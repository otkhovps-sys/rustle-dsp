//! Voice-processing chain around RNNoise (all at 48 kHz mono, per 480-sample frame):
//!
//! ```text
//!  mic ─► high-pass 80 Hz ─► RNNoise ─► strength mix ─► voice gate ─► AGC ─► limiter ─► out
//!                              │  dry (delay-aligned) ──┘      ▲
//!                              └─────────── VAD probability ───┘ (also drives AGC)
//! ```
//!
//! * **High-pass** removes rumble (fans, AC, desk thumps) below the voice band
//!   before the network sees it.
//! * **Strength** mixes the delay-aligned dry signal back in: 100 % = pure
//!   RNNoise, lower values trade residual noise for a more natural voice.
//! * **Voice gate** uses RNNoise's own voice-activity probability to pull the
//!   level down between phrases (residual keyboard clicks, breathing). It has
//!   hysteresis, a hold time and smooth gain ramps so words are not chopped.
//! * **AGC** brings speech to a consistent level, adapting only while voice is
//!   detected (noise is never pumped up during pauses).
//! * **Limiter** guarantees no clipping (instant attack, smooth release).
//!
//! Parameters live in [`VoiceControls`] (atomics), so the UI can change them
//! while audio runs without locks. Everything here is plain arithmetic on
//! preallocated state: no allocation, no locks, RT-friendly.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};

use crate::fx::{FxChain, FxSettings};

pub const SAMPLE_RATE: f32 = 48_000.0;
const FRAME: usize = crate::denoiser::FRAME_SIZE;

// ---------------------------------------------------------------------------
// Settings / presets
// ---------------------------------------------------------------------------

/// How hard the voice gate attenuates between phrases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "gui", derive(serde::Serialize, serde::Deserialize))]
pub enum GateMode {
    Off,
    /// −15 dB between phrases: noise is lower but the line never sounds "dead".
    Light,
    /// −45 dB between phrases: near silence when you are not talking.
    Strong,
}

impl GateMode {
    fn to_u8(self) -> u8 {
        match self {
            GateMode::Off => 0,
            GateMode::Light => 1,
            GateMode::Strong => 2,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            1 => GateMode::Light,
            2 => GateMode::Strong,
            _ => GateMode::Off,
        }
    }
    /// Gain applied while the gate is closed.
    fn floor_gain(self) -> f32 {
        match self {
            GateMode::Off => 1.0,
            GateMode::Light => db_to_gain(-15.0),
            GateMode::Strong => db_to_gain(-45.0),
        }
    }
}

/// Voice relief of the Light / Balanced presets, chosen from measurements on
/// real speech (LibriSpeech + fan/hum/keyboard noise): at 10 dB SNR the
/// overall SI-SDR peaks around 0.6 and clean-speech distortion keeps falling
/// with more relief, but at 0 dB SNR VAD false positives let relief raise
/// the noise in pauses — so the Strong (noisy-room) preset uses none.
pub const RELIEF_LIGHT: f32 = 0.7;
pub const RELIEF_BALANCED: f32 = 0.5;
/// Upper bound of `voice_relief`.
pub const RELIEF_MAX: f32 = 0.8;

/// Ready-made combinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "gui", derive(serde::Serialize, serde::Deserialize))]
pub enum Preset {
    /// Most natural voice (70 % voice relief) + light gate (quiet rooms).
    Light,
    /// Natural voice (50 % voice relief) + light gate (recommended default).
    Balanced,
    /// Maximum cleaning, no relief, strong gate (noisy rooms, cafés).
    Strong,
}

/// A complete processing configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "gui", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiceSettings {
    pub highpass: bool,
    /// 0.0 = dry only, 1.0 = pure RNNoise.
    pub strength: f32,
    /// "Keep the voice natural": while voice is detected, the strength is
    /// reduced by this fraction (0.0..=RELIEF_MAX). Speech masks the residual noise,
    /// and RNNoise's own artefacts ("watery" voice) are what you would
    /// otherwise hear; pauses stay fully cleaned.
    pub voice_relief: f32,
    pub gate: GateMode,
    pub agc: bool,
    /// Pass the (delay-aligned) raw microphone through untouched, for A/B.
    pub bypass: bool,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self::preset(Preset::Balanced)
    }
}

impl VoiceSettings {
    pub fn preset(p: Preset) -> Self {
        match p {
            // AGC is off in every preset: Discord / Zoom / Meet / Teams all
            // run their own, and two AGCs in series pump. Enable it only for
            // apps that have none (OBS, recorders).
            Preset::Light => Self {
                highpass: true,
                strength: 1.0,
                voice_relief: RELIEF_LIGHT,
                gate: GateMode::Light,
                agc: false,
                bypass: false,
            },
            Preset::Balanced => Self {
                highpass: true,
                strength: 1.0,
                voice_relief: RELIEF_BALANCED,
                gate: GateMode::Light,
                agc: false,
                bypass: false,
            },
            Preset::Strong => Self {
                highpass: true,
                strength: 1.0,
                voice_relief: 0.0,
                gate: GateMode::Strong,
                agc: false,
                bypass: false,
            },
        }
    }

    /// Which preset these settings correspond to, if any (bypass ignored).
    pub fn matching_preset(&self) -> Option<Preset> {
        [Preset::Light, Preset::Balanced, Preset::Strong]
            .into_iter()
            .find(|p| {
                let s = Self::preset(*p);
                s.highpass == self.highpass
                    && (s.strength - self.strength).abs() < 0.005
                    && (s.voice_relief - self.voice_relief).abs() < 0.005
                    && s.gate == self.gate
                    && s.agc == self.agc
            })
    }
}

/// Live, lock-free parameter block shared between the UI and the DSP thread.
#[derive(Debug)]
pub struct VoiceControls {
    highpass: AtomicBool,
    strength_bits: AtomicU32,
    relief_bits: AtomicU32,
    gate: AtomicU8,
    agc: AtomicBool,
    bypass: AtomicBool,
}

impl VoiceControls {
    pub fn new(s: VoiceSettings) -> Self {
        let c = Self {
            highpass: AtomicBool::new(false),
            strength_bits: AtomicU32::new(0),
            relief_bits: AtomicU32::new(0),
            gate: AtomicU8::new(0),
            agc: AtomicBool::new(false),
            bypass: AtomicBool::new(false),
        };
        c.set(s);
        c
    }

    pub fn set(&self, s: VoiceSettings) {
        self.highpass.store(s.highpass, Ordering::Relaxed);
        self.strength_bits
            .store(s.strength.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.relief_bits
            .store(s.voice_relief.clamp(0.0, RELIEF_MAX).to_bits(), Ordering::Relaxed);
        self.gate.store(s.gate.to_u8(), Ordering::Relaxed);
        self.agc.store(s.agc, Ordering::Relaxed);
        self.bypass.store(s.bypass, Ordering::Relaxed);
    }

    pub fn get(&self) -> VoiceSettings {
        VoiceSettings {
            highpass: self.highpass.load(Ordering::Relaxed),
            strength: f32::from_bits(self.strength_bits.load(Ordering::Relaxed)),
            voice_relief: f32::from_bits(self.relief_bits.load(Ordering::Relaxed)),
            gate: GateMode::from_u8(self.gate.load(Ordering::Relaxed)),
            agc: self.agc.load(Ordering::Relaxed),
            bypass: self.bypass.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

pub fn db_to_gain(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

pub fn gain_to_db(g: f32) -> f32 {
    20.0 * g.max(1e-9).log10()
}

/// One-pole smoothing coefficient for a time constant `ms` at 48 kHz.
fn coeff(ms: f32) -> f32 {
    (-1.0 / (ms * 0.001 * SAMPLE_RATE)).exp()
}

/// 2nd-order Butterworth biquad (RBJ cookbook), transposed direct form II.
#[derive(Debug, Clone)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

/// The 80 Hz rumble filter uses the high-pass form.
pub type HighPass = Biquad;

impl Biquad {
    /// High-pass (kept as `new` for the rumble filter).
    pub fn new(cutoff_hz: f32) -> Self {
        Self::highpass(cutoff_hz)
    }

    pub fn highpass(cutoff_hz: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz / SAMPLE_RATE;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 + cos) / 2.0 / a0,
            b1: -(1.0 + cos) / a0,
            b2: (1.0 + cos) / 2.0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    pub fn lowpass(cutoff_hz: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * cutoff_hz / SAMPLE_RATE;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 - cos) / 2.0 / a0,
            b1: (1.0 - cos) / a0,
            b2: (1.0 - cos) / 2.0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            let y = self.b0 * *x + self.z1;
            self.z1 = self.b1 * *x - self.a1 * y + self.z2;
            self.z2 = self.b2 * *x - self.a2 * y;
            *x = y;
        }
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Voice gate driven by RNNoise's VAD probability.
#[derive(Debug, Clone)]
struct VoiceGate {
    open: bool,
    hold_left: u32,
    gain: f32,
    attack: f32,
    release: f32,
}

impl VoiceGate {
    /// Open as soon as VAD ≥ 0.6; start closing only after VAD has stayed
    /// < 0.3 for 200 ms (hold). Hysteresis + hold keep word endings and the
    /// short gaps between words intact; the gate acts on real pauses.
    const OPEN_AT: f32 = 0.6;
    const CLOSE_BELOW: f32 = 0.3;
    const HOLD_FRAMES: u32 = 20; // 20 × 10 ms

    fn new() -> Self {
        Self {
            open: false,
            hold_left: 0,
            gain: 1.0,
            // Opening: fast, so onsets are not clipped.
            attack: coeff(3.0),
            // Closing: 40 ms time constant = 8.7 dB per 40 ms, so −45 dB is
            // reached ~0.2 s after the hold (measured: a 120 ms constant took
            // ~0.6 s and barely touched typical conversational pauses). The
            // denoised signal is already quiet here, so this is not audible
            // as a "chop".
            release: coeff(40.0),
        }
    }

    fn update(&mut self, vad: f32) {
        if vad >= Self::OPEN_AT {
            self.open = true;
            self.hold_left = Self::HOLD_FRAMES;
        } else if vad < Self::CLOSE_BELOW {
            if self.hold_left > 0 {
                self.hold_left -= 1;
            } else {
                self.open = false;
            }
        }
        // Between the thresholds: keep the current state.
    }

    fn process(&mut self, frame: &mut [f32], floor: f32) {
        let target = if self.open { 1.0 } else { floor };
        let c = if target > self.gain { self.attack } else { self.release };
        for x in frame.iter_mut() {
            self.gain = target + (self.gain - target) * c;
            *x *= self.gain;
        }
    }
}

/// Speech-level automatic gain control.
///
/// Deliberately slow: it tracks the *average* speech level over about a
/// second of voiced frames and moves the gain by at most 3 dB/s, so it
/// corrects "too quiet / too loud" without touching the dynamics of words.
/// (v0.3.0 reacted per 10 ms frame and audibly squashed every loud
/// syllable — measured as a 7 dB SI-SDR loss on clean speech.)
#[derive(Debug, Clone)]
struct Agc {
    level_db: f32,
    gain_db: f32,
    applied: f32,
}

impl Agc {
    const TARGET_DB: f32 = -23.0; // long-term speech RMS target (dBFS)
    const MAX_DB: f32 = 12.0;
    const MIN_DB: f32 = -6.0;
    const MIN_SPEECH_DB: f32 = -50.0; // ignore near-silent frames
    const LEVEL_ALPHA: f32 = 0.01; // per voiced frame: ~1 s of speech
    const MAX_STEP_DB: f32 = 0.03; // per frame = 3 dB/s

    fn new() -> Self {
        Self {
            level_db: Self::TARGET_DB,
            gain_db: 0.0,
            applied: 1.0,
        }
    }

    /// `frame` is measured *before* this stage's gain (input level).
    fn process(&mut self, frame: &mut [f32], vad: f32, enabled: bool) {
        if enabled && vad > 0.8 {
            let rms = (frame.iter().map(|v| v * v).sum::<f32>() / frame.len() as f32).sqrt();
            let rms_db = gain_to_db(rms);
            if rms_db > Self::MIN_SPEECH_DB {
                self.level_db += (rms_db - self.level_db) * Self::LEVEL_ALPHA;
            }
            let want = (Self::TARGET_DB - self.level_db).clamp(Self::MIN_DB, Self::MAX_DB);
            self.gain_db += (want - self.gain_db).clamp(-Self::MAX_STEP_DB, Self::MAX_STEP_DB);
        }
        let target = if enabled { db_to_gain(self.gain_db) } else { 1.0 };
        // Ramp linearly across the frame: no zipper noise.
        let step = (target - self.applied) / frame.len() as f32;
        for x in frame.iter_mut() {
            self.applied += step;
            *x *= self.applied;
        }
        self.applied = target;
    }
}

/// Peak limiter: instant attack (no overshoot possible), 60 ms release.
#[derive(Debug, Clone)]
pub struct Limiter {
    env: f32,
    release: f32,
    ceiling: f32,
}

impl Default for Limiter {
    fn default() -> Self {
        Self::new()
    }
}

impl Limiter {
    pub fn new() -> Self {
        Self {
            env: 0.0,
            release: coeff(60.0),
            ceiling: db_to_gain(-1.0),
        }
    }

    pub fn process(&mut self, frame: &mut [f32]) {
        for x in frame.iter_mut() {
            self.env = x.abs().max(self.env * self.release);
            if self.env > self.ceiling {
                *x *= self.ceiling / self.env;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The chain
// ---------------------------------------------------------------------------

/// Post-RNNoise chain state (one per stream).
#[derive(Debug, Clone)]
pub struct VoiceChain {
    gate: VoiceGate,
    agc: Agc,
    limiter: Limiter,
    /// Smoothed VAD for the speech-aware strength (no fast modulation).
    vad_smooth: f32,
    /// Mix factor applied at the end of the previous frame (ramped).
    k_prev: f32,
}

impl Default for VoiceChain {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceChain {
    pub fn new() -> Self {
        Self {
            gate: VoiceGate::new(),
            agc: Agc::new(),
            limiter: Limiter::new(),
            vad_smooth: 0.0,
            k_prev: 1.0,
        }
    }

    /// Process one 480-sample frame. `denoised` and `dry` must be
    /// time-aligned (see `Denoiser`), `out` receives the result.
    pub fn process(
        &mut self,
        denoised: &[f32],
        dry: &[f32],
        vad: f32,
        s: &VoiceSettings,
        out: &mut [f32; FRAME],
    ) {
        if s.bypass {
            out.copy_from_slice(dry);
            return;
        }
        // Speech-aware strength: full cleaning in pauses, `voice_relief`
        // less while someone talks. VAD is smoothed (~30 ms) and the mix
        // factor ramps across the frame, so there is no audible modulation.
        self.vad_smooth += (vad - self.vad_smooth) * 0.3;
        let relief = s.voice_relief.clamp(0.0, RELIEF_MAX) * self.vad_smooth;
        let k = (s.strength.clamp(0.0, 1.0) * (1.0 - relief)).clamp(0.0, 1.0);
        let step = (k - self.k_prev) / FRAME as f32;
        let mut kk = self.k_prev;
        for ((o, d), r) in out.iter_mut().zip(denoised).zip(dry) {
            kk += step;
            *o = kk * d + (1.0 - kk) * r;
        }
        self.k_prev = k;
        self.gate.update(vad);
        if s.gate != GateMode::Off {
            self.gate.process(out, s.gate.floor_gain());
        } else {
            self.gate.gain = 1.0;
        }
        self.agc.process(out, vad, s.agc);
        self.limiter.process(out);
    }

    /// Current AGC gain (dB) for meters.
    pub fn agc_gain_db(&self) -> f32 {
        gain_to_db(self.agc.applied)
    }

    /// Current gate gain (0..1) for meters.
    pub fn gate_gain(&self) -> f32 {
        self.gate.gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn highpass_cuts_rumble_keeps_voice() {
        for (f, lo, hi) in [(30.0, -40.0, -14.0), (1000.0, -0.3, 0.3)] {
            let mut hp = HighPass::new(80.0);
            let mut x = sine(f, 48_000, 0.5);
            hp.process(&mut x);
            let db = gain_to_db(rms(&x[24_000..]) / (0.5 / 2f32.sqrt()));
            assert!(db > lo && db < hi, "{f} Hz -> {db:.1} dB");
        }
    }

    #[test]
    fn strength_zero_is_dry_and_one_is_denoised() {
        let mut c = VoiceChain::new();
        let den = [0.1f32; FRAME];
        let dry = [0.3f32; FRAME];
        let mut out = [0.0f32; FRAME];
        let mut s = VoiceSettings { highpass: false, strength: 0.0, voice_relief: 0.0, gate: GateMode::Off, agc: false, bypass: false };
        // The mix factor ramps over one frame; the second frame is steady.
        c.process(&den, &dry, 1.0, &s, &mut out);
        c.process(&den, &dry, 1.0, &s, &mut out);
        assert!(out.iter().all(|v| (v - 0.3).abs() < 1e-6));
        s.strength = 1.0;
        c.process(&den, &dry, 1.0, &s, &mut out);
        c.process(&den, &dry, 1.0, &s, &mut out);
        assert!(out.iter().all(|v| (v - 0.1).abs() < 1e-6));
    }

    #[test]
    fn gate_holds_then_closes_to_floor_and_reopens_fast() {
        let mut c = VoiceChain::new();
        let s = VoiceSettings { highpass: false, strength: 1.0, voice_relief: 0.0, gate: GateMode::Strong, agc: false, bypass: false };
        let x = [0.2f32; FRAME];
        let mut out = [0.0f32; FRAME];
        // Speech, then silence-probability frames.
        for _ in 0..10 {
            c.process(&x, &x, 0.95, &s, &mut out);
        }
        for _ in 0..19 {
            c.process(&x, &x, 0.05, &s, &mut out);
        }
        // Still inside the 200 ms hold: fully open.
        assert!((out[FRAME - 1] - 0.2).abs() < 1e-3, "hold: {}", out[FRAME - 1]);
        // Hold ends, then ~0.25 s of release reaches the floor.
        for _ in 0..30 {
            c.process(&x, &x, 0.05, &s, &mut out);
        }
        let floor = 0.2 * db_to_gain(-45.0);
        assert!(out[FRAME - 1] < floor * 1.5, "closed: {}", out[FRAME - 1]);
        // Voice returns: within one frame (3 ms attack) the gain is back up.
        c.process(&x, &x, 0.95, &s, &mut out);
        assert!(out[FRAME - 1] > 0.19, "reopen: {}", out[FRAME - 1]);
    }

    #[test]
    fn agc_raises_quiet_speech_but_not_noise() {
        let mut c = VoiceChain::new();
        let s = VoiceSettings { highpass: false, strength: 1.0, voice_relief: 0.0, gate: GateMode::Off, agc: true, bypass: false };
        let quiet = sine(200.0, FRAME, 0.02); // ≈ -37 dBFS RMS
        let mut out = [0.0f32; FRAME];
        // Noise-only frames (VAD low) must not move the gain.
        for _ in 0..200 {
            c.process(&quiet, &quiet, 0.1, &s, &mut out);
        }
        assert!(c.agc_gain_db().abs() < 0.01);
        // Slow by design (≤ 3 dB/s, ~1 s level estimate): after 10 s of
        // speech the +12 dB cap is reached.
        for _ in 0..1000 {
            c.process(&quiet, &quiet, 0.95, &s, &mut out);
        }
        assert!((c.agc_gain_db() - 12.0).abs() < 0.5, "gain {} dB", c.agc_gain_db());
    }

    #[test]
    fn agc_does_not_pump_on_syllables() {
        // Speech-like dynamics: 120 ms loud (-14 dBFS) / 80 ms soft (-34 dBFS).
        let mut c = VoiceChain::new();
        let s = VoiceSettings { highpass: false, strength: 1.0, voice_relief: 0.0, gate: GateMode::Off, agc: true, bypass: false };
        let loud = sine(220.0, FRAME, 0.28);
        let soft = sine(220.0, FRAME, 0.028);
        let mut out = [0.0f32; FRAME];
        let mut gains = Vec::new();
        for i in 0..600 {
            let x = if i % 20 < 12 { &loud } else { &soft };
            c.process(x, x, 0.95, &s, &mut out);
            if i >= 500 {
                gains.push(c.agc_gain_db());
            }
        }
        let (lo, hi) = gains.iter().fold((f32::MAX, f32::MIN), |(a, b), g| (a.min(*g), b.max(*g)));
        // v0.3.0 swung by several dB within each syllable cycle.
        assert!(hi - lo < 0.7, "gain swing {:.2} dB within syllables", hi - lo);
    }

    #[test]
    fn voice_relief_mixes_dry_only_while_talking() {
        let mut c = VoiceChain::new();
        let s = VoiceSettings { highpass: false, strength: 1.0, voice_relief: 0.25, gate: GateMode::Off, agc: false, bypass: false };
        let den = [0.0f32; FRAME];
        let dry = [1.0f32; FRAME];
        let mut out = [0.0f32; FRAME];
        for _ in 0..50 {
            c.process(&den, &dry, 0.0, &s, &mut out);
        }
        assert!(out[FRAME - 1].abs() < 1e-3, "pause must be fully cleaned: {}", out[FRAME - 1]);
        for _ in 0..50 {
            c.process(&den, &dry, 1.0, &s, &mut out);
        }
        assert!((out[FRAME - 1] - 0.25).abs() < 0.01, "speech keeps 25 % dry: {}", out[FRAME - 1]);
    }

    #[test]
    fn limiter_never_exceeds_ceiling() {
        let mut c = VoiceChain::new();
        let s = VoiceSettings { highpass: false, strength: 1.0, voice_relief: 0.0, gate: GateMode::Off, agc: true, bypass: false };
        let loud = sine(300.0, FRAME, 0.99);
        let mut out = [0.0f32; FRAME];
        for _ in 0..300 {
            c.process(&loud, &loud, 0.99, &s, &mut out);
            assert!(out.iter().all(|v| v.abs() <= db_to_gain(-1.0) + 1e-6));
        }
    }

    #[test]
    fn bypass_is_exact_dry() {
        let mut c = VoiceChain::new();
        let s = VoiceSettings { bypass: true, ..VoiceSettings::default() };
        let den = [0.0f32; FRAME];
        let dry: Vec<f32> = (0..FRAME).map(|i| i as f32 / FRAME as f32 - 0.5).collect();
        let mut out = [0.0f32; FRAME];
        c.process(&den, &dry, 0.0, &s, &mut out);
        assert_eq!(&out[..], &dry[..]);
    }

    #[test]
    fn presets_round_trip_and_controls_store_values() {
        for p in [Preset::Light, Preset::Balanced, Preset::Strong] {
            let s = VoiceSettings::preset(p);
            assert_eq!(s.matching_preset(), Some(p));
            let c = VoiceControls::new(s);
            assert_eq!(c.get(), s);
        }
        let custom = VoiceSettings { strength: 0.55, ..VoiceSettings::default() };
        assert_eq!(custom.matching_preset(), None);
    }
}

// ---------------------------------------------------------------------------
// Complete 48 kHz voice processor (used by the real-time worker and offline)
// ---------------------------------------------------------------------------

/// High-pass → RNNoise → [`VoiceChain`] → voice effects ([`FxChain`]), fed
/// with arbitrary-length 48 kHz mono slices. Output lags input by
/// [`crate::denoiser::MODEL_DELAY`] plus [`FxChain::latency`] (0 unless a
/// pitch effect is enabled).
pub struct VoiceProcessor {
    highpass: HighPass,
    denoiser: crate::denoiser::Denoiser,
    chain: VoiceChain,
    fx: FxChain,
    /// Scratch for the high-passed input (sized for the largest push).
    pre: Vec<f32>,
    frame: [f32; FRAME],
}

impl VoiceProcessor {
    /// `max_push`: largest slice ever passed to `process` (scratch capacity).
    pub fn new(max_push: usize) -> Self {
        Self {
            highpass: HighPass::new(80.0),
            denoiser: crate::denoiser::Denoiser::new(),
            chain: VoiceChain::new(),
            fx: FxChain::new(),
            pre: Vec::with_capacity(max_push),
            frame: [0.0; FRAME],
        }
    }

    pub fn last_vad(&self) -> f32 {
        self.denoiser.last_vad()
    }

    pub fn frames_processed(&self) -> u64 {
        self.denoiser.frames_processed()
    }

    pub fn pending(&self) -> usize {
        self.denoiser.pending()
    }

    pub fn agc_gain_db(&self) -> f32 {
        self.chain.agc_gain_db()
    }

    pub fn gate_gain(&self) -> f32 {
        self.chain.gate_gain()
    }

    /// Latency currently added by the effects stage (samples).
    pub fn fx_latency(&self) -> usize {
        self.fx.latency()
    }

    /// Process `input` with noise-reduction settings `s` and effects `fx`;
    /// every finished 480-sample frame is passed to `sink`.
    pub fn process(
        &mut self,
        input: &[f32],
        s: &VoiceSettings,
        fx: &FxSettings,
        sink: &mut dyn FnMut(&[f32]),
    ) {
        self.pre.clear();
        self.pre.extend_from_slice(input);
        if s.highpass && !s.bypass {
            self.highpass.process(&mut self.pre);
        }
        let Self {
            denoiser,
            chain,
            fx: fx_chain,
            pre,
            frame,
            ..
        } = self;
        denoiser.push(pre, &mut |den, dry, vad| {
            chain.process(den, dry, vad, s, frame);
            fx_chain.process(frame, fx, s.bypass);
            sink(&frame[..]);
        });
    }

    /// Emit everything still held (partial frame, model delay, effects
    /// look-ahead).
    pub fn flush(&mut self, s: &VoiceSettings, fx: &FxSettings, sink: &mut dyn FnMut(&[f32])) {
        let Self {
            denoiser,
            chain,
            fx: fx_chain,
            frame,
            ..
        } = self;
        let mut den_pad = [0.0f32; FRAME];
        let mut dry_pad = [0.0f32; FRAME];
        denoiser.flush(&mut |den, dry, vad| {
            let n = den.len();
            den_pad.fill(0.0);
            dry_pad.fill(0.0);
            den_pad[..n].copy_from_slice(den);
            dry_pad[..n].copy_from_slice(dry);
            chain.process(&den_pad, &dry_pad, vad, s, frame);
            fx_chain.process(&mut frame[..n], fx, s.bypass);
            sink(&frame[..n]);
        });
        // Drain the effects look-ahead with silence.
        let mut left = fx_chain.latency();
        while left > 0 {
            let n = left.min(FRAME);
            frame[..n].fill(0.0);
            fx_chain.process(&mut frame[..n], fx, s.bypass);
            sink(&frame[..n]);
            left -= n;
        }
    }
}

/// Offline helper: process a whole 48 kHz mono buffer (noise reduction only)
/// and return a buffer of the same length, compensated for the model delay.
pub fn process_buffer(input: &[f32], s: &VoiceSettings) -> Vec<f32> {
    process_buffer_fx(input, s, &FxSettings::default())
}

/// Like [`process_buffer`] with voice effects; compensates their latency too.
pub fn process_buffer_fx(input: &[f32], s: &VoiceSettings, fx: &FxSettings) -> Vec<f32> {
    let mut p = VoiceProcessor::new(4096);
    let mut out = Vec::with_capacity(input.len() + FRAME + crate::fx::FX_LATENCY);
    for chunk in input.chunks(4096) {
        p.process(chunk, s, fx, &mut |f| out.extend_from_slice(f));
    }
    let delay = crate::denoiser::MODEL_DELAY + p.fx_latency();
    p.flush(s, fx, &mut |f| out.extend_from_slice(f));
    out.drain(..delay.min(out.len()));
    out.truncate(input.len());
    out
}

#[cfg(test)]
mod processor_tests {
    use super::*;

    #[test]
    fn effects_keep_length_and_alignment_in_bypass() {
        // A/B with a pitch effect selected: same latency path, exact input.
        let input: Vec<f32> = (0..48_000 + 77).map(|i| ((i * 13) % 89) as f32 / 300.0 - 0.15).collect();
        let s = VoiceSettings { bypass: true, ..VoiceSettings::default() };
        let fx = crate::fx::FxSettings::preset(crate::fx::FxPreset::Deep);
        let out = process_buffer_fx(&input, &s, &fx);
        assert_eq!(out.len(), input.len());
        let err = out.iter().zip(&input).skip(4_000).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-4, "max error {err}");
    }

    #[test]
    fn process_buffer_preserves_length_and_alignment_in_bypass() {
        let input: Vec<f32> = (0..48_000 + 123).map(|i| ((i * 31) % 97) as f32 / 200.0 - 0.2).collect();
        let s = VoiceSettings { bypass: true, ..VoiceSettings::default() };
        let out = process_buffer(&input, &s);
        assert_eq!(out.len(), input.len());
        assert_eq!(out, input, "bypass must be the exact input once delay is compensated");
    }
}
