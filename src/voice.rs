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

/// Ready-made combinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "gui", derive(serde::Serialize, serde::Deserialize))]
pub enum Preset {
    /// Natural voice: 80 % strength + light gate (quiet rooms).
    Light,
    /// Full RNNoise + light gate (recommended default).
    Balanced,
    /// Full RNNoise + strong gate (noisy rooms, open offices, cafés).
    Strong,
}

/// A complete processing configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "gui", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiceSettings {
    pub highpass: bool,
    /// 0.0 = dry only, 1.0 = pure RNNoise.
    pub strength: f32,
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
            Preset::Light => Self {
                highpass: true,
                strength: 0.8,
                gate: GateMode::Light,
                agc: true,
                bypass: false,
            },
            Preset::Balanced => Self {
                highpass: true,
                strength: 1.0,
                gate: GateMode::Light,
                agc: true,
                bypass: false,
            },
            Preset::Strong => Self {
                highpass: true,
                strength: 1.0,
                gate: GateMode::Strong,
                agc: true,
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
    gate: AtomicU8,
    agc: AtomicBool,
    bypass: AtomicBool,
}

impl VoiceControls {
    pub fn new(s: VoiceSettings) -> Self {
        let c = Self {
            highpass: AtomicBool::new(false),
            strength_bits: AtomicU32::new(0),
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
        self.gate.store(s.gate.to_u8(), Ordering::Relaxed);
        self.agc.store(s.agc, Ordering::Relaxed);
        self.bypass.store(s.bypass, Ordering::Relaxed);
    }

    pub fn get(&self) -> VoiceSettings {
        VoiceSettings {
            highpass: self.highpass.load(Ordering::Relaxed),
            strength: f32::from_bits(self.strength_bits.load(Ordering::Relaxed)),
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

/// 2nd-order Butterworth high-pass (RBJ cookbook), transposed direct form II.
#[derive(Debug, Clone)]
pub struct HighPass {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl HighPass {
    pub fn new(cutoff_hz: f32) -> Self {
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
#[derive(Debug, Clone)]
struct Agc {
    gain_db: f32,
    applied: f32,
}

impl Agc {
    const TARGET_DB: f32 = -20.0; // speech RMS target (dBFS)
    const MAX_DB: f32 = 18.0;
    const MIN_DB: f32 = -10.0;
    const MIN_SPEECH_DB: f32 = -55.0; // ignore near-silent frames
    const UP_RATE: f32 = 0.02; // per speech frame (slow: ~0.5 s)
    const DOWN_RATE: f32 = 0.15; // per speech frame (fast: loud speech)

    fn new() -> Self {
        Self {
            gain_db: 0.0,
            applied: 1.0,
        }
    }

    fn process(&mut self, frame: &mut [f32], vad: f32, enabled: bool) {
        if enabled && vad > 0.7 {
            let rms = (frame.iter().map(|v| v * v).sum::<f32>() / frame.len() as f32).sqrt();
            let rms_db = gain_to_db(rms);
            if rms_db > Self::MIN_SPEECH_DB {
                let want = (Self::TARGET_DB - rms_db).clamp(Self::MIN_DB, Self::MAX_DB);
                let rate = if want < self.gain_db { Self::DOWN_RATE } else { Self::UP_RATE };
                self.gain_db += (want - self.gain_db) * rate;
            }
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
struct Limiter {
    env: f32,
    release: f32,
    ceiling: f32,
}

impl Limiter {
    fn new() -> Self {
        Self {
            env: 0.0,
            release: coeff(60.0),
            ceiling: db_to_gain(-1.0),
        }
    }

    fn process(&mut self, frame: &mut [f32]) {
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
        let k = s.strength.clamp(0.0, 1.0);
        for ((o, d), r) in out.iter_mut().zip(denoised).zip(dry) {
            *o = k * d + (1.0 - k) * r;
        }
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
        let mut s = VoiceSettings { highpass: false, strength: 0.0, gate: GateMode::Off, agc: false, bypass: false };
        c.process(&den, &dry, 1.0, &s, &mut out);
        assert!(out.iter().all(|v| (v - 0.3).abs() < 1e-6));
        s.strength = 1.0;
        c.process(&den, &dry, 1.0, &s, &mut out);
        assert!(out.iter().all(|v| (v - 0.1).abs() < 1e-6));
    }

    #[test]
    fn gate_holds_then_closes_to_floor_and_reopens_fast() {
        let mut c = VoiceChain::new();
        let s = VoiceSettings { highpass: false, strength: 1.0, gate: GateMode::Strong, agc: false, bypass: false };
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
        let s = VoiceSettings { highpass: false, strength: 1.0, gate: GateMode::Off, agc: true, bypass: false };
        let quiet = sine(200.0, FRAME, 0.02); // ≈ -37 dBFS RMS
        let mut out = [0.0f32; FRAME];
        // Noise-only frames (VAD low) must not move the gain.
        for _ in 0..200 {
            c.process(&quiet, &quiet, 0.1, &s, &mut out);
        }
        assert!(c.agc_gain_db().abs() < 0.01);
        for _ in 0..400 {
            c.process(&quiet, &quiet, 0.95, &s, &mut out);
        }
        assert!((c.agc_gain_db() - 17.0).abs() < 1.5, "gain {} dB", c.agc_gain_db());
    }

    #[test]
    fn limiter_never_exceeds_ceiling() {
        let mut c = VoiceChain::new();
        let s = VoiceSettings { highpass: false, strength: 1.0, gate: GateMode::Off, agc: true, bypass: false };
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

/// High-pass → RNNoise → [`VoiceChain`], fed with arbitrary-length 48 kHz
/// mono slices. Output lags input by [`crate::denoiser::MODEL_DELAY`].
pub struct VoiceProcessor {
    highpass: HighPass,
    denoiser: crate::denoiser::Denoiser,
    chain: VoiceChain,
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

    /// Process `input` with settings `s`; every finished 480-sample frame is
    /// passed to `sink`.
    pub fn process(&mut self, input: &[f32], s: &VoiceSettings, sink: &mut dyn FnMut(&[f32])) {
        self.pre.clear();
        self.pre.extend_from_slice(input);
        if s.highpass && !s.bypass {
            self.highpass.process(&mut self.pre);
        }
        let Self {
            denoiser,
            chain,
            pre,
            frame,
            ..
        } = self;
        denoiser.push(pre, &mut |den, dry, vad| {
            chain.process(den, dry, vad, s, frame);
            sink(&frame[..]);
        });
    }

    /// Emit everything still held (partial frame + model delay).
    pub fn flush(&mut self, s: &VoiceSettings, sink: &mut dyn FnMut(&[f32])) {
        let Self {
            denoiser,
            chain,
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
            sink(&frame[..n]);
        });
    }
}

/// Offline helper: process a whole 48 kHz mono buffer and return a buffer of
/// the same length, already compensated for the model delay.
pub fn process_buffer(input: &[f32], s: &VoiceSettings) -> Vec<f32> {
    let mut p = VoiceProcessor::new(4096);
    let mut out = Vec::with_capacity(input.len() + FRAME);
    for chunk in input.chunks(4096) {
        p.process(chunk, s, &mut |f| out.extend_from_slice(f));
    }
    p.flush(s, &mut |f| out.extend_from_slice(f));
    out.drain(..crate::denoiser::MODEL_DELAY.min(out.len()));
    out.truncate(input.len());
    out
}

#[cfg(test)]
mod processor_tests {
    use super::*;

    #[test]
    fn process_buffer_preserves_length_and_alignment_in_bypass() {
        let input: Vec<f32> = (0..48_000 + 123).map(|i| ((i * 31) % 97) as f32 / 200.0 - 0.2).collect();
        let s = VoiceSettings { bypass: true, ..VoiceSettings::default() };
        let out = process_buffer(&input, &s);
        assert_eq!(out.len(), input.len());
        assert_eq!(out, input, "bypass must be the exact input once delay is compensated");
    }
}
