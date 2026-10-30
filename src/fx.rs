//! Voice effects (48 kHz mono, after noise reduction):
//!
//! * **Pitch shift** and **formant shift** with TD-PSOLA: the voice is cut
//!   into pitch-synchronous grains (two periods, Hann window) that are laid
//!   down again at a new spacing. Changing the spacing changes the pitch
//!   while each grain keeps its spectral envelope (formants), so +5 semitones
//!   does not sound like a chipmunk. Resampling each grain moves the
//!   formants independently (male ↔ female timbre).
//! * **Auto-tune**: the detected pitch is snapped to the nearest note of a
//!   key/scale, with an adjustable retune speed (0 = hard "T-Pain" effect).
//! * **Robot**: the pitch is flattened to a single note.
//! * **Radio**, **echo**, **reverb**: simple time-domain effects.
//!
//! Pitch detection: YIN on a 4× decimated (12 kHz) copy of the signal, every
//! 5 ms, 75–800 Hz, median of the last three estimates against octave jumps.
//!
//! Latency: PSOLA needs to see ahead by up to one grain plus the pitch
//! analysis window, so while pitch effects are enabled the output is delayed
//! by [`FX_LATENCY`] samples (38 ms). With effects disabled the stage is a
//! no-op and adds nothing. Unvoiced sounds (consonants, breath, silence) are
//! reconstructed exactly (grains at ratio 1 sum back to the input), so only
//! voiced speech is altered.
//!
//! All buffers are allocated in `new()`; processing is allocation- and
//! lock-free. Parameters are read from [`FxControls`] (atomics) every frame.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};

use crate::voice::{Biquad, Limiter, SAMPLE_RATE};

/// Look-ahead of the pitch engine (samples at 48 kHz) = 38 ms.
///
/// Budget: a grain centred up to `H_MAX` past the emission point, taken from
/// an analysis mark up to half a period (≤ 320) further, reads up to
/// `H_MAX × 2^(5/12)` (max formant factor) around it: 640 + 320 + 855 ≤ 1824.
pub const FX_LATENCY: usize = 1824;

// Ring sizes (powers of two).
const HIST: usize = 8192;
const DEC_HIST: usize = 2048;
const MARKS: usize = 64;
const PITCH_HIST: usize = 64;

// Pitch analysis (12 kHz decimated domain).
const DECIM: usize = 4;
const DEC_RATE: f32 = SAMPLE_RATE / DECIM as f32;
const YIN_W: usize = 256;
const F0_MIN: f32 = 75.0;
const F0_MAX: f32 = 800.0;
const TAU_MAX: usize = (DEC_RATE / F0_MIN) as usize + 1; // 134
const TAU_MIN: usize = (DEC_RATE / F0_MAX) as usize; // 15
const YIN_HOP: usize = 60; // 5 ms at 12 kHz
/// CMNDF threshold, chosen on real speech (LibriSpeech male ~100 Hz and
/// female ~220 Hz vs. an independent autocorrelation reference): 0.25
/// detects 83 % / 92 % of voiced frames with 93 % / 91 % within 5 %; looser
/// settings or a global-minimum fallback mostly add octave errors and
/// breath detected as voice.
const YIN_THRESHOLD: f32 = 0.25;
const YIN_FALLBACK: f32 = 0.0;

/// Grain half-length cap (samples) = one period at F0_MIN; unvoiced spacing.
const H_MAX: usize = 640;
const UNVOICED_PERIOD: f32 = 240.0;
/// Formant shift range (semitones) — bounded so grains fit in the look-ahead.
pub const FORMANT_RANGE: f32 = 5.0;
pub const PITCH_RANGE: f32 = 12.0;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "gui", derive(serde::Serialize, serde::Deserialize))]
pub enum Scale {
    Chromatic,
    Major,
    Minor,
}

impl Scale {
    fn degrees(self) -> &'static [i32] {
        match self {
            Scale::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            Scale::Major => &[0, 2, 4, 5, 7, 9, 11],
            Scale::Minor => &[0, 2, 3, 5, 7, 8, 10],
        }
    }
    fn to_u8(self) -> u8 {
        self as u8
    }
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Scale::Major,
            2 => Scale::Minor,
            _ => Scale::Chromatic,
        }
    }
}

/// Extra time-domain effect after the pitch engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "gui", derive(serde::Serialize, serde::Deserialize))]
pub enum Extra {
    None,
    /// Telephone / walkie-talkie: 300–3400 Hz band + light saturation.
    Radio,
    /// Single echo, 220 ms, a few repeats.
    Echo,
    /// Large room / cave reverb.
    Reverb,
}

impl Extra {
    fn to_u8(self) -> u8 {
        self as u8
    }
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Extra::Radio,
            2 => Extra::Echo,
            3 => Extra::Reverb,
            _ => Extra::None,
        }
    }
}

/// Ready-made voices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "gui", derive(serde::Serialize, serde::Deserialize))]
pub enum FxPreset {
    Off,
    /// Deeper, darker voice.
    Deep,
    /// Higher, brighter voice.
    High,
    /// Child / cartoon.
    Child,
    /// Monster.
    Monster,
    /// Monotone robot.
    Robot,
    /// Hard auto-tune (chromatic).
    AutoTune,
    Radio,
    Echo,
    Cave,
}

impl FxPreset {
    pub const ALL: [FxPreset; 10] = [
        FxPreset::Off,
        FxPreset::Deep,
        FxPreset::High,
        FxPreset::Child,
        FxPreset::Monster,
        FxPreset::Robot,
        FxPreset::AutoTune,
        FxPreset::Radio,
        FxPreset::Echo,
        FxPreset::Cave,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "gui", derive(serde::Serialize, serde::Deserialize))]
pub struct FxSettings {
    /// Master switch. Off = no processing and no added latency.
    pub enabled: bool,
    /// Pitch shift in semitones (−12..=12).
    pub pitch: f32,
    /// Formant shift in semitones (−5..=5); 0 keeps the natural timbre.
    pub formant: f32,
    pub autotune: bool,
    /// Key root, 0 = C … 11 = B.
    pub key: u8,
    pub scale: Scale,
    /// Retune time in ms (0 = instant / robotic, ~150 = natural).
    pub retune_ms: f32,
    /// Flatten the pitch to one note (robot voice).
    pub robot: bool,
    pub extra: Extra,
}

impl Default for FxSettings {
    fn default() -> Self {
        Self::preset(FxPreset::Off)
    }
}

impl FxSettings {
    pub fn preset(p: FxPreset) -> Self {
        let base = Self {
            enabled: true,
            pitch: 0.0,
            formant: 0.0,
            autotune: false,
            key: 0,
            scale: Scale::Chromatic,
            retune_ms: 0.0,
            robot: false,
            extra: Extra::None,
        };
        match p {
            FxPreset::Off => Self { enabled: false, ..base },
            FxPreset::Deep => Self { pitch: -4.0, formant: -2.0, ..base },
            FxPreset::High => Self { pitch: 5.0, formant: 3.0, ..base },
            FxPreset::Child => Self { pitch: 8.0, formant: 5.0, ..base },
            FxPreset::Monster => Self { pitch: -8.0, formant: -5.0, extra: Extra::Reverb, ..base },
            FxPreset::Robot => Self { robot: true, extra: Extra::Echo, ..base },
            FxPreset::AutoTune => Self { autotune: true, retune_ms: 0.0, ..base },
            FxPreset::Radio => Self { extra: Extra::Radio, ..base },
            FxPreset::Echo => Self { extra: Extra::Echo, ..base },
            FxPreset::Cave => Self { pitch: -1.0, extra: Extra::Reverb, ..base },
        }
    }

    pub fn matching_preset(&self) -> Option<FxPreset> {
        FxPreset::ALL.into_iter().find(|p| {
            let s = Self::preset(*p);
            if !self.enabled && !s.enabled {
                return true;
            }
            s.enabled == self.enabled
                && (s.pitch - self.pitch).abs() < 0.05
                && (s.formant - self.formant).abs() < 0.05
                && s.autotune == self.autotune
                && (!s.autotune || (s.key == self.key && s.scale == self.scale && (s.retune_ms - self.retune_ms).abs() < 0.5))
                && s.robot == self.robot
                && s.extra == self.extra
        })
    }

    /// Does the PSOLA engine need to run (and add its latency)?
    fn needs_pitch_engine(&self) -> bool {
        self.enabled
            && (self.pitch.abs() > 0.01 || self.formant.abs() > 0.01 || self.autotune || self.robot)
    }
}

/// Lock-free live parameters.
#[derive(Debug)]
pub struct FxControls {
    enabled: AtomicBool,
    pitch: AtomicU32,
    formant: AtomicU32,
    autotune: AtomicBool,
    key: AtomicU8,
    scale: AtomicU8,
    retune: AtomicU32,
    robot: AtomicBool,
    extra: AtomicU8,
}

impl FxControls {
    pub fn new(s: FxSettings) -> Self {
        let c = Self {
            enabled: AtomicBool::new(false),
            pitch: AtomicU32::new(0),
            formant: AtomicU32::new(0),
            autotune: AtomicBool::new(false),
            key: AtomicU8::new(0),
            scale: AtomicU8::new(0),
            retune: AtomicU32::new(0),
            robot: AtomicBool::new(false),
            extra: AtomicU8::new(0),
        };
        c.set(s);
        c
    }

    pub fn set(&self, s: FxSettings) {
        let r = Ordering::Relaxed;
        self.enabled.store(s.enabled, r);
        self.pitch.store(s.pitch.clamp(-PITCH_RANGE, PITCH_RANGE).to_bits(), r);
        self.formant
            .store(s.formant.clamp(-FORMANT_RANGE, FORMANT_RANGE).to_bits(), r);
        self.autotune.store(s.autotune, r);
        self.key.store(s.key % 12, r);
        self.scale.store(s.scale.to_u8(), r);
        self.retune.store(s.retune_ms.clamp(0.0, 1000.0).to_bits(), r);
        self.robot.store(s.robot, r);
        self.extra.store(s.extra.to_u8(), r);
    }

    pub fn get(&self) -> FxSettings {
        let r = Ordering::Relaxed;
        FxSettings {
            enabled: self.enabled.load(r),
            pitch: f32::from_bits(self.pitch.load(r)),
            formant: f32::from_bits(self.formant.load(r)),
            autotune: self.autotune.load(r),
            key: self.key.load(r),
            scale: Scale::from_u8(self.scale.load(r)),
            retune_ms: f32::from_bits(self.retune.load(r)),
            robot: self.robot.load(r),
            extra: Extra::from_u8(self.extra.load(r)),
        }
    }
}

// ---------------------------------------------------------------------------
// Pitch detection (YIN, decimated)
// ---------------------------------------------------------------------------

/// Streaming YIN tracker. Publishes (centre time, period in 48 kHz samples,
/// 0 = unvoiced) every 5 ms.
struct PitchTracker {
    lp1: Biquad,
    lp2: Biquad,
    phase: usize,
    dec: Vec<f32>,
    dec_count: u64,
    since_hop: usize,
    diff: Vec<f32>,
    est_time: Vec<u64>,
    est_period: Vec<f32>,
    est_count: usize,
    raw: [f32; 3],
    threshold: f32,
    fallback: f32,
}

impl PitchTracker {
    fn new() -> Self {
        Self {
            // Anti-alias for the 4× decimation (12 kHz, voice f0 ≤ 800 Hz).
            lp1: Biquad::lowpass(2600.0),
            lp2: Biquad::lowpass(2600.0),
            phase: 0,
            dec: vec![0.0; DEC_HIST],
            dec_count: 0,
            since_hop: 0,
            diff: vec![0.0; TAU_MAX + 2],
            est_time: vec![0; PITCH_HIST],
            est_period: vec![0.0; PITCH_HIST],
            est_count: 0,
            raw: [0.0; 3],
            threshold: YIN_THRESHOLD,
            fallback: YIN_FALLBACK,
        }
    }

    /// Feed one 48 kHz sample with absolute index `t`.
    fn push(&mut self, x: f32, t: u64) {
        let mut v = [x];
        self.lp1.process(&mut v);
        self.lp2.process(&mut v);
        self.phase += 1;
        if self.phase < DECIM {
            return;
        }
        self.phase = 0;
        self.dec[(self.dec_count as usize) % DEC_HIST] = v[0];
        self.dec_count += 1;
        self.since_hop += 1;
        if self.since_hop >= YIN_HOP && self.dec_count as usize >= YIN_W + TAU_MAX {
            self.since_hop = 0;
            self.analyse(t);
        }
    }

    fn analyse(&mut self, t_now: u64) {
        let n = YIN_W + TAU_MAX;
        let start = self.dec_count as usize - n;
        let at = |i: usize| self.dec[(start + i) % DEC_HIST];
        // Energy gate.
        let mut e = 0.0f32;
        for i in 0..YIN_W {
            let v = at(i);
            e += v * v;
        }
        let rms = (e / YIN_W as f32).sqrt();
        let mut period = 0.0f32;
        if rms > 10f32.powf(-55.0 / 20.0) {
            // Difference function + cumulative mean normalisation.
            self.diff[0] = 1.0;
            let mut running = 0.0f32;
            for tau in 1..=TAU_MAX {
                let mut d = 0.0f32;
                for j in 0..YIN_W {
                    let a = at(j) - at(j + tau);
                    d += a * a;
                }
                running += d;
                self.diff[tau] = if running > 0.0 { d * tau as f32 / running } else { 1.0 };
            }
            let mut tau = TAU_MIN;
            let mut found = 0usize;
            while tau < TAU_MAX {
                if self.diff[tau] < self.threshold {
                    while tau + 1 < TAU_MAX && self.diff[tau + 1] < self.diff[tau] {
                        tau += 1;
                    }
                    found = tau;
                    break;
                }
                tau += 1;
            }
            if found == 0 && self.fallback > 0.0 {
                // No dip under the threshold: accept the global minimum if it
                // is still clearly periodic (breathy / low voices).
                let (mut best, mut bv) = (0usize, f32::MAX);
                for t in TAU_MIN..TAU_MAX {
                    if self.diff[t] < bv {
                        bv = self.diff[t];
                        best = t;
                    }
                }
                if bv < self.fallback {
                    found = best;
                }
            }
            if found > 0 {
                // Parabolic interpolation around the minimum.
                let (a, b, c) = (self.diff[found - 1], self.diff[found], self.diff[found + 1]);
                let den = a - 2.0 * b + c;
                let shift = if den.abs() > 1e-9 { 0.5 * (a - c) / den } else { 0.0 };
                period = (found as f32 + shift.clamp(-0.5, 0.5)) * DECIM as f32;
            }
        }
        // Median of the last three raw estimates (unvoiced counts as 0).
        self.raw = [self.raw[1], self.raw[2], period];
        let mut m = self.raw;
        m.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let smoothed = if period == 0.0 { 0.0 } else if m[1] == 0.0 { period } else { m[1] };
        // Centre of the analysis window, in 48 kHz samples. The two low-pass
        // biquads add ~2 samples of group delay at voice frequencies: ignored.
        let centre = t_now.saturating_sub((n * DECIM / 2) as u64);
        let i = self.est_count % PITCH_HIST;
        self.est_time[i] = centre;
        self.est_period[i] = smoothed;
        self.est_count += 1;
    }

    /// Period (48 kHz samples) of the estimate closest to `t`; 0 = unvoiced.
    fn period_at(&self, t: u64) -> f32 {
        let n = self.est_count.min(PITCH_HIST);
        let mut best = (u64::MAX, 0.0f32);
        for k in 0..n {
            let i = (self.est_count - 1 - k) % PITCH_HIST;
            let d = self.est_time[i].abs_diff(t);
            if d < best.0 {
                best = (d, self.est_period[i]);
            }
            if self.est_time[i] + 2000 < t {
                break;
            }
        }
        best.1
    }

    fn reset(&mut self) {
        self.est_count = 0;
        self.raw = [0.0; 3];
    }
}

// ---------------------------------------------------------------------------
// PSOLA engine
// ---------------------------------------------------------------------------

struct Psola {
    hist: Vec<f32>,
    /// Overlap-add accumulator (output timeline).
    acc: Vec<f32>,
    /// Absolute index of the next input sample.
    now: u64,
    /// Output timeline: next sample to emit corresponds to input time
    /// `emitted` (negative during start-up = silence). Always
    /// `emitted == now - FX_LATENCY` between calls once running.
    emitted: i64,
    /// Analysis marks (absolute positions) and their periods (0 = unvoiced).
    marks: Vec<(i64, f32)>,
    mark_head: usize,
    mark_count: usize,
    next_ta: i64,
    /// Next synthesis mark (same timeline as the input).
    next_ts: f64,
    tracker: PitchTracker,
    /// Smoothed auto-tune correction (semitones).
    correction: f32,
}

impl Psola {
    fn new() -> Self {
        let mut p = Self {
            hist: vec![0.0; HIST],
            acc: vec![0.0; HIST],
            now: 0,
            emitted: 0,
            marks: vec![(0, 0.0); MARKS],
            mark_head: 0,
            mark_count: 0,
            next_ta: 0,
            next_ts: 0.0,
            tracker: PitchTracker::new(),
            correction: 0.0,
        };
        p.restart();
        p
    }

    /// Input sample at absolute time `i` (0 outside the valid window).
    #[inline]
    fn h(&self, i: i64) -> f32 {
        if i < 0 || i as u64 >= self.now || self.now - i as u64 > (HIST - 1) as u64 {
            return 0.0;
        }
        self.hist[(i as usize) % HIST]
    }

    #[inline]
    fn h_frac(&self, pos: f64) -> f32 {
        let i = pos.floor();
        let f = (pos - i) as f32;
        let i = i as i64;
        self.h(i) * (1.0 - f) + self.h(i + 1) * f
    }

    /// Only record input (engine idle): keeps the history warm so the engine
    /// can start without a gap.
    fn feed(&mut self, frame: &[f32]) {
        for &x in frame {
            self.hist[(self.now as usize) % HIST] = x;
            self.now += 1;
        }
    }

    /// Restart so that output resumes at `now - FX_LATENCY`.
    fn restart(&mut self) {
        self.acc.fill(0.0);
        self.emitted = self.now as i64 - FX_LATENCY as i64;
        self.next_ta = self.emitted - 2 * H_MAX as i64;
        self.next_ts = (self.emitted - H_MAX as i64) as f64;
        self.mark_count = 0;
        self.mark_head = 0;
        self.correction = 0.0;
        self.tracker.reset();
    }

    fn push_mark(&mut self, pos: i64, period: f32) {
        let i = (self.mark_head + self.mark_count) % MARKS;
        self.marks[i] = (pos, period);
        if self.mark_count < MARKS {
            self.mark_count += 1;
        } else {
            self.mark_head = (self.mark_head + 1) % MARKS;
        }
    }

    /// Analysis marks up to `limit`: pitch-synchronous while voiced, aligned
    /// to the strongest positive peak within ±P/4 so that consecutive grains
    /// are in phase; fixed 5 ms spacing while unvoiced.
    fn extend_analysis_marks(&mut self, limit: i64) {
        while self.next_ta <= limit {
            let p = if self.next_ta >= 0 { self.tracker.period_at(self.next_ta as u64) } else { 0.0 };
            let mut pos = self.next_ta;
            if p > 0.0 {
                let r = (p / 4.0) as i64;
                let mut best = f32::MIN;
                for i in (self.next_ta - r)..=(self.next_ta + r) {
                    let v = self.h(i);
                    if v > best {
                        best = v;
                        pos = i;
                    }
                }
            }
            if self.mark_count > 0 {
                let last = self.marks[(self.mark_head + self.mark_count - 1) % MARKS].0;
                if pos <= last + 8 {
                    pos = last + 8;
                }
            }
            self.push_mark(pos, p);
            let step = if p > 0.0 { p } else { UNVOICED_PERIOD };
            self.next_ta = pos + step.max(8.0) as i64;
        }
    }

    /// Mark `k` (0 = oldest kept).
    #[inline]
    fn mark(&self, k: usize) -> (i64, f32) {
        self.marks[(self.mark_head + k) % MARKS]
    }

    /// Index of the analysis mark nearest to `t`.
    fn nearest_mark(&self, t: f64) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for k in 0..self.mark_count {
            let d = (self.mark(k).0 as f64 - t).abs();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((k, d));
            }
        }
        best.map(|(k, _)| k)
    }

    /// Semitone shift for a voiced grain with input period `period`.
    fn shift_for(&mut self, period: f32, s: &FxSettings, dt_ms: f32) -> f32 {
        let f0 = SAMPLE_RATE / period;
        let midi = 69.0 + 12.0 * (f0 / 440.0).log2();
        if s.robot {
            // Monotone at A2 (110 Hz), transposed by the pitch setting.
            return (45.0 - midi) + s.pitch;
        }
        let mut shift = s.pitch;
        if s.autotune {
            let want = midi + s.pitch;
            let corr = snap(want, s.key, s.scale) - want;
            if s.retune_ms <= 0.5 {
                self.correction = corr;
            } else {
                let a = 1.0 - (-dt_ms / s.retune_ms).exp();
                self.correction += (corr - self.correction) * a;
            }
            shift += self.correction;
        }
        shift
    }

    /// Push `frame` in, replace it with the output delayed by `FX_LATENCY`.
    /// `identity` keeps the engine (and its latency) but changes nothing —
    /// used for the A/B comparison.
    fn process(&mut self, frame: &mut [f32], s: &FxSettings, identity: bool) {
        let n = frame.len();
        for &x in frame.iter() {
            self.hist[(self.now as usize) % HIST] = x;
            self.tracker.push(x, self.now);
            self.now += 1;
        }
        let emit_end = self.emitted + n as i64; // == now - FX_LATENCY
        let formant = if identity {
            1.0
        } else {
            2f64.powf(f64::from(s.formant.clamp(-FORMANT_RANGE, FORMANT_RANGE)) / 12.0)
        };

        self.extend_analysis_marks(emit_end + 2 * H_MAX as i64);

        // Every grain that can touch [emitted, emit_end) must be added now.
        //
        // Windows are asymmetric raised-cosine halves: the left half spans
        // back to the previous analysis mark, the right half forward to the
        // next one. Neighbouring unmodified grains therefore sum to exactly
        // 1, so "no change" reproduces the input bit-for-bit (up to float
        // rounding). Modified grains are placed `spacing` apart and scaled by
        // the inverse of their average window overlap; there is no per-sample
        // normalisation, which would erase the new periodicity.
        while self.next_ts <= (emit_end + H_MAX as i64) as f64 {
            let Some(k) = self.nearest_mark(self.next_ts) else { break };
            let (ta, period) = self.mark(k);
            let fallback = if period > 0.0 { period } else { UNVOICED_PERIOD };
            let clamp = |g: f32| g.clamp(8.0, H_MAX as f32);
            let right = clamp(if k + 1 < self.mark_count { (self.mark(k + 1).0 - ta) as f32 } else { fallback });
            let left = clamp(if k > 0 { (ta - self.mark(k - 1).0) as f32 } else { fallback });
            // The grain's own period (distance to the next peak-aligned mark)
            // is more accurate in time than the tracker's 5 ms estimate; use
            // it when it agrees with the tracker (guards against a missed or
            // spurious mark).
            let p_grain = if period > 0.0 && right > 0.7 * period && right < 1.4 * period { right } else { period };
            let shift = if period > 0.0 && !identity { self.shift_for(p_grain, s, right / 48.0) } else { 0.0 };
            let ratio = 2f32.powf(shift / 12.0).clamp(0.25, 4.0);

            let unchanged = (ratio - 1.0).abs() < 1e-4 && (formant - 1.0).abs() < 1e-9;
            let (ts, spacing, lo, ro, gain) = if unchanged {
                // Lay the grain exactly on its analysis mark; continue at the next.
                self.next_ts = ta as f64;
                (ta, right, left, right, 1.0)
            } else {
                let spacing = right / ratio;
                let f = formant as f32;
                let (lo, ro) = (left / f, right / f);
                // Average sum of windows of total length lo+ro at hop `spacing`
                // is (lo+ro)/(2·spacing): compensate so loudness is kept.
                let gain = (2.0 * spacing / (lo + ro)).min(2.0);
                (self.next_ts.round() as i64, spacing, lo, ro, gain)
            };

            let lo_i = lo.max(1.0) as i64;
            let ro_i = ro.max(1.0) as i64;
            for j in -lo_i..=ro_i {
                let o = ts + j;
                if o < self.emitted {
                    continue;
                }
                let w = if j < 0 {
                    0.5 + 0.5 * (std::f32::consts::PI * j as f32 / lo).cos()
                } else {
                    0.5 + 0.5 * (std::f32::consts::PI * j as f32 / ro).cos()
                };
                if w <= 0.0 {
                    continue;
                }
                let v = self.h_frac(ta as f64 + j as f64 * formant);
                self.acc[(o as usize) % HIST] += gain * w * v;
            }
            self.next_ts += f64::from(spacing.max(8.0));
        }

        // Emit the overlap-add result.
        for (i, out) in frame.iter_mut().enumerate() {
            let t = self.emitted + i as i64;
            if t < 0 {
                *out = 0.0;
                continue;
            }
            let idx = (t as usize) % HIST;
            *out = self.acc[idx];
            self.acc[idx] = 0.0;
        }
        self.emitted = emit_end;
    }
}

/// Nearest note of `key`/`scale` to the (fractional) MIDI pitch `m`.
fn snap(m: f32, key: u8, scale: Scale) -> f32 {
    let base = m.round() as i32;
    let mut best = (f32::MAX, m);
    for cand in (base - 3)..=(base + 3) {
        let deg = (cand - i32::from(key)).rem_euclid(12);
        if scale.degrees().contains(&deg) {
            let d = (cand as f32 - m).abs();
            if d < best.0 {
                best = (d, cand as f32);
            }
        }
    }
    best.1
}

// ---------------------------------------------------------------------------
// Extra effects
// ---------------------------------------------------------------------------

struct Radio {
    hp: Biquad,
    lp: Biquad,
}

impl Radio {
    fn new() -> Self {
        Self {
            hp: Biquad::highpass(300.0),
            lp: Biquad::lowpass(3400.0),
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        self.hp.process(x);
        self.lp.process(x);
        // Soft saturation; the band-limited signal loses ~3 dB, so drive a bit.
        let k = 2.5f32;
        let norm = 1.0 / k.tanh();
        for v in x.iter_mut() {
            *v = (k * *v * 1.4).tanh() * norm * 0.5;
        }
    }
}

struct Echo {
    buf: Vec<f32>,
    pos: usize,
}

impl Echo {
    const DELAY: usize = 10_560; // 220 ms
    const FEEDBACK: f32 = 0.35;
    const WET: f32 = 0.4;
    fn new() -> Self {
        Self { buf: vec![0.0; Self::DELAY], pos: 0 }
    }
    fn process(&mut self, x: &mut [f32]) {
        for v in x.iter_mut() {
            let d = self.buf[self.pos];
            self.buf[self.pos] = *v + d * Self::FEEDBACK;
            self.pos = (self.pos + 1) % Self::DELAY;
            *v += d * Self::WET;
        }
    }
    fn clear(&mut self) {
        self.buf.fill(0.0);
    }
}

/// Schroeder / Freeverb-style reverb: 4 damped combs + 2 all-passes.
struct Reverb {
    combs: [(Vec<f32>, usize, f32); 4],
    allpasses: [(Vec<f32>, usize); 2],
}

impl Reverb {
    const COMBS: [usize; 4] = [1695, 1760, 1623, 1548];
    const ALLPASSES: [usize; 2] = [605, 480];
    const FEEDBACK: f32 = 0.82;
    const DAMP: f32 = 0.3;
    const WET: f32 = 0.3;

    fn new() -> Self {
        Self {
            combs: Self::COMBS.map(|n| (vec![0.0; n], 0, 0.0)),
            allpasses: Self::ALLPASSES.map(|n| (vec![0.0; n], 0)),
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        for v in x.iter_mut() {
            let input = *v * 0.25;
            let mut sum = 0.0;
            for (buf, pos, filt) in self.combs.iter_mut() {
                let y = buf[*pos];
                *filt = y * (1.0 - Self::DAMP) + *filt * Self::DAMP;
                buf[*pos] = input + *filt * Self::FEEDBACK;
                *pos = (*pos + 1) % buf.len();
                sum += y;
            }
            for (buf, pos) in self.allpasses.iter_mut() {
                let b = buf[*pos];
                buf[*pos] = sum + b * 0.5;
                *pos = (*pos + 1) % buf.len();
                sum = b - sum;
            }
            *v += sum * Self::WET;
        }
    }
    fn clear(&mut self) {
        for (b, _, f) in self.combs.iter_mut() {
            b.fill(0.0);
            *f = 0.0;
        }
        for (b, _) in self.allpasses.iter_mut() {
            b.fill(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// The effects stage
// ---------------------------------------------------------------------------

/// Voice-effects stage for one stream.
pub struct FxChain {
    psola: Psola,
    engine_on: bool,
    radio: Radio,
    echo: Echo,
    reverb: Reverb,
    last_extra: Extra,
    limiter: Limiter,
    scratch: Vec<f32>,
}

impl Default for FxChain {
    fn default() -> Self {
        Self::new()
    }
}

impl FxChain {
    pub fn new() -> Self {
        Self {
            psola: Psola::new(),
            engine_on: false,
            radio: Radio::new(),
            echo: Echo::new(),
            reverb: Reverb::new(),
            last_extra: Extra::None,
            limiter: Limiter::new(),
            scratch: Vec::with_capacity(8192),
        }
    }

    /// Latency this stage currently adds (samples at 48 kHz).
    pub fn latency(&self) -> usize {
        if self.engine_on { FX_LATENCY } else { 0 }
    }

    /// Process one frame in place. `bypass` (A/B) keeps the current latency
    /// but applies no effect.
    pub fn process(&mut self, frame: &mut [f32], s: &FxSettings, bypass: bool) {
        let want_engine = s.needs_pitch_engine();

        if want_engine != self.engine_on {
            // Switch the pitch engine on/off: its output is FX_LATENCY behind
            // the direct signal, so crossfade the two over this frame.
            self.scratch.clear();
            self.scratch.extend_from_slice(frame);
            if want_engine {
                self.psola.restart();
                self.psola.process(frame, s, bypass);
            } else {
                self.psola.process(frame, s, true);
            }
            let n = frame.len() as f32;
            for (i, (y, d)) in frame.iter_mut().zip(&self.scratch).enumerate() {
                let a = (i as f32 + 0.5) / n; // 0 → 1 across the frame
                let (from, to) = if want_engine { (*d, *y) } else { (*y, *d) };
                *y = from * (1.0 - a) + to * a;
            }
            self.engine_on = want_engine;
        } else if self.engine_on {
            self.psola.process(frame, s, bypass);
        } else {
            self.psola.feed(frame);
        }

        let extra = if bypass || !s.enabled { Extra::None } else { s.extra };
        if extra != self.last_extra {
            self.echo.clear();
            self.reverb.clear();
            self.last_extra = extra;
        }
        match extra {
            Extra::None => {}
            Extra::Radio => self.radio.process(frame),
            Extra::Echo => self.echo.process(frame),
            Extra::Reverb => self.reverb.process(frame),
        }
        if s.enabled && !bypass {
            self.limiter.process(frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// Harmonic "voice": f0 with 1/h harmonics shaped by two formant
    /// resonances (700 Hz and 1200 Hz), like an /a/ vowel.
    fn vowel(f0: impl Fn(f32) -> f32, secs: f32) -> Vec<f32> {
        let n = (secs * SR) as usize;
        let mut phase = 0.0f32;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / SR;
            let f = f0(t);
            phase += f / SR;
            let mut v = 0.0;
            let mut h = 1;
            while (h as f32) * f < 4000.0 {
                let fh = h as f32 * f;
                let env = 1.0 / (1.0 + ((fh - 700.0) / 150.0).powi(2)) + 0.6 / (1.0 + ((fh - 1200.0) / 200.0).powi(2)) + 0.05;
                v += env * (2.0 * std::f32::consts::PI * h as f32 * phase).sin();
                h += 1;
            }
            out.push(v * 0.1);
        }
        out
    }

    fn run(x: &[f32], s: &FxSettings) -> Vec<f32> {
        let mut fx = FxChain::new();
        let mut out = Vec::with_capacity(x.len());
        for c in x.chunks(480) {
            let mut f = c.to_vec();
            fx.process(&mut f, s, false);
            out.extend_from_slice(&f);
        }
        out
    }

    /// Median f0 (Hz) measured by our own tracker over `x[from..]`.
    fn measure_f0(x: &[f32], from: usize) -> f32 {
        let mut t = PitchTracker::new();
        let mut v = Vec::new();
        for (i, &s) in x.iter().enumerate() {
            let before = t.est_count;
            t.push(s, i as u64);
            if t.est_count != before && i > from {
                let p = t.est_period[(t.est_count - 1) % PITCH_HIST];
                if p > 0.0 {
                    v.push(SR / p);
                }
            }
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// Spectral centroid (Hz) of 200–3000 Hz on a 4096-point Hann window
    /// from the middle of `x` (plain DFT, test only).
    fn centroid(x: &[f32]) -> f32 {
        let n = 4096;
        let start = x.len() / 2;
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for k in (200 * n / 48_000)..(3000 * n / 48_000) {
            let f = k as f64 * 48_000.0 / n as f64;
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for j in 0..n {
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * j as f64 / n as f64).cos();
                let a = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
                re += w * x[start + j] as f64 * a.cos();
                im += w * x[start + j] as f64 * a.sin();
            }
            let p = re * re + im * im;
            num += f * p;
            den += p;
        }
        (num / den) as f32
    }

    #[test]
    fn yin_tracks_pitch_within_one_percent() {
        for f0 in [78.0f32, 95.0, 120.0, 200.0, 330.0, 500.0, 700.0] {
            let x = vowel(|_| f0, 0.5);
            let m = measure_f0(&x, 6000);
            assert!((m / f0 - 1.0).abs() < 0.01, "{f0} Hz measured as {m:.1}");
        }
    }

    #[test]
    fn engine_without_changes_is_an_exact_delay() {
        // Voiced + unvoiced material through the running pitch engine in
        // identity mode (A/B): output == input delayed by FX_LATENCY.
        let mut x = vowel(|t| 140.0 + 30.0 * t, 1.0);
        let mut seed = 1u32;
        for v in x[20_000..30_000].iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.05;
        }
        let s = FxSettings { pitch: 3.0, ..FxSettings::preset(FxPreset::Off) };
        let s = FxSettings { enabled: true, ..s };
        let mut fx = FxChain::new();
        let mut out = Vec::new();
        for c in x.chunks(480) {
            let mut f = c.to_vec();
            fx.process(&mut f, &s, true);
            out.extend_from_slice(&f);
        }
        assert_eq!(fx.latency(), FX_LATENCY);
        // Skip the switch-on crossfade frame and the start-up.
        let mut max_err = 0.0f32;
        for i in (FX_LATENCY + 4_000)..x.len() {
            max_err = max_err.max((out[i] - x[i - FX_LATENCY]).abs());
        }
        assert!(max_err < 1e-4, "identity error {max_err}");
    }

    #[test]
    fn pitch_shift_hits_the_target_interval() {
        for (st, f0) in [(7.0f32, 150.0f32), (-5.0, 220.0), (12.0, 130.0), (-12.0, 260.0)] {
            let x = vowel(|_| f0, 1.5);
            let s = FxSettings { pitch: st, ..FxSettings::preset(FxPreset::Deep) };
            let s = FxSettings { formant: 0.0, ..s };
            let y = run(&x, &s);
            let want = f0 * 2f32.powf(st / 12.0);
            let got = measure_f0(&y, 24_000);
            assert!((got / want - 1.0).abs() < 0.015, "{st:+} st from {f0} Hz: want {want:.1}, got {got:.1}");
            assert!(y.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn pitch_shift_keeps_formants_and_formant_shift_moves_them() {
        let x = vowel(|_| 130.0, 1.5);
        let base = centroid(&x);
        let c = |pitch: f32, formant: f32| {
            let s = FxSettings { pitch, formant, ..FxSettings::preset(FxPreset::High) };
            centroid(&run(&x, &s)) / base
        };
        // Pitch +5 st alone: the spectral envelope (formants) stays put.
        let pitch_only = c(5.0, 0.0);
        assert!((0.9..1.12).contains(&pitch_only), "pitch-only centroid ratio {pitch_only:.3}");
        // Formant +5 st (×1.335): the envelope moves up by about that factor.
        let formant_up = c(5.0, 5.0);
        assert!((1.2..1.45).contains(&formant_up), "formant +5 centroid ratio {formant_up:.3}");
        // Formant −5 st moves it down.
        let formant_down = c(0.0, -5.0);
        assert!((0.68..0.85).contains(&formant_down), "formant -5 centroid ratio {formant_down:.3}");
    }

    #[test]
    fn autotune_snaps_a_glide_to_semitones() {
        // Slow glide 205 → 235 Hz (G#3 … A#3): hard chromatic auto-tune must
        // output only notes, never the in-between pitches.
        let x = vowel(|t| 205.0 + 20.0 * t, 1.5);
        let s = FxSettings::preset(FxPreset::AutoTune);
        let y = run(&x, &s);
        let mut t = PitchTracker::new();
        let mut off = Vec::new();
        for (i, &v) in y.iter().enumerate() {
            let before = t.est_count;
            t.push(v, i as u64);
            if t.est_count != before && i > 12_000 {
                let p = t.est_period[(t.est_count - 1) % PITCH_HIST];
                if p > 0.0 {
                    let midi = 69.0 + 12.0 * ((SR / p) / 440.0).log2();
                    off.push((midi - midi.round()).abs());
                }
            }
        }
        off.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p90 = off[off.len() * 9 / 10];
        assert!(p90 < 0.2, "90th percentile distance to nearest semitone: {p90:.2}");
    }

    #[test]
    fn robot_is_monotone_at_110_hz() {
        for f0 in [150.0f32, 240.0] {
            let x = vowel(|t| f0 + 20.0 * t, 1.5);
            let s = FxSettings { extra: Extra::None, ..FxSettings::preset(FxPreset::Robot) };
            let got = measure_f0(&run(&x, &s), 24_000);
            assert!((got / 110.0 - 1.0).abs() < 0.02, "robot from {f0} Hz: {got:.1}");
        }
    }

    #[test]
    fn switching_effects_live_is_finite_and_bounded() {
        let x = vowel(|t| 160.0 + 40.0 * (t * 3.0).sin(), 3.0);
        let mut fx = FxChain::new();
        let presets = FxPreset::ALL;
        for (i, c) in x.chunks(480).enumerate() {
            let s = FxSettings::preset(presets[(i / 20) % presets.len()]);
            let mut f = c.to_vec();
            fx.process(&mut f, &s, i % 37 == 0);
            assert!(f.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "frame {i}");
        }
    }

    #[test]
    fn presets_round_trip_through_controls() {
        for p in FxPreset::ALL {
            let s = FxSettings::preset(p);
            assert_eq!(s.matching_preset(), Some(p), "{p:?}");
            assert_eq!(FxControls::new(s).get(), s);
        }
    }

    #[test]
    fn snap_respects_scale() {
        assert_eq!(snap(60.9, 0, Scale::Major), 60.0); // between C and D, nearer C
        assert_eq!(snap(61.4, 0, Scale::Major), 62.0); // C# is not in C major: D is nearer
        assert_eq!(snap(63.2, 0, Scale::Minor), 63.0); // Eb in C minor
        assert_eq!(snap(60.4, 0, Scale::Chromatic), 60.0);
    }
}

