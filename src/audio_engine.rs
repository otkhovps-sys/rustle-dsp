//! Device setup, stream configuration and the lock-free pipeline:
//!
//! ```text
//!  CPAL input callback          DSP worker thread                 CPAL output callback
//!  (device rate, N ch)          (not real-time, but RT-friendly)   (device rate, M ch)
//!  ┌──────────────────┐  SPSC   ┌──────────────────────────────┐  SPSC  ┌──────────────────┐
//!  │ downmix -> mono  │ ──────► │ resample -> 48k              │ ─────► │ mono -> M ch dup │
//!  │ try_push()       │ in_rb   │ RNNoise (480-sample frames)  │ out_rb │ try_pop()        │
//!  │ unpark(worker)   │         │ resample -> out rate         │        │ silence on empty │
//!  └──────────────────┘         └──────────────────────────────┘        └──────────────────┘
//! ```
//!
//! Real-time rules enforced in the two CPAL callbacks:
//!   * no locks, no allocation, no syscalls that can block;
//!   * the only cross-thread communication is the SPSC ring buffer
//!     (`ringbuf`, wait-free push/pop) plus relaxed atomics for metrics;
//!   * `Thread::unpark()` is used to wake the worker. It is wait-free for the
//!     caller (a futex wake) and is the standard way to signal a worker from
//!     an audio callback without a mutex/condvar.
//!
//! The DSP worker owns everything heavy (the RNNoise state, the resamplers)
//! and does all its allocation up front, so it also never allocates while
//! running.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    BufferSize, Device, FromSample, Host, Sample, SampleFormat, SampleRate, SizedSample, Stream,
    StreamConfig, SupportedBufferSize, SupportedStreamConfig,
};
use log::{debug, error, info, warn};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};

use crate::denoiser::{Denoiser, FRAME_SIZE};
use crate::resampler::{StreamResampler, RNNOISE_RATE};

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Input device selector: index from `list`, case-insensitive substring, or `None` = default.
    pub input: Option<String>,
    /// Output device selector (same rules).
    pub output: Option<String>,
    /// Requested hardware period size in frames (0 = let the backend decide).
    pub buffer_frames: u32,
    /// Force a specific device sample rate (useful to exercise the resampler).
    pub force_rate: Option<u32>,
    /// Bypass RNNoise (A/B testing); the rest of the pipeline is unchanged.
    pub bypass: bool,
    /// Print runtime statistics every N seconds (0 = never).
    pub stats_secs: u64,
}

// ---------------------------------------------------------------------------
// Shared state between threads (atomics only — no locks anywhere)
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct Metrics {
    /// Input callbacks that found `in_rb` full (samples were lost).
    pub input_overruns: AtomicU64,
    /// Output callbacks that found `out_rb` empty (silence was inserted).
    pub output_underruns: AtomicU64,
    /// Worker pushes that did not fit into `out_rb`.
    pub output_overruns: AtomicU64,
    /// Samples deliberately dropped by the latency clamp.
    pub dropped_samples: AtomicU64,
    /// Fill levels (frames) published by the worker / output callback.
    pub in_fill: AtomicUsize,
    pub out_fill: AtomicUsize,
    /// Last RNNoise VAD probability (f32 bits).
    pub vad_bits: AtomicU32,
    pub frames_processed: AtomicU64,
    /// Peak |sample| seen by the last input / output callback (f32 bits).
    pub in_peak_bits: AtomicU32,
    pub out_peak_bits: AtomicU32,
    /// Errors reported by CPAL on either stream (device unplugged, ...).
    pub stream_errors: AtomicU64,
    /// Largest number of frames requested by a single input / output callback.
    /// Sound-server plugins (PulseAudio/PipeWire ALSA plugins) often use much
    /// bigger and irregular chunks than the requested period.
    pub in_cb_max: AtomicUsize,
    pub out_cb_max: AtomicUsize,
}

impl Metrics {
    pub fn vad(&self) -> f32 {
        f32::from_bits(self.vad_bits.load(Ordering::Relaxed))
    }
    pub fn in_peak(&self) -> f32 {
        f32::from_bits(self.in_peak_bits.load(Ordering::Relaxed))
    }
    pub fn out_peak(&self) -> f32 {
        f32::from_bits(self.out_peak_bits.load(Ordering::Relaxed))
    }
}

/// Snapshot of all counters, convenient for UIs.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetricsSnapshot {
    pub frames_processed: u64,
    pub vad: f32,
    pub in_peak: f32,
    pub out_peak: f32,
    pub in_fill_ms: f64,
    pub out_fill_ms: f64,
    pub input_overruns: u64,
    pub output_underruns: u64,
    pub output_overruns: u64,
    pub dropped_samples: u64,
    pub stream_errors: u64,
    pub in_cb_max: usize,
    pub out_cb_max: usize,
}

/// Control flags for shutdown sequencing.
#[derive(Default)]
struct Control {
    /// Ask the worker to drain and exit.
    stop: AtomicBool,
    /// Worker has flushed everything into `out_rb` and exited.
    drained: AtomicBool,
    /// Output callback should play out whatever is left, ignoring the priming threshold.
    draining: AtomicBool,
}

// ---------------------------------------------------------------------------
// Device enumeration / selection
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    Input,
    Output,
}

impl Direction {
    fn label(self) -> &'static str {
        match self {
            Direction::Input => "input",
            Direction::Output => "output",
        }
    }
}

fn devices_for(host: &Host, dir: Direction) -> Result<Vec<Device>> {
    Ok(match dir {
        Direction::Input => host.input_devices()?.collect(),
        Direction::Output => host.output_devices()?.collect(),
    })
}

fn default_device_for(host: &Host, dir: Direction) -> Option<Device> {
    match dir {
        Direction::Input => host.default_input_device(),
        Direction::Output => host.default_output_device(),
    }
}

fn default_config_for(device: &Device, dir: Direction) -> Result<SupportedStreamConfig> {
    Ok(match dir {
        Direction::Input => device.default_input_config()?,
        Direction::Output => device.default_output_config()?,
    })
}

fn supported_configs_for(
    device: &Device,
    dir: Direction,
) -> Result<Vec<cpal::SupportedStreamConfigRange>> {
    Ok(match dir {
        Direction::Input => device.supported_input_configs()?.collect(),
        Direction::Output => device.supported_output_configs()?.collect(),
    })
}

/// Print all devices of the default host, in the same order/index that
/// `--input N` / `--output N` accept.
pub fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("Host: {:?}", host.id());

    for dir in [Direction::Input, Direction::Output] {
        println!();
        println!("{} devices:", capitalize(dir.label()));
        let default_name = default_device_for(&host, dir).and_then(|d| d.name().ok());
        let devices = devices_for(&host, dir)?;
        if devices.is_empty() {
            println!("  (none)");
            continue;
        }
        for (idx, dev) in devices.iter().enumerate() {
            let name = dev.name().unwrap_or_else(|_| "<unknown>".into());
            let marker = if Some(&name) == default_name.as_ref() { "*" } else { " " };
            match default_config_for(dev, dir) {
                Ok(cfg) => println!(
                    "{marker} [{idx}] {name}  (default: {} Hz, {} ch, {:?})",
                    cfg.sample_rate().0,
                    cfg.channels(),
                    cfg.sample_format()
                ),
                Err(e) => println!("{marker} [{idx}] {name}  (no default config: {e})"),
            }
            if let Ok(ranges) = supported_configs_for(dev, dir) {
                for r in ranges {
                    println!(
                        "        {} ch, {}..{} Hz, {:?}, buffer {}",
                        r.channels(),
                        r.min_sample_rate().0,
                        r.max_sample_rate().0,
                        r.sample_format(),
                        describe_buffer_size(r.buffer_size())
                    );
                }
            }
        }
    }
    println!();
    println!("* = default device. Select with --input/--output by index or name substring.");
    Ok(())
}

/// One entry of the device catalog used by UIs. `index` is what
/// `EngineConfig::input/output` accept as a selector.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub is_default: bool,
    pub default_rate: Option<u32>,
    pub default_channels: Option<u16>,
}

impl DeviceInfo {
    pub fn label(&self) -> String {
        let mut l = self.name.clone();
        if let (Some(r), Some(c)) = (self.default_rate, self.default_channels) {
            l.push_str(&format!("  ({r} Hz, {c} ch)"));
        }
        if self.is_default {
            l.push_str("  [default]");
        }
        l
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeviceCatalog {
    pub host: String,
    pub inputs: Vec<DeviceInfo>,
    pub outputs: Vec<DeviceInfo>,
}

/// Enumerate devices of the default host (same indices as `list`).
pub fn device_catalog() -> DeviceCatalog {
    let host = cpal::default_host();
    let collect = |dir: Direction| -> Vec<DeviceInfo> {
        let default_name = default_device_for(&host, dir).and_then(|d| d.name().ok());
        devices_for(&host, dir)
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(index, dev)| {
                let name = dev.name().unwrap_or_else(|_| format!("device {index}"));
                let cfg = default_config_for(dev, dir).ok();
                DeviceInfo {
                    index,
                    is_default: Some(&name) == default_name.as_ref(),
                    default_rate: cfg.as_ref().map(|c| c.sample_rate().0),
                    default_channels: cfg.as_ref().map(|c| c.channels()),
                    name,
                }
            })
            .collect()
    };
    DeviceCatalog {
        host: format!("{:?}", host.id()),
        inputs: collect(Direction::Input),
        outputs: collect(Direction::Output),
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn describe_buffer_size(b: &SupportedBufferSize) -> String {
    match b {
        SupportedBufferSize::Range { min, max } => format!("{min}..{max} frames"),
        SupportedBufferSize::Unknown => "unknown".into(),
    }
}

fn resolve_device(host: &Host, dir: Direction, selector: Option<&str>) -> Result<Device> {
    let Some(sel) = selector.map(str::trim).filter(|s| !s.is_empty()) else {
        return default_device_for(host, dir)
            .ok_or_else(|| anyhow!("no default {} device available", dir.label()));
    };

    let devices = devices_for(host, dir)?;

    // Numeric index as printed by `list`.
    if let Ok(idx) = sel.parse::<usize>() {
        return devices
            .into_iter()
            .nth(idx)
            .ok_or_else(|| anyhow!("{} device index {idx} out of range", dir.label()));
    }

    // Otherwise: exact (case-insensitive) match first, then substring.
    let needle = sel.to_lowercase();
    let mut substring: Option<Device> = None;
    for dev in devices {
        let name = dev.name().unwrap_or_default().to_lowercase();
        if name == needle {
            return Ok(dev);
        }
        if substring.is_none() && name.contains(&needle) {
            substring = Some(dev);
        }
    }
    substring.ok_or_else(|| anyhow!("no {} device matching {sel:?} (see `list`)", dir.label()))
}

// ---------------------------------------------------------------------------
// Stream configuration
// ---------------------------------------------------------------------------

struct StreamSpec {
    config: StreamConfig,
    format: SampleFormat,
    /// Period size actually requested (`None` = backend default).
    buffer_frames: Option<u32>,
}

impl StreamSpec {
    fn rate(&self) -> u32 {
        self.config.sample_rate.0
    }
    fn channels(&self) -> usize {
        usize::from(self.config.channels)
    }
}

/// Rank sample formats: f32 avoids any conversion; i16 is universally supported.
fn format_rank(f: SampleFormat) -> u8 {
    match f {
        SampleFormat::F32 => 0,
        SampleFormat::I16 => 1,
        SampleFormat::I32 => 2,
        SampleFormat::F64 => 3,
        SampleFormat::U16 => 4,
        SampleFormat::U32 => 5,
        SampleFormat::I8 => 6,
        SampleFormat::U8 => 7,
        _ => 8,
    }
}

/// Pick a stream configuration for `device`.
///
/// Strategy:
///  1. Keep the device's default channel count (so we never fight the OS mixer).
///  2. Prefer running the device natively at 48 kHz: it removes the resampler
///     from the chain entirely. If the device cannot do 48 kHz (or `force_rate`
///     asks for something else) we fall back to the device default rate and the
///     worker resamples.
///  3. Clamp the requested period size to what the backend reports as supported.
fn choose_config(
    device: &Device,
    dir: Direction,
    force_rate: Option<u32>,
    buffer_frames: u32,
) -> Result<StreamSpec> {
    let default = default_config_for(device, dir)
        .with_context(|| format!("querying default {} config", dir.label()))?;
    let channels = default.channels();
    let want_rate = force_rate.unwrap_or(RNNOISE_RATE);

    let mut chosen: Option<SupportedStreamConfig> = None;
    if default.sample_rate().0 == want_rate {
        chosen = Some(default.clone());
    } else if let Ok(ranges) = supported_configs_for(device, dir) {
        let mut best: Option<SupportedStreamConfig> = None;
        for r in ranges.into_iter().filter(|r| r.channels() == channels) {
            if let Some(cfg) = r.try_with_sample_rate(SampleRate(want_rate)) {
                let better = match &best {
                    None => true,
                    Some(b) => format_rank(cfg.sample_format()) < format_rank(b.sample_format()),
                };
                if better {
                    best = Some(cfg);
                }
            }
        }
        chosen = best;
    }

    let supported = match chosen {
        Some(c) => c,
        None => {
            if let Some(r) = force_rate {
                bail!(
                    "{} device does not support {} Hz with {} channels",
                    dir.label(),
                    r,
                    channels
                );
            }
            info!(
                "{} device cannot run at {} Hz natively; using {} Hz + resampler",
                dir.label(),
                RNNOISE_RATE,
                default.sample_rate().0
            );
            default
        }
    };

    // Period (buffer) size. Latency contribution of a period is
    // frames / rate: 128 frames @ 48 kHz = 2.67 ms per direction.
    let buffer = if buffer_frames == 0 {
        None
    } else {
        match supported.buffer_size() {
            SupportedBufferSize::Range { min, max } => {
                let clamped = buffer_frames.clamp(*min, *max);
                if clamped != buffer_frames {
                    warn!(
                        "{} buffer {} frames outside supported range {}..{}, using {}",
                        dir.label(),
                        buffer_frames,
                        min,
                        max,
                        clamped
                    );
                }
                Some(clamped)
            }
            SupportedBufferSize::Unknown => Some(buffer_frames),
        }
    };

    let format = supported.sample_format();
    let mut config: StreamConfig = supported.into();
    config.buffer_size = match buffer {
        Some(n) => BufferSize::Fixed(n),
        None => BufferSize::Default,
    };

    Ok(StreamSpec {
        config,
        format,
        buffer_frames: buffer,
    })
}

// ---------------------------------------------------------------------------
// Drift detection
// ---------------------------------------------------------------------------

/// Detects a queue that is *persistently* longer than needed.
///
/// Sound servers (PulseAudio, PipeWire, WASAPI shared mode, ...) pull audio
/// in irregular bursts, so the instantaneous queue level swings by several
/// periods without anything being wrong. Trimming on peaks would throw away
/// good audio (audible as crackle / missing syllables). What signals real
/// excess latency — clock drift between two devices, or a burst that was
/// never consumed — is the queue's *minimum* staying high. So we track the
/// lowest level seen over a window (~1 s of audio) and, at the end of the
/// window, report how much can be dropped while still leaving `floor`
/// frames at that lowest point (no underrun if the pattern repeats).
///
/// Pure arithmetic: safe to use inside an audio callback.
#[derive(Debug, Clone)]
struct BacklogWatch {
    window: usize,
    floor: usize,
    tolerance: usize,
    elapsed: usize,
    min_level: usize,
}

impl BacklogWatch {
    /// `window`: frames of audio per decision; `floor`: frames to keep at the
    /// lowest point; `tolerance`: excess (above `floor`) ignored to avoid
    /// chasing tiny drifts with frequent clicks.
    fn new(window: usize, floor: usize, tolerance: usize) -> Self {
        Self {
            window: window.max(1),
            floor,
            tolerance,
            elapsed: 0,
            min_level: usize::MAX,
        }
    }

    /// Record the queue level at its low point and that `advanced` frames of
    /// audio time passed. Returns the number of frames to drop now (0 = none).
    fn observe(&mut self, level: usize, advanced: usize) -> usize {
        self.min_level = self.min_level.min(level);
        self.elapsed += advanced;
        if self.elapsed < self.window {
            return 0;
        }
        let min = self.min_level;
        self.reset();
        if min > self.floor + self.tolerance {
            min - self.floor
        } else {
            0
        }
    }

    fn reset(&mut self) {
        self.elapsed = 0;
        self.min_level = usize::MAX;
    }
}

/// What the output callback should do this period.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputPlan {
    /// Not primed yet: output silence, consume nothing.
    Silence,
    /// Drop `skip` queued frames first (latency correction), then play.
    Play { skip: usize },
}

/// Output-queue policy, separated from the CPAL callback so it can be unit
/// tested with simulated devices (bursty pulls, clock drift).
///
/// * Priming: wait until `prime` frames are queued before playing. The worker
///   delivers audio in RNNoise-sized chunks (`chunk` = 10 ms), so between two
///   chunks the device consumes up to `chunk` frames on top of the request
///   being served. `prime = largest request + chunk + floor` therefore leaves
///   about `floor` frames at the queue's low point, whatever the phase
///   between producer and device.
/// * Drift correction: [`BacklogWatch`] over ~1 s windows.
/// * Hard cap: if the queue ever exceeds `prime + hard_max` (e.g. a device
///   that stalled for a long time, or two wildly mismatched clocks), trim
///   back to `prime` immediately so latency can never run away.
///
/// Pure arithmetic: RT-safe.
#[derive(Debug, Clone)]
struct OutputPacer {
    prime: usize,
    chunk: usize,
    floor: usize,
    hard_max: usize,
    primed: bool,
    watch: BacklogWatch,
}

impl OutputPacer {
    /// Priming level for a device requesting `request` frames per callback.
    fn prime_for(request: usize, q: &OutputQueueParams) -> usize {
        request + q.chunk + q.floor
    }

    fn new(period_hint: usize, q: OutputQueueParams) -> Self {
        Self {
            prime: Self::prime_for(period_hint, &q),
            chunk: q.chunk,
            floor: q.floor,
            hard_max: q.hard_max,
            primed: false,
            watch: BacklogWatch::new(q.window, q.floor, q.tolerance),
        }
    }

    /// Decide for a callback that wants `wanted` frames with `available` queued.
    fn plan(&mut self, available: usize, wanted: usize, draining: bool) -> OutputPlan {
        self.prime = self.prime.max(wanted + self.chunk + self.floor);
        if draining {
            return OutputPlan::Play { skip: 0 };
        }
        if !self.primed {
            if available < self.prime {
                return OutputPlan::Silence;
            }
            self.primed = true;
            self.watch.reset();
        }
        if available > self.prime + self.hard_max {
            self.watch.reset();
            return OutputPlan::Play {
                skip: available - self.prime,
            };
        }
        let low_point = available.saturating_sub(wanted);
        OutputPlan::Play {
            skip: self.watch.observe(low_point, wanted),
        }
    }

    /// The callback ran out of data: re-prime (one short gap, not crackle).
    fn on_underrun(&mut self) {
        self.primed = false;
        self.watch.reset();
    }
}

// ---------------------------------------------------------------------------
// CPAL callbacks (real-time context!)
// ---------------------------------------------------------------------------

/// Build the capture stream. `T` is the device's native sample type; samples are
/// converted to f32, down-mixed to mono and pushed to `in_rb`.
fn build_input_stream<T>(
    device: &Device,
    config: &StreamConfig,
    mut prod: HeapProd<f32>,
    metrics: Arc<Metrics>,
    worker: Thread,
) -> Result<Stream>
where
    T: SizedSample + Send + 'static,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels).max(1);
    let inv_channels = 1.0 / channels as f32;

    let err_metrics = metrics.clone();
    let data_cb = move |data: &[T], _: &cpal::InputCallbackInfo| {
        // RT-safe: only arithmetic, a wait-free push per frame, relaxed atomics.
        let mut overrun = false;
        let mut peak = 0.0f32;
        for frame in data.chunks_exact(channels) {
            // Downmix: arithmetic mean of all channels. Equal-weight averaging
            // keeps a mono source recorded on both channels at unity gain.
            let mut acc = 0.0f32;
            for s in frame {
                acc += f32::from_sample(*s);
            }
            let mono = acc * inv_channels;
            peak = peak.max(mono.abs());
            if prod.try_push(mono).is_err() {
                overrun = true;
            }
        }
        if overrun {
            metrics.input_overruns.fetch_add(1, Ordering::Relaxed);
        }
        metrics.in_peak_bits.store(peak.to_bits(), Ordering::Relaxed);
        metrics.in_cb_max.fetch_max(data.len() / channels, Ordering::Relaxed);
        // Wake the DSP worker. Futex wake = non-blocking for us.
        worker.unpark();
    };

    let err_cb = move |e: cpal::StreamError| {
        err_metrics.stream_errors.fetch_add(1, Ordering::Relaxed);
        error!("input stream error: {e}");
    };
    device
        .build_input_stream(config, data_cb, err_cb, None)
        .context("building input stream")
}

/// Build the playback stream. Pops mono samples from `out_rb`, duplicates them
/// into every output channel, converts to the device's native sample type.
fn build_output_stream<T>(
    device: &Device,
    config: &StreamConfig,
    mut cons: HeapCons<f32>,
    metrics: Arc<Metrics>,
    ctl: Arc<Control>,
    prime_frames: usize,
    q: OutputQueueParams,
) -> Result<Stream>
where
    T: SizedSample + FromSample<f32> + Send + 'static,
{
    let channels = usize::from(config.channels).max(1);
    let silence = T::from_sample(0.0f32);
    let err_metrics = metrics.clone();

    // Priming, drift correction and the latency hard cap live in
    // `OutputPacer` (pure logic, unit-tested with simulated devices).
    let mut pacer = OutputPacer::new(prime_frames, q);

    let data_cb = move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
        let mut available = cons.occupied_len();
        let wanted = data.len() / channels;
        metrics.out_cb_max.fetch_max(wanted, Ordering::Relaxed);

        let draining = ctl.draining.load(Ordering::Relaxed);
        match pacer.plan(available, wanted, draining) {
            OutputPlan::Silence => {
                metrics.out_fill.store(available, Ordering::Relaxed);
                data.fill(silence);
                return;
            }
            OutputPlan::Play { skip } if skip > 0 => {
                // Only the consumer can drop from an SPSC ring; `skip` just
                // advances the read index (wait-free, RT-safe).
                let skipped = cons.skip(skip);
                metrics.dropped_samples.fetch_add(skipped as u64, Ordering::Relaxed);
                available -= skipped;
            }
            OutputPlan::Play { .. } => {}
        }
        metrics.out_fill.store(available, Ordering::Relaxed);

        let mut underrun = false;
        let mut peak = 0.0f32;
        for frame in data.chunks_exact_mut(channels) {
            let s = match cons.try_pop() {
                Some(s) => s,
                None => {
                    underrun = true;
                    0.0
                }
            };
            peak = peak.max(s.abs());
            let v = T::from_sample(s);
            // Mono -> M channels: identical sample on every channel.
            for out in frame.iter_mut() {
                *out = v;
            }
        }
        // Any trailing partial frame (should not happen with sane backends).
        let rem = data.len() % channels;
        if rem != 0 {
            let n = data.len();
            data[n - rem..].fill(silence);
        }

        metrics.out_peak_bits.store(peak.to_bits(), Ordering::Relaxed);
        if underrun && !draining {
            metrics.output_underruns.fetch_add(1, Ordering::Relaxed);
            pacer.on_underrun();
        }
    };

    let err_cb = move |e: cpal::StreamError| {
        err_metrics.stream_errors.fetch_add(1, Ordering::Relaxed);
        error!("output stream error: {e}");
    };
    device
        .build_output_stream(config, data_cb, err_cb, None)
        .context("building output stream")
}

/// Dispatch on the device's sample format. Falls back to `BufferSize::Default`
/// if the backend rejects the fixed period size.
fn build_input_any(
    device: &Device,
    spec: &mut StreamSpec,
    prod: HeapProd<f32>,
    metrics: Arc<Metrics>,
    worker: Thread,
) -> Result<Stream> {
    macro_rules! build {
        ($cfg:expr, $prod:expr) => {
            match spec.format {
                SampleFormat::F32 => build_input_stream::<f32>(device, $cfg, $prod, metrics.clone(), worker.clone()),
                SampleFormat::I16 => build_input_stream::<i16>(device, $cfg, $prod, metrics.clone(), worker.clone()),
                SampleFormat::U16 => build_input_stream::<u16>(device, $cfg, $prod, metrics.clone(), worker.clone()),
                SampleFormat::I32 => build_input_stream::<i32>(device, $cfg, $prod, metrics.clone(), worker.clone()),
                SampleFormat::U32 => build_input_stream::<u32>(device, $cfg, $prod, metrics.clone(), worker.clone()),
                SampleFormat::F64 => build_input_stream::<f64>(device, $cfg, $prod, metrics.clone(), worker.clone()),
                SampleFormat::I8 => build_input_stream::<i8>(device, $cfg, $prod, metrics.clone(), worker.clone()),
                SampleFormat::U8 => build_input_stream::<u8>(device, $cfg, $prod, metrics.clone(), worker.clone()),
                other => bail!("unsupported input sample format {other:?}"),
            }
        };
    }

    // The producer is moved into the callback; if the first attempt fails we
    // need it back, so attempt the fixed size through an Option dance.
    let mut prod_slot = Some(prod);
    if spec.buffer_frames.is_some() {
        // Try with the fixed period size first.
        let p = prod_slot.take().expect("producer present");
        // We cannot recover `p` from a failed closure, so probe the config by
        // building; on error we re-create the stream with the default size
        // using a *fresh* ring buffer (the old producer is dropped with the
        // failed closure, but no consumer has read from it yet, so nothing is lost).
        match build!(&spec.config, p) {
            Ok(s) => return Ok(s),
            Err(e) => {
                warn!(
                    "input stream rejected fixed buffer of {} frames ({e:#}); retrying with backend default",
                    spec.buffer_frames.unwrap_or(0)
                );
                spec.config.buffer_size = BufferSize::Default;
                spec.buffer_frames = None;
                bail!(RetryWithDefault);
            }
        }
    }
    let p = prod_slot.take().expect("producer present");
    build!(&spec.config, p)
}

fn build_output_any(
    device: &Device,
    spec: &mut StreamSpec,
    cons: HeapCons<f32>,
    metrics: Arc<Metrics>,
    ctl: Arc<Control>,
    prime_frames: usize,
    q: OutputQueueParams,
) -> Result<Stream> {
    macro_rules! build {
        ($cfg:expr, $cons:expr) => {
            match spec.format {
                SampleFormat::F32 => build_output_stream::<f32>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames, q),
                SampleFormat::I16 => build_output_stream::<i16>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames, q),
                SampleFormat::U16 => build_output_stream::<u16>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames, q),
                SampleFormat::I32 => build_output_stream::<i32>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames, q),
                SampleFormat::U32 => build_output_stream::<u32>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames, q),
                SampleFormat::F64 => build_output_stream::<f64>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames, q),
                SampleFormat::I8 => build_output_stream::<i8>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames, q),
                SampleFormat::U8 => build_output_stream::<u8>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames, q),
                other => bail!("unsupported output sample format {other:?}"),
            }
        };
    }

    let mut cons_slot = Some(cons);
    if spec.buffer_frames.is_some() {
        let c = cons_slot.take().expect("consumer present");
        match build!(&spec.config, c) {
            Ok(s) => return Ok(s),
            Err(e) => {
                warn!(
                    "output stream rejected fixed buffer of {} frames ({e:#}); retrying with backend default",
                    spec.buffer_frames.unwrap_or(0)
                );
                spec.config.buffer_size = BufferSize::Default;
                spec.buffer_frames = None;
                bail!(RetryWithDefault);
            }
        }
    }
    let c = cons_slot.take().expect("consumer present");
    build!(&spec.config, c)
}

/// Drift-correction parameters for the output queue (frames at output rate).
#[derive(Debug, Clone, Copy)]
struct OutputQueueParams {
    /// Size of the producer's chunks (one RNNoise frame at the output rate).
    chunk: usize,
    /// Decision window (≈ 1 s of audio).
    window: usize,
    /// Frames to keep at the queue's lowest point (jitter margin).
    floor: usize,
    /// Excess above `floor` that is tolerated without trimming.
    tolerance: usize,
    /// Queue length above `prime` that is trimmed immediately (hard cap).
    hard_max: usize,
}

/// Marker error: the fixed-size attempt failed and the ring buffer endpoint
/// was consumed by the failed closure; the caller must recreate the ring buffer
/// and call again (the spec has already been downgraded to `BufferSize::Default`).
#[derive(Debug)]
struct RetryWithDefault;

impl std::fmt::Display for RetryWithDefault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("retry with default buffer size")
    }
}
impl std::error::Error for RetryWithDefault {}

// ---------------------------------------------------------------------------
// DSP worker
// ---------------------------------------------------------------------------

struct WorkerParams {
    in_rate: u32,
    out_rate: u32,
    bypass: bool,
    /// Frames (at `in_rate`) above which the input backlog is trimmed. This
    /// only fires after a long stall of this thread; normal device drift is
    /// corrected on the output side.
    in_backlog_limit: usize,
    /// Frames (at `in_rate`) the backlog is trimmed *to*.
    in_backlog_target: usize,
}

/// Scale a frame count from one rate to another, rounding up.
fn scale_frames(frames: usize, from: u32, to: u32) -> usize {
    (frames as u64 * u64::from(to)).div_ceil(u64::from(from)) as usize
}

fn worker_loop(
    mut in_cons: HeapCons<f32>,
    mut out_prod: HeapProd<f32>,
    p: WorkerParams,
    ctl: Arc<Control>,
    metrics: Arc<Metrics>,
) -> Result<()> {
    // Everything below is allocated once, before the loop.
    let mut to48 = StreamResampler::new(p.in_rate, RNNOISE_RATE)?;
    let mut from48 = StreamResampler::new(RNNOISE_RATE, p.out_rate)?;
    let mut denoiser = Denoiser::new(p.bypass);

    // Pull at most this many frames per iteration. Slightly more than one
    // RNNoise frame at the highest plausible device rate, so one wake-up is
    // normally enough to complete a frame.
    const PULL_FRAMES: usize = 1024;
    let mut scratch = vec![0.0f32; PULL_FRAMES];

    // Stage buffers. Capacities are generous upper bounds (ratio ≤ 4, plus a
    // resampler chunk and a full RNNoise frame) so `extend_from_slice` never
    // reallocates in steady state.
    let mut stage48: Vec<f32> = Vec::with_capacity(PULL_FRAMES * 4 + 2 * FRAME_SIZE);
    let mut clean48: Vec<f32> = Vec::with_capacity(stage48.capacity() + FRAME_SIZE);
    let mut outdev: Vec<f32> = Vec::with_capacity(clean48.capacity() * 4);

    // Worst-case per-stage output exceeding capacity would only cost one
    // reallocation on the worker thread (never in a CPAL callback).

    let mut idle_wakeups: u64 = 0;

    debug!(
        "worker ready: in {} Hz -> 48000 ({}), 48000 -> out {} Hz ({})",
        to48.in_rate(),
        if to48.is_passthrough() { "pass-through" } else { "sinc resampler" },
        from48.out_rate(),
        if from48.is_passthrough() { "pass-through" } else { "sinc resampler" },
    );

    loop {
        let stopping = ctl.stop.load(Ordering::Acquire);
        let in_backlog = in_cons.occupied_len();
        metrics.in_fill.store(in_backlog, Ordering::Relaxed);

        // --- Stall recovery ------------------------------------------------
        // If this thread was descheduled for a long time (system under heavy
        // load, laptop resumed from sleep) the input ring holds a large
        // backlog; playing it all would add that much latency permanently.
        // Drop the oldest audio instead (one click, latency stays bounded).
        // Ordinary drift / bursty devices are handled by the output callback.
        if !stopping && in_backlog > p.in_backlog_limit {
            let drop = in_backlog - p.in_backlog_target;
            let skipped = in_cons.skip(drop);
            metrics.dropped_samples.fetch_add(skipped as u64, Ordering::Relaxed);
            debug!("stall recovery: dropped {skipped} input frames (backlog {in_backlog})");
        }

        let n = in_cons.pop_slice(&mut scratch);
        if n == 0 {
            if stopping {
                break;
            }
            idle_wakeups += 1;
            // Sleep until the input callback unparks us. The timeout is only a
            // safety net (e.g. if the input stream died) and is short enough
            // not to matter for latency.
            thread::park_timeout(Duration::from_millis(2));
            continue;
        }

        // --- Stage 1: device rate -> 48 kHz ------------------------------
        stage48.clear();
        to48.process(&scratch[..n], &mut |x| stage48.extend_from_slice(x))?;

        // --- Stage 2: RNNoise on 480-sample frames -----------------------
        clean48.clear();
        denoiser.push(&stage48, &mut |f| clean48.extend_from_slice(f));

        // --- Stage 3: 48 kHz -> output device rate -----------------------
        outdev.clear();
        from48.process(&clean48, &mut |y| outdev.extend_from_slice(y))?;

        push_out(&mut out_prod, &outdev, &metrics);

        metrics
            .vad_bits
            .store(denoiser.last_vad().to_bits(), Ordering::Relaxed);
        metrics
            .frames_processed
            .store(denoiser.frames_processed(), Ordering::Relaxed);
    }

    // --- Shutdown flush: push everything still held in the three stages ----
    stage48.clear();
    to48.flush(&mut |x| stage48.extend_from_slice(x))?;
    clean48.clear();
    denoiser.push(&stage48, &mut |f| clean48.extend_from_slice(f));
    denoiser.flush(&mut |f| clean48.extend_from_slice(f));
    outdev.clear();
    from48.process(&clean48, &mut |y| outdev.extend_from_slice(y))?;
    from48.flush(&mut |y| outdev.extend_from_slice(y))?;
    push_out(&mut out_prod, &outdev, &metrics);

    debug_assert_eq!(denoiser.pending(), 0, "flush must leave no partial frame");
    debug!(
        "worker exiting: {} frames denoised, {} idle wake-ups",
        denoiser.frames_processed(),
        idle_wakeups
    );
    ctl.drained.store(true, Ordering::Release);
    Ok(())
}

fn push_out(out_prod: &mut HeapProd<f32>, data: &[f32], metrics: &Metrics) {
    if data.is_empty() {
        return;
    }
    let pushed = out_prod.push_slice(data);
    if pushed < data.len() {
        metrics.output_overruns.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Ring buffer capacity: one full second at the device rate, rounded up to a
/// power of two. Capacity is *not* latency — latency is governed by the fill
/// level (priming threshold + latency clamp). A large capacity only means a
/// scheduler hiccup causes a latency bump (then trimmed) instead of lost audio.
fn ring_capacity(rate: u32) -> usize {
    (rate as usize).next_power_of_two()
}

fn ms(frames: usize, rate: u32) -> f64 {
    frames as f64 * 1000.0 / f64::from(rate)
}

/// Estimated one-way latency, mic -> output, broken down per stage (ms).
#[derive(Debug, Clone, Copy, Default)]
pub struct LatencyBudget {
    pub input_period_ms: f64,
    pub rnnoise_frame_ms: f64,
    /// Input period + RNNoise framing (framing is counted once for the
    /// whole chain; see `latency_budget`).
    pub input_side_ms: f64,
    pub resample_in_ms: f64,
    pub resample_out_ms: f64,
    pub output_queue_ms: f64,
    pub output_queue_frames: usize,
    pub output_period_ms: f64,
    pub total_ms: f64,
}

/// Everything a UI wants to know about a running engine.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub host: String,
    pub input_name: String,
    pub output_name: String,
    pub input_rate: u32,
    pub output_rate: u32,
    pub input_channels: usize,
    pub output_channels: usize,
    pub input_format: String,
    pub output_format: String,
    pub input_period: Option<u32>,
    pub output_period: Option<u32>,
    pub bypass: bool,
    pub latency: LatencyBudget,
}

/// A running pipeline. Dropping the handle stops it gracefully.
///
/// CPAL streams are `!Send`, so they live on a dedicated control thread owned
/// by this handle; `start()` returns once the streams are playing and
/// `stop()` performs the ordered shutdown (stop capture → flush DSP → drain
/// output → stop playback).
pub struct Engine {
    info: EngineInfo,
    metrics: Arc<Metrics>,
    stop_flag: Arc<AtomicBool>,
    control: Option<JoinHandle<Result<()>>>,
}

enum ControlMsg {
    Ready(Box<EngineInfo>),
    Failed(anyhow::Error),
}

impl Engine {
    /// Open the devices, start the worker and both streams.
    pub fn start(cfg: EngineConfig) -> Result<Engine> {
        let metrics = Arc::new(Metrics::default());
        let stop_flag = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel::<ControlMsg>();

        let thread_metrics = metrics.clone();
        let thread_stop = stop_flag.clone();
        let control = thread::Builder::new()
            .name("crisp-control".into())
            .spawn(move || control_thread(cfg, thread_metrics, thread_stop, tx))
            .context("spawning control thread")?;

        match rx.recv() {
            Ok(ControlMsg::Ready(info)) => Ok(Engine {
                info: *info,
                metrics,
                stop_flag,
                control: Some(control),
            }),
            Ok(ControlMsg::Failed(e)) => {
                let _ = control.join();
                Err(e)
            }
            Err(_) => {
                // Thread died before reporting (panic).
                match control.join() {
                    Ok(Ok(())) => bail!("control thread exited before the engine started"),
                    Ok(Err(e)) => Err(e),
                    Err(_) => bail!("control thread panicked during start-up"),
                }
            }
        }
    }

    pub fn info(&self) -> &EngineInfo {
        &self.info
    }

    pub fn metrics(&self) -> &Arc<Metrics> {
        &self.metrics
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        snapshot(&self.metrics, self.info.input_rate, self.info.output_rate)
    }

    /// True once the control thread has exited (after `stop()` or a fatal error).
    pub fn is_finished(&self) -> bool {
        self.control.as_ref().is_none_or(|h| h.is_finished())
    }

    /// Ask the pipeline to stop without waiting (poll `is_finished()`).
    pub fn request_stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(h) = &self.control {
            h.thread().unpark();
        }
    }

    /// Graceful shutdown; blocks until the output queue has drained (≤ ~0.6 s).
    pub fn stop(mut self) -> Result<()> {
        self.shutdown()
    }

    fn shutdown(&mut self) -> Result<()> {
        self.request_stop();
        match self.control.take() {
            Some(h) => h
                .join()
                .map_err(|_| anyhow!("control thread panicked"))?,
            None => Ok(()),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Err(e) = self.shutdown() {
            error!("engine shutdown: {e:#}");
        }
    }
}

/// Convenience for the CLI: run until `running` is cleared, logging stats.
pub fn run(cfg: EngineConfig, running: Arc<AtomicBool>) -> Result<()> {
    run_with(cfg, running, |_| Ok(()))
}

/// Like [`run`], but calls `on_started` once the streams are playing (used to
/// route the output into a virtual microphone). An error stops the engine.
pub fn run_with(
    cfg: EngineConfig,
    running: Arc<AtomicBool>,
    on_started: impl FnOnce(&Engine) -> Result<()>,
) -> Result<()> {
    let stats_secs = cfg.stats_secs;
    let engine = Engine::start(cfg)?;
    log_info(engine.info());
    on_started(&engine)?;
    info!("running — press Ctrl+C to stop");

    let stats_every = (stats_secs > 0).then(|| Duration::from_secs(stats_secs));
    let mut last_stats = Instant::now();
    while running.load(Ordering::Relaxed) && !engine.is_finished() {
        thread::sleep(Duration::from_millis(50));
        if let Some(every) = stats_every {
            if last_stats.elapsed() >= every {
                last_stats = Instant::now();
                log_stats(&engine.snapshot());
            }
        }
    }

    info!("shutting down");
    let snap = engine.snapshot();
    engine.stop()?;
    log_stats(&snap);
    info!("stopped cleanly");
    Ok(())
}

/// Owns the CPAL streams for their whole lifetime.
fn control_thread(
    cfg: EngineConfig,
    metrics: Arc<Metrics>,
    stop_flag: Arc<AtomicBool>,
    tx: std::sync::mpsc::Sender<ControlMsg>,
) -> Result<()> {
    let started = match start_pipeline(&cfg, metrics.clone()) {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("{e:#}");
            let _ = tx.send(ControlMsg::Failed(e));
            bail!("{msg}");
        }
    };
    let Started {
        in_stream,
        out_stream,
        worker,
        ctl,
        info,
    } = started;

    if tx.send(ControlMsg::Ready(Box::new(info))).is_err() {
        // Nobody is listening any more; shut down right away.
        stop_flag.store(true, Ordering::Release);
    }

    // Wait for a stop request (or the worker dying).
    while !stop_flag.load(Ordering::Acquire) && !worker.is_finished() {
        thread::park_timeout(Duration::from_millis(100));
    }

    // ---- Graceful shutdown ----------------------------------------------------
    // 1. Stop capturing: no new samples enter the pipeline.
    drop(in_stream);
    // 2. Tell the worker to process + flush whatever is left and exit.
    ctl.stop.store(true, Ordering::Release);
    worker.thread().unpark();
    let worker_result = worker
        .join()
        .map_err(|_| anyhow!("DSP worker thread panicked"));
    // 3. Let the output play out the remaining queued audio (bypass priming).
    ctl.draining.store(true, Ordering::Release);
    let deadline = Instant::now() + Duration::from_millis(500);
    while metrics.out_fill.load(Ordering::Relaxed) > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    // 4. Stop playback.
    drop(out_stream);

    worker_result?
}

struct Started {
    in_stream: Stream,
    out_stream: Stream,
    worker: JoinHandle<Result<()>>,
    ctl: Arc<Control>,
    info: EngineInfo,
}

fn start_pipeline(cfg: &EngineConfig, metrics: Arc<Metrics>) -> Result<Started> {
    let host = cpal::default_host();
    info!("audio host: {:?}", host.id());

    let in_dev = resolve_device(&host, Direction::Input, cfg.input.as_deref())?;
    let out_dev = resolve_device(&host, Direction::Output, cfg.output.as_deref())?;
    let in_name = in_dev.name().unwrap_or_else(|_| "<unknown>".into());
    let out_name = out_dev.name().unwrap_or_else(|_| "<unknown>".into());

    let mut in_spec = choose_config(&in_dev, Direction::Input, cfg.force_rate, cfg.buffer_frames)?;
    let mut out_spec =
        choose_config(&out_dev, Direction::Output, cfg.force_rate, cfg.buffer_frames)?;

    let ctl = Arc::new(Control::default());

    // Streams are built in a small retry loop because a backend may reject
    // the fixed period size; in that case the ring buffer endpoint moved into
    // the failed closure is gone, so we simply recreate both halves.
    let (in_stream, out_stream, worker, prime_frames) = loop {
        let in_rate = in_spec.rate();
        let out_rate = out_spec.rate();

        let (in_prod, in_cons) = HeapRb::<f32>::new(ring_capacity(in_rate)).split();
        let (out_prod, out_cons) = HeapRb::<f32>::new(ring_capacity(out_rate)).split();

        // ---- Frame sizing / latency parameters -----------------------------
        // Effective device period (frames). If the backend chooses, assume a
        // conservative 10 ms for the purposes of priming and clamping.
        let in_period = in_spec
            .buffer_frames
            .map(|n| n as usize)
            .unwrap_or_else(|| scale_frames(480, RNNOISE_RATE, in_rate));
        let out_period = out_spec
            .buffer_frames
            .map(|n| n as usize)
            .unwrap_or_else(|| scale_frames(480, RNNOISE_RATE, out_rate));
        let frame_out = scale_frames(FRAME_SIZE, RNNOISE_RATE, out_rate);
        let frame_in = scale_frames(FRAME_SIZE, RNNOISE_RATE, in_rate);

        // Output queue policy (see `OutputPacer` / `BacklogWatch`):
        //  * floor: aim for ~5 ms (half an RNNoise frame) at the queue's low
        //    point — the jitter margin for worker scheduling;
        //  * drift: if over ~1 s the low point stays > 10 ms above the floor,
        //    drop the excess in one step;
        //  * hard cap: latency can never exceed prime + 250 ms.
        let out_q = OutputQueueParams {
            chunk: frame_out,
            window: out_rate as usize,
            floor: frame_out / 2,
            tolerance: frame_out,
            hard_max: out_rate as usize / 4,
        };
        // Output priming threshold: one device period + one RNNoise chunk +
        // the jitter margin.
        let prime = OutputPacer::prime_for(out_period, &out_q);

        let params = WorkerParams {
            in_rate,
            out_rate,
            bypass: cfg.bypass,
            // Stall recovery: more than 250 ms (plus one input chunk) waiting
            // means this thread was not scheduled; keep one frame and go on.
            in_backlog_limit: in_rate as usize / 4 + in_period,
            in_backlog_target: frame_in,
        };

        let worker_ctl = ctl.clone();
        let worker_metrics = metrics.clone();
        let worker: JoinHandle<Result<()>> = thread::Builder::new()
            .name("crisp-dsp".into())
            .spawn(move || worker_loop(in_cons, out_prod, params, worker_ctl, worker_metrics))
            .context("spawning DSP worker")?;
        let worker_thread = worker.thread().clone();

        let in_result = build_input_any(
            &in_dev,
            &mut in_spec,
            in_prod,
            metrics.clone(),
            worker_thread.clone(),
        );
        let in_stream = match in_result {
            Ok(s) => s,
            Err(e) if e.is::<RetryWithDefault>() => {
                stop_worker(&ctl, worker, &worker_thread);
                continue;
            }
            Err(e) => {
                stop_worker(&ctl, worker, &worker_thread);
                return Err(e);
            }
        };

        let out_result = build_output_any(
            &out_dev,
            &mut out_spec,
            out_cons,
            metrics.clone(),
            ctl.clone(),
            prime,
            out_q,
        );
        let out_stream = match out_result {
            Ok(s) => s,
            Err(e) if e.is::<RetryWithDefault>() => {
                drop(in_stream);
                stop_worker(&ctl, worker, &worker_thread);
                continue;
            }
            Err(e) => {
                drop(in_stream);
                stop_worker(&ctl, worker, &worker_thread);
                return Err(e);
            }
        };

        break (in_stream, out_stream, worker, prime);
    };

    let latency = latency_budget(&in_spec, &out_spec, prime_frames);
    let info = EngineInfo {
        host: format!("{:?}", host.id()),
        input_name: in_name,
        output_name: out_name,
        input_rate: in_spec.rate(),
        output_rate: out_spec.rate(),
        input_channels: in_spec.channels(),
        output_channels: out_spec.channels(),
        input_format: format!("{:?}", in_spec.format),
        output_format: format!("{:?}", out_spec.format),
        input_period: in_spec.buffer_frames,
        output_period: out_spec.buffer_frames,
        bypass: cfg.bypass,
        latency,
    };

    // Start output first so it is already waiting (silently, un-primed) when
    // the first input period arrives.
    out_stream.play().context("starting output stream")?;
    in_stream.play().context("starting input stream")?;

    Ok(Started {
        in_stream,
        out_stream,
        worker,
        ctl,
        info,
    })
}

fn stop_worker(ctl: &Arc<Control>, worker: JoinHandle<Result<()>>, thread: &Thread) {
    ctl.stop.store(true, Ordering::Release);
    thread.unpark();
    let _ = worker.join();
    ctl.stop.store(false, Ordering::Release);
    ctl.drained.store(false, Ordering::Release);
}

fn latency_budget(in_spec: &StreamSpec, out_spec: &StreamSpec, prime: usize) -> LatencyBudget {
    let in_rate = in_spec.rate();
    let out_rate = out_spec.rate();
    let input_period_ms = in_spec
        .buffer_frames
        .map(|n| ms(n as usize, in_rate))
        .unwrap_or(10.0);
    let out_period_frames = out_spec
        .buffer_frames
        .map(|n| n as usize)
        .unwrap_or_else(|| scale_frames(480, RNNOISE_RATE, out_rate));
    let output_period_ms = ms(out_period_frames, out_rate);
    let rnnoise_frame_ms = ms(FRAME_SIZE, RNNOISE_RATE);
    let resample_in_ms = StreamResampler::new(in_rate, RNNOISE_RATE)
        .map(|r| r.latency_ms())
        .unwrap_or(0.0);
    let resample_out_ms = StreamResampler::new(RNNOISE_RATE, out_rate)
        .map(|r| r.latency_ms())
        .unwrap_or(0.0);

    // Latency model, per sample:
    //  * input side: up to one input period until the callback delivers it,
    //    plus RNNoise framing. A sample at position x of a 480-sample frame
    //    waits (480 - x) for the frame to complete and then x more while the
    //    chunk drains on the output — together exactly one frame (10 ms) for
    //    every sample, so framing is counted once, here.
    //  * output queue: what is still queued in front of the chunk when it
    //    arrives — the low point the pacer steers to (floor = half a frame),
    //    which can sit up to one device period above it depending on phase.
    //  * output period: the device's own buffer.
    let input_side_ms = input_period_ms + rnnoise_frame_ms;
    let floor = scale_frames(FRAME_SIZE, RNNOISE_RATE, out_rate) / 2;
    let output_queue_ms = ms(floor + out_period_frames, out_rate);
    let total_ms =
        input_side_ms + resample_in_ms + resample_out_ms + output_queue_ms + output_period_ms;

    LatencyBudget {
        input_period_ms,
        rnnoise_frame_ms,
        input_side_ms,
        resample_in_ms,
        resample_out_ms,
        output_queue_ms,
        output_queue_frames: prime,
        output_period_ms,
        total_ms,
    }
}

fn log_info(info: &EngineInfo) {
    let period = |p: Option<u32>, rate: u32| match p {
        Some(n) => format!("{n} frames ({:.2} ms)", ms(n as usize, rate)),
        None => "backend default".into(),
    };
    info!(
        "input : {} @ {} Hz, {} ch, {}, period {}",
        info.input_name,
        info.input_rate,
        info.input_channels,
        info.input_format,
        period(info.input_period, info.input_rate)
    );
    info!(
        "output: {} @ {} Hz, {} ch, {}, period {}",
        info.output_name,
        info.output_rate,
        info.output_channels,
        info.output_format,
        period(info.output_period, info.output_rate)
    );
    let l = &info.latency;
    info!("latency budget (mic -> output, typical worst case, excluding driver/hardware):");
    info!(
        "  input period + RNNoise framing : {:>6.2} ms  ({:.2} + {:.2} ms)",
        l.input_side_ms, l.input_period_ms, l.rnnoise_frame_ms
    );
    info!("  resampler in  ({} -> 48000)    : {:>6.2} ms", info.input_rate, l.resample_in_ms);
    info!("  resampler out (48000 -> {})    : {:>6.2} ms", info.output_rate, l.resample_out_ms);
    info!(
        "  output queue (steady state)    : {:>6.2} ms  (primes at {} frames)",
        l.output_queue_ms, l.output_queue_frames
    );
    info!("  output period                  : {:>6.2} ms", l.output_period_ms);
    info!(
        "  ≈ total                        : {:>6.2} ms{}",
        l.total_ms,
        if info.bypass { "  [RNNoise bypassed]" } else { "" }
    );
    if l.total_ms > 25.0 {
        warn!("estimated latency exceeds 25 ms; try a smaller --buffer-frames or a 48 kHz device");
    }
}

fn snapshot(m: &Metrics, in_rate: u32, out_rate: u32) -> MetricsSnapshot {
    MetricsSnapshot {
        frames_processed: m.frames_processed.load(Ordering::Relaxed),
        vad: m.vad(),
        in_peak: m.in_peak(),
        out_peak: m.out_peak(),
        in_fill_ms: ms(m.in_fill.load(Ordering::Relaxed), in_rate),
        out_fill_ms: ms(m.out_fill.load(Ordering::Relaxed), out_rate),
        input_overruns: m.input_overruns.load(Ordering::Relaxed),
        output_underruns: m.output_underruns.load(Ordering::Relaxed),
        output_overruns: m.output_overruns.load(Ordering::Relaxed),
        dropped_samples: m.dropped_samples.load(Ordering::Relaxed),
        stream_errors: m.stream_errors.load(Ordering::Relaxed),
        in_cb_max: m.in_cb_max.load(Ordering::Relaxed),
        out_cb_max: m.out_cb_max.load(Ordering::Relaxed),
    }
}

fn log_stats(s: &MetricsSnapshot) {
    info!(
        "stats: frames={} vad={:.2} in_q={:.1}ms out_q={:.1}ms cb_max={}/{} | in_overrun={} out_underrun={} out_overrun={} dropped={} errors={}",
        s.frames_processed,
        s.vad,
        s.in_fill_ms,
        s.out_fill_ms,
        s.in_cb_max,
        s.out_cb_max,
        s.input_overruns,
        s.output_underruns,
        s.output_overruns,
        s.dropped_samples,
        s.stream_errors,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_frames_rounds_up() {
        assert_eq!(scale_frames(480, 48_000, 44_100), 441);
        assert_eq!(scale_frames(480, 48_000, 96_000), 960);
        assert_eq!(scale_frames(480, 48_000, 48_000), 480);
        assert_eq!(scale_frames(1, 48_000, 44_100), 1);
    }

    #[test]
    fn backlog_watch_ignores_bursty_consumer() {
        // Consumer pulls 4 x 512 frames every ~43 ms: queue swings 0..2048
        // but its minimum stays ~0 -> never trim.
        let mut w = BacklogWatch::new(48_000, 240, 480);
        let mut dropped = 0;
        for i in 0..2000 {
            let level = if i % 4 == 3 { 0 } else { 2048 - (i % 4) * 512 };
            dropped += w.observe(level, 512);
        }
        assert_eq!(dropped, 0);
    }

    #[test]
    fn backlog_watch_trims_persistent_excess_to_floor() {
        let mut w = BacklogWatch::new(48_000, 240, 480);
        let mut total = 0;
        // Queue never drops below 3000 for a full window.
        for _ in 0..(48_000 / 128) {
            total += w.observe(3000 + 37, 128);
        }
        assert_eq!(total, 3037 - 240);
    }

    #[test]
    fn backlog_watch_tolerates_small_excess() {
        let mut w = BacklogWatch::new(48_000, 240, 480);
        let mut total = 0;
        for _ in 0..1000 {
            total += w.observe(600, 128);
        }
        assert_eq!(total, 0);
    }

    /// Simulate `seconds` of a producer emitting 480-frame RNNoise bursts at
    /// `48 kHz * (1 + drift)` and a device pulling `burst` x `period` frames
    /// at a time at exactly 48 kHz. Returns (max queue, underruns, dropped,
    /// final queue), all in frames.
    fn simulate(drift: f64, period: usize, burst: usize, seconds: f64) -> (usize, usize, usize, usize) {
        let rate = 48_000.0;
        let frame = 480usize;
        let q = OutputQueueParams {
            chunk: frame,
            window: 48_000,
            floor: frame / 2,
            tolerance: frame,
            hard_max: 12_000,
        };
        let mut pacer = OutputPacer::new(period, q);
        let (mut queue, mut max_q, mut underruns, mut dropped) = (0usize, 0usize, 0usize, 0usize);
        let prod_dt = frame as f64 / (rate * (1.0 + drift));
        let cons_dt = (period * burst) as f64 / rate;
        let (mut t_prod, mut t_cons) = (0.0f64, cons_dt);
        while t_cons < seconds {
            // Producer events up to the next consumer wake-up.
            while t_prod <= t_cons {
                queue += frame;
                t_prod += prod_dt;
            }
            max_q = max_q.max(queue);
            // Sound server wakes and pulls `burst` periods back to back.
            for _ in 0..burst {
                match pacer.plan(queue, period, false) {
                    OutputPlan::Silence => {}
                    OutputPlan::Play { skip } => {
                        queue -= skip;
                        dropped += skip;
                        if queue < period {
                            queue = 0;
                            underruns += 1;
                            pacer.on_underrun();
                        } else {
                            queue -= period;
                        }
                    }
                }
            }
            t_cons += cons_dt;
        }
        eprintln!(
            "simulate drift={drift:+} period={period}x{burst} {seconds}s: max_q={max_q} underruns={underruns} dropped={dropped} final_q={queue}"
        );
        (max_q, underruns, dropped, queue)
    }

    #[test]
    fn pacer_bursty_device_no_drift_is_glitch_free() {
        // PipeWire-like: 4 x 256 frames every ~21 ms, perfect clocks, 60 s.
        let (max_q, underruns, dropped, _) = simulate(0.0, 256, 4, 60.0);
        assert_eq!(underruns, 0, "underruns");
        assert_eq!(dropped, 0, "nothing should be dropped without drift");
        assert!(max_q <= 256 * 4 + 480 + 240 + 2 * 480, "max queue {max_q}");
    }

    #[test]
    fn pacer_fast_input_clock_keeps_latency_bounded() {
        // Mic clock 0.1 % fast vs. output (≈ 48 extra frames/s), 120 s.
        let (max_q, underruns, dropped, final_q) = simulate(0.001, 128, 1, 120.0);
        assert_eq!(underruns, 0);
        assert!(dropped > 0, "excess must be trimmed");
        // Without correction the queue would end ~5760 frames (120 ms) long.
        assert!(final_q <= 128 + 480 + 240 + 2 * 480, "final queue {final_q}");
        assert!(max_q <= 128 + 480 + 240 + 3 * 480, "max queue {max_q}");
    }

    #[test]
    fn pacer_slow_input_clock_rarely_reprimes() {
        // Mic clock 0.1 % slow (48 frames/s deficit): the queue slowly
        // empties. The ~240-frame margin lasts ~5 s, so a short re-prime gap
        // every few seconds is unavoidable — but never continuous crackle.
        let (_, underruns, dropped, _) = simulate(-0.001, 128, 1, 60.0);
        assert_eq!(dropped, 0);
        assert!(underruns <= 15, "underruns {underruns}");
    }

    #[test]
    fn pacer_hard_cap_bounds_wild_mismatch() {
        // Producer 30 % faster than the device (broken clocks): latency must
        // stay under prime + hard cap instead of filling the ring.
        let (max_q, _, dropped, final_q) = simulate(0.30, 128, 1, 20.0);
        assert!(dropped > 0);
        let prime = 128 + 480 + 240;
        assert!(final_q <= prime + 12_000 + 480, "final queue {final_q}");
        assert!(max_q <= prime + 12_000 + 2 * 480, "max queue {max_q}");
    }

    #[test]
    fn pacer_large_requests_raise_prime() {
        let q = OutputQueueParams { chunk: 480, window: 48_000, floor: 240, tolerance: 480, hard_max: 12_000 };
        let mut p = OutputPacer::new(128, q);
        // Device asks for 4800 frames at once; 1000 queued must not start playback.
        assert_eq!(p.plan(1000, 4800, false), OutputPlan::Silence);
        assert_eq!(p.plan(4800 + 480 + 240, 4800, false), OutputPlan::Play { skip: 0 });
    }

    #[test]
    fn ring_capacity_is_pow2_and_at_least_one_second() {
        assert_eq!(ring_capacity(48_000), 65_536);
        assert_eq!(ring_capacity(44_100), 65_536);
        assert!(ring_capacity(96_000) >= 96_000);
    }
}
