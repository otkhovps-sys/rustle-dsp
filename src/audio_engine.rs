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
}

impl Metrics {
    pub fn vad(&self) -> f32 {
        f32::from_bits(self.vad_bits.load(Ordering::Relaxed))
    }
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

    let data_cb = move |data: &[T], _: &cpal::InputCallbackInfo| {
        // RT-safe: only arithmetic, a wait-free push per frame, relaxed atomics.
        let mut overrun = false;
        for frame in data.chunks_exact(channels) {
            // Downmix: arithmetic mean of all channels. Equal-weight averaging
            // keeps a mono source recorded on both channels at unity gain.
            let mut acc = 0.0f32;
            for s in frame {
                acc += f32::from_sample(*s);
            }
            if prod.try_push(acc * inv_channels).is_err() {
                overrun = true;
            }
        }
        if overrun {
            metrics.input_overruns.fetch_add(1, Ordering::Relaxed);
        }
        // Wake the DSP worker. Futex wake = non-blocking for us.
        worker.unpark();
    };

    let err_cb = |e: cpal::StreamError| error!("input stream error: {e}");
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
) -> Result<Stream>
where
    T: SizedSample + FromSample<f32> + Send + 'static,
{
    let channels = usize::from(config.channels).max(1);
    let silence = T::from_sample(0.0f32);

    // Priming: stay silent until the worker has queued `prime_frames`. That
    // backlog is the jitter margin that absorbs the 10 ms burstiness of
    // RNNoise output versus the much smaller device period. After an underrun
    // we go back to priming so the margin is rebuilt in one go (one short gap)
    // instead of crackling continuously.
    let mut primed = false;

    let data_cb = move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
        let available = cons.occupied_len();
        metrics.out_fill.store(available, Ordering::Relaxed);

        let draining = ctl.draining.load(Ordering::Relaxed);
        if !primed && !draining {
            if available >= prime_frames {
                primed = true;
            } else {
                data.fill(silence);
                return;
            }
        }

        let mut underrun = false;
        for frame in data.chunks_exact_mut(channels) {
            let s = match cons.try_pop() {
                Some(s) => s,
                None => {
                    underrun = true;
                    0.0
                }
            };
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

        if underrun && !draining {
            metrics.output_underruns.fetch_add(1, Ordering::Relaxed);
            primed = false;
        }
    };

    let err_cb = |e: cpal::StreamError| error!("output stream error: {e}");
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
) -> Result<Stream> {
    macro_rules! build {
        ($cfg:expr, $cons:expr) => {
            match spec.format {
                SampleFormat::F32 => build_output_stream::<f32>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames),
                SampleFormat::I16 => build_output_stream::<i16>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames),
                SampleFormat::U16 => build_output_stream::<u16>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames),
                SampleFormat::I32 => build_output_stream::<i32>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames),
                SampleFormat::U32 => build_output_stream::<u32>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames),
                SampleFormat::F64 => build_output_stream::<f64>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames),
                SampleFormat::I8 => build_output_stream::<i8>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames),
                SampleFormat::U8 => build_output_stream::<u8>(device, $cfg, $cons, metrics.clone(), ctl.clone(), prime_frames),
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
    /// Frames (at `in_rate`) above which the input backlog is trimmed.
    in_backlog_limit: usize,
    /// Frames (at `in_rate`) the backlog is trimmed *to*.
    in_backlog_target: usize,
    /// Frames (at `out_rate`) above which output is considered over-queued.
    out_backlog_limit: usize,
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

    let frame_in = scale_frames(FRAME_SIZE, RNNOISE_RATE, p.in_rate);
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

        // --- Latency clamp -------------------------------------------------
        // Clock drift between two different devices, or a long scheduler
        // stall, lets the backlog grow without bound; that would silently
        // turn into ever-increasing latency. Trim the oldest input instead.
        // Costs one audible click, bounds the worst case.
        if !stopping {
            if in_backlog > p.in_backlog_limit {
                let drop = in_backlog - p.in_backlog_target;
                let skipped = in_cons.skip(drop);
                metrics.dropped_samples.fetch_add(skipped as u64, Ordering::Relaxed);
                debug!("latency clamp: dropped {skipped} input frames (backlog {in_backlog})");
            } else if out_prod.occupied_len() > p.out_backlog_limit && in_backlog >= frame_in {
                let skipped = in_cons.skip(frame_in);
                metrics.dropped_samples.fetch_add(skipped as u64, Ordering::Relaxed);
                debug!("latency clamp: output over-queued, dropped {skipped} input frames");
            }
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

fn fmt_buffer(spec: &StreamSpec) -> String {
    match spec.buffer_frames {
        Some(n) => format!("{n} frames ({:.2} ms)", ms(n as usize, spec.rate())),
        None => "backend default".into(),
    }
}

pub fn run(cfg: EngineConfig, running: Arc<AtomicBool>) -> Result<()> {
    let host = cpal::default_host();
    info!("audio host: {:?}", host.id());

    let in_dev = resolve_device(&host, Direction::Input, cfg.input.as_deref())?;
    let out_dev = resolve_device(&host, Direction::Output, cfg.output.as_deref())?;
    let in_name = in_dev.name().unwrap_or_else(|_| "<unknown>".into());
    let out_name = out_dev.name().unwrap_or_else(|_| "<unknown>".into());

    let mut in_spec = choose_config(&in_dev, Direction::Input, cfg.force_rate, cfg.buffer_frames)?;
    let mut out_spec =
        choose_config(&out_dev, Direction::Output, cfg.force_rate, cfg.buffer_frames)?;

    info!(
        "input : {in_name} @ {} Hz, {} ch, {:?}, period {}",
        in_spec.rate(),
        in_spec.channels(),
        in_spec.format,
        fmt_buffer(&in_spec)
    );
    info!(
        "output: {out_name} @ {} Hz, {} ch, {:?}, period {}",
        out_spec.rate(),
        out_spec.channels(),
        out_spec.format,
        fmt_buffer(&out_spec)
    );

    let metrics = Arc::new(Metrics::default());
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

        // Output priming threshold = one device period + half an RNNoise frame.
        // The worker emits 480-frame bursts every 10 ms while the output
        // callback drains `out_period` every period; the half-frame margin
        // covers worker scheduling jitter of up to 5 ms without underrunning.
        let prime = out_period + frame_out / 2;

        let params = WorkerParams {
            in_rate,
            out_rate,
            bypass: cfg.bypass,
            // Never let more than ~4 RNNoise frames (40 ms) queue up on input...
            in_backlog_limit: 4 * frame_in + in_period,
            // ...and when trimming, leave exactly one frame so processing can continue.
            in_backlog_target: frame_in,
            // Output queue should hover around `prime`; beyond prime + 2 frames
            // we are accumulating drift and start dropping input.
            out_backlog_limit: prime + 2 * frame_out,
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

    print_latency_budget(&in_spec, &out_spec, prime_frames, cfg.bypass);

    // Start output first so it is already waiting (silently, un-primed) when
    // the first input period arrives.
    out_stream.play().context("starting output stream")?;
    in_stream.play().context("starting input stream")?;
    info!("running — press Ctrl+C to stop");

    // ---- Main loop: just report stats until asked to stop -----------------
    let stats_every = if cfg.stats_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(cfg.stats_secs))
    };
    let mut last_stats = Instant::now();
    while running.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(50));
        if let Some(every) = stats_every {
            if last_stats.elapsed() >= every {
                last_stats = Instant::now();
                log_stats(&metrics, in_spec.rate(), out_spec.rate());
            }
        }
        if worker.is_finished() {
            break;
        }
    }

    // ---- Graceful shutdown --------------------------------------------------
    info!("shutting down");
    // 1. Stop capturing: no new samples enter the pipeline.
    drop(in_stream);
    // 2. Tell the worker to process + flush whatever is left and exit.
    ctl.stop.store(true, Ordering::Release);
    worker.thread().unpark();
    let worker_result = worker
        .join()
        .map_err(|_| anyhow!("DSP worker thread panicked"))?;
    // 3. Let the output play out the remaining queued audio (bypass priming).
    ctl.draining.store(true, Ordering::Release);
    let deadline = Instant::now() + Duration::from_millis(500);
    while metrics.out_fill.load(Ordering::Relaxed) > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    // 4. Stop playback.
    drop(out_stream);

    log_stats(&metrics, in_spec.rate(), out_spec.rate());
    worker_result?;
    info!("stopped cleanly");
    Ok(())
}

fn stop_worker(ctl: &Arc<Control>, worker: JoinHandle<Result<()>>, thread: &Thread) {
    ctl.stop.store(true, Ordering::Release);
    thread.unpark();
    let _ = worker.join();
    ctl.stop.store(false, Ordering::Release);
    ctl.drained.store(false, Ordering::Release);
}

fn print_latency_budget(in_spec: &StreamSpec, out_spec: &StreamSpec, prime: usize, bypass: bool) {
    let in_rate = in_spec.rate();
    let out_rate = out_spec.rate();
    let in_period = in_spec
        .buffer_frames
        .map(|n| ms(n as usize, in_rate))
        .unwrap_or(10.0);
    let out_period = out_spec
        .buffer_frames
        .map(|n| ms(n as usize, out_rate))
        .unwrap_or(10.0);
    let rnnoise_frame = ms(FRAME_SIZE, RNNOISE_RATE);
    let resample_in = StreamResampler::new(in_rate, RNNOISE_RATE)
        .map(|r| r.latency_ms())
        .unwrap_or(0.0);
    let resample_out = StreamResampler::new(RNNOISE_RATE, out_rate)
        .map(|r| r.latency_ms())
        .unwrap_or(0.0);
    let prime_ms = ms(prime, out_rate);

    // Input period and RNNoise framing overlap: a sample waits until its
    // 480-frame block is complete, which already includes waiting for the
    // input period that delivers it. So the input side costs max(period, frame).
    let input_side = in_period.max(rnnoise_frame);
    let total = input_side + resample_in + resample_out + prime_ms + out_period;

    info!("latency budget (mic -> output, worst case, excluding driver/hardware):");
    info!("  input period + RNNoise framing : {:>6.2} ms  (max of {:.2} ms, {:.2} ms)", input_side, in_period, rnnoise_frame);
    info!("  resampler in  ({} -> 48000)    : {:>6.2} ms", in_rate, resample_in);
    info!("  resampler out (48000 -> {})    : {:>6.2} ms", out_rate, resample_out);
    info!("  output queue (priming level)   : {:>6.2} ms  ({} frames)", prime_ms, prime);
    info!("  output period                  : {:>6.2} ms", out_period);
    info!("  ≈ total                        : {:>6.2} ms{}", total, if bypass { "  [RNNoise bypassed]" } else { "" });
    if total > 25.0 {
        warn!("estimated latency exceeds 25 ms; try a smaller --buffer-frames or a 48 kHz device");
    }
}

fn log_stats(m: &Metrics, in_rate: u32, out_rate: u32) {
    let in_fill = m.in_fill.load(Ordering::Relaxed);
    let out_fill = m.out_fill.load(Ordering::Relaxed);
    info!(
        "stats: frames={} vad={:.2} in_q={:.1}ms out_q={:.1}ms | in_overrun={} out_underrun={} out_overrun={} dropped={}",
        m.frames_processed.load(Ordering::Relaxed),
        m.vad(),
        ms(in_fill, in_rate),
        ms(out_fill, out_rate),
        m.input_overruns.load(Ordering::Relaxed),
        m.output_underruns.load(Ordering::Relaxed),
        m.output_overruns.load(Ordering::Relaxed),
        m.dropped_samples.load(Ordering::Relaxed),
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
    fn ring_capacity_is_pow2_and_at_least_one_second() {
        assert_eq!(ring_capacity(48_000), 65_536);
        assert_eq!(ring_capacity(44_100), 65_536);
        assert!(ring_capacity(96_000) >= 96_000);
    }
}
