//! crisp-rs — real-time microphone denoiser (RNNoise) as a CLI / daemon.
//!
//! ```text
//! crisp-rs list
//! crisp-rs run --input "USB Mic" --output "CABLE Input"
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use crisp_rs::resampler::{StreamResampler, RNNOISE_RATE};
use crisp_rs::fx::{Extra, FxPreset, FxSettings, Scale};
use crisp_rs::voice::{GateMode, Preset, VoiceSettings};

use crisp_rs::virtual_mic::VirtualMic;
use crisp_rs::{audio_engine, EngineConfig};

#[derive(Parser, Debug)]
#[command(
    name = "crisp-rs",
    version,
    about = "Low-latency real-time microphone noise suppression (RNNoise)",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Verbose logging (same as RUST_LOG=debug).
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List input/output devices of the default audio host.
    List,
    /// Capture, denoise and play back (default command).
    Run(RunArgs),
    /// Denoise a WAV file offline with exactly the same processing chain.
    /// Output: 48 kHz mono 16-bit WAV.
    Process {
        /// Input WAV (any rate / channel count / sample format).
        input: PathBuf,
        /// Output WAV path.
        output: PathBuf,
        #[command(flatten)]
        voice: VoiceArgs,
        #[command(flatten)]
        fx: FxArgs,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum VoicePresetArg {
    Off,
    Deep,
    High,
    Child,
    Monster,
    Robot,
    Autotune,
    Radio,
    Echo,
    Cave,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ScaleArg {
    Chromatic,
    Major,
    Minor,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EffectArg {
    None,
    Radio,
    Echo,
    Reverb,
}

/// Voice-effect options shared by `run` and `process`.
#[derive(Args, Debug, Clone)]
struct FxArgs {
    /// Voice-changer preset; the options below override parts of it.
    #[arg(long = "voice", value_enum, default_value_t = VoicePresetArg::Off)]
    voice_preset: VoicePresetArg,

    /// Pitch shift in semitones (-12..12). Adds ~38 ms latency.
    #[arg(long, allow_hyphen_values = true)]
    pitch: Option<f32>,

    /// Formant (timbre) shift in semitones (-5..5). 0 keeps the natural timbre.
    #[arg(long, allow_hyphen_values = true)]
    formant: Option<f32>,

    /// Auto-tune to this key (C, C#, D, ..., B). Adds ~38 ms latency.
    #[arg(long, value_name = "KEY")]
    autotune: Option<String>,

    /// Scale for --autotune.
    #[arg(long, value_enum, default_value_t = ScaleArg::Chromatic)]
    scale: ScaleArg,

    /// Auto-tune retune time in ms (0 = hard/robotic, ~150 = natural).
    #[arg(long, default_value_t = 0.0)]
    retune_ms: f32,

    /// Robot voice (monotone).
    #[arg(long)]
    robot: bool,

    /// Extra effect after the voice changer.
    #[arg(long, value_enum)]
    effect: Option<EffectArg>,
}

impl FxArgs {
    fn settings(&self) -> Result<FxSettings> {
        let mut s = FxSettings::preset(match self.voice_preset {
            VoicePresetArg::Off => FxPreset::Off,
            VoicePresetArg::Deep => FxPreset::Deep,
            VoicePresetArg::High => FxPreset::High,
            VoicePresetArg::Child => FxPreset::Child,
            VoicePresetArg::Monster => FxPreset::Monster,
            VoicePresetArg::Robot => FxPreset::Robot,
            VoicePresetArg::Autotune => FxPreset::AutoTune,
            VoicePresetArg::Radio => FxPreset::Radio,
            VoicePresetArg::Echo => FxPreset::Echo,
            VoicePresetArg::Cave => FxPreset::Cave,
        });
        let mut touched = false;
        if let Some(p) = self.pitch {
            anyhow::ensure!((-12.0..=12.0).contains(&p), "--pitch must be within -12..12");
            s.pitch = p;
            touched = true;
        }
        if let Some(f) = self.formant {
            anyhow::ensure!((-5.0..=5.0).contains(&f), "--formant must be within -5..5");
            s.formant = f;
            touched = true;
        }
        if let Some(k) = &self.autotune {
            s.autotune = true;
            s.key = parse_key(k)?;
            touched = true;
        }
        s.scale = match self.scale {
            ScaleArg::Chromatic => Scale::Chromatic,
            ScaleArg::Major => Scale::Major,
            ScaleArg::Minor => Scale::Minor,
        };
        if self.retune_ms > 0.0 {
            s.retune_ms = self.retune_ms;
        }
        if self.robot {
            s.robot = true;
            touched = true;
        }
        if let Some(e) = self.effect {
            s.extra = match e {
                EffectArg::None => Extra::None,
                EffectArg::Radio => Extra::Radio,
                EffectArg::Echo => Extra::Echo,
                EffectArg::Reverb => Extra::Reverb,
            };
            touched = true;
        }
        if touched {
            s.enabled = true;
        }
        Ok(s)
    }

    fn defaults() -> Self {
        Self {
            voice_preset: VoicePresetArg::Off,
            pitch: None,
            formant: None,
            autotune: None,
            scale: ScaleArg::Chromatic,
            retune_ms: 0.0,
            robot: false,
            effect: None,
        }
    }
}

/// "C", "c#", "Db", "A#" … → 0..=11.
fn parse_key(k: &str) -> Result<u8> {
    let k = k.trim();
    let mut chars = k.chars();
    let base = match chars.next().map(|c| c.to_ascii_uppercase()) {
        Some('C') => 0,
        Some('D') => 2,
        Some('E') => 4,
        Some('F') => 5,
        Some('G') => 7,
        Some('A') => 9,
        Some('B') => 11,
        _ => anyhow::bail!("unknown key {k:?} (use C, C#, Db, ..., B)"),
    };
    let acc = match chars.as_str() {
        "" => 0,
        "#" | "s" => 1,
        "b" => -1,
        other => anyhow::bail!("unknown accidental {other:?} in key {k:?}"),
    };
    Ok(((base + acc + 12) % 12) as u8)
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PresetArg {
    /// Most natural voice (70 % voice relief) + light gate (quiet rooms).
    Light,
    /// Natural voice (50 % voice relief) + light gate (recommended).
    Balanced,
    /// Maximum cleaning, no relief, strong gate (noisy rooms).
    Strong,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum GateArg {
    Off,
    Light,
    Strong,
}

/// Voice-processing options shared by `run` and `process`.
#[derive(Args, Debug, Clone)]
struct VoiceArgs {
    /// Processing preset; the options below override parts of it.
    #[arg(long, value_enum, default_value_t = PresetArg::Balanced)]
    preset: PresetArg,

    /// Noise-reduction strength in percent (0 = raw mic, 100 = full RNNoise).
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=100))]
    strength: Option<u8>,

    /// Keep the voice natural: reduce the strength by this percentage while
    /// voice is detected (0-80). Pauses stay fully cleaned. Presets: light 70,
    /// balanced 50, strong 0.
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=80))]
    natural: Option<u8>,

    /// Voice gate between phrases: off, light (-15 dB) or strong (-45 dB).
    #[arg(long, value_enum)]
    gate: Option<GateArg>,

    /// Enable slow automatic gain control (speech level normalisation). Off by
    /// default: Discord / Zoom / Meet / Teams already have one.
    #[arg(long)]
    agc: bool,

    /// Disable the 80 Hz high-pass (rumble) filter.
    #[arg(long)]
    no_highpass: bool,

    /// Pass the raw microphone through (A/B comparison, same latency).
    #[arg(long)]
    bypass: bool,
}

impl VoiceArgs {
    fn settings(&self) -> VoiceSettings {
        let mut s = VoiceSettings::preset(match self.preset {
            PresetArg::Light => Preset::Light,
            PresetArg::Balanced => Preset::Balanced,
            PresetArg::Strong => Preset::Strong,
        });
        if let Some(p) = self.strength {
            s.strength = f32::from(p) / 100.0;
        }
        if let Some(p) = self.natural {
            s.voice_relief = f32::from(p) / 100.0;
        }
        if let Some(g) = self.gate {
            s.gate = match g {
                GateArg::Off => GateMode::Off,
                GateArg::Light => GateMode::Light,
                GateArg::Strong => GateMode::Strong,
            };
        }
        s.agc |= self.agc;
        s.highpass &= !self.no_highpass;
        s.bypass = self.bypass;
        s
    }

    fn defaults() -> Self {
        Self {
            preset: PresetArg::Balanced,
            strength: None,
            natural: None,
            gate: None,
            agc: false,
            no_highpass: false,
            bypass: false,
        }
    }
}

#[derive(Args, Debug, Clone)]
struct RunArgs {
    /// Input device: index from `list`, name substring, or omit for default.
    #[arg(short, long)]
    input: Option<String>,

    /// Output device (speakers or a virtual cable): index, name substring, or default.
    #[arg(short, long, conflicts_with = "virtual_mic")]
    output: Option<String>,

    /// Send the denoised audio to a virtual microphone that chat apps
    /// (Discord, Zoom, Meet, OBS...) can select. Linux: created automatically
    /// via PulseAudio/PipeWire. Windows/macOS: uses VB-Cable / BlackHole.
    #[arg(long)]
    virtual_mic: bool,

    /// Hardware period size in frames for both streams (0 = backend default).
    /// 128 frames @ 48 kHz = 2.67 ms per direction.
    #[arg(long, default_value_t = 128)]
    buffer_frames: u32,

    /// Force the device sample rate (Hz). Anything other than 48000 engages the
    /// rubato resampler. Default: 48000 if the device supports it, else its native rate.
    #[arg(long)]
    rate: Option<u32>,

    #[command(flatten)]
    voice: VoiceArgs,

    #[command(flatten)]
    fx: FxArgs,

    /// Print pipeline statistics every N seconds (0 = off).
    #[arg(long, default_value_t = 5)]
    stats_secs: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let default_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_level))
        .format_timestamp_millis()
        .init();

    match cli.command.unwrap_or(Command::Run(RunArgs::default_args())) {
        Command::List => audio_engine::list_devices(),
        Command::Run(args) => run(args),
        Command::Process {
            input,
            output,
            voice,
            fx,
        } => process_file(&input, &output, &voice.settings(), &fx.settings()?),
    }
}

/// Offline: WAV in → mono → 48 kHz → voice chain → 48 kHz mono 16-bit WAV.
fn process_file(input: &Path, output: &Path, s: &VoiceSettings, fx: &FxSettings) -> Result<()> {
    let mut reader = hound::WavReader::open(input)
        .with_context(|| format!("opening {}", input.display()))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|v| v.map(|x| x as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = interleaved
        .chunks_exact(channels)
        .map(|f| f.iter().sum::<f32>() / channels as f32)
        .collect();

    let mut at48 = Vec::with_capacity(mono.len() * 48_000 / spec.sample_rate as usize + 1024);
    let mut rs = StreamResampler::new(spec.sample_rate, RNNOISE_RATE)?;
    for chunk in mono.chunks(4096) {
        rs.process(chunk, &mut |x| at48.extend_from_slice(x))?;
    }
    rs.flush(&mut |x| at48.extend_from_slice(x))?;

    let started = std::time::Instant::now();
    let out = crisp_rs::voice::process_buffer_fx(&at48, s, fx);
    let elapsed = started.elapsed().as_secs_f64();

    let mut writer = hound::WavWriter::create(
        output,
        hound::WavSpec {
            channels: 1,
            sample_rate: RNNOISE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .with_context(|| format!("creating {}", output.display()))?;
    for v in &out {
        writer.write_sample((v.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    writer.finalize()?;

    let audio_secs = out.len() as f64 / f64::from(RNNOISE_RATE);
    log::info!(
        "{} → {}: {:.1} s of audio processed in {:.2} s ({:.0}× real time)",
        input.display(),
        output.display(),
        audio_secs,
        elapsed,
        audio_secs / elapsed.max(1e-9)
    );
    Ok(())
}

impl RunArgs {
    fn default_args() -> Self {
        Self {
            input: None,
            output: None,
            virtual_mic: false,
            buffer_frames: 128,
            rate: None,
            voice: VoiceArgs::defaults(),
            fx: FxArgs::defaults(),
            stats_secs: 5,
        }
    }
}

fn run(args: RunArgs) -> Result<()> {
    if let Some(r) = args.rate {
        anyhow::ensure!(
            (8_000..=384_000).contains(&r),
            "--rate {r} is outside the sane range 8000..=384000 Hz"
        );
    }

    // Graceful shutdown: Ctrl+C / SIGTERM flips a flag; the engine notices,
    // stops capture, flushes the DSP stages and drains the output queue.
    let running = Arc::new(AtomicBool::new(true));
    {
        let running = running.clone();
        ctrlc::set_handler(move || {
            if !running.swap(false, Ordering::SeqCst) {
                // Second signal: the user is impatient; exit immediately.
                eprintln!("forced exit");
                std::process::exit(130);
            }
        })
        .context("installing Ctrl+C handler")?;
    }

    let mut cfg = EngineConfig {
        input: args.input,
        output: args.output,
        buffer_frames: args.buffer_frames,
        force_rate: args.rate,
        voice: args.voice.settings(),
        fx: args.fx.settings()?,
        stats_secs: args.stats_secs,
    };

    if cfg.fx.enabled {
        log::info!("voice effects: {:?}", cfg.fx);
    }
    if !args.virtual_mic {
        return audio_engine::run(cfg, running);
    }

    // Virtual-mic mode: the VirtualMic must outlive the engine (dropping it
    // removes the Linux virtual device), so it is created first.
    let catalog = crisp_rs::device_catalog();
    let vmic = VirtualMic::prepare(&catalog.outputs)?;
    let input_name = match cfg.input.as_deref() {
        Some(sel) => sel.to_string(),
        None => "default".to_string(),
    };
    vmic.check_input(&input_name)?;
    cfg.output = Some(vmic.output_selector().to_string());
    log::info!("virtual mic via {}", vmic.backend());

    audio_engine::run_with(cfg, running, |engine| {
        vmic.check_input(&engine.info().input_name)?;
        vmic.route_engine_output()?;
        log::info!(
            "==> In Discord / Zoom / Meet select the microphone: \"{}\"",
            vmic.mic_name()
        );
        Ok(())
    })
}
