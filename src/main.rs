//! crisp-rs — real-time microphone denoiser (RNNoise) as a CLI / daemon.
//!
//! ```text
//! crisp-rs list
//! crisp-rs run --input "USB Mic" --output "CABLE Input"
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

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

    /// Bypass RNNoise (A/B comparison). Latency path is identical.
    #[arg(long)]
    bypass: bool,

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
    }
}

impl RunArgs {
    fn default_args() -> Self {
        Self {
            input: None,
            output: None,
            virtual_mic: false,
            buffer_frames: 128,
            rate: None,
            bypass: false,
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
        bypass: args.bypass,
        stats_secs: args.stats_secs,
    };

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
