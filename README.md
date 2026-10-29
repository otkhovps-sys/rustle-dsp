# crisp-rs

Real-time microphone noise suppression as a small CLI / daemon **and a desktop app**,
written in Rust.

![crisp-gui](docs/screenshot-gui.png)

* Captures a microphone with [`cpal`](https://crates.io/crates/cpal)
* Denoises with RNNoise via [`nnnoiseless`](https://crates.io/crates/nnnoiseless)
* **Virtual microphone mode**: the clean signal appears as a new input device
  ("Crisp Mic (denoised)" on Linux, VB-Cable / BlackHole on Windows / macOS) that
  Discord, Zoom, Meet, Teams or OBS select as their microphone.
* Monitor mode plays the clean signal on speakers / headphones for testing.
* ~23 ms mic-to-output latency at 48 kHz with lock-free SPSC ring buffers between the
  audio callbacks and a dedicated DSP thread, no allocation and no locks in the
  real-time callbacks, and drift correction that keeps latency bounded without
  crackling on bursty sound servers.

## Pipeline

```
CPAL input cb            DSP worker thread                     CPAL output cb
(device rate, N ch)      (allocates nothing after start-up)    (device rate, M ch)
┌────────────────┐ SPSC  ┌───────────────────────────────┐ SPSC ┌──────────────────┐
│ downmix → mono │ ────► │ rubato  → 48 kHz              │ ───► │ mono → M ch      │
│ try_push       │ in_rb │ RNNoise 480-sample frames     │ out  │ try_pop          │
│ unpark(worker) │       │ rubato  → output device rate  │ _rb  │ silence on empty │
└────────────────┘       └───────────────────────────────┘      └──────────────────┘
```

* `src/resampler.rs` – streaming wrapper over `rubato::SincFixedIn` (pass-through
  when the device already runs at 48 kHz).
* `src/denoiser.rs` – wrapper over `nnnoiseless::DenoiseState`, gathers 480-sample
  frames and converts between ±1.0 floats and the i16 range RNNoise expects.
* `src/audio_engine.rs` – device selection, stream configuration, ring buffers,
  worker thread, latency clamp, graceful shutdown.
* `src/main.rs` – `clap` CLI (`crisp-rs`).
* `src/bin/gui.rs` – desktop GUI (`crisp-gui`, egui/eframe): device pickers, period /
  sample-rate / bypass settings, Start/Stop, live mic & output meters, VAD, latency
  budget and xrun counters. The UI thread only reads atomics; audio runs exactly the
  same engine as the CLI.

## Downloads

Prebuilt binaries for **Windows x86_64** and **Linux x86_64** are on the
[Releases page](https://github.com/otkhovps-sys/rustle-dsp/releases). Each archive
contains both `crisp-gui` (desktop app) and `crisp-rs` (CLI).

GitHub Actions (`.github/workflows/build.yml`) builds them on every push (see the run's
**Artifacts**) and publishes a Release when a `v*` tag is pushed, or when
**Run workflow** is started with a `release_tag` (e.g. `v0.1.1`). A macOS arm64 build
is available on demand via **Actions → build → Run workflow → macos**.

## Build

```bash
cargo build --release                       # CLI + GUI
cargo build --release --no-default-features # CLI only (no egui)
```

Linux needs the ALSA headers (`libasound2-dev` on Debian/Ubuntu, `alsa-lib-devel`
on Fedora); the GUI additionally needs an OpenGL-capable X11/Wayland session at run
time. macOS and Windows need nothing extra. Cross-compiling for Windows from Linux
works with `rustup target add x86_64-pc-windows-gnu` + `gcc-mingw-w64-x86-64`.

## Virtual microphone (use it in Discord / Zoom / Meet / Teams / OBS)

```
real mic ──► crisp-rs (RNNoise) ──► virtual mic ──► chat app selects it as "Microphone"
```

| OS | What crisp-rs does | One-time setup |
|---|---|---|
| Linux (PulseAudio or PipeWire) | Creates **"Crisp Mic (denoised)"** itself (`module-null-sink` + `module-remap-source` via `pactl`), routes its output there, removes it on exit. | none (`pactl` must exist: `pulseaudio-utils` or `pipewire-pulse`) |
| Windows | Sends audio to **CABLE Input**; chat apps select **CABLE Output (VB-Audio Virtual Cable)**. | install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (free) |
| macOS | Sends audio to **BlackHole 2ch**; chat apps select **BlackHole 2ch**. | install [BlackHole](https://existential.audio/blackhole/) (free) |

Windows and macOS only let signed drivers create audio devices, which an application
cannot ship, hence the one-time cable install. crisp-rs detects the cable automatically.

Then in the chat app: **Settings → Audio → Microphone / Input → the name above**, and
turn off the app's own noise suppression (Krisp etc.) so the voice is not processed twice.

On Linux the ALSA plugin of the sound server re-creates its stream after every xrun;
crisp-rs listens to `pactl subscribe` and moves every new stream of its own process
onto the virtual mic within milliseconds, so routing survives glitches and restarts.

## Desktop app

```bash
crisp-gui
```

1. Mode **Micro ảo** (virtual mic, default) or **Nghe thử** (monitor on speakers).
2. Pick the real microphone. Virtual mics are hidden from the input list to prevent
   feedback loops.
3. **Bắt đầu lọc ồn** (Start). A green box shows the exact device name to select in
   the chat app, with a copy button. The meters show input / output level and the
   voice-activity probability; "Chi tiết pipeline" shows the latency budget and
   xrun counters.
4. Settings (mode, devices by name, period, rate, auto-start on launch) are saved.
   Closing the window stops gracefully and removes the Linux virtual mic.

The GUI embeds DejaVu Sans so Vietnamese labels render correctly on every platform.

## Usage

```bash
# list devices (index + name, default marked with *)
crisp-rs list

# default mic → virtual microphone for chat apps
crisp-rs run --virtual-mic

# default mic → default speakers (monitor / testing)
crisp-rs run

# pick devices by index or (case-insensitive) name substring
crisp-rs run --input "USB Mic" --output "CABLE Input"

# A/B the denoiser without changing latency
crisp-rs run --bypass

# force a device rate to exercise the resampler path
crisp-rs run --rate 44100

# smaller period for lower latency (if the backend accepts it)
crisp-rs run --buffer-frames 64
```

`Ctrl+C` (or `SIGTERM`) stops capture, flushes the resamplers and the last partial
RNNoise frame, lets the output queue drain, removes the Linux virtual mic and exits.
A second `Ctrl+C` exits immediately (a left-over virtual mic is adopted and cleaned up
by the next run).

## Latency budget

Printed at start-up, e.g. for a 48 kHz device with 128-frame periods:

```
input period + RNNoise framing :  12.67 ms  (2.67 + 10.00 ms)
resampler in  (48000 -> 48000)  :   0.00 ms
resampler out (48000 -> 48000)  :   0.00 ms
output queue (steady state)     :   7.67 ms  (primes at 848 frames)
output period                   :   2.67 ms
≈ total                         :  23.00 ms
```

| Devices | Period | Estimated latency |
|---|---|---|
| 48 kHz both ends | 128 | 23.0 ms |
| 44.1 kHz both ends (resampled) | 128 | 27.9 ms |
| 44.1 kHz both ends (resampled) | 64 | ≈ 23.5 ms |

Where the numbers come from:

* **RNNoise framing** – 480 samples = 10 ms, counted once for the whole chain: a sample
  at position *x* of a frame waits (480 − *x*) for the frame to complete and then *x*
  while that chunk drains on the output, i.e. exactly 10 ms for every sample.
* **Device periods** – `--buffer-frames` (default 128 = 2.67 ms @ 48 kHz) per direction.
* **Output queue** – the pacer steers the queue's low point to half a frame (5 ms of
  jitter margin for the DSP thread); it sits up to one period above that depending
  on phase. Playback starts once `period + 480 + 240` frames are queued, which
  guarantees that low point whatever the phase between worker and device.
* **Resampler** – only when a device cannot run at 48 kHz: sinc group delay
  (32 frames) + chunk gathering (64 frames) ≈ 2 ms per direction.

### Run-time safeguards

* **Drift correction** – sound servers pull audio in irregular bursts, so the queue
  swings by several periods without anything being wrong. crisp-rs therefore watches
  the queue's *minimum* over ~1 s windows; only if it stays more than 10 ms above the
  5 ms floor (two devices with different clocks) is the excess dropped, in one step.
  Simulated 120 s at +0.1 % clock drift: 0 underruns, final queue 10 ms.
* **Hard cap** – whatever the devices do, the output queue is trimmed immediately if
  it exceeds the priming level by 250 ms.
* **Stall recovery** – if the DSP thread was not scheduled for > 250 ms, the input
  backlog is trimmed instead of being played late.
* **Underrun handling** – the output callback emits silence and re-primes, so a stall
  produces one short gap instead of continuous crackling. Simulated 60 s at −0.1 %
  drift: 6 short re-primes.
* `--stats-secs N` logs frames processed, VAD probability, queue depths, the largest
  callback sizes and all overrun/underrun/drop counters.

## Tuning tips

* Prefer a 48 kHz device configuration for both ends: it removes both resamplers.
* On Linux, PipeWire users can lower the quantum (`PIPEWIRE_LATENCY=128/48000`) for
  the ALSA plugin, or run with the JACK backend.
* If you see `out_underrun` counts growing, raise `--buffer-frames` (256) or pin the
  process to a core / give it real-time priority (`chrt -f 10 crisp-rs run ...`).
