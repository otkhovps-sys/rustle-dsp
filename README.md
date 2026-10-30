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
* Noise-reduction presets (Light / Balanced / Strong) with high-pass, speech-aware
  strength, voice gate, optional AGC and limiter around RNNoise, adjustable live, plus
  offline WAV processing.
* Voice effects: voice changer (pitch + formant, TD-PSOLA), auto-tune, robot, radio,
  echo, reverb.
* ~33 ms mic-to-output latency at 48 kHz (20 ms of it is RNNoise itself) with
  lock-free SPSC ring buffers between the audio callbacks and a dedicated DSP thread,
  no allocation and no locks in the real-time callbacks, and drift correction that
  keeps latency bounded without crackling on bursty sound servers.

## Pipeline

```
CPAL input cb            DSP worker thread                     CPAL output cb
(device rate, N ch)      (allocates nothing after start-up)    (device rate, M ch)
┌────────────────┐ SPSC  ┌───────────────────────────────┐ SPSC ┌──────────────────┐
│ downmix → mono │ ────► │ rubato  → 48 kHz              │ ───► │ mono → M ch      │
│ try_push       │ in_rb │ voice chain (below)           │ out  │ try_pop          │
│ unpark(worker) │       │ rubato  → output device rate  │ _rb  │ silence on empty │
└────────────────┘       └───────────────────────────────┘      └──────────────────┘
```

Voice chain (48 kHz mono, per 10 ms frame, `src/voice.rs`, `src/fx.rs`):

```
high-pass 80 Hz ─► RNNoise ─► speech-aware mix ─► voice gate ─► AGC ─► limiter ─► effects ─► limiter
                     │  dry (delay-aligned) ─┘           ▲
                     └──── VAD probability ──────────────┘ (also drives mix and AGC)
```

* `src/resampler.rs` – streaming wrapper over `rubato::SincFixedIn` (pass-through
  when the device already runs at 48 kHz).
* `src/denoiser.rs` – wrapper over `nnnoiseless::DenoiseState`, gathers 480-sample
  frames, converts between ±1.0 floats and the i16 range RNNoise expects, and returns
  the dry signal delay-aligned with the network output (its 480-sample model delay).
* `src/voice.rs` – the voice chain above, presets, and lock-free live controls.
* `src/fx.rs` – voice effects: YIN pitch tracker, TD-PSOLA pitch/formant engine,
  auto-tune, robot, radio, echo, reverb; lock-free live controls.
* `src/audio_engine.rs` – device selection, stream configuration, ring buffers,
  worker thread, latency clamp, graceful shutdown.
* `src/main.rs` – `clap` CLI (`crisp-rs`).
* `src/bin/gui.rs` – desktop GUI (`crisp-gui`, egui/eframe): device pickers, noise
  reduction presets and live controls, Start/Stop, live mic & output meters, VAD, gate
  and AGC indicators, latency budget and xrun counters. The UI thread only reads atomics; audio runs exactly the
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

## Noise reduction levels

| Preset | Voice relief | Voice gate | Use it for |
|---|---|---|---|
| **Nhẹ / Light** | 70 % | light (−15 dB) | quiet rooms, most natural voice |
| **Cân bằng / Balanced** (default) | 50 % | light (−15 dB) | everyday calls |
| **Mạnh / Strong** | 0 % | strong (−45 dB) | noisy rooms, cafés, open offices |

All presets run RNNoise at full strength in pauses and enable the 80 Hz high-pass
(fan / AC / desk rumble) and a −1 dBFS limiter. Every part can be tuned under
"Chỉnh chi tiết" / with CLI flags, **while audio is running**, and the A/B switch plays
the raw microphone through the same latency path for a fair comparison.

* **Voice relief** ("Giữ giọng tự nhiên"): while voice is detected, part of the
  delay-aligned raw signal is mixed back in. Speech masks the residual noise, and what
  you would otherwise hear are RNNoise's own artefacts ("watery" voice). Pauses stay
  fully cleaned. In very loud rooms the voice detector fires on noise, so the Strong
  preset uses none.
* **Voice gate** uses RNNoise's voice-activity probability (hysteresis 0.6 / 0.3,
  200 ms hold, 3 ms open, 40 ms close), so words and short gaps stay intact and the
  level drops only in real pauses — e.g. while you listen to others.
* **AGC** is **off** in every preset: Discord / Zoom / Meet / Teams run their own, and
  two AGCs in series pump. When enabled (OBS, recorders) it is deliberately slow: ~1 s
  speech-level estimate, at most 3 dB/s, target −23 dBFS.

Measured on real speech (two LibriSpeech readers, male ~100 Hz and female ~220 Hz,
normalised to −26 dBFS) with fan noise, 50 Hz hum and keyboard clicks added.
"Clean speech" = processing the speech with no noise at all, i.e. pure distortion:

| Configuration | Clean speech SI-SDR | Noisy (10 dB SNR) SI-SDR | Pause noise (median) | Noisy (0 dB SNR) pause noise |
|---|---|---|---|---|
| Unprocessed | ∞ | 10.0 dB | −36 dB | −26 dB |
| v0.3.0 Balanced (fast AGC) | 10.1 – 10.9 dB | | | |
| RNNoise only | 14.5 dB | 10.7 dB | −70.8 dB | −57.4 dB |
| Light | **23.9 dB** | 13.0 dB | −82.2 dB | −47.3 dB |
| Balanced | 21.3 dB | 12.9 dB | −82.7 dB | −49.1 dB |
| Strong | 16.4 dB | 11.5 dB | **−98.2 dB** | **−58.2 dB** |

v0.3.0's AGC reacted every 10 ms and squashed each loud syllable; that, not RNNoise,
was the main audible distortion. Defaults for voice relief were chosen from a sweep:
at 10 dB SNR overall SI-SDR peaks around 50–70 %, at 0 dB SNR relief lets noise into
pauses — hence 0 % for Strong.

## Voice effects (voice changer, auto-tune, ...)

Applied after noise reduction, live, in the app ("Hiệu ứng giọng") or with CLI flags.

| Preset | What it does |
|---|---|
| Trầm / Deep | −4 semitones, formant −2 |
| Cao / High | +5 semitones, formant +3 |
| Trẻ con / Child | +8 semitones, formant +5 |
| Quái vật / Monster | −8 semitones, formant −5, reverb |
| Robot | pitch flattened to 110 Hz, echo |
| Auto-tune | snaps to the nearest note (key + chromatic/major/minor, retune 0–300 ms) |
| Radio | 300–3400 Hz band + light saturation |
| Vọng / Echo | 220 ms echo |
| Hang động / Cave | reverb, −1 semitone |

* **Pitch** uses TD-PSOLA: pitch-synchronous grains (asymmetric raised-cosine windows
  from mark to mark) are laid down at a new spacing, so the pitch changes while the
  formants stay, and +5 semitones does not sound like a chipmunk. **Formant** shift
  resamples each grain (±5 semitones) to change timbre independently.
* **Pitch detection**: YIN on a 4× decimated copy, 75–800 Hz, every 5 ms; the grain's
  own mark-to-mark period refines it. On the two real voices it finds 83 % / 92 % of
  voiced frames, 91–93 % of them within 5 %.
* Unvoiced sounds and "no change" are reconstructed exactly (unit test: identity within
  1e-4), so consonants are not smeared.
* **Latency**: +38 ms while a pitch effect (pitch, formant, auto-tune, robot) is on,
  nothing otherwise. Switching on/off live crossfades over one 10 ms frame.

Measured on the same real voices (independent autocorrelation pitch reference):

| Effect | Male voice (~100 Hz) | Female voice (~220 Hz) |
|---|---|---|
| Deep −4 | median −3.9 st | −4.0 st (95 % of frames within ±1 st) |
| High +5 | +5.0 st (92 %) | +5.0 st (96 %) |
| Child +8 | +8.0 st (89 %) | +8.0 st (95 %) |
| Monster −8 | −7.9 st (74 %) | −8.0 st (88 %) |
| Robot | 110 Hz, pitch spread ×1.009 (input ×1.41) | 110 Hz, ×1.006 (input ×1.54) |
| Auto-tune | 66 % of frames within ±0.2 st of a note (input 36 %) | 62 % (input 39 %); median deviation 0.25 → 0.11 st |

Auto-tune on speech is limited by voicing detection (frames not recognised as voiced
are left alone); on sustained or sung notes it locks fully (unit test).

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

# noise reduction level / fine tuning
crisp-rs run --virtual-mic --preset strong
crisp-rs run --virtual-mic --strength 70 --gate light --no-agc

# A/B: raw microphone through the same latency path
crisp-rs run --bypass

# voice changer / auto-tune / effects
crisp-rs run --virtual-mic --voice deep
crisp-rs run --virtual-mic --pitch 3 --formant 2
crisp-rs run --virtual-mic --autotune A --scale minor --retune-ms 40
crisp-rs run --virtual-mic --robot --effect echo

# denoise a recording offline with the same chain (output: 48 kHz mono WAV)
crisp-rs process meeting.wav meeting-clean.wav --preset balanced

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
input period + RNNoise         :  22.67 ms  (2.67 + 20.00 ms)
resampler in  (48000 -> 48000)  :   0.00 ms
resampler out (48000 -> 48000)  :   0.00 ms
output queue (steady state)     :   7.67 ms  (primes at 848 frames)
output period                   :   2.67 ms
≈ total                         :  33.00 ms
```

| Devices | Period | Estimated latency |
|---|---|---|
| 48 kHz both ends | 128 | 33.0 ms |
| 48 kHz both ends | 64 | 29.0 ms |
| 44.1 kHz both ends (resampled) | 128 | 37.9 ms |

RNNoise alone accounts for 20 ms, so a < 25 ms end-to-end target is not reachable with
it; ~30 ms is far below what chat apps add over the network. Pitch effects add 38 ms
while enabled.

Where the numbers come from:

* **RNNoise** – 20 ms, counted once for the whole chain. Framing: a sample at position
  *x* of a frame waits (480 − *x*) for the frame to complete and then *x* while that
  chunk drains on the output, i.e. exactly 10 ms for every sample. Model delay: the
  network analyses a 20 ms window and resynthesises by overlap-add, so its output lags
  its input by one more frame (measured: 480 samples).
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
