//! "Virtual microphone" routing: mic → crisp-rs → a *new input device* that
//! Discord / Zoom / Meet / OBS can select as their microphone.
//!
//! * **Linux (PulseAudio or PipeWire via pipewire-pulse):** we create the
//!   virtual mic ourselves with `pactl`:
//!     - `module-null-sink`   `crisp_denoised` (48 kHz mono) — we play into it,
//!     - `module-remap-source` `crisp_mic` on its monitor — apps record from it.
//!
//!   Our playback stream is then moved onto the null sink. Modules are
//!   unloaded when the `VirtualMic` is dropped (re-used if left over from a crash).
//! * **Windows / macOS:** creating an audio endpoint requires a signed kernel
//!   / HAL driver, which a normal application cannot ship. We therefore look
//!   for an installed virtual cable (VB-Audio Cable, BlackHole, Loopback…),
//!   play into its *input/playback* side and tell the user which *recording*
//!   device to pick in their chat app.

use anyhow::{bail, Result};

use crate::audio_engine::DeviceInfo;

/// Name shown to chat apps on Linux.
pub const LINUX_MIC_DESCRIPTION: &str = "Crisp Mic (denoised)";
#[cfg(target_os = "linux")]
const LINUX_SINK_NAME: &str = "crisp_denoised";
#[cfg(target_os = "linux")]
const LINUX_SOURCE_NAME: &str = "crisp_mic";

/// Known virtual-cable products: (lower-case substring of the *playback*
/// device we write to, the *recording* device the user selects, product name).
const KNOWN_CABLES: &[(&str, &str, &str)] = &[
    ("cable input", "CABLE Output (VB-Audio Virtual Cable)", "VB-Audio Virtual Cable"),
    ("cable-a input", "CABLE-A Output (VB-Audio Cable A)", "VB-Audio Cable A"),
    ("cable-b input", "CABLE-B Output (VB-Audio Cable B)", "VB-Audio Cable B"),
    ("voicemeeter aux input", "Voicemeeter Out B2 (VB-Audio Voicemeeter AUX VAIO)", "Voicemeeter"),
    ("blackhole", "", "BlackHole"),
    ("loopback audio", "", "Rogue Amoeba Loopback"),
];

/// Lower-case substrings identifying devices that *are* a virtual mic. Using
/// one of them as the *input* would create a feedback loop.
const VIRTUAL_INPUT_MARKERS: &[&str] = &[
    "crisp mic",
    "crisp_mic",
    "crisp_denoised",
    "cable output",
    "cable-a output",
    "cable-b output",
    "voicemeeter out",
    "blackhole",
    "loopback audio",
];

/// True if `name` looks like a virtual-mic recording device (never use it as input).
pub fn is_virtual_input(name: &str) -> bool {
    let n = name.to_lowercase();
    VIRTUAL_INPUT_MARKERS.iter().any(|m| n.contains(m))
}

/// Where to install a virtual cable on this platform.
pub fn install_hint() -> (&'static str, &'static str) {
    if cfg!(target_os = "macos") {
        ("BlackHole 2ch (miễn phí)", "https://existential.audio/blackhole/")
    } else if cfg!(target_os = "windows") {
        ("VB-Audio Virtual Cable (miễn phí)", "https://vb-audio.com/Cable/")
    } else {
        (
            "PulseAudio hoặc PipeWire (gói pipewire-pulse) và lệnh pactl",
            "https://wiki.archlinux.org/title/PipeWire",
        )
    }
}

/// A virtual cable found among the output devices.
#[derive(Debug, Clone)]
pub struct CableMatch {
    pub output: DeviceInfo,
    /// Recording device the user must select in their chat app.
    pub mic_name: String,
    pub product: &'static str,
}

/// Find an installed virtual cable among `outputs` (first known product wins).
pub fn find_cable(outputs: &[DeviceInfo]) -> Option<CableMatch> {
    for (needle, mic, product) in KNOWN_CABLES {
        if let Some(d) = outputs.iter().find(|d| d.name.to_lowercase().contains(needle)) {
            let mic_name = if mic.is_empty() { d.name.clone() } else { (*mic).to_string() };
            return Some(CableMatch {
                output: d.clone(),
                mic_name,
                product,
            });
        }
    }
    None
}

/// What the virtual-mic mode would do on this machine (for UIs, no side effects).
#[derive(Debug, Clone)]
pub enum Availability {
    /// Linux: we can create the virtual mic ourselves.
    CreatedByApp { mic_name: String },
    /// An installed cable will be used.
    Cable(CableMatch),
    /// Nothing usable; `reason` explains, `install_hint()` says what to install.
    Unavailable { reason: String },
}

pub fn availability(outputs: &[DeviceInfo]) -> Availability {
    #[cfg(target_os = "linux")]
    {
        if pulse::server_available() {
            return Availability::CreatedByApp {
                mic_name: LINUX_MIC_DESCRIPTION.to_string(),
            };
        }
    }
    match find_cable(outputs) {
        Some(c) => Availability::Cable(c),
        None => Availability::Unavailable {
            reason: if cfg!(target_os = "linux") {
                "Không thấy PulseAudio/PipeWire (pactl) để tạo micro ảo.".into()
            } else {
                "Chưa cài virtual audio cable nên chưa thể tạo micro ảo.".into()
            },
        },
    }
}

/// An active virtual microphone. Keep it alive for as long as chat apps
/// should see the device; dropping it removes a Linux virtual mic.
pub struct VirtualMic {
    /// Selector for `EngineConfig::output`.
    output_selector: String,
    mic_name: String,
    backend: String,
    // Field order matters: the router is dropped (stopped) before the
    // modules it routes to are unloaded.
    /// Keeps our playback streams on the virtual-mic sink (Linux).
    #[cfg(target_os = "linux")]
    router: std::sync::OnceLock<pulse::StreamRouter>,
    #[cfg(target_os = "linux")]
    pulse: Option<pulse::PulseVirtualMic>,
}

impl VirtualMic {
    /// Create (Linux) or locate (cable) the virtual mic.
    pub fn prepare(outputs: &[DeviceInfo]) -> Result<VirtualMic> {
        #[cfg(target_os = "linux")]
        {
            if pulse::server_available() {
                let pm = pulse::PulseVirtualMic::create()?;
                // Prefer the ALSA "pulse"/"pipewire" plugin device so the
                // stream is visible to the sound server and can be moved.
                let selector = ["pulse", "pipewire", "default"]
                    .iter()
                    .find(|n| outputs.iter().any(|d| d.name == **n))
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "default".into());
                return Ok(VirtualMic {
                    output_selector: selector,
                    mic_name: LINUX_MIC_DESCRIPTION.to_string(),
                    backend: if pm.is_pipewire() { "PipeWire" } else { "PulseAudio" }.into(),
                    router: std::sync::OnceLock::new(),
                    pulse: Some(pm),
                });
            }
        }
        match find_cable(outputs) {
            Some(c) => Ok(VirtualMic {
                output_selector: c.output.index.to_string(),
                mic_name: c.mic_name,
                backend: c.product.to_string(),
                #[cfg(target_os = "linux")]
                router: std::sync::OnceLock::new(),
                #[cfg(target_os = "linux")]
                pulse: None,
            }),
            None => {
                let (what, url) = install_hint();
                bail!("chưa có micro ảo: hãy cài {what} ({url}) rồi bấm Làm mới / chạy lại")
            }
        }
    }

    pub fn output_selector(&self) -> &str {
        &self.output_selector
    }

    /// Recording-device name to select in Discord / Zoom / Meet.
    pub fn mic_name(&self) -> &str {
        &self.mic_name
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// Refuse configurations that would loop the virtual mic back into itself.
    pub fn check_input(&self, input_name: &str) -> Result<()> {
        if is_virtual_input(input_name) {
            bail!("micro đầu vào \"{input_name}\" chính là micro ảo — hãy chọn micro thật");
        }
        #[cfg(target_os = "linux")]
        if self.pulse.is_some() && pulse::default_source_is_ours() {
            bail!(
                "micro mặc định của hệ thống đang là \"{}\" — hãy chọn micro thật trong danh sách",
                LINUX_MIC_DESCRIPTION
            );
        }
        Ok(())
    }

    /// Call right after the engine started: routes our playback stream into
    /// the virtual mic (Linux). No-op for cables (the device was selected directly).
    ///
    /// The sound server's ALSA plugin re-creates its stream after every xrun
    /// (and a new engine start creates a new one), and a new stream lands on
    /// the *default* sink. So besides moving the current stream we keep a
    /// router running (`pactl subscribe`) that moves every new playback
    /// stream of this process onto the virtual-mic sink as soon as it appears.
    pub fn route_engine_output(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        if self.pulse.is_some() {
            let pid = std::process::id();
            if self.router.get().is_none() {
                match pulse::StreamRouter::start(pid) {
                    Ok(r) => {
                        let _ = self.router.set(r);
                    }
                    Err(e) => log::warn!("stream router not available ({e:#}); routing once"),
                }
            }
            return pulse::move_own_streams_to_sink(pid);
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod pulse {
    use std::io::{BufRead, BufReader};
    use std::process::{Child, Command, Stdio};
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    use anyhow::{anyhow, bail, Context, Result};
    use log::{debug, info, warn};

    use super::{LINUX_MIC_DESCRIPTION, LINUX_SINK_NAME, LINUX_SOURCE_NAME};

    fn pactl(args: &[&str]) -> Result<String> {
        let out = Command::new("pactl")
            .args(args)
            .env("LC_ALL", "C")
            .output()
            .context("không chạy được `pactl` (cài pulseaudio-utils / pipewire-pulse)")?;
        if !out.status.success() {
            bail!(
                "pactl {} thất bại: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    pub fn server_available() -> bool {
        pactl(&["info"]).is_ok()
    }

    pub fn default_source_is_ours() -> bool {
        pactl(&["get-default-source"])
            .map(|s| s.trim() == LINUX_SOURCE_NAME)
            .unwrap_or(false)
    }

    /// Second column of `pactl list short <kind>` (object names).
    fn short_names(kind: &str) -> Result<Vec<String>> {
        Ok(pactl(&["list", "short", kind])?
            .lines()
            .filter_map(|l| l.split('\t').nth(1).map(str::to_string))
            .collect())
    }

    pub struct PulseVirtualMic {
        /// Modules we loaded (unloaded in reverse order on drop).
        modules: Vec<u32>,
        pipewire: bool,
    }

    impl PulseVirtualMic {
        pub fn create() -> Result<Self> {
            let pipewire = pactl(&["info"])
                .map(|s| s.to_lowercase().contains("pipewire"))
                .unwrap_or(false);
            let mut me = PulseVirtualMic {
                modules: Vec::new(),
                pipewire,
            };

            if !short_names("sinks")?.iter().any(|n| n == LINUX_SINK_NAME) {
                let id = load_module(&[
                    "module-null-sink",
                    &format!("sink_name={LINUX_SINK_NAME}"),
                    // 48 kHz mono = RNNoise's native format: no resampling anywhere.
                    "rate=48000",
                    "channels=1",
                    "sink_properties='device.description=\"Crisp internal sink (do not select)\"'",
                ])?;
                me.modules.push(id);
            } else {
                // Left over from a crashed / killed run: adopt it so it is
                // removed when we exit.
                me.modules.extend(module_ids_with_arg(&format!("sink_name={LINUX_SINK_NAME}"))?);
                info!("re-using existing sink {LINUX_SINK_NAME}");
            }

            if !short_names("sources")?.iter().any(|n| n == LINUX_SOURCE_NAME) {
                let id = load_module(&[
                    "module-remap-source",
                    &format!("master={LINUX_SINK_NAME}.monitor"),
                    &format!("source_name={LINUX_SOURCE_NAME}"),
                    "channels=1",
                    &format!(
                        "source_properties='device.description=\"{LINUX_MIC_DESCRIPTION}\" device.icon_name=\"audio-input-microphone\"'"
                    ),
                ])?;
                me.modules.push(id);
            } else {
                me.modules.extend(module_ids_with_arg(&format!("source_name={LINUX_SOURCE_NAME}"))?);
                info!("re-using existing source {LINUX_SOURCE_NAME}");
            }

            info!(
                "virtual mic \"{LINUX_MIC_DESCRIPTION}\" ready ({})",
                if pipewire { "PipeWire" } else { "PulseAudio" }
            );
            Ok(me)
        }

        pub fn is_pipewire(&self) -> bool {
            self.pipewire
        }
    }

    impl Drop for PulseVirtualMic {
        fn drop(&mut self) {
            debug!("removing virtual mic modules {:?}", self.modules);
            for id in self.modules.drain(..).rev() {
                if let Err(e) = pactl(&["unload-module", &id.to_string()]) {
                    warn!("unloading module {id}: {e:#}");
                }
            }
        }
    }

    /// Ids of loaded modules whose argument string contains `arg` as a word.
    fn module_ids_with_arg(arg: &str) -> Result<Vec<u32>> {
        Ok(pactl(&["list", "short", "modules"])?
            .lines()
            .filter(|l| l.split(['\t', ' ']).any(|w| w == arg))
            .filter_map(|l| l.split('\t').next()?.trim().parse().ok())
            .collect())
    }

    fn load_module(args: &[&str]) -> Result<u32> {
        let mut full = vec!["load-module"];
        full.extend_from_slice(args);
        let out = pactl(&full)?;
        out.trim()
            .parse::<u32>()
            .map_err(|_| anyhow!("unexpected `pactl load-module` output: {out:?}"))
    }

    /// Index of our null sink (sink-inputs report their sink by index).
    fn our_sink_index() -> Result<String> {
        pactl(&["list", "short", "sinks"])?
            .lines()
            .find_map(|l| {
                let mut cols = l.split('\t');
                let idx = cols.next()?;
                (cols.next()? == LINUX_SINK_NAME).then(|| idx.to_string())
            })
            .ok_or_else(|| anyhow!("sink {LINUX_SINK_NAME} not found"))
    }

    /// Parse `pactl list sink-inputs` (LC_ALL=C) into (id, sink index) of
    /// the streams owned by process `pid`.
    pub(super) fn parse_sink_inputs(text: &str, pid: u32) -> Vec<(String, String)> {
        let wanted = format!("application.process.id = \"{pid}\"");
        let mut out = Vec::new();
        let mut id: Option<String> = None;
        let mut sink: Option<String> = None;
        for line in text.lines() {
            let t = line.trim();
            if let Some(rest) = line.strip_prefix("Sink Input #") {
                id = Some(rest.trim().to_string());
                sink = None;
            } else if let Some(rest) = t.strip_prefix("Sink: ") {
                sink = Some(rest.trim().to_string());
            } else if t == wanted {
                if let (Some(i), Some(s)) = (id.take(), sink.clone()) {
                    out.push((i, s));
                }
            }
        }
        out
    }

    /// Move this process's playback streams that are not yet on our sink.
    /// Returns how many streams of ours exist (moved or already in place).
    fn move_once(pid: u32) -> Result<usize> {
        let target = our_sink_index()?;
        let streams = parse_sink_inputs(&pactl(&["list", "sink-inputs"])?, pid);
        for (id, sink) in &streams {
            if *sink != target {
                pactl(&["move-sink-input", id, LINUX_SINK_NAME])?;
                debug!("moved sink-input #{id} to {LINUX_SINK_NAME}");
            }
        }
        Ok(streams.len())
    }

    /// Move every playback stream of this process onto the virtual-mic sink.
    /// The stream shows up a few ms after CPAL opens it, so retry briefly.
    pub fn move_own_streams_to_sink(pid: u32) -> Result<()> {
        for _ in 0..40 {
            if move_once(pid)? > 0 {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(50));
        }
        bail!("không tìm thấy luồng phát của crisp-rs trong sound server để nối vào micro ảo")
    }

    /// Background watcher: `pactl subscribe` prints one line per server
    /// event; on every new sink-input we re-apply the routing. Event-driven,
    /// so a re-created stream is moved within milliseconds.
    pub struct StreamRouter {
        child: Child,
        thread: Option<JoinHandle<()>>,
    }

    impl StreamRouter {
        pub fn start(pid: u32) -> Result<Self> {
            let mut child = Command::new("pactl")
                .arg("subscribe")
                .env("LC_ALL", "C")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .context("starting `pactl subscribe`")?;
            let stdout = child.stdout.take().context("pactl subscribe stdout")?;
            let thread = thread::Builder::new()
                .name("crisp-router".into())
                .spawn(move || {
                    for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
                        if line.contains("'new' on sink-input") {
                            if let Err(e) = move_once(pid) {
                                debug!("router: {e:#}");
                            }
                        }
                    }
                })
                .context("spawning router thread")?;
            Ok(Self {
                child,
                thread: Some(thread),
            })
        }
    }

    impl Drop for StreamRouter {
        fn drop(&mut self) {
            // Killing pactl closes its stdout, which ends the reader thread.
            let _ = self.child.kill();
            let _ = self.child.wait();
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(index: usize, name: &str) -> DeviceInfo {
        DeviceInfo {
            index,
            name: name.into(),
            is_default: false,
            default_rate: Some(48_000),
            default_channels: Some(2),
        }
    }

    #[test]
    fn finds_vb_cable_and_names_its_recording_side() {
        let outs = vec![dev(0, "Speakers (Realtek)"), dev(1, "CABLE Input (VB-Audio Virtual Cable)")];
        let c = find_cable(&outs).expect("cable");
        assert_eq!(c.output.index, 1);
        assert_eq!(c.mic_name, "CABLE Output (VB-Audio Virtual Cable)");
    }

    #[test]
    fn blackhole_uses_same_device_name_for_mic() {
        let outs = vec![dev(0, "MacBook Speakers"), dev(3, "BlackHole 2ch")];
        let c = find_cable(&outs).expect("cable");
        assert_eq!(c.output.index, 3);
        assert_eq!(c.mic_name, "BlackHole 2ch");
    }

    #[test]
    fn no_cable_among_plain_devices() {
        assert!(find_cable(&[dev(0, "Speakers"), dev(1, "HDMI")]).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_sink_inputs_of_our_pid() {
        let text = "Sink Input #12\n\tDriver: protocol-native.c\n\tSink: 0\n\tProperties:\n\t\tapplication.name = \"ALSA plug-in [crisp-gui]\"\n\t\tapplication.process.id = \"4242\"\n\nSink Input #13\n\tSink: 3\n\tProperties:\n\t\tapplication.process.id = \"999\"\n";
        assert_eq!(
            pulse::parse_sink_inputs(text, 4242),
            vec![("12".to_string(), "0".to_string())]
        );
        assert!(pulse::parse_sink_inputs(text, 1).is_empty());
    }

    #[test]
    fn detects_virtual_inputs() {
        assert!(is_virtual_input("CABLE Output (VB-Audio Virtual Cable)"));
        assert!(is_virtual_input("Crisp Mic (denoised)"));
        assert!(!is_virtual_input("Microphone (USB Audio)"));
    }
}
