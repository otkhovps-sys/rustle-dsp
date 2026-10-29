//! crisp-gui — desktop front-end for the crisp-rs denoising engine.
//!
//! Default use case: **virtual microphone**. Real mic → RNNoise → a virtual
//! input device that Discord / Zoom / Meet / OBS select as their microphone.
//! A "monitor" mode plays the result on speakers/headphones for testing.
//!
//! The UI thread never touches audio: it only reads atomics published by the
//! engine (`MetricsSnapshot`) and starts/stops the `Engine`, which owns the
//! CPAL streams on its own control thread.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::time::{Duration, Instant};

use crisp_rs::virtual_mic::{self, Availability, VirtualMic};
use crisp_rs::{device_catalog, DeviceCatalog, DeviceInfo, Engine, EngineConfig, MetricsSnapshot};
use eframe::egui::{self, Color32, RichText, Vec2};
use serde::{Deserialize, Serialize};

const BUFFER_CHOICES: &[u32] = &[0, 64, 128, 256, 512, 1024];
const RATE_CHOICES: &[u32] = &[0, 44_100, 48_000, 96_000];
const METER_DECAY_PER_SEC: f32 = 2.5;

const GREEN: Color32 = Color32::from_rgb(80, 200, 120);
const AMBER: Color32 = Color32::from_rgb(235, 180, 60);
const RED: Color32 = Color32::from_rgb(230, 70, 70);

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([580.0, 700.0])
            .with_min_inner_size([480.0, 540.0])
            .with_title("crisp-rs — micro khử ồn"),
        ..Default::default()
    };
    eframe::run_native(
        "crisp-rs",
        options,
        Box::new(|cc| Ok(Box::new(CrispApp::new(cc)))),
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Mode {
    /// Denoised audio goes into a virtual mic for chat apps.
    VirtualMic,
    /// Denoised audio is played on a normal output device (testing).
    Monitor,
}

/// Persisted between launches (eframe storage). Devices are stored by name
/// because indices change when hardware is plugged / unplugged.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    mode: Mode,
    input_name: Option<String>,
    output_name: Option<String>,
    buffer_frames: u32,
    rate: u32,
    bypass: bool,
    auto_start: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            mode: Mode::VirtualMic,
            input_name: None,
            output_name: None,
            buffer_frames: 128,
            rate: 0,
            bypass: false,
            auto_start: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Idle,
    Running,
    Error,
}

struct CrispApp {
    catalog: DeviceCatalog,
    availability: Availability,
    settings: Settings,
    // Selection is by device index (what `EngineConfig` accepts); `None` = default.
    input_sel: Option<usize>,
    output_sel: Option<usize>,

    engine: Option<Engine>,
    /// Kept for the whole app lifetime once created, so chat apps keep seeing
    /// the device across Start/Stop (dropped → Linux virtual mic removed).
    vmic: Option<VirtualMic>,
    /// Mic name to show while running in virtual-mic mode.
    active_mic: Option<String>,
    status: Status,
    message: String,
    started_at: Option<Instant>,
    pending_auto_start: bool,

    // Smoothed meters (0..1) for display.
    in_meter: f32,
    out_meter: f32,
    vad_meter: f32,
    last_frame: Instant,
    last_snapshot: MetricsSnapshot,
}

fn find_by_name(devs: &[DeviceInfo], name: &Option<String>) -> Option<usize> {
    let name = name.as_ref()?;
    devs.iter().find(|d| &d.name == name).map(|d| d.index)
}

impl CrispApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_fonts(&cc.egui_ctx);
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(8.0, 8.0);
        style.spacing.button_padding = Vec2::new(12.0, 6.0);
        cc.egui_ctx.set_style(style);

        let settings: Settings = cc
            .storage
            .and_then(|s| eframe::get_value(s, eframe::APP_KEY))
            .unwrap_or_default();
        let catalog = device_catalog();
        let availability = virtual_mic::availability(&catalog.outputs);

        Self {
            input_sel: find_by_name(&catalog.inputs, &settings.input_name),
            output_sel: find_by_name(&catalog.outputs, &settings.output_name),
            pending_auto_start: settings.auto_start,
            catalog,
            availability,
            settings,
            engine: None,
            vmic: None,
            active_mic: None,
            status: Status::Idle,
            message: "Chọn micro rồi bấm Bắt đầu.".into(),
            started_at: None,
            in_meter: 0.0,
            out_meter: 0.0,
            vad_meter: 0.0,
            last_frame: Instant::now(),
            last_snapshot: MetricsSnapshot::default(),
        }
    }

    fn input_name(&self) -> String {
        self.input_sel
            .and_then(|i| self.catalog.inputs.get(i))
            .map(|d| d.name.clone())
            .unwrap_or_else(|| "default".into())
    }

    fn fail(&mut self, msg: String) {
        self.status = Status::Error;
        self.message = msg;
    }

    fn start(&mut self) {
        let mut cfg = EngineConfig {
            input: self.input_sel.map(|i| i.to_string()),
            output: self.output_sel.map(|i| i.to_string()),
            buffer_frames: self.settings.buffer_frames,
            force_rate: (self.settings.rate != 0).then_some(self.settings.rate),
            bypass: self.settings.bypass,
            stats_secs: 0,
        };

        let virtual_mode = self.settings.mode == Mode::VirtualMic;
        if virtual_mode {
            if self.vmic.is_none() {
                match VirtualMic::prepare(&self.catalog.outputs) {
                    Ok(v) => self.vmic = Some(v),
                    Err(e) => return self.fail(format!("Không tạo được micro ảo: {e:#}")),
                }
            }
            let vmic = self.vmic.as_ref().expect("created above");
            if let Err(e) = vmic.check_input(&self.input_name()) {
                return self.fail(format!("{e:#}"));
            }
            cfg.output = Some(vmic.output_selector().to_string());
        }

        let engine = match Engine::start(cfg) {
            Ok(e) => e,
            Err(e) => return self.fail(format!("Không khởi động được: {e:#}")),
        };

        if virtual_mode {
            let vmic = self.vmic.as_ref().expect("created above");
            let routed = vmic
                .check_input(&engine.info().input_name)
                .and_then(|_| vmic.route_engine_output());
            if let Err(e) = routed {
                let _ = engine.stop();
                return self.fail(format!("Không nối được vào micro ảo: {e:#}"));
            }
            self.active_mic = Some(vmic.mic_name().to_string());
            self.message = format!(
                "Đang lọc ồn: {} → micro ảo “{}” (trễ ≈ {:.1} ms)",
                engine.info().input_name,
                vmic.mic_name(),
                engine.info().latency.total_ms
            );
        } else {
            self.active_mic = None;
            let info = engine.info();
            self.message = format!(
                "Đang nghe thử: {} → {}  (trễ ≈ {:.1} ms)",
                info.input_name, info.output_name, info.latency.total_ms
            );
        }
        self.engine = Some(engine);
        self.status = Status::Running;
        self.started_at = Some(Instant::now());
    }

    fn stop(&mut self) {
        if let Some(engine) = self.engine.take() {
            self.last_snapshot = engine.snapshot();
            match engine.stop() {
                Ok(()) => {
                    self.status = Status::Idle;
                    self.message = "Đã dừng lọc ồn.".into();
                }
                Err(e) => self.fail(format!("Dừng với lỗi: {e:#}")),
            }
        }
        self.active_mic = None;
        self.started_at = None;
        self.in_meter = 0.0;
        self.out_meter = 0.0;
        self.vad_meter = 0.0;
        // Keep the counters visible after stop, but clear instantaneous values.
        self.last_snapshot.in_peak = 0.0;
        self.last_snapshot.out_peak = 0.0;
        self.last_snapshot.vad = 0.0;
    }

    fn refresh_devices(&mut self) {
        let in_name = self.input_sel.and_then(|i| self.catalog.inputs.get(i)).map(|d| d.name.clone());
        let out_name = self.output_sel.and_then(|i| self.catalog.outputs.get(i)).map(|d| d.name.clone());
        self.catalog = device_catalog();
        self.input_sel = find_by_name(&self.catalog.inputs, &in_name);
        self.output_sel = find_by_name(&self.catalog.outputs, &out_name);
        // A newly installed cable is picked up here.
        if self.vmic.is_none() {
            self.availability = virtual_mic::availability(&self.catalog.outputs);
        }
    }

    fn sync_settings(&mut self) {
        self.settings.input_name = self.input_sel.and_then(|i| self.catalog.inputs.get(i)).map(|d| d.name.clone());
        self.settings.output_name = self.output_sel.and_then(|i| self.catalog.outputs.get(i)).map(|d| d.name.clone());
    }

    fn update_meters(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.25);
        self.last_frame = now;
        let decay = METER_DECAY_PER_SEC * dt;

        if let Some(engine) = &self.engine {
            let snap = engine.snapshot();
            self.last_snapshot = snap;
            // Peak-hold with linear decay so short bursts stay visible.
            self.in_meter = (self.in_meter - decay).max(amp_to_meter(snap.in_peak));
            self.out_meter = (self.out_meter - decay).max(amp_to_meter(snap.out_peak));
            // VAD is already 0..1; light smoothing.
            self.vad_meter += (snap.vad - self.vad_meter) * (dt * 10.0).min(1.0);

            if engine.is_finished() {
                // Worker or control thread exited on its own (fatal stream error).
                self.stop();
                self.fail("Engine đã dừng bất thường (thiết bị bị ngắt?). Bấm Làm mới rồi Bắt đầu lại.".into());
            }
        }
    }

    fn virtual_unavailable(&self) -> bool {
        self.vmic.is_none() && matches!(self.availability, Availability::Unavailable { .. })
    }
}

/// egui's bundled fonts lack most Vietnamese glyphs, so we embed DejaVu Sans
/// (full Latin Extended Additional coverage, free license — see
/// `assets/fonts/DejaVu-LICENSE.txt`) and put it first in both font families.
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "DejaVuSans".to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
            "../../assets/fonts/DejaVuSans.ttf"
        ))),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, "DejaVuSans".to_owned());
    }
    ctx.set_fonts(fonts);
}

/// Map linear peak amplitude to a 0..1 meter position using a dB scale
/// (-60 dBFS .. 0 dBFS), which is how humans perceive level.
fn amp_to_meter(peak: f32) -> f32 {
    if peak <= 1e-6 {
        return 0.0;
    }
    let db = 20.0 * peak.log10();
    ((db + 60.0) / 60.0).clamp(0.0, 1.0)
}

fn meter_color(v: f32) -> Color32 {
    if v > 0.95 {
        RED
    } else if v > 0.8 {
        AMBER
    } else {
        GREEN
    }
}

fn level_bar(ui: &mut egui::Ui, label: &str, value: f32, text: String) {
    ui.horizontal(|ui| {
        ui.add_sized([80.0, 18.0], egui::Label::new(label));
        let bar = egui::ProgressBar::new(value)
            .fill(meter_color(value))
            .text(RichText::new(text).small());
        ui.add_sized([ui.available_width(), 18.0], bar);
    });
}

fn device_combo(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    devices: &[&DeviceInfo],
    sel: &mut Option<usize>,
    enabled: bool,
) {
    ui.horizontal(|ui| {
        ui.add_sized([80.0, 20.0], egui::Label::new(label));
        let selected_text = match sel {
            Some(i) => devices
                .iter()
                .find(|d| d.index == *i)
                .map(|d| d.name.clone())
                .unwrap_or_else(|| "?".into()),
            None => "Mặc định của hệ thống".to_string(),
        };
        ui.add_enabled_ui(enabled, |ui| {
            egui::ComboBox::from_id_salt(id)
                .width(ui.available_width())
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    ui.selectable_value(sel, None, "Mặc định của hệ thống");
                    for d in devices {
                        ui.selectable_value(sel, Some(d.index), d.label());
                    }
                });
        });
    });
}

impl eframe::App for CrispApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.pending_auto_start {
            self.pending_auto_start = false;
            if !(self.settings.mode == Mode::VirtualMic && self.virtual_unavailable()) {
                self.start();
            }
        }
        self.update_meters();
        // Keep meters alive while running; idle UI repaints only on input.
        if self.engine.is_some() {
            ctx.request_repaint_after(Duration::from_millis(33));
        }

        let running = self.engine.is_some();

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading("crisp-rs");
                ui.label(RichText::new("micro khử ồn thời gian thực").weak());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (txt, col) = match self.status {
                        Status::Idle => ("● ĐANG TẮT", Color32::GRAY),
                        Status::Running => ("● ĐANG LỌC", GREEN),
                        Status::Error => ("● LỖI", RED),
                    };
                    ui.label(RichText::new(txt).color(col).strong());
                });
            });
            ui.add_space(6.0);
        });

        egui::TopBottomPanel::bottom("footer").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                let col = match self.status {
                    Status::Error => RED,
                    _ => ui.visuals().text_color(),
                };
                ui.label(RichText::new(&self.message).color(col));
            });
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // ---------------- Mode ----------------
                ui.group(|ui| {
                    ui.set_width(ui.available_width());
                    ui.strong("Chế độ");
                    ui.add_enabled_ui(!running, |ui| {
                        ui.radio_value(
                            &mut self.settings.mode,
                            Mode::VirtualMic,
                            "Micro ảo — dùng cho Discord, Zoom, Meet, Teams, OBS…",
                        );
                        ui.radio_value(
                            &mut self.settings.mode,
                            Mode::Monitor,
                            "Nghe thử — phát tiếng đã lọc ra loa / tai nghe",
                        );
                    });
                });

                // ---------------- Devices ----------------
                ui.group(|ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.strong("Thiết bị");
                        ui.label(RichText::new(format!("host: {}", self.catalog.host)).weak());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.add_enabled(!running, egui::Button::new("⟳ Làm mới")).clicked() {
                                self.refresh_devices();
                            }
                        });
                    });

                    // Never offer a virtual mic as the *input* (feedback loop).
                    let inputs: Vec<&DeviceInfo> = self
                        .catalog
                        .inputs
                        .iter()
                        .filter(|d| !virtual_mic::is_virtual_input(&d.name))
                        .collect();
                    device_combo(ui, "in", "Micro thật", &inputs, &mut self.input_sel, !running);

                    match self.settings.mode {
                        Mode::Monitor => {
                            let outputs: Vec<&DeviceInfo> = self.catalog.outputs.iter().collect();
                            device_combo(ui, "out", "Loa / tai nghe", &outputs, &mut self.output_sel, !running);
                        }
                        Mode::VirtualMic => {
                            ui.horizontal(|ui| {
                                ui.add_sized([80.0, 20.0], egui::Label::new("Micro ảo"));
                                match (&self.vmic, &self.availability) {
                                    (Some(v), _) => {
                                        ui.label(RichText::new(format!("“{}” ({})", v.mic_name(), v.backend())).color(GREEN));
                                    }
                                    (None, Availability::CreatedByApp { mic_name }) => {
                                        ui.label(RichText::new(format!("“{mic_name}” — sẽ được tạo tự động")).color(GREEN));
                                    }
                                    (None, Availability::Cable(c)) => {
                                        ui.label(RichText::new(format!("“{}” qua {}", c.mic_name, c.product)).color(GREEN));
                                    }
                                    (None, Availability::Unavailable { reason }) => {
                                        ui.label(RichText::new(reason).color(AMBER));
                                    }
                                }
                            });
                            if self.virtual_unavailable() {
                                let (what, url) = virtual_mic::install_hint();
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Cài");
                                    ui.hyperlink_to(what, url);
                                    ui.label("(một lần duy nhất), khởi động lại máy nếu được yêu cầu, rồi bấm Làm mới.");
                                });
                            }
                        }
                    }
                    if self.catalog.inputs.is_empty() {
                        ui.label(
                            RichText::new("Không tìm thấy micro. Cắm micro / kiểm tra driver rồi bấm Làm mới.")
                                .color(AMBER),
                        );
                    }
                });

                // ---------------- Transport ----------------
                ui.add_space(2.0);
                let blocked = self.settings.mode == Mode::VirtualMic && self.virtual_unavailable();
                ui.horizontal(|ui| {
                    let size = [ui.available_width(), 44.0];
                    if running {
                        let b = egui::Button::new(RichText::new("■  Dừng").size(18.0))
                            .fill(Color32::from_rgb(120, 50, 50));
                        if ui.add_sized(size, b).clicked() {
                            self.stop();
                        }
                    } else {
                        let b = egui::Button::new(RichText::new("▶  Bắt đầu lọc ồn").size(18.0))
                            .fill(Color32::from_rgb(40, 110, 70));
                        if ui.add_enabled_ui(!blocked, |ui| ui.add_sized(size, b)).inner.clicked() {
                            self.start();
                        }
                    }
                });

                // ---------------- How to use the virtual mic ----------------
                if let Some(mic) = self.active_mic.clone() {
                    egui::Frame::group(ui.style())
                        .fill(Color32::from_rgb(28, 56, 40))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(RichText::new("✔ Micro ảo đang hoạt động").color(GREEN).strong());
                            ui.label("Trong Discord / Zoom / Meet / Teams → Cài đặt âm thanh → Micro (Input), chọn:");
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(&mic).size(16.0).strong());
                                if ui.button("Sao chép tên").clicked() {
                                    ui.ctx().copy_text(mic.clone());
                                }
                            });
                            ui.label(
                                RichText::new("Tắt chống ồn có sẵn của app đó (Krisp, Noise suppression) để tránh lọc 2 lần làm méo tiếng.")
                                    .weak()
                                    .small(),
                            );
                        });
                }
                ui.add_space(2.0);

                // ---------------- Meters ----------------
                ui.group(|ui| {
                    ui.set_width(ui.available_width());
                    ui.strong("Tín hiệu");
                    let s = self.last_snapshot;
                    level_bar(ui, "Micro thật", self.in_meter, format!("{:.1} dBFS", to_db(s.in_peak)));
                    level_bar(ui, "Sau lọc", self.out_meter, format!("{:.1} dBFS", to_db(s.out_peak)));
                    ui.horizontal(|ui| {
                        ui.add_sized([80.0, 18.0], egui::Label::new("Giọng nói"));
                        let bar = egui::ProgressBar::new(self.vad_meter)
                            .fill(Color32::from_rgb(90, 140, 230))
                            .text(RichText::new(format!("VAD {:.0}%", self.vad_meter * 100.0)).small());
                        ui.add_sized([ui.available_width(), 18.0], bar);
                    });
                });

                // ---------------- Settings ----------------
                egui::CollapsingHeader::new("Cài đặt nâng cao")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.add_enabled_ui(!running, |ui| {
                            ui.horizontal(|ui| {
                                ui.add_sized([80.0, 20.0], egui::Label::new("Period"));
                                egui::ComboBox::from_id_salt("buf")
                                    .selected_text(buffer_label(self.settings.buffer_frames))
                                    .show_ui(ui, |ui| {
                                        for &b in BUFFER_CHOICES {
                                            ui.selectable_value(&mut self.settings.buffer_frames, b, buffer_label(b));
                                        }
                                    });
                                ui.label(RichText::new("nhỏ = trễ thấp, dễ rè hơn").weak().small());
                            });
                            ui.horizontal(|ui| {
                                ui.add_sized([80.0, 20.0], egui::Label::new("Sample rate"));
                                egui::ComboBox::from_id_salt("rate")
                                    .selected_text(rate_label(self.settings.rate))
                                    .show_ui(ui, |ui| {
                                        for &r in RATE_CHOICES {
                                            ui.selectable_value(&mut self.settings.rate, r, rate_label(r));
                                        }
                                    });
                            });
                            ui.checkbox(&mut self.settings.bypass, "Tắt RNNoise (so sánh A/B, độ trễ giữ nguyên)");
                        });
                        ui.checkbox(&mut self.settings.auto_start, "Tự bắt đầu lọc ồn khi mở app");
                    });

                // ---------------- Engine details ----------------
                if let Some(engine) = &self.engine {
                    let info = engine.info();
                    let l = &info.latency;
                    egui::CollapsingHeader::new("Chi tiết pipeline").show(ui, |ui| {
                        egui::Grid::new("info").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
                            ui.label("Vào");
                            ui.label(format!("{} — {} Hz, {} ch, {}", info.input_name, info.input_rate, info.input_channels, info.input_format));
                            ui.end_row();
                            ui.label("Ra");
                            ui.label(format!("{} — {} Hz, {} ch, {}", info.output_name, info.output_rate, info.output_channels, info.output_format));
                            ui.end_row();
                            ui.label("Resampler");
                            ui.label(if l.resample_in_ms + l.resample_out_ms > 0.0 {
                                format!("bật ({:.1} ms + {:.1} ms)", l.resample_in_ms, l.resample_out_ms)
                            } else {
                                "không cần (48 kHz)".into()
                            });
                            ui.end_row();
                            ui.label("Độ trễ ước tính");
                            let col = if l.total_ms <= 25.0 { GREEN } else { AMBER };
                            ui.label(RichText::new(format!("{:.1} ms", l.total_ms)).color(col));
                            ui.end_row();
                            ui.label("Thời gian chạy");
                            ui.label(format_duration(self.started_at.map(|t| t.elapsed()).unwrap_or_default()));
                            ui.end_row();
                            let s = self.last_snapshot;
                            ui.label("Rè / mất mẫu");
                            let bad = s.input_overruns + s.output_underruns + s.stream_errors;
                            ui.label(RichText::new(format!(
                                "underrun {} · overrun {} · lỗi {}",
                                s.output_underruns, s.input_overruns, s.stream_errors
                            )).color(if bad > 0 { AMBER } else { ui.visuals().text_color() }));
                            ui.end_row();
                        });
                    });
                }
            });
        });

        self.sync_settings();
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.sync_settings();
        eframe::set_value(storage, eframe::APP_KEY, &self.settings);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Window closed while running: stop gracefully (flush + drain), then
        // remove the Linux virtual mic.
        self.stop();
        self.vmic = None;
    }
}

fn to_db(peak: f32) -> f32 {
    if peak <= 1e-6 {
        -120.0
    } else {
        20.0 * peak.log10()
    }
}

fn buffer_label(b: u32) -> String {
    if b == 0 {
        "Mặc định của driver".into()
    } else {
        format!("{b} frames ({:.1} ms @48k)", b as f64 * 1000.0 / 48_000.0)
    }
}

fn rate_label(r: u32) -> String {
    if r == 0 {
        "Tự động (ưu tiên 48000)".into()
    } else {
        format!("{r} Hz")
    }
}

fn format_duration(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}
