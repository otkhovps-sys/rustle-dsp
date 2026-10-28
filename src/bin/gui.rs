//! crisp-gui — desktop front-end for the crisp-rs denoising engine.
//!
//! The UI thread never touches audio: it only reads atomics published by the
//! engine (`MetricsSnapshot`) and starts/stops the `Engine`, which owns the
//! CPAL streams on its own control thread.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, RichText, Vec2};
use crisp_rs::{device_catalog, DeviceCatalog, Engine, EngineConfig, MetricsSnapshot};

const BUFFER_CHOICES: &[u32] = &[0, 64, 128, 256, 512, 1024];
const RATE_CHOICES: &[u32] = &[0, 44_100, 48_000, 96_000];
const METER_DECAY_PER_SEC: f32 = 2.5;

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([560.0, 640.0])
            .with_min_inner_size([480.0, 520.0])
            .with_title("crisp-rs — microphone denoiser"),
        ..Default::default()
    };
    eframe::run_native(
        "crisp-rs",
        options,
        Box::new(|cc| Ok(Box::new(CrispApp::new(cc)))),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Idle,
    Running,
    Error,
}

struct CrispApp {
    catalog: DeviceCatalog,
    // Selection is by device index (what `EngineConfig` accepts); `None` = default.
    input_sel: Option<usize>,
    output_sel: Option<usize>,
    buffer_frames: u32,
    rate: u32,
    bypass: bool,

    engine: Option<Engine>,
    status: Status,
    message: String,
    started_at: Option<Instant>,

    // Smoothed meters (0..1) for display.
    in_meter: f32,
    out_meter: f32,
    vad_meter: f32,
    last_frame: Instant,
    last_snapshot: MetricsSnapshot,
}

impl CrispApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_fonts(&cc.egui_ctx);
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(8.0, 8.0);
        style.spacing.button_padding = Vec2::new(12.0, 6.0);
        cc.egui_ctx.set_style(style);

        Self {
            catalog: device_catalog(),
            input_sel: None,
            output_sel: None,
            buffer_frames: 128,
            rate: 0,
            bypass: false,
            engine: None,
            status: Status::Idle,
            message: "Chọn thiết bị rồi bấm Start.".into(),
            started_at: None,
            in_meter: 0.0,
            out_meter: 0.0,
            vad_meter: 0.0,
            last_frame: Instant::now(),
            last_snapshot: MetricsSnapshot::default(),
        }
    }

    fn config(&self) -> EngineConfig {
        EngineConfig {
            input: self.input_sel.map(|i| i.to_string()),
            output: self.output_sel.map(|i| i.to_string()),
            buffer_frames: self.buffer_frames,
            force_rate: (self.rate != 0).then_some(self.rate),
            bypass: self.bypass,
            stats_secs: 0,
        }
    }

    fn start(&mut self) {
        match Engine::start(self.config()) {
            Ok(engine) => {
                let info = engine.info();
                self.message = format!(
                    "Đang chạy: {} → {}  (ước tính trễ {:.1} ms)",
                    info.input_name, info.output_name, info.latency.total_ms
                );
                self.engine = Some(engine);
                self.status = Status::Running;
                self.started_at = Some(Instant::now());
            }
            Err(e) => {
                self.status = Status::Error;
                self.message = format!("Không khởi động được: {e:#}");
            }
        }
    }

    fn stop(&mut self) {
        if let Some(engine) = self.engine.take() {
            self.last_snapshot = engine.snapshot();
            match engine.stop() {
                Ok(()) => {
                    self.status = Status::Idle;
                    self.message = "Đã dừng (buffer đã được flush).".into();
                }
                Err(e) => {
                    self.status = Status::Error;
                    self.message = format!("Dừng với lỗi: {e:#}");
                }
            }
        }
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
        self.catalog = device_catalog();
        if self.input_sel.is_some_and(|i| i >= self.catalog.inputs.len()) {
            self.input_sel = None;
        }
        if self.output_sel.is_some_and(|i| i >= self.catalog.outputs.len()) {
            self.output_sel = None;
        }
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
                self.status = Status::Error;
                self.message = "Engine đã dừng bất thường (thiết bị bị ngắt?). Xem log.".into();
            }
        }
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
        Color32::from_rgb(230, 70, 70)
    } else if v > 0.8 {
        Color32::from_rgb(235, 180, 60)
    } else {
        Color32::from_rgb(80, 200, 120)
    }
}

fn level_bar(ui: &mut egui::Ui, label: &str, value: f32, text: String) {
    ui.horizontal(|ui| {
        ui.add_sized([70.0, 18.0], egui::Label::new(label));
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
    devices: &[crisp_rs::DeviceInfo],
    sel: &mut Option<usize>,
    enabled: bool,
) {
    ui.horizontal(|ui| {
        ui.add_sized([70.0, 20.0], egui::Label::new(label));
        let selected_text = match sel {
            Some(i) => devices
                .get(*i)
                .map(|d| d.name.clone())
                .unwrap_or_else(|| "?".into()),
            None => "Hệ thống mặc định".to_string(),
        };
        ui.add_enabled_ui(enabled, |ui| {
            egui::ComboBox::from_id_salt(id)
                .width(ui.available_width())
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    ui.selectable_value(sel, None, "Hệ thống mặc định");
                    for d in devices {
                        ui.selectable_value(sel, Some(d.index), d.label());
                    }
                });
        });
    });
}

impl eframe::App for CrispApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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
                ui.label(RichText::new("RNNoise real-time mic denoiser").weak());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (txt, col) = match self.status {
                        Status::Idle => ("● IDLE", Color32::GRAY),
                        Status::Running => ("● RUNNING", Color32::from_rgb(80, 200, 120)),
                        Status::Error => ("● ERROR", Color32::from_rgb(230, 70, 70)),
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
                    Status::Error => Color32::from_rgb(230, 70, 70),
                    _ => ui.visuals().text_color(),
                };
                ui.label(RichText::new(&self.message).color(col));
            });
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // ---------------- Devices ----------------
                ui.group(|ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.strong("Thiết bị");
                        ui.label(RichText::new(format!("host: {}", self.catalog.host)).weak());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add_enabled(!running, egui::Button::new("⟳ Làm mới"))
                                .clicked()
                            {
                                self.refresh_devices();
                            }
                        });
                    });
                    device_combo(ui, "in", "Micro", &self.catalog.inputs, &mut self.input_sel, !running);
                    device_combo(ui, "out", "Đầu ra", &self.catalog.outputs, &mut self.output_sel, !running);
                    if self.catalog.inputs.is_empty() || self.catalog.outputs.is_empty() {
                        ui.label(
                            RichText::new("Không tìm thấy thiết bị âm thanh. Cắm micro / kiểm tra driver rồi bấm Làm mới.")
                                .color(Color32::from_rgb(235, 180, 60)),
                        );
                    }
                });

                // ---------------- Settings ----------------
                ui.group(|ui| {
                    ui.set_width(ui.available_width());
                    ui.strong("Cấu hình");
                    ui.add_enabled_ui(!running, |ui| {
                        ui.horizontal(|ui| {
                            ui.add_sized([70.0, 20.0], egui::Label::new("Period"));
                            egui::ComboBox::from_id_salt("buf")
                                .selected_text(buffer_label(self.buffer_frames))
                                .show_ui(ui, |ui| {
                                    for &b in BUFFER_CHOICES {
                                        ui.selectable_value(&mut self.buffer_frames, b, buffer_label(b));
                                    }
                                });
                            ui.label(RichText::new("nhỏ hơn = trễ thấp hơn, dễ xrun hơn").weak().small());
                        });
                        ui.horizontal(|ui| {
                            ui.add_sized([70.0, 20.0], egui::Label::new("Sample rate"));
                            egui::ComboBox::from_id_salt("rate")
                                .selected_text(rate_label(self.rate))
                                .show_ui(ui, |ui| {
                                    for &r in RATE_CHOICES {
                                        ui.selectable_value(&mut self.rate, r, rate_label(r));
                                    }
                                });
                            ui.label(RichText::new("khác 48 kHz → dùng resampler").weak().small());
                        });
                    });
                    ui.horizontal(|ui| {
                        ui.add_sized([70.0, 20.0], egui::Label::new("Bypass"));
                        ui.add_enabled(!running, egui::Checkbox::new(&mut self.bypass, "tắt RNNoise để so sánh A/B (độ trễ giữ nguyên)"));
                    });
                });

                // ---------------- Transport ----------------
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let size = [ui.available_width(), 40.0];
                    if running {
                        let b = egui::Button::new(RichText::new("■  Stop").size(18.0))
                            .fill(Color32::from_rgb(120, 50, 50));
                        if ui.add_sized(size, b).clicked() {
                            self.stop();
                        }
                    } else {
                        let b = egui::Button::new(RichText::new("▶  Start").size(18.0))
                            .fill(Color32::from_rgb(40, 110, 70));
                        if ui.add_sized(size, b).clicked() {
                            self.start();
                        }
                    }
                });
                ui.add_space(4.0);

                // ---------------- Meters ----------------
                ui.group(|ui| {
                    ui.set_width(ui.available_width());
                    ui.strong("Tín hiệu");
                    let s = self.last_snapshot;
                    level_bar(ui, "Mic", self.in_meter, format!("{:.1} dBFS", to_db(s.in_peak)));
                    level_bar(ui, "Đầu ra", self.out_meter, format!("{:.1} dBFS", to_db(s.out_peak)));
                    ui.horizontal(|ui| {
                        ui.add_sized([70.0, 18.0], egui::Label::new("Giọng nói"));
                        let bar = egui::ProgressBar::new(self.vad_meter)
                            .fill(Color32::from_rgb(90, 140, 230))
                            .text(RichText::new(format!("VAD {:.0}%", self.vad_meter * 100.0)).small());
                        ui.add_sized([ui.available_width(), 18.0], bar);
                    });
                });

                // ---------------- Engine details ----------------
                if let Some(engine) = &self.engine {
                    let info = engine.info();
                    let l = &info.latency;
                    ui.group(|ui| {
                        ui.set_width(ui.available_width());
                        ui.strong("Pipeline");
                        egui::Grid::new("info").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
                            ui.label("Micro");
                            ui.label(format!("{} — {} Hz, {} ch, {}", info.input_name, info.input_rate, info.input_channels, info.input_format));
                            ui.end_row();
                            ui.label("Đầu ra");
                            ui.label(format!("{} — {} Hz, {} ch, {}", info.output_name, info.output_rate, info.output_channels, info.output_format));
                            ui.end_row();
                            ui.label("Resampler");
                            ui.label(if l.resample_in_ms + l.resample_out_ms > 0.0 {
                                format!("bật ({:.1} ms + {:.1} ms)", l.resample_in_ms, l.resample_out_ms)
                            } else {
                                "không cần (48 kHz native)".into()
                            });
                            ui.end_row();
                            ui.label("Độ trễ ước tính");
                            let col = if l.total_ms <= 25.0 { Color32::from_rgb(80, 200, 120) } else { Color32::from_rgb(235, 180, 60) };
                            ui.label(RichText::new(format!("{:.1} ms  (frame {:.0} + queue {:.1} + period {:.1})",
                                l.total_ms, l.input_side_ms, l.output_queue_ms, l.output_period_ms)).color(col));
                            ui.end_row();
                            ui.label("Thời gian chạy");
                            ui.label(format_duration(self.started_at.map(|t| t.elapsed()).unwrap_or_default()));
                            ui.end_row();
                        });
                    });
                }

                // ---------------- Health counters ----------------
                ui.group(|ui| {
                    ui.set_width(ui.available_width());
                    ui.strong("Sức khoẻ pipeline");
                    let s = self.last_snapshot;
                    egui::Grid::new("health").num_columns(4).spacing([16.0, 4.0]).show(ui, |ui| {
                        counter(ui, "Frames", s.frames_processed, false);
                        counter(ui, "Input overrun", s.input_overruns, true);
                        ui.end_row();
                        counter(ui, "Output underrun", s.output_underruns, true);
                        counter(ui, "Dropped (clamp)", s.dropped_samples, true);
                        ui.end_row();
                        counter(ui, "Stream errors", s.stream_errors, true);
                        ui.label("Hàng đợi");
                        ui.label(format!("in {:.1} ms / out {:.1} ms", s.in_fill_ms, s.out_fill_ms));
                        ui.end_row();
                    });
                    ui.label(RichText::new("Underrun/overrun tăng liên tục → chọn period lớn hơn (256) hoặc dùng thiết bị 48 kHz.").weak().small());
                });
            });
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Window closed while running: stop gracefully (flush + drain).
        self.stop();
    }
}

fn counter(ui: &mut egui::Ui, label: &str, v: u64, warn_nonzero: bool) {
    ui.label(label);
    let txt = RichText::new(v.to_string()).monospace();
    if warn_nonzero && v > 0 {
        ui.label(txt.color(Color32::from_rgb(235, 180, 60)));
    } else {
        ui.label(txt);
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
