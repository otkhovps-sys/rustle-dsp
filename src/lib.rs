//! crisp-rs — real-time microphone denoiser (RNNoise) with a lock-free,
//! low-latency CPAL pipeline. This library is shared by the CLI (`crisp-rs`)
//! and the desktop GUI (`crisp-gui`).

pub mod audio_engine;
pub mod denoiser;
pub mod resampler;
pub mod virtual_mic;

pub use audio_engine::{
    device_catalog, list_devices, DeviceCatalog, DeviceInfo, Engine, EngineConfig, EngineInfo,
    LatencyBudget, Metrics, MetricsSnapshot,
};
