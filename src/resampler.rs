//! Sample-rate conversion between the physical device rate and the fixed
//! 48 kHz that RNNoise requires.
//!
//! The wrapper is *streaming*: callers feed arbitrary-length slices and
//! receive output through a sink callback. Internally `rubato::SincFixedIn`
//! wants exactly `CHUNK_FRAMES` input frames per call, so a small FIFO gathers
//! input until a chunk is complete.
//!
//! When the two rates are equal the resampler is a zero-cost pass-through, so
//! the rest of the pipeline never has to special-case "device already runs at
//! 48 kHz".
//!
//! All buffers are allocated once in `new()`. `process()`/`flush()` never
//! allocate, which keeps the DSP worker free of latency spikes.

use anyhow::{bail, Context, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// RNNoise is trained for exactly 48 kHz.
pub const RNNOISE_RATE: u32 = 48_000;

/// Input frames handed to rubato per call.
///
/// Latency trade-off: the FIFO can hold up to `CHUNK_FRAMES - 1` frames before
/// a chunk completes, so this adds at most 64 / 48 000 s ≈ 1.3 ms of
/// buffering latency per resampler. Smaller chunks reduce that further but
/// increase per-call overhead; 64 keeps a 44.1 kHz <-> 48 kHz chain (two
/// resamplers) inside the 25 ms budget alongside the 10 ms RNNoise frame.
pub const CHUNK_FRAMES: usize = 64;

/// Half the sinc kernel length is the group delay of the filter
/// (64 taps -> 32 frames ≈ 0.67 ms at 48 kHz). 64 taps still gives > 90 dB
/// stop-band attenuation for speech material, which is plenty after RNNoise.
const SINC_LEN: usize = 64;

pub struct StreamResampler {
    /// `None` when `in_rate == out_rate` (pure pass-through).
    inner: Option<SincFixedIn<f32>>,
    in_rate: u32,
    out_rate: u32,
    /// Pending input frames (always exactly `CHUNK_FRAMES` long, `fifo_len` used).
    fifo: Vec<f32>,
    fifo_len: usize,
    /// rubato output buffer, 1 channel, sized `output_frames_max()`.
    out_buf: Vec<Vec<f32>>,
}

impl StreamResampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Result<Self> {
        if in_rate == 0 || out_rate == 0 {
            bail!("invalid sample rate: {in_rate} -> {out_rate}");
        }

        if in_rate == out_rate {
            return Ok(Self {
                inner: None,
                in_rate,
                out_rate,
                fifo: Vec::new(),
                fifo_len: 0,
                out_buf: Vec::new(),
            });
        }

        let params = SincInterpolationParameters {
            sinc_len: SINC_LEN,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };

        // Ratio is output/input. `max_resample_ratio_relative = 1.0` disables
        // run-time ratio changes, which lets rubato pre-compute everything.
        let ratio = f64::from(out_rate) / f64::from(in_rate);
        let resampler = SincFixedIn::<f32>::new(ratio, 1.0, params, CHUNK_FRAMES, 1)
            .with_context(|| format!("creating resampler {in_rate} Hz -> {out_rate} Hz"))?;

        // Pre-allocate the output buffer for the worst case so `process()` never allocates.
        let out_buf = resampler.output_buffer_allocate(true);

        Ok(Self {
            inner: Some(resampler),
            in_rate,
            out_rate,
            fifo: vec![0.0; CHUNK_FRAMES],
            fifo_len: 0,
            out_buf,
        })
    }

    pub fn is_passthrough(&self) -> bool {
        self.inner.is_none()
    }

    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }

    /// Steady-state algorithmic delay introduced by this stage, in milliseconds
    /// (sinc group delay + worst-case chunk gathering). Used for the latency
    /// budget printed at start-up.
    pub fn latency_ms(&self) -> f64 {
        match &self.inner {
            None => 0.0,
            Some(r) => {
                let group_delay = r.output_delay() as f64 / f64::from(self.out_rate);
                let chunking = CHUNK_FRAMES as f64 / f64::from(self.in_rate);
                (group_delay + chunking) * 1000.0
            }
        }
    }

    /// Feed `input` (mono, `in_rate`). Every completed chunk is resampled and
    /// delivered to `sink` as a mono slice at `out_rate`.
    pub fn process(&mut self, input: &[f32], sink: &mut dyn FnMut(&[f32])) -> Result<()> {
        let Some(resampler) = self.inner.as_mut() else {
            if !input.is_empty() {
                sink(input);
            }
            return Ok(());
        };

        let mut rest = input;
        while !rest.is_empty() {
            let room = CHUNK_FRAMES - self.fifo_len;
            let take = room.min(rest.len());
            self.fifo[self.fifo_len..self.fifo_len + take].copy_from_slice(&rest[..take]);
            self.fifo_len += take;
            rest = &rest[take..];

            if self.fifo_len == CHUNK_FRAMES {
                let (_consumed, produced) = resampler
                    .process_into_buffer(&[&self.fifo[..]], &mut self.out_buf, None)
                    .context("resampler process_into_buffer")?;
                self.fifo_len = 0;
                if produced > 0 {
                    sink(&self.out_buf[0][..produced]);
                }
            }
        }
        Ok(())
    }

    /// Push out whatever is still buffered (partial chunk + filter tail).
    /// Called once at shutdown so the last few milliseconds are not lost.
    pub fn flush(&mut self, sink: &mut dyn FnMut(&[f32])) -> Result<()> {
        let Some(resampler) = self.inner.as_mut() else {
            return Ok(());
        };

        if self.fifo_len > 0 {
            // Zero-pad the incomplete chunk and run it through normally.
            self.fifo[self.fifo_len..].fill(0.0);
            self.fifo_len = 0;
            let (_c, produced) = resampler
                .process_into_buffer(&[&self.fifo[..]], &mut self.out_buf, None)
                .context("resampler flush (partial chunk)")?;
            if produced > 0 {
                sink(&self.out_buf[0][..produced]);
            }
        }

        // Drain the sinc filter's internal delay line.
        let (_c, produced) = resampler
            .process_partial_into_buffer(None::<&[&[f32]]>, &mut self.out_buf, None)
            .context("resampler flush (tail)")?;
        if produced > 0 {
            sink(&self.out_buf[0][..produced]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_is_identity() {
        let mut r = StreamResampler::new(48_000, 48_000).unwrap();
        assert!(r.is_passthrough());
        let input: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        let mut out = Vec::new();
        r.process(&input, &mut |s| out.extend_from_slice(s)).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn converts_44100_to_48000_with_expected_frame_ratio() {
        let mut r = StreamResampler::new(44_100, 48_000).unwrap();
        let seconds = 2.0;
        let n_in = (44_100.0 * seconds) as usize;
        // Feed in odd-sized slices to exercise the FIFO path.
        let input: Vec<f32> =
            (0..n_in).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let mut out = Vec::new();
        for chunk in input.chunks(333) {
            r.process(chunk, &mut |s| out.extend_from_slice(s)).unwrap();
        }
        r.flush(&mut |s| out.extend_from_slice(s)).unwrap();
        let expected = 48_000.0 * seconds;
        let err = (out.len() as f64 - expected).abs() / expected;
        assert!(err < 0.01, "got {} frames, expected ~{}", out.len(), expected);
        assert!(out.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn latency_budget_is_small() {
        let r = StreamResampler::new(44_100, 48_000).unwrap();
        assert!(r.latency_ms() < 5.0, "latency {} ms", r.latency_ms());
    }
}
