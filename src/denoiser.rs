//! Thin wrapper around `nnnoiseless::DenoiseState`.
//!
//! RNNoise works on fixed frames of 480 samples (10 ms @ 48 kHz). Callers can
//! push arbitrary-length mono slices; this type gathers them into full frames,
//! runs the network, and hands back clean 480-sample frames.
//!
//! Scaling: RNNoise was trained on 16-bit PCM, so `nnnoiseless` expects sample
//! values in the i16 range (±32768) even though they are `f32`. CPAL gives us
//! normalised ±1.0 floats, so we scale on the way in and back out.

use nnnoiseless::DenoiseState;

/// 480 samples = 10 ms at 48 kHz. This is the dominant term of the end-to-end
/// latency budget: a sample may have to wait up to a full frame before the
/// network can run.
pub const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;

const PCM_SCALE: f32 = 32_768.0;
const INV_PCM_SCALE: f32 = 1.0 / PCM_SCALE;

pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    in_frame: [f32; FRAME_SIZE],
    out_frame: [f32; FRAME_SIZE],
    /// Number of valid samples currently gathered in `in_frame`.
    filled: usize,
    /// Voice-activity probability reported by the last processed frame.
    last_vad: f32,
    /// Bypass the network (A/B testing). Framing is kept identical so the
    /// latency is the same with and without denoising.
    bypass: bool,
    frames_processed: u64,
}

impl Denoiser {
    pub fn new(bypass: bool) -> Self {
        Self {
            state: DenoiseState::new(),
            in_frame: [0.0; FRAME_SIZE],
            out_frame: [0.0; FRAME_SIZE],
            filled: 0,
            last_vad: 0.0,
            bypass,
            frames_processed: 0,
        }
    }

    pub fn last_vad(&self) -> f32 {
        self.last_vad
    }

    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Samples buffered but not yet forming a complete frame.
    pub fn pending(&self) -> usize {
        self.filled
    }

    /// Feed mono 48 kHz samples in ±1.0. For every complete 480-sample frame
    /// `sink` is called once with the denoised frame (also ±1.0, clamped).
    pub fn push(&mut self, input: &[f32], sink: &mut dyn FnMut(&[f32])) {
        let mut rest = input;
        while !rest.is_empty() {
            let room = FRAME_SIZE - self.filled;
            let take = room.min(rest.len());

            // Scale to the i16 range the model expects while copying.
            for (dst, src) in self.in_frame[self.filled..self.filled + take]
                .iter_mut()
                .zip(&rest[..take])
            {
                *dst = src * PCM_SCALE;
            }
            self.filled += take;
            rest = &rest[take..];

            if self.filled == FRAME_SIZE {
                self.run_frame();
                sink(&self.out_frame);
            }
        }
    }

    /// Flush a partially filled frame (zero-padded) at shutdown so the tail of
    /// the stream is not dropped. Returns the number of *real* samples emitted.
    pub fn flush(&mut self, sink: &mut dyn FnMut(&[f32])) -> usize {
        if self.filled == 0 {
            return 0;
        }
        let real = self.filled;
        self.in_frame[real..].fill(0.0);
        self.filled = FRAME_SIZE;
        self.run_frame();
        sink(&self.out_frame[..real]);
        real
    }

    fn run_frame(&mut self) {
        debug_assert_eq!(self.filled, FRAME_SIZE);
        if self.bypass {
            self.out_frame.copy_from_slice(&self.in_frame);
        } else {
            self.last_vad = self.state.process_frame(&mut self.out_frame, &self.in_frame);
        }
        // Back to normalised floats; clamp in case the network overshoots.
        for v in self.out_frame.iter_mut() {
            *v = (*v * INV_PCM_SCALE).clamp(-1.0, 1.0);
        }
        self.filled = 0;
        self.frames_processed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_size_is_480() {
        assert_eq!(FRAME_SIZE, 480);
    }

    #[test]
    fn emits_one_frame_per_480_samples_regardless_of_slice_sizes() {
        let mut d = Denoiser::new(false);
        let input = vec![0.01f32; 480 * 5 + 17];
        let mut frames = 0;
        let mut total = 0;
        for chunk in input.chunks(97) {
            d.push(chunk, &mut |f| {
                frames += 1;
                total += f.len();
                assert_eq!(f.len(), FRAME_SIZE);
            });
        }
        assert_eq!(frames, 5);
        assert_eq!(total, 480 * 5);
        assert_eq!(d.pending(), 17);
        let flushed = d.flush(&mut |f| total += f.len());
        assert_eq!(flushed, 17);
        assert_eq!(total, input.len());
        assert_eq!(d.pending(), 0);
    }

    #[test]
    fn white_noise_is_attenuated() {
        // Deterministic LCG noise so the test is reproducible.
        let mut seed: u32 = 0x1234_5678;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32) * 0.2 - 0.1
        };
        let input: Vec<f32> = (0..480 * 100).map(|_| next()).collect();
        let mut d = Denoiser::new(false);
        let mut out = Vec::with_capacity(input.len());
        d.push(&input, &mut |f| out.extend_from_slice(f));
        assert_eq!(out.len(), input.len());

        // Compare RMS over the second half (after the network has adapted).
        // Pure stationary white noise is not what RNNoise is tuned for, so we
        // only assert a clear, monotone reduction (the default model gives
        // roughly -2..-3 dB here); real-world speech+noise does far better.
        let half = input.len() / 2;
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        let rin = rms(&input[half..]);
        let rout = rms(&out[half..]);
        assert!(rout < rin * 0.9, "noise not attenuated: in {rin} out {rout}");
        assert!(out.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn bypass_is_transparent() {
        let mut d = Denoiser::new(true);
        let input: Vec<f32> = (0..960).map(|i| ((i as f32) * 0.01).sin() * 0.3).collect();
        let mut out = Vec::new();
        d.push(&input, &mut |f| out.extend_from_slice(f));
        assert_eq!(out.len(), input.len());
        for (a, b) in input.iter().zip(&out) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
