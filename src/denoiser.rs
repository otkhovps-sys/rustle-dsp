//! Thin wrapper around `nnnoiseless::DenoiseState`.
//!
//! RNNoise works on fixed frames of 480 samples (10 ms at 48 kHz). Callers can
//! push arbitrary-length mono slices; this type gathers them into full frames,
//! runs the network and hands back, per frame:
//!
//! * the denoised frame,
//! * the **dry** (unprocessed) frame *time-aligned* with it, and
//! * the network's voice-activity probability.
//!
//! Delay: RNNoise analyses a 20 ms window (the current and the previous
//! frame) and resynthesises by overlap-add, so the frame it returns
//! corresponds to the *previous* input frame — exactly `FRAME_SIZE` samples
//! of algorithmic delay on top of the 10 ms framing (measured by
//! cross-correlation: 480 samples). The dry signal is delayed by the same
//! amount so that mixing dry/denoised never comb-filters and bypass has the
//! same latency as processing (fair A/B).
//!
//! Scaling: RNNoise was trained on 16-bit PCM, so `nnnoiseless` expects sample
//! values in the i16 range (±32768) even though they are `f32`. CPAL gives us
//! normalised ±1.0 floats, so we scale on the way in and back out.

use nnnoiseless::DenoiseState;

/// 480 samples = 10 ms at 48 kHz.
pub const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;

/// Algorithmic delay of the network output relative to its input (samples).
pub const MODEL_DELAY: usize = FRAME_SIZE;

/// Receives `(denoised, dry_aligned, vad)` for each processed frame.
pub type FrameSink<'a> = dyn FnMut(&[f32], &[f32], f32) + 'a;

const PCM_SCALE: f32 = 32_768.0;
const INV_PCM_SCALE: f32 = 1.0 / PCM_SCALE;

pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    /// Input frame being gathered (±1.0 scale).
    in_frame: [f32; FRAME_SIZE],
    /// Same frame in the i16 range for the network.
    scaled: [f32; FRAME_SIZE],
    out_frame: [f32; FRAME_SIZE],
    /// Dry signal aligned with `out_frame` (= the previous input frame).
    dry_frame: [f32; FRAME_SIZE],
    /// Last complete input frame, waiting to become `dry_frame`.
    held: [f32; FRAME_SIZE],
    /// Number of valid samples currently gathered in `in_frame`.
    filled: usize,
    /// Voice-activity probability reported by the last processed frame.
    last_vad: f32,
    frames_processed: u64,
}

impl Default for Denoiser {
    fn default() -> Self {
        Self::new()
    }
}

impl Denoiser {
    pub fn new() -> Self {
        Self {
            state: DenoiseState::new(),
            in_frame: [0.0; FRAME_SIZE],
            scaled: [0.0; FRAME_SIZE],
            out_frame: [0.0; FRAME_SIZE],
            dry_frame: [0.0; FRAME_SIZE],
            held: [0.0; FRAME_SIZE],
            filled: 0,
            last_vad: 0.0,
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

    /// Feed mono 48 kHz samples in ±1.0. For every complete frame `sink`
    /// receives `(denoised, dry_aligned, vad)`, both slices `FRAME_SIZE` long.
    pub fn push(&mut self, input: &[f32], sink: &mut FrameSink<'_>) {
        let mut rest = input;
        while !rest.is_empty() {
            let room = FRAME_SIZE - self.filled;
            let take = room.min(rest.len());
            self.in_frame[self.filled..self.filled + take].copy_from_slice(&rest[..take]);
            self.filled += take;
            rest = &rest[take..];

            if self.filled == FRAME_SIZE {
                self.run_frame();
                sink(&self.out_frame, &self.dry_frame, self.last_vad);
            }
        }
    }

    /// Shutdown flush: emit everything still inside the denoiser — the
    /// partial frame and the one frame held by the model delay — so the tail
    /// of the stream is not lost. `sink` gets slices shorter than a frame for
    /// the very last chunk. Returns the number of samples emitted.
    pub fn flush(&mut self, sink: &mut FrameSink<'_>) -> usize {
        if self.frames_processed == 0 && self.filled == 0 {
            return 0;
        }
        let partial = self.filled;
        let mut emitted = 0;
        if self.frames_processed > 0 {
            // Zero-pad the partial frame; this releases the held full frame.
            self.in_frame[partial..].fill(0.0);
            self.filled = FRAME_SIZE;
            self.run_frame();
            sink(&self.out_frame, &self.dry_frame, self.last_vad);
            emitted += FRAME_SIZE;
            if partial > 0 {
                // One more (silent) frame releases the real part of the partial one.
                self.in_frame.fill(0.0);
                self.filled = FRAME_SIZE;
                self.run_frame();
                sink(&self.out_frame[..partial], &self.dry_frame[..partial], self.last_vad);
                emitted += partial;
            }
        } else {
            // Only a partial frame was ever received.
            self.in_frame[partial..].fill(0.0);
            self.filled = FRAME_SIZE;
            self.run_frame();
            self.in_frame.fill(0.0);
            self.filled = FRAME_SIZE;
            self.run_frame();
            sink(&self.out_frame[..partial], &self.dry_frame[..partial], self.last_vad);
            emitted += partial;
        }
        emitted
    }

    fn run_frame(&mut self) {
        debug_assert_eq!(self.filled, FRAME_SIZE);
        for (d, s) in self.scaled.iter_mut().zip(&self.in_frame) {
            *d = s * PCM_SCALE;
        }
        self.last_vad = self.state.process_frame(&mut self.out_frame, &self.scaled);
        // Back to normalised floats; clamp in case the network overshoots.
        for v in self.out_frame.iter_mut() {
            *v = (*v * INV_PCM_SCALE).clamp(-1.0, 1.0);
        }
        // The network output lags its input by one frame: the aligned dry
        // signal is the previous input frame.
        self.dry_frame.copy_from_slice(&self.held);
        self.held.copy_from_slice(&self.in_frame);
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
        let mut d = Denoiser::new();
        let input = vec![0.01f32; 480 * 5 + 17];
        let mut frames = 0;
        let mut total = 0;
        for chunk in input.chunks(97) {
            d.push(chunk, &mut |f, dry, _| {
                frames += 1;
                total += f.len();
                assert_eq!(f.len(), FRAME_SIZE);
                assert_eq!(dry.len(), FRAME_SIZE);
            });
        }
        assert_eq!(frames, 5);
        assert_eq!(d.pending(), 17);
        let flushed = d.flush(&mut |f, _, _| total += f.len());
        assert_eq!(flushed, 480 + 17);
        // Output = model delay (leading frame) + every input sample.
        assert_eq!(total, MODEL_DELAY + input.len());
        assert_eq!(d.pending(), 0);
    }

    #[test]
    fn dry_output_is_input_delayed_by_exactly_one_frame() {
        let input: Vec<f32> = (0..480 * 6).map(|i| ((i * 7919) % 1000) as f32 / 2000.0 - 0.25).collect();
        let mut d = Denoiser::new();
        let mut dry_out = Vec::new();
        d.push(&input, &mut |_, dry, _| dry_out.extend_from_slice(dry));
        d.flush(&mut |_, dry, _| dry_out.extend_from_slice(dry));
        assert_eq!(dry_out.len(), input.len() + MODEL_DELAY);
        assert!(dry_out[..MODEL_DELAY].iter().all(|v| *v == 0.0));
        assert_eq!(&dry_out[MODEL_DELAY..], &input[..]);
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
        let mut d = Denoiser::new();
        let mut out = Vec::with_capacity(input.len());
        d.push(&input, &mut |f, _, _| out.extend_from_slice(f));
        assert_eq!(out.len(), input.len());

        // Pure stationary white noise is not what RNNoise is tuned for, so we
        // only assert a clear reduction; real-world speech+noise does far better.
        let half = input.len() / 2;
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        let rin = rms(&input[half..]);
        let rout = rms(&out[half..]);
        assert!(rout < rin * 0.9, "noise not attenuated: in {rin} out {rout}");
        assert!(out.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }
}
