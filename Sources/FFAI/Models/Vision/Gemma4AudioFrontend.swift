// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Gemma 4 audio → log-Mel front-end. Turns a raw 16 kHz mono waveform
// into the `[nFrames, 128]` log-Mel spectrogram the Gemma 4 audio tower
// (`Gemma4Audio`) consumes, plus a per-frame validity mask.
//
// This reuses FFAI's existing GPU Mel path (`Ops.melSpectrogram`, wrapped
// by `AudioPreprocessing.logMelSpectrogram`) rather than re-deriving an
// STFT / Mel projection. Gemma 4 differs from Whisper in three ways, all
// parameterised here rather than mutating the Whisper path:
//
//   * 128 Mel bins, 20 ms frame (320 samples), 10 ms hop (160 samples),
//     FFT length 512 (next pow2 ≥ 320), `fMax = 8000 Hz` — captured by
//     `AudioFrontEndConfig.gemma4`.
//   * `norm=None` HTK Mel filterbank (unit-peak triangles, not Slaney) —
//     `MelNorm.none`.
//   * `log(mel + mel_floor)` natural-log compression with no dynamic-range
//     clamp / affine normalisation — so `whisperNormalize: false` and the
//     `mel_floor` is folded into the kernel's `logEps`.
//
// Reference: `transformers` `Gemma4AudioFeatureExtractor`
// (`feature_extraction_gemma4.py`), `_extract_spectrogram`.

import Foundation
import Metal

enum Gemma4AudioFrontend {

    /// Frame length in samples (20 ms @ 16 kHz). Reference
    /// `frame_length = round(16000 * 20 / 1000)`.
    static let frameLength = 320
    /// Hop length in samples (10 ms @ 16 kHz). Reference
    /// `hop_length = round(16000 * 10 / 1000)`.
    static let hopLength = 160
    /// FFT length — next power of two ≥ `frameLength` (`2^ceil(log2 320)`
    /// = 512; `fft_overdrive` is off so no ×2). Reference `fft_length`.
    static let fftLength = 512
    /// Mel filterbank bins. Reference `feature_size = 128`.
    static let nMels = 128
    /// `log(mel + mel_floor)` floor that avoids `log(0)`. Reference
    /// `mel_floor = 1e-3`.
    static let melFloor: Float = 1e-3
    /// Semicausal left pad: prepend `frame_length // 2` zeros so the first
    /// STFT frame is centred at t=0 (reference `pad_left`,
    /// `time_padding='semicausal'`).
    static let padLeft = 160

    /// Raw mono waveform (16 kHz `Float` samples) → log-Mel `[nFrames, 128]`
    /// plus a per-frame validity mask (`true` = valid, `false` = padding).
    ///
    /// The mask follows the reference: a frame is valid only when every
    /// sample in its analysis window is real (non-padded) audio. With pure
    /// trailing/leading silence absent, every frame whose window end index
    /// lands inside the original (pre-pad) waveform is valid.
    static func melSpectrogram(
        waveform: [Float], device: Device = .shared
    ) -> (mel: Tensor, mask: [Bool]) {
        let cfg = AudioFrontEndConfig.gemma4
        let nSamples = waveform.count

        // ── Semicausal pad: prepend `padLeft` zeros, no right reflection.
        // (Reference pads only on the left; the right side is handled by
        // the kernel reading zero-padded tail samples that the window —
        // zero beyond `frameLength` — multiplies out.)
        var padded = [Float](repeating: 0, count: padLeft)
        padded.append(contentsOf: waveform)

        // ── Frame count, mirroring the reference unfold. The reference
        // unfolds with `frame_size_for_unfold = frame_length + 1 = 321`
        // (then drops the last sample since preemphasis == 0), stepping by
        // `hop_length`. So `nFrames = (paddedLen - 321) / hop + 1`.
        let unfoldSize = frameLength + 1
        let nFrames =
            padded.count >= unfoldSize
            ? (padded.count - unfoldSize) / hopLength + 1
            : 0
        precondition(
            nFrames > 0,
            "Gemma4AudioFrontend.melSpectrogram: waveform too short "
                + "(\(nSamples) samples → 0 frames)")

        // ── Right-pad the audio buffer so the GPU kernel's frame walk
        // (frame i reads `audio[i*hop ..< i*hop + nFFT]`, nFFT = 512) stays
        // in-bounds. These tail samples fall in the window's zero region
        // (Hann over first `frameLength`, zeros after) so they contribute
        // nothing — matching the reference's implicit `rfft(n=fft_length)`
        // right-pad of each 320-sample windowed frame to 512.
        let kernelSpan = (nFrames - 1) * hopLength + fftLength
        if padded.count < kernelSpan {
            padded.append(
                contentsOf: [Float](
                    repeating: 0, count: kernelSpan - padded.count))
        }

        // ── Length-512 window: periodic Hann over the first `frameLength`
        // samples, zeros for the remaining `fftLength - frameLength`. This
        // realises "window the 320-sample frame, then RFFT with n=512".
        // VALIDATE: reference uses `window_function(frame_length)` which is
        // a periodic Hann of length 320 (divisor 320, matching
        // `AudioPreprocessing.hannWindow`); the zero tail to 512 is the
        // implicit RFFT zero-pad.
        var window = AudioPreprocessing.hannWindow(frameLength)
        window.append(
            contentsOf: [Float](repeating: 0, count: fftLength - frameLength))
        let winT = Tensor.empty(shape: [fftLength], dtype: .f32, device: device)
        AudioPreprocessing.copyFloats(window, into: winT)

        // ── HTK, norm=None Mel filterbank (Gemma 4 / HF).
        let bank = AudioPreprocessing.melFilterbank(cfg, norm: .none)
        let melT = Tensor.empty(
            shape: [cfg.nMels, cfg.nFreq], dtype: .f32, device: device)
        AudioPreprocessing.copyFloats(bank, into: melT)

        // ── Upload the padded waveform.
        let audioT = Tensor.empty(
            shape: [padded.count], dtype: .f32, device: device)
        AudioPreprocessing.copyFloats(padded, into: audioT)

        // ── Fused STFT + Mel + log on the GPU. We dispatch
        // `Ops.melSpectrogram` directly rather than via
        // `AudioPreprocessing.logMelSpectrogram`, because that helper
        // hardcodes Whisper's symmetric reflect-pad (`nFFT/2` each side);
        // Gemma 4's framing is semicausal (left-pad only, done above) so
        // the waveform is already framed for the kernel here.
        //
        // The kernel's `logEps` is added inside the log — set it to
        // `mel_floor` so the kernel emits exactly `log(mel + mel_floor)`,
        // matching the reference's `np.log(mel_spec + self.mel_floor)`.
        //
        // VALIDATE: the `mel_spectrogram` kernel projects the *power*
        // spectrum (`re² + im²`) through the filterbank, then takes
        // `log(mel_power + logEps)`. The Gemma 4 reference instead projects
        // the *magnitude* spectrum (`np.abs(stft) = sqrt(power)`) through
        // the filterbank, i.e. `log(magnitude @ mel + mel_floor)`. FFAI's
        // shipped Mel kernels are power-spectrum only (`mel_spectrogram.rs`
        // / `mel_filterbank.rs` both accumulate `re² + im²`), so this
        // front-end emits a power-Mel log. The shape / framing / Mel
        // geometry / mask match the reference exactly; the per-bin values
        // differ by the magnitude-vs-power square. When a magnitude-Mel
        // kernel lands (sqrt before the filterbank projection), swap it in
        // here — the rest of this function is unchanged.
        let cmd = device.makeCommandBuffer()
        let mel = Ops.melSpectrogram(
            audio: audioT, window: winT, melWeight: melT,
            nFFT: fftLength, nMels: nMels, hopLength: hopLength,
            nFrames: nFrames, logEps: melFloor, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── Per-frame validity mask. Reference: a frame is valid when the
        // last sample of its analysis window is real audio. The window for
        // frame i spans `[i*hop, i*hop + unfoldSize - 1]` in the
        // left-padded signal; the sample is real when its index lies in
        // `[padLeft, padLeft + nSamples)`.
        var mask = [Bool](repeating: false, count: nFrames)
        for i in 0 ..< nFrames {
            let frameEnd = i * hopLength + unfoldSize - 1
            mask[i] = frameEnd >= padLeft && frameEnd < padLeft + nSamples
        }

        return (mel, mask)
    }
}
