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
    /// The reference `__call__` pads the raw waveform length up to a
    /// multiple of this before framing (`pad_to_multiple_of=128`, "optimal
    /// TPU support"). Those trailing samples are zeros, marked invalid by
    /// the frame mask and zeroed out of the final mel — but they shift the
    /// frame count (and therefore the post-SSCP soft-token count) by up to
    /// one frame, so FFAI must replicate the padding to match the reference
    /// token geometry.
    static let padToMultiple = 128

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

        // ── `pad_to_multiple_of=128`: the reference `__call__` right-pads
        // the raw waveform with silence up to a multiple of 128 samples
        // BEFORE any framing. These samples are real zeros (not reflected),
        // get marked invalid by the frame mask, and are zeroed out of the
        // final mel — but they grow the frame count by up to one frame, so
        // FFAI replicates the padding to keep the soft-token geometry
        // aligned with the reference. `[0, nSamples)` stays the real region.
        let paddedSamples =
            ((nSamples + padToMultiple - 1) / padToMultiple) * padToMultiple

        // ── Semicausal pad: prepend `padLeft` zeros, no right reflection.
        // (Reference pads only on the left; the right side is handled by
        // the kernel reading zero-padded tail samples that the window —
        // zero beyond `frameLength` — multiplies out.)
        var padded = [Float](repeating: 0, count: padLeft)
        padded.append(contentsOf: waveform)
        // Right-pad the real audio up to the 128-multiple length.
        if paddedSamples > nSamples {
            padded.append(
                contentsOf: [Float](
                    repeating: 0, count: paddedSamples - nSamples))
        }

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

        // ── HTK, norm=None Mel filterbank (Gemma 4 / HF). `bank` is
        // `[nMels, nFreq]` row-major (filter weight per (mel_bin, freq)).
        let bank = AudioPreprocessing.melFilterbank(cfg, norm: .none)

        // ── STFT → MAGNITUDE Mel → log, on the CPU (per-frame parallel).
        //
        // FFAI's fused `mel_spectrogram` GPU kernel projects the *power*
        // spectrum (`re² + im²`) through the filterbank; the Gemma 4
        // reference projects the *magnitude* (`np.abs(stft) = sqrt(power)`),
        // i.e. `log(magnitude @ mel + mel_floor)`. There is no magnitude-Mel
        // kernel, so the magnitude DFT runs here on the CPU — a direct port
        // of the kernel's per-(frame,freq) DFT (`re = Σ_t win[t]·x[f·hop+t]·
        // cos(-2π·k·t/nFFT)`, `im = Σ … sin`) with a `sqrt` before the
        // filterbank projection. The front-end runs once per clip (not a
        // hot path), so the O(nFrames·nFreq·nFFT) CPU DFT is fine; if a
        // magnitude-Mel kernel ever lands, swap it back in here.
        //
        // The accumulation runs in `Double`: the reference (`np.fft.rfft`)
        // is an FFT, not a naive DFT, so it carries far less summation
        // error. The sine's energy is concentrated near one bin, leaving
        // most bins at ~`log(mel_floor)` where `log` amplifies tiny
        // magnitude differences — single-precision accumulation there drifts
        // ~0.1; `Double` brings the whole spectrum to f32-rounding agreement.
        let nFreq = cfg.nFreq
        let nFFT = fftLength
        let angStep = -2.0 * Double.pi / Double(nFFT)
        let win = window  // length-nFFT window (captured by value)
        let pad = padded
        var melFlat = [Float](repeating: 0, count: nFrames * nMels)
        melFlat.withUnsafeMutableBufferPointer { out in
            let outBase = out.baseAddress!
            // Each frame writes a disjoint `[f*nMels, (f+1)*nMels)` slice.
            let outAddr = UInt(bitPattern: outBase)
            DispatchQueue.concurrentPerform(iterations: nFrames) { f in
                let dst = UnsafeMutablePointer<Float>(bitPattern: outAddr)!
                let base = f * hopLength
                var mag = [Double](repeating: 0, count: nFreq)
                for k in 0 ..< nFreq {
                    var re: Double = 0
                    var im: Double = 0
                    let a = angStep * Double(k)
                    for t in 0 ..< nFFT {
                        let s = Double(pad[base + t]) * Double(win[t])
                        re += s * cos(a * Double(t))
                        im += s * sin(a * Double(t))
                    }
                    mag[k] = (re * re + im * im).squareRoot()
                }
                for m in 0 ..< nMels {
                    var acc: Double = 0
                    let bb = m * nFreq
                    for k in 0 ..< nFreq { acc += Double(bank[bb + k]) * mag[k] }
                    dst[f * nMels + m] = Float(Foundation.log(acc + Double(melFloor)))
                }
            }
        }
        _ = winT  // (window tensor no longer needed for the CPU path)

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

        // ── Zero the masked-out (padding) frames, matching the reference's
        // `prepared_speech = speech * mask[..., None]`. The Conformer's SSCP
        // conv reads raw mel values, so the tail frames must be hard zeros
        // (not `log(mel_floor)`) for the boundary activations to match.
        for i in 0 ..< nFrames where !mask[i] {
            let base = i * nMels
            for m in 0 ..< nMels { melFlat[base + m] = 0 }
        }

        let mel = ImagePreprocessing.makeTensor(
            from: melFlat, shape: [nFrames, nMels], dtype: .f32, device: device)

        return (mel, mask)
    }
}
