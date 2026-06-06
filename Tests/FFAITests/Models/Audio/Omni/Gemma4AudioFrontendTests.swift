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
// Gemma4AudioFrontend — structural contract for the audio → log-Mel
// front-end. A 1 s 16 kHz tone must produce a `[~100, 128]` log-Mel with
// finite values and a mask whose length matches the frame count.

import Foundation
import Metal
import Testing

@testable import FFAI

@Suite("Gemma4 audio front-end")
struct Gemma4AudioFrontendTests {

    /// 1 s of a 440 Hz sine at 16 kHz, amplitude 0.5 — via the shared
    /// production generator (`MediaUtils.sineWave`).
    private func sineWaveform(seconds: Double = 1.0, hz: Double = 440)
        -> [Float]
    {
        MediaUtils.sineWave(
            frequency: hz, seconds: seconds, sampleRate: 16_000, amplitude: 0.5)
    }

    @Test("1 s sine → ~100 frames × 128 mels, finite, mask length matches")
    func shapeAndFiniteness() {
        let waveform = sineWaveform()
        let (mel, mask) = Gemma4AudioFrontend.melSpectrogram(waveform: waveform)

        // ── Shape: [nFrames, 128]. 1 s @ 10 ms hop ≈ 100 frames (the
        // semicausal left-pad + 20 ms window shift it by a frame or two).
        #expect(mel.shape.count == 2)
        #expect(mel.shape[1] == 128)
        let nFrames = mel.shape[0]
        #expect(nFrames >= 95 && nFrames <= 105)

        // ── Mask length tracks the frame count exactly.
        #expect(mask.count == nFrames)

        // ── A 1 s real tone with only the semicausal left-pad leaves the
        // vast majority of frames valid; assert at least the bulk are.
        let validCount = mask.filter { $0 }.count
        #expect(validCount >= nFrames - 2)

        // ── Every log-Mel value is finite (no NaN / Inf from log(0)).
        let vals = mel.toFloatArray()
        #expect(vals.count == nFrames * 128)
        #expect(vals.allSatisfy { $0.isFinite })
    }

    @Test("frame count scales with waveform length")
    func framesScaleWithLength() {
        let half = Gemma4AudioFrontend.melSpectrogram(
            waveform: sineWaveform(seconds: 0.5)).mel.shape[0]
        let full = Gemma4AudioFrontend.melSpectrogram(
            waveform: sineWaveform(seconds: 1.0)).mel.shape[0]
        // ~50 frames vs ~100 frames — roughly doubles.
        #expect(full > half)
        #expect(half >= 45 && half <= 55)
    }

    @Test("HTK norm=None filterbank differs from Slaney")
    func filterbankNormDiffers() {
        let cfg = AudioFrontEndConfig.gemma4
        let slaney = MediaUtils.melFilterbank(cfg, norm: .slaney)
        let none = MediaUtils.melFilterbank(cfg, norm: .none)
        #expect(slaney.count == none.count)
        #expect(none.count == cfg.nMels * cfg.nFreq)
        // norm=None leaves unit-peak triangles; Slaney rescales each by
        // 2/(f_hi - f_lo), so the two banks must not be identical.
        #expect(zip(slaney, none).contains { abs($0 - $1) > 1e-6 })
    }
}
