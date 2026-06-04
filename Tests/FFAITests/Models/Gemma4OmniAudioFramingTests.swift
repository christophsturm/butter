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
// Encoder-free audio framing for Gemma 4 Omni (`Gemma4OmniAudio.frame`):
// raw 16 kHz samples reshaped into [nFrames, 640] (40 ms/token), right
// zero-padded, capped at 750 frames (30 s). No mel / STFT. Pure CPU.

import Testing

@testable import FFAI

@Suite("Gemma4 Omni Audio Framing")
struct Gemma4OmniAudioFramingTests {

    private static let spt = 640  // samples per token (40 ms @ 16 kHz)

    private func frameShape(_ count: Int) -> [Int] {
        let wav = [Float](repeating: 0.1, count: count)
        return Gemma4OmniAudio.frame(
            wav, samplesPerToken: Self.spt, dtype: .f32, device: .shared
        ).shape
    }

    @Test("frame count = ceil(N / 640), always ≥ 1")
    func frameCounts() {
        #expect(frameShape(0) == [1, 640])  // empty → one zero frame
        #expect(frameShape(1) == [1, 640])
        #expect(frameShape(300) == [1, 640])  // sub-frame → one frame
        #expect(frameShape(640) == [1, 640])  // exact
        #expect(frameShape(641) == [2, 640])
        #expect(frameShape(1280) == [2, 640])
        #expect(frameShape(1281) == [3, 640])
        #expect(frameShape(16_000) == [25, 640])  // 1 s → 25 frames
    }

    @Test("30 s cap — never more than 750 frames")
    func capAt750() {
        #expect(frameShape(750 * 640) == [750, 640])  // exactly 30 s
        #expect(frameShape(750 * 640 + 1) == [750, 640])  // truncated
        #expect(frameShape(2_000_000) == [750, 640])  // way over → capped
    }

    @Test("right zero-pad of the final partial frame; samples preserved in order")
    func zeroPadAndOrder() {
        // 700 samples → 2 frames; the first 700 values are the input, the
        // trailing 1280-700 = 580 are zero pad.
        var wav = [Float](repeating: 0, count: 700)
        for i in 0 ..< 700 { wav[i] = Float(i) + 1 }  // 1...700, all non-zero
        let t = Gemma4OmniAudio.frame(
            wav, samplesPerToken: Self.spt, dtype: .f32, device: .shared)
        #expect(t.shape == [2, 640])
        let flat = t.toFloatArray()
        #expect(flat.count == 1280)
        // Input preserved in raster order.
        for i in 0 ..< 700 { #expect(flat[i] == Float(i) + 1) }
        // Tail zero-padded.
        for i in 700 ..< 1280 { #expect(flat[i] == 0) }
    }
}
