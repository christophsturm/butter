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
// Integration test: loads a real Kokoro-82M checkpoint from the HF
// cache and exercises the GPU iSTFTNet vocoder tail. A load failure
// FAILS the suite — `loadKokoro()` is `throws` and the checkpoint is a
// hard requirement, not a "skip if missing".
//
// Kokoro-82M is the published checkpoint; FFAI's Phase 7 contribution
// is the iSTFTNet vocoder (Ops.vocoderISTFT). This suite verifies the
// model loads, the vocoder constructs from the checkpoint config, and a
// synthesized waveform is non-degenerate (finite, non-silent).
//
// KokoroModel.load is config-driven — the vocoder geometry comes from
// `config.json`'s `istftnet` block and no safetensors weights are read
// (the StyleTTS2 acoustic stack is a separate port). The 4-bit MLX
// conversion is the smallest published Kokoro variant and carries the
// canonical config.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Kokoro Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct KokoroIntegrationTests {

    /// Canonical HF repo id. The 4-bit MLX conversion is the smallest
    /// published Kokoro variant.
    private static let repoId = "mlx-community/Kokoro-82M-4bit"

    /// Load Kokoro from the HF cache. Throws on failure so a missing
    /// checkpoint fails the test instead of skipping it.
    private func loadKokoro() async throws -> KokoroModel {
        let dir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
        return try KokoroModel.load(directory: dir)
    }

    @Test("load — Kokoro config binds the iSTFTNet vocoder")
    func loadKokoro_bindsVocoder() async throws {
        let model = try await loadKokoro()
        // Kokoro's iSTFTNet head uses a tiny FFT (20) with hop 5.
        #expect(model.vocoder.nFFT > 0)
        #expect(model.vocoder.hopLength > 0)
        #expect(model.config.sampleRate == 24_000)
        // The phoneme vocabulary should have loaded from config.json.
        #expect(!model.phonemeVocab.isEmpty)
    }

    @Test("synthesize — vocoder produces a non-degenerate waveform")
    func synthesize_nonDegenerateWaveform() async throws {
        let model = try await loadKokoro()
        // A predicted complex spectrogram (the acoustic decoder's
        // output) — frequency-sweep content so the reconstruction is
        // a real, non-constant utterance-length waveform.
        let nFrames = 200  // ~ a short utterance at hop 5
        let nFreq = model.vocoder.nFFT / 2 + 1
        var re = [Float](repeating: 0, count: nFrames * nFreq)
        var im = [Float](repeating: 0, count: nFrames * nFreq)
        for f in 0 ..< nFrames {
            for k in 0 ..< nFreq {
                let phase = 2.0 * Float.pi * Float(k) * Float(f) / Float(nFrames)
                re[f * nFreq + k] = 0.4 * cos(phase)
                im[f * nFreq + k] = 0.4 * sin(phase)
            }
        }
        let reT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
        reT.copyIn(from: re)
        let imT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
        imT.copyIn(from: im)

        let waveform = model.synthesizeFromSpectrogram(specRe: reT, specIm: imT)
        let expectedLen =
            (nFrames - 1) * model.vocoder.hopLength
            + model.vocoder.nFFT
        #expect(waveform.shape == [expectedLen])

        let samples = waveform.toFloatArray()
        // Non-degenerate: finite, non-silent, not a constant.
        #expect(samples.allSatisfy { $0.isFinite })
        let energy = samples.map { $0 * $0 }.reduce(0, +)
        #expect(energy > 1e-4, "Kokoro vocoder produced a silent waveform")
        let distinct = Set(samples.map { ($0 * 1000).rounded() }).count
        #expect(distinct > 10, "Kokoro vocoder produced a constant waveform")
        print(
            "Kokoro synthesized \(expectedLen) samples, "
                + "energy=\(energy), distinct=\(distinct)")
    }

    @Test("synthesize — full acoustic chain produces non-degenerate speech")
    func synthesize_fullChain() async throws {
        // Full-precision acoustic weights (the 4-bit pack quantizes them).
        let dir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: "mlx-community/Kokoro-82M-bf16")
        }
        let model = try KokoroModel.load(directory: dir)
        // A short IPA phoneme string mapped through the checkpoint vocab.
        let ids = model.phonemeIds(for: "hɛlˈoʊ wˈɜːld")
        try #require(!ids.isEmpty, "no phonemes mapped — vocab issue")
        let waveform = try model.synthesize(phonemeIds: ids)
        let samples = waveform.toFloatArray()
        #expect(samples.count > 1000, "waveform too short: \(samples.count)")
        #expect(samples.allSatisfy { $0.isFinite }, "non-finite samples")
        let energy = samples.map { $0 * $0 }.reduce(0, +)
        #expect(energy > 1e-4, "silent waveform")
        let distinct = Set(samples.map { ($0 * 1000).rounded() }).count
        #expect(distinct > 50, "degenerate waveform")
        print("Kokoro full-chain: \(samples.count) samples, energy=\(energy), distinct=\(distinct)")
    }

    @Test("generator — GPU matches the CPU iSTFTNet reference (re/im parity)")
    func generatorGPUParity() async throws {
        let dir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: "mlx-community/Kokoro-82M-bf16")
        }
        let model = try KokoroModel.load(directory: dir)
        let acoustic = try #require(model.acoustic, "no acoustic weights")
        // Synthetic decoder output + F0 at a small frame count so the CPU
        // reference is fast; the generator is shape-generic, so this exercises
        // every conv / adain / snake / leaky path at the real channel widths.
        let t2 = 8
        func ramp(_ n: Int, _ seed: Int, _ amp: Float) -> [Float] {
            (0 ..< n).map { (Float(($0 * 7 + seed) % 23) / 23 - 0.5) * amp }
        }
        let x0 = (0 ..< t2).map { t in ramp(512, t * 3 + 1, 0.6) }  // [T2, 512]
        let f0 = (0 ..< t2).map { Float(120 + ($0 % 5) * 30) }  // voiced curve
        let style = ramp(128, 99, 0.4)

        let (reCPU, imCPU) = acoustic.generate(
            x0, style: style, f0: f0, vocoder: model.vocoder)
        let gw = KokoroGPUWeights(acoustic)
        let (reGPU, imGPU) = acoustic.generateGPU(
            x0, style: style, f0: f0, vocoder: model.vocoder, gw: gw)

        try #require(reGPU.count == reCPU.count, "frame count differs: GPU \(reGPU.count) vs CPU \(reCPU.count)")
        var maxRe: Float = 0, maxIm: Float = 0, denom: Float = 1e-9
        for t in 0 ..< reCPU.count {
            for k in 0 ..< reCPU[t].count {
                maxRe = max(maxRe, abs(reGPU[t][k] - reCPU[t][k]))
                maxIm = max(maxIm, abs(imGPU[t][k] - imCPU[t][k]))
                denom = max(denom, abs(reCPU[t][k]), abs(imCPU[t][k]))
            }
        }
        print("generator GPU↔CPU parity: maxRe=\(maxRe) maxIm=\(maxIm) scale=\(denom)")
        #expect(maxRe / denom < 1e-2, "re diverges (max \(maxRe), scale \(denom))")
        #expect(maxIm / denom < 1e-2, "im diverges (max \(maxIm), scale \(denom))")

        // Full decoder + generator parity: CPU decode() vs GPU decodeGPU().
        // asr is the aligned rate T; F0/N are the 2T predictor curves (the
        // decoder's stride-2 F0_conv/N_conv bring them back to T).
        let asr = (0 ..< t2).map { t in ramp(512, t * 5 + 2, 0.5) }  // [T, 512]
        let f0Dec = (0 ..< 2 * t2).map { Float(110 + ($0 % 6) * 25) }  // [2T]
        let nDec = (0 ..< 2 * t2).map { Float(($0 % 7) - 3) * 3 }  // [2T]
        let (dReCPU, dImCPU) = acoustic.decode(asr: asr, f0: f0Dec, n: nDec, style: style, vocoder: model.vocoder)
        let (dReGPU, dImGPU) = acoustic.decodeGPU(
            asr: asr, f0: f0Dec, n: nDec, style: style, vocoder: model.vocoder, gw: gw)
        try #require(dReGPU.count == dReCPU.count, "decoder frame count differs: GPU \(dReGPU.count) vs CPU \(dReCPU.count)")
        var dRe: Float = 0, dIm: Float = 0, dDen: Float = 1e-9
        for t in 0 ..< dReCPU.count {
            for k in 0 ..< dReCPU[t].count {
                dRe = max(dRe, abs(dReGPU[t][k] - dReCPU[t][k]))
                dIm = max(dIm, abs(dImGPU[t][k] - dImCPU[t][k]))
                dDen = max(dDen, abs(dReCPU[t][k]), abs(dImCPU[t][k]))
            }
        }
        print("decoder GPU↔CPU parity: maxRe=\(dRe) maxIm=\(dIm) scale=\(dDen)")
        #expect(dRe / dDen < 2e-2, "decoder re diverges (max \(dRe), scale \(dDen))")
        #expect(dIm / dDen < 2e-2, "decoder im diverges (max \(dIm), scale \(dDen))")
    }
}
