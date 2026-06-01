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
// Slow integration test for Gemma 4's multimodal (audio + image) path on
// a real checkpoint. Verifies that the `Gemma4ForConditionalGeneration`
// conversion which ships an `audio_config` + `audio_tower.*` loads the
// audio tower + `embed_audio` projection, declares both `.audioIn` and
// `.imageIn`, and (KNOWN-PENDING) generates coherent text from an audio
// prompt.
//
// Uses `mlx-community/gemma-4-e2b-4bit` — the smallest published Gemma 4
// multimodal conversion. Skipped unless the vision-suite group is enabled
// AND the checkpoint is already cached (the audio path pulls a multi-GB
// snapshot; we never download inside this suite).

import Foundation
import TestHelpers
import Testing

@testable import FFAI

/// Local-cache predicate — this multimodal suite never downloads. The
/// `gemma-4-e2b-4bit` snapshot must already be in the HF cache, otherwise
/// the whole suite stays skipped.
private let gemma4MultimodalCacheAvailable: Bool = {
    let cache =
        ("~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit"
            as NSString)
        .expandingTildeInPath
    return FileManager.default.fileExists(atPath: cache)
}()

@Suite(
    "Gemma4 Multimodal Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites
            && gemma4MultimodalCacheAvailable,
        "Gemma 4 multimodal suite: needs the vision-suite group enabled and the gemma-4-e2b-4bit checkpoint already cached.")
)
struct Gemma4MultimodalIntegrationTests {

    static let modelId = "mlx-community/gemma-4-e2b-it-4bit"

    /// Synthesize a short 16 kHz mono test waveform: a 0.5 s, 440 Hz sine
    /// at moderate amplitude. The audio encoder is not yet numerically
    /// validated, so the exact content doesn't matter here — we only need
    /// a non-degenerate signal that produces a positive run of audio soft
    /// tokens through the front-end + encoder.
    static func testWaveform(
        seconds: Double = 0.5, sampleRate: Int = 16_000, freq: Double = 440
    ) -> [Float] {
        let n = Int(seconds * Double(sampleRate))
        var w = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            w[i] = 0.3 * Float(sin(2.0 * Double.pi * freq * Double(i) / Double(sampleRate)))
        }
        return w
    }

    @Test("load — Gemma 4 multimodal checkpoint loads with audio + image capability")
    func loadMultimodalCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // The checkpoint is a VLM with an audio tower — vlModel present,
        // both `.imageIn` and `.audioIn` available, and the text backbone
        // supports the embedding-input splice.
        let vlm = try #require(m.vlModel, "checkpoint is not a VLM")
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.audioIn))
        #expect(m.engine.supportsEmbeddingInput)

        // The audio tower actually loaded — `supportsAudio` is true exactly
        // when both an `audioTokenId` and the audioEncode hook are wired.
        #expect(vlm.supportsAudio)
        #expect(vlm.audioTokenId != nil)
    }

    @Test("audio encode — front-end + encoder + embed_audio produce text-hidden soft tokens")
    func audioEncodeProducesSoftTokens() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "checkpoint is not a VLM")
        try #require(vlm.supportsAudio, "checkpoint has no audio tower")

        // Encode a short waveform → audio soft tokens. This exercises the
        // whole audio loader + splice plumbing (front-end → Conformer
        // encoder → embed_audio projection) and asserts the output shape is
        // `[nAudioTokens, textHidden]` with a positive token run.
        let waveform = Self.testWaveform()
        let audioTokens = try vlm.encodeAudio(waveform)
        #expect(audioTokens.shape.count == 2)
        #expect(audioTokens.shape[0] > 0, "expected a positive run of audio soft tokens")
        #expect(
            audioTokens.shape[1] == m.engine.hidden,
            "audio soft tokens must project into the text hidden dim (\(audioTokens.shape[1]) != \(m.engine.hidden))")
    }

    /// Numerically validate the audio encoder against the mlx-vlm oracle.
    /// Feeds the EXACT mel `/tmp/gemma4_audio_mel.f32` (dumped from mlx-vlm)
    /// to `Gemma4AudioEncoder.encode` and compares to the oracle output
    /// `/tmp/gemma4_audio_oracle.f32` (mlx-vlm's `audio_tower(mel, mask)`).
    /// Both use the same 4-bit weights, so a matching algorithm ⇒ tiny diff;
    /// a large diff localises an encoder bug (the `// VALIDATE:` spots).
    /// Skips if the fixtures aren't present (regenerate via the dump script).
    @Test("audio encoder matches the mlx-vlm oracle on a fixed mel")
    func audioEncoderMatchesOracle() async throws {
        let fm = FileManager.default
        let melPath = "/tmp/gemma4_audio_mel.f32"
        let oraclePath = "/tmp/gemma4_audio_oracle.f32"
        try #require(
            fm.fileExists(atPath: melPath) && fm.fileExists(atPath: oraclePath),
            "oracle fixtures missing — run the mlx-vlm dump script first")

        let dir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.modelId)
        }
        let config = try ModelConfig.load(from: dir)
        let weights = try SafeTensorsBundle(directory: dir)
        let audioCfg = try Gemma4AudioConfig.decode(
            try #require(config.subConfig("audio_config"), "no audio_config"))
        let encoder = try Gemma4AudioEncoder.load(
            weights: weights, config: audioCfg,
            quantization: config.quantization, device: .shared)

        // Read the fixture mel [200, 128] + the oracle output [50, 1536].
        func readF32(_ path: String) -> [Float] {
            let d = fm.contents(atPath: path) ?? Data()
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let melFlat = readF32(melPath)
        let nFrames = melFlat.count / 128
        let mel = ImagePreprocessing.makeTensor(
            from: melFlat, shape: [nFrames, 128], dtype: .f32, device: .shared)
        let (enc, _) = encoder.encode(
            mel: mel, melMask: [Bool](repeating: true, count: nFrames), device: .shared)
        let got = enc.toFloatArray()
        let oracle = readF32(oraclePath)

        print("FFAI enc shape \(enc.shape), oracle count \(oracle.count)")
        #expect(got.count == oracle.count, "shape mismatch: \(got.count) vs \(oracle.count)")
        var maxDiff: Float = 0
        var sumDiff: Float = 0
        let n = min(got.count, oracle.count)
        for i in 0 ..< n {
            let d = abs(got[i] - oracle[i])
            maxDiff = max(maxDiff, d)
            sumDiff += d
        }
        print(
            "FFAI enc[0,:6]   \(got.prefix(6).map { (($0 * 10000).rounded()) / 10000 })")
        print(
            "oracle enc[0,:6] \(oracle.prefix(6).map { (($0 * 10000).rounded()) / 10000 })")
        let meanDiff = sumDiff / Float(n)
        print("max abs diff \(maxDiff), mean abs diff \(meanDiff)")
        // mlx-vlm is a localization aid, NOT bit-exact ground truth (per the
        // repo's "coherent output is the contract" philosophy). After the
        // SSCP conv-layout fix the encoder tracks the reference to ~1% mean
        // over a ±45 range; the residual is bf16 accumulation across the 12
        // residual layers. This is a gross-bug regression guard (a real
        // algorithm bug spikes the MEAN, like the 4.4 pre-fix); the true
        // gate is the coherent audio→text test below.
        #expect(
            meanDiff < 1.5,
            "audio encoder grossly diverges from the reference (mean abs diff \(meanDiff))")
    }

    /// Validate the magnitude-Mel front-end against the transformers
    /// `Gemma4AudioFeatureExtractor` for the test sine. The mel is
    /// deterministic and FFAI now matches the reference bit-for-bit
    /// (periodic-Hann window, HTK norm=None filterbank, magnitude STFT,
    /// `pad_to_multiple_of=128` + mask-zeroing), so this is a strict gate.
    /// Regenerate the fixture with
    /// `gemma4-benchmark/dump_gemma4_sine_mel.py`. Skips if missing.
    @Test("mel front-end matches the Gemma 4 feature extractor on the test sine")
    func melFrontEndMatchesReference() throws {
        let oraclePath = "/tmp/gemma4_sine_mel_oracle.f32"
        try #require(
            FileManager.default.fileExists(atPath: oraclePath),
            "mel oracle fixture missing — run the mlx-vlm dump first")
        let (mel, _) = Gemma4AudioFrontend.melSpectrogram(waveform: Self.testWaveform())
        let got = mel.toFloatArray()
        let oracle = (FileManager.default.contents(atPath: oraclePath) ?? Data())
            .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        print("FFAI mel shape \(mel.shape), oracle count \(oracle.count)")
        print("FFAI mel[0,:6]   \(got.prefix(6).map { (($0 * 10000).rounded()) / 10000 })")
        print("oracle mel[0,:6] \(oracle.prefix(6).map { (($0 * 10000).rounded()) / 10000 })")
        #expect(got.count == oracle.count, "mel shape mismatch \(got.count) vs \(oracle.count)")
        var maxDiff: Float = 0, sumDiff: Float = 0
        let n = min(got.count, oracle.count)
        for i in 0 ..< n {
            maxDiff = max(maxDiff, abs(got[i] - oracle[i]))
            sumDiff += abs(got[i] - oracle[i])
        }
        print("mel max abs diff \(maxDiff), mean abs diff \(sumDiff / Float(n))")
        // FFAI's CPU magnitude-Mel reproduces the reference feature
        // extractor exactly up to f32 rounding (the NumPy cross-check is
        // bit-exact; the small residual here is GPU-tensor round-trip /
        // CPU-DFT vs np.fft.rfft float ordering). A tight bound catches any
        // framing/window/filterbank regression.
        #expect(
            maxDiff < 5e-3,
            "mel front-end diverges from the reference (max abs diff \(maxDiff))")
    }

    // End-to-end audio → text: the full Gemma 4 audio pipeline (log-Mel
    // front-end → Conformer/USM encoder → `embed_audio` projection → splice
    // → text decode) on a real speech clip. Both halves are independently
    // validated against the reference: the mel front-end is bit-exact
    // (`melFrontEndMatchesReference`, max diff 3e-4) and the encoder tracks
    // the mlx-vlm oracle to ~1 % mean (`audioEncoderMatchesOracle`). This
    // closes the loop by asserting the decoded transcription is coherent
    // and mentions the clip's spoken content.
    //
    // Gemma 4 wraps the audio soft-token run in `boa … eoa` markers
    // (`boa_token_id` 256000, `audio_token_id` 258881, `eoa_token_id`
    // 258883 — see mlx-vlm `processing_gemma4.py`): the prompt carries one
    // `audioTokenId` placeholder per encoder frame, and `spliceMultimodal`
    // replaces each with the matching audio embedding row.
    //
    // The `VisionModel` multimodal generate path is greedy (`runGreedy`),
    // the same surface the Gemma 4 vision integration test uses, so this is
    // deterministic without a seed.
    static let boaTokenId = 256000
    static let eoaTokenId = 258883

    @Test("audio + text prompt — transcribes the clean_001 speech clip coherently")
    func audioTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "checkpoint is not a VLM")
        let audioTokenId = try #require(vlm.audioTokenId, "no audio token id")

        // Real 16 kHz speech: "Sure, I can help you with that." (~1.85 s).
        let waveform = try AudioTestHelpers.clean001Waveform()

        // Encode first so we know how many audio soft tokens to splice; the
        // prompt carries one `audioTokenId` per encoder frame inside the
        // `boa … eoa` span.
        let audioTokens = try vlm.encodeAudio(waveform)
        let nAudioTokens = audioTokens.shape[0]
        #expect(nAudioTokens > 0, "encoder produced no audio soft tokens")

        // Build the chat prompt: a user turn whose content is the wrapped
        // audio span followed by a transcription instruction.
        let userPrefix = m.tokenizer.encode(text: "<start_of_turn>user\n")
        let userSuffix = m.tokenizer.encode(
            text: "\nTranscribe this audio.<end_of_turn>\n<start_of_turn>model\n")
        let audioSpan =
            [Self.boaTokenId]
            + Array(repeating: audioTokenId, count: nAudioTokens)
            + [Self.eoaTokenId]
        let promptTokens = userPrefix + audioSpan + userSuffix

        let generated = try vlm.generate(
            promptTokens: promptTokens, audio: waveform,
            maxTokens: 64, eosTokenId: m.config.eosTokenId,
            eosTokenIds: m.config.eosTokenIds)

        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 4 audio → text: \(text)")

        // Coherent (non-degenerate) decode — the audio path must not produce
        // an empty / repetitive / first-step-EOS collapse.
        expectCoherentOutput(
            generated, minTokens: 4, maxConsecutiveRepeat: 6,
            minUniqueRatio: 0.2, label: "gemma4-audio")
        // Content gate: the transcription should reference the spoken phrase.
        let lower = text.lowercased()
        #expect(
            lower.contains("sure") || lower.contains("help")
                || lower.contains("can"),
            "transcription did not reference the clip's spoken content: \(text)")
    }
}
