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
// Consolidated integration suite for Gemma 4 VL / multimodal
// (`Gemma4ForConditionalGeneration`, the bespoke Gemma 4 ViT tower +
// USM/Conformer audio tower + multi-modal embedder + Gemma 4 backbone).
// ONE @Suite that loads `gemma-4-e2b-it-4bit` ONCE (static memoized Task)
// and runs a per-capability inference pass — text + image + audio + video —
// modeled on `MiniCPM45OmniIntegrationTests`. Merges the former
// `Gemma4VisionIntegrationTests` + `Gemma4MultimodalIntegrationTests` so the
// shared e2b checkpoint loads a single time instead of once per file/@Test.
//
// The numeric mlx-vlm oracle tests (vision tower / audio encoder / mel
// front-end) are retained here, fixture-gated on `/tmp/*.f32` dumps — they
// skip unless the dumps are present.

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

/// Local-cache predicate — this suite never downloads (the audio + vision
/// paths pull a multi-GB snapshot). The `gemma-4-e2b-it-4bit` snapshot must
/// already be in the HF cache, otherwise the whole suite stays skipped.
private let gemma4CacheAvailable: Bool = {
    let cache =
        ("~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit"
            as NSString)
        .expandingTildeInPath
    return FileManager.default.fileExists(atPath: cache)
}()

@Suite(
    "Gemma 4 Multimodal Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites && gemma4CacheAvailable,
        "Gemma 4 multimodal suite: needs the vision-suite group enabled and the gemma-4-e2b-it-4bit checkpoint already cached.")
)
struct Gemma4IntegrationTests {

    static let modelId = "mlx-community/gemma-4-e2b-it-4bit"

    /// The Gemma 4 VL/multimodal checkpoint, loaded ONCE and shared across
    /// every `@Test`. `loadModel` serializes through the global model-load
    /// lock; tests recover it with `try await Self.loaded.value`.
    static let loaded = Task { try await loadModel(modelId) }

    // Gemma 4 image/audio span markers (the multimodal generate path does
    // not auto-frame; tests build the placeholder runs explicitly).
    static let boiTokenId = 255_999
    static let eoiTokenId = 258_882
    static let boaTokenId = 256_000
    static let eoaTokenId = 258_883

    // MARK: - load

    @Test("load — Gemma 4 loads as a VLM with image + video + audio")
    func load() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "checkpoint is not a VLM")
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.videoIn))
        #expect(m.availableCapabilities.contains(.audioIn))
        #expect(m.textEngine.supportsEmbeddingInput)  // VLM splice prerequisite
        #expect(vlm.imageTokenCount > 0)
        #expect(vlm.videoTokenId != nil)
        // The audio tower loaded — `supportsAudio` is true exactly when both
        // an `audioTokenId` and the audioEncode hook are wired.
        #expect(vlm.supportsAudio)
        #expect(vlm.audioTokenId != nil)
    }

    @Test("enable / disable .imageIn — runtime capability flip")
    func capabilityFlip() async throws {
        let m = try await Self.loaded.value
        #expect(m.availableCapabilities.contains(.imageIn))
        m.disable(.imageIn)
        #expect(!m.isEnabled(.imageIn))
        m.enable(.imageIn)
        #expect(m.isEnabled(.imageIn))
    }

    /// E2B backbone shape + the sliding/global KV-cache geometry. The E2B
    /// `layer_types` make every 5th layer (idx 4, 9, …, 34) full_attention ⇒
    /// 7 global (.unbounded) + 28 sliding (.window). Preserved from the former
    /// `Gemma4TextIntegrationTests` E2B case so this one shared load covers it.
    @Test("backbone — E2B PLE shape + sliding/global cache geometry")
    func backboneShape() async throws {
        let m = try await Self.loaded.value
        // E2B canonical shapes (HF text_config): hidden 1536, 35 layers,
        // 8 heads, sliding head_dim 256.
        #expect(m.textEngine.hidden == 1536)
        #expect(m.textEngine.nLayers == 35)
        #expect(m.textEngine.nHeads == 8)
        #expect(m.textEngine.headDim == 256)
        // Resolved as the PLE (Gemma4E) variant.
        let gemma4 = m.engine as? Gemma4Model
        #expect(gemma4 != nil, "expected Gemma4Model engine")
        #expect(gemma4?.ple != nil, "E2B is a PLE variant — Gemma4PLE must be present")
        // Per-layer cache geometry: 7 global unbounded, 28 sliding windowed.
        let caches = m.textEngine.makeLayerCaches()
        var slidingCount = 0, globalCount = 0
        for (i, c) in caches.enumerated() {
            guard let kv = c as? any KVCacheProtocol else { continue }
            let isGlobal = (i + 1) % 5 == 0
            switch kv.eviction {
            case .window:
                #expect(!isGlobal, "layer \(i): expected global (.unbounded)")
                slidingCount += 1
            case .unbounded:
                #expect(isGlobal, "layer \(i): expected sliding (.window)")
                globalCount += 1
            }
        }
        #expect(globalCount == 7)
        #expect(slidingCount == 28)
    }

    // MARK: - text

    @Test("text — Gemma 4 backbone chat-generates coherently")
    func textGeneration() async throws {
        let m = try await Self.loaded.value
        let r = try await m.generate(
            messages: [
                ChatMessage(
                    role: .user,
                    content: "Name three primary colors, comma-separated.")
            ],
            parameters: m.defaultGenerationParameters.with {
                $0.maxTokens = 64
                $0.temperature = 0
            })
        let text = m.textTokenizer.decode(tokens: r.generatedTokens, skipSpecialTokens: true)
        print("Gemma 4 text → \(text)")
        expectCoherentOutput(r.generatedTokens, minTokens: 4, label: "gemma4-text")
    }

    // MARK: - image → text

    @Test("image + text — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "Gemma 4 VL checkpoint is not a VLM")

        // Gemma 4 chat prompt with the image span INSIDE the user turn:
        // `<bos><start_of_turn>user\n` + boi + <image>×N + eoi + instruction
        // + `<end_of_turn>\n<start_of_turn>model\n`. Gemma 4 is BOS-critical
        // and the multimodal path doesn't auto-prepend one → explicit <bos>.
        let imageTokenId = vlm.imageTokenId
        let bos = try #require(
            m.textTokenizer.bosTokenId ?? m.config.bosTokenId, "no BOS token id")
        let userPrefix = m.textTokenizer.encode(text: "<start_of_turn>user\n")
        let userSuffix = m.textTokenizer.encode(
            text: "\nDescribe this image.<end_of_turn>\n<start_of_turn>model\n")

        // The variable-resolution processor sizes the soft-token count per
        // image, so encode first and match the placeholder run.
        let image = try VisionTestHelpers.dogImage()
        let nImageTokens = vlm.encodeImage(image).shape[0]
        let imageSpan =
            [Self.boiTokenId]
            + Array(repeating: imageTokenId, count: nImageTokens)
            + [Self.eoiTokenId]
        let promptTokens = [bos] + userPrefix + imageSpan + userSuffix

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200,
            parameters: m.defaultGenerationParameters.with { $0.seed = 42 },
            eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        expectCoherentOutput(generated, minTokens: 8, label: "Gemma 4 VL image+text")
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 4 VL image → \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "Gemma 4 VL")
    }

    // MARK: - audio → text

    @Test("audio encode — front-end + encoder + embed_audio produce text-hidden soft tokens")
    func audioEncodeProducesSoftTokens() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "checkpoint is not a VLM")
        try #require(vlm.supportsAudio, "checkpoint has no audio tower")

        // Encode a short tone → audio soft tokens, exercising the whole
        // front-end → Conformer encoder → embed_audio plumbing; assert the
        // output shape is `[nAudioTokens, textHidden]` with a positive run.
        let audioTokens = try vlm.encodeAudio(AudioTestHelpers.tone())
        #expect(audioTokens.shape.count == 2)
        #expect(audioTokens.shape[0] > 0, "expected a positive run of audio soft tokens")
        #expect(
            audioTokens.shape[1] == m.textEngine.hidden,
            "audio soft tokens must project into the text hidden dim (\(audioTokens.shape[1]) != \(m.textEngine.hidden))")
    }

    @Test("audio + text prompt — transcribes the clean_001 speech clip coherently")
    func audioTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "checkpoint is not a VLM")
        let audioTokenId = try #require(vlm.audioTokenId, "no audio token id")

        // Real 16 kHz speech: "Sure, I can help you with that." (~1.85 s).
        let waveform = try AudioTestHelpers.clean001Waveform()

        // Encode first so we know how many audio soft tokens to splice; the
        // prompt carries one `audioTokenId` per encoder frame inside the
        // `boa … eoa` span (Gemma 4 is BOS-critical → explicit leading <bos>).
        let audioTokens = try vlm.encodeAudio(waveform)
        let nAudioTokens = audioTokens.shape[0]
        #expect(nAudioTokens > 0, "encoder produced no audio soft tokens")

        let bos = try #require(
            m.textTokenizer.bosTokenId ?? m.config.bosTokenId, "no BOS token id")
        let userPrefix = m.textTokenizer.encode(text: "<start_of_turn>user\n")
        let userSuffix = m.textTokenizer.encode(
            text: "\nTranscribe this audio.<end_of_turn>\n<start_of_turn>model\n")
        let audioSpan =
            [Self.boaTokenId]
            + Array(repeating: audioTokenId, count: nAudioTokens)
            + [Self.eoaTokenId]
        let promptTokens = [bos] + userPrefix + audioSpan + userSuffix

        let generated = try vlm.generate(
            promptTokens: promptTokens, audio: waveform,
            maxTokens: 64,
            parameters: m.defaultGenerationParameters.with { $0.seed = 42 },
            eosTokenId: m.config.eosTokenId,
            eosTokenIds: m.config.eosTokenIds)

        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 4 audio → text: \(text)")
        expectCoherentOutput(
            generated, minTokens: 4, maxConsecutiveRepeat: 6,
            minUniqueRatio: 0.2, label: "gemma4-audio")
        let lower = text.lowercased()
        #expect(
            lower.contains("sure") || lower.contains("help") || lower.contains("can"),
            "transcription did not reference the clip's spoken content: \(text)")
    }

    // MARK: - video → text

    @Test("video + text — identifies the cat in the cat video")
    func videoTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "Gemma 4 VL checkpoint is not a VLM")
        _ = try #require(vlm.videoTokenId, "no video token id")

        // Gemma 4's video prompt interleaves a `"MM:SS "` timestamp +
        // begin/end-of-image markers around EACH frame's soft-token run.
        // `vlm.generate(videoURL:…)` decodes cat.mp4 and sizes each frame's
        // placeholder run from the encoded rows via the per-frame closures.
        let bos = try #require(
            m.textTokenizer.bosTokenId ?? m.config.bosTokenId, "no BOS token id")
        let tokenizer = m.textTokenizer
        let userPrefix = tokenizer.encode(text: "<start_of_turn>user\n")
        let userSuffix = tokenizer.encode(
            text: "\nWhat animal is in this video?<end_of_turn>\n"
                + "<start_of_turn>model\n")

        let generated = try vlm.generate(
            preTokens: [bos] + userPrefix,
            videoURL: VisionTestHelpers.catVideoURL(),
            maxFrames: 4,
            postTokens: userSuffix,
            framePrefix: { i in
                tokenizer.encode(text: "00:0\(i) ") + [Self.boiTokenId]
            },
            frameSuffix: { _ in [Self.eoiTokenId] + tokenizer.encode(text: " ") },
            maxTokens: 160,
            parameters: m.defaultGenerationParameters.with { $0.seed = 42 },
            eosTokenId: m.config.eosTokenId,
            eosTokenIds: m.config.eosTokenIds)
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 4 video → text: \(text)")

        expectCoherentOutput(generated, minTokens: 4, label: "gemma4-video")
        #expect(
            text.lowercased().contains("cat"),
            "video caption should mention a cat — got: \(text)")
    }

    // MARK: - numeric mlx-vlm oracle tests (fixture-gated)

    /// Vision tower vs the mlx-vlm `embed_vision` oracle: feed Butter's
    /// `encode` the EXACT preprocessed pixels mlx-vlm produced for the dog
    /// fixture (`[3, 672, 912]` → 42×57 grid → 266 soft tokens) and compare
    /// the `[266, 1536]` soft tokens. Isolates the tower from preprocessing.
    @Test("vision tower matches the mlx-vlm oracle on the dog pixels")
    func visionTowerMatchesOracle() async throws {
        let pixelsPath = "/tmp/gemma4_pixels_oracle.f32"
        let embedPath = "/tmp/gemma4_embed_out_oracle.f32"
        let fm = FileManager.default
        try #require(
            fm.fileExists(atPath: pixelsPath) && fm.fileExists(atPath: embedPath),
            "oracle fixtures missing — run dump_gemma4_vision.py + dump_pixels.py")

        func readF32(_ path: String) -> [Float] {
            (fm.contents(atPath: path) ?? Data())
                .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "not a VLM")

        let gridH = 42, gridW = 57, patch = 16
        let imgH = gridH * patch, imgW = gridW * patch  // 672 × 912
        let pixels = readF32(pixelsPath)
        #expect(pixels.count == 3 * imgH * imgW)
        let pixT = Tensor.empty(shape: [1, 3, imgH, imgW], dtype: .f32)
        pixT.copyIn(from: pixels)
        let soft = vlm.visionEncoder.encode(image: pixT)
        let got = soft.toFloatArray()
        let oracle = readF32(embedPath)
        print("Butter soft \(soft.shape), oracle count \(oracle.count)")
        #expect(
            soft.shape == [266, 1536],
            "expected [266,1536] soft tokens, got \(soft.shape)")
        #expect(got.count == oracle.count, "count \(got.count) vs \(oracle.count)")
        let n = min(got.count, oracle.count)
        var maxDiff: Float = 0, sumDiff: Float = 0
        for i in 0 ..< n {
            maxDiff = max(maxDiff, abs(got[i] - oracle[i]))
            sumDiff += abs(got[i] - oracle[i])
        }
        let meanDiff = sumDiff / Float(n)
        print("vision tower mean abs diff \(meanDiff), max \(maxDiff)")
        #expect(
            meanDiff < 0.05,
            "vision tower diverges from the oracle (mean \(meanDiff))")
    }

    /// Audio encoder vs the mlx-vlm oracle: feed the EXACT mel
    /// `/tmp/gemma4_audio_mel.f32` to `Gemma4AudioEncoder.encode` and compare
    /// to `/tmp/gemma4_audio_oracle.f32`. Loads the raw encoder (not the
    /// `Model`) since it gates the tower math directly.
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

        func readF32(_ path: String) -> [Float] {
            let d = fm.contents(atPath: path) ?? Data()
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let melFlat = readF32(melPath)
        let nFrames = melFlat.count / 128
        let mel = MediaUtils.makeTensor(
            from: melFlat, shape: [nFrames, 128], dtype: .f32, device: .shared)
        let (enc, _) = encoder.encode(
            mel: mel, melMask: [Bool](repeating: true, count: nFrames), device: .shared)
        let got = enc.toFloatArray()
        let oracle = readF32(oraclePath)

        #expect(got.count == oracle.count, "shape mismatch: \(got.count) vs \(oracle.count)")
        var maxDiff: Float = 0, sumDiff: Float = 0
        let n = min(got.count, oracle.count)
        for i in 0 ..< n {
            let d = abs(got[i] - oracle[i])
            maxDiff = max(maxDiff, d)
            sumDiff += d
        }
        let meanDiff = sumDiff / Float(n)
        print("audio encoder max abs diff \(maxDiff), mean abs diff \(meanDiff)")
        // Gross-bug regression guard (a real algorithm bug spikes the MEAN);
        // the true gate is the coherent audio→text test above.
        #expect(
            meanDiff < 1.5,
            "audio encoder grossly diverges from the reference (mean abs diff \(meanDiff))")
    }

    /// Magnitude-Mel front-end vs the transformers `Gemma4AudioFeatureExtractor`
    /// for the shared test tone — bit-for-bit up to f32 rounding, so a tight
    /// gate catches any framing/window/filterbank regression. No model load.
    @Test("mel front-end matches the Gemma 4 feature extractor on the test tone")
    func melFrontEndMatchesReference() throws {
        let oraclePath = "/tmp/gemma4_sine_mel_oracle.f32"
        try #require(
            FileManager.default.fileExists(atPath: oraclePath),
            "mel oracle fixture missing — run the mlx-vlm dump first")
        let (mel, _) = Gemma4AudioFrontend.melSpectrogram(waveform: AudioTestHelpers.tone())
        let got = mel.toFloatArray()
        let oracle = (FileManager.default.contents(atPath: oraclePath) ?? Data())
            .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        #expect(got.count == oracle.count, "mel shape mismatch \(got.count) vs \(oracle.count)")
        var maxDiff: Float = 0
        let n = min(got.count, oracle.count)
        for i in 0 ..< n {
            maxDiff = max(maxDiff, abs(got[i] - oracle[i]))
        }
        print("mel max abs diff \(maxDiff)")
        #expect(
            maxDiff < 5e-3,
            "mel front-end diverges from the reference (max abs diff \(maxDiff))")
    }
}
