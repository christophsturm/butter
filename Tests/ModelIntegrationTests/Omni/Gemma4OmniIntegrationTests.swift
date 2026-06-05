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
// Consolidated integration suite for the Gemma 4 Omni (`gemma4_unified`)
// checkpoint: ONE @Suite that loads the model ONCE and exercises every
// capability it supports — text, image, audio, video — with a per-modality
// `@Test`. This is the canonical "omni" test shape: a single load shared
// across all capability tests (see `loaded`), rather than one file +
// reload per modality.
//
// The encoder-free omni path projects raw inputs (48×48 image patches,
// 640-sample audio frames, per-frame video) straight into the Gemma4Dense
// text backbone via lightweight RMSNorm+Linear embedders — no ViT /
// Conformer encoder. Resolves the published `ekryski/gemma-4-12B-it-mlx-4bit`
// conversion, preferring a local `ffai convert` output dir when present.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

/// Local `ffai convert` output, then the published HF repo.
let gemma4OmniLocalDir =
    ("~/.cache/ffai/converts/gemma-4-12B-it-mlx-4bit" as NSString)
    .expandingTildeInPath
let gemma4OmniHFCache =
    ("~/.cache/huggingface/hub/models--ekryski--gemma-4-12B-it-mlx-4bit"
        as NSString).expandingTildeInPath

/// Path to load: the local convert dir if present, else the HF repo id.
let gemma4OmniModelPath: String =
    FileManager.default.fileExists(atPath: gemma4OmniLocalDir)
    ? gemma4OmniLocalDir : "ekryski/gemma-4-12B-it-mlx-4bit"

/// True when the model is resolvable offline (local convert dir or HF cache).
let gemma4OmniCacheAvailable: Bool =
    FileManager.default.fileExists(atPath: gemma4OmniLocalDir)
    || FileManager.default.fileExists(atPath: gemma4OmniHFCache)

@Suite(
    "Gemma4 Omni Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableOmniSuites && gemma4OmniCacheAvailable,
        "Gemma 4 Omni suite: needs the omni-suite group enabled and the omni 4-bit checkpoint cached (or a local `ffai convert` output dir).")
)
struct Gemma4OmniIntegrationTests {

    // Image/audio placeholder begin/end markers (omni config).
    static let boi = 255_999, eoi = 258_882
    static let boa = 256_000, eoa = 258_883

    /// The omni checkpoint, loaded ONCE and shared across every `@Test`.
    /// A lazily-started memoized `Task`: the first capability test that
    /// awaits `.value` triggers the single load; the rest reuse it. A
    /// skipped suite never accesses it, so no load happens when gated off.
    static let loaded = Task { try await loadModel(gemma4OmniModelPath) }

    /// 0.5 s 440 Hz mono sine at 16 kHz — a non-degenerate signal for the
    /// raw-frame audio path (content doesn't matter; we gate on a positive
    /// soft-token run + coherent decode).
    static func sine(seconds: Double = 0.5) -> [Float] {
        let n = Int(seconds * 16_000)
        return (0 ..< n).map {
            0.3 * Float(sin(2.0 * .pi * 440.0 * Double($0) / 16_000.0))
        }
    }

    // MARK: - load + text

    @Test("load — omni checkpoint loads as a VLM with full omni capabilities")
    func load() async throws {
        let m = try await Self.loaded.value
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.audioIn))
        #expect(m.availableCapabilities.contains(.videoIn))
        #expect(m.engine.supportsEmbeddingInput)
        // The omni text backbone is the 48-layer, 3840-hidden Gemma4Dense.
        #expect(m.engine.hidden == 3840)
        #expect(m.engine.nLayers == 48)
    }

    @Test("text — encoder-free backbone chat-generates coherently")
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
        expectCoherentOutput(r.generatedTokens, minTokens: 4, label: "gemma4-omni-text")
    }

    // MARK: - encode shapes

    @Test("encode shapes — image / audio / video project into the text hidden dim")
    func encodeShapes() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "omni checkpoint is not a VLM")
        let hidden = m.engine.hidden

        let img = vlm.encodeImage(try VisionTestHelpers.dogImage())
        #expect(img.shape.count == 2 && img.shape[0] > 0 && img.shape[1] == hidden)

        try #require(vlm.supportsAudio)
        let aud = try vlm.encodeAudio(Self.sine())  // ~13 frames at 640 samples
        #expect(aud.shape.count == 2 && aud.shape[0] > 0 && aud.shape[1] == hidden)

        let vid = try vlm.encodeVideo(VisionTestHelpers.catVideoFrames(maxFrames: 2))
        #expect(vid.shape.count == 2 && vid.shape[0] > 0 && vid.shape[1] == hidden)
    }

    // MARK: - image / audio / video → text

    @Test("image + text — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "omni checkpoint is not a VLM")

        let image = try VisionTestHelpers.dogImage()
        let nImg = vlm.encodeImage(image).shape[0]
        let bos = try #require(m.tokenizer.bosTokenId ?? m.config.bosTokenId)
        let pre = m.tokenizer.encode(text: "<start_of_turn>user\n")
        let suf = m.tokenizer.encode(
            text: "\nDescribe this image.<end_of_turn>\n<start_of_turn>model\n")
        let span =
            [Self.boi] + Array(repeating: vlm.imageTokenId, count: nImg) + [Self.eoi]
        let prompt = [bos] + pre + span + suf

        let generated = try vlm.generate(
            promptTokens: prompt, image: image, maxTokens: 96,
            parameters: m.defaultGenerationParameters.with { $0.temperature = 0 },
            eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 4 Omni image → text: \(text)")
        expectCoherentOutput(generated, minTokens: 6, label: "gemma4-omni-image")
        VisionTestHelpers.expectMentionsDog(text, label: "Gemma 4 Omni")
    }

    @Test("audio + text — raw-frame audio path decodes coherently")
    func audioTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "omni checkpoint is not a VLM")
        let audioTokenId = try #require(vlm.audioTokenId)

        let wav = Self.sine()
        let nAud = try vlm.encodeAudio(wav).shape[0]
        #expect(nAud > 0)
        let bos = try #require(m.tokenizer.bosTokenId ?? m.config.bosTokenId)
        let pre = m.tokenizer.encode(text: "<start_of_turn>user\n")
        let suf = m.tokenizer.encode(
            text: "\nWhat do you hear?<end_of_turn>\n<start_of_turn>model\n")
        let span =
            [Self.boa] + Array(repeating: audioTokenId, count: nAud) + [Self.eoa]
        let prompt = [bos] + pre + span + suf

        let generated = try vlm.generate(
            promptTokens: prompt, audio: wav, maxTokens: 48,
            parameters: m.defaultGenerationParameters.with { $0.temperature = 0 },
            eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)
        print(
            "Gemma 4 Omni audio → text: "
                + m.tokenizer.decode(tokens: generated, skipSpecialTokens: true))
        expectCoherentOutput(generated, minTokens: 3, label: "gemma4-omni-audio")
    }

    @Test("video + text — identifies the cat video")
    func videoTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "omni checkpoint is not a VLM")
        let videoTokenId = try #require(vlm.videoTokenId)

        let frames = try VisionTestHelpers.catVideoFrames(maxFrames: 3)
        let perFrame = vlm.encodeImage(frames[0]).shape[0]
        let bos = try #require(m.tokenizer.bosTokenId ?? m.config.bosTokenId)
        let pre = m.tokenizer.encode(text: "<start_of_turn>user\n")
        let suf = m.tokenizer.encode(
            text: "\nWhat animal is in this video?<end_of_turn>\n<start_of_turn>model\n")
        var span: [Int] = []
        for i in 0 ..< frames.count {
            span += m.tokenizer.encode(text: "00:0\(i) ")
            span += [Self.boi]
            span += Array(repeating: videoTokenId, count: perFrame)
            span += [Self.eoi] + m.tokenizer.encode(text: " ")
        }
        let prompt = [bos] + pre + span + suf

        let generated = try vlm.generate(
            promptTokens: prompt, videoFrames: frames, maxTokens: 64,
            parameters: m.defaultGenerationParameters.with { $0.temperature = 0 },
            eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 4 Omni video → text: \(text)")
        expectCoherentOutput(generated, minTokens: 4, label: "gemma4-omni-video")
        #expect(
            text.lowercased().contains("cat"),
            "video caption should mention a cat — got: \(text)")
    }
}
