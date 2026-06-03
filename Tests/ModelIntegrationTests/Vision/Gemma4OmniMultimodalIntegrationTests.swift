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
// Integration test for the Gemma 4 Omni encoder-free multimodal path:
// image / audio / video → text on the real 4-bit checkpoint. Each modality
// projects raw inputs (48×48 patches, 640-sample audio frames, per-frame
// video) straight into the text backbone via the lightweight RMSNorm+Linear
// embedders — no ViT / Conformer encoder.
//
// Reuses the shared `gemma4OmniModelPath` / `gemma4OmniCacheAvailable`
// resolver from `Gemma4OmniTextIntegrationTests`. Gated on the vision-suite
// group (like the Gemma 4 VL suite) AND model availability.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Gemma4 Omni Multimodal Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites && gemma4OmniCacheAvailable,
        "Gemma 4 Omni multimodal suite: needs the vision-suite group enabled and the omni 4-bit checkpoint cached.")
)
struct Gemma4OmniMultimodalIntegrationTests {

    // Image/audio/video placeholder ids + begin/end markers (omni config).
    static let boi = 255_999, eoi = 258_882
    static let boa = 256_000, eoa = 258_883

    /// 0.5 s 440 Hz mono sine at 16 kHz — a non-degenerate signal for the
    /// raw-frame audio path (content doesn't matter; we gate on a positive
    /// soft-token run + coherent decode).
    static func sine(seconds: Double = 0.5) -> [Float] {
        let n = Int(seconds * 16_000)
        return (0 ..< n).map {
            0.3 * Float(sin(2.0 * .pi * 440.0 * Double($0) / 16_000.0))
        }
    }

    @Test("encode shapes — image / audio / video project into the text hidden dim")
    func encodeShapes() async throws {
        let m = try await loadModel(gemma4OmniModelPath)
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

    @Test("image + text — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await loadModel(gemma4OmniModelPath)
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
        let m = try await loadModel(gemma4OmniModelPath)
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
        let m = try await loadModel(gemma4OmniModelPath)
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
