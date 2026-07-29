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
// Consolidated integration suite for the Qwen 3.6-VL checkpoint. There is
// no small Qwen 3.6-VL conversion, so this uses the 35B-A3B MoE 4-bit
// (`mlx-community/Qwen3.6-35B-A3B-4bit`, model_type `qwen3_5_moe` + a real
// `vision_tower.*` ViT). It shares the Qwen3-VL vision tower + the Qwen 3.5
// hybrid (here MoE) text backbone, joined via `Qwen35VL.load`.
//
// ONE @Suite that loads the (large) checkpoint ONCE and exercises every
// capability — text, image, video — with a per-modality @Test. NOTE: this
// is a 35B MoE; the load + per-prompt decode are multi-minute.

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite(
    "Qwen3.6-VL Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct Qwen36VLIntegrationTests {

    static let frameCount = 4

    /// The Qwen 3.6-VL checkpoint, loaded ONCE and shared across every
    /// `@Test` (lazily-started memoized Task). A skipped suite never accesses
    /// it, so no load happens when gated off.
    static let loaded = Task { try await loadModel("mlx-community/Qwen3.6-35B-A3B-4bit") }

    // MARK: - load + text

    @Test("load — Qwen 3.6-VL checkpoint loads with image + video capabilities")
    func load() async throws {
        let m = try await Self.loaded.value
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.videoIn))
        #expect(m.textEngine.supportsEmbeddingInput)
        let vlm = try #require(m.vlModel)
        #expect(vlm.imageTokenCount > 0)
        #expect(vlm.videoTokenId != nil)
    }

    // KNOWN-FAILING CANARY: like the other VL checkpoints, Qwen 3.6-VL
    // degenerates on pure text-only generation (image + video are coherent).
    // Kept per the team decision to keep surfacing the VL text-only path.
    @Test("text — VL backbone text-only generation (known-failing canary)")
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
        expectCoherentOutput(r.generatedTokens, minTokens: 4, label: "qwen36vl-text")
    }

    // MARK: - image / video → text

    @Test("image + text — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "Qwen 3.6-VL checkpoint is not a VLM")

        let image = try VisionTestHelpers.dogImage()

        // Native-resolution token count wrapped in the <|vision_start|> …
        // <|vision_end|> markers (the Qwen 3-VL tower convention).
        let nImg = vlm.encodeImage(image).shape[0]
        let preTokens = m.textTokenizer.encode(
            text: "<|im_start|>user\n<|vision_start|>")
        let postTokens = m.textTokenizer.encode(
            text: "<|vision_end|>Describe this image.<|im_end|>\n"
                + "<|im_start|>assistant\n")
        let promptTokens =
            preTokens + Array(repeating: vlm.imageTokenId, count: nImg) + postTokens

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        expectCoherentOutput(
            generated, minTokens: 8, label: "Qwen 3.6-VL image+text")
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3.6-VL generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "Qwen 3.6-VL")
    }

    @Test("video + text — describes the cat clip")
    func videoTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "Qwen 3.6-VL checkpoint is not a VLM")
        _ = try #require(vlm.videoTokenId, "Qwen 3.6-VL must be video-capable")

        let preTokens = m.textTokenizer.encode(
            text: "<|im_start|>user\n<|vision_start|>")
        let postTokens = m.textTokenizer.encode(
            text: "<|vision_end|>What's in this video?<|im_end|>\n"
                + "<|im_start|>assistant\n")

        let generated = try vlm.generate(
            preTokens: preTokens,
            videoURL: VisionTestHelpers.catVideoURL(),
            maxFrames: Self.frameCount,
            postTokens: postTokens,
            maxTokens: 200,
            eosTokenId: m.config.eosTokenId,
            eosTokenIds: m.config.eosTokenIds)

        expectCoherentOutput(
            generated, minTokens: 8, label: "Qwen 3.6-VL video+text")
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3.6-VL video generated: \(text)")
        VisionTestHelpers.expectMentionsCat(text, label: "Qwen 3.6-VL video")
    }
}
