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
// Slow integration test for Qwen 3.5-VL (the
// `Qwen3_5ForConditionalGeneration` checkpoint: the Qwen3-VL ViT tower +
// the Qwen 3.5 dense hybrid text backbone, joined via `Qwen35VL.load` in
// `Models/Vision/Qwen3xVision.swift`).
//
// Verifies the vision path end-to-end on a real checkpoint: the
// Qwen3-VL vision tower loads, runs its full-attention + M-RoPE forward
// and patch-merger, the cross-modal splice injects the merged image
// tokens, and the fused stream decodes coherent text through the
// Qwen 3.5 hybrid backbone.
//
// Uses the mlx-community Qwen3.5-2B 4-bit conversion (model_type
// `qwen3_5`, `image_token_id` 248056 / `video_token_id` 248057) — a
// genuinely different model from Qwen3-VL-MoE, which carries its own
// `qwen3_vl` orchestrator + the larger 30B-A3B checkpoint. The checkpoint
// MUST load — a load failure fails the test.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Qwen35 Vision Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct Qwen35VisionIntegrationTests {

    static let modelId = "mlx-community/Qwen3.5-2B-4bit"
    static let frameCount = 4

    @Test("load — Qwen 3.5-VL checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // The checkpoint is a VLM — vlModel is present, .imageIn is
        // available, and the text backbone supports the splice.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.engine.supportsEmbeddingInput)  // VLM splice prerequisite

        let vlm = try #require(m.vlModel)
        // The vision tower contributes a positive run of merged tokens.
        #expect(vlm.imageTokenCount > 0)
    }

    @Test("enable / disable .imageIn — runtime capability flip")
    func capabilityFlip() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        #expect(m.availableCapabilities.contains(.imageIn))
        m.disable(.imageIn)
        #expect(!m.isEnabled(.imageIn))
        m.enable(.imageIn)
        #expect(m.isEnabled(.imageIn))
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "Qwen 3.5-VL checkpoint is not a VLM")

        // Build an image+text prompt: a run of `imageTokenCount`
        // <|image_pad|> placeholders followed by a text question.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "<|im_start|>user\nDescribe this image.<|im_end|>\n"
                + "<|im_start|>assistant\n")
        let promptTokens =
            Array(
                repeating: imageTokenId,
                count: vlm.imageTokenCount) + questionTokens

        // A real photograph — the golden-retriever fixture.
        let image = try VisionTestHelpers.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence first, then the content check: the caption should
        // mention a dog.
        expectCoherentOutput(
            generated, minTokens: 8,
            label: "Qwen 3.5-VL image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3.5-VL generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "Qwen 3.5-VL")
    }

    @Test("load — checkpoint reports .videoIn capability")
    func loadVLCheckpointVideo() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        // The Qwen 3.5-VL family declares text + vision + video.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.videoIn))

        let vlm = try #require(m.vlModel)
        // The video splice needs the family loader to thread
        // `video_token_id` through to VisionModel.init.
        #expect(vlm.videoTokenId != nil)
    }

    @Test("video + text prompt — describes the cat clip")
    func videoTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "Qwen 3.5-VL checkpoint is not a VLM")
        _ = try #require(vlm.videoTokenId, "Qwen 3.5-VL must be video-capable")

        // Build the standard Qwen 3-VL video prompt:
        //   <|im_start|>user\n<|vision_start|><|video_pad|>…<|vision_end|>What's in this video?<|im_end|>\n
        //   <|im_start|>assistant\n
        // The model decodes cat.mp4 (`maxFrames` evenly-spaced frames), folds
        // them into temporal patches, and inserts exactly the matching number
        // of `<|video_pad|>` placeholders between the markers — the test no
        // longer decodes frames or hand-computes the placeholder count.
        let preTokens = m.tokenizer.encode(
            text: "<|im_start|>user\n<|vision_start|>")
        let postTokens = m.tokenizer.encode(
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

        // Coherence first, then the content check: the caption should
        // mention a cat (or kitten — model verbosity varies).
        expectCoherentOutput(
            generated, minTokens: 8,
            label: "Qwen 3.5-VL video+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3.5-VL video generated: \(text)")
        VisionTestHelpers.expectMentionsCat(text, label: "Qwen 3.5-VL video")
    }
}
