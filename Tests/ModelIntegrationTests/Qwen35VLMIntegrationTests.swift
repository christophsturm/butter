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
// Consolidated integration suite for the Qwen 3.5-VL
// (`Qwen3_5ForConditionalGeneration`) checkpoint: the Qwen3-VL ViT tower +
// the Qwen 3.5 dense hybrid text backbone, joined via `Qwen35VL.load` in
// `Models/Vision/Qwen3xVision.swift`. ONE @Suite that loads the model
// ONCE and exercises every capability it supports — text, image, video —
// with a per-modality `@Test`. This is the canonical multimodal test
// shape: a single load shared across all capability tests (see `loaded`),
// rather than one file + reload per modality.
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
// `qwen3_vl` orchestrator + the larger 30B-A3B checkpoint. Qwen 3.5-VL is
// text + image + video (no audio). The checkpoint MUST load — a load
// failure fails the test.

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite(
    "Qwen3.5-VLM Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct Qwen35VLMIntegrationTests {

    static let frameCount = 4

    /// The Qwen 3.5-VL checkpoint, loaded ONCE and shared across every
    /// `@Test`. A lazily-started memoized `Task`: the first capability
    /// test that awaits `.value` triggers the single load; the rest reuse
    /// it. A skipped suite never accesses it, so no load happens when
    /// gated off.
    static let loaded = Task { try await loadModel("mlx-community/Qwen3.5-2B-4bit") }

    // MARK: - load + text

    @Test("load — Qwen 3.5-VL checkpoint loads with image + video capabilities")
    func load() async throws {
        let m = try await Self.loaded.value

        // The checkpoint is a VLM — vlModel is present, .imageIn and
        // .videoIn are available, and the text backbone supports the
        // splice.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.videoIn))
        #expect(m.textEngine.supportsEmbeddingInput)  // VLM splice prerequisite

        let vlm = try #require(m.vlModel)
        // The vision tower contributes a positive run of merged tokens.
        #expect(vlm.imageTokenCount > 0)
        // The video splice needs the family loader to thread
        // `video_token_id` through to VisionModel.init.
        #expect(vlm.videoTokenId != nil)
    }

    // KNOWN-FAILING CANARY: the Qwen 3.5-VL checkpoint degenerates on pure
    // text-only generation (image + video are coherent — see those tests).
    // Kept, per the team decision, to keep surfacing the VL text-only path
    // until it's investigated; it is expected to fail in the bisect today.
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
        expectCoherentOutput(r.generatedTokens, minTokens: 4, label: "qwen35vl-text")
    }

    // MARK: - image / video → text

    @Test("image + text — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "Qwen 3.5-VL checkpoint is not a VLM")

        // A real photograph — the golden-retriever fixture.
        let image = try VisionTestHelpers.dogImage()

        // Qwen 3.5-VL shares the Qwen 3-VL native-resolution tower: size the
        // <|image_pad|> run to the image's NATIVE encoded token count and wrap
        // it in the <|vision_start|> … <|vision_end|> markers (the nominal
        // `imageTokenCount` with no markers is malformed → degenerate output).
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

        // Coherence first, then the content check: the caption should
        // mention a dog.
        expectCoherentOutput(
            generated, minTokens: 8,
            label: "Qwen 3.5-VL image+text")
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3.5-VL generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "Qwen 3.5-VL")
    }

    @Test("video + text — describes the cat clip")
    func videoTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "Qwen 3.5-VL checkpoint is not a VLM")
        _ = try #require(vlm.videoTokenId, "Qwen 3.5-VL must be video-capable")

        // Build the standard Qwen 3-VL video prompt:
        //   <|im_start|>user\n<|vision_start|><|video_pad|>…<|vision_end|>What's in this video?<|im_end|>\n
        //   <|im_start|>assistant\n
        // The model decodes cat.mp4 (`maxFrames` evenly-spaced frames), folds
        // them into temporal patches, and inserts exactly the matching number
        // of `<|video_pad|>` placeholders between the markers — the test no
        // longer decodes frames or hand-computes the placeholder count.
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

        // Coherence first, then the content check: the caption should
        // mention a cat (or kitten — model verbosity varies).
        expectCoherentOutput(
            generated, minTokens: 8,
            label: "Qwen 3.5-VL video+text")
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3.5-VL video generated: \(text)")
        VisionTestHelpers.expectMentionsCat(text, label: "Qwen 3.5-VL video")
    }
}
