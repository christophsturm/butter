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
// Consolidated integration suite for the Qwen 3-VL
// (`Qwen3VLForConditionalGeneration`) checkpoint: ONE @Suite that loads the
// model ONCE and exercises every capability it supports — text, image, video
// — with a per-modality `@Test`. This is the canonical "omni" test shape: a
// single load shared across all capability tests (see `loaded`), rather than
// one file + reload per modality (merged from the prior
// `Qwen3VisionIntegrationTests.swift`).
//
// Verifies the vision path end-to-end on a real checkpoint: the Qwen 3-VL
// vision tower loads, runs its full-attention + M-RoPE forward and
// patch-merger, the cross-modal splice injects the merged image/video tokens,
// and the fused stream decodes coherent text through the Qwen 3 dense backbone
// (which supports embedding-input forward for the splice).
//
// Uses the mlx-community 2B-Instruct 4-bit conversion — the smallest published
// Qwen3-VL with a complete local snapshot. The checkpoint MUST load — a load
// failure fails the test. Qwen 3-VL = text + image + video; there is NO audio
// path for this checkpoint.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Qwen3-VLM Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct Qwen3VLMIntegrationTests {

    static let modelId = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

    /// Number of evenly-spaced frames to pull out of `cat.mp4`. Must be
    /// a multiple of `temporal_patch_size` (2) so the vision tower
    /// doesn't have to pad with the last frame — keeps the placeholder
    /// arithmetic exact.
    static let frameCount = 4

    /// The Qwen 3-VL checkpoint, loaded ONCE and shared across every `@Test`.
    /// A lazily-started memoized `Task`: the first capability test that awaits
    /// `.value` triggers the single load; the rest reuse it. A skipped suite
    /// never accesses it, so no load happens when gated off.
    static let loaded = Task { try await loadModel(Self.modelId) }

    // MARK: - load + text

    @Test("load — Qwen 3-VL checkpoint loads with image + video capability")
    func load() async throws {
        let m = try await Self.loaded.value

        // The checkpoint is a VLM — vlModel is present, .imageIn and .videoIn
        // are available, and the text backbone supports the splice.
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

    @Test("text — VL backbone generates coherently (text-only prompt)")
    func textGeneration() async throws {
        let m = try await Self.loaded.value
        // Hand-frame the chat turn (the image/video tests do the same) rather
        // than go through `generate(messages:)`: the Qwen 3-VL chat template
        // is geared for vision turns and degenerates on a text-only message.
        let prompt =
            "<|im_start|>user\nName three primary colors, comma-separated."
            + "<|im_end|>\n<|im_start|>assistant\n"
        let r = try await m.generate(
            prompt: prompt,
            parameters: m.defaultGenerationParameters.with {
                $0.maxTokens = 64
                $0.temperature = 0
            })
        expectCoherentOutput(r.generatedTokens, minTokens: 4, label: "qwen3vl-text")
    }

    // MARK: - image / video → text

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "Qwen 3-VL checkpoint is not a VLM")

        // A real photograph — the golden-retriever fixture.
        let image = try VisionTestHelpers.dogImage()

        // Build the standard Qwen 3-VL image prompt: the <|vision_start|> …
        // <|vision_end|> markers wrapping a run of <|image_pad|> placeholders
        // sized to the image's NATIVE-resolution token count (Qwen 3-VL is a
        // native-resolution tower — `imageTokenCount` is only the nominal
        // square count; the actual count comes from `encodeImage`).
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
            label: "Qwen 3-VL image+text")
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3-VL generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "Qwen 3-VL")
    }

    /// Same caption, but through the UNIFIED message-based surface —
    /// `Model.generate(.vision(messages:media:))`. Exercises the whole
    /// template-driven path with NO hand-built promptTokens: the Qwen 3-VL
    /// chat template renders the `.image` content part into
    /// `<|vision_start|><|image_pad|><|vision_end|>`, the builder expands the
    /// single `<|image_pad|>` to the native-resolution token count, and
    /// splice/decode run. Proves the message-based vision path end-to-end.
    @Test("image via unified Model.generate(.vision) — describes the dog photo")
    func imageTextGenerationUnified() async throws {
        let m = try await Self.loaded.value
        let image = try VisionTestHelpers.dogImage()

        let messages = [
            ChatMessage(
                role: .user, content: [.image, .text("Describe this image.")])
        ]
        let out = try await m.generate(
            .vision(
                VisionRequest(
                    messages: messages, media: [.image(image)],
                    maxTokens: 200,
                    parameters: m.defaultGenerationParameters.with {
                        $0.temperature = 0
                    },
                    eosTokenId: m.config.eosTokenId,
                    eosTokenIds: m.config.eosTokenIds)))

        guard case .text(let result) = out else {
            Issue.record("expected .text output from a VLM image request")
            return
        }
        expectCoherentOutput(
            result.generatedTokens, minTokens: 8,
            label: "Qwen 3-VL unified image+text")
        print("Qwen 3-VL unified generated: \(result.text)")
        VisionTestHelpers.expectMentionsDog(result.text, label: "Qwen 3-VL unified")
    }

    @Test("video + text prompt — describes the cat clip")
    func videoTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "Qwen 3-VL checkpoint is not a VLM")
        _ = try #require(vlm.videoTokenId, "Qwen 3-VL must be video-capable")

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
            label: "Qwen 3-VL video+text")
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3-VL video generated: \(text)")
        VisionTestHelpers.expectMentionsCat(text, label: "Qwen 3-VL video")
    }
}
