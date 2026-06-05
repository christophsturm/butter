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
// Consolidated integration suite for MiniCPM-V 4.6 (the SigLIP2-400M +
// Qwen3.5 `MiniCPMV4_6ForConditionalGeneration` checkpoint): ONE @Suite
// that loads the model ONCE and exercises every capability it supports —
// text, image, video — with a per-modality `@Test`. MiniCPM-V 4.6 has no
// audio path, so there is no audio test.
//
// Verifies the v1 vision path end-to-end on the real
// `mlx-community/MiniCPM-V-4.6-4bit` checkpoint: SigLIP2 vision tower
// loads, position embedding is bilinearly resampled to the 32×32 runtime
// grid, `vit_merger` runs after encoder layer 6 (16× mode), the final
// `merger` projects into the Qwen 3.5 text hidden dim, the cross-modal
// splice injects 64 vision tokens, and the fused stream decodes coherent
// text through the Qwen 3.5 backbone. The video path maps each frame 1:1
// to an `imageTokenCount` run (no temporal folding).
//
// The base checkpoint is ~9B params (~18 GB bf16) — the omni-suite group
// gate keeps it out of the default run.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "MiniCPM-V 4.6 Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct MiniCPM46VIntegrationTests {

    static let modelId = "mlx-community/MiniCPM-V-4.6-4bit"

    /// Number of evenly-spaced frames to pull from cat.mp4.  MiniCPM-V 4.6
    /// has no temporal-patch requirement (no folding step), so any positive
    /// frame count is valid. 4 frames is a lightweight integration sweep.
    static let frameCount = 4

    /// The MiniCPM-V 4.6 checkpoint, loaded ONCE and shared across every
    /// `@Test`. A lazily-started memoized `Task`: the first capability test
    /// that awaits `.value` triggers the single load; the rest reuse it. A
    /// skipped suite never accesses it, so no load happens when gated off.
    static let loaded = Task { try await loadModel(modelId) }

    // MARK: - load

    @Test("load — MiniCPM-V 4.6 loads with image + video capability")
    func load() async throws {
        let m = try await Self.loaded.value

        // The checkpoint is a VLM — vlModel is present, .imageIn and
        // .videoIn are available, and the text backbone is Qwen 3.5
        // (hidden 1024). MiniCPM-V 4.6 has no audio path.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.videoIn))
        #expect(m.engine.hidden == 1024)
        #expect(m.engine.supportsEmbeddingInput)
        // Qwen 3.5 is the text backbone (`qwen3_5_text` text_config).
        #expect(m.qwen35 != nil)

        let vlm = try #require(m.vlModel, "MiniCPM-V 4.6 checkpoint is not a VLM")
        // v1: 448×448 / patch-14 → 32×32 patches → vit_merger 16×16 →
        // merger 8×8 = 64 tokens per image (matches `query_num: 64`).
        #expect(vlm.imageTokenCount == 64)
        #expect(vlm.imageTokenId == MiniCPM4.defaultImageTokenId)
        #expect(vlm.visionEncoder.config.imageSize == MiniCPM4.runtimeImageSize)
        #expect(vlm.visionEncoder.config.patchSize == 14)
        #expect(vlm.visionEncoder.config.hidden == 1152)
        #expect(vlm.visionEncoder.config.nLayers == 27)

        // The MiniCPM-V 4.6 family loader threads video_token_id (248057)
        // through to VisionModel.init; confirm it landed.
        #expect(
            vlm.videoTokenId != nil,
            "video_token_id should be threaded through from config")
        #expect(
            vlm.videoTokenId == MiniCPM4.defaultVideoTokenId,
            "video_token_id should be 248057 (<|video_pad|>)")
        #expect(
            vlm.imageTokenId == MiniCPM4.defaultImageTokenId,
            "image_token_id should be 248056 (<|image_pad|>)")
    }

    // MARK: - text

    // KNOWN-FAILING CANARY: MiniCPM-V 4.6's Qwen 3.5 backbone (like the bare
    // Qwen 3.5-VL) degenerates on pure text-only generation; image + video
    // are coherent. Kept per the team decision to keep surfacing the VL
    // text-only path until investigated; expected to fail in the bisect.
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
        expectCoherentOutput(r.generatedTokens, minTokens: 4, label: "minicpm4-text")
    }

    // MARK: - image → text

    @Test("image + text prompt — coherent multi-modal generation")
    func imageTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "MiniCPM-V 4.6 checkpoint is not a VLM")

        // Build an image+text prompt. MiniCPM-V uses the Qwen-style chat
        // wrapper (<|im_start|>user … <|im_end|> <|im_start|>assistant)
        // around a run of <|image_pad|> placeholders, NO <|vision_start|>
        // / <|vision_end|> markers (the MiniCPM-V chat template only
        // emits <|image_pad|>; checkpoint config + chat_template.jinja
        // confirm this). Without the chat wrapping the model has no
        // signal that it's in assistant-response mode and falls into a
        // degenerate emit-only-space loop (token 220 = `Ġ`), which is
        // exactly how this test had been failing.
        let imageTokenId = vlm.imageTokenId
        let header = m.tokenizer.encode(text: "<|im_start|>user\n")
        // The chat_template.jinja with the default `enable_thinking=false`
        // appends `<think>\n\n</think>\n\n` after `<|im_start|>assistant\n`.
        // Skipping that opener leaves the model in an undefined state
        // (the Qwen3.5 backbone was fine-tuned with the marker present)
        // and it loops on newlines as the lowest-perplexity continuation.
        let trailer = m.tokenizer.encode(
            text: "\nDescribe this image briefly.<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        let promptTokens =
            header
            + Array(repeating: imageTokenId, count: vlm.imageTokenCount)
            + trailer

        // A solid test image at the encoder's runtime resolution (448).
        let cfg = vlm.visionEncoder.config
        let image = RGBImage.solid(
            width: cfg.imageSize, height: cfg.imageSize,
            r: 0.55, g: 0.45, b: 0.30)

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence-only contract: a real image+text prompt should
        // decode a non-degenerate run of tokens.
        expectCoherentOutput(
            generated, minTokens: 16,
            label: "MiniCPM-V 4.6 image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("MiniCPM-V 4.6 generated: \(text)")
    }

    // MARK: - video → text

    @Test("video + text prompt — describes the cat clip")
    func videoTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(
            m.vlModel,
            "MiniCPM-V 4.6 checkpoint is not a VLM")
        _ = try #require(
            vlm.videoTokenId,
            "video_token_id must be set for video generation")

        // Build the MiniCPM-V chat-template prompt for a video query. The
        // model decodes cat.mp4 (`maxFrames` frames; MiniCPM-V maps each
        // frame 1:1 to an `imageTokenCount` run, no temporal folding) and
        // inserts the matching number of `<|video_pad|>` placeholders
        // between the markers — the test no longer decodes frames or
        // hand-computes the count. Prompt structure:
        //   <|im_start|>user\n
        //   <|video_pad|> × (frames · imageTokenCount)
        //   \nWhat's in this video?<|im_end|>\n
        //   <|im_start|>assistant\n<think>\n\n</think>\n\n
        let header = m.tokenizer.encode(text: "<|im_start|>user\n")
        // Mirror the image-path opener (`<think>\n\n</think>\n\n` after
        // `<|im_start|>assistant\n`) so video-mode generation gets the
        // same assistant-turn signal the backbone was fine-tuned on.
        let trailer = m.tokenizer.encode(
            text: "\nWhat's in this video?<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n")

        let generated = try vlm.generate(
            preTokens: header,
            videoURL: VisionTestHelpers.catVideoURL(),
            maxFrames: Self.frameCount,
            postTokens: trailer,
            maxTokens: 200,
            eosTokenId: m.config.eosTokenId,
            eosTokenIds: m.config.eosTokenIds)

        // Coherence first — the model should produce a non-degenerate run.
        expectCoherentOutput(
            generated, minTokens: 8,
            label: "MiniCPM-V 4.6 video+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("MiniCPM-V 4.6 video generated: \(text)")
        VisionTestHelpers.expectMentionsCat(text, label: "MiniCPM-V 4.6 video")
    }
}
