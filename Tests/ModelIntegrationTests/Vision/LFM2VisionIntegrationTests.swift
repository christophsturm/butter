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
// Slow integration test for LFM2-VL (the SigLIP2 + LFM2
// `Lfm2VlForConditionalGeneration` checkpoint).
//
// Verifies the end-to-end vision path on a real checkpoint:
// the SigLIP2 NaFlex tower loads into the shared `VisionEncoder` and runs
// at native (aspect-preserving) resolution — the patch grid is capped at
// `max_num_patches` (1024), so the per-image ViT-token count varies. The
// pixel-unshuffle(2) + MLP projector reduces and projects to per-image
// super-patch embeddings, the cross-modal splice injects the image tokens,
// and the fused stream decodes coherent text through the LFM2 hybrid text
// backbone (short-conv / attention layers).
//
// Uses the mlx-community 4-bit conversion (LFM2-VL-1.6B-4bit). The
// checkpoint MUST load — a load failure fails the test.
//
// DO NOT RUN — heavyweight test, requires the HuggingFace checkpoint at
// ~/.cache/huggingface/hub/models--mlx-community--LFM2-VL-1.6B-4bit/.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "LFM2 Vision Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct LFM2VisionIntegrationTests {

    static let modelId = "mlx-community/LFM2-VL-1.6B-4bit"

    @Test("load — LFM2-VL checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await loadModel(Self.modelId)
        }

        // The checkpoint is a VLM — vlModel is present, .imageIn is
        // available, and the text backbone is an LFM2 hybrid engine.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.textEngine.hidden == 2048)  // LFM2-1.6B text hidden
        #expect(m.textEngine.supportsEmbeddingInput)  // VLM splice prerequisite

        let vlm = try #require(m.vlModel)
        // `imageTokenCount` is the NOMINAL square count (256 patches →
        // pixel-unshuffle(2) → 64); the actual per-image token count is
        // derived at native resolution (see `nativeTokenCount`).
        #expect(vlm.imageTokenCount == 64)
        #expect(vlm.imageTokenId == 396)
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await loadModel(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "LFM2-VL checkpoint is not a VLM")

        // Build the LFM2-VL chat prompt:
        //   <|startoftext|><|im_start|>user\n<image>Describe this image.
        //   <|im_end|>\n<|im_start|>assistant\n
        // where `<image>` expands to the per-image vision-token run. Without
        // the chat framing (BOS + im_start/assistant turn) the hybrid LFM2
        // backbone degenerates to whitespace — the vision features alone are
        // correct (they match the reference distribution).
        let imageTokenId = vlm.imageTokenId
        let bos = try #require(
            m.textTokenizer.bosTokenId ?? m.config.bosTokenId, "no BOS token id")
        let userPrefix = m.textTokenizer.encode(text: "<|im_start|>user\n")
        let userSuffix = m.textTokenizer.encode(
            text: "Describe this image.<|im_end|>\n<|im_start|>assistant\n")

        // The shared golden-retriever fixture.
        let image = try VisionTestHelpers.dogImage()

        // Native resolution: the projected vision-token count is per-image
        // (the aspect-preserving smart-resize), so build exactly that many
        // image-placeholder tokens. Encode once up front to size the run;
        // `generate` re-encodes internally.
        let imageTokens = vlm.encodeImage(image).shape[0]
        let promptTokens =
            [bos] + userPrefix
            + Array(repeating: imageTokenId, count: imageTokens)
            + userSuffix

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence first: at least 8 tokens generated, no degenerate
        // repeats — then the content check.
        expectCoherentOutput(
            generated, minTokens: 8,
            label: "LFM2-VL image+text")
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("LFM2-VL generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "LFM2-VL")
    }

    @Test("image + text prompt — STREAMING yields tokens incrementally")
    func imageTextGenerationStreaming() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await loadModel(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "LFM2-VL checkpoint is not a VLM")

        let imageTokenId = vlm.imageTokenId
        let bos = try #require(
            m.textTokenizer.bosTokenId ?? m.config.bosTokenId, "no BOS token id")
        let userPrefix = m.textTokenizer.encode(text: "<|im_start|>user\n")
        let userSuffix = m.textTokenizer.encode(
            text: "Describe this image.<|im_end|>\n<|im_start|>assistant\n")
        let image = try VisionTestHelpers.dogImage()
        let imageTokens = vlm.encodeImage(image).shape[0]
        let promptTokens =
            [bos] + userPrefix
            + Array(repeating: imageTokenId, count: imageTokens)
            + userSuffix

        // Consume the multimodal stream token-by-token. The model owns the
        // splice + decode; chunks carry token ids (decoded to text here via
        // the tokenizer — VisionModel has no tokenizer). The stream yields
        // one token per chunk, then a final stats-bearing chunk.
        let stream = vlm.generateStream(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId,
            eosTokenIds: m.config.eosTokenIds)
        var collected: [Int] = []
        var tokenChunks = 0
        var sawFinalStats = false
        for try await chunk in stream {
            if chunk.stats != nil { sawFinalStats = true }
            if !chunk.tokens.isEmpty { tokenChunks += 1 }
            collected.append(contentsOf: chunk.tokens)
        }
        #expect(sawFinalStats, "stream must end with a stats-bearing chunk")
        #expect(
            tokenChunks == collected.count,
            "streaming must yield exactly one token per chunk")
        expectCoherentOutput(
            collected, minTokens: 8, label: "LFM2-VL streaming")
        let text = m.textTokenizer.decode(tokens: collected, skipSpecialTokens: true)
        print("LFM2-VL streamed: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "LFM2-VL streaming")
    }
}
