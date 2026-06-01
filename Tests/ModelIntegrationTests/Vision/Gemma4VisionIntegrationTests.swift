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
// Slow integration test for Gemma 4 VL (the bespoke Gemma 4 ViT tower +
// multi-modal embedder + Gemma 4 text backbone, the
// `Gemma4ForConditionalGeneration` checkpoint).
//
// Verifies the vision path end-to-end on a real checkpoint: the Gemma 4
// vision tower loads, runs its RoPE-attention forward and
// attention-pooling head, the multi-modal embedder projects the pooled
// soft tokens into the text hidden dim, the cross-modal splice injects
// them, and the fused stream decodes coherent text through the Gemma 4
// backbone (which now supports embedding-input forward for the splice).
//
// Uses the mlx-community gemma-4-e2b-it 4-bit conversion — the smallest
// published Gemma 4 VL (its `config.json` carries a `vision_config`, so
// it routes through the Gemma4VL loader). The checkpoint MUST load — a
// load failure fails the test.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Gemma4 Vision Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct Gemma4VisionIntegrationTests {

    static let modelId = "mlx-community/gemma-4-e2b-it-4bit"

    @Test("load — Gemma 4 VL checkpoint loads with vision capability")
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
        // The vision tower contributes a positive run of soft tokens.
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
        let vlm = try #require(m.vlModel, "Gemma 4 VL checkpoint is not a VLM")

        // Build the Gemma 4 chat prompt with the image span INSIDE the
        // user turn: `<bos><start_of_turn>user\n` + boi + <image>×N + eoi +
        // `\nDescribe this image.<end_of_turn>\n<start_of_turn>model\n`.
        let imageTokenId = vlm.imageTokenId
        // Gemma 4 is BOS-critical (requiresLeadingBOS) and its tokenizer
        // post-processor does NOT add one — the multimodal generate path
        // doesn't auto-prepend it the way Generate.swift does, so the
        // prompt carries an explicit leading <bos>.
        let bos = try #require(
            m.tokenizer.bosTokenId ?? m.config.bosTokenId, "no BOS token id")
        // Gemma 4 wraps the image soft-token run in begin/end-of-image
        // markers (boi 255999, eoi 258882 — see mlx-vlm
        // processing_gemma4.py `full_image_sequence`).
        let boiTokenId = 255999
        let eoiTokenId = 258882
        let userPrefix = m.tokenizer.encode(text: "<start_of_turn>user\n")
        let userSuffix = m.tokenizer.encode(
            text: "\nDescribe this image.<end_of_turn>\n<start_of_turn>model\n")

        // A real photograph — the golden-retriever fixture. Gemma 4's image
        // processor is variable-resolution, so encode first to learn the
        // exact soft-token count and size the placeholder run to match.
        let image = try VisionTestHelpers.dogImage()
        let nImageTokens = vlm.encodeImage(image).shape[0]
        let imageSpan =
            [boiTokenId]
            + Array(repeating: imageTokenId, count: nImageTokens)
            + [eoiTokenId]
        let promptTokens =
            [bos] + userPrefix + imageSpan + userSuffix

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence first, then the content check: the caption should
        // mention a dog.
        expectCoherentOutput(
            generated, minTokens: 8,
            label: "Gemma 4 VL image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 4 VL generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "Gemma 4 VL")
    }

    /// Validate the variable-resolution vision tower against the mlx-vlm
    /// oracle: feed FFAI's `encode` the EXACT preprocessed pixels mlx-vlm
    /// produced for the dog fixture (`[3, 672, 912]`, aspect-ratio resized
    /// to a 42×57 patch grid → 266 soft tokens) and compare the
    /// `[266, 1536]` image soft tokens to the reference `embed_vision`
    /// output. Isolates the tower (patch-embed → blocks → pool → embed)
    /// from preprocessing. Regenerate fixtures with
    /// `gemma4-benchmark/dump_gemma4_vision.py` + `/tmp/dump_pixels.py`.
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
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "not a VLM")

        // Build the preprocessed-pixel tensor [1, 3, 672, 912] and run the
        // tower directly (bypassing FFAI preprocessing — this test gates
        // the tower math, not the resize).
        let gridH = 42, gridW = 57, patch = 16
        let imgH = gridH * patch, imgW = gridW * patch  // 672 × 912
        let pixels = readF32(pixelsPath)
        #expect(pixels.count == 3 * imgH * imgW)
        let pixT = Tensor.empty(shape: [1, 3, imgH, imgW], dtype: .f32)
        pixT.copyIn(from: pixels)
        let soft = vlm.visionEncoder.encode(image: pixT)
        let got = soft.toFloatArray()
        let oracle = readF32(embedPath)
        print("FFAI soft \(soft.shape), oracle count \(oracle.count)")
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
        // The variable-resolution tower now matches the mlx-vlm oracle:
        // correct 266-token grid AND numerically faithful (per-block
        // correlation 1.0 against the reference after the RMSNorm fold fix +
        // clipped-linears). A small residual remains from the 4-bit
        // quantized weights + bf16 accumulation over 16 blocks.
        #expect(
            meanDiff < 0.05,
            "vision tower diverges from the oracle (mean \(meanDiff))")
    }
}
