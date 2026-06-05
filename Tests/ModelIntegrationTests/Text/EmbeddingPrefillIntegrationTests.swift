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
// Numerics guard for the batched VLM-prefill primitive
// `LanguageModel.forwardMulti(inputEmbeddings:)` — the entry point a
// `VisionModel` uses to prefill the spliced embedding stream in one (or a
// few) command buffers instead of one dispatch per position.
//
// The VLM prefill must be numerically faithful to the per-position
// `forward(inputEmbedding:)` loop it replaced. This exercises the
// **fused** Llama / Qwen 2 path (one `Ops.sdpaMulti(causal: true)` per
// layer + batched projections): Qwen 2.5 1.5B is the smallest loadable
// checkpoint with the same backbone shape as Qwen 2-VL 2B —
// `head_dim = 128`, which is the gate for the fused `ffai_sdpa_multi`
// kernel. (Qwen 2-VL 2B itself can't be loaded yet: a pre-existing
// quant-load crash blocks its e2e suite, independent of the prefill
// batching this guards.)
//
// A text-only embedding stream (text-token rows only) is sufficient — the
// engine code path is identical whether a row came from `textEmbedding`
// or a vision encoder; only the row's provenance differs.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("Embedding-Prefill Numerics (batched == per-position)", .serialized)
struct EmbeddingPrefillIntegrationTests {

    /// Qwen 2.5 1.5B Instruct 4-bit: hidden 1536, 12 heads → head_dim 128.
    /// Same Llama backbone shape as Qwen 2-VL 2B, so it drives the fused
    /// `forwardMulti(inputEmbeddings:)` SDPA path (not the head_dim-64
    /// fallback the 0.5B would take).
    static let modelId = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

    @Test("batched forwardMulti(inputEmbeddings:) matches per-position forward(inputEmbedding:)")
    func batchedEmbeddingPrefillMatchesPerPosition() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await loadModel(Self.modelId)
        }
        let engine = m.textEngine
        #expect(engine.supportsEmbeddingInput, "Qwen 2.5 backbone must support the splice")
        #expect(engine.headDim == 128, "this guard targets the fused head_dim-128 path")

        // Build a text-only embedding stream — the same kind of contiguous
        // `[seqLen, hidden]` block `VisionModel.assembleEmbeddingBlock`
        // packs, minus the vision rows.
        let promptTokens = m.textTokenizer.encode(
            text: "A photograph of a dog standing in a grassy field near a calm lake at dawn.")
        precondition(
            promptTokens.count >= 4,
            "embedding-prefill guard: prompt encoded to \(promptTokens.count) tokens; need ≥ 4")
        let stream = promptTokens.map {
            engine.textEmbedding(tokenId: $0, device: .shared)
        }

        let hidden = engine.hidden
        let dtype = stream[0].dtype
        let rowBytes = hidden * dtype.byteSize
        let block = Tensor.empty(
            shape: [stream.count, hidden], dtype: dtype, device: .shared)
        let dstBase = block.buffer.contents().advanced(by: block.offset)
        for (i, row) in stream.enumerated() {
            memcpy(
                dstBase.advanced(by: i * rowBytes),
                row.buffer.contents().advanced(by: row.offset),
                rowBytes)
        }

        // ── Reference: per-position embedding forward, committing per
        // position (the path that shipped before the batched prefill).
        let refCaches = engine.makeLayerCaches()
        var refTail: Tensor!
        for (i, row) in stream.enumerated() {
            runAndWait { cmd in
                refTail = engine.forward(
                    inputEmbedding: row, position: i,
                    caches: refCaches, on: cmd, device: .shared)
            }
        }
        let refLogits = refTail.toFloatArray()
        let refArgmax = refLogits.enumerated().max(by: { $0.element < $1.element })!.offset

        // ── Batched: one `forwardMulti(inputEmbeddings:)` over the block.
        let manyCaches = engine.makeLayerCaches()
        var manyTail: Tensor!
        runAndWait { cmd in
            manyTail = engine.forwardMulti(
                inputEmbeddings: block, startingAt: 0,
                pleTokenIds: nil, caches: manyCaches, on: cmd, device: .shared)
        }
        let manyLogits = manyTail.toFloatArray()
        let manyArgmax = manyLogits.enumerated().max(by: { $0.element < $1.element })!.offset

        // Same assertion convention as the text `forwardMany` equivalence
        // test: exact argmax match + top-1 logit within 0.5.
        print(
            "embedding-prefill T=\(stream.count): ref argmax=\(refArgmax) "
                + "batched argmax=\(manyArgmax)")
        let refTop5 = refLogits.enumerated().sorted { $0.element > $1.element }.prefix(5)
            .map { (id: $0.offset, logit: $0.element) }
        let manyTop5 = manyLogits.enumerated().sorted { $0.element > $1.element }.prefix(5)
            .map { (id: $0.offset, logit: $0.element) }
        print("  ref  top5: \(refTop5)")
        print("  many top5: \(manyTop5)")

        #expect(
            refArgmax == manyArgmax,
            "batched embedding-prefill argmax \(manyArgmax) ≠ per-position \(refArgmax)")
        let absDelta = abs(refLogits[refArgmax] - manyLogits[manyArgmax])
        #expect(
            absDelta < 0.5,
            "batched embedding-prefill top-1 logit drifted \(absDelta) from per-position")
    }
}
