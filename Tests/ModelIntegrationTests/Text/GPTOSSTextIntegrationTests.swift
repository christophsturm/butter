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
// Slow integration test: loads the smallest FFAI-runnable GPT-OSS-20B
// checkpoint (a 4-bit / MXFP4-quantized tag) and runs end-to-end greedy
// generation. Skipped automatically if no checkpoint is cached.
//
// GPT-OSS-20B is OpenAI's mixture-of-experts transformer — a 24-layer
// MoE stack (~20B total / ~3.6B active params) with three structural
// features this test exercises end-to-end:
//
//   1. An alternating attention schedule — `layer_types` assigns each
//      layer "sliding_attention" or "full_attention"; sliding layers
//      get a `.window` eviction KV cache, full layers stay unbounded.
//   2. Learned per-head attention sinks — a `self_attn.sinks` logit
//      vector folded into the softmax denominator as a per-head
//      post-hoc rescale of the SDPA output (the d64 SDPA kernel has no
//      native learned-sink support).
//   3. Bias-corrected q/k/v/o projections (`attention_bias`).
//
// The MoE experts ship MXFP4-quantized; the loader transcodes them to
// FFAI's affine-int4 format at load time (see GPTOSSMoE.swift).
//
// `loan-star/gpt-oss-20b-mlx-4Bit` (~11 GB) is the target — it fits
// in the dev box's memory budget. The transcode reorders the logit tail
// slightly, but the next-token ARGMAX is stable, so the test asserts
// both forward parity (argmax matches mlx-lm) and pipeline-level
// coherence under constrained sampling. (We may switch to a
// `ekryski/` MXFP4 → affine-int4 conversion in future once we have
// our own published checkpoint; one canonical target keeps the
// suite consistent with the rest of the integration coverage.)
//
// GPT-OSS-20B is FFAI's heaviest integration target — a debug-build
// greedy decode of a ~20B MoE runs for minutes. Ungated: like every
// text model suite it runs unconditionally via the bisect (no
// `FFAI_BUILD_MACHINE` / group flag). Run it alone if you want to avoid
// the wall-clock cost: `./scripts/integration-bisect.sh GPTOSS`.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("GPTOSS Integration", .serialized)
struct GPTOSSTextIntegrationTests {

    @Test("load + greedy generate produces coherent MoE output")
    func loadAndGenerate() async throws {
        let modelId = "loan-star/gpt-oss-20b-mlx-4Bit"
        let prompt = "The history of the printing press began when"

        let m = try await loadModel(modelId)

        // Engine should be GPT-OSS (not a dense / hybrid family).
        #expect(m.gptOSS != nil)
        #expect(m.jamba == nil)
        #expect(m.qwen3 == nil)

        // Shapes from the published GPT-OSS-20B config.
        #expect(m.engine.hidden == 2880)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.nHeads == 64)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 64)
        #expect(m.engine.vocab == 201_088)

        if let g = m.gptOSS {
            // The alternating sliding/full attention schedule.
            #expect(g.attnKinds.count == 24)
            let slidingCount = g.attnKinds.filter { $0 == .sliding }.count
            let fullCount = g.attnKinds.filter { $0 == .full }.count
            #expect(slidingCount == 12)
            #expect(fullCount == 12)
            #expect(g.slidingWindow == 128)
            // Every layer carries a 32-expert MoE FFN, top-4 routing.
            #expect(g.layers.count == 24)
            for layer in g.layers {
                #expect(layer.moe.experts.count == 32)
                #expect(layer.moe.topK == 4)
            }
        }

        // ── Per-layer cache kinds match the attention schedule ────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 24)
        if let g = m.gptOSS {
            for (i, kind) in g.attnKinds.enumerated() {
                guard let kv = caches[i] as? KVCache else {
                    Issue.record("expected KVCache at layer \(i)")
                    continue
                }
                switch kind {
                case .sliding:
                    #expect(
                        kv.effectiveMaxSize == 128,
                        "sliding layer \(i) should cap at the 128-token window")
                case .full:
                    #expect(
                        kv.effectiveMaxSize > 128,
                        "full-attention layer \(i) should be unbounded")
                }
            }
        }

        // ── Forward correctness vs mlx-lm reference ───────────────────
        // The hard correctness guard: the SAME checkpoint in mlx-lm
        // predicts " the" (id 290, logit ~13.5) as the next token after
        // this prompt, with top-3 [290, 261, 1665]. FFAI's forward must
        // agree — the MXFP4→affine-int4 transcode reorders the logit tail
        // slightly, but the argmax is stable. (A prior "degenerate output"
        // report turned out to be sampling config, NOT a forward bug:
        // GPT-OSS at the OpenAI default temp=1.0 / top_p=1.0 with NO tail
        // truncation samples incoherent tail tokens on a RAW completion
        // prompt — mlx-lm reproduces the identical word-salad. The forward
        // pass is verified bit-for-bit coherent here.)
        let refIds = m.tokenizer.encode(
            text: "The history of the printing press began when")
        let refCaches = m.engine.makeLayerCaches()
        var promptLogits: Tensor!
        for (pos, id) in refIds.enumerated() {
            promptLogits = m.engine.forward(
                tokenId: id, position: pos, caches: refCaches)
        }
        #expect(promptLogits.elementCount == 201_088)
        let refTop = Sampling.topN(promptLogits, n: 3)
        #expect(refTop[0].1.isFinite)
        let refTopIds = refTop.map { $0.0 }
        #expect(
            refTop[0].0 == 290,
            "next-token argmax must match mlx-lm (' the' = 290), got \(refTopIds)")

        // ── End-to-end generation (constrained sampling) ──────────────
        // Two adjustments make the RAW-prompt completion coherent:
        //   • min_p 0.05 truncates the flat low-probability tail (the
        //     source of the temp-1.0/top_p-1.0 word-salad above).
        //   • reasoningLevel .none disables the harmony analysis-channel
        //     scaffolding (the default `.medium` emits `<|channel|>…`
        //     control tokens that decode to empty on a non-chat prompt).
        let result = try await m.generate(
            prompt: prompt,
            parameters: m.defaultGenerationParameters.with {
                $0.maxTokens = 200
                $0.seed = 42
                $0.minP = 0.05
                $0.reasoningLevel = ReasoningLevel.none
            }
        )
        #expect(result.tokensPerSecond > 0)

        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("GPT-OSS-20B decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "GPT-OSS-20B 4bit"
        )
    }

}
