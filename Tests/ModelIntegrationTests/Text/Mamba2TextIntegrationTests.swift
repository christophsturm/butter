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
// Slow integration test: downloads (or hits cache) Mamba 2 130M and
// runs end-to-end greedy generation. Skipped automatically if the
// network or checkpoint isn't available.
//
// 130M is the smallest Mamba 2 family checkpoint (~260MB), so the
// integration suite stays fast. The architecture (Mamba2Dense) is the
// same for 370M / 780M / 1.3B / 2.7B — those would be drop-in swaps
// once we want a perf test.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Mamba2 Integration", .serialized
)
struct Mamba2TextIntegrationTests {

    @Test("load + greedy generate produces non-degenerate text")
    func loadAndGenerate() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await loadModel("mlx-community/mamba2-130m")
        }

        // Engine should be Mamba 2 (not Llama / Qwen3).
        #expect(m.mamba2 != nil)
        #expect(m.llama == nil)
        #expect(m.qwen3 == nil)

        // Shapes from the published 130M config.
        #expect(m.textEngine.hidden == 768)
        #expect(m.textEngine.nLayers == 24)
        #expect(m.textEngine.nHeads == 24)
        #expect(m.textEngine.headDim == 64)
        #expect(m.textEngine.vocab == 50_288)
        if let mamba = m.mamba2 {
            #expect(mamba.stateDim == 128)
            #expect(mamba.convKernel == 4)
            #expect(mamba.dInner == 1536)
            // d_inner + 2 * n_groups * state_dim = 1536 + 256 = 1792
            #expect(mamba.convDim == 1792)
        }

        // Forward one BOS-style token. Logits should be finite and
        // non-uniform (the model has been pre-trained — top tokens
        // should pull ahead of the noise floor).
        let caches = m.textEngine.makeLayerCaches()
        let logits = m.textEngine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        // Top logit should be strictly greater than the 5th — degenerate
        // (all-equal) logits would indicate a forward-pass numerical bug.
        #expect(top[0].1 > top[4].1)

        // Greedy generation over a FIXED 64-token window to verify the
        // SSM + conv1d forward + state cache sustain a long decode.
        //
        // `stopOnEOS = false`: Mamba 2 130M is a *base* LM, and on this
        // self-completing pangram it emits `<|endoftext|>` (id 0) after
        // only ~6 fluent tokens (" lazy dog\n\n…") — a correct early
        // termination, not a forward bug. Disabling EOS-stop forces the
        // full window so the test exercises sustained decode rather than
        // the model's natural stop point. At greedy temperature=0 the
        // 130M falls into an ~11-token cycle past ~token 80, so 64 stays
        // inside the fluent range.
        let result = try await m.generate(
            prompt: "The quick brown fox jumps over the",
            parameters: m.defaultGenerationParameters.with {
                $0.maxTokens = 64
                $0.seed = 42
                $0.stopOnEOS = false
            }
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            minUniqueRatio: 0.15,
            label: "Mamba 2 130M"
        )
    }

    /// 4-bit conversion — exercises the FUSED pre-gated-RMSNorm + int4
    /// out_proj GEMV path (`Ops.gatedRmsNormPregateQgemvInt4Fast`). The
    /// bf16 test above takes the unfused fallback (out_proj isn't int4);
    /// only a quantized out_proj triggers the fused kernel, so this is the
    /// only suite that covers it end-to-end. Coherence parity with the
    /// bf16 case confirms the gate-then-norm fusion is mathematically
    /// correct (the kernel is GPU-correctness-tested against a CPU oracle
    /// in iron; this is the in-model integration check).
    @Test("4-bit — fused pre-gated RMSNorm + int4 out_proj decodes coherently")
    func loadAndGenerate4bit() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await loadModel("ekryski/mamba2-130m-4bit")
        }
        #expect(m.mamba2 != nil)
        #expect(m.textEngine.hidden == 768)
        #expect(m.textEngine.nLayers == 24)

        let result = try await m.generate(
            prompt: "The quick brown fox jumps over the",
            parameters: m.defaultGenerationParameters.with { $0.maxTokens = 64; $0.seed = 42 }
        )
        let text = m.textTokenizer.decode(tokens: result.generatedTokens)
        print("Mamba2-130M-4bit generated: \(text)")
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            minUniqueRatio: 0.15,
            label: "Mamba 2 130M 4bit (fused path)"
        )
    }
}
