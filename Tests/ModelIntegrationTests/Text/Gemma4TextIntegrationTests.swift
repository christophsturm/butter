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
// Slow integration test for Gemma 4. First-light verification that the
// family's architectural quirks compose into coherent generated text on
// a real checkpoint:
//
//   • two attention geometries (sliding head_dim 256 + global 512)
//   • ProportionalRoPE on the global layers
//   • value RMSNorm, per-head q/k norms, plain RMSNorm (no Gemma
//     `(1 + weight)` fold — Gemma 4 dropped that convention)
//   • leading `<bos>` prefix (Gemma 4's tokenizer post-processor
//     does not add one — `Gemma4Model.requiresLeadingBOS`)
//   • Per-Layer Embeddings (PLE) — E2B / E4B are PLE variants
//   • per-layer learned scalar
//   • sqrt(hidden) embed scale, GELU MLP, tied embeddings
//   • final logit soft-capping
//
// The smallest Gemma 4 checkpoint on mlx-community is the E2B size,
// which is itself a PLE (Gemma4E) variant — there is no PLE-free dense
// checkpoint below 31B and no MoE checkpoint below 26B, so the dense
// and MoE variants are guard-skipped here (no small/raw checkpoint).
// Tests are skipped entirely if the checkpoint is not available
// locally.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Gemma4 Text Integration", .serialized
)
struct Gemma4TextIntegrationTests {

    // E2B (`gemma-4-e2b-it-4bit`) load + text + image + audio + video +
    // backbone-shape / cache-geometry coverage now lives in the consolidated
    // `Gemma4IntegrationTests` (loaded once there). This suite keeps only the
    // larger sizes that the combined e2b suite can't cover in one load.

    @Test("Gemma4Dense (31B): load + greedy generate produces coherent output")
    func loadAndGenerateDense() async throws {
        // The smallest non-PLE dense Gemma 4 is 31B — only available as
        // a 4-/8-bit quantization. Build-machine only: the checkpoint is
        // large (4-bit ~17 GB) and greedy-decoding 48 tokens is slow.
        // The 5376-wide hidden state routes through the wide-row RMSNorm
        // kernel (the old 4096-row cap is gone). The E2B test above
        // covers the same code paths unconditionally; this is the
        // dense-variant spot check.
        let modelId = "mlx-community/gemma-4-31b-it-4bit"
        let prompt = "Once upon a time, in a quiet village"

        let m = try await loadModel(modelId)
        let gemma4 = m.engine as? Gemma4Model
        #expect(gemma4 != nil)
        #expect(gemma4?.ple == nil, "31B is a dense (non-PLE) variant")

        let result = try await m.generate(
            prompt: prompt,
            parameters: m.defaultGenerationParameters.with { $0.maxTokens = 200; $0.seed = 42 }
        )
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "Gemma 4 31B-it 4bit"
        )
    }

    @Test("Gemma4MoE (26B-A4B): load + greedy generate produces coherent output")
    func loadAndGenerateMoE() async throws {
        // 26B-A4B is the mixture-of-experts variant: every layer runs a
        // shared dense MLP and an 8-of-128 routed expert mixture in
        // parallel, each branch with its own pre/post norm, plus the
        // Gemma 4 router (input RMSNorm + per-expert scale). This test
        // uses the uniformly-8-bit checkpoint; the 4-bit checkpoint
        // mixes 4-/8-bit per layer, now handled by the per-tensor
        // `deriveAffineQuantBits` bit-width derivation. Build-machine
        // only: the 4-bit checkpoint is ~13 GB and slow to greedy-decode.
        let modelId = "mlx-community/gemma-4-26b-a4b-it-4bit"
        let prompt = "The capital of France is"

        let m = try await loadModel(modelId)
        let gemma4 = m.engine as? Gemma4Model
        #expect(gemma4 != nil, "expected Gemma4Model engine")
        // 26B-A4B has hidden_size_per_layer_input = 0 → MoE, not PLE.
        #expect(gemma4?.ple == nil, "26B-A4B is the MoE (non-PLE) variant")

        let result = try await m.generate(
            prompt: prompt,
            parameters: m.defaultGenerationParameters.with { $0.maxTokens = 200; $0.seed = 42 }
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "Gemma 4 26B-A4B-it 4bit"
        )
    }
}
