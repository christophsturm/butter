// Copyright 2026 Tom Turney (@TheTom)
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
// Slow integration test for Hy3 (Tencent Hunyuan hy_v3). First-light
// verification that the family's architectural pieces — sigmoid+bias
// (Path B) router, ungated shared expert, QK-RMSNorm before RoPE —
// compose into coherent generated text on a real checkpoint.
//
// Hy3 is a 295B MoE checkpoint (oQ2 weights ≈ 75-92 GB), so this test
// is gated on a local checkpoint supplied via the `BUTTER_HY3_MODEL_PATH`
// env var rather than an HF download. Skips quietly when unset — no
// machine-specific default, matching `AuraKLDIntegrationTests`.

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("Hy3 Integration", .serialized)
struct Hy3IntegrationTests {

    @Test("load + config basics + generation defaults + greedy generate")
    func loadAndGenerate() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["BUTTER_HY3_MODEL_PATH"] else {
            print("[skipped] Hy3IntegrationTests: BUTTER_HY3_MODEL_PATH not set")
            return
        }

        // (a) Load via the family-dispatch registry (Model.load routes
        // through ModelRegistry.dispatchAndLoad, which resolves Hy3 by
        // architecture/model_type — see Loader/Model.swift).
        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelPath) }

        // (b) Shapes/config basics. 80 layers is the canonical Hy3
        // depth per the HF config; fall back to whatever the checkpoint
        // actually advertises so this doesn't false-fail on a
        // scaled-down test fixture.
        let expectedLayers = m.config.numLayers ?? 80
        #expect(m.textEngine.nLayers == expectedLayers)
        if let hidden = m.config.hiddenSize {
            #expect(m.textEngine.hidden == hidden)
        }
        if let vocab = m.config.vocabSize {
            #expect(m.textEngine.vocab == vocab)
        }
        let hy3 = m.engine as? Hy3Model
        #expect(hy3 != nil, "expected Hy3Model engine")

        // (c) Generation defaults — Hy3Variant's Tencent recipe:
        // temperature 0.9, top_p 1.0.
        #expect(m.defaultGenerationParameters.temperature == 0.9)
        #expect(m.defaultGenerationParameters.topP == 1.0)

        // (d) Greedy decode from a short prompt must produce non-empty,
        // non-repeating output — exercises the router (sigmoid+bias
        // top-K), expert-sorted gather QMM, and shared-expert fuse path
        // end-to-end.
        let prompt = "The capital of France is"
        let maxTokens = 16
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        print("Hy3 decoded: \"\(prompt)\(result.text)\"")
        #expect(!result.generatedTokens.isEmpty)
        expectCoherentOutput(
            result.generatedTokens, minTokens: maxTokens,
            label: "Hy3")
    }
}
