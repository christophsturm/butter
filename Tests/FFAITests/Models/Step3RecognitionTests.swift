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
// Step-3 scaffold smoke tests — no checkpoint, runs in CI. Verifies the
// family is recognized (model_type / architecture sets + variant
// dispatch), the hybrid text-config decoder reads the per-layer
// full/sliding pattern, and the default generation parameters follow
// the convention (no model-specific maxTokens). The forward path is not
// implemented yet (loads throw `Step3Error.notYetImplemented`); end-to-end
// generation lands in follow-ups.

import Foundation
import Testing

@testable import FFAI

@Suite("Step-3 scaffold")
struct Step3RecognitionTests {

    @Test("Family is recognized by model_type and architecture")
    func recognition() throws {
        #expect(Step3.modelTypes.contains("step3p7"))
        #expect(Step3.modelTypes.contains("step3p5"))
        #expect(!Step3.architectures.isEmpty)
        #expect(!Step3.vlArchitectures.isEmpty)

        // A step3p7 text config dispatches to a Step-3 variant (not an
        // "unsupported architecture" error).
        let cfg = ModelConfig(
            architecture: nil, modelType: "step3p7", raw: ["model_type": "step3p7"])
        #expect(throws: Never.self) { _ = try Step3.variant(for: cfg) }
    }

    @Test("Hybrid text-config decoder reads the full/sliding layer pattern")
    func decodesHybridConfig() throws {
        // 4-layer [full, sliding, sliding, sliding] stack with a distinct
        // SWA attention shape — the Step-3 hybrid marker.
        let raw: [String: Any] = [
            "num_hidden_layers": 4,
            "hidden_size": 4096,
            "vocab_size": 128_000,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "sliding_window": 512,
            "layer_types": [
                "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            ],
            "attention_other_setting": ["num_attention_heads": 96, "num_key_value_heads": 8],
        ]
        let cfg = ModelConfig(architecture: nil, modelType: "step3p7", raw: raw)
        let tc = try Step3TextConfig.decode(cfg)

        #expect(tc.nLayers == 4)
        #expect(tc.hidden == 4096)
        #expect(tc.headDim == 128)
        #expect(tc.nHeads == 64)
        // Layer 0 is full-attention (window 0); layers 1-3 slide at 512.
        #expect(tc.perLayerSlidingWindow[0] == 0)
        #expect(tc.perLayerSlidingWindow[1] == 512)
        // SWA layers override the head count (96 vs the full-attn 64).
        #expect(tc.perLayerHeads[1] == 96)
    }

    @Test("Default generation parameters follow the convention (no model-specific maxTokens)")
    func defaultGenerationParameters() {
        #expect(Step3Hybrid.defaultGenerationParameters.maxTokens == GenerationParameters().maxTokens)
        #expect(Step3Hybrid.defaultGenerationParameters.prefillStepSize == 4096)
    }
}
