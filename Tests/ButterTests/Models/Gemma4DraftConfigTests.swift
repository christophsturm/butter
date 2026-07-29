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
// Config recognition for the Gemma 4 Omni EAGLE drafter
// (`gemma4_unified_assistant`). Generation is Phase 8.7; this pins the
// decoded EAGLE shape (the testable part of "load + register" while the
// draft loop is deferred).

import Testing

@testable import ButterSwift

@Suite("Gemma4 Omni Drafter config")
struct Gemma4DraftConfigTests {

    /// A faithful slice of `google/gemma-4-12B-it-assistant`'s config.json.
    private func drafterConfig() -> ModelConfig {
        ModelConfig(
            architecture: "Gemma4UnifiedAssistantForCausalLM",
            modelType: "gemma4_unified_assistant",
            raw: [
                "model_type": "gemma4_unified_assistant",
                "backbone_hidden_size": 3840,
                "num_centroids": 2048,
                "centroid_intermediate_top_k": 32,
                "text_config": [
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 1024,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "num_kv_shared_layers": 4,
                    "intermediate_size": 8192,
                    "vocab_size": 262_144,
                    "tie_word_embeddings": true,
                ],
            ])
    }

    @Test("decodes the EAGLE drafter shape")
    func decodesDrafter() throws {
        let cfg = try Gemma4DraftConfig.decode(drafterConfig())
        #expect(cfg.backboneHiddenSize == 3840)  // pre_projection in = 2·3840
        #expect(cfg.hidden == 1024)
        #expect(cfg.numLayers == 4)
        // All layers KV-shared from the TARGET (the EAGLE wiring → Phase 8.7).
        #expect(cfg.numKvSharedLayers == 4)
        #expect(cfg.numKvSharedLayers == cfg.numLayers)
        #expect(cfg.numCentroids == 2048)
        #expect(cfg.centroidTopK == 32)
        #expect(cfg.vocabSize == 262_144)
    }

    @Test("missing backbone_hidden_size throws")
    func missingThrows() {
        let bad = ModelConfig(
            architecture: "Gemma4UnifiedAssistantForCausalLM",
            modelType: "gemma4_unified_assistant",
            raw: ["text_config": ["hidden_size": 1024, "num_hidden_layers": 4]])
        #expect(throws: (any Error).self) {
            _ = try Gemma4DraftConfig.decode(bad)
        }
    }

    @Test("drafter is recognized by the namespace forwarding")
    func recognized() {
        #expect(Gemma4Draft.architectures.contains("Gemma4UnifiedAssistantForCausalLM"))
        #expect(Gemma4Draft.modelTypes.contains("gemma4_unified_assistant"))
    }
}
