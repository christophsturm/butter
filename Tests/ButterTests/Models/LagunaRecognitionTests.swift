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
// Laguna scaffold smoke tests (S + XS shapes) — no checkpoint, runs in CI. Verifies
// family recognition, hybrid text-config decode (iSWA + MoE + YaRN), and
// default generation parameters. Forward path not implemented yet.

import Foundation
import Testing

@testable import ButterSwift

@Suite("Laguna scaffold")
struct LagunaRecognitionTests {

    @Test("Family is recognized by model_type and architecture")
    func recognition() throws {
        #expect(Laguna.modelTypes.contains("laguna"))
        #expect(Laguna.architectures.contains("LagunaForCausalLM"))

        let cfg = ModelConfig(
            architecture: "LagunaForCausalLM",
            modelType: "laguna",
            raw: ["model_type": "laguna"])
        let v = try Laguna.variant(for: cfg)
        #expect(String(describing: v) == String(describing: LagunaHybrid.self))
    }

    @Test("Hybrid text-config decoder reads S-2.1 shape from a mini fixture")
    func decodesHybridConfig() throws {
        // 4-layer [full, sliding, sliding, sliding] slice with per-layer
        // head counts 48/72/72/72 and nested rope_parameters.
        let raw: [String: Any] = [
            "num_hidden_layers": 4,
            "hidden_size": 3072,
            "vocab_size": 100_352,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "sliding_window": 512,
            "layer_types": [
                "full_attention", "sliding_attention", "sliding_attention",
                "sliding_attention",
            ],
            "num_attention_heads_per_layer": [48, 72, 72, 72],
            "mlp_only_layers": [0],
            "num_experts": 256,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 1024,
            "moe_routed_scaling_factor": 2.5,
            "norm_topk_prob": true,
            "gating": "per-head",
            "rope_parameters": [
                "full_attention": [
                    "rope_theta": 500_000.0,
                    "rope_type": "yarn",
                    "factor": 128.0,
                    "original_max_position_embeddings": 8192,
                    "partial_rotary_factor": 0.5,
                ],
                "sliding_attention": [
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                    "partial_rotary_factor": 1.0,
                ],
            ],
            "max_position_embeddings": 1_048_576,
            "eos_token_id": [2, 24],
        ]
        let cfg = ModelConfig(
            architecture: "LagunaForCausalLM", modelType: "laguna", raw: raw)
        let tc = try LagunaTextConfig.decode(cfg)

        #expect(tc.nLayers == 4)
        #expect(tc.hidden == 3072)
        #expect(tc.headDim == 128)
        #expect(tc.nHeads == 48)
        #expect(tc.nKVHeads == 8)
        #expect(tc.vocab == 100_352)
        #expect(tc.maxSeq == 1_048_576)

        // Layer 0 full (window 0); layers 1-3 slide at 512.
        #expect(tc.perLayerSlidingWindow[0] == 0)
        #expect(tc.perLayerSlidingWindow[1] == 512)
        #expect(tc.isSliding(layer: 0) == false)
        #expect(tc.isSliding(layer: 1) == true)

        // Asymmetric GQA head counts.
        #expect(tc.nHeads(layer: 0) == 48)
        #expect(tc.nHeads(layer: 1) == 72)

        // Layer 0 dense; others MoE.
        #expect(tc.usesMoE(layer: 0) == false)
        #expect(tc.usesMoE(layer: 1) == true)
        #expect(tc.nExperts == 256)
        #expect(tc.nExpertsPerToken == 10)
        #expect(tc.routerScalingFactor == 2.5)
        #expect(tc.useSoftplusHeadwiseAttnGate)

        // RoPE split.
        #expect(tc.yarnFactor == 128)
        #expect(tc.yarnOriginalContext == 8192)
        #expect(tc.fullPartialRotary == 0.5)
        #expect(tc.slidingPartialRotary == 1.0)
        #expect(tc.perLayerRopeTheta[0] == 500_000)
        #expect(tc.perLayerRopeTheta[1] == 10_000)
        #expect(tc.eosTokenIds == [2, 24])
    }

    @Test("Full S-2.1 config fixture decodes (48 layers)")
    func decodesFullFixture() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Models/
            .appendingPathComponent("..")
            .appendingPathComponent("Fixtures")
            .appendingPathComponent("laguna-s-2.1-config.json")
            .standardizedFileURL
        let data = try Data(contentsOf: url)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let cfg = ModelConfig(
            architecture: (obj["architectures"] as? [String])?.first,
            modelType: obj["model_type"] as? String,
            raw: obj)
        let tc = try LagunaTextConfig.decode(cfg)
        #expect(tc.nLayers == 48)
        #expect(tc.nExperts == 256)
        #expect(tc.nExpertsPerToken == 10)
        // 12 full + 36 sliding
        let fullCount = tc.layerTypes.filter { $0 == "full_attention" }.count
        let swaCount = tc.layerTypes.filter { $0 == "sliding_attention" }.count
        #expect(fullCount == 12)
        #expect(swaCount == 36)
        #expect(tc.mlpOnlyLayers == [0])
        #expect(tc.usesMoE(layer: 0) == false)
        #expect(tc.usesMoE(layer: 1) == true)
    }


    @Test("Full XS-2.1 config fixture decodes (40 layers)")
    func decodesFullXSFixture() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Models/
            .appendingPathComponent("..")
            .appendingPathComponent("Fixtures")
            .appendingPathComponent("laguna-xs-2.1-config.json")
            .standardizedFileURL
        let data = try Data(contentsOf: url)
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let cfg = ModelConfig(
            architecture: (obj["architectures"] as? [String])?.first,
            modelType: obj["model_type"] as? String,
            raw: obj)
        let tc = try LagunaTextConfig.decode(cfg)
        #expect(tc.nLayers == 40)
        #expect(tc.hidden == 2048)
        #expect(tc.nExperts == 256)
        #expect(tc.nExpertsPerToken == 8)
        #expect(tc.moeIntermediate == 512)
        let fullCount = tc.layerTypes.filter { $0 == "full_attention" }.count
        let swaCount = tc.layerTypes.filter { $0 == "sliding_attention" }.count
        #expect(fullCount == 10)
        #expect(swaCount == 30)
        #expect(tc.nHeads(layer: 0) == 48)
        #expect(tc.nHeads(layer: 1) == 64)
        #expect(tc.yarnFactor == 32)
    }

    @Test("Default generation parameters follow the convention (no model-specific maxTokens)")
    func defaultGenerationParameters() {
        #expect(
            LagunaHybrid.defaultGenerationParameters.maxTokens
                == GenerationParameters().maxTokens)
        #expect(LagunaHybrid.defaultGenerationParameters.prefillStepSize == 4096)
    }


    @Test("XS-2.1 mini fixture decodes (40L shape: topk=8 mid=512 heads 48/64)")
    func decodesXSMiniConfig() throws {
        let raw: [String: Any] = [
            "num_hidden_layers": 4,
            "hidden_size": 2048,
            "vocab_size": 100_352,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "sliding_window": 512,
            "layer_types": [
                "full_attention", "sliding_attention", "sliding_attention",
                "sliding_attention",
            ],
            "num_attention_heads_per_layer": [48, 64, 64, 64],
            "mlp_only_layers": [0],
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "intermediate_size": 8192,
            "moe_routed_scaling_factor": 2.5,
            "norm_topk_prob": true,
            "gating": "per-head",
            "rope_parameters": [
                "full_attention": [
                    "rope_theta": 500_000.0,
                    "rope_type": "yarn",
                    "factor": 32.0,
                    "original_max_position_embeddings": 8192,
                    "beta_slow": 1.0,
                    "beta_fast": 64.0,
                    "attention_factor": 1.3465735902799727,
                    "partial_rotary_factor": 0.5,
                ],
                "sliding_attention": [
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                    "partial_rotary_factor": 1.0,
                ],
            ],
            "max_position_embeddings": 262_144,
            "eos_token_id": [2, 24],
        ]
        let cfg = ModelConfig(
            architecture: "LagunaForCausalLM", modelType: "laguna", raw: raw)
        let tc = try LagunaTextConfig.decode(cfg)

        #expect(tc.nLayers == 4)
        #expect(tc.hidden == 2048)
        #expect(tc.headDim == 128)
        #expect(tc.nHeads(layer: 0) == 48)
        #expect(tc.nHeads(layer: 1) == 64)
        #expect(tc.nExpertsPerToken == 8)
        #expect(tc.moeIntermediate == 512)
        #expect(tc.denseIntermediate == 8192)
        #expect(tc.yarnFactor == 32)
        #expect(tc.yarnAttentionFactor == Float(1.3465735902799727))
        #expect(tc.yarnBetaFast == 64)
        #expect(tc.maxSeq == 262_144)
        #expect(tc.usesMoE(layer: 0) == false)
        #expect(tc.usesMoE(layer: 1) == true)
    }

    @Test("LagunaError stringifies every case")
    func errorDescriptions() {
        #expect(LagunaError.missingConfig("x").description.contains("Laguna"))
        #expect(LagunaError.missingConfig("x").description.contains("x"))
        #expect(
            LagunaError.unsupportedLayerType("foo").description.contains("foo"))
        #expect(
            LagunaError.notYetImplemented("fwd").description.contains("fwd"))
    }
}
