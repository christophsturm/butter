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
// Config parsing for Gemma 4 Omni (encoder-free `gemma4_unified`): the text
// tower resolves to `Gemma4Dense`, the encoder-free `Gemma4OmniVisionConfig`
// decodes from `vision_config`, and the placeholder ids match the checkpoint.

import Testing

@testable import ButterSwift

@Suite("Gemma4 Omni config")
struct Gemma4OmniConfigTests {

    /// A faithful slice of `google/gemma-4-12B-it`'s config.json — the bits
    /// the loader dispatches + decodes on.
    private func omniConfig() -> ModelConfig {
        ModelConfig(
            architecture: "Gemma4UnifiedForConditionalGeneration",
            modelType: "gemma4_unified",
            raw: [
                "model_type": "gemma4_unified",
                "image_token_id": 258_880,
                "audio_token_id": 258_881,
                "video_token_id": 258_884,
                "boi_token_id": 255_999,
                "eoi_token_id": 258_882,
                "boa_token_id": 256_000,
                "eoa_token_index": 258_883,
                "text_config": [
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 3840,
                    "num_hidden_layers": 48,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "num_global_key_value_heads": 1,
                    "enable_moe_block": false,
                    "hidden_size_per_layer_input": 0,
                    "final_logit_softcapping": 30.0,
                    "tie_word_embeddings": true,
                    "sliding_window": 1024,
                    "vocab_size": 262_144,
                ],
                "vision_config": [
                    "model_type": "gemma4_unified_vision",
                    "mm_embed_dim": 3840,
                    "model_patch_size": 48,
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "num_soft_tokens": 280,
                    "mm_posemb_size": 1120,
                    "rms_norm_eps": 1e-6,
                ],
                "audio_config": [
                    "model_type": "gemma4_unified_audio",
                    "audio_embed_dim": 640,
                    "audio_samples_per_token": 640,
                    "rms_norm_eps": 1e-6,
                ],
            ])
    }

    @Test("omni text tower resolves to the Gemma4Dense variant")
    func textVariantIsDense() throws {
        let variant = try Gemma4.variant(for: omniConfig())
        #expect(
            ObjectIdentifier(variant) == ObjectIdentifier(Gemma4Dense.self),
            "expected Gemma4Dense for the encoder-free omni text_config")
    }

    @Test("vision_config decodes the encoder-free embedder shape")
    func visionConfigDecode() throws {
        let vc = try #require(omniConfig().subConfig("vision_config"))
        let cfg = try Gemma4OmniVisionConfig.decode(vc)
        #expect(cfg.modelPatchSize == 48)  // patch_dense input = 3·48² = 6912
        #expect(cfg.patchSize == 16)
        #expect(cfg.poolingKernelSize == 3)  // resize divisibility = 16·3 = 48
        #expect(cfg.hidden == 3840)
        #expect(cfg.numSoftTokens == 280)
        #expect(cfg.posEmbSize == 1120)
        #expect(cfg.layerNormEps == Float(1e-6))
    }

    @Test("vision_config requires mm_embed_dim + model_patch_size")
    func visionConfigMissingThrows() {
        let bad = ModelConfig(
            architecture: "x", modelType: "gemma4_unified_vision",
            raw: ["patch_size": 16])  // no mm_embed_dim / model_patch_size
        #expect(throws: (any Error).self) {
            _ = try Gemma4OmniVisionConfig.decode(bad)
        }
    }

    @Test("placeholder ids read from config, with the checkpoint defaults")
    func placeholderIds() {
        let c = omniConfig()
        #expect(c.int("image_token_id") == 258_880)
        #expect(c.int("audio_token_id") == 258_881)
        #expect(c.int("video_token_id") == 258_884)
        // The family defaults match the checkpoint ids (so a re-host that
        // drops the explicit fields still wires the right placeholders).
        #expect(Gemma4Omni.defaultImageTokenId == 258_880)
        #expect(Gemma4Omni.defaultAudioTokenId == 258_881)
        #expect(Gemma4Omni.defaultVideoTokenId == 258_884)
    }

    @Test("drafter strings are recognized but kept out of the generate sets")
    func drafterRecognition() {
        // The drafter must NOT auto-route to standalone text generation.
        #expect(!Gemma4.architectures.contains("Gemma4UnifiedAssistantForCausalLM"))
        #expect(!Gemma4.modelTypes.contains("gemma4_unified_assistant"))
        #expect(Gemma4.drafterArchitectures.contains("Gemma4UnifiedAssistantForCausalLM"))
        #expect(Gemma4.drafterModelTypes.contains("gemma4_unified_assistant"))
        // The omni model itself routes through the Gemma4 family.
        #expect(Gemma4.architectures.contains("Gemma4UnifiedForConditionalGeneration"))
        #expect(Gemma4.modelTypes.contains("gemma4_unified"))
    }

    @Test("omni default generation params match the Gemma card (temp 1.0 / p .95 / k 64)")
    func defaultGenerationParameters() {
        let p = Gemma4Dense.defaultGenerationParameters
        #expect(p.temperature == 1.0)
        #expect(p.topP == 0.95)
        #expect(p.topK == 64)
    }
}
