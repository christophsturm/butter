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
// Phase 1 of the Gemma 4 multimodal build: `Gemma4AudioConfig` parsing.

import Testing

@testable import FFAI

@Suite("Gemma4 audio config")
struct Gemma4AudioConfigTests {

    /// The exact `audio_config` block from `google/gemma-4-E2B-it`.
    private func e2bAudioConfig() -> ModelConfig {
        ModelConfig(
            architecture: "Gemma4ForConditionalGeneration",
            modelType: "gemma4_audio",
            raw: [
                "model_type": "gemma4_audio",
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "output_proj_dims": 1536,
                "rms_norm_eps": 1e-6,
                "attention_chunk_size": 12,
                "attention_context_left": 13,
                "attention_context_right": 0,
                "attention_logit_cap": 50.0,
                "attention_invalid_logits_value": -1_000_000_000.0,
                "conv_kernel_size": 5,
                "subsampling_conv_channels": [128, 32],
                "residual_weight": 0.5,
                "use_clipped_linears": true,
                "hidden_act": "silu",
            ])
    }

    @Test("decodes the E2B audio_config exactly")
    func decodesE2B() throws {
        let cfg = try Gemma4AudioConfig.decode(e2bAudioConfig())
        #expect(cfg.hidden == 1024)
        #expect(cfg.numLayers == 12)
        #expect(cfg.numHeads == 8)
        #expect(cfg.outputProjDims == 1536)  // must match the text hidden dim
        #expect(cfg.rmsNormEps == Float(1e-6))
        #expect(cfg.attentionChunkSize == 12)
        #expect(cfg.attentionContextLeft == 13)
        #expect(cfg.attentionContextRight == 0)
        #expect(cfg.attentionLogitCap == 50.0)
        #expect(cfg.convKernelSize == 5)
        #expect(cfg.subsamplingConvChannels == [128, 32])
        #expect(cfg.residualWeight == 0.5)
        #expect(cfg.useClippedLinears)
        #expect(cfg.hiddenActivation == "silu")
    }

    @Test("missing required keys throw")
    func missingThrows() {
        let bad = ModelConfig(
            architecture: "Gemma4ForConditionalGeneration",
            modelType: "gemma4_audio",
            raw: ["model_type": "gemma4_audio"])  // no hidden_size/layers/heads
        #expect(throws: (any Error).self) {
            _ = try Gemma4AudioConfig.decode(bad)
        }
    }

    @Test("output_proj_dims defaults to hidden when absent")
    func outputProjDefaults() throws {
        let cfg = try Gemma4AudioConfig.decode(
            ModelConfig(
                architecture: "x", modelType: "gemma4_audio",
                raw: ["hidden_size": 1024, "num_hidden_layers": 12, "num_attention_heads": 8]))
        #expect(cfg.outputProjDims == 1024)
        #expect(cfg.useClippedLinears)  // defaults true
        #expect(cfg.subsamplingConvChannels == [128, 32])
    }
}
