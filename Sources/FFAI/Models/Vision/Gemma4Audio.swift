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
// Gemma 4 audio tower (`gemma4_audio`) — a Conformer / USM speech encoder
// that turns a waveform into `[frames, output_proj_dims]` soft tokens
// spliced into the text stream at `audio_token_id` placeholders (between
// the `boa` / `eoa` span markers). This file is PHASE 1 of the multimodal
// build (see `planning/gemma4-multimodal-plan.md`): the config surface.
// The mel front-end, conv subsampling, the 12 Conformer layers (macaron
// FFN + local chunked attention + depthwise-conv module + clipped
// linears), and the cross-modal splice land in subsequent phases.

import Foundation

/// Parsed `audio_config` block of a Gemma 4 multimodal checkpoint
/// (`google/gemma-4-*` / `mlx-community/gemma-4-*`). Mirrors the values the
/// reference `Gemma4AudioEncoder` consumes.
struct Gemma4AudioConfig {
    /// Encoder hidden width (1024 on E2B).
    let hidden: Int
    /// Number of Conformer layers (12 on E2B).
    let numLayers: Int
    /// Self-attention heads (8 on E2B).
    let numHeads: Int
    /// Width the encoder projects to before splicing into the text stream
    /// — must equal the text backbone's hidden size (1536 on E2B).
    let outputProjDims: Int
    let rmsNormEps: Float

    // ── Local chunked self-attention geometry ──
    /// Frames per attention chunk.
    let attentionChunkSize: Int
    /// Extra left-context frames each chunk may attend to.
    let attentionContextLeft: Int
    /// Extra right-context frames (0 = causal local window on E2B).
    let attentionContextRight: Int
    /// tanh logit soft-cap applied to attention scores.
    let attentionLogitCap: Float
    /// Fill value for masked-out (out-of-window) attention positions.
    let attentionInvalidLogitsValue: Float

    // ── Conv module + subsampling ──
    /// Depthwise conv kernel width in the conv module (5 on E2B).
    let convKernelSize: Int
    /// Channel widths of the two mel-subsampling conv layers ([128, 32]).
    let subsamplingConvChannels: [Int]

    /// Macaron-FFN residual scale (0.5 on E2B).
    let residualWeight: Float
    /// Whether the audio linears carry per-tensor input/output clipping
    /// ranges (the Gemma-4-audio "clipped linear" scheme).
    let useClippedLinears: Bool
    /// Hidden activation ("silu" on E2B).
    let hiddenActivation: String

    static func decode(_ c: ModelConfig) throws -> Gemma4AudioConfig {
        guard let hidden = c.int("hidden_size"),
            let numLayers = c.int("num_hidden_layers"),
            let numHeads = c.int("num_attention_heads")
        else {
            throw Gemma4Error.missingConfig("audio_config")
        }
        return Gemma4AudioConfig(
            hidden: hidden,
            numLayers: numLayers,
            numHeads: numHeads,
            outputProjDims: c.int("output_proj_dims") ?? hidden,
            rmsNormEps: Float(c.float("rms_norm_eps") ?? 1e-6),
            attentionChunkSize: c.int("attention_chunk_size") ?? 12,
            attentionContextLeft: c.int("attention_context_left") ?? 13,
            attentionContextRight: c.int("attention_context_right") ?? 0,
            attentionLogitCap: Float(c.float("attention_logit_cap") ?? 50.0),
            attentionInvalidLogitsValue: Float(
                c.float("attention_invalid_logits_value") ?? -1e9),
            convKernelSize: c.int("conv_kernel_size") ?? 5,
            subsamplingConvChannels: c.intArray("subsampling_conv_channels") ?? [128, 32],
            residualWeight: Float(c.float("residual_weight") ?? 0.5),
            useClippedLinears: c.bool("use_clipped_linears") ?? true,
            hiddenActivation: c.string("hidden_act") ?? "silu")
    }
}
