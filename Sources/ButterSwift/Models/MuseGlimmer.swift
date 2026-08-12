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
// Muse Glimmer family root — Meta's Muse Glimmer 30B, a dense causal
// LM distilled from Muse Spark for local agentic use, plus a ~1.8B
// ViT-G/14 perception encoder we do NOT wire yet.
//
// This file is the **main model interface** for the family:
//   • the family enum `MuseGlimmer` (modelTypes, architectures, variant
//     dispatch),
//   • the `MuseGlimmerVariant` protocol every concrete variant
//     conforms to,
//   • the `MuseGlimmerError` type the loader / decode site raises.
//
// Concrete variants + the dense decoder + per-layer impl live under
// `Models/Text/MuseGlimmerText.swift`:
//   - `MuseGlimmerDense` — the (currently only) 30B text decoder.
//   - `MuseGlimmerModel` — the full LanguageModel decoder.
//
// TEXT-ONLY first light. The checkpoint's `architectures` is always
// `MuseGlimmerForConditionalGeneration` (the multimodal wrapper) even
// for text-only use — there is no separate `*ForCausalLM` architecture
// string the way Gemma 3 ships one. We route on that architecture (and
// on the `muse_glimmer` / `muse_glimmer_text` model_type strings as a
// fallback) straight to the text decoder, reading fields from the
// checkpoint's `text_config` sub-object and reading only the
// `language_model.*`-prefixed tensors — `model.vision_tower.*`,
// `model.vision_adapter.*`, `model.vision_projection.*`, and
// `model.perception_emb_norm.*` are silently ignored. Perception
// (image/video) support is a follow-up; so is the DFlash speculative
// drafter.
//
// Architectural notes (verified against the HF
// `modeling_muse_glimmer.py` / `configuration_muse_glimmer.py`, not
// just the announcement blog — several details diverge from a stock
// Gemma 3/4 sandwich-norm decoder):
//
//   1. **52 layers, pattern (3× sliding + 1× full) × 13.** `layer_idx
//      % 4 == 3` is a full-attention layer; the other three of every
//      four are sliding-window (window 2048).
//
//   2. **NoPE on full layers.** Full-attention layers receive *no*
//      rotary embedding at all (not even a different base) — the
//      reference model signals this via `layer_rope_theta[i] == 0`
//      and passes `position_embeddings=None` into attention for those
//      layers. Sliding layers all share one RoPE base (500000).
//
//   3. **Weightless Q/K RMSNorm + an extra scalar on Q only.** Both Q
//      and K get a per-head RMSNorm with NO learned scale
//      (`with_scale=False`); Q is then multiplied by a scalar
//      `qk_scale_factor` (checkpoint value 3.87) — K is not. We fold
//      both behaviours into the existing `RMSNorm` kernel by
//      synthesizing a constant `[headDim]` "weight" tensor (all
//      `qk_scale_factor` for Q, all `1.0` for K) rather than adding a
//      new elementwise-multiply dispatch.
//
//   4. **Sigmoid-gated attention output.** `attn_out = sdpa_out *
//      sigmoid(gate_proj(x))`, where `x` is the SAME
//      `input_layernorm`-normalized hidden state that feeds Q/K/V —
//      not the raw residual, and not the attention output itself.
//      `gate_proj` has its own weights (`self_attn.gate_proj.weight`,
//      shape `[hidden, nHeads*headDim]`, no bias). This is the
//      `Ops.sigmoidMul` primitive Qwen 3.5's gated attention already
//      uses.
//
//   5. **Gemma-style sandwich norm — but with TWO epsilons.**
//      `input_layernorm` / `pre_feedforward_layernorm` use
//      `rms_norm_eps` (1e-5 on the shipped checkpoint);
//      `post_attention_layernorm` / `post_feedforward_layernorm` use a
//      separate, much tighter `post_norm_eps` (1e-8). All four are the
//      "centered" `1 + weight` RMSNorm (HF's
//      `MuseGlimmerTextCenteredRMSNorm`) — the same fold-`+1`-into-
//      the-loaded-weight trick `Gemma3Text.swift` already uses via
//      `loadGemmaRMSNorm`, just called twice with two different `eps`
//      values instead of once.
//
//   6. **The FINAL norm is a *different*, non-centered RMSNorm class**
//      (`weight` used directly, no `+1`) — do not run it through the
//      centered-fold helper.
//
//   7. **No `sqrt(hidden)` embedding scale.** Unlike Gemma 1-4, the
//      embedded token row is passed through a weightless RMSNorm
//      instead (`MuseGlimmerTextNormedEmbedding`) — we synthesize a
//      constant-`1.0` `[hidden]` weight and run it through the same
//      `RMSNorm` kernel Gemma 3 uses for `embedScale`, except as a
//      norm rather than a multiply.
//
//   8. **Gemma2-style final-logit soft-cap, WITH a pre-multiplier.**
//      `logits = cap · tanh(logits · output_multiplier / cap)` where
//      `cap = final_logit_softcapping` (20.0) and `output_multiplier`
//      (≈0.19612, checkpoint-specific) is folded into the argument of
//      `tanh` rather than applied as a separate elementwise pass.
//
//   9. **GQA 32 query / 2 KV heads, head_dim 128** — `head_dim * nHeads
//      (4096) ≠ hidden_size (6656)`, same "head_dim is an independent
//      hyperparameter" shape as Gemma 3/4. `headsPerGroup = 16`; the
//      d128 SDPA decode kernel (dense + sliding/sink fast path) is
//      already generic in `nKVHeads`, no kernel changes needed.
//
// See `Models/Text/MuseGlimmerText.swift` for the concrete decoder.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum MuseGlimmer {
    public static let modelTypes: Set<String> = ["muse_glimmer", "muse_glimmer_text"]
    public static let architectures: Set<String> = ["MuseGlimmerForConditionalGeneration"]

    public static func variant(for config: ModelConfig) throws -> any MuseGlimmerVariant.Type {
        _ = config
        return MuseGlimmerDense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol MuseGlimmerVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> MuseGlimmerModel
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum MuseGlimmerError: Error, CustomStringConvertible {
    case missingConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let field):
            return "MuseGlimmer: required config field missing: \(field)"
        }
    }
}
