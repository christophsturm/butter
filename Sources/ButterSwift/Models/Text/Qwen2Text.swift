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
// Qwen2Dense — the TEXT component of the Qwen 2 family (the `Qwen2Variant`
// the `Qwen2` root dispatches to). Qwen 2.x ships the 3-series Llama-style
// architecture with one structural difference: the QKV projections carry
// per-output biases, while o_proj does not (the 3-series uses no biases
// anywhere). `loadLinear` in `Sources/Butter/ModelLayers/Linear.swift` detects
// `<base>.bias` automatically, so Qwen 2.x routes through the same
// `Llama3Dense` engine as the 3-series + Mistral7B with no extra plumbing.
//
// Quantized Qwen 2.x (mlx-community 4-bit) is supported transparently: the
// converter KEEPS the additive output biases on the QKV projections
// (`q_proj.bias` next to `q_proj.{weight,scales,biases}`) even after 4-bit
// conversion. `QuantizedLinear` carries an optional `additiveBias` folded
// in after the gemv, and `loadLinear` picks it up when present. Dropping
// these biases gave silent degenerate output (DeepSeek-R1-Distill-Qwen-1.5B
// emitted token-15 "0" forever before this fix). The per-group quant
// `biases` (plural) are a different axis — the dequant offset applied
// during the gemv itself — and have always been handled.

import Foundation

/// The single Qwen 2 text variant. Text-only; reuses the `Llama3Dense`
/// engine (Qwen 2.x is a standard GQA Llama shape with QKV biases).
public enum Qwen2Dense: Qwen2Variant {
    public static let availableCapabilities: Set<Capability> = Capability.textOnly

    /// Qwen 2.x inherits the dense-attention generation defaults (the same
    /// values the family previously got by routing through `Llama3Dense`).
    /// Declared here so the variant owns its parameters per the family
    /// convention; override per-checkpoint via the model card if needed.
    public static let defaultGenerationParameters =
        Llama3Dense.defaultGenerationParameters

    /// Load the Qwen 2 text backbone via the `Llama3Dense` engine. `weights`
    /// must already be scoped to the text sub-tree (e.g. `language_model.`-
    /// prefixed for a Qwen2-VL checkpoint).
    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Qwen2Model {
        try Llama3Dense.loadModel(
            config: config, weights: weights, options: options, device: device)
    }
}
