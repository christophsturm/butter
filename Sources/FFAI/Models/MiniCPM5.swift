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
// MiniCPM5 — OpenBMB's MiniCPM5 family entry point. The only v5 model today
// is MiniCPM5-1B, a text-only checkpoint that declares a plain
// `LlamaForCausalLM` / `model_type: llama` architecture — indistinguishable
// from a generic Llama at the architecture level. The loader therefore
// routes here by a `_name_or_path` heuristic (see `handles`).
//
// This is intentionally a thin facade over the Llama stack today: it
// delegates text loading to `MiniCPM5Text`. The named entrypoint exists so
// that when multimodal v5 checkpoints arrive (OpenBMB ships omni variants
// of each generation), they compose vision / audio components here the same
// way `MiniCPM4` does — without re-plumbing the dispatch.

import Foundation

public enum MiniCPM5 {
    /// MiniCPM5-1B is text-only. Multimodal v5 variants will widen this.
    public static let availableCapabilities: Set<Capability> = Capability.textOnly

    /// Whether a decoded `config.json` describes a MiniCPM5 checkpoint.
    ///
    /// MiniCPM5-1B declares `architectures: ["LlamaForCausalLM"]` +
    /// `model_type: "llama"`, so there's no architecture marker to match —
    /// we route by the `_name_or_path` field the conversion preserves
    /// (`"MiniCPM5-1B-MLX"`). Checked BEFORE the generic Llama branch so a
    /// MiniCPM5 checkpoint lands on this named entrypoint; every other Llama
    /// checkpoint (no "minicpm5" marker) falls through unchanged.
    public static func handles(_ config: ModelConfig) -> Bool {
        let name =
            config.string("_name_or_path") ?? config.string("name_or_path") ?? ""
        return name.lowercased().contains("minicpm5")
    }

    /// Load a MiniCPM5 checkpoint. Delegates the text engine to
    /// `MiniCPM5Text`; the family is text-only today.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> ModelRegistry.Loaded {
        let engine = try MiniCPM5Text.load(
            config: config, weights: weights, options: options, device: device)
        return ModelRegistry.Loaded(
            engine: engine,
            defaultGenerationParameters:
                try Llama.variant(for: config).defaultGenerationParameters,
            availableCapabilities: availableCapabilities)
    }
}
