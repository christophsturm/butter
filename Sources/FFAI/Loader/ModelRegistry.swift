// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// ModelRegistry — config → family dispatch. Decodes a checkpoint's
// `architecture` / `model_type` and routes to the right family loader,
// returning a `Loaded` (engine + capabilities + default generation
// parameters + optional `vlModel`). Owns the vision-language architecture
// set + the Nemotron VL / diffusion-VL detection predicates. The public
// `Model` wrapper (`FFAI/Model.swift`) calls `ModelRegistry.dispatchAndLoad`.

import Foundation
import Metal
import Tokenizers

/// Architecture strings that identify a vision-language checkpoint. A
/// VL checkpoint carries a `vision_config` block and prefixes its text
/// weights under `language_model.*`; the registry recognizes these so a
/// VL load fails with an actionable `visionModelNotIntegrated` error
/// rather than a generic "unsupported architecture".
public enum VisionLanguageArchitectures {
    public static let architectures: Set<String> = [
        "Gemma3ForConditionalGeneration",
        "GlmOcrForConditionalGeneration",
        "Idefics3ForConditionalGeneration",
        "Lfm2VlForConditionalGeneration",
        "LlavaForConditionalGeneration",  // Pixtral-12B
        "LlavaQwen2ForCausalLM",  // FastVLM (Apple FastViTHD + Qwen2)
        "MiniCPMV4_6ForConditionalGeneration",
        "Mistral3ForConditionalGeneration",  // Mistral Small 3.1
        "PaliGemmaForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "SmolVLMForConditionalGeneration",
        // Note: `Gemma4ForConditionalGeneration` is intentionally NOT
        // listed — it is shared by text-only Gemma 4 checkpoints. The
        // `vision_config`-presence check below distinguishes the VL
        // conversion, which the dispatch routes to `Gemma4VL.load`.
    ]

    /// True if `config` describes a VL checkpoint — by architecture
    /// string, by the presence of a `vision_config` block, or by a
    /// VL-only `model_type` (e.g. `paligemma` checkpoints carry the
    /// model_type with or without a vision_config nested block on
    /// partial test fixtures).
    public static func isVisionLanguage(_ config: ModelConfig) -> Bool {
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        if config.nested("vision_config") != nil {
            return true
        }
        // VL-only model_types — none of these have a text-only
        // counterpart, so a bare model_type is enough signal.
        if let mt = config.modelType, Paligemma.modelTypes.contains(mt) {
            return true
        }
        return false
    }
}

/// True if `config` is a Nemotron Nano VL checkpoint — a VL checkpoint
/// (`vision_config` present) whose text backbone is a NemotronH hybrid
/// (its `text_config.model_type` is `nemotron_h`). The Nemotron Nano VL
/// conversion does not carry a single canonical top-level architecture
/// string, so the text backbone's `model_type` is the reliable signal.
func isNemotronVisionLanguage(_ config: ModelConfig) -> Bool {
    guard config.nested("vision_config") != nil,
        let tc = config.nested("text_config")
    else { return false }
    let textModelType = (tc["model_type"] as? String) ?? ""
    if NemotronH.modelTypes.contains(textModelType) { return true }
    let textArch = (tc["architectures"] as? [String])?.first ?? ""
    return NemotronH.architectures.contains(textArch)
}

/// True if `config` is a Nemotron-Labs-Diffusion VLM checkpoint — a VL
/// checkpoint (`vision_config` present) whose top-level model_type /
/// architecture declares the diffusion VLM. Unlike the hybrid Nemotron
/// VL above, the diffusion VLM ships a canonical top-level model_type
/// (`nemotron_labs_diffusion_vlm`) and architecture string
/// (`NemotronLabsDiffusionVLMModel`), so we route on those directly.
func isNemotronDiffusionVisionLanguage(_ config: ModelConfig) -> Bool {
    guard config.nested("vision_config") != nil else { return false }
    if let mt = config.modelType,
        NemotronDiffusionVL.modelTypes.contains(mt)
    {
        return true
    }
    if let arch = config.architecture,
        NemotronDiffusionVL.architectures.contains(arch)
    {
        return true
    }
    return false
}

/// Routes a config to the right family file. Family files declare which
/// architecture / model_type strings they handle. Add a new family by
/// extending `dispatchAndLoad` here.
public enum ModelRegistry {
    /// Engine + the variant-declared generation defaults. The defaults
    /// flow into the `Model` so callers can read them off without
    /// knowing the concrete family.
    public struct Loaded {
        public let engine: any LanguageModel
        public let defaultGenerationParameters: GenerationParameters
        /// Capabilities the loaded variant supports. Text-only families
        /// report `Capability.textOnly`; VL variants add `.imageIn`.
        public let availableCapabilities: Set<Capability>
        /// The composed vision-language model, when the checkpoint is a
        /// VLM. `nil` for text-only families. The `engine` is the VL
        /// model's text backbone, so text-only generation works
        /// regardless; `vlModel` adds the cross-modal image path.
        public let vlModel: VisionModel?

        public init(
            engine: any LanguageModel,
            defaultGenerationParameters: GenerationParameters,
            availableCapabilities: Set<Capability> = Capability.textOnly,
            vlModel: VisionModel? = nil
        ) {
            self.engine = engine
            self.defaultGenerationParameters = defaultGenerationParameters
            self.availableCapabilities = availableCapabilities
            self.vlModel = vlModel
        }
    }

    public static func dispatchAndLoad(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Loaded {
        // Vision-language checkpoints carry a nested `vision_config` and
        // prefix their text weights under `language_model.*`. They route
        // through the VL tier only — a VL config the tier doesn't claim
        // fails with an actionable `visionModelNotIntegrated` rather than
        // falling through to a text loader.
        if VisionLanguageArchitectures.isVisionLanguage(config) {
            guard let entry = visionEntries.first(where: { $0.matches(config) })
            else {
                throw ModelError.visionModelNotIntegrated(
                    config.architecture ?? config.modelType ?? "<unknown>")
            }
            return try entry.load(config, weights, options, device)
        }
        guard let entry = textEntries.first(where: { $0.matches(config) }) else {
            throw ModelError.unsupportedArchitecture(
                config.architecture ?? config.modelType ?? "<unknown>"
            )
        }
        return try entry.load(config, weights, options, device)
    }

    public static func loadLlama3(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Llama3.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    /// Granite 3.x dense (`GraniteForCausalLM`). Same shape as Llama 3
    /// dense, so it reuses `Llama3Dense` — but with the four µP
    /// multipliers folded in via `loadModel(muP:)`. Granite 3 MoE
    /// (`GraniteMoeForCausalLM`) is a distinct arch and does not route
    /// here.
    public static func loadGranite3(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Granite3.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights, options: options, device: device)
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters,
            availableCapabilities: variant.availableCapabilities)
    }

    /// Mistral 7B / Nemo / Small (`MistralForCausalLM`). Same shape as
    /// Llama 3 dense, so it reuses `Llama3Dense` via the `MistralDense`
    /// variant.
    public static func loadMistral(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Mistral.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights, options: options, device: device)
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters,
            availableCapabilities: variant.availableCapabilities)
    }

    /// OLMo 1 (`OlmoForCausalLM`). Pre-norm Llama-shaped; reuses
    /// `Llama3Dense` via the `OLMoDense` variant. OLMo 2 (`olmo2`) is
    /// post-norm with q/k-norm and routes through `loadOLMo2` instead.
    public static func loadOLMo1(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try OLMo.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights, options: options, device: device)
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters,
            availableCapabilities: variant.availableCapabilities)
    }

    /// Yi (`YiForCausalLM`). Same shape as Llama 3 dense, so it reuses
    /// `Llama3Dense` via the `YiDense` variant.
    public static func loadYi(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Yi.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights, options: options, device: device)
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters,
            availableCapabilities: variant.availableCapabilities)
    }

    /// SmolLM 1 / 2 / 3 (`SmolLM*ForCausalLM`). Same shape as Llama 3
    /// dense, so it reuses `Llama3Dense` via the `SmolLMDense` variant.
    public static func loadSmolLM(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try SmolLM.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights, options: options, device: device)
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters,
            availableCapabilities: variant.availableCapabilities)
    }

    /// InternLM2 (`InternLM2ForCausalLM`). Llama-shaped forward; the
    /// `InternLM2Dense` variant's loader remaps InternLM2's tensor names
    /// and splits the fused `wqkv` into q/k/v. See
    /// `Models/Text/InternLM2Text.swift`.
    public static func loadInternLM2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try InternLM2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights, options: options, device: device)
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters,
            availableCapabilities: variant.availableCapabilities)
    }

    /// OLMo 2 (`Olmo2ForCausalLM`). Post-norm + q/k-norm decoder; the
    /// dedicated engine lives in `Models/Text/OLMo2Text.swift`. OLMo 1
    /// stays on `loadLlama3`.
    public static func loadOLMo2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let engine = try OLMo2Dense.loadModel(
            config: config, weights: weights, options: options, device: device)
        return Loaded(
            engine: engine,
            defaultGenerationParameters: OLMo2Dense.defaultGenerationParameters)
    }

    public static func loadStarcoder2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Starcoder2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters,
            availableCapabilities: variant.availableCapabilities)
    }

    public static func loadQwen2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Qwen2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters,
            availableCapabilities: variant.availableCapabilities)
    }

    public static func loadQwen3(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Qwen3.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadPhi(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Phi.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGemma3(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Gemma3.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGemma2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Gemma2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGemma4(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Gemma4.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadMamba2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Mamba2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadFalconH1(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try FalconH1.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadNemotronH(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try NemotronH.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGranite4(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Granite4.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadJamba(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Jamba.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadQwen35(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Qwen35.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGPTOSS(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try GPTOSS.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadLFM2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try LFM2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        // The LFM2.5 collection (128000-token vocab) ships its own
        // recommended sampling defaults (temp 0.2 / top-p 0.8 /
        // repetition-penalty 1.05, per LiquidAI). LFM2 (65536 vocab)
        // keeps the conservative greedy default. Detected by vocab size
        // since both load through the same `lfm2` / `lfm2_moe` variant.
        let isLFM25 = config.vocabSize == 128_000
        let defaults =
            isLFM25 ? lfm25DefaultGenerationParameters : variant.defaultGenerationParameters
        return Loaded(
            engine: engine,
            defaultGenerationParameters: defaults)
    }

    public static func loadNemotronDiffusion(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try NemotronDiffusion.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }
}
