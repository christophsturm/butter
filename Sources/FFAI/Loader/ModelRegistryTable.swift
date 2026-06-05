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
// ModelRegistry table — the data-driven dispatch list that replaces the
// former hand-written `dispatchAndLoad` if-chain. Two tiers, matching the
// original control flow exactly:
//
//   • `visionEntries` — consulted ONLY when the config is a vision-language
//     checkpoint (`VisionLanguageArchitectures.isVisionLanguage`). A VL
//     config that matches none of these throws `visionModelNotIntegrated`
//     (it never falls through to text).
//   • `textEntries` — consulted for every non-VL config; no match throws
//     `unsupportedArchitecture`.
//
// Entry order within each tier preserves the original precedence. Each
// `load` closure is the family's former branch body verbatim — it
// self-validates config + weights and throws a typed family error.
//
// The `summary` / `repos` catalog fields are populated in the next phase
// (the `ffai models` catalog reads this same table); they're left empty
// here so this phase stays a behavior-preserving dispatch swap.

import Foundation

extension ModelRegistry {

    // ─── Vision-language tier ────────────────────────────────────────

    static let visionEntries: [ModelEntry] = [
        ModelEntry(
            family: "Gemma 3 VL", kind: .vlm,
            architectures: ["Gemma3ForConditionalGeneration"],
            load: { c, w, o, d in
                let vlm = try Gemma3VL.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Gemma3Dense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "LFM2-VL", kind: .vlm,
            architectures: ["Lfm2VlForConditionalGeneration"],
            load: { c, w, o, d in
                let vlm = try LFM2VL.load(config: c, weights: w, options: o, device: d)
                let lfm2DefaultParams = GenerationParameters(
                    maxTokens: 256, prefillStepSize: 1024,
                    temperature: 0.0, topP: 1.0, topK: 0,
                    minP: 0.0, repetitionPenalty: 1.0)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: lfm2DefaultParams,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "Qwen 2.5-VL", kind: .vlm,
            architectures: ["Qwen2_5_VLForConditionalGeneration"],
            load: { c, w, o, d in
                let vlm = try Qwen25VL.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Qwen25VL.defaultGenerationParameters,
                    availableCapabilities: Qwen25VL.availableCapabilities,
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "Qwen 2-VL", kind: .vlm,
            architectures: ["Qwen2VLForConditionalGeneration"],
            load: { c, w, o, d in
                let vlm = try Qwen2VL.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Qwen2VL.defaultGenerationParameters,
                    availableCapabilities: Qwen2VL.availableCapabilities,
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "Qwen 3-VL", kind: .vlm,
            architectures: ["Qwen3VLForConditionalGeneration"],
            load: { c, w, o, d in
                let vlm = try Qwen3VL.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Qwen3Dense.defaultGenerationParameters,
                    availableCapabilities: Qwen3VL.availableCapabilities,
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "Qwen 3-VL-MoE", kind: .vlm,
            architectures: ["Qwen3VLMoeForConditionalGeneration"],
            load: { c, w, o, d in
                let vlm = try Qwen3VLMoe.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Qwen35Hybrid.defaultGenerationParameters,
                    availableCapabilities: Qwen3VLMoe.availableCapabilities,
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "Gemma 4 VL", kind: .vlm,
            architectures: ["Gemma4ForConditionalGeneration"],
            load: { c, w, o, d in
                let vlm = try Gemma4VL.load(config: c, weights: w, options: o, device: d)
                // Gate `.audioIn` / `.videoIn` on what the checkpoint wired.
                var caps = Capability.textOnly.union([.imageIn])
                if vlm.supportsAudio { caps.insert(.audioIn) }
                if vlm.videoTokenId != nil { caps.insert(.videoIn) }
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Gemma4Dense.defaultGenerationParameters,
                    availableCapabilities: caps,
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "Gemma 4 Omni", kind: .omni,
            modelTypes: ["gemma4_unified"],
            architectures: ["Gemma4UnifiedForConditionalGeneration"],
            load: { c, w, o, d in
                let vlm = try Gemma4Omni.load(config: c, weights: w, options: o, device: d)
                var caps = Capability.textOnly.union([.imageIn])
                if vlm.supportsAudio { caps.insert(.audioIn) }
                if vlm.videoTokenId != nil { caps.insert(.videoIn) }
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Gemma4Dense.defaultGenerationParameters,
                    availableCapabilities: caps,
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "FastVLM", kind: .vlm,
            modelTypes: FastVLM.modelTypes, architectures: FastVLM.architectures,
            load: { c, w, o, d in
                let vlm = try FastVLM.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: FastVLM.defaultGenerationParameters,
                    availableCapabilities: FastVLM.availableCapabilities,
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "Pixtral", kind: .vlm,
            modelTypes: Pixtral.modelTypes,
            load: { c, w, o, d in
                let vlm = try Pixtral.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Pixtral.defaultGenerationParameters,
                    availableCapabilities: Pixtral.availableCapabilities,
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "Mistral 3 VL", kind: .vlm,
            modelTypes: Mistral3.modelTypes,
            load: { c, w, o, d in
                let vlm = try Mistral3.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Mistral3.defaultGenerationParameters,
                    availableCapabilities: Mistral3.availableCapabilities,
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "GLM-OCR", kind: .ocr,
            modelTypes: GlmOcr.modelTypes, architectures: GlmOcr.architectures,
            load: { c, w, o, d in
                let variant = try GlmOcr.variant(for: c)
                let m = try variant.loadModel(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: m,
                    defaultGenerationParameters: variant.defaultGenerationParameters,
                    availableCapabilities: variant.availableCapabilities)
            }),
        ModelEntry(
            family: "Idefics3", kind: .vlm,
            modelTypes: Idefics3.modelTypes, architectures: Idefics3.architectures,
            load: { c, w, o, d in
                let variant = try Idefics3.variant(for: c)
                let m = try variant.loadModel(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: m,
                    defaultGenerationParameters: variant.defaultGenerationParameters,
                    availableCapabilities: variant.availableCapabilities)
            }),
        ModelEntry(
            family: "PaliGemma", kind: .vlm,
            modelTypes: Paligemma.modelTypes, architectures: Paligemma.architectures,
            load: { c, w, o, d in
                let variant = try Paligemma.variant(for: c)
                let m = try variant.loadModel(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: m,
                    defaultGenerationParameters: variant.defaultGenerationParameters,
                    availableCapabilities: variant.availableCapabilities)
            }),
        ModelEntry(
            family: "MiniCPM-V 4.x", kind: .vlm,
            sniff: { MiniCPM4.handles($0) },
            load: { c, w, o, d in
                let variant = try MiniCPM4.variant(for: c)
                let vlm = try variant.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: variant.defaultGenerationParameters,
                    availableCapabilities: variant.capabilities(for: vlm),
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "Nemotron Nano VL", kind: .vlm,
            sniff: { isNemotronVisionLanguage($0) },
            load: { c, w, o, d in
                let vlm = try NemotronVL.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: NemotronHHybrid.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }),
        ModelEntry(
            family: "Nemotron-Labs-Diffusion VLM", kind: .vlm,
            sniff: { isNemotronDiffusionVisionLanguage($0) },
            load: { c, w, o, d in
                let vlm = try NemotronDiffusionVL.load(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: NemotronDiffusionDense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }),
        // Qwen 3.5 / 3.6 — shares its architecture string with the
        // text-only release; the only reliable VL signal is an actual
        // vision tower in the weights. Matches on architecture (vision_config
        // is implied inside the VL gate); the weight probe lives in `load`,
        // which routes a vestigial-vision-config checkpoint to the text loader.
        ModelEntry(
            family: "Qwen 3.5-VL", kind: .vlm,
            architectures: Qwen35.architectures,
            load: { c, w, o, d in
                if w.has("model.visual.patch_embed.proj.weight")
                    || w.has("vision_tower.patch_embed.proj.weight")
                {
                    let vlm = try Qwen35VL.load(config: c, weights: w, options: o, device: d)
                    return Loaded(
                        engine: vlm.engine,
                        defaultGenerationParameters: Qwen35Hybrid.defaultGenerationParameters,
                        availableCapabilities: Qwen35VL.availableCapabilities,
                        vlModel: vlm)
                }
                // Vestigial vision_config, no vision tensors → text-only.
                return try ModelRegistry.loadQwen35(
                    config: c, weights: w, options: o, device: d)
            }),
        ModelEntry(
            family: "SmolVLM2", kind: .vlm,
            modelTypes: SmolVLM2.modelTypes, architectures: SmolVLM2.architectures,
            load: { c, w, o, d in
                let variant = try SmolVLM2.variant(for: c)
                let engine = try variant.loadModel(config: c, weights: w, options: o, device: d)
                return Loaded(
                    engine: engine,
                    defaultGenerationParameters: variant.defaultGenerationParameters,
                    availableCapabilities: variant.availableCapabilities)
            }),
    ]

    // ─── Text / non-VL tier ──────────────────────────────────────────

    static let textEntries: [ModelEntry] = [
        // MiniCPM5 declares a plain `LlamaForCausalLM` / `llama` arch, so it
        // MUST be checked before the generic Llama entry — it routes by a
        // `_name_or_path` marker (`MiniCPM5.handles`).
        ModelEntry(
            family: "MiniCPM5", kind: .llm,
            sniff: { MiniCPM5.handles($0) },
            load: { c, w, o, d in
                try MiniCPM5.load(config: c, weights: w, options: o, device: d)
            }),
        ModelEntry(
            family: "Llama 3.x", kind: .llm,
            modelTypes: Llama3.modelTypes, architectures: Llama3.architectures,
            load: { c, w, o, d in try ModelRegistry.loadLlama3(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Mistral", kind: .llm,
            modelTypes: Mistral.modelTypes, architectures: Mistral.architectures,
            load: { c, w, o, d in try ModelRegistry.loadMistral(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "SmolLM", kind: .llm,
            modelTypes: SmolLM.modelTypes, architectures: SmolLM.architectures,
            load: { c, w, o, d in try ModelRegistry.loadSmolLM(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "OLMo 1", kind: .llm,
            modelTypes: OLMo.olmo1ModelTypes, architectures: OLMo.olmo1Architectures,
            load: { c, w, o, d in try ModelRegistry.loadOLMo1(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Yi", kind: .llm,
            modelTypes: Yi.modelTypes, architectures: Yi.architectures,
            load: { c, w, o, d in try ModelRegistry.loadYi(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Granite 3.x", kind: .llm,
            modelTypes: Granite3.modelTypes, architectures: Granite3.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGranite3(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "InternLM2", kind: .llm,
            modelTypes: InternLM2.modelTypes, architectures: InternLM2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadInternLM2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "OLMo 2", kind: .llm,
            modelTypes: OLMo.olmo2ModelTypes, architectures: OLMo.olmo2Architectures,
            load: { c, w, o, d in try ModelRegistry.loadOLMo2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Starcoder 2", kind: .llm,
            modelTypes: Starcoder2.modelTypes, architectures: Starcoder2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadStarcoder2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Phi", kind: .llm,
            modelTypes: Phi.modelTypes, architectures: Phi.architectures,
            load: { c, w, o, d in try ModelRegistry.loadPhi(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Qwen 2 / 2.5", kind: .llm,
            modelTypes: Qwen2.modelTypes, architectures: Qwen2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadQwen2(config: c, weights: w, options: o, device: d) }),
        // Gemma 2 before Gemma 3 before Gemma 4 (distinct model_types; order
        // is belt-and-suspenders).
        ModelEntry(
            family: "Gemma 2", kind: .llm,
            modelTypes: Gemma2.modelTypes, architectures: Gemma2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGemma2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Gemma 3", kind: .llm,
            modelTypes: Gemma3.modelTypes, architectures: Gemma3.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGemma3(config: c, weights: w, options: o, device: d) }),
        // Gemma 4 Omni drafter — loads + lists but can't generate standalone
        // (EAGLE/MTP draft model). Must precede the Gemma 4 entry.
        ModelEntry(
            family: "Gemma 4 drafter", kind: .llm,
            modelTypes: Gemma4.drafterModelTypes, architectures: Gemma4.drafterArchitectures,
            load: { c, _, _, _ in
                throw ModelError.draftModelNotGeneratable(
                    c.architecture ?? c.modelType ?? "<unknown>")
            }),
        ModelEntry(
            family: "Gemma 4", kind: .llm,
            modelTypes: Gemma4.modelTypes, architectures: Gemma4.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGemma4(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Qwen 3", kind: .llm,
            modelTypes: Qwen3.modelTypes, architectures: Qwen3.architectures,
            load: { c, w, o, d in try ModelRegistry.loadQwen3(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Mamba 2", kind: .llm,
            modelTypes: Mamba2.modelTypes, architectures: Mamba2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadMamba2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "FalconH1", kind: .llm,
            modelTypes: FalconH1.modelTypes, architectures: FalconH1.architectures,
            load: { c, w, o, d in try ModelRegistry.loadFalconH1(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "NemotronH", kind: .llm,
            modelTypes: NemotronH.modelTypes, architectures: NemotronH.architectures,
            load: { c, w, o, d in try ModelRegistry.loadNemotronH(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Granite 4", kind: .llm,
            modelTypes: Granite4.modelTypes, architectures: Granite4.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGranite4(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Jamba", kind: .llm,
            modelTypes: Jamba.modelTypes, architectures: Jamba.architectures,
            load: { c, w, o, d in try ModelRegistry.loadJamba(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Qwen 3.5", kind: .llm,
            modelTypes: Qwen35.modelTypes, architectures: Qwen35.architectures,
            load: { c, w, o, d in try ModelRegistry.loadQwen35(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "GPT-OSS", kind: .llm,
            modelTypes: GPTOSS.modelTypes, architectures: GPTOSS.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGPTOSS(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "LFM2", kind: .llm,
            modelTypes: LFM2.modelTypes, architectures: LFM2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadLFM2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Nemotron-Labs-Diffusion", kind: .llm,
            modelTypes: NemotronDiffusion.modelTypes, architectures: NemotronDiffusion.architectures,
            load: { c, w, o, d in try ModelRegistry.loadNemotronDiffusion(config: c, weights: w, options: o, device: d) }),
    ]

    /// The entry that WOULD load `config` — VL tier when the config is a
    /// vision-language checkpoint, else the text tier. `nil` means: a VL
    /// config with no specific handler (→ `visionModelNotIntegrated`), or a
    /// non-VL config no family claims (→ `unsupportedArchitecture`).
    /// Exposed for routing tests (no weights needed).
    public static func entry(for config: ModelConfig) -> ModelEntry? {
        if VisionLanguageArchitectures.isVisionLanguage(config) {
            return visionEntries.first { $0.matches(config) }
        }
        return textEntries.first { $0.matches(config) }
    }
}
