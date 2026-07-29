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
// (the `butter models` catalog reads this same table); they're left empty
// here so this phase stays a behavior-preserving dispatch swap.

import Foundation
import Tokenizers

extension ModelRegistry {

    // ─── Vision-language tier ────────────────────────────────────────

    static let visionEntries: [ModelEntry] = [
        ModelEntry(
            family: "Gemma 3 VL", kind: .vlm,
            summary: "SigLIP ViT + multi-modal projector + Gemma 3 text backbone.",
            repos: ["mlx-community/gemma-3-4b-it-4bit", "mlx-community/gemma-3-27b-it-4bit"],
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
            summary: "LiquidAI LFM2-VL — SigLIP2 ViT + LFM2 backbone + pixel-unshuffle projector.",
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
            summary: "Dynamic-res windowed-attention ViT + Qwen 2.5 backbone.",
            repos: [
                "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            ],
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
            summary: "Dynamic-res full-attention ViT + Qwen 2 text backbone.",
            repos: [
                "mlx-community/Qwen2-VL-2B-Instruct-4bit",
                "mlx-community/Qwen2-VL-7B-Instruct-4bit",
            ],
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
            summary: "Dynamic-res full-attention ViT + Qwen 3 dense backbone.",
            repos: [
                "mlx-community/Qwen3-VL-2B-Instruct-4bit",
                "mlx-community/Qwen3-VL-8B-Instruct-4bit",
            ],
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
            summary: "Dynamic-res ViT + Qwen 3.5-shaped GDN ↔ attention MoE backbone.",
            repos: ["mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"],
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
            summary: "Bespoke Gemma 4 ViT + multi-modal embedder + Gemma 4 text backbone.",
            repos: ["mlx-community/gemma-4-e4b-it-4bit", "mlx-community/gemma-4-31b-it-4bit"],
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
            summary: "Encoder-free omni: raw-patch image + raw-640-frame audio + per-frame video onto the Gemma 4 backbone.",
            repos: ["ekryski/gemma-4-12B-it-mlx-4bit"],
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
            summary: "Apple FastVLM — FastViTHD tower + mlp2x_gelu projector + Qwen2 backbone.",
            repos: ["mlx-community/FastVLM-0.5B-bf16", "ekryski/FastVLM-0.5B-4bit"],
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
            summary: "Mistral Pixtral 12B — bespoke RoPE-2D ViT + Mistral text backbone.",
            repos: ["mlx-community/pixtral-12b-4bit"],
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
            summary: "Mistral Small 3.1 / 3.2 — Pixtral ViT + patch-merger projector + Mistral backbone.",
            repos: [
                "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
                "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
            ],
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
            summary: "Zhipu GLM-OCR — dynamic-resolution ViT + GLM-Lite text backbone (document OCR).",
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
            summary: "HuggingFace Idefics3 — Llama 3 backbone + SigLIP ViT + pixel-shuffle connector.",
            repos: ["mlx-community/Idefics3-8B-Llama3-4bit"],
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
            summary: "Google PaliGemma 1 / 2 — SigLIP ViT + Gemma backbone (Gemma 2 on PG2).",
            repos: [
                "mlx-community/paligemma2-3b-mix-224-4bit",
                "mlx-community/paligemma2-10b-mix-448-4bit",
            ],
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
            summary: "MiniCPM-V 4.5 / 4.6 + MiniCPM-o — Qwen3 / 3.5 backbone + SigLIP, resampler / merger projector.",
            repos: ["mlx-community/MiniCPM-V-4.6-4bit"],
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
            summary: "NVIDIA Nemotron-H text backbone + ViT + projector (Nano VL).",
            repos: ["nvidia/Nemotron-Nano-VL-8B-V1", "ekryski/Llama-3.1-Nemotron-Nano-VL-8B-V1-4bit"],
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
            summary: "Tri-mode diffusion text backbone + Pixtral ViT vision tower.",
            repos: ["nvidia/Nemotron-Labs-Diffusion-VLM-8B"],
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
            summary: "Qwen 3.5 GDN ↔ attention MoE backbone + dynamic-res ViT.",
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
            summary: "HuggingFace SmolVLM 1 / 2 — SmolLM backbone + SigLIP-So400m ViT.",
            repos: ["mlx-community/SmolVLM-Instruct-4bit", "mlx-community/SmolVLM2-2.2B-Instruct-mlx"],
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
            summary: "OpenBMB MiniCPM5-1B — Llama-shaped text (routed by name marker).",
            sniff: { MiniCPM5.handles($0) },
            load: { c, w, o, d in
                try MiniCPM5.load(config: c, weights: w, options: o, device: d)
            }),
        ModelEntry(
            family: "Llama 3.x", kind: .llm,
            summary: "Meta Llama 3 / 3.1 / 3.2 / 3.3 dense GQA transformer.",
            repos: [
                "unsloth/Llama-3.2-1B", "mlx-community/Llama-3.1-8B-Instruct-4bit",
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "mlx-community/Llama-3.3-70B-Instruct-4bit",
            ],
            modelTypes: Llama3.modelTypes, architectures: Llama3.architectures,
            load: { c, w, o, d in try ModelRegistry.loadLlama3(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Mistral", kind: .llm,
            summary: "Mistral 7B / Nemo / Small — Llama-shaped GQA (Llama-compatible loader).",
            repos: [
                "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                "mlx-community/Mistral-7B-Instruct-v0.3-bf16",
            ],
            modelTypes: Mistral.modelTypes, architectures: Mistral.architectures,
            load: { c, w, o, d in try ModelRegistry.loadMistral(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "SmolLM", kind: .llm,
            summary: "HuggingFace SmolLM 1 / 2 / 3 — 135M–3B Llama-shaped models.",
            repos: [
                "mlx-community/SmolLM-135M-Instruct-4bit", "mlx-community/SmolLM3-3B-4bit",
            ],
            modelTypes: SmolLM.modelTypes, architectures: SmolLM.architectures,
            load: { c, w, o, d in try ModelRegistry.loadSmolLM(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "OLMo 1", kind: .llm,
            summary: "AI2 OLMo 1 — open pre-norm Llama-shaped research model.",
            repos: ["mlx-community/OLMo-1B-hf-4bit-mlx", "mlx-community/OLMo-7B-hf-4bit-mlx"],
            modelTypes: OLMo.olmo1ModelTypes, architectures: OLMo.olmo1Architectures,
            load: { c, w, o, d in try ModelRegistry.loadOLMo1(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Yi", kind: .llm,
            summary: "01.AI Yi — Llama-shaped dense backbone.",
            repos: ["mlx-community/Yi-1.5-6B-Chat-4bit", "mlx-community/Yi-1.5-9B-Chat-4bit"],
            modelTypes: Yi.modelTypes, architectures: Yi.architectures,
            load: { c, w, o, d in try ModelRegistry.loadYi(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Granite 3.x", kind: .llm,
            summary: "IBM Granite v3 dense — Llama-shaped GQA backbone with µP multipliers.",
            repos: [
                "mlx-community/granite-3.3-2b-instruct-4bit",
                "mlx-community/granite-3.3-8b-instruct-4bit",
            ],
            modelTypes: Granite3.modelTypes, architectures: Granite3.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGranite3(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "InternLM2", kind: .llm,
            summary: "Shanghai AI Lab InternLM 2 / 2.5 — Llama-shaped (fused wqkv remap).",
            repos: ["mlx-community/internlm2_5-7b-chat-4bit"],
            modelTypes: InternLM2.modelTypes, architectures: InternLM2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadInternLM2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "OLMo 2", kind: .llm,
            summary: "AI2 OLMo 2 — post-norm Llama shape with q/k-norm.",
            repos: [
                "mlx-community/OLMo-2-1124-7B-Instruct-4bit",
                "mlx-community/OLMo-2-0325-32B-Instruct-4bit",
            ],
            modelTypes: OLMo.olmo2ModelTypes, architectures: OLMo.olmo2Architectures,
            load: { c, w, o, d in try ModelRegistry.loadOLMo2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Starcoder 2", kind: .llm,
            summary: "BigCode Starcoder 2 — code model (LayerNorm + GELU MLP, attention biases).",
            repos: ["mlx-community/starcoder2-7b-4bit", "mlx-community/starcoder2-15b-4bit"],
            modelTypes: Starcoder2.modelTypes, architectures: Starcoder2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadStarcoder2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Phi", kind: .llm,
            summary: "Microsoft Phi-3 / 3.5 dense transformer.",
            repos: [
                "mlx-community/Phi-3-mini-4k-instruct-4bit",
                "mlx-community/Phi-3.5-mini-instruct-4bit",
            ],
            modelTypes: Phi.modelTypes, architectures: Phi.architectures,
            load: { c, w, o, d in try ModelRegistry.loadPhi(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Qwen 2 / 2.5", kind: .llm,
            summary: "Qwen 2 / 2.5 dense transformer (Llama-shaped + QKV biases).",
            repos: [
                "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "mlx-community/Qwen2.5-7B-Instruct-4bit",
                "mlx-community/Qwen2.5-32B-Instruct-4bit",
            ],
            modelTypes: Qwen2.modelTypes, architectures: Qwen2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadQwen2(config: c, weights: w, options: o, device: d) }),
        // Gemma 2 before Gemma 3 before Gemma 4 (distinct model_types; order
        // is belt-and-suspenders).
        ModelEntry(
            family: "Gemma 2", kind: .llm,
            summary: "Google Gemma 2 — Llama-shaped GQA + final-logit soft-cap.",
            repos: [
                "mlx-community/gemma-2-2b-it-4bit", "mlx-community/gemma-2-9b-it-4bit",
                "mlx-community/gemma-2-27b-it-4bit",
            ],
            modelTypes: Gemma2.modelTypes, architectures: Gemma2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGemma2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Gemma 3", kind: .llm,
            summary: "Google Gemma 3 text decoder.",
            repos: ["mlx-community/gemma-3-1b-it-4bit", "mlx-community/gemma-3-text-4b-it-4bit"],
            modelTypes: Gemma3.modelTypes, architectures: Gemma3.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGemma3(config: c, weights: w, options: o, device: d) }),
        // Gemma 4 Omni drafter — loads + lists but can't generate standalone
        // (EAGLE/MTP draft model). Must precede the Gemma 4 entry.
        ModelEntry(
            family: "Gemma 4 drafter", kind: .llm,
            summary: "EAGLE/MTP speculative-decoding drafter for Gemma 4 Omni — lists + quantizes but can't generate standalone.",
            repos: ["ekryski/gemma-4-12B-it-assistant-mlx-4bit"],
            modelTypes: Gemma4.drafterModelTypes, architectures: Gemma4.drafterArchitectures,
            load: { c, _, _, _ in
                throw ModelError.draftModelNotGeneratable(
                    c.architecture ?? c.modelType ?? "<unknown>")
            }),
        ModelEntry(
            family: "Gemma 4", kind: .llm,
            summary: "Gemma 4 — Dense / E-series PLE / 26B-A4B MoE variants.",
            repos: [
                "mlx-community/gemma-4-e2b-it-mxfp4", "mlx-community/gemma-4-e4b-it-4bit",
                "mlx-community/gemma-4-26b-a4b-it-4bit",
            ],
            modelTypes: Gemma4.modelTypes, architectures: Gemma4.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGemma4(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Qwen 3", kind: .llm,
            summary: "Qwen 3 dense — per-head q/k RMSNorm before RoPE.",
            repos: [
                "mlx-community/Qwen3-0.6B-4bit", "mlx-community/Qwen3-8B-4bit",
                "mlx-community/Qwen3-32B-4bit",
            ],
            modelTypes: Qwen3.modelTypes, architectures: Qwen3.architectures,
            load: { c, w, o, d in try ModelRegistry.loadQwen3(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Mamba 2", kind: .llm,
            summary: "Dense Mamba 2 selective-SSM (no attention).",
            repos: ["mlx-community/mamba2-370m-4bit", "mlx-community/mamba2-1.3b-4bit"],
            modelTypes: Mamba2.modelTypes, architectures: Mamba2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadMamba2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "FalconH1", kind: .llm,
            summary: "Per-layer Mamba 2 + attention (within-layer hybrid). Raw bf16/f16 only.",
            repos: [
                "mlx-community/Falcon-H1-1.5B-Instruct-bf16",
                "mlx-community/Falcon-H1-7B-Instruct-4bit",
            ],
            modelTypes: FalconH1.modelTypes, architectures: FalconH1.architectures,
            load: { c, w, o, d in try ModelRegistry.loadFalconH1(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "NemotronH", kind: .llm,
            summary: "Stack-interleaved Mamba 2 / attention / MLP. Raw bf16/f16 only.",
            repos: ["nvidia/Nemotron-H-4B-Base-8K", "ekryski/Nemotron-H-4B-Base-8K-4bit"],
            modelTypes: NemotronH.modelTypes, architectures: NemotronH.architectures,
            load: { c, w, o, d in try ModelRegistry.loadNemotronH(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Hy3", kind: .llm,
            summary:
                "Tencent Hy3 295B-A21B MoE: sigmoid+bias router, shared expert, QK-norm→RoPE. Prefer mlx 2-bit oQ2 on Apple Silicon.",
            repos: ["tencent/Hy3", "tencent/Hy3-FP8", "mlx-community/Hy3-oQ2"],
            modelTypes: Hy3.modelTypes, architectures: Hy3.architectures,
            load: { c, w, o, d in try ModelRegistry.loadHy3(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Granite 4", kind: .llm,
            summary: "Stack-interleaved Mamba 2 / attention + dense or MoE FFN.",
            repos: ["mlx-community/granite-4.0-1b-4bit", "mlx-community/granite-4.0-h-tiny-4bit"],
            modelTypes: Granite4.modelTypes, architectures: Granite4.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGranite4(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Jamba", kind: .llm,
            summary: "Mamba 1 + attention + dense / MoE FFN. Raw bf16/f16 only.",
            repos: ["mlx-community/AI21-Jamba-Reasoning-3B-bf16"],
            modelTypes: Jamba.modelTypes, architectures: Jamba.architectures,
            load: { c, w, o, d in try ModelRegistry.loadJamba(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Qwen 3.5", kind: .llm,
            summary: "Qwen 3.5 / 3.6 — Gated Delta Net ↔ attention hybrid; dense or MoE FFN.",
            repos: [
                "mlx-community/Qwen3.5-0.8B-4bit", "mlx-community/Qwen3.5-9B-4bit",
                "mlx-community/Qwen3.5-35B-A3B-4bit",
            ],
            modelTypes: Qwen35.modelTypes, architectures: Qwen35.architectures,
            load: { c, w, o, d in try ModelRegistry.loadQwen35(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "GPT-OSS", kind: .llm,
            summary: "OpenAI GPT-OSS — 32-expert MoE, sliding/full attention, learned sinks.",
            repos: ["mlx-community/gpt-oss-20b-MXFP4-Q8", "mlx-community/gpt-oss-120b-4bit"],
            modelTypes: GPTOSS.modelTypes, architectures: GPTOSS.architectures,
            load: { c, w, o, d in try ModelRegistry.loadGPTOSS(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "LFM2", kind: .llm,
            summary: "LiquidAI LFM2 / LFM2.5 — stack-interleaved short-conv + attention, optional MoE.",
            repos: [
                "mlx-community/LFM2-1.2B-4bit", "mlx-community/LFM2-8B-A1B-4bit",
                "mlx-community/LFM2.5-1.2B-Instruct-4bit",
            ],
            modelTypes: LFM2.modelTypes, architectures: LFM2.architectures,
            load: { c, w, o, d in try ModelRegistry.loadLFM2(config: c, weights: w, options: o, device: d) }),
        ModelEntry(
            family: "Nemotron-Labs-Diffusion", kind: .llm,
            summary: "Tri-mode — autoregressive / block diffusion / self-speculation.",
            repos: ["nvidia/Nemotron-Labs-Diffusion-8B"],
            modelTypes: NemotronDiffusion.modelTypes, architectures: NemotronDiffusion.architectures,
            load: { c, w, o, d in try ModelRegistry.loadNemotronDiffusion(config: c, weights: w, options: o, device: d) }),
    ]


    /// Every registered family — the VL + text loadable entries plus the
    /// catalog-only audio / VAD rows. The single source of truth behind the
    /// `butter models` catalog.
    public static var catalogEntries: [ModelEntry] {
        visionEntries + textEntries + audioEntries
    }

    /// All registered families grouped by `ModelType` in catalog display
    /// order (empty kinds omitted). Backs `butter models`.
    public static var catalog: [(kind: ModelType, entries: [ModelEntry])] {
        let order: [ModelType] = [.llm, .vlm, .ocr, .omni, .stt, .tts, .sts, .vad]
        return order.compactMap { kind in
            let entries = catalogEntries
                .filter { $0.kind == kind }
                .sorted { $0.family < $1.family }
            return entries.isEmpty ? nil : (kind, entries)
        }
    }

    /// The entry that WOULD load `config`, across every tier — the single
    /// routing predicate the unified `Model.load` and the routing tests use.
    /// `nil` means no family claims it.
    ///
    /// Precedence (preserves the pre-unification behavior + folds in audio):
    ///   1. A **VL** checkpoint (`isVisionLanguage`) routes to the VL tier;
    ///      Gemma 4's VL/omni conversions carry a top-level `audio_config`,
    ///      so the VL tier MUST win over the audio structural sniffs for
    ///      them — hence the VL gate is checked first.
    ///   2. A VL checkpoint with no VL handler may still be an omni-audio
    ///      model (Qwen-Omni configs carry a `vision_config` but load through
    ///      the audio path) — try the audio tier before giving up.
    ///   3. A non-VL checkpoint: the audio tier's specific structural sniffs
    ///      (`audio_config` / `talker_config` / name markers) win over the
    ///      generic text architecture match (e.g. an Orpheus checkpoint is a
    ///      `LlamaForCausalLM` but a TTS model), then the text tier.
    ///
    /// A non-nil result whose `loadFromDirectory` is set is an audio / VAD
    /// family; otherwise it's a text/VL family with a `load` closure.
    public static func entry(for config: ModelConfig) -> ModelEntry? {
        if VisionLanguageArchitectures.isVisionLanguage(config) {
            if let vl = visionEntries.first(where: { $0.matches(config) }) {
                return vl
            }
            // VL config, no VL handler → maybe an omni-audio family.
            return audioEntries.first { $0.matches(config) }
        }
        if let audio = audioEntries.first(where: { $0.matches(config) }) {
            return audio
        }
        return textEntries.first { $0.matches(config) }
    }

    /// Wrap a freshly-constructed audio engine into a `Loaded`. For the
    /// `TokenizerAttachable` STT/omni decoders (whose `transcribeStream`
    /// renders token ids to text) it loads the snapshot tokenizer and
    /// attaches it to the engine — the single uniform attach the former
    /// `AudioModelRegistry.load` did — and surfaces it as `Model.tokenizer`.
    fileprivate static func audioLoaded(
        _ engine: any ModelEngine,
        _ capabilities: Set<Capability>,
        directory: URL,
        attachTokenizer: Bool = false
    ) async -> Loaded {
        var tok: (any Tokenizer)?
        if attachTokenizer {
            // `try?`: a missing / unsupported tokenizer just leaves the
            // streaming path to surface its own error; it must not fail the
            // whole load.
            tok = try? await TokenizerLoader().load(from: directory)
            (engine as? any TokenizerAttachable)?.tokenizer = tok
        }
        return Loaded(
            engine: engine, availableCapabilities: capabilities, tokenizer: tok)
    }

    // ─── Audio / VAD tier ────────────────────────────────────────────
    //
    // Order = dispatch precedence (mirrors `AudioModelRegistry.resolve`).

    static let audioEntries: [ModelEntry] = [
        // ── FishSpeech (first — its `fish_qwen3_omni` contains "qwen3_omni") ──
        ModelEntry(
            family: "FishSpeech", kind: .tts,
            summary: "FishAudio FishSpeech — GQA dual-autoregressive neural-codec TTS.",
            capabilities: Capability.textToSpeech,
            sniff: { FishSpeechModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try FishSpeechModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),

        // ── Speech-to-text / omni (Qwen3TTS before the Qwen ASR/Omni checks) ──
        ModelEntry(
            family: "Qwen3-TTS", kind: .tts,
            summary: "Qwen3-TTS — talker + code predictor + ECAPA speaker + intrinsic codec.",
            repos: ["mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"],
            capabilities: Capability.textToSpeech,
            sniff: { Qwen3TTSModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await Qwen3TTSModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Qwen3-ASR", kind: .stt,
            summary: "Qwen3-ASR — Qwen 3 backbone + audio encoder front-end.",
            repos: ["mlx-community/Qwen3-ASR-0.6B-4bit", "mlx-community/Qwen3-ASR-1.7B-4bit"],
            capabilities: Capability.speechToText,
            sniff: { Qwen3ASRModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try Qwen3ASRModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.speechToText, directory: dir, attachTokenizer: true)
            }),
        ModelEntry(
            family: "Qwen-Omni", kind: .omni,
            summary: "Whisper-style audio encoder + Qwen text backbone (omni-modal).",
            repos: ["Qwen/Qwen2.5-Omni-7B", "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"],
            capabilities: Capability.omniAudio,
            sniff: { QwenOmniModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try QwenOmniModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.omniAudio, directory: dir, attachTokenizer: true)
            }),
        ModelEntry(
            family: "Marvis (CSM)", kind: .tts,
            summary: "Sesame CSM-shaped dual-transformer TTS — Mimi-code frame generator.",
            repos: ["Marvis-AI/marvis-tts-250m-v0.2-MLX-4bit", "sesame/csm-1b"],
            capabilities: Capability.textToSpeech,
            sniff: { MarvisModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await MarvisModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Orpheus (LlamaTTS)", kind: .tts,
            summary: "Orpheus-style TTS — Llama 3 acoustic backbone + SNAC code decode loop.",
            repos: ["mlx-community/orpheus-3b-0.1-ft-4bit"],
            capabilities: Capability.textToSpeech,
            sniff: { LlamaTTSModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await LlamaTTSModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Whisper STT", kind: .stt,
            summary: "OpenAI Whisper — encoder + cross-attending text decoder.",
            repos: [
                "openai/whisper-tiny", "mlx-community/whisper-tiny-mlx",
                "mlx-community/whisper-base-mlx", "mlx-community/whisper-small-mlx",
                "mlx-community/whisper-medium-mlx", "mlx-community/whisper-large-v3-mlx",
                "mlx-community/whisper-large-v3-turbo-4bit",
            ],
            capabilities: Capability.speechToText,
            sniff: { WhisperModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try WhisperModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.speechToText, directory: dir, attachTokenizer: true)
            }),
        ModelEntry(
            family: "SenseVoice", kind: .stt,
            summary: "Alibaba SenseVoice — SAN-M encoder + CTC head (non-AR STT).",
            repos: ["mlx-community/SenseVoiceSmall"],
            capabilities: Capability.speechToText,
            sniff: { SenseVoiceModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try SenseVoiceModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.speechToText, directory: dir, attachTokenizer: true)
            }),
        // StyleTTS2 / Kitten before Kokoro (Kokoro's structural fallback also
        // matches Kitten's `kitten_tts`).
        ModelEntry(
            family: "StyleTTS2", kind: .tts,
            summary: "StyleTTS2 — style-diffusion acoustic model + iSTFTNet vocoder.",
            capabilities: Capability.textToSpeech,
            sniff: { StyleTTS2Model.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try StyleTTS2Model.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Kokoro TTS", kind: .tts,
            summary: "Kokoro 82M — StyleTTS2 acoustic + iSTFTNet GPU vocoder tail.",
            repos: ["mlx-community/Kokoro-82M-bf16", "hexgrad/Kokoro-82M"],
            capabilities: Capability.textToSpeech,
            sniff: { KokoroModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try KokoroModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Echo TTS", kind: .tts,
            summary: "Echo TTS — compact neural-codec speech synthesizer.",
            capabilities: Capability.textToSpeech,
            sniff: { EchoTTSModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try EchoTTSModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Qwen3-TTS Base", kind: .tts,
            summary: "Qwen3-TTS base talker variant.",
            capabilities: Capability.textToSpeech,
            sniff: { Qwen3TTSBaseModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await Qwen3TTSBaseModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Parakeet", kind: .stt,
            summary: "NVIDIA Parakeet — FastConformer encoder + TDT/CTC decoder.",
            capabilities: Capability.speechToText,
            sniff: { ParakeetModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try ParakeetModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToText, directory: dir)
            }),
        ModelEntry(
            family: "Chatterbox TTS", kind: .tts,
            summary: "Resemble AI Chatterbox — Llama-shaped TTS backbone + S3 tokenizer.",
            repos: ["mlx-community/Chatterbox-TTS-4bit", "mlx-community/Chatterbox-Turbo-TTS-4bit"],
            capabilities: Capability.textToSpeech,
            sniff: { ChatterboxModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try ChatterboxModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "FireRedASR2", kind: .stt,
            summary: "FireRedTeam FireRedASR2-AED — conformer encoder + Llama decoder.",
            repos: ["mlx-community/FireRedASR2-AED-mlx"],
            capabilities: Capability.speechToText,
            sniff: { FireRedASR2Model.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try FireRedASR2Model.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToText, directory: dir)
            }),
        // Moss Nano before Moss-8B (Nano's distinctive `gpt2_config` block).
        ModelEntry(
            family: "Moss TTS Nano", kind: .tts,
            summary: "Moss TTS Nano — compact on-device variant.",
            capabilities: Capability.textToSpeech,
            sniff: { MossTTSNanoModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try MossTTSNanoModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Moss TTS", kind: .tts,
            summary: "Moss TTS — autoregressive neural-codec speech synthesizer.",
            capabilities: Capability.textToSpeech,
            sniff: { MossTTSModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try MossTTSModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "LFM Audio", kind: .omni,
            summary: "LiquidAI LFM Audio — omni-modal audio-in path.",
            capabilities: Capability.omniAudio,
            sniff: { LFMAudioModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try LFMAudioModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.omniAudio, directory: dir, attachTokenizer: true)
            }),
        ModelEntry(
            family: "Voxtral", kind: .stt,
            summary: "Mistral Voxtral — realtime STT / TTS (streaming).",
            repos: [
                "mlx-community/Voxtral-Mini-3B-2507-bf16",
                "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            ],
            capabilities: Capability.speechToText,
            sniff: { VoxtralRealtimeModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try VoxtralRealtimeModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToText, directory: dir)
            }),
        ModelEntry(
            family: "GLM-ASR", kind: .stt,
            summary: "Zhipu GLM-ASR — conformer encoder + GLM text decoder.",
            capabilities: Capability.speechToText,
            sniff: { GLMASRModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try GLMASRModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.speechToText, directory: dir, attachTokenizer: true)
            }),
        ModelEntry(
            family: "Pocket TTS", kind: .tts,
            summary: "Pocket TTS — compact StyleTTS2-style on-device synth.",
            repos: ["mlx-community/pocket-tts-4bit"],
            capabilities: Capability.textToSpeech,
            sniff: { PocketTTSModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try PocketTTSModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Soprano TTS", kind: .tts,
            summary: "Soprano 80M — compact StyleTTS2-flavored on-device synth.",
            repos: ["mlx-community/Soprano-80M-bf16", "ekryski/Soprano-1.1-80M-4bit"],
            capabilities: Capability.textToSpeech,
            sniff: { SopranoModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await SopranoModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Granite Speech", kind: .stt,
            summary: "IBM Granite Speech — conformer encoder + QFormer projector + Granite LLM.",
            capabilities: Capability.speechToText,
            sniff: { GraniteSpeech.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await GraniteSpeech.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToText, directory: dir)
            }),

        // ── Speech-to-speech (enhancement / separation / segmentation) ──
        ModelEntry(
            family: "DeepFilterNet", kind: .sts,
            summary: "DeepFilterNet — real-time speech enhancement / denoise.",
            capabilities: Capability.speechToSpeech,
            sniff: { DeepFilterNetModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try DeepFilterNetModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Cohere Transcribe", kind: .stt,
            summary: "Cohere transcription model — audio encoder + Command text decoder.",
            capabilities: Capability.speechToText,
            sniff: { CohereTranscribeModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await CohereTranscribeModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToText, directory: dir)
            }),
        ModelEntry(
            family: "MossFormer2-SE", kind: .sts,
            summary: "MossFormer2 — speech enhancement / source separation.",
            capabilities: Capability.speechToSpeech,
            sniff: { MossFormer2SEModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try MossFormer2SEModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "SAM Audio", kind: .sts,
            summary: "SAM Audio — promptable audio segmentation.",
            capabilities: Capability.speechToSpeech,
            sniff: { SAMAudio.handles($0) },
            loadFromDirectory: { dir, config, _, dev in
                let samConfig = SAMAudioModel.loadConfig(from: dir)
                let bundle = try SafeTensorsBundle(directory: dir, device: dev)
                let variant = try SAMAudio.variant(for: config)
                let m = try variant.loadModel(
                    directory: dir, config: samConfig, weights: bundle, device: dev)
                return await audioLoaded(m, Capability.speechToSpeech, directory: dir)
            }),

        // ── Voice activity / diarization (matched by model_type) ──
        ModelEntry(
            family: "Silero VAD", kind: .vad,
            summary: "Streaming VAD — STFT + small gated-conv encoder.",
            repos: ["snakers4/silero-vad"],
            capabilities: [],
            modelTypes: ["silero_vad"],
            loadFromDirectory: { dir, _, _, dev in
                let m = try SileroVADModel.load(directory: dir, device: dev)
                return await audioLoaded(m, [], directory: dir)
            }),
        ModelEntry(
            family: "SmartTurn VAD", kind: .vad,
            summary: "Conversational endpoint / turn detection (Pipecat).",
            repos: ["pipecat-ai/smart-turn-v3"],
            capabilities: [],
            modelTypes: ["smart_turn", "smart_turn_v3", "smart-turn"],
            loadFromDirectory: { dir, _, _, dev in
                let m = try SmartTurnModel.load(directory: dir, device: dev)
                return await audioLoaded(m, [], directory: dir)
            }),
        ModelEntry(
            family: "Sortformer", kind: .vad,
            summary: "NVIDIA Sortformer — multi-speaker diarization.",
            capabilities: [],
            modelTypes: ["sortformer", "diar_sortformer"],
            loadFromDirectory: { dir, _, _, dev in
                let m = try SortformerModel.load(directory: dir, device: dev)
                return await audioLoaded(m, [], directory: dir)
            }),
        ModelEntry(
            family: "TenVAD", kind: .vad,
            summary: "TEN-framework lightweight VAD (native macOS framework).",
            capabilities: [],
            modelTypes: ["ten_vad", "ten-vad", "tenvad"],
            loadFromDirectory: { dir, _, _, dev in
                let m = try TenVADModel.load(directory: dir, device: dev)
                return await audioLoaded(m, [], directory: dir)
            }),
        ModelEntry(
            family: "FireRedVAD", kind: .vad,
            summary: "FireRedTeam DFSMN-based VAD.",
            capabilities: [],
            modelTypes: ["firered_vad", "firered-vad", "fireredvad"],
            loadFromDirectory: { dir, _, _, dev in
                let m = try FireRedVADModel.load(directory: dir, device: dev)
                return await audioLoaded(m, [], directory: dir)
            }),
    ]
}
