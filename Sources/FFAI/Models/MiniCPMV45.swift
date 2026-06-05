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
// MiniCPM-V 4.5 — OpenBMB's `MiniCPMV` checkpoint (`model_type: minicpmv`,
// `version: 4.5`). An image + video → text VLM:
//
//   • LLM    — Qwen3-8B dense (`llm.model.*` + `llm.lm_head`), 4096 hidden,
//              36 layers, 32 heads, 8 KV, head_dim 128.
//   • Vision — SigLIP encoder (`vpm.*`, 1152 hidden, 27 layers) + perceiver
//              resampler (`resampler.*`) → 64 tokens / image.
//
// This is MiniCPM-o-4.5 MINUS the audio tower: the same Qwen3-8B backbone,
// the same SigLIP + perceiver `MiniCPMOResampler`, just under the
// MiniCPM-V weight prefixes (`llm.` / `vpm.` rather than the omni
// checkpoint's `language_model.` / `vision_tower.`). It carries NO
// `text_config` sub-block — the Qwen3 hyper-parameters live at the top
// level of config.json, exactly like MiniCPM-o.
//
// The vision token runs are spliced into the text stream by the shared
// `VisionModel` at each `<|image_pad|>` (image) / `<|video_pad|>` (video)
// placeholder — distinct ids so the splice can disambiguate the two
// modalities in a single prompt.
//
// Vision internals are shared with MiniCPM-o:
// `Models/Vision/MiniCPMOVision.swift` (SigLIP + resampler) +
// `Models/Vision/MiniCPMOResampler.swift` (the perceiver). This file is
// the family orchestrator (load entry-point + public constants).
//
// Reference: (HF `openbmb/MiniCPM-V-4_5`).

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum MiniCPMV45 {
    public static let modelTypes: Set<String> = ["minicpmv"]
    public static let architectures: Set<String> = ["MiniCPMV"]

    /// The `version` field that distinguishes this checkpoint from the
    /// older `MiniCPMV` releases (2.x / 3.x, which ride a different —
    /// non-Qwen3 — backbone). MiniCPM-V-4.6 uses the distinct
    /// `MiniCPMV4_6ForConditionalGeneration` arch and routes elsewhere.
    public static let supportedVersion = 4.5
    /// Tolerance for the float `version` compare (config stores `4.5`).
    private static let versionEpsilon = 0.05

    /// Distinct per-modality placeholder ids (the MiniCPM-V vocab's own
    /// `<|image_pad|>` / `<|video_pad|>`). The processor scatters image vs
    /// video features over a shared run, but the `VisionModel` splice
    /// validates + substitutes EACH modality by token id — a shared id
    /// would make an image run look like a video run. The placeholder id is
    /// only a splice marker (replaced by the resampler embedding before the
    /// LLM sees it), so using distinct vocab ids is output-identical and
    /// lets one prompt carry both modalities unambiguously.
    /// Source: `added_tokens.json` (`<|image_pad|>` 151655 / `<|video_pad|>`
    /// 151656).
    public static let imageTokenId = 151_655  // <|image_pad|>
    public static let videoTokenId = 151_656  // <|video_pad|>

    /// Per-image / per-frame vision-token count (`query_num`, 64) — the
    /// perceiver resampler emits exactly this many text-hidden tokens.
    /// Source: config.json `query_num`.
    public static let defaultQueryNum = 64

    /// Runtime single-tile image resolution. 448 = 32×32 patches at
    /// `patch_size: 14` (matches `slice_config` scale resolution 448).
    public static let runtimeImageSize = 448

    /// Capabilities a MiniCPM-V-4.5 checkpoint exposes: text + image +
    /// video. NO audio (that is the MiniCPM-o omni checkpoint).
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.imageIn, .videoIn])

    /// Whether a decoded `config.json` describes a MiniCPM-V-4.5 checkpoint:
    /// the bare `MiniCPMV` arch / `minicpmv` model_type AND `version ≈ 4.5`.
    /// The version gate keeps this from grabbing older `MiniCPMV` releases.
    public static func handles(_ config: ModelConfig) -> Bool {
        let archMatch =
            (config.architecture.map(architectures.contains) ?? false)
            || (config.modelType.map(modelTypes.contains) ?? false)
        guard archMatch else { return false }
        guard let version = config.float("version") else { return false }
        return abs(version - supportedVersion) < versionEpsilon
    }

    /// Build a `VisionModel` from a `MiniCPMV` (v4.5) checkpoint: the
    /// Qwen3-8B dense LLM (`llm.model.*`) + the SigLIP + resampler vision
    /// tower (`vpm.*` / `resampler.*`), joined by `VisionModel`'s
    /// cross-modal splice (image / video).
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw MiniCPMOError.missingConfig("vision_config")
        }

        // ── Text backbone — Qwen3-8B dense ────────────────────────────
        // MiniCPM-V declares the Qwen3 text hyper-parameters at the TOP
        // level of config.json (no `text_config` sub-tree); the weights live
        // under `llm.model.*` + `llm.lm_head`. `Qwen3Dense.loadModel` reads
        // top-level config keys and probes `model.*` / `lm_head` in its
        // bundle, so re-wrap the bundle view at `llm.` and forward the
        // config as the text config. A `quantization` block (when present on
        // a converted checkpoint) is top-level, so the embedding / linears /
        // lm_head all load quantized; the bf16 base loads full-precision.
        let textConfig = ModelConfig(
            architecture: "Qwen3ForCausalLM",
            modelType: "qwen3",
            raw: config.raw)
        let textWeights = weights.prefixed("llm.")
        let textEngine = try Qwen3Dense.loadModel(
            config: textConfig, weights: textWeights,
            options: options, device: device)

        let queryNum = config.int("query_num") ?? defaultQueryNum

        // ── Vision tower — SigLIP encoder + perceiver resampler ───────
        // Reuses the MiniCPM-o vision tower verbatim; only the bundle
        // prefixes differ (`vpm.` / `resampler.`).
        let visionWeights = weights.prefixed("vpm.")
        let resamplerWeights = weights.prefixed("resampler.")
        let vision = try MiniCPMOVision.load(
            visionConfig: visionConfig,
            runtimeImageSize: runtimeImageSize, textHidden: textEngine.hidden,
            queryNum: queryNum, visionWeights: visionWeights,
            resamplerWeights: resamplerWeights,
            quantization: config.quantization, device: device)

        // ── Cross-modal splice (image / video — no audio) ─────────────
        return try VisionModel(
            visionEncoder: vision, engine: textEngine,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .siglip,
            imageTokenCount: vision.outputTokenCount)
    }
}
