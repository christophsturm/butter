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
// MiniCPM 4.x family root — OpenBMB's MiniCPM-V / MiniCPM-o 4.x line.
// One entry point that dispatches by variant (like the Qwen3 root):
//
//   • v4.5  (`MiniCPMV`, version 4.5)            — image + video.
//        Qwen3-8B dense (`llm.model.*`) + SigLIP (`vpm.*`) + perceiver
//        resampler (`resampler.*`) → 64 tokens/image.
//   • v4.6  (`MiniCPMV4_6ForConditionalGeneration`) — image + video.
//        SigLIP2-400M (`vision_tower.*`) + `vit_merger` (mid-stack window
//        cross-attn merger) + final `merger` (2×2 → text hidden) + the
//        Qwen3.5 text backbone.
//   • o-4.5 (`MiniCPMO`, version 4.5)            — + audio-IN.
//        Qwen3-8B dense (`language_model.model.*`) + SigLIP
//        (`vision_tower.*`) + the SAME perceiver resampler + a
//        Whisper-medium `audio_tower` + avg-pool + 4-bit projector. The TTS
//        half (`tts.*`) is ignored — FFAI does audio-IN only.
//
// Component stacks live alongside the rest of the family:
//   • text   — Qwen3.5 for 4.6 (`Models/Text/MiniCPM4Text.swift`); Qwen3
//              dense for 4.5 / o (`Qwen3Dense`).
//   • vision — `vit_merger` tower in `Models/Vision/MiniCPM4Vision.swift`
//              (4.6); the resampler tower in
//              `Models/Vision/MiniCPMOVision.swift` +
//              `Models/Vision/MiniCPMOResampler.swift` (4.5 / o).
//   • audio  — `Models/Audio/MiniCPM4Audio.swift` (o only).
//
// This file is the family orchestrator (variant detection + load
// dispatch + public constants). The vision / audio token runs are spliced
// into the text stream by the shared `VisionModel`.
//
// Reference: `modeling_minicpmv.py` / `modeling_minicpmo.py` /
// `processing_minicpm{v,o}.py` (HF `openbmb/MiniCPM-{V,o}-4_*`).

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum MiniCPM4 {
    /// Which concrete MiniCPM 4.x checkpoint a config describes.
    public enum Variant: Sendable { case v45, v46, omni }

    public static let modelTypes: Set<String> = [
        "minicpmv", "minicpmv4_6", "minicpmo",
    ]
    public static let architectures: Set<String> = [
        "MiniCPMV", "MiniCPMV4_6ForConditionalGeneration", "MiniCPMO",
    ]

    // ── v4.5 / o-4.5 token ids (the resampler variants) ───────────────
    // Distinct per-modality placeholder ids (the model's own `<|image_pad|>`
    // / `<|video_pad|>` / `<|audio|>` vocab ids). The processor scatters
    // image / video / audio features over a shared `<unk>` run, but the
    // `VisionModel` splice validates + substitutes EACH modality by token
    // id — a shared id makes an image run look like a video run. The
    // placeholder id is only a splice marker (replaced by the encoder
    // embedding before the LLM sees it), so distinct vocab ids are
    // output-identical and let one prompt carry more than one modality.
    public static let resamplerImageTokenId = 151_655  // <|image_pad|>
    public static let resamplerVideoTokenId = 151_656  // <|video_pad|>
    public static let omniAudioPlaceholderTokenId = 151_698  // <|audio|>

    // ── v4.6 token ids (`config.json` image/video_token_id) ───────────
    // These keep the family-default names (the original `MiniCPM4` was the
    // 4.6 checkpoint); the resampler variants carry their own ids above.
    public static let defaultImageTokenId = 248_056
    public static let defaultVideoTokenId = 248_057

    /// Capabilities every MiniCPM 4.x checkpoint exposes (text + image +
    /// video). Audio is added at load time for the omni variant when its
    /// audio tower is wired (`vlm.supportsAudio`).
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.imageIn, .videoIn])

    /// Per-image / per-frame resampler token count (`query_num`, 64).
    public static let defaultQueryNum = 64
    /// Average-pool stride applied to the omni audio features
    /// (`audio_pool_step`, 5).
    public static let defaultAudioPoolStep = 5
    /// Runtime single-tile image resolution. 448 = 32×32 patches at
    /// `patch_size: 14`.
    public static let runtimeImageSize = 448

    /// The `version` field the resampler variants (4.5 / o-4.5) carry.
    public static let resamplerVersion = 4.5
    /// Tolerance for the float `version` compare (config stores `4.5`).
    private static let versionEpsilon = 0.05

    /// Detect the variant a config describes; `nil` when it is not a
    /// MiniCPM 4.x checkpoint.
    public static func variant(for config: ModelConfig) -> Variant? {
        if config.modelType == "minicpmo" || config.architecture == "MiniCPMO" {
            return .omni
        }
        if config.modelType == "minicpmv4_6"
            || config.architecture == "MiniCPMV4_6ForConditionalGeneration"
        {
            return .v46
        }
        // v4.5 — the bare `MiniCPMV` arch + `version ≈ 4.5` (the version
        // gate keeps this from grabbing older `MiniCPMV` releases, which
        // ride a non-Qwen3 backbone).
        let isBareV =
            config.modelType == "minicpmv" || config.architecture == "MiniCPMV"
        if isBareV, let v = config.float("version"),
            abs(v - resamplerVersion) < versionEpsilon
        {
            return .v45
        }
        return nil
    }

    /// Whether a decoded `config.json` describes a MiniCPM 4.x checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        variant(for: config) != nil
    }

    /// Generation defaults per variant: 4.6 rides the Qwen3.5 backbone;
    /// 4.5 / o ride the Qwen3 dense backbone.
    public static func defaultGenerationParameters(for config: ModelConfig)
        -> GenerationParameters
    {
        switch variant(for: config) {
        case .v46: return Qwen35Hybrid.defaultGenerationParameters
        case .v45, .omni, .none: return Qwen3Dense.defaultGenerationParameters
        }
    }

    /// Capabilities the loaded model exposes. Image + video are always
    /// present; audio only when the omni checkpoint wired an audio tower
    /// (`vlm.supportsAudio`).
    public static func capabilities(for vlm: VisionModel) -> Set<Capability> {
        var caps = availableCapabilities
        if vlm.supportsAudio { caps.insert(.audioIn) }
        return caps
    }

    /// Build a `VisionModel` for whichever MiniCPM 4.x variant the config
    /// describes, joined by `VisionModel`'s cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        switch variant(for: config) {
        case .v46:
            return try loadVitMerger(
                config: config, weights: weights,
                options: options, device: device)
        case .v45:
            return try loadResampler(
                config: config, weights: weights, options: options,
                textPrefix: "llm.", visionPrefix: "vpm.",
                withAudio: false, device: device)
        case .omni:
            return try loadResampler(
                config: config, weights: weights, options: options,
                textPrefix: "language_model.", visionPrefix: "vision_tower.",
                withAudio: true, device: device)
        case .none:
            throw MiniCPM4Error.unsupportedConfig(
                "not a MiniCPM 4.x checkpoint")
        }
    }

    // ── v4.5 / o-4.5 — resampler tower (+ optional audio) ──────────────
    //
    // Identical assembly for both: a Qwen3-8B dense LLM (config at the TOP
    // level of config.json — no `text_config` sub-tree) + the SigLIP +
    // perceiver resampler vision tower. They differ only in the weight
    // prefixes (`llm.` / `vpm.` for 4.5; `language_model.` / `vision_tower.`
    // for o) and whether the audio tower is wired.
    private static func loadResampler(
        config: ModelConfig, weights: SafeTensorsBundle, options: LoadOptions,
        textPrefix: String, visionPrefix: String, withAudio: Bool,
        device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw MiniCPM4Error.missingConfig("vision_config")
        }

        // Text backbone — Qwen3-8B dense. `Qwen3Dense.loadModel` reads
        // top-level config keys and probes `model.*` / `lm_head` in its
        // bundle, so forward the config as the text config and re-wrap the
        // bundle at the variant's text prefix. A `quantization` block (when
        // present) is top-level, so embedding / linears / lm_head load
        // quantized; the bf16 base loads full-precision.
        let textConfig = ModelConfig(
            architecture: "Qwen3ForCausalLM", modelType: "qwen3",
            raw: config.raw)
        let textEngine = try Qwen3Dense.loadModel(
            config: textConfig, weights: weights.prefixed(textPrefix),
            options: options, device: device)

        let queryNum = config.int("query_num") ?? defaultQueryNum

        // Vision tower — SigLIP encoder + perceiver resampler.
        let vision = try MiniCPMOVision.load(
            visionConfig: visionConfig,
            runtimeImageSize: runtimeImageSize, textHidden: textEngine.hidden,
            queryNum: queryNum,
            visionWeights: weights.prefixed(visionPrefix),
            resamplerWeights: weights.prefixed("resampler."),
            quantization: config.quantization, device: device)

        // Audio tower — Whisper encoder + projector + avg-pool. Only wired
        // when the checkpoint declares an `audio_config` AND ships
        // `audio_tower.*` tensors (every shipped MiniCPM-o does).
        var audioTokenId: Int? = nil
        var audioEncode:
            (@Sendable (_ waveform: [Float], _ device: Device) -> Tensor)? = nil
        if withAudio, let audioConfig = config.subConfig("audio_config"),
            weights.has("audio_tower.conv1.weight")
        {
            let poolStep = config.int("audio_pool_step") ?? defaultAudioPoolStep
            let audio = try MiniCPM4Audio.load(
                weights: weights, audioConfig: audioConfig,
                textHidden: textEngine.hidden, poolStep: poolStep,
                quantization: config.quantization, device: device)
            audioTokenId = omniAudioPlaceholderTokenId
            audioEncode = { waveform, dev in
                audio.encode(waveform: waveform, device: dev)
            }
        }

        return try VisionModel(
            visionEncoder: vision, engine: textEngine,
            imageTokenId: resamplerImageTokenId,
            videoTokenId: resamplerVideoTokenId,
            normalization: .siglip,
            imageTokenCount: vision.outputTokenCount,
            audioTokenId: audioTokenId, audioEncode: audioEncode)
    }

    // ── v4.6 — SigLIP2-400M + vit_merger + merger + Qwen3.5 text ───────
    private static func loadVitMerger(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config"),
            config.nested("text_config") != nil
        else {
            throw MiniCPM4Error.missingConfig("vision_config / text_config")
        }

        // Text backbone: Qwen3.5 from `text_config` (the family reuses the
        // Qwen3.5 hybrid stack). mlx-community ships the text weights under
        // `language_model.model.*`; the Qwen3.5 loader probes that prefix
        // itself, so the full bundle is passed through unmodified.
        let textEngine = try MiniCPM4Text.load(
            config: config, weights: weights,
            options: options, device: device)

        // Vision tower: SigLIP2 encoder + `vit_merger` + `merger`. All
        // vision weights live at the top level — `vision_tower.*`,
        // `vit_merger.*`, `merger.*` — with no `model.` wrapper.
        let composed = try MiniCPM4ComposedEncoder.load(
            visionConfig: visionConfig,
            insertLayerId: config.int("insert_layer_id") ?? 6,
            mergerTimes: config.int("merger_times") ?? 1,
            runtimeImageSize: runtimeImageSize,
            textHidden: textEngine.hidden,
            visionWeights: weights.prefixed("vision_tower."),
            mergerWeights: weights.prefixed("merger."),
            vitMergerOverride: weights.prefixed("vit_merger."),
            device: device)

        return try VisionModel(
            visionEncoder: composed, engine: textEngine,
            imageTokenId: config.int("image_token_id") ?? defaultImageTokenId,
            videoTokenId: config.int("video_token_id") ?? defaultVideoTokenId,
            normalization: .siglip,
            imageTokenCount: composed.outputTokenCount)
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Unified MiniCPM 4.x family error — raised by the family orchestrator
/// and its vision / audio component stacks (`MiniCPMOVision`,
/// `MiniCPM4Audio`, `MiniCPM4ComposedEncoder`).
public enum MiniCPM4Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "MiniCPM-V: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "MiniCPM-V: unsupported config: \(m)"
        }
    }
}
