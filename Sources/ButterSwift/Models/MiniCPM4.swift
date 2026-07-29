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
//
// This file is the **main model interface** for the family (the same shape
// as `Qwen2.swift` / `Qwen3.swift`):
//   • the family enum `MiniCPM4` (modelTypes, architectures, the shared
//     family constants, and the `variant(for:)` dispatch),
//   • the `MiniCPM4Variant` protocol every concrete variant conforms to —
//     each declaring the capabilities + generation defaults it exposes and
//     loading its full multimodal assembly into a `VisionModel`,
//   • the unified `MiniCPM4Error` type the family's loaders raise.
//
// Unlike the text-only families (whose `Variant` loads a text engine), every
// MiniCPM 4.x checkpoint is multimodal, so each concrete variant loads a
// whole image / video (+ audio) assembly and returns a `VisionModel`:
//
//   • `MiniCPM4V45`  (`MiniCPMV`, version 4.5)            — image + video.
//        Qwen3-8B dense (`llm.model.*`) + SigLIP (`vpm.*`) + perceiver
//        resampler (`resampler.*`) → 64 tokens/image.
//   • `MiniCPM4V46`  (`MiniCPMV4_6ForConditionalGeneration`) — image + video.
//        SigLIP2-400M (`vision_tower.*`) + `vit_merger` (mid-stack window
//        cross-attn merger) + final `merger` (2×2 → text hidden) + the
//        Qwen3.5 text backbone.
//   • `MiniCPM4Omni` (`MiniCPMO`, version 4.5)            — + audio-IN.
//        Qwen3-8B dense (`language_model.model.*`) + SigLIP
//        (`vision_tower.*`) + the SAME perceiver resampler + a
//        Whisper-medium `audio_tower` + avg-pool + 4-bit projector. The TTS
//        half (`tts.*`) is ignored — Butter does audio-IN only.
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
// The vision / audio token runs are spliced into the text stream by the
// shared `VisionModel`.
//

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum MiniCPM4 {
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
    /// video) — the family baseline. The omni variant widens this with
    /// `.audioIn`; the audio capability is then gated at load time on
    /// whether the audio tower actually wired (see
    /// `MiniCPM4Variant.capabilities(for:)`).
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

    /// Detect the concrete variant type a config describes; `nil` when it
    /// is not a MiniCPM 4.x checkpoint. Backs both `handles(_:)` and the
    /// throwing `variant(for:)` dispatch.
    static func variantType(for config: ModelConfig) -> (any MiniCPM4Variant.Type)? {
        if config.modelType == "minicpmo" || config.architecture == "MiniCPMO" {
            return MiniCPM4Omni.self
        }
        if config.modelType == "minicpmv4_6"
            || config.architecture == "MiniCPMV4_6ForConditionalGeneration"
        {
            return MiniCPM4V46.self
        }
        // v4.5 — the bare `MiniCPMV` arch + `version ≈ 4.5` (the version
        // gate keeps this from grabbing older `MiniCPMV` releases, which
        // ride a non-Qwen3 backbone).
        let isBareV =
            config.modelType == "minicpmv" || config.architecture == "MiniCPMV"
        if isBareV, let v = config.float("version"),
            abs(v - resamplerVersion) < versionEpsilon
        {
            return MiniCPM4V45.self
        }
        return nil
    }

    /// Pick the concrete variant a config describes (the family's Variant
    /// dispatch). Throws when the config is not a MiniCPM 4.x checkpoint.
    public static func variant(for config: ModelConfig) throws -> any MiniCPM4Variant.Type {
        guard let variant = variantType(for: config) else {
            throw MiniCPM4Error.unsupportedConfig("not a MiniCPM 4.x checkpoint")
        }
        return variant
    }

    /// Whether a decoded `config.json` describes a MiniCPM 4.x checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        variantType(for: config) != nil
    }

    // ── Shared assembly loaders ───────────────────────────────────────
    //
    // Both resampler variants (4.5 / o-4.5) share `loadResampler`; 4.6 owns
    // `loadVitMerger`. They are `fileprivate` so only the concrete variant
    // types below call them.

    /// v4.5 / o-4.5 — Qwen3-8B dense LLM + SigLIP + perceiver resampler
    /// (+ optional Whisper audio tower).
    ///
    /// Identical assembly for both: a Qwen3-8B dense LLM (config at the TOP
    /// level of config.json — no `text_config` sub-tree) + the SigLIP +
    /// perceiver resampler vision tower. They differ only in the weight
    /// prefixes (`llm.` / `vpm.` for 4.5; `language_model.` / `vision_tower.`
    /// for o) and whether the audio tower is wired.
    fileprivate static func loadResampler(
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

    /// v4.6 — SigLIP2-400M + `vit_merger` + `merger` + Qwen3.5 text.
    fileprivate static func loadVitMerger(
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

// ─── Variant protocol ────────────────────────────────────────────────

/// Every concrete MiniCPM 4.x variant declares the capabilities it exposes
/// and the generation defaults it recommends, and loads its full multimodal
/// assembly into a `VisionModel`. Mirrors `Qwen2Variant` / `Qwen3Variant`,
/// except `load` returns a `VisionModel` (the whole family is multimodal —
/// there is no text-only variant).
public protocol MiniCPM4Variant {
    /// Capabilities this variant declares — the maximum it can expose.
    /// Audio is refined at load time (see `capabilities(for:)`).
    static var availableCapabilities: Set<Capability> { get }
    /// Generation defaults for this variant's text backbone.
    static var defaultGenerationParameters: GenerationParameters { get }
    /// Build the variant's image / video (+ audio) assembly, joined by
    /// `VisionModel`'s cross-modal splice.
    static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel
    /// Refine the declared capabilities against the loaded model.
    static func capabilities(for vlm: VisionModel) -> Set<Capability>
}

extension MiniCPM4Variant {
    /// The omni variant only exposes `.audioIn` when its audio tower
    /// actually wired (`vlm.supportsAudio`) — a checkpoint shipped without
    /// `audio_tower.*` tensors loads image / video only. The image / video
    /// variants declare no `.audioIn`, so this is a no-op for them.
    public static func capabilities(for vlm: VisionModel) -> Set<Capability> {
        var caps = availableCapabilities
        if !vlm.supportsAudio { caps.remove(.audioIn) }
        return caps
    }
}

// ─── Concrete variants ───────────────────────────────────────────────

/// MiniCPM-V 4.5 — image + video. Qwen3-8B dense (`llm.*`) + SigLIP
/// (`vpm.*`) + perceiver resampler.
public enum MiniCPM4V45: MiniCPM4Variant {
    public static let availableCapabilities = MiniCPM4.availableCapabilities
    public static let defaultGenerationParameters =
        Qwen3Dense.defaultGenerationParameters

    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        try MiniCPM4.loadResampler(
            config: config, weights: weights, options: options,
            textPrefix: "llm.", visionPrefix: "vpm.",
            withAudio: false, device: device)
    }
}

/// MiniCPM-V 4.6 — image + video. SigLIP2-400M + `vit_merger` + `merger`
/// + the Qwen3.5 hybrid text backbone.
public enum MiniCPM4V46: MiniCPM4Variant {
    public static let availableCapabilities = MiniCPM4.availableCapabilities
    public static let defaultGenerationParameters =
        Qwen35Hybrid.defaultGenerationParameters

    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        try MiniCPM4.loadVitMerger(
            config: config, weights: weights,
            options: options, device: device)
    }
}

/// MiniCPM-o 4.5 — image + video + audio-IN. Qwen3-8B dense
/// (`language_model.*`) + SigLIP (`vision_tower.*`) + the resampler + a
/// Whisper-medium audio tower. `.audioIn` is gated on the audio tower
/// actually wiring (see `MiniCPM4Variant.capabilities(for:)`).
public enum MiniCPM4Omni: MiniCPM4Variant {
    public static let availableCapabilities =
        MiniCPM4.availableCapabilities.union([.audioIn])
    public static let defaultGenerationParameters =
        Qwen3Dense.defaultGenerationParameters

    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        try MiniCPM4.loadResampler(
            config: config, weights: weights, options: options,
            textPrefix: "language_model.", visionPrefix: "vision_tower.",
            withAudio: true, device: device)
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
