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
// MiniCPM-o 4.5 — OpenBMB's `MiniCPMO` omni checkpoint (`model_type:
// minicpmo`, version 4.5). A text + image + video + audio-IN model:
//
//   • LLM    — Qwen3-8B dense (`language_model.model.*` + `lm_head`),
//              4096 hidden, 36 layers, 32 heads, 8 KV, head_dim 128.
//   • Vision — SigLIP encoder (`vision_tower.*`, 1152 hidden, 27 layers)
//              + perceiver resampler (`resampler.*`) → 64 tokens / image.
//   • Audio  — Whisper-medium encoder (`audio_tower.*`, 1024 hidden,
//              24 layers) + avg-pool (stride 5) + 4-bit projector
//              (`audio_projection_layer.*`) → audio soft tokens.
//
// The TTS half (`tts.*`) is intentionally ignored — FFAI does audio-IN
// only. The vision / audio token runs are spliced into the text stream by
// the shared `VisionModel` at each `<unk>` placeholder (the id the
// MiniCPM-o processor scatters image / audio / video features at).
//
// Vision internals live in `Models/Vision/MiniCPMOVision.swift` +
// `Models/Vision/MiniCPMOResampler.swift`; the audio tower in
// `Models/MiniCPM4Audio.swift`. This file is the family orchestrator
// (load entry-point + public constants).
//
// Reference: `modeling_minicpmo.py` / `processing_minicpmo.py`
// (HF `openbmb/MiniCPM-o-4_5`, mlx-community `MiniCPM-o-4_5-4bit`).

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum MiniCPMO {
    public static let modelTypes: Set<String> = ["minicpmo"]
    public static let architectures: Set<String> = ["MiniCPMO"]

    /// Placeholder token the MiniCPM-o processor scatters image / audio /
    /// video features at — the `<unk>` token (`self.unk_token *
    /// feature_size` in `processing_minicpmo.py`, audio likewise wraps
    /// `<unk>` runs in `audio_start … audio_end`). The same id is used for
    /// DISTINCT per-modality placeholder ids. MiniCPM-o's processor scatters
    /// vision/audio features over a shared `<unk>` (128244) run, but the
    /// `VisionModel` splice validates + substitutes EACH modality by token
    /// id — a shared id makes an image run look like a video run (the splice
    /// can't tell them apart). The placeholder id is only a splice marker
    /// (it is replaced by the encoder embedding before the LLM ever sees
    /// it), so using the model's own distinct `<|image_pad|>` /
    /// `<|video_pad|>` / `<|audio|>` vocab ids is output-identical and lets
    /// a single prompt carry more than one modality unambiguously.
    public static let imageTokenId = 151_655  // <|image_pad|>
    public static let videoTokenId = 151_656  // <|video_pad|>
    public static let audioPlaceholderTokenId = 151_698  // <|audio|>

    /// Per-image / per-frame vision-token count (`query_num`, 64) — the
    /// perceiver resampler emits exactly this many text-hidden tokens.
    /// Source: config.json `query_num`.
    public static let defaultQueryNum = 64

    /// Average-pool stride applied to the projected audio features
    /// (`audio_pool_step`, 5). Source: config.json `audio_pool_step`.
    public static let defaultAudioPoolStep = 5

    /// Runtime single-tile image resolution. 448 = 32×32 patches at
    /// `patch_size: 14` (matches `slice_config.scale_resolution: 448`).
    public static let runtimeImageSize = 448

    /// Capabilities a MiniCPM-o checkpoint exposes: text + image + video +
    /// audio-IN.
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.imageIn, .videoIn, .audioIn])

    /// Whether a decoded `config.json` describes a MiniCPM-o checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        return false
    }

    /// Build a `VisionModel` from a `MiniCPMO` checkpoint: the Qwen3-8B
    /// dense LLM (`language_model.model.*`) + the SigLIP + resampler vision
    /// tower + the Whisper + projector audio tower, joined by
    /// `VisionModel`'s cross-modal splice (image / video / audio).
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw MiniCPMOError.missingConfig("vision_config")
        }

        // ── Text backbone — Qwen3-8B dense ────────────────────────────
        // MiniCPM-o declares the Qwen3 text hyper-parameters at the TOP
        // level of config.json (no `text_config` sub-tree); the weights
        // live under `language_model.model.*` + `language_model.lm_head`.
        // `Qwen3Dense.loadModel` reads top-level config keys and probes
        // `model.*` / `lm_head` in its bundle, so re-wrap the bundle view
        // at `language_model.` and forward the config as the text config.
        // `quantization` is already top-level, so the embedding / linears /
        // lm_head all load quantized.
        let textConfig = ModelConfig(
            architecture: "Qwen3ForCausalLM",
            modelType: "qwen3",
            raw: config.raw)
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try Qwen3Dense.loadModel(
            config: textConfig, weights: textWeights,
            options: options, device: device)

        let queryNum = config.int("query_num") ?? defaultQueryNum

        // ── Vision tower — SigLIP encoder + perceiver resampler ───────
        let visionWeights = weights.prefixed("vision_tower.")
        let resamplerWeights = weights.prefixed("resampler.")
        let vision = try MiniCPMOVision.load(
            visionConfig: visionConfig,
            runtimeImageSize: runtimeImageSize, textHidden: textEngine.hidden,
            queryNum: queryNum, visionWeights: visionWeights,
            resamplerWeights: resamplerWeights,
            quantization: config.quantization, device: device)

        // ── Audio tower — Whisper encoder + projector + avg-pool ──────
        // Optional: only wired when the checkpoint declares an
        // `audio_config` AND ships `audio_tower.*` tensors (every shipped
        // MiniCPM-o does; a vision-only variant would skip it).
        var audioTokenId: Int? = nil
        var audioEncode:
            (@Sendable (_ waveform: [Float], _ device: Device) -> Tensor)? = nil
        let hasAudioTower = weights.has("audio_tower.conv1.weight")
        if let audioConfig = config.subConfig("audio_config"), hasAudioTower {
            let poolStep = config.int("audio_pool_step") ?? defaultAudioPoolStep
            let audio = try MiniCPM4Audio.load(
                weights: weights, audioConfig: audioConfig,
                textHidden: textEngine.hidden, poolStep: poolStep,
                quantization: config.quantization, device: device)
            audioTokenId = audioPlaceholderTokenId
            // The audio encode hook: waveform → log-Mel front-end → Whisper
            // encoder → projector → avg-pool. Mirrors the reference
            // `get_audio_embedding`.
            audioEncode = { waveform, dev in
                audio.encode(waveform: waveform, device: dev)
            }
        }

        // ── Cross-modal splice ────────────────────────────────────────
        // Distinct per-modality placeholder ids (see the constants above) so
        // the splice disambiguates image / video / audio runs.
        return try VisionModel(
            visionEncoder: vision, engine: textEngine,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .siglip,
            imageTokenCount: vision.outputTokenCount,
            audioTokenId: audioTokenId,
            audioEncode: audioEncode)
    }
}

public enum MiniCPMOError: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "MiniCPM-o: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "MiniCPM-o: unsupported config: \(m)"
        }
    }
}
