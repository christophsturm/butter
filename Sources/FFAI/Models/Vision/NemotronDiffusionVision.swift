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
// Nemotron-Labs-Diffusion-VLM — the diffusion vision-language
// orchestrator (`enum NemotronDiffusionVL`). Mirrors the per-family
// vision-file layout the rest of the Nemotron line uses (`NemotronVL`
// lives in `Models/Vision/NemotronHVision.swift`). The family entry-point
// metadata (`enum Nemotron` / `NemotronDiffusion`) lives in
// `Models/Nemotron.swift` + `Models/NemotronDiffusion.swift`; this
// orchestrator raises the NemotronDiffusion family's unified
// `NemotronDiffusionError`.
//
// The diffusion VLM composes types from THREE unrelated families: the
// Pixtral 2D-RoPE ViT vision tower, the Mistral3-style multi-modal
// projector (RMSNorm → 2×2 patch merger → linear_1 → GELU → linear_2),
// and the NemotronDiffusion tri-mode text backbone. `vision_config.
// model_type` is explicitly `"pixtral"`, so the shipped
// `PixtralVisionEncoder` is the correct tower and we reuse
// `Mistral3Projector` for the merger.
//
// Coherence-first port: vision attention runs on the CPU (already the
// case in `PixtralVisionEncoder`); the text backbone runs through the
// existing GPU-accelerated `NemotronDiffusionDense` engine. Image-only
// inference for now; the `forwardBlock` diffusion / self-speculation
// paths take the spliced embeddings unchanged.

import Foundation
import Metal

// ─── NemotronDiffusionVL — diffusion vision-language orchestrator ────

public enum NemotronDiffusionVL {
    /// `model_type` labels this orchestrator recognises.
    public static let modelTypes: Set<String> = ["nemotron_labs_diffusion_vlm"]
    /// `architectures[0]` labels this orchestrator recognises.
    public static let architectures: Set<String> = ["NemotronLabsDiffusionVLMModel"]

    /// Patch-merger size default for the Nemotron-Labs-Diffusion VLM —
    /// the shipped 8B config sets `spatial_merge_size: 2`.
    public static let defaultSpatialMergeSize = 2
    /// `image_token_id` fallback. The 8B config doesn't specify one
    /// explicitly; the tokenizer's `[IMG]` (Pixtral lineage) resolves
    /// to id 10. Override via `config.image_token_id` /
    /// `image_token_index` when a checkpoint sets it.
    public static let defaultImageTokenId = 10

    /// Build a `VisionModel` from a Nemotron-Labs-Diffusion-VLM
    /// checkpoint: the Pixtral 2D-RoPE ViT + the Mistral3 patch-merger
    /// projector + the NemotronDiffusion tri-mode text backbone,
    /// joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config")
        else {
            throw NemotronDiffusionError.missingConfig
        }

        // ── Text backbone — NemotronDiffusion tri-mode engine ──
        // The diffusion VLM stores text hyper-parameters at the *root*
        // of the config (not under a nested `text_config` — confirmed
        // against the shipped 8B config.json). The text weights live
        // under `language_model.*` (standard VLM namespacing) so the
        // existing `encoder.*` / `diffusion_head.*` keys the standalone
        // loader expects become `language_model.encoder.*` /
        // `language_model.diffusion_head.*` once we prefix the bundle.
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try NemotronDiffusionDense.loadModel(
            config: config, weights: textWeights,
            options: options, device: device)

        // ── Vision tower — Pixtral ViT ──
        // `vision_config.model_type` is `"pixtral"`. Reuse the shipped
        // `PixtralVisionEncoder` verbatim. Weights live under
        // `vision_tower.*` (the mlx-community VLM convention); the
        // encoder reads them as `vision_model.*` so the prefix strip
        // matches what Mistral3 does for the same tower.
        let visionWeights = weights.prefixed("vision_tower.")
        let visionCfg = try PixtralVisionConfig.decode(visionConfig)
        let vision = try PixtralVisionEncoder.load(
            cfg: visionCfg, weights: visionWeights,
            dtype: textEngine.dtype, device: device)

        // ── Multi-modal projector — Mistral3 patch-merger ──
        // The 8B config sets `multimodal_projector_bias: false` and
        // `projector_hidden_act: "gelu"` — Mistral3Projector's defaults
        // match (GELU between linear_1 and linear_2, biases optional).
        // The merger reduces token count by spatialMergeSize².
        let spatialMergeSize = config.int("spatial_merge_size") ?? defaultSpatialMergeSize
        let projectorBias = config.bool("multimodal_projector_bias") ?? false
        let projQuant = weights.mistral3ProjectorQuantization
        let projector = try Mistral3Projector.load(
            visionHidden: visionCfg.hiddenSize,
            textHidden: textEngine.hidden,
            spatialMergeSize: spatialMergeSize,
            hasBias: projectorBias,
            quantization: projQuant,
            weights: weights, device: device)

        // Image token count for the default square-image path; the
        // composed encoder adjusts at encode time for dynamic images.
        let mergedPatches =
            visionCfg.numPatches
            / (spatialMergeSize * spatialMergeSize)

        let imageTokenId =
            config.int("image_token_id")
            ?? config.int("image_token_index")
            ?? defaultImageTokenId

        let composedTower = Mistral3ComposedTower(
            encoder: vision, projector: projector,
            visionCfg: visionCfg, spatialMergeSize: spatialMergeSize,
            textHidden: textEngine.hidden, dtype: textEngine.dtype)

        return try VisionModel(
            visionEncoder: composedTower.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .clip,
            imageTokenCount: mergedPatches)
    }
}
