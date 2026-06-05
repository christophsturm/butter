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
// MiniCPM-o vision tower — a SigLIP encoder (`vision_tower.*`) followed by
// the perceiver `MiniCPMOResampler` (`resampler.*`).
//
//   image ──SigLIP patch-embed + posEmb + 27 encoder layers + post_LN──▶
//         [num_patches, 1152]
//         ──resampler (64 learned queries cross-attend the patches)──▶
//         [64, 4096]  (text hidden)
//
// This is the MiniCPM-o analogue of `MiniCPM4ComposedEncoder`: the SigLIP
// half is identical (same `vision_tower.*` key layout), but the
// `vit_merger` + `merger` reduction path is replaced by the resampler.
// Each image yields exactly `query_num` (64) text-hidden tokens; video is
// encoded one frame at a time through the same path.
//
// Reference: `get_vllm_embedding` / `Resampler` in `modeling_minicpmo.py`
// (HF `openbmb/MiniCPM-o-4_5`).

import Foundation
import Metal

// ─── MiniCPMOVision ──────────────────────────────────────────────────

/// A `VisionEncoder` subclass: SigLIP encoder + perceiver resampler.
/// `encode(image:)` returns `[query_num, textHidden]` soft tokens.
final class MiniCPMOVision: VisionEncoder {
    /// The perceiver resampler that maps the SigLIP patch grid to a fixed
    /// `query_num` run of text-hidden soft tokens.
    let resampler: MiniCPMOResampler
    /// Patch grid side at the runtime image size (`imageSize / patchSize`).
    let gridSide: Int
    /// Per-image text-hidden token count (`query_num`, 64) — the count
    /// `VisionModel` splices per image-placeholder run.
    let outputTokenCount: Int

    init(
        config: VisionEncoderConfig,
        patchEmbedWeight: Tensor, patchEmbedBias: Tensor,
        positionEmbedding: Tensor, layers: [VisionEncoderLayer],
        postLayerNorm: LayerNorm,
        resampler: MiniCPMOResampler,
        gridSide: Int, outputTokenCount: Int, dtype: DType
    ) {
        self.resampler = resampler
        self.gridSide = gridSide
        self.outputTokenCount = outputTokenCount
        super.init(
            config: config, patchEmbedWeight: patchEmbedWeight,
            patchEmbedBias: patchEmbedBias, positionEmbedding: positionEmbedding,
            layers: layers, postLayerNorm: postLayerNorm,
            projection: nil, dtype: dtype)
    }

    /// Load the SigLIP encoder (`vision_tower.*`) + resampler
    /// (`resampler.*`). `runtimeImageSize` fixes the single-tile input
    /// resolution; the shipped `[stored², hidden]` position table is
    /// bilinearly interpolated to the runtime grid once at load (the
    /// reference uses bicubic — see `interpolatePositionEmbedding`).
    static func load(
        visionConfig: ModelConfig,
        runtimeImageSize: Int, textHidden: Int, queryNum: Int,
        visionWeights: SafeTensorsBundle, resamplerWeights: SafeTensorsBundle,
        quantization: ModelConfig.QuantizationConfig?,
        device: Device
    ) throws -> MiniCPMOVision {
        guard let hidden = visionConfig.int("hidden_size"),
            let storedImageSize = visionConfig.int("image_size"),
            let patchSize = visionConfig.int("patch_size"),
            let intermediate = visionConfig.int("intermediate_size"),
            let nLayers = visionConfig.int("num_hidden_layers"),
            let nHeads = visionConfig.int("num_attention_heads")
        else {
            throw MiniCPM4Error.missingConfig(
                "vision_config hidden_size / image_size / patch_size / "
                    + "intermediate_size / num_hidden_layers / num_attention_heads")
        }
        let eps = Float(visionConfig.float("layer_norm_eps") ?? 1e-6)

        precondition(
            runtimeImageSize % patchSize == 0,
            "MiniCPM-o: runtimeImageSize \(runtimeImageSize) not divisible "
                + "by patch_size \(patchSize)")
        let gridSide = runtimeImageSize / patchSize

        // ── Patch embed + interpolated position embedding ─────────────
        // patch_embedding.weight may ship in OHWI layout `[hidden, p, p, 3]`;
        // `Ops.conv2d` expects OIHW `[hidden, 3, p, p]` — transpose when the
        // trailing dim is the in-channel count (mirrors the MiniCPM-V 4.6 /
        // Gemma3 / Pixtral detection).
        let patchWRaw = try visionWeights.tensor(
            named: "embeddings.patch_embedding.weight")
        let patchW =
            patchWRaw.shape.count == 4 && patchWRaw.shape[3] == 3
            ? transposeOHWItoOIHW(patchWRaw)
            : patchWRaw
        let patchB = try visionWeights.tensor(
            named: "embeddings.patch_embedding.bias")
        let posEmbRaw = try visionWeights.tensor(
            named: "embeddings.position_embedding.weight")
        let storedGridSide = storedImageSize / patchSize
        let posEmb = interpolatePositionEmbedding(
            posEmbRaw, storedSide: storedGridSide,
            targetSide: gridSide, hidden: hidden, device: device)

        // ── SigLIP encoder layers — reuse VisionEncoderLayer ──────────
        // SigLIP v1 (`siglip_vision_model`, `gelu_pytorch_tanh`) → the
        // tanh GELU (`useExactGelu: false`).
        var layers: [VisionEncoderLayer] = []
        layers.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let p = "encoder.layers.\(i)"
            let ln1 = LayerNorm(
                weight: try visionWeights.tensor(named: "\(p).layer_norm1.weight"),
                bias: try visionWeights.tensor(named: "\(p).layer_norm1.bias"),
                eps: eps)
            let ln2 = LayerNorm(
                weight: try visionWeights.tensor(named: "\(p).layer_norm2.weight"),
                bias: try visionWeights.tensor(named: "\(p).layer_norm2.bias"),
                eps: eps)
            func lin(_ name: String) throws -> Linear {
                Linear(
                    weight: try visionWeights.tensor(named: "\(p).\(name).weight"),
                    bias: try visionWeights.tensor(named: "\(p).\(name).bias"))
            }
            layers.append(
                VisionEncoderLayer(
                    layerNorm1: ln1,
                    qProj: try lin("self_attn.q_proj"),
                    kProj: try lin("self_attn.k_proj"),
                    vProj: try lin("self_attn.v_proj"),
                    oProj: try lin("self_attn.out_proj"),
                    layerNorm2: ln2,
                    fc1: try lin("mlp.fc1"), fc2: try lin("mlp.fc2"),
                    hidden: hidden, nHeads: nHeads, intermediate: intermediate))
        }
        let postLN = LayerNorm(
            weight: try visionWeights.tensor(named: "post_layernorm.weight"),
            bias: try visionWeights.tensor(named: "post_layernorm.bias"),
            eps: eps)

        // ── Resampler ──────────────────────────────────────────────────
        let resampler = try MiniCPMOResampler.load(
            weights: resamplerWeights, embedDim: textHidden,
            visionHidden: hidden, quantization: quantization,
            eps: eps, device: device)
        precondition(
            resampler.numQueries == queryNum,
            "MiniCPM-o: resampler query count \(resampler.numQueries) ≠ "
                + "query_num \(queryNum)")

        let cfg = VisionEncoderConfig(
            imageSize: runtimeImageSize, patchSize: patchSize, hidden: hidden,
            intermediate: intermediate, nLayers: nLayers, nHeads: nHeads,
            layerNormEps: eps, textHidden: textHidden)

        return MiniCPMOVision(
            config: cfg, patchEmbedWeight: patchW, patchEmbedBias: patchB,
            positionEmbedding: posEmb, layers: layers, postLayerNorm: postLN,
            resampler: resampler, gridSide: gridSide,
            outputTokenCount: queryNum, dtype: patchW.dtype)
    }

    /// Encode one preprocessed `[1, 3, imageSize, imageSize]` NCHW image
    /// into `[query_num, textHidden]` soft tokens: SigLIP patch tokens via
    /// the base `VisionEncoder.encode`, then the resampler.
    ///
    /// The base `encode` runs patch-embed → posEmb → encoder layers →
    /// post_layernorm and (since `projection == nil`) returns the raw
    /// `[num_patches, visionHidden]` grid — exactly the resampler's input.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        let patches = super.encode(image: image, device: device)
        let numPatches = gridSide * gridSide
        precondition(
            patches.shape == [numPatches, config.hidden],
            "MiniCPM-o encode: SigLIP patches \(patches.shape) ≠ "
                + "[\(numPatches), \(config.hidden)]")
        return resampler.forward(
            patches: patches, gridH: gridSide, gridW: gridSide,
            device: device)
    }

    /// Encode a sequence of video frames into a single
    /// `[frameCount × query_num, textHidden]` token tensor. Each frame is
    /// resampled independently through the same single-image path; the
    /// per-frame `[query_num, textHidden]` slices are concatenated along
    /// dim 0 (the video splice substitutes one row per `video_pad` slot).
    override func encode(frames: [Tensor], device: Device = .shared) throws -> Tensor {
        precondition(
            !frames.isEmpty,
            "MiniCPMOVision.encode(frames:): expected at least one frame")
        let expectedShape = [1, config.inChannels, config.imageSize, config.imageSize]
        for (i, frame) in frames.enumerated() {
            precondition(
                frame.shape == expectedShape,
                "MiniCPMOVision.encode(frames:): frame[\(i)] shape "
                    + "\(frame.shape) ≠ \(expectedShape)")
        }
        let perFrame = frames.map { encode(image: $0, device: device) }

        let textHidden = config.textHidden
        let rowsPerFrame = outputTokenCount
        let totalRows = frames.count * rowsPerFrame
        let out = Tensor.empty(
            shape: [totalRows, textHidden],
            dtype: perFrame[0].dtype, device: device)
        let rowBytes = textHidden * perFrame[0].dtype.byteSize
        for (fi, ft) in perFrame.enumerated() {
            let srcPtr = ft.buffer.contents().advanced(by: ft.offset)
            let dstPtr = out.buffer.contents()
                .advanced(by: out.offset + fi * rowsPerFrame * rowBytes)
            dstPtr.copyMemory(from: srcPtr, byteCount: rowsPerFrame * rowBytes)
        }
        return out
    }
}
