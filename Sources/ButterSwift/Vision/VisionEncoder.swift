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
// VisionEncoder — the vision-transformer (ViT) stack a VLM runs an image
// through before splicing the resulting tokens into the text stream.
//
// Declared in for the capability API; lit up here initially.
// The architecture is the SigLIP / CLIP ViT shape every shipped VLM
// vision tower uses:
//
//   image  ──conv2d patch embed──▶  [num_patches, hidden]
//          ──+ learned position embedding──▶
//          ──▶ N × { LayerNorm → MHA (bidirectional) → +residual
//                    LayerNorm → MLP (GELU)          → +residual }
//          ──post-LayerNorm──▶
//          ──projection──▶  [num_patches, text_hidden]
//
// The encoder processes ALL patch tokens at once (vision attention is
// bidirectional — no causal mask, no KV cache), so it uses the
// multi-query `Ops.sdpaMulti` path rather than the single-token decode
// SDPA the text backbone uses.
//
// Qwen-VL's windowed-attention / dynamic-resolution towers are a
// superset of this; the family files that need them layer the extra
// behaviour on top. This file is the shared SigLIP-style core.

import Foundation
import Metal

// ─── Configuration ───────────────────────────────────────────────────

/// Static shape + hyper-parameters of a ViT vision tower, decoded from
/// the checkpoint's `vision_config`.
public struct VisionEncoderConfig: Sendable {
    /// Channels of the input image (3 for RGB).
    public let inChannels: Int
    /// Square input resolution the encoder expects (e.g. 224, 896).
    public let imageSize: Int
    /// Square patch side (14 for SigLIP/Qwen-VL, 16 for CLIP/Gemma-VL).
    public let patchSize: Int
    /// Encoder hidden dimension.
    public let hidden: Int
    /// Encoder feed-forward intermediate dimension.
    public let intermediate: Int
    /// Number of transformer blocks.
    public let nLayers: Int
    /// Number of attention heads per block.
    public let nHeads: Int
    /// LayerNorm epsilon.
    public let layerNormEps: Float
    /// Text-model hidden dimension the encoder output is projected into.
    /// When equal to `hidden` the projection is identity / absent.
    public let textHidden: Int

    public init(
        inChannels: Int = 3, imageSize: Int, patchSize: Int,
        hidden: Int, intermediate: Int, nLayers: Int, nHeads: Int,
        layerNormEps: Float = 1e-6, textHidden: Int
    ) {
        self.inChannels = inChannels
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.hidden = hidden
        self.intermediate = intermediate
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.layerNormEps = layerNormEps
        self.textHidden = textHidden
    }

    /// Patches along one axis (`imageSize / patchSize`).
    public var patchesPerSide: Int { imageSize / patchSize }
    /// Total patch tokens an image produces (`patchesPerSide²`).
    public var numPatches: Int { patchesPerSide * patchesPerSide }
    /// Per-head dimension.
    public var headDim: Int { hidden / nHeads }
}

// ─── Vision encoder ──────────────────────────────────────────────────

/// A SigLIP / CLIP-style ViT vision tower. Holds the patch-embed conv
/// weights, the learned position embedding, the encoder block stack,
/// the post-LayerNorm, and the projection into the text hidden dim.
// Non-final so family files can subclass it — e.g. Gemma 3 VL's
// composed encoder+projector tower (`Gemma3VLComposedEncoder`) overrides
// `encode` to append its multi-modal projector.
public class VisionEncoder: Module {
    public let config: VisionEncoderConfig

    /// conv2d patch-embed weight `[hidden, inChannels, patchSize, patchSize]`.
    public let patchEmbedWeight: Tensor
    /// conv2d patch-embed bias `[hidden]`.
    public let patchEmbedBias: Tensor
    /// Learned position embedding `[numPatches, hidden]`.
    public let positionEmbedding: Tensor
    /// Encoder blocks.
    public let layers: [VisionEncoderLayer]
    /// Post-encoder LayerNorm.
    /// Post-encoder LayerNorm. `nil` for towers consumed as an
    /// intermediate-layer feature extractor (e.g. LFM2-VL reads
    /// `hidden_states[vision_feature_layer]`, which is taken BEFORE
    /// `post_layernorm` — applying it shifts the feature distribution the
    /// projector was trained on).
    public let postLayerNorm: LayerNorm?
    /// Optional projection into the text hidden dim. `nil` when the
    /// encoder hidden already equals the text hidden.
    public let projection: Linear?

    public let dtype: DType

    public init(
        config: VisionEncoderConfig,
        patchEmbedWeight: Tensor, patchEmbedBias: Tensor,
        positionEmbedding: Tensor, layers: [VisionEncoderLayer],
        postLayerNorm: LayerNorm?, projection: Linear?,
        dtype: DType
    ) {
        self.config = config
        self.patchEmbedWeight = patchEmbedWeight
        self.patchEmbedBias = patchEmbedBias
        self.positionEmbedding = positionEmbedding
        self.layers = layers
        self.postLayerNorm = postLayerNorm
        self.projection = projection
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        out.append(("embeddings.patch_embedding.weight", patchEmbedWeight))
        out.append(("embeddings.patch_embedding.bias", patchEmbedBias))
        out.append(("embeddings.position_embedding.weight", positionEmbedding))
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("encoder.layers.\(i).\(k)", v))
            }
        }
        if let postLayerNorm {
            for (k, v) in postLayerNorm.parameters() { out.append(("post_layernorm.\(k)", v)) }
        }
        if let proj = projection {
            for (k, v) in proj.parameters() { out.append(("projection.\(k)", v)) }
        }
        return out
    }

    /// Optional family-specific preprocessing from a raw `RGBImage` to the
    /// NCHW pixel tensor `encode(image:)` consumes. Returns `nil` to fall
    /// back to the generic fixed-square `MediaUtils.preprocess`
    /// (`VisionModel.encodeImage`). Gemma 4 overrides this for its
    /// variable-resolution / aspect-ratio-preserving image processor, which
    /// produces a non-square `[1, 3, H, W]` tensor.
    public func preprocessImage(
        _ image: RGBImage, device: Device = .shared
    ) -> Tensor? {
        nil
    }

    /// Encode a preprocessed image into patch-token embeddings in the
    /// text hidden dim.
    ///
    /// `image` is a normalized NCHW tensor `[1, inChannels, imageSize,
    /// imageSize]` (the `MediaUtils.preprocess` output).
    /// Returns `[numPatches, textHidden]` — the tokens a VLM splices
    /// into its text stream. All GPU work is queued on a private command
    /// buffer that is committed + waited before returning, since callers
    /// consume the result on the CPU (the cross-modal splice).
    public func encode(image: Tensor, device: Device = .shared) -> Tensor {
        // Accept square OR non-square `[1, C, H, W]`. Native-resolution towers
        // (SigLIP2 NaFlex) feed a variable grid; the patch conv + token path
        // are grid-agnostic and the learned posEmb is resized to the actual
        // grid (`resolvedPositionEmbedding`). Fixed-square towers (Paligemma,
        // Gemma3) pass `imageSize×imageSize` and hit the raw-table fast path.
        precondition(
            image.shape.count == 4 && image.shape[0] == 1
                && image.shape[1] == config.inChannels,
            "VisionEncoder.encode: image shape \(image.shape) != [1,\(config.inChannels),H,W]")
        let inH = image.shape[2]
        let inW = image.shape[3]
        precondition(
            inH % config.patchSize == 0 && inW % config.patchSize == 0,
            "VisionEncoder.encode: \(inH)x\(inW) not divisible by patch \(config.patchSize)")
        let gridH = inH / config.patchSize
        let gridW = inW / config.patchSize
        let numPatches = gridH * gridW
        let cmd = device.makeCommandBuffer()

        // Patch embedding — fused conv (one thread per output element).
        // conv2d output is NCHW [1, hidden, gridH, gridW]; flatten the
        // spatial grid to [numPatches, hidden] token-major.
        let conv = Ops.conv2d(
            input: image, weight: patchEmbedWeight, bias: patchEmbedBias,
            strideH: config.patchSize, strideW: config.patchSize, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        var tokens = channelMajorToTokenMajor(
            conv, hidden: config.hidden, numPatches: numPatches, device: device)

        // Add the learned position embedding (bicubic-resized to the grid
        // when the image isn't the table's native square).
        let posEmb = resolvedPositionEmbedding(gridH: gridH, gridW: gridW, device: device)
        let cmd2 = device.makeCommandBuffer()
        var h = Ops.add(tokens, posEmb, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        // Run the encoder block stack. Each block commits its own work
        // internally (its attention core needs a CPU sync point).
        for layer in layers {
            let cmd = device.makeCommandBuffer()
            h = layer.forward(h, nTokens: numPatches, device: device, on: cmd)
        }

        // Post-encoder LayerNorm — skipped when the tower is read as an
        // intermediate-layer feature extractor (postLayerNorm == nil).
        if let postLayerNorm {
            let cmdN = device.makeCommandBuffer()
            h = Ops.layerNorm(
                h, weight: postLayerNorm.weight,
                bias: postLayerNorm.bias, eps: postLayerNorm.eps,
                nRows: numPatches, rowSize: config.hidden, on: cmdN)
            cmdN.commit()
            cmdN.waitUntilCompleted()
        }
        tokens = h

        // Project into the text hidden dim if the encoder hidden differs.
        guard let proj = projection else { return tokens }
        let cmd3 = device.makeCommandBuffer()
        var projected = Ops.gemm(
            weight: proj.weight, input: tokens, nRows: numPatches, on: cmd3)
        if let bias = proj.bias {
            let biasVals = bias.toFloatArray()
            var flat = [Float](repeating: 0, count: numPatches * config.textHidden)
            for r in 0 ..< numPatches {
                for c in 0 ..< config.textHidden {
                    flat[r * config.textHidden + c] = biasVals[c]
                }
            }
            let biasTiled = Tensor.empty(
                shape: [numPatches, config.textHidden], dtype: dtype)
            MediaUtils.copyFloats(flat, into: biasTiled)
            projected = Ops.add(projected, biasTiled, on: cmd3)
        }
        cmd3.commit()
        cmd3.waitUntilCompleted()
        return projected
    }

    /// The learned position embedding resolved to a `gridH × gridW` patch
    /// grid. Returns the raw `[posSize², hidden]` table when the grid is the
    /// table's native square (fixed-square towers); otherwise bicubic-resizes
    /// it to `[gridH·gridW, hidden]` (SigLIP2 NaFlex
    /// `resize_positional_embeddings`).
    private func resolvedPositionEmbedding(
        gridH: Int, gridW: Int, device: Device
    ) -> Tensor {
        let total = positionEmbedding.shape[0]
        let posSize = Int(Double(total).squareRoot().rounded())
        if gridH == posSize && gridW == posSize {
            return positionEmbedding
        }
        let table = positionEmbedding.toFloatArray()
        let resized = MediaUtils.bicubicResize(
            table, srcH: posSize, srcW: posSize, channels: config.hidden,
            dstH: gridH, dstW: gridW)
        return MediaUtils.makeTensor(
            from: resized, shape: [gridH * gridW, config.hidden],
            dtype: dtype, device: device)
    }

    /// Encode a sequence of frames as a temporally-batched token stream.
    /// Default implementation throws `VisionEncoderError.videoUnsupported`
    /// — towers that wire real video patches (Qwen 2/2.5/3 VL,
    /// MiniCPM-V, SmolVLM2, QwenOmni) override this with the temporal-
    /// patch unfold path. Per-model overrides decide how frames map to
    /// the temporal axis (typically `temporal_patch_size`-wide groups).
    public func encode(frames: [Tensor], device: Device = .shared) throws -> Tensor {
        throw VisionEncoderError.videoUnsupported(
            family: String(describing: type(of: self)))
    }

    /// Reinterpret a conv2d `[1, hidden, P, P]` channel-major output as
    /// `[numPatches, hidden]` token-major patch tokens. CPU transpose —
    /// patch counts are at most a few thousand.
    private func channelMajorToTokenMajor(
        _ conv: Tensor, hidden: Int, numPatches: Int, device: Device
    ) -> Tensor {
        let src = conv.toFloatArray()
        var dst = [Float](repeating: 0, count: numPatches * hidden)
        for c in 0 ..< hidden {
            for p in 0 ..< numPatches {
                dst[p * hidden + c] = src[c * numPatches + p]
            }
        }
        let out = Tensor.empty(
            shape: [numPatches, hidden], dtype: dtype,
            device: device)
        MediaUtils.copyFloats(dst, into: out)
        return out
    }
}

// `MediaUtils.copyFloats(_:into:)` lives in `Media/AudioPreprocessing.swift`
// (the shared host→GPU staging helper, used by both the vision encoder and
// the audio front-ends).

public enum VisionEncoderError: Error, CustomStringConvertible {
    /// The base `VisionEncoder.encode(frames:)` default throws this.
    /// Override on towers that wire the multi-frame temporal-patch path.
    case videoUnsupported(family: String)

    public var description: String {
        switch self {
        case .videoUnsupported(let family):
            return "VisionEncoder: \(family) does not implement video encode(frames:)."
                + " The single-image encode(image:) path is fine; video requires the"
                + " multi-frame temporal-patch unfold to be wired per-family."
        }
    }
}
