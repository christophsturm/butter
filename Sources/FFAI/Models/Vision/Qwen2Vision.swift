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
// Qwen 2-VL — vision tower internals.
//
// This file contains the `Qwen2VL` vision-language orchestrator (the
// load entry-point + the `<|image_pad|>` / `<|video_pad|>` token ids the
// splice needs) plus the tower internals: `Qwen2VLVisionConfig`,
// `Qwen2VLVisionBlock`, `Qwen2VLVisionModel`, `Qwen2VLComposedEncoder`,
// and all supporting CPU / GPU helpers for the dynamic-resolution ViT
// (Conv3d patch embed, LayerNorm pre-norms, M-RoPE, full bidirectional
// attention, patch merger). The family root (enum + `Qwen2Variant`
// protocol + `Qwen2Error`) lives in `Models/Qwen2.swift`.

import Foundation
import Metal

// ─── Qwen 2-VL orchestrator ──────────────────────────────────────────

/// The Qwen2-VL vision-language orchestrator: ties the dynamic-resolution
/// vision tower (below) to the Qwen 2 text backbone, joined by
/// `VisionModel`'s cross-modal splice. Declares the capabilities + the
/// generation defaults the VL checkpoint exposes (per the family
/// convention — see `Qwen2Variant`).
public enum Qwen2VL {
    /// `image_token_id` default for Qwen 2-VL checkpoints.
    public static let defaultImageTokenId = 151_655
    /// `video_token_id` default for Qwen 2-VL checkpoints
    /// (`<|video_pad|>` — same id as Qwen 2.5-VL).
    public static let defaultVideoTokenId = 151_656

    /// Capabilities a Qwen 2-VL checkpoint declares to the loader.
    /// Text + image + video — the vision tower's Conv3d patch embed and
    /// temporal-patch unfold handle both single-image and multi-frame
    /// video paths.
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.imageIn, .videoIn])

    /// Generation defaults for the VL path — the Qwen 2 text-variant
    /// defaults (the VL checkpoint shares the Qwen 2 dense backbone).
    public static let defaultGenerationParameters =
        Qwen2Dense.defaultGenerationParameters

    /// Build a `VisionModel` from a `Qwen2VLForConditionalGeneration`
    /// checkpoint: the dynamic-resolution vision tower + the Qwen 2 text
    /// backbone (via the `Qwen2Dense` text variant), joined by the splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Qwen2Error.missingConfig
        }

        // ── Text backbone ──
        // Qwen2-VL stores text hyper-parameters at the top level (same
        // layout as a standalone Qwen 2 checkpoint). Text weights are
        // under `language_model.*`; the `Qwen2Dense` text variant loads
        // the Qwen 2.x architecture (tied embeddings, SiLU MLP, GQA).
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try Qwen2Dense.loadModel(
            config: config, weights: textWeights, options: options, device: device)

        // ── Vision tower ──
        // Vision weights are under `vision_tower.*`.
        let visionWeights = weights.prefixed("vision_tower.")
        let vision = try Qwen2VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, dtype: textEngine.dtype,
            quantization: config.quantization, device: device)

        let imageTokenId = config.int("image_token_id") ?? defaultImageTokenId
        let videoTokenId = config.int("video_token_id") ?? defaultVideoTokenId
        return try VisionModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .clip,
            imageTokenCount: vision.mergedTokenCount)
    }
}

// ─── Qwen 2-VL vision tower ──────────────────────────────────────────

/// Static shape of the Qwen 2-VL vision tower, decoded from
/// `vision_config`.
struct Qwen2VLVisionConfig {
    let depth: Int
    /// Vision hidden dim (from `embed_dim` in the config).
    let hidden: Int
    /// MLP intermediate dim (`mlp_ratio × hidden`).
    let intermediate: Int
    /// Text projection output dim (`hidden_size` in the vision_config).
    let outHidden: Int
    let numHeads: Int
    let patchSize: Int
    let spatialMergeSize: Int
    let temporalPatchSize: Int
    let inChannels: Int
    let layerNormEps: Float

    /// Per-head dimension.
    var headDim: Int { hidden / numHeads }
    /// `mergeSize²` — patches pooled into one merged token.
    var mergeUnit: Int { spatialMergeSize * spatialMergeSize }

    static func decode(_ c: ModelConfig) throws -> Qwen2VLVisionConfig {
        // `embed_dim` is the vision hidden (Qwen2-VL naming convention).
        guard let depth = c.int("depth"),
            let hidden = c.int("embed_dim"),
            let numHeads = c.int("num_heads"),
            let patchSize = c.int("patch_size"),
            let mergeSize = c.int("spatial_merge_size")
        else {
            throw Qwen2Error.missingConfig
        }
        // `mlp_ratio` scales the hidden dim to the MLP intermediate.
        let mlpRatio = c.float("mlp_ratio") ?? 4.0
        let intermediate = Int(Double(hidden) * mlpRatio)
        // `hidden_size` is the text backbone hidden (output of the merger).
        let outHidden = c.int("hidden_size") ?? hidden
        return Qwen2VLVisionConfig(
            depth: depth, hidden: hidden, intermediate: intermediate,
            outHidden: outHidden, numHeads: numHeads, patchSize: patchSize,
            spatialMergeSize: mergeSize,
            temporalPatchSize: c.int("temporal_patch_size") ?? 2,
            inChannels: c.int("in_chans") ?? c.int("in_channels") ?? 3,
            layerNormEps: Float(c.float("layer_norm_eps") ?? 1e-6))
    }
}

/// One Qwen 2-VL vision block: LayerNorm → full MHA + M-RoPE → residual,
/// LayerNorm → GELU MLP → residual. Held as plain weight tensors; the
/// forward runs CPU attention + GPU GEMMs.
final class Qwen2VLVisionBlock {
    let norm1: LayerNorm
    let norm2: LayerNorm
    let qkv: Linear  // fused [3·hidden, hidden] (+ bias)
    let proj: Linear  // [hidden, hidden] (+ bias)
    let fc1: Linear  // GELU-MLP up   [paddedIntermediate, hidden]
    let fc2: Linear  // GELU-MLP down [hidden, paddedIntermediate]
    let cfg: Qwen2VLVisionConfig
    /// MLP intermediate dim rounded up to the GEMM K-tile width — `fc1`'s
    /// output rows are zero-extended and `fc2`'s input columns zero-padded
    /// to it, so the `fc2` projection's `inDim` is aligned.
    let paddedIntermediate: Int

    init(
        norm1: LayerNorm, norm2: LayerNorm, qkv: Linear, proj: Linear,
        fc1: Linear, fc2: Linear,
        paddedIntermediate: Int, cfg: Qwen2VLVisionConfig
    ) {
        self.norm1 = norm1
        self.norm2 = norm2
        self.qkv = qkv
        self.proj = proj
        self.fc1 = fc1
        self.fc2 = fc2
        self.paddedIntermediate = paddedIntermediate
        self.cfg = cfg
    }

    /// Forward `[nTokens, hidden]` activations through one block.
    /// `cosTable` / `sinTable` are the precomputed per-token rotary
    /// tables `[nTokens, headDim]`.
    func forward(
        _ h: Tensor, nTokens: Int,
        cosTable: [Float], sinTable: [Float],
        device: Device
    ) -> Tensor {
        let hidden = cfg.hidden
        // ── Attention sub-block ──
        let cmd = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            h, weight: norm1.weight, bias: norm1.bias,
            eps: norm1.eps, nRows: nTokens,
            rowSize: hidden, on: cmd)
        let qkvOut = projectRows(
            qkv, normed, nTokens: nTokens,
            outDim: 3 * hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let attn = attention(
            qkv: qkvOut, nTokens: nTokens,
            cosTable: cosTable, sinTable: sinTable,
            device: device)

        // ── Residual + MLP sub-block ──
        let cmd2 = device.makeCommandBuffer()
        let attnProj = projectRows(
            proj, attn, nTokens: nTokens,
            outDim: hidden, on: cmd2)
        let postAttn = Ops.add(h, attnProj, on: cmd2)
        let normed2 = Ops.layerNorm(
            postAttn, weight: norm2.weight,
            bias: norm2.bias, eps: norm2.eps,
            nRows: nTokens, rowSize: hidden, on: cmd2)
        let up = projectRows(
            fc1, normed2, nTokens: nTokens,
            outDim: paddedIntermediate, on: cmd2)
        let act = Ops.gelu(up, on: cmd2)
        let down = projectRows(
            fc2, act, nTokens: nTokens,
            outDim: hidden, on: cmd2)
        let result = Ops.add(postAttn, down, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return result
    }

    /// Full bidirectional multi-head attention with M-RoPE on the **GPU**
    /// `sdpaBidirectional` kernel. `qkv` is the fused `[nTokens, 3·hidden]`
    /// projection. Returns the context token-major `[nTokens, hidden]`.
    ///
    /// Hybrid CPU→GPU split (mirrors the Gemma 4 vision migration, #23):
    ///   1. **CPU (cheap, `O(n·d)`):** split q/k/v per head and apply Qwen's
    ///      M-RoPE from the precomputed `cos`/`sin` tables
    ///      (`out[d] = x[d]·cos − x[d±half]·sin`, GPT-NeoX rotate-half),
    ///      producing token-major q/k/v `[nTokens, nHeads·headDim]`. Kept on
    ///      CPU so the rope numerics stay byte-identical to the validated
    ///      table build; it is negligible next to the attention.
    ///   2. **GPU (`O(n²·d)`, the former CPU pin):** transpose k/v
    ///      token-major → head-major (`Ops.transposeTokensToHeads`), then
    ///      `Ops.sdpaBidirectional` (MHA, `nKVHeads == nQHeads`), scale
    ///      `1/√headDim`, baseKV 0, `kvStride == nTokens`. The
    ///      `ffai_sdpa_bidirectional_d80_*` kernel is GPU-correct at this
    ///      production geometry (`sdpa_bidirectional_gpu_correctness.rs`),
    ///      so the native-resolution grid (thousands of patches) no longer
    ///      pins the cores.
    private func attention(
        qkv: Tensor, nTokens: Int,
        cosTable: [Float], sinTable: [Float],
        device: Device
    ) -> Tensor {
        let nHeads = cfg.numHeads
        let headDim = cfg.headDim
        let hidden = cfg.hidden
        let half = headDim / 2
        let qkvA = qkv.toFloatArray()

        // ── Stage 1 (CPU): split per head + M-RoPE → token-major q/k/v ──
        // Each (head, token) owns a disjoint `[dst, dst+headDim)` slot, so
        // the concurrent writes are race-free. `nonisolated(unsafe)`: the
        // index math guarantees no aliasing.
        nonisolated(unsafe) var qTok = [Float](repeating: 0, count: nTokens * hidden)
        nonisolated(unsafe) var kTok = [Float](repeating: 0, count: nTokens * hidden)
        nonisolated(unsafe) var vTok = [Float](repeating: 0, count: nTokens * hidden)
        qkvA.withUnsafeBufferPointer { srcBuf in
            nonisolated(unsafe) let s = srcBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nHeads * nTokens) { work in
                let head = work / nTokens
                let t = work % nTokens
                let hOff = head * headDim
                let qBase = t * 3 * hidden + hOff
                let kBase = t * 3 * hidden + hidden + hOff
                let vBase = t * 3 * hidden + 2 * hidden + hOff
                let cb = t * headDim
                let dst = t * hidden + hOff
                for d in 0 ..< headDim {
                    let c = cosTable[cb + d]
                    let sn = sinTable[cb + d]
                    // Rotate-half within the head's headDim slice.
                    let qRot = d < half ? -s[qBase + d + half] : s[qBase + d - half]
                    let kRot = d < half ? -s[kBase + d + half] : s[kBase + d - half]
                    qTok[dst + d] = s[qBase + d] * c + qRot * sn
                    kTok[dst + d] = s[kBase + d] * c + kRot * sn
                    vTok[dst + d] = s[vBase + d]
                }
            }
        }

        // ── Stage 2 (GPU): bidirectional MHA via sdpaBidirectional ──
        let qT = Tensor.empty(shape: [nTokens, hidden], dtype: qkv.dtype, device: device)
        let kT = Tensor.empty(shape: [nTokens, hidden], dtype: qkv.dtype, device: device)
        let vT = Tensor.empty(shape: [nTokens, hidden], dtype: qkv.dtype, device: device)
        MediaUtils.copyFloats(qTok, into: qT)
        MediaUtils.copyFloats(kTok, into: kT)
        MediaUtils.copyFloats(vTok, into: vT)

        let cmd = device.makeCommandBuffer()
        let kHead = Ops.transposeTokensToHeads(
            kT, nTokens: nTokens, nHeads: nHeads, headDim: headDim, on: cmd)
        let vHead = Ops.transposeTokensToHeads(
            vT, nTokens: nTokens, nHeads: nHeads, headDim: headDim, on: cmd)
        let scale = 1.0 / Float(Double(headDim).squareRoot())
        let attnOut = Ops.sdpaBidirectional(
            q: qT, k: kHead, v: vHead,
            nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
            baseKV: 0, nQuery: nTokens, kvStride: nTokens,
            scale: scale, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        // `[nTokens, nHeads, headDim]` contiguous == `[nTokens, hidden]`.
        return attnOut.reshaped(to: [nTokens, hidden])
    }

    /// Apply a `Linear` to every row of `[nTokens, *]` via `Ops.gemm`,
    /// then broadcast-add the bias if present.
    private func projectRows(
        _ linear: Linear, _ x: Tensor, nTokens: Int,
        outDim: Int, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let y = Ops.gemm(weight: linear.weight, input: x, nRows: nTokens, on: cmd)
        guard let bias = linear.bias else { return y }
        return addRowBias(
            y, bias: bias, nRows: nTokens,
            rowSize: outDim, on: cmd)
    }
}

/// The Qwen 2-VL vision tower. Holds the patch-embed projection, the block
/// stack, and the patch-merger; `encode` runs the full forward and returns
/// merged tokens in the text hidden dim.
final class Qwen2VLVisionModel: @unchecked Sendable {
    let cfg: Qwen2VLVisionConfig
    /// Flattened patch-embed weight `[hidden, patchDimPadded]`.
    let patchEmbedWeight: Tensor
    /// `patchDim` padded up to a multiple of the GEMM K-tile width.
    let patchDimPadded: Int
    let blocks: [Qwen2VLVisionBlock]
    /// Patch-merger LayerNorm over the pre-merge hidden (key: `merger.ln_q`).
    let mergerNorm: LayerNorm
    /// Patch-merger MLP: `[mergeUnit·hidden] → [mergeUnit·hidden] → [outHidden]`.
    let mergerFC1: Linear
    let mergerFC2: Linear
    let textHidden: Int
    let dtype: DType
    /// Patch grid side (square test image at the encoder's fixed size).
    let gridSide: Int

    /// Number of merged vision tokens one image contributes.
    var mergedTokenCount: Int {
        let llmSide = gridSide / cfg.spatialMergeSize
        return llmSide * llmSide
    }

    /// Number of merged vision tokens a video of `frameCount` frames
    /// contributes — `(T/tp) * (gridSide/m)²`. Used by callers to
    /// compute placeholder counts.
    func mergedTokenCount(frameCount: Int) -> Int {
        MediaUtils.mergedVideoTokenCount(
            frameCount: frameCount, gridSide: gridSide,
            spatialMergeSize: cfg.spatialMergeSize,
            temporalPatchSize: cfg.temporalPatchSize)
    }

    /// The standard Qwen-family `smartResize` pixel bounds for this config
    /// (`factor = patch · merge`, HF default token budget). Shared by the
    /// `preprocessImage` override and the placeholder-count helper.
    static func smartResizeBounds(
        cfg: Qwen2VLVisionConfig
    ) -> MediaUtils.SmartResizeBounds {
        MediaUtils.SmartResizeBounds.qwen(
            patchSize: cfg.patchSize, mergeSize: cfg.spatialMergeSize)
    }

    /// The native `(gridH, gridW)` patch grid `smartResize` selects for an
    /// image of the given pixel dims, plus the merged vision-token count it
    /// yields (`(gridH/m)·(gridW/m)`). Pure geometry — lets a caller size
    /// the image-placeholder run before encoding (the variable-resolution
    /// analogue of the fixed `mergedTokenCount`). Throws on an extreme
    /// aspect ratio (`MediaUtils.SmartResizeError`).
    static func smartResizeGrid(
        width: Int, height: Int, cfg: Qwen2VLVisionConfig
    ) throws -> (gridH: Int, gridW: Int, mergedTokens: Int) {
        try MediaUtils.smartResizeGrid(
            width: width, height: height,
            patchSize: cfg.patchSize, spatialMergeSize: cfg.spatialMergeSize)
    }

    /// Round `frameCount` up to a multiple of `temporal_patch_size` by
    /// repeating the last frame (matches the mlx-vlm reference's
    /// `patchify` padding).
    func paddedFrameCount(_ frameCount: Int) -> Int {
        MediaUtils.paddedFrameCount(
            frameCount, temporalPatchSize: cfg.temporalPatchSize)
    }

    init(
        cfg: Qwen2VLVisionConfig, patchEmbedWeight: Tensor,
        patchDimPadded: Int,
        blocks: [Qwen2VLVisionBlock], mergerNorm: LayerNorm,
        mergerFC1: Linear, mergerFC2: Linear, textHidden: Int,
        dtype: DType, gridSide: Int
    ) {
        self.cfg = cfg
        self.patchEmbedWeight = patchEmbedWeight
        self.patchDimPadded = patchDimPadded
        self.blocks = blocks
        self.mergerNorm = mergerNorm
        self.mergerFC1 = mergerFC1
        self.mergerFC2 = mergerFC2
        self.textHidden = textHidden
        self.dtype = dtype
        self.gridSide = gridSide
    }

    /// The patch grid side for a square image at the standard Qwen2-VL
    /// test resolution. 28×28 patches at patch-14 → 392×392 px; the grid
    /// divides the merge size cleanly (`28 % 2 == 0`).
    static let defaultGridSide = 28

    static func load(
        visionConfig: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundle, dtype: DType,
        quantization: ModelConfig.QuantizationConfig?, device: Device
    ) throws -> Qwen2VLVisionModel {
        let cfg = try Qwen2VLVisionConfig.decode(visionConfig)

        // Patch-embed: the checkpoint stores a Conv3d weight
        // `[hidden, in_ch, tPatch, patch, patch]` (PyTorch) or
        // `[hidden, tPatch, patch, patch, in_ch]` (MLX channel-last).
        // Flatten to a 2D GEMM weight `[hidden, patchDim]`.
        let rawPatch = try weights.tensor(named: "patch_embed.proj.weight")
        let patchDim =
            cfg.inChannels * cfg.temporalPatchSize
            * cfg.patchSize * cfg.patchSize
        let kTile = gemmKTileWidth
        let patchDimPadded = ((patchDim + kTile - 1) / kTile) * kTile
        let patchEmbedWeight = flattenPatchEmbed(
            rawPatch, hidden: cfg.hidden, patchDim: patchDim,
            patchDimPadded: patchDimPadded, device: device)

        // Block stack — GELU-MLP intermediate padded to the K-tile width.
        let paddedIntermediate =
            ((cfg.intermediate + kTile - 1) / kTile) * kTile
        var blocks: [Qwen2VLVisionBlock] = []
        blocks.reserveCapacity(cfg.depth)
        for i in 0 ..< cfg.depth {
            let p = "blocks.\(i)"
            // Dequantize at load: the 4-/8-bit Qwen2-VL conversions quantize
            // the ViT linears too, but the block holds plain `Linear` fields
            // driving `Ops.gemm` on `.weight`, so a packed u32 weight must be
            // materialized to a float weight here.
            func lin(_ name: String) throws -> Linear {
                try loadDequantizedLinear(
                    base: "\(p).\(name)", in: weights,
                    quantization: quantization, dtype: dtype, device: device)
            }
            // Qwen2-VL blocks use LayerNorm (weight + bias).
            func norm(_ name: String) throws -> LayerNorm {
                LayerNorm(
                    weight: try weights.tensor(named: "\(p).\(name).weight"),
                    bias: try weights.tensor(named: "\(p).\(name).bias"),
                    eps: cfg.layerNormEps)
            }
            // MLP uses `mlp.fc1` / `mlp.fc2` (GELU, not SwiGLU).
            let fc1 = try lin("mlp.fc1")
            let fc2 = try lin("mlp.fc2")
            blocks.append(
                Qwen2VLVisionBlock(
                    norm1: try norm("norm1"), norm2: try norm("norm2"),
                    qkv: try lin("attn.qkv"), proj: try lin("attn.proj"),
                    fc1: padLinearRows(
                        fc1, toRows: paddedIntermediate, device: device),
                    fc2: padLinearCols(
                        fc2, toCols: paddedIntermediate, device: device),
                    paddedIntermediate: paddedIntermediate, cfg: cfg))
        }

        // Patch-merger: LayerNorm under `merger.ln_q` (with bias), then
        // a 2-layer MLP indexed as `merger.mlp.0` / `merger.mlp.2`.
        let mergerNorm = LayerNorm(
            weight: try weights.tensor(named: "merger.ln_q.weight"),
            bias: try weights.tensor(named: "merger.ln_q.bias"),
            eps: cfg.layerNormEps)
        func mergerLin(_ idx: Int) throws -> Linear {
            try loadDequantizedLinear(
                base: "merger.mlp.\(idx)", in: weights,
                quantization: quantization, dtype: dtype, device: device)
        }
        let mergerFC1 = try mergerLin(0)
        let mergerFC2 = try mergerLin(2)

        return Qwen2VLVisionModel(
            cfg: cfg, patchEmbedWeight: patchEmbedWeight,
            patchDimPadded: patchDimPadded, blocks: blocks,
            mergerNorm: mergerNorm, mergerFC1: mergerFC1, mergerFC2: mergerFC2,
            textHidden: textHidden, dtype: dtype, gridSide: defaultGridSide)
    }

    /// Run the full vision forward on a preprocessed image. `image` is a
    /// normalized NCHW tensor `[1, inChannels, H, W]` at the native
    /// `smartResize` resolution (`H`, `W` multiples of `patch · merge`, not
    /// necessarily square); the patch grid `(H/patch, W/patch)` is derived
    /// from the tensor. Returns `[mergedTokenCount, textHidden]`.
    ///
    /// Thin wrapper: delegates to the shared multi-frame path with
    /// `gridT=1` and a single frame replicated `temporalPatchSize` times
    /// in the unfold — identical to the Qwen 2.5-VL reference.
    func encode(image: Tensor, device: Device) -> Tensor {
        let (gridH, gridW) = Self.gridDims(for: image, cfg: cfg)
        return runForward(
            framePixels: [image.toFloatArray()], gridT: 1,
            gridH: gridH, gridW: gridW, device: device)
    }

    /// Derive the patch grid `(gridH, gridW)` from a preprocessed NCHW
    /// pixel tensor `[1, C, H, W]`: `gridH = H / patch`, `gridW = W /
    /// patch`. Both must be whole and divisible by `spatialMergeSize` —
    /// the `smartResize` processor guarantees this (sides are multiples of
    /// `patch · merge`); a non-conforming tensor is a caller bug.
    static func gridDims(
        for image: Tensor, cfg: Qwen2VLVisionConfig
    ) -> (gridH: Int, gridW: Int) {
        precondition(
            image.shape.count == 4 && image.shape[0] == 1
                && image.shape[1] == cfg.inChannels,
            "Qwen2VL.encode: expected [1,\(cfg.inChannels),H,W], got \(image.shape)")
        let h = image.shape[2]
        let w = image.shape[3]
        let p = cfg.patchSize
        let m = cfg.spatialMergeSize
        precondition(
            h % (p * m) == 0 && w % (p * m) == 0,
            "Qwen2VL.encode: H=\(h) W=\(w) must be multiples of "
                + "patch·merge=\(p * m) (use MediaUtils.smartResize)")
        return (h / p, w / p)
    }

    /// Run the full vision forward on `frames` preprocessed video frames
    /// (each a normalized NCHW tensor `[1, inChannels, side, side]`).
    /// Frames are padded to a multiple of `temporal_patch_size` by
    /// repeating the last frame. Returns
    /// `[mergedTokenCount(frameCount:), textHidden]`.
    ///
    /// Precondition: every frame shares one native NCHW shape
    /// `[1, inChannels, H, W]` (the caller resizes all frames to the same
    /// `smartResize` target); the grid `(H/patch, W/patch)` is derived from
    /// frame 0.
    func encode(frames: [Tensor], device: Device) -> Tensor {
        precondition(
            !frames.isEmpty,
            "Qwen2VL.encode(frames:): expected at least one frame")
        // All frames share one native grid (the caller resizes every frame
        // to the same `smartResize` target); derive it from frame 0.
        let (gridH, gridW) = Self.gridDims(for: frames[0], cfg: cfg)
        for (i, frame) in frames.enumerated() {
            precondition(
                frame.shape == frames[0].shape,
                "Qwen2VL.encode(frames:): frame[\(i)] shape \(frame.shape) "
                    + "!= frame[0] \(frames[0].shape) (all frames must share "
                    + "one resolution)")
        }

        // Pad frame count up to a multiple of `temporal_patch_size` by
        // repeating the last frame — same as mlx-vlm's `patchify`.
        let tp = cfg.temporalPatchSize
        let mod = frames.count % tp
        var framePixels: [[Float]] = frames.map { $0.toFloatArray() }
        if mod != 0 {
            let pad = tp - mod
            if let last = framePixels.last {
                for _ in 0 ..< pad { framePixels.append(last) }
            }
        }
        let gridT = framePixels.count / tp
        return runForward(
            framePixels: framePixels, gridT: gridT,
            gridH: gridH, gridW: gridW, device: device)
    }

    /// Shared forward path for image / video. `framePixels` has either
    /// 1 element (image, replicated `tp` times in the unfold) or
    /// `gridT * tp` elements (video, one per real frame in display order).
    /// `gridT` is the number of temporal patches the unfold will produce;
    /// `(gridH, gridW)` is the per-frame patch grid (`H/patch × W/patch`),
    /// which may be non-square under `smartResize` native resolution.
    private func runForward(
        framePixels: [[Float]], gridT: Int,
        gridH: Int, gridW: Int,
        device: Device
    ) -> Tensor {
        let nPatches = gridT * gridH * gridW

        // ── Patch unfold + embed ── (unfold → GEMM on one cmd buffer)
        let cmd = device.makeCommandBuffer()
        let unfolded = unfoldPatches(
            framePixels: framePixels, gridT: gridT,
            gridH: gridH, gridW: gridW, on: cmd)
        let h = Ops.gemm(
            weight: patchEmbedWeight, input: unfolded,
            nRows: nPatches, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── M-RoPE tables ──
        let (cosTable, sinTable) = ropeTables(
            gridT: gridT, gridH: gridH, gridW: gridW)

        // ── Block stack (full attention on every block) ──
        var hidden = h
        for block in blocks {
            hidden = block.forward(
                hidden, nTokens: nPatches,
                cosTable: cosTable, sinTable: sinTable,
                device: device)
        }

        // ── Patch merger ──
        return mergePatches(hidden, nPatches: nPatches, device: device)
    }

    /// Present the tower as a `VisionEncoder` so `VisionModel` accepts it.
    func asVisionEncoder() -> VisionEncoder {
        Qwen2VLComposedEncoder(tower: self)
    }

    // ── Patch unfold ──

    /// Unfold normalized NCHW frames into a `[nPatches, patchDimPadded]`
    /// tensor. `framePixels` is a list of length 1 (image, replicated
    /// `tp` times along the temporal axis) or `gridT · tp` (video, one
    /// entry per real frame in display order). Each row holds one
    /// `C · tPatch · patch · patch` patch in merge-block raster order;
    /// the temporal dimension follows the layout expected by the flattened
    /// patch-embed weight.
    ///
    /// Row order is `(tGroup, mergeBlockRow, mergeBlockCol, intraRow,
    /// intraCol)` — temporal patches are blocked together so the merger's
    /// reshape lines up with one temporal patch group of spatial
    /// neighbourhoods.
    private func unfoldPatches(
        framePixels: [[Float]], gridT: Int, gridH: Int, gridW: Int,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        let p = cfg.patchSize
        let c = cfg.inChannels
        let tp = cfg.temporalPatchSize
        let m = cfg.spatialMergeSize
        let mergeBlocksH = gridH / m
        let mergeBlocksW = gridW / m
        let imgH = gridH * p
        let imgW = gridW * p
        // Image fast path: a single frame reused for every temporal slot.
        let isImage = framePixels.count == 1
        // Concatenate the per-frame pixel buffers into one
        // `[nFrames, c, imgH, imgW]` tensor; the GPU kernel emits the
        // merge-block raster + `(t, c, py, px)` columns, zero-padded.
        let flat = framePixels.flatMap { $0 }
        let framesT = MediaUtils.makeTensor(
            from: flat, shape: [framePixels.count, c, imgH, imgW],
            dtype: dtype, device: .shared)
        return Ops.patchUnfoldQwen(
            frames: framesT, channels: c, patch: p, temporalPatch: tp,
            merge: m, mergeBlocksH: mergeBlocksH, mergeBlocksW: mergeBlocksW,
            imgH: imgH, imgW: imgW, gridT: gridT,
            patchDimPadded: patchDimPadded, isImage: isImage, on: cmd)
    }

    // ── M-RoPE ──

    /// Build the per-token rotary `cos` / `sin` tables `[nPatches,
    /// headDim]`. Qwen's vision M-RoPE interleaves a height-rotary and a
    /// width-rotary half over `headDim`, each driven by the patch's
    /// `(h, w)` grid coordinate. Positions are in merge-block raster order.
    ///
    /// For video (`gridT > 1`), the `(h, w)` pattern is tiled identically
    /// per temporal patch — the temporal axis is carried by the text
    /// decoder's M-RoPE 3-section split, not by the vision tower's rotary
    /// table. This matches the Qwen 2.5-VL reference implementation.
    private func ropeTables(
        gridT: Int, gridH: Int, gridW: Int
    ) -> (cos: [Float], sin: [Float]) {
        let nPatches = gridT * gridH * gridW
        let headDim = cfg.headDim
        let half = headDim / 2  // height half | width half
        let quarter = half / 2  // distinct rotary frequencies
        // Inverse frequencies with theta 10000 (VisionRotaryEmbedding).
        var invFreq = [Float](repeating: 0, count: quarter)
        for i in 0 ..< quarter {
            invFreq[i] = 1.0 / pow(10_000, Float(2 * i) / Float(half))
        }
        let (hPos, wPos) = mergeReorderedPositions(gridH: gridH, gridW: gridW)
        let perFrame = gridH * gridW

        var cosT = [Float](repeating: 0, count: nPatches * headDim)
        var sinT = [Float](repeating: 0, count: nPatches * headDim)
        for t in 0 ..< nPatches {
            // Spatial position is the same per-frame tile for all temporal
            // groups — only (h, w) coordinates drive the vision RoPE.
            let spatial = t % perFrame
            let base = t * headDim
            // Height rotary fills [0, half); width rotary fills [half, 2·half).
            for i in 0 ..< quarter {
                let fh = Float(hPos[spatial]) * invFreq[i]
                let fw = Float(wPos[spatial]) * invFreq[i]
                // Frequencies tiled to half (duplicate for rotate-half).
                for (off, f) in [(i, fh), (i + quarter, fh)] {
                    cosT[base + off] = cos(f)
                    sinT[base + off] = sin(f)
                }
                for (off, f) in [(half + i, fw), (half + i + quarter, fw)] {
                    cosT[base + off] = cos(f)
                    sinT[base + off] = sin(f)
                }
            }
        }
        return (cosT, sinT)
    }

    /// Per-patch `(h, w)` grid coordinates in merge-block raster order
    /// (matches `unfoldPatches` row order within one temporal group).
    private func mergeReorderedPositions(
        gridH: Int, gridW: Int
    ) -> (h: [Int], w: [Int]) {
        let m = cfg.spatialMergeSize
        let mergeBlocksH = gridH / m
        let mergeBlocksW = gridW / m
        var hPos = [Int](repeating: 0, count: gridH * gridW)
        var wPos = [Int](repeating: 0, count: gridH * gridW)
        var idx = 0
        for br in 0 ..< mergeBlocksH {
            for bc in 0 ..< mergeBlocksW {
                for ir in 0 ..< m {
                    for ic in 0 ..< m {
                        hPos[idx] = br * m + ir
                        wPos[idx] = bc * m + ic
                        idx += 1
                    }
                }
            }
        }
        return (hPos, wPos)
    }

    // ── Patch merger ──

    /// Pool each `mergeSize²` neighbourhood of post-encoder tokens into
    /// one token, then project into the text hidden dim. Tokens are in
    /// `(tGroup, merge-block raster)` order, so each consecutive run of
    /// `mergeUnit` tokens is one neighbourhood. Returns
    /// `[nPatches/mergeUnit, textHidden]`.
    private func mergePatches(_ h: Tensor, nPatches: Int, device: Device) -> Tensor {
        let hidden = cfg.hidden
        let mergeUnit = cfg.mergeUnit
        let merged = nPatches / mergeUnit

        // LayerNorm each token, then group `mergeUnit` tokens into one row.
        let cmd = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            h, weight: mergerNorm.weight,
            bias: mergerNorm.bias, eps: mergerNorm.eps,
            nRows: nPatches, rowSize: hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // The merger MLP input dim is `mergeUnit · hidden`; tokens are
        // already contiguous per neighbourhood, so a plain reshape suffices.
        let grouped = normed.reshaped(to: [merged, mergeUnit * hidden])

        let cmd2 = device.makeCommandBuffer()
        var x = Ops.gemm(
            weight: mergerFC1.weight, input: grouped,
            nRows: merged, on: cmd2)
        if let b = mergerFC1.bias {
            x = addRowBias(
                x, bias: b, nRows: merged,
                rowSize: mergerFC1.weight.shape[0], on: cmd2)
        }
        x = Ops.gelu(x, on: cmd2)
        var y = Ops.gemm(
            weight: mergerFC2.weight, input: x,
            nRows: merged, on: cmd2)
        if let b = mergerFC2.bias {
            y = addRowBias(
                y, bias: b, nRows: merged,
                rowSize: mergerFC2.weight.shape[0], on: cmd2)
        }
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return y
    }
}

/// A `VisionEncoder` subclass whose `encode` runs the Qwen 2-VL vision
/// tower — so `VisionModel` (which holds a `VisionEncoder`) transparently
/// gets the merged, projected vision tokens.
final class Qwen2VLComposedEncoder: VisionEncoder {
    let tower: Qwen2VLVisionModel

    init(tower: Qwen2VLVisionModel) {
        self.tower = tower
        let c = tower.cfg
        let side = tower.gridSide * c.patchSize
        let facadeConfig = VisionEncoderConfig(
            inChannels: c.inChannels, imageSize: side,
            patchSize: side / Int(Double(tower.mergedTokenCount).squareRoot()),
            hidden: c.hidden, intermediate: c.intermediate,
            nLayers: c.depth, nHeads: c.numHeads,
            layerNormEps: c.layerNormEps, textHidden: tower.textHidden)
        // The patch-embed / position / layers are unused by the override
        // below; pass minimal placeholders.
        let placeholderW = tower.patchEmbedWeight
        super.init(
            config: facadeConfig,
            patchEmbedWeight: placeholderW, patchEmbedBias: placeholderW,
            positionEmbedding: placeholderW, layers: [],
            postLayerNorm: tower.mergerNorm,
            projection: nil, dtype: tower.dtype)
    }

    /// Native-resolution preprocessing: `smartResize` the raw image to the
    /// largest aspect-preserving grid under Qwen2-VL's token budget (sides a
    /// multiple of `patch · merge`), then GPU bilinear-resize + CLIP-
    /// normalize → NCHW `[1, 3, H, W]`. Replaces the old fixed 392² square
    /// that distorted + truncated every non-square image. Falls back to the
    /// generic fixed square (returns `nil`) only for a degenerate aspect
    /// ratio `smartResize` rejects (> 200:1) — those never occur in
    /// practice.
    override func preprocessImage(
        _ image: RGBImage, device: Device = .shared
    ) -> Tensor? {
        guard
            let (th, tw) = try? MediaUtils.smartResize(
                height: image.height, width: image.width,
                bounds: Qwen2VLVisionModel.smartResizeBounds(cfg: tower.cfg))
        else {
            return nil
        }
        // Bicubic: smart_resize routinely down-scales multi-MP photos to a
        // few hundred px/side; bicubic keeps text + fine detail bilinear softens.
        return MediaUtils.resizeNormalize(
            image, targetW: tw, targetH: th,
            normalization: .clip, dtype: tower.dtype,
            mode: .bicubic, device: device)
    }

    /// Run the Qwen 2-VL vision tower on a single preprocessed image.
    /// Returns `[mergedTokenCount, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        tower.encode(image: image, device: device)
    }

    /// Run the Qwen 2-VL vision tower on a sequence of preprocessed
    /// video frames. Returns
    /// `[(T/temporalPatchSize) · mergedTokenCount, textHidden]`.
    override func encode(frames: [Tensor], device: Device = .shared) throws -> Tensor {
        tower.encode(frames: frames, device: device)
    }
}

// ════════════════════════════════════════════════════════════════════
// Qwen 2.5-VL — consolidated from the former Qwen25 family. Qwen 2.5's
// text backbone is Qwen 2 (`Qwen2Dense`); only the vision tower +
// orchestrator differ (windowed attention, learned pos table).
// ════════════════════════════════════════════════════════════════════

public enum Qwen25VL {
    /// `image_token_id` default for Qwen 2.5-VL checkpoints
    /// (`<|image_pad|>`).
    public static let defaultImageTokenId = 151_655
    /// `video_token_id` default for Qwen 2.5-VL checkpoints
    /// (`<|video_pad|>`).
    public static let defaultVideoTokenId = 151_656

    /// Capabilities a Qwen 2.5-VL checkpoint declares to the loader.
    /// Text + image + video — every published Qwen 2.5-VL conversion
    /// ships the same vision tower that handles both image (one frame
    /// repeated `temporal_patch_size` times) and video (N consecutive
    /// frames per temporal patch). The actual video-token splice is
    /// gated on the caller passing `videoFrames` to
    /// `VisionModel.generate(...)`.
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.imageIn, .videoIn])

    /// Generation defaults — the Qwen 2.5 text variant's (`Qwen2Dense`).
    public static let defaultGenerationParameters =
        Qwen2Dense.defaultGenerationParameters

    /// Build a `VisionModel` from a `Qwen2_5_VLForConditionalGeneration`
    /// checkpoint: the dynamic-resolution vision tower + the Qwen 2.x
    /// text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Qwen2Error.missingConfig
        }

        // ── Text backbone ──
        // The mlx-community Qwen 2.5-VL conversion namespaces its text
        // weights under `language_model.model.*` / `language_model.norm`;
        // the text hyper-parameters live at the config top level. Route
        // the `language_model.`-prefixed sub-tree through the Llama dense
        // engine (Qwen 2.x == LlamaDense).
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try Qwen2.variant(for: config).loadModel(
            config: config, weights: textWeights, options: options, device: device)

        // ── Vision tower ──
        // Vision weights are under `vision_tower.*`. The common conversions
        // set `skip_vision`, so the ViT is plain f16 / bf16 — but pass the
        // quantization config anyway; `loadDequantizedLinear` dequantizes
        // per-tensor only if a checkpoint did quantize the tower.
        let visionWeights = weights.prefixed("vision_tower.")
        let vision = try Qwen25VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, dtype: textEngine.dtype,
            quantization: config.quantization, device: device)

        let imageTokenId = config.int("image_token_id") ?? defaultImageTokenId
        let videoTokenId = config.int("video_token_id") ?? defaultVideoTokenId
        // The vision tower decides its own merged-token count from the
        // dynamic image geometry; the encoder facade reports it.
        return try VisionModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .clip,
            imageTokenCount: vision.mergedTokenCount)
    }
}


// ─── Vision tower configuration ──────────────────────────────────────

/// Static shape of the Qwen 2.5-VL vision tower, decoded from
/// `vision_config`.
struct Qwen25VLVisionConfig {
    let depth: Int
    let hidden: Int
    let intermediate: Int
    let outHidden: Int
    let numHeads: Int
    let patchSize: Int
    let spatialMergeSize: Int
    let temporalPatchSize: Int
    let windowSize: Int
    let fullattBlockIndexes: Set<Int>
    let inChannels: Int
    let rmsNormEps: Float

    /// Per-head dimension.
    var headDim: Int { hidden / numHeads }
    /// `mergeSize²` — patches pooled into one merged token.
    var mergeUnit: Int { spatialMergeSize * spatialMergeSize }

    static func decode(_ c: ModelConfig) throws -> Qwen25VLVisionConfig {
        guard let depth = c.int("depth"),
            let hidden = c.int("hidden_size"),
            let numHeads = c.int("num_heads"),
            let patchSize = c.int("patch_size"),
            let mergeSize = c.int("spatial_merge_size")
        else {
            throw Qwen2Error.missingConfig
        }
        // `intermediate_size` / `out_hidden_size` are present on real
        // checkpoints; fall back to documented Qwen2.5-VL defaults.
        let intermediate = c.int("intermediate_size") ?? hidden * 4
        let outHidden = c.int("out_hidden_size") ?? hidden
        return Qwen25VLVisionConfig(
            depth: depth, hidden: hidden, intermediate: intermediate,
            outHidden: outHidden, numHeads: numHeads, patchSize: patchSize,
            spatialMergeSize: mergeSize,
            temporalPatchSize: c.int("temporal_patch_size") ?? 2,
            windowSize: c.int("window_size") ?? 112,
            fullattBlockIndexes: Set(c.intArray("fullatt_block_indexes") ?? []),
            inChannels: c.int("in_chans") ?? c.int("in_channels") ?? 3,
            rmsNormEps: Float(c.float("layer_norm_eps") ?? c.float("rms_norm_eps") ?? 1e-6))
    }
}

// ─── Vision block ────────────────────────────────────────────────────

/// One Qwen 2.5-VL vision block: RMSNorm → windowed/full MHA + M-RoPE →
/// residual, RMSNorm → SwiGLU MLP → residual. Held as plain weight
/// tensors; the forward runs CPU attention + GPU GEMMs.
final class Qwen25VLVisionBlock {
    let norm1: RMSNorm
    let norm2: RMSNorm
    let qkv: Linear  // fused [3·hidden, hidden] (+ bias)
    let proj: Linear  // [hidden, hidden] (+ bias)
    let gate: Linear  // SwiGLU gate [paddedIntermediate, hidden]
    let up: Linear  // SwiGLU up   [paddedIntermediate, hidden]
    let down: Linear  // SwiGLU down [hidden, paddedIntermediate]
    let cfg: Qwen25VLVisionConfig
    /// SwiGLU intermediate dim rounded up to the GEMM K-tile width — the
    /// `gate`/`up` outputs are zero-extended and `down`'s input columns
    /// zero-padded to it, so the `down` projection's `inDim` is aligned.
    let paddedIntermediate: Int

    init(
        norm1: RMSNorm, norm2: RMSNorm, qkv: Linear, proj: Linear,
        gate: Linear, up: Linear, down: Linear,
        paddedIntermediate: Int, cfg: Qwen25VLVisionConfig
    ) {
        self.norm1 = norm1
        self.norm2 = norm2
        self.qkv = qkv
        self.proj = proj
        self.gate = gate
        self.up = up
        self.down = down
        self.paddedIntermediate = paddedIntermediate
        self.cfg = cfg
    }

    /// Forward `[nTokens, hidden]` activations through one block.
    /// `cosTable` / `sinTable` are the precomputed per-token rotary
    /// tables `[nTokens, headDim]`; `windowGroups` partitions the tokens
    /// into attention windows (one group = full attention over the whole
    /// sequence for full-attention blocks).
    func forward(
        _ h: Tensor, nTokens: Int,
        cosTable: [Float], sinTable: [Float],
        windowGroups: [[Int]], device: Device
    ) -> Tensor {
        let hidden = cfg.hidden
        // ── Attention sub-block ──
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(
            h, weight: norm1.weight, eps: norm1.eps,
            nRows: nTokens, rowSize: hidden, on: cmd)
        let qkvOut = projectRows(
            qkv, normed, nTokens: nTokens,
            outDim: 3 * hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let attn = cpuAttention(
            qkv: qkvOut, nTokens: nTokens,
            cosTable: cosTable, sinTable: sinTable,
            windowGroups: windowGroups, device: device)

        // ── Residual + MLP sub-block ──
        let cmd2 = device.makeCommandBuffer()
        let attnProj = projectRows(
            proj, attn, nTokens: nTokens,
            outDim: hidden, on: cmd2)
        let postAttn = Ops.add(h, attnProj, on: cmd2)
        let normed2 = Ops.rmsNormRows(
            postAttn, weight: norm2.weight,
            eps: norm2.eps, nRows: nTokens,
            rowSize: hidden, on: cmd2)
        let g = projectRows(
            gate, normed2, nTokens: nTokens,
            outDim: paddedIntermediate, on: cmd2)
        let u = projectRows(
            up, normed2, nTokens: nTokens,
            outDim: paddedIntermediate, on: cmd2)
        let act = Ops.silu(g, on: cmd2)
        let gated = Ops.mul(act, u, on: cmd2)
        let d = projectRows(
            down, gated, nTokens: nTokens,
            outDim: hidden, on: cmd2)
        let result = Ops.add(postAttn, d, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return result
    }

    /// CPU windowed multi-head attention with M-RoPE. `qkv` is the fused
    /// `[nTokens, 3·hidden]` projection. Returns the context token-major
    /// `[nTokens, hidden]`.
    ///
    /// Two-stage parallelization with `DispatchQueue.concurrentPerform`:
    ///   1. Setup — M-RoPE: each `(head, token)` pair writes its own
    ///      `qH[head*nTokens+t]` / `kH[head*nTokens+t]` / `vH[head*nTokens+t]`
    ///      row — disjoint, race-free.
    ///   2. Attention — parallelized over heads: within each head, window
    ///      groups are processed serially (each (head, i) writes to a
    ///      disjoint `[oBase, oBase + headDim)` output slice — race-free
    ///      across heads because different heads use different `hOff`).
    private func cpuAttention(
        qkv: Tensor, nTokens: Int,
        cosTable: [Float], sinTable: [Float],
        windowGroups: [[Int]], device: Device
    ) -> Tensor {
        let nHeads = cfg.numHeads
        let headDim = cfg.headDim
        let hidden = cfg.hidden
        let qkvA = qkv.toFloatArray()
        var out = [Float](repeating: 0, count: nTokens * hidden)
        let scale = 1.0 / Float(Double(headDim).squareRoot())
        let half = headDim / 2

        // Stage 1: Extract and RoPE every (head, token) slice.
        // Index layout: qH[head * nTokens + t] — each (head, t) pair owns
        // its slot, so concurrent writes are race-free.
        // `nonisolated(unsafe)`: disjoint indices, no aliasing.
        nonisolated(unsafe) var qH = [[Float]](repeating: [], count: nHeads * nTokens)
        nonisolated(unsafe) var kH = [[Float]](repeating: [], count: nHeads * nTokens)
        nonisolated(unsafe) var vH = [[Float]](repeating: [], count: nHeads * nTokens)
        DispatchQueue.concurrentPerform(iterations: nHeads * nTokens) { work in
            let head = work / nTokens
            let t = work % nTokens
            let hOff = head * headDim
            let base = t * 3 * hidden
            var q = Array(qkvA[(base + hOff) ..< (base + hOff + headDim)])
            var k = Array(qkvA[(base + hidden + hOff) ..< (base + hidden + hOff + headDim)])
            let v = Array(qkvA[(base + 2 * hidden + hOff) ..< (base + 2 * hidden + hOff + headDim)])
            // Apply M-RoPE inline. Qwen's rotate-half scheme:
            // out[d] = x[d]·cos − x[d±half]·sin. Defined inside the
            // closure to avoid @Sendable capture warnings.
            func applyRope(_ x: inout [Float], tok: Int) {
                let cb = tok * headDim
                for d in 0 ..< headDim {
                    let rotated = d < half ? -x[d + half] : x[d - half]
                    let c = cosTable[cb + d]
                    let s = sinTable[cb + d]
                    x[d] = x[d] * c + rotated * s
                }
            }
            applyRope(&q, tok: t)
            applyRope(&k, tok: t)
            qH[head * nTokens + t] = q
            kH[head * nTokens + t] = k
            vH[head * nTokens + t] = v
        }

        // Stage 2: Attention — GPU.
        //   • Full-attention blocks (`fullattBlockIndexes`, one group over
        //     every token) → dense `Ops.sdpaBidirectional`. d80 dense is
        //     validated at production image resolution by Qwen2-VL (same
        //     kernel + dims, e2e coherent caption), so it's no longer gated.
        //   • Windowed blocks (the bulk: 28/32) → the windowed kernel below.
        // The CPU loop remains only as a fallback for head dims neither
        // kernel covers.
        let isFullAttention =
            windowGroups.count == 1
            && windowGroups[0].count == nTokens
        if isFullAttention
            && OpsValidation.sdpaBidirectionalIsProductionSafe(headDim: headDim)
        {
            // GPU path. qH/kH/vH are already laid out as
            // [nHeads, nTokens, headDim] — kernel K/V layout natively.
            // Q needs a transpose into [nTokens, nHeads, headDim].
            var qFlat = [Float](repeating: 0, count: nTokens * nHeads * headDim)
            var kFlat = [Float](repeating: 0, count: nHeads * nTokens * headDim)
            var vFlat = [Float](repeating: 0, count: nHeads * nTokens * headDim)
            for head in 0 ..< nHeads {
                for t in 0 ..< nTokens {
                    let qRow = qH[head * nTokens + t]
                    let kRow = kH[head * nTokens + t]
                    let vRow = vH[head * nTokens + t]
                    let qDst = (t * nHeads + head) * headDim
                    let kvDst = (head * nTokens + t) * headDim
                    for d in 0 ..< headDim {
                        qFlat[qDst + d] = qRow[d]
                        kFlat[kvDst + d] = kRow[d]
                        vFlat[kvDst + d] = vRow[d]
                    }
                }
            }
            let qT = Tensor.empty(
                shape: [nTokens, nHeads, headDim], dtype: .f32,
                device: device)
            MediaUtils.copyFloats(qFlat, into: qT)
            let kT = Tensor.empty(
                shape: [nHeads, nTokens, headDim], dtype: .f32,
                device: device)
            MediaUtils.copyFloats(kFlat, into: kT)
            let vT = Tensor.empty(
                shape: [nHeads, nTokens, headDim], dtype: .f32,
                device: device)
            MediaUtils.copyFloats(vFlat, into: vT)
            let cmd = device.makeCommandBuffer()
            let outT = Ops.sdpaBidirectional(
                q: qT, k: kT, v: vT,
                nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
                baseKV: 0, nQuery: nTokens, kvStride: nTokens,
                scale: scale, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            // outT is [nTokens, nHeads, headDim] — byte-identical to the
            // [nTokens, hidden] layout the patch-merger/proj expects.
            let outFlat = outT.toFloatArray()
            let result = Tensor.empty(
                shape: [nTokens, hidden], dtype: qkv.dtype,
                device: device)
            MediaUtils.copyFloats(outFlat, into: result)
            return result
        }

        // GPU windowed path — the windowed-attention blocks (the bulk:
        // 28/32 for Qwen 2.5-VL). Permute tokens into window-contiguous
        // order so each window is one `[seg_start, seg_len)` slice, run the
        // windowed SDPA kernel, then scatter the context back to original
        // order. d80 only (the kernel's head dim); other dims → CPU.
        if headDim == 80 {
            // Window permutation: new-order index → old token index; each
            // window group occupies a contiguous run and every token in it
            // carries that run's (offset, size).
            var perm = [Int]()
            perm.reserveCapacity(nTokens)
            var segStart = [UInt32](repeating: 0, count: nTokens)
            var segLen = [UInt32](repeating: 0, count: nTokens)
            for group in windowGroups {
                let off = perm.count
                let sz = group.count
                for oldT in group {
                    segStart[perm.count] = UInt32(off)
                    segLen[perm.count] = UInt32(sz)
                    perm.append(oldT)
                }
            }
            // Permuted q (token-major) + k/v (head-major) from the
            // already-RoPE'd qH/kH/vH.
            var qPerm = [Float](repeating: 0, count: nTokens * nHeads * headDim)
            var kPerm = [Float](repeating: 0, count: nHeads * nTokens * headDim)
            var vPerm = [Float](repeating: 0, count: nHeads * nTokens * headDim)
            for newIdx in 0 ..< nTokens {
                let oldT = perm[newIdx]
                for head in 0 ..< nHeads {
                    let qRow = qH[head * nTokens + oldT]
                    let kRow = kH[head * nTokens + oldT]
                    let vRow = vH[head * nTokens + oldT]
                    let qDst = (newIdx * nHeads + head) * headDim
                    let kvDst = (head * nTokens + newIdx) * headDim
                    for d in 0 ..< headDim {
                        qPerm[qDst + d] = qRow[d]
                        kPerm[kvDst + d] = kRow[d]
                        vPerm[kvDst + d] = vRow[d]
                    }
                }
            }
            let qT = Tensor.empty(shape: [nTokens, nHeads, headDim], dtype: .f32, device: device)
            MediaUtils.copyFloats(qPerm, into: qT)
            let kT = Tensor.empty(shape: [nHeads, nTokens, headDim], dtype: .f32, device: device)
            MediaUtils.copyFloats(kPerm, into: kT)
            let vT = Tensor.empty(shape: [nHeads, nTokens, headDim], dtype: .f32, device: device)
            MediaUtils.copyFloats(vPerm, into: vT)
            let ssT = Tensor.empty(shape: [nTokens], dtype: .u32, device: device)
            ssT.copyIn(from: segStart)
            let slT = Tensor.empty(shape: [nTokens], dtype: .u32, device: device)
            slT.copyIn(from: segLen)
            let cmd = device.makeCommandBuffer()
            let outT = Ops.sdpaBidirectionalWindowed(
                q: qT, k: kT, v: vT, segStart: ssT, segLen: slT,
                nQHeads: nHeads, nKVHeads: nHeads,
                nQuery: nTokens, kvStride: nTokens, scale: scale, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            let outFlat = outT.toFloatArray()
            // Scatter the per-window context back to original token order,
            // token-major `[nTokens, hidden]`.
            var ctx = [Float](repeating: 0, count: nTokens * hidden)
            for newIdx in 0 ..< nTokens {
                let oldT = perm[newIdx]
                for head in 0 ..< nHeads {
                    let src = (newIdx * nHeads + head) * headDim
                    let dst = oldT * hidden + head * headDim
                    for d in 0 ..< headDim { ctx[dst + d] = outFlat[src + d] }
                }
            }
            let result = Tensor.empty(
                shape: [nTokens, hidden], dtype: qkv.dtype, device: device)
            MediaUtils.copyFloats(ctx, into: result)
            return result
        }

        // CPU fallback path — windowed attention. Same loop as before,
        // parallelized over heads. Within a head, token `i` writes to
        // `out[i*hidden+hOff .. +headDim)` — disjoint across heads
        // (different `hOff`) and disjoint across tokens within a head
        // (different `i`).
        out.withUnsafeMutableBufferPointer { outBuf in
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nHeads) { head in
                let hOff = head * headDim
                for group in windowGroups {
                    for i in group {
                        var scores = [Float](repeating: 0, count: group.count)
                        var maxS = -Float.greatestFiniteMagnitude
                        let qVec = qH[head * nTokens + i]
                        for (gi, j) in group.enumerated() {
                            var dot: Float = 0
                            let kVec = kH[head * nTokens + j]
                            for d in 0 ..< headDim { dot += qVec[d] * kVec[d] }
                            let s = dot * scale
                            scores[gi] = s
                            if s > maxS { maxS = s }
                        }
                        var sum: Float = 0
                        for gi in 0 ..< group.count {
                            let e = exp(scores[gi] - maxS)
                            scores[gi] = e
                            sum += e
                        }
                        let inv = sum > 0 ? 1 / sum : 0
                        let oBase = i * hidden + hOff
                        for (gi, j) in group.enumerated() {
                            let w = scores[gi] * inv
                            let vVec = vH[head * nTokens + j]
                            for d in 0 ..< headDim { outPtr[oBase + d] += w * vVec[d] }
                        }
                    }
                }
            }
        }
        let result = Tensor.empty(
            shape: [nTokens, hidden], dtype: qkv.dtype,
            device: device)
        MediaUtils.copyFloats(out, into: result)
        return result
    }

    /// Apply a `Linear` to every row of `[nTokens, *]` via `Ops.gemm`,
    /// then broadcast-add the bias if present.
    private func projectRows(
        _ linear: Linear, _ x: Tensor, nTokens: Int,
        outDim: Int, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let y = Ops.gemm(weight: linear.weight, input: x, nRows: nTokens, on: cmd)
        guard let bias = linear.bias else { return y }
        return addRowBias(
            y, bias: bias, nRows: nTokens,
            rowSize: outDim, on: cmd)
    }
}

// ─── Vision tower ────────────────────────────────────────────────────

/// The Qwen 2.5-VL vision tower. Holds the patch-embed projection, the
/// block stack, and the patch-merger; `encode(image:)` / `encode(frames:)`
/// run the full forward and return merged tokens in the text hidden dim.
final class Qwen25VLVisionModel: @unchecked Sendable {
    let cfg: Qwen25VLVisionConfig
    /// Flattened patch-embed weight `[hidden, patchDimPadded]` — the
    /// unfold `patchDim` rounded up to the GEMM K-tile width (16) with
    /// zero-pad columns, so the patch-embed projection is a single
    /// `Ops.gemm`.
    let patchEmbedWeight: Tensor
    /// `patchDim` padded up to a multiple of the GEMM K-tile width.
    let patchDimPadded: Int
    let blocks: [Qwen25VLVisionBlock]
    /// Patch-merger RMSNorm over the pre-merge hidden.
    let mergerNorm: RMSNorm
    /// Patch-merger MLP: `[mergeUnit·hidden] → [mergeUnit·hidden] → [outHidden]`.
    let mergerFC1: Linear
    let mergerFC2: Linear
    let textHidden: Int
    let dtype: DType
    /// Patch grid side (square test image at the encoder's fixed size).
    let gridSide: Int

    /// Number of merged vision tokens one image contributes — the
    /// `imageTokenCount` `VisionModel` splices.
    var mergedTokenCount: Int {
        let llmSide = gridSide / cfg.spatialMergeSize
        return llmSide * llmSide
    }

    /// Number of merged vision tokens a video of `frameCount` frames
    /// contributes — `(T/tp) * (gridSide/m)²`. Used by the family
    /// orchestrator + callers to compute placeholder counts.
    func mergedTokenCount(frameCount: Int) -> Int {
        MediaUtils.mergedVideoTokenCount(
            frameCount: frameCount, gridSide: gridSide,
            spatialMergeSize: cfg.spatialMergeSize,
            temporalPatchSize: cfg.temporalPatchSize)
    }

    /// Round `frameCount` up to a multiple of `temporal_patch_size` by
    /// repeating the last frame.
    func paddedFrameCount(_ frameCount: Int) -> Int {
        MediaUtils.paddedFrameCount(
            frameCount, temporalPatchSize: cfg.temporalPatchSize)
    }

    /// Derive the patch grid `(gridH, gridW)` from a preprocessed NCHW
    /// pixel tensor `[1, C, H, W]`: `gridH = H/patch`, `gridW = W/patch`.
    /// Both must be whole and divisible by `spatialMergeSize` — the
    /// `smartResize` processor guarantees this (sides multiples of
    /// `patch · merge`).
    static func gridDims(
        for image: Tensor, cfg: Qwen25VLVisionConfig
    ) -> (gridH: Int, gridW: Int) {
        precondition(
            image.shape.count == 4 && image.shape[0] == 1
                && image.shape[1] == cfg.inChannels,
            "Qwen25VL.encode: expected [1,\(cfg.inChannels),H,W], got \(image.shape)")
        let h = image.shape[2]
        let w = image.shape[3]
        let p = cfg.patchSize
        let m = cfg.spatialMergeSize
        precondition(
            h % (p * m) == 0 && w % (p * m) == 0,
            "Qwen25VL.encode: H=\(h) W=\(w) must be multiples of "
                + "patch·merge=\(p * m) (use MediaUtils.smartResize)")
        return (h / p, w / p)
    }

    /// Standard Qwen-family `smartResize` bounds for this config
    /// (`factor = patch · merge`, HF default token budget).
    static func smartResizeBounds(
        cfg: Qwen25VLVisionConfig
    ) -> MediaUtils.SmartResizeBounds {
        MediaUtils.SmartResizeBounds.qwen(
            patchSize: cfg.patchSize, mergeSize: cfg.spatialMergeSize)
    }

    /// The native `(gridH, gridW)` patch grid `smartResize` selects for an
    /// image of the given pixel dims, plus the merged vision-token count it
    /// yields. Lets a caller size the placeholder run before encoding.
    static func smartResizeGrid(
        width: Int, height: Int, cfg: Qwen25VLVisionConfig
    ) throws -> (gridH: Int, gridW: Int, mergedTokens: Int) {
        try MediaUtils.smartResizeGrid(
            width: width, height: height,
            patchSize: cfg.patchSize, spatialMergeSize: cfg.spatialMergeSize)
    }

    init(
        cfg: Qwen25VLVisionConfig, patchEmbedWeight: Tensor,
        patchDimPadded: Int,
        blocks: [Qwen25VLVisionBlock], mergerNorm: RMSNorm,
        mergerFC1: Linear, mergerFC2: Linear, textHidden: Int,
        dtype: DType, gridSide: Int
    ) {
        self.cfg = cfg
        self.patchEmbedWeight = patchEmbedWeight
        self.patchDimPadded = patchDimPadded
        self.blocks = blocks
        self.mergerNorm = mergerNorm
        self.mergerFC1 = mergerFC1
        self.mergerFC2 = mergerFC2
        self.textHidden = textHidden
        self.dtype = dtype
        self.gridSide = gridSide
    }

    /// The patch grid side for a square image at the standard Qwen2.5-VL
    /// test resolution. We size the test image so the grid divides the
    /// merge size cleanly and the window size; 28×28 patches at
    /// patch-14 → 392×392 px works for the default 112-px window.
    static let defaultGridSide = 28

    static func load(
        visionConfig: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundle, dtype: DType,
        quantization: ModelConfig.QuantizationConfig?, device: Device
    ) throws -> Qwen25VLVisionModel {
        let cfg = try Qwen25VLVisionConfig.decode(visionConfig)

        // Patch-embed: the checkpoint stores a Conv3d weight
        // `[hidden, in_ch, tPatch, patch, patch]`. We flatten it to a
        // 2D GEMM weight `[hidden, in_ch·tPatch·patch·patch]` so each
        // unfolded patch row projects with one matmul.
        let rawPatch = try weights.tensor(named: "patch_embed.proj.weight")
        let patchDim =
            cfg.inChannels * cfg.temporalPatchSize
            * cfg.patchSize * cfg.patchSize
        // Round the unfold dim up to the GEMM K-tile width so the
        // patch-embed projection dispatches as a single `Ops.gemm`.
        let patchDimPadded =
            ((patchDim + gemmKTileWidth - 1) / gemmKTileWidth) * gemmKTileWidth
        let patchEmbedWeight = flattenPatchEmbed(
            rawPatch, hidden: cfg.hidden, patchDim: patchDim,
            patchDimPadded: patchDimPadded, device: device)

        // Block stack. The SwiGLU intermediate is rounded up to the
        // GEMM K-tile width so the `down` projection's `inDim` aligns;
        // `gate`/`up` output rows and `down` input columns are
        // zero-extended to the padded dim.
        let paddedIntermediate =
            ((cfg.intermediate + gemmKTileWidth - 1) / gemmKTileWidth)
            * gemmKTileWidth
        var blocks: [Qwen25VLVisionBlock] = []
        blocks.reserveCapacity(cfg.depth)
        for i in 0 ..< cfg.depth {
            let p = "blocks.\(i)"
            // Dequantize at load: the 4-/8-bit conversions quantize the ViT
            // linears too, but the block holds plain `Linear` fields driving
            // `Ops.gemm` on `.weight`, so a packed u32 weight must be
            // materialized to a float weight here.
            func lin(_ name: String) throws -> Linear {
                try loadDequantizedLinear(
                    base: "\(p).\(name)", in: weights,
                    quantization: quantization, dtype: dtype, device: device)
            }
            func norm(_ name: String) throws -> RMSNorm {
                RMSNorm(
                    weight: try weights.tensor(named: "\(p).\(name).weight"),
                    eps: cfg.rmsNormEps)
            }
            let gate = try lin("mlp.gate_proj")
            let up = try lin("mlp.up_proj")
            let down = try lin("mlp.down_proj")
            blocks.append(
                Qwen25VLVisionBlock(
                    norm1: try norm("norm1"), norm2: try norm("norm2"),
                    qkv: try lin("attn.qkv"), proj: try lin("attn.proj"),
                    gate: padLinearRows(gate, toRows: paddedIntermediate, device: device),
                    up: padLinearRows(up, toRows: paddedIntermediate, device: device),
                    down: padLinearCols(down, toCols: paddedIntermediate, device: device),
                    paddedIntermediate: paddedIntermediate, cfg: cfg))
        }

        // Patch-merger.
        let mergerNorm = RMSNorm(
            weight: try weights.tensor(named: "merger.ln_q.weight"),
            eps: cfg.rmsNormEps)
        func mergerLin(_ idx: Int) throws -> Linear {
            try loadDequantizedLinear(
                base: "merger.mlp.\(idx)", in: weights,
                quantization: quantization, dtype: dtype, device: device)
        }
        let mergerFC1 = try mergerLin(0)
        let mergerFC2 = try mergerLin(2)

        return Qwen25VLVisionModel(
            cfg: cfg, patchEmbedWeight: patchEmbedWeight,
            patchDimPadded: patchDimPadded, blocks: blocks,
            mergerNorm: mergerNorm, mergerFC1: mergerFC1, mergerFC2: mergerFC2,
            textHidden: textHidden, dtype: dtype, gridSide: defaultGridSide)
    }

    /// Run the full vision forward on a preprocessed image. `image` is a
    /// normalized NCHW tensor `[1, inChannels, H, W]` at the native
    /// `smartResize` resolution (`H`, `W` multiples of `patch · merge`, not
    /// necessarily square); the grid `(H/patch, W/patch)` is derived from
    /// the tensor. Returns `[mergedTokenCount, textHidden]`.
    func encode(image: Tensor, device: Device) -> Tensor {
        // Single-image path: gridT = 1, the unfold copies the same frame
        // `temporal_patch_size` times in the temporal axis. Reuse the
        // multi-frame path so all the temporal-aware bookkeeping
        // (M-RoPE, window groups, merger) goes through one code path.
        let (gridH, gridW) = Self.gridDims(for: image, cfg: cfg)
        return runForward(
            framePixels: [image.toFloatArray()], gridT: 1,
            gridH: gridH, gridW: gridW,
            framePadIndices: nil, device: device)
    }

    /// Run the full vision forward on `frames` preprocessed video frames
    /// (each a normalized NCHW tensor `[1, inChannels, side, side]`).
    /// Frames are padded to a multiple of `temporal_patch_size` by
    /// repeating the last frame (matches the mlx-vlm reference's
    /// `patchify` path). Returns `[mergedTokenCount, textHidden]` where
    /// `mergedTokenCount = (T/tp) · gridSide² / mergeUnit`.
    ///
    /// Precondition: every frame shares one native NCHW shape
    /// `[1, inChannels, H, W]` (the caller resizes all frames to the same
    /// `smartResize` target); the grid `(H/patch, W/patch)` is derived from
    /// frame 0.
    func encode(frames: [Tensor], device: Device) -> Tensor {
        precondition(
            !frames.isEmpty,
            "Qwen25VL.encode(frames:): expected at least one frame")
        let (gridH, gridW) = Self.gridDims(for: frames[0], cfg: cfg)
        for (i, frame) in frames.enumerated() {
            precondition(
                frame.shape == frames[0].shape,
                "Qwen25VL.encode(frames:): frame[\(i)] shape \(frame.shape) "
                    + "!= frame[0] \(frames[0].shape) (all frames must share "
                    + "one resolution)")
        }

        // Pad frame count up to a multiple of `temporal_patch_size` by
        // repeating the last frame — same as mlx-vlm's `patchify`.
        let tp = cfg.temporalPatchSize
        let mod = frames.count % tp
        var framePixels: [[Float]] = frames.map { $0.toFloatArray() }
        if mod != 0 {
            let pad = tp - mod
            if let last = framePixels.last {
                for _ in 0 ..< pad { framePixels.append(last) }
            }
        }
        let paddedT = framePixels.count
        let gridT = paddedT / tp
        return runForward(
            framePixels: framePixels, gridT: gridT,
            gridH: gridH, gridW: gridW,
            framePadIndices: nil, device: device)
    }

    /// Shared forward path for image / video. `framePixels` has either
    /// 1 element (image, replicated `tp` times in the unfold) or
    /// `gridT * tp` elements (video, one per real frame). `gridT` is the
    /// number of temporal patches the unfold will produce.
    private func runForward(
        framePixels: [[Float]], gridT: Int,
        gridH: Int, gridW: Int,
        framePadIndices: [Int]?,
        device: Device
    ) -> Tensor {
        let nPatches = gridT * gridH * gridW

        // ── Patch unfold + embed ── (unfold → GEMM on one cmd buffer)
        let cmd = device.makeCommandBuffer()
        let unfolded = unfoldPatches(
            framePixels: framePixels, gridT: gridT,
            gridH: gridH, gridW: gridW, on: cmd)
        var h = Ops.gemm(
            weight: patchEmbedWeight, input: unfolded,
            nRows: nPatches, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── M-RoPE tables + window groups ──
        let (cosTable, sinTable) = ropeTables(
            gridT: gridT, gridH: gridH, gridW: gridW)
        let windowGroups = windowAttentionGroups(
            gridT: gridT, gridH: gridH, gridW: gridW)
        let fullGroup = [Array(0 ..< nPatches)]

        // ── Block stack ──
        for (i, block) in blocks.enumerated() {
            let groups =
                cfg.fullattBlockIndexes.contains(i)
                ? fullGroup : windowGroups
            h = block.forward(
                h, nTokens: nPatches,
                cosTable: cosTable, sinTable: sinTable,
                windowGroups: groups, device: device)
        }

        // ── Patch merger ──
        return mergePatches(h, nPatches: nPatches, device: device)
    }

    /// Present the tower as a `VisionEncoder` so `VisionModel` accepts it.
    func asVisionEncoder() -> VisionEncoder {
        Qwen25VLComposedEncoder(tower: self)
    }

    // ── Patch unfold ──

    /// Unfold normalized NCHW frames into a `[nPatches, patchDim]`
    /// tensor. `framePixels` is a list of length 1 (image, replicated
    /// `tp` times along the temporal axis) or `gridT · tp` (video, one
    /// entry per real frame in display order). Each row holds one
    /// `tP · C · py · px` patch flattened in `(temporal, channel, py, px)`
    /// order, matching the flattened patch-embed weight's column layout.
    ///
    /// Row order is `(tGroup, mergeBlockRow, mergeBlockCol, intraRow,
    /// intraCol)` — temporal patches are blocked together and each run of
    /// `m²` rows is one 2×2 spatial neighbourhood, so the merger's reshape
    /// `[nPatches/mergeUnit, mergeUnit·hidden]`, the window grouping, and
    /// the M-RoPE position table all line up. (Previously this emitted
    /// plain row-major raster, which disagreed with the merge-block layout
    /// the other three stages assume — every token past index 1 was pooled
    /// / positioned / windowed against the wrong patch. Now matches the
    /// validated Qwen2-VL unfold.)
    private func unfoldPatches(
        framePixels: [[Float]], gridT: Int, gridH: Int, gridW: Int,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        let p = cfg.patchSize
        let c = cfg.inChannels
        let tp = cfg.temporalPatchSize
        let m = cfg.spatialMergeSize
        let mergeBlocksH = gridH / m
        let mergeBlocksW = gridW / m
        let imgH = gridH * p
        let imgW = gridW * p
        // Image fast path: a single frame reused for every temporal slot.
        let isImage = framePixels.count == 1
        // Concatenate the per-frame pixel buffers into one
        // `[nFrames, c, imgH, imgW]` tensor; the GPU kernel emits the
        // merge-block raster + `(t, c, py, px)` columns, zero-padded.
        let flat = framePixels.flatMap { $0 }
        let framesT = MediaUtils.makeTensor(
            from: flat, shape: [framePixels.count, c, imgH, imgW],
            dtype: dtype, device: .shared)
        return Ops.patchUnfoldQwen(
            frames: framesT, channels: c, patch: p, temporalPatch: tp,
            merge: m, mergeBlocksH: mergeBlocksH, mergeBlocksW: mergeBlocksW,
            imgH: imgH, imgW: imgW, gridT: gridT,
            patchDimPadded: patchDimPadded, isImage: isImage, on: cmd)
    }

    // ── M-RoPE ──

    /// Build the per-token rotary `cos` / `sin` tables `[nPatches,
    /// headDim]`. Qwen 2.5-VL's vision M-RoPE interleaves a height-rotary
    /// and a width-rotary half over `headDim`, each driven by the patch's
    /// `(h, w)` grid coordinate. The merge-size block reorder of the
    /// position ids matches the reference `rot_pos_emb`. For video, the
    /// `(h, w)` pattern is tiled identically per temporal patch — the
    /// temporal axis is carried by the M-RoPE 3-section split in the
    /// text decoder, not by the vision tower's rotary table (the
    /// reference vision tower only rotates over (h, w)).
    private func ropeTables(
        gridT: Int, gridH: Int, gridW: Int
    ) -> (cos: [Float], sin: [Float]) {
        let nPatches = gridT * gridH * gridW
        let headDim = cfg.headDim
        let half = headDim / 2  // height half | width half
        let quarter = half / 2  // distinct rotary frequencies
        // inv_freq over quarter dims, theta 10000 — VisionRotaryEmbedding.
        var invFreq = [Float](repeating: 0, count: quarter)
        for i in 0 ..< quarter {
            invFreq[i] = 1.0 / pow(10_000, Float(2 * i) / Float(half))
        }
        // Per-patch (h, w) coordinates within one frame, reordered into
        // merge-size blocks. The same per-frame table is tiled `gridT`
        // times along the temporal axis (the reference vision tower's
        // rotary embedding only depends on spatial position).
        let (hPos, wPos) = mergeReorderedPositions(gridH: gridH, gridW: gridW)
        let perFrame = gridH * gridW

        var cosT = [Float](repeating: 0, count: nPatches * headDim)
        var sinT = [Float](repeating: 0, count: nPatches * headDim)
        for t in 0 ..< nPatches {
            let spatial = t % perFrame
            let base = t * headDim
            // Height rotary fills [0, half); width rotary fills [half, 2·half).
            for i in 0 ..< quarter {
                let fh = Float(hPos[spatial]) * invFreq[i]
                let fw = Float(wPos[spatial]) * invFreq[i]
                // freqs are tiled to half (duplicate for rotate-half).
                for (off, f) in [(i, fh), (i + quarter, fh)] {
                    cosT[base + off] = cos(f)
                    sinT[base + off] = sin(f)
                }
                for (off, f) in [(half + i, fw), (half + i + quarter, fw)] {
                    cosT[base + off] = cos(f)
                    sinT[base + off] = sin(f)
                }
            }
        }
        return (cosT, sinT)
    }

    /// Per-patch `(h, w)` grid coordinates within one frame, reordered
    /// into the merge-size block layout the reference's `rot_pos_emb`
    /// produces.
    private func mergeReorderedPositions(
        gridH: Int, gridW: Int
    ) -> (h: [Int], w: [Int]) {
        let m = cfg.spatialMergeSize
        let mergeBlocksH = gridH / m
        let mergeBlocksW = gridW / m
        var hPos = [Int](repeating: 0, count: gridH * gridW)
        var wPos = [Int](repeating: 0, count: gridH * gridW)
        var idx = 0
        for br in 0 ..< mergeBlocksH {
            for bc in 0 ..< mergeBlocksW {
                for ir in 0 ..< m {
                    for ic in 0 ..< m {
                        hPos[idx] = br * m + ir
                        wPos[idx] = bc * m + ic
                        idx += 1
                    }
                }
            }
        }
        return (hPos, wPos)
    }

    /// Partition the patch tokens into window-attention groups. The
    /// patches are already in `(t, merge-block raster)` order, so each
    /// temporal patch is a `gridSide × gridSide` block of merge-units —
    /// window groups are per-frame (matching the reference, which keeps
    /// windowed attention strictly within one temporal patch).
    private func windowAttentionGroups(
        gridT: Int, gridH: Int, gridW: Int
    ) -> [[Int]] {
        let m = cfg.spatialMergeSize
        // LLM (merged) grid dims — windows tile this grid, with separate
        // row/column extents for a non-square image.
        let mergeBlocksH = gridH / m
        let mergeBlocksW = gridW / m
        // window in units of merge-blocks.
        let winBlocks = max(1, cfg.windowSize / (m * cfg.patchSize))
        let nWinH = (mergeBlocksH + winBlocks - 1) / winBlocks
        let nWinW = (mergeBlocksW + winBlocks - 1) / winBlocks
        let perFrame = gridH * gridW
        var groups: [[Int]] = []
        for tIdx in 0 ..< gridT {
            let tBase = tIdx * perFrame
            for wr in 0 ..< nWinH {
                for wc in 0 ..< nWinW {
                    var group: [Int] = []
                    for lbr in (wr * winBlocks) ..< min((wr + 1) * winBlocks, mergeBlocksH) {
                        for lbc in (wc * winBlocks) ..< min((wc + 1) * winBlocks, mergeBlocksW) {
                            // Row stride is the number of block-columns
                            // (`mergeBlocksW`), not a square side — each
                            // merge-block owns `m²` consecutive token slots.
                            let blockBase = tBase + (lbr * mergeBlocksW + lbc) * m * m
                            for k in 0 ..< (m * m) { group.append(blockBase + k) }
                        }
                    }
                    if !group.isEmpty { groups.append(group) }
                }
            }
        }
        return groups
    }

    // ── Patch merger ──

    /// Pool each `mergeSize²` neighbourhood of post-encoder tokens into
    /// one token, then project into the text hidden dim. The tokens are
    /// in `(t, merge-block raster)` order, so each consecutive run of
    /// `mergeUnit` tokens is one neighbourhood. Returns
    /// `[nPatches/mergeUnit, textHidden]`.
    private func mergePatches(
        _ h: Tensor, nPatches: Int,
        device: Device
    ) -> Tensor {
        let hidden = cfg.hidden
        let mergeUnit = cfg.mergeUnit
        let merged = nPatches / mergeUnit

        // RMSNorm each token, then group `mergeUnit` tokens into one row.
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(
            h, weight: mergerNorm.weight,
            eps: mergerNorm.eps, nRows: nPatches,
            rowSize: hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // The merger MLP input dim is `mergeUnit · hidden`; the tokens
        // are already contiguous per neighbourhood, so a plain reshape
        // of the buffer view to `[merged, mergeUnit·hidden]` suffices.
        let grouped = normed.reshaped(to: [merged, mergeUnit * hidden])

        let cmd2 = device.makeCommandBuffer()
        var x = Ops.gemm(
            weight: mergerFC1.weight, input: grouped,
            nRows: merged, on: cmd2)
        if let b = mergerFC1.bias {
            x = addRowBias(
                x, bias: b, nRows: merged,
                rowSize: mergerFC1.weight.shape[0], on: cmd2)
        }
        x = Ops.gelu(x, on: cmd2)
        var y = Ops.gemm(
            weight: mergerFC2.weight, input: x,
            nRows: merged, on: cmd2)
        if let b = mergerFC2.bias {
            y = addRowBias(
                y, bias: b, nRows: merged,
                rowSize: mergerFC2.weight.shape[0], on: cmd2)
        }
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return y
    }

    // Shared `addRowBias`, `padLinearRows`, `padLinearCols`, and
    // `flattenPatchEmbed` helpers live in `VisionTowerOps.swift`.
}

// ─── Encoder facade ──────────────────────────────────────────────────

/// A `VisionEncoder` subclass whose `encode` runs the Qwen 2.5-VL vision
/// tower — so `VisionModel` (which holds a `VisionEncoder`) transparently
/// gets the merged, projected vision tokens.
final class Qwen25VLComposedEncoder: VisionEncoder {
    let tower: Qwen25VLVisionModel

    init(tower: Qwen25VLVisionModel) {
        self.tower = tower
        let c = tower.cfg
        // The facade config reports the square image side the tower
        // expects (`gridSide · patchSize`) and the merged token count
        // as `numPatches` so `VisionModel.imageTokenCount` is correct.
        let side = tower.gridSide * c.patchSize
        let facadeConfig = VisionEncoderConfig(
            inChannels: c.inChannels, imageSize: side,
            patchSize: side / Int(Double(tower.mergedTokenCount).squareRoot()),
            hidden: c.hidden, intermediate: c.intermediate,
            nLayers: c.depth, nHeads: c.numHeads,
            layerNormEps: c.rmsNormEps, textHidden: tower.textHidden)
        // The patch-embed / position / layers are unused by the override
        // below; pass minimal placeholders.
        let placeholderW = tower.patchEmbedWeight
        super.init(
            config: facadeConfig,
            patchEmbedWeight: placeholderW, patchEmbedBias: placeholderW,
            positionEmbedding: placeholderW, layers: [],
            postLayerNorm: tower.mergerNorm.asLayerNorm(),
            projection: nil, dtype: tower.dtype)
    }

    /// Native-resolution preprocessing: `smartResize` the raw image to the
    /// largest aspect-preserving grid under Qwen2.5-VL's token budget (sides
    /// a multiple of `patch · merge`), then GPU bilinear-resize + CLIP-
    /// normalize → NCHW `[1, 3, H, W]`. Replaces the old fixed 392² square.
    /// Falls back to the generic fixed square (`nil`) only for a degenerate
    /// aspect ratio `smartResize` rejects (> 200:1).
    override func preprocessImage(
        _ image: RGBImage, device: Device = .shared
    ) -> Tensor? {
        guard
            let (th, tw) = try? MediaUtils.smartResize(
                height: image.height, width: image.width,
                bounds: Qwen25VLVisionModel.smartResizeBounds(cfg: tower.cfg))
        else {
            return nil
        }
        // Bicubic: smart_resize routinely down-scales multi-MP photos to a
        // few hundred px/side; bicubic keeps text + fine detail bilinear softens.
        return MediaUtils.resizeNormalize(
            image, targetW: tw, targetH: th,
            normalization: .clip, dtype: tower.dtype,
            mode: .bicubic, device: device)
    }

    /// Run the Qwen 2.5-VL vision tower on a single preprocessed image.
    /// Returns `[mergedTokenCount, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        tower.encode(image: image, device: device)
    }

    /// Run the Qwen 2.5-VL vision tower on a sequence of preprocessed
    /// video frames. Returns
    /// `[(T/temporalPatchSize) · mergedTokenCount, textHidden]`.
    override func encode(frames: [Tensor], device: Device = .shared) throws -> Tensor {
        tower.encode(frames: frames, device: device)
    }
}

extension RMSNorm {
    /// A throw-away `LayerNorm` view used only to satisfy the
    /// `VisionEncoder` initializer for the Qwen 2.5-VL facade (whose
    /// `encode` is fully overridden, so the post-norm is never invoked).
    func asLayerNorm() -> LayerNorm {
        let zeroBias = Tensor.empty(shape: weight.shape, dtype: weight.dtype)
        return LayerNorm(weight: weight, bias: zeroBias, eps: eps)
    }
}
