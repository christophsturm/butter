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
// Muse Glimmer vision — the ~1.8B ViT-G/14 perception encoder + the
// vision-adapter/vision-projection MLP that maps merged patch tokens
// into the text hidden dim.
//
// The family root (`Models/MuseGlimmer.swift`) documents the text side.
// This file adds the vision path: `MuseGlimmerVisionConfig`,
// `MuseGlimmerVisionBlock`, `MuseGlimmerVisionModel`,
// `MuseGlimmerComposedEncoder`, and the `MuseGlimmerVL` family
// orchestrator that composes this tower with `MuseGlimmerDense` (the
// existing text engine, extended with embedding-input forward — see
// `MuseGlimmerModel.forward(inputEmbedding:...)` in `MuseGlimmerText.swift`)
// via `VisionModel`'s cross-modal splice.
//
// ── Architecture (from `vision_config` + the checkpoint's tensor map —
//    no external reference implementation exists for this checkpoint;
//    every geometric fact below was cross-checked against the actual
//    safetensors shapes) ──
//
//   • Patch embed is a PLAIN LINEAR, not a Conv3d: the checkpoint stores
//     `vision_tower.patch_embedder.patch_embedding.weight` as a 2-D
//     `[hidden, inChannels·temporalPatch·patch·patch]` matrix directly
//     (1536×1176 on this checkpoint) — one `Ops.gemm` per patch, no
//     `flattenPatchEmbed` reshape needed. Column order is `(temporal,
//     channel, patchRow, patchCol)`, matching the Qwen-VL family
//     convention every other dynamic-resolution tower in this file uses.
//   • A LEARNED absolute position-embedding table
//     (`patch_embedder.position_embedding_table.weight`,
//     `[pos_emb_height·pos_emb_width, hidden]` = 32×32 = 1024 rows) is
//     added post patch-embed, pre-`ln_pre` — same mechanism as
//     `Qwen3VLVisionModel.posEmbedTable` (direct gather when the test
//     grid matches the table's native side, nearest-neighbour
//     otherwise).
//   • 2-D RoPE ALSO runs inside attention (`rope_parameters.rope_theta`
//     = 10000) — this tower combines a learned table AND rotary
//     position info, unlike Qwen 2.5-VL (rope only) or Qwen 3-VL
//     (learned table only). The rotary math (height/width split over
//     `headDim`) is bit-identical to both Qwen towers' `ropeTables`.
//   • Pre-norm transformer blocks: `LayerNorm(norm1) → MHA(+RoPE) →
//     residual, LayerNorm(norm2) → GELU MLP(fc1/fc2) → residual` — same
//     block shape as `Qwen3VLVisionBlock`, except attention uses THREE
//     separate `q_proj`/`k_proj`/`v_proj` projections (all with bias)
//     instead of one fused `qkv` — the checkpoint ships them unfused.
//   • `layer_types` (50 entries, `"window_attention"` / `"full_attention"`,
//     pattern (3× window + 1× full) × 12 + one extra window layer) picks
//     per-layer attention scope, same mechanism as Qwen 2.5-VL's
//     `fullatt_block_indexes` — but `vision_config` carries NO explicit
//     pixel `window_size` the way Qwen 2.5-VL does. **Documented
//     assumption**: the window granularity is the tower's OWN learned
//     position-embedding tile — `windowPatches = pos_emb_height` (32) —
//     i.e. "window" attention is scoped to one native absolute-position
//     tile, and "full" attention spans the whole (possibly multi-tile)
//     grid. This is a reasoned default, not a verified fact (no
//     reference implementation exists for this checkpoint to check
//     against): for the single-tile 32×32-patch default test image the
//     default `encode(image:)` path uses, `windowPatches == gridSide`,
//     so windowed and full-attention layers are NUMERICALLY IDENTICAL
//     (one window == the whole image) — the assumption therefore does
//     not affect end-to-end correctness for standard-size images, only
//     for a hypothetical multi-tile input larger than 32×32 patches.
//     The windowing MACHINERY itself (grouping + masking) is exercised
//     directly by `MuseGlimmerVisionTests`'s synthetic small-grid case.
//   • Patch merger: NO extra learned norm at the merge step (unlike
//     Qwen 3-VL's `merger.norm`) — `ln_post` already normalizes the
//     tower's final hidden state, so merging is a pure reshape:
//     `mergeSize²` (2×2=4) consecutive merge-block-order tokens
//     concatenate into one `[mergeUnit·hidden]` = `[6144]` row.
//   • Projector: `vision_adapter.fc1` (6144→4096, GELU) →
//     `vision_adapter.fc2` (4096→4096, no activation) →
//     `vision_projection` (4096→6656 = text hidden, no activation). All
//     three are 4-bit affine-quantized (group_size 64) — unlike the
//     vision tower itself, which ships bf16 throughout. **Documented
//     assumption**: exactly one GELU, between `fc1` and `fc2` (the
//     standard 2-layer-MLP-adapter shape `projector_hidden_act: "gelu"`
//     names), with `vision_projection` a final plain linear resize with
//     no activation before it — the natural reading of three separate
//     named linears where only two (`fc1`/`fc2`) are grouped under one
//     `vision_adapter` module and the config exposes a single activation
//     field.

import Foundation
import Metal

// ─── Vision tower configuration ──────────────────────────────────────

/// Static shape of the Muse Glimmer ViT-G/14 vision tower, decoded from
/// `vision_config`.
struct MuseGlimmerVisionConfig {
    let depth: Int
    let hidden: Int
    let intermediate: Int
    let numHeads: Int
    let patchSize: Int
    let mergeSize: Int
    let temporalPatchSize: Int
    let layerNormEps: Float
    let posEmbHeight: Int
    let posEmbWidth: Int
    /// Per-layer `"window_attention"` / `"full_attention"` — `nil` falls
    /// back to the documented (3× window + 1× full) × N pattern.
    let layerTypes: [String]?
    let ropeTheta: Float
    let inChannels: Int

    /// Per-head dimension.
    var headDim: Int { hidden / numHeads }
    /// `mergeSize²` — patches pooled into one merged token.
    var mergeUnit: Int { mergeSize * mergeSize }

    static func decode(_ c: ModelConfig) throws -> MuseGlimmerVisionConfig {
        guard let depth = c.int("num_hidden_layers"),
            let hidden = c.int("hidden_size"),
            let numHeads = c.int("num_attention_heads"),
            let patchSize = c.int("patch_size"),
            let mergeSize = c.int("merge_size")
        else {
            throw MuseGlimmerError.missingConfig(
                "vision_config: num_hidden_layers/hidden_size/"
                    + "num_attention_heads/patch_size/merge_size")
        }
        let intermediate = c.int("intermediate_size") ?? hidden * 4
        let ropeTheta = Float(c.nested("rope_parameters")?["rope_theta"] as? Double ?? 10_000.0)
        return MuseGlimmerVisionConfig(
            depth: depth, hidden: hidden, intermediate: intermediate,
            numHeads: numHeads, patchSize: patchSize, mergeSize: mergeSize,
            temporalPatchSize: c.int("patch_temporal") ?? 2,
            layerNormEps: Float(c.float("layer_norm_eps") ?? 1e-5),
            posEmbHeight: c.int("pos_emb_height") ?? 32,
            posEmbWidth: c.int("pos_emb_width") ?? 32,
            layerTypes: c.raw["layer_types"] as? [String],
            ropeTheta: ropeTheta,
            inChannels: c.int("in_channels") ?? c.int("in_chans") ?? 3)
    }

    /// Whether layer `i` is a full-attention layer. Falls back to the
    /// documented (3× window + 1× full) × N pattern — same shape as the
    /// text backbone's `layer_types` fallback — when the checkpoint
    /// doesn't ship `layer_types`.
    func isFullAttentionLayer(_ i: Int) -> Bool {
        if let types = layerTypes, i < types.count {
            return types[i] == "full_attention"
        }
        return (i + 1) % 4 == 0
    }

    /// Window granularity in PATCHES (not merge-blocks) — see the file
    /// header's "documented assumption" note. Defaults to the learned
    /// position table's native side, `pos_emb_height`.
    var windowPatches: Int { posEmbHeight }
}

/// Partition a `gridSide × gridSide` (per temporal group) merge-block-
/// raster-ordered patch grid into window-attention groups of
/// `windowPatches × windowPatches` (`winBlocks = windowPatches /
/// mergeSize` merge-blocks per window side). A standalone, weight-free
/// function — both `MuseGlimmerVisionModel.windowAttentionGroups` and
/// `MuseGlimmerVisionTests`' oracle test call it directly.
///
/// Patch index within one temporal group follows the merge-block raster
/// order every other geometry function in this file uses: for merge
/// block `(br, bc)`, the `mergeSize²` patches `(br·m+ir, bc·m+ic)` occupy
/// `mergeSize²` CONSECUTIVE indices starting at `(br·mergeBlocks+bc)·m²`.
func museGlimmerWindowAttentionGroups(
    gridSide: Int, mergeSize: Int, windowPatches: Int, gridT: Int
) -> [[Int]] {
    let side = gridSide
    let m = mergeSize
    let mergeBlocks = side / m
    let winBlocks = max(1, windowPatches / m)
    let nWin = (mergeBlocks + winBlocks - 1) / winBlocks
    let perFrame = side * side
    var groups: [[Int]] = []
    for tIdx in 0 ..< gridT {
        let tBase = tIdx * perFrame
        for wr in 0 ..< nWin {
            for wc in 0 ..< nWin {
                var group: [Int] = []
                for lbr in (wr * winBlocks) ..< min((wr + 1) * winBlocks, mergeBlocks) {
                    for lbc in (wc * winBlocks) ..< min((wc + 1) * winBlocks, mergeBlocks) {
                        let blockBase = tBase + (lbr * mergeBlocks + lbc) * m * m
                        for k in 0 ..< (m * m) { group.append(blockBase + k) }
                    }
                }
                if !group.isEmpty { groups.append(group) }
            }
        }
    }
    return groups
}

// ─── Vision block ────────────────────────────────────────────────────

/// One Muse Glimmer vision block: LayerNorm → windowed/full MHA + 2-D
/// RoPE → residual, LayerNorm → GELU MLP → residual. Unlike the Qwen-VL
/// family's fused `qkv` projection, this checkpoint ships separate
/// `q_proj`/`k_proj`/`v_proj` (all with bias).
final class MuseGlimmerVisionBlock {
    let norm1: LayerNorm
    let norm2: LayerNorm
    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let proj: Linear
    let fc1: Linear
    let fc2: Linear
    let cfg: MuseGlimmerVisionConfig
    /// GELU-MLP intermediate dim rounded up to the GEMM K-tile width.
    let paddedIntermediate: Int

    init(
        norm1: LayerNorm, norm2: LayerNorm,
        qProj: Linear, kProj: Linear, vProj: Linear, proj: Linear,
        fc1: Linear, fc2: Linear,
        paddedIntermediate: Int, cfg: MuseGlimmerVisionConfig
    ) {
        self.norm1 = norm1
        self.norm2 = norm2
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.proj = proj
        self.fc1 = fc1
        self.fc2 = fc2
        self.paddedIntermediate = paddedIntermediate
        self.cfg = cfg
    }

    /// Forward `[nTokens, hidden]` activations through one block.
    func forward(
        _ h: Tensor, nTokens: Int,
        cosTable: [Float], sinTable: [Float],
        windowGroups: [[Int]], device: Device
    ) -> Tensor {
        let hidden = cfg.hidden
        // ── Attention sub-block ──
        let cmd = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            h, weight: norm1.weight, bias: norm1.bias,
            eps: norm1.eps, nRows: nTokens, rowSize: hidden, on: cmd)
        let q = projectRows(qProj, normed, nTokens: nTokens, outDim: hidden, on: cmd)
        let k = projectRows(kProj, normed, nTokens: nTokens, outDim: hidden, on: cmd)
        let v = projectRows(vProj, normed, nTokens: nTokens, outDim: hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let attn = cpuAttention(
            q: q, k: k, v: v, nTokens: nTokens,
            cosTable: cosTable, sinTable: sinTable,
            windowGroups: windowGroups, device: device)

        // ── Residual + MLP sub-block ──
        let cmd2 = device.makeCommandBuffer()
        let attnProj = projectRows(proj, attn, nTokens: nTokens, outDim: hidden, on: cmd2)
        let postAttn = Ops.add(h, attnProj, on: cmd2)
        let normed2 = Ops.layerNorm(
            postAttn, weight: norm2.weight, bias: norm2.bias,
            eps: norm2.eps, nRows: nTokens, rowSize: hidden, on: cmd2)
        let up = projectRows(
            fc1, normed2, nTokens: nTokens, outDim: paddedIntermediate, on: cmd2)
        let act = Ops.gelu(up, on: cmd2)
        let down = projectRows(fc2, act, nTokens: nTokens, outDim: hidden, on: cmd2)
        let result = Ops.add(postAttn, down, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return result
    }

    /// CPU windowed multi-head attention with 2-D RoPE. `q`/`k`/`v` are
    /// separate `[nTokens, hidden]` projections. Returns the context
    /// token-major `[nTokens, hidden]`.
    ///
    /// Same two-stage `DispatchQueue.concurrentPerform` parallelization
    /// as `Qwen25VLVisionBlock.cpuAttention` (see that file for the
    /// race-free indexing argument) — adapted to read from three
    /// separate flat arrays instead of slicing one fused `qkv` buffer.
    private func cpuAttention(
        q: Tensor, k: Tensor, v: Tensor, nTokens: Int,
        cosTable: [Float], sinTable: [Float],
        windowGroups: [[Int]], device: Device
    ) -> Tensor {
        let nHeads = cfg.numHeads
        let headDim = cfg.headDim
        let hidden = cfg.hidden
        let qA = q.toFloatArray()
        let kA = k.toFloatArray()
        let vA = v.toFloatArray()
        var out = [Float](repeating: 0, count: nTokens * hidden)
        let scale = 1.0 / Float(Double(headDim).squareRoot())
        let half = headDim / 2

        // Stage 1: extract + RoPE every (head, token) slice.
        nonisolated(unsafe) var qH = [[Float]](repeating: [], count: nHeads * nTokens)
        nonisolated(unsafe) var kH = [[Float]](repeating: [], count: nHeads * nTokens)
        nonisolated(unsafe) var vH = [[Float]](repeating: [], count: nHeads * nTokens)
        DispatchQueue.concurrentPerform(iterations: nHeads * nTokens) { work in
            let head = work / nTokens
            let t = work % nTokens
            let hOff = head * headDim
            let base = t * hidden
            var qv = Array(qA[(base + hOff) ..< (base + hOff + headDim)])
            var kv = Array(kA[(base + hOff) ..< (base + hOff + headDim)])
            let vv = Array(vA[(base + hOff) ..< (base + hOff + headDim)])
            func applyRope(_ x: inout [Float], tok: Int) {
                let cb = tok * headDim
                for d in 0 ..< headDim {
                    let rotated = d < half ? -x[d + half] : x[d - half]
                    let c = cosTable[cb + d]
                    let s = sinTable[cb + d]
                    x[d] = x[d] * c + rotated * s
                }
            }
            applyRope(&qv, tok: t)
            applyRope(&kv, tok: t)
            qH[head * nTokens + t] = qv
            kH[head * nTokens + t] = kv
            vH[head * nTokens + t] = vv
        }

        // Stage 2: attention. Full-attention groups (a single group
        // spanning every token) dispatch through the GPU
        // `Ops.sdpaBidirectional` kernel at head_dim 96 (supported);
        // real windowed groups keep the parallel CPU loop.
        let isFullAttention = windowGroups.count == 1 && windowGroups[0].count == nTokens
        if isFullAttention && OpsValidation.sdpaBidirectionalSupportedHeadDims.contains(headDim) {
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
            let qT = Tensor.empty(shape: [nTokens, nHeads, headDim], dtype: .f32, device: device)
            ImagePreprocessing.copyFloats(qFlat, into: qT)
            let kT = Tensor.empty(shape: [nHeads, nTokens, headDim], dtype: .f32, device: device)
            ImagePreprocessing.copyFloats(kFlat, into: kT)
            let vT = Tensor.empty(shape: [nHeads, nTokens, headDim], dtype: .f32, device: device)
            ImagePreprocessing.copyFloats(vFlat, into: vT)
            let cmd = device.makeCommandBuffer()
            let outT = Ops.sdpaBidirectional(
                q: qT, k: kT, v: vT,
                nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
                baseKV: 0, nQuery: nTokens, kvStride: nTokens,
                scale: scale, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            let outFlat = outT.toFloatArray()
            let result = Tensor.empty(shape: [nTokens, hidden], dtype: q.dtype, device: device)
            ImagePreprocessing.copyFloats(outFlat, into: result)
            return result
        }

        // CPU fallback — real windowed attention (or an unsupported head
        // dim's full-attention group; not applicable at headDim 96 here
        // but kept general).
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
        let result = Tensor.empty(shape: [nTokens, hidden], dtype: q.dtype, device: device)
        ImagePreprocessing.copyFloats(out, into: result)
        return result
    }

    private func projectRows(
        _ linear: Linear, _ x: Tensor, nTokens: Int,
        outDim: Int, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let y = Ops.gemm(weight: linear.weight, input: x, nRows: nTokens, on: cmd)
        guard let bias = linear.bias else { return y }
        return addRowBias(y, bias: bias, nRows: nTokens, rowSize: outDim, on: cmd)
    }
}

// ─── Vision tower ────────────────────────────────────────────────────

/// The Muse Glimmer ViT-G/14 vision tower. Holds the patch-embed
/// projection, the learned position table, the block stack, and the
/// (quantized) adapter + projection; `encode(image:)` runs the full
/// forward and returns merged tokens in the text hidden dim.
final class MuseGlimmerVisionModel: @unchecked Sendable {
    let cfg: MuseGlimmerVisionConfig
    /// Patch-embed weight `[hidden, patchDimPadded]` — already a plain
    /// 2-D matrix on this checkpoint (no Conv3d reshape needed), just
    /// zero-padded to the GEMM K-tile width.
    let patchEmbedWeight: Tensor
    let patchDimPadded: Int
    /// Learned position embedding table `[posEmbHeight·posEmbWidth, hidden]`.
    let posEmbedTable: Tensor
    let lnPre: LayerNorm
    let blocks: [MuseGlimmerVisionBlock]
    let lnPost: LayerNorm
    /// Quantized projector: `fc1(6144→4096) → GELU → fc2(4096→4096) →
    /// vision_projection(4096→textHidden)`.
    let adapterFC1: AnyLinear
    let adapterFC2: AnyLinear
    let projection: AnyLinear
    let textHidden: Int
    let dtype: DType
    /// Patch grid side for the default single-tile test image — sized
    /// to the learned position table's native side so the position
    /// lookup is a direct gather.
    let gridSide: Int

    /// Number of merged vision tokens one image contributes.
    var mergedTokenCount: Int {
        let llmSide = gridSide / cfg.mergeSize
        return llmSide * llmSide
    }

    init(
        cfg: MuseGlimmerVisionConfig, patchEmbedWeight: Tensor, patchDimPadded: Int,
        posEmbedTable: Tensor, lnPre: LayerNorm, blocks: [MuseGlimmerVisionBlock],
        lnPost: LayerNorm, adapterFC1: AnyLinear, adapterFC2: AnyLinear,
        projection: AnyLinear, textHidden: Int, dtype: DType, gridSide: Int
    ) {
        self.cfg = cfg
        self.patchEmbedWeight = patchEmbedWeight
        self.patchDimPadded = patchDimPadded
        self.posEmbedTable = posEmbedTable
        self.lnPre = lnPre
        self.blocks = blocks
        self.lnPost = lnPost
        self.adapterFC1 = adapterFC1
        self.adapterFC2 = adapterFC2
        self.projection = projection
        self.textHidden = textHidden
        self.dtype = dtype
        self.gridSide = gridSide
    }

    /// Default test-grid side — matches the learned position table's
    /// native side (`pos_emb_height`, 32 on this checkpoint) so the
    /// default `encode(image:)` path needs no position interpolation.
    static let defaultGridSide = 32

    static func load(
        visionConfig: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundle, quantization: ModelConfig.QuantizationConfig?,
        dtype: DType, device: Device
    ) throws -> MuseGlimmerVisionModel {
        let cfg = try MuseGlimmerVisionConfig.decode(visionConfig)
        let towerWeights = weights.prefixed("vision_tower.")

        // Patch-embed — a plain 2-D matrix on this checkpoint (see the
        // file header): `[hidden, inChannels·temporalPatch·patch·patch]`,
        // no Conv3d reshape needed. Zero-pad the column count up to the
        // GEMM K-tile width.
        let rawPatch = try towerWeights.tensor(named: "patch_embedder.patch_embedding.weight")
        let patchDim = cfg.inChannels * cfg.temporalPatchSize * cfg.patchSize * cfg.patchSize
        precondition(
            rawPatch.shape == [cfg.hidden, patchDim],
            "MuseGlimmerVisionModel.load: patch-embed weight \(rawPatch.shape) "
                + "≠ [\(cfg.hidden), \(patchDim)]")
        let patchDimPadded = ((patchDim + gemmKTileWidth - 1) / gemmKTileWidth) * gemmKTileWidth
        let patchEmbedWeight = padLinearColsTo(rawPatch, toCols: patchDimPadded, device: device)

        let posEmbedTable = try towerWeights.tensor(
            named: "patch_embedder.position_embedding_table.weight")

        let lnPre = LayerNorm(
            weight: try towerWeights.tensor(named: "ln_pre.weight"),
            bias: try towerWeights.tensor(named: "ln_pre.bias"),
            eps: cfg.layerNormEps)
        let lnPost = LayerNorm(
            weight: try towerWeights.tensor(named: "ln_post.weight"),
            bias: try towerWeights.tensor(named: "ln_post.bias"),
            eps: cfg.layerNormEps)

        let paddedIntermediate =
            ((cfg.intermediate + gemmKTileWidth - 1) / gemmKTileWidth) * gemmKTileWidth
        var blocks: [MuseGlimmerVisionBlock] = []
        blocks.reserveCapacity(cfg.depth)
        for i in 0 ..< cfg.depth {
            let p = "layers.\(i)"
            func lin(_ name: String) throws -> Linear {
                let w = try towerWeights.tensor(named: "\(p).\(name).weight")
                let b = try? towerWeights.tensor(named: "\(p).\(name).bias")
                return Linear(weight: w, bias: b)
            }
            func norm(_ name: String) throws -> LayerNorm {
                LayerNorm(
                    weight: try towerWeights.tensor(named: "\(p).\(name).weight"),
                    bias: try towerWeights.tensor(named: "\(p).\(name).bias"),
                    eps: cfg.layerNormEps)
            }
            let fc1 = try lin("mlp.fc1")
            let fc2 = try lin("mlp.fc2")
            blocks.append(
                MuseGlimmerVisionBlock(
                    norm1: try norm("norm1"), norm2: try norm("norm2"),
                    qProj: try lin("attn.q_proj"), kProj: try lin("attn.k_proj"),
                    vProj: try lin("attn.v_proj"), proj: try lin("attn.proj"),
                    fc1: padLinearRows(fc1, toRows: paddedIntermediate, device: device),
                    fc2: padLinearCols(fc2, toCols: paddedIntermediate, device: device),
                    paddedIntermediate: paddedIntermediate, cfg: cfg))
        }

        // Projector — 4-bit affine-quantized on this checkpoint;
        // `loadLinear` reads the (weight, scales, biases) triplet.
        let adapterFC1 = try loadLinear(
            base: "vision_adapter.fc1", in: weights, quantization: quantization)
        let adapterFC2 = try loadLinear(
            base: "vision_adapter.fc2", in: weights, quantization: quantization)
        let projection = try loadLinear(
            base: "vision_projection", in: weights, quantization: quantization)

        return MuseGlimmerVisionModel(
            cfg: cfg, patchEmbedWeight: patchEmbedWeight, patchDimPadded: patchDimPadded,
            posEmbedTable: posEmbedTable, lnPre: lnPre, blocks: blocks, lnPost: lnPost,
            adapterFC1: adapterFC1, adapterFC2: adapterFC2, projection: projection,
            textHidden: textHidden, dtype: dtype, gridSide: defaultGridSide)
    }

    /// Run the full vision forward on a preprocessed image. `image` is a
    /// normalized NCHW tensor `[1, inChannels, side, side]` where
    /// `side = gridSide · patchSize`. Returns `[mergedTokenCount, textHidden]`.
    func encode(image: Tensor, device: Device) -> Tensor {
        runForward(framePixels: [image.toFloatArray()], gridT: 1, device: device)
    }

    private func runForward(framePixels: [[Float]], gridT: Int, device: Device) -> Tensor {
        let side = gridSide
        let nPatches = gridT * side * side

        // ── Patch unfold + embed + learned position embedding ──
        let unfolded = unfoldPatches(framePixels: framePixels, gridT: gridT)
        let cmd = device.makeCommandBuffer()
        var h = Ops.gemm(weight: patchEmbedWeight, input: unfolded, nRows: nPatches, on: cmd)
        let posEmb = mergeReorderedPositionEmbedding(gridT: gridT)
        h = Ops.add(h, posEmb, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── ln_pre ──
        let cmdPre = device.makeCommandBuffer()
        h = Ops.layerNorm(
            h, weight: lnPre.weight, bias: lnPre.bias, eps: lnPre.eps,
            nRows: nPatches, rowSize: cfg.hidden, on: cmdPre)
        cmdPre.commit()
        cmdPre.waitUntilCompleted()

        // ── 2-D RoPE tables + window groups ──
        let (cosTable, sinTable) = ropeTables(gridT: gridT)
        let windowGroups = windowAttentionGroups(gridT: gridT)
        let fullGroup = [Array(0 ..< nPatches)]

        // ── Block stack ──
        for (i, block) in blocks.enumerated() {
            let groups = cfg.isFullAttentionLayer(i) ? fullGroup : windowGroups
            h = block.forward(
                h, nTokens: nPatches, cosTable: cosTable, sinTable: sinTable,
                windowGroups: groups, device: device)
        }

        // ── ln_post ──
        let cmdPost = device.makeCommandBuffer()
        h = Ops.layerNorm(
            h, weight: lnPost.weight, bias: lnPost.bias, eps: lnPost.eps,
            nRows: nPatches, rowSize: cfg.hidden, on: cmdPost)
        cmdPost.commit()
        cmdPost.waitUntilCompleted()

        // ── Merge + projector (no extra norm at merge — ln_post already
        //    normalized) ──
        return projectMerged(h, nPatches: nPatches, device: device)
    }

    func asVisionEncoder() -> VisionEncoder {
        MuseGlimmerComposedEncoder(tower: self)
    }

    // ── Patch unfold ──

    /// Unfold normalized NCHW frames into `[nPatches, patchDimPadded]`,
    /// merge-block-raster ordered (matches `mergeReorderedPositionEmbedding`
    /// / `mergeReorderedPositions` / `windowAttentionGroups` — every
    /// consecutive run of `mergeUnit` rows is one 2×2 spatial
    /// neighbourhood). Column order within a row is `(temporal,
    /// channel, patchRow, patchCol)`.
    private func unfoldPatches(framePixels: [[Float]], gridT: Int) -> Tensor {
        let side = gridSide
        let p = cfg.patchSize
        let c = cfg.inChannels
        let tp = cfg.temporalPatchSize
        let m = cfg.mergeSize
        let mergeBlocks = side / m
        let imgSide = side * p
        let nPatches = gridT * side * side
        let isImage = framePixels.count == 1

        var rows = [Float](repeating: 0, count: nPatches * patchDimPadded)
        var patch = 0
        for tGroup in 0 ..< gridT {
            for br in 0 ..< mergeBlocks {
                for bc in 0 ..< mergeBlocks {
                    for ir in 0 ..< m {
                        for ic in 0 ..< m {
                            let pr = br * m + ir
                            let pc = bc * m + ic
                            var col = 0
                            for tWithin in 0 ..< tp {
                                let frameIdx = isImage ? 0 : tGroup * tp + tWithin
                                let pix = framePixels[frameIdx]
                                for ch in 0 ..< c {
                                    for py in 0 ..< p {
                                        let yy = pr * p + py
                                        for px in 0 ..< p {
                                            let xx = pc * p + px
                                            let v = pix[(ch * imgSide + yy) * imgSide + xx]
                                            rows[patch * patchDimPadded + col] = v
                                            col += 1
                                        }
                                    }
                                }
                            }
                            patch += 1
                        }
                    }
                }
            }
        }
        return ImagePreprocessing.makeTensor(
            from: rows, shape: [nPatches, patchDimPadded], dtype: dtype, device: .shared)
    }

    // ── Learned position embedding ──

    /// Build the per-patch learned position embedding `[nPatches, hidden]`,
    /// merge-block-raster ordered. Direct gather when `gridSide` equals
    /// the table's native side (the default test grid); nearest-neighbour
    /// otherwise (coherence-first — this port doesn't implement bicubic
    /// position-embedding interpolation for arbitrary resolutions).
    private func mergeReorderedPositionEmbedding(gridT: Int) -> Tensor {
        let side = gridSide
        let m = cfg.mergeSize
        let mergeBlocks = side / m
        let hidden = cfg.hidden
        let posSideH = cfg.posEmbHeight
        let posSideW = cfg.posEmbWidth
        let table = posEmbedTable.toFloatArray()
        let perFrame = side * side
        let nPatches = gridT * perFrame

        var frameDst = [Float](repeating: 0, count: perFrame * hidden)
        var idx = 0
        for br in 0 ..< mergeBlocks {
            for bc in 0 ..< mergeBlocks {
                for ir in 0 ..< m {
                    for ic in 0 ..< m {
                        let pr = br * m + ir
                        let pc = bc * m + ic
                        let ty = posSideH > 1 ? min(posSideH - 1, pr * posSideH / max(1, side)) : 0
                        let tx = posSideW > 1 ? min(posSideW - 1, pc * posSideW / max(1, side)) : 0
                        let src = (ty * posSideW + tx) * hidden
                        for d in 0 ..< hidden { frameDst[idx * hidden + d] = table[src + d] }
                        idx += 1
                    }
                }
            }
        }
        var dst = [Float](repeating: 0, count: nPatches * hidden)
        for t in 0 ..< gridT {
            let dstOff = t * perFrame * hidden
            for i in 0 ..< (perFrame * hidden) { dst[dstOff + i] = frameDst[i] }
        }
        return ImagePreprocessing.makeTensor(
            from: dst, shape: [nPatches, hidden], dtype: dtype, device: .shared)
    }

    // ── 2-D RoPE ──

    /// Per-token rotary `cos`/`sin` tables `[nPatches, headDim]` — height
    /// rotary fills the low half, width rotary fills the high half (the
    /// same construction as the Qwen-VL family's vision M-RoPE).
    private func ropeTables(gridT: Int) -> (cos: [Float], sin: [Float]) {
        let side = gridSide
        let nPatches = gridT * side * side
        let headDim = cfg.headDim
        let half = headDim / 2
        let quarter = half / 2
        var invFreq = [Float](repeating: 0, count: quarter)
        for i in 0 ..< quarter {
            invFreq[i] = 1.0 / pow(cfg.ropeTheta, Float(2 * i) / Float(half))
        }
        let (hPos, wPos) = mergeReorderedPositions()
        let perFrame = side * side

        var cosT = [Float](repeating: 0, count: nPatches * headDim)
        var sinT = [Float](repeating: 0, count: nPatches * headDim)
        for t in 0 ..< nPatches {
            let spatial = t % perFrame
            let base = t * headDim
            for i in 0 ..< quarter {
                let fh = Float(hPos[spatial]) * invFreq[i]
                let fw = Float(wPos[spatial]) * invFreq[i]
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

    /// Per-patch `(h, w)` grid coordinates in merge-block raster order.
    private func mergeReorderedPositions() -> (h: [Int], w: [Int]) {
        let side = gridSide
        let m = cfg.mergeSize
        let mergeBlocks = side / m
        var hPos = [Int](repeating: 0, count: side * side)
        var wPos = [Int](repeating: 0, count: side * side)
        var idx = 0
        for br in 0 ..< mergeBlocks {
            for bc in 0 ..< mergeBlocks {
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

    // ── Window attention grouping ──

    /// Partition the merge-block-raster-ordered patch tokens into
    /// window-attention groups of `windowPatches × windowPatches`
    /// (config `windowPatches`, in merge-blocks: `winBlocks =
    /// windowPatches / mergeSize`). See the file header's "documented
    /// assumption" note on window granularity. Thin forward to the
    /// free function below — kept as a standalone, weight-free
    /// function so it's directly unit-testable without constructing a
    /// full tower.
    func windowAttentionGroups(gridT: Int) -> [[Int]] {
        museGlimmerWindowAttentionGroups(
            gridSide: gridSide, mergeSize: cfg.mergeSize,
            windowPatches: cfg.windowPatches, gridT: gridT)
    }

    // ── Merge + projector ──

    /// Group each `mergeSize²` neighbourhood of post-encoder tokens into
    /// one `[mergeUnit·hidden]` row (pure reshape — tokens are already
    /// merge-block-contiguous), then run the quantized adapter +
    /// projection. Returns `[nPatches/mergeUnit, textHidden]`.
    private func projectMerged(_ h: Tensor, nPatches: Int, device: Device) -> Tensor {
        let hidden = cfg.hidden
        let mergeUnit = cfg.mergeUnit
        let merged = nPatches / mergeUnit
        let grouped = h.reshaped(to: [merged, mergeUnit * hidden])

        let cmd = device.makeCommandBuffer()
        let up = adapterFC1.callMany(grouped, t: merged, on: cmd, device: device)
        let act = Ops.gelu(up, on: cmd)
        let down = adapterFC2.callMany(act, t: merged, on: cmd, device: device)
        let proj = projection.callMany(down, t: merged, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        return proj
    }
}

// ─── Encoder facade ──────────────────────────────────────────────────

/// A `VisionEncoder` subclass whose `encode` runs the Muse Glimmer
/// vision tower — so `VisionModel` (which holds a `VisionEncoder`)
/// transparently gets the merged, projected vision tokens.
final class MuseGlimmerComposedEncoder: VisionEncoder {
    let tower: MuseGlimmerVisionModel

    init(tower: MuseGlimmerVisionModel) {
        self.tower = tower
        let c = tower.cfg
        let side = tower.gridSide * c.patchSize
        let facadeConfig = VisionEncoderConfig(
            inChannels: c.inChannels, imageSize: side,
            patchSize: side / Int(Double(tower.mergedTokenCount).squareRoot()),
            hidden: c.hidden, intermediate: c.intermediate,
            nLayers: c.depth, nHeads: c.numHeads,
            layerNormEps: c.layerNormEps, textHidden: tower.textHidden)
        let placeholderW = tower.patchEmbedWeight
        super.init(
            config: facadeConfig,
            patchEmbedWeight: placeholderW, patchEmbedBias: placeholderW,
            positionEmbedding: placeholderW, layers: [],
            postLayerNorm: tower.lnPost,
            projection: nil, dtype: tower.dtype)
    }

    /// Run the Muse Glimmer vision tower on a single preprocessed image.
    /// Returns `[mergedTokenCount, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        tower.encode(image: image, device: device)
    }
}

// ─── Muse Glimmer VL family orchestrator ─────────────────────────────
//
// Composes the vision tower above with `MuseGlimmerDense` (the existing
// text engine — extended with embedding-input forward, see
// `MuseGlimmerModel.forward(inputEmbedding:...)` in
// `Models/Text/MuseGlimmerText.swift`), joined by `VisionModel`'s
// cross-modal token splice: each `<|patch|>` placeholder
// (`image_token_id` = 200092) in the prompt takes one merged vision
// token. TEXT-ONLY behaviour is unaffected — `Loader/Model.swift` only
// routes here when the checkpoint's safetensors actually carry a
// `vision_tower.*` tensor; checkpoints without one still take the
// pre-existing `MuseGlimmerDense`-only path.
//
// Video (`video_token_id` = 200091 / `patch_temporal` framing) is not
// wired in this pass — same scope cut Qwen 3-VL's `deepstack` and
// Qwen 3.5-VL's video-in-this-file precedent set: images first, video
// is a follow-up.

public enum MuseGlimmerVL {
    /// Capabilities a Muse Glimmer checkpoint with a vision tower
    /// exposes: text + image (no video yet — see the header note above).
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.imageIn])

    /// Build a `VisionModel` from a `MuseGlimmerForConditionalGeneration`
    /// checkpoint that carries real `vision_tower.*` tensors: the
    /// dynamic-resolution ViT-G/14 tower + the Muse Glimmer text
    /// backbone, joined by the splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw MuseGlimmerError.missingConfig("vision_config")
        }

        // ── Text backbone — reuse the existing text loader unchanged ──
        let textEngine = try MuseGlimmerDense.loadModel(
            config: config, weights: weights, options: options, device: device)

        // ── Vision tower ──
        // Quantization metadata lives at the checkpoint's TOP LEVEL
        // (same fact `MuseGlimmerDense.loadModel` already documents for
        // the text side) — the vision tower itself ships bf16, but the
        // adapter/projection are quantized under that same root config.
        let quant = config.quantization
        let vision = try MuseGlimmerVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: weights, quantization: quant,
            dtype: textEngine.dtype, device: device)

        let imageTokenId = config.int("image_token_id") ?? 200_092
        return try VisionModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            videoTokenId: nil,
            normalization: .siglip,
            imageTokenCount: vision.mergedTokenCount)
    }
}
