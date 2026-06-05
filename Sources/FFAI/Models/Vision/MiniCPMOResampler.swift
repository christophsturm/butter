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
// MiniCPM-o perceiver resampler — the cross-modal projector that turns a
// variable-length SigLIP patch grid into a fixed `query_num` (64) run of
// text-hidden-dim soft tokens.
//
// It is a one-layer 2D perceiver: a learned `[64, embed_dim]` query set
// cross-attends (plain bidirectional MHA, no causal mask, no RoPE) to the
// projected vision patches, which carry an additive 2D sin-cos position
// embedding generated at runtime for the actual `(H, W)` patch grid.
//
//   x      = kv_proj(patches)            # [num_patches, embed_dim]
//   x      = ln_kv(x)
//   q      = ln_q(query)                 # [64, embed_dim]
//   out    = MHA(q, key = x + posEmb, value = x)   # [64, embed_dim]
//   out    = ln_post(out)
//   out    = out @ proj                  # [64, embed_dim]
//
// Reference: `Resampler` in `modeling_minicpmo.py` (HF
// `openbmb/MiniCPM-o-4_5`, version 4.5). `num_heads = embed_dim // 128`
// (32 for the 4096-dim text hidden). `kv_proj` lifts the 1152-dim SigLIP
// hidden to `embed_dim`; `proj` is a square `[embed_dim, embed_dim]`
// parameter applied as a right-multiply (`out @ proj`).
//
// The reference attention is `nn.MultiheadAttention`, whose fused
// `in_proj_weight` is split by the mlx-community conversion into separate
// `attn.{q,k,v}_proj` + `attn.out_proj` linears (all 4-bit, with bias).
// We load them through `loadLinear` and run a standard scaled-dot-product
// MHA on the GPU GEMMs + a head-dim-agnostic CPU attention core (64
// queries × num_patches keys — tiny relative to the encoder).

import Foundation
import Metal

// ─── MiniCPMOResampler ───────────────────────────────────────────────

/// The MiniCPM-o perceiver resampler. One cross-attention layer mapping a
/// `[num_patches, visionHidden]` SigLIP grid to `[numQueries, embedDim]`
/// text-hidden soft tokens.
final class MiniCPMOResampler {
    /// Learned query set `[numQueries, embedDim]` (`resampler.query`).
    let query: Tensor
    /// `kv_proj` — `visionHidden → embedDim`, no bias (4-bit).
    let kvProj: AnyLinear
    /// MHA query / key / value / output projections — each
    /// `[embedDim, embedDim]`, 4-bit with additive bias.
    let qProj, kProj, vProj, outProj: AnyLinear
    /// `ln_q` over the query, `ln_kv` over the projected patches,
    /// `ln_post` over the attention output.
    let lnQ, lnKV, lnPost: LayerNorm
    /// Square output projection `[embedDim, embedDim]` (`resampler.proj`),
    /// applied as a right-multiply `out @ proj`.
    let proj: Tensor

    /// Number of learned queries (`query_num`, 64) — the per-image token
    /// count the resampler emits.
    let numQueries: Int
    /// Text hidden dim the resampler maps into (`embed_dim`, 4096).
    let embedDim: Int
    /// SigLIP hidden dim of the incoming patches (`vision_dim`, 1152).
    let visionHidden: Int
    /// Attention head count (`embed_dim // 128`, 32).
    let nHeads: Int
    /// Per-head dimension (`embed_dim / nHeads`, 128).
    let headDim: Int
    /// Softmax scale `1 / sqrt(headDim)`.
    let scale: Float
    /// Working dtype (matches the learned `query` / `proj` params).
    let dtype: DType

    init(
        query: Tensor, kvProj: AnyLinear,
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, outProj: AnyLinear,
        lnQ: LayerNorm, lnKV: LayerNorm, lnPost: LayerNorm,
        proj: Tensor, numQueries: Int, embedDim: Int, visionHidden: Int,
        nHeads: Int, dtype: DType
    ) {
        self.query = query
        self.kvProj = kvProj
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.outProj = outProj
        self.lnQ = lnQ
        self.lnKV = lnKV
        self.lnPost = lnPost
        self.proj = proj
        self.numQueries = numQueries
        self.embedDim = embedDim
        self.visionHidden = visionHidden
        self.nHeads = nHeads
        self.headDim = embedDim / nHeads
        self.scale = 1.0 / Float(Double(embedDim / nHeads).squareRoot())
        self.dtype = dtype
    }

    /// Per-head dimension used as the 2D sin-cos position-embedding width.
    /// 128 for the 4096-dim text hidden at 32 heads.
    static let headDimConstant = 128

    /// Load the resampler from a sub-bundle prefixed at `resampler.`.
    /// `embedDim` is the text hidden dim, `visionHidden` the SigLIP hidden
    /// dim. `numQueries` is read from the shipped `query` parameter (which
    /// equals `query_num`).
    static func load(
        weights: SafeTensorsBundle,
        embedDim: Int, visionHidden: Int,
        quantization: ModelConfig.QuantizationConfig?,
        eps: Float, device: Device
    ) throws -> MiniCPMOResampler {
        let query = try weights.tensor(named: "query")
        precondition(
            query.shape.count == 2 && query.shape[1] == embedDim,
            "MiniCPMOResampler: query shape \(query.shape) ≠ "
                + "[numQueries, \(embedDim)]")
        let numQueries = query.shape[0]
        let proj = try weights.tensor(named: "proj")
        precondition(
            proj.shape == [embedDim, embedDim],
            "MiniCPMOResampler: proj shape \(proj.shape) ≠ "
                + "[\(embedDim), \(embedDim)]")

        let kvProj = try loadLinear(
            base: "kv_proj", in: weights, quantization: quantization)
        let qProj = try loadLinear(
            base: "attn.q_proj", in: weights, quantization: quantization)
        let kProj = try loadLinear(
            base: "attn.k_proj", in: weights, quantization: quantization)
        let vProj = try loadLinear(
            base: "attn.v_proj", in: weights, quantization: quantization)
        let outProj = try loadLinear(
            base: "attn.out_proj", in: weights, quantization: quantization)

        let lnQ = LayerNorm(
            weight: try weights.tensor(named: "ln_q.weight"),
            bias: try weights.tensor(named: "ln_q.bias"), eps: eps)
        let lnKV = LayerNorm(
            weight: try weights.tensor(named: "ln_kv.weight"),
            bias: try weights.tensor(named: "ln_kv.bias"), eps: eps)
        let lnPost = LayerNorm(
            weight: try weights.tensor(named: "ln_post.weight"),
            bias: try weights.tensor(named: "ln_post.bias"), eps: eps)

        // The MHA head count is `embed_dim // 128` (MiniCPM-V/-o
        // resamplers; see `init_resampler` in the reference).
        let nHeads = embedDim / headDimConstant
        precondition(
            nHeads * headDimConstant == embedDim,
            "MiniCPMOResampler: embedDim \(embedDim) not a multiple of "
                + "\(headDimConstant)")

        return MiniCPMOResampler(
            query: query, kvProj: kvProj,
            qProj: qProj, kProj: kProj, vProj: vProj, outProj: outProj,
            lnQ: lnQ, lnKV: lnKV, lnPost: lnPost,
            proj: proj, numQueries: numQueries, embedDim: embedDim,
            visionHidden: visionHidden, nHeads: nHeads, dtype: query.dtype)
    }

    /// Resample a single image's `[numPatches, visionHidden]` SigLIP grid
    /// (row-major over the `gridH × gridW` patch grid, `gridH·gridW ==
    /// numPatches`) into `[numQueries, embedDim]` text-hidden soft tokens.
    func forward(
        patches: Tensor, gridH: Int, gridW: Int,
        device: Device
    ) -> Tensor {
        let numPatches = gridH * gridW
        precondition(
            patches.shape == [numPatches, visionHidden],
            "MiniCPMOResampler.forward: patches \(patches.shape) ≠ "
                + "[\(numPatches), \(visionHidden)]")

        // ── 1. kv_proj + ln_kv over the patches → [numPatches, embedDim] ─
        let cmd = device.makeCommandBuffer()
        let projected = kvProj.callMany(patches, t: numPatches, on: cmd, device: device)
        let kv = Ops.layerNorm(
            projected, weight: lnKV.weight, bias: lnKV.bias,
            eps: lnKV.eps, nRows: numPatches, rowSize: embedDim, on: cmd)

        // ── 2. ln_q over the learned queries → [numQueries, embedDim] ────
        let qNorm = Ops.layerNorm(
            query, weight: lnQ.weight, bias: lnQ.bias,
            eps: lnQ.eps, nRows: numQueries, rowSize: embedDim, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── 3. Additive 2D sin-cos position embedding on the KEY only ────
        // Reference: `key = x + pos_embed`, `value = x` (pos added only to
        // the attention key path). Generated at runtime for the actual
        // patch grid, dtype-matched, then added to the projected patches.
        let posEmb = MiniCPMOResampler.sincos2DPositionEmbedding(
            gridH: gridH, gridW: gridW, embedDim: embedDim,
            dtype: dtype, device: device)
        let cmdPos = device.makeCommandBuffer()
        let keyInput = Ops.add(kv, posEmb, on: cmdPos)
        cmdPos.commit()
        cmdPos.waitUntilCompleted()

        // ── 4. MHA: q_proj(qNorm), k_proj(key), v_proj(kv) ───────────────
        let cmd2 = device.makeCommandBuffer()
        let q = qProj.callMany(qNorm, t: numQueries, on: cmd2, device: device)
        let k = kProj.callMany(keyInput, t: numPatches, on: cmd2, device: device)
        let v = vProj.callMany(kv, t: numPatches, on: cmd2, device: device)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        // Scaled-dot-product attention (64 queries × numPatches keys).
        let attn = crossAttention(
            q: q, k: k, v: v, nQuery: numQueries, nKV: numPatches,
            device: device)

        // ── 5. out_proj → ln_post → @ proj → [numQueries, embedDim] ──────
        let cmd3 = device.makeCommandBuffer()
        let attnProj = outProj.callMany(attn, t: numQueries, on: cmd3, device: device)
        let post = Ops.layerNorm(
            attnProj, weight: lnPost.weight, bias: lnPost.bias,
            eps: lnPost.eps, nRows: numQueries, rowSize: embedDim, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()

        // `out @ proj` — `proj` is `[embedDim, embedDim]`. `Ops.gemm`
        // computes `out[r] = weight · input[r]` (i.e. `input @ weightᵀ`),
        // so to realise the reference `post @ proj` we feed `projᵀ` as the
        // weight. The CPU transpose is one-time and tiny (4096²).
        let projT = MiniCPMOResampler.transpose2D(
            proj, rows: embedDim, cols: embedDim, dtype: dtype, device: device)
        let cmd4 = device.makeCommandBuffer()
        let result = Ops.gemm(
            weight: projT, input: post, nRows: numQueries, on: cmd4)
        cmd4.commit()
        cmd4.waitUntilCompleted()
        return result
    }

    /// CPU scaled-dot-product multi-head cross attention. `q` is
    /// query-major `[nQuery, embedDim]`, `k`/`v` are key-major
    /// `[nKV, embedDim]`. No mask. Returns the context query-major
    /// `[nQuery, embedDim]`, ready for `out_proj`.
    ///
    /// Fans the `(head, query-row)` index space across cores with
    /// `concurrentPerform`; each iteration writes a disjoint
    /// `[oBase, oBase + headDim)` slice (race-free). Mirrors the
    /// `VisionEncoderLayer` / `AudioEncoderLayer` CPU cores.
    private func crossAttention(
        q: Tensor, k: Tensor, v: Tensor,
        nQuery: Int, nKV: Int, device: Device
    ) -> Tensor {
        let qa = q.toFloatArray()
        let ka = k.toFloatArray()
        let va = v.toFloatArray()
        let stride = nHeads * headDim
        var out = [Float](repeating: 0, count: nQuery * stride)

        let nHeadsLocal = nHeads
        let headDimLocal = headDim
        let scaleLocal = scale
        out.withUnsafeMutableBufferPointer { outBuf in
            // Each iteration owns a distinct `(head, query)` output slice;
            // the pointers outlive the parallel region.
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            qa.withUnsafeBufferPointer { qPtr in
                ka.withUnsafeBufferPointer { kPtr in
                    va.withUnsafeBufferPointer { vPtr in
                        nonisolated(unsafe) let qb = qPtr.baseAddress!
                        nonisolated(unsafe) let kb = kPtr.baseAddress!
                        nonisolated(unsafe) let vb = vPtr.baseAddress!
                        DispatchQueue.concurrentPerform(
                            iterations: nHeadsLocal * nQuery
                        ) { work in
                            let head = work / nQuery
                            let i = work % nQuery
                            let hOff = head * headDimLocal
                            var scores = [Float](repeating: 0, count: nKV)
                            var maxScore = -Float.greatestFiniteMagnitude
                            let qBase = i * stride + hOff
                            for j in 0 ..< nKV {
                                var dot: Float = 0
                                let kBase = j * stride + hOff
                                for d in 0 ..< headDimLocal {
                                    dot += qb[qBase + d] * kb[kBase + d]
                                }
                                let s = dot * scaleLocal
                                scores[j] = s
                                if s > maxScore { maxScore = s }
                            }
                            var sumExp: Float = 0
                            for j in 0 ..< nKV {
                                let e = exp(scores[j] - maxScore)
                                scores[j] = e
                                sumExp += e
                            }
                            let inv = sumExp > 0 ? 1 / sumExp : 0
                            let oBase = i * stride + hOff
                            for j in 0 ..< nKV {
                                let w = scores[j] * inv
                                let vBase = j * stride + hOff
                                for d in 0 ..< headDimLocal {
                                    outPtr[oBase + d] += w * vb[vBase + d]
                                }
                            }
                        }
                    }
                }
            }
        }
        let result = Tensor.empty(
            shape: [nQuery, stride], dtype: q.dtype, device: device)
        MediaUtils.copyFloats(out, into: result)
        return result
    }

    /// CPU transpose of a 2D `[rows, cols]` tensor to `[cols, rows]`.
    static func transpose2D(
        _ x: Tensor, rows: Int, cols: Int,
        dtype: DType, device: Device
    ) -> Tensor {
        let src = x.toFloatArray()
        var dst = [Float](repeating: 0, count: rows * cols)
        for r in 0 ..< rows {
            for c in 0 ..< cols { dst[c * rows + r] = src[r * cols + c] }
        }
        return MediaUtils.makeTensor(
            from: dst, shape: [cols, rows], dtype: dtype, device: device)
    }

    // ─── 2D sin-cos position embedding ───────────────────────────────
    //
    // Standard `get_2d_sincos_pos_embed(embed_dim, (H, W))`: the embedding
    // dim is split in half — the first half encodes the H-grid index, the
    // second half the W-grid index — each via interleaved sin/cos of
    // `position × frequency`. Reference: `get_2d_sincos_pos_embed` +
    // `get_2d_sincos_pos_embed_from_grid` in `modeling_minicpmo.py`.

    /// Build the `[H·W, embedDim]` additive 2D sin-cos position embedding
    /// for a `gridH × gridW` patch grid, row-major over the grid (the
    /// patch order the SigLIP tower emits). `embedDim` must be divisible
    /// by 4 (split into two halves, each split into sin/cos).
    static func sincos2DPositionEmbedding(
        gridH: Int, gridW: Int, embedDim: Int,
        dtype: DType, device: Device
    ) -> Tensor {
        precondition(
            embedDim % 4 == 0,
            "MiniCPMOResampler.sincos2D: embedDim \(embedDim) must be "
                + "divisible by 4")
        let halfDim = embedDim / 2
        // Per the reference, each axis embedding is built over the grid
        // with `get_1d_sincos_pos_embed_from_grid_new(embed_dim/2, grid)`,
        // where the H grid varies down rows and the W grid across cols.
        // For grid position (h, w): first `halfDim` channels encode `h`,
        // next `halfDim` channels encode `w`.
        let hEmb = sincos1D(positions: gridH, dim: halfDim)  // [gridH, halfDim]
        let wEmb = sincos1D(positions: gridW, dim: halfDim)  // [gridW, halfDim]

        var dst = [Float](repeating: 0, count: gridH * gridW * embedDim)
        for h in 0 ..< gridH {
            for w in 0 ..< gridW {
                let row = (h * gridW + w) * embedDim
                for c in 0 ..< halfDim {
                    dst[row + c] = hEmb[h * halfDim + c]
                    dst[row + halfDim + c] = wEmb[w * halfDim + c]
                }
            }
        }
        return MediaUtils.makeTensor(
            from: dst, shape: [gridH * gridW, embedDim],
            dtype: dtype, device: device)
    }

    /// 1D sin-cos embedding: for each of `positions` indices, a `dim`-wide
    /// vector whose first `dim/2` channels are `sin(pos·omega_k)` and last
    /// `dim/2` channels are `cos(pos·omega_k)`, with
    /// `omega_k = 1 / 10000^(2k/dim)`. Matches
    /// `get_1d_sincos_pos_embed_from_grid_new`: `omega = arange(dim/2) /
    /// (dim/2)`, `omega = 1 / 10000**omega`, `out = pos ⊗ omega`, then
    /// `concat(sin(out), cos(out))`. Returns a flat `[positions·dim]` array.
    private static func sincos1D(positions: Int, dim: Int) -> [Float] {
        precondition(dim % 2 == 0, "sincos1D: dim \(dim) must be even")
        let half = dim / 2
        var omega = [Float](repeating: 0, count: half)
        for k in 0 ..< half {
            // omega = arange(half) / half; omega = 1 / 10000**omega.
            let exponent = Float(k) / Float(half)
            omega[k] = 1.0 / powf(10_000.0, exponent)
        }
        var out = [Float](repeating: 0, count: positions * dim)
        for p in 0 ..< positions {
            let base = p * dim
            for k in 0 ..< half {
                let v = Float(p) * omega[k]
                out[base + k] = sin(v)
                out[base + half + k] = cos(v)
            }
        }
        return out
    }
}
