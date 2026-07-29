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
// VisionEncoderLayer — one pre-norm ViT transformer encoder block, the
// reusable building block the shared `VisionEncoder` tower stacks. Kept
// here in `ModelLayers/` alongside the other shared layer primitives; the
// encoder stack itself (`VisionEncoder`, config + error) lives in
// `Vision/VisionEncoder.swift`.

import Foundation
import Metal

// ─── Encoder block ───────────────────────────────────────────────────

/// One pre-norm ViT transformer block: LayerNorm → bidirectional MHA →
/// residual, then LayerNorm → GELU MLP → residual.
public final class VisionEncoderLayer: Module {
    let layerNorm1: LayerNorm
    let qProj, kProj, vProj, oProj: Linear
    let layerNorm2: LayerNorm
    let fc1, fc2: Linear

    let hidden, nHeads, headDim, intermediate: Int
    let scale: Float
    /// Use the exact erf-based GELU (SigLIP2 `nn.GELU(approx="precise")`)
    /// instead of the tanh approximation (SigLIP v1 / CLIP `gelu_pytorch_tanh`).
    let useExactGelu: Bool

    init(
        layerNorm1: LayerNorm,
        qProj: Linear, kProj: Linear, vProj: Linear, oProj: Linear,
        layerNorm2: LayerNorm, fc1: Linear, fc2: Linear,
        hidden: Int, nHeads: Int, intermediate: Int,
        useExactGelu: Bool = false
    ) {
        self.useExactGelu = useExactGelu
        self.layerNorm1 = layerNorm1
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.layerNorm2 = layerNorm2
        self.fc1 = fc1
        self.fc2 = fc2
        self.hidden = hidden
        self.nHeads = nHeads
        self.headDim = hidden / nHeads
        self.intermediate = intermediate
        self.scale = 1.0 / Float(Double(hidden / nHeads).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in layerNorm1.parameters() { out.append(("layer_norm1.\(k)", v)) }
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.out_proj.\(k)", v)) }
        for (k, v) in layerNorm2.parameters() { out.append(("layer_norm2.\(k)", v)) }
        for (k, v) in fc1.parameters() { out.append(("mlp.fc1.\(k)", v)) }
        for (k, v) in fc2.parameters() { out.append(("mlp.fc2.\(k)", v)) }
        return out
    }

    /// Forward `[nTokens, hidden]` patch-token activations through one
    /// encoder block. The GEMM-heavy projections + norms run on the GPU
    /// queued on `cmd`; the bidirectional attention core then dispatches
    /// to a head-dim-matched GPU kernel where one exists, falling back
    /// to a parallel CPU pass only for head_dim values that have no
    /// kernel yet.
    ///
    /// Available kernels (Butter/Ops.swift):
    ///   - head_dim ∈ {32, 64, 72, 80, 96} → `Ops.sdpaBidirectional`
    ///     (multi-query, non-causal)
    ///   - head_dim == 128 → `Ops.sdpaMulti(causal: false)`
    ///     (multi-query, hardcoded d=128)
    ///   - anything else   → CPU `cpuAttention` (parallel by
    ///                       (head, query-row), see below)
    ///
    /// At SigLIP-So400m's 4096 patches × 27 layers × 16 heads × O(n²)
    /// the CPU path used to take 25+ minutes per encoder pass; the GPU
    /// path collapses that into the per-layer GEMM cost.
    func forward(
        _ h: Tensor, nTokens: Int, device: Device,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        // ── Attention sub-block ──
        let normed = Ops.layerNorm(
            h, weight: layerNorm1.weight,
            bias: layerNorm1.bias, eps: layerNorm1.eps,
            nRows: nTokens, rowSize: hidden, on: cmd)
        // Q/K/V projections over every token, one multi-row GEMM each.
        let q = projectRows(qProj, normed, nTokens: nTokens, on: cmd)
        let k = projectRows(kProj, normed, nTokens: nTokens, on: cmd)
        let v = projectRows(vProj, normed, nTokens: nTokens, on: cmd)

        // Bidirectional multi-head attention. GPU path when the kernel
        // is known-safe at this head_dim; CPU fallback otherwise.
        //
        // GPU vs CPU bidirectional attention is gated on
        // `OpsValidation.sdpaBidirectionalIsProductionSafe` — the single
        // source of truth for which head_dims have a passing iron
        // production-shape GPU-correctness cell (16 heads × 576-patch tower,
        // 9216 threadgroups). The 2026-05-27 GlmOcr "pin" at d64 was a
        // wrong-dispatch bug (since fixed in `Ops.sdpaBidirectional`); d64,
        // d72, d80, d96 are now all verified correct + unpinned at
        // production shape (see `sdpa_bidirectional.rs`
        // `*_vision_tower` / `*_audio_encoder` cells). d32 (FastViT-HD) has
        // a kernel but no production cell yet, so it stays on CPU.
        let attnFlat: Tensor
        if OpsValidation.sdpaBidirectionalIsProductionSafe(headDim: headDim) {
            attnFlat = gpuAttention(
                q: q, k: k, v: v, nTokens: nTokens,
                device: device, on: cmd)
        } else if headDim == 128 {
            // Pixtral, Mistral3 Pixtral-based, GlmOcr d=128 variants.
            // `sdpaMulti(causal: false)` hasn't been observed to pin
            // at vision-tower shapes but also hasn't been independently
            // verified — leave it on the GPU path for now; revisit if
            // a Pixtral integration test pins the GPU.
            attnFlat = gpuAttentionMulti(
                q: q, k: k, v: v, nTokens: nTokens,
                device: device, on: cmd)
        } else {
            // d ∈ {32, 64, 80, 96} → CPU fallback under the GUARD.
            // d ∉ supported → CPU per the original contract.
            // Flush the projection GEMMs so their results are CPU-
            // readable for the fallback attention core.
            cmd.commit()
            cmd.waitUntilCompleted()
            attnFlat = cpuAttention(
                q: q, k: k, v: v, nTokens: nTokens,
                device: device)
        }

        // ── Residual + MLP sub-block ──
        let cmd2 = device.makeCommandBuffer()
        let attnProj = projectRows(oProj, attnFlat, nTokens: nTokens, on: cmd2)
        let postAttn = Ops.add(h, attnProj, on: cmd2)
        let normed2 = Ops.layerNorm(
            postAttn, weight: layerNorm2.weight,
            bias: layerNorm2.bias, eps: layerNorm2.eps,
            nRows: nTokens, rowSize: hidden, on: cmd2)
        let ff1 = projectRows(
            fc1, normed2, nTokens: nTokens,
            outDim: intermediate, on: cmd2)
        let act = useExactGelu ? Ops.geluErf(ff1, on: cmd2) : Ops.gelu(ff1, on: cmd2)
        let ff2 = projectRows(
            fc2, act, nTokens: nTokens,
            inDim: intermediate, outDim: hidden, on: cmd2)
        let result = Ops.add(postAttn, ff2, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return result
    }

    /// GPU bidirectional multi-head attention via
    /// `Ops.sdpaBidirectional(headDim:)`. head_dim must be one of
    /// {32, 64, 72} — the kernel surface Butter ships today.
    ///
    /// Layout note: `Ops.sdpaBidirectional` takes
    ///   Q at `[nQuery, nQHeads, headDim]` (token-major, head-second)
    ///   K/V at `[nKVHeads, kvStride, headDim]` (head-major).
    ///
    /// The caller hands us Q/K/V as `[nTokens, nHeads*headDim]`. Q's
    /// memory layout is byte-identical to the kernel's
    /// `[nQuery, nQHeads, headDim]` (same row-major tensor with a
    /// different shape annotation). K/V need a `[nTokens, nHeads,
    /// headDim]` → `[nHeads, nTokens, headDim]` transpose; we do it
    /// CPU-side (one readback + copy) for the same reason the
    /// Paligemma / SmolVLM2 migrations do — there is no
    /// `Ops.transpose` wrapper. The transpose cost is dwarfed by the
    /// O(n²·d) attention kernel, and we get a fully GPU-resident
    /// softmax / weighted-V loop in exchange.
    private func gpuAttention(
        q: Tensor, k: Tensor, v: Tensor,
        nTokens: Int, device: Device,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        // Flush Q/K/V projections so we can read K/V back for the
        // CPU-side head-major transpose.
        cmd.commit()
        cmd.waitUntilCompleted()

        let kHeadMajor = transposeTokenToHeadMajor(
            k, nTokens: nTokens, device: device)
        let vHeadMajor = transposeTokenToHeadMajor(
            v, nTokens: nTokens, device: device)
        // Q in its current `[nTokens, nHeads*headDim]` layout is
        // byte-identical to `[nQuery, nQHeads, headDim]` — no copy.
        let qReshaped = q.reshaped(to: [nTokens, nHeads, headDim])

        let cmd2 = device.makeCommandBuffer()
        let attn = Ops.sdpaBidirectional(
            q: qReshaped, k: kHeadMajor, v: vHeadMajor,
            nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
            baseKV: 0, nQuery: nTokens, kvStride: nTokens,
            scale: scale, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        // Kernel output is `[nQuery, nQHeads, headDim]` — byte-identical
        // to the `[nTokens, nHeads*headDim]` flat layout o_proj expects.
        return attn.reshaped(to: [nTokens, nHeads * headDim])
    }

    /// GPU bidirectional MHA dispatched via `Ops.sdpaMulti(causal:false)`
    /// — the head_dim=128 variant. Same layout contract as
    /// `gpuAttention`. Used when this tower's head_dim is 128.
    private func gpuAttentionMulti(
        q: Tensor, k: Tensor, v: Tensor,
        nTokens: Int, device: Device,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        cmd.commit()
        cmd.waitUntilCompleted()

        let kHeadMajor = transposeTokenToHeadMajor(
            k, nTokens: nTokens, device: device)
        let vHeadMajor = transposeTokenToHeadMajor(
            v, nTokens: nTokens, device: device)
        let qReshaped = q.reshaped(to: [nTokens, nHeads, headDim])

        let cmd2 = device.makeCommandBuffer()
        let attn = Ops.sdpaMulti(
            q: qReshaped, k: kHeadMajor, v: vHeadMajor,
            nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
            baseKV: 0, nQuery: nTokens, kvStride: nTokens,
            causal: false, scale: scale, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return attn.reshaped(to: [nTokens, nHeads * headDim])
    }

    /// `[nTokens, nHeads*headDim]` (token-major) →
    /// `[nHeads, nTokens, headDim]` (head-major) via a CPU pass.
    /// Used to repack K/V into the layout the sdpa kernels expect.
    private func transposeTokenToHeadMajor(
        _ t: Tensor, nTokens: Int, device: Device
    ) -> Tensor {
        let src = t.toFloatArray()
        var dst = [Float](repeating: 0, count: nTokens * nHeads * headDim)
        let stride = nHeads * headDim
        // src[tok, h, d] = src[tok*stride + h*headDim + d]
        // dst[h, tok, d] = dst[h*nTokens*headDim + tok*headDim + d]
        for tok in 0 ..< nTokens {
            let srcRow = tok * stride
            for h in 0 ..< nHeads {
                let srcH = srcRow + h * headDim
                let dstH = h * nTokens * headDim + tok * headDim
                for d in 0 ..< headDim { dst[dstH + d] = src[srcH + d] }
            }
        }
        let out = Tensor.empty(
            shape: [nHeads, nTokens, headDim],
            dtype: t.dtype, device: device)
        MediaUtils.copyFloats(dst, into: out)
        return out
    }

    /// CPU bidirectional multi-head attention over `nTokens` patch
    /// tokens. `q` / `k` / `v` are token-major `[nTokens, nHeads*headDim]`.
    /// Returns the context, token-major `[nTokens, nHeads*headDim]`,
    /// ready for the output projection.
    ///
    /// The work is O(`nHeads · nTokens² · headDim`). SigLIP-896 vision
    /// towers produce `nTokens` = 4096 (64×64 patch grid), so a naive
    /// single-threaded scalar pass is ~10¹² float ops per encoder forward
    /// — minutes of wall time. Each `(head, query-row)` pair writes a
    /// disjoint `headDim`-wide slice of `out`, so the outer
    /// `nHeads · nTokens` index space is embarrassingly parallel: it is
    /// fanned out across cores with `concurrentPerform`, which is
    /// race-free precisely because the write targets never overlap.
    private func cpuAttention(
        q: Tensor, k: Tensor, v: Tensor,
        nTokens: Int, device: Device
    ) -> Tensor {
        let qa = q.toFloatArray()
        let ka = k.toFloatArray()
        let va = v.toFloatArray()
        let stride = nHeads * headDim
        var out = [Float](repeating: 0, count: nTokens * stride)

        // Fan the (head, query-row) index space across cores. `out` is
        // mutated through an unsafe pointer because each iteration owns a
        // disjoint `[oBase, oBase+headDim)` slice — no two iterations
        // touch the same element, so the writes need no synchronization.
        let nHeadsLocal = nHeads
        let headDimLocal = headDim
        let scaleLocal = scale
        out.withUnsafeMutableBufferPointer { outBuf in
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            qa.withUnsafeBufferPointer { qPtr in
                ka.withUnsafeBufferPointer { kPtr in
                    va.withUnsafeBufferPointer { vPtr in
                        nonisolated(unsafe) let qb = qPtr.baseAddress!
                        nonisolated(unsafe) let kb = kPtr.baseAddress!
                        nonisolated(unsafe) let vb = vPtr.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: nHeadsLocal * nTokens) { work in
                            let head = work / nTokens
                            let i = work % nTokens
                            let hOff = head * headDimLocal
                            // Scaled dot-product scores against every token.
                            var scores = [Float](repeating: 0, count: nTokens)
                            var maxScore = -Float.greatestFiniteMagnitude
                            let qBase = i * stride + hOff
                            for j in 0 ..< nTokens {
                                var dot: Float = 0
                                let kBase = j * stride + hOff
                                for d in 0 ..< headDimLocal {
                                    dot += qb[qBase + d] * kb[kBase + d]
                                }
                                let s = dot * scaleLocal
                                scores[j] = s
                                if s > maxScore { maxScore = s }
                            }
                            // Softmax (numerically stable).
                            var sumExp: Float = 0
                            for j in 0 ..< nTokens {
                                let e = exp(scores[j] - maxScore)
                                scores[j] = e
                                sumExp += e
                            }
                            let inv = sumExp > 0 ? 1 / sumExp : 0
                            // Weighted sum of V into this row's disjoint slice.
                            let oBase = i * stride + hOff
                            for j in 0 ..< nTokens {
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
            shape: [nTokens, stride], dtype: q.dtype,
            device: device)
        MediaUtils.copyFloats(out, into: result)
        return result
    }

    /// Apply a `Linear` to every row of a `[nTokens, *]` tensor via
    /// `Ops.gemm` (one tiled multi-row GEMM), then broadcast-add the
    /// bias to every row. The vision Linears always carry a bias.
    /// `outDim` defaults to the block hidden; pass it for the MLP's
    /// `hidden → intermediate → hidden` shape changes.
    private func projectRows(
        _ linear: Linear, _ x: Tensor, nTokens: Int,
        inDim: Int? = nil, outDim: Int? = nil,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        _ = inDim  // kept for call-site documentation symmetry with outDim
        let outD = outDim ?? hidden
        let y = Ops.gemm(weight: linear.weight, input: x, nRows: nTokens, on: cmd)
        guard let bias = linear.bias else { return y }
        return addRowBias(y, bias: bias, nRows: nTokens, rowSize: outD, on: cmd)
    }

    /// Add a `[rowSize]` bias to each of `nRows` rows of a flat
    /// `[nRows, rowSize]` tensor. `Ops.add` is element-wise same-shape,
    /// so the bias is tiled into a full `[nRows*rowSize]` buffer once.
    private func addRowBias(
        _ x: Tensor, bias: Tensor, nRows: Int,
        rowSize: Int, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let tiled = Tensor.empty(shape: [nRows, rowSize], dtype: x.dtype)
        let biasVals = bias.toFloatArray()
        var flat = [Float](repeating: 0, count: nRows * rowSize)
        for r in 0 ..< nRows {
            for c in 0 ..< rowSize { flat[r * rowSize + c] = biasVals[c] }
        }
        MediaUtils.copyFloats(flat, into: tiled)
        return Ops.add(x, tiled, on: cmd)
    }
}
