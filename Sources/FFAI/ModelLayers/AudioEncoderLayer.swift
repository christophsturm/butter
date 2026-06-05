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
// AudioEncoderLayer — one pre-norm Whisper-style transformer encoder
// block, the reusable building block the shared `AudioEncoder` stack
// repeats. Kept here in `ModelLayers/` alongside the other shared layer
// primitives; the encoder stack itself (`AudioEncoder`, its config + conv
// stem + `addRowBias` helper) lives in `Audio/AudioEncoder.swift`.

import Foundation
import Metal

// ─── Encoder block ───────────────────────────────────────────────────

/// One pre-norm Whisper encoder block: LayerNorm → bidirectional MHA →
/// residual, then LayerNorm → GELU MLP → residual.
public final class AudioEncoderLayer: Module {
    let layerNorm1: LayerNorm
    let qProj, kProj, vProj, oProj: Linear
    let layerNorm2: LayerNorm
    let fc1, fc2: Linear

    let hidden, nHeads, headDim, intermediate: Int
    let scale: Float

    init(
        layerNorm1: LayerNorm,
        qProj: Linear, kProj: Linear, vProj: Linear, oProj: Linear,
        layerNorm2: LayerNorm, fc1: Linear, fc2: Linear,
        hidden: Int, nHeads: Int, intermediate: Int
    ) {
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
        for (k, v) in layerNorm1.parameters() { out.append(("self_attn_layer_norm.\(k)", v)) }
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.out_proj.\(k)", v)) }
        for (k, v) in layerNorm2.parameters() { out.append(("final_layer_norm.\(k)", v)) }
        for (k, v) in fc1.parameters() { out.append(("fc1.\(k)", v)) }
        for (k, v) in fc2.parameters() { out.append(("fc2.\(k)", v)) }
        return out
    }

    /// Forward `[nFrames, hidden]` frame activations through one encoder
    /// block. Fully GPU-resident when the head dim is one the
    /// `sdpaBidirectional` kernel supports (Whisper-family encoders use
    /// 64 / 80 / 96, all supported): the bidirectional attention core runs
    /// on the GPU on the same command buffer as the projection GEMMs (no
    /// CPU readback). Unsupported head dims fall back to the parallel CPU
    /// attention (still correct).
    func forward(
        _ h: Tensor, nFrames: Int, device: Device,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        // ── Attention sub-block ──
        let normed = Ops.layerNorm(
            h, weight: layerNorm1.weight,
            bias: layerNorm1.bias, eps: layerNorm1.eps,
            nRows: nFrames, rowSize: hidden, on: cmd)
        let q = projectRows(qProj, normed, nRows: nFrames, on: cmd)
        let k = projectRows(kProj, normed, nRows: nFrames, on: cmd)
        let v = projectRows(vProj, normed, nRows: nFrames, on: cmd)

        // Audio self-attention is plain bidirectional MHA (no RoPE, no
        // mask), so it maps straight onto the GPU `sdpaBidirectional`
        // kernel — q stays frame-major, k/v transpose to head-major. The
        // MLP sub-block then continues on the SAME command buffer, so the
        // whole block is one GPU submission for supported head dims.
        let attnFlat: Tensor
        let mlpCmd: MTLCommandBuffer
        if OpsValidation.sdpaBidirectionalIsProductionSafe(headDim: headDim) {
            let kHead = Ops.transposeTokensToHeads(
                k, nTokens: nFrames, nHeads: nHeads, headDim: headDim, on: cmd)
            let vHead = Ops.transposeTokensToHeads(
                v, nTokens: nFrames, nHeads: nHeads, headDim: headDim, on: cmd)
            attnFlat = Ops.sdpaBidirectional(
                q: q, k: kHead, v: vHead,
                nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
                baseKV: 0, nQuery: nFrames, kvStride: nFrames,
                scale: scale, on: cmd
            ).reshaped(to: [nFrames, nHeads * headDim])
            mlpCmd = cmd
        } else {
            // Flush the projection GEMMs so their results are CPU-readable.
            cmd.commit()
            cmd.waitUntilCompleted()
            attnFlat = cpuAttention(
                q: q, k: k, v: v, nFrames: nFrames, device: device)
            mlpCmd = device.makeCommandBuffer()
        }

        // ── Residual + MLP sub-block ──
        let attnProj = projectRows(oProj, attnFlat, nRows: nFrames, on: mlpCmd)
        let postAttn = Ops.add(h, attnProj, on: mlpCmd)
        let normed2 = Ops.layerNorm(
            postAttn, weight: layerNorm2.weight,
            bias: layerNorm2.bias, eps: layerNorm2.eps,
            nRows: nFrames, rowSize: hidden, on: mlpCmd)
        let ff1 = projectRows(
            fc1, normed2, nRows: nFrames,
            outDim: intermediate, on: mlpCmd)
        let act = Ops.gelu(ff1, on: mlpCmd)
        let ff2 = projectRows(
            fc2, act, nRows: nFrames, outDim: hidden,
            on: mlpCmd)
        let result = Ops.add(postAttn, ff2, on: mlpCmd)
        mlpCmd.commit()
        mlpCmd.waitUntilCompleted()
        return result
    }

    /// CPU bidirectional multi-head attention over `nFrames` audio
    /// frames. `q` / `k` / `v` are frame-major `[nFrames, nHeads*headDim]`.
    /// Returns the context, frame-major, ready for the output projection.
    ///
    /// Fans the `(head, query-row)` index space across CPU cores with
    /// `DispatchQueue.concurrentPerform`. Mirrors the parallelization of
    /// `VisionEncoderLayer.cpuAttention` — at Whisper's nAudioCtx = 1500
    /// and 6 heads the prior single-threaded loop took 15-18 minutes per
    /// encoder pass, which surfaced as Whisper `transcribe` timing out.
    /// Race-free because each iteration writes to a disjoint
    /// `[oBase, oBase + headDim)` output slice — no two iterations touch
    /// the same element, so the writes need no synchronization.
    private func cpuAttention(
        q: Tensor, k: Tensor, v: Tensor,
        nFrames: Int, device: Device
    ) -> Tensor {
        let qa = q.toFloatArray()
        let ka = k.toFloatArray()
        let va = v.toFloatArray()
        let stride = nHeads * headDim
        var out = [Float](repeating: 0, count: nFrames * stride)

        let nHeadsLocal = nHeads
        let headDimLocal = headDim
        let scaleLocal = scale
        out.withUnsafeMutableBufferPointer { outBuf in
            // Pointers / handles below cross a `@Sendable` boundary. Safety:
            // each `concurrentPerform` iteration writes to a distinct `(head, i)`
            // pair of output rows, and the pointers' lifetime strictly contains
            // the parallel work.
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            qa.withUnsafeBufferPointer { qPtr in
                ka.withUnsafeBufferPointer { kPtr in
                    va.withUnsafeBufferPointer { vPtr in
                        nonisolated(unsafe) let qb = qPtr.baseAddress!
                        nonisolated(unsafe) let kb = kPtr.baseAddress!
                        nonisolated(unsafe) let vb = vPtr.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: nHeadsLocal * nFrames) { work in
                            let head = work / nFrames
                            let i = work % nFrames
                            let hOff = head * headDimLocal
                            var scores = [Float](repeating: 0, count: nFrames)
                            var maxScore = -Float.greatestFiniteMagnitude
                            let qBase = i * stride + hOff
                            for j in 0 ..< nFrames {
                                var dot: Float = 0
                                let kBase = j * stride + hOff
                                for d in 0 ..< headDimLocal { dot += qb[qBase + d] * kb[kBase + d] }
                                let s = dot * scaleLocal
                                scores[j] = s
                                if s > maxScore { maxScore = s }
                            }
                            var sumExp: Float = 0
                            for j in 0 ..< nFrames {
                                let e = exp(scores[j] - maxScore)
                                scores[j] = e
                                sumExp += e
                            }
                            let inv = sumExp > 0 ? 1 / sumExp : 0
                            let oBase = i * stride + hOff
                            for j in 0 ..< nFrames {
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
            shape: [nFrames, stride], dtype: q.dtype,
            device: device)
        MediaUtils.copyFloats(out, into: result)
        return result
    }

    /// Apply a `Linear` to every row of a `[nRows, *]` tensor via
    /// `Ops.gemm`, then broadcast-add the bias. Whisper Linears all
    /// carry a bias except `k_proj` (which has none — handled by the
    /// optional `Linear.bias`).
    private func projectRows(
        _ linear: Linear, _ x: Tensor, nRows: Int,
        outDim: Int? = nil,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        let outD = outDim ?? linear.weight.shape[0]
        let y = Ops.gemm(weight: linear.weight, input: x, nRows: nRows, on: cmd)
        guard let bias = linear.bias else { return y }
        return AudioEncoder.addRowBias(
            y, bias: bias, nRows: nRows,
            rowSize: outD, on: cmd)
    }
}
