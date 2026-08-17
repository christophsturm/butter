// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// Ops — ergonomic Tensor-based dispatch over IronKernels.
//
// Each op picks the right kernel for the input dtype, fills in default
// grid/threadgroup sizing, encodes on the supplied command buffer, and
// returns a fresh output Tensor (or writes into a caller-supplied one).
//
// Initial cut: only the kernels Llama needs. Adding more in follow-ups.

import Foundation
import Metal
import IronSwift

public enum Ops {
    public static let device: Device = .shared

    // ─── Sizing helpers ──────────────────────────────────────────────

    /// Threadgroup width for elementwise kernels. Matches what we know
    /// PSO maxTotalThreadsPerThreadgroup will accept on M-series.
    public static let elementwiseTgSize = 256

    /// Internal because the per-Ops extension files (`OpsMath.swift`,
    /// `OpsLogits.swift`, `OpsFused.swift`) call it.
    static func elementwiseGrid(_ n: Int) -> (MTLSize, MTLSize) {
        let tg = MTLSize(width: min(elementwiseTgSize, n), height: 1, depth: 1)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        return (grid, tg)
    }

    // ─── GPU copy (blit, no kernel dispatch) ─────────────────────────

    /// GPU-side copy of `src` into `dst`. Encoded as an `MTLBlit`
    /// command on `cmd` — no compute kernel, no PSO dispatch overhead.
    /// Same dtype, same elementCount; layouts must match.
    ///
    /// Use when the caller already has `dst` allocated at a specific
    /// row inside a larger contiguous buffer and needs to deposit the
    /// op-produced `src` there (e.g. per-row GDN-layer write-back in
    /// batched-prefill, where the per-token decode returns a fresh
    /// tensor but the model's `[T, hidden]` running buffer needs that
    /// row updated). Prefer `Ops.add(into:)` when an arithmetic
    /// combine is what's wanted; reach for `copy` for the pure-move
    /// case.
    public static func copy(
        _ src: Tensor, into dst: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            src.elementCount == dst.elementCount,
            "Ops.copy: src/dst element-count mismatch (\(src.elementCount) vs \(dst.elementCount))")
        precondition(
            src.dtype == dst.dtype,
            "Ops.copy: src/dst dtype mismatch (\(src.dtype) vs \(dst.dtype))")
        let bytes = src.elementCount * src.dtype.byteSize
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(
            from: src.buffer, sourceOffset: src.offset,
            to: dst.buffer, destinationOffset: dst.offset,
            size: bytes)
        blit.endEncoding()
    }

    // ─── Element-wise binary: add ────────────────────────────────────

    /// In-place scaled add: `accum[i] += scalar * src[i]`.
    public static func axpyScalarInplace(
        _ accum: Tensor, _ src: Tensor, scalar: Float, on cmd: MTLCommandBuffer
    ) {
        precondition(accum.shape == src.shape, "axpyScalarInplace: shape mismatch")
        precondition(accum.dtype == src.dtype, "axpyScalarInplace: dtype mismatch")
        let n = accum.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch accum.dtype {
        case .f32:
            IronKernels.iron_axpy_scalar_inplace_f32(
                src: src.buffer, srcOffset: src.offset,
                accum: accum.buffer, accumOffset: accum.offset,
                scalar: scalar,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_axpy_scalar_inplace_f16(
                src: src.buffer, srcOffset: src.offset,
                accum: accum.buffer, accumOffset: accum.offset,
                scalar: scalar,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_axpy_scalar_inplace_bf16(
                src: src.buffer, srcOffset: src.offset,
                accum: accum.buffer, accumOffset: accum.offset,
                scalar: scalar,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("axpyScalarInplace: unsupported dtype \(accum.dtype)")
        }
    }

    /// Fused multi-expert (top-K=6) down gemv + weighted accumulate.
    /// `accum[r] += Σ_k w[k] * (downs[k][r] · inners[k])`.
    /// 1 dispatch replaces 18 (6 gemv + 6 mul + 6 add).
    public static func moeDownWeightedSum6(
        downs: [Tensor], inners: [Tensor], weights: Tensor, accum: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(downs.count == 6 && inners.count == 6,
                     "moeDownWeightedSum6: expects exactly 6 of each")
        precondition(weights.dtype == .f32 && weights.elementCount >= 6,
                     "moeDownWeightedSum6: weights must be f32 of len 6")
        let m = downs[0].shape[0]
        let k = downs[0].shape[1]
        precondition(k <= 2048, "moeDownWeightedSum6: k \(k) exceeds kernel TG staging cap 2048")
        let dt = downs[0].dtype
        let tgWidth = 256
        let grid = MTLSize(width: m * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch dt {
        case .f16:
            IronKernels.iron_moe_down_weighted_sum_6_f16(
                down_0: downs[0].buffer, down_0Offset: downs[0].offset,
                inner_0: inners[0].buffer, inner_0Offset: inners[0].offset,
                down_1: downs[1].buffer, down_1Offset: downs[1].offset,
                inner_1: inners[1].buffer, inner_1Offset: inners[1].offset,
                down_2: downs[2].buffer, down_2Offset: downs[2].offset,
                inner_2: inners[2].buffer, inner_2Offset: inners[2].offset,
                down_3: downs[3].buffer, down_3Offset: downs[3].offset,
                inner_3: inners[3].buffer, inner_3Offset: inners[3].offset,
                down_4: downs[4].buffer, down_4Offset: downs[4].offset,
                inner_4: inners[4].buffer, inner_4Offset: inners[4].offset,
                down_5: downs[5].buffer, down_5Offset: downs[5].offset,
                inner_5: inners[5].buffer, inner_5Offset: inners[5].offset,
                weights: weights.buffer, weightsOffset: weights.offset,
                accum: accum.buffer, accumOffset: accum.offset,
                k: UInt32(k),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f32:
            IronKernels.iron_moe_down_weighted_sum_6_f32(
                down_0: downs[0].buffer, down_0Offset: downs[0].offset,
                inner_0: inners[0].buffer, inner_0Offset: inners[0].offset,
                down_1: downs[1].buffer, down_1Offset: downs[1].offset,
                inner_1: inners[1].buffer, inner_1Offset: inners[1].offset,
                down_2: downs[2].buffer, down_2Offset: downs[2].offset,
                inner_2: inners[2].buffer, inner_2Offset: inners[2].offset,
                down_3: downs[3].buffer, down_3Offset: downs[3].offset,
                inner_3: inners[3].buffer, inner_3Offset: inners[3].offset,
                down_4: downs[4].buffer, down_4Offset: downs[4].offset,
                inner_4: inners[4].buffer, inner_4Offset: inners[4].offset,
                down_5: downs[5].buffer, down_5Offset: downs[5].offset,
                inner_5: inners[5].buffer, inner_5Offset: inners[5].offset,
                weights: weights.buffer, weightsOffset: weights.offset,
                accum: accum.buffer, accumOffset: accum.offset,
                k: UInt32(k),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_down_weighted_sum_6_bf16(
                down_0: downs[0].buffer, down_0Offset: downs[0].offset,
                inner_0: inners[0].buffer, inner_0Offset: inners[0].offset,
                down_1: downs[1].buffer, down_1Offset: downs[1].offset,
                inner_1: inners[1].buffer, inner_1Offset: inners[1].offset,
                down_2: downs[2].buffer, down_2Offset: downs[2].offset,
                inner_2: inners[2].buffer, inner_2Offset: inners[2].offset,
                down_3: downs[3].buffer, down_3Offset: downs[3].offset,
                inner_3: inners[3].buffer, inner_3Offset: inners[3].offset,
                down_4: downs[4].buffer, down_4Offset: downs[4].offset,
                inner_4: inners[4].buffer, inner_4Offset: inners[4].offset,
                down_5: downs[5].buffer, down_5Offset: downs[5].offset,
                inner_5: inners[5].buffer, inner_5Offset: inners[5].offset,
                weights: weights.buffer, weightsOffset: weights.offset,
                accum: accum.buffer, accumOffset: accum.offset,
                k: UInt32(k),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("moeDownWeightedSum6: unsupported dtype \(dt)")
        }
    }

    /// Fused gate gemv + up gemv + SwiGLU. 1 dispatch replaces 3.
    /// `inner[r] = silu(gate_w[r] · x) * (up_w[r] · x)`.
    public static func gateUpSwigluFused(
        gateW: Tensor, upW: Tensor, x: Tensor, inner: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(gateW.shape.count == 2 && upW.shape.count == 2, "gateUpSwigluFused: weights 2D")
        precondition(x.shape.count == 1 && inner.shape.count == 1, "gateUpSwigluFused: vec/inner 1D")
        precondition(gateW.shape == upW.shape, "gateUpSwigluFused: gate/up shape mismatch")
        precondition(gateW.shape[0] == inner.shape[0], "gateUpSwigluFused: m mismatch")
        precondition(gateW.shape[1] == x.shape[0], "gateUpSwigluFused: k mismatch")
        let m = gateW.shape[0]
        let k = gateW.shape[1]
        let tgWidth = 256
        let grid = MTLSize(width: m * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch gateW.dtype {
        case .f32:
            IronKernels.iron_gate_up_swiglu_fused_f32(
                gate_w: gateW.buffer, gate_wOffset: gateW.offset,
                up_w: upW.buffer, up_wOffset: upW.offset,
                x: x.buffer, xOffset: x.offset,
                inner: inner.buffer, innerOffset: inner.offset,
                k: UInt32(k),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gate_up_swiglu_fused_f16(
                gate_w: gateW.buffer, gate_wOffset: gateW.offset,
                up_w: upW.buffer, up_wOffset: upW.offset,
                x: x.buffer, xOffset: x.offset,
                inner: inner.buffer, innerOffset: inner.offset,
                k: UInt32(k),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gate_up_swiglu_fused_bf16(
                gate_w: gateW.buffer, gate_wOffset: gateW.offset,
                up_w: upW.buffer, up_wOffset: upW.offset,
                x: x.buffer, xOffset: x.offset,
                inner: inner.buffer, innerOffset: inner.offset,
                k: UInt32(k),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("gateUpSwigluFused: unsupported dtype \(gateW.dtype)")
        }
    }

    /// Fused gemv + weighted in-place accumulate:
    /// `accum[r] += weight * (Σ_j mat[r, j] * vec[j])`.
    /// Saves 2 dispatches per call (vs gemv → Tensor.filled(w) → mul → add).
    public static func gemvAxpyInplace(
        mat: Tensor, vec: Tensor, accum: Tensor, weight: Float,
        on cmd: MTLCommandBuffer
    ) {
        precondition(mat.shape.count == 2, "gemvAxpyInplace: mat must be 2D")
        precondition(vec.shape.count == 1, "gemvAxpyInplace: vec must be 1D")
        precondition(accum.shape.count == 1, "gemvAxpyInplace: accum must be 1D")
        precondition(mat.shape[1] == vec.shape[0], "gemvAxpyInplace: k mismatch")
        precondition(mat.shape[0] == accum.shape[0], "gemvAxpyInplace: m mismatch")
        precondition(mat.dtype == vec.dtype && mat.dtype == accum.dtype, "gemvAxpyInplace: dtype mismatch")
        let m = mat.shape[0]
        let k = mat.shape[1]
        let tgWidth = 256
        let grid = MTLSize(width: m * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch mat.dtype {
        case .f32:
            IronKernels.iron_gemv_axpy_inplace_f32(
                mat: mat.buffer, matOffset: mat.offset,
                vec: vec.buffer, vecOffset: vec.offset,
                accum: accum.buffer, accumOffset: accum.offset,
                k: UInt32(k), weight: weight,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gemv_axpy_inplace_f16(
                mat: mat.buffer, matOffset: mat.offset,
                vec: vec.buffer, vecOffset: vec.offset,
                accum: accum.buffer, accumOffset: accum.offset,
                k: UInt32(k), weight: weight,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gemv_axpy_inplace_bf16(
                mat: mat.buffer, matOffset: mat.offset,
                vec: vec.buffer, vecOffset: vec.offset,
                accum: accum.buffer, accumOffset: accum.offset,
                k: UInt32(k), weight: weight,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("gemvAxpyInplace: unsupported dtype \(mat.dtype)")
        }
    }

    public static func add(
        _ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(a.shape == b.shape, "add: shape mismatch \(a.shape) vs \(b.shape)")
        precondition(a.dtype == b.dtype, "add: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let n = a.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch a.dtype {
        case .f32:
            IronKernels.vector_add_f32(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.vector_add_f16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.vector_add_bf16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.add: unsupported dtype \(a.dtype)")
        }
        return result
    }

    public static func mul(
        _ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(a.shape == b.shape, "mul: shape mismatch")
        precondition(a.dtype == b.dtype, "mul: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let n = a.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch a.dtype {
        case .f32:
            IronKernels.iron_mul_f32(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_mul_f16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_mul_bf16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.mul: unsupported dtype \(a.dtype)")
        }
        return result
    }

    /// Broadcast head-gate multiply: `out[i] = attn[i] * gate[i / headDim]`.
    ///
    /// Fuses the `gather(expand_ids) + mul` pair the softplus head-gate used
    /// (host-built `[n]` expand-id buffer → general `iron_gather` → separate
    /// `iron_mul`) into ONE `iron_head_gate_mul` broadcast pass. `gate` is the
    /// flat `[rows * nHeads]` pre-activated per-head scalar buffer; `attn` is
    /// the flat `[rows * nHeads * headDim]` attention output. Byte-identical
    /// to the gather+mul it replaces (the gather is a bit-exact copy of
    /// `gate[i / headDim]` and the product is the same native-`T` multiply).
    public static func headGateMul(
        attn: Tensor, gate: Tensor, headDim: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(attn.dtype == gate.dtype, "headGateMul: dtype mismatch")
        precondition(
            attn.elementCount % headDim == 0,
            "headGateMul: attn count \(attn.elementCount) not a multiple of headDim \(headDim)")
        let result = out ?? Tensor.empty(shape: attn.shape, dtype: attn.dtype)
        let n = attn.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch attn.dtype {
        case .f32:
            IronKernels.iron_head_gate_mul_f32(
                attn: attn.buffer, attnOffset: attn.offset,
                gate: gate.buffer, gateOffset: gate.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_head_gate_mul_f16(
                attn: attn.buffer, attnOffset: attn.offset,
                gate: gate.buffer, gateOffset: gate.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_head_gate_mul_bf16(
                attn: attn.buffer, attnOffset: attn.offset,
                gate: gate.buffer, gateOffset: gate.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.headGateMul: unsupported dtype \(attn.dtype)")
        }
        return result
    }

    public static func silu(
        _ x: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch x.dtype {
        case .f32:
            IronKernels.iron_silu_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_silu_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_silu_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.silu: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// Sigmoid: out[i] = 1 / (1 + exp(-x[i])). Wraps iron's
    /// `mt_sigmoid_*` element-wise kernel. Qwen3.5's gated attention
    /// output (`attn_output_gate`) multiplies the SDPA result by
    /// `sigmoid(gate)` before `o_proj`.
    public static func sigmoid(
        _ x: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch x.dtype {
        case .f32:
            IronKernels.iron_sigmoid_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_sigmoid_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_sigmoid_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sigmoid: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// ReLU: out[i] = max(x[i], 0). Wraps iron's `mt_relu_*`
    /// element-wise kernel. NemotronH's MLP / MoE feed-forward blocks
    /// use squared-ReLU (`relu(x)^2`) as their activation; the squaring
    /// is a follow-up `Ops.mul(relu, relu)` on the caller side.
    public static func relu(
        _ x: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch x.dtype {
        case .f32:
            IronKernels.iron_relu_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_relu_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_relu_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.relu: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// GELU (tanh-approximate, matching the math used by Gemma 3 / GPT-2
    /// / etc.). out[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/π) * (x[i] + 0.044715 * x[i]³))).
    /// Wraps iron's `mt_gelu_*` element-wise kernel.
    public static func gelu(
        _ x: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch x.dtype {
        case .f32:
            IronKernels.iron_gelu_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gelu_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gelu_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gelu: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// Softplus: out[i] = log(1 + exp(x[i])). Numerically stable across
    /// the full input range. Used by Mamba 2's `dt = softplus(dt_raw +
    /// dt_bias)` per-head time-step computation.
    public static func softplus(
        _ x: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch x.dtype {
        case .f32:
            IronKernels.iron_softplus_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_softplus_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_softplus_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.softplus: unsupported dtype \(x.dtype)")
        }
        return result
    }

    // Note: `Ops.exp` (wrapping `iron_exp_*`) already exists in
    // `OpsMath.swift` — `gdnGateBeta` below reuses it directly rather than
    // redeclaring it here.

    /// Embedding lookup. `table` is [vocab, dim], `tokenIds` is [n_tokens]
    /// (u32), output is [n_tokens, dim].
    public static func gather(
        table: Tensor, tokenIds: Tensor,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(table.shape.count == 2, "gather: table must be 2D")
        precondition(tokenIds.dtype == .u32, "gather: tokenIds must be u32")
        let dim = table.shape[1]
        let n = tokenIds.elementCount
        let result = out ?? Tensor.empty(shape: [n, dim], dtype: table.dtype)
        let totalThreads = n * dim
        let (grid, tg) = elementwiseGrid(totalThreads)
        switch table.dtype {
        case .f32:
            IronKernels.iron_gather_f32(
                table: table.buffer, tableOffset: table.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                dim: UInt32(dim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gather_f16(
                table: table.buffer, tableOffset: table.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                dim: UInt32(dim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gather_bf16(
                table: table.buffer, tableOffset: table.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                dim: UInt32(dim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gather: unsupported dtype \(table.dtype)")
        }
        return result
    }

    /// Two embedding lookups against the SAME `table` on one compute
    /// encoder. Used when two parallel id streams need to read the
    /// same embedding (or, in Qwen3 attention, when a single linear
    /// projection result is split into two head-half slices via
    /// `ids = [0..nHeads]` + `ids = [nHeads..2·nHeads]`). The table
    /// binding + `dim` constant are set once and the per-call
    /// `(ids, out)` pair rotates per dispatch.
    public static func gatherTwo(
        table: Tensor,
        ids1: Tensor, into out1: Tensor,
        ids2: Tensor, into out2: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(table.shape.count == 2, "Ops.gatherTwo: table must be 2D")
        precondition(
            ids1.dtype == .u32 && ids2.dtype == .u32,
            "Ops.gatherTwo: ids must be u32")
        precondition(
            out1.dtype == table.dtype && out2.dtype == table.dtype,
            "Ops.gatherTwo: out dtype must match table")
        let dim = table.shape[1]
        let psoName: String
        switch table.dtype {
        case .f32: psoName = "iron_gather_f32"
        case .f16: psoName = "iron_gather_f16"
        case .bf16: psoName = "iron_gather_bf16"
        default: fatalError("Ops.gatherTwo: unsupported dtype \(table.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        var dimV = UInt32(dim)
        enc.setBytes(&dimV, length: 4, index: 3)
        enc.setBuffer(table.buffer, offset: table.offset, index: 0)
        @inline(__always)
        func dispatch(_ ids: Tensor, _ out: Tensor) {
            let n = ids.elementCount
            let totalThreads = n * dim
            let (grid, tg) = elementwiseGrid(totalThreads)
            enc.setBuffer(ids.buffer, offset: ids.offset, index: 1)
            enc.setBuffer(out.buffer, offset: out.offset, index: 2)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(ids1, out1)
        dispatch(ids2, out2)
        enc.endEncoding()
    }

    /// Cooperative-thread matrix-vector multiply. weight: [out_dim, in_dim],
    /// input: [in_dim], output: [out_dim]. One threadgroup per output row;
    /// threads cooperate on the dot-product reduction.
    /// Device-side MoE dispatch plan (iron_moe_sort_plan). From the router's
    /// top-k expert ids [mTotal] produces sortedExperts/sourceTokens/invPerm
    /// in one dispatch — no readback / wait / host sort.
    public static func moeSortPlan(
        topkIds: Tensor, sortedExperts: Tensor, sourceTokens: Tensor, invPerm: Tensor,
        mTotal: Int, k: Int, on cmd: MTLCommandBuffer
    ) {
        let tgWidth = min(mTotal, 256)
        let grid = MTLSize(width: mTotal, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        IronKernels.iron_moe_sort_plan_f32(
            topk_ids: topkIds.buffer, topk_idsOffset: topkIds.offset,
            sorted_experts: sortedExperts.buffer, sorted_expertsOffset: sortedExperts.offset,
            source_tokens: sourceTokens.buffer, source_tokensOffset: sourceTokens.offset,
            inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
            m_total: UInt32(mTotal), k: UInt32(k),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    /// Bucketed counting-sort replacement for `moeSortPlan` (F-85 prefill
    /// gate). Same stable-order output contract, three dispatches on the
    /// caller's command buffer instead of one, no readback between them,
    /// same no-sync DEVPLAN2 property as the kernel it replaces:
    ///   1. `iron_moe_sort_plan_hist`    computes per-(block, expert) row counts.
    ///   2. `iron_moe_sort_plan_offsets` turns counts into each block's per-expert
    ///      write offset.
    ///   3. `iron_moe_sort_plan_scatter` has each row write itself using its
    ///      block's offset plus an in-block stable rank.
    /// `blockCounts`/`blockOffsets` are fresh scratch each call (sized off
    /// `mTotal`, which varies per prefill chunk), not cached, matching how
    /// `sortedExperts`/`sourceTokens`/`invPerm` are already allocated fresh
    /// per `decodeMany` call at this call site.
    public static func moeSortPlanCounting(
        topkIds: Tensor, sortedExperts: Tensor, sourceTokens: Tensor, invPerm: Tensor,
        mTotal: Int, k: Int, nExperts: Int, blockSize: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(
            nExperts <= 256,
            "moeSortPlanCounting: nExperts (\(nExperts)) exceeds the kernel's 256-lane threadgroup scratch capacity"
        )
        let nBlocks = (mTotal + blockSize - 1) / blockSize
        let blockCountsBuf = device.makeBuffer(length: nBlocks * nExperts * 4)
        let blockOffsetsBuf = device.makeBuffer(length: nBlocks * nExperts * 4)

        // Pass 1: per-block histogram. grid = nBlocks threadgroups, one
        // lane per expert.
        IronKernels.iron_moe_sort_plan_hist_f32_threadgroups(
            topk_ids: topkIds.buffer, topk_idsOffset: topkIds.offset,
            block_counts: blockCountsBuf, block_countsOffset: 0,
            m_total: UInt32(mTotal), n_experts: UInt32(nExperts), block_size: UInt32(blockSize),
            gridSize: MTLSize(width: nBlocks, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: nExperts, height: 1, depth: 1),
            on: cmd)

        // Pass 2: counts -> per-block per-expert write offsets. One
        // threadgroup, one lane per expert.
        IronKernels.iron_moe_sort_plan_offsets_f32_threadgroups(
            block_counts: blockCountsBuf, block_countsOffset: 0,
            block_offsets: blockOffsetsBuf, block_offsetsOffset: 0,
            n_blocks: UInt32(nBlocks), n_experts: UInt32(nExperts),
            gridSize: MTLSize(width: 1, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: nExperts, height: 1, depth: 1),
            on: cmd)

        // Pass 3: scatter. Same dispatchThreads convention as the
        // original `iron_moe_sort_plan` (grid in threads, data-sized).
        let tgWidth = min(mTotal, 256)
        IronKernels.iron_moe_sort_plan_scatter_f32(
            topk_ids: topkIds.buffer, topk_idsOffset: topkIds.offset,
            block_offsets: blockOffsetsBuf, block_offsetsOffset: 0,
            sorted_experts: sortedExperts.buffer, sorted_expertsOffset: sortedExperts.offset,
            source_tokens: sourceTokens.buffer, source_tokensOffset: sourceTokens.offset,
            inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
            m_total: UInt32(mTotal), k: UInt32(k), n_experts: UInt32(nExperts), block_size: UInt32(blockSize),
            gridSize: MTLSize(width: mTotal, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: tgWidth, height: 1, depth: 1),
            on: cmd)
    }

    /// Single-dispatch fusion of `moeSortPlanCounting`'s three-pass chain
    /// (F-85 Stage-1 fusion #1 / W61). Same stable-order output contract,
    /// same no-sync DEVPLAN2 property, ONE dispatch instead of three — no
    /// `blockCounts`/`blockOffsets` intermediate buffers at all, since
    /// every threadgroup recomputes what it needs directly from
    /// `topkIds` (see `iron_moe_route_sort_plan_fused`'s module doc for
    /// why that recompute is cheap and why the algorithm still needs zero
    /// cross-threadgroup synchronization). Same grid shape as
    /// `moeSortPlanCounting`'s pass 1 (`nBlocks` threadgroups, `nExperts`
    /// lanes) — this fusion trades away the two intermediate-buffer round
    /// trips and two dispatch/encoder overheads, not occupancy.
    public static func moeSortPlanFused(
        topkIds: Tensor, sortedExperts: Tensor, sourceTokens: Tensor, invPerm: Tensor,
        mTotal: Int, k: Int, nExperts: Int, blockSize: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(
            nExperts <= 256,
            "moeSortPlanFused: nExperts (\(nExperts)) exceeds the kernel's 256-lane threadgroup scratch capacity"
        )
        let nBlocks = (mTotal + blockSize - 1) / blockSize
        IronKernels.iron_moe_route_sort_plan_fused_f32_threadgroups(
            topk_ids: topkIds.buffer, topk_idsOffset: topkIds.offset,
            sorted_experts: sortedExperts.buffer, sorted_expertsOffset: sortedExperts.offset,
            source_tokens: sourceTokens.buffer, source_tokensOffset: sourceTokens.offset,
            inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
            m_total: UInt32(mTotal), k: UInt32(k), n_experts: UInt32(nExperts), block_size: UInt32(blockSize),
            gridSize: MTLSize(width: max(nBlocks, 1), height: 1, depth: 1),
            threadgroupSize: MTLSize(width: nExperts, height: 1, depth: 1),
            on: cmd)
    }

    /// Dequantize an mlx int4 affine-quantized `[outDim, inDim]` weight to a
    /// dense `[outDim, inDim]` tensor in one dispatch (one thread per packed
    /// u32 = 8 nibbles). Used to batch narrow-outDim projections (e.g. a
    /// `hidden -> 1` gate) through a single dense `gemm` instead of a
    /// per-row `dequantGemv` T-loop.
    public static func affineDequantize(
        weight: Tensor, scales: Tensor, biases: Tensor,
        outDim: Int, inDim: Int, groupSize: Int, bits: Int,
        on cmd: MTLCommandBuffer, into out: Tensor
    ) {
        precondition(weight.dtype == .u32, "affineDequantize: weight must be u32 packed")
        precondition(bits == 4 || bits == 8, "affineDequantize: only bits 4/8 supported")
        precondition(out.elementCount == outDim * inDim,
            "affineDequantize: out must be outDim*inDim (\(outDim*inDim)), got \(out.elementCount)")
        let packFactor = 32 / bits           // int4 -> 8, int8 -> 4 elems / u32
        let packs = (outDim * inDim) / packFactor
        let tgWidth = min(packs, 256)
        let grid = MTLSize(width: packs, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let gs = UInt32(groupSize)
        switch (bits, out.dtype) {
        case (4, .f32): IronKernels.iron_affine_dequantize_int4_f32(w: weight.buffer, wOffset: weight.offset, scales: scales.buffer, scalesOffset: scales.offset, biases: biases.buffer, biasesOffset: biases.offset, out: out.buffer, outOffset: out.offset, group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f16): IronKernels.iron_affine_dequantize_int4_f16(w: weight.buffer, wOffset: weight.offset, scales: scales.buffer, scalesOffset: scales.offset, biases: biases.buffer, biasesOffset: biases.offset, out: out.buffer, outOffset: out.offset, group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .bf16): IronKernels.iron_affine_dequantize_int4_bf16(w: weight.buffer, wOffset: weight.offset, scales: scales.buffer, scalesOffset: scales.offset, biases: biases.buffer, biasesOffset: biases.offset, out: out.buffer, outOffset: out.offset, group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f32): IronKernels.iron_affine_dequantize_int8_f32(w: weight.buffer, wOffset: weight.offset, scales: scales.buffer, scalesOffset: scales.offset, biases: biases.buffer, biasesOffset: biases.offset, out: out.buffer, outOffset: out.offset, group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f16): IronKernels.iron_affine_dequantize_int8_f16(w: weight.buffer, wOffset: weight.offset, scales: scales.buffer, scalesOffset: scales.offset, biases: biases.buffer, biasesOffset: biases.offset, out: out.buffer, outOffset: out.offset, group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .bf16): IronKernels.iron_affine_dequantize_int8_bf16(w: weight.buffer, wOffset: weight.offset, scales: scales.buffer, scalesOffset: scales.offset, biases: biases.buffer, biasesOffset: biases.offset, out: out.buffer, outOffset: out.offset, group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        default: fatalError("affineDequantize: unsupported bits/dtype \(bits)/\(out.dtype)")
        }
    }

    public static func gemv(
        weight: Tensor, input: Tensor,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.shape.count == 2, "gemv: weight must be 2D")
        precondition(input.shape.count == 1, "gemv: input must be 1D")
        precondition(
            weight.shape[1] == input.shape[0],
            "gemv: in_dim mismatch \(weight.shape[1]) vs \(input.shape[0])")
        precondition(weight.dtype == input.dtype, "gemv: dtype mismatch")
        let outDim = weight.shape[0]
        let inDim = weight.shape[1]
        if let reason = OpsValidation.validateGemv(outDim: outDim, inDim: inDim) {
            preconditionFailure("Ops.gemv: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: [outDim], dtype: weight.dtype)
        // Reduction kernel: one threadgroup per output row.
        // dispatchThreads dispatches outDim*tgWidth threads in groups
        // of tgWidth — yielding outDim threadgroups, each cooperating
        // over the in_dim axis.
        let tgWidth = 256
        let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch weight.dtype {
        case .f32:
            IronKernels.iron_gemv_f32(
                mat: weight.buffer, matOffset: weight.offset,
                vec: input.buffer, vecOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                k: UInt32(inDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gemv_f16(
                mat: weight.buffer, matOffset: weight.offset,
                vec: input.buffer, vecOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                k: UInt32(inDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gemv_bf16(
                mat: weight.buffer, matOffset: weight.offset,
                vec: input.buffer, vecOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                k: UInt32(inDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gemv: unsupported dtype \(weight.dtype)")
        }
        return result
    }

    /// Multi-row GEMM — `out[r, :] = weight · input[r, :]` for a block
    /// of `nRows` rows in one dispatch. `weight` is `[outDim, inDim]`,
    /// `input` is `[nRows, inDim]`, output is `[nRows, outDim]`.
    ///
    /// `iron_gemm` tiles the output 32×32 and stages weight + input
    /// tiles in threadgroup memory, so the weight is read once and
    /// reused across the block's rows — the projection-bandwidth win
    /// the diffusion / self-speculation block forward depends on.
    /// Reduction-mode kernel; the 1024-thread dispatch is hard.
    public static func gemm(
        weight: Tensor, input: Tensor, nRows: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.shape.count == 2, "Ops.gemm: weight must be 2D")
        let outDim = weight.shape[0]
        let inDim = weight.shape[1]
        if let reason = OpsValidation.validateGemm(inDim: inDim, outDim: outDim, nRows: nRows) {
            preconditionFailure("Ops.gemm: \(reason)")
        }
        precondition(
            input.elementCount == nRows * inDim,
            "Ops.gemm: input has \(input.elementCount) elements, expected "
                + "nRows*inDim = \(nRows * inDim)")
        precondition(weight.dtype == input.dtype, "Ops.gemm: weight/input dtype mismatch")
        let result = out ?? Tensor.empty(shape: [nRows, outDim], dtype: weight.dtype)
        // 32×32 output tiles, TPG = 1024. 1-D thread grid that Metal
        // slices into (outDim/32 ceil) × (nRows/32 ceil) threadgroups.
        let threadsPerGroup = 1024
        let nTiles = (outDim + 31) / 32
        let mTiles = (nRows + 31) / 32
        let grid = MTLSize(width: nTiles * threadsPerGroup, height: mTiles, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        switch weight.dtype {
        case .f32:
            IronKernels.iron_gemm_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                input: input.buffer, inputOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim), n_rows: UInt32(nRows),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gemm_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                input: input.buffer, inputOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim), n_rows: UInt32(nRows),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gemm_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                input: input.buffer, inputOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim), n_rows: UInt32(nRows),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gemm: unsupported dtype \(weight.dtype)")
        }
        return result
    }

    /// RMSNorm. x: [n], weight: [n], eps: scalar.
    /// Internally bound as a 1-element f32 buffer.
    /// Reduction kernel — one threadgroup per row.
    public static func rmsNorm(
        _ x: Tensor, weight: Tensor, eps: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(x.shape == weight.shape, "rmsNorm: weight/x shape mismatch")
        precondition(x.dtype == weight.dtype, "rmsNorm: dtype mismatch")
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount

        // Kernel-invariant validation. See OpsValidation.swift for the
        // full reasoning + a CI-runnable test of each precondition.
        if let reason = OpsValidation.validateRmsNorm(n: n) {
            preconditionFailure("Ops.rmsNorm: \(reason)")
        }
        dispatchRmsNorm(
            x: x, weight: weight, result: result,
            eps: eps, n: n, nRows: 1, on: cmd)
        return result
    }

    /// Multi-row RMSNorm. Input is [nRows, n]; weight is [n] (shared
    /// across all rows). Each row gets its own threadgroup. Used by
    /// Qwen3 to dispatch all per-head q_norm / k_norm in one call
    /// instead of one per head.
    public static func rmsNormRows(
        _ x: Tensor, weight: Tensor, eps: Float,
        nRows: Int, rowSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            x.elementCount == nRows * rowSize,
            "rmsNormRows: x size \(x.elementCount) ≠ nRows*rowSize")
        precondition(weight.elementCount == rowSize, "rmsNormRows: weight must be [rowSize]")
        precondition(x.dtype == weight.dtype, "rmsNormRows: dtype mismatch")
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)

        // Reduction kernel: one threadgroup per row. Per-row invariant
        // is the same as the single-row dispatch — see OpsValidation
        // for the full reasoning and CI-runnable tests.
        if let reason = OpsValidation.validateRmsNorm(n: rowSize) {
            preconditionFailure("Ops.rmsNormRows: \(reason)")
        }
        dispatchRmsNorm(
            x: x, weight: weight, result: result,
            eps: eps, n: rowSize, nRows: nRows, on: cmd)
        return result
    }

    /// Two `rmsNormRows` dispatches in ONE compute encoder. Used by
    /// the Qwen3 attention mixer's pre-RoPE Q-norm + K-norm pair: both
    /// run the fast `iron_rms_norm_*` kernel (TPG = rowSize / 4), share
    /// the same eps when set from the same model config, and there's
    /// no data dependency between them.
    ///
    /// `rowSize1` and `rowSize2` must both clear the fast-path bounds
    /// (`≤ 4096` and a multiple of 128 — checked via
    /// `OpsValidation.validateRmsNorm`). For wider rows the caller
    /// must fall back to two `rmsNormRows` calls so the wide kernel is
    /// picked individually.
    ///
    /// `eps` is forwarded via 1-element f32 buffers. If both eps
    /// values are equal we share a single alloc, otherwise two are
    /// allocated.
    public static func rmsNormRowsTwo(
        _ x1: Tensor, weight w1: Tensor, eps1: Float,
        nRows1: Int, rowSize1: Int, into out1: Tensor,
        _ x2: Tensor, weight w2: Tensor, eps2: Float,
        nRows2: Int, rowSize2: Int, into out2: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            x1.dtype == w1.dtype && x2.dtype == w2.dtype && x1.dtype == x2.dtype,
            "Ops.rmsNormRowsTwo: dtype mismatch")
        precondition(
            x1.elementCount == nRows1 * rowSize1,
            "Ops.rmsNormRowsTwo: x1 size \(x1.elementCount) ≠ nRows1·rowSize1")
        precondition(
            x2.elementCount == nRows2 * rowSize2,
            "Ops.rmsNormRowsTwo: x2 size \(x2.elementCount) ≠ nRows2·rowSize2")
        precondition(
            w1.elementCount == rowSize1 && w2.elementCount == rowSize2,
            "Ops.rmsNormRowsTwo: weight size must equal rowSize")
        precondition(
            out1.elementCount == x1.elementCount && out2.elementCount == x2.elementCount,
            "Ops.rmsNormRowsTwo: output element-count must match input")
        precondition(
            out1.dtype == x1.dtype && out2.dtype == x2.dtype,
            "Ops.rmsNormRowsTwo: output dtype must match input")
        if let reason = OpsValidation.validateRmsNorm(n: rowSize1) {
            preconditionFailure("Ops.rmsNormRowsTwo (#1): \(reason)")
        }
        if let reason = OpsValidation.validateRmsNorm(n: rowSize2) {
            preconditionFailure("Ops.rmsNormRowsTwo (#2): \(reason)")
        }
        let psoName: String
        switch x1.dtype {
        case .f32: psoName = "iron_rms_norm_f32"
        case .f16: psoName = "iron_rms_norm_f16"
        case .bf16: psoName = "iron_rms_norm_bf16"
        default: fatalError("Ops.rmsNormRowsTwo: unsupported dtype \(x1.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        // Allocate one or two eps buffers depending on whether the two
        // eps values match — RMSNorm pairs typically come from the
        // same checkpoint config and so share a single eps in practice.
        let epsBuf1: MTLBuffer = device.scalarBuffer(eps1)
        let epsBuf2: MTLBuffer = {
            if eps1 == eps2 { return epsBuf1 }
            return device.scalarBuffer(eps2)
        }()
        @inline(__always)
        func dispatch(
            _ x: Tensor, _ w: Tensor, _ out: Tensor,
            _ epsBuf: MTLBuffer, _ n: Int, _ nRows: Int
        ) {
            let tgWidth = n / 4
            let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            enc.setBuffer(x.buffer, offset: x.offset, index: 0)
            enc.setBuffer(w.buffer, offset: w.offset, index: 1)
            enc.setBuffer(out.buffer, offset: out.offset, index: 2)
            enc.setBuffer(epsBuf, offset: 0, index: 3)
            var nU = UInt32(n)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(x1, w1, out1, epsBuf1, rowSize1, nRows1)
        dispatch(x2, w2, out2, epsBuf2, rowSize2, nRows2)
        enc.endEncoding()
    }

    /// Fused (residual `a + b`) + RMSNorm. Returns **both** outputs:
    /// `residual` = `a + b` (the next layer's residual stream) and
    /// `normed` = `RMSNorm(a + b, weight)` (the input to the next
    /// sublayer). One dispatch instead of two — eliminates one round
    /// trip to GPU for the post-attention / post-FFN residual+norm
    /// pattern that every transformer block runs twice per token.
    ///
    /// Shape: `a`, `b` are `[rows, n]` (or any contiguous layout that
    /// flattens to `rows * n`); `weight` is `[n]`. Both outputs match
    /// `a`'s shape + dtype. `eps` is passed via a 1-element f32 buffer
    /// the wrapper allocates.
    ///
    /// **Row-width limit.** Kernel uses TPG = n / 4, so n must be ≤
    /// `OpsValidation.addRmsNormMaxRowSize` (4096) and a multiple of 4.
    /// Larger hidden sizes (Gemma 4 27B+ at 5376) MUST use separate
    /// `Ops.add` + `Ops.rmsNorm` calls. Pre-check with
    /// `OpsValidation.validateAddRmsNorm(n:)` if you don't know the
    /// model's hidden size statically.
    public static func addAndRmsNorm(
        _ a: Tensor, _ b: Tensor, weight: Tensor, eps: Float,
        nRows: Int, rowSize: Int, on cmd: MTLCommandBuffer,
        residualOut: Tensor? = nil, normedOut: Tensor? = nil
    ) -> (residual: Tensor, normed: Tensor) {
        precondition(
            a.elementCount == nRows * rowSize,
            "Ops.addAndRmsNorm: a size \(a.elementCount) ≠ nRows*rowSize")
        precondition(
            b.elementCount == nRows * rowSize,
            "Ops.addAndRmsNorm: b size \(b.elementCount) ≠ nRows*rowSize")
        precondition(
            weight.elementCount == rowSize,
            "Ops.addAndRmsNorm: weight must be [rowSize]")
        precondition(
            a.dtype == b.dtype && a.dtype == weight.dtype,
            "Ops.addAndRmsNorm: dtype mismatch")

        // Kernel-invariant validation. See OpsValidation.swift.
        if let reason = OpsValidation.validateAddRmsNorm(n: rowSize) {
            preconditionFailure("Ops.addAndRmsNorm: \(reason)")
        }

        let resid = residualOut ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let normed = normedOut ?? Tensor.empty(shape: a.shape, dtype: a.dtype)

        // eps as a 1-element f32 buffer.
        let epsBuf = device.scalarBuffer(eps)

        // TPG = n / 4 (kernel vectorises in 4-elem chunks). One
        // threadgroup per row → grid = nRows * (n/4) threads × 1 × 1.
        // Validator above ensures n/4 ∈ [32, 1024].
        let tgWidth = rowSize / 4
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let nU = UInt32(rowSize)
        switch a.dtype {
        case .f32:
            IronKernels.iron_add_rms_norm_f32(
                a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset,
                w: weight.buffer, wOffset: weight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                residual_out: resid.buffer, residual_outOffset: resid.offset,
                normed_out: normed.buffer, normed_outOffset: normed.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_add_rms_norm_f16(
                a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset,
                w: weight.buffer, wOffset: weight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                residual_out: resid.buffer, residual_outOffset: resid.offset,
                normed_out: normed.buffer, normed_outOffset: normed.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_add_rms_norm_bf16(
                a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset,
                w: weight.buffer, wOffset: weight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                residual_out: resid.buffer, residual_outOffset: resid.offset,
                normed_out: normed.buffer, normed_outOffset: normed.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.addAndRmsNorm: unsupported dtype \(a.dtype)")
        }
        return (resid, normed)
    }

    /// Shared RMSNorm dispatch for `rmsNorm` (nRows = 1) and
    /// `rmsNormRows`. Routes by row width:
    ///   • `n ≤ 4096` AND `n` is a multiple of 128 → `iron_rms_norm`,
    ///     4 elements per thread, TPG = n / 4 (the fast straight-line
    ///     kernel — every standard transformer hidden size).
    ///   • Anything else → `iron_rms_norm_wide`, whose strided loop covers
    ///     any width at a fixed TPG of 1024. Used by large-hidden text
    ///     models (Gemma 4 27B+ hidden 5376) AND by vision/audio towers
    ///     whose hidden dim isn't a 128-multiple (SmolVLM2 d=960 etc).
    /// One threadgroup per row in both cases.
    private static func dispatchRmsNorm(
        x: Tensor, weight: Tensor, result: Tensor,
        eps: Float, n: Int, nRows: Int, on cmd: MTLCommandBuffer
    ) {
        // eps as a 1-element f32 buffer.
        let epsBuf = device.scalarBuffer(eps)

        // Fast kernel needs TPG = n/4 with TPG a multiple of 32 and
        // ≤ 1024. Anything outside that — too wide, too narrow, or not
        // a 128-multiple — goes through the always-correct wide kernel.
        let useWide = n > 4096 || !n.isMultiple(of: 128)
        let tgWidth = useWide ? 1024 : n / 4
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch (useWide, x.dtype) {
        case (false, .f32):
            IronKernels.iron_rms_norm_f32(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (false, .f16):
            IronKernels.iron_rms_norm_f16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (false, .bf16):
            IronKernels.iron_rms_norm_bf16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (true, .f32):
            IronKernels.iron_rms_norm_wide_f32(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (true, .f16):
            IronKernels.iron_rms_norm_wide_f16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (true, .bf16):
            IronKernels.iron_rms_norm_wide_bf16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rmsNorm: unsupported dtype \(x.dtype)")
        }
    }

    /// Llama-3-style RoPE with frequency-band scaling. Pass `scaleFactor=1`
    /// + `originalMaxPosition` very large to disable scaling.
    public struct RoPEScaling: Sendable {
        public var scaleFactor: Float
        public var lowFreqFactor: Float
        public var highFreqFactor: Float
        public var originalMaxPosition: Float

        public init(
            scaleFactor: Float = 1, lowFreqFactor: Float = 1,
            highFreqFactor: Float = 4,
            originalMaxPosition: Float = 1e9
        ) {
            self.scaleFactor = scaleFactor
            self.lowFreqFactor = lowFreqFactor
            self.highFreqFactor = highFreqFactor
            self.originalMaxPosition = originalMaxPosition
        }

        public static let none = RoPEScaling()
    }

    public static func rope(
        _ qk: Tensor, position: Int, headDim: Int,
        thetaBase: Float,
        scaling: RoPEScaling = .none,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil,
        device: Device = .shared
    ) -> Tensor {
        precondition(qk.elementCount % headDim == 0, "rope: qk size must be multiple of headDim")
        let nHeads = qk.elementCount / headDim
        let halfDim = headDim / 2
        let result = out ?? Tensor.empty(shape: qk.shape, dtype: qk.dtype)
        let grid = MTLSize(width: 1, height: nHeads, depth: halfDim)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        // Single-token decode: 1-element positions buffer, row_stride = per-token
        // qk stride (n_heads * head_dim). See iron_rope_banded contract.
        let posBuf = device.makeBuffer(length: 4)
        var posVal = UInt32(position)
        memcpy(posBuf.contents(), &posVal, 4)
        let rowStride = UInt32(nHeads * headDim)
        switch qk.dtype {
        case .f32:
            IronKernels.iron_rope_banded_f32(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: posBuf, positionsOffset: 0,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim),
                row_stride: rowStride,
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_rope_banded_f16(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: posBuf, positionsOffset: 0,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim),
                row_stride: rowStride,
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_rope_banded_bf16(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: posBuf, positionsOffset: 0,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim),
                row_stride: rowStride,
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rope: unsupported dtype \(qk.dtype)")
        }
        return result
    }

    /// Partial-rotary RoPE. Rotates only the first `rotaryDim` elements
    /// of each `headDim`-strided head, leaving the remaining
    /// `headDim - rotaryDim` elements untouched. Qwen3.5 sets
    /// `partial_rotary_factor = 0.25` so a 256-dim head rotates only its
    /// first 64 dims.
    ///
    /// The `iron_rope_llama_*` kernel takes the per-head stride
    /// (`head_dim`) and the rotate-half pairing offset / grid height
    /// (`half_dim`) as independent constants. Driving it with
    /// `head_dim = headDim` (true stride) but `half_dim = rotaryDim / 2`
    /// rotates the pairs `(i, i + rotaryDim/2)` for `i ∈ [0, rotaryDim/2)`
    /// inside each head — exactly the partial-rotary subset. Dims
    /// `[rotaryDim, headDim)` are never written, so this MUST run
    /// in-place (`out` aliasing `qk`) — the caller's buffer already holds
    /// the correct pass-through values for the unrotated tail.
    ///
    /// `rotaryDim` must be even and ≤ `headDim`; `qk` must be a flat
    /// `[nHeads * headDim]` tensor.
    public static func ropePartial(
        _ qk: Tensor, position: Int,
        headDim: Int, rotaryDim: Int,
        thetaBase: Float,
        scaling: RoPEScaling = .none,
        on cmd: MTLCommandBuffer,
        device: Device = .shared
    ) {
        precondition(
            qk.elementCount % headDim == 0,
            "ropePartial: qk size must be a multiple of headDim")
        precondition(
            rotaryDim > 0 && rotaryDim <= headDim,
            "ropePartial: rotaryDim (\(rotaryDim)) must be in 1...headDim (\(headDim))")
        precondition(
            rotaryDim % 2 == 0,
            "ropePartial: rotaryDim (\(rotaryDim)) must be even (rotate-half pairs)")
        let nHeads = qk.elementCount / headDim
        let halfRotary = rotaryDim / 2
        // Grid: one thread per (head, rotary-pair). Writes in-place.
        let grid = MTLSize(width: 1, height: nHeads, depth: halfRotary)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        // Single-token decode: 1-element positions buffer, row_stride = per-token
        // qk stride (n_heads * head_dim). See iron_rope_banded contract.
        let posBuf = device.makeBuffer(length: 4)
        var posVal = UInt32(position)
        memcpy(posBuf.contents(), &posVal, 4)
        let rowStride = UInt32(nHeads * headDim)
        switch qk.dtype {
        case .f32:
            IronKernels.iron_rope_banded_f32(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: posBuf, positionsOffset: 0,
                out: qk.buffer, outOffset: qk.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfRotary),
                row_stride: rowStride,
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_rope_banded_f16(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: posBuf, positionsOffset: 0,
                out: qk.buffer, outOffset: qk.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfRotary),
                row_stride: rowStride,
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_rope_banded_bf16(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: posBuf, positionsOffset: 0,
                out: qk.buffer, outOffset: qk.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfRotary),
                row_stride: rowStride,
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.ropePartial: unsupported dtype \(qk.dtype)")
        }
    }

    /// Partial RoPE rotation on TWO tensors (typically Q + K) in ONE
    /// compute encoder. Both tensors share the same `(headDim,
    /// rotaryDim, position, thetaBase, scaling)` and dtype. The
    /// kernel writes in-place exactly like `ropePartial`, so passing
    /// `out == qk` is required.
    public static func ropePartialTwo(
        _ q: Tensor, _ k: Tensor, position: Int,
        headDim: Int, rotaryDim: Int,
        thetaBase: Float,
        scaling: RoPEScaling = .none,
        on cmd: MTLCommandBuffer,
        device: Device = .shared
    ) {
        precondition(
            q.dtype == k.dtype,
            "Ops.ropePartialTwo: dtype mismatch")
        precondition(
            q.elementCount % headDim == 0 && k.elementCount % headDim == 0,
            "Ops.ropePartialTwo: sizes must be multiples of headDim")
        precondition(
            rotaryDim > 0 && rotaryDim <= headDim && rotaryDim % 2 == 0,
            "Ops.ropePartialTwo: rotaryDim (\(rotaryDim)) must be even and in 1...headDim")
        let psoName: String
        switch q.dtype {
        case .f32: psoName = "iron_rope_banded_f32"
        case .f16: psoName = "iron_rope_banded_f16"
        case .bf16: psoName = "iron_rope_banded_bf16"
        default: fatalError("Ops.ropePartialTwo: unsupported dtype \(q.dtype)")
        }
        let halfRotary = rotaryDim / 2
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        // iron_rope_banded buffer layout: qk@0, positions@1, out@2, then
        // head_dim@3, half_dim@4, row_stride@5, theta@6, scale factors@7-10.
        // Decode = 1-element positions buffer (shared across q and k).
        let posBuf = device.makeBuffer(length: 4)
        var posVal = UInt32(position)
        memcpy(posBuf.contents(), &posVal, 4)
        enc.setBuffer(posBuf, offset: 0, index: 1)
        // RoPE constants are shared across q and k — set ONCE.
        var hd = UInt32(headDim)
        var half = UInt32(halfRotary)
        var theta = thetaBase
        var scaleFactor = scaling.scaleFactor
        var lowFreq = scaling.lowFreqFactor
        var highFreq = scaling.highFreqFactor
        var origMax = scaling.originalMaxPosition
        enc.setBytes(&hd, length: 4, index: 3)
        enc.setBytes(&half, length: 4, index: 4)
        enc.setBytes(&theta, length: 4, index: 6)
        enc.setBytes(&scaleFactor, length: 4, index: 7)
        enc.setBytes(&lowFreq, length: 4, index: 8)
        enc.setBytes(&highFreq, length: 4, index: 9)
        enc.setBytes(&origMax, length: 4, index: 10)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        @inline(__always)
        func dispatch(_ t: Tensor) {
            let nHeads = t.elementCount / headDim
            var rowStride = UInt32(nHeads * headDim)
            enc.setBytes(&rowStride, length: 4, index: 5)
            let grid = MTLSize(width: 1, height: nHeads, depth: halfRotary)
            enc.setBuffer(t.buffer, offset: t.offset, index: 0)
            enc.setBuffer(t.buffer, offset: t.offset, index: 2)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(q)
        dispatch(k)
        enc.endEncoding()
    }

    /// Batched per-row RoPE: applies position-dependent rotation to T
    /// rows of `qk` in ONE dispatch. `positions` is a `[T]` u32 buffer
    /// (per-row physical slot index). In-place rotation. Saves T-1
    /// encoder begin/end pairs versus a per-row `ropePartial` T-loop;
    /// at Qwen3.6-A3B prefill T=512 × 10 attn layers that's ≈ 5100
    /// fewer dispatches per prefill call.
    public static func ropePartialMany(
        _ qk: Tensor, positions: Tensor,
        t: Int, nHeads: Int,
        headDim: Int, rotaryDim: Int,
        rowStride: Int,
        thetaBase: Float,
        scaling: RoPEScaling = .none,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            positions.dtype == .u32,
            "Ops.ropePartialMany: positions must be .u32")
        precondition(
            positions.elementCount == t,
            "Ops.ropePartialMany: positions count must equal T")
        precondition(
            qk.elementCount == t * rowStride,
            "Ops.ropePartialMany: qk size \(qk.elementCount) ≠ T·rowStride = \(t * rowStride)")
        precondition(
            rotaryDim > 0 && rotaryDim <= headDim && rotaryDim % 2 == 0,
            "Ops.ropePartialMany: rotaryDim must be even and in 1...headDim")
        let halfRotary = rotaryDim / 2
        let psoName: String
        switch qk.dtype {
        case .f32: psoName = "iron_rope_banded_f32"
        case .f16: psoName = "iron_rope_banded_f16"
        case .bf16: psoName = "iron_rope_banded_bf16"
        default: fatalError("Ops.ropePartialMany: unsupported dtype \(qk.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qk.buffer, offset: qk.offset, index: 0)
        enc.setBuffer(positions.buffer, offset: positions.offset, index: 1)
        enc.setBuffer(qk.buffer, offset: qk.offset, index: 2)
        var hd = UInt32(headDim)
        var half = UInt32(halfRotary)
        var stride = UInt32(rowStride)
        var theta = thetaBase
        var sf = scaling.scaleFactor
        var lf = scaling.lowFreqFactor
        var hf = scaling.highFreqFactor
        var om = scaling.originalMaxPosition
        enc.setBytes(&hd, length: 4, index: 3)
        enc.setBytes(&half, length: 4, index: 4)
        enc.setBytes(&stride, length: 4, index: 5)
        enc.setBytes(&theta, length: 4, index: 6)
        enc.setBytes(&sf, length: 4, index: 7)
        enc.setBytes(&lf, length: 4, index: 8)
        enc.setBytes(&hf, length: 4, index: 9)
        enc.setBytes(&om, length: 4, index: 10)
        let grid = MTLSize(width: t, height: nHeads, depth: halfRotary)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// Pair of `ropePartialMany` calls (Q + K) sharing ONE compute
    /// encoder. Both buffers consume the same `positions`, `headDim`,
    /// `rotaryDim`, `thetaBase`, and `scaling`; only the per-buffer
    /// `nHeads` × `rowStride` differs.
    public static func ropePartialManyTwo(
        q: Tensor, qNHeads: Int, qRowStride: Int,
        k: Tensor, kNHeads: Int, kRowStride: Int,
        positions: Tensor, t: Int,
        headDim: Int, rotaryDim: Int,
        thetaBase: Float,
        scaling: RoPEScaling = .none,
        on cmd: MTLCommandBuffer
    ) {
        precondition(q.dtype == k.dtype, "Ops.ropePartialManyTwo: dtype mismatch")
        precondition(
            positions.dtype == .u32,
            "Ops.ropePartialManyTwo: positions must be .u32")
        precondition(
            positions.elementCount == t,
            "Ops.ropePartialManyTwo: positions count must equal T")
        precondition(
            rotaryDim > 0 && rotaryDim <= headDim && rotaryDim % 2 == 0,
            "Ops.ropePartialManyTwo: rotaryDim must be even and in 1...headDim")
        let halfRotary = rotaryDim / 2
        let psoName: String
        switch q.dtype {
        case .f32: psoName = "iron_rope_banded_f32"
        case .f16: psoName = "iron_rope_banded_f16"
        case .bf16: psoName = "iron_rope_banded_bf16"
        default: fatalError("Ops.ropePartialManyTwo: unsupported dtype \(q.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        // Shared bindings: positions + constants set ONCE.
        enc.setBuffer(positions.buffer, offset: positions.offset, index: 1)
        var hd = UInt32(headDim)
        var half = UInt32(halfRotary)
        var theta = thetaBase
        var sf = scaling.scaleFactor
        var lf = scaling.lowFreqFactor
        var hf = scaling.highFreqFactor
        var om = scaling.originalMaxPosition
        enc.setBytes(&hd, length: 4, index: 3)
        enc.setBytes(&half, length: 4, index: 4)
        enc.setBytes(&theta, length: 4, index: 6)
        enc.setBytes(&sf, length: 4, index: 7)
        enc.setBytes(&lf, length: 4, index: 8)
        enc.setBytes(&hf, length: 4, index: 9)
        enc.setBytes(&om, length: 4, index: 10)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        @inline(__always)
        func dispatch(_ buf: Tensor, _ nHeads: Int, _ rowStride: Int) {
            var stride = UInt32(rowStride)
            enc.setBuffer(buf.buffer, offset: buf.offset, index: 0)
            enc.setBuffer(buf.buffer, offset: buf.offset, index: 2)
            enc.setBytes(&stride, length: 4, index: 5)
            let grid = MTLSize(width: t, height: nHeads, depth: halfRotary)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(q, qNHeads, qRowStride)
        dispatch(k, kNHeads, kRowStride)
        enc.endEncoding()
    }

    /// YaRN RoPE parameters. `low` / `high` are the correction-range
    /// bounds — precomputed via `RoPEYaRN.from(...)` since they need a
    /// `floor`/`ceil`/`ln` computation that is constant across the
    /// dispatch. `factor == 1` collapses YaRN to plain RoPE.
    public struct RoPEYaRN: Sendable {
        public var factor: Float
        public var low: Float
        public var high: Float
        public var attnFactor: Float

        public init(factor: Float, low: Float, high: Float, attnFactor: Float = 1) {
            self.factor = factor
            self.low = low
            self.high = high
            self.attnFactor = attnFactor
        }

        /// Plain RoPE — `factor == 1` makes interpolation == extrapolation,
        /// so the ramp blend is a no-op.
        public static let plain = RoPEYaRN(factor: 1, low: 0, high: 1, attnFactor: 1)

        /// Build YaRN parameters from a checkpoint's `rope_parameters`
        /// block. Computes the correction-range bounds (`low` / `high`)
        /// from `beta_fast` / `beta_slow` and the YaRN mscale attention
        /// factor from `mscale` / `mscale_all_dim`.
        public static func from(
            headDim: Int, thetaBase: Float, factor: Float,
            betaFast: Float, betaSlow: Float,
            originalMaxPosition: Float,
            mscale: Float = 1, mscaleAllDim: Float = 1
        ) -> RoPEYaRN {
            // find_correction_dim — the dimension index at which a given
            // number of rotations occurs over the original context.
            func correctionDim(_ numRotations: Float) -> Float {
                (Float(headDim) * Foundation.log(originalMaxPosition / (numRotations * 2 * .pi)))
                    / (2 * Foundation.log(thetaBase))
            }
            var low = correctionDim(betaFast).rounded(.down)
            var high = correctionDim(betaSlow).rounded(.up)
            low = max(low, 0)
            high = min(high, Float(headDim - 1))
            if high <= low { high = low + 0.001 }  // avoid a zero-width ramp

            // YaRN mscale attention factor. When mscale == mscale_all_dim
            // (the common case) the ratio is exactly 1.
            func yarnMscale(_ scale: Float, _ m: Float) -> Float {
                scale <= 1 ? 1 : 0.1 * m * Foundation.log(scale) + 1
            }
            let attnFactor = yarnMscale(factor, mscale) / yarnMscale(factor, mscaleAllDim)
            return RoPEYaRN(factor: factor, low: low, high: high, attnFactor: attnFactor)
        }
    }

    /// YaRN RoPE — context-extended rotary embedding. Same Grid3D
    /// dispatch shape as `rope`: one threadgroup per (head, half-dim
    /// index). `factor == 1` reproduces plain RoPE bit-for-bit.
    public static func ropeYaRN(
        _ qk: Tensor, position: Int, headDim: Int,
        thetaBase: Float, yarn: RoPEYaRN,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            qk.elementCount % headDim == 0,
            "ropeYaRN: qk size must be a multiple of headDim")
        let nHeads = qk.elementCount / headDim
        let halfDim = headDim / 2
        let result = out ?? Tensor.empty(shape: qk.shape, dtype: qk.dtype)
        let grid = MTLSize(width: nHeads, height: halfDim, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch qk.dtype {
        case .f32:
            IronKernels.iron_rope_yarn_f32(
                qk: qk.buffer, qkOffset: qk.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), half_dim: UInt32(halfDim),
                position: UInt32(position), theta_base: thetaBase,
                factor: yarn.factor, low: yarn.low, high: yarn.high,
                attn_factor: yarn.attnFactor,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_rope_yarn_f16(
                qk: qk.buffer, qkOffset: qk.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), half_dim: UInt32(halfDim),
                position: UInt32(position), theta_base: thetaBase,
                factor: yarn.factor, low: yarn.low, high: yarn.high,
                attn_factor: yarn.attnFactor,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_rope_yarn_bf16(
                qk: qk.buffer, qkOffset: qk.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), half_dim: UInt32(halfDim),
                position: UInt32(position), theta_base: thetaBase,
                factor: yarn.factor, low: yarn.low, high: yarn.high,
                attn_factor: yarn.attnFactor,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.ropeYaRN: unsupported dtype \(qk.dtype)")
        }
        return result
    }

    /// MLX-format dequantizing gather (embedding lookup). bits ∈ {4, 8}.
    public static func dequantGather(
        weight: Tensor, scales: Tensor, biases: Tensor,
        tokenIds: Tensor, hidden: Int, bits: Int, groupSize: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.dtype == .u32, "dequantGather: weight must be u32 packed")
        precondition(tokenIds.dtype == .u32, "dequantGather: tokenIds must be u32")
        precondition(scales.dtype == biases.dtype, "dequantGather: scales/biases dtype mismatch")
        // Kernel-invariant validation (bit-width + silent-miscompute
        // footgun: partial trailing group). See
        // OpsValidation.validateDequantGather.
        if let reason = OpsValidation.validateDequantGather(
            hidden: hidden, bits: bits, groupSize: groupSize
        ) {
            preconditionFailure("Ops.dequantGather: \(reason)")
        }
        let n = tokenIds.elementCount
        let result = out ?? Tensor.empty(shape: [n, hidden], dtype: scales.dtype)
        let totalThreads = n * hidden
        let (grid, tg) = elementwiseGrid(totalThreads)
        let hiddenU = UInt32(hidden)
        let groupSizeU = UInt32(groupSize)
        switch (bits, scales.dtype) {
        case (4, .f32):
            IronKernels.iron_dequant_gather_int4_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f16):
            IronKernels.iron_dequant_gather_int4_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .bf16):
            IronKernels.iron_dequant_gather_int4_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f32):
            IronKernels.iron_dequant_gather_int8_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f16):
            IronKernels.iron_dequant_gather_int8_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .bf16):
            IronKernels.iron_dequant_gather_int8_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .f32):
            IronKernels.iron_dequant_gather_int6_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .f16):
            IronKernels.iron_dequant_gather_int6_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .bf16):
            IronKernels.iron_dequant_gather_int6_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f32):
            IronKernels.iron_dequant_gather_int3_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f16):
            IronKernels.iron_dequant_gather_int3_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .bf16):
            IronKernels.iron_dequant_gather_int3_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .f32):
            IronKernels.iron_dequant_gather_int5_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .f16):
            IronKernels.iron_dequant_gather_int5_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .bf16):
            IronKernels.iron_dequant_gather_int5_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f32):
            IronKernels.iron_dequant_gather_int2_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f16):
            IronKernels.iron_dequant_gather_int2_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .bf16):
            IronKernels.iron_dequant_gather_int2_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.dequantGather: unsupported (bits=\(bits), dtype=\(scales.dtype))")
        }
        return result
    }

    /// Backwards-compatible 4-bit alias.
    public static func dequantGatherInt4(
        weight: Tensor, scales: Tensor, biases: Tensor,
        tokenIds: Tensor, hidden: Int, groupSize: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        dequantGather(
            weight: weight, scales: scales, biases: biases,
            tokenIds: tokenIds, hidden: hidden, bits: 4,
            groupSize: groupSize, on: cmd, into: out)
    }

    /// MLX-format dequantizing GEMV. Weight is packed uint32; either 8
    /// 4-bit values per word (`bits == 4`) or 4 8-bit values per word
    /// (`bits == 8`). Per `groupSize` stripe of the in_dim axis, a
    /// per-row (scale, bias) pair dequantizes via `w_real = q*scale + bias`.
    /// Fused with the gemv accumulator — full weight matrix never
    /// materialized.
    public static func dequantGemv(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, bits: Int, groupSize: Int = 64,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.shape.count == 2, "dequantGemv: weight must be 2D")
        precondition(weight.dtype == .u32, "dequantGemv: weight must be u32 (packed)")
        precondition(
            scales.dtype == input.dtype && biases.dtype == input.dtype,
            "dequantGemv: scales/biases dtype must match input")
        let outDim = weight.shape[0]
        let packedPerRow = weight.shape[1]
        // Storage layout: bytes per row = in_dim * bits / 8.
        // packedPerRow uint32 = (in_dim * bits / 8) / 4 bytes, so:
        //   in_dim = packedPerRow * 32 / bits
        let inDim = packedPerRow * 32 / bits
        precondition(
            input.elementCount == inDim,
            "dequantGemv: input \(input.elementCount) ≠ in_dim \(inDim)")
        // Kernel-invariant validation (silent-miscompute footguns:
        // partial trailing group, unaligned pack tail, undersized
        // scales/biases). See OpsValidation.validateDequantGemv.
        if let reason = OpsValidation.validateDequantGemv(
            outDim: outDim, inDim: inDim, bits: bits, groupSize: groupSize,
            scalesCount: scales.elementCount, biasesCount: biases.elementCount
        ) {
            preconditionFailure("Ops.dequantGemv: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: [outDim], dtype: input.dtype)
        // Reduction kernel: one threadgroup per output row.
        let tgWidth = 256
        let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let inDimU = UInt32(inDim)
        let groupSizeU = UInt32(groupSize)

        switch (bits, input.dtype) {
        case (4, .f32):
            IronKernels.iron_dequant_gemv_int4_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f16):
            IronKernels.iron_dequant_gemv_int4_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .bf16):
            IronKernels.iron_dequant_gemv_int4_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f32):
            IronKernels.iron_dequant_gemv_int8_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f16):
            IronKernels.iron_dequant_gemv_int8_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .bf16):
            IronKernels.iron_dequant_gemv_int8_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .f32):
            IronKernels.iron_dequant_gemv_int6_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .f16):
            IronKernels.iron_dequant_gemv_int6_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .bf16):
            IronKernels.iron_dequant_gemv_int6_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f32):
            IronKernels.iron_dequant_gemv_int3_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f16):
            IronKernels.iron_dequant_gemv_int3_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .bf16):
            IronKernels.iron_dequant_gemv_int3_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .f32):
            IronKernels.iron_dequant_gemv_int5_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .f16):
            IronKernels.iron_dequant_gemv_int5_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .bf16):
            IronKernels.iron_dequant_gemv_int5_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f32):
            IronKernels.iron_dequant_gemv_int2_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f16):
            IronKernels.iron_dequant_gemv_int2_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .bf16):
            IronKernels.iron_dequant_gemv_int2_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.dequantGemv: unsupported (bits=\(bits), dtype=\(input.dtype))")
        }
        return result
    }

    /// Backwards-compatible 4-bit alias.
    public static func dequantGemvInt4(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, groupSize: Int = 64,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        dequantGemv(
            weight: weight, scales: scales, biases: biases,
            input: input, bits: 4, groupSize: groupSize,
            on: cmd, into: out)
    }

    /// Batched int4 dequantGemv on TWO projections sharing one input
    /// in ONE compute encoder. Used by `Qwen35MoEFFN.forward` for the
    /// per-expert gate + up pair: both projections read the same
    /// `[hidden]` post-norm activation, so we set the `input` binding
    /// once and rotate `(weight, scales, biases, output)` per dispatch.
    /// Saves one encoder begin/end pair per call.
    ///
    /// All outputs must share dtype with `input`; `groupSize` and the
    /// 4-bit packing are identical across the two projections.
    public static func dequantGemvInt4Two(
        input: Tensor,
        w0: Tensor, s0: Tensor, b0: Tensor, out0: Tensor,
        w1: Tensor, s1: Tensor, b1: Tensor, out1: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            input.dtype == out0.dtype && input.dtype == out1.dtype,
            "Ops.dequantGemvInt4Two: dtype mismatch")
        let psoName: String
        switch input.dtype {
        case .f32: psoName = "iron_dequant_gemv_int4_f32"
        case .f16: psoName = "iron_dequant_gemv_int4_f16"
        case .bf16: psoName = "iron_dequant_gemv_int4_bf16"
        default: fatalError("Ops.dequantGemvInt4Two: unsupported dtype \(input.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 3)
        // `inDim` derives from the packed-row width: 4-bit packs 8
        // weights per u32 word and `weight.shape[1]` counts words.
        let packedPerRow = w0.shape[1]
        let inDim = packedPerRow * 32 / 4
        var inDimV = UInt32(inDim)
        var groupSizeV = UInt32(groupSize)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&groupSizeV, length: 4, index: 6)
        let tgWidth = 256
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        @inline(__always)
        func dispatch(_ w: Tensor, _ s: Tensor, _ b: Tensor, _ out: Tensor) {
            enc.setBuffer(w.buffer, offset: w.offset, index: 0)
            enc.setBuffer(s.buffer, offset: s.offset, index: 1)
            enc.setBuffer(b.buffer, offset: b.offset, index: 2)
            enc.setBuffer(out.buffer, offset: out.offset, index: 4)
            let outDim = w.shape[0]
            let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(w0, s0, b0, out0)
        dispatch(w1, s1, b1, out1)
        enc.endEncoding()
    }

    /// Batched int4 dequantGemv on THREE projections sharing one input
    /// in ONE compute encoder. Used by the Qwen3.5/3.6 attention mixer
    /// for the q/k/v projection triplet.
    public static func dequantGemvInt4Three(
        input: Tensor,
        w0: Tensor, s0: Tensor, b0: Tensor, out0: Tensor,
        w1: Tensor, s1: Tensor, b1: Tensor, out1: Tensor,
        w2: Tensor, s2: Tensor, b2: Tensor, out2: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            input.dtype == out0.dtype,
            "Ops.dequantGemvInt4Three: dtype mismatch")
        let psoName: String
        switch input.dtype {
        case .f32: psoName = "iron_dequant_gemv_int4_f32"
        case .f16: psoName = "iron_dequant_gemv_int4_f16"
        case .bf16: psoName = "iron_dequant_gemv_int4_bf16"
        default: fatalError("Ops.dequantGemvInt4Three: unsupported dtype \(input.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 3)
        let packedPerRow = w0.shape[1]
        let inDim = packedPerRow * 32 / 4
        var inDimV = UInt32(inDim)
        var groupSizeV = UInt32(groupSize)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&groupSizeV, length: 4, index: 6)
        let tgWidth = 256
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        @inline(__always)
        func dispatch(_ w: Tensor, _ s: Tensor, _ b: Tensor, _ out: Tensor) {
            enc.setBuffer(w.buffer, offset: w.offset, index: 0)
            enc.setBuffer(s.buffer, offset: s.offset, index: 1)
            enc.setBuffer(b.buffer, offset: b.offset, index: 2)
            enc.setBuffer(out.buffer, offset: out.offset, index: 4)
            let outDim = w.shape[0]
            let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(w0, s0, b0, out0)
        dispatch(w1, s1, b1, out1)
        dispatch(w2, s2, b2, out2)
        enc.endEncoding()
    }

    /// Batched int4 dequantGemv on FOUR projections sharing one input
    /// in ONE compute encoder. Used by the Qwen3.5/3.6 GDN mixer where
    /// the four input projections (qkv, z, b, a) all read the same
    /// xNorm output.
    public static func dequantGemvInt4Four(
        input: Tensor,
        w0: Tensor, s0: Tensor, b0: Tensor, out0: Tensor,
        w1: Tensor, s1: Tensor, b1: Tensor, out1: Tensor,
        w2: Tensor, s2: Tensor, b2: Tensor, out2: Tensor,
        w3: Tensor, s3: Tensor, b3: Tensor, out3: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            input.dtype == out0.dtype,
            "Ops.dequantGemvInt4Four: dtype mismatch")
        let psoName: String
        switch input.dtype {
        case .f32: psoName = "iron_dequant_gemv_int4_f32"
        case .f16: psoName = "iron_dequant_gemv_int4_f16"
        case .bf16: psoName = "iron_dequant_gemv_int4_bf16"
        default: fatalError("Ops.dequantGemvInt4Four: unsupported dtype \(input.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 3)
        let packedPerRow = w0.shape[1]
        let inDim = packedPerRow * 32 / 4
        var inDimV = UInt32(inDim)
        var groupSizeV = UInt32(groupSize)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&groupSizeV, length: 4, index: 6)
        let tgWidth = 256
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        @inline(__always)
        func dispatch(_ w: Tensor, _ s: Tensor, _ b: Tensor, _ out: Tensor) {
            enc.setBuffer(w.buffer, offset: w.offset, index: 0)
            enc.setBuffer(s.buffer, offset: s.offset, index: 1)
            enc.setBuffer(b.buffer, offset: b.offset, index: 2)
            enc.setBuffer(out.buffer, offset: out.offset, index: 4)
            let outDim = w.shape[0]
            let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(w0, s0, b0, out0)
        dispatch(w1, s1, b1, out1)
        dispatch(w2, s2, b2, out2)
        dispatch(w3, s3, b3, out3)
        enc.endEncoding()
    }

    /// Batched int4 dequant-GEMV on N projections with DIFFERENT
    /// inputs sharing ONE compute encoder. Unlike `dequantGemvInt4Two`
    /// / `Three` / `Four` (which all share a single input), the per-
    /// expert MoE down phase has N independent activations — one per
    /// chosen expert — but all projections share PSO, `inDim`, and
    /// `groupSize`. The wrapper sets the constexprs once and rotates
    /// `(weight, scales, biases, input, output)` per dispatch. Saves
    /// N-1 encoder begin/end pairs versus N independent
    /// `dequantGemvInt4` calls.
    public static func dequantGemvInt4Many(
        weights: [Tensor], scales: [Tensor], biases: [Tensor],
        inputs: [Tensor], outputs: [Tensor],
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        dequantGemvMany(
            bits: 4, weights: weights, scales: scales, biases: biases,
            inputs: inputs, outputs: outputs, groupSize: groupSize, on: cmd)
    }

    /// Encoder-batched `dequantGemv` for bits ∈ {2, 3, 4, 5, 6, 8}.
    /// One PSO + one compute encoder; N dispatches rotate the five
    /// buffers. Used by Hy3 MoE (topK=8 × gate/up/down = up to 24
    /// gemvs per layer) to drop N-1 encoder begin/end pairs.
    public static func dequantGemvMany(
        bits: Int,
        weights: [Tensor], scales: [Tensor], biases: [Tensor],
        inputs: [Tensor], outputs: [Tensor],
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        let n = weights.count
        precondition(
            scales.count == n && biases.count == n && inputs.count == n && outputs.count == n,
            "Ops.dequantGemvMany: count mismatch")
        guard n > 0 else { return }
        let dtype = inputs[0].dtype
        let psoName: String
        switch (bits, dtype) {
        case (2, .f32): psoName = "iron_dequant_gemv_int2_f32"
        case (2, .f16): psoName = "iron_dequant_gemv_int2_f16"
        case (2, .bf16): psoName = "iron_dequant_gemv_int2_bf16"
        case (3, .f32): psoName = "iron_dequant_gemv_int3_f32"
        case (3, .f16): psoName = "iron_dequant_gemv_int3_f16"
        case (3, .bf16): psoName = "iron_dequant_gemv_int3_bf16"
        case (4, .f32): psoName = "iron_dequant_gemv_int4_f32"
        case (4, .f16): psoName = "iron_dequant_gemv_int4_f16"
        case (4, .bf16): psoName = "iron_dequant_gemv_int4_bf16"
        case (5, .f32): psoName = "iron_dequant_gemv_int5_f32"
        case (5, .f16): psoName = "iron_dequant_gemv_int5_f16"
        case (5, .bf16): psoName = "iron_dequant_gemv_int5_bf16"
        case (6, .f32): psoName = "iron_dequant_gemv_int6_f32"
        case (6, .f16): psoName = "iron_dequant_gemv_int6_f16"
        case (6, .bf16): psoName = "iron_dequant_gemv_int6_bf16"
        case (8, .f32): psoName = "iron_dequant_gemv_int8_f32"
        case (8, .f16): psoName = "iron_dequant_gemv_int8_f16"
        case (8, .bf16): psoName = "iron_dequant_gemv_int8_bf16"
        default:
            fatalError("Ops.dequantGemvMany: unsupported (bits=\(bits), dtype=\(dtype))")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        let packedPerRow = weights[0].shape[1]
        let inDim = packedPerRow * 32 / bits
        var inDimV = UInt32(inDim)
        var groupSizeV = UInt32(groupSize)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&groupSizeV, length: 4, index: 6)
        let tgWidth = 256
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        for i in 0 ..< n {
            precondition(
                weights[i].shape[1] == packedPerRow,
                "Ops.dequantGemvMany: inDim varies at index \(i)")
            precondition(
                inputs[i].dtype == dtype && outputs[i].dtype == dtype,
                "Ops.dequantGemvMany: dtype varies at index \(i)")
            enc.setBuffer(weights[i].buffer, offset: weights[i].offset, index: 0)
            enc.setBuffer(scales[i].buffer, offset: scales[i].offset, index: 1)
            enc.setBuffer(biases[i].buffer, offset: biases[i].offset, index: 2)
            enc.setBuffer(inputs[i].buffer, offset: inputs[i].offset, index: 3)
            enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 4)
            let outDim = weights[i].shape[0]
            let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
    }

    /// Fused Q/K/V int4 dequant-GEMV in ONE dispatch via
    /// `iron_batched_qkv_qgemv_fast`. Replaces the 3-dispatch
    /// `dequantGemvInt4Three` shared-encoder form: the matrix is
    /// selected by `program_id<2>()` across a `[ceil(maxOut/8), 1, 3]`
    /// threadgroup grid, so all 3 projections share a single kernel
    /// launch and read `x` once into TG memory instead of three times
    /// from DRAM. Saves 2 encoder begin/end pairs per attention layer
    /// at decode T=1.
    ///
    /// Output layout: q, k, v concatenated in that order in `out`;
    /// caller slices into the three regions.
    ///
    /// Kernel constraints (mirror `iron_rms_norm_qgemv_fast`):
    /// - `in_dim` MUST be a multiple of 512.
    /// - Each of `out_q`, `out_k`, `out_v` MUST be a multiple of 8.
    /// - `group_size` MUST be 64.
    /// - TPG = 64.
    public static func batchedQkvQgemvInt4Fast(
        x: Tensor,
        wQ: Tensor, scalesQ: Tensor, biasesQ: Tensor,
        wK: Tensor, scalesK: Tensor, biasesK: Tensor,
        wV: Tensor, scalesV: Tensor, biasesV: Tensor,
        outQ: Int, outK: Int, outV: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            wQ.dtype == .u32 && wK.dtype == .u32 && wV.dtype == .u32,
            "Ops.batchedQkvQgemvInt4Fast: w_* must be u32-packed")
        let packedPerRow = wQ.shape[1]
        let inDim = packedPerRow * 8
        // K and V share the same in_dim as Q — the kernel reads ONE x row
        // and walks all three weight tensors at that packed width. A
        // mis-packed wK/wV would silently miscompute the K/V projections.
        precondition(
            wK.shape[1] == packedPerRow && wV.shape[1] == packedPerRow,
            "Ops.batchedQkvQgemvInt4Fast: wK/wV packed width (\(wK.shape[1])/\(wV.shape[1])) must equal wQ (\(packedPerRow))"
        )
        precondition(
            x.elementCount == inDim,
            "Ops.batchedQkvQgemvInt4Fast: x.elementCount \(x.elementCount) ≠ inDim \(inDim)")
        precondition(
            out.elementCount == outQ + outK + outV,
            "Ops.batchedQkvQgemvInt4Fast: out.elementCount must be q+k+v")
        precondition(
            inDim % 512 == 0,
            "Ops.batchedQkvQgemvInt4Fast: in_dim must be a multiple of 512")
        precondition(
            outQ % 8 == 0 && outK % 8 == 0 && outV % 8 == 0,
            "Ops.batchedQkvQgemvInt4Fast: out_q/k/v must each be a multiple of 8")
        let groupSize = 64
        let maxOut = max(outQ, max(outK, outV))
        // Grid: [ceil(maxOut/8) * TPG, 1, 3] — TG count per matrix axis
        // times TPG=64 lanes; 3 matrices.
        let tpg = 64
        let nTiles = maxOut / 8
        let grid = MTLSize(width: nTiles * tpg, height: 1, depth: 3)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            IronKernels.iron_batched_qkv_qgemv_fast_f32(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                out: out.buffer, outOffset: out.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_batched_qkv_qgemv_fast_f16(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                out: out.buffer, outOffset: out.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_batched_qkv_qgemv_fast_bf16(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                out: out.buffer, outOffset: out.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.batchedQkvQgemvInt4Fast: unsupported dtype \(x.dtype)")
        }
    }

    /// M>1 (batched-prefill) sibling of `batchedQkvQgemvInt4Fast`. Reads
    /// `x = [M, in_dim]` once into TG memory and produces all 3
    /// projections into THREE separate contiguous output tensors
    /// (`qBuf [M, out_q]`, `kBuf [M, out_k]`, `vBuf [M, out_v]`). The
    /// split-output layout lets downstream Q/K/V code read each
    /// projection as `[M, dim]` without strided views (vs. a single
    /// concat buffer). One dispatch replaces three independent qmm
    /// dispatches; the `xNorm` DRAM roundtrip is paid once instead of
    /// three times.
    ///
    /// Constraints (mirror `iron_batched_qkv_qmm_fast`):
    /// - `in_dim` MUST be a multiple of 512.
    /// - Each of `out_q`, `out_k`, `out_v` MUST be a multiple of 8.
    /// - `group_size` MUST be 64. TPG = 64.
    /// - All three outputs MUST be pre-zeroed (kernel only writes
    ///   valid `row0 < out_*` tiles per matrix branch). The wrapper
    ///   handles this — callers do not need to zero ahead of time.
    public static func batchedQkvQmmFast(
        x: Tensor,  // [M, in_dim]
        wQ: Tensor, scalesQ: Tensor, biasesQ: Tensor,
        wK: Tensor, scalesK: Tensor, biasesK: Tensor,
        wV: Tensor, scalesV: Tensor, biasesV: Tensor,
        m: Int, outQ: Int, outK: Int, outV: Int,
        on cmd: MTLCommandBuffer,
        qBuf: Tensor,  // [M, out_q]
        kBuf: Tensor,  // [M, out_k]
        vBuf: Tensor  // [M, out_v]
    ) {
        precondition(
            wQ.dtype == .u32 && wK.dtype == .u32 && wV.dtype == .u32,
            "Ops.batchedQkvQmmFast: w_* must be u32-packed")
        let packedPerRow = wQ.shape[1]
        let inDim = packedPerRow * 8
        // K and V share the same in_dim as Q — the kernel reads ONE x row
        // and walks all three weight tensors at that packed width. A
        // mis-packed wK/wV would silently miscompute the K/V projections.
        precondition(
            wK.shape[1] == packedPerRow && wV.shape[1] == packedPerRow,
            "Ops.batchedQkvQmmFast: wK/wV packed width (\(wK.shape[1])/\(wV.shape[1])) must equal wQ (\(packedPerRow))"
        )
        precondition(
            x.elementCount == m * inDim,
            "Ops.batchedQkvQmmFast: x.elementCount \(x.elementCount) ≠ M·inDim \(m * inDim)")
        precondition(
            qBuf.elementCount == m * outQ,
            "Ops.batchedQkvQmmFast: qBuf.elementCount must be M·out_q")
        precondition(
            kBuf.elementCount == m * outK,
            "Ops.batchedQkvQmmFast: kBuf.elementCount must be M·out_k")
        precondition(
            vBuf.elementCount == m * outV,
            "Ops.batchedQkvQmmFast: vBuf.elementCount must be M·out_v")
        precondition(
            inDim % 512 == 0,
            "Ops.batchedQkvQmmFast: in_dim must be a multiple of 512")
        precondition(
            outQ % 8 == 0 && outK % 8 == 0 && outV % 8 == 0,
            "Ops.batchedQkvQmmFast: out_q/k/v must each be a multiple of 8")
        // Pre-zero outputs: the kernel tile mechanic depends on the
        // buffer starting at zero where the smaller two matrices' tiles
        // past their `out_*` count no-op the store.
        qBuf.zero()
        kBuf.zero()
        vBuf.zero()
        let groupSize = 64
        let tpg = 64
        let maxOut = max(outQ, max(outK, outV))
        let nTiles = maxOut / 8
        let grid = MTLSize(width: nTiles * tpg, height: m, depth: 3)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            IronKernels.iron_batched_qkv_qmm_fast_f32(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                q_buf: qBuf.buffer, q_bufOffset: qBuf.offset,
                k_buf: kBuf.buffer, k_bufOffset: kBuf.offset,
                v_buf: vBuf.buffer, v_bufOffset: vBuf.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_batched_qkv_qmm_fast_f16(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                q_buf: qBuf.buffer, q_bufOffset: qBuf.offset,
                k_buf: kBuf.buffer, k_bufOffset: kBuf.offset,
                v_buf: vBuf.buffer, v_bufOffset: vBuf.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_batched_qkv_qmm_fast_bf16(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                q_buf: qBuf.buffer, q_bufOffset: qBuf.offset,
                k_buf: kBuf.buffer, k_bufOffset: kBuf.offset,
                v_buf: vBuf.buffer, v_bufOffset: vBuf.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.batchedQkvQmmFast: unsupported dtype \(x.dtype)")
        }
    }

    /// Fused 4-output int4 dequant-GEMV in ONE dispatch via
    /// `iron_batched_4_qgemv_fast`. Extends the QKV (3-output) pattern
    /// to 4 projections sharing the same input. Used by the Qwen3.5
    /// GDN mixer where the four input projections (qkv, z, b, a) all
    /// read the same `xNorm` and can be fused into a single kernel
    /// launch. Replaces the 4-dispatch shared-encoder form
    /// (`Ops.dequantGemvInt4Four`) with one launch, paying the input
    /// DRAM roundtrip once instead of four times.
    ///
    /// Constraints (mirror `iron_batched_qkv_qgemv_fast`):
    /// - `in_dim` MUST be a multiple of 512.
    /// - Each of `out_a / out_b / out_c / out_d` MUST be a multiple
    ///   of 8.
    /// - `group_size` MUST be 64.
    /// - TPG = 64. Grid: `[ceil(max(out_*) / 8) * TPG, 1, 4]`.
    public static func batched4QgemvInt4Fast(
        input x: Tensor,
        wA: Tensor, scalesA: Tensor, biasesA: Tensor, outA: Tensor,
        wB: Tensor, scalesB: Tensor, biasesB: Tensor, outB: Tensor,
        wC: Tensor, scalesC: Tensor, biasesC: Tensor, outC: Tensor,
        wD: Tensor, scalesD: Tensor, biasesD: Tensor, outD: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            wA.dtype == .u32 && wB.dtype == .u32 && wC.dtype == .u32 && wD.dtype == .u32,
            "Ops.batched4QgemvInt4Fast: w_* must be u32-packed")
        let packedPerRow = wA.shape[1]
        let inDim = packedPerRow * 8
        // All four weight tensors share the same in_dim — the kernel reads
        // ONE x row and walks all four at that packed width. A mis-packed
        // wB/wC/wD would silently miscompute its branch's output.
        precondition(
            wB.shape[1] == packedPerRow && wC.shape[1] == packedPerRow
                && wD.shape[1] == packedPerRow,
            "Ops.batched4QgemvInt4Fast: wB/wC/wD packed width must equal wA (\(packedPerRow))"
        )
        precondition(
            x.elementCount == inDim,
            "Ops.batched4QgemvInt4Fast: x.elementCount \(x.elementCount) ≠ inDim \(inDim)")
        let outA_ = outA.elementCount
        let outB_ = outB.elementCount
        let outC_ = outC.elementCount
        let outD_ = outD.elementCount
        precondition(
            inDim % 512 == 0,
            "Ops.batched4QgemvInt4Fast: in_dim must be a multiple of 512")
        precondition(
            outA_ % 8 == 0 && outB_ % 8 == 0 && outC_ % 8 == 0 && outD_ % 8 == 0,
            "Ops.batched4QgemvInt4Fast: each out_* must be a multiple of 8")
        let tpg = 64
        let maxOut = max(max(outA_, outB_), max(outC_, outD_))
        let nTiles = maxOut / 8
        let grid = MTLSize(width: nTiles * tpg, height: 1, depth: 4)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            IronKernels.iron_batched_4_qgemv_fast_f32(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_out: outA.buffer, a_outOffset: outA.offset,
                b_out: outB.buffer, b_outOffset: outB.offset,
                c_out: outC.buffer, c_outOffset: outC.offset,
                d_out: outD.buffer, d_outOffset: outD.offset,
                out_a: UInt32(outA_), out_b: UInt32(outB_),
                out_c: UInt32(outC_), out_d: UInt32(outD_),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_batched_4_qgemv_fast_f16(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_out: outA.buffer, a_outOffset: outA.offset,
                b_out: outB.buffer, b_outOffset: outB.offset,
                c_out: outC.buffer, c_outOffset: outC.offset,
                d_out: outD.buffer, d_outOffset: outD.offset,
                out_a: UInt32(outA_), out_b: UInt32(outB_),
                out_c: UInt32(outC_), out_d: UInt32(outD_),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_batched_4_qgemv_fast_bf16(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_out: outA.buffer, a_outOffset: outA.offset,
                b_out: outB.buffer, b_outOffset: outB.offset,
                c_out: outC.buffer, c_outOffset: outC.offset,
                d_out: outD.buffer, d_outOffset: outD.offset,
                out_a: UInt32(outA_), out_b: UInt32(outB_),
                out_c: UInt32(outC_), out_d: UInt32(outD_),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.batched4QgemvInt4Fast: unsupported dtype \(x.dtype)")
        }
    }

    /// M>1 (batched-prefill) sibling of `batched4QgemvInt4Fast`. Reads
    /// `x = [M, in_dim]` once into TG memory and produces all 4
    /// outputs (each `[M, out_*]`) in a single dispatch. Eliminates
    /// the 3 redundant input DRAM reads of the unfused `callMany`
    /// chain used by the GDN mixer's batched forward (qkv, z, b, a all
    /// project from the same `xNorm`).
    ///
    /// Output tensors are caller-allocated and indexed independently
    /// (`outA / outB / outC / outD`, each `[M, out_*]`).
    ///
    /// Constraints (mirror `iron_batched_4_qgemv_fast`):
    /// - `in_dim` MUST be a multiple of 512.
    /// - Each of `out_a / out_b / out_c / out_d` MUST be a multiple
    ///   of 8.
    /// - `group_size` MUST be 64. TPG = 64.
    public static func batched4QmmFast(
        input x: Tensor, m: Int,
        wA: Tensor, scalesA: Tensor, biasesA: Tensor, outA: Tensor,
        wB: Tensor, scalesB: Tensor, biasesB: Tensor, outB: Tensor,
        wC: Tensor, scalesC: Tensor, biasesC: Tensor, outC: Tensor,
        wD: Tensor, scalesD: Tensor, biasesD: Tensor, outD: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            wA.dtype == .u32 && wB.dtype == .u32 && wC.dtype == .u32 && wD.dtype == .u32,
            "Ops.batched4QmmFast: w_* must be u32-packed")
        let packedPerRow = wA.shape[1]
        let inDim = packedPerRow * 8
        // All four weight tensors share the same in_dim — the kernel reads
        // ONE x row and walks all four at that packed width. A mis-packed
        // wB/wC/wD would silently miscompute its branch's output.
        precondition(
            wB.shape[1] == packedPerRow && wC.shape[1] == packedPerRow
                && wD.shape[1] == packedPerRow,
            "Ops.batched4QmmFast: wB/wC/wD packed width must equal wA (\(packedPerRow))"
        )
        precondition(
            x.elementCount == m * inDim,
            "Ops.batched4QmmFast: x size \(x.elementCount) ≠ M·inDim \(m * inDim)")
        let outADim = wA.shape[0]
        let outBDim = wB.shape[0]
        let outCDim = wC.shape[0]
        let outDDim = wD.shape[0]
        precondition(
            outA.elementCount == m * outADim,
            "Ops.batched4QmmFast: outA size \(outA.elementCount) ≠ M·outADim \(m * outADim)")
        precondition(
            outB.elementCount == m * outBDim,
            "Ops.batched4QmmFast: outB size \(outB.elementCount) ≠ M·outBDim \(m * outBDim)")
        precondition(
            outC.elementCount == m * outCDim,
            "Ops.batched4QmmFast: outC size \(outC.elementCount) ≠ M·outCDim \(m * outCDim)")
        precondition(
            outD.elementCount == m * outDDim,
            "Ops.batched4QmmFast: outD size \(outD.elementCount) ≠ M·outDDim \(m * outDDim)")
        precondition(
            inDim % 512 == 0,
            "Ops.batched4QmmFast: in_dim must be a multiple of 512")
        precondition(
            outADim % 8 == 0 && outBDim % 8 == 0 && outCDim % 8 == 0 && outDDim % 8 == 0,
            "Ops.batched4QmmFast: each out_* must be a multiple of 8")
        let tpg = 64
        let maxOut = max(max(outADim, outBDim), max(outCDim, outDDim))
        let nTiles = maxOut / 8
        let grid = MTLSize(width: nTiles * tpg, height: m, depth: 4)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            IronKernels.iron_batched_4_qmm_fast_f32(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_buf: outA.buffer, a_bufOffset: outA.offset,
                b_buf: outB.buffer, b_bufOffset: outB.offset,
                c_buf: outC.buffer, c_bufOffset: outC.offset,
                d_buf: outD.buffer, d_bufOffset: outD.offset,
                out_a: UInt32(outADim), out_b: UInt32(outBDim),
                out_c: UInt32(outCDim), out_d: UInt32(outDDim),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_batched_4_qmm_fast_f16(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_buf: outA.buffer, a_bufOffset: outA.offset,
                b_buf: outB.buffer, b_bufOffset: outB.offset,
                c_buf: outC.buffer, c_bufOffset: outC.offset,
                d_buf: outD.buffer, d_bufOffset: outD.offset,
                out_a: UInt32(outADim), out_b: UInt32(outBDim),
                out_c: UInt32(outCDim), out_d: UInt32(outDDim),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_batched_4_qmm_fast_bf16(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_buf: outA.buffer, a_bufOffset: outA.offset,
                b_buf: outB.buffer, b_bufOffset: outB.offset,
                c_buf: outC.buffer, c_bufOffset: outC.offset,
                d_buf: outD.buffer, d_bufOffset: outD.offset,
                out_a: UInt32(outADim), out_b: UInt32(outBDim),
                out_c: UInt32(outCDim), out_d: UInt32(outDDim),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.batched4QmmFast: unsupported dtype \(x.dtype)")
        }
    }

    /// Exact-row sibling of `batched4QmmFast` that dispatches four BM3 or
    /// BM4 projections through one compute encoder. Each projection reuses
    /// its weight tile across the tile's activation rows, while the shared
    /// encoder keeps the four-dispatch host cost close to the fused entry.
    public static func batched4QmmReuseRowsFast(
        input x: Tensor, m: Int,
        wA: Tensor, scalesA: Tensor, biasesA: Tensor, outA: Tensor,
        wB: Tensor, scalesB: Tensor, biasesB: Tensor, outB: Tensor,
        wC: Tensor, scalesC: Tensor, biasesC: Tensor, outC: Tensor,
        wD: Tensor, scalesD: Tensor, biasesD: Tensor, outD: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition((m == 6 || m == 8 || m == 9) && groupSize == 64)
        precondition(
            wA.dtype == .u32 && wB.dtype == .u32 && wC.dtype == .u32 && wD.dtype == .u32)
        let k = wA.shape[1] * 8
        precondition(k % 256 == 0)
        precondition(
            wB.shape[1] * 8 == k && wC.shape[1] * 8 == k && wD.shape[1] * 8 == k)
        let tileRows = m == 8 ? 4 : 3
        let kernelStem = m == 8 ? "iron_qmm_bm4" : "iron_qmm_bm3"
        let psoName: String
        switch x.dtype {
        case .f32: psoName = "\(kernelStem)_f32"
        case .f16: psoName = "\(kernelStem)_f16"
        case .bf16: psoName = "\(kernelStem)_bf16"
        default: fatalError("Ops.batched4QmmReuseRowsFast: unsupported dtype \(x.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        var kU = UInt32(k)
        var gsU = UInt32(k / groupSize)
        @inline(__always)
        func dispatch(
            weight: Tensor, scales: Tensor, biases: Tensor,
            output: Tensor
        ) {
            let n = weight.shape[0]
            precondition(n % 8 == 0 && output.elementCount == m * n)
            var nU = UInt32(n)
            enc.setBuffer(weight.buffer, offset: weight.offset, index: 0)
            enc.setBuffer(scales.buffer, offset: scales.offset, index: 1)
            enc.setBuffer(biases.buffer, offset: biases.offset, index: 2)
            enc.setBuffer(x.buffer, offset: x.offset, index: 3)
            enc.setBuffer(output.buffer, offset: output.offset, index: 4)
            enc.setBytes(&kU, length: 4, index: 5)
            enc.setBytes(&nU, length: 4, index: 6)
            enc.setBytes(&gsU, length: 4, index: 7)
            enc.dispatchThreadgroups(
                MTLSize(width: n / 8, height: m / tileRows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        dispatch(weight: wA, scales: scalesA, biases: biasesA, output: outA)
        dispatch(weight: wB, scales: scalesB, biases: biasesB, output: outB)
        dispatch(weight: wC, scales: scalesC, biases: biasesC, output: outC)
        dispatch(weight: wD, scales: scalesD, biases: biasesD, output: outD)
        enc.endEncoding()
    }

    /// Per-expert indexed int4 dequant-GEMV. The caller stacks every
    /// expert's weight slab into one `[nExperts, outDim, inDim/8]`
    /// u32-packed tensor (and matching scales / biases stacks); the
    /// kernel reads `expertIndex[0]` to pick the slab to dequantize on
    /// this call. Paired with `Ops.moeRouterTopK`, the top-K indices
    /// stay GPU-resident and the host-side `route → dispatch expert
    /// gemv` sync at every MoE layer collapses to one
    /// `expertIndex.buffer + offset = slot · 4` view per slot.
    public static func dequantGemvInt4ExpertIndexed(
        weightsStacked: Tensor, scalesStacked: Tensor, biasesStacked: Tensor,
        input: Tensor, expertIndex: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            weightsStacked.shape.count == 3,
            "Ops.dequantGemvInt4ExpertIndexed: weightsStacked must be [nExperts, outDim, inDim/8]")
        precondition(
            weightsStacked.dtype == .u32,
            "Ops.dequantGemvInt4ExpertIndexed: weightsStacked must be u32 (packed)")
        precondition(
            expertIndex.dtype == .u32 && expertIndex.elementCount == 1,
            "Ops.dequantGemvInt4ExpertIndexed: expertIndex must be [1] u32")
        precondition(
            scalesStacked.dtype == input.dtype && biasesStacked.dtype == input.dtype,
            "Ops.dequantGemvInt4ExpertIndexed: scales / biases dtype must match input")
        precondition(
            out.dtype == input.dtype,
            "Ops.dequantGemvInt4ExpertIndexed: out dtype must match input")
        let outDim = weightsStacked.shape[1]
        let packedPerRow = weightsStacked.shape[2]
        let inDim = packedPerRow * 8  // int4 packs 8 weights per u32 word
        precondition(
            input.elementCount == inDim,
            "Ops.dequantGemvInt4ExpertIndexed: input \(input.elementCount) ≠ inDim \(inDim)")
        precondition(
            out.elementCount == outDim,
            "Ops.dequantGemvInt4ExpertIndexed: out \(out.elementCount) ≠ outDim \(outDim)")
        // Kernel runs in Reduction mode with tpg=32 (one simdgroup per
        // output row). `dispatchThreads` counts THREADS not threadgroups
        // — total threads = outDim · 32 gives outDim threadgroups, one
        // per output row.
        // Dispatch via PSOCache + encoder (Generated IronKernels may omit
        // the affine int4 expert-indexed wrappers after a partial emit).
        dequantGemvInt4ExpertIndexedMany(
            weightsStacked: [weightsStacked], scalesStacked: [scalesStacked],
            biasesStacked: [biasesStacked], inputs: [input],
            expertIndices: [expertIndex], outputs: [out],
            groupSize: groupSize, on: cmd)
    }

    /// Encoder-batched form of `dequantGemvInt4ExpertIndexed`. All N
    /// calls must share `(inDim, outDim, groupSize, dtype)` so they
    /// reuse the same PSO + constexpr binding; only the per-call
    /// `(weights, scales, biases, input, expertIndex, output)` rotate
    /// per dispatch. Saves N-1 encoder begin/end pairs versus N
    /// independent calls. At Qwen3.6-A3B's topK=8 the gate+up phase
    /// becomes 16 calls on one encoder and the down phase 8 calls on
    /// a second encoder per MoE layer.
    public static func dequantGemvInt4ExpertIndexedMany(
        weightsStacked: [Tensor], scalesStacked: [Tensor], biasesStacked: [Tensor],
        inputs: [Tensor], expertIndices: [Tensor], outputs: [Tensor],
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        let n = weightsStacked.count
        precondition(
            scalesStacked.count == n && biasesStacked.count == n
                && inputs.count == n && expertIndices.count == n
                && outputs.count == n,
            "Ops.dequantGemvInt4ExpertIndexedMany: count mismatch")
        guard n > 0 else { return }
        let dtype = inputs[0].dtype
        let psoName: String
        switch dtype {
        case .f32: psoName = "iron_dequant_gemv_int4_expert_indexed_f32"
        case .f16: psoName = "iron_dequant_gemv_int4_expert_indexed_f16"
        case .bf16: psoName = "iron_dequant_gemv_int4_expert_indexed_bf16"
        default:
            fatalError(
                "Ops.dequantGemvInt4ExpertIndexedMany: unsupported dtype \(dtype)")
        }
        let outDim = weightsStacked[0].shape[1]
        let packedPerRow = weightsStacked[0].shape[2]
        let inDim = packedPerRow * 8
        var inDimV = UInt32(inDim)
        var outDimV = UInt32(outDim)
        var groupSizeV = UInt32(groupSize)
        let pso = PSOCache.shared.pipelineState(for: psoName)
        // Concurrent default: gate‖up and multi-expert gemvs write distinct
        // outs (Hy3 int2 path). Serial escape: BUTTER_MOE_SERIAL_EXPERT_GEMV=1.
        let useConcurrent =
            ButterEnv["BUTTER_MOE_SERIAL_EXPERT_GEMV"] == nil
        let enc: MTLComputeCommandEncoder?
        if useConcurrent {
            enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        } else {
            enc = cmd.makeComputeCommandEncoder()
        }
        guard let enc else { return }
        enc.setComputePipelineState(pso)
        // Shared constexprs (kernel buffer indices 6, 7, 8) set once.
        enc.setBytes(&inDimV, length: 4, index: 6)
        enc.setBytes(&outDimV, length: 4, index: 7)
        enc.setBytes(&groupSizeV, length: 4, index: 8)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let grid = MTLSize(width: outDim * 32, height: 1, depth: 1)
        for i in 0 ..< n {
            precondition(
                weightsStacked[i].shape[1] == outDim
                    && weightsStacked[i].shape[2] == packedPerRow,
                "Ops.dequantGemvInt4ExpertIndexedMany: weight shape varies at \(i)")
            precondition(
                inputs[i].dtype == dtype && outputs[i].dtype == dtype,
                "Ops.dequantGemvInt4ExpertIndexedMany: dtype varies at \(i)")
            precondition(
                expertIndices[i].dtype == .u32 && expertIndices[i].elementCount == 1,
                "Ops.dequantGemvInt4ExpertIndexedMany: expertIndices[\(i)] must be [1] u32")
            enc.setBuffer(weightsStacked[i].buffer, offset: weightsStacked[i].offset, index: 0)
            enc.setBuffer(scalesStacked[i].buffer, offset: scalesStacked[i].offset, index: 1)
            enc.setBuffer(biasesStacked[i].buffer, offset: biasesStacked[i].offset, index: 2)
            enc.setBuffer(inputs[i].buffer, offset: inputs[i].offset, index: 3)
            enc.setBuffer(expertIndices[i].buffer, offset: expertIndices[i].offset, index: 4)
            enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 5)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
    }

    /// Encoder-batched int2 affine expert-indexed dequant-GEMV (MLX oQ2
    /// pack: 16 vals/u32, scales+biases per group). Kernel name
    /// `iron_dequant_gemv_int2_expert_indexed_*` — may live in
    /// `hy3_extra.metallib` or be live-compiled from
    /// `Resources/kernels/`. Same buffer layout as the int4 affine form.
    public static func dequantGemvInt2ExpertIndexedMany(
        weightsStacked: [Tensor], scalesStacked: [Tensor], biasesStacked: [Tensor],
        inputs: [Tensor], expertIndices: [Tensor], outputs: [Tensor],
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer,
        concurrent: Bool = true
    ) {
        let n = weightsStacked.count
        precondition(
            scalesStacked.count == n && biasesStacked.count == n
                && inputs.count == n && expertIndices.count == n
                && outputs.count == n,
            "Ops.dequantGemvInt2ExpertIndexedMany: count mismatch")
        guard n > 0 else { return }
        let dtype = inputs[0].dtype
        let psoName: String
        switch dtype {
        case .f32: psoName = "iron_dequant_gemv_int2_expert_indexed_f32"
        case .f16: psoName = "iron_dequant_gemv_int2_expert_indexed_f16"
        case .bf16: psoName = "iron_dequant_gemv_int2_expert_indexed_bf16"
        default:
            fatalError(
                "Ops.dequantGemvInt2ExpertIndexedMany: unsupported dtype \(dtype)")
        }
        let outDim = weightsStacked[0].shape[1]
        let packedPerRow = weightsStacked[0].shape[2]
        let inDim = packedPerRow * 16  // int2 packs 16 weights per u32
        var inDimV = UInt32(inDim)
        var outDimV = UInt32(outDim)
        var groupSizeV = UInt32(groupSize)
        let pso = PSOCache.shared.pipelineState(for: psoName)
        // Concurrent default: gate‖up and multi-expert gemvs have no
        // producer-consumer edges within the batch (distinct out buffers).
        // BUTTER_HY3_SERIAL_EXPERT_GEMV=1 forces serial encoder.
        let useConcurrent =
            concurrent
            && ButterEnv["BUTTER_HY3_SERIAL_EXPERT_GEMV"] == nil
        let enc: MTLComputeCommandEncoder?
        if useConcurrent {
            enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        } else {
            enc = cmd.makeComputeCommandEncoder()
        }
        guard let enc else { return }
        enc.setComputePipelineState(pso)
        enc.setBytes(&inDimV, length: 4, index: 6)
        enc.setBytes(&outDimV, length: 4, index: 7)
        enc.setBytes(&groupSizeV, length: 4, index: 8)
        // One SG per output row (tg=32). Wider TGs measured slower on
        // Hy3 oQ2 decode (K=4096, outDim=1536/4096).
        let tgWidth = 32
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
        for i in 0 ..< n {
            precondition(
                weightsStacked[i].shape[1] == outDim
                    && weightsStacked[i].shape[2] == packedPerRow,
                "Ops.dequantGemvInt2ExpertIndexedMany: weight shape varies at \(i)")
            precondition(
                inputs[i].dtype == dtype && outputs[i].dtype == dtype,
                "Ops.dequantGemvInt2ExpertIndexedMany: dtype varies at \(i)")
            precondition(
                expertIndices[i].dtype == .u32 && expertIndices[i].elementCount == 1,
                "Ops.dequantGemvInt2ExpertIndexedMany: expertIndices[\(i)] must be [1] u32")
            enc.setBuffer(weightsStacked[i].buffer, offset: weightsStacked[i].offset, index: 0)
            enc.setBuffer(scalesStacked[i].buffer, offset: scalesStacked[i].offset, index: 1)
            enc.setBuffer(biasesStacked[i].buffer, offset: biasesStacked[i].offset, index: 2)
            enc.setBuffer(inputs[i].buffer, offset: inputs[i].offset, index: 3)
            enc.setBuffer(expertIndices[i].buffer, offset: expertIndices[i].offset, index: 4)
            enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 5)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
    }

    /// Multi-row Path B pre-score: for each of `tRows` tokens,
    /// `unbiased[t·E+e] = sigmoid(logits[...])`,
    /// `biased[t·E+e] = unbiased + bias[e]`. One dispatch over T·E.
    public static func moeSigmoidBiasRows(
        logits: Tensor, bias: Tensor,
        unbiased: Tensor, biased: Tensor,
        tRows: Int, nExperts: Int,
        on cmd: MTLCommandBuffer
    ) {
        let total = tRows * nExperts
        precondition(
            logits.elementCount == total && unbiased.elementCount == total
                && biased.elementCount == total,
            "moeSigmoidBiasRows: logits/unbiased/biased must be [T·E]")
        precondition(
            bias.elementCount == nExperts,
            "moeSigmoidBiasRows: bias must be [nExperts]")
        precondition(
            logits.dtype == bias.dtype && bias.dtype == unbiased.dtype
                && unbiased.dtype == biased.dtype,
            "moeSigmoidBiasRows: dtype mismatch")
        let psoName: String
        switch logits.dtype {
        case .f32: psoName = "iron_moe_sigmoid_bias_rows_f32"
        case .f16: psoName = "iron_moe_sigmoid_bias_rows_f16"
        case .bf16: psoName = "iron_moe_sigmoid_bias_rows_bf16"
        default:
            fatalError("Ops.moeSigmoidBiasRows: unsupported dtype \(logits.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(logits.buffer, offset: logits.offset, index: 0)
        enc.setBuffer(bias.buffer, offset: bias.offset, index: 1)
        enc.setBuffer(unbiased.buffer, offset: unbiased.offset, index: 2)
        enc.setBuffer(biased.buffer, offset: biased.offset, index: 3)
        var nE = UInt32(nExperts)
        var tR = UInt32(tRows)
        enc.setBytes(&nE, length: 4, index: 4)
        enc.setBytes(&tR, length: 4, index: 5)
        let tgWidth = min(total, 256)
        let grid = MTLSize(width: total, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// Hy3 / NemotronH Path B on device: top-K by biased scores, weights
    /// from unbiased (renorm sum-to-1). Does NOT apply
    /// `routed_scaling_factor` — caller multiplies after.
    /// Single-row form — scores `[nExperts]`.
    public static func moeRouterTopKBiased(
        scoreBiased: Tensor, scoreUnbiased: Tensor,
        indicesOut: Tensor, weightsOut: Tensor,
        nExperts: Int, k: Int,
        on cmd: MTLCommandBuffer
    ) {
        moeRouterTopKBiasedMany(
            scoreBiased: scoreBiased, scoreUnbiased: scoreUnbiased,
            indicesOut: indicesOut, weightsOut: weightsOut,
            t: 1, nExperts: nExperts, k: k, on: cmd)
    }

    /// T-row Path B top-k. Scores `[T·nExperts]` (row-major), indices
    /// `[T·k]` u32, weights `[T·k]`.
    public static func moeRouterTopKBiasedMany(
        scoreBiased: Tensor, scoreUnbiased: Tensor,
        indicesOut: Tensor, weightsOut: Tensor,
        t: Int, nExperts: Int, k: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(t > 0, "Ops.moeRouterTopKBiasedMany: t must be positive")
        precondition(
            scoreBiased.elementCount == t * nExperts
                && scoreUnbiased.elementCount == t * nExperts,
            "Ops.moeRouterTopKBiasedMany: scores must be [T·nExperts]")
        precondition(
            indicesOut.elementCount == t * k && indicesOut.dtype == .u32,
            "Ops.moeRouterTopKBiasedMany: indicesOut must be [T·k] u32")
        precondition(
            weightsOut.elementCount == t * k
                && weightsOut.dtype == scoreBiased.dtype,
            "Ops.moeRouterTopKBiasedMany: weightsOut must be [T·k] matching scores")
        let grd = MTLSize(width: t * 32, height: 1, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        switch scoreBiased.dtype {
        case .f32:
            IronKernels.iron_moe_router_topk_biased_f32(
                score_biased: scoreBiased.buffer, score_biasedOffset: scoreBiased.offset,
                score_unbiased: scoreUnbiased.buffer, score_unbiasedOffset: scoreUnbiased.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k),
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_router_topk_biased_f16(
                score_biased: scoreBiased.buffer, score_biasedOffset: scoreBiased.offset,
                score_unbiased: scoreUnbiased.buffer, score_unbiasedOffset: scoreUnbiased.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k),
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_router_topk_biased_bf16(
                score_biased: scoreBiased.buffer, score_biasedOffset: scoreBiased.offset,
                score_unbiased: scoreUnbiased.buffer, score_unbiasedOffset: scoreUnbiased.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k),
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.moeRouterTopKBiasedMany: unsupported dtype \(scoreBiased.dtype)")
        }
    }

    /// W69: single-dispatch fusion of `moeSigmoidBiasRows` +
    /// `moeRouterTopKBiasedMany` (`iron_moe_router_topk_biased_rows_fused`).
    /// Reads RAW router GEMV `logits` directly — no precomputed
    /// `unbiased`/`biased` score tensors — and computes sigmoid+bias inline
    /// per masked-argmax candidate, replicating the two-kernel chain's
    /// per-stage rounding exactly (see the kernel's module doc). Does NOT
    /// apply `routedScalingFactor` — same contract as
    /// `moeRouterTopKBiasedMany`, caller multiplies after.
    public static func moeRouterTopKBiasedRowsFused(
        logits: Tensor, bias: Tensor,
        indicesOut: Tensor, weightsOut: Tensor,
        t: Int, nExperts: Int, k: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(t > 0, "Ops.moeRouterTopKBiasedRowsFused: t must be positive")
        precondition(
            logits.elementCount == t * nExperts,
            "Ops.moeRouterTopKBiasedRowsFused: logits must be [T·nExperts]")
        precondition(
            bias.elementCount == nExperts,
            "Ops.moeRouterTopKBiasedRowsFused: bias must be [nExperts]")
        precondition(
            indicesOut.elementCount == t * k && indicesOut.dtype == .u32,
            "Ops.moeRouterTopKBiasedRowsFused: indicesOut must be [T·k] u32")
        precondition(
            weightsOut.elementCount == t * k && weightsOut.dtype == logits.dtype,
            "Ops.moeRouterTopKBiasedRowsFused: weightsOut must be [T·k] matching logits")
        precondition(
            logits.dtype == bias.dtype,
            "Ops.moeRouterTopKBiasedRowsFused: logits/bias dtype mismatch")
        let grd = MTLSize(width: t * 32, height: 1, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        switch logits.dtype {
        case .f32:
            IronKernels.iron_moe_router_topk_biased_rows_fused_f32(
                logits: logits.buffer, logitsOffset: logits.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k),
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_router_topk_biased_rows_fused_f16(
                logits: logits.buffer, logitsOffset: logits.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k),
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_router_topk_biased_rows_fused_bf16(
                logits: logits.buffer, logitsOffset: logits.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k),
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.moeRouterTopKBiasedRowsFused: unsupported dtype \(logits.dtype)")
        }
    }

    /// MoE top-K router. Reads `[nExperts]` raw logits and writes
    /// `[k]` u32 expert indices + `[k]` weights of the same dtype.
    /// `normTopkProb` matches Qwen3-MoE convention when `true`
    /// (softmax restricted to the chosen k, weights renormalise to
    /// sum-to-1); `false` matches the Qwen3-Next style (softmax over
    /// all experts, then pick top-k without renormalisation).
    ///
    /// Single-row form — `logits: [nExperts]`. Internally calls the
    /// T-batched variant with `t = 1`.
    public static func moeRouterTopK(
        logits: Tensor, indicesOut: Tensor, weightsOut: Tensor,
        nExperts: Int, k: Int, normTopkProb: Bool,
        on cmd: MTLCommandBuffer
    ) {
        moeRouterTopKMany(
            logits: logits, indicesOut: indicesOut, weightsOut: weightsOut,
            t: 1, nExperts: nExperts, k: k, normTopkProb: normTopkProb,
            on: cmd)
    }

    /// T-batched MoE top-K router. `logits: [T, nExperts]`,
    /// `indicesOut: [T, k]` u32, `weightsOut: [T, k]` (matching logits
    /// dtype). The kernel iterates `T` rows along `program_id<0>()`
    /// (one threadgroup per row), so a single dispatch covers all
    /// prefill rows without the host commit + wait + CPU
    /// `MoERouter.route` round-trip that the per-row form forced.
    public static func moeRouterTopKMany(
        logits: Tensor, indicesOut: Tensor, weightsOut: Tensor,
        t: Int, nExperts: Int, k: Int, normTopkProb: Bool,
        on cmd: MTLCommandBuffer
    ) {
        precondition(t > 0, "Ops.moeRouterTopKMany: t must be positive")
        precondition(
            logits.elementCount == t * nExperts,
            "Ops.moeRouterTopKMany: logits must be [T·nExperts]")
        precondition(
            indicesOut.elementCount == t * k && indicesOut.dtype == .u32,
            "Ops.moeRouterTopKMany: indicesOut must be [T·k] u32")
        precondition(
            weightsOut.elementCount == t * k && weightsOut.dtype == logits.dtype,
            "Ops.moeRouterTopKMany: weightsOut must be [T·k] matching logits dtype")
        // Kernel pins tpg = 32 (one simdgroup per token row, Reduction
        // mode). Grid total threads = T · 32 → T threadgroups.
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let grid = MTLSize(width: t * 32, height: 1, depth: 1)
        let normFlag: UInt32 = normTopkProb ? 1 : 0
        switch logits.dtype {
        case .f32:
            IronKernels.iron_moe_router_topk_f32(
                router_logits: logits.buffer, router_logitsOffset: logits.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k), norm_topk_prob: normFlag,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_router_topk_f16(
                router_logits: logits.buffer, router_logitsOffset: logits.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k), norm_topk_prob: normFlag,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_router_topk_bf16(
                router_logits: logits.buffer, router_logitsOffset: logits.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k), norm_topk_prob: normFlag,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.moeRouterTopKMany: unsupported logits dtype \(logits.dtype)")
        }
    }

    /// GPU argmax over a 1D logits tensor. Caller supplies a 1-element
    /// u32 output buffer. Uses the cooperative 256-thread Reduction
    /// kernel — one threadgroup, ~80-300 KB / vocab logits in registers.
    /// Mamba 2 / Mamba 1D depthwise causal-conv step — streaming
    /// decode form. One thread per channel. `state` is the rolling
    /// window of the last `kernelSize - 1` inputs (shape
    /// `[kernelSize - 1, nChannels]`); shifted in-place after compute.
    /// Activation (Mamba 2 follows the conv with SiLU) is the caller's
    /// concern — kept separate for composability.
    public static func conv1dCausalStep(
        x: Tensor, w: Tensor, b: Tensor,
        state: Tensor, into y: Tensor,
        nChannels: Int, kernelSize: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            w.dtype == x.dtype && b.dtype == x.dtype
                && state.dtype == x.dtype && y.dtype == x.dtype,
            "Ops.conv1dCausalStep: every tensor must share dtype")
        let grid = MTLSize(width: nChannels, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            IronKernels.iron_conv1d_causal_step_f32(
                x: x.buffer, xOffset: x.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state: state.buffer, stateOffset: state.offset,
                y: y.buffer, yOffset: y.offset,
                n_channels: UInt32(nChannels), kernel_size: UInt32(kernelSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_conv1d_causal_step_f16(
                x: x.buffer, xOffset: x.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state: state.buffer, stateOffset: state.offset,
                y: y.buffer, yOffset: y.offset,
                n_channels: UInt32(nChannels), kernel_size: UInt32(kernelSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_conv1d_causal_step_bf16(
                x: x.buffer, xOffset: x.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state: state.buffer, stateOffset: state.offset,
                y: y.buffer, yOffset: y.offset,
                n_channels: UInt32(nChannels), kernel_size: UInt32(kernelSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.conv1dCausalStep: unsupported dtype \(x.dtype)")
        }
    }

    /// Batched depthwise causal conv1d + SiLU + cast-to-f32. Sweeps
    /// `t` tokens in ONE dispatch with the conv state held in
    /// per-channel registers across the sweep — `stateIn` is read
    /// once at the top, `stateOut` is written once at the bottom.
    /// Replaces the per-token `conv1dCausalStep` + `siluCastToF32`
    /// T-loop in `Qwen35GDNMixer.forwardManyChunked`.
    ///
    /// `convKernel` must be 4 (Qwen3.5 / 3.6, Mamba 2, NemotronH all
    /// use kernel=4; the kernel signature carries it as a constexpr
    /// for shape uniformity but the body is K=4 hardcoded).
    ///
    /// State in-place safety: each grid thread owns one channel;
    /// reads `stateIn[*, c]` once at the start of the sweep and
    /// writes `stateOut[*, c]` once at the end. Passing the same
    /// buffer for both is safe.
    /// `statePlanes` / `planesEnabled` — spec-decode rollback side channel,
    /// same contract as `gatedDeltaPrepChunk`'s (see
    /// `iron_conv1d_causal_step_silu_cast_many`'s "Per-token conv-state
    /// planes" doc). `statePlanes` must be `[T, conv_kernel-1, convDim]`
    /// when enabled; pass `Ops.disabledGdnPlanes(dtype:device:)` otherwise.
    public static func conv1dCausalStepSiluCastMany(
        src: Tensor, w: Tensor, b: Tensor,
        stateIn: Tensor, outF32: Tensor, stateOut: Tensor,
        statePlanes: Tensor, planesEnabled: Tensor,
        t: Int, convDim: Int, convKernel: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            convKernel == 4,
            "Ops.conv1dCausalStepSiluCastMany: requires conv_kernel = 4")
        precondition(
            src.dtype == w.dtype && src.dtype == b.dtype
                && src.dtype == stateIn.dtype && src.dtype == stateOut.dtype
                && src.dtype == statePlanes.dtype,
            "Ops.conv1dCausalStepSiluCastMany: dtype mismatch")
        precondition(
            outF32.dtype == .f32,
            "Ops.conv1dCausalStepSiluCastMany: outF32 must be .f32")
        precondition(
            planesEnabled.dtype == .u32 && planesEnabled.elementCount == 1,
            "Ops.conv1dCausalStepSiluCastMany: planesEnabled must be a [1] u32 scalar buffer")
        let grid = MTLSize(width: convDim, height: 1, depth: 1)
        let tg = MTLSize(width: min(convDim, 256), height: 1, depth: 1)
        let tLenU = UInt32(t)
        let cdU = UInt32(convDim)
        let ckU = UInt32(convKernel)
        switch src.dtype {
        case .f32:
            IronKernels.iron_conv1d_causal_step_silu_cast_many_f32(
                src: src.buffer, srcOffset: src.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                out_f32: outF32.buffer, out_f32Offset: outF32.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                t_len: tLenU, conv_dim: cdU, conv_kernel: ckU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_conv1d_causal_step_silu_cast_many_f16(
                src: src.buffer, srcOffset: src.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                out_f32: outF32.buffer, out_f32Offset: outF32.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                t_len: tLenU, conv_dim: cdU, conv_kernel: ckU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_conv1d_causal_step_silu_cast_many_bf16(
                src: src.buffer, srcOffset: src.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                out_f32: outF32.buffer, out_f32Offset: outF32.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                t_len: tLenU, conv_dim: cdU, conv_kernel: ckU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.conv1dCausalStepSiluCastMany: unsupported dtype \(src.dtype)")
        }
    }

    /// Mamba 2 selective-scan single-token decode step. Updates the
    /// per-layer recurrent state `h` in place and writes the output
    /// channel vector `y`. `h` lives in fp32 (state accumulates over
    /// many decode steps; bf16's 7-bit mantissa drifts fast). One
    /// thread per `(head, channel)` — total `nHeads * headDim` threads.
    ///
    /// See `SSMStateCache` for the storage class that wraps the per-layer
    /// `h` buffer; Mamba 2 family files call this through
    /// that cache.
    public static func ssmStep(
        x: Tensor, a: Tensor, b: Tensor, c: Tensor, dt: Tensor,
        state h: Tensor, into y: Tensor,
        nHeads: Int, headDim: Int, stateDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(h.dtype == .f32, "Ops.ssmStep: state h must be f32")
        precondition(x.dtype == y.dtype, "Ops.ssmStep: x and y dtype must match")
        precondition(
            a.dtype == x.dtype && b.dtype == x.dtype
                && c.dtype == x.dtype && dt.dtype == x.dtype,
            "Ops.ssmStep: a/b/c/dt dtype must match x")
        let grid = MTLSize(width: nHeads * headDim, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            IronKernels.iron_ssm_step_f32(
                x: x.buffer, xOffset: x.offset,
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: c.buffer, cOffset: c.offset,
                dt: dt.buffer, dtOffset: dt.offset,
                h: h.buffer, hOffset: h.offset,
                y: y.buffer, yOffset: y.offset,
                head_dim: UInt32(headDim), state_dim: UInt32(stateDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_ssm_step_f16(
                x: x.buffer, xOffset: x.offset,
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: c.buffer, cOffset: c.offset,
                dt: dt.buffer, dtOffset: dt.offset,
                h: h.buffer, hOffset: h.offset,
                y: y.buffer, yOffset: y.offset,
                head_dim: UInt32(headDim), state_dim: UInt32(stateDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_ssm_step_bf16(
                x: x.buffer, xOffset: x.offset,
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: c.buffer, cOffset: c.offset,
                dt: dt.buffer, dtOffset: dt.offset,
                h: h.buffer, hOffset: h.offset,
                y: y.buffer, yOffset: y.offset,
                head_dim: UInt32(headDim), state_dim: UInt32(stateDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.ssmStep: unsupported dtype \(x.dtype)")
        }
    }

    /// Gated Delta Net (GDN) recurrent single-token decode step.
    ///
    /// Computes, per decode step and per value head, the recurrence
    ///   S_t = g_t · S_{t-1} + β_t · k_t · (v_t − k_tᵀ · S_{t-1})ᵀ
    /// then the output `o_t = S_t · q_t`. The per-head state matrix
    /// `S [Hv, Dv, Dk]` stays fp32 throughout — the gate + rank-1
    /// update accumulate across many steps and bf16 drifts fast.
    ///
    /// The underlying kernel is **reduction-mode** with strict
    /// dispatch-shape invariants (see
    /// `crates/iron-std/src/butter/gated_delta.rs`
    /// §"DISPATCH INVARIANTS" and `OpsValidation.validateGatedDeltaStep`):
    /// TPG = 32 (one simdgroup), one threadgroup per `(dv, n)` pair,
    /// `Dk % 32 == 0`, `Hv % Hk == 0`. The `(Dk, Dv, Hk, Hv)` dimensions
    /// are runtime `#[constexpr]` scalars — a single PSO serves every
    /// model config, no per-tuple specialization.
    ///
    /// `q` / `k` are expected pre-normalised (the GDN block applies the
    /// rmsNorm + scale before calling this — the standard, non-fused
    /// kernel variant). The kernel reads `stateIn` and writes a distinct
    /// `stateOut`; callers double-buffer via `GDNStateCache.swap()`.
    ///
    /// All tensors are f32 — the kernel runs the recurrence state in f32
    /// regardless of activation dtype. This wraps `iron_gated_delta_step`,
    /// which performs exactly one recurrence step; multi-step prefill
    /// loops this call per token at the caller.
    public static func gatedDeltaStep(
        q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
        stateIn: Tensor, into y: Tensor, stateOut: Tensor,
        numKeyHeads: Int, numValueHeads: Int,
        keyHeadDim: Int, valueHeadDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        // Kernel-invariant validation — see OpsValidation.swift. A bad
        // dispatch shape on a reduction kernel ranges from silent
        // miscompute to a non-preemptive GPU pin.
        if let reason = OpsValidation.validateGatedDeltaStep(
            keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim,
            numKeyHeads: numKeyHeads, numValueHeads: numValueHeads
        ) {
            preconditionFailure("Ops.gatedDeltaStep: \(reason)")
        }
        precondition(
            q.dtype == .f32 && k.dtype == .f32 && v.dtype == .f32
                && g.dtype == .f32 && beta.dtype == .f32,
            "Ops.gatedDeltaStep: q/k/v/g/beta must be f32")
        precondition(
            stateIn.dtype == .f32 && stateOut.dtype == .f32
                && y.dtype == .f32,
            "Ops.gatedDeltaStep: state + output tensors must be f32")

        // Dispatch derived from the invariants: 32 threads per group
        // (one simdgroup), one group per (dv_idx, n) pair. The kernel
        // reads tid as the lane (dk_idx), tgid_x as dv_idx, tgid_y as n
        // = batch·Hv + hv. Decode is single-batch so n ranges [0, Hv).
        // Generated bindings use `dispatchThreads`, so the grid is
        // counted in THREADS: width = Dv · 32, height = Hv.
        let lanesPerGroup = 32
        let grid = MTLSize(
            width: valueHeadDim * lanesPerGroup,
            height: numValueHeads, depth: 1)
        let tg = MTLSize(width: lanesPerGroup, height: 1, depth: 1)

        // `iron_gated_delta_step` is a single PSO: the (Dk, Dv, Hv, Hk)
        // dimensions are runtime `#[constexpr]` scalars rather than
        // baked compile-time constants, so one dispatch serves every
        // model config.
        IronKernels.iron_gated_delta_step_f32(
            q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
            v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
            beta: beta.buffer, betaOffset: beta.offset,
            state_in: stateIn.buffer, state_inOffset: stateIn.offset,
            state_out: stateOut.buffer, state_outOffset: stateOut.offset,
            y: y.buffer, yOffset: y.offset,
            dk: UInt32(keyHeadDim), dv: UInt32(valueHeadDim),
            hv: UInt32(numValueHeads), hk: UInt32(numKeyHeads),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    public static func argmax(
        _ logits: Tensor, into out: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(out.dtype == .u32, "Ops.argmax: output must be u32")
        precondition(out.elementCount == 1, "Ops.argmax: output must be a single element")
        let n = logits.elementCount
        // Reduction: dispatch 256 threads in 1 group (full simdgroups).
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: 256, height: 1, depth: 1)
        switch logits.dtype {
        case .f32:
            IronKernels.iron_argmax_f32(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_argmax_f16(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_argmax_bf16(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.argmax: unsupported dtype \(logits.dtype)")
        }
    }

    /// GPU softmax + categorical sample over a 1D logits tensor.
    /// Caller supplies:
    ///   - 1-element u32 output buffer for the sampled token id
    ///   - 1-element f32 temperature buffer (must be > 0; T=0 should
    ///     route to `argmax` instead)
    ///   - 1-element f32 uniform draw in [0, 1) — CPU-generated each
    ///     decode step from `GenerationParameters.makeRNG()`
    ///
    /// Used by the Generate decode loop for the pure-temperature
    /// sampling path (T > 0, no top-K / top-P / min-P / rep-penalty)
    /// to keep logits on the GPU. Top-K / top-P / min-P /
    /// rep-penalty still flow through the CPU `Sampling.sample(...)`
    /// path until separate filter kernels land.
    public static func softmaxCategoricalSample(
        _ logits: Tensor,
        into out: Tensor,
        temperature: Tensor,
        uniform: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(out.dtype == .u32, "Ops.softmaxCategoricalSample: output must be u32")
        precondition(
            out.elementCount == 1, "Ops.softmaxCategoricalSample: output must be a single element")
        precondition(
            temperature.dtype == .f32 && temperature.elementCount == 1,
            "Ops.softmaxCategoricalSample: temperature must be a 1-element f32 tensor")
        precondition(
            uniform.dtype == .f32 && uniform.elementCount == 1,
            "Ops.softmaxCategoricalSample: uniform must be a 1-element f32 tensor")
        let n = logits.elementCount
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: 256, height: 1, depth: 1)
        switch logits.dtype {
        case .f32:
            IronKernels.iron_softmax_categorical_sample_f32(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                temperature_in: temperature.buffer, temperature_inOffset: temperature.offset,
                uniform_in: uniform.buffer, uniform_inOffset: uniform.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_softmax_categorical_sample_f16(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                temperature_in: temperature.buffer, temperature_inOffset: temperature.offset,
                uniform_in: uniform.buffer, uniform_inOffset: uniform.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_softmax_categorical_sample_bf16(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                temperature_in: temperature.buffer, temperature_inOffset: temperature.offset,
                uniform_in: uniform.buffer, uniform_inOffset: uniform.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.softmaxCategoricalSample: unsupported dtype \(logits.dtype)")
        }
    }

    /// Affine-quantize one K (or V) row into an int4 or int8 KV cache
    /// slot. Dispatches to the right `quantize_kv_int{4,8}` kernel
    /// based on `bits`.
    public static func quantizeKVAffine(
        src: Tensor,
        weights: Tensor, scales: Tensor, biases: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, position: Int, bits: Int,
        on cmd: MTLCommandBuffer
    ) {
        switch bits {
        case 4:
            quantizeKVInt4(
                src: src, weights: weights, scales: scales, biases: biases,
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                groupSize: groupSize, position: position, on: cmd)
        case 8:
            quantizeKVInt8(
                src: src, weights: weights, scales: scales, biases: biases,
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                groupSize: groupSize, position: position, on: cmd)
        default:
            fatalError("Ops.quantizeKVAffine: unsupported bits=\(bits) (use 4 or 8)")
        }
    }

    /// Bulk-dequant int4/int8 KV cache → working buffer. Dispatches to
    /// the right `bulk_dequant_kv_int{4,8}` kernel based on `bits`.
    public static func bulkDequantKVAffine(
        weights: Tensor, scales: Tensor, biases: Tensor,
        into out: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, nPositions: Int, bits: Int,
        on cmd: MTLCommandBuffer
    ) {
        switch bits {
        case 4:
            bulkDequantKVInt4(
                weights: weights, scales: scales, biases: biases,
                into: out, nKVHeads: nKVHeads, headDim: headDim,
                maxSeq: maxSeq, groupSize: groupSize,
                nPositions: nPositions, on: cmd)
        case 8:
            bulkDequantKVInt8(
                weights: weights, scales: scales, biases: biases,
                into: out, nKVHeads: nKVHeads, headDim: headDim,
                maxSeq: maxSeq, groupSize: groupSize,
                nPositions: nPositions, on: cmd)
        default:
            fatalError("Ops.bulkDequantKVAffine: unsupported bits=\(bits) (use 4 or 8)")
        }
    }

    /// Affine-quantize one K (or V) row into an int4 KV cache slot.
    /// Packs 8 nibbles per uint32. One thread per group.
    public static func quantizeKVInt4(
        src: Tensor,
        weights: Tensor, scales: Tensor, biases: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, position: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(weights.dtype == .u32, "quantizeKVInt4: weights must be u32")
        precondition(
            scales.dtype == src.dtype && biases.dtype == src.dtype,
            "quantizeKVInt4: scales/biases dtype must match src")
        // Kernel-invariant validation (silent-miscompute footguns:
        // partial trailing group, unaligned pack stride). See
        // OpsValidation.validateQuantizeKV.
        if let reason = OpsValidation.validateQuantizeKV(
            nKVHeads: nKVHeads, headDim: headDim, groupSize: groupSize, bits: 4
        ) {
            preconditionFailure("Ops.quantizeKVInt4: \(reason)")
        }
        let groupsPerHead = headDim / groupSize
        let grid = MTLSize(width: nKVHeads * groupsPerHead, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch src.dtype {
        case .f32:
            IronKernels.iron_quantize_kv_int4_f32(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_quantize_kv_int4_f16(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_quantize_kv_int4_bf16(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.quantizeKVInt4: unsupported dtype \(src.dtype)")
        }
    }

    /// Bulk-dequant the live slice of an int4 KV cache into a
    /// working buffer for SDPA. Unpacks 8 nibbles per uint32.
    public static func bulkDequantKVInt4(
        weights: Tensor, scales: Tensor, biases: Tensor,
        into out: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, nPositions: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(weights.dtype == .u32, "bulkDequantKVInt4: weights must be u32")
        precondition(
            scales.dtype == out.dtype && biases.dtype == out.dtype,
            "bulkDequantKVInt4: scales/biases dtype must match output")
        // Kernel-invariant validation — shares the int4 group/pack
        // alignment contract with quantizeKVInt4. See
        // OpsValidation.validateQuantizeKV.
        if let reason = OpsValidation.validateQuantizeKV(
            nKVHeads: nKVHeads, headDim: headDim, groupSize: groupSize, bits: 4
        ) {
            preconditionFailure("Ops.bulkDequantKVInt4: \(reason)")
        }
        let total = nKVHeads * nPositions * headDim
        let grid = MTLSize(width: total, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch out.dtype {
        case .f32:
            IronKernels.iron_bulk_dequant_kv_int4_f32(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_bulk_dequant_kv_int4_f16(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_bulk_dequant_kv_int4_bf16(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.bulkDequantKVInt4: unsupported dtype \(out.dtype)")
        }
    }

    /// Affine-quantize one K (or V) row into an int8 KV cache slot.
    /// `src` is `[nKVHeads, headDim]` in fp16/bf16; outputs are the
    /// cache's packed weights + per-group scales + biases (see
    /// `AffineQuantizedKVCache`). One thread per group.
    public static func quantizeKVInt8(
        src: Tensor,
        weights: Tensor, scales: Tensor, biases: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, position: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(weights.dtype == .u32, "quantizeKVInt8: weights must be u32")
        precondition(
            scales.dtype == src.dtype && biases.dtype == src.dtype,
            "quantizeKVInt8: scales/biases dtype must match src")
        // Kernel-invariant validation (silent-miscompute footguns:
        // partial trailing group, unaligned pack stride). See
        // OpsValidation.validateQuantizeKV.
        if let reason = OpsValidation.validateQuantizeKV(
            nKVHeads: nKVHeads, headDim: headDim, groupSize: groupSize, bits: 8
        ) {
            preconditionFailure("Ops.quantizeKVInt8: \(reason)")
        }
        let groupsPerHead = headDim / groupSize
        let grid = MTLSize(width: nKVHeads * groupsPerHead, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch src.dtype {
        case .f32:
            IronKernels.iron_quantize_kv_int8_f32(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_quantize_kv_int8_f16(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_quantize_kv_int8_bf16(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.quantizeKVInt8: unsupported dtype \(src.dtype)")
        }
    }

    /// Bulk-dequantize the live slice of an int8 KV cache into a
    /// working buffer that SDPA can read directly. Output buffer
    /// shape `[nKVHeads, maxSeq, headDim]`; only positions `[0,
    /// nPositions)` are written. One thread per output element.
    public static func bulkDequantKVInt8(
        weights: Tensor, scales: Tensor, biases: Tensor,
        into out: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, nPositions: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(weights.dtype == .u32, "bulkDequantKVInt8: weights must be u32")
        precondition(
            scales.dtype == out.dtype && biases.dtype == out.dtype,
            "bulkDequantKVInt8: scales/biases dtype must match output")
        // Kernel-invariant validation — shares the int8 group/pack
        // alignment contract with quantizeKVInt8. See
        // OpsValidation.validateQuantizeKV.
        if let reason = OpsValidation.validateQuantizeKV(
            nKVHeads: nKVHeads, headDim: headDim, groupSize: groupSize, bits: 8
        ) {
            preconditionFailure("Ops.bulkDequantKVInt8: \(reason)")
        }
        let total = nKVHeads * nPositions * headDim
        let grid = MTLSize(width: total, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch out.dtype {
        case .f32:
            IronKernels.iron_bulk_dequant_kv_int8_f32(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_bulk_dequant_kv_int8_f16(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_bulk_dequant_kv_int8_bf16(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.bulkDequantKVInt8: unsupported dtype \(out.dtype)")
        }
    }

    /// Append one timestep to a KV cache on the GPU. `src` is
    /// [nKVHeads, headDim] (rotated K or V for the current token);
    /// `cache` is the full [nKVHeads, maxSeq, headDim] buffer.
    /// `position` is the slot to write into. Replaces the CPU-side
    /// memcpy + mid-layer commit/wait pattern from the initial cut.
    public static func kvCacheUpdate(
        src: Tensor, into cache: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int, position: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(src.dtype == cache.dtype, "kvCacheUpdate: dtype mismatch")
        let total = nKVHeads * headDim
        let (grid, tg) = elementwiseGrid(total)
        let hd = UInt32(headDim)
        let ms = UInt32(maxSeq)
        let pos = UInt32(position)
        switch src.dtype {
        case .f32:
            IronKernels.iron_kv_cache_update_f32(
                src: src.buffer, srcOffset: src.offset,
                out: cache.buffer, outOffset: cache.offset,
                head_dim: hd, max_seq: ms, position: pos,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_kv_cache_update_f16(
                src: src.buffer, srcOffset: src.offset,
                out: cache.buffer, outOffset: cache.offset,
                head_dim: hd, max_seq: ms, position: pos,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_kv_cache_update_bf16(
                src: src.buffer, srcOffset: src.offset,
                out: cache.buffer, outOffset: cache.offset,
                head_dim: hd, max_seq: ms, position: pos,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.kvCacheUpdate: unsupported dtype \(src.dtype)")
        }
    }

    /// Append the current token's K AND V rows to their caches in ONE
    /// compute encoder. The cache buffers, layout, and writing thread
    /// are identical to `kvCacheUpdate` — this just amortises the
    /// encoder begin/end across both projections (saving one pair per
    /// attention layer per decode token).
    public static func kvCacheUpdateKV(
        kSrc: Tensor, kCache: Tensor,
        vSrc: Tensor, vCache: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int, position: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            kSrc.dtype == kCache.dtype && vSrc.dtype == vCache.dtype
                && kSrc.dtype == vSrc.dtype,
            "Ops.kvCacheUpdateKV: dtype mismatch")
        let total = nKVHeads * headDim
        let (grid, tg) = elementwiseGrid(total)
        let hd = UInt32(headDim)
        let ms = UInt32(maxSeq)
        let pos = UInt32(position)
        let psoName: String
        switch kSrc.dtype {
        case .f32: psoName = "iron_kv_cache_update_f32"
        case .f16: psoName = "iron_kv_cache_update_f16"
        case .bf16: psoName = "iron_kv_cache_update_bf16"
        default: fatalError("Ops.kvCacheUpdateKV: unsupported dtype \(kSrc.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        @inline(__always)
        func dispatch(_ src: Tensor, _ cache: Tensor) {
            enc.setBuffer(src.buffer, offset: src.offset, index: 0)
            enc.setBuffer(cache.buffer, offset: cache.offset, index: 1)
            var headDimV = hd
            var maxSeqV = ms
            var positionV = pos
            enc.setBytes(&headDimV, length: 4, index: 2)
            enc.setBytes(&maxSeqV, length: 4, index: 3)
            enc.setBytes(&positionV, length: 4, index: 4)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(kSrc, kCache)
        dispatch(vSrc, vCache)
        enc.endEncoding()
    }

    /// Batched K + V cache append: writes T tokens' K rows AND V rows
    /// into their respective caches in ONE shared encoder (2 batched
    /// dispatches). Each per-token write goes to a slot indexed by
    /// `positions[t]`. Replaces the per-token `kvCacheUpdateKV` loop
    /// at `decodeMany` / `forwardMany` prefill — at T=512 × 10 attn
    /// layers that's ≈ 5100 fewer dispatches per prefill call.
    ///
    /// Shapes:
    ///   - `kSrc` / `vSrc`: `[T, nKVHeads, headDim]` (flat).
    ///   - `kCache` / `vCache`: `[nKVHeads, maxSeq, headDim]`.
    ///   - `positions`: `[T]` u32 slot indices into the caches.
    public static func kvCacheUpdateKVMany(
        kSrc: Tensor, kCache: Tensor,
        vSrc: Tensor, vCache: Tensor,
        positions: Tensor, t: Int,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            kSrc.dtype == kCache.dtype && vSrc.dtype == vCache.dtype
                && kSrc.dtype == vSrc.dtype,
            "Ops.kvCacheUpdateKVMany: dtype mismatch")
        precondition(
            positions.dtype == .u32,
            "Ops.kvCacheUpdateKVMany: positions must be .u32")
        precondition(
            positions.elementCount == t,
            "Ops.kvCacheUpdateKVMany: positions count must equal T")
        let elementsPerRow = nKVHeads * headDim
        let total = t * elementsPerRow
        let (grid, tg) = elementwiseGrid(total)
        let psoName: String
        switch kSrc.dtype {
        case .f32: psoName = "iron_kv_cache_update_many_f32"
        case .f16: psoName = "iron_kv_cache_update_many_f16"
        case .bf16: psoName = "iron_kv_cache_update_many_bf16"
        default: fatalError("Ops.kvCacheUpdateKVMany: unsupported dtype \(kSrc.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        // Shared bindings: positions + scalars set ONCE.
        enc.setBuffer(positions.buffer, offset: positions.offset, index: 1)
        var hd = UInt32(headDim)
        var ms = UInt32(maxSeq)
        var nhd = UInt32(elementsPerRow)
        enc.setBytes(&hd, length: 4, index: 3)
        enc.setBytes(&ms, length: 4, index: 4)
        enc.setBytes(&nhd, length: 4, index: 5)
        @inline(__always)
        func dispatch(_ src: Tensor, _ cache: Tensor) {
            enc.setBuffer(src.buffer, offset: src.offset, index: 0)
            enc.setBuffer(cache.buffer, offset: cache.offset, index: 2)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(kSrc, kCache)
        dispatch(vSrc, vCache)
        enc.endEncoding()
    }

    /// SDPA decode. q: [n_q_heads, head_dim]. k/v cache layout:
    /// [n_kv_heads, kv_stride, head_dim] where kv_stride is the physical
    /// capacity (maxSeq) and nKV is how many positions to attend to.
    /// Output: [n_q_heads, head_dim].
    ///
    /// Dispatch invariants (see `crates/iron-std/src/butter/sdpa_decode.rs`
    /// + `sdpa_decode_d64.rs`):
    ///   * `head_dim ∈ {64, 128}` — one threadgroup is 32 simdgroups ×
    ///     32 lanes; each lane owns `head_dim / 32` consecutive Q/K/V
    ///     elements. 128 covers full attention heads on Llama 3.2 3B+ /
    ///     Qwen3 / GPT-OSS full layers; 64 covers Llama 3.2 1B and
    ///     GPT-OSS sliding-window layers. head_dim=256 (some Gemma
    ///     configs) is queued but not yet emitted.
    ///   * 1 threadgroup per Q head, 1024 threads per threadgroup
    ///     (32 simdgroups × 32 lanes). `tgid_x = q_head`.
    ///
    /// Sliding-window / attention-sink fast path (`head_dim = 128`
    /// variants only — iron PR #50): when `sinkEnd` and/or
    /// `windowStart` are non-zero the kernel attends `[0, sinkEnd)`
    /// (the pinned attention sinks) plus `[windowStart, nKV)` (the
    /// sliding window) and skips the masked range
    /// `[sinkEnd, windowStart)` at the loop-bound level — no
    /// per-position branching. Both default to 0, which is exactly
    /// dense full attention over `[0, nKV)`. Callers with a windowed
    /// KV cache derive these from the eviction policy: for a window of
    /// `W` retained positions with `S` pinned sink slots,
    /// `windowStart = max(0, nKV - W)` and `sinkEnd = S`. The d64 /
    /// d256 kernel variants are dense-only — passing non-zero sink /
    /// window with those head dims is a precondition failure.
    public static func sdpaDecode(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        nKV: Int, kvStride: Int,
        scale: Float, on cmd: MTLCommandBuffer,
        sinkEnd: Int = 0, windowStart: Int = 0,
        into out: Tensor? = nil
    ) -> Tensor {
        // Kernel-invariant validation — see OpsValidation.swift for the
        // full reasoning + CI-runnable tests. The 2026-05-19 GPU freeze
        // came from this wrapper accepting head_dim=4 with no check.
        if let reason = OpsValidation.validateSdpaDecode(
            headDim: headDim, nQHeads: nQHeads, nKVHeads: nKVHeads,
            nKV: nKV, kvStride: kvStride,
            sinkEnd: sinkEnd, windowStart: windowStart
        ) {
            preconditionFailure("Ops.sdpaDecode: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: [nQHeads, headDim], dtype: q.dtype)
        // One threadgroup per q-head. Reduction-mode kernel reads `tgid_x`
        // as the q-head; we lay out a 1D thread grid of
        // `nQHeads * threadsPerGroup` threads so Metal slices it into
        // `nQHeads` threadgroups.
        //
        // The d64/d128/d256 variants run 1024 threads (32 simdgroups).
        // The d512 variant runs 512 (16 simdgroups): its 16-wide per-lane
        // register footprint pushes the pipeline's
        // maxTotalThreadsPerThreadgroup below 1024, and a 1024-thread
        // dispatch silently no-ops (command buffer errors, output stays
        // zero). The kernel body is parametric in `n_simd`, so 512 is
        // correct — see sdpa_decode_d512.rs §"DISPATCH INVARIANTS".
        let threadsPerGroup = headDim == 512 ? 512 : 1024
        let grid = MTLSize(width: nQHeads * threadsPerGroup, height: 1, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let headsPerGroup = nQHeads / nKVHeads

        // Route to the head_dim-specialized kernel. Each variant has its
        // own per-lane width baked into the kernel (head_dim=128 → 4
        // elt/lane, head_dim=64 → 2 elt/lane). A generic kernel would
        // either lose register efficiency or trip up the codegen's
        // vectorize pass; per-head-dim variants keep both kernels fast.
        switch (headDim, q.dtype) {
        case (128, .f32):
            IronKernels.iron_sdpa_decode_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                // Sliding-window / sink fast path. Both 0 → dense full
                // attention over [0, n_kv); non-zero values let the
                // kernel skip the masked range [sink_end, window_start)
                // at the loop-bound level. See the doc comment above.
                sink_end: UInt32(sinkEnd), window_start: UInt32(windowStart),
                // has_sink: 0 → no learned attention-sink logit folded
                // into the softmax denominator (GPT-OSS-style sinks).
                // Ops.sdpaDecode does not yet expose a sink-logit param;
                // 0 reproduces the pre-#145 dense-attention behavior.
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (128, .f16):
            IronKernels.iron_sdpa_decode_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                sink_end: UInt32(sinkEnd), window_start: UInt32(windowStart),
                // has_sink: 0 → dense softmax denominator (see d128 f32
                // case above for the GPT-OSS attention-sink rationale).
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (128, .bf16):
            IronKernels.iron_sdpa_decode_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                sink_end: UInt32(sinkEnd), window_start: UInt32(windowStart),
                // has_sink: 0 → dense softmax denominator (see d128 f32
                // case above for the GPT-OSS attention-sink rationale).
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        // NOTE: the d64/d256 specialized kernels in iron-std
        // don't accept `sink_end` / `window_start` parameters — they're
        // perf-tuned for dense full attention. Callers needing sliding-
        // window / attention-sink at these head_dims must precondition
        // `sinkEnd == 0 && windowStart == 0` (validated upstream by
        // OpsValidation.validateSdpaDecode). The d128 / d512 cases use
        // the generic `iron_sdpa_decode_*` kernel with constexpr
        // head_dim which DOES carry both params.
        case (64, .f32):
            IronKernels.iron_sdpa_decode_d64_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, .f16):
            IronKernels.iron_sdpa_decode_d64_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, .bf16):
            IronKernels.iron_sdpa_decode_d64_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (256, .f32):
            IronKernels.iron_sdpa_decode_d256_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (256, .f16):
            IronKernels.iron_sdpa_decode_d256_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (256, .bf16):
            IronKernels.iron_sdpa_decode_d256_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        // d512 routes to the dedicated `iron_sdpa_decode_d512_*` kernel.
        // iron #145 auto-discovers `sdpa_decode_d512.rs`, so the
        // per-head-dim kernel is emitted again — the generic-kernel
        // reroute used while it was missing produced all-zero output
        // past offset 128 (the generic body's per-lane width assumes
        // head_dim ≤ 128). Like d64/d256, the d512 kernel is perf-tuned
        // for dense full attention and does not accept sink params;
        // OpsValidation preconditions `sinkEnd == 0 && windowStart == 0`.
        case (512, .f32):
            IronKernels.iron_sdpa_decode_d512_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (512, .f16):
            IronKernels.iron_sdpa_decode_d512_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (512, .bf16):
            IronKernels.iron_sdpa_decode_d512_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            // OpsValidation rejected anything we don't have a kernel for;
            // an unsupported dtype lands here.
            fatalError("Ops.sdpaDecode: unsupported (head_dim=\(headDim), dtype=\(q.dtype))")
        }
        return result
    }

    /// Two-pass Flash-Decoding SDPA. Pass 1 partitions the KV cache
    /// into `blocks` even tiles and writes a per-block partial
    /// `(O, max, lse)` triple; pass 2 merges them with the log-sum-exp
    /// trick. Wins over single-pass `sdpaDecode` at long KV (≥ 1K
    /// typical), where the single-pass kernel starves the GPU with
    /// one threadgroup per Q-head.
    ///
    /// **Block-count contract.** The pass2 kernel hardcodes a 32-wide
    /// reduction (`bn = 32` per lane) over `blocks / 32` chunks, so
    /// `blocks` MUST be a multiple of 32 and at least 32. Passing
    /// `blocks < 32` causes pass2 to iterate zero chunks and emit an
    /// all-zero output silently — the precondition below catches that.
    /// 32 is the standard short-context tile; 64 / 96 / 128 are
    /// reasonable for very long contexts.
    ///
    /// **Scratch buffer sizing.** Caller owns the partials; reuse
    /// across decode steps is safe because the shape depends only on
    /// `(nQHeads, headDim, blocks)`:
    ///   - `partialO` : `[nQHeads, blocks, headDim]` matching `q.dtype`
    ///   - `partialM` : `[nQHeads, blocks]` fp32 (running max)
    ///   - `partialL` : `[nQHeads, blocks]` fp32 (running lse)
    public static func sdpaDecode2Pass(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        nKV: Int, kvStride: Int, blocks: Int,
        scale: Float,
        partialO: Tensor, partialM: Tensor, partialL: Tensor,
        into out: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            nQHeads % nKVHeads == 0,
            "Ops.sdpaDecode2Pass: nQHeads must be a multiple of nKVHeads")
        let gqaFactor = nQHeads / nKVHeads
        precondition(
            blocks >= 32 && blocks % 32 == 0,
            "Ops.sdpaDecode2Pass: blocks (\(blocks)) must be a multiple of 32 "
                + "and ≥ 32 — pass2 hardcodes a 32-lane block-merge reduction")
        precondition(
            partialO.elementCount == nQHeads * blocks * headDim,
            "Ops.sdpaDecode2Pass: partialO must be [nQHeads, blocks, headDim]")
        precondition(
            partialM.elementCount == nQHeads * blocks && partialM.dtype == .f32,
            "Ops.sdpaDecode2Pass: partialM must be [nQHeads, blocks] f32")
        precondition(
            partialL.elementCount == nQHeads * blocks && partialL.dtype == .f32,
            "Ops.sdpaDecode2Pass: partialL must be [nQHeads, blocks] f32")
        precondition(
            partialO.dtype == q.dtype && out.dtype == q.dtype,
            "Ops.sdpaDecode2Pass: partialO / out dtype must match q")

        // Pass 1: per-block partial O / m / l. One TG per
        // (kv_head, block) with `gqa_factor` simdgroups × 32 lanes,
        // each lane owning `headDim / 32` consecutive elements.
        let pass1TgWidth = gqaFactor * 32
        let pass1Grid = MTLSize(
            width: nKVHeads * pass1TgWidth, height: blocks, depth: 1)
        let pass1Tg = MTLSize(width: pass1TgWidth, height: 1, depth: 1)

        // Pass 2: merge partials across blocks. One TG per Q-head with
        // 32 simdgroups × 32 lanes = 1024 threads.
        let pass2TgWidth = 1024
        let pass2Grid = MTLSize(
            width: nQHeads * pass2TgWidth, height: 1, depth: 1)
        let pass2Tg = MTLSize(width: pass2TgWidth, height: 1, depth: 1)

        switch q.dtype {
        case .f32:
            IronKernels.iron_sdpa_decode_2pass_pass1_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                gqa_factor: UInt32(gqaFactor), blocks: UInt32(blocks),
                scale: scale,
                gridSize: pass1Grid, threadgroupSize: pass1Tg, on: cmd)
            IronKernels.iron_sdpa_decode_2pass_pass2_f32(
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), blocks: UInt32(blocks),
                gridSize: pass2Grid, threadgroupSize: pass2Tg, on: cmd)
        case .f16:
            IronKernels.iron_sdpa_decode_2pass_pass1_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                gqa_factor: UInt32(gqaFactor), blocks: UInt32(blocks),
                scale: scale,
                gridSize: pass1Grid, threadgroupSize: pass1Tg, on: cmd)
            IronKernels.iron_sdpa_decode_2pass_pass2_f16(
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), blocks: UInt32(blocks),
                gridSize: pass2Grid, threadgroupSize: pass2Tg, on: cmd)
        case .bf16:
            IronKernels.iron_sdpa_decode_2pass_pass1_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                gqa_factor: UInt32(gqaFactor), blocks: UInt32(blocks),
                scale: scale,
                gridSize: pass1Grid, threadgroupSize: pass1Tg, on: cmd)
            IronKernels.iron_sdpa_decode_2pass_pass2_bf16(
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), blocks: UInt32(blocks),
                gridSize: pass2Grid, threadgroupSize: pass2Tg, on: cmd)
        default:
            fatalError("Ops.sdpaDecode2Pass: unsupported dtype \(q.dtype)")
        }
    }

    /// Multi-query SDPA — attends `nQuery` query rows against a shared
    /// K/V cache in one dispatch. `q` / output are `[nQuery, nQHeads,
    /// headDim]`; `k` / `v` are the cache buffers `[nKVHeads, kvStride,
    /// headDim]`. `causal == false` → every query attends
    /// `[0, baseKV + nQuery)` (bidirectional); `causal == true` → query
    /// `r` attends `[0, baseKV + r + 1)`.
    ///
    /// `iron_sdpa_multi` is a reduction kernel — its threadgroup
    /// geometry is part of the contract. Each invariant below is
    /// `precondition`-checked, citing `crates/iron-std/src/butter/
    /// sdpa_multi.rs`. The 1024-thread dispatch is hard: a smaller TPG
    /// makes the kernel's `n_simd` zero and the K walk an infinite GPU
    /// loop (the documented machine-freeze hazard).
    public static func sdpaMulti(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        baseKV: Int, nQuery: Int, kvStride: Int,
        causal: Bool, scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        // ## DISPATCH INVARIANTS — butter/sdpa_multi.rs. See
        // OpsValidation.validateSdpaMulti for the full reasoning +
        // CI-runnable tests.
        if let reason = OpsValidation.validateSdpaMulti(
            headDim: headDim, nQHeads: nQHeads, nKVHeads: nKVHeads,
            baseKV: baseKV, nQuery: nQuery, kvStride: kvStride
        ) {
            preconditionFailure("Ops.sdpaMulti: \(reason)")
        }
        let headsPerGroup = nQHeads / nKVHeads
        let result = out ?? Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: q.dtype)
        // TPG = 1024 (32 simdgroups × 32 lanes), one threadgroup per
        // (query, q_head). Lay out a 1D thread grid so Metal slices it
        // into nQHeads*nQuery threadgroups of 1024 — NEVER fewer than
        // 32 threads per group (the freeze condition).
        let threadsPerGroup = 1024
        let grid = MTLSize(width: nQHeads * nQuery * threadsPerGroup, height: 1, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let causalFlag = UInt32(causal ? 1 : 0)
        // Route by (dtype, headDim). d=128 → `iron_sdpa_multi_*`; d=256
        // → `iron_sdpa_multi_d256_*`, the variant added for Qwen3.6-A3B
        // full-attention layers (head_dim=256). Both kernels share the
        // same Swift wrapper shape; the dispatch grid / TPG / online
        // softmax / causal mask / GQA fan-out semantics are identical.
        // The d=256 kernel uses a 2-phase output reduction internally
        // to stay under Apple's 32 KB threadgroup-memory cap.
        switch (q.dtype, headDim) {
        case (.f32, 128):
            IronKernels.iron_sdpa_multi_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.f16, 128):
            IronKernels.iron_sdpa_multi_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.bf16, 128):
            IronKernels.iron_sdpa_multi_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.f32, 256):
            IronKernels.iron_sdpa_multi_d256_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.f16, 256):
            IronKernels.iron_sdpa_multi_d256_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.bf16, 256):
            IronKernels.iron_sdpa_multi_d256_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.sdpaMulti: unsupported dtype \(q.dtype) at headDim \(headDim) "
                    + "— only (f32/f16/bf16) × (128, 256) are emitted")
        }
        return result
    }

    /// Q-tiled prefill-append SDPA for `headDim == 256`, the default d=256
    /// prefill path (see `Qwen35AttentionMixer.forwardMany`;
    /// `BUTTER_SDPA_LEGACY_D256=1` reverts to `sdpaMulti`). One threadgroup
    /// owns `BQ = 16` query rows for a single q_head and reuses each
    /// loaded K/V block across all 16 rows before advancing, vs
    /// `sdpaMulti`'s one-threadgroup-per-query-row (no K/V reuse across
    /// rows, the mechanism behind the F-85 prefill-attention gap).
    /// Wraps `iron_sdpa_prefill_qtiled_d256`; see
    /// `crates/iron-std/src/kernels/sdpa/
    /// sdpa_prefill_qtiled_d256.rs` for the tiling / masking contract.
    /// TPG = 128 (4 simdgroups × 32 lanes); grid = `(ceil(nQuery/16),
    /// nQHeads, 1)`. Ragged tiles (`nQuery` not a multiple of 16) are
    /// handled inside the kernel (padded rows guarded off), so no extra
    /// precondition beyond `sdpaMulti`'s own headDim/GQA/kvStride ones.
    public static func sdpaPrefillQtiledD256(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        baseKV: Int, nQuery: Int, kvStride: Int,
        causal: Bool, scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        if let reason = OpsValidation.validateSdpaMulti(
            headDim: headDim, nQHeads: nQHeads, nKVHeads: nKVHeads,
            baseKV: baseKV, nQuery: nQuery, kvStride: kvStride
        ) {
            preconditionFailure("Ops.sdpaPrefillQtiledD256: \(reason)")
        }
        precondition(
            headDim == 256,
            "Ops.sdpaPrefillQtiledD256: only headDim == 256 is emitted, got \(headDim)")
        let headsPerGroup = nQHeads / nKVHeads
        let result = out ?? Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: q.dtype)
        let bq = 16
        let tgX = (nQuery + bq - 1) / bq
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        // `dispatchThreads` wants the grid in TOTAL THREADS, not
        // threadgroup counts (the kernel reads `tgid` via
        // `threadgroup_position_in_grid`, so what matters is that
        // grid/threadgroupSize divides out to (tgX, nQHeads, 1)
        // threadgroups). Width is threadgroups-in-x times TPG.x, height
        // is threadgroups-in-y times TPG.y (TPG.y == 1, so height equals
        // nQHeads directly). Same convention as `Ops.gemm`'s 2-D tile
        // dispatch above (`nTiles * threadsPerGroup` width, `mTiles`
        // height, TPG height 1).
        let grid = MTLSize(width: tgX * tg.width, height: nQHeads, depth: 1)
        let causalFlag = UInt32(causal ? 1 : 0)
        switch q.dtype {
        case .f32:
            IronKernels.iron_sdpa_prefill_qtiled_d256_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_sdpa_prefill_qtiled_d256_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_sdpa_prefill_qtiled_d256_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.sdpaPrefillQtiledD256: unsupported dtype \(q.dtype) "
                    + "(only f32/f16/bf16 are emitted)")
        }
        return result
    }

    /// MMA (simdgroup-matrix) Q-tiled prefill-append SDPA for
    /// `headDim == 256`. Same external contract as
    /// `sdpaPrefillQtiledD256` (baseKV/nQuery/kvStride/causal-flag
    /// prefill-append signature, GQA via nQHeads/nKVHeads, ragged
    /// nQuery guarded inside the kernel) but each threadgroup owns
    /// `BQ = 32` query rows for a single q_head, computed via
    /// multi-simdgroup `simdgroup_matrix` MMA fragments rather than
    /// `sdpaPrefillQtiledD256`'s scalar simd_sum dot products. Wraps
    /// `iron_sdpa_prefill_mma_d256`; see `crates/iron-std/src/
    /// kernels/sdpa/sdpa_prefill_mma_d256.rs`.
    /// TPG = 128 (4 simdgroups x 32 lanes); grid = `(ceil(nQuery/32),
    /// nQHeads, 1)` THREADGROUPS -- this is a coop-matrix MMA kernel,
    /// so it must be dispatched with `dispatchThreadgroups` (the
    /// `_threadgroups` generated variant), never the plain
    /// `dispatchThreads` entry point (see `sdpaPrefillMma`'s note
    /// below on this being a silent-corruption bug for coop-matrix
    /// kernels, not just a perf one).
    ///
    /// f16/bf16 default to the BK=16 follow-up kernel
    /// (`iron_sdpa_prefill_mma_d256_bk16`, see `crates/iron-std/
    /// src/kernels/sdpa/sdpa_prefill_mma_d256_bk16.rs`): it halves the
    /// K-block loop's iteration count by reusing each cooperative K/V
    /// load across twice as many query rows, at a threadgroup-memory
    /// cost that only fits f16/bf16 (33792 B at f32 would blow the 32
    /// KiB ceiling, so f32 always stays on the BK=8 kernel below).
    /// `BUTTER_SDPA_BK16=0` reverts f16/bf16 to BK=8 for bisection. Cached
    /// once (`sdpaBK16Enabled` below) rather than read from
    /// `ProcessInfo.environment` on every call - this is the default
    /// d=256 prefill attention kernel, dispatched once per attention
    /// layer per prefill chunk and once per attention layer per decode
    /// token on the batched decode path.
    private static let sdpaBK16Enabled: Bool =
        ButterEnv["BUTTER_SDPA_BK16"] != "0"

    public static func sdpaPrefillMmaD256(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        baseKV: Int, nQuery: Int, kvStride: Int,
        causal: Bool, scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        if let reason = OpsValidation.validateSdpaMulti(
            headDim: headDim, nQHeads: nQHeads, nKVHeads: nKVHeads,
            baseKV: baseKV, nQuery: nQuery, kvStride: kvStride
        ) {
            preconditionFailure("Ops.sdpaPrefillMmaD256: \(reason)")
        }
        precondition(
            headDim == 256,
            "Ops.sdpaPrefillMmaD256: only headDim == 256 is emitted, got \(headDim)")
        let headsPerGroup = nQHeads / nKVHeads
        let result = out ?? Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: q.dtype)
        // Grid in THREADGROUPS directly: (ceil(nQuery/32), nQHeads, 1).
        // Do NOT scale by tg.width the way the dispatchThreads-based
        // sdpaPrefillQtiledD256 does above -- dispatchThreadgroups
        // takes a threadgroup-count grid, not a total-thread grid.
        let bq = 32
        let tgX = (nQuery + bq - 1) / bq
        let tgGrid = MTLSize(width: tgX, height: nQHeads, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let causalFlag = UInt32(causal ? 1 : 0)
        let useBK16 = Self.sdpaBK16Enabled
        switch q.dtype {
        case .f32:
            IronKernels.iron_sdpa_prefill_mma_d256_f32_threadgroups(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: tgGrid, threadgroupSize: tg, on: cmd)
        case .f16:
            if useBK16 {
                IronKernels.iron_sdpa_prefill_mma_d256_bk16_f16_threadgroups(
                    q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                    v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                    head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                    base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                    kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                    causal: causalFlag, scale: scale,
                    gridSize: tgGrid, threadgroupSize: tg, on: cmd)
            } else {
                IronKernels.iron_sdpa_prefill_mma_d256_f16_threadgroups(
                    q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                    v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                    head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                    base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                    kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                    causal: causalFlag, scale: scale,
                    gridSize: tgGrid, threadgroupSize: tg, on: cmd)
            }
        case .bf16:
            if useBK16 {
                IronKernels.iron_sdpa_prefill_mma_d256_bk16_bf16_threadgroups(
                    q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                    v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                    head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                    base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                    kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                    causal: causalFlag, scale: scale,
                    gridSize: tgGrid, threadgroupSize: tg, on: cmd)
            } else {
                IronKernels.iron_sdpa_prefill_mma_d256_bf16_threadgroups(
                    q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                    v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                    head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                    base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                    kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                    causal: causalFlag, scale: scale,
                    gridSize: tgGrid, threadgroupSize: tg, on: cmd)
            }
        default:
            fatalError(
                "Ops.sdpaPrefillMmaD256: unsupported dtype \(q.dtype) "
                    + "(only f32/f16/bf16 are emitted)")
        }
        return result
    }

    /// Tree-causal `sdpaMulti` — same dispatch contract but the
    /// scalar `causal` boolean is replaced by an additive
    /// `[nQuery, nQuery]` mask tensor consulted only for in-block KV
    /// positions (positions `< baseKV` — the cached prefix — are
    /// always fully attended). The mask is `0.0` for "allow" and
    /// `-inf` for "block"; the kernel adds it directly to the
    /// pre-softmax scores.
    ///
    /// Used by speculative-decode tree-verify: one verifier call
    /// attends every leaf-to-root path of the draft tree at once,
    /// gated by a tree-causal mask that lets each leaf only see its
    /// own ancestors. Companion to `DraftTreeNode.treeCausalMask()`.
    ///
    /// Mask dtype must match `q`. `headDim` shares `sdpaMulti`'s
    /// constraint: only the 128-variant kernel is emitted on dev
    /// (tree-verify has no d256 use case today).
    public static func sdpaMultiTreeMask(
        q: Tensor, k: Tensor, v: Tensor, mask: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        baseKV: Int, nQuery: Int, kvStride: Int,
        scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        if let reason = OpsValidation.validateSdpaMulti(
            headDim: headDim, nQHeads: nQHeads, nKVHeads: nKVHeads,
            baseKV: baseKV, nQuery: nQuery, kvStride: kvStride
        ) {
            preconditionFailure("Ops.sdpaMultiTreeMask: \(reason)")
        }
        precondition(
            headDim == 128,
            "Ops.sdpaMultiTreeMask: only headDim=128 has a tree-mask kernel variant")
        precondition(
            mask.dtype == q.dtype,
            "Ops.sdpaMultiTreeMask: mask dtype (\(mask.dtype)) must match q dtype (\(q.dtype))")
        precondition(
            mask.elementCount == nQuery * nQuery,
            "Ops.sdpaMultiTreeMask: mask elementCount \(mask.elementCount) ≠ nQuery·nQuery \(nQuery * nQuery)")
        let headsPerGroup = nQHeads / nKVHeads
        let result = out ?? Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: q.dtype)
        let threadsPerGroup = 1024
        let grid = MTLSize(
            width: nQHeads * nQuery * threadsPerGroup,
            height: 1, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        switch q.dtype {
        case .f32:
            IronKernels.iron_sdpa_multi_tree_mask_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                mask: mask.buffer, maskOffset: mask.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_sdpa_multi_tree_mask_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                mask: mask.buffer, maskOffset: mask.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_sdpa_multi_tree_mask_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                mask: mask.buffer, maskOffset: mask.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sdpaMultiTreeMask: unsupported dtype \(q.dtype)")
        }
        return result
    }

    /// Multi-query **bidirectional** SDPA — every query attends the
    /// full `[0, baseKV + nQuery)` range. Specialized variants for
    /// head_dim ∈ {32, 64, 72} cover the VLM vision-tower spectrum
    /// (FastViT-HD = 32; SigLIP / CLIP-L = 64; PaliGemma SigLIP-So400m
    /// = 72). For head_dim = 128 use `sdpaMulti(causal: false)`.
    ///
    /// Same shape contract as `sdpaMulti`: Q / `out` layout
    /// `[nQuery, nQHeads, headDim]`, K / V layout `[nKVHeads, kvStride,
    /// headDim]`. The d=72 kernel uses a ragged 3-elements-per-lane
    /// layout with bounds masking — lanes 24..31 are idle (25% lane
    /// occupancy loss, kernel-internal, transparent to the wrapper).
    ///
    /// `iron_sdpa_bidirectional_dN` is a reduction kernel — TPG=1024
    /// is hard. See `OpsValidation.validateSdpaBidirectional` for the
    /// invariant list and `iron/crates/iron-std/src/butter/
    /// sdpa_bidirectional.rs` for the dispatch contract.
    public static func sdpaBidirectional(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        baseKV: Int, nQuery: Int, kvStride: Int,
        scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        if let reason = OpsValidation.validateSdpaBidirectional(
            headDim: headDim, nQHeads: nQHeads, nKVHeads: nKVHeads,
            baseKV: baseKV, nQuery: nQuery, kvStride: kvStride
        ) {
            preconditionFailure("Ops.sdpaBidirectional: \(reason)")
        }
        let headsPerGroup = nQHeads / nKVHeads
        let result = out ?? Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: q.dtype)
        // TPG = 1024 (32 simdgroups × 32 lanes), one threadgroup per
        // (query, q_head). Same machine-freeze hazard as sdpa_multi —
        // never use elementwiseGrid (would make n_simd=0).
        let threadsPerGroup = 1024
        let grid = MTLSize(width: nQHeads * nQuery * threadsPerGroup, height: 1, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let hd = UInt32(headDim)
        let nqh = UInt32(nQHeads)
        let bkv = UInt32(baseKV)
        let nq = UInt32(nQuery)
        let kvs = UInt32(kvStride)
        let hpg = UInt32(headsPerGroup)
        switch (headDim, q.dtype) {
        case (32, .f32):
            IronKernels.iron_sdpa_bidirectional_d32_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (32, .f16):
            IronKernels.iron_sdpa_bidirectional_d32_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (32, .bf16):
            IronKernels.iron_sdpa_bidirectional_d32_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, .f32):
            IronKernels.iron_sdpa_bidirectional_d64_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, .f16):
            IronKernels.iron_sdpa_bidirectional_d64_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, .bf16):
            IronKernels.iron_sdpa_bidirectional_d64_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (72, .f32):
            IronKernels.iron_sdpa_bidirectional_d72_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (72, .f16):
            IronKernels.iron_sdpa_bidirectional_d72_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (72, .bf16):
            IronKernels.iron_sdpa_bidirectional_d72_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (80, .f32):
            IronKernels.iron_sdpa_bidirectional_d80_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (80, .f16):
            IronKernels.iron_sdpa_bidirectional_d80_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (80, .bf16):
            IronKernels.iron_sdpa_bidirectional_d80_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (96, .f32):
            IronKernels.iron_sdpa_bidirectional_d96_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (96, .f16):
            IronKernels.iron_sdpa_bidirectional_d96_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (96, .bf16):
            IronKernels.iron_sdpa_bidirectional_d96_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sdpaBidirectional: unsupported (headDim=\(headDim), dtype=\(q.dtype))")
        }
        return result
    }

    // MARK: - AURA

    /// AURA fused encode for `rows` flat vectors of length `dim`.
    /// Computes per-row L2 norm, rotates by `rotation` (`[dim×dim]`
    /// f32), quantises against `boundaries`, packs codebook indices
    /// into `packed_out`, and writes the norm-correction factor to
    /// `norms_out`. One threadgroup per row, `dim` threads per group.
    ///
    /// `input` accepts f32 / f16 / bf16 — the kernel casts to f32 at
    /// the load, so production code can feed the K/V projection
    /// directly without an explicit upcast pass. `rotation`,
    /// `boundaries`, and `codebook` are always f32 (precision matters
    /// for the rotation matmul + Lloyd-Max boundary comparisons).
    ///
    /// Per-(bits, dtype) dispatch. `bits ∈ {2, 3, 4, 8}` × dtype ∈
    /// {f32, f16, bf16} → 12 generated kernel variants.
    public static func auraEncode(
        input: Tensor, rotation: Tensor, boundaries: Tensor, codebook: Tensor,
        packedOut: Tensor, normsOut: Tensor,
        rows: Int, dim: Int, packedWidth: Int, bits: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            rotation.dtype == input.dtype && boundaries.dtype == input.dtype,
            "Ops.auraEncode: rotation/boundaries dtype must match input "
                + "dtype (got rotation=\(rotation.dtype), "
                + "boundaries=\(boundaries.dtype) vs input=\(input.dtype)) "
                + "— iron #226 unified to Tensor<T>; the Π matrix "
                + "dominates encoder bandwidth so narrowing it at f16/bf16 "
                + "halves the dominant read")
        precondition(
            codebook.dtype == input.dtype,
            "Ops.auraEncode: codebook dtype must match input dtype "
                + "(got \(codebook.dtype) vs \(input.dtype))")
        precondition(packedOut.dtype == .u32, "Ops.auraEncode: packed_out must be u32")
        precondition(
            normsOut.dtype == input.dtype,
            "Ops.auraEncode: norms_out dtype must match input dtype "
                + "(got \(normsOut.dtype) vs \(input.dtype))")
        // Kernel-invariant validation — see OpsValidation.swift.
        if let reason = OpsValidation.validateAuraEncode(rows: rows, dim: dim, bits: bits) {
            preconditionFailure("Ops.auraEncode: \(reason)")
        }
        // One threadgroup per row; `dim` threads per group (each thread
        // owns one rotated coordinate). The kernel reads
        // `row = program_id::<0>() = tgid_x`. We dispatch `rows * dim`
        // total threads in a 1D grid so Metal slices into `rows`
        // threadgroups of `dim` threads — that gives `tgid_x` the row
        // index we want.
        //
        // The earlier shape `(dim, rows, 1)` with `dispatchThreads`
        // computed threadgroups as `(1, rows, 1)`, making `tgid_x` always
        // 0 and processing only row 0. Latent because no production
        // caller existed; AURAQuantizedKVCache is the first.
        let grid = MTLSize(width: dim * rows, height: 1, depth: 1)
        let tg = MTLSize(width: dim, height: 1, depth: 1)
        let dimU = UInt32(dim)
        let pwU = UInt32(packedWidth)

        // Per-(bits, dtype) routing. The kernel layout, argument order,
        // and grid shape are identical across all 12 variants — only the
        // function name changes — so a small local helper keeps the
        // switch readable.
        @inline(__always)
        func dispatchEncode(
            _ kernel: (
                MTLBuffer, Int, MTLBuffer, Int, MTLBuffer, Int,
                MTLBuffer, Int, MTLBuffer, Int, MTLBuffer, Int,
                UInt32, UInt32, MTLSize, MTLSize, MTLCommandBuffer
            ) -> Void
        ) {
            kernel(
                input.buffer, input.offset,
                rotation.buffer, rotation.offset,
                boundaries.buffer, boundaries.offset,
                codebook.buffer, codebook.offset,
                packedOut.buffer, packedOut.offset,
                normsOut.buffer, normsOut.offset,
                dimU, pwU, grid, tg, cmd
            )
        }

        switch (bits, input.dtype) {
        case (2, .f32):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int2_f32(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (2, .f16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int2_f16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (2, .bf16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int2_bf16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (3, .f32):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int3_f32(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (3, .f16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int3_f16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (3, .bf16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int3_bf16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (4, .f32):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int4_f32(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (4, .f16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int4_f16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (4, .bf16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int4_bf16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (8, .f32):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int8_f32(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (8, .f16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int8_f16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (8, .bf16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                IronKernels.iron_aura_encode_int8_bf16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        default:
            fatalError("Ops.auraEncode: unsupported (bits=\(bits), input.dtype=\(input.dtype))")
        }
    }

    /// AURA bulk dequant — codebook lookup + norm rescale for every
    /// stored token, into a fp32 / fp16 / bf16 working buffer in
    /// rotated codec space. The caller applies the inverse rotation
    /// (e.g. by folding Π into W_o offline, or via a follow-up
    /// matmul / fused pass2_with_rot kernel).
    ///
    /// Grid: `(packed_width, tokens, B*H)` — `dispatchThreads` does
    /// not pad, so no per-thread bounds guards are needed.
    /// Bulk-dequant AURA-packed K/V into `out`.
    ///
    /// `tokens` is the number of valid token rows to process (the
    /// dispatch grid height). `cacheStride` is the per-head row stride
    /// of the `packed` / `norms` / `out` buffers — i.e. the *allocated*
    /// sequence capacity. The dequant kernel keys all per-head offset
    /// arithmetic off its `tokens` constexpr, so for a buffer laid out
    /// `[nKVHeads, maxSeq, …]` the kernel must be fed `cacheStride =
    /// maxSeq`, NOT the fill count — otherwise heads 1…n read/write at
    /// the wrong offset and the error grows with fill length (the AURA
    /// "coherent then collapse" bug). `cacheStride` defaults to `tokens`
    /// for callers whose buffers are exactly `[nKVHeads, tokens, …]`.
    public static func auraDequantRotated(
        packed: Tensor, norms: Tensor, codebook: Tensor,
        into out: Tensor,
        nKVHeads: Int, dim: Int, packedWidth: Int, tokens: Int, bits: Int,
        cacheStride: Int? = nil,
        on cmd: MTLCommandBuffer
    ) {
        precondition(packed.dtype == .u32, "Ops.auraDequantRotated: packed must be u32")
        precondition(
            norms.dtype == out.dtype && codebook.dtype == out.dtype,
            "Ops.auraDequantRotated: norms/codebook must match output "
                + "dtype (got norms=\(norms.dtype), codebook=\(codebook.dtype) "
                + "vs out=\(out.dtype)) — post-AURA-dtype-unification")
        let stride = cacheStride ?? tokens
        // Kernel-invariant validation (cacheStride row-stride contract,
        // packedWidth dim coverage, bit-width). See
        // OpsValidation.validateAuraDequantRotated.
        if let reason = OpsValidation.validateAuraDequantRotated(
            dim: dim, packedWidth: packedWidth, tokens: tokens, bits: bits,
            cacheStride: stride
        ) {
            preconditionFailure("Ops.auraDequantRotated: \(reason)")
        }
        // Grid height = rows to process (`tokens`); the kernel's `tokens`
        // constexpr carries the per-head buffer stride (`stride`).
        let grid = MTLSize(width: packedWidth, height: tokens, depth: nKVHeads)
        let tg = MTLSize(width: packedWidth, height: 1, depth: 1)
        switch (bits, out.dtype) {
        case (2, .f32):
            IronKernels.iron_aura_dequant_rotated_int2_f32(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f16):
            IronKernels.iron_aura_dequant_rotated_int2_f16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .bf16):
            IronKernels.iron_aura_dequant_rotated_int2_bf16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f32):
            IronKernels.iron_aura_dequant_rotated_int3_f32(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f16):
            IronKernels.iron_aura_dequant_rotated_int3_f16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .bf16):
            IronKernels.iron_aura_dequant_rotated_int3_bf16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f32):
            IronKernels.iron_aura_dequant_rotated_int4_f32(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f16):
            IronKernels.iron_aura_dequant_rotated_int4_f16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .bf16):
            IronKernels.iron_aura_dequant_rotated_int4_bf16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f32):
            IronKernels.iron_aura_dequant_rotated_int8_f32(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f16):
            IronKernels.iron_aura_dequant_rotated_int8_f16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .bf16):
            IronKernels.iron_aura_dequant_rotated_int8_bf16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.auraDequantRotated: unsupported (bits=\(bits), dtype=\(out.dtype))")
        }
    }

    /// Apply a shared `[headDim, headDim]` rotation matrix to each
    /// head's `[headDim]` slice of a flat `[nHeads * headDim]` tensor.
    /// Used for AURA's Q post-RoPE rotation and the attention-output
    /// un-rotation (see `AURAQuantizedKVCache` header).
    ///
    /// The rotation matrix is shared across heads within a layer; we
    /// fan out one `Ops.gemv` dispatch per head with per-head buffer
    /// views, so the cost is `nHeads` gemv launches per call. For Qwen3
    /// 1.7B that's 16 dispatches per Q rotation × 2 calls per layer × 28
    /// layers = 896 dispatches per token. Fine for correctness-first
    /// Stage 1a; Stage 1b folds this work into the compressed-domain
    /// `aura_flash_p1` / pass2 kernels which expect pre-rotated Q and
    /// produce un-rotated output directly.
    ///
    /// Preconditions:
    ///   * `x` is `[nHeads * headDim]`
    ///   * `rotation` is `[headDim, headDim]`
    ///   * `x.dtype == rotation.dtype` (gemv requires matched dtypes —
    ///     the caller is expected to keep an activation-dtype copy of
    ///     the rotation alongside the f32 copy required by `auraEncode`)
    /// AURA flash-SDPA (compressed decode). Walks the packed K + V
    /// buffers directly — no dequant-to-mirror — saving a per-decode
    /// `nKVHeads × maxSeq × headDim` materialisation. At Qwen3-1.7B /
    /// maxSeq=32K that's ~1.8 GB of working-buffer reads eliminated per
    /// decode step.
    ///
    /// Caller must:
    ///   - have already applied Π to `q` (matches the existing
    ///     `dequantMirror` decode path — Π applied to Q before SDPA so
    ///     scores cancel; Π^T applied to the output before oProj).
    ///   - confirm `supportsFlashSdpa(scheme:headDim:)` is true; this
    ///     wrapper only covers the iron flash variants currently
    ///     emitted: (kb=4, vb=2) and (kb=4, vb=4) × d ∈ {64, 128}.
    ///   - own + pass a `sinks` tensor of shape `[nQHeads]` (zeroed +
    ///     `hasSinks=false` gives a no-op attention-sink path).
    ///
    /// The kernel takes `q_rot` as fp32; this wrapper casts on the
    /// caller's behalf into a scratch buffer if needed. Output dtype
    /// is the model's activation dtype.
    /// - Parameter liveLength: number of populated K/V rows the loop
    ///   should iterate over (the attention upper bound).
    /// - Parameter kvStride: per-KV-head row stride of the cache —
    ///   typically `maxSeq` of the AURA KVCache. MUST equal the cache's
    ///   allocated stride, not `liveLength`, otherwise off-head reads
    ///   alias into the previous head's tail bytes when the cache isn't
    ///   fully populated.
    public static func auraFlashSdpa(
        q: Tensor, sinks: Tensor,
        kPacked: Tensor, kNorms: Tensor, kCodebook: Tensor,
        vPacked: Tensor, vNorms: Tensor, vCodebook: Tensor,
        into out: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        kPackedWidth: Int, vPackedWidth: Int,
        liveLength: Int, kvStride: Int,
        keyBits: Int, valueBits: Int,
        scale: Float,
        hasSinks: Bool = false, windowSize: Int = 0,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            kvStride >= liveLength,
            "Ops.auraFlashSdpa: kvStride (\(kvStride)) must be ≥ liveLength (\(liveLength))")
        precondition(
            nQHeads % nKVHeads == 0,
            "Ops.auraFlashSdpa: nQHeads (\(nQHeads)) must be divisible by nKVHeads (\(nKVHeads))")
        precondition(
            kPacked.dtype == .u32 && vPacked.dtype == .u32,
            "Ops.auraFlashSdpa: packed K/V must be u32")
        precondition(
            kNorms.dtype == out.dtype && vNorms.dtype == out.dtype
                && kCodebook.dtype == out.dtype && vCodebook.dtype == out.dtype,
            "Ops.auraFlashSdpa: norms + codebooks must match activation "
                + "dtype (got \(kNorms.dtype) vs \(out.dtype))")
        precondition(
            sinks.dtype == out.dtype,
            "Ops.auraFlashSdpa: sinks must match activation dtype "
                + "(got \(sinks.dtype) vs \(out.dtype))")
        precondition(
            q.dtype == out.dtype,
            "Ops.auraFlashSdpa: q dtype must match activation dtype "
                + "(got \(q.dtype) vs \(out.dtype))")

        // The flash kernel's contract (aura_flash_sdpa.rs header):
        // `q_rot` is WHT-rotated AND **pre-scaled** by the caller. The
        // kernel computes `(Q · centroid) * k_norm` with no internal
        // scale — caller must bake the softmax scale (1/√headDim) into
        // Q before dispatch. Scratch + scale buffer cached by
        // (shape, dtype, scale) so the hot decode path doesn't pay 2×
        // Metal buffer allocations per attn layer per token.
        //
        // Post-dtype-unification the multiply happens in the activation
        // dtype — kernel does cast-at-load to f32 internally, matching
        // the precision of every other AURA kernel (encode / score /
        // value / 2-pass), and the C++ TQ+ fork's production pattern.
        let qScratch = AuraFlashScratchCache.qScratch(
            shape: q.shape, dtype: out.dtype)
        let scaleBuf = AuraFlashScratchCache.scaleBuffer(
            shape: q.shape, scale: scale, dtype: out.dtype)
        _ = Ops.mul(q, scaleBuf, on: cmd, into: qScratch)

        let dim = UInt32(headDim)
        let kpw = UInt32(kPackedWidth)
        let vpw = UInt32(vPackedWidth)
        let tokens = UInt32(liveLength)
        let kvStrideU = UInt32(kvStride)
        let repeatCount = UInt32(nQHeads / nKVHeads)
        let numQ = UInt32(nQHeads)
        let sinksFlag: UInt32 = hasSinks ? 1 : 0
        let win = UInt32(windowSize)

        // Grid invariant from `aura_flash_sdpa.rs`: one simdgroup per
        // Q head. grid = [32, nQHeads, 1], tg = [32, 1, 1].
        let grid = MTLSize(width: 32, height: nQHeads, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)

        switch (keyBits, valueBits, headDim, out.dtype) {
        case (4, 2, 128, .f32):
            IronKernels.iron_aura_flash_sdpa_kb4_vb2_d128_f32(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                sinks: sinks.buffer, sinksOffset: sinks.offset,
                out: out.buffer, outOffset: out.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount, num_q_heads: numQ,
                has_sinks: sinksFlag, window_size: win,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, 2, 128, .f16):
            IronKernels.iron_aura_flash_sdpa_kb4_vb2_d128_f16(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                sinks: sinks.buffer, sinksOffset: sinks.offset,
                out: out.buffer, outOffset: out.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount, num_q_heads: numQ,
                has_sinks: sinksFlag, window_size: win,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, 2, 128, .bf16):
            IronKernels.iron_aura_flash_sdpa_kb4_vb2_d128_bf16(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                sinks: sinks.buffer, sinksOffset: sinks.offset,
                out: out.buffer, outOffset: out.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount, num_q_heads: numQ,
                has_sinks: sinksFlag, window_size: win,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, 4, 128, .f32):
            IronKernels.iron_aura_flash_sdpa_kb4_vb4_d128_f32(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                sinks: sinks.buffer, sinksOffset: sinks.offset,
                out: out.buffer, outOffset: out.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount, num_q_heads: numQ,
                has_sinks: sinksFlag, window_size: win,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, 4, 128, .f16):
            IronKernels.iron_aura_flash_sdpa_kb4_vb4_d128_f16(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                sinks: sinks.buffer, sinksOffset: sinks.offset,
                out: out.buffer, outOffset: out.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount, num_q_heads: numQ,
                has_sinks: sinksFlag, window_size: win,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, 4, 128, .bf16):
            IronKernels.iron_aura_flash_sdpa_kb4_vb4_d128_bf16(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                sinks: sinks.buffer, sinksOffset: sinks.offset,
                out: out.buffer, outOffset: out.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount, num_q_heads: numQ,
                has_sinks: sinksFlag, window_size: win,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            preconditionFailure(
                "Ops.auraFlashSdpa: unsupported (keyBits=\(keyBits), "
                    + "valueBits=\(valueBits), headDim=\(headDim), "
                    + "dtype=\(out.dtype)). Caller must check "
                    + "supportsFlashSdpa(scheme:headDim:dtype:) first.")
        }
    }

    /// Predicate for the `auraFlashSdpa` supported-scheme set. Returns
    /// true when the iron flash kernel exists for this combo.
    public static func supportsAuraFlashSdpa(
        keyBits: Int, valueBits: Int, headDim: Int, dtype: DType
    ) -> Bool {
        guard [DType.f32, .f16, .bf16].contains(dtype) else { return false }
        // d=64 iron kernels exist but Butter side hasn't wired their
        // Ops dispatch yet — gate to d=128 (Qwen3 / Llama-3 family).
        guard headDim == 128 else { return false }
        if keyBits == 4 && (valueBits == 2 || valueBits == 4) { return true }
        return false
    }

    /// AURA flash-SDPA — two-pass token-parallel variant. Same external
    /// contract as `auraFlashSdpa` but dispatches `aura_flash_p1` +
    /// `aura_flash_pass2` instead of the single-pass kernel. Pass 1
    /// fans tokens across `num_blocks × nQHeads` simdgroups (single-pass
    /// dispatches one simdgroup per Q head, which starves the GPU at
    /// long context). Pass 2 reduces the per-block partials.
    ///
    /// **Block-size contract.** `num_blocks = ceil(liveLength /
    /// blockSize)`. `blockSize = 64` is the FA-2 conventional choice and
    /// saturates an M5 Max around liveLength≈4K (~16 q-heads × 64 blocks
    /// = 1024 active simdgroups). The kernel takes `block_size` as a
    /// runtime scalar so the wrapper is free to tune per-call.
    ///
    /// Uses the non-causal `aura_flash_p1` variant — safe for decode
    /// T=1 because `q_position == liveLength-1` and all populated K rows
    /// satisfy `t ≤ q_position`. The causal variant exists for kb4_vb2
    /// only; using non-causal everywhere unifies the dispatch surface.
    ///
    /// Caller owns the partials. Reuse across decode steps is safe — the
    /// shapes depend only on `(nQHeads, headDim, blockSize, kvStride)`
    /// and the wrapper writes the full surface each call.
    public static func auraFlashSdpa2Pass(
        q: Tensor,
        kPacked: Tensor, kNorms: Tensor, kCodebook: Tensor,
        vPacked: Tensor, vNorms: Tensor, vCodebook: Tensor,
        into out: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        kPackedWidth: Int, vPackedWidth: Int,
        liveLength: Int, kvStride: Int,
        keyBits: Int, valueBits: Int,
        scale: Float,
        blockSize: Int,
        partialO: Tensor, partialM: Tensor, partialL: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            kvStride >= liveLength,
            "Ops.auraFlashSdpa2Pass: kvStride (\(kvStride)) must be ≥ liveLength (\(liveLength))")
        precondition(
            nQHeads % nKVHeads == 0,
            "Ops.auraFlashSdpa2Pass: nQHeads (\(nQHeads)) must be divisible by nKVHeads (\(nKVHeads))")
        precondition(
            kPacked.dtype == .u32 && vPacked.dtype == .u32,
            "Ops.auraFlashSdpa2Pass: packed K/V must be u32")
        precondition(
            kNorms.dtype == out.dtype && vNorms.dtype == out.dtype
                && kCodebook.dtype == out.dtype && vCodebook.dtype == out.dtype,
            "Ops.auraFlashSdpa2Pass: norms + codebooks must match "
                + "activation dtype (got \(kNorms.dtype) vs \(out.dtype))")
        precondition(
            q.dtype == out.dtype,
            "Ops.auraFlashSdpa2Pass: q dtype must match activation dtype")
        precondition(
            blockSize > 0 && blockSize % 32 == 0,
            "Ops.auraFlashSdpa2Pass: blockSize (\(blockSize)) must be a "
                + "positive multiple of 32")
        let numBlocks = (liveLength + blockSize - 1) / blockSize
        precondition(
            partialO.dtype == out.dtype && partialM.dtype == out.dtype
                && partialL.dtype == out.dtype,
            "Ops.auraFlashSdpa2Pass: partials must match activation dtype")
        precondition(
            partialO.elementCount >= nQHeads * numBlocks * headDim,
            "Ops.auraFlashSdpa2Pass: partialO too small for nQHeads * numBlocks * headDim")
        precondition(
            partialM.elementCount >= nQHeads * numBlocks
                && partialL.elementCount >= nQHeads * numBlocks,
            "Ops.auraFlashSdpa2Pass: partialM/L too small for nQHeads * numBlocks")

        // Same q pre-scale flow as the single-pass wrapper — kernel
        // expects pre-scaled Q in activation dtype.
        let qScratch = AuraFlashScratchCache.qScratch(
            shape: q.shape, dtype: out.dtype)
        let scaleBuf = AuraFlashScratchCache.scaleBuffer(
            shape: q.shape, scale: scale, dtype: out.dtype)
        _ = Ops.mul(q, scaleBuf, on: cmd, into: qScratch)

        let dim = UInt32(headDim)
        let kpw = UInt32(kPackedWidth)
        let vpw = UInt32(vPackedWidth)
        let tokens = UInt32(liveLength)
        let kvStrideU = UInt32(kvStride)
        let repeatCount = UInt32(nQHeads / nKVHeads)
        let numBlocksU = UInt32(numBlocks)
        let blockSizeU = UInt32(blockSize)
        // Causal masking unused by the non-causal variant; pass 0.
        let qPosition: UInt32 = 0

        // Pass 1: Grid3D (lane, q_idx, block_idx), tg = (32, 1, 1).
        // Raw threads = [32, nQHeads, numBlocks] groups into
        // [1, nQHeads, numBlocks] TGs of 32 lanes each.
        let p1Grid = MTLSize(width: 32, height: nQHeads, depth: numBlocks)
        let p1Tg = MTLSize(width: 32, height: 1, depth: 1)
        // Pass 2: tg = (32, 1, 1) per q_idx; kernel reads
        // `q_idx = tgid_x`. Raw threads = [nQHeads*32, 1, 1] groups into
        // [nQHeads, 1, 1] TGs, one per q_idx — matches the end-to-end
        // `aura_flash_gpu_correctness.rs` dispatch shape.
        let p2Grid = MTLSize(width: nQHeads * 32, height: 1, depth: 1)
        let p2Tg = MTLSize(width: 32, height: 1, depth: 1)

        switch (keyBits, valueBits, headDim, out.dtype) {
        case (4, 2, 128, .f32):
            IronKernels.iron_aura_flash_p1_kb4_vb2_d128_f32(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount,
                num_blocks: numBlocksU, block_size: blockSizeU,
                q_position: qPosition,
                gridSize: p1Grid, threadgroupSize: p1Tg, on: cmd)
            IronKernels.iron_aura_flash_pass2_d128_f32(
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                output: out.buffer, outputOffset: out.offset,
                dim: dim, num_blocks: numBlocksU,
                gridSize: p2Grid, threadgroupSize: p2Tg, on: cmd)
        case (4, 2, 128, .f16):
            IronKernels.iron_aura_flash_p1_kb4_vb2_d128_f16(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount,
                num_blocks: numBlocksU, block_size: blockSizeU,
                q_position: qPosition,
                gridSize: p1Grid, threadgroupSize: p1Tg, on: cmd)
            IronKernels.iron_aura_flash_pass2_d128_f16(
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                output: out.buffer, outputOffset: out.offset,
                dim: dim, num_blocks: numBlocksU,
                gridSize: p2Grid, threadgroupSize: p2Tg, on: cmd)
        case (4, 2, 128, .bf16):
            IronKernels.iron_aura_flash_p1_kb4_vb2_d128_bf16(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount,
                num_blocks: numBlocksU, block_size: blockSizeU,
                q_position: qPosition,
                gridSize: p1Grid, threadgroupSize: p1Tg, on: cmd)
            IronKernels.iron_aura_flash_pass2_d128_bf16(
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                output: out.buffer, outputOffset: out.offset,
                dim: dim, num_blocks: numBlocksU,
                gridSize: p2Grid, threadgroupSize: p2Tg, on: cmd)
        case (4, 4, 128, .f32):
            IronKernels.iron_aura_flash_p1_kb4_vb4_d128_f32(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount,
                num_blocks: numBlocksU, block_size: blockSizeU,
                q_position: qPosition,
                gridSize: p1Grid, threadgroupSize: p1Tg, on: cmd)
            IronKernels.iron_aura_flash_pass2_d128_f32(
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                output: out.buffer, outputOffset: out.offset,
                dim: dim, num_blocks: numBlocksU,
                gridSize: p2Grid, threadgroupSize: p2Tg, on: cmd)
        case (4, 4, 128, .f16):
            IronKernels.iron_aura_flash_p1_kb4_vb4_d128_f16(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount,
                num_blocks: numBlocksU, block_size: blockSizeU,
                q_position: qPosition,
                gridSize: p1Grid, threadgroupSize: p1Tg, on: cmd)
            IronKernels.iron_aura_flash_pass2_d128_f16(
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                output: out.buffer, outputOffset: out.offset,
                dim: dim, num_blocks: numBlocksU,
                gridSize: p2Grid, threadgroupSize: p2Tg, on: cmd)
        case (4, 4, 128, .bf16):
            IronKernels.iron_aura_flash_p1_kb4_vb4_d128_bf16(
                q_rot: qScratch.buffer, q_rotOffset: qScratch.offset,
                key_packed: kPacked.buffer, key_packedOffset: kPacked.offset,
                key_norms: kNorms.buffer, key_normsOffset: kNorms.offset,
                key_codebook: kCodebook.buffer, key_codebookOffset: kCodebook.offset,
                val_packed: vPacked.buffer, val_packedOffset: vPacked.offset,
                val_norms: vNorms.buffer, val_normsOffset: vNorms.offset,
                val_codebook: vCodebook.buffer, val_codebookOffset: vCodebook.offset,
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                dim: dim, key_packed_width: kpw, value_packed_width: vpw,
                tokens: tokens, kv_stride: kvStrideU,
                repeat_count: repeatCount,
                num_blocks: numBlocksU, block_size: blockSizeU,
                q_position: qPosition,
                gridSize: p1Grid, threadgroupSize: p1Tg, on: cmd)
            IronKernels.iron_aura_flash_pass2_d128_bf16(
                o_partials: partialO.buffer, o_partialsOffset: partialO.offset,
                m_partials: partialM.buffer, m_partialsOffset: partialM.offset,
                l_partials: partialL.buffer, l_partialsOffset: partialL.offset,
                output: out.buffer, outputOffset: out.offset,
                dim: dim, num_blocks: numBlocksU,
                gridSize: p2Grid, threadgroupSize: p2Tg, on: cmd)
        default:
            preconditionFailure(
                "Ops.auraFlashSdpa2Pass: unsupported (keyBits=\(keyBits), "
                    + "valueBits=\(valueBits), headDim=\(headDim), "
                    + "dtype=\(out.dtype)). Caller must check "
                    + "supportsAuraFlashSdpa2Pass(...) first.")
        }
    }

    /// Predicate for the `auraFlashSdpa2Pass` supported-scheme set.
    /// Matches `supportsAuraFlashSdpa` today — both wrappers cover the
    /// same iron-emitted (kb=4, vb∈{2,4}) × d=128 cells.
    public static func supportsAuraFlashSdpa2Pass(
        keyBits: Int, valueBits: Int, headDim: Int, dtype: DType
    ) -> Bool {
        guard [DType.f32, .f16, .bf16].contains(dtype) else { return false }
        guard headDim == 128 else { return false }
        if keyBits == 4 && (valueBits == 2 || valueBits == 4) { return true }
        return false
    }

    public static func auraRotatePerHead(
        _ x: Tensor, rotation: Tensor,
        nHeads: Int, headDim: Int,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        // Pure shape / dtype contract validation — see
        // OpsValidation.validateAuraRotatePerHead.
        if let reason = OpsValidation.validateAuraRotatePerHead(
            xElementCount: x.elementCount, rotationShape: rotation.shape,
            rotationDtypeMatchesX: rotation.dtype == x.dtype,
            nHeads: nHeads, headDim: headDim
        ) {
            preconditionFailure("Ops.auraRotatePerHead: \(reason)")
        }

        let result = Tensor.empty(shape: [nHeads * headDim], dtype: x.dtype)
        let bytesPerHead = headDim * x.dtype.byteSize

        for h in 0 ..< nHeads {
            let xView = Tensor(
                buffer: x.buffer,
                offset: x.offset + h * bytesPerHead,
                shape: [headDim], dtype: x.dtype)
            let outView = Tensor(
                buffer: result.buffer,
                offset: result.offset + h * bytesPerHead,
                shape: [headDim], dtype: x.dtype)
            _ = Ops.gemv(weight: rotation, input: xView, on: cmd, into: outView)
        }
        return result
    }

    // ─── MoE batched gather GEMM (int4) ──────────────────────────────
    //
    // Wraps `iron_moe_gather_qmm_mma_int4_bm16_*` — one kernel launch that
    // processes `mTotal` rows of activations, each row tagged with the
    // expert id it routes to via `indices[mTotal]`. Replaces a Python-
    // style serial-expert loop (`topK` dispatches × 3 projections = 3K
    // launches) with one batched dispatch per projection.
    //
    // Weight layout (mlx affine int4 stacked):
    //   `weight   [E, N, K/8]` u32 packed, row-major within each expert.
    //   `scales`, `biases` `[E, N, K/groupSize]` in the activation dtype.
    // Activation layout:
    //   `input    [mTotal, K]` contiguous in `input.dtype`.
    //   `indices  [mTotal]` u32 — expert id per row. ROWS MUST BE SORTED
    //                            BY EXPERT (the kernel scans forward
    //                            looking for an expert-id change to find
    //                            the per-tile sub-range).
    // Output:
    //   `out      [mTotal, N]` in `input.dtype` (caller pre-allocates).
    //
    // Tile contract: N % 32 == 0, K % 32 == 0 (the kernel pads m_total
    // internally — BM=16 with boundary masking). Grid =
    //   [N / 32, ceil(mTotal / 16), 1]
    // Threadgroup = [64, 1, 1] (2 simdgroups).
    /// Grow-only identity row-map (`x_rows[i] = i`) for the bm16 MPP
    /// gather family's pre-gathered-X layout. An iota of capacity N
    /// serves any mTotal <= N — the kernel reads only the first m_total
    /// entries. Refilled only on growth.
    private static let identityRowsLock = NSLock()
    nonisolated(unsafe) private static var identityRowsBuf: MTLBuffer?
    static func identityRowsBuffer(minCount: Int, device: Device) -> MTLBuffer {
        identityRowsLock.lock()
        defer { identityRowsLock.unlock() }
        if let buf = identityRowsBuf, buf.length >= minCount * 4 { return buf }
        let cap = max(minCount, 8192)
        let buf = device.makeBuffer(length: cap * 4)
        let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: cap)
        for i in 0 ..< cap { ptr[i] = UInt32(i) }
        identityRowsBuf = buf
        return buf
    }

    /// Whether the MPP auto-default may fire: Apple GPU Family 9+ (M3+)
    /// and macOS 26+ (Metal 4 — the MPP header live-compile requires it).
    /// Cached once; explicit BUTTER_MOE_BGEMM_MPP=1 bypasses this.
    ///
    /// Module-visible (not `private`) rather than duplicated: `MoELayer`'s
    /// F-85 coop-core MoE gather gate (`coopEnabled`) also wraps a
    /// `matmul2d`/cooperative-tensor kernel (`iron_moe_gather_qmm_coop`)
    /// with the same Family-9/macOS-26 requirement, and unlike
    /// `dispatchQmmMma`'s coop tier below (which falls back to a cheaper
    /// tier on incapable hardware), that kernel has no lower-level
    /// fallback of its own once dispatched - the capability check has to
    /// happen at the `MoELayer` call site, so it reuses this same cached
    /// check instead of re-deriving it.
    static let mppAutoCapable: Bool = {
        // Diagnostic force-off: BUTTER_FORCE_NO_MPP=1 makes the auto-gate
        // report incapable, so every coop-tensor path takes its non-MPP
        // fallback and every MPP-guarded test skips. This lets the full suite
        // be validated on real hardware exactly as it would run on a paravirt
        // CI GPU (Family < 9) that cannot build cooperative-tensor PSOs.
        // Default off; genuine hardware detection below is unchanged.
        if let raw = ButterEnv["BUTTER_FORCE_NO_MPP"],
            raw == "1" || raw.lowercased() == "true"
        {
            return false
        }
        guard
            ProcessInfo.processInfo.isOperatingSystemAtLeast(
                OperatingSystemVersion(majorVersion: 26, minorVersion: 0, patchVersion: 0))
        else { return false }
        return Device.shared.mtlDevice.supportsFamily(.apple9)
    }()

    public static func moeGatherDequantGemmInt4(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        // Shape / dtype invariants. The kernel itself does no validation
        // so any mismatch silently produces garbage.
        precondition(weight.dtype == .u32, "moeGatherDequantGemmInt4: weight must be u32 packed")
        precondition(
            indices.dtype == .u32,
            "moeGatherDequantGemmInt4: indices must be u32 (\(indices.dtype))")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4: input/scales/biases dtype must match")
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4: out dtype must match input")
        precondition(
            input.elementCount == mTotal * kIn,
            "moeGatherDequantGemmInt4: input has \(input.elementCount) elements, expected \(mTotal * kIn) (mTotal=\(mTotal) * kIn=\(kIn))"
        )
        precondition(
            out.elementCount == mTotal * nOut,
            "moeGatherDequantGemmInt4: out has \(out.elementCount) elements, expected \(mTotal * nOut)"
        )
        precondition(
            indices.elementCount == mTotal,
            "moeGatherDequantGemmInt4: indices has \(indices.elementCount) elements, expected mTotal=\(mTotal)"
        )
        precondition(
            nOut % 32 == 0,
            "moeGatherDequantGemmInt4: nOut (\(nOut)) must be multiple of 32 for bm16 kernel")
        precondition(
            kIn % 32 == 0,
            "moeGatherDequantGemmInt4: kIn (\(kIn)) must be multiple of 32 for bm16 kernel")

        // BM=16, BN=32. The simdgroup-matrix variant uses 64 threads
        // (2 SGs); the MPP / NAX cooperative-tensor variant uses 32
        // threads (1 SG). Canonical iron dispatch
        //   dispatch_with_grid(grid=[N/32, ceil(T/16), 1], tg=[TG_W, 1, 1])
        // calls `dispatchThreadgroups` (grid counted in TGs); the
        // generated Swift wrapper calls `dispatchThreads` (grid counted
        // in TOTAL threads) so total threads = grid × tg per axis.
        //
        // NAX cooperative-tensor (MPP): macOS 26+ / Apple GPU Family 9+ (M3+).
        // Force on: BUTTER_MOE_BGEMM_MPP=1. Force off: BUTTER_MOE_BGEMM_NO_MPP=1.
        // Optional floor: BUTTER_MOE_BGEMM_MPP_MIN=<mTotal> (default 0 = always
        // when hardware allows). Measured 2026-07-18 M5 Max Qwen3.6-A3B:
        // force-on beats simdgroup at short T too (T=57 +9%, T=82 +6%)
        // and is neutral/better at long T — the old mTotal≥1024 floor was
        // leaving short/mid prefill on the slower path. Auto must still
        // not route M1–M2 or pre-Metal-4 into a live-compile that cannot
        // find the MPP header (same .apple9 gate the MPP tests use).
        // Force-on skips the gate: explicit opt-in owns unsupported HW.
        let useMpp: Bool
        if ButterEnv["BUTTER_MOE_BGEMM_NO_MPP"] != nil {
            useMpp = false
        } else if ButterEnv["BUTTER_MOE_BGEMM_MPP"] != nil {
            useMpp = true
        } else {
            let mppMin = Int(ButterEnv["BUTTER_MOE_BGEMM_MPP_MIN"] ?? "") ?? 0
            useMpp = mTotal >= mppMin && Self.mppAutoCapable
        }
        let tgWidth = useMpp ? 32 : 64
        let grid = MTLSize(
            width: (nOut / 32) * tgWidth,
            height: (mTotal + 15) / 16,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let mTotalU = UInt32(mTotal)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let groupSizeU = UInt32(groupSize)

        if useMpp {
            // Identity x_rows: X is pre-gathered, row i reads x[i, :].
            let idRows = Self.identityRowsBuffer(minCount: mTotal, device: Device.shared)
            switch input.dtype {
            case .f32:
                IronKernels.iron_moe_gather_qmm_mma_int4_bm16_mpp_f32(
                    x: input.buffer, xOffset: input.offset,
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    indices: indices.buffer, indicesOffset: indices.offset,
                    x_rows: idRows, x_rowsOffset: 0,
                    out: out.buffer, outOffset: out.offset,
                    m_total: mTotalU, n_out: nOutU, k_in: kInU,
                    group_size: groupSizeU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            case .f16:
                IronKernels.iron_moe_gather_qmm_mma_int4_bm16_mpp_f16(
                    x: input.buffer, xOffset: input.offset,
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    indices: indices.buffer, indicesOffset: indices.offset,
                    x_rows: idRows, x_rowsOffset: 0,
                    out: out.buffer, outOffset: out.offset,
                    m_total: mTotalU, n_out: nOutU, k_in: kInU,
                    group_size: groupSizeU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            case .bf16:
                // Qwen3.x activations are often bf16; kernel is already emitted.
                IronKernels.iron_moe_gather_qmm_mma_int4_bm16_mpp_bf16(
                    x: input.buffer, xOffset: input.offset,
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    indices: indices.buffer, indicesOffset: indices.offset,
                    x_rows: idRows, x_rowsOffset: 0,
                    out: out.buffer, outOffset: out.offset,
                    m_total: mTotalU, n_out: nOutU, k_in: kInU,
                    group_size: groupSizeU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            default:
                fatalError(
                    "Ops.moeGatherDequantGemmInt4: MPP unsupported dtype \(input.dtype)"
                )
            }
            return
        }

        switch input.dtype {
        case .f32:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm16_f32(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm16_f16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm16_bf16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4: unsupported dtype \(input.dtype)")
        }
    }

    // MARK: - F-85 expert-aligned tile-plan MoE gather GEMM

    /// Host-computable worst-case tile-plan capacity for `moeBuildTilePlan` /
    /// `moeGatherDequantGemmInt4TilePlan`: `ceil(mTotal/16) + nExperts`. This
    /// is a provable upper bound on `sum_e ceil(count[e]/16)` (every expert
    /// contributes at most one "ceiling" tile beyond its exact `count/16`
    /// share) that depends only on `mTotal`/`nExperts` - both known on the
    /// host before routing runs - so callers can size and dispatch the
    /// plan-builder AND the tileplan GEMM with a fully static grid, no
    /// GPU→CPU sync to read back the real tile count. See
    /// `iron_moe_build_tile_plan`'s dispatch-invariants doc (kernels repo,
    /// `moe_tile_plan_builder.rs`) for the proof sketch and the zero-fill
    /// contract that makes the padding entries this bound reserves safe to
    /// dispatch as inert tiles.
    public static func moeTilePlanCapacity(mTotal: Int, nExperts: Int) -> Int {
        (mTotal + 15) / 16 + nExperts
    }

    /// Device-side tile-plan builder (`iron_moe_build_tile_plan`). Reads the
    /// already expert-sorted `sortedExperts [mTotal]` device buffer (the
    /// routing step's own sort output - no host readback) and writes
    /// `tileExpert`/`tileRowStart`/`tileRowCount`, one entry per expert-
    /// aligned BM=16 tile, ascending by expert id. Real entries occupy
    /// indices `[0, realTileCount)`; the caller must have zero-filled the
    /// buffers out to `capacity` (from `moeTilePlanCapacity`) before this
    /// call - this kernel only writes the real entries, it does not know
    /// `capacity` and never touches padding.
    ///
    /// One threadgroup, `nExperts` lanes - the kernel's own threadgroup
    /// scratch array is sized for 256 lanes (Qwen3.6-35B-A3B's expert
    /// count); `nExperts` above that would silently under-allocate.
    public static func moeBuildTilePlan(
        sortedExperts: Tensor,
        tileExpert: Tensor, tileRowStart: Tensor, tileRowCount: Tensor,
        mTotal: Int, nExperts: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            sortedExperts.dtype == .u32, "moeBuildTilePlan: sortedExperts must be u32")
        precondition(
            tileExpert.dtype == .u32 && tileRowStart.dtype == .u32 && tileRowCount.dtype == .u32,
            "moeBuildTilePlan: tile_expert/tile_row_start/tile_row_count must be u32")
        precondition(
            nExperts <= 256,
            "moeBuildTilePlan: nExperts (\(nExperts)) exceeds the kernel's 256-lane threadgroup scratch capacity"
        )
        let grid = MTLSize(width: nExperts, height: 1, depth: 1)
        let tg = MTLSize(width: nExperts, height: 1, depth: 1)
        IronKernels.iron_moe_build_tile_plan_f32(
            sorted_experts: sortedExperts.buffer, sorted_expertsOffset: sortedExperts.offset,
            tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
            tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
            tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
            m_total: UInt32(mTotal), n_experts: UInt32(nExperts),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    /// Expert-aligned tile-plan MoE gather GEMM
    /// (`iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan`, F-85). Same
    /// dequant math, same BM=16/BN=32/BK=16 MPP geometry as
    /// `moeGatherDequantGemmInt4`'s MPP path - the only difference is tile
    /// boundaries come from a host-dispatched, device-built plan
    /// (`tileExpert`/`tileRowStart`/`tileRowCount`, see `moeBuildTilePlan`)
    /// instead of a fixed 16-row M-stride, so every dispatched tile is
    /// single-expert by construction and pays exactly one K-dimension pass.
    ///
    /// `capacity` is the dispatch grid's tile-axis extent - the host
    /// worst-case bound from `moeTilePlanCapacity`, NOT the real tile
    /// count (which would need a GPU→CPU sync to read back). Tiles at or
    /// past the real count read `tileRowCount == 0` (the caller's
    /// zero-fill) and are masked out by the same `mr < row_count` tail
    /// check that handles an ordinary trailing partial tile - they stage
    /// no X row and write no output, just redundant W-dequant work.
    public static func moeGatherDequantGemmInt4TilePlan(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        tileExpert: Tensor, tileRowStart: Tensor, tileRowCount: Tensor,
        xRows: Tensor,
        capacity: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(weight.dtype == .u32, "moeGatherDequantGemmInt4TilePlan: weight must be u32 packed")
        precondition(
            tileExpert.dtype == .u32 && tileRowStart.dtype == .u32 && tileRowCount.dtype == .u32,
            "moeGatherDequantGemmInt4TilePlan: tile_expert/tile_row_start/tile_row_count must be u32")
        precondition(
            xRows.dtype == .u32, "moeGatherDequantGemmInt4TilePlan: xRows must be u32")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4TilePlan: input/scales/biases dtype must match")
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4TilePlan: out dtype must match input")
        precondition(
            nOut % 32 == 0,
            "moeGatherDequantGemmInt4TilePlan: nOut (\(nOut)) must be multiple of 32")
        precondition(
            kIn % 16 == 0,
            "moeGatherDequantGemmInt4TilePlan: kIn (\(kIn)) must be multiple of 16")

        // Single simdgroup (32-thread) MPP kernel, same as the sibling
        // bm16_mpp dispatch: thread-counted grid, tile axis = capacity
        // (host worst-case bound, static - see moeTilePlanCapacity).
        let grid = MTLSize(width: (nOut / 32) * 32, height: capacity, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let groupSizeU = UInt32(groupSize)

        switch input.dtype {
        case .f32:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan_f32(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
                tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
                tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
                x_rows: xRows.buffer, x_rowsOffset: xRows.offset,
                out: out.buffer, outOffset: out.offset,
                n_out: nOutU, k_in: kInU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan_f16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
                tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
                tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
                x_rows: xRows.buffer, x_rowsOffset: xRows.offset,
                out: out.buffer, outOffset: out.offset,
                n_out: nOutU, k_in: kInU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm16_mpp_tileplan_bf16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
                tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
                tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
                x_rows: xRows.buffer, x_rowsOffset: xRows.offset,
                out: out.buffer, outOffset: out.offset,
                n_out: nOutU, k_in: kInU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4TilePlan: unsupported dtype \(input.dtype)")
        }
    }

    // MARK: - F-85 coop-core MoE gather GEMM (BM=32 tile plan)

    /// BM=32 sibling of `moeTilePlanCapacity`: `ceil(mTotal/32) + nExperts`,
    /// the worst-case tile count for `moeBuildTilePlanBm32` /
    /// `moeGatherDequantGemmInt4CoopTilePlan`. Same proof as the BM=16
    /// bound (every expert contributes at most one "ceiling" tile beyond
    /// its exact `count/32` share) with the tile height doubled.
    public static func moeTilePlanCapacityBm32(mTotal: Int, nExperts: Int) -> Int {
        (mTotal + 31) / 32 + nExperts
    }

    /// Device-side BM=32 tile-plan builder (`iron_moe_build_tile_plan_bm32`).
    /// Same contract as `moeBuildTilePlan` - reads the already expert-sorted
    /// `sortedExperts` buffer, writes `tileExpert`/`tileRowStart`/
    /// `tileRowCount` for BM=32 tiles - just the tile-height constant
    /// doubled. Caller must zero-fill the plan buffers out to
    /// `moeTilePlanCapacityBm32` before this call, same padding contract.
    /// `tileCountGateUp`/`tileCountDown` (F-85 idle-tile-cost follow-up):
    /// one-element u32 views the caller has already positioned at the
    /// "count" word (word 1) of its own two
    /// `MTLDispatchThreadgroupsIndirectArguments`-shaped triples - see
    /// `Ops.moeGatherDequantGemmInt4CoopTilePlanIndirect`. The two legs
    /// need independent triples because they differ in `n_out` (word 0),
    /// but both receive the SAME device-computed real tile count here.
    public static func moeBuildTilePlanBm32(
        sortedExperts: Tensor,
        tileExpert: Tensor, tileRowStart: Tensor, tileRowCount: Tensor,
        tileCountGateUp: Tensor, tileCountDown: Tensor,
        mTotal: Int, nExperts: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            sortedExperts.dtype == .u32, "moeBuildTilePlanBm32: sortedExperts must be u32")
        precondition(
            tileExpert.dtype == .u32 && tileRowStart.dtype == .u32 && tileRowCount.dtype == .u32,
            "moeBuildTilePlanBm32: tile_expert/tile_row_start/tile_row_count must be u32")
        precondition(
            tileCountGateUp.dtype == .u32 && tileCountDown.dtype == .u32,
            "moeBuildTilePlanBm32: tileCountGateUp/tileCountDown must be u32")
        precondition(
            nExperts <= 256,
            "moeBuildTilePlanBm32: nExperts (\(nExperts)) exceeds the kernel's 256-lane threadgroup scratch capacity"
        )
        let grid = MTLSize(width: nExperts, height: 1, depth: 1)
        let tg = MTLSize(width: nExperts, height: 1, depth: 1)
        IronKernels.iron_moe_build_tile_plan_bm32_f32(
            sorted_experts: sortedExperts.buffer, sorted_expertsOffset: sortedExperts.offset,
            tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
            tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
            tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
            tile_count_gateup: tileCountGateUp.buffer, tile_count_gateupOffset: tileCountGateUp.offset,
            tile_count_down: tileCountDown.buffer, tile_count_downOffset: tileCountDown.offset,
            m_total: UInt32(mTotal), n_experts: UInt32(nExperts),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    /// Coop-core MoE gather GEMM (`iron_moe_gather_qmm_coop`, F-85 gather
    /// variant). Reuses `iron_qmm_coop`'s inner loop/staging (TPG=128, 4 SG
    /// 2x2 warp grid, BN=64) against a BM=32 tile plan - the tile height
    /// matches this core's native M-tile exactly, so a plan tile maps to
    /// one threadgroup with no pairing logic, mirroring
    /// `moeGatherDequantGemmInt4TilePlan`'s BM=16 dispatch shape one tile
    /// height up. `capacity` is `moeTilePlanCapacityBm32`'s host worst-case
    /// bound (the dispatch grid's tile-axis extent), not the real tile
    /// count - same zero-fill/inert-tile padding contract as the BM=16
    /// sibling. Requires `nOut % 64 == 0` (BN=64) and `groupSize == 64`
    /// (baked group-size assumption shared with the dense coop core);
    /// callers outside that shape must use the BM=16 tileplan path instead.
    public static func moeGatherDequantGemmInt4CoopTilePlan(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        tileExpert: Tensor, tileRowStart: Tensor, tileRowCount: Tensor,
        xRows: Tensor,
        capacity: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(weight.dtype == .u32, "moeGatherDequantGemmInt4CoopTilePlan: weight must be u32 packed")
        precondition(
            tileExpert.dtype == .u32 && tileRowStart.dtype == .u32 && tileRowCount.dtype == .u32,
            "moeGatherDequantGemmInt4CoopTilePlan: tile_expert/tile_row_start/tile_row_count must be u32")
        precondition(
            xRows.dtype == .u32, "moeGatherDequantGemmInt4CoopTilePlan: xRows must be u32")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4CoopTilePlan: input/scales/biases dtype must match")
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4CoopTilePlan: out dtype must match input")
        precondition(
            nOut % 64 == 0,
            "moeGatherDequantGemmInt4CoopTilePlan: nOut (\(nOut)) must be multiple of 64 (BN tile)")
        precondition(
            kIn % 32 == 0,
            "moeGatherDequantGemmInt4CoopTilePlan: kIn (\(kIn)) must be multiple of 32 (BK tile)")
        precondition(
            kIn.isMultiple(of: groupSize) && kIn / groupSize > 0,
            "moeGatherDequantGemmInt4CoopTilePlan: groupSize (\(groupSize)) must divide kIn (\(kIn))")

        // 4-SG coop-core dispatch: threadgroup-counted grid, tile axis =
        // capacity (host worst-case bound, static - see
        // moeTilePlanCapacityBm32).
        let grid = MTLSize(width: nOut / 64, height: capacity, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let groupSizeU = UInt32(groupSize)

        switch input.dtype {
        case .f32:
            IronKernels.iron_moe_gather_qmm_coop_f32_threadgroups(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
                tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
                tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
                x_rows: xRows.buffer, x_rowsOffset: xRows.offset,
                out: out.buffer, outOffset: out.offset,
                n_out: nOutU, k_in: kInU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_gather_qmm_coop_f16_threadgroups(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
                tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
                tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
                x_rows: xRows.buffer, x_rowsOffset: xRows.offset,
                out: out.buffer, outOffset: out.offset,
                n_out: nOutU, k_in: kInU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_gather_qmm_coop_bf16_threadgroups(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
                tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
                tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
                x_rows: xRows.buffer, x_rowsOffset: xRows.offset,
                out: out.buffer, outOffset: out.offset,
                n_out: nOutU, k_in: kInU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4CoopTilePlan: unsupported dtype \(input.dtype)")
        }
    }

    /// Indirect-dispatch sibling of `moeGatherDequantGemmInt4CoopTilePlan`
    /// (F-85 idle-tile-cost follow-up). An ablation on the production
    /// coop kernel's guard structure found no early exit before the
    /// K-loop: every dispatched threadgroup, including worst-case-capacity
    /// padding whose `tileRowCount == 0`, pays the full W dequant and
    /// `coop_tile_run` matmul, not a cheap masked skip - same-binary
    /// capacity-vs-real-tile-count dispatch measured 10-21% idle-tile cost
    /// at the F-85 low-context mTotal targets. Same kernel, same buffer
    /// contract as `moeGatherDequantGemmInt4CoopTilePlan` - the only
    /// difference is the dispatch grid comes from a device-written
    /// `MTLDispatchThreadgroupsIndirectArguments` triple
    /// (`indirectArgs`/`indirectArgsOffset`, produced by
    /// `Ops.moeBuildTilePlanBm32`'s `tileCountGateUp`/`tileCountDown`
    /// outputs) instead of the host worst-case `capacity` bound, so idle
    /// tiles beyond the REAL tile count launch zero threadgroups instead
    /// of running a masked-but-still-full K-loop.
    ///
    /// Hand-encoded rather than routed through a generated `_threadgroups`
    /// wrapper: the generated Swift bindings only emit a fixed
    /// `dispatchThreadgroups(_:threadsPerThreadgroup:)` call taking a
    /// host-computed `MTLSize`, with no indirect-buffer variant (iron
    /// codegen limitation - out of scope to change for this fix). The PSO
    /// itself is unchanged (same pipeline the generated wrapper uses,
    /// fetched by the same name from the shared `PSOCache`), and every
    /// `setBuffer`/`setBytes` call below mirrors the generated
    /// `..._threadgroups` wrapper's binding order index-for-index - see
    /// `iron_moe_gather_qmm_coop_<dtype>_threadgroups` in
    /// `IronKernels.swift` for the reference this must stay in sync with
    /// if that kernel's signature ever changes.
    public static func moeGatherDequantGemmInt4CoopTilePlanIndirect(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        tileExpert: Tensor, tileRowStart: Tensor, tileRowCount: Tensor,
        xRows: Tensor,
        indirectArgs: MTLBuffer, indirectArgsOffset: Int = 0,
        nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            weight.dtype == .u32, "moeGatherDequantGemmInt4CoopTilePlanIndirect: weight must be u32 packed")
        precondition(
            tileExpert.dtype == .u32 && tileRowStart.dtype == .u32 && tileRowCount.dtype == .u32,
            "moeGatherDequantGemmInt4CoopTilePlanIndirect: tile_expert/tile_row_start/tile_row_count must be u32"
        )
        precondition(
            xRows.dtype == .u32, "moeGatherDequantGemmInt4CoopTilePlanIndirect: xRows must be u32")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4CoopTilePlanIndirect: input/scales/biases dtype must match")
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4CoopTilePlanIndirect: out dtype must match input")
        precondition(
            nOut % 64 == 0,
            "moeGatherDequantGemmInt4CoopTilePlanIndirect: nOut (\(nOut)) must be multiple of 64 (BN tile)")
        precondition(
            kIn % 32 == 0,
            "moeGatherDequantGemmInt4CoopTilePlanIndirect: kIn (\(kIn)) must be multiple of 32 (BK tile)")
        precondition(
            kIn.isMultiple(of: groupSize) && kIn / groupSize > 0,
            "moeGatherDequantGemmInt4CoopTilePlanIndirect: groupSize (\(groupSize)) must divide kIn (\(kIn))")

        let kernelName: String
        switch input.dtype {
        case .f32: kernelName = "iron_moe_gather_qmm_coop_f32"
        case .f16: kernelName = "iron_moe_gather_qmm_coop_f16"
        case .bf16: kernelName = "iron_moe_gather_qmm_coop_bf16"
        default:
            fatalError("Ops.moeGatherDequantGemmInt4CoopTilePlanIndirect: unsupported dtype \(input.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: kernelName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(biases.buffer, offset: biases.offset, index: 3)
        enc.setBuffer(tileExpert.buffer, offset: tileExpert.offset, index: 4)
        enc.setBuffer(tileRowStart.buffer, offset: tileRowStart.offset, index: 5)
        enc.setBuffer(tileRowCount.buffer, offset: tileRowCount.offset, index: 6)
        enc.setBuffer(xRows.buffer, offset: xRows.offset, index: 7)
        enc.setBuffer(out.buffer, offset: out.offset, index: 8)
        var nOutV = UInt32(nOut)
        enc.setBytes(&nOutV, length: 4, index: 9)
        var kInV = UInt32(kIn)
        enc.setBytes(&kInV, length: 4, index: 10)
        var groupSizeV = UInt32(groupSize)
        enc.setBytes(&groupSizeV, length: 4, index: 11)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        enc.dispatchThreadgroups(
            indirectBuffer: indirectArgs, indirectBufferOffset: indirectArgsOffset,
            threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// Worst-case tile-plan capacity for `moeBuildTilePlanBm32Paired` /
    /// `moeGatherDequantGemmInt4CoopTilePlanPaired`: `ceil(nExperts / 2)`
    /// paired tiles at most (every candidate expert pairs with at most one
    /// other; an odd leftover still costs one tile with an inert B half).
    public static func moePairedTilePlanCapacity(nExperts: Int) -> Int {
        (nExperts + 1) / 2
    }

    /// Device-side "own" BM=32 tile-plan builder for the F-85 tile-pairing
    /// lever (`iron_moe_build_tile_plan_bm32_own`). Same contract and same
    /// worst-case capacity as `moeBuildTilePlanBm32` - the only difference
    /// is that an expert's `1..=16`-row remainder does not get a tile here
    /// (those rows are pairing candidates, handled by the separate
    /// `moeBuildTilePlanBm32Paired` / `moeGatherDequantGemmInt4CoopTilePlanPaired`
    /// dispatch instead). Caller zero-fills the plan buffers out to
    /// `moeTilePlanCapacityBm32` before this call, same padding contract.
    /// `tileCountGateUp`/`tileCountDown` (F-85 idle-tile-cost follow-up):
    /// same contract as `moeBuildTilePlanBm32`'s own outputs - the real
    /// "own" tile count, which is typically a much smaller fraction of the
    /// shared worst-case capacity than the unpaired builder's own count
    /// (stripping the pairing candidates out shrinks the real count while
    /// the capacity bound stays the same), so this leg's indirect-dispatch
    /// win is larger whenever the pairing lever is opted into.
    public static func moeBuildTilePlanBm32Own(
        sortedExperts: Tensor,
        tileExpert: Tensor, tileRowStart: Tensor, tileRowCount: Tensor,
        tileCountGateUp: Tensor, tileCountDown: Tensor,
        mTotal: Int, nExperts: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            sortedExperts.dtype == .u32, "moeBuildTilePlanBm32Own: sortedExperts must be u32")
        precondition(
            tileExpert.dtype == .u32 && tileRowStart.dtype == .u32 && tileRowCount.dtype == .u32,
            "moeBuildTilePlanBm32Own: tile_expert/tile_row_start/tile_row_count must be u32")
        precondition(
            tileCountGateUp.dtype == .u32 && tileCountDown.dtype == .u32,
            "moeBuildTilePlanBm32Own: tileCountGateUp/tileCountDown must be u32")
        precondition(
            nExperts <= 256,
            "moeBuildTilePlanBm32Own: nExperts (\(nExperts)) exceeds the kernel's 256-lane threadgroup scratch capacity"
        )
        let grid = MTLSize(width: nExperts, height: 1, depth: 1)
        let tg = MTLSize(width: nExperts, height: 1, depth: 1)
        IronKernels.iron_moe_build_tile_plan_bm32_own_f32(
            sorted_experts: sortedExperts.buffer, sorted_expertsOffset: sortedExperts.offset,
            tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
            tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
            tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
            tile_count_gateup: tileCountGateUp.buffer, tile_count_gateupOffset: tileCountGateUp.offset,
            tile_count_down: tileCountDown.buffer, tile_count_downOffset: tileCountDown.offset,
            m_total: UInt32(mTotal), n_experts: UInt32(nExperts),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    /// Phase 1 of the parallelized tile-plan builders (F-85 small-M
    /// follow-up, `iron_moe_tile_plan_expert_counts`). Dispatches ONE
    /// THREADGROUP PER EXPERT (`nExperts` groups, one thread each) so every
    /// expert's binary search over `sortedExperts` runs concurrently across
    /// the GPU's cores instead of serially on the single core the original
    /// builders' one-threadgroup dispatch confines all `nExperts` lanes to.
    /// Writes `expertRowBase[e]`/`expertCount[e]`, the same row-range values
    /// any of the three single-threadgroup builders would compute locally
    /// for expert `e` - shared by all three tile-height variants below,
    /// since the row range a sorted-expert array implies for an expert does
    /// not depend on the tile height it will be sliced into.
    ///
    /// Callers (`moeBuildTilePlanParallel`/`Bm32Parallel`/`Bm32OwnParallel`
    /// below) allocate `expertRowBase`/`expertCount` as fresh per-call
    /// scratch, matching `moeSortPlanCounting`'s `blockCounts`/
    /// `blockOffsets` convention - not cached, since `nExperts` is fixed
    /// per model but this keeps every caller's allocation lifetime obvious.
    public static func moeTilePlanExpertCounts(
        sortedExperts: Tensor,
        expertRowBase: Tensor, expertCount: Tensor,
        mTotal: Int, nExperts: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            sortedExperts.dtype == .u32, "moeTilePlanExpertCounts: sortedExperts must be u32")
        precondition(
            expertRowBase.dtype == .u32 && expertCount.dtype == .u32,
            "moeTilePlanExpertCounts: expertRowBase/expertCount must be u32")
        precondition(
            nExperts <= 256,
            "moeTilePlanExpertCounts: nExperts (\(nExperts)) exceeds the kernel's 256-expert scope"
        )
        let grid = MTLSize(width: nExperts, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        IronKernels.iron_moe_tile_plan_expert_counts_f32_threadgroups(
            sorted_experts: sortedExperts.buffer, sorted_expertsOffset: sortedExperts.offset,
            expert_row_base: expertRowBase.buffer, expert_row_baseOffset: expertRowBase.offset,
            expert_count: expertCount.buffer, expert_countOffset: expertCount.offset,
            m_total: UInt32(mTotal),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    /// Parallelized sibling of `moeBuildTilePlan` (F-85 small-M follow-up):
    /// dispatches `moeTilePlanExpertCounts` (phase 1, per-expert parallel)
    /// then `iron_moe_build_tile_plan_parallel` (phase 2 - prefix sum +
    /// tile emission only, no per-lane binary search left). Identical
    /// output format/contract to `moeBuildTilePlan` - same
    /// `tileExpert`/`tileRowStart`/`tileRowCount`, same worst-case capacity
    /// and zero-fill padding convention. Gated by `BUTTER_MOE_PARALLEL_PLAN`
    /// in `MoELayer.swift`; `moeBuildTilePlan` remains the escape hatch.
    public static func moeBuildTilePlanParallel(
        sortedExperts: Tensor,
        tileExpert: Tensor, tileRowStart: Tensor, tileRowCount: Tensor,
        mTotal: Int, nExperts: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            sortedExperts.dtype == .u32, "moeBuildTilePlanParallel: sortedExperts must be u32")
        precondition(
            tileExpert.dtype == .u32 && tileRowStart.dtype == .u32 && tileRowCount.dtype == .u32,
            "moeBuildTilePlanParallel: tile_expert/tile_row_start/tile_row_count must be u32")
        precondition(
            nExperts <= 256,
            "moeBuildTilePlanParallel: nExperts (\(nExperts)) exceeds the kernel's 256-lane threadgroup scratch capacity"
        )
        let rowBaseBuf = device.makeBuffer(length: nExperts * 4)
        let countBuf = device.makeBuffer(length: nExperts * 4)
        let expertRowBase = Tensor(buffer: rowBaseBuf, offset: 0, shape: [nExperts], dtype: .u32)
        let expertCount = Tensor(buffer: countBuf, offset: 0, shape: [nExperts], dtype: .u32)
        moeTilePlanExpertCounts(
            sortedExperts: sortedExperts, expertRowBase: expertRowBase, expertCount: expertCount,
            mTotal: mTotal, nExperts: nExperts, on: cmd)

        let grid = MTLSize(width: 1, height: 1, depth: 1)
        let tg = MTLSize(width: nExperts, height: 1, depth: 1)
        IronKernels.iron_moe_build_tile_plan_parallel_f32_threadgroups(
            expert_row_base: expertRowBase.buffer, expert_row_baseOffset: expertRowBase.offset,
            expert_count: expertCount.buffer, expert_countOffset: expertCount.offset,
            tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
            tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
            tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
            n_experts: UInt32(nExperts),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    /// Parallelized sibling of `moeBuildTilePlanBm32` (F-85 small-M
    /// follow-up) - same two-phase split as `moeBuildTilePlanParallel`,
    /// BM=32 tile height, including the `tileCountGateUp`/`tileCountDown`
    /// indirect-dispatch outputs. Gated by `BUTTER_MOE_PARALLEL_PLAN`;
    /// `moeBuildTilePlanBm32` remains the escape hatch.
    public static func moeBuildTilePlanBm32Parallel(
        sortedExperts: Tensor,
        tileExpert: Tensor, tileRowStart: Tensor, tileRowCount: Tensor,
        tileCountGateUp: Tensor, tileCountDown: Tensor,
        mTotal: Int, nExperts: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            sortedExperts.dtype == .u32, "moeBuildTilePlanBm32Parallel: sortedExperts must be u32")
        precondition(
            tileExpert.dtype == .u32 && tileRowStart.dtype == .u32 && tileRowCount.dtype == .u32,
            "moeBuildTilePlanBm32Parallel: tile_expert/tile_row_start/tile_row_count must be u32")
        precondition(
            tileCountGateUp.dtype == .u32 && tileCountDown.dtype == .u32,
            "moeBuildTilePlanBm32Parallel: tileCountGateUp/tileCountDown must be u32")
        precondition(
            nExperts <= 256,
            "moeBuildTilePlanBm32Parallel: nExperts (\(nExperts)) exceeds the kernel's 256-lane threadgroup scratch capacity"
        )
        let rowBaseBuf = device.makeBuffer(length: nExperts * 4)
        let countBuf = device.makeBuffer(length: nExperts * 4)
        let expertRowBase = Tensor(buffer: rowBaseBuf, offset: 0, shape: [nExperts], dtype: .u32)
        let expertCount = Tensor(buffer: countBuf, offset: 0, shape: [nExperts], dtype: .u32)
        moeTilePlanExpertCounts(
            sortedExperts: sortedExperts, expertRowBase: expertRowBase, expertCount: expertCount,
            mTotal: mTotal, nExperts: nExperts, on: cmd)

        let grid = MTLSize(width: 1, height: 1, depth: 1)
        let tg = MTLSize(width: nExperts, height: 1, depth: 1)
        IronKernels.iron_moe_build_tile_plan_bm32_parallel_f32_threadgroups(
            expert_row_base: expertRowBase.buffer, expert_row_baseOffset: expertRowBase.offset,
            expert_count: expertCount.buffer, expert_countOffset: expertCount.offset,
            tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
            tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
            tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
            tile_count_gateup: tileCountGateUp.buffer, tile_count_gateupOffset: tileCountGateUp.offset,
            tile_count_down: tileCountDown.buffer, tile_count_downOffset: tileCountDown.offset,
            n_experts: UInt32(nExperts),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    /// Parallelized sibling of `moeBuildTilePlanBm32Own` (F-85 small-M
    /// follow-up) - same two-phase split, excludes `1..=16`-row pairing
    /// candidates like the original "own" builder. Gated by
    /// `BUTTER_MOE_PARALLEL_PLAN`; `moeBuildTilePlanBm32Own` remains the
    /// escape hatch.
    public static func moeBuildTilePlanBm32OwnParallel(
        sortedExperts: Tensor,
        tileExpert: Tensor, tileRowStart: Tensor, tileRowCount: Tensor,
        tileCountGateUp: Tensor, tileCountDown: Tensor,
        mTotal: Int, nExperts: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            sortedExperts.dtype == .u32, "moeBuildTilePlanBm32OwnParallel: sortedExperts must be u32"
        )
        precondition(
            tileExpert.dtype == .u32 && tileRowStart.dtype == .u32 && tileRowCount.dtype == .u32,
            "moeBuildTilePlanBm32OwnParallel: tile_expert/tile_row_start/tile_row_count must be u32")
        precondition(
            tileCountGateUp.dtype == .u32 && tileCountDown.dtype == .u32,
            "moeBuildTilePlanBm32OwnParallel: tileCountGateUp/tileCountDown must be u32")
        precondition(
            nExperts <= 256,
            "moeBuildTilePlanBm32OwnParallel: nExperts (\(nExperts)) exceeds the kernel's 256-lane threadgroup scratch capacity"
        )
        let rowBaseBuf = device.makeBuffer(length: nExperts * 4)
        let countBuf = device.makeBuffer(length: nExperts * 4)
        let expertRowBase = Tensor(buffer: rowBaseBuf, offset: 0, shape: [nExperts], dtype: .u32)
        let expertCount = Tensor(buffer: countBuf, offset: 0, shape: [nExperts], dtype: .u32)
        moeTilePlanExpertCounts(
            sortedExperts: sortedExperts, expertRowBase: expertRowBase, expertCount: expertCount,
            mTotal: mTotal, nExperts: nExperts, on: cmd)

        let grid = MTLSize(width: 1, height: 1, depth: 1)
        let tg = MTLSize(width: nExperts, height: 1, depth: 1)
        IronKernels.iron_moe_build_tile_plan_bm32_own_parallel_f32_threadgroups(
            expert_row_base: expertRowBase.buffer, expert_row_baseOffset: expertRowBase.offset,
            expert_count: expertCount.buffer, expert_countOffset: expertCount.offset,
            tile_expert: tileExpert.buffer, tile_expertOffset: tileExpert.offset,
            tile_row_start: tileRowStart.buffer, tile_row_startOffset: tileRowStart.offset,
            tile_row_count: tileRowCount.buffer, tile_row_countOffset: tileRowCount.offset,
            tile_count_gateup: tileCountGateUp.buffer, tile_count_gateupOffset: tileCountGateUp.offset,
            tile_count_down: tileCountDown.buffer, tile_count_downOffset: tileCountDown.offset,
            n_experts: UInt32(nExperts),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    /// Device-side PAIR-ONLY BM=32 tile-plan builder for the F-85
    /// tile-pairing lever (`iron_moe_build_tile_plan_bm32_paired`). Reads
    /// the SAME `sortedExperts` array as `moeBuildTilePlanBm32Own` and
    /// produces the complementary `1..=16`-row remainder tiles those rows
    /// were excluded from, packed two experts to a tile (`_a`/`_b` halves,
    /// `tile_row_count_b == 0` marks an odd leftover's inert B half).
    /// Caller zero-fills all six plan buffers out to
    /// `moePairedTilePlanCapacity(nExperts:)` before this call - required,
    /// not just a courtesy, since the odd-leftover case depends on the
    /// zero fill.
    /// `pairCountGateUp`/`pairCountDown` (F-85 follow-up): one-element u32
    /// views the caller has already positioned at the "count" word (word 1)
    /// of its own two `MTLDispatchThreadgroupsIndirectArguments`-shaped
    /// triples - see `Ops.moeGatherDequantGemmInt4CoopTilePlanPairedIndirect`.
    /// The two legs need independent triples because they differ in
    /// `n_out` (word 0), but both receive the SAME device-computed value
    /// here (the real paired-tile count does not depend on `n_out`).
    public static func moeBuildTilePlanBm32Paired(
        sortedExperts: Tensor,
        tileExpertA: Tensor, tileRowStartA: Tensor, tileRowCountA: Tensor,
        tileExpertB: Tensor, tileRowStartB: Tensor, tileRowCountB: Tensor,
        pairCountGateUp: Tensor, pairCountDown: Tensor,
        mTotal: Int, nExperts: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            sortedExperts.dtype == .u32, "moeBuildTilePlanBm32Paired: sortedExperts must be u32")
        precondition(
            tileExpertA.dtype == .u32 && tileRowStartA.dtype == .u32 && tileRowCountA.dtype == .u32
                && tileExpertB.dtype == .u32 && tileRowStartB.dtype == .u32
                && tileRowCountB.dtype == .u32,
            "moeBuildTilePlanBm32Paired: all six tile_*_a/_b buffers must be u32")
        precondition(
            pairCountGateUp.dtype == .u32 && pairCountDown.dtype == .u32,
            "moeBuildTilePlanBm32Paired: pairCountGateUp/pairCountDown must be u32")
        precondition(
            nExperts <= 256,
            "moeBuildTilePlanBm32Paired: nExperts (\(nExperts)) exceeds the kernel's 256-lane threadgroup scratch capacity"
        )
        let grid = MTLSize(width: nExperts, height: 1, depth: 1)
        let tg = MTLSize(width: nExperts, height: 1, depth: 1)
        IronKernels.iron_moe_build_tile_plan_bm32_paired_f32(
            sorted_experts: sortedExperts.buffer, sorted_expertsOffset: sortedExperts.offset,
            tile_expert_a: tileExpertA.buffer, tile_expert_aOffset: tileExpertA.offset,
            tile_row_start_a: tileRowStartA.buffer, tile_row_start_aOffset: tileRowStartA.offset,
            tile_row_count_a: tileRowCountA.buffer, tile_row_count_aOffset: tileRowCountA.offset,
            tile_expert_b: tileExpertB.buffer, tile_expert_bOffset: tileExpertB.offset,
            tile_row_start_b: tileRowStartB.buffer, tile_row_start_bOffset: tileRowStartB.offset,
            tile_row_count_b: tileRowCountB.buffer, tile_row_count_bOffset: tileRowCountB.offset,
            pair_count_gateup: pairCountGateUp.buffer, pair_count_gateupOffset: pairCountGateUp.offset,
            pair_count_down: pairCountDown.buffer, pair_count_downOffset: pairCountDown.offset,
            m_total: UInt32(mTotal), n_experts: UInt32(nExperts),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    /// Two-expert PAIRED-tile MoE gather GEMM (`iron_moe_gather_qmm_coop_paired`,
    /// F-85 tile-pairing lever). Separate dispatch from
    /// `moeGatherDequantGemmInt4CoopTilePlan` - fed the `moeBuildTilePlanBm32Paired`
    /// plan instead of the regular tile plan, over the SAME sorted-row
    /// space (`xRows`/`out`), covering only the `1..=16`-row remainder
    /// rows that plan's "own" sibling excludes. `capacity` is
    /// `moePairedTilePlanCapacity(nExperts:)`'s host worst-case bound, not
    /// the real paired-tile count - same zero-fill/inert-tile padding
    /// contract as the unpaired sibling. Same shape requirements
    /// (`nOut % 64 == 0`, `groupSize == 64`) as the unpaired coop core -
    /// this kernel reuses its inner loop/staging geometry per-half.
    public static func moeGatherDequantGemmInt4CoopTilePlanPaired(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        tileExpertA: Tensor, tileRowStartA: Tensor, tileRowCountA: Tensor,
        tileExpertB: Tensor, tileRowStartB: Tensor, tileRowCountB: Tensor,
        xRows: Tensor,
        capacity: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            weight.dtype == .u32, "moeGatherDequantGemmInt4CoopTilePlanPaired: weight must be u32 packed")
        precondition(
            tileExpertA.dtype == .u32 && tileRowStartA.dtype == .u32 && tileRowCountA.dtype == .u32
                && tileExpertB.dtype == .u32 && tileRowStartB.dtype == .u32
                && tileRowCountB.dtype == .u32,
            "moeGatherDequantGemmInt4CoopTilePlanPaired: all six tile_*_a/_b buffers must be u32")
        precondition(
            xRows.dtype == .u32, "moeGatherDequantGemmInt4CoopTilePlanPaired: xRows must be u32")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4CoopTilePlanPaired: input/scales/biases dtype must match")
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4CoopTilePlanPaired: out dtype must match input")
        precondition(
            nOut % 64 == 0,
            "moeGatherDequantGemmInt4CoopTilePlanPaired: nOut (\(nOut)) must be multiple of 64 (BN tile)")
        precondition(
            kIn % 32 == 0,
            "moeGatherDequantGemmInt4CoopTilePlanPaired: kIn (\(kIn)) must be multiple of 32 (BK tile)")
        precondition(
            kIn.isMultiple(of: groupSize) && kIn / groupSize > 0,
            "moeGatherDequantGemmInt4CoopTilePlanPaired: groupSize (\(groupSize)) must divide kIn (\(kIn))")

        let grid = MTLSize(width: nOut / 64, height: capacity, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let groupSizeU = UInt32(groupSize)

        switch input.dtype {
        case .f32:
            IronKernels.iron_moe_gather_qmm_coop_paired_f32_threadgroups(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                tile_expert_a: tileExpertA.buffer, tile_expert_aOffset: tileExpertA.offset,
                tile_row_start_a: tileRowStartA.buffer, tile_row_start_aOffset: tileRowStartA.offset,
                tile_row_count_a: tileRowCountA.buffer, tile_row_count_aOffset: tileRowCountA.offset,
                tile_expert_b: tileExpertB.buffer, tile_expert_bOffset: tileExpertB.offset,
                tile_row_start_b: tileRowStartB.buffer, tile_row_start_bOffset: tileRowStartB.offset,
                tile_row_count_b: tileRowCountB.buffer, tile_row_count_bOffset: tileRowCountB.offset,
                x_rows: xRows.buffer, x_rowsOffset: xRows.offset,
                out: out.buffer, outOffset: out.offset,
                n_out: nOutU, k_in: kInU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_gather_qmm_coop_paired_f16_threadgroups(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                tile_expert_a: tileExpertA.buffer, tile_expert_aOffset: tileExpertA.offset,
                tile_row_start_a: tileRowStartA.buffer, tile_row_start_aOffset: tileRowStartA.offset,
                tile_row_count_a: tileRowCountA.buffer, tile_row_count_aOffset: tileRowCountA.offset,
                tile_expert_b: tileExpertB.buffer, tile_expert_bOffset: tileExpertB.offset,
                tile_row_start_b: tileRowStartB.buffer, tile_row_start_bOffset: tileRowStartB.offset,
                tile_row_count_b: tileRowCountB.buffer, tile_row_count_bOffset: tileRowCountB.offset,
                x_rows: xRows.buffer, x_rowsOffset: xRows.offset,
                out: out.buffer, outOffset: out.offset,
                n_out: nOutU, k_in: kInU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_gather_qmm_coop_paired_bf16_threadgroups(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                tile_expert_a: tileExpertA.buffer, tile_expert_aOffset: tileExpertA.offset,
                tile_row_start_a: tileRowStartA.buffer, tile_row_start_aOffset: tileRowStartA.offset,
                tile_row_count_a: tileRowCountA.buffer, tile_row_count_aOffset: tileRowCountA.offset,
                tile_expert_b: tileExpertB.buffer, tile_expert_bOffset: tileExpertB.offset,
                tile_row_start_b: tileRowStartB.buffer, tile_row_start_bOffset: tileRowStartB.offset,
                tile_row_count_b: tileRowCountB.buffer, tile_row_count_bOffset: tileRowCountB.offset,
                x_rows: xRows.buffer, x_rowsOffset: xRows.offset,
                out: out.buffer, outOffset: out.offset,
                n_out: nOutU, k_in: kInU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4CoopTilePlanPaired: unsupported dtype \(input.dtype)")
        }
    }

    /// Indirect-dispatch sibling of `moeGatherDequantGemmInt4CoopTilePlanPaired`
    /// (F-85 follow-up - the root-cause fix for the padding regression that
    /// gate documented). Same kernel, same buffer contract - the only
    /// difference is the dispatch grid comes from a device-written
    /// `MTLDispatchThreadgroupsIndirectArguments` triple
    /// (`indirectArgs`/`indirectArgsOffset`, produced by
    /// `Ops.moeBuildTilePlanBm32Paired`'s `pairCountGateUp`/`pairCountDown`
    /// outputs) instead of the host worst-case `capacity` bound, so idle
    /// tiles beyond the REAL paired-tile count launch zero threadgroups
    /// instead of running a masked-but-still-full K-loop.
    ///
    /// Hand-encoded rather than routed through a generated `_threadgroups`
    /// wrapper: the generated Swift bindings only emit a fixed
    /// `dispatchThreadgroups(_:threadsPerThreadgroup:)` call taking a
    /// host-computed `MTLSize`, with no indirect-buffer variant (iron
    /// codegen limitation - out of scope to change for this fix). The PSO
    /// itself is unchanged (same pipeline the generated wrapper uses,
    /// fetched by the same name from the shared `PSOCache`), and every
    /// `setBuffer`/`setBytes` call below mirrors the generated
    /// `..._threadgroups` wrapper's binding order index-for-index - see
    /// `iron_moe_gather_qmm_coop_paired_<dtype>_threadgroups` in
    /// `IronKernels.swift` for the reference this must stay in sync with
    /// if that kernel's signature ever changes.
    public static func moeGatherDequantGemmInt4CoopTilePlanPairedIndirect(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        tileExpertA: Tensor, tileRowStartA: Tensor, tileRowCountA: Tensor,
        tileExpertB: Tensor, tileRowStartB: Tensor, tileRowCountB: Tensor,
        xRows: Tensor,
        indirectArgs: MTLBuffer, indirectArgsOffset: Int = 0,
        nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            weight.dtype == .u32,
            "moeGatherDequantGemmInt4CoopTilePlanPairedIndirect: weight must be u32 packed")
        precondition(
            tileExpertA.dtype == .u32 && tileRowStartA.dtype == .u32 && tileRowCountA.dtype == .u32
                && tileExpertB.dtype == .u32 && tileRowStartB.dtype == .u32
                && tileRowCountB.dtype == .u32,
            "moeGatherDequantGemmInt4CoopTilePlanPairedIndirect: all six tile_*_a/_b buffers must be u32"
        )
        precondition(
            xRows.dtype == .u32,
            "moeGatherDequantGemmInt4CoopTilePlanPairedIndirect: xRows must be u32")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4CoopTilePlanPairedIndirect: input/scales/biases dtype must match"
        )
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4CoopTilePlanPairedIndirect: out dtype must match input")
        precondition(
            nOut % 64 == 0,
            "moeGatherDequantGemmInt4CoopTilePlanPairedIndirect: nOut (\(nOut)) must be multiple of 64 (BN tile)"
        )
        precondition(
            kIn % 32 == 0,
            "moeGatherDequantGemmInt4CoopTilePlanPairedIndirect: kIn (\(kIn)) must be multiple of 32 (BK tile)"
        )
        precondition(
            kIn.isMultiple(of: groupSize) && kIn / groupSize > 0,
            "moeGatherDequantGemmInt4CoopTilePlanPairedIndirect: groupSize (\(groupSize)) must divide kIn (\(kIn))"
        )

        let kernelName: String
        switch input.dtype {
        case .f32: kernelName = "iron_moe_gather_qmm_coop_paired_f32"
        case .f16: kernelName = "iron_moe_gather_qmm_coop_paired_f16"
        case .bf16: kernelName = "iron_moe_gather_qmm_coop_paired_bf16"
        default:
            fatalError(
                "Ops.moeGatherDequantGemmInt4CoopTilePlanPairedIndirect: unsupported dtype \(input.dtype)"
            )
        }
        let pso = PSOCache.shared.pipelineState(for: kernelName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(biases.buffer, offset: biases.offset, index: 3)
        enc.setBuffer(tileExpertA.buffer, offset: tileExpertA.offset, index: 4)
        enc.setBuffer(tileRowStartA.buffer, offset: tileRowStartA.offset, index: 5)
        enc.setBuffer(tileRowCountA.buffer, offset: tileRowCountA.offset, index: 6)
        enc.setBuffer(tileExpertB.buffer, offset: tileExpertB.offset, index: 7)
        enc.setBuffer(tileRowStartB.buffer, offset: tileRowStartB.offset, index: 8)
        enc.setBuffer(tileRowCountB.buffer, offset: tileRowCountB.offset, index: 9)
        enc.setBuffer(xRows.buffer, offset: xRows.offset, index: 10)
        enc.setBuffer(out.buffer, offset: out.offset, index: 11)
        var nOutV = UInt32(nOut)
        enc.setBytes(&nOutV, length: 4, index: 12)
        var kInV = UInt32(kIn)
        enc.setBytes(&kInV, length: 4, index: 13)
        var groupSizeV = UInt32(groupSize)
        enc.setBytes(&groupSizeV, length: 4, index: 14)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        enc.dispatchThreadgroups(
            indirectBuffer: indirectArgs, indirectBufferOffset: indirectArgsOffset,
            threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    // MARK: - Batched-T prefill kernels (Qwen3.5 / Qwen3.6 hybrid prefill)

    /// Multi-token Gated Delta Net recurrence over a chunk of `T` tokens.
    ///
    /// Wraps `iron_gated_delta_chunk` - same recurrence math as
    /// `gatedDeltaStep` but runs the per-token loop *inside* the kernel
    /// with the recurrent state kept in per-lane registers across the
    /// entire `T` sweep. A single dispatch replaces `T` independent
    /// `gatedDeltaStep` calls; the state buffer is read once at entry
    /// and written once at exit.
    ///
    /// Layout (matches the MLX-LM `_make_gated_delta_kernel` convention
    /// the iron-butter kernel implements):
    ///   * `q, k`  : `[T, Hk, Dk]`   row-major  (B=1)
    ///   * `v, y`  : `[T, Hv, Dv]`   row-major
    ///   * `g, beta` : `[T, Hv]`     row-major
    ///   * `stateIn / stateOut` : `[Hv, Dv, Dk]` (one state per `hv`)
    ///   * `tLen`  : `[1]` u32 - number of tokens in this chunk (runtime
    ///                scalar, NOT a constexpr; same PSO works for every
    ///                chunk length).
    ///
    /// All input tensors must share the activation dtype `T` - the
    /// kernel is emitted in f32 / f16 / bf16 variants. For Qwen3.5 the
    /// state is f32 (see `GDNStateCache.dtype`); pass q/k/v/g/beta as f32
    /// tensors as well.
    ///
    /// Dispatch: grid `(Dv, Hv)` threadgroups, 32 threads (one simdgroup)
    /// per group - identical to `iron_gated_delta_step` apart from the
    /// runtime `tLen`. `Dk % 32 == 0` invariant applies (max Dk = 256, so
    /// `n_per_t = Dk/32 ≤ 8` register entries per lane).
    public static func gatedDeltaChunk(
        q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
        stateIn: Tensor, into y: Tensor, stateOut: Tensor,
        tLen: Tensor,
        numKeyHeads: Int, numValueHeads: Int,
        keyHeadDim: Int, valueHeadDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        if let reason = OpsValidation.validateGatedDeltaStep(
            keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim,
            numKeyHeads: numKeyHeads, numValueHeads: numValueHeads
        ) {
            preconditionFailure("Ops.gatedDeltaChunk: \(reason)")
        }
        precondition(
            q.dtype == k.dtype && k.dtype == v.dtype
                && v.dtype == g.dtype && g.dtype == beta.dtype
                && beta.dtype == stateIn.dtype && stateIn.dtype == stateOut.dtype
                && stateOut.dtype == y.dtype,
            "Ops.gatedDeltaChunk: every tensor must share dtype")
        precondition(
            tLen.dtype == .u32 && tLen.elementCount == 1,
            "Ops.gatedDeltaChunk: tLen must be a [1] u32 scalar buffer")

        let lanesPerGroup = 32
        let grid = MTLSize(
            width: valueHeadDim * lanesPerGroup,
            height: numValueHeads, depth: 1)
        let tg = MTLSize(width: lanesPerGroup, height: 1, depth: 1)

        switch q.dtype {
        case .f32:
            IronKernels.iron_gated_delta_chunk_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: UInt32(keyHeadDim), dv: UInt32(valueHeadDim),
                hv: UInt32(numValueHeads), hk: UInt32(numKeyHeads),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gated_delta_chunk_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: UInt32(keyHeadDim), dv: UInt32(valueHeadDim),
                hv: UInt32(numValueHeads), hk: UInt32(numKeyHeads),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gated_delta_chunk_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: UInt32(keyHeadDim), dv: UInt32(valueHeadDim),
                hv: UInt32(numValueHeads), hk: UInt32(numKeyHeads),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedDeltaChunk: unsupported dtype \(q.dtype)")
        }
    }

    /// Transpose `[T, H, D]` → `[H, T, D]` (f16). Live-compile.
    public static func transposeTHD_f16(
        _ input: Tensor, t: Int, h: Int, d: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(input.dtype == .f16)
        let result = out ?? Tensor.empty(shape: [h, t, d], dtype: .f16)
        var tU = UInt32(t), hU = UInt32(h), dU = UInt32(d)
        let pso = PSOCache.shared.pipelineState(for: "iron_transpose_thd_f16")
        guard let enc = cmd.makeComputeCommandEncoder() else { return result }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(result.buffer, offset: result.offset, index: 1)
        enc.setBytes(&tU, length: 4, index: 2)
        enc.setBytes(&hU, length: 4, index: 3)
        enc.setBytes(&dU, length: 4, index: 4)
        // 2D grid (t, h*d) — depth axis capped on Metal.
        let hd = h * d
        enc.dispatchThreads(
            MTLSize(width: t, height: hd, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(t, 32), height: min(hd, 32), depth: 1))
        enc.endEncoding()
        return result
    }

    /// Transpose `[H, T, D]` → `[T, H, D]` (f16).
    public static func transposeHTD_f16(
        _ input: Tensor, t: Int, h: Int, d: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(input.dtype == .f16)
        let result = out ?? Tensor.empty(shape: [t, h, d], dtype: .f16)
        var tU = UInt32(t), hU = UInt32(h), dU = UInt32(d)
        let pso = PSOCache.shared.pipelineState(for: "iron_transpose_htd_f16")
        guard let enc = cmd.makeComputeCommandEncoder() else { return result }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(result.buffer, offset: result.offset, index: 1)
        enc.setBytes(&tU, length: 4, index: 2)
        enc.setBytes(&hU, length: 4, index: 3)
        enc.setBytes(&dU, length: 4, index: 4)
        let hd = h * d
        enc.dispatchThreads(
            MTLSize(width: t, height: hd, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(t, 32), height: min(hd, 32), depth: 1))
        enc.endEncoding()
        return result
    }

    /// Compact K/V `[nKV, maxSeq, D]` → packed `[nKV, kLen, D]` (f16).
    public static func compactKV_f16(
        src: Tensor, nKV: Int, maxSeq: Int, kLen: Int, headDim: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil,
        encoder: MTLComputeCommandEncoder? = nil
    ) -> Tensor {
        precondition(src.dtype == .f16 && kLen > 0 && kLen <= maxSeq)
        let result = out ?? Tensor.empty(shape: [nKV, kLen, headDim], dtype: .f16)
        var nKVU = UInt32(nKV), maxU = UInt32(maxSeq), kU = UInt32(kLen), dU = UInt32(headDim)
        let pso = PSOCache.shared.pipelineState(for: "iron_compact_kv_f16")
        let ownsEnc = encoder == nil
        guard let enc = encoder ?? cmd.makeComputeCommandEncoder() else { return result }
        enc.setComputePipelineState(pso)
        enc.setBuffer(src.buffer, offset: src.offset, index: 0)
        enc.setBuffer(result.buffer, offset: result.offset, index: 1)
        enc.setBytes(&nKVU, length: 4, index: 2)
        enc.setBytes(&maxU, length: 4, index: 3)
        enc.setBytes(&kU, length: 4, index: 4)
        enc.setBytes(&dU, length: 4, index: 5)
        let td = kLen * headDim
        enc.dispatchThreads(
            MTLSize(width: nKV, height: td, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: min(td, 256), depth: 1))
        if ownsEnc { enc.endEncoding() }
        return result
    }

    /// Batched-Q causal SDPA prefill via the simdgroup-matrix MMA kernel.
    /// Replaces `T` independent `sdpaDecode` calls (each O(K) per
    /// q-head, T separate dispatches) with one dispatch that tiles the
    /// `T × K` attention matrix 32×16 with simdgroup_matrix MMAs.
    ///
    /// Layout (per `iron_sdpa_prefill_mma` in iron-butter):
    ///   * `q`   : `[n_q_heads, q_len, head_dim]`   row-major (B=1)
    ///   * `k,v` : `[n_kv_heads, k_len, head_dim]`  row-major
    ///   * `out` : `[n_q_heads, q_len, head_dim]`
    ///
    /// `qLen` must be a multiple of 32 (BQ tile). Caller is responsible
    /// for padding (or for splitting the prefill into a 32-aligned chunk
    /// plus a per-token tail that falls back to `sdpaDecode`).
    /// `kLen ≥ qLen` covers the causal mask; the kernel internally bounds
    /// each row's K walk to its absolute position (assumes the last
    /// `qLen` queries correspond to the last `qLen` K rows — i.e. the
    /// queries are appended at the end of the K cache).
    ///
    /// Grid: `(qLen / 32, n_q_heads, 1)` threadgroups, 128 threads per
    /// group (4 simdgroups). Reduction-mode kernel.
    /// - Parameters position/kReal: W48 direct-KV extension (see
    ///   `iron_sdpa_prefill_mma`'s doc comment in iron's
    ///   `steel_attention_mma.rs`). `kLen` is now the K/V ADDRESSING
    ///   STRIDE (buffer row pitch) rather than necessarily the real
    ///   length — `position` (absolute sequence offset of this Q
    ///   chunk) and `kReal` (real, non-pad KV length; any k >= kReal
    ///   is masked to -inf alongside causal) are explicit params
    ///   instead of being implicitly derived. Both default to the
    ///   legacy compacted-KV contract when omitted (`position = kLen -
    ///   qLen`, `kReal = kLen`) — every existing call site (Hy3Text,
    ///   LagunaSdpa's compacted path) is byte-identical without
    ///   changes. Pass them explicitly only for a direct (uncompacted)
    ///   raw-strided K/V read where `kLen` (the cache's `maxSeq`) may
    ///   exceed the real content length.
    public static func sdpaPrefillMma(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        qLen: Int, kLen: Int, scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil,
        position: Int? = nil,
        kReal: Int? = nil
    ) -> Tensor {
        precondition(
            qLen > 0 && qLen % 32 == 0,
            "Ops.sdpaPrefillMma: qLen \(qLen) must be a positive multiple of 32")
        precondition(
            kLen >= qLen,
            "Ops.sdpaPrefillMma: kLen \(kLen) must be >= qLen \(qLen)")
        precondition(
            nQHeads % nKVHeads == 0,
            "Ops.sdpaPrefillMma: nQHeads \(nQHeads) must be a multiple of nKVHeads \(nKVHeads)")
        precondition(
            q.dtype == k.dtype && k.dtype == v.dtype,
            "Ops.sdpaPrefillMma: q/k/v must share dtype")
        let result = out ?? Tensor.empty(shape: [nQHeads, qLen, headDim], dtype: q.dtype)
        // Grid in THREADGROUPS: (qLen/BQ, n_q_heads, 1), TPG=128
        // (4 simdgroups). Must use dispatchThreadgroups for coop-matrix
        // MMA — dispatchThreads was a silent perf bug (≈ sdpaMulti cost).
        let threadsPerGroup = 128
        let bq = 32
        let tgGrid = MTLSize(width: qLen / bq, height: nQHeads, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let gqaFactor = nQHeads / nKVHeads
        let positionV = position ?? (kLen - qLen)
        let kRealV = kReal ?? kLen
        precondition(positionV >= 0, "Ops.sdpaPrefillMma: position \(positionV) must be >= 0")
        precondition(
            kRealV <= kLen, "Ops.sdpaPrefillMma: kReal \(kRealV) must be <= kLen \(kLen)")

        switch q.dtype {
        case .f32:
            IronKernels.iron_sdpa_prefill_mma_f32_threadgroups(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                q_len: UInt32(qLen), k_len: UInt32(kLen),
                gqa_factor: UInt32(gqaFactor),
                n_q_heads: UInt32(nQHeads), n_kv_heads: UInt32(nKVHeads),
                scale: scale,
                position: UInt32(positionV), k_real: UInt32(kRealV),
                gridSize: tgGrid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_sdpa_prefill_mma_f16_threadgroups(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                q_len: UInt32(qLen), k_len: UInt32(kLen),
                gqa_factor: UInt32(gqaFactor),
                n_q_heads: UInt32(nQHeads), n_kv_heads: UInt32(nKVHeads),
                scale: scale,
                position: UInt32(positionV), k_real: UInt32(kRealV),
                gridSize: tgGrid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_sdpa_prefill_mma_bf16_threadgroups(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                q_len: UInt32(qLen), k_len: UInt32(kLen),
                gqa_factor: UInt32(gqaFactor),
                n_q_heads: UInt32(nQHeads), n_kv_heads: UInt32(nKVHeads),
                scale: scale,
                position: UInt32(positionV), k_real: UInt32(kRealV),
                gridSize: tgGrid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sdpaPrefillMma: unsupported dtype \(q.dtype)")
        }
        return result
    }

    /// W58 fused-epilogue prefill SDPA: f16 q/k/v in, **transposed bf16**
    /// out. Identical MMA math to `sdpaPrefillMma`, but the store epilogue
    /// writes each element straight to the transposed row-major layout
    /// `out[t, q_head*head_dim + d]` (shape `[qLen, nQHeads*headDim]`) and
    /// narrows f32→f16→bf16 in-register — folding the separate
    /// `transposeHTDCastToBf16` dispatch (and the `outHTD` round trip: a full
    /// write + read of the `[nHeads, qLen, headDim]` f16 attn output per
    /// layer) into the producer kernel.
    ///
    /// The f32→f16→bf16 double-round is bit-identical to the native-layout
    /// path (`sdpaPrefillMma` stores f16 via `.cast::<f16>()`, then
    /// `transposeHTDCastToBf16` does `bfloat(float(x))`).
    ///
    /// `out` must be `[qLen, nQHeads*headDim]` bf16 (caller slices the first
    /// `nRows` rows). Same grid/tpg contract as `sdpaPrefillMma`.
    public static func sdpaPrefillMmaTransBf16(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        qLen: Int, kLen: Int, scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            qLen > 0 && qLen % 32 == 0,
            "Ops.sdpaPrefillMmaTransBf16: qLen \(qLen) must be a positive multiple of 32")
        precondition(
            kLen >= qLen,
            "Ops.sdpaPrefillMmaTransBf16: kLen \(kLen) must be >= qLen \(qLen)")
        precondition(
            nQHeads % nKVHeads == 0,
            "Ops.sdpaPrefillMmaTransBf16: nQHeads \(nQHeads) must be a multiple of nKVHeads \(nKVHeads)")
        precondition(
            q.dtype == .f16 && k.dtype == .f16 && v.dtype == .f16,
            "Ops.sdpaPrefillMmaTransBf16: q/k/v must be f16")
        precondition(
            out.dtype == .bf16, "Ops.sdpaPrefillMmaTransBf16: out must be bf16")
        precondition(
            headDim == 128, "Ops.sdpaPrefillMmaTransBf16: kernel is head_dim=128 only")
        let threadsPerGroup = 128
        let bq = 32
        let tgGrid = MTLSize(width: qLen / bq, height: nQHeads, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let gqaFactor = nQHeads / nKVHeads
        IronKernels.iron_sdpa_prefill_mma_transbf16_f16_threadgroups(
            q: q.buffer, qOffset: q.offset,
            k: k.buffer, kOffset: k.offset,
            v: v.buffer, vOffset: v.offset,
            out: out.buffer, outOffset: out.offset,
            q_len: UInt32(qLen), k_len: UInt32(kLen),
            gqa_factor: UInt32(gqaFactor),
            n_q_heads: UInt32(nQHeads), n_kv_heads: UInt32(nKVHeads),
            scale: scale,
            gridSize: tgGrid, threadgroupSize: tg, on: cmd)
    }

    // ─── MoE gather BGEMM int4 — BM=8 MPP variant ────────────────────
    //
    // Half-height counterpart of `moeGatherDequantGemmInt4` tuned for
    // topK=8 decode where `mTotal = 8`. BM=16 wastes 50% of the tile
    // rows on the trailing boundary; BM=8 fills the tile exactly.
    // Uses the MPP destination-only-cooperative path, so the kernel
    // descriptor (M=8, N=32, K=16) clears the simdgroup-scope
    // constraint that the all-cooperative `matmul2d` would reject.
    // Requires macOS 26.2+ / Apple10 GPU (M5 Max).
    /// BM=64 MPP / NAX cooperative-tensor MoE gather BGEMM. Half-height
    /// counterpart of the Bm16 default — fills `ceil(m/64)` tiles
    /// instead of `ceil(m/16)`. Sweet spot for the batched-prefill
    /// regime where `mTotal = T·topK ≥ 64`: at Qwen3.6-A3B T=32 topK=8
    /// mTotal=256 = exactly 4 BM=64 tiles, no boundary waste.
    ///
    /// Requires macOS 26.2+ on Apple10+ GPU (M5 Max) — the MPP path
    /// uses the cooperative-tensor `mpp::tensor_ops::matmul2d` per the
    /// iron MPP NAX primitive landing.
    ///
    /// FIXED (2026-07-18, round-3 MoE tile lever). The 2026-05-21 note
    /// below described a real bug, now root-caused: this call used the
    /// plain `dispatchThreads`-based wrapper with `gridSize` scaled to
    /// TOTAL THREADS (`(nOut/64)*128`), but `iron_moe_gather_qmm_mma_
    /// int4_bm64_mpp` is a coop_tile / cooperative-tensor kernel, and the
    /// codegen itself documents that `dispatchThreads` puts Metal in
    /// non-uniform-threadgroup mode. That silently produces wrong
    /// results for `matmul2d`, even when the grid divides evenly by the
    /// threadgroup size (see `emit_swift_wrapper_threadgroups` in
    /// iron-codegen/src/emit.rs). The already-shipped Hy3 int2
    /// BM64 / expert-grid path (`moeGatherDequantGemmInt2` above) never
    /// hit this because it hand-encodes `dispatchThreadgroups` directly
    /// instead of going through the generated wrapper. That is the
    /// reference fix, applied here via the codegen's `_threadgroups`
    /// wrapper variant (grid now in THREADGROUP counts, not threads).
    /// A CPU-oracle test extended with a skewed/ragged routing fixture
    /// (empty experts plus one heavy expert straddling tile boundaries,
    /// `moe_gather_qmm_mma_int4_bm64_mpp_matches_m1_skewed_ragged`)
    /// confirms the kernel's own math was already correct, so the bug
    /// was Swift-side dispatch, not the tile logic.
    ///
    /// Old warning (2026-05-21, root cause now known): dispatch shape
    /// matched the iron-std `moe_gather_qmm_mpp_bm64_correctness`
    /// test (grid = `[ceil(N/64), ceil(M/64), 1]` threadgroups, tg =
    /// `[128, 1, 1]` for 4 SGs WM=WN=2), but `forwardManyEquivalence` at
    /// T=8 mTotal=64 produced top-1 argmax 279 with logit 12.25 vs
    /// reference argmax 52290 logit 17.125. The tokens were present in
    /// both top-5 lists but with a scale drift, the signature of a
    /// non-uniform-dispatch coop_tile miscompile rather than a wrong
    /// offset. Still opt-in via `BUTTER_MOE_BGEMM_BM64=1`, default stays
    /// off pending a measured e2e win (bm16 ships the shipped baseline).
    public static func moeGatherDequantGemmInt4Bm64Mpp(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            weight.dtype == .u32,
            "moeGatherDequantGemmInt4Bm64Mpp: weight must be u32 packed")
        precondition(
            indices.dtype == .u32,
            "moeGatherDequantGemmInt4Bm64Mpp: indices must be u32")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4Bm64Mpp: input/scales/biases dtype must match")
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4Bm64Mpp: out dtype must match input")
        precondition(
            input.elementCount == mTotal * kIn,
            "moeGatherDequantGemmInt4Bm64Mpp: input elements \(input.elementCount) ≠ mTotal·kIn \(mTotal * kIn)"
        )
        precondition(
            out.elementCount == mTotal * nOut,
            "moeGatherDequantGemmInt4Bm64Mpp: out elements \(out.elementCount) ≠ mTotal·nOut \(mTotal * nOut)"
        )
        precondition(
            indices.elementCount == mTotal,
            "moeGatherDequantGemmInt4Bm64Mpp: indices elements \(indices.elementCount) ≠ mTotal \(mTotal)"
        )
        precondition(
            nOut % 64 == 0,
            "moeGatherDequantGemmInt4Bm64Mpp: nOut (\(nOut)) must be multiple of 64 (BN tile)")
        precondition(
            kIn % 32 == 0,
            "moeGatherDequantGemmInt4Bm64Mpp: kIn (\(kIn)) must be multiple of 32 (BK tile)")

        // Per `iron-std/src/butter/moe_mpp_bm64.rs`:
        //   BM = BN = 64, BK = 32. 4 SGs per TG, WM = WN = 2.
        //   Threadgroup size [128, 1, 1]. Grid [N/64, ceil(M/64), 1].
        // This is a coop_tile (cooperative-tensor matmul2d) kernel, so the
        // grid MUST be dispatched in THREADGROUP counts via the codegen's
        // `_threadgroups` wrapper (raw `dispatchThreadgroups`), matching
        // the already-shipped Hy3 int2 BM64 / expert-grid path just above.
        // The plain (thread-count, `dispatchThreads`) wrapper puts Metal
        // in non-uniform-threadgroup mode, which is documented (see
        // iron-codegen/src/emit.rs) to silently corrupt
        // cooperative-tensor kernels even on an evenly-divided grid, and
        // was the actual cause of the 2026-05-21 correctness bug this
        // kernel was gated off for.
        let tgWidth = 128
        let grid = MTLSize(
            width: nOut / 64,
            height: (mTotal + 63) / 64,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let mTotalU = UInt32(mTotal)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let groupSizeU = UInt32(groupSize)

        switch input.dtype {
        case .f32:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm64_mpp_f32_threadgroups(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm64_mpp_f16_threadgroups(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm64_mpp_bf16_threadgroups(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4Bm64Mpp: unsupported dtype \(input.dtype)")
        }
    }

    /// Fused MoE unpermute + weighted scatter-sum back to per-token
    /// outputs. Replaces the per-row `mTotal·2` scalar mul+add loop
    /// (`Tensor.filled([hidden])` × mTotal) on the Butter batched-prefill
    /// MoE path — one dispatch over T·hidden elements vs the
    /// `mTotal × 2` separate dispatches.
    ///
    /// Layout (per `iron-std/src/butter/moe.rs`):
    ///   - `expertOutputs`: `[k·B·T, hidden]` per-expert dense outputs at
    ///     expert-sorted positions
    ///   - `invPerm`: `[B·T, k]` u32 — where (token, slot) was placed in
    ///     `expertOutputs` (caller's sort step)
    ///   - `topKWeights`: `[B·T, k]` routing weights in `out.dtype`
    ///   - `out`: `[B·T, hidden]` weighted sum across `k` experts
    ///
    /// Geometry: TG=`[128, 1, 1]`, grid=`[B·T, 1, 1]` TGs. Reduction
    /// mode — one TG per token. Bandwidth-bound.
    public static func moeUnpermute(
        expertOutputs: Tensor,
        invPerm: Tensor,
        topKWeights: Tensor,
        into out: Tensor,
        nRows: Int, hidden: Int, k: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            invPerm.dtype == .u32,
            "Ops.moeUnpermute: invPerm must be u32 (got \(invPerm.dtype))")
        precondition(
            expertOutputs.dtype == topKWeights.dtype && topKWeights.dtype == out.dtype,
            "Ops.moeUnpermute: expertOutputs/topKWeights/out must share dtype")
        precondition(
            expertOutputs.elementCount == nRows * k * hidden,
            "Ops.moeUnpermute: expertOutputs has \(expertOutputs.elementCount) elements, expected nRows·k·hidden = \(nRows * k * hidden)"
        )
        precondition(
            invPerm.elementCount == nRows * k,
            "Ops.moeUnpermute: invPerm has \(invPerm.elementCount) elements, expected nRows·k = \(nRows * k)"
        )
        precondition(
            topKWeights.elementCount == nRows * k,
            "Ops.moeUnpermute: topKWeights has \(topKWeights.elementCount) elements, expected nRows·k = \(nRows * k)"
        )
        precondition(
            out.elementCount == nRows * hidden,
            "Ops.moeUnpermute: out has \(out.elementCount) elements, expected nRows·hidden = \(nRows * hidden)"
        )

        let tgWidth = 128
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let hiddenU = UInt32(hidden)
        let kU = UInt32(k)
        switch out.dtype {
        case .f32:
            IronKernels.iron_moe_unpermute_f32(
                expert_outputs: expertOutputs.buffer, expert_outputsOffset: expertOutputs.offset,
                inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
                top_k_weights: topKWeights.buffer, top_k_weightsOffset: topKWeights.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU, k: kU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_unpermute_f16(
                expert_outputs: expertOutputs.buffer, expert_outputsOffset: expertOutputs.offset,
                inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
                top_k_weights: topKWeights.buffer, top_k_weightsOffset: topKWeights.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU, k: kU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_unpermute_bf16(
                expert_outputs: expertOutputs.buffer, expert_outputsOffset: expertOutputs.offset,
                inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
                top_k_weights: topKWeights.buffer, top_k_weightsOffset: topKWeights.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU, k: kU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeUnpermute: unsupported dtype \(out.dtype)")
        }
    }

    /// W57: fused MoE combine tail — `moeUnpermute` + shared-expert add +
    /// residual add in ONE dispatch, replacing the Laguna prefill sorted-MoE
    /// tail's three separate `[T, hidden]` passes (`moeUnpermute` ->
    /// `Ops.add` shared -> `Ops.add` residual) with one read of each input
    /// and one write of `out`. Flag-gated via
    /// `BUTTER_LAGUNA_FUSED_MOE_TAIL` (see `LagunaFusedMoeTail`); default OFF
    /// for the W57 A/B, zero behavior change unless set.
    ///
    /// `sharedExpert` is optional — pass `nil` on paths/layers with no
    /// shared expert (e.g. `MoEEnvFlags.noShared`, or a model variant
    /// without one). When `nil`, `residual` is bound at the kernel's
    /// `shared_expert_out` slot as a same-shape dummy buffer and
    /// `has_shared=0` guards the load so it is never read — matching the
    /// iron kernel's documented dummy-buffer contract
    /// (`iron_moe_unpermute_tail` in `moe/moe_unpermute_tail.rs`).
    ///
    /// Accumulation order matches the unfused call sites exactly (unpermute
    /// -> += shared -> += residual, kept in one f32 accumulator instead of
    /// round-tripping through `T` between steps) — see that kernel's file
    /// header for why this is not a reassociation and should be
    /// byte-identical, not just within-tolerance.
    public static func moeUnpermuteTail(
        expertOutputs: Tensor,
        invPerm: Tensor,
        topKWeights: Tensor,
        sharedExpert: Tensor?,
        residual: Tensor,
        into out: Tensor,
        nRows: Int, hidden: Int, k: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            invPerm.dtype == .u32,
            "Ops.moeUnpermuteTail: invPerm must be u32 (got \(invPerm.dtype))")
        precondition(
            expertOutputs.dtype == topKWeights.dtype && topKWeights.dtype == out.dtype
                && residual.dtype == out.dtype,
            "Ops.moeUnpermuteTail: expertOutputs/topKWeights/residual/out must share dtype")
        if let shared = sharedExpert {
            precondition(
                shared.dtype == out.dtype,
                "Ops.moeUnpermuteTail: sharedExpert must share dtype with out")
            precondition(
                shared.elementCount == nRows * hidden,
                "Ops.moeUnpermuteTail: sharedExpert has \(shared.elementCount) elements, expected nRows·hidden = \(nRows * hidden)"
            )
        }
        precondition(
            expertOutputs.elementCount == nRows * k * hidden,
            "Ops.moeUnpermuteTail: expertOutputs has \(expertOutputs.elementCount) elements, expected nRows·k·hidden = \(nRows * k * hidden)"
        )
        precondition(
            invPerm.elementCount == nRows * k,
            "Ops.moeUnpermuteTail: invPerm has \(invPerm.elementCount) elements, expected nRows·k = \(nRows * k)"
        )
        precondition(
            topKWeights.elementCount == nRows * k,
            "Ops.moeUnpermuteTail: topKWeights has \(topKWeights.elementCount) elements, expected nRows·k = \(nRows * k)"
        )
        precondition(
            residual.elementCount == nRows * hidden,
            "Ops.moeUnpermuteTail: residual has \(residual.elementCount) elements, expected nRows·hidden = \(nRows * hidden)"
        )
        precondition(
            out.elementCount == nRows * hidden,
            "Ops.moeUnpermuteTail: out has \(out.elementCount) elements, expected nRows·hidden = \(nRows * hidden)"
        )

        let tgWidth = 128
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let hiddenU = UInt32(hidden)
        let kU = UInt32(k)
        let hasSharedU: UInt32 = sharedExpert != nil ? 1 : 0
        // Dummy buffer for the shared_expert_out slot when there is no
        // shared expert: same-shape `residual` is a safe bind (correct
        // size, never read — has_shared=0 guards the kernel's load).
        let sharedBuf = sharedExpert ?? residual

        switch out.dtype {
        case .f32:
            IronKernels.iron_moe_unpermute_tail_f32(
                expert_outputs: expertOutputs.buffer, expert_outputsOffset: expertOutputs.offset,
                inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
                top_k_weights: topKWeights.buffer, top_k_weightsOffset: topKWeights.offset,
                shared_expert_out: sharedBuf.buffer, shared_expert_outOffset: sharedBuf.offset,
                residual: residual.buffer, residualOffset: residual.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU, k: kU, has_shared: hasSharedU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_unpermute_tail_f16(
                expert_outputs: expertOutputs.buffer, expert_outputsOffset: expertOutputs.offset,
                inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
                top_k_weights: topKWeights.buffer, top_k_weightsOffset: topKWeights.offset,
                shared_expert_out: sharedBuf.buffer, shared_expert_outOffset: sharedBuf.offset,
                residual: residual.buffer, residualOffset: residual.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU, k: kU, has_shared: hasSharedU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_unpermute_tail_bf16(
                expert_outputs: expertOutputs.buffer, expert_outputsOffset: expertOutputs.offset,
                inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
                top_k_weights: topKWeights.buffer, top_k_weightsOffset: topKWeights.offset,
                shared_expert_out: sharedBuf.buffer, shared_expert_outOffset: sharedBuf.offset,
                residual: residual.buffer, residualOffset: residual.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU, k: kU, has_shared: hasSharedU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeUnpermuteTail: unsupported dtype \(out.dtype)")
        }
    }

    public static func moeGatherDequantGemmInt4Bm8(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            weight.dtype == .u32,
            "moeGatherDequantGemmInt4Bm8: weight must be u32 packed")
        precondition(
            indices.dtype == .u32,
            "moeGatherDequantGemmInt4Bm8: indices must be u32")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4Bm8: input/scales/biases dtype must match")
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4Bm8: out dtype must match input")
        precondition(
            input.elementCount == mTotal * kIn,
            "moeGatherDequantGemmInt4Bm8: input elements \(input.elementCount) != mTotal*kIn \(mTotal*kIn)"
        )
        precondition(
            out.elementCount == mTotal * nOut,
            "moeGatherDequantGemmInt4Bm8: out elements \(out.elementCount) != mTotal*nOut \(mTotal*nOut)"
        )
        precondition(
            indices.elementCount == mTotal,
            "moeGatherDequantGemmInt4Bm8: indices elements \(indices.elementCount) != mTotal \(mTotal)"
        )
        precondition(
            nOut % 32 == 0,
            "moeGatherDequantGemmInt4Bm8: nOut \(nOut) must be multiple of 32")
        precondition(
            kIn % 32 == 0,
            "moeGatherDequantGemmInt4Bm8: kIn \(kIn) must be multiple of 32")

        // 1 simdgroup per TG, BM=8 → grid Y = ceil(m/8).
        let tgWidth = 32
        let grid = MTLSize(
            width: (nOut / 32) * tgWidth,
            height: (mTotal + 7) / 8,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let mTotalU = UInt32(mTotal)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let groupSizeU = UInt32(groupSize)

        switch input.dtype {
        case .f32:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm8_mpp_f32(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm8_mpp_f16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_gather_qmm_mma_int4_bm8_mpp_bf16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4Bm8: unsupported dtype \(input.dtype)")
        }
    }

    // ─── Fused GDN prep + recurrence step ─────────────────────────────
    //
    // One dispatch absorbs the per-head q/k RMSNorm + g/beta math +
    // the existing GDN recurrence step, collapsing the 3 host
    // commit+wait pairs in Qwen35GDNMixer.forward down to 1. Same
    // dispatch geometry as `iron_gated_delta_step`:
    //   grid = [dv, B·hv, 1]
    //   tg   = [32, 1, 1]
    // `convOut` layout: `[B, 2·Hk·Dk + Hv·Dv]` — q | k | v slabs.
    // `qNormWeight` / `kNormWeight`: `[Hk·Dk]` (pass 1.0 × invKeyScale
    // for the unweighted path).
    public static func gatedDeltaPrepStep(
        convOut: Tensor,
        aLog: Tensor, dtBias: Tensor,
        aRaw: Tensor, bRaw: Tensor,
        qNormWeight: Tensor, kNormWeight: Tensor,
        stateIn: Tensor, stateOut: Tensor, y: Tensor,
        batchSize: Int, dk: Int, dv: Int, hv: Int, hk: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            convOut.dtype == aLog.dtype && aLog.dtype == dtBias.dtype
                && dtBias.dtype == aRaw.dtype && aRaw.dtype == bRaw.dtype
                && bRaw.dtype == qNormWeight.dtype && qNormWeight.dtype == kNormWeight.dtype
                && kNormWeight.dtype == stateIn.dtype && stateIn.dtype == stateOut.dtype
                && stateOut.dtype == y.dtype,
            "Ops.gatedDeltaPrepStep: every tensor must share dtype")
        precondition(
            dk % 32 == 0,
            "Ops.gatedDeltaPrepStep: dk \(dk) must be multiple of 32")
        precondition(
            dv % 32 == 0,
            "Ops.gatedDeltaPrepStep: dv \(dv) must be multiple of 32")
        precondition(
            hv % hk == 0,
            "Ops.gatedDeltaPrepStep: hv \(hv) must be a multiple of hk \(hk) (GQA)")

        // `dispatchThreads` counts TOTAL threads per axis, so the
        // X axis is `dv` (TGs along X) × `tgWidth` (threads per TG).
        // `tgid_x` inside the kernel therefore ranges 0..dv-1, matching
        // the kernel's `dv_idx = tgid_x` contract. Y axis is one TG
        // per (batch · Hv) slab.
        let tgWidth = 32
        let grid = MTLSize(
            width: dv * tgWidth,
            height: batchSize * hv,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let dkU = UInt32(dk)
        let dvU = UInt32(dv)
        let hvU = UInt32(hv)
        let hkU = UInt32(hk)

        switch convOut.dtype {
        case .f32:
            IronKernels.iron_gated_delta_prep_step_f32(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gated_delta_prep_step_f16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gated_delta_prep_step_bf16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedDeltaPrepStep: unsupported dtype \(convOut.dtype)")
        }
    }

    /// q/k RMSNorm pre-pass for `gatedDeltaPrepChunk`. `gatedDeltaPrepChunk`'s
    /// dispatch grid is `[Dv, batchSize·hv, 1]` — every one of the `Dv`
    /// threadgroups sharing a given `(b, hv_idx)` would otherwise
    /// redundantly recompute the identical q/k RMSNorm (q/k don't depend
    /// on `dv_idx`), and GQA groups recompute it again across their
    /// shared `hk_idx`. This kernel computes it once per `(b, t, hk_idx)`
    /// — grid `[tLen, batchSize·hk, 1]`, fully parallel across tokens —
    /// and writes dense `qNormed`/`kNormed` that `gatedDeltaPrepChunk`
    /// loads directly instead of recomputing.
    ///
    /// Shape contract (all share dtype except `tLen`):
    ///   - `convOut`     : `[batchSize, tLen, 2·hk·dk + hv·dv]`
    ///   - `qNormWeight` / `kNormWeight` : `[hk·dk]`
    ///   - `qNormed` / `kNormed` : `[batchSize, tLen, hk, dk]` (output)
    public static func gatedDeltaQKNormPrepass(
        convOut: Tensor,
        qNormWeight: Tensor, kNormWeight: Tensor,
        qNormed: Tensor, kNormed: Tensor,
        tLen: Tensor, tCount: Int,
        batchSize: Int, dk: Int, dv: Int, hv: Int, hk: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            convOut.dtype == qNormWeight.dtype && qNormWeight.dtype == kNormWeight.dtype
                && kNormWeight.dtype == qNormed.dtype && qNormed.dtype == kNormed.dtype,
            "Ops.gatedDeltaQKNormPrepass: every tensor must share dtype")
        precondition(
            dk % 32 == 0,
            "Ops.gatedDeltaQKNormPrepass: dk (\(dk)) must be a multiple of 32")
        precondition(
            hv % hk == 0,
            "Ops.gatedDeltaQKNormPrepass: hv (\(hv)) must be a multiple of hk (\(hk)) (GQA)")
        precondition(
            tLen.dtype == .u32 && tLen.elementCount == 1,
            "Ops.gatedDeltaQKNormPrepass: tLen must be a [1] u32 scalar buffer")
        // `tCount` (host Int) drives dispatch grid width; `tLen` (GPU
        // scalar) drives the kernel body's `bt = b·t_total + t` — same
        // split as `gatedDeltaPrepChunk`'s runtime-`tLen` convention, just
        // that here T is also a grid dimension so the host needs it too.
        let tgWidth = 32
        let grid = MTLSize(width: tCount * tgWidth, height: batchSize * hk, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let dkU = UInt32(dk)
        let dvU = UInt32(dv)
        let hvU = UInt32(hv)
        let hkU = UInt32(hk)
        switch convOut.dtype {
        case .f32:
            IronKernels.iron_gated_delta_qknorm_prepass_f32(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gated_delta_qknorm_prepass_f16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gated_delta_qknorm_prepass_bf16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedDeltaQKNormPrepass: unsupported dtype \(convOut.dtype)")
        }
    }

    /// Chunked-prefill counterpart to `gatedDeltaPrepStep`. Runs the
    /// fused GDN prep + recurrence over a contiguous range of tokens
    /// in one dispatch instead of looping the single-token kernel,
    /// keeping the per-head recurrent state in registers across the
    /// chunk. `tLen` is a GPU-resident `[1]` u32 scalar so the host
    /// can chain chunked dispatches without a CPU readback for the
    /// chunk-size constant.
    ///
    /// Shape contract (all share dtype except `tLen`):
    ///   - `convOut`   : `[batchSize, tLen, hv·dv]`
    ///   - `aLog`      : `[hv]` per-head log-decay
    ///   - `dtBias`    : `[hv]` per-head delta-t bias
    ///   - `aRaw`/`bRaw` : `[batchSize, tLen, hv]` chunk gates
    ///   - `qNormed` / `kNormed` : `[batchSize, tLen, hk, dk]`, from
    ///     `gatedDeltaQKNormPrepass` — must be dispatched on the same
    ///     `convOut`/`tLen` chunk beforehand.
    ///   - `stateIn` / `stateOut` : `[batchSize, hv, dv, dk]` fp32
    ///     recurrent state (read-once / write-once across the chunk)
    ///   - `y`         : `[batchSize, tLen, hv·dv]` output activations
    /// `statePlanes` / `planesEnabled` — spec-decode rollback side channel
    /// (see `iron_gated_delta_prep_chunk`'s "Per-token state planes" doc).
    /// `statePlanes` must be `[T, batchSize·hv, dv, dk]` sized (same dtype
    /// as everything else) when `planesEnabled` carries a nonzero u32, and
    /// may be any nonempty buffer of the right dtype (never indexed)
    /// otherwise — pass `Ops.disabledGdnPlanes(dtype:device:)` for the
    /// common "planes off" case.
    public static func gatedDeltaPrepChunk(
        convOut: Tensor,
        aLog: Tensor, dtBias: Tensor,
        aRaw: Tensor, bRaw: Tensor,
        qNormed: Tensor, kNormed: Tensor,
        stateIn: Tensor, stateOut: Tensor, y: Tensor,
        tLen: Tensor,
        statePlanes: Tensor, planesEnabled: Tensor,
        batchSize: Int, dk: Int, dv: Int, hv: Int, hk: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            convOut.dtype == aLog.dtype && aLog.dtype == dtBias.dtype
                && dtBias.dtype == aRaw.dtype && aRaw.dtype == bRaw.dtype
                && bRaw.dtype == qNormed.dtype && qNormed.dtype == kNormed.dtype
                && kNormed.dtype == stateIn.dtype && stateIn.dtype == stateOut.dtype
                && stateOut.dtype == y.dtype && y.dtype == statePlanes.dtype,
            "Ops.gatedDeltaPrepChunk: every tensor must share dtype")
        precondition(
            dk % 32 == 0,
            "Ops.gatedDeltaPrepChunk: dk (\(dk)) must be a multiple of 32")
        precondition(
            dv % 32 == 0,
            "Ops.gatedDeltaPrepChunk: dv (\(dv)) must be a multiple of 32")
        precondition(
            hv % hk == 0,
            "Ops.gatedDeltaPrepChunk: hv (\(hv)) must be a multiple of hk (\(hk)) (GQA)")
        precondition(
            tLen.dtype == .u32 && tLen.elementCount == 1,
            "Ops.gatedDeltaPrepChunk: tLen must be a [1] u32 scalar buffer")
        precondition(
            planesEnabled.dtype == .u32 && planesEnabled.elementCount == 1,
            "Ops.gatedDeltaPrepChunk: planesEnabled must be a [1] u32 scalar buffer")
        // 4 SGs per TG pack 4 Dv slots (reference Metal GDN geometry).
        // Grid TGs: [ceil(Dv/4), batch·Hv, 1], TG 128. Kernel walks tLen
        // tokens internally with state register-resident.
        let tgWidth = 128
        let dvTiles = (dv + 3) / 4
        let grid = MTLSize(
            width: dvTiles * tgWidth,
            height: batchSize * hv,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let dkU = UInt32(dk)
        let dvU = UInt32(dv)
        let hvU = UInt32(hv)
        let hkU = UInt32(hk)
        let nTotalU = UInt32(batchSize * hv)
        switch convOut.dtype {
        case .f32:
            IronKernels.iron_gated_delta_prep_chunk_f32(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU, n_total: nTotalU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gated_delta_prep_chunk_f16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU, n_total: nTotalU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gated_delta_prep_chunk_bf16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU, n_total: nTotalU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedDeltaPrepChunk: unsupported dtype \(convOut.dtype)")
        }
    }

    /// Small reusable dummy "planes off" pair for `gatedDeltaPrepChunk(Fast)`
    /// call sites that never roll back via planes (regular decode, prefill,
    /// the `rewindState` prefix recompute). `statePlanes` is a 1-element
    /// buffer of `dtype` — never indexed since `planesEnabled` is 0 — and
    /// `planesEnabled` is a 1-element zeroed u32 buffer. Allocated fresh
    /// (4-16 bytes) rather than cached: these call sites are not hot enough
    /// to justify a shared/tagged scratch slot.
    public static func disabledGdnPlanes(dtype: DType, device: Device) -> (statePlanes: Tensor, planesEnabled: Tensor) {
        // Content is never read (planesEnabled == 0 short-circuits every
        // store in the kernel), so `statePlanes` skips the zero-fill other
        // callers give freshly allocated tensors.
        let planes = Tensor.empty(shape: [1], dtype: dtype, device: device)
        let flagBuf = device.makeBuffer(length: 4)
        memset(flagBuf.contents(), 0, 4)
        let flag = Tensor(buffer: flagBuf, offset: 0, shape: [1], dtype: .u32)
        return (planes, flag)
    }

    /// Shape-specialized fast path for `gatedDeltaPrepChunk`, valid at the
    /// Dk=Dv=128, Hk=16 production shapes with Hv=32 or Hv=48.
    ///
    /// Root cause this works around: the generic kernel's `dk`/`dv`/`hv`/`hk`
    /// are `#[constexpr]` DSL params, but the MSL backend lowers every
    /// `#[constexpr]` to an ordinary runtime `constant T &name [[buffer(N)]]`,
    /// so `n_per_t = dk / 32` is NOT a compile-time constant to the Metal
    /// shader compiler, and the per-lane `state_reg`/`k_cache` arrays can't
    /// be unrolled into registers; every iteration of the T-token recurrence
    /// pays a real indexed-memory round trip instead. This dispatches a
    /// sibling kernel (`iron_gated_delta_prep_chunk_fast_d128_128_32_16`)
    /// that bakes `Dk`/`Dv`/`Hv`/`Hk` as true Rust-level compile-time
    /// constants (`#[kernel(variants(...))]`, the same mechanism already
    /// used by `iron_gated_delta_step_record`'s shape specialization), so
    /// the Metal compiler can fully unroll and register-promote. Measured
    /// ~3.7-4.2x faster than the generic kernel at this exact shape (see
    /// `bench_gated_delta_prep_chunk_fast_t1024` / `_t4096` vs
    /// `bench_gated_delta_prep_chunk_prod_t1024` / `_t4096` in
    /// `gated_delta_prep_chunk.rs`). Same grid/TG geometry as
    /// `gatedDeltaPrepChunk`; this does not change grid-level parallelism.
    ///
    /// No `dk`/`dv`/`hv`/`hk` parameters. Callers must confirm the shape
    /// matches before calling (see `Qwen35GDNMixer.forwardManyChunked`'s
    /// guard). `batchSize`/`hv` are still needed for grid sizing.
    /// `statePlanes` / `planesEnabled` — see `gatedDeltaPrepChunk`'s doc;
    /// same contract here (`statePlanes` shape
    /// `[T, batchSize·numValueHeads, 128, 128]` when enabled).
    public static func gatedDeltaPrepChunkFast(
        convOut: Tensor,
        aLog: Tensor, dtBias: Tensor,
        aRaw: Tensor, bRaw: Tensor,
        qNormed: Tensor, kNormed: Tensor,
        stateIn: Tensor, stateOut: Tensor, y: Tensor,
        tLen: Tensor,
        statePlanes: Tensor, planesEnabled: Tensor,
        batchSize: Int, numValueHeads: Int = 32,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            convOut.dtype == aLog.dtype && aLog.dtype == dtBias.dtype
                && dtBias.dtype == aRaw.dtype && aRaw.dtype == bRaw.dtype
                && bRaw.dtype == qNormed.dtype && qNormed.dtype == kNormed.dtype
                && kNormed.dtype == stateIn.dtype && stateIn.dtype == stateOut.dtype
                && stateOut.dtype == y.dtype && y.dtype == statePlanes.dtype,
            "Ops.gatedDeltaPrepChunkFast: every tensor must share dtype")
        precondition(
            tLen.dtype == .u32 && tLen.elementCount == 1,
            "Ops.gatedDeltaPrepChunkFast: tLen must be a [1] u32 scalar buffer")
        precondition(
            planesEnabled.dtype == .u32 && planesEnabled.elementCount == 1,
            "Ops.gatedDeltaPrepChunkFast: planesEnabled must be a [1] u32 scalar buffer")
        precondition(
            numValueHeads == 32 || numValueHeads == 48,
            "Ops.gatedDeltaPrepChunkFast: numValueHeads must be 32 or 48")
        // Fixed production dimensions: Dv=128 and Hk=16. Hv selects the
        // matching compile-time kernel variant and controls grid sizing.
        let dv = 128
        let hv = numValueHeads
        let tgWidth = 128
        let dvTiles = (dv + 3) / 4
        let grid = MTLSize(
            width: dvTiles * tgWidth,
            height: batchSize * hv,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let nTotalU = UInt32(batchSize * hv)
        if hv == 48 {
            switch convOut.dtype {
            case .f32:
                IronKernels.iron_gated_delta_prep_chunk_fast_d128_128_48_16_f32(
                    conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                    a_log: aLog.buffer, a_logOffset: aLog.offset,
                    dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                    a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                    b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                    q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                    k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                    state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                    state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                    y: y.buffer, yOffset: y.offset,
                    t_len: tLen.buffer, t_lenOffset: tLen.offset,
                    state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                    planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                    n_total: nTotalU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            case .f16:
                IronKernels.iron_gated_delta_prep_chunk_fast_d128_128_48_16_f16(
                    conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                    a_log: aLog.buffer, a_logOffset: aLog.offset,
                    dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                    a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                    b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                    q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                    k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                    state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                    state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                    y: y.buffer, yOffset: y.offset,
                    t_len: tLen.buffer, t_lenOffset: tLen.offset,
                    state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                    planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                    n_total: nTotalU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            case .bf16:
                IronKernels.iron_gated_delta_prep_chunk_fast_d128_128_48_16_bf16(
                    conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                    a_log: aLog.buffer, a_logOffset: aLog.offset,
                    dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                    a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                    b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                    q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                    k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                    state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                    state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                    y: y.buffer, yOffset: y.offset,
                    t_len: tLen.buffer, t_lenOffset: tLen.offset,
                    state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                    planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                    n_total: nTotalU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            default:
                fatalError("Ops.gatedDeltaPrepChunkFast: unsupported dtype \(convOut.dtype)")
            }
            return
        }
        switch convOut.dtype {
        case .f32:
            IronKernels.iron_gated_delta_prep_chunk_fast_d128_128_32_16_f32(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                n_total: nTotalU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gated_delta_prep_chunk_fast_d128_128_32_16_f16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                n_total: nTotalU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gated_delta_prep_chunk_fast_d128_128_32_16_bf16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_normed: qNormed.buffer, q_normedOffset: qNormed.offset,
                k_normed: kNormed.buffer, k_normedOffset: kNormed.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                state_planes: statePlanes.buffer, state_planesOffset: statePlanes.offset,
                planes_enabled: planesEnabled.buffer, planes_enabledOffset: planesEnabled.offset,
                n_total: nTotalU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedDeltaPrepChunkFast: unsupported dtype \(convOut.dtype)")
        }
    }

    // ─── Chunked-WY GDN prefill (`iron_gdn_wy_plan` + `iron_gdn_wy_scan`) ──
    //
    // Opt-in alternative to `gatedDeltaPrepChunk` — see
    // `Qwen35GDNMixer.forwardManyChunked` for the `BUTTER_GDN_WY=1` gate and
    // the `t % c == 0` fallback to `gatedDeltaPrepChunk` for ragged chunks.

    /// Grow-once tagged scratch tensor for the chunked-WY GDN prefill path.
    /// Same discipline as `Hy3Text.swift`'s (file-private) `hy3ScratchTensor`
    /// — reuses `Device.intermediateScratch`'s tagged, grow-to-largest-size
    /// buffer instead of a fresh `Tensor.empty` per layer/chunk, keyed by a
    /// tag that encodes the shape so distinct shapes never alias.
    private static func wyScratchTensor(
        shape: [Int], dtype: DType, tag: String, device: Device
    ) -> Tensor {
        let n = shape.reduce(1, *)
        let bytes = max(n * dtype.byteSize, 64)
        let buf = device.intermediateScratch(tag: tag, minBytes: bytes)
        return Tensor(buffer: buf, offset: 0, shape: shape, dtype: dtype)
    }

    /// General N-D strided copy/broadcast, wrapping `iron_strided_copy_nd`
    /// (module doc: handles padded copies, transposes, broadcasts via a
    /// zero stride, and arbitrary slices). `shape`/`strides` are **element**
    /// counts, not bytes; `rank = shape.count`. `srcBuffer`/`srcByteOffset`
    /// are taken raw (rather than a `Tensor`) so callers can point at an
    /// arbitrary sub-slab of a larger buffer (e.g. the `v` slice inside
    /// `gatedDeltaWYPrefill`'s packed `conv_out`).
    ///
    /// `shape`/`strides` are packed into two fresh tiny `MTLBuffer`s per
    /// call (mirrors `forwardManyChunked`'s `tLenBuf` — a fresh buffer per
    /// dispatch, not a tagged scratch slab, since these are 2-rank/8-byte
    /// constants and a shared tagged buffer would race across pipelined
    /// command buffers).
    private static func stridedCopyNd(
        srcBuffer: MTLBuffer, srcByteOffset: Int, dtype: DType,
        shape: [Int], strides: [Int],
        outBuffer: MTLBuffer, outByteOffset: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(
            shape.count == strides.count,
            "Ops.stridedCopyNd: shape/strides rank mismatch")
        let rank = shape.count
        let nOut = shape.reduce(1, *)
        var shapeVals = shape.map { UInt32($0) }
        var strideVals = strides.map { UInt32($0) }
        let shapeBuf = device.makeBuffer(length: rank * 4)
        let stridesBuf = device.makeBuffer(length: rank * 4)
        _ = shapeVals.withUnsafeMutableBytes { raw in
            memcpy(shapeBuf.contents(), raw.baseAddress!, rank * 4)
        }
        _ = strideVals.withUnsafeMutableBytes { raw in
            memcpy(stridesBuf.contents(), raw.baseAddress!, rank * 4)
        }
        let (grid, tg) = elementwiseGrid(nOut)
        switch dtype {
        case .f32:
            IronKernels.iron_strided_copy_nd_f32(
                src: srcBuffer, srcOffset: srcByteOffset,
                shape: shapeBuf, shapeOffset: 0,
                strides: stridesBuf, stridesOffset: 0,
                out: outBuffer, outOffset: outByteOffset,
                rank: UInt32(rank),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_strided_copy_nd_f16(
                src: srcBuffer, srcOffset: srcByteOffset,
                shape: shapeBuf, shapeOffset: 0,
                strides: stridesBuf, stridesOffset: 0,
                out: outBuffer, outOffset: outByteOffset,
                rank: UInt32(rank),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_strided_copy_nd_bf16(
                src: srcBuffer, srcOffset: srcByteOffset,
                shape: shapeBuf, shapeOffset: 0,
                strides: stridesBuf, stridesOffset: 0,
                out: outBuffer, outOffset: outByteOffset,
                rank: UInt32(rank),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.stridedCopyNd: unsupported dtype \(dtype)")
        }
    }

    /// Per-token GDN gate/beta for the chunked-WY prefill pipeline
    /// (`iron_gdn_wy_plan`/`iron_gdn_wy_scan`), which — unlike
    /// `gatedDeltaPrepChunk` — consume precomputed `g`/`beta` tensors
    /// instead of computing them internally from raw `a_raw`/`b_raw` +
    /// `a_log`/`dt_bias`.
    ///
    /// Dispatches the fused `iron_gdn_gate_beta` kernel (one dispatch),
    /// replacing what used to be a ~6-dispatch elementwise chain (sigmoid,
    /// two broadcasts, add, softplus, mul, exp — see git history for the
    /// old implementation).
    ///
    /// Math (identical to `gatedDeltaPrepChunk`'s internal per-token g/beta
    /// — same un-clamped softplus `iron_gated_delta_prep_chunk` bakes in,
    /// `log(exp(x)+1)`; the fused kernel now matches that production
    /// oracle's formula exactly, superseding the old chain's use of the
    /// numerically-stable `iron_softplus` kernel — the same real value
    /// away from overflow, which the learned `a_raw`/`dt_bias` activations
    /// never approach):
    ///   - `dt   = log(exp(a_raw + dt_bias) + 1)`
    ///   - `g    = exp(-exp(a_log) * dt) = exp(negExpALog * dt)`
    ///   - `beta = sigmoid(b_raw)`
    ///
    /// `negExpALog = -exp(a_log)` is precomputed once on the host at model
    /// load (`a_log` is a static per-head weight, not runtime data — see
    /// `Qwen35GDNMixer.negExpALogTF32`).
    ///
    /// `dt_bias`/`negExpALog` are `[Hv]`; `a_raw`/`b_raw` (and `g`/`beta`)
    /// are `[rows, Hv]` (`rows = B·T`, B=1 here) — the kernel broadcasts
    /// the `[Hv]` constants against the `[rows, Hv]` runtime chunk via
    /// `idx % hv` internally (no host-side broadcast dispatch needed).
    public static func gdnGateBeta(
        aRaw: Tensor, bRaw: Tensor,
        dtBias: Tensor, negExpALog: Tensor,
        g: Tensor, beta: Tensor,
        rows: Int, hv: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(
            aRaw.elementCount == rows * hv && bRaw.elementCount == rows * hv
                && g.elementCount == rows * hv && beta.elementCount == rows * hv,
            "Ops.gdnGateBeta: aRaw/bRaw/g/beta must be [rows·hv]")
        precondition(
            dtBias.elementCount == hv && negExpALog.elementCount == hv,
            "Ops.gdnGateBeta: dtBias/negExpALog must be [hv]")
        precondition(
            aRaw.dtype == .f32 && bRaw.dtype == .f32 && dtBias.dtype == .f32
                && negExpALog.dtype == .f32 && g.dtype == .f32 && beta.dtype == .f32,
            "Ops.gdnGateBeta: fp32 only (matches gatedDeltaPrepChunk's fp32 g/beta convention)")

        let n = rows * hv
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        IronKernels.iron_gdn_gate_beta_f32(
            a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
            b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
            dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
            neg_exp_a_log: negExpALog.buffer, neg_exp_a_logOffset: negExpALog.offset,
            g: g.buffer, gOffset: g.offset,
            beta: beta.buffer, betaOffset: beta.offset,
            hv: UInt32(hv),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    /// Chunked-WY GDN prefill — two-kernel (`iron_gdn_wy_plan` +
    /// `iron_gdn_wy_scan`) alternative to `gatedDeltaPrepChunk`. Opt-in via
    /// `BUTTER_GDN_WY=1`, default OFF (see `Qwen35GDNMixer.forwardManyChunked`).
    ///
    /// `_plan`/`_scan`'s v1 require `t_len` to be an exact multiple of
    /// their fixed chunk size `c=64` — both kernels' module docs document
    /// this as: Butter pads a short prefill tail up to a multiple of `c`
    /// upstream with `g=1, β=0` no-op tokens (a true no-op on the
    /// recurrence: decay ×1, delta ×0 → state unchanged; the padded rows'
    /// `y` output is simply never read). This function IS that upstream
    /// pad: real `t` only needs to be `>= c` (the caller — see
    /// `Qwen35GDNMixer.forwardManyChunked` — still falls back to
    /// `gatedDeltaPrepChunk` for `t < c`, where there's no full chunk to
    /// pad up to). When `t` is already a multiple of `c` (the common case
    /// — `prefillStepSize` keeps every chunk but the last aligned) this is
    /// a pure passthrough with zero extra dispatches; padding only kicks
    /// in for a genuinely ragged tail/short prompt.
    ///
    /// Shape contract (B=1, matching the mixer's batching):
    ///   - `convOut`         : `[T, 2·Hk·Dk + Hv·Dv]` fp32 — the `v` slab is
    ///     sliced out of its packed row here (`iron_gdn_wy_plan` wants a
    ///     dense `[T,Hv,Dv]`, unlike `gatedDeltaPrepChunk`, which reads the
    ///     packed layout directly).
    ///   - `qNormed`/`kNormed` : `[T, Hk, Dk]` fp32, from
    ///     `gatedDeltaQKNormPrepass` (reused as-is — already the dense
    ///     layout `iron_gdn_wy_plan` wants).
    ///   - `aRaw`/`bRaw`     : `[T, Hv]` fp32, from `castToF32Two`.
    ///   - `stateIn`/`stateOut` : `[Hv, Dv, Dk]` fp32 recurrent state.
    ///   - `y`               : `[T, Hv, Dv]` fp32 (matches `gatedDeltaPrepChunk`'s `y`) —
    ///     always real-`t`-sized regardless of internal padding.
    public static func gatedDeltaWYPrefill(
        convOut: Tensor,
        dtBias: Tensor, negExpALog: Tensor,
        aRaw: Tensor, bRaw: Tensor,
        qNormed: Tensor, kNormed: Tensor,
        stateIn: Tensor, stateOut: Tensor, y: Tensor,
        tLen: Tensor, t: Int,
        dk: Int, dv: Int, hv: Int, hk: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) {
        let c = 64
        precondition(
            t >= c,
            "Ops.gatedDeltaWYPrefill: t (\(t)) must be >= c=\(c) — "
                + "caller must fall back to gatedDeltaPrepChunk when there's no full chunk")
        precondition(
            dk % 16 == 0 && dv % 16 == 0,
            "Ops.gatedDeltaWYPrefill: dk/dv (\(dk)/\(dv)) must be multiples of 16 (iron_gdn_wy_plan MMA tiling)")
        precondition(
            dv % 32 == 0,
            "Ops.gatedDeltaWYPrefill: dv (\(dv)) must be a multiple of 32 (iron_gdn_wy_scan v1 requirement)")
        precondition(
            hv % hk == 0,
            "Ops.gatedDeltaWYPrefill: hv (\(hv)) must be a multiple of hk (\(hk)) (GQA)")
        precondition(
            convOut.dtype == .f32 && qNormed.dtype == .f32 && kNormed.dtype == .f32
                && stateIn.dtype == .f32 && stateOut.dtype == .f32 && y.dtype == .f32,
            "Ops.gatedDeltaWYPrefill: fp32 only (matches the mixer's fused-prep fp32 convention)")

        let tPadded = ((t + c - 1) / c) * c
        let nc = tPadded / c
        let n = hv  // B=1
        let convDim = 2 * hk * dk + hv * dv
        let dtBytes = DType.f32.byteSize
        let ragged = tPadded != t

        // ── v: dense [Tpadded, Hv, Dv] sliced out of conv_out's packed
        // row; zero-filled tail when ragged (zero v-row is itself already
        // a no-op contribution, but the tail's real no-op-ness comes from
        // g=1/beta=0 below — v's own value there doesn't matter).
        let vAll = wyScratchTensor(
            shape: [tPadded * hv * dv], dtype: .f32, tag: "gdn_wy_v_\(tPadded)x\(hv)x\(dv)",
            device: device)
        stridedCopyNd(
            srcBuffer: convOut.buffer,
            srcByteOffset: convOut.offset + (2 * hk * dk) * dtBytes,
            dtype: .f32,
            shape: [t, hv * dv], strides: [convDim, 1],
            outBuffer: vAll.buffer, outByteOffset: vAll.offset,
            on: cmd, device: device)
        if ragged {
            let validBytes = t * hv * dv * dtBytes
            let tailBytes = (tPadded - t) * hv * dv * dtBytes
            let blit = cmd.makeBlitCommandEncoder()!
            blit.fill(
                buffer: vAll.buffer,
                range: (vAll.offset + validBytes) ..< (vAll.offset + validBytes + tailBytes),
                value: 0)
            blit.endEncoding()
        }

        // ── g/beta from a_raw/b_raw + dt_bias/negExpALog over the real
        // `t` rows; ragged tail padded to the documented (g=1, beta=0)
        // no-op via a stride-0 broadcast from a 1-element scalar buffer
        // (same broadcast idiom `stridedCopyNd` already uses for
        // dt_bias/negExpALog above — `blit.fill` can't express a non-zero
        // fp32 bit pattern like 1.0f, which isn't a uniform byte).
        let gAll = wyScratchTensor(
            shape: [tPadded * hv], dtype: .f32, tag: "gdn_wy_g_\(tPadded)x\(hv)", device: device)
        let betaAll = wyScratchTensor(
            shape: [tPadded * hv], dtype: .f32, tag: "gdn_wy_beta_\(tPadded)x\(hv)", device: device)
        gdnGateBeta(
            aRaw: aRaw, bRaw: bRaw, dtBias: dtBias, negExpALog: negExpALog,
            g: Tensor(buffer: gAll.buffer, offset: gAll.offset, shape: [t * hv], dtype: .f32),
            beta: Tensor(buffer: betaAll.buffer, offset: betaAll.offset, shape: [t * hv], dtype: .f32),
            rows: t, hv: hv, on: cmd, device: device)
        if ragged {
            let padRows = tPadded - t
            stridedCopyNd(
                srcBuffer: device.scalarBuffer(1.0), srcByteOffset: 0, dtype: .f32,
                shape: [padRows * hv], strides: [0],
                outBuffer: gAll.buffer, outByteOffset: gAll.offset + t * hv * dtBytes,
                on: cmd, device: device)
            stridedCopyNd(
                srcBuffer: device.scalarBuffer(0.0), srcByteOffset: 0, dtype: .f32,
                shape: [padRows * hv], strides: [0],
                outBuffer: betaAll.buffer, outByteOffset: betaAll.offset + t * hv * dtBytes,
                on: cmd, device: device)
        }

        // ── q/k: passthrough when already chunk-aligned; zero-padded
        // grow-once scratch copies when ragged (mirrors the ragged-T qmm
        // pad pattern in `dequantGemmDynamicM`).
        let qForKernel: Tensor
        let kForKernel: Tensor
        if ragged {
            let qPadded = wyScratchTensor(
                shape: [tPadded * hk * dk], dtype: .f32, tag: "gdn_wy_qpad_\(tPadded)x\(hk)x\(dk)",
                device: device)
            let kPadded = wyScratchTensor(
                shape: [tPadded * hk * dk], dtype: .f32, tag: "gdn_wy_kpad_\(tPadded)x\(hk)x\(dk)",
                device: device)
            let validBytes = t * hk * dk * dtBytes
            let tailBytes = (tPadded - t) * hk * dk * dtBytes
            let blit = cmd.makeBlitCommandEncoder()!
            blit.copy(
                from: qNormed.buffer, sourceOffset: qNormed.offset,
                to: qPadded.buffer, destinationOffset: qPadded.offset, size: validBytes)
            blit.copy(
                from: kNormed.buffer, sourceOffset: kNormed.offset,
                to: kPadded.buffer, destinationOffset: kPadded.offset, size: validBytes)
            blit.fill(
                buffer: qPadded.buffer,
                range: (qPadded.offset + validBytes) ..< (qPadded.offset + validBytes + tailBytes),
                value: 0)
            blit.fill(
                buffer: kPadded.buffer,
                range: (kPadded.offset + validBytes) ..< (kPadded.offset + validBytes + tailBytes),
                value: 0)
            blit.endEncoding()
            qForKernel = qPadded
            kForKernel = kPadded
        } else {
            qForKernel = qNormed
            kForKernel = kNormed
        }

        // ── y: write straight into the caller's buffer when aligned;
        // otherwise scratch at Tpadded, sliced back to the real `t` rows
        // the caller actually allocated/reads.
        let yForKernel: Tensor
        if ragged {
            yForKernel = wyScratchTensor(
                shape: [tPadded * hv * dv], dtype: .f32, tag: "gdn_wy_ypad_\(tPadded)x\(hv)x\(dv)",
                device: device)
        } else {
            yForKernel = y
        }

        // ── t_len: the plan/scan kernels need the PADDED length (their
        // `NC = t_len/c` sizes the chunk loop) — a fresh tiny buffer per
        // ragged dispatch, same discipline as `stridedCopyNd`'s
        // shape/strides buffers (not tagged scratch: a shared buffer would
        // race across pipelined command buffers).
        let tLenForKernel: Tensor
        if ragged {
            let buf = device.makeBuffer(length: 4)
            var tPaddedU32 = UInt32(tPadded)
            memcpy(buf.contents(), &tPaddedU32, 4)
            tLenForKernel = Tensor(buffer: buf, offset: 0, shape: [1], dtype: .u32)
        } else {
            tLenForKernel = tLen
        }

        // ── K1 intermediates (grow-once tagged scratch, tag by shape) ───
        let u = wyScratchTensor(
            shape: [n * nc * c * dv], dtype: .f32, tag: "gdn_wy_u_\(n)x\(nc)x\(c)x\(dv)", device: device)
        let yLocal = wyScratchTensor(
            shape: [n * nc * c * dv], dtype: .f32, tag: "gdn_wy_ylocal_\(n)x\(nc)x\(c)x\(dv)",
            device: device)
        let qEff = wyScratchTensor(
            shape: [n * nc * c * dk], dtype: .f32, tag: "gdn_wy_qeff_\(n)x\(nc)x\(c)x\(dk)",
            device: device)
        let pS = wyScratchTensor(
            shape: [n * nc * dk * dk], dtype: .f32, tag: "gdn_wy_ps_\(n)x\(nc)x\(dk)x\(dk)", device: device)
        let uS = wyScratchTensor(
            shape: [n * nc * dv * dk], dtype: .f32, tag: "gdn_wy_us_\(n)x\(nc)x\(dv)x\(dk)", device: device)

        // ── K1: iron_gdn_wy_plan — chunk-parallel, dispatchThreadgroups
        // (coop_tile/MMA kernel). Grid (TGs): [Tpadded/C, B·Hv, 1], TG [512,1,1].
        let gridK1 = MTLSize(width: nc, height: n, depth: 1)
        let tgK1 = MTLSize(width: 512, height: 1, depth: 1)
        IronKernels.iron_gdn_wy_plan_f32_threadgroups(
            q: qForKernel.buffer, qOffset: qForKernel.offset,
            k: kForKernel.buffer, kOffset: kForKernel.offset,
            v: vAll.buffer, vOffset: vAll.offset,
            g: gAll.buffer, gOffset: gAll.offset,
            beta: betaAll.buffer, betaOffset: betaAll.offset,
            t_len: tLenForKernel.buffer, t_lenOffset: tLenForKernel.offset,
            u: u.buffer, uOffset: u.offset,
            y_local: yLocal.buffer, y_localOffset: yLocal.offset,
            q_eff: qEff.buffer, q_effOffset: qEff.offset,
            p_s: pS.buffer, p_sOffset: pS.offset,
            u_s: uS.buffer, u_sOffset: uS.offset,
            dk: UInt32(dk), dv: UInt32(dv), hv: UInt32(hv), hk: UInt32(hk), c: UInt32(c),
            gridSize: gridK1, threadgroupSize: tgK1, on: cmd)

        // ── K2: iron_gdn_wy_scan — chunk-sequential, dispatchThreadgroups.
        // Grid (TGs): [Dv/32, B·Hv, 1], TG [128,1,1].
        let gridK2 = MTLSize(width: dv / 32, height: n, depth: 1)
        let tgK2 = MTLSize(width: 128, height: 1, depth: 1)
        IronKernels.iron_gdn_wy_scan_f32_threadgroups(
            q_eff: qEff.buffer, q_effOffset: qEff.offset,
            y_local: yLocal.buffer, y_localOffset: yLocal.offset,
            p_s: pS.buffer, p_sOffset: pS.offset,
            u_s: uS.buffer, u_sOffset: uS.offset,
            state_in: stateIn.buffer, state_inOffset: stateIn.offset,
            state_out: stateOut.buffer, state_outOffset: stateOut.offset,
            y: yForKernel.buffer, yOffset: yForKernel.offset,
            t_len: tLenForKernel.buffer, t_lenOffset: tLenForKernel.offset,
            dk: UInt32(dk), dv: UInt32(dv), hv: UInt32(hv), c: UInt32(c),
            gridSize: gridK2, threadgroupSize: tgK2, on: cmd)

        // ── Slice yForKernel's real-t rows back into the caller's `y`
        // when we dispatched into padded scratch.
        if ragged {
            let blit = cmd.makeBlitCommandEncoder()!
            blit.copy(
                from: yForKernel.buffer, sourceOffset: yForKernel.offset,
                to: y.buffer, destinationOffset: y.offset,
                size: t * hv * dv * dtBytes)
            blit.endEncoding()
        }
    }

    // ─── Dynamic-M batched-prefill int4 qmm ───────────────────────────
    //
    // Host-side driver around `iron_qmm_mma` (and, at bits=4 on eligible
    // shapes/hardware, the faster `iron_qmm_coop` / `iron_qmm_mma_mpp`
    // cooperative-tensor tiers; see `dispatchQmmMma`'s doc for the
    // priority order). The BN=32 kernels' dispatch grid is
    // `[N/32, ceil(M/32), 1]` × tg `[128, 1, 1]` — M is purely grid-Y
    // sized at runtime, so any `M % 32 == 0` is a clean dispatch. The
    // driver pads ragged `T → mPadded = ceil(T/32) * 32` with zero rows
    // and slices the first `T` output rows on return; zero-row contributions
    // collapse to `0` at every valid output column (`s · Σ q·x + b · Σ x`
    // with `x ≡ 0`).
    //
    // The Rust-side iron-butter counterpart lives in
    // `crates/iron-std/src/mlx/quantized_mma_dynamic_m.rs` and has
    // 7/7 GPU correctness cells including the bf16 T=4096 N=K=2048 Qwen3.6
    // production shape (cos 0.999999). This Swift wrapper is the host-side
    // equivalent — same padding contract, same dispatch geometry, dispatched
    // through the regenerated `IronKernels.iron_qmm_*` bindings.
    //
    // Use this when `T > 1` (the batched-prefill case). For T=1 the regular
    // per-token dequant path is faster (the mma kernel pays full weight-tile
    // load cost regardless of how many valid rows are in the M tile).
    public static func dequantGemmDynamicM(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        t: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        device: Device,
        into out: Tensor,
        bits: Int = 4
    ) {
        precondition(
            weight.dtype == .u32,
            "Ops.dequantGemmDynamicM: weight must be u32 packed (got \(weight.dtype))")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "Ops.dequantGemmDynamicM: input/scales/biases must share dtype")
        precondition(
            out.dtype == input.dtype,
            "Ops.dequantGemmDynamicM: out dtype must match input (got \(out.dtype) vs \(input.dtype))"
        )
        precondition(
            input.elementCount == t * kIn,
            "Ops.dequantGemmDynamicM: input has \(input.elementCount) elements, expected t*kIn = \(t * kIn)"
        )
        precondition(
            out.elementCount == t * nOut,
            "Ops.dequantGemmDynamicM: out has \(out.elementCount) elements, expected t*nOut = \(t * nOut)"
        )
        precondition(
            nOut % 32 == 0,
            "Ops.dequantGemmDynamicM: nOut (\(nOut)) must be multiple of 32 (BN tile)")
        precondition(
            kIn % 32 == 0,
            "Ops.dequantGemmDynamicM: kIn (\(kIn)) must be multiple of 32 (BK tile)")
        precondition(
            bits == 2 || bits == 4 || bits == 8,
            "Ops.dequantGemmDynamicM: only int2/int4/int8 supported (got bits=\(bits))")

        let gsPerRow = kIn / groupSize

        // ── BM8 MMA fast path (BUTTER_QMM_MMA_BM8, default OFF — opt-in
        // experiment) ───────────────────────────────────────────────
        // `iron_qmm_mma_bm8` maps M=8 to exactly one 8-row frag position
        // (bandwidth-bound marginal cost) vs the BM=2/4 small-M kernels'
        // compute-bound marginal cost below — see the kernel's Rust doc
        // comment for the full rationale (T≈1-8 spec-decode verify
        // batches). Takes precedence over BOTH the small-M fast path and
        // the padded BM=32 MMA path when enabled — checked first, before
        // either. Env is read FRESH every call (not a static let): sibling
        // flags like `qmmSmallMMax` cache at process start, which breaks
        // toggling the env mid-process from in-process tests.
        if bits == 4, nOut % 32 == 0, kIn % 32 == 0, t >= 1, t <= 16,
            ButterEnv["BUTTER_QMM_MMA_BM8"] == "1"
        {
            let mPadded8 = ((t + 7) / 8) * 8
            let grid = MTLSize(width: nOut / 32, height: mPadded8 / 8, depth: 1)
            let tg = MTLSize(width: 64, height: 1, depth: 1)
            if mPadded8 == t {
                if dispatchQmmMmaBm8(
                    weight: weight, scales: scales, biases: biases,
                    input: input, output: out,
                    n: nOut, k: kIn, gsPerRow: gsPerRow,
                    grid: grid, tg: tg, on: cmd)
                {
                    return
                }
            } else {
                // Pad up to the BM8 tile (8 or 16 for t in 1...16) rather
                // than 32: at most 7 wasted rows instead of 31. Same
                // blit-pad-slice shape as the small-M path just below.
                var handled = false
                autoreleasepool {
                    let xPad = Tensor.empty(
                        shape: [mPadded8, kIn], dtype: input.dtype, device: device)
                    let oPad = Tensor.empty(
                        shape: [mPadded8, nOut], dtype: input.dtype, device: device)
                    let validIn = t * kIn * input.dtype.byteSize
                    let tailIn = (mPadded8 - t) * kIn * input.dtype.byteSize
                    guard let b1 = cmd.makeBlitCommandEncoder() else { return }
                    b1.copy(
                        from: input.buffer, sourceOffset: input.offset,
                        to: xPad.buffer, destinationOffset: xPad.offset,
                        size: validIn)
                    if tailIn > 0 {
                        b1.fill(
                            buffer: xPad.buffer,
                            range: (xPad.offset + validIn)
                                ..< (xPad.offset + validIn + tailIn),
                            value: 0)
                    }
                    b1.endEncoding()
                    guard
                        dispatchQmmMmaBm8(
                            weight: weight, scales: scales, biases: biases,
                            input: xPad, output: oPad,
                            n: nOut, k: kIn, gsPerRow: gsPerRow,
                            grid: grid, tg: tg, on: cmd)
                    else { return }
                    let validOut = t * nOut * input.dtype.byteSize
                    guard let b2 = cmd.makeBlitCommandEncoder() else { return }
                    b2.copy(
                        from: oPad.buffer, sourceOffset: oPad.offset,
                        to: out.buffer, destinationOffset: out.offset,
                        size: validOut)
                    b2.endEncoding()
                    handled = true
                }
                if handled { return }
            }
        }

        // ── Scalar BM8 fast path (BUTTER_QMM_BM8_SCALAR, default OFF —
        // opt-in experiment) ───────────────────────────────────────────
        // `iron_qmm_bm8` extends `iron_qmm_bm4`'s own W-reuse geometry
        // (2 SG x 32 lanes, BN=8, grid [n/8, m/BM, 1]) from BM=4 to
        // BM=8: one M-tile covers T=5..8 instead of bm4 needing two
        // M-tiles (i.e. two full W reads) over that range, halving W
        // traffic again relative to bm4 at the cost of ~2x per-lane
        // register pressure (see the kernel's Rust doc comment —
        // ~800B/lane vs bm4's ~440B, near the ~1KB Apple GPU budget).
        // Motivated by `iron_qmm_mma_bm8` (the simdgroup-matrix MMA
        // BM8 kernel above) losing to bm4 at T=8 — this is the scalar
        // follow-up experiment. Takes precedence over the small-M fast
        // path below for t in 5...qmmSmallMMax when enabled. Env is
        // read FRESH every call — same reasoning as BUTTER_QMM_MMA_BM8
        // above (sibling flags like `qmmSmallMMax` cache at process
        // start, which breaks toggling this env mid-process from
        // in-process tests).
        if bits == 4, t >= 5, t <= Self.qmmSmallMMax, nOut % 8 == 0,
            ButterEnv["BUTTER_QMM_BM8_SCALAR"] == "1"
        {
            let mSmall8 = ((t + 7) / 8) * 8
            let grid = MTLSize(width: nOut / 8, height: mSmall8 / 8, depth: 1)
            let tg = MTLSize(width: 64, height: 1, depth: 1)
            if mSmall8 == t {
                if dispatchQmmBm8Scalar(
                    weight: weight, scales: scales, biases: biases,
                    input: input, output: out,
                    n: nOut, k: kIn, gsPerRow: gsPerRow,
                    grid: grid, tg: tg, on: cmd)
                {
                    return
                }
            } else {
                // T=5..7 pad up to 8 (a single BM tile) — at most 3
                // wasted rows. Same blit-pad-slice shape as the
                // small-M / BM8-MMA paths above/below.
                var handled = false
                autoreleasepool {
                    let xPad = Tensor.empty(
                        shape: [mSmall8, kIn], dtype: input.dtype, device: device)
                    let oPad = Tensor.empty(
                        shape: [mSmall8, nOut], dtype: input.dtype, device: device)
                    let validIn = t * kIn * input.dtype.byteSize
                    let tailIn = (mSmall8 - t) * kIn * input.dtype.byteSize
                    guard let b1 = cmd.makeBlitCommandEncoder() else { return }
                    b1.copy(
                        from: input.buffer, sourceOffset: input.offset,
                        to: xPad.buffer, destinationOffset: xPad.offset,
                        size: validIn)
                    if tailIn > 0 {
                        b1.fill(
                            buffer: xPad.buffer,
                            range: (xPad.offset + validIn)
                                ..< (xPad.offset + validIn + tailIn),
                            value: 0)
                    }
                    b1.endEncoding()
                    guard
                        dispatchQmmBm8Scalar(
                            weight: weight, scales: scales, biases: biases,
                            input: xPad, output: oPad,
                            n: nOut, k: kIn, gsPerRow: gsPerRow,
                            grid: grid, tg: tg, on: cmd)
                    else { return }
                    let validOut = t * nOut * input.dtype.byteSize
                    guard let b2 = cmd.makeBlitCommandEncoder() else { return }
                    b2.copy(
                        from: oPad.buffer, sourceOffset: oPad.offset,
                        to: out.buffer, destinationOffset: out.offset,
                        size: validOut)
                    b2.endEncoding()
                    handled = true
                }
                if handled { return }
            }
        }

        // Widths 5 and 7 otherwise pad to two BM4 tiles, including an
        // input copy, tail fill, and output copy for every projection.
        // Exact 3+2 and 4+3 splits keep the same two weight reads while
        // dispatching directly from and into row-offset views.
        if bits == 4, groupSize == 64, kIn % 256 == 0,
            (t == 5 || t == 7), nOut % 8 == 0,
            (input.dtype == .f16 || input.dtype == .bf16),
            ButterEnv["BUTTER_QMM_SPLIT_M57"] != "0"
        {
            let firstRows = t == 5 ? 3 : 4
            let secondRows = t - firstRows
            let dtBytes = input.dtype.byteSize
            let xFirst = Tensor(
                buffer: input.buffer, offset: input.offset,
                shape: [firstRows, kIn], dtype: input.dtype)
            let outFirst = Tensor(
                buffer: out.buffer, offset: out.offset,
                shape: [firstRows, nOut], dtype: out.dtype)
            let xSecond = Tensor(
                buffer: input.buffer,
                offset: input.offset + firstRows * kIn * dtBytes,
                shape: [secondRows, kIn], dtype: input.dtype)
            let outSecond = Tensor(
                buffer: out.buffer,
                offset: out.offset + firstRows * nOut * dtBytes,
                shape: [secondRows, nOut], dtype: out.dtype)
            let tg = MTLSize(width: 64, height: 1, depth: 1)
            let firstGrid = MTLSize(width: nOut / 8, height: 1, depth: 1)
            let firstHandled: Bool
            if firstRows == 3 {
                firstHandled = dispatchQmmBm3(
                    weight: weight, scales: scales, biases: biases,
                    input: xFirst, output: outFirst,
                    n: nOut, k: kIn, gsPerRow: gsPerRow,
                    grid: firstGrid, tg: tg, on: cmd)
            } else {
                firstHandled = dispatchQmmSmallM(
                    weight: weight, scales: scales, biases: biases,
                    input: xFirst, output: outFirst,
                    n: nOut, k: kIn, gsPerRow: gsPerRow, useBm4: true,
                    grid: firstGrid, tg: tg, on: cmd)
            }
            if firstHandled {
                let secondGrid = MTLSize(width: nOut / 8, height: 1, depth: 1)
                let secondHandled: Bool
                if secondRows == 3 {
                    secondHandled = dispatchQmmBm3(
                        weight: weight, scales: scales, biases: biases,
                        input: xSecond, output: outSecond,
                        n: nOut, k: kIn, gsPerRow: gsPerRow,
                        grid: secondGrid, tg: tg, on: cmd)
                } else {
                    secondHandled = dispatchQmmSmallM(
                        weight: weight, scales: scales, biases: biases,
                        input: xSecond, output: outSecond,
                        n: nOut, k: kIn, gsPerRow: gsPerRow, useBm4: false,
                        grid: secondGrid, tg: tg, on: cmd)
                }
                if secondHandled { return }
            }
        }

        // BM3 shares each int4 weight tile across exactly three activation
        // rows. Widths 3, 6, and 9 avoid the padding and extra register pressure
        // of the four-row route used below. Keep an opt-out for paired model
        // measurements and restrict this to the kernel's affine group-64
        // contract. Its K loop has no ragged-block bounds checks, so only
        // route widths composed of complete 256-element blocks.
        if bits == 4, groupSize == 64, kIn % 256 == 0,
            (t == 3 || t == 6
                || (t == 9 && ButterEnv["BUTTER_QMM_BM3_M9"] != "0")),
            nOut % 8 == 0,
            ButterEnv["BUTTER_QMM_BM3"] != "0"
        {
            let grid = MTLSize(width: nOut / 8, height: t / 3, depth: 1)
            let tg = MTLSize(width: 64, height: 1, depth: 1)
            if dispatchQmmBm3(
                weight: weight, scales: scales, biases: biases,
                input: input, output: out,
                n: nOut, k: kIn, gsPerRow: gsPerRow,
                grid: grid, tg: tg, on: cmd)
            {
                return
            }
        }

        // ── Small-M fast path (BUTTER_QMM_SMALL_M, default ON) ────────
        // The BM=32 tile below pads M up to 32, so a 2-row spec-decode
        // verify does 16x the needed compute and flips the pass from
        // weight-bound to compute-bound. Iron already ships BM=2 / BM=4
        // variants of the same int4 kernel (`iron_qmm_bm2` / `_bm4`,
        // 2 SG x 32 lanes, BN=8, grid [n/8, m/BM, 1]) whose whole point
        // is exactly this shape. Route T <= smallMMax through them and
        // pad only up to BM (2 or 4) instead of 32.
        if bits == 4, t >= 1, t <= Self.qmmSmallMMax, nOut % 8 == 0 {
            let useBm4 = t > 2
            let bm = useBm4 ? 4 : 2
            let mSmall = ((t + bm - 1) / bm) * bm
            let grid = MTLSize(width: nOut / 8, height: mSmall / bm, depth: 1)
            let tg = MTLSize(width: 64, height: 1, depth: 1)
            if mSmall == t {
                if dispatchQmmSmallM(
                    weight: weight, scales: scales, biases: biases,
                    input: input, output: out,
                    n: nOut, k: kIn, gsPerRow: gsPerRow, useBm4: useBm4,
                    grid: grid, tg: tg, on: cmd)
                {
                    return
                }
            } else {
                // Pad up to the SMALL BM (2 or 4) rather than 32: at most
                // 3 wasted rows instead of 31. Same blit-pad-slice shape as
                // the BM=32 path below.
                var handled = false
                autoreleasepool {
                    let xPad = Tensor.empty(
                        shape: [mSmall, kIn], dtype: input.dtype, device: device)
                    let oPad = Tensor.empty(
                        shape: [mSmall, nOut], dtype: input.dtype, device: device)
                    let validIn = t * kIn * input.dtype.byteSize
                    let tailIn = (mSmall - t) * kIn * input.dtype.byteSize
                    guard let b1 = cmd.makeBlitCommandEncoder() else { return }
                    b1.copy(
                        from: input.buffer, sourceOffset: input.offset,
                        to: xPad.buffer, destinationOffset: xPad.offset,
                        size: validIn)
                    if tailIn > 0 {
                        b1.fill(
                            buffer: xPad.buffer,
                            range: (xPad.offset + validIn)
                                ..< (xPad.offset + validIn + tailIn),
                            value: 0)
                    }
                    b1.endEncoding()
                    guard
                        dispatchQmmSmallM(
                            weight: weight, scales: scales, biases: biases,
                            input: xPad, output: oPad,
                            n: nOut, k: kIn, gsPerRow: gsPerRow, useBm4: useBm4,
                            grid: grid, tg: tg, on: cmd)
                    else { return }
                    let validOut = t * nOut * input.dtype.byteSize
                    guard let b2 = cmd.makeBlitCommandEncoder() else { return }
                    b2.copy(
                        from: oPad.buffer, sourceOffset: oPad.offset,
                        to: out.buffer, destinationOffset: out.offset,
                        size: validOut)
                    b2.endEncoding()
                    handled = true
                }
                if handled { return }
            }
        }

        let mPadded = ((t + 31) / 32) * 32

        // Fast path: T already multiple of 32 — dispatch directly without
        // padding. The hot-loop case for Qwen3.6 prefill (T = 32, 256, 1024,
        // 4096, 32768 are all naturally aligned).
        if t == mPadded {
            dispatchQmmMma(
                weight: weight, scales: scales, biases: biases,
                input: input, output: out,
                m: mPadded, n: nOut, k: kIn, gsPerRow: gsPerRow, bits: bits,
                on: cmd)
            return
        }

        // Slow path: pad X → mPadded, qmm, slice first T rows to `out`.
        // Per-call pads (not one shared intermediateScratch pair): a
        // single pad false-hazards every concurrent callMany on the same
        // cmd when T%32!=0. `Tensor.empty` / scratch-slab bumps stay unique;
        // the cmd retains the buffer until GPU completion.
        //
        // autoreleasepool is CRITICAL here: outside scratch mode each call
        // allocates a fresh pad pair via `mtlDevice.makeBuffer` (no pool),
        // and ObjC-bridged buffers are pinned by the enclosing autorelease
        // pool until it drains — hundreds of ragged callManys per prefill
        // would accumulate GBs of RSS (the DeepSeekV4 prefill freeze class).
        // The cmd retains the pads at encode time, so releasing our
        // references at pool exit is safe.
        autoreleasepool {
            let xPadded = Tensor.empty(
                shape: [mPadded, kIn], dtype: input.dtype, device: device)
            let outPadded = Tensor.empty(
                shape: [mPadded, nOut], dtype: input.dtype, device: device)
            let validInBytes = t * kIn * input.dtype.byteSize
            let tailZeroBytes = (mPadded - t) * kIn * input.dtype.byteSize
            let blit = cmd.makeBlitCommandEncoder()!
            blit.copy(
                from: input.buffer, sourceOffset: input.offset,
                to: xPadded.buffer, destinationOffset: xPadded.offset,
                size: validInBytes)
            if tailZeroBytes > 0 {
                blit.fill(
                    buffer: xPadded.buffer,
                    range: (xPadded.offset + validInBytes)
                        ..< (xPadded.offset + validInBytes + tailZeroBytes),
                    value: 0)
            }
            blit.endEncoding()
            dispatchQmmMma(
                weight: weight, scales: scales, biases: biases,
                input: xPadded, output: outPadded,
                m: mPadded, n: nOut, k: kIn, gsPerRow: gsPerRow, bits: bits,
                on: cmd)

            // Slice first T rows of outPadded → out (MTLBlit on `cmd`).
            let validOutBytes = t * nOut * input.dtype.byteSize
            let outBlit = cmd.makeBlitCommandEncoder()!
            outBlit.copy(
                from: outPadded.buffer, sourceOffset: outPadded.offset,
                to: out.buffer, destinationOffset: out.offset,
                size: validOutBytes)
            outBlit.endEncoding()
        }
    }

    /// Encode an int8 affine MPP QMM onto an existing (possibly concurrent)
    /// encoder. Same geometry as `dispatchQmmMma` bits=8 MPP path.
    /// Caller owns encoder lifetime. `m`/`n`/`k` must already satisfy
    /// tile multiples (m%32==0, n%32==0, k%32==0).
    public static func encodeDequantGemmInt8Mpp(
        input: Tensor, weight: Tensor, scales: Tensor, biases: Tensor,
        m: Int, n: Int, k: Int, groupSize: Int,
        into out: Tensor,
        encoder: MTLComputeCommandEncoder
    ) {
        precondition(weight.dtype == .u32)
        precondition(n % 32 == 0 && m % 32 == 0 && k % 32 == 0)
        precondition(k.isMultiple(of: groupSize))
        let gsPerRow = k / groupSize
        let tgGrid = MTLSize(width: n / 32, height: m / 32, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        var kU = UInt32(k)
        var nU = UInt32(n)
        var gsU = UInt32(gsPerRow)
        let psoName: String
        switch input.dtype {
        case .f32: psoName = "iron_qmm_mma_mpp_int8_f32"
        case .f16: psoName = "iron_qmm_mma_mpp_int8_f16"
        case .bf16: psoName = "iron_qmm_mma_mpp_int8_bf16"
        default: fatalError("encodeDequantGemmInt8Mpp: dtype \(input.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(weight.buffer, offset: weight.offset, index: 0)
        encoder.setBuffer(scales.buffer, offset: scales.offset, index: 1)
        encoder.setBuffer(biases.buffer, offset: biases.offset, index: 2)
        encoder.setBuffer(input.buffer, offset: input.offset, index: 3)
        encoder.setBuffer(out.buffer, offset: out.offset, index: 4)
        encoder.setBytes(&kU, length: 4, index: 5)
        encoder.setBytes(&nU, length: 4, index: 6)
        encoder.setBytes(&gsU, length: 4, index: 7)
        encoder.dispatchThreadgroups(tgGrid, threadsPerThreadgroup: tg)
    }

    /// `BUTTER_QMM_MPP` / `BUTTER_QMM_COOP` raw env strings, cached once.
    /// `dispatchQmmMma` below is the shared entry for every dense int4
    /// projection at prefill (GDN in-projections, attention QKV/out,
    /// every other quantized linear), so this fires many times per layer
    /// per forward call; re-fetching the whole process environment
    /// dictionary that often was pure overhead once the tier decision
    /// stabilizes for the life of the process.
    private static let qmmMppEnvRaw: String? =
        ButterEnv["BUTTER_QMM_MPP"]
    private static let qmmCoopEnvRaw: String? =
        ButterEnv["BUTTER_QMM_COOP"]

    /// Inner dispatcher for `dequantGemmDynamicM`. Grid is `[N/32, M/32, 1]`
    /// × tg `[128, 1, 1]` (4 SGs WM=WN=2 per the canonical `iron_qmm_mma`).
    /// `dispatchThreads` counts total threads per axis, so grid.x = N/32·128.
    /// `bits` selects the int4 (unsuffixed) or int2 variant; the int2 kernel
    /// uses the same geometry but unpacks 16 codes per u32 (vs 8 nibbles).
    ///
    /// `m` must already be a multiple of 32 (BM tile) — callers pad ragged
    /// `T` up to 32 before reaching here (`dequantGemmDynamicM`'s slow
    /// path); the `m/32` grid math below silently truncates otherwise.
    ///
    /// bits=4 additionally prefers, in order:
    ///  1. `iron_qmm_coop` (BN=64 wide-N cooperative-tensor kernel, two
    ///     `matmul2d` ops per simdgroup sharing one staged X-tile; see its
    ///     module doc for the isolated-bench case for widening past MPP's
    ///     BN=32). Requires `n % 64 == 0` in addition to the group-size
    ///     gate below; a caller with a narrower N (not a multiple of 64,
    ///     e.g. some vision-tower FFN dims) falls through to tier 2/3
    ///     regardless of the gate. Gate: `BUTTER_QMM_COOP=0` (force off) /
    ///     `=1` (force on); unset auto-detects on `Ops.mppAutoCapable`
    ///     hardware (default ON there; this is the fastest of the three
    ///     tiers on every dense int4 shape measured so far).
    ///  2. `iron_qmm_mma_mpp` (MPP/NAX cooperative-tensor kernel,
    ///     live-compiled via `MetalPerformancePrimitives`) over the manual
    ///     simdgroup ladder, same idea as the existing bits=8 MPP arm
    ///     below. Gate: `Ops.mppAutoCapable` (Apple GPU Family 9+ /
    ///     macOS 26+, the same cached capability check the MoE bgemm work
    ///     added; reused here rather than duplicated), overridable via
    ///     `BUTTER_QMM_MPP=0` (force off) / `BUTTER_QMM_MPP=1` (force on).
    ///  3. `iron_qmm_mma` (manual simdgroup ladder), the fallback for any
    ///     shape or hardware neither tier above can serve.
    /// Both cooperative-tensor kernels bake `group_size=64` (Qwen3.6-A3B's
    /// default; see their module docs); a caller quantized at a different
    /// group size falls back to the simdgroup path regardless of either
    /// gate, since those kernels would silently read scales/biases at the
    /// wrong stride otherwise.
    /// Max T routed to the BM=2 / BM=4 small-M quantized matmul kernels.
    /// 0 disables the fast path (BUTTER_QMM_SMALL_M=0).
    static let qmmSmallMMax: Int = {
        if let raw = ButterEnv["BUTTER_QMM_SMALL_M"] {
            return Int(raw) ?? 8
        }
        return 8
    }()

    /// Dispatch `iron_qmm_bm2` / `iron_qmm_bm4` for exact small M.
    /// Returns false when no kernel exists for the dtype (caller falls
    /// through to the padded BM=32 path).
    private static func dispatchQmmSmallM(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, output: Tensor,
        n: Int, k: Int, gsPerRow: Int, useBm4: Bool,
        grid: MTLSize, tg: MTLSize, on cmd: MTLCommandBuffer
    ) -> Bool {
        let kU = UInt32(k), nU = UInt32(n), gsU = UInt32(gsPerRow)
        switch (input.dtype, useBm4) {
        case (.bf16, false):
            IronKernels.iron_qmm_bm2_bf16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.bf16, true):
            IronKernels.iron_qmm_bm4_bf16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.f16, false):
            IronKernels.iron_qmm_bm2_f16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.f16, true):
            IronKernels.iron_qmm_bm4_f16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            return false
        }
        return true
    }

    /// Dispatch the exact-three-row scalar int4 kernel. Its grid is
    /// `[N/8, M/3, 1]`, with two SIMD groups producing eight output columns
    /// while reusing one weight tile across three activation rows.
    private static func dispatchQmmBm3(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, output: Tensor,
        n: Int, k: Int, gsPerRow: Int,
        grid: MTLSize, tg: MTLSize, on cmd: MTLCommandBuffer
    ) -> Bool {
        let kU = UInt32(k), nU = UInt32(n), gsU = UInt32(gsPerRow)
        switch input.dtype {
        case .f32:
            IronKernels.iron_qmm_bm3_f32_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_qmm_bm3_f16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_qmm_bm3_bf16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            return false
        }
        return true
    }

    /// Dispatch `iron_qmm_mma_bm8` (BUTTER_QMM_MMA_BM8 opt-in). Same
    /// buffer signature and quarter-height MMA geometry as `iron_qmm_mma`
    /// / `iron_qmm_mma_m16`, just BM=8: grid `[N/32, M/8, 1]`, tg
    /// `[64, 1, 1]`. Returns false when no kernel exists for the dtype
    /// (caller falls through to the small-M / padded BM=32 paths).
    private static func dispatchQmmMmaBm8(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, output: Tensor,
        n: Int, k: Int, gsPerRow: Int,
        grid: MTLSize, tg: MTLSize, on cmd: MTLCommandBuffer
    ) -> Bool {
        let kU = UInt32(k), nU = UInt32(n), gsU = UInt32(gsPerRow)
        switch input.dtype {
        case .f32:
            IronKernels.iron_qmm_mma_bm8_f32_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_qmm_mma_bm8_f16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_qmm_mma_bm8_bf16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            return false
        }
        return true
    }

    /// Dispatch `iron_qmm_bm8` (BUTTER_QMM_BM8_SCALAR opt-in). Scalar
    /// (non-MMA) extension of `iron_qmm_bm4`'s own geometry to BM=8:
    /// grid `[N/8, M/8, 1]`, tg `[64, 1, 1]` — same 2 SG x 32 lane, BN=8
    /// tile as `dispatchQmmSmallM`'s bm2/bm4, just BM=8. Returns false
    /// when no kernel exists for the dtype (caller falls through to the
    /// small-M / padded BM=32 paths).
    private static func dispatchQmmBm8Scalar(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, output: Tensor,
        n: Int, k: Int, gsPerRow: Int,
        grid: MTLSize, tg: MTLSize, on cmd: MTLCommandBuffer
    ) -> Bool {
        let kU = UInt32(k), nU = UInt32(n), gsU = UInt32(gsPerRow)
        switch input.dtype {
        case .f32:
            IronKernels.iron_qmm_bm8_f32_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_qmm_bm8_f16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_qmm_bm8_bf16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            return false
        }
        return true
    }

    private static func dispatchQmmMma(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, output: Tensor,
        m: Int, n: Int, k: Int, gsPerRow: Int, bits: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            m % 32 == 0,
            "Ops.dispatchQmmMma: m (\(m)) must be a multiple of 32 (BM tile) — caller must pad")
        // Prefer threadgroups dispatch for MMA (grid = tile counts).
        let tgWidth = 128
        let tgGrid = MTLSize(width: n / 32, height: m / 32, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        // Generated wrappers take a "total threads" grid; convert.
        let grid = MTLSize(
            width: tgGrid.width * tgWidth,
            height: tgGrid.height,
            depth: 1)
        // iron_qmm_coop's BN=64 tile halves the N-tile count vs the BN=32
        // tiers above.
        let coopGrid = MTLSize(width: n / 64, height: m / 32, depth: 1)
        let kU = UInt32(k)
        let nU = UInt32(n)
        let gsU = UInt32(gsPerRow)
        let mppRequested4: Bool
        switch Self.qmmMppEnvRaw {
        case "0": mppRequested4 = false
        case "1": mppRequested4 = true
        default: mppRequested4 = Self.mppAutoCapable
        }
        let useMpp4 = mppRequested4 && gsPerRow > 0 && k % gsPerRow == 0 && k / gsPerRow == 64
        let coopRequested4: Bool
        switch Self.qmmCoopEnvRaw {
        case "0": coopRequested4 = false
        case "1": coopRequested4 = true
        default: coopRequested4 = Self.mppAutoCapable
        }
        let useCoop4 =
            coopRequested4 && n % 64 == 0 && gsPerRow > 0 && k % gsPerRow == 0
            && k / gsPerRow == 64
        let useCoop128 =
            useCoop4 && m >= 256 && n % 128 == 0
            && ButterEnv["BUTTER_QMM_COOP_BN128"] != "0"
        switch (bits, input.dtype) {
        case (4, .f32):
            if useCoop128 {
                IronKernels.iron_qmm_coop_bn128_f32_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: MTLSize(width: n / 128, height: m / 32, depth: 1),
                    threadgroupSize: tg, on: cmd)
            } else if useCoop4 {
                IronKernels.iron_qmm_coop_f32_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: coopGrid, threadgroupSize: tg, on: cmd)
            } else if useMpp4 {
                IronKernels.iron_qmm_mma_mpp_f32_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: tgGrid, threadgroupSize: tg, on: cmd)
            } else {
                IronKernels.iron_qmm_mma_f32(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            }
        case (4, .f16):
            if useCoop128 {
                IronKernels.iron_qmm_coop_bn128_f16_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: MTLSize(width: n / 128, height: m / 32, depth: 1),
                    threadgroupSize: tg, on: cmd)
            } else if useCoop4 {
                IronKernels.iron_qmm_coop_f16_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: coopGrid, threadgroupSize: tg, on: cmd)
            } else if useMpp4 {
                IronKernels.iron_qmm_mma_mpp_f16_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: tgGrid, threadgroupSize: tg, on: cmd)
            } else {
                IronKernels.iron_qmm_mma_f16(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            }
        case (4, .bf16):
            if useCoop128 {
                IronKernels.iron_qmm_coop_bn128_bf16_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: MTLSize(width: n / 128, height: m / 32, depth: 1),
                    threadgroupSize: tg, on: cmd)
            } else if useCoop4 {
                IronKernels.iron_qmm_coop_bf16_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: coopGrid, threadgroupSize: tg, on: cmd)
            } else if useMpp4 {
                IronKernels.iron_qmm_mma_mpp_bf16_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: tgGrid, threadgroupSize: tg, on: cmd)
            } else {
                IronKernels.iron_qmm_mma_bf16(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            }
        case (2, .f32):
            IronKernels.iron_qmm_mma_int2_f32(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f16):
            IronKernels.iron_qmm_mma_int2_f16(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .bf16):
            IronKernels.iron_qmm_mma_int2_bf16(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f32):
            // Prefer MPP/NAX int8 (live-compile) unless BUTTER_HY3_QMM_MMA=1.
            // Gate on `Self.mppAutoCapable` too — same as the bits=4 tier's
            // `useMpp4` a few lines up: unlike that tier this int8 default
            // had no capability check at all, so on Apple GPU Family <9 /
            // macOS <26 it unconditionally requested the `matmul2d`
            // cooperative-tensor PSO and PSOCache.pipelineState fatalErrors
            // compiling it. `iron_qmm_mma_int8_*` below is a plain
            // simdgroup kernel and already the `BUTTER_HY3_QMM_MMA=1`
            // escape — reuse it as the capability fallback too.
            if ButterEnv["BUTTER_HY3_QMM_MMA"] == nil && Self.mppAutoCapable {
                IronKernels.iron_qmm_mma_mpp_int8_f32_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: tgGrid, threadgroupSize: tg, on: cmd)
            } else {
                IronKernels.iron_qmm_mma_int8_f32(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            }
        case (8, .f16):
            // Gate on `Self.mppAutoCapable` too — same as the bits=4 tier's
            // `useMpp4` a few lines up: unlike that tier this int8 default
            // had no capability check at all, so on Apple GPU Family <9 /
            // macOS <26 it unconditionally requested the `matmul2d`
            // cooperative-tensor PSO and PSOCache.pipelineState fatalErrors
            // compiling it. `iron_qmm_mma_int8_*` below is a plain
            // simdgroup kernel and already the `BUTTER_HY3_QMM_MMA=1`
            // escape — reuse it as the capability fallback too.
            if ButterEnv["BUTTER_HY3_QMM_MMA"] == nil && Self.mppAutoCapable {
                IronKernels.iron_qmm_mma_mpp_int8_f16_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: tgGrid, threadgroupSize: tg, on: cmd)
            } else {
                IronKernels.iron_qmm_mma_int8_f16(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            }
        case (8, .bf16):
            // Gate on `Self.mppAutoCapable` too — same as the bits=4 tier's
            // `useMpp4` a few lines up: unlike that tier this int8 default
            // had no capability check at all, so on Apple GPU Family <9 /
            // macOS <26 it unconditionally requested the `matmul2d`
            // cooperative-tensor PSO and PSOCache.pipelineState fatalErrors
            // compiling it. `iron_qmm_mma_int8_*` below is a plain
            // simdgroup kernel and already the `BUTTER_HY3_QMM_MMA=1`
            // escape — reuse it as the capability fallback too.
            if ButterEnv["BUTTER_HY3_QMM_MMA"] == nil && Self.mppAutoCapable {
                IronKernels.iron_qmm_mma_mpp_int8_bf16_threadgroups(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: tgGrid, threadgroupSize: tg, on: cmd)
            } else {
                IronKernels.iron_qmm_mma_int8_bf16(
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    x: input.buffer, xOffset: input.offset,
                    out: output.buffer, outOffset: output.offset,
                    k: kU, n: nU, gs_per_row: gsU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            }
        default:
            fatalError("Ops.dequantGemmDynamicM: unsupported (bits=\(bits), dtype=\(input.dtype))")
        }
    }

    // ─── GPU-side cast to fp32 ────────────────────────────────────────
    //
    // Wraps `iron_cast_to_f32_{bf16,f16,f32}`. Per-element copy with dtype
    // promotion. Used to bridge bf16 / f16 model activations into kernels
    // that require fp32 inputs (the fused GDN prep step is the immediate
    // consumer — it runs against the fp32 recurrence state to avoid the
    // 7-bit-mantissa drift that bf16 state would accumulate). f32→f32 is
    // a memory copy on the GPU, retained for dispatch-table uniformity.
    /// Fused silu + bf16/f16 → fp32 cast in one dispatch. Collapses the
    /// `Ops.silu(...) → Ops.castToF32(...)` two-dispatch chain used in
    /// Butter's batched-prefill GDN inner loop into one. At Qwen3.6-A3B
    /// T=512 × 30 GDN layers = 15360 dispatches removed per prefill.
    /// Input is bf16 or f16; output must be f32.
    public static func siluCastToF32(
        _ input: Tensor, into output: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            output.dtype == .f32,
            "Ops.siluCastToF32: output dtype must be f32, got \(output.dtype)")
        precondition(
            input.elementCount == output.elementCount,
            "Ops.siluCastToF32: element count mismatch (\(input.elementCount) vs \(output.elementCount))"
        )
        let n = input.elementCount
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch input.dtype {
        case .f16:
            IronKernels.iron_silu_cast_to_f32_f16(
                input: input.buffer, inputOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_silu_cast_to_f32_bf16(
                input: input.buffer, inputOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.siluCastToF32: unsupported input dtype \(input.dtype) — bf16 / f16 only")
        }
    }

    /// Cast TWO same-dtype tensors to f32 on the same compute encoder.
    /// Saves one encoder begin/end pair versus two `castToF32` calls.
    /// Used inside the Qwen3 GDN T-loop where the two raw gate inputs
    /// (`aRaw`, `bRaw`) need to be promoted to fp32 every token.
    public static func castToF32Two(
        _ a: Tensor, into outA: Tensor,
        _ b: Tensor, into outB: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            a.dtype == b.dtype,
            "Ops.castToF32Two: inputs must share dtype")
        precondition(
            outA.dtype == .f32 && outB.dtype == .f32,
            "Ops.castToF32Two: outputs must be f32")
        precondition(
            a.elementCount == outA.elementCount,
            "Ops.castToF32Two: a / outA element-count mismatch")
        precondition(
            b.elementCount == outB.elementCount,
            "Ops.castToF32Two: b / outB element-count mismatch")
        let psoName: String
        switch a.dtype {
        case .bf16: psoName = "iron_cast_to_f32_bf16"
        case .f16: psoName = "iron_cast_to_f32_f16"
        case .f32: psoName = "iron_cast_to_f32_f32"
        default: fatalError("Ops.castToF32Two: unsupported dtype \(a.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        @inline(__always)
        func dispatch(_ input: Tensor, _ out: Tensor) {
            enc.setBuffer(input.buffer, offset: input.offset, index: 0)
            enc.setBuffer(out.buffer, offset: out.offset, index: 1)
            let n = input.elementCount
            let tgWidth = min(n, 256)
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        dispatch(a, outA)
        dispatch(b, outB)
        enc.endEncoding()
    }

    /// Cast THREE same-dtype tensors to f32 on the same compute
    /// encoder. Used in the Qwen3 GDN fused-prep path where
    /// `convAct`, `aRaw`, and `bRaw` all need f32 promotion before
    /// the recurrence kernel.
    public static func castToF32Three(
        _ a: Tensor, into outA: Tensor,
        _ b: Tensor, into outB: Tensor,
        _ c: Tensor, into outC: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            a.dtype == b.dtype && b.dtype == c.dtype,
            "Ops.castToF32Three: all inputs must share dtype")
        precondition(
            outA.dtype == .f32 && outB.dtype == .f32 && outC.dtype == .f32,
            "Ops.castToF32Three: outputs must all be f32")
        precondition(
            a.elementCount == outA.elementCount,
            "Ops.castToF32Three: a / outA element-count mismatch")
        precondition(
            b.elementCount == outB.elementCount,
            "Ops.castToF32Three: b / outB element-count mismatch")
        precondition(
            c.elementCount == outC.elementCount,
            "Ops.castToF32Three: c / outC element-count mismatch")
        let psoName: String
        switch a.dtype {
        case .bf16: psoName = "iron_cast_to_f32_bf16"
        case .f16: psoName = "iron_cast_to_f32_f16"
        case .f32: psoName = "iron_cast_to_f32_f32"
        default: fatalError("Ops.castToF32Three: unsupported dtype \(a.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        @inline(__always)
        func dispatch(_ input: Tensor, _ out: Tensor) {
            enc.setBuffer(input.buffer, offset: input.offset, index: 0)
            enc.setBuffer(out.buffer, offset: out.offset, index: 1)
            let n = input.elementCount
            let tgWidth = min(n, 256)
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        dispatch(a, outA)
        dispatch(b, outB)
        dispatch(c, outC)
        enc.endEncoding()
    }

    public static func castToF32(
        _ input: Tensor, into output: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            output.dtype == .f32,
            "Ops.castToF32: output dtype must be f32, got \(output.dtype)")
        precondition(
            input.elementCount == output.elementCount,
            "Ops.castToF32: element count mismatch (\(input.elementCount) vs \(output.elementCount))"
        )
        let n = input.elementCount
        // Pick a TG width that divides n evenly when possible; cap at
        // 256 (well under the Apple TG limit of 1024).
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch input.dtype {
        case .f32:
            IronKernels.iron_cast_to_f32_f32(
                input: input.buffer, inputOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_cast_to_f32_f16(
                input: input.buffer, inputOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_cast_to_f32_bf16(
                input: input.buffer, inputOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.castToF32: unsupported input dtype \(input.dtype)")
        }
    }

    /// bf16 → f16 elementwise cast (2-byte → 2-byte). Supports **in-place**
    /// when `input` and `output` share buffer+offset (Hy3 oQ2 load path:
    /// promote checkpoint BF16 scales without doubling resident weight
    /// storage). Live-compiled from `Resources/kernels/iron_cast_bf16_to_f16.metal`.
    /// Returns a Tensor view with `dtype == .f16` over `output`'s storage.
    @discardableResult
    public static func castBf16ToF16(
        _ input: Tensor,
        into output: Tensor? = nil,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        precondition(
            input.dtype == .bf16,
            "Ops.castBf16ToF16: input must be bf16, got \(input.dtype)")
        let out: Tensor
        if let o = output {
            precondition(
                o.elementCount == input.elementCount,
                "Ops.castBf16ToF16: count mismatch")
            // In-place or separate: out may be labeled bf16 or f16.
            out = o
        } else {
            out = input  // in-place on same buffer
        }
        let n = input.elementCount
        guard n > 0 else {
            return Tensor(
                buffer: out.buffer, offset: out.offset, shape: out.shape, dtype: .f16)
        }
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let pso = PSOCache.shared.pipelineState(for: "iron_cast_bf16_to_f16_f32")
        guard let enc = cmd.makeComputeCommandEncoder() else {
            return Tensor(
                buffer: out.buffer, offset: out.offset, shape: out.shape, dtype: .f16)
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(out.buffer, offset: out.offset, index: 1)
        // T30 fix: `iron_cast_bf16_to_f16_f32` reads a bounds check
        // `constant uint &n [[buffer(2)]]` that this call site never
        // bound; the kernel's `n > tid` guard was reading whatever
        // was left in that slot by an unrelated prior encoder (Metal
        // does not zero-initialize unbound buffer slots). Usually
        // "safe" in practice because the dispatches that typically
        // precede this call in this codebase bind a large `n`
        // elsewhere first, but not guaranteed: found via
        // LagunaPrefillFusedQKRopeProbeTests where a small preceding
        // dispatch left a residual `n` smaller than this call's actual
        // element count, silently truncating the cast.
        var nU = UInt32(n)
        enc.setBytes(&nU, length: 4, index: 2)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        return Tensor(
            buffer: out.buffer, offset: out.offset, shape: out.shape, dtype: .f16)
    }

    /// One silu-cast plus two plain casts to f32 on the SAME compute
    /// encoder. Used inside the Qwen3.5 GDN T-loop where, every token,
    /// the input goes through `siluCastToF32` while `aRaw` / `bRaw`
    /// (the gate scalars) go through plain `castToF32`. The wrapper
    /// sets the silu PSO once, dispatches the silu, then switches to
    /// the plain-cast PSO and dispatches the two raw casts. Saves the
    /// encoder begin/end pair between dispatches that would otherwise
    /// fire on every GDN token.
    public static func siluCastF32PlusCastF32Two(
        siluIn: Tensor, into siluOut: Tensor,
        _ a: Tensor, into outA: Tensor,
        _ b: Tensor, into outB: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            siluIn.dtype == a.dtype && a.dtype == b.dtype,
            "Ops.siluCastF32PlusCastF32Two: all inputs must share dtype")
        precondition(
            siluOut.dtype == .f32 && outA.dtype == .f32 && outB.dtype == .f32,
            "Ops.siluCastF32PlusCastF32Two: outputs must be f32")
        precondition(
            siluIn.elementCount == siluOut.elementCount,
            "Ops.siluCastF32PlusCastF32Two: silu in/out count mismatch")
        precondition(
            a.elementCount == outA.elementCount,
            "Ops.siluCastF32PlusCastF32Two: a/outA count mismatch")
        precondition(
            b.elementCount == outB.elementCount,
            "Ops.siluCastF32PlusCastF32Two: b/outB count mismatch")
        let siluPso: String
        let castPso: String
        switch siluIn.dtype {
        case .f16:
            siluPso = "iron_silu_cast_to_f32_f16"
            castPso = "iron_cast_to_f32_f16"
        case .bf16:
            siluPso = "iron_silu_cast_to_f32_bf16"
            castPso = "iron_cast_to_f32_bf16"
        default:
            fatalError(
                "Ops.siluCastF32PlusCastF32Two: unsupported dtype \(siluIn.dtype)")
        }
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        @inline(__always)
        func dispatch(_ input: Tensor, _ out: Tensor) {
            enc.setBuffer(input.buffer, offset: input.offset, index: 0)
            enc.setBuffer(out.buffer, offset: out.offset, index: 1)
            let n = input.elementCount
            let tgWidth = min(n, 256)
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        enc.setComputePipelineState(PSOCache.shared.pipelineState(for: siluPso))
        dispatch(siluIn, siluOut)
        enc.setComputePipelineState(PSOCache.shared.pipelineState(for: castPso))
        dispatch(a, outA)
        dispatch(b, outB)
        enc.endEncoding()
    }

    // ─── Fused gated mixer norm ───────────────────────────────────────
    //
    // Wraps `iron_gated_mixer_norm_{f32,f16,bf16}`. Computes
    // `out = rms_norm(y, w) · silu(z)` per row across `[Hv, Dv]` in a
    // single dispatch. Used by Butter's Qwen3.5 / Qwen3.6 GDN mixer to
    // eliminate the phase-2 host round-trip the legacy path needed
    // for the gated norm + silu math — 30 host commit+waits per
    // Qwen3.6-A3B decode token recovered (one per GDN layer).
    //
    // Tensor contracts (matching the kernel sig):
    //   y       [Hv, Dv]  fp32     — recurrence output (fp32 state)
    //   z       [Hv, Dv]  T        — gate, in model dtype
    //   w       [Dv]      T        — `mixer.norm.weight`, in model dtype
    //   out     [Hv, Dv]  T        — gated output, in model dtype
    //   eps_buf [1]       fp32     — epsilon as a 1-element buffer
    // Constexpr:
    //   n = Dv (must be a multiple of 4 — kernel reads 4 elts/thread)
    // Dispatch:
    //   grid = [Hv * (Dv/4), 1, 1] threads (Hv TGs × Dv/4 threads per TG)
    //   tg   = [Dv/4, 1, 1] threads
    public static func gatedMixerNorm(
        y: Tensor, z: Tensor, weight: Tensor,
        epsBuf: Tensor,
        into out: Tensor,
        numValueHeads: Int, valueHeadDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            y.dtype == .f32,
            "Ops.gatedMixerNorm: y must be f32 (got \(y.dtype))")
        precondition(
            z.dtype == weight.dtype && weight.dtype == out.dtype,
            "Ops.gatedMixerNorm: z / weight / out must share dtype")
        precondition(
            epsBuf.dtype == .f32,
            "Ops.gatedMixerNorm: epsBuf must be f32")
        precondition(
            valueHeadDim.isMultiple(of: 4),
            "Ops.gatedMixerNorm: valueHeadDim (\(valueHeadDim)) must be multiple of 4")
        precondition(
            y.elementCount == numValueHeads * valueHeadDim,
            "Ops.gatedMixerNorm: y has \(y.elementCount) elements, expected \(numValueHeads * valueHeadDim)"
        )
        precondition(
            z.elementCount == numValueHeads * valueHeadDim,
            "Ops.gatedMixerNorm: z has \(z.elementCount) elements, expected \(numValueHeads * valueHeadDim)"
        )
        precondition(
            weight.elementCount == valueHeadDim,
            "Ops.gatedMixerNorm: weight has \(weight.elementCount) elements, expected \(valueHeadDim)"
        )
        precondition(
            out.elementCount == numValueHeads * valueHeadDim,
            "Ops.gatedMixerNorm: out has \(out.elementCount) elements, expected \(numValueHeads * valueHeadDim)"
        )

        // One thread per 4 consecutive Dv elements → tpg = Dv / 4.
        // `dispatchThreads` counts total threads per axis, so grid.x =
        // numValueHeads × tpg (Hv TGs × tpg threads per TG).
        let tpg = valueHeadDim / 4
        let grid = MTLSize(width: numValueHeads * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        let nU = UInt32(valueHeadDim)
        switch out.dtype {
        case .f32:
            IronKernels.iron_gated_mixer_norm_f32(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gated_mixer_norm_f16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gated_mixer_norm_bf16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedMixerNorm: unsupported out dtype \(out.dtype)")
        }
    }

    /// T-batched `gatedMixerNorm`. Same kernel, same per-row geometry
    /// (one threadgroup per row, `Dv / 4` threads per TG); the grid X
    /// axis widens from `Hv` to `T · Hv` so a single dispatch covers
    /// every prefill row. Used in the Qwen3.5 GDN mixer's prefill
    /// path where the T-loop over per-token norms was an explicit
    /// performance hotspot.
    ///
    /// Tensor shapes:
    ///   - `y` `[T, Hv, Dv]` fp32 (recurrence output stays fp32)
    ///   - `z` / `out` `[T, Hv, Dv]` in model dtype
    ///   - `weight` `[Dv]` in model dtype
    ///   - `epsBuf` `[1]` fp32
    public static func gatedMixerNormMany(
        y: Tensor, z: Tensor, weight: Tensor,
        epsBuf: Tensor,
        into out: Tensor,
        t: Int, numValueHeads: Int, valueHeadDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(t > 0, "Ops.gatedMixerNormMany: t must be positive")
        precondition(
            y.dtype == .f32,
            "Ops.gatedMixerNormMany: y must be f32 (got \(y.dtype))")
        precondition(
            z.dtype == weight.dtype && weight.dtype == out.dtype,
            "Ops.gatedMixerNormMany: z / weight / out must share dtype")
        precondition(
            epsBuf.dtype == .f32,
            "Ops.gatedMixerNormMany: epsBuf must be f32")
        precondition(
            valueHeadDim.isMultiple(of: 4),
            "Ops.gatedMixerNormMany: valueHeadDim (\(valueHeadDim)) must be a multiple of 4")
        let expected = t * numValueHeads * valueHeadDim
        precondition(
            y.elementCount == expected,
            "Ops.gatedMixerNormMany: y has \(y.elementCount), expected T·Hv·Dv = \(expected)")
        precondition(
            z.elementCount == expected,
            "Ops.gatedMixerNormMany: z has \(z.elementCount), expected \(expected)")
        precondition(
            weight.elementCount == valueHeadDim,
            "Ops.gatedMixerNormMany: weight has \(weight.elementCount), expected Dv = \(valueHeadDim)")
        precondition(
            out.elementCount == expected,
            "Ops.gatedMixerNormMany: out has \(out.elementCount), expected \(expected)")
        let tpg = valueHeadDim / 4
        let nRows = t * numValueHeads
        let grid = MTLSize(width: nRows * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        let nU = UInt32(valueHeadDim)
        switch out.dtype {
        case .f32:
            IronKernels.iron_gated_mixer_norm_f32(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gated_mixer_norm_f16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gated_mixer_norm_bf16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedMixerNormMany: unsupported out dtype \(out.dtype)")
        }
    }

    // ─── Fused RMSNorm + int4 dequant-GEMV ────────────────────────────
    //
    // Wraps `iron_rms_norm_qgemv_fast` (RMSNorm-fused) and
    // `iron_gated_rms_norm_qgemv_int4_fast` (Mamba2-gated-RMSNorm-
    // fused). Both keep the normalised activation in registers and
    // feed it straight into the int4 dequant-GEMV, replacing a
    // 2-dispatch chain (`Ops.rmsNorm` / `Ops.gatedRmsNorm` + `Ops.
    // dequantGemvInt4`) with a single kernel. Saves a full `[in_dim]`
    // DRAM roundtrip on the intermediate `normed` per call.
    //
    // Use at every site where a pre-norm immediately precedes a
    // single int4 qmm projection — most notably the finalNorm + lmHead
    // boundary (one dispatch per token) and the GDN mixer's
    // post-recurrence gatedRmsNorm + outProj pair.

    /// Fused RMSNorm + int4 dequant-GEMV in ONE dispatch via
    /// `iron_rms_norm_qgemv_fast`. Computes
    ///   `y[row] = Σ_i (q[row,i]·scale + bias) ·
    ///            (x[i] · norm_weight[i] · inv_rms)`
    /// with `inv_rms = rsqrt(mean(x²) + eps)` and the normalised
    /// activation never leaving registers.
    ///
    /// Kernel constraints (`iron_rms_norm_qgemv_fast`):
    /// - `in_dim` MUST be a multiple of 512 (kernel block size = 512
    ///   K-elements per outer iter).
    /// - `out_dim` MUST be a multiple of 8 (kernel processes 8 output
    ///   rows per TG).
    /// - `group_size` MUST equal 64.
    /// - TPG = 64 (2 simdgroups × 32 lanes).
    public static func rmsNormQgemvInt4Fast(
        x: Tensor, normWeight: Tensor, eps: Float,
        qWeight: Tensor, qScales: Tensor, qBiases: Tensor,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            qWeight.dtype == .u32,
            "Ops.rmsNormQgemvInt4Fast: qWeight must be u32-packed")
        precondition(
            qWeight.shape.count == 2,
            "Ops.rmsNormQgemvInt4Fast: qWeight must be [outDim, inDim/8]")
        let outDim = qWeight.shape[0]
        let packedPerRow = qWeight.shape[1]
        let inDim = packedPerRow * 8
        precondition(
            x.elementCount == inDim,
            "Ops.rmsNormQgemvInt4Fast: x.elementCount \(x.elementCount) ≠ inDim \(inDim)")
        precondition(
            normWeight.elementCount == inDim,
            "Ops.rmsNormQgemvInt4Fast: normWeight.elementCount \(normWeight.elementCount) ≠ inDim \(inDim)")
        precondition(
            out.elementCount == outDim,
            "Ops.rmsNormQgemvInt4Fast: out.elementCount \(out.elementCount) ≠ outDim \(outDim)")
        precondition(
            x.dtype == normWeight.dtype && normWeight.dtype == qScales.dtype
                && qScales.dtype == qBiases.dtype && qBiases.dtype == out.dtype,
            "Ops.rmsNormQgemvInt4Fast: all non-weight tensors must share dtype")
        precondition(
            inDim % 512 == 0,
            "Ops.rmsNormQgemvInt4Fast: in_dim \(inDim) must be a multiple of 512 (fast variant)")
        precondition(
            outDim % 8 == 0,
            "Ops.rmsNormQgemvInt4Fast: out_dim \(outDim) must be a multiple of 8")
        // INVARIANT: kernel pins group_size=64 + TPG=64 for the fast
        // 8-rows-per-TG variant. group_size is checked as a constexpr
        // by the kernel; we re-assert here for early failure.
        let groupSize = 64
        precondition(
            inDim % groupSize == 0,
            "Ops.rmsNormQgemvInt4Fast: in_dim must divide group_size=64")
        // eps as a 1-element f32 buffer.
        let epsBuf = device.scalarBuffer(eps)
        let tg = MTLSize(width: 64, height: 1, depth: 1)
        let nTiles = outDim / 8
        let grid = MTLSize(width: nTiles * 64, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            IronKernels.iron_rms_norm_qgemv_fast_f32(
                x: x.buffer, xOffset: x.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                weight: qWeight.buffer, weightOffset: qWeight.offset,
                scales: qScales.buffer, scalesOffset: qScales.offset,
                biases: qBiases.buffer, biasesOffset: qBiases.offset,
                output: out.buffer, outputOffset: out.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_rms_norm_qgemv_fast_f16(
                x: x.buffer, xOffset: x.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                weight: qWeight.buffer, weightOffset: qWeight.offset,
                scales: qScales.buffer, scalesOffset: qScales.offset,
                biases: qBiases.buffer, biasesOffset: qBiases.offset,
                output: out.buffer, outputOffset: out.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_rms_norm_qgemv_fast_bf16(
                x: x.buffer, xOffset: x.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                weight: qWeight.buffer, weightOffset: qWeight.offset,
                scales: qScales.buffer, scalesOffset: qScales.offset,
                biases: qBiases.buffer, biasesOffset: qBiases.offset,
                output: out.buffer, outputOffset: out.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rmsNormQgemvInt4Fast: unsupported dtype \(x.dtype)")
        }
    }

    /// Fused Mamba2-style gated RMSNorm + int4 dequant-GEMV in ONE
    /// dispatch via `iron_gated_rms_norm_qgemv_int4_fast`. Computes
    /// the GDN mixer's `silu(z) · w · rmsnorm(y)` and feeds the result
    /// straight into the int4 outProj — replaces the 2-dispatch
    /// chain (`Ops.gatedRmsNorm` + `Ops.dequantGemvInt4`) with a
    /// single kernel. Saves a full `[Hv·Dv]` DRAM roundtrip on the
    /// intermediate gated-norm output per call.
    ///
    /// Use at the GDN mixer post-recurrence boundary, where the
    /// gated-norm output is the direct input to the outProj projection.
    ///
    /// Kernel constraints:
    /// - `Hv·Dv` (in_dim) MUST be a multiple of 512.
    /// - `Hv·Dv` MUST be ≤ 8192 (kernel TG-memory cap; the gated-norm
    ///   pass stages the full input in TG memory).
    /// - `out_dim` MUST be a multiple of 8.
    /// - `group_size` MUST be 64. TPG = 64.
    public static func gatedRmsNormQgemvInt4Fast(
        y: Tensor,  // [Hv, Dv] f32 (GDN recurrence output)
        z: Tensor,  // [Hv*Dv] T
        normWeight: Tensor,  // [Dv] T
        eps: Float,
        qWeight: Tensor,  // [out_dim, in_dim/8] u32
        qScales: Tensor,  // [out_dim, in_dim/group_size] T
        qBiases: Tensor,
        hv: Int, dv: Int, outDim: Int, groupSize: Int = 64,
        on cmd: MTLCommandBuffer,
        into out: Tensor  // [out_dim] T
    ) {
        precondition(
            y.dtype == .f32,
            "Ops.gatedRmsNormQgemvInt4Fast: y must be f32")
        precondition(
            z.dtype == normWeight.dtype && normWeight.dtype == out.dtype,
            "Ops.gatedRmsNormQgemvInt4Fast: z/weight/out dtype mismatch")
        precondition(
            qWeight.dtype == .u32,
            "Ops.gatedRmsNormQgemvInt4Fast: q_weight must be u32-packed")
        let inDim = hv * dv
        precondition(
            inDim % 512 == 0,
            "Ops.gatedRmsNormQgemvInt4Fast: Hv·Dv (\(inDim)) must be multiple of 512")
        precondition(
            inDim <= 8192,
            "Ops.gatedRmsNormQgemvInt4Fast: Hv·Dv (\(inDim)) must be ≤ 8192 (kernel TG-mem cap)")
        precondition(
            outDim % 8 == 0,
            "Ops.gatedRmsNormQgemvInt4Fast: out_dim must be multiple of 8")
        precondition(
            groupSize == 64,
            "Ops.gatedRmsNormQgemvInt4Fast: group_size must be 64")
        precondition(
            out.elementCount == outDim,
            "Ops.gatedRmsNormQgemvInt4Fast: out element count mismatch")
        let tpg = 64
        let nTiles = outDim / 8
        let grid = MTLSize(width: nTiles * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        // eps as a 1-element f32 buffer.
        let epsBuf = device.scalarBuffer(eps)
        switch out.dtype {
        case .f32:
            IronKernels.iron_gated_rms_norm_qgemv_int4_fast_f32(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                q_weight: qWeight.buffer, q_weightOffset: qWeight.offset,
                q_scales: qScales.buffer, q_scalesOffset: qScales.offset,
                q_biases: qBiases.buffer, q_biasesOffset: qBiases.offset,
                out: out.buffer, outOffset: out.offset,
                hv: UInt32(hv), dv: UInt32(dv),
                out_dim: UInt32(outDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_gated_rms_norm_qgemv_int4_fast_f16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                q_weight: qWeight.buffer, q_weightOffset: qWeight.offset,
                q_scales: qScales.buffer, q_scalesOffset: qScales.offset,
                q_biases: qBiases.buffer, q_biasesOffset: qBiases.offset,
                out: out.buffer, outOffset: out.offset,
                hv: UInt32(hv), dv: UInt32(dv),
                out_dim: UInt32(outDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_gated_rms_norm_qgemv_int4_fast_bf16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                q_weight: qWeight.buffer, q_weightOffset: qWeight.offset,
                q_scales: qScales.buffer, q_scalesOffset: qScales.offset,
                q_biases: qBiases.buffer, q_biasesOffset: qBiases.offset,
                out: out.buffer, outOffset: out.offset,
                hv: UInt32(hv), dv: UInt32(dv),
                out_dim: UInt32(outDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedRmsNormQgemvInt4Fast: unsupported dtype \(out.dtype)")
        }
    }

    // ─── Fused scalar-sigmoid fan-out + FMA ───────────────────────────
    //
    // Wraps `mt_sigmoid_scalar_fma_{f32,f16,bf16}`. Computes
    //   `out[i] = base[i] + sigmoid(gate[0]) * value[i]`
    // for `i in 0..hidden`, broadcasting the scalar `gate` across the
    // `[hidden]` vectors. Used by `Qwen35MoEFFN.forward` to fuse the
    // shared-expert gate's sigmoid + broadcast-mul + residual add into
    // one GPU dispatch, eliminating the `gateLogit.toFloatArray()` host
    // detour and the `commit + wait` that comes with it.
    //
    // Tensor contracts (matching the kernel sig):
    //   gate    [1]       T — scalar logit (raw, not yet sigmoided)
    //   value   [hidden]  T — shared-expert output
    //   base    [hidden]  T — running combine (routed-expert sum)
    //   out     [hidden]  T — gated + accumulated result
    // Dispatch:
    //   grid = [hidden, 1, 1] threads (one per element)
    //   tg   = [tgWidth, 1, 1] threads
    public static func sigmoidScalarFMA(
        gate: Tensor, value: Tensor, base: Tensor,
        into out: Tensor, on cmd: MTLCommandBuffer
    ) {
        precondition(
            gate.dtype == value.dtype && value.dtype == base.dtype && base.dtype == out.dtype,
            "Ops.sigmoidScalarFMA: all tensors must share dtype")
        precondition(
            gate.elementCount == 1,
            "Ops.sigmoidScalarFMA: gate must be [1] (got \(gate.elementCount))")
        precondition(
            value.elementCount == base.elementCount && base.elementCount == out.elementCount,
            "Ops.sigmoidScalarFMA: value / base / out must have matching elementCount")

        let n = value.elementCount
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch out.dtype {
        case .f32:
            IronKernels.iron_sigmoid_scalar_fma_f32(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_sigmoid_scalar_fma_f16(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_sigmoid_scalar_fma_bf16(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sigmoidScalarFMA: unsupported dtype \(out.dtype)")
        }
    }

    /// Row-broadcast sigmoid FMA over a `[T, hidden]` layout:
    /// `out[r, h] = base[r, h] + sigmoid(gate[r]) * value[r, h]`.
    /// The T-batched form of `sigmoidScalarFMA` — one dispatch replaces
    /// the per-row command-buffer fan-out at batched prefill.
    public static func sigmoidRowFMA(
        gate: Tensor, value: Tensor, base: Tensor,
        into out: Tensor, t: Int, hidden: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(
            gate.dtype == value.dtype && value.dtype == base.dtype && base.dtype == out.dtype,
            "Ops.sigmoidRowFMA: all tensors must share dtype")
        precondition(
            gate.elementCount == t,
            "Ops.sigmoidRowFMA: gate must be [T] (got \(gate.elementCount), T=\(t))")
        precondition(
            value.elementCount == t * hidden && base.elementCount == t * hidden
                && out.elementCount == t * hidden,
            "Ops.sigmoidRowFMA: value / base / out must be T*hidden")

        let n = t * hidden
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let hiddenU = UInt32(hidden)
        switch out.dtype {
        case .f32:
            IronKernels.iron_sigmoid_row_fma_f32(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_sigmoid_row_fma_f16(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_sigmoid_row_fma_bf16(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sigmoidRowFMA: unsupported dtype \(out.dtype)")
        }
    }

    // ─── Fused SwiGLU ─────────────────────────────────────────────────
    //
    // Wraps `iron_swiglu_{f32,f16,bf16}`. Computes `out[i] = silu(gate[i]) * up[i]`
    // in one element-wise dispatch — replaces the two-launch `silu` +
    // `mul` chain (`Ops.silu(gate) → Ops.mul(_, up)`). Half the bandwidth
    // on the activation tensor (the intermediate `silu(gate)` stays in
    // registers) plus one fewer commit/encode round-trip per call.
    //
    // Used by Butter's per-expert MoE SwiGLU + Qwen3 dense MLPs. fp32 path
    // is the canonical reference; f16 / bf16 narrow on store, accumulate
    // in fp32.
    public static func swiglu(
        gate: Tensor, up: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            gate.dtype == up.dtype,
            "Ops.swiglu: gate / up dtype mismatch")
        precondition(
            gate.elementCount == up.elementCount,
            "Ops.swiglu: gate / up size mismatch")
        let result = out ?? Tensor.empty(shape: gate.shape, dtype: gate.dtype)
        precondition(
            result.dtype == gate.dtype && result.elementCount == gate.elementCount,
            "Ops.swiglu: out dtype / size mismatch")
        let n = gate.elementCount
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch gate.dtype {
        case .f32:
            IronKernels.iron_swiglu_f32(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_swiglu_f16(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_swiglu_bf16(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.swiglu: unsupported dtype \(gate.dtype)")
        }
        return result
    }

    /// N independent SwiGLU dispatches sharing ONE compute encoder.
    /// Each `(gate, up, out)` triple is dispatched in sequence after
    /// setting the same `iron_swiglu_*` PSO once. Used by the MoE
    /// per-expert SwiGLU phase at decode (one dispatch per chosen
    /// expert × topK experts × MoE layers) where the encoder
    /// begin/end pairs dominated CPU side.
    public static func swigluMany(
        gates: [Tensor], ups: [Tensor], outs: [Tensor],
        on cmd: MTLCommandBuffer
    ) {
        let n = gates.count
        precondition(
            ups.count == n && outs.count == n,
            "Ops.swigluMany: count mismatch")
        guard n > 0 else { return }
        let dtype = gates[0].dtype
        let psoName: String
        switch dtype {
        case .f32: psoName = "iron_swiglu_f32"
        case .f16: psoName = "iron_swiglu_f16"
        case .bf16: psoName = "iron_swiglu_bf16"
        default: fatalError("Ops.swigluMany: unsupported dtype \(dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        for i in 0 ..< n {
            let count = gates[i].elementCount
            precondition(
                ups[i].elementCount == count && outs[i].elementCount == count,
                "Ops.swigluMany: shape mismatch at index \(i)")
            precondition(
                ups[i].dtype == dtype && outs[i].dtype == dtype,
                "Ops.swigluMany: dtype mismatch at index \(i)")
            let tgWidth = min(count, 256)
            enc.setBuffer(gates[i].buffer, offset: gates[i].offset, index: 0)
            enc.setBuffer(ups[i].buffer, offset: ups[i].offset, index: 1)
            enc.setBuffer(outs[i].buffer, offset: outs[i].offset, index: 2)
            enc.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    /// DSv4 SwiGLU-with-limit (single): `silu(min(gate,limit)) * clip(up,±limit)`.
    public static func swigluLimit(
        gate: Tensor, up: Tensor, limit: Float, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(gate.dtype == up.dtype && gate.elementCount == up.elementCount,
            "Ops.swigluLimit: gate/up mismatch")
        let result = out ?? Tensor.empty(shape: gate.shape, dtype: gate.dtype)
        swigluLimitMany(gates: [gate], ups: [up], outs: [result], limit: limit, on: cmd)
        return result
    }

    /// DSv4 SwiGLU-with-limit, N dispatches sharing one encoder:
    /// `out = silu(min(gate, limit)) * clip(up, -limit, +limit)`. DSv4
    /// trains with swiglu_limit=10; the clamp is essential — unclamped
    /// silu(gate)*up overflows fp16 in the deep high-magnitude layers.
    public static func swigluLimitMany(
        gates: [Tensor], ups: [Tensor], outs: [Tensor], limit: Float,
        on cmd: MTLCommandBuffer
    ) {
        let n = gates.count
        precondition(ups.count == n && outs.count == n, "Ops.swigluLimitMany: count mismatch")
        guard n > 0 else { return }
        let dtype = gates[0].dtype
        let psoName: String
        switch dtype {
        case .f32: psoName = "iron_swiglu_limit_f32"
        case .f16: psoName = "iron_swiglu_limit_f16"
        case .bf16: psoName = "iron_swiglu_limit_bf16"
        default: fatalError("Ops.swigluLimitMany: unsupported dtype \(dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        var lim = limit
        for i in 0 ..< n {
            let count = gates[i].elementCount
            precondition(ups[i].elementCount == count && outs[i].elementCount == count,
                "Ops.swigluLimitMany: shape mismatch at index \(i)")
            precondition(ups[i].dtype == dtype && outs[i].dtype == dtype,
                "Ops.swigluLimitMany: dtype mismatch at index \(i)")
            let tgWidth = min(count, 256)
            enc.setBuffer(gates[i].buffer, offset: gates[i].offset, index: 0)
            enc.setBuffer(ups[i].buffer, offset: ups[i].offset, index: 1)
            enc.setBuffer(outs[i].buffer, offset: outs[i].offset, index: 2)
            enc.setBytes(&lim, length: 4, index: 3)
            enc.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    // ─── MoE gather quantised matmul, scalar m1 ────────────────────────
    //
    // Wraps `iron_moe_gather_qmm_int4_{f32,f16,bf16}`. One TG per
    // (output column m, input row t). Each TG resolves the row's
    // expert via the CSR `expert_offsets` and computes one scalar
    // dot-product with `simd_sum` reduction. No cooperative-tensor
    // (MPP) overhead — at decode T=1 the cooperative variants
    // (bm8 / bm16 / bm64) regress because the descriptor + tensor
    // setup dominates the tiny compute. This kernel skips that
    // overhead and walks K with the same per-lane stride pattern as
    // `iron_dequant_gemv_int4`.
    //
    // Inputs:
    //   x              [T_rows, k_in]                 T  — sorted by expert (caller responsibility)
    //   weight         [n_experts, m_out, k_in/8]     u32 (int4 packed)
    //   scales         [n_experts, m_out, k_in/group] T
    //   biases         [n_experts, m_out, k_in/group] T
    //   expertOffsets  [n_experts + 1]                u32 — CSR row offsets
    //   out            [T_rows, m_out]                T
    //
    // Constexpr: k_in, m_out, n_experts, group_size.
    public static func moeGatherDequantGemmInt4M1(
        _ x: Tensor, _ weight: Tensor, _ scales: Tensor, _ biases: Tensor,
        _ expertOffsets: Tensor,
        _ tRows: Int, _ mOut: Int, _ kIn: Int,
        _ nExperts: Int, _ groupSize: Int,
        _ cmd: MTLCommandBuffer, _ out: Tensor
    ) {
        precondition(weight.dtype == .u32, "moeGatherDequantGemmInt4M1: weight must be u32 packed")
        precondition(
            expertOffsets.dtype == .u32, "moeGatherDequantGemmInt4M1: expertOffsets must be u32")
        precondition(
            scales.dtype == x.dtype && biases.dtype == x.dtype && out.dtype == x.dtype,
            "moeGatherDequantGemmInt4M1: dtype mismatch")
        precondition(
            kIn.isMultiple(of: 32), "moeGatherDequantGemmInt4M1: k_in must be multiple of 32")
        precondition(
            kIn.isMultiple(of: groupSize), "moeGatherDequantGemmInt4M1: group_size must divide k_in"
        )
        precondition(
            expertOffsets.elementCount == nExperts + 1,
            "moeGatherDequantGemmInt4M1: expertOffsets must have n_experts+1 entries")
        // Swift binding uses dispatchThreads — total-thread semantics.
        // Kernel wants ONE TG per (output col m, input row t). With TG
        // width 32, grid.x = mOut * 32 gives mOut threadgroups in x; the
        // y axis maps one TG per row.
        let grid = MTLSize(width: mOut * 32, height: tRows, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let kInU = UInt32(kIn)
        let mOutU = UInt32(mOut)
        let nExpertsU = UInt32(nExperts)
        let groupSizeU = UInt32(groupSize)
        switch x.dtype {
        case .f32:
            IronKernels.iron_moe_gather_qmm_int4_f32(
                x: x.buffer, xOffset: x.offset,
                weight_packed: weight.buffer, weight_packedOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                expert_offsets: expertOffsets.buffer, expert_offsetsOffset: expertOffsets.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, n_experts: nExpertsU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_moe_gather_qmm_int4_f16(
                x: x.buffer, xOffset: x.offset,
                weight_packed: weight.buffer, weight_packedOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                expert_offsets: expertOffsets.buffer, expert_offsetsOffset: expertOffsets.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, n_experts: nExpertsU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_moe_gather_qmm_int4_bf16(
                x: x.buffer, xOffset: x.offset,
                weight_packed: weight.buffer, weight_packedOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                expert_offsets: expertOffsets.buffer, expert_offsetsOffset: expertOffsets.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, n_experts: nExpertsU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4M1: unsupported dtype \(x.dtype)")
        }
    }

    /// Compact sorted expert ids into an active-only list + row spans and
    /// write `MTLDispatchThreadgroupsIndirectArguments` for gate/up and down.
    /// Single TG, thread 0. Live-compile `iron_moe_compact_active_experts`.
    public static func moeCompactActiveExperts(
        sortedExperts: Tensor,
        activeExperts: Tensor,
        rowLo: Tensor,
        rowHi: Tensor,
        nActiveOut: Tensor,
        indirectGU: Tensor,
        indirectDown: Tensor,
        mTotal: Int,
        nOutGU: Int,
        nOutDown: Int,
        nExpertsCap: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(sortedExperts.dtype == .u32)
        precondition(activeExperts.dtype == .u32 && rowLo.dtype == .u32 && rowHi.dtype == .u32)
        precondition(nActiveOut.dtype == .u32 && nActiveOut.elementCount >= 1)
        precondition(indirectGU.dtype == .u32 && indirectGU.elementCount >= 3)
        precondition(indirectDown.dtype == .u32 && indirectDown.elementCount >= 3)
        var mTotalU = UInt32(mTotal)
        var nOutGUU = UInt32(nOutGU)
        var nOutDownU = UInt32(nOutDown)
        var capU = UInt32(nExpertsCap)
        let pso = PSOCache.shared.pipelineState(for: "iron_moe_compact_active_experts_f32")
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(sortedExperts.buffer, offset: sortedExperts.offset, index: 0)
        enc.setBuffer(activeExperts.buffer, offset: activeExperts.offset, index: 1)
        enc.setBuffer(rowLo.buffer, offset: rowLo.offset, index: 2)
        enc.setBuffer(rowHi.buffer, offset: rowHi.offset, index: 3)
        enc.setBuffer(nActiveOut.buffer, offset: nActiveOut.offset, index: 4)
        enc.setBuffer(indirectGU.buffer, offset: indirectGU.offset, index: 5)
        enc.setBuffer(indirectDown.buffer, offset: indirectDown.offset, index: 6)
        enc.setBytes(&mTotalU, length: 4, index: 7)
        enc.setBytes(&nOutGUU, length: 4, index: 8)
        enc.setBytes(&nOutDownU, length: 4, index: 9)
        enc.setBytes(&capU, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Densify active expert int2 weights + f16 scales/biases into contiguous
    /// `[nActive, nOut, …]` slabs. One TG per active expert.
    public static func moeDensifyInt2Experts(
        weightSrc: Tensor, scalesSrc: Tensor, biasesSrc: Tensor,
        activeExperts: Tensor,
        weightDst: Tensor, scalesDst: Tensor, biasesDst: Tensor,
        nOut: Int, kIn: Int, groupSize: Int, nActive: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(weightSrc.dtype == .u32 && weightDst.dtype == .u32)
        precondition(scalesSrc.dtype == .f16 && biasesSrc.dtype == .f16)
        precondition(kIn % 16 == 0 && kIn.isMultiple(of: groupSize))
        precondition(nActive > 0)
        let packsPerRow = kIn / 16
        let groupsPerRow = kIn / groupSize
        var nOutU = UInt32(nOut)
        var packsU = UInt32(packsPerRow)
        var groupsU = UInt32(groupsPerRow)
        var nActiveU = UInt32(nActive)
        let pso = PSOCache.shared.pipelineState(for: "iron_moe_densify_int2_experts_f16")
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weightSrc.buffer, offset: weightSrc.offset, index: 0)
        enc.setBuffer(scalesSrc.buffer, offset: scalesSrc.offset, index: 1)
        enc.setBuffer(biasesSrc.buffer, offset: biasesSrc.offset, index: 2)
        enc.setBuffer(activeExperts.buffer, offset: activeExperts.offset, index: 3)
        enc.setBuffer(weightDst.buffer, offset: weightDst.offset, index: 4)
        enc.setBuffer(scalesDst.buffer, offset: scalesDst.offset, index: 5)
        enc.setBuffer(biasesDst.buffer, offset: biasesDst.offset, index: 6)
        enc.setBytes(&nOutU, length: 4, index: 7)
        enc.setBytes(&packsU, length: 4, index: 8)
        enc.setBytes(&groupsU, length: 4, index: 9)
        enc.setBytes(&nActiveU, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: nActive, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Remap sorted expert ids → dense `[0, nActive)` using active list.
    public static func moeRemapExpertsDense(
        sortedExperts: Tensor,
        activeExperts: Tensor,
        denseIndices: Tensor,
        mTotal: Int, nActive: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(sortedExperts.dtype == .u32 && activeExperts.dtype == .u32)
        precondition(denseIndices.dtype == .u32 && denseIndices.elementCount >= mTotal)
        var mTotalU = UInt32(mTotal)
        var nActiveU = UInt32(nActive)
        let pso = PSOCache.shared.pipelineState(for: "iron_moe_remap_experts_dense_f32")
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(sortedExperts.buffer, offset: sortedExperts.offset, index: 0)
        enc.setBuffer(activeExperts.buffer, offset: activeExperts.offset, index: 1)
        enc.setBuffer(denseIndices.buffer, offset: denseIndices.offset, index: 2)
        enc.setBytes(&mTotalU, length: 4, index: 3)
        enc.setBytes(&nActiveU, length: 4, index: 4)
        let tpg = 256
        let grid = MTLSize(width: (mTotal + tpg - 1) / tpg * tpg, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Active-only expert-outer int2 gather QMM (f16). `tgid.y` indexes the
    /// compact active list from `moeCompactActiveExperts`. Prefer
    /// `dispatchIndirect` so empty experts never launch.
    public static func moeGatherDequantGemmInt2Active(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        activeExperts: Tensor, rowLo: Tensor, rowHi: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor,
        encoder: MTLComputeCommandEncoder? = nil,
        indirectArgs: Tensor? = nil,
        nActiveFallback: Int = 0
    ) {
        precondition(weight.dtype == .u32)
        precondition(
            input.dtype == .f16 || input.dtype == .bf16,
            "moeGatherDequantGemmInt2Active: only f16/bf16 (got \(input.dtype))")
        precondition(nOut % 32 == 0 && kIn % 16 == 0)
        precondition(kIn.isMultiple(of: groupSize))
        let psoName: String
        switch input.dtype {
        case .f16: psoName = "iron_moe_gather_qmm_mma_eg_int2_active_mpp_f16"
        case .bf16: psoName = "iron_moe_gather_qmm_mma_eg_int2_active_mpp_bf16"
        default: fatalError("moeGatherDequantGemmInt2Active: \(input.dtype)")
        }
        // bf16 active kernel may be missing — fall back to f16 name only.
        let pso: MTLComputePipelineState
        do {
            pso = try PSOCache.shared.pipelineStateThrowing(for: psoName)
        } catch {
            // No bf16 active kernel yet — callers should use f16 promote.
            fatalError("moeGatherDequantGemmInt2Active: PSO \(psoName): \(error)")
        }
        let ownsEnc = encoder == nil
        guard let enc = encoder ?? cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        var mTotalU = UInt32(mTotal)
        var nOutU = UInt32(nOut)
        var kInU = UInt32(kIn)
        var groupSizeU = UInt32(groupSize)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(biases.buffer, offset: biases.offset, index: 3)
        enc.setBuffer(activeExperts.buffer, offset: activeExperts.offset, index: 4)
        enc.setBuffer(rowLo.buffer, offset: rowLo.offset, index: 5)
        enc.setBuffer(rowHi.buffer, offset: rowHi.offset, index: 6)
        enc.setBuffer(out.buffer, offset: out.offset, index: 7)
        enc.setBytes(&mTotalU, length: 4, index: 8)
        enc.setBytes(&nOutU, length: 4, index: 9)
        enc.setBytes(&kInU, length: 4, index: 10)
        enc.setBytes(&groupSizeU, length: 4, index: 11)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        if let ind = indirectArgs {
            enc.dispatchThreadgroups(
                indirectBuffer: ind.buffer,
                indirectBufferOffset: ind.offset,
                threadsPerThreadgroup: tg)
        } else {
            let nA = max(nActiveFallback, 1)
            enc.dispatchThreadgroups(
                MTLSize(width: nOut / 32, height: nA, depth: 1),
                threadsPerThreadgroup: tg)
        }
        if ownsEnc { enc.endEncoding() }
    }

    /// Affine int2 MoE gather QMM — BM=16/BN=32 (default), BM=BN=64 (opt-in),
    /// or expert-outer grid. Prefer expert-sorted `indices` so contiguous
    /// runs share weight loads.
    ///
    /// Default: MPP `iron_moe_gather_qmm_mma_int2_bm16_mpp_*` (1 SG, tpg=32,
    /// live-compile from Resources/kernels).
    /// - `BUTTER_HY3_MOE_BM64=1` (+ `nOut%64==0`, `kIn%32==0`): BM64 MPP
    ///   (4 SG, tpg=128) — better for fat experts / long runs.
    /// - `BUTTER_HY3_MOE_MMA=1`: simdgroup-MMA BM16 (tpg=64), live-compiled.
    /// - `BUTTER_HY3_EXPERT_GRID=1` + `nExperts`: expert-outer MPP.
    public static func moeGatherDequantGemmInt2(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor,
        encoder: MTLComputeCommandEncoder? = nil,
        nExperts: Int = 0
    ) {
        precondition(weight.dtype == .u32, "moeGatherDequantGemmInt2: weight must be u32 packed")
        precondition(
            indices.dtype == .u32 && indices.elementCount == mTotal,
            "moeGatherDequantGemmInt2: indices must be [mTotal] u32")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype
                && out.dtype == input.dtype,
            "moeGatherDequantGemmInt2: dtype mismatch")
        precondition(
            input.elementCount == mTotal * kIn,
            "moeGatherDequantGemmInt2: input elements \(input.elementCount) ≠ mTotal·kIn \(mTotal * kIn)")
        precondition(
            out.elementCount == mTotal * nOut,
            "moeGatherDequantGemmInt2: out elements \(out.elementCount) ≠ mTotal·nOut \(mTotal * nOut)")
        precondition(
            nOut % 32 == 0 && kIn % 16 == 0,
            "moeGatherDequantGemmInt2: nOut multiple of 32, kIn of 16 (got nOut=\(nOut) kIn=\(kIn))")
        precondition(
            kIn.isMultiple(of: groupSize),
            "moeGatherDequantGemmInt2: group_size must divide k_in")

        // Expert-outer grid when mTotal is large. BUTTER_HY3_NO_EXPERT_GRID=1
        // disables; BUTTER_HY3_EXPERT_GRID=1 forces.
        let useExpertGrid: Bool = {
            guard nExperts > 0 else { return false }
            if ButterEnv["BUTTER_HY3_NO_EXPERT_GRID"] != nil {
                return false
            }
            if ButterEnv["BUTTER_HY3_EXPERT_GRID"] != nil {
                return true
            }
            return mTotal >= 1536
        }()
        let useBm64 =
            !useExpertGrid
            && ButterEnv["BUTTER_HY3_MOE_BM64"] != nil
            && nOut % 64 == 0
            && kIn % 32 == 0
            && mTotal >= 64
        let useBm8 =
            !useExpertGrid && !useBm64
            && ButterEnv["BUTTER_HY3_MOE_BM8"] != nil
        let useMma =
            !useExpertGrid && !useBm64 && !useBm8
            && ButterEnv["BUTTER_HY3_MOE_MMA"] != nil
        // BM64: 4 SG; BM16 MMA: 2 SG; expert-grid/BM16/BM8 MPP: 1 SG.
        let tgWidth = useBm64 ? 128 : (useMma ? 64 : 32)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        var mTotalU = UInt32(mTotal)
        var nOutU = UInt32(nOut)
        var kInU = UInt32(kIn)
        var groupSizeU = UInt32(groupSize)
        var nExpertsU = UInt32(nExperts)
        let psoName: String
        if useExpertGrid {
            switch input.dtype {
            case .f32: psoName = "iron_moe_gather_qmm_mma_eg_int2_expert_grid_mpp_f32"
            case .f16: psoName = "iron_moe_gather_qmm_mma_eg_int2_expert_grid_mpp_f16"
            case .bf16: psoName = "iron_moe_gather_qmm_mma_eg_int2_expert_grid_mpp_bf16"
            default:
                fatalError("Ops.moeGatherDequantGemmInt2: unsupported dtype \(input.dtype)")
            }
        } else if useBm64 {
            switch input.dtype {
            case .f32: psoName = "iron_moe_gather_qmm_mma_int2_bm64_mpp_f32"
            case .f16: psoName = "iron_moe_gather_qmm_mma_int2_bm64_mpp_f16"
            case .bf16: psoName = "iron_moe_gather_qmm_mma_int2_bm64_mpp_bf16"
            default:
                fatalError("Ops.moeGatherDequantGemmInt2: unsupported dtype \(input.dtype)")
            }
        } else if useBm8 {
            switch input.dtype {
            case .f32: psoName = "iron_moe_gather_qmm_mma_int2_bm8_mpp_f32"
            case .f16: psoName = "iron_moe_gather_qmm_mma_int2_bm8_mpp_f16"
            case .bf16: psoName = "iron_moe_gather_qmm_mma_int2_bm8_mpp_bf16"
            default:
                fatalError("Ops.moeGatherDequantGemmInt2: unsupported dtype \(input.dtype)")
            }
        } else if useMma {
            switch input.dtype {
            case .f32: psoName = "iron_moe_gather_qmm_mma_int2_bm16_f32"
            case .f16: psoName = "iron_moe_gather_qmm_mma_int2_bm16_f16"
            case .bf16: psoName = "iron_moe_gather_qmm_mma_int2_bm16_bf16"
            default:
                fatalError("Ops.moeGatherDequantGemmInt2: unsupported dtype \(input.dtype)")
            }
        } else {
            switch input.dtype {
            case .f32: psoName = "iron_moe_gather_qmm_mma_int2_bm16_mpp_f32"
            case .f16: psoName = "iron_moe_gather_qmm_mma_int2_bm16_mpp_f16"
            case .bf16: psoName = "iron_moe_gather_qmm_mma_int2_bm16_mpp_bf16"
            default:
                fatalError("Ops.moeGatherDequantGemmInt2: unsupported dtype \(input.dtype)")
            }
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        let ownsEnc = encoder == nil
        guard let enc = encoder ?? cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(biases.buffer, offset: biases.offset, index: 3)
        enc.setBuffer(indices.buffer, offset: indices.offset, index: 4)
        // The bm16_mpp family carries an x_rows row-map (identity here —
        // X is pre-gathered); its out/constexpr bindings shift by one.
        // The other tiles keep the original ABI.
        let hasXRows = !useExpertGrid && !useBm64 && !useBm8 && !useMma
        if hasXRows {
            let idRows = Self.identityRowsBuffer(minCount: mTotal, device: Device.shared)
            enc.setBuffer(idRows, offset: 0, index: 5)
            enc.setBuffer(out.buffer, offset: out.offset, index: 6)
            enc.setBytes(&mTotalU, length: 4, index: 7)
            enc.setBytes(&nOutU, length: 4, index: 8)
            enc.setBytes(&kInU, length: 4, index: 9)
            enc.setBytes(&groupSizeU, length: 4, index: 10)
        } else {
            enc.setBuffer(out.buffer, offset: out.offset, index: 5)
            enc.setBytes(&mTotalU, length: 4, index: 6)
            enc.setBytes(&nOutU, length: 4, index: 7)
            enc.setBytes(&kInU, length: 4, index: 8)
            enc.setBytes(&groupSizeU, length: 4, index: 9)
            if useExpertGrid {
                enc.setBytes(&nExpertsU, length: 4, index: 10)
            }
        }
        let tgGrid: MTLSize
        if useExpertGrid {
            // One TG per 32 N-cols.
            tgGrid = MTLSize(width: nOut / 32, height: nExperts, depth: 1)
        } else if useBm64 {
            tgGrid = MTLSize(
                width: nOut / 64,
                height: (mTotal + 63) / 64,
                depth: 1)
        } else if useBm8 {
            tgGrid = MTLSize(
                width: nOut / 32,
                height: (mTotal + 7) / 8,
                depth: 1)
        } else {
            tgGrid = MTLSize(
                width: nOut / 32,
                height: (mTotal + 15) / 16,
                depth: 1)
        }
        enc.dispatchThreadgroups(tgGrid, threadsPerThreadgroup: tg)
        if ownsEnc { enc.endEncoding() }
    }

    /// Gather QMM for **int2 affine** (MLX oQ2, scales+biases) — Hy3
    /// reduction-style fallback. `expertIds` is `[tRows]` u32 expert index per
    /// gathered row (token-order or expert-sorted). Kernel is
    /// live-compiled from `Resources/kernels/`.
    public static func moeGatherDequantGemmInt2M1(
        _ x: Tensor, _ weight: Tensor, _ scales: Tensor, _ biases: Tensor,
        _ expertIds: Tensor,
        _ tRows: Int, _ mOut: Int, _ kIn: Int,
        _ nExperts: Int, _ groupSize: Int,
        _ cmd: MTLCommandBuffer, _ out: Tensor
    ) {
        precondition(weight.dtype == .u32, "moeGatherDequantGemmInt2M1: weight must be u32 packed")
        precondition(
            expertIds.dtype == .u32 && expertIds.elementCount == tRows,
            "moeGatherDequantGemmInt2M1: expertIds must be [tRows] u32")
        precondition(
            scales.dtype == x.dtype && biases.dtype == x.dtype && out.dtype == x.dtype,
            "moeGatherDequantGemmInt2M1: dtype mismatch")
        precondition(
            kIn.isMultiple(of: groupSize),
            "moeGatherDequantGemmInt2M1: group_size must divide k_in")
        // One threadgroup per gathered row; threads partition m_out.
        // (Older form was one TG per (row, out_col) → millions of launches.)
        let tgWidth = 256
        let grid = MTLSize(width: tRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        var kInV = UInt32(kIn)
        var mOutV = UInt32(mOut)
        var nExpertsV = UInt32(nExperts)
        var groupSizeV = UInt32(groupSize)
        let psoName: String
        switch x.dtype {
        case .f32: psoName = "iron_moe_gather_qmm_int2_f32"
        case .f16: psoName = "iron_moe_gather_qmm_int2_f16"
        case .bf16: psoName = "iron_moe_gather_qmm_int2_bf16"
        default:
            fatalError("Ops.moeGatherDequantGemmInt2M1: unsupported dtype \(x.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(biases.buffer, offset: biases.offset, index: 3)
        enc.setBuffer(expertIds.buffer, offset: expertIds.offset, index: 4)
        enc.setBuffer(out.buffer, offset: out.offset, index: 5)
        enc.setBytes(&kInV, length: 4, index: 6)
        enc.setBytes(&mOutV, length: 4, index: 7)
        enc.setBytes(&nExpertsV, length: 4, index: 8)
        enc.setBytes(&groupSizeV, length: 4, index: 9)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// LayerNorm: `out = (x − mean) / sqrt(var + eps) · weight + bias`.
    /// Backed by `mt_layer_norm_*` (one TG per row, TPG=1024). Used by
    /// vision-transformer encoders (SigLIP / CLIP) and audio frontends —
    /// not on the Qwen3.5/3.6 hot path, but kept here so VLM/audio
    /// models in this package compile against `Ops`.
    public static func layerNorm(
        _ x: Tensor, weight: Tensor, bias: Tensor,
        eps: Float, nRows: Int, rowSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            x.elementCount == nRows * rowSize,
            "Ops.layerNorm: x size \(x.elementCount) ≠ nRows*rowSize \(nRows * rowSize)")
        precondition(
            weight.elementCount == rowSize,
            "Ops.layerNorm: weight must be [rowSize]")
        precondition(
            bias.elementCount == rowSize,
            "Ops.layerNorm: bias must be [rowSize]")
        precondition(
            x.dtype == weight.dtype && weight.dtype == bias.dtype,
            "Ops.layerNorm: x/weight/bias dtype mismatch")
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let epsBuf = device.scalarBuffer(eps)
        // TPG=1024 per the kernel's reduce-tree contract. One TG per row.
        let tgWidth = 1024
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            IronKernels.iron_layer_norm_f32(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                b: bias.buffer, bOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(rowSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_layer_norm_f16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                b: bias.buffer, bOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(rowSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_layer_norm_bf16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                b: bias.buffer, bOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(rowSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.layerNorm: unsupported dtype \(x.dtype)")
        }
        return result
    }

    // ─── Vision: conv2d ──────────────────────────────────────────────

    /// 2D convolution for vision-transformer patch embedding.
    ///
    /// Backed by `iron-std/src/butter/conv2d.rs` (`conv2d_generic`).
    /// One thread per output element `(n, oc, oh, ow)` — a genuine
    /// Grid3D / one-thread-per-output kernel, dispatched flat over the
    /// `batch * out_ch * out_h * out_w` output count, exactly like
    /// `gather`.
    ///
    /// Layouts (NCHW input, OIHW weight — the PyTorch / safetensors
    /// default every VLM checkpoint ships):
    ///   input  `[batch, in_ch,  in_h,  in_w]`
    ///   weight `[out_ch, in_ch, kh,    kw]`
    ///   bias   `[out_ch]`
    ///   out    `[batch, out_ch, out_h, out_w]`
    ///
    /// ## DISPATCH INVARIANTS (conv2d.rs)
    ///   * `out_h = (in_h + 2*pad_h - kh) / stride_h + 1`
    ///   * `out_w = (in_w + 2*pad_w - kw) / stride_w + 1`
    ///   * `input`, `weight`, `bias`, `out` share one floating dtype.
    public static func conv2d(
        input: Tensor, weight: Tensor, bias: Tensor,
        strideH: Int, strideW: Int,
        padH: Int = 0, padW: Int = 0,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            input.shape.count == 4,
            "Ops.conv2d: input must be 4D [batch,in_ch,in_h,in_w] (conv2d.rs)")
        precondition(
            weight.shape.count == 4,
            "Ops.conv2d: weight must be 4D [out_ch,in_ch,kh,kw] (conv2d.rs)")
        precondition(
            bias.shape.count == 1,
            "Ops.conv2d: bias must be 1D [out_ch] (conv2d.rs)")
        precondition(
            input.dtype == weight.dtype && weight.dtype == bias.dtype,
            "Ops.conv2d: input/weight/bias dtype mismatch (conv2d.rs)")
        precondition(
            strideH > 0 && strideW > 0,
            "Ops.conv2d: stride must be positive (conv2d.rs)")

        let batch = input.shape[0]
        let inCh = input.shape[1]
        let inH = input.shape[2]
        let inW = input.shape[3]
        let outCh = weight.shape[0]
        let kh = weight.shape[2]
        let kw = weight.shape[3]
        precondition(
            weight.shape[1] == inCh,
            "Ops.conv2d: weight in_ch \(weight.shape[1]) != input in_ch \(inCh) (conv2d.rs)")
        precondition(
            bias.shape[0] == outCh,
            "Ops.conv2d: bias \(bias.shape[0]) != out_ch \(outCh) (conv2d.rs)")

        // Output spatial dims — the conv2d.rs DISPATCH INVARIANT.
        let outH = (inH + 2 * padH - kh) / strideH + 1
        let outW = (inW + 2 * padW - kw) / strideW + 1
        precondition(
            outH > 0 && outW > 0,
            "Ops.conv2d: degenerate output \(outH)x\(outW) — kernel "
                + "larger than padded input (conv2d.rs)")

        let result =
            out
            ?? Tensor.empty(
                shape: [batch, outCh, outH, outW],
                dtype: input.dtype)
        precondition(
            result.shape == [batch, outCh, outH, outW],
            "Ops.conv2d: out shape \(result.shape) != expected "
                + "\([batch, outCh, outH, outW]) (conv2d.rs)")

        // Grid3D — one thread per output element, dispatched flat.
        let totalThreads = batch * outCh * outH * outW
        let (grid, tg) = elementwiseGrid(totalThreads)

        func dispatch(
            _ fn: (
                MTLBuffer, Int, MTLBuffer, Int, MTLBuffer, Int, MTLBuffer, Int,
                UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                MTLSize, MTLSize, MTLCommandBuffer
            ) -> Void
        ) {
            fn(
                input.buffer, input.offset, weight.buffer, weight.offset,
                bias.buffer, bias.offset, result.buffer, result.offset,
                UInt32(batch), UInt32(inCh), UInt32(inH), UInt32(inW),
                UInt32(outCh), UInt32(outH), UInt32(outW),
                UInt32(kh), UInt32(kw), UInt32(strideH), UInt32(strideW),
                UInt32(padH), UInt32(padW), grid, tg, cmd)
        }
        switch input.dtype {
        case .f32:
            dispatch {
                IronKernels.conv2d_generic_f32(
                    input: $0, inputOffset: $1, weight: $2, weightOffset: $3,
                    bias: $4, biasOffset: $5, out: $6, outOffset: $7,
                    batch: $8, in_ch: $9, in_h: $10, in_w: $11,
                    out_ch: $12, out_h: $13, out_w: $14,
                    kh: $15, kw: $16, stride_h: $17, stride_w: $18,
                    pad_h: $19, pad_w: $20, gridSize: $21, threadgroupSize: $22,
                    on: $23)
            }
        case .f16:
            dispatch {
                IronKernels.conv2d_generic_f16(
                    input: $0, inputOffset: $1, weight: $2, weightOffset: $3,
                    bias: $4, biasOffset: $5, out: $6, outOffset: $7,
                    batch: $8, in_ch: $9, in_h: $10, in_w: $11,
                    out_ch: $12, out_h: $13, out_w: $14,
                    kh: $15, kw: $16, stride_h: $17, stride_w: $18,
                    pad_h: $19, pad_w: $20, gridSize: $21, threadgroupSize: $22,
                    on: $23)
            }
        case .bf16:
            dispatch {
                IronKernels.conv2d_generic_bf16(
                    input: $0, inputOffset: $1, weight: $2, weightOffset: $3,
                    bias: $4, biasOffset: $5, out: $6, outOffset: $7,
                    batch: $8, in_ch: $9, in_h: $10, in_w: $11,
                    out_ch: $12, out_h: $13, out_w: $14,
                    kh: $15, kw: $16, stride_h: $17, stride_w: $18,
                    pad_h: $19, pad_w: $20, gridSize: $21, threadgroupSize: $22,
                    on: $23)
            }
        default:
            fatalError("Ops.conv2d: unsupported dtype \(input.dtype)")
        }
        return result
    }

    // ─── Vision: patch_embed ─────────────────────────────────────────

    /// Fused image-unfold + linear-projection patch embedding for vision
    /// transformers — the ViT stem in one dispatch.
    ///
    /// Backed by `iron-std/src/butter/patch_embed.rs`. One thread per
    /// output element `(patch, h)` — Grid3D / one-thread-per-output,
    /// dispatched flat over `num_patches * hidden`.
    ///
    /// Layouts (NCHW image, flat linear weight):
    ///   image  `[in_ch, in_h, in_w]`  (single image)
    ///   weight `[hidden, in_ch * patch_h * patch_w]`
    ///   bias   `[hidden]`
    ///   out    `[num_patches, hidden]`
    ///
    /// ## DISPATCH INVARIANTS (patch_embed.rs)
    ///   * `in_h` divisible by `patch_h`, `in_w` by `patch_w` — the
    ///     patch grid tiles the image exactly (no padding / clamp).
    ///   * `weight` second dim == `in_ch * patch_h * patch_w`.
    ///   * `image`, `weight`, `bias`, `out` share one floating dtype.
    public static func patchEmbed(
        image: Tensor, weight: Tensor, bias: Tensor,
        patchH: Int, patchW: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            image.shape.count == 3,
            "Ops.patchEmbed: image must be 3D [in_ch,in_h,in_w] (patch_embed.rs)")
        precondition(
            weight.shape.count == 2,
            "Ops.patchEmbed: weight must be 2D [hidden,patch_dim] (patch_embed.rs)")
        precondition(
            bias.shape.count == 1,
            "Ops.patchEmbed: bias must be 1D [hidden] (patch_embed.rs)")
        precondition(
            image.dtype == weight.dtype && weight.dtype == bias.dtype,
            "Ops.patchEmbed: image/weight/bias dtype mismatch (patch_embed.rs)")

        let inCh = image.shape[0]
        let inH = image.shape[1]
        let inW = image.shape[2]
        let hidden = weight.shape[0]
        precondition(
            inH % patchH == 0 && inW % patchW == 0,
            "Ops.patchEmbed: image \(inH)x\(inW) not divisible by patch "
                + "\(patchH)x\(patchW) (patch_embed.rs)")
        let patchDim = inCh * patchH * patchW
        precondition(
            weight.shape[1] == patchDim,
            "Ops.patchEmbed: weight patch_dim \(weight.shape[1]) != "
                + "in_ch*patch_h*patch_w \(patchDim) (patch_embed.rs)")
        precondition(
            bias.shape[0] == hidden,
            "Ops.patchEmbed: bias \(bias.shape[0]) != hidden \(hidden) (patch_embed.rs)")

        let numPatches = (inH / patchH) * (inW / patchW)
        let result =
            out
            ?? Tensor.empty(
                shape: [numPatches, hidden],
                dtype: image.dtype)
        precondition(
            result.shape == [numPatches, hidden],
            "Ops.patchEmbed: out shape \(result.shape) != expected "
                + "\([numPatches, hidden]) (patch_embed.rs)")

        let totalThreads = numPatches * hidden
        let (grid, tg) = elementwiseGrid(totalThreads)
        switch image.dtype {
        case .f32:
            IronKernels.iron_patch_embed_f32(
                image: image.buffer, imageOffset: image.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                in_ch: UInt32(inCh), in_h: UInt32(inH), in_w: UInt32(inW),
                patch_h: UInt32(patchH), patch_w: UInt32(patchW),
                hidden: UInt32(hidden),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_patch_embed_f16(
                image: image.buffer, imageOffset: image.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                in_ch: UInt32(inCh), in_h: UInt32(inH), in_w: UInt32(inW),
                patch_h: UInt32(patchH), patch_w: UInt32(patchW),
                hidden: UInt32(hidden),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_patch_embed_bf16(
                image: image.buffer, imageOffset: image.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                in_ch: UInt32(inCh), in_h: UInt32(inH), in_w: UInt32(inW),
                patch_h: UInt32(patchH), patch_w: UInt32(patchW),
                hidden: UInt32(hidden),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.patchEmbed: unsupported dtype \(image.dtype)")
        }
        return result
    }

    // ─── Vision: rope_2d ─────────────────────────────────────────────

    /// 2D positional RoPE for vision transformers — the "M-RoPE" spatial
    /// component. Splits each head's `head_dim` into two halves: the
    /// first rotated by the token's row index, the second by its column.
    ///
    /// Backed by `iron-std/src/butter/rope_2d.rs`. Grid3D — one thread
    /// per `(token, head, j)` with `j ∈ [0, quarter_dim)`; each thread
    /// emits four output values.
    ///
    /// Layout:
    ///   qk        `[n_tokens, n_heads, head_dim]`
    ///   positions `[n_tokens, 2]`  u32 — `(row, col)` per token
    ///   out       `[n_tokens, n_heads, head_dim]`
    ///
    /// ## DISPATCH INVARIANTS (rope_2d.rs)
    ///   * `head_dim` divisible by 4 (`quarter_dim = head_dim / 4`).
    ///   * `positions` is u32, `[n_tokens, 2]`.
    ///   * `qk` element count == `n_tokens * n_heads * head_dim`.
    public static func rope2D(
        _ qk: Tensor, positions: Tensor,
        nTokens: Int, nHeads: Int, headDim: Int,
        thetaBase: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            headDim % 4 == 0,
            "Ops.rope2D: head_dim \(headDim) must be a multiple of 4 (rope_2d.rs)")
        precondition(
            positions.dtype == .u32,
            "Ops.rope2D: positions must be u32 (rope_2d.rs)")
        precondition(
            positions.elementCount == nTokens * 2,
            "Ops.rope2D: positions count \(positions.elementCount) != "
                + "n_tokens*2 \(nTokens * 2) (rope_2d.rs)")
        precondition(
            qk.elementCount == nTokens * nHeads * headDim,
            "Ops.rope2D: qk count \(qk.elementCount) != "
                + "n_tokens*n_heads*head_dim \(nTokens * nHeads * headDim) (rope_2d.rs)")

        let halfDim = headDim / 2
        let quarterDim = headDim / 4
        let result = out ?? Tensor.empty(shape: qk.shape, dtype: qk.dtype)

        // Grid3D: one thread per (token, head, j).
        let grid = MTLSize(width: nTokens, height: nHeads, depth: quarterDim)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch qk.dtype {
        case .f32:
            IronKernels.iron_rope_2d_f32(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: positions.buffer, positionsOffset: positions.offset,
                out: result.buffer, outOffset: result.offset,
                n_heads: UInt32(nHeads), head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim), quarter_dim: UInt32(quarterDim),
                theta_base: thetaBase,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_rope_2d_f16(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: positions.buffer, positionsOffset: positions.offset,
                out: result.buffer, outOffset: result.offset,
                n_heads: UInt32(nHeads), head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim), quarter_dim: UInt32(quarterDim),
                theta_base: thetaBase,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_rope_2d_bf16(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: positions.buffer, positionsOffset: positions.offset,
                out: result.buffer, outOffset: result.offset,
                n_heads: UInt32(nHeads), head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim), quarter_dim: UInt32(quarterDim),
                theta_base: thetaBase,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rope2D: unsupported dtype \(qk.dtype)")
        }
        return result
    }

    // ─── Audio front-end / vocoder ───────────────────────────────────

    /// Log-Mel spectrogram — the STT / audio-in front-end. Fuses the
    /// short-time Fourier transform, the Mel filterbank projection and
    /// the log into one dispatch. Wraps `mel_spectrogram_{f32,f16}`.
    ///
    /// Inputs (all share `dtype`, f32 or f16):
    ///   * `audio`      — `[nSamples]` mono waveform, pre-padded so every
    ///                    frame is in-bounds (Whisper reflect-pads by
    ///                    `nFFT/2` on each side before calling this).
    ///   * `window`     — `[nFFT]` analysis window (periodic Hann).
    ///   * `melWeight`  — `[nMels, nFreq]` Mel filterbank, row-major.
    ///
    /// Output: `[nFrames, nMels]` log-Mel.
    ///
    /// The kernel is a plain Grid3D one-thread-per-output kernel
    /// (`KernelMode::Grid3D` in `mel_spectrogram.rs`), so `elementwiseGrid`
    /// is the correct dispatch — NOT a reduction kernel.
    ///
    /// ## DISPATCH INVARIANTS (from `butter/mel_spectrogram.rs`)
    ///   * `nFreq == nFFT / 2 + 1` — the non-redundant real-FFT bins.
    ///   * `nSamples >= (nFrames - 1) * hopLength + nFFT` — the kernel
    ///     does no bounds check on the frame walk; the caller pre-pads.
    public static func melSpectrogram(
        audio: Tensor, window: Tensor, melWeight: Tensor,
        nFFT: Int, nMels: Int, hopLength: Int,
        nFrames: Int, logEps: Float = 1e-10,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            audio.dtype == window.dtype && audio.dtype == melWeight.dtype,
            "Ops.melSpectrogram: audio/window/melWeight must share dtype")
        precondition(
            audio.dtype == .f32 || audio.dtype == .f16,
            "Ops.melSpectrogram: dtype must be f32 or f16")
        let nFreq = nFFT / 2 + 1
        // Invariants cited from butter/mel_spectrogram.rs §"Layouts".
        precondition(
            window.elementCount == nFFT,
            "Ops.melSpectrogram: window must be [nFFT=\(nFFT)] "
                + "(butter/mel_spectrogram.rs)")
        precondition(
            melWeight.elementCount == nMels * nFreq,
            "Ops.melSpectrogram: melWeight must be [nMels, nFreq] "
                + "= [\(nMels), \(nFreq)] (butter/mel_spectrogram.rs)")
        precondition(
            audio.elementCount >= (nFrames - 1) * hopLength + nFFT,
            "Ops.melSpectrogram: audio too short — kernel does no "
                + "bounds check on the frame walk; pre-pad so "
                + "nSamples >= (nFrames-1)*hop + nFFT "
                + "(butter/mel_spectrogram.rs)")
        let result = out ?? Tensor.empty(shape: [nFrames, nMels], dtype: audio.dtype)
        // One thread per output element (frame, mel_bin).
        let (grid, tg) = elementwiseGrid(nFrames * nMels)
        switch audio.dtype {
        case .f32:
            IronKernels.iron_mel_spectrogram_f32(
                audio: audio.buffer, audioOffset: audio.offset,
                window: window.buffer, windowOffset: window.offset,
                mel_weight: melWeight.buffer, mel_weightOffset: melWeight.offset,
                out: result.buffer, outOffset: result.offset,
                n_fft: UInt32(nFFT), n_freq: UInt32(nFreq),
                n_mels: UInt32(nMels), hop_length: UInt32(hopLength),
                log_eps: logEps,
                n_out: UInt32(nFrames * nMels),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_mel_spectrogram_f16(
                audio: audio.buffer, audioOffset: audio.offset,
                window: window.buffer, windowOffset: window.offset,
                mel_weight: melWeight.buffer, mel_weightOffset: melWeight.offset,
                out: result.buffer, outOffset: result.offset,
                n_fft: UInt32(nFFT), n_freq: UInt32(nFreq),
                n_mels: UInt32(nMels), hop_length: UInt32(hopLength),
                log_eps: logEps,
                n_out: UInt32(nFrames * nMels),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.melSpectrogram: unsupported dtype \(audio.dtype)")
        }
        return result
    }

    /// Wide-stride multi-channel 1D convolution — the STT audio patch
    /// embedding. Dense, strided, NCL layout (PyTorch `nn.Conv1d`).
    /// Wraps `audio_conv1d_{f32,f16,bf16}`.
    ///
    /// Inputs (all share `dtype`):
    ///   * `input`  — `[batch, inCh, inLen]`.
    ///   * `weight` — `[outCh, inCh, k]`.
    ///   * `bias`   — `[outCh]`.
    ///
    /// Output: `[batch, outCh, outLen]` with
    /// `outLen = (inLen + 2*pad - k) / stride + 1`.
    ///
    /// Grid3D one-thread-per-output kernel (`KernelMode::Grid3D` in
    /// `audio_conv1d.rs`); `elementwiseGrid` is the correct dispatch.
    ///
    /// ## DISPATCH INVARIANTS (from `butter/audio_conv1d.rs`)
    ///   * `outLen == (inLen + 2*pad - k) / stride + 1`.
    ///   * `stride >= 1`, `k >= 1`.
    public static func audioConv1d(
        input: Tensor, weight: Tensor, bias: Tensor,
        batch: Int, inCh: Int, inLen: Int, outCh: Int,
        k: Int, stride: Int, pad: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            input.dtype == weight.dtype && input.dtype == bias.dtype,
            "Ops.audioConv1d: input/weight/bias must share dtype")
        precondition(
            stride >= 1 && k >= 1,
            "Ops.audioConv1d: stride and k must be >= 1 "
                + "(butter/audio_conv1d.rs)")
        precondition(
            input.elementCount == batch * inCh * inLen,
            "Ops.audioConv1d: input must be [batch, inCh, inLen]")
        precondition(
            weight.elementCount == outCh * inCh * k,
            "Ops.audioConv1d: weight must be [outCh, inCh, k]")
        precondition(
            bias.elementCount == outCh,
            "Ops.audioConv1d: bias must be [outCh]")
        let outLen = (inLen + 2 * pad - k) / stride + 1
        precondition(
            outLen >= 1,
            "Ops.audioConv1d: degenerate outLen=\(outLen) "
                + "(butter/audio_conv1d.rs)")
        let result =
            out
            ?? Tensor.empty(
                shape: [batch, outCh, outLen],
                dtype: input.dtype)
        // One thread per output element (n, oc, op).
        let (grid, tg) = elementwiseGrid(batch * outCh * outLen)
        switch input.dtype {
        case .f32:
            IronKernels.iron_conv1d_dense_direct_f32(
                input: input.buffer, inputOffset: input.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                batch: UInt32(batch), in_ch: UInt32(inCh), channels: UInt32(inCh), in_len: UInt32(inLen),
                out_ch: UInt32(outCh), out_len: UInt32(outLen),
                k: UInt32(k), stride: UInt32(stride), pad: UInt32(pad),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_conv1d_dense_direct_f16(
                input: input.buffer, inputOffset: input.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                batch: UInt32(batch), in_ch: UInt32(inCh), channels: UInt32(inCh), in_len: UInt32(inLen),
                out_ch: UInt32(outCh), out_len: UInt32(outLen),
                k: UInt32(k), stride: UInt32(stride), pad: UInt32(pad),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_conv1d_dense_direct_bf16(
                input: input.buffer, inputOffset: input.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                batch: UInt32(batch), in_ch: UInt32(inCh), channels: UInt32(inCh), in_len: UInt32(inLen),
                out_ch: UInt32(outCh), out_len: UInt32(outLen),
                k: UInt32(k), stride: UInt32(stride), pad: UInt32(pad),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.audioConv1d: unsupported dtype \(input.dtype)")
        }
        return result
    }

    /// Inverse-STFT overlap-add — the TTS vocoder waveform-synthesis
    /// tail. Inverse-DFTs each frame, applies the synthesis window,
    /// overlap-adds with COLA normalisation. Wraps
    /// `vocoder_istft_{f32,f16,bf16}`.
    ///
    /// Inputs (all share `dtype`):
    ///   * `specRe` / `specIm` — `[nFrames, nFreq]` real / imaginary
    ///     planes of the predicted STFT.
    ///   * `window` — `[nFFT]` synthesis window.
    ///
    /// Output: `[outLen]` reconstructed waveform with
    /// `outLen = (nFrames - 1) * hopLength + nFFT`.
    ///
    /// Grid3D one-thread-per-output-sample kernel (`KernelMode::Grid3D`
    /// in `vocoder.rs`); `elementwiseGrid` is the correct dispatch.
    ///
    /// ## DISPATCH INVARIANTS (from `butter/vocoder.rs`)
    ///   * `nFreq == nFFT / 2 + 1`.
    ///   * `outLen == (nFrames - 1) * hopLength + nFFT`.
    public static func vocoderISTFT(
        specRe: Tensor, specIm: Tensor, window: Tensor,
        nFrames: Int, nFFT: Int, hopLength: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            specRe.dtype == specIm.dtype && specRe.dtype == window.dtype,
            "Ops.vocoderISTFT: specRe/specIm/window must share dtype")
        let nFreq = nFFT / 2 + 1
        precondition(
            window.elementCount == nFFT,
            "Ops.vocoderISTFT: window must be [nFFT=\(nFFT)] "
                + "(butter/vocoder.rs)")
        precondition(
            specRe.elementCount == nFrames * nFreq,
            "Ops.vocoderISTFT: specRe must be [nFrames, nFreq] "
                + "= [\(nFrames), \(nFreq)] (butter/vocoder.rs)")
        precondition(
            specIm.elementCount == nFrames * nFreq,
            "Ops.vocoderISTFT: specIm must be [nFrames, nFreq] "
                + "= [\(nFrames), \(nFreq)] (butter/vocoder.rs)")
        let outLen = (nFrames - 1) * hopLength + nFFT
        let result = out ?? Tensor.empty(shape: [outLen], dtype: specRe.dtype)
        // One thread per output sample.
        let (grid, tg) = elementwiseGrid(outLen)
        switch specRe.dtype {
        case .f32:
            IronKernels.iron_vocoder_istft_f32(
                spec_re: specRe.buffer, spec_reOffset: specRe.offset,
                spec_im: specIm.buffer, spec_imOffset: specIm.offset,
                window: window.buffer, windowOffset: window.offset,
                out: result.buffer, outOffset: result.offset,
                n_frames: UInt32(nFrames), n_fft: UInt32(nFFT),
                n_freq: UInt32(nFreq), hop_length: UInt32(hopLength),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_vocoder_istft_f16(
                spec_re: specRe.buffer, spec_reOffset: specRe.offset,
                spec_im: specIm.buffer, spec_imOffset: specIm.offset,
                window: window.buffer, windowOffset: window.offset,
                out: result.buffer, outOffset: result.offset,
                n_frames: UInt32(nFrames), n_fft: UInt32(nFFT),
                n_freq: UInt32(nFreq), hop_length: UInt32(hopLength),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_vocoder_istft_bf16(
                spec_re: specRe.buffer, spec_reOffset: specRe.offset,
                spec_im: specIm.buffer, spec_imOffset: specIm.offset,
                window: window.buffer, windowOffset: window.offset,
                out: result.buffer, outOffset: result.offset,
                n_frames: UInt32(nFrames), n_fft: UInt32(nFFT),
                n_freq: UInt32(nFreq), hop_length: UInt32(hopLength),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.vocoderISTFT: unsupported dtype \(specRe.dtype)")
        }
        return result
    }
}

/// Process-wide scratch cache for `Ops.auraFlashSdpa`. The flash
/// wrapper allocates a same-shape Q scratch (for the scale multiply)
/// and a same-shape scale buffer (filled with the softmax scale value)
/// on every call. Per-decode-token across ~28 attention layers, those
/// Metal buffer allocations cost ~3 ms — enough to make the
/// compressed path lose to dequant-mirror at short KV. This cache
/// allocates each scratch ONCE per (shape, dtype, scale) tuple and
/// reuses it across all subsequent calls. The decode pipeline is
/// single-inference-per-instance (see `Qwen35AttentionMixer` threading
/// contract) so an `NSLock` around the dictionary is sufficient.
///
/// Post-AURA-dtype-unification, the Q scratch + scale buffer follow
/// the activation dtype directly (was f32-only pre-unification, paired
/// with the legacy `Tensor<f32>` kernel sig).
final class AuraFlashScratchCache: @unchecked Sendable {
    static let shared = AuraFlashScratchCache()

    private let lock = NSLock()
    private var qScratchByKey: [ScratchKey: Tensor] = [:]
    private var scaleBufByKey: [ScaleKey: Tensor] = [:]
    private var partialsByKey: [PartialsKey: PartialsScratch] = [:]

    /// Optional override for the FA-2 block size used by
    /// `Ops.auraFlashSdpa2Pass` via `Qwen3Layer.forward`. Set to a value
    /// in {32, 64, 128, 256} (or any multiple of 32) to override the
    /// default 64. Bench-only knob — production should leave nil.
    /// `nonisolated(unsafe)` because the single-inference-per-instance
    /// threading contract (see `AuraFlashScratchCache` header) makes
    /// races a non-issue and the bench surface needs simple writes.
    nonisolated(unsafe) public static var blockSizeOverride: Int?

    private struct ScratchKey: Hashable {
        let count: Int
        let dtype: DType
    }

    private struct ScaleKey: Hashable {
        let count: Int
        let dtype: DType
        let scaleBits: UInt32
    }

    /// Keys the 2-pass partial-O / M / L scratch by the longest decode-
    /// time shape: `(nQHeads, maxBlocks, headDim, dtype)`. Cache hot path
    /// allocates once at first call (sized for `maxSeq/blockSize` blocks)
    /// and reuses; the kernel overwrites the full surface each call.
    private struct PartialsKey: Hashable {
        let nQHeads: Int
        let maxBlocks: Int
        let headDim: Int
        let dtype: DType
    }

    /// Bundled partials triple — `Ops.auraFlashSdpa2Pass` takes the three
    /// buffers separately so the caller can reuse them across schemes.
    struct PartialsScratch {
        let partialO: Tensor
        let partialM: Tensor
        let partialL: Tensor
    }

    static func qScratch(shape: [Int], dtype: DType) -> Tensor {
        shared.qScratch(shape: shape, dtype: dtype)
    }

    static func scaleBuffer(shape: [Int], scale: Float, dtype: DType) -> Tensor {
        shared.scaleBuffer(shape: shape, scale: scale, dtype: dtype)
    }

    /// Cached 2-pass partials sized for the worst-case `(nQHeads,
    /// maxBlocks, headDim, dtype)`. `maxBlocks` should be sized for the
    /// cache's `maxSeq / blockSize` so all decode-step sizes fit.
    static func partials(
        nQHeads: Int, maxBlocks: Int, headDim: Int, dtype: DType
    ) -> PartialsScratch {
        shared.partials(
            nQHeads: nQHeads, maxBlocks: maxBlocks, headDim: headDim, dtype: dtype)
    }

    private func partials(
        nQHeads: Int, maxBlocks: Int, headDim: Int, dtype: DType
    ) -> PartialsScratch {
        let key = PartialsKey(
            nQHeads: nQHeads, maxBlocks: maxBlocks, headDim: headDim, dtype: dtype)
        lock.lock()
        defer { lock.unlock() }
        if let cached = partialsByKey[key] { return cached }
        let o = Tensor.empty(
            shape: [nQHeads, maxBlocks, headDim], dtype: dtype, device: .shared)
        let m = Tensor.empty(
            shape: [nQHeads, maxBlocks], dtype: dtype, device: .shared)
        let l = Tensor.empty(
            shape: [nQHeads, maxBlocks], dtype: dtype, device: .shared)
        let scratch = PartialsScratch(partialO: o, partialM: m, partialL: l)
        partialsByKey[key] = scratch
        return scratch
    }

    private func qScratch(shape: [Int], dtype: DType) -> Tensor {
        let count = shape.reduce(1, *)
        let key = ScratchKey(count: count, dtype: dtype)
        lock.lock()
        defer { lock.unlock() }
        if let cached = qScratchByKey[key], cached.shape == shape {
            return cached
        }
        let fresh = Tensor.empty(shape: shape, dtype: dtype, device: .shared)
        qScratchByKey[key] = fresh
        return fresh
    }

    private func scaleBuffer(shape: [Int], scale: Float, dtype: DType) -> Tensor {
        let count = shape.reduce(1, *)
        let key = ScaleKey(count: count, dtype: dtype, scaleBits: scale.bitPattern)
        lock.lock()
        defer { lock.unlock() }
        if let cached = scaleBufByKey[key], cached.shape == shape {
            return cached
        }
        // Host-side conversion from Float into the target dtype. The
        // scale is a single value broadcast across the buffer; bf16/f16
        // narrow it once at construction.
        let fresh = Tensor.empty(shape: shape, dtype: dtype, device: .shared)
        switch dtype {
        case .f32:
            fresh.copyIn(from: [Float](repeating: scale, count: count))
        case .f16:
            fresh.copyIn(from: [Float16](repeating: Float16(scale), count: count))
        case .bf16:
            let bf16Bits = UInt16(truncatingIfNeeded: scale.bitPattern >> 16)
            fresh.copyIn(from: [UInt16](repeating: bf16Bits, count: count))
        default:
            fatalError("AuraFlashScratchCache: unsupported scale dtype \(dtype)")
        }
        scaleBufByKey[key] = fresh
        return fresh
    }
}
