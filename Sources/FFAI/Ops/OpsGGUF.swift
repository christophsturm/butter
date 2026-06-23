// Copyright 2026 Tom Turney (@TheTom)
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
// GGUF block-dequant Ops — Swift wrappers over the metaltile
// `ffai_gguf_dequant_*` kernel family.
//
// Each Op takes the GPU-resident split the loader produces (packed
// quants + per-block scales + LUT tables) and dispatches the matching
// per-dtype kernel. The input/output dtypes are fixed by the kernel
// family:
//
//   - `Tensor<u8>` / `Tensor<u32>` — packed quant bytes
//   - `Tensor<f32>`                — per-block fp16-converted scales
//   - `Tensor<u8>`                 — iq2xxs grid + ksigns tables
//   - `Tensor<T>`                  — output (T = f32 / f16 / bf16)

import Foundation
import Metal
import MetalTileSwift

extension Ops {
    /// Q8_0 — `out[i] = qs_signed[i] * scales[i/32]`. Block size 32.
    ///
    /// - Parameters:
    ///   - qsSigned: `[n_blocks * 32]` `u8` — int8 quants, sign-reconstructed
    ///     inside the kernel via `select(q >= 128, q-256, q)`.
    ///   - scales: `[n_blocks]` `f32` — host-extracted block super-scales
    ///     (fp16 → f32 at load time).
    ///   - outDtype: target output dtype. Allocates the result tensor.
    public static func ggufDequantQ8_0(
        qsSigned: Tensor, scales: Tensor, nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(qsSigned.dtype == .u8, "ggufDequantQ8_0: qsSigned must be u8")
        precondition(scales.dtype == .f32, "ggufDequantQ8_0: scales must be f32")
        precondition(nValues % 32 == 0, "ggufDequantQ8_0: nValues must be multiple of 32")
        let result = out ?? Tensor.empty(shape: [nValues], dtype: outDtype)
        let (grid, tg) = elementwiseGrid(nValues)
        let n = UInt32(nValues)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_gguf_dequant_q8_0_f32(
                qs_signed: qsSigned.buffer, qs_signedOffset: qsSigned.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gguf_dequant_q8_0_f16(
                qs_signed: qsSigned.buffer, qs_signedOffset: qsSigned.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gguf_dequant_q8_0_bf16(
                qs_signed: qsSigned.buffer, qs_signedOffset: qsSigned.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("ggufDequantQ8_0: unsupported output dtype \(outDtype)")
        }
        return result
    }

    /// Q2_K — `out[i] = d * scale_4bit * q_2bit - dmin * min_4bit`.
    /// Block size 256, two-level scales.
    public static func ggufDequantQ2_K(
        qsPacked: Tensor, scales: Tensor, dF32: Tensor, dminF32: Tensor,
        nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(qsPacked.dtype == .u32, "ggufDequantQ2_K: qsPacked must be u32")
        precondition(scales.dtype == .u8, "ggufDequantQ2_K: scales must be u8")
        precondition(dF32.dtype == .f32, "ggufDequantQ2_K: d_f32 must be f32")
        precondition(dminF32.dtype == .f32, "ggufDequantQ2_K: dmin_f32 must be f32")
        precondition(nValues % 256 == 0, "ggufDequantQ2_K: nValues must be multiple of 256")
        let result = out ?? Tensor.empty(shape: [nValues], dtype: outDtype)
        let (grid, tg) = elementwiseGrid(nValues)
        let n = UInt32(nValues)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_gguf_dequant_q2_k_f32(
                qs_packed: qsPacked.buffer, qs_packedOffset: qsPacked.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                dmin_f32: dminF32.buffer, dmin_f32Offset: dminF32.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gguf_dequant_q2_k_f16(
                qs_packed: qsPacked.buffer, qs_packedOffset: qsPacked.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                dmin_f32: dminF32.buffer, dmin_f32Offset: dminF32.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gguf_dequant_q2_k_bf16(
                qs_packed: qsPacked.buffer, qs_packedOffset: qsPacked.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                dmin_f32: dminF32.buffer, dmin_f32Offset: dminF32.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("ggufDequantQ2_K: unsupported output dtype \(outDtype)")
        }
        return result
    }

    /// Fused 6-expert IQ2_XXS gather GEMV. `qsAll`/`dAll` hold the
    /// slot-major split buffers for `nSlots` routed experts (from
    /// `bundle.stageGatherIQ2XXS`); `x` is the shared activation
    /// `[kIn]`. Writes `out[nSlots * mOut]` = per-expert gemv result,
    /// inline-dequanting the quant bytes — no f16 weight buffer, ONE
    /// dispatch for all experts of the role.
    public static func moeGatherGemvIQ2XXS(
        x: Tensor, qsAll: Tensor, dAll: Tensor, expertIds: Tensor, grid: Tensor, signs: Tensor,
        nSlots: Int, mOut: Int, kIn: Int,
        on cmd: MTLCommandBuffer, into out: Tensor
    ) {
        precondition(qsAll.dtype == .u32, "moeGatherGemvIQ2XXS: qsAll must be u32")
        precondition(dAll.dtype == .f32, "moeGatherGemvIQ2XXS: dAll must be f32")
        precondition(expertIds.dtype == .u32, "moeGatherGemvIQ2XXS: expertIds must be u32")
        precondition(kIn % 256 == 0, "moeGatherGemvIQ2XXS: kIn must be multiple of 256")
        let tgWidth = 32
        let grd = MTLSize(width: mOut * tgWidth, height: nSlots, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let kInU = UInt32(kIn)
        let mOutU = UInt32(mOut)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_gather_gemv_iq2xxs_f16(
                x: x.buffer, xOffset: x.offset,
                qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset,
                expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_gather_gemv_iq2xxs_f32(
                x: x.buffer, xOffset: x.offset,
                qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset,
                expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_gather_gemv_iq2xxs_bf16(
                x: x.buffer, xOffset: x.offset,
                qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset,
                expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default:
            fatalError("moeGatherGemvIQ2XXS: unsupported dtype \(out.dtype)")
        }
    }

    /// Q8_0 inline-dequant gemv: `out[m] = Σ_k dequant(W[m,k])·x[k]`,
    /// reading the resident Q8 split (qs int8 + per-block f32 scale).
    public static func gemvQ8(
        q8: ResidentQ8, x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor
    ) {
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let grd = MTLSize(width: q8.mOut * 32, height: 1, depth: 1)
        let qsT = q8.qs; let dT = q8.d
        let kInU = UInt32(q8.kIn); let mOutU = UInt32(q8.mOut)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_gemv_q8_f16(
                qs: qsT, d_f32: dT, x: x.buffer, xOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_gemv_q8_f32(
                qs: qsT, d_f32: dT, x: x.buffer, xOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gemv_q8_bf16(
                qs: qsT, d_f32: dT, x: x.buffer, xOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, gridSize: grd, threadgroupSize: tg, on: cmd)
        default:
            fatalError("gemvQ8: unsupported dtype \(out.dtype)")
        }
    }

    /// Grouped Q8 gemv: each contiguous block of `rowsPerGroup` output
    /// rows reads its own `kIn`-slice of `x`. Fuses the 8-group O-LoRA
    /// into one dispatch. `x` must hold `(mOut/rowsPerGroup) * kIn` values.
    public static func groupedGemvQ8(
        q8: ResidentQ8, x: Tensor, rowsPerGroup: Int,
        on cmd: MTLCommandBuffer, into out: Tensor
    ) {
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let grd = MTLSize(width: q8.mOut * 32, height: 1, depth: 1)
        let kInU = UInt32(q8.kIn); let mOutU = UInt32(q8.mOut); let rpg = UInt32(rowsPerGroup)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_grouped_gemv_q8_f16(
                qs: q8.qs, d_f32: q8.d, x: x.buffer, xOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_grouped_gemv_q8_f32(
                qs: q8.qs, d_f32: q8.d, x: x.buffer, xOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_grouped_gemv_q8_bf16(
                qs: q8.qs, d_f32: q8.d, x: x.buffer, xOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default:
            fatalError("groupedGemvQ8: unsupported dtype \(out.dtype)")
        }
    }

    /// BATCHED grouped Q8 gemv over `nTokens` rows in ONE dispatch. x is
    /// [nTokens, nGroups*kIn], out [nTokens, mOut]. Replaces the prefill
    /// O-LoRA per-token loop (N dispatches → 1).
    public static func groupedGemvQ8Rows(
        q8: ResidentQ8, x: Tensor, rowsPerGroup: Int, nTokens: Int,
        on cmd: MTLCommandBuffer, into out: Tensor
    ) {
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let grd = MTLSize(width: q8.mOut * 32, height: nTokens, depth: 1)
        let kInU = UInt32(q8.kIn); let mOutU = UInt32(q8.mOut); let rpg = UInt32(rowsPerGroup)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_grouped_gemv_q8_rows_f16(
                qs: q8.qs, d_f32: q8.d, x: x.buffer, xOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_grouped_gemv_q8_rows_f32(
                qs: q8.qs, d_f32: q8.d, x: x.buffer, xOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_grouped_gemv_q8_rows_bf16(
                qs: q8.qs, d_f32: q8.d, x: x.buffer, xOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default:
            fatalError("groupedGemvQ8Rows: unsupported dtype \(out.dtype)")
        }
    }

    /// Q8 gemv over a row sub-range of a resident Q8 weight (for the
    /// grouped O-LoRA: each group is a contiguous row block with its own
    /// input slice). `rowStart`/`nRows` select W rows; byte offsets into
    /// the qs/d buffers are derived from the row-contiguous block layout.
    public static func gemvQ8Rows(
        q8: ResidentQ8, rowStart: Int, nRows: Int, x: Tensor,
        on cmd: MTLCommandBuffer, into out: Tensor
    ) {
        let bpr = q8.kIn / 32                      // blocks per row
        let qsOff = rowStart * bpr * 8 * 4         // u32 → bytes
        let dOff = rowStart * bpr * 4              // f32 → bytes
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let grd = MTLSize(width: nRows * 32, height: 1, depth: 1)
        let kInU = UInt32(q8.kIn); let mOutU = UInt32(nRows)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_gemv_q8_f16(
                qs: q8.qs, qsOffset: qsOff, d_f32: q8.d, d_f32Offset: dOff,
                x: x.buffer, xOffset: x.offset, out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_gemv_q8_f32(
                qs: q8.qs, qsOffset: qsOff, d_f32: q8.d, d_f32Offset: dOff,
                x: x.buffer, xOffset: x.offset, out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gemv_q8_bf16(
                qs: q8.qs, qsOffset: qsOff, d_f32: q8.d, d_f32Offset: dOff,
                x: x.buffer, xOffset: x.offset, out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, gridSize: grd, threadgroupSize: tg, on: cmd)
        default:
            fatalError("gemvQ8Rows: unsupported dtype \(out.dtype)")
        }
    }

    /// DSv4 GPU router top-K: top-K experts by biased score, weights =
    /// unbiased[chosen] renormalized sum-to-1. Keeps routing on the GPU
    /// (no CPU readback). `scoreBiased`/`scoreUnbiased` are f32 [nExperts];
    /// writes `indicesOut` [k] u32 + `weightsOut` [k] f32.
    public static func dsv4RouterTopK(
        scoreBiased: Tensor, scoreUnbiased: Tensor,
        indicesOut: Tensor, weightsOut: Tensor,
        nExperts: Int, k: Int, on cmd: MTLCommandBuffer
    ) {
        let grd = MTLSize(width: 32, height: 1, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        MetalTileKernels.mt_dsv4_router_topk_f32(
            score_biased: scoreBiased.buffer, score_biasedOffset: scoreBiased.offset,
            score_unbiased: scoreUnbiased.buffer, score_unbiasedOffset: scoreUnbiased.offset,
            indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
            weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
            n_experts: UInt32(nExperts), k: UInt32(k),
            gridSize: grd, threadgroupSize: tg, on: cmd)
    }

    /// `out[i] = table[idx[i]]` (u32 gather) — remap routed expert ids
    /// into resident-pool packed slots on the GPU.
    public static func remapU32(
        table: Tensor, idx: Tensor, out: Tensor, n: Int, on cmd: MTLCommandBuffer
    ) {
        let tg = MTLSize(width: min(n, 32), height: 1, depth: 1)
        let grd = MTLSize(width: n, height: 1, depth: 1)
        MetalTileKernels.mt_remap_u32_f32(
            table: table.buffer, tableOffset: table.offset,
            idx: idx.buffer, idxOffset: idx.offset,
            out: out.buffer, outOffset: out.offset,
            n: UInt32(n), gridSize: grd, threadgroupSize: tg, on: cmd)
    }

    /// Fused 6-expert Q2_K gather down-projection + router-weighted sum.
    /// `innersAll` holds the per-slot SwiGLU inner `[nSlots * kIn]`; the
    /// Q2_K split buffers hold all routed experts' down weights. Writes
    /// `out[mOut]` = Σ_slot weights[slot] · (downW_slot · inner_slot) —
    /// the routed MoE output — in ONE dispatch.
    public static func moeGatherDownQ2K(
        innersAll: Tensor, qsAll: Tensor, scalesAll: Tensor,
        dAll: Tensor, dminAll: Tensor, expertIds: Tensor, weights: Tensor,
        nSlots: Int, mOut: Int, kIn: Int,
        on cmd: MTLCommandBuffer, into out: Tensor
    ) {
        precondition(qsAll.dtype == .u32 && scalesAll.dtype == .u8)
        precondition(dAll.dtype == .f32 && dminAll.dtype == .f32 && weights.dtype == .f32)
        precondition(expertIds.dtype == .u32, "moeGatherDownQ2K: expertIds must be u32")
        precondition(kIn % 256 == 0, "moeGatherDownQ2K: kIn must be multiple of 256")
        let tgWidth = 32
        let grd = MTLSize(width: mOut * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let kInU = UInt32(kIn); let mOutU = UInt32(mOut); let nSlotsU = UInt32(nSlots)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_gather_down_q2k_f16(
                inners_all: innersAll.buffer, inners_allOffset: innersAll.offset,
                qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                scales_all: scalesAll.buffer, scales_allOffset: scalesAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset,
                dmin_all: dminAll.buffer, dmin_allOffset: dminAll.offset,
                expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                weights: weights.buffer, weightsOffset: weights.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, n_slots: nSlotsU,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_gather_down_q2k_f32(
                inners_all: innersAll.buffer, inners_allOffset: innersAll.offset,
                qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                scales_all: scalesAll.buffer, scales_allOffset: scalesAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset,
                dmin_all: dminAll.buffer, dmin_allOffset: dminAll.offset,
                expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                weights: weights.buffer, weightsOffset: weights.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, n_slots: nSlotsU,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_gather_down_q2k_bf16(
                inners_all: innersAll.buffer, inners_allOffset: innersAll.offset,
                qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                scales_all: scalesAll.buffer, scales_allOffset: scalesAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset,
                dmin_all: dminAll.buffer, dmin_allOffset: dminAll.offset,
                expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                weights: weights.buffer, weightsOffset: weights.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, n_slots: nSlotsU,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default:
            fatalError("moeGatherDownQ2K: unsupported dtype \(out.dtype)")
        }
    }

    /// IQ2_XXS — codebook lookup against `iq2xxs_grid[256][8]` modulated
    /// by the `ksigns_iq2xs[128]` sign-mask table. Block size 256.
    public static func ggufDequantIQ2_XXS(
        qsU32: Tensor, dF32: Tensor, grid: Tensor, signs: Tensor,
        nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(qsU32.dtype == .u32, "ggufDequantIQ2_XXS: qsU32 must be u32")
        precondition(dF32.dtype == .f32, "ggufDequantIQ2_XXS: d_f32 must be f32")
        precondition(grid.dtype == .u8, "ggufDequantIQ2_XXS: grid must be u8")
        precondition(signs.dtype == .u8, "ggufDequantIQ2_XXS: signs must be u8")
        precondition(
            grid.elementCount == 2048, "ggufDequantIQ2_XXS: grid must be 2048 bytes (256×8)")
        precondition(signs.elementCount == 128, "ggufDequantIQ2_XXS: signs must be 128 bytes")
        precondition(nValues % 256 == 0, "ggufDequantIQ2_XXS: nValues must be multiple of 256")
        let result = out ?? Tensor.empty(shape: [nValues], dtype: outDtype)
        let (gridDim, tg) = elementwiseGrid(nValues)
        let n = UInt32(nValues)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_gguf_dequant_iq2_xxs_f32(
                qs_u32: qsU32.buffer, qs_u32Offset: qsU32.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: gridDim, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gguf_dequant_iq2_xxs_f16(
                qs_u32: qsU32.buffer, qs_u32Offset: qsU32.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: gridDim, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gguf_dequant_iq2_xxs_bf16(
                qs_u32: qsU32.buffer, qs_u32Offset: qsU32.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: gridDim, threadgroupSize: tg, on: cmd)
        default:
            fatalError("ggufDequantIQ2_XXS: unsupported output dtype \(outDtype)")
        }
        return result
    }

    /// IQ2_XXS qs extract — raw bytes → packed qs_u32 staging buffer.
    /// Tiny GPU prologue kernel that replaces the per-block CPU
    /// memcpy(64) loop. 1 thread per output u32 word.
    public static func ggufIQ2_XXS_extractQs(
        rawBytes: Tensor, qsU32: Tensor, nBlocks: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(rawBytes.dtype == .u8)
        precondition(qsU32.dtype == .u32)
        let (gridDim, tg) = elementwiseGrid(nBlocks * 16)
        MetalTileKernels.ffai_gguf_iq2_xxs_extract_qs_u32(
            raw_bytes: rawBytes.buffer, raw_bytesOffset: rawBytes.offset,
            qs_u32: qsU32.buffer, qs_u32Offset: qsU32.offset,
            n_blocks: UInt32(nBlocks),
            gridSize: gridDim, threadgroupSize: tg, on: cmd)
    }

    /// IQ2_XXS dequant — raw-bytes variant. Reads qs from the on-disk
    /// 66-byte block layout directly, skipping the CPU preprocess that
    /// split each block into a separate qs_u32 buffer. `dF32` is still
    /// pre-staged (the DSL has no bit_cast<u32→f32> intrinsic for
    /// in-kernel fp16 → f32, and staging the 32K-element d-vector on
    /// CPU is ~30 ms per token vs ~470 ms the qs memcpy was costing).
    public static func ggufDequantIQ2_XXS_raw(
        rawBytes: Tensor, dF32: Tensor, grid: Tensor, signs: Tensor,
        nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(rawBytes.dtype == .u8, "ggufDequantIQ2_XXS_raw: rawBytes must be u8")
        precondition(dF32.dtype == .f32, "ggufDequantIQ2_XXS_raw: d_f32 must be f32")
        precondition(grid.dtype == .u8, "ggufDequantIQ2_XXS_raw: grid must be u8")
        precondition(signs.dtype == .u8, "ggufDequantIQ2_XXS_raw: signs must be u8")
        precondition(nValues % 256 == 0, "ggufDequantIQ2_XXS_raw: nValues must be multiple of 256")
        let result = out ?? Tensor.empty(shape: [nValues], dtype: outDtype)
        let (gridDim, tg) = elementwiseGrid(nValues)
        let n = UInt32(nValues)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_gguf_dequant_iq2_xxs_raw_f32(
                raw_bytes: rawBytes.buffer, raw_bytesOffset: rawBytes.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: gridDim, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gguf_dequant_iq2_xxs_raw_f16(
                raw_bytes: rawBytes.buffer, raw_bytesOffset: rawBytes.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: gridDim, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gguf_dequant_iq2_xxs_raw_bf16(
                raw_bytes: rawBytes.buffer, raw_bytesOffset: rawBytes.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: gridDim, threadgroupSize: tg, on: cmd)
        default:
            fatalError("ggufDequantIQ2_XXS_raw: unsupported output dtype \(outDtype)")
        }
        return result
    }
}
