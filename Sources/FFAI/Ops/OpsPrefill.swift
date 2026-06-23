// Copyright 2026 Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//
// Prefill matmul Ops — batched (M>1) GGUF-quant matmuls that read each
// weight once and reuse it across all rows in a chunk (the amortization
// that turns DSv4 prefill from per-token gemv into batched throughput).
// Backed by the metaltile kernels ffai_gemm_q8 (dense/attn/lmhead),
// ffai_moe_gather_bgemm_iq2xxs_mpp (gate/up), ffai_moe_gather_bgemm_q2k_mpp
// (down). Bindings use dispatchThreads, so gridSize is in THREADS.

import Foundation
import Metal
import MetalTileSwift

extension Ops {
    /// Q8_0 tiled GEMM: `out[r, :] = dequant(W) · input[r, :]` for `nRows`
    /// rows. `W` is the resident Q8 split `[outDim, inDim]`. 32×32 tile,
    /// 1024 threads/TG. `inDim % 16 == 0` (and `% 32` for the Q8 block).
    public static func gemmQ8(
        qs: Tensor, dF32: Tensor, input: Tensor, out: Tensor,
        inDim: Int, outDim: Int, nRows: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(qs.dtype == .u32 && dF32.dtype == .f32)
        let gx = (outDim + 31) / 32
        let gy = (nRows + 31) / 32
        let grd = MTLSize(width: gx * 1024, height: gy, depth: 1)
        let tg = MTLSize(width: 1024, height: 1, depth: 1)
        let i = UInt32(inDim); let o = UInt32(outDim); let n = UInt32(nRows)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_gemm_q8_f16(
                qs: qs.buffer, qsOffset: qs.offset, d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                input: input.buffer, inputOffset: input.offset, out: out.buffer, outOffset: out.offset,
                in_dim: i, out_dim: o, n_rows: n, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_gemm_q8_f32(
                qs: qs.buffer, qsOffset: qs.offset, d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                input: input.buffer, inputOffset: input.offset, out: out.buffer, outOffset: out.offset,
                in_dim: i, out_dim: o, n_rows: n, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gemm_q8_bf16(
                qs: qs.buffer, qsOffset: qs.offset, d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                input: input.buffer, inputOffset: input.offset, out: out.buffer, outOffset: out.offset,
                in_dim: i, out_dim: o, n_rows: n, gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("gemmQ8: unsupported dtype \(out.dtype)")
        }
    }

    /// Multi-query causal sliding-window SDPA (d512, sink, MQA) for prefill.
    /// q/out `[nQuery, nQHeads, headDim]`; k/v `[kvStride, headDim]` per kv
    /// head (absolute position). Query q_pos attends KV
    /// `[max(0, kvBase+q_pos+1-window) .. kvBase+q_pos]`. Threads = TGs×32.
    public static func sdpaPrefillD512Sink(
        q: Tensor, k: Tensor, v: Tensor, sinkLogit: Tensor, out: Tensor,
        headDim: Int, nQHeads: Int, kvStride: Int, headsPerGroup: Int,
        window: Int, kvBase: Int, scale: Float, nQuery: Int, on cmd: MTLCommandBuffer
    ) {
        let grd = MTLSize(width: nQHeads * 32, height: nQuery, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let hd = UInt32(headDim); let nq = UInt32(nQHeads); let ks = UInt32(kvStride)
        let hpg = UInt32(headsPerGroup); let w = UInt32(window); let kb = UInt32(kvBase)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_sdpa_prefill_d512_sink_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, sink_logit: sinkLogit.buffer, sink_logitOffset: sinkLogit.offset,
                out: out.buffer, outOffset: out.offset, head_dim: hd, n_q_heads: nq, kv_stride: ks,
                heads_per_group: hpg, window: w, kv_base: kb, scale: scale,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_sdpa_prefill_d512_sink_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, sink_logit: sinkLogit.buffer, sink_logitOffset: sinkLogit.offset,
                out: out.buffer, outOffset: out.offset, head_dim: hd, n_q_heads: nq, kv_stride: ks,
                heads_per_group: hpg, window: w, kv_base: kb, scale: scale,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_sdpa_prefill_d512_sink_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, sink_logit: sinkLogit.buffer, sink_logitOffset: sinkLogit.offset,
                out: out.buffer, outOffset: out.offset, head_dim: hd, n_q_heads: nq, kv_stride: ks,
                heads_per_group: hpg, window: w, kv_base: kb, scale: scale,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("sdpaPrefillD512Sink: unsupported dtype \(out.dtype)")
        }
    }

    /// IQ2_XXS grouped BGEMM (prefill gate/up). Rows pre-permuted by expert;
    /// `indices[row]` = expert id (into the resident pool). `qs`/`dF32` hold
    /// all experts. out `[mTotal, nOut]`. BM=16/BN=32 MMA, 32 threads/TG.
    public static func moeBgemmIQ2XXS(
        x: Tensor, qsAll: Tensor, dAll: Tensor, grid: Tensor, signs: Tensor,
        indices: Tensor, out: Tensor, mTotal: Int, nOut: Int, kIn: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(qsAll.dtype == .u32 && dAll.dtype == .f32 && indices.dtype == .u32)
        let grd = MTLSize(width: (nOut / 32) * 32, height: (mTotal + 15) / 16, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let m = UInt32(mTotal); let n = UInt32(nOut); let k = UInt32(kIn)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_gather_bgemm_iq2xxs_mpp_f16(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                d_f32: dAll.buffer, d_f32Offset: dAll.offset, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_gather_bgemm_iq2xxs_mpp_f32(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                d_f32: dAll.buffer, d_f32Offset: dAll.offset, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_gather_bgemm_iq2xxs_mpp_bf16(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                d_f32: dAll.buffer, d_f32Offset: dAll.offset, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeBgemmIQ2XXS: unsupported dtype \(out.dtype)")
        }
    }

    /// ZERO-COPY IQ2_XXS grouped BGEMM reading raw blocks from an mmap
    /// view buffer (no repack pool). `viewBuf` is the no-copy MTLBuffer;
    /// bound twice (as u8 for qs bytes, f16 for block d). `indices[row]`
    /// is the EXPERT id; `tensorByteOff`/`expertByteStride` locate the
    /// tensor + per-expert stride within the view (bytes).
    public static func moeBgemmIQ2XXSView(
        x: Tensor, viewBuf: MTLBuffer, viewByteOffset: Int,
        grid: Tensor, signs: Tensor, indices: Tensor, out: Tensor,
        mTotal: Int, nOut: Int, kIn: Int, tensorByteOff: Int, expertByteStride: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(indices.dtype == .u32)
        let grd = MTLSize(width: (nOut / 32) * 32, height: (mTotal + 15) / 16, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let m = UInt32(mTotal); let n = UInt32(nOut); let k = UInt32(kIn)
        // qs read raw from the no-copy view (zero-copy bulk); d from a small
        // separate per-expert f32 array (avoids aliasing the buffer as f16).
        let tOff = UInt32(tensorByteOff + viewByteOffset)
        let eStride = UInt32(expertByteStride)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_bgemm_iq2xxs_view_f16(
                x: x.buffer, xOffset: x.offset, view_u8: viewBuf, view_u8Offset: 0, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_bgemm_iq2xxs_view_f32(
                x: x.buffer, xOffset: x.offset, view_u8: viewBuf, view_u8Offset: 0, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_bgemm_iq2xxs_view_bf16(
                x: x.buffer, xOffset: x.offset, view_u8: viewBuf, view_u8Offset: 0, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeBgemmIQ2XXSView: unsupported dtype \(out.dtype)")
        }
    }

    /// POOL-FREE amortized view bgemm: bm64 MMA reading raw resident IQ2
    /// blocks via aligned u16 (114 GB/s raw read; 34 GB/s amortized — faster
    /// than the pool bm64, with NO repack). viewBuf = resident mmap view;
    /// indices = GLOBAL expert ids. See moe_bgemm_iq2xxs_view_u16_bm64.rs.
    public static func moeBgemmIQ2XXSViewU16Bm64(
        x: Tensor, viewBuf: MTLBuffer, viewByteOffset: Int,
        grid: Tensor, signs: Tensor, indices: Tensor, out: Tensor,
        mTotal: Int, nOut: Int, kIn: Int, tensorByteOff: Int, expertByteStride: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(indices.dtype == .u32)
        let grd = MTLSize(width: nOut / 64, height: (mTotal + 63) / 64, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let m = UInt32(mTotal); let n = UInt32(nOut); let k = UInt32(kIn)
        let tOff = UInt32(tensorByteOff + viewByteOffset); let eStride = UInt32(expertByteStride)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_bgemm_iq2xxs_view_u16_bm64_f16_threadgroups(
                x: x.buffer, xOffset: x.offset, view_u16: viewBuf, view_u16Offset: 0, view_f16: viewBuf, view_f16Offset: 0, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_bgemm_iq2xxs_view_u16_bm64_f32_threadgroups(
                x: x.buffer, xOffset: x.offset, view_u16: viewBuf, view_u16Offset: 0, view_f16: viewBuf, view_f16Offset: 0, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_bgemm_iq2xxs_view_u16_bm64_bf16_threadgroups(
                x: x.buffer, xOffset: x.offset, view_u16: viewBuf, view_u16Offset: 0, view_f16: viewBuf, view_f16Offset: 0, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeBgemmIQ2XXSViewU16Bm64: unsupported dtype \(out.dtype)")
        }
    }

    /// Q2_K grouped BGEMM (prefill down). See `moeBgemmIQ2XXS`.
    public static func moeBgemmQ2K(
        x: Tensor, qsAll: Tensor, scalesAll: Tensor, dAll: Tensor, dminAll: Tensor,
        indices: Tensor, out: Tensor, mTotal: Int, nOut: Int, kIn: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(qsAll.dtype == .u32 && scalesAll.dtype == .u8 && dAll.dtype == .f32 && dminAll.dtype == .f32)
        let grd = MTLSize(width: (nOut / 32) * 32, height: (mTotal + 15) / 16, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let m = UInt32(mTotal); let n = UInt32(nOut); let k = UInt32(kIn)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_gather_bgemm_q2k_mpp_f16(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                scales: scalesAll.buffer, scalesOffset: scalesAll.offset, d_f32: dAll.buffer, d_f32Offset: dAll.offset,
                dmin_f32: dminAll.buffer, dmin_f32Offset: dminAll.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_gather_bgemm_q2k_mpp_f32(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                scales: scalesAll.buffer, scalesOffset: scalesAll.offset, d_f32: dAll.buffer, d_f32Offset: dAll.offset,
                dmin_f32: dminAll.buffer, dmin_f32Offset: dminAll.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_gather_bgemm_q2k_mpp_bf16(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                scales: scalesAll.buffer, scalesOffset: scalesAll.offset, d_f32: dAll.buffer, d_f32Offset: dAll.offset,
                dmin_f32: dminAll.buffer, dmin_f32Offset: dminAll.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeBgemmQ2K: unsupported dtype \(out.dtype)")
        }
    }

    /// ZERO-COPY Q2_K grouped BGEMM (prefill down): reads raw 84-byte Q2_K
    /// blocks straight from a no-copy mmap view buffer, indexed by expert id.
    /// See `moeBgemmIQ2XXSView`. expertByteStride = nBlocksPerExpert*84.
    /// Q2_K view-BM64: raw 84-byte Q2_K blocks → bm64 64×64×32 coop_tile MMA,
    /// NO deinterleave pool. Same speed as the pool bm64, eliminates the ~342ms/
    /// layer Q2_K pool build. indices = slot/expert id; d/dmin via the f16 view
    /// (= viewBuf), scales/qs via the u16 view. Live-compiled (name has bgemm).
    public static func moeBgemmQ2KViewU16Bm64(
        x: Tensor, viewBuf: MTLBuffer, viewByteOffset: Int,
        indices: Tensor, out: Tensor,
        mTotal: Int, nOut: Int, kIn: Int, tensorByteOff: Int, expertByteStride: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(indices.dtype == .u32)
        let grd = MTLSize(width: nOut / 64, height: (mTotal + 63) / 64, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let m = UInt32(mTotal); let n = UInt32(nOut); let k = UInt32(kIn)
        let tOff = UInt32(tensorByteOff + viewByteOffset); let eStride = UInt32(expertByteStride)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_bgemm_q2k_view_u16_bm64_f16_threadgroups(
                x: x.buffer, xOffset: x.offset, view_u16: viewBuf, view_u16Offset: 0, view_f16: viewBuf, view_f16Offset: 0,
                indices: indices.buffer, indicesOffset: indices.offset, out: out.buffer, outOffset: out.offset,
                m_total: m, n_out: n, k_in: k, tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_bgemm_q2k_view_u16_bm64_f32_threadgroups(
                x: x.buffer, xOffset: x.offset, view_u16: viewBuf, view_u16Offset: 0, view_f16: viewBuf, view_f16Offset: 0,
                indices: indices.buffer, indicesOffset: indices.offset, out: out.buffer, outOffset: out.offset,
                m_total: m, n_out: n, k_in: k, tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_bgemm_q2k_view_u16_bm64_bf16_threadgroups(
                x: x.buffer, xOffset: x.offset, view_u16: viewBuf, view_u16Offset: 0, view_f16: viewBuf, view_f16Offset: 0,
                indices: indices.buffer, indicesOffset: indices.offset, out: out.buffer, outOffset: out.offset,
                m_total: m, n_out: n, k_in: k, tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeBgemmQ2KViewU16Bm64: unsupported dtype \(out.dtype)")
        }
    }

    public static func moeBgemmQ2KView(
        x: Tensor, viewBuf: MTLBuffer, viewByteOffset: Int,
        indices: Tensor, out: Tensor,
        mTotal: Int, nOut: Int, kIn: Int, tensorByteOff: Int, expertByteStride: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(indices.dtype == .u32)
        let grd = MTLSize(width: (nOut / 32) * 32, height: (mTotal + 15) / 16, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let m = UInt32(mTotal); let n = UInt32(nOut); let k = UInt32(kIn)
        let tOff = UInt32(tensorByteOff + viewByteOffset)
        let eStride = UInt32(expertByteStride)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_bgemm_q2k_view_f16(
                x: x.buffer, xOffset: x.offset, view_u8: viewBuf, view_u8Offset: 0,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_bgemm_q2k_view_f32(
                x: x.buffer, xOffset: x.offset, view_u8: viewBuf, view_u8Offset: 0,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_bgemm_q2k_view_bf16(
                x: x.buffer, xOffset: x.offset, view_u8: viewBuf, view_u8Offset: 0,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                tensor_byte_off: tOff, expert_byte_stride: eStride, gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeBgemmQ2KView: unsupported dtype \(out.dtype)")
        }
    }

    /// FAST prefill MoE IQ2_XXS GEMV-over-rows: direct simd_sum dot-product
    /// over all M rows in one dispatch (~24x the coop-tile bgemm). Reads the
    /// SAME resident split pool the bgemm uses; `expertIds[row]` = the row's
    /// packed pool slot. out[row, mOut]. grid (mOut, mTotal).
    public static func moeGemvRowsIQ2XXS(
        x: Tensor, qsAll: Tensor, dAll: Tensor, expertIds: Tensor, grid: Tensor, signs: Tensor,
        out: Tensor, mTotal: Int, mOut: Int, kIn: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(qsAll.dtype == .u32 && dAll.dtype == .f32 && expertIds.dtype == .u32)
        let grd = MTLSize(width: mOut * 32, height: mTotal, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let k = UInt32(kIn); let mo = UInt32(mOut); let mt = UInt32(mTotal)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_gemv_rows_iq2xxs_f16(
                x: x.buffer, xOffset: x.offset, qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset, expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                grid: grid.buffer, gridOffset: grid.offset, signs: signs.buffer, signsOffset: signs.offset,
                out: out.buffer, outOffset: out.offset, k_in: k, m_out: mo, m_total: mt,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_gemv_rows_iq2xxs_f32(
                x: x.buffer, xOffset: x.offset, qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset, expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                grid: grid.buffer, gridOffset: grid.offset, signs: signs.buffer, signsOffset: signs.offset,
                out: out.buffer, outOffset: out.offset, k_in: k, m_out: mo, m_total: mt,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_gemv_rows_iq2xxs_bf16(
                x: x.buffer, xOffset: x.offset, qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset, expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                grid: grid.buffer, gridOffset: grid.offset, signs: signs.buffer, signsOffset: signs.offset,
                out: out.buffer, outOffset: out.offset, k_in: k, m_out: mo, m_total: mt,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeGemvRowsIQ2XXS: unsupported dtype \(out.dtype)")
        }
    }

    /// Dense Q8 GEMM via cooperative-tensor MMA (~6× the scalar gemmQ8).
    /// Drop-in for gemmQ8 (q/kv/q_a/q_b/O-LoRA-B/shared-expert). Live-compiled
    /// (name has _mpp_ → PSOCache.isMppKernel). 64×64×32 coop_tile, 128 thr/tg.
    public static func gemmQ8Mpp(
        qs: Tensor, dF32: Tensor, input: Tensor, out: Tensor,
        inDim: Int, outDim: Int, nRows: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(qs.dtype == .u32 && dF32.dtype == .f32)
        let grd = MTLSize(width: (outDim + 63) / 64, height: (nRows + 63) / 64, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let n = UInt32(nRows); let o = UInt32(outDim); let k = UInt32(inDim)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_gemm_q8_mpp_f16(
                x: input.buffer, xOffset: input.offset, qs: qs.buffer, qsOffset: qs.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset, out: out.buffer, outOffset: out.offset,
                n_rows: n, out_dim: o, k_in: k, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_gemm_q8_mpp_f32(
                x: input.buffer, xOffset: input.offset, qs: qs.buffer, qsOffset: qs.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset, out: out.buffer, outOffset: out.offset,
                n_rows: n, out_dim: o, k_in: k, gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gemm_q8_mpp_bf16(
                x: input.buffer, xOffset: input.offset, qs: qs.buffer, qsOffset: qs.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset, out: out.buffer, outOffset: out.offset,
                n_rows: n, out_dim: o, k_in: k, gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("gemmQ8Mpp: unsupported dtype \(out.dtype)")
        }
    }

    /// GROUPED Q8 GEMM via cooperative-tensor MMA — the MMA O-LoRA-A (vs the
    /// scalar groupedGemmQ8). 64×64×32 coop_tile, live-compiled (_mpp_).
    public static func groupedGemmQ8Mpp(
        qs: Tensor, dF32: Tensor, input: Tensor, out: Tensor,
        inDim: Int, outDim: Int, nRows: Int, nGroups: Int, rowsPerGroup: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(qs.dtype == .u32 && dF32.dtype == .f32)
        let grd = MTLSize(width: (outDim + 63) / 64, height: (nRows + 63) / 64, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let n = UInt32(nRows); let o = UInt32(outDim); let k = UInt32(inDim)
        let ng = UInt32(nGroups); let rpg = UInt32(rowsPerGroup)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_grouped_gemm_q8_mpp_f16(
                x: input.buffer, xOffset: input.offset, qs: qs.buffer, qsOffset: qs.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset, out: out.buffer, outOffset: out.offset,
                n_rows: n, out_dim: o, k_in: k, n_groups: ng, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_grouped_gemm_q8_mpp_f32(
                x: input.buffer, xOffset: input.offset, qs: qs.buffer, qsOffset: qs.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset, out: out.buffer, outOffset: out.offset,
                n_rows: n, out_dim: o, k_in: k, n_groups: ng, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_grouped_gemm_q8_mpp_bf16(
                x: input.buffer, xOffset: input.offset, qs: qs.buffer, qsOffset: qs.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset, out: out.buffer, outOffset: out.offset,
                n_rows: n, out_dim: o, k_in: k, n_groups: ng, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("groupedGemmQ8Mpp: unsupported dtype \(out.dtype)")
        }
    }

    /// GROUPED Q8 GEMM (scalar) — amortized O-LoRA-A (replaces the per-token
    /// groupedGemvQ8Rows, the #1 prefill attention hotspot ~47ms/layer).
    /// Tiled 32×32, weight dequanted once/tile, input grouped: output col o
    /// reads input group g=o/rowsPerGroup of an (nGroups*inDim)-wide row.
    public static func groupedGemmQ8(
        qs: Tensor, dF32: Tensor, input: Tensor, out: Tensor,
        inDim: Int, outDim: Int, nRows: Int, nGroups: Int, rowsPerGroup: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(qs.dtype == .u32 && dF32.dtype == .f32)
        let gx = (outDim + 31) / 32; let gy = (nRows + 31) / 32
        let grd = MTLSize(width: gx * 1024, height: gy, depth: 1)
        let tg = MTLSize(width: 1024, height: 1, depth: 1)
        let i = UInt32(inDim); let o = UInt32(outDim); let n = UInt32(nRows)
        let ng = UInt32(nGroups); let rpg = UInt32(rowsPerGroup)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_grouped_gemm_q8_f16(
                qs: qs.buffer, qsOffset: qs.offset, d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                input: input.buffer, inputOffset: input.offset, out: out.buffer, outOffset: out.offset,
                in_dim: i, out_dim: o, n_rows: n, n_groups: ng, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_grouped_gemm_q8_f32(
                qs: qs.buffer, qsOffset: qs.offset, d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                input: input.buffer, inputOffset: input.offset, out: out.buffer, outOffset: out.offset,
                in_dim: i, out_dim: o, n_rows: n, n_groups: ng, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_grouped_gemm_q8_bf16(
                qs: qs.buffer, qsOffset: qs.offset, d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                input: input.buffer, inputOffset: input.offset, out: out.buffer, outOffset: out.offset,
                in_dim: i, out_dim: o, n_rows: n, n_groups: ng, rows_per_group: rpg,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("groupedGemmQ8: unsupported dtype \(out.dtype)")
        }
    }

    /// WEIGHT-STATIONARY prefill MoE IQ2_XXS gemv — dequants each expert's
    /// weight row ONCE into threadgroup mem and reuses it across its rows in
    /// the tile (amortized like bm64 but at gemv speed, ~3.7x bm64). Plain
    /// gemv (no MMA) so it loads correctly from the metallib. rowsPerTile=8.
    public static func moeGemvWsIQ2XXS(
        x: Tensor, qsAll: Tensor, dAll: Tensor, expertIds: Tensor, grid: Tensor, signs: Tensor,
        out: Tensor, mTotal: Int, mOut: Int, kIn: Int, rowsPerTile: Int = 8, on cmd: MTLCommandBuffer
    ) {
        precondition(qsAll.dtype == .u32 && dAll.dtype == .f32 && expertIds.dtype == .u32)
        let nTiles = (mTotal + rowsPerTile - 1) / rowsPerTile
        let grd = MTLSize(width: mOut * 32, height: nTiles, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let k = UInt32(kIn); let mo = UInt32(mOut); let mt = UInt32(mTotal); let rpt = UInt32(rowsPerTile)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_gemv_ws_iq2xxs_f16(
                x: x.buffer, xOffset: x.offset, qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset, expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                grid: grid.buffer, gridOffset: grid.offset, signs: signs.buffer, signsOffset: signs.offset,
                out: out.buffer, outOffset: out.offset, k_in: k, m_out: mo, m_total: mt, rows_per_tile: rpt,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_gemv_ws_iq2xxs_f32(
                x: x.buffer, xOffset: x.offset, qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset, expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                grid: grid.buffer, gridOffset: grid.offset, signs: signs.buffer, signsOffset: signs.offset,
                out: out.buffer, outOffset: out.offset, k_in: k, m_out: mo, m_total: mt, rows_per_tile: rpt,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_gemv_ws_iq2xxs_bf16(
                x: x.buffer, xOffset: x.offset, qs_all: qsAll.buffer, qs_allOffset: qsAll.offset,
                d_all: dAll.buffer, d_allOffset: dAll.offset, expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                grid: grid.buffer, gridOffset: grid.offset, signs: signs.buffer, signsOffset: signs.offset,
                out: out.buffer, outOffset: out.offset, k_in: k, m_out: mo, m_total: mt, rows_per_tile: rpt,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeGemvWsIQ2XXS: unsupported dtype \(out.dtype)")
        }
    }

    /// FAST prefill MoE Q2_K GEMV-over-rows (down). See moeGemvRowsIQ2XXS.
    public static func moeGemvRowsQ2K(
        x: Tensor, qsAll: Tensor, scalesAll: Tensor, dAll: Tensor, dminAll: Tensor, expertIds: Tensor,
        out: Tensor, mTotal: Int, mOut: Int, kIn: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(qsAll.dtype == .u32 && scalesAll.dtype == .u8 && dAll.dtype == .f32 && dminAll.dtype == .f32 && expertIds.dtype == .u32)
        let grd = MTLSize(width: mOut * 32, height: mTotal, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let k = UInt32(kIn); let mo = UInt32(mOut); let mt = UInt32(mTotal)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_gemv_rows_q2k_f16(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                scales: scalesAll.buffer, scalesOffset: scalesAll.offset, d_f32: dAll.buffer, d_f32Offset: dAll.offset,
                dmin_f32: dminAll.buffer, dmin_f32Offset: dminAll.offset, expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                out: out.buffer, outOffset: out.offset, k_in: k, m_out: mo, m_total: mt,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_gemv_rows_q2k_f32(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                scales: scalesAll.buffer, scalesOffset: scalesAll.offset, d_f32: dAll.buffer, d_f32Offset: dAll.offset,
                dmin_f32: dminAll.buffer, dmin_f32Offset: dminAll.offset, expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                out: out.buffer, outOffset: out.offset, k_in: k, m_out: mo, m_total: mt,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_gemv_rows_q2k_bf16(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                scales: scalesAll.buffer, scalesOffset: scalesAll.offset, d_f32: dAll.buffer, d_f32Offset: dAll.offset,
                dmin_f32: dminAll.buffer, dmin_f32Offset: dminAll.offset, expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                out: out.buffer, outOffset: out.offset, k_in: k, m_out: mo, m_total: mt,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeGemvRowsQ2K: unsupported dtype \(out.dtype)")
        }
    }

    /// bm64 IQ2_XXS BGEMM (64×64×32, 4 simdgroups) — ~2x the 16×32 bgemm,
    /// amortized + byte-exact. Same pool/indices as moeBgemmIQ2XXS; grid is
    /// n_out/64 × ceil(M/64), 128 threads/tg.
    public static func moeBgemmIQ2XXSBm64(
        x: Tensor, qsAll: Tensor, dAll: Tensor, grid: Tensor, signs: Tensor,
        indices: Tensor, out: Tensor, mTotal: Int, nOut: Int, kIn: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(qsAll.dtype == .u32 && dAll.dtype == .f32 && indices.dtype == .u32)
        // coop_tile kernel → MUST dispatch THREADGROUPS (dispatchThreads breaks
        // simdgroup-matrix). gridSize = threadgroup counts.
        let grd = MTLSize(width: nOut / 64, height: (mTotal + 63) / 64, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let m = UInt32(mTotal); let n = UInt32(nOut); let k = UInt32(kIn)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_bgemm_iq2xxs_bm64_f16_threadgroups(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                d_f32: dAll.buffer, d_f32Offset: dAll.offset, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_bgemm_iq2xxs_bm64_f32_threadgroups(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                d_f32: dAll.buffer, d_f32Offset: dAll.offset, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_bgemm_iq2xxs_bm64_bf16_threadgroups(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                d_f32: dAll.buffer, d_f32Offset: dAll.offset, grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeBgemmIQ2XXSBm64: unsupported dtype \(out.dtype)")
        }
    }

    /// bm64 Q2_K BGEMM (down). See moeBgemmIQ2XXSBm64.
    public static func moeBgemmQ2KBm64(
        x: Tensor, qsAll: Tensor, scalesAll: Tensor, dAll: Tensor, dminAll: Tensor,
        indices: Tensor, out: Tensor, mTotal: Int, nOut: Int, kIn: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(qsAll.dtype == .u32 && scalesAll.dtype == .u8 && dAll.dtype == .f32 && dminAll.dtype == .f32 && indices.dtype == .u32)
        // coop_tile → dispatchThreadgroups (gridSize in threadgroups).
        let grd = MTLSize(width: nOut / 64, height: (mTotal + 63) / 64, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let m = UInt32(mTotal); let n = UInt32(nOut); let k = UInt32(kIn)
        switch out.dtype {
        case .f16:
            MetalTileKernels.ffai_moe_bgemm_q2k_bm64_f16_threadgroups(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                scales: scalesAll.buffer, scalesOffset: scalesAll.offset, d_f32: dAll.buffer, d_f32Offset: dAll.offset,
                dmin_f32: dminAll.buffer, dmin_f32Offset: dminAll.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .f32:
            MetalTileKernels.ffai_moe_bgemm_q2k_bm64_f32_threadgroups(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                scales: scalesAll.buffer, scalesOffset: scalesAll.offset, d_f32: dAll.buffer, d_f32Offset: dAll.offset,
                dmin_f32: dminAll.buffer, dmin_f32Offset: dminAll.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_bgemm_q2k_bm64_bf16_threadgroups(
                x: x.buffer, xOffset: x.offset, qs: qsAll.buffer, qsOffset: qsAll.offset,
                scales: scalesAll.buffer, scalesOffset: scalesAll.offset, d_f32: dAll.buffer, d_f32Offset: dAll.offset,
                dmin_f32: dminAll.buffer, dmin_f32Offset: dminAll.offset, indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset, m_total: m, n_out: n, k_in: k,
                gridSize: grd, threadgroupSize: tg, on: cmd)
        default: fatalError("moeBgemmQ2KBm64: unsupported dtype \(out.dtype)")
        }
    }
}
