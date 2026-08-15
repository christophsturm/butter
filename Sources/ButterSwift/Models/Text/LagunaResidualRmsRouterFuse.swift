// Copyright 2026 Tom Turney (@TheTom)
//
// T28 lever 2: decode-only fold of the post-attention residual add +
// RMSNorm + MoE router GEMV into one dispatch. Butter already fuses
// residual+RMSNorm via `Ops.addAndRmsNorm` (`iron_add_rms_norm_bf16`, one
// threadgroup owns the whole 2048-wide row); this file's kernel is that
// same math, byte-matched, plus a router GEMV appended in the SAME
// threadgroup. The normed row is already resident in registers across the
// TG's 512 threads, so staging it once into 4 KiB threadgroup memory and
// letting each of the 16 SIMD groups own a 16-row slice of the 256-expert
// router GEMV costs no extra threadgroup launch, only more work inside the
// one that's already running. Router logits therefore are NOT claimed
// byte-identical to the stock separate `gate(h, on: cmd)` GEMV (different
// reduction order/thread partition, the same class of FP-reorder as every
// other decode fuse in this file family); residual/normed outputs ARE
// claimed byte-identical (same unrolled-by-4, same accumulate order, same
// rsqrt(mean+eps) formula as `iron_add_rms_norm_bf16`).
//
// Escape: BUTTER_LAGUNA_RESIDUAL_RMS_ROUTER=0. Default ON when hidden==2048
// && nExperts==256 (Laguna XS's exact shape, no generic remainder path is
// implemented) && the MoE gate is a plain (unquantized) BF16 Linear with no
// bias && `OpsValidation.validateAddRmsNorm` already allows the fast
// addAndRmsNorm path (same eligibility Butter uses today).

import Foundation
import Metal
import IronSwift

enum LagunaResidualRmsRouterFuse {
    /// Opt-in, OFF by default, see T28 README: two attempts (single-TG,
    /// then the rows_per_group=8 multi-TG layout below) both regressed
    /// decode (-50% and -5.3% respectively) vs the stock separate
    /// `Ops.gemv` router dispatch, which already spreads the 256-row router
    /// GEMV across 256 threadgroups (one per output row), 8x more
    /// GPU-core parallelism than this fold's 32 threadgroups (8 rows/TG)
    /// can offer without a much bigger redesign. Kept as a correctness-
    /// verified, tested opt-in (`BUTTER_LAGUNA_RESIDUAL_RMS_ROUTER=1`) per
    /// this repo's own precedent (`BUTTER_LAGUNA_FUSED_SHARED_SWIGLU`,
    /// `BUTTER_NVFP4_MOE_BM64`) for correct-but-slower kernels.
    static var enabled: Bool {
        ButterEnv["BUTTER_LAGUNA_RESIDUAL_RMS_ROUTER"] == "1"
    }

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        if let p = try? PSOCache.shared.pipelineStateThrowing(for: name) {
            psoCache[name] = p
            return p
        }
        let opts = MTLCompileOptions()
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    private static let src = """
    #include <metal_stdlib>
    using namespace metal;

    // Multi-threadgroup, rows-per-group=8 layout:
    // `n_experts / rows_per_group` threadgroups (e.g. 256/8 = 32), each
    // independently redoes the (cheap, ~2048-element) residual+RMSNorm row.
    // This is redundant across threadgroups, but that's what keeps the
    // router GEMV parallel across GPU cores instead of collapsing to one
    // threadgroup / one core. A first version dispatched a single TG for
    // everything (normed row "already resident, GEMV free") and measured a
    // 2x DECODE REGRESSION (49 vs 100 tok/s): the separate stock
    // `gate(h,on:cmd)` GEMV was already spread across many threadgroups, and
    // collapsing residual+norm+router into one TG serialized all 256 rows
    // onto a single core. This redundant-recompute layout is the fix (see
    // T28 README for the measured numbers of both attempts).
    //
    // TPG = 512 threads (16 SIMD groups, matches hidden/4). Only the first
    // `rows_per_group` SIMD groups do GEMV work per threadgroup; only
    // threadgroup 0 writes residual_out/normed_out (every TG computes the
    // same values, so writing from just one avoids redundant global bandwidth).
    kernel void laguna_residual_rms_router_bf16_2048_rpg8(
        const device bfloat *residual [[buffer(0)]],
        const device bfloat *attn_out [[buffer(1)]],
        const device bfloat *norm_weight [[buffer(2)]],
        const device float *eps_buf [[buffer(3)]],
        const device bfloat *router_weight [[buffer(4)]],
        device bfloat *residual_out [[buffer(5)]],
        device bfloat *normed_out [[buffer(6)]],
        device bfloat *logits_out [[buffer(7)]],
        constant uint &n [[buffer(8)]],
        constant uint &n_experts [[buffer(9)]],
        uint tid [[thread_position_in_threadgroup]],
        uint lsize [[threads_per_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]
    ) {
        constexpr uint rows_per_group = 8u;
        uint n_simd = lsize / 32u;
        threadgroup float ssq_sg[32];
        threadgroup float normed_tg[2048];

        uint col = tid * 4u;
        float a0 = float(residual[col + 0u]);
        float a1 = float(residual[col + 1u]);
        float a2 = float(residual[col + 2u]);
        float a3 = float(residual[col + 3u]);
        float b0 = float(attn_out[col + 0u]);
        float b1 = float(attn_out[col + 1u]);
        float b2 = float(attn_out[col + 2u]);
        float b3 = float(attn_out[col + 3u]);
        float s0 = a0 + b0;
        float s1 = a1 + b1;
        float s2 = a2 + b2;
        float s3 = a3 + b3;
        // Same left-to-right accumulate order as iron_add_rms_norm_bf16's
        // `((((s0*s0)+s1*s1)+s2*s2)+s3*s3)`.
        float partial_ssq = ((((s0 * s0) + (s1 * s1)) + (s2 * s2)) + (s3 * s3));

        float tg_ssq;
        {
            float sv = simd_sum(partial_ssq);
            if (simd_lane == 0u) ssq_sg[simd_group] = sv;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0u) {
                float wv = simd_lane < n_simd ? ssq_sg[simd_lane] : 0.0f;
                ssq_sg[0] = simd_sum(wv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tg_ssq = ssq_sg[0];
        }
        float eps = eps_buf[0];
        float rms = rsqrt((tg_ssq / float(n)) + eps);

        bfloat n0 = bfloat((s0 * rms) * float(norm_weight[col + 0u]));
        bfloat n1 = bfloat((s1 * rms) * float(norm_weight[col + 1u]));
        bfloat n2 = bfloat((s2 * rms) * float(norm_weight[col + 2u]));
        bfloat n3 = bfloat((s3 * rms) * float(norm_weight[col + 3u]));
        normed_tg[col + 0u] = float(n0);
        normed_tg[col + 1u] = float(n1);
        normed_tg[col + 2u] = float(n2);
        normed_tg[col + 3u] = float(n3);
        if (tgid == 0u) {
            residual_out[col + 0u] = bfloat(s0);
            residual_out[col + 1u] = bfloat(s1);
            residual_out[col + 2u] = bfloat(s2);
            residual_out[col + 3u] = bfloat(s3);
            normed_out[col + 0u] = n0;
            normed_out[col + 1u] = n1;
            normed_out[col + 2u] = n2;
            normed_out[col + 3u] = n3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Router GEMV over this threadgroup's `rows_per_group` output rows:
        // SIMD groups [0, rows_per_group) each own one row; the remaining
        // SIMD groups (recompute-only, per the redundant layout above) sit
        // idle for this phase, the tradeoff of this row-ownership knob.
        if (simd_group < rows_per_group) {
            uint row = tgid * rows_per_group + simd_group;
            if (row < n_experts) {
                float acc = 0.0f;
                uint row_off = row * n;
                for (uint k = simd_lane; k < n; k += 32u) {
                    acc += float(router_weight[row_off + k]) * normed_tg[k];
                }
                acc = simd_sum(acc);
                if (simd_lane == 0u) logits_out[row] = bfloat(acc);
            }
        }
    }
    """

    /// Fused residual add + RMSNorm + MoE router GEMV. Returns
    /// (residual: h2, normed: ffnIn, logits).
    static func apply(
        residual: Tensor, attnOut: Tensor,
        normWeight: Tensor, eps: Float,
        routerWeight: Tensor,
        hidden: Int, nExperts: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> (residual: Tensor, normed: Tensor, logits: Tensor) {
        precondition(hidden == 2048 && nExperts == 256)
        precondition(residual.dtype == .bf16 && attnOut.dtype == .bf16)
        precondition(routerWeight.dtype == .bf16)

        let residualOut = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
        let normedOut = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
        let logitsOut = Tensor.empty(shape: [nExperts], dtype: .bf16, device: device)

        let mtl = device.mtlDevice
        let p = pso(
            "laguna_residual_rms_router_bf16_2048_rpg8", source: src, device: mtl)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            return (residualOut, normedOut, logitsOut)
        }
        enc.setComputePipelineState(p)
        enc.setBuffer(residual.buffer, offset: residual.offset, index: 0)
        enc.setBuffer(attnOut.buffer, offset: attnOut.offset, index: 1)
        enc.setBuffer(normWeight.buffer, offset: normWeight.offset, index: 2)
        var epsVal = eps
        enc.setBytes(&epsVal, length: 4, index: 3)
        enc.setBuffer(routerWeight.buffer, offset: routerWeight.offset, index: 4)
        enc.setBuffer(residualOut.buffer, offset: residualOut.offset, index: 5)
        enc.setBuffer(normedOut.buffer, offset: normedOut.offset, index: 6)
        enc.setBuffer(logitsOut.buffer, offset: logitsOut.offset, index: 7)
        var n = UInt32(hidden)
        var ne = UInt32(nExperts)
        enc.setBytes(&n, length: 4, index: 8)
        enc.setBytes(&ne, length: 4, index: 9)
        let rowsPerGroup = 8
        let numGroups = (nExperts + rowsPerGroup - 1) / rowsPerGroup
        enc.dispatchThreadgroups(
            MTLSize(width: numGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: hidden / 4, height: 1, depth: 1))
        enc.endEncoding()
        return (residualOut, normedOut, logitsOut)
    }
}
