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
// Decode-only SDPA fusion (T24 "TASK 2"): GQA head-pairing + exact-alpha
// rescale skip, two new techniques applied to *butter's own*
// iron-DSL-generated decode kernel algorithm, which this file mirrors
// line-for-line minus the two changes below (see
// `Sources/IronSwift/Resources/kernels/iron_sdpa_decode_bf16.metal` for the
// generated baseline this was checked against). Hand-written live-compiled
// Metal via the `LagunaNvfp4Fuse`/`LagunaAttnInt8` `pso()` pattern: this
// is a fixed-shape decode-only specialization, not a generic primitive, so
// it stays out of the iron DSL (same rationale as `LagunaGatedOProjFuse`).
//
// 1. **GQA head-pairing**: one threadgroup now serves 2 adjacent query
//    heads instead of 1, reading each K/V element once and reusing it for
//    both heads' dot products / weighted-value accumulation. Halves both
//    the K/V bytes streamed from HBM for the paired portion of a GQA
//    group *and* the threadgroup dispatch count (nHeads/2 instead of
//    nHeads), directly on T20's "dispatch-count, not bandwidth" finding.
//    Only valid when `headsPerGroup` (nHeads/nKVHeads) is even, so a pair
//    never straddles two different KV heads: true for both Laguna head
//    counts (48/8=6, 64/8=8). `headDim` is fixed at 128 (Laguna's only
//    decode head width); `has_sink` (GPT-OSS-style learned sink logit) is
//    not modeled, since Laguna's `Ops.sdpaDecode` call site never sets it,
//    and algebraically it's a no-op at `has_sink=0` in the generated kernel
//    this mirrors (see file history / PR description for the derivation).
//    `sinkEnd`/`windowStart` (the *positional* sink+sliding-window fast
//    path, unrelated to the logit sink) are still fully supported.
//
// 2. **ALPHASKIP**: the online-softmax rescale factor `exp(oldMax -
//    newMax)` is exactly `1.0f` (bit-exact, not approximately 1) whenever
//    the running max didn't change this step. `os[i] * 1.0f == os[i]`
//    exactly under IEEE 754 (multiply by exact 1.0 has zero rounding
//    error), so `if (factor == 1.0f) { os[i] += weight*v[i]; } else {
//    os[i] = os[i]*factor + weight*v[i]; }` is bit-identical to always
//    taking the multiply branch, so this stays an exact-comparison skip
//    rather than an approximation. (A `fma`-fused single-rounding-step
//    rewrite of the *general* branch would NOT be bit-exact against the
//    original two-op `mul` then `add`, so that form is deliberately not
//    shipped here.)
//
// Escape: `BUTTER_LAGUNA_SDPA_GQA2=0` restores the stock `Ops.sdpaDecode`
// path. Default ON when shapes allow (`headDim==128`, `headsPerGroup`
// even, `nHeads` even, bf16).

import Foundation
import Metal
import IronSwift

enum LagunaSdpaDecodeGqa2 {
    /// Default ON; set `BUTTER_LAGUNA_SDPA_GQA2=0` to restore
    /// `Ops.sdpaDecode` unconditionally.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_SDPA_GQA2"] != "0"
    }

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        let opts = MTLCompileOptions()
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    /// `headDim` must be 128 (32 simdgroups × 32 lanes × 4 elt/lane, same
    /// geometry as `iron_sdpa_decode_bf16`); `nHeads` and `heads_per_group`
    /// (`nHeads/nKVHeads`) must both be even so consecutive-index pairing
    /// never straddles a KV-head boundary.
    static func canFuse(headDim: Int, nHeads: Int, nKVHeads: Int, dtype: DType) -> Bool {
        guard enabled, headDim == 128, dtype == .bf16, nKVHeads > 0,
            nHeads.isMultiple(of: nKVHeads), nHeads.isMultiple(of: 2)
        else { return false }
        let headsPerGroup = nHeads / nKVHeads
        return headsPerGroup.isMultiple(of: 2)
    }

    // One threadgroup per Q-head PAIR (`tgid_x` = pair index). 1024
    // threads (32 simdgroups × 32 lanes), matching `iron_sdpa_decode_bf16`.
    // `has_sink` is not modeled (see file header), algebraically identical
    // to the generated kernel's own `has_sink=0` path.
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void laguna_sdpa_decode_gqa2_bf16_128(
        const device bfloat *q [[buffer(0)]],
        const device bfloat *k [[buffer(1)]],
        const device bfloat *v [[buffer(2)]],
        device bfloat *out [[buffer(3)]],
        constant uint &n_kv [[buffer(4)]],
        constant uint &kv_stride [[buffer(5)]],
        constant uint &heads_per_group [[buffer(6)]],
        constant uint &sink_end [[buffer(7)]],
        constant uint &window_start [[buffer(8)]],
        constant float &scale [[buffer(9)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]])
    {
        constexpr uint head_dim = 128u;
        constexpr uint n_simd = 32u;
        threadgroup float tg_max[32];
        threadgroup float tg_sum[32];
        threadgroup float tg_out0[32 * 33];
        threadgroup float tg_out1[32 * 33];
        threadgroup float tg_out2[32 * 33];
        threadgroup float tg_out3[32 * 33];

        uint pair = tgid.x;
        uint qHeadA = pair * 2u;
        uint qHeadB = qHeadA + 1u;
        uint kvHead = qHeadA / heads_per_group;  // == qHeadB / heads_per_group (even group size)
        uint kvBase = kvHead * kv_stride * head_dim;
        uint d0 = lane * 4u;

        // Scaled Q, both heads.
        float qA[4];
        float qB[4];
        {
            const device bfloat *qaP = q + qHeadA * head_dim + d0;
            const device bfloat *qbP = q + qHeadB * head_dim + d0;
            for (uint i = 0; i < 4u; ++i) {
                qA[i] = float(qaP[i]) * scale;
                qB[i] = float(qbP[i]) * scale;
            }
        }

        float runMaxA = -INFINITY, runSumA = 0.0f;
        float runMaxB = -INFINITY, runSumB = 0.0f;
        float osA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float osB[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Shared K/V scan: [0, sink_end) then [window_start, n_kv), same
        // dense-attention-when-both-zero contract as the stock kernel.
        for (uint phase = 0u; phase < 2u; ++phase) {
            uint lo = phase == 0u ? sg : (window_start + sg);
            uint hi = phase == 0u ? sink_end : n_kv;
            for (uint i = lo; i < hi; i += n_simd) {
                uint kv0 = kvBase + i * head_dim + d0;
                const device bfloat *kP = k + kv0;
                float kv[4];
                for (uint j = 0; j < 4u; ++j) { kv[j] = float(kP[j]); }

                float partialA = 0.0f, partialB = 0.0f;
                for (uint j = 0; j < 4u; ++j) {
                    partialA += qA[j] * kv[j];
                    partialB += qB[j] * kv[j];
                }
                float scoreA = simd_sum(partialA);
                float scoreB = simd_sum(partialB);

                const device bfloat *vP = v + kv0;
                float vv[4];
                for (uint j = 0; j < 4u; ++j) { vv[j] = float(vP[j]); }

                // Head A online-softmax update, ALPHASKIP on exact factor==1.
                {
                    float newMax = max(runMaxA, scoreA);
                    float factor = exp(runMaxA - newMax);
                    float weight = exp(scoreA - newMax);
                    runSumA = runSumA * factor + weight;
                    runMaxA = newMax;
                    if (factor == 1.0f) {
                        for (uint j = 0; j < 4u; ++j) { osA[j] = osA[j] + weight * vv[j]; }
                    } else {
                        for (uint j = 0; j < 4u; ++j) { osA[j] = osA[j] * factor + weight * vv[j]; }
                    }
                }
                // Head B, same K/V values reused.
                {
                    float newMax = max(runMaxB, scoreB);
                    float factor = exp(runMaxB - newMax);
                    float weight = exp(scoreB - newMax);
                    runSumB = runSumB * factor + weight;
                    runMaxB = newMax;
                    if (factor == 1.0f) {
                        for (uint j = 0; j < 4u; ++j) { osB[j] = osB[j] + weight * vv[j]; }
                    } else {
                        for (uint j = 0; j < 4u; ++j) { osB[j] = osB[j] * factor + weight * vv[j]; }
                    }
                }
            }
        }

        // Combine across the 32 simdgroups and write, head A then head B
        // (threadgroup scratch reused sequentially; barriers make this
        // safe and keep threadgroup memory footprint == the single-head
        // kernel's).
        uint qOffA = qHeadA * head_dim;
        uint qOffB = qHeadB * head_dim;
        for (uint headSel = 0u; headSel < 2u; ++headSel) {
            float runMax = headSel == 0u ? runMaxA : runMaxB;
            float runSum = headSel == 0u ? runSumA : runSumB;
            thread float *os = headSel == 0u ? osA : osB;
            uint qOff = headSel == 0u ? qOffA : qOffB;

            if (lane == 0u) {
                tg_max[sg] = runMax;
                tg_sum[sg] = runSum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float localMax = (lane < n_simd) ? tg_max[lane] : -INFINITY;
                float gMax = simd_max(localMax);
                float localSum = (lane < n_simd) ? (tg_sum[lane] * exp(localMax - gMax)) : 0.0f;
                float gSum = simd_sum(localSum);
                if (lane == 0u) {
                    tg_max[0] = gMax;
                    tg_sum[0] = gSum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float gMax = tg_max[0];
            float gSum = tg_sum[0];
            float rescale = gSum > 0.0f ? (exp(runMax - gMax) / gSum) : 0.0f;
            uint stride = n_simd + 1u;
            uint idx = lane * stride + sg;
            tg_out0[idx] = os[0] * rescale;
            tg_out1[idx] = os[1] * rescale;
            tg_out2[idx] = os[2] * rescale;
            tg_out3[idx] = os[3] * rescale;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float so0 = 0.0f, so1 = 0.0f, so2 = 0.0f, so3 = 0.0f;
                uint rowBase = lane * stride;
                for (uint i = 0u; i < n_simd; ++i) {
                    uint ri = rowBase + i;
                    so0 += tg_out0[ri];
                    so1 += tg_out1[ri];
                    so2 += tg_out2[ri];
                    so3 += tg_out3[ri];
                }
                uint outOff = qOff + d0;
                out[outOff] = bfloat(so0);
                out[outOff + 1u] = bfloat(so1);
                out[outOff + 2u] = bfloat(so2);
                out[outOff + 3u] = bfloat(so3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    """

    /// Fused GQA-paired decode SDPA. Declines (`nil`) unless
    /// `canFuse(...)` holds; caller falls back to `Ops.sdpaDecode`.
    /// Same call contract as `Ops.sdpaDecode`: `q` is `[nQHeads, headDim]`,
    /// `k`/`v` are `[nKVHeads, kvStride, headDim]`, output is
    /// `[nQHeads, headDim]`.
    static func apply(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        nKV: Int, kvStride: Int, scale: Float,
        on cmd: MTLCommandBuffer, device: Device,
        sinkEnd: Int = 0, windowStart: Int = 0,
        into out: Tensor? = nil
    ) -> Tensor? {
        guard canFuse(headDim: headDim, nHeads: nQHeads, nKVHeads: nKVHeads, dtype: q.dtype),
            q.dtype == .bf16, k.dtype == .bf16, v.dtype == .bf16,
            q.elementCount == nQHeads * headDim,
            nKV >= 0, nKV <= kvStride
        else { return nil }

        let result = out ?? Tensor.empty(shape: [nQHeads, headDim], dtype: .bf16, device: device)
        let p = pso(
            "laguna_sdpa_decode_gqa2_bf16_128", source: source, device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(p)
        enc.setBuffer(q.buffer, offset: q.offset, index: 0)
        enc.setBuffer(k.buffer, offset: k.offset, index: 1)
        enc.setBuffer(v.buffer, offset: v.offset, index: 2)
        enc.setBuffer(result.buffer, offset: result.offset, index: 3)
        var nKvU = UInt32(nKV)
        var kvStrideU = UInt32(kvStride)
        var headsPerGroupU = UInt32(nQHeads / nKVHeads)
        var sinkEndU = UInt32(sinkEnd)
        var windowStartU = UInt32(windowStart)
        var scaleV = scale
        enc.setBytes(&nKvU, length: 4, index: 4)
        enc.setBytes(&kvStrideU, length: 4, index: 5)
        enc.setBytes(&headsPerGroupU, length: 4, index: 6)
        enc.setBytes(&sinkEndU, length: 4, index: 7)
        enc.setBytes(&windowStartU, length: 4, index: 8)
        enc.setBytes(&scaleV, length: 4, index: 9)
        let pairs = nQHeads / 2
        enc.dispatchThreadgroups(
            MTLSize(width: pairs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        enc.endEncoding()
        return result
    }
}
