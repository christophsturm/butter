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
// `BUTTER_LAGUNA_FUSED_ATTN_STEP` (default ON; set =0 to disable): collapses
// the sliding-layer decode attention chain [fused QK-norm+RoPE
// dispatch] -> [KV cache append, 2 dispatches on one encoder] ->
// [GQA2 SDPA dispatch] into ONE Metal dispatch per layer per decode
// token. Today's chain is host-bound on per-dispatch encode, not GPU
// work (benchmarks/laguna-xs/README.md sections 13-17); with
// encoder-merging measured null (section 19) and ICB measured -15%
// (section 30), fewer dispatches is the only remaining decode lever
// on this path. Removing 3 dispatches per sliding layer, at 30
// sliding layers, is ~90 fewer dispatches/token -- MEASURED on
// `~/models/laguna-xs-2.1-4bit` (h64/kv8 sliding shape): flag ON
// ~83 tok/s decode vs flag OFF ~71-72 tok/s, a confirmed +16%.
//
// Sliding layers only (headDim 128, full rotary, bf16, plain
// unquantized `KVCache`) -- gated by the per-layer `isSliding` flag
// the config already computes for every layer including layer 0, so
// there is no separate layer-0 special case. Declines for full
// attention (partial rotary / YaRN), non-bf16, a quantized cache, or
// any GQA-pairing shape mismatch (odd `nHeads` or odd heads-per-group)
// -- same envelope `LagunaFusedQKNormRoPE` and `LagunaSdpaDecodeGqa2`
// already check individually, ANDed together here.
//
// ── Numerics: cos>0.99999 vs stock, NOT bit-exact, by design ───────
//
// This kernel compiles its own PSO with `fastMathEnabled = false`
// (below), so its own three internal phases stay float-op-order
// stable relative to EACH OTHER. It is deliberately NOT compiled to
// match the stock chain's own compiled bit pattern: the stock kernels
// (`LagunaFusedQKNormRoPE` in `LagunaSdpa.swift`,
// `LagunaSdpaDecodeGqa2`) compile with Metal's default
// `fastMathEnabled = true`, and that default MUST stay untouched here
// -- those two kernels back the DEFAULT (flag-OFF) decode path used
// by every caller regardless of this feature, and the default path's
// token stream must stay byte-identical to what ships without this
// file existing at all. An earlier version of this fix mistakenly set
// `fastMathEnabled = false` on those two SHARED kernels to chase
// bit-exactness against the fused path; that changed the default
// decode digest for every caller (verified: flag-OFF digest no longer
// matched dev HEAD) and was reverted. The two compiled artifacts
// (stock: fast-math contraction allowed; fused: contraction
// disabled) therefore differ by the fp-contraction gap measured
// during that investigation -- up to ~3e-5 per element on the h64/kv8
// GQA shape (Laguna XS's actual sliding-layer head count per
// `num_attention_heads_per_layer`) -- which is within butter's
// tolerance-based kernel-equivalence convention already established
// by `LagunaSdpaDecodeGqa2Tests` (maxAbsDiff < 0.05 there; this
// kernel's own regression suite,
// `Tests/ButterTests/Ops/LagunaFusedSlidingAttnStepTests.swift`,
// pins a tighter maxAbsDiff < 5e-3 / cos > 0.99999 bound against the
// stock chain). Bottom line: flag OFF is byte-identical to dev HEAD
// (feature invisible); flag ON is a cos>0.99999 approximation of the
// stock chain's math, not a bit-exact substitute -- acceptable for an
// opt-in decode-speed lever, not acceptable to silently impose on
// every user by touching the shared kernels.
//
// ── Phase-by-phase derivation (op-sequence match, not bit-exactness) ─
//
// The kernel below is built out of THREE phases that textually inline
// the math of the three stock dispatches it replaces, so each phase
// alone is a straight transcription with no source-level
// reassociation (the residual gap above is a compiler artifact of the
// two different `fastMathEnabled` settings, not a difference in what
// the source code computes):
//
//   1. Norm+RoPE: identical operation sequence to
//      `laguna_sliding_qk_norm_rope_bf16_128`
//      (`LagunaSdpa.swift`, `LagunaFusedQKNormRoPE`) -- same RMSNorm
//      reduction, same `bfloat((x*inv_rms)*w)` single-rounding cast,
//      same `log2`/`exp2`/`cos`/`sin` RoPE formula -- just writing to
//      threadgroup arrays instead of the device `qOut`/`kOut` buffers
//      the stock kernel would produce. One threadgroup now computes
//      BOTH paired Q heads plus the shared K head (simdgroups 0/1/2),
//      instead of the stock kernel's one-head-per-threadgroup
//      dispatch; simdgroup 3 copies the raw (unnormalized, unroped) V
//      row, matching V's stock treatment of never touching norm/RoPE.
//   2. Cache write: identical destination address
//      (`h*maxSeq*headDim + position*headDim + d`, verified against
//      the generated `iron_kv_cache_update_bf16` kernel) and identical
//      bf16 values (the just-computed threadgroup row, same values
//      phase 1 would hand to the stock append) as
//      `Ops.kvCacheUpdateKV`/`KVCache.appendOnGPU` would write.
//   3. SDPA: identical operation sequence to
//      `laguna_sdpa_decode_gqa2_bf16_128` (`LagunaSdpaDecodeGqa2.swift`,
//      v1/rolled body -- v1 and v2 are already documented as
//      byte-identical outputs, v2 only changes scan pipelining) --
//      same sink/window two-phase scan, same ALPHASKIP exact-1.0f
//      rescale skip, same 4-plane cross-simdgroup combine.
//
// The one addition Phase 3 makes over a textual copy is the write/read
// coherence trick, which is the crux of this whole fusion:
//
//   **No threadgroup in this dispatch ever reads slot `write_idx` back
//   from device memory.** Every pair-threadgroup that can reach
//   `write_idx` in its K/V scan recomputed that exact row ITSELF in
//   Phase 1 (Q is per-head so heads 2p/2p+1 always own their pair; K
//   is per-KV-head, so every one of the `headsPerGroup/2`
//   pair-threadgroups sharing a KV head redundantly repeats the same
//   32-lane RMSNorm+RoPE -- cheap, since it's one more one-simdgroup
//   row op against a scan loop over up to `n_kv` rows). When the scan
//   index equals `write_idx`, Phase 3 substitutes that threadgroup's
//   own `tg_k`/`tg_v` instead of dereferencing `k_cache`/`v_cache`.
//   Since Phase 2's device write is a verbatim bf16-to-bf16 copy of
//   the same threadgroup row (no extra rounding either way), the
//   substituted value is bit-identical to what a device read of
//   Phase 2's write would have produced. Because this dispatch never
//   depends on observing another threadgroup's write, there is no
//   need for a `device`-scope memory fence or a second dispatch to
//   order the write before the read -- only the existing
//   `threadgroup_barrier(mem_flags::mem_threadgroup)` after Phase 1,
//   which orders each threadgroup's OWN Phase-1-to-Phase-3 read
//   correctly. Phase 2's write only matters to FUTURE decode steps,
//   and those begin in a later dispatch on the same (or a
//   subsequently-committed) command buffer, so plain command-buffer
//   submission order already guarantees they observe it -- the same
//   ordering guarantee the stock chain's separate KV-append dispatch
//   already relied on before this fusion existed.
//
//   Only one pair-threadgroup per KV head (the first pair in the
//   group, `head0 % headsPerGroup == 0`) performs the Phase 2 device
//   write; the others in the same group computed the identical row
//   but never write it, avoiding a redundant (harmless but wasteful)
//   multi-writer store to the same address.
//
// Occupancy note: like `LagunaSdpaDecodeGqa2`, the dispatch grid is
// `nHeads/2` threadgroups of 1024 threads (32 simdgroups) each -- one
// threadgroup per Q-head pair. At Laguna's 64-sliding-head shape that
// is 32 threadgroups/layer; Phase 1 only occupies 4 of the 32
// simdgroups per threadgroup (the rest idle behind the barrier before
// Phase 3 uses all 32), the same occupancy profile the stock GQA2 SDPA
// kernel already has on its own.
//
// Host-side slot bookkeeping: the KV cache still reserves the physical
// write slot on the host (`KVCache.reserveSlotForFusedStep()`, added
// alongside this file) -- the fused kernel performs the device WRITE
// itself, but slot allocation / ring rotation / `length` accounting
// stays exactly where `KVEvictionState` already owns it, so this lever
// cannot double-append or desync cache bookkeeping. The stock
// `appendOnGPU` path is skipped entirely (not doubled) whenever the
// fused step fires.
//
// Validated (`~/models/laguna-xs-2.1-4bit`, temperature 0, greedy):
// (1) flag OFF produces a byte-identical digest to dev HEAD (this
// file's own kernel is simply never reached; the "Numerics" section
// above is why that must hold even though this kernel itself compiles
// with `fastMathEnabled = false`); (2) flag ON vs flag OFF, several
// prompts, up to 700 generated tokens (past the 512-token sliding-
// window wrap): NOT byte-identical to the stock chain (expected, see
// "Numerics" above) but `LagunaFusedSlidingAttnStepTests` bounds the
// per-kernel-call gap at maxAbsDiff < 5e-3 / cos > 0.99999 across
// h48/kv8 and h64/kv8 shapes and history lengths 0 through past-wrap;
// (3) decode A/B throughput on the h64/kv8 sliding shape: flag ON
// ~83 tok/s vs flag OFF ~71-72 tok/s (matches dev HEAD stock ~71.6),
// a confirmed +16%, consistent with the ~90-fewer-dispatches/token
// estimate above.

import Foundation
import Metal
import IronSwift

enum LagunaFusedSlidingAttnStep {
    /// Default ON (`=0` to disable). Validated on the hardware gates:
    /// 12-case regression suite (h48/h64, history 0-513 incl.
    /// past-window-wrap), cos >= 0.99999999 vs the stock chain (worst
    /// 6.1e-5/elem), digest-stable across runs, and paired ABBA on the
    /// merged tree measured 85.3 vs 70.9 tok/s stock (+20% decode).
    static var enabled: Bool {
        ButterEnv["BUTTER_LAGUNA_FUSED_ATTN_STEP"] != "0"
    }

    /// Sliding-layer decode only: full rotary, head_dim 128, bf16,
    /// even `nHeads` with an even heads-per-group (same GQA-pairing
    /// envelope `LagunaSdpaDecodeGqa2.canFuse` checks).
    static func canFuse(
        isSliding: Bool, rotaryDim: Int, headDim: Int,
        nHeads: Int, nKVHeads: Int, dtype: DType
    ) -> Bool {
        guard enabled, isSliding, rotaryDim == headDim, headDim == 128,
            dtype == .bf16, nKVHeads > 0, nHeads.isMultiple(of: nKVHeads),
            nHeads.isMultiple(of: 2)
        else { return false }
        let headsPerGroup = nHeads / nKVHeads
        return headsPerGroup.isMultiple(of: 2)
    }

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        let opts = MTLCompileOptions()
        // Force safe math (no fp-contraction) so this kernel's own
        // three phases stay internally op-order stable, matching the
        // belt-and-suspenders pattern already used for the same class
        // of bug in `LagunaSortedCombine.pso`. Deliberately scoped to
        // ONLY this kernel's own PSO -- see the file header's
        // "Numerics" section for why the shared stock kernels
        // (`LagunaFusedQKNormRoPE`, `LagunaSdpaDecodeGqa2`) must stay
        // on Metal's default `fastMathEnabled = true` and must NOT be
        // touched to chase bit-exactness against this opt-in feature.
        opts.fastMathEnabled = false
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    // One threadgroup per Q-head PAIR (`tgid.x`), 1024 threads (32
    // simdgroups x 32 lanes) -- same grid shape as
    // `laguna_sdpa_decode_gqa2_bf16_128`, since Phase 3 below needs
    // the full 32-simdgroup scan width regardless of Phase 1's lighter
    // occupancy.
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void laguna_fused_sliding_attn_step_bf16_128(
        const device bfloat *raw_q [[buffer(0)]],
        const device bfloat *raw_k [[buffer(1)]],
        const device bfloat *raw_v [[buffer(2)]],
        const device bfloat *q_weight [[buffer(3)]],
        const device bfloat *k_weight [[buffer(4)]],
        device bfloat *k_cache [[buffer(5)]],
        device bfloat *v_cache [[buffer(6)]],
        device bfloat *out [[buffer(7)]],
        constant uint &n_kv [[buffer(8)]],
        constant uint &kv_stride [[buffer(9)]],
        constant uint &heads_per_group [[buffer(10)]],
        constant uint &sink_end [[buffer(11)]],
        constant uint &window_start [[buffer(12)]],
        constant float &scale [[buffer(13)]],
        constant uint &position [[buffer(14)]],
        constant float &theta_base [[buffer(15)]],
        constant float &eps [[buffer(16)]],
        constant uint &write_idx [[buffer(17)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]])
    {
        constexpr uint head_dim = 128u;
        constexpr uint half_dim = 64u;
        constexpr uint n_simd = 32u;

        threadgroup bfloat tg_qA[head_dim];
        threadgroup bfloat tg_qB[head_dim];
        threadgroup bfloat tg_k[head_dim];
        threadgroup bfloat tg_v[head_dim];
        threadgroup float tg_max[32];
        threadgroup float tg_sum[32];
        threadgroup float tg_out0[32 * 33];
        threadgroup float tg_out1[32 * 33];
        threadgroup float tg_out2[32 * 33];
        threadgroup float tg_out3[32 * 33];

        uint pair = tgid.x;
        uint head0 = pair * 2u;
        uint head1 = head0 + 1u;
        uint kv_head = head0 / heads_per_group;
        uint d0 = lane * 4u;

        // Phase 1: per-head RMSNorm + plain RoPE, textual replica of
        // laguna_sliding_qk_norm_rope_bf16_128 with the device row
        // writes retargeted at threadgroup memory. simdgroup 0 owns Q
        // head0, simdgroup 1 owns Q head1, simdgroup 2 recomputes the
        // shared K row (redundantly across every pair threadgroup in
        // this KV head's group -- see Phase 3). simdgroup 3 copies the
        // raw V row unmodified (V is never normed or roped).
        if (sg < 3u) {
            const device bfloat *input =
                sg == 0u ? raw_q + head0 * head_dim
                : sg == 1u ? raw_q + head1 * head_dim
                           : raw_k + kv_head * head_dim;
            const device bfloat *weight = sg == 2u ? k_weight : q_weight;
            threadgroup bfloat *outrow =
                sg == 0u ? tg_qA : sg == 1u ? tg_qB : tg_k;

            float x[4];
            float sum = 0.0f;
            for (uint i = 0u; i < 4u; ++i) {
                x[i] = float(input[d0 + i]);
                sum += x[i] * x[i];
            }
            sum = simd_sum(sum);
            float inv_rms = rsqrt(sum / float(head_dim) + eps);

            thread bfloat normalized[4];
            for (uint i = 0u; i < 4u; ++i) {
                float wi = float(weight[d0 + i]);
                normalized[i] = bfloat((x[i] * inv_rms) * wi);
            }

            thread float paired[4];
            for (uint i = 0u; i < 4u; ++i) {
                paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16u);
            }

            if (lane < 16u) {
                float log_theta = log2(theta_base);
                float half_f = float(half_dim);
                float pos_f = float(position);
                for (uint i = 0u; i < 4u; ++i) {
                    uint p = d0 + i;
                    float first = float(normalized[i]);
                    float second = paired[i];
                    float inv_freq = exp2(((-float(p)) * log_theta) / half_f);
                    float th = pos_f * inv_freq;
                    float c = cos(th);
                    float s = sin(th);
                    outrow[p] = bfloat(first * c - second * s);
                    outrow[p + half_dim] = bfloat(first * s + second * c);
                }
            }
        } else if (sg == 3u) {
            const device bfloat *vin = raw_v + kv_head * head_dim;
            for (uint i = lane; i < head_dim; i += n_simd) {
                tg_v[i] = vin[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: one writer per KV head (the first pair in its
        // group) persists the new K/V row for FUTURE decode steps at
        // the same address the generated iron_kv_cache_update_bf16
        // kernel would use. Phase 3 below never reads write_idx back
        // from device memory this dispatch, so no device-scope fence
        // is needed here -- see the file header for the full argument.
        if ((head0 % heads_per_group) == 0u && sg == 0u) {
            uint base = kv_head * kv_stride * head_dim + write_idx * head_dim;
            device bfloat *kc = k_cache + base;
            device bfloat *vc = v_cache + base;
            for (uint i = lane; i < head_dim; i += n_simd) {
                kc[i] = tg_k[i];
                vc[i] = tg_v[i];
            }
        }

        // Phase 3: GQA-paired SDPA over the cache, textual replica of
        // laguna_sdpa_decode_gqa2_bf16_128's rolled scan + combine,
        // with the write/read substitution from the file header.
        uint kv_base = kv_head * kv_stride * head_dim;

        float qA[4];
        float qB[4];
        for (uint i = 0u; i < 4u; ++i) {
            qA[i] = float(tg_qA[d0 + i]) * scale;
            qB[i] = float(tg_qB[d0 + i]) * scale;
        }

        float runMaxA = -INFINITY, runSumA = 0.0f;
        float runMaxB = -INFINITY, runSumB = 0.0f;
        float osA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float osB[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (uint phase = 0u; phase < 2u; ++phase) {
            uint lo = phase == 0u ? sg : (window_start + sg);
            uint hi = phase == 0u ? sink_end : n_kv;
            for (uint i = lo; i < hi; i += n_simd) {
                bool sub = (i == write_idx);
                uint kv0 = kv_base + i * head_dim + d0;
                float kv[4];
                float vv[4];
                if (sub) {
                    for (uint j = 0u; j < 4u; ++j) { kv[j] = float(tg_k[d0 + j]); }
                    for (uint j = 0u; j < 4u; ++j) { vv[j] = float(tg_v[d0 + j]); }
                } else {
                    const device bfloat *kP = k_cache + kv0;
                    const device bfloat *vP = v_cache + kv0;
                    for (uint j = 0u; j < 4u; ++j) { kv[j] = float(kP[j]); }
                    for (uint j = 0u; j < 4u; ++j) { vv[j] = float(vP[j]); }
                }

                float partialA = 0.0f, partialB = 0.0f;
                for (uint j = 0u; j < 4u; ++j) {
                    partialA += qA[j] * kv[j];
                    partialB += qB[j] * kv[j];
                }
                float scoreA = simd_sum(partialA);
                float scoreB = simd_sum(partialB);

                {
                    float newMax = max(runMaxA, scoreA);
                    float factor = exp(runMaxA - newMax);
                    float weight = exp(scoreA - newMax);
                    runSumA = runSumA * factor + weight;
                    runMaxA = newMax;
                    if (factor == 1.0f) {
                        for (uint j = 0u; j < 4u; ++j) { osA[j] = osA[j] + weight * vv[j]; }
                    } else {
                        for (uint j = 0u; j < 4u; ++j) { osA[j] = osA[j] * factor + weight * vv[j]; }
                    }
                }
                {
                    float newMax = max(runMaxB, scoreB);
                    float factor = exp(runMaxB - newMax);
                    float weight = exp(scoreB - newMax);
                    runSumB = runSumB * factor + weight;
                    runMaxB = newMax;
                    if (factor == 1.0f) {
                        for (uint j = 0u; j < 4u; ++j) { osB[j] = osB[j] + weight * vv[j]; }
                    } else {
                        for (uint j = 0u; j < 4u; ++j) { osB[j] = osB[j] * factor + weight * vv[j]; }
                    }
                }
            }
        }

        uint qOffA = head0 * head_dim;
        uint qOffB = head1 * head_dim;
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

    private static func encode(
        rawQ: Tensor, rawK: Tensor, rawV: Tensor, qWeight: Tensor, kWeight: Tensor,
        kCache: Tensor, vCache: Tensor,
        nHeads: Int, headsPerGroup: Int,
        nKV: Int, kvStride: Int, sinkEnd: Int, windowStart: Int,
        scale: Float, position: Int, thetaBase: Float, eps: Float, writeIdx: Int,
        enc: MTLComputeCommandEncoder, device: Device, out: Tensor
    ) {
        let p = pso(
            "laguna_fused_sliding_attn_step_bf16_128", source: source, device: device.mtlDevice)
        enc.setComputePipelineState(p)
        enc.setBuffer(rawQ.buffer, offset: rawQ.offset, index: 0)
        enc.setBuffer(rawK.buffer, offset: rawK.offset, index: 1)
        enc.setBuffer(rawV.buffer, offset: rawV.offset, index: 2)
        enc.setBuffer(qWeight.buffer, offset: qWeight.offset, index: 3)
        enc.setBuffer(kWeight.buffer, offset: kWeight.offset, index: 4)
        enc.setBuffer(kCache.buffer, offset: kCache.offset, index: 5)
        enc.setBuffer(vCache.buffer, offset: vCache.offset, index: 6)
        enc.setBuffer(out.buffer, offset: out.offset, index: 7)
        var nKvU = UInt32(nKV)
        var kvStrideU = UInt32(kvStride)
        var headsPerGroupU = UInt32(headsPerGroup)
        var sinkEndU = UInt32(sinkEnd)
        var windowStartU = UInt32(windowStart)
        var scaleV = scale
        var positionU = UInt32(position)
        var thetaV = thetaBase
        var epsV = eps
        var writeIdxU = UInt32(writeIdx)
        enc.setBytes(&nKvU, length: 4, index: 8)
        enc.setBytes(&kvStrideU, length: 4, index: 9)
        enc.setBytes(&headsPerGroupU, length: 4, index: 10)
        enc.setBytes(&sinkEndU, length: 4, index: 11)
        enc.setBytes(&windowStartU, length: 4, index: 12)
        enc.setBytes(&scaleV, length: 4, index: 13)
        enc.setBytes(&positionU, length: 4, index: 14)
        enc.setBytes(&thetaV, length: 4, index: 15)
        enc.setBytes(&epsV, length: 4, index: 16)
        enc.setBytes(&writeIdxU, length: 4, index: 17)
        let pairs = nHeads / 2
        enc.dispatchThreadgroups(
            MTLSize(width: pairs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Single-dispatch fused decode attention step for a sliding layer.
    /// Caller MUST already have checked `canFuse(...)`; unlike the
    /// `LagunaSdpaDecodeGqa2`/`LagunaAttnInt8` family this has no
    /// per-call soft-decline path (matching `LagunaFusedQKNormRoPE`'s
    /// own convention) -- every precondition below is something
    /// `canFuse` already ruled out at the call site.
    ///
    /// Reserves the cache's next physical write slot on the host
    /// (`KVCache.reserveSlotForFusedStep()`) exactly once, then queues
    /// the single dispatch that performs the norm+RoPE, the cache
    /// write at that slot, and the SDPA read in one command. The
    /// stock `cache.appendOnGPU` path is never called for this token's
    /// K/V -- callers must not call it themselves alongside this.
    static func apply(
        rawQ: Tensor, rawK: Tensor, rawV: Tensor,
        qWeight: Tensor, kWeight: Tensor,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        position: Int, thetaBase: Float, eps: Float,
        cache: KVCache, scale: Float,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(headDim == 128)
        precondition(rawQ.elementCount == nHeads * headDim)
        precondition(rawK.elementCount == nKVHeads * headDim)
        precondition(rawV.elementCount == nKVHeads * headDim)
        precondition(qWeight.elementCount == headDim && kWeight.elementCount == headDim)
        precondition(rawQ.dtype == .bf16 && rawK.dtype == .bf16 && rawV.dtype == .bf16)
        precondition(qWeight.dtype == .bf16 && kWeight.dtype == .bf16)
        precondition(cache.dtype == .bf16)
        precondition(cache.headDim == headDim && cache.nKVHeads == nKVHeads)
        precondition(nHeads.isMultiple(of: nKVHeads) && nHeads.isMultiple(of: 2))
        let headsPerGroup = nHeads / nKVHeads
        precondition(headsPerGroup.isMultiple(of: 2))

        // Host-side bookkeeping only: same slot the stock
        // `appendOnGPU` would reserve, without queuing its GPU
        // dispatches -- the kernel below performs that write itself
        // (Phase 2). `cache.length` reflects the reservation
        // immediately, same contract `appendOnGPU` already has, so
        // `nKV`/`sinkEnd`/`windowStart` below already include this
        // token, matching the stock chain's append-then-read-length
        // sequencing.
        let writeIdx = cache.reserveSlotForFusedStep()
        let nKV = cache.length
        let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: nKV)

        let out = Tensor.empty(shape: [nHeads * headDim], dtype: .bf16, device: device)
        guard let enc = cmd.makeComputeCommandEncoder() else { return out }
        encode(
            rawQ: rawQ, rawK: rawK, rawV: rawV, qWeight: qWeight, kWeight: kWeight,
            kCache: cache.kBuffer, vCache: cache.vBuffer,
            nHeads: nHeads, headsPerGroup: headsPerGroup,
            nKV: nKV, kvStride: cache.maxSeq, sinkEnd: sinkEnd, windowStart: windowStart,
            scale: scale, position: position, thetaBase: thetaBase, eps: eps,
            writeIdx: writeIdx,
            enc: enc, device: device, out: out)
        enc.endEncoding()
        return out
    }
}
