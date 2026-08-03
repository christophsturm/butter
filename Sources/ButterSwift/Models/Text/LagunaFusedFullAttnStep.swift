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
// `BUTTER_LAGUNA_FUSED_FULL_ATTN_STEP` (default ON; set =0 to disable):
// the full-attention twin of `LagunaFusedSlidingAttnStep`
// (`BUTTER_LAGUNA_FUSED_ATTN_STEP`). Collapses the full-attention-layer
// decode attention chain [RMSNorm + partial YaRN RoPE dispatch] -> [KV
// cache append, 2 dispatches on one encoder] -> [GQA2 SDPA dispatch]
// into ONE Metal dispatch per full-attention layer per decode token,
// same lever as the sliding twin: fewer dispatches on a decode path
// that is host-bound on per-dispatch encode, not GPU work
// (benchmarks/laguna-xs/README.md sections 13-17). On
// `~/models/laguna-xs-2.1-4bit` (40 layers, `layer_types` pattern
// repeats every 4 starting at index 0) there are 10 full-attention
// layers (h48/kv8) -- removing 3 dispatches per full layer is ~30
// fewer dispatches/token on top of the sliding fusion's ~90.
//
// Full-attention layers only (headDim 128, PARTIAL rotary via YaRN,
// bf16, plain unquantized `KVCache`) -- gated by the per-layer
// `isSliding == false` flag the config already computes for every
// layer. Declines for sliding layers (plain full-rotary RoPE, handled
// by the sibling file), non-bf16, a quantized cache, or any
// GQA-pairing shape mismatch -- same envelope
// `LagunaFusedSlidingAttnStep.canFuse` checks for its own layer class.
//
// Opt-OUT (`!= "0"`), matching the sliding flag: validated on the
// hardware gates (18-case suite across both fused files, deterministic
// 300-token digests, paired ABBA 87.65 -> 91.96 tok/s = +4.9% decode on
// top of the sliding fusion). Set =0 to restore the stock chain.
//
// ── Cache eviction: full layers are `.unbounded`, and that's fine ──
//
// Full-attention layers back an `.unbounded` `KVCache` (grow-only;
// `KVEvictionState.reserveNextSlot()` returns `slot = absoluteCount`
// then increments -- no ring wrap, panics past `maxSeq`).
// `KVCacheProtocol.sdpaSinkWindow(nKV:)`'s default implementation
// (`KVCache.swift`) always returns `(sinkEnd: 0, windowStart: 0)` for
// this eviction shape, and the sliding kernel's own Phase-3 two-phase
// scan loop already degenerates correctly to a dense `[0, nKV)` scan
// when given `(0, 0)`: `phase==0`'s loop bound becomes `lo=sg, hi=0`
// (zero iterations), `phase==1`'s becomes `lo=0+sg, hi=n_kv` (the
// entire dense range). So Phase 3 (SDPA scan+combine) and Phase 2
// (cache write) below are copied VERBATIM from the sliding kernel --
// nothing about them is full-attention-specific. Only Phase 1's
// rotation math differs (see below), and the host-side wiring just has
// to call `cache.sdpaSinkWindow(nKV:)` the same way the sliding step
// does for `sinkEnd`/`windowStart` to come out `(0, 0)` automatically.
//
// ── Rotary: PARTIAL YaRN, not plain full-head RoPE ──────────────────
//
// Full-attention layers use YaRN on a partial-rotary subset (see
// `lagunaRopeYaRNPartial` in `LagunaText.swift`, which dispatches
// `IronKernels.iron_rope_yarn_bf16` --
// `Sources/IronSwift/Resources/kernels/iron_rope_yarn_bf16.metal`).
// Only the first `rotaryDim` of each `headDim`-strided head is
// rotated (pairs `(i, i + rotaryDim/2)` for `i in [0, rotaryDim/2)`);
// dims `[rotaryDim, headDim)` are left as the plain RMSNorm output,
// untouched. On `~/models/laguna-xs-2.1-4bit`, `headDim == 128` and
// `rotaryDim == 64` (`partial_rotary_factor: 0.5` in
// `rope_parameters.full_attention`). The RMSNorm reduction itself
// (the `bfloat((x*inv_rms)*w)` single-rounding cast) is identical
// head-wide math to the sliding kernel's Phase 1 -- only the
// post-norm rotation step differs, and the kernel below transcribes
// `iron_rope_yarn_bf16.metal`'s exact generated op order (ramp/blend
// included) rather than a "cleaned up" equivalent, on purpose -- see
// the numerics note below.
//
// ── Numerics: cos>0.99999 vs stock, NOT bit-exact, by design ───────
//
// Same convention as `LagunaFusedSlidingAttnStep`: this kernel
// compiles its own PSO with `fastMathEnabled = false` (below), so its
// own phases stay float-op-order stable relative to EACH OTHER. It is
// deliberately NOT compiled to match the stock chain's own compiled
// bit pattern -- the stock kernels (`Ops.rmsNormRows`,
// `IronKernels.iron_rope_yarn_bf16`, `LagunaSdpaDecodeGqa2`) compile
// with Metal's default `fastMathEnabled = true`, and that default MUST
// stay untouched: those kernels back the DEFAULT (flag-OFF) decode
// path used by every caller regardless of this feature, and the
// default path's token stream must stay byte-identical to what ships
// without this file existing at all. This kernel's own regression
// suite, `Tests/ButterTests/Ops/LagunaFusedFullAttnStepTests.swift`,
// pins maxAbsDiff < 5e-3 / cos > 0.99999 against the stock chain,
// mirroring the sliding suite's bound.
//
// ── Phase-by-phase derivation ───────────────────────────────────────
//
//   1. RMSNorm + partial YaRN RoPE: identical RMSNorm reduction /
//      single-rounding cast to the sliding kernel's Phase 1, but the
//      post-norm rotation is `iron_rope_yarn_bf16.metal`'s op
//      sequence restricted to lanes that own a rotary pair. With
//      `d0 = lane*4` (each lane owns 4 contiguous head elements),
//      lanes `[0, rotaryDim/8)` own a rotary pair across their whole
//      4-element group and WRITE both halves (`p` and `p +
//      rotaryDim/2`); lanes `[rotaryDim/8, rotaryDim/4)` hold the
//      second half of a pair already written by their partner lane
//      (`lane ^ (rotaryDim/8)`) and do not write; lanes
//      `[rotaryDim/4, 32)` are entirely outside `rotaryDim` and pass
//      their normalized value straight through unrotated. This is why
//      `canFuse` requires `rotaryDim.isMultiple(of: 8)`: `d0`'s
//      4-element lane group must not straddle the rotary/passthrough
//      boundary (`rotaryDim/2`) or the pair-partner offset
//      (`rotaryDim/4`), i.e. both must be exact multiples of 4 with no
//      remainder inside one lane's group. `rotaryDim == 64` (this
//      checkpoint's shape) satisfies this trivially (64/8 == 8); a
//      future config that doesn't will just decline and fall back to
//      the stock chain, same soft-decline convention as every other
//      `canFuse` in this codebase.
//   2. Cache write: BYTE-IDENTICAL to the sliding kernel's Phase 2 --
//      same destination address, same bf16 values.
//   3. SDPA: BYTE-IDENTICAL to the sliding kernel's Phase 3 -- same
//      write/read substitution trick (no threadgroup in this dispatch
//      ever reads `write_idx` back from device memory; see the sliding
//      file's header for the full argument), same sink/window
//      two-phase scan (which degenerates to a dense full scan here, as
//      explained above), same ALPHASKIP rescale, same 4-plane
//      cross-simdgroup combine.
//
// Host-side slot bookkeeping matches the sliding file exactly:
// `KVCache.reserveSlotForFusedStep()` reserves the physical write slot
// on the host: the fused kernel performs the device WRITE itself, but
// slot allocation / `length` accounting stays exactly where
// `KVEvictionState` already owns it (here, grow-only, no ring). The
// stock `appendOnGPU` path is skipped entirely whenever the fused step
// fires.
//
// Validated (`~/models/laguna-xs-2.1-4bit`, temperature 0, greedy,
// h48/kv8 -- this checkpoint's 10 full-attention layers out of 40):
// (1) flag OFF (unset) produces a token stream byte-identical to a
// clean `origin/dev` @ c357840c build (this file's kernel is simply
// never reached -- only wall-clock timing lines differed in the
// `diff`); (2) flag ON vs flag ON, two full 300-token greedy runs:
// STABLE (byte-identical token stream both times), and the generated
// text is coherent; (3) `LagunaFusedFullAttnStepTests` (6/6 passing)
// bounds the per-kernel-call gap at maxAbsDiff < 5e-3 / cos > 0.99999
// across history lengths 0 through 1000 (well past any sliding-window
// size, proving this path has no window truncation bug), plus decline
// checks (unset/`=0`/sliding-layer/non-multiple-of-8-rotaryDim all
// correctly decline) and a same-input repeatability check; the sibling
// `LagunaFusedSlidingAttnStepTests` suite (12/12) stays green,
// unaffected; (4) ABBA decode throughput, 12 runs (6 A / 6 B,
// ABBA-BAAB block order, "Write a rust fib function", 300 tokens,
// steady-state tok/s from tokens 11+) -- A = default (sliding fusion
// only, this flag off), B = this flag on (sliding + full-attention
// fusion both on): A = [87.45, 87.61, 87.88, 87.88, 88.37, 86.72]
// tok/s, mean 87.65; B = [92.56, 92.18, 90.36, 93.24, 92.03, 91.39]
// tok/s, mean 91.96 -- a +4.9% decode win, above the +1.5% ship gate.
// See `benchmarks/laguna-xs/README.md`'s "Fused single-dispatch decode
// attention step (full-attention layers)" entry for the full writeup.

import Foundation
import Metal
import IronSwift

enum LagunaFusedFullAttnStep {
    /// Opt-out (`!= "0"`), same contract as the sliding flag. The
    /// hardware gate below (mppAutoCapable) still declines on
    /// non-target devices regardless of this flag.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_FUSED_FULL_ATTN_STEP"] != "0"
    }

    /// Full-attention-layer decode only: partial rotary via YaRN,
    /// head_dim 128, bf16, even `nHeads` with an even heads-per-group
    /// (same GQA-pairing envelope the sliding twin checks).
    /// `rotaryDim.isMultiple(of: 8)` is required by the Phase-1
    /// lane-partition math -- see the file header's phase-1 derivation
    /// for why the rotary/passthrough boundary and pair-shuffle offset
    /// must land on 4-element lane boundaries with no remainder.
    static func canFuse(
        isSliding: Bool, rotaryDim: Int, headDim: Int,
        nHeads: Int, nKVHeads: Int, dtype: DType
    ) -> Bool {
        // Hardware gate: same repo-wide capability check the MPP kernels
        // use. On the CI paravirt GPU this kernel showed a single-element
        // output discrepancy at history <= 1 (maxAbsDiff ~1.0, cos 1.0)
        // that does not reproduce on real Apple Silicon, where 300-token
        // greedy streams are deterministic and digest-stable. Root cause
        // on paravirt not yet isolated; the fusion targets current
        // hardware, so decline (stock chain) anywhere the MPP gate says
        // the device is not a known-good target. Revisit if the feature
        // should ever enable on older or virtualized devices.
        guard Ops.mppAutoCapable else { return false }
        guard enabled, !isSliding, headDim == 128,
            rotaryDim > 0, rotaryDim <= headDim, rotaryDim.isMultiple(of: 8),
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
        // phases stay internally op-order stable -- same
        // belt-and-suspenders pattern as `LagunaFusedSlidingAttnStep`.
        // Deliberately scoped to ONLY this kernel's own PSO -- see the
        // file header's "Numerics" section for why the shared stock
        // kernels (`Ops.rmsNormRows`, `IronKernels.iron_rope_yarn_bf16`,
        // `LagunaSdpaDecodeGqa2`) must stay on Metal's default
        // `fastMathEnabled = true` and must NOT be touched to chase
        // bit-exactness against this opt-in feature.
        opts.fastMathEnabled = false
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    // One threadgroup per Q-head PAIR (`tgid.x`), 1024 threads (32
    // simdgroups x 32 lanes) -- same grid shape as
    // `laguna_fused_sliding_attn_step_bf16_128` / the stock
    // `laguna_sdpa_decode_gqa2_bf16_128`.
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void laguna_fused_full_attn_step_bf16_128(
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
        constant uint &rotary_dim [[buffer(18)]],
        constant float &yarn_factor [[buffer(19)]],
        constant float &yarn_low [[buffer(20)]],
        constant float &yarn_high [[buffer(21)]],
        constant float &yarn_attn_factor [[buffer(22)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]])
    {
        constexpr uint head_dim = 128u;
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

        uint half_rotary = rotary_dim / 2u;
        uint quarter_lanes = rotary_dim / 8u;
        uint half_dim_lanes = rotary_dim / 4u;

        // Phase 1: per-head RMSNorm + partial YaRN RoPE, textual replica of
        // iron_rope_yarn_bf16.metal (fed by the same single-rounding RMSNorm
        // cast Phase 1 of the sliding twin uses), with device row writes
        // retargeted at threadgroup memory. simdgroup 0 owns Q head0,
        // simdgroup 1 owns Q head1, simdgroup 2 recomputes the shared K row
        // (redundantly across every pair threadgroup sharing this KV head --
        // see the sliding file's Phase-3 substitution argument, unchanged
        // here), simdgroup 3 copies the raw V row.
        //
        // Lanes [0, quarter_lanes) own a rotary pair and WRITE both halves
        // (p and p+half_rotary). Lanes [quarter_lanes, half_dim_lanes) hold
        // the second half of a pair already written by their partner lane
        // and do not write. Lanes [half_dim_lanes, 32) are entirely outside
        // rotary_dim and pass their normalized value straight through
        // unrotated. All 32 lanes call simd_shuffle uniformly (required for
        // simdgroup-scope permute ops) even though only the first group uses
        // the result.
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
                paired[i] = simd_shuffle(float(normalized[i]), lane ^ quarter_lanes);
            }

            if (lane < quarter_lanes) {
                float log_theta = log2(theta_base);
                float half_f = float(half_rotary);
                float pos_f = float(position);
                for (uint i = 0u; i < 4u; ++i) {
                    uint p = d0 + i;
                    float p_f = float(p);
                    float inv_freq_extrap = exp2(((-p_f) * log_theta) / half_f);
                    float t = (p_f - yarn_low) / (yarn_high - yarn_low);
                    float ramp_hi = select(t, 1.0f, bool(t > 1.0f));
                    float ramp = select(ramp_hi, 0.0f, bool(0.0f > t));
                    float extrap_term = inv_freq_extrap * (1.0f - ramp);
                    float interp_term = (inv_freq_extrap / yarn_factor) * ramp;
                    float inv_freq = interp_term + extrap_term;
                    float th = pos_f * inv_freq;
                    float c = cos(th) * yarn_attn_factor;
                    float s = sin(th) * yarn_attn_factor;
                    float first = float(normalized[i]);
                    float second = paired[i];
                    outrow[p] = bfloat(first * c - second * s);
                    outrow[p + half_rotary] = bfloat(first * s + second * c);
                }
            } else if (lane >= half_dim_lanes) {
                for (uint i = 0u; i < 4u; ++i) {
                    outrow[d0 + i] = normalized[i];
                }
            }
        } else if (sg == 3u) {
            const device bfloat *vin = raw_v + kv_head * head_dim;
            for (uint i = lane; i < head_dim; i += n_simd) {
                tg_v[i] = vin[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2 + Phase 3: BYTE-IDENTICAL to LagunaFusedSlidingAttnStep's own
        // Phase 2 (cache write) / Phase 3 (GQA-paired SDPA scan+combine) -- see
        // that file for the full derivation, copied verbatim here. sink_end /
        // window_start are always (0, 0) for calls into this kernel (full
        // layers back an `.unbounded` cache; `KVCacheProtocol.sdpaSinkWindow`'s
        // default always returns (0, 0) for it), so phase 0 of the two-phase
        // scan is a no-op and phase 1 is a dense [0, n_kv) scan -- full
        // attention over the grown cache, no window/sink math needed, same
        // dense-path argument the sliding file's KVCache.swift header already
        // makes for ring caches.
        if ((head0 % heads_per_group) == 0u && sg == 0u) {
            uint base = kv_head * kv_stride * head_dim + write_idx * head_dim;
            device bfloat *kc = k_cache + base;
            device bfloat *vc = v_cache + base;
            for (uint i = lane; i < head_dim; i += n_simd) {
                kc[i] = tg_k[i];
                vc[i] = tg_v[i];
            }
        }

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
        rotaryDim: Int, yarn: Ops.RoPEYaRN,
        enc: MTLComputeCommandEncoder, device: Device, out: Tensor
    ) {
        let p = pso(
            "laguna_fused_full_attn_step_bf16_128", source: source, device: device.mtlDevice)
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
        var rotaryDimU = UInt32(rotaryDim)
        var yarnFactorV = yarn.factor
        var yarnLowV = yarn.low
        var yarnHighV = yarn.high
        var yarnAttnFactorV = yarn.attnFactor
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
        enc.setBytes(&rotaryDimU, length: 4, index: 18)
        enc.setBytes(&yarnFactorV, length: 4, index: 19)
        enc.setBytes(&yarnLowV, length: 4, index: 20)
        enc.setBytes(&yarnHighV, length: 4, index: 21)
        enc.setBytes(&yarnAttnFactorV, length: 4, index: 22)
        let pairs = nHeads / 2
        enc.dispatchThreadgroups(
            MTLSize(width: pairs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Single-dispatch fused decode attention step for a full-attention
    /// layer. Caller MUST already have checked `canFuse(...)`; like
    /// `LagunaFusedSlidingAttnStep.apply` this has no per-call
    /// soft-decline path -- every precondition below is something
    /// `canFuse` already ruled out at the call site.
    ///
    /// Reserves the cache's next physical write slot on the host
    /// (`KVCache.reserveSlotForFusedStep()`) exactly once, then queues
    /// the single dispatch that performs the RMSNorm + partial YaRN
    /// RoPE, the cache write at that slot, and the SDPA read in one
    /// command. The stock `cache.appendOnGPU` path is never called for
    /// this token's K/V -- callers must not call it themselves
    /// alongside this.
    static func apply(
        rawQ: Tensor, rawK: Tensor, rawV: Tensor,
        qWeight: Tensor, kWeight: Tensor,
        nHeads: Int, nKVHeads: Int, headDim: Int, rotaryDim: Int,
        position: Int, thetaBase: Float, eps: Float,
        yarn: Ops.RoPEYaRN,
        cache: KVCache, scale: Float,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(headDim == 128)
        precondition(rotaryDim > 0 && rotaryDim <= headDim && rotaryDim.isMultiple(of: 8))
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
            writeIdx: writeIdx, rotaryDim: rotaryDim, yarn: yarn,
            enc: enc, device: device, out: out)
        enc.endEncoding()
        return out
    }
}
