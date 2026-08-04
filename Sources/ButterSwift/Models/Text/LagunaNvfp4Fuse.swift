// Copyright 2026 Tom Turney (@TheTom)
//
// Decode-path NVFP4 fusions (mlxfast DARKBLOOM-style).
//
// Shared expert decode (opt-in): BUTTER_LAGUNA_FUSED_SHARED_SWIGLU=1
//   one QMV emits SwiGLU(gate, up) over fused [gate; up] bank.
//   Opt-in: custom runtime Metal is slower than iron QMV on M5.
//
// Shared expert prefill (default ON): iron QMM on fused [gate; up] bank +
// interleaved SwiGLU. Escape: BUTTER_LAGUNA_SHARED_PREFILL_FUSED_GATEUP=0.
//
// Routed Path B fuses (decode T=1 only):
//   BUTTER_LAGUNA_PATHB_ROUTED_SWIGLU!=0 — gate+up QMV+SwiGLU, 8 experts one
//                                          dispatch (mlxfast FUSED_ROUTED_SWIGLU_QMV).
//                                          Default ON.
//   BUTTER_LAGUNA_PATHB_DOWN_REDUCE!=0   — concurrent 8-slot down QMV +
//                                          BF16 weighted reduce (mlxfast
//                                          FUSED_ROUTED_DOWN_REDUCE). Default ON.
//   BUTTER_LAGUNA_PATHB_NVFP4_FUSED=1     — serial chain8 (SLOW on M5, opt-in)
//   BUTTER_LAGUNA_PATHB_NVFP4_SLOT_FUSE=1 — per-slot swiglu+down (recomputes
//                                          silu per out-row; usually loss)
//   BUTTER_LAGUNA_PATHB_GATEUP_FUSE=1    — older per-row gateup fuse (opt-in;
//                                          superseded by ROUTED_SWIGLU)
//   BUTTER_LAGUNA_GATEUP_SPLIT=1          : T28 nine-slot gate-up dispatch
//                                          split back into routed-8 +
//                                          shared-1 (one concurrent-
//                                          dispatch encoder, shared first).
//                                          Down+residual nine-slot merge
//                                          stays intact either way. Opt-in
//                                          until validated; see the T28
//                                          nine-slot MARK section below.
//
// Kernel packaging:
//   Default-ON Path B kernels ship in `kernels_extra.metallib` (see
//   Resources/kernel_overlays/ + scripts/link-extra-metals.sh). `pso()`
//   prefers IronSwift PSOCache / metallib, falls back to live makeLibrary.
//   Fast half-bitpath qdot also overlays
//   `iron_nvfp4_dequant_gemv_expert_indexed_{bf16,f16,f32}` for stock Path B.

import Foundation
import Metal
import IronSwift

enum LagunaNvfp4Fuse {
    /// Decode shared expert: single NVFP4 QMV + SwiGLU over fused [gate;up].
    /// Opt-in only — runtime Metal loses to iron dual QMV on M5 (~−10%).
    static var sharedSwigluEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_FUSED_SHARED_SWIGLU"] == "1"
    }

    /// Serial all-8-slot fuse (slow on M5). Opt-in only.
    static var pathBDownFusedEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_PATHB_NVFP4_FUSED"] == "1"
    }

    /// Per-slot concurrent swiglu+down fuse. Opt-in (usually a loss at mid=512).
    static var pathBSlotFuseEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_PATHB_NVFP4_SLOT_FUSE"] == "1"
    }

    /// Fused gate+up eidx → SwiGLU mid. Opt-in (runtime Metal; no win vs iron yet).
    static var pathBGateUpFuseEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_PATHB_GATEUP_FUSE"] == "1"
    }

    /// Decode routed down: one dispatch over 8 expert mids → weighted sum.
    /// mlxfast `DARKBLOOM_FUSED_ROUTED_DOWN_REDUCE`. Default ON; escape `=0`.
    /// Constraints: mid=512, hidden=2048, topK=8, bf16, block=16.
    static var pathBDownReduceEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_PATHB_DOWN_REDUCE"] != "0"
    }

    /// Decode routed gate+up QMV + SwiGLU (mlxfast FUSED_ROUTED_SWIGLU_QMV).
    /// One dispatch over 8 expert slots × 128 tiles; reuses input loads for
    /// gate and up. Default ON; escape `BUTTER_LAGUNA_PATHB_ROUTED_SWIGLU=0`.
    /// Constraints: mid=512, hidden=2048, topK=8, bf16, block=16.
    static var pathBRoutedSwigluEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_PATHB_ROUTED_SWIGLU"]
            != "0"
    }

    /// Prefill shared: single iron NVFP4 QMM on fused [gate;up] then interleaved
    /// SwiGLU. Default ON (same iron dequant as two QMMs). Escape `=0`.
    static var sharedPrefillFusedGateUpEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_SHARED_PREFILL_FUSED_GATEUP"]
            != "0"
    }

    /// T28: nine-slot merged MoE dispatch, one threadgroup family owns all
    /// 8 routed + 1 shared expert slots (geometry only, no residual fold,
    /// see T28 README notes). Replaces the routed SwiGLU dispatch plus
    /// separate shared SwiGLU/down dispatches plus the final acc+shared
    /// add with two dispatches (swiglu, down+reduce). Default ON when
    /// eligible; escape `BUTTER_LAGUNA_NINE_SLOT=0`.
    static var nineSlotEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_NINE_SLOT"] != "0"
    }

    /// T28 follow-up: split the nine-slot GATE-UP dispatch only into
    /// routed-8 (needs router top-8 indices) + shared-1 (needs only the
    /// layer input). Down+residual stays the merged nine-slot dispatch
    /// regardless of this flag; see `moeGateUpSplitSwigluQmv` for the
    /// dependency-overlap rationale. Opt-in until validated by the A/B
    /// harness; escape hatch is simply not setting this (nine-slot merged
    /// gate-up, `moeNineSlotSwigluQmv`, stays the default).
    static var gateUpSplitEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_GATEUP_SPLIT"] == "1"
    }

    /// T43: fold the post-MoE residual add into the nine-slot down-reduce
    /// kernel's lane-0 epilogue (`laguna_nine_slot_nvfp4_down_reduce_residual_bf16`),
    /// removing the separate `Ops.add(h2, ffnOut, ...)` dispatch on the
    /// nine-slot decode path. Default ON (`=0` to disable, restoring the
    /// stock two-dispatch chain): validated bit-exact vs stock (element
    /// diff 0/2048, KV digests equal, four-prompt 200-token digests
    /// match) with +6 percent paired-ABBA decode. See
    /// `LagunaNvfp4Fuse.nineSlotDownReduceSource` / `moeNineSlotDownReduce`.
    static var downResidualFoldEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_DOWN_RESIDUAL_FOLD"] != "0"
    }

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static let header = """
    #include <metal_stdlib>
    using namespace metal;

    // BUTTER_NVFP4_FAST_DEQUANT-style E4M3 micro-scale decode via CARRY
    // SIGN-FOLD (unconditional -- proved bit-exact for all 256 byte
    // values, see Tests/ButterTests/Ops/Nvfp4FastDequantTests.swift).
    //
    // `uint(bits) + (bits & 128u)`: when sign(bit7)=0 this is a no-op
    // (adds 0). When sign=1, adding 128 to a byte whose bit7 is already
    // set carries: bit7+bit7 -> bit7 clears, bit8 sets; the low 7 bits
    // (exp:4 + mantissa:3) are untouched (128's low 7 bits are 0). So the
    // 9-bit sum is always `(sign << 8) | (exp:mantissa payload)`.
    // Shifting left by 7 (the half-vs-e4m3 mantissa width delta, 10-3=7)
    // lands: sign at half bit15, the 4 exponent bits at half bits[13:10],
    // the 3 mantissa bits at half bits[9:7]. That is exactly the e4m3
    // (bias 7) value reinterpreted with half's (bias 15) exponent field
    // and top-aligned mantissa -- i.e. `half_value == true_value * 2^-8`
    // for both the normal case (2^(exp-15)*(1+mant/8) vs true
    // 2^(exp-7)*(1+mant/8)) and the subnormal case (exp=0 in both e4m3
    // and half; mant/8*2^-14 vs true mant*2^-9 = mant/8*2^-14*2^8).
    // So `float(as_type<half>(h)) * 256.0f` (2^8) is bit-exact for every
    // (sign, exp, mantissa) combination -- all 256 byte values.
    // Replaces a mask/shift/select ~7-op sequence with 5 ops (AND, ADD,
    // SHL + 2 casts, no branch/exp2/select).
    static inline float iron_nvfp4_e4m3_scale(uchar bits) {
        ushort h = ushort((uint(bits) + (uint(bits) & 128u)) << 7);
        return float(as_type<half>(h)) * 256.0f;
    }

    // Iron-matching E2M1 element from 4-bit nibble (slow path / fallback).
    static inline float iron_nvfp4_e2m1(uint nibble) {
        uint mag_code = nibble & 7u;
        float mag;
        switch (mag_code) {
            case 0u: mag = 0.0f; break;
            case 1u: mag = 0.5f; break;
            case 2u: mag = 1.0f; break;
            case 3u: mag = 1.5f; break;
            case 4u: mag = 2.0f; break;
            case 5u: mag = 3.0f; break;
            case 6u: mag = 4.0f; break;
            default: mag = 6.0f; break;
        }
        return ((nibble & 8u) != 0u) ? -mag : mag;
    }

    // Bit-twiddle E2M1 decode of one u32 pack (8 nibbles),
    // via NIBBLE-SPLIT word-parallel decode (19 -> 13 int ops per pack;
    // see Tests/ButterTests/Ops/Nvfp4FastDequantTests.swift for the
    // exhaustive 16-code equivalence table vs iron_nvfp4_e2m1):
    //   xe = c & 0x0F0F0F0F;  ge = xe | (xe << 3)   -- spread even nibbles
    //   yo = c & 0xF0F0F0F0;  go = yo | (yo >> 3)   -- spread odd  nibbles
    //   p0=(ge<<9)&M  p1=(go<<8)&M  p2=(ge<<1)&M  p3=go&M   M=0x8E008E00
    // `ge`/`go` are each computed ONCE (1 AND + 1 SHL/SHR + 1 OR) and
    // shared across two of the four p0..p3 outputs; the original formula
    // recomputed an independent 2×(AND+SHL/SHR)+OR pair per output (5 ops
    // × 4 = 20, or 19 counting p3's unshifted second term) with no shared
    // work. `float2(as_type<half2>(p))` is bit-exact against
    // iron_nvfp4_e2m1 (mag/sign) for every nibble up to the caller-scale
    // multiply below.
    //
    // NOTE (2^22 scale fold, task 3): this function no longer applies the
    // `*16384.0f` half-domain restore itself -- the caller (`scale`
    // argument) is required to have that factor folded in already. Both
    // wrappers below (`iron_nvfp4_qdot_16`, `iron_nvfp4_pack_dot`) do this
    // ONCE per call instead of 4x per pack (8x per qdot_16, since it
    // covers 2 packs) as the old per-value `* 16384.0f` did. Combined with
    // the E4M3 scale fold above (256 = 2^8), a caller that sources `scale`
    // from `iron_nvfp4_e4m3_scale(...) * 16384.0f` is applying a combined
    // 256 * 16384 = 4,194,304 = 2^22 constant across the two folds -- both
    // foldings are individually exact (power-of-2 float multiplies, no
    // rounding, verified in-range for e4m3's finite domain), so splitting
    // them across the two functions is bit-identical to a single 2^22
    // multiply at the scale site; kept split so the *unrelated* slow
    // scalar fallback path (`iron_nvfp4_e2m1(nibble) * iron_nvfp4_e4m3_scale(...)`,
    // used by e.g. laguna_moe_gateup_swiglu_nvfp4_bf16 below) keeps
    // getting the plain (non-16384-folded) E4M3 magnitude it expects.
    // Composing further with the T23 `kNvfp4StageBias` (2^-14) half-
    // staging fold used by the NAX/NAX64 MPP gather kernels would give
    // 2^22 * 2^-14 = 2^8 -- N/A here since this qdot family never stages
    // through `half` tiles (T23 doesn't touch these call sites), but
    // documented for any future fuse of the two families.
    static inline float iron_nvfp4_pack_dot_fast(
        uint packed, float scale, const thread float* input, uint p_off
    ) {
        float accum = 0.0f;
        const uint xe = packed & 0x0F0F0F0Fu;
        const uint ge = xe | (xe << 3);
        const uint yo = packed & 0xF0F0F0F0u;
        const uint go = yo | (yo >> 3);
        const uint M = 0x8E008E00u;
        const uint p0 = (ge << 9) & M;
        const uint p1 = (go << 8) & M;
        const uint p2 = (ge << 1) & M;
        const uint p3 = go & M;
        const float2 v04 = float2(as_type<half2>(p0));
        const float2 v15 = float2(as_type<half2>(p1));
        const float2 v26 = float2(as_type<half2>(p2));
        const float2 v37 = float2(as_type<half2>(p3));
        // half2 packs even/odd nibbles: (n0,n4), (n1,n5), (n2,n6), (n3,n7)
        accum += v04.x * scale * input[p_off + 0u];
        accum += v15.x * scale * input[p_off + 1u];
        accum += v26.x * scale * input[p_off + 2u];
        accum += v37.x * scale * input[p_off + 3u];
        accum += v04.y * scale * input[p_off + 4u];
        accum += v15.y * scale * input[p_off + 5u];
        accum += v26.y * scale * input[p_off + 6u];
        accum += v37.y * scale * input[p_off + 7u];
        return accum;
    }

    // Dot 16 inputs against 8 bytes of packed E2M1 (16 nibbles) × scale.
    // External contract unchanged: `scale` is the plain e4m3*global value
    // (same as every existing call site already passes). The 2^22-class
    // *16384.0f half-domain-restore fold (task 3) happens ONCE here per
    // 16-element dot instead of 8x inside iron_nvfp4_pack_dot_fast (4x
    // per pack × 2 packs) -- exact (power-of-2 multiply, see that
    // function's header comment for the full equivalence argument).
    static inline float iron_nvfp4_qdot_16(
        const device uchar* weight_bytes,
        const thread float* input,
        float scale
    ) {
        float accum = 0.0f;
        const device uint* packs = (const device uint*)weight_bytes;
        const float scale16k = scale * 16384.0f;
        accum += iron_nvfp4_pack_dot_fast(packs[0], scale16k, input, 0u);
        accum += iron_nvfp4_pack_dot_fast(packs[1], scale16k, input, 8u);
        return accum;
    }

    // Threadgroup-pointer sibling of iron_nvfp4_qdot_16 (single pack).
    // Same external contract (plain e4m3*global `scale`); same *16384.0f
    // fold applied once here instead of inside pack_dot_fast.
    static inline float iron_nvfp4_pack_dot(
        uint packed, float scale, const threadgroup float* inner, uint p_off
    ) {
        // Promote TG pointer values into thread registers for the fast path.
        thread float tmp[8];
        for (uint i = 0; i < 8u; ++i) tmp[i] = inner[p_off + i];
        return iron_nvfp4_pack_dot_fast(packed, scale * 16384.0f, tmp, 0u);
    }
    """

    private static let sharedSwigluSrc = header + """
    // TPG=64 (2 SGs). 128 tiles × 4 rows = 512 mid.
    // buffer(4) = global scale (float constant).
    kernel void laguna_shared_nvfp4_swiglu_qmv(
        device const bfloat *input [[buffer(0)]],
        device const uint *fused_weight [[buffer(1)]],
        device const uchar *fused_scales [[buffer(2)]],
        device bfloat *activated [[buffer(3)]],
        constant float &global_scale [[buffer(4)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
        constexpr uint input_width = 2048u;
        constexpr uint output_width = 512u;
        constexpr uint packed_row_bytes = 1024u;
        constexpr uint scale_row_bytes = 128u;
        constexpr uint block_width = 512u;
        constexpr uint values_per_lane = 16u;

        uint tile = tgid;
        uint first_row = tile * 4u + sgid * 2u;

        thread float gate_result[2] = {0.f, 0.f};
        thread float up_result[2] = {0.f, 0.f};
        thread float input_values[values_per_lane];

        const device uchar* wbytes = (const device uchar*)fused_weight;

        for (uint block = 0; block < input_width; block += block_width) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(
                    input + block + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4u; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4 * i] = float(values[0]);
                input_values[4 * i + 1] = float(values[1]);
                input_values[4 * i + 2] = float(values[2]);
                input_values[4 * i + 3] = float(values[3]);
            }
            for (uint row = 0; row < 2u; ++row) {
                uint gate_row = first_row + row;
                uint up_row = gate_row + output_width;
                const device uchar* gate_weight =
                    wbytes + gate_row * packed_row_bytes + block / 2u + lane * 8u;
                const device uchar* up_weight =
                    wbytes + up_row * packed_row_bytes + block / 2u + lane * 8u;
                const device uchar* gate_scale =
                    fused_scales + gate_row * scale_row_bytes + block / 16u + lane;
                const device uchar* up_scale =
                    fused_scales + up_row * scale_row_bytes + block / 16u + lane;
                float gs = iron_nvfp4_e4m3_scale(gate_scale[0]) * global_scale;
                float us = iron_nvfp4_e4m3_scale(up_scale[0]) * global_scale;
                gate_result[row] += iron_nvfp4_qdot_16(gate_weight, input_values, gs);
                up_result[row] += iron_nvfp4_qdot_16(up_weight, input_values, us);
            }
        }
        for (uint row = 0; row < 2u; ++row) {
            gate_result[row] = simd_sum(gate_result[row]);
            up_result[row] = simd_sum(up_result[row]);
            if (lane == 0) {
                float g = gate_result[row];
                float u = up_result[row];
                float sig = 1.f / (1.f + metal::exp(-g));
                activated[first_row + row] = bfloat(g * sig * u);
            }
        }
    }
    """

    private static let downSwigluAccumSrc = header + """
    // Path B routed: for each out-row TG, over topK=8 slots:
    //   tg_inner[d] = silu(gate_s[d]) * up_s[d]
    //   acc += slot_w[s] * (W_down[expert_s, row] · tg_inner)
    // Collapses swigluMany + 8× eidx down GEMV + scalarFMAChain8.
    // Constraints: in_dim ≤ 768, block_size=16, topK=8, bf16, TPG=64.
    kernel void laguna_moe_down_swiglu_accum_nvfp4_chain8_bf16(
        device const bfloat *gate_0 [[buffer(0)]],
        device const bfloat *up_0 [[buffer(1)]],
        device const bfloat *gate_1 [[buffer(2)]],
        device const bfloat *up_1 [[buffer(3)]],
        device const bfloat *gate_2 [[buffer(4)]],
        device const bfloat *up_2 [[buffer(5)]],
        device const bfloat *gate_3 [[buffer(6)]],
        device const bfloat *up_3 [[buffer(7)]],
        device const bfloat *gate_4 [[buffer(8)]],
        device const bfloat *up_4 [[buffer(9)]],
        device const bfloat *gate_5 [[buffer(10)]],
        device const bfloat *up_5 [[buffer(11)]],
        device const bfloat *gate_6 [[buffer(12)]],
        device const bfloat *up_6 [[buffer(13)]],
        device const bfloat *gate_7 [[buffer(14)]],
        device const bfloat *up_7 [[buffer(15)]],
        device const uint *expert_indices [[buffer(16)]],
        device const bfloat *slot_weights [[buffer(17)]],
        device const uint *weights_stacked [[buffer(18)]],
        device const uchar *scales_stacked [[buffer(19)]],
        device bfloat *output [[buffer(20)]],
        constant uint &in_dim [[buffer(21)]],
        constant uint &out_dim [[buffer(22)]],
        constant uint &block_size [[buffer(23)]],
        constant float &global_scale [[buffer(24)]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]],
        uint lsize [[threads_per_threadgroup]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]
    ) {
        threadgroup float tg_inner[768];
        threadgroup float v_total_sg[32];

        uint n_simd = lsize / 32u;
        if (n_simd == 0u) { return; }

        uint row = tgid;
        uint n_packs = in_dim / 8u;
        uint n_blocks = in_dim / block_size;
        uint packs_per_block = block_size / 8u;

        // Pointer tables for the 8 gate/up pairs.
        const device bfloat *gates[8] = {
            gate_0, gate_1, gate_2, gate_3, gate_4, gate_5, gate_6, gate_7};
        const device bfloat *ups[8] = {
            up_0, up_1, up_2, up_3, up_4, up_5, up_6, up_7};

        float acc = 0.0f;

        for (uint slot = 0u; slot < 8u; ++slot) {
            uint expert = expert_indices[slot];
            float sw = float(slot_weights[slot]);

            // Stage SwiGLU(gate, up) into tg_inner.
            for (uint d = tid; d < in_dim; d += lsize) {
                float g = float(gates[slot][d]);
                float u = float(ups[slot][d]);
                float silu = g / (1.0f + metal::exp(-g));
                tg_inner[d] = silu * u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Down GEMV for this expert's row: W[expert, row, :] · tg_inner.
            uint exp_base_packs = expert * out_dim * n_packs;
            uint exp_base_blocks = expert * out_dim * n_blocks;
            uint row_pack_off = exp_base_packs + row * n_packs;
            uint row_block_off = exp_base_blocks + row * n_blocks;

            for (uint pack_idx = tid; pack_idx < n_packs; pack_idx += lsize) {
                uint blk = pack_idx / packs_per_block;
                float scale =
                    iron_nvfp4_e4m3_scale(scales_stacked[row_block_off + blk])
                    * global_scale;
                uint packed = weights_stacked[row_pack_off + pack_idx];
                uint p_off = pack_idx * 8u;
                acc += sw * iron_nvfp4_pack_dot(packed, scale, tg_inner, p_off);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Reduce across TG.
        float v_total;
        {
            float _sv = simd_sum(acc);
            if (simd_lane == 0) v_total_sg[simd_group] = _sv;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                float _wv = simd_lane < n_simd ? v_total_sg[simd_lane] : 0.0f;
                v_total_sg[0] = simd_sum(_wv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            v_total = v_total_sg[0];
        }
        if (tid == 0u) {
            output[row] = bfloat(v_total);
        }
    }
    """

    /// Prefer precompiled metallib PSO (via IronSwift `PSOCache`) when the
    /// kernel was linked into `kernels.metallib` (Path B routed fuses + fast
    /// qdot). Fall back to live `makeLibrary(source:)` for opt-in soft kernels
    /// that are not yet in the metallib, or when metallib load fails.
    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        // 1) Metallib / Iron PSOCache (no runtime MSL parse).
        if let p = try? PSOCache.shared.pipelineStateThrowing(for: name) {
            psoCache[name] = p
            if ProcessInfo.processInfo.environment["BUTTER_LAGUNA_QDOT_VERBOSE"]
                == "1"
            {
                fputs("[LagunaNvfp4Fuse] metallib PSO: \(name)\n", stderr)
            }
            return p
        }
        // 2) Live compile from embedded source (first-use only; cached).
        let opts = MTLCompileOptions()
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        if ProcessInfo.processInfo.environment["BUTTER_LAGUNA_QDOT_VERBOSE"]
            == "1"
        {
            fputs("[LagunaNvfp4Fuse] live-compile PSO: \(name)\n", stderr)
        }
        return p
    }

    /// Build row-concatenated [gate; up] banks from separate NVFP4 linears.
    static func fuseGateUp(
        gate: Nvfp4Linear, up: Nvfp4Linear, device: Device
    ) -> (weight: Tensor, scales: Tensor, global: Float)? {
        guard gate.outDim == up.outDim, gate.inDim == up.inDim,
            gate.outDim == 512, gate.inDim == 2048,
            gate.weight.shape == [512, 256], up.weight.shape == [512, 256],
            abs(gate.global - up.global) < 1e-6
        else { return nil }
        let mid = gate.outDim
        let packs = gate.inDim / 8
        let groups = gate.inDim / 16
        let wBytes = mid * packs * 4 * 2
        let sBytes = mid * groups * 2
        let wBuf = device.makeBuffer(length: wBytes)
        let sBuf = device.makeBuffer(length: sBytes)
        let gw = gate.weight.buffer.contents().advanced(by: gate.weight.offset)
        let uw = up.weight.buffer.contents().advanced(by: up.weight.offset)
        let gs = gate.scales.buffer.contents().advanced(by: gate.scales.offset)
        let us = up.scales.buffer.contents().advanced(by: up.scales.offset)
        let halfW = mid * packs * 4
        let halfS = mid * groups
        memcpy(wBuf.contents(), gw, halfW)
        memcpy(wBuf.contents().advanced(by: halfW), uw, halfW)
        memcpy(sBuf.contents(), gs, halfS)
        memcpy(sBuf.contents().advanced(by: halfS), us, halfS)
        let weight = Tensor(
            buffer: wBuf, offset: 0, shape: [2 * mid, packs], dtype: .u32)
        let scales = Tensor(
            buffer: sBuf, offset: 0, shape: [2 * mid, groups], dtype: .u8)
        // Fused [gate; up] bank replaces two separate NVFP4 linears for
        // the shared-expert SwiGLU path and is read on every forward
        // pass; pin it like any other persistent weight buffer.
        device.markWeightsResident([wBuf, sBuf])
        return (weight, scales, gate.global)
    }

    /// `activated = SwiGLU(W_gate x, W_up x)` for shared expert, BF16.
    static func sharedSwigluQmv(
        input: Tensor,
        fusedWeight: Tensor,
        fusedScales: Tensor,
        global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        device: Device
    ) -> Tensor {
        precondition(input.dtype == .bf16 && input.elementCount == 2048)
        precondition(fusedWeight.dtype == .u32 && fusedScales.dtype == .u8)
        let out = Tensor.empty(shape: [512], dtype: .bf16, device: device)
        let mtl = device.mtlDevice
        let p = pso(
            "laguna_shared_nvfp4_swiglu_qmv", source: sharedSwigluSrc, device: mtl)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(input.buffer, offset: input.offset, index: 0)
            enc.setBuffer(fusedWeight.buffer, offset: fusedWeight.offset, index: 1)
            enc.setBuffer(fusedScales.buffer, offset: fusedScales.offset, index: 2)
            enc.setBuffer(out.buffer, offset: out.offset, index: 3)
            var g = global
            enc.setBytes(&g, length: 4, index: 4)
            // 128 tiles × 64 threads (2 SGs)
            enc.dispatchThreadgroups(
                MTLSize(width: 128, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
        }
        return out
    }

    private static let swigluGateUpInterleavedSrc = """
    #include <metal_stdlib>
    using namespace metal;
    // gu layout [t, 2*mid]: per token [gate[0..mid), up[0..mid)]
    // out layout [t, mid]: silu(g)*u
    kernel void laguna_swiglu_gateup_interleaved_bf16(
        device const bfloat *gu [[buffer(0)]],
        device bfloat *out [[buffer(1)]],
        constant uint &mid [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
        uint token = i / mid;
        uint j = i - token * mid;
        uint base = token * (mid << 1);
        float g = float(gu[base + j]);
        float u = float(gu[base + mid + j]);
        float sig = 1.f / (1.f + metal::exp(-g));
        out[i] = bfloat(g * sig * u);
    }
    kernel void laguna_swiglu_gateup_interleaved_f16(
        device const half *gu [[buffer(0)]],
        device half *out [[buffer(1)]],
        constant uint &mid [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
        uint token = i / mid;
        uint j = i - token * mid;
        uint base = token * (mid << 1);
        float g = float(gu[base + j]);
        float u = float(gu[base + mid + j]);
        float sig = 1.f / (1.f + metal::exp(-g));
        out[i] = half(g * sig * u);
    }
    kernel void laguna_swiglu_gateup_interleaved_f32(
        device const float *gu [[buffer(0)]],
        device float *out [[buffer(1)]],
        constant uint &mid [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
        uint token = i / mid;
        uint j = i - token * mid;
        uint base = token * (mid << 1);
        float g = gu[base + j];
        float u = gu[base + mid + j];
        float sig = 1.f / (1.f + metal::exp(-g));
        out[i] = g * sig * u;
    }
    """

    /// SwiGLU over fused gate‖up rows: `gu` is `[t, 2*mid]`, `out` is `[t, mid]`.
    /// Used after iron `nvfp4Qmm` on fused [gate; up] weight bank.
    static func swigluGateUpInterleaved(
        gu: Tensor, mid: Int, on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor
    ) {
        precondition(mid > 0)
        precondition(gu.elementCount == out.elementCount * 2)
        precondition(out.elementCount % mid == 0)
        precondition(gu.dtype == out.dtype)
        let n = out.elementCount
        let name: String
        switch gu.dtype {
        case .bf16: name = "laguna_swiglu_gateup_interleaved_bf16"
        case .f16: name = "laguna_swiglu_gateup_interleaved_f16"
        case .f32: name = "laguna_swiglu_gateup_interleaved_f32"
        default: fatalError("swigluGateUpInterleaved: bad dtype \(gu.dtype)")
        }
        let p = pso(name, source: swigluGateUpInterleavedSrc, device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(gu.buffer, offset: gu.offset, index: 0)
            enc.setBuffer(out.buffer, offset: out.offset, index: 1)
            var midV = UInt32(mid)
            enc.setBytes(&midV, length: 4, index: 2)
            let tg = min(n, 256)
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        }
    }

    /// Fused SwiGLU + NVFP4 down eidx GEMV + weighted sum for topK=8.
    /// Serial over slots inside one TG — often slower than concurrent eidx.
    /// Opt-in via `pathBDownFusedEnabled`.
    static func moeDownSwigluAccumChain8(
        gates: [Tensor],
        ups: [Tensor],
        expertIndices: Tensor,
        slotWeights: Tensor,
        downWeight: Tensor,
        downScales: Tensor,
        output: Tensor,
        inDim: Int,
        outDim: Int,
        blockSize: Int = 16,
        global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        device: Device
    ) {
        precondition(gates.count == 8 && ups.count == 8)
        precondition(inDim <= 768 && inDim > 0)
        precondition(blockSize == 16)
        precondition(expertIndices.dtype == .u32 && expertIndices.elementCount == 8)
        precondition(slotWeights.dtype == .bf16 && slotWeights.elementCount == 8)
        precondition(output.dtype == .bf16 && output.elementCount == outDim)
        precondition(downWeight.dtype == .u32 && downScales.dtype == .u8)
        for i in 0 ..< 8 {
            precondition(gates[i].dtype == .bf16 && gates[i].elementCount == inDim)
            precondition(ups[i].dtype == .bf16 && ups[i].elementCount == inDim)
        }

        let mtl = device.mtlDevice
        let p = pso(
            "laguna_moe_down_swiglu_accum_nvfp4_chain8_bf16",
            source: downSwigluAccumSrc, device: mtl)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        for i in 0 ..< 8 {
            enc.setBuffer(gates[i].buffer, offset: gates[i].offset, index: i * 2)
            enc.setBuffer(ups[i].buffer, offset: ups[i].offset, index: i * 2 + 1)
        }
        enc.setBuffer(expertIndices.buffer, offset: expertIndices.offset, index: 16)
        enc.setBuffer(slotWeights.buffer, offset: slotWeights.offset, index: 17)
        enc.setBuffer(downWeight.buffer, offset: downWeight.offset, index: 18)
        enc.setBuffer(downScales.buffer, offset: downScales.offset, index: 19)
        enc.setBuffer(output.buffer, offset: output.offset, index: 20)
        var inD = UInt32(inDim)
        var outD = UInt32(outDim)
        var bs = UInt32(blockSize)
        var g = global
        enc.setBytes(&inD, length: 4, index: 21)
        enc.setBytes(&outD, length: 4, index: 22)
        enc.setBytes(&bs, length: 4, index: 23)
        enc.setBytes(&g, length: 4, index: 24)
        enc.dispatchThreadgroups(
            MTLSize(width: outDim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    private static let slotSwigluDownSrc = header + """
    // One expert slot: SwiGLU(gate,up) in TG mem → NVFP4 down eidx GEMV → out.
    // Concurrent across slots (dispatchType concurrent) to keep occupancy.
    kernel void laguna_moe_slot_swiglu_down_nvfp4_bf16(
        device const bfloat *gate [[buffer(0)]],
        device const bfloat *up [[buffer(1)]],
        device const uint *expert_index [[buffer(2)]],
        device const uint *weights_stacked [[buffer(3)]],
        device const uchar *scales_stacked [[buffer(4)]],
        device bfloat *output [[buffer(5)]],
        constant uint &in_dim [[buffer(6)]],
        constant uint &out_dim [[buffer(7)]],
        constant uint &block_size [[buffer(8)]],
        constant float &global_scale [[buffer(9)]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]],
        uint lsize [[threads_per_threadgroup]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]
    ) {
        threadgroup float tg_inner[768];
        threadgroup float v_total_sg[32];
        uint n_simd = lsize / 32u;
        if (n_simd == 0u) { return; }

        uint row = tgid;
        uint n_packs = in_dim / 8u;
        uint n_blocks = in_dim / block_size;
        uint packs_per_block = block_size / 8u;
        uint expert = expert_index[0];

        for (uint d = tid; d < in_dim; d += lsize) {
            float g = float(gate[d]);
            float u = float(up[d]);
            tg_inner[d] = (g / (1.0f + metal::exp(-g))) * u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float acc = 0.0f;
        uint exp_base_packs = expert * out_dim * n_packs;
        uint exp_base_blocks = expert * out_dim * n_blocks;
        uint row_pack_off = exp_base_packs + row * n_packs;
        uint row_block_off = exp_base_blocks + row * n_blocks;
        for (uint pack_idx = tid; pack_idx < n_packs; pack_idx += lsize) {
            uint blk = pack_idx / packs_per_block;
            float scale =
                iron_nvfp4_e4m3_scale(scales_stacked[row_block_off + blk])
                * global_scale;
            uint packed = weights_stacked[row_pack_off + pack_idx];
            acc += iron_nvfp4_pack_dot(packed, scale, tg_inner, pack_idx * 8u);
        }

        float v_total;
        {
            float _sv = simd_sum(acc);
            if (simd_lane == 0) v_total_sg[simd_group] = _sv;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                float _wv = simd_lane < n_simd ? v_total_sg[simd_lane] : 0.0f;
                v_total_sg[0] = simd_sum(_wv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            v_total = v_total_sg[0];
        }
        if (tid == 0u) {
            output[row] = bfloat(v_total);
        }
    }
    """

    private static let gateUpSwigluSrc = header + """
    // Fused expert-indexed gate+up GEMV → SwiGLU into mid[out_dim].
    // One TG per mid-row; concurrent across topK slots.
    // Layout: weights [E, mid, hidden/8] u32, scales [E, mid, hidden/16] u8.
    kernel void laguna_moe_gateup_swiglu_nvfp4_bf16(
        device const uint *gate_weights [[buffer(0)]],
        device const uchar *gate_scales [[buffer(1)]],
        device const uint *up_weights [[buffer(2)]],
        device const uchar *up_scales [[buffer(3)]],
        device const bfloat *input [[buffer(4)]],
        device const uint *expert_index [[buffer(5)]],
        device bfloat *activated [[buffer(6)]],
        constant uint &in_dim [[buffer(7)]],   // hidden
        constant uint &out_dim [[buffer(8)]],  // mid
        constant uint &block_size [[buffer(9)]],
        constant float &global_scale [[buffer(10)]],
        uint tid [[thread_position_in_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]],
        uint lsize [[threads_per_threadgroup]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]
    ) {
        threadgroup float v_total_sg[32];
        uint n_simd = lsize / 32u;
        if (n_simd == 0u) { return; }

        uint row = tgid;  // mid-row
        uint expert = expert_index[0];
        uint n_packs = in_dim / 8u;
        uint n_blocks = in_dim / block_size;
        uint packs_per_block = block_size / 8u;

        uint exp_base_packs = expert * out_dim * n_packs;
        uint exp_base_blocks = expert * out_dim * n_blocks;
        uint row_pack_off = exp_base_packs + row * n_packs;
        uint row_block_off = exp_base_blocks + row * n_blocks;

        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (uint pack_idx = tid; pack_idx < n_packs; pack_idx += lsize) {
            uint blk = pack_idx / packs_per_block;
            float gscale =
                iron_nvfp4_e4m3_scale(gate_scales[row_block_off + blk]) * global_scale;
            float uscale =
                iron_nvfp4_e4m3_scale(up_scales[row_block_off + blk]) * global_scale;
            uint gpacked = gate_weights[row_pack_off + pack_idx];
            uint upacked = up_weights[row_pack_off + pack_idx];
            uint p_off = pack_idx * 8u;
            for (uint i = 0; i < 8u; ++i) {
                float x = float(input[p_off + i]);
                uint gnib = (gpacked >> (i * 4u)) & 15u;
                uint unib = (upacked >> (i * 4u)) & 15u;
                gate_acc += iron_nvfp4_e2m1(gnib) * gscale * x;
                up_acc += iron_nvfp4_e2m1(unib) * uscale * x;
            }
        }

        // Reduce gate and up independently across the TG.
        float g_total;
        float u_total;
        {
            float _sg = simd_sum(gate_acc);
            float _su = simd_sum(up_acc);
            // Pack two floats into consecutive SG slots: even=gate, odd=up.
            // Simpler: two sequential reductions reusing v_total_sg.
            if (simd_lane == 0) v_total_sg[simd_group] = _sg;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                float _wv = simd_lane < n_simd ? v_total_sg[simd_lane] : 0.0f;
                v_total_sg[0] = simd_sum(_wv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            g_total = v_total_sg[0];

            if (simd_lane == 0) v_total_sg[simd_group] = _su;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                float _wv = simd_lane < n_simd ? v_total_sg[simd_lane] : 0.0f;
                v_total_sg[0] = simd_sum(_wv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            u_total = v_total_sg[0];
        }
        if (tid == 0u) {
            float silu = g_total / (1.0f + metal::exp(-g_total));
            activated[row] = bfloat(silu * u_total);
        }
    }
    """

    /// Concurrent fused gate+up eidx → SwiGLU mid for each topK slot.
    static func moeGateUpSwigluMany(
        gateWeight: Tensor,
        gateScales: Tensor,
        upWeight: Tensor,
        upScales: Tensor,
        input: Tensor,
        expertIndices: [Tensor],
        outputs: [Tensor],  // mid activations per slot
        inDim: Int,         // hidden
        outDim: Int,        // mid
        blockSize: Int = 16,
        global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        device: Device
    ) {
        let n = expertIndices.count
        precondition(n > 0 && outputs.count == n)
        precondition(input.dtype == .bf16 && input.elementCount == inDim)
        precondition(gateWeight.dtype == .u32 && upWeight.dtype == .u32)
        let mtl = device.mtlDevice
        let p = pso(
            "laguna_moe_gateup_swiglu_nvfp4_bf16",
            source: gateUpSwigluSrc, device: mtl)
        let useConcurrent =
            ProcessInfo.processInfo.environment["BUTTER_MOE_SERIAL_EXPERT_GEMV"] == nil
        let enc: MTLComputeCommandEncoder?
        if useConcurrent {
            enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        } else {
            enc = cmd.makeComputeCommandEncoder()
        }
        guard let enc else { return }
        enc.setComputePipelineState(p)
        var inD = UInt32(inDim)
        var outD = UInt32(outDim)
        var bs = UInt32(blockSize)
        var g = global
        for i in 0 ..< n {
            enc.setBuffer(gateWeight.buffer, offset: gateWeight.offset, index: 0)
            enc.setBuffer(gateScales.buffer, offset: gateScales.offset, index: 1)
            enc.setBuffer(upWeight.buffer, offset: upWeight.offset, index: 2)
            enc.setBuffer(upScales.buffer, offset: upScales.offset, index: 3)
            enc.setBuffer(input.buffer, offset: input.offset, index: 4)
            enc.setBuffer(
                expertIndices[i].buffer, offset: expertIndices[i].offset, index: 5)
            enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 6)
            enc.setBytes(&inD, length: 4, index: 7)
            enc.setBytes(&outD, length: 4, index: 8)
            enc.setBytes(&bs, length: 4, index: 9)
            enc.setBytes(&g, length: 4, index: 10)
            // mid rows (outDim=512), TPG=64
            enc.dispatchThreadgroups(
                MTLSize(width: outDim, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    /// Concurrent per-slot SwiGLU+down (skips writing mid to DRAM).
    /// Then caller runs scalarFMAChain8 on the 8 outs.
    static func moeSlotSwigluDownMany(
        gates: [Tensor],
        ups: [Tensor],
        expertIndices: [Tensor],  // each [1] u32
        downWeight: Tensor,
        downScales: Tensor,
        outputs: [Tensor],
        inDim: Int,
        outDim: Int,
        blockSize: Int = 16,
        global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        device: Device
    ) {
        let n = gates.count
        precondition(n > 0 && ups.count == n && expertIndices.count == n
            && outputs.count == n)
        precondition(inDim <= 768 && blockSize == 16)
        let mtl = device.mtlDevice
        let p = pso(
            "laguna_moe_slot_swiglu_down_nvfp4_bf16",
            source: slotSwigluDownSrc, device: mtl)
        let useConcurrent =
            ProcessInfo.processInfo.environment["BUTTER_MOE_SERIAL_EXPERT_GEMV"] == nil
        let enc: MTLComputeCommandEncoder?
        if useConcurrent {
            enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        } else {
            enc = cmd.makeComputeCommandEncoder()
        }
        guard let enc else { return }
        enc.setComputePipelineState(p)
        var inD = UInt32(inDim)
        var outD = UInt32(outDim)
        var bs = UInt32(blockSize)
        var g = global
        for i in 0 ..< n {
            enc.setBuffer(gates[i].buffer, offset: gates[i].offset, index: 0)
            enc.setBuffer(ups[i].buffer, offset: ups[i].offset, index: 1)
            enc.setBuffer(
                expertIndices[i].buffer, offset: expertIndices[i].offset, index: 2)
            enc.setBuffer(downWeight.buffer, offset: downWeight.offset, index: 3)
            enc.setBuffer(downScales.buffer, offset: downScales.offset, index: 4)
            enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 5)
            enc.setBytes(&inD, length: 4, index: 6)
            enc.setBytes(&outD, length: 4, index: 7)
            enc.setBytes(&bs, length: 4, index: 8)
            enc.setBytes(&g, length: 4, index: 9)
            enc.dispatchThreadgroups(
                MTLSize(width: outDim, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    // MARK: - Routed down-reduce (mlxfast FUSED_ROUTED_DOWN_REDUCE)

    /// Concurrent 8-expert down QMV + BF16 weighted reduction.
    ///
    /// One TG = 8 simdgroups (one per topK slot). Each SG dequants its
    /// expert's down row(s) against the slot mid activation; SG0 then
    /// multiplies by router weights (already including `routedScalingFactor`)
    /// and reduces in BF16 order matching `scalarFMAChain8`.
    ///
    /// Hard-shaped for Laguna XS: mid=512, hidden=2048, topK=8, block=16.
    private static let routedDownReduceSrc = header + """
    kernel void laguna_moe_routed_down_reduce_nvfp4_bf16(
        device const bfloat *mid_0 [[buffer(0)]],
        device const bfloat *mid_1 [[buffer(1)]],
        device const bfloat *mid_2 [[buffer(2)]],
        device const bfloat *mid_3 [[buffer(3)]],
        device const bfloat *mid_4 [[buffer(4)]],
        device const bfloat *mid_5 [[buffer(5)]],
        device const bfloat *mid_6 [[buffer(6)]],
        device const bfloat *mid_7 [[buffer(7)]],
        device const uint *indices [[buffer(8)]],
        device const bfloat *slot_weights [[buffer(9)]],
        device const uint *down_weight [[buffer(10)]],
        device const uchar *down_scales [[buffer(11)]],
        device bfloat *output [[buffer(12)]],
        constant float &global_scale [[buffer(13)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]
    ) {
        constexpr uint input_width = 512u;
        constexpr uint experts_per_token = 8u;
        constexpr uint outputs_per_simd = 4u;
        constexpr uint values_per_lane = 16u;
        constexpr uint n_packs = input_width / 8u;       // 64
        constexpr uint n_blocks = input_width / 16u;     // 32
        constexpr uint packs_per_block = 16u / 8u;       // 2
        constexpr uint out_dim = 2048u;

        uint expert_slot = simd_group;  // 0..7
        uint lane = simd_lane;          // 0..31
        uint first_row = tgid * outputs_per_simd;
        if (expert_slot >= experts_per_token || first_row >= out_dim) return;

        // Pick the mid activation for this expert slot.
        const device bfloat* expert_input;
        switch (expert_slot) {
            case 0u: expert_input = mid_0; break;
            case 1u: expert_input = mid_1; break;
            case 2u: expert_input = mid_2; break;
            case 3u: expert_input = mid_3; break;
            case 4u: expert_input = mid_4; break;
            case 5u: expert_input = mid_5; break;
            case 6u: expert_input = mid_6; break;
            default: expert_input = mid_7; break;
        }

        uint expert = indices[expert_slot];
        uint exp_base_packs = expert * out_dim * n_packs;
        uint exp_base_blocks = expert * out_dim * n_blocks;

        // Stage this lane's 16 mid values (covers full mid=512 across 32 lanes).
        thread float input_values[values_per_lane];
        {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(
                    expert_input + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4u; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4u * i] = float(values[0]);
                input_values[4u * i + 1u] = float(values[1]);
                input_values[4u * i + 2u] = float(values[2]);
                input_values[4u * i + 3u] = float(values[3]);
            }
        }

        // Each lane owns packs [lane*2, lane*2+1] (one NVFP4 block = 16 vals).
        uint pack_base = lane * packs_per_block;
        uint blk = pack_base / packs_per_block;  // == lane
        thread float result[outputs_per_simd] = {0.f, 0.f, 0.f, 0.f};
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            uint row_pack_off = exp_base_packs + output_row * n_packs + pack_base;
            uint row_block_off = exp_base_blocks + output_row * n_blocks + blk;
            float scale =
                iron_nvfp4_e4m3_scale(down_scales[row_block_off]) * global_scale;
            // Two u32 packs cover 16 nibbles — fast half bitpath qdot.
            const device uchar* wbytes =
                (const device uchar*)(down_weight + row_pack_off);
            result[row] = iron_nvfp4_qdot_16(wbytes, input_values, scale);
            result[row] = simd_sum(result[row]);
        }

        // Park BF16 expert outputs in TG mem for the weighted reduce.
        threadgroup bfloat expert_outputs[experts_per_token * outputs_per_simd];
        if (lane == 0u) {
            for (uint row = 0; row < outputs_per_simd; ++row) {
                expert_outputs[expert_slot * outputs_per_simd + row] =
                    bfloat(result[row]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // BF16 product + accumulate in slot order 0..7 (matches scalarFMAChain8).
        // Router weights already include routedScalingFactor (applied on host/GPU
        // before this kernel); do NOT re-scale by 2.5 here.
        if (expert_slot == 0u && lane < outputs_per_simd) {
            bfloat total = bfloat(0.0f);
            for (uint slot = 0u; slot < experts_per_token; ++slot) {
                bfloat route_weight = slot_weights[slot];
                bfloat product = bfloat(
                    float(expert_outputs[slot * outputs_per_simd + lane])
                    * float(route_weight));
                total = bfloat(float(product) + float(total));
            }
            output[first_row + lane] = total;
        }
    }
    """

    // MARK: - Routed SwiGLU QMV (mlxfast FUSED_ROUTED_SWIGLU_QMV)

    /// One dispatch: 8 expert slots × 128 tiles → mid activations.
    /// Separate gate/up stacked banks (no fused-bank build). Tile-major
    /// expert interleave for weight-bank latency hiding. Fast half bitpath
    /// qdot (iron-exact).
    private static let routedSwigluSrc = header + """
    kernel void laguna_moe_routed_swiglu_qmv_nvfp4_bf16(
        device const bfloat *input [[buffer(0)]],
        device const uint *gate_weight [[buffer(1)]],
        device const uchar *gate_scales [[buffer(2)]],
        device const uint *up_weight [[buffer(3)]],
        device const uchar *up_scales [[buffer(4)]],
        device const uint *indices [[buffer(5)]],
        device bfloat *activated_0 [[buffer(6)]],
        device bfloat *activated_1 [[buffer(7)]],
        device bfloat *activated_2 [[buffer(8)]],
        device bfloat *activated_3 [[buffer(9)]],
        device bfloat *activated_4 [[buffer(10)]],
        device bfloat *activated_5 [[buffer(11)]],
        device bfloat *activated_6 [[buffer(12)]],
        device bfloat *activated_7 [[buffer(13)]],
        constant float &global_scale [[buffer(14)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        constexpr uint input_width = 2048u;
        constexpr uint output_width = 512u;
        constexpr uint packed_row_bytes = 1024u;   // 2048/2
        constexpr uint scale_row_bytes = 128u;     // 2048/16
        constexpr uint packed_expert_bytes = output_width * packed_row_bytes;
        constexpr uint scale_expert_bytes = output_width * scale_row_bytes;
        constexpr uint block_width = 512u;
        constexpr uint values_per_lane = 16u;
        constexpr uint routed_experts = 8u;

        // Tile-major: interleave expert slots so banks overlap in the wave.
        uint expert_slot = tgid % routed_experts;
        uint tile = tgid / routed_experts;
        uint expert = indices[expert_slot];
        uint first_row = tile * 4u + sgid * 2u;

        const device uchar* g_wbase =
            (const device uchar*)gate_weight + expert * packed_expert_bytes;
        const device uchar* u_wbase =
            (const device uchar*)up_weight + expert * packed_expert_bytes;
        const device uchar* g_sbase =
            gate_scales + expert * scale_expert_bytes;
        const device uchar* u_sbase =
            up_scales + expert * scale_expert_bytes;

        thread float gate_result[2] = {0.f, 0.f};
        thread float up_result[2] = {0.f, 0.f};
        thread float input_values[values_per_lane];

        for (uint block = 0; block < input_width; block += block_width) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(
                    input + block + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4u; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4u * i] = float(values[0]);
                input_values[4u * i + 1u] = float(values[1]);
                input_values[4u * i + 2u] = float(values[2]);
                input_values[4u * i + 3u] = float(values[3]);
            }
            for (uint row = 0; row < 2u; ++row) {
                uint out_row = first_row + row;
                const device uchar* gate_w =
                    g_wbase + out_row * packed_row_bytes
                    + block / 2u + lane * 8u;
                const device uchar* up_w =
                    u_wbase + out_row * packed_row_bytes
                    + block / 2u + lane * 8u;
                const device uchar* gate_s =
                    g_sbase + out_row * scale_row_bytes
                    + block / 16u + lane;
                const device uchar* up_s =
                    u_sbase + out_row * scale_row_bytes
                    + block / 16u + lane;
                float gs = iron_nvfp4_e4m3_scale(gate_s[0]) * global_scale;
                float us = iron_nvfp4_e4m3_scale(up_s[0]) * global_scale;
                gate_result[row] += iron_nvfp4_qdot_16(gate_w, input_values, gs);
                up_result[row] += iron_nvfp4_qdot_16(up_w, input_values, us);
            }
        }

        // Destination mid buffer for this expert slot.
        device bfloat* activated;
        switch (expert_slot) {
            case 0u: activated = activated_0; break;
            case 1u: activated = activated_1; break;
            case 2u: activated = activated_2; break;
            case 3u: activated = activated_3; break;
            case 4u: activated = activated_4; break;
            case 5u: activated = activated_5; break;
            case 6u: activated = activated_6; break;
            default: activated = activated_7; break;
        }

        for (uint row = 0; row < 2u; ++row) {
            gate_result[row] = simd_sum(gate_result[row]);
            up_result[row] = simd_sum(up_result[row]);
            if (lane == 0u) {
                float g = gate_result[row];
                float u = up_result[row];
                // Match Ops.swiglu: silu(g) = g * sigmoid(g) in float, then *u.
                float sig = 1.f / (1.f + metal::exp(-g));
                activated[first_row + row] = bfloat(g * sig * u);
            }
        }
    }
    """

    /// Fused routed gate+up QMV + SwiGLU → 8 mid buffers (Laguna shapes).
    /// Replaces 16× eidx gate/up GEMV + `swigluMany`.
    static func moeRoutedSwigluQmv(
        input: Tensor,
        gateWeight: Tensor,
        gateScales: Tensor,
        upWeight: Tensor,
        upScales: Tensor,
        expertIndices: Tensor,
        outputs: [Tensor],
        global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        device: Device
    ) {
        precondition(outputs.count == 8)
        precondition(input.dtype == .bf16 && input.elementCount == 2048)
        precondition(expertIndices.dtype == .u32 && expertIndices.elementCount == 8)
        precondition(gateWeight.dtype == .u32 && upWeight.dtype == .u32)
        precondition(gateScales.dtype == .u8 && upScales.dtype == .u8)
        for i in 0 ..< 8 {
            precondition(outputs[i].dtype == .bf16 && outputs[i].elementCount == 512)
        }

        let mtl = device.mtlDevice
        let p = pso(
            "laguna_moe_routed_swiglu_qmv_nvfp4_bf16",
            source: routedSwigluSrc, device: mtl)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(gateWeight.buffer, offset: gateWeight.offset, index: 1)
        enc.setBuffer(gateScales.buffer, offset: gateScales.offset, index: 2)
        enc.setBuffer(upWeight.buffer, offset: upWeight.offset, index: 3)
        enc.setBuffer(upScales.buffer, offset: upScales.offset, index: 4)
        enc.setBuffer(expertIndices.buffer, offset: expertIndices.offset, index: 5)
        for i in 0 ..< 8 {
            enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 6 + i)
        }
        var g = global
        enc.setBytes(&g, length: 4, index: 14)
        // 8 experts × 128 tiles; TPG=64 (2 SGs × 32)
        enc.dispatchThreadgroups(
            MTLSize(width: 8 * 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Fused routed down QMV + weighted sum for topK=8 (Laguna mid=512).
    /// Replaces 8× eidx down GEMV + `scalarFMAChain8`.
    static func moeRoutedDownReduce(
        mids: [Tensor],
        expertIndices: Tensor,
        slotWeights: Tensor,
        downWeight: Tensor,
        downScales: Tensor,
        output: Tensor,
        global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        device: Device
    ) {
        precondition(mids.count == 8)
        precondition(expertIndices.dtype == .u32 && expertIndices.elementCount == 8)
        precondition(slotWeights.dtype == .bf16 && slotWeights.elementCount == 8)
        precondition(output.dtype == .bf16 && output.elementCount == 2048)
        precondition(downWeight.dtype == .u32 && downScales.dtype == .u8)
        for i in 0 ..< 8 {
            precondition(mids[i].dtype == .bf16 && mids[i].elementCount == 512)
        }

        let mtl = device.mtlDevice
        let p = pso(
            "laguna_moe_routed_down_reduce_nvfp4_bf16",
            source: routedDownReduceSrc, device: mtl)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        for i in 0 ..< 8 {
            enc.setBuffer(mids[i].buffer, offset: mids[i].offset, index: i)
        }
        enc.setBuffer(expertIndices.buffer, offset: expertIndices.offset, index: 8)
        enc.setBuffer(slotWeights.buffer, offset: slotWeights.offset, index: 9)
        enc.setBuffer(downWeight.buffer, offset: downWeight.offset, index: 10)
        enc.setBuffer(downScales.buffer, offset: downScales.offset, index: 11)
        enc.setBuffer(output.buffer, offset: output.offset, index: 12)
        var g = global
        enc.setBytes(&g, length: 4, index: 13)
        // 512 tiles × 4 rows = 2048; TPG=256 (8 SGs × 32 lanes)
        enc.dispatchThreadgroups(
            MTLSize(width: 512, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - T28 nine-slot merged MoE dispatch (routed + shared, one family)

    /// Same per-row arithmetic as `routedSwigluSrc` (routed slots) and
    /// `sharedSwigluSrc` (shared slot, slot index 8): both already share the
    /// identical `iron_nvfp4_qdot_16` fast bitpath + `bfloat(g*sig*u)`
    /// rounding, so this is a dispatch-count merge, not a numerics change.
    /// Tile-major, TG=64 (2 SGs, matches both source kernels): 9 slots × 128
    /// tiles = 1152 threadgroups. A 1-row-per-SIMD/256-TG layout was tried
    /// and regressed, so this keeps the 2-row-per-SIMD/1152-TG shape.
    private static let nineSlotSwigluSrc = header + """
    kernel void laguna_nine_slot_nvfp4_swiglu_qmv_bf16(
        device const bfloat *input [[buffer(0)]],
        device const uint *gate_weight [[buffer(1)]],
        device const uchar *gate_scales [[buffer(2)]],
        device const uint *up_weight [[buffer(3)]],
        device const uchar *up_scales [[buffer(4)]],
        device const uint *indices [[buffer(5)]],
        device const uint *shared_weight [[buffer(6)]],
        device const uchar *shared_scales [[buffer(7)]],
        device bfloat *activated_0 [[buffer(8)]],
        device bfloat *activated_1 [[buffer(9)]],
        device bfloat *activated_2 [[buffer(10)]],
        device bfloat *activated_3 [[buffer(11)]],
        device bfloat *activated_4 [[buffer(12)]],
        device bfloat *activated_5 [[buffer(13)]],
        device bfloat *activated_6 [[buffer(14)]],
        device bfloat *activated_7 [[buffer(15)]],
        device bfloat *shared_activated [[buffer(16)]],
        constant float &global_scale [[buffer(17)]],
        constant float &shared_global_scale [[buffer(18)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        constexpr uint input_width = 2048u;
        constexpr uint output_width = 512u;
        constexpr uint packed_row_bytes = 1024u;   // 2048/2
        constexpr uint scale_row_bytes = 128u;     // 2048/16
        constexpr uint packed_expert_bytes = output_width * packed_row_bytes;
        constexpr uint scale_expert_bytes = output_width * scale_row_bytes;
        constexpr uint block_width = 512u;
        constexpr uint values_per_lane = 16u;
        constexpr uint routed_experts = 8u;
        constexpr uint slots = routed_experts + 1u;

        uint expert_slot = tgid % slots;
        uint tile = tgid / slots;
        bool is_routed = expert_slot < routed_experts;
        uint first_row = tile * 4u + sgid * 2u;

        const device uchar* g_wbase;
        const device uchar* u_wbase;
        const device uchar* g_sbase;
        const device uchar* u_sbase;
        float scale_mul;
        uint up_extra_rows;
        if (is_routed) {
            uint expert = indices[expert_slot];
            g_wbase = (const device uchar*)gate_weight + expert * packed_expert_bytes;
            u_wbase = (const device uchar*)up_weight + expert * packed_expert_bytes;
            g_sbase = gate_scales + expert * scale_expert_bytes;
            u_sbase = up_scales + expert * scale_expert_bytes;
            scale_mul = global_scale;
            up_extra_rows = 0u;
        } else {
            // Fused [gate(512); up(512)] row-concatenated bank (LagunaNvfp4Fuse.fuseGateUp).
            g_wbase = (const device uchar*)shared_weight;
            u_wbase = (const device uchar*)shared_weight;
            g_sbase = shared_scales;
            u_sbase = shared_scales;
            scale_mul = shared_global_scale;
            up_extra_rows = output_width;
        }

        thread float gate_result[2] = {0.f, 0.f};
        thread float up_result[2] = {0.f, 0.f};
        thread float input_values[values_per_lane];

        for (uint block = 0; block < input_width; block += block_width) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(
                    input + block + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4u; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4u * i] = float(values[0]);
                input_values[4u * i + 1u] = float(values[1]);
                input_values[4u * i + 2u] = float(values[2]);
                input_values[4u * i + 3u] = float(values[3]);
            }
            for (uint row = 0; row < 2u; ++row) {
                uint out_row = first_row + row;
                uint up_row = out_row + up_extra_rows;
                const device uchar* gate_w =
                    g_wbase + out_row * packed_row_bytes + block / 2u + lane * 8u;
                const device uchar* up_w =
                    u_wbase + up_row * packed_row_bytes + block / 2u + lane * 8u;
                const device uchar* gate_s =
                    g_sbase + out_row * scale_row_bytes + block / 16u + lane;
                const device uchar* up_s =
                    u_sbase + up_row * scale_row_bytes + block / 16u + lane;
                float gs = iron_nvfp4_e4m3_scale(gate_s[0]) * scale_mul;
                float us = iron_nvfp4_e4m3_scale(up_s[0]) * scale_mul;
                gate_result[row] += iron_nvfp4_qdot_16(gate_w, input_values, gs);
                up_result[row] += iron_nvfp4_qdot_16(up_w, input_values, us);
            }
        }

        device bfloat* activated;
        switch (expert_slot) {
            case 0u: activated = activated_0; break;
            case 1u: activated = activated_1; break;
            case 2u: activated = activated_2; break;
            case 3u: activated = activated_3; break;
            case 4u: activated = activated_4; break;
            case 5u: activated = activated_5; break;
            case 6u: activated = activated_6; break;
            case 7u: activated = activated_7; break;
            default: activated = shared_activated; break;
        }

        for (uint row = 0; row < 2u; ++row) {
            gate_result[row] = simd_sum(gate_result[row]);
            up_result[row] = simd_sum(up_result[row]);
            if (lane == 0u) {
                float g = gate_result[row];
                float u = up_result[row];
                float sig = 1.f / (1.f + metal::exp(-g));
                activated[first_row + row] = bfloat(g * sig * u);
            }
        }
    }
    """

    /// Fused nine-slot (8 routed + 1 shared) gate+up QMV + SwiGLU. Replaces
    /// the routed `moeRoutedSwigluQmv` dispatch plus the separate shared
    /// gate/up/swiglu chain. `sharedWeight`/`sharedScales` are the fused
    /// [gate;up] bank from `fuseGateUp`.
    static func moeNineSlotSwigluQmv(
        input: Tensor,
        gateWeight: Tensor, gateScales: Tensor,
        upWeight: Tensor, upScales: Tensor,
        expertIndices: Tensor,
        routedOutputs: [Tensor],
        sharedWeight: Tensor, sharedScales: Tensor,
        sharedOutput: Tensor,
        global: Float, sharedGlobal: Float,
        on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(routedOutputs.count == 8)
        precondition(input.dtype == .bf16 && input.elementCount == 2048)
        precondition(expertIndices.dtype == .u32 && expertIndices.elementCount == 8)
        precondition(sharedOutput.dtype == .bf16 && sharedOutput.elementCount == 512)

        let mtl = device.mtlDevice
        let p = pso(
            "laguna_nine_slot_nvfp4_swiglu_qmv_bf16",
            source: nineSlotSwigluSrc, device: mtl)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(gateWeight.buffer, offset: gateWeight.offset, index: 1)
        enc.setBuffer(gateScales.buffer, offset: gateScales.offset, index: 2)
        enc.setBuffer(upWeight.buffer, offset: upWeight.offset, index: 3)
        enc.setBuffer(upScales.buffer, offset: upScales.offset, index: 4)
        enc.setBuffer(expertIndices.buffer, offset: expertIndices.offset, index: 5)
        enc.setBuffer(sharedWeight.buffer, offset: sharedWeight.offset, index: 6)
        enc.setBuffer(sharedScales.buffer, offset: sharedScales.offset, index: 7)
        for i in 0 ..< 8 {
            enc.setBuffer(routedOutputs[i].buffer, offset: routedOutputs[i].offset, index: 8 + i)
        }
        enc.setBuffer(sharedOutput.buffer, offset: sharedOutput.offset, index: 16)
        var g = global
        var sg = sharedGlobal
        enc.setBytes(&g, length: 4, index: 17)
        enc.setBytes(&sg, length: 4, index: 18)
        // 9 slots × 128 tiles, TPG=64 (2 SGs × 32): 2-row-per-SIMD geometry,
        // see the dispatch-shape note on `nineSlotSwigluSrc` above.
        enc.dispatchThreadgroups(
            MTLSize(width: 9 * 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Nine-slot down QMV + weighted reduce. Routed slots (0-7) weighted by
    /// `slotWeights`/`indices` exactly as `moeRoutedDownReduce`; the shared
    /// slot (8) uses a dedicated weight bank (no expert gather) and is added
    /// unconditionally (implicit weight 1.0, no router gate), the same
    /// accumulation ORDER as the stock path (routed weighted-sum first, then
    /// `+ shared`), so this reduction is expected to match the stock
    /// `moeRoutedDownReduce` + `Ops.add(acc, sharedOut)` chain's rounding
    /// trajectory row-for-row (not claimed byte-identical against the
    /// *shared* leg specifically, since the shared leg's own dequant-GEMV
    /// numerics differ, see T28 README).
    /// T43: template shared by the stock and residual-folding down-reduce
    /// kernels (no function-constant/#if machinery elsewhere in this file
    /// -- the established convention is a full source string per PSO
    /// variant, see the T28 gate-up-split doc comment above). With
    /// `foldResidual: false` this reproduces `laguna_nine_slot_nvfp4_down_reduce_bf16`
    /// byte-for-byte (stock kernel untouched). With `foldResidual: true` it
    /// emits the `_residual_bf16` sibling: one extra `residual` input
    /// (buffer 18, next free index after the two `setBytes` scalar slots
    /// at 16/17) and the lane-0 epilogue's final store becomes
    /// `bfloat(float(bfloat(total)) + float(residual[row]))` instead of a
    /// bare store. The inner `bfloat(total)` cast deliberately emulates
    /// the stock chain's DOUBLE rounding: the stock down-reduce stores
    /// `bfloat(total)` (round 1) and `Ops.add` then computes
    /// `bfloat(float(a) + float(b))` over the two bf16 inputs (round 2).
    /// Skipping round 1 (adding the unrounded float `total`) is more
    /// accurate but diverges from the two-dispatch chain by 1 ulp on
    /// round-boundary values; emulating both rounds makes the fused
    /// result bit-exact vs. stock, which is the ship gate here.
    private static func nineSlotDownReduceSource(foldResidual: Bool) -> String {
        let fnName = foldResidual
            ? "laguna_nine_slot_nvfp4_down_reduce_residual_bf16"
            : "laguna_nine_slot_nvfp4_down_reduce_bf16"
        let residualParam = foldResidual
            ? "\n        device const bfloat *residual [[buffer(18)]],"
            : ""
        // T43 fix: the residual epilogue's "round 1" MUST be a genuine
        // device-memory round-trip, not just a `bfloat(total)` cast held in
        // registers. `total` is already bf16-typed by the time we get here
        // (see the accumulation loop above), so an in-register
        // `bfloat(total)` cast is a no-op AS FAR AS THE SOURCE IS
        // CONCERNED -- but under Metal's default fastMathEnabled=true (this
        // kernel is compiled with the same default as every other
        // `LagunaNvfp4Fuse.pso(...)` caller, on purpose: the qdot/dequant
        // math above must stay numerically identical to the stock kernel's
        // compile), the compiler is free to treat that cast as an advisory
        // precision hint rather than a hard rounding boundary, since
        // `total` never actually left the function before. The STOCK
        // two-dispatch chain gets its rounding boundary for free: dispatch
        // 1 stores `total` to `output` (device memory, real bf16 truncation
        // no optimizer can undo), dispatch 2 (`Ops.add`) reloads that
        // exact stored value as its `a` operand. Reproducing that here
        // needs an ACTUAL store + reload through `output` (device address
        // space), with a `threadgroup_barrier(mem_flags::mem_device)` in
        // between so the reload can't be forwarded from the just-computed
        // register value -- measured to close a ~2 ULP bf16 drift on ~25%
        // of a layer's output elements that an in-register cast alone left
        // open (present from the very first isolated decode call, no
        // command buffers or pipelining involved). The barrier must be
        // called by every thread in the threadgroup (Metal requirement),
        // so it sits OUTSIDE the `expert_slot == 0u && lane < outputs_per_simd`
        // guard, same pattern the existing `mem_threadgroup` barrier above
        // already uses for its own guarded producer.
        let residualEpilogue =
            foldResidual
            ? """

                threadgroup_barrier(mem_flags::mem_device);
                if (expert_slot == 0u && lane < outputs_per_simd) {
                    // Round 1 already committed to device memory by the
                    // store above; reload it for a real bf16 round-trip
                    // (immune to fast-math register-level reassociation),
                    // then round 2 matches `Ops.add`'s
                    // `bfloat(float(a) + float(b))` exactly.
                    bfloat rounded_total = output[first_row + lane];
                    output[first_row + lane] = bfloat(
                        float(rounded_total)
                        + float(residual[first_row + lane]));
                }
            """
            : ""
        return header + """
    kernel void \(fnName)(
        device const bfloat *mid_0 [[buffer(0)]],
        device const bfloat *mid_1 [[buffer(1)]],
        device const bfloat *mid_2 [[buffer(2)]],
        device const bfloat *mid_3 [[buffer(3)]],
        device const bfloat *mid_4 [[buffer(4)]],
        device const bfloat *mid_5 [[buffer(5)]],
        device const bfloat *mid_6 [[buffer(6)]],
        device const bfloat *mid_7 [[buffer(7)]],
        device const bfloat *shared_mid [[buffer(8)]],
        device const uint *indices [[buffer(9)]],
        device const bfloat *slot_weights [[buffer(10)]],
        device const uint *down_weight [[buffer(11)]],
        device const uchar *down_scales [[buffer(12)]],
        device const uint *shared_down_weight [[buffer(13)]],
        device const uchar *shared_down_scales [[buffer(14)]],
        device bfloat *output [[buffer(15)]],
        constant float &global_scale [[buffer(16)]],
        constant float &shared_global_scale [[buffer(17)]],\(residualParam)
        uint tgid [[threadgroup_position_in_grid]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]
    ) {
        constexpr uint input_width = 512u;
        constexpr uint routed_experts = 8u;
        constexpr uint slots = routed_experts + 1u;
        constexpr uint outputs_per_simd = 4u;
        constexpr uint values_per_lane = 16u;
        constexpr uint n_packs = input_width / 8u;       // 64
        constexpr uint n_blocks = input_width / 16u;     // 32
        constexpr uint packs_per_block = 16u / 8u;       // 2
        constexpr uint out_dim = 2048u;

        uint expert_slot = simd_group;  // 0..8
        uint lane = simd_lane;          // 0..31
        uint first_row = tgid * outputs_per_simd;
        if (expert_slot >= slots || first_row >= out_dim) return;
        bool is_routed = expert_slot < routed_experts;

        const device bfloat* expert_input;
        switch (expert_slot) {
            case 0u: expert_input = mid_0; break;
            case 1u: expert_input = mid_1; break;
            case 2u: expert_input = mid_2; break;
            case 3u: expert_input = mid_3; break;
            case 4u: expert_input = mid_4; break;
            case 5u: expert_input = mid_5; break;
            case 6u: expert_input = mid_6; break;
            case 7u: expert_input = mid_7; break;
            default: expert_input = shared_mid; break;
        }

        const device uint* wbase;
        const device uchar* sbase;
        float scale_mul;
        uint exp_base_packs;
        uint exp_base_blocks;
        if (is_routed) {
            uint expert = indices[expert_slot];
            wbase = down_weight;
            sbase = down_scales;
            exp_base_packs = expert * out_dim * n_packs;
            exp_base_blocks = expert * out_dim * n_blocks;
            scale_mul = global_scale;
        } else {
            wbase = shared_down_weight;
            sbase = shared_down_scales;
            exp_base_packs = 0u;
            exp_base_blocks = 0u;
            scale_mul = shared_global_scale;
        }

        thread float input_values[values_per_lane];
        {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(
                    expert_input + lane * values_per_lane);
            for (uint i = 0; i < values_per_lane / 4u; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4u * i] = float(values[0]);
                input_values[4u * i + 1u] = float(values[1]);
                input_values[4u * i + 2u] = float(values[2]);
                input_values[4u * i + 3u] = float(values[3]);
            }
        }

        uint pack_base = lane * packs_per_block;
        uint blk = pack_base / packs_per_block;
        thread float result[outputs_per_simd] = {0.f, 0.f, 0.f, 0.f};
        for (uint row = 0; row < outputs_per_simd; ++row) {
            uint output_row = first_row + row;
            uint row_pack_off = exp_base_packs + output_row * n_packs + pack_base;
            uint row_block_off = exp_base_blocks + output_row * n_blocks + blk;
            float scale =
                iron_nvfp4_e4m3_scale(sbase[row_block_off]) * scale_mul;
            const device uchar* wbytes =
                (const device uchar*)(wbase + row_pack_off);
            result[row] = iron_nvfp4_qdot_16(wbytes, input_values, scale);
            result[row] = simd_sum(result[row]);
        }

        threadgroup bfloat expert_outputs[slots * outputs_per_simd];
        if (lane == 0u) {
            for (uint row = 0; row < outputs_per_simd; ++row) {
                expert_outputs[expert_slot * outputs_per_simd + row] = bfloat(result[row]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (expert_slot == 0u && lane < outputs_per_simd) {
            bfloat total = bfloat(0.0f);
            for (uint slot = 0u; slot < routed_experts; ++slot) {
                bfloat route_weight = slot_weights[slot];
                bfloat product = bfloat(
                    float(expert_outputs[slot * outputs_per_simd + lane])
                    * float(route_weight));
                total = bfloat(float(product) + float(total));
            }
            // Shared expert: unconditional add, implicit weight 1.0.
            total = bfloat(
                float(total)
                + float(expert_outputs[routed_experts * outputs_per_simd + lane]));
            output[first_row + lane] = total;
        }\(residualEpilogue)
    }
    """
    }

    private static let nineSlotDownReduceSrc = nineSlotDownReduceSource(foldResidual: false)

    /// T43 residual-folding sibling of `nineSlotDownReduceSrc`: identical
    /// kernel body plus a `residual` input and a final store that adds it
    /// in (`Ops.add`-bit-exact rounding, see `nineSlotDownReduceSource`).
    private static let nineSlotDownReduceResidualSrc = nineSlotDownReduceSource(foldResidual: true)

    /// Nine-slot down QMV + weighted reduce (routed + shared, one dispatch).
    /// T43: when `residual` is non-nil, the fused `_residual_bf16` PSO
    /// variant runs instead, folding `output = residual + (routed-weighted-
    /// sum + shared)` into the same dispatch -- removing the separate
    /// `Ops.add(h2, ffnOut, ...)` the caller would otherwise need. `residual`
    /// must be `[hidden]` bf16, same shape/dtype as `output`; the stock
    /// (residual == nil) PSO/dispatch geometry is completely unchanged.
    static func moeNineSlotDownReduce(
        mids: [Tensor], sharedMid: Tensor,
        expertIndices: Tensor, slotWeights: Tensor,
        downWeight: Tensor, downScales: Tensor,
        sharedDownWeight: Tensor, sharedDownScales: Tensor,
        output: Tensor,
        global: Float, sharedGlobal: Float,
        residual: Tensor? = nil,
        on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(mids.count == 8)
        precondition(sharedMid.dtype == .bf16 && sharedMid.elementCount == 512)
        precondition(output.dtype == .bf16 && output.elementCount == 2048)
        if let residual {
            precondition(residual.dtype == .bf16 && residual.elementCount == 2048)
        }

        let mtl = device.mtlDevice
        let p = residual == nil
            ? pso(
                "laguna_nine_slot_nvfp4_down_reduce_bf16",
                source: nineSlotDownReduceSrc, device: mtl)
            : pso(
                "laguna_nine_slot_nvfp4_down_reduce_residual_bf16",
                source: nineSlotDownReduceResidualSrc, device: mtl)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        for i in 0 ..< 8 {
            enc.setBuffer(mids[i].buffer, offset: mids[i].offset, index: i)
        }
        enc.setBuffer(sharedMid.buffer, offset: sharedMid.offset, index: 8)
        enc.setBuffer(expertIndices.buffer, offset: expertIndices.offset, index: 9)
        enc.setBuffer(slotWeights.buffer, offset: slotWeights.offset, index: 10)
        enc.setBuffer(downWeight.buffer, offset: downWeight.offset, index: 11)
        enc.setBuffer(downScales.buffer, offset: downScales.offset, index: 12)
        enc.setBuffer(sharedDownWeight.buffer, offset: sharedDownWeight.offset, index: 13)
        enc.setBuffer(sharedDownScales.buffer, offset: sharedDownScales.offset, index: 14)
        enc.setBuffer(output.buffer, offset: output.offset, index: 15)
        var g = global
        var sg = sharedGlobal
        enc.setBytes(&g, length: 4, index: 16)
        enc.setBytes(&sg, length: 4, index: 17)
        if let residual {
            enc.setBuffer(residual.buffer, offset: residual.offset, index: 18)
        }
        // 512 tiles × 4 rows = 2048; TPG=288 (9 SGs × 32 lanes)
        enc.dispatchThreadgroups(
            MTLSize(width: 512, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 288, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - T28 follow-up: gate-up split (routed-8 + shared-1)

    /// Split the nine-slot GATE-UP dispatch back into its two constituent
    /// kernels: `routedSwigluSrc` (slots 0-7, needs `expertIndices`) and
    /// `sharedSwigluSrc` (slot 8, needs only `input`), encoded on ONE
    /// `.concurrent`-dispatch-type encoder, shared first. Down+residual
    /// stays on the merged `moeNineSlotDownReduce` dispatch regardless;
    /// only the gate-up half is split here.
    ///
    /// Why this helps (frontier-measured, +0.98% over the fully-merged
    /// nine-slot gate-up): `laguna_nine_slot_nvfp4_swiglu_qmv_bf16` is a
    /// single kernel launch whose routed threadgroups (1024 of 1152) read
    /// `indices`. Metal's hazard tracker therefore can't start ANY of that
    /// dispatch's threadgroups, including the 128 shared-slot ones, which
    /// never touch `indices`, until the router chain's write to
    /// `indices`/`routerWeightsScratch` has retired: one dispatch means one
    /// hazard set, so the shared leg's independence is invisible to the
    /// tracker. Splitting shared into its own dispatch removes that false
    /// dependency: shared only touches `input` + its own fused weight bank
    /// + `sharedOutput`, none of which the router chain writes, so once
    /// it's its own dispatch the hazard tracker is free to schedule it
    /// without waiting on `indices`. Placing both dispatches on one
    /// `.concurrent` encoder (rather than two separate default/serial
    /// encoders) additionally drops the blanket ordering `.serial` would
    /// otherwise impose between the shared and routed dispatches
    /// themselves; same mechanism this file already leans on in
    /// `moeGateUpSwigluMany` / `moeSlotSwigluDownMany` above for
    /// independent per-slot dispatches. Shared is encoded first simply
    /// because it's the one with nothing left to wait on; routed is
    /// encoded second because it must follow the router chain's write to
    /// `expertIndices` in program order regardless of encoder/dispatch
    /// type (that's a true hazard, not a false one, so it's not something
    /// `.concurrent` scheduling removes).
    ///
    /// Reuses `sharedSwigluSrc` / `routedSwigluSrc` verbatim (same PSOs the
    /// pre-T28 code path and `moeRoutedSwigluQmv` already use elsewhere in
    /// this file); the nine-slot kernel's own header comment already
    /// documents its per-row arithmetic as identical to these two, so this
    /// is purely a dispatch-count change, not a new numerics path.
    ///
    /// Placement (caller side, see `MoELayer.decodeNineSlot`): encode
    /// AFTER `runPathBRouteChain` (routed needs `expertIndices` ready) and
    /// on the SAME command buffer/queue as everything else in the nine-slot
    /// decode path. `decodeNineSlot` does not use the dual-queue (`cmdB` +
    /// sync-event) pattern; that pattern only exists in the older,
    /// pre-T28 `decodeGPUPathBNvfp4` fallback below the nine-slot branch,
    /// so there is no second queue to hand the shared leg to here without
    /// reintroducing the ~1 CB + 2 events/layer overhead T28 specifically
    /// removed. A same-encoder concurrent split is the available overlap
    /// lever that doesn't reopen that cost.
    static func moeGateUpSplitSwigluQmv(
        input: Tensor,
        gateWeight: Tensor, gateScales: Tensor,
        upWeight: Tensor, upScales: Tensor,
        expertIndices: Tensor,
        routedOutputs: [Tensor],
        sharedWeight: Tensor, sharedScales: Tensor,
        sharedOutput: Tensor,
        global: Float, sharedGlobal: Float,
        on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(routedOutputs.count == 8)
        precondition(input.dtype == .bf16 && input.elementCount == 2048)
        precondition(expertIndices.dtype == .u32 && expertIndices.elementCount == 8)
        precondition(sharedOutput.dtype == .bf16 && sharedOutput.elementCount == 512)

        let mtl = device.mtlDevice
        let pShared = pso(
            "laguna_shared_nvfp4_swiglu_qmv", source: sharedSwigluSrc, device: mtl)
        let pRouted = pso(
            "laguna_moe_routed_swiglu_qmv_nvfp4_bf16",
            source: routedSwigluSrc, device: mtl)
        guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { return }

        // Shared first: no dependency on `expertIndices`, see rationale
        // above. Same buffer layout as `sharedSwigluQmv`, but writes into
        // the caller's `sharedOutput` scratch instead of allocating fresh.
        enc.setComputePipelineState(pShared)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(sharedWeight.buffer, offset: sharedWeight.offset, index: 1)
        enc.setBuffer(sharedScales.buffer, offset: sharedScales.offset, index: 2)
        enc.setBuffer(sharedOutput.buffer, offset: sharedOutput.offset, index: 3)
        var sg = sharedGlobal
        enc.setBytes(&sg, length: 4, index: 4)
        // 128 tiles × 64 threads (2 SGs); same geometry as `sharedSwigluQmv`.
        enc.dispatchThreadgroups(
            MTLSize(width: 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        // Routed second: depends on `expertIndices` (written by
        // `runPathBRouteChain` before this function is called). Same
        // buffer layout/geometry as `moeRoutedSwigluQmv`.
        enc.setComputePipelineState(pRouted)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(gateWeight.buffer, offset: gateWeight.offset, index: 1)
        enc.setBuffer(gateScales.buffer, offset: gateScales.offset, index: 2)
        enc.setBuffer(upWeight.buffer, offset: upWeight.offset, index: 3)
        enc.setBuffer(upScales.buffer, offset: upScales.offset, index: 4)
        enc.setBuffer(expertIndices.buffer, offset: expertIndices.offset, index: 5)
        for i in 0 ..< 8 {
            enc.setBuffer(
                routedOutputs[i].buffer, offset: routedOutputs[i].offset, index: 6 + i)
        }
        var g = global
        enc.setBytes(&g, length: 4, index: 14)
        // 8 experts × 128 tiles; TPG=64 (2 SGs × 32); same geometry as
        // `moeRoutedSwigluQmv`.
        enc.dispatchThreadgroups(
            MTLSize(width: 8 * 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        enc.endEncoding()
    }
}
