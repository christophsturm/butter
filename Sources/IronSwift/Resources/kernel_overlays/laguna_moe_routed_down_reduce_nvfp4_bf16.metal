// Hand-written Laguna Path B: fused routed down NVFP4 QMV + BF16 weighted reduce.
// Geometry: mid=512, hidden=2048, topK=8, block=16.
// Grid: 512 tiles × 4 rows; TPG=256 (8 SGs × 32 lanes, one SG per expert slot).
// Iron-bitexact E2M1 via mlxfast half bit-twiddle (*16384).
// Prefer metallib PSO (this file) over live makeLibrary(source:).

#include <metal_stdlib>
using namespace metal;

// CARRY SIGN-FOLD E4M3 decode (unconditional, bit-exact for all 256 byte
// values -- see Tests/ButterTests/Ops/Nvfp4FastDequantTests.swift and the
// derivation in LagunaNvfp4Fuse.swift's `header`). Replaces a
// mask/shift/select ~7-op sequence with 5 ops.
static inline float iron_nvfp4_e4m3_scale(uchar bits) {
    ushort h = ushort((uint(bits) + (uint(bits) & 128u)) << 7);
    return float(as_type<half>(h)) * 256.0f;
}

// NIBBLE-SPLIT word-parallel E2M1 decode (19 -> 13 int ops per pack; see
// LagunaNvfp4Fuse.swift's `header` for the full derivation + equivalence
// table, and Tests/ButterTests/Ops/Nvfp4FastDequantTests.swift for the
// exhaustive per-nibble test). `scale` must already carry the *16384.0f
// half-domain-restore fold -- see iron_nvfp4_qdot_16 below (task 3).
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

// External contract unchanged (`scale` = plain e4m3*global). The
// *16384.0f fold (2^22-class combined with the E4M3 fold above once
// composed at a caller that also pre-scales) now happens ONCE per
// 16-element dot instead of 8x inside pack_dot_fast.
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
    uint blk = pack_base / packs_per_block;  // == lane
    thread float result[outputs_per_simd] = {0.f, 0.f, 0.f, 0.f};
    for (uint row = 0; row < outputs_per_simd; ++row) {
        uint output_row = first_row + row;
        uint row_pack_off = exp_base_packs + output_row * n_packs + pack_base;
        uint row_block_off = exp_base_blocks + output_row * n_blocks + blk;
        float scale =
            iron_nvfp4_e4m3_scale(down_scales[row_block_off]) * global_scale;
        const device uchar* wbytes =
            (const device uchar*)(down_weight + row_pack_off);
        result[row] = iron_nvfp4_qdot_16(wbytes, input_values, scale);
        result[row] = simd_sum(result[row]);
    }

    threadgroup bfloat expert_outputs[experts_per_token * outputs_per_simd];
    if (lane == 0u) {
        for (uint row = 0; row < outputs_per_simd; ++row) {
            expert_outputs[expert_slot * outputs_per_simd + row] =
                bfloat(result[row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // BF16 product + accumulate in slot order 0..7 (matches scalarFMAChain8).
    // Router weights already include routedScalingFactor.
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
