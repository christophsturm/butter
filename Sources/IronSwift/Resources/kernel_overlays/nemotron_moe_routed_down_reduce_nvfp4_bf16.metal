// Hand-written NemotronH-Lightning Path B: fused routed down NVFP4 QMV +
// BF16 weighted reduce. Analog of
// laguna_moe_routed_down_reduce_nvfp4_bf16.metal, adapted for topK=6 and
// moeIntermediate=1856 (down-projection input width).
//
// Geometry: moeIntermediate=1856, hidden=2688, topK=6, block=16.
// Grid: 672 tiles x 4 rows; TPG=192 (6 SGs x 32 lanes, one SG per
// expert slot).
// Iron-bitexact E2M1 via mlxfast half bit-twiddle (*16384).
//
// Laguna's down-reduce loads its whole mid=512 input in ONE shot (32
// lanes x 16 values/lane = 512, exact). Lightning's moeIntermediate=1856
// is not a multiple of 512 (1856 / 512 = 3.625; even at the 16-wide
// NVFP4-block granularity, 1856/16 = 116 is not a multiple of 32 lanes),
// so this kernel loops `ceil(n_blocks/32)` iterations with the same
// masked-lane pattern as the sibling relu2-QMV kernel.

#include <metal_stdlib>
using namespace metal;

static inline float iron_nvfp4_e4m3_scale(uchar bits) {
    ushort h = ushort((uint(bits) + (uint(bits) & 128u)) << 7);
    return float(as_type<half>(h)) * 256.0f;
}

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

kernel void nemotron_moe_routed_down_reduce_nvfp4_bf16(
    device const bfloat *mid_0 [[buffer(0)]],
    device const bfloat *mid_1 [[buffer(1)]],
    device const bfloat *mid_2 [[buffer(2)]],
    device const bfloat *mid_3 [[buffer(3)]],
    device const bfloat *mid_4 [[buffer(4)]],
    device const bfloat *mid_5 [[buffer(5)]],
    device const uint *indices [[buffer(6)]],
    device const bfloat *slot_weights [[buffer(7)]],
    device const uint *down_weight [[buffer(8)]],
    device const uchar *down_scales [[buffer(9)]],
    device bfloat *output [[buffer(10)]],
    constant float &global_scale [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint input_width = 1856u;     // moeIntermediate
    constexpr uint experts_per_token = 6u;
    constexpr uint outputs_per_simd = 4u;
    constexpr uint n_blocks = input_width / 16u;          // 116
    constexpr uint n_iters = (n_blocks + 31u) / 32u;       // 4 (ceil)
    constexpr uint out_dim = 2688u;                        // hidden
    constexpr uint n_packs_row = input_width / 8u;         // 232 (u32 words/row)

    uint expert_slot = simd_group;  // 0..5
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
        default: expert_input = mid_5; break;
    }

    uint expert = indices[expert_slot];
    uint exp_base_packs = expert * out_dim * n_packs_row;
    uint exp_base_blocks = expert * out_dim * n_blocks;

    thread float result[outputs_per_simd] = {0.f, 0.f, 0.f, 0.f};

    for (uint iter = 0u; iter < n_iters; ++iter) {
        uint groupIdx = iter * 32u + lane;
        bool valid = groupIdx < n_blocks;
        thread float input_values[16];
        if (valid) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(
                    expert_input + groupIdx * 16u);
            for (uint i = 0; i < 4u; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4u * i + 0u] = float(values[0]);
                input_values[4u * i + 1u] = float(values[1]);
                input_values[4u * i + 2u] = float(values[2]);
                input_values[4u * i + 3u] = float(values[3]);
            }
        }
        for (uint row = 0u; row < outputs_per_simd; ++row) {
            if (!valid) continue;
            uint output_row = first_row + row;
            // 2 u32 packs per 16-wide group -- index in u32 (word) units.
            uint row_pack_off = exp_base_packs + output_row * n_packs_row + groupIdx * 2u;
            uint row_block_off = exp_base_blocks + output_row * n_blocks + groupIdx;
            float scale =
                iron_nvfp4_e4m3_scale(down_scales[row_block_off]) * global_scale;
            const device uchar* wbytes =
                (const device uchar*)(down_weight + row_pack_off);
            result[row] += iron_nvfp4_qdot_16(wbytes, input_values, scale);
        }
    }
    for (uint row = 0u; row < outputs_per_simd; ++row) {
        result[row] = simd_sum(result[row]);
    }

    threadgroup bfloat expert_outputs[experts_per_token * outputs_per_simd];
    if (lane == 0u) {
        for (uint row = 0u; row < outputs_per_simd; ++row) {
            expert_outputs[expert_slot * outputs_per_simd + row] =
                bfloat(result[row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // BF16 product + accumulate in slot order 0..5 (matches the unfused
    // chain's `headGateMul` + serial `Ops.add` accumulate order --
    // slot 0 seeds the accumulator directly, slots 1..5 chain-add).
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
