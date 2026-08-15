// Hand-written Laguna Path B: fused routed gate+up NVFP4 QMV + SwiGLU (bf16).
// Geometry: mid=512, hidden=2048, topK=8, block=16.
// Grid: 8 experts × 128 tiles; TPG=64 (2 SGs × 32).
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
// *16384.0f fold now happens ONCE per 16-element dot instead of 8x
// inside pack_dot_fast.
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
