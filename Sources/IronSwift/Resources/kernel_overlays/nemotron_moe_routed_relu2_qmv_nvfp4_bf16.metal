// Hand-written NemotronH-Lightning Path B: fused routed up-projection
// NVFP4 QMV + squared-ReLU (bf16). Gate-less analog of
// laguna_moe_routed_swiglu_qmv_nvfp4_bf16.metal — Lightning's routed
// experts are `down(relu(up(x))^2)` (no gate projection).
//
// Geometry: hidden=2688, moeIntermediate=1856, topK=6, block=16.
// Grid: 6 experts x 464 tiles; TPG=64 (2 SGs x 32).
// Iron-bitexact E2M1 via mlxfast half bit-twiddle (*16384).
//
// Unlike Laguna (mid=512, hidden=2048 -- both exact multiples of
// 32 lanes x 16 values/lane = 512), Lightning's hidden=2688 is NOT a
// multiple of 512 (2688 / 512 = 5.25), so this kernel can't reuse
// Laguna's fixed `block_width=512` outer-loop shape. Instead each lane
// owns one 16-wide NVFP4 group per iteration (`groupIdx = iter*32 +
// lane`) and the loop runs `ceil(n_blocks/32)` iterations with the
// last iteration's excess lanes masked off (`valid`) -- n_blocks =
// hidden/16 = 168 = 5*32 + 8, so iteration 5 only has 8 active lanes.
// This generalizes cleanly to any hidden width that's a multiple of 16
// (the NVFP4 block size) without requiring a multiple of 512.
//
// Epilogue MATCHES the unfused chain's bf16 rounding exactly (see
// NemotronLightningNvfp4Fuse.swift's doc comment on
// `moeRoutedRelu2Qmv`): one bf16 round after the QMV (mimics the
// up-GEMV's bf16 store), relu is lossless on an already-bf16 value, one
// more bf16 round after squaring (mimics Ops.mul's bf16 store).

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
// half-domain-restore fold -- see iron_nvfp4_qdot_16 below.
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

kernel void nemotron_moe_routed_relu2_qmv_nvfp4_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uint *up_weight [[buffer(1)]],
    device const uchar *up_scales [[buffer(2)]],
    device const uint *indices [[buffer(3)]],
    device bfloat *activated_0 [[buffer(4)]],
    device bfloat *activated_1 [[buffer(5)]],
    device bfloat *activated_2 [[buffer(6)]],
    device bfloat *activated_3 [[buffer(7)]],
    device bfloat *activated_4 [[buffer(8)]],
    device bfloat *activated_5 [[buffer(9)]],
    constant float &global_scale [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint input_width = 2688u;   // hidden
    constexpr uint output_width = 1856u;  // moeIntermediate
    constexpr uint packed_row_bytes = input_width / 2u;   // 1344
    constexpr uint scale_row_bytes = input_width / 16u;   // 168
    constexpr uint packed_expert_bytes = output_width * packed_row_bytes;
    constexpr uint scale_expert_bytes = output_width * scale_row_bytes;
    constexpr uint n_blocks = input_width / 16u;           // 168
    constexpr uint n_iters = (n_blocks + 31u) / 32u;       // 6 (ceil)
    constexpr uint routed_experts = 6u;

    // Tile-major: interleave expert slots so banks overlap in the wave.
    uint expert_slot = tgid % routed_experts;
    uint tile = tgid / routed_experts;
    uint expert = indices[expert_slot];
    uint first_row = tile * 4u + sgid * 2u;

    const device uchar* u_wbase =
        (const device uchar*)up_weight + expert * packed_expert_bytes;
    const device uchar* u_sbase =
        up_scales + expert * scale_expert_bytes;

    thread float up_result[2] = {0.f, 0.f};
    thread float input_values[16];

    for (uint iter = 0u; iter < n_iters; ++iter) {
        uint groupIdx = iter * 32u + lane;
        bool valid = groupIdx < n_blocks;
        if (valid) {
            const device vec<bfloat, 4>* input_vectors =
                (const device vec<bfloat, 4>*)(input + groupIdx * 16u);
            for (uint i = 0; i < 4u; ++i) {
                const vec<bfloat, 4> values = input_vectors[i];
                input_values[4u * i + 0u] = float(values[0]);
                input_values[4u * i + 1u] = float(values[1]);
                input_values[4u * i + 2u] = float(values[2]);
                input_values[4u * i + 3u] = float(values[3]);
            }
        }
        for (uint row = 0u; row < 2u; ++row) {
            if (!valid) continue;
            uint out_row = first_row + row;
            const device uchar* up_w =
                u_wbase + out_row * packed_row_bytes + groupIdx * 8u;
            const device uchar* up_s =
                u_sbase + out_row * scale_row_bytes + groupIdx;
            float us = iron_nvfp4_e4m3_scale(up_s[0]) * global_scale;
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
        default: activated = activated_5; break;
    }

    for (uint row = 0u; row < 2u; ++row) {
        up_result[row] = simd_sum(up_result[row]);
        if (lane == 0u) {
            // Match Ops.relu(Ops.nvfp4DequantGemvExpertIndexedMany(...))
            // then Ops.mul(r, r): the unfused chain rounds to bf16 after
            // the GEMV (up_bf), relu is exact on an already-bf16 value
            // (0 and positives round-trip losslessly), then rounds again
            // after the square.
            bfloat up_bf = bfloat(up_result[row]);
            float r = (up_bf > bfloat(0.0f)) ? float(up_bf) : 0.0f;
            activated[first_row + row] = bfloat(r * r);
        }
    }
}
