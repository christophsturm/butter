// Iron-ABI NVFP4 expert-indexed dequant GEMV (f16) with mlxfast half bit-twiddle qdot.
// See iron_nvfp4_dequant_gemv_expert_indexed_bf16.metal for notes.

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
// half-domain-restore fold -- folded once at this file's call site below
// (task 3; no qdot_16 wrapper in this kernel family).
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

kernel void iron_nvfp4_dequant_gemv_expert_indexed_f16(
    const device uint *weights_stacked [[buffer(0)]],
    const device uchar *scales_stacked [[buffer(1)]],
    const device half *input [[buffer(2)]],
    const device uint *expert_index [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &in_dim [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    constant uint &block_size [[buffer(7)]],
    constant float &global [[buffer(8)]],
    uint3 _tid3  [[thread_position_in_threadgroup]],
    uint3 _tgid3 [[threadgroup_position_in_grid]],
    uint3 _lsize3 [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint tid    = _tid3.x;
    uint tgid_x = _tgid3.x;
    uint lsize  = _lsize3.x;
    uint n_simd = lsize / 32u;
    if (n_simd == 0u) { return; }
    threadgroup float v_total_sg[32];

    uint row = tgid_x;
    uint n_blocks = in_dim / block_size;
    uint expert = expert_index[0];
    uint n_packs_per_row = in_dim / 8u;
    uint packs_per_block = block_size / 8u;
    uint row_pack_off = expert * out_dim * n_packs_per_row + row * n_packs_per_row;
    uint row_block_off = expert * out_dim * n_blocks + row * n_blocks;

    float acc = 0.0f;
    uint p_iters = (n_packs_per_row + lsize - 1u) / lsize;
    for (uint p_iter = 0u; p_iter < p_iters; p_iter += 1u) {
        uint pack_idx = p_iter * lsize + tid;
        if (pack_idx < n_packs_per_row) {
            uint blk = pack_idx / packs_per_block;
            // Task 3: *16384.0f folded here (once per pack) instead of
            // 4x inside pack_dot_fast; exact power-of-2 multiply.
            float scale =
                iron_nvfp4_e4m3_scale(scales_stacked[row_block_off + blk])
                * global * 16384.0f;
            uint packed = weights_stacked[row_pack_off + pack_idx];
            uint p_off = pack_idx * 8u;
            thread float xv[8];
            for (uint i = 0u; i < 8u; ++i) {
                xv[i] = float(input[p_off + i]);
            }
            acc += iron_nvfp4_pack_dot_fast(packed, scale, xv, 0u);
        }
    }

    float sg_sum = simd_sum(acc);
    if (simd_lane == 0u) {
        v_total_sg[simd_group] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (simd_group == 0u) {
        float lane_v = (simd_lane < n_simd) ? v_total_sg[simd_lane] : 0.0f;
        total = simd_sum(lane_v);
        if (simd_lane == 0u) {
            v_total_sg[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        output[row] = half(v_total_sg[0]);
    }
}
