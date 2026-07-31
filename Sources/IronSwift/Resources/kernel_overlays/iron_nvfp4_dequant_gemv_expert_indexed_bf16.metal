// Iron-ABI NVFP4 expert-indexed dequant GEMV (bf16) with mlxfast half bit-twiddle qdot.
// Same buffers/dispatch as iron-generated `iron_nvfp4_dequant_gemv_expert_indexed_bf16`:
//   Mode Reduction; grid [out_dim]; TPG 64 (multiple of 32).
// Bit-exact vs iron_decode_e2m1 + iron_decode_e4m3 * global.
// NOTE: `iron build --emit` overwrites this file — butter Makefile re-applies
// the fast-qdot overlay from kernel_overlays/ after emit (see link-extra-metals).

#include <metal_stdlib>
using namespace metal;

static inline float iron_nvfp4_e4m3_scale(uchar bits) {
    uint b = uint(bits);
    uint exp = (b >> 3) & 15u;
    float mant = float(b & 7u);
    float mag;
    if (exp == 0u) {
        mag = mant * 0.001953125f;
    } else {
        mag = (1.0f + mant * 0.125f) * exp2(float(exp) - 7.0f);
    }
    return ((b >> 7) & 1u) != 0u ? -mag : mag;
}

// mlxfast half bit-twiddle: 8 nibbles per u32 pack, bit-exact vs iron e2m1.
static inline float iron_nvfp4_pack_dot_fast(
    uint packed, float scale, const thread float* input, uint p_off
) {
    float accum = 0.0f;
    const uint p0 =
        ((packed & 0x00070007u) << 9) | ((packed & 0x00080008u) << 12);
    const uint p1 =
        ((packed & 0x00700070u) << 5) | ((packed & 0x00800080u) << 8);
    const uint p2 =
        ((packed & 0x07000700u) << 1) | ((packed & 0x08000800u) << 4);
    const uint p3 =
        ((packed & 0x70007000u) >> 3) | (packed & 0x80008000u);
    const float2 v04 = float2(as_type<half2>(p0)) * 16384.0f;
    const float2 v15 = float2(as_type<half2>(p1)) * 16384.0f;
    const float2 v26 = float2(as_type<half2>(p2)) * 16384.0f;
    const float2 v37 = float2(as_type<half2>(p3)) * 16384.0f;
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

kernel void iron_nvfp4_dequant_gemv_expert_indexed_bf16(
    const device uint *weights_stacked [[buffer(0)]],
    const device uchar *scales_stacked [[buffer(1)]],
    const device bfloat *input [[buffer(2)]],
    const device uint *expert_index [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
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
            float scale =
                iron_nvfp4_e4m3_scale(scales_stacked[row_block_off + blk]) * global;
            uint packed = weights_stacked[row_pack_off + pack_idx];
            uint p_off = pack_idx * 8u;
            // Stage 8 inputs for this pack into thread regs for the fast path.
            thread float xv[8];
            for (uint i = 0u; i < 8u; ++i) {
                xv[i] = float(input[p_off + i]);
            }
            acc += iron_nvfp4_pack_dot_fast(packed, scale, xv, 0u);
        }
    }

    // SG reduce then TG reduce (matches iron Reduction contract).
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
        output[row] = bfloat(v_total_sg[0]);
    }
}
