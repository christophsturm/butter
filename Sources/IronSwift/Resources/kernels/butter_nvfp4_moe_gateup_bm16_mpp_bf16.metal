// Hand-written NVFP4 MoE gather MPP: fused gate+up, shared X load, x_rows gather (bf16).
// Geometry: BM=16 BN=32 BK=16, 1 SG. X is [n_tokens, k_in] bf16; x_rows selects token row.
#include <metal_stdlib>
using namespace metal;

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

static inline float butter_nvfp4_scale_b(uint sraw, float global_scale) {
    uint exp_bits = (sraw >> 3) & 15u;
    float mant = float(sraw & 7u);
    float scale;
    if (exp_bits < 1u) {
        scale = mant * 0.001953125f;
    } else {
        scale = (1.0f + mant * 0.125f) * exp2(float(exp_bits) - 7.0f);
    }
    if (((sraw >> 7) & 1u) != 0u) scale = -scale;
    return scale * global_scale;
}

static inline float butter_nvfp4_nibble_b(uint nib) {
    uint mag = nib & 7u;
    float v = select(6.0f, 4.0f, mag < 7u);
    v = select(v, 3.0f, mag < 6u);
    v = select(v, 2.0f, mag < 5u);
    v = select(v, 1.5f, mag < 4u);
    v = select(v, 1.0f, mag < 3u);
    v = select(v, 0.5f, mag < 2u);
    v = select(v, 0.0f, mag < 1u);
    return ((nib & 8u) != 0u) ? -v : v;
}

static inline void butter_dequant_w_tile_half(
    const device uint *w,
    const device uchar *scales,
    threadgroup half *ws,
    uint expert, uint n_tile_base, uint k_base,
    uint n_out, uint k_in, uint block_size, float global_scale,
    uint lane
) {
    uint packs_per_row = k_in / 8u;
    uint groups_per_row = k_in / block_size;
    uint w_expert_pack = expert * n_out * packs_per_row;
    uint sb_expert_base = expert * n_out * groups_per_row;
    for (uint i = 0; i < 2u; i++) {
        uint pack_id = lane * 2u + i;
        uint w_row = pack_id / 2u;
        uint stripe = pack_id % 2u;
        uint g_row = n_tile_base + w_row;
        uint sb_off = sb_expert_base + g_row * groups_per_row + ((k_base + stripe * 8u) / block_size);
        float scale = butter_nvfp4_scale_b(uint(scales[sb_off]), global_scale);
        uint packed = w[w_expert_pack + g_row * packs_per_row + (k_base / 8u) + stripe];
        uint dst = w_row * 16u + stripe * 8u;
        for (uint j = 0; j < 8u; j++) {
            uint nib = (packed >> (j * 4u)) & 15u;
            ws[dst + j] = half(butter_nvfp4_nibble_b(nib) * scale);
        }
    }
}

kernel void butter_nvfp4_moe_gateup_bm16_mpp_bf16(
    const device bfloat *x [[buffer(0)]],
    const device uint *x_rows [[buffer(1)]],
    const device uint *w_gate [[buffer(2)]],
    const device uchar *scales_gate [[buffer(3)]],
    const device uint *w_up [[buffer(4)]],
    const device uchar *scales_up [[buffer(5)]],
    const device uint *indices [[buffer(6)]],
    device bfloat *out_gate [[buffer(7)]],
    device bfloat *out_up [[buffer(8)]],
    constant uint &m_total [[buffer(9)]],
    constant uint &n_out [[buffer(10)]],
    constant uint &k_in [[buffer(11)]],
    constant uint &block_size [[buffer(12)]],
    constant float &global_scale [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
    const uint n_tile_base = tgid.x * 32u;
    const uint m_tile_base = tgid.y * 16u;

    threadgroup half xs[256];
    threadgroup half ws[512];
    threadgroup float out_g[512];
    threadgroup float out_u[512];

    constexpr auto gemm_desc = mpp::tensor_ops::matmul2d_descriptor(
        /*M=*/16, /*N=*/32, /*K=*/16,
        /*ta=*/false, /*tb=*/true, /*tc=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<gemm_desc, metal::execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ct_cg = gemm_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
    auto ct_cu = gemm_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    uint sub = 0u;
    while (sub < 16u) {
        uint cur_row = m_tile_base + sub;
        bool cur_ok = cur_row < m_total;
        uint cur_expert = cur_ok ? indices[cur_row] : 0xFFFFFFFFu;
        uint sub_end = 16u;
        for (uint p = sub + 1u; p < 16u; p++) {
            uint pr = m_tile_base + p;
            if (pr >= m_total) {
                sub_end = p;
                break;
            }
            if (indices[pr] != cur_expert) {
                sub_end = p;
                break;
            }
        }
        if (!cur_ok) break;

        for (uint16_t i = 0; i < ct_cg.get_capacity(); ++i) ct_cg[i] = {};
        for (uint16_t i = 0; i < ct_cu.get_capacity(); ++i) ct_cu[i] = {};

        for (uint k0 = 0u; k0 < k_in; k0 += 16u) {
            for (uint t = 0u; t < 8u; t++) {
                uint flat = lane * 8u + t;
                uint mr = flat / 16u;
                uint kc = flat % 16u;
                uint gr = m_tile_base + mr;
                bool in_run = (mr >= sub) && (mr < sub_end) && (gr < m_total);
                float xv = 0.0f;
                if (in_run) {
                    uint xrow = x_rows[gr];
                    xv = float(x[xrow * k_in + k0 + kc]);
                }
                xs[mr * 16u + kc] = half(xv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            butter_dequant_w_tile_half(
                w_gate, scales_gate, ws, cur_expert, n_tile_base, k0,
                n_out, k_in, block_size, global_scale, lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                metal::tensor<threadgroup half, metal::extents<int, 16, 16>, metal::tensor_inline>
                    tA(xs, metal::extents<int, 16, 16>{});
                metal::tensor<threadgroup half, metal::extents<int, 16, 32>, metal::tensor_inline>
                    tB(ws, metal::extents<int, 16, 32>{});
                ct_a.load(tA);
                ct_b.load(tB);
                gemm_op.run(ct_a, ct_b, ct_cg);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            butter_dequant_w_tile_half(
                w_up, scales_up, ws, cur_expert, n_tile_base, k0,
                n_out, k_in, block_size, global_scale, lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                metal::tensor<threadgroup half, metal::extents<int, 16, 16>, metal::tensor_inline>
                    tA(xs, metal::extents<int, 16, 16>{});
                metal::tensor<threadgroup half, metal::extents<int, 16, 32>, metal::tensor_inline>
                    tB(ws, metal::extents<int, 16, 32>{});
                ct_a.load(tA);
                ct_b.load(tB);
                gemm_op.run(ct_a, ct_b, ct_cu);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        {
            metal::tensor<threadgroup float, metal::extents<int, 32, 16>, metal::tensor_inline>
                tCg(out_g, metal::extents<int, 32, 16>{});
            metal::tensor<threadgroup float, metal::extents<int, 32, 16>, metal::tensor_inline>
                tCu(out_u, metal::extents<int, 32, 16>{});
            ct_cg.store(tCg);
            ct_cu.store(tCu);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0u; i < 16u; i++) {
            uint idx = lane * 16u + i;
            uint mr = idx / 32u;
            uint nc = idx % 32u;
            uint gr = m_tile_base + mr;
            uint gc = n_tile_base + nc;
            bool valid = (mr >= sub) && (mr < sub_end) && (gr < m_total) && (gc < n_out);
            if (valid) {
                uint o = gr * n_out + gc;
                out_gate[o] = bfloat(out_g[mr * 32u + nc]);
                out_up[o] = bfloat(out_u[mr * 32u + nc]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sub = sub_end;
    }
#endif
}
