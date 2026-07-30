// Hand-written NVFP4 MoE gather QMM: NAX-class geometry, TRUE BK=64 staging.
// f32 activation path (late-layer f32 promote). Same MPP matmul2d body as
// bf16; see butter_nvfp4_gather_qmm_rhs_nax64_mpp_bf16.metal for the BK=64
// staging design notes (two contiguous 64×32 half-blocks per K64 tile, one
// barrier pair, two back-to-back SK=32 matmul2d calls).

#include <metal_stdlib>
using namespace metal;

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

static inline float butter_nax64_nvfp4_scale_f(uint sraw, float global_scale) {
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

// T23 half-staging overflow fix, see butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16.metal
// for the full writeup (benchmarks/laguna-xs/README.md "T23"). Fold a fixed
// power-of-two bias into both staged operands before the half cast;
// compensate once at the f32 accumulator store. Exact (power-of-2 scale),
// factors cleanly out of the whole K-sum since the bias is a constant.
constant float kNvfp4StageBias = 0.00006103515625f;  // 2^-14, folded into both operands
constant float kNvfp4StageComp = 268435456.0f;        // 2^28 == 1/(bias*bias), applied once at output

static inline float butter_nax64_nvfp4_nibble_f(uint nib) {
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

kernel void butter_nvfp4_gather_qmm_rhs_nax64_mpp_f32(
    const device float *x [[buffer(0)]],
    const device uint *w [[buffer(1)]],
    const device uchar *scales [[buffer(2)]],
    const device uint *indices [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant uint &m_total [[buffer(5)]],
    constant uint &n_out [[buffer(6)]],
    constant uint &k_in [[buffer(7)]],
    constant uint &block_size [[buffer(8)]],
    constant float &global_scale [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
    const uint n_tile_base = tgid.x * 64u;
    const uint m_tile_base = tgid.y * 64u;
    const uint lane_in_tg = simd_group * 32u + simd_lane;
    const uint sg_m_base = (simd_group / 2u) * 32u;
    const uint sg_n_base = (simd_group & 1u) * 32u;

    // MEMORY BUDGET FIX (T22): see butter_nvfp4_gather_qmm_rhs_nax64_mpp_bf16.metal
    // for the full root-cause writeup (per-layer NaN scan on the real
    // 39-layer Laguna forward: clean through layer 37, corruption starting
    // exactly at layer 38 once real residual magnitude exceeded what any
    // synthetic unit test reached -- consistent with the original separate
    // Xs[2][2048]+Ws[2][2048]+OutScratch[4096] declaration sitting exactly
    // at the 32KB compute-threadgroup-memory ceiling with zero headroom for
    // MPP's own internal needs). Alias Xs/Ws onto OutScratch's backing
    // store instead (disjoint lifetimes, separated by the barrier after the
    // K-loop) -- total explicit threadgroup memory drops from 32768B to
    // 16384B.
    threadgroup float OutScratch[4096];
    threadgroup half* const XsWsBase = (threadgroup half*)OutScratch;
    threadgroup half* const Xs[2] = {XsWsBase, XsWsBase + 2048};
    threadgroup half* const Ws[2] = {XsWsBase + 4096, XsWsBase + 6144};

    constexpr auto gemm_desc = mpp::tensor_ops::matmul2d_descriptor(
        /*M=*/32, /*N=*/32, /*K=*/32,
        /*ta=*/false, /*tb=*/true, /*tc=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<gemm_desc, metal::execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    const uint packs_per_row = k_in / 8u;
    const uint groups_per_row = k_in / block_size;
    const uint x_m_row = lane_in_tg / 2u;
    const uint x_k_base = (lane_in_tg & 1u) * 16u;
    const uint packs_in_row = 4u;
    const uint packs_per_lane = 2u;

    uint sub = 0u;
    while (sub < 64u) {
        const uint cur_row = m_tile_base + sub;
        const bool cur_ok = cur_row < m_total;
        const uint cur_expert = cur_ok ? indices[cur_row] : 0xFFFFFFFFu;
        uint sub_end = 64u;
        for (uint p = sub + 1u; p < 64u; p++) {
            const uint pr = m_tile_base + p;
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

        const uint w_expert_pack = cur_expert * n_out * packs_per_row;
        const uint sb_expert_base = cur_expert * n_out * groups_per_row;
        for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) ct_c[i] = {};

        for (uint k0 = 0u; k0 < k_in; k0 += 64u) {
            for (uint h = 0u; h < 2u; h++) {
                const uint kOff = k0 + h * 32u;

                {
                    const uint gr = m_tile_base + x_m_row;
                    const bool in_run =
                        (x_m_row >= sub) && (x_m_row < sub_end) && (gr < m_total);
                    const uint x_dev = (in_run ? gr : 0u) * k_in + kOff + x_k_base;
                    const uint x_ws = x_m_row * 32u + x_k_base;
                    for (uint t = 0u; t < 16u; t++) {
                        float xv = 0.0f;
                        if (in_run) xv = x[x_dev + t];
                        Xs[h][x_ws + t] = half(xv * kNvfp4StageBias);
                    }
                }
                for (uint i = 0u; i < packs_per_lane; i++) {
                    const uint pack_id = lane_in_tg * packs_per_lane + i;
                    const uint w_row = pack_id / packs_in_row;
                    const uint pack_in_row = pack_id % packs_in_row;
                    const uint g_row = n_tile_base + w_row;
                    const uint k_pack = kOff / 8u + pack_in_row;
                    const uint sb_off =
                        sb_expert_base + g_row * groups_per_row
                        + ((kOff + pack_in_row * 8u) / block_size);
                    const float scale =
                        butter_nax64_nvfp4_scale_f(uint(scales[sb_off]), global_scale);
                    const uint packed =
                        w[w_expert_pack + g_row * packs_per_row + k_pack];
                    const uint ws_base = w_row * 32u + pack_in_row * 8u;
                    for (uint j = 0u; j < 8u; j++) {
                        uint nib = (packed >> (j * 4u)) & 15u;
                        Ws[h][ws_base + j] =
                            half(butter_nax64_nvfp4_nibble_f(nib) * scale * kNvfp4StageBias);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                metal::tensor<threadgroup half, metal::extents<int, 32, 32>, metal::tensor_inline>
                    tA0(Xs[0] + sg_m_base * 32u, metal::extents<int, 32, 32>{});
                metal::tensor<threadgroup half, metal::extents<int, 32, 32>, metal::tensor_inline>
                    tB0(Ws[0] + sg_n_base * 32u, metal::extents<int, 32, 32>{});
                ct_a.load(tA0);
                ct_b.load(tB0);
                gemm_op.run(ct_a, ct_b, ct_c);

                metal::tensor<threadgroup half, metal::extents<int, 32, 32>, metal::tensor_inline>
                    tA1(Xs[1] + sg_m_base * 32u, metal::extents<int, 32, 32>{});
                metal::tensor<threadgroup half, metal::extents<int, 32, 32>, metal::tensor_inline>
                    tB1(Ws[1] + sg_n_base * 32u, metal::extents<int, 32, 32>{});
                ct_a.load(tA1);
                ct_b.load(tB1);
                gemm_op.run(ct_a, ct_b, ct_c);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        {
            metal::tensor<threadgroup float, metal::extents<int, 32, 32>, metal::tensor_inline>
                tC(OutScratch + simd_group * 1024u, metal::extents<int, 32, 32>{});
            ct_c.store(tC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0u; i < 32u; i++) {
            const uint flat = lane_in_tg * 32u + i;
            const uint mr = flat / 64u;
            const uint nc = flat % 64u;
            const uint gr = m_tile_base + mr;
            const uint gc = n_tile_base + nc;
            const bool valid =
                (mr >= sub) && (mr < sub_end) && (gr < m_total) && (gc < n_out);
            if (valid) {
                const uint sg = (mr / 32u) * 2u + (nc / 32u);
                const uint local =
                    sg * 1024u + (mr % 32u) * 32u + (nc % 32u);
                out[gr * n_out + gc] = OutScratch[local] * kNvfp4StageComp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sub = sub_end;
    }
#endif
}
