// Hand-written NVFP4 MoE gather QMM: NAX-class geometry (BM=BN=64, TPG=128).
//
// GEOMETRY LABEL: this is MPP `matmul2d` 32×32×32 × 4 SGs (2×2), the
// cooperative-tensor MMA primitive Metal exposes on this hardware class
// via MetalPerformancePrimitives. Geometry matches the board's 64-wide
// gather QMM shape (64³ / 128 thr).
//
// Weight layout: iron/butter NVFP4 `[E, N, K/8] u32` + scales u8 E4M3 × global.
// Indices: RHS expert ids `[m_total]` with contiguous-run walk (reload W on change).
// Grid: [n_out/64, ceil(m_total/64), 1] threadgroups; TPG=128.
// Requires: n_out % 64 == 0, k_in % 32 == 0, block_size == 16.

#include <metal_stdlib>
using namespace metal;

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

static inline float butter_nax_nvfp4_scale(uint sraw, float global_scale) {
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

// T23 half-staging overflow fix (see benchmarks/laguna-xs/README.md "T23").
// All MPP gather kernels in this family stage dequantized NVFP4 weight
// values AND raw activation values into Metal `half` tiles before the
// `matmul2d` call. Both operands individually fit in half's ~65504 ceiling
// for typical magnitudes, but MPP's half-compute-type multiply can still
// round an individual A*B product to Inf once real late-layer residual
// magnitude (SwiGLU-amplified activation entering down_proj) crosses into
// the thousands -- confirmed via a real-weight magnitude sweep
// (Tests/ButterTests/Ops/Nvfp4GatherNax64MagnitudeTests.swift) that goes
// non-finite around activation magnitude ~50 while the f32 GEMV oracle
// stays finite past 1600.
//
// Fix: fold a constant power-of-two bias into BOTH staged operands before
// the half cast, and multiply the compensation back into the f32
// accumulator once at the final store. Because the bias is a fixed
// constant (not data/group dependent), it factors out of the entire K-sum
// exactly -- `sum_k((a_k*B) * (b_k*B)) == B^2 * sum_k(a_k*b_k)` -- so this
// is bit-for-bit equivalent to the unbiased computation up to ordinary
// half-precision rounding (no additional approximation), the only real
// risk being subnormal precision loss for already-tiny operands.
//
// kNvfp4StageBias = 2^-14 was reached empirically, not by a priori
// derivation, and by balancing TWO gates, not one: the real-weight
// magnitude sweep (Tests/ButterTests/Ops/Nvfp4GatherNax64MagnitudeTests.swift,
// requiring zero NaN/Inf + cos>0.9999 vs the f32 GEMV oracle up to
// activation magnitude 800) AND the pre-existing real-Laguna-weight tests
// in Nvfp4GatherNax64Tests.swift (gate/down/sort-plan, real trained
// weights at normal/typical activation magnitude, also gated at
// cos>0.9999). These two gates pull in opposite directions: bias small
// enough to preserve full precision at typical magnitude (e.g. 2^-12)
// still overflows past activation magnitude ~400-800; bias big enough to
// stay finite at magnitude 1600 (2^-15 or higher) pushes the real
// gate/down weights' typically-small dequantized values into half's
// subnormal range and drops the pre-existing real-weight tests from
// cos~1.0 to ~0.9998 -- a real regression, not just a synthetic-sweep
// concern. 2^-14 is the largest bias (most overflow headroom) that still
// keeps every real-weight test at cos~1.0: finite + cos>0.9999 through
// magnitude 800 (16x T22's observed ~50 cliff), zero regression on
// existing real-weight coverage. A SINGLE fixed bias cannot cover
// unbounded activation magnitude without this tradeoff -- full coverage
// needs per-tile/per-magnitude adaptive scaling (Option C in the T23
// plan: f32 tiles for late layers), out of scope for this pass.
constant float kNvfp4StageBias = 0.00006103515625f;  // 2^-14, folded into both operands
constant float kNvfp4StageComp = 268435456.0f;        // 2^28 == 1/(bias*bias), applied once at output

static inline float butter_nax_nvfp4_nibble(uint nib) {
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

kernel void butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16(
    const device bfloat *x [[buffer(0)]],
    const device uint *w [[buffer(1)]],
    const device uchar *scales [[buffer(2)]],
    const device uint *indices [[buffer(3)]],
    device bfloat *out [[buffer(4)]],
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
    // BM=BN=64, BK=32 staging, 4 SGs in 2×2 of 32×32 matmul2d tiles.
    const uint n_tile_base = tgid.x * 64u;
    const uint m_tile_base = tgid.y * 64u;
    const uint lane_in_tg = simd_group * 32u + simd_lane;
    const uint sg_m_base = (simd_group / 2u) * 32u;
    const uint sg_n_base = (simd_group & 1u) * 32u;

    threadgroup half Xs[2048];       // 64 × 32
    threadgroup half Ws[2048];       // 64 × 32
    threadgroup float OutScratch[4096]; // 4 × 32 × 32

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

    // X staging: 128 lanes × 16 elems → 64 rows × 32 K.
    const uint x_m_row = lane_in_tg / 2u;
    const uint x_k_base = (lane_in_tg & 1u) * 16u;

    // W dequant: 256 packs / 128 lanes = 2 packs/lane (BN=64, BK=32).
    const uint packs_in_row = 4u; // 32 / 8
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

        for (uint k0 = 0u; k0 < k_in; k0 += 32u) {
            // --- Load X tile (only rows in [sub, sub_end)) ---
            {
                const uint gr = m_tile_base + x_m_row;
                const bool in_run =
                    (x_m_row >= sub) && (x_m_row < sub_end) && (gr < m_total);
                const uint x_dev = (in_run ? gr : 0u) * k_in + k0 + x_k_base;
                const uint x_ws = x_m_row * 32u + x_k_base;
                for (uint t = 0u; t < 16u; t++) {
                    float xv = 0.0f;
                    if (in_run) xv = float(x[x_dev + t]);
                    Xs[x_ws + t] = half(xv * kNvfp4StageBias);
                }
            }

            // --- Dequant W tile for this expert / N-tile / K-slice ---
            for (uint i = 0u; i < packs_per_lane; i++) {
                const uint pack_id = lane_in_tg * packs_per_lane + i;
                const uint w_row = pack_id / packs_in_row;
                const uint pack_in_row = pack_id % packs_in_row;
                const uint g_row = n_tile_base + w_row;
                const uint k_pack = k0 / 8u + pack_in_row;
                const uint sb_off =
                    sb_expert_base + g_row * groups_per_row
                    + ((k0 + pack_in_row * 8u) / block_size);
                const float scale =
                    butter_nax_nvfp4_scale(uint(scales[sb_off]), global_scale);
                const uint packed =
                    w[w_expert_pack + g_row * packs_per_row + k_pack];
                const uint ws_base = w_row * 32u + pack_in_row * 8u;
                for (uint j = 0u; j < 8u; j++) {
                    uint nib = (packed >> (j * 4u)) & 15u;
                    Ws[ws_base + j] =
                        half(butter_nax_nvfp4_nibble(nib) * scale * kNvfp4StageBias);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                metal::tensor<threadgroup half, metal::extents<int, 32, 32>, metal::tensor_inline>
                    tA(Xs + sg_m_base * 32u, metal::extents<int, 32, 32>{});
                metal::tensor<threadgroup half, metal::extents<int, 32, 32>, metal::tensor_inline>
                    tB(Ws + sg_n_base * 32u, metal::extents<int, 32, 32>{});
                ct_a.load(tA);
                ct_b.load(tB);
                gemm_op.run(ct_a, ct_b, ct_c);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store cooperative C → TG scratch (per-SG 32×32 block).
        {
            metal::tensor<threadgroup float, metal::extents<int, 32, 32>, metal::tensor_inline>
                tC(OutScratch + simd_group * 1024u, metal::extents<int, 32, 32>{});
            ct_c.store(tC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scatter 64×64 from 2×2 SG tiles. Each of 128 lanes writes 32 elems.
        for (uint i = 0u; i < 32u; i++) {
            const uint flat = lane_in_tg * 32u + i;
            const uint mr = flat / 64u;
            const uint nc = flat % 64u;
            const uint gr = m_tile_base + mr;
            const uint gc = n_tile_base + nc;
            const bool valid =
                (mr >= sub) && (mr < sub_end) && (gr < m_total) && (gc < n_out);
            if (valid) {
                // SG layout: m-half = mr/32, n-half = nc/32 → sg = m*2 + n
                const uint sg = (mr / 32u) * 2u + (nc / 32u);
                const uint local =
                    sg * 1024u + (mr % 32u) * 32u + (nc % 32u);
                out[gr * n_out + gc] = bfloat(OutScratch[local] * kNvfp4StageComp);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sub = sub_end;
    }
#endif
}
