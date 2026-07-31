// Hand-written NVFP4 MoE gather QMM: NAX-class geometry, TRUE BK=64 staging
// (BM=BN=BK=64, TPG=128). Stage A of the T22 NAX-64 port (see
// benchmarks/laguna-xs/README.md "T22, NAX gather QMM").
//
// GEOMETRY LABEL: this uses MPP `matmul2d` 32×32×32 × 4 SGs (2×2) as the
// cooperative-tensor MMA primitive; that is the public Metal surface for
// this class of hardware (MetalPerformancePrimitives `matmul2d`).
//
// WHAT CHANGED vs `butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16.metal` (BK=32,
// itself numerically/structurally identical to iron's
// `iron_nvfp4_moe_gather_qmm_bm64_mpp_bf16`): the K loop now advances 64
// columns per outer iteration instead of 32. Each 64-column K-tile stages
// TWO independent 64×32 half blocks (`Xs[0]`/`Ws[0]` cols [0,32),
// `Xs[1]`/`Ws[1]` cols [32,64)) inside ONE barrier pair, then issues both
// SK=32 matmul2d calls back-to-back before the next barrier pair. This
// amortizes dequant and halves the barrier count over BK=64 by staging
// once and walking two SK=32 sub-steps, using two physically-separate
// contiguous 32-wide staging blocks instead of a single stride-64 buffer
// sliced via `metal::tensor`: this repo has no confirmed strided/
// device-address-space `metal::tensor` constructor to verify against, so
// two contiguous blocks (same per-block layout as the already-proven
// BK=32 kernel) keeps every `tensor_inline` load in the exact shape
// that's already known to compile and dequant correctly.
//
// EXACTNESS: for a fixed (m0, n0) output tile, the sequence of matmul2d
// calls into `ct_c` happens at K offsets 0, 32, 64, 96, ... in that order,
// bit-identical call order to the BK=32 kernel's own K-loop (k0=0, then
// k0=32, etc., one matmul per k0). No accumulation reordering, no dequant
// formula change. The only behavioral difference is barrier count (halved)
// and staging burst size (doubled): a scheduling change, not a numerical
// one. Half of `Xs`/`Ws` (2×2048 halfs each) doubles threadgroup memory
// footprint vs the BK32 kernel (8 KiB → 16 KiB combined), well inside the
// 32 KiB/TG budget alongside `OutScratch` (16 KiB).
//
// Weight layout: iron/butter NVFP4 `[E, N, K/8] u32` + scales u8 E4M3 × global.
// Indices: RHS expert ids `[m_total]` with contiguous-run walk (reload W on change).
// Grid: [n_out/64, ceil(m_total/64), 1] threadgroups; TPG=128.
// Requires: n_out % 64 == 0, k_in % 64 == 0, block_size == 16.

#include <metal_stdlib>
using namespace metal;

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

// CARRY SIGN-FOLD E4M3 decode (unconditional, bit-exact for all 256 byte
// values, `sraw` always in [0,255] since callers widen from `device
// uchar*` -- see Tests/ButterTests/Ops/Nvfp4FastDequantTests.swift and
// the derivation in LagunaNvfp4Fuse.swift's `header`). Replaces the
// branch/exp2 sequence above with 5 branchless ops.
static inline float butter_nax64_nvfp4_scale(uint sraw, float global_scale) {
    ushort h = ushort((sraw + (sraw & 128u)) << 7);
    return float(as_type<half>(h)) * 256.0f * global_scale;
}

// T23 half-staging overflow fix (see benchmarks/laguna-xs/README.md "T23").
// Same fragility as every other MPP gather kernel in this family (see
// butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16.metal for the full writeup):
// staging dequantized weight AND raw activation into `half` tiles before
// `matmul2d` can round an individual A*B product to Inf once real
// late-layer (SwiGLU-amplified) activation magnitude crosses into the
// thousands, even though this restructuring's K-tile call order is
// bit-identical to the BK=32 kernel it replaces. Fold a fixed power-of-two
// bias into both staged operands before the half cast; compensate once at
// the f32 accumulator store. Exact (power-of-2 scale, constant across the
// whole K-sum, so it factors out cleanly): no additional approximation
// beyond ordinary half-precision rounding.
constant float kNvfp4StageBias = 0.00006103515625f;  // 2^-14, folded into both operands
constant float kNvfp4StageComp = 268435456.0f;        // 2^28 == 1/(bias*bias), applied once at output

static inline float butter_nax64_nvfp4_nibble(uint nib) {
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

kernel void butter_nvfp4_gather_qmm_rhs_nax64_mpp_bf16(
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
    const uint n_tile_base = tgid.x * 64u;
    const uint m_tile_base = tgid.y * 64u;
    const uint lane_in_tg = simd_group * 32u + simd_lane;
    const uint sg_m_base = (simd_group / 2u) * 32u;
    const uint sg_n_base = (simd_group & 1u) * 32u;

    // Two independent contiguous 64×32 half-blocks per K64 tile (see file
    // header). Xs[h]/Ws[h] is byte-for-byte the same 64×32 layout the BK=32
    // kernel uses for its single Xs/Ws.
    //
    // MEMORY BUDGET FIX (T22): Xs+Ws as separate `threadgroup half
    // [2][2048]` arrays alongside a separate `float OutScratch[4096]` sums
    // to 32768B of *explicit* threadgroup memory -- exactly the 32KB
    // compute-threadgroup-memory ceiling on Apple GPU families, with zero
    // headroom for whatever the MPP `matmul2d` cooperative-tensor machinery
    // needs internally. Root-caused via a per-layer NaN scan on the real
    // 39-layer Laguna forward pass: layers 1-37 were clean, corruption
    // (finite-but-wrong values, then NaN/Inf) started exactly at layer 38
    // once real accumulated-residual magnitude grew past what any isolated
    // kernel-level unit test (small synthetic activations) reached --
    // consistent with an over-budget/near-boundary threadgroup allocation
    // rather than an algorithmic bug (the K-tile matmul call order is
    // bit-identical to the already-proven BK=32 kernel this replaces).
    //
    // Fix: Xs/Ws are only live *before* the barrier that follows the K-loop;
    // OutScratch is only live *after* it. Alias Xs/Ws onto OutScratch's own
    // 16384-byte backing store instead of allocating them separately, so
    // total explicit threadgroup memory drops to 16384B (half of even the
    // BK=32 kernel's 24576B). `float` is 4-byte aligned, so every 2-byte
    // `half` slot carved out of it stays correctly aligned.
    threadgroup float OutScratch[4096]; // 4 × 32 × 32 (16384 bytes)
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

    // X staging: 128 lanes × 16 elems → 64 rows × 32 K (per half-block).
    const uint x_m_row = lane_in_tg / 2u;
    const uint x_k_base = (lane_in_tg & 1u) * 16u;

    // W dequant: 256 packs / 128 lanes = 2 packs/lane (BN=64, BK=32/half-block).
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

        for (uint k0 = 0u; k0 < k_in; k0 += 64u) {
            // --- Stage both 32-wide halves of this 64-wide K tile ---
            for (uint h = 0u; h < 2u; h++) {
                const uint kOff = k0 + h * 32u;

                // X tile (only rows in [sub, sub_end)).
                {
                    const uint gr = m_tile_base + x_m_row;
                    const bool in_run =
                        (x_m_row >= sub) && (x_m_row < sub_end) && (gr < m_total);
                    const uint x_dev = (in_run ? gr : 0u) * k_in + kOff + x_k_base;
                    const uint x_ws = x_m_row * 32u + x_k_base;
                    for (uint t = 0u; t < 16u; t++) {
                        float xv = 0.0f;
                        if (in_run) xv = float(x[x_dev + t]);
                        Xs[h][x_ws + t] = half(xv * kNvfp4StageBias);
                    }
                }

                // Dequant W tile for this expert / N-tile / K-half-slice.
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
                        butter_nax64_nvfp4_scale(uint(scales[sb_off]), global_scale);
                    const uint packed =
                        w[w_expert_pack + g_row * packs_per_row + k_pack];
                    const uint ws_base = w_row * 32u + pack_in_row * 8u;
                    for (uint j = 0u; j < 8u; j++) {
                        uint nib = (packed >> (j * 4u)) & 15u;
                        Ws[h][ws_base + j] =
                            half(butter_nax64_nvfp4_nibble(nib) * scale * kNvfp4StageBias);
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
