// Copyright 2026 Tom Turney (@TheTom)
//
// F-85 prefill decisive experiment: dense (non-gather, non-quantized) bf16
// GEMM via MPP `mpp::tensor_ops::matmul2d`, at the shipped steel simdgroup
// MMA kernel's exact output-tile granularity (BM=BN=64, TPG=128, 2x2 SG
// grid of 32x32x32 matmul2d calls), so it can be A/B-benchmarked directly
// against `iron_steel_gemm_64x64x16_2x2_bf16` at real Laguna prefill
// projection shapes. Pure microbench infrastructure -- not wired into any
// model forward path.
//
// GEOMETRY LABEL: this is the same MPP `matmul2d` 32x32x32 x 4 SGs (2x2)
// cooperative-tensor structure every existing hand-written `_mpp_*.metal`
// kernel in this repo uses (see e.g.
// `butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16.metal`), stripped of the
// NVFP4 dequant + expert-gather machinery: this is the DENSE case those
// kernels' K-loop and store/scatter structure already generalizes to.
//
// SEMANTICS -- matches Butter's `Linear` convention exactly: out[M,N] =
// X[M,K] @ W^T, W stored [N,K] row-major (K/inDim contiguous). That is
// the SAME layout real Q/K/V/O/gate/up/down weights live in (Butter
// `Linear.weight` is `[outDim, inDim]`), so `b` here takes an
// UNTRANSPOSED Linear weight directly -- unlike
// `iron_steel_gemm_64x64x16_2x2_bf16`, which needs the weight
// pre-transposed to `[K,N]` first (see `LagunaSteelGemm.transposeWeightBf16`).
// `tb=true` in the `matmul2d_descriptor` below is what buys this: MPP
// transposes the right operand internally, so staging directly from
// `[N,K]` needs no host-side or on-device transpose pass at all -- a
// real structural difference between the two kernel families worth
// noting for whoever reads the bench numbers (steel pays a one-time
// transpose at weight-load time that this MPP path never needs).
//
// COOPERATIVE-TENSOR OPERAND DTYPE: staged into `half`, matching every
// other MPP `matmul2d` kernel in this repo (all 8 existing `_mpp_*.metal`
// kernels use `half,half,float` -- confirmed via
// `grep -n "get_left_input_cooperative_tensor" Sources/IronSwift/Resources/kernels/*_mpp_*.metal`).
// Whether `mpp::tensor_ops::matmul2d` accepts `bfloat` cooperative-tensor
// operands directly is an OPEN QUESTION in this repo's own history --
// see `benchmarks/laguna-xs/README.md` T25 ("not verified this
// session... an open question whether MPP's matmul2d on this
// hardware/Metal-version accepts bfloat cooperative-tensor operands the
// same way it accepts half"). T33's "bfloat staging needs no fold"
// finding was demonstrated on a DIFFERENT Metal surface -- the steel
// kernel's raw `simdgroup_matrix<bfloat,8,8>` MMA path, not
// `mpp::tensor_ops::matmul2d`'s cooperative tensor. This session cannot
// build/run (`swift build`/`swift test` are off-limits -- a separate
// validation lane compiles and runs this), so risking an unverified
// operand type on the one class of kernel that MUST live-compile against
// the running OS's MPP header (see `PSOCache.liveCompileMppFunction`)
// is the wrong tradeoff for a correctness-gated decisive experiment.
// Staying on `half`, the only operand dtype every existing MPP kernel in
// this repo has actually compiled and run correctly with, is the
// lower-risk choice. (A `bfloat`-cooperative-tensor variant is a natural
// follow-up once this file is confirmed to build -- flagged in the PR
// description, not attempted here.)
//
// NO T23 magnitude-bias fold: that fold exists to survive NVFP4-dequant's
// pathological post-scale magnitude range (observed non-finite past
// activation magnitude ~50 without it, see T23 in the README). This
// kernel stages plain bf16 activations/weights at ordinary transformer
// magnitude -- post-RMSNorm activations and trained projection weights,
// O(1)-O(10), nowhere near half's ~65504 ceiling -- so the fold's
// overflow protection has nothing to protect against here and would only
// cost precision (subnormal crowding) for zero benefit. See T33's own
// "bfloat staging needs no fold" finding for the same reasoning applied
// to the steel path.
//
// Grid: [N/64, ceil(M/64), 1] threadgroups; TPG=128. Requires N % 64 ==
// 0 and K % 32 == 0 (both required for the `Bs`/matmul2d tiling below to
// stay in-bounds; there is no N or K masking, only M is boundary-masked).
// All shapes this experiment benchmarks (Laguna XS QKV/O/L00-FFN
// projections) satisfy N%64==0 and K%32==0 exactly; M boundary-masking
// exists so the correctness tests can probe a non-64-multiple M without
// a second kernel variant.

#include <metal_stdlib>
using namespace metal;

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

kernel void butter_dense_gemm_mpp_bf16(
    const device bfloat *a [[buffer(0)]],   // [M, K] row-major, K contiguous
    const device bfloat *b [[buffer(1)]],   // [N, K] row-major, K contiguous (Linear.weight layout)
    device bfloat *out [[buffer(2)]],       // [M, N] row-major
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
    // BM=BN=64, BK=32 staging, 4 SGs in 2x2 of 32x32 matmul2d tiles --
    // identical geometry to the proven NAX BK32 gather kernel this is
    // modeled on.
    const uint n_tile_base = tgid.x * 64u;
    const uint m_tile_base = tgid.y * 64u;
    const uint lane_in_tg = simd_group * 32u + simd_lane;
    const uint sg_m_base = (simd_group / 2u) * 32u;
    const uint sg_n_base = (simd_group & 1u) * 32u;

    threadgroup half As[2048];          // 64 (M) x 32 (K)
    threadgroup half Bs[2048];          // 64 (N) x 32 (K)
    threadgroup float OutScratch[4096]; // 4 x 32 x 32

    constexpr auto gemm_desc = mpp::tensor_ops::matmul2d_descriptor(
        /*M=*/32, /*N=*/32, /*K=*/32,
        /*ta=*/false, /*tb=*/true, /*tc=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<gemm_desc, metal::execution_simdgroup> gemm_op;
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    // A staging: 128 lanes x 16 elems -> 64 rows x 32 K.
    const uint a_m_row = lane_in_tg / 2u;
    const uint a_k_base = (lane_in_tg & 1u) * 16u;
    // B staging: same index math, N-major instead of M-major (both are
    // "64 rows x 32 K" tiles read straight off a [rows, K] row-major
    // device array -- A off `a` indexed by M, B off `b` indexed by N).
    const uint b_n_row = lane_in_tg / 2u;
    const uint b_k_base = (lane_in_tg & 1u) * 16u;

    for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) ct_c[i] = {};

    for (uint k0 = 0u; k0 < K; k0 += 32u) {
        // --- Stage A tile: rows [m_tile_base, m_tile_base+64), K-slice [k0, k0+32) ---
        {
            const uint gr = m_tile_base + a_m_row;
            const bool in_range = gr < M;
            const uint a_dev = (in_range ? gr : 0u) * K + k0 + a_k_base;
            const uint a_ws = a_m_row * 32u + a_k_base;
            for (uint t = 0u; t < 16u; t++) {
                float av = 0.0f;
                if (in_range) av = float(a[a_dev + t]);
                As[a_ws + t] = half(av);
            }
        }

        // --- Stage B tile: rows [n_tile_base, n_tile_base+64), K-slice [k0, k0+32) ---
        // N%64==0 is a hard kernel requirement (see file header), so
        // `gc < N` is always true here -- no masking needed, unlike A's
        // M-boundary check.
        {
            const uint gc = n_tile_base + b_n_row;
            const uint b_dev = gc * K + k0 + b_k_base;
            const uint b_ws = b_n_row * 32u + b_k_base;
            for (uint t = 0u; t < 16u; t++) {
                Bs[b_ws + t] = half(float(b[b_dev + t]));
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            metal::tensor<threadgroup half, metal::extents<int, 32, 32>, metal::tensor_inline>
                tA(As + sg_m_base * 32u, metal::extents<int, 32, 32>{});
            metal::tensor<threadgroup half, metal::extents<int, 32, 32>, metal::tensor_inline>
                tB(Bs + sg_n_base * 32u, metal::extents<int, 32, 32>{});
            ct_a.load(tA);
            ct_b.load(tB);
            gemm_op.run(ct_a, ct_b, ct_c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store cooperative C -> TG scratch (per-SG 32x32 block).
    {
        metal::tensor<threadgroup float, metal::extents<int, 32, 32>, metal::tensor_inline>
            tC(OutScratch + simd_group * 1024u, metal::extents<int, 32, 32>{});
        ct_c.store(tC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter 64x64 from 2x2 SG tiles. Each of 128 lanes writes 32 elems.
    // N%64==0 means `gc < N` always holds; `gr < M` is the only real mask.
    for (uint i = 0u; i < 32u; i++) {
        const uint flat = lane_in_tg * 32u + i;
        const uint mr = flat / 64u;
        const uint nc = flat % 64u;
        const uint gr = m_tile_base + mr;
        const uint gc = n_tile_base + nc;
        if (gr < M) {
            const uint sg = (mr / 32u) * 2u + (nc / 32u);
            const uint local = sg * 1024u + (mr % 32u) * 32u + (nc % 32u);
            out[gr * N + gc] = bfloat(OutScratch[local]);
        }
    }
#endif
}
