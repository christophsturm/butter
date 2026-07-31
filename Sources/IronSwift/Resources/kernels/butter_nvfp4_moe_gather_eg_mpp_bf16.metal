// w4/mpp-moegather: NVFP4 MoE gather QMM with the dequant FUSED into an MPP
// tensor-ops (`mpp::tensor_ops::matmul2d`) gather-GEMM, using EG-style
// ("Expert-Grouped") ownership -- each threadgroup owns exactly ONE
// expert's rows for exactly one 64-row sub-tile of that expert's run, never
// a mix of two experts' rows the way the flat-M-tile BM16/NAX/NAX64
// families can. See `LagunaNvfp4MppGather.swift` for the full design
// writeup (tile-plan prepass, why this avoids indirect dispatch, and how
// it composes with `Nvfp4GatherEG.expertOffsets`).
//
// GEOMETRY: identical MPP `matmul2d` 32x32x32 x 4 SGs (2x2) cooperative-
// tensor structure as `butter_dense_gemm_mpp_bf16.metal` (the F-85 dense
// prefill GEMM this mirrors) and
// `butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16.metal` (the existing fused-
// dequant MPP gather kernel this borrows its dequant math from
// byte-for-byte -- E4M3 carry-sign-fold scale decode, E2M1 nibble decode,
// T23 half-staging bias fold). BM=BN=64, BK=32, TPG=128.
//
// WHAT'S DIFFERENT FROM THE EXISTING NAX KERNEL: NAX dispatches a grid over
// the flat `[ceil(m_total/64)]` M-tile axis and walks a `while(sub<64)`
// loop INSIDE each threadgroup to find each expert-run boundary within
// that fixed 64-row window (a tile can and does straddle 2+ experts'
// runs whenever a run doesn't end on a 64-row seam). This kernel instead
// dispatches over a PRECOMPUTED tile-plan axis (`tile_expert` /
// `tile_row_start` / `tile_row_count`, built once per gather call by the
// `butter_nvfp4_moe_gather_eg_tileplan_u32` prepass -- see the Swift
// wrapper): tile `tgid.y` already IS one specific expert's rows
// `[tile_row_start, tile_row_start+tile_row_count)`, `tile_row_count <=
// 64`, no run-boundary scan needed inside the K-loop at all. The tradeoff
// this buys: strictly less per-tile control flow / no straddling logic,
// at the cost of the tile-plan prepass itself and a somewhat larger grid
// (`ceil(m_total/bm) + n_experts` tiles in the worst case, vs
// exactly `ceil(m_total/bm)` for the straddle-aware NAX approach -- see
// the Swift wrapper's `maxTiles(mTotal:nExperts:)` for the exact bound and
// its derivation). Whether this net wins over NAX's straddle-scan is an
// open empirical question this pass does not resolve; it ships opt-in
// (`BUTTER_NVFP4_MPP_GATHER=1`, default OFF) specifically so it can be
// benched against NAX/NAX64/BM16 without touching the default path.
//
// RAGGED SEGMENTS: `tile_row_count` (<=64, from the tile-plan prepass) is
// the number of VALID rows in this tile -- the last tile of any expert
// whose run length isn't a multiple of 64 is ragged by construction. Rows
// `[tile_row_count, 64)` are zero-filled into the X stage (never read past
// `tile_row_start + tile_row_count - 1` in `x`) and masked out of the
// final scatter-store, mirroring exactly how `butter_dense_gemm_mpp_bf16`
// masks its M boundary and NAX masks its per-sub-run boundary. A tile
// slot with `tile_expert == 0xFFFFFFFF` (padding beyond the last real
// tile, from the `maxTiles` over-provisioning) exits before touching any
// buffer at all.
//
// SCALE-GROUP BOUNDARY: `block_size` (16 for Poolside Laguna NVFP4) scale
// groups are HALF a BK=32 K-tile (2 packs of 8 elems = 1 group; a BK=32
// tile spans 4 packs = 2 groups). The per-PACK (not per-tile) scale
// lookup below (`(k0 + pack_in_row * 8) / block_size`) is what makes this
// correct across that boundary -- each of the 4 packs in a K-tile
// independently resolves its own group index, so a K-tile straddling a
// group-16 seam dequantizes the first 2 packs against one scale byte and
// the last 2 against the next, exactly matching the packed layout's own
// group structure. Reused verbatim from the NAX kernel, which already
// established this is correct (see that file's header + this repo's
// Nvfp4GatherNax64Tests.swift boundary-straddle coverage) -- not
// re-derived here.
//
// T23 STAGE FOLD (mandatory, unconditional): same `kNvfp4StageBias` /
// `kNvfp4StageComp` (2^-14 / 2^28) as every other MPP gather kernel in
// this family. SwiGLU-amplified activations reach magnitude 40-800 in
// real Laguna late-layer down_proj rows; folding this fixed power-of-two
// bias into BOTH staged half operands before the coop-tensor multiply
// (and undoing it once, exactly, via `kNvfp4StageComp == 1/bias^2`, at the
// f32 accumulator store) keeps the half-compute-type A*B product finite
// through that whole range without changing the result beyond ordinary
// half-precision rounding -- see `butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16.metal`'s
// header for the full empirical derivation of the constant.
//
// Requires (enforced by `LagunaNvfp4MppGather.eligible`, not re-checked
// here): n_out % 64 == 0, k_in % 32 == 0, block_size == 16, dtype bf16.

#include <metal_stdlib>
using namespace metal;

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

// Same fold constants every MPP gather/GEMM kernel in this repo uses (see
// `butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16.metal`'s T23 header for the
// full derivation): 2^-14 folded into both operands before the half cast,
// 2^28 == 1/bias^2 multiplied back once at store.
constant float kEgGatherStageBias = 0.00006103515625f;  // 2^-14
constant float kEgGatherStageComp = 268435456.0f;        // 2^28

// CARRY SIGN-FOLD E4M3 decode -- identical to
// `butter_nax_nvfp4_scale` in the NAX kernel, renamed to keep this file's
// symbols self-contained (each `_mpp_` kernel here is live-compiled from
// its own single-file `makeLibrary(source:)` call, so there is no actual
// link-time collision risk, but distinct names avoid any confusion when
// reading the two files side by side).
static inline float butter_eg_nvfp4_scale(uint sraw, float global_scale) {
    ushort h = ushort((sraw + (sraw & 128u)) << 7);
    return float(as_type<half>(h)) * 256.0f * global_scale;
}

static inline float butter_eg_nvfp4_nibble(uint nib) {
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

kernel void butter_nvfp4_moe_gather_eg_mpp_bf16(
    const device bfloat *x [[buffer(0)]],              // [m_total, k_in] gathered/sorted rows
    const device uint *w [[buffer(1)]],                 // [n_experts, n_out, k_in/8]
    const device uchar *scales [[buffer(2)]],            // [n_experts, n_out, k_in/block_size]
    const device uint *tile_expert [[buffer(3)]],        // [max_tiles] expert id, or 0xFFFFFFFF = padding
    const device uint *tile_row_start [[buffer(4)]],     // [max_tiles] row offset into x/out
    const device uint *tile_row_count [[buffer(5)]],     // [max_tiles] valid rows in this tile, 1...64
    device bfloat *out [[buffer(6)]],                    // [m_total, n_out]
    constant uint &n_out [[buffer(7)]],
    constant uint &k_in [[buffer(8)]],
    constant uint &block_size [[buffer(9)]],
    constant float &global_scale [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
    const uint tile_slot = tgid.y;
    const uint expert = tile_expert[tile_slot];
    if (expert == 0xFFFFFFFFu) return;  // padding tile past the real tile count: no rows owned, no work

    const uint row_start = tile_row_start[tile_slot];
    const uint row_count = tile_row_count[tile_slot];  // 1...64, this TG's single expert's rows only

    const uint n_tile_base = tgid.x * 64u;
    const uint lane_in_tg = simd_group * 32u + simd_lane;
    const uint sg_m_base = (simd_group / 2u) * 32u;
    const uint sg_n_base = (simd_group & 1u) * 32u;

    threadgroup half Xs[2048];           // 64 (rows) x 32 (K)
    threadgroup half Ws[2048];           // 64 (N) x 32 (K)
    threadgroup float OutScratch[4096];  // 4 x 32 x 32

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
    const uint w_expert_pack = expert * n_out * packs_per_row;
    const uint sb_expert_base = expert * n_out * groups_per_row;

    // X staging: 128 lanes x 16 elems -> 64 rows x 32 K, single expert, no
    // run-boundary scan needed (unlike NAX) -- every valid row in this
    // tile already belongs to `expert` by construction of the tile plan.
    const uint x_m_row = lane_in_tg / 2u;
    const uint x_k_base = (lane_in_tg & 1u) * 16u;

    // W dequant: 256 packs / 128 lanes = 2 packs/lane (BN=64, BK=32) --
    // same staging math as the NAX kernel (proven correct there).
    const uint packs_in_row = 4u;  // 32 / 8
    const uint packs_per_lane = 2u;

    for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) ct_c[i] = {};

    for (uint k0 = 0u; k0 < k_in; k0 += 32u) {
        // --- Stage X tile: this TG's row_count rows (single expert) ---
        {
            const bool in_run = x_m_row < row_count;
            const uint gr = row_start + (in_run ? x_m_row : 0u);
            const uint x_dev = gr * k_in + k0 + x_k_base;
            const uint x_ws = x_m_row * 32u + x_k_base;
            for (uint t = 0u; t < 16u; t++) {
                float xv = 0.0f;
                if (in_run) xv = float(x[x_dev + t]);
                Xs[x_ws + t] = half(xv * kEgGatherStageBias);
            }
        }

        // --- Dequant W tile for this expert / N-tile / K-slice ---
        for (uint i = 0u; i < packs_per_lane; i++) {
            const uint pack_id = lane_in_tg * packs_per_lane + i;
            const uint w_row = pack_id / packs_in_row;
            const uint pack_in_row = pack_id % packs_in_row;
            const uint g_row = n_tile_base + w_row;
            const uint k_pack = k0 / 8u + pack_in_row;
            // Per-PACK (not per-tile) scale group lookup -- see file
            // header "SCALE-GROUP BOUNDARY": correctly resolves each
            // pack's own group even when this K-tile straddles a
            // block_size=16 (2-pack) group seam.
            const uint sb_off =
                sb_expert_base + g_row * groups_per_row
                + ((k0 + pack_in_row * 8u) / block_size);
            const float scale =
                butter_eg_nvfp4_scale(uint(scales[sb_off]), global_scale);
            const uint packed =
                w[w_expert_pack + g_row * packs_per_row + k_pack];
            const uint ws_base = w_row * 32u + pack_in_row * 8u;
            for (uint j = 0u; j < 8u; j++) {
                uint nib = (packed >> (j * 4u)) & 15u;
                Ws[ws_base + j] =
                    half(butter_eg_nvfp4_nibble(nib) * scale * kEgGatherStageBias);
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

    // Store cooperative C -> TG scratch (per-SG 32x32 block). Both
    // operands were folded by `kEgGatherStageBias` above; undo once here
    // via `kEgGatherStageComp == 1/kEgGatherStageBias^2`.
    {
        metal::tensor<threadgroup float, metal::extents<int, 32, 32>, metal::tensor_inline>
            tC(OutScratch + simd_group * 1024u, metal::extents<int, 32, 32>{});
        ct_c.store(tC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter 64x64 from 2x2 SG tiles, masked to `row_count` valid rows.
    // n_out % 64 == 0 is a hard eligibility requirement (see file header),
    // so the N side never needs masking, matching
    // `butter_dense_gemm_mpp_bf16`'s own M-only masking convention.
    for (uint i = 0u; i < 32u; i++) {
        const uint flat = lane_in_tg * 32u + i;
        const uint mr = flat / 64u;
        const uint nc = flat % 64u;
        if (mr < row_count) {
            const uint gr = row_start + mr;
            const uint gc = n_tile_base + nc;
            const uint sg = (mr / 32u) * 2u + (nc / 32u);
            const uint local = sg * 1024u + (mr % 32u) * 32u + (nc % 32u);
            out[gr * n_out + gc] = bfloat(OutScratch[local] * kEgGatherStageComp);
        }
    }
#endif
}
