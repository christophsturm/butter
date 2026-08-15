// Copyright 2026 Tom Turney (@TheTom)
//
// T33 stage 1: dense (single-expert) NVFP4 GEMM using the steel MMA
// structure (see iron_steel_gemm_64x64x16_2x2_bf16.metal, T26): raw
// simdgroup_matrix<bfloat,8,8> multiply-accumulate with BOTH operands'
// per-lane thread_elements() filled directly from device memory -- no
// threadgroup staging buffer, no threadgroup_barrier, for either operand.
// This mirrors T25's structural note for closing the NVFP4 gather-QMM
// kernel-efficiency gap: the shipped bm16 MPP gather kernel
// (butter_nvfp4_moe_gather_bm16_mpp_xrows_bf16.metal) stages BOTH the
// dequantized weight AND the raw activation into `half` tiles in
// threadgroup memory before calling mpp::tensor_ops::matmul2d, 16-wide K
// per step with 2 barriers/step (256 barriers at K=2048) -- the exact
// fragility that needed the 2^-14 exponent-bias fold (T23) to avoid Inf
// on late-layer activation magnitude. Staging into `bfloat` instead of
// `half` keeps the full f32-width exponent range, so that overflow class
// structurally cannot occur (empirically verified by the magnitude-ramp
// test, not merely assumed) and the fold is not needed here.
//
// The only new work vs the dense steel GEMM is the B-operand fetch:
// instead of a plain bfloat device read, each element is dequantized
// on the fly from the NVFP4 packed layout (StackedNvfp4Experts, see
// OpsNVFP4.swift): `w` is u32 `[n, k/8]` (4-bit E2M1 codes, 8 nibbles/word),
// `scales` is u8 `[n, k/block_size]` (E4M3 micro-scale, block_size=16),
// multiplied by a single f32 `global_scale`. No threadgroup write for
// either operand -- the dequant result goes straight into
// `thread_elements()`, same zero-staging discipline as the A fetch.
//
// Stage 1 is DENSE: exactly one logical expert across the whole `[m,k]`
// input (single weight/scales base, no per-row expert lookup, no ragged-
// segment masking). Stage 2 (butter_nvfp4_gather_steel_bgemm_bf16.metal)
// adds the per-row expert-run masking for the real sorted-gather MoE path.
// Kept as its own kernel/test/commit per the T33 staged plan so Stage 1's
// dequant-into-MMA correctness is validated in isolation first.
//
// No bounds checking (same contract as the dense steel GEMM): m, n, k
// must be exact multiples of 64/64/16 respectively, k %% block_size == 0.
// Named with a `bgemm` substring so PSOCache live-compiles it from this
// source against the running OS's Metal toolchain (same mechanism the
// NAX/NAX64 MPP kernels use) -- fastest iteration path, no offline
// `iron build --emit all` metallib regen required. This kernel does NOT
// use mpp::tensor_ops (plain simdgroup_matrix, a stable non-MPP-header
// API) so it does not strictly need runtime-header-skew protection, but
// piggybacking on the live-compile path keeps overlay .metal iteration
// fast per the T33 mission brief; the AOT-baked steel dense GEMM
// (iron_steel_gemm_64x64x16_2x2_bf16, no MPP-pattern name) shows the
// non-live-compiled route works too if this ever needs to move to the
// prebuilt metallib.
//
// T34 update: the B-operand (NVFP4 dequant) depends only on (tj, K), never
// on `ti` (the 4 row-subtiles within a 32x32 simdgroup output block). T33's
// original loop nested `ti` outermost, so the dequant was redundantly
// recomputed 4x per K-step -- free to eat for a plain device read (the
// dense steel GEMM this structure was copied from) but real ALU cost for
// NVFP4 (nibble unpack + E4M3 scale decode), measured as a 3.4-4.0x
// slowdown vs the shipped mpp default. The loop below is `tj` outer, `ti`
// inner, with 4 parallel per-lane accumulator registers so each B element
// is dequantized once per (tj, K-step) and reused across all 4 `ti`
// iterations -- still zero threadgroup memory, zero barriers.
#include <metal_stdlib>
using namespace metal;
#include <metal_simdgroup_matrix>

// E4M3 micro-scale decode (sign:1, exp:4 bias-7, mantissa:3), matches the
// shipped bm16 MPP gather kernel exactly (butter_nvfp4_moe_gather_bm16_mpp_xrows_bf16.metal).
// CARRY SIGN-FOLD E4M3 decode (unconditional, bit-exact for all 256 byte
// values -- see Tests/ButterTests/Ops/Nvfp4FastDequantTests.swift and the
// derivation in LagunaNvfp4Fuse.swift's `header`). Replaces the
// mask/shift/select ~7-op sequence above with 5 ops.
inline float nvfp4_decode_scale_byte(uchar raw, float global_scale) {
    ushort h = ushort((uint(raw) + (uint(raw) & 128u)) << 7);
    return float(as_type<half>(h)) * 256.0f * global_scale;
}

// E2M1 4-bit code -> magnitude LUT {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign in bit 3.
inline float nvfp4_decode_nibble(uint nib, float scale) {
    uint mag_idx = nib & 0x7u;
    float mag = select(6.0f, 4.0f, mag_idx < 7u);
    mag = select(mag, 3.0f, mag_idx < 6u);
    mag = select(mag, 2.0f, mag_idx < 5u);
    mag = select(mag, 1.5f, mag_idx < 4u);
    mag = select(mag, 1.0f, mag_idx < 3u);
    mag = select(mag, 0.5f, mag_idx < 2u);
    mag = select(mag, 0.0f, mag_idx < 1u);
    bool negative = (nib & 0x8u) != 0u;
    return select(mag, -mag, negative) * scale;
}

kernel void butter_nvfp4_gather_steel_bgemm_dense_bf16(
    const device bfloat *a       [[buffer(0)]],   // [m, k] row-major
    const device uint   *w       [[buffer(1)]],   // [n, k/8] packed NVFP4
    const device uchar  *scales  [[buffer(2)]],   // [n, k/block_size] e4m3
    device bfloat       *out     [[buffer(3)]],   // [m, n] row-major
    [[maybe_unused]] constant uint &m [[buffer(4)]],
    constant uint &n                [[buffer(5)]],
    constant uint &k                [[buffer(6)]],
    constant uint &block_size      [[buffer(7)]],
    constant float &global_scale   [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    simdgroup_matrix<bfloat, 8, 8> subA;
    simdgroup_matrix<bfloat, 8, 8> subB;

    // 2x2 simdgroup grid per threadgroup (4 SGs), each owns a 32x32 output
    // sub-block built from 4x4 8x8 MMA tiles. Identical layout/derivation
    // to iron_steel_gemm_64x64x16_2x2_bf16.metal (T26, byte-identical-
    // verified correctness) -- reused verbatim, not re-derived, since this
    // is Apple's documented-by-precedent per-lane float8x8 fragment
    // mapping and re-deriving it risks a silent transposition bug.
    const uint wn = 2u;
    const uint subM = 32u, subN = 32u;
    const uint tileFm = subM / 8u, tileFn = subN / 8u;

    uint quad = simd_lane / 4u;
    uint fm = (quad & 4u) + ((simd_lane / 2u) % 4u);
    uint fn0 = ((quad & 2u) * 2u) + ((simd_lane % 2u) * 2u);
    uint fn1 = fn0 + 1u;

    const uint bm = 64u, bn = 64u;
    uint sgRow0 = (simd_group / wn) * subM;
    uint sgCol0 = (simd_group % wn) * subN;
    uint blockM0 = tid.y * bm + sgRow0;
    uint blockN0 = tid.x * bn + sgCol0;

    uint packsPerRow = k / 8u;
    uint groupsPerRow = k / block_size;

    // T34: loop order is `tj` outer, `ti` inner, with `tileFm` (4) parallel
    // accumulators held in per-lane registers. The B-operand (NVFP4-
    // dequantized weight) depends only on (tj, K), never on `ti` -- T33's
    // original `ti`-outer nesting re-fetched and re-dequantized the same
    // weight element 4x per K-step (once per row-subtile), which is free to
    // eat for a plain device read (the dense steel GEMM this kernel's
    // structure was copied from) but not for NVFP4 dequant (nibble unpack +
    // E4M3 scale decode is real ALU work). Hoisting the dequant out of the
    // `ti` loop and reusing it across all 4 row-subtiles removes that 4x
    // redundancy while keeping the zero-threadgroup-memory, zero-barrier
    // discipline (still pure per-lane registers, `acc` is just an array of
    // 4 simdgroup_matrix accumulators instead of 1).
    simdgroup_matrix<float, 8, 8> accs[4];

    for (uint tj = 0u; tj < tileFn; tj += 1u) {
        for (uint ti = 0u; ti < tileFm; ti += 1u) {
            accs[ti].thread_elements()[0] = 0.0f;
            accs[ti].thread_elements()[1] = 0.0f;
        }
        uint nCol = blockN0 + tj * 8u;
        uint nCol0 = nCol + fn0;
        uint nCol1 = nCol + fn1;

        for (uint kBase = 0u; kBase < k; kBase += 16u) {
            for (uint half_ = 0u; half_ < 2u; half_ += 1u) {
                uint kf = kBase + half_ * 8u;
                uint kRow = kf + fm;

                // B: NVFP4 dequant computed ONCE per (tj, K-step), reused
                // across all 4 `ti` row-subtiles below. fm plays the
                // K-local-row role here (mirrors A's row/col-local
                // addressing lambda, applied to B's [K,N] logical tile
                // instead of A's [M,K] tile).
                uint pack0 = w[nCol0 * packsPerRow + kRow / 8u];
                uint nib0 = (pack0 >> ((kRow % 8u) * 4u)) & 0xFu;
                uchar sraw0 = scales[nCol0 * groupsPerRow + kRow / block_size];
                float scale0 = nvfp4_decode_scale_byte(sraw0, global_scale);

                uint pack1 = w[nCol1 * packsPerRow + kRow / 8u];
                uint nib1 = (pack1 >> ((kRow % 8u) * 4u)) & 0xFu;
                uchar sraw1 = scales[nCol1 * groupsPerRow + kRow / block_size];
                float scale1 = nvfp4_decode_scale_byte(sraw1, global_scale);

                subB.thread_elements()[0] = bfloat(nvfp4_decode_nibble(nib0, scale0));
                subB.thread_elements()[1] = bfloat(nvfp4_decode_nibble(nib1, scale1));

                for (uint ti = 0u; ti < tileFm; ti += 1u) {
                    uint mRow = blockM0 + ti * 8u;
                    // A: straight device read, no threadgroup stage.
                    uint aRow = mRow + fm;
                    subA.thread_elements()[0] = a[aRow * k + kf + fn0];
                    subA.thread_elements()[1] = a[aRow * k + kf + fn1];

                    simdgroup_multiply_accumulate(accs[ti], subA, subB, accs[ti]);
                }
            }
        }

        for (uint ti = 0u; ti < tileFm; ti += 1u) {
            uint mRow = blockM0 + ti * 8u;
            float r0 = accs[ti].thread_elements()[0];
            float r1 = accs[ti].thread_elements()[1];
            uint outRow = mRow + fm;
            out[outRow * n + nCol + fn0] = bfloat(r0);
            out[outRow * n + nCol + fn1] = bfloat(r1);
        }
    }
}
