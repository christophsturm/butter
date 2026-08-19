// Copyright 2026 Tom Turney (@TheTom)
//
// W34 flagship prefill lever: dense bf16 GEMM restructured around
// register-direct NAX-class fragment MMA, replacing (behind a flag) the
// threadgroup-staged `butter_dense_gemm_mpp_bf16` structure for the big
// aligned prefill projection shapes. The predecessor kernel runs at
// 10-15% of peak; profiling attributed the gap to its per-K-step
// threadgroup round-trip (stage A/B tiles to threadgroup memory, 2 full
// `mem_threadgroup` barriers every 32-deep K-step, then `ct.load()` back
// out of threadgroup memory into the cooperative tensors). This kernel
// removes that staging entirely.
//
// STRUCTURE (the four load-bearing differences vs the predecessor):
//
// 1. REGISTER-DIRECT OPERAND LOADS. A and B fragments are read straight
//    from device memory into `matmul2d` cooperative tensors. No
//    threadgroup memory is touched anywhere in the kernel. Importantly,
//    the cooperative-tensor implementation owns the mapping from matrix
//    coordinates to per-lane registers: Metal documents that mapping as
//    implementation-defined and dependent on the operation, data type,
//    and execution scope. The original M5-tuned version of this kernel
//    hard-coded that mapping and produced permuted fragments on M4 GPUs.
//    This family-9 variant hard-codes the layout captured by a one-shot
//    probe built with the documented coordinate accessors, but keeps those
//    accessors out of the production hot path. Redundant
//    A-tile reads across the 4 simdgroups that share an M-row (and
//    B-tile reads across the 2 that share an N-column) are served by
//    the cache hierarchy, which this structure trades for the removed
//    barrier/staging cost.
//
// 2. DEEP BARRIER-FREE K-LOOP. BK=256: the K reduction runs 8 fragment
//    K-steps (SK=32) per outer iteration with NO memory barriers inside
//    -- the only fence is one `threadgroup_barrier(mem_flags::mem_none)`
//    execution fence per outer BK iteration (an execution regrouper for
//    the simdgroups, not a memory fence; there is no shared memory to
//    order). For the production K=2048 projections that is 8 outer
//    iterations total, vs the predecessor's 64 barrier-pair K-steps.
//
// 3. PER-SIMDGROUP REGISTER OUTPUT TILE with PAIRED matmul2d CALLS.
//    Each of the 8 simdgroups (WM=2 x WN=4) owns a 32x32 f32 output
//    tile held in registers as a 2x2 grid of 16x16 fragments, fed by a
//    (16,32,16) `matmul2d_descriptor`: each call multiplies one 16x16 A
//    fragment against a PAIR of adjacent B fragments (nn, nn+1),
//    accumulating a 16x32 output strip -- two fragment columns per
//    issue, halving issue count vs a square 16x16x16 walk.
//
// 4. NATIVE bfloat OPERANDS. The cooperative tensors are created with
//    <bfloat, bfloat, float> directly -- no half staging, and therefore
//    no T23 magnitude fold: products accumulate in f32 with bfloat's
//    full e8 exponent range, so the half-overflow cliff class the
//    predecessor's fold exists to suppress cannot occur here at all.
//    (The predecessor's header called bfloat operands an open question;
//    the answer turned out to be that `matmul2d` accepts them fine when
//    fragments are written element-wise into the cooperative tensors --
//    it was the threadgroup-tensor `ct.load()` staging path that was
//    only ever exercised with half.)
//
// GEOMETRY: BM=64, BN=128, BK=256, WM=2, WN=4 -> 8 simdgroups, 256
// threads per threadgroup. Grid: [tiles_n(<<swizzle_log), ceil(tiles_m>>
// swizzle_log), 1] with tiles_n = N/128, tiles_m = M/64. `swizzle_log`
// remaps threadgroup IDs so adjacent threadgroups share B columns
// (L2-locality knob); 0 = identity mapping. Out-of-range remapped IDs
// (possible only when swizzle rounds the grid up) exit early.
//
// ALIGNMENT CONTRACT -- this kernel implements ONLY the fully aligned
// fast path: M % 64 == 0, N % 128 == 0, K % 256 == 0. There is no edge
// masking of any kind. `LagunaMppGemm` enforces all three at dispatch
// (M is already 64-padded upstream by the pad-M path; every Laguna
// prefill projection satisfies the N/K conditions -- see
// `nax2ShapeEligible`) and routes anything else to the staged MPP
// kernel. Direct callers MUST honor the same contract: an unaligned
// dispatch reads out of bounds.
//
// SEMANTICS -- identical to `butter_dense_gemm_mpp_bf16`: out[M,N] =
// X[M,K] @ W^T with W in Butter's native `Linear.weight` [N,K] row-major
// layout (K contiguous). The (16,32,16) descriptor is built with
// transpose_right=true, so B fragments are read straight out of [N,K]
// rows -- no host-side or on-device transpose pass, same as the
// predecessor.
//
// TODO(w34+1): edge-tile variant (fragment loads bounds-clamped per row/
// col) to lift the N%128/K%256 gate; BM=128 WM=4 config for very long
// prompts; revisit swizzle_log > 0 once a shape shows L2 pressure.

#include <metal_stdlib>
using namespace metal;

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Apple-family-9 register origins measured from the public cooperative-
// tensor coordinate accessors for this exact (16,32,16), bfloat/bfloat/
// float, execution_simdgroup operation. A and C share the same per-lane
// origin. B swaps the lane-bit roles because transpose_right=true.
//
// Keep these generation-specific helpers in this family-9-only source:
// the original family-10 source has a different fixed layout and remains
// unchanged.
static inline ushort2 butter_nax2_apple9_ac_origin(ushort lane) {
    const ushort x = ((lane & 1u) << 1u) | ((lane & 8u) >> 1u);
    const ushort y = ((lane >> 1u) & 3u) | ((lane & 16u) >> 2u);
    return ushort2(x, y);
}

static inline ushort2 butter_nax2_apple9_b_origin(ushort lane) {
    const ushort x = ((lane >> 1u) & 3u) | ((lane & 16u) >> 2u);
    const ushort y = ((lane & 1u) << 1u) | ((lane & 8u) >> 1u);
    return ushort2(x, y);
}

#endif  // __METAL_VERSION__ >= 400

kernel void butter_dense_gemm_nax2_mpp_bf16_apple9(
    const device bfloat *a [[buffer(0)]],   // [M, K] row-major, K contiguous
    const device bfloat *b [[buffer(1)]],   // [N, K] row-major, K contiguous (Linear.weight layout)
    device bfloat *out [[buffer(2)]],       // [M, N] row-major
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    constant uint &swizzleLog [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
    // Block geometry (see file header). SM/SN are each simdgroup's
    // output sub-tile; TM/TN/TK its 16-granular fragment counts.
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 256;
    constexpr int WN = 4;
    constexpr int SM = 32;   // BM / WM
    constexpr int SN = 32;   // BN / WN

    // Threadgroup-ID swizzle: consecutive hardware tgids walk a
    // (1<<swizzleLog)-tall column of M-tiles before advancing N,
    // improving B reuse in L2. swizzleLog=0 is the identity mapping.
    const int tid_y = int((tgid.y << swizzleLog) + (tgid.x & ((1u << swizzleLog) - 1u)));
    const int tid_x = int(tgid.x >> swizzleLog);
    const int tiles_m = int(M) / BM;
    const int tiles_n = int(N) / BN;
    if (tid_x >= tiles_n || tid_y >= tiles_m) {
        return;
    }

    const int lda = int(K);
    const int ldb = int(K);
    const int ldd = int(N);

    // Threadgroup block, then this simdgroup's sub-tile within it.
    const int c_row = tid_y * BM;
    const int c_col = tid_x * BN;
    const int sg_row = SM * (int(simd_group) / WN);
    const int sg_col = SN * (int(simd_group) % WN);

    const device bfloat *A = a + size_t(c_row + sg_row) * size_t(lda);
    const device bfloat *B = b + size_t(c_col + sg_col) * size_t(ldb);
    device bfloat *D = out + size_t(c_row + sg_row) * size_t(ldd) + size_t(c_col + sg_col);
    const ushort2 ac = butter_nax2_apple9_ac_origin(simd_lane);
    const ushort2 bc = butter_nax2_apple9_b_origin(simd_lane);

    // One 16x32 accumulator per 16-row half of the simdgroup's 32x32
    // output tile. The paired-N descriptor covers both adjacent
    // 16-column output fragments in one operation.
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        /*M=*/16, /*N=*/32, /*K=*/16,
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ct_a0 = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_a1 = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c0 = op.template get_destination_cooperative_tensor<
        decltype(ct_a0), decltype(ct_b), float>();
    auto ct_c1 = op.template get_destination_cooperative_tensor<
        decltype(ct_a1), decltype(ct_b), float>();

#pragma clang loop unroll(full)
    for (ushort i = 0; i < 16; i++) {
        ct_c0[i] = 0.0f;
        ct_c1[i] = 0.0f;
    }

    // K reduction: outer BK=256 blocks with a single execution fence
    // each (no memory to order -- this keeps the simdgroups marching
    // in loose lockstep so their redundant A/B reads stay cache-hot);
    // inner 16-deep fragment steps are barrier-free.
    const uint kIterations = K / uint(BK);
#pragma clang loop unroll(disable)
    for (uint kk0 = 0; kk0 < kIterations; kk0++) {
#pragma clang loop unroll(disable)
        for (int kk1 = 0; kk1 < BK; kk1 += 16) {
            // Family-9 A layout by cooperative-tensor storage index:
            // x offsets [0,1,8,9] repeated at y+8. Both A fragments
            // share that layout; the second starts 16 matrix rows down.
#pragma clang loop unroll(full)
            for (ushort i = 0; i < 8; i++) {
                const ushort x = ac.x + (i & 1u) + ((i & 2u) << 2u);
                const ushort y = ac.y + ((i & 4u) << 1u);
                const size_t off = size_t(y) * size_t(lda) +
                                   size_t(kk1 + int(x));
                ct_a0[i] = A[off];
                ct_a1[i] = A[size_t(16) * size_t(lda) + off];
            }

            // Family-9 B layout: storage indices 0...7 cover K x and
            // N-row offsets [0,1,8,9,16,17,24,25]; 8...15 repeat those
            // rows at K x+8. This is the family-9 counterpart to the
            // family-10 kernel's b0/b1 register packing.
#pragma clang loop unroll(full)
            for (ushort i = 0; i < 16; i++) {
                const ushort x = bc.x + (i & 8u);
                const ushort y = bc.y + (i & 1u) + ((i & 6u) << 2u);
                const size_t off = size_t(y) * size_t(ldb) +
                                   size_t(kk1 + int(x));
                ct_b[i] = B[off];
            }
            op.run(ct_a0, ct_b, ct_c0);
            op.run(ct_a1, ct_b, ct_c1);
        }

        A += BK;
        B += BK;
    }

    // Family-9 C layout: N offsets [0,1,8,9,16,17,24,25], repeated at
    // M+8. The second accumulator starts another 16 M rows down.
#pragma clang loop unroll(full)
    for (ushort i = 0; i < 16; i++) {
        const ushort x = ac.x + (i & 1u) + ((i & 6u) << 2u);
        const ushort y = ac.y + (i & 8u);
        const size_t off = size_t(y) * size_t(ldd) + size_t(x);
        D[off] = bfloat(ct_c0[i]);
        D[size_t(16) * size_t(ldd) + off] = bfloat(ct_c1[i]);
    }
#endif
}
