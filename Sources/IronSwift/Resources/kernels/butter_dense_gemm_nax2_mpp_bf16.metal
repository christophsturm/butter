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
//    from device memory into per-thread fragment registers (the
//    `frag_load` helper below: each thread owns 8 elements of every
//    16x16 fragment, at a fixed lane->coordinate mapping) and written
//    element-by-element into the `matmul2d` cooperative tensors. No
//    threadgroup memory is touched anywhere in the kernel. Redundant
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

// ── Fragment primitives ─────────────────────────────────────────────
//
// A 16x16 fragment is distributed over the 32 lanes of a simdgroup as 8
// elements per lane: 2 row-groups (rows sc.y and sc.y+8) of 4
// consecutive columns starting at sc.x. The lane->coordinate mapping
// below is the fixed layout `matmul2d`'s cooperative tensors use for
// 16x16 operand fragments on Apple GPU family 9; writing fragment
// elements at these coordinates in this order makes a plain indexed
// copy into the cooperative tensor land every element where the MMA
// expects it.

typedef vec<bfloat, 8> butter_nax2_frag_bf16;
typedef vec<float, 8> butter_nax2_frag_f32;

// Lane -> (col=x, row=y) of the lane's first element within a 16x16
// fragment.
static inline short2 butter_nax2_frag_coord(ushort lane) {
    const short qid = short(lane >> 2);
    const short fm = (qid & 4) | ((short(lane) >> 1) & 3);
    const short fn = ((qid & 2) | (short(lane) & 1)) * 4;
    return short2(fn, fm);
}

// Load this lane's 8 elements of the 16x16 fragment whose top-left
// corner is `src` (row-major, leading dimension `ld`, unit column
// stride). Aligned fast path only -- no bounds checks (see the file
// header's alignment contract).
static inline void butter_nax2_frag_load(
    thread butter_nax2_frag_bf16 &dst,
    const device bfloat *src,
    int ld,
    short2 sc
) {
    const device bfloat *p = src + sc.y * ld + sc.x;
#pragma clang loop unroll(full)
    for (short i = 0; i < 2; i++) {
#pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
            dst[i * 4 + j] = p[i * 8 * ld + j];
        }
    }
}

// Store this lane's 8 elements of a 16x16 f32 accumulator fragment to
// `dst` (row-major, leading dimension `ld`), casting to bfloat.
static inline void butter_nax2_frag_store(
    const thread butter_nax2_frag_f32 &src,
    device bfloat *dst,
    int ld,
    short2 sc
) {
    device bfloat *p = dst + sc.y * ld + sc.x;
#pragma clang loop unroll(full)
    for (short i = 0; i < 2; i++) {
#pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
            p[i * 8 * ld + j] = bfloat(src[i * 4 + j]);
        }
    }
}

// One paired-N MMA: {c0|c1}[16x32] += a[16x16] @ {b0|b1}[32x16]^T.
// b0/b1 are two adjacent 16-column output fragments' worth of B, each
// read as a 16(N)x16(K) fragment of the [N,K] weight;
// transpose_right=true in the descriptor makes the op consume them as
// K x N. Native bfloat operands, f32 accumulate; relaxed_precision OFF:
// A/B'd both settings at the 1351-token end-to-end gate and they are
// perf-identical within run noise (2269 vs 2290-2320 tok/s prefill),
// while strict tightens final-position prefill-logit agreement with the
// staged kernel from cos 0.971 to 0.984 -- the same noise class as the
// already-shipped staged-vs-steel delta (0.987) on the identical prompt.
// Free accuracy, so strict wins.
static inline void butter_nax2_mma_pair(
    thread butter_nax2_frag_f32 &c0,
    thread butter_nax2_frag_f32 &c1,
    const thread butter_nax2_frag_bf16 &a,
    const thread butter_nax2_frag_bf16 &b0,
    const thread butter_nax2_frag_bf16 &b1
) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        /*M=*/16, /*N=*/32, /*K=*/16,
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ct_a = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<
        decltype(ct_a), decltype(ct_b), float>();

#pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        ct_a[i] = a[i];
    }
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        ct_b[i] = b0[i];
        ct_b[8 + i] = b1[i];
    }
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        ct_c[i] = c0[i];
        ct_c[8 + i] = c1[i];
    }

    op.run(ct_a, ct_b, ct_c);

#pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
        c0[i] = ct_c[i];
        c1[i] = ct_c[8 + i];
    }
}

#endif  // __METAL_VERSION__ >= 400

kernel void butter_dense_gemm_nax2_mpp_bf16(
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
    constexpr int SK = 32;
    constexpr int TM = SM / 16;  // 2
    constexpr int TN = SN / 16;  // 2
    constexpr int TK = SK / 16;  // 2

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

    const short2 sc = butter_nax2_frag_coord(simd_lane);

    // Per-simdgroup 32x32 f32 accumulator: 2x2 fragments, D[mm][nn].
    butter_nax2_frag_f32 Dfrag[TM][TN];
#pragma clang loop unroll(full)
    for (short mm = 0; mm < TM; mm++) {
#pragma clang loop unroll(full)
        for (short nn = 0; nn < TN; nn++) {
            Dfrag[mm][nn] = butter_nax2_frag_f32(0.0f);
        }
    }

    // K reduction: outer BK=256 blocks with a single execution fence
    // each (no memory to order -- this keeps the simdgroups marching
    // in loose lockstep so their redundant A/B reads stay cache-hot);
    // inner SK=32 fragment steps are barrier-free.
    const uint kIterations = K / uint(BK);
#pragma clang loop unroll(disable)
    for (uint kk0 = 0; kk0 < kIterations; kk0++) {
        threadgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(disable)
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            // Fresh operand fragments each step; the volatile keeps the
            // compiler from coalescing/hoisting loads across K-steps
            // (register-pressure fence, same trick the steel NAX loop
            // uses).
            butter_nax2_frag_bf16 Afrag[TM][TK];
            butter_nax2_frag_bf16 Bfrag[TN][TK];
            volatile int compiler_barrier;

#pragma clang loop unroll(full)
            for (short mm = 0; mm < TM; mm++) {
#pragma clang loop unroll(full)
                for (short kf = 0; kf < TK; kf++) {
                    butter_nax2_frag_load(
                        Afrag[mm][kf], A + mm * 16 * lda + kk1 + kf * 16, lda, sc);
                }
            }
#pragma clang loop unroll(full)
            for (short nn = 0; nn < TN; nn++) {
#pragma clang loop unroll(full)
                for (short kf = 0; kf < TK; kf++) {
                    butter_nax2_frag_load(
                        Bfrag[nn][kf], B + nn * 16 * ldb + kk1 + kf * 16, ldb, sc);
                }
            }

            // Paired-N fragment matmul walk: for each A row-fragment,
            // one (16,32,16) call per K-fragment covers both N
            // fragments at once.
#pragma clang loop unroll(full)
            for (short mm = 0; mm < TM; mm++) {
#pragma clang loop unroll(full)
                for (short kf = 0; kf < TK; kf++) {
                    butter_nax2_mma_pair(
                        Dfrag[mm][0], Dfrag[mm][1],
                        Afrag[mm][kf],
                        Bfrag[0][kf], Bfrag[1][kf]);
                }
            }

            (void)compiler_barrier;
        }

        A += BK;
        B += BK;
    }

    // Aligned store: each lane writes its 8 elements of each of the 4
    // output fragments, f32 -> bfloat.
#pragma clang loop unroll(full)
    for (short mm = 0; mm < TM; mm++) {
#pragma clang loop unroll(full)
        for (short nn = 0; nn < TN; nn++) {
            butter_nax2_frag_store(
                Dfrag[mm][nn], D + mm * 16 * ldd + nn * 16, ldd, sc);
        }
    }
#endif
}
