// Copyright 2026 Tom Turney (@TheTom)
//
// T33 stage 2: sorted-gather multi-expert NVFP4 GEMM using the steel MMA
// structure. Extends butter_nvfp4_gather_steel_bgemm_dense_bf16.metal
// (stage 1, single expert) with the ragged-segment handling the real MoE
// gather path needs: within a 64-row output tile, sorted rows (by
// `moeSortPlan`) can straddle a boundary between two or more experts, or
// come from a run shorter than the tile.
//
// Same zero-threadgroup-staging, zero-barrier steel MMA core as stage 1.
// The ragged-segment technique itself (find the contiguous same-expert
// run starting at the tile's current row offset, redundantly per-lane
// since it is a read-only scan over `indices` with no cross-lane
// dependency, then loop: full K-reduction pass over the WHOLE tile with
// out-of-run rows' A operand masked to zero, write output only for
// in-run rows, advance to the next run) is the same algorithm the shipped
// bm16 MPP gather kernel uses (`__ml_sub_offset`/`__ml_sub_end` in
// butter_nvfp4_moe_gather_bm16_mpp_xrows_bf16.metal), just applied to a
// 64-row tile instead of 16 and re-expressed for the barrier-free
// thread_elements() MMA path instead of mpp::tensor_ops::matmul2d.
//
// `indices[mTotal]` gives the expert id per SORTED row (same contract as
// every other gather kernel in this family); `x` is already gathered to
// `[mTotal, kIn]` (same convention as `nvfp4MoeGatherNax64` -- gather is a
// separate `Ops.gather` dispatch upstream, not fused into this kernel).
//
// No bounds checking on N or K (nOut %% 64 == 0, kIn %% 16 == 0,
// kIn %% block_size == 0 required); M is bounds-checked per-row since
// ragged/partial tiles are the whole point of this kernel.
//
// T34 update: same B-dequant hoist as the stage-1 dense kernel. Per run,
// the loop is `tj` outer, `ti` inner, with 4 parallel per-lane accumulators
// so each B element is dequantized once per (tj, K-step) instead of once
// per (ti, tj, K-step) -- removes the 4x redundant nibble-unpack +
// E4M3-scale-decode T33 measured as a 3.4-4.0x slowdown vs the shipped mpp
// default. Only the A-operand masking (`rowInRun`) stays per-`ti`, since
// that is genuinely per-row-subtile state, not redundant.
#include <metal_stdlib>
using namespace metal;
#include <metal_simdgroup_matrix>

// CARRY SIGN-FOLD E4M3 decode (unconditional, bit-exact for all 256 byte
// values -- see Tests/ButterTests/Ops/Nvfp4FastDequantTests.swift and the
// derivation in LagunaNvfp4Fuse.swift's `header`). Replaces the
// mask/shift/select ~7-op sequence above with 5 ops.
inline float nvfp4_decode_scale_byte(uchar raw, float global_scale) {
    ushort h = ushort((uint(raw) + (uint(raw) & 128u)) << 7);
    return float(as_type<half>(h)) * 256.0f * global_scale;
}

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

kernel void butter_nvfp4_gather_steel_bgemm_bf16(
    const device bfloat *x       [[buffer(0)]],   // [m_total, k_in], gathered
    const device uint   *w       [[buffer(1)]],   // [n_experts, n_out, k_in/8]
    const device uchar  *scales  [[buffer(2)]],   // [n_experts, n_out, k_in/block_size]
    const device uint   *indices [[buffer(3)]],   // [m_total], expert id per sorted row
    device bfloat       *out     [[buffer(4)]],   // [m_total, n_out]
    constant uint &m_total        [[buffer(5)]],
    constant uint &n_out          [[buffer(6)]],
    constant uint &k_in           [[buffer(7)]],
    constant uint &block_size     [[buffer(8)]],
    constant float &global_scale  [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint wn = 2u;
    const uint subM = 32u, subN = 32u;
    const uint tileFm = subM / 8u, tileFn = subN / 8u;
    const uint bm = 64u, bn = 64u;

    uint quad = simd_lane / 4u;
    uint fm = (quad & 4u) + ((simd_lane / 2u) % 4u);
    uint fn0 = ((quad & 2u) * 2u) + ((simd_lane % 2u) * 2u);
    uint fn1 = fn0 + 1u;

    uint sgRow0 = (simd_group / wn) * subM;
    uint sgCol0 = (simd_group % wn) * subN;
    uint tileRowBase = tid.y * bm;
    uint blockM0 = tileRowBase + sgRow0;
    uint blockN0 = tid.x * bn + sgCol0;

    uint packsPerRow = k_in / 8u;
    uint groupsPerRow = k_in / block_size;
    uint expertWStride = n_out * packsPerRow;
    uint expertSStride = n_out * groupsPerRow;

    uint subOffset = 0u;
    for (uint iter = 0u; iter < bm; iter += 1u) {
        if (subOffset >= bm) break;

        uint curRow = tileRowBase + subOffset;
        bool curInRange = curRow < m_total;
        if (!curInRange) { subOffset = bm; continue; }
        uint curExpert = indices[curRow];

        // Redundant per-lane scan (read-only over `indices`, no cross-lane
        // dependency -- matches the shipped bm16 MPP gather kernel's
        // approach) for the end of this contiguous same-expert run within
        // the tile, or the tile/m_total boundary, whichever comes first.
        uint subEnd = bm;
        bool found = false;
        for (uint p = subOffset + 1u; p < bm; p += 1u) {
            uint probeRow = tileRowBase + p;
            if (!found && probeRow < m_total) {
                if (indices[probeRow] != curExpert) { subEnd = p; found = true; }
            }
            if (!found && probeRow >= m_total) { subEnd = p; found = true; }
        }

        if (curInRange) {
            uint wBase = curExpert * expertWStride;
            uint sBase = curExpert * expertSStride;

            // T34: `tj` outer, `ti` inner, 4 parallel per-lane accumulators
            // -- same hoist as the dense stage-1 kernel. The B-operand
            // (NVFP4 dequant) depends only on (tj, K), never on `ti`; only
            // the A-operand masking (`rowInRun`) is per-`ti`. This still
            // does a full K-reduction pass per run for every `ti` subtile
            // regardless of whether that subtile's rows are in the current
            // run (same as T33's original -- the masked-out subtiles' work
            // is thrown away at the write, not skipped), so this hoist does
            // not change which rows get redundant per-run passes, only
            // removes the redundant weight dequant within a pass.
            for (uint tj = 0u; tj < tileFn; tj += 1u) {
                simdgroup_matrix<float, 8, 8> accs[4];
                simdgroup_matrix<bfloat, 8, 8> subA;
                simdgroup_matrix<bfloat, 8, 8> subB;
                for (uint ti = 0u; ti < tileFm; ti += 1u) {
                    accs[ti].thread_elements()[0] = 0.0f;
                    accs[ti].thread_elements()[1] = 0.0f;
                }
                uint nCol = blockN0 + tj * 8u;
                uint nCol0 = nCol + fn0;
                uint nCol1 = nCol + fn1;

                for (uint kBase = 0u; kBase < k_in; kBase += 16u) {
                    for (uint half_ = 0u; half_ < 2u; half_ += 1u) {
                        uint kf = kBase + half_ * 8u;
                        uint kRow = kf + fm;

                        uint pack0 = w[wBase + nCol0 * packsPerRow + kRow / 8u];
                        uint nib0 = (pack0 >> ((kRow % 8u) * 4u)) & 0xFu;
                        uchar sraw0 = scales[sBase + nCol0 * groupsPerRow + kRow / block_size];
                        float scale0 = nvfp4_decode_scale_byte(sraw0, global_scale);

                        uint pack1 = w[wBase + nCol1 * packsPerRow + kRow / 8u];
                        uint nib1 = (pack1 >> ((kRow % 8u) * 4u)) & 0xFu;
                        uchar sraw1 = scales[sBase + nCol1 * groupsPerRow + kRow / block_size];
                        float scale1 = nvfp4_decode_scale_byte(sraw1, global_scale);

                        subB.thread_elements()[0] = bfloat(nvfp4_decode_nibble(nib0, scale0));
                        subB.thread_elements()[1] = bfloat(nvfp4_decode_nibble(nib1, scale1));

                        for (uint ti = 0u; ti < tileFm; ti += 1u) {
                            uint mRow = blockM0 + ti * 8u;
                            uint rowLocal = sgRow0 + ti * 8u + fm;
                            bool rowInRun = rowLocal >= subOffset && rowLocal < subEnd
                                && (mRow + fm) < m_total;
                            uint aRow = mRow + fm;

                            subA.thread_elements()[0] = rowInRun
                                ? x[aRow * k_in + kf + fn0] : bfloat(0.0);
                            subA.thread_elements()[1] = rowInRun
                                ? x[aRow * k_in + kf + fn1] : bfloat(0.0);

                            simdgroup_multiply_accumulate(accs[ti], subA, subB, accs[ti]);
                        }
                    }
                }

                for (uint ti = 0u; ti < tileFm; ti += 1u) {
                    uint mRow = blockM0 + ti * 8u;
                    uint rowLocal = sgRow0 + ti * 8u + fm;
                    bool rowInRun = rowLocal >= subOffset && rowLocal < subEnd
                        && (mRow + fm) < m_total;
                    if (rowInRun) {
                        uint outRow = mRow + fm;
                        float r0 = accs[ti].thread_elements()[0];
                        float r1 = accs[ti].thread_elements()[1];
                        out[outRow * n_out + nCol + fn0] = bfloat(r0);
                        out[outRow * n_out + nCol + fn1] = bfloat(r1);
                    }
                }
            }
        }

        subOffset = subEnd;
    }
}
