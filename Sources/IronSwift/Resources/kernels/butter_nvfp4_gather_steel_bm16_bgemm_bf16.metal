// Copyright 2026 Tom Turney (@TheTom)
//
// T35: BM16 row-tile variant of the T33/T34 ragged-gather steel-structure
// NVFP4 GEMM. T34 fixed the redundant per-`ti` weight dequant (hoisted B
// to `tj`-outer/`ti`-inner) but found the remaining loss at Laguna XS's
// real production shape (topK=8, nExperts=256, ~16 rows/expert average
// run) was NOT the dequant redundancy -- it was `BM64` tile granularity.
// A 64-row output tile at ~16-row average run length straddles ~4
// experts, and this kernel family does one full K-reduction pass over
// the WHOLE tile per contiguous same-expert run (masking out-of-run rows
// at the write) -- so a 4-way-straddled tile pays ~4x the K-reduction
// work of a tile that holds exactly one run. The shipped mpp default
// already uses BM16 tiles, which is why it doesn't pay this multiplier.
//
// This kernel shrinks the M-tile from 64 to 16 rows (bn stays 64 -- N
// straddling isn't the fragmentation axis, only M/expert-boundary
// straddling is), so a typical 16-row tile now holds close to exactly
// one expert's run at production scale, collapsing the redundant-pass
// multiplier back toward 1x. Same T34 B-dequant hoist structure (`tj`
// outer, `ti` inner, per-lane accumulator array) is kept -- with bm/subM
// shrunk to 16, that array drops from 4 accumulators to 2, freeing
// registers the T35 mission brief suggested spending on a wider K-step
// or wider per-simdgroup N-tile; this first cut does neither (keeps the
// K-step and per-simdgroup N-tile identical to the BM64 kernel) to land
// a correctness-verified baseline first and let the bench data decide
// whether that headroom is worth spending.
//
// Simdgroup layout: BM64's kernel used a 2x2 simdgroup grid (`wn=2`,
// `subM=subN=32`) since a 64-row tile split naturally into two 32-row
// bands. A 16-row tile has only one row-band, so all 4 simdgroups here
// split the N dimension instead (`wn=4`, `subM=16`, `subN=16`) --
// `sgRow0` reduces to 0 for every simdgroup, `sgCol0` becomes
// `simd_group * 16`, covering the same 64-wide `bn` tile as BM64's
// kernel with 4 simdgroups side-by-side instead of a 2x2 block grid.
//
// `indices[mTotal]` gives the expert id per SORTED row (same contract as
// every other gather kernel in this family); `x` is already gathered to
// `[mTotal, kIn]`. No bounds checking on N or K (nOut %% 64 == 0,
// kIn %% 16 == 0, kIn %% block_size == 0 required); M is bounds-checked
// per-row since ragged/partial tiles are the whole point of this kernel.
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

kernel void butter_nvfp4_gather_steel_bm16_bgemm_bf16(
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
    const uint wn = 4u;
    const uint subM = 16u, subN = 16u;
    const uint tileFm = subM / 8u, tileFn = subN / 8u;
    const uint bm = 16u, bn = 64u;

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
        // dependency) for the end of this contiguous same-expert run within
        // the tile, or the tile/m_total boundary, whichever comes first.
        // Same technique as the BM64 kernel, just over a 16-row tile instead
        // of 64 -- at production scale (~16-row average run) this usually
        // finds `subEnd == bm` on the first iteration (one run fills the
        // whole tile), which is exactly the fragmentation fix this kernel
        // exists for.
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

            // T34-style hoist: `tj` outer, `ti` inner, `tileFm` (2 here,
            // vs BM64's 4) parallel per-lane accumulators. The B-operand
            // (NVFP4 dequant) depends only on (tj, K), never on `ti`; only
            // the A-operand masking (`rowInRun`) is per-`ti`.
            for (uint tj = 0u; tj < tileFn; tj += 1u) {
                simdgroup_matrix<float, 8, 8> accs[2];
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
