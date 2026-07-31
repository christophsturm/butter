// Copyright 2026 Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Tests for the T-blockmap prefill SDPA lever (`LagunaSdpa.swift`,
// `BUTTER_LAGUNA_PREFILL_BLOCKMAP=1`):
//
//   1. `LagunaPrefillBlockmapCPUTests`: pure CPU, no GPU/Metal touched.
//      Proves `LagunaSdpa.queryBlockBounds`'s closed form against an
//      independent brute-force per-element mask scan: for every row in
//      a query tile, the set of K columns the kernel would actually
//      visit (sink phase + edge-lo/full-mid/edge-hi, replaying the same
//      phase logic the MSL kernel runs) must exactly equal the set the
//      brute-force scan calls valid: no false negatives (a live pair
//      silently dropped would corrupt the softmax) and no false
//      positives (the "full" no-mask claim would be unsound).
//   2. `LagunaPrefillBlockmapClassifyTests`: same idea for the coarser
//      per-(query-block, key-block) `classifyBlock` form ("block (qi,ki)
//      fully dead iff…").
//   3. `LagunaPrefillBlockmapGPUTests`: GPU equivalence for the new
//      `laguna_sdpa_prefill_blockmap_f16_128` kernel vs. (a) the
//      existing `Ops.sdpaPrefillMma` (real simdgroup_matrix MMA) for the
//      causal-only case, and (b) a per-row `Ops.sdpaDecode` loop (the
//      existing sliding-window/sink fast path) as the oracle for the
//      window/sink cases `Ops.sdpaPrefillMma` has no equivalent for.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

// MARK: - 1. Pure CPU: queryBlockBounds vs. brute force

@Suite("LagunaPrefillBlockmap - closed-form bounds vs brute force")
struct LagunaPrefillBlockmapCPUTests {
    /// Ground truth, independently re-derived from `queryBlockBounds`'s
    /// doc comment (not a call into production code): query row `q` may
    /// attend to key column `k` iff `k <= q` (causal) AND (`k < sinkEnd`
    /// (sink override) OR `window <= 0` (no window) OR
    /// `k >= q - window + 1` (in-window)).
    static func isValid(q: Int, k: Int, window: Int, sinkEnd: Int) -> Bool {
        guard k <= q else { return false }
        if sinkEnd > 0 && k < sinkEnd { return true }
        if window <= 0 { return true }
        return k >= q - window + 1
    }

    /// Replays exactly what the MSL kernel's four phases would visit for
    /// one row, given the bounds `queryBlockBounds` computed for the
    /// row's tile: sink prefix (causal-masked), edge-lo (causal+window
    /// masked), full-mid (unmasked), edge-hi (causal+window masked).
    static func kernelWouldVisit(
        q: Int, bounds: LagunaSdpa.PrefillBlockmapBounds, window: Int
    ) -> Set<Int> {
        var live = Set<Int>()
        if bounds.sinkEnd > 0 {
            for k in 0 ..< bounds.sinkEnd where k <= q { live.insert(k) }
        }
        for k in bounds.mainStart ..< bounds.fullLo {
            let causalOk = k <= q
            let windowOk = window == 0 || (k + window > q)
            if causalOk && windowOk { live.insert(k) }
        }
        for k in bounds.fullLo ..< bounds.fullHi { live.insert(k) }  // no mask test
        for k in bounds.fullHi ..< bounds.kvEnd {
            let causalOk = k <= q
            let windowOk = window == 0 || (k + window > q)
            if causalOk && windowOk { live.insert(k) }
        }
        return live
    }

    private func checkTile(
        qiMin: Int, qiMax: Int, kPad: Int, window: Int, sinkEnd: Int,
        label: String
    ) {
        let bounds = LagunaSdpa.queryBlockBounds(
            qiMin: qiMin, qiMax: qiMax, kPad: kPad, window: window, sinkEnd: sinkEnd)

        // Ordering invariant the kernel's loop bounds depend on.
        #expect(bounds.sinkEnd >= 0, "\(label): sinkEnd negative")
        #expect(bounds.sinkEnd <= bounds.mainStart, "\(label): sinkEnd > mainStart")
        #expect(bounds.mainStart <= bounds.fullLo, "\(label): mainStart > fullLo")
        #expect(bounds.fullLo <= bounds.fullHi, "\(label): fullLo > fullHi")
        #expect(bounds.fullHi <= bounds.kvEnd, "\(label): fullHi > kvEnd")
        #expect(bounds.kvEnd <= kPad, "\(label): kvEnd > kPad")

        for q in qiMin ... qiMax {
            let expected = Set(
                (0 ..< kPad).filter {
                    Self.isValid(q: q, k: $0, window: window, sinkEnd: sinkEnd)
                })
            let got = LagunaPrefillBlockmapCPUTests.kernelWouldVisit(
                q: q, bounds: bounds, window: window)
            #expect(
                got == expected,
                Comment(
                    rawValue: "\(label): q=\(q) mismatch. missing=\(expected.subtracting(got).sorted()) "
                        + "extra=\(got.subtracting(expected).sorted())"))
        }
    }

    @Test("pure causal, first tile and second tile")
    func causalSmall() {
        checkTile(qiMin: 0, qiMax: 31, kPad: 64, window: 0, sinkEnd: 0, label: "causal tile 0")
        checkTile(qiMin: 32, qiMax: 63, kPad: 64, window: 0, sinkEnd: 0, label: "causal tile 1")
    }

    @Test("pure causal, nonzero position (continuing prefill)")
    func causalPositioned() {
        checkTile(qiMin: 100, qiMax: 131, kPad: 200, window: 0, sinkEnd: 0, label: "causalPositioned")
    }

    @Test("SWA: seq < window (degenerates to pure causal)")
    func swaSeqLessThanWindow() {
        checkTile(qiMin: 0, qiMax: 31, kPad: 32, window: 256, sinkEnd: 0, label: "seqLtWindow")
    }

    @Test("SWA: seq == window exactly")
    func swaSeqEqualsWindow() {
        checkTile(qiMin: 0, qiMax: 63, kPad: 64, window: 64, sinkEnd: 0, label: "seqEqWindow")
    }

    @Test("SWA: window narrower than one tile")
    func swaNarrowWindow() {
        checkTile(qiMin: 200, qiMax: 231, kPad: 256, window: 16, sinkEnd: 0, label: "narrowWindow")
    }

    @Test("SWA: window boundary sweeps across the tile mid-scan")
    func swaBoundaryStraddle() {
        checkTile(qiMin: 0, qiMax: 63, kPad: 96, window: 32, sinkEnd: 0, label: "boundaryStraddle")
    }

    @Test("SWA: window == 1 (degenerate, only self is live)")
    func swaWindowOfOne() {
        checkTile(qiMin: 200, qiMax: 231, kPad: 256, window: 1, sinkEnd: 0, label: "windowOfOne")
    }

    @Test("sink prefix only, no window")
    func sinkOnly() {
        checkTile(qiMin: 0, qiMax: 31, kPad: 64, window: 0, sinkEnd: 4, label: "sinkOnly")
    }

    @Test("sink + window, positioned so no gap ambiguity")
    func sinkWindowNoOverlap() {
        checkTile(qiMin: 200, qiMax: 231, kPad: 256, window: 32, sinkEnd: 4, label: "sinkWindowNoOverlap")
    }

    @Test("sink + window: sink extends past the window's own lower edge for early rows")
    func sinkWindowOverlap() {
        checkTile(qiMin: 0, qiMax: 31, kPad: 64, window: 16, sinkEnd: 8, label: "sinkWindowOverlap")
    }

    @Test("sink prefix exceeds the tile's causal horizon (early prompt, big configured sink)")
    func sinkExceedsHorizon() {
        checkTile(qiMin: 0, qiMax: 7, kPad: 64, window: 4, sinkEnd: 100, label: "sinkExceedsHorizon")
    }

    @Test("single-row tile (qiMin == qiMax)")
    func singleRowTile() {
        checkTile(qiMin: 50, qiMax: 50, kPad: 64, window: 8, sinkEnd: 2, label: "singleRow")
    }

    @Test("degenerate: kPad == 0 (empty cache)")
    func emptyCache() {
        checkTile(qiMin: 0, qiMax: 31, kPad: 0, window: 16, sinkEnd: 4, label: "emptyCache")
    }

    @Test("randomized sweep vs. brute force")
    func randomizedSweep() {
        var seed: UInt64 = 0xD1B5_4A32_D192_ED03
        func nextInt(_ bound: Int) -> Int {
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            return Int(seed % UInt64(bound))
        }
        let widths = [1, 2, 8, 32]
        for i in 0 ..< 300 {
            let kPad = 1 + nextInt(200)
            let tileWidth = widths[nextInt(widths.count)]
            let qiMin = nextInt(kPad + 32)
            let qiMax = qiMin + tileWidth - 1
            let window = nextInt(3) == 0 ? 0 : 1 + nextInt(64)
            let sinkEnd = nextInt(3) == 0 ? 0 : nextInt(16)
            checkTile(
                qiMin: qiMin, qiMax: qiMax, kPad: kPad, window: window, sinkEnd: sinkEnd,
                label:
                    "random#\(i) qiMin=\(qiMin) qiMax=\(qiMax) kPad=\(kPad) "
                    + "window=\(window) sinkEnd=\(sinkEnd)")
        }
    }
}

// MARK: - 2. Pure CPU: classifyBlock vs. brute force

@Suite("LagunaPrefillBlockmap - classifyBlock vs brute force")
struct LagunaPrefillBlockmapClassifyTests {
    private func bruteForceAllValid(
        qiMin: Int, qiMax: Int, kiMin: Int, kiMax: Int, window: Int, sinkEnd: Int
    ) -> Bool {
        for q in qiMin ... qiMax {
            for k in kiMin ... kiMax
            where !LagunaPrefillBlockmapCPUTests.isValid(
                q: q, k: k, window: window, sinkEnd: sinkEnd)
            {
                return false
            }
        }
        return true
    }

    private func bruteForceAnyValid(
        qiMin: Int, qiMax: Int, kiMin: Int, kiMax: Int, window: Int, sinkEnd: Int
    ) -> Bool {
        for q in qiMin ... qiMax {
            for k in kiMin ... kiMax
            where LagunaPrefillBlockmapCPUTests.isValid(
                q: q, k: k, window: window, sinkEnd: sinkEnd)
            {
                return true
            }
        }
        return false
    }

    private func check(
        qiMin: Int, qiMax: Int, kiMin: Int, kiMax: Int, window: Int, sinkEnd: Int, label: String
    ) {
        let cls = LagunaSdpa.classifyBlock(
            qiMin: qiMin, qiMax: qiMax, kiMin: kiMin, kiMax: kiMax, window: window, sinkEnd: sinkEnd)
        switch cls {
        case .dead:
            #expect(
                !bruteForceAnyValid(
                    qiMin: qiMin, qiMax: qiMax, kiMin: kiMin, kiMax: kiMax,
                    window: window, sinkEnd: sinkEnd),
                "\(label): classified .dead but brute force found a live (q,k) pair")
        case .full:
            #expect(
                bruteForceAllValid(
                    qiMin: qiMin, qiMax: qiMax, kiMin: kiMin, kiMax: kiMax,
                    window: window, sinkEnd: sinkEnd),
                "\(label): classified .full but brute force found a dead (q,k) pair")
        case .edge:
            break  // No unconditional claim either way; always safe by construction.
        }
    }

    @Test("dead: strictly future block (causal)")
    func deadFuture() {
        check(qiMin: 0, qiMax: 31, kiMin: 64, kiMax: 95, window: 0, sinkEnd: 0, label: "deadFuture")
    }

    @Test("dead: strictly window-expired block")
    func deadWindowExpired() {
        check(
            qiMin: 500, qiMax: 531, kiMin: 0, kiMax: 31, window: 64, sinkEnd: 0,
            label: "deadWindowExpired")
    }

    @Test("not dead: window-expired but rescued by an overlapping sink")
    func rescuedBySink() {
        check(
            qiMin: 500, qiMax: 531, kiMin: 0, kiMax: 31, window: 64, sinkEnd: 16,
            label: "rescuedBySink")
    }

    @Test("full: entirely below the diagonal, no window")
    func fullCausalSafe() {
        check(qiMin: 64, qiMax: 95, kiMin: 0, kiMax: 31, window: 0, sinkEnd: 0, label: "fullCausalSafe")
    }

    @Test("full: causal-safe and window-safe")
    func fullWindowSafe() {
        check(
            qiMin: 200, qiMax: 231, kiMin: 100, kiMax: 131, window: 256, sinkEnd: 0,
            label: "fullWindowSafe")
    }

    @Test("full: entirely inside the sink prefix")
    func fullInsideSink() {
        check(
            qiMin: 500, qiMax: 531, kiMin: 0, kiMax: 7, window: 16, sinkEnd: 8,
            label: "fullInsideSink")
    }

    @Test("edge: causal diagonal band")
    func edgeDiagonal() {
        check(qiMin: 32, qiMax: 63, kiMin: 32, kiMax: 63, window: 0, sinkEnd: 0, label: "edgeDiagonal")
    }

    @Test("edge: window boundary band")
    func edgeWindowBoundary() {
        check(
            qiMin: 500, qiMax: 531, kiMin: 460, kiMax: 491, window: 64, sinkEnd: 0,
            label: "edgeWindowBoundary")
    }

    @Test("randomized sweep vs. brute force")
    func randomizedSweep() {
        var seed: UInt64 = 0x9E37_79B9_2545_F491
        func nextInt(_ bound: Int) -> Int {
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            return Int(seed % UInt64(bound))
        }
        for i in 0 ..< 400 {
            let base = nextInt(200)
            let qiMin = base
            let qiMax = qiMin + nextInt(32)
            let kiMin = nextInt(base + 64)
            let kiMax = kiMin + nextInt(32)
            let window = nextInt(3) == 0 ? 0 : 1 + nextInt(64)
            let sinkEnd = nextInt(3) == 0 ? 0 : nextInt(16)
            check(
                qiMin: qiMin, qiMax: qiMax, kiMin: kiMin, kiMax: kiMax,
                window: window, sinkEnd: sinkEnd, label: "random#\(i)")
        }
    }
}

// MARK: - 3. GPU equivalence: new kernel vs. existing shipped kernels


/// Apple GPU Family 9+ required for the MPP cooperative-tensor kernels the
/// MoE and prefill paths dispatch. CI paravirtualized GPUs cannot build
/// those PSOs, so the GPU-equivalence suites skip there rather than crash.
private func mppCoopTensorsAvailable() -> Bool {
    guard let dev = MTLCreateSystemDefaultDevice() else { return false }
    return dev.supportsFamily(.apple9)
}

@Suite("LagunaPrefillBlockmap - GPU equivalence")
struct LagunaPrefillBlockmapGPUTests {
    private func randomF16(_ shape: [Int], seed: inout UInt64) -> Tensor {
        let n = shape.reduce(1, *)
        var vals = [Float16](repeating: 0, count: n)
        for i in 0 ..< n {
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            let u = Double(seed % 2_000_001) / 1_000_000.0 - 1.0  // [-1, 1)
            vals[i] = Float16(Float(u))
        }
        let t = Tensor.empty(shape: shape, dtype: .f16)
        t.copyIn(from: vals)
        return t
    }

    private func compareArrays(_ a: [Float], _ b: [Float], label: String) {
        #expect(a.count == b.count, "\(label): length mismatch \(a.count) vs \(b.count)")
        guard a.count == b.count else { return }
        var maxAbsDiff: Float = 0
        var dot: Double = 0, na: Double = 0, nb: Double = 0
        for i in 0 ..< a.count {
            maxAbsDiff = max(maxAbsDiff, abs(a[i] - b[i]))
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        let cos = (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 1.0
        #expect(maxAbsDiff < 0.05, "\(label): maxAbsDiff=\(maxAbsDiff)")
        #expect(cos > 0.999, "\(label): cos=\(cos)")
    }

    /// Causal-only (window=0, sinkEnd=0): the block-map kernel vs. the
    /// existing `Ops.sdpaPrefillMma` (real simdgroup_matrix MMA). Same
    /// masking, two independently-written kernels: the direct "the new
    /// kernel computes the same thing the shipped one does" check.
    private func compareVsMma(nHeads: Int, nKVHeads: Int, tPad: Int, position: Int, label: String) {
        guard mppCoopTensorsAvailable() else { return }
        autoreleasepool {
            let headDim = 128
            let kPad = position + tPad
            var seed: UInt64 = 0x1234_5678_9ABC_DEF0
            let qHTD = randomF16([nHeads, tPad, headDim], seed: &seed)
            let kPacked = randomF16([nKVHeads, kPad, headDim], seed: &seed)
            let vPacked = randomF16([nKVHeads, kPad, headDim], seed: &seed)
            let scale: Float = 1.0 / Float(headDim).squareRoot()

            var mmaOut: Tensor!
            runAndWait { cb in
                mmaOut = Ops.sdpaPrefillMma(
                    q: qHTD, k: kPacked, v: vPacked,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    qLen: tPad, kLen: kPad, scale: scale, on: cb)
            }
            let bmOut = Tensor.empty(shape: [nHeads, tPad, headDim], dtype: .f16)
            runAndWait { cb in
                LagunaSdpa.sdpaPrefillBlockmap(
                    qHTD: qHTD, kPacked: kPacked, vPacked: vPacked,
                    nHeads: nHeads, nKVHeads: nKVHeads,
                    tPad: tPad, kPad: kPad, position: position,
                    window: 0, sinkEnd: 0, scale: scale,
                    on: cb, device: .shared, into: bmOut)
            }
            compareArrays(mmaOut.toFloatArray(), bmOut.toFloatArray(), label: label)
        }
    }

    @Test("causal only, position=0, one query tile")
    func causalPositionZeroOneTile() {
        compareVsMma(nHeads: 8, nKVHeads: 2, tPad: 32, position: 0, label: "causalPositionZeroOneTile")
    }

    @Test("causal only, position=0, two query tiles")
    func causalTwoTiles() {
        compareVsMma(nHeads: 8, nKVHeads: 2, tPad: 64, position: 0, label: "causalTwoTiles")
    }

    @Test("causal only, nonzero position (continuing a prefill chunk)")
    func causalNonzeroPosition() {
        compareVsMma(nHeads: 8, nKVHeads: 2, tPad: 64, position: 100, label: "causalNonzeroPosition")
    }

    @Test("causal only, GQA factor 1 (nHeads == nKVHeads)")
    func causalNoGqa() {
        compareVsMma(nHeads: 4, nKVHeads: 4, tPad: 32, position: 17, label: "causalNoGqa")
    }

    /// Sliding-window (+ optional sink): the block-map kernel vs. a
    /// per-row `Ops.sdpaDecode` loop: the existing, already-shipped
    /// sliding-window/sink fast path, and the correct oracle here since
    /// `Ops.sdpaPrefillMma` has no windowing to compare against at all
    /// (this is the newly-enabled capability, see `LagunaSdpa.swift`'s
    /// `prefillBlockmapEnabled` header comment).
    private func compareVsDecodeOracle(
        nHeads: Int, nKVHeads: Int, tPad: Int, position: Int,
        window: Int, sinkEnd: Int, label: String
    ) {
        guard mppCoopTensorsAvailable() else { return }
        autoreleasepool {
            let headDim = 128
            let kPad = position + tPad
            var seed: UInt64 = 0x0FED_CBA9_8765_4321
            let qTHD = randomF16([tPad, nHeads, headDim], seed: &seed)
            let kPacked = randomF16([nKVHeads, kPad, headDim], seed: &seed)
            let vPacked = randomF16([nKVHeads, kPad, headDim], seed: &seed)
            let scale: Float = 1.0 / Float(headDim).squareRoot()

            var qHTD: Tensor!
            runAndWait { cb in
                qHTD = Ops.transposeTHD_f16(qTHD, t: tPad, h: nHeads, d: headDim, on: cb)
            }

            let bmOutHTD = Tensor.empty(shape: [nHeads, tPad, headDim], dtype: .f16)
            runAndWait { cb in
                LagunaSdpa.sdpaPrefillBlockmap(
                    qHTD: qHTD, kPacked: kPacked, vPacked: vPacked,
                    nHeads: nHeads, nKVHeads: nKVHeads,
                    tPad: tPad, kPad: kPad, position: position,
                    window: window, sinkEnd: sinkEnd, scale: scale,
                    on: cb, device: .shared, into: bmOutHTD)
            }
            var bmOutTHD: Tensor!
            runAndWait { cb in
                bmOutTHD = Ops.transposeHTD_f16(bmOutHTD, t: tPad, h: nHeads, d: headDim, on: cb)
            }

            // Oracle: `nKV`/`windowStart` computed per row exactly the
            // way `KVEvictionState`'s own convention works (see
            // `LagunaSdpa.queryBlockBounds`'s doc comment):
            // `windowStart = max(0, nKV - window)`, `nKV = q + 1`.
            let oracleTHD = Tensor.empty(shape: [tPad, nHeads, headDim], dtype: .f16)
            runAndWait { cb in
                for r in 0 ..< tPad {
                    let qRow = qTHD.slicedRows(start: r, count: 1).reshaped(to: [nHeads, headDim])
                    let outRow = oracleTHD.slicedRows(start: r, count: 1).reshaped(
                        to: [nHeads, headDim])
                    let nKV = position + r + 1
                    let windowStart = window > 0 ? max(0, nKV - window) : 0
                    _ = Ops.sdpaDecode(
                        q: qRow, k: kPacked, v: vPacked,
                        nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                        nKV: nKV, kvStride: kPad, scale: scale, on: cb,
                        sinkEnd: sinkEnd, windowStart: windowStart, into: outRow)
                }
            }

            compareArrays(oracleTHD.toFloatArray(), bmOutTHD.toFloatArray(), label: label)
        }
    }

    @Test("sliding window: boundary activates partway through the chunk")
    func windowBoundaryMidChunk() {
        compareVsDecodeOracle(
            nHeads: 8, nKVHeads: 2, tPad: 64, position: 0, window: 32, sinkEnd: 0,
            label: "windowBoundaryMidChunk")
    }

    @Test("sliding window: seq < window everywhere (degenerate causal)")
    func windowSeqLessThanWindow() {
        compareVsDecodeOracle(
            nHeads: 8, nKVHeads: 2, tPad: 32, position: 0, window: 256, sinkEnd: 0,
            label: "windowSeqLessThanWindow")
    }

    @Test("sliding window, nonzero starting position, window narrower than the chunk")
    func windowNonzeroPosition() {
        compareVsDecodeOracle(
            nHeads: 8, nKVHeads: 2, tPad: 64, position: 200, window: 48, sinkEnd: 0,
            label: "windowNonzeroPosition")
    }

    @Test("sink + window, positioned so no gap ambiguity")
    func sinkAndWindow() {
        compareVsDecodeOracle(
            nHeads: 8, nKVHeads: 2, tPad: 64, position: 40, window: 32, sinkEnd: 4,
            label: "sinkAndWindow")
    }
}
