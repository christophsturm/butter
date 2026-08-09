// Copyright 2026 Tom Turney (@TheTom)
//
// Follow-on lever gate for `LagunaNvfp4ExpertGather` — the
// scaffolding-removal levers stacked on the endgame expert-aligned
// gather QMM (`BUTTER_LAGUNA_GATHER_V2` path):
//
//   1. skipSort   — skip the identity k=1 re-sort on MoELayer's
//                   already-sorted plan path (+ direct GEMM output
//                   writes, + direct token-table gather-cast).
//   2. fusedSilu  — SiLU·mul fused into the scatter-back / inner store.
//   3. f16Inner   — SwiGLU inner stored f16 (bf16-roundtrip), down GEMM
//                   reads it device-direct with no cast pass.
//
// THE BAR IS BYTE-IDENTITY, not cosine: every lever is a pure
// scaffolding removal and must reproduce the base V2 chain's output
// bits exactly (the full-model convergence gate — identical greedy
// token streams — rests on this). Tests follow the sibling suites'
// idioms (LagunaMppGatherTests.swift): synthetic NVFP4 weights, sorted
// ragged multi-expert runs incl. zero-count experts, MPP capability
// skip-not-fail guard.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("LagunaNvfp4ExpertGather follow-on levers", .serialized)
struct LagunaExpertGatherLeverTests {
    private var mppAvailable: Bool { Ops.mppAutoCapable }

    // MARK: - Helpers

    private func bytesU16(_ t: Tensor) -> [UInt16] {
        let n = t.elementCount
        let ptr = t.buffer.contents().advanced(by: t.offset)
            .bindMemory(to: UInt16.self, capacity: n)
        return (0 ..< n).map { ptr[$0] }
    }

    private func expectByteIdentical(
        _ a: Tensor, _ b: Tensor, _ label: String
    ) {
        let ab = bytesU16(a)
        let bb = bytesU16(b)
        #expect(ab.count == bb.count, "\(label): element count mismatch")
        var firstDiff = -1
        var diffCount = 0
        for i in 0 ..< min(ab.count, bb.count) where ab[i] != bb[i] {
            if firstDiff < 0 { firstDiff = i }
            diffCount += 1
        }
        let msg =
            "\(label): \(diffCount) byte diffs, first at \(firstDiff) "
            + "(a=0x\(String(ab[max(firstDiff, 0)], radix: 16)) "
            + "b=0x\(String(bb[max(firstDiff, 0)], radix: 16)))"
        #expect(diffCount == 0, Comment(rawValue: msg))
    }

    /// Deterministic pseudo-random bf16 payload (LCG; values in ±~4 with
    /// a few amplified rows so magnitudes differ across rows/cols).
    private func bf16Payload(count: Int, seed: UInt64, scale: Float = 4.0) -> [UInt16] {
        var s = seed
        return (0 ..< count).map { _ in
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let u = Float(s >> 40) / Float(1 << 24)  // [0,1)
            let v = (u - 0.5) * 2.0 * scale
            return UInt16(truncatingIfNeeded: v.bitPattern >> 16)
        }
    }

    private struct Fixture {
        let device: Device
        let mTotal: Int
        let t: Int
        let kIn: Int
        let nOut: Int
        let nExperts: Int
        let hRows: Tensor      // [t, kIn] bf16 token table
        let srcTok: Tensor     // [mTotal] u32 sorted-slot → token row
        let indices: Tensor    // [mTotal] u32 expert ids, sorted ascending
        let weight: Tensor     // [nExperts, nOut, kIn/8] u32
        let scales: Tensor     // [nExperts, nOut, kIn/16] u8
        let weight2: Tensor    // second bank (up / distinct from gate)
        let scales2: Tensor
    }

    /// Sorted ragged runs incl. zero-count experts: expert 1 × 50 rows,
    /// expert 4 × 77, expert 7 × 1 (unit run), expert 9 × 72; experts
    /// 0/2/3/5/6/8 and 10+ never appear. mTotal = 200 (non-multiple of
    /// 16 and 64 → exercises masked partial tiles).
    private func makeFixture(kIn: Int, nOut: Int) throws -> Fixture {
        let device = Device.shared
        let nExperts = 16
        let runs: [(expert: UInt32, count: Int)] = [(1, 50), (4, 77), (7, 1), (9, 72)]
        let mTotal = runs.reduce(0) { $0 + $1.count }
        let t = 25

        var idxHost = [UInt32]()
        for r in runs { idxHost.append(contentsOf: [UInt32](repeating: r.expert, count: r.count)) }
        var srcHost = [UInt32]()
        var s: UInt64 = 0x5151_7171
        for _ in 0 ..< mTotal {
            s = s &* 2_862_933_555_777_941_757 &+ 3_037_000_493
            srcHost.append(UInt32(s % UInt64(t)))
        }

        func u32Tensor(_ host: [UInt32], shape: [Int]) -> Tensor {
            let buf = device.makeBuffer(length: host.count * 4)
            host.withUnsafeBytes { _ = memcpy(buf.contents(), $0.baseAddress!, host.count * 4) }
            return Tensor(buffer: buf, offset: 0, shape: shape, dtype: .u32)
        }

        let hHost = bf16Payload(count: t * kIn, seed: 0xDEAD_BEEF)
        let hBuf = device.makeBuffer(length: hHost.count * 2)
        hHost.withUnsafeBytes { _ = memcpy(hBuf.contents(), $0.baseAddress!, hHost.count * 2) }
        let hRows = Tensor(buffer: hBuf, offset: 0, shape: [t, kIn], dtype: .bf16)

        let packs = kIn / 8
        let groups = kIn / 16
        func bank(_ seed: UInt32) -> (Tensor, Tensor) {
            var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
            for e in 0 ..< nExperts {
                for r in 0 ..< nOut {
                    for p in 0 ..< packs {
                        wHost[(e * nOut + r) * packs + p] =
                            0x1111_1111 &+ UInt32(e) &* 0x0101 &+ UInt32(p % 5) &+ seed
                    }
                }
            }
            var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
            for i in 0 ..< sHost.count { sHost[i] = 0x40 &+ UInt8((i &+ Int(seed)) % 12) }
            let wBuf = device.makeBuffer(length: wHost.count * 4)
            let sBuf = device.makeBuffer(length: sHost.count)
            wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
            sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
            return (
                Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32),
                Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)
            )
        }
        let (w1, s1) = bank(0)
        let (w2, s2) = bank(7)
        return Fixture(
            device: device, mTotal: mTotal, t: t, kIn: kIn, nOut: nOut,
            nExperts: nExperts, hRows: hRows,
            srcTok: u32Tensor(srcHost, shape: [mTotal]),
            indices: u32Tensor(idxHost, shape: [mTotal]),
            weight: w1, scales: s1, weight2: w2, scales2: s2)
    }

    private func emptyBf16(_ shape: [Int], _ device: Device) -> Tensor {
        let t = Tensor.empty(shape: shape, dtype: .bf16, device: device)
        memset(t.buffer.contents().advanced(by: t.offset), 0, t.elementCount * 2)
        return t
    }

    // MARK: - Lever 1: skip identity re-sort

    @Test("callGather sortedIndices path is byte-identical to the re-sort path")
    func skipSortCallGatherByteIdentical() throws {
        guard mppAvailable else { return }  // MPP unavailable: skip, not fail
        for (kIn, nOut) in [(2048, 512), (512, 2048)] {
            let fx = try makeFixture(kIn: kIn, nOut: nOut)
            let device = fx.device
            // Input rows already in plan order (what MoELayer's sorted
            // path feeds V2): materialise the sorted gather up front.
            let cmd0 = device.makeCommandBuffer()
            let x = Ops.gather(table: fx.hRows, tokenIds: fx.srcTok, on: cmd0)
            cmd0.commit(); cmd0.waitUntilCompleted()

            let outRef = emptyBf16([fx.mTotal, nOut], device)
            let cmdA = device.makeCommandBuffer()
            LagunaNvfp4ExpertGather.callGather(
                input: x, weight: fx.weight, scales: fx.scales, indices: fx.indices,
                mTotal: fx.mTotal, nOut: nOut, kIn: kIn,
                on: cmdA, device: device, into: outRef)
            cmdA.commit(); cmdA.waitUntilCompleted()
            #expect(cmdA.status == .completed)

            let outLever = emptyBf16([fx.mTotal, nOut], device)
            let cmdB = device.makeCommandBuffer()
            LagunaNvfp4ExpertGather.callGather(
                input: x, weight: fx.weight, scales: fx.scales, indices: fx.indices,
                mTotal: fx.mTotal, nOut: nOut, kIn: kIn,
                sortedIndices: true,
                on: cmdB, device: device, into: outLever)
            cmdB.commit(); cmdB.waitUntilCompleted()
            #expect(cmdB.status == .completed)

            expectByteIdentical(outRef, outLever, "skipSort callGather k\(kIn)")
        }
    }

    @Test("gateUpChain(skipSort) + downChain match the base V2 reference chain byte-exactly")
    func skipSortChainByteIdentical() throws {
        guard mppAvailable else { return }
        let kIn = 2048, nOut = 512
        let fx = try makeFixture(kIn: kIn, nOut: nOut)
        let device = fx.device
        let levers = LagunaNvfp4ExpertGather.Levers(skipSort: true)

        // Reference: exactly what the base fused-gateup branch does —
        // materialise gather, V2 pair (internal identity re-sort +
        // scatter-back), separate SwiGLU, V2 down (identity re-sort).
        let refGate = emptyBf16([fx.mTotal, nOut], device)
        let refUp = emptyBf16([fx.mTotal, nOut], device)
        let refDown = emptyBf16([fx.mTotal, kIn], device)
        let cmdA = device.makeCommandBuffer()
        let gathered = Ops.gather(table: fx.hRows, tokenIds: fx.srcTok, on: cmdA)
        LagunaNvfp4ExpertGather.callGatherPair(
            input: gathered,
            weight0: fx.weight, scales0: fx.scales, out0: refGate,
            weight1: fx.weight2, scales1: fx.scales2, out1: refUp,
            indices: fx.indices,
            mTotal: fx.mTotal, nOut: nOut, kIn: kIn,
            on: cmdA, device: device)
        let refInner = Ops.swiglu(gate: refGate, up: refUp, on: cmdA, into: refGate)
        LagunaNvfp4ExpertGather.callGather(
            input: refInner, weight: fx.weight, scales: fx.scales, indices: fx.indices,
            mTotal: fx.mTotal, nOut: kIn, kIn: nOut,
            on: cmdA, device: device, into: refDown)
        cmdA.commit(); cmdA.waitUntilCompleted()
        #expect(cmdA.status == .completed)

        // Lever chain.
        let levGate = emptyBf16([fx.mTotal, nOut], device)
        let levUp = emptyBf16([fx.mTotal, nOut], device)
        let levDown = emptyBf16([fx.mTotal, kIn], device)
        let cmdB = device.makeCommandBuffer()
        let levInner = LagunaNvfp4ExpertGather.gateUpChain(
            hRows: fx.hRows, srcTok: fx.srcTok, indices: fx.indices,
            gateWeight: fx.weight, gateScales: fx.scales,
            upWeight: fx.weight2, upScales: fx.scales2,
            gateOut: levGate, upOut: levUp,
            mTotal: fx.mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0,
            levers: levers, on: cmdB, device: device)
        LagunaNvfp4ExpertGather.downChain(
            inner: levInner, indices: fx.indices,
            weight: fx.weight, scales: fx.scales,
            mTotal: fx.mTotal, nOut: kIn, kIn: nOut,
            blockSize: 16, global: 1.0,
            levers: levers, on: cmdB, device: device, into: levDown)
        cmdB.commit(); cmdB.waitUntilCompleted()
        #expect(cmdB.status == .completed)

        expectByteIdentical(refInner, levInner, "skipSort chain inner")
        expectByteIdentical(refUp, levUp, "skipSort chain up")
        expectByteIdentical(refDown, levDown, "skipSort chain down")
    }

    // MARK: - Lever 2: fused SiLU·mul in the scatter-back

    /// Reference chain (base V2 semantics) built from public pieces;
    /// returns (inner, up, down) for byte-comparison.
    private func referenceChain(
        _ fx: Fixture, nOut: Int, kIn: Int
    ) -> (inner: Tensor, up: Tensor, down: Tensor) {
        let device = fx.device
        let refGate = emptyBf16([fx.mTotal, nOut], device)
        let refUp = emptyBf16([fx.mTotal, nOut], device)
        let refDown = emptyBf16([fx.mTotal, kIn], device)
        let cmd = device.makeCommandBuffer()
        let gathered = Ops.gather(table: fx.hRows, tokenIds: fx.srcTok, on: cmd)
        LagunaNvfp4ExpertGather.callGatherPair(
            input: gathered,
            weight0: fx.weight, scales0: fx.scales, out0: refGate,
            weight1: fx.weight2, scales1: fx.scales2, out1: refUp,
            indices: fx.indices,
            mTotal: fx.mTotal, nOut: nOut, kIn: kIn,
            on: cmd, device: device)
        let refInner = Ops.swiglu(gate: refGate, up: refUp, on: cmd, into: refGate)
        LagunaNvfp4ExpertGather.callGather(
            input: refInner, weight: fx.weight, scales: fx.scales, indices: fx.indices,
            mTotal: fx.mTotal, nOut: kIn, kIn: nOut,
            on: cmd, device: device, into: refDown)
        cmd.commit(); cmd.waitUntilCompleted()
        return (refInner, refUp, refDown)
    }

    private func leverChain(
        _ fx: Fixture, nOut: Int, kIn: Int,
        levers: LagunaNvfp4ExpertGather.Levers
    ) -> (inner: Tensor, down: Tensor) {
        let device = fx.device
        let levGate = emptyBf16([fx.mTotal, nOut], device)
        let levUp = emptyBf16([fx.mTotal, nOut], device)
        let levDown = emptyBf16([fx.mTotal, kIn], device)
        let cmd = device.makeCommandBuffer()
        let inner = LagunaNvfp4ExpertGather.gateUpChain(
            hRows: fx.hRows, srcTok: fx.srcTok, indices: fx.indices,
            gateWeight: fx.weight, gateScales: fx.scales,
            upWeight: fx.weight2, upScales: fx.scales2,
            gateOut: levGate, upOut: levUp,
            mTotal: fx.mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0,
            levers: levers, on: cmd, device: device)
        LagunaNvfp4ExpertGather.downChain(
            inner: inner, indices: fx.indices,
            weight: fx.weight, scales: fx.scales,
            mTotal: fx.mTotal, nOut: kIn, kIn: nOut,
            blockSize: 16, global: 1.0,
            levers: levers, on: cmd, device: device, into: levDown)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)
        return (inner, levDown)
    }

    @Test("fusedSilu chain is byte-identical to the base chain (with and without skipSort)")
    func fusedSiluByteIdentical() throws {
        guard mppAvailable else { return }
        let kIn = 2048, nOut = 512
        let fx = try makeFixture(kIn: kIn, nOut: nOut)
        let ref = referenceChain(fx, nOut: nOut, kIn: kIn)

        // fusedSilu alone (skipSort off → internal identity re-sort).
        let alone = leverChain(
            fx, nOut: nOut, kIn: kIn,
            levers: .init(skipSort: false, fusedSilu: true))
        expectByteIdentical(ref.inner, alone.inner, "fusedSilu-alone inner")
        expectByteIdentical(ref.down, alone.down, "fusedSilu-alone down")

        // fusedSilu + skipSort stacked.
        let stacked = leverChain(
            fx, nOut: nOut, kIn: kIn,
            levers: .init(skipSort: true, fusedSilu: true))
        expectByteIdentical(ref.inner, stacked.inner, "fusedSilu+skipSort inner")
        expectByteIdentical(ref.down, stacked.down, "fusedSilu+skipSort down")
    }

    @Test("mutation check: dropping the SiLU factor must change the inner bytes")
    func fusedSiluMutationCheck() throws {
        guard mppAvailable else { return }
        let kIn = 2048, nOut = 512
        let fx = try makeFixture(kIn: kIn, nOut: nOut)
        // Raw gate/up GEMM outputs (no SwiGLU): the pair call scattered
        // into gate/up targets, un-SwiGLU'd.
        let device = fx.device
        let rawGate = emptyBf16([fx.mTotal, nOut], device)
        let rawUp = emptyBf16([fx.mTotal, nOut], device)
        let cmd = device.makeCommandBuffer()
        let gathered = Ops.gather(table: fx.hRows, tokenIds: fx.srcTok, on: cmd)
        LagunaNvfp4ExpertGather.callGatherPair(
            input: gathered,
            weight0: fx.weight, scales0: fx.scales, out0: rawGate,
            weight1: fx.weight2, scales1: fx.scales2, out1: rawUp,
            indices: fx.indices,
            mTotal: fx.mTotal, nOut: nOut, kIn: kIn,
            on: cmd, device: device)
        cmd.commit(); cmd.waitUntilCompleted()

        // A silu-less mutant would produce plain g·u. Verify the fused
        // path's output differs from that on a large fraction of
        // elements — i.e. this suite HAS the power to catch a dropped
        // SiLU (the byte-identity tests above then pin the real math).
        let fused = leverChain(
            fx, nOut: nOut, kIn: kIn,
            levers: .init(skipSort: true, fusedSilu: true))
        let g = bytesU16(rawGate)
        let u = bytesU16(rawUp)
        let inner = bytesU16(fused.inner)
        var mulOnlyMatches = 0
        for i in 0 ..< inner.count {
            let gf = Float(bitPattern: UInt32(g[i]) << 16)
            let uf = Float(bitPattern: UInt32(u[i]) << 16)
            // bf16 RNE of the silu-less product.
            let prod = gf * uf
            var bits = prod.bitPattern
            let lsb = (bits >> 16) & 1
            bits = bits &+ 0x7FFF &+ lsb
            if UInt16(truncatingIfNeeded: bits >> 16) == inner[i] { mulOnlyMatches += 1 }
        }
        // Zeros match trivially (0·u == silu(0)·u == 0 at bf16) and the
        // synthetic NVFP4 banks produce many exact-zero products, so
        // don't demand a majority — demand a LARGE ABSOLUTE disagreement
        // (a silu-less mutant flips ≥20% of all elements, far beyond any
        // rounding-level fluke), which is what gives the byte-identity
        // tests above their power against a dropped SiLU.
        let differing = inner.count - mulOnlyMatches
        #expect(
            differing > inner.count / 5,
            "only \(differing)/\(inner.count) elements differ from the silu-less mutant — mutation would go undetected")
    }

    // MARK: - Lever 3: f16-native down input

    /// Exact bf16→f16 conversion with the gatherCastBf16ToF16 clamp —
    /// this is the CPU mirror of what the deleted cast pass does, used
    /// to check the f16 inner against the bf16 reference inner.
    private func bf16BitsToF16Bits(_ b: UInt16) -> UInt16 {
        let f = Float(bitPattern: UInt32(b) << 16)
        let c = max(-65504.0, min(65504.0, f))
        return Float16(c).bitPattern
    }

    @Test("f16Inner chain is byte-identical to the base chain (alone and stacked)")
    func f16InnerByteIdentical() throws {
        guard mppAvailable else { return }
        let kIn = 2048, nOut = 512
        let fx = try makeFixture(kIn: kIn, nOut: nOut)
        let ref = referenceChain(fx, nOut: nOut, kIn: kIn)
        let refInnerBytes = bytesU16(ref.inner)

        let combos: [(String, LagunaNvfp4ExpertGather.Levers)] = [
            ("f16Inner-alone", .init(skipSort: false, fusedSilu: false, f16Inner: true)),
            ("f16Inner+skipSort", .init(skipSort: true, fusedSilu: false, f16Inner: true)),
            ("f16Inner+fusedSilu", .init(skipSort: false, fusedSilu: true, f16Inner: true)),
            ("all-three", .init(skipSort: true, fusedSilu: true, f16Inner: true)),
        ]
        for (label, levers) in combos {
            let lev = leverChain(fx, nOut: nOut, kIn: kIn, levers: levers)
            // Down output is the chain's contract — must be bit-exact.
            expectByteIdentical(ref.down, lev.down, "\(label) down")
            // The f16 inner must equal the exact bf16→f16(+clamp) image
            // of the reference bf16 inner (the deleted cast's output).
            #expect(lev.inner.dtype == .f16, "\(label): inner must be f16")
            let innerBytes = bytesU16(lev.inner)
            var diffs = 0
            var firstDiff = -1
            for i in 0 ..< innerBytes.count
            where innerBytes[i] != bf16BitsToF16Bits(refInnerBytes[i]) {
                if firstDiff < 0 { firstDiff = i }
                diffs += 1
            }
            let msg = "\(label): f16 inner deviates from the exact cast image "
                + "on \(diffs) elements (first at \(firstDiff))"
            #expect(diffs == 0, Comment(rawValue: msg))
        }
    }

    @Test("f16 inner range safety: finite at 10x the measured |inner| bound, clamped at the half ceiling")
    func f16InnerRangeSafety() throws {
        guard mppAvailable else { return }
        // Measured Laguna inner activation bound: |silu(g)*u| ~ 800
        // (T23 calibration; see the kernel module docs). This test
        // drives the f16 store at >= 10x that (8000+) and past the half
        // ceiling (65504) to pin the saturating clamp — 80x documented
        // headroom between measured reality and the clamp.
        let device = Device.shared
        let mTotal = 64
        let dimN = 64
        func bf16Tensor(_ vals: [Float]) -> Tensor {
            let bits = vals.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let buf = device.makeBuffer(length: bits.count * 2)
            bits.withUnsafeBytes { _ = memcpy(buf.contents(), $0.baseAddress!, bits.count * 2) }
            return Tensor(buffer: buf, offset: 0, shape: [mTotal, dimN], dtype: .bf16)
        }
        // g = 15 → silu(g) ≈ g; u sweeps ±(1 … 5000) → products up to
        // ~±75000: covers the 8000-class range AND the clamp region.
        var gVals = [Float](), uVals = [Float]()
        for i in 0 ..< mTotal * dimN {
            gVals.append(15.0)
            let mag = 1.0 + Float(i % 100) * 50.0  // 1 … 4951
            uVals.append(i % 2 == 0 ? mag : -mag)
        }
        let gs = bf16Tensor(gVals)
        let us = bf16Tensor(uVals)
        let idBuf = Ops.identityRowsBuffer(minCount: mTotal, device: device)
        let ip = Tensor(buffer: idBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let innerB = emptyBf16([mTotal, dimN], device)
        let innerFBuf = device.makeBuffer(length: (mTotal + 64) * dimN * 2)
        let innerF = Tensor(buffer: innerFBuf, offset: 0, shape: [mTotal, dimN], dtype: .f16)
        let cmd = device.makeCommandBuffer()
        LagunaNvfp4ExpertGather.siluScatter(
            gs: gs, us: us, ip: ip, mTotal: mTotal, dimN: dimN, on: cmd, into: innerB)
        LagunaNvfp4ExpertGather.siluScatter(
            gs: gs, us: us, ip: ip, mTotal: mTotal, dimN: dimN, on: cmd, into: innerF)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        let bBytes = bytesU16(innerB)
        let fBytes = bytesU16(innerF)
        var maxAbs: Float = 0
        var clamped = 0
        var mismatches = 0
        for i in 0 ..< fBytes.count {
            let f16v = Float(Float16(bitPattern: fBytes[i]))
            #expect(f16v.isFinite, "f16 inner produced non-finite at \(i)")
            maxAbs = max(maxAbs, abs(f16v))
            if abs(f16v) == 65504.0 { clamped += 1 }
            // The f16 store must be the exact cast image of the bf16
            // store (no independent rounding path).
            if fBytes[i] != bf16BitsToF16Bits(bBytes[i]) { mismatches += 1 }
        }
        #expect(mismatches == 0, "f16 store deviates from the bf16 cast image on \(mismatches) elements")
        #expect(maxAbs >= 8000.0, "range test never reached 10x the measured bound (max |v| = \(maxAbs))")
        #expect(clamped > 0, "clamp region never exercised")
    }

    @Test("moeSortPlan over already-sorted plan ids is the identity (skipSort premise)")
    func sortPlanIdentityPremise() throws {
        guard mppAvailable else { return }
        // The skipSort lever's entire safety argument: on MoELayer's
        // sorted-plan path, V2's local k=1 re-sort of the (already
        // sorted) expert ids yields identity sourceRows/invPerm. Pin it.
        let fx = try makeFixture(kIn: 512, nOut: 2048)
        let device = fx.device
        let mTotal = fx.mTotal
        let se = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        let st = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        let ip = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        let cmd = device.makeCommandBuffer()
        Ops.moeSortPlan(
            topkIds: fx.indices, sortedExperts: se, sourceTokens: st,
            invPerm: ip, mTotal: mTotal, k: 1, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let stHost = st.toArray(as: UInt32.self)
        let ipHost = ip.toArray(as: UInt32.self)
        let seHost = se.toArray(as: UInt32.self)
        let idxHost = fx.indices.toArray(as: UInt32.self)
        for i in 0 ..< mTotal {
            #expect(stHost[i] == UInt32(i), "sourceRows[\(i)]=\(stHost[i]) not identity")
            #expect(ipHost[i] == UInt32(i), "invPerm[\(i)]=\(ipHost[i]) not identity")
            #expect(seHost[i] == idxHost[i], "sortedExperts[\(i)] changed")
        }
    }
}
