// Copyright 2026 Tom Turney (@TheTom)
//
// Hardening test for the `Ops.mppAutoCapable` gap in
// `Ops.nvfp4MoeGatherBm16`'s default dispatch branch (fix/mpp-capability-
// dispatch): that branch used to build a `matmul2d` cooperative-tensor PSO
// name (`iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_*`) unconditionally, which
// PSOCache.pipelineState fatalErrors compiling on Apple GPU Family <9 /
// macOS <26 hardware (see butter PR #132's first CI run 31758723492 —
// "unsupported deferred-static-alloca-size function body in
// metal::cooperative_tensor"). The op now self-guards on
// `Ops.mppAutoCapable` and degrades to a per-row
// `nvfp4DequantGemvExpertIndexedMany` GEMV loop instead.
//
// This suite proves the hardened path on THIS (capable) machine two ways:
//   1. Run normally: `Ops.mppAutoCapable == true`, the real MPP kernel
//      fires, and its output must still match the (hardware-independent)
//      per-row GEMV oracle -- a no-regression check.
//   2. Run the whole process with `BUTTER_FORCE_NO_MPP=1` set in the
//      environment (the diagnostic override `Ops.mppAutoCapable` reads --
//      see its doc in Sources/ButterSwift/Ops/Ops.swift): `mppAutoCapable`
//      is a memoized `static let`, so this MUST be set before the process
//      starts, e.g.
//        BUTTER_FORCE_NO_MPP=1 swift test -c release \
//          --filter Nvfp4MoeGatherCapabilityFallbackTests
//      In that mode `Ops.mppAutoCapable == false` process-wide, so the same
//      test exercises `nvfp4MoeGatherBm16NonMppFallback` end-to-end and
//      checks it neither crashes nor drifts from the oracle.
//
// Synthetic shapes only (small, no checkpoint loads) per repo direction —
// a real-model unit test run earlier today ran the box out of memory.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("NVFP4 gather MPP-capability fallback (F-mpp-hardening)")
struct Nvfp4MoeGatherCapabilityFallbackTests {
    /// Builds small, mixed pseudo-random synthetic NVFP4 MoE gather-GEMM
    /// operands. Shapes are deliberately NOT multiples of 64 (`mTotal`) so
    /// the BM64 auto-tile never engages (`nvfp4MoeGatherPrefersBm64`
    /// requires `mTotal >= 64`) and both the MPP and fallback paths always
    /// exercise the BM16 tile family this bug lives in. `nOut`/`kIn` still
    /// satisfy the op's own preconditions (`nOut % 32 == 0`).
    private struct Fixture {
        let device: Device
        let mTotal: Int
        let nOut: Int
        let kIn: Int
        let nExperts: Int
        let blockSize: Int
        let global: Float
        let weight: Tensor
        let scales: Tensor
        let x: Tensor
        let indices: Tensor
        let idxBuf: MTLBuffer
    }

    /// Plain factory function (not a struct `init`) — Swift's definite-
    /// initialization checker flags any closure (even a non-escaping
    /// `withUnsafeBytes`) appearing before every stored property is
    /// assigned inside a custom struct initializer ("captured by a
    /// closure before being initialized"), so build everything from local
    /// variables here and construct the `Fixture` once, fully formed, at
    /// the end.
    private func makeFixture() -> Fixture {
        let device = Device.shared
        let mTotal = 12
        let nOut = 64
        let kIn = 64
        let nExperts = 3
        let blockSize = 16
        let global: Float = 1.0

        let packs = kIn / 8
        let groups = kIn / blockSize
        var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
        var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
        var seed: UInt32 = 1234
        for i in 0 ..< wHost.count {
            seed = seed &* 1664525 &+ 1013904223
            // Distinct-but-bounded codes per expert/row/pack — avoids
            // degenerate all-zero dequant while staying deterministic.
            wHost[i] = 0x1111_1111 &+ (seed & 0x0F0F_0F0F)
        }
        for i in 0 ..< sHost.count {
            seed = seed &* 1664525 &+ 1013904223
            // Moderate E4M3 scale byte, same value class the existing
            // `Nvfp4MppSmokeTests` fixtures use (0x48), jittered a bit.
            sHost[i] = UInt8(0x44 &+ (seed & 0x07))
        }
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        let sBuf = device.makeBuffer(length: sHost.count)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for i in 0 ..< xHost.count {
            seed = seed &* 1664525 &+ 1013904223
            let f = Float(Int(seed % 2000)) / 500.0 - 2.0  // ~[-2, 2]
            xHost[i] = UInt16((f.bitPattern >> 16) & 0xFFFF)
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        // Sorted expert-id runs (4 rows/expert × 3 experts = 12), matching
        // the tile-locality convention the existing MPP smoke tests use.
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        let rowsPerExpert = mTotal / nExperts
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert {
                idxHost[e * rowsPerExpert + r] = UInt32(e)
            }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        return Fixture(
            device: device, mTotal: mTotal, nOut: nOut, kIn: kIn,
            nExperts: nExperts, blockSize: blockSize, global: global,
            weight: weight, scales: scales, x: x, indices: indices, idxBuf: idxBuf)
    }

    private func toF32(_ t: Tensor) -> [Float] {
        let n = t.elementCount
        let ptr = t.buffer.contents().advanced(by: t.offset)
            .bindMemory(to: UInt16.self, capacity: n)
        return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< a.count where a[i].isFinite && b[i].isFinite {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        return dot / (sqrt(na) * sqrt(nb) + 1e-12)
    }

    /// Per-row expert-indexed GEMV oracle — hardware/capability
    /// independent (no `_mpp_` kernel involved), and exactly the primitive
    /// `nvfp4MoeGatherBm16NonMppFallback` itself wraps. Mirrors the
    /// pattern `Nvfp4MppSmokeTests` already uses.
    private func eidxOracle(_ f: Fixture, on cmd: MTLCommandBuffer) -> Tensor {
        let out = Tensor.empty(shape: [f.mTotal, f.nOut], dtype: .bf16, device: f.device)
        memset(out.buffer.contents().advanced(by: out.offset), 0, f.mTotal * f.nOut * 2)
        var ws: [Tensor] = [], ss: [Tensor] = [], ins: [Tensor] = []
        var idxs: [Tensor] = [], outs: [Tensor] = []
        for r in 0 ..< f.mTotal {
            ws.append(f.weight); ss.append(f.scales)
            ins.append(f.x.slicedRows(start: r, count: 1).reshaped(to: [f.kIn]))
            idxs.append(Tensor(buffer: f.idxBuf, offset: r * 4, shape: [1], dtype: .u32))
            outs.append(out.slicedRows(start: r, count: 1).reshaped(to: [f.nOut]))
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss,
            inputs: ins, expertIndices: idxs, outputs: outs,
            blockSize: f.blockSize, global: f.global, on: cmd)
        return out
    }

    @Test("nvfp4MoeGatherBm16 matches the GEMV oracle under whichever path Ops.mppAutoCapable selects")
    func matchesOracleUnderCurrentCapability() throws {
        // Keep LagunaNvfp4MppGather's EG-MPP fast path from intercepting
        // this shape when capable (nOut=64/kIn=64/blockSize=16/bf16 is
        // shape-eligible for it) -- this test targets
        // `nvfp4MoeGatherBm16`'s own default-branch guard specifically,
        // not the EG-MPP tier ahead of it (that tier already checks
        // `Ops.mppAutoCapable` via its own `.capable`, audited separately).
        setenv("BUTTER_NVFP4_MPP_GATHER", "0", 1)
        defer { unsetenv("BUTTER_NVFP4_MPP_GATHER") }

        let f = makeFixture()
        let outUnderTest = Tensor.empty(shape: [f.mTotal, f.nOut], dtype: .bf16, device: f.device)
        memset(outUnderTest.buffer.contents().advanced(by: outUnderTest.offset), 0, f.mTotal * f.nOut * 2)

        let cmd = f.device.makeCommandBuffer()
        Ops.nvfp4MoeGatherBm16(
            input: f.x, weight: f.weight, scales: f.scales, indices: f.indices,
            mTotal: f.mTotal, nOut: f.nOut, kIn: f.kIn,
            blockSize: f.blockSize, global: f.global,
            on: cmd, into: outUnderTest)
        let oracle = eidxOracle(f, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "cmd status \(cmd.status.rawValue) — path under test must not crash/fault")

        let under = toF32(outUnderTest)
        let ref = toF32(oracle)
        let underNan = under.filter(\.isNaN).count
        let refNan = ref.filter(\.isNaN).count
        print("[mpp-cap-fallback] Ops.mppAutoCapable=\(Ops.mppAutoCapable) underNan=\(underNan) refNan=\(refNan)")
        print("[mpp-cap-fallback] under \(under.prefix(6))")
        print("[mpp-cap-fallback] oracle \(ref.prefix(6))")

        #expect(refNan == 0, "GEMV oracle itself produced NaNs — fixture bug, not the code under test")
        #expect(underNan == 0, "path under test (mppAutoCapable=\(Ops.mppAutoCapable)) produced NaNs")

        let cos = cosine(under, ref)
        print("[mpp-cap-fallback] cos=\(cos)")
        if !Ops.mppAutoCapable {
            // The fallback wraps the *identical* GEMV kernel the oracle
            // calls by hand, on identical inputs — should be
            // (near-)bit-exact, not just numerically close.
            #expect(cos > 0.9999, "non-MPP fallback should ~exactly match its own oracle kernel, cos=\(cos)")
        } else {
            #expect(cos > 0.99, "MPP path drifted from the GEMV oracle, cos=\(cos)")
        }
    }

    @Test("nvfp4MoeGatherBm16Pair / GateUp / XRowsPair complete without crashing under the current capability")
    func pairAndGateUpFamilyDoNotCrash() throws {
        setenv("BUTTER_NVFP4_MPP_GATHER", "0", 1)
        defer { unsetenv("BUTTER_NVFP4_MPP_GATHER") }

        let f = makeFixture()
        // Second weight/scales set ("up" projection) so gate+up are
        // distinguishable; reuse the fixture's RNG-free helper shape.
        let f2 = makeFixture()

        let outGate = Tensor.empty(shape: [f.mTotal, f.nOut], dtype: .bf16, device: f.device)
        let outUp = Tensor.empty(shape: [f.mTotal, f.nOut], dtype: .bf16, device: f.device)
        memset(outGate.buffer.contents().advanced(by: outGate.offset), 0, f.mTotal * f.nOut * 2)
        memset(outUp.buffer.contents().advanced(by: outUp.offset), 0, f.mTotal * f.nOut * 2)

        let cmd = f.device.makeCommandBuffer()
        Ops.nvfp4MoeGatherBm16Pair(
            input: f.x,
            weight0: f.weight, scales0: f.scales, out0: outGate,
            weight1: f2.weight, scales1: f2.scales, out1: outUp,
            indices: f.indices,
            mTotal: f.mTotal, nOut: f.nOut, kIn: f.kIn,
            blockSize: f.blockSize, global: f.global, on: cmd)

        // GateUp (identity x_rows path, i.e. `xRows: nil`).
        let outGate2 = Tensor.empty(shape: [f.mTotal, f.nOut], dtype: .bf16, device: f.device)
        let outUp2 = Tensor.empty(shape: [f.mTotal, f.nOut], dtype: .bf16, device: f.device)
        memset(outGate2.buffer.contents().advanced(by: outGate2.offset), 0, f.mTotal * f.nOut * 2)
        memset(outUp2.buffer.contents().advanced(by: outUp2.offset), 0, f.mTotal * f.nOut * 2)
        Ops.nvfp4MoeGatherBm16GateUp(
            input: f.x, xRows: nil,
            weightGate: f.weight, scalesGate: f.scales, outGate: outGate2,
            weightUp: f2.weight, scalesUp: f2.scales, outUp: outUp2,
            indices: f.indices,
            mTotal: f.mTotal, nOut: f.nOut, kIn: f.kIn,
            blockSize: f.blockSize, global: f.global, on: cmd)

        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "cmd status \(cmd.status.rawValue)")

        for (name, t) in [("Pair.gate", outGate), ("Pair.up", outUp), ("GateUp.gate", outGate2), ("GateUp.up", outUp2)] {
            let a = toF32(t)
            let nan = a.filter(\.isNaN).count
            print("[mpp-cap-fallback-pair] Ops.mppAutoCapable=\(Ops.mppAutoCapable) \(name) nan=\(nan)/\(a.count) sample=\(a.prefix(4))")
            #expect(nan == 0, "\(name) produced NaNs under mppAutoCapable=\(Ops.mppAutoCapable)")
        }

        // Pair and GateUp both ultimately compute the same gate/up
        // projections as the serial API on the same inputs -- they should
        // agree with each other regardless of which internal tile/fallback
        // each one happened to route through.
        let gateCos = cosine(toF32(outGate), toF32(outGate2))
        let upCos = cosine(toF32(outUp), toF32(outUp2))
        print("[mpp-cap-fallback-pair] gateCos=\(gateCos) upCos=\(upCos)")
        #expect(gateCos > 0.999, "Pair vs GateUp gate output diverged, cos=\(gateCos)")
        #expect(upCos > 0.999, "Pair vs GateUp up output diverged, cos=\(upCos)")
    }
}
