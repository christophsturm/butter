// Copyright 2026 Tom Turney (@TheTom)
//
// T34 kernel-level perf probe: T33 steel-structure NVFP4 gather GEMM
// (`Ops.nvfp4GatherSteel`, `BUTTER_NVFP4_GATHER_STEEL` opt-in path) vs the
// shipped bm16 MPP default (`Ops.nvfp4MoeGatherBm16` with no env override)
// on the real Laguna gate/up gather shape: 512 sorted rows across 8
// experts, kIn=2048, nOut=1024. T33's full-model MoE-phase timing measured
// mpp gateup at ~217ms/39 layers (~5.6ms/layer) and the un-hoisted steel
// kernel at ~743ms/39 layers (~19ms/layer, 3.4x worse), root-caused to a
// redundant per-`ti` NVFP4 weight dequant (4x repeated nibble-unpack +
// E4M3 scale decode per K-step). This probe measures the T34 hoisted
// kernel's isolated per-dispatch time against the same mpp baseline,
// same pattern as MoEInt2DenseQmmVsEgTests.swift (warmup iters discarded,
// then averaged wall-clock per dispatch via CFAbsoluteTimeGetCurrent).
//
//   swift test -c release --filter Nvfp4GatherSteelKernelBenchTests

import Foundation
import Metal
import IronSwift
import Testing

@testable import ButterSwift

@Suite("NVFP4 steel-structure gather GEMM kernel bench (T34)", .serialized)
struct Nvfp4GatherSteelKernelBenchTests {
    static let steelAvailable: Bool = MTLCreateSystemDefaultDevice() != nil

    static func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let cmd = Device.shared.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)
    }

    @Test(
        "steel vs mpp-default per-dispatch time, gateup shape (m=512,n=1024,k=2048,8 experts)",
        .enabled(if: steelAvailable))
    func steelVsMppDefault() throws {
        let mTotal = 512
        let nOut = 1024
        let kIn = 2048
        let nExperts = 8
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        let rowsPerExpert = mTotal / nExperts
        let iters = 8
        let warmup = 2

        let device = Device.shared

        // Sorted indices: 8 experts, 64 sorted rows each, contiguous runs
        // (no straddle) -- the steady-state gateup dispatch shape.
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert { idxHost[e * rowsPerExpert + r] = UInt32(e) }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
        for i in 0 ..< wHost.count { wHost[i] = 0x3210_7654 &+ UInt32(i % 251) }
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)

        var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
        for i in 0 ..< sHost.count { sHost[i] = UInt8(0x40 &+ UInt8(i % 24)) }
        let sBuf = device.makeBuffer(length: sHost.count)
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            for k in 0 ..< kIn {
                let v = Float(0.35) * sin(Float(k) * 0.7 + Float(r) * 0.13)
                xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)

        func timeSteel() -> Double {
            for _ in 0 ..< warmup {
                Self.runAndWait { cmd in
                    Ops.nvfp4GatherSteel(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outSteel)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                Self.runAndWait { cmd in
                    Ops.nvfp4GatherSteel(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outSteel)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        // mpp default: nvfp4MoeGatherBm16 dispatches steel only when
        // BUTTER_NVFP4_GATHER_STEEL is set -- explicitly clear it so this
        // measures the shipped default path regardless of outer-env state.
        func timeMppDefault() -> Double {
            unsetenv("BUTTER_NVFP4_GATHER_STEEL")
            for _ in 0 ..< warmup {
                Self.runAndWait { cmd in
                    Ops.nvfp4MoeGatherBm16(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outMpp)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                Self.runAndWait { cmd in
                    Ops.nvfp4MoeGatherBm16(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outMpp)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        let mppMs = timeMppDefault()
        let steelMs = timeSteel()

        print("[T34-kernel-bench] shape m=\(mTotal) n=\(nOut) k=\(kIn) experts=\(nExperts)")
        print("[T34-kernel-bench] mpp-default   = \(mppMs) ms/dispatch")
        print("[T34-kernel-bench] steel-hoisted = \(steelMs) ms/dispatch")
        print("[T34-kernel-bench] ratio (steel/mpp) = \(steelMs / mppMs)")
        // No hard pass/fail gate here -- this is a diagnostic probe per the
        // T34 mission brief, not a correctness test. The README records the
        // number either way (win or residual gap) rather than gating CI on
        // a perf threshold that could flake with thermal/host drift.
    }

    /// Production-scale probe: the small 512-row/8-expert shape above
    /// matches T33's own stated "gateup shape" language, but the REAL
    /// Laguna XS dispatch is `mTotal = t * topK` with `topK=8`,
    /// `nExperts=256` (see `MoELayer.decodeManyNvfp4`) -- e.g. `t=512`
    /// tokens gives `mTotal=4096` sorted rows spread across up to 256
    /// experts, not 8. That means far shorter average per-expert runs
    /// (mTotal/nExperts ~= 16 rows here vs 64 above) and many more
    /// same-tile expert-boundary straddles per 64-row tile than the probe
    /// above exercises. Full-model `BUTTER_LAGUNA_MOE_PHASE_TIMING=1`
    /// measurement after the T34 hoist still shows steel losing to mpp
    /// (~2-3x, down from T33's original 3.4-4.0x) despite this probe's
    /// small-shape win above -- this second probe exists to check whether
    /// that discrepancy is a shape/fragmentation artifact of the first
    /// probe rather than a real win reversing at production scale.
    @Test(
        "steel vs mpp-default per-dispatch time, PRODUCTION shape (m=4096,n=512,k=2048,256 experts,~16 rows/expert)",
        .enabled(if: steelAvailable))
    func steelVsMppDefaultProductionScale() throws {
        let t = 512
        let topK = 8
        let mTotal = t * topK  // 4096, matches MoELayer's `t * router.topK`
        let nOut = 512  // matches Laguna XS gate/up nOut (T33 header)
        let kIn = 2048
        let nExperts = 256  // matches Laguna XS router.nExperts
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        let iters = 8
        let warmup = 2

        let device = Device.shared

        // Sorted indices: uniform ~mTotal/nExperts rows per expert (16),
        // ascending expert id -- the steady-state moeSortPlan output shape
        // for a roughly-uniform top-k routing distribution. Real routing
        // can be skewed (some experts hotter than others), but uniform is
        // the WORST case for tile fragmentation (every expert's run is
        // short, so every 64-row tile straddles ~4 experts) -- if steel
        // still wins here despite that, the small-shape probe's win was
        // real and the full-model gap must come from elsewhere (attention
        // interleaving, occupancy, etc.); if it loses, tile-granularity
        // fragmentation vs the mpp default's 16-row BM16 tiles is the
        // likely next lever, not something this session's B-dequant hoist
        // addresses.
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        let rowsPerExpert = mTotal / nExperts
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert { idxHost[e * rowsPerExpert + r] = UInt32(e) }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
        for i in 0 ..< wHost.count { wHost[i] = 0x3210_7654 &+ UInt32(i % 251) }
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)

        var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
        for i in 0 ..< sHost.count { sHost[i] = UInt8(0x40 &+ UInt8(i % 24)) }
        let sBuf = device.makeBuffer(length: sHost.count)
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            for k in 0 ..< kIn {
                let v = Float(0.35) * sin(Float(k) * 0.7 + Float(r) * 0.13)
                xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)

        func timeSteel() -> Double {
            for _ in 0 ..< warmup {
                Self.runAndWait { cmd in
                    Ops.nvfp4GatherSteel(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outSteel)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                Self.runAndWait { cmd in
                    Ops.nvfp4GatherSteel(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outSteel)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        func timeMppDefault() -> Double {
            unsetenv("BUTTER_NVFP4_GATHER_STEEL")
            for _ in 0 ..< warmup {
                Self.runAndWait { cmd in
                    Ops.nvfp4MoeGatherBm16(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outMpp)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                Self.runAndWait { cmd in
                    Ops.nvfp4MoeGatherBm16(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outMpp)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        let mppMs = timeMppDefault()
        let steelMs = timeSteel()

        print("[T34-kernel-bench-prod] shape m=\(mTotal) n=\(nOut) k=\(kIn) experts=\(nExperts) rowsPerExpert=\(rowsPerExpert)")
        print("[T34-kernel-bench-prod] mpp-default   = \(mppMs) ms/dispatch")
        print("[T34-kernel-bench-prod] steel-hoisted = \(steelMs) ms/dispatch")
        print("[T34-kernel-bench-prod] ratio (steel/mpp) = \(steelMs / mppMs)")
    }

    /// Isolates whether the production-scale loss above is driven by total
    /// row count (`mTotal=4096`) or by per-expert RUN LENGTH
    /// (`mTotal/nExperts`): same `mTotal=4096` as the probe above, but only
    /// 8 experts (512 rows/expert, long contiguous runs) instead of 256
    /// (16 rows/expert). If steel wins here despite the 8x larger `mTotal`
    /// than the first probe, that confirms run length -- not row count --
    /// is what flips the result, and the production-scale loss above is a
    /// tile-fragmentation effect (many short per-expert runs forcing
    /// repeated full-K-reduction passes per 64-row tile), not merely "more
    /// total work."
    @Test(
        "steel vs mpp-default per-dispatch time, m=4096 but LONG runs (8 experts, 512 rows/expert)",
        .enabled(if: steelAvailable))
    func steelVsMppDefaultLongRunsAtScale() throws {
        let mTotal = 4096
        let nOut = 512
        let kIn = 2048
        let nExperts = 8
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        let iters = 8
        let warmup = 2

        let device = Device.shared

        var idxHost = [UInt32](repeating: 0, count: mTotal)
        let rowsPerExpert = mTotal / nExperts
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert { idxHost[e * rowsPerExpert + r] = UInt32(e) }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
        for i in 0 ..< wHost.count { wHost[i] = 0x3210_7654 &+ UInt32(i % 251) }
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)

        var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
        for i in 0 ..< sHost.count { sHost[i] = UInt8(0x40 &+ UInt8(i % 24)) }
        let sBuf = device.makeBuffer(length: sHost.count)
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            for k in 0 ..< kIn {
                let v = Float(0.35) * sin(Float(k) * 0.7 + Float(r) * 0.13)
                xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)

        func timeSteel() -> Double {
            for _ in 0 ..< warmup {
                Self.runAndWait { cmd in
                    Ops.nvfp4GatherSteel(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outSteel)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                Self.runAndWait { cmd in
                    Ops.nvfp4GatherSteel(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outSteel)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        func timeMppDefault() -> Double {
            unsetenv("BUTTER_NVFP4_GATHER_STEEL")
            for _ in 0 ..< warmup {
                Self.runAndWait { cmd in
                    Ops.nvfp4MoeGatherBm16(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outMpp)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                Self.runAndWait { cmd in
                    Ops.nvfp4MoeGatherBm16(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outMpp)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        let mppMs = timeMppDefault()
        let steelMs = timeSteel()

        print("[T34-kernel-bench-longrun] shape m=\(mTotal) n=\(nOut) k=\(kIn) experts=\(nExperts) rowsPerExpert=\(rowsPerExpert)")
        print("[T34-kernel-bench-longrun] mpp-default   = \(mppMs) ms/dispatch")
        print("[T34-kernel-bench-longrun] steel-hoisted = \(steelMs) ms/dispatch")
        print("[T34-kernel-bench-longrun] ratio (steel/mpp) = \(steelMs / mppMs)")
    }

    // MARK: - T35: BM16 row-tile variant

    /// Shared shape harness for the T35 BM16-vs-BM64-vs-mpp comparisons
    /// below -- same body as the three T34 probes above, parameterized by
    /// `mTotal`/`nOut`/`kIn`/`nExperts` and printing mpp/BM64-steel/BM16-
    /// steel side by side so the crossover (which run length favors which
    /// tile size) is visible in one place instead of three separate runs.
    static func runBm16Probe(
        label: String, mTotal: Int, nOut: Int, kIn: Int, nExperts: Int
    ) {
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        let iters = 8
        let warmup = 2
        let device = Device.shared
        let rowsPerExpert = mTotal / nExperts

        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert { idxHost[e * rowsPerExpert + r] = UInt32(e) }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
        for i in 0 ..< wHost.count { wHost[i] = 0x3210_7654 &+ UInt32(i % 251) }
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)

        var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
        for i in 0 ..< sHost.count { sHost[i] = UInt8(0x40 &+ UInt8(i % 24)) }
        let sBuf = device.makeBuffer(length: sHost.count)
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            for k in 0 ..< kIn {
                let v = Float(0.35) * sin(Float(k) * 0.7 + Float(r) * 0.13)
                xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outBm16 = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)

        func timeIt(_ body: @escaping (MTLCommandBuffer) -> Void) -> Double {
            for _ in 0 ..< warmup { runAndWait(body) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters { runAndWait(body) }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        unsetenv("BUTTER_NVFP4_GATHER_STEEL")
        let mppMs = timeIt { cmd in
            Ops.nvfp4MoeGatherBm16(
                input: x, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: 1.0, on: cmd, into: outMpp)
        }
        let bm64Ms = timeIt { cmd in
            Ops.nvfp4GatherSteel(
                input: x, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: 1.0, on: cmd, into: outSteel)
        }
        let bm16Ms = timeIt { cmd in
            Ops.nvfp4GatherSteelBm16(
                input: x, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: 1.0, on: cmd, into: outBm16)
        }

        print("[T35-kernel-bench-\(label)] shape m=\(mTotal) n=\(nOut) k=\(kIn) experts=\(nExperts) rowsPerExpert=\(rowsPerExpert)")
        print("[T35-kernel-bench-\(label)] mpp-default      = \(mppMs) ms/dispatch")
        print("[T35-kernel-bench-\(label)] steel-BM64-hoist = \(bm64Ms) ms/dispatch (ratio vs mpp \(bm64Ms / mppMs))")
        print("[T35-kernel-bench-\(label)] steel-BM16       = \(bm16Ms) ms/dispatch (ratio vs mpp \(bm16Ms / mppMs), ratio vs BM64 \(bm16Ms / bm64Ms))")
    }

    /// Main T35 target: the real production shape (m=4096, n=512, k=2048,
    /// 256 experts, ~16 rows/expert) where T34 measured BM64-steel losing
    /// 1.94-2.54x to mpp. BM16's whole reason for existing is to win here.
    @Test(
        "T35 BM16 vs BM64-steel vs mpp, PRODUCTION shape (m=4096,n=512,k=2048,256 experts,~16 rows/expert)",
        .enabled(if: steelAvailable))
    func bm16VsBm64VsMppProductionScale() throws {
        Self.runBm16Probe(label: "prod", mTotal: 4096, nOut: 512, kIn: 2048, nExperts: 256)
    }

    /// Regression check: the 64-row/8-expert shape where T34's BM64-steel
    /// already won (0.45-0.64x). BM16 tiles here are 4x smaller than a run,
    /// so each run now needs 4 BM16-tile dispatches' worth of K-reduction
    /// instead of 1 BM64-tile's -- expected to lose ground vs BM64 (each
    /// tile still pays a fixed per-tile scan/dequant-setup overhead, and a
    /// run this long no longer benefits from BM16's fragmentation fix,
    /// since there was no fragmentation to fix at this run length).
    @Test(
        "T35 BM16 vs BM64-steel vs mpp, gateup shape (m=512,n=1024,k=2048,8 experts,64 rows/expert)",
        .enabled(if: steelAvailable))
    func bm16VsBm64VsMppGateupShape() throws {
        Self.runBm16Probe(label: "gateup64", mTotal: 512, nOut: 1024, kIn: 2048, nExperts: 8)
    }

    /// Regression check: m=4096 but long runs (512 rows/expert, 8 experts)
    /// -- BM64-steel was near parity here (0.95). Same expectation as the
    /// 64-row shape: BM16 tiles are far smaller than the run, so this
    /// should cost BM16 relative to BM64, quantifying the per-tile
    /// overhead BM16 pays when fragmentation isn't actually a problem.
    @Test(
        "T35 BM16 vs BM64-steel vs mpp, m=4096 LONG runs (8 experts, 512 rows/expert)",
        .enabled(if: steelAvailable))
    func bm16VsBm64VsMppLongRunsAtScale() throws {
        Self.runBm16Probe(label: "longrun", mTotal: 4096, nOut: 512, kIn: 2048, nExperts: 8)
    }

    /// BM16 with bn doubled to 128 (subN 16->32) at production scale --
    /// spends the register headroom BM16 freed on a wider per-simdgroup
    /// N-tile instead of leaving it idle, amortizing the ragged-run scan's
    /// fixed per-tile cost over 2x the output columns per dispatch.
    @Test(
        "T35 BM16-bn128 vs BM16-bn64 vs mpp, PRODUCTION shape (m=4096,n=512,k=2048,256 experts)",
        .enabled(if: steelAvailable))
    func bm16Bn128VsBn64VsMppProductionScale() throws {
        let mTotal = 4096
        let nOut = 512
        let kIn = 2048
        let nExperts = 256
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        let iters = 8
        let warmup = 2
        let device = Device.shared
        let rowsPerExpert = mTotal / nExperts

        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert { idxHost[e * rowsPerExpert + r] = UInt32(e) }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
        for i in 0 ..< wHost.count { wHost[i] = 0x3210_7654 &+ UInt32(i % 251) }
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)

        var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
        for i in 0 ..< sHost.count { sHost[i] = UInt8(0x40 &+ UInt8(i % 24)) }
        let sBuf = device.makeBuffer(length: sHost.count)
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            for k in 0 ..< kIn {
                let v = Float(0.35) * sin(Float(k) * 0.7 + Float(r) * 0.13)
                xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outBm16 = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outBn128 = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)

        func timeIt(_ body: @escaping (MTLCommandBuffer) -> Void) -> Double {
            for _ in 0 ..< warmup { Self.runAndWait(body) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters { Self.runAndWait(body) }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        unsetenv("BUTTER_NVFP4_GATHER_STEEL")
        let mppMs = timeIt { cmd in
            Ops.nvfp4MoeGatherBm16(
                input: x, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: 1.0, on: cmd, into: outMpp)
        }
        let bm16Ms = timeIt { cmd in
            Ops.nvfp4GatherSteelBm16(
                input: x, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: 1.0, on: cmd, into: outBm16)
        }
        let bn128Ms = timeIt { cmd in
            Ops.nvfp4GatherSteelBm16N128(
                input: x, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: 1.0, on: cmd, into: outBn128)
        }

        print("[T35-kernel-bench-bn128] shape m=\(mTotal) n=\(nOut) k=\(kIn) experts=\(nExperts) rowsPerExpert=\(rowsPerExpert)")
        print("[T35-kernel-bench-bn128] mpp-default        = \(mppMs) ms/dispatch")
        print("[T35-kernel-bench-bn128] steel-BM16-bn64     = \(bm16Ms) ms/dispatch (ratio vs mpp \(bm16Ms / mppMs))")
        print("[T35-kernel-bench-bn128] steel-BM16-bn128    = \(bn128Ms) ms/dispatch (ratio vs mpp \(bn128Ms / mppMs), ratio vs bn64 \(bn128Ms / bm16Ms))")
    }

    // MARK: - T36: bench-vs-in-model discrepancy probes

    /// T36's own diagnosis: EVERY probe above (T33/T34/T35) hardcodes
    /// `nOut=512/1024, kIn=2048` -- Laguna XS gate/up's shape
    /// (`hidden=2048, moeIntermediate=512`,
    /// `Tests/ButterTests/Fixtures/laguna-xs-2.1-config.json`). Down-proj's
    /// REAL shape is the reverse -- `kIn=moeIntermediate=512,
    /// nOut=hidden=2048` (`MoELayer.swift` line ~2914:
    /// `nOut: hidden, kIn: moeIntermediate`) -- 4x shorter K-reduction, 4x
    /// wider N grid. No probe in this file has ever measured that shape.
    /// This one does, at production row-length (256 experts, ~16
    /// rows/expert, matching `bm16VsBm64VsMppProductionScale` above except
    /// for the swapped kIn/nOut) -- same serial single-dispatch pattern as
    /// down's real in-model call (`Ops.nvfp4MoeGatherBm16`, not the
    /// concurrent Pair gate+up uses), so dispatch pattern matches down's
    /// real call exactly and only shape was previously untested.
    @Test(
        "T36 BM16-steel vs mpp, DOWN-PROJ shape (m=4096,n=2048,k=512,256 experts,~16 rows/expert)",
        .enabled(if: steelAvailable))
    func bm16VsMppDownProjShape() throws {
        Self.runBm16Probe(label: "downproj", mTotal: 4096, nOut: 2048, kIn: 512, nExperts: 256)
    }

    /// T36's second diagnosed gap: gate+up's shape DOES match the T34/T35
    /// "production" probes (`nOut=512, kIn=2048, 256 experts`), but every
    /// probe in this file measures the SERIAL single-dispatch entry points
    /// (`Ops.nvfp4GatherSteel`/`nvfp4GatherSteelBm16`/`nvfp4MoeGatherBm16`).
    /// The real in-model gate+up call
    /// (`MoELayer.swift` line ~2948/~2897, `Ops.nvfp4MoeGatherBm16Pair`)
    /// issues gate and up as TWO `dispatchThreadgroups` calls on ONE
    /// `.concurrent` `MTLComputeCommandEncoder` -- not two independent
    /// command buffers. This probe uses the exact same
    /// `Ops.nvfp4MoeGatherBm16Pair` entry point MoELayer calls (env-toggled
    /// steel on/off, same as the other probes toggle it), at the identical
    /// production shape, to check whether concurrent-encoder dispatch
    /// overhead (not measured by any prior probe) explains gateup's
    /// 1.89x in-model delta despite the serial-dispatch probe showing
    /// near parity (0.76-1.06x) at the same shape.
    @Test(
        "T36 BM16-steel vs mpp, concurrent Pair dispatch (m=4096,n=512,k=2048,256 experts) -- matches in-model gateup call pattern",
        .enabled(if: steelAvailable))
    func bm16PairVsMppPairGateupShape() throws {
        let mTotal = 4096
        let nOut = 512
        let kIn = 2048
        let nExperts = 256
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        let iters = 8
        let warmup = 2
        let device = Device.shared
        let rowsPerExpert = mTotal / nExperts

        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert { idxHost[e * rowsPerExpert + r] = UInt32(e) }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        func makeWeight() -> (Tensor, Tensor) {
            var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
            for i in 0 ..< wHost.count { wHost[i] = 0x3210_7654 &+ UInt32(i % 251) }
            let wBuf = device.makeBuffer(length: wHost.count * 4)
            wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
            let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)
            var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
            for i in 0 ..< sHost.count { sHost[i] = UInt8(0x40 &+ UInt8(i % 24)) }
            let sBuf = device.makeBuffer(length: sHost.count)
            sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
            let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)
            return (weight, scales)
        }
        let (weightGate, scalesGate) = makeWeight()
        let (weightUp, scalesUp) = makeWeight()

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            for k in 0 ..< kIn {
                let v = Float(0.35) * sin(Float(k) * 0.7 + Float(r) * 0.13)
                xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outGateMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outUpMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outGateSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outUpSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)

        func timeIt(_ body: @escaping (MTLCommandBuffer) -> Void) -> Double {
            for _ in 0 ..< warmup { Self.runAndWait(body) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters { Self.runAndWait(body) }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        unsetenv("BUTTER_NVFP4_GATHER_STEEL")
        let mppMs = timeIt { cmd in
            Ops.nvfp4MoeGatherBm16Pair(
                input: x,
                weight0: weightGate, scales0: scalesGate, out0: outGateMpp,
                weight1: weightUp, scales1: scalesUp, out1: outUpMpp,
                indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: 1.0, on: cmd)
        }
        setenv("BUTTER_NVFP4_GATHER_STEEL", "1", 1)
        let steelMs = timeIt { cmd in
            Ops.nvfp4MoeGatherBm16Pair(
                input: x,
                weight0: weightGate, scales0: scalesGate, out0: outGateSteel,
                weight1: weightUp, scales1: scalesUp, out1: outUpSteel,
                indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: 1.0, on: cmd)
        }
        unsetenv("BUTTER_NVFP4_GATHER_STEEL")

        print("[T36-kernel-bench-pair] shape m=\(mTotal) n=\(nOut) k=\(kIn) experts=\(nExperts) rowsPerExpert=\(rowsPerExpert) (concurrent Pair dispatch, matches in-model gateup)")
        print("[T36-kernel-bench-pair] mpp-default-Pair  = \(mppMs) ms/dispatch")
        print("[T36-kernel-bench-pair] steel-BM16-Pair   = \(steelMs) ms/dispatch (ratio vs mpp \(steelMs / mppMs))")

        // T36 fix candidate: same two dispatches, but as separate serial
        // encoders instead of one `.concurrent` encoder -- see
        // `Ops.nvfp4GatherSteelBm16PairSerial`'s doc comment for the theory
        // (heavy-footprint steel kernel loses more to `.concurrent`
        // resource contention than it gains from latency-hiding).
        let outGateSerial = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outUpSerial = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let steelSerialMs = timeIt { cmd in
            Ops.nvfp4GatherSteelBm16PairSerial(
                input: x,
                weight0: weightGate, scales0: scalesGate, out0: outGateSerial,
                weight1: weightUp, scales1: scalesUp, out1: outUpSerial,
                indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: 1.0, on: cmd)
        }
        print("[T36-kernel-bench-pair] steel-BM16-PairSerial = \(steelSerialMs) ms/dispatch (ratio vs mpp \(steelSerialMs / mppMs), ratio vs steel-concurrent-Pair \(steelSerialMs / steelMs))")
    }

    /// T36 down-proj residual-gap probe. The dedicated down-shape probe
    /// above (`bm16VsMppDownProjShape`) already matches down's real shape
    /// (kIn=512, nOut=2048) AND down's real dispatch pattern (serial single
    /// encoder, not the concurrent Pair gate+up uses) -- yet still showed
    /// near parity (1.05x), nowhere near the in-model 1.61x. Every probe in
    /// this file up to this point still shares one property real MoELayer
    /// dispatch does NOT have: each timed op gets its OWN command buffer,
    /// submitted and waited on in isolation with nothing else queued. In
    /// the real model, down's encoder is the last of several encoders
    /// (gate+up Pair, swiglu, down) queued back-to-back on the SAME `work`
    /// command buffer per MoE layer, with no intermediate wait -- so
    /// whatever GPU-side residency/scheduling state gate+up's dispatch
    /// left behind is still live when down's encoder starts executing.
    /// This probe checks whether that adjacency itself costs something a
    /// fully-isolated single-encoder command buffer cannot show: time
    /// gate+up Pair alone, down alone, and gate+up Pair immediately
    /// followed by down on ONE shared command buffer (one commit+wait for
    /// both), then compare the chained time against the sum of the two
    /// isolated times for both mpp and steel.
    @Test(
        "T36 chained gateup+down on one command buffer vs isolated sum, production shape",
        .enabled(if: steelAvailable))
    func chainedGateupDownVsIsolatedSum() throws {
        let mTotal = 4096
        let nExperts = 256
        let blockSize = 16
        let hidden = 2048
        let moeIntermediate = 512
        let iters = 8
        let warmup = 2
        let device = Device.shared
        let rowsPerExpert = mTotal / nExperts

        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert { idxHost[e * rowsPerExpert + r] = UInt32(e) }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        func makeWeight(nOut: Int, kIn: Int) -> (Tensor, Tensor) {
            let packs = kIn / 8
            let groups = kIn / blockSize
            var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
            for i in 0 ..< wHost.count { wHost[i] = 0x3210_7654 &+ UInt32(i % 251) }
            let wBuf = device.makeBuffer(length: wHost.count * 4)
            wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
            let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)
            var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
            for i in 0 ..< sHost.count { sHost[i] = UInt8(0x40 &+ UInt8(i % 24)) }
            let sBuf = device.makeBuffer(length: sHost.count)
            sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
            let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)
            return (weight, scales)
        }
        let (weightGate, scalesGate) = makeWeight(nOut: moeIntermediate, kIn: hidden)
        let (weightUp, scalesUp) = makeWeight(nOut: moeIntermediate, kIn: hidden)
        let (weightDown, scalesDown) = makeWeight(nOut: hidden, kIn: moeIntermediate)

        func makeX(mTotal: Int, kIn: Int) -> Tensor {
            var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
            for r in 0 ..< mTotal {
                for k in 0 ..< kIn {
                    let v = Float(0.35) * sin(Float(k) * 0.7 + Float(r) * 0.13)
                    xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
                }
            }
            let xBuf = device.makeBuffer(length: xHost.count * 2)
            xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
            return Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)
        }
        let xGateUp = makeX(mTotal: mTotal, kIn: hidden)
        // Down's real input is gate+up's swiglu output ([mTotal, moeIntermediate]);
        // synthetic data here since this probe targets GPU scheduling/occupancy,
        // not numerics -- shape and dispatch adjacency are what's under test.
        let xDown = makeX(mTotal: mTotal, kIn: moeIntermediate)

        let outGate = Tensor.empty(shape: [mTotal, moeIntermediate], dtype: .bf16, device: device)
        let outUp = Tensor.empty(shape: [mTotal, moeIntermediate], dtype: .bf16, device: device)
        let outDown = Tensor.empty(shape: [mTotal, hidden], dtype: .bf16, device: device)

        func timeIt(_ body: @escaping (MTLCommandBuffer) -> Void) -> Double {
            for _ in 0 ..< warmup { Self.runAndWait(body) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters { Self.runAndWait(body) }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        func gateUpOnly(_ cmd: MTLCommandBuffer) {
            Ops.nvfp4MoeGatherBm16Pair(
                input: xGateUp,
                weight0: weightGate, scales0: scalesGate, out0: outGate,
                weight1: weightUp, scales1: scalesUp, out1: outUp,
                indices: indices,
                mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                blockSize: blockSize, global: 1.0, on: cmd)
        }
        func downOnly(_ cmd: MTLCommandBuffer) {
            Ops.nvfp4MoeGatherBm16(
                input: xDown, weight: weightDown, scales: scalesDown, indices: indices,
                mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                blockSize: blockSize, global: 1.0, on: cmd, into: outDown)
        }
        func chained(_ cmd: MTLCommandBuffer) {
            gateUpOnly(cmd)
            downOnly(cmd)
        }

        for (label, steelOn) in [("mpp", false), ("steel", true)] {
            if steelOn {
                setenv("BUTTER_NVFP4_GATHER_STEEL", "1", 1)
            } else {
                unsetenv("BUTTER_NVFP4_GATHER_STEEL")
            }
            let gateUpMs = timeIt(gateUpOnly)
            let downMs = timeIt(downOnly)
            let chainedMs = timeIt(chained)
            let isolatedSum = gateUpMs + downMs
            print("[T36-kernel-bench-chain-\(label)] gateup-alone=\(gateUpMs)ms down-alone=\(downMs)ms isolated-sum=\(isolatedSum)ms chained-one-cmdbuf=\(chainedMs)ms chained/isolatedSum=\(chainedMs / isolatedSum)")
        }
        unsetenv("BUTTER_NVFP4_GATHER_STEEL")
    }

    /// T36 root-cause probe. The Pair-dispatch finding above (steel/mpp
    /// ratio jumps from 1.06x at N=1 dispatch/cmd-buffer to 1.75-1.77x at
    /// N=2, REGARDLESS of concurrent vs serial encoding) suggests the N=1
    /// "isolated" bench methodology itself is the confound: a single
    /// dispatch's measured wall time is `fixed_overhead + device_time`,
    /// where `fixed_overhead` (CPU encode + commit + GPU round-trip wait)
    /// is roughly constant per command buffer regardless of what's
    /// dispatched. If mpp's true `device_time` is small relative to
    /// `fixed_overhead` (its 32-thread/1-simdgroup threadgroups are cheap)
    /// while steel's `device_time` is a much larger fraction of its own
    /// N=1 measurement (128-thread/4-simdgroup threadgroups, heavier real
    /// compute), then N=1 wall-clock ratios are dominated by
    /// `fixed_overhead` and UNDERSTATE the true device-time ratio -- and
    /// chaining more dispatches into one command buffer amortizes
    /// `fixed_overhead` away, revealing the real per-dispatch compute-time
    /// gap. This probe chains N identical same-kernel dispatches (same
    /// weight buffer reused, only `out` rotated through a small pool to
    /// avoid write-after-write hazards forcing serialization) into ONE
    /// command buffer for N in {1, 2, 4, 8} and reports ms/dispatch for
    /// mpp and steel at each N -- if the theory holds, mpp's ms/dispatch
    /// should drop sharply as N grows (fixed cost amortizing over more
    /// dispatches) while steel's should stay comparatively flat (already
    /// compute-time-dominated even at N=1), and the steel/mpp ratio should
    /// climb toward the full-model MoE-phase deltas (1.89x gateup, 1.61x
    /// down) as N increases -- explaining why 3 sessions running N=1
    /// probes (T33/T34/T35) all found the isolated number optimistic
    /// relative to the full 39-layer, many-dispatches-per-command-buffer
    /// model run.
    @Test(
        "T36 dispatch-chain-length probe: does N=1 fixed overhead mask steel's true device-time cost",
        .enabled(if: steelAvailable))
    func dispatchChainLengthRevealsTrueDeviceCost() throws {
        let mTotal = 4096
        let nOut = 512
        let kIn = 2048
        let nExperts = 256
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        let iters = 8
        let warmup = 2
        let device = Device.shared
        let rowsPerExpert = mTotal / nExperts

        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert { idxHost[e * rowsPerExpert + r] = UInt32(e) }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
        for i in 0 ..< wHost.count { wHost[i] = 0x3210_7654 &+ UInt32(i % 251) }
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)

        var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
        for i in 0 ..< sHost.count { sHost[i] = UInt8(0x40 &+ UInt8(i % 24)) }
        let sBuf = device.makeBuffer(length: sHost.count)
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            for k in 0 ..< kIn {
                let v = Float(0.35) * sin(Float(k) * 0.7 + Float(r) * 0.13)
                xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let maxN = 8
        let outsMpp = (0 ..< maxN).map { _ in Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device) }
        let outsSteel = (0 ..< maxN).map { _ in Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device) }

        func timeChain(steel: Bool, n: Int) -> Double {
            if steel {
                setenv("BUTTER_NVFP4_GATHER_STEEL", "1", 1)
            } else {
                unsetenv("BUTTER_NVFP4_GATHER_STEEL")
            }
            let outs = steel ? outsSteel : outsMpp
            func body(_ cmd: MTLCommandBuffer) {
                for i in 0 ..< n {
                    Ops.nvfp4MoeGatherBm16(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outs[i])
                }
            }
            for _ in 0 ..< warmup { Self.runAndWait(body) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters { Self.runAndWait(body) }
            let totalMs = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
            return totalMs / Double(n)
        }

        print("[T36-kernel-bench-chainlen] shape m=\(mTotal) n=\(nOut) k=\(kIn) experts=\(nExperts) (N same-kernel dispatches chained on one command buffer)")
        for n in [1, 2, 4, 8] {
            let mppPerDispatch = timeChain(steel: false, n: n)
            let steelPerDispatch = timeChain(steel: true, n: n)
            print("[T36-kernel-bench-chainlen] N=\(n): mpp=\(mppPerDispatch) ms/dispatch  steel=\(steelPerDispatch) ms/dispatch  ratio=\(steelPerDispatch / mppPerDispatch)")
        }
        unsetenv("BUTTER_NVFP4_GATHER_STEEL")
    }

    /// Same chain-length methodology as the probe above, but at down-proj's
    /// real shape (kIn=512, nOut=2048) instead of gate/up's (kIn=2048,
    /// nOut=512) -- down is never paired/concurrent-dispatched with
    /// anything in the real model (`Ops.nvfp4MoeGatherBm16` serial single
    /// dispatch only), so the gateup finding above (concurrent-dispatch
    /// occupancy asymmetry) does not apply to it by construction. This
    /// checks whether the SAME fixed-overhead-amortization mechanism
    /// (N=1 wall time dominated by per-command-buffer overhead, masking
    /// the true device-time ratio) reproduces at down's shape and, if so,
    /// how large the effect is there -- down's in-model gap is 1.61x,
    /// noticeably smaller than gateup's 1.89x, so a smaller amortization
    /// effect here would be consistent rather than contradictory.
    @Test(
        "T36 dispatch-chain-length probe, DOWN-PROJ shape (k=512,n=2048)",
        .enabled(if: steelAvailable))
    func dispatchChainLengthDownProjShape() throws {
        let mTotal = 4096
        let nOut = 2048
        let kIn = 512
        let nExperts = 256
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        let iters = 8
        let warmup = 2
        let device = Device.shared
        let rowsPerExpert = mTotal / nExperts

        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert { idxHost[e * rowsPerExpert + r] = UInt32(e) }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
        for i in 0 ..< wHost.count { wHost[i] = 0x3210_7654 &+ UInt32(i % 251) }
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)

        var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
        for i in 0 ..< sHost.count { sHost[i] = UInt8(0x40 &+ UInt8(i % 24)) }
        let sBuf = device.makeBuffer(length: sHost.count)
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            for k in 0 ..< kIn {
                let v = Float(0.35) * sin(Float(k) * 0.7 + Float(r) * 0.13)
                xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let maxN = 8
        let outsMpp = (0 ..< maxN).map { _ in Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device) }
        let outsSteel = (0 ..< maxN).map { _ in Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device) }

        func timeChain(steel: Bool, n: Int) -> Double {
            if steel {
                setenv("BUTTER_NVFP4_GATHER_STEEL", "1", 1)
            } else {
                unsetenv("BUTTER_NVFP4_GATHER_STEEL")
            }
            let outs = steel ? outsSteel : outsMpp
            func body(_ cmd: MTLCommandBuffer) {
                for i in 0 ..< n {
                    Ops.nvfp4MoeGatherBm16(
                        input: x, weight: weight, scales: scales, indices: indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn,
                        blockSize: blockSize, global: 1.0, on: cmd, into: outs[i])
                }
            }
            for _ in 0 ..< warmup { Self.runAndWait(body) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters { Self.runAndWait(body) }
            let totalMs = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
            return totalMs / Double(n)
        }

        print("[T36-kernel-bench-chainlen-down] shape m=\(mTotal) n=\(nOut) k=\(kIn) experts=\(nExperts) (N same-kernel dispatches chained on one command buffer, DOWN shape)")
        for n in [1, 2, 4, 8] {
            let mppPerDispatch = timeChain(steel: false, n: n)
            let steelPerDispatch = timeChain(steel: true, n: n)
            print("[T36-kernel-bench-chainlen-down] N=\(n): mpp=\(mppPerDispatch) ms/dispatch  steel=\(steelPerDispatch) ms/dispatch  ratio=\(steelPerDispatch / mppPerDispatch)")
        }
        unsetenv("BUTTER_NVFP4_GATHER_STEEL")
    }

    /// T36 routing-skew probe. Down's in-model gap (1.61x) is not explained
    /// by shape (`bm16VsMppDownProjShape` above: 1.05x at down's real
    /// shape with UNIFORM exactly-16-rows/expert routing), dispatch
    /// pattern (down is a plain serial single dispatch in both bench and
    /// model, no Pair/concurrent involvement), or fixed-overhead
    /// amortization (`dispatchChainLengthDownProjShape` above: ratio stays
    /// ~1.0-1.05x from N=1 to N=8). Every probe in this file (T33/T34/T35
    /// and this session's own two above) constructs `indices` as an exact
    /// round-robin: expert 0 gets rows 0..<16, expert 1 gets 16..<32, etc
    /// -- perfectly uniform run length with zero variance. Mission
    /// hypothesis (a) flags this directly: real top-k routing is not
    /// uniform, and `nvfp4GatherSteelPrefersBm16`'s BM16-vs-BM64 choice
    /// uses only the STATIC average `mTotal/nExperts`, blind to the real
    /// distribution's shape. This probe replaces the uniform assignment
    /// with an i.i.d. random draw (each of `mTotal` rows independently
    /// picks one of `nExperts` uniformly at random, then sorted by expert
    /// id) -- a more realistic model of what real top-k routing plus
    /// `moeSortPlan`'s sort actually produces: same mean run length (16)
    /// but real (roughly Poisson) variance, so some experts get 0 rows,
    /// others get 2-3x the mean, instead of every expert getting exactly
    /// 16. Tests both down's shape and gate/up's shape so the routing-skew
    /// effect (if any) can be compared against the already-diagnosed
    /// concurrent-Pair effect rather than confused with it.
    @Test(
        "T36 skewed (non-uniform) routing probe, down and gateup shapes",
        .enabled(if: steelAvailable))
    func skewedRoutingProbe() throws {
        let nExperts = 256
        let topK = 8
        let t = 512
        let mTotal = t * topK
        let blockSize = 16

        // i.i.d. random routing: each row picks an expert uniformly at
        // random (mean run length mTotal/nExperts = 16, Poisson-ish
        // variance from the multinomial draw) instead of the exact
        // round-robin every other probe in this file uses. Fixed seed via
        // a simple LCG for reproducibility across runs.
        func skewedIndices(seed: UInt64) -> ([UInt32], [Int]) {
            var state = seed
            func nextRand() -> UInt32 {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                return UInt32(truncatingIfNeeded: state >> 33)
            }
            var assigned = [UInt32](repeating: 0, count: mTotal)
            var counts = [Int](repeating: 0, count: nExperts)
            for i in 0 ..< mTotal {
                let e = Int(nextRand() % UInt32(nExperts))
                assigned[i] = UInt32(e)
                counts[e] += 1
            }
            // Sort rows by expert id (moeSortPlan's actual contract: rows
            // grouped contiguously by expert for the gather kernels).
            let sortedExpertIds = assigned.sorted()
            return (sortedExpertIds, counts)
        }
        let (idxHostSkewed, counts) = skewedIndices(seed: 0xC0FFEE)
        let minRun = counts.min() ?? 0
        let maxRun = counts.max() ?? 0
        let zeroRunExperts = counts.filter { $0 == 0 }.count

        let device = Device.shared
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHostSkewed.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        func runShape(label: String, nOut: Int, kIn: Int) {
            let packs = kIn / 8
            let groups = kIn / blockSize
            let iters = 8
            let warmup = 2

            var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
            for i in 0 ..< wHost.count { wHost[i] = 0x3210_7654 &+ UInt32(i % 251) }
            let wBuf = device.makeBuffer(length: wHost.count * 4)
            wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
            let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)

            var sHost = [UInt8](repeating: 0, count: nExperts * nOut * groups)
            for i in 0 ..< sHost.count { sHost[i] = UInt8(0x40 &+ UInt8(i % 24)) }
            let sBuf = device.makeBuffer(length: sHost.count)
            sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
            let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

            var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
            for r in 0 ..< mTotal {
                for k in 0 ..< kIn {
                    let v = Float(0.35) * sin(Float(k) * 0.7 + Float(r) * 0.13)
                    xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
                }
            }
            let xBuf = device.makeBuffer(length: xHost.count * 2)
            xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
            let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

            let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
            let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)

            func timeIt(_ body: @escaping (MTLCommandBuffer) -> Void) -> Double {
                for _ in 0 ..< warmup { Self.runAndWait(body) }
                let t0 = CFAbsoluteTimeGetCurrent()
                for _ in 0 ..< iters { Self.runAndWait(body) }
                return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
            }

            unsetenv("BUTTER_NVFP4_GATHER_STEEL")
            let mppMs = timeIt { cmd in
                Ops.nvfp4MoeGatherBm16(
                    input: x, weight: weight, scales: scales, indices: indices,
                    mTotal: mTotal, nOut: nOut, kIn: kIn,
                    blockSize: blockSize, global: 1.0, on: cmd, into: outMpp)
            }
            setenv("BUTTER_NVFP4_GATHER_STEEL", "1", 1)
            let steelMs = timeIt { cmd in
                Ops.nvfp4MoeGatherBm16(
                    input: x, weight: weight, scales: scales, indices: indices,
                    mTotal: mTotal, nOut: nOut, kIn: kIn,
                    blockSize: blockSize, global: 1.0, on: cmd, into: outSteel)
            }
            unsetenv("BUTTER_NVFP4_GATHER_STEEL")
            print("[T36-kernel-bench-skew-\(label)] mpp=\(mppMs)ms steel=\(steelMs)ms ratio=\(steelMs / mppMs)")
        }

        print("[T36-kernel-bench-skew] mTotal=\(mTotal) nExperts=\(nExperts) i.i.d.-random routing: minRun=\(minRun) maxRun=\(maxRun) zeroRunExperts=\(zeroRunExperts) (mean=\(mTotal / nExperts))")
        runShape(label: "downproj", nOut: 2048, kIn: 512)
        runShape(label: "gateup", nOut: 512, kIn: 2048)
    }
}
