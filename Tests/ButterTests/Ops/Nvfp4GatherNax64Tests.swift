// Copyright 2026 Tom Turney (@TheTom)
//
// T22 stage A/B kernel-level correctness: NAX-class gather QMM with TRUE
// BK=64 staging (`butter_nvfp4_gather_qmm_rhs_nax64_mpp_{bf16,f32}`) vs the
// expert-indexed GEMV oracle. Mirrors the pattern in Nvfp4MppSmokeTests.swift
// but specifically exercises 64-row tile-boundary-straddling expert runs and
// zero-count experts, the two failure classes noted in prior NAX spike
// history (benchmarks/laguna-xs/README.md T13).

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("NVFP4 NAX64 gather QMM", .serialized)
struct Nvfp4GatherNax64Tests {
    private var mppAvailable: Bool {
        guard let dev = MTLCreateSystemDefaultDevice() else { return false }
        return dev.supportsFamily(.apple9)
    }

    private func toF32(_ t: Tensor) -> [Float] {
        let n = t.elementCount
        let ptr = t.buffer.contents().advanced(by: t.offset)
            .bindMemory(to: UInt16.self, capacity: n)
        return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> (cos: Float, nan: Int) {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        var nanCount = 0
        for i in 0 ..< a.count {
            if a[i].isNaN { nanCount += 1; continue }
            guard b[i].isFinite else { continue }
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        return (dot / (sqrt(na) * sqrt(nb) + 1e-12), nanCount)
    }

    /// Oracle: one expert-indexed GEMV per row.
    private func eidxOracle(
        x: Tensor, weight: Tensor, scales: Tensor, idxBuf: MTLBuffer,
        mTotal: Int, kIn: Int, nOut: Int, device: Device, cmd: MTLCommandBuffer
    ) -> Tensor {
        let outEidx = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outEidx.buffer.contents().advanced(by: outEidx.offset), 0, mTotal * nOut * 2)
        var ws: [Tensor] = [], ss: [Tensor] = [], ins: [Tensor] = []
        var idxs: [Tensor] = [], outs: [Tensor] = []
        for r in 0 ..< mTotal {
            ws.append(weight); ss.append(scales)
            ins.append(x.slicedRows(start: r, count: 1).reshaped(to: [kIn]))
            idxs.append(Tensor(buffer: idxBuf, offset: r * 4, shape: [1], dtype: .u32))
            outs.append(outEidx.slicedRows(start: r, count: 1).reshaped(to: [nOut]))
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss,
            inputs: ins, expertIndices: idxs, outputs: outs,
            blockSize: 16, global: 1.0, on: cmd)
        return outEidx
    }

    @Test("NAX64 same-expert synthetic matches eidx", .enabled(if: true))
    func nax64SameExpertMatchesEidx() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let device = Device.shared
        let mTotal = 16
        let nOut = 512
        let kIn = 2048  // %64==0
        let nExperts = 4
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
        for e in 0 ..< nExperts {
            for r in 0 ..< nOut {
                for p in 0 ..< packs {
                    wHost[(e * nOut + r) * packs + p] = UInt32(0x1111_1111 &+ UInt32(e))
                }
            }
        }
        let sHost = [UInt8](repeating: 0x48, count: nExperts * nOut * groups)
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        let sBuf = device.makeBuffer(length: sHost.count)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        let xHost = [UInt16](repeating: 0x3F80, count: mTotal * kIn)
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let idxHost = [UInt32](repeating: 0, count: mTotal)
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let outNax = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outNax.buffer.contents().advanced(by: outNax.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4MoeGatherNax64(
            input: x, weight: weight, scales: scales, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: blockSize, global: 1.0, on: cmd, into: outNax)
        let outEidx = eidxOracle(
            x: x, weight: weight, scales: scales, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        let nax = toF32(outNax)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(nax, eidx)
        print("[nax64-synth] nan=\(nan) cos=\(cos) sample nax=\(nax.prefix(4)) eidx=\(eidx.prefix(4))")
        #expect(nan == 0, "NAX64 produced \(nan) NaNs")
        #expect(cos > 0.9999, "NAX64 vs eidx cos \(cos)")
    }

    /// Real Laguna gate weights, multi-expert runs that straddle the 64-row
    /// tile boundary (tile 0 = rows [0,64), tile 1 = [64,128), ...), plus
    /// experts that never appear (zero-count) interleaved between used ones.
    @Test("NAX64 real Laguna gate: boundary straddle + zero-count experts", .enabled(if: true))
    func nax64RealGateBoundaryStraddle() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("[nax64-gate] skip: no model at \(modelDir.path)")
            return
        }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        let nOut = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 64 == 0, "gate kIn \(kIn) must be %64==0 for NAX64")

        // Run lengths crafted to straddle tile boundaries at 64/128/192:
        // expert 2 spans rows [50,80) (crosses tile0|tile1 @64),
        // expert 9 spans rows [120,150) (crosses tile1|tile2 @128),
        // expert 20 spans rows [180,210) (crosses tile2|tile3 @192).
        // Experts 5, 15 are skipped entirely (zero-count).
        struct Run { let expert: UInt32; let start: Int; let end: Int }
        let runs = [
            Run(expert: 0, start: 0, end: 50),
            Run(expert: 2, start: 50, end: 80),
            Run(expert: 7, start: 80, end: 120),
            Run(expert: 9, start: 120, end: 150),
            Run(expert: 12, start: 150, end: 180),
            Run(expert: 20, start: 180, end: 210),
            Run(expert: 30, start: 210, end: 224),
        ]
        let mTotal = runs.last!.end
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for run in runs {
            for r in run.start ..< run.end { idxHost[r] = run.expert }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        // Distinct-ish input per row so a row-alignment bug shows up as a
        // cos mismatch rather than accidentally cancelling.
        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            let v: Float = 0.5 + Float(r % 7) * 0.1
            let bits = v.bitPattern
            let bf16 = UInt16(bits >> 16)
            for k in 0 ..< kIn { xHost[r * kIn + k] = bf16 }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outNax = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outNax.buffer.contents().advanced(by: outNax.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4MoeGatherNax64(
            input: x, weight: gw, scales: gs, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outNax)
        let outEidx = eidxOracle(
            x: x, weight: gw, scales: gs, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        let nax = toF32(outNax)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(nax, eidx)
        print("[nax64-gate] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        print("[nax64-gate] nax sample \(nax.prefix(4)) eidx sample \(eidx.prefix(4))")
        #expect(nan == 0, "NAX64 real gate produced \(nan) NaNs")
        #expect(cos > 0.9999, "NAX64 real gate boundary-straddle cos \(cos)")

        // Per-row check: also verify a row right at each straddle boundary
        // individually (row 63/64, 127/128, 191/192) to localize any tile-
        // seam bug rather than only seeing it diluted in the aggregate cos.
        for boundary in [64, 128, 192] {
            for r in [boundary - 1, boundary] {
                guard r < mTotal else { continue }
                let naxRow = Array(nax[(r * nOut) ..< ((r + 1) * nOut)])
                let eidxRow = Array(eidx[(r * nOut) ..< ((r + 1) * nOut)])
                let (rowCos, rowNan) = cosine(naxRow, eidxRow)
                print("[nax64-gate] row \(r) (expert \(idxHost[r])) cos=\(rowCos) nan=\(rowNan)")
                #expect(rowNan == 0, "row \(r) NaN")
                #expect(rowCos > 0.999, "row \(r) at tile seam cos \(rowCos)")
            }
        }
    }

    /// Real Laguna down proj (kIn=512, nOut=2048): different shape class
    /// (kIn < nOut) from gate/up.
    @Test("NAX64 real Laguna down proj", .enabled(if: true))
    func nax64RealDownProj() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let dw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.down_proj.weight")
        let ds = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.down_proj.scales")
        let nOut = dw.shape[1]
        let packs = dw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 64 == 0, "down kIn \(kIn) must be %64==0 for NAX64")
        try #require(nOut % 64 == 0)

        // 96 rows across 3 experts, uneven split (not a multiple of 64) so
        // the last tile is a partial 32-row tile.
        let mTotal = 96
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for i in 0 ..< mTotal { idxHost[i] = UInt32(i / 32) }  // experts 0,1,2 (32 rows each)
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            let v: Float = 0.3 + Float(r % 5) * 0.15
            let bf16 = UInt16(v.bitPattern >> 16)
            for k in 0 ..< kIn { xHost[r * kIn + k] = bf16 }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outNax = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outNax.buffer.contents().advanced(by: outNax.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4MoeGatherNax64(
            input: x, weight: dw, scales: ds, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outNax)
        let outEidx = eidxOracle(
            x: x, weight: dw, scales: ds, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let nax = toF32(outNax)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(nax, eidx)
        print("[nax64-down] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "NAX64 down proj produced \(nan) NaNs")
        #expect(cos > 0.9999, "NAX64 down proj cos \(cos)")
    }

    /// Full MoELayer-shaped integration: real `moeSortPlan` output (many
    /// zero-count experts among 256, since mTotal=80 « 256) feeding NAX64
    /// gate proj, same as the wiring in MoELayer's gather path.
    @Test("NAX64 after moeSortPlan+gather like MoELayer", .enabled(if: true))
    func nax64AfterSortPlanGather() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        let mid = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 64 == 0)
        let t = 10
        let topK = 8
        let mTotal = t * topK  // 80: crosses the 64-row tile boundary
        let nExperts = 256

        var hHost = [UInt16](repeating: 0, count: t * kIn)
        for r in 0 ..< t {
            let v: Float = 0.4 + Float(r) * 0.05
            let bf16 = UInt16(v.bitPattern >> 16)
            for k in 0 ..< kIn { hHost[r * kIn + k] = bf16 }
        }
        let hBuf = device.makeBuffer(length: hHost.count * 2)
        hHost.withUnsafeBytes { _ = memcpy(hBuf.contents(), $0.baseAddress!, hHost.count * 2) }
        let hRows = Tensor(buffer: hBuf, offset: 0, shape: [t, kIn], dtype: .bf16)

        var topkHost = [UInt32](repeating: 0, count: mTotal)
        var seed: UInt32 = 13
        for r in 0 ..< t {
            var used = Set<UInt32>()
            var s = 0
            while s < topK {
                seed = seed &* 1664525 &+ 1013904223
                let e = seed % UInt32(nExperts)
                if used.insert(e).inserted {
                    topkHost[r * topK + s] = e
                    s += 1
                }
            }
        }
        let topkBuf = device.makeBuffer(length: mTotal * 4)
        topkHost.withUnsafeBytes { _ = memcpy(topkBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let topkIds = Tensor(buffer: topkBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let se = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        let st = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        let ip = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)

        let cmd = device.makeCommandBuffer()
        Ops.moeSortPlan(
            topkIds: topkIds, sortedExperts: se, sourceTokens: st, invPerm: ip,
            mTotal: mTotal, k: topK, on: cmd)
        let xGathered = Ops.gather(table: hRows, tokenIds: st, on: cmd)

        let outNax = Tensor.empty(shape: [mTotal, mid], dtype: .bf16, device: device)
        memset(outNax.buffer.contents().advanced(by: outNax.offset), 0, mTotal * mid * 2)
        Ops.nvfp4MoeGatherNax64(
            input: xGathered, weight: gw, scales: gs, indices: se,
            mTotal: mTotal, nOut: mid, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outNax)

        let outEidx = eidxOracle(
            x: xGathered, weight: gw, scales: gs, idxBuf: se.buffer,
            mTotal: mTotal, kIn: kIn, nOut: mid, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let seHost = se.toArray(as: UInt32.self)
        print("[nax64-sort] sorted experts prefix \(seHost.prefix(16)) is sorted \(seHost == seHost.sorted())")
        let uniqueExperts = Set(seHost).count
        print("[nax64-sort] unique experts used = \(uniqueExperts)/\(nExperts) (rest zero-count)")

        let nax = toF32(outNax)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(nax, eidx)
        print("[nax64-sort] mTotal=\(mTotal) nan=\(nan) cos=\(cos)")
        #expect(nan == 0)
        #expect(cos > 0.9999, "NAX64 after sort+gather cos \(cos)")
    }

    /// RAW bf16 kernel (no f32 auto-promote): this is the exact path
    /// `nvfp4MoeGatherNax64Pair` takes unconditionally for bf16 input
    /// (MoELayer's default gate+up dispatch for early/native-bf16 layers).
    /// Every other test in this file passes bf16 input through the
    /// *singular* `nvfp4MoeGatherNax64`, which auto-promotes bf16 -> f32
    /// unless `preferNativeBf16: true`, so none of them actually exercised
    /// `butter_nvfp4_gather_qmm_rhs_nax64_mpp_bf16` itself until now.
    @Test("NAX64 RAW bf16 (preferNativeBf16) real Laguna gate", .enabled(if: true))
    func nax64RawBf16RealGate() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        let nOut = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 64 == 0)

        let mTotal = 64  // exactly one 64-row tile, matches real T=8,topK=8 prefill
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        var seed: UInt32 = 41
        for i in 0 ..< mTotal {
            seed = seed &* 1664525 &+ 1013904223
            idxHost[i] = seed % 256
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            let v: Float = 0.2 + Float(r % 11) * 0.07
            let bf16 = UInt16(v.bitPattern >> 16)
            for k in 0 ..< kIn { xHost[r * kIn + k] = bf16 }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outNax = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outNax.buffer.contents().advanced(by: outNax.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4MoeGatherNax64(
            input: x, weight: gw, scales: gs, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0,
            preferNativeBf16: true,  // <-- forces the raw bf16 kernel, no promote
            on: cmd, into: outNax)
        let outEidx = eidxOracle(
            x: x, weight: gw, scales: gs, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let nax = toF32(outNax)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(nax, eidx)
        print("[nax64-rawbf16] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        print("[nax64-rawbf16] nax sample \(nax.prefix(6))")
        print("[nax64-rawbf16] eidx sample \(eidx.prefix(6))")
        #expect(nan == 0, "raw bf16 NAX64 produced \(nan) NaNs")
        #expect(cos > 0.999, "raw bf16 NAX64 real gate cos \(cos)")
    }

    /// `nvfp4MoeGatherNax64Pair` directly: the concurrent gate+up dispatch
    /// MoELayer actually calls. No other test in this file exercises the
    /// Pair entry point at all (only the serial `nvfp4MoeGatherNax64`).
    @Test("NAX64 Pair (gate+up concurrent) real Laguna weights", .enabled(if: true))
    func nax64PairRealGateUp() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        let uw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.up_proj.weight")
        let us = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.up_proj.scales")
        let nOut = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 64 == 0)

        let mTotal = 64
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        var seed: UInt32 = 97
        for i in 0 ..< mTotal {
            seed = seed &* 1664525 &+ 1013904223
            idxHost[i] = seed % 256
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            let v: Float = 0.15 + Float(r % 9) * 0.08
            let bf16 = UInt16(v.bitPattern >> 16)
            for k in 0 ..< kIn { xHost[r * kIn + k] = bf16 }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outGateNax = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outUpNax = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outGateNax.buffer.contents().advanced(by: outGateNax.offset), 0, mTotal * nOut * 2)
        memset(outUpNax.buffer.contents().advanced(by: outUpNax.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        // This is bf16 input into the Pair entry point directly -> raw bf16
        // kernel, unconditionally (Pair has no promote path, matches
        // MoELayer's actual call site for native-bf16-MPP layers).
        Ops.nvfp4MoeGatherNax64Pair(
            input: x,
            weight0: gw, scales0: gs, out0: outGateNax,
            weight1: uw, scales1: us, out1: outUpNax,
            indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd)
        let outGateEidx = eidxOracle(
            x: x, weight: gw, scales: gs, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        let outUpEidx = eidxOracle(
            x: x, weight: uw, scales: us, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "cmd status \(cmd.status.rawValue)")

        let gateNax = toF32(outGateNax)
        let gateEidx = toF32(outGateEidx)
        let upNax = toF32(outUpNax)
        let upEidx = toF32(outUpEidx)
        let (gateCos, gateNan) = cosine(gateNax, gateEidx)
        let (upCos, upNan) = cosine(upNax, upEidx)
        print("[nax64-pair] gate nan=\(gateNan) cos=\(gateCos) sample nax=\(gateNax.prefix(4)) eidx=\(gateEidx.prefix(4))")
        print("[nax64-pair] up   nan=\(upNan) cos=\(upCos) sample nax=\(upNax.prefix(4)) eidx=\(upEidx.prefix(4))")
        #expect(gateNan == 0, "Pair gate produced \(gateNan) NaNs")
        #expect(upNan == 0, "Pair up produced \(upNan) NaNs")
        #expect(gateCos > 0.999, "Pair gate cos \(gateCos)")
        #expect(upCos > 0.999, "Pair up cos \(upCos)")
    }
}
