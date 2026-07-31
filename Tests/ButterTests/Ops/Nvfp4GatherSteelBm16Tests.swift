// Copyright 2026 Tom Turney (@TheTom)
//
// T35 kernel-level correctness: BM16 row-tile variant of the sorted-gather
// multi-expert steel-structure NVFP4 GEMM
// (`butter_nvfp4_gather_steel_bm16_bgemm_bf16`) vs the expert-indexed GEMV
// dequant oracle. Same test ladder as Nvfp4GatherSteelGatherTests.swift
// (the BM64 kernel's own suite) -- synthetic same-expert, real-weight
// boundary straddle + zero-count experts, ragged/partial tail tile, real
// moeSortPlan integration, magnitude ramp to 800 -- since this kernel
// targets the same production dispatch shape, just with a 16-row tile
// instead of 64. The boundary-straddle test checks every 16-row tile
// boundary (not just one) since BM16 has 4x as many tile boundaries per
// row range as BM64.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("NVFP4 steel-structure gather GEMM BM16 (T35)", .serialized)
struct Nvfp4GatherSteelBm16Tests {
    private var steelAvailable: Bool {
        MTLCreateSystemDefaultDevice() != nil
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

    @Test("steel BM16 gather same-expert synthetic matches eidx", .enabled(if: true))
    func gatherSyntheticSameExpertMatchesEidx() throws {
        try #require(steelAvailable)
        let device = Device.shared
        let mTotal = 64
        let nOut = 128
        let kIn = 256
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

        let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outSteel.buffer.contents().advanced(by: outSteel.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4GatherSteelBm16(
            input: x, weight: weight, scales: scales, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: blockSize, global: 1.0, on: cmd, into: outSteel)
        let outEidx = eidxOracle(
            x: x, weight: weight, scales: scales, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        let steel = toF32(outSteel)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(steel, eidx)
        print("[steel-gather-bm16-synth] nan=\(nan) cos=\(cos) sample steel=\(steel.prefix(4)) eidx=\(eidx.prefix(4))")
        #expect(nan == 0, "steel BM16 gather produced \(nan) NaNs")
        #expect(cos > 0.9999, "steel BM16 gather vs eidx cos \(cos)")
    }

    /// Real Laguna gate weights, runs straddling 16-row tile boundaries
    /// (BM16 has far more tile boundaries than BM64 over the same row
    /// range), plus zero-count experts interleaved.
    @Test("steel BM16 gather real Laguna gate: boundary straddle + zero-count experts", .enabled(if: true))
    func gatherRealGateBoundaryStraddle() throws {
        try #require(steelAvailable)
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("[steel-gather-bm16-gate] skip: no model at \(modelDir.path)")
            return
        }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        let nOut = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 16 == 0 && nOut % 64 == 0)

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

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            let v: Float = 0.5 + Float(r % 7) * 0.1
            let bf16 = UInt16(v.bitPattern >> 16)
            for k in 0 ..< kIn { xHost[r * kIn + k] = bf16 }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outSteel.buffer.contents().advanced(by: outSteel.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4GatherSteelBm16(
            input: x, weight: gw, scales: gs, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outSteel)
        let outEidx = eidxOracle(
            x: x, weight: gw, scales: gs, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        let steel = toF32(outSteel)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(steel, eidx)
        print("[steel-gather-bm16-gate] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "steel BM16 gather real gate produced \(nan) NaNs")
        #expect(cos > 0.9999, "steel BM16 gather real gate boundary-straddle cos \(cos)")

        // Every 16-row tile boundary in range, not just one -- BM16 has 4x
        // the boundaries BM64 does over the same row span.
        let boundaries = stride(from: 16, to: mTotal, by: 16)
        for boundary in boundaries {
            for r in [boundary - 1, boundary] {
                guard r < mTotal else { continue }
                let steelRow = Array(steel[(r * nOut) ..< ((r + 1) * nOut)])
                let eidxRow = Array(eidx[(r * nOut) ..< ((r + 1) * nOut)])
                let (rowCos, rowNan) = cosine(steelRow, eidxRow)
                print("[steel-gather-bm16-gate] row \(r) (expert \(idxHost[r])) cos=\(rowCos) nan=\(rowNan)")
                #expect(rowNan == 0, "row \(r) NaN")
                #expect(rowCos > 0.999, "row \(r) at tile seam cos \(rowCos)")
            }
        }
    }

    /// Real Laguna down proj (kIn=512, nOut=2048), ragged mTotal=96 (not a
    /// multiple of 16) across 3 experts -- exercises the partial-tail-tile
    /// bounds checking path directly.
    @Test("steel BM16 gather real Laguna down proj, ragged mTotal", .enabled(if: true))
    func gatherRealDownProjRagged() throws {
        try #require(steelAvailable)
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
        try #require(kIn % 16 == 0 && nOut % 64 == 0)

        let mTotal = 96
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for i in 0 ..< mTotal { idxHost[i] = UInt32(i / 32) }
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

        let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outSteel.buffer.contents().advanced(by: outSteel.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4GatherSteelBm16(
            input: x, weight: dw, scales: ds, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outSteel)
        let outEidx = eidxOracle(
            x: x, weight: dw, scales: ds, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let steel = toF32(outSteel)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(steel, eidx)
        print("[steel-gather-bm16-down] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "steel BM16 gather down proj produced \(nan) NaNs")
        #expect(cos > 0.9999, "steel BM16 gather down proj cos \(cos)")
    }

    /// Full MoELayer-shaped integration: real `moeSortPlan` output (many
    /// zero-count experts among 256, since mTotal=80 << 256).
    @Test("steel BM16 gather after moeSortPlan+gather like MoELayer", .enabled(if: true))
    func gatherAfterSortPlanGather() throws {
        try #require(steelAvailable)
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
        try #require(kIn % 16 == 0 && mid % 64 == 0)
        let t = 10
        let topK = 8
        let mTotal = t * topK  // 80: crosses several 16-row tile boundaries
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

        let outSteel = Tensor.empty(shape: [mTotal, mid], dtype: .bf16, device: device)
        memset(outSteel.buffer.contents().advanced(by: outSteel.offset), 0, mTotal * mid * 2)
        Ops.nvfp4GatherSteelBm16(
            input: xGathered, weight: gw, scales: gs, indices: se,
            mTotal: mTotal, nOut: mid, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outSteel)

        let outEidx = eidxOracle(
            x: xGathered, weight: gw, scales: gs, idxBuf: se.buffer,
            mTotal: mTotal, kIn: kIn, nOut: mid, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let seHost = se.toArray(as: UInt32.self)
        print("[steel-gather-bm16-sort] sorted experts prefix \(seHost.prefix(16)) is sorted \(seHost == seHost.sorted())")
        let uniqueExperts = Set(seHost).count
        print("[steel-gather-bm16-sort] unique experts used = \(uniqueExperts)/\(nExperts) (rest zero-count)")

        let steel = toF32(outSteel)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(steel, eidx)
        print("[steel-gather-bm16-sort] mTotal=\(mTotal) nan=\(nan) cos=\(cos)")
        #expect(nan == 0)
        #expect(cos > 0.9999, "steel BM16 gather after sort+gather cos \(cos)")
    }

    /// Magnitude ramp to 800 through the sorted-gather path, real layer-38
    /// expert-161 weights (the T22/T23 failing shape), multi-expert tile.
    @Test("steel BM16 gather magnitude ramp to 800, multi-expert tile", .enabled(if: true))
    func gatherMagnitudeRamp() throws {
        try #require(steelAvailable)
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("[steel-gather-bm16-ramp] skip: no model at \(modelDir.path)")
            return
        }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.38.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.38.mlp.switch_mlp.gate_proj.scales")
        let nOut = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 16 == 0 && nOut % 64 == 0)

        let mTotal = 64
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        // Two experts straddling the 32-row midpoint -- also straddles two
        // separate 16-row BM16 tiles (16-31 and 32-47 both mixed... here
        // actually the switch lands exactly on a 16-row tile boundary at
        // row 32, so this also covers the "clean tile-aligned switch" case;
        // the boundary-straddle test above covers the misaligned case).
        for i in 0 ..< mTotal { idxHost[i] = i < 32 ? 161 : 200 }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var worstCos: Float = 2
        var anyNan = 0
        for magnitude: Float in [1, 10, 50, 100, 200, 400, 600, 800] {
            var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
            for r in 0 ..< mTotal {
                for k in 0 ..< kIn {
                    let v = magnitude * sin(Float(k) * 0.7 + Float(r) * 0.13)
                    xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
                }
            }
            let xBuf = device.makeBuffer(length: xHost.count * 2)
            xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
            let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

            let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
            memset(outSteel.buffer.contents().advanced(by: outSteel.offset), 0, mTotal * nOut * 2)
            let cmd = device.makeCommandBuffer()
            Ops.nvfp4GatherSteelBm16(
                input: x, weight: gw, scales: gs, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: 16, global: 1.0, on: cmd, into: outSteel)
            let outEidx = eidxOracle(
                x: x, weight: gw, scales: gs, idxBuf: idxBuf,
                mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
            cmd.commit(); cmd.waitUntilCompleted()
            let steel = toF32(outSteel)
            let eidx = toF32(outEidx)
            let (cos, nan) = cosine(steel, eidx)
            print("[steel-gather-bm16-ramp] magnitude=\(magnitude) nan=\(nan) cos=\(cos)")
            worstCos = min(worstCos, cos)
            anyNan += nan
        }
        #expect(anyNan == 0, "steel BM16 gather magnitude ramp produced NaNs at some magnitude")
        #expect(worstCos > 0.999, "steel BM16 gather magnitude ramp worst cos \(worstCos)")
    }
}
