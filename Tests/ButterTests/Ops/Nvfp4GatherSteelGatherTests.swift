// Copyright 2026 Tom Turney (@TheTom)
//
// T33 stage 2 kernel-level correctness: sorted-gather multi-expert steel-
// structure NVFP4 GEMM (`butter_nvfp4_gather_steel_bgemm_bf16`) vs the
// expert-indexed GEMV dequant oracle. Same test ladder as
// Nvfp4GatherNax64Tests.swift (tile-boundary straddle, zero-count experts,
// ragged/partial tail tiles, real moeSortPlan integration) since this
// kernel targets the exact same production dispatch shape.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("NVFP4 steel-structure gather GEMM (T33 stage 2)", .serialized)
struct Nvfp4GatherSteelGatherTests {
    // Steel gather dispatches a coop-tensor `bgemm` PSO that a paravirt CI
    // GPU (Family < 9) cannot live-compile - it fatal-errors PSOCache and
    // takes the whole test bundle down with SIGTRAP. Gate on the same
    // auto-capability check the production path uses (honors
    // BUTTER_FORCE_NO_MPP) so incapable runners skip cleanly.
    private var steelAvailable: Bool {
        Ops.mppAutoCapable
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

    @Test("steel gather same-expert synthetic matches eidx", .enabled(if: true))
    func gatherSyntheticSameExpertMatchesEidx() throws {
        guard steelAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
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
        Ops.nvfp4GatherSteel(
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
        print("[steel-gather-synth] nan=\(nan) cos=\(cos) sample steel=\(steel.prefix(4)) eidx=\(eidx.prefix(4))")
        #expect(nan == 0, "steel gather produced \(nan) NaNs")
        #expect(cos > 0.9999, "steel gather vs eidx cos \(cos)")
    }

    /// Real Laguna gate weights, runs straddling the 64-row tile boundary,
    /// plus zero-count experts interleaved -- mirrors
    /// Nvfp4GatherNax64Tests.nax64RealGateBoundaryStraddle exactly.
    @Test("steel gather real Laguna gate: boundary straddle + zero-count experts", .enabled(if: true))
    func gatherRealGateBoundaryStraddle() throws {
        guard steelAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("[steel-gather-gate] skip: no model at \(modelDir.path)")
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
        Ops.nvfp4GatherSteel(
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
        print("[steel-gather-gate] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "steel gather real gate produced \(nan) NaNs")
        #expect(cos > 0.9999, "steel gather real gate boundary-straddle cos \(cos)")

        for boundary in [64] {
            for r in [boundary - 1, boundary] {
                guard r < mTotal else { continue }
                let steelRow = Array(steel[(r * nOut) ..< ((r + 1) * nOut)])
                let eidxRow = Array(eidx[(r * nOut) ..< ((r + 1) * nOut)])
                let (rowCos, rowNan) = cosine(steelRow, eidxRow)
                print("[steel-gather-gate] row \(r) (expert \(idxHost[r])) cos=\(rowCos) nan=\(rowNan)")
                #expect(rowNan == 0, "row \(r) NaN")
                #expect(rowCos > 0.999, "row \(r) at tile seam cos \(rowCos)")
            }
        }
    }

    /// Real Laguna down proj (kIn=512, nOut=2048), ragged mTotal=96 (not a
    /// multiple of 64) across 3 experts -- exercises the partial-tail-tile
    /// bounds checking path directly (mirrors nax64RealDownProj).
    @Test("steel gather real Laguna down proj, ragged mTotal", .enabled(if: true))
    func gatherRealDownProjRagged() throws {
        guard steelAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
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
        Ops.nvfp4GatherSteel(
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
        print("[steel-gather-down] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "steel gather down proj produced \(nan) NaNs")
        #expect(cos > 0.9999, "steel gather down proj cos \(cos)")
    }

    /// Full MoELayer-shaped integration: real `moeSortPlan` output (many
    /// zero-count experts among 256, since mTotal=80 << 256), mirrors
    /// nax64AfterSortPlanGather.
    @Test("steel gather after moeSortPlan+gather like MoELayer", .enabled(if: true))
    func gatherAfterSortPlanGather() throws {
        guard steelAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
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

        let outSteel = Tensor.empty(shape: [mTotal, mid], dtype: .bf16, device: device)
        memset(outSteel.buffer.contents().advanced(by: outSteel.offset), 0, mTotal * mid * 2)
        Ops.nvfp4GatherSteel(
            input: xGathered, weight: gw, scales: gs, indices: se,
            mTotal: mTotal, nOut: mid, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outSteel)

        let outEidx = eidxOracle(
            x: xGathered, weight: gw, scales: gs, idxBuf: se.buffer,
            mTotal: mTotal, kIn: kIn, nOut: mid, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let seHost = se.toArray(as: UInt32.self)
        print("[steel-gather-sort] sorted experts prefix \(seHost.prefix(16)) is sorted \(seHost == seHost.sorted())")
        let uniqueExperts = Set(seHost).count
        print("[steel-gather-sort] unique experts used = \(uniqueExperts)/\(nExperts) (rest zero-count)")

        let steel = toF32(outSteel)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(steel, eidx)
        print("[steel-gather-sort] mTotal=\(mTotal) nan=\(nan) cos=\(cos)")
        #expect(nan == 0)
        #expect(cos > 0.9999, "steel gather after sort+gather cos \(cos)")
    }

    /// Magnitude ramp to 800 through the sorted-gather path, real layer-38
    /// expert-161 weights (the T22/T23 failing shape), multi-expert tile.
    @Test("steel gather magnitude ramp to 800, multi-expert tile", .enabled(if: true))
    func gatherMagnitudeRamp() throws {
        guard steelAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("[steel-gather-ramp] skip: no model at \(modelDir.path)")
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
        // Two experts straddling the 32-row midpoint of the tile.
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
            Ops.nvfp4GatherSteel(
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
            print("[steel-gather-ramp] magnitude=\(magnitude) nan=\(nan) cos=\(cos)")
            worstCos = min(worstCos, cos)
            anyNan += nan
        }
        #expect(anyNan == 0, "steel gather magnitude ramp produced NaNs at some magnitude")
        #expect(worstCos > 0.999, "steel gather magnitude ramp worst cos \(worstCos)")
    }
}
