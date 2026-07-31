// Copyright 2026 Tom Turney (@TheTom)
//
// T33 stage 1 kernel-level correctness: dense (single-expert) steel-
// structure NVFP4 GEMM (`butter_nvfp4_gather_steel_bgemm_dense_bf16`) vs
// the expert-indexed GEMV dequant oracle (`Ops.nvfp4DequantGemvExpertIndexedMany`),
// same oracle pattern as Nvfp4GatherNax64Tests.swift. Covers: synthetic
// uniform weights, real Laguna gate/up/down weight slices, and a magnitude
// ramp to 800 (mirrors Nvfp4GatherNax64MagnitudeTests.swift) to verify the
// bfloat-staged kernel needs no T23-style exponent-bias fold.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("NVFP4 steel-structure dense gather GEMM (T33 stage 1)", .serialized)
struct Nvfp4GatherSteelDenseTests {
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

    /// Oracle: one expert-indexed GEMV per row, single "expert" (index 0)
    /// shared by every row -- exactly what the dense steel kernel computes.
    private func denseOracle(
        x: Tensor, weight: Tensor, scales: Tensor,
        mTotal: Int, kIn: Int, nOut: Int, device: Device, cmd: MTLCommandBuffer
    ) -> Tensor {
        let outEidx = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outEidx.buffer.contents().advanced(by: outEidx.offset), 0, mTotal * nOut * 2)
        let weightStacked = weight.reshaped(to: [1, nOut, weight.shape[1]])
        let scalesStacked = scales.reshaped(to: [1, nOut, scales.shape[1]])
        let idxHost = [UInt32](repeating: 0, count: mTotal)
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        var ws: [Tensor] = [], ss: [Tensor] = [], ins: [Tensor] = []
        var idxs: [Tensor] = [], outs: [Tensor] = []
        for r in 0 ..< mTotal {
            ws.append(weightStacked); ss.append(scalesStacked)
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

    private func rampInput(device: Device, mTotal: Int, kIn: Int, magnitude: Float) -> Tensor {
        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            for k in 0 ..< kIn {
                let v = magnitude * sin(Float(k) * 0.7 + Float(r) * 0.13)
                xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        return Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)
    }

    @Test("dense steel synthetic uniform weight matches eidx", .enabled(if: true))
    func denseSyntheticMatchesEidx() throws {
        try #require(steelAvailable)
        let device = Device.shared
        let mTotal = 64
        let nOut = 128
        let kIn = 256
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        var wHost = [UInt32](repeating: 0, count: nOut * packs)
        for r in 0 ..< nOut {
            for p in 0 ..< packs {
                wHost[r * packs + p] = 0x3210_7654 &+ UInt32(r % 7)
            }
        }
        let sHost = [UInt8](repeating: 0x48, count: nOut * groups)
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        let sBuf = device.makeBuffer(length: sHost.count)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nOut, packs], dtype: .u32)
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nOut, groups], dtype: .u8)

        let x = rampInput(device: device, mTotal: mTotal, kIn: kIn, magnitude: 1.0)

        let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outSteel.buffer.contents().advanced(by: outSteel.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4GatherSteelDense(
            input: x, weight: weight, scales: scales,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: blockSize, global: 1.0, on: cmd, into: outSteel)
        let outEidx = denseOracle(
            x: x, weight: weight, scales: scales,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        let steel = toF32(outSteel)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(steel, eidx)
        print("[steel-dense-synth] nan=\(nan) cos=\(cos) sample steel=\(steel.prefix(4)) eidx=\(eidx.prefix(4))")
        #expect(nan == 0, "steel dense produced \(nan) NaNs")
        #expect(cos > 0.9999, "steel dense vs eidx cos \(cos)")
    }

    @Test("dense steel real Laguna gate proj slice", .enabled(if: true))
    func denseRealGateProj() throws {
        try #require(steelAvailable)
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("[steel-dense-gate] skip: no model at \(modelDir.path)")
            return
        }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gwStacked = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gsStacked = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        // Slice a single expert (index 3) out of the [nExperts, nOut, packs] stack.
        let expert = 3
        let nOut = gwStacked.shape[1]
        let packs = gwStacked.shape[2]
        let groups = gsStacked.shape[2]
        let kIn = packs * 8
        try #require(kIn % 16 == 0 && nOut % 64 == 0)
        let gw = gwStacked.slicedRows(start: expert, count: 1).reshaped(to: [nOut, packs])
        let gs = gsStacked.slicedRows(start: expert, count: 1).reshaped(to: [nOut, groups])

        let mTotal = 128
        let x = rampInput(device: device, mTotal: mTotal, kIn: kIn, magnitude: 2.5)

        let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outSteel.buffer.contents().advanced(by: outSteel.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4GatherSteelDense(
            input: x, weight: gw, scales: gs,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outSteel)
        let outEidx = denseOracle(
            x: x, weight: gw, scales: gs,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        let steel = toF32(outSteel)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(steel, eidx)
        print("[steel-dense-gate] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "steel dense real gate produced \(nan) NaNs")
        #expect(cos > 0.9999, "steel dense real gate cos \(cos)")
    }

    @Test("dense steel real Laguna down proj slice (kIn<nOut shape class)", .enabled(if: true))
    func denseRealDownProj() throws {
        try #require(steelAvailable)
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let dwStacked = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.down_proj.weight")
        let dsStacked = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.down_proj.scales")
        let expert = 5
        let nOut = dwStacked.shape[1]
        let packs = dwStacked.shape[2]
        let groups = dsStacked.shape[2]
        let kIn = packs * 8
        try #require(kIn % 16 == 0 && nOut % 64 == 0)
        let dw = dwStacked.slicedRows(start: expert, count: 1).reshaped(to: [nOut, packs])
        let ds = dsStacked.slicedRows(start: expert, count: 1).reshaped(to: [nOut, groups])

        let mTotal = 64
        let x = rampInput(device: device, mTotal: mTotal, kIn: kIn, magnitude: 1.5)

        let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outSteel.buffer.contents().advanced(by: outSteel.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4GatherSteelDense(
            input: x, weight: dw, scales: ds,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outSteel)
        let outEidx = denseOracle(
            x: x, weight: dw, scales: ds,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let steel = toF32(outSteel)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(steel, eidx)
        print("[steel-dense-down] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "steel dense down produced \(nan) NaNs")
        #expect(cos > 0.9999, "steel dense down cos \(cos)")
    }

    /// Magnitude ramp to 800 through real layer-38 expert-161 gate weights
    /// (the exact T22/T23 failing shape) -- verifies the bfloat-staged
    /// kernel does NOT need the half-staging 2^-14 fold; if this NaNs or
    /// diverges, that is a real finding to report, not to paper over.
    @Test("dense steel magnitude ramp to 800, real layer-38 expert-161 gate", .enabled(if: true))
    func denseMagnitudeRamp() throws {
        try #require(steelAvailable)
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("[steel-dense-ramp] skip: no model at \(modelDir.path)")
            return
        }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gwStacked = try weights.tensor(named: "model.layers.38.mlp.switch_mlp.gate_proj.weight")
        let gsStacked = try weights.tensor(named: "model.layers.38.mlp.switch_mlp.gate_proj.scales")
        let expert = 161
        let nOut = gwStacked.shape[1]
        let packs = gwStacked.shape[2]
        let groups = gsStacked.shape[2]
        let kIn = packs * 8
        try #require(kIn % 16 == 0 && nOut % 64 == 0)
        let gw = gwStacked.slicedRows(start: expert, count: 1).reshaped(to: [nOut, packs])
        let gs = gsStacked.slicedRows(start: expert, count: 1).reshaped(to: [nOut, groups])

        let mTotal = 64
        var worstCos: Float = 2
        var anyNan = 0
        for magnitude: Float in [1, 10, 50, 100, 200, 400, 600, 800] {
            let x = rampInput(device: device, mTotal: mTotal, kIn: kIn, magnitude: magnitude)
            let outSteel = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
            memset(outSteel.buffer.contents().advanced(by: outSteel.offset), 0, mTotal * nOut * 2)
            let cmd = device.makeCommandBuffer()
            Ops.nvfp4GatherSteelDense(
                input: x, weight: gw, scales: gs,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: 16, global: 1.0, on: cmd, into: outSteel)
            let outEidx = denseOracle(
                x: x, weight: gw, scales: gs,
                mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
            cmd.commit(); cmd.waitUntilCompleted()
            let steel = toF32(outSteel)
            let eidx = toF32(outEidx)
            let (cos, nan) = cosine(steel, eidx)
            print("[steel-dense-ramp] magnitude=\(magnitude) nan=\(nan) cos=\(cos)")
            worstCos = min(worstCos, cos)
            anyNan += nan
        }
        #expect(anyNan == 0, "steel dense magnitude ramp produced NaNs at some magnitude")
        #expect(worstCos > 0.999, "steel dense magnitude ramp worst cos \(worstCos)")
    }
}
