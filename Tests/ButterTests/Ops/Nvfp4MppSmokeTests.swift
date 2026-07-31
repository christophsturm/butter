// Copyright 2026 Tom Turney (@TheTom)
//
// Numerical smoke: NVFP4 MPP gather vs expert-indexed GEMV (same expert,
// m=16). Diagnoses all-NaN Laguna prefill MPP path.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("NVFP4 MPP smoke")
struct Nvfp4MppSmokeTests {
    /// Apple GPU Family 9+ required for MPP cooperative tensors.
    private var mppAvailable: Bool {
        guard let dev = MTLCreateSystemDefaultDevice() else { return false }
        return dev.supportsFamily(.apple9)
    }

    @Test("MPP BM16 same-expert matches eidx", .enabled(if: true))
    func mppBm16SameExpertMatchesEidx() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail

        let device = Device.shared
        let mTotal = 16
        let nOut = 512
        let kIn = 2048
        let nExperts = 4
        let blockSize = 16
        let global: Float = 1.0

        // Synthetic stacked weights: expert e has codes = e+1 (nonzero).
        let packs = kIn / 8
        let groups = kIn / blockSize
        let wElems = nExperts * nOut * packs
        let sElems = nExperts * nOut * groups
        var wHost = [UInt32](repeating: 0x1111_1111, count: wElems)
        var sHost = [UInt8](repeating: 0x48, count: sElems)  // moderate E4M3 scale
        // Vary by expert a bit.
        for e in 0 ..< nExperts {
            for r in 0 ..< nOut {
                for p in 0 ..< packs {
                    wHost[(e * nOut + r) * packs + p] = UInt32(0x1111_1111 &+ UInt32(e))
                }
            }
        }
        let wBuf = device.makeBuffer(length: wElems * 4)
        let sBuf = device.makeBuffer(length: sElems)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wElems * 4) }
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sElems) }
        let weight = Tensor(
            buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)
        let scales = Tensor(
            buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        // Input: all rows = ones.
        var xHost = [UInt16](repeating: 0x3F80, count: mTotal * kIn)  // bf16 1.0 = 0x3F80? 
        // bf16 1.0 is 0x3F80 yes (same top 16 of f32 0x3F800000).
        let xBuf = device.makeBuffer(length: mTotal * kIn * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, mTotal * kIn * 2) }
        let x = Tensor(
            buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        // All rows expert 0.
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(
            buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outEidx = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)

        // Zero outs so empty-kernel shows zeros not NaN from uninit.
        memset(outMpp.buffer.contents().advanced(by: outMpp.offset), 0, mTotal * nOut * 2)
        memset(outEidx.buffer.contents().advanced(by: outEidx.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        // Force BM16.
        setenv("BUTTER_NVFP4_GATHER_BM", "16", 1)
        Ops.nvfp4MoeGatherBm16(
            input: x, weight: weight, scales: scales, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: blockSize, global: global,
            on: cmd, into: outMpp)

        // Expert-indexed: one GEMV per row, expert 0.
        var ws: [Tensor] = [], ss: [Tensor] = [], ins: [Tensor] = []
        var idxs: [Tensor] = [], outs: [Tensor] = []
        for r in 0 ..< mTotal {
            let xRow = x.slicedRows(start: r, count: 1).reshaped(to: [kIn])
            let oRow = outEidx.slicedRows(start: r, count: 1).reshaped(to: [nOut])
            let eIdx = Tensor(
                buffer: idxBuf, offset: 0, shape: [1], dtype: .u32)
            ws.append(weight); ss.append(scales)
            ins.append(xRow); idxs.append(eIdx); outs.append(oRow)
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss,
            inputs: ins, expertIndices: idxs, outputs: outs,
            blockSize: blockSize, global: global, on: cmd)

        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "cmd status \(cmd.status.rawValue)")

        // Read back as bf16 → f32.
        func toF32(_ t: Tensor) -> [Float] {
            let n = t.elementCount
            let ptr = t.buffer.contents().advanced(by: t.offset)
                .bindMemory(to: UInt16.self, capacity: n)
            var out = [Float](repeating: 0, count: n)
            for i in 0 ..< n {
                let bits = UInt32(ptr[i]) << 16
                out[i] = Float(bitPattern: bits)
            }
            return out
        }
        let mpp = toF32(outMpp)
        let eidx = toF32(outEidx)

        let mppFinite = mpp.filter { $0.isFinite }.count
        let eidxFinite = eidx.filter { $0.isFinite }.count
        let mppNan = mpp.filter { $0.isNaN }.count
        print("[mpp-smoke] mpp finite=\(mppFinite)/\(mpp.count) nan=\(mppNan) max=\(mpp.filter(\.isFinite).max() ?? .nan)")
        print("[mpp-smoke] eidx finite=\(eidxFinite)/\(eidx.count) max=\(eidx.filter(\.isFinite).max() ?? .nan)")
        print("[mpp-smoke] mpp sample \(mpp.prefix(8))")
        print("[mpp-smoke] eidx sample \(eidx.prefix(8))")

        #expect(eidxFinite == eidx.count, "eidx should be all finite")
        #expect(mppNan == 0, "MPP produced \(mppNan) NaNs")
        #expect(mppFinite == mpp.count, "MPP finite \(mppFinite)/\(mpp.count)")

        // Cosine similarity on finite pairs.
        var dot: Float = 0, nm: Float = 0, ne: Float = 0
        for i in 0 ..< mpp.count where mpp[i].isFinite && eidx[i].isFinite {
            dot += mpp[i] * eidx[i]
            nm += mpp[i] * mpp[i]
            ne += eidx[i] * eidx[i]
        }
        let cos = dot / (sqrt(nm) * sqrt(ne) + 1e-12)
        print("[mpp-smoke] cos=\(cos)")
        #expect(cos > 0.99, "MPP vs eidx cos \(cos)")
    }

    @Test("MPP BM16 multi-expert sorted runs", .enabled(if: true))
    func mppBm16MultiExpertSorted() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let device = Device.shared
        // 4 experts × 8 rows each = 32, sorted by expert.
        let rowsPerExpert = 8
        let nExperts = 4
        let mTotal = rowsPerExpert * nExperts
        let nOut = 512
        let kIn = 2048
        let blockSize = 16
        let global: Float = 1.0
        let packs = kIn / 8
        let groups = kIn / blockSize

        var wHost = [UInt32](repeating: 0x1111_1111, count: nExperts * nOut * packs)
        var sHost = [UInt8](repeating: 0x48, count: nExperts * nOut * groups)
        for e in 0 ..< nExperts {
            for r in 0 ..< nOut {
                for p in 0 ..< min(4, packs) {
                    // Distinct codes per expert so rows differ.
                    wHost[(e * nOut + r) * packs + p] = UInt32(0x1111_1111 &+ UInt32(e * 0x1111))
                }
            }
        }
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        let sBuf = device.makeBuffer(length: sHost.count)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let weight = Tensor(
            buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)
        let scales = Tensor(
            buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        var xHost = [UInt16](repeating: 0x3F80, count: mTotal * kIn)
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for e in 0 ..< nExperts {
            for r in 0 ..< rowsPerExpert {
                idxHost[e * rowsPerExpert + r] = UInt32(e)
            }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outEidx = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outMpp.buffer.contents().advanced(by: outMpp.offset), 0, mTotal * nOut * 2)
        memset(outEidx.buffer.contents().advanced(by: outEidx.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        setenv("BUTTER_NVFP4_GATHER_BM", "16", 1)
        Ops.nvfp4MoeGatherBm16(
            input: x, weight: weight, scales: scales, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: blockSize, global: global,
            on: cmd, into: outMpp)

        var ws: [Tensor] = [], ss: [Tensor] = [], ins: [Tensor] = []
        var idxs: [Tensor] = [], outs: [Tensor] = []
        for r in 0 ..< mTotal {
            let xRow = x.slicedRows(start: r, count: 1).reshaped(to: [kIn])
            let oRow = outEidx.slicedRows(start: r, count: 1).reshaped(to: [nOut])
            let eIdx = Tensor(
                buffer: idxBuf, offset: r * 4, shape: [1], dtype: .u32)
            ws.append(weight); ss.append(scales)
            ins.append(xRow); idxs.append(eIdx); outs.append(oRow)
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss,
            inputs: ins, expertIndices: idxs, outputs: outs,
            blockSize: blockSize, global: global, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        func toF32(_ t: Tensor) -> [Float] {
            let n = t.elementCount
            let ptr = t.buffer.contents().advanced(by: t.offset)
                .bindMemory(to: UInt16.self, capacity: n)
            return (0 ..< n).map { i in
                Float(bitPattern: UInt32(ptr[i]) << 16)
            }
        }
        let mpp = toF32(outMpp)
        let eidx = toF32(outEidx)
        let mppNan = mpp.filter(\.isNaN).count
        print("[mpp-multi] mpp nan=\(mppNan) eidx nan=\(eidx.filter(\.isNaN).count)")
        print("[mpp-multi] mpp row0 \(mpp.prefix(4)) row8 \(mpp[nOut..<(nOut+4)])")
        print("[mpp-multi] eidx row0 \(eidx.prefix(4)) row8 \(eidx[nOut..<(nOut+4)])")
        #expect(mppNan == 0)
        var dot: Float = 0, nm: Float = 0, ne: Float = 0
        for i in 0 ..< mpp.count where mpp[i].isFinite && eidx[i].isFinite {
            dot += mpp[i] * eidx[i]; nm += mpp[i] * mpp[i]; ne += eidx[i] * eidx[i]
        }
        let cos = dot / (sqrt(nm) * sqrt(ne) + 1e-12)
        print("[mpp-multi] cos=\(cos)")
        #expect(cos > 0.99)
    }

    @Test("MPP with real Laguna gate weight slice", .enabled(if: true))
    func mppRealLagunaGate() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("[mpp-real] skip — no model at \(modelDir.path)")
            return
        }
        // Load via SafeTensors if available.
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        print("[mpp-real] gate weight \(gw.shape) \(gw.dtype) scales \(gs.shape) \(gs.dtype)")
        #expect(gw.shape.count == 3)
        let nExperts = gw.shape[0]
        let nOut = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        let mTotal = 32
        // Random-ish bf16 input (ones).
        var xHost = [UInt16](repeating: 0x3F80, count: mTotal * kIn)
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        // Sorted: 8 rows each of experts 0,1,2,3
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for i in 0 ..< mTotal { idxHost[i] = UInt32(i / 8) }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outEidx = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outMpp.buffer.contents().advanced(by: outMpp.offset), 0, mTotal * nOut * 2)
        memset(outEidx.buffer.contents().advanced(by: outEidx.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        setenv("BUTTER_NVFP4_GATHER_BM", "16", 1)
        Ops.nvfp4MoeGatherBm16(
            input: x, weight: gw, scales: gs, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0,
            on: cmd, into: outMpp)
        var ws: [Tensor] = [], ss: [Tensor] = [], ins: [Tensor] = []
        var idxs: [Tensor] = [], outs: [Tensor] = []
        for r in 0 ..< mTotal {
            ws.append(gw); ss.append(gs)
            ins.append(x.slicedRows(start: r, count: 1).reshaped(to: [kIn]))
            idxs.append(Tensor(buffer: idxBuf, offset: r * 4, shape: [1], dtype: .u32))
            outs.append(outEidx.slicedRows(start: r, count: 1).reshaped(to: [nOut]))
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss,
            inputs: ins, expertIndices: idxs, outputs: outs,
            blockSize: 16, global: 1.0, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        func toF32(_ t: Tensor) -> [Float] {
            let n = t.elementCount
            let ptr = t.buffer.contents().advanced(by: t.offset)
                .bindMemory(to: UInt16.self, capacity: n)
            return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
        }
        let mpp = toF32(outMpp)
        let eidx = toF32(outEidx)
        let mppNan = mpp.filter(\.isNaN).count
        print("[mpp-real] nExperts=\(nExperts) nOut=\(nOut) kIn=\(kIn)")
        print("[mpp-real] mpp nan=\(mppNan)/\(mpp.count) eidx nan=\(eidx.filter(\.isNaN).count)")
        print("[mpp-real] mpp sample \(mpp.prefix(8))")
        print("[mpp-real] eidx sample \(eidx.prefix(8))")
        #expect(mppNan == 0, "real Laguna gate MPP NaN count \(mppNan)")
        var dot: Float = 0, nm: Float = 0, ne: Float = 0
        for i in 0 ..< mpp.count where mpp[i].isFinite && eidx[i].isFinite {
            dot += mpp[i] * eidx[i]; nm += mpp[i] * mpp[i]; ne += eidx[i] * eidx[i]
        }
        let cos = dot / (sqrt(nm) * sqrt(ne) + 1e-12)
        print("[mpp-real] cos=\(cos)")
        #expect(cos > 0.99, "real gate cos \(cos)")
    }

    @Test("MPP BM64 real Laguna gate", .enabled(if: true))
    func mppBm64RealLagunaGate() throws {
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
        let mTotal = 64  // triggers BM64 auto + height=1
        var xHost = [UInt16](repeating: 0x3F80, count: mTotal * kIn)
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for i in 0 ..< mTotal { idxHost[i] = UInt32(i / 8) }  // 8 rows × 8 experts
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outEidx = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outMpp.buffer.contents().advanced(by: outMpp.offset), 0, mTotal * nOut * 2)
        memset(outEidx.buffer.contents().advanced(by: outEidx.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        // Preferred opt-in; GATHER_BM=64 still works as legacy alias.
        setenv("BUTTER_NVFP4_MOE_BM64", "1", 1)
        setenv("BUTTER_NVFP4_GATHER_BM", "64", 1)
        Ops.nvfp4MoeGatherBm16(
            input: x, weight: gw, scales: gs, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0,
            on: cmd, into: outMpp)
        var ws: [Tensor] = [], ss: [Tensor] = [], ins: [Tensor] = []
        var idxs: [Tensor] = [], outs: [Tensor] = []
        for r in 0 ..< mTotal {
            ws.append(gw); ss.append(gs)
            ins.append(x.slicedRows(start: r, count: 1).reshaped(to: [kIn]))
            idxs.append(Tensor(buffer: idxBuf, offset: r * 4, shape: [1], dtype: .u32))
            outs.append(outEidx.slicedRows(start: r, count: 1).reshaped(to: [nOut]))
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss,
            inputs: ins, expertIndices: idxs, outputs: outs,
            blockSize: 16, global: 1.0, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        func toF32(_ t: Tensor) -> [Float] {
            let n = t.elementCount
            let ptr = t.buffer.contents().advanced(by: t.offset)
                .bindMemory(to: UInt16.self, capacity: n)
            return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
        }
        let mpp = toF32(outMpp)
        let eidx = toF32(outEidx)
        let mppNan = mpp.filter(\.isNaN).count
        print("[mpp-bm64] nan=\(mppNan)/\(mpp.count) sample \(mpp.prefix(8))")
        print("[mpp-bm64] eidx sample \(eidx.prefix(8))")
        #expect(mppNan == 0, "BM64 NaN count \(mppNan)")
        var dot: Float = 0, nm: Float = 0, ne: Float = 0
        for i in 0 ..< mpp.count where mpp[i].isFinite && eidx[i].isFinite {
            dot += mpp[i] * eidx[i]; nm += mpp[i] * mpp[i]; ne += eidx[i] * eidx[i]
        }
        let cos = dot / (sqrt(nm) * sqrt(ne) + 1e-12)
        print("[mpp-bm64] cos=\(cos)")
        #expect(cos > 0.99, "BM64 cos \(cos)")
    }
}

// MARK: - Down proj (kIn=512, nOut=2048)
extension Nvfp4MppSmokeTests {
    @Test("MPP real Laguna down proj BM16", .enabled(if: true))
    func mppRealLagunaDown() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let dw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.down_proj.weight")
        let ds = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.down_proj.scales")
        print("[mpp-down] weight \(dw.shape) scales \(ds.shape)")
        let nOut = dw.shape[1]
        let packs = dw.shape[2]
        let kIn = packs * 8
        let mTotal = 32
        var xHost = [UInt16](repeating: 0x3F80, count: mTotal * kIn)
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for i in 0 ..< mTotal { idxHost[i] = UInt32(i / 8) }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)
        let outMpp = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        let outEidx = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outMpp.buffer.contents().advanced(by: outMpp.offset), 0, mTotal * nOut * 2)
        memset(outEidx.buffer.contents().advanced(by: outEidx.offset), 0, mTotal * nOut * 2)
        let cmd = device.makeCommandBuffer()
        setenv("BUTTER_NVFP4_GATHER_BM", "16", 1)
        Ops.nvfp4MoeGatherBm16(
            input: x, weight: dw, scales: ds, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outMpp)
        var ws: [Tensor] = [], ss: [Tensor] = [], ins: [Tensor] = []
        var idxs: [Tensor] = [], outs: [Tensor] = []
        for r in 0 ..< mTotal {
            ws.append(dw); ss.append(ds)
            ins.append(x.slicedRows(start: r, count: 1).reshaped(to: [kIn]))
            idxs.append(Tensor(buffer: idxBuf, offset: r * 4, shape: [1], dtype: .u32))
            outs.append(outEidx.slicedRows(start: r, count: 1).reshaped(to: [nOut]))
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss,
            inputs: ins, expertIndices: idxs, outputs: outs,
            blockSize: 16, global: 1.0, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        func toF32(_ t: Tensor) -> [Float] {
            let n = t.elementCount
            let ptr = t.buffer.contents().advanced(by: t.offset).bindMemory(to: UInt16.self, capacity: n)
            return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
        }
        let mpp = toF32(outMpp); let eidx = toF32(outEidx)
        let mppNan = mpp.filter(\.isNaN).count
        print("[mpp-down] nan=\(mppNan) sample \(mpp.prefix(8)) eidx \(eidx.prefix(8))")
        #expect(mppNan == 0)
        var dot: Float = 0, nm: Float = 0, ne: Float = 0
        for i in 0 ..< mpp.count where mpp[i].isFinite && eidx[i].isFinite {
            dot += mpp[i]*eidx[i]; nm += mpp[i]*mpp[i]; ne += eidx[i]*eidx[i]
        }
        let cos = dot / (sqrt(nm)*sqrt(ne)+1e-12)
        print("[mpp-down] cos=\(cos)")
        #expect(cos > 0.99)
    }
}

extension Nvfp4MppSmokeTests {
    @Test("MPP full SwiGLU gate+up+down real weights", .enabled(if: true))
    func mppFullSwigluReal() throws {
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
        let dw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.down_proj.weight")
        let ds = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.down_proj.scales")
        let mid = gw.shape[1]
        let hidden = dw.shape[1]
        let kIn = gw.shape[2] * 8
        let mTotal = 64
        // Pseudo-random bf16 inputs (top bits of f32 normals via bit cast table).
        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        var seed: UInt32 = 42
        for i in 0 ..< xHost.count {
            seed = seed &* 1664525 &+ 1013904223
            // Map to bf16 in ~[-2,2]
            let f = Float(Int(seed % 1000)) / 250.0 - 2.0
            xHost[i] = UInt16((f.bitPattern >> 16) & 0xFFFF)
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for i in 0 ..< mTotal { idxHost[i] = UInt32((i / 4) % 16) } // sorted runs of 4
        idxHost.sort()
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let gateOut = Tensor.empty(shape: [mTotal, mid], dtype: .bf16, device: device)
        let upOut = Tensor.empty(shape: [mTotal, mid], dtype: .bf16, device: device)
        let downOut = Tensor.empty(shape: [mTotal, hidden], dtype: .bf16, device: device)

        let cmd = device.makeCommandBuffer()
        setenv("BUTTER_NVFP4_GATHER_BM", "16", 1)
        Ops.nvfp4MoeGatherBm16(
            input: x, weight: gw, scales: gs, indices: indices,
            mTotal: mTotal, nOut: mid, kIn: kIn, blockSize: 16, global: 1.0,
            on: cmd, into: gateOut)
        Ops.nvfp4MoeGatherBm16(
            input: x, weight: uw, scales: us, indices: indices,
            mTotal: mTotal, nOut: mid, kIn: kIn, blockSize: 16, global: 1.0,
            on: cmd, into: upOut)
        let inner = Ops.swiglu(gate: gateOut, up: upOut, on: cmd, into: gateOut)
        Ops.nvfp4MoeGatherBm16(
            input: inner, weight: dw, scales: ds, indices: indices,
            mTotal: mTotal, nOut: hidden, kIn: mid, blockSize: 16, global: 1.0,
            on: cmd, into: downOut)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        func toF32(_ t: Tensor) -> [Float] {
            let n = t.elementCount
            let ptr = t.buffer.contents().advanced(by: t.offset).bindMemory(to: UInt16.self, capacity: n)
            return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
        }
        for (name, t) in [("gate", gateOut), ("down", downOut)] {
            let a = toF32(t)
            let nan = a.filter(\.isNaN).count
            print("[mpp-swiglu] \(name) nan=\(nan)/\(a.count) max=\(a.filter(\.isFinite).map(abs).max() ?? -1)")
            #expect(nan == 0, "\(name) has \(nan) NaNs")
        }
    }
}

extension Nvfp4MppSmokeTests {
    @Test("MPP after moeSortPlan+gather like MoELayer", .enabled(if: true))
    func mppAfterSortPlanGather() throws {
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
        let t = 6
        let topK = 8
        let mTotal = t * topK
        let nExperts = 256

        // Fake hRows: ones
        var hHost = [UInt16](repeating: 0x3F80, count: t * kIn)
        let hBuf = device.makeBuffer(length: hHost.count * 2)
        hHost.withUnsafeBytes { _ = memcpy(hBuf.contents(), $0.baseAddress!, hHost.count * 2) }
        let hRows = Tensor(buffer: hBuf, offset: 0, shape: [t, kIn], dtype: .bf16)

        // Random topk indices per token (not sorted)
        var topkHost = [UInt32](repeating: 0, count: mTotal)
        var seed: UInt32 = 7
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
        let outMpp = Tensor.empty(shape: [mTotal, mid], dtype: .bf16, device: device)
        memset(outMpp.buffer.contents().advanced(by: outMpp.offset), 0, mTotal * mid * 2)
        setenv("BUTTER_NVFP4_GATHER_BM", "16", 1)
        Ops.nvfp4MoeGatherBm16(
            input: xGathered, weight: gw, scales: gs, indices: se,
            mTotal: mTotal, nOut: mid, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, into: outMpp)

        // Reference: eidx on same gathered x + sorted experts
        let outEidx = Tensor.empty(shape: [mTotal, mid], dtype: .bf16, device: device)
        memset(outEidx.buffer.contents().advanced(by: outEidx.offset), 0, mTotal * mid * 2)
        var ws: [Tensor] = [], ss: [Tensor] = [], ins: [Tensor] = []
        var idxs: [Tensor] = [], outs: [Tensor] = []
        for r in 0 ..< mTotal {
            ws.append(gw); ss.append(gs)
            ins.append(xGathered.slicedRows(start: r, count: 1).reshaped(to: [kIn]))
            idxs.append(Tensor(buffer: se.buffer, offset: se.offset + r * 4, shape: [1], dtype: .u32))
            outs.append(outEidx.slicedRows(start: r, count: 1).reshaped(to: [mid]))
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss,
            inputs: ins, expertIndices: idxs, outputs: outs,
            blockSize: 16, global: 1.0, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        // Host check sorted experts
        let seHost = se.toArray(as: UInt32.self)
        print("[mpp-sort] sorted experts prefix \(seHost.prefix(16))")
        print("[mpp-sort] is sorted", seHost == seHost.sorted())
        for e in seHost { #expect(e < 256) }

        func toF32(_ t: Tensor) -> [Float] {
            let n = t.elementCount
            let ptr = t.buffer.contents().advanced(by: t.offset).bindMemory(to: UInt16.self, capacity: n)
            return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
        }
        let mpp = toF32(outMpp); let eidx = toF32(outEidx)
        let mppNan = mpp.filter(\.isNaN).count
        print("[mpp-sort] mpp nan=\(mppNan) eidx nan=\(eidx.filter(\.isNaN).count)")
        print("[mpp-sort] mpp \(mpp.prefix(4)) eidx \(eidx.prefix(4))")
        #expect(mppNan == 0)
        var dot: Float = 0, nm: Float = 0, ne: Float = 0
        for i in 0 ..< mpp.count where mpp[i].isFinite && eidx[i].isFinite {
            dot += mpp[i]*eidx[i]; nm += mpp[i]*mpp[i]; ne += eidx[i]*eidx[i]
        }
        let cos = dot / (sqrt(nm)*sqrt(ne)+1e-12)
        print("[mpp-sort] cos=\(cos)")
        #expect(cos > 0.99)
    }
}
