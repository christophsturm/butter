// Copyright 2026 Tom Turney (@TheTom)
//
// Diagnostic scan (T22): the full-model NAX64 prefill produces all-NaN
// logits (`--dump-prefill-logits`) even though every kernel-level cos test
// in Nvfp4GatherNax64Tests.swift passes cleanly on layer-1 weights. This
// scans NAX64 output for NaN/Inf across EVERY real Laguna MoE layer's
// gate/up/down weights (no oracle, just self-consistency) to localize
// which layer index (if any) triggers a NaN that layer-1-only tests missed.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("NVFP4 NAX64 all-layer NaN scan", .serialized)
struct Nvfp4GatherNax64ScanTests {
    private var mppAvailable: Bool {
        guard let dev = MTLCreateSystemDefaultDevice() else { return false }
        return dev.supportsFamily(.apple9)
    }

    @Test("NAX64 scan all 39 layers gate/up/down for NaN", .enabled(if: true))
    func scanAllLayers() throws {
        try #require(mppAvailable)
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("[nax64-scan] skip: no model")
            return
        }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)

        let mTotal = 64
        // Realistic-ish magnitude, varied per row/col so it isn't degenerate,
        // but still small enough that a *correct* kernel stays finite.
        func makeX(kIn: Int) -> Tensor {
            var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
            for r in 0 ..< mTotal {
                for k in 0 ..< kIn {
                    let v: Float = 0.3 * sin(Float(r * 37 + k) * 0.017) + 0.1
                    xHost[r * kIn + k] = UInt16(v.bitPattern >> 16)
                }
            }
            let xBuf = device.makeBuffer(length: xHost.count * 2)
            xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
            return Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)
        }
        // All 64 rows spread across 8 distinct experts (8 rows each),
        // matches T=8,topK=8 prefill shape without needing per-token routing.
        let nExperts = 256
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for i in 0 ..< mTotal { idxHost[i] = UInt32((i / 8) * 7 % nExperts) }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        func toF32(_ t: Tensor) -> [Float] {
            let n = t.elementCount
            let ptr = t.buffer.contents().advanced(by: t.offset)
                .bindMemory(to: UInt16.self, capacity: n)
            return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
        }

        var badLayers: [(Int, String)] = []
        for layer in 1 ... 39 {
            for (proj, wname, sname) in [
                ("gate", "gate_proj", "gate_proj"),
                ("up", "up_proj", "up_proj"),
                ("down", "down_proj", "down_proj"),
            ] {
                let wKey = "model.layers.\(layer).mlp.switch_mlp.\(wname).weight"
                let sKey = "model.layers.\(layer).mlp.switch_mlp.\(sname).scales"
                guard let w = try? weights.tensor(named: wKey),
                    let s = try? weights.tensor(named: sKey)
                else { continue }
                let nOut = w.shape[1]
                let packs = w.shape[2]
                let kIn = packs * 8
                guard kIn % 64 == 0, nOut % 64 == 0 else { continue }

                let x = makeX(kIn: kIn)
                let outNax = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
                memset(outNax.buffer.contents().advanced(by: outNax.offset), 0, mTotal * nOut * 2)
                let cmd = device.makeCommandBuffer()
                // preferNativeBf16: true forces the raw bf16 kernel unconditionally,
                // matching what MoELayer's Pair path does for native-bf16 layers.
                Ops.nvfp4MoeGatherNax64(
                    input: x, weight: w, scales: s, indices: indices,
                    mTotal: mTotal, nOut: nOut, kIn: kIn,
                    blockSize: 16, global: 1.0, preferNativeBf16: true,
                    on: cmd, into: outNax)
                cmd.commit(); cmd.waitUntilCompleted()

                let vals = toF32(outNax)
                let nanCount = vals.filter { $0.isNaN }.count
                let infCount = vals.filter { $0.isInfinite }.count
                if nanCount > 0 || infCount > 0 {
                    print("[nax64-scan] LAYER \(layer) \(proj) BAD: nan=\(nanCount) inf=\(infCount) / \(vals.count) kIn=\(kIn) nOut=\(nOut)")
                    badLayers.append((layer, proj))
                }
            }
        }
        print("[nax64-scan] done. bad=\(badLayers)")
        #expect(badLayers.isEmpty, "NAX64 produced non-finite output on layers/projs: \(badLayers)")
    }

    /// Every prior NAX64 test zeroes the output buffer before dispatch; if
    /// the kernel leaves ANY element unwritten, a zeroed buffer hides it
    /// (unwritten == 0.0, finite, doesn't fail a cos check badly). The real
    /// MoELayer prefill path REUSES scratch buffers across all 39 layers
    /// and explicitly SKIPS a defensive zero-fill blit whenever
    /// `mTotal % tileM == 0` ("full tiles overwrite every output element,
    /// skip 3 blit fills × 39 layers", tileM=64 for the NAX64/BM64 path),
    /// exactly the T=8,topK=8 -> mTotal=64 shape real prefill uses. Poison
    /// the buffer with NaN bit patterns first: any element the kernel
    /// doesn't actually write shows up directly as a leftover NaN.
    @Test("NAX64 writes every output element (poisoned-buffer check)", .enabled(if: true))
    func poisonedBufferFullCoverage() throws {
        try #require(mppAvailable)
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

        let mTotal = 64  // exact tile multiple -> real MoELayer skips the zero-fill blit
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        var seed: UInt32 = 5
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
        // Poison with bf16 NaN (0x7FC0 = quiet NaN top-16) instead of zero.
        let nanBits: UInt16 = 0x7FC0
        let poison = [UInt16](repeating: nanBits, count: mTotal * nOut)
        poison.withUnsafeBytes {
            _ = memcpy(outNax.buffer.contents().advanced(by: outNax.offset), $0.baseAddress!, mTotal * nOut * 2)
        }

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4MoeGatherNax64(
            input: x, weight: gw, scales: gs, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, preferNativeBf16: true,
            on: cmd, into: outNax)
        cmd.commit(); cmd.waitUntilCompleted()

        let n = outNax.elementCount
        let ptr = outNax.buffer.contents().advanced(by: outNax.offset)
            .bindMemory(to: UInt16.self, capacity: n)
        var unwritten: [Int] = []
        for i in 0 ..< n where ptr[i] == nanBits { unwritten.append(i) }
        print("[nax64-poison] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) unwritten=\(unwritten.count)/\(n)")
        if !unwritten.isEmpty {
            let sample = unwritten.prefix(20).map { (row: $0 / nOut, col: $0 % nOut) }
            print("[nax64-poison] unwritten sample (row,col): \(sample)")
            // Row histogram: is it whole rows, whole cols, or scattered?
            let rowsHit = Set(unwritten.map { $0 / nOut })
            let colsHit = Set(unwritten.map { $0 % nOut })
            print("[nax64-poison] distinct rows touched=\(rowsHit.count)/\(mTotal) cols touched=\(colsHit.count)/\(nOut)")
        }
        #expect(unwritten.isEmpty, "NAX64 left \(unwritten.count)/\(n) output elements unwritten (poison leak)")
    }
}
