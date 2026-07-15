// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Dense int2 QMM (dequantGemmDynamicM) vs expert-grid gather for Hy3-ish
// GU shape. Answers: is the gather/EG path leaving multi-x on the table
// vs the rectangular int2 MMA path?
//
//   swift test -c release --filter MoEInt2DenseQmmVsEg

import Foundation
import Metal
import MetalTileSwift
import Testing

@testable import FFAI

@Suite("MoE int2 dense QMM vs EG")
struct MoEInt2DenseQmmVsEgTests {
    static let mppAvailable: Bool = {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple9)
    }()

    static func pack16(_ codes: [UInt32]) -> UInt32 {
        var word: UInt32 = 0
        for i in 0 ..< 16 { word |= (codes[i] & 0x3) << UInt32(i * 2) }
        return word
    }

    static func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let cmd = Device.shared.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)
    }

    @Test("microbench EG vs per-expert dense int2 QMM vs one fat dense QMM")
    func microbenchPaths() throws {
        try #require(Self.mppAvailable)
        // Hy3 GU: mTotal=2048, N=1536, K=4096, ~90 active of 192.
        let mTotal = 2048
        let nOut = 1536
        let kIn = 4096
        let nExperts = 192
        let nActive = 90
        let groupSize = 64
        let te = mTotal / nActive  // 22
        let tePad = ((te + 31) / 32) * 32  // 32
        let packs = kIn / 16
        let groups = kIn / groupSize
        let iters = 6

        // Sorted indices: nActive experts, te rows each (tail on last).
        var idx = [UInt32](repeating: 0, count: mTotal)
        for a in 0 ..< nActive {
            let e = UInt32(a * nExperts / nActive)
            let lo = a * te
            let hi = min((a + 1) * te, mTotal)
            for r in lo ..< hi { idx[r] = e }
        }
        let lastE = UInt32((nActive - 1) * nExperts / nActive)
        for r in (nActive * te) ..< mTotal { idx[r] = lastE }
        let indices = Tensor.empty(shape: [mTotal], dtype: .u32)
        indices.copyIn(from: idx)

        let x = Tensor.empty(shape: [mTotal, kIn], dtype: .f16)
        x.copyIn(from: (0 ..< mTotal * kIn).map { Float16(Float($0 % 17) * 0.01 - 0.08) })

        let wCount = nExperts * nOut * packs
        let w = Tensor.empty(shape: [nExperts, nOut, packs], dtype: .u32)
        var wh = [UInt32](repeating: 0, count: wCount)
        for i in 0 ..< wCount {
            var c = [UInt32](repeating: 0, count: 16)
            for j in 0 ..< 16 { c[j] = UInt32((i &+ j) & 3) }
            wh[i] = Self.pack16(c)
        }
        w.copyIn(from: wh)
        let sb = nExperts * nOut * groups
        let scales = Tensor.empty(shape: [nExperts, nOut, groups], dtype: .f16)
        let biases = Tensor.empty(shape: [nExperts, nOut, groups], dtype: .f16)
        var sh = [Float16](repeating: 0, count: sb)
        var bh = [Float16](repeating: 0, count: sb)
        for i in 0 ..< sb {
            sh[i] = Float16(0.05 + Float(i % 7) * 0.01)
            bh[i] = Float16(Float(i % 4) * -0.01)
        }
        scales.copyIn(from: sh)
        biases.copyIn(from: bh)

        let outEG = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        let outDense = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        // Per-expert padded workspace for dense path.
        let xPad = Tensor.empty(shape: [tePad, kIn], dtype: .f16)
        let oPad = Tensor.empty(shape: [tePad, nOut], dtype: .f16)

        setenv("FFAI_HY3_EXPERT_GRID", "1", 1)
        setenv("FFAI_HY3_NO_MOE_BM32", "1", 1)
        defer {
            unsetenv("FFAI_HY3_EXPERT_GRID")
            unsetenv("FFAI_HY3_NO_MOE_BM32")
        }

        func timeEG() -> Double {
            for _ in 0 ..< 2 {
                outEG.zero()
                Self.runAndWait { cmd in
                    Ops.moeGatherDequantGemmInt2(
                        input: x, weight: w, scales: scales, biases: biases,
                        indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                        groupSize: groupSize, on: cmd, into: outEG, nExperts: nExperts)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                outEG.zero()
                Self.runAndWait { cmd in
                    Ops.moeGatherDequantGemmInt2(
                        input: x, weight: w, scales: scales, biases: biases,
                        indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                        groupSize: groupSize, on: cmd, into: outEG, nExperts: nExperts)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        // Per-expert dense: slice weight expert row as [nOut, packs], x rows, pad M to 32.
        func timePerExpertDense() -> Double {
            // Precompute expert row ranges from host indices.
            var ranges: [(e: Int, lo: Int, hi: Int)] = []
            var i = 0
            while i < mTotal {
                let e = Int(idx[i])
                let lo = i
                while i < mTotal && Int(idx[i]) == e { i += 1 }
                ranges.append((e, lo, i))
            }
            for _ in 0 ..< 2 {
                outDense.zero()
                Self.runAndWait { cmd in
                    for r in ranges {
                        let n = r.hi - r.lo
                        // Copy x rows into pad (blit).
                        let blit = cmd.makeBlitCommandEncoder()!
                        let rowBytes = kIn * 2
                        blit.copy(
                            from: x.buffer, sourceOffset: r.lo * rowBytes,
                            to: xPad.buffer, destinationOffset: 0,
                            size: n * rowBytes)
                        if tePad > n {
                            blit.fill(
                                buffer: xPad.buffer,
                                range: (n * rowBytes) ..< (tePad * rowBytes),
                                value: 0)
                        }
                        blit.endEncoding()
                        let wE = Tensor(
                            buffer: w.buffer,
                            offset: w.offset + r.e * nOut * packs * 4,
                            shape: [nOut, packs], dtype: .u32)
                        let sE = Tensor(
                            buffer: scales.buffer,
                            offset: scales.offset + r.e * nOut * groups * 2,
                            shape: [nOut, groups], dtype: .f16)
                        let bE = Tensor(
                            buffer: biases.buffer,
                            offset: biases.offset + r.e * nOut * groups * 2,
                            shape: [nOut, groups], dtype: .f16)
                        Ops.dequantGemmDynamicM(
                            input: xPad, weight: wE, scales: sE, biases: bE,
                            t: tePad, nOut: nOut, kIn: kIn, groupSize: groupSize,
                            on: cmd, device: Device.shared, into: oPad, bits: 2)
                        let blit2 = cmd.makeBlitCommandEncoder()!
                        blit2.copy(
                            from: oPad.buffer, sourceOffset: 0,
                            to: outDense.buffer, destinationOffset: r.lo * nOut * 2,
                            size: n * nOut * 2)
                        blit2.endEncoding()
                    }
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                outDense.zero()
                Self.runAndWait { cmd in
                    for r in ranges {
                        let n = r.hi - r.lo
                        let blit = cmd.makeBlitCommandEncoder()!
                        let rowBytes = kIn * 2
                        blit.copy(
                            from: x.buffer, sourceOffset: r.lo * rowBytes,
                            to: xPad.buffer, destinationOffset: 0,
                            size: n * rowBytes)
                        if tePad > n {
                            blit.fill(
                                buffer: xPad.buffer,
                                range: (n * rowBytes) ..< (tePad * rowBytes),
                                value: 0)
                        }
                        blit.endEncoding()
                        let wE = Tensor(
                            buffer: w.buffer,
                            offset: w.offset + r.e * nOut * packs * 4,
                            shape: [nOut, packs], dtype: .u32)
                        let sE = Tensor(
                            buffer: scales.buffer,
                            offset: scales.offset + r.e * nOut * groups * 2,
                            shape: [nOut, groups], dtype: .f16)
                        let bE = Tensor(
                            buffer: biases.buffer,
                            offset: biases.offset + r.e * nOut * groups * 2,
                            shape: [nOut, groups], dtype: .f16)
                        Ops.dequantGemmDynamicM(
                            input: xPad, weight: wE, scales: sE, biases: bE,
                            t: tePad, nOut: nOut, kIn: kIn, groupSize: groupSize,
                            on: cmd, device: Device.shared, into: oPad, bits: 2)
                        let blit2 = cmd.makeBlitCommandEncoder()!
                        blit2.copy(
                            from: oPad.buffer, sourceOffset: 0,
                            to: outDense.buffer, destinationOffset: r.lo * nOut * 2,
                            size: n * nOut * 2)
                        blit2.endEncoding()
                    }
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        // Fat upper bound: one dense QMM M=2048 (as if single expert).
        func timeFatDense() -> Double {
            // Use expert 0 weights, full x as if dense.
            let w0 = Tensor(
                buffer: w.buffer, offset: w.offset,
                shape: [nOut, packs], dtype: .u32)
            let s0 = Tensor(
                buffer: scales.buffer, offset: scales.offset,
                shape: [nOut, groups], dtype: .f16)
            let b0 = Tensor(
                buffer: biases.buffer, offset: biases.offset,
                shape: [nOut, groups], dtype: .f16)
            // M must be multiple of 32.
            let mPad = ((mTotal + 31) / 32) * 32
            let xFat: Tensor
            if mPad == mTotal {
                xFat = x
            } else {
                let xf = Tensor.empty(shape: [mPad, kIn], dtype: .f16)
                Self.runAndWait { cmd in
                    let blit = cmd.makeBlitCommandEncoder()!
                    blit.copy(
                        from: x.buffer, sourceOffset: 0,
                        to: xf.buffer, destinationOffset: 0,
                        size: mTotal * kIn * 2)
                    blit.fill(
                        buffer: xf.buffer,
                        range: (mTotal * kIn * 2) ..< (mPad * kIn * 2),
                        value: 0)
                    blit.endEncoding()
                }
                xFat = xf
            }
            let oFat = Tensor.empty(shape: [mPad, nOut], dtype: .f16)
            for _ in 0 ..< 2 {
                Self.runAndWait { cmd in
                    Ops.dequantGemmDynamicM(
                        input: xFat, weight: w0, scales: s0, biases: b0,
                        t: mPad, nOut: nOut, kIn: kIn, groupSize: groupSize,
                        on: cmd, device: Device.shared, into: oFat, bits: 2)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                Self.runAndWait { cmd in
                    Ops.dequantGemmDynamicM(
                        input: xFat, weight: w0, scales: s0, biases: b0,
                        t: mPad, nOut: nOut, kIn: kIn, groupSize: groupSize,
                        on: cmd, device: Device.shared, into: oFat, bits: 2)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        // Per-expert dense without host blits: pad once into contiguous
        // [nActive, tePad, k] workspace, then 90 dense M=tePad QMMs on one CB.
        // Tests whether serial dense QMM beats EG when encode thrash is gone.
        func timePerExpertPacked() -> Double {
            var ranges: [(e: Int, lo: Int, hi: Int)] = []
            var i = 0
            while i < mTotal {
                let e = Int(idx[i])
                let lo = i
                while i < mTotal && Int(idx[i]) == e { i += 1 }
                ranges.append((e, lo, i))
            }
            let nA = ranges.count
            let xPack = Tensor.empty(shape: [nA * tePad, kIn], dtype: .f16)
            let oPack = Tensor.empty(shape: [nA * tePad, nOut], dtype: .f16)
            let outPacked = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)

            func packAndRun(on cmd: MTLCommandBuffer) {
                let blit = cmd.makeBlitCommandEncoder()!
                let rowBytes = kIn * 2
                for (a, r) in ranges.enumerated() {
                    let n = r.hi - r.lo
                    let dstBase = a * tePad * rowBytes
                    blit.copy(
                        from: x.buffer, sourceOffset: r.lo * rowBytes,
                        to: xPack.buffer, destinationOffset: dstBase,
                        size: n * rowBytes)
                    if tePad > n {
                        blit.fill(
                            buffer: xPack.buffer,
                            range: (dstBase + n * rowBytes) ..< (dstBase + tePad * rowBytes),
                            value: 0)
                    }
                }
                blit.endEncoding()
                for (a, r) in ranges.enumerated() {
                    let xE = Tensor(
                        buffer: xPack.buffer,
                        offset: a * tePad * kIn * 2,
                        shape: [tePad, kIn], dtype: .f16)
                    let oE = Tensor(
                        buffer: oPack.buffer,
                        offset: a * tePad * nOut * 2,
                        shape: [tePad, nOut], dtype: .f16)
                    let wE = Tensor(
                        buffer: w.buffer,
                        offset: w.offset + r.e * nOut * packs * 4,
                        shape: [nOut, packs], dtype: .u32)
                    let sE = Tensor(
                        buffer: scales.buffer,
                        offset: scales.offset + r.e * nOut * groups * 2,
                        shape: [nOut, groups], dtype: .f16)
                    let bE = Tensor(
                        buffer: biases.buffer,
                        offset: biases.offset + r.e * nOut * groups * 2,
                        shape: [nOut, groups], dtype: .f16)
                    Ops.dequantGemmDynamicM(
                        input: xE, weight: wE, scales: sE, biases: bE,
                        t: tePad, nOut: nOut, kIn: kIn, groupSize: groupSize,
                        on: cmd, device: Device.shared, into: oE, bits: 2)
                }
                // Unpack first n rows per expert back to token order (sorted).
                let blit2 = cmd.makeBlitCommandEncoder()!
                for (a, r) in ranges.enumerated() {
                    let n = r.hi - r.lo
                    blit2.copy(
                        from: oPack.buffer, sourceOffset: a * tePad * nOut * 2,
                        to: outPacked.buffer, destinationOffset: r.lo * nOut * 2,
                        size: n * nOut * 2)
                }
                blit2.endEncoding()
            }

            for _ in 0 ..< 2 {
                outPacked.zero()
                Self.runAndWait { packAndRun(on: $0) }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                outPacked.zero()
                Self.runAndWait { packAndRun(on: $0) }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        let tEG = timeEG()
        let tPer = timePerExpertDense()
        let tPacked = timePerExpertPacked()
        let tFat = timeFatDense()
        let line = String(
            format:
                "EG=%.2f ms  perExpertDense=%.2f ms (%.2fx)  packedDense=%.2f ms (%.2fx)  fatDenseM2048=%.2f ms (%.2fx vs EG)\n",
            tEG, tPer, tEG / max(tPer, 1e-9), tPacked, tEG / max(tPacked, 1e-9),
            tFat, tEG / max(tFat, 1e-9))
        fputs(line, stderr)
        if let scratch = ProcessInfo.processInfo.environment["HY3_SCRATCH"] {
            let path = (scratch as NSString).appendingPathComponent("hy3-dense-vs-eg.log")
            try? line.write(toFile: path, atomically: true, encoding: .utf8)
        }
        #expect(tEG > 0 && tPer > 0 && tPacked > 0 && tFat > 0)
    }
}
