// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Dual-M expert-grid int2 gather (BM=32) vs BM=16 EG:
// correctness + wall microbench at Hy3-ish GU shape (Te≈23).
//
//   swift test -c release --filter MoEInt2EgBm32

import Foundation
import Metal
import MetalTileSwift
import TestHelpers
import Testing

@testable import FFAI

@Suite("MoE int2 EG BM32 gather")
struct MoEInt2EgBm32Tests {
    static let mppAvailable: Bool = {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple9)
    }()

    static func pack16(_ codes: [UInt32]) -> UInt32 {
        precondition(codes.count == 16)
        var word: UInt32 = 0
        for i in 0 ..< 16 {
            word |= (codes[i] & 0x3) << UInt32(i * 2)
        }
        return word
    }

    static func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let cmd = Device.shared.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "cmd status \(cmd.status.rawValue)")
    }

    static func makeProblem(
        mTotal: Int, nOut: Int, kIn: Int, nExperts: Int, nActive: Int, groupSize: Int
    ) -> (
        x: Tensor, w: Tensor, scales: Tensor, biases: Tensor,
        indices: Tensor
    ) {
        precondition(nOut % 32 == 0 && kIn % 16 == 0 && kIn % groupSize == 0)
        let packsPerRow = kIn / 16
        let groupsPerRow = kIn / groupSize
        let rowsPer = max(mTotal / nActive, 1)
        var idx = [UInt32](repeating: 0, count: mTotal)
        for a in 0 ..< nActive {
            let e = UInt32(a * nExperts / nActive)
            let lo = a * rowsPer
            let hi = min((a + 1) * rowsPer, mTotal)
            if lo < hi {
                for r in lo ..< hi { idx[r] = e }
            }
        }
        let lastE = UInt32((nActive - 1) * nExperts / nActive)
        for r in (nActive * rowsPer) ..< mTotal { idx[r] = lastE }

        let x = Tensor.empty(shape: [mTotal, kIn], dtype: .f16)
        x.copyIn(from: (0 ..< (mTotal * kIn)).map { Float16(Float($0 % 17) * 0.01 - 0.08) })

        let wCount = nExperts * nOut * packsPerRow
        let w = Tensor.empty(shape: [nExperts, nOut, packsPerRow], dtype: .u32)
        var wHost = [UInt32](repeating: 0, count: wCount)
        for i in 0 ..< wCount {
            var codes = [UInt32](repeating: 0, count: 16)
            for c in 0 ..< 16 { codes[c] = UInt32((i + c) & 3) }
            wHost[i] = pack16(codes)
        }
        w.copyIn(from: wHost)

        let sbCount = nExperts * nOut * groupsPerRow
        let scales = Tensor.empty(shape: [nExperts, nOut, groupsPerRow], dtype: .f16)
        let biases = Tensor.empty(shape: [nExperts, nOut, groupsPerRow], dtype: .f16)
        var sHost = [Float16](repeating: 0, count: sbCount)
        var bHost = [Float16](repeating: 0, count: sbCount)
        for i in 0 ..< sbCount {
            sHost[i] = Float16(0.05 + Float(i % 7) * 0.01)
            bHost[i] = Float16(Float(i % 5) * -0.01)
        }
        scales.copyIn(from: sHost)
        biases.copyIn(from: bHost)

        let indices = Tensor.empty(shape: [mTotal], dtype: .u32)
        indices.copyIn(from: idx)
        return (x, w, scales, biases, indices)
    }

    @Test("BM32 EG matches BM16 EG on multi-expert f16 shape")
    func bm32MatchesBm16() throws {
        try #require(Self.mppAvailable)
        let mTotal = 128
        let nOut = 128
        let kIn = 128
        let nExperts = 8
        let groupSize = 64

        setenv("FFAI_HY3_EXPERT_GRID", "1", 1)
        defer {
            unsetenv("FFAI_HY3_EXPERT_GRID")
            unsetenv("FFAI_HY3_MOE_BM32")
            unsetenv("FFAI_HY3_NO_MOE_BM32")
            unsetenv("FFAI_HY3_MOE_BN64")
        }
        unsetenv("FFAI_HY3_MOE_BN64")

        let p = Self.makeProblem(
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            nExperts: nExperts, nActive: nExperts, groupSize: groupSize)

        setenv("FFAI_HY3_NO_MOE_BM32", "1", 1)
        let out16 = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        out16.zero()
        Self.runAndWait { cmd in
            Ops.moeGatherDequantGemmInt2(
                input: p.x, weight: p.w, scales: p.scales, biases: p.biases,
                indices: p.indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                on: cmd, into: out16, nExperts: nExperts)
        }
        let ref = out16.toArray(as: Float16.self).map { Float($0) }

        unsetenv("FFAI_HY3_NO_MOE_BM32")
        setenv("FFAI_HY3_MOE_BM32", "1", 1)
        let out32 = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        out32.zero()
        Self.runAndWait { cmd in
            Ops.moeGatherDequantGemmInt2(
                input: p.x, weight: p.w, scales: p.scales, biases: p.biases,
                indices: p.indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                on: cmd, into: out32, nExperts: nExperts)
        }
        let got = out32.toArray(as: Float16.self).map { Float($0) }

        #expect(got.count == ref.count)
        var maxAbs: Float = 0
        var nBad = 0
        for i in 0 ..< got.count {
            #expect(got[i].isFinite, "non-finite at \(i): \(got[i])")
            let d = abs(got[i] - ref[i])
            if d > maxAbs { maxAbs = d }
            let tol = max(1e-2, 0.02 * abs(ref[i]))
            if d > tol { nBad += 1 }
        }
        #expect(nBad == 0, "BM32 vs BM16: \(nBad)/\(got.count) exceed tol; maxAbs=\(maxAbs)")
    }

    @Test("microbench BM16 vs BM32 EG at Hy3 GU shape (synthetic)")
    func microbenchHy3Shape() throws {
        try #require(Self.mppAvailable)
        // Hy3 GU: mTotal=2048, N=1536, K=4096, ~90 active of 192.
        let mTotal = 2048
        let nOut = 1536
        let kIn = 4096
        let nExperts = 192
        let nActive = 90
        let groupSize = 64
        let iters = 8

        setenv("FFAI_HY3_EXPERT_GRID", "1", 1)
        defer {
            unsetenv("FFAI_HY3_EXPERT_GRID")
            unsetenv("FFAI_HY3_MOE_BM32")
            unsetenv("FFAI_HY3_NO_MOE_BM32")
            unsetenv("FFAI_HY3_MOE_BN64")
        }
        unsetenv("FFAI_HY3_MOE_BN64")

        let p = Self.makeProblem(
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            nExperts: nExperts, nActive: nActive, groupSize: groupSize)
        let out = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)

        func timePath(bm32: Bool) -> Double {
            if bm32 {
                unsetenv("FFAI_HY3_NO_MOE_BM32")
                setenv("FFAI_HY3_MOE_BM32", "1", 1)
            } else {
                setenv("FFAI_HY3_NO_MOE_BM32", "1", 1)
                unsetenv("FFAI_HY3_MOE_BM32")
            }
            // Warmup
            for _ in 0 ..< 2 {
                out.zero()
                Self.runAndWait { cmd in
                    Ops.moeGatherDequantGemmInt2(
                        input: p.x, weight: p.w, scales: p.scales, biases: p.biases,
                        indices: p.indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                        on: cmd, into: out, nExperts: nExperts)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                out.zero()
                Self.runAndWait { cmd in
                    Ops.moeGatherDequantGemmInt2(
                        input: p.x, weight: p.w, scales: p.scales, biases: p.biases,
                        indices: p.indices,
                        mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                        on: cmd, into: out, nExperts: nExperts)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000.0
        }

        let t16 = timePath(bm32: false)
        let t32 = timePath(bm32: true)
        let speedup = t16 / t32
        let line = String(
            format: "EG BM16: %.2f ms  BM32: %.2f ms  speedup: %.2fx  (M=%d N=%d K=%d Eactive=%d)\n",
            t16, t32, speedup, mTotal, nOut, kIn, nActive)
        fputs(line, stderr)
        // Write to SCRATCH if present for residual logs.
        if let scratch = ProcessInfo.processInfo.environment["HY3_SCRATCH"] {
            let path = (scratch as NSString).appendingPathComponent("hy3-bm32-microbench.log")
            try? line.write(toFile: path, atomically: true, encoding: .utf8)
        }
        #expect(t16 > 0 && t32 > 0)
    }
}
