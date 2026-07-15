// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Dual-N expert-grid int2 gather (BN=64) vs BN=32 EG:
// correctness (match) + wall microbench at Hy3-ish GU shape.
//
// Run:
//   swift test -c release --filter MoEInt2EgBn64
// Does NOT load the 92G Hy3 checkpoint — synthetic tensors only.

import Foundation
import Metal
import MetalTileSwift
import TestHelpers
import Testing

@testable import FFAI

@Suite("MoE int2 EG BN64 gather")
struct MoEInt2EgBn64Tests {
    static let mppAvailable: Bool = {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple9)
    }()

    /// Pack 16 int2 codes (0..3) into one u32 (low bits first).
    static func pack16(_ codes: [UInt32]) -> UInt32 {
        precondition(codes.count == 16)
        var word: UInt32 = 0
        for i in 0 ..< 16 {
            word |= (codes[i] & 0x3) << UInt32(i * 2)
        }
        return word
    }

    static func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let device = Device.shared
        let cmd = device.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "cmd status \(cmd.status.rawValue)")
    }

    /// Build sorted multi-expert int2 gather problem.
    /// Weight layout: [nExperts, nOut, packsPerRow] u32, packsPerRow = kIn/16.
    static func makeProblem(
        mTotal: Int, nOut: Int, kIn: Int, nExperts: Int, nActive: Int, groupSize: Int
    ) -> (
        x: Tensor, w: Tensor, scales: Tensor, biases: Tensor,
        indices: Tensor, out: Tensor
    ) {
        precondition(nOut % 64 == 0)
        precondition(kIn % groupSize == 0)
        precondition(kIn % 16 == 0)
        let packsPerRow = kIn / 16
        let groupsPerRow = kIn / groupSize
        let rowsPer = max(mTotal / nActive, 1)
        var idx = [UInt32](repeating: 0, count: mTotal)
        for a in 0 ..< nActive {
            let e = UInt32(a * nExperts / nActive)  // spread across expert ids
            let lo = a * rowsPer
            let hi = min((a + 1) * rowsPer, mTotal)
            if lo < hi {
                for r in lo ..< hi { idx[r] = e }
            }
        }
        // Tail → last active
        let lastE = UInt32((nActive - 1) * nExperts / nActive)
        for r in (nActive * rowsPer) ..< mTotal { idx[r] = lastE }

        let x = Tensor.empty(shape: [mTotal, kIn], dtype: .f16)
        var xHost = [Float16](repeating: 0, count: mTotal * kIn)
        for i in 0 ..< xHost.count {
            xHost[i] = Float16(Float(i % 17) * 0.01 - 0.08)
        }
        x.copyIn(from: xHost)

        let wCount = nExperts * nOut * packsPerRow
        let w = Tensor.empty(shape: [nExperts, nOut, packsPerRow], dtype: .u32)
        var wHost = [UInt32](repeating: 0, count: wCount)
        for i in 0 ..< wCount {
            // Deterministic 2-bit pattern 0,1,2,3,0,...
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

        let out = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        out.zero()
        return (x, w, scales, biases, indices, out)
    }

    @Test("BN64 EG matches BN32 EG on multi-expert f16 shape")
    func bn64MatchesBn32() throws {
        try #require(Self.mppAvailable)
        // Modest shape: M=128, N=128, K=128, 8 experts all active.
        let mTotal = 128
        let nOut = 128
        let kIn = 128
        let nExperts = 8
        let groupSize = 64

        setenv("FFAI_HY3_EXPERT_GRID", "1", 1)
        defer {
            unsetenv("FFAI_HY3_EXPERT_GRID")
            unsetenv("FFAI_HY3_MOE_BN64")
            unsetenv("FFAI_HY3_NO_MOE_BN64")
        }

        let p = Self.makeProblem(
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            nExperts: nExperts, nActive: nExperts, groupSize: groupSize)

        // BN32 reference
        setenv("FFAI_HY3_NO_MOE_BN64", "1", 1)
        unsetenv("FFAI_HY3_MOE_BN64")
        let out32 = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        out32.zero()
        Self.runAndWait { cmd in
            Ops.moeGatherDequantGemmInt2(
                input: p.x, weight: p.w, scales: p.scales, biases: p.biases,
                indices: p.indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                on: cmd, into: out32, nExperts: nExperts)
        }
        let ref = out32.toArray(as: Float16.self).map { Float($0) }

        // BN64
        unsetenv("FFAI_HY3_NO_MOE_BN64")
        setenv("FFAI_HY3_MOE_BN64", "1", 1)
        let out64 = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        out64.zero()
        Self.runAndWait { cmd in
            Ops.moeGatherDequantGemmInt2(
                input: p.x, weight: p.w, scales: p.scales, biases: p.biases,
                indices: p.indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                on: cmd, into: out64, nExperts: nExperts)
        }
        let got = out64.toArray(as: Float16.self).map { Float($0) }

        #expect(got.count == ref.count)
        var maxAbs: Float = 0
        var nBad = 0
        for i in 0 ..< got.count {
            #expect(got[i].isFinite, "non-finite at \(i): \(got[i])")
            let d = abs(got[i] - ref[i])
            if d > maxAbs { maxAbs = d }
            // f16 accum noise: allow small relative error
            let tol = max(1e-2, 0.02 * abs(ref[i]))
            if d > tol { nBad += 1 }
        }
        #expect(nBad == 0, "BN64 vs BN32: \(nBad)/\(got.count) exceed tol; maxAbs=\(maxAbs)")
    }

    @Test("microbench BN32 vs BN64 EG at Hy3 GU shape (synthetic)")
    func microbenchHy3Shape() throws {
        try #require(Self.mppAvailable)
        // Hy3 gate: M=2048 (T256×top8), N=1536, K=4096, E=192, ~90 active.
        let mTotal = 2048
        let nOut = 1536
        let kIn = 4096
        let nExperts = 192
        let nActive = 90
        let groupSize = 64
        let warmup = 3
        let iters = 10

        setenv("FFAI_HY3_EXPERT_GRID", "1", 1)
        defer {
            unsetenv("FFAI_HY3_EXPERT_GRID")
            unsetenv("FFAI_HY3_MOE_BN64")
            unsetenv("FFAI_HY3_NO_MOE_BN64")
        }

        let p = Self.makeProblem(
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            nExperts: nExperts, nActive: nActive, groupSize: groupSize)

        func timeMs(bn64: Bool) -> Double {
            if bn64 {
                unsetenv("FFAI_HY3_NO_MOE_BN64")
                setenv("FFAI_HY3_MOE_BN64", "1", 1)
            } else {
                setenv("FFAI_HY3_NO_MOE_BN64", "1", 1)
                unsetenv("FFAI_HY3_MOE_BN64")
            }
            let out = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
            for _ in 0 ..< warmup {
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
            let t1 = CFAbsoluteTimeGetCurrent()
            return (t1 - t0) / Double(iters) * 1000.0
        }

        let ms32 = timeMs(bn64: false)
        let ms64 = timeMs(bn64: true)
        let speedup = ms32 / max(ms64, 1e-9)
        let line = String(
            format: "EG BN32: %.2f ms  BN64: %.2f ms  speedup: %.2fx  (M=%d N=%d K=%d Eactive=%d)",
            ms32, ms64, speedup, mTotal, nOut, kIn, nActive)
        print(line)
        // Write to scratch for residual
        let scratch = ProcessInfo.processInfo.environment["FFAI_TEST_SCRATCH"]
            ?? "/var/folders/y2/mjncn_qn211bjbdr94zpzps40000gn/T/grok-goal-3df794e08dd7/implementer"
        try? (line + "\n").write(
            toFile: scratch + "/hy3-bn64-microbench.log",
            atomically: true, encoding: .utf8)
        // Sanity: both produce finite (already waited). Don't fail on speedup —
        // battery may hide wins; just ensure BN64 is not catastrophically slower.
        #expect(ms64 < ms32 * 3.0, "BN64 >3× slower than BN32: \(line)")
    }

    @Test("microbench empty-TG tax: nExperts=90 dense vs 192 sparse")
    func microbenchEmptyTGTax() throws {
        try #require(Self.mppAvailable)
        let mTotal = 2048
        let nOut = 1536
        let kIn = 4096
        let groupSize = 64
        let nActive = 90
        let warmup = 2
        let iters = 8

        setenv("FFAI_HY3_EXPERT_GRID", "1", 1)
        setenv("FFAI_HY3_NO_MOE_BN64", "1", 1)
        defer {
            unsetenv("FFAI_HY3_EXPERT_GRID")
            unsetenv("FFAI_HY3_NO_MOE_BN64")
        }

        func timeWith(nExperts: Int, denseActive: Bool) -> Double {
            let p = Self.makeProblem(
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                nExperts: nExperts, nActive: denseActive ? nExperts : nActive,
                groupSize: groupSize)
            let out = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
            for _ in 0 ..< warmup {
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

        // Dense: only 90 expert slots, all used.
        let msDense = timeWith(nExperts: nActive, denseActive: true)
        // Sparse: 192 slots, ~90 active (empty early-exit).
        let msSparse = timeWith(nExperts: 192, denseActive: false)
        let speedup = msSparse / max(msDense, 1e-9)
        let line = String(
            format: "EG empty-TG tax: sparse192=%.2f ms dense90=%.2f ms  dense_speedup=%.2fx",
            msSparse, msDense, speedup)
        print(line)
        let scratch = ProcessInfo.processInfo.environment["FFAI_TEST_SCRATCH"]
            ?? "/var/folders/y2/mjncn_qn211bjbdr94zpzps40000gn/T/grok-goal-3df794e08dd7/implementer"
        try? (line + "\n").write(
            toFile: scratch + "/hy3-empty-tg-tax.log",
            atomically: true, encoding: .utf8)
        #expect(msDense.isFinite && msSparse.isFinite)
    }
}
