// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Active-expert densify + dense EG vs sparse EG (empty TGs).
//
//   swift test -c release --filter MoEInt2Densify

import Foundation
import Metal
import IronSwift
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("MoE int2 active densify")
struct MoEInt2DensifyTests {
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

    static func makeW(
        nExperts: Int, nOut: Int, kIn: Int, groupSize: Int, seed: Int
    ) -> (w: Tensor, s: Tensor, b: Tensor) {
        let packs = kIn / 16
        let groups = kIn / groupSize
        let wCount = nExperts * nOut * packs
        let w = Tensor.empty(shape: [nExperts, nOut, packs], dtype: .u32)
        var wh = [UInt32](repeating: 0, count: wCount)
        for i in 0 ..< wCount {
            var c = [UInt32](repeating: 0, count: 16)
            for j in 0 ..< 16 { c[j] = UInt32((seed &+ i &+ j) & 3) }
            wh[i] = pack16(c)
        }
        w.copyIn(from: wh)
        let sb = nExperts * nOut * groups
        let s = Tensor.empty(shape: [nExperts, nOut, groups], dtype: .f16)
        let b = Tensor.empty(shape: [nExperts, nOut, groups], dtype: .f16)
        var sh = [Float16](repeating: 0, count: sb)
        var bh = [Float16](repeating: 0, count: sb)
        for i in 0 ..< sb {
            sh[i] = Float16(0.05 + Float((i + seed) % 7) * 0.01)
            bh[i] = Float16(Float((i + seed) % 4) * -0.01)
        }
        s.copyIn(from: sh); b.copyIn(from: bh)
        return (w, s, b)
    }

    @Test(
        "densify+remap+dense EG matches sparse EG",
        .enabled(if: mppAvailable, "MPP cooperative tensor — Apple GPU Family 9+"))
    func densifyMatchesSparse() throws {
        let mTotal = 256
        let nOut = 128
        let kIn = 128
        let nExperts = 16
        let nActive = 8
        let groupSize = 64

        setenv("BUTTER_HY3_EXPERT_GRID", "1", 1)
        setenv("BUTTER_HY3_NO_MOE_BN64", "1", 1)
        defer {
            unsetenv("BUTTER_HY3_EXPERT_GRID")
            unsetenv("BUTTER_HY3_NO_MOE_BN64")
        }

        // Sparse expert ids: 0,2,4,...,14
        var idx = [UInt32](repeating: 0, count: mTotal)
        let rowsPer = mTotal / nActive
        var activeHost = [UInt32]()
        for a in 0 ..< nActive {
            let e = UInt32(a * 2)
            activeHost.append(e)
            for r in (a * rowsPer) ..< ((a + 1) * rowsPer) { idx[r] = e }
        }
        let indices = Tensor.empty(shape: [mTotal], dtype: .u32)
        indices.copyIn(from: idx)
        let active = Tensor.empty(shape: [nActive], dtype: .u32)
        active.copyIn(from: activeHost)

        let x = Tensor.empty(shape: [mTotal, kIn], dtype: .f16)
        x.copyIn(from: (0 ..< mTotal * kIn).map { Float16(Float($0 % 13) * 0.01 - 0.06) })
        let src = Self.makeW(
            nExperts: nExperts, nOut: nOut, kIn: kIn, groupSize: groupSize, seed: 3)

        // Sparse EG reference
        let outSparse = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        outSparse.zero()
        Self.runAndWait { cmd in
            Ops.moeGatherDequantGemmInt2(
                input: x, weight: src.w, scales: src.s, biases: src.b,
                indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                groupSize: groupSize, on: cmd, into: outSparse, nExperts: nExperts)
        }

        // Densify + remap + dense EG
        let packs = kIn / 16
        let groups = kIn / groupSize
        let wD = Tensor.empty(shape: [nActive, nOut, packs], dtype: .u32)
        let sD = Tensor.empty(shape: [nActive, nOut, groups], dtype: .f16)
        let bD = Tensor.empty(shape: [nActive, nOut, groups], dtype: .f16)
        let denseIdx = Tensor.empty(shape: [mTotal], dtype: .u32)
        let outDense = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        outDense.zero()
        Self.runAndWait { cmd in
            Ops.moeDensifyInt2Experts(
                weightSrc: src.w, scalesSrc: src.s, biasesSrc: src.b,
                activeExperts: active,
                weightDst: wD, scalesDst: sD, biasesDst: bD,
                nOut: nOut, kIn: kIn, groupSize: groupSize, nActive: nActive,
                on: cmd)
            Ops.moeRemapExpertsDense(
                sortedExperts: indices, activeExperts: active,
                denseIndices: denseIdx, mTotal: mTotal, nActive: nActive, on: cmd)
            Ops.moeGatherDequantGemmInt2(
                input: x, weight: wD, scales: sD, biases: bD,
                indices: denseIdx, mTotal: mTotal, nOut: nOut, kIn: kIn,
                groupSize: groupSize, on: cmd, into: outDense, nExperts: nActive)
        }

        let ref = outSparse.toArray(as: Float16.self).map { Float($0) }
        let got = outDense.toArray(as: Float16.self).map { Float($0) }
        var nBad = 0
        var maxAbs: Float = 0
        for i in 0 ..< got.count {
            #expect(got[i].isFinite)
            let d = abs(got[i] - ref[i])
            if d > maxAbs { maxAbs = d }
            if d > max(1e-2, 0.02 * abs(ref[i])) { nBad += 1 }
        }
        #expect(nBad == 0, "densify mismatch: \(nBad)/\(got.count) maxAbs=\(maxAbs)")
    }

    @Test(
        "microbench densify+dense EG vs sparse EG Hy3 shape",
        .enabled(if: mppAvailable, "MPP cooperative tensor — Apple GPU Family 9+"))
    func microbenchHy3() throws {
        let mTotal = 2048
        let nOut = 1536
        let kIn = 4096
        let nExperts = 192
        let nActive = 90
        let groupSize = 64
        let warmup = 2
        let iters = 6

        setenv("BUTTER_HY3_EXPERT_GRID", "1", 1)
        setenv("BUTTER_HY3_NO_MOE_BN64", "1", 1)
        defer {
            unsetenv("BUTTER_HY3_EXPERT_GRID")
            unsetenv("BUTTER_HY3_NO_MOE_BN64")
        }

        // Sparse indices over 192 with 90 active
        var idx = [UInt32](repeating: 0, count: mTotal)
        var activeHost = [UInt32]()
        let rowsPer = max(mTotal / nActive, 1)
        for a in 0 ..< nActive {
            let e = UInt32(a * nExperts / nActive)
            activeHost.append(e)
            let lo = a * rowsPer
            let hi = min((a + 1) * rowsPer, mTotal)
            if lo < hi { for r in lo ..< hi { idx[r] = e } }
        }
        // Sort indices for EG (required)
        idx.sort()
        activeHost = Array(Set(activeHost)).sorted()
        let nAct = activeHost.count

        let indices = Tensor.empty(shape: [mTotal], dtype: .u32)
        indices.copyIn(from: idx)
        let active = Tensor.empty(shape: [nAct], dtype: .u32)
        active.copyIn(from: activeHost)

        let x = Tensor.empty(shape: [mTotal, kIn], dtype: .f16)
        x.copyIn(from: (0 ..< mTotal * kIn).map { Float16(Float($0 % 17) * 0.01 - 0.08) })
        let src = Self.makeW(
            nExperts: nExperts, nOut: nOut, kIn: kIn, groupSize: groupSize, seed: 1)

        let packs = kIn / 16
        let groups = kIn / groupSize
        let wD = Tensor.empty(shape: [nAct, nOut, packs], dtype: .u32)
        let sD = Tensor.empty(shape: [nAct, nOut, groups], dtype: .f16)
        let bD = Tensor.empty(shape: [nAct, nOut, groups], dtype: .f16)
        let denseIdx = Tensor.empty(shape: [mTotal], dtype: .u32)
        let out = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)

        func timeMs(densify: Bool) -> Double {
            for _ in 0 ..< warmup {
                out.zero()
                Self.runAndWait { cmd in
                    if densify {
                        Ops.moeDensifyInt2Experts(
                            weightSrc: src.w, scalesSrc: src.s, biasesSrc: src.b,
                            activeExperts: active,
                            weightDst: wD, scalesDst: sD, biasesDst: bD,
                            nOut: nOut, kIn: kIn, groupSize: groupSize, nActive: nAct,
                            on: cmd)
                        Ops.moeRemapExpertsDense(
                            sortedExperts: indices, activeExperts: active,
                            denseIndices: denseIdx, mTotal: mTotal, nActive: nAct, on: cmd)
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: wD, scales: sD, biases: bD,
                            indices: denseIdx, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: out, nExperts: nAct)
                    } else {
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: src.w, scales: src.s, biases: src.b,
                            indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: out, nExperts: nExperts)
                    }
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                out.zero()
                Self.runAndWait { cmd in
                    if densify {
                        Ops.moeDensifyInt2Experts(
                            weightSrc: src.w, scalesSrc: src.s, biasesSrc: src.b,
                            activeExperts: active,
                            weightDst: wD, scalesDst: sD, biasesDst: bD,
                            nOut: nOut, kIn: kIn, groupSize: groupSize, nActive: nAct,
                            on: cmd)
                        Ops.moeRemapExpertsDense(
                            sortedExperts: indices, activeExperts: active,
                            denseIndices: denseIdx, mTotal: mTotal, nActive: nAct, on: cmd)
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: wD, scales: sD, biases: bD,
                            indices: denseIdx, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: out, nExperts: nAct)
                    } else {
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: src.w, scales: src.s, biases: src.b,
                            indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: out, nExperts: nExperts)
                    }
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000.0
        }

        let msSparse = timeMs(densify: false)
        let msDense = timeMs(densify: true)
        let speedup = msSparse / max(msDense, 1e-9)
        let line = String(
            format: "densify path: sparse=%.2f ms densify+dense=%.2f ms speedup=%.2fx (incl densify cost)",
            msSparse, msDense, speedup)
        print(line)
        let scratch = ProcessInfo.processInfo.environment["BUTTER_TEST_SCRATCH"]
            ?? "/var/folders/y2/mjncn_qn211bjbdr94zpzps40000gn/T/grok-goal-3df794e08dd7/implementer"
        try? (line + "\n").write(
            toFile: scratch + "/hy3-densify-microbench.log",
            atomically: true, encoding: .utf8)
        #expect(msDense.isFinite && msSparse.isFinite)
    }
}
