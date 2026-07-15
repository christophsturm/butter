// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Fused gate+up EG int2 gather vs two separate EG dispatches.
//
//   swift test -c release --filter MoEInt2FusedGU

import Foundation
import Metal
import MetalTileSwift
import TestHelpers
import Testing

@testable import FFAI

@Suite("MoE int2 fused gate+up EG")
struct MoEInt2FusedGUTests {
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

    static func makeWeights(
        nExperts: Int, nOut: Int, kIn: Int, groupSize: Int, seed: UInt32
    ) -> (w: Tensor, scales: Tensor, biases: Tensor) {
        let packsPerRow = kIn / 16
        let groupsPerRow = kIn / groupSize
        let wCount = nExperts * nOut * packsPerRow
        let w = Tensor.empty(shape: [nExperts, nOut, packsPerRow], dtype: .u32)
        var wHost = [UInt32](repeating: 0, count: wCount)
        for i in 0 ..< wCount {
            var codes = [UInt32](repeating: 0, count: 16)
            for c in 0 ..< 16 { codes[c] = UInt32((Int(seed) &+ i &+ c) & 3) }
            wHost[i] = pack16(codes)
        }
        w.copyIn(from: wHost)
        let sbCount = nExperts * nOut * groupsPerRow
        let scales = Tensor.empty(shape: [nExperts, nOut, groupsPerRow], dtype: .f16)
        let biases = Tensor.empty(shape: [nExperts, nOut, groupsPerRow], dtype: .f16)
        var sHost = [Float16](repeating: 0, count: sbCount)
        var bHost = [Float16](repeating: 0, count: sbCount)
        for i in 0 ..< sbCount {
            sHost[i] = Float16(0.04 + Float((i + Int(seed)) % 9) * 0.01)
            bHost[i] = Float16(Float((i + Int(seed)) % 5) * -0.008)
        }
        scales.copyIn(from: sHost)
        biases.copyIn(from: bHost)
        return (w, scales, biases)
    }

    @Test("fused GU matches two separate EG dispatches")
    func fusedMatchesSeparate() throws {
        try #require(Self.mppAvailable)
        let mTotal = 128
        let nOut = 128
        let kIn = 128
        let nExperts = 8
        let groupSize = 64

        setenv("FFAI_HY3_EXPERT_GRID", "1", 1)
        setenv("FFAI_HY3_NO_MOE_BN64", "1", 1)
        defer {
            unsetenv("FFAI_HY3_EXPERT_GRID")
            unsetenv("FFAI_HY3_NO_MOE_BN64")
        }

        var idx = [UInt32](repeating: 0, count: mTotal)
        let rowsPer = mTotal / nExperts
        for e in 0 ..< nExperts {
            for r in (e * rowsPer) ..< ((e + 1) * rowsPer) { idx[r] = UInt32(e) }
        }
        let indices = Tensor.empty(shape: [mTotal], dtype: .u32)
        indices.copyIn(from: idx)

        let x = Tensor.empty(shape: [mTotal, kIn], dtype: .f16)
        x.copyIn(from: (0 ..< (mTotal * kIn)).map { Float16(Float($0 % 19) * 0.01 - 0.09) })

        let gate = Self.makeWeights(
            nExperts: nExperts, nOut: nOut, kIn: kIn, groupSize: groupSize, seed: 1)
        let up = Self.makeWeights(
            nExperts: nExperts, nOut: nOut, kIn: kIn, groupSize: groupSize, seed: 7)

        let gSep = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        let uSep = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        gSep.zero(); uSep.zero()
        Self.runAndWait { cmd in
            Ops.moeGatherDequantGemmInt2(
                input: x, weight: gate.w, scales: gate.scales, biases: gate.biases,
                indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                groupSize: groupSize, on: cmd, into: gSep, nExperts: nExperts)
            Ops.moeGatherDequantGemmInt2(
                input: x, weight: up.w, scales: up.scales, biases: up.biases,
                indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                groupSize: groupSize, on: cmd, into: uSep, nExperts: nExperts)
        }

        let gF = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        let uF = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        gF.zero(); uF.zero()
        Self.runAndWait { cmd in
            Ops.moeGatherDequantGemmInt2FusedGU(
                input: x,
                gateWeight: gate.w, gateScales: gate.scales, gateBiases: gate.biases,
                upWeight: up.w, upScales: up.scales, upBiases: up.biases,
                indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                nExperts: nExperts, on: cmd, intoGate: gF, intoUp: uF)
        }

        let gRef = gSep.toArray(as: Float16.self).map { Float($0) }
        let uRef = uSep.toArray(as: Float16.self).map { Float($0) }
        let gGot = gF.toArray(as: Float16.self).map { Float($0) }
        let uGot = uF.toArray(as: Float16.self).map { Float($0) }

        func check(_ got: [Float], _ ref: [Float], _ name: String) {
            var nBad = 0
            var maxAbs: Float = 0
            for i in 0 ..< got.count {
                #expect(got[i].isFinite, "\(name) non-finite @\(i)")
                let d = abs(got[i] - ref[i])
                if d > maxAbs { maxAbs = d }
                let tol = max(1e-2, 0.02 * abs(ref[i]))
                if d > tol { nBad += 1 }
            }
            #expect(nBad == 0, "\(name): \(nBad)/\(got.count) bad maxAbs=\(maxAbs)")
        }
        check(gGot, gRef, "gate")
        check(uGot, uRef, "up")
    }

    @Test("microbench fused GU vs 2× EG at Hy3 shape")
    func microbenchHy3Shape() throws {
        try #require(Self.mppAvailable)
        let mTotal = 2048
        let nOut = 1536
        let kIn = 4096
        let nExperts = 192
        let nActive = 90
        let groupSize = 64
        let warmup = 2
        let iters = 8

        setenv("FFAI_HY3_EXPERT_GRID", "1", 1)
        setenv("FFAI_HY3_NO_MOE_BN64", "1", 1)
        defer {
            unsetenv("FFAI_HY3_EXPERT_GRID")
            unsetenv("FFAI_HY3_NO_MOE_BN64")
        }

        var idx = [UInt32](repeating: 0, count: mTotal)
        let rowsPer = max(mTotal / nActive, 1)
        for a in 0 ..< nActive {
            let e = UInt32(a * nExperts / nActive)
            let lo = a * rowsPer
            let hi = min((a + 1) * rowsPer, mTotal)
            if lo < hi { for r in lo ..< hi { idx[r] = e } }
        }
        let lastE = UInt32((nActive - 1) * nExperts / nActive)
        for r in (nActive * rowsPer) ..< mTotal { idx[r] = lastE }
        let indices = Tensor.empty(shape: [mTotal], dtype: .u32)
        indices.copyIn(from: idx)

        let x = Tensor.empty(shape: [mTotal, kIn], dtype: .f16)
        x.copyIn(from: (0 ..< (mTotal * kIn)).map { Float16(Float($0 % 17) * 0.01 - 0.08) })
        let gate = Self.makeWeights(
            nExperts: nExperts, nOut: nOut, kIn: kIn, groupSize: groupSize, seed: 3)
        let up = Self.makeWeights(
            nExperts: nExperts, nOut: nOut, kIn: kIn, groupSize: groupSize, seed: 11)

        let gOut = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        let uOut = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)

        func timeMs(fused: Bool) -> Double {
            for _ in 0 ..< warmup {
                gOut.zero(); uOut.zero()
                Self.runAndWait { cmd in
                    if fused {
                        Ops.moeGatherDequantGemmInt2FusedGU(
                            input: x,
                            gateWeight: gate.w, gateScales: gate.scales, gateBiases: gate.biases,
                            upWeight: up.w, upScales: up.scales, upBiases: up.biases,
                            indices: indices,
                            mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                            nExperts: nExperts, on: cmd, intoGate: gOut, intoUp: uOut)
                    } else {
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: gate.w, scales: gate.scales, biases: gate.biases,
                            indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: gOut, nExperts: nExperts)
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: up.w, scales: up.scales, biases: up.biases,
                            indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: uOut, nExperts: nExperts)
                    }
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                gOut.zero(); uOut.zero()
                Self.runAndWait { cmd in
                    if fused {
                        Ops.moeGatherDequantGemmInt2FusedGU(
                            input: x,
                            gateWeight: gate.w, gateScales: gate.scales, gateBiases: gate.biases,
                            upWeight: up.w, upScales: up.scales, upBiases: up.biases,
                            indices: indices,
                            mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                            nExperts: nExperts, on: cmd, intoGate: gOut, intoUp: uOut)
                    } else {
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: gate.w, scales: gate.scales, biases: gate.biases,
                            indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: gOut, nExperts: nExperts)
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: up.w, scales: up.scales, biases: up.biases,
                            indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: uOut, nExperts: nExperts)
                    }
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000.0
        }

        let ms2 = timeMs(fused: false)
        let msF = timeMs(fused: true)
        let speedup = ms2 / max(msF, 1e-9)
        let line = String(
            format: "GU 2×EG: %.2f ms  fused: %.2f ms  speedup: %.2fx  (M=%d N=%d K=%d)",
            ms2, msF, speedup, mTotal, nOut, kIn)
        print(line)
        let scratch = ProcessInfo.processInfo.environment["FFAI_TEST_SCRATCH"]
            ?? "/var/folders/y2/mjncn_qn211bjbdr94zpzps40000gn/T/grok-goal-3df794e08dd7/implementer"
        try? (line + "\n").write(
            toFile: scratch + "/hy3-fused-gu-microbench.log",
            atomically: true, encoding: .utf8)
        // Don't fail on speedup under battery; fail only if fused is catastrophically broken.
        #expect(msF < ms2 * 2.5, "fused GU >2.5× slower: \(line)")
    }
}
