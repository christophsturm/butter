// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Fused GU+SwiGLU EG gather vs separate gate EG + up EG + swiglu.

import Foundation
import Metal
import MetalTileSwift
import TestHelpers
import Testing

@testable import FFAI

@Suite("MoE int2 GU+SwiGLU EG")
struct MoEInt2GUSwigluTests {
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
        s.copyIn(from: sh)
        b.copyIn(from: bh)
        return (w, s, b)
    }

    @Test("GU+SwiGLU matches separate gate+up+swiglu")
    func guSwigluMatchesSeparate() throws {
        try #require(Self.mppAvailable)
        let mTotal = 128
        let nOut = 128
        let kIn = 128
        let nExperts = 8
        let groupSize = 64

        setenv("FFAI_HY3_EXPERT_GRID", "1", 1)
        defer { unsetenv("FFAI_HY3_EXPERT_GRID") }

        var idx = [UInt32](repeating: 0, count: mTotal)
        let rowsPer = mTotal / nExperts
        for e in 0 ..< nExperts {
            for r in (e * rowsPer) ..< ((e + 1) * rowsPer) { idx[r] = UInt32(e) }
        }
        let indices = Tensor.empty(shape: [mTotal], dtype: .u32)
        indices.copyIn(from: idx)
        let x = Tensor.empty(shape: [mTotal, kIn], dtype: .f16)
        x.copyIn(from: (0 ..< mTotal * kIn).map { Float16(Float($0 % 13) * 0.01 - 0.06) })
        let gate = Self.makeW(
            nExperts: nExperts, nOut: nOut, kIn: kIn, groupSize: groupSize, seed: 2)
        let up = Self.makeW(
            nExperts: nExperts, nOut: nOut, kIn: kIn, groupSize: groupSize, seed: 9)

        let gOut = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        let uOut = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        let innerSep = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        gOut.zero(); uOut.zero(); innerSep.zero()
        Self.runAndWait { cmd in
            Ops.moeGatherDequantGemmInt2(
                input: x, weight: gate.w, scales: gate.s, biases: gate.b,
                indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                groupSize: groupSize, on: cmd, into: gOut, nExperts: nExperts)
            Ops.moeGatherDequantGemmInt2(
                input: x, weight: up.w, scales: up.s, biases: up.b,
                indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                groupSize: groupSize, on: cmd, into: uOut, nExperts: nExperts)
            _ = Ops.swiglu(gate: gOut, up: uOut, on: cmd, into: innerSep)
        }

        let innerF = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        innerF.zero()
        Self.runAndWait { cmd in
            Ops.moeGatherDequantGemmInt2GUSwiglu(
                input: x,
                gateWeight: gate.w, gateScales: gate.s, gateBiases: gate.b,
                upWeight: up.w, upScales: up.s, upBiases: up.b,
                indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                nExperts: nExperts, on: cmd, intoInner: innerF)
        }

        let ref = innerSep.toArray(as: Float16.self).map { Float($0) }
        let got = innerF.toArray(as: Float16.self).map { Float($0) }
        var nBad = 0
        var maxAbs: Float = 0
        for i in 0 ..< got.count {
            #expect(got[i].isFinite)
            let d = abs(got[i] - ref[i])
            if d > maxAbs { maxAbs = d }
            // f16 accum + silu: allow modest tol
            if d > max(2e-2, 0.05 * abs(ref[i])) { nBad += 1 }
        }
        #expect(nBad == 0, "GU+SwiGLU mismatch \(nBad)/\(got.count) maxAbs=\(maxAbs)")
    }

    @Test("microbench GU+SwiGLU vs 2xEG+swiglu Hy3 shape")
    func microbenchHy3() throws {
        try #require(Self.mppAvailable)
        let mTotal = 2048
        let nOut = 1536
        let kIn = 4096
        let nExperts = 192
        let nActive = 90
        let groupSize = 64
        let warmup = 2
        let iters = 6

        setenv("FFAI_HY3_EXPERT_GRID", "1", 1)
        defer { unsetenv("FFAI_HY3_EXPERT_GRID") }

        var idx = [UInt32](repeating: 0, count: mTotal)
        let rowsPer = max(mTotal / nActive, 1)
        for a in 0 ..< nActive {
            let e = UInt32(a * nExperts / nActive)
            let lo = a * rowsPer
            let hi = min((a + 1) * rowsPer, mTotal)
            if lo < hi { for r in lo ..< hi { idx[r] = e } }
        }
        idx.sort()
        let indices = Tensor.empty(shape: [mTotal], dtype: .u32)
        indices.copyIn(from: idx)
        let x = Tensor.empty(shape: [mTotal, kIn], dtype: .f16)
        x.copyIn(from: (0 ..< mTotal * kIn).map { Float16(Float($0 % 17) * 0.01 - 0.08) })
        let gate = Self.makeW(
            nExperts: nExperts, nOut: nOut, kIn: kIn, groupSize: groupSize, seed: 1)
        let up = Self.makeW(
            nExperts: nExperts, nOut: nOut, kIn: kIn, groupSize: groupSize, seed: 5)

        let gOut = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        let uOut = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)
        let inner = Tensor.empty(shape: [mTotal, nOut], dtype: .f16)

        func timeMs(fused: Bool) -> Double {
            for _ in 0 ..< warmup {
                gOut.zero(); uOut.zero(); inner.zero()
                Self.runAndWait { cmd in
                    if fused {
                        Ops.moeGatherDequantGemmInt2GUSwiglu(
                            input: x,
                            gateWeight: gate.w, gateScales: gate.s, gateBiases: gate.b,
                            upWeight: up.w, upScales: up.s, upBiases: up.b,
                            indices: indices,
                            mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                            nExperts: nExperts, on: cmd, intoInner: inner)
                    } else {
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: gate.w, scales: gate.s, biases: gate.b,
                            indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: gOut, nExperts: nExperts)
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: up.w, scales: up.s, biases: up.b,
                            indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: uOut, nExperts: nExperts)
                        _ = Ops.swiglu(gate: gOut, up: uOut, on: cmd, into: inner)
                    }
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                gOut.zero(); uOut.zero(); inner.zero()
                Self.runAndWait { cmd in
                    if fused {
                        Ops.moeGatherDequantGemmInt2GUSwiglu(
                            input: x,
                            gateWeight: gate.w, gateScales: gate.s, gateBiases: gate.b,
                            upWeight: up.w, upScales: up.s, upBiases: up.b,
                            indices: indices,
                            mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                            nExperts: nExperts, on: cmd, intoInner: inner)
                    } else {
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: gate.w, scales: gate.s, biases: gate.b,
                            indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: gOut, nExperts: nExperts)
                        Ops.moeGatherDequantGemmInt2(
                            input: x, weight: up.w, scales: up.s, biases: up.b,
                            indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                            groupSize: groupSize, on: cmd, into: uOut, nExperts: nExperts)
                        _ = Ops.swiglu(gate: gOut, up: uOut, on: cmd, into: inner)
                    }
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000.0
        }

        let msSep = timeMs(fused: false)
        let msF = timeMs(fused: true)
        let speedup = msSep / max(msF, 1e-9)
        let line = String(
            format: "GU path: 2xEG+swiglu=%.2f ms  GU+SwiGLU=%.2f ms  speedup=%.2fx",
            msSep, msF, speedup)
        print(line)
        let scratch = ProcessInfo.processInfo.environment["FFAI_TEST_SCRATCH"]
            ?? "/var/folders/y2/mjncn_qn211bjbdr94zpzps40000gn/T/grok-goal-3df794e08dd7/implementer"
        try? (line + "\n").write(
            toFile: scratch + "/hy3-gu-swiglu-microbench.log",
            atomically: true, encoding: .utf8)
        #expect(msF.isFinite && msSep.isFinite)
    }
}
