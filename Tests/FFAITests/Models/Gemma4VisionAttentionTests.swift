// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Numerically validates the Gemma 4 vision tower's GPU attention pipeline
// (per-head RMSNorm → 2D vision RoPE → token↔head-major transpose →
// bidirectional SDPA) against a from-scratch CPU reference — the exact
// algorithm the tower used before the CPU→GPU migration. This is the
// de-risk gate for replacing the `concurrentPerform` attention with the
// `Ops.rmsNormRows` + `Ops.rope2D` + `Ops.transposeTokensToHeads` +
// `Ops.sdpaBidirectional` composition.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import FFAI

@Suite("Gemma4 Vision Attention (GPU vs CPU)")
struct Gemma4VisionAttentionTests {

    /// CPU reference: the pre-migration algorithm (per-head weighted
    /// RMSNorm on q/k, unweighted on v; multi-dimensional vision RoPE on
    /// q/k; full bidirectional GQA attention, scale 1.0).
    private func cpuReference(
        q: [Float], k: [Float], v: [Float],
        qWeight: [Float], kWeight: [Float],
        nTokens: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        xPos: [Int], yPos: [Int], ropeTheta: Float, eps: Float
    ) -> [Float] {
        let qStride = nHeads * headDim
        let kvStride = nKVHeads * headDim
        let numDims = 2
        let chPerDim = 2 * (headDim / (2 * numDims))
        let halfPerDim = chPerDim / 2

        func rmsNorm(_ x: inout [Float], weight: [Float]?) {
            var ss: Float = 0
            for d in 0 ..< headDim { ss += x[d] * x[d] }
            let inv = 1.0 / (ss / Float(headDim) + eps).squareRoot()
            for d in 0 ..< headDim {
                x[d] = x[d] * inv * (weight != nil ? weight![d] : 1.0)
            }
        }
        func rope(_ x: inout [Float], xp: Int, yp: Int) {
            for axis in 0 ..< numDims {
                let start = axis * chPerDim
                let pos = Float(axis == 0 ? xp : yp)
                for i in 0 ..< halfPerDim {
                    let exponent = (2.0 / Float(chPerDim)) * Float(i)
                    let timescale = pow(ropeTheta, exponent)
                    let theta = pos / timescale
                    let c = cos(theta)
                    let s = sin(theta)
                    let a = x[start + i]
                    let b = x[start + halfPerDim + i]
                    x[start + i] = a * c - b * s
                    x[start + halfPerDim + i] = b * c + a * s
                }
            }
        }

        var qH = [[Float]](repeating: [], count: nTokens * nHeads)
        var kH = [[Float]](repeating: [], count: nTokens * nKVHeads)
        var vH = [[Float]](repeating: [], count: nTokens * nKVHeads)
        for t in 0 ..< nTokens {
            for h in 0 ..< nHeads {
                var x = Array(
                    q[(t * qStride + h * headDim) ..< (t * qStride + (h + 1) * headDim)])
                rmsNorm(&x, weight: qWeight)
                rope(&x, xp: xPos[t], yp: yPos[t])
                qH[t * nHeads + h] = x
            }
            for h in 0 ..< nKVHeads {
                var xk = Array(
                    k[(t * kvStride + h * headDim) ..< (t * kvStride + (h + 1) * headDim)])
                rmsNorm(&xk, weight: kWeight)
                rope(&xk, xp: xPos[t], yp: yPos[t])
                kH[t * nKVHeads + h] = xk
                var xv = Array(
                    v[(t * kvStride + h * headDim) ..< (t * kvStride + (h + 1) * headDim)])
                rmsNorm(&xv, weight: nil)
                vH[t * nKVHeads + h] = xv
            }
        }

        let headsPerGroup = nHeads / nKVHeads
        let H = nHeads * headDim
        var out = [Float](repeating: 0, count: nTokens * H)
        for head in 0 ..< nHeads {
            let kvHead = head / headsPerGroup
            for i in 0 ..< nTokens {
                let qRow = qH[i * nHeads + head]
                var scores = [Float](repeating: 0, count: nTokens)
                var maxScore = -Float.greatestFiniteMagnitude
                for j in 0 ..< nTokens {
                    let kRow = kH[j * nKVHeads + kvHead]
                    var dot: Float = 0
                    for d in 0 ..< headDim { dot += qRow[d] * kRow[d] }
                    scores[j] = dot  // scale 1.0
                    if dot > maxScore { maxScore = dot }
                }
                var sumExp: Float = 0
                for j in 0 ..< nTokens {
                    let e = exp(scores[j] - maxScore)
                    scores[j] = e
                    sumExp += e
                }
                let invSum = 1.0 / sumExp
                let outBase = i * H + head * headDim
                for j in 0 ..< nTokens {
                    let w = scores[j] * invSum
                    let vRow = vH[j * nKVHeads + kvHead]
                    for d in 0 ..< headDim { out[outBase + d] += w * vRow[d] }
                }
            }
        }
        return out
    }

    @Test("GPU pipeline matches CPU reference (GQA, headDim 64)")
    func gpuMatchesCPU() {
        autoreleasepool {
            // GQA config mirroring the Gemma 4 vision tower (16:16 here,
            // but exercise a fan-out too via nKVHeads < nHeads).
            let nTokens = 24, nHeads = 8, nKVHeads = 4, headDim = 64
            let ropeTheta: Float = 100.0
            let eps: Float = 1e-6
            let qCount = nTokens * nHeads * headDim
            let kvCount = nTokens * nKVHeads * headDim

            func ramp(_ n: Int, _ seed: Int) -> [Float] {
                (0 ..< n).map { Float(($0 + seed) % 37 - 18) * 0.05 }
            }
            let q = ramp(qCount, 3)
            let k = ramp(kvCount, 7)
            let v = ramp(kvCount, 11)
            let qWeight = ramp(headDim, 1).map { $0 + 1.0 }
            let kWeight = ramp(headDim, 2).map { $0 + 1.0 }
            let xPos = (0 ..< nTokens).map { $0 % 6 }
            let yPos = (0 ..< nTokens).map { $0 / 6 }

            let expected = cpuReference(
                q: q, k: k, v: v, qWeight: qWeight, kWeight: kWeight,
                nTokens: nTokens, nHeads: nHeads, nKVHeads: nKVHeads,
                headDim: headDim, xPos: xPos, yPos: yPos,
                ropeTheta: ropeTheta, eps: eps)

            // GPU pipeline — the exact ops the vision tower now uses.
            let qT = Tensor.empty(shape: [nTokens, nHeads * headDim], dtype: .f32)
            qT.copyIn(from: q)
            let kT = Tensor.empty(shape: [nTokens, nKVHeads * headDim], dtype: .f32)
            kT.copyIn(from: k)
            let vT = Tensor.empty(shape: [nTokens, nKVHeads * headDim], dtype: .f32)
            vT.copyIn(from: v)
            let qW = Tensor.empty(shape: [headDim], dtype: .f32)
            qW.copyIn(from: qWeight)
            let kW = Tensor.empty(shape: [headDim], dtype: .f32)
            kW.copyIn(from: kWeight)
            let ones = Tensor.filled(1.0, shape: [headDim], dtype: .f32)
            var posFlat = [UInt32](repeating: 0, count: nTokens * 2)
            for t in 0 ..< nTokens {
                posFlat[t * 2] = UInt32(xPos[t])
                posFlat[t * 2 + 1] = UInt32(yPos[t])
            }
            let positions = Tensor.empty(shape: [nTokens * 2], dtype: .u32)
            positions.copyIn(from: posFlat)

            var outT: Tensor!
            var qnT: Tensor!
            var qrT: Tensor!
            runAndWait { cmd in
                let qn = Ops.rmsNormRows(
                    qT, weight: qW, eps: eps,
                    nRows: nTokens * nHeads, rowSize: headDim, on: cmd)
                let kn = Ops.rmsNormRows(
                    kT, weight: kW, eps: eps,
                    nRows: nTokens * nKVHeads, rowSize: headDim, on: cmd)
                let vn = Ops.rmsNormRows(
                    vT, weight: ones, eps: eps,
                    nRows: nTokens * nKVHeads, rowSize: headDim, on: cmd)
                let qr = Ops.rope2D(
                    qn, positions: positions, nTokens: nTokens,
                    nHeads: nHeads, headDim: headDim, thetaBase: ropeTheta, on: cmd)
                let kr = Ops.rope2D(
                    kn, positions: positions, nTokens: nTokens,
                    nHeads: nKVHeads, headDim: headDim, thetaBase: ropeTheta, on: cmd)
                let kHead = Ops.transposeTokensToHeads(
                    kr, nTokens: nTokens, nHeads: nKVHeads, headDim: headDim, on: cmd)
                let vHead = Ops.transposeTokensToHeads(
                    vn, nTokens: nTokens, nHeads: nKVHeads, headDim: headDim, on: cmd)
                qnT = qn
                qrT = qr
                outT = Ops.sdpaBidirectional(
                    q: qr, k: kHead, v: vHead,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    baseKV: 0, nQuery: nTokens, kvStride: nTokens,
                    scale: 1.0, on: cmd)
            }
            // Read back AFTER the command buffer has completed.
            let got = outT.toArray(as: Float.self)
            _ = qnT
            _ = qrT

            #expect(got.count == expected.count)
            var maxDiff: Float = 0
            for i in 0 ..< min(got.count, expected.count) {
                maxDiff = max(maxDiff, abs(got[i] - expected[i]))
            }
            #expect(
                maxDiff < 1e-3,
                "GPU vision attention diverges from CPU reference (max |Δ| \(maxDiff))")
        }
    }
}
