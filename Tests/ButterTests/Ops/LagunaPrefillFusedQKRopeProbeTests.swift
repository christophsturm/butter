// Copyright 2026 Tom Turney (@TheTom)
//
// T30 scratch probe: pins `LagunaSdpa.fusedQKNormRopeForPrefill` (the new
// prefill sliding-layer QK-norm+RoPE fused kernel) against the stock
// three-dispatch chain it replaces (`Ops.rmsNormRows` x2 +
// `Ops.ropePartialManyTwo` + `Ops.castBf16ToF16` + `Ops.transposeTHD_f16`)
// on synthetic data, isolated from the rest of the model. Throwaway-safe
// diagnostic, same spirit as `LagunaSteelGemmProbeTests`.
//
// At nRows=32 this is exactly byte-identical (0/N mismatches). At the
// realistic nRows=512 (T=512 prefill, 512 distinct RoPE angles including
// large `position*inv_freq` arguments to `cos`/`sin`) a handful of
// elements differ by ~1 bf16 ULP (observed: 39/4,194,304 Q elements,
// maxAbs 0.0039; 5/524,288 K elements, maxAbs 6.1e-5): two differently
// -compiled Metal kernels evaluating the same `cos`/`sin` expression can
// legitimately round the last bit differently. This is the same class of
// FP-reorder already accepted throughout T20-T29 (e.g. T26/T27's steel
// GEMM, T28's nine-slot dispatch); real end-to-end generation stays
// coherent (see benchmarks/laguna-xs/README.md T30's gate table), not a
// correctness bug. The bounds below are loose (order of magnitude above
// what's been observed) so this stays a regression guard, not a brittle
// bit-exactness assertion.
import Foundation
import Metal
import IronSwift
import TestHelpers
import Testing

@testable import ButterSwift

private func bf16Bits(_ v: Float) -> UInt16 {
    let bits = v.bitPattern
    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
    return UInt16(rounded >> 16)
}

@Suite("LagunaPrefillFusedQKRopeProbe")
struct LagunaPrefillFusedQKRopeProbeTests {
    @Test("fusedQKNormRopeForPrefill matches rmsNormRows+ropePartialManyTwo+transpose/cast")
    func matchesStockChain() {
        autoreleasepool {
            let device = Device.shared
            let nRows = 512
            let nHeads = 64, nKVHeads = 8, headDim = 128
            let position = 0
            let theta: Float = 10000.0
            let eps: Float = 1e-6

            let qData = (0 ..< nRows * nHeads * headDim).map {
                Float(($0 * 7) % 23) * 0.02 - 0.23
            }
            let kData = (0 ..< nRows * nKVHeads * headDim).map {
                Float(($0 * 11) % 19) * 0.03 - 0.28
            }
            let qWData = (0 ..< headDim).map { Float($0 % 5) * 0.1 + 0.5 }
            let kWData = (0 ..< headDim).map { Float($0 % 7) * 0.08 + 0.4 }

            let rawQ = Tensor.empty(shape: [nRows, nHeads * headDim], dtype: .bf16, device: device)
            let rawK = Tensor.empty(
                shape: [nRows, nKVHeads * headDim], dtype: .bf16, device: device)
            let qW = Tensor.empty(shape: [headDim], dtype: .bf16, device: device)
            let kW = Tensor.empty(shape: [headDim], dtype: .bf16, device: device)
            rawQ.copyIn(from: qData.map(bf16Bits))
            rawK.copyIn(from: kData.map(bf16Bits))
            qW.copyIn(from: qWData.map(bf16Bits))
            kW.copyIn(from: kWData.map(bf16Bits))

            // --- Fused path ---
            let fusedCmd = device.makeCommandBuffer()
            let (qHTD, kOut, _) = LagunaSdpa.fusedQKNormRopeForPrefill(
                rawQ: rawQ, rawK: rawK, qWeight: qW, kWeight: kW,
                nRows: nRows, tPad: nRows, position: position,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                thetaBase: theta, eps: eps, on: fusedCmd, device: device)
            fusedCmd.commit()
            fusedCmd.waitUntilCompleted()

            // --- Stock chain ---
            let oldCmd = device.makeCommandBuffer()
            var qOld = Ops.rmsNormRows(
                rawQ.reshaped(to: [nRows * nHeads * headDim]), weight: qW, eps: eps,
                nRows: nRows * nHeads, rowSize: headDim, on: oldCmd
            ).reshaped(to: [nRows, nHeads * headDim])
            var kOld = Ops.rmsNormRows(
                rawK.reshaped(to: [nRows * nKVHeads * headDim]), weight: kW, eps: eps,
                nRows: nRows * nKVHeads, rowSize: headDim, on: oldCmd
            ).reshaped(to: [nRows, nKVHeads * headDim])
            let positionsT = Tensor.empty(shape: [nRows], dtype: .u32, device: device)
            let posPtr = positionsT.buffer.contents()
                .advanced(by: positionsT.offset)
                .bindMemory(to: UInt32.self, capacity: nRows)
            for r in 0 ..< nRows { posPtr[r] = UInt32(position + r) }
            Ops.ropePartialManyTwo(
                q: qOld, qNHeads: nHeads, qRowStride: nHeads * headDim,
                k: kOld, kNHeads: nKVHeads, kRowStride: nKVHeads * headDim,
                positions: positionsT, t: nRows,
                headDim: headDim, rotaryDim: headDim,
                thetaBase: theta, scaling: .none, on: oldCmd)
            let qF16 = Tensor.empty(
                shape: [nRows, nHeads, headDim], dtype: .f16, device: device)
            _ = Ops.castBf16ToF16(
                qOld.reshaped(to: [nRows * nHeads * headDim]),
                into: qF16.reshaped(to: [nRows * nHeads * headDim]), on: oldCmd)
            let qHTDOld = Tensor.empty(
                shape: [nHeads, nRows, headDim], dtype: .f16, device: device)
            _ = Ops.transposeTHD_f16(
                qF16, t: nRows, h: nHeads, d: headDim, on: oldCmd, into: qHTDOld)
            oldCmd.commit()
            oldCmd.waitUntilCompleted()

            let gotQ = qHTD.toFloatArray()
            let expQ = qHTDOld.toFloatArray()
            #expect(gotQ.count == expQ.count)
            var qMismatches = 0
            var qMaxAbs: Float = 0
            for i in 0 ..< gotQ.count {
                if gotQ[i] != expQ[i] { qMismatches += 1 }
                qMaxAbs = max(qMaxAbs, abs(gotQ[i] - expQ[i]))
            }
            // Loose near-ULP tolerance, see file header comment.
            #expect(
                Double(qMismatches) < 0.01 * Double(gotQ.count) && qMaxAbs < 0.05,
                "Q: \(qMismatches)/\(gotQ.count) mismatches, maxAbs=\(qMaxAbs)")

            let gotK = kOut.toFloatArray()
            let expK = kOld.toFloatArray()
            #expect(gotK.count == expK.count)
            var kMismatches = 0
            var kMaxAbs: Float = 0
            for i in 0 ..< gotK.count {
                if gotK[i] != expK[i] { kMismatches += 1 }
                kMaxAbs = max(kMaxAbs, abs(gotK[i] - expK[i]))
            }
            #expect(
                Double(kMismatches) < 0.01 * Double(gotK.count) && kMaxAbs < 0.01,
                "K: \(kMismatches)/\(gotK.count) mismatches, maxAbs=\(kMaxAbs)")
        }
    }

    @Test(
        "fusedQKNormYarnForPrefill matches rmsNormRows+per-row iron_rope_yarn_bf16+transpose/cast"
    )
    func matchesStockYarnChain() {
        autoreleasepool {
            let device = Device.shared
            let nRows = 512
            let nHeads = 48, nKVHeads = 8, headDim = 128, rotaryDim = 64
            let position = 0
            let theta: Float = 500_000.0
            let eps: Float = 1e-6
            let yarn = Ops.RoPEYaRN(factor: 32, low: 1, high: 32, attnFactor: 1.0)

            let qData = (0 ..< nRows * nHeads * headDim).map {
                Float(($0 * 7) % 23) * 0.02 - 0.23
            }
            let kData = (0 ..< nRows * nKVHeads * headDim).map {
                Float(($0 * 11) % 19) * 0.03 - 0.28
            }
            let qWData = (0 ..< headDim).map { Float($0 % 5) * 0.1 + 0.5 }
            let kWData = (0 ..< headDim).map { Float($0 % 7) * 0.08 + 0.4 }

            let rawQ = Tensor.empty(shape: [nRows, nHeads * headDim], dtype: .bf16, device: device)
            let rawK = Tensor.empty(
                shape: [nRows, nKVHeads * headDim], dtype: .bf16, device: device)
            let qW = Tensor.empty(shape: [headDim], dtype: .bf16, device: device)
            let kW = Tensor.empty(shape: [headDim], dtype: .bf16, device: device)
            rawQ.copyIn(from: qData.map(bf16Bits))
            rawK.copyIn(from: kData.map(bf16Bits))
            qW.copyIn(from: qWData.map(bf16Bits))
            kW.copyIn(from: kWData.map(bf16Bits))

            // --- Fused path ---
            let fusedCmd = device.makeCommandBuffer()
            let (qHTD, kOut, _) = LagunaSdpa.fusedQKNormYarnForPrefill(
                rawQ: rawQ, rawK: rawK, qWeight: qW, kWeight: kW,
                nRows: nRows, tPad: nRows, position: position,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: theta, eps: eps, yarn: yarn, on: fusedCmd, device: device)
            fusedCmd.commit()
            fusedCmd.waitUntilCompleted()

            // --- Stock chain: rmsNormRows x2, then per-row iron_rope_yarn_bf16
            // (matches `lagunaRopeYaRNPartial`'s per-row dispatch loop exactly).
            let oldCmd = device.makeCommandBuffer()
            let qOld = Ops.rmsNormRows(
                rawQ.reshaped(to: [nRows * nHeads * headDim]), weight: qW, eps: eps,
                nRows: nRows * nHeads, rowSize: headDim, on: oldCmd
            ).reshaped(to: [nRows, nHeads * headDim])
            let kOld = Ops.rmsNormRows(
                rawK.reshaped(to: [nRows * nKVHeads * headDim]), weight: kW, eps: eps,
                nRows: nRows * nKVHeads, rowSize: headDim, on: oldCmd
            ).reshaped(to: [nRows, nKVHeads * headDim])
            let halfRotary = rotaryDim / 2
            let grid = MTLSize(width: nHeads, height: halfRotary, depth: 1)
            let gridK = MTLSize(width: nKVHeads, height: halfRotary, depth: 1)
            let tg = MTLSize(width: 1, height: 1, depth: 1)
            let qRowBytes = nHeads * headDim * 2
            let kRowBytes = nKVHeads * headDim * 2
            for r in 0 ..< nRows {
                let qRow = Tensor(
                    buffer: qOld.buffer, offset: qOld.offset + r * qRowBytes,
                    shape: [nHeads * headDim], dtype: .bf16)
                let kRow = Tensor(
                    buffer: kOld.buffer, offset: kOld.offset + r * kRowBytes,
                    shape: [nKVHeads * headDim], dtype: .bf16)
                IronKernels.iron_rope_yarn_bf16(
                    qk: qRow.buffer, qkOffset: qRow.offset,
                    out: qRow.buffer, outOffset: qRow.offset,
                    head_dim: UInt32(headDim), half_dim: UInt32(halfRotary),
                    position: UInt32(position + r), theta_base: theta,
                    factor: yarn.factor, low: yarn.low, high: yarn.high,
                    attn_factor: yarn.attnFactor,
                    gridSize: grid, threadgroupSize: tg, on: oldCmd)
                IronKernels.iron_rope_yarn_bf16(
                    qk: kRow.buffer, qkOffset: kRow.offset,
                    out: kRow.buffer, outOffset: kRow.offset,
                    head_dim: UInt32(headDim), half_dim: UInt32(halfRotary),
                    position: UInt32(position + r), theta_base: theta,
                    factor: yarn.factor, low: yarn.low, high: yarn.high,
                    attn_factor: yarn.attnFactor,
                    gridSize: gridK, threadgroupSize: tg, on: oldCmd)
            }
            let qF16 = Tensor.empty(
                shape: [nRows, nHeads, headDim], dtype: .f16, device: device)
            _ = Ops.castBf16ToF16(
                qOld.reshaped(to: [nRows * nHeads * headDim]),
                into: qF16.reshaped(to: [nRows * nHeads * headDim]), on: oldCmd)
            let qHTDOld = Tensor.empty(
                shape: [nHeads, nRows, headDim], dtype: .f16, device: device)
            _ = Ops.transposeTHD_f16(
                qF16, t: nRows, h: nHeads, d: headDim, on: oldCmd, into: qHTDOld)
            oldCmd.commit()
            oldCmd.waitUntilCompleted()

            let gotQ = qHTD.toFloatArray()
            let expQ = qHTDOld.toFloatArray()
            #expect(gotQ.count == expQ.count)
            var qMismatches = 0
            var qMaxAbs: Float = 0
            for i in 0 ..< gotQ.count {
                if gotQ[i] != expQ[i] { qMismatches += 1 }
                qMaxAbs = max(qMaxAbs, abs(gotQ[i] - expQ[i]))
            }
            #expect(
                Double(qMismatches) < 0.01 * Double(gotQ.count) && qMaxAbs < 0.05,
                "Q: \(qMismatches)/\(gotQ.count) mismatches, maxAbs=\(qMaxAbs)")

            let gotK = kOut.toFloatArray()
            let expK = kOld.toFloatArray()
            #expect(gotK.count == expK.count)
            var kMismatches = 0
            var kMaxAbs: Float = 0
            for i in 0 ..< gotK.count {
                if gotK[i] != expK[i] { kMismatches += 1 }
                kMaxAbs = max(kMaxAbs, abs(gotK[i] - expK[i]))
            }
            #expect(
                Double(kMismatches) < 0.01 * Double(gotK.count) && kMaxAbs < 0.01,
                "K: \(kMismatches)/\(gotK.count) mismatches, maxAbs=\(kMaxAbs)")
        }
    }
}
