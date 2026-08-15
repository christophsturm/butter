// Copyright 2026 Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// T41 kernel-level correctness for `LagunaAttnNvfp4Tail` (group-16 NVFP4
// attention tail for late layers): quantize->dequant round-trip fidelity
// vs a CPU reference decoder, the batched QMV vs a CPU dot-product
// reference over the same quantized bytes, a magnitude sweep (finite +
// accurate at real late-layer activation scale, same class of check T23
// added for the MPP gather family, see benchmarks/laguna-xs/README.md
// T23/T41), and the gate-folded o_proj QMV vs a manual softplus-gate +
// plain-QMV reference.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("Laguna attention NVFP4 tail (T41)", .serialized)
struct LagunaAttnNvfp4TailTests {
    // MARK: - Test data generation

    /// Uniform `[-scale, scale)` BF16 fill, xorshift64 deterministic (no
    /// external dependency), same generator shape as
    /// `LagunaSdpaDecodeGqa2Tests.randomBf16`.
    private func randomUniformBf16(_ shape: [Int], scale: Float = 1.0, seed: inout UInt64) -> Tensor {
        let n = shape.reduce(1, *)
        var f32 = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            let u = Double(seed % 2_000_001) / 1_000_000.0 - 1.0  // [-1, 1)
            f32[i] = Float(u) * scale
        }
        let f32T = Tensor.empty(shape: shape, dtype: .f32)
        f32T.copyIn(from: f32)
        let bf16T = Tensor.empty(shape: shape, dtype: .bf16)
        runAndWait { cb in Ops.castF32ToBf16(f32T, into: bf16T, on: cb) }
        return bf16T
    }

    /// Approximately-normal `std`-scaled BF16 fill (Box-Muller over the
    /// same xorshift64 stream), closer to real trained-weight shape than
    /// uniform noise, used for the round-trip fidelity gate so the
    /// measured cosine reflects the case this lever actually targets.
    private func randomGaussianBf16(_ shape: [Int], std: Float = 1.0, seed: inout UInt64) -> Tensor {
        let n = shape.reduce(1, *)
        var f32 = [Float](repeating: 0, count: n)
        var i = 0
        while i < n {
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            let u1 = max(Double(seed % 1_000_000) / 1_000_000.0, 1e-9)
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            let u2 = Double(seed % 1_000_000) / 1_000_000.0
            let r = (-2.0 * log(u1)).squareRoot()
            let theta = 2.0 * Double.pi * u2
            f32[i] = Float(r * cos(theta)) * std
            i += 1
            if i < n {
                f32[i] = Float(r * sin(theta)) * std
                i += 1
            }
        }
        let f32T = Tensor.empty(shape: shape, dtype: .f32)
        f32T.copyIn(from: f32)
        let bf16T = Tensor.empty(shape: shape, dtype: .bf16)
        runAndWait { cb in Ops.castF32ToBf16(f32T, into: bf16T, on: cb) }
        return bf16T
    }

    // MARK: - CPU reference decode (bit-exact vs the Metal helper functions
    // in `LagunaAttnNvfp4Tail.helperSrc`)

    private func e4m3Decode(_ byte: UInt8) -> Float {
        let exp4 = (UInt32(byte) >> 3) & 0xF
        let mant3 = Float(byte & 0x7)
        let mag: Float
        if exp4 == 0 {
            mag = mant3 * 0.001953125
        } else {
            mag = (1.0 + mant3 * 0.125) * powf(2.0, Float(exp4) - 7.0)
        }
        return (byte >> 7) & 1 == 1 ? -mag : mag
    }

    private static let e2m1Lut: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    private func e2m1Decode(_ nib: UInt8) -> Float {
        let mag = Self.e2m1Lut[Int(nib & 0x7)]
        return (nib >> 3) & 1 == 1 ? -mag : mag
    }

    /// CPU dequant of a `LagunaAttnNvfp4Tail.Bank` to `[outDim][inDim]`
    /// `Float`, reading the packed/scale bytes straight off the GPU
    /// buffers (storage-shared, no extra sync needed once the quantize
    /// command buffer has completed).
    private func dequantBank(_ bank: LagunaAttnNvfp4Tail.Bank) -> [[Float]] {
        let packed = bank.packed.toArray(as: UInt32.self)
        let scales = bank.scales.toArray(as: UInt8.self)
        let packsPerRow = bank.inDim / 8
        let groupsPerRow = bank.inDim / LagunaAttnNvfp4Tail.groupSize
        var rows = [[Float]](
            repeating: [Float](repeating: 0, count: bank.inDim), count: bank.outDim)
        for row in 0 ..< bank.outDim {
            for pack in 0 ..< packsPerRow {
                let p = packed[row * packsPerRow + pack]
                let g = (pack * 8) / LagunaAttnNvfp4Tail.groupSize
                let scale = e4m3Decode(scales[row * groupsPerRow + g])
                for i in 0 ..< 8 {
                    let nib = UInt8((p >> (4 * UInt32(i))) & 0xF)
                    rows[row][pack * 8 + i] = e2m1Decode(nib) * scale
                }
            }
        }
        return rows
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    }

    // MARK: - Tests

    /// Quantize->dequant round trip vs the original BF16 source. Gaussian
    /// (`std=1.0`) weight shape [64, 256] (16 groups/row, 1024 groups
    /// total), large enough that the measured cosine is a stable
    /// estimate, not a single-draw fluke. `0.98` is the calibrated floor
    /// for group-16 4-bit NVFP4 (E2M1 8-magnitude-class alphabet + a
    /// per-16-element E4M3 scale), coarser than the group-32 affine INT8
    /// side layout this replaces for late layers, by design (T20's own
    /// rationale for leaving NVFP4 unimplemented was exactly this
    /// precision/complexity tradeoff).
    @Test("quantize round-trip cosine floor at group-16 4-bit")
    func quantizeRoundTripCosineFloor() {
        let device = Device.shared
        let outDim = 64
        let inDim = 256
        var seed: UInt64 = 0x1234_5678_9ABC_DEF0
        let weight = randomGaussianBf16([outDim, inDim], std: 1.0, seed: &seed)

        var bank: LagunaAttnNvfp4Tail.Bank!
        runAndWait { cb in
            bank = LagunaAttnNvfp4Tail.quantizeG16(weight: weight, device: device, cmd: cb)
        }
        #expect(bank != nil, "quantizeG16 declined on a shape it should accept")
        guard let bank else { return }

        let original = weight.toFloatArray()
        let dequant = dequantBank(bank).flatMap { $0 }
        let cos = cosine(original, dequant)
        print("[nvfp4-tail-roundtrip] cos=\(cos)")
        #expect(cos > 0.98, "group-16 NVFP4 round-trip cos \(cos) below calibrated floor 0.98")
    }

    /// Batched QMV (`LagunaAttnNvfp4Tail.qmv`) vs a CPU dot-product
    /// reference computed from the SAME dequantized bytes (isolates the
    /// GPU dispatch/reduction from quantization error, this is a kernel
    /// correctness check, not a quantization-fidelity check). `outDim=24`
    /// is not a multiple of 8 (exercises the row-bounds check in the
    /// "ceil(outDim/8)" tiling, same discipline `LagunaAttnInt8.qmvSrc`'s
    /// own tests would want).
    @Test("batched QMV vs CPU reference over the quantized bytes")
    func qmvVsCpuReference() {
        let device = Device.shared
        let outDim = 24
        let inDim = 128
        var seed: UInt64 = 0xC0FF_EE00_1234_5678
        let weight = randomUniformBf16([outDim, inDim], scale: 1.5, seed: &seed)
        let input = randomUniformBf16([inDim], scale: 1.0, seed: &seed)

        var bank: LagunaAttnNvfp4Tail.Bank!
        runAndWait { cb in
            bank = LagunaAttnNvfp4Tail.quantizeG16(weight: weight, device: device, cmd: cb)
        }
        #expect(bank != nil)
        guard let bank else { return }

        let out = Tensor.empty(shape: [outDim], dtype: .bf16, device: device)
        runAndWait { cb in
            LagunaAttnNvfp4Tail.qmv(input: input, bank: bank, on: cb, device: device, into: out)
        }

        let dequantRows = dequantBank(bank)
        let xF = input.toFloatArray()
        var refOut = [Float](repeating: 0, count: outDim)
        for row in 0 ..< outDim {
            var acc: Float = 0
            for i in 0 ..< inDim { acc += dequantRows[row][i] * xF[i] }
            refOut[row] = acc
        }

        let gpuOut = out.toFloatArray()
        let cos = cosine(gpuOut, refOut)
        print("[nvfp4-tail-qmv] cos=\(cos) sample gpu=\(gpuOut.prefix(4)) ref=\(refOut.prefix(4))")
        #expect(cos > 0.9999, "QMV vs CPU-reference cos \(cos) below 0.9999")
    }

    /// Magnitude sweep, same class of regression T23 added for the MPP
    /// gather family after its half-staging overflow bug (see
    /// benchmarks/laguna-xs/README.md T23). This kernel family never
    /// stages into `half` (accumulates in `float` end to end, see the
    /// design note atop `LagunaAttnNvfp4Tail.swift`), so this is a
    /// permanent confirmation of that property rather than a fix
    /// verification: `nan==0`, `inf==0`, and `cos>0.999` at every swept
    /// magnitude, including real late-layer (SwiGLU-amplified) scale.
    @Test("magnitude sweep stays finite and accurate")
    func magnitudeSweepStaysFinite() {
        let device = Device.shared
        let outDim = 16
        let inDim = 128
        let magnitudes: [Float] = [1, 10, 100, 1000, 10000]
        for magnitude in magnitudes {
            var seed: UInt64 = 0x9E37_79B9_0000_0001 &+ UInt64(magnitude)
            let weight = randomGaussianBf16([outDim, inDim], std: magnitude, seed: &seed)
            let input = randomGaussianBf16([inDim], std: magnitude, seed: &seed)

            var bank: LagunaAttnNvfp4Tail.Bank!
            runAndWait { cb in
                bank = LagunaAttnNvfp4Tail.quantizeG16(weight: weight, device: device, cmd: cb)
            }
            #expect(bank != nil)
            guard let bank else { continue }

            let out = Tensor.empty(shape: [outDim], dtype: .bf16, device: device)
            runAndWait { cb in
                LagunaAttnNvfp4Tail.qmv(input: input, bank: bank, on: cb, device: device, into: out)
            }

            let dequantRows = dequantBank(bank)
            let xF = input.toFloatArray()
            var refOut = [Float](repeating: 0, count: outDim)
            for row in 0 ..< outDim {
                var acc: Float = 0
                for i in 0 ..< inDim { acc += dequantRows[row][i] * xF[i] }
                refOut[row] = acc
            }

            let gpuOut = out.toFloatArray()
            let nanCount = gpuOut.filter { $0.isNaN }.count
            let infCount = gpuOut.filter { $0.isInfinite }.count
            let cos = cosine(gpuOut, refOut)
            print("[nvfp4-tail-magnitude] mag=\(magnitude) nan=\(nanCount) inf=\(infCount) cos=\(cos)")
            #expect(nanCount == 0, "magnitude \(magnitude): \(nanCount) NaNs")
            #expect(infCount == 0, "magnitude \(magnitude): \(infCount) Infs")
            #expect(cos > 0.999, "magnitude \(magnitude): cos \(cos) below floor")
        }
    }

    /// Gate-folded o_proj QMV (`LagunaAttnNvfp4Tail.qmvGated`) vs a manual
    /// softplus-gate-then-plain-QMV reference, the exact fusion decode
    /// takes for the o_proj bank when both `LagunaAttnInt8.gatedEnabled`
    /// and the NVFP4 tail apply to a layer (`LagunaText.forward`).
    @Test("gated o_proj QMV matches manual softplus-gate + plain QMV")
    func qmvGatedMatchesManualPath() {
        let device = Device.shared
        let heads = 4
        let headDim = 32  // multiple of 8
        let inDim = heads * headDim
        let outDim = 40
        var seed: UInt64 = 0xABCD_EF01_2345_6789
        let weight = randomUniformBf16([outDim, inDim], scale: 1.2, seed: &seed)
        let attnOut = randomUniformBf16([inDim], scale: 1.0, seed: &seed)
        let gateLogits = randomUniformBf16([heads], scale: 2.0, seed: &seed)

        var bank: LagunaAttnNvfp4Tail.Bank!
        runAndWait { cb in
            bank = LagunaAttnNvfp4Tail.quantizeG16(weight: weight, device: device, cmd: cb)
        }
        #expect(bank != nil)
        guard let bank else { return }

        setenv("BUTTER_LAGUNA_INT8_OPROJ_GATED", "1", 1)
        let gatedOut = Tensor.empty(shape: [outDim], dtype: .bf16, device: device)
        var gatedDone = false
        runAndWait { cb in
            gatedDone = LagunaAttnNvfp4Tail.qmvGated(
                input: attnOut, gateLogits: gateLogits, bank: bank,
                heads: heads, headDim: headDim,
                on: cb, device: device, into: gatedOut)
        }
        #expect(gatedDone, "qmvGated declined on a shape it should accept")

        // Manual reference: softplus-gate the input per head, then a plain
        // CPU dot product over the same dequantized bytes.
        let rawX = attnOut.toFloatArray()
        let gl = gateLogits.toFloatArray()
        var gatedX = [Float](repeating: 0, count: inDim)
        for h in 0 ..< heads {
            let g = gl[h]
            let softplus = max(g, 0) + log(1 + exp(-abs(g)))
            for d in 0 ..< headDim {
                gatedX[h * headDim + d] = rawX[h * headDim + d] * softplus
            }
        }
        let dequantRows = dequantBank(bank)
        var refOut = [Float](repeating: 0, count: outDim)
        for row in 0 ..< outDim {
            var acc: Float = 0
            for i in 0 ..< inDim { acc += dequantRows[row][i] * gatedX[i] }
            refOut[row] = acc
        }

        let gpuOut = gatedOut.toFloatArray()
        let cos = cosine(gpuOut, refOut)
        print("[nvfp4-tail-qmvgated] cos=\(cos)")
        #expect(cos > 0.9999, "gated QMV vs manual-gate reference cos \(cos) below 0.9999")
    }
}
