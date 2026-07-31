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
// Correctness tests for `LagunaAttnInt8.qmvGatedV2` (the reworked gated
// o_proj INT8 QMV: threadgroup-precomputed per-head softplus gate table,
// values_per_thread=8 / block_size=256 stride, scalar uint8_t weight
// reads, gate applied to the activation at the same fp32->bf16 rounding
// point the unfused gate-product kernel uses). `qmvGatedV1` is the
// kernel this file shipped with before this change (unmodified body,
// just moved into its own callable function so tests can dispatch it
// directly), kept as the `BUTTER_LAGUNA_QMV_V2=0` escape hatch.
//
// Every synthetic bank here is built by hand (random int8 codes packed
// 4/word + random bf16 scale/bias), bypassing `LagunaAttnInt8.quantizeG32`
// entirely, so these tests exercise the QMV kernels in isolation from the
// requant path. Shapes mirror the real Laguna XS o_proj bank (see
// `Tests/ButterTests/Fixtures/laguna-xs-2.1-config.json`):
// `num_attention_heads=48`, `head_dim=128` -> bank `inDim=6144`;
// `hidden_size=2048` -> bank `outDim=2048`.
//
// Both kernels are compared either against an independent Double-
// precision CPU reference (loose cosine + abs-diff tolerance, same
// convention `LagunaSdpaDecodeGqa2Tests` uses for a different fused
// kernel) or, for the no-gate case, directly against each other under
// the same tolerance. See `noGateMatchesV1` below for why that case
// uses deliberately small shapes rather than the full Laguna XS bank.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("LagunaOProjQmvV2")
struct LagunaOProjQmvV2Tests {

    // MARK: - synthetic data helpers

    /// Deterministic xorshift64 PRNG, same construction
    /// `LagunaSdpaDecodeGqa2Tests` uses: no external dependency, fully
    /// reproducible across runs.
    private func nextUniform(_ seed: inout UInt64) -> Double {
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        return Double(seed % 2_000_001) / 1_000_000.0 - 1.0  // [-1, 1)
    }

    private func randomBf16(_ shape: [Int], scale: Float, seed: inout UInt64) -> Tensor {
        let n = shape.reduce(1, *)
        var f32 = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            f32[i] = Float(nextUniform(&seed)) * scale
        }
        let f32T = Tensor.empty(shape: shape, dtype: .f32)
        f32T.copyIn(from: f32)
        let bf16T = Tensor.empty(shape: shape, dtype: .bf16)
        runAndWait { cb in Ops.castF32ToBf16(f32T, into: bf16T, on: cb) }
        return bf16T
    }

    /// Build a group-32 affine INT8 bank directly from CPU-side codes /
    /// scales / biases, bypassing `LagunaAttnInt8.quantizeG32` so the
    /// test controls the exact quantized values feeding the kernel.
    /// `codes` is `[outDim][inDim]` in `0...255`; `scalesF`/`biasesF`
    /// are `[outDim][inDim/32]`. Scale/bias are round-tripped through a
    /// GPU f32->bf16 cast (not hand-rounded) so the returned Bank holds
    /// exactly what the kernel will read. Callers that need a CPU
    /// reference should read the bf16-rounded values back out via
    /// `.toFloatArray()` rather than reusing `scalesF`/`biasesF`.
    private func makeBank(
        outDim: Int, inDim: Int,
        codes: [UInt8], scalesF: [Float], biasesF: [Float]
    ) -> LagunaAttnInt8.Bank {
        precondition(inDim % 32 == 0, "group size 32 must divide inDim")
        let packsPerRow = inDim / 4
        let groupsPerRow = inDim / 32
        precondition(codes.count == outDim * inDim)
        precondition(scalesF.count == outDim * groupsPerRow)
        precondition(biasesF.count == outDim * groupsPerRow)

        var packedWords = [UInt32](repeating: 0, count: outDim * packsPerRow)
        for r in 0 ..< outDim {
            for p in 0 ..< packsPerRow {
                var word: UInt32 = 0
                for i in 0 ..< 4 {
                    let c = UInt32(codes[r * inDim + p * 4 + i])
                    word |= c << (8 * UInt32(i))
                }
                packedWords[r * packsPerRow + p] = word
            }
        }
        let packed = Tensor.empty(shape: [outDim, packsPerRow], dtype: .u32)
        packed.copyIn(from: packedWords)

        let scalesF32 = Tensor.empty(shape: [outDim, groupsPerRow], dtype: .f32)
        scalesF32.copyIn(from: scalesF)
        let scalesBf16 = Tensor.empty(shape: [outDim, groupsPerRow], dtype: .bf16)
        runAndWait { cb in Ops.castF32ToBf16(scalesF32, into: scalesBf16, on: cb) }

        let biasesF32 = Tensor.empty(shape: [outDim, groupsPerRow], dtype: .f32)
        biasesF32.copyIn(from: biasesF)
        let biasesBf16 = Tensor.empty(shape: [outDim, groupsPerRow], dtype: .bf16)
        runAndWait { cb in Ops.castF32ToBf16(biasesF32, into: biasesBf16, on: cb) }

        return LagunaAttnInt8.Bank(
            packed: packed, scales: scalesBf16, biases: biasesBf16, outDim: outDim, inDim: inDim)
    }

    /// bf16 round-trip via the exact round-to-nearest-even conversion
    /// `Gemma3Text`'s cast helpers use, matching the GPU `bfloat(...)`
    /// cast the kernels round through.
    private func bf16Round(_ x: Float) -> Float {
        bf16BitsToFloatForTest(floatToBf16BitsForTest(x))
    }

    /// Numerically-stable softplus, same form as `Ops.softplus`'s CPU
    /// reference in `OpsTests.swift` (`max(x,0) + log1p(exp(-|x|))`,
    /// computed in Double) and the kernels' in-shader form.
    private func stableSoftplus(_ x: Float) -> Float {
        if x.isNaN { return Float.nan }
        return max(x, 0) + Float(log1p(Double(exp(-abs(Double(x))))))
    }

    /// Independent Double-precision CPU reference for the gated o_proj
    /// QMV: per-head softplus gate (bf16-rounded once), per-column
    /// gated activation (bf16-rounded at the same point the kernel
    /// rounds it), dequantized weight dot product accumulated in
    /// Double, final result bf16-rounded. Does not mirror the kernels'
    /// block/lane partitioning, just the mathematical contract, so
    /// comparisons against it use a tolerance, not exact equality.
    private func cpuReference(
        x: [Float], gateLogitsRaw: [Float], heads: Int, headDim: Int,
        outDim: Int, inDim: Int,
        codes: [UInt8], scales: [Float], biases: [Float]
    ) -> [Float] {
        let groupsPerRow = inDim / 32
        let gate = (0 ..< heads).map { bf16Round(stableSoftplus(gateLogitsRaw[$0])) }

        var gatedX = [Float](repeating: 0, count: inDim)
        for j in 0 ..< inDim {
            let head = j / headDim
            gatedX[j] = bf16Round(x[j] * gate[head])
        }

        var out = [Float](repeating: 0, count: outDim)
        for r in 0 ..< outDim {
            var acc: Double = 0
            let rowBase = r * inDim
            let groupBase = r * groupsPerRow
            for j in 0 ..< inDim {
                let g = groupBase + j / 32
                let dequant = Double(scales[g]) * Double(codes[rowBase + j]) + Double(biases[g])
                acc += dequant * Double(gatedX[j])
            }
            out[r] = bf16Round(Float(acc))
        }
        return out
    }

    private func cosAndMaxAbsDiff(_ a: [Float], _ b: [Float]) -> (cos: Double, maxAbsDiff: Float) {
        precondition(a.count == b.count)
        var maxAbsDiff: Float = 0
        var dot: Double = 0, na: Double = 0, nb: Double = 0
        for i in 0 ..< a.count {
            maxAbsDiff = max(maxAbsDiff, abs(a[i] - b[i]))
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        let cos = (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 1.0
        return (cos, maxAbsDiff)
    }

    // MARK: - CPU reference vs kernel, real Laguna XS o_proj shapes

    /// `heads * headDim` and `outDim` match the real Laguna XS o_proj
    /// bank (`num_attention_heads=48`, `head_dim=128`, `hidden_size=2048`
    /// in `laguna-xs-2.1-config.json`); `inDim = 6144` is a multiple of
    /// 256 so `qmvGatedV2` doesn't decline.
    @Test("v2 vs Double-precision CPU reference, real Laguna XS o_proj shape, cos > 0.99999")
    func v2MatchesCpuReference() {
        autoreleasepool {
            let heads = 48
            let headDim = 128
            let inDim = heads * headDim  // 6144
            let outDim = 2048

            var seed: UInt64 = 0xC0FF_EE12_3456_789A
            var codes = [UInt8](repeating: 0, count: outDim * inDim)
            for i in 0 ..< codes.count {
                codes[i] = UInt8((nextUniform(&seed) * 0.5 + 0.5) * 255.0)
            }
            let groupsPerRow = inDim / 32
            var scalesF = [Float](repeating: 0, count: outDim * groupsPerRow)
            var biasesF = [Float](repeating: 0, count: outDim * groupsPerRow)
            for i in 0 ..< scalesF.count {
                scalesF[i] = 0.001 + Float((nextUniform(&seed) * 0.5 + 0.5)) * 0.009  // (0.001, 0.01)
                biasesF[i] = Float(nextUniform(&seed)) * 0.01  // [-0.01, 0.01)
            }
            let bank = makeBank(outDim: outDim, inDim: inDim, codes: codes, scalesF: scalesF, biasesF: biasesF)

            let xTensor = randomBf16([inDim], scale: 1.0, seed: &seed)
            let gateTensor = randomBf16([heads], scale: 3.0, seed: &seed)
            let x = xTensor.toFloatArray()
            let gateLogits = gateTensor.toFloatArray()
            // Read the bf16-rounded scale/bias back so the CPU reference
            // uses exactly what the kernel reads, not the pre-round values.
            let scalesRounded = bank.scales.toFloatArray()
            let biasesRounded = bank.biases.toFloatArray()

            let expected = cpuReference(
                x: x, gateLogitsRaw: gateLogits, heads: heads, headDim: headDim,
                outDim: outDim, inDim: inDim, codes: codes, scales: scalesRounded,
                biases: biasesRounded)

            let out = Tensor.empty(shape: [outDim], dtype: .bf16)
            var ok = false
            runAndWait { cb in
                ok = LagunaAttnInt8.qmvGatedV2(
                    input: xTensor, gateLogits: gateTensor, bank: bank, heads: heads,
                    headDim: headDim, on: cb, device: .shared, into: out)
            }
            #expect(ok, "qmvGatedV2 declined at a shape it should support")

            let got = out.toFloatArray()
            let (cos, maxAbsDiff) = cosAndMaxAbsDiff(expected, got)
            #expect(cos > 0.99999, "cos=\(cos)")
            #expect(maxAbsDiff < 0.05, "maxAbsDiff=\(maxAbsDiff)")
        }
    }

    // MARK: - magnitude case

    /// Same real shape, wider-range codes/scale/bias/gate-logit magnitude
    /// (still finite, this is not an Inf/NaN edge-case test) to stress
    /// the accumulation and the softplus stable form away from the
    /// small-value regime the primary test uses.
    @Test("v2 vs CPU reference, large-magnitude codes/scale/bias/gate, cos > 0.99999")
    func v2MagnitudeCase() {
        autoreleasepool {
            let heads = 48
            let headDim = 128
            let inDim = heads * headDim
            let outDim = 2048

            var seed: UInt64 = 0x1357_9BDF_2468_ACE0
            var codes = [UInt8](repeating: 0, count: outDim * inDim)
            for i in 0 ..< codes.count {
                codes[i] = UInt8((nextUniform(&seed) * 0.5 + 0.5) * 255.0)
            }
            let groupsPerRow = inDim / 32
            var scalesF = [Float](repeating: 0, count: outDim * groupsPerRow)
            var biasesF = [Float](repeating: 0, count: outDim * groupsPerRow)
            for i in 0 ..< scalesF.count {
                scalesF[i] = 0.05 + Float((nextUniform(&seed) * 0.5 + 0.5)) * 0.45  // (0.05, 0.5)
                biasesF[i] = Float(nextUniform(&seed)) * 1.0  // [-1, 1)
            }
            let bank = makeBank(outDim: outDim, inDim: inDim, codes: codes, scalesF: scalesF, biasesF: biasesF)

            let xTensor = randomBf16([inDim], scale: 8.0, seed: &seed)
            let gateTensor = randomBf16([heads], scale: 20.0, seed: &seed)
            let x = xTensor.toFloatArray()
            let gateLogits = gateTensor.toFloatArray()
            let scalesRounded = bank.scales.toFloatArray()
            let biasesRounded = bank.biases.toFloatArray()

            let expected = cpuReference(
                x: x, gateLogitsRaw: gateLogits, heads: heads, headDim: headDim,
                outDim: outDim, inDim: inDim, codes: codes, scales: scalesRounded,
                biases: biasesRounded)

            let out = Tensor.empty(shape: [outDim], dtype: .bf16)
            var ok = false
            runAndWait { cb in
                ok = LagunaAttnInt8.qmvGatedV2(
                    input: xTensor, gateLogits: gateTensor, bank: bank, heads: heads,
                    headDim: headDim, on: cb, device: .shared, into: out)
            }
            #expect(ok, "qmvGatedV2 declined at a shape it should support")

            let got = out.toFloatArray()
            #expect(got.allSatisfy { $0.isFinite }, "kernel produced a non-finite value")
            let (cos, maxAbsDiff) = cosAndMaxAbsDiff(expected, got)
            let maxExpected = expected.reduce(0) { max($0, abs($1)) }
            #expect(cos > 0.99999, "cos=\(cos)")
            // Relative bound: this case's outputs run much larger than the
            // primary test's, so a fixed absolute bound isn't meaningful.
            #expect(
                maxAbsDiff < max(0.05, 0.01 * maxExpected),
                "maxAbsDiff=\(maxAbsDiff) maxExpected=\(maxExpected)")
        }
    }

    // MARK: - no-gate case: v2 must match v1 bit-for-bit

    /// Gate logit chosen so `softplus(logit)` rounds to exactly bf16
    /// `1.0` (`ln(e - 1) ~= 0.5413`, the point where softplus is exactly
    /// 1 in exact math; bf16's rounding bucket around 1.0 is wide enough
    /// (ULP 2^-7 near 1.0) that any of the small perturbations here
    /// (the logit itself gets bf16-rounded before either kernel sees it,
    /// and softplus's own float32 evaluation isn't bit-exact to the
    /// Double literal below) lands well inside it. At gate == 1 exactly,
    /// `float(bfloat(float(x) * 1.0))` is a lossless round-trip of an
    /// already-bf16 value in both kernels, so v2's gated activation is
    /// bit-identical to v1's ungated one. The only remaining source of
    /// divergence is float32 summation-order noise from the different
    /// block/lane partitioning (v1: 4 values/lane/block; v2: 8
    /// values/lane/block). That noise is not mathematically zero, and in
    /// practice (Metal toolchain 32023.883) it does occasionally land a
    /// partial sum on the opposite side of a bf16 rounding boundary,
    /// producing last-ULP differences (e.g. 3.78125 vs 3.765625), the
    /// legitimate FP-reordering flake this comment originally
    /// anticipated, not a gate-order bug. A gate-order bug would scale
    /// the whole activation and miss by far more than one ULP, so the
    /// comparison uses the same cos/maxAbsDiff tolerance as the CPU-
    /// reference tests above (relative bound, per the magnitude case)
    /// rather than exact bit equality. The shape stays SMALL (one v2
    /// block, few output rows) to keep accumulated noise near the
    /// rounding step, see file header.
    @Test("v2 gate==1 output matches v1 within FP-reordering tolerance (small shape, one block)")
    func noGateMatchesV1() {
        autoreleasepool {
            let heads = 2
            let headDim = 128
            let inDim = heads * headDim  // 256 == one v2 block
            let outDim = 10  // not a multiple of 8: exercises the row-bounds guard too

            var seed: UInt64 = 0x0BAD_F00D_DEAD_BEEF
            var codes = [UInt8](repeating: 0, count: outDim * inDim)
            for i in 0 ..< codes.count {
                codes[i] = UInt8((nextUniform(&seed) * 0.5 + 0.5) * 255.0)
            }
            let groupsPerRow = inDim / 32
            var scalesF = [Float](repeating: 0, count: outDim * groupsPerRow)
            var biasesF = [Float](repeating: 0, count: outDim * groupsPerRow)
            for i in 0 ..< scalesF.count {
                scalesF[i] = 0.001 + Float((nextUniform(&seed) * 0.5 + 0.5)) * 0.009
                biasesF[i] = Float(nextUniform(&seed)) * 0.01
            }
            let bank = makeBank(outDim: outDim, inDim: inDim, codes: codes, scalesF: scalesF, biasesF: biasesF)

            let xTensor = randomBf16([inDim], scale: 1.0, seed: &seed)
            // softplus(x) == 1 at x = ln(e - 1); both heads get it so the
            // whole row is gate-neutral.
            let softplusInverseOfOne: Float = 0.541_324_854_612_918
            let gateTensor = Tensor.empty(shape: [heads], dtype: .f32)
            gateTensor.copyIn(from: [Float](repeating: softplusInverseOfOne, count: heads))
            let gateBf16 = Tensor.empty(shape: [heads], dtype: .bf16)
            runAndWait { cb in Ops.castF32ToBf16(gateTensor, into: gateBf16, on: cb) }

            let outV1 = Tensor.empty(shape: [outDim], dtype: .bf16)
            let outV2 = Tensor.empty(shape: [outDim], dtype: .bf16)
            var okV1 = false, okV2 = false
            runAndWait { cb in
                okV1 = LagunaAttnInt8.qmvGatedV1(
                    input: xTensor, gateLogits: gateBf16, bank: bank, heads: heads,
                    headDim: headDim, on: cb, device: .shared, into: outV1)
            }
            runAndWait { cb in
                okV2 = LagunaAttnInt8.qmvGatedV2(
                    input: xTensor, gateLogits: gateBf16, bank: bank, heads: heads,
                    headDim: headDim, on: cb, device: .shared, into: outV2)
            }
            #expect(okV1, "qmvGatedV1 declined at a shape it should support")
            #expect(okV2, "qmvGatedV2 declined at a shape it should support")

            let gotV1 = outV1.toFloatArray()
            let gotV2 = outV2.toFloatArray()
            #expect(gotV1.count == gotV2.count)
            let (cos, maxAbsDiff) = cosAndMaxAbsDiff(gotV1, gotV2)
            let maxV1 = gotV1.reduce(0) { max($0, abs($1)) }
            #expect(cos > 0.99999, "cos=\(cos)")
            #expect(
                maxAbsDiff < max(0.05, 0.01 * maxV1),
                "maxAbsDiff=\(maxAbsDiff) maxV1=\(maxV1)")
        }
    }

    // MARK: - shape guards

    /// `qmvGatedV2` requires `bank.inDim` a multiple of 256 (its
    /// `block_size`); `qmvGatedV1` has no such restriction (32-wide
    /// per-pack loop with a bounds check on the last partial block).
    /// `inDim = 192` (heads=6, headDim=32) is a multiple of 32 (valid
    /// group-32 bank) but not of 256, so v2 should decline while v1
    /// still runs.
    @Test("qmvGatedV2 declines when inDim is not a multiple of 256; v1 still runs")
    func v2DeclinesOnNonBlockAlignedInDim() {
        autoreleasepool {
            let heads = 6
            let headDim = 32
            let inDim = heads * headDim  // 192
            let outDim = 8

            let codes = [UInt8](repeating: 1, count: outDim * inDim)
            let groupsPerRow = inDim / 32
            let scalesF = [Float](repeating: 0.01, count: outDim * groupsPerRow)
            let biasesF = [Float](repeating: 0.0, count: outDim * groupsPerRow)
            let bank = makeBank(outDim: outDim, inDim: inDim, codes: codes, scalesF: scalesF, biasesF: biasesF)

            var seed: UInt64 = 0x9999_1111_2222_3333
            let xTensor = randomBf16([inDim], scale: 1.0, seed: &seed)
            let gateTensor = randomBf16([heads], scale: 1.0, seed: &seed)

            let outV1 = Tensor.empty(shape: [outDim], dtype: .bf16)
            let outV2 = Tensor.empty(shape: [outDim], dtype: .bf16)
            var okV1 = false, okV2 = true
            runAndWait { cb in
                okV2 = LagunaAttnInt8.qmvGatedV2(
                    input: xTensor, gateLogits: gateTensor, bank: bank, heads: heads,
                    headDim: headDim, on: cb, device: .shared, into: outV2)
            }
            runAndWait { cb in
                okV1 = LagunaAttnInt8.qmvGatedV1(
                    input: xTensor, gateLogits: gateTensor, bank: bank, heads: heads,
                    headDim: headDim, on: cb, device: .shared, into: outV1)
            }
            #expect(!okV2, "qmvGatedV2 should decline a non-256-aligned inDim")
            #expect(okV1, "qmvGatedV1 should accept a non-256-aligned inDim")
        }
    }
}
