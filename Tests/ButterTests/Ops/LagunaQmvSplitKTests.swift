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
// Correctness tests for `LagunaAttnInt8.qmvSplitK` / `qmvGatedSplitK`
// (the NR0=2/NSG=4 two-stage-reduce split-K restructure of the group-32
// affine INT8 QMV: 4 simdgroups jointly stride the K dimension of ONE
// row pair instead of each simdgroup independently walking the full K
// dimension for its own 4 rows). Only the reduction order changes, not
// the affine math, so the split-K kernels are compared against the same
// kind of Double-precision CPU reference `LagunaOProjQmvV2Tests` uses
// for its sibling kernels — tight cosine/abs-diff tolerance, not exact
// equality (K-split changes float32 summation order).
//
// Shapes mirror the real Laguna XS decode banks (see
// `Tests/ButterTests/Fixtures/laguna-xs-2.1-config.json`): the QKV(+gate)
// bank is `hidden_size=2048` wide with `num_attention_heads` either 48
// (full-attn, gated: Q=6144, K=V=1024 each, gate=48 rows) or 64
// (sliding-attn, ungated: Q=8192, K=V=1024 each); the o_proj bank is
// `inDim=6144` (`num_attention_heads=48` * `head_dim=128`),
// `outDim=hidden_size=2048`.
//
// Every synthetic bank here is built by hand (random int8 codes packed
// 4/word + random bf16 scale/bias), bypassing `LagunaAttnInt8.quantizeG32`
// entirely, same as `LagunaOProjQmvV2Tests` — these tests exercise the
// QMV kernels in isolation from the requant path.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("LagunaQmvSplitK")
struct LagunaQmvSplitKTests {

    // MARK: - GPU availability guard

    /// Pure INT8 QMV kernels (no MPP cooperative tensors), so unlike the
    /// MPP suites this only needs a device to exist, not Apple GPU
    /// Family 9 — same guard `GpuTokenLoopTests` uses.
    private var gpuAvailable: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }

    // MARK: - synthetic data helpers

    /// Deterministic xorshift64 PRNG, same construction
    /// `LagunaOProjQmvV2Tests`/`LagunaSdpaDecodeGqa2Tests` use: no
    /// external dependency, fully reproducible across runs.
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
    /// `codes` is `[outDim][inDim]` in `0...255`; `scalesF`/`biasesF` are
    /// `[outDim][inDim/32]`. Scale/bias are round-tripped through a GPU
    /// f32->bf16 cast (not hand-rounded) so the returned Bank holds
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
    /// the kernels round through (`Gemma3Text`'s cast helpers).
    private func bf16Round(_ x: Float) -> Float {
        bf16BitsToFloatForTest(floatToBf16BitsForTest(x))
    }

    /// Numerically-stable softplus, same form as the o_proj gated
    /// kernels' in-shader form and `LagunaOProjQmvV2Tests`'s reference.
    private func stableSoftplus(_ x: Float) -> Float {
        if x.isNaN { return Float.nan }
        return max(x, 0) + Float(log1p(Double(exp(-abs(Double(x))))))
    }

    /// Independent Double-precision CPU reference for the plain
    /// (non-gated) group-32 affine QMV: dequantized weight dot product
    /// accumulated in Double, per-row result bf16-rounded once at the
    /// end. Does not mirror the kernels' block/lane/simdgroup
    /// partitioning (there is more than one valid summation order for
    /// the same affine decomposition — see `LagunaAttnInt8.qmvSplitKSrc`'s
    /// header comment for why the split-K reduction order still nets the
    /// same value up to float32 rounding), so comparisons against it use
    /// a tolerance, not exact equality.
    private func cpuReference(
        x: [Float], outDim: Int, inDim: Int,
        codes: [UInt8], scales: [Float], biases: [Float]
    ) -> [Float] {
        let groupsPerRow = inDim / 32
        var out = [Float](repeating: 0, count: outDim)
        for r in 0 ..< outDim {
            var acc: Double = 0
            let rowBase = r * inDim
            let groupBase = r * groupsPerRow
            for j in 0 ..< inDim {
                let g = groupBase + j / 32
                let dequant = Double(scales[g]) * Double(codes[rowBase + j]) + Double(biases[g])
                acc += dequant * Double(x[j])
            }
            out[r] = bf16Round(Float(acc))
        }
        return out
    }

    /// Same as `cpuReference` but with the o_proj gate-product-softplus
    /// prologue applied to the activation first, matching
    /// `LagunaAttnInt8.qmvGatedSplitKSrc` / `qmvGatedSrc`'s semantics
    /// (per-head softplus gate, bf16-rounded once each; per-column gated
    /// activation, bf16-rounded at the same point the kernel rounds it).
    private func cpuReferenceGated(
        x: [Float], gateLogitsRaw: [Float], heads: Int, headDim: Int,
        outDim: Int, inDim: Int,
        codes: [UInt8], scales: [Float], biases: [Float]
    ) -> [Float] {
        let gate = (0 ..< heads).map { bf16Round(stableSoftplus(gateLogitsRaw[$0])) }
        var gatedX = [Float](repeating: 0, count: inDim)
        for j in 0 ..< inDim {
            let head = j / headDim
            gatedX[j] = bf16Round(x[j] * gate[head])
        }
        return cpuReference(
            x: gatedX, outDim: outDim, inDim: inDim, codes: codes, scales: scales, biases: biases)
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

    /// Random codes/scales/biases in the "primary" small-magnitude range
    /// `LagunaOProjQmvV2Tests.v2MatchesCpuReference` uses.
    private func randomBankData(outDim: Int, inDim: Int, seed: inout UInt64) -> (
        codes: [UInt8], scalesF: [Float], biasesF: [Float]
    ) {
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
        return (codes, scalesF, biasesF)
    }

    // MARK: - CPU reference vs kernel, real Laguna XS QKV(+gate) bank shapes

    /// Full-attn, gated QKV bank: `hidden=2048`, `num_attention_heads=48`,
    /// `head_dim=128` -> Q=6144; `num_kv_heads=8` -> K=V=1024 each;
    /// gate=48 (one row per query head). `outDim = 6144+1024+1024+48 =
    /// 8240`, matching `LagunaQkvNormFoldTests.fullAttnGated`'s shape.
    @Test("qmvSplitK vs Double-precision CPU reference, full-attn+gate QKV bank shape, cos > 0.99999")
    func splitKMatchesCpuReferenceFullAttnGated() throws {
        try #require(gpuAvailable, "Metal required for this test")
        autoreleasepool {
            let hidden = 2048
            let qDim = 48 * 128  // 6144
            let kvDim = 8 * 128  // 1024
            let gateDim = 48
            let outDim = qDim + kvDim + kvDim + gateDim  // 8240

            var seed: UInt64 = 0xC0FF_EE12_3456_789A
            let (codes, scalesF, biasesF) = randomBankData(outDim: outDim, inDim: hidden, seed: &seed)
            let bank = makeBank(outDim: outDim, inDim: hidden, codes: codes, scalesF: scalesF, biasesF: biasesF)

            let xTensor = randomBf16([hidden], scale: 1.0, seed: &seed)
            let x = xTensor.toFloatArray()
            let scalesRounded = bank.scales.toFloatArray()
            let biasesRounded = bank.biases.toFloatArray()

            let expected = cpuReference(
                x: x, outDim: outDim, inDim: hidden, codes: codes, scales: scalesRounded,
                biases: biasesRounded)

            let out = Tensor.empty(shape: [outDim], dtype: .bf16)
            var ok = false
            runAndWait { cb in
                ok = LagunaAttnInt8.qmvSplitK(input: xTensor, bank: bank, on: cb, device: .shared, into: out)
            }
            #expect(ok, "qmvSplitK declined at a shape it should support")

            let got = out.toFloatArray()
            let (cos, maxAbsDiff) = cosAndMaxAbsDiff(expected, got)
            // Magnitude-relative maxAbsDiff, same form as `splitKMagnitudeCase`
            // and `splitKBoundaryRows` (and the sibling `LagunaOProjQmvV2Tests`
            // magnitude case). The split-K reduction is near-ULP BY DESIGN
            // (the float32 K-sum is split across `NSG=4` simdgroups instead of
            // walked serially by one -- see `LagunaAttnInt8.splitKEnabled`'s
            // header), not bit-identical to the Double CPU reference. The
            // per-element affine math is byte-for-byte `qmvSrc` (activation
            // load, dequant, per-group scale/bias, raw dot, bias term, f32
            // accumulate), so the only gap is last-bit summation order. At
            // these QKV(+gate) bank magnitudes the outputs run ~O(16), where a
            // SINGLE bf16 ULP is already 0.125 -- so the worst of these 8240+
            // rows straddling a bf16 rounding boundary trips a fixed 0.05 gate
            // even though nothing is wrong. `0.01 * maxExpected` always covers
            // a 1-ULP flip (1 ULP = maxExpected/128 ~= 0.0078*maxExpected)
            // while still catching any structured (many-row / super-ULP)
            // divergence, which `cos > 0.99999` guards independently. The
            // narrower o_proj shapes (2048 rows) stay under a bare 0.05, which
            // is why the sibling non-split v2 CPU-reference test can keep it.
            let maxExpected = expected.reduce(0) { max($0, abs($1)) }
            #expect(cos > 0.99999, "cos=\(cos)")
            #expect(
                maxAbsDiff < max(0.05, 0.01 * maxExpected),
                "maxAbsDiff=\(maxAbsDiff) maxExpected=\(maxExpected)")
        }
    }

    /// Sliding-attn, ungated QKV bank: `num_attention_heads=64` -> Q=8192,
    /// K=V=1024 each, no gate rows. `outDim = 8192+1024+1024 = 10240`,
    /// matching `LagunaQkvNormFoldTests.slidingAttnUngated`'s shape.
    @Test("qmvSplitK vs Double-precision CPU reference, sliding-attn QKV bank shape, cos > 0.99999")
    func splitKMatchesCpuReferenceSlidingUngated() throws {
        try #require(gpuAvailable, "Metal required for this test")
        autoreleasepool {
            let hidden = 2048
            let qDim = 64 * 128  // 8192
            let kvDim = 8 * 128  // 1024
            let outDim = qDim + kvDim + kvDim  // 10240

            var seed: UInt64 = 0x1357_9BDF_2468_ACE0
            let (codes, scalesF, biasesF) = randomBankData(outDim: outDim, inDim: hidden, seed: &seed)
            let bank = makeBank(outDim: outDim, inDim: hidden, codes: codes, scalesF: scalesF, biasesF: biasesF)

            let xTensor = randomBf16([hidden], scale: 1.0, seed: &seed)
            let x = xTensor.toFloatArray()
            let scalesRounded = bank.scales.toFloatArray()
            let biasesRounded = bank.biases.toFloatArray()

            let expected = cpuReference(
                x: x, outDim: outDim, inDim: hidden, codes: codes, scales: scalesRounded,
                biases: biasesRounded)

            let out = Tensor.empty(shape: [outDim], dtype: .bf16)
            var ok = false
            runAndWait { cb in
                ok = LagunaAttnInt8.qmvSplitK(input: xTensor, bank: bank, on: cb, device: .shared, into: out)
            }
            #expect(ok, "qmvSplitK declined at a shape it should support")

            let got = out.toFloatArray()
            let (cos, maxAbsDiff) = cosAndMaxAbsDiff(expected, got)
            // Magnitude-relative maxAbsDiff, same form as `splitKMagnitudeCase`
            // and `splitKBoundaryRows` (and the sibling `LagunaOProjQmvV2Tests`
            // magnitude case). The split-K reduction is near-ULP BY DESIGN
            // (the float32 K-sum is split across `NSG=4` simdgroups instead of
            // walked serially by one -- see `LagunaAttnInt8.splitKEnabled`'s
            // header), not bit-identical to the Double CPU reference. The
            // per-element affine math is byte-for-byte `qmvSrc` (activation
            // load, dequant, per-group scale/bias, raw dot, bias term, f32
            // accumulate), so the only gap is last-bit summation order. At
            // these QKV(+gate) bank magnitudes the outputs run ~O(16), where a
            // SINGLE bf16 ULP is already 0.125 -- so the worst of these 8240+
            // rows straddling a bf16 rounding boundary trips a fixed 0.05 gate
            // even though nothing is wrong. `0.01 * maxExpected` always covers
            // a 1-ULP flip (1 ULP = maxExpected/128 ~= 0.0078*maxExpected)
            // while still catching any structured (many-row / super-ULP)
            // divergence, which `cos > 0.99999` guards independently. The
            // narrower o_proj shapes (2048 rows) stay under a bare 0.05, which
            // is why the sibling non-split v2 CPU-reference test can keep it.
            let maxExpected = expected.reduce(0) { max($0, abs($1)) }
            #expect(cos > 0.99999, "cos=\(cos)")
            #expect(
                maxAbsDiff < max(0.05, 0.01 * maxExpected),
                "maxAbsDiff=\(maxAbsDiff) maxExpected=\(maxExpected)")
        }
    }

    // MARK: - output-slice-boundary case

    /// The concatenated QKV(+gate) bank's consumer slices `out` by fixed
    /// Q/K/V/gate row offsets (`LagunaText.forward`'s `attnInt8QScratch`
    /// / `KScratch` / `VScratch` / `GateScratch`), so a bug localized to
    /// a handful of rows near a segment boundary could hide inside an
    /// aggregate cosine-similarity check over 8240 rows. This checks the
    /// specific rows at and adjacent to each Q|K|V|gate boundary against
    /// the CPU reference individually, with a per-row tolerance instead
    /// of an aggregate one. All boundaries here (6144, 7168, 8192) are
    /// even, so every `NR0=2` row pair the kernel dispatches falls
    /// entirely within one segment — this test exists precisely to catch
    /// an off-by-one in that assumption (e.g. a stray `tgid.x` or
    /// `rowBase` miscount) rather than assume it holds.
    @Test("qmvSplitK output matches CPU reference at every Q/K/V/gate boundary row")
    func splitKBoundaryRows() throws {
        try #require(gpuAvailable, "Metal required for this test")
        autoreleasepool {
            let hidden = 2048
            let qDim = 48 * 128  // 6144
            let kvDim = 8 * 128  // 1024
            let gateDim = 48
            let outDim = qDim + kvDim + kvDim + gateDim  // 8240

            var seed: UInt64 = 0x0BAD_F00D_DEAD_BEEF
            let (codes, scalesF, biasesF) = randomBankData(outDim: outDim, inDim: hidden, seed: &seed)
            let bank = makeBank(outDim: outDim, inDim: hidden, codes: codes, scalesF: scalesF, biasesF: biasesF)

            let xTensor = randomBf16([hidden], scale: 1.0, seed: &seed)
            let x = xTensor.toFloatArray()
            let scalesRounded = bank.scales.toFloatArray()
            let biasesRounded = bank.biases.toFloatArray()

            let expected = cpuReference(
                x: x, outDim: outDim, inDim: hidden, codes: codes, scales: scalesRounded,
                biases: biasesRounded)

            let out = Tensor.empty(shape: [outDim], dtype: .bf16)
            var ok = false
            runAndWait { cb in
                ok = LagunaAttnInt8.qmvSplitK(input: xTensor, bank: bank, on: cb, device: .shared, into: out)
            }
            #expect(ok, "qmvSplitK declined at a shape it should support")
            let got = out.toFloatArray()

            let qEnd = qDim
            let kEnd = qDim + kvDim
            let vEnd = qDim + kvDim + kvDim
            let boundaryRows = [
                0, qEnd - 2, qEnd - 1, qEnd, qEnd + 1,
                kEnd - 2, kEnd - 1, kEnd, kEnd + 1,
                vEnd - 2, vEnd - 1, vEnd, vEnd + 1,
                outDim - 2, outDim - 1,
            ]
            for row in boundaryRows {
                let diff = abs(got[row] - expected[row])
                #expect(
                    diff < max(0.02, 0.02 * abs(expected[row])),
                    "row \(row): got=\(got[row]) expected=\(expected[row]) diff=\(diff)")
            }
        }
    }

    // MARK: - magnitude case

    /// Same full-attn+gate shape, wider-range codes/scale/bias/activation
    /// magnitude to stress the accumulation away from the small-value
    /// regime the primary test uses (mirrors
    /// `LagunaOProjQmvV2Tests.v2MagnitudeCase`).
    @Test("qmvSplitK vs CPU reference, large-magnitude codes/scale/bias/activation, cos > 0.99999")
    func splitKMagnitudeCase() throws {
        try #require(gpuAvailable, "Metal required for this test")
        autoreleasepool {
            let hidden = 2048
            let qDim = 48 * 128
            let kvDim = 8 * 128
            let gateDim = 48
            let outDim = qDim + kvDim + kvDim + gateDim

            var seed: UInt64 = 0x1111_2222_3333_4444
            var codes = [UInt8](repeating: 0, count: outDim * hidden)
            for i in 0 ..< codes.count {
                codes[i] = UInt8((nextUniform(&seed) * 0.5 + 0.5) * 255.0)
            }
            let groupsPerRow = hidden / 32
            var scalesF = [Float](repeating: 0, count: outDim * groupsPerRow)
            var biasesF = [Float](repeating: 0, count: outDim * groupsPerRow)
            for i in 0 ..< scalesF.count {
                scalesF[i] = 0.05 + Float((nextUniform(&seed) * 0.5 + 0.5)) * 0.45  // (0.05, 0.5)
                biasesF[i] = Float(nextUniform(&seed)) * 1.0  // [-1, 1)
            }
            let bank = makeBank(outDim: outDim, inDim: hidden, codes: codes, scalesF: scalesF, biasesF: biasesF)

            let xTensor = randomBf16([hidden], scale: 8.0, seed: &seed)
            let x = xTensor.toFloatArray()
            let scalesRounded = bank.scales.toFloatArray()
            let biasesRounded = bank.biases.toFloatArray()

            let expected = cpuReference(
                x: x, outDim: outDim, inDim: hidden, codes: codes, scales: scalesRounded,
                biases: biasesRounded)

            let out = Tensor.empty(shape: [outDim], dtype: .bf16)
            var ok = false
            runAndWait { cb in
                ok = LagunaAttnInt8.qmvSplitK(input: xTensor, bank: bank, on: cb, device: .shared, into: out)
            }
            #expect(ok, "qmvSplitK declined at a shape it should support")

            let got = out.toFloatArray()
            #expect(got.allSatisfy { $0.isFinite }, "kernel produced a non-finite value")
            let (cos, maxAbsDiff) = cosAndMaxAbsDiff(expected, got)
            let maxExpected = expected.reduce(0) { max($0, abs($1)) }
            #expect(cos > 0.99999, "cos=\(cos)")
            #expect(
                maxAbsDiff < max(0.05, 0.01 * maxExpected),
                "maxAbsDiff=\(maxAbsDiff) maxExpected=\(maxExpected)")
        }
    }

    // MARK: - row-count-not-multiple-of-2 decline

    /// `qmvSplitK`'s fixed `NR0 = 2` row-pair tiling has no partial-pair
    /// tail path, so an odd `bank.outDim` must decline cleanly (`out`
    /// left untouched) rather than under- or over-write.
    @Test("qmvSplitK declines when bank.outDim is not a multiple of 2")
    func splitKDeclinesOnOddOutDim() throws {
        try #require(gpuAvailable, "Metal required for this test")
        autoreleasepool {
            let hidden = 64
            let outDim = 9  // odd

            let codes = [UInt8](repeating: 1, count: outDim * hidden)
            let groupsPerRow = hidden / 32
            let scalesF = [Float](repeating: 0.01, count: outDim * groupsPerRow)
            let biasesF = [Float](repeating: 0.0, count: outDim * groupsPerRow)
            let bank = makeBank(outDim: outDim, inDim: hidden, codes: codes, scalesF: scalesF, biasesF: biasesF)

            var seed: UInt64 = 0x9999_1111_2222_3333
            let xTensor = randomBf16([hidden], scale: 1.0, seed: &seed)

            // Sentinel fill so a decline that silently wrote to `out`
            // would be caught, not just a `false` return. 3.0 is exactly
            // representable in bf16 (small integer, zero mantissa bits
            // needed), so the round-trip through the GPU f32->bf16 cast
            // used to seed it is lossless and the comparison below can
            // use exact equality.
            let sentinel: Float = 3.0
            let out = Tensor.empty(shape: [outDim], dtype: .bf16)
            let sentinelF32 = Tensor.empty(shape: [outDim], dtype: .f32)
            sentinelF32.copyIn(from: [Float](repeating: sentinel, count: outDim))
            runAndWait { cb in Ops.castF32ToBf16(sentinelF32, into: out, on: cb) }

            var ok = true
            runAndWait { cb in
                ok = LagunaAttnInt8.qmvSplitK(input: xTensor, bank: bank, on: cb, device: .shared, into: out)
            }
            #expect(!ok, "qmvSplitK should decline an odd bank.outDim")
            let untouched = out.toFloatArray()
            #expect(
                untouched.allSatisfy { $0 == sentinel },
                "qmvSplitK wrote to `out` despite declining")
        }
    }

    // MARK: - gated o_proj split-K

    /// Real Laguna XS o_proj bank shape (`num_attention_heads=48`,
    /// `head_dim=128` -> `inDim=6144`; `hidden_size=2048` -> `outDim=
    /// 2048`), matching `LagunaOProjQmvV2Tests`'s shape.
    @Test("qmvGatedSplitK vs Double-precision CPU reference, real Laguna XS o_proj shape, cos > 0.99999")
    func gatedSplitKMatchesCpuReference() throws {
        try #require(gpuAvailable, "Metal required for this test")
        autoreleasepool {
            let heads = 48
            let headDim = 128
            let inDim = heads * headDim  // 6144
            let outDim = 2048

            var seed: UInt64 = 0xABCD_EF01_2345_6789
            let (codes, scalesF, biasesF) = randomBankData(outDim: outDim, inDim: inDim, seed: &seed)
            let bank = makeBank(outDim: outDim, inDim: inDim, codes: codes, scalesF: scalesF, biasesF: biasesF)

            let xTensor = randomBf16([inDim], scale: 1.0, seed: &seed)
            let gateTensor = randomBf16([heads], scale: 3.0, seed: &seed)
            let x = xTensor.toFloatArray()
            let gateLogits = gateTensor.toFloatArray()
            let scalesRounded = bank.scales.toFloatArray()
            let biasesRounded = bank.biases.toFloatArray()

            let expected = cpuReferenceGated(
                x: x, gateLogitsRaw: gateLogits, heads: heads, headDim: headDim,
                outDim: outDim, inDim: inDim, codes: codes, scales: scalesRounded,
                biases: biasesRounded)

            let out = Tensor.empty(shape: [outDim], dtype: .bf16)
            var ok = false
            runAndWait { cb in
                ok = LagunaAttnInt8.qmvGatedSplitK(
                    input: xTensor, gateLogits: gateTensor, bank: bank, heads: heads,
                    headDim: headDim, on: cb, device: .shared, into: out)
            }
            #expect(ok, "qmvGatedSplitK declined at a shape it should support")

            let got = out.toFloatArray()
            let (cos, maxAbsDiff) = cosAndMaxAbsDiff(expected, got)
            #expect(cos > 0.99999, "cos=\(cos)")
            #expect(maxAbsDiff < 0.05, "maxAbsDiff=\(maxAbsDiff)")
        }
    }

    /// `qmvGatedSplitK`'s fixed `NR0 = 2` tiling declines an odd
    /// `bank.outDim`, same guard as the ungated `qmvSplitK`.
    @Test("qmvGatedSplitK declines when bank.outDim is not a multiple of 2")
    func gatedSplitKDeclinesOnOddOutDim() throws {
        try #require(gpuAvailable, "Metal required for this test")
        autoreleasepool {
            let heads = 4
            let headDim = 32
            let inDim = heads * headDim  // 128
            let outDim = 5  // odd

            let codes = [UInt8](repeating: 1, count: outDim * inDim)
            let groupsPerRow = inDim / 32
            let scalesF = [Float](repeating: 0.01, count: outDim * groupsPerRow)
            let biasesF = [Float](repeating: 0.0, count: outDim * groupsPerRow)
            let bank = makeBank(outDim: outDim, inDim: inDim, codes: codes, scalesF: scalesF, biasesF: biasesF)

            var seed: UInt64 = 0x4444_5555_6666_7777
            let xTensor = randomBf16([inDim], scale: 1.0, seed: &seed)
            let gateTensor = randomBf16([heads], scale: 1.0, seed: &seed)

            let out = Tensor.empty(shape: [outDim], dtype: .bf16)
            var ok = true
            runAndWait { cb in
                ok = LagunaAttnInt8.qmvGatedSplitK(
                    input: xTensor, gateLogits: gateTensor, bank: bank, heads: heads,
                    headDim: headDim, on: cb, device: .shared, into: out)
            }
            #expect(!ok, "qmvGatedSplitK should decline an odd bank.outDim")
        }
    }
}
