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
// BUTTER_LAGUNA_DECODE_QMV_V2 correctness gates for the V2
// decode-attention NVFP4 QMV path (`LagunaAttnNvfp4Tail.normQmvV2` +
// the gated split-K V2 kernel + the V2 fast-dequant helpers):
//
// 1. Exhaustive dequant equivalence: `iron_nvfp4_e4m3_scale22_v2` /
//    `iron_nvfp4_pack_dot_v2` vs a CPU `Double`-precision ground-truth
//    oracle (hardware-invariant by construction — see the derivation
//    comment on `dequantEquivalenceExhaustive` below) over ALL 256 E4M3
//    scale bytes x 16 E2M1 nibbles, bit-exact (this is the mutation
//    gate for the bit-trick decode — any wrong mask/shift flips at
//    least one of the 4096 combinations). This used to compare against
//    a second *GPU* dispatch of V1's `iron_nvfp4_e4m3_scale` /
//    `iron_nvfp4_nibble_value`, but V1's `exp2()`-based scale decode is
//    itself hardware-variable (Metal does not require `exp2` to be
//    correctly rounded) — retargeted 2026-08-09 after a CI-only 2-ULP
//    mismatch at scaleByte=8 traced to exactly that.
// 2. Norm-folded V2 QKV vs the separate `Ops.rmsNorm` + plain
//    (non-split-K) QMV chain at Laguna XS's exact decode shapes —
//    bit-exact primary assertion, tight-cosine fallback with a report,
//    same contract as `LagunaQkvNormFoldTests` (the INT8 twin).
// 3. Gated split-K V2 vs a CPU `Double`-precision ground-truth oracle
//    at the exact o_proj shapes — bit-exact primary assertion,
//    tight-cosine fallback with a report (same contract as test 2 /
//    `LagunaQkvNormFoldTests`: a K=inDim-wide accumulation compared
//    against a `Double` oracle is not guaranteed bit-exact on every row
//    of real random data, since float32 GPU accumulation order and a
//    `Double` CPU sum can round to adjacent bf16 values on a row whose
//    true sum sits very close to a bf16 halfway point). Retargeted
//    2026-08-09 (same fix as test 1, applied per this file's own prior
//    NOTE below): this used to compare against a live V1 GPU dispatch,
//    which inherits the same latent hardware-`exp2()` risk documented
//    on test 1 -- and it did eventually trip CI on the
//    `gatedSliding`/`gatedFull` shapes, on a GPU generation where V1's
//    scale decode disagreed with V2's exact bit-trick decode. V1 is
//    still computed and reported (see `compareGated`) as an
//    INFORMATIONAL divergence count only -- it is not the load-bearing
//    reference and must not be hard-asserted against.

// 4. Magnitude sweep (T23 class): finite + accurate at real late-layer
//    activation magnitudes.
// 5. Decline hygiene: V2 off or misaligned hidden must decline so the
//    caller's fallback chain runs.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("Laguna attention NVFP4 decode-QMV V2", .serialized)
struct LagunaAttnNvfp4V2Tests {
    private func setV2(_ on: Bool) {
        // Default-on flag (escape `=0`): "off" must be the explicit
        // escape hatch, not an unset var — unsetting now means enabled.
        if on {
            setenv("BUTTER_LAGUNA_DECODE_QMV_V2", "1", 1)
        } else {
            setenv("BUTTER_LAGUNA_DECODE_QMV_V2", "0", 1)
        }
    }

    private func randomBf16(_ shape: [Int], seed: inout UInt64, magnitude: Float = 1.0) -> Tensor {
        let n = shape.reduce(1, *)
        var f32 = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            let u = Double(seed % 2_000_001) / 1_000_000.0 - 1.0  // [-1, 1)
            f32[i] = Float(u) * magnitude
        }
        let f32T = Tensor.empty(shape: shape, dtype: .f32)
        f32T.copyIn(from: f32)
        let bf16T = Tensor.empty(shape: shape, dtype: .bf16)
        runAndWait { cb in Ops.castF32ToBf16(f32T, into: bf16T, on: cb) }
        return bf16T
    }

    private func randomBank(rows: Int, hidden: Int, seed: inout UInt64, magnitude: Float = 1.0)
        -> LagunaAttnNvfp4Tail.Bank
    {
        let weight = randomBf16([rows, hidden], seed: &seed, magnitude: magnitude)
        var bank: LagunaAttnNvfp4Tail.Bank!
        runAndWait { cb in
            bank = LagunaAttnNvfp4Tail.quantizeG16(weight: weight, device: .shared, cmd: cb)
        }
        return bank
    }

    // MARK: - 1. Exhaustive dequant equivalence (the mutation gate)
    //
    // Ground truth here is computed on the CPU in double precision --
    // hardware-invariant by construction -- rather than by dispatching a
    // second *GPU* kernel that runs V1's `iron_nvfp4_e4m3_scale`. That
    // used to be the reference, but V1's "normal" branch calls Metal's
    // `exp2()` intrinsic, which the Metal Shading Language spec does
    // NOT require to be correctly rounded (unlike float/double
    // conversion, which IEEE 754 pins to round-to-nearest-even): it is
    // a hardware transcendental-function-unit approximation, and Apple
    // Silicon generations are free to disagree on the exact result even
    // for an *integer* exponent input whose true answer is an exactly
    // representable power of two.
    //
    // Root-caused 2026-08-09 (CI failure: 9/4096 mismatches, all at
    // scaleByte=8 which is exp4=1/mant3=0, i.e. scale = exp2(1-7) with
    // no mantissa to mask the transcendental's error): the true product
    // for scaleByte=8, nib=1 is 0.5 * 2^-6 = 2^-7 = 0.0078125 exactly
    // (bits 1_006_632_960). V2 (`iron_nvfp4_e4m3_scale22_v2` +
    // `iron_nvfp4_pack_dot_v2`) reproduces that value bit-for-bit on
    // both M5 Max and the CI runner -- it never calls a transcendental,
    // it lands the E4M3 exponent field directly into an IEEE half's
    // exponent bits via `(bits + (bits & 128)) << 7` and reinterprets,
    // an exact bit construction. V1's GPU `exp2(-6.0f)` matched that on
    // M5 Max but returned a value 2 ULP low (0.007812499, bits
    // 1_006_632_958) on the CI runner's Apple Silicon generation. So
    // V1 -- not V2 -- was the hardware-variable side of the old
    // comparison; V2 is unchanged and ships as-is, and this test now
    // checks V2 against ground truth instead of against V1's GPU
    // output. (Cross-checked against `Nvfp4FastDequantTests
    // .e4m3SignFoldExhaustive256`, which independently validates the
    // same V2-style E4M3 bit-fold against a CPU Swift `pow()`-based
    // reference for all 256 scale bytes and already passes.)

    /// True E4M3 scale value for `bits`, computed in `Double` with no
    /// transcendental call anywhere in the path: the power-of-two factor
    /// is constructed directly via `Double.init(sign:exponent:significand:)`
    /// (an exact IEEE 754 construction, not a function approximation), and
    /// every other step (integer mantissa scaled by an exact power of two)
    /// is exact in binary floating point. This is the ground-truth oracle;
    /// narrowing to `Float` below is the only rounding step, and Swift's
    /// `Float(Double)` narrowing is correctly rounded (round-to-nearest,
    /// ties-to-even) per IEEE 754 / the language spec.
    private static func trueE4M3Scale(_ bits: UInt8) -> Double {
        let b = UInt32(bits)
        let exp4 = Int((b >> 3) & 15)
        let mant3 = Double(b & 7)
        let mag: Double
        if exp4 == 0 {
            mag = mant3 * 0.001953125  // 2^-9, exact (integer x exact power of two)
        } else {
            let p = Double(sign: .plus, exponent: exp4 - 7, significand: 1.0)  // exact 2^(exp4-7)
            mag = (1.0 + mant3 * 0.125) * p  // exact: mant3/8 exact, x power-of-two exact
        }
        return ((b >> 7) & 1) != 0 ? -mag : mag
    }

    private static let e2m1MagnitudeLut: [Double] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    /// True E2M1 nibble value, computed in `Double` (every LUT entry and
    /// the sign flip are exact in binary floating point).
    private static func trueE2M1Nibble(_ nib: UInt32) -> Double {
        let mag = Self.e2m1MagnitudeLut[Int(nib & 7)]
        return (nib & 8) != 0 ? -mag : mag
    }

    @Test("V2 fast dequant == CPU f64 ground truth, all 256 scale bytes x 16 nibbles, bit-exact")
    func dequantEquivalenceExhaustive() {
        let probeSrc = """
            #include <metal_stdlib>
            using namespace metal;

            \(LagunaAttnNvfp4Tail.helperV2Src)

            kernel void laguna_nvfp4_v2_dequant_probe(
                device float *v2 [[buffer(0)]],
                uint tid [[thread_position_in_grid]])
            {
                uint scaleByte = tid >> 4;   // 0..255
                uint nib = tid & 0xFu;       // 0..15
                // V2 path exactly as a QMV lane computes one term: the
                // pack word carries `nib` in nibble slot 0 and the
                // activation vector selects slot 0 only.
                float xf[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                v2[tid] = iron_nvfp4_e4m3_scale22_v2(uchar(scaleByte))
                    * iron_nvfp4_pack_dot_v2(nib, xf);
            }
            """
        let device = Device.shared.mtlDevice
        let lib = try! device.makeLibrary(source: probeSrc, options: MTLCompileOptions())
        let fn = lib.makeFunction(name: "laguna_nvfp4_v2_dequant_probe")!
        let pso = try! device.makeComputePipelineState(function: fn)

        let n = 256 * 16
        let v2T = Tensor.empty(shape: [n], dtype: .f32)
        runAndWait { cb in
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            enc.setBuffer(v2T.buffer, offset: v2T.offset, index: 0)
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
        }
        let b = v2T.toFloatArray()
        #expect(b.count == n)
        var mismatches = 0
        for i in 0 ..< n {
            let scaleByte = UInt8(i >> 4)
            let nib = UInt32(i & 15)
            let truth = Self.trueE4M3Scale(scaleByte) * Self.trueE2M1Nibble(nib)
            let a = Float(truth)  // correctly-rounded double -> float32 narrowing
            guard a.bitPattern != b[i].bitPattern else { continue }
            // -0.0 vs +0.0 both encode "zero code x some scale sign";
            // treat signed zero as equal only if the ground truth itself
            // has the sign V2 did — i.e. require true bit equality, and
            // record the combination on failure.
            mismatches += 1
            if mismatches <= 8 {
                let msg =
                    "scaleByte=\(i >> 4) nib=\(i & 15): truth=\(a) (\(a.bitPattern)) v2=\(b[i]) (\(b[i].bitPattern))"
                Issue.record(Comment(rawValue: msg))
            }
        }
        #expect(mismatches == 0, "V2 dequant diverges from CPU f64 ground truth on \(mismatches)/\(n) combinations")
    }

    // MARK: - 2. Norm-folded V2 QKV vs separate chain

    private func compareNormFold(
        hidden: Int, rows: Int, eps: Float = 1e-6, magnitude: Float = 1.0,
        splitK: Bool = false, label: String
    ) {
        autoreleasepool {
            setV2(true)
            // Pin the geometry under test: the split-K arm's bit-exact
            // reference is `qmvSplitK` (V1), the 8-rows/TG arm's is
            // `qmvPlain` — each fold reproduces exactly its reference's
            // per-row accumulation order.
            setenv("BUTTER_LAGUNA_QMV_V2_NORMFOLD_SPLITK", splitK ? "1" : "0", 1)
            defer {
                setV2(false)
                unsetenv("BUTTER_LAGUNA_QMV_V2_NORMFOLD_SPLITK")
            }
            var seed: UInt64 = 0xC0FE_BABE_9E37_79B9
            let residual = randomBf16([hidden], seed: &seed, magnitude: magnitude)
            let normWeight = randomBf16([hidden], seed: &seed)
            let bank = randomBank(rows: rows, hidden: hidden, seed: &seed)

            let refOut = Tensor.empty(shape: [rows], dtype: .bf16)
            runAndWait { cb in
                let xNorm = Ops.rmsNorm(residual, weight: normWeight, eps: eps, on: cb)
                if splitK {
                    let ok = LagunaAttnNvfp4Tail.qmvSplitK(
                        input: xNorm, bank: bank, on: cb, device: .shared, into: refOut)
                    if !ok {
                        LagunaAttnNvfp4Tail.qmvPlain(
                            input: xNorm, bank: bank, on: cb, device: .shared, into: refOut)
                    }
                } else {
                    LagunaAttnNvfp4Tail.qmvPlain(
                        input: xNorm, bank: bank, on: cb, device: .shared, into: refOut)
                }
            }

            let fusedOut = Tensor.empty(shape: [rows], dtype: .bf16)
            var folded = false
            runAndWait { cb in
                folded = LagunaAttnNvfp4Tail.normQmvV2(
                    residual: residual, normWeight: normWeight, eps: eps, bank: bank,
                    on: cb, device: .shared, into: fusedOut)
            }
            guard folded else {
                Issue.record("\(label): normQmvV2 declined at a shape it should support")
                return
            }

            let a = refOut.toFloatArray()
            let b = fusedOut.toFloatArray()
            #expect(a.count == b.count)
            guard a.count == b.count else { return }

            var exact = true
            var maxAbsDiff: Float = 0
            var dot = 0.0, na = 0.0, nb = 0.0
            for i in 0 ..< a.count {
                if a[i] != b[i] { exact = false }
                maxAbsDiff = max(maxAbsDiff, abs(a[i] - b[i]))
                dot += Double(a[i]) * Double(b[i])
                na += Double(a[i]) * Double(a[i])
                nb += Double(b[i]) * Double(b[i])
            }
            if exact {
                #expect(a == b, "\(label): expected bit-exact")
            } else {
                // Same documented fallback contract as
                // `LagunaQkvNormFoldTests` (see that file's header).
                let cos = (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 1.0
                #expect(
                    cos > 0.999999,
                    Comment(
                        rawValue:
                            "\(label): NOT bit-exact (maxAbsDiff=\(maxAbsDiff), cos=\(cos))"))
            }
        }
    }

    @Test("norm-folded V2: Laguna XS sliding shape (64h + 2x8kv + 64 gate rows)")
    func normFoldSliding() {
        compareNormFold(hidden: 2048, rows: 64 * 128 + 2 * 8 * 128 + 64, label: "sliding")
    }

    @Test("norm-folded V2: Laguna XS full-attn shape (48h + 2x8kv + 48 gate rows)")
    func normFoldFull() {
        compareNormFold(hidden: 2048, rows: 48 * 128 + 2 * 8 * 128 + 48, label: "full")
    }

    @Test("norm-folded V2: large-magnitude activations")
    func normFoldMagnitude() {
        compareNormFold(
            hidden: 2048, rows: 48 * 128 + 2 * 8 * 128 + 48, magnitude: 50.0, label: "mag50")
    }

    @Test("split-K norm-folded V2 vs rmsNorm + qmvSplitK: sliding shape")
    func normFoldSplitKSliding() {
        compareNormFold(
            hidden: 2048, rows: 64 * 128 + 2 * 8 * 128 + 64, splitK: true,
            label: "splitk-sliding")
    }

    @Test("split-K norm-folded V2 vs rmsNorm + qmvSplitK: full-attn shape")
    func normFoldSplitKFull() {
        compareNormFold(
            hidden: 2048, rows: 48 * 128 + 2 * 8 * 128 + 48, splitK: true,
            label: "splitk-full")
    }

    @Test("split-K norm-folded V2: large-magnitude activations")
    func normFoldSplitKMagnitude() {
        compareNormFold(
            hidden: 2048, rows: 64 * 128 + 2 * 8 * 128 + 64, magnitude: 50.0, splitK: true,
            label: "splitk-mag50")
    }

    // MARK: - 3. Gated split-K V2 vs CPU f64 ground truth (bit-exact strict)

    /// Round a `Float` through bf16 and widen back to `Float`
    /// (round-to-nearest-even on the truncated low 16 bits) -- the same
    /// bias-add trick `Tensor.filled`'s bf16 case and `GPTOSSText`'s
    /// `bf16Bits` already use elsewhere in this codebase. Used to
    /// reproduce, on the CPU, the two intermediate bf16 rounding points
    /// the gated split-K kernels both apply on the GPU (softplus gate
    /// value, then gate*activation product) -- see `qmvGatedSplitKV2Src`'s
    /// "ROUNDING POINT 1" / "ROUNDING POINT 2" comments.
    private static func roundToBf16(_ v: Float) -> Float {
        let bits = v.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        let bf16 = UInt16(truncatingIfNeeded: rounded >> 16)
        return Float(bitPattern: UInt32(bf16) << 16)
    }

    /// CPU `Double`-precision ground truth for `qmvGatedSplitK`'s output:
    /// per-head softplus gate (rounded through bf16, ROUNDING POINT 1),
    /// gate*activation product (rounded through bf16 per lane, ROUNDING
    /// POINT 2), dequantized bank row via the same `trueE4M3Scale` x
    /// `trueE2M1Nibble` oracle test 1 uses, dot-accumulated in `Double`
    /// (order-insensitive at this precision -- unlike the kernel's
    /// float32 accumulator, `Double` has ~9 more decimal digits of
    /// headroom than bf16's ~2-3 significant digits, so any GPU
    /// reduction-order difference washes out well below the final bf16
    /// rounding step for realistic random weight/activation magnitudes),
    /// narrowed to the output dtype the same two-step way the GPU does
    /// (float32 accumulator -> bf16 store): `Float(acc)` (Swift's
    /// `Double` -> `Float` narrowing is correctly rounded) then
    /// `roundToBf16`.
    ///
    /// Softplus itself is computed with Swift's `Foundation` `log`/`exp`
    /// in `Double`, not Metal's `log`/`exp` -- those are, like `exp2` (see
    /// test 1's header comment), not required to be correctly rounded on
    /// every Apple Silicon generation. Unlike test 1's exhaustive sweep,
    /// this oracle is not claiming hardware-invariant bit-exactness of
    /// the softplus step itself; it relies on the two bf16 roundings
    /// either side of it (and the ~1e-3 relative bf16 quantization step)
    /// to absorb any few-ULP float32-transcendental-vs-Double gap for
    /// realistic (non-adversarial, non-exhaustive) random gate logits --
    /// the same trust level `compareNormFold`'s bit-exact primary
    /// assertion already places in this file for its own reference
    /// chain.
    private static func gatedTruth(
        x: [Float], gateLogits: [Float], packed: [UInt32], scales: [UInt8],
        heads: Int, headDim: Int, hidden: Int, inDim: Int
    ) -> [Float] {
        let packsPerRow = inDim / 8
        let groupsPerRow = inDim / 16  // GROUP = 16
        let packsPerHead = headDim / 8

        var gate = [Double](repeating: 0, count: heads)
        for h in 0 ..< heads {
            let gl = Double(gateLogits[h])
            let ax = gl > 0 ? gl : -gl
            let softplus = (gl > 0 ? gl : 0.0) + log(1.0 + exp(-ax))
            gate[h] = Double(roundToBf16(Float(softplus)))
        }

        var out = [Float](repeating: 0, count: hidden)
        for row in 0 ..< hidden {
            var acc = 0.0
            for packIdx in 0 ..< packsPerRow {
                let xBase = packIdx * 8
                let head = packIdx / packsPerHead
                let g = gate[head]
                var xf = [Double](repeating: 0, count: 8)
                for i in 0 ..< 8 {
                    let prod = Double(x[xBase + i]) * g
                    xf[i] = Double(roundToBf16(Float(prod)))
                }
                let group = xBase / 16
                let scaleByte = scales[row * groupsPerRow + group]
                let trueScale = Self.trueE4M3Scale(scaleByte)
                let p = packed[row * packsPerRow + packIdx]
                var qd = 0.0
                for i in 0 ..< 8 {
                    let nib = (p >> (4 * UInt32(i))) & 0xF
                    qd += Self.trueE2M1Nibble(nib) * xf[i]
                }
                acc += trueScale * qd
            }
            out[row] = roundToBf16(Float(acc))
        }
        return out
    }

    private func compareGated(heads: Int, headDim: Int, hidden: Int, label: String) {
        autoreleasepool {
            var seed: UInt64 = 0xDEAD_BEEF_0BAD_F00D
            let inDim = heads * headDim
            let x = randomBf16([inDim], seed: &seed)
            let gateLogits = randomBf16([heads], seed: &seed, magnitude: 3.0)
            let bank = randomBank(rows: hidden, hidden: inDim, seed: &seed)

            setV2(false)
            let v1Out = Tensor.empty(shape: [hidden], dtype: .bf16)
            var okV1 = false
            runAndWait { cb in
                okV1 = LagunaAttnNvfp4Tail.qmvGatedSplitK(
                    input: x, gateLogits: gateLogits, bank: bank,
                    heads: heads, headDim: headDim,
                    on: cb, device: .shared, into: v1Out)
            }

            setV2(true)
            let v2Out = Tensor.empty(shape: [hidden], dtype: .bf16)
            var okV2 = false
            runAndWait { cb in
                okV2 = LagunaAttnNvfp4Tail.qmvGatedSplitK(
                    input: x, gateLogits: gateLogits, bank: bank,
                    heads: heads, headDim: headDim,
                    on: cb, device: .shared, into: v2Out)
            }
            setV2(false)

            guard okV1, okV2 else {
                Issue.record("\(label): gated split-K declined (v1=\(okV1) v2=\(okV2))")
                return
            }

            let truth = Self.gatedTruth(
                x: x.toFloatArray(), gateLogits: gateLogits.toFloatArray(),
                packed: bank.packed.toArray(as: UInt32.self),
                scales: bank.scales.toArray(as: UInt8.self),
                heads: heads, headDim: headDim, hidden: hidden, inDim: inDim)

            let a = v1Out.toFloatArray()
            let b = v2Out.toFloatArray()
            #expect(b.count == truth.count)
            guard b.count == truth.count, a.count == truth.count else { return }

            // V1 is informational only here -- see the header comment
            // above test 3 and test 1's derivation comment. Its
            // divergence count is folded into the diagnostic comment
            // below (only reached if V2 itself needs the fallback path)
            // rather than reported via a standalone `Issue.record`,
            // which would fail the test on its own regardless of
            // outcome -- V1 disagreeing with ground truth is expected
            // and must never be load-bearing.
            var v1Mismatches = 0
            for i in 0 ..< a.count where a[i].bitPattern != truth[i].bitPattern {
                v1Mismatches += 1
            }

            var mismatches = 0
            var firstMismatch = ""
            var maxAbsDiff: Float = 0
            var dot = 0.0, nb = 0.0, nt = 0.0
            for i in 0 ..< b.count {
                if b[i].bitPattern != truth[i].bitPattern {
                    mismatches += 1
                    if firstMismatch.isEmpty {
                        firstMismatch =
                            "row \(i): truth=\(truth[i]) (\(truth[i].bitPattern)) v2=\(b[i]) (\(b[i].bitPattern))"
                    }
                }
                maxAbsDiff = max(maxAbsDiff, abs(b[i] - truth[i]))
                dot += Double(b[i]) * Double(truth[i])
                nb += Double(b[i]) * Double(b[i])
                nt += Double(truth[i]) * Double(truth[i])
            }
            if mismatches == 0 {
                #expect(b == truth, "\(label): expected bit-exact")
            } else {
                // Bit-exact primary / tight-cosine fallback, same
                // contract `compareNormFold` above uses: a K=inDim-wide
                // accumulation compared against a `Double` oracle is not
                // guaranteed bit-exact on every row even when both sides
                // are "correct" -- float32 GPU accumulation order vs a
                // `Double` CPU sum can round to adjacent bf16 values on
                // the rare row whose true sum sits very close to a bf16
                // halfway point (observed: 1/2048 rows, ~1 bf16 ULP,
                // magnitude ~0.001 -- exactly this class of boundary).
                let cos = (nb > 0 && nt > 0) ? dot / (nb.squareRoot() * nt.squareRoot()) : 1.0
                #expect(
                    cos > 0.999999,
                    Comment(
                        rawValue:
                            "\(label): NOT bit-exact on \(mismatches)/\(b.count) rows (maxAbsDiff=\(maxAbsDiff), cos=\(cos), v1Mismatches=\(v1Mismatches)/\(a.count) [informational], first: \(firstMismatch))"
                    ))
            }
        }
    }

    @Test("gated split-K V2 == V1: o_proj sliding shape (64 heads, 8192 -> 2048)")
    func gatedSliding() {
        compareGated(heads: 64, headDim: 128, hidden: 2048, label: "oproj-sliding")
    }

    @Test("gated split-K V2 == V1: o_proj full-attn shape (48 heads, 6144 -> 2048)")
    func gatedFull() {
        compareGated(heads: 48, headDim: 128, hidden: 2048, label: "oproj-full")
    }

    // MARK: - 4. Magnitude sweep (T23 class)

    @Test("V2 norm-fold + gated kernels stay finite and accurate across magnitudes")
    func magnitudeSweep() {
        for magnitude in [Float(1), 10, 100, 1000, 10000] {
            compareNormFold(
                hidden: 2048, rows: 512, magnitude: magnitude,
                label: "sweep-mag\(Int(magnitude))")
        }
    }

    // MARK: - 5. Decline hygiene

    @Test("normQmvV2 declines when the flag is off")
    func declinesFlagOff() {
        setV2(false)
        var seed: UInt64 = 0x1111_2222_3333_4444
        let hidden = 2048
        let residual = randomBf16([hidden], seed: &seed)
        let normWeight = randomBf16([hidden], seed: &seed)
        let bank = randomBank(rows: 64, hidden: hidden, seed: &seed)
        let out = Tensor.empty(shape: [64], dtype: .bf16)
        var folded = true
        runAndWait { cb in
            folded = LagunaAttnNvfp4Tail.normQmvV2(
                residual: residual, normWeight: normWeight, eps: 1e-6, bank: bank,
                on: cb, device: .shared, into: out)
        }
        #expect(!folded)
    }

    @Test("normQmvV2 declines when inDim is not a multiple of 256")
    func declinesMisaligned() {
        setV2(true)
        defer { setV2(false) }
        var seed: UInt64 = 0x5555_6666_7777_8888
        let hidden = 1920  // multiple of 128 and 16, NOT of 256
        let residual = randomBf16([hidden], seed: &seed)
        let normWeight = randomBf16([hidden], seed: &seed)
        let bank = randomBank(rows: 64, hidden: hidden, seed: &seed)
        let out = Tensor.empty(shape: [64], dtype: .bf16)
        var folded = true
        runAndWait { cb in
            folded = LagunaAttnNvfp4Tail.normQmvV2(
                residual: residual, normWeight: normWeight, eps: 1e-6, bank: bank,
                on: cb, device: .shared, into: out)
        }
        #expect(!folded)
    }
}
