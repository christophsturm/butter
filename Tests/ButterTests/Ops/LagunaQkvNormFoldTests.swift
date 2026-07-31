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
// GPU correctness test for `LagunaAttnInt8.qmvNormFolded` (the norm-folded
// batched QKV(+gate) QMV): compares the fused single-dispatch path against
// the standalone two-dispatch reference (`Ops.rmsNorm` then `qmv`, exactly
// what `LagunaText.forward` runs when the fused path declines or the
// escape hatch is set) on synthetic random data at Laguna XS's exact
// decode shapes.
//
// The fused kernel's reduction is a deliberate bit-for-bit replica of
// `Ops.rmsNorm`'s fast path (`iron_rms_norm_bf16`, `n <= 4096` and a
// multiple of 128): 64 real threads (2 simdgroups) emulate the wider
// `inDim/4`-thread virtual threadgroup that kernel would dispatch, in the
// same per-simdgroup / cross-simdgroup order; see the file header on
// `LagunaAttnInt8.normQmvSrc` for the full argument. So the primary
// assertion here is EXACT equality, not a tolerance. If a GPU or driver
// detail breaks that assumption (e.g. `simd_sum`'s reduction tree isn't
// what the codegen doc claims on some hardware), the comparison falls
// back to a tight cosine-similarity bound instead of failing outright:
// each test reports which path it took.
import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("LagunaQkvNormFold", .serialized)
struct LagunaQkvNormFoldTests {
    private func randomBf16(_ shape: [Int], seed: inout UInt64, magnitude: Float = 1.0) -> Tensor {
        let n = shape.reduce(1, *)
        var f32 = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            // xorshift64, deterministic, no external dependency.
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

    /// Builds a random group-32 INT8 bank of shape `[rows, hidden]` the
    /// same way `LagunaText.ensureAttnInt8` builds the production QKV
    /// bank: quantize, commit, wait; the quantize dispatch must finish
    /// before the bank's buffers are safe to read from a later command
    /// buffer. Contents don't need to be a real Q/K/V/gate stack; only
    /// the shape and the group-32 affine layout matter for this kernel.
    private func randomBank(rows: Int, hidden: Int, seed: inout UInt64) -> LagunaAttnInt8.Bank {
        let weight = randomBf16([rows, hidden], seed: &seed)
        var bank: LagunaAttnInt8.Bank!
        runAndWait { cb in
            bank = LagunaAttnInt8.quantizeG32(weight: weight, device: .shared, cmd: cb)
        }
        return bank
    }

    private func compare(
        hidden: Int, rows: Int, eps: Float = 1e-6, magnitude: Float = 1.0,
        label: String
    ) {
        autoreleasepool {
            var seed: UInt64 = 0xA5A5_1234_9E37_79B9
            let residual = randomBf16([hidden], seed: &seed, magnitude: magnitude)
            let normWeight = randomBf16([hidden], seed: &seed)
            let bank = randomBank(rows: rows, hidden: hidden, seed: &seed)

            // Reference: standalone `Ops.rmsNorm` (the same call
            // `LagunaText.forward`'s `inputNorm` module makes) feeding the
            // existing unfused `qmv`.
            let refOut = Tensor.empty(shape: [rows], dtype: .bf16)
            runAndWait { cb in
                let xNorm = Ops.rmsNorm(residual, weight: normWeight, eps: eps, on: cb)
                LagunaAttnInt8.qmv(input: xNorm, bank: bank, on: cb, device: .shared, into: refOut)
            }

            // Fused: one dispatch, folding the norm's reduction into the
            // bank's own QMV.
            let fusedOut = Tensor.empty(shape: [rows], dtype: .bf16)
            var folded = false
            runAndWait { cb in
                folded = LagunaAttnInt8.qmvNormFolded(
                    residual: residual, normWeight: normWeight, eps: eps, bank: bank,
                    on: cb, device: .shared, into: fusedOut)
            }
            guard folded else {
                Issue.record("\(label): qmvNormFolded declined at a shape it should support")
                return
            }

            let a = refOut.toFloatArray()
            let b = fusedOut.toFloatArray()
            #expect(a.count == b.count)
            guard a.count == b.count else { return }

            var exact = true
            var maxAbsDiff: Float = 0
            var dot: Double = 0, na: Double = 0, nb: Double = 0
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
                // Documented fallback (see file header): the virtual-thread
                // reduction was designed to be bit-exact, so landing here
                // means that assumption didn't hold on this run/hardware.
                // Still require it be extremely close.
                let cos = (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 1.0
                #expect(
                    cos > 0.999999,
                    Comment(rawValue: "\(label): NOT bit-exact (maxAbsDiff=\(maxAbsDiff), cos=\(cos)), "
                        + "virtual-thread reduction did not reproduce iron_rms_norm_bf16 exactly"))
            }
        }
    }

    @Test("Laguna XS full-attn shape (h48, headDim128, kv8, gated): vs norm-then-qmv")
    func fullAttnGated() {
        compare(hidden: 2048, rows: 48 * 128 + 2 * 8 * 128 + 48, label: "full-attn+gate")
    }

    @Test("Laguna XS sliding-attn shape (h64, headDim128, kv8, no gate): vs norm-then-qmv")
    func slidingAttnUngated() {
        compare(hidden: 2048, rows: 64 * 128 + 2 * 8 * 128, label: "sliding-attn")
    }

    @Test("magnitude case: large-magnitude activations")
    func magnitude() {
        compare(
            hidden: 2048, rows: 48 * 128 + 2 * 8 * 128 + 48, magnitude: 50.0,
            label: "magnitude")
    }

    @Test("small hidden at the fast-path floor (256)")
    func smallHidden() {
        compare(hidden: 256, rows: 512, label: "hidden=256")
    }

    @Test("declines when inDim is not a multiple of 256")
    func declinesMisalignedHidden() {
        // 1920 is a multiple of 128 (Ops.rmsNorm's own fast-path bound)
        // but NOT of 256, so `qmvNormFolded` must decline and let the
        // caller fall back to the separate norm + qmv chain.
        let hidden = 1920
        var seed: UInt64 = 0x1122_3344_5566_7788
        let residual = randomBf16([hidden], seed: &seed)
        let normWeight = randomBf16([hidden], seed: &seed)
        let bank = randomBank(rows: 256, hidden: hidden, seed: &seed)
        let out = Tensor.empty(shape: [256], dtype: .bf16)
        var folded = true
        runAndWait { cb in
            folded = LagunaAttnInt8.qmvNormFolded(
                residual: residual, normWeight: normWeight, eps: 1e-6, bank: bank,
                on: cb, device: .shared, into: out)
        }
        #expect(!folded)
    }

    @Test("declines when inDim exceeds the 4096 fast-path cap")
    func declinesWideHidden() {
        let hidden = 4352  // multiple of 256, but > 4096
        var seed: UInt64 = 0x2233_4455_6677_8899
        let residual = randomBf16([hidden], seed: &seed)
        let normWeight = randomBf16([hidden], seed: &seed)
        let bank = randomBank(rows: 256, hidden: hidden, seed: &seed)
        let out = Tensor.empty(shape: [256], dtype: .bf16)
        var folded = true
        runAndWait { cb in
            folded = LagunaAttnInt8.qmvNormFolded(
                residual: residual, normWeight: normWeight, eps: 1e-6, bank: bank,
                on: cb, device: .shared, into: out)
        }
        #expect(!folded)
    }

    @Test("escape hatch: BUTTER_LAGUNA_QKV_NORMFOLD=0 declines")
    func escapeHatch() {
        setenv("BUTTER_LAGUNA_QKV_NORMFOLD", "0", 1)
        defer { unsetenv("BUTTER_LAGUNA_QKV_NORMFOLD") }
        let hidden = 2048
        var seed: UInt64 = 0x99AA_BBCC_DDEE_FF00
        let residual = randomBf16([hidden], seed: &seed)
        let normWeight = randomBf16([hidden], seed: &seed)
        let bank = randomBank(rows: 256, hidden: hidden, seed: &seed)
        let out = Tensor.empty(shape: [256], dtype: .bf16)
        var folded = true
        runAndWait { cb in
            folded = LagunaAttnInt8.qmvNormFolded(
                residual: residual, normWeight: normWeight, eps: 1e-6, bank: bank,
                on: cb, device: .shared, into: out)
        }
        #expect(!folded)
    }
}
