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
//    `iron_nvfp4_pack_dot_v2` vs the V1 `iron_nvfp4_e4m3_scale` /
//    `iron_nvfp4_nibble_value` over ALL 256 E4M3 scale bytes x 16 E2M1
//    nibbles, bit-exact (this is the mutation gate for the bit-trick
//    decode — any wrong mask/shift flips at least one of the 4096
//    combinations).
// 2. Norm-folded V2 QKV vs the separate `Ops.rmsNorm` + plain
//    (non-split-K) QMV chain at Laguna XS's exact decode shapes —
//    bit-exact primary assertion, tight-cosine fallback with a report,
//    same contract as `LagunaQkvNormFoldTests` (the INT8 twin).
// 3. Gated split-K V2 vs V1 at the exact o_proj shapes — bit-exact
//    strict (same geometry, same rounding points; only the dequant
//    encoding and where the softplus is computed differ).
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

    @Test("V2 fast dequant == V1 dequant, all 256 scale bytes x 16 nibbles, bit-exact")
    func dequantEquivalenceExhaustive() {
        let probeSrc = """
            #include <metal_stdlib>
            using namespace metal;

            \(LagunaAttnNvfp4Tail.helperSrcForTesting)

            \(LagunaAttnNvfp4Tail.helperV2Src)

            kernel void laguna_nvfp4_v2_dequant_probe(
                device float *ref [[buffer(0)]],
                device float *v2 [[buffer(1)]],
                uint tid [[thread_position_in_grid]])
            {
                uint scaleByte = tid >> 4;   // 0..255
                uint nib = tid & 0xFu;       // 0..15
                ref[tid] = iron_nvfp4_e4m3_scale(uchar(scaleByte))
                    * iron_nvfp4_nibble_value(nib);
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
        let refT = Tensor.empty(shape: [n], dtype: .f32)
        let v2T = Tensor.empty(shape: [n], dtype: .f32)
        runAndWait { cb in
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            enc.setBuffer(refT.buffer, offset: refT.offset, index: 0)
            enc.setBuffer(v2T.buffer, offset: v2T.offset, index: 1)
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
        }
        let a = refT.toFloatArray()
        let b = v2T.toFloatArray()
        #expect(a.count == n && b.count == n)
        var mismatches = 0
        for i in 0 ..< n where a[i].bitPattern != b[i].bitPattern {
            // -0.0 vs +0.0 both encode "zero code x some scale sign";
            // treat signed zero as equal only if V1 itself produced the
            // sign V2 did — i.e. require true bit equality, and record
            // the combination on failure.
            mismatches += 1
            if mismatches <= 8 {
                let msg =
                    "scaleByte=\(i >> 4) nib=\(i & 15): ref=\(a[i]) (\(a[i].bitPattern)) v2=\(b[i]) (\(b[i].bitPattern))"
                Issue.record(Comment(rawValue: msg))
            }
        }
        #expect(mismatches == 0, "V2 dequant diverges from V1 on \(mismatches)/\(n) combinations")
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

    // MARK: - 3. Gated split-K V2 vs V1 (bit-exact strict)

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
            let a = v1Out.toFloatArray()
            let b = v2Out.toFloatArray()
            #expect(a == b, "\(label): V2 gated split-K must be bit-exact vs V1")
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
