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
// GPU correctness test for `LagunaSdpaDecodeGqa2` (T24 "TASK 2"): compares
// the GQA-paired + ALPHASKIP decode SDPA against `Ops.sdpaDecode` (the
// stock, proven iron-DSL-generated kernel) on synthetic random data at
// Laguna XS's exact decode shapes (headDim=128; nHeads=48/heads_per_group=6
// and nHeads=64/heads_per_group=8, both nKVHeads=8). Both kernels compute
// the same online-softmax formulas in the same order (see file header of
// `LagunaSdpaDecodeGqa2.swift` for the derivation); this asserts they
// land within a tight, non-bit-exact tolerance (float32-internal GPU math
// with different source text can reorder a handful of ops even when
// algebraically identical) plus dense cosine similarity per head.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("LagunaSdpaDecodeGqa2")
struct LagunaSdpaDecodeGqa2Tests {
    private func randomBf16(_ shape: [Int], seed: inout UInt64) -> Tensor {
        let n = shape.reduce(1, *)
        var f32 = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            // xorshift64, deterministic, no external dependency.
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            let u = Double(seed % 2_000_001) / 1_000_000.0 - 1.0  // [-1, 1)
            f32[i] = Float(u)
        }
        let f32T = Tensor.empty(shape: shape, dtype: .f32)
        f32T.copyIn(from: f32)
        let bf16T = Tensor.empty(shape: shape, dtype: .bf16)
        runAndWait { cb in Ops.castF32ToBf16(f32T, into: bf16T, on: cb) }
        return bf16T
    }

    private func compare(
        nHeads: Int, nKVHeads: Int, nKV: Int, kvStride: Int,
        sinkEnd: Int = 0, windowStart: Int = 0
    ) {
        autoreleasepool {
            let headDim = 128
            var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
            let q = randomBf16([nHeads, headDim], seed: &seed)
            let k = randomBf16([nKVHeads, kvStride, headDim], seed: &seed)
            let v = randomBf16([nKVHeads, kvStride, headDim], seed: &seed)
            let scale: Float = 1.0 / Float(headDim).squareRoot()

            #expect(
                LagunaSdpaDecodeGqa2.canFuse(
                    headDim: headDim, nHeads: nHeads, nKVHeads: nKVHeads, dtype: .bf16))

            var stock: Tensor!
            runAndWait { cb in
                stock = Ops.sdpaDecode(
                    q: q, k: k, v: v,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    nKV: nKV, kvStride: kvStride, scale: scale, on: cb,
                    sinkEnd: sinkEnd, windowStart: windowStart)
            }

            var fused: Tensor?
            runAndWait { cb in
                fused = LagunaSdpaDecodeGqa2.apply(
                    q: q, k: k, v: v,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    nKV: nKV, kvStride: kvStride, scale: scale,
                    on: cb, device: .shared,
                    sinkEnd: sinkEnd, windowStart: windowStart)
            }
            guard let fused else {
                Issue.record("LagunaSdpaDecodeGqa2.apply declined at a shape it should support")
                return
            }

            let a = stock.toFloatArray()
            let b = fused.toFloatArray()
            #expect(a.count == b.count)
            var maxAbsDiff: Float = 0
            var dot: Double = 0, na: Double = 0, nb: Double = 0
            for i in 0 ..< a.count {
                maxAbsDiff = max(maxAbsDiff, abs(a[i] - b[i]))
                dot += Double(a[i]) * Double(b[i])
                na += Double(a[i]) * Double(a[i])
                nb += Double(b[i]) * Double(b[i])
            }
            let cos = (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 1.0
            let ctx =
                "nHeads=\(nHeads) nKV=\(nKV) sinkEnd=\(sinkEnd) windowStart=\(windowStart)"
            #expect(maxAbsDiff < 0.05, "\(ctx): maxAbsDiff=\(maxAbsDiff)")
            #expect(cos > 0.9999, "\(ctx): cos=\(cos)")
        }
    }

    @Test("h48/kv8 (full-attn head count): dense, small nKV")
    func h48Small() { compare(nHeads: 48, nKVHeads: 8, nKV: 5, kvStride: 64) }

    @Test("h48/kv8: dense, mid nKV")
    func h48Mid() { compare(nHeads: 48, nKVHeads: 8, nKV: 130, kvStride: 256) }

    @Test("h64/kv8 (sliding-attn head count): dense, small nKV")
    func h64Small() { compare(nHeads: 64, nKVHeads: 8, nKV: 5, kvStride: 64) }

    @Test("h64/kv8: dense, mid nKV")
    func h64Mid() { compare(nHeads: 64, nKVHeads: 8, nKV: 130, kvStride: 256) }

    @Test("h64/kv8: sink + sliding window")
    func h64SinkWindow() {
        compare(nHeads: 64, nKVHeads: 8, nKV: 100, kvStride: 256, sinkEnd: 4, windowStart: 50)
    }

    @Test("h48/kv8: sink + sliding window")
    func h48SinkWindow() {
        compare(nHeads: 48, nKVHeads: 8, nKV: 100, kvStride: 256, sinkEnd: 4, windowStart: 50)
    }

    @Test("nKV == 1 edge case")
    func singlePosition() {
        compare(nHeads: 48, nKVHeads: 8, nKV: 1, kvStride: 8)
    }

    // MARK: - v2 (pipelined + unrolled) vs v1 (rolled loop)
    //
    // v2 (`LagunaSdpaDecodeGqa2.swift` file header point 3) is a straight
    // reordering + manual unroll of v1's per-position math: same FP op
    // sequence, same ALPHASKIP branch, just with loads hoisted two
    // positions ahead and the inner loops written out by hand. That
    // means v1 and v2 aren't just close, they should be bit-identical on
    // the same input, so this compares them directly rather than against
    // `Ops.sdpaDecode` (unlike `compare` above, which tolerates the small
    // reordering-vs-the-generated-kernel drift that's expected there).
    // `BUTTER_LAGUNA_SDPA_GQA2_V2` forces each variant in turn on the
    // same q/k/v tensors.

    private func compareV1VsV2(
        nHeads: Int, nKVHeads: Int, nKV: Int, kvStride: Int,
        sinkEnd: Int = 0, windowStart: Int = 0
    ) {
        autoreleasepool {
            let headDim = 128
            var seed: UInt64 = 0xC0FF_EE15_BEEF_D00D
            let q = randomBf16([nHeads, headDim], seed: &seed)
            let k = randomBf16([nKVHeads, kvStride, headDim], seed: &seed)
            let v = randomBf16([nKVHeads, kvStride, headDim], seed: &seed)
            let scale: Float = 1.0 / Float(headDim).squareRoot()

            #expect(
                LagunaSdpaDecodeGqa2.canFuse(
                    headDim: headDim, nHeads: nHeads, nKVHeads: nKVHeads, dtype: .bf16))

            defer { unsetenv("BUTTER_LAGUNA_SDPA_GQA2_V2") }

            setenv("BUTTER_LAGUNA_SDPA_GQA2_V2", "0", 1)
            var v1Out: Tensor?
            runAndWait { cb in
                v1Out = LagunaSdpaDecodeGqa2.apply(
                    q: q, k: k, v: v,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    nKV: nKV, kvStride: kvStride, scale: scale,
                    on: cb, device: .shared,
                    sinkEnd: sinkEnd, windowStart: windowStart)
            }
            guard let v1Out else {
                Issue.record("v1 kernel declined at a shape it should support")
                return
            }

            setenv("BUTTER_LAGUNA_SDPA_GQA2_V2", "1", 1)
            var v2Out: Tensor?
            runAndWait { cb in
                v2Out = LagunaSdpaDecodeGqa2.apply(
                    q: q, k: k, v: v,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    nKV: nKV, kvStride: kvStride, scale: scale,
                    on: cb, device: .shared,
                    sinkEnd: sinkEnd, windowStart: windowStart)
            }
            guard let v2Out else {
                Issue.record("v2 kernel declined at a shape it should support")
                return
            }

            let a = v1Out.toFloatArray()
            let b = v2Out.toFloatArray()
            let ctx =
                "nHeads=\(nHeads) nKV=\(nKV) sinkEnd=\(sinkEnd) windowStart=\(windowStart)"
            #expect(a.count == b.count, Comment(rawValue: ctx))
            var mismatches = 0
            for i in 0 ..< min(a.count, b.count) where a[i] != b[i] {
                mismatches += 1
                if mismatches <= 3 {
                    Issue.record("\(ctx): v1[\(i)]=\(a[i]) != v2[\(i)]=\(b[i])")
                }
            }
            #expect(mismatches == 0, "\(ctx): \(mismatches) bit-exact mismatches")
        }
    }

    @Test("v2/h48/kv8: dense, small nKV bit-identical to v1")
    func v2H48Small() { compareV1VsV2(nHeads: 48, nKVHeads: 8, nKV: 5, kvStride: 64) }

    @Test("v2/h48/kv8: dense, mid nKV bit-identical to v1")
    func v2H48Mid() { compareV1VsV2(nHeads: 48, nKVHeads: 8, nKV: 130, kvStride: 256) }

    @Test("v2/h64/kv8: dense, small nKV bit-identical to v1")
    func v2H64Small() { compareV1VsV2(nHeads: 64, nKVHeads: 8, nKV: 5, kvStride: 64) }

    @Test("v2/h64/kv8: dense, mid nKV bit-identical to v1")
    func v2H64Mid() { compareV1VsV2(nHeads: 64, nKVHeads: 8, nKV: 130, kvStride: 256) }

    @Test("v2/h64/kv8: sink + sliding window bit-identical to v1")
    func v2H64SinkWindow() {
        compareV1VsV2(nHeads: 64, nKVHeads: 8, nKV: 100, kvStride: 256, sinkEnd: 4, windowStart: 50)
    }

    @Test("v2/h48/kv8: sink + sliding window bit-identical to v1")
    func v2H48SinkWindow() {
        compareV1VsV2(nHeads: 48, nKVHeads: 8, nKV: 100, kvStride: 256, sinkEnd: 4, windowStart: 50)
    }

    @Test("v2: nKV == 1 edge case bit-identical to v1")
    func v2SinglePosition() {
        compareV1VsV2(nHeads: 48, nKVHeads: 8, nKV: 1, kvStride: 8)
    }

    @Test("v2: odd pipeline-trip-count (one 2-deep trip + single-position tail) bit-identical to v1")
    func v2OddTripCount() {
        // nKV=65, dense: lane sg=0's phase-1 range is [0, 65), which is
        // 3 rolled-loop steps (0, 32, 64) -- an odd count, so v2 takes
        // exactly one paired trip (0, 32) then falls into the
        // single-position tail for 64. Exercises the tail path deliberately
        // rather than incidentally.
        compareV1VsV2(nHeads: 64, nKVHeads: 8, nKV: 65, kvStride: 128)
    }

    @Test("v2: long KV (many pipelined trips per lane) bit-identical to v1")
    func v2LongKV() {
        compareV1VsV2(nHeads: 64, nKVHeads: 8, nKV: 4097, kvStride: 4097)
    }
}
