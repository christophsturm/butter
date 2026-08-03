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
// GPU correctness test for `LagunaFusedSlidingAttnStep`: compares its
// single-dispatch output against the stock 3-dispatch chain
// (LagunaFusedQKNormRoPE -> KVCache.appendOnGPU -> LagunaSdpaDecodeGqa2)
// on two IDENTICAL, independently-populated caches, isolated from any
// real generation loop so a divergence here can't be explained by
// cache-history drift between the two paths.
//
// This suite is what localized the original BUTTER_LAGUNA_FUSED_ATTN_STEP
// byte-identical-gate failure (real `butter generate` output diverging
// from stock after ~140 decode tokens): h48/kv8 (headsPerGroup=6)
// looked bit-exact with a handful of random seeds, but h64/kv8
// (headsPerGroup=8, Laguna XS's actual sliding-layer shape per
// `num_attention_heads_per_layer`) reproducibly showed up to ~3e-5
// per-element gaps. Root cause: fp-contraction (fma) applied
// differently by the Metal compiler to the textually-identical
// scan+combine body in this kernel's separately-compiled instance vs
// `LagunaSdpaDecodeGqa2`/`LagunaFusedQKNormRoPE`'s own compiled
// kernels, even under those kernels' existing source-level
// anti-reassociation precautions (ALPHASKIP branch, "do not
// reassociate" comments).
//
// RESCOPED fix (see `LagunaFusedSlidingAttnStep.swift`'s "Numerics"
// header section): `fastMathEnabled = false` stays ONLY on this
// kernel's own PSO. The shared stock kernels
// (`LagunaFusedQKNormRoPE`/`LagunaSdpaDecodeGqa2`) were reverted to
// Metal's default `fastMathEnabled = true` -- an earlier version of
// this fix touched them too, which silently changed the DEFAULT
// (flag-OFF) decode digest for every caller; that is not acceptable
// for an opt-in feature. Consequence: this kernel is no longer
// expected to be bit-exact against the stock chain (stock still
// allows fp-contraction, this kernel doesn't), so the assertion below
// is tolerance-based -- maxAbsDiff < 5e-3 and cosine > 0.99999 --
// mirroring the existing convention `LagunaSdpaDecodeGqa2Tests` already
// uses for comparing two independently-compiled kernels computing the
// same math (that suite's own bound is the looser maxAbsDiff < 0.05).
// The `h48`-shape "pass" seen during the original investigation was
// just data-dependent luck absorbed by bf16 rounding, not evidence the
// gap was h64-shape-specific -- both shapes are kept in the matrix
// below as a regression guard, plus a past-window-wrap case (history
// longer than the cache's ring `maxSize`) since that's the shape the
// original bug report worried about first.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

// .serialized: every test in this suite mutates the shared process-global
// BUTTER_LAGUNA_FUSED_ATTN_STEP env var (setenv/unsetenv); running tests
// in parallel races that mutation across tests (one test's canFuse-should-
// default-OFF check can observe another concurrently-running test's "1").
@Suite("LagunaFusedSlidingAttnStep", .serialized)
struct LagunaFusedSlidingAttnStepTests {
    private func randomBf16(_ shape: [Int], seed: inout UInt64) -> Tensor {
        let n = shape.reduce(1, *)
        var f32 = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            let u = Double(seed % 2_000_001) / 1_000_000.0 - 1.0
            f32[i] = Float(u)
        }
        let f32T = Tensor.empty(shape: shape, dtype: .f32)
        f32T.copyIn(from: f32)
        let bf16T = Tensor.empty(shape: shape, dtype: .bf16)
        runAndWait { cb in Ops.castF32ToBf16(f32T, into: bf16T, on: cb) }
        return bf16T
    }

    private func compare(
        nHeads: Int, nKVHeads: Int, historyLen: Int, maxSize: Int
    ) {
        autoreleasepool {
            let headDim = 128
            let eps: Float = 1e-6
            let theta: Float = 10_000
            let scale: Float = 1.0 / Float(headDim).squareRoot()
            var seed: UInt64 = 0xABCD_1234_5678_EF01

            let qWeight = randomBf16([headDim], seed: &seed)
            let kWeight = randomBf16([headDim], seed: &seed)

            let cacheStock = KVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSize, dtype: .bf16,
                eviction: .window(maxSize: maxSize, keep: 0))
            let cacheFused = KVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSize, dtype: .bf16,
                eviction: .window(maxSize: maxSize, keep: 0))

            // Identical pre-existing history on both caches (simulates prior
            // decode steps already committed): same random K/V rows, written
            // via the same appendOnGPU path both stock and fused rely on for
            // everything except the CURRENT token.
            for _ in 0 ..< historyLen {
                var hseed = seed
                let kRow = randomBf16([nKVHeads, headDim], seed: &hseed)
                let vRow = randomBf16([nKVHeads, headDim], seed: &hseed)
                seed = hseed
                runAndWait { cb in cacheStock.appendOnGPU(kFlat: kRow, vFlat: vRow, on: cb) }
                runAndWait { cb in cacheFused.appendOnGPU(kFlat: kRow, vFlat: vRow, on: cb) }
            }
            // `.window` eviction saturates `length` at `maxSize` once the
            // ring has wrapped -- `min(historyLen, maxSize)`, not
            // `historyLen`, is the correct post-prefill expectation.
            let expectedLength = min(historyLen, maxSize)
            #expect(cacheStock.length == expectedLength)
            #expect(cacheFused.length == expectedLength)

            // The CURRENT token's raw QKV -- identical inputs to both paths.
            let rawQ = randomBf16([nHeads * headDim], seed: &seed)
            let rawK = randomBf16([nKVHeads * headDim], seed: &seed)
            let rawV = randomBf16([nKVHeads * headDim], seed: &seed)
            let position = historyLen

            #expect(
                LagunaFusedSlidingAttnStep.canFuse(
                    isSliding: true, rotaryDim: headDim, headDim: headDim,
                    nHeads: nHeads, nKVHeads: nKVHeads, dtype: .bf16) == false,
                "canFuse should require BUTTER_LAGUNA_FUSED_ATTN_STEP=1 (default OFF)")

            setenv("BUTTER_LAGUNA_FUSED_ATTN_STEP", "1", 1)
            defer { unsetenv("BUTTER_LAGUNA_FUSED_ATTN_STEP") }

            #expect(
                LagunaFusedSlidingAttnStep.canFuse(
                    isSliding: true, rotaryDim: headDim, headDim: headDim,
                    nHeads: nHeads, nKVHeads: nKVHeads, dtype: .bf16))

            // ── Stock chain ──────────────────────────────────────────
            var stockOut: Tensor!
            runAndWait { cb in
                let fused = LagunaFusedQKNormRoPE.apply(
                    rawQ: rawQ, rawK: rawK, qWeight: qWeight, kWeight: kWeight,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    position: position, thetaBase: theta, eps: eps,
                    on: cb, device: .shared)
                let q2d = fused.q.reshaped(to: [nHeads, headDim])
                let k2d = fused.k.reshaped(to: [nKVHeads, headDim])
                let v2d = rawV.reshaped(to: [nKVHeads, headDim])
                cacheStock.appendOnGPU(kFlat: k2d, vFlat: v2d, on: cb)
                let (cacheK, cacheV) = cacheStock.prepareForAttention(on: cb)
                let (sinkEnd, windowStart) = cacheStock.sdpaSinkWindow(nKV: cacheStock.length)
                stockOut =
                    (LagunaSdpaDecodeGqa2.apply(
                        q: q2d, k: cacheK, v: cacheV,
                        nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                        nKV: cacheStock.length, kvStride: cacheStock.maxSeq, scale: scale,
                        on: cb, device: .shared,
                        sinkEnd: sinkEnd, windowStart: windowStart)
                    ?? Ops.sdpaDecode(
                        q: q2d, k: cacheK, v: cacheV,
                        nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                        nKV: cacheStock.length, kvStride: cacheStock.maxSeq,
                        scale: scale, on: cb,
                        sinkEnd: sinkEnd, windowStart: windowStart))
                    .reshaped(to: [nHeads * headDim])
            }

            // ── Fused single-dispatch step ───────────────────────────
            var fusedOut: Tensor!
            runAndWait { cb in
                fusedOut = LagunaFusedSlidingAttnStep.apply(
                    rawQ: rawQ, rawK: rawK, rawV: rawV,
                    qWeight: qWeight, kWeight: kWeight,
                    nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    position: position, thetaBase: theta, eps: eps,
                    cache: cacheFused, scale: scale,
                    on: cb, device: .shared)
            }

            #expect(cacheStock.length == cacheFused.length)

            let a = stockOut.toFloatArray()
            let b = fusedOut.toFloatArray()
            #expect(a.count == b.count)
            var maxAbsDiff: Float = 0
            var maxAbsDiffIdx = -1
            var dot: Double = 0, na: Double = 0, nb: Double = 0
            for i in 0 ..< min(a.count, b.count) {
                let d = abs(a[i] - b[i])
                if d > maxAbsDiff {
                    maxAbsDiff = d
                    maxAbsDiffIdx = i
                }
                dot += Double(a[i]) * Double(b[i])
                na += Double(a[i]) * Double(a[i])
                nb += Double(b[i]) * Double(b[i])
            }
            let cos = (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 1.0
            let headIdx = maxAbsDiffIdx >= 0 ? maxAbsDiffIdx / headDim : -1
            let ctx =
                "nHeads=\(nHeads) nKVHeads=\(nKVHeads) historyLen=\(historyLen) maxSize=\(maxSize) maxAbsDiff=\(maxAbsDiff) at idx \(maxAbsDiffIdx) (head \(headIdx)) cos=\(cos)"
            #expect(maxAbsDiff < 5e-3, "\(ctx)")
            #expect(cos > 0.99999, "\(ctx)")
        }
    }

    @Test("h48/kv8: single history row (position 1, no other-row reads)")
    func h48Pos1() { compare(nHeads: 48, nKVHeads: 8, historyLen: 1, maxSize: 64) }

    @Test("h48/kv8: small history, dense reads")
    func h48Small() { compare(nHeads: 48, nKVHeads: 8, historyLen: 5, maxSize: 64) }

    @Test("h48/kv8: mid history")
    func h48Mid() { compare(nHeads: 48, nKVHeads: 8, historyLen: 130, maxSize: 256) }

    @Test("h48/kv8: history past the window (ring has wrapped)")
    func h48PastWrap() { compare(nHeads: 48, nKVHeads: 8, historyLen: 90, maxSize: 32) }

    @Test("h64/kv8: mid history")
    func h64Mid() { compare(nHeads: 64, nKVHeads: 8, historyLen: 130, maxSize: 256) }

    @Test("h64/kv8: single history row (position 1, no other-row reads)")
    func h64Pos1() { compare(nHeads: 64, nKVHeads: 8, historyLen: 1, maxSize: 64) }

    @Test("h64/kv8: no history at all (position 0)")
    func h64Pos0() { compare(nHeads: 64, nKVHeads: 8, historyLen: 0, maxSize: 64) }

    @Test("h64/kv8: small history, dense reads")
    func h64Small() { compare(nHeads: 64, nKVHeads: 8, historyLen: 5, maxSize: 64) }

    @Test("h64/kv8: 9 history rows")
    func h64Nine() { compare(nHeads: 64, nKVHeads: 8, historyLen: 9, maxSize: 64) }

    @Test("h64/kv8: history past the window (ring has wrapped)")
    func h64PastWrap() { compare(nHeads: 64, nKVHeads: 8, historyLen: 90, maxSize: 32) }

    @Test("h64/kv8: history several full ring rotations past the window")
    func h64ManyWraps() { compare(nHeads: 64, nKVHeads: 8, historyLen: 513, maxSize: 32) }

    // Sanity check: is GQA2 itself deterministic run-to-run on identical
    // inputs? If NOT, the fused-vs-stock comparison above is chasing a
    // moving target and the "divergence" is inherent to GQA2's own
    // fast-math compiled kernel, not something this fusion introduced.
    @Test("stock GQA2 alone is repeatable on identical inputs (h64)")
    func stockRepeatableH64() {
        autoreleasepool {
            let headDim = 128
            let nHeads = 64, nKVHeads = 8, nKV = 131, kvStride = 256
            var seed: UInt64 = 0x1111_2222_3333_4444
            let q = randomBf16([nHeads, headDim], seed: &seed)
            let k = randomBf16([nKVHeads, kvStride, headDim], seed: &seed)
            let v = randomBf16([nKVHeads, kvStride, headDim], seed: &seed)
            let scale: Float = 1.0 / Float(headDim).squareRoot()

            var run1: Tensor!
            runAndWait { cb in
                run1 = LagunaSdpaDecodeGqa2.apply(
                    q: q, k: k, v: v, nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    nKV: nKV, kvStride: kvStride, scale: scale, on: cb, device: .shared)
            }
            var run2: Tensor!
            runAndWait { cb in
                run2 = LagunaSdpaDecodeGqa2.apply(
                    q: q, k: k, v: v, nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    nKV: nKV, kvStride: kvStride, scale: scale, on: cb, device: .shared)
            }
            let a = run1.toFloatArray()
            let b = run2.toFloatArray()
            var mismatches = 0
            for i in 0 ..< min(a.count, b.count) where a[i] != b[i] {
                mismatches += 1
                if mismatches <= 3 {
                    Issue.record("run1[\(i)]=\(a[i]) != run2[\(i)]=\(b[i])")
                }
            }
            #expect(mismatches == 0, "GQA2 non-deterministic across identical repeated calls: \(mismatches) mismatches")
        }
    }
}
