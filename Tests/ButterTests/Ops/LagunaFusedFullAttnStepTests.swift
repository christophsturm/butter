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
// GPU correctness test for `LagunaFusedFullAttnStep`: compares its
// single-dispatch output against the stock 3-dispatch chain
// (Ops.rmsNormRows + IronKernels.iron_rope_yarn_bf16 -> KVCache.appendOnGPU
// -> LagunaSdpaDecodeGqa2) on two IDENTICAL, independently-populated
// caches, isolated from any real generation loop -- same isolation
// discipline as `LagunaFusedSlidingAttnStepTests`.
//
// Unlike the sliding suite, full-attention layers back an `.unbounded`
// `KVCache` (grow-only, no ring wrap), so both the stock and fused
// comparison caches use `.unbounded` eviction here, and the large-history
// case below sizes `maxSeq` generously (`.unbounded` panics past
// capacity instead of wrapping).
//
// Tolerance convention matches the sliding suite: this kernel compiles
// its own PSO with `fastMathEnabled = false` while the stock chain
// (`Ops.rmsNormRows`, `IronKernels.iron_rope_yarn_bf16`,
// `LagunaSdpaDecodeGqa2`) keeps Metal's default `fastMathEnabled = true`,
// so the two are NOT expected to be bit-exact -- maxAbsDiff < 5e-3 and
// cosine > 0.99999 is the bound, mirroring
// `LagunaFusedSlidingAttnStepTests`.

import Foundation
import IronSwift
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

// .serialized: every test in this suite mutates the shared process-global
// BUTTER_LAGUNA_FUSED_FULL_ATTN_STEP env var (setenv/unsetenv); running
// tests in parallel races that mutation across tests, same concern as
// LagunaFusedSlidingAttnStepTests.
@Suite("LagunaFusedFullAttnStep", .serialized)
struct LagunaFusedFullAttnStepTests {
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

    /// `~/models/laguna-xs-2.1-4bit`'s `rope_parameters.full_attention`
    /// block: rope_theta 500000, YaRN factor 32, original_max_position
    /// 8192, beta_fast 64, beta_slow 1, attention_factor 1.0 (which
    /// matches the default mscale/mscale_all_dim == 1 ratio, so
    /// `Ops.RoPEYaRN.from`'s default mscale args reproduce it exactly).
    /// `headDim: 64` here is the *rotated* dim (rotaryDim), matching how
    /// `LagunaModel.load` computes `yarnBounds` from `rotaryDimFull`, not
    /// the full `headDim` of 128.
    private func yarnParams() -> Ops.RoPEYaRN {
        Ops.RoPEYaRN.from(
            headDim: 64, thetaBase: 500_000, factor: 32,
            betaFast: 64, betaSlow: 1, originalMaxPosition: 8192)
    }

    private func compare(
        nHeads: Int, nKVHeads: Int, historyLen: Int, maxSize: Int
    ) {
        // Same skip-not-fail convention as LagunaMppGemmTests: the fused
        // kernel declines via canFuse on non-target hardware (CI paravirt
        // GPU), so there is nothing to compare there.
        guard Ops.mppAutoCapable else { return }
        autoreleasepool {
            let headDim = 128
            let rotaryDim = 64
            let eps: Float = 1e-6
            let theta: Float = 500_000
            let scale: Float = 1.0 / Float(headDim).squareRoot()
            let yarn = yarnParams()
            var seed: UInt64 = 0x9E37_79B9_7F4A_7C15

            let qWeight = randomBf16([headDim], seed: &seed)
            let kWeight = randomBf16([headDim], seed: &seed)

            // Full-attention layers back an `.unbounded` cache -- grow-only,
            // never wraps. Sized `maxSize` generously for the large-history
            // case since `.unbounded` panics on overflow rather than
            // rotating.
            let cacheStock = KVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSize, dtype: .bf16,
                eviction: .unbounded)
            let cacheFused = KVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSize, dtype: .bf16,
                eviction: .unbounded)

            // Identical pre-existing history on both caches.
            for _ in 0 ..< historyLen {
                var hseed = seed
                let kRow = randomBf16([nKVHeads, headDim], seed: &hseed)
                let vRow = randomBf16([nKVHeads, headDim], seed: &hseed)
                seed = hseed
                runAndWait { cb in cacheStock.appendOnGPU(kFlat: kRow, vFlat: vRow, on: cb) }
                runAndWait { cb in cacheFused.appendOnGPU(kFlat: kRow, vFlat: vRow, on: cb) }
            }
            // `.unbounded` never saturates below `maxSeq` -- length tracks
            // historyLen exactly (unlike the sliding suite's `.window` case).
            #expect(cacheStock.length == historyLen)
            #expect(cacheFused.length == historyLen)

            // The CURRENT token's raw QKV -- identical inputs to both paths.
            let rawQ = randomBf16([nHeads * headDim], seed: &seed)
            let rawK = randomBf16([nKVHeads * headDim], seed: &seed)
            let rawV = randomBf16([nKVHeads * headDim], seed: &seed)
            let position = historyLen

            // Opt-out contract: default (unset) fuses; =0 must decline.
            setenv("BUTTER_LAGUNA_FUSED_FULL_ATTN_STEP", "0", 1)
            #expect(
                LagunaFusedFullAttnStep.canFuse(
                    isSliding: false, rotaryDim: rotaryDim, headDim: headDim,
                    nHeads: nHeads, nKVHeads: nKVHeads, dtype: .bf16) == false,
                "BUTTER_LAGUNA_FUSED_FULL_ATTN_STEP=0 must decline")

            setenv("BUTTER_LAGUNA_FUSED_FULL_ATTN_STEP", "1", 1)
            defer { unsetenv("BUTTER_LAGUNA_FUSED_FULL_ATTN_STEP") }

            #expect(
                LagunaFusedFullAttnStep.canFuse(
                    isSliding: false, rotaryDim: rotaryDim, headDim: headDim,
                    nHeads: nHeads, nKVHeads: nKVHeads, dtype: .bf16))
            // Must never fire for sliding layers, even with otherwise
            // matching shapes -- that's the sibling file's job.
            #expect(
                LagunaFusedFullAttnStep.canFuse(
                    isSliding: true, rotaryDim: rotaryDim, headDim: headDim,
                    nHeads: nHeads, nKVHeads: nKVHeads, dtype: .bf16) == false,
                "must decline for sliding layers")
            // rotaryDim must be a multiple of 8 (Phase-1 lane-partition math).
            #expect(
                LagunaFusedFullAttnStep.canFuse(
                    isSliding: false, rotaryDim: 65, headDim: headDim,
                    nHeads: nHeads, nKVHeads: nKVHeads, dtype: .bf16) == false,
                "rotaryDim not a multiple of 8 must decline")

            // ── Stock chain ──────────────────────────────────────────
            var stockOut: Tensor!
            runAndWait { cb in
                var qTmp = Ops.rmsNormRows(
                    rawQ, weight: qWeight, eps: eps,
                    nRows: nHeads, rowSize: headDim, on: cb)
                var kTmp = Ops.rmsNormRows(
                    rawK, weight: kWeight, eps: eps,
                    nRows: nKVHeads, rowSize: headDim, on: cb)
                let halfRotary = rotaryDim / 2
                let grid = MTLSize(width: nHeads, height: halfRotary, depth: 1)
                let kGrid = MTLSize(width: nKVHeads, height: halfRotary, depth: 1)
                let tg = MTLSize(width: 1, height: 1, depth: 1)
                IronKernels.iron_rope_yarn_bf16(
                    qk: qTmp.buffer, qkOffset: qTmp.offset,
                    out: qTmp.buffer, outOffset: qTmp.offset,
                    head_dim: UInt32(headDim), half_dim: UInt32(halfRotary),
                    position: UInt32(position), theta_base: theta,
                    factor: yarn.factor, low: yarn.low, high: yarn.high,
                    attn_factor: yarn.attnFactor,
                    gridSize: grid, threadgroupSize: tg, on: cb)
                IronKernels.iron_rope_yarn_bf16(
                    qk: kTmp.buffer, qkOffset: kTmp.offset,
                    out: kTmp.buffer, outOffset: kTmp.offset,
                    head_dim: UInt32(headDim), half_dim: UInt32(halfRotary),
                    position: UInt32(position), theta_base: theta,
                    factor: yarn.factor, low: yarn.low, high: yarn.high,
                    attn_factor: yarn.attnFactor,
                    gridSize: kGrid, threadgroupSize: tg, on: cb)

                let q2d = qTmp.reshaped(to: [nHeads, headDim])
                let k2d = kTmp.reshaped(to: [nKVHeads, headDim])
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
                fusedOut = LagunaFusedFullAttnStep.apply(
                    rawQ: rawQ, rawK: rawK, rawV: rawV,
                    qWeight: qWeight, kWeight: kWeight,
                    nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, rotaryDim: rotaryDim,
                    position: position, thetaBase: theta, eps: eps,
                    yarn: yarn,
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

    @Test("h48/kv8: no history at all (position 0)")
    func h48Pos0() { compare(nHeads: 48, nKVHeads: 8, historyLen: 0, maxSize: 64) }

    @Test("h48/kv8: single history row (position 1)")
    func h48Pos1() { compare(nHeads: 48, nKVHeads: 8, historyLen: 1, maxSize: 64) }

    @Test("h48/kv8: small history, dense reads")
    func h48Small() { compare(nHeads: 48, nKVHeads: 8, historyLen: 5, maxSize: 64) }

    @Test("h48/kv8: mid history")
    func h48Mid() { compare(nHeads: 48, nKVHeads: 8, historyLen: 130, maxSize: 256) }

    @Test("h48/kv8: large history, well past any sliding-window size")
    func h48Large() { compare(nHeads: 48, nKVHeads: 8, historyLen: 1000, maxSize: 1200) }

    // Sanity check: is the fused step itself deterministic run-to-run on
    // identical inputs? Mirrors LagunaFusedSlidingAttnStepTests'
    // stockRepeatableH64 -- rules out the comparison chasing a moving
    // target rather than something this fusion introduced.
    @Test("fused step is repeatable on identical inputs (h48)")
    func fusedRepeatableH48() {
        autoreleasepool {
            let headDim = 128
            let rotaryDim = 64
            let nHeads = 48, nKVHeads = 8
            let historyLen = 32, maxSize = 64
            let eps: Float = 1e-6
            let theta: Float = 500_000
            let scale: Float = 1.0 / Float(headDim).squareRoot()
            let yarn = yarnParams()
            var seed: UInt64 = 0x1234_5678_9ABC_DEF0

            let qWeight = randomBf16([headDim], seed: &seed)
            let kWeight = randomBf16([headDim], seed: &seed)
            let rawQ = randomBf16([nHeads * headDim], seed: &seed)
            let rawK = randomBf16([nKVHeads * headDim], seed: &seed)
            let rawV = randomBf16([nKVHeads * headDim], seed: &seed)

            setenv("BUTTER_LAGUNA_FUSED_FULL_ATTN_STEP", "1", 1)
            defer { unsetenv("BUTTER_LAGUNA_FUSED_FULL_ATTN_STEP") }

            func runOnce() -> Tensor {
                let cache = KVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSize, dtype: .bf16,
                    eviction: .unbounded)
                for _ in 0 ..< historyLen {
                    var hseed: UInt64 = 0xFEED_FACE_BEEF_0000
                    let kRow = randomBf16([nKVHeads, headDim], seed: &hseed)
                    let vRow = randomBf16([nKVHeads, headDim], seed: &hseed)
                    runAndWait { cb in cache.appendOnGPU(kFlat: kRow, vFlat: vRow, on: cb) }
                }
                var out: Tensor!
                runAndWait { cb in
                    out = LagunaFusedFullAttnStep.apply(
                        rawQ: rawQ, rawK: rawK, rawV: rawV,
                        qWeight: qWeight, kWeight: kWeight,
                        nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                        rotaryDim: rotaryDim,
                        position: historyLen, thetaBase: theta, eps: eps,
                        yarn: yarn,
                        cache: cache, scale: scale,
                        on: cb, device: .shared)
                }
                return out
            }

            let run1 = runOnce()
            let run2 = runOnce()
            let a = run1.toFloatArray()
            let b = run2.toFloatArray()
            var mismatches = 0
            for i in 0 ..< min(a.count, b.count) where a[i] != b[i] {
                mismatches += 1
                if mismatches <= 3 {
                    Issue.record("run1[\(i)]=\(a[i]) != run2[\(i)]=\(b[i])")
                }
            }
            #expect(
                mismatches == 0,
                "fused full-attn step non-deterministic across identical repeated calls: \(mismatches) mismatches"
            )
        }
    }
}
