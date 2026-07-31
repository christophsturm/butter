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
// Terminal-layer last-row prefill (`LagunaLayer.forwardMultiLastRow`,
// see `LagunaLastRowPrefill` in LagunaText.swift). Two kinds of coverage:
//
//   1. CPU-checkable plumbing: the row-slice index math and the
//      "residual add commutes with slicing to the last row" property
//      the whole design leans on (Phase B slices `xNorm`/`residual`'s
//      last row instead of recomputing them).
//
//   2. GPU equivalence: a full `forwardMulti` pass (all rows) sliced to
//      row T-1 vs `forwardMultiLastRow`'s direct output, on a real
//      (small) `LagunaLayer` with two fresh, identically-seeded KV
//      caches. Both arms are the REAL production code paths; this is
//      NOT expected to be bit-exact:
//        - Stock's bulk attention (`prefillMmaBf16`/`sdpaMulti`, a
//          multi-row MMA-GEMM causal kernel) vs the lastrow path's
//          single-row decode SDPA (`LagunaSdpaDecodeGqa2`/`sdpaDecode`)
//          are different kernels computing the same causal-attention
//          math for the last row: the same class of FP-reorder already
//          accepted throughout this file family (see
//          `LagunaSdpaDecodeGqa2Tests`, `LagunaPrefillFusedQKRopeProbeTests`).
//        - `BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE=0` is forced for the
//          comparison so stock's K-cache write takes the SAME unfused
//          `rmsNormRows`+`ropePartialMany`/YaRN chain
//          `forwardMultiLastRow`'s Phase A always uses (it never takes
//          the T30 fused-QK-RoPE-for-prefill kernels, see the "QK-norm
//          +RoPE" bullet in `forwardMultiLastRow`'s doc comment),
//          without this, the two arms' cached K bytes could differ by
//          up to the ~1 ULP T30 already measured, on top of the SDPA
//          delta below.
//        - `BUTTER_LAGUNA_ATTN_INT8=0` is forced so o_proj takes the
//          same plain-bf16 GEMV family in both arms; DEFAULT config
//          (INT8 enabled) makes `forwardMultiLastRow`'s o_proj eligible
//          for the group-32 INT8 QMV bank on qualifying shapes, same
//          as decode already does, but a NEW call site during prefill
//          (previously prefill's o_proj was always exact bf16). Flagged
//          in the commit report as a real behavioral note, not just a
//          test-isolation knob.
import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

private func bf16Bits(_ v: Float) -> UInt16 {
    let bits = v.bitPattern
    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
    return UInt16(rounded >> 16)
}

private func randomBf16(_ shape: [Int], seed: inout UInt64, scale: Float = 0.05) -> Tensor {
    let n = shape.reduce(1, *)
    var vals = [UInt16](repeating: 0, count: n)
    for i in 0 ..< n {
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        let u = Double(seed % 2_000_001) / 1_000_000.0 - 1.0  // [-1, 1)
        vals[i] = bf16Bits(Float(u) * scale)
    }
    let t = Tensor.empty(shape: shape, dtype: .bf16)
    t.copyIn(from: vals)
    return t
}

/// Small-but-realistic Laguna-shaped layer: headDim fixed at 128 (the
/// only width `LagunaSdpaDecodeGqa2`/`Ops.sdpaDecode` support), hidden
/// and the Q/K/V/O out-dims all multiples of 64 (steel-GEMM eligible)
/// and of 32 (INT8 group eligible) so the test exercises the same
/// kernel-eligibility envelope Laguna XS/S hit in production, just at a
/// fraction of the width/expert-count for speed.
private func makeTestLayer(
    hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int, rotaryDim: Int,
    ropeTheta: Float, isSliding: Bool, isFullYarn: Bool,
    useMoE: Bool, nExperts: Int, topK: Int, moeIntermediate: Int,
    seed: inout UInt64
) -> LagunaLayer {
    let device = Device.shared
    func lin(_ outDim: Int, _ inDim: Int) -> AnyLinear {
        AnyLinear(Linear(weight: randomBf16([outDim, inDim], seed: &seed)))
    }
    func norm(_ n: Int) -> RMSNorm {
        RMSNorm(weight: randomBf16([n], seed: &seed, scale: 1.0), eps: 1e-6)
    }

    let qProj = lin(nHeads * headDim, hidden)
    let kProj = lin(nKVHeads * headDim, hidden)
    let vProj = lin(nKVHeads * headDim, hidden)
    let oProj = lin(hidden, nHeads * headDim)
    let gProj = lin(nHeads, hidden)
    let qNorm = norm(headDim)
    let kNorm = norm(headDim)
    let inputNorm = norm(hidden)
    let postAttnNorm = norm(hidden)

    let ffn: LagunaFFN
    if useMoE {
        let gate = lin(nExperts, hidden)
        var gateProj: [AnyLinear] = []
        var upProj: [AnyLinear] = []
        var downProj: [AnyLinear] = []
        for _ in 0 ..< nExperts {
            gateProj.append(lin(moeIntermediate, hidden))
            upProj.append(lin(moeIntermediate, hidden))
            downProj.append(lin(hidden, moeIntermediate))
        }
        let expertBias = (0 ..< nExperts).map { _ -> Float in
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            return Float(Double(seed % 2_000_001) / 1_000_000.0 - 1.0) * 0.1
        }
        let router = MoERouter(
            nExperts: nExperts, topK: topK, gatingMode: .sigmoidBiasedTopK,
            normTopKProb: true, expertBias: expertBias, routedScalingFactor: 1.0)
        let moe = MoELayer(
            gate: gate, gateProj: gateProj, upProj: upProj, downProj: downProj,
            router: router, hidden: hidden, layerIndex: 0, dtype: .bf16, device: device)
        ffn = .moe(moe)
    } else {
        let inter = hidden * 2
        let gate = lin(inter, hidden)
        let up = lin(inter, hidden)
        let down = lin(hidden, inter)
        ffn = .dense(
            LagunaDenseMLP(gate: gate, up: up, down: down, hidden: hidden, dtype: .bf16, device: device))
    }

    return LagunaLayer(
        qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
        gProj: gProj, qNorm: qNorm, kNorm: kNorm,
        inputNorm: inputNorm, postAttnNorm: postAttnNorm,
        ffn: ffn,
        hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, rotaryDim: rotaryDim,
        ropeTheta: ropeTheta, isSliding: isSliding, isFullYarn: isFullYarn, dtype: .bf16,
        device: device, layerIndex: 0)
}

/// Save/restore an env var around a test body so cross-test/cross-suite
/// state doesn't leak (Swift Testing may run suites concurrently).
private func withEnv(_ overrides: [String: String], _ body: () -> Void) {
    var saved: [String: String?] = [:]
    for (k, v) in overrides {
        saved[k] = ProcessInfo.processInfo.environment[k]
        setenv(k, v, 1)
    }
    defer {
        for (k, old) in saved {
            if let old { setenv(k, old, 1) } else { unsetenv(k) }
        }
    }
    body()
}

// ─── CPU-checkable plumbing ────────────────────────────────────────────

@Suite("LagunaLastRowPrefill plumbing")
struct LagunaLastRowPrefillPlumbingTests {
    @Test("slicedRows(start: T-1) picks the physically-last row")
    func lastRowSliceMatchesManualIndex() {
        let t = 9
        let hidden = 16
        var host = [Float](repeating: 0, count: t * hidden)
        for r in 0 ..< t {
            for c in 0 ..< hidden { host[r * hidden + c] = Float(r * 100 + c) }
        }
        let full = Tensor.empty(shape: [t, hidden], dtype: .f32)
        full.copyIn(from: host)

        let last = full.slicedRows(start: t - 1, count: 1).reshaped(to: [hidden])
        let got = last.toFloatArray()
        let expected = Array(host[(t - 1) * hidden ..< t * hidden])
        #expect(got == expected)

        // Same contract `forwardMultiLastRow` relies on for BOTH
        // `xNorm.slicedRows(start: nRows-1, count: 1)` and
        // `residualFull.slicedRows(start: nRows-1, count: 1)`: an
        // arbitrary interior row is untouched by the slice.
        let mid = full.slicedRows(start: 3, count: 1).reshaped(to: [hidden])
        #expect(mid.toFloatArray() == Array(host[3 * hidden ..< 4 * hidden]))
    }

    @Test("residual-row add commutes with slicing to the last row")
    func residualRowMathCommutesWithSlicing() {
        // `forwardMultiLastRow`'s Phase B computes
        // `h2 = residualLast + oOut` (single row) instead of computing
        // the full-width `h2 = residual + oOut` and slicing afterward.
        // This only produces the same answer as the stock full-width
        // path because elementwise add is row-independent; verify that
        // property directly (`Ops.add` is a straight elementwise kernel,
        // so this also stands in as a plumbing regression guard on the
        // exact row-stride arithmetic Phase B uses).
        let t = 5
        let hidden = 32
        var seed: UInt64 = 0xABCD_1234_5678_9101
        func randomF32(_ n: Int) -> [Float] {
            var out = [Float](repeating: 0, count: n)
            for i in 0 ..< n {
                seed ^= seed << 13
                seed ^= seed >> 7
                seed ^= seed << 17
                out[i] = Float(Double(seed % 2_000_001) / 1_000_000.0 - 1.0)
            }
            return out
        }
        let residualHost = randomF32(t * hidden)
        let attnHost = randomF32(t * hidden)
        let residual = Tensor.empty(shape: [t * hidden], dtype: .f32)
        let attn = Tensor.empty(shape: [t * hidden], dtype: .f32)
        residual.copyIn(from: residualHost)
        attn.copyIn(from: attnHost)

        // Full-width add, then slice the last row (what stock forwardMulti
        // + the model-level `slicedRows` post-loop step effectively do).
        var h2Full: Tensor!
        runAndWait { cb in h2Full = Ops.add(residual, attn, on: cb) }
        let fromFullSlice = h2Full.reshaped(to: [t, hidden])
            .slicedRows(start: t - 1, count: 1).reshaped(to: [hidden]).toFloatArray()

        // Slice first, then add just the one row (what `forwardMultiLastRow`
        // does: `residualLast`/`oOut` are already single rows).
        let residualLast = residual.reshaped(to: [t, hidden])
            .slicedRows(start: t - 1, count: 1).reshaped(to: [hidden])
        let attnLast = attn.reshaped(to: [t, hidden])
            .slicedRows(start: t - 1, count: 1).reshaped(to: [hidden])
        var h2Last: Tensor!
        runAndWait { cb in h2Last = Ops.add(residualLast, attnLast, on: cb) }
        let fromRowAdd = h2Last.toFloatArray()

        #expect(fromFullSlice.count == hidden)
        #expect(fromRowAdd.count == hidden)
        for i in 0 ..< hidden {
            #expect(fromFullSlice[i] == fromRowAdd[i])
        }
    }
}

// ─── GPU equivalence ────────────────────────────────────────────────────

@Suite("LagunaLastRowPrefill GPU equivalence", .serialized)
struct LagunaLastRowPrefillGPUTests {
    private func compare(
        isSliding: Bool, isFullYarn: Bool, useMoE: Bool, label: String
    ) {
        autoreleasepool {
            withEnv([
                // Isolate the comparison to the SDPA-algorithm delta
                // (bulk MMA-GEMM vs single-row decode SDPA); see file
                // header. Both env vars restored by `withEnv` on exit.
                "BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE": "0",
                "BUTTER_LAGUNA_ATTN_INT8": "0",
            ]) {
                let device = Device.shared
                let hidden = 128
                let nHeads = 4, nKVHeads = 2, headDim = 128
                let rotaryDim = isFullYarn ? 64 : headDim
                let ropeTheta: Float = isFullYarn ? 500_000 : 10_000
                let nRows = 64
                let maxSeq = 128
                var seed: UInt64 = 0x1357_9BDF_2468_ACE0

                let layer = makeTestLayer(
                    hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    rotaryDim: rotaryDim, ropeTheta: ropeTheta,
                    isSliding: isSliding, isFullYarn: isFullYarn,
                    useMoE: useMoE, nExperts: 4, topK: 2, moeIntermediate: 64,
                    seed: &seed)

                let h = randomBf16([nRows, hidden], seed: &seed, scale: 0.1)
                let yarn = Ops.RoPEYaRN(factor: 32, low: 1, high: 32, attnFactor: 1.0)

                let cacheStock = KVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq, dtype: .bf16,
                    eviction: .unbounded, device: device)
                let cacheLastRow = KVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq, dtype: .bf16,
                    eviction: .unbounded, device: device)

                let stockCmd = device.makeCommandBuffer()
                let stockOut = layer.forwardMulti(
                    h, startingAt: 0, cache: cacheStock, yarn: yarn,
                    cmd: stockCmd, device: device)
                if stockCmd.status != .committed && stockCmd.status != .completed
                    && stockCmd.status != .error
                {
                    stockCmd.commit()
                }
                stockCmd.waitUntilCompleted()
                let stockLast = stockOut.slicedRows(start: nRows - 1, count: 1)
                    .reshaped(to: [hidden])

                let lastRowCmd = device.makeCommandBuffer()
                let lastRowOut = layer.forwardMultiLastRow(
                    h, startingAt: 0, cache: cacheLastRow, yarn: yarn,
                    cmd: lastRowCmd, device: device)
                if lastRowCmd.status != .committed && lastRowCmd.status != .completed
                    && lastRowCmd.status != .error
                {
                    lastRowCmd.commit()
                }
                lastRowCmd.waitUntilCompleted()

                let a = stockLast.toFloatArray()
                let b = lastRowOut.toFloatArray()
                #expect(a.count == hidden)
                #expect(b.count == hidden)
                guard a.count == b.count else { return }

                var maxAbsDiff: Float = 0
                var dot: Double = 0, na: Double = 0, nb: Double = 0
                for i in 0 ..< a.count {
                    maxAbsDiff = max(maxAbsDiff, abs(a[i] - b[i]))
                    dot += Double(a[i]) * Double(b[i])
                    na += Double(a[i]) * Double(a[i])
                    nb += Double(b[i]) * Double(b[i])
                }
                let cos = (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 1.0
                // Loose-but-meaningful bound: same class of tolerance
                // `LagunaSdpaDecodeGqa2Tests` / `LagunaPrefillFusedQKRopeProbeTests`
                // already use for "different kernel, same math" FP-reorder,
                // not a quantization or algorithmic-error bound (INT8 is
                // forced off above).
                #expect(cos > 0.999, "\(label): cos=\(cos) maxAbsDiff=\(maxAbsDiff)")
                #expect(maxAbsDiff < 0.1, "\(label): maxAbsDiff=\(maxAbsDiff) cos=\(cos)")
            }
        }
    }

    @Test("sliding layer + dense FFN: full path row T-1 == lastrow path")
    func slidingDense() {
        compare(isSliding: true, isFullYarn: false, useMoE: false, label: "sliding+dense")
    }

    @Test("full/YaRN layer + dense FFN: full path row T-1 == lastrow path")
    func fullYarnDense() {
        compare(isSliding: false, isFullYarn: true, useMoE: false, label: "full-yarn+dense")
    }

    @Test("sliding layer + MoE FFN: full path row T-1 == lastrow path")
    func slidingMoE() {
        compare(isSliding: true, isFullYarn: false, useMoE: true, label: "sliding+moe")
    }
}
