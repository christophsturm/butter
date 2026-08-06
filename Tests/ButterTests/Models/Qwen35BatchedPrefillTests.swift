// Copyright 2026 Eric Kryski (@ekryski)
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
// Regression test for the dense-checkpoint batched-prefill dropped-
// command-buffer bug (w42): `Qwen35GDNLayer.decodeMany` declares
// `commitsCommandBuffer == true`, and the model-level batched driver
// (`_forwardManyBatched`) replaces its `workCmd` after every GDN layer
// on that promise. The dense-FFN branch of `qwen35ApplyFFNMany` leaves
// all work queued on the caller's cmd, so before the fix the entire
// buffer — embedding gather included — was dropped un-executed on
// dense checkpoints (Qwen3.5 0.8B/4B, Qwen3.6-27B). The hidden state
// stayed zero and batched-prefill logits collapsed to argmax token 0
// ("!" loops) while single-token decode stayed correct.
//
// The test builds a tiny synthetic dense GDN hybrid (2 GDN layers,
// dense SwiGLU FFN, unquantized f32 weights — the exact FFN kind that
// triggered the bug) and asserts the batched `forwardMany` prefill
// logits match a per-token `forward(tokenId:)` reference sweep.

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("Qwen35 dense batched prefill (dropped-cmd regression)", .serialized)
struct Qwen35BatchedPrefillTests {

    // Small deterministic LCG so the test is reproducible.
    struct LCG: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
    }

    static func bf16Bits(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(rounded >> 16)
    }

    /// bf16 tensors throughout — the fused single-token GDN path
    /// (`siluCastF32PlusCastF32Two`) only supports f16/bf16 activations,
    /// and bf16 is the production Qwen3.5/3.6 dtype.
    static func randTensor(
        _ shape: [Int], scale: Float, rng: inout LCG
    ) -> Tensor {
        let n = shape.reduce(1, *)
        let vals = (0 ..< n).map { _ in
            bf16Bits(Float.random(in: -scale ... scale, using: &rng))
        }
        let t = Tensor.empty(shape: shape, dtype: .bf16)
        t.copyIn(from: vals)
        return t
    }

    static func onesTensor(_ shape: [Int]) -> Tensor {
        let n = shape.reduce(1, *)
        let t = Tensor.empty(shape: shape, dtype: .bf16)
        t.copyIn(from: [UInt16](repeating: bf16Bits(1), count: n))
        return t
    }

    /// Build a 2-layer all-GDN dense hybrid at a minimal legal shape:
    /// hidden 512, Dk = Dv = 128 (Dv must be a multiple of 128 for the
    /// gated mixer norm), Hk = 1, Hv = 2 → convDim = 2·128 + 256 = 512.
    static func makeTinyDenseModel() -> Qwen35Model {
        let vocab = 64
        let hidden = 512
        let dk = 128, dv = 128, hk = 1, hv = 2
        let keyDim = hk * dk  // 128
        let valueDim = hv * dv  // 256
        let convDim = 2 * keyDim + valueDim  // 512
        let convKernel = 4
        let eps: Float = 1e-6
        let invKeyScale = Foundation.pow(Float(dk), -0.5)
        var rng = LCG(state: 0xC0FFEE)

        var layers: [any DecoderLayer] = []
        for _ in 0 ..< 2 {
            let mixer = Qwen35GDNMixer(
                inProjQKV: AnyLinear(
                    Linear(weight: randTensor([convDim, hidden], scale: 0.05, rng: &rng))),
                inProjZ: AnyLinear(
                    Linear(weight: randTensor([valueDim, hidden], scale: 0.05, rng: &rng))),
                inProjB: AnyLinear(
                    Linear(weight: randTensor([hv, hidden], scale: 0.05, rng: &rng))),
                inProjA: AnyLinear(
                    Linear(weight: randTensor([hv, hidden], scale: 0.05, rng: &rng))),
                outProj: AnyLinear(
                    Linear(weight: randTensor([hidden, valueDim], scale: 0.05, rng: &rng))),
                convW: randTensor([convKernel, convDim], scale: 0.2, rng: &rng),
                convB: {
                    let b = Tensor.empty(shape: [convDim], dtype: .bf16)
                    b.zero()
                    return b
                }(),
                mixerNorm: RMSNorm(weight: onesTensor([dv]), eps: eps),
                aLog: (0 ..< hv).map { _ in Float.random(in: -1 ... 0, using: &rng) },
                dtBias: [Float](repeating: 0, count: hv),
                hidden: hidden,
                numKeyHeads: hk, numValueHeads: hv,
                keyHeadDim: dk, valueHeadDim: dv,
                keyDim: keyDim, valueDim: valueDim, convDim: convDim,
                convKernel: convKernel, eps: eps, invKeyScale: invKeyScale,
                dtype: .bf16)
            let ffn = Qwen35FFN.dense(
                Qwen35DenseMLP(
                    gateProj: AnyLinear(
                        Linear(weight: randTensor([768, hidden], scale: 0.05, rng: &rng))),
                    upProj: AnyLinear(
                        Linear(weight: randTensor([768, hidden], scale: 0.05, rng: &rng))),
                    downProj: AnyLinear(
                        Linear(weight: randTensor([hidden, 768], scale: 0.05, rng: &rng)))))
            layers.append(
                Qwen35GDNLayer(
                    inputNorm: RMSNorm(weight: onesTensor([hidden]), eps: eps),
                    postNorm: RMSNorm(weight: onesTensor([hidden]), eps: eps),
                    mixer: mixer, ffn: ffn, hidden: hidden))
        }

        let embedW = randTensor([vocab, hidden], scale: 0.5, rng: &rng)
        return Qwen35Model(
            embedTokens: AnyEmbedding(Embedding(weight: embedW)),
            layers: layers,
            finalNorm: RMSNorm(weight: onesTensor([hidden]), eps: eps),
            lmHead: AnyLinear(Linear(weight: randTensor([vocab, hidden], scale: 0.1, rng: &rng))),
            hidden: hidden, nLayers: 2, nHeads: 4, nKVHeads: 1, headDim: 128,
            numKeyHeads: hk, numValueHeads: hv,
            keyHeadDim: dk, valueHeadDim: dv,
            convDim: convDim, convKernel: convKernel,
            vocab: vocab, maxSeq: 128, dtype: .bf16)
    }

    @Test("batched forwardMany matches the per-token reference on a dense hybrid")
    func batchedMatchesPerToken() {
        let model = Self.makeTinyDenseModel()
        let tokens = [3, 17, 42, 5, 60]
        let device = Device.shared

        // Reference: per-token single-forward sweep (the known-good
        // decode path — commits every GDN layer via qwen35ApplyFFN).
        let refCaches = model.makeLayerCaches(maxSeq: nil, device: device)
        var refLogits: Tensor!
        for (i, t) in tokens.enumerated() {
            let cmd = device.makeCommandBuffer()
            refLogits = model.forward(
                tokenId: t, position: i, caches: refCaches,
                on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let ref = refLogits.toFloatArray()

        // Batched prefill: one forwardMany over all 5 tokens — the path
        // that dropped its workCmd un-committed before the fix.
        let batCaches = model.makeLayerCaches(maxSeq: nil, device: device)
        let cmd = device.makeCommandBuffer()
        let batLogits = model.forwardMany(
            tokenIds: tokens, startPosition: 0, caches: batCaches,
            on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        let bat = batLogits.toFloatArray()

        #expect(ref.count == bat.count)

        // Bug signature: the batched path returned lm_head(norm(0)) —
        // an (almost) all-zero logits row. Guard against it explicitly.
        let batL2 = bat.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        #expect(batL2 > 1e-3, "batched logits are ~zero (dropped-cmd signature)")

        // And the batched logits must track the per-token reference.
        // bf16 activations + a different (chunked) recurrence kernel →
        // small rounding-scale differences are expected; the dropped-cmd
        // bug produced a wholesale zero row, orders of magnitude away.
        var maxAbsDiff: Float = 0
        for i in 0 ..< ref.count {
            maxAbsDiff = max(maxAbsDiff, abs(ref[i] - bat[i]))
        }
        #expect(
            maxAbsDiff < 0.05,
            "batched vs per-token logits diverge: max_abs_diff=\(maxAbsDiff)")

        // Greedy argmax must agree.
        let refArgmax = ref.indices.max(by: { ref[$0] < ref[$1] })!
        let batArgmax = bat.indices.max(by: { bat[$0] < bat[$1] })!
        #expect(refArgmax == batArgmax)
    }

    /// Per-ROW batched-vs-sequential parity.
    ///
    /// `batchedMatchesPerToken` above only compares the LAST logits row,
    /// which hides a divergence concentrated in the earlier rows. The MTP
    /// speculative loop consumes every row (row j decides whether draft j
    /// is accepted, and the trailing row is emitted directly on a full
    /// accept), so each row has to track the per-token reference — not
    /// just the last one.
    ///
    /// Measured on the real Qwen3.6-27B checkpoint, row 0 disagreed with
    /// the single-token path on 45% of full-accept cycles at gamma=3 and
    /// 21% at gamma=1, at logit margins up to 7.0 (a confidently
    /// different token, not an argmax tie-break). This test is the
    /// checkpoint-free reproduction of that.
    @Test("batched forwardMany matches the per-token reference on EVERY row")
    func batchedMatchesPerTokenAllRows() {
        let model = Self.makeTinyDenseModel()
        let tokens = [3, 17, 42, 5, 60]
        let device = Device.shared

        // Reference: one logits row per position from the single-token
        // decode path, which is the exactness baseline.
        let refCaches = model.makeLayerCaches(maxSeq: nil, device: device)
        var refRows: [[Float]] = []
        for (i, t) in tokens.enumerated() {
            let cmd = device.makeCommandBuffer()
            let lg = model.forward(
                tokenId: t, position: i, caches: refCaches,
                on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            refRows.append(lg.toFloatArray())
        }

        // Batched: all rows from a single forwardMany over the block.
        let batCaches = model.makeLayerCaches(maxSeq: nil, device: device)
        let cmd = device.makeCommandBuffer()
        let batAll = model.forwardManyAllLogits(
            tokenIds: tokens, startPosition: 0, caches: batCaches,
            on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        let flat = batAll.toFloatArray()
        let vocab = refRows[0].count
        #expect(flat.count == tokens.count * vocab)

        var worstRow = -1
        var worstDiff: Float = 0
        var argmaxMismatches: [Int] = []
        for r in 0 ..< tokens.count {
            let ref = refRows[r]
            var maxAbsDiff: Float = 0
            for v in 0 ..< vocab {
                maxAbsDiff = max(maxAbsDiff, abs(ref[v] - flat[r * vocab + v]))
            }
            if maxAbsDiff > worstDiff {
                worstDiff = maxAbsDiff
                worstRow = r
            }
            let refArg = ref.indices.max(by: { ref[$0] < ref[$1] })!
            var batArg = 0
            var best = -Float.infinity
            for v in 0 ..< vocab where flat[r * vocab + v] > best {
                best = flat[r * vocab + v]
                batArg = v
            }
            if refArg != batArg { argmaxMismatches.append(r) }
        }

        #expect(
            argmaxMismatches.isEmpty,
            "batched argmax differs from per-token on rows \(argmaxMismatches); worst |diff| \(worstDiff) on row \(worstRow)")
        #expect(
            worstDiff < 0.05,
            "per-row batched vs per-token divergence: max_abs_diff=\(worstDiff) on row \(worstRow)")
    }

}
