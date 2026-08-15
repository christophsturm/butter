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
import Foundation
import Testing

@testable import ButterSwift

@Suite("Sampling")
struct SamplingTests {
    @Test("argmax over f32 picks the largest")
    func argmaxF32() {
        let t = Tensor.empty(shape: [5], dtype: .f32)
        t.copyIn(from: [Float(0.1), 0.5, -2.0, 4.2, 1.0])
        #expect(Sampling.argmax(t) == 3)
    }

    @Test("argmax over f16 picks the largest")
    func argmaxF16() {
        let t = Tensor.empty(shape: [4], dtype: .f16)
        t.copyIn(from: [Float16(1), 7, 3, -1])
        #expect(Sampling.argmax(t) == 1)
    }

    @Test("argmax over bf16 picks the largest")
    func argmaxBf16() {
        let t = Tensor.empty(shape: [4], dtype: .bf16)
        // bf16 = top 16 bits of f32; build by shifting our floats down.
        let bits: [UInt16] = [0x3F80, 0x40C0, 0x3FC0, 0xBF80]  // 1.0, 6.0, 1.5, -1.0
        t.copyIn(from: bits)
        #expect(Sampling.argmax(t) == 1)
    }

    @Test("topN returns N entries sorted descending")
    func topN() {
        let t = Tensor.empty(shape: [6], dtype: .f32)
        t.copyIn(from: [Float(1), 5, 2, 9, 0, 7])
        let top3 = Sampling.topN(t, n: 3)
        #expect(top3.count == 3)
        #expect(top3[0].0 == 3)  // value 9
        #expect(top3[1].0 == 5)  // value 7
        #expect(top3[2].0 == 1)  // value 5
        #expect(top3[0].1 == 9)
    }

    @Test("topN works for f16 + bf16 too")
    func topNDtypes() {
        let f16 = Tensor.empty(shape: [3], dtype: .f16)
        f16.copyIn(from: [Float16(1), 5, 2])
        #expect(Sampling.topN(f16, n: 1)[0].0 == 1)

        let bf16 = Tensor.empty(shape: [3], dtype: .bf16)
        bf16.copyIn(from: [UInt16(0x3F80), 0x40A0, 0x4000])  // 1, 5, 2
        #expect(Sampling.topN(bf16, n: 1)[0].0 == 1)
    }

    // MARK: - Sampling pipeline

    private func makeLogits(_ vs: [Float]) -> Tensor {
        let t = Tensor.empty(shape: [vs.count], dtype: .f32)
        t.copyIn(from: vs)
        return t
    }

    @Test("sample at temperature=0 matches argmax (no RNG draw)")
    func tempZeroIsArgmax() {
        let t = makeLogits([0.1, 0.5, -2.0, 4.2, 1.0])
        var rng = SeededRandomNumberGenerator(seed: 42)
        let p = GenerationParameters(temperature: 0)
        #expect(Sampling.sample(t, parameters: p, rng: &rng) == 3)
    }

    @Test("sample is reproducible with the same seed")
    func reproducibleSeed() {
        let t = makeLogits([0.1, 1.0, 0.5, 2.0, 0.3, 1.5])
        let p = GenerationParameters(temperature: 0.7)
        var rng1 = SeededRandomNumberGenerator(seed: 12345)
        var rng2 = SeededRandomNumberGenerator(seed: 12345)
        let seq1 = (0 ..< 8).map { _ in Sampling.sample(t, parameters: p, rng: &rng1) }
        let seq2 = (0 ..< 8).map { _ in Sampling.sample(t, parameters: p, rng: &rng2) }
        #expect(seq1 == seq2)
    }

    @Test("top-K cuts the candidate set to K largest logits")
    func topKFilter() {
        // logits — argmax is index 4 (3.0). Top-2 are indices 4, 1.
        let t = makeLogits([0.0, 2.0, -1.0, 0.5, 3.0])
        var rng = SeededRandomNumberGenerator(seed: 1)
        let p = GenerationParameters(temperature: 1.0, topK: 2)
        var hits = Set<Int>()
        for _ in 0 ..< 200 {
            hits.insert(Sampling.sample(t, parameters: p, rng: &rng))
        }
        // Across 200 draws with topK=2 we should only ever see {1, 4}.
        #expect(hits.isSubset(of: [1, 4]))
        #expect(hits.contains(4))  // the dominant prob should fire
    }

    @Test("top-P keeps only the cumulative-prob nucleus")
    func topPFilter() {
        // Very peaked distribution: idx 0 has nearly all the mass.
        let t = makeLogits([10.0, 0.0, 0.0, 0.0])
        var rng = SeededRandomNumberGenerator(seed: 2)
        let p = GenerationParameters(temperature: 1.0, topP: 0.5)
        // top-P=0.5 with the peak >>> rest should always pick the peak.
        for _ in 0 ..< 50 {
            #expect(Sampling.sample(t, parameters: p, rng: &rng) == 0)
        }
    }

    @Test("min-P drops tokens below min_p × max_prob")
    func minPFilter() {
        // Peak prob ~ 0.999, others near 0. min-P=0.5 should only keep the peak.
        let t = makeLogits([8.0, 0.0, 0.0, 0.0])
        var rng = SeededRandomNumberGenerator(seed: 3)
        let p = GenerationParameters(temperature: 1.0, minP: 0.5)
        for _ in 0 ..< 50 {
            #expect(Sampling.sample(t, parameters: p, rng: &rng) == 0)
        }
    }

    @Test("repetition penalty discounts already-seen tokens with positive logits")
    func repPenaltyDiscountsSeen() {
        // Without penalty: argmax would be index 0 (logit 5.0).
        // With penalty 2.0 applied to {0}: logit 0 → 5/2 = 2.5; new max is index 1 (3.0).
        let t = makeLogits([5.0, 3.0, 1.0, 0.5])
        var rng = SeededRandomNumberGenerator(seed: 4)
        let p = GenerationParameters(temperature: 0, repetitionPenalty: 2.0)
        let plain = Sampling.sample(
            t, parameters: GenerationParameters(temperature: 0),
            rng: &rng, tokenHistory: [0])
        #expect(plain == 0)
        let penalized = Sampling.sample(
            t, parameters: p,
            rng: &rng, tokenHistory: [0])
        #expect(penalized == 1)
    }

    @Test("repetition penalty multiplies tokens with negative logits")
    func repPenaltyAmplifiesNegative() {
        // Argmax would be index 0 (logit 1.0). Index 2 has -0.5.
        // With penalty 2.0 on {2}: -0.5 * 2 = -1.0 (further suppressed).
        let t = makeLogits([1.0, 0.5, -0.5, -1.0])
        var rng = SeededRandomNumberGenerator(seed: 5)
        let p = GenerationParameters(temperature: 0, repetitionPenalty: 2.0)
        let r = Sampling.sample(t, parameters: p, rng: &rng, tokenHistory: [2])
        // index 0 still wins; index 2 is further suppressed.
        #expect(r == 0)
    }

    @Test("makeRNG honors the seed")
    func makeRNGSeedHonored() {
        let p1 = GenerationParameters(seed: 99)
        let p2 = GenerationParameters(seed: 99)
        var r1 = p1.makeRNG()
        var r2 = p2.makeRNG()
        #expect(r1.next() == r2.next())
        #expect(r1.next() == r2.next())
    }

    @Test("makeRNG with no seed gives a system random generator")
    func makeRNGNoSeed() {
        var rng = GenerationParameters().makeRNG()
        // Just exercise: must not crash and must produce some bits.
        _ = rng.next()
    }

    @Test("SeededRandomNumberGenerator is deterministic across instances")
    func seededDeterminism() {
        var r1 = SeededRandomNumberGenerator(seed: 0xDEAD_BEEF)
        var r2 = SeededRandomNumberGenerator(seed: 0xDEAD_BEEF)
        let s1 = (0 ..< 10).map { _ in r1.next() }
        let s2 = (0 ..< 10).map { _ in r2.next() }
        #expect(s1 == s2)
    }

    @Test("sample bf16 + f16 logits both work")
    func sampleDtypes() {
        var rng = SeededRandomNumberGenerator(seed: 7)
        let p = GenerationParameters(temperature: 0)
        let f16 = Tensor.empty(shape: [4], dtype: .f16)
        f16.copyIn(from: [Float16(1), 7, 3, -1])
        #expect(Sampling.sample(f16, parameters: p, rng: &rng) == 1)

        let bf16 = Tensor.empty(shape: [4], dtype: .bf16)
        bf16.copyIn(from: [UInt16(0x3F80), 0x40C0, 0x3FC0, 0xBF80])
        #expect(Sampling.sample(bf16, parameters: p, rng: &rng) == 1)
    }
}

// MARK: - T43: fast partial top-K vs full-sort reference equivalence

/// `Sampling.buildFilteredWeights`'s fast path (bounded min-heap partial
/// selection when top-K is active) must be byte-identical — same
/// SURVIVING SET, same WEIGHTS, and critically same ORDER (the
/// categorical draw in `sampleFromWeights` accumulates over `indices`/
/// `weights` in order, so a reordering would silently change which
/// token a given RNG draw lands on) — to the pre-T43 always-sort
/// implementation kept as `Sampling.buildFilteredWeightsFullSort` for
/// exactly this purpose. Covers every filter combo the task brief calls
/// out (temp only / +top-K / +top-P / +min-P / all) across several
/// vocab sizes and random rows, plus tie-heavy rows (mirroring
/// `LagunaLmHeadPrune`'s -infinity-masked candidates) since ties are
/// where a naive partial-selection rewrite is most likely to silently
/// diverge from a stable sort's tie-break.
@Suite("Sampling filter equivalence — fast top-K vs full-sort reference")
struct SamplingFilterEquivalenceTests {

    /// Deterministic pseudo-random logits row. `tieFraction` of entries
    /// are pinned to `tieValue` (when set) so ties get real coverage —
    /// generic random floats essentially never collide.
    private static func randomRow(
        vocab: Int, seed: UInt64, range: ClosedRange<Float> = -10 ... 10,
        tieValue: Float? = nil, tieFraction: Double = 0.0
    ) -> [Float] {
        var rng = SeededRandomNumberGenerator(seed: seed)
        var row = [Float](repeating: 0, count: vocab)
        for i in 0 ..< vocab {
            if let tieValue, Double.random(in: 0 ..< 1, using: &rng) < tieFraction {
                row[i] = tieValue
            } else {
                row[i] = Float.random(in: range, using: &rng)
            }
        }
        return row
    }

    private static func assertEquivalent(
        _ values: [Float], _ parameters: GenerationParameters, label: String
    ) {
        let fast = Sampling.buildFilteredWeights(values, parameters: parameters)
        let reference = Sampling.buildFilteredWeightsFullSort(values, parameters: parameters)
        #expect(fast.indices == reference.indices, "\(label): surviving index set/order differs")
        #expect(
            fast.weights.count == reference.weights.count,
            "\(label): weight counts differ (\(fast.weights.count) vs \(reference.weights.count))")
        #expect(abs(fast.total - reference.total) < 1e-6, "\(label): total differs")
        for i in 0 ..< min(fast.weights.count, reference.weights.count) {
            #expect(
                abs(fast.weights[i] - reference.weights[i]) < 1e-6,
                "\(label): weight[\(i)] differs (\(fast.weights[i]) vs \(reference.weights[i]))")
        }
    }

    private static let paramCombos: [(String, GenerationParameters)] = [
        ("temp only", GenerationParameters().with {
            $0.temperature = 0.6; $0.topK = 0; $0.topP = 1.0; $0.minP = 0.0
        }),
        ("+topK", GenerationParameters().with {
            $0.temperature = 0.6; $0.topK = 20; $0.topP = 1.0; $0.minP = 0.0
        }),
        ("+topK tiny", GenerationParameters().with {
            $0.temperature = 1.0; $0.topK = 3; $0.topP = 1.0; $0.minP = 0.0
        }),
        ("+topP (no topK)", GenerationParameters().with {
            $0.temperature = 0.6; $0.topK = 0; $0.topP = 0.9; $0.minP = 0.0
        }),
        ("+topK+topP", GenerationParameters().with {
            $0.temperature = 0.6; $0.topK = 20; $0.topP = 0.95; $0.minP = 0.0
        }),
        ("+topK+minP", GenerationParameters().with {
            $0.temperature = 0.6; $0.topK = 20; $0.topP = 1.0; $0.minP = 0.05
        }),
        ("all (topK+topP+minP)", GenerationParameters().with {
            $0.temperature = 0.6; $0.topK = 20; $0.topP = 0.95; $0.minP = 0.05
        }),
        ("topK >= vocab (degenerate, exercises the non-fast branch)", GenerationParameters().with {
            $0.temperature = 0.6; $0.topK = 999_999; $0.topP = 0.95; $0.minP = 0.0
        }),
    ]

    @Test("fast top-K path matches full-sort reference across param combos and random rows")
    func fastPathMatchesReferenceAcrossCombos() {
        for vocab in [8, 37, 500, 4096] {
            for (label, params) in Self.paramCombos {
                for seed: UInt64 in [1, 2, 3] {
                    let row = Self.randomRow(vocab: vocab, seed: seed &+ UInt64(vocab))
                    Self.assertEquivalent(row, params, label: "\(label) vocab=\(vocab) seed=\(seed)")
                }
                // ~40% of entries pinned to a shared value, simulating
                // masked (-infinity-style) candidates.
                let tieRow = Self.randomRow(
                    vocab: vocab, seed: 99 &+ UInt64(vocab), tieValue: -1000, tieFraction: 0.4)
                Self.assertEquivalent(tieRow, params, label: "\(label) vocab=\(vocab) ties")
            }
        }
    }

    @Test("fast top-K path matches reference when every logit is exactly tied")
    func exactTieBreakMatchesReference() {
        // Every entry the SAME value: the entire surviving order is
        // decided purely by the tie-break rule (ascending index). If the
        // heap's tie-break ever diverged from the stable-sort convention
        // this is where `indices` would come back scrambled.
        let vocab = 1000
        let row = [Float](repeating: 3.0, count: vocab)
        let params = GenerationParameters().with {
            $0.temperature = 1.0
            $0.topK = 17
            $0.topP = 1.0
            $0.minP = 0.0
        }
        Self.assertEquivalent(row, params, label: "all-tied")
    }

    @Test("fast top-K path matches reference at topK == 1 (boundary)")
    func topKOneBoundary() {
        for vocab in [1, 2, 8, 500] {
            let row = Self.randomRow(vocab: vocab, seed: 12345 &+ UInt64(vocab))
            let params = GenerationParameters().with {
                $0.temperature = 0.6
                $0.topK = 1
                $0.topP = 1.0
                $0.minP = 0.0
            }
            Self.assertEquivalent(row, params, label: "topK=1 vocab=\(vocab)")
        }
    }
}
