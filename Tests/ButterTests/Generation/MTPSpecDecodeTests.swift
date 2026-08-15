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
// MTPSpecDecodeTests — pure-CPU coverage of the MTP spec-decode
// iterator's accept-length arithmetic (`SpecDecode.planMTPAccept`,
// spec 030 phase 2). The drafts here play the role of a mocked
// drafter: literal candidate arrays paired with literal verifier
// argmax rows. GPU/model coverage lives in
// Tests/ModelIntegrationTests/Text/MTPSpecDecodeIntegrationTests.swift.

import Testing

@testable import ButterSwift

@Suite("MTPSpecDecode — accept-length arithmetic")
struct MTPSpecDecodeTests {

    @Test("full accept commits prev + all drafts, bonus seeds next iteration, no replay")
    func fullAcceptPlusBonus() {
        let plan = SpecDecode.planMTPAccept(
            prev: 100, drafts: [1, 2, 3],
            verifierTokens: [1, 2, 3], batchedBonus: 42)
        #expect(plan.committed == [100, 1, 2, 3])
        #expect(plan.nextToken == 42)
        #expect(!plan.needsReplay)
        // Next burst seeds from the verify hidden at the last committed
        // position — row 3 of the [γ+1, hidden] output.
        #expect(plan.nextHiddenRow == 3)
    }

    @Test("partial accept commits the matching prefix and forces replay")
    func partialAccept() {
        let plan = SpecDecode.planMTPAccept(
            prev: 100, drafts: [1, 2, 3, 4],
            verifierTokens: [1, 2, 99, 4], batchedBonus: 42)
        #expect(plan.committed == [100, 1, 2])
        // Accept-time candidate is the verifier token at the first
        // mismatch; the loop recomputes it on the exact single-token
        // path before use (needsReplay).
        #expect(plan.nextToken == 99)
        #expect(plan.needsReplay)
        #expect(plan.nextHiddenRow == 2)
    }

    @Test("immediate reject commits only prev and replays it")
    func immediateReject() {
        let plan = SpecDecode.planMTPAccept(
            prev: 100, drafts: [7, 8],
            verifierTokens: [5, 8], batchedBonus: 42)
        #expect(plan.committed == [100])
        #expect(plan.nextToken == 5)
        #expect(plan.needsReplay)
        #expect(plan.nextHiddenRow == 0)
    }

    @Test("gamma == 1, accepted")
    func gammaOneAccept() {
        let plan = SpecDecode.planMTPAccept(
            prev: 100, drafts: [7],
            verifierTokens: [7], batchedBonus: 42)
        #expect(plan.committed == [100, 7])
        #expect(plan.nextToken == 42)
        #expect(!plan.needsReplay)
        #expect(plan.nextHiddenRow == 1)
    }

    @Test("gamma == 1, rejected")
    func gammaOneReject() {
        let plan = SpecDecode.planMTPAccept(
            prev: 100, drafts: [7],
            verifierTokens: [9], batchedBonus: 42)
        #expect(plan.committed == [100])
        #expect(plan.nextToken == 9)
        #expect(plan.needsReplay)
        #expect(plan.nextHiddenRow == 0)
    }

    @Test("priming iteration (no drafts) commits prev alone and forces replay")
    func primingIteration() {
        let plan = SpecDecode.planMTPAccept(
            prev: 100, drafts: [],
            verifierTokens: [], batchedBonus: 42)
        #expect(plan.committed == [100])
        #expect(plan.needsReplay)
        #expect(plan.nextHiddenRow == 0)
    }

    @Test("per-iteration commit count is always accepted + 1 (never 0)")
    func commitCountInvariant() {
        for (drafts, verifier) in [
            ([Int](), [Int]()),
            ([1], [1]), ([1], [2]),
            ([1, 2, 3, 4], [1, 2, 3, 4]),
            ([1, 2, 3, 4], [9, 2, 3, 4]),
        ] {
            let plan = SpecDecode.planMTPAccept(
                prev: 0, drafts: drafts,
                verifierTokens: verifier, batchedBonus: -1)
            let accepted = zip(drafts, verifier).prefix(while: { $0 == $1 }).count
            #expect(plan.committed.count == accepted + 1)
            #expect(plan.needsReplay == (accepted < drafts.count || drafts.isEmpty))
        }
    }
}

// MARK: - Rejection sampling (lossless temperature support)

/// Pure-math coverage of the standard speculative-sampling accept/residual
/// arithmetic (`Sampling.acceptProbability`, `Sampling.residual`) against
/// hand-computed values, plus `SpecDecode.planMTPAcceptSampled` end-to-end
/// on synthetic logits rows. `topK: 1` is used throughout the plan-level
/// tests specifically because it makes the filtered distributions
/// EXACTLY one-hot (probability 1.0 for the surviving token, EXACTLY 0.0
/// for everyone else, not just "small") — the accept/reject outcome is
/// then deterministic for every draw in `Double.random(in: 0..<1)`'s
/// range, so the tests don't depend on the seeded RNG's specific stream.
@Suite("MTP rejection sampling — sampler math")
struct MTPRejectionSamplingTests {

    // ─── Sampling.acceptProbability ───────────────────────────────────

    @Test("accept probability is min(1, p/q)")
    func acceptProbabilityRatio() {
        // p > q: ratio > 1, clamped to 1.
        #expect(Sampling.acceptProbability(p: 0.5, q: 0.2) == 1.0)
        // p < q: ratio < 1, passed through.
        #expect(abs(Sampling.acceptProbability(p: 0.3, q: 0.5) - 0.6) < 1e-12)
        #expect(abs(Sampling.acceptProbability(p: 0.2, q: 0.3) - (0.2 / 0.3)) < 1e-12)
        // p == q: ratio exactly 1.
        #expect(Sampling.acceptProbability(p: 0.4, q: 0.4) == 1.0)
        // p == 0: always reject.
        #expect(Sampling.acceptProbability(p: 0.0, q: 0.5) == 0.0)
    }

    @Test("accept probability rejects (rather than divides by zero) when q <= 0")
    func acceptProbabilityDegenerateQ() {
        #expect(Sampling.acceptProbability(p: 0.5, q: 0.0) == 0.0)
        #expect(Sampling.acceptProbability(p: 0.0, q: 0.0) == 0.0)
    }

    // ─── Sampling.residual ─────────────────────────────────────────────

    @Test("residual is normalize(max(0, p - q)), single-support case")
    func residualSingleSupport() {
        // p - q = [0.3, -0.2, -0.1] -> max(0, .) = [0.3, 0, 0] -> normalize -> [1, 0, 0].
        let r = Sampling.residual(p: [0.5, 0.3, 0.2], q: [0.2, 0.5, 0.3])
        #expect(abs(r[0] - 1.0) < 1e-12)
        #expect(abs(r[1] - 0.0) < 1e-12)
        #expect(abs(r[2] - 0.0) < 1e-12)
    }

    @Test("residual is normalize(max(0, p - q)), multi-support case")
    func residualMultiSupport() {
        // p - q = [0.3, 0.2, -0.5] -> max(0, .) = [0.3, 0.2, 0], sum 0.5
        // -> normalize -> [0.6, 0.4, 0].
        let r = Sampling.residual(p: [0.5, 0.5, 0.0], q: [0.2, 0.3, 0.5])
        #expect(abs(r[0] - 0.6) < 1e-12)
        #expect(abs(r[1] - 0.4) < 1e-12)
        #expect(abs(r[2] - 0.0) < 1e-12)
        // Always a valid probability distribution.
        #expect(abs(r.reduce(0, +) - 1.0) < 1e-12)
    }

    @Test("residual is all-zero when p == q everywhere (degenerate; shouldn't occur post-reject)")
    func residualDegenerateEqual() {
        let r = Sampling.residual(p: [0.5, 0.5], q: [0.5, 0.5])
        #expect(r == [0.0, 0.0])
    }

    // ─── SpecDecode.planMTPAcceptSampled — deterministic end-to-end ────

    private static let onehotParams = GenerationParameters().with {
        $0.temperature = 1.0
        $0.topK = 1  // keep exactly the argmax -> exact 0/1 probabilities
        $0.topP = 1.0
        $0.minP = 0.0
    }

    @Test("full accept when q's drafted token IS p's argmax (accept prob == 1 exactly)")
    func fullAcceptDeterministic() {
        // q and p both collapse (topK=1) to token 1 -- the drafted token
        // matches p's own argmax, so p(x)/q(x) == 1/1 == 1: ALWAYS accept,
        // for any u in [0, 1). Bonus row collapses to token 3.
        let qRow: [Float] = [-100, 100, -100, -100]
        let pRow: [Float] = [-100, 100, -100, -100]
        let bonusRow: [Float] = [-100, -100, -100, 100]
        var rng = AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: 42))
        let qDist = Sampling.FilteredDistribution(rawValues: qRow, parameters: Self.onehotParams)
        let plan = SpecDecode.planMTPAcceptSampled(
            prev: 999, drafts: [1], draftQDistributions: [qDist],
            verifyFlat: pRow + bonusRow, rows: 2, vocab: 4,
            parameters: Self.onehotParams, rng: &rng)
        #expect(plan.committed == [999, 1])
        #expect(plan.nextToken == 3)
        #expect(!plan.needsReplay)
        #expect(plan.nextHiddenRow == 1)
    }

    @Test("reject + exact residual sample when p filters the drafted token to exactly 0")
    func rejectWithResidualDeterministic() {
        // q collapses to token 1 (the drafted token); p collapses to
        // token 0 -- p(token 1) is EXACTLY 0 under top-K=1, so accept
        // probability is exactly 0: ALWAYS reject, for any u in [0, 1).
        // Residual = normalize(max(0, [1,0,0,0] - [0,1,0,0])) = [1,0,0,0],
        // so the resample is token 0 with certainty.
        let qRow: [Float] = [-100, 100, -100, -100]
        let pRow: [Float] = [100, -100, -100, -100]
        let bonusRow: [Float] = [0, 0, 0, 0]  // unused on a reject
        var rng = AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: 7))
        let qDist = Sampling.FilteredDistribution(rawValues: qRow, parameters: Self.onehotParams)
        let plan = SpecDecode.planMTPAcceptSampled(
            prev: 999, drafts: [1], draftQDistributions: [qDist],
            verifyFlat: pRow + bonusRow, rows: 2, vocab: 4,
            parameters: Self.onehotParams, rng: &rng)
        #expect(plan.committed == [999])
        #expect(plan.nextToken == 0)
        #expect(plan.needsReplay)
        #expect(plan.nextHiddenRow == 0)
    }

    @Test("priming iteration (no drafts) commits prev alone, samples bonus, forces replay")
    func primingIterationSampled() {
        let bonusRow: [Float] = [-100, -100, 100, -100]
        var rng = AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: 1))
        let plan = SpecDecode.planMTPAcceptSampled(
            prev: 999, drafts: [], draftQDistributions: [],
            verifyFlat: bonusRow, rows: 1, vocab: 4,
            parameters: Self.onehotParams, rng: &rng)
        #expect(plan.committed == [999])
        #expect(plan.needsReplay)
        #expect(plan.nextHiddenRow == 0)
    }

    // ─── SpecDecode.planMTPAcceptMatch — sample-target-then-match ─────
    //
    // The new DEFAULT sampled-mode scheme (SAMPLING_RESEARCH.md #1):
    // greedy drafts (tested at the `mtpDecodeLoop` level, not here —
    // this covers only the accept-plan half) + one pre-sampled draw per
    // verify row, matched/replaced via `planMTPAccept`'s own
    // longest-matching-prefix machinery. `topK: 1` one-hot rows again
    // make the pre-sample deterministic for any RNG draw, same trick as
    // the ratio-scheme tests above.

    @Test("match scheme: full accept when draft equals every row's one-hot pre-sample")
    func matchSchemeFullAccept() {
        let row0: [Float] = [-100, 100, -100, -100]  // token 1
        let row1: [Float] = [-100, -100, 100, -100]  // token 2
        let bonusRow: [Float] = [-100, -100, -100, 100]  // token 3
        var rng = AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: 42))
        let plan = SpecDecode.planMTPAcceptMatch(
            prev: 999, drafts: [1, 2],
            verifyFlat: row0 + row1 + bonusRow, rows: 3, vocab: 4,
            parameters: Self.onehotParams, rng: &rng)
        #expect(plan.committed == [999, 1, 2])
        #expect(plan.nextToken == 3)
        #expect(!plan.needsReplay)
        #expect(plan.nextHiddenRow == 2)
    }

    @Test("match scheme: mismatch commits the pre-sampled target token as the correction (no residual math)")
    func matchSchemeMismatchCorrection() {
        let row0: [Float] = [-100, 100, -100, -100]  // token 1 -- draft matches
        let row1: [Float] = [-100, -100, 100, -100]  // token 2 -- draft (9) mismatches
        let bonusRow: [Float] = [0, 0, 0, 0]  // unused on a partial accept
        var rng = AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: 7))
        let plan = SpecDecode.planMTPAcceptMatch(
            prev: 999, drafts: [1, 9],
            verifyFlat: row0 + row1 + bonusRow, rows: 3, vocab: 4,
            parameters: Self.onehotParams, rng: &rng)
        #expect(plan.committed == [999, 1])
        // The correction IS the pre-sampled target token at the
        // mismatch row -- already a lawful draw from p_1, no residual
        // math (unlike the ratio scheme's resample-from-residual).
        #expect(plan.nextToken == 2)
        #expect(plan.needsReplay)
        #expect(plan.nextHiddenRow == 1)
    }

    @Test("match scheme: immediate mismatch at row 0 commits prev alone")
    func matchSchemeImmediateMismatch() {
        let row0: [Float] = [-100, 100, -100, -100]  // token 1
        let bonusRow: [Float] = [0, 0, 0, 0]
        var rng = AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: 3))
        let plan = SpecDecode.planMTPAcceptMatch(
            prev: 999, drafts: [7],
            verifyFlat: row0 + bonusRow, rows: 2, vocab: 4,
            parameters: Self.onehotParams, rng: &rng)
        #expect(plan.committed == [999])
        #expect(plan.nextToken == 1)
        #expect(plan.needsReplay)
        #expect(plan.nextHiddenRow == 0)
    }

    @Test("match scheme: priming iteration commits prev alone, samples bonus from p, forces replay")
    func matchSchemePriming() {
        let bonusRow: [Float] = [-100, -100, 100, -100]  // token 2
        var rng = AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: 1))
        let plan = SpecDecode.planMTPAcceptMatch(
            prev: 999, drafts: [],
            verifyFlat: bonusRow, rows: 1, vocab: 4,
            parameters: Self.onehotParams, rng: &rng)
        #expect(plan.committed == [999])
        #expect(plan.needsReplay)
        #expect(plan.nextHiddenRow == 0)
    }

    @Test(
        "match scheme: row-order pre-sampling doesn't depend on the draft content (deterministic RNG consumption)"
    )
    func matchSchemeRowOrderIndependentOfDrafts() {
        // Non-one-hot rows (topK=2) so the pre-sample is a genuine
        // categorical draw, not a foregone conclusion -- this is what
        // actually exercises the "row order j=0..gamma" determinism
        // claim from the design: if planMTPAcceptMatch sampled in a
        // different order, or skipped rows past the first mismatch, the
        // draws below would depend on what was drafted.
        let weightedParams = GenerationParameters().with {
            $0.temperature = 1.0
            $0.topK = 2
            $0.topP = 1.0
            $0.minP = 0.0
        }
        let row0: [Float] = [1.0, 0.5, -100, -100]
        let row1: [Float] = [0.2, 0.8, -100, -100]
        let bonusRow: [Float] = [0.9, 0.1, -100, -100]
        let flat = row0 + row1 + bonusRow

        // Ground truth: sample row0, row1, bonus in that fixed order
        // from a fresh seed -- the only order planMTPAcceptMatch should
        // ever produce, regardless of what's drafted.
        var rngTruth = AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: 4242))
        let t0 = Sampling.FilteredDistribution(rawValues: row0, parameters: weightedParams)
            .sample(&rngTruth)
        let t1 = Sampling.FilteredDistribution(rawValues: row1, parameters: weightedParams)
            .sample(&rngTruth)
        let tBonus = Sampling.FilteredDistribution(rawValues: bonusRow, parameters: weightedParams)
            .sample(&rngTruth)

        // Variant A: drafts happen to equal the pre-samples -> full accept.
        var rngA = AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: 4242))
        let planA = SpecDecode.planMTPAcceptMatch(
            prev: 500, drafts: [t0, t1],
            verifyFlat: flat, rows: 3, vocab: 4,
            parameters: weightedParams, rng: &rngA)
        #expect(planA.committed == [500, t0, t1])
        #expect(planA.nextToken == tBonus)
        #expect(!planA.needsReplay)

        // Variant B: drafts mismatch immediately -- the SAME row0
        // pre-sample (t0) must still surface as the mismatch
        // correction, proving the target draws are computed from
        // verifyFlat/rng alone, independent of the draft content.
        var rngB = AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: 4242))
        let mismatchDraft = (t0 == 0) ? 1 : 0
        let planB = SpecDecode.planMTPAcceptMatch(
            prev: 500, drafts: [mismatchDraft, t1],
            verifyFlat: flat, rows: 3, vocab: 4,
            parameters: weightedParams, rng: &rngB)
        #expect(planB.committed == [500])
        #expect(planB.nextToken == t0)
        #expect(planB.needsReplay)
    }
}
