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
// T40 follow-up: pure-CPU coverage for the certified lm_head prune's sampling-path
// awareness (Sources/ButterSwift/Models/Text/LagunaLmHeadPrune.swift's
// `samplingMode` / `maskNonCandidates`). Neither depends on Metal/GPU:
// `samplingMode` is a plain decision function over (temperature,
// repetitionPenalty, opt-in flag) and `maskNonCandidates` operates on
// plain `[Float]` certificate arrays, so both are exercised directly
// here without standing up a real Laguna model or command buffer.

import Foundation
import Testing

@testable import ButterSwift

@Suite("LmHeadSamplingPath")
struct LmHeadSamplingPathTests {

    // MARK: - samplingMode: pure path-selection logic

    @Test("greedy (T=0, no rep-penalty) keeps the certified prune")
    func greedyKeepsPrune() {
        let mode = LagunaLmHeadPrune.samplingMode(
            temperature: 0, repetitionPenalty: 1.0,
            candidateRestrictedSamplingEnabled: false)
        #expect(mode == .pruneCertified)
    }

    @Test("greedy keeps the prune even with the candidate-restricted flag on")
    func greedyKeepsPruneRegardlessOfFlag() {
        // The opt-in is scoped to CPU sampling; it must have zero effect
        // on the T=0/no-penalty case, which reduces to a plain argmax
        // either way (see `Sampling.sample`'s greedy short-circuit).
        let mode = LagunaLmHeadPrune.samplingMode(
            temperature: 0, repetitionPenalty: 1.0,
            candidateRestrictedSamplingEnabled: true)
        #expect(mode == .pruneCertified)
    }

    @Test("temperature > 0 default disables the prune (exact full vocab)")
    func temperatureDefaultDisablesPrune() {
        let mode = LagunaLmHeadPrune.samplingMode(
            temperature: 0.7, repetitionPenalty: 1.0,
            candidateRestrictedSamplingEnabled: false)
        #expect(mode == .exactFullVocab)
    }

    @Test("candidate-restricted opt-in flag is honored at temperature > 0")
    func candidateModeFlagHonored() {
        let mode = LagunaLmHeadPrune.samplingMode(
            temperature: 0.7, repetitionPenalty: 1.0,
            candidateRestrictedSamplingEnabled: true)
        #expect(mode == .pruneCandidateRestricted)
    }

    @Test("repetition penalty forces exact even with the candidate flag on")
    func repetitionPenaltyForcesExact() {
        // Repetition penalty rescales logits the certificate never
        // vetted against -- the opt-in must NOT override this, at
        // either temperature.
        let atTempZero = LagunaLmHeadPrune.samplingMode(
            temperature: 0, repetitionPenalty: 1.3,
            candidateRestrictedSamplingEnabled: true)
        #expect(atTempZero == .exactFullVocab)

        let atTempPositive = LagunaLmHeadPrune.samplingMode(
            temperature: 0.7, repetitionPenalty: 1.3,
            candidateRestrictedSamplingEnabled: true)
        #expect(atTempPositive == .exactFullVocab)
    }

    @Test("repetition penalty forces exact with the flag off too")
    func repetitionPenaltyForcesExactFlagOff() {
        let mode = LagunaLmHeadPrune.samplingMode(
            temperature: 0, repetitionPenalty: 0.8,
            candidateRestrictedSamplingEnabled: false)
        #expect(mode == .exactFullVocab)
    }

    // MARK: - maskNonCandidates: exact numeric behavior

    /// Shared synthetic 8-row vocab used by several tests below. `coarse
    /// == assembled` (delta alone carries the certificate's error
    /// bound), `delta` uniform 0.1. `L = max(coarse - delta) = 4.9`,
    /// `thr = L - |L|/64 = 4.823437500`. Rows 0-3 clear
    /// `coarse[i] + delta[i] >= thr` (i.e. `coarse[i] >= 4.7234375`);
    /// rows 4-7 don't -- but sit only 0.4-0.7 below the max, so a naive
    /// *unmasked* softmax over the same array would still give them
    /// real weight (`exp(-0.4...-0.7) ~= 0.50-0.67x` the peak). That
    /// makes this a case where masking + renormalizing actually changes
    /// the distribution, not just zeroes out already-negligible rows.
    private func syntheticCertificate() -> (
        assembled: [Float], coarse: [Float], delta: [Float], thr: Float, candidates: [Int]
    ) {
        let assembled: [Float] = [5.0, 4.95, 4.9, 4.85, 4.6, 4.5, 4.4, 4.3]
        let coarse = assembled
        let delta = [Float](repeating: 0.1, count: assembled.count)
        let lowerBound = zip(coarse, delta).map { $0 - $1 }.max()!
        let thr = lowerBound - abs(lowerBound) / 64
        return (assembled, coarse, delta, thr, [0, 1, 2, 3])
    }

    @Test("maskNonCandidates keeps candidate rows, -infinities the rest")
    func maskKeepsCandidatesOnly() {
        let s = syntheticCertificate()
        let masked = LagunaLmHeadPrune.maskNonCandidates(
            assembled: s.assembled, coarse: s.coarse, delta: s.delta, thr: s.thr)

        for i in s.candidates {
            #expect(masked[i] == s.assembled[i], "candidate row \(i) must be untouched")
        }
        for i in 0 ..< masked.count where !s.candidates.contains(i) {
            #expect(masked[i] == -Float.infinity, "excluded row \(i) must be -infinity")
        }
    }

    @Test("maskNonCandidates includes a row exactly at the threshold (>=, not >)")
    func maskBoundaryEquality() {
        let assembled: [Float] = [1.0, 2.0]
        let coarse: [Float] = [1.0, 2.0]
        let delta: [Float] = [0.0, 0.0]
        let thr: Float = 1.0  // exactly row 0's coarse + delta
        let masked = LagunaLmHeadPrune.maskNonCandidates(
            assembled: assembled, coarse: coarse, delta: delta, thr: thr)
        #expect(masked[0] == 1.0)  // 1.0 >= 1.0 -> candidate, kept exactly
        #expect(masked[1] == 2.0)  // 2.0 >= 1.0 -> candidate, kept exactly
    }

    @Test("maskNonCandidates treats a NaN certificate value as non-candidate")
    func maskNaNIsExcluded() {
        // Mirrors the kernel's `is_cand = (coarse[i]+delta[i] >= thr)`:
        // any IEEE-754 comparison against NaN is false, so a NaN
        // certificate value must fall out as non-candidate, not silently
        // pass through.
        let assembled: [Float] = [1.0, 2.0]
        let coarse: [Float] = [Float.nan, 2.0]
        let delta: [Float] = [0.0, 0.0]
        let thr: Float = 0.5
        let masked = LagunaLmHeadPrune.maskNonCandidates(
            assembled: assembled, coarse: coarse, delta: delta, thr: thr)
        #expect(masked[0] == -Float.infinity)
        #expect(masked[1] == 2.0)
    }

    // MARK: - Renormalization math vs. an independent reference

    /// Independent reference: plain softmax restricted to `candidates`,
    /// normalized to sum to 1. Deliberately does NOT share any code with
    /// `Sampling.sample` or `LagunaLmHeadPrune.maskNonCandidates`, so
    /// comparing against it below is a real cross-check, not a
    /// tautology.
    private func referenceCandidateSoftmax(
        _ logits: [Float], candidates: [Int]
    ) -> [Int: Double] {
        let maxL = candidates.map { logits[$0] }.max()!
        var weights: [Int: Double] = [:]
        var total = 0.0
        for c in candidates {
            let w = Double(expf(logits[c] - maxL))
            weights[c] = w
            total += w
        }
        return weights.mapValues { $0 / total }
    }

    @Test("reference candidate-only softmax renormalizes to sum 1")
    func referenceProbabilitiesSumToOne() {
        let s = syntheticCertificate()
        let reference = referenceCandidateSoftmax(s.assembled, candidates: s.candidates)
        let total = reference.values.reduce(0, +)
        #expect(abs(total - 1.0) < 1e-9)
    }

    @Test(
        "candidate-restricted sampling matches the reference softmax over just the candidate set"
    )
    func candidateRestrictedMatchesReference() {
        let s = syntheticCertificate()
        let masked = LagunaLmHeadPrune.maskNonCandidates(
            assembled: s.assembled, coarse: s.coarse, delta: s.delta, thr: s.thr)
        let reference = referenceCandidateSoftmax(s.assembled, candidates: s.candidates)

        let params = GenerationParameters(temperature: 1.0)
        var rng = SeededRandomNumberGenerator(seed: 0xC0FF_EE)
        var counts: [Int: Int] = [:]
        let draws = 100_000
        for _ in 0 ..< draws {
            let idx = Sampling.sample(
                values: masked, parameters: params, rng: &rng, tokenHistory: [])
            counts[idx, default: 0] += 1
        }

        // Never draws an excluded row -- the restriction is exact, not
        // just "usually small".
        for i in 0 ..< masked.count where !s.candidates.contains(i) {
            #expect(counts[i] == nil, "excluded row \(i) must never be sampled")
        }
        // Empirical frequency tracks the analytically-computed reference
        // probability closely. At p ~= 0.25 and N = 100_000 the sampling
        // std dev is ~0.0014; 0.01 is a wide, non-flaky margin (fixed
        // seed makes this fully deterministic run-to-run regardless).
        for c in s.candidates {
            let empirical = Double(counts[c] ?? 0) / Double(draws)
            let expected = reference[c]!
            #expect(
                abs(empirical - expected) < 0.01,
                "candidate \(c): empirical=\(empirical) expected=\(expected)")
        }
    }
}
