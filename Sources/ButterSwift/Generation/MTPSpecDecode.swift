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
// MTPSpecDecode — the MTP self-speculative token iterator for
// Qwen 3.5 / 3.6 (spec 030 phase 2; `MTPSelfSpeculativeTokenIterator`
// in planning/architecture.md).
//
// Same driver shape as `SpecDecode.generateGreedy` (the n-gram
// draft-verify loop), with the external `Drafter` replaced by the
// checkpoint's own single-layer MTP head (`Qwen3xMTPHead`):
//
//   1. Chain-draft γ candidate tokens with `mtpHead.draftStep`: the
//      first step fuses the backbone's last pre-final-norm hidden with
//      the last committed/sampled token; each subsequent step chains
//      the head's OWN output hidden h' with its previous draft (the
//      DeepSeek-nextn recipe; `mtp_num_hidden_layers = 1`, so the one
//      layer is reused every step). Host commit between steps per the
//      head's threading contract.
//   2. Verify with ONE batched backbone pass —
//      `forwardManyHiddenAndLogits([prev] + drafts)` — which also
//      returns the per-position hidden states the next draft burst and
//      the head's KV lockstep need.
//   3. Accept via `SpeculativeAccept.verify` (longest matching prefix).
//   4. EXACTNESS DISCIPLINE, greedy mode (mirrors DFlashSpecDecode.swift
//      — see that file's header, point 5): the batched verify logits are
//      used ONLY for accept-length decisions, NEVER emission or cache
//      content. On EVERY cycle — partial, zero, AND full accept alike —
//      the emitted next token comes from REPLAYING the accepted prefix
//      through the single-token forward path after a full cache restore
//      (`mtpDecodeLoop`'s `mustReplayForExactness`) — batched kernels
//      can drift vs decode kernels by ULPs even when every drafted
//      token's identity already matched (SpecDecode.swift ~168-176;
//      confirmed here too via `mtpSpecMatchesBaselineExactly`, which
//      caught a full-accept-only divergence before this fix closed it).
//      Sampled mode keeps the original full-accept fast path — the
//      bonus token comes straight from the trailing batched logits row,
//      matching `SpecDecode.generateGreedy`'s bonus handling — since
//      sampled decode's losslessness contract doesn't depend on
//      kernel-level bit-exactness the way greedy's baseline-matching
//      gate does (see `mtpDecodeLoop`'s `mustReplayForExactness` doc
//      comment).
//   5. MTP KV LOCKSTEP: the draft head runs as a normal decoder layer
//      with a persistent KV cache mirroring the backbone's committed
//      history (see Qwen3xMTPHead's header). After each iteration the
//      burst's draft rows are rolled back (`kvCache.truncate`) and the
//      head is re-advanced over every committed token using the hidden
//      states the verify pass returned — one draftStep-shaped pass per
//      committed token, logits discarded (the simple per-token version;
//      BUTTER_QWEN_MTP_BATCHED_LOCKSTEP=1 batches this into ONE
//      T-batched `Qwen3xMTPHead.draftStepMany` pass instead, off by
//      default).
//
// PRIMING: the loop needs the backbone hidden at `position - 1` to
// seed the first draft burst. When the caller can't supply it (the
// standard `Generate.swift` prefill path only returns logits), the
// first iteration runs draft-free: snapshot, a 1-token
// `forwardManyHiddenAndLogits([prev])` to capture the hidden, then the
// standard restore + single-token replay so the emitted token stays on
// the exact decode path. The head's KV history simply misses the
// pre-priming rows (prompt + the priming position) — positions are
// passed through unchanged so later rows still line up; this only
// (mildly) affects draft quality, never correctness.
//
// Greedy-only, like `generateGreedy` — argmax everywhere.

import Foundation
import Metal

/// Stats for one `generateGreedyMTP` run. Mirrors `SpecDecodeStats`
/// plus the MTP-specific replay / lockstep costs.
public struct MTPSpecDecodeStats {
    public let tokensGenerated: Int
    /// Batched verify passes run (incl. the priming pass, which
    /// proposes zero candidates).
    public let verifySteps: Int
    public let candidatesProposed: Int
    public let candidatesAccepted: Int
    /// Single-token backbone forwards run to replay accepted prefixes
    /// on partial/zero accepts (the exactness discipline).
    public let replaySingleSteps: Int
    /// Draft-head passes spent advancing the MTP KV cache over
    /// committed tokens (the lockstep cost; excludes draft-burst steps).
    public let lockstepSteps: Int
    public let wallclockSeconds: Double

    public var acceptanceRate: Double {
        guard candidatesProposed > 0 else { return 0 }
        return Double(candidatesAccepted) / Double(candidatesProposed)
    }
    public var tps: Double {
        guard wallclockSeconds > 0 else { return 0 }
        return Double(tokensGenerated) / wallclockSeconds
    }
}

extension SpecDecode {

    /// Pure accept-arithmetic for one MTP spec iteration — split out of
    /// the loop so the commit/replay bookkeeping is unit-testable
    /// without a model. `drafts` may be empty (the priming iteration).
    struct MTPAcceptPlan: Equatable {
        /// Tokens committed this iteration, in order:
        /// `[prev] + acceptedDraftPrefix`.
        let committed: [Int]
        /// Next iteration's seed token. From the batched bonus row on a
        /// full accept; from the single-token replay otherwise (the
        /// value here is the *accept-time* candidate — the replay path
        /// recomputes it on the exact decode path before use).
        let nextToken: Int
        /// True on a partial/zero accept: restore every cache and
        /// replay `committed` through the single-token forward path.
        let needsReplay: Bool
        /// Row of the verify pass's `[T, hidden]` output that feeds the
        /// next draft burst (the hidden at the last committed position).
        let nextHiddenRow: Int
    }

    /// Adaptive per-cycle draft-depth (gamma) controller, gated on
    /// REALIZED trailing acceptance rather than drafter self-confidence
    /// — `BUTTER_QWEN_MTP_PMIN` (this file's confidence gate) was
    /// measured uninformative on the Qwen3.6 MTP head (it's confident
    /// everywhere); acceptance instead decays via CONTENT DRIFT over
    /// generation length (the `BUTTER_QWEN_MTP_ACCEPT_CURVE=1`
    /// instrumentation: per-64-token buckets oscillate 73->47% over 512
    /// tokens). Realized accept/reject outcomes are the honest signal.
    /// `BUTTER_QWEN_MTP_ADAPTIVE=1` wires this into `mtpDecodeLoop` —
    /// see its env-var block for the knobs (window/lo/hi) and default
    /// values.
    ///
    /// Pure logic, no model/tensor dependency — unit-testable directly
    /// (`MTPAdaptiveGammaTests`). Never affects accept semantics: it
    /// only chooses how many tokens the NEXT burst drafts, so
    /// LOSSLESSNESS is untouched (see the file header's exactness
    /// discipline) — a bad gamma choice costs speed, never correctness.
    struct MTPAdaptiveGamma {
        /// Ceiling: the loop's configured `gamma`. Effective gamma never
        /// exceeds this.
        let cap: Int
        /// Trailing window size, in per-candidate accept/reject outcomes
        /// (not cycles — a cycle can propose 0..cap candidates).
        let window: Int
        /// Step gamma DOWN by 1 (floor 1) once trailing acceptance falls
        /// strictly below this.
        let lo: Double
        /// Step gamma UP by 1 (ceiling `cap`) once trailing acceptance
        /// rises strictly above this. Requires `lo < hi`; the band
        /// between them is the hysteresis zone — trailing rates inside
        /// it hold gamma steady instead of adjusting every cycle.
        let hi: Double

        /// Effective gamma for the cycle about to draft. Starts at
        /// `cap` and is stepped by at most 1 per `record` call.
        private(set) var gamma: Int

        // Trailing window: fixed-capacity ring buffer of per-candidate
        // accept outcomes (true = accepted, false = rejected or never
        // reached — see `record`'s doc). `ringFilled` is the number of
        // valid slots so far, capped at `window`; `trueCount` is
        // maintained incrementally so the trailing rate is O(1) per
        // push rather than a full rescan.
        private var ring: [Bool]
        private var ringNext = 0
        private var ringFilled = 0
        private var trueCount = 0

        // Cross-cycle stats for the `[mtp-prof]` line (avg/min/max
        // effective gamma actually used).
        private(set) var gammaCycleSum = 0
        private(set) var gammaCycles = 0
        private(set) var gammaMin = Int.max
        private(set) var gammaMax = 0

        init(cap: Int, window: Int, lo: Double, hi: Double) {
            precondition(cap >= 1, "MTPAdaptiveGamma: cap must be >= 1")
            precondition(window >= 1, "MTPAdaptiveGamma: window must be >= 1")
            precondition(
                lo < hi, "MTPAdaptiveGamma: lo (\(lo)) must be < hi (\(hi))")
            self.cap = cap
            self.window = window
            self.lo = lo
            self.hi = hi
            self.gamma = cap
            self.ring = Array(repeating: false, count: window)
        }

        /// Effective gamma to draft with THIS cycle — call exactly once
        /// per cycle, before drafting; folds the value into the running
        /// avg/min/max stats as a side effect.
        mutating func nextCycleGamma() -> Int {
            gammaCycleSum += gamma
            gammaCycles += 1
            gammaMin = min(gammaMin, gamma)
            gammaMax = max(gammaMax, gamma)
            return gamma
        }

        /// Record this cycle's per-candidate outcomes and step gamma by
        /// at most 1 for the NEXT cycle. Mirrors the longest-matching-
        /// prefix shape the loop's own `candidatesProposed`/
        /// `candidatesAccepted` tally already uses: of `proposed`
        /// drafted candidates, the first `accepted` were committed and
        /// the rest were not (either verified-and-rejected, or, for a
        /// chained draft past the first rejection, never independently
        /// verified at all — both count as "not accepted" for the
        /// trailing rate, same as the existing global tally). A
        /// draft-free cycle (`proposed == 0`, i.e. the priming
        /// iteration) is a no-op — nothing to record, gamma unchanged.
        mutating func record(accepted: Int, proposed: Int) {
            guard proposed > 0 else { return }
            precondition(
                accepted >= 0 && accepted <= proposed,
                "MTPAdaptiveGamma.record: accepted (\(accepted)) out of "
                    + "range for proposed (\(proposed))")
            for i in 0 ..< proposed {
                push(i < accepted)
            }
            // Warmup: hold at the cap until the window has observed a
            // full `window` candidates — the trailing rate is noisy
            // (and, early on, computed over a partially-stale/all-false
            // ring) before then.
            guard ringFilled >= window else { return }
            let rate = Double(trueCount) / Double(ringFilled)
            if rate < lo {
                gamma = max(1, gamma - 1)
            } else if rate > hi {
                gamma = min(cap, gamma + 1)
            }
            // rate in [lo, hi]: hold — the hysteresis band.
        }

        private mutating func push(_ accepted: Bool) {
            if ringFilled == window {
                if ring[ringNext] { trueCount -= 1 }
            } else {
                ringFilled += 1
            }
            ring[ringNext] = accepted
            if accepted { trueCount += 1 }
            ringNext = (ringNext + 1) % window
        }

        var avgGamma: Double {
            gammaCycles > 0 ? Double(gammaCycleSum) / Double(gammaCycles) : 0
        }
    }

    /// Per-draft-position cost model adapted from the promoted Qwen3.8
    /// MLX.fast schedule. Position `i` tracks
    /// P(draft i accepted | every earlier draft accepted), then chooses
    /// the depth maximizing expected committed tokens per unit round cost.
    /// A full round transfers capped optimism to the first untried
    /// position so easy prompts can widen without making cold prompts pay.
    struct MTPCostModelGamma {
        let cap: Int
        let headStepCostRatio: Double
        let alpha: Double
        private(set) var positionAcceptEMA: [Double]

        private(set) var gammaCycleSum = 0
        private(set) var gammaCycles = 0
        private(set) var gammaMin = Int.max
        private(set) var gammaMax = 0

        init(cap: Int, headStepCostRatio: Double = 0.22, alpha: Double = 0.15) {
            precondition(cap >= 1, "MTPCostModelGamma: cap must be >= 1")
            precondition(
                headStepCostRatio > 0,
                "MTPCostModelGamma: headStepCostRatio must be positive")
            precondition(
                alpha > 0 && alpha <= 1,
                "MTPCostModelGamma: alpha must be in (0, 1]")
            self.cap = cap
            self.headStepCostRatio = headStepCostRatio
            self.alpha = alpha
            self.positionAcceptEMA = (0 ..< cap).map { 0.85 * pow(0.98, Double($0)) }
        }

        mutating func nextCycleGamma() -> Int {
            let h = headStepCostRatio
            var reach = 1.0
            var expected = 0.0
            var depth = 0
            while depth < cap {
                reach *= positionAcceptEMA[depth]
                let threshold = h * (1.0 + expected) / (1.0 + Double(depth) * h)
                guard reach > threshold else { break }
                expected += reach
                depth += 1
            }
            gammaCycleSum += depth
            gammaCycles += 1
            gammaMin = min(gammaMin, depth)
            gammaMax = max(gammaMax, depth)
            return depth
        }

        mutating func record(accepted: Int, proposed: Int) {
            guard proposed > 0 else { return }
            precondition(
                accepted >= 0 && accepted <= proposed,
                "MTPCostModelGamma.record: accepted out of range")
            for index in 0 ..< min(accepted, positionAcceptEMA.count) {
                positionAcceptEMA[index] += alpha * (1.0 - positionAcceptEMA[index])
            }
            if accepted < proposed, accepted < positionAcceptEMA.count {
                positionAcceptEMA[accepted] += alpha * (0.0 - positionAcceptEMA[accepted])
            } else if accepted == proposed, accepted < positionAcceptEMA.count,
                positionAcceptEMA[accepted] < 0.95
            {
                positionAcceptEMA[accepted] += alpha * (0.95 - positionAcceptEMA[accepted])
            }
        }

        var avgGamma: Double {
            gammaCycles > 0 ? Double(gammaCycleSum) / Double(gammaCycles) : 0
        }
    }

    /// Compute the commit plan from the draft block and the verifier's
    /// own (argmax) tokens. `verifierTokens[j]` verifies `drafts[j]`;
    /// `batchedBonus` is the verifier's token one past the draft block.
    static func planMTPAccept(
        prev: Int, drafts: [Int], verifierTokens: [Int], batchedBonus: Int
    ) -> MTPAcceptPlan {
        if drafts.isEmpty {
            // Priming / draft-free iteration: commit `prev` alone and
            // force the replay path (nothing verified backs the batched
            // pass, so the emitted next token must come from the exact
            // single-token forward).
            return MTPAcceptPlan(
                committed: [prev], nextToken: batchedBonus,
                needsReplay: true, nextHiddenRow: 0)
        }
        let outcome = SpeculativeAccept.verify(
            draft: drafts, verifierTokens: verifierTokens,
            bonusToken: batchedBonus)
        let k = outcome.acceptedDraft.count
        let committed = [prev] + outcome.acceptedDraft
        // On a FULL accept the next token is `batchedBonus`, taken from
        // the batched verify logits and emitted without ever passing
        // through the single-token decode path. If the two kernels
        // disagree on an argmax tie the stream diverges from baseline
        // greedy. BUTTER_QWEN_MTP_ALWAYS_REPLAY=1 forces the replay path
        // on every iteration to isolate that as the divergence source.
        let forceReplay =
            ButterEnv["BUTTER_QWEN_MTP_ALWAYS_REPLAY"] == "1"
        return MTPAcceptPlan(
            committed: committed,
            nextToken: outcome.bonusToken,
            needsReplay: forceReplay || k < drafts.count,
            nextHiddenRow: committed.count - 1)
    }

    /// Compute the commit plan via lossless rejection sampling instead
    /// of exact-match verification. Standard speculative-sampling
    /// (Leviathan et al.), same `MTPAcceptPlan` shape as
    /// `planMTPAccept` so every downstream consumer (replay, KV
    /// lockstep, commit/emit) is unmodified — only the accept DECISION
    /// differs.
    ///
    /// SUPERSEDED as the sampled path's default by `planMTPAcceptMatch`
    /// (see its doc comment): sampled drafts + a p/q ratio test give
    /// per-position acceptance Σ min(p,q), which is *worse* than greedy
    /// drafts' p(argmax q) once temperature softens both distributions
    /// (measured 53.6% here vs vLLM's 87.9% / llama.cpp's 0.72-0.83 on
    /// the same model class with greedy drafts — see
    /// `SAMPLING_RESEARCH.md` §2). Kept as a correctness oracle behind
    /// `BUTTER_QWEN_MTP_ACCEPT=ratio` — `mtpDecodeLoop` only reaches
    /// this function (and only samples drafts from q, see its `.sampled`
    /// branch) when that env var is set; the default sampled path calls
    /// `planMTPAcceptMatch` instead.
    ///
    /// Per drafted token x_j (j = 0..<drafts.count): reuse q_j from
    /// `draftQDistributions[j]` — the SAME `Sampling.FilteredDistribution`
    /// the draft step already built to sample x_j from q_j in the first
    /// place (see `mtpDecodeLoop`'s `.sampled` branch) — and build p_j
    /// fresh from `verifyFlat`'s row j (the batched verify pass), through
    /// the IDENTICAL filter pipeline (same temp/top-K/top-P/min-P as the
    /// non-spec sampled path). Passing in the already-built q_j instead
    /// of the raw logits row avoids rebuilding (and, pre-T43, re-sorting)
    /// the SAME distribution twice per draft — once at draft time, once
    /// here — which used to roughly double this path's CPU cost on top
    /// of the O(vocab log vocab) sort itself (see Sampling.swift's T43
    /// perf note).
    ///
    /// Accept x_j with probability `min(1, p_j(x_j)/q_j(x_j))`. On the
    /// first rejection, resample the replacement from the residual
    /// `normalize(max(0, p_j - q_j))` over the full vocab and stop —
    /// this is what makes the algorithm lossless (the emitted token's
    /// marginal distribution is exactly p_j regardless of q_j). If
    /// every draft is accepted, sample the bonus token fresh from
    /// `verifyFlat`'s trailing row (p_gamma), matching the greedy
    /// path's full-accept bonus handling.
    ///
    /// Repetition penalty is NOT applied (`Sampling.buildFilteredWeights`
    /// doesn't take a token history) — callers gate the sampled MTP path
    /// on `repetitionPenalty == 1.0` (see `Generate.swift`).
    static func planMTPAcceptSampled(
        prev: Int, drafts: [Int], draftQDistributions: [Sampling.FilteredDistribution],
        verifyFlat: [Float], rows: Int, vocab: Int,
        parameters: GenerationParameters, rng: inout AnyRandomNumberGenerator
    ) -> MTPAcceptPlan {
        precondition(
            drafts.count == draftQDistributions.count,
            "planMTPAcceptSampled: drafts/draftQDistributions length mismatch")
        precondition(
            verifyFlat.count == rows * vocab,
            "planMTPAcceptSampled: verifyFlat has \(verifyFlat.count) elements; "
                + "expected rows*vocab = \(rows * vocab)")

        func bonusDistribution() -> Sampling.FilteredDistribution {
            Sampling.FilteredDistribution(
                rawValues: Array(verifyFlat[(rows - 1) * vocab ..< rows * vocab]),
                parameters: parameters)
        }

        if drafts.isEmpty {
            // Priming / draft-free iteration: same contract as
            // `planMTPAccept` — commit `prev` alone, force replay. The
            // candidate here is discarded by the caller's replay
            // handling exactly like the greedy path's `batchedBonus`
            // placeholder (see the file header's PRIMING note); still
            // sample it from p (not argmax) so a caller that skipped
            // the discard would see a temperature-consistent value.
            let bonus = bonusDistribution().sample(&rng)
            return MTPAcceptPlan(
                committed: [prev], nextToken: bonus,
                needsReplay: true, nextHiddenRow: 0)
        }

        var accepted: [Int] = []
        var nextToken = -1
        var j = 0
        while j < drafts.count {
            let x = drafts[j]
            let q = draftQDistributions[j]
            let p = Sampling.FilteredDistribution(
                rawValues: Array(verifyFlat[j * vocab ..< (j + 1) * vocab]),
                parameters: parameters)
            let acceptProb = Sampling.acceptProbability(
                p: p.probability(of: x), q: q.probability(of: x))
            let u = Double.random(in: 0 ..< 1, using: &rng)
            if u < acceptProb {
                accepted.append(x)
                j += 1
                continue
            }
            // First rejection: resample from the residual over the
            // FULL vocab and stop — later drafts are discarded (they
            // were chained off a draft hidden state that's now wrong).
            let residual = Sampling.residual(
                p: p.denseVector(vocab: vocab), q: q.denseVector(vocab: vocab))
            nextToken = Sampling.sampleFromVector(residual, rng: &rng)
            break
        }
        if accepted.count == drafts.count {
            // Full accept: bonus token sampled fresh from p_gamma, the
            // trailing verify row one past the drafted block.
            nextToken = bonusDistribution().sample(&rng)
        }
        let committed = [prev] + accepted
        return MTPAcceptPlan(
            committed: committed,
            nextToken: nextToken,
            needsReplay: accepted.count < drafts.count,
            nextHiddenRow: committed.count - 1)
    }

    /// Compute the commit plan via "sample-target-then-match" — the
    /// DEFAULT sampled-mode accept scheme (`SAMPLING_RESEARCH.md` #1),
    /// replacing `planMTPAcceptSampled`'s p/q ratio test. Pairs with
    /// GREEDY drafts (`mtpDecodeLoop`'s `.sampled` branch draws `drafts`
    /// via argmax, identical to `.greedy` mode, when
    /// `BUTTER_QWEN_MTP_ACCEPT` isn't "ratio" — no q is built or passed
    /// in here at all).
    ///
    /// For every verify row j = 0...gamma (drafted rows THEN the
    /// trailing bonus row, in that order so RNG consumption stays
    /// deterministic regardless of where the draft/target match later
    /// breaks): build p_j — the SAME temp/top-K/top-P/min-P filtered
    /// distribution `Sampling.sample` uses, via
    /// `Sampling.buildFilteredWeights` (heap-fast since T43) — and
    /// PRE-SAMPLE one token t_j from it with `FilteredDistribution.sample`
    /// (one categorical draw per row).
    ///
    /// The accept/correct/bonus rule is then EXACTLY `planMTPAccept`'s
    /// longest-matching-prefix machinery (`SpeculativeAccept.verify`)
    /// with `t_j` standing in for the verifier's argmax: accept the
    /// longest prefix where `drafts[j] == t_j`; the first mismatch
    /// commits `t_j` itself (already a lawful draw from p_j — no
    /// residual math needed, unlike the ratio scheme); a full accept's
    /// bonus is `t_gamma`, the trailing row's pre-sample. This is why no
    /// separate plan struct or bookkeeping exists here — it's the
    /// greedy plan's own arithmetic, just fed sampled rather than
    /// argmax'd verifier tokens.
    ///
    /// Losslessness is by construction: every emitted token (matched
    /// prefix, mismatch correction, and full-accept bonus alike) is
    /// literally a draw from its row's filtered target distribution —
    /// there is nothing left to prove, unlike the ratio scheme's
    /// min(1,p/q)-plus-residual argument.
    ///
    /// Priming iteration (`drafts` empty, `rows == 1`): the loop below
    /// runs zero times and `bonusDist` is built from the lone row, so
    /// this samples the priming bonus from p rather than taking an
    /// argmax — `planMTPAccept`'s `drafts.isEmpty` branch then commits
    /// `prev` alone and forces replay, same contract as
    /// `planMTPAcceptSampled`'s priming case.
    static func planMTPAcceptMatch(
        prev: Int, drafts: [Int],
        verifyFlat: [Float], rows: Int, vocab: Int,
        parameters: GenerationParameters, rng: inout AnyRandomNumberGenerator
    ) -> MTPAcceptPlan {
        precondition(
            verifyFlat.count == rows * vocab,
            "planMTPAcceptMatch: verifyFlat has \(verifyFlat.count) elements; "
                + "expected rows*vocab = \(rows * vocab)")
        precondition(
            rows == drafts.count + 1,
            "planMTPAcceptMatch: rows (\(rows)) must be drafts.count + 1 (\(drafts.count + 1))"
        )

        var verifierTokens: [Int] = []
        verifierTokens.reserveCapacity(drafts.count)
        for j in 0 ..< drafts.count {
            let p = Sampling.FilteredDistribution(
                rawValues: Array(verifyFlat[j * vocab ..< (j + 1) * vocab]),
                parameters: parameters)
            verifierTokens.append(p.sample(&rng))
        }
        let bonusDist = Sampling.FilteredDistribution(
            rawValues: Array(verifyFlat[(rows - 1) * vocab ..< rows * vocab]),
            parameters: parameters)
        let batchedBonus = bonusDist.sample(&rng)

        return planMTPAccept(
            prev: prev, drafts: drafts,
            verifierTokens: verifierTokens, batchedBonus: batchedBonus)
    }

    /// Decode-loop accept strategy: `.greedy` (exact-match verification,
    /// deterministic) or `.sampled` (lossless rejection sampling,
    /// requires `temperature > 0`). Both share `mtpDecodeLoop` below —
    /// this is the ONLY thing that differs between
    /// `generateGreedyMTP` and `generateSampledMTP`.
    enum MTPDecodeMode {
        case greedy
        case sampled(GenerationParameters)

        var isGreedy: Bool {
            if case .greedy = self { return true }
            return false
        }
    }

    /// Run the greedy MTP self-speculative decode loop — argmax
    /// everywhere, deterministic. Thin wrapper over `mtpDecodeLoop`
    /// (`.greedy` mode); the RNG is unused on this path (nothing draws
    /// from it) but threaded through for a single shared loop body.
    ///
    /// Caller contract matches `generateGreedy`: `caches` already
    /// reflect the prompt, `lastToken` is the most recent
    /// (already-sampled, not yet cache-committed) token, `position` is
    /// where `lastToken` lands on the next decode step.
    ///
    /// - `head`: the checkpoint's MTP draft head (`model.mtpHead`). Its
    ///   KV cache is truncated/advanced by this loop; callers starting
    ///   a fresh generation should `head.resetCache()` first.
    /// - `lastHidden`: backbone pre-final-norm hidden at `position - 1`
    ///   (e.g. the last row of a `forwardManyHiddenAndLogits` prefill).
    ///   Pass `nil` to let the loop prime itself (one extra batched +
    ///   single-token pass; see the file header).
    /// - `emit`: called once per committed token, in stream order —
    ///   the exact tokens the baseline greedy loop would emit. Stop
    ///   tokens are NOT emitted (matching `Generate.driveGeneration`).
    ///
    /// Never emits more than `maxNewTokens` tokens (a full-accept block
    /// is trimmed at the budget, unlike `generateGreedy`'s overshoot).
    public static func generateGreedyMTP(
        model: Qwen35Model,
        head: Qwen3xMTPHead,
        gamma: Int,
        lastToken: Int,
        position: Int,
        lastHidden: Tensor? = nil,
        caches: [any LayerCacheProtocol],
        history: inout [Int],
        maxNewTokens: Int,
        stopTokens: Set<Int> = [],
        device: Device = .shared,
        emit: ((Int) -> Void)? = nil
    ) -> MTPSpecDecodeStats {
        var unusedRNG = AnyRandomNumberGenerator(SystemRandomNumberGenerator())
        return mtpDecodeLoop(
            model: model, head: head, gamma: gamma, mode: .greedy,
            lastToken: lastToken, position: position, lastHidden: lastHidden,
            caches: caches, history: &history, maxNewTokens: maxNewTokens,
            stopTokens: stopTokens, device: device, rng: &unusedRNG, emit: emit)
    }

    /// Run the MTP self-speculative decode loop with lossless
    /// temperature support (`parameters.temperature > 0`) instead of
    /// greedy argmax. Thin wrapper over `mtpDecodeLoop` (`.sampled`
    /// mode); same caller contract as `generateGreedyMTP` otherwise,
    /// plus:
    ///
    /// - `parameters`: sampling knobs (temperature / topP / topK /
    ///   minP). `repetitionPenalty` must be `1.0` — the sampled MTP
    ///   path does not support it (see `Sampling.buildFilteredWeights`);
    ///   `Generate.swift` gates dispatch on this.
    /// - `rng`: drives every draw — pass `parameters.makeRNG()` for the
    ///   same seeded/system-random behavior as the non-spec sampled
    ///   path.
    ///
    /// DEFAULT scheme (see `planMTPAcceptMatch`'s doc):
    /// "sample-target-then-match" — greedy drafts, one pre-sampled draw
    /// per verify row from the filtered target distribution, accept by
    /// match/replace. `BUTTER_QWEN_MTP_ACCEPT=ratio` switches to the
    /// original sampled-draft + min(1,p/q) rejection-sampling scheme
    /// (`planMTPAcceptSampled`'s doc), kept as a correctness oracle —
    /// see `mtpDecodeLoop`'s doc comment for the full fork.
    public static func generateSampledMTP(
        model: Qwen35Model,
        head: Qwen3xMTPHead,
        gamma: Int,
        lastToken: Int,
        position: Int,
        lastHidden: Tensor? = nil,
        caches: [any LayerCacheProtocol],
        history: inout [Int],
        maxNewTokens: Int,
        stopTokens: Set<Int> = [],
        device: Device = .shared,
        parameters: GenerationParameters,
        rng: inout AnyRandomNumberGenerator,
        emit: ((Int) -> Void)? = nil
    ) -> MTPSpecDecodeStats {
        precondition(
            parameters.temperature > 0,
            "SpecDecode.generateSampledMTP: temperature must be > 0 (use generateGreedyMTP for temperature == 0)"
        )
        return mtpDecodeLoop(
            model: model, head: head, gamma: gamma, mode: .sampled(parameters),
            lastToken: lastToken, position: position, lastHidden: lastHidden,
            caches: caches, history: &history, maxNewTokens: maxNewTokens,
            stopTokens: stopTokens, device: device, rng: &rng, emit: emit)
    }

    /// Shared MTP self-speculative decode loop body for both
    /// `generateGreedyMTP` and `generateSampledMTP` — see their doc
    /// comments for the caller contract. `mode` selects exactly THREE
    /// divergence points, marked `── mode:` in the body below: (1) how
    /// each draft token is chosen, (2) how the accept plan is built, (3)
    /// whether a partial/zero-accept replay is allowed to overwrite
    /// `nextTok` via argmax (greedy's baseline-matching exactness
    /// discipline — the sampled path's `nextTok` already came from a
    /// correct draw and must NOT be clobbered). Every other line — draft
    /// chaining, snapshot/restore, the batched verify pass, KV
    /// lockstep, commit/emit, stats/profiling — runs identically
    /// regardless of `mode`.
    ///
    /// `.sampled` itself carries a SECOND, env-gated fork nested inside
    /// divergences (1) and (2) — `BUTTER_QWEN_MTP_ACCEPT=ratio` selects
    /// which sampled scheme runs:
    ///
    ///   - Default (unset, or any value other than "ratio") —
    ///     "sample-target-then-match" (`SAMPLING_RESEARCH.md` #1):
    ///     drafts are argmax, IDENTICAL to `.greedy`'s divergence-1
    ///     branch (no q built at all); the accept plan
    ///     (`planMTPAcceptMatch`) pre-samples one token per verify row
    ///     from the filtered target distribution and reuses
    ///     `planMTPAccept`'s own match/replace machinery with those
    ///     pre-samples standing in for the verifier's argmax tokens.
    ///   - `ratio` — the original scheme: drafts are sampled from q_j
    ///     (`draftQDistributions`), and the accept plan
    ///     (`planMTPAcceptSampled`) runs the standard
    ///     min(1,p_j(x_j)/q_j(x_j)) accept/residual test. Kept only as a
    ///     correctness oracle for the match scheme above — see
    ///     `planMTPAcceptSampled`'s doc comment.
    private static func mtpDecodeLoop(
        model: Qwen35Model,
        head: Qwen3xMTPHead,
        gamma: Int,
        mode: MTPDecodeMode,
        lastToken: Int,
        position: Int,
        lastHidden: Tensor? = nil,
        caches: [any LayerCacheProtocol],
        history: inout [Int],
        maxNewTokens: Int,
        stopTokens: Set<Int> = [],
        device: Device = .shared,
        rng: inout AnyRandomNumberGenerator,
        emit: ((Int) -> Void)? = nil
    ) -> MTPSpecDecodeStats {
        precondition(gamma >= 1, "SpecDecode.mtpDecodeLoop: gamma must be ≥ 1")
        precondition(
            maxNewTokens >= 0,
            "SpecDecode.mtpDecodeLoop: maxNewTokens must be ≥ 0")

        // Resolved once per run — feeds `Qwen3xMTPHead.draftStep`'s
        // `isGreedy` param (`.greedyOnly` draft-lm-head mode). Matches
        // the temperature==0 definition `generateGreedyMTP` /
        // `generateSampledMTP` already assert on entry.
        let isGreedyRun: Bool
        switch mode {
        case .greedy: isGreedyRun = true
        case .sampled: isGreedyRun = false
        }

        var pos = position
        var prev = lastToken
        var prevHidden = lastHidden
        var tokensGenerated = 0
        var verifySteps = 0
        var candidatesProposed = 0
        var candidatesAccepted = 0
        var replaySingleSteps = 0
        var lockstepSteps = 0
        let t0 = Date()
        // BUTTER_QWEN_MTP_PROF=1 attribution buckets (seconds).
        var msDraft = 0.0, msVerify = 0.0, msReplay = 0.0
        var msLockstep = 0.0, msSnap = 0.0
        let stateRewind =
            ButterEnv["BUTTER_QWEN_MTP_STATE_REWIND"] != "0"
        var rewindSteps = 0
        // State planes are enabled by default; BUTTER_GDN_STATE_PLANES=0
        // opts the fused conv+scan kernels out of writing per-token state
        // planes (see
        // `Qwen35GDNLayer.forwardManyChunked`) — tried FIRST on partial
        // accept, ahead of `stateRewind`: `model.adoptPlanes` selects the
        // `(keep-1)`th captured plane directly into the live cache
        // buffers, skipping both the pre-verify snapshot restore AND the
        // conv+scan re-run `rewindCaches` needs. Falls back to
        // `stateRewind` (if set) then full replay when planes weren't
        // captured for this cycle (e.g. `t` exceeded the plane-eligible
        // cap — see `forwardManyChunked`'s `maxPlaneT`).
        let statePlanes =
            ButterEnv["BUTTER_GDN_STATE_PLANES"] != "0"
        var planeSteps = 0
        let batchedReplay =
            ButterEnv["BUTTER_QWEN_MTP_BATCHED_REPLAY"] != "0"
        // Batched lockstep is the Qwen3.8 default; set
        // BUTTER_QWEN_MTP_BATCHED_LOCKSTEP=0 to restore the sequential
        // diagnostic path. It runs the post-accept MTP
        // KV re-advance (below) as ONE T-batched `draftStepMany` pass
        // instead of k+1 sequential `draftStep` calls. The flag is
        // read fresh on every call (no hoisted/cached global) so a
        // per-process env change always takes effect.
        let batchedLockstep =
            ButterEnv["BUTTER_QWEN_MTP_BATCHED_LOCKSTEP"] != "0"
        let parityProbe =
            ButterEnv["BUTTER_QWEN_MTP_PARITY"] == "1"
        var parityChecked = 0
        var parityMismatch = 0
        var parityMismatchMargin = 0.0
        var parityMatchMargin = 0.0
        var parityRowChecked: [Int] = []
        var parityRowMismatch: [Int] = []
        // BUTTER_QWEN_MTP_PMIN=<float>: confidence-gated dynamic draft
        // length. Unset/empty ⇒ feature OFF, behavior byte-identical to
        // the fixed-γ loop (every burst drafts exactly `gamma` tokens).
        // When set, `gamma` doubles as the burst's n_max: the burst
        // still stops at `gamma` tokens, but may stop EARLIER once the
        // drafter's own top-token probability drops below `pMin` — read
        // fresh every call, no static caching, matching the other
        // env-gated knobs above.
        let pMinEnv = ButterEnv["BUTTER_QWEN_MTP_PMIN"]
        let pMin: Float? = (pMinEnv?.isEmpty == false) ? Float(pMinEnv!) : nil
        // BUTTER_QWEN_MTP_ADAPTIVE=1: trailing-window adaptive draft depth,
        // gated on REALIZED trailing acceptance rather than drafter
        // self-confidence (BUTTER_QWEN_MTP_PMIN above was measured
        // uninformative on this head — confident everywhere; see the
        // BUTTER_QWEN_MTP_ACCEPT_CURVE=1 diagnosis: acceptance decays
        // via CONTENT DRIFT, oscillating 73->47% over 512 tokens, not
        // via a confidence signal). `cost` is now the Qwen3.8 default;
        // set the variable to `0` for the fixed-γ diagnostic path. Read
        // fresh every call, with no static caching.
        //
        // When set, `gamma` becomes the CEILING (`MTPAdaptiveGamma.cap`):
        // each cycle's effective draft length starts there and is
        // stepped by AT MOST 1 per cycle from the trailing acceptance
        // rate over the last BUTTER_QWEN_MTP_ADAPT_WINDOW per-candidate
        // accept/reject outcomes (default 32) — below
        // BUTTER_QWEN_MTP_ADAPT_LO (default 0.55) steps DOWN (floor 1),
        // above BUTTER_QWEN_MTP_ADAPT_HI (default 0.70) steps UP
        // (ceiling `gamma`); the gap between the two is the hysteresis
        // band that holds gamma steady rather than flapping every
        // cycle. Before the window has observed `window` candidates
        // (warmup), gamma stays at the cap. See `MTPAdaptiveGamma`'s
        // doc for the controller itself.
        //
        // BUTTER_QWEN_MTP_ADAPTIVE=cost selects the promoted Qwen3.8
        // per-position acceptance-EMA cost model instead. Its head/verify
        // cost ratio defaults to the measured 0.22 and can be re-fit with
        // BUTTER_QWEN_MTP_HEAD_COST_RATIO=<float>.
        //
        // Both controllers work identically in `.greedy` and `.sampled` mode: the
        // controller only consumes `drafts.count` /
        // `plan.committed.count - 1`, the same per-cycle accept
        // arithmetic `candidatesProposed`/`candidatesAccepted` already
        // tally below — no new verification or acceptance semantics, so
        // LOSSLESSNESS is untouched (gamma choice changes only how many
        // tokens are DRAFTED per cycle, never what gets accepted or
        // emitted; a greedy exact-match run is gamma-INVARIANT in its
        // output stream by construction).
        let adaptiveMode = ButterEnv["BUTTER_QWEN_MTP_ADAPTIVE"] ?? "cost"
        let adaptiveEnabled = adaptiveMode == "1"
        let adaptWindowEnv =
            ButterEnv["BUTTER_QWEN_MTP_ADAPT_WINDOW"]
        let adaptWindow =
            (adaptWindowEnv?.isEmpty == false) ? (Int(adaptWindowEnv!) ?? 32) : 32
        let adaptLoEnv = ButterEnv["BUTTER_QWEN_MTP_ADAPT_LO"]
        let adaptLo = (adaptLoEnv?.isEmpty == false) ? (Double(adaptLoEnv!) ?? 0.55) : 0.55
        let adaptHiEnv = ButterEnv["BUTTER_QWEN_MTP_ADAPT_HI"]
        let adaptHi = (adaptHiEnv?.isEmpty == false) ? (Double(adaptHiEnv!) ?? 0.70) : 0.70
        var adaptiveGamma: MTPAdaptiveGamma? =
            adaptiveEnabled
            ? MTPAdaptiveGamma(cap: gamma, window: adaptWindow, lo: adaptLo, hi: adaptHi)
            : nil
        let headCostEnv = ButterEnv["BUTTER_QWEN_MTP_HEAD_COST_RATIO"]
        let headCost =
            (headCostEnv?.isEmpty == false) ? (Double(headCostEnv!) ?? 0.22) : 0.22
        var costModelGamma: MTPCostModelGamma? =
            adaptiveMode == "cost"
            ? MTPCostModelGamma(cap: gamma, headStepCostRatio: headCost)
            : nil
        // BUTTER_QWEN_MTP_ACCEPT=ratio: correctness-oracle override for
        // `.sampled` mode, reproducing the original sampled-draft +
        // min(1,p/q) scheme (`planMTPAcceptSampled`) verbatim. Any other
        // value (default: unset) selects the "sample-target-then-match"
        // scheme (`planMTPAcceptMatch`) — greedy drafts, pre-sampled
        // target match/replace — which is the sampled path's default per
        // `SAMPLING_RESEARCH.md` #1. Read fresh every call, matching the
        // other env-gated knobs above; no effect in `.greedy` mode.
        let useRatioAccept =
            ButterEnv["BUTTER_QWEN_MTP_ACCEPT"] == "ratio"
        // avgdraft observability (BUTTER_QWEN_MTP_PROF=1 line below):
        // draftCycles counts bursts actually run (excludes the priming
        // iteration, which has no seed hidden and drafts nothing).
        var draftCycles = 0
        var minDraftLen = Int.max
        var maxDraftLen = 0
        // BUTTER_QWEN_MTP_ACCEPT_CURVE=1: positional acceptance
        // instrumentation (diagnostic only, T39 investigation into
        // acceptance decay over generation length). Buckets candidates
        // proposed/accepted by the OUTPUT position (tokensGenerated) at
        // the START of the cycle that proposed them — a cycle straddling
        // a bucket boundary is attributed whole to its start bucket
        // (gamma is small, so this never blurs the curve materially).
        // Printed as a single [mtp-accept] stderr line at the end, one
        // field per 64-token bucket.
        let acceptCurve =
            ButterEnv["BUTTER_QWEN_MTP_ACCEPT_CURVE"] == "1"
        let acceptCurveBucket = 64
        var bucketProposed: [Int] = []
        var bucketAccepted: [Int] = []

        // `statePlanes` also needs `specCapture` on: it gates
        // `forwardManyChunked`'s plane STASH onto the cache (the WRITE
        // itself is separately gated on the `BUTTER_GDN_STATE_PLANES` env
        // opt-out alone — see that method's doc), same as `stateRewind`
        // needing it for the qkv/a_raw/b_raw stash.
        if stateRewind || statePlanes { model.setSpecCapture(true, caches: caches) }
        defer { if stateRewind || statePlanes { model.setSpecCapture(false, caches: caches) } }
        loop: while tokensGenerated < maxNewTokens {
            // Pre-burst MTP cache length — the truncation target for
            // the post-accept lockstep (every burst row is rolled back
            // before the canonical re-advance). MUST be read before
            // the burst appends its rows.
            let mtpLen = head.kvCache.length
            // Output-position bucket for this cycle's candidates —
            // captured before `tokensGenerated` advances below (see
            // BUTTER_QWEN_MTP_ACCEPT_CURVE above).
            let curveBucketIdx = tokensGenerated / acceptCurveBucket

            // ── Draft burst (skipped on the priming iteration) ────────
            // Chain up to γ tokens through the single MTP layer. Each
            // step commits its own command buffer (the head's mixer
            // reuses per-instance scratch — see its threading contract)
            // and reads one argmax back to feed the chain.
            var drafts: [Int] = []
            // q_j's filtered distribution, one per draft step — only
            // populated in `.sampled` mode's RATIO-ORACLE branch
            // (BUTTER_QWEN_MTP_ACCEPT=ratio; ── mode: divergence 1).
            // Built ONCE here (to sample x_j from q_j) and reused
            // as-is by `planMTPAcceptSampled` at the accept step — see
            // its doc comment for why rebuilding it there used to
            // roughly double this path's CPU cost. Stays empty (unused)
            // under the default match scheme, which drafts greedily —
            // no q is ever built.
            var draftQDistributions: [Sampling.FilteredDistribution] = []
            if let seedHidden = prevHidden {
                var chainHidden = seedHidden
                var chainTok = prev
                // Effective burst length: `gamma` (the cap) unless
                // BUTTER_QWEN_MTP_ADAPTIVE=1, in which case this cycle's
                // controller-chosen value (see the env-var block above).
                // Recorded into `adaptiveGamma`'s avg/min/max stats as a
                // side effect — call exactly once per drafting cycle,
                // which this `if let seedHidden` branch already is (the
                // priming iteration never reaches here, so it never
                // pollutes the stats or the trailing window).
                let effectiveGamma =
                    costModelGamma?.nextCycleGamma()
                    ?? adaptiveGamma?.nextCycleGamma()
                    ?? gamma
                for j in 0 ..< effectiveGamma {
                    let dT = Date()
                    let dCmd = device.makeCommandBuffer()
                    let (h, dLogits) = head.draftStep(
                        prevHidden: chainHidden, prevToken: chainTok,
                        position: pos + j,
                        draftStepIndex: j, isGreedy: isGreedyRun,
                        on: dCmd, device: device)
                    dCmd.commit()
                    dCmd.waitUntilCompleted()
                    // Single CPU readback of this step's logits row,
                    // shared by both the argmax (always) and the
                    // confidence check (only when BUTTER_QWEN_MTP_PMIN
                    // is set) — no extra GPU work either way.
                    let flatD = dLogits.toFloatArray()
                    let c: Int
                    var pTop: Float = 1
                    // ── mode: divergence 1 — draft token choice.
                    // Greedy, and sampled's DEFAULT match scheme: BOTH
                    // argmax — identical computation, no q construction,
                    // no draft-side sampling at all
                    // (`SAMPLING_RESEARCH.md` #1: greedy drafts are what
                    // make the accept step's p(argmax q) beat the old
                    // ratio scheme's Σ min(p,q)). Sampled +
                    // BUTTER_QWEN_MTP_ACCEPT=ratio (correctness oracle
                    // only): draw from q_j (the SAME filter pipeline
                    // `Sampling.sample` uses on the non-spec path) and
                    // stash the raw row for the accept step, reproducing
                    // the pre-match scheme verbatim.
                    let draftsGreedily: Bool
                    switch mode {
                    case .greedy: draftsGreedily = true
                    case .sampled: draftsGreedily = !useRatioAccept
                    }
                    if draftsGreedily {
                        if pMin != nil {
                            let (idx, top) = argmaxAndTopProb(
                                flatD, offset: 0, count: flatD.count)
                            c = idx
                            pTop = top
                        } else {
                            c = argmax(flatD, offset: 0, count: flatD.count)
                        }
                    } else if case .sampled(let params) = mode {
                        // Ratio-oracle branch (BUTTER_QWEN_MTP_ACCEPT=ratio):
                        // build q_j's filtered distribution ONCE, sample
                        // x_j from it, and stash it for reuse both by
                        // the confidence gate below (probability of the
                        // token actually drawn) and by the accept step
                        // in `planMTPAcceptSampled` — avoids the double
                        // (pre-T43, triple under BUTTER_QWEN_MTP_PMIN)
                        // rebuild of the same row's distribution.
                        let qDist = Sampling.FilteredDistribution(
                            rawValues: flatD, parameters: params)
                        c = qDist.sample(&rng)
                        draftQDistributions.append(qDist)
                        if pMin != nil {
                            // Confidence gate reads the SAMPLED token's
                            // own q-probability (not argmax's), so a
                            // low-probability draw still triggers an
                            // early burst stop under BUTTER_QWEN_MTP_PMIN.
                            pTop = Float(qDist.probability(of: c))
                        }
                    } else {
                        // Unreachable: draftsGreedily is false only when
                        // mode is .sampled (see the switch above).
                        fatalError("MTPSpecDecode: unreachable draft-choice branch")
                    }
                    drafts.append(c)
                    chainHidden = h
                    chainTok = c
                    msDraft += Date().timeIntervalSince(dT)
                    // Confidence gate: stop the burst once the drafter's
                    // own top-token probability drops below pMin. Always
                    // draft at least 1 token first — the seed hidden's
                    // (pre-burst) confidence isn't available without an
                    // extra head pass, so the earliest possible stop is
                    // right after a draft, never before one.
                    //
                    // Floor is 2, not 1: `Ops.dequantGemmDynamicM`'s
                    // default-on small-M fast path (BUTTER_QMM_SMALL_M)
                    // routes verify batches of T=2 (drafts.count==1)
                    // through its BM=2 kernel, which has a pre-existing
                    // correctness bug — confirmed independent of this
                    // feature by reproducing the exact same corrupted
                    // digest via BUTTER_QWEN_MTP_GAMMA=1 (T=2, every
                    // cycle) on the unmodified tree, and confirming
                    // BUTTER_QMM_SMALL_M=0 fixes it. T=3/4 (BM=4) and
                    // T=8 verified clean. Rather than disabling the fast
                    // path for every pMin run (which would cost real
                    // throughput on the T≥3 cycles too, and stop this
                    // arm from being an apples-to-apples comparison
                    // against the fixed-γ baseline), the gate simply
                    // never stops at drafts.count == 1 — costing at most
                    // one extra (cheap, single-layer) draft step per
                    // early-stopped cycle while the underlying kernel
                    // bug gets its own fix.
                    if let pMin, drafts.count >= 2, pTop < pMin { break }
                }
                candidatesProposed += drafts.count
                draftCycles += 1
                minDraftLen = min(minDraftLen, drafts.count)
                maxDraftLen = max(maxDraftLen, drafts.count)
                // BUTTER_QWEN_MTP_TRACE=1: first few cycles only. Tells
                // apart a numerically broken head (constant/degenerate
                // draft ids) from a misaligned-but-live one (varied,
                // confident ids that simply do not match the verifier).
                if ButterEnv["BUTTER_QWEN_MTP_TRACE"] == "1",
                    verifySteps < 6
                {
                    FileHandle.standardError.write(
                        "[mtp-trace] cycle=\(verifySteps) prev=\(prev) "
                            .data(using: .utf8)!)
                    FileHandle.standardError.write(
                        "drafts=\(drafts)\n".data(using: .utf8)!)
                }
            }

            // ── Snapshot ALL caches before the speculative forward ────
            // Backbone layer caches via the composite snapshot; the MTP
            // head's own KV cache alongside (`mtpLen` above — same
            // length-only rollback semantics as the composite's KVCache
            // case). Taking the backbone snapshot AFTER the draft burst
            // is safe: the burst only touches the head's own cache.
            let snapT = Date()
            let snap = caches.snapshotAll(device: device)
            msSnap += Date().timeIntervalSince(snapT)

            // ── Verify: ONE batched backbone pass ─────────────────────
            let inputIds = [prev] + drafts
            let vT = Date()
            let vCmd = device.makeCommandBuffer()
            var (hiddenAll, logitsAll) = model.forwardManyHiddenAndLogits(
                tokenIds: inputIds, startPosition: pos,
                caches: caches, on: vCmd, device: device)
            vCmd.commit()
            vCmd.waitUntilCompleted()
            msVerify += Date().timeIntervalSince(vT)
            verifySteps += 1

            let flat = logitsAll.toFloatArray()
            let rows = inputIds.count
            let vocab = flat.count / rows
            precondition(
                flat.count == rows * vocab,
                "generateGreedyMTP: logits have \(flat.count) elements; "
                    + "expected rows*vocab = \(rows * vocab)")

            // ── Accept plan ── mode: divergence 2 ──────────────────────
            // Greedy: argmax per draft-aligned row (verifierTokens) +
            // the trailing bonus row, then longest-matching-prefix via
            // `planMTPAccept` — byte-identical to the original loop.
            // Sampled + default (match scheme): `planMTPAcceptMatch`
            // PRE-SAMPLES one token per row from each row's filtered
            // target distribution (row order, deterministic RNG
            // consumption) and feeds those into the SAME `planMTPAccept`
            // match/replace machinery greedy mode uses above — see its
            // doc comment. Sampled + BUTTER_QWEN_MTP_ACCEPT=ratio
            // (correctness oracle): `planMTPAcceptSampled` rebuilds p_j
            // from `flat` itself and makes the original
            // accept/reject/residual/bonus draws.
            let plan: MTPAcceptPlan
            switch mode {
            case .greedy:
                // Verifier tokens: argmax per draft-aligned row. Row j
                // verifies drafts[j] (the token at position pos + 1 + j);
                // the trailing row is the full-accept bonus.
                var verifierTokens: [Int] = []
                verifierTokens.reserveCapacity(drafts.count)
                for j in 0 ..< drafts.count {
                    verifierTokens.append(argmax(flat, offset: j * vocab, count: vocab))
                }
                let batchedBonus = argmax(
                    flat, offset: (rows - 1) * vocab, count: vocab)
                if ButterEnv["BUTTER_QWEN_MTP_TRACE"] == "1",
                    verifySteps <= 6
                {
                    FileHandle.standardError.write(
                        "[mtp-trace] verifier=\(verifierTokens) bonus=\(batchedBonus)\n"
                            .data(using: .utf8)!)
                }
                plan = planMTPAccept(
                    prev: prev, drafts: drafts,
                    verifierTokens: verifierTokens, batchedBonus: batchedBonus)
            case .sampled(let params):
                if useRatioAccept {
                    plan = planMTPAcceptSampled(
                        prev: prev, drafts: drafts, draftQDistributions: draftQDistributions,
                        verifyFlat: flat, rows: rows, vocab: vocab,
                        parameters: params, rng: &rng)
                } else {
                    plan = planMTPAcceptMatch(
                        prev: prev, drafts: drafts,
                        verifyFlat: flat, rows: rows, vocab: vocab,
                        parameters: params, rng: &rng)
                }
            }
            candidatesAccepted += plan.committed.count - 1
            // Feed the realized outcome back into the controller for
            // the NEXT cycle's gamma decision. No-ops on the priming
            // cycle (`drafts.count == 0`) — see `record`'s doc.
            adaptiveGamma?.record(
                accepted: plan.committed.count - 1, proposed: drafts.count)
            costModelGamma?.record(
                accepted: plan.committed.count - 1, proposed: drafts.count)
            if acceptCurve {
                while bucketProposed.count <= curveBucketIdx {
                    bucketProposed.append(0)
                    bucketAccepted.append(0)
                }
                bucketProposed[curveBucketIdx] += drafts.count
                bucketAccepted[curveBucketIdx] += plan.committed.count - 1
            }

            var nextTok = plan.nextToken
            var replayed = false
            // ── EXACTNESS DISCIPLINE (greedy only) ─────────────────────
            // Mirrors the fix `DFlashSpecDecode.swift` already made (see
            // that file's header, EXACTNESS DISCIPLINE section, point 5):
            // a full accept's batched verify-pass argmax can drift by
            // ULPs from what the single-token forward path would compute
            // at the SAME position, even though every drafted token's
            // IDENTITY already matched row-by-row. Trusting that drift
            // straight into the emitted bonus token and the cache
            // content on every full accept (the old `if plan.needsReplay`
            // gate below skipped this whole block whenever
            // `plan.needsReplay == false`) is exactly what
            // `mtpSpecMatchesBaselineExactly` caught: a reproducible
            // divergence at token index 16 under the shipped fixed-γ
            // config (γ ∈ {1, 2, 4}), present since before the PR #129
            // adversarial audit — see that test's git history for the
            // bisect. Forcing greedy-mode full accepts through the same
            // replay path partial accepts already use closes the gap:
            // `nextTok` and the cache both end up backed by the exact
            // single-token kernel, matching baseline greedy by
            // construction, same as DFlash.
            //
            // Sampled mode's full-accept fast path is UNCHANGED: its
            // bonus token already comes from a valid draw over the
            // batched pass's own probabilities (see
            // `planMTPAcceptSampled`/`planMTPAcceptMatch`'s doc
            // comments) — a losslessness contract that does not depend
            // on kernel-level bit-exactness the way greedy's
            // baseline-matching gate does, so sampled decode keeps full
            // accept's speed win.
            let mustReplayForExactness = mode.isGreedy && !plan.needsReplay
            if plan.needsReplay || mustReplayForExactness {
              // `autoreleasepool` wrap — this block now runs on nearly
              // EVERY greedy cycle instead of only partial/zero accepts
              // (the whole point of `mustReplayForExactness`), and each
              // pass through it allocates several MTLCommandBuffers /
              // Tensors (the snapshot restore, the per-token or batched
              // replay forward). Confirmed by direct repro during this
              // fix: at γ=1 (where nearly every cycle takes this path)
              // running unwrapped drove the process to a silent kill
              // partway through a 120-token run — undrained Metal
              // objects piling up across ~120 iterations in one
              // process, the same class of issue as the repo's
              // Metal-autoreleasepool-leak note (wrap any
              // `makeBuffer`-heavy loop). Scoped to this block rather
              // than the whole `loop: while` above because that outer
              // loop has `break loop` statements a closure can't target.
              autoreleasepool {
                // ── Partial/zero accept: restore + replay ─────────────
                // The batched pass's cache writes and logits are
                // discarded; the accepted prefix replays through the
                // single-token forward so the emitted stream (and the
                // cache contents backing it) stay bit-identical to the
                // baseline greedy loop. Next-prev comes from the LAST
                // single-step forward, not the batched row.
                let rT = Date()
                if statePlanes {
                    // ── Partial accept via plane select ────────────────
                    // Cheaper still than `stateRewind` below: the last
                    // verify pass's fused conv+scan kernels already wrote
                    // every token's state out to a plane (see
                    // `forwardManyChunked`'s per-token-planes block), so
                    // recovering the k-token state is a direct SELECT —
                    // `model.adoptPlanes` blits `plane[keep-1]` straight
                    // into the live cache buffers. No pre-verify snapshot
                    // restore (unlike `stateRewind`, this doesn't need
                    // `caches.restoreAll(..., recurrentOnly: true)` first
                    // — see `Qwen35GDNLayer.adoptPlaneState`'s doc), and
                    // no re-scan dispatch at all.
                    let wCmd = device.makeCommandBuffer()
                    let ok = model.adoptPlanes(
                        keep: plan.committed.count, absolutePosition: pos,
                        caches: caches, on: wCmd)
                    wCmd.commit()
                    wCmd.waitUntilCompleted()
                    if ok {
                        msReplay += Date().timeIntervalSince(rT)
                        // nextTok stays plan.nextToken; hiddenAll stays
                        // the verify pass's rows — same contract as
                        // `stateRewind` below.
                        planeSteps += 1
                        replayed = true
                    }
                    // else: planes weren't captured this cycle (flag was
                    // on but `t` exceeded the eligible cap, or this is
                    // the priming cycle) — fall through to `stateRewind`
                    // / full replay below. Caches are untouched so far
                    // (adoptPlanes only writes on success), so both
                    // fallbacks below still see the verify pass's
                    // (wrong, full-T) state exactly as they expect.
                }
                if !replayed, stateRewind {
                    // ── Partial accept WITHOUT replay ─────────────────
                    // The batched pass is causal, so everything it
                    // produced for accepted rows is already correct: the
                    // KV entries, the hidden states, and `plan.nextToken`
                    // (the verifier row one past the accepted prefix).
                    // The ONLY thing wrong is the recurrent state, which
                    // advanced over the rejected tokens too.
                    //
                    // So restore just the recurrent state, re-run the
                    // conv + chunk scan over the accepted prefix (no
                    // weight traffic), and shorten KV to match. This
                    // replaces a full forward — all 14 GB of weights —
                    // with two small dispatches per GDN layer.
                    caches.restoreAll(
                        from: snap, device: device, recurrentOnly: true)
                    let wCmd = device.makeCommandBuffer()
                    let ok = model.rewindCaches(
                        keep: plan.committed.count, absolutePosition: pos,
                        caches: caches, on: wCmd, device: device)
                    wCmd.commit()
                    wCmd.waitUntilCompleted()
                    if ok {
                        msReplay += Date().timeIntervalSince(rT)
                        // nextTok stays plan.nextToken; hiddenAll stays
                        // the verify pass's rows.
                        rewindSteps += 1
                        replayed = true
                    } else {
                        // Shape not eligible — fall back to replaying,
                        // which needs the FULL snapshot restored first.
                        caches.restoreAll(from: snap, device: device)
                    }
                }
                if !replayed {
                caches.restoreAll(from: snap, device: device)
                if batchedReplay {
                    // The rejected rows polluted every cache, so the
                    // accepted prefix has to be re-run from the restored
                    // snapshot either way. Doing it as ONE batched pass
                    // costs a single T=k+1 forward instead of k+1
                    // separate T=1 forwards, and the weights are loaded
                    // once rather than k+1 times.
                    //
                    // This gives up the bit-exactness discipline: the
                    // next token now comes from batched logits on EVERY
                    // path, not just full accepts. That is the same
                    // contract llama.cpp and vLLM use (they never replay
                    // at all). BUTTER_QWEN_MTP_ALWAYS_REPLAY=1 with this
                    // flag unset remains the bit-exact mode.
                    let rCmd = device.makeCommandBuffer()
                    let (rHidden, rLogits) = model.forwardManyHiddenAndLogits(
                        tokenIds: plan.committed, startPosition: pos,
                        caches: caches, on: rCmd, device: device)
                    rCmd.commit()
                    rCmd.waitUntilCompleted()
                    replaySingleSteps += 1
                    // ── mode: divergence 3 ─────────────────────────────
                    // Greedy: re-derive nextTok via argmax on the exact
                    // replay-kernel logits (the baseline-matching
                    // exactness discipline — see the file header).
                    // Sampled (both match and ratio schemes):
                    // `plan.nextToken` already came from the correct
                    // pre-sample/residual/bonus draw over the BATCHED
                    // pass's p; that's the algorithm's contract
                    // (lossless w.r.t. p regardless of which kernel
                    // computed it), so it must NOT be clobbered here.
                    // The replay above still ran for its cache-write
                    // side effect — the accepted prefix's KV entries
                    // must come from the restored snapshot, not the
                    // rejected-draft-polluted batched pass.
                    let rf = rLogits.toFloatArray()
                    let rv = rf.count / plan.committed.count
                    if mode.isGreedy {
                        nextTok = argmax(
                            rf, offset: (plan.committed.count - 1) * rv,
                            count: rv)
                    }
                    // Seed the next burst from THIS pass's hiddens: the
                    // verify pass's rows were computed before the
                    // restore and its trailing rows are backed by
                    // rejected tokens.
                    hiddenAll = rHidden
                } else {
                    var lastLogits: Tensor!
                    for (i, tok) in plan.committed.enumerated() {
                        let sCmd = device.makeCommandBuffer()
                        lastLogits = model.forward(
                            tokenId: tok, position: pos + i,
                            caches: caches, on: sCmd, device: device)
                        sCmd.commit()
                        sCmd.waitUntilCompleted()
                        replaySingleSteps += 1
                    }
                    // ── mode: divergence 3 (see above) ──────────────────
                    if mode.isGreedy {
                        nextTok = argmax(lastLogits)
                    }
                }
                msReplay += Date().timeIntervalSince(rT)
                }
              }
            }
            // Full accept, greedy: handled by the `mustReplayForExactness`
            // branch above — caches and `nextTok` both come from the
            // single-token replay, not the batched pass.
            // Full accept, sampled: backbone caches already hold the
            // verified block (all rows backed by accepted tokens);
            // `nextTok` is the batched bonus, per SpecDecode's existing
            // handling (see the `mustReplayForExactness` doc comment
            // above for why sampled mode keeps this fast path).

            // ── Parity probe (diagnostic only) ────────────────────────
            // BUTTER_QWEN_MTP_PARITY=1 recomputes the bonus through the
            // single-token path on FULL accepts and reports how far the
            // batched logits drift from it. Since the greedy full-accept
            // fix above (`mustReplayForExactness`) now already replays
            // through the single-token path for real, this block runs
            // AFTER the cache has been corrected — so on greedy runs it
            // measures single-token-vs-single-token (always zero drift)
            // rather than the pre-fix batched-vs-single-token gap. Left
            // in place (harmless — restores back to the already-correct
            // post-replay content either way) as a live check that the
            // fix actually closed the gap; the historical
            // batched-vs-single-token numbers this block used to report
            // are in the PR history for this fix. The emitted stream
            // still uses the fast path, so timings stay representative
            // and only the diagnostic cost is added.
            //
            // The margin is |top1 - top2| of the batched bonus row. A
            // mismatch at a tiny margin is an argmax tie-break between
            // near-equal logits; a mismatch at a wide margin means the
            // batched pass computed materially different logits, which
            // would also be silently costing acceptance on the verifier
            // rows.
            // Greedy-only diagnostic: "batched vs single-token argmax"
            // parity has no meaning once tokens are drawn probabilistically.
            //
            // SIDE-EFFECT HAZARD (found + fixed by the adversarial audit
            // on this branch): the block below truncates the backbone
            // caches to `snap` (pre-verify) and replays `plan.committed`
            // through the single-token forward — a kernel path that is
            // NUMERICALLY DIFFERENT from the batched verify pass (that
            // gap is the entire point of the measurement). That replay
            // writes real K/V into the SAME physical attention-cache
            // slots the batched pass already filled. `caches.restoreAll`
            // only undoes this for GDN/conv state (a true tensor-content
            // restore — see `CacheSnapshot.swift`'s `.gdnLayer`/`.gdn`
            // cases); its KVCache case is LENGTH-ONLY BY DESIGN (correct
            // for the reject/replay path elsewhere in this loop, where
            // the truncated tail gets reappended with identical content
            // — see the case's doc comment), so restoring `saved`
            // afterward is a no-op on KV bytes and silently leaves the
            // probe's single-token-computed K/V sitting where the
            // batched pass's K/V used to be. Every later decode step
            // attends over those slots, so the drift compounds forward
            // until it flips an argmax — confirmed on
            // `mtpSpecMatchesBaselineOnAdversarialPrompts` case 0
            // (γ=2, creative-highentropy): exact match with the flag
            // off, diverges at token 86/180 with it on. Fix: snapshot
            // the ACTUAL K/V bytes for the verified range up front via
            // `KVCache.snapshotRangeEncode` (content-level, not the
            // composite's length-only snapshot) and blit them back with
            // `restoreRangeEncode` after the probe, so the probe's own
            // replay never leaks into what the fast path reads next.
            if parityProbe, mode.isGreedy, !plan.needsReplay {
                // Recomputed locally (not read from the `.greedy` accept-plan
                // branch's own local) — this diagnostic block only runs in
                // greedy mode, where it equals the batched trailing row's
                // argmax, same value `planMTPAccept`'s caller derived.
                let batchedBonus = argmax(flat, offset: (rows - 1) * vocab, count: vocab)
                // Content-level KV snapshot over the verified range —
                // see the SIDE-EFFECT HAZARD note above. `pos` is the
                // batched verify pass's startPosition and `plan.committed
                // .count` is its row count, so `[pos, pos+count)` is
                // exactly the slot range that pass just filled.
                let rangeStart = pos
                let rangeCount = plan.committed.count
                var kvRangeSnaps: [(cache: KVCache, k: Tensor, v: Tensor)] = []
                let kvSnapCmd = device.makeCommandBuffer()
                for c in caches {
                    if let kv = c as? KVCache {
                        let s = kv.snapshotRangeEncode(
                            start: rangeStart, count: rangeCount,
                            on: kvSnapCmd, device: device)
                        kvRangeSnaps.append((kv, s.k, s.v))
                    }
                }
                kvSnapCmd.commit()
                kvSnapCmd.waitUntilCompleted()

                let saved = caches.snapshotAll(device: device)
                caches.restoreAll(from: snap, device: device)
                var truthLogits: Tensor!
                for (i, tok) in plan.committed.enumerated() {
                    let pCmd = device.makeCommandBuffer()
                    truthLogits = model.forward(
                        tokenId: tok, position: pos + i,
                        caches: caches, on: pCmd, device: device)
                    pCmd.commit()
                    pCmd.waitUntilCompleted()
                    // On a full accept `committed == inputIds`, so replay
                    // step i is the ground truth for batched row i. Track
                    // per-row agreement to tell a last-row boundary bug
                    // apart from the batched pass being wrong throughout.
                    let rowTruth = argmax(truthLogits)
                    let rowBatched = argmax(
                        flat, offset: i * vocab, count: vocab)
                    while parityRowChecked.count <= i {
                        parityRowChecked.append(0)
                        parityRowMismatch.append(0)
                    }
                    parityRowChecked[i] += 1
                    if rowTruth != rowBatched { parityRowMismatch[i] += 1 }
                }
                let truth = argmax(truthLogits)
                parityChecked += 1
                let base = (rows - 1) * vocab
                var t1 = -Float.infinity
                var t2 = -Float.infinity
                for v in 0 ..< vocab {
                    let x = flat[base + v]
                    if x > t1 {
                        t2 = t1
                        t1 = x
                    } else if x > t2 {
                        t2 = x
                    }
                }
                let margin = t1 - t2
                if truth != batchedBonus {
                    parityMismatch += 1
                    parityMismatchMargin += Double(margin)
                    let msg =
                        "[mtp-parity] MISMATCH batched=\(batchedBonus) "
                        + "single=\(truth) margin=\(margin)\n"
                    FileHandle.standardError.write(msg.data(using: .utf8)!)
                } else {
                    parityMatchMargin += Double(margin)
                }
                caches.restoreAll(from: saved, device: device)
                // Undo the probe's KV pollution: blit the batched pass's
                // ACTUAL K/V bytes back over the range the replay just
                // overwrote (composite restoreAll above only fixed
                // length + GDN/conv content, not this).
                let kvRestoreCmd = device.makeCommandBuffer()
                for entry in kvRangeSnaps {
                    entry.cache.restoreRangeEncode(
                        from: (k: entry.k, v: entry.v),
                        start: rangeStart, count: rangeCount, on: kvRestoreCmd)
                }
                kvRestoreCmd.commit()
                kvRestoreCmd.waitUntilCompleted()
            }

            // ── Commit + emit (stop tokens and budget respected) ──────
            var stopped = false
            var emitted = 0
            for tok in plan.committed {
                if stopTokens.contains(tok) {
                    stopped = true
                    break
                }
                history.append(tok)
                emit?(tok)
                emitted += 1
                tokensGenerated += 1
                if tokensGenerated >= maxNewTokens { break }
            }
            if stopped || emitted < plan.committed.count { break loop }

            // ── MTP KV lockstep over the committed block ──────────────
            // Roll back the burst's draft rows (steps ≥ 1 were fed the
            // head's own h' chain, not backbone hiddens), then advance
            // the head once per committed token with the verify pass's
            // backbone hidden at the previous position — the canonical
            // lockstep feed. Logits are discarded; no sampling.
            head.kvCache.truncate(toLength: mtpLen)
            if batchedLockstep {
                // ── Batched re-advance — ONE T-batched head pass ──────
                // Only row 0 can be missing a hidden (the priming
                // iteration, where `prevHidden` is nil for `committed[0]
                // == prev`); when that happens the row is DROPPED from
                // the batch (never zero-filled — see the task note in
                // Qwen3xMTPHead.draftStepMany) and the remaining suffix,
                // which is always contiguous, is batched instead. Every
                // post-priming iteration has `prevHidden != nil`, so
                // `startIdx` is 0 from the second iteration on.
                let startIdx = (prevHidden == nil) ? 1 : 0
                if startIdx < plan.committed.count {
                    let rowsT = plan.committed.count - startIdx
                    let lT = Date()
                    // Assemble the `[rowsT, hidden]` hidden batch: row 0
                    // (when startIdx == 0) is `prevHidden` — a tensor
                    // from a DIFFERENT buffer than `hiddenAll` (last
                    // iteration's carry) — every other row is a
                    // `hiddenAll` view. Blit both sources into one
                    // contiguous buffer so the batched pass reads a
                    // single tensor.
                    let hiddenBytes = model.hidden * hiddenAll.dtype.byteSize
                    let hiddensFlat = Tensor.empty(
                        shape: [rowsT, model.hidden], dtype: hiddenAll.dtype,
                        device: device)
                    let lCmd = device.makeCommandBuffer()
                    for row in 0 ..< rowsT {
                        let j = startIdx + row
                        let src =
                            (j == 0)
                            ? prevHidden!
                            : hiddenRow(hiddenAll, j - 1, hidden: model.hidden)
                        let dst = Tensor(
                            buffer: hiddensFlat.buffer,
                            offset: hiddensFlat.offset + row * hiddenBytes,
                            shape: [model.hidden], dtype: hiddenAll.dtype)
                        Ops.copy(src, into: dst, on: lCmd)
                    }
                    let tokens = Array(plan.committed[startIdx...])
                    head.draftStepMany(
                        prevHiddensFlat: hiddensFlat.reshaped(
                            to: [rowsT * model.hidden]),
                        prevTokens: tokens, startPosition: pos + startIdx,
                        on: lCmd, device: device)
                    lCmd.commit()
                    // The next head pass is submitted to the same primary
                    // queue, which preserves the KV update dependency.
                    msLockstep += Date().timeIntervalSince(lT)
                    lockstepSteps += 1
                }
            } else {
                for (j, tok) in plan.committed.enumerated() {
                    // committed[0] == prev at position pos pairs with the
                    // hidden at pos - 1 (last iteration's carry); committed
                    // [j ≥ 1] pairs with the verify hidden at pos + j - 1.
                    guard let ph = (j == 0) ? prevHidden : hiddenRow(hiddenAll, j - 1, hidden: model.hidden)
                    else {
                        // Priming iteration only: no hidden exists for
                        // `prev`'s position — the head's history skips this
                        // one row (see the file header).
                        continue
                    }
                    let lT = Date()
                    let lCmd = device.makeCommandBuffer()
                    // Not a speculative burst — this re-advances the
                    // head's KV cache over already-COMMITTED tokens and
                    // discards the logits (see the surrounding comment),
                    // so which lm_head computes them is irrelevant to
                    // correctness. Pass draftStepIndex 0 (always maps to
                    // the 4-bit head under `.depth2Plus`, the cheapest
                    // safe default) rather than threading `j` here.
                    _ = head.draftStep(
                        prevHidden: ph, prevToken: tok, position: pos + j,
                        draftStepIndex: 0, isGreedy: isGreedyRun,
                        on: lCmd, device: device)
                    lCmd.commit()
                    lCmd.waitUntilCompleted()
                    msLockstep += Date().timeIntervalSince(lT)
                    lockstepSteps += 1
                }
            }
            // Diagnostic: head KV cache length vs. the backbone position
            // it should be tracking (offset by a constant -1 from the
            // documented priming gap — see the file header's PRIMING
            // note). Printed every 32 verify steps under
            // BUTTER_QWEN_MTP_ACCEPT_CURVE=1 to rule out a slow leak
            // (offset growing instead of staying pinned at 1).
            if acceptCurve, verifySteps % 32 == 0 {
                let kvLen = head.kvCache.length
                let posAfter = pos + plan.committed.count
                FileHandle.standardError.write(
                    "[mtp-kvcheck] verifystep=\(verifySteps) posAfter=\(posAfter) "
                        .data(using: .utf8)!)
                FileHandle.standardError.write(
                    "kvlen=\(kvLen) offset=\(posAfter - kvLen)\n".data(using: .utf8)!)
            }

            // ── Advance ───────────────────────────────────────────────
            // Next burst seeds from the verify hidden at the last
            // committed position. The verify pass blits `hiddenAll`
            // into a fresh tensor on its own command buffer, so the
            // row view stays valid across later forwards.
            prevHidden = hiddenRow(hiddenAll, plan.nextHiddenRow, hidden: model.hidden)
            pos += plan.committed.count
            prev = nextTok
            if stopTokens.contains(nextTok) { break loop }
        }
        // Like `generateGreedy`: `prev` is next-iteration input (its KV
        // slot may not be committed); it is NOT appended to `history`.

        if ButterEnv["BUTTER_QWEN_MTP_PROF"] == "1" {
            let tot = Date().timeIntervalSince(t0)
            let avgDraft =
                draftCycles > 0
                ? Double(candidatesProposed) / Double(draftCycles) : 0.0
            let minD = draftCycles > 0 ? minDraftLen : 0
            // Acceptance stats: accepted-per-cycle mirrors avgdraft
            // (average, not per-cycle detail — this loop doesn't retain
            // a per-cycle history array) and acceptance-rate is
            // candidatesAccepted/candidatesProposed, same definition as
            // `MTPSpecDecodeStats.acceptanceRate`. Reported on both
            // modes; most useful for `.sampled` where acceptance isn't
            // "1.0 unless the drafter is wrong" the way greedy's is.
            let acceptedPerCycle =
                draftCycles > 0
                ? Double(candidatesAccepted) / Double(draftCycles) : 0.0
            let acceptRate =
                candidatesProposed > 0
                ? 100.0 * Double(candidatesAccepted) / Double(candidatesProposed) : 0.0
            // BUTTER_QWEN_MTP_ADAPTIVE=1 effective-gamma stats — always
            // printed as `adaptive=on/off` so log-scraping tooling has a
            // stable field either way; avg/min/max only populated `on`
            // (mirrors `adaptiveGamma`'s own optionality).
            let adaptiveSuffix: String
            if let ag = costModelGamma {
                let mn = ag.gammaMin == Int.max ? 0 : ag.gammaMin
                adaptiveSuffix =
                    " adaptive=cost avggamma=\(String(format: "%.2f", ag.avgGamma)) "
                    + "mingamma=\(mn) maxgamma=\(ag.gammaMax)"
            } else if let ag = adaptiveGamma {
                let mn = ag.gammaMin == Int.max ? 0 : ag.gammaMin
                adaptiveSuffix =
                    " adaptive=on avggamma=\(String(format: "%.2f", ag.avgGamma)) "
                    + "mingamma=\(mn) maxgamma=\(ag.gammaMax)"
            } else {
                adaptiveSuffix = " adaptive=off"
            }
            let line = String(
                format:
                    "[mtp-prof] total=%.2fs draft=%.2f verify=%.2f replay=%.2f "
                    + "lockstep=%.2f snap=%.2f other=%.2f rewinds=\(rewindSteps) "
                    + "planes=\(planeSteps) "
                    + "avgdraft=\(String(format: "%.2f", avgDraft)) "
                    + "mindraft=\(minD) maxdraft=\(maxDraftLen) draftcycles=\(draftCycles) "
                    + "accepted=\(candidatesAccepted)/\(candidatesProposed) "
                    + "acceptrate=\(String(format: "%.1f", acceptRate))%% "
                    + "acceptedpercycle=\(String(format: "%.2f", acceptedPerCycle))"
                    + "\(adaptiveSuffix)\n",
                tot, msDraft, msVerify, msReplay, msLockstep, msSnap,
                tot - msDraft - msVerify - msReplay - msLockstep - msSnap)
            FileHandle.standardError.write(line.data(using: .utf8)!)
        }
        if parityProbe, parityChecked > 0 {
            let matched = parityChecked - parityMismatch
            let line = String(
                format:
                    "[mtp-parity] full-accepts=%d mismatch=%d (%.1f%%) "
                    + "avg-margin match=%.4f mismatch=%.4f\n",
                parityChecked, parityMismatch,
                100.0 * Double(parityMismatch) / Double(parityChecked),
                matched > 0 ? parityMatchMargin / Double(matched) : 0.0,
                parityMismatch > 0
                    ? parityMismatchMargin / Double(parityMismatch) : 0.0)
            FileHandle.standardError.write(line.data(using: .utf8)!)
            var rows = "[mtp-parity] per-row mismatch:"
            for i in 0 ..< parityRowChecked.count {
                rows += " row\(i)=\(parityRowMismatch[i])/\(parityRowChecked[i])"
            }
            FileHandle.standardError.write((rows + "\n").data(using: .utf8)!)
        }
        if acceptCurve, !bucketProposed.isEmpty {
            var line = "[mtp-accept]"
            for i in 0 ..< bucketProposed.count {
                let lo = i * acceptCurveBucket
                let hi = lo + acceptCurveBucket - 1
                let p = bucketProposed[i]
                let a = bucketAccepted[i]
                let rate = p > 0 ? 100.0 * Double(a) / Double(p) : 0.0
                line += " b\(lo)-\(hi)=\(a)/\(p)(\(String(format: "%.1f", rate))%)"
            }
            FileHandle.standardError.write((line + "\n").data(using: .utf8)!)
        }
        return MTPSpecDecodeStats(
            tokensGenerated: tokensGenerated,
            verifySteps: verifySteps,
            candidatesProposed: candidatesProposed,
            candidatesAccepted: candidatesAccepted,
            replaySingleSteps: replaySingleSteps,
            lockstepSteps: lockstepSteps,
            wallclockSeconds: Date().timeIntervalSince(t0))
    }
}

// ─── helpers ─────────────────────────────────────────────────────────

/// `[hidden]` view of one row of a `[T, hidden]` hidden-state tensor.
@inline(__always)
private func hiddenRow(_ hiddenAll: Tensor, _ row: Int, hidden: Int) -> Tensor {
    Tensor(
        buffer: hiddenAll.buffer,
        offset: hiddenAll.offset + row * hidden * hiddenAll.dtype.byteSize,
        shape: [hidden], dtype: hiddenAll.dtype)
}

@inline(__always)
private func argmax(_ logits: Tensor) -> Int {
    let host = logits.toFloatArray()
    return argmax(host, offset: 0, count: host.count)
}

@inline(__always)
private func argmax(_ flat: [Float], offset: Int, count: Int) -> Int {
    precondition(count > 0)
    var bestIdx = 0
    var bestVal = flat[offset]
    for i in 1 ..< count {
        let v = flat[offset + i]
        if v > bestVal {
            bestVal = v
            bestIdx = i
        }
    }
    return bestIdx
}

/// Argmax + exact softmax top-1 probability over one logits row, in the
/// same host readback (BUTTER_QWEN_MTP_PMIN's confidence gate). Two
/// passes over `count` floats: the first finds `bestIdx`/`bestVal`
/// (identical to `argmax` above), the second computes the numerically
/// stable normalizer `s = Σ exp(l_i - bestVal)`. Since
/// `exp(bestVal - bestVal) == 1`, the top probability is `1 / s` —
/// no separate `exp(l_max - logsumexp(l))` term needed.
@inline(__always)
private func argmaxAndTopProb(_ flat: [Float], offset: Int, count: Int) -> (
    Int, Float
) {
    precondition(count > 0)
    var bestIdx = 0
    var bestVal = flat[offset]
    for i in 1 ..< count {
        let v = flat[offset + i]
        if v > bestVal {
            bestVal = v
            bestIdx = i
        }
    }
    var s: Float = 0
    for i in 0 ..< count {
        s += Foundation.exp(flat[offset + i] - bestVal)
    }
    return (bestIdx, 1 / s)
}
