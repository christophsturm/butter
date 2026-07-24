// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// WindowPlanTests — pure-logic coverage for the PPL/KLD context-windowing
// plan (design §6.1) and the KLD distribution summary (§6.4). No model /
// GPU required: the plan is integer bookkeeping and the summary is array
// math, so these run in the fast unit suite.

import Foundation
import Testing

@testable import ButterSwift

@Suite("PPL/KLD windowing + distribution")
struct WindowPlanTests {

    /// Reconstruct the set of absolute target tokens a plan scores. A
    /// window forces local positions `0 ..< length-1`; positions
    /// `>= scoredLocalStart` are scored, with target `start + t + 1`.
    private func scoredTargets(_ plan: [Perplexity.Window]) -> [Int] {
        var targets: [Int] = []
        for w in plan {
            for t in 0 ..< (w.length - 1) where t >= w.scoredLocalStart {
                targets.append(w.start + t + 1)
            }
        }
        return targets
    }

    @Test("window=0 collapses to a single full-corpus pass")
    func legacySinglePass() {
        let n = 50
        let plan = Perplexity.WindowPlan.plan(tokenCount: n, contextWindow: 0)
        #expect(plan == [Perplexity.Window(start: 0, length: n, scoredLocalStart: 0)])
        #expect(scoredTargets(plan) == Array(1 ..< n))
    }

    @Test("window >= corpus collapses to a single pass")
    func windowLargerThanCorpus() {
        let n = 20
        let plan = Perplexity.WindowPlan.plan(tokenCount: n, contextWindow: 1000)
        #expect(plan.count == 1)
        #expect(scoredTargets(plan) == Array(1 ..< n))
    }

    @Test("strided windows score every corpus token exactly once")
    func stridedCoverageExactlyOnce() {
        // A spread of (n, ctx) pairs, including non-multiples so the tail
        // window is exercised.
        for (n, ctx) in [(100, 40), (101, 40), (256, 64), (200, 30), (77, 16)] {
            let plan = Perplexity.WindowPlan.plan(tokenCount: n, contextWindow: ctx)
            let targets = scoredTargets(plan)
            // Each corpus token in [1, n) scored exactly once — no gaps,
            // no double-counting.
            #expect(targets.sorted() == Array(1 ..< n), "coverage broke for n=\(n) ctx=\(ctx)")
            #expect(Set(targets).count == targets.count, "double-count for n=\(n) ctx=\(ctx)")
        }
    }

    @Test("every scored token carries >= window/2 left-context")
    func scoredTokensHaveContext() {
        let n = 256
        let ctx = 64
        let plan = Perplexity.WindowPlan.plan(tokenCount: n, contextWindow: ctx)
        for w in plan where w.start > 0 {
            // Non-first windows score only the second half, so the first
            // scored local position has >= stride-1 tokens of context.
            #expect(w.scoredLocalStart >= ctx / 2 - 1)
        }
    }

    @Test("KLD distribution percentiles + top-1 agreement")
    func distributionSummary() {
        // samples 0..9; nearest-rank over 10 elements:
        //   median q=.5 → rank round(.5*9)=round(4.5)=5 (half-away) → 5
        //   p90    q=.9 → rank round(.9*9)=round(8.1)=8 → 8
        //   p99    q=.99→ rank round(.99*9)=round(8.91)=9 → 9
        let samples = (0 ..< 10).map { Double($0) }
        let d = Perplexity.KLDistribution.from(klSamples: samples, agreements: 7, scored: 10)
        #expect(d.mean == 4.5)
        #expect(d.median == 5)
        #expect(d.p90 == 8)
        #expect(d.p99 == 9)
        #expect(d.max == 9)
        #expect(abs(d.top1Agreement - 0.7) < 1e-9)
    }

    @Test("empty distribution is well-defined")
    func emptyDistribution() {
        let d = Perplexity.KLDistribution.from(klSamples: [], agreements: 0, scored: 0)
        #expect(d.mean == 0)
        #expect(d.max == 0)
        #expect(d.top1Agreement == 0)
    }
}
