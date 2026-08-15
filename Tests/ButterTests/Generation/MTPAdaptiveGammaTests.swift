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
// MTPAdaptiveGammaTests — pure-CPU coverage of the MTP loop's adaptive
// draft-depth controller (`SpecDecode.MTPAdaptiveGamma`,
// BUTTER_QWEN_MTP_ADAPTIVE=1). Window math, hysteresis, and warmup are
// exercised directly against the controller, no model/tensor
// dependency — same spirit as `MTPSpecDecodeTests`'s coverage of
// `planMTPAccept`. GPU/model-level engagement of the feature is
// verified separately by the gamma-invariance digest gate (a greedy
// exact-match run's emitted stream doesn't depend on how many
// candidates were drafted per cycle, so BUTTER_QWEN_MTP_ADAPTIVE=1 must
// reproduce the fixed-gamma digest bit-for-bit — see the commit that
// introduced this file for the measured result).

import Testing

@testable import ButterSwift

@Suite("MTPAdaptiveGamma — controller")
struct MTPAdaptiveGammaTests {

    // ─── Warmup ─────────────────────────────────────────────────────

    @Test("stays at cap until the window has observed a full `window` candidates")
    func warmupHoldsAtCap() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 4, lo: 0.5, hi: 0.75)
        #expect(g.gamma == 3)
        // 2 rejects, then 1 more reject: 3 of 4 window slots filled,
        // still warming up even though the trailing rate (0/3) is
        // already well below `lo`.
        g.record(accepted: 0, proposed: 2)
        #expect(g.gamma == 3)
        g.record(accepted: 0, proposed: 1)
        #expect(g.gamma == 3)
    }

    @Test("adjusts the instant the window reaches exactly `window` candidates")
    func warmupBoundaryTriggersFirstAdjustment() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 4, lo: 0.5, hi: 0.75)
        g.record(accepted: 0, proposed: 3)
        #expect(g.gamma == 3)  // 3 of 4 -- still warmup
        g.record(accepted: 0, proposed: 1)  // window now exactly full: 4 candidates, rate 0
        #expect(g.gamma == 2)  // rate 0 < lo(0.5) -> step down
    }

    @Test("a priming/draft-free cycle (proposed == 0) is a no-op: doesn't fill the window or touch gamma")
    func primingCycleIsNoOp() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 2, lo: 0.5, hi: 0.75)
        g.record(accepted: 0, proposed: 0)
        g.record(accepted: 0, proposed: 0)
        g.record(accepted: 0, proposed: 0)
        #expect(g.gamma == 3)  // never left warmup -- window never filled
        // Confirm the window really is untouched: 2 real rejects now
        // should fill a window of size 2 and adjust, exactly as if the
        // priming no-ops never happened.
        g.record(accepted: 0, proposed: 2)
        #expect(g.gamma == 2)
    }

    // ─── Hysteresis: step down / hold / step up ────────────────────

    @Test("steps DOWN by exactly 1 when trailing acceptance falls below lo")
    func stepsDownBelowLo() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 4, lo: 0.5, hi: 0.75)
        g.record(accepted: 0, proposed: 4)  // rate 0/4 = 0 < 0.5
        #expect(g.gamma == 2)
    }

    @Test("steps UP by exactly 1 when trailing acceptance rises above hi")
    func stepsUpAboveHi() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 4, lo: 0.5, hi: 0.75)
        // Start below cap so the step up is observable (a cap-start
        // controller can only go down or hold at the ceiling).
        g.record(accepted: 0, proposed: 4)  // -> gamma 2
        #expect(g.gamma == 2)
        g.record(accepted: 4, proposed: 4)  // ring now all-true, rate 1.0 > 0.75
        #expect(g.gamma == 3)
    }

    @Test("holds gamma steady when trailing acceptance lands inside [lo, hi] (the hysteresis band)")
    func holdsInsideHysteresisBand() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 4, lo: 0.5, hi: 0.75)
        g.record(accepted: 0, proposed: 4)  // -> gamma 2 (rate 0)
        #expect(g.gamma == 2)
        // Ring is window-sized (4): pushing exactly 4 new outcomes
        // fully replaces it, landing the rate exactly at lo (0.5) --
        // NOT strictly below it, so this holds rather than stepping
        // down again.
        g.record(accepted: 2, proposed: 4)
        #expect(g.gamma == 2)
        // Same trick at exactly hi (0.75): holds, doesn't step up.
        g.record(accepted: 3, proposed: 4)
        #expect(g.gamma == 2)
    }

    @Test("boundary values lo and hi are both inclusive of the hold zone (strict < / > only)")
    func boundariesAreInclusiveHold() {
        // window=1 isolates a single record call's rate exactly at the
        // boundary with no averaging from prior pushes.
        var atLo = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 2, lo: 0.5, hi: 0.75)
        atLo.record(accepted: 1, proposed: 2)  // rate exactly 0.5 == lo
        #expect(atLo.gamma == 3)  // holds at cap, does not step down

        var atHi = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 4, lo: 0.5, hi: 0.75)
        atHi.record(accepted: 0, proposed: 4)  // -> gamma 2
        atHi.record(accepted: 3, proposed: 4)  // rate exactly 0.75 == hi
        #expect(atHi.gamma == 2)  // holds, does not step up
    }

    // ─── Floor / ceiling clamps ─────────────────────────────────────

    @Test("never steps below 1, even with sustained zero acceptance")
    func floorsAtOne() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 5, window: 1, lo: 0.5, hi: 0.75)
        #expect(g.gamma == 5)
        g.record(accepted: 0, proposed: 1)
        #expect(g.gamma == 4)
        g.record(accepted: 0, proposed: 1)
        #expect(g.gamma == 3)
        g.record(accepted: 0, proposed: 1)
        #expect(g.gamma == 2)
        g.record(accepted: 0, proposed: 1)
        #expect(g.gamma == 1)
        // Already at the floor: further zero-acceptance cycles hold,
        // never go negative or to 0.
        g.record(accepted: 0, proposed: 1)
        #expect(g.gamma == 1)
        g.record(accepted: 0, proposed: 1)
        #expect(g.gamma == 1)
    }

    @Test("never steps above the configured cap, even with sustained full acceptance")
    func ceilingsAtCap() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 1, lo: 0.5, hi: 0.75)
        #expect(g.gamma == 3)  // starts at cap
        g.record(accepted: 1, proposed: 1)  // rate 1.0 > hi, but already at cap
        #expect(g.gamma == 3)
        g.record(accepted: 1, proposed: 1)
        #expect(g.gamma == 3)
    }

    @Test("adjusts by at most 1 per cycle -- an extreme rate never skips a step")
    func adjustsAtMostOnePerCycle() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 5, window: 4, lo: 0.5, hi: 0.75)
        // Fill the window with a single call, rate 0 -- as extreme a
        // signal as possible -- and confirm gamma moves by exactly 1,
        // not straight to the floor.
        g.record(accepted: 0, proposed: 4)
        #expect(g.gamma == 4)
    }

    // ─── Sliding window: old outcomes actually get evicted ─────────

    @Test("trailing rate is over the last `window` candidates only -- older outcomes are evicted")
    func slidingWindowEvictsOldOutcomes() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 4, lo: 0.5, hi: 0.75)
        // Fill the window with all-accepted (rate 1.0 > hi already at
        // cap, so this just confirms the window state, not gamma).
        g.record(accepted: 4, proposed: 4)
        #expect(g.gamma == 3)
        // Push 4 more, all-rejected: this should COMPLETELY replace the
        // old all-true window (window == proposed count), landing the
        // trailing rate at 0, not some blend with the evicted history.
        g.record(accepted: 0, proposed: 4)
        #expect(g.gamma == 2)
    }

    @Test("partial window replacement blends correctly (older survivors still count)")
    func partialWindowReplacementBlends() {
        // window=4: seed [T,T,T,T] (rate 1.0), then push 2 more
        // rejects -- ring becomes [F,F,T,T] (FIFO eviction of the 2
        // oldest), rate 2/4 = 0.5 == lo -> holds, does not step down.
        var g = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 4, lo: 0.5, hi: 0.75)
        g.record(accepted: 4, proposed: 4)
        #expect(g.gamma == 3)
        g.record(accepted: 0, proposed: 2)
        #expect(g.gamma == 3)  // rate exactly 0.5 -- hold, not a step down
        // One more reject: ring becomes [F,T,T,F] pushed as [F] over
        // the oldest T -- FIFO order is [F,F,T,T] then push F evicts
        // the front F -> [F,T,T,F]... rather than hand-trace FIFO
        // order (which doesn't matter for a rate-only decision), just
        // confirm the rate crosses below lo once true count drops to
        // 1/4 = 0.25.
        g.record(accepted: 0, proposed: 1)
        #expect(g.gamma == 2)
    }

    // ─── nextCycleGamma() stats bookkeeping ─────────────────────────

    @Test("nextCycleGamma reports the CURRENT gamma and accumulates avg/min/max")
    func nextCycleGammaStats() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 3, window: 4, lo: 0.5, hi: 0.75)
        #expect(g.nextCycleGamma() == 3)  // cycle 1: cap, pre-warmup
        g.record(accepted: 0, proposed: 4)  // -> gamma 2
        #expect(g.nextCycleGamma() == 2)  // cycle 2: reflects the step down
        #expect(g.nextCycleGamma() == 2)  // cycle 3: no record() in between, holds
        #expect(g.avgGamma == (3.0 + 2.0 + 2.0) / 3.0)
        #expect(g.gammaMin == 2)
        #expect(g.gammaMax == 3)
        #expect(g.gammaCycles == 3)
    }

    @Test("avgGamma is 0 when nextCycleGamma was never called (no cycles observed)")
    func avgGammaZeroWithNoCycles() {
        let g = SpecDecode.MTPAdaptiveGamma(cap: 4, window: 8, lo: 0.5, hi: 0.75)
        #expect(g.avgGamma == 0)
        #expect(g.gammaCycles == 0)
    }

    // ─── End-to-end: warmup -> down -> hold -> up, gamma bounded [1, cap] throughout ────

    @Test("full lifecycle stays within [1, cap] across a mixed sequence of outcomes")
    func fullLifecycleStaysInBounds() {
        var g = SpecDecode.MTPAdaptiveGamma(cap: 4, window: 3, lo: 0.55, hi: 0.70)
        let cycles: [(accepted: Int, proposed: Int)] = [
            (0, 3), (0, 3), (1, 3), (0, 3), (3, 3), (3, 3), (3, 3),
            (0, 1), (0, 1), (0, 1), (3, 3), (3, 3), (3, 3),
        ]
        for c in cycles {
            _ = g.nextCycleGamma()
            g.record(accepted: c.accepted, proposed: c.proposed)
            #expect(g.gamma >= 1 && g.gamma <= 4)
        }
    }
}
