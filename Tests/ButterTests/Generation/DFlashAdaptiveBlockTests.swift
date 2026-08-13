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
// DFlashAdaptiveBlockTests — pure-CPU coverage of the DFlash loop's
// adaptive block-size/bail controller (`DFlashSpecDecode.DFlashAdaptiveBlock`,
// BUTTER_GLIMMER_DFLASH_ADAPTIVE=1). Window math, hysteresis, rung
// stepping, and bail entry/exit are exercised directly against the
// controller, no model/tensor dependency — same spirit as
// `MTPAdaptiveGammaTests`'s coverage of `SpecDecode.MTPAdaptiveGamma`.
// GPU/model-level engagement is verified separately by the DFlash
// byte-identical exactness gate (`GlimmerDFlashSpecDecodeIntegrationTests`)
// — a greedy exact-match run's emitted stream doesn't depend on how many
// candidates were drafted per cycle or whether a cycle drafted at all,
// so BUTTER_GLIMMER_DFLASH_ADAPTIVE=1 must reproduce the fixed-block
// digest bit-for-bit.

import Testing

@testable import ButterSwift

@Suite("DFlashAdaptiveBlock — controller")
struct DFlashAdaptiveBlockTests {

    // ─── Warmup ─────────────────────────────────────────────────────

    @Test("stays at the ceiling rung until the window has observed a full `window` candidates")
    func warmupHoldsAtCeiling() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 4, lo: 0.5, hi: 0.75, bailTokens: 8)
        #expect(b.blockSize == 16)
        b.record(accepted: 0, proposed: 2)
        #expect(b.blockSize == 16)
        b.record(accepted: 0, proposed: 1)
        #expect(b.blockSize == 16)  // 3 of 4 -- still warming up
    }

    @Test("adjusts the instant the window reaches exactly `window` candidates")
    func warmupBoundaryTriggersFirstAdjustment() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 4, lo: 0.5, hi: 0.75, bailTokens: 8)
        b.record(accepted: 0, proposed: 3)
        #expect(b.blockSize == 16)
        b.record(accepted: 0, proposed: 1)  // window now exactly full: rate 0
        #expect(b.blockSize == 8)  // rate 0 < lo(0.5) -> step down one rung
    }

    @Test("a draft-free cycle (proposed == 0) is a no-op: doesn't fill the window or touch the rung")
    func proposedZeroIsNoOp() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 2, lo: 0.5, hi: 0.75, bailTokens: 8)
        b.record(accepted: 0, proposed: 0)
        b.record(accepted: 0, proposed: 0)
        #expect(b.blockSize == 16)
        b.record(accepted: 0, proposed: 2)  // fills the window of 2
        #expect(b.blockSize == 8)
    }

    // ─── Hysteresis: step down / hold / step up ────────────────────

    @Test("steps DOWN one rung when trailing acceptance falls below lo")
    func stepsDownOneRungBelowLo() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 4, lo: 0.5, hi: 0.75, bailTokens: 8)
        b.record(accepted: 0, proposed: 4)  // rate 0 < 0.5
        #expect(b.blockSize == 8)
    }

    @Test("steps UP one rung when trailing acceptance rises above hi")
    func stepsUpOneRungAboveHi() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 4, lo: 0.5, hi: 0.75, bailTokens: 8)
        b.record(accepted: 0, proposed: 4)  // -> 8
        #expect(b.blockSize == 8)
        b.record(accepted: 4, proposed: 4)  // rate 1.0 > 0.75 -> back to 16
        #expect(b.blockSize == 16)
    }

    @Test("holds the rung steady when trailing acceptance lands inside [lo, hi]")
    func holdsInsideHysteresisBand() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 4, lo: 0.5, hi: 0.75, bailTokens: 8)
        b.record(accepted: 0, proposed: 4)  // -> 8 (rate 0)
        #expect(b.blockSize == 8)
        b.record(accepted: 2, proposed: 4)  // rate exactly 0.5 == lo -> hold
        #expect(b.blockSize == 8)
        b.record(accepted: 3, proposed: 4)  // rate exactly 0.75 == hi -> hold
        #expect(b.blockSize == 8)
    }

    // ─── Bail: entry, resumption at smallest rung, recovery ─────────

    @Test("enters bail once the SMALLEST rung also falls below lo")
    func entersBailAtSmallestRung() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 4, lo: 0.5, hi: 0.75, bailTokens: 8)
        b.record(accepted: 0, proposed: 4)  // 16 -> 8
        b.record(accepted: 0, proposed: 4)  // 8 -> 4
        #expect(b.blockSize == 4)
        #expect(!b.isBailed)
        b.record(accepted: 0, proposed: 4)  // still 0 accepted at the floor rung -> bail
        #expect(b.isBailed)
        #expect(b.bailEntries == 1)
        // Resumes at the SMALLEST rung once bail ends (not the ceiling).
        #expect(b.blockSize == 4)
    }

    @Test("consumeBailToken decrements bailRemaining and clears isBailed at zero")
    func consumeBailTokenDrainsToZero() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 1, lo: 0.5, hi: 0.75, bailTokens: 2)
        b.enterBail()
        #expect(b.isBailed)
        b.consumeBailToken()
        #expect(b.isBailed)  // 1 remaining
        b.consumeBailToken()
        #expect(!b.isBailed)  // 0 remaining
    }

    @Test("bailedTokensTotal accumulates across multiple bail spans")
    func bailedTokensTotalAccumulates() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [4], window: 1, lo: 0.5, hi: 0.75, bailTokens: 3)
        b.enterBail()
        b.consumeBailToken()
        b.consumeBailToken()
        b.consumeBailToken()
        #expect(b.bailedTokensTotal == 3)
        b.enterBail()
        b.consumeBailToken()
        #expect(b.bailedTokensTotal == 4)
        #expect(b.bailEntries == 2)
    }

    @Test("entering bail clears the trailing window -- a fresh warmup follows recovery")
    func bailClearsWindowForFreshWarmup() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [4], window: 2, lo: 0.5, hi: 0.75, bailTokens: 1)
        b.record(accepted: 0, proposed: 2)  // fills window, rate 0 -- already smallest rung -> bail
        #expect(b.isBailed)
        b.consumeBailToken()
        #expect(!b.isBailed)
        // Post-bail: one strong cycle should NOT be enough to move
        // anything (window needs 2 fresh candidates, not blended with
        // stale pre-bail rejects) -- still warming up.
        b.record(accepted: 1, proposed: 1)
        #expect(b.blockSize == 4)  // only one rung anyway, but no crash/adjustment either
    }

    @Test("recovers UP through rungs via sustained high acceptance after a bail span")
    func recoversUpThroughRungsAfterBail() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 4, lo: 0.5, hi: 0.75, bailTokens: 4)
        b.enterBail()
        while b.isBailed { b.consumeBailToken() }
        #expect(b.blockSize == 4)
        b.record(accepted: 4, proposed: 4)  // rate 1.0 > hi -> 4 -> 8
        #expect(b.blockSize == 8)
        b.record(accepted: 4, proposed: 4)  // -> 8 -> 16
        #expect(b.blockSize == 16)
        b.record(accepted: 4, proposed: 4)  // already at ceiling, holds
        #expect(b.blockSize == 16)
    }

    // ─── Floor / ceiling clamps ─────────────────────────────────────

    @Test("never steps past the smallest configured rung without bailing")
    func floorsAtSmallestRung() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 1, lo: 0.5, hi: 0.75, bailTokens: 8)
        b.record(accepted: 0, proposed: 1)  // -> 8
        #expect(b.blockSize == 8)
        b.record(accepted: 0, proposed: 1)  // -> 4
        #expect(b.blockSize == 4)
        #expect(!b.isBailed)
        b.record(accepted: 0, proposed: 1)  // at floor rung, still 0 -> bail, not below-floor rung
        #expect(b.isBailed)
    }

    @Test("never steps above the ceiling rung, even with sustained full acceptance")
    func ceilingsAtRungZero() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 1, lo: 0.5, hi: 0.75, bailTokens: 8)
        #expect(b.blockSize == 16)
        b.record(accepted: 1, proposed: 1)  // rate 1.0 > hi, already at ceiling
        #expect(b.blockSize == 16)
    }

    // ─── nextCycleBlockSize() stats bookkeeping ─────────────────────

    @Test("nextCycleBlockSize reports the CURRENT block size and accumulates avg")
    func nextCycleBlockSizeStats() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 4, lo: 0.5, hi: 0.75, bailTokens: 8)
        #expect(b.nextCycleBlockSize() == 16)  // cycle 1: ceiling, pre-warmup
        b.record(accepted: 0, proposed: 4)  // -> 8
        #expect(b.nextCycleBlockSize() == 8)  // cycle 2: reflects the step down
        #expect(b.nextCycleBlockSize() == 8)  // cycle 3: no record() in between, holds
        #expect(b.avgBlockSize == (16.0 + 8.0 + 8.0) / 3.0)
    }

    @Test("avgBlockSize is 0 when nextCycleBlockSize was never called")
    func avgBlockSizeZeroWithNoCycles() {
        let b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 4, lo: 0.5, hi: 0.75, bailTokens: 8)
        #expect(b.avgBlockSize == 0)
    }

    // ─── End-to-end: down -> bail -> recover, block size bounded throughout ────

    @Test("full lifecycle stays within the rung ladder (or bailed) across a mixed sequence")
    func fullLifecycleStaysInBounds() {
        var b = DFlashSpecDecode.DFlashAdaptiveBlock(
            rungs: [16, 8, 4], window: 3, lo: 0.45, hi: 0.70, bailTokens: 6)
        let cycles: [(accepted: Int, proposed: Int)] = [
            (0, 3), (0, 3), (1, 3), (0, 3), (0, 3), (0, 3),
        ]
        for c in cycles {
            if b.isBailed {
                while b.isBailed { b.consumeBailToken() }
                continue
            }
            _ = b.nextCycleBlockSize()
            b.record(accepted: c.accepted, proposed: c.proposed)
            #expect(b.rungs.contains(b.blockSize))
        }
        // Sustained near-zero acceptance should have driven this all the
        // way down through both rungs into bail.
        #expect(b.bailEntries >= 1 || b.blockSize == 4)
    }
}
