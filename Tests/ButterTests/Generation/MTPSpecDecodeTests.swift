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
