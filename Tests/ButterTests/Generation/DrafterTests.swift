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
// Host-side unit test for the n-gram + tree drafters — no GPU needed.

import Foundation
import Testing

@testable import ButterSwift

@Suite("TreeDrafter — scaffolding")
struct TreeDrafterScaffoldTests {

    @Test("DraftTreeNode size + depth on a linear chain")
    func linearChainStructure() {
        // Build: 1 → 2 → 3 (depth 3, size 3, single child each).
        let leaf = DraftTreeNode(token: 3)
        let mid = DraftTreeNode(token: 2, children: [leaf])
        let root = DraftTreeNode(token: 1, children: [mid])
        #expect(root.size == 3)
        #expect(root.depth == 3)
        #expect(root.children.count == 1)
    }

    @Test("DraftTreeNode size + depth on a branching tree")
    func branchingTreeStructure() {
        // Root → [A, B]; A → [C]; B → [D, E]. size = 6, depth = 3.
        let c = DraftTreeNode(token: 3)
        let d = DraftTreeNode(token: 4)
        let e = DraftTreeNode(token: 5)
        let a = DraftTreeNode(token: 1, children: [c])
        let b = DraftTreeNode(token: 2, children: [d, e])
        let root = DraftTreeNode(token: 0, children: [a, b])
        #expect(root.size == 6)
        #expect(root.depth == 3)
    }

    @Test("Drafter.proposeTreeLinear converts linear propose to a chain tree")
    func linearAdapterMatchesChain() {
        let the = 1
        let cat = 2
        let sat = 3
        let on = 4
        let history = [the, cat, sat, on, the, cat, sat]
        let nGram = NGramDrafter(maxNMatch: 3, minNMatch: 2)
        let linear = nGram.propose(history: history, gamma: 2)
        guard let tree = nGram.proposeTreeLinear(history: history, maxDepth: 2)
        else {
            Issue.record("expected a non-nil tree")
            return
        }
        #expect(tree.size == linear.count)
        #expect(tree.depth == linear.count)
        // Walk single-child chain comparing tokens.
        var node: DraftTreeNode? = tree
        for tok in linear {
            #expect(node?.token == tok)
            node = node?.children.first
        }
    }

    @Test("LinearTreeAdapter wraps any Drafter as a TreeDrafter")
    func linearAdapterIsTreeDrafter() {
        let drafter = NGramDrafter(maxNMatch: 3, minNMatch: 1)
        let tree: TreeDrafter = LinearTreeAdapter(drafter)
        let history = [1, 2, 3, 4, 1, 2, 3]
        let result = tree.proposeTree(
            history: history, maxDepth: 2, maxNodes: 16)
        #expect(result != nil)
        if let r = result {
            var node: DraftTreeNode? = r
            while let n = node {
                #expect(n.children.count <= 1)
                node = n.children.first
            }
        }
    }

    @Test("LinearTreeAdapter returns nil when the underlying drafter is empty")
    func linearAdapterEmptyProposal() {
        let drafter = NeverDrafter()
        let tree = LinearTreeAdapter(drafter)
        let result = tree.proposeTree(
            history: [1, 2, 3], maxDepth: 4, maxNodes: 16)
        #expect(result == nil)
    }
}

@Suite("NGramTreeDrafter — branching tree from n-gram history")
struct NGramTreeDrafterTests {

    @Test("proposeTree branches on multiple continuations of the same n-gram")
    func branchesOnMultipleContinuations() {
        // History where trigram "the cat sat" appears 3 times with
        // different continuations: ["on", "on", "down"]. Top-2 should
        // be "on" (count = 2) then "down" (count = 1).
        let the = 1
        let cat = 2
        let sat = 3
        let on = 4
        let down = 5
        let pad = 99
        let history = [
            the, cat, sat, on, pad,
            the, cat, sat, on, pad,
            the, cat, sat, down, pad,
            the, cat, sat,
        ]
        let drafter = NGramTreeDrafter(
            maxNMatch: 3, minNMatch: 3, branchingFactor: 2)
        guard
            let tree = drafter.proposeTree(
                history: history, maxDepth: 1, maxNodes: 16)
        else {
            Issue.record("expected a non-nil tree")
            return
        }
        // Root token = top-1 = "on" (count = 2).
        #expect(tree.token == on)
        // depth = 1, branching = 2 → root has 2 children: "on" + "down".
        #expect(tree.children.count == 2)
        #expect(tree.children.map(\.token).sorted() == [on, down].sorted())
    }

    @Test("proposeTree returns nil when the n-gram never repeats")
    func nilOnNoMatch() {
        let history = [1, 2, 3, 4, 5]
        let drafter = NGramTreeDrafter(
            maxNMatch: 3, minNMatch: 2, branchingFactor: 2)
        let tree = drafter.proposeTree(
            history: history, maxDepth: 2, maxNodes: 16)
        #expect(tree == nil)
    }

    @Test("propose (linear) falls back to a top-1 chain")
    func linearFallbackProposeChain() {
        // Bigram "a b" → "c" appears 3×, then once more.
        let a = 10
        let b = 11
        let c = 12
        let history = [a, b, c, 99, a, b, c, 99, a, b, c, 99, a, b]
        let drafter = NGramTreeDrafter(
            maxNMatch: 2, minNMatch: 2, branchingFactor: 3)
        let linear = drafter.propose(history: history, gamma: 1)
        #expect(linear == [c])
    }

    @Test("nodeBudget caps tree growth")
    func nodeBudgetCap() {
        let t = [1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 2, 3, 4, 7, 1, 2, 3, 4]
        let drafter = NGramTreeDrafter(
            maxNMatch: 3, minNMatch: 2, branchingFactor: 3)
        guard
            let tree = drafter.proposeTree(
                history: t, maxDepth: 4, maxNodes: 4)
        else {
            Issue.record("expected non-nil")
            return
        }
        #expect(tree.size <= 4)
    }

    @Test("flatten + treeCausalMask: linear chain → lower-triangular mask")
    func flattenLinearChain() {
        // chain 1 → 2 → 3 → 4 (depth 4, branching 1)
        let leaf = DraftTreeNode(token: 4)
        let n3 = DraftTreeNode(token: 3, children: [leaf])
        let n2 = DraftTreeNode(token: 2, children: [n3])
        let root = DraftTreeNode(token: 1, children: [n2])
        let (tokens, parent, paths) = root.flatten()
        #expect(tokens == [1, 2, 3, 4])
        #expect(parent == [-1, 0, 1, 2])
        #expect(paths[0] == [0])
        #expect(paths[1] == [0, 1])
        #expect(paths[2] == [0, 1, 2])
        #expect(paths[3] == [0, 1, 2, 3])
        // Linear chain mask = lower triangular (each token attends to
        // every prior).
        let (mask, t) = root.treeCausalMask()
        #expect(t == 4)
        for i in 0 ..< t {
            for j in 0 ..< t {
                let m = mask[i * t + j]
                let expected: Float = (j <= i) ? 0.0 : -Float.infinity
                #expect(m == expected)
            }
        }
    }

    /// Helper: build the branching tree used by verify-case tests.
    private func cousinsTree() -> DraftTreeNode {
        //         0 (root, tok = 10)
        //        / \
        //      1     2     (tok = 20, tok = 30)
        //     /|     |
        //    3 4     5     (tok = 40, tok = 50, tok = 60)
        let n3 = DraftTreeNode(token: 40)
        let n4 = DraftTreeNode(token: 50)
        let n5 = DraftTreeNode(token: 60)
        let n1 = DraftTreeNode(token: 20, children: [n3, n4])
        let n2 = DraftTreeNode(token: 30, children: [n5])
        return DraftTreeNode(token: 10, children: [n1, n2])
    }

    @Test("verify: root mismatch returns empty accepted + the bonus token")
    func verifyRootMismatch() {
        let tree = cousinsTree()
        // Oracle says the target wants token 99 first (not the root's 10).
        let oracleHistoryEnd = 99
        let result = tree.verify(
            oracleAtHistoryEnd: oracleHistoryEnd,
            oracle: { _ in 0 })  // unused — root rejected
        #expect(result.acceptedTokens.isEmpty)
        #expect(result.bonusToken == 99)
    }

    @Test("verify: oracle agrees with path 0→1→3, then diverges")
    func verifyAcceptsPathToN3() {
        let tree = cousinsTree()
        // After history, target wants 10 (root). At root (flat 0),
        // target wants 20 (descend into n1). At n1 (flat 1), target
        // wants 40 (descend into n3). At n3 (flat 2), target wants
        // 999 (off-tree) — stop.
        let oracle: (Int) -> Int = { flat in
            switch flat {
            case 0: return 20  // pick n1 (token = 20)
            case 1: return 40  // pick n3 (token = 40)
            case 2: return 999  // off-tree, bonus
            default: return -1
            }
        }
        let result = tree.verify(oracleAtHistoryEnd: 10, oracle: oracle)
        #expect(result.acceptedTokens == [10, 20, 40])
        #expect(result.bonusToken == 999)
    }

    @Test("verify: oracle picks a sibling at depth 1 → only root accepted")
    func verifySiblingOnlyRoot() {
        let tree = cousinsTree()
        let oracle: (Int) -> Int = { flat in
            switch flat {
            case 0: return 30  // pick n2 over n1
            case 4: return 60  // descend to n5
            case 5: return 999  // off-tree bonus
            default: return -1
            }
        }
        let result = tree.verify(oracleAtHistoryEnd: 10, oracle: oracle)
        #expect(result.acceptedTokens == [10, 30, 60])
        #expect(result.bonusToken == 999)
    }

    @Test("verify: full path accept on a linear chain tree")
    func verifyFullLinearPath() {
        let leaf = DraftTreeNode(token: 4)
        let n3 = DraftTreeNode(token: 3, children: [leaf])
        let n2 = DraftTreeNode(token: 2, children: [n3])
        let root = DraftTreeNode(token: 1, children: [n2])
        let oracle: (Int) -> Int = { flat in
            switch flat {
            case 0: return 2
            case 1: return 3
            case 2: return 4
            case 3: return 999  // bonus after the leaf
            default: return -1
            }
        }
        let result = root.verify(oracleAtHistoryEnd: 1, oracle: oracle)
        #expect(result.acceptedTokens == [1, 2, 3, 4])
        #expect(result.bonusToken == 999)
    }

    @Test("flatten + treeCausalMask: branching tree masks cousins/siblings")
    func flattenBranchingTreeMasksCousins() {
        //         0 (root)
        //        / \
        //       1   2
        //      /|   |
        //     3 4   5
        // DFS order: 0, 1, 3, 4, 2, 5
        let n3 = DraftTreeNode(token: 30)
        let n4 = DraftTreeNode(token: 40)
        let n5 = DraftTreeNode(token: 50)
        let n1 = DraftTreeNode(token: 10, children: [n3, n4])
        let n2 = DraftTreeNode(token: 20, children: [n5])
        let root = DraftTreeNode(token: 0, children: [n1, n2])
        let (tokens, parent, _) = root.flatten()
        #expect(tokens == [0, 10, 30, 40, 20, 50])
        #expect(parent == [-1, 0, 1, 1, 0, 4])

        let (mask, t) = root.treeCausalMask()
        #expect(t == 6)
        // Which flat-indices is each node allowed to attend?
        let allow0: Set<Int> = [0]
        let allow1: Set<Int> = [0, 1]
        let allow2: Set<Int> = [0, 1, 2]
        let allow3: Set<Int> = [0, 1, 3]  // n4 — NOT n3 (sibling)
        let allow4: Set<Int> = [0, 4]  // n2 — NOT n1 / n3 / n4
        let allow5: Set<Int> = [0, 4, 5]  // n5 — NOT n1 / n3 / n4
        let allowed: [Set<Int>] = [
            allow0, allow1, allow2, allow3, allow4, allow5,
        ]

        for i in 0 ..< t {
            for j in 0 ..< t {
                let m = mask[i * t + j]
                let expected: Float =
                    allowed[i].contains(j) ? 0.0 : -Float.infinity
                #expect(m == expected)
            }
        }
    }

    @Test("frequency-tie deterministic order")
    func frequencyTieDeterministicOrder() {
        // Bigram "a b" → "c" once, → "d" once (tie). Top-2 returns
        // both; tie-break is by token id ascending → [c, d] when c < d.
        let a = 1
        let b = 2
        let c = 3
        let d = 4
        let history = [a, b, c, 99, a, b, d, 99, a, b]
        let drafter = NGramTreeDrafter(
            maxNMatch: 2, minNMatch: 2, branchingFactor: 2)
        guard
            let tree = drafter.proposeTree(
                history: history, maxDepth: 1, maxNodes: 8)
        else {
            Issue.record("expected non-nil")
            return
        }
        // Both c and d have count = 1; tie-break ascending → root = c.
        #expect(tree.token == c)
        #expect(tree.children.map(\.token) == [c, d])
    }
}

@Suite("Drafter — n-gram prompt-lookup")
struct DrafterTests {

    @Test("NGramDrafter proposes the continuation after a repeated trigram")
    func nGramFindsTrigramContinuation() throws {
        let the = 1
        let cat = 2
        let sat = 3
        let on = 4
        let mat = 5
        let dog = 6
        let ran = 7
        // [the, cat, sat, on, the, mat, dog, ran, the, cat, sat]
        // Latest 3 = "the cat sat"; earliest at indices 0..2;
        // next 3 after that = [on, the, mat].
        let history = [the, cat, sat, on, the, mat, dog, ran, the, cat, sat]

        let drafter = NGramDrafter(maxNMatch: 3, minNMatch: 2)
        let proposal = drafter.propose(history: history, gamma: 3)
        #expect(
            proposal == [on, the, mat],
            "expected [on, the, mat], got \(proposal)")
    }

    @Test("NGramDrafter falls back to a shorter match when longest absent")
    func nGramBigramFallback() throws {
        let a = 1
        let b = 2
        let c = 3
        let d = 4
        // [a, b, c, d, a, b] — bigram "a, b" repeats; trigram absent.
        let history = [a, b, c, d, a, b]
        let drafter = NGramDrafter(maxNMatch: 3, minNMatch: 2)
        let proposal = drafter.propose(history: history, gamma: 2)
        #expect(
            proposal == [c, d],
            "expected bigram fallback [c, d], got \(proposal)")
    }

    @Test("NGramDrafter returns empty when no match is found")
    func nGramNoMatchReturnsEmpty() throws {
        let history = [1, 2, 3, 4, 5]
        let drafter = NGramDrafter(maxNMatch: 3, minNMatch: 2)
        let proposal = drafter.propose(history: history, gamma: 3)
        #expect(proposal.isEmpty, "expected empty, got \(proposal)")
    }

    @Test("NeverDrafter always proposes empty")
    func neverDrafterEmpty() throws {
        let drafter = NeverDrafter()
        #expect(drafter.propose(history: [1, 2, 3], gamma: 4).isEmpty)
    }
}
