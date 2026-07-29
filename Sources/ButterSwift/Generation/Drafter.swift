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
// Drafter — interface for speculative-decode candidate proposal.
//
// A drafter takes the token history so far and proposes one or more
// candidate next tokens. The target model then verifies via a batched
// forward over `[lastAccepted, ...candidates]` and either accepts
// (commit candidates) or rejects (restore caches, commit the model's
// actual choice from the verify logits).
//
// Implementations:
//   * `NGramDrafter` — prompt-lookup n-gram. Zero ML cost; works well
//     on repetitive contexts (code, structured chat).
//   * `NeverDrafter` — stub that always returns nothing. Used to
//     measure pure spec-decode driver overhead vs raw decode.
//   * `NGramTreeDrafter` — real branching n-gram tree drafter for
//     tree-verify spec-decode (top-K continuations per depth).

import Foundation

/// A drafter proposes candidate next tokens for the target model to
/// verify in a speculative-decode loop.
public protocol Drafter: AnyObject {
    /// Propose up to `gamma` candidate tokens. Implementations may
    /// return fewer (or an empty array) if they can't make a confident
    /// proposal — the driver falls back to a plain decode step in
    /// that case.
    ///
    /// `history` is the full sequence so far (prompt + generated).
    /// `gamma` is the maximum candidates the driver is asking for.
    func propose(history: [Int], gamma: Int) -> [Int]
}

/// Prompt-lookup n-gram drafter — zero ML cost; works well on
/// repetitive contexts (code, structured chat).
///
/// Algorithm:
///   1. Take the last `nMatch` tokens of `history` as the lookup key.
///   2. Scan backwards through `history` looking for a previous
///      occurrence of that key.
///   3. If found, return the next `gamma` tokens AFTER that earlier
///      occurrence — those are the candidate next tokens.
///   4. If not found, fall back to shorter keys (`nMatch - 1`,
///      `nMatch - 2`, ...) before giving up.
public final class NGramDrafter: Drafter {
    /// Largest match length to try first. Falls back to shorter
    /// lengths if no longer match is found. Typical: 3 (trigram).
    public let maxNMatch: Int
    /// Smallest match length the drafter will try before giving up.
    /// Default 2 (unigram lookup is too noisy in practice and pushes
    /// average acceptance below the γ=2 break-even of ~70%).
    public let minNMatch: Int

    public init(maxNMatch: Int = 3, minNMatch: Int = 2) {
        precondition(
            maxNMatch >= minNMatch && minNMatch >= 1,
            "NGramDrafter: maxNMatch (\(maxNMatch)) must be ≥ minNMatch (\(minNMatch)) ≥ 1")
        self.maxNMatch = maxNMatch
        self.minNMatch = minNMatch
    }

    public func propose(history: [Int], gamma: Int) -> [Int] {
        precondition(gamma >= 0, "NGramDrafter.propose: gamma must be ≥ 0")
        guard gamma > 0, !history.isEmpty else { return [] }

        for nMatch in stride(from: maxNMatch, through: minNMatch, by: -1) {
            guard history.count >= nMatch else { continue }
            let keyStart = history.count - nMatch
            let key = Array(history[keyStart ..< history.count])
            // Scan backwards from just-before the key (so we don't
            // match the key against itself).
            var probe = keyStart - 1
            while probe >= nMatch - 1 {
                if matches(history, at: probe - (nMatch - 1), key: key) {
                    let candidateStart = probe + 1
                    let candidateEnd = Swift.min(
                        candidateStart + gamma, history.count)
                    if candidateEnd > candidateStart {
                        return Array(history[candidateStart ..< candidateEnd])
                    }
                }
                probe -= 1
            }
        }
        return []
    }

    @inline(__always)
    private func matches(_ history: [Int], at start: Int, key: [Int]) -> Bool {
        if start < 0 || start + key.count > history.count { return false }
        for i in 0 ..< key.count {
            if history[start + i] != key[i] { return false }
        }
        return true
    }
}

/// Stub drafter that never proposes anything. Used to measure pure
/// spec-decode driver overhead against the no-spec baseline.
public final class NeverDrafter: Drafter {
    public init() {}
    public func propose(history: [Int], gamma: Int) -> [Int] { [] }
}

// ─── Tree drafter (tree-verify spec decode) ──────────────────────────
//
// Linear γ=2 spec decode caps at 1.83 expected accepted tokens per
// verify cycle (83% acceptance × 2 candidates + 1 verified). Tree
// drafters expand multiple continuations per draft step, verifying
// ALL of them in one forward pass with a tree-causal attention mask.
// At γ=8 tree (e.g., 2-branch at depth 3) with 50–65% per-branch
// acceptance, expected accepted tokens jumps to 3–4 → 1.7–2× decode
// over linear.

/// One node in a draft tree. The root represents the first proposed
/// token following `history`; each child represents a possible
/// continuation after its parent. A linear chain (one child per node)
/// degenerates to the existing `[Int]` candidate sequence.
public struct DraftTreeNode: Sendable {
    /// The token this node proposes.
    public let token: Int
    /// Continuations after this token. Empty at leaves.
    public let children: [DraftTreeNode]

    public init(token: Int, children: [DraftTreeNode] = []) {
        self.token = token
        self.children = children
    }

    /// Total node count (root + all descendants).
    public var size: Int {
        return 1 + children.reduce(0) { $0 + $1.size }
    }

    /// Maximum depth (root alone = 1).
    public var depth: Int {
        return 1 + (children.map(\.depth).max() ?? 0)
    }

    /// Depth-first flatten of the tree. Returns:
    ///   - `tokens[i]`: the token at flat position `i` (root at 0).
    ///   - `parentIndex[i]`: the flat-index of node `i`'s parent, or
    ///     `-1` for the root (`i == 0`).
    ///   - `pathFromRoot[i]`: indices of node `i`'s ancestors INCLUDING
    ///     `i` itself, root-first (length = node `i`'s depth in the
    ///     tree). Used by the verify driver to walk the accepted prefix.
    public func flatten() -> (
        tokens: [Int], parentIndex: [Int], pathFromRoot: [[Int]]
    ) {
        var tokens: [Int] = []
        var parent: [Int] = []
        var paths: [[Int]] = []
        var indexStack: [Int] = []  // DFS path-to-this-node, by flat-index
        func recurse(_ node: DraftTreeNode, parentIdx: Int) {
            let myIdx = tokens.count
            tokens.append(node.token)
            parent.append(parentIdx)
            indexStack.append(myIdx)
            paths.append(indexStack)
            for child in node.children {
                recurse(child, parentIdx: myIdx)
            }
            indexStack.removeLast()
        }
        recurse(self, parentIdx: -1)
        return (tokens, parent, paths)
    }

    /// Result of walking a draft tree against the target's argmax
    /// predictions.
    public struct VerifyResult: Equatable, Sendable {
        /// The accepted-path tokens — root + each child whose token
        /// matched the target's argmax at the prior depth, inclusive.
        /// Empty if the root didn't match the target's first prediction.
        public let acceptedTokens: [Int]
        /// The "bonus" token from the target's argmax at the deepest
        /// accepted flat-position (or at the pre-tree position if
        /// even the root failed). This is the standard spec-decode
        /// guaranteed-correct-token-for-free.
        public let bonusToken: Int
    }

    /// Walk the tree against the target's per-position argmax oracle;
    /// return the longest accepted path + the bonus token.
    ///
    /// Protocol:
    ///   1. `oracleAtHistoryEnd` is the target's argmax of logits at
    ///      the position JUST BEFORE the tree (the last token in
    ///      history). It's the target's preferred root token. If it
    ///      doesn't equal `self.token`, the root is rejected — return
    ///      no accepted tokens + `oracleAtHistoryEnd` as the bonus.
    ///   2. Otherwise the root is accepted; descend by repeatedly
    ///      looking up `oracle(currentFlatIndex)` and accepting
    ///      whichever child has that token. Stop when no child
    ///      matches; return accepted path + `oracle(lastAcceptedIdx)`
    ///      as the bonus.
    ///
    /// Pure function. No model / cache dependencies. The driver wires
    /// this up with `oracle = { i in target_logits[i].argmax() }`.
    public func verify(
        oracleAtHistoryEnd: Int, oracle: (Int) -> Int
    ) -> VerifyResult {
        if self.token != oracleAtHistoryEnd {
            return VerifyResult(
                acceptedTokens: [], bonusToken: oracleAtHistoryEnd)
        }
        let (tokens, parents, _) = flatten()
        let n = tokens.count
        var childrenOf: [[Int]] = Array(repeating: [], count: n)
        for i in 1 ..< n {
            childrenOf[parents[i]].append(i)
        }
        var acceptedPath: [Int] = [tokens[0]]
        var currentFlat = 0
        while true {
            let predictedNext = oracle(currentFlat)
            if let nextFlat = childrenOf[currentFlat]
                .first(where: { tokens[$0] == predictedNext })
            {
                acceptedPath.append(tokens[nextFlat])
                currentFlat = nextFlat
                continue
            }
            return VerifyResult(
                acceptedTokens: acceptedPath, bonusToken: predictedNext)
        }
    }

    /// Tree-causal additive attention mask for the in-tree positions.
    /// Returns a flat `[T·T]` array where `mask[i·T + j]` is:
    ///   - `0.0` if flat-index `j` is an ancestor of `i` in the tree,
    ///     or `j == i` itself (the diagonal — every token attends to
    ///     itself).
    ///   - `-Float.infinity` otherwise (siblings / cousins / disjoint
    ///     branches — must NOT attend across alternative paths).
    ///
    /// Caller adds this mask onto the attention scores BEFORE softmax.
    /// The cached-prefix portion (positions < `baseKV`) is always
    /// attended (full causal-to-cache) and is NOT included here —
    /// wrap this mask into the kernel's `mask` param only for the
    /// in-block region.
    public func treeCausalMask() -> (mask: [Float], t: Int) {
        let (_, parent, _) = flatten()
        let t = parent.count
        var mask = [Float](repeating: -Float.infinity, count: t * t)
        for i in 0 ..< t {
            var node = i
            while node != -1 {
                mask[i * t + node] = 0.0
                node = parent[node]
            }
        }
        return (mask, t)
    }
}

/// A drafter that proposes a tree of candidate continuations. The
/// SpecDecode driver walks the tree, flattens to a tree-causal
/// attention mask, and verifies all candidates in ONE forward pass —
/// accepting the longest matching prefix.
public protocol TreeDrafter: AnyObject {
    /// Propose a tree rooted at the first token after `history`.
    /// Returns `nil` when the drafter has no confident proposal — the
    /// driver falls back to a plain decode step.
    ///
    /// `maxDepth` caps the depth of the returned tree (root counts as
    /// depth 1). `maxNodes` caps the total node count so the driver
    /// can budget the verify forward pass.
    func proposeTree(
        history: [Int], maxDepth: Int, maxNodes: Int
    ) -> DraftTreeNode?
}

extension Drafter {
    /// Convert a linear `propose` result into a degenerate tree (one
    /// child per node). Used to expose any existing `Drafter` as a
    /// `TreeDrafter` without adding a real branching policy.
    public func proposeTreeLinear(
        history: [Int], maxDepth: Int
    ) -> DraftTreeNode? {
        let linear = propose(history: history, gamma: maxDepth)
        guard let last = linear.last else { return nil }
        // Build from leaf inward so children-of-children compose
        // correctly.
        var node = DraftTreeNode(token: last, children: [])
        for t in linear.dropLast().reversed() {
            node = DraftTreeNode(token: t, children: [node])
        }
        return node
    }
}

/// Adapter: any linear `Drafter` becomes a `TreeDrafter` that emits a
/// degenerate (single-branch) tree. Useful for A/B-ing the
/// tree-verify driver against the linear baseline using the existing
/// `NGramDrafter`.
public final class LinearTreeAdapter: TreeDrafter {
    public let inner: Drafter
    public init(_ inner: Drafter) { self.inner = inner }
    public func proposeTree(
        history: [Int], maxDepth: Int, maxNodes _: Int
    ) -> DraftTreeNode? {
        inner.proposeTreeLinear(history: history, maxDepth: maxDepth)
    }
}

/// Real branching n-gram drafter.
///
/// Extends `NGramDrafter`'s "scan history for n-gram matches" lookup
/// from "first match → linear chain" to "all matches → top-K
/// continuations per depth → branching tree." Each node expands its
/// `branchingFactor` most-frequent continuations from past
/// occurrences of the current key, recursively up to `maxDepth` or
/// `maxNodes` total.
///
/// Conforms to both `Drafter` (linear γ fallback via `propose`) and
/// `TreeDrafter` (real branching via `proposeTree`).
public final class NGramTreeDrafter: Drafter, TreeDrafter {
    public let maxNMatch: Int
    public let minNMatch: Int
    /// Number of children per node (top-K continuations).
    public let branchingFactor: Int

    public init(
        maxNMatch: Int = 3, minNMatch: Int = 2,
        branchingFactor: Int = 2
    ) {
        precondition(
            maxNMatch >= minNMatch && minNMatch >= 1,
            "NGramTreeDrafter: maxNMatch (\(maxNMatch)) must be ≥ minNMatch (\(minNMatch)) ≥ 1")
        precondition(
            branchingFactor >= 1,
            "NGramTreeDrafter: branchingFactor must be ≥ 1")
        self.maxNMatch = maxNMatch
        self.minNMatch = minNMatch
        self.branchingFactor = branchingFactor
    }

    // MARK: Drafter — linear γ fallback (top-1 chain).

    public func propose(history: [Int], gamma: Int) -> [Int] {
        precondition(
            gamma >= 0, "NGramTreeDrafter.propose: gamma must be ≥ 0")
        guard gamma > 0, !history.isEmpty else { return [] }
        var chain: [Int] = []
        chain.reserveCapacity(gamma)
        var extended = history
        for _ in 0 ..< gamma {
            guard let token = topKContinuations(of: extended, k: 1).first
            else { break }
            chain.append(token)
            extended.append(token)
        }
        return chain
    }

    // MARK: TreeDrafter — branching tree.

    public func proposeTree(
        history: [Int], maxDepth: Int, maxNodes: Int
    ) -> DraftTreeNode? {
        guard maxDepth > 0, maxNodes > 0, !history.isEmpty else {
            return nil
        }
        var nodeBudget = maxNodes
        let roots = topKContinuations(of: history, k: branchingFactor)
        guard let rootTok = roots.first else { return nil }
        nodeBudget -= 1
        // Each `roots[i]` becomes a child of the root at depth 1, then
        // recurse downwards `maxDepth - 1`.
        var rootChildren: [DraftTreeNode] = []
        rootChildren.reserveCapacity(roots.count)
        for tok in roots {
            guard nodeBudget > 0 else { break }
            nodeBudget -= 1  // RESERVE this child's slot before recursing.
            var path = history
            path.append(tok)
            let subtree = buildSubtree(
                extendedHistory: path,
                remainingDepth: maxDepth - 1,
                nodeBudget: &nodeBudget)
            rootChildren.append(
                DraftTreeNode(
                    token: tok, children: subtree?.children ?? []))
        }
        return DraftTreeNode(token: rootTok, children: rootChildren)
    }

    /// Recursive helper: build a chain-or-tree rooted at the implicit
    /// next-token, capped by `remainingDepth` and `nodeBudget`. Budget
    /// is reserved BEFORE recursing so the cap is a hard upper bound
    /// on `tree.size`.
    private func buildSubtree(
        extendedHistory: [Int],
        remainingDepth: Int,
        nodeBudget: inout Int
    ) -> DraftTreeNode? {
        guard remainingDepth > 0, nodeBudget > 0 else { return nil }
        let toks = topKContinuations(of: extendedHistory, k: branchingFactor)
        guard let firstTok = toks.first else { return nil }
        var children: [DraftTreeNode] = []
        children.reserveCapacity(toks.count)
        for tok in toks {
            guard nodeBudget > 0 else { break }
            nodeBudget -= 1  // RESERVE before recursing.
            var deeper = extendedHistory
            deeper.append(tok)
            let sub = buildSubtree(
                extendedHistory: deeper,
                remainingDepth: remainingDepth - 1,
                nodeBudget: &nodeBudget)
            children.append(
                DraftTreeNode(token: tok, children: sub?.children ?? []))
        }
        return DraftTreeNode(token: firstTok, children: children)
    }

    /// Top-K most frequent continuations after `history` (longest
    /// available n-gram → fall back to shorter). Returns up to `k`
    /// tokens sorted by occurrence count descending; ties broken by
    /// token id ascending for determinism.
    private func topKContinuations(of history: [Int], k: Int) -> [Int] {
        guard !history.isEmpty, k > 0 else { return [] }
        for nMatch in stride(from: maxNMatch, through: minNMatch, by: -1) {
            guard history.count >= nMatch else { continue }
            let keyStart = history.count - nMatch
            let key = Array(history[keyStart ..< history.count])
            var counts: [Int: Int] = [:]
            var probe = keyStart - 1
            while probe >= nMatch - 1 {
                if matches(history, at: probe - (nMatch - 1), key: key) {
                    let candidateIdx = probe + 1
                    if candidateIdx < history.count {
                        counts[history[candidateIdx], default: 0] += 1
                    }
                }
                probe -= 1
            }
            if !counts.isEmpty {
                return counts.sorted {
                    if $0.value != $1.value { return $0.value > $1.value }
                    return $0.key < $1.key
                }
                .prefix(k)
                .map(\.key)
            }
        }
        return []
    }

    @inline(__always)
    private func matches(_ history: [Int], at start: Int, key: [Int]) -> Bool {
        if start < 0 || start + key.count > history.count { return false }
        for i in 0 ..< key.count {
            if history[start + i] != key[i] { return false }
        }
        return true
    }
}
