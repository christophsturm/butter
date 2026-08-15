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
// Sampling — CPU sampling pipeline over a vocab-sized logits tensor.
//
// The greedy fast path is handled by Ops.argmax + LanguageModel.forwardSample
// (single 4-byte readback, no logits cross CPU↔GPU). When the user picks
// non-greedy sampling — temperature > 0 with any of top-K / top-P / min-P /
// repetition-penalty — we read the full vocab to CPU once per token and run
// the pipeline here. At Qwen3 vocab=151_936 fp16 that's ~304 KB per token,
// negligible bandwidth (~18 MB/s at 60 tok/s) on Apple Silicon's unified
// memory. A GPU softmax+categorical sample kernel is on the iron
// `ek/sampling-kernels` branch for the future fully-on-GPU path.
//
// Pipeline order (matches mlx-swift-lm / mlx-lm):
//
//   1. Repetition penalty (logit / penalty for seen tokens with positive
//      logits; logit * penalty for negative ones — Hugging Face convention)
//   2. Temperature scaling (logit / T; T == 0 → argmax shortcut)
//   3. Top-K filter (keep K largest logits)
//   4. Top-P / nucleus filter (keep tokens until cumulative softmax ≥ P)
//   5. Min-P filter (keep tokens with prob ≥ min_p × max_prob)
//   6. Softmax + categorical sample with a uniform draw from the RNG

import Foundation

public enum Sampling {

    // ─── Debug helpers ───────────────────────────────────────────────

    /// Debug: top-N highest logits as (index, value) pairs.
    public static func topN(_ logits: Tensor, n: Int) -> [(Int, Float)] {
        let values = decodeF32(logits)
        let indexed = values.enumerated().sorted { $0.element > $1.element }
        return indexed.prefix(n).map { ($0.offset, $0.element) }
    }

    /// Greedy: argmax over a 1D logits tensor. Tie-breaks toward the
    /// smallest index, matching the GPU argmax kernel.
    public static func argmax(_ logits: Tensor) -> Int {
        let values = decodeF32(logits)
        var best = 0
        var bestVal = values[0]
        for i in 1 ..< values.count where values[i] > bestVal {
            bestVal = values[i]
            best = i
        }
        return best
    }

    // ─── Full CPU sampling pipeline ──────────────────────────────────

    /// Sample a token id from `logits` per the given `parameters`. When
    /// `parameters.temperature == 0` the call short-circuits to a CPU
    /// argmax (deterministic, no RNG draw). `tokenHistory` is the
    /// already-generated prefix (prompt + decode so far); only used
    /// when `repetitionPenalty != 1.0`.
    public static func sample(
        _ logits: Tensor,
        parameters: GenerationParameters,
        rng: inout some RandomNumberGenerator,
        tokenHistory: [Int] = []
    ) -> Int {
        sample(
            values: decodeF32(logits), parameters: parameters,
            rng: &rng, tokenHistory: tokenHistory)
    }

    /// Same pipeline as `sample(_:parameters:rng:tokenHistory:)` but
    /// takes already-decoded f32 values directly, skipping the Tensor
    /// round-trip. Used by `Generate.swift`'s
    /// `BUTTER_LMHEAD_PRUNE_SAMPLE_CANDIDATES=1` path (T40 follow-up),
    /// which needs to feed pre-masked values (non-candidate rows forced to
    /// `-infinity` by `LagunaLmHeadPrune.maskNonCandidates`) through the
    /// same top-K/top-P/min-P/softmax pipeline.
    public static func sample(
        values rawValues: [Float],
        parameters: GenerationParameters,
        rng: inout some RandomNumberGenerator,
        tokenHistory: [Int] = []
    ) -> Int {
        var values = rawValues
        let vocab = values.count

        // (1) Repetition penalty
        if parameters.repetitionPenalty != 1.0 && !tokenHistory.isEmpty {
            let p = parameters.repetitionPenalty
            var seen = Set<Int>()
            for t in tokenHistory where t >= 0 && t < vocab {
                seen.insert(t)
            }
            for i in seen {
                let v = values[i]
                values[i] = v > 0 ? v / p : v * p
            }
        }

        // Greedy short-circuit. Skips the softmax / RNG draw entirely.
        if parameters.temperature == 0 {
            return argmaxOf(values)
        }

        // (2)-(5) temperature scale + top-K/top-P/min-P filtering, then
        // (6) softmax + categorical sample — factored into
        // `buildFilteredWeights` / `sampleFromWeights` so the MTP
        // rejection-sampling path (`MTPSpecDecode.swift`) can build the
        // identical p_j / q_j distributions without a second hand-rolled
        // pipeline. See `FilteredDistribution` below.
        let (indices, weights, total) = buildFilteredWeights(values, parameters: parameters)
        if indices.isEmpty {
            // Defensive: every filter rejected everything. Fall back to
            // the unfiltered argmax so we still emit something sensible.
            return argmaxOf(values)
        }
        return sampleFromWeights(indices: indices, weights: weights, total: total, rng: &rng)
    }

    // ─── Shared filter pipeline (temp / top-K / top-P / min-P) ───────

    /// Steps (2)-(5) of the pipeline above: temperature scale, then
    /// top-K / top-P / min-P filtering, returning the surviving
    /// `(vocab index, unnormalized softmax weight)` pairs plus their
    /// sum. Pulled out of `sample(values:...)` verbatim (byte-identical
    /// control flow — same filter order, same edge-case fallbacks) so
    /// both the plain sampler and `FilteredDistribution` (MTP's
    /// rejection-sampling p_j/q_j) share one pipeline. Repetition
    /// penalty (step 1) is NOT applied here — it needs the evolving
    /// token history, which has no well-defined meaning mid-draft-burst
    /// (drafted tokens aren't committed to history yet); callers that
    /// need it apply it to `rawValues` before calling in (as
    /// `sample(values:...)` does above).
    ///
    /// Caller contract: `parameters.temperature != 0` (the greedy
    /// short-circuit is the caller's job, same as `sample` above).
    ///
    /// T43 perf note: the old code unconditionally did a full
    /// `O(vocab log vocab)` sort of the entire logits row whenever
    /// top-K or top-P was active — ~9ms/token pure CPU at Qwen3's
    /// 248,320-row vocab, and the MTP rejection-sampling path
    /// (`SpecDecode.planMTPAcceptSampled`) called this up to `2*gamma+1`
    /// times per cycle, making the "sampled MTP" path SLOWER than plain
    /// sampled decode. When top-K is active (every shipped family
    /// default that sets `topP < 1` also sets a finite `topK` — see
    /// `LlamaText`/`Gemma*Text`/`DeepSeekV4`/`Laguna`/`Step3`/etc. — and
    /// this file's bench gates use `topK: 20`), `selectTopKSorted` below
    /// finds the K best candidates via a bounded min-heap in
    /// `O(vocab log K)` instead, then sorts only those K (K is
    /// typically ≤64, so that inner sort is noise). Its output is
    /// byte-identical in SET, VALUES, and ORDER to the old
    /// sort-then-truncate-to-K path — see its doc comment for the
    /// tie-break argument — which matters beyond correctness: the
    /// categorical draw in `sampleFromWeights` accumulates over
    /// `indices`/`weights` IN ORDER, so a reordering would change which
    /// token a given RNG draw lands on and break seeded-sample
    /// reproducibility, not just the marginal distribution.
    ///
    /// Pure top-P with top-K disabled (`topK == 0` or `topK >= vocab`)
    /// keeps the full-sort path below unchanged — no shipped family
    /// default pairs `topP < 1` with `topK == 0`, so a quickselect
    /// rewrite for that combination isn't worth the risk/time here (see
    /// the T43 task brief); it remains exactly the pre-T43 code.
    static func buildFilteredWeights(
        _ rawValues: [Float], parameters: GenerationParameters
    ) -> (indices: [Int], weights: [Double], total: Double) {
        var values = rawValues
        let vocab = values.count

        // (2) Temperature
        if parameters.temperature != 1.0 {
            let invT = 1.0 / parameters.temperature
            for i in 0 ..< vocab { values[i] *= invT }
        }

        // (3) Top-K + initial candidate set.
        var indexed: [(idx: Int, logit: Float)]
        if parameters.topK > 0 && parameters.topK < vocab {
            // Fast path: partial selection, already sorted descending
            // (see `selectTopKSorted`'s doc comment for the exact
            // equivalence argument vs. the old full-sort path).
            indexed = selectTopKSorted(values, k: parameters.topK)
        } else {
            // Unmodified pre-T43 path: build the full working list of
            // (logit, index) and sort once if top-P needs the order to
            // walk cumulative softmax. (topK > 0 here only when
            // topK >= vocab — "keep everything" — a degenerate config
            // no shipped default hits; kept byte-identical to before.)
            indexed = []
            indexed.reserveCapacity(vocab)
            for i in 0 ..< vocab { indexed.append((i, values[i])) }

            let needsSort = parameters.topK > 0 || parameters.topP < 1.0
            if needsSort {
                indexed.sort { $0.logit > $1.logit }
            }
            if parameters.topK > 0 && parameters.topK < indexed.count {
                indexed.removeLast(indexed.count - parameters.topK)
            }
        }

        // (4) Top-P — keep until cumulative softmax ≥ topP
        if parameters.topP < 1.0 && !indexed.isEmpty {
            // Stable softmax over the (sorted) candidates.
            let maxL = indexed.first?.logit ?? 0
            var cum = 0.0
            var sumExp = 0.0
            for x in indexed { sumExp += Double(expf(x.logit - maxL)) }
            var cutoff = indexed.count
            for i in 0 ..< indexed.count {
                cum += Double(expf(indexed[i].logit - maxL)) / sumExp
                if cum >= Double(parameters.topP) {
                    cutoff = i + 1
                    break
                }
            }
            indexed.removeLast(indexed.count - cutoff)
        }

        // (5) Min-P — keep tokens with prob ≥ min_p × max_prob.
        // exp(logit - maxL) is the unnormalized prob ratio; threshold
        // becomes logit ≥ maxL + log(min_p).
        if parameters.minP > 0 && !indexed.isEmpty {
            let maxL = indexed.first?.logit ?? indexed.map(\.logit).max() ?? 0
            let threshLogit = maxL + logf(parameters.minP)
            indexed = indexed.filter { $0.logit >= threshLogit }
        }

        if indexed.isEmpty { return ([], [], 0) }

        // (6) Softmax weights (unnormalized)
        let maxL = indexed.map(\.logit).max() ?? 0
        var weights = [Double](repeating: 0, count: indexed.count)
        var total = 0.0
        for i in 0 ..< indexed.count {
            let w = Double(expf(indexed[i].logit - maxL))
            weights[i] = w
            total += w
        }
        return (indexed.map(\.idx), weights, total)
    }

    /// Categorical sample over `(indices, weights, total)` from
    /// `buildFilteredWeights` — the tail of `sample(values:...)`'s step
    /// (6), extracted so `FilteredDistribution.sample` can reuse it.
    static func sampleFromWeights(
        indices: [Int], weights: [Double], total: Double,
        rng: inout some RandomNumberGenerator
    ) -> Int {
        guard total > 0, !indices.isEmpty else { return 0 }
        let u = Double.random(in: 0 ..< 1, using: &rng) * total
        var acc = 0.0
        for i in 0 ..< indices.count {
            acc += weights[i]
            if acc >= u { return indices[i] }
        }
        return indices[indices.count - 1]
    }

    /// Categorical sample over a dense `[vocab]` probability vector
    /// (many zero entries, e.g. a rejection-sampling residual
    /// distribution). O(vocab) linear scan — used at most once per MTP
    /// spec-decode cycle (only on a reject or a full-accept bonus), not
    /// per draft step.
    static func sampleFromVector(_ probs: [Double], rng: inout some RandomNumberGenerator) -> Int {
        let total = probs.reduce(0, +)
        guard total > 0 else { return 0 }
        let u = Double.random(in: 0 ..< 1, using: &rng) * total
        var acc = 0.0
        for i in 0 ..< probs.count {
            acc += probs[i]
            if acc >= u { return i }
        }
        return probs.count - 1
    }

    // ─── Rejection sampling primitives (lossless spec decode) ────────
    //
    // Standard speculative-sampling accept/residual math (Leviathan et
    // al.): given the drafter's distribution q and the target's
    // distribution p over the SAME token, accept the drafted token x
    // with probability min(1, p(x)/q(x)); on rejection, resample from
    // the residual r = normalize(max(0, p - q)). This is lossless: the
    // marginal distribution of the emitted token is exactly p,
    // regardless of q. Pure/deterministic given their inputs — no
    // RandomNumberGenerator, no logits — so they're unit-testable
    // directly against hand-computed values
    // (Tests/ButterTests/Generation/MTPSpecDecodeTests.swift).

    /// `min(1, p/q)` — the standard speculative-sampling accept
    /// probability for one token. `q <= 0` only happens when the
    /// drafted token was filtered out of q's own support (shouldn't
    /// happen since the token was SAMPLED from q — defensive only):
    /// reject unconditionally rather than divide by zero.
    public static func acceptProbability(p: Double, q: Double) -> Double {
        guard q > 0 else { return 0 }
        return min(1.0, p / q)
    }

    /// `normalize(max(0, p - q))` elementwise over two same-length
    /// distributions. All-zero input (p == q everywhere) returns an
    /// all-zero vector — callers should not hit this in practice (a
    /// rejection implies p(x) < q(x) at the drafted token, so p != q).
    public static func residual(p: [Double], q: [Double]) -> [Double] {
        precondition(p.count == q.count, "Sampling.residual: p/q length mismatch")
        var r = [Double](repeating: 0, count: p.count)
        var total = 0.0
        for i in 0 ..< p.count {
            let v = max(0, p[i] - q[i])
            r[i] = v
            total += v
        }
        if total > 0 {
            for i in 0 ..< r.count { r[i] /= total }
        }
        return r
    }

    /// One row's filtered-and-softmaxed distribution (temp + top-K/P +
    /// min-P, same pipeline as `sample(values:...)`), materialized so
    /// MTP rejection sampling can both draw from it (the draft step's
    /// q_j) and query specific probabilities against it (the accept
    /// step's p_j(x)/q_j(x)) without recomputing the filter pipeline
    /// twice. Cheap to build (one `buildFilteredWeights` call); does
    /// NOT eagerly materialize a dense `[vocab]` vector — see
    /// `Sampling.residual` / `denseVector(vocab:)` for that, needed
    /// only on a reject or a full-accept bonus draw.
    public struct FilteredDistribution {
        public let indices: [Int]
        public let weights: [Double]
        public let total: Double

        public init(rawValues: [Float], parameters: GenerationParameters) {
            (indices, weights, total) = Sampling.buildFilteredWeights(
                rawValues, parameters: parameters)
        }

        /// Probability of `token` under this distribution, 0 if it was
        /// filtered out. Linear scan over the surviving candidates —
        /// called at most a couple of times per row (the drafted
        /// token's p and q), not per-vocab-entry, so a dictionary
        /// build would cost more than it saves here.
        public func probability(of token: Int) -> Double {
            guard total > 0 else { return 0 }
            guard let pos = indices.firstIndex(of: token) else { return 0 }
            return weights[pos] / total
        }

        /// Categorical sample from this distribution.
        public func sample(_ rng: inout some RandomNumberGenerator) -> Int {
            Sampling.sampleFromWeights(indices: indices, weights: weights, total: total, rng: &rng)
        }

        /// Dense `[vocab]` probability vector (0 outside the surviving
        /// support) — only needed to build a rejection residual against
        /// another distribution's dense vector.
        public func denseVector(vocab: Int) -> [Double] {
            var v = [Double](repeating: 0, count: vocab)
            guard total > 0 else { return v }
            for (i, idx) in indices.enumerated() { v[idx] = weights[i] / total }
            return v
        }
    }

    // ─── Partial top-K selection (T43 hot-loop fix) ──────────────────

    /// Select the `k` best `(index, logit)` pairs out of `values` and
    /// return them SORTED descending — set, values, and order
    /// byte-identical to
    /// `values.enumerated().sorted { $0.1 > $1.1 }.prefix(k)`, but
    /// `O(n log k)` via a bounded min-heap instead of `O(n log n)`.
    ///
    /// Equivalence argument: Swift's `sort(by:)` is a STABLE sort (Swift
    /// 5+ guarantee), so `indexed.sort { $0.logit > $1.logit }` orders
    /// by logit descending and, among ties, preserves the elements'
    /// original relative order — i.e. ascending index. That's a total,
    /// unambiguous order: `(logit desc, index asc)`. `worse(a, b)` below
    /// encodes exactly that same total order (a sorts after b), so the
    /// heap keeps the true top-K under it, and the final `sort` (over
    /// only the k survivors, using the identical total order rather
    /// than relying on stability) reproduces the full-sort-then-`prefix`
    /// result exactly. No ties are possible in the final comparator
    /// (index is unique), so this order is deterministic regardless of
    /// heap internals.
    ///
    /// Kept `private` — the hot path always goes through
    /// `buildFilteredWeights`. `SamplingFilterEquivalenceTests` verifies
    /// this against the old always-sort implementation
    /// (`buildFilteredWeightsFullSort`, kept below for exactly that
    /// purpose) across randomized rows and parameter combinations.
    private static func selectTopKSorted(
        _ values: [Float], k: Int
    ) -> [(idx: Int, logit: Float)] {
        typealias Cand = (idx: Int, logit: Float)

        // a is "worse" than b iff a would sort strictly after b under
        // (logit desc, index asc) — i.e. b is the one a stable full
        // sort would keep first.
        func worse(_ a: Cand, _ b: Cand) -> Bool {
            if a.logit != b.logit { return a.logit < b.logit }
            return a.idx > b.idx
        }

        var heap: [Cand] = []
        heap.reserveCapacity(k)

        // Standard bounded min-heap (worst-of-the-kept at the root)
        // keyed by `worse`, so we can cheaply test/evict against the
        // current worst survivor as new candidates stream in.
        func siftUp(_ start: Int) {
            var i = start
            while i > 0 {
                let parent = (i - 1) / 2
                if worse(heap[i], heap[parent]) {
                    heap.swapAt(i, parent)
                    i = parent
                } else {
                    break
                }
            }
        }
        func siftDown(_ start: Int) {
            var i = start
            let n = heap.count
            while true {
                var worst = i
                let left = 2 * i + 1
                let right = 2 * i + 2
                if left < n && worse(heap[left], heap[worst]) { worst = left }
                if right < n && worse(heap[right], heap[worst]) { worst = right }
                if worst == i { break }
                heap.swapAt(i, worst)
                i = worst
            }
        }

        for i in 0 ..< values.count {
            let cand: Cand = (i, values[i])
            if heap.count < k {
                heap.append(cand)
                siftUp(heap.count - 1)
            } else if worse(heap[0], cand) {
                // Root is the current worst-of-the-kept; the new
                // candidate beats it, so evict.
                heap[0] = cand
                siftDown(0)
            }
        }

        heap.sort { a, b in
            if a.logit != b.logit { return a.logit > b.logit }
            return a.idx < b.idx
        }
        return heap
    }

    /// Pre-T43 reference implementation: full `O(vocab log vocab)` sort
    /// whenever top-K or top-P is active, unconditionally. Kept ONLY as
    /// the equivalence test's ground truth for `buildFilteredWeights`'s
    /// fast top-K path — not called from any production path. See
    /// `SamplingFilterEquivalenceTests`.
    static func buildFilteredWeightsFullSort(
        _ rawValues: [Float], parameters: GenerationParameters
    ) -> (indices: [Int], weights: [Double], total: Double) {
        var values = rawValues
        let vocab = values.count

        if parameters.temperature != 1.0 {
            let invT = 1.0 / parameters.temperature
            for i in 0 ..< vocab { values[i] *= invT }
        }

        var indexed: [(idx: Int, logit: Float)] = []
        indexed.reserveCapacity(vocab)
        for i in 0 ..< vocab { indexed.append((i, values[i])) }

        let needsSort = parameters.topK > 0 || parameters.topP < 1.0
        if needsSort {
            indexed.sort { $0.logit > $1.logit }
        }

        if parameters.topK > 0 && parameters.topK < indexed.count {
            indexed.removeLast(indexed.count - parameters.topK)
        }

        if parameters.topP < 1.0 && !indexed.isEmpty {
            let maxL = indexed.first?.logit ?? 0
            var cum = 0.0
            var sumExp = 0.0
            for x in indexed { sumExp += Double(expf(x.logit - maxL)) }
            var cutoff = indexed.count
            for i in 0 ..< indexed.count {
                cum += Double(expf(indexed[i].logit - maxL)) / sumExp
                if cum >= Double(parameters.topP) {
                    cutoff = i + 1
                    break
                }
            }
            indexed.removeLast(indexed.count - cutoff)
        }

        if parameters.minP > 0 && !indexed.isEmpty {
            let maxL = indexed.first?.logit ?? indexed.map(\.logit).max() ?? 0
            let threshLogit = maxL + logf(parameters.minP)
            indexed = indexed.filter { $0.logit >= threshLogit }
        }

        if indexed.isEmpty { return ([], [], 0) }

        let maxL = indexed.map(\.logit).max() ?? 0
        var weights = [Double](repeating: 0, count: indexed.count)
        var total = 0.0
        for i in 0 ..< indexed.count {
            let w = Double(expf(indexed[i].logit - maxL))
            weights[i] = w
            total += w
        }
        return (indexed.map(\.idx), weights, total)
    }

    // ─── Internal helpers ────────────────────────────────────────────

    /// Decode logits to a fp32 Swift array. Handles fp32 / fp16 / bf16.
    public static func decodeF32(_ logits: Tensor) -> [Float] {
        let n = logits.elementCount
        var out = [Float](repeating: 0, count: n)
        switch logits.dtype {
        case .f32:
            let arr = logits.toArray(as: Float.self)
            for i in 0 ..< n { out[i] = arr[i] }
        case .f16:
            let arr = logits.toArray(as: Float16.self)
            for i in 0 ..< n { out[i] = Float(arr[i]) }
        case .bf16:
            let bits = logits.toArray(as: UInt16.self)
            for i in 0 ..< n { out[i] = Float(bitPattern: UInt32(bits[i]) << 16) }
        default:
            fatalError("Sampling: unsupported logits dtype \(logits.dtype)")
        }
        return out
    }

    private static func argmaxOf(_ values: [Float]) -> Int {
        var best = 0
        var bestVal = values[0]
        for i in 1 ..< values.count where values[i] > bestVal {
            bestVal = values[i]
            best = i
        }
        return best
    }
}

// MARK: - Seeded RNG

/// SplitMix64 PRNG — same seed always produces the same sequence.
/// Used by `GenerationParameters.makeRNG()` when `seed != nil` so
/// non-greedy sampling is reproducible run-to-run.
public struct SeededRandomNumberGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { self.state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Type-erased RNG so callers can hand around a concrete `RandomNumberGenerator`
/// without leaking whether it's seeded vs system-random.
public struct AnyRandomNumberGenerator: RandomNumberGenerator {
    private var fill: () -> UInt64
    public mutating func next() -> UInt64 { fill() }

    public init<R: RandomNumberGenerator>(_ wrapped: R) {
        var w = wrapped
        self.fill = { w.next() }
    }
}

extension GenerationParameters {
    /// Build an RNG for this generation: seeded + deterministic when
    /// `seed != nil`, system random otherwise.
    public func makeRNG() -> AnyRandomNumberGenerator {
        if let s = seed {
            return AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: s))
        }
        return AnyRandomNumberGenerator(SystemRandomNumberGenerator())
    }
}
