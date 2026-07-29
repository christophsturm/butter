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
// Perplexity — log-likelihood / perplexity / KL-divergence over a
// fixed token sequence.
//
// Standalone helper. Not folded into `generate(...)` because perplexity
// computation requires capturing logits at every step (extra work +
// memory) which would tax the greedy fast path. The bench harness
// calls this directly when `--method wikitext2` runs; the CLI / your
// own code can opt into `--stats` plus a separate
// `Perplexity.compute(model:tokens:)` call.
//
// ─── Why fp32 ────────────────────────────────────────────────────────
// log_softmax computes log(Σ exp(x_i − max_x)) over the full vocab. With
// 128–152K vocab and a peaky distribution, the partial sum lives in
// roughly [1, 10⁵]. bf16's 7 mantissa bits can't represent that range
// without precision loss that pollutes the perplexity 3rd decimal —
// enough to obscure the differences between quantization tiers we're
// usually trying to measure. fp16 has 11 mantissa bits, also marginal.
// fp32 it is. The extra work is ~1µs per token; we only do this
// offline for evaluation, so the cost is irrelevant.
//
// ─── KL divergence (paired-run) ──────────────────────────────────────
// `Perplexity.klDivergence(reference:candidate:tokens:)` runs both
// models forward in lockstep, capturing per-position log-softmax
// distributions and accumulating KL(p_ref || q_cand) per position.
// Both models must share the same vocabulary — the natural pairing is a
// quantized variant against its bf16 unquantized parent. The reference
// should be the unquantized variant if it fits in memory; using a
// smaller reference makes the KL number a measure of family closeness
// rather than quantization fidelity.

import Foundation

public enum Perplexity {
    public struct Result: Sendable, Equatable {
        /// `exp(-mean(log p(token_t | tokens_<t)))`.
        public let perplexity: Double
        /// Mean negative log-likelihood (nats).
        public let meanNegLogLikelihood: Double
        /// Number of scored positions (tokens.count - 1).
        public let scoredTokens: Int
    }

    public struct KLDResult: Sendable, Equatable {
        /// Mean KL(p_ref || q_cand) across scored positions, in nats.
        public let meanKLDivergence: Double
        /// Number of positions scored. Equal to `tokens.count - 1`
        /// minus any positions skipped due to vocab-size mismatch.
        public let scoredTokens: Int
        /// Per-position KLD distribution. Mean hides tail distortion —
        /// exactly where bad quants live — so we surface the spread too.
        /// `nil` only for callers that built a bare
        /// `KLDResult(meanKLDivergence:scoredTokens:)`.
        public let distribution: KLDistribution?

        public init(
            meanKLDivergence: Double,
            scoredTokens: Int,
            distribution: KLDistribution? = nil
        ) {
            self.meanKLDivergence = meanKLDivergence
            self.scoredTokens = scoredTokens
            self.distribution = distribution
        }
    }

    /// The shape of a KLD run, not just its mean. Mean KLD is a poor
    /// "is this quant broken" signal because a handful of catastrophically
    /// distorted positions get averaged away. The tail percentiles + the
    /// top-1 flip rate are the legible ones (design §6.4).
    public struct KLDistribution: Sendable, Equatable {
        /// Mean KL(p_ref || q_cand) across scored positions, in nats.
        public let mean: Double
        /// Median (p50) per-position KLD, in nats.
        public let median: Double
        /// 90th-percentile per-position KLD, in nats.
        public let p90: Double
        /// 99th-percentile per-position KLD, in nats.
        public let p99: Double
        /// Worst single-position KLD, in nats.
        public let max: Double
        /// Fraction of scored positions where `argmax(ref) == argmax(cand)`,
        /// in `[0, 1]`. The most legible quant-health signal: 1.0 means the
        /// quant never changed the model's top pick.
        public let top1Agreement: Double

        public init(
            mean: Double, median: Double, p90: Double, p99: Double,
            max: Double, top1Agreement: Double
        ) {
            self.mean = mean
            self.median = median
            self.p90 = p90
            self.p99 = p99
            self.max = max
            self.top1Agreement = top1Agreement
        }

        /// Summarize per-position samples into the reported distribution.
        /// `agreements` counts positions where the two models' argmax
        /// matched; `scored` is the total scored-position count.
        public static func from(klSamples: [Double], agreements: Int, scored: Int)
            -> KLDistribution
        {
            guard !klSamples.isEmpty else {
                return KLDistribution(
                    mean: 0, median: 0, p90: 0, p99: 0, max: 0,
                    top1Agreement: scored > 0 ? Double(agreements) / Double(scored) : 0)
            }
            let sorted = klSamples.sorted()
            let mean = klSamples.reduce(0, +) / Double(klSamples.count)
            return KLDistribution(
                mean: mean,
                median: Self.percentile(sorted, 0.50),
                p90: Self.percentile(sorted, 0.90),
                p99: Self.percentile(sorted, 0.99),
                max: sorted[sorted.count - 1],
                top1Agreement: scored > 0 ? Double(agreements) / Double(scored) : 0)
        }

        /// Nearest-rank percentile over an already-sorted ascending array.
        /// `q` in `[0, 1]`. No interpolation — fine for a telemetry readout.
        static func percentile(_ sorted: [Double], _ q: Double) -> Double {
            guard !sorted.isEmpty else { return 0 }
            let rank = Int((q * Double(sorted.count - 1)).rounded())
            return sorted[Swift.min(Swift.max(rank, 0), sorted.count - 1)]
        }
    }

    /// Score the sequence against the model. Token at index `t+1` is
    /// the target for the prediction conditioned on tokens `[0..=t]`.
    /// Uses `forward(...)` (not `forwardSample`) so we get logits, then
    /// reads them back to CPU once per step. Cost is one extra readback
    /// per token vs greedy; acceptable for offline evaluation.
    ///
    /// **Context windowing (`contextWindow > 0`).** A single growing cache
    /// over the whole corpus RoPE-extrapolates past `n_ctx` (garbage) and
    /// scores the first tokens with ~no left-context (noise). Pass a fixed
    /// window and the corpus is scored in `n_ctx`-sized windows strided by
    /// `n_ctx/2`, scoring only each window's second half so every scored
    /// token carries ≥ `n_ctx/2` of real left-context, counted once
    /// (design §6.1). `contextWindow <= 0` keeps the legacy single-pass
    /// behaviour.
    public static func compute(
        model: some QualityScorable,
        tokens: [Int],
        contextWindow: Int = 0,
        device: Device = .shared
    ) -> Result {
        precondition(
            tokens.count >= 2,
            "Perplexity.compute requires at least 2 tokens (one context + one target)")
        var nll = 0.0
        var scored = 0
        for window in WindowPlan.plan(tokenCount: tokens.count, contextWindow: contextWindow) {
            let caches = model.makeScoringCaches(device: device)
            for t in 0 ..< (window.length - 1) {
                let logits = model.scoringForward(
                    tokenId: tokens[window.start + t], position: t,
                    caches: caches, device: device)
                guard t >= window.scoredLocalStart else { continue }
                let target = tokens[window.start + t + 1]
                nll += negLogSoftmaxAt(logits: logits, index: target)
                scored += 1
            }
        }
        let mean = scored > 0 ? nll / Double(scored) : 0
        return Result(
            perplexity: exp(mean),
            meanNegLogLikelihood: mean,
            scoredTokens: scored)
    }

    /// KL(reference || candidate) over a fixed token sequence. Both
    /// models must use the same tokenizer / vocab. Runs them in
    /// lockstep — at each position, compute reference probs +
    /// candidate log-probs from full logits, then accumulate
    /// `Σ p_ref(v) * (log p_ref(v) − log q_cand(v))`.
    ///
    /// Memory cost: two simultaneously-loaded models + two KV caches.
    /// Pick the reference and candidate sizes such that both fit in
    /// device memory. The recommended pairing is the bf16 variant of
    /// the same checkpoint as `reference` against the quantized
    /// variant as `candidate`.
    /// `contextWindow` mirrors `compute` (design §6.1) — windowed strided
    /// scoring so the KLD positions carry real left-context. Both models are
    /// driven over the identical window plan so their scored positions align
    /// exactly. The result carries the full per-position distribution +
    /// top-1 agreement, not just the mean (design §6.4).
    public static func klDivergence(
        reference: some QualityScorable,
        candidate: some QualityScorable,
        tokens: [Int],
        contextWindow: Int = 0,
        device: Device = .shared
    ) -> KLDResult {
        precondition(
            tokens.count >= 2,
            "klDivergence requires at least 2 tokens")

        var klSamples: [Double] = []
        var agreements = 0
        var scored = 0
        for window in WindowPlan.plan(tokenCount: tokens.count, contextWindow: contextWindow) {
            let refCaches = reference.makeScoringCaches(device: device)
            let candCaches = candidate.makeScoringCaches(device: device)
            for t in 0 ..< (window.length - 1) {
                let refLogits = reference.scoringForward(
                    tokenId: tokens[window.start + t], position: t,
                    caches: refCaches, device: device)
                let candLogits = candidate.scoringForward(
                    tokenId: tokens[window.start + t], position: t,
                    caches: candCaches, device: device)
                guard t >= window.scoredLocalStart else { continue }
                // Skip positions where vocab sizes mismatch (defensive — KL
                // over heterogeneous vocabs isn't meaningful).
                let refV = refLogits.shape.last ?? 0
                let candV = candLogits.shape.last ?? 0
                guard refV > 0, refV == candV else { continue }
                klSamples.append(klAtPosition(refLogits: refLogits, candLogits: candLogits))
                if argmaxLogits(refLogits) == argmaxLogits(candLogits) { agreements += 1 }
                scored += 1
            }
        }
        let dist = KLDistribution.from(klSamples: klSamples, agreements: agreements, scored: scored)
        return KLDResult(
            meanKLDivergence: dist.mean, scoredTokens: scored, distribution: dist)
    }

    // MARK: - Context windowing

    /// One scoring window. The model is forced-decoded over
    /// `tokens[start ..< start+length]` from a fresh cache; logits at local
    /// position `t >= scoredLocalStart` are accumulated (their target is
    /// `tokens[start + t + 1]`). Local positions below `scoredLocalStart`
    /// are decoded for context only — they were already scored by the
    /// previous overlapping window, so counting each corpus token exactly
    /// once falls out of the plan.
    struct Window: Equatable {
        let start: Int
        let length: Int
        let scoredLocalStart: Int
    }

    enum WindowPlan {
        /// Build the strided window plan (design §6.1). `contextWindow <= 0`
        /// (or ≥ the corpus) collapses to a single full-corpus window scored
        /// from local position 0 — identical to the legacy single-pass loop.
        /// Otherwise: `n_ctx`-sized windows strided by `n_ctx/2`, with each
        /// non-first window scoring only its second half so every scored
        /// token has ≥ `n_ctx/2` of left-context.
        static func plan(tokenCount n: Int, contextWindow: Int) -> [Window] {
            guard n >= 2 else { return [] }
            if contextWindow <= 0 || contextWindow >= n {
                return [Window(start: 0, length: n, scoredLocalStart: 0)]
            }
            let stride = Swift.max(1, contextWindow / 2)
            var out: [Window] = []
            var start = 0
            var first = true
            while start < n - 1 {
                let length = Swift.min(contextWindow, n - start)
                let scoredLocalStart = first ? 0 : (stride - 1)
                // Keep only windows that actually score ≥ 1 position
                // (the tail window can be pure overlap — already counted).
                if scoredLocalStart <= length - 2 {
                    out.append(
                        Window(start: start, length: length, scoredLocalStart: scoredLocalStart))
                }
                first = false
                if start + length >= n { break }
                start += stride
            }
            return out
        }
    }

    // MARK: - Internal: argmax over a logits tensor

    /// Index of the maximum logit (full vocab), dtype-aware. Used for the
    /// top-1-agreement signal and greedy live-capture.
    internal static func argmaxLogits(_ logits: Tensor) -> Int {
        let n = logits.shape.last ?? 0
        guard n > 0 else { return 0 }
        let ptr = logits.buffer.contents().advanced(by: logits.offset)
        var best = 0
        var bestV = -Double.infinity
        switch logits.dtype {
        case .f32:
            let f = ptr.assumingMemoryBound(to: Float.self)
            for i in 0 ..< n where Double(f[i]) > bestV { bestV = Double(f[i]); best = i }
        case .f16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n {
                let v = Double(float16ToFloat32(f[i]))
                if v > bestV { bestV = v; best = i }
            }
        case .bf16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n {
                let v = Double(bfloat16ToFloat32(f[i]))
                if v > bestV { bestV = v; best = i }
            }
        default:
            fatalError("Perplexity: unsupported logits dtype \(logits.dtype)")
        }
        return best
    }

    // MARK: - Internal: log-softmax(logits)[index], in fp32
    // Exposed at internal access for direct test coverage (see
    // PerplexityTests). The public surface remains compute(...) / klDivergence(...).

    internal static func negLogSoftmaxAt(logits: Tensor, index: Int) -> Double {
        let n = logits.shape.last ?? 0
        precondition(index >= 0 && index < n, "target index out of range")
        let ptr = logits.buffer.contents().advanced(by: logits.offset)

        var maxV: Double = -.infinity
        var targetLogit: Double = 0
        switch logits.dtype {
        case .f32:
            let f = ptr.assumingMemoryBound(to: Float.self)
            for i in 0 ..< n {
                let v = Double(f[i])
                if v > maxV { maxV = v }
            }
            targetLogit = Double(f[index])
        case .f16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n {
                let v = Double(float16ToFloat32(f[i]))
                if v > maxV { maxV = v }
            }
            targetLogit = Double(float16ToFloat32(f[index]))
        case .bf16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n {
                let v = Double(bfloat16ToFloat32(f[i]))
                if v > maxV { maxV = v }
            }
            targetLogit = Double(bfloat16ToFloat32(f[index]))
        default:
            fatalError("Perplexity: unsupported logits dtype \(logits.dtype)")
        }

        var sum: Double = 0
        switch logits.dtype {
        case .f32:
            let f = ptr.assumingMemoryBound(to: Float.self)
            for i in 0 ..< n { sum += exp(Double(f[i]) - maxV) }
        case .f16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n { sum += exp(Double(float16ToFloat32(f[i])) - maxV) }
        case .bf16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n { sum += exp(Double(bfloat16ToFloat32(f[i])) - maxV) }
        default:
            fatalError("Perplexity: unsupported logits dtype \(logits.dtype)")
        }
        let logZ = maxV + log(sum)
        return logZ - targetLogit  // = -log p(target)
    }

    /// Per-position KL(p_ref || q_cand). Decodes both logit vectors to
    /// fp32, computes log-softmax of each, then accumulates
    /// `Σ p_ref(v) * (log p_ref(v) − log q_cand(v))`. Internal access
    /// for direct test coverage.
    internal static func klAtPosition(refLogits: Tensor, candLogits: Tensor) -> Double {
        let n = refLogits.shape.last ?? 0
        precondition((candLogits.shape.last ?? 0) == n, "KL: vocab size mismatch")
        let refLog = decodeLogSoftmax(logits: refLogits)
        let candLog = decodeLogSoftmax(logits: candLogits)
        var kl = 0.0
        for v in 0 ..< n {
            let p = exp(refLog[v])
            kl += p * (refLog[v] - candLog[v])
        }
        return kl
    }

    /// Decode logits to a fp32 log-softmax vector (`log p(v)` for all v).
    /// Internal access for direct test coverage.
    internal static func decodeLogSoftmax(logits: Tensor) -> [Double] {
        let n = logits.shape.last ?? 0
        var raw = [Double](repeating: 0, count: n)
        let ptr = logits.buffer.contents().advanced(by: logits.offset)
        switch logits.dtype {
        case .f32:
            let f = ptr.assumingMemoryBound(to: Float.self)
            for i in 0 ..< n { raw[i] = Double(f[i]) }
        case .f16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n { raw[i] = Double(float16ToFloat32(f[i])) }
        case .bf16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n { raw[i] = Double(bfloat16ToFloat32(f[i])) }
        default:
            fatalError("Perplexity: unsupported logits dtype \(logits.dtype)")
        }
        var maxV = -Double.infinity
        for v in raw where v > maxV { maxV = v }
        var sum = 0.0
        for v in raw { sum += exp(v - maxV) }
        let logZ = maxV + log(sum)
        for i in 0 ..< n { raw[i] -= logZ }
        return raw
    }

    private static func float16ToFloat32(_ x: UInt16) -> Float {
        var v = x
        return withUnsafePointer(to: &v) {
            $0.withMemoryRebound(to: Float16.self, capacity: 1) { Float($0.pointee) }
        }
    }

    private static func bfloat16ToFloat32(_ x: UInt16) -> Float {
        var bits = UInt32(x) << 16
        return withUnsafePointer(to: &bits) {
            $0.withMemoryRebound(to: Float.self, capacity: 1) { $0.pointee }
        }
    }
}
