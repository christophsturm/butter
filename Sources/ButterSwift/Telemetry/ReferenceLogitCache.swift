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
// ReferenceLogitCache — two-phase KLD so the full-precision reference is
// run exactly once (design §6.3).
//
// KLD against a bf16 reference is the trustworthy quant-fidelity number,
// but co-running reference + candidate live is infeasible (two 80 GB-class
// models resident at once) and wasteful (the reference is re-run for every
// quant you compare). Instead:
//
//   Phase A  run the reference once → dump per-position full-vocab
//            log-probs to disk (f16, to halve the file).
//   Phase B  score each candidate against the cached file — no reference
//            model in memory.
//
// The CLI surfaces this as `--save-ref-logits <file>` (phase A) then
// `--ref-logits <file>` (phase B); see BenchCommand.
//
// ─── File layout ─────────────────────────────────────────────────────
//   header: 8 × UInt32, little-endian
//     [magic, version, vocab, positions, contextWindow, tokenCount, 0, 0]
//   body:   positions × vocab × UInt16   (Float16 bits of log p(v))
//
// Rows are written in WindowPlan order, so a candidate that replays the
// identical (tokenCount, contextWindow) plan visits the same scored
// positions in the same order — no per-row position index needed.
//
// ─── Size warning ────────────────────────────────────────────────────
// Full-vocab f16 is large: 2047 positions × 150K vocab × 2 B ≈ 600 MB for
// a 2K-token WikiText-2 cap. That's the cost of an apples-to-apples KLD
// (we need the whole reference distribution, not just the top-k). Cap the
// corpus with --wikitext2-max-tokens if disk is tight.

import Foundation

public enum ReferenceLogitCache {
    static let magic: UInt32 = 0x4646_414B  // "FFAK"
    static let version: UInt32 = 1
    private static let headerWords = 8
    private static var headerBytes: Int { headerWords * 4 }

    public enum CacheError: Error, CustomStringConvertible {
        case createFailed(URL)
        case truncatedHeader(URL)
        case badMagic(URL)
        case versionMismatch(URL, found: UInt32)
        case planMismatch(URL, expectedPositions: Int, found: Int)
        case truncatedBody(URL, position: Int)

        public var description: String {
            switch self {
            case .createFailed(let u):
                return "ReferenceLogitCache: could not create \(u.path)"
            case .truncatedHeader(let u):
                return "ReferenceLogitCache: \(u.path) is too short to hold a header"
            case .badMagic(let u):
                return "ReferenceLogitCache: \(u.path) is not a reference-logit cache (bad magic)"
            case .versionMismatch(let u, let v):
                return
                    "ReferenceLogitCache: \(u.path) version \(v) != supported \(ReferenceLogitCache.version)"
            case .planMismatch(let u, let e, let f):
                return
                    "ReferenceLogitCache: \(u.path) has \(f) positions but the candidate's window plan scores \(e) — same --model tokenization, --wikitext2-max-tokens, and --wikitext2-context required across phases"
            case .truncatedBody(let u, let p):
                return "ReferenceLogitCache: \(u.path) ran out of rows at position \(p)"
            }
        }
    }

    // MARK: - Phase A — dump the reference distribution

    /// Run `reference` over `tokens` (windowed identically to
    /// `Perplexity.klDivergence`) and write its per-position full-vocab
    /// log-probs to `url` as Float16. Returns the number of scored
    /// positions written.
    @discardableResult
    public static func write(
        reference: some QualityScorable,
        tokens: [Int],
        contextWindow: Int = 0,
        to url: URL,
        device: Device = .shared
    ) throws -> Int {
        precondition(tokens.count >= 2, "ReferenceLogitCache.write requires at least 2 tokens")
        let plan = Perplexity.WindowPlan.plan(
            tokenCount: tokens.count, contextWindow: contextWindow)

        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw CacheError.createFailed(url)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }

        // Reserve the header; patch it once vocab + count are known.
        try handle.write(contentsOf: Data(count: headerBytes))

        var vocab = 0
        var positions = 0
        for window in plan {
            let caches = reference.makeScoringCaches(device: device)
            for t in 0 ..< (window.length - 1) {
                let logits = reference.scoringForward(
                    tokenId: tokens[window.start + t], position: t,
                    caches: caches, device: device)
                guard t >= window.scoredLocalStart else { continue }
                let logp = Perplexity.decodeLogSoftmax(logits: logits)
                if vocab == 0 { vocab = logp.count }
                try handle.write(contentsOf: f16Row(logp))
                positions += 1
            }
        }

        try handle.seek(toOffset: 0)
        try handle.write(
            contentsOf: header(
                vocab: vocab, positions: positions,
                contextWindow: contextWindow, tokenCount: tokens.count))
        return positions
    }

    // MARK: - Phase B — score a candidate against the cached reference

    /// KL(reference || candidate) reading the reference distribution from
    /// `referenceFile` (phase A) and running only `candidate` live. The
    /// candidate replays the identical window plan, so its scored
    /// positions align row-for-row with the cache.
    public static func klDivergence(
        candidate: some QualityScorable,
        tokens: [Int],
        referenceFile url: URL,
        contextWindow: Int = 0,
        device: Device = .shared
    ) throws -> Perplexity.KLDResult {
        precondition(tokens.count >= 2, "klDivergence requires at least 2 tokens")
        let plan = Perplexity.WindowPlan.plan(
            tokenCount: tokens.count, contextWindow: contextWindow)
        let expected = plan.reduce(0) { $0 + (($1.length - 1) - $1.scoredLocalStart) }

        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let head = try readHeader(handle, url: url)
        guard head.positions == expected else {
            throw CacheError.planMismatch(url, expectedPositions: expected, found: head.positions)
        }
        let vocab = head.vocab
        let rowBytes = vocab * 2

        var klSamples: [Double] = []
        var agreements = 0
        var scored = 0
        for window in plan {
            let caches = candidate.makeScoringCaches(device: device)
            for t in 0 ..< (window.length - 1) {
                let candLogits = candidate.scoringForward(
                    tokenId: tokens[window.start + t], position: t,
                    caches: caches, device: device)
                guard t >= window.scoredLocalStart else { continue }
                guard let rowData = try handle.read(upToCount: rowBytes), rowData.count == rowBytes
                else { throw CacheError.truncatedBody(url, position: scored) }
                // Vocab mismatch is unscorable — KL needs aligned vocabs.
                guard (candLogits.shape.last ?? 0) == vocab else { continue }
                let refLogp = decodeF16Row(rowData, vocab: vocab)
                let candLogp = Perplexity.decodeLogSoftmax(logits: candLogits)
                var kl = 0.0
                var refArg = 0
                var refBest = -Double.infinity
                for v in 0 ..< vocab {
                    let p = exp(refLogp[v])
                    kl += p * (refLogp[v] - candLogp[v])
                    if refLogp[v] > refBest { refBest = refLogp[v]; refArg = v }
                }
                klSamples.append(kl)
                if refArg == Perplexity.argmaxLogits(candLogits) { agreements += 1 }
                scored += 1
            }
        }
        let dist = Perplexity.KLDistribution.from(
            klSamples: klSamples, agreements: agreements, scored: scored)
        return Perplexity.KLDResult(
            meanKLDivergence: dist.mean, scoredTokens: scored, distribution: dist)
    }

    // MARK: - Encoding helpers

    private static func header(vocab: Int, positions: Int, contextWindow: Int, tokenCount: Int)
        -> Data
    {
        let words: [UInt32] = [
            magic, version, UInt32(vocab), UInt32(positions),
            UInt32(max(contextWindow, 0)), UInt32(tokenCount), 0, 0,
        ]
        return words.withUnsafeBytes { Data(bytes: $0.baseAddress!, count: $0.count) }
    }

    private struct Header {
        let vocab: Int
        let positions: Int
        let contextWindow: Int
        let tokenCount: Int
    }

    private static func readHeader(_ handle: FileHandle, url: URL) throws -> Header {
        try handle.seek(toOffset: 0)
        guard let data = try handle.read(upToCount: headerBytes), data.count == headerBytes else {
            throw CacheError.truncatedHeader(url)
        }
        let words = data.withUnsafeBytes { raw -> [UInt32] in
            Array(raw.bindMemory(to: UInt32.self))
        }
        guard words[0] == magic else { throw CacheError.badMagic(url) }
        guard words[1] == version else { throw CacheError.versionMismatch(url, found: words[1]) }
        return Header(
            vocab: Int(words[2]), positions: Int(words[3]),
            contextWindow: Int(words[4]), tokenCount: Int(words[5]))
    }

    /// fp32/fp64 log-probs → Float16 bit row.
    private static func f16Row(_ logp: [Double]) -> Data {
        var bits = [UInt16](repeating: 0, count: logp.count)
        for i in 0 ..< logp.count { bits[i] = Float16(Float(logp[i])).bitPattern }
        return bits.withUnsafeBytes { Data(bytes: $0.baseAddress!, count: $0.count) }
    }

    /// Float16 bit row → fp64 log-probs.
    private static func decodeF16Row(_ data: Data, vocab: Int) -> [Double] {
        var out = [Double](repeating: 0, count: vocab)
        data.withUnsafeBytes { raw in
            let p = raw.bindMemory(to: UInt16.self)
            for v in 0 ..< vocab { out[v] = Double(Float(Float16(bitPattern: p[v]))) }
        }
        return out
    }
}
