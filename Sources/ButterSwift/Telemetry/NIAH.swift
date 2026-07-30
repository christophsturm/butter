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
// NIAH — needle-in-a-haystack retrieval, the long-context quality eval
// (design §5/§8). A single "needle" fact is buried at a known depth inside
// a long filler "haystack", then the model is asked to recall it. Run
// across a grid of (context length × needle depth); report the recall
// accuracy. A quant or a RoPE-extrapolation bug that silently breaks long
// context shows up here as recall collapsing at deeper positions / longer
// contexts even when short-context PPL looks fine.
//
// Rides the same `QualityScorable` contract as PPL/KLD: the answer is
// produced by greedy argmax over `scoringForward` logits — no sampler, no
// per-family code. The `Model` is taken whole only for its tokenizer
// (build the haystack, decode the answer) and EOS set; all scoring goes
// through the protocol primitives so every family is evaluated identically.

import Foundation

public enum NIAH {
    /// The fact to bury + the question that retrieves it. `expected` is the
    /// substring the model's answer must contain to count as a recall. The
    /// passphrase is deliberately a coined, low-frequency string so a model
    /// can only produce it by reading it from context, not from priors.
    public struct Needle: Sendable {
        public let fact: String
        public let question: String
        public let expected: String

        public init(fact: String, question: String, expected: String) {
            self.fact = fact
            self.question = question
            self.expected = expected
        }
    }

    public static let defaultNeedle = Needle(
        fact:
            "Important note: the secret passphrase for the Waffle House vault is velvet-otter-3175. Remember it.",
        question: "What is the secret passphrase for the Waffle House vault?",
        expected: "velvet-otter-3175")

    /// One cell of the grid.
    public struct Trial: Sendable {
        public let contextTokens: Int
        public let depthFraction: Double
        public let passed: Bool
        public let answerPreview: String
    }

    public struct Result: Sendable {
        public let trials: [Trial]
        public var passed: Int { trials.filter(\.passed).count }
        public var total: Int { trials.count }
        public var accuracy: Double {
            total > 0 ? Double(passed) / Double(total) : 0
        }

        /// Compact one-line summary for the bench report's preview column.
        public var summaryLine: String {
            String(format: "NIAH acc=%.1f%% (%d/%d)", accuracy * 100, passed, total)
        }
    }

    // Neutral, self-authored filler — repeated to pad the haystack. Kept
    // bland on purpose so the needle is the only distinctive content.
    private static let fillerSentence =
        "The morning fog rolled over the quiet harbor as the tide moved slowly against the wooden pier. "
        + "Gulls drifted in wide circles above the grey water, and the air carried the faint smell of salt and rope. "
        + "A small boat rocked gently at its mooring while the town behind it stayed asleep under a pale sky. "

    /// Run the grid. `contextLengths` are clamped to the model's max
    /// sequence so we never request a window the cache can't hold.
    public static func run(
        model: Model,
        contextLengths: [Int] = [1024, 2048, 4096],
        depths: [Double] = [0.0, 0.25, 0.5, 0.75, 1.0],
        needle: Needle = defaultNeedle,
        maxAnswerTokens: Int = 24,
        device: Device = .shared
    ) -> Result {
        let maxSeq = model.textEngine.maxContextWindow
        let lengths =
            contextLengths
            .map { Swift.min($0, maxSeq) }
            .filter { $0 > 0 }
            .reduce(into: [Int]()) { acc, l in if !acc.contains(l) { acc.append(l) } }

        let fillerTokens = model.textTokenizer.encode(text: fillerSentence)
        let needleTokens = model.textTokenizer.encode(text: " " + needle.fact + " ")
        let questionTokens = model.textTokenizer.encode(
            text: "\n\nQuestion: " + needle.question + "\nAnswer:")
        let bos: Int? =
            (model.textEngine.requiresLeadingBOS)
            ? (model.textTokenizer.bosTokenId ?? model.config.bosTokenId) : nil

        var trials: [Trial] = []
        for length in lengths {
            for depth in depths {
                let prompt = buildPrompt(
                    targetTokens: length, depth: depth,
                    filler: fillerTokens, needle: needleTokens,
                    question: questionTokens, bos: bos)
                let answer = greedyAnswer(
                    model: model, prompt: prompt,
                    maxAnswerTokens: maxAnswerTokens, device: device)
                let passed = normalized(answer).contains(normalized(needle.expected))
                trials.append(
                    Trial(
                        contextTokens: prompt.count, depthFraction: depth,
                        passed: passed,
                        answerPreview: String(answer.prefix(80))
                            .replacingOccurrences(of: "\n", with: " ")))
                Debug.log(
                    .bench,
                    "niah: len=\(prompt.count) depth=\(depth) passed=\(passed)")
            }
        }
        return Result(trials: trials)
    }

    // MARK: - Prompt construction

    /// Assemble `[bos?] + fillerHead + needle + fillerTail + question`,
    /// padding the filler (repeating its base tokens) so the whole prompt
    /// lands near `targetTokens`. `depth` places the needle as a fraction
    /// of the filler budget (0 = very start, 1 = just before the question).
    private static func buildPrompt(
        targetTokens: Int, depth: Double,
        filler: [Int], needle: [Int], question: [Int], bos: Int?
    ) -> [Int] {
        let overhead = (bos == nil ? 0 : 1) + needle.count + question.count
        let fillerBudget = Swift.max(0, targetTokens - overhead)
        let padded = repeated(filler, toCount: fillerBudget)
        let cut = Swift.min(padded.count, Int((Double(padded.count) * depth).rounded()))

        var prompt: [Int] = []
        prompt.reserveCapacity(targetTokens + needle.count + question.count + 1)
        if let bos { prompt.append(bos) }
        prompt.append(contentsOf: padded[0 ..< cut])
        prompt.append(contentsOf: needle)
        prompt.append(contentsOf: padded[cut ..< padded.count])
        prompt.append(contentsOf: question)
        return prompt
    }

    /// Repeat `base` until reaching `count` tokens (truncating the last
    /// copy). Empty `base` yields an empty array.
    private static func repeated(_ base: [Int], toCount count: Int) -> [Int] {
        guard !base.isEmpty, count > 0 else { return [] }
        var out: [Int] = []
        out.reserveCapacity(count)
        while out.count < count {
            let remaining = count - out.count
            if remaining >= base.count {
                out.append(contentsOf: base)
            } else {
                out.append(contentsOf: base[0 ..< remaining])
            }
        }
        return out
    }

    // MARK: - Greedy continuation over the QualityScorable contract

    /// Force-decode `prompt`, then greedily extend by argmax over
    /// `scoringForward` logits until EOS or `maxAnswerTokens`. Decodes the
    /// generated tokens to text for the recall check.
    private static func greedyAnswer(
        model: some QualityScorable & Tokenizing & EOSProviding,
        prompt: [Int],
        maxAnswerTokens: Int,
        device: Device
    ) -> String {
        guard prompt.count >= 1 else { return "" }
        let caches = model.makeScoringCaches(device: device)
        var logits = model.scoringForward(
            tokenId: prompt[0], position: 0, caches: caches, device: device)
        for t in 1 ..< prompt.count {
            logits = model.scoringForward(
                tokenId: prompt[t], position: t, caches: caches, device: device)
        }
        // `logits` now predicts the first answer token.
        var generated: [Int] = []
        var pos = prompt.count
        for _ in 0 ..< maxAnswerTokens {
            let next = Perplexity.argmaxLogits(logits)
            if model.eosTokenIds.contains(next) { break }
            generated.append(next)
            logits = model.scoringForward(
                tokenId: next, position: pos, caches: caches, device: device)
            pos += 1
        }
        return model.decodeTokens(generated)
    }

    /// Lowercase + collapse whitespace so recall matching is robust to
    /// tokenizer spacing / casing quirks.
    private static func normalized(_ s: String) -> String {
        s.lowercased().split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" })
            .joined(separator: " ")
    }
}

// Small capability protocols so `greedyAnswer` stays on the uniform
// scoring contract while still reaching the tokenizer + EOS set it needs.
// `Model` satisfies all three.
public protocol Tokenizing {
    func decodeTokens(_ tokens: [Int]) -> String
}

public protocol EOSProviding {
    var eosTokenIds: Set<Int> { get }
}

extension Model: Tokenizing {
    public func decodeTokens(_ tokens: [Int]) -> String {
        tokenizer?.decode(tokens: tokens, skipSpecialTokens: true) ?? ""    }
}

extension Model: EOSProviding {
    public var eosTokenIds: Set<Int> { Set(config.eosTokenIds) }
}
