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
// BenchRunner — drives a `BenchMethod` against a loaded model and
// hands the resulting row to a `BenchmarkWriter`.
//
// Implemented methods bottom out in `Model.generate(...)` /
// `Perplexity.compute(...)`. Unimplemented methods (NIAH, multi-turn,
// tool-calling, ngram-*, vision) `throw .notImplemented(...)` with
// the dependency name so callers see exactly what's missing.

import Foundation

public enum BenchRunnerError: Error, CustomStringConvertible {
    case notImplemented(method: BenchMethod, dependency: String)
    case missingPrompt
    case wikitext2CorpusMissing(URL)
    case kldRequiresReferenceModel

    public var description: String {
        switch self {
        case .notImplemented(let m, let dep):
            return
                "ffai bench --method \(m.rawValue): not implemented yet — needs \(dep). Tracked alongside its parent feature in planning/plan.md."
        case .missingPrompt:
            return "ffai bench: --prompt is required for this method"
        case .wikitext2CorpusMissing(let url):
            return
                "ffai bench: WikiText-2 corpus not found at \(url.path). Provide --wikitext2-corpus </path/to/wiki.test.raw>"
        case .kldRequiresReferenceModel:
            return "ffai bench: KLD computation requires --ref-model"
        }
    }
}

public struct BenchOptions: Sendable {
    public var prompt: String?
    public var maxTokens: Int
    public var contextSize: Int?
    public var quantization: String?
    public var wikitext2Corpus: URL?
    public var wikitext2MaxTokens: Int
    /// Sliding `n_ctx` window for PPL/KLD scoring (design §6.1). `0` keeps
    /// the legacy single-pass scoring (every token from position 0 of one
    /// growing cache); `> 0` strides by `window/2` and scores only each
    /// window's second half so scored tokens carry real left-context.
    public var wikitext2ContextWindow: Int
    public var referenceModel: Model?
    /// Phase B of two-phase KLD: score the candidate against a reference
    /// distribution cached on disk by a prior `--save-ref-logits` run,
    /// instead of co-loading a reference model (design §6.3).
    public var referenceLogits: URL?
    /// Phase A of two-phase KLD: dump this model's per-position log-probs
    /// to disk so future candidates can KLD against it without re-running
    /// the reference. Use with the bf16 reference as `--model`.
    public var saveReferenceLogits: URL?

    public init(
        prompt: String? = nil,
        maxTokens: Int = 64,
        contextSize: Int? = nil,
        quantization: String? = nil,
        wikitext2Corpus: URL? = nil,
        wikitext2MaxTokens: Int = 2048,
        wikitext2ContextWindow: Int = 0,
        referenceModel: Model? = nil,
        referenceLogits: URL? = nil,
        saveReferenceLogits: URL? = nil
    ) {
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.contextSize = contextSize
        self.quantization = quantization
        self.wikitext2Corpus = wikitext2Corpus
        self.wikitext2MaxTokens = wikitext2MaxTokens
        self.wikitext2ContextWindow = wikitext2ContextWindow
        self.referenceModel = referenceModel
        self.referenceLogits = referenceLogits
        self.saveReferenceLogits = saveReferenceLogits
    }
}

public struct BenchRunner {
    public let model: Model
    public let modelLabel: String

    public init(model: Model, modelLabel: String) {
        self.model = model
        self.modelLabel = modelLabel
    }

    /// Run `method` against the loaded model, returning a `BenchRow`
    /// the caller can hand to `BenchmarkWriter.append(_:)`.
    public func run(
        method: BenchMethod,
        options: BenchOptions
    ) async throws -> BenchRow {
        Debug.log(.bench, "running method=\(method.rawValue) model=\(modelLabel)")
        switch method {
        case .simple:
            return try await runSimple(options: options)
        case .summarization:
            return try await runSummarization(options: options)
        case .wikitext2:
            return try await runWikiText2(options: options)
        case .niah:
            return try await runNIAH(options: options)
        default:
            throw BenchRunnerError.notImplemented(
                method: method,
                dependency: method.dependency ?? "<see BenchMethod.dependency>"
            )
        }
    }

    // MARK: - Simple — single-prompt generation, throughput + memory

    private func runSimple(options: BenchOptions) async throws -> BenchRow {
        guard let prompt = options.prompt else { throw BenchRunnerError.missingPrompt }
        let params = model.defaultGenerationParameters.with { $0.maxTokens = options.maxTokens }
        let result = try await model.generate(prompt: prompt, parameters: params)

        let preview = String(result.text.prefix(160))
            .replacingOccurrences(of: "\n", with: " ")

        // KLD opt-in: if a reference model is supplied, score the
        // generated continuation under both. Cheap because we only
        // touch the generated tokens, not WikiText-2.
        var kld: Double?
        if let ref = options.referenceModel {
            let kldResult = Perplexity.klDivergence(
                reference: ref, candidate: model,
                tokens: result.promptTokens + result.generatedTokens
            )
            kld = kldResult.meanKLDivergence
        }

        return BenchRow(
            model: modelLabel, method: BenchMethod.simple.rawValue,
            quantization: options.quantization,
            stats: result.stats, outputPreview: preview,
            genPerplexity: nil, genKLDivergence: kld
        )
    }

    // MARK: - Summarization — fixed-shape long-prompt generation

    private func runSummarization(options: BenchOptions) async throws -> BenchRow {
        // For now, a thin wrapper over Simple that just records the
        // method name so the report column reflects it. mlx-swift-lm
        // sweeps multiple context sizes here; we'll add the matrix
        // sweep loop when we ship --ctx <list>. The single-shot
        // shape is already useful for "does this prompt size work".
        guard let prompt = options.prompt else { throw BenchRunnerError.missingPrompt }
        let params = model.defaultGenerationParameters.with { $0.maxTokens = options.maxTokens }
        let result = try await model.generate(prompt: prompt, parameters: params)
        let preview = String(result.text.prefix(160))
            .replacingOccurrences(of: "\n", with: " ")
        return BenchRow(
            model: modelLabel, method: BenchMethod.summarization.rawValue,
            quantization: options.quantization,
            stats: result.stats, outputPreview: preview,
            genPerplexity: nil, genKLDivergence: nil
        )
    }

    // MARK: - WikiText2 — perplexity over a corpus

    private func runWikiText2(options: BenchOptions) async throws -> BenchRow {
        guard let corpus = options.wikitext2Corpus else {
            throw BenchRunnerError.wikitext2CorpusMissing(
                URL(fileURLWithPath: "wiki.test.raw")
            )
        }
        let raw = try String(contentsOf: corpus, encoding: .utf8)
        var tokens = model.textTokenizer.encode(text: raw)
        if tokens.count > options.wikitext2MaxTokens {
            tokens = Array(tokens.prefix(options.wikitext2MaxTokens))
        }
        // BOS handling (design §6.2): prepend the model's <bos> once so the
        // first scored token has a real left-context anchor. Idempotent —
        // skip when the tokenizer already emitted one or the family is not
        // BOS-critical.
        if model.engine.requiresLeadingBOS,
            let bos = model.tokenizer.bosTokenId ?? model.config.bosTokenId,
            tokens.first != bos
        {
            tokens.insert(bos, at: 0)
        }
        let window = options.wikitext2ContextWindow
        Debug.log(
            .bench,
            "wikitext2: scoring \(tokens.count) tokens, contextWindow=\(window == 0 ? "off" : String(window))"
        )

        // Capture a pre-PPL memory snapshot so the report still
        // surfaces realistic numbers (no decode loop populates them
        // for this method otherwise).
        let memTracker = PhaseMemoryTracker()
        let pplResult = Perplexity.compute(
            model: model, tokens: tokens, contextWindow: window)
        memTracker.endPrefill()
        memTracker.endDecode()

        // Phase A: dump this model's reference distribution and stop — no
        // candidate KLD here (this run's --model *is* the reference).
        if let saveURL = options.saveReferenceLogits {
            let n = try ReferenceLogitCache.write(
                reference: model, tokens: tokens,
                contextWindow: window, to: saveURL)
            print("[BENCH] saved \(n) reference log-prob rows → \(saveURL.path)")
        }

        var kld: Double?
        var kldResult: Perplexity.KLDResult?
        if let refLogits = options.referenceLogits {
            // Phase B: cached reference, no second model in memory.
            kldResult = try ReferenceLogitCache.klDivergence(
                candidate: model, tokens: tokens,
                referenceFile: refLogits, contextWindow: window)
        } else if let ref = options.referenceModel {
            // Live paired KLD (both models resident).
            kldResult = Perplexity.klDivergence(
                reference: ref, candidate: model,
                tokens: tokens, contextWindow: window)
        }
        if let r = kldResult {
            kld = r.meanKLDivergence
            printKLDDistribution(r)
        }

        let weightsBytes = model.textEngine.parameters().reduce(0) { $0 + $1.1.byteCount }
        let stats = GenerationStats(
            promptTokens: tokens.count, generatedTokens: 0,
            contextSize: model.textEngine.maxContextWindow,
            prefillTimeS: 0, decodeTimeS: 0, timeToFirstTokenMs: 0,
            steadyTokensPerSecond: nil,
            baselineGPUBytes: memTracker.baseline.gpuBytes,
            postPrefillGPUBytes: memTracker.postPrefill?.gpuBytes ?? 0,
            postDecodeGPUBytes: memTracker.postDecode?.gpuBytes ?? 0,
            prefillPeakGPUBytes: memTracker.prefillPeakBytes,
            decodePeakGPUBytes: memTracker.decodePeakBytes,
            wiredTicketBytes: memTracker.baseline.wiredTicketBytes,
            weightsBytes: weightsBytes,
            kvCacheAllocatedBytes: 0, kvCacheUsedBytes: 0,
            thinkPerplexity: nil, genPerplexity: pplResult.perplexity,
            thinkKLDivergence: nil, genKLDivergence: kld,
            thinkTokenCount: nil, genTokenCount: nil
        )
        return BenchRow(
            model: modelLabel, method: BenchMethod.wikitext2.rawValue,
            quantization: options.quantization,
            stats: stats, outputPreview: nil,
            genPerplexity: pplResult.perplexity,
            genKLDivergence: kld
        )
    }

    // MARK: - NIAH — needle-in-a-haystack retrieval

    private func runNIAH(options: BenchOptions) async throws -> BenchRow {
        // A single --ctx pins one context length; otherwise sweep the
        // default grid (clamped to the model's max sequence by NIAH.run).
        let lengths = options.contextSize.map { [$0] } ?? [1024, 2048, 4096]
        let memTracker = PhaseMemoryTracker()
        let result = NIAH.run(model: model, contextLengths: lengths)
        memTracker.endPrefill()
        memTracker.endDecode()

        print("[BENCH] niah \(modelLabel)")
        print("  ctx-tokens  depth  result  answer")
        for t in result.trials {
            let ctxStr = String(t.contextTokens)
            let ctx = String(repeating: " ", count: max(0, 9 - ctxStr.count)) + ctxStr
            let depth = String(format: "%.2f", t.depthFraction)
            let verdict = t.passed ? "PASS" : "miss"
            print("  \(ctx)  \(depth)  \(verdict)  \(t.answerPreview)")
        }
        print("  \(result.summaryLine)")

        let weightsBytes = model.engine.parameters().reduce(0) { $0 + $1.1.byteCount }
        let stats = GenerationStats(
            promptTokens: result.trials.map(\.contextTokens).max() ?? 0,
            generatedTokens: 0,
            contextSize: model.engine.maxSeq,
            prefillTimeS: 0, decodeTimeS: 0, timeToFirstTokenMs: 0,
            steadyTokensPerSecond: nil,
            baselineGPUBytes: memTracker.baseline.gpuBytes,
            postPrefillGPUBytes: memTracker.postPrefill?.gpuBytes ?? 0,
            postDecodeGPUBytes: memTracker.postDecode?.gpuBytes ?? 0,
            prefillPeakGPUBytes: memTracker.prefillPeakBytes,
            decodePeakGPUBytes: memTracker.decodePeakBytes,
            wiredTicketBytes: memTracker.baseline.wiredTicketBytes,
            weightsBytes: weightsBytes,
            kvCacheAllocatedBytes: 0, kvCacheUsedBytes: 0,
            thinkPerplexity: nil, genPerplexity: nil,
            thinkKLDivergence: nil, genKLDivergence: nil,
            thinkTokenCount: nil, genTokenCount: nil
        )
        return BenchRow(
            model: modelLabel, method: BenchMethod.niah.rawValue,
            quantization: options.quantization,
            stats: stats, outputPreview: result.summaryLine,
            genPerplexity: nil, genKLDivergence: nil
        )
    }

    /// Print the full per-position KLD distribution + top-1 agreement.
    /// Mean alone hides tail distortion — the p99 / max / top-1 flip rate
    /// are the legible "is this quant broken" signals (design §6.4). The
    /// report row still stores the mean for cross-comparability.
    private func printKLDDistribution(_ r: Perplexity.KLDResult) {
        guard let d = r.distribution else { return }
        print("  KLD distribution over \(r.scoredTokens) positions (nats):")
        print(String(format: "    mean   %.4f", d.mean))
        print(String(format: "    median %.4f", d.median))
        print(String(format: "    p90    %.4f", d.p90))
        print(String(format: "    p99    %.4f", d.p99))
        print(String(format: "    max    %.4f", d.max))
        print(String(format: "    top-1 agreement %.2f%%", d.top1Agreement * 100))
    }
}
