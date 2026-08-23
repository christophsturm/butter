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
// Generate — autoregressive text generation, both buffered and streaming.
//
// Streaming is the primitive: `generateStream(prompt:parameters:)` yields
// one `GenerationChunk` per generated token plus a final chunk carrying
// the full `GenerationStats`. The buffered `generate(prompt:parameters:)`
// is a thin collector over the same stream so there's one source of truth
// for the prefill + decode loop.
//
// strategy: simple slow prefill (decode each prompt token
// sequentially through the same forward path used at decode time), then
// greedy GPU argmax decode. Sampling parameters on `GenerationParameters`
// (temperature, topP, …) are no-ops on the greedy fast path until GPU
// sampling kernels land in planned. `prefillStepSize` is honored once
// chunked prefill ships; today's per-token prefill ignores it.
//
// Stats: a `PhaseMemoryTracker` samples GPU memory at each token boundary
// and at the prefill→decode transition. Cost is one
// `MTLDevice.currentAllocatedSize` read per token (sub-µs);
// `GenerationStats` is always populated.

import Foundation
import IronSwift
import Tokenizers
#if canImport(Darwin)
    import Darwin
#endif

// MARK: - Public types

public struct GenerationResult: Sendable {
    public let promptTokens: [Int]
    public let generatedTokens: [Int]
    public let text: String
    public let stats: GenerationStats

    public init(
        promptTokens: [Int], generatedTokens: [Int],
        text: String, stats: GenerationStats
    ) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.text = text
        self.stats = stats
    }

    // Convenience read-throughs to the underlying stats.
    public var tokensPerSecond: Double { stats.decodeTokensPerSecond }
    public var prefillTokensPerSecond: Double { stats.prefillTokensPerSecond }
    public var prefillTimeS: Double { stats.prefillTimeS }
    public var decodeTimeS: Double { stats.decodeTimeS }
    public var timeToFirstTokenS: Double { stats.timeToFirstTokenMs / 1000.0 }
}

/// One yield from the streaming generation API. For decode tokens,
/// `text` and `tokens` carry the delta since the last yield. The
/// stream's final chunk has `tokens.isEmpty`, `text == ""`, and a
/// non-nil `stats` populated with full memory + timing numbers.
public struct GenerationChunk: Sendable {
    public let text: String
    public let tokens: [Int]
    public let position: Int
    public let stats: GenerationStats?

    public var isFinal: Bool { stats != nil }

    public init(
        text: String, tokens: [Int], position: Int,
        stats: GenerationStats? = nil
    ) {
        self.text = text
        self.tokens = tokens
        self.position = position
        self.stats = stats
    }
}

// MARK: - Public API

extension Model {
    /// Buffered generation. Collects every chunk from `generateStream`
    /// and returns the final result. Use this when you want the full
    /// text in one shot; use `generateStream` for token-by-token UI
    /// streaming.
    /// - Parameter profile: telemetry sink for this generation. Defaults
    ///   to the process-wide `Profile.shared`; pass a dedicated instance
    ///   to accumulate phase timings + signposts independently (the
    ///   prerequisite for per-sequence telemetry under planned's batched
    ///   decode, where one shared singleton would conflate sequences).
    public func generate(
        prompt: String,
        parameters: GenerationParameters? = nil,
        profile: Profile = .shared
    ) async throws -> GenerationResult {
        let promptTokens = encodePrompt(prompt)
        let params = parameters ?? defaultGenerationParameters
        let stream = generateStreamInternal(
            promptTokens: promptTokens,
            parameters: params, profile: profile)
        return try await collectStream(stream, promptTokens: promptTokens)
    }

    /// Streaming generation. Yields one `GenerationChunk` per generated
    /// token (with the decoded delta text), then a final chunk with the
    /// full `GenerationStats`. Cancel the consuming `Task` to abort
    /// generation early — the producer task notices the cancellation at
    /// the next token boundary.
    ///
    /// - Parameter profile: telemetry sink for this generation. See
    ///   `generate(prompt:parameters:profile:)` for the rationale.
    public func generateStream(
        prompt: String,
        parameters: GenerationParameters? = nil,
        profile: Profile = .shared
    )
        -> AsyncThrowingStream<GenerationChunk, Error>
    {
        let promptTokens = encodePrompt(prompt)
        let params = parameters ?? defaultGenerationParameters
        return generateStreamInternal(
            promptTokens: promptTokens,
            parameters: params, profile: profile)
    }

    // MARK: - Prompt encoding

    /// Encode a raw prompt string to token ids, prepending the model's
    /// `<bos>` token when the engine declares `requiresLeadingBOS` and
    /// the tokenizer's post-processor did not already add one.
    ///
    /// Gemma 4 is the motivating case: it is BOS-critical, but its
    /// `tokenizer.json` post-processor's `single` template is bare, so
    /// `Tokenizer.encode` returns no leading BOS. Without this prefix
    /// the model generates incoherent text. Gemma 3 and most other
    /// families list `<bos>` in their post-processor and need no fixup
    /// here. The guard against a BOS already being present keeps the
    /// helper idempotent and safe for families whose post-processor
    /// does add one.
    /// Public because it is the CANONICAL prompt tokenization — any
    /// caller that reports prompt-token counts (e.g. the CLI's streaming
    /// display) must use this, not a bare `tokenizer.encode`, or its
    /// count drifts from what generation actually prefills (BOS).
    public func encodePrompt(_ prompt: String) -> [Int] {
        var tokens = tokenizer.encode(text: prompt)
        guard engine.requiresLeadingBOS,
            let bos = tokenizer.bosTokenId ?? config.bosTokenId
        else { return tokens }
        if tokens.first != bos {
            tokens.insert(bos, at: 0)
        }
        return tokens
    }

    // MARK: - Internal entry points (shared by chat overloads)

    /// Build the producer stream from already-encoded prompt tokens.
    /// Used by both the `prompt:` and `messages:` public overloads so
    /// the tokenizer is invoked exactly once per call.
    internal func generateStreamInternal(
        promptTokens: [Int],
        parameters params: GenerationParameters,
        profile: Profile = .shared,
        caches: [any LayerCacheProtocol]? = nil,
        prefillStart: Int = 0
    )
        -> AsyncThrowingStream<GenerationChunk, Error>
    {
        runStream(
            promptTokens: promptTokens, parameters: params, profile: profile,
            caches: caches, prefillStart: prefillStart)
    }

    /// Drain a generation stream into a buffered `GenerationResult`.
    internal func collectStream(
        _ stream: AsyncThrowingStream<GenerationChunk, Error>,
        promptTokens: [Int]
    ) async throws -> GenerationResult {
        var generated: [Int] = []
        var text = ""
        var stats: GenerationStats?
        for try await chunk in stream {
            generated.append(contentsOf: chunk.tokens)
            text += chunk.text
            if let s = chunk.stats { stats = s }
        }
        guard let stats else {
            throw GenerationError.streamEndedWithoutFinalChunk
        }
        return GenerationResult(
            promptTokens: promptTokens,
            generatedTokens: generated,
            text: text, stats: stats
        )
    }

    // MARK: - Core stream producer

    private func runStream(
        promptTokens: [Int],
        parameters params: GenerationParameters,
        profile: Profile,
        caches: [any LayerCacheProtocol]?,
        prefillStart: Int
    )
        -> AsyncThrowingStream<GenerationChunk, Error>
    {
        AsyncThrowingStream { continuation in
            // Capture the values we need on the producer task. `Model`
            // is @unchecked Sendable, so this is fine.
            let model = self
            // T38: decode is a tight latency-sensitive per-token host loop
            // (unlike prefill's one big batched dispatch). An unstructured
            // `Task {}` here previously had no explicit priority and just
            // inherited whatever the ambient context handed it. Request the
            // highest priority Swift's structured concurrency exposes
            // (`.high` == QOS_CLASS_USER_INITIATED) so the token loop isn't
            // left at medium/default if the calling context is anything
            // less than interactive. Kill switch: BUTTER_DECODE_QOS=0
            // restores the old bare `Task {}` for A/B.
            let qosFixEnabled =
                ButterEnv["BUTTER_DECODE_QOS"] != "0"
            let producer = Task(priority: qosFixEnabled ? .high : nil) {
                do {
                    try await model.driveGeneration(
                        promptTokens: promptTokens,
                        parameters: params,
                        profile: profile,
                        continuation: continuation,
                        sessionCaches: caches,
                        prefillStart: prefillStart
                    )
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                producer.cancel()
            }
        }
    }

    /// Session HTTP abort runs this on the session producer Task so
    /// `Task.checkCancellation` in prefill sees the client close. One-shot
    /// `generate()` still goes through `runStream`.
    @discardableResult
    func driveGeneration(
        promptTokens: [Int],
        parameters params: GenerationParameters,
        profile: Profile,
        continuation: AsyncThrowingStream<GenerationChunk, Error>.Continuation,
        sessionCaches: [any LayerCacheProtocol]?,
        prefillStart: Int,
        onPrefillComplete: (@Sendable () -> Void)? = nil
    ) async throws -> [Int] {
        // T38: pin the executing thread's OS-visible QoS class to
        // user-interactive for the duration of this generate call (covers
        // both prefill and decode; prefill was already unaffected by
        // scheduling pressure so this is a no-op cost there). Swift's
        // structured-concurrency `Task priority:` only reaches
        // QOS_CLASS_USER_INITIATED at the top (`.high`). This goes one
        // step further and asks XNU directly via the pthread QoS API,
        // covering hops onto worker threads Task priority alone doesn't
        // guarantee. Best-effort: `pthread_set_qos_class_self_np` can fail
        // (e.g. thread already opted out of QoS); a failure here is not
        // fatal, decode just runs at whatever QoS it already had.
        // Kill switch: BUTTER_DECODE_QOS=0 skips this for A/B.
        // `Thread.current` is unavailable from async contexts (Swift
        // concurrency can hop this call onto any worker thread), so the
        // QoS pin goes straight through the pthread C API instead, which
        // has no such restriction and applies to whichever OS thread is
        // actually running this code at the time of the call.
        #if canImport(Darwin)
            if ButterEnv["BUTTER_DECODE_QOS"] != "0" {
                pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
            }
        #endif
        // T21 hygiene: re-pin the weight residency set at the top of
        // every generate call, not just at load: cheap
        // (`MTLResidencySet.requestResidency()` re-touches the whole set
        // in one driver call) and guards against the OS paging weights
        // back out between calls under memory pressure (long-idle chat
        // sessions, another process squeezing this one). See
        // `Device.refreshWeightResidency` for the full rationale.
        Device.shared.refreshWeightResidency()
        // Session path reuses live KV — do not release/rebuild. One-shot
        // `generate()` still drops the previous generation's caches.
        let caches: [any LayerCacheProtocol]
        if let sessionCaches {
            caches = sessionCaches
        } else {
            Device.shared.releaseGenerationCaches()
            caches = engine.makeLayerCaches()
        }

        // D9: fail fast, before any prefill work, when this call can
        // never fit the KV cache's fixed physical capacity
        // (`engine.maxSeq` — the ceiling every `.unbounded`
        // attention-cache layer allocates its buffer at, sized once at
        // load from `--max-context` / the family default; see
        // `KVCache.init`'s `KVEvictionState(policy:bufferCapacity:)`).
        // NOT every layer's physical buffer is `maxSeq`-sized, though:
        // a `.window` layer allocates only `min(slidingWindow, maxSeq)`
        // slots (see e.g. `Gemma4Text.makeLayerCaches`'s `physicalCap`)
        // — that's fine, because a windowed layer's whole point is to
        // never need more than its window regardless of how long the
        // conversation runs. `promptTokens` is always the FULL
        // cumulative token count — for session calls it includes prior
        // turns, not just the new suffix (`GenerationSession.complete`
        // passes the whole conversation) — so this one check covers
        // both entry points.
        //
        // Skipped only when EVERY attention-cache layer uses `.window`
        // eviction: a windowed cache is *designed* to run past its
        // physical capacity forever (FIFO rotation, e.g.
        // streaming-LLM-style attention sinks — see
        // `KVCacheEviction.swift`), so `promptTokens.count + maxTokens`
        // legitimately exceeding `maxSeq` is the normal, intended way
        // to use it there, not an error. Eviction policy is NOT
        // engine-wide, though: several hybrid families mix `.window`
        // local layers with `.unbounded` global layers within the same
        // engine (Gemma 2/3/4, GPT-OSS, Laguna, MuseGlimmer), and those
        // `.unbounded` layers have the same fixed physical ceiling as a
        // plain model — checking only the first attention layer, or
        // exempting on ANY `.window` hit, silently disabled this guard
        // for exactly those families. So: collect every attention-cache
        // layer and only skip the check when the set is non-empty AND
        // every one of them is `.window`. Without this guard, an
        // `.unbounded` cache instead runs the full (possibly
        // multi-chunk) prefill and only then hits
        // `KVEvictionState.reserveNextSlot()`'s precondition mid-decode
        // — a much later, less actionable failure for the same root
        // cause.
        let attentionCaches = caches.compactMap { $0 as? any KVCacheProtocol }
        let allWindowed =
            !attentionCaches.isEmpty
            && attentionCaches.allSatisfy {
                if case .window = $0.eviction { return true }
                return false
            }
        if !allWindowed {
            let required = promptTokens.count + params.maxTokens
            if required > engine.maxSeq {
                throw GenerationError.contextOverflow(
                    promptTokens: promptTokens.count,
                    maxNewTokens: params.maxTokens,
                    maxContextLength: engine.maxSeq)
            }
        }

        let nPast = min(max(prefillStart, 0), promptTokens.count)
        // Gemma 3+, several Qwen variants, and a few llama-tuned models
        // publish `eos_token_id` as a *list* (model EOS plus end-of-turn
        // tokens like `<|im_end|>`). Stop on any of them.
        let eosIds = config.eosTokenIds
        let stopSet: Set<Int> = {
            var s = params.extraStopTokens
            if params.stopOnEOS { s.formUnion(eosIds) }
            return s
        }()
        let weightsBytes = engine.parameters().reduce(0) { $0 + $1.1.byteCount }
        let memTracker = PhaseMemoryTracker()

        // T40 follow-up: clear any candidate certificate left over from a previous
        // generation call in this process (e.g. a prior Laguna run right
        // before a different, non-Laguna `engine` -- that family's
        // forward() never populates this, so a stale value would
        // otherwise carry an unrelated vocab size into this generation's
        // `sampleNext`). `currentSamplingMode` doesn't need the same
        // reset: it's read-then-restored around every `cpuSample` call
        // below and defaults to `.pruneCertified` at rest.
        LagunaLmHeadPrune.resetCandidateCertificate()
        // BUTTER_DEBUG_LOGIT_MARGIN: fresh per-generation token counter so
        // t=0 always lands on the prefill-computed seed token below,
        // regardless of which decode loop (GPU token loop / fallback
        // forwardSample / cpuSample) produces the rest. No-op when off.
        LogitMarginDebug.reset()
        // BUTTER_DEBUG_ROUTER_SETS: same per-generation reset, sibling
        // instrument for butter#108 (see RouterSetsDebug.swift). No-op
        // when off.
        RouterSetsDebug.reset()

        // Path selector for the sampling step:
        //
        //   greedy-GPU       T == 0, rep-penalty == 1 - existing argmax
        //                    kernel (4-byte readback per token, fastest
        //                    path).
        //   gpu-categorical  T > 0, no filters — forward + GPU softmax +
        //                    categorical sample kernel. Logits stay on GPU;
        //                    only the chosen token id flows back (4 bytes).
        //   cpu-sample       T > 0 with any of top-K / top-P / min-P /
        //                    rep-penalty, or T == 0 with rep-penalty != 1:
        //                    full vocab readback + Sampling.sample pipeline.
        let hasFilters =
            params.topK > 0 || params.topP < 1.0 || params.minP > 0
            || params.repetitionPenalty != 1.0
        // T39 finding (benchmarks/laguna-xs/README.md): family defaults
        // like Laguna's `topK: 64, topP: 0.95` used to route every plain
        // `--temperature 0` call through cpuSample even though greedy
        // argmax is mathematically invariant to top-K / top-P / min-P
        // filtering - those filters only ever remove candidates, never
        // rescale the winning logit's value, and the raw-argmax token
        // always survives all three (top-K keeps >= 1 candidates
        // including the max; top-P's cumulative-softmax walk always
        // includes the top token before it can stop; min-P's threshold
        // is defined relative to the max token itself, so the max always
        // clears it). Repetition penalty is the one filter that does not
        // have this property - it rescales logit values in place before
        // argmax ever runs (see `Sampling.sample`'s step (1), applied
        // ahead of its own `temperature == 0` short-circuit), so it can
        // change which token wins and must keep routing through cpuSample.
        //
        // `BUTTER_GREEDY_GPU_ARGMAX=0` disables this and restores the old
        // routing (any hasFilters sends greedy through cpuSample too).
        let greedyGPUFastPathEnabled =
            ButterEnv["BUTTER_GREEDY_GPU_ARGMAX"] != "0"
        enum Path: String { case greedyGPU, gpuCategorical, cpuSample }
        let path: Path = {
            if greedyGPUFastPathEnabled, params.temperature == 0,
                params.repetitionPenalty == 1.0
            {
                return .greedyGPU
            }
            if hasFilters { return .cpuSample }
            if params.temperature == 0 { return .greedyGPU }
            return .gpuCategorical
        }()

        var rng = params.makeRNG()
        var tokenHistory = promptTokens  // grows as decode produces tokens; used by cpu-sample

        // ─── Live quality-metric capture (design §5) ─────────────────────
        // The single InspectTap drives this. `BUTTER_TELEMETRY=ppl` opts into
        // accumulating the model's self-perplexity over its own stream — the
        // NLL of each token it actually picks. Off by default: when the flag
        // is unset `capturePPL` is false and the hot path is byte-for-byte
        // the existing fused-kernel sampler (no logit readback, no softmax).
        //
        // `.kld` / `.niah` are not actionable in the live decode loop — KLD
        // needs a paired reference and NIAH is its own retrieval harness; both
        // run through `butter bench`. The tap flags still gate them there.
        let tap = InspectTap.fromEnvironment
        let capturePPL = tap.captures(.perplexity)
        var liveNLLSum = 0.0
        var liveNLLCount = 0

        func sampleNext(tokenId t: Int, position i: Int) -> Int {
            // T39: brackets one whole decode token (forward + sample) for
            // the "sampling/host_gap(residual)" row in
            // BUTTER_LAGUNA_DECODE_KERNEL_TIMING=1's report - the wall time
            // inside a token no checkpoint bucket already accounts for. No-op
            // when the env var is unset.
            LagunaDecodeKernelTiming.beginToken()
            defer { LagunaDecodeKernelTiming.endToken() }
            // Capturing path: route through full logits so we can read the
            // chosen token's probability. Only taken when the flag is on.
            if capturePPL {
                let logits = engine.forward(tokenId: t, position: i, caches: caches)
                let chosen: Int
                switch path {
                case .greedyGPU:
                    chosen = Perplexity.argmaxLogits(logits)
                case .gpuCategorical, .cpuSample:
                    chosen = Sampling.sample(
                        logits, parameters: params,
                        rng: &rng, tokenHistory: tokenHistory)
                }
                liveNLLSum += Perplexity.negLogSoftmaxAt(logits: logits, index: chosen)
                liveNLLCount += 1
                return chosen
            }
            switch path {
            case .greedyGPU:
                return engine.forwardSample(tokenId: t, position: i, caches: caches)
            case .gpuCategorical:
                let u = Float(Double.random(in: 0 ..< 1, using: &rng))
                return engine.forwardSampleCategorical(
                    tokenId: t, position: i, caches: caches,
                    temperature: params.temperature, uniformDraw: u
                )
            case .cpuSample:
                // T40 follow-up: cpuSample is the one path that reads more than the
                // winning logit's identity (full softmax at temperature>0,
                // repetition-penalty rescale before argmax/sampling) --
                // exactly the case the certified lm_head prune's coarse
                // approximation on non-candidate rows corrupts. Tell
                // Laguna's `lagunaFinalNormLmHead` which strategy this
                // token needs via the static context var (see
                // `LagunaLmHeadPrune.currentSamplingMode` doc for why a
                // protocol-level parameter isn't used); always restore
                // the certified default afterward so every other path
                // (and every non-Laguna family, which never reads this
                // var) is unaffected.
                let mode = LagunaLmHeadPrune.samplingMode(
                    temperature: params.temperature,
                    repetitionPenalty: params.repetitionPenalty)
                LagunaLmHeadPrune.currentSamplingMode = mode
                defer { LagunaLmHeadPrune.currentSamplingMode = .pruneCertified }
                let logits = engine.forward(tokenId: t, position: i, caches: caches)
                if mode == .pruneCandidateRestricted,
                    let cert = LagunaLmHeadPrune.lastCandidateCertificate,
                    cert.coarse.elementCount == logits.elementCount
                {
                    // Certificate came from THIS token's forward pass
                    // (vocab size matches the logits we just got back) --
                    // restrict + renormalize to the certified candidate
                    // set instead of paying for a full BF16 GEMV. See
                    // `LagunaLmHeadPrune.maskNonCandidates` for the
                    // statistical justification.
                    let masked = LagunaLmHeadPrune.maskNonCandidates(
                        assembled: Sampling.decodeF32(logits),
                        coarse: Sampling.decodeF32(cert.coarse),
                        delta: Sampling.decodeF32(cert.delta),
                        thr: Sampling.decodeF32(cert.thr)[0])
                    let chosen = Sampling.sample(
                        values: masked, parameters: params,
                        rng: &rng, tokenHistory: tokenHistory)
                    // BUTTER_DEBUG_LOGIT_MARGIN: CPU path, candidate-
                    // restricted branch — margin computed from the same
                    // masked values Sampling.sample just argmaxed over.
                    if params.temperature == 0 {
                        LogitMarginDebug.report(values: masked, chosenId: chosen)
                    }
                    return chosen
                }
                let chosen = Sampling.sample(
                    logits, parameters: params,
                    rng: &rng, tokenHistory: tokenHistory)
                // BUTTER_DEBUG_LOGIT_MARGIN: CPU path, unmasked branch —
                // the "CPU argmax" arm of the drift investigation
                // (BUTTER_GREEDY_GPU_ARGMAX=0 routes Laguna's default
                // top-K/top-P-having params here even at temperature 0).
                if params.temperature == 0 {
                    LogitMarginDebug.report(logits: logits, chosenId: chosen)
                }
                return chosen
            }
        }

        Debug.log(
            .generate,
            "begin prefill: \(promptTokens.count) tokens, maxTokens=\(params.maxTokens), path=\(path.rawValue)"
        )

        // ─── Prefill ─────────────────────────────────────────────────
        //
        // Chunked prefill: `engine.forwardMulti(...)` in
        // `prefillStepSize`-sized chunks over the full prompt. The last
        // multi returns logits at the final prompt position; we sample
        // the first generate token from those logits (no second stack).
        //
        // The default `forwardMulti` implementation in `LanguageModel`
        // falls back to a per-token loop, so families that haven't
        // overridden it retain correct behaviour. Optimised families
        // override `forwardMulti` with a batched attention/MLP path.
        //
        // Empty prompts hit the early-return below.
        // Census tap (additive, env-gated): reset the standing
        // per-kernel-name dispatch histogram (Sources/IronSwift/
        // DispatchCensus.swift) right before prefill so counts below
        // exclude model-load dispatches. No-op when
        // BUTTER_DISPATCH_CENSUS is unset.
        if DispatchCensus.enabled { DispatchCensus.reset() }
        let prefillStart = Date()
        var nextToken = 0
        // Qwen3.8 MTP attends over the committed prompt, not just tokens
        // generated after prefill. Capture post-final-norm trunk hiddens while
        // the prompt is already flowing through the backbone and use them to
        // prime the sidecar's one-layer KV cache. This is gated by the same
        // explicit speculative-decode switch as the decode loop.
        let qwenMTPPrefillModel: Qwen35Model? =
            ButterEnv["BUTTER_QWEN_MTP_SPEC"] == "1"
            ? engine as? Qwen35Model : nil
        let qwenMTPPrefillHead = qwenMTPPrefillModel?.mtpHead
        var qwenMTPLastPromptHidden: Tensor?
        qwenMTPPrefillHead?.resetCache()
        try profile.signpost("prefill") {
            let n = promptTokens.count
            // Honour the caller's explicit `prefillStepSize` if set;
            // otherwise fall through to the engine's tuned default
            // (1024 generic, 2048 GPT-OSS, 4096 Gemma 4 / Qwen 3.5 MoE).
            // BUTTER_PREFILL_STEP=<n> overrides both for chunk-size sweeps —
            // larger chunks amortize per-chunk costs (KV append, gate/beta,
            // sort plans, attention prefix re-reads) at the cost of larger
            // transient activations.
            let chunkSize = max(
                1,
                ButterEnv["BUTTER_PREFILL_STEP"].flatMap(Int.init)
                    ?? params.prefillStepSize
                    ?? engine.defaultPrefillStepSize)
            // Multi over the full prompt (all n tokens). The last multi
            // returns logits at the final prompt position so we can sample
            // the first generate token without a second full layer stack
            // (previously multi was n-1 then sampleNext re-forwarded last).
            var pos = nPast
            var lastLogits: Tensor?
            while pos < n {
                PrefillAbortProbe.markBegun(
                    promptCount: n, firstToken: promptTokens.first, chunkSize: chunkSize)
                try Task.checkCancellation()
                let end = min(pos + chunkSize, n)
                let chunk = Array(promptTokens[pos ..< end])
                // Census tap (additive, env-gated): REAL (unforced) wall
                // time for this chunk's forward + commit + wait, using the
                // chunk boundary this loop already pays for — no extra
                // synchronization added, so unlike BUTTER_PROFILE_LAYERS
                // this number is NOT inflated. Reuses BUTTER_DISPATCH_CENSUS
                // rather than adding a new flag.
                let chunkWallT0 = DispatchCensus.enabled ? CFAbsoluteTimeGetCurrent() : 0
                let cmd = Device.shared.makeCommandBuffer()
                var qwenPromptHidden: Tensor?
                if let qwen = qwenMTPPrefillModel,
                    qwenMTPPrefillHead != nil
                {
                    let out = qwen.forwardManyHiddenAndLastLogits(
                        tokenIds: chunk, startPosition: pos,
                        caches: caches, on: cmd, device: Device.shared)
                    lastLogits = out.logits
                    qwenPromptHidden = out.hidden
                } else {
                    lastLogits = engine.forwardMulti(
                        tokenIds: chunk, startingAt: pos,
                        caches: caches, on: cmd,
                        device: Device.shared)
                }
                cmd.commit()
                cmd.waitUntilCompleted()
                if DispatchCensus.enabled {
                    let ms = (CFAbsoluteTimeGetCurrent() - chunkWallT0) * 1000
                    fputs(
                        "[census] prefill chunk REAL startPos=\(pos) T=\(chunk.count) wallMs=\(String(format: "%.2f", ms))\n",
                        stderr)
                }
                PrefillAbortProbe.markCommitted(
                    promptCount: n, firstToken: promptTokens.first, chunkSize: chunkSize)
                if let head = qwenMTPPrefillHead,
                    let hiddenRows = qwenPromptHidden,
                    let qwen = qwenMTPPrefillModel
                {
                    // Across a chunk boundary, the previous chunk's last
                    // hidden pairs with this chunk's first token.
                    if let previous = qwenMTPLastPromptHidden {
                        let primeCmd = Device.shared.makeCommandBuffer()
                        head.draftStepMany(
                            prevHiddensFlat: previous,
                            prevTokens: [chunk[0]], startPosition: pos,
                            on: primeCmd, device: Device.shared)
                        primeCmd.commit()
                        primeCmd.waitUntilCompleted()
                    }
                    // Within a chunk, hidden row i pairs with token i+1.
                    if chunk.count > 1 {
                        let rows = chunk.count - 1
                        let primeHidden = Tensor(
                            buffer: hiddenRows.buffer, offset: hiddenRows.offset,
                            shape: [rows * qwen.hidden], dtype: hiddenRows.dtype)
                        let primeCmd = Device.shared.makeCommandBuffer()
                        head.draftStepMany(
                            prevHiddensFlat: primeHidden,
                            prevTokens: Array(chunk.dropFirst()),
                            startPosition: pos + 1,
                            on: primeCmd, device: Device.shared)
                        primeCmd.commit()
                        primeCmd.waitUntilCompleted()
                    }
                    let rowBytes = qwen.hidden * hiddenRows.dtype.byteSize
                    qwenMTPLastPromptHidden = Tensor(
                        buffer: hiddenRows.buffer,
                        offset: hiddenRows.offset + (chunk.count - 1) * rowBytes,
                        shape: [qwen.hidden], dtype: hiddenRows.dtype)
                }
                memTracker.sample()
                pos = end
            }
            if n > 0, let logits = lastLogits {
                try Task.checkCancellation()
                // Sample first generate token from multi logits (no re-forward).
                let chosen: Int
                if path == .greedyGPU || params.temperature == 0 {
                    chosen = Sampling.argmax(logits)
                    // BUTTER_DEBUG_LOGIT_MARGIN: t=0 is always this
                    // prefill-computed seed token — identical code path
                    // regardless of BUTTER_GREEDY_GPU_ARGMAX /
                    // BUTTER_GPU_TOKEN_LOOP, so any divergence those
                    // flags cause only shows up from t=1 on. Skip when
                    // maxTokens == 0: `chosen` is computed either way but
                    // never actually emitted in that case.
                    if params.maxTokens > 0 {
                        LogitMarginDebug.report(logits: logits, chosenId: chosen)
                    }
                } else {
                    chosen = Sampling.sample(
                        logits, parameters: params,
                        rng: &rng, tokenHistory: tokenHistory)
                }
                if capturePPL {
                    liveNLLSum += Perplexity.negLogSoftmaxAt(
                        logits: logits, index: chosen)
                    liveNLLCount += 1
                }
                nextToken = chosen
                memTracker.sample()
            }
        }
        memTracker.endPrefill()
        onPrefillComplete?()
        if DispatchCensus.enabled {
            fputs("[dispatch-census] === PREFILL (\(promptTokens.count) tokens) ===\n", stderr)
            DispatchCensus.report()
            DispatchCensus.reset()
        }
        let prefillTime = Date().timeIntervalSince(prefillStart)
        profile.recordPhase("prefill", durationS: prefillTime)
        profile.event("ttft")
        Debug.log(
            .generate,
            String(
                format: "prefill done in %.3fs (%.1f tok/s)", prefillTime,
                Double(promptTokens.count) / max(prefillTime, 1e-9)))

        if promptTokens.isEmpty {
            let stats = makeStats(
                promptTokens: promptTokens, generatedCount: 0,
                contextSize: engine.maxSeq, prefillTime: 0, decodeTime: 0,
                ttftMs: 0, perTokenWallclock: [],
                memTracker: memTracker, caches: caches,
                weightsBytes: weightsBytes, splitTokens: nil
            )
            continuation.yield(
                GenerationChunk(
                    text: "", tokens: [],
                    position: 0, stats: stats))
            return []
        }

        // TTFT for the slow per-token prefill is identical to prefill time.
        let ttftMs = prefillTime * 1000

        // ─── Decode ──────────────────────────────────────────────────
        let decodeStart = Date()
        var generated: [Int] = []
        var perTokenWallclock: [Double] = []
        var lastStep = decodeStart
        var pos = promptTokens.count

        // F-token-loop: GPU-resident decode feedback for the greedy-argmax
        // fast path (see GpuTokenLoop.swift). The old loop below reads the
        // chosen token id back to the host (`forwardSample`'s
        // `cmd.commit()` + `waitUntilCompleted()`) before the *next* step
        // can even start encoding, even though the next step's embedding
        // gather only needs the token id as a GPU buffer, not a host
        // value. `GpuTokenLoop.runGPU` chains steps directly off each
        // other's `Ops.argmax` output buffer instead, and only reads
        // token ids back asynchronously: for detokenization and the
        // stop-token check, the two things that actually need a host
        // value. Requires the greedy-argmax path (no PPL capture, no
        // sampling filters, those still need the full logits or CPU
        // randomness per step) and family support
        // (`engine.supportsGPUTokenFeedback`; currently Laguna).
        // Escape: BUTTER_GPU_TOKEN_LOOP=0 restores the old per-token
        // synchronous loop unconditionally.
        let gpuTokenLoopEnabled =
            ButterEnv["BUTTER_GPU_TOKEN_LOOP"] != "0"
        let useGpuTokenLoop =
            gpuTokenLoopEnabled && path == .greedyGPU && !capturePPL
            && engine.supportsGPUTokenFeedback

        // Spec 030 phase 2: MTP self-speculative decode. Opt-in via
        // BUTTER_QWEN_MTP_SPEC=1 (default OFF), Qwen3x-only, and only
        // when the checkpoint shipped an `mtp-sidecar/` (`mtpHead !=
        // nil`). γ (draft-chain ceiling) defaults to 8; the promoted cost
        // model normally selects a smaller effective depth and expands only
        // after observed accepts. Override with
        // BUTTER_QWEN_MTP_GAMMA=<n>.
        //
        // Two arms: `temperature == 0` dispatches the original
        // greedy-argmax loop (the `.greedyGPU` path already implies
        // repetitionPenalty == 1); `temperature > 0` dispatches lossless
        // rejection-sampling MTP (`SpecDecode.generateSampledMTP`) —
        // gated additionally on `repetitionPenalty == 1.0` since the
        // sampled path's p_j/q_j pipeline doesn't support it (it has no
        // well-defined meaning mid-draft-burst; see
        // `Sampling.buildFilteredWeights`'s doc). Falls through to the
        // non-spec `cpuSample`/`gpuCategorical` path otherwise.
        let mtpSpecEnabled =
            ButterEnv["BUTTER_QWEN_MTP_SPEC"] == "1"
        let mtpModel = engine as? Qwen35Model
        let useMTPSpecGreedy =
            mtpSpecEnabled && path == .greedyGPU && !capturePPL
            && params.temperature == 0
            && (mtpModel?.mtpHead) != nil
        let useMTPSpecSampled =
            mtpSpecEnabled && !capturePPL
            && params.temperature > 0
            && params.repetitionPenalty == 1.0
            && (mtpModel?.mtpHead) != nil
        // MTP sidecar is primed during a full-prompt prefill. A session
        // suffix prefill would leave the draft cache missing the prefix.
        let useMTPSpec =
            (useMTPSpecGreedy || useMTPSpecSampled) && nPast == 0

        // F-131: NemotronH MTP self-speculative decode. Opt-in via
        // BUTTER_NEMOTRON_MTP_SPEC=1 (default OFF — measured a
        // throughput LOSS on this codebase's NemotronH forward path;
        // see `NemotronHMTPSpecDecode.swift`'s file header for why it's
        // architectural, not a tuning gap). Greedy-only — the driver
        // doesn't support temperature > 0 (unlike Qwen3x's MTP path).
        // Requires BUTTER_NEMOTRON_MTP=1 at load time too (separate
        // gate — see `NemotronHMTPHead.swift`), which is what makes
        // `nemotronModel?.mtpHead` non-nil here.
        let nemotronMTPSpecEnabled =
            ButterEnv["BUTTER_NEMOTRON_MTP_SPEC"] == "1"
        let nemotronModel = engine as? NemotronHModel
        let useNemotronMTPSpec =
            nemotronMTPSpecEnabled && path == .greedyGPU && !capturePPL
            && params.temperature == 0
            && (nemotronModel?.mtpHead) != nil
            && nPast == 0

        if useMTPSpec, let qwen = mtpModel, let mtpHead = qwen.mtpHead {
            try Task.checkCancellation()
            if params.maxTokens > 0, !stopSet.contains(nextToken) {
                let gamma = max(
                    1,
                    Int(
                        ButterEnv[
                            "BUTTER_QWEN_MTP_GAMMA"] ?? "") ?? 8)
                // Prefill above already reset and primed the head cache with
                // the committed prompt history, and captured the final seed
                // hidden for the first drafting round.
                var mtpRNG = params.makeRNG()
                // The seed (`nextToken`) sits at `pos` — same contract as
                // the GPU-loop's seedPosition (#108): capture it by name
                // BEFORE the emit closure below starts mutating `pos` for
                // display bookkeeping, so the MTP entry position can never
                // drift if this block is reordered.
                let mtpEntryPosition = pos
                let stats = profile.signpost("decode_mtp_spec") {
                    let emitToken: (Int) -> Void = { tokenId in
                        generated.append(tokenId)
                        let text = self.tokenizer.decode(
                            tokens: [tokenId], skipSpecialTokens: true)
                        continuation.yield(
                            GenerationChunk(
                                text: text, tokens: [tokenId],
                                position: pos + 1, stats: nil))
                        memTracker.sample()
                        let now = Date()
                        perTokenWallclock.append(now.timeIntervalSince(lastStep))
                        lastStep = now
                        pos += 1
                    }
                    if useMTPSpecSampled {
                        return SpecDecode.generateSampledMTP(
                            model: qwen, head: mtpHead, gamma: gamma,
                            lastToken: nextToken, position: mtpEntryPosition,
                            lastHidden: qwenMTPLastPromptHidden,
                            caches: caches, history: &tokenHistory,
                            maxNewTokens: params.maxTokens,
                            stopTokens: stopSet, device: Device.shared,
                            parameters: params, rng: &mtpRNG,
                            emit: emitToken)
                    }
                    return SpecDecode.generateGreedyMTP(
                        model: qwen, head: mtpHead, gamma: gamma,
                        lastToken: nextToken, position: mtpEntryPosition,
                        lastHidden: qwenMTPLastPromptHidden,
                        caches: caches, history: &tokenHistory,
                        maxNewTokens: params.maxTokens,
                        stopTokens: stopSet, device: Device.shared,
                        emit: emitToken)
                }
                Debug.log(
                    .generate,
                    "MTP spec decode: γ=\(gamma), "
                        + "accepted \(stats.candidatesAccepted)/\(stats.candidatesProposed) "
                        + "(\(String(format: "%.1f", stats.acceptanceRate * 100))%), "
                        + "verify=\(stats.verifySteps), replay=\(stats.replaySingleSteps), "
                        + "lockstep=\(stats.lockstepSteps)")
            }
        } else if useNemotronMTPSpec, let nemotron = nemotronModel, let mtpHead = nemotron.mtpHead {
            try Task.checkCancellation()
            if params.maxTokens > 0, !stopSet.contains(nextToken) {
                let gamma = max(
                    1,
                    Int(ButterEnv["BUTTER_NEMOTRON_MTP_GAMMA"] ?? "") ?? 2)
                mtpHead.resetCache()
                let mtpEntryPosition = pos
                let stats = profile.signpost("decode_nemotron_mtp_spec") {
                    let emitToken: (Int) -> Void = { tokenId in
                        generated.append(tokenId)
                        let text = self.tokenizer.decode(
                            tokens: [tokenId], skipSpecialTokens: true)
                        continuation.yield(
                            GenerationChunk(
                                text: text, tokens: [tokenId],
                                position: pos + 1, stats: nil))
                        memTracker.sample()
                        let now = Date()
                        perTokenWallclock.append(now.timeIntervalSince(lastStep))
                        lastStep = now
                        pos += 1
                    }
                    return NemotronHMTPSpecDecode.generateGreedy(
                        model: nemotron, head: mtpHead, gamma: gamma,
                        lastToken: nextToken, position: mtpEntryPosition,
                        caches: caches, history: &tokenHistory,
                        maxNewTokens: params.maxTokens,
                        stopTokens: stopSet, device: Device.shared,
                        emit: emitToken)
                }
                Debug.log(
                    .generate,
                    "NemotronH MTP spec decode: γ=\(gamma), "
                        + "accepted \(stats.candidatesAccepted)/\(stats.candidatesProposed) "
                        + "(\(String(format: "%.1f", stats.acceptanceRate * 100))%), "
                        + "verify=\(stats.verifySteps), replay=\(stats.replaySingleSteps), "
                        + "lockstep=\(stats.lockstepSteps)")
            }
        } else if useGpuTokenLoop {
            try Task.checkCancellation()
            if params.maxTokens > 0, !stopSet.contains(nextToken) {
                // The seed token is already known (picked by prefill's
                // sampler): emit it exactly as the old loop's first
                // iteration would, no GPU call needed to produce it.
                generated.append(nextToken)
                let seedText = tokenizer.decode(
                    tokens: [nextToken], skipSpecialTokens: true)
                continuation.yield(
                    GenerationChunk(
                        text: seedText, tokens: [nextToken],
                        position: pos + 1, stats: nil))
                tokenHistory.append(nextToken)
                memTracker.sample()
                let seedNow = Date()
                perTokenWallclock.append(seedNow.timeIntervalSince(lastStep))
                lastStep = seedNow
                // The seed token SITS at `pos`: prefill consumed the
                // prompt through pos-1 and sampled the seed as the
                // token at `pos`. The chained loop's first forward must
                // therefore process the seed AT `pos` (its RoPE angle,
                // its KV slot) — exactly like the host branch below,
                // which calls sampleNext(position: pos) BEFORE
                // incrementing. Capture that before the display
                // bookkeeping bumps `pos`, or the whole chained stream
                // runs one position late (butter#108: wrong RoPE +
                // KV written to slot pos+1 leaving slot pos stale —
                // observably different logits from layer 0 onward).
                let seedPosition = pos
                pos += 1

                // W35: under the non-blocking decode window
                // (BUTTER_LAGUNA_NONBLOCK_DECODE, default ON) the engine's
                // forward returns after *committing* the layer stack
                // instead of draining it, so raising the loop lag to 3
                // lets tokens N+1..N+3 encode on the host while token N
                // executes on the GPU. Ring size follows automatically
                // (`runGPU`: max(lag+3, 4)); stop-token semantics are
                // unchanged (speculative steps past a stop token are
                // computed and discarded, by design). Override via
                // BUTTER_GPU_LOOP_LAG. Default (flag off) stays 1.
                let nonblockDecode =
                    ButterEnv[
                        "BUTTER_LAGUNA_NONBLOCK_DECODE"] != "0"
                let gpuLoopLag = max(
                    1,
                    Int(ButterEnv["BUTTER_GPU_LOOP_LAG"] ?? "")
                        ?? (nonblockDecode ? 3 : 1))
                let remainingBudget = params.maxTokens - 1
                if remainingBudget > 0 {
                    // One signpost interval for the whole chained batch,
                    // not one per token: under pipelining there's no
                    // single discrete "this GPU call produced token N"
                    // moment on the host any more (that's the point),
                    // so per-token "decode_step" signposts (as the
                    // fallback loop below still emits) wouldn't mean
                    // the same thing here.
                    _ = try profile.signpost("decode_gpu_loop") {
                        try GpuTokenLoop.runGPU(
                            engine: engine, seedToken: nextToken,
                            startPosition: seedPosition, caches: caches,
                            remainingBudget: remainingBudget,
                            stopSet: stopSet, device: Device.shared,
                            lag: gpuLoopLag
                        ) { tokenId in
                            try Task.checkCancellation()
                            generated.append(tokenId)
                            let text = tokenizer.decode(
                                tokens: [tokenId], skipSpecialTokens: true)
                            continuation.yield(
                                GenerationChunk(
                                    text: text, tokens: [tokenId],
                                    position: pos + 1, stats: nil))
                            tokenHistory.append(tokenId)
                            memTracker.sample()
                            let now = Date()
                            perTokenWallclock.append(now.timeIntervalSince(lastStep))
                            lastStep = now
                            pos += 1
                        }
                    }
                }
            }
        } else {
            for _ in 0 ..< params.maxTokens {
                try Task.checkCancellation()
                if stopSet.contains(nextToken) { break }
                generated.append(nextToken)
                let chunkText = tokenizer.decode(
                    tokens: [nextToken],
                    skipSpecialTokens: true)
                continuation.yield(
                    GenerationChunk(
                        text: chunkText,
                        tokens: [nextToken],
                        position: pos + 1, stats: nil))
                let priorToken = nextToken
                // `autoreleasepool` per token — `sampleNext` allocates a
                // fresh command buffer + per-layer intermediate tensors
                // on every call; without draining here those Metal-side
                // allocations accumulate on the host's implicit
                // autorelease pool for the whole decode loop instead of
                // being freed per token. Immaterial on narrow/shallow
                // models but grows host RSS unboundedly (tens of GB
                // over a few hundred tokens) on a wide backbone —
                // observed while exercising this fallback path with
                // Muse Glimmer's 6656-hidden / 52-layer text engine.
                nextToken = profile.signpost("decode_step") {
                    autoreleasepool {
                        sampleNext(tokenId: priorToken, position: pos)
                    }
                }
                tokenHistory.append(priorToken)
                memTracker.sample()
                let now = Date()
                perTokenWallclock.append(now.timeIntervalSince(lastStep))
                lastStep = now
                pos += 1
            }
        }
        if DispatchCensus.enabled {
            fputs("[dispatch-census] === DECODE (\(generated.count) tokens) ===\n", stderr)
            DispatchCensus.report()
        }
        let decodeTime = Date().timeIntervalSince(decodeStart)
        memTracker.endDecode()
        profile.recordPhase("ttft", durationS: ttftMs / 1000)
        profile.recordPhase("decode", durationS: decodeTime)
        profile.recordPhase("generation_total", durationS: prefillTime + decodeTime)
        Debug.log(
            .generate,
            String(
                format: "decode done: %d tokens in %.3fs (%.1f tok/s)", generated.count, decodeTime,
                Double(generated.count) / max(decodeTime, 1e-9)))

        let split = ThinkingSplit.split(tokens: generated, model: self)
        // Live self-perplexity over the generated stream, when captured.
        let livePerplexity: Double? =
            (capturePPL && liveNLLCount > 0) ? exp(liveNLLSum / Double(liveNLLCount)) : nil
        let stats = makeStats(
            promptTokens: promptTokens, generatedCount: generated.count,
            contextSize: engine.maxSeq, prefillTime: prefillTime,
            decodeTime: decodeTime, ttftMs: ttftMs,
            perTokenWallclock: perTokenWallclock,
            memTracker: memTracker, caches: caches,
            weightsBytes: weightsBytes, splitTokens: split,
            genPerplexity: livePerplexity
        )

        continuation.yield(
            GenerationChunk(
                text: "", tokens: [],
                position: pos, stats: stats))
        return generated
    }

    // MARK: - Stats assembly

    private func makeStats(
        promptTokens: [Int], generatedCount: Int, contextSize: Int,
        prefillTime: Double, decodeTime: Double, ttftMs: Double,
        perTokenWallclock: [Double],
        memTracker: PhaseMemoryTracker, caches: [any LayerCacheProtocol],
        weightsBytes: Int, splitTokens: ThinkingSplit.Split?,
        genPerplexity: Double? = nil
    ) -> GenerationStats {
        let steady: Double? = {
            guard perTokenWallclock.count > 10 else { return nil }
            let tail = perTokenWallclock.dropFirst(10)
            let totalS = tail.reduce(0, +)
            return totalS > 0 ? Double(tail.count) / totalS : nil
        }()

        return GenerationStats(
            promptTokens: promptTokens.count,
            generatedTokens: generatedCount,
            contextSize: contextSize,
            prefillTimeS: prefillTime,
            decodeTimeS: decodeTime,
            timeToFirstTokenMs: ttftMs,
            steadyTokensPerSecond: steady,
            baselineGPUBytes: memTracker.baseline.gpuBytes,
            postPrefillGPUBytes: memTracker.postPrefill?.gpuBytes ?? memTracker.baseline.gpuBytes,
            postDecodeGPUBytes: memTracker.postDecode?.gpuBytes ?? memTracker.baseline.gpuBytes,
            prefillPeakGPUBytes: memTracker.prefillPeakBytes,
            decodePeakGPUBytes: memTracker.decodePeakBytes,
            wiredTicketBytes: memTracker.baseline.wiredTicketBytes,
            weightsBytes: weightsBytes,
            kvCacheAllocatedBytes: caches.totalBytesAllocated,
            kvCacheUsedBytes: caches.totalBytesInUse,
            thinkPerplexity: nil,
            genPerplexity: genPerplexity,
            thinkKLDivergence: nil,
            genKLDivergence: nil,
            thinkTokenCount: splitTokens.map { $0.thinkTokens.count },
            genTokenCount: splitTokens.map { $0.genTokens.count }
        )
    }
}

public enum GenerationError: Error, CustomStringConvertible {
    case streamEndedWithoutFinalChunk
    /// D9: `promptTokens + maxNewTokens` exceeds the KV cache's fixed
    /// physical capacity (`engine.maxSeq`). See the fail-fast check in
    /// `driveGeneration` for when this fires (and when it doesn't —
    /// exempt only when EVERY attention-cache layer uses `.window`
    /// eviction; a hybrid engine with even one `.unbounded` layer is
    /// still checked).
    case contextOverflow(promptTokens: Int, maxNewTokens: Int, maxContextLength: Int)
    public var description: String {
        switch self {
        case .streamEndedWithoutFinalChunk:
            return "GenerationStream finished without yielding the final stats chunk"
        case .contextOverflow(let promptTokens, let maxNewTokens, let maxContextLength):
            let total = promptTokens + maxNewTokens
            return
                "prompt (\(promptTokens) tokens) + max-tokens (\(maxNewTokens)) = "
                + "\(total) exceeds the KV cache capacity (\(maxContextLength) positions). "
                + "Raise --max-context to fit the full request, or pass --kv-window-size "
                + "to run with a bounded sliding-window cache instead."
        }
    }
}
