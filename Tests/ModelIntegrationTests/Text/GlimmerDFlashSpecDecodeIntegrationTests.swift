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
// GlimmerDFlashSpecDecodeIntegrationTests — the byte-identical
// EXACTNESS gate on Muse Glimmer 30B's DFlash block-speculative decode
// path (see `DFlashSpecDecode.swift`'s header).
//
// The gate: greedy decode is deterministic, so the DFlash speculative
// path MUST emit the exact same token stream as the plain single-token
// greedy loop, on every prompt — in BOTH the original fixed-block
// configuration AND the adaptive-control configuration
// (BUTTER_GLIMMER_DFLASH_PMIN + BUTTER_GLIMMER_DFLASH_ADAPTIVE, see
// `DFlashSpecDecode.swift`'s ADAPTIVE CONTROL header section). Also
// reports acceptance rate + decode tok/s for all three arms
// (informational — Meta's own reference reports ~1.8x on this exact
// model/hardware pairing at block_size=16, though that's on templated
// text; the adaptive arm exists specifically to fix the fixed-block
// arm's measured worst-case NET SLOWDOWN on open-ended prompts).
//
// 5 prompts: the original 3 (short continuation, templated
// multi-topic, codeish) plus 2 adversarial ones added for the adaptive
// wave — a high-entropy creative prompt and a long factual list — which
// together span the acceptance-rate range from the templated ceiling
// down to the open-ended floor.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

private let glimmerModelPath = "/Users/tom/models/muse-glimmer-30b-4bit"
private let glimmerDFlashPath = "/Users/tom/models/muse-glimmer-30b-dflash"
private let glimmerModelAvailable =
    FileManager.default.fileExists(
        atPath: "\(glimmerModelPath)/model.safetensors.index.json")
private let glimmerDFlashAvailable =
    FileManager.default.fileExists(atPath: "\(glimmerDFlashPath)/model.safetensors")

@Suite(
    "DFlash spec decode vs baseline greedy (Muse Glimmer 30B)", .serialized,
    .enabled(if: modelTestsOptedIn),
    .enabled(if: glimmerModelAvailable, "requires \(glimmerModelPath)"),
    .enabled(if: glimmerDFlashAvailable, "requires the DFlash drafter at \(glimmerDFlashPath)")
)
struct GlimmerDFlashSpecDecodeIntegrationTests {

    /// Adaptive-control knobs shared by every arm/test in this suite —
    /// see DFlashSpecDecode.swift's file header (ADAPTIVE CONTROL
    /// section) for the rationale, and its EXACTNESS DISCIPLINE section
    /// for why `lo`/`hi` were retuned well above their original
    /// 0.45/0.70: once every cycle unconditionally replays (the
    /// bail-bug fix), a drafting cycle's cost is ALWAYS verify-overhead
    /// PLUS a full plain-decode-equivalent replay of whatever gets
    /// committed — there is no more "full accept is free" fast path —
    /// so a cycle only pays for itself when its committed length
    /// amortizes that overhead, which the measured per-arm timings in
    /// dflash_fix_run.log (12-tok's 81.9%-accept 1.43x win vs
    /// 150-tok/60-tok's 68.9%/73.8%-accept 0.52x/0.72x LOSSES at
    /// blockSize=16) show only holds at raw acceptance rates well
    /// above the old lo=0.45/hi=0.70 band.
    private let adaptiveEnv: [String: String] = [
        "BUTTER_GLIMMER_DFLASH_PMIN": "0.7",
        "BUTTER_GLIMMER_DFLASH_ADAPTIVE": "1",
        "BUTTER_GLIMMER_DFLASH_ADAPT_WINDOW": "24",
        "BUTTER_GLIMMER_DFLASH_ADAPT_LO": "0.75",
        "BUTTER_GLIMMER_DFLASH_ADAPT_HI": "0.90",
        "BUTTER_GLIMMER_DFLASH_BAIL_TOKENS": "32",
    ]

    /// Chunked prefill (`forwardMulti`, all but the last prompt token)
    /// + one `forwardOneWithFeatures` call for the LAST prompt token —
    /// gives us the seed token AND the `targetLayerIDs` features DFlash
    /// needs to condition its very first draft block on, without
    /// double-processing that last token.
    private func prefill(
        _ glimmer: MuseGlimmerModel, promptTokens: [Int], targetLayerIDs: [Int],
        caches: [any LayerCacheProtocol], device: Device
    ) -> (seed: Int, features: Tensor) {
        precondition(!promptTokens.isEmpty)
        let bulk = Array(promptTokens.dropLast())
        var pos = 0
        if !bulk.isEmpty {
            let chunkSize = glimmer.defaultPrefillStepSize
            while pos < bulk.count {
                let end = min(pos + chunkSize, bulk.count)
                // `autoreleasepool` per chunk — Swift concurrency's
                // cooperative-executor threads (this whole test runs
                // inside an `async` test function) have NO implicit
                // per-iteration autorelease-pool drain the way a
                // traditional run-loop-driven thread would; without an
                // explicit pool here, every chunk's transient ObjC-
                // bridged Metal objects (MTLCommandBuffer, encoders,
                // etc.) accumulate for the whole PROCESS lifetime, not
                // just this loop — see `ffai-metal-autoreleasepool-leak`
                // / `DFlashSpecDecode.swift`'s EXACTNESS DISCIPLINE
                // section for the full incident writeup.
                autoreleasepool {
                    // `forwardMulti` self-commits+waits its OWN internal
                    // command buffer and never touches `cmd` — do NOT
                    // commit/wait `cmd` here (see that function's doc
                    // comment; double-committing wedged the Metal queue
                    // — see this suite's file header history).
                    let cmd = device.makeCommandBuffer()
                    _ = glimmer.forwardMulti(
                        tokenIds: Array(bulk[pos ..< end]), startingAt: pos,
                        caches: caches, on: cmd, device: device)
                }
                pos = end
            }
        }
        return autoreleasepool { () -> (seed: Int, features: Tensor) in
            // Same self-commit contract as `forwardMulti` above — see
            // `forwardOneWithFeatures`'s doc comment.
            let cmd = device.makeCommandBuffer()
            let (logits, features) = glimmer.forwardOneWithFeatures(
                tokenId: promptTokens.last!, position: pos, targetLayerIDs: targetLayerIDs,
                caches: caches, on: cmd, device: device)
            return (argmaxHost(logits), features)
        }
    }

    /// Baseline greedy stream of `count` tokens starting from `seed`.
    private func baselineGreedy(
        _ glimmer: MuseGlimmerModel, seed: Int, position: Int, count: Int,
        caches: [any LayerCacheProtocol], device: Device
    ) -> (tokens: [Int], decodeSeconds: Double) {
        var tokens: [Int] = []
        var prev = seed
        var pos = position
        let t0 = Date()
        for _ in 0 ..< count {
            tokens.append(prev)
            // `autoreleasepool` per token — see `prefill`'s comment
            // above for why this matters on Swift concurrency's
            // cooperative executor. This loop runs `count` times per
            // arm (80 in the full gate) — the single highest-volume
            // Metal-object-creation site of the baseline arm.
            let nextTok = autoreleasepool { () -> Int in
                // `forward(tokenId:...)` self-commits+waits its own
                // private buffer too — do NOT commit/wait `cmd` (same
                // contract as `forwardMulti`/`forwardOneWithFeatures`
                // above).
                let cmd = device.makeCommandBuffer()
                let logits = glimmer.forward(
                    tokenId: prev, position: pos, caches: caches, on: cmd, device: device)
                return argmaxHost(logits)
            }
            prev = nextTok
            pos += 1
        }
        return (tokens, Date().timeIntervalSince(t0))
    }

    /// Run one DFlash spec-decode arm (fixed-block when `envVars` is
    /// empty, adaptive-control when it carries the
    /// BUTTER_GLIMMER_DFLASH_* knobs) against a fresh cache/prompt pair
    /// and return the token stream + stats. Env vars are set for the
    /// duration of the call and restored (unset) afterward — `ButterEnv`
    /// reads live via `getenv(3)`, so this toggles the feature without
    /// needing a subprocess per arm.
    private func runDFlashArm(
        _ glimmer: MuseGlimmerModel, draft: DFlashDraftModel,
        promptTokens: [Int], targetLayerIDs: [Int], count: Int,
        envVars: [String: String], device: Device
    ) -> (tokens: [Int], stats: DFlashSpecDecodeStats) {
        for (k, v) in envVars { setenv(k, v, 1) }
        defer { for k in envVars.keys { unsetenv(k) } }

        // Cross-arm retention fix (see `Device.releaseGenerationCaches`):
        // drop residency for whichever arm's caches were built last
        // before allocating this arm's. Each arm's `caches`/`draftCaches`
        // are local to this function and go out of scope at return, so
        // by the time the NEXT arm calls this again their only
        // remaining strong reference was the (now-released) residency
        // set — this is what makes the 15-arm/5-prompt gate flat instead
        // of growing ~6.6GB/arm.
        device.releaseGenerationCaches()
        let caches = glimmer.makeLayerCaches(maxSeq: nil, device: device)
        let (seed, seedFeatures) = prefill(
            glimmer, promptTokens: promptTokens, targetLayerIDs: targetLayerIDs,
            caches: caches, device: device)
        let draftCaches = draft.makeCache(maxSeq: 4096, device: device)
        var history = promptTokens
        let stats = DFlashSpecDecode.generateGreedy(
            model: glimmer, draft: draft, draftCaches: draftCaches,
            lastToken: seed, position: promptTokens.count, seedFeatures: seedFeatures,
            caches: caches, history: &history,
            maxNewTokens: count, device: device)
        return (Array(history[promptTokens.count...]), stats)
    }

    // Bail-then-reenter regression: the creative-highentropy prompt is
    // the smallest/cheapest natural trigger for the adaptive
    // controller's bail path (see DFlashSpecDecode.swift's file header,
    // EXACTNESS DISCIPLINE section, for the root cause this guards
    // against — a batched-vs-single-token-kernel ULP drift that a
    // "trust the batch on full accept" shortcut used to let leak into
    // the cache, discovered via exactly this bail-then-reenter shape).
    // 40 tokens (not the full suite's 80) is enough to walk through
    // several drafting cycles, ONE full bail span, AND re-entry —
    // proven sufficient to reproduce/verify the original failure — at
    // roughly a tenth of the full 5-prompt gate's wall-clock cost, so
    // this can run standalone as a fast, targeted regression check.
    @Test("DFlash adaptive bail-then-reenter == baseline greedy (fast, targeted)")
    func adaptiveBailThenReenterMatchesBaseline() async throws {
        let device = Device.shared
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder

        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(glimmerModelPath, options: opts)
        }
        let glimmer = try #require(m.museGlimmer, "expected MuseGlimmerModel engine")
        let draft = try DFlashDraftModel.loadModel(path: glimmerDFlashPath, device: device)
        let targetLayerIDs = draft.config.targetLayerIDs
        let count = 40

        let promptTokens = m.tokenizer.encode(
            text: """
                Write a short surreal story, about 150 words, about a
                lighthouse keeper who discovers the ocean itself is
                alive and dreaming. Make it strange, unpredictable, and
                richly imagistic — avoid clichés and generic phrasing.
                """)

        print("physFootprint before baseline arm=\(String(format: "%.2f", physFootprintGB()))GB")
        let baseline = autoreleasepool {
            () -> (tokens: [Int], decodeSeconds: Double) in
            // See `runDFlashArm`'s matching call — this arm doesn't go
            // through that helper, so it needs its own release before
            // building a fresh cache set (`Device.releaseGenerationCaches`).
            device.releaseGenerationCaches()
            let baseCaches = glimmer.makeLayerCaches(maxSeq: nil, device: device)
            let (baseSeed, _) = prefill(
                glimmer, promptTokens: promptTokens, targetLayerIDs: targetLayerIDs,
                caches: baseCaches, device: device)
            return baselineGreedy(
                glimmer, seed: baseSeed, position: promptTokens.count, count: count,
                caches: baseCaches, device: device)
        }
        print("physFootprint after baseline arm=\(String(format: "%.2f", physFootprintGB()))GB")

        let (adaptiveTokens, stats) = autoreleasepool {
            runDFlashArm(
                glimmer, draft: draft, promptTokens: promptTokens,
                targetLayerIDs: targetLayerIDs, count: count, envVars: adaptiveEnv, device: device)
        }
        print("physFootprint after adaptive arm=\(String(format: "%.2f", physFootprintGB()))GB")

        let matchPrefix = zip(baseline.tokens, adaptiveTokens).prefix(while: { $0 == $1 }).count
        print(
            "DFlash bail-regression: exact_match=\(adaptiveTokens == baseline.tokens) "
                + "prefix=\(matchPrefix)/\(count) bailSteps=\(stats.bailSteps) "
                + "bailEntries=\(stats.bailEntries)")
        #expect(
            stats.bailEntries > 0,
            "this regression test is only meaningful if it actually exercises the bail path — it didn't; the adaptive thresholds may have drifted"
        )
        #expect(
            adaptiveTokens == baseline.tokens,
            "bail-then-reenter diverged from baseline at prefix \(matchPrefix)/\(count)")
    }

    // Cross-arm retention MINIMAL REPRO — see `Device.markCacheResident`/
    // `releaseGenerationCaches` in Device.swift for the fix this
    // verifies. Runs the SAME arm (fixed-block DFlash spec-decode
    // against the SAME prompt) `reps` times in one process and prints
    // phys_footprint before/after each repetition. This isolates the
    // leak's SIGNATURE from any prompt- or arm-type-dependence: the
    // original 15-arm gate varied both prompt and arm kind per
    // iteration, which left open whether the growth was keyed on the
    // (prompt, arm) pair or was arm-count-only. Root cause was the
    // opposite of prompt-keyed — the leak is `Device.shared`'s weight
    // MTLResidencySet gaining one KV/GDN/conv cache buffer set per
    // `makeLayerCaches()` call with no matching removal, so ANY repeated
    // cache allocation reproduces it, same-arm-repeated included. Before
    // the fix this test's footprint delta grows ~linearly rep over rep;
    // after, it is flat once warmup settles (reps 0-1).
    @Test("cross-arm retention: same arm repeated 6x stays footprint-flat")
    func sameArmRepeatedStaysFootprintFlat() async throws {
        let device = Device.shared
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder

        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(glimmerModelPath, options: opts)
        }
        let glimmer = try #require(m.museGlimmer, "expected MuseGlimmerModel engine")
        let draft = try DFlashDraftModel.loadModel(path: glimmerDFlashPath, device: device)
        let targetLayerIDs = draft.config.targetLayerIDs
        let count = 16
        let reps = 6

        let promptTokens = m.tokenizer.encode(
            text: "The quick brown fox jumps over the lazy dog and then")

        var footprints: [Double] = []
        var lastTokens: [Int]?
        for i in 0 ..< reps {
            let before = physFootprintGB()
            let (tokens, _) = autoreleasepool {
                runDFlashArm(
                    glimmer, draft: draft, promptTokens: promptTokens,
                    targetLayerIDs: targetLayerIDs, count: count, envVars: [:], device: device)
            }
            let after = physFootprintGB()
            footprints.append(after)
            print(
                "[repeat-arm] rep\(i): before=\(String(format: "%.3f", before))GB "
                    + "after=\(String(format: "%.3f", after))GB "
                    + "delta=\(String(format: "%+.3f", after - before))GB")
            // Determinism sanity: greedy decode, same prompt, same arm —
            // every repetition must emit the exact same tokens. A
            // mismatch here would mean cross-arm STATE bleed (a
            // correctness bug), not just a memory leak — the residency
            // fix only touches which residency set buffers are pinned
            // in, never the buffers' contents, so this must keep
            // passing.
            if let last = lastTokens {
                #expect(
                    tokens == last,
                    "rep\(i): token stream diverged from rep\(i - 1) — cross-arm state bleed, not just a memory leak"
                )
            }
            lastTokens = tokens
        }

        // Leak-shape check, same discipline as
        // `Gemma4MoEAllocTraceTests.perTokenStability`: a leak
        // accumulates roughly monotonically across reps, so tail-vs-head
        // growth is large relative to noise. Skip reps 0-1 (model/kernel
        // warmup: PSO compile, first-touch paging) and compare the mean
        // of the last 2 reps against the mean of reps 2-3.
        let head = Array(footprints[2 ..< 4])
        let tail = Array(footprints[(reps - 2)...])
        let headMean = head.reduce(0, +) / Double(head.count)
        let tailMean = tail.reduce(0, +) / Double(tail.count)
        print(
            "[repeat-arm] headMean(reps 2-3)=\(String(format: "%.3f", headMean))GB "
                + "tailMean(reps \(reps - 2)-\(reps - 1))=\(String(format: "%.3f", tailMean))GB "
                + "growth=\(String(format: "%+.3f", tailMean - headMean))GB")
        let growthMessage =
            "footprint grew \(tailMean - headMean)GB from reps 2-3 to reps \(reps - 2)-\(reps - 1) — cross-arm retention regression (see Device.releaseGenerationCaches)"
        #expect(tailMean - headMean < 1.0, "\(growthMessage)")
    }

    /// Optional single-prompt selector for the 5-prompt gate below — set
    /// to an index 0-4 (`BUTTER_DFLASH_GATE_PROMPT=<n>`) to run just
    /// that one prompt (still all 3 arms) as its own process; unset runs
    /// the full 5-prompt sweep, preserving prior behavior. Exists
    /// because 5 prompts × 3 arms in a single process leaks ~6.6GB of
    /// physical footprint PER ARM (retention across arms — root cause
    /// not yet found, separate investigation filed): 15 arms ≈ +100GB,
    /// which wedges the Metal queue (submit thread stuck in
    /// `iokit_user_client_trap`) around the 50-70 min mark. Splitting
    /// into 5 sequential single-prompt processes keeps each run's
    /// footprint and wall-clock bounded until the leak itself is fixed.
    private var gatePromptIndex: Int? {
        guard let raw = ProcessInfo.processInfo.environment["BUTTER_DFLASH_GATE_PROMPT"] else {
            return nil
        }
        guard let idx = Int(raw) else {
            preconditionFailure("BUTTER_DFLASH_GATE_PROMPT=\(raw) is not an integer")
        }
        return idx
    }

    @Test(
        "DFlash spec (fixed-block AND adaptive-control) == baseline greedy: 5 prompts, 80 new tokens each"
    )
    func dflashSpecMatchesBaselineExactly() async throws {
        let device = Device.shared
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder

        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(glimmerModelPath, options: opts)
        }
        let glimmer = try #require(m.museGlimmer, "expected MuseGlimmerModel engine")
        let draft = try DFlashDraftModel.loadModel(path: glimmerDFlashPath, device: device)

        let allPrompts: [(name: String, text: String)] = [
            ("12-tok", "The quick brown fox jumps over"),
            (
                "150-tok-multitopic",
                """
                Give a brief overview of three unrelated topics: first,
                explain how photosynthesis converts light energy into
                chemical energy in plant cells, mentioning chlorophyll
                and the role of chloroplasts. Second, describe the
                economic causes that are commonly cited for the fall of
                the Roman Empire, including currency debasement and
                overextension of the military. Third, explain at a high
                level how a hash table achieves average O(1) lookup
                time, mentioning collision resolution strategies like
                chaining and open addressing. Keep each explanation to
                a couple of sentences.
                """
            ),
            (
                "60-tok-codeish",
                """
                Write a Python function called `is_prime` that takes an
                integer `n` and returns True if `n` is prime and False
                otherwise. Include a docstring and handle edge cases
                for n less than 2.
                """
            ),
            (
                "creative-highentropy",
                """
                Write a short surreal story, about 150 words, about a
                lighthouse keeper who discovers the ocean itself is
                alive and dreaming. Make it strange, unpredictable, and
                richly imagistic — avoid clichés and generic phrasing.
                """
            ),
            (
                "factual-list",
                """
                List 25 distinct countries and their capital cities,
                one pair per line, formatted as "Country: Capital".
                Cover a mix of continents and avoid repeating any
                country.
                """
            ),
        ]
        let prompts: [(name: String, text: String)]
        if let idx = gatePromptIndex {
            precondition(
                allPrompts.indices.contains(idx),
                "BUTTER_DFLASH_GATE_PROMPT=\(idx) out of range 0..<\(allPrompts.count)")
            prompts = [allPrompts[idx]]
        } else {
            prompts = allPrompts
        }
        let count = 80
        let targetLayerIDs = draft.config.targetLayerIDs

        for (name, text) in prompts {
            let promptTokens = m.tokenizer.encode(text: text)
            let n = promptTokens.count
            print(
                "DFlash prompt=\(name) tokens=\(n) "
                    + "physFootprint=\(String(format: "%.2f", physFootprintGB()))GB")

            // ── Baseline arm ────────────────────────────────────────
            // `autoreleasepool` per ARM (not just per inner loop) — see
            // `prefill`'s comment for the cooperative-executor
            // rationale. 15 arms run in this suite (5 prompts × 3
            // arms); each allocates a fresh `caches` array (KVCache
            // buffers for all 52 layers) that should be fully released,
            // Metal-side, before the NEXT arm starts, not just when the
            // Swift reference count hits zero on a thread with no
            // implicit drain point.
            let (baseline, baseTps) = autoreleasepool {
                () -> ((tokens: [Int], decodeSeconds: Double), Double) in
                // Cross-arm retention fix — see `runDFlashArm`'s matching
                // call and `Device.releaseGenerationCaches`'s doc comment.
                // This is what makes the "should be fully released"
                // comment above actually true, instead of aspirational.
                device.releaseGenerationCaches()
                let baseCaches = glimmer.makeLayerCaches(maxSeq: nil, device: device)
                let (baseSeed, _) = prefill(
                    glimmer, promptTokens: promptTokens, targetLayerIDs: targetLayerIDs,
                    caches: baseCaches, device: device)
                let result = baselineGreedy(
                    glimmer, seed: baseSeed, position: n, count: count,
                    caches: baseCaches, device: device)
                return (result, Double(count) / result.decodeSeconds)
            }
            print(
                "DFlash baseline \(name): "
                    + "\(String(format: "%.2f", baseTps)) tok/s "
                    + "physFootprint=\(String(format: "%.2f", physFootprintGB()))GB")

            // ── DFlash fixed-block arm (original, no adaptive knobs) ──
            let (fixedTokens, fixedStats) = autoreleasepool {
                runDFlashArm(
                    glimmer, draft: draft, promptTokens: promptTokens,
                    targetLayerIDs: targetLayerIDs, count: count, envVars: [:], device: device)
            }
            #expect(fixedStats.tokensGenerated == count)
            let fixedMatchPrefix = zip(baseline.tokens, fixedTokens).prefix(while: { $0 == $1 }).count
            #expect(
                fixedTokens == baseline.tokens,
                "\(name): DFlash FIXED spec stream diverged from baseline (prefix \(fixedMatchPrefix)/\(count))"
            )
            let fixedTps = fixedStats.tps
            print(
                "DFlash-fixed \(name): exact_match=\(fixedTokens == baseline.tokens) "
                    + "tps=\(String(format: "%.2f", fixedTps)) "
                    + "(\(String(format: "%.2fx", fixedTps / baseTps)) vs baseline), "
                    + "accepted=\(fixedStats.candidatesAccepted)/\(fixedStats.candidatesProposed) "
                    + "(\(String(format: "%.1f", fixedStats.acceptanceRate * 100))%), "
                    + "verify=\(fixedStats.verifySteps), replay=\(fixedStats.replaySingleSteps) "
                    + "physFootprint=\(String(format: "%.2f", physFootprintGB()))GB")

            // ── DFlash adaptive-control arm ───────────────────────────
            let (adaptiveTokens, adaptiveStats) = autoreleasepool {
                runDFlashArm(
                    glimmer, draft: draft, promptTokens: promptTokens,
                    targetLayerIDs: targetLayerIDs, count: count, envVars: adaptiveEnv,
                    device: device)
            }
            #expect(adaptiveStats.tokensGenerated == count)
            let adaptiveMatchPrefix =
                zip(baseline.tokens, adaptiveTokens).prefix(while: { $0 == $1 }).count
            #expect(
                adaptiveTokens == baseline.tokens,
                "\(name): DFlash ADAPTIVE spec stream diverged from baseline (prefix \(adaptiveMatchPrefix)/\(count))"
            )
            let adaptiveTps = adaptiveStats.tps
            print(
                "DFlash-adaptive \(name): exact_match=\(adaptiveTokens == baseline.tokens) "
                    + "tps=\(String(format: "%.2f", adaptiveTps)) "
                    + "(\(String(format: "%.2fx", adaptiveTps / baseTps)) vs baseline), "
                    + "accepted=\(adaptiveStats.candidatesAccepted)/\(adaptiveStats.candidatesProposed) "
                    + "(\(String(format: "%.1f", adaptiveStats.acceptanceRate * 100))%), "
                    + "verify=\(adaptiveStats.verifySteps), replay=\(adaptiveStats.replaySingleSteps), "
                    + "avgBlock=\(String(format: "%.2f", adaptiveStats.avgBlockSize)), "
                    + "truncated=\(adaptiveStats.truncatedCycles), "
                    + "bailSteps=\(adaptiveStats.bailSteps), bailEntries=\(adaptiveStats.bailEntries) "
                    + "physFootprint=\(String(format: "%.2f", physFootprintGB()))GB")
        }
    }
}

@inline(__always)
private func argmaxHost(_ logits: Tensor) -> Int {
    let host = logits.toFloatArray()
    var bestIdx = 0
    var bestVal = host[0]
    for i in 1 ..< host.count {
        if host[i] > bestVal {
            bestVal = host[i]
            bestIdx = i
        }
    }
    return bestIdx
}

/// This process's `phys_footprint` in GiB — the same figure Instruments
/// / `sample`'s "Physical footprint" line reports, and (unlike RSS)
/// what actually accounts for Metal/IOSurface GPU-resident allocations
/// on Apple Silicon's unified memory. Printed between arms so a gate
/// log shows whether footprint stays flat (autoreleasepool coverage
/// working) or climbs across arms (a leak) — see
/// `DFlashSpecDecode.swift`'s EXACTNESS DISCIPLINE section for the
/// incident this instruments.
private func physFootprintGB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard result == KERN_SUCCESS else { return -1 }
    return Double(info.phys_footprint) / 1_073_741_824.0
}
