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
// MTPSpecDecodeIntegrationTests — spec 030 phase 2 EXACTNESS gate on
// the 27B checkpoint (mtp-sidecar auto-attaches at load).
//
// The gate: greedy decode is deterministic, so the MTP speculative
// path MUST emit the exact same token stream as the plain single-token
// greedy loop — the accepted-prefix replay discipline makes the
// emitted tokens (and the KV/GDN contents backing them) come from the
// same single-token forward path in both arms. This holds regardless
// of backbone output quality (both arms share the backbone), so the
// gate is valid even while the dense-backbone degeneration (separate
// investigation) is open.
//
// Also reports acceptance rate + decode tok/s per γ (informational —
// degenerate/repetitive output can inflate acceptance; not tuned here).

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

private let mtpModelPath = "/Users/tom/models/qwen36-27b-4bit-mlx"
private let mtpModelAvailable =
    FileManager.default.fileExists(
        atPath: "\(mtpModelPath)/model.safetensors.index.json")
    && FileManager.default.fileExists(atPath: "\(mtpModelPath)/mtp-sidecar")

@Suite(
    "MTP spec decode vs baseline greedy (27B)", .serialized,
    .enabled(if: modelTestsOptedIn),
    .enabled(
        if: mtpModelAvailable,
        "requires the 27B checkpoint + mtp-sidecar at \(mtpModelPath)")
)
struct MTPSpecDecodeIntegrationTests {

    /// Chunked prefill (the same `forwardMulti` path
    /// `Generate.driveGeneration` uses) + argmax of the tail logits →
    /// the seed token both arms start from.
    private func prefill(
        _ qwen: Qwen35Model, promptTokens: [Int],
        caches: [any LayerCacheProtocol], device: Device
    ) -> Int {
        let chunkSize = qwen.defaultPrefillStepSize
        var pos = 0
        var lastLogits: Tensor!
        while pos < promptTokens.count {
            let end = min(pos + chunkSize, promptTokens.count)
            let cmd = device.makeCommandBuffer()
            lastLogits = qwen.forwardMulti(
                tokenIds: Array(promptTokens[pos ..< end]), startingAt: pos,
                caches: caches, on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            pos = end
        }
        return argmaxHost(lastLogits)
    }

    /// Baseline greedy stream of `count` tokens starting from `seed`
    /// (the stream `Generate.driveGeneration`'s fallback loop emits):
    /// emit token, single-token forward, argmax, repeat.
    private func baselineGreedy(
        _ qwen: Qwen35Model, seed: Int, position: Int, count: Int,
        caches: [any LayerCacheProtocol], device: Device
    ) -> (tokens: [Int], decodeSeconds: Double) {
        var tokens: [Int] = []
        var prev = seed
        var pos = position
        let t0 = Date()
        for _ in 0 ..< count {
            tokens.append(prev)
            let cmd = device.makeCommandBuffer()
            let logits = qwen.forward(
                tokenId: prev, position: pos,
                caches: caches, on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            prev = argmaxHost(logits)
            pos += 1
        }
        return (tokens, Date().timeIntervalSince(t0))
    }

    @Test("spec on == spec off: 120 tokens, γ ∈ {1, 2, 4}; 300 tokens at γ = 2")
    func mtpSpecMatchesBaselineExactly() async throws {
        let device = Device.shared
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(mtpModelPath, options: opts)
        }
        let qwen = try #require(m.qwen35, "expected Qwen35Model engine")
        let head = try #require(
            qwen.mtpHead, "expected the mtp-sidecar to attach an MTP head")

        let prompt = """
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n - 1) + fibonacci(n - 2)

            def main():
            """
        let promptTokens = m.tokenizer.encode(text: prompt)
        let n = promptTokens.count
        print("MTPSpec prompt=\(n) tokens")

        // Runs: (maxNewTokens, gamma). 120-token exactness across γ,
        // plus one long 300-token run.
        let runs: [(count: Int, gamma: Int)] = [
            (120, 1), (120, 2), (120, 4), (300, 2),
        ]

        // ── Baseline arm per length (compute once per count) ──────────
        var baselines: [Int: (tokens: [Int], decodeSeconds: Double)] = [:]
        for count in Set(runs.map(\.count)) {
            // Cross-arm retention fix — drop the previous iteration's
            // cache residency before allocating a fresh one; see
            // `Device.releaseGenerationCaches` for the full writeup.
            device.releaseGenerationCaches()
            let caches = qwen.makeLayerCaches()
            let seed = prefill(
                qwen, promptTokens: promptTokens, caches: caches, device: device)
            baselines[count] = baselineGreedy(
                qwen, seed: seed, position: n, count: count,
                caches: caches, device: device)
            let b = baselines[count]!
            print(
                "MTPSpec baseline count=\(count): "
                    + "\(String(format: "%.2f", Double(count) / b.decodeSeconds)) tok/s")
        }

        // ── Spec arm per (count, γ) ───────────────────────────────────
        for (count, gamma) in runs {
            let baseline = baselines[count]!
            device.releaseGenerationCaches()
            let caches = qwen.makeLayerCaches()
            let seed = prefill(
                qwen, promptTokens: promptTokens, caches: caches, device: device)
            #expect(
                seed == baseline.tokens[0],
                "prefill seed must match between arms")

            head.resetCache()
            var history = promptTokens
            let t0 = Date()
            let stats = SpecDecode.generateGreedyMTP(
                model: qwen, head: head, gamma: gamma,
                lastToken: seed, position: n,
                caches: caches, history: &history,
                maxNewTokens: count, device: device)
            let specSeconds = Date().timeIntervalSince(t0)

            let specTokens = Array(history[n...])
            #expect(stats.tokensGenerated == count)
            #expect(specTokens.count == count)

            // ── THE EXACTNESS GATE ────────────────────────────────────
            let matchPrefix = zip(baseline.tokens, specTokens)
                .prefix(while: { $0 == $1 }).count
            let baseCtx = Array(
                baseline.tokens.dropFirst(max(0, matchPrefix - 2)).prefix(6))
            let specCtx = Array(
                specTokens.dropFirst(max(0, matchPrefix - 2)).prefix(6))
            #expect(
                specTokens == baseline.tokens,
                "γ=\(gamma) count=\(count): spec stream diverged from baseline at index \(matchPrefix) — baseline \(baseCtx) vs spec \(specCtx)"
            )

            let specTps = Double(count) / specSeconds
            let baseTps = Double(count) / baseline.decodeSeconds
            print(
                "MTPSpec γ=\(gamma) count=\(count): exact_match=\(specTokens == baseline.tokens) "
                    + "(prefix \(matchPrefix)/\(count)), "
                    + "tps=\(String(format: "%.2f", specTps)) "
                    + "(\(String(format: "%.2fx", specTps / baseTps)) vs baseline), "
                    + "accepted=\(stats.candidatesAccepted)/\(stats.candidatesProposed) "
                    + "(\(String(format: "%.1f", stats.acceptanceRate * 100))%), "
                    + "verify=\(stats.verifySteps), replay=\(stats.replaySingleSteps), "
                    + "lockstep=\(stats.lockstepSteps)")
        }
    }

    // ── Adversarial exactness coverage ─────────────────────────────────
    //
    // `mtpSpecMatchesBaselineExactly` above only exercises ONE prompt —
    // a low-entropy fibonacci continuation, where the backbone is
    // extremely confident at nearly every position. That is exactly the
    // condition under which two kernel paths (the batched
    // `forwardManyHiddenAndLogits` verify pass vs. the single-token
    // `forward` decode pass) are LEAST likely to disagree: a wide
    // top1/top2 logit margin swallows small ULP-level drift before it
    // can flip an argmax. It cannot rule out drift on lower-margin,
    // higher-entropy content — the exact failure mode confirmed on Muse
    // Glimmer's DFlash path (see `DFlashSpecDecode.swift`'s file header,
    // EXACTNESS DISCIPLINE section): full accept there trusted the
    // batched pass's cache writes directly, token-identity agreement on
    // full accept did NOT guarantee bit-identical cache, and accumulated
    // drift eventually flipped an argmax on a high-entropy creative
    // prompt. `MTPSpecDecode.swift`'s full-accept path used to inherit
    // the identical shortcut — and DID diverge from it, just not on
    // these 3 adversarial prompts: `mtpSpecMatchesBaselineExactly`
    // above caught it first, on its own low-entropy fibonacci prompt
    // (diverges at token 16 under the shipped γ ∈ {1, 2, 4} config —
    // the margin-based intuition in this comment's second paragraph is
    // about AVERAGE margin across a prompt, not a guarantee at every
    // position, and index 16 was evidently a tight one). Fixed by
    // routing greedy-mode full accepts through the same single-token
    // replay partial accepts already use — see `mtpDecodeLoop`'s
    // `mustReplayForExactness` — matching DFlash's discipline exactly.
    //
    // These 3 prompts mirror the adversarial half of
    // `GlimmerDFlashSpecDecodeIntegrationTests`'s 5-prompt matrix
    // (`creative-highentropy`, `150-tok-multitopic`, `factual-list` —
    // copied verbatim from that file so both gates are stressing the
    // same content) plus the two γ values most likely to differ in how
    // often the full-accept shortcut actually fires: γ=2 (the value
    // profiled for the shipped decode-speed figure) and γ=4 (a longer
    // burst gives ULP drift more positions to accumulate over per
    // full-accept cycle before the next verify pass resets against a
    // fresh snapshot). 180 new tokens per run — generous runway for
    // drift to compound into a flipped argmax, matching DFlash's
    // adversarial-prompt budget.
    //
    // Optional single-(prompt,gamma) selector, same shape as
    // `GlimmerDFlashSpecDecodeIntegrationTests`'s
    // `BUTTER_DFLASH_GATE_PROMPT` — set `BUTTER_MTP_GATE_CASE=<n>` to run
    // just one (prompt, gamma) pair as its own process. Exists for the
    // same reason: keeping a single process's model-load + KV-cache
    // footprint bounded when other model jobs may be sharing this box.
    private var gateCaseIndex: Int? {
        guard let raw = ProcessInfo.processInfo.environment["BUTTER_MTP_GATE_CASE"] else {
            return nil
        }
        guard let idx = Int(raw) else {
            preconditionFailure("BUTTER_MTP_GATE_CASE=\(raw) is not an integer")
        }
        return idx
    }

    // Prompt texts copied verbatim from
    // `GlimmerDFlashSpecDecodeIntegrationTests.dflashSpecMatchesBaselineExactly`'s
    // `allPrompts` — same adversarial content, different model/driver.
    // Shared between `mtpSpecMatchesBaselineOnAdversarialPrompts` and
    // `mtpSpecMatchesBaselineUnderBoostedRegime` below (same (prompt,
    // gamma) matrix — the boosted-regime gate only adds env-configured
    // PMIN/ADAPTIVE promotion configs on top).
    private static let adversarialPrompts: [(name: String, text: String)] = [
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
            "factual-list",
            """
            List 25 distinct countries and their capital cities,
            one pair per line, formatted as "Country: Capital".
            Cover a mix of continents and avoid repeating any
            country.
            """
        ),
    ]
    private static let adversarialGammas = [2, 4]

    @Test("spec on == spec off: adversarial high-entropy prompts, γ ∈ {2, 4}, 180 tokens")
    func mtpSpecMatchesBaselineOnAdversarialPrompts() async throws {
        let device = Device.shared
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(mtpModelPath, options: opts)
        }
        let qwen = try #require(m.qwen35, "expected Qwen35Model engine")
        let head = try #require(
            qwen.mtpHead, "expected the mtp-sidecar to attach an MTP head")

        let allPrompts = Self.adversarialPrompts
        let gammas = Self.adversarialGammas
        var allCases: [(prompt: (name: String, text: String), gamma: Int)] = []
        for p in allPrompts { for g in gammas { allCases.append((p, g)) } }

        let cases: [(prompt: (name: String, text: String), gamma: Int)]
        if let idx = gateCaseIndex {
            precondition(
                allCases.indices.contains(idx),
                "BUTTER_MTP_GATE_CASE=\(idx) out of range 0..<\(allCases.count)")
            cases = [allCases[idx]]
        } else {
            cases = allCases
        }
        let count = 180

        // Baseline is per-PROMPT (gamma-independent) — compute once and
        // reuse across both γ values for the same prompt.
        var baselines: [String: (tokens: [Int], decodeSeconds: Double)] = [:]
        for (name, text) in Set(cases.map { $0.prompt.name }).sorted().compactMap({ n in
            allPrompts.first(where: { $0.name == n })
        }) {
            let promptTokens = m.tokenizer.encode(text: text)
            // Cross-arm retention fix — see `Device.releaseGenerationCaches`.
            device.releaseGenerationCaches()
            let caches = qwen.makeLayerCaches()
            let seed = autoreleasepool {
                prefill(qwen, promptTokens: promptTokens, caches: caches, device: device)
            }
            baselines[name] = autoreleasepool {
                baselineGreedy(
                    qwen, seed: seed, position: promptTokens.count, count: count,
                    caches: caches, device: device)
            }
        }

        for (prompt, gamma) in cases {
            let promptTokens = m.tokenizer.encode(text: prompt.text)
            let n = promptTokens.count
            let baseline = baselines[prompt.name]!

            device.releaseGenerationCaches()
            let caches = qwen.makeLayerCaches()
            let seed = autoreleasepool {
                prefill(qwen, promptTokens: promptTokens, caches: caches, device: device)
            }
            #expect(
                seed == baseline.tokens[0],
                "\(prompt.name): prefill seed must match between arms")

            head.resetCache()
            var history = promptTokens
            let t0 = Date()
            let stats = autoreleasepool {
                SpecDecode.generateGreedyMTP(
                    model: qwen, head: head, gamma: gamma,
                    lastToken: seed, position: n,
                    caches: caches, history: &history,
                    maxNewTokens: count, device: device)
            }
            let specSeconds = Date().timeIntervalSince(t0)

            let specTokens = Array(history[n...])
            #expect(stats.tokensGenerated == count)
            #expect(specTokens.count == count)

            // ── THE EXACTNESS GATE ────────────────────────────────────
            let matchPrefix = zip(baseline.tokens, specTokens)
                .prefix(while: { $0 == $1 }).count
            let baseCtx = Array(
                baseline.tokens.dropFirst(max(0, matchPrefix - 2)).prefix(6))
            let specCtx = Array(
                specTokens.dropFirst(max(0, matchPrefix - 2)).prefix(6))
            #expect(
                specTokens == baseline.tokens,
                "\(prompt.name) γ=\(gamma) count=\(count): spec stream diverged from baseline at index \(matchPrefix)/\(count) — baseline \(baseCtx) vs spec \(specCtx)"
            )

            let specTps = Double(count) / specSeconds
            let baseTps = Double(count) / baseline.decodeSeconds
            print(
                "MTPSpec-adversarial \(prompt.name) γ=\(gamma) count=\(count): "
                    + "exact_match=\(specTokens == baseline.tokens) "
                    + "(prefix \(matchPrefix)/\(count)), "
                    + "tps=\(String(format: "%.2f", specTps)) "
                    + "(\(String(format: "%.2fx", specTps / baseTps)) vs baseline), "
                    + "accepted=\(stats.candidatesAccepted)/\(stats.candidatesProposed) "
                    + "(\(String(format: "%.1f", stats.acceptanceRate * 100))%), "
                    + "verify=\(stats.verifySteps), replay=\(stats.replaySingleSteps), "
                    + "lockstep=\(stats.lockstepSteps)")
        }
    }

    // ── Boosted-regime exactness coverage (phase 2) ──────────────────────
    //
    // `mtpSpecMatchesBaselineOnAdversarialPrompts` above proved the
    // SHIPPED config (fixed γ, no confidence truncation, no adaptive
    // gamma) holds exact despite real batched-vs-single-token argmax
    // drift on full-accept bonus positions (see `BUTTER_QWEN_MTP_PARITY`,
    // now side-effect-free per the KVCache range-snapshot fix this file's
    // sibling commit added). The working theory there: frequent
    // partial/zero-accept cycles force a single-token replay that resets
    // the cache before drift compounds, and the shipped config's
    // acceptance rates (40-73%) keep those resets frequent.
    //
    // `BUTTER_QWEN_MTP_PMIN` (confidence-gated draft truncation) and
    // `BUTTER_QWEN_MTP_ADAPTIVE` (trailing-acceptance gamma controller)
    // are exactly the two levers a throughput-chasing config would flip
    // to shrink bursts and raise full-accept frequency — the DANGER
    // regime, since DFlash (Muse Glimmer) diverged in precisely this
    // regime (see `DFlashSpecDecode.swift`'s EXACTNESS DISCIPLINE
    // header). This gate runs the same adversarial (prompt, gamma)
    // matrix under three plausible promotion configs someone chasing
    // decode speed would pick:
    //
    //   - `pmin-moderate` (BUTTER_QWEN_MTP_PMIN=0.8): empirically a
    //     NO-OP on this checkpoint — the MTP head's own top-1 probability
    //     saturates at (float) 1.0 on essentially every draft step, so
    //     no threshold below 1.0 (verified up to 0.999) ever trips the
    //     gate. Kept in the matrix as documentation of that finding
    //     rather than removed, since a future checkpoint or head
    //     retraining could un-saturate it.
    //   - `adaptive-default` (BUTTER_QWEN_MTP_ADAPTIVE=1, file defaults:
    //     window=32 lo=0.55 hi=0.70): raises full-accept frequency
    //     materially over the fixed-γ baseline on some (prompt, γ=4)
    //     cases (e.g. 24 → 52 full-accepts on creative-highentropy) by
    //     holding gamma near its cap once trailing acceptance clears hi.
    //   - `adaptive-aggressive` (lo=0.3 hi=0.4): pins gamma at its cap
    //     almost immediately (typical acceptance here is 45-73%, well
    //     above hi=0.4) — the largest-burst-per-full-accept regime this
    //     file can reach, maximizing how many drifted positions a single
    //     full-accept cycle can commit before the next verify pass.
    //
    // Audit result (2026-08-13, 27B checkpoint, all 3 configs × all 6
    // (prompt, γ) cases): EXACT MATCH throughout — 18/18 runs — despite
    // full-accept counts up to 56/89 cycles and per-run parity mismatch
    // rates up to 20% (margins 0.25-7.6, not all tie-breaks). The hazard
    // proven real by `BUTTER_QWEN_MTP_PARITY` (batched-vs-single-token
    // drift on full-accept bonus positions) stayed only THEORETICAL on
    // this specific matrix (the 3 adversarial prompts, all 3 promotion
    // configs): the same partial/zero-accept replay frequency that
    // protected the shipped config continued to reset the cache often
    // enough to interrupt drift before it compounded into a flipped
    // argmax, across every config tested here. It was NOT theoretical
    // in general — `mtpSpecMatchesBaselineExactly`'s own fixed-γ
    // fibonacci-prompt gate hit it directly (diverges at token 16,
    // reproducible, see that test's comment for the bisect). Fixed by
    // `mtpDecodeLoop`'s `mustReplayForExactness`: greedy-mode full
    // accepts now always replay through the single-token path, exactly
    // like partial accepts already did, so this hazard is closed
    // structurally rather than surviving on this suite by the
    // replay-frequency coincidence described above — see the commit
    // message for the full per-config/per-case numbers this audit
    // found before the fix.
    private static let boostedRegimeConfigs:
        [(name: String, env: [String: String])] = [
            ("pmin-moderate", ["BUTTER_QWEN_MTP_PMIN": "0.8"]),
            ("adaptive-default", ["BUTTER_QWEN_MTP_ADAPTIVE": "1"]),
            (
                "adaptive-aggressive",
                [
                    "BUTTER_QWEN_MTP_ADAPTIVE": "1",
                    "BUTTER_QWEN_MTP_ADAPT_LO": "0.3",
                    "BUTTER_QWEN_MTP_ADAPT_HI": "0.4",
                ]
            ),
        ]

    // MEMORY FOOTPRINT: `BUTTER_MTP_GATE_CASE` is MANDATORY for this
    // test (unlike the fixed-config gate above, where it's optional) —
    // see the precondition in the test body. This test allocates a
    // fresh `makeLayerCaches()` per (case, config) triple — up to 6
    // cases x 3 configs = 18, plus 3 per-prompt baselines, all in ONE
    // process if every case ran together. Confirmed by direct repro
    // during this audit: even with every per-config body wrapped in its
    // own `autoreleasepool` (see below), running the full unscoped
    // matrix in one process drove `vm.swapusage` to 48.7/50GB and then
    // to single-digit-thousands of free VM pages (near-OOM) — twice,
    // once before the `autoreleasepool` wrap and once after, so the
    // wrap alone does not bound this test's footprint (the per-prompt
    // baseline loop and/or the 27B checkpoint's own load-time working
    // set are the likelier culprits, not just the per-config cache
    // allocations). Rather than chase that further, this test requires
    // the same one-process-per-case discipline the shell loop this
    // audit used already relied on: run cases 0..5 as separate process
    // invocations (see the commit message for the exact loop).
    @Test(
        "spec on == spec off: adversarial prompts under PMIN/ADAPTIVE promotion configs, γ ∈ {2, 4}, 180 tokens"
    )
    func mtpSpecMatchesBaselineUnderBoostedRegime() async throws {
        let device = Device.shared
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(mtpModelPath, options: opts)
        }
        let qwen = try #require(m.qwen35, "expected Qwen35Model engine")
        let head = try #require(
            qwen.mtpHead, "expected the mtp-sidecar to attach an MTP head")

        let allPrompts = Self.adversarialPrompts
        let gammas = Self.adversarialGammas
        var allCases: [(prompt: (name: String, text: String), gamma: Int)] = []
        for p in allPrompts { for g in gammas { allCases.append((p, g)) } }

        // `BUTTER_MTP_GATE_CASE` selects one (prompt, gamma) pair; this
        // test then runs it under every config in `boostedRegimeConfigs`
        // in turn. MANDATORY here (unlike the fixed-config gate above) —
        // see the MEMORY FOOTPRINT note on this test's declaration.
        guard let idx = gateCaseIndex else {
            preconditionFailure(
                "mtpSpecMatchesBaselineUnderBoostedRegime requires BUTTER_MTP_GATE_CASE=<0..\(allCases.count - 1)> "
                    + "— this test's per-case x per-config cache footprint is NOT safe to run "
                    + "unscoped in one process (confirmed: drives this box to near-OOM). "
                    + "Run each case as its own process invocation.")
        }
        precondition(
            allCases.indices.contains(idx),
            "BUTTER_MTP_GATE_CASE=\(idx) out of range 0..<\(allCases.count)")
        let cases = [allCases[idx]]
        let count = 180

        // Baseline is per-PROMPT (gamma- and config-independent) —
        // compute once and reuse across every (gamma, config) combo for
        // the same prompt.
        var baselines: [String: (tokens: [Int], decodeSeconds: Double)] = [:]
        for (name, text) in Set(cases.map { $0.prompt.name }).sorted().compactMap({ n in
            allPrompts.first(where: { $0.name == n })
        }) {
            let promptTokens = m.tokenizer.encode(text: text)
            // Cross-arm retention fix — see `Device.releaseGenerationCaches`.
            device.releaseGenerationCaches()
            let caches = qwen.makeLayerCaches()
            let seed = autoreleasepool {
                prefill(qwen, promptTokens: promptTokens, caches: caches, device: device)
            }
            baselines[name] = autoreleasepool {
                baselineGreedy(
                    qwen, seed: seed, position: promptTokens.count, count: count,
                    caches: caches, device: device)
            }
        }

        for (prompt, gamma) in cases {
            let promptTokens = m.tokenizer.encode(text: prompt.text)
            let n = promptTokens.count
            let baseline = baselines[prompt.name]!

            for config in Self.boostedRegimeConfigs {
                // ── Apply this config's env, always alongside the
                // side-effect-free parity probe (full-accept frequency +
                // drift stats) — `ButterEnv` reads live via `getenv(3)`,
                // so `setenv`/`unsetenv` from here is observed on the
                // very next `mtpDecodeLoop` call (see `ButterEnv.swift`).
                var appliedKeys = Array(config.env.keys)
                appliedKeys.append("BUTTER_QWEN_MTP_PARITY")
                for (key, value) in config.env {
                    setenv(key, value, 1)
                }
                setenv("BUTTER_QWEN_MTP_PARITY", "1", 1)
                defer { for key in appliedKeys { unsetenv(key) } }

                // Whole per-config body (cache allocation through the
                // spec run) in ONE autoreleasepool — this loop allocates
                // a fresh `makeLayerCaches()` per (case, config) triple
                // (up to 6 cases x 3 configs = 18, plus 3 baselines, in a
                // single process when `BUTTER_MTP_GATE_CASE` is unset).
                // Without this, the 27B checkpoint's KV/GDN cache MTLBuffers
                // only get released at the next autoreleasepool drain,
                // which without an explicit pool here is the run loop's
                // own (much coarser) drain point — 21 undrained cache
                // sets is enough to swap-thrash this box (confirmed:
                // vm.swapusage hit 48.7/50GB and the run stalled for
                // 40+ minutes before being killed, see the commit
                // message). See the repo's Metal-autoreleasepool-leak
                // note: wrap any `makeBuffer`-heavy loop.
                autoreleasepool {
                    // Cross-arm retention fix — see
                    // `Device.releaseGenerationCaches`. Complements the
                    // `autoreleasepool` wrap this comment block already
                    // documents: that drains transient ObjC/Metal
                    // objects, this drops the residency-set strong
                    // reference the old `markWeightsResident` call left
                    // on every prior (case, config) triple's cache
                    // buffers — the two are independent leaks that
                    // happened to compound in this test.
                    device.releaseGenerationCaches()
                    let caches = qwen.makeLayerCaches()
                    let seed = autoreleasepool {
                        prefill(qwen, promptTokens: promptTokens, caches: caches, device: device)
                    }
                    #expect(
                        seed == baseline.tokens[0],
                        "\(prompt.name)/\(config.name): prefill seed must match between arms")

                    head.resetCache()
                    var history = promptTokens
                    let t0 = Date()
                    let stats = autoreleasepool {
                        SpecDecode.generateGreedyMTP(
                            model: qwen, head: head, gamma: gamma,
                            lastToken: seed, position: n,
                            caches: caches, history: &history,
                            maxNewTokens: count, device: device)
                    }
                    let specSeconds = Date().timeIntervalSince(t0)

                    let specTokens = Array(history[n...])
                    #expect(stats.tokensGenerated == count)
                    #expect(specTokens.count == count)

                    // ── THE EXACTNESS GATE (boosted regime) ────────────
                    let matchPrefix = zip(baseline.tokens, specTokens)
                        .prefix(while: { $0 == $1 }).count
                    let baseCtx = Array(
                        baseline.tokens.dropFirst(max(0, matchPrefix - 2)).prefix(6))
                    let specCtx = Array(
                        specTokens.dropFirst(max(0, matchPrefix - 2)).prefix(6))
                    #expect(
                        specTokens == baseline.tokens,
                        "\(prompt.name) γ=\(gamma) config=\(config.name) count=\(count): spec stream diverged from baseline at index \(matchPrefix)/\(count) — baseline \(baseCtx) vs spec \(specCtx)"
                    )

                    let specTps = Double(count) / specSeconds
                    let baseTps = Double(count) / baseline.decodeSeconds
                    print(
                        "MTPSpec-boosted \(prompt.name) γ=\(gamma) config=\(config.name) count=\(count): "
                            + "exact_match=\(specTokens == baseline.tokens) "
                            + "(prefix \(matchPrefix)/\(count)), "
                            + "tps=\(String(format: "%.2f", specTps)) "
                            + "(\(String(format: "%.2fx", specTps / baseTps)) vs baseline), "
                            + "accepted=\(stats.candidatesAccepted)/\(stats.candidatesProposed) "
                            + "(\(String(format: "%.1f", stats.acceptanceRate * 100))%), "
                            + "verify=\(stats.verifySteps), replay=\(stats.replaySingleSteps), "
                            + "lockstep=\(stats.lockstepSteps)")
                }
            }
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
