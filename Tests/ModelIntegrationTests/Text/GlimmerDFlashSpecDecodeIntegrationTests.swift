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
// greedy loop, on every prompt. Also reports acceptance rate + decode
// tok/s (informational — Meta's own reference reports ~1.8x on this
// exact model/hardware pairing at block_size=16).

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
                let cmd = device.makeCommandBuffer()
                _ = glimmer.forwardMulti(
                    tokenIds: Array(bulk[pos ..< end]), startingAt: pos,
                    caches: caches, on: cmd, device: device)
                cmd.commit()
                cmd.waitUntilCompleted()
                pos = end
            }
        }
        let cmd = device.makeCommandBuffer()
        let (logits, features) = glimmer.forwardOneWithFeatures(
            tokenId: promptTokens.last!, position: pos, targetLayerIDs: targetLayerIDs,
            caches: caches, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        return (argmaxHost(logits), features)
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
            let cmd = device.makeCommandBuffer()
            let logits = glimmer.forward(
                tokenId: prev, position: pos, caches: caches, on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            prev = argmaxHost(logits)
            pos += 1
        }
        return (tokens, Date().timeIntervalSince(t0))
    }

    @Test("DFlash spec == baseline greedy: 3 prompts (12/150/60-tok), 80 new tokens each")
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

        let prompts: [(name: String, text: String)] = [
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
        ]
        let count = 80
        let targetLayerIDs = draft.config.targetLayerIDs

        for (name, text) in prompts {
            let promptTokens = m.tokenizer.encode(text: text)
            let n = promptTokens.count
            print("DFlash prompt=\(name) tokens=\(n)")

            // ── Baseline arm ────────────────────────────────────────
            let baseCaches = glimmer.makeLayerCaches(maxSeq: nil, device: device)
            let (baseSeed, _) = prefill(
                glimmer, promptTokens: promptTokens, targetLayerIDs: targetLayerIDs,
                caches: baseCaches, device: device)
            let baseline = baselineGreedy(
                glimmer, seed: baseSeed, position: n, count: count,
                caches: baseCaches, device: device)
            let baseTps = Double(count) / baseline.decodeSeconds
            print(
                "DFlash baseline \(name): "
                    + "\(String(format: "%.2f", baseTps)) tok/s")

            // ── DFlash spec arm ──────────────────────────────────────
            let specCaches = glimmer.makeLayerCaches(maxSeq: nil, device: device)
            let (specSeed, seedFeatures) = prefill(
                glimmer, promptTokens: promptTokens, targetLayerIDs: targetLayerIDs,
                caches: specCaches, device: device)
            #expect(specSeed == baseSeed, "prefill seed must match between arms")

            let draftCaches = draft.makeCache(maxSeq: 4096, device: device)
            var history = promptTokens
            let stats = DFlashSpecDecode.generateGreedy(
                model: glimmer, draft: draft, draftCaches: draftCaches,
                lastToken: specSeed, position: n, seedFeatures: seedFeatures,
                caches: specCaches, history: &history,
                maxNewTokens: count, device: device)

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
                "\(name): DFlash spec stream diverged from baseline at index \(matchPrefix) — baseline \(baseCtx) vs spec \(specCtx)"
            )

            let specTps = stats.tps
            print(
                "DFlash \(name): exact_match=\(specTokens == baseline.tokens) "
                    + "(prefix \(matchPrefix)/\(count)), "
                    + "tps=\(String(format: "%.2f", specTps)) "
                    + "(\(String(format: "%.2fx", specTps / baseTps)) vs baseline), "
                    + "accepted=\(stats.candidatesAccepted)/\(stats.candidatesProposed) "
                    + "(\(String(format: "%.1f", stats.acceptanceRate * 100))%), "
                    + "verify=\(stats.verifySteps), replay=\(stats.replaySingleSteps), "
                    + "draft=\(String(format: "%.2f", stats.draftSeconds))s "
                    + "verifyT=\(String(format: "%.2f", stats.verifySeconds))s "
                    + "replayT=\(String(format: "%.2f", stats.replaySeconds))s")
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
