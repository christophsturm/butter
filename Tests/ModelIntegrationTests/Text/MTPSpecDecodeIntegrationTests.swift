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
