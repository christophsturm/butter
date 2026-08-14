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
// KV-cache sizing gap regression: `MuseGlimmerDense.loadModel` used to
// hardcode `maxSeq` from `max_position_embeddings` (131072) and never
// consulted `LoadOptions.maxContextLength`. That value sizes the
// UNBOUNDED KV cache on the 13 full-attention NoPE layers (sliding
// layers stay bounded by `slidingWindow` regardless — see
// `MuseGlimmerModel.makeLayerCaches`), so every load — including a
// plain text-only `generate()` — paid ~21 GB of KV for the full 131072
// window with no public-API way to shrink it (measured ~46.9 GB peak
// RSS, watchdog-killed during a separate investigation). See
// `MuseGlimmerMaxSeqTests` (ButterTests) for the config-only unit
// coverage of the fix; this suite proves it end-to-end against the
// real checkpoint.
//
// Uses the local 4-bit checkpoint at `~/models/muse-glimmer-30b-4bit`,
// text-only (no vision splice — see `MuseGlimmerVisionIntegrationTests`
// for the VLM path).

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("Muse Glimmer text — maxContextLength KV sizing", .serialized, .enabled(if: modelTestsOptedIn))
struct MuseGlimmerTextIntegrationTests {

    static let modelPath = "~/models/muse-glimmer-30b-4bit"

    @Test("LoadOptions.maxContextLength bounds the loaded engine's maxSeq and generate() stays coherent + deterministic")
    func boundedContextLoadsAndGenerates() async throws {
        let m = try await loadModel(Self.modelPath, options: LoadOptions(maxContextLength: 4096))

        // The fix: the loaded engine's stored maxSeq reflects the
        // LoadOptions override, not the checkpoint's advertised 131072.
        let glimmer = try #require(m.museGlimmer, "expected engine to be MuseGlimmerModel")
        #expect(glimmer.maxSeq == 4096)

        // `Loader/Model.swift`'s `prewarm()` and `Generation/Generate.swift`'s
        // `generate()` both call `engine.makeLayerCaches()` with no
        // per-call override (default `maxSeq: nil`), so they pick up
        // `glimmer.maxSeq` here. Confirm the full-attention (non-sliding)
        // layers' caches are actually bounded to it, not the checkpoint
        // window.
        let caches = glimmer.makeLayerCaches(device: Device.shared)
        let fullLayerIdx = glimmer.layers.firstIndex { !$0.isSliding }
        let idx = try #require(fullLayerIdx, "expected at least one full-attention layer")
        let fullCache = try #require(caches[idx] as? KVCache)
        #expect(fullCache.maxSeq == 4096)

        // Deterministic greedy decode, run twice against the one loaded
        // engine — KV sizing must not change the math (position indices
        // are unaffected by cache capacity), so both runs should match
        // each other and read as coherent text. This substitutes for a
        // byte-identity check against the unbounded (131072) path,
        // which would need a second 20+ GB load on this box.
        let prompt =
            "<|start|>user<|message|>Explain what a hash table is in two sentences.<|eot|>"
            + "<|start|>assistant<|message|>"
        let params = m.defaultGenerationParameters.with {
            $0.maxTokens = 100
            $0.temperature = 0
        }
        let run1 = try await m.generate(prompt: prompt, parameters: params)
        let run2 = try await m.generate(prompt: prompt, parameters: params)

        #expect(
            run1.generatedTokens == run2.generatedTokens,
            Comment(
                rawValue:
                    "bounded-context generation is non-deterministic across two runs of the "
                    + "same loaded engine — run1=\(run1.generatedTokens.prefix(20))... "
                    + "run2=\(run2.generatedTokens.prefix(20))..."))
        expectCoherentOutput(
            run1.generatedTokens, minTokens: 8,
            label: "Muse Glimmer text-only, maxContextLength=4096")
        print("Muse Glimmer (maxContextLength=4096) generated: \(run1.text)")
    }
}
