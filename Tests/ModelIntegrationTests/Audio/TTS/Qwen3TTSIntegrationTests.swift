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
// Integration test: loads a real Qwen3TTS (12 Hz Flash) checkpoint from
// the HF cache and exercises the config decode, family detection, and the
// staged-synthesis error surface — including the streaming conformance.
//
// DO NOT RUN this suite directly — `ModelTests` loads multi-GB HuggingFace
// snapshots and must be serialized. Use:
//
//   make test-integration
//
// A load failure FAILS the suite. A missing checkpoint surfaces as a
// thrown error so the test fails rather than silently passing.
//
// Qwen3TTS is a STAGED PORT (see the scope note in Qwen3TTS.swift): this
// build ships stage 1 (config + family detection). The talker / code
// predictor / ECAPA speaker encoder / speech-tokenizer codec are follow-on
// stages, so `synthesize` (and therefore `synthesizeStream`) throw the
// typed `Qwen3TTSError.synthesisNotWired` by design. This suite verifies
// the model loads and the streaming surface is wired to propagate that
// staged error in lock-step with the batch path.
//
// ## Checkpoint
//
// Primary: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit`
// The smallest published Qwen3-TTS Flash variant (4-bit quantized).

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite(
    "Qwen3TTS Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct Qwen3TTSIntegrationTests {

    /// Canonical HF repo id. The 4-bit variant is the smallest published
    /// Qwen3-TTS Flash checkpoint.
    private static let repoId = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"

    /// Resolve and load the Qwen3TTS checkpoint. Throws on failure so a
    /// missing checkpoint fails the test instead of skipping it.
    private func loadModel() async throws -> Qwen3TTSModel {
        let dir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
        return try Qwen3TTSModel.load(directory: dir)
    }

    @Test("load — Qwen3TTS checkpoint decodes the talker config")
    func load_decodesTalkerConfig() async throws {
        let model = try await loadModel()
        #expect(model.config.talker.nLayers > 0)
        #expect(model.config.talker.hidden > 0)
        #expect(model.sampleRate == 24_000)
    }

    @Test("synthesize — staged port reports synthesis as not wired")
    func synthesize_reportsNotWired() async throws {
        let model = try await loadModel()
        #expect(throws: Qwen3TTSError.self) {
            _ = try model.synthesize(text: "Hello.")
        }
    }

    @Test("synthesizeStream — concatenated stream matches a single full synthesize")
    func synthesizeStreamingFromText() async throws {
        let model = try await loadModel()
        let text = "Hello there."

        // Drive the streaming surface and concatenate every yielded chunk.
        // The conformance is a SINGLE-yield (talker→codec has no clean
        // per-frame split — see synthesizeStream's doc comment), so the
        // concatenation equals the one chunk it yields.
        func drainStream() async throws -> [Float] {
            var samples: [Float] = []
            for try await chunk in model.synthesizeStream(text: text) {
                samples.append(contentsOf: chunk.toFloatArray())
            }
            return samples
        }

        // The talker / code-predictor / codec are a separate staged port.
        // While they are unwired, both the batch and streaming paths surface
        // the same typed error; once they land, the concatenated stream must
        // equal the batch synthesis sample-for-sample.
        let batchThrows = (try? model.synthesize(text: text)) == nil
        if batchThrows {
            // Streaming must propagate the SAME typed error as the batch path.
            await #expect(throws: Qwen3TTSError.self) {
                _ = try await drainStream()
            }
        } else {
            let streamed = try await drainStream()
            let batch = try model.synthesize(text: text).toFloatArray()
            #expect(
                streamed.count == batch.count,
                "streamed (\(streamed.count)) != batch (\(batch.count))")
            // SINGLE-yield ⇒ streamed concatenation is bit-identical to batch.
            var maxDiff: Float = 0
            for i in 0 ..< min(streamed.count, batch.count) {
                maxDiff = max(maxDiff, abs(streamed[i] - batch[i]))
            }
            #expect(maxDiff < 1e-5, "stream/batch mismatch (max \(maxDiff))")
        }
    }
}
