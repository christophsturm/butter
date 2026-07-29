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
// Mistral3 integration test — Mistral AI's Mistral Small 3.1
// vision-language model (`mistral3` model_type,
// `Mistral3ForConditionalGeneration` architecture).
//
// Uses the mlx-community 4-bit quantized conversion:
//   `mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit`
//
// The checkpoint is cached locally at the time this test was written
// (2026-05-23) — the suite runs without the `.disabled` tag.
//
// DO NOT RUN `swift test` on this target — individual ModelTests load
// multi-GB checkpoints and must be run via `make test-integration`
// (serialized, --num-workers 1).

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite(
    "Mistral3 Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct Mistral3IntegrationTests {

    static let modelId = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit"

    /// The checkpoint, loaded ONCE and shared across every `@Test`.
    static let loaded = Task { try await loadModel(modelId) }

    @Test("load — Mistral3 checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m = try await Self.loaded.value

        // The checkpoint is a VLM — vlModel is present, .imageIn is
        // available, and the text backbone supports the splice.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.textEngine.supportsEmbeddingInput)

        let vlm = try #require(m.vlModel)
        // The vision tower contributes a positive run of merged patch tokens.
        #expect(vlm.imageTokenCount > 0)
        // For the mlx-community Mistral-Small-3.1 4bit conversion with
        // image_size=1540, patchSize=14, spatialMergeSize=2:
        //   110×110 patches → 55×55 = 3025 merged tokens.
        #expect(vlm.imageTokenCount == 3025)
    }

    @Test("enable / disable .imageIn — runtime capability flip")
    func capabilityFlip() async throws {
        let m = try await Self.loaded.value
        #expect(m.availableCapabilities.contains(.imageIn))
        m.disable(.imageIn)
        #expect(!m.isEnabled(.imageIn))
        m.enable(.imageIn)
        #expect(m.isEnabled(.imageIn))
    }

    @Test("text — VL backbone generates coherently (text-only prompt)")
    func textGeneration() async throws {
        let m = try await Self.loaded.value
        let r = try await m.generate(
            messages: [
                ChatMessage(
                    role: .user, content: "Name three primary colors, comma-separated.")
            ],
            parameters: m.defaultGenerationParameters.with {
                $0.maxTokens = 64
                $0.temperature = 0
            })
        expectCoherentOutput(r.generatedTokens, minTokens: 4, label: "mistral3-text")
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "Mistral3 checkpoint is not a VLM")

        // Build an image+text prompt. Mistral3 uses the `[IMG]` special
        // token (id = 10 by default) as the image placeholder in its
        // prompt format; one placeholder per merged patch.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.textTokenizer.encode(
            text: "\nDescribe this image.")
        let promptTokens =
            Array(
                repeating: imageTokenId,
                count: vlm.imageTokenCount) + questionTokens

        // The golden-retriever fixture shared by all VLM tests.
        let image = try VisionTestHelpers.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId)

        // Coherence gate: the model must produce at least 8 tokens of
        // non-trivial text.
        expectCoherentOutput(
            generated, minTokens: 8,
            label: "Mistral3 image+text")
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Mistral3 generated: \(text)")
        // Content gate: the caption should mention a dog.
        VisionTestHelpers.expectMentionsDog(text, label: "Mistral3")
    }
}
