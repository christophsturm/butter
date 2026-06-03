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
// Integration test for the Gemma 4 Omni (`gemma4_unified`) TEXT backbone on
// a real 4-bit checkpoint: the encoder-free omni text_config resolves to a
// `Gemma4Dense` backbone (dual head_dim, proportional RoPE, softcap 30,
// tied embeddings) loaded from the `model.language_model.`-prefixed sub-tree.
//
// Resolves the published `ekryski/gemma-4-12B-it-mlx-4bit` conversion,
// preferring a local `ffai convert` output dir when present so the suite
// runs against the freshly-converted model without a 7 GB download. Skipped
// (and passes) when neither is available.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

/// Local `ffai convert` output, then the published HF repo.
let gemma4OmniLocalDir =
    ("~/.cache/ffai/converts/gemma-4-12B-it-mlx-4bit" as NSString)
    .expandingTildeInPath
let gemma4OmniHFCache =
    ("~/.cache/huggingface/hub/models--ekryski--gemma-4-12B-it-mlx-4bit"
        as NSString).expandingTildeInPath

/// Path to load: the local convert dir if present, else the HF repo id.
let gemma4OmniModelPath: String =
    FileManager.default.fileExists(atPath: gemma4OmniLocalDir)
    ? gemma4OmniLocalDir : "ekryski/gemma-4-12B-it-mlx-4bit"

/// True when the model is resolvable offline (local convert dir or HF cache).
let gemma4OmniCacheAvailable: Bool =
    FileManager.default.fileExists(atPath: gemma4OmniLocalDir)
    || FileManager.default.fileExists(atPath: gemma4OmniHFCache)

@Suite(
    "Gemma4 Omni Text Integration", .serialized,
    .enabled(
        if: gemma4OmniCacheAvailable,
        "Gemma 4 Omni text suite: needs ekryski/gemma-4-12B-it-mlx-4bit cached (or a local `ffai convert` output dir).")
)
struct Gemma4OmniTextIntegrationTests {

    @Test("load — omni checkpoint loads as a VLM with full omni capabilities")
    func loadOmni() async throws {
        let m = try await loadModel(gemma4OmniModelPath)
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.audioIn))
        #expect(m.availableCapabilities.contains(.videoIn))
        #expect(m.engine.supportsEmbeddingInput)
        // The omni text backbone is the 48-layer, 3840-hidden Gemma4Dense.
        #expect(m.engine.hidden == 3840)
        #expect(m.engine.nLayers == 48)
    }

    @Test("chat generate — encoder-free text backbone produces coherent output")
    func loadAndGenerate() async throws {
        let m = try await loadModel(gemma4OmniModelPath)
        let r = try await m.generate(
            messages: [
                ChatMessage(
                    role: .user,
                    content: "Name three primary colors, comma-separated.")
            ],
            parameters: m.defaultGenerationParameters.with {
                $0.maxTokens = 64
                $0.temperature = 0
            })
        expectCoherentOutput(r.generatedTokens, minTokens: 4, label: "gemma4-omni-text")
    }
}
