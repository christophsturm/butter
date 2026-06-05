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
// Integration test for the Gemma 4 Omni EAGLE drafter
// (`gemma4_unified_assistant`). The drafter loads + quantizes + lists in
// `ffai models`, but CANNOT generate standalone (it drafts from the target's
// hidden state), so the dispatcher rejects standalone generation. The
// end-to-end EAGLE speculative-decode loop is Phase 8.7.
//
// `google/gemma-4-12B-it-assistant` is a GATED HuggingFace repo (separate
// approval from the omni 12B). Until access is granted + the conversion is
// hosted, this suite is skipped (and passes) — flip on by caching either the
// local `ffai convert` output or `ekryski/gemma-4-12B-it-assistant-mlx-4bit`.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

private let drafterLocalDir =
    ("~/.cache/ffai/converts/gemma-4-12B-it-assistant-mlx-4bit" as NSString)
    .expandingTildeInPath
private let drafterHFCache =
    ("~/.cache/huggingface/hub/models--ekryski--gemma-4-12B-it-assistant-mlx-4bit"
        as NSString).expandingTildeInPath
private let drafterPath: String =
    FileManager.default.fileExists(atPath: drafterLocalDir)
    ? drafterLocalDir : "ekryski/gemma-4-12B-it-assistant-mlx-4bit"
private let drafterAvailable: Bool =
    FileManager.default.fileExists(atPath: drafterLocalDir)
    || FileManager.default.fileExists(atPath: drafterHFCache)

@Suite(
    "Gemma4 Omni Drafter Integration", .serialized,
    .enabled(
        if: drafterAvailable,
        "Gemma 4 Omni drafter suite: google/gemma-4-12B-it-assistant is gated — needs access approval + a hosted/local conversion of ekryski/gemma-4-12B-it-assistant-mlx-4bit.")
)
struct Gemma4DraftIntegrationTests {

    @Test("EAGLE drafter config decodes to the expected shape")
    func config() async throws {
        let dir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: drafterPath)
        }
        let cfg = try Gemma4Draft.config(from: dir)
        #expect(cfg.backboneHiddenSize == 3840)
        #expect(cfg.hidden == 1024)
        #expect(cfg.numLayers == 4)
        #expect(cfg.numKvSharedLayers == cfg.numLayers)  // target-KV-shared (Phase 8.7)
        #expect(cfg.numCentroids == 2048)
        #expect(cfg.centroidTopK == 32)
    }

    @Test("dispatcher rejects standalone generation with an actionable error")
    func rejectsStandaloneGeneration() async throws {
        await #expect(throws: ModelError.self) {
            _ = try await Model.load(drafterPath)
        }
    }

    @Test("convert preserves the EAGLE fusion heads (pre_projection / post_projection)")
    func convertPreservesEagleHeads() async throws {
        let dir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: drafterPath)
        }
        let weights = try SafeTensorsBundle(directory: dir)
        let keys = weights.allKeys
        func hasSuffix(_ s: String) -> Bool { keys.contains { $0.hasSuffix(s) } }
        // The EAGLE fusion: pre_projection (2·backbone_hidden -> hidden) consumes
        // the target's hidden state; post_projection (hidden -> backbone_hidden)
        // feeds it back. Both must survive `ffai convert`.
        #expect(hasSuffix("pre_projection.weight"), "pre_projection dropped by convert")
        #expect(hasSuffix("post_projection.weight"), "post_projection dropped by convert")
    }
}
