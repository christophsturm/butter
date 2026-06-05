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
// ModelEntryTests — unit tests for `Sources/FFAI/Loader/ModelEntry.swift`,
// specifically the `matches(_:)` routing predicate (model_type /
// architecture membership + the optional custom `sniff`).

import Foundation
import Testing

@testable import FFAI

@Suite("ModelEntry")
struct ModelEntryTests {
    /// A throwaway entry whose `load` is never invoked by these tests —
    /// we only exercise the routing predicate.
    private func entry(
        modelTypes: Set<String> = [],
        architectures: Set<String> = [],
        sniff: (@Sendable (ModelConfig) -> Bool)? = nil
    ) -> ModelEntry {
        ModelEntry(
            family: "Test", kind: .llm,
            modelTypes: modelTypes, architectures: architectures,
            sniff: sniff,
            load: { _, _, _, _ in
                throw ModelError.unsupportedModelType("test")
            })
    }

    private func config(modelType: String? = nil, architecture: String? = nil)
        -> ModelConfig
    {
        ModelConfig(architecture: architecture, modelType: modelType, raw: [:])
    }

    @Test("matches by model_type")
    func matchesByModelType() {
        let e = entry(modelTypes: ["llama"])
        #expect(e.matches(config(modelType: "llama")))
        #expect(!e.matches(config(modelType: "qwen3")))
    }

    @Test("matches by architecture")
    func matchesByArchitecture() {
        let e = entry(architectures: ["LlamaForCausalLM"])
        #expect(e.matches(config(architecture: "LlamaForCausalLM")))
        #expect(!e.matches(config(architecture: "Qwen3ForCausalLM")))
    }

    @Test("custom sniff matches when sets miss")
    func matchesBySniff() {
        let e = entry(sniff: { cfg in cfg.modelType == "secret" })
        #expect(e.matches(config(modelType: "secret")))
        #expect(!e.matches(config(modelType: "other")))
    }

    @Test("no match when nothing applies")
    func noMatch() {
        let e = entry(modelTypes: ["llama"], architectures: ["LlamaForCausalLM"])
        #expect(!e.matches(config(modelType: "alien", architecture: "NotAFamily")))
    }
}
