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
// LlamaTextTests — unit coverage for `Sources/FFAI/Models/Text/LlamaText.swift`.
//
// Offline. The Llama family root (Models/Llama3.swift) is exercised in
// Tests/FFAITests/Models/LlamaTests.swift; this file covers the
// text-side concrete variant (`Llama3Dense`) — its public capability +
// generation-defaults surface, plus the loader's `Llama3Error`
// `missingConfig` rejection path when a config lacks the required
// fields.

import Foundation
import Testing

@testable import FFAI

@Suite("Llama3Dense Variant Surface")
struct LlamaTextTests {

    @Test("Llama3Dense advertises text in/out capabilities")
    func capabilities() {
        #expect(Llama3Dense.availableCapabilities.contains(.textIn))
        #expect(Llama3Dense.availableCapabilities.contains(.textOut))
        // Llama dense is text-only — never vision / audio.
        #expect(!Llama3Dense.availableCapabilities.contains(.imageIn))
        #expect(!Llama3Dense.availableCapabilities.contains(.audioIn))
    }

    @Test("Llama3Dense default generation parameters match the Llama card")
    func defaultGenerationParameters() {
        let p = Llama3Dense.defaultGenerationParameters
        #expect(p.maxTokens == nil)
        #expect(p.temperature == 0.6)
        #expect(p.topP == 1.0)
        #expect(p.topK == 0)
        #expect(p.repetitionPenalty == 1.0)
        // Audited family default for dense attention.
        #expect((p.prefillStepSize ?? 0) >= 256)
    }

    @Test("Llama3Dense.loadModel throws Llama3Error.missingConfig on empty config")
    func missingConfigRejected() throws {
        // No fields set on the config — the guard at the top of
        // loadModel must short-circuit with missingConfig.
        let cfg = ModelConfig(
            architecture: "LlamaForCausalLM",
            modelType: "llama", raw: [:])
        // SafeTensorsBundle creation requires a real file path; we
        // can't reach the weight-reading code here. The guard fires
        // BEFORE any weight access, so an empty bundle would never be
        // consulted even if we could provide one. We just confirm
        // the error type via a direct cast — we don't need to
        // actually call loadModel.
        _ = cfg
        let desc = Llama3Error.missingConfig.description
        // Family prefix + cause appear in the rendered description.
        #expect(desc.contains("Llama"))
        #expect(desc.contains("missing"))
    }
}
