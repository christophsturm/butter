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
// Qwen2TextTests — unit coverage for `Sources/ButterSwift/Models/Text/Qwen2Text.swift`.
//
// Offline. Qwen 2 / 2.5 dense is the `Qwen2Dense` variant, a thin adapter
// over the Llama dense engine (loadLinear auto-detects the QKV projection
// biases the Qwen 2 line carries). Tests guard the family-metadata surface
// (on the `Qwen2` root) and the `variant(for:)` contract: every Qwen 2 /
// 2.5 config resolves to `Qwen2Dense`, which declares text-only
// capabilities + the dense generation defaults.

import Foundation
import Testing

@testable import ButterSwift

@Suite("Qwen2 Text Family Routing")
struct Qwen2TextTests {

    @Test("Qwen2.modelTypes advertises qwen2 (covers 2.x + 2.5)")
    func modelTypes() {
        #expect(Qwen2.modelTypes.contains("qwen2"))
    }

    @Test("Qwen2.architectures advertises Qwen2ForCausalLM")
    func architectures() {
        #expect(Qwen2.architectures.contains("Qwen2ForCausalLM"))
    }

    @Test("Qwen2.variant(for:) returns Qwen2Dense for any qwen2 config")
    func routesThroughQwen2Dense() throws {
        let cfg = ModelConfig(
            architecture: "Qwen2ForCausalLM",
            modelType: "qwen2", raw: [:])
        let v = try Qwen2.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Qwen2Dense.self))
    }

    @Test("Qwen2Dense advertises text-only capabilities + dense gen defaults")
    func variantContract() {
        #expect(Qwen2Dense.availableCapabilities == Capability.textOnly)
        #expect(
            Qwen2Dense.defaultGenerationParameters.temperature
                == Llama3Dense.defaultGenerationParameters.temperature)
    }
}
