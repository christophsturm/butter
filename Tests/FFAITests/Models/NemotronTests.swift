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
// NemotronTests — unit tests for the Nemotron family entry points.
//
// Offline. Covers the `NemotronH` family root
// (`Sources/FFAI/Models/Nemotron.swift`) — registration + variant
// dispatch + the unified `NemotronHError` description shape — and the
// `NemotronDiffusionVL` orchestrator
// (`Sources/FFAI/Models/Vision/NemotronDiffusionVision.swift`)
// registration + defaults. (The NemotronDiffusion text family root has
// its own coverage; the two architectures are deliberately separate.)

import Foundation
import Testing

@testable import FFAI

@Suite("Nemotron Family Roots")
struct NemotronFamilyRootTests {

    @Test("NemotronH owns nemotron_h + NemotronHForCausalLM")
    func nemotronHRegistration() {
        #expect(NemotronH.modelTypes.contains("nemotron_h"))
        #expect(NemotronH.architectures.contains("NemotronHForCausalLM"))
    }

    @Test("NemotronH.variant(for:) returns NemotronHHybrid")
    func nemotronHVariant() throws {
        let cfg = ModelConfig(
            architecture: "NemotronHForCausalLM",
            modelType: "nemotron_h", raw: [:])
        let v = try NemotronH.variant(for: cfg)
        #expect(String(describing: v) == String(describing: NemotronHHybrid.self))
    }

    @Test("NemotronHError stringifies every case with its payload")
    func nemotronHErrorDescriptions() {
        #expect(NemotronHError.missingConfig("layers").description.contains("layers"))
        #expect(NemotronHError.missingTensor("foo").description.contains("foo"))
        #expect(NemotronHError.unsupportedConfig("bad").description.contains("bad"))
        #expect(NemotronHError.missingConfig("x").description.contains("NemotronH"))
    }

    @Test("NemotronDiffusionVL owns the VLM model_type + arch")
    func nemotronDiffusionVLRegistration() {
        #expect(NemotronDiffusionVL.modelTypes.contains("nemotron_labs_diffusion_vlm"))
        #expect(NemotronDiffusionVL.architectures.contains("NemotronLabsDiffusionVLMModel"))
    }

    @Test("NemotronDiffusionVL defaults match the shipped 8B config")
    func nemotronDiffusionVLDefaults() {
        #expect(NemotronDiffusionVL.defaultSpatialMergeSize == 2)
        #expect(NemotronDiffusionVL.defaultImageTokenId == 10)
    }
}
