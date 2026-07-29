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
// Qwen2Tests — unit tests for the Qwen 2 family root + the `Qwen2VL`
// orchestrator (`Models/Vision/Qwen2Vision.swift`).
//
// Offline. The `Qwen2` root declares the text dispatch + the unified
// `Qwen2Error`; `Qwen2VL` declares the image / video token defaults +
// capability set + generation defaults. The `vision_config` decode surface
// is covered by `Tests/ButterTests/Models/Vision/Qwen2VisionConfigTests.swift`
// — this file focuses on the constants + error shape.

import Foundation
import Testing

@testable import ButterSwift

@Suite("Qwen2VL Family Root")
struct Qwen2VLRootTests {

    @Test("Qwen2VL exposes the canonical image / video placeholder ids")
    func tokenIdDefaults() {
        #expect(Qwen2VL.defaultImageTokenId == 151_655)
        #expect(Qwen2VL.defaultVideoTokenId == 151_656)
    }

    @Test("availableCapabilities advertise text + image + video in")
    func capabilities() {
        let caps = Qwen2VL.availableCapabilities
        #expect(caps.contains(.textIn))
        #expect(caps.contains(.textOut))
        #expect(caps.contains(.imageIn))
        #expect(caps.contains(.videoIn))
    }

    @Test("Qwen2VL declares the dense generation defaults")
    func vlGenerationDefaults() {
        #expect(
            Qwen2VL.defaultGenerationParameters.temperature
                == Qwen2Dense.defaultGenerationParameters.temperature)
    }

    @Test("Qwen2Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(Qwen2Error.missingConfig.description.contains("Qwen2"))
        #expect(
            Qwen2Error.missingTensor("vision_tower.x").description
                .contains("vision_tower.x"))
    }
}
