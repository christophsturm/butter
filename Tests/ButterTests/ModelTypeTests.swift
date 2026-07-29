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
// ModelTypeTests — unit tests for `Sources/Butter/ModelType.swift`.

import Foundation
import Testing

@testable import ButterSwift

@Suite("ModelType")
struct ModelTypeTests {
    @Test("every case has a non-empty catalog title")
    func titles() {
        for kind in ModelType.allCases {
            #expect(!kind.title.isEmpty)
        }
    }

    @Test("implemented kinds are the 8 shipping today; reserved are not")
    func implementedFlags() {
        let implemented = ModelType.allCases.filter(\.isImplemented)
        #expect(Set(implemented) == [.llm, .vlm, .ocr, .omni, .stt, .tts, .sts, .vad])
        // The reserved generative kinds aren't wired yet.
        #expect(!ModelType.tti.isImplemented)
        #expect(!ModelType.segmentation.isImplemented)
    }

    @Test("raw values round-trip through Codable")
    func codableRoundTrip() throws {
        for kind in ModelType.allCases {
            let data = try JSONEncoder().encode(kind)
            let decoded = try JSONDecoder().decode(ModelType.self, from: data)
            #expect(decoded == kind)
        }
    }
}
