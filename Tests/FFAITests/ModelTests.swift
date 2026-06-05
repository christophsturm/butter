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
// ModelTests — unit tests for `Sources/FFAI/Model.swift` (the public
// `Model` wrapper + the shared `ModelError`). End-to-end `Model.load`
// coverage lives in the integration suites (multi-GB snapshots); this
// file covers the error vocabulary in isolation.

import Foundation
import Testing

@testable import FFAI

@Suite("Model")
struct ModelTests {
    @Test("ModelError.description renders each case")
    func errorDescriptions() {
        let a = ModelError.unsupportedArchitecture("foo")
        #expect(String(describing: a).contains("foo"))
        let b = ModelError.unsupportedModelType("bar")
        #expect(String(describing: b).contains("bar"))
        let c = ModelError.capabilityNotAvailable(.audioOut)
        #expect(String(describing: c).contains("audioOut"))
    }
}
