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
// Mistral3Dense — the TEXT component of the Mistral 3 family. Mistral
// Small 3.1's text backbone is Mistral (Llama-shaped), so it reuses the
// in-tree `LlamaDense` engine. This thin variant lets the `Mistral3` VL
// orchestrator load its text stack through the family's own variant rather
// than reaching for `LlamaDense` directly.

import Foundation

/// Mistral 3's text engine is the shared `LlamaModel` (Mistral == Llama-shaped).
public typealias Mistral3Model = LlamaModel

public protocol Mistral3Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Mistral3Model
}

/// The single Mistral 3 text variant — the Mistral dense backbone.
public enum Mistral3Dense: Mistral3Variant {
    public static let availableCapabilities: Set<Capability> = Capability.textOnly
    public static let defaultGenerationParameters =
        LlamaDense.defaultGenerationParameters

    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Mistral3Model {
        try LlamaDense.loadModel(
            config: config, weights: weights, options: options, device: device)
    }
}
