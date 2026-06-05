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
// PixtralDense — the TEXT component of the Pixtral family. Pixtral-12B's
// text backbone is Mistral, which is architecturally identical to Llama, so
// it reuses the in-tree `LlamaDense` engine. This thin variant lets the
// `Pixtral` VL orchestrator load its text stack through the family's own
// variant (per the family-root convention) rather than reaching for
// `LlamaDense` directly.

import Foundation

/// Pixtral's text engine is the shared `LlamaModel` (Mistral == Llama-shaped).
public typealias PixtralModel = LlamaModel

public protocol PixtralVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> PixtralModel
}

/// The single Pixtral text variant — the Mistral dense backbone.
public enum PixtralDense: PixtralVariant {
    public static let availableCapabilities: Set<Capability> = Capability.textOnly
    public static let defaultGenerationParameters =
        LlamaDense.defaultGenerationParameters

    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> PixtralModel {
        try LlamaDense.loadModel(
            config: config, weights: weights, options: options, device: device)
    }
}
