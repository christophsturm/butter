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
// FastVLMDense — the TEXT component of the FastVLM family. FastVLM's text
// backbone is Qwen 2 (Llama-shaped with QKV biases), so it reuses the
// in-tree `LlamaDense` engine. This thin variant lets the `FastVLM` VL
// orchestrator load its text stack through the family's own variant rather
// than reaching for `LlamaDense` directly.

import Foundation

/// FastVLM's text engine is the shared `LlamaModel` (Qwen 2 == Llama-shaped).
public typealias FastVLMModel = LlamaModel

public protocol FastVLMVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> FastVLMModel
}

/// The single FastVLM text variant — the Qwen 2 dense backbone.
public enum FastVLMDense: FastVLMVariant {
    public static let availableCapabilities: Set<Capability> = Capability.textOnly
    public static let defaultGenerationParameters =
        LlamaDense.defaultGenerationParameters

    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> FastVLMModel {
        try LlamaDense.loadModel(
            config: config, weights: weights, options: options, device: device)
    }
}
