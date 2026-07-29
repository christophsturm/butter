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
// Yi family — 01.ai's Yi dense text models. Llama-3-shaped weights
// with optional QKV biases that `loadLinear` auto-detects; the family
// root just declares dispatch metadata and routes through `Llama3Dense`.

import Foundation

public enum Yi {
    public static let modelTypes: Set<String> = ["yi"]
    public static let architectures: Set<String> = ["YiForCausalLM"]

    /// Pick the concrete text variant. Yi ships a single dense variant
    /// (`YiDense`).
    public static func variant(for _: ModelConfig) throws -> any YiVariant.Type {
        YiDense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

/// Yi's text engine is the shared `Llama3Model` (Llama-3-shaped).
public typealias YiModel = Llama3Model

public protocol YiVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> YiModel
}

/// The single Yi text variant — the Llama dense backbone (optional QKV
/// biases auto-detected by `loadLinear`).
public enum YiDense: YiVariant {
    public static let availableCapabilities: Set<Capability> = Capability.textOnly
    public static let defaultGenerationParameters =
        Llama3Dense.defaultGenerationParameters

    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> YiModel {
        try Llama3Dense.loadModel(
            config: config, weights: weights, options: options, device: device)
    }
}
