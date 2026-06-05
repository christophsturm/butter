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
// MiniCPM5Text — the TEXT component of the MiniCPM5 family. MiniCPM5-1B
// declares a plain `LlamaForCausalLM` / `llama` architecture, so its text
// backbone reuses the in-tree Llama dense stack via the `MiniCPM5Dense`
// variant. This is the seam where a future multimodal v5 adds vision /
// audio components alongside the text variant.

import Foundation

// ─── Variant protocol ────────────────────────────────────────────────

/// MiniCPM5's text engine is the shared `Llama3Model` (Llama-3-shaped).
public typealias MiniCPM5Model = Llama3Model

public protocol MiniCPM5Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> MiniCPM5Model
}

/// The single MiniCPM5 text variant — the Llama dense backbone.
/// MiniCPM5-1B is a standard GQA Llama shape, so it reuses `Llama3Dense`.
public enum MiniCPM5Dense: MiniCPM5Variant {
    public static let availableCapabilities: Set<Capability> = Capability.textOnly
    public static let defaultGenerationParameters =
        Llama3Dense.defaultGenerationParameters

    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> MiniCPM5Model {
        try Llama3Dense.loadModel(
            config: config, weights: weights, options: options, device: device)
    }
}
