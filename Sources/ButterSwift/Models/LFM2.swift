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
// LFM2 family root — LiquidAI's Liquid Foundation Models 2 line
// (`lfm2` / `lfm2_moe` model_type).
//
// This file is the **main model interface** for the family:
//   • the text family enum `LFM2` (modelTypes, architectures, variant
//     dispatch — picks dense or MoE),
//   • the `LFM2Variant` protocol every concrete text variant conforms to,
//   • the unified `LFM2Error` type the loaders / decode sites raise.
//
// Concrete text variants + the hybrid decoder + per-layer impl live
// under `Models/Text/LFM2Text.swift`:
//   - `LFM2Dense` / `LFM2MoE` — stack-interleaved conv + attention
//     mixers; per-layer SwiGLU MLP (`lfm2`) or block-sparse MoE FFN
//     (`lfm2_moe`).
//   - `LFM2LayerKind`, `LFM2ConvCache`, `LFM2Model` — per-layer +
//     full-model impl.
//
// The LFM2-VL orchestrator (`LFM2VL`, `Lfm2VlForConditionalGeneration`) +
// the SigLIP2 vision tower + pixel-unshuffle projector live in
// `Models/Vision/LFM2Vision.swift`.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum LFM2 {
    public static let modelTypes: Set<String> = ["lfm2", "lfm2_moe"]
    public static let architectures: Set<String> =
        ["Lfm2ForCausalLM", "Lfm2MoeForCausalLM"]

    /// True when the config names the mixture-of-experts checkpoint.
    static func isMoE(_ config: ModelConfig) -> Bool {
        config.modelType == "lfm2_moe"
            || config.architecture == "Lfm2MoeForCausalLM"
    }

    public static func variant(for config: ModelConfig) throws -> any LFM2Variant.Type {
        return isMoE(config) ? LFM2MoE.self : LFM2Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol LFM2Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LFM2Model
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Unified LFM2 family error — raised by the text loaders and the LFM2-VL
/// orchestrator + vision tower (`Models/Vision/LFM2Vision.swift`).
public enum LFM2Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingTensor(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "LFM2: required config field missing: \(f)"
        case .missingTensor(let name):
            return "LFM2: checkpoint is missing tensor '\(name)'"
        case .unsupportedConfig(let m):
            return "LFM2: unsupported config: \(m)"
        }
    }
}
