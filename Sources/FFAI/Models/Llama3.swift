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
// Llama family root — Meta's Llama 3.x line.
//
// This file is the **main model interface** for the family:
//   • the family enum `Llama` (modelTypes, architectures, variant
//     dispatch),
//   • the `Llama3Variant` protocol every concrete variant conforms to,
//   • the `Llama3Error` type every loader / decode site raises.
//
// Concrete variants + the dense decoder + per-layer impl live under
// `Models/Text/LlamaText.swift`:
//   - `Llama3Dense` — Llama 3 / 3.1 / 3.2 dense GQA transformer (the
//     only variant shipped today; 1B / 3B / 8B / 70B; 405B with
//     quantization).
//   - `Llama3Layer`, `Llama3Model` — per-layer + full-model impl.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Llama3 {
    /// HuggingFace `model_type` strings this family handles.
    public static let modelTypes: Set<String> = ["llama"]
    /// HuggingFace `architectures[0]` strings this family handles.
    public static let architectures: Set<String> = ["LlamaForCausalLM"]

    /// Pick the concrete variant for a config. Only `Llama3Dense`
    /// ships today; future variants (Llama 4 MoE, etc.) dispatch here.
    public static func variant(for config: ModelConfig) throws -> any Llama3Variant.Type {
        _ = config
        return Llama3Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Llama3Variant {
    /// Capabilities a checkpoint of this variant exposes.
    static var availableCapabilities: Set<Capability> { get }
    /// Generation defaults for this variant. The user can override any
    /// field; absent overrides fall back to the values declared here.
    /// See `planning/roadmap.md` for which fields are honored today vs
    /// staged for planned (sampling kernels).
    static var defaultGenerationParameters: GenerationParameters { get }
    /// Build a `Llama3Model` decoder from a checkpoint.
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Llama3Model
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum Llama3Error: Error, CustomStringConvertible {
    case missingConfig
    public var description: String {
        switch self {
        case .missingConfig: return "Llama: required config field missing"
        }
    }
}
