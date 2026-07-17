// Copyright 2026 Tom Turney (@TheTom)
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
// Hy3 family root — Tencent Hunyuan Hy3 (`hy_v3` / `HYV3ForCausalLM`).
//
// Architecture (confirmed against the HF config and reference
// implementations of hy_v3):
//   • 80 decoder layers, hidden=4096, GQA 64/8 head_dim=128
//   • first_k_dense_replace=1 dense SwiGLU layer, then MoE layers
//   • MoE: 192 experts top-8, expert intermediate 1536, 1 ungated shared
//   • Router Path B: sigmoid + expert_bias selection, unbiased weights,
//     route_norm, router_scaling_factor (≈2.826)
//   • QK-RMSNorm before RoPE
//   • Optional MTP block (num_nextn_predict_layers=1) — not wired yet
//
// Concrete loader / decoder: `Models/Text/Hy3Text.swift`.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Hy3 {
    /// HuggingFace `model_type` strings this family handles.
    public static let modelTypes: Set<String> = ["hy_v3", "hy-v3"]

    /// HuggingFace `architectures[0]` strings this family handles.
    public static let architectures: Set<String> = [
        "HYV3ForCausalLM",
        "HyV3ForCausalLM",
        "Hy3ForCausalLM",
    ]

    public static func variant(for config: ModelConfig) throws -> any Hy3Variant.Type {
        _ = config
        return Hy3Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Hy3Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Hy3Model
}

extension Hy3Variant {
    public static var availableCapabilities: Set<Capability> { [.textIn, .textOut] }
    public static var defaultGenerationParameters: GenerationParameters {
        // Tencent recipe: temperature=0.9, top_p=1.0. 256K context.
        GenerationParameters(
            prefillStepSize: 4096,
            temperature: 0.9, topP: 1.0, topK: 0,
            repetitionPenalty: 1.0)
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum Hy3Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingTensor(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .missingConfig(let k): return "Hy3: missing config key \(k)"
        case .missingTensor(let n): return "Hy3: missing tensor \(n)"
        case .unsupported(let m): return "Hy3: \(m)"
        }
    }
}
