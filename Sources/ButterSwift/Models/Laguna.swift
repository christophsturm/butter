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
// Laguna family root — poolside Laguna line (S-2.1 and XS-2.1).
//
// Architecture (from upstream HF config + modeling; shape is config-driven):
//   • Hybrid iSWA: full attention every 4th layer (dense-first), else SWA-512
//   • Asymmetric GQA via num_attention_heads_per_layer (S: 48/72, XS: 48/64)
//   • Softplus per-head attention output gate (g_proj) before o_proj
//   • QK-RMSNorm pre-RoPE
//   • RoPE: full = YaRN partial 0.5 θ=5e5; sliding = default θ=1e4
//   • Layer 0 dense SwiGLU; remaining layers MoE (256 experts + shared)
//   • Sigmoid + e_score_correction_bias router; routed_scaling from config
//
// S-2.1: 48L hidden=3072 topk=10 mid=1024 YaRN×128 (1M card)
// XS-2.1: 40L hidden=2048 topk=8 mid=512 YaRN×32 (256K)
//
// This file is the main model interface:
//   • family enum `Laguna` (modelTypes, architectures, variant dispatch)
//   • `LagunaVariant` protocol
//   • `LagunaError`
//
// Concrete config decode + decoder live in `Models/Text/LagunaText.swift`.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Laguna {
    /// HuggingFace `model_type` strings this family handles.
    public static let modelTypes: Set<String> = ["laguna"]

    /// HuggingFace `architectures[0]` strings this family handles.
    public static let architectures: Set<String> = [
        "LagunaForCausalLM",
    ]

    public static func variant(for config: ModelConfig) throws -> any LagunaVariant.Type {
        _ = config
        return LagunaHybrid.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol LagunaVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LagunaModel
}

extension LagunaVariant {
    public static var availableCapabilities: Set<Capability> { [.textIn, .textOut] }
    public static var defaultGenerationParameters: GenerationParameters {
        // 1M context with YaRN on full-attention layers and SWA-512 on
        // the other 3-of-4. Chunk size matches the hybrid-family default.
        // No model-specific maxTokens — callers own generation length.
        GenerationParameters(
            prefillStepSize: 4096,
            temperature: 1.0, topP: 0.95, topK: 64,
            repetitionPenalty: 1.0)
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum LagunaError: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingTensor(String)
    case unsupportedLayerType(String)
    case unsupportedConfig(String)
    case notYetImplemented(String)

    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "Laguna: required config field missing: \(f)"
        case .missingTensor(let name):
            return "Laguna: checkpoint is missing tensor '\(name)'"
        case .unsupportedLayerType(let t):
            return "Laguna: unknown layer_type '\(t)' (expected 'full_attention' or 'sliding_attention')"
        case .unsupportedConfig(let m):
            return "Laguna: unsupported config: \(m)"
        case .notYetImplemented(let what):
            return "Laguna: \(what) — not yet implemented"
        }
    }
}
