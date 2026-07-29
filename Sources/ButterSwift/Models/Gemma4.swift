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
// Gemma 4 family root — Google's Gemma 4 line.
//
// This file is the **main model interface** for the family:
//   • the family enum `Gemma4` (modelTypes, architectures, variant
//     dispatch),
//   • the `Gemma4Variant` protocol every concrete variant conforms to,
//   • the unified `Gemma4Error` type every loader / decode site raises
//     (covers both the text path and the Gemma 4 VL path).
//
// Concrete variants + the dense / E / MoE decoder + per-layer impl live
// under `Models/Text/Gemma4Text.swift`:
//   - `Gemma4Dense` — 31B Gemma-style backbone, no PLE, dense MLP.
//   - `Gemma4E`     — E2B / E4B; adds Per-Layer Embeddings.
//   - `Gemma4MoE`   — 26B-A4B; mixture-of-experts feed-forward.
//   - `Gemma4Model` — the full TextGenerating decoder.
//
// The Gemma 4 VL vision-language orchestrator (`enum Gemma4VL`) — which
// ties the Gemma 4 text backbone to the bespoke Gemma 4 ViT vision
// tower + multi-modal embedder — lives in
// `Models/Vision/Gemma4Vision.swift` alongside the tower internals.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Gemma4 {
    public static let modelTypes: Set<String> = [
        "gemma4", "gemma4_text",
        // Gemma 4 Omni (encoder-free) — the omni checkpoint's text tower
        // is byte-for-byte a Gemma4Dense backbone; `gemma4_unified_text` is
        // the text_config model_type. The top-level `gemma4_unified` carries
        // a `vision_config`, so the dispatcher routes it through the VL
        // section to the omni loader; a bare text-only `gemma4_unified_text`
        // checkpoint (if one ships) loads as a plain Gemma4Dense backbone.
        "gemma4_unified", "gemma4_unified_text",
    ]
    public static let architectures: Set<String> = [
        "Gemma4ForCausalLM", "Gemma4TextForCausalLM",
        "Gemma4ForConditionalGeneration",
        // Gemma 4 Omni (the VL section intercepts this — it has a
        // vision_config — and routes to `Gemma4Omni.load`).
        "Gemma4UnifiedForConditionalGeneration",
    ]

    /// The EAGLE/MTP drafter (`google/gemma-4-12B-it-assistant`). Its
    /// architecture is kept OUT of `Gemma4.architectures` so it never
    /// auto-routes to standalone text generation — it cannot generate
    /// without the target model's hidden state (spec-decode, Phase 8.7).
    /// `Gemma4Draft` provides a typed loader for tests + the future
    /// draft loop; the dispatcher recognizes these strings to raise an
    /// actionable error.
    public static let drafterModelTypes: Set<String> = ["gemma4_unified_assistant"]
    public static let drafterArchitectures: Set<String> = ["Gemma4UnifiedAssistantForCausalLM"]

    /// Resolve the concrete variant from config. MoE wins over PLE wins
    /// over plain dense.
    public static func variant(for config: ModelConfig) throws -> any Gemma4Variant.Type {
        let tc = Gemma4Config.textConfig(config)
        if (tc["enable_moe_block"] as? Bool) ?? false {
            return Gemma4MoE.self
        }
        if let ple = tc["hidden_size_per_layer_input"] as? Int, ple > 0 {
            return Gemma4E.self
        }
        return Gemma4Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Gemma4Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Gemma4Model
}

extension Gemma4Variant {
    public static var availableCapabilities: Set<Capability> { [.textIn, .textOut] }
    public static var defaultGenerationParameters: GenerationParameters {
        // Gemma 4: 4096-token prefill chunk is the audited family
        // optimum (pure-attention backbone, no SSM bottleneck).
        GenerationParameters(
            prefillStepSize: 4096,
            temperature: 1.0, topP: 0.95, topK: 64,
            repetitionPenalty: 1.0)
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Unified Gemma 4 family error — raised by both the text loaders
/// (`Gemma4Dense` / `Gemma4E` / `Gemma4MoE.loadModel`) and the Gemma 4
/// VL orchestrator (`Gemma4VL.load` in `Models/Vision/Gemma4Vision.swift`).
public enum Gemma4Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingTensor(String)
    case unsupportedHeadDim(Int)
    case unalignedNorm(Int)

    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "Gemma4: required config field missing: \(f)"
        case .missingTensor(let name):
            return "Gemma4: checkpoint is missing tensor '\(name)'"
        case .unsupportedHeadDim(let d):
            return "Gemma4: head_dim \(d) unsupported (Ops.sdpaDecode needs 64/128/256/512)"
        case .unalignedNorm(let n):
            return "Gemma4: norm row size \(n) must be 128-aligned"
        }
    }
}
