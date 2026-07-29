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
// MiniCPM5 — OpenBMB's MiniCPM5 family entry point. The only v5 model today
// is MiniCPM5-1B, a text-only checkpoint that declares a plain
// `LlamaForCausalLM` / `model_type: llama` architecture — indistinguishable
// from a generic Llama at the architecture level. The loader therefore
// routes here by a `_name_or_path` heuristic (see `handles`).
//
// This file is the **main model interface** for the family (the same shape
// as `Qwen2.swift` / `Qwen3.swift`):
//   • the family enum `MiniCPM5` (variant dispatch + the `_name_or_path`
//     `handles` heuristic + load),
//   • the `MiniCPM5Variant` protocol every concrete TEXT variant conforms
//     to (the `MiniCPM5Model` engine alias lives here too).
//
// It is intentionally a thin facade over the Llama stack today: the single
// `MiniCPM5Dense` variant (in `Models/Text/MiniCPM5Text.swift`) delegates
// text loading to `Llama3Dense`. The named entrypoint exists so that when
// multimodal v5 checkpoints arrive (OpenBMB ships omni variants of each
// generation), they compose vision / audio components here the same way
// `MiniCPM4` does — without re-plumbing the dispatch.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum MiniCPM5 {
    /// Whether a decoded `config.json` describes a MiniCPM5 checkpoint.
    ///
    /// MiniCPM5-1B declares `architectures: ["LlamaForCausalLM"]` +
    /// `model_type: "llama"`, so there's no architecture marker to match —
    /// we route by the `_name_or_path` field the conversion preserves
    /// (`"MiniCPM5-1B-MLX"`). Checked BEFORE the generic Llama branch so a
    /// MiniCPM5 checkpoint lands on this named entrypoint; every other Llama
    /// checkpoint (no "minicpm5" marker) falls through unchanged.
    public static func handles(_ config: ModelConfig) -> Bool {
        let name =
            config.string("_name_or_path") ?? config.string("name_or_path") ?? ""
        return name.lowercased().contains("minicpm5")
    }

    /// Pick the concrete text variant. MiniCPM5 ships a single dense
    /// variant (`MiniCPM5Dense`); multimodal v5 variants will widen this.
    public static func variant(for _: ModelConfig) throws -> any MiniCPM5Variant.Type {
        MiniCPM5Dense.self
    }

    /// Load a MiniCPM5 checkpoint through its text variant; the family is
    /// text-only today.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> ModelRegistry.Loaded {
        let variant = try MiniCPM5.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights, options: options, device: device)
        return ModelRegistry.Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters,
            availableCapabilities: variant.availableCapabilities)
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

/// MiniCPM5's text engine is the shared `Llama3Model` (Llama-3-shaped).
public typealias MiniCPM5Model = Llama3Model

/// Every concrete MiniCPM5 TEXT variant declares the capabilities it
/// exposes and the generation defaults it recommends, and loads its text
/// stack. Mirrors `Qwen2Variant` / `Qwen3Variant`. The concrete
/// `MiniCPM5Dense` variant lives in `Models/Text/MiniCPM5Text.swift`.
public protocol MiniCPM5Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> MiniCPM5Model
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Unified MiniCPM5 family error — the family's error vocabulary, raised by
/// the family orchestrator and its (future) vision / audio component stacks.
/// Mirrors `MiniCPM4Error`; the text-only stack delegates to `Llama3Dense`
/// today, so these surface once a multimodal v5 variant is wired.
public enum MiniCPM5Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "MiniCPM5: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "MiniCPM5: unsupported config: \(m)"
        }
    }
}
