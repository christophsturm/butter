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
// Qwen 2 family root — Alibaba's Qwen 2 / 2.5 line.
//
// This file is the **main model interface** for the family (the same shape
// as `Qwen3.swift` / `Granite4.swift`):
//   • the family enum `Qwen2` (modelTypes, architectures, text-variant
//     dispatch, and the Qwen2.5-Omni ownership),
//   • the `Qwen2Variant` protocol every concrete TEXT variant conforms to,
//   • the unified `Qwen2Error` type the family's loaders raise.
//
// Per-capability component stacks live in their own files, each declaring
// the capabilities + generation defaults it requires:
//   - text   — `Models/Text/Qwen2Text.swift` (`Qwen2Dense`, a thin variant
//              over the in-tree Llama dense engine; Qwen 2.x is Llama-shaped
//              with QKV biases that `loadLinear` handles transparently).
//   - vision — `Models/Vision/Qwen2Vision.swift` (`Qwen2VL` orchestrator +
//              the dynamic-resolution ViT tower) for Qwen2-VL.
//   - audio  — Qwen2.5-Omni reuses the shared `QwenOmniModel` engine
//              (`Models/Qwen3.swift`, the canonical omni home); this file
//              owns only the `qwen2_5_omni*` detection (`isOmni`).

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Qwen2 {
    /// Both 2.x and 2.5.x ship `model_type = "qwen2"`. The 2.5 refresh
    /// changed weights + tokenizer but kept the same arch + config keys.
    public static let modelTypes: Set<String> = ["qwen2"]
    public static let architectures: Set<String> = ["Qwen2ForCausalLM"]

    /// Pick the concrete TEXT variant for a config. Qwen 2.x ships a single
    /// Llama-shaped dense variant; the Qwen2-VL vision path is the `Qwen2VL`
    /// orchestrator (`Models/Vision/Qwen2Vision.swift`) and the Qwen2.5-Omni
    /// path the shared `QwenOmniModel` engine (see `isOmni`).
    public static func variant(for _: ModelConfig) throws -> any Qwen2Variant.Type {
        Qwen2Dense.self
    }

    // ─── Qwen2.5-Omni ownership ───────────────────────────────────────
    //
    // Qwen2.5-Omni (text + vision + audio-IN) reuses the SAME shared
    // `QwenOmniModel` engine as Qwen3-Omni (it lives in `Models/Qwen3.swift`,
    // the canonical omni home). The only family-specific piece is detection:
    // the Qwen 2.x family owns the `qwen2_5_omni*` model_type / architecture
    // strings here, and the shared engine's `QwenOmniModel.handles` unions
    // these with `Qwen3.omniModelTypes`. Loading goes through
    // `ModelRegistry` → `QwenOmniModel.load`, which composes the Qwen 2.5
    // text backbone + Whisper-style audio tower (and, once wired, the Qwen 2.5
    // vision tower) — the right back-end stacks for the checkpoint.

    /// Omni model_type / architecture strings owned by the Qwen 2.x family
    /// (Qwen2.5-Omni). Unioned into `QwenOmniModel.modelTypes` /
    /// `architectures` by the shared engine.
    public static let omniModelTypes: Set<String> = [
        "qwen2_5_omni", "qwen2_5_omni_thinker",
    ]
    public static let omniArchitectures: Set<String> = [
        "Qwen2_5OmniForConditionalGeneration",
    ]

    /// Whether a config describes a **Qwen2.5-Omni** checkpoint. The omni
    /// model is assembled by the shared `QwenOmniModel` engine; this is the
    /// Qwen 2.x family's ownership of its omni model_type / architecture.
    public static func isOmni(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, omniModelTypes.contains(mt) { return true }
        return config.architecture.map(omniArchitectures.contains) ?? false
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

/// Qwen 2.x is Llama-shaped, so its text engine IS the in-tree
/// `Llama3Model`. The alias keeps the family's public surface self-named
/// while reusing the shared decoder.
public typealias Qwen2Model = Llama3Model

/// Every concrete Qwen 2 TEXT variant declares the capabilities it exposes
/// and the generation defaults it recommends (overriding / extending the
/// family baseline), and loads its text stack. Mirrors `Qwen3Variant` /
/// `Granite4Variant`.
public protocol Qwen2Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Qwen2Model
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Unified Qwen 2 family error — raised by the text variant
/// (`Qwen2Dense`) and the Qwen2-VL orchestrator + vision tower
/// (`Models/Vision/Qwen2Vision.swift`).
public enum Qwen2Error: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Qwen2: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "Qwen2: checkpoint is missing tensor '\(name)'"
        }
    }
}
