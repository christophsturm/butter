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
// Nemotron family — NVIDIA's Nemotron model line. Unified entry point
// covering four variants:
//
//   • NemotronH (text) — the stack-interleaved hybrid (Mamba 2 /
//     attention / dense-MLP / MoE) decoder. Covers the dense
//     Nemotron-H-4B-Base lineage AND the MoE Cascade-2-30B-A3B /
//     Nemotron-3 Nano/Super/Ultra lineage (all `nemotron_h` model_type).
//     Family enum + variant protocol + unified error live below; impl
//     lives in `Models/Text/NemotronHText.swift`.
//   • NemotronH (vision-language) — Nemotron Nano VL: the ViT vision
//     tower + multi-modal projector + the NemotronH hybrid text
//     backbone. Orchestrator (`enum NemotronVL`) lives in
//     `Models/Vision/NemotronHVision.swift` alongside the tower
//     internals; raises the same unified `NemotronHError` as the text
//     path.
//   • NemotronDiffusion (text) — Nemotron-Labs-Diffusion, the
//     tri-mode (AR / block-diffusion / self-speculation) decoder.
//     Family enum + variant protocol + unified error live in
//     `Models/NemotronDiffusion.swift`; impl lives in
//     `Models/Text/NemotronDiffusionText.swift`.
//   • NemotronDiffusion (vision-language) — Nemotron-Labs-Diffusion-
//     VLM-8B: the Pixtral 2D-RoPE ViT + Mistral3-style patch-merger
//     projector + the NemotronDiffusion tri-mode text backbone.
//     Orchestrator (`enum NemotronDiffusionVL`) lives in
//     `Models/Vision/NemotronDiffusionVision.swift` (it composes types
//     from three unrelated families — Pixtral tower + Mistral3 projector
//     + NemotronDiffusion text).
//
// The `enum Nemotron` below is the unified family root — it advertises
// the union of every variant's `modelTypes` + `architectures` so the
// ModelRegistry can ask "is this a Nemotron checkpoint?" with one
// lookup. Per-variant dispatch still routes to the right loader (the
// NemotronH text impl, the NemotronVL block in
// `Models/Vision/NemotronHVision.swift`, the NemotronDiffusion text
// impl, the NemotronDiffusionVL block in
// `Models/Vision/NemotronDiffusionVision.swift`).

import Foundation
import Metal

/// Unified Nemotron family root. Covers the text NemotronH backbone
/// (dense + MoE), the NemotronH vision-language wrapper, the
/// NemotronDiffusion tri-mode decoder, and its vision-language
/// wrapper. Each variant retains its own per-variant enum
/// (`NemotronH`, `NemotronVL`, `NemotronDiffusion`,
/// `NemotronDiffusionVL`) — this root just unions the metadata so the
/// registry can dispatch with one membership check.
public enum Nemotron {
    /// Union of every Nemotron variant's HuggingFace `model_type`
    /// labels. Used by `ModelRegistry` to recognise "this is a
    /// Nemotron checkpoint" before dispatching to the per-variant
    /// loader.
    public static var modelTypes: Set<String> {
        NemotronH.modelTypes
            .union(NemotronVL.modelTypes)
            .union(NemotronDiffusion.modelTypes)
            .union(NemotronDiffusionVL.modelTypes)
    }

    /// Union of every Nemotron variant's HuggingFace `architectures[0]`
    /// labels.
    public static var architectures: Set<String> {
        NemotronH.architectures
            .union(NemotronDiffusion.architectures)
            .union(NemotronDiffusionVL.architectures)
        // NemotronVL doesn't ship a distinct architecture string — the
        // VL checkpoints carry `text_config.model_type = nemotron_h`
        // and the registry routes them via the vision-config sniff.
    }
}

// ─── NemotronH text family entry point ───────────────────────────────

public enum NemotronH {
    public static let modelTypes: Set<String> = ["nemotron_h"]
    public static let architectures: Set<String> = ["NemotronHForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any NemotronHVariant.Type {
        _ = config
        return NemotronHHybrid.self
    }
}

// ─── NemotronH variant protocol ──────────────────────────────────────

public protocol NemotronHVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> NemotronHModel
}

// ─── NemotronH errors ────────────────────────────────────────────────

/// Unified NemotronH family error — raised by both the text loaders
/// (`NemotronHHybrid.loadModel`) and the NemotronH vision-language
/// orchestrator (`NemotronVL.load` in
/// `Models/Vision/NemotronHVision.swift`).
public enum NemotronHError: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingTensor(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f): return "NemotronH: required config field missing: \(f)"
        case .missingTensor(let name): return "NemotronH: checkpoint is missing tensor '\(name)'"
        case .unsupportedConfig(let m): return "NemotronH: unsupported config: \(m)"
        }
    }
}
