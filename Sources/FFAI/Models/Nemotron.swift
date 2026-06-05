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
// NemotronH family root — NVIDIA's Nemotron-H line, the stack-
// interleaved hybrid (Mamba 2 / attention / dense-MLP / MoE) decoder.
// Covers the dense Nemotron-H-4B-Base lineage AND the MoE
// Cascade-2-30B-A3B / Nemotron-3 Nano/Super/Ultra lineage (all
// `nemotron_h` model_type).
//
// This file is the **main model interface** for the NemotronH family:
//   • the family enum `NemotronH` (modelTypes, architectures, variant
//     dispatch),
//   • the `NemotronHVariant` protocol every concrete variant conforms to,
//   • the unified `NemotronHError` type the text loaders AND the
//     vision-language orchestrator raise.
//
// Concrete variant + per-layer impl live in
// `Models/Text/NemotronHText.swift` (`NemotronHHybrid`). The Nemotron
// Nano VL orchestrator (`enum NemotronVL`) — the ViT tower + multi-modal
// projector + the NemotronH hybrid text backbone — lives in
// `Models/Vision/NemotronHVision.swift` and raises the same
// `NemotronHError`.
//
// The sibling Nemotron-Labs-Diffusion line is a DIFFERENT architecture
// and lives in its own files: `Models/NemotronDiffusion.swift` (family
// root + variant protocol + error), `Models/Text/NemotronDiffusionText.swift`
// (the tri-mode decoder), and `Models/Vision/NemotronDiffusionVision.swift`
// (`enum NemotronDiffusionVL`).

import Foundation
import Metal

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
