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
// LoRAAdaptable — the capability protocol for engines that support
// hot-swappable LoRA adapters. `Model`'s LoRA surface (`hasLoRA` /
// `loadLoRA` / `unloadLoRA`) delegates to whichever loaded engine conforms,
// so adding LoRA support to a new family is a one-line conformance with no
// special-casing in the `Model` wrapper. The default for a non-conforming
// model is "no LoRA" — `Model.hasLoRA` returns `false` and load/unload no-op.

import Foundation

/// An engine that can attach / detach a LoRA adapter at runtime. The
/// conformer owns the adapter state.
///
/// `attachLoRA(from:)` MUST **silently no-op when no adapter is found** at
/// `directory` — that contract is what lets `Model.load` auto-attach a bundled
/// adapter for *any* conforming family without knowing whether the checkpoint
/// actually shipped one.
public protocol LoRAAdaptable {
    /// Whether a LoRA adapter is currently attached.
    var hasLoRA: Bool { get }

    /// Load + attach a LoRA adapter from `directory` — either a model
    /// directory whose adapter subfolder is resolved, or a directory holding
    /// the adapter outright. Replaces any currently-attached adapter.
    /// **No-ops when no adapter is present.** Do not call during an active
    /// generate.
    func attachLoRA(from directory: URL, device: Device)

    /// Detach the current adapter. No-op when none is attached.
    func detachLoRA()
}

// NemotronLabs-Diffusion is the first LoRA-adaptable family — its optional
// `linear_spec_lora/` adapter sharpens the self-speculation diffusion
// drafter. The model already implements the full surface, so this is a marker
// conformance.
extension NemotronDiffusionModel: LoRAAdaptable {}
