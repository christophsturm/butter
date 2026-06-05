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
// MiniCPM4Text — the TEXT component of the MiniCPM4 family (MiniCPM-V 4.5 /
// 4.6, and the MiniCPM-o 4.5 omni backbone). The MiniCPM4 line reuses the
// Qwen3.5 hybrid text stack (`qwen3_5_text`), so this is a thin adapter
// that loads the `Qwen35Hybrid` engine from the checkpoint's text sub-tree.
//
// It exists as a named component so the `MiniCPM4` entrypoint composes its
// text / vision (`MiniCPM4Vision`) / audio (`MiniCPM4Audio`, omni-only)
// stacks uniformly — mirroring the per-modality component split the other
// multimodal families use.

import Foundation

public enum MiniCPM4Text {
    /// Load the Qwen3.5 hybrid text backbone for a MiniCPM4 checkpoint.
    ///
    /// mlx-community ships the text weights under `language_model.model.*`
    /// (an outer `language_model.` wrapping the inner `model.`). The
    /// Qwen3.5 loader probes that prefix candidate itself, so the full
    /// bundle is passed through unmodified.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Qwen35Model {
        try Qwen35Hybrid.loadModel(
            config: config, weights: weights, options: options, device: device)
    }
}
