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
// MiniCPM5Text — the TEXT component of the MiniCPM5 family. MiniCPM5-1B
// declares a plain `LlamaForCausalLM` / `llama` architecture, so its text
// backbone reuses the in-tree Llama dense stack. This is a thin adapter so
// the `MiniCPM5` entrypoint composes its component stacks uniformly — the
// seam where a future multimodal v5 adds vision / audio components.

import Foundation

public enum MiniCPM5Text {
    /// Load the text engine for a MiniCPM5 checkpoint via the Llama variant
    /// loader (MiniCPM5-1B is a standard GQA Llama shape).
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> any LanguageModel {
        try Llama.variant(for: config).loadModel(
            config: config, weights: weights, options: options, device: device)
    }
}
