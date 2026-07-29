// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// RMSNorm — root-mean-square layer normalization with a learned per-channel
// scale (no shift). The normalization the Llama / Qwen / Gemma transformer
// backbones use. Holds its weight tensor as a field and exposes
// `parameters()` for SafeTensors weight binding.

import Foundation
import Metal

public final class RMSNorm: Module {
    /// weight shape [n] — per-channel scale.
    public let weight: Tensor
    public let eps: Float

    public init(weight: Tensor, eps: Float) {
        self.weight = weight
        self.eps = eps
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.rmsNorm(x, weight: weight, eps: eps, on: cmd)
    }
}
