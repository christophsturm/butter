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
// LayerNorm — standard layer normalization with a learned scale + shift.
// The normalization the vision-transformer encoders (SigLIP / CLIP) use
// instead of RMSNorm. Holds its weight + bias tensors as fields and exposes
// `parameters()` for SafeTensors weight binding.

import Foundation
import Metal

/// Standard LayerNorm with learned scale + shift — the normalization
/// vision-transformer encoders (SigLIP / CLIP) use instead of RMSNorm.
/// `callAsFunction` treats `x` as a single `[rowSize]` row; multi-row
/// callers (the vision encoder, processing all patch tokens at once)
/// call `Ops.layerNorm` directly with an explicit `nRows`.
public final class LayerNorm: Module {
    /// weight shape [n] — per-channel scale (`gamma`).
    public let weight: Tensor
    /// bias shape [n] — per-channel shift (`beta`).
    public let bias: Tensor
    public let eps: Float

    public init(weight: Tensor, bias: Tensor, eps: Float) {
        precondition(
            weight.shape == bias.shape,
            "LayerNorm: weight/bias shape mismatch")
        self.weight = weight
        self.bias = bias
        self.eps = eps
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("bias", bias)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.layerNorm(
            x, weight: weight, bias: bias, eps: eps,
            nRows: 1, rowSize: x.elementCount, on: cmd)
    }
}
