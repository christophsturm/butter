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
// Embedding layer building blocks — the dense `Embedding`, the mlx-format
// `QuantizedEmbedding`, the type-erasing `AnyEmbedding` wrapper, and the
// `loadEmbedding` helper that builds the right variant from a SafeTensors
// bundle. Each layer holds its weight tensors as fields and exposes
// `parameters()` for SafeTensors weight binding. The shared
// `deriveAffineQuantBits` quant helper lives in `Linear.swift`.

import Foundation
import Metal

// ─── Embedding ───────────────────────────────────────────────────────

public final class Embedding: Module {
    /// weight shape [vocab_size, hidden_size]
    public let weight: Tensor

    public init(weight: Tensor) {
        precondition(weight.shape.count == 2, "Embedding: weight must be 2D")
        self.weight = weight
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight)]
    }

    /// Look up `tokenIds` (one-element u32 tensor for decode) and return
    /// [n_tokens, hidden] in the table's dtype.
    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.gather(table: weight, tokenIds: tokenIds, on: cmd)
    }
}

// ─── QuantizedEmbedding (mlx int4 format) ─────────────────────────────

public final class QuantizedEmbedding: Module {
    public let weight: Tensor  // [vocab, hidden/pack_factor] uint32
    public let scales: Tensor
    public let biases: Tensor
    public let hidden: Int
    public let bits: Int
    public let groupSize: Int

    public init(
        weight: Tensor, scales: Tensor, biases: Tensor,
        hidden: Int, bits: Int, groupSize: Int
    ) {
        precondition(
            bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
            "QuantizedEmbedding: bits must be one of 2, 3, 4, 5, 6, or 8")
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.hidden = hidden
        self.bits = bits
        self.groupSize = groupSize
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("scales", scales), ("biases", biases)]
    }

    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.dequantGather(
            weight: weight, scales: scales, biases: biases,
            tokenIds: tokenIds, hidden: hidden, bits: bits, groupSize: groupSize,
            on: cmd
        )
    }
}

/// Type-erasing wrapper over Embedding / QuantizedEmbedding so model
/// loaders don't need to template every call site.
public final class AnyEmbedding: Module {
    public let inner: any Module
    public let weight: Tensor  // expose for tying with lm_head
    private let forward: (Tensor, MTLCommandBuffer) -> Tensor

    public init(_ embed: Embedding) {
        self.inner = embed
        self.weight = embed.weight
        self.forward = { embed($0, on: $1) }
    }

    public init(_ embed: QuantizedEmbedding) {
        self.inner = embed
        self.weight = embed.weight
        self.forward = { embed($0, on: $1) }
    }

    public func parameters() -> [(String, Tensor)] { inner.parameters() }

    /// The float dtype activations flow in — i.e. the embedding's *output*
    /// dtype. For a quantized embedding the packed `weight` is `u32`, so the
    /// activation dtype is the quant scales' (float) dtype; a plain embedding
    /// uses its weight dtype. Callers building activation buffers must use
    /// this, never `weight.dtype`, or they crash on quantized checkpoints.
    public var activationDtype: DType {
        if let q = inner as? QuantizedEmbedding { return q.scales.dtype }
        return weight.dtype
    }

    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        forward(tokenIds, cmd)
    }
}

/// Build the right Embedding variant depending on quantization presence.
public func loadEmbedding(
    base: String, in bundle: SafeTensorsBundle,
    hidden: Int, quantization: ModelConfig.QuantizationConfig?
) throws -> AnyEmbedding {
    if let q = quantization, bundle.isQuantized(base) {
        let t = try bundle.quantizedTriplet(base)
        let bits = deriveAffineQuantBits(
            weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
            scaleCols: t.scales.shape[t.scales.shape.count - 1],
            groupSize: q.groupSize)
        if [2, 3, 4, 5, 6, 8].contains(bits) {
            return AnyEmbedding(
                QuantizedEmbedding(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    hidden: hidden, bits: bits, groupSize: q.groupSize
                ))
        }
    }
    return AnyEmbedding(Embedding(weight: try bundle.tensor(named: "\(base).weight")))
}
