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
// QualityScorable — the single model-facing contract for quality
// telemetry. The metric math (perplexity, KL-divergence, NIAH-later)
// lives in Telemetry/ behind this protocol, so every model family is
// scored identically: the model only supplies its distributions.

import Foundation

/// A model that can be scored for quality telemetry.
///
/// Conformance exposes the two teacher-forced primitives the `Telemetry/`
/// metrics drive — make a fresh per-layer cache, and run one forced step that
/// returns full next-token logits. `Perplexity.compute` / `klDivergence`
/// consume this rather than a concrete model, so adding a new family is just a
/// conformance.
///
/// **Logits, not log-probs.** `scoringForward` returns raw logits so the
/// softmax / NLL / KL math stays centralized in `Telemetry/` and KLD compares a
/// reference's and a candidate's distributions apples-to-apples (same code path,
/// same precision). It also streams one position at a time — corpus-scale
/// scoring never materializes N vocab-wide tensors at once.
public protocol QualityScorable {
    /// A fresh set of per-layer state caches for a forced-decode scoring pass,
    /// independent of any live-generation caches.
    func makeScoringCaches(device: Device) -> [any LayerCacheProtocol]

    /// One teacher-forced step: queue `tokenId` at `position` against `caches`
    /// and return the next-token logits (full vocab).
    func scoringForward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol], device: Device
    ) -> Tensor
}

// The concrete `Model` conforms by delegating to its `engine`, so every current
// family is scorable for free — `makeLayerCaches` / `forward` are exactly the
// primitives the metrics need.
extension Model: QualityScorable {
    public func makeScoringCaches(device: Device) -> [any LayerCacheProtocol] {
        engine.makeLayerCaches(maxSeq: nil, device: device)
    }

    public func scoringForward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol], device: Device
    ) -> Tensor {
        engine.forward(tokenId: tokenId, position: position, caches: caches, device: device)
    }
}
