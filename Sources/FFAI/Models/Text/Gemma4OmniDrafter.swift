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
// Gemma 4 Omni drafter (`gemma4_unified_assistant`,
// `google/gemma-4-12B-it-assistant`) — the EAGLE/MTP speculative-decoding
// draft model for the Gemma 4 Omni 12B target.
//
// SCOPE (this file): config recognition + the decoded shape of the EAGLE
// drafter, so the checkpoint is RECOGNIZED (the dispatcher raises an
// actionable `draftModelNotGeneratable` for it), QUANTIZABLE (`ffai convert`
// is architecture-agnostic — it quantizes by tensor name/shape, no loader
// needed), and LISTED in `ffai models`. The drafter CANNOT generate
// standalone: it predicts tokens conditioned on the target model's hidden
// state, which the target then verifies.
//
// DEFERRED to Phase 8.7 (see `planning/plan.md`): the actual EAGLE draft
// loop. Unlike a plain Gemma4Dense backbone, the drafter's 4 layers ALL
// declare `num_kv_shared_layers == num_hidden_layers`, i.e. they reuse
// `shared_kv_states` from the TARGET's KV cache (not a within-model donor),
// so the existing `Gemma4Loader` KV-sharing path does not apply. The
// remaining work is the target-hidden fusion (`pre_projection`,
// `2·backbone_hidden → hidden`), the centroid head (`centroids`,
// `hidden → num_centroids` + top-`centroid_intermediate_top_k`),
// `post_projection` / `masked_embedding`, a `Drafter` / `SpeculativeVerifier`
// implementation, and a `--draft-model` CLI flag.

import Foundation

/// Decoded shape of the Gemma 4 Omni EAGLE drafter. The text dims come from
/// the nested `text_config` (a 4-layer Gemma-4-shaped backbone); the EAGLE
/// fields are top-level.
public struct Gemma4OmniDrafterConfig: Sendable {
    /// Target backbone hidden size the drafter fuses (`backbone_hidden_size`,
    /// 3840). `pre_projection` consumes `2 · backboneHiddenSize`.
    public let backboneHiddenSize: Int
    /// Drafter's own hidden size (`text_config.hidden_size`, 1024).
    public let hidden: Int
    /// Drafter backbone depth (`text_config.num_hidden_layers`, 4).
    public let numLayers: Int
    /// KV-shared layer count (`text_config.num_kv_shared_layers`). Equal to
    /// `numLayers` for the EAGLE drafter — all layers reuse the target's
    /// `shared_kv_states`, which is the Phase 8.7 wiring.
    public let numKvSharedLayers: Int
    /// Centroid-head vocabulary clusters (`num_centroids`, 2048).
    public let numCentroids: Int
    /// Top-k centroids evaluated per step (`centroid_intermediate_top_k`, 32).
    public let centroidTopK: Int
    /// Drafter vocabulary (`text_config.vocab_size`, 262144 — shared with the
    /// target).
    public let vocabSize: Int

    public static func decode(_ config: ModelConfig) throws -> Gemma4OmniDrafterConfig {
        guard let backbone = config.int("backbone_hidden_size") else {
            throw Gemma4Error.missingConfig("backbone_hidden_size")
        }
        let tc = config.nested("text_config") ?? config.raw
        guard let hidden = tc["hidden_size"] as? Int,
            let layers = tc["num_hidden_layers"] as? Int
        else {
            throw Gemma4Error.missingConfig("text_config.hidden_size / num_hidden_layers")
        }
        return Gemma4OmniDrafterConfig(
            backboneHiddenSize: backbone,
            hidden: hidden,
            numLayers: layers,
            numKvSharedLayers: (tc["num_kv_shared_layers"] as? Int) ?? 0,
            numCentroids: config.int("num_centroids") ?? 0,
            centroidTopK: config.int("centroid_intermediate_top_k") ?? 0,
            vocabSize: (tc["vocab_size"] as? Int) ?? 262_144)
    }
}

/// Namespace for the Gemma 4 Omni drafter. Generation is Phase 8.7; today
/// this only surfaces the recognized config shape (the dispatcher rejects
/// standalone generation via `ModelError.draftModelNotGeneratable`).
public enum Gemma4OmniDrafter {
    /// Recognized architecture / model_type strings (forwarded from the
    /// Gemma 4 family). Kept here too so the spec-decode wiring can find them.
    public static var architectures: Set<String> { Gemma4.drafterArchitectures }
    public static var modelTypes: Set<String> { Gemma4.drafterModelTypes }

    /// Decode the drafter config from a resolved checkpoint directory. Used
    /// by tests (and the future Phase 8.7 draft loop) to confirm the EAGLE
    /// shape without standing up a full generation engine.
    public static func config(from directory: URL) throws -> Gemma4OmniDrafterConfig {
        try Gemma4OmniDrafterConfig.decode(ModelConfig.load(from: directory))
    }
}
