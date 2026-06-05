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
// ModelEntry — one row of the `ModelRegistry` table: everything needed to
// (a) RECOGNISE a checkpoint, (b) LOAD it, and (c) LIST it in the catalog.
// One source of truth for both the loader and the `ffai models` CLI,
// replacing the hand-written dispatch chain + the separate CLI catalog.
//
// `matches(_:)` is the routing predicate — model_type / architecture
// membership, plus an optional `sniff` closure for families that can't be
// identified by those alone (a `text_config` model_type, a `_name_or_path`
// marker, a vision-weight probe). `load` is the family's former dispatch
// branch verbatim: it self-validates config + weights and throws a typed
// family error on mismatch.

import Foundation

public struct ModelEntry: Sendable {
    /// Human-facing family name (e.g. "Qwen 3"). Catalog display + logs.
    public let family: String
    /// Primary catalog category.
    public let kind: ModelKind
    /// `config.json` `model_type` strings this family dispatches on.
    public let modelTypes: Set<String>
    /// `config.json` `architectures[0]` strings this family dispatches on.
    public let architectures: Set<String>
    /// One-line catalog summary for `ffai models`.
    public let summary: String
    /// Example HuggingFace repo ids that load + decode through FFAI.
    public let repos: [String]
    /// Optional custom matcher for families not identifiable by
    /// model_type / architecture alone (runs in addition to the set
    /// membership check, OR'd in).
    let sniff: (@Sendable (ModelConfig) -> Bool)?
    /// Load the checkpoint through the text/VL dispatch path. The closure
    /// is the family's dispatch branch — it self-validates and throws a
    /// typed family error on mismatch. `nil` for **catalog-only** rows: the
    /// audio / VAD families list in `ffai models` but load through their own
    /// (audio) registry until the unified-engine phase wires them here.
    let load:
        (@Sendable (ModelConfig, SafeTensorsBundle, LoadOptions, Device) throws ->
            ModelRegistry.Loaded)?

    public init(
        family: String,
        kind: ModelKind,
        modelTypes: Set<String> = [],
        architectures: Set<String> = [],
        summary: String = "",
        repos: [String] = [],
        sniff: (@Sendable (ModelConfig) -> Bool)? = nil,
        load: (@Sendable (ModelConfig, SafeTensorsBundle, LoadOptions, Device)
            throws -> ModelRegistry.Loaded)? = nil
    ) {
        self.family = family
        self.kind = kind
        self.modelTypes = modelTypes
        self.architectures = architectures
        self.summary = summary
        self.repos = repos
        self.sniff = sniff
        self.load = load
    }

    /// Whether this entry handles `config` — by `model_type`,
    /// `architectures[0]`, or the custom `sniff`.
    public func matches(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) { return true }
        return sniff?(config) ?? false
    }
}
