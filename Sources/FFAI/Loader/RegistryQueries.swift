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
// Routing-query helpers on the unified `ModelRegistry` — the "what would load
// this / what can it do" questions, answered against the one table. These
// replaced the query API on the deleted `AudioModelRegistry` /
// `VADModelRegistry` (their LOADERS folded into `Model.load` + the audio
// `ModelEntry.loadFromDirectory` closures; these read-only queries live on).

import Foundation

// ─── VAD taxonomy ─────────────────────────────────────────────────────
//
// The fine-grained VAD architecture label. The unified `ModelEntry.kind` is
// `.vad` for every VAD family; this enum is the sub-classification the
// VAD-routing query / tests use to tell silero from smart-turn from
// sortformer. (Formerly `AudioModelKind` on `VADModelRegistry`.)

public enum AudioModelKind: String, Sendable, CaseIterable {
    case sileroVAD
    case smartTurn
    case sortformer
    case tenVAD
    case fireRedVAD

    /// The `model_type` strings (from `config.json`) that map to this
    /// architecture.
    public var modelTypes: Set<String> {
        switch self {
        case .sileroVAD: return ["silero_vad"]
        case .smartTurn: return ["smart_turn", "smart_turn_v3", "smart-turn"]
        case .sortformer: return ["sortformer", "diar_sortformer"]
        case .tenVAD: return ["ten_vad", "ten-vad", "tenvad"]
        case .fireRedVAD: return ["firered_vad", "firered-vad", "fireredvad"]
        }
    }
}

public enum AudioModelError: Error, CustomStringConvertible {
    case unknownArchitecture(String)
    case missingConfig(URL)

    public var description: String {
        switch self {
        case .unknownArchitecture(let m):
            return "ModelRegistry: unrecognized audio model_type: \(m)"
        case .missingConfig(let url):
            return "ModelRegistry: config.json not found at \(url.path)"
        }
    }
}

extension ModelRegistry {

    /// The `ModelKind`s that denote an audio family (anything `Model.load`
    /// routes through the audio `loadFromDirectory` path).
    static let audioKinds: Set<ModelKind> = [.stt, .tts, .sts, .vad, .omni]

    /// The declared capability set of the AUDIO family that would load
    /// `config`, or `nil` when `config` is not an audio family (it's a
    /// text/VL model, or unrecognized). Audio-scoped to exactly mirror the
    /// former `AudioModelRegistry.capabilities(for:)` — a non-audio config
    /// answers `nil`, not its text capabilities. Use
    /// `entry(for:)?.capabilities` for the any-kind declared set.
    public static func capabilities(for config: ModelConfig) -> Set<Capability>? {
        guard let e = entry(for: config), audioKinds.contains(e.kind) else { return nil }
        return e.capabilities
    }

    /// The capability set a checkpoint at `directory` would expose, without
    /// loading its weights. `nil` when no family claims it or the config
    /// can't be read. Replaces `AudioModelRegistry.capabilities(forConfigAt:)`.
    public static func capabilities(forConfigAt directory: URL) -> Set<Capability>? {
        guard let config = try? ModelConfig.load(from: directory) else { return nil }
        return capabilities(for: config)
    }

    /// Whether `config` is recognized as an audio family (STT / TTS / STS /
    /// VAD / omni). Replaces `AudioModelRegistry.handles(_:)`.
    public static func handlesAudio(_ config: ModelConfig) -> Bool {
        guard let kind = entry(for: config)?.kind else { return false }
        return audioKinds.contains(kind)
    }

    /// The fine-grained VAD architecture a checkpoint directory describes —
    /// by `model_type`, falling back to directory-name heuristics. Replaces
    /// `VADModelRegistry.detectKind(in:)`.
    public static func detectVADKind(in directory: URL) throws -> AudioModelKind {
        let configURL = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
            let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        {
            let candidates = [
                raw["model_type"] as? String,
                raw["architecture"] as? String,
            ].compactMap { $0?.lowercased() }
            for candidate in candidates {
                for kind in AudioModelKind.allCases where kind.modelTypes.contains(candidate) {
                    return kind
                }
            }
        }
        // Fallback: infer from the directory name.
        let name = directory.lastPathComponent.lowercased()
        if name.contains("silero") { return .sileroVAD }
        if name.contains("smart-turn") || name.contains("smart_turn") { return .smartTurn }
        if name.contains("sortformer") || name.contains("diar") { return .sortformer }
        if name.contains("ten-vad") || name.contains("ten_vad") || name.contains("tenvad") {
            return .tenVAD
        }
        if name.contains("firered") && name.contains("vad") { return .fireRedVAD }
        throw AudioModelError.unknownArchitecture(directory.lastPathComponent)
    }
}
