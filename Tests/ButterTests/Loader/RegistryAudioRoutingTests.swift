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
// RegistryAudioRoutingTests — the audio / VAD tier of the unified
// `ModelRegistry` (Phase 4). Asserts the audio families fold into the one
// table as directory-loaded entries, the routing-query helpers answer
// correctly, and capabilities track kind. No GPU / model loading.

import Foundation
import Testing

@testable import ButterSwift

@Suite("Registry — audio tier")
struct RegistryAudioRoutingTests {

    /// Every audio entry is a directory-loaded family (has `loadFromDirectory`,
    /// no text `load` closure).
    @Test("audio entries are directory-loaded, never text-loaded")
    func audioEntriesAreDirectoryLoaded() {
        #expect(!ModelRegistry.audioEntries.isEmpty)
        for e in ModelRegistry.audioEntries {
            #expect(e.loadFromDirectory != nil, "\(e.family) missing loadFromDirectory")
            #expect(e.load == nil, "\(e.family) should not have a text load closure")
        }
    }

    /// Declared capabilities track the catalog kind.
    @Test("audio entry capabilities match kind")
    func capabilitiesMatchKind() {
        for e in ModelRegistry.audioEntries {
            switch e.kind {
            case .stt: #expect(e.capabilities == Capability.speechToText)
            case .tts: #expect(e.capabilities == Capability.textToSpeech)
            case .sts: #expect(e.capabilities == Capability.speechToSpeech)
            case .omni: #expect(e.capabilities == Capability.omniAudio)
            case .vad: #expect(e.capabilities.isEmpty)
            default: Issue.record("unexpected audio-tier kind \(e.kind) for \(e.family)")
            }
        }
    }

    /// A VAD checkpoint (matched by `model_type`) routes to the VAD tier.
    @Test("VAD config routes to a .vad entry")
    func vadConfigRoutes() {
        let config = ModelConfig(architecture: nil, modelType: "silero_vad", raw: [:])
        let entry = ModelRegistry.entry(for: config)
        #expect(entry?.kind == .vad)
        #expect(entry?.family == "Silero VAD")
        #expect(entry?.loadFromDirectory != nil)
        #expect(ModelRegistry.handlesAudio(config))
        // VAD has no output capability.
        #expect(ModelRegistry.capabilities(for: config)?.isEmpty == true)
    }

    /// A plain text checkpoint is never routed to an audio family.
    @Test("text config is not audio")
    func textConfigNotAudio() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama", raw: [:])
        #expect(!ModelRegistry.handlesAudio(config))
        #expect(ModelRegistry.entry(for: config)?.kind == .llm)
    }

    /// `detectVADKind` resolves the fine-grained VAD architecture by
    /// `model_type` and by directory-name fallback.
    @Test("detectVADKind by model_type and by directory name")
    func detectVADKind() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("butter-vadkind-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }

        // By model_type.
        let typed = base.appendingPathComponent("anything")
        try FileManager.default.createDirectory(at: typed, withIntermediateDirectories: true)
        try #"{"model_type": "smart_turn_v3"}"#.write(
            to: typed.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)
        #expect(try ModelRegistry.detectVADKind(in: typed) == .smartTurn)

        // By directory-name fallback (no recognizable model_type).
        let named = base.appendingPathComponent("silero-vad-v5")
        try FileManager.default.createDirectory(at: named, withIntermediateDirectories: true)
        #expect(try ModelRegistry.detectVADKind(in: named) == .sileroVAD)
    }

    /// The catalog still represents every implemented kind with no empty
    /// groups (audio kinds now load, not just list).
    @Test("catalog covers every audio kind")
    func catalogCoversAudioKinds() {
        let kinds = Set(ModelRegistry.catalog.map { $0.kind })
        for k in [ModelType.stt, .tts, .sts, .vad, .omni] {
            #expect(kinds.contains(k), "catalog missing \(k)")
        }
    }
}
