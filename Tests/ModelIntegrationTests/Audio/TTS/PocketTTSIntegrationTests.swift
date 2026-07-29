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
// Integration test: loads a real PocketTTS checkpoint from the HF cache
// and exercises config decoding + registry detection + the streaming
// synthesizer surface.
//
// ## Scope note — STAGED PORT
//
// PocketTTS's synthesis pipeline (sentencepiece tokenisation, flow LM
// autoregressive decode, Mimi codec decode) is not wired yet, so
// `synthesize` / `synthesizeStream` throw `PocketTTSError.synthesisNotWired`.
// This suite verifies the checkpoint loads, the config decodes, and the
// streaming surface surfaces that error cleanly (rather than shipping an
// empty or wrong waveform). When the synthesis pipeline lands, the streaming
// test should flip to assert a non-degenerate waveform (mirroring Kokoro /
// Soprano).
//
// DO NOT RUN this suite via `make test-integration` during CI — the
// checkpoint is multi-GB and requires an ML-capable Mac.

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite(
    "PocketTTS Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct PocketTTSIntegrationTests {

    /// Canonical HF repo id (Kyutai's published PocketTTS checkpoint).
    private static let repoId = "mlx-community/pocket-tts"

    private static var hfCacheRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    private static func isCompleteSnapshot(_ dir: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        else { return false }
        guard let entries = try? fm.contentsOfDirectory(atPath: dir.path)
        else { return false }
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    /// Resolve a PocketTTS checkpoint: local HF cache first, then download.
    private func resolveCheckpoint() async throws -> URL {
        let root = Self.hfCacheRoot
        let slug = "models--mlx-community--pocket-tts"
        let snapshots = root.appendingPathComponent(slug)
            .appendingPathComponent("snapshots")
        let fm = FileManager.default
        if let subs = try? fm.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil),
            let dir = subs.first(where: { Self.isCompleteSnapshot($0) })
        {
            return dir
        }
        return try await ModelLocator(downloader: ModelDownloader())
            .resolve(idOrPath: Self.repoId)
    }

    private func loadPocketTTS() async throws -> PocketTTSModel {
        let dir = try await resolveCheckpoint()
        return try PocketTTSModel.load(directory: dir)
    }

    // ─── Tests ────────────────────────────────────────────────────────

    @Test("load — PocketTTS config decodes flow LM + Mimi fields from real checkpoint")
    func load_decodesConfig() async throws {
        let model = try await loadPocketTTS()
        #expect(model.config.flowLM.transformer.dModel > 0)
        #expect(model.config.flowLM.transformer.numLayers > 0)
        #expect(model.config.mimi.sampleRate > 0)
        #expect(model.sampleRate == model.config.mimi.sampleRate)
        print(
            "[PocketTTS] Loaded: dModel=\(model.config.flowLM.transformer.dModel), "
                + "layers=\(model.config.flowLM.transformer.numLayers), "
                + "sampleRate=\(model.sampleRate)")
    }

    @Test("registry — the registry routes PocketTTS checkpoint to .pocketTTS")
    func registry_routesPocketTTS() async throws {
        let dir = try await resolveCheckpoint()
        let loaded = try await Model.load(dir.path)
        #expect(loaded.availableCapabilities == Capability.textToSpeech)
    }

    @Test("synthesizeStream — surfaces synthesisNotWired until the pipeline lands")
    func synthesizeStreamingFromText() async throws {
        let model = try await loadPocketTTS()

        // The synthesis pipeline is a follow-on stage; the stream must finish
        // by throwing the explicit `synthesisNotWired` error rather than
        // yielding any (empty / wrong) audio.
        var yielded = 0
        var caught: Error?
        do {
            for try await chunk in model.synthesizeStream(text: "Hello world.") {
                yielded += chunk.shape.first ?? 0
            }
        } catch {
            caught = error
        }
        #expect(yielded == 0, "stream yielded audio before synthesis is wired")
        guard case PocketTTSError.synthesisNotWired? = caught else {
            Issue.record(
                "expected PocketTTSError.synthesisNotWired, got \(String(describing: caught))")
            return
        }
        print("[PocketTTS] synthesizeStream correctly surfaced synthesisNotWired")
    }
}
