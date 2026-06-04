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
// Chatterbox integration test — loads a real Chatterbox checkpoint from the
// HF cache (resolved through the standard ModelLocator + ModelLoadLock
// pattern) and exercises the `StreamingSynthesizer` conformance.
//
// ## Scope note — STAGED PORT
//
// Chatterbox's two-stage pipeline (T3 backbone → S3Gen flow decoder →
// VoiceEncoder → HiFi-GAN vocoder) is not yet wired — `synthesize` throws
// `ChatterboxError.synthesisNotWired` (see Chatterbox.swift). So
// `synthesizeStream(text:voice:speed:)` finishes by throwing that same
// error rather than emitting incorrect audio.
//
// This suite therefore asserts the CONTRACT that holds today:
//   1. load + sampleRate + conformance,
//   2. streaming finishes-throwing the not-wired error AND its concatenated
//      samples are empty (no partial / wrong audio leaks through).
//
// When the synthesis pipeline lands, swap the error assertion for the
// equivalence assertion mirrored in KokoroIntegrationTests: concatenated
// streamed samples ≈ a single full `synthesize`.
//
// DO NOT RUN this suite via `make test-unit`. Run serialised with
//   make test-integration
// Heavy GPU usage; do NOT run in parallel with other ModelTests.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Chatterbox Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct ChatterboxIntegrationTests {

    /// Canonical HF repo id — the published mlx-community Chatterbox Turbo
    /// (GPT-2 backbone) fp16 conversion. The Regular LLaMA variant lives at
    /// `mlx-community/Chatterbox-TTS-fp16`.
    private static let repoId = "mlx-community/chatterbox-turbo-fp16"

    /// Resolve the cached snapshot directory through the standard
    /// ModelLocator path, serialised by ModelLoadLock so concurrent
    /// integration tests don't double-resolve the same checkpoint.
    private func resolveDirectory() async throws -> URL {
        try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
    }

    /// Load Chatterbox from the resolved snapshot directory.
    private func loadModel() async throws -> ChatterboxModel {
        let dir = try await resolveDirectory()
        return try ChatterboxModel.load(directory: dir, device: .shared)
    }

    @Test("load — Chatterbox config binds the 24 kHz output sample rate")
    func loadChatterbox_bindsSampleRate() async throws {
        let model = try await loadModel()
        #expect(model.sampleRate == model.config.sampleRate)
        #expect(model.sampleRate == 24_000)
        // The T3 backbone geometry decoded from config.json.
        #expect(model.config.t3.speechTokensDictSize > 0)
    }

    @Test("synthesizeStreamingFromText — streams a full waveform or throws clearly")
    func synthesizeStreamingFromText() async throws {
        let model = try await loadModel()

        // Drive the protocol-shaped streaming entry point from raw text
        // (Chatterbox's T3 stage tokenizes text internally — no external g2p).
        var streamed: [Float] = []
        var threwNotWired = false
        do {
            for try await chunk in model.synthesizeStream(
                text: "Hello, world.", voice: nil, speed: 1.0)
            {
                streamed.append(contentsOf: chunk.toFloatArray())
            }
        } catch let error as ChatterboxError {
            // Staged port: the T3 + S3Gen + HiFi-GAN pipeline is not wired,
            // so the stream MUST finish by throwing the descriptive FFAI
            // error rather than emitting incorrect audio.
            threwNotWired = true
            #expect(
                String(describing: error).contains("S3Gen"),
                "expected the synthesis-not-wired error, got: \(error)")
        }

        if threwNotWired {
            // No partial / wrong audio may leak through before the throw.
            #expect(
                streamed.isEmpty,
                "stream threw not-wired but still emitted \(streamed.count) samples")
            print("Chatterbox streaming: synthesis not wired — threw as expected.")
        } else {
            // Forward-compatible path: once synthesis lands, the concatenated
            // streamed samples must equal a single full `synthesize`.
            let full = try model.synthesize(text: "Hello, world.")
            #expect(
                streamed.count == full.count,
                "streamed (\(streamed.count)) != full synth (\(full.count))")
            #expect(streamed.allSatisfy { $0.isFinite }, "non-finite streamed samples")
            print("Chatterbox streaming: \(streamed.count) samples == full synth.")
        }
    }
}
