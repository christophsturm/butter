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
// Slow integration test: downloads (or hits cache) the SileroVAD
// checkpoint and runs voice-activity detection on a synthetic clip.
//
// SileroVAD is tiny (~1.5M params, a few MB on disk), so the
// integration suite stays fast. The test clip is generated in-process —
// a 200 Hz tone burst surrounded by silence — so no audio fixture is
// needed; the burst should register as at least one speech segment.
//
// This suite is assertive: a load failure FAILS the test rather than
// silently skipping. The contract is "the SileroVAD checkpoint loads
// and actually detects the tone burst as speech".

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "SileroVAD Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct SileroVADIntegrationTests {

    /// The published SileroVAD checkpoint.
    static let repoId = "mlx-community/silero-vad"

    /// A mono 16 kHz silence→two-tone-burst→silence clip energetic enough
    /// to trip the speech detector — `AudioTestHelpers.voiceLikeClip`
    /// defaults (200 Hz fundamental, 600 ms silence, 1200 ms burst).
    private func syntheticClip() -> [Float] { AudioTestHelpers.voiceLikeClip() }

    /// Load the SileroVAD model, holding the global model-load lock. A
    /// load failure throws — it is NOT caught and skipped, so a broken
    /// loader fails the suite.
    private func loadSileroVAD() async throws -> SileroVADModel {
        try await ModelLoadLock.shared.loadSerially {
            try await SileroVADModel.fromPretrained(Self.repoId)
        }
    }

    @Test("load + detect produces a finite probability stream with speech")
    func loadAndDetect() async throws {
        let model = try await loadSileroVAD()

        // Branch geometry from the published config.
        #expect(model.config.branch16k.chunkSize == 512)
        #expect(model.config.branch16k.contextSize == 64)
        #expect(model.config.branch8k.chunkSize == 256)

        let clip = syntheticClip()
        let output = try model.detect(audio: clip, sampleRate: 16000)

        // The probability stream must be finite and in [0, 1].
        #expect(output.isWellFormed)
        #expect(!output.probabilities.isEmpty)
        #expect(output.frameStrideSamples == 512)
        #expect(output.sampleRate == 16000)

        // Frame count should match: ceil(clipLen / chunkSize).
        let expectedFrames = (clip.count + 511) / 512
        #expect(output.probabilities.count == expectedFrames)

        // The tone burst should produce at least one speech segment, and
        // some frame should cross a generous probability floor.
        let maxProb = output.probabilities.max() ?? 0
        #expect(
            maxProb > 0.3,
            "max speech probability \(maxProb) — the forward pass should detect the tone burst")
        #expect(!output.segments.isEmpty)

        // Detected speech should not span the whole clip — the leading
        // and trailing silence should keep total speech below the full
        // duration.
        if let first = output.segments.first {
            #expect(first.durationSeconds > 0)
        }
        #expect(output.totalSpeechSeconds < Double(clip.count) / 16000.0)
    }

    @Test("empty clip yields an empty, well-formed result")
    func emptyClip() async throws {
        let model = try await loadSileroVAD()
        let output = try model.detect(audio: [], sampleRate: 16000)
        #expect(output.probabilities.isEmpty)
        #expect(output.segments.isEmpty)
        #expect(output.isWellFormed)
    }

    @Test("unsupported sample rate is rejected")
    func unsupportedSampleRate() async throws {
        let model = try await loadSileroVAD()
        #expect(throws: SileroVADError.self) {
            _ = try model.detect(audio: [0, 0, 0], sampleRate: 44100)
        }
    }
}
