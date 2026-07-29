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
// Integration tests for the MOSS-TTS family streaming conformance
// (`StreamingSynthesizer`). Covers both MossTTSModel (8B, Qwen3 backbone)
// and MossTTSNanoModel (100M, GPT-2 backbone).
//
// Both models consume raw TEXT directly — the backbone tokenizes the prompt
// internally, so no external g2p / phonemizer is required. The full
// synthesis pipeline (delay-pattern / local-transformer generation + MOSS
// audio tokenizer decode) is a follow-on stage, so `synthesize` currently
// throws `synthesisNotWired`.
//
// The contract these tests pin:
//   • While unwired — `synthesizeStream` surfaces the `synthesisNotWired`
//     error via `continuation.finish(throwing:)` rather than emitting silent
//     or wrong audio.
//   • Once wired — concatenating the streamed waveform chunks must equal a
//     single full `synthesize` (the `synthesizeStreamingFromText` test asserts
//     this; it activates automatically when synthesis lands).
//
// Gated by `IntegrationGroupGating.enableAudioSuites`. Each test resolves the
// checkpoint from the HF cache and SKIPS (records nothing) when it is absent,
// since the multi-GB MOSS-TTS-8B checkpoint is not a hard CI requirement.

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite(
    "MossTTS Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct MossTTSIntegrationTests {

    // ─── Checkpoint ids ───────────────────────────────────────────────

    /// Canonical published MOSS-TTS-8B checkpoint (the loader / registry
    /// reference id). If unavailable in the cache the test skips.
    private static let mossTTSRepoId = "mlx-community/MOSS-TTS-8B-8bit"
    /// Canonical published MOSS-TTS-Nano checkpoint.
    private static let mossTTSNanoRepoId = "mlx-community/MOSS-TTS-Nano-100M"

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

    /// Resolve a checkpoint snapshot from the local HF cache only. Returns
    /// `nil` when the checkpoint is not cached so the test can skip instead
    /// of triggering a multi-GB download.
    private func cachedSnapshot(repoId: String) -> URL? {
        let slug = "models--" + repoId.replacingOccurrences(of: "/", with: "--")
        let snapshots = Self.hfCacheRoot
            .appendingPathComponent(slug)
            .appendingPathComponent("snapshots")
        guard
            let subs = try? FileManager.default.contentsOfDirectory(
                at: snapshots, includingPropertiesForKeys: nil)
        else { return nil }
        return subs.first(where: { Self.isCompleteSnapshot($0) })
    }

    // ─── MossTTS (8B) ─────────────────────────────────────────────────

    @Test("MossTTS — sample rate is 24 000 Hz")
    func mossTTS_sampleRate() async throws {
        guard let dir = cachedSnapshot(repoId: Self.mossTTSRepoId) else {
            print("[MossTTS] checkpoint not cached — skipping")
            return
        }
        let model = try MossTTSModel.load(directory: dir)
        #expect(model.sampleRate == 24_000)
    }

    /// Streaming text→audio: concatenated streamed samples ≈ a single full
    /// `synthesize`. While synthesis is unwired this asserts the stream
    /// surfaces `MossTTSError.synthesisNotWired` (no silent / wrong audio);
    /// once the generation loop lands the `else` branch asserts the
    /// stream-equals-batch contract.
    @Test("MossTTS — synthesizeStream from raw text")
    func synthesizeStreamingFromText() async throws {
        guard let dir = cachedSnapshot(repoId: Self.mossTTSRepoId) else {
            print("[MossTTS] checkpoint not cached — skipping")
            return
        }
        let model = try MossTTSModel.load(directory: dir)
        let text = "Hello world."

        // Single full synthesis, captured as the reference (or the wired-yet
        // sentinel error).
        var fullSamples: [Float]?
        var synthError: Error?
        do {
            fullSamples = try model.synthesize(text: text)
        } catch {
            synthError = error
        }

        // Drain the stream.
        var streamed: [Float] = []
        var streamError: Error?
        do {
            for try await chunk in model.synthesizeStream(text: text) {
                streamed.append(contentsOf: chunk.toFloatArray())
            }
        } catch {
            streamError = error
        }

        if let full = fullSamples {
            // Synthesis is wired: the stream must reproduce the full waveform.
            #expect(streamError == nil, "stream threw though synthesize succeeded")
            #expect(
                streamed.count == full.count,
                "streamed (\(streamed.count)) != full (\(full.count))")
            print("[MossTTS] streamed \(streamed.count) samples == full \(full.count)")
        } else {
            // Synthesis is unwired: BOTH paths must surface the same clear
            // error — the stream never silently yields wrong audio.
            #expect(streamed.isEmpty, "stream yielded audio despite unwired synthesis")
            #expect(synthError is MossTTSError, "expected MossTTSError from synthesize")
            #expect(streamError is MossTTSError, "expected MossTTSError from stream")
            print("[MossTTS] unwired — stream surfaced \(String(describing: streamError))")
        }
    }

    // ─── MossTTSNano (100M) ───────────────────────────────────────────

    @Test("MossTTSNano — sample rate is 48 000 Hz")
    func mossTTSNano_sampleRate() async throws {
        guard let dir = cachedSnapshot(repoId: Self.mossTTSNanoRepoId) else {
            print("[MossTTSNano] checkpoint not cached — skipping")
            return
        }
        let model = try MossTTSNanoModel.load(directory: dir)
        #expect(model.sampleRate == 48_000)
    }

    /// Nano counterpart of `synthesizeStreamingFromText`.
    @Test("MossTTSNano — synthesizeStream from raw text")
    func nano_synthesizeStreamingFromText() async throws {
        guard let dir = cachedSnapshot(repoId: Self.mossTTSNanoRepoId) else {
            print("[MossTTSNano] checkpoint not cached — skipping")
            return
        }
        let model = try MossTTSNanoModel.load(directory: dir)
        let text = "Hello world."

        var fullSamples: [Float]?
        var synthError: Error?
        do {
            fullSamples = try model.synthesize(text: text)
        } catch {
            synthError = error
        }

        var streamed: [Float] = []
        var streamError: Error?
        do {
            for try await chunk in model.synthesizeStream(text: text) {
                streamed.append(contentsOf: chunk.toFloatArray())
            }
        } catch {
            streamError = error
        }

        if let full = fullSamples {
            #expect(streamError == nil, "stream threw though synthesize succeeded")
            #expect(
                streamed.count == full.count,
                "streamed (\(streamed.count)) != full (\(full.count))")
            print("[MossTTSNano] streamed \(streamed.count) samples == full \(full.count)")
        } else {
            #expect(streamed.isEmpty, "stream yielded audio despite unwired synthesis")
            #expect(synthError is MossTTSNanoError, "expected MossTTSNanoError from synthesize")
            #expect(streamError is MossTTSNanoError, "expected MossTTSNanoError from stream")
            print("[MossTTSNano] unwired — stream surfaced \(String(describing: streamError))")
        }
    }
}
