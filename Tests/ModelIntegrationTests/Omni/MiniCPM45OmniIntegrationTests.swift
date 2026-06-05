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
// Consolidated integration suite for MiniCPM-o-4.5 (`model_type: minicpmo`),
// OpenBMB's omni VLM: text + image + video + audio-in → text. A Qwen3-8B
// dense backbone + SigLIP vision + perceiver resampler + a Whisper audio
// tower. ONE @Suite that loads the model ONCE (static memoized Task) and
// runs a per-capability inference pass — modeled on
// `Gemma4OmniIntegrationTests`.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "MiniCPM-o 4.5 Omni Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableOmniSuites,
        IntegrationGroupGating.omniSkipReason)
)
struct MiniCPM45OmniIntegrationTests {

    static let modelId = "mlx-community/MiniCPM-o-4_5-4bit"

    /// The omni checkpoint, loaded ONCE and shared across every `@Test`.
    static let loaded = Task { try await loadModel(modelId) }

    /// 0.5 s 440 Hz mono sine at 16 kHz — a non-degenerate audio signal.
    static func sine(seconds: Double = 0.5) -> [Float] {
        let n = Int(seconds * 16_000)
        return (0 ..< n).map {
            0.3 * Float(sin(2.0 * .pi * 440.0 * Double($0) / 16_000.0))
        }
    }

    // MARK: - load

    @Test("load — MiniCPM-o 4.5 loads as a VLM with image + video + audio")
    func load() async throws {
        let m = try await Self.loaded.value
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.videoIn))
        #expect(m.availableCapabilities.contains(.audioIn))
        #expect(m.textEngine.supportsEmbeddingInput)
        // Qwen3-8B dense backbone.
        #expect(m.textEngine.hidden == 4096)
        let vlm = try #require(m.vlModel)
        #expect(vlm.imageTokenCount == 64)  // resampler query_num
        #expect(vlm.supportsAudio)
    }

    // MARK: - text

    @Test("text — Qwen3-8B backbone chat-generates coherently")
    func textGeneration() async throws {
        let m = try await Self.loaded.value
        let r = try await m.generate(
            messages: [
                ChatMessage(
                    role: .user,
                    content: "Name three primary colors, comma-separated.")
            ],
            parameters: m.defaultGenerationParameters.with {
                $0.maxTokens = 64
                $0.temperature = 0
            })
        let text = m.textTokenizer.decode(tokens: r.generatedTokens, skipSpecialTokens: true)
        print("MiniCPM-o text → \(text)")
        expectCoherentOutput(r.generatedTokens, minTokens: 4, label: "minicpmo-text")
    }

    // MARK: - image → text

    @Test("image + text — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "MiniCPM-o is not a VLM")

        let image = try VisionTestHelpers.dogImage()
        // MiniCPM-o image framing: `<image>` + `<unk>×image_feature_size(64)` + `</image>`,
        // inside a chat turn. The 64 `<unk>` (vlm.imageTokenId) placeholders are replaced 
        // by the resampler's 64 image tokens.
        let nImg = vlm.encodeImage(image).shape[0]
        let pre = m.textTokenizer.encode(text: "<|im_start|>user\n<image>")
        let post = m.textTokenizer.encode(
            text: "</image>\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n")
        let prompt = pre + Array(repeating: vlm.imageTokenId, count: nImg) + post

        let generated = try vlm.generate(
            promptTokens: prompt, image: image, maxTokens: 200,
            eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("MiniCPM-o image → \(text)")
        expectCoherentOutput(generated, minTokens: 8, label: "MiniCPM-o image")
        VisionTestHelpers.expectMentionsDog(text, label: "MiniCPM-o")
    }

    // MARK: - audio → text

    @Test("audio + text — raw-frame audio path decodes coherently")
    func audioTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "MiniCPM-o is not a VLM")
        let audioTokenId = try #require(vlm.audioTokenId)

        let wav = Self.sine()
        // MiniCPM-o audio framing: `<|audio_start|>`
        // + `<unk>×output_len` + `<|audio_end|>`. The placeholder count must
        // match the whisper-tower → projector → avg-pool(5) output length.
        let nAud = try vlm.encodeAudio(wav).shape[0]
        #expect(nAud > 0)
        let pre = m.textTokenizer.encode(text: "<|im_start|>user\n<|audio_start|>")
        let post = m.textTokenizer.encode(
            text: "<|audio_end|>\nWhat do you hear?<|im_end|>\n<|im_start|>assistant\n")
        let prompt = pre + Array(repeating: audioTokenId, count: nAud) + post

        let generated = try vlm.generate(
            promptTokens: prompt, audio: wav, maxTokens: 64,
            eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("MiniCPM-o audio → \(text)")
        expectCoherentOutput(generated, minTokens: 3, label: "MiniCPM-o audio")
    }

    // MARK: - video → text

    @Test("video + text — identifies the cat video")
    func videoTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "MiniCPM-o is not a VLM")
        let videoTokenId = try #require(vlm.videoTokenId)

        // MiniCPM-o encodes each video frame as an independent image (64
        // resampler tokens/frame). Frame each `<image>`-wrapped placeholder
        // run; the splice substitutes per-frame video tokens IN ORDER.
        let frames = try VisionTestHelpers.catVideoFrames(maxFrames: 3)
        let perFrame = vlm.encodeImage(frames[0]).shape[0]
        let pre = m.textTokenizer.encode(text: "<|im_start|>user\n")
        let post = m.textTokenizer.encode(
            text: "\nWhat animal is in this video?<|im_end|>\n"
                + "<|im_start|>assistant\n")
        var span: [Int] = []
        for _ in frames {
            span += m.textTokenizer.encode(text: "<image>")
            span += Array(repeating: videoTokenId, count: perFrame)
            span += m.textTokenizer.encode(text: "</image>")
        }
        let prompt = pre + span + post

        let generated = try vlm.generate(
            promptTokens: prompt, videoFrames: frames, maxTokens: 96,
            eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)
        let text = m.textTokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("MiniCPM-o video → \(text)")
        expectCoherentOutput(generated, minTokens: 4, label: "MiniCPM-o video")
        #expect(
            text.lowercased().contains("cat"),
            "video caption should mention a cat — got: \(text)")
    }
}
