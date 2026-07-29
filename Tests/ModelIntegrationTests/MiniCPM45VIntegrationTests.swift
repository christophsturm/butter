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
// Consolidated integration suite for MiniCPM-V-4.5 (`model_type: minicpmv`,
// `version: 4.5`): OpenBMB's image + video → text VLM. A Qwen3-8B dense
// backbone + SigLIP vision + perceiver resampler (the same resampler tower
// as MiniCPM-o, minus the audio half). ONE @Suite that loads the model
// ONCE (static memoized Task) and runs a per-capability inference pass —
// modeled on `MiniCPM46VIntegrationTests` / `MiniCPM45OmniIntegrationTests`.
//
// Runs against the Butter 4-bit conversion, referenced LOCALLY from the
// `ffai convert` output cache (`~/.cache/ffai/converts/…`). There is no
// mlx-community 4-bit of MiniCPM-V-4.5, so quantize the `openbmb/MiniCPM-V-4_5`
// bf16 base once:
//
//   ffai convert openbmb/MiniCPM-V-4_5 --bits 4 --embedding-bits 4 --lm-head-bits 4
//
// `Model.load` takes an id-or-path, so the test points at the local dir.

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite(
    "MiniCPM-V 4.5 Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct MiniCPM45VIntegrationTests {

    /// Local `ffai convert` output for the 4-bit MiniCPM-V-4.5 (no HF
    /// upload — referenced straight from the converts cache).
    static let modelId = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".cache/ffai/converts/openbmb--MiniCPM-V-4_5-4bit")
        .path

    /// The checkpoint, loaded ONCE and shared across every `@Test`.
    static let loaded = Task { try await loadModel(modelId) }

    // MARK: - load

    @Test("load — MiniCPM-V 4.5 loads as a VLM with image + video")
    func load() async throws {
        let m = try await Self.loaded.value
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.videoIn))
        // No audio — this is the V checkpoint, not the omni.
        #expect(!m.availableCapabilities.contains(.audioIn))
        #expect(m.textEngine.supportsEmbeddingInput)
        // Qwen3-8B dense backbone.
        #expect(m.textEngine.hidden == 4096)
        let vlm = try #require(m.vlModel)
        #expect(vlm.imageTokenCount == 64)  // resampler query_num
        #expect(vlm.imageTokenId == MiniCPM4.resamplerImageTokenId)
        #expect(vlm.videoTokenId == MiniCPM4.resamplerVideoTokenId)
        #expect(!vlm.supportsAudio)
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
        print("MiniCPM-V 4.5 text → \(text)")
        expectCoherentOutput(r.generatedTokens, minTokens: 4, label: "minicpmv45-text")
    }

    // MARK: - image → text

    @Test("image + text — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "MiniCPM-V 4.5 is not a VLM")

        let image = try VisionTestHelpers.dogImage()
        // MiniCPM-V image framing: `<image>` + `placeholder × image_feature_size(64)` + `</image>`, 
        // inside a chat turn. The 64 placeholders (vlm.imageTokenId, `<|image_pad|>`) are replaced 
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
        print("MiniCPM-V 4.5 image → \(text)")
        expectCoherentOutput(generated, minTokens: 8, label: "MiniCPM-V 4.5 image")
        VisionTestHelpers.expectMentionsDog(text, label: "MiniCPM-V 4.5")
    }

    // MARK: - video → text

    @Test("video + text — identifies the cat video")
    func videoTextGeneration() async throws {
        let m = try await Self.loaded.value
        let vlm = try #require(m.vlModel, "MiniCPM-V 4.5 is not a VLM")
        let videoTokenId = try #require(vlm.videoTokenId)

        // MiniCPM-V encodes each video frame as an independent image (64
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
        print("MiniCPM-V 4.5 video → \(text)")
        expectCoherentOutput(generated, minTokens: 4, label: "MiniCPM-V 4.5 video")
        #expect(
            text.lowercased().contains("cat"),
            "video caption should mention a cat — got: \(text)")
    }
}
