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
// GenerateImage / GenerateVideo / GenerateAudio — the multimodal-input
// subcommands of `ffai generate`. Each loads a vision/omni model, encodes
// the media to soft tokens, builds a family-agnostic prompt
// (MultimodalPrompt), and streams the decoded text. Streaming is the
// primary route; `--no-streaming` buffers and prints once.

import ArgumentParser
import FFAI
import Foundation
import Tokenizers

/// The audio sample rate VLM/omni audio front-ends expect (16 kHz mono).
private let kAudioInputSampleRate = 16000

// MARK: - Shared stream consumer

/// The result of draining a VLM generation stream — enough to print the
/// standard footer.
struct VLMStreamOutcome {
    let tokens: [Int]
    let text: String
    let stats: GenerationStats?
}

enum VLMStreamConsumer {
    /// Drain a `VisionModel` generation stream. `VisionModel` chunks carry
    /// token ids only (it has no tokenizer), so we detokenize here. When
    /// `streaming`, we decode the FULL running token buffer each step and
    /// print only the newly-revealed suffix — this is the robust
    /// detokenization trick: it survives multi-token glyphs and BPE merges
    /// that a per-token `decode` would mangle.
    static func consume(
        _ stream: AsyncThrowingStream<GenerationChunk, Error>,
        tokenizer: any Tokenizer,
        streaming: Bool
    ) async throws -> VLMStreamOutcome {
        var collected: [Int] = []
        var printed = ""
        var stats: GenerationStats?
        for try await chunk in stream {
            collected.append(contentsOf: chunk.tokens)
            if let s = chunk.stats { stats = s }
            guard streaming else { continue }
            let full = tokenizer.decode(tokens: collected, skipSpecialTokens: true)
            if full.count > printed.count {
                print(String(full.dropFirst(printed.count)), terminator: "")
                fflush(stdout)
                printed = full
            }
        }
        let full = tokenizer.decode(tokens: collected, skipSpecialTokens: true)
        if streaming {
            print("")
        } else {
            print(full)
        }
        return VLMStreamOutcome(tokens: collected, text: full, stats: stats)
    }
}

/// Shared helper: resolve the VLM, assert the capability, print the
/// prompt banner, and return the model + vision model.
private func requireVLM(
    _ common: CommonGenerateOptions, capability: Capability, label: String
) async throws -> (Model, VisionModel) {
    try common.applyGlobalFlags()
    let m = try await common.loadModel()
    guard let vlm = m.vlModel else {
        throw ValidationError("\(common.model) is not a vision/omni model (no vlModel).")
    }
    guard m.availableCapabilities.contains(capability) else {
        throw ValidationError(
            "\(common.model) does not support \(label) input (missing \(capability.rawValue)).")
    }
    return (m, vlm)
}

// MARK: - image

/// `ffai generate image <file>` — image + prompt → text.
struct GenerateImageCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "image",
        abstract: "Describe or answer questions about an image."
    )

    @OptionGroup var common: CommonGenerateOptions

    @Argument(help: "Path to an image file (JPEG / PNG / any ImageIO format).")
    var imageFile: String

    @Option(name: .long, help: "Instruction / question about the image.")
    var prompt: String = "Describe this image."

    func run() async throws {
        let (m, vlm) = try await requireVLM(common, capability: .imageIn, label: "image")
        let image = try RGBImage.load(contentsOf: URL(fileURLWithPath: imageFile))

        // Native-resolution token count: encode once to size the
        // placeholder run; the stream re-encodes deterministically.
        let count = vlm.encodeImage(image).shape[0]
        let promptTokens = try MultimodalPrompt.build(
            model: m, userText: prompt, mediaToken: vlm.imageTokenId, count: count)
        let params = common.resolvedParameters(for: m)

        print("---")
        let stream = vlm.generateStream(
            promptTokens: promptTokens, image: image,
            maxTokens: params.maxTokens, parameters: params,
            eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)
        let outcome = try await VLMStreamConsumer.consume(
            stream, tokenizer: m.tokenizer, streaming: common.streaming)
        printMediaFooter(common, promptTokens: promptTokens, outcome: outcome)
    }
}

// MARK: - video

/// `ffai generate video <file>` — video + prompt → text.
struct GenerateVideoCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "video",
        abstract: "Describe or answer questions about a video."
    )

    @OptionGroup var common: CommonGenerateOptions

    @Argument(help: "Path to a video file (any AVFoundation-decodable format).")
    var videoFile: String

    @Option(name: .long, help: "Instruction / question about the video.")
    var prompt: String = "Describe this video."

    @Option(name: .long, help: "Maximum number of frames to sample from the video.")
    var maxFrames: Int = 8

    func run() async throws {
        let (m, vlm) = try await requireVLM(common, capability: .videoIn, label: "video")
        guard let videoToken = vlm.videoTokenId else {
            throw ValidationError("\(common.model) was built without a video token id.")
        }
        #if canImport(AVFoundation)
            let frames = try MediaUtils.decodeVideoFrames(
                url: URL(fileURLWithPath: videoFile), maxFrames: maxFrames)
            let count = try vlm.encodeVideo(frames).shape[0]
            let promptTokens = try MultimodalPrompt.build(
                model: m, userText: prompt, mediaToken: videoToken, count: count)
            let params = common.resolvedParameters(for: m)

            print("---")
            let stream = try vlm.generateStream(
                promptTokens: promptTokens, videoFrames: frames,
                maxTokens: params.maxTokens, parameters: params,
                eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)
            let outcome = try await VLMStreamConsumer.consume(
                stream, tokenizer: m.tokenizer, streaming: common.streaming)
            printMediaFooter(common, promptTokens: promptTokens, outcome: outcome)
        #else
            throw ValidationError("Video decoding requires AVFoundation (Apple platforms only).")
        #endif
    }
}

// MARK: - audio

/// `ffai generate audio <file>` — audio + prompt → text (omni VLMs).
struct GenerateAudioCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "audio",
        abstract: "Describe or answer questions about an audio clip (omni models)."
    )

    @OptionGroup var common: CommonGenerateOptions

    @Argument(help: "Path to an audio file (WAV / any AVFoundation-decodable format).")
    var audioFile: String

    @Option(name: .long, help: "Instruction / question about the audio.")
    var prompt: String = "Describe this audio."

    func run() async throws {
        let (m, vlm) = try await requireVLM(common, capability: .audioIn, label: "audio")
        guard let audioToken = vlm.audioTokenId else {
            throw ValidationError("\(common.model) was built without an audio token id.")
        }
        let waveform = try MediaUtils.loadWaveform(
            url: URL(fileURLWithPath: audioFile), targetRate: kAudioInputSampleRate)
        let count = try vlm.encodeAudio(waveform).shape[0]
        let promptTokens = try MultimodalPrompt.build(
            model: m, userText: prompt, mediaToken: audioToken, count: count)
        let params = common.resolvedParameters(for: m)

        print("---")
        let stream = try vlm.generateStream(
            promptTokens: promptTokens, audio: waveform,
            maxTokens: params.maxTokens, parameters: params,
            eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)
        let outcome = try await VLMStreamConsumer.consume(
            stream, tokenizer: m.tokenizer, streaming: common.streaming)
        printMediaFooter(common, promptTokens: promptTokens, outcome: outcome)
    }
}

// MARK: - footer

/// VLM footer: token counts + decode tok/s from the final stats chunk.
private func printMediaFooter(
    _ common: CommonGenerateOptions, promptTokens: [Int], outcome: VLMStreamOutcome
) {
    print("---")
    print("prompt: \(promptTokens.count) tokens")
    if let s = outcome.stats {
        print(
            "generated: \(outcome.tokens.count) tokens "
                + "in \(String(format: "%.2f", s.decodeTimeS))s "
                + "(\(String(format: "%.2f", s.decodeTokensPerSecond)) tok/s)")
        if common.stats { print(s.formatted()) }
    } else {
        print("generated: \(outcome.tokens.count) tokens")
    }
}
