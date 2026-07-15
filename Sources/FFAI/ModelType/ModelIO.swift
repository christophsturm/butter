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
// ModelInput / ModelOutput / ModelOutputChunk — the modality-tagged sum types
// for the unified `Model.generate(_:)` / `generateStream(_:)` surface.
//
// Tagged-union enums (NOT open protocols) because FFAI owns the whole curated
// modality set: an exhaustive `switch` forces every call site to handle a new
// modality when one is added (the compile-time safety we want), and it matches
// the codebase's enum idiom. Each case carries a dedicated STRUCT payload so a
// modality's fields evolve independently of the others.
//
// Naming (T6): the plan originally named these `GenerationResult` /
// `GenerationChunk`, but those names are already taken by the text-generation
// structs in `Generation/Generate.swift`. To avoid clobbering the
// deeply-wired text path, the unified sum types are `ModelOutput` (batch) and
// `ModelOutputChunk` (streaming); the existing text structs are reused inside
// the `.text` cases.

import Foundation

// MARK: - Shared payloads

/// A raw audio waveform with its sample rate — the payload for STS / VAD
/// inputs and audio outputs.
public struct AudioWaveform: Sendable {
    public var samples: [Float]
    public var sampleRate: Int

    public init(samples: [Float], sampleRate: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}

/// A text-generation request: chat messages + optional template / sampling
/// overrides. Reuses the existing `ChatMessage` / `ChatTemplateOptions` /
/// `GenerationParameters` types (T3) — no parallel chat vocabulary. `nil`
/// parameters fall back to the model's family-tuned defaults.
public struct TextRequest: Sendable {
    public var messages: [ChatMessage]
    public var parameters: GenerationParameters?
    public var templateOptions: ChatTemplateOptions?

    public init(
        messages: [ChatMessage],
        parameters: GenerationParameters? = nil,
        templateOptions: ChatTemplateOptions? = nil
    ) {
        self.messages = messages
        self.parameters = parameters
        self.templateOptions = templateOptions
    }
}

/// A speech-to-text request: a waveform plus decode parameters.
public struct TranscribeRequest: Sendable {
    public var waveform: [Float]
    public var parameters: TranscribeParameters

    public init(waveform: [Float], parameters: TranscribeParameters = .init()) {
        self.waveform = waveform
        self.parameters = parameters
    }
}

/// A text-to-speech request: text plus synthesis parameters.
public struct SpeakRequest: Sendable {
    public var text: String
    public var parameters: SynthesizeParameters

    public init(text: String, parameters: SynthesizeParameters = .init()) {
        self.text = text
        self.parameters = parameters
    }
}

/// One media item bound positionally to a `.image` / `.video` / `.audio`
/// slot in a `ChatMessage`'s content parts. The i-th media slot across all
/// messages (in message + part order) pairs with `media[i]`.
public enum VisionMedia: Sendable {
    case image(RGBImage)
    case video([RGBImage])
    case audio([Float])
}

/// A vision / omni request routed through the composed `VisionModel`
/// (image / video / audio-in → text). Uniform with `.text`: pass chat
/// `messages` whose content parts carry media slots, plus the `media`
/// bound to those slots in order. `Model` renders the family's chat
/// template, frames each slot's placeholder run, encodes the media, and
/// generates. Output is text (the same `.text` / `.textDelta` the LLM
/// path emits).
///
/// The framing is template-driven where the family's chat template
/// understands image/video/audio content parts (it emits the family's
/// markers + one placeholder, which FFAI expands to the encoded count),
/// with an FFAI-controlled per-family fallback otherwise. See
/// `planning/message-based-vision-input-plan.md`.
///
/// `promptTokens` is a deprecated escape hatch for callers that want to
/// supply a fully framed prompt themselves (bypasses the builder); when
/// non-nil it takes precedence over `messages` / `media`.
public struct VisionRequest: Sendable {
    /// Chat turns, with media slots in their content parts (primary path).
    public var messages: [ChatMessage]
    /// Media bound positionally to the messages' media slots.
    public var media: [VisionMedia]
    /// Chat-template rendering options (mirrors `TextRequest`).
    public var templateOptions: ChatTemplateOptions
    public var maxTokens: Int
    public var parameters: GenerationParameters?
    public var eosTokenId: Int?
    public var eosTokenIds: [Int]

    /// Advanced escape hatch: a fully framed token prompt. Non-nil
    /// bypasses the message-based builder.
    public var promptTokens: [Int]?

    /// Message-based request (primary).
    public init(
        messages: [ChatMessage], media: [VisionMedia] = [],
        templateOptions: ChatTemplateOptions = .init(),
        maxTokens: Int = 512, parameters: GenerationParameters? = nil,
        eosTokenId: Int? = nil, eosTokenIds: [Int] = []
    ) {
        self.messages = messages
        self.media = media
        self.templateOptions = templateOptions
        self.maxTokens = maxTokens
        self.parameters = parameters
        self.eosTokenId = eosTokenId
        self.eosTokenIds = eosTokenIds
        self.promptTokens = nil
    }

    /// Token-based escape hatch: caller supplies the fully framed prompt
    /// and the raw media to encode/splice. Prefer the message-based init.
    @available(
        *, deprecated,
        message:
            "Token-based framing is the advanced escape hatch; prefer the message-based init (messages:media:)."
    )
    public init(
        promptTokens: [Int], media: [VisionMedia] = [],
        maxTokens: Int = 512, parameters: GenerationParameters? = nil,
        eosTokenId: Int? = nil, eosTokenIds: [Int] = []
    ) {
        self.messages = []
        self.media = media
        self.templateOptions = .init()
        self.maxTokens = maxTokens
        self.parameters = parameters
        self.eosTokenId = eosTokenId
        self.eosTokenIds = eosTokenIds
        self.promptTokens = promptTokens
    }
}

// MARK: - Input

/// Modality-tagged input for `Model.generate(_:)` / `generateStream(_:)`.
/// `Model` validates the case against its `kind` / `capabilities` and throws
/// `ModelError.unsupportedInput` on a mismatch.
public enum ModelInput: Sendable {
    /// Text / chat generation (LLM, VLM text path).
    case text(TextRequest)
    /// Vision / omni generation routed through the composed `VisionModel`
    /// (image / video / audio-in → text). Available only when `vlModel != nil`.
    case vision(VisionRequest)
    /// Speech-to-text transcription (STT, omni audio-in).
    case transcribe(TranscribeRequest)
    /// Text-to-speech synthesis (TTS).
    case speak(SpeakRequest)
    /// Speech-to-speech transform — denoise / enhance / separate (STS).
    case transform(AudioWaveform)
    /// Voice-activity / diarization / endpoint detection (VAD).
    case detect(AudioWaveform)
}

// MARK: - Output (batch)

/// Modality-tagged buffered result from `Model.generate(_:)`.
public enum ModelOutput: Sendable {
    /// Text generation — the full `GenerationResult` (tokens + stats + text).
    case text(GenerationResult)
    /// Transcription — the decoded transcript + decode stats.
    case transcript(TranscriptionResult)
    /// Synthesized audio.
    case audio(AudioWaveform)
    /// Detected audio events.
    case detections([DetectionEvent])
}

// MARK: - Output (streaming)

/// Modality-tagged streamed chunk from `Model.generateStream(_:)`. Streaming
/// is available for every kind; the final text chunk carries `stats` (per
/// `GenerationChunk.isFinal`).
public enum ModelOutputChunk: Sendable {
    /// One text-generation delta (the existing streaming chunk).
    case textDelta(GenerationChunk)
    /// One incremental transcript delta.
    case transcriptDelta(String)
    /// One synthesized waveform segment.
    case audioSegment(AudioWaveform)
    /// Detected events (emitted once, terminal, for the non-streaming VAD /
    /// STS kinds surfaced through the streaming API).
    case detections([DetectionEvent])
}
