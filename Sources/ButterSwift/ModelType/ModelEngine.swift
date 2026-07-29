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
// ModelEngine + the per-modality engine protocols — the protocol-composition
// layer that lets ONE `Model` wrapper hold ANY modality's concrete engine
// (`engine: any ModelEngine`).
//
// Rationale (see planning/model-registry-and-inheritance-plan.md, "Model-type
// layers = protocols + default impls"): the modality taxonomy is overlapping
// SETS, not a tree — omni = text + audio-in, ocr ≈ vlm — which Swift's single
// inheritance can't model. So each modality is a protocol that REFINES the
// `ModelEngine` marker, and a family conforms to one or several:
//
//   ModelEngine                     (marker — the existential `Model.engine` holds)
//   ├─ TextEngine        → TextGenerating: Module, TextEngine   (LLM / VLM backbone)
//   ├─ Transcribing      → STT / omni-audio-in  (refines StreamingTranscriber)
//   ├─ Speaking      → TTS                  (refines StreamingSynthesizer)
//   ├─ SpeechTransforming→ STS denoise / separate
//   └─ VoiceActivating         → VAD / diarization / endpointing
//
// `ModelEngine` deliberately carries NO structural requirements: the 30+
// heterogeneous audio / VAD concrete types (`@unchecked Sendable` classes,
// most not even `Module`) conform for free with a one-line marker. Per-engine
// metadata (capabilities, dtype, shape) lives on `Model` / the modality
// protocols, not on the base — `Model.availableCapabilities` (sourced from the
// registry `ModelEntry`) is the single source of truth for "what can it do".

import Foundation

/// The base every loadable model engine conforms to — the existential the
/// unified `Model` wrapper stores as `engine`. A pure marker: refined by the
/// modality protocols below. Conformance is free for any concrete engine.
public protocol ModelEngine {}

// MARK: - Text (LLM / VLM)

/// Text-generation engines (LLM backbones + the text backbone of a VLM). The
/// existing `TextGenerating` protocol refines this (T1: kept the name, added
/// `: TextEngine`), so every one of the ~40 `TextGenerating` conformers is a
/// `TextEngine` — and thus a `ModelEngine` — with zero churn. The text
/// forward / cache surface stays on `TextGenerating`; `TextEngine` is the
/// modality marker `Model` dispatches on.
public protocol TextEngine: ModelEngine {}

// MARK: - Transcribe (STT / omni audio-in)

/// Parameters for a transcription request. Maps 1:1 onto the existing
/// `StreamingTranscriber.transcribeStream(_:maxNewTokens:temperature:)` args
/// so the unified path is a pure pass-through (T4).
public struct TranscribeParameters: Sendable {
    public var maxNewTokens: Int
    public var temperature: Float

    public init(maxNewTokens: Int = 512, temperature: Float = 0.0) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
    }
}

/// Speech-to-text / omni-audio-in engines. Refines the existing
/// `StreamingTranscriber` (the uniform streaming contract every STT decoder
/// already conforms to), so a conformance is a one-line marker; the batch
/// `transcribe(...)` default consumes the stream and joins it — bit-identical
/// to concatenating the deltas, per `StreamingTranscriber`'s own contract.
public protocol Transcribing: ModelEngine, StreamingTranscriber {
    /// Buffered transcription — runs the decode to completion and returns the
    /// full transcript. Default consumes `transcribeStream`.
    func transcribe(_ waveform: [Float], parameters: TranscribeParameters)
        async throws -> TranscriptionResult
}

extension Transcribing {
    public func transcribe(_ waveform: [Float], parameters: TranscribeParameters)
        async throws -> TranscriptionResult
    {
        let start = Date()
        var text = ""
        var tokens = 0
        let stream = transcribeStream(
            waveform, maxNewTokens: parameters.maxNewTokens,
            temperature: parameters.temperature)
        for try await delta in stream {
            text += delta
            tokens += 1
        }
        return TranscriptionResult(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
            generatedTokens: tokens,
            totalTimeS: Date().timeIntervalSince(start))
    }
}

// MARK: - Synthesize (TTS)

/// Parameters for a speech-synthesis request. Mirrors the
/// `StreamingSynthesizer.synthesizeStream(text:voice:speed:)` knobs (T4).
public struct SynthesizeParameters: Sendable {
    /// Speaker / voice id for multi-speaker models. `nil` → the model's
    /// default voice.
    public var voice: String?
    /// Speaking-rate multiplier (1.0 = natural rate).
    public var speed: Float

    public init(voice: String? = nil, speed: Float = 1.0) {
        self.voice = voice
        self.speed = speed
    }
}

/// Text-to-speech engines. Refines the existing `StreamingSynthesizer`
/// (segment-streaming waveform `Tensor`s), so a conformance is a one-line
/// marker; the batch `synthesize(...)` default concatenates the streamed
/// segments — equal to a single-shot synthesis of the joined text, per
/// `StreamingSynthesizer`'s contract.
public protocol Speaking: ModelEngine, StreamingSynthesizer {
    /// Buffered synthesis — runs synthesis to completion and returns the full
    /// waveform as flat `Float` samples at `sampleRate`. Default concatenates
    /// `synthesizeStream`.
    func synthesize(_ text: String, parameters: SynthesizeParameters)
        async throws -> [Float]
}

extension Speaking {
    public func synthesize(_ text: String, parameters: SynthesizeParameters)
        async throws -> [Float]
    {
        var samples: [Float] = []
        let stream = synthesizeStream(
            text: text, voice: parameters.voice, speed: parameters.speed)
        for try await chunk in stream {
            samples.append(contentsOf: chunk.toFloatArray())
        }
        return samples
    }
}

// MARK: - Speech transform (STS denoise / separate)

/// Speech-to-speech engines whose contract is uniform waveform → waveform —
/// real-time denoise / enhancement / source-separation (DeepFilterNet,
/// MossFormer2-SE). Promptable / text-conditioned audio editors (SAM-Audio)
/// do NOT fit this contract and are reached via `model.engine as?` instead.
public protocol SpeechTransforming: ModelEngine {
    /// Process an input waveform and return the transformed waveform (same
    /// sample rate as the input unless the model documents otherwise).
    func process(_ waveform: [Float], device: Device) async throws -> [Float]
}

// MARK: - Detect (VAD / diarization / endpointing)

/// One detected audio event — the common shape the three VAD output types
/// (`VADOutput` per-frame speech runs, `DiarizationOutput` per-speaker runs,
/// `VADEndpointOutput` a single turn-end decision) all map down to (T4).
public struct DetectionEvent: Sendable, Equatable {
    /// Event start, in samples from the start of the input waveform.
    public let startSample: Int
    /// Event end, in samples. Equals `startSample` for instantaneous events
    /// (e.g. a turn-end endpoint decision).
    public let endSample: Int
    /// Sample rate the `startSample` / `endSample` indices are expressed in.
    public let sampleRate: Int
    /// Event label — `"speech"` for VAD, `"endpoint"` for turn detection,
    /// `"speaker"` for diarization runs.
    public let label: String
    /// Confidence / probability in `[0, 1]`.
    public let score: Float
    /// Speaker index for diarization; `nil` for single-stream VAD.
    public let speaker: Int?

    public init(
        startSample: Int, endSample: Int, sampleRate: Int,
        label: String, score: Float, speaker: Int? = nil
    ) {
        self.startSample = startSample
        self.endSample = endSample
        self.sampleRate = sampleRate
        self.label = label
        self.score = score
        self.speaker = speaker
    }

    /// Event start in seconds.
    public var startSeconds: Double { Double(startSample) / Double(max(1, sampleRate)) }
    /// Event end in seconds.
    public var endSeconds: Double { Double(endSample) / Double(max(1, sampleRate)) }
}

/// Voice-activity / diarization / endpointing engines — audio in, detected
/// events out (no generative output). The three concrete VAD output types map
/// into `[DetectionEvent]` in each family's conformance.
public protocol VoiceActivating: ModelEngine {
    /// Detect events in an input waveform.
    func detect(_ waveform: [Float], sampleRate: Int) throws -> [DetectionEvent]
}
