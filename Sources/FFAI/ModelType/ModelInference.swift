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
// The unified inference surface on `Model` — the generic `generate(_:)` /
// `generateStream(_:)` sum-type API plus the typed convenience wrappers
// (`transcribe` / `synthesize` / `detect`) the CLI drives.
//
// `generate` dispatches on `engine as? <ModalityProtocol>` and throws
// `ModelError.unsupportedInput` when the loaded model's kind can't service the
// request. The text path delegates to the existing `generate(messages:)` /
// `generateStream(messages:)` (so the rich `GenerationResult` / streaming
// behaviour is unchanged); audio paths route through the modality protocols
// (`Transcribing` / `Speaking` / `SpeechTransforming` / `VoiceActivating`).

import Foundation
import Metal

extension Model {

    // MARK: - Generic (sum-type) surface

    /// Run inference for `input`, returning a modality-tagged batch result.
    /// Throws `ModelError.unsupportedInput` if this model's kind can't service
    /// the request (e.g. `.speak` to an LLM).
    public func generate(_ input: ModelInput) async throws -> ModelOutput {
        switch input {
        case .text(let req):
            let result = try await generate(
                messages: req.messages,
                templateOptions: req.templateOptions ?? .init(),
                parameters: req.parameters)
            return .text(result)

        case .vision(let req):
            // Drain the VLM stream into a GenerationResult — the same
            // collect-the-stream batch path the text case uses.
            let result = try await collectStream(
                try visionStream(req), promptTokens: req.promptTokens)
            return .text(result)

        case .transcribe(let req):
            let engine = try transcribingEngine(for: input)
            let transcript = try await engine.transcribe(
                req.waveform, parameters: req.parameters)
            return .transcript(transcript)

        case .speak(let req):
            let engine = try synthesizingEngine(for: input)
            let samples = try await engine.synthesize(
                req.text, parameters: req.parameters)
            return .audio(AudioWaveform(samples: samples, sampleRate: engine.sampleRate))

        case .transform(let waveform):
            let engine = try transformingEngine(for: input)
            let out = try await engine.process(waveform.samples, device: .shared)
            return .audio(AudioWaveform(samples: out, sampleRate: waveform.sampleRate))

        case .detect(let waveform):
            let engine = try detectingEngine(for: input)
            let events = try engine.detect(waveform.samples, sampleRate: waveform.sampleRate)
            return .detections(events)
        }
    }

    /// Streaming counterpart of `generate(_:)`. Every kind streams: text
    /// yields one `.textDelta` per token (final carries stats), STT yields
    /// `.transcriptDelta`s, TTS yields `.audioSegment`s, and the
    /// single-shot kinds (STS / VAD) yield one terminal `.detections` /
    /// `.audioSegment`. Errors — including an `unsupportedInput` kind
    /// mismatch — surface through the stream.
    public func generateStream(_ input: ModelInput)
        -> AsyncThrowingStream<ModelOutputChunk, Error>
    {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    switch input {
                    case .text(let req):
                        let stream = try generateStream(
                            messages: req.messages,
                            templateOptions: req.templateOptions ?? .init(),
                            parameters: req.parameters)
                        for try await chunk in stream {
                            continuation.yield(.textDelta(chunk))
                        }

                    case .vision(let req):
                        let stream = try visionStream(req)
                        for try await chunk in stream {
                            continuation.yield(.textDelta(chunk))
                        }

                    case .transcribe(let req):
                        let engine = try transcribingEngine(for: input)
                        let stream = engine.transcribeStream(
                            req.waveform,
                            maxNewTokens: req.parameters.maxNewTokens,
                            temperature: req.parameters.temperature)
                        for try await delta in stream {
                            continuation.yield(.transcriptDelta(delta))
                        }

                    case .speak(let req):
                        let engine = try synthesizingEngine(for: input)
                        let rate = engine.sampleRate
                        let stream = engine.synthesizeStream(
                            text: req.text, voice: req.parameters.voice,
                            speed: req.parameters.speed)
                        for try await chunk in stream {
                            continuation.yield(
                                .audioSegment(
                                    AudioWaveform(
                                        samples: chunk.toFloatArray(), sampleRate: rate)))
                        }

                    case .transform(let waveform):
                        let engine = try transformingEngine(for: input)
                        let out = try await engine.process(waveform.samples, device: .shared)
                        continuation.yield(
                            .audioSegment(
                                AudioWaveform(samples: out, sampleRate: waveform.sampleRate)))

                    case .detect(let waveform):
                        let engine = try detectingEngine(for: input)
                        let events = try engine.detect(
                            waveform.samples, sampleRate: waveform.sampleRate)
                        continuation.yield(.detections(events))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Typed convenience

    /// Transcribe a waveform to text (STT / omni). Throws
    /// `ModelError.unsupportedInput` if this isn't a transcription model.
    public func transcribe(
        _ waveform: [Float], parameters: TranscribeParameters = .init()
    ) async throws -> TranscriptionResult {
        try await transcribingEngine(for: .transcribe(.init(waveform: waveform)))
            .transcribe(waveform, parameters: parameters)
    }

    /// Streaming transcription — yields incremental transcript text. Drives
    /// `ffai transcribe`. A kind mismatch surfaces through the stream.
    public func transcribeStream(
        _ waveform: [Float], maxNewTokens: Int = 512, temperature: Float = 0.0
    ) -> AsyncThrowingStream<String, Error> {
        guard let engine = engine as? any Transcribing else {
            return errorStream(unsupportedInput("transcribe"))
        }
        return engine.transcribeStream(
            waveform, maxNewTokens: maxNewTokens, temperature: temperature)
    }

    /// Synthesize speech for `text` (TTS), returning the full waveform.
    public func synthesize(
        _ text: String, parameters: SynthesizeParameters = .init()
    ) async throws -> [Float] {
        try await synthesizingEngine(for: .speak(.init(text: text)))
            .synthesize(text, parameters: parameters)
    }

    /// Streaming synthesis — yields one waveform `Tensor` per segment. Drives
    /// `ffai speak`. A kind mismatch surfaces through the stream.
    public func synthesizeStream(
        text: String, voice: String? = nil, speed: Float = 1.0
    ) -> AsyncThrowingStream<Tensor, Error> {
        guard let engine = engine as? any Speaking else {
            return errorStream(unsupportedInput("speak"))
        }
        return engine.synthesizeStream(text: text, voice: voice, speed: speed)
    }

    /// The synthesis sample rate, or `nil` if this isn't a synthesis model.
    /// `ffai speak` reads this for the WAV header.
    public var audioSampleRate: Int? { (engine as? any Speaking)?.sampleRate }

    /// Detect audio events (VAD / diarization / endpointing).
    public func detect(
        _ waveform: [Float], sampleRate: Int = 16000
    ) async throws -> [DetectionEvent] {
        try detectingEngine(for: .detect(.init(samples: waveform, sampleRate: sampleRate)))
            .detect(waveform, sampleRate: sampleRate)
    }

    // MARK: - Engine resolution helpers

    private func transcribingEngine(for input: ModelInput) throws -> any Transcribing {
        guard let e = engine as? any Transcribing else {
            throw unsupportedInput("transcribe")
        }
        return e
    }
    private func synthesizingEngine(for input: ModelInput) throws -> any Speaking {
        guard let e = engine as? any Speaking else {
            throw unsupportedInput("speak")
        }
        return e
    }
    private func transformingEngine(for input: ModelInput) throws -> any SpeechTransforming {
        guard let e = engine as? any SpeechTransforming else {
            throw unsupportedInput("transform")
        }
        return e
    }
    private func detectingEngine(for input: ModelInput) throws -> any VoiceActivating {
        guard let e = engine as? any VoiceActivating else {
            throw unsupportedInput("detect")
        }
        return e
    }

    /// Resolve a `.vision` request to the matching composed-`VisionModel`
    /// token stream. Routes by the one present media modality (videoFrames →
    /// audio → image; all empty/nil runs the backbone text-only). Throws
    /// `unsupportedInput` when the model isn't a VLM / omni (`vlModel == nil`).
    private func visionStream(_ req: VisionRequest) throws
        -> AsyncThrowingStream<GenerationChunk, Error>
    {
        guard let vlm = vlModel else { throw unsupportedInput("vision") }
        if !req.videoFrames.isEmpty {
            return try vlm.generateStream(
                promptTokens: req.promptTokens, videoFrames: req.videoFrames,
                maxTokens: req.maxTokens, parameters: req.parameters,
                eosTokenId: req.eosTokenId, eosTokenIds: req.eosTokenIds)
        }
        if let audio = req.audio {
            return try vlm.generateStream(
                promptTokens: req.promptTokens, audio: audio,
                maxTokens: req.maxTokens, parameters: req.parameters,
                eosTokenId: req.eosTokenId, eosTokenIds: req.eosTokenIds)
        }
        return vlm.generateStream(
            promptTokens: req.promptTokens, image: req.image,
            maxTokens: req.maxTokens, parameters: req.parameters,
            eosTokenId: req.eosTokenId, eosTokenIds: req.eosTokenIds)
    }

    private func unsupportedInput(_ what: String) -> ModelError {
        ModelError.unsupportedInput(kind: kind, input: what)
    }

    private func errorStream<T>(_ error: Error) -> AsyncThrowingStream<T, Error> {
        AsyncThrowingStream { $0.finish(throwing: error) }
    }
}
