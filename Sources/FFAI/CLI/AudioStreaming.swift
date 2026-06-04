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
// AudioStreaming — the uniform streaming contract the CLI (`ffai
// transcribe` / `ffai speak`) drives every audio model through, plus the
// `LoadedAudioModel` accessors that surface a loaded model as a
// transcriber (speech-to-text / omni) or synthesizer (text-to-speech).
//
// Each concrete model conforms in its own file (the conformance is a thin
// wrapper over the model's existing prefill + decode loop, so the buffered
// and streamed paths stay bit-identical). Speech-to-speech / enhancement /
// VAD models (DeepFilterNet, MossFormer2-SE, SAM-Audio) are neither — their
// accessors return nil.

import Foundation

/// Speech-to-text streaming. Yields incremental transcript TEXT (one delta
/// per decoded token for autoregressive decoders; a single yield for
/// single-pass CTC / RNN-T encoders). Concatenating the deltas reproduces
/// the batch transcript (modulo trailing-whitespace trim).
public protocol StreamingTranscriber: Sendable {
    func transcribeStream(
        _ waveform: [Float],
        maxNewTokens: Int,
        temperature: Float
    ) -> AsyncThrowingStream<String, Error>
}

/// Text-to-speech streaming. Yields one waveform `Tensor` per segment
/// (audio chunks instead of tokens) so playback can begin on segment 1
/// while later segments synthesize. Concatenating the chunks equals a
/// single-shot synthesis of the joined text. `sampleRate` is the rate of
/// the yielded samples (for WAV headers / playback).
public protocol StreamingSynthesizer: Sendable {
    var sampleRate: Int { get }
    func synthesizeStream(
        text: String,
        voice: String?,
        speed: Float
    ) -> AsyncThrowingStream<Tensor, Error>
}

extension LoadedAudioModel {
    /// The loaded model as a speech-to-text streamer, or nil if it isn't a
    /// transcription model. Mirrors the `speechToText` / `omniAudio`
    /// buckets in `capabilities(for:)`.
    public var transcriber: (any StreamingTranscriber)? {
        switch self {
        case .graniteSpeech(let m): return m
        default: return nil
        }
    }

    /// The loaded model as a text-to-speech streamer, or nil if it isn't a
    /// synthesis model. Mirrors the `textToSpeech` bucket in
    /// `capabilities(for:)`. Cases are added as each model gains a
    /// `synthesizeStream(text:voice:speed:)` conformance (rollout).
    public var synthesizer: (any StreamingSynthesizer)? {
        switch self {
        default: return nil
        }
    }
}

// MARK: - GraniteSpeech (reference STT conformance)

// GraniteSpeechModel already ships `transcribeStream(_:maxNewTokens:
// temperature:)` with the exact protocol shape — conformance is a marker.
extension GraniteSpeechModel: StreamingTranscriber {}
