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
import Tokenizers

/// An audio model whose streaming-transcription path needs the snapshot's
/// tokenizer attached at load time. The `StreamingTranscriber` protocol
/// shape carries no tokenizer argument, and several decoders (GLMASR,
/// Qwen3ASR, QwenOmni, LFMAudio, Whisper, SenseVoice) emit token ids that a
/// tokenizer renders to text — so `AudioModelRegistry.load` attaches the
/// snapshot tokenizer uniformly after construction (the synchronous
/// per-model loaders can't run the async tokenizer load themselves).
public protocol TokenizerAttachable: AnyObject {
    var tokenizer: (any Tokenizer)? { get set }
}

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
        case .whisper(let m): return m
        case .senseVoice(let m): return m
        case .parakeet(let m): return m
        case .qwen3ASR(let m): return m
        case .fireRedASR2(let m): return m
        case .glmASR(let m): return m
        case .cohereTranscribe(let m): return m
        case .voxtralRealtime(let m): return m
        case .qwenOmni(let m): return m
        case .lfmAudio(let m): return m
        default: return nil
        }
    }

    /// The loaded model as a tokenizer-attachable, or nil if its decode
    /// path doesn't need the snapshot tokenizer wired at load. The registry
    /// uses this for a single uniform attach instead of per-branch wiring.
    var tokenizerAttachable: (any TokenizerAttachable)? {
        switch self {
        case .glmASR(let m): return m
        case .qwen3ASR(let m): return m
        case .qwenOmni(let m): return m
        case .lfmAudio(let m): return m
        case .whisper(let m): return m
        case .senseVoice(let m): return m
        default: return nil
        }
    }

    /// The loaded model as a text-to-speech streamer, or nil if it isn't a
    /// synthesis model. Mirrors the `textToSpeech` bucket in
    /// `capabilities(for:)`. Cases are added as each model gains a
    /// `synthesizeStream(text:voice:speed:)` conformance (rollout).
    public var synthesizer: (any StreamingSynthesizer)? {
        switch self {
        case .kokoro(let m): return m
        case .styleTTS2(let m): return m
        case .llamaTTS(let m): return m
        case .marvis(let m): return m
        case .qwen3TTS(let m): return m
        case .qwen3TTSBase(let m): return m
        case .echoTTS(let m): return m
        case .chatterbox(let m): return m
        case .mossTTS(let m): return m
        case .mossTTSNano(let m): return m
        case .pocketTTS(let m): return m
        case .soprano(let m): return m
        case .fishSpeech(let m): return m
        default: return nil
        }
    }
}

// MARK: - GraniteSpeech (reference STT conformance)

// GraniteSpeechModel already ships `transcribeStream(_:maxNewTokens:
// temperature:)` with the exact protocol shape — conformance is a marker.
extension GraniteSpeechModel: StreamingTranscriber {}

// MARK: - TokenizerAttachable conformances

// Each of these exposes a settable `tokenizer` property; the registry wires
// the snapshot tokenizer into it after load so the token-id → text decode in
// `transcribeStream` (and the batch transcribe) can render real text.
extension GLMASRModel: TokenizerAttachable {}
extension Qwen3ASRModel: TokenizerAttachable {}
extension QwenOmniModel: TokenizerAttachable {}
extension LFMAudioModel: TokenizerAttachable {}
extension WhisperModel: TokenizerAttachable {}
extension SenseVoiceModel: TokenizerAttachable {}
