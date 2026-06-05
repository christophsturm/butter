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
// Modality-protocol conformances for the audio / VAD families — the bridge
// that lets every concrete audio engine slot into the unified `Model.engine`
// (`any ModelEngine`) and be dispatched by `Model.generate(_:)`.
//
// STT / TTS conformances are pure markers: those types already conform to the
// streaming protocols (`StreamingTranscriber` / `StreamingSynthesizer`) that
// `Transcribing` / `Synthesizing` refine, and the batch method comes from the
// protocol-extension default. STS / VAD conformances carry a thin body that
// adapts the family's existing entry point (`enhance` / `detect` /
// `predictEndpoint`) onto the uniform protocol shape.

import Foundation

// MARK: - Transcribing (STT / omni audio-in)

// Each already conforms to `StreamingTranscriber`; `Transcribing` adds only
// the defaulted batch `transcribe(_:parameters:)`. (Same set as
// `LoadedAudioModel.transcriber`.)
extension GraniteSpeechModel: Transcribing {}
extension WhisperModel: Transcribing {}
extension SenseVoiceModel: Transcribing {}
extension ParakeetModel: Transcribing {}
extension Qwen3ASRModel: Transcribing {}
extension FireRedASR2Model: Transcribing {}
extension GLMASRModel: Transcribing {}
extension CohereTranscribeModel: Transcribing {}
extension VoxtralRealtimeModel: Transcribing {}
extension QwenOmniModel: Transcribing {}
extension LFMAudioModel: Transcribing {}

// MARK: - Synthesizing (TTS)

// Each already conforms to `StreamingSynthesizer`; `Synthesizing` adds only
// the defaulted batch `synthesize(_:parameters:)`. (Same set as
// `LoadedAudioModel.synthesizer`.)
extension KokoroModel: Synthesizing {}
extension StyleTTS2Model: Synthesizing {}
extension LlamaTTSModel: Synthesizing {}
extension MarvisModel: Synthesizing {}
extension Qwen3TTSModel: Synthesizing {}
extension Qwen3TTSBaseModel: Synthesizing {}
extension EchoTTSModel: Synthesizing {}
extension ChatterboxModel: Synthesizing {}
extension MossTTSModel: Synthesizing {}
extension MossTTSNanoModel: Synthesizing {}
extension PocketTTSModel: Synthesizing {}
extension SopranoModel: Synthesizing {}
extension FishSpeechModel: Synthesizing {}

// MARK: - SpeechTransforming (STS denoise / separate)

// Waveform → waveform denoisers. SAM-Audio's promptable `segment(...)` does
// not fit the uniform contract and is reached via `model.engine as?` instead.
extension DeepFilterNetModel: SpeechTransforming {
    public func process(_ waveform: [Float], device: Device) async throws -> [Float] {
        try enhance(waveform: waveform)
    }
}
extension MossFormer2SEModel: SpeechTransforming {
    public func process(_ waveform: [Float], device: Device) async throws -> [Float] {
        try enhance(waveform: waveform)
    }
}
// SAM-Audio still flows through the unified loader (kind `.sts`,
// `speechToSpeech` capabilities) — it just isn't a `SpeechTransforming`.
extension SAMAudioModel: ModelEngine {}

// MARK: - Detecting (VAD / diarization / endpointing)

// The three concrete VAD output types map down to `[DetectionEvent]`.

extension VADOutput {
    /// Each post-processed speech run becomes a `"speech"` detection event.
    var detectionEvents: [DetectionEvent] {
        segments.map {
            DetectionEvent(
                startSample: $0.startSample, endSample: $0.endSample,
                sampleRate: $0.sampleRate, label: "speech", score: 1.0)
        }
    }
}

extension DiarizationOutput {
    /// Each speaker run becomes a `"speaker"` event tagged with its speaker
    /// index; sample indices are derived from the segment's seconds.
    var detectionEvents: [DetectionEvent] {
        segments.map {
            DetectionEvent(
                startSample: Int($0.startSeconds * Double(sampleRate)),
                endSample: Int($0.endSeconds * Double(sampleRate)),
                sampleRate: sampleRate, label: "speaker", score: 1.0,
                speaker: $0.speaker)
        }
    }
}

extension VADEndpointOutput {
    /// A single `"endpoint"` event when the turn-end threshold was crossed;
    /// empty otherwise. Instantaneous, so `startSample == endSample == 0`.
    func detectionEvents(sampleRate: Int) -> [DetectionEvent] {
        guard prediction == 1 else { return [] }
        return [
            DetectionEvent(
                startSample: 0, endSample: 0, sampleRate: sampleRate,
                label: "endpoint", score: probability)
        ]
    }
}

extension SileroVADModel: Detecting {
    public func detect(_ waveform: [Float], sampleRate: Int) throws -> [DetectionEvent] {
        try detect(audio: waveform, sampleRate: sampleRate).detectionEvents
    }
}
extension TenVADModel: Detecting {
    public func detect(_ waveform: [Float], sampleRate: Int) throws -> [DetectionEvent] {
        try detect(audio: waveform, sampleRate: sampleRate).detectionEvents
    }
}
extension FireRedVADModel: Detecting {
    public func detect(_ waveform: [Float], sampleRate: Int) throws -> [DetectionEvent] {
        try detect(audio: waveform, sampleRate: sampleRate).detectionEvents
    }
}
extension SortformerModel: Detecting {
    public func detect(_ waveform: [Float], sampleRate: Int) throws -> [DetectionEvent] {
        detect(audio: waveform, sampleRate: sampleRate).detectionEvents
    }
}
extension SmartTurnModel: Detecting {
    public func detect(_ waveform: [Float], sampleRate: Int) throws -> [DetectionEvent] {
        predictEndpoint(audio: waveform).detectionEvents(sampleRate: sampleRate)
    }
}
