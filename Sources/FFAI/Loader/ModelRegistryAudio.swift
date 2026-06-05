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
// The audio / VAD tier of the unified `ModelRegistry` table — STT, TTS, STS,
// omni-audio, and VAD families as first-class `ModelEntry`s with a
// `loadFromDirectory` closure. This is the data that replaced the separate
// `AudioModelRegistry` / `VADModelRegistry` loaders + their `LoadedAudioModel`
// / `LoadedVADModel` enums: `Model.load` routes every kind through one table.
//
// ENTRY ORDER IS DISPATCH PRECEDENCE. It mirrors the former
// `AudioModelRegistry.resolve` if-chain exactly — the order matters because
// several families share structural markers (FishSpeech's `fish_qwen3_omni`
// contains `qwen3_omni`; Qwen3-ASR's `audio_config` is also matched by
// QwenOmni's structural fallback; StyleTTS2 / Kitten before Kokoro). The
// `ffai models` catalog renders from the kind-grouped, family-sorted
// `catalog`, so this dispatch order is independent of display order.

import Foundation
import Tokenizers

extension ModelRegistry {

    /// Wrap a freshly-constructed audio engine into a `Loaded`. For the
    /// `TokenizerAttachable` STT/omni decoders (whose `transcribeStream`
    /// renders token ids to text) it loads the snapshot tokenizer and
    /// attaches it to the engine — the single uniform attach the former
    /// `AudioModelRegistry.load` did — and surfaces it as `Model.tokenizer`.
    fileprivate static func audioLoaded(
        _ engine: any ModelEngine,
        _ capabilities: Set<Capability>,
        directory: URL,
        attachTokenizer: Bool = false
    ) async -> Loaded {
        var tok: (any Tokenizer)?
        if attachTokenizer {
            // `try?`: a missing / unsupported tokenizer just leaves the
            // streaming path to surface its own error; it must not fail the
            // whole load.
            tok = try? await TokenizerLoader().load(from: directory)
            (engine as? any TokenizerAttachable)?.tokenizer = tok
        }
        return Loaded(
            engine: engine, availableCapabilities: capabilities, tokenizer: tok)
    }

    // ─── Audio / VAD tier ────────────────────────────────────────────
    //
    // Order = dispatch precedence (mirrors `AudioModelRegistry.resolve`).

    static let audioEntries: [ModelEntry] = [
        // ── FishSpeech (first — its `fish_qwen3_omni` contains "qwen3_omni") ──
        ModelEntry(
            family: "FishSpeech", kind: .tts,
            summary: "FishAudio FishSpeech — GQA dual-autoregressive neural-codec TTS.",
            capabilities: Capability.textToSpeech,
            sniff: { FishSpeechModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try FishSpeechModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),

        // ── Speech-to-text / omni (Qwen3TTS before the Qwen ASR/Omni checks) ──
        ModelEntry(
            family: "Qwen3-TTS", kind: .tts,
            summary: "Qwen3-TTS — talker + code predictor + ECAPA speaker + intrinsic codec.",
            repos: ["mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"],
            capabilities: Capability.textToSpeech,
            sniff: { Qwen3TTSModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await Qwen3TTSModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Qwen3-ASR", kind: .stt,
            summary: "Qwen3-ASR — Qwen 3 backbone + audio encoder front-end.",
            repos: ["mlx-community/Qwen3-ASR-0.6B-4bit", "mlx-community/Qwen3-ASR-1.7B-4bit"],
            capabilities: Capability.speechToText,
            sniff: { Qwen3ASRModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try Qwen3ASRModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.speechToText, directory: dir, attachTokenizer: true)
            }),
        ModelEntry(
            family: "Qwen-Omni", kind: .omni,
            summary: "Whisper-style audio encoder + Qwen text backbone (omni-modal).",
            repos: ["Qwen/Qwen2.5-Omni-7B", "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit"],
            capabilities: Capability.omniAudio,
            sniff: { QwenOmniModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try QwenOmniModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.omniAudio, directory: dir, attachTokenizer: true)
            }),
        ModelEntry(
            family: "Marvis (CSM)", kind: .tts,
            summary: "Sesame CSM-shaped dual-transformer TTS — Mimi-code frame generator.",
            repos: ["Marvis-AI/marvis-tts-250m-v0.2-MLX-4bit", "sesame/csm-1b"],
            capabilities: Capability.textToSpeech,
            sniff: { MarvisModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await MarvisModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Orpheus (LlamaTTS)", kind: .tts,
            summary: "Orpheus-style TTS — Llama 3 acoustic backbone + SNAC code decode loop.",
            repos: ["mlx-community/orpheus-3b-0.1-ft-4bit"],
            capabilities: Capability.textToSpeech,
            sniff: { LlamaTTSModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await LlamaTTSModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Whisper STT", kind: .stt,
            summary: "OpenAI Whisper — encoder + cross-attending text decoder.",
            repos: [
                "openai/whisper-tiny", "mlx-community/whisper-tiny-mlx",
                "mlx-community/whisper-base-mlx", "mlx-community/whisper-small-mlx",
                "mlx-community/whisper-medium-mlx", "mlx-community/whisper-large-v3-mlx",
                "mlx-community/whisper-large-v3-turbo-4bit",
            ],
            capabilities: Capability.speechToText,
            sniff: { WhisperModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try WhisperModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.speechToText, directory: dir, attachTokenizer: true)
            }),
        ModelEntry(
            family: "SenseVoice", kind: .stt,
            summary: "Alibaba SenseVoice — SAN-M encoder + CTC head (non-AR STT).",
            repos: ["mlx-community/SenseVoiceSmall"],
            capabilities: Capability.speechToText,
            sniff: { SenseVoiceModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try SenseVoiceModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.speechToText, directory: dir, attachTokenizer: true)
            }),
        // StyleTTS2 / Kitten before Kokoro (Kokoro's structural fallback also
        // matches Kitten's `kitten_tts`).
        ModelEntry(
            family: "StyleTTS2", kind: .tts,
            summary: "StyleTTS2 — style-diffusion acoustic model + iSTFTNet vocoder.",
            capabilities: Capability.textToSpeech,
            sniff: { StyleTTS2Model.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try StyleTTS2Model.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Kokoro TTS", kind: .tts,
            summary: "Kokoro 82M — StyleTTS2 acoustic + iSTFTNet GPU vocoder tail.",
            repos: ["mlx-community/Kokoro-82M-bf16", "hexgrad/Kokoro-82M"],
            capabilities: Capability.textToSpeech,
            sniff: { KokoroModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try KokoroModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Echo TTS", kind: .tts,
            summary: "Echo TTS — compact neural-codec speech synthesizer.",
            capabilities: Capability.textToSpeech,
            sniff: { EchoTTSModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try EchoTTSModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Qwen3-TTS Base", kind: .tts,
            summary: "Qwen3-TTS base talker variant.",
            capabilities: Capability.textToSpeech,
            sniff: { Qwen3TTSBaseModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await Qwen3TTSBaseModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Parakeet", kind: .stt,
            summary: "NVIDIA Parakeet — FastConformer encoder + TDT/CTC decoder.",
            capabilities: Capability.speechToText,
            sniff: { ParakeetModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try ParakeetModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToText, directory: dir)
            }),
        ModelEntry(
            family: "Chatterbox TTS", kind: .tts,
            summary: "Resemble AI Chatterbox — Llama-shaped TTS backbone + S3 tokenizer.",
            repos: ["mlx-community/Chatterbox-TTS-4bit", "mlx-community/Chatterbox-Turbo-TTS-4bit"],
            capabilities: Capability.textToSpeech,
            sniff: { ChatterboxModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try ChatterboxModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "FireRedASR2", kind: .stt,
            summary: "FireRedTeam FireRedASR2-AED — conformer encoder + Llama decoder.",
            repos: ["mlx-community/FireRedASR2-AED-mlx"],
            capabilities: Capability.speechToText,
            sniff: { FireRedASR2Model.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try FireRedASR2Model.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToText, directory: dir)
            }),
        // Moss Nano before Moss-8B (Nano's distinctive `gpt2_config` block).
        ModelEntry(
            family: "Moss TTS Nano", kind: .tts,
            summary: "Moss TTS Nano — compact on-device variant.",
            capabilities: Capability.textToSpeech,
            sniff: { MossTTSNanoModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try MossTTSNanoModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Moss TTS", kind: .tts,
            summary: "Moss TTS — autoregressive neural-codec speech synthesizer.",
            capabilities: Capability.textToSpeech,
            sniff: { MossTTSModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try MossTTSModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "LFM Audio", kind: .omni,
            summary: "LiquidAI LFM Audio — omni-modal audio-in path.",
            capabilities: Capability.omniAudio,
            sniff: { LFMAudioModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try LFMAudioModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.omniAudio, directory: dir, attachTokenizer: true)
            }),
        ModelEntry(
            family: "Voxtral", kind: .stt,
            summary: "Mistral Voxtral — realtime STT / TTS (streaming).",
            repos: [
                "mlx-community/Voxtral-Mini-3B-2507-bf16",
                "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            ],
            capabilities: Capability.speechToText,
            sniff: { VoxtralRealtimeModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try VoxtralRealtimeModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToText, directory: dir)
            }),
        ModelEntry(
            family: "GLM-ASR", kind: .stt,
            summary: "Zhipu GLM-ASR — conformer encoder + GLM text decoder.",
            capabilities: Capability.speechToText,
            sniff: { GLMASRModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try GLMASRModel.load(directory: dir, device: dev)
                return await audioLoaded(
                    m, Capability.speechToText, directory: dir, attachTokenizer: true)
            }),
        ModelEntry(
            family: "Pocket TTS", kind: .tts,
            summary: "Pocket TTS — compact StyleTTS2-style on-device synth.",
            repos: ["mlx-community/pocket-tts-4bit"],
            capabilities: Capability.textToSpeech,
            sniff: { PocketTTSModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try PocketTTSModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Soprano TTS", kind: .tts,
            summary: "Soprano 80M — compact StyleTTS2-flavored on-device synth.",
            repos: ["mlx-community/Soprano-80M-bf16", "ekryski/Soprano-1.1-80M-4bit"],
            capabilities: Capability.textToSpeech,
            sniff: { SopranoModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await SopranoModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.textToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Granite Speech", kind: .stt,
            summary: "IBM Granite Speech — conformer encoder + QFormer projector + Granite LLM.",
            capabilities: Capability.speechToText,
            sniff: { GraniteSpeech.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await GraniteSpeech.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToText, directory: dir)
            }),

        // ── Speech-to-speech (enhancement / separation / segmentation) ──
        ModelEntry(
            family: "DeepFilterNet", kind: .sts,
            summary: "DeepFilterNet — real-time speech enhancement / denoise.",
            capabilities: Capability.speechToSpeech,
            sniff: { DeepFilterNetModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try DeepFilterNetModel.load(from: dir, device: dev)
                return await audioLoaded(m, Capability.speechToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "Cohere Transcribe", kind: .stt,
            summary: "Cohere transcription model — audio encoder + Command text decoder.",
            capabilities: Capability.speechToText,
            sniff: { CohereTranscribeModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try await CohereTranscribeModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToText, directory: dir)
            }),
        ModelEntry(
            family: "MossFormer2-SE", kind: .sts,
            summary: "MossFormer2 — speech enhancement / source separation.",
            capabilities: Capability.speechToSpeech,
            sniff: { MossFormer2SEModel.handles($0) },
            loadFromDirectory: { dir, _, _, dev in
                let m = try MossFormer2SEModel.load(directory: dir, device: dev)
                return await audioLoaded(m, Capability.speechToSpeech, directory: dir)
            }),
        ModelEntry(
            family: "SAM Audio", kind: .sts,
            summary: "SAM Audio — promptable audio segmentation.",
            capabilities: Capability.speechToSpeech,
            sniff: { SAMAudio.handles($0) },
            loadFromDirectory: { dir, config, _, dev in
                let samConfig = SAMAudioModel.loadConfig(from: dir)
                let bundle = try SafeTensorsBundle(directory: dir, device: dev)
                let variant = try SAMAudio.variant(for: config)
                let m = try variant.loadModel(
                    directory: dir, config: samConfig, weights: bundle, device: dev)
                return await audioLoaded(m, Capability.speechToSpeech, directory: dir)
            }),

        // ── Voice activity / diarization (matched by model_type) ──
        ModelEntry(
            family: "Silero VAD", kind: .vad,
            summary: "Streaming VAD — STFT + small gated-conv encoder.",
            repos: ["snakers4/silero-vad"],
            capabilities: [],
            modelTypes: ["silero_vad"],
            loadFromDirectory: { dir, _, _, dev in
                let m = try SileroVADModel.loadFromDirectory(dir, device: dev)
                return await audioLoaded(m, [], directory: dir)
            }),
        ModelEntry(
            family: "SmartTurn VAD", kind: .vad,
            summary: "Conversational endpoint / turn detection (Pipecat).",
            repos: ["pipecat-ai/smart-turn-v3"],
            capabilities: [],
            modelTypes: ["smart_turn", "smart_turn_v3", "smart-turn"],
            loadFromDirectory: { dir, _, _, dev in
                let m = try SmartTurnModel.loadFromDirectory(dir, device: dev)
                return await audioLoaded(m, [], directory: dir)
            }),
        ModelEntry(
            family: "Sortformer", kind: .vad,
            summary: "NVIDIA Sortformer — multi-speaker diarization.",
            capabilities: [],
            modelTypes: ["sortformer", "diar_sortformer"],
            loadFromDirectory: { dir, _, _, dev in
                let m = try SortformerModel.loadFromDirectory(dir, device: dev)
                return await audioLoaded(m, [], directory: dir)
            }),
        ModelEntry(
            family: "TenVAD", kind: .vad,
            summary: "TEN-framework lightweight VAD (native macOS framework).",
            capabilities: [],
            modelTypes: ["ten_vad", "ten-vad", "tenvad"],
            loadFromDirectory: { dir, _, _, dev in
                let m = try TenVADModel.loadFromDirectory(dir, device: dev)
                return await audioLoaded(m, [], directory: dir)
            }),
        ModelEntry(
            family: "FireRedVAD", kind: .vad,
            summary: "FireRedTeam DFSMN-based VAD.",
            capabilities: [],
            modelTypes: ["firered_vad", "firered-vad", "fireredvad"],
            loadFromDirectory: { dir, _, _, dev in
                let m = try FireRedVADModel.loadFromDirectory(dir, device: dev)
                return await audioLoaded(m, [], directory: dir)
            }),
    ]
}
