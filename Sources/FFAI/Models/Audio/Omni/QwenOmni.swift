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
// QwenOmni — Qwen's omni-modal family (Qwen2.5-Omni / Qwen3-Omni):
// text + vision + audio in, text (and optionally audio) out. The model
// is a Qwen text backbone fronted by per-modality encoders whose
// outputs are spliced into the text token stream.
//
//   audio   ──audio encoder──▶ audio tokens ──┐
//   image   ──vision encoder──▶ image tokens ─┼─▶ text backbone ──▶ text
//   text    ──token embed──────▶ text tokens ─┘
//
// ## Scope
//
// This file wires the **audio-in path**: a Whisper-style `AudioEncoder`
// turns a waveform into audio feature tokens in the backbone's hidden
// dim. The vision path is infrastructure (`VisionEncoder`)
// and is scaffolded but not spliced here — Qwen-VL's dynamic-resolution
// tower is a separate port. The text backbone is the existing Qwen3
// engine; QwenOmni composes it rather than reimplementing it.
//
// `QwenOmni.encodeAudio` is the supported entry point: it produces the
// `[nAudioTokens, textHidden]` features a caller splices into a Qwen3
// prompt's embedding stream.
//
// ## TODO — follow-up: video inference
//
// Qwen-Omni supports video input upstream (it composes Qwen 2.5-VL's
// dynamic-resolution vision tower with the omni text backbone). FFAI's
// QwenOmni currently exposes ONLY the audio path — there is no vision
// tower wired here at all, no `VisionModel` adapter, and no video token id
// surfaced. Adding video requires first wiring vision (the Qwen-VL
// tower would compose cleanly: load `vision_tower.*` via
// `Qwen25VLVisionModel.load` and wrap the QwenOmniModel in a `VisionModel`
// alongside the audio encoder). Once vision is wired, video falls out
// of `Qwen25VLVisionModel.encode(frames:)` for free since the omni
// variant shares the 2.5-VL tower. Tracked under task #150.

import Foundation
import Metal
import Tokenizers

// ─── Configuration ───────────────────────────────────────────────────

/// QwenOmni audio-tower hyper-parameters, decoded from the checkpoint's
/// nested `audio_config` (a Whisper-style encoder config) plus the
/// backbone text hidden dim.
public struct QwenOmniAudioConfig: Sendable {
    /// Mel bins the audio front-end produces.
    public let nMels: Int
    /// Audio-encoder hidden dim (`d_model` of the audio tower).
    public let encoderHidden: Int
    /// Audio-encoder feed-forward intermediate dim.
    public let encoderIntermediate: Int
    /// Audio-encoder transformer blocks.
    public let encoderLayers: Int
    /// Audio-encoder attention heads.
    public let encoderHeads: Int
    /// Maximum audio-context length.
    public let maxAudioCtx: Int
    /// The text backbone's hidden dim — the audio features are
    /// projected into this so they can be spliced into the text stream.
    public let textHidden: Int

    public init(
        nMels: Int, encoderHidden: Int, encoderIntermediate: Int,
        encoderLayers: Int, encoderHeads: Int, maxAudioCtx: Int,
        textHidden: Int
    ) {
        self.nMels = nMels
        self.encoderHidden = encoderHidden
        self.encoderIntermediate = encoderIntermediate
        self.encoderLayers = encoderLayers
        self.encoderHeads = encoderHeads
        self.maxAudioCtx = maxAudioCtx
        self.textHidden = textHidden
    }

    /// The `audio_config` block in a Qwen-Omni `config.json`. Qwen2.5-
    /// Omni nests it under `thinker_config.audio_config`; some Qwen3-
    /// Omni / mlx conversions hoist it to the top level. Returns the
    /// `(audio_config, text_config)` pair, looking in both places.
    static func locateConfigs(_ config: ModelConfig)
        -> (audio: [String: Any], text: [String: Any]?)?
    {
        // Top-level (hoisted) layout.
        if let audio = config.nested("audio_config") {
            return (audio, config.nested("text_config"))
        }
        // Qwen2.5-Omni nested-under-thinker layout.
        if let thinker = config.nested("thinker_config"),
            let audio = thinker["audio_config"] as? [String: Any]
        {
            return (audio, thinker["text_config"] as? [String: Any])
        }
        return nil
    }

    /// Build from a decoded `config.json`. Qwen-Omni nests the audio
    /// tower under `audio_config` (a Whisper-style block) and the text
    /// backbone under `text_config`, both inside `thinker_config` for
    /// Qwen2.5-Omni.
    public static func from(_ config: ModelConfig) -> QwenOmniAudioConfig? {
        guard let located = locateConfigs(config) else { return nil }
        let audio = located.audio
        func ai(_ k: String) -> Int? {
            if let v = audio[k] as? Int { return v }
            if let v = audio[k] as? Double { return Int(v) }
            return nil
        }
        guard let encHidden = ai("d_model") else { return nil }
        // Whisper-style configs name the layer count `encoder_layers`;
        // Qwen-Omni's audio block also carries `num_hidden_layers`.
        guard let encLayers = ai("encoder_layers") ?? ai("num_hidden_layers"),
            let encHeads = ai("encoder_attention_heads")
        else { return nil }
        let nMels = ai("num_mel_bins") ?? 128
        let encInter = ai("encoder_ffn_dim") ?? (4 * encHidden)
        let maxAud = ai("max_source_positions") ?? 1500
        // The text hidden dim — the audio features project into it.
        // `output_dim` on the audio block is the projection target;
        // fall back to the text config's hidden_size.
        let textHidden =
            ai("output_dim")
            ?? (located.text?["hidden_size"] as? Int)
            ?? config.hiddenSize
            ?? encHidden
        return QwenOmniAudioConfig(
            nMels: nMels, encoderHidden: encHidden,
            encoderIntermediate: encInter, encoderLayers: encLayers,
            encoderHeads: encHeads, maxAudioCtx: maxAud,
            textHidden: textHidden)
    }

    /// The front-end config the audio tower expects (16 kHz log-Mel).
    public var frontEnd: AudioFrontEndConfig {
        AudioFrontEndConfig(
            sampleRate: 16_000, nFFT: 400, hopLength: 160,
            nMels: nMels)
    }
}

// ─── QwenOmni model ──────────────────────────────────────────────────

public enum QwenOmniError: Error, CustomStringConvertible {
    /// The text backbone is not loaded — `QwenOmni` was constructed
    /// audio-only. `encodeAudio` works regardless.
    case textBackboneUnavailable
    /// `transcribe` / `transcribeStream` was called before a tokenizer
    /// was attached. The synchronous `load(directory:)` path can't run the
    /// async tokenizer load, so the registry's async loader (or a caller)
    /// must set `tokenizer` before driving the audio→text path.
    case tokenizerUnavailable

    public var description: String {
        switch self {
        case .textBackboneUnavailable:
            return "QwenOmni: the text backbone is not loaded in this "
                + "audio-only build; use encodeAudio to obtain audio "
                + "feature tokens and run them through a Qwen3 model"
        case .tokenizerUnavailable:
            return "QwenOmni: no tokenizer attached — set `tokenizer` "
                + "(the async registry loader does this) before calling "
                + "transcribe / transcribeStream"
        }
    }
}

/// A loaded QwenOmni model. Owns the audio encoder (always available —
/// the FFAI contribution) and, when wired, the text backbone +
/// vision tower.
public final class QwenOmniModel: @unchecked Sendable {
    public let config: QwenOmniAudioConfig
    /// The Whisper-style audio encoder.
    public let audioEncoder: AudioEncoder
    /// Projection from the audio-encoder hidden dim into the text
    /// backbone hidden dim. `nil` when they already match.
    public let audioProjection: Linear?
    /// The text backbone — `nil` in an audio-only build.
    public let textBackbone: Qwen3Model?
    let dtype: DType

    /// Token id that marks an audio-pad slot in the prompt — the position
    /// whose embedding is replaced by an encoded audio feature row. Matches
    /// the Qwen-Omni / Qwen3-ASR `<|audio_pad|>` id.
    public let audioTokenId: Int
    /// EOS token ids — autoregressive decode stops when any is produced.
    /// Qwen-Omni emits either the generic `<|endoftext|>` (151643) or the
    /// chat-turn `<|im_end|>` (151645).
    public let eosTokenIds: [Int]

    /// Tokenizer used by the `transcribe` / `transcribeStream` audio→text
    /// path. Optional because the synchronous `load(directory:)` can't run
    /// the async tokenizer load; the async registry loader (or a caller
    /// that already holds a tokenizer) sets it before transcribing. Marked
    /// mutable; the class-level `@unchecked Sendable` already covers it.
    public var tokenizer: (any Tokenizer)?

    public init(
        config: QwenOmniAudioConfig, audioEncoder: AudioEncoder,
        audioProjection: Linear?, textBackbone: Qwen3Model?,
        dtype: DType,
        audioTokenId: Int = 151_676,
        eosTokenIds: [Int] = [151_643, 151_645],
        tokenizer: (any Tokenizer)? = nil
    ) {
        self.config = config
        self.audioEncoder = audioEncoder
        self.audioProjection = audioProjection
        self.textBackbone = textBackbone
        self.dtype = dtype
        self.audioTokenId = audioTokenId
        self.eosTokenIds = eosTokenIds
        self.tokenizer = tokenizer
    }

    /// Encode a waveform into audio feature tokens in the text backbone
    /// hidden dim. Returns `[nAudioTokens, textHidden]` — the tokens a
    /// caller splices into a Qwen3 prompt's embedding stream.
    public func encodeAudio(waveform: [Float], device: Device = .shared)
        -> Tensor
    {
        // The mel_spectrogram kernel only emits f32 / f16; a bf16 model
        // gets the front-end run in f32, then the spectrogram cast to
        // the model's activation dtype before the conv stem.
        let melDtype: DType = dtype == .f16 ? .f16 : .f32
        let cmd = device.makeCommandBuffer()
        // `whisperNormalize: true` — `logMelSpectrogram` commits + waits
        // on `cmd` itself (it normalises the kernel result on the CPU),
        // so the result is already CPU-synced; do NOT re-commit `cmd`.
        let melRaw = MediaUtils.logMelSpectrogram(
            waveform: waveform, cfg: config.frontEnd, dtype: melDtype,
            whisperNormalize: true, device: device, on: cmd)
        let mel = MediaUtils.castTensor(
            melRaw, to: dtype,
            device: device)

        // Run the Whisper-style encoder over the log-Mel.
        var features = audioEncoder.encode(
            mel: mel, melFrameMajor: true,
            device: device)

        // Project into the text hidden dim if the dims differ.
        guard let proj = audioProjection else { return features }
        let nTokens = features.shape[0]
        let cmd2 = device.makeCommandBuffer()
        var projected = Ops.gemm(
            weight: proj.weight, input: features,
            nRows: nTokens, on: cmd2)
        if let bias = proj.bias {
            projected = AudioEncoder.addRowBias(
                projected, bias: bias, nRows: nTokens,
                rowSize: config.textHidden, on: cmd2)
        }
        cmd2.commit()
        cmd2.waitUntilCompleted()
        features = projected
        return features
    }

    /// Whether this build has a text backbone wired.
    public var hasTextBackbone: Bool { textBackbone != nil }
}

// ─── Audio → text transcription ──────────────────────────────────────
//
// The audio→text path encodes the waveform into feature tokens, splices
// them into a Qwen-Omni chat prompt's embedding stream (replacing the
// `<|audio_pad|>` placeholders), prefills the Qwen3 backbone, then
// greedily / temperature-samples tokens until EOS. The batch
// `transcribe` and streaming `transcribeStream` share one
// `prepareTranscribe` + an identical per-token decode loop, so the
// concatenated stream deltas reproduce the batch transcript exactly
// (modulo the batch's trailing-whitespace trim).

extension QwenOmniModel {
    /// State carried from prefill into the decode loop, shared by the
    /// batch and streaming transcription paths.
    private struct TranscribeState {
        let backbone: Qwen3Model
        let tokenizer: any Tokenizer
        let caches: [any LayerCacheProtocol]
        let firstToken: Int
        let promptLen: Int
    }

    /// Sample one token from a `[vocab]` logits tensor — argmax at
    /// `temperature == 0`, otherwise temperature sampling.
    private func sampleTok(_ logits: Tensor, temperature: Float) -> Int {
        if temperature == 0 { return Sampling.argmax(logits) }
        var rng = SystemRandomNumberGenerator()
        return Sampling.sample(
            logits,
            parameters: GenerationParameters(temperature: temperature),
            rng: &rng)
    }

    /// Forward one decode token through the backbone, returning `[vocab]`
    /// logits. The GPU `commit` + `waitUntilCompleted` live here, in a
    /// synchronous helper, so the async `transcribeStream` `Task` only
    /// calls into it and never blocks on the GPU directly.
    private func forwardDecodeToken(
        _ backbone: Qwen3Model, tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol], device: Device
    ) -> Tensor {
        let cmd = device.makeCommandBuffer()
        let logits = backbone.forward(
            tokenId: tokenId, position: position,
            caches: caches, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        return logits
    }

    /// Build the Qwen-Omni transcription prompt token ids, embed them on
    /// the backbone, splice the encoded audio rows into the
    /// `<|audio_pad|>` positions, prefill the KV cache, and sample the
    /// first decode token. Throws when the backbone or tokenizer is
    /// absent (an audio-only build, or a tokenizer that was never
    /// attached).
    private func prepareTranscribe(
        _ waveform: [Float], temperature: Float, device: Device
    ) throws -> TranscribeState {
        guard let backbone = textBackbone else {
            throw QwenOmniError.textBackboneUnavailable
        }
        guard let tokenizer = tokenizer else {
            throw QwenOmniError.tokenizerUnavailable
        }

        // 1. Encode the waveform → [nAudioTokens, textHidden] feature rows.
        let audioFeatures = encodeAudio(waveform: waveform, device: device)
        let nAudioTokens = audioFeatures.shape[0]

        // 2. Build the chat prompt with `nAudioTokens` audio-pad slots
        //    wrapped in the audio-span markers, then the assistant turn.
        let audioPad = String(repeating: "<|audio_pad|>", count: nAudioTokens)
        let prompt =
            "<|im_start|>system\n<|im_end|>\n"
            + "<|im_start|>user\n<|audio_start|>"
            + audioPad
            + "<|audio_end|><|im_end|>\n"
            + "<|im_start|>assistant\n"
        let promptIds = tokenizer.encode(text: prompt)
        let promptLen = promptIds.count

        // 3. Build the spliced prompt-embedding stream: text tokens get
        //    their table embedding, audio-pad positions get the next
        //    encoded audio feature row.
        let caches = backbone.makeLayerCaches(maxSeq: nil, device: device)
        var lastLogits: Tensor!
        var audioIdx = 0
        for (pos, tokenId) in promptIds.enumerated() {
            let rowEmbed: Tensor
            if tokenId == audioTokenId, audioIdx < nAudioTokens {
                rowEmbed = audioFeatures
                    .slicedRows(start: audioIdx, count: 1)
                    .reshaped(to: [config.textHidden])
                audioIdx += 1
            } else {
                rowEmbed = backbone.textEmbedding(
                    tokenId: tokenId, device: device)
            }
            let cmd = device.makeCommandBuffer()
            lastLogits = backbone.forward(
                inputEmbedding: rowEmbed, position: pos,
                caches: caches, on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // 4. Sample the first decode token from the prefill tail.
        let firstToken = sampleTok(lastLogits, temperature: temperature)
        return TranscribeState(
            backbone: backbone, tokenizer: tokenizer, caches: caches,
            firstToken: firstToken, promptLen: promptLen)
    }

    /// Batch transcription — prefill, then decode to EOS / `maxNewTokens`,
    /// returning the full trimmed transcript. Shares `prepareTranscribe`
    /// + the per-token loop with `transcribeStream`.
    public func transcribe(
        _ waveform: [Float],
        maxNewTokens: Int = 512,
        temperature: Float = 0.0,
        device: Device = .shared
    ) throws -> TranscriptionResult {
        let start = Date()
        let st = try prepareTranscribe(
            waveform, temperature: temperature, device: device)
        let eosIds = Set(eosTokenIds)
        var generated: [Int] = []
        var lastTok = st.firstToken
        for _ in 0 ..< maxNewTokens {
            if eosIds.contains(lastTok) { break }
            generated.append(lastTok)
            let nextLogits = forwardDecodeToken(
                st.backbone, tokenId: lastTok,
                position: st.promptLen + generated.count - 1,
                caches: st.caches, device: device)
            lastTok = sampleTok(nextLogits, temperature: temperature)
        }
        let text = st.tokenizer.decode(tokens: generated)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return TranscriptionResult(
            text: text, generatedTokens: generated.count,
            totalTimeS: Date().timeIntervalSince(start))
    }

    /// Streaming transcription — yields the incremental decoded TEXT (one
    /// delta per generated token) as the backbone decodes. Same
    /// audio-encode + prefill + per-token loop as `transcribe`, so
    /// concatenating the yielded deltas reproduces the batch transcript
    /// (modulo the batch's trailing trim). Cancelling the consuming task
    /// stops decode at the next step.
    ///
    /// The GPU sync stays inside the synchronous `prepareTranscribe` /
    /// `backbone.forward(...)` helpers; the async `Task` only orchestrates
    /// them, checks cancellation, and yields text — never blocks on the
    /// GPU itself.
    public func transcribeStream(
        _ waveform: [Float],
        maxNewTokens: Int = 512,
        temperature: Float = 0.0
    ) -> AsyncThrowingStream<String, Error> {
        let model = self
        let device = Device.shared
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let st = try model.prepareTranscribe(
                        waveform, temperature: temperature, device: device)
                    let eosIds = Set(model.eosTokenIds)
                    var generated: [Int] = []
                    var lastTok = st.firstToken
                    for _ in 0 ..< maxNewTokens {
                        try Task.checkCancellation()
                        if eosIds.contains(lastTok) { break }
                        generated.append(lastTok)
                        // Decode the running prefix each step and emit the
                        // new suffix — robust to BPE tokens that don't
                        // decode standalone; concatenation == batch text.
                        let prev =
                            generated.count == 1
                            ? ""
                            : st.tokenizer.decode(
                                tokens: Array(generated.dropLast()))
                        let cur = st.tokenizer.decode(tokens: generated)
                        if cur.count > prev.count {
                            continuation.yield(String(cur.dropFirst(prev.count)))
                        }
                        let nextLogits = model.forwardDecodeToken(
                            st.backbone, tokenId: lastTok,
                            position: st.promptLen + generated.count - 1,
                            caches: st.caches, device: device)
                        lastTok = model.sampleTok(
                            nextLogits, temperature: temperature)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// Streaming-transcription conformance. The protocol-shaped
/// `transcribeStream(_:maxNewTokens:temperature:)` is implemented above;
/// the conformance is a marker so a loaded QwenOmni surfaces as a
/// `StreamingTranscriber` (see `AudioStreaming.swift`).
extension QwenOmniModel: StreamingTranscriber {}

// ─── Loading ─────────────────────────────────────────────────────────

extension QwenOmniModel {
    public static let modelTypes: Set<String> = [
        "qwen2_5_omni", "qwen3_omni", "qwen2_5_omni_thinker",
    ]
    public static let architectures: Set<String> = [
        "Qwen2_5OmniForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
    ]

    /// Whether a decoded `config.json` describes a QwenOmni checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        // Fall back to structural detection — an `audio_config` block,
        // top-level or nested under `thinker_config`.
        return QwenOmniAudioConfig.locateConfigs(config) != nil
    }

    /// Load the QwenOmni audio path from a resolved snapshot directory.
    /// The audio encoder is always constructed; the text backbone is
    /// loaded when the checkpoint's text weights are present and the
    /// vision path is left for the Qwen-VL port (see the scope note).
    public static func load(directory: URL, device: Device = .shared)
        throws -> QwenOmniModel
    {
        let config = try ModelConfig.load(from: directory)
        guard let qc = QwenOmniAudioConfig.from(config) else {
            throw ModelError.unsupportedModelType(
                "config.json has no audio_config — not a QwenOmni checkpoint")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return try build(config: qc, bundle: bundle, modelConfig: config)
    }

    /// Assemble a `QwenOmniModel` audio path from a decoded config + a
    /// weight bundle. Factored out so tests can drive it directly.
    ///
    /// `modelConfig` is the full decoded `config.json` — used to wire the
    /// Qwen3 text backbone (the audio→text decode path). When `nil` (the
    /// encoder-only test path), the backbone is left unwired and the
    /// model is the documented audio-only build.
    public static func build(
        config qc: QwenOmniAudioConfig,
        bundle: SafeTensorsBundle,
        modelConfig: ModelConfig? = nil
    ) throws -> QwenOmniModel {
        // Qwen-Omni prefixes the audio tower weights; conversions vary
        // (`audio_tower.` / `thinker.audio_tower.`). Detect the prefix.
        let prefixes = [
            "thinker.audio_tower.", "audio_tower.",
            "model.audio_tower.", "",
        ]
        guard
            let prefix = prefixes.first(where: {
                bundle.has("\($0)conv1.weight")
            })
        else {
            throw ModelError.unsupportedModelType(
                "QwenOmni: no audio_tower weights found in the checkpoint")
        }
        let dtype = try bundle.tensor(named: "\(prefix)conv1.weight").dtype

        func t(_ name: String) throws -> Tensor {
            try bundle.tensor(named: prefix + name)
        }
        func ln(_ base: String) throws -> LayerNorm {
            LayerNorm(
                weight: try t("\(base).weight"),
                bias: try t("\(base).bias"), eps: 1e-5)
        }
        func linear(_ base: String, hasBias: Bool = true) throws -> Linear {
            let w = try t("\(base).weight")
            let b =
                hasBias && bundle.has(prefix + "\(base).bias")
                ? try t("\(base).bias") : nil
            return Linear(weight: w, bias: b)
        }

        var encLayers: [AudioEncoderLayer] = []
        for i in 0 ..< qc.encoderLayers {
            let base = "layers.\(i)"
            encLayers.append(
                AudioEncoderLayer(
                    layerNorm1: try ln("\(base).self_attn_layer_norm"),
                    qProj: try linear("\(base).self_attn.q_proj"),
                    kProj: try linear("\(base).self_attn.k_proj", hasBias: false),
                    vProj: try linear("\(base).self_attn.v_proj"),
                    oProj: try linear("\(base).self_attn.out_proj"),
                    layerNorm2: try ln("\(base).final_layer_norm"),
                    fc1: try linear("\(base).fc1"),
                    fc2: try linear("\(base).fc2"),
                    hidden: qc.encoderHidden, nHeads: qc.encoderHeads,
                    intermediate: qc.encoderIntermediate))
        }
        let encoderConfig = AudioEncoderConfig(
            nMels: qc.nMels, hidden: qc.encoderHidden,
            intermediate: qc.encoderIntermediate, nLayers: qc.encoderLayers,
            nHeads: qc.encoderHeads, maxAudioCtx: qc.maxAudioCtx,
            layerNormEps: 1e-5)

        // Qwen-Omni's audio encoder uses a fixed sinusoidal position
        // encoding computed at runtime — unlike Whisper it does NOT
        // store `embed_positions.weight` in the checkpoint. Synthesize
        // the table when the weight is absent.
        let posEmbedding: Tensor
        if bundle.has(prefix + "embed_positions.weight") {
            posEmbedding = try t("embed_positions.weight")
        } else {
            let table = MediaUtils.sinusoidalPositions(
                length: qc.maxAudioCtx, dim: qc.encoderHidden)
            let pe = Tensor.empty(
                shape: [qc.maxAudioCtx, qc.encoderHidden],
                dtype: dtype)
            MediaUtils.copyFloats(table, into: pe)
            posEmbedding = pe
        }

        let encoder = AudioEncoder(
            config: encoderConfig,
            conv1Weight: try t("conv1.weight"),
            conv1Bias: try t("conv1.bias"),
            conv2Weight: try t("conv2.weight"),
            conv2Bias: try t("conv2.bias"),
            positionEmbedding: posEmbedding,
            layers: encLayers,
            postLayerNorm: try ln("ln_post"),
            dtype: dtype)

        // Audio→text projection, if the checkpoint carries one and the
        // dims differ.
        var audioProjection: Linear? = nil
        if qc.encoderHidden != qc.textHidden {
            for projBase in [
                "proj", "audio_bos_eos_token.proj",
                "multi_modal_projector",
            ] {
                if bundle.has(prefix + "\(projBase).weight") {
                    audioProjection = try linear(projBase)
                    break
                }
            }
        }

        // Wire the Qwen3 text backbone when the checkpoint carries text
        // weights. Qwen-Omni nests the text decoder under `thinker.model.`
        // (and the text hyper-params under `thinker_config.text_config`);
        // some mlx conversions hoist them to `model.` / `text_config`.
        // Best-effort: leave `textBackbone` nil (the documented
        // audio-only build) when no text weights are present, so the
        // audio-encode path and tests that build encoder-only still work.
        let textBackbone = try? loadTextBackbone(
            modelConfig: modelConfig, bundle: bundle)

        return QwenOmniModel(
            config: qc, audioEncoder: encoder,
            audioProjection: audioProjection, textBackbone: textBackbone,
            dtype: dtype)
    }

    /// Load the Qwen3 text backbone from a Qwen-Omni checkpoint, or return
    /// `nil` when the text weights / config aren't present (an audio-only
    /// build). The text decoder is the standard Qwen3 dense engine —
    /// QwenOmni composes it rather than reimplementing it.
    static func loadTextBackbone(
        modelConfig: ModelConfig?, bundle: SafeTensorsBundle
    ) throws -> Qwen3Model? {
        guard let modelConfig else { return nil }

        // The text sub-config lives under `thinker_config.text_config`
        // (Qwen2.5-Omni) or top-level `text_config` (hoisted conversions).
        let textRaw: [String: Any]?
        if let thinker = modelConfig.nested("thinker_config"),
            let text = thinker["text_config"] as? [String: Any]
        {
            textRaw = text
        } else {
            textRaw = modelConfig.nested("text_config")
        }
        guard let textRaw else { return nil }

        // Detect the prefix the text decoder weights live under. Qwen-Omni
        // stores them beneath the `thinker.` namespace; hoisted
        // conversions use the canonical `model.` prefix the Qwen3 loader
        // expects directly.
        let textPrefixes = ["thinker.model.", "model.", "thinker."]
        guard
            let textPrefix = textPrefixes.first(where: {
                bundle.has("\($0)embed_tokens.weight")
                    || bundle.has("\($0)layers.0.input_layernorm.weight")
            })
        else { return nil }

        // The Qwen3 dense loader reads weights under the canonical
        // `model.X` / `lm_head.X` prefix. Strip the detected prefix and
        // re-add `model.` so the standard loader binds unmodified; the
        // tied / standalone lm_head is read from the root namespace.
        let textBundle = bundle.prefixed(textPrefix).withAddedPrefix("model.")
        let textConfig = ModelConfig(
            architecture: "Qwen3ForCausalLM",
            modelType: "qwen3",
            raw: textRaw)
        return try Qwen3Dense.loadModel(
            config: textConfig, weights: textBundle,
            options: LoadOptions(), device: .shared)
    }
}
