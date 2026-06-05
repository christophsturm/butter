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
// MiniCPM-o audio tower — the audio-IN half of the omni model.
//
//   waveform ──Whisper log-Mel front-end (nMels 80, nFFT 400, hop 160)──▶
//            ──Whisper encoder (`audio_tower.*`, 24 layers, 1024 hidden)──▶
//            [seq, 1024]
//            ──projector (linear1 1024→4096 → ReLU → linear2 4096→4096)──▶
//            [seq, 4096]
//            ──avg-pool stride `audio_pool_step` (5) along seq──▶
//            [seq/5, 4096]   (audio soft tokens in the text hidden dim)
//
// The order is encode → PROJECT → POOL, matching the reference
// `get_audio_embedding`: `audio_projection_layer(audio_states)` then
// `audio_avg_pooler(...)`. The Whisper encoder reuses the shared
// `AudioEncoder` (its `audio_tower.*` key layout matches exactly); the
// projector is the 4-bit `MultiModalProjector` (`audio_projection_layer.*`).
//
// Reference: `get_audio_embedding` / `MultiModalProjector` in
// `modeling_minicpmo.py` (HF `openbmb/MiniCPM-o-4_5`). The `audio_config`
// is `openai/whisper-medium` (d_model 1024, 24 encoder layers, 16 heads,
// encoder_ffn_dim 4096, num_mel_bins 80, max_source_positions 1500).

import Foundation
import Metal

// ─── MiniCPM4Audio ───────────────────────────────────────────────────

/// The MiniCPM-o audio tower: Whisper-style encoder + 4-bit projector +
/// average pool. `encode(waveform:)` turns a raw 16 kHz mono signal into
/// `[nAudioTokens, textHidden]` soft tokens for the cross-modal splice.
///
/// `@unchecked Sendable`: every stored property is an immutable
/// GPU-backed value (the encoder, two linears, and scalar config) read but
/// never mutated by `encode`, so the `audioEncode` `@Sendable` hook can
/// capture it safely. Mirrors `Gemma4AudioEncoder`.
final class MiniCPM4Audio: @unchecked Sendable {
    /// The Whisper-style audio encoder (`audio_tower.*`).
    let encoder: AudioEncoder
    /// Projector first linear (`audio_projection_layer.linear1`):
    /// `audioHidden → textHidden`, 4-bit with bias.
    let linear1: AnyLinear
    /// Projector second linear (`audio_projection_layer.linear2`):
    /// `textHidden → textHidden`, 4-bit with bias.
    let linear2: AnyLinear
    /// Average-pool stride along the sequence axis (`audio_pool_step`, 5).
    let poolStep: Int
    /// Text hidden dim the projector maps into (`embed_dim`, 4096).
    let textHidden: Int
    /// Working dtype (matches the encoder activation dtype).
    let dtype: DType

    init(
        encoder: AudioEncoder, linear1: AnyLinear, linear2: AnyLinear,
        poolStep: Int, textHidden: Int, dtype: DType
    ) {
        self.encoder = encoder
        self.linear1 = linear1
        self.linear2 = linear2
        self.poolStep = poolStep
        self.textHidden = textHidden
        self.dtype = dtype
    }

    /// MiniCPM-o / Whisper-medium fixed analysis window: 30 s of 16 kHz
    /// audio. The encoder's 1500-row positional table assumes it, so the
    /// waveform is padded / trimmed to exactly this length (OpenAI's
    /// `pad_or_trim`).
    static let audioWindowSamples = 30 * 16_000

    /// The Whisper log-Mel front-end config that pairs with the audio
    /// tower (`num_mel_bins` 80, 16 kHz, 400-sample FFT, 160-sample hop).
    var frontEnd: AudioFrontEndConfig {
        AudioFrontEndConfig(
            sampleRate: 16_000, nFFT: 400, hopLength: 160,
            nMels: encoder.config.nMels)
    }

    /// Load the audio tower from the top-level bundle. The Whisper encoder
    /// loads from the `audio_tower.` prefix (mirroring Whisper `build`);
    /// the projector from `audio_projection_layer.`.
    static func load(
        weights: SafeTensorsBundle, audioConfig: ModelConfig,
        textHidden: Int, poolStep: Int,
        quantization: ModelConfig.QuantizationConfig?,
        device: Device
    ) throws -> MiniCPM4Audio {
        guard let hidden = audioConfig.int("d_model"),
            let nLayers = audioConfig.int("encoder_layers"),
            let nHeads = audioConfig.int("encoder_attention_heads")
        else {
            throw MiniCPMOError.missingConfig(
                "audio_config d_model / encoder_layers / encoder_attention_heads")
        }
        let nMels = audioConfig.int("num_mel_bins") ?? 80
        let intermediate = audioConfig.int("encoder_ffn_dim") ?? (4 * hidden)
        let maxAudioCtx = audioConfig.int("max_source_positions") ?? 1500

        let audio = weights.prefixed("audio_tower.")
        let dtype = try audio.tensor(named: "conv1.weight").dtype

        // ── Whisper encoder blocks (reuse AudioEncoderLayer) ──────────
        // The `audio_tower.*` key layout is exactly the Whisper encoder's
        // (`layers.N.self_attn.{q,k,v,out}_proj`, `self_attn_layer_norm`,
        // `final_layer_norm`, `fc1`/`fc2`, plus the conv stem +
        // `embed_positions` + post `layer_norm`). `k_proj` ships no bias.
        func t(_ name: String) throws -> Tensor { try audio.tensor(named: name) }
        func ln(_ base: String) throws -> LayerNorm {
            LayerNorm(
                weight: try t("\(base).weight"),
                bias: try t("\(base).bias"), eps: 1e-5)
        }
        func linear(_ base: String, hasBias: Bool = true) throws -> Linear {
            let w = try t("\(base).weight")
            let b =
                hasBias && audio.has("\(base).bias")
                ? try t("\(base).bias") : nil
            return Linear(weight: w, bias: b)
        }

        var encLayers: [AudioEncoderLayer] = []
        encLayers.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
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
                    hidden: hidden, nHeads: nHeads,
                    intermediate: intermediate))
        }
        let encoderConfig = AudioEncoderConfig(
            nMels: nMels, hidden: hidden, intermediate: intermediate,
            nLayers: nLayers, nHeads: nHeads,
            maxAudioCtx: maxAudioCtx, layerNormEps: 1e-5)
        let encoder = AudioEncoder(
            config: encoderConfig,
            conv1Weight: try t("conv1.weight"),
            conv1Bias: try t("conv1.bias"),
            conv2Weight: try t("conv2.weight"),
            conv2Bias: try t("conv2.bias"),
            positionEmbedding: try t("embed_positions.weight"),
            layers: encLayers,
            postLayerNorm: try ln("layer_norm"),
            dtype: dtype)

        // ── Projector (4-bit MultiModalProjector) ─────────────────────
        let projWeights = weights.prefixed("audio_projection_layer.")
        let linear1 = try loadLinear(
            base: "linear1", in: projWeights, quantization: quantization)
        let linear2 = try loadLinear(
            base: "linear2", in: projWeights, quantization: quantization)

        return MiniCPM4Audio(
            encoder: encoder, linear1: linear1, linear2: linear2,
            poolStep: poolStep, textHidden: textHidden, dtype: dtype)
    }

    /// Encode a raw 16 kHz mono waveform into `[nAudioTokens, textHidden]`
    /// audio soft tokens: log-Mel front-end → Whisper encoder → projector
    /// (linear1 → ReLU → linear2) → average pool (stride `poolStep`).
    func encode(waveform: [Float], device: Device = .shared) -> Tensor {
        // ── Log-Mel front-end (Whisper framing) ───────────────────────
        // The mel_spectrogram kernel only emits f32 / f16; a bf16 model
        // runs the front-end in f32, then casts the spectrogram to the
        // encoder dtype before the conv stem.
        let melDtype: DType = dtype == .f16 ? .f16 : .f32
        let framed = MiniCPM4Audio.padOrTrim(
            waveform, to: MiniCPM4Audio.audioWindowSamples)
        let cmd = device.makeCommandBuffer()
        // `whisperNormalize: true` — `logMelSpectrogram` commits + waits on
        // `cmd` internally (CPU-side normalisation), so the result is
        // already CPU-synced.
        let melRaw = MediaUtils.logMelSpectrogram(
            waveform: framed, cfg: frontEnd, dtype: melDtype,
            whisperNormalize: true, device: device, on: cmd)
        var mel = MediaUtils.castTensor(melRaw, to: dtype, device: device)
        // Trim the surplus front-end frame (the kernel emits one more
        // column than Whisper's `stft[..., :-1]`) so `nAudioCtx` lands at
        // `maxAudioCtx`, not `maxAudioCtx + 1`, after the stride-2 conv.
        let maxMelFrames = 2 * encoder.config.maxAudioCtx
        if mel.shape[0] > maxMelFrames {
            mel = mel.slicedRows(start: 0, count: maxMelFrames)
        }

        // ── Whisper encoder → [nAudioCtx, audioHidden] ────────────────
        let encoded = encoder.encode(
            mel: mel, melFrameMajor: true, device: device)
        let nAudioCtx = encoded.shape[0]

        // ── Projector: linear1 → ReLU → linear2 → [nAudioCtx, textHidden]
        let cmdP = device.makeCommandBuffer()
        let h1 = linear1.callMany(encoded, t: nAudioCtx, on: cmdP, device: device)
        let act = Ops.relu(h1, on: cmdP)
        let projected = linear2.callMany(act, t: nAudioCtx, on: cmdP, device: device)
        cmdP.commit()
        cmdP.waitUntilCompleted()

        // ── Average pool stride `poolStep` along the sequence axis ─────
        return averagePool(projected, seq: nAudioCtx, device: device)
    }

    /// Average-pool `[seq, textHidden]` along the sequence axis with a
    /// `poolStep`-wide, stride-`poolStep`, non-padded window — `nn.AvgPool1d
    /// (poolStep, stride=poolStep)`. Output rows = `seq / poolStep` (the
    /// trailing remainder is dropped, matching PyTorch's default no-padding
    /// `AvgPool1d`). CPU pass (a few thousand rows × 4096 — cheap relative
    /// to the encoder GEMMs).
    private func averagePool(
        _ x: Tensor, seq: Int, device: Device
    ) -> Tensor {
        let outRows = seq / poolStep
        precondition(
            outRows > 0,
            "MiniCPM4Audio.averagePool: \(seq) frames < poolStep \(poolStep)")
        let src = x.toFloatArray()
        var dst = [Float](repeating: 0, count: outRows * textHidden)
        let inv = 1.0 / Float(poolStep)
        for r in 0 ..< outRows {
            let dstBase = r * textHidden
            for s in 0 ..< poolStep {
                let srcBase = (r * poolStep + s) * textHidden
                for c in 0 ..< textHidden {
                    dst[dstBase + c] += src[srcBase + c]
                }
            }
            for c in 0 ..< textHidden { dst[dstBase + c] *= inv }
        }
        return MediaUtils.makeTensor(
            from: dst, shape: [outRows, textHidden],
            dtype: dtype, device: device)
    }

    /// Pad (trailing zeros) or trim a waveform to exactly `length`
    /// samples — OpenAI Whisper's `pad_or_trim`.
    static func padOrTrim(_ waveform: [Float], to length: Int) -> [Float] {
        if waveform.count == length { return waveform }
        if waveform.count > length { return Array(waveform[0 ..< length]) }
        return waveform + [Float](repeating: 0, count: length - waveform.count)
    }
}
