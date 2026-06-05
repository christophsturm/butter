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
// AudioEncoder — the speech-transformer stack a speech model runs a
// waveform through before the text decoder (Whisper STT) or the
// cross-modal splice (Qwen-Omni audio-in).
//
// Declared in for the capability API; lit up here in .
// The architecture is the Whisper-style audio encoder every shipped
// STT / audio-in tower builds on:
//
//   log-Mel  ──conv1d(k=3,s=1) → GELU──▶
//            ──conv1d(k=3,s=2) → GELU──▶  [n_frames/2, hidden]
//            ──+ sinusoidal position embedding──▶
//            ──▶ N × { LayerNorm → MHA (bidirectional) → +residual
//                      LayerNorm → MLP (GELU)          → +residual }
//            ──post-LayerNorm──▶  [n_audio_ctx, hidden]
//
// The encoder processes ALL audio frames at once (audio-encoder
// attention is bidirectional — no causal mask, no KV cache), so it
// mirrors `VisionEncoder`: GPU GEMMs / norms / convs queued on a
// command buffer, a CPU bidirectional attention core (head-dim-agnostic
// and unambiguously correct — a head-dim-aware audio SDPA kernel is a
// later performance pass).
//
// This file is the shared core; Whisper and Qwen-Omni both build one.

import Foundation
import Metal

// ─── Configuration ───────────────────────────────────────────────────

/// Static shape + hyper-parameters of a Whisper-style audio encoder,
/// decoded from the checkpoint's `config.json`.
public struct AudioEncoderConfig: Sendable {
    /// Number of Mel filterbank bins the front-end produces (the conv
    /// stem's input channels). Whisper tiny→medium: 80; large-v3: 128.
    public let nMels: Int
    /// Encoder hidden dimension (`d_model`).
    public let hidden: Int
    /// Encoder feed-forward intermediate dimension.
    public let intermediate: Int
    /// Number of transformer blocks.
    public let nLayers: Int
    /// Number of attention heads per block.
    public let nHeads: Int
    /// Maximum audio-context length (positional-embedding rows). Whisper
    /// fixes this at 1500 — a 30 s clip at the 16 kHz / hop-160 / conv
    /// stride-2 framing.
    public let maxAudioCtx: Int
    /// LayerNorm epsilon.
    public let layerNormEps: Float

    public init(
        nMels: Int, hidden: Int, intermediate: Int,
        nLayers: Int, nHeads: Int, maxAudioCtx: Int = 1500,
        layerNormEps: Float = 1e-5
    ) {
        self.nMels = nMels
        self.hidden = hidden
        self.intermediate = intermediate
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.maxAudioCtx = maxAudioCtx
        self.layerNormEps = layerNormEps
    }

    /// Per-head dimension.
    public var headDim: Int { hidden / nHeads }
}

// ─── Audio encoder ───────────────────────────────────────────────────

/// A Whisper-style audio encoder. Holds the two conv-stem layers, the
/// (fixed sinusoidal) position embedding, the encoder block stack and
/// the post-encoder LayerNorm.
public final class AudioEncoder: Module {
    public let config: AudioEncoderConfig

    /// First conv: `[hidden, nMels, 3]`, stride 1, pad 1.
    public let conv1Weight: Tensor
    /// First conv bias `[hidden]`.
    public let conv1Bias: Tensor
    /// Second conv: `[hidden, hidden, 3]`, stride 2, pad 1.
    public let conv2Weight: Tensor
    /// Second conv bias `[hidden]`.
    public let conv2Bias: Tensor
    /// Position embedding `[maxAudioCtx, hidden]` (Whisper bakes a fixed
    /// sinusoidal table into the checkpoint as `embed_positions.weight`).
    public let positionEmbedding: Tensor
    /// Encoder blocks.
    public let layers: [AudioEncoderLayer]
    /// Post-encoder LayerNorm.
    public let postLayerNorm: LayerNorm

    public let dtype: DType

    public init(
        config: AudioEncoderConfig,
        conv1Weight: Tensor, conv1Bias: Tensor,
        conv2Weight: Tensor, conv2Bias: Tensor,
        positionEmbedding: Tensor, layers: [AudioEncoderLayer],
        postLayerNorm: LayerNorm, dtype: DType
    ) {
        self.config = config
        self.conv1Weight = conv1Weight
        self.conv1Bias = conv1Bias
        self.conv2Weight = conv2Weight
        self.conv2Bias = conv2Bias
        self.positionEmbedding = positionEmbedding
        self.layers = layers
        self.postLayerNorm = postLayerNorm
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        out.append(("conv1.weight", conv1Weight))
        out.append(("conv1.bias", conv1Bias))
        out.append(("conv2.weight", conv2Weight))
        out.append(("conv2.bias", conv2Bias))
        out.append(("embed_positions.weight", positionEmbedding))
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("layers.\(i).\(k)", v))
            }
        }
        for (k, v) in postLayerNorm.parameters() {
            out.append(("layer_norm.\(k)", v))
        }
        return out
    }

    /// Encode a log-Mel spectrogram into audio-frame embeddings.
    ///
    /// `mel` is the `[nMels, nFrames]` channel-major log-Mel from the
    /// front-end (`Ops.melSpectrogram` produces `[nFrames, nMels]` —
    /// transpose before calling, or pass `melFrameMajor: true` to have
    /// `encode` accept the frame-major layout and transpose internally).
    /// Returns `[nAudioCtx, hidden]` — the audio tokens a decoder
    /// cross-attends to. `nAudioCtx = nFrames / 2` (the stride-2 conv).
    ///
    /// All GPU work is queued on private command buffers committed +
    /// waited before returning, since callers consume the result on the
    /// CPU or hand it to a decoder loop.
    public func encode(
        mel: Tensor, melFrameMajor: Bool = false,
        device: Device = .shared
    ) -> Tensor {
        precondition(mel.shape.count == 2, "AudioEncoder.encode: mel must be 2D")
        // Normalise to channel-major [nMels, nFrames] — the conv-stem
        // input layout.
        let nFrames: Int
        let melCM: Tensor
        if melFrameMajor {
            precondition(
                mel.shape[1] == config.nMels,
                "AudioEncoder.encode: frame-major mel must be "
                    + "[nFrames, nMels=\(config.nMels)]")
            nFrames = mel.shape[0]
            melCM = transpose2D(
                mel, rows: nFrames, cols: config.nMels,
                device: device)
        } else {
            precondition(
                mel.shape[0] == config.nMels,
                "AudioEncoder.encode: channel-major mel must be "
                    + "[nMels=\(config.nMels), nFrames]")
            nFrames = mel.shape[1]
            melCM = mel
        }

        // ── Conv stem ──
        // conv1: [nMels → hidden], k=3, s=1, pad=1 → [hidden, nFrames].
        let cmd = device.makeCommandBuffer()
        let conv1In = melCM.reshaped(to: [1, config.nMels, nFrames])
        let conv1Out = Ops.audioConv1d(
            input: conv1In, weight: conv1Weight, bias: conv1Bias,
            batch: 1, inCh: config.nMels, inLen: nFrames, outCh: config.hidden,
            k: 3, stride: 1, pad: 1, on: cmd)
        let conv1Act = Ops.gelu(conv1Out, on: cmd)
        // conv2: [hidden → hidden], k=3, s=2, pad=1 → [hidden, nFrames/2].
        let nAudioCtx = (nFrames + 2 - 3) / 2 + 1
        let conv2Out = Ops.audioConv1d(
            input: conv1Act.reshaped(to: [1, config.hidden, nFrames]),
            weight: conv2Weight, bias: conv2Bias,
            batch: 1, inCh: config.hidden, inLen: nFrames, outCh: config.hidden,
            k: 3, stride: 2, pad: 1, on: cmd)
        let conv2Act = Ops.gelu(conv2Out, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // conv output is channel-major [hidden, nAudioCtx]; the
        // transformer wants frame-major [nAudioCtx, hidden].
        var h = transpose2D(
            conv2Act.reshaped(to: [config.hidden, nAudioCtx]),
            rows: config.hidden, cols: nAudioCtx,
            device: device)

        // ── + position embedding ──
        precondition(
            nAudioCtx <= config.maxAudioCtx,
            "AudioEncoder.encode: \(nAudioCtx) frames exceed "
                + "maxAudioCtx \(config.maxAudioCtx)")
        let posSlice = positionEmbedding.slicedRows(start: 0, count: nAudioCtx)
        let cmdP = device.makeCommandBuffer()
        h = Ops.add(h, posSlice, on: cmdP)
        cmdP.commit()
        cmdP.waitUntilCompleted()

        // ── Encoder block stack ──
        for layer in layers {
            let cmdL = device.makeCommandBuffer()
            h = layer.forward(h, nFrames: nAudioCtx, device: device, on: cmdL)
        }

        // ── Post-encoder LayerNorm ──
        let cmdN = device.makeCommandBuffer()
        h = Ops.layerNorm(
            h, weight: postLayerNorm.weight,
            bias: postLayerNorm.bias, eps: postLayerNorm.eps,
            nRows: nAudioCtx, rowSize: config.hidden, on: cmdN)
        cmdN.commit()
        cmdN.waitUntilCompleted()
        return h
    }

    /// CPU transpose of a 2D `[rows, cols]` tensor to `[cols, rows]`.
    /// Frame counts × hidden are at most a couple million elements —
    /// cheap relative to the encoder GEMMs.
    private func transpose2D(
        _ x: Tensor, rows: Int, cols: Int,
        device: Device
    ) -> Tensor {
        let src = x.toFloatArray()
        var dst = [Float](repeating: 0, count: rows * cols)
        for r in 0 ..< rows {
            for c in 0 ..< cols {
                dst[c * rows + r] = src[r * cols + c]
            }
        }
        let out = Tensor.empty(shape: [cols, rows], dtype: dtype, device: device)
        MediaUtils.copyFloats(dst, into: out)
        return out
    }

    /// Add a `[rowSize]` bias to each of `nRows` rows of a flat
    /// `[nRows, rowSize]` tensor. Shared by `AudioEncoderLayer`.
    static func addRowBias(
        _ x: Tensor, bias: Tensor, nRows: Int,
        rowSize: Int, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let tiled = Tensor.empty(shape: [nRows, rowSize], dtype: x.dtype)
        let biasVals = bias.toFloatArray()
        var flat = [Float](repeating: 0, count: nRows * rowSize)
        for r in 0 ..< nRows {
            for c in 0 ..< rowSize { flat[r * rowSize + c] = biasVals[c] }
        }
        MediaUtils.copyFloats(flat, into: tiled)
        return Ops.add(x, tiled, on: cmd)
    }
}
