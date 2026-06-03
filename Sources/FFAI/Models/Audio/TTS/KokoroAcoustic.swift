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
// Kokoro / StyleTTS2 acoustic front-end — the text→spectrogram stack that
// feeds the iSTFTNet vocoder (`KokoroVocoder`, GPU). Architecture, mapped
// 1:1 from the `kokoro-v1_0.safetensors` checkpoint (548 weights):
//
//   phonemes ─▶ PLBERT (ALBERT, 12 shared layers) ─▶ bert_encoder (→512)
//            ─▶ text_encoder (3× conv-norm + BiLSTM, acoustic features)
//            ─▶ predictor (duration LSTM + F0/N AdaIN blocks)
//            ─▶ length regulation (duration → frame expansion)
//            ─▶ decoder (AdaIN res-blocks)
//            ─▶ generator (iSTFTNet: NSF source + upsample + Snake/AdaIN
//                          res-blocks → [re, im]) ─▶ iSTFT ─▶ waveform
//
// Build order: this file lands the loader + PLBERT first (verifiable: the
// encoder is a standard transformer whose output is shape/finiteness-
// checkable via InspectTap). The prosody predictor + decoder + generator
// reuse FFAI's CPU audio primitives (`WeightNorm`, `AudioPrimitives.conv1d`,
// `AudioMath.snake`, `BigVGANAMPBlock`) and land incrementally on top.
//
// Numerics run CPU-first (the acoustic stack is tiny — 82M params, run once
// per utterance); the heavy iSTFT tail is already on the GPU. GPU-residency
// of the front-end is a follow-up, mirroring the GraniteSpeech encoder port.

import Foundation
import Metal

// ─── Weight access ───────────────────────────────────────────────────

/// Thin CPU weight reader over the Kokoro safetensors bundle: pulls tensors
/// to `[Float]` and reconstructs weight-normalized conv weights `(g, v) →
/// g·v/‖v‖` (StyleTTS2 stores every conv as a `weight_g`/`weight_v` pair).
struct KokoroWeights {
    let bundle: SafeTensorsBundle

    /// A plain float tensor by key.
    func vec(_ name: String) -> [Float] {
        (try? bundle.tensor(named: name))?.toFloatArray() ?? []
    }

    /// Shape of a stored tensor.
    func shape(_ name: String) -> [Int] {
        (try? bundle.tensor(named: name))?.shape ?? []
    }

    /// Whether a key exists.
    func has(_ name: String) -> Bool { bundle.has(name) }

    /// Effective conv weight from a weight-normalized `base` (`base.weight_g`,
    /// `base.weight_v`). Returns `[Cout, Cin, K]` row-major.
    func convWeight(_ base: String) -> (w: [Float], shape: [Int]) {
        let g = vec("\(base).weight_g")
        let v = vec("\(base).weight_v")
        let sh = shape("\(base).weight_v")
        guard sh.count == 3 else { return (v, sh) }
        return (WeightNorm.effectiveWeight(g: g, v: v, shape: sh), sh)
    }

    /// Conv bias (may be absent for weight-norm-only modules).
    func convBias(_ base: String) -> [Float] { vec("\(base).bias") }
}

// ─── PLBERT (ALBERT) text encoder ────────────────────────────────────

/// PLBERT — the phoneme-level ALBERT used by Kokoro. A single shared
/// transformer layer applied `numLayers` times over the embedded phoneme
/// sequence. Bidirectional (no mask); the sequence is one utterance.
///
/// Checkpoint layout (`bert.*`): `embeddings.{word,position,token_type}`
/// (all 128-dim), `embeddings.LayerNorm`, `encoder.embedding_hidden_mapping_in`
/// (128→768), and one shared `albert_layer_groups.0.albert_layers.0` block:
/// `attention.{query,key,value,dense,LayerNorm}` (768), `ffn` (768→2048),
/// `ffn_output` (2048→768), `full_layer_layer_norm`.
struct KokoroPLBert {
    let w: KokoroWeights
    let hidden = 768
    let embDim = 128
    let heads = 12
    let intermediate = 2048
    let numLayers = 12

    /// Encode phoneme ids → contextual features `[L, 768]` (CPU `[[Float]]`).
    func encode(_ ids: [Int]) -> [[Float]] {
        let L = ids.count
        let wordE = w.vec("bert.embeddings.word_embeddings.weight")  // [178,128]
        let posE = w.vec("bert.embeddings.position_embeddings.weight")  // [512,128]
        let ttE = w.vec("bert.embeddings.token_type_embeddings.weight")  // [2,128]
        let lnW = w.vec("bert.embeddings.LayerNorm.weight")
        let lnB = w.vec("bert.embeddings.LayerNorm.bias")

        // Embedding sum (word + position + token_type[0]) then LayerNorm(128).
        var emb = [[Float]](repeating: [Float](repeating: 0, count: embDim), count: L)
        for i in 0 ..< L {
            let wb = ids[i] * embDim
            let pb = i * embDim
            for d in 0 ..< embDim { emb[i][d] = wordE[wb + d] + posE[pb + d] + ttE[d] }
            KokoroMath.layerNorm(&emb[i], weight: lnW, bias: lnB, eps: 1e-12)
        }

        // Embedding → hidden (128 → 768).
        let mapW = w.vec("bert.encoder.embedding_hidden_mapping_in.weight")  // [768,128]
        let mapB = w.vec("bert.encoder.embedding_hidden_mapping_in.bias")
        var h = KokoroMath.linear(emb, weight: mapW, bias: mapB, outDim: hidden, inDim: embDim)

        let p = "bert.encoder.albert_layer_groups.0.albert_layers.0"
        let qW = w.vec("\(p).attention.query.weight"), qB = w.vec("\(p).attention.query.bias")
        let kW = w.vec("\(p).attention.key.weight"), kB = w.vec("\(p).attention.key.bias")
        let vW = w.vec("\(p).attention.value.weight"), vB = w.vec("\(p).attention.value.bias")
        let dW = w.vec("\(p).attention.dense.weight"), dB = w.vec("\(p).attention.dense.bias")
        let aLnW = w.vec("\(p).attention.LayerNorm.weight"), aLnB = w.vec("\(p).attention.LayerNorm.bias")
        let fW = w.vec("\(p).ffn.weight"), fB = w.vec("\(p).ffn.bias")
        let foW = w.vec("\(p).ffn_output.weight"), foB = w.vec("\(p).ffn_output.bias")
        let flLnW = w.vec("\(p).full_layer_layer_norm.weight"), flLnB = w.vec("\(p).full_layer_layer_norm.bias")

        for _ in 0 ..< numLayers {
            // Multi-head self-attention.
            let q = KokoroMath.linear(h, weight: qW, bias: qB, outDim: hidden, inDim: hidden)
            let k = KokoroMath.linear(h, weight: kW, bias: kB, outDim: hidden, inDim: hidden)
            let v = KokoroMath.linear(h, weight: vW, bias: vB, outDim: hidden, inDim: hidden)
            let attn = KokoroMath.mha(q: q, k: k, v: v, heads: heads, dim: hidden)
            var dense = KokoroMath.linear(attn, weight: dW, bias: dB, outDim: hidden, inDim: hidden)
            for i in 0 ..< L {
                for d in 0 ..< hidden { dense[i][d] += h[i][d] }
                KokoroMath.layerNorm(&dense[i], weight: aLnW, bias: aLnB, eps: 1e-12)
            }
            // FFN (GELU).
            var ff = KokoroMath.linear(dense, weight: fW, bias: fB, outDim: intermediate, inDim: hidden)
            for i in 0 ..< L { KokoroMath.gelu(&ff[i]) }
            var ffOut = KokoroMath.linear(ff, weight: foW, bias: foB, outDim: hidden, inDim: intermediate)
            for i in 0 ..< L {
                for d in 0 ..< hidden { ffOut[i][d] += dense[i][d] }
                KokoroMath.layerNorm(&ffOut[i], weight: flLnW, bias: flLnB, eps: 1e-12)
            }
            h = ffOut
        }
        return h
    }
}

// ─── bert_encoder + acoustic text encoder ────────────────────────────

extension KokoroWeights {
    /// `bert_encoder` — projects PLBERT features 768 → 512.
    func bertEncode(_ h: [[Float]]) -> [[Float]] {
        KokoroMath.linear(
            h, weight: vec("bert_encoder.weight"), bias: vec("bert_encoder.bias"),
            outDim: 512, inDim: 768)
    }

    /// Weight-norm `Conv1d` over an `[L, Cin]` sequence → `[L, Cout]`.
    /// `pad` is symmetric (StyleTTS2 uses `same` padding, `k/2`).
    func conv1dSeq(_ x: [[Float]], base: String, pad: Int, dilation: Int = 1) -> [[Float]] {
        let (w, sh) = convWeight(base)
        let (cOut, cIn, _) = (sh[0], sh[1], sh[2])
        let L = x.count
        // [L, Cin] → channels-first flat [Cin, L].
        var xf = [Float](repeating: 0, count: cIn * L)
        for t in 0 ..< L { for c in 0 ..< cIn { xf[c * L + t] = x[t][c] } }
        let (yf, ysh) = AudioMath.conv1d(
            x: xf, xShape: [1, cIn, L], weight: w, wShape: sh, bias: convBias(base),
            stride: 1, padding: pad, dilation: dilation, groups: 1)
        let lOut = ysh[2]
        var out = [[Float]](repeating: [Float](repeating: 0, count: cOut), count: lOut)
        for c in 0 ..< cOut { for t in 0 ..< lOut { out[t][c] = yf[c * lOut + t] } }
        return out
    }

    /// Acoustic `text_encoder`: phoneme embedding → 3× (weight-norm conv k5 +
    /// channel-LayerNorm + LeakyReLU) → BiLSTM → `[L, 512]`.
    func textEncode(_ ids: [Int]) -> [[Float]] {
        let embW = vec("text_encoder.embedding.weight")  // [178, 512]
        var h = [[Float]](repeating: [Float](repeating: 0, count: 512), count: ids.count)
        for i in ids.indices { for d in 0 ..< 512 { h[i][d] = embW[ids[i] * 512 + d] } }
        for layer in 0 ..< 3 {
            h = conv1dSeq(h, base: "text_encoder.cnn.\(layer).0", pad: 2)
            KokoroMath.channelLayerNorm(
                &h, gamma: vec("text_encoder.cnn.\(layer).1.gamma"),
                beta: vec("text_encoder.cnn.\(layer).1.beta"))
            KokoroMath.leakyReLU(&h)
        }
        return KokoroMath.biLSTM(
            h,
            wihF: vec("text_encoder.lstm.weight_ih_l0"), whhF: vec("text_encoder.lstm.weight_hh_l0"),
            bihF: vec("text_encoder.lstm.bias_ih_l0"), bhhF: vec("text_encoder.lstm.bias_hh_l0"),
            wihR: vec("text_encoder.lstm.weight_ih_l0_reverse"),
            whhR: vec("text_encoder.lstm.weight_hh_l0_reverse"),
            bihR: vec("text_encoder.lstm.bias_ih_l0_reverse"),
            bhhR: vec("text_encoder.lstm.bias_hh_l0_reverse"), inDim: 512, hidden: 256)
    }
}

// ─── Small CPU math helpers (acoustic front-end is tiny, run once) ────

enum KokoroMath {
    /// LayerNorm over a single vector in-place.
    static func layerNorm(_ x: inout [Float], weight: [Float], bias: [Float], eps: Float) {
        let n = x.count
        var mean: Float = 0
        for v in x { mean += v }
        mean /= Float(n)
        var variance: Float = 0
        for v in x { let d = v - mean; variance += d * d }
        variance /= Float(n)
        let inv = 1.0 / (variance + eps).squareRoot()
        for i in 0 ..< n { x[i] = (x[i] - mean) * inv * weight[i] + bias[i] }
    }

    /// Batched row-wise linear `[n, inDim] · Wᵀ (+b) → [n, outDim]`.
    /// `weight` is `[outDim, inDim]` row-major (PyTorch Linear).
    static func linear(
        _ rows: [[Float]], weight: [Float], bias: [Float], outDim: Int, inDim: Int
    ) -> [[Float]] {
        let n = rows.count
        nonisolated(unsafe) var out = [[Float]](
            repeating: [Float](repeating: 0, count: outDim), count: n)
        nonisolated(unsafe) let w = weight
        nonisolated(unsafe) let b = bias
        DispatchQueue.concurrentPerform(iterations: n) { r in
            let x = rows[r]
            var y = [Float](repeating: 0, count: outDim)
            for o in 0 ..< outDim {
                var acc: Float = b.isEmpty ? 0 : b[o]
                let base = o * inDim
                for i in 0 ..< inDim { acc += w[base + i] * x[i] }
                y[o] = acc
            }
            out[r] = y
        }
        return out
    }

    /// Tanh-approximation GELU in-place (the BERT/ALBERT variant).
    static func gelu(_ x: inout [Float]) {
        let c: Float = 0.7978845608028654  // sqrt(2/pi)
        for i in x.indices {
            let v = x[i]
            x[i] = 0.5 * v * (1 + tanhf(c * (v + 0.044715 * v * v * v)))
        }
    }

    /// LeakyReLU in-place.
    static func leakyReLU(_ x: inout [[Float]], slope: Float = 0.2) {
        for i in x.indices { for j in x[i].indices where x[i][j] < 0 { x[i][j] *= slope } }
    }

    /// LayerNorm1d over the channel dim of a `[L, C]` sequence (StyleTTS2's
    /// `LayerNorm` that transposes channels-last, normalizes, transposes back).
    static func channelLayerNorm(_ x: inout [[Float]], gamma: [Float], beta: [Float]) {
        for i in x.indices { layerNorm(&x[i], weight: gamma, bias: beta, eps: 1e-5) }
    }

    /// One LSTM direction over `seq` `[L, inDim]`. `wih [4H,inDim]`, `whh
    /// [4H,H]`, gate order i,f,g,o (PyTorch). Returns `[L, H]`.
    static func lstmDir(
        _ seq: [[Float]], wih: [Float], whh: [Float], bih: [Float], bhh: [Float],
        inDim: Int, hidden H: Int, reverse: Bool
    ) -> [[Float]] {
        let L = seq.count
        var out = [[Float]](repeating: [Float](repeating: 0, count: H), count: L)
        var h = [Float](repeating: 0, count: H)
        var c = [Float](repeating: 0, count: H)
        for step in 0 ..< L {
            let t = reverse ? L - 1 - step : step
            let x = seq[t]
            var g = [Float](repeating: 0, count: 4 * H)
            for o in 0 ..< 4 * H {
                var acc = bih[o] + bhh[o]
                let ib = o * inDim
                for k in 0 ..< inDim { acc += wih[ib + k] * x[k] }
                let hb = o * H
                for k in 0 ..< H { acc += whh[hb + k] * h[k] }
                g[o] = acc
            }
            for j in 0 ..< H {
                let ig = 1 / (1 + expf(-g[j]))
                let fg = 1 / (1 + expf(-g[H + j]))
                let gg = tanhf(g[2 * H + j])
                let og = 1 / (1 + expf(-g[3 * H + j]))
                c[j] = fg * c[j] + ig * gg
                h[j] = og * tanhf(c[j])
            }
            out[t] = h
        }
        return out
    }

    /// Bidirectional LSTM: forward ⊕ reverse → `[L, 2H]`.
    static func biLSTM(
        _ seq: [[Float]], wihF: [Float], whhF: [Float], bihF: [Float], bhhF: [Float],
        wihR: [Float], whhR: [Float], bihR: [Float], bhhR: [Float], inDim: Int, hidden H: Int
    ) -> [[Float]] {
        let f = lstmDir(seq, wih: wihF, whh: whhF, bih: bihF, bhh: bhhF, inDim: inDim, hidden: H, reverse: false)
        let r = lstmDir(seq, wih: wihR, whh: whhR, bih: bihR, bhh: bhhR, inDim: inDim, hidden: H, reverse: true)
        return (0 ..< seq.count).map { f[$0] + r[$0] }
    }

    /// Bidirectional multi-head self-attention. `q/k/v` are `[L, heads·dim]`.
    static func mha(q: [[Float]], k: [[Float]], v: [[Float]], heads: Int, dim: Int) -> [[Float]] {
        let L = q.count
        let hd = dim / heads
        let scale = 1.0 / Float(hd).squareRoot()
        nonisolated(unsafe) var out = [[Float]](
            repeating: [Float](repeating: 0, count: dim), count: L)
        nonisolated(unsafe) let qq = q, kk = k, vv = v
        DispatchQueue.concurrentPerform(iterations: heads) { h in
            let base = h * hd
            for i in 0 ..< L {
                var scores = [Float](repeating: 0, count: L)
                var mx = -Float.greatestFiniteMagnitude
                for j in 0 ..< L {
                    var d: Float = 0
                    for c in 0 ..< hd { d += qq[i][base + c] * kk[j][base + c] }
                    d *= scale
                    scores[j] = d
                    if d > mx { mx = d }
                }
                var sum: Float = 0
                for j in 0 ..< L { scores[j] = expf(scores[j] - mx); sum += scores[j] }
                let invSum = 1.0 / sum
                for c in 0 ..< hd {
                    var acc: Float = 0
                    for j in 0 ..< L { acc += scores[j] * vv[j][base + c] }
                    out[i][base + c] = acc * invSum
                }
            }
        }
        return out
    }
}
