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
public struct KokoroWeights {
    let bundle: SafeTensorsBundle

    public init(bundle: SafeTensorsBundle) { self.bundle = bundle }

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
        let cb = convBias(base)
        let (yf, ysh) = AudioMath.conv1d(
            x: xf, xShape: [1, cIn, L], weight: w, wShape: sh, bias: cb.isEmpty ? nil : cb,
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

// ─── Prosody predictor (durations + F0/N) ────────────────────────────

extension KokoroWeights {
    /// AdaIN1d: instance-norm `[L, C]` over time per channel, then
    /// StyleTTS2 affine `(1+γ)·norm + β`, with `[γ,β] = fc(style)`.
    func adaIN1d(_ x: [[Float]], style: [Float], fcBase: String, channels C: Int) -> [[Float]] {
        let gb = KokoroMath.linear([style], weight: vec("\(fcBase).weight"), bias: vec("\(fcBase).bias"),
            outDim: 2 * C, inDim: style.count)[0]
        let L = x.count
        var out = x
        for c in 0 ..< C {
            var mean: Float = 0
            for t in 0 ..< L { mean += x[t][c] }
            mean /= Float(L)
            var v: Float = 0
            for t in 0 ..< L { let d = x[t][c] - mean; v += d * d }
            let inv = 1 / (v / Float(L) + 1e-5).squareRoot()
            let g = gb[c], b = gb[C + c]
            for t in 0 ..< L { out[t][c] = (1 + g) * ((x[t][c] - mean) * inv) + b }
        }
        return out
    }

    /// AdainResBlk1d: `norm1→act→[pool]→conv1→norm2→act→conv2`, residual
    /// `(x + shortcut)/√2`. `pool` (weight-norm convT, when present)
    /// up-samples ×2; `conv1x1` adapts channels on the shortcut.
    func adainResBlk(
        _ x: [[Float]], style: [Float], base: String, inC: Int, outC: Int,
        upsample: Bool
    ) -> [[Float]] {
        var h = adaIN1d(x, style: style, fcBase: "\(base).norm1.fc", channels: inC)
        KokoroMath.leakyReLU(&h)
        if upsample { h = poolUp2(h, base: "\(base).pool", channels: inC) }
        h = conv1dSeq(h, base: "\(base).conv1", pad: 1)
        h = adaIN1d(h, style: style, fcBase: "\(base).norm2.fc", channels: outC)
        KokoroMath.leakyReLU(&h)
        h = conv1dSeq(h, base: "\(base).conv2", pad: 1)
        // Shortcut: optional ×2 upsample then optional 1×1 conv.
        var sc = x
        if upsample { sc = upsampleNearest2(sc) }
        if has("\(base).conv1x1.weight_v") { sc = conv1dSeq(sc, base: "\(base).conv1x1", pad: 0) }
        let invSqrt2: Float = 0.7071067811865476
        let T = min(h.count, sc.count)
        var out = [[Float]](repeating: [Float](repeating: 0, count: outC), count: T)
        for t in 0 ..< T { for c in 0 ..< outC { out[t][c] = (h[t][c] + sc[t][c]) * invSqrt2 } }
        return out
    }

    /// ×2 nearest-neighbour upsample of an `[L, C]` sequence.
    func upsampleNearest2(_ x: [[Float]]) -> [[Float]] {
        var out = [[Float]]()
        out.reserveCapacity(x.count * 2)
        for r in x { out.append(r); out.append(r) }
        return out
    }

    /// The `pool` up-sampler inside an `AdainResBlk1d` (StyleTTS2): a
    /// weight-norm **grouped** (depthwise) `ConvTranspose1d` — kernel 3,
    /// stride 2, padding 1, `groups = C` — followed by a left zero-pad of 1
    /// frame. Weight `[C, 1, 3]`. Output length is `2·L` (convT gives
    /// `2L-1`, the left-pad makes it `2L`), matching the nearest-×2 shortcut.
    func poolUp2(_ x: [[Float]], base: String, channels C: Int) -> [[Float]] {
        let g = vec("\(base).weight_g"), v = vec("\(base).weight_v")
        let sh = shape("\(base).weight_v")  // [C, 1, 3]
        let w = WeightNorm.effectiveWeight(g: g, v: v, shape: sh, exceptDim: 0)
        let b = convBias(base)
        let L = x.count
        // [L, C] → channels-first flat [C, L].
        var xf = [Float](repeating: 0, count: C * L)
        for t in 0 ..< L { for c in 0 ..< C { xf[c * L + t] = x[t][c] } }
        let (yf, ysh) = AudioMath.convTransposed1d(
            x: xf, xShape: [1, C, L], weight: w, wShape: sh,
            bias: b.isEmpty ? nil : b, stride: 2, padding: 1, dilation: 1,
            outputPadding: 0, groups: C)
        let lOut = ysh[2]  // 2L - 1
        // Channels-first → [time, C], prepending one zero frame (left-pad 1).
        var out = [[Float]](repeating: [Float](repeating: 0, count: C), count: lOut + 1)
        for c in 0 ..< C { for t in 0 ..< lOut { out[t + 1][c] = yf[c * lOut + t] } }
        return out
    }

    /// Duration encoder: 3×(BiLSTM[640→512] over `[feat ⊕ style]`, then
    /// AdaLayerNorm). Returns `[L, 512]`.
    func durationEncoder(_ feat: [[Float]], style: [Float]) -> [[Float]] {
        var x = feat
        for pair in 0 ..< 3 {
            let li = pair * 2
            let withStyle = x.map { $0 + style }  // [L, 512+128=640]
            x = KokoroMath.biLSTM(
                withStyle,
                wihF: vec("predictor.text_encoder.lstms.\(li).weight_ih_l0"),
                whhF: vec("predictor.text_encoder.lstms.\(li).weight_hh_l0"),
                bihF: vec("predictor.text_encoder.lstms.\(li).bias_ih_l0"),
                bhhF: vec("predictor.text_encoder.lstms.\(li).bias_hh_l0"),
                wihR: vec("predictor.text_encoder.lstms.\(li).weight_ih_l0_reverse"),
                whhR: vec("predictor.text_encoder.lstms.\(li).weight_hh_l0_reverse"),
                bihR: vec("predictor.text_encoder.lstms.\(li).bias_ih_l0_reverse"),
                bhhR: vec("predictor.text_encoder.lstms.\(li).bias_hh_l0_reverse"),
                inDim: 640, hidden: 256)
            // AdaLayerNorm: channel-LayerNorm then style affine (1+γ)·+β.
            let fc = "predictor.text_encoder.lstms.\(li + 1).fc"
            let gb = KokoroMath.linear([style], weight: vec("\(fc).weight"), bias: vec("\(fc).bias"),
                outDim: 1024, inDim: 128)[0]
            for i in x.indices {
                KokoroMath.layerNorm(&x[i], weight: [Float](repeating: 1, count: 512),
                    bias: [Float](repeating: 0, count: 512), eps: 1e-5)
                for c in 0 ..< 512 { x[i][c] = (1 + gb[c]) * x[i][c] + gb[512 + c] }
            }
        }
        return x
    }

    /// Predict per-token integer durations + the aligned F0/N curves.
    /// `dEn` is bert_encoder output `[L,512]`; `style` the 128-dim voice
    /// half used by the predictor. Returns `(durations, F0[T], N[T])` and the
    /// `[L,T]` expansion is applied by the caller to the acoustic features.
    func predict(dEn: [[Float]], style: [Float], speed: Float) -> (dur: [Int], f0: [Float], n: [Float], dAligned: [[Float]]) {
        let d = durationEncoder(dEn, style: style)
        let x = KokoroMath.biLSTM(
            d.map { $0 + style },
            wihF: vec("predictor.lstm.weight_ih_l0"), whhF: vec("predictor.lstm.weight_hh_l0"),
            bihF: vec("predictor.lstm.bias_ih_l0"), bhhF: vec("predictor.lstm.bias_hh_l0"),
            wihR: vec("predictor.lstm.weight_ih_l0_reverse"), whhR: vec("predictor.lstm.weight_hh_l0_reverse"),
            bihR: vec("predictor.lstm.bias_ih_l0_reverse"), bhhR: vec("predictor.lstm.bias_hh_l0_reverse"),
            inDim: 640, hidden: 256)
        let dp = KokoroMath.linear(x, weight: vec("predictor.duration_proj.linear_layer.weight"),
            bias: vec("predictor.duration_proj.linear_layer.bias"), outDim: 50, inDim: 512)
        // Duration per token: round(Σ sigmoid(logits) / speed), clipped to [1,100].
        let maxFramesPerPhoneme = 100
        var dur = [Int]()
        for row in dp {
            var s: Float = 0
            for v in row { s += 1 / (1 + expf(-v)) }
            let r = Int((s / max(speed, 0.1)).rounded())
            dur.append(min(max(r, 1), maxFramesPerPhoneme))
        }
        // Length-regulate the duration-encoder output → [T, 512].
        var dAligned = [[Float]]()
        for (i, r) in d.enumerated() { for _ in 0 ..< dur[i] { dAligned.append(r) } }
        // Shared LSTM → F0/N AdaIN res-block stacks.
        let sh = KokoroMath.biLSTM(
            dAligned.map { $0 + style },
            wihF: vec("predictor.shared.weight_ih_l0"), whhF: vec("predictor.shared.weight_hh_l0"),
            bihF: vec("predictor.shared.bias_ih_l0"), bhhF: vec("predictor.shared.bias_hh_l0"),
            wihR: vec("predictor.shared.weight_ih_l0_reverse"), whhR: vec("predictor.shared.weight_hh_l0_reverse"),
            bihR: vec("predictor.shared.bias_ih_l0_reverse"), bhhR: vec("predictor.shared.bias_hh_l0_reverse"),
            inDim: 640, hidden: 256)
        func curve(_ tag: String) -> [Float] {
            var h = adainResBlk(sh, style: style, base: "predictor.\(tag).0", inC: 512, outC: 512, upsample: false)
            h = adainResBlk(h, style: style, base: "predictor.\(tag).1", inC: 512, outC: 256, upsample: true)
            h = adainResBlk(h, style: style, base: "predictor.\(tag).2", inC: 256, outC: 256, upsample: false)
            let projW = vec("predictor.\(tag)_proj.weight")  // plain [1,256,1]
            let pb = vec("predictor.\(tag)_proj.bias")
            return h.map { row -> Float in
                var acc: Float = pb.isEmpty ? 0 : pb[0]
                for c in 0 ..< 256 { acc += projW[c] * row[c] }
                return acc
            }
        }
        return (dur, curve("F0"), curve("N"), dAligned)
    }
}

// ─── Decoder + iSTFTNet generator ────────────────────────────────────

extension KokoroWeights {
    /// Snake-activated, AdaIN-conditioned generator res-block: for each of
    /// the 3 dilations `d`, `x += convs2(snake(α2, convs1(snake(α1, adain1(x)))))`
    /// — the StyleTTS2 iSTFTNet `AdaINResBlock1`.
    func genResBlock(_ x: [[Float]], style: [Float], base: String, channels C: Int) -> [[Float]] {
        let dils = [1, 3, 5]
        var h = x
        for (j, dil) in dils.enumerated() {
            // `same` padding derived from the conv's own kernel size: convs1
            // is dilated (pad = dil·(k-1)/2), convs2 is dilation-1 (pad =
            // (k-1)/2). Kernel size differs per res-block (3 / 7 / 11), so it
            // must come from the weight shape, not be hard-coded.
            let k1 = shape("\(base).convs1.\(j).weight_v")[2]
            let k2 = shape("\(base).convs2.\(j).weight_v")[2]
            var a = adaIN1d(h, style: style, fcBase: "\(base).adain1.\(j).fc", channels: C)
            snake(&a, base: "\(base).alpha1.\(j)", channels: C)
            a = conv1dSeq(a, base: "\(base).convs1.\(j)", pad: dil * (k1 - 1) / 2, dilation: dil)
            a = adaIN1d(a, style: style, fcBase: "\(base).adain2.\(j).fc", channels: C)
            snake(&a, base: "\(base).alpha2.\(j)", channels: C)
            a = conv1dSeq(a, base: "\(base).convs2.\(j)", pad: (k2 - 1) / 2)
            let T = min(h.count, a.count)
            for t in 0 ..< T { for c in 0 ..< C { h[t][c] += a[t][c] } }
        }
        return h
    }

    /// Snake activation `x + (1/α)·sin²(αx)` in-place; `α` stored `[1,C,1]`.
    func snake(_ x: inout [[Float]], base: String, channels C: Int) {
        let alpha = vec(base)  // [C]
        for t in x.indices {
            for c in 0 ..< C {
                let a = alpha[c]
                let s = sinf(a * x[t][c])
                x[t][c] += (1 / (a + 1e-9)) * s * s
            }
        }
    }

    /// Weight-norm `ConvTranspose1d` up-sample. Weight `[Cin, Cout, K]`,
    /// `stride`, symmetric `padding` (cropped from both ends). Weight-norm is
    /// over dim 0 (`g` is `[Cin,1,1]`).
    func convTranspose1d(_ x: [[Float]], base: String, stride: Int, padding: Int) -> [[Float]] {
        let g = vec("\(base).weight_g"), v = vec("\(base).weight_v")
        let sh = shape("\(base).weight_v")  // [Cin, Cout, K]
        let (cIn, cOut) = (sh[0], sh[1])
        let w = WeightNorm.effectiveWeight(g: g, v: v, shape: sh, exceptDim: 0)
        let b = convBias(base)
        let L = x.count
        var xf = [Float](repeating: 0, count: cIn * L)
        for t in 0 ..< L { for c in 0 ..< cIn { xf[c * L + t] = x[t][c] } }
        let (yf, ysh) = AudioMath.convTransposed1d(
            x: xf, xShape: [1, cIn, L], weight: w, wShape: sh,
            bias: b.isEmpty ? nil : b, stride: stride, padding: padding,
            dilation: 1, outputPadding: 0, groups: 1)
        let lOut = ysh[2]
        var out = [[Float]](repeating: [Float](repeating: 0, count: cOut), count: lOut)
        for c in 0 ..< cOut { for t in 0 ..< lOut { out[t][c] = yf[c * lOut + t] } }
        return out
    }

    /// iSTFTNet generator: NSF harmonic source + 2 up-sample stages (×10, ×6)
    /// with Snake-AdaIN res-blocks → conv_post (22ch) → mag/phase → [re,im].
    /// Returns the `[nFrames, nFreq]` re/im planes for `KokoroVocoder`.
    ///
    /// `f0` is the predictor F0 curve at the (2T) decoder rate; the source is
    /// built at full sample resolution then down-projected per stage by the
    /// `noise_convs`. Leaky-ReLU slopes match the reference (0.1 in the loop,
    /// 0.01 at the tail).
    func generate(_ x0: [[Float]], style: [Float], f0: [Float], vocoder: KokoroVocoder) -> (re: [[Float]], im: [[Float]]) {
        var x = x0
        let upRates = [10, 6], upPads = [5, 3], upChans = [256, 128]
        let nFFT = vocoder.nFFT, hop = vocoder.hopLength
        let nFreq = nFFT / 2 + 1  // 11
        let totalUp = (upRates[0] * upRates[1]) * hop  // 300
        // Harmonic NSF source spectrogram at the iSTFT framing: [framesFull, 22].
        let har = nsfSource(f0Curve: f0, totalUpsample: totalUp, nFFT: nFFT, hop: hop)
        let numUp = upRates.count
        for stage in 0 ..< numUp {
            KokoroMath.leakyReLU(&x, slope: 0.1)
            let C = upChans[stage]
            // Source branch: noise_convs (strided plain conv) → noise_res block.
            // stage 0: kernel 2·stride_f0, stride stride_f0=6, pad (stride_f0+1)/2.
            // stage 1 (last): kernel 1, stride 1, pad 0.
            // Kernel/stride come from the stored weight; `stride_f0` is the
            // product of the *later* upsample rates (6 for stage 0, 1 for the
            // last stage), and padding is (stride_f0 + 1) / 2.
            let strideF0 = (stage == 0) ? upRates[1] : 1
            let padSrc = (stage == 0) ? (strideF0 + 1) / 2 : 0
            var src = conv1dSeqPlain(har, base: "decoder.generator.noise_convs.\(stage)",
                outC: C, pad: padSrc, stride: strideF0)
            src = genResBlock(src, style: style, base: "decoder.generator.noise_res.\(stage)", channels: C)
            // Main branch: transposed-conv up-sample (+ left zero-pad on the
            // last stage, the reference `reflection_pad((1,0))`), then add source.
            x = convTranspose1d(x, base: "decoder.generator.ups.\(stage)",
                stride: upRates[stage], padding: upPads[stage])
            if stage == numUp - 1 {
                x.insert([Float](repeating: 0, count: C), at: 0)  // left-pad 1 frame
            }
            let T = min(x.count, src.count)
            x = Array(x.prefix(T))
            for t in 0 ..< T { for c in 0 ..< C { x[t][c] += src[t][c] } }
            // 3 res-blocks (kernels 3/7/11) averaged.
            var acc = [[Float]](repeating: [Float](repeating: 0, count: C), count: x.count)
            for j in 0 ..< 3 {
                let rb = genResBlock(x, style: style, base: "decoder.generator.resblocks.\(stage * 3 + j)", channels: C)
                let TT = min(acc.count, rb.count)
                for t in 0 ..< TT { for c in 0 ..< C { acc[t][c] += rb[t][c] / 3 } }
            }
            x = acc
        }
        KokoroMath.leakyReLU(&x, slope: 0.01)
        let post = conv1dSeq(x, base: "decoder.generator.conv_post", pad: 3)  // → 22ch
        // conv_post → magnitude/phase: spec = exp(x[:nFreq]), phase = sin(x[nFreq:]),
        // then the complex spectrum is mag·e^{iφ} → (re, im) for the iSTFT.
        var re = [[Float]](repeating: [Float](repeating: 0, count: nFreq), count: post.count)
        var im = re
        for t in post.indices {
            for k in 0 ..< nFreq {
                let mag = expf(post[t][k])
                let ph = sinf(post[t][nFreq + k])
                re[t][k] = mag * cosf(ph)
                im[t][k] = mag * sinf(ph)
            }
        }
        return (re, im)
    }

    /// NSF harmonic source spectrogram `[frames, 2·nFreq]` (magnitude ⊕ phase)
    /// from the F0 curve — the StyleTTS2 `SourceModuleHnNSF` + STFT. Builds a
    /// 9-harmonic sine excitation at the full sample rate (anti-aliased via the
    /// reference's down/cumsum/up-interpolate dance), merges it through
    /// `m_source.l_linear` + tanh, and STFTs it to the iSTFT framing. Noise is
    /// zeroed (deterministic); the sine excitation carries the voiced energy.
    func nsfSource(f0Curve: [Float], totalUpsample: Int, nFFT: Int, hop: Int) -> [[Float]] {
        let SR: Float = 24_000
        let harmonics = 9
        let sineAmp: Float = 0.1
        let voicedThresh: Float = 10
        let F = f0Curve.count
        let N = F * totalUpsample
        guard F > 0 else { return [] }
        let lW = vec("decoder.generator.m_source.l_linear.weight")  // [1,9] → 9
        let lB = vec("decoder.generator.m_source.l_linear.bias")  // [1]
        // Per-frame phase for each harmonic: cumsum of (f0·h/SR mod 1) over the
        // F frames, ×2π, then ×totalUpsample (the reference scales before the
        // up-interpolate). phaseScaled[m][h].
        var phaseScaled = [[Float]](repeating: [Float](repeating: 0, count: harmonics), count: F)
        var running = [Float](repeating: 0, count: harmonics)
        let twoPi = 2 * Float.pi
        for m in 0 ..< F {
            for h in 0 ..< harmonics {
                let rad = (f0Curve[m] * Float(h + 1) / SR).truncatingRemainder(dividingBy: 1)
                running[h] += rad
                phaseScaled[m][h] = running[h] * twoPi * Float(totalUpsample)
            }
        }
        // Up-sample phase to N samples (linear interp, align_corners=None):
        // xu = n·(F/N) + 0.5·(F/N) − 0.5; negative xLow wraps to F−1 (mlx).
        // Parallelised over a raw mutable buffer (no `[Float]` CoW race).
        var merged = [Float](repeating: 0, count: N)
        let scaleFN = Float(F) / Float(N)
        nonisolated(unsafe) let psU = phaseScaled
        nonisolated(unsafe) let f0U = f0Curve
        nonisolated(unsafe) let lWU = lW, lBU = lB
        merged.withUnsafeMutableBufferPointer { buf in
            nonisolated(unsafe) let ptr = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: N) { n in
                let xu = Float(n) * scaleFN + 0.5 * scaleFN - 0.5
                let xLowRaw = Int(floorf(xu))
                let frac = xu - Float(xLowRaw)
                let lo = xLowRaw < 0 ? F - 1 : min(xLowRaw, F - 1)
                let hi = min(xLowRaw + 1, F - 1)
                let block = min(n / totalUpsample, F - 1)
                let uv: Float = f0U[block] > voicedThresh ? 1 : 0
                var sum: Float = lBU.isEmpty ? 0 : lBU[0]
                for h in 0 ..< harmonics {
                    let ph = psU[lo][h] * (1 - frac) + psU[hi][h] * frac
                    let sine = sinf(ph) * sineAmp * uv
                    sum += lWU[h] * sine
                }
                ptr[n] = tanhf(sum)
            }
        }
        // STFT(merged) → [frames, nFreq] magnitude + phase, concatenated.
        let (mag, phase) = KokoroMath.forwardSTFT(merged, nFFT: nFFT, hop: hop)
        let frames = mag.count
        var out = [[Float]](repeating: [Float](repeating: 0, count: 2 * (nFFT / 2 + 1)), count: frames)
        let nFreq = nFFT / 2 + 1
        for f in 0 ..< frames {
            for k in 0 ..< nFreq { out[f][k] = mag[f][k]; out[f][nFreq + k] = phase[f][k] }
        }
        return out
    }

    /// Plain (non-weight-norm) `Conv1d` over `[L, Cin]` → `[L, Cout]` with
    /// `stride`. `noise_convs` weights are PyTorch `[Cout, Cin, K]` row-major.
    func conv1dSeqPlain(_ x: [[Float]], base: String, outC: Int, pad: Int, stride: Int = 1) -> [[Float]] {
        let w = vec("\(base).weight")
        let sh = shape("\(base).weight")
        let cIn = sh[1]
        let L = x.count
        var xf = [Float](repeating: 0, count: cIn * L)
        for t in 0 ..< L { for c in 0 ..< cIn { xf[c * L + t] = x[t][c] } }
        let (yf, ysh) = AudioMath.conv1d(
            x: xf, xShape: [1, cIn, L], weight: w, wShape: sh, bias: vec("\(base).bias"),
            stride: stride, padding: pad, dilation: 1, groups: 1)
        let lOut = ysh[2]
        var out = [[Float]](repeating: [Float](repeating: 0, count: outC), count: lOut)
        for c in 0 ..< outC { for t in 0 ..< lOut { out[t][c] = yf[c * lOut + t] } }
        return out
    }

    /// Decoder: concat(asr, F0, N) → encode → concat(asr_res, F0, N) → decode
    /// ×4 (last up-samples ×2) → generator → re/im planes. The F0/N curves come
    /// in at 2T (predictor upsamples once); the decoder's F0_conv/N_conv are
    /// stride-2 depthwise convs that bring them back to the asr rate T.
    func decode(asr: [[Float]], f0: [Float], n: [Float], style: [Float], vocoder: KokoroVocoder) -> (re: [[Float]], im: [[Float]]) {
        let T = asr.count
        // F0/N down-conv: weight-norm Conv1d(1→1, k3, stride 2, pad 1) over the
        // 2T curve → T frames. Reference `ConvWeighted(1,1,3,stride=2,padding=1)`.
        func single(_ curve: [Float], base: String) -> [Float] {
            let (w, _) = convWeight(base)  // [1,1,3]
            let b = convBias(base)
            let L = curve.count
            let lOut = (L + 2 * 1 - 3) / 2 + 1
            return (0 ..< lOut).map { t -> Float in
                var acc: Float = b.isEmpty ? 0 : b[0]
                let start = t * 2 - 1  // stride 2, pad 1
                for k in 0 ..< 3 {
                    let s = start + k
                    if s >= 0 && s < L { acc += w[k] * curve[s] }
                }
                return acc
            }
        }
        let f0d = single(f0, base: "decoder.F0_conv")  // length T
        let nd = single(n, base: "decoder.N_conv")  // length T
        // encode: AdainResBlk on [asr ⊕ F0 ⊕ N] (514 → 1024).
        var x = (0 ..< T).map { asr[$0] + [f0d[min($0, f0d.count - 1)], nd[min($0, nd.count - 1)]] }
        x = adainResBlk(x, style: style, base: "decoder.encode", inC: 514, outC: 1024, upsample: false)
        let asrRes = conv1dSeq(asr, base: "decoder.asr_res.0", pad: 0)  // 512 → 64
        for blk in 0 ..< 4 {
            let cat = (0 ..< x.count).map { t -> [Float] in
                let i = min(T - 1, t)
                return x[t] + asrRes[i] + [f0d[i], nd[i]]  // 1024+64+1+1 = 1090
            }
            let up = (blk == 3)
            let outC = (blk == 3) ? 512 : 1024
            x = adainResBlk(cat, style: style, base: "decoder.decode.\(blk)", inC: 1090, outC: outC, upsample: up)
        }
        // Generator runs on the original (2T) F0 curve, not the down-sampled one.
        return generate(x, style: style, f0: f0, vocoder: vocoder)
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

    /// Exact (erf-based) GELU in-place — ALBERT's `nn.GELU()` default. The
    /// tanh approximation drifts enough over 12 shared layers to matter, so
    /// we match the reference exactly: `0.5·x·(1 + erf(x/√2))`.
    static func gelu(_ x: inout [Float]) {
        let invSqrt2: Float = 0.7071067811865476
        for i in x.indices {
            let v = x[i]
            x[i] = 0.5 * v * (1 + erff(v * invSqrt2))
        }
    }

    /// LeakyReLU in-place.
    static func leakyReLU(_ x: inout [[Float]], slope: Float = 0.2) {
        for i in x.indices { for j in x[i].indices where x[i][j] < 0 { x[i][j] *= slope } }
    }

    /// Real STFT matching `mlx_audio.utils.stft`: center reflect-pad by
    /// `nFFT/2`, a symmetric Hann window (`denom = nFFT−1`), `rfft` per frame.
    /// Returns `(magnitude, phase)` each `[frames][nFreq]`, `nFreq = nFFT/2+1`,
    /// `frames = 1 + (paddedLen − nFFT)/hop`. The FFT is a naïve DFT — fine for
    /// Kokoro's tiny `nFFT = 20`.
    static func forwardSTFT(_ sig: [Float], nFFT: Int, hop: Int) -> (mag: [[Float]], phase: [[Float]]) {
        let pad = nFFT / 2
        let n = sig.count
        // Reflect pad: prefix = sig[1...pad] reversed, suffix = sig[n-pad-1 ... n-2] reversed.
        var padded = [Float]()
        padded.reserveCapacity(n + 2 * pad)
        if pad > 0 { for i in stride(from: pad, through: 1, by: -1) { padded.append(sig[i]) } }
        padded.append(contentsOf: sig)
        if pad > 0 { for i in stride(from: n - 2, through: n - 1 - pad, by: -1) { padded.append(sig[i]) } }
        // Symmetric Hann window (periodic = false): 0.5·(1 − cos(2πk/(nFFT−1))).
        var win = [Float](repeating: 0, count: nFFT)
        let denom = Float(max(nFFT - 1, 1))
        for k in 0 ..< nFFT { win[k] = 0.5 * (1 - cosf(2 * Float.pi * Float(k) / denom)) }
        let nFreq = nFFT / 2 + 1
        let frames = 1 + (padded.count - nFFT) / hop
        // Precompute DFT twiddles.
        var cosT = [Float](repeating: 0, count: nFreq * nFFT)
        var sinT = [Float](repeating: 0, count: nFreq * nFFT)
        for kk in 0 ..< nFreq {
            for nn in 0 ..< nFFT {
                let ang = -2 * Float.pi * Float(kk) * Float(nn) / Float(nFFT)
                cosT[kk * nFFT + nn] = cosf(ang)
                sinT[kk * nFFT + nn] = sinf(ang)
            }
        }
        var mag = [[Float]](repeating: [Float](repeating: 0, count: nFreq), count: frames)
        var phase = mag
        // Serial: nFFT = 20 → ~frames·11·20 mults, microseconds. (A parallel
        // version here would need careful buffer-pointer handling to avoid the
        // CoW data race that concurrent `[[Float]]` element writes trigger.)
        for f in 0 ..< frames {
            let start = f * hop
            for kk in 0 ..< nFreq {
                var re: Float = 0, im: Float = 0
                let base = kk * nFFT
                for nn in 0 ..< nFFT {
                    let v = padded[start + nn] * win[nn]
                    re += v * cosT[base + nn]
                    im += v * sinT[base + nn]
                }
                mag[f][kk] = (re * re + im * im).squareRoot()
                phase[f][kk] = atan2f(im, re)
            }
        }
        return (mag, phase)
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
        // Parallelize over query positions — each iteration writes a distinct
        // `out[i]` row (all heads for that row), so no inner-array data race.
        nonisolated(unsafe) var out = [[Float]](
            repeating: [Float](repeating: 0, count: dim), count: L)
        nonisolated(unsafe) let qq = q, kk = k, vv = v
        DispatchQueue.concurrentPerform(iterations: L) { i in
            var row = [Float](repeating: 0, count: dim)
            for h in 0 ..< heads {
                let base = h * hd
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
                    row[base + c] = acc * invSum
                }
            }
            out[i] = row
        }
        return out
    }
}

