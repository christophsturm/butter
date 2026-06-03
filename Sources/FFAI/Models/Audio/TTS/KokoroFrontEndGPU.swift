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
// Kokoro / StyleTTS2 front-end — GPU-resident port of PLBERT, the acoustic
// text_encoder, and the prosody predictor. These run at the token rate
// (L ≤ ~70) so they were never the cost bottleneck (the decoder + generator
// were — see `KokoroAcousticGPU`); this is the last stage for full residency.
//
// Layout: token-rate features are row-major `[L, C]` (one row per token), the
// layout `Ops.gemm` / `Ops.layerNorm` / `Ops.lstm` consume. The conv layers
// inside the acoustic text_encoder transpose to channels-first `[1, C, L]`.
// Embedding lookups stay on the CPU (a tiny gather — the idiomatic split,
// cf. Paligemma's `CpuEmbedding`); everything matrix-heavy runs on the GPU.
//
// Each stage returns `[[Float]]` so it is a drop-in replacement for its CPU
// twin and is parity-tested in isolation (`KokoroFrontEndGPUParityTests`).

import Foundation
import Metal

extension KokoroWeights {
    /// `weight·xᵀ (+ bias)` on the GPU. `x` is `[nRows, inDim]`, weight key is
    /// PyTorch `[outDim, inDim]`; returns `[nRows, outDim]`.
    func linearGPU(
        _ x: Tensor, wName: String, bName: String?, nRows: Int, inDim: Int, outDim: Int,
        gw: KokoroGPUWeights, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let wt = gw.vecT(wName, shape: [outDim, inDim])
        var y = Ops.gemm(weight: wt, input: x, nRows: nRows, on: cmd)  // [nRows, outDim]
        if let bName { y = Ops.add(y, gw.biasTile(bName, rows: nRows, dim: outDim), on: cmd) }
        return y
    }

    /// GPU PLBERT (ALBERT) encode: phoneme ids → contextual `[L, 768]`. The
    /// embedding sum + 128-dim LayerNorm + 128→768 map run on the CPU (a tiny
    /// gather + small matmul); the 12 shared transformer layers run on the GPU
    /// in a single command buffer.
    func plbertEncodeGPU(_ ids: [Int], gw: KokoroGPUWeights, device: Device = .shared) -> [[Float]] {
        let L = ids.count
        let hidden = 768, embDim = 128, heads = 12, intermediate = 2048, numLayers = 12
        let hd = hidden / heads
        let scale = 1.0 / Float(hd).squareRoot()

        // CPU embedding: word + position + token_type[0] → LayerNorm(128) → map 128→768.
        let wordE = vec("bert.embeddings.word_embeddings.weight")
        let posE = vec("bert.embeddings.position_embeddings.weight")
        let ttE = vec("bert.embeddings.token_type_embeddings.weight")
        let lnW = vec("bert.embeddings.LayerNorm.weight"), lnB = vec("bert.embeddings.LayerNorm.bias")
        var emb = [[Float]](repeating: [Float](repeating: 0, count: embDim), count: L)
        for i in 0 ..< L {
            let wb = ids[i] * embDim, pb = i * embDim
            for d in 0 ..< embDim { emb[i][d] = wordE[wb + d] + posE[pb + d] + ttE[d] }
            KokoroMath.layerNorm(&emb[i], weight: lnW, bias: lnB, eps: 1e-12)
        }
        let mapW = vec("bert.encoder.embedding_hidden_mapping_in.weight")
        let mapB = vec("bert.encoder.embedding_hidden_mapping_in.bias")
        let h0 = KokoroMath.linear(emb, weight: mapW, bias: mapB, outDim: hidden, inDim: embDim)

        var h = Tensor.empty(shape: [L, hidden], dtype: .f32, device: device)
        h.copyIn(from: h0.flatMap { $0 })

        let p = "bert.encoder.albert_layer_groups.0.albert_layers.0"
        let aLnW = gw.vecT("\(p).attention.LayerNorm.weight", shape: [hidden])
        let aLnB = gw.vecT("\(p).attention.LayerNorm.bias", shape: [hidden])
        let flLnW = gw.vecT("\(p).full_layer_layer_norm.weight", shape: [hidden])
        let flLnB = gw.vecT("\(p).full_layer_layer_norm.bias", shape: [hidden])
        let cmd = device.makeCommandBuffer()
        for _ in 0 ..< numLayers {
            // Multi-head self-attention.
            let q = linearGPU(h, wName: "\(p).attention.query.weight", bName: "\(p).attention.query.bias",
                nRows: L, inDim: hidden, outDim: hidden, gw: gw, on: cmd)
            let k = linearGPU(h, wName: "\(p).attention.key.weight", bName: "\(p).attention.key.bias",
                nRows: L, inDim: hidden, outDim: hidden, gw: gw, on: cmd)
            let v = linearGPU(h, wName: "\(p).attention.value.weight", bName: "\(p).attention.value.bias",
                nRows: L, inDim: hidden, outDim: hidden, gw: gw, on: cmd)
            let qh = q.reshaped(to: [L, heads, hd])  // token-major
            let kh = Ops.transposeTokensToHeads(k, nTokens: L, nHeads: heads, headDim: hd, on: cmd)
            let vh = Ops.transposeTokensToHeads(v, nTokens: L, nHeads: heads, headDim: hd, on: cmd)
            let attn = Ops.sdpaBidirectional(
                q: qh, k: kh, v: vh, nQHeads: heads, nKVHeads: heads, headDim: hd,
                baseKV: 0, nQuery: L, kvStride: L, scale: scale, on: cmd
            ).reshaped(to: [L, hidden])
            var dense = linearGPU(attn, wName: "\(p).attention.dense.weight", bName: "\(p).attention.dense.bias",
                nRows: L, inDim: hidden, outDim: hidden, gw: gw, on: cmd)
            dense = Ops.add(dense, h, on: cmd)
            dense = Ops.layerNorm(dense, weight: aLnW, bias: aLnB, eps: 1e-12, nRows: L, rowSize: hidden, on: cmd)
            // FFN (exact GELU).
            var ff = linearGPU(dense, wName: "\(p).ffn.weight", bName: "\(p).ffn.bias",
                nRows: L, inDim: hidden, outDim: intermediate, gw: gw, on: cmd)
            ff = Ops.geluErf(ff, on: cmd)
            var ffOut = linearGPU(ff, wName: "\(p).ffn_output.weight", bName: "\(p).ffn_output.bias",
                nRows: L, inDim: intermediate, outDim: hidden, gw: gw, on: cmd)
            ffOut = Ops.add(ffOut, dense, on: cmd)
            ffOut = Ops.layerNorm(ffOut, weight: flLnW, bias: flLnB, eps: 1e-12, nRows: L, rowSize: hidden, on: cmd)
            h = ffOut
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        let flat = h.toFloatArray()
        return (0 ..< L).map { i in Array(flat[i * hidden ..< (i + 1) * hidden]) }
    }

    /// Plain 2-D transpose `[A, B] → [B, A]` via `transposeTokensToHeads` with
    /// `headDim = 1` (channels-last ↔ channels-first for the text_encoder convs).
    private func transpose2D(_ x: Tensor, _ a: Int, _ b: Int, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.transposeTokensToHeads(x, nTokens: a, nHeads: b, headDim: 1, on: cmd).reshaped(to: [b, a])
    }

    /// `bert_encoder` 768→512 on the GPU.
    func bertEncodeGPU(_ h: [[Float]], gw: KokoroGPUWeights, device: Device = .shared) -> [[Float]] {
        let L = h.count
        let hT = Tensor.empty(shape: [L, 768], dtype: .f32, device: device); hT.copyIn(from: h.flatMap { $0 })
        let cmd = device.makeCommandBuffer()
        let out = linearGPU(hT, wName: "bert_encoder.weight", bName: "bert_encoder.bias",
            nRows: L, inDim: 768, outDim: 512, gw: gw, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let flat = out.toFloatArray()
        return (0 ..< L).map { Array(flat[$0 * 512 ..< ($0 + 1) * 512]) }
    }

    /// GPU acoustic `text_encoder`: phoneme embedding (CPU gather) → 3× (weight-
    /// norm conv k5 + channel-LayerNorm + LeakyReLU) → BiLSTM → `[L, 512]`. The
    /// convs run channels-first `[512, L]`; LayerNorm/LeakyReLU run channels-last
    /// `[L, 512]`, so each layer transposes around the conv.
    func textEncodeGPU(_ ids: [Int], gw: KokoroGPUWeights, device: Device = .shared) -> [[Float]] {
        let L = ids.count
        let C = 512
        let embW = vec("text_encoder.embedding.weight")
        var emb = [Float](repeating: 0, count: L * C)
        for i in 0 ..< L { for d in 0 ..< C { emb[i * C + d] = embW[ids[i] * C + d] } }
        // Start channels-first [C, L].
        var hcf = [Float](repeating: 0, count: C * L)
        for i in 0 ..< L { for d in 0 ..< C { hcf[d * L + i] = emb[i * C + d] } }
        var x = Tensor.empty(shape: [C, L], dtype: .f32, device: device); x.copyIn(from: hcf)
        let cmd = device.makeCommandBuffer()
        for layer in 0 ..< 3 {
            // conv k5 pad2 (channels-first).
            let base = "text_encoder.cnn.\(layer).0"
            let k = shape("\(base).weight_v")[2]
            x = Ops.conv1dDilated(
                x, weight: gw.convW(base), bias: gw.convB(base, outC: C),
                batch: 1, inCh: C, inLen: L, outCh: C, k: k, stride: 1, pad: (k - 1) / 2, dilation: 1, on: cmd
            ).reshaped(to: [C, L])
            // → channels-last [L, C] for LayerNorm + LeakyReLU.
            var cl = transpose2D(x, C, L, on: cmd)  // [L, C]
            cl = Ops.layerNorm(cl,
                weight: gw.vecT("text_encoder.cnn.\(layer).1.gamma", shape: [C]),
                bias: gw.vecT("text_encoder.cnn.\(layer).1.beta", shape: [C]),
                eps: 1e-5, nRows: L, rowSize: C, on: cmd)
            cl = Ops.leakyReLU(cl, slope: 0.2, on: cmd)
            x = transpose2D(cl, L, C, on: cmd)  // back to [C, L]
        }
        // BiLSTM over channels-last [L, 512] — on the CPU (tiny; the GPU LSTM
        // kernel has a long-sequence NaN, see the spawned "Fix GPU LSTM" task).
        let seqT = transpose2D(x, C, L, on: cmd)  // [L, C]
        cmd.commit(); cmd.waitUntilCompleted()
        let seqFlat = seqT.toFloatArray()
        let seq = (0 ..< L).map { Array(seqFlat[$0 * C ..< ($0 + 1) * C]) }
        return KokoroMath.biLSTM(
            seq,
            wihF: vec("text_encoder.lstm.weight_ih_l0"), whhF: vec("text_encoder.lstm.weight_hh_l0"),
            bihF: vec("text_encoder.lstm.bias_ih_l0"), bhhF: vec("text_encoder.lstm.bias_hh_l0"),
            wihR: vec("text_encoder.lstm.weight_ih_l0_reverse"), whhR: vec("text_encoder.lstm.weight_hh_l0_reverse"),
            bihR: vec("text_encoder.lstm.bias_ih_l0_reverse"), bhhR: vec("text_encoder.lstm.bias_hh_l0_reverse"),
            inDim: 512, hidden: 256)
    }

    /// GPU prosody predictor: durations + aligned F0/N curves. Mirrors the CPU
    /// `predict`. The LSTMs, AdaLayerNorms, and F0/N AdaIN res-blocks run on the
    /// GPU; the `[feat ⊕ style]` concat, the duration rounding, and the
    /// length-regulation (inherently data-dependent) stay on the CPU.
    func predictGPU(
        dEn: [[Float]], style: [Float], speed: Float, gw: KokoroGPUWeights, device: Device = .shared
    ) -> (dur: [Int], f0: [Float], n: [Float], dAligned: [[Float]]) {
        // The LSTMs run on the CPU (negligible at L ≤ ~70, and the GPU LSTM
        // kernel has a data-dependent NaN at longer sequences — see the spawned
        // "Fix GPU LSTM" task). durationEncoder, predictor.lstm, duration_proj,
        // and the shared LSTM are the verified CPU helpers; the F0/N AdaIN
        // res-block curves run on the GPU.
        let d = durationEncoder(dEn, style: style)  // [L, 512]
        let x = KokoroMath.biLSTM(
            d.map { $0 + style },
            wihF: vec("predictor.lstm.weight_ih_l0"), whhF: vec("predictor.lstm.weight_hh_l0"),
            bihF: vec("predictor.lstm.bias_ih_l0"), bhhF: vec("predictor.lstm.bias_hh_l0"),
            wihR: vec("predictor.lstm.weight_ih_l0_reverse"), whhR: vec("predictor.lstm.weight_hh_l0_reverse"),
            bihR: vec("predictor.lstm.bias_ih_l0_reverse"), bhhR: vec("predictor.lstm.bias_hh_l0_reverse"),
            inDim: 640, hidden: 256)
        let dp = KokoroMath.linear(x, weight: vec("predictor.duration_proj.linear_layer.weight"),
            bias: vec("predictor.duration_proj.linear_layer.bias"), outDim: 50, inDim: 512)
        let maxFramesPerPhoneme = 100
        var dur = [Int]()
        for row in dp {
            var s: Float = 0
            for v in row { s += 1 / (1 + expf(-v)) }
            dur.append(min(max(Int((s / max(speed, 0.1)).rounded()), 1), maxFramesPerPhoneme))
        }
        var dAligned = [[Float]]()
        for (i, r) in d.enumerated() { for _ in 0 ..< dur[i] { dAligned.append(r) } }
        let T = dAligned.count
        let sh = KokoroMath.biLSTM(
            dAligned.map { $0 + style },
            wihF: vec("predictor.shared.weight_ih_l0"), whhF: vec("predictor.shared.weight_hh_l0"),
            bihF: vec("predictor.shared.bias_ih_l0"), bhhF: vec("predictor.shared.bias_hh_l0"),
            wihR: vec("predictor.shared.weight_ih_l0_reverse"), whhR: vec("predictor.shared.weight_hh_l0_reverse"),
            bihR: vec("predictor.shared.bias_ih_l0_reverse"), bhhR: vec("predictor.shared.bias_hh_l0_reverse"),
            inDim: 640, hidden: 256)
        // F0/N curves via the AdaIN res-block stack (GPU, channels-first).
        let f0 = curveGPU("F0", sh.flatMap { $0 }, rows: T, style: style, gw: gw, device: device)
        let n = curveGPU("N", sh.flatMap { $0 }, rows: T, style: style, gw: gw, device: device)
        return (dur, f0, n, dAligned)
    }

    /// AdaLayerNorm reduces to `LayerNorm(weight = 1+γ, bias = β)` with
    /// `[γ,β] = fc(style)`. Computes `fc(style)` on the CPU (tiny) → two `[C]`
    /// tensors.
    private func adaLayerNormParams(
        _ fcBase: String, style: [Float], channels C: Int, gw: KokoroGPUWeights, device: Device
    ) -> (weight: Tensor, bias: Tensor) {
        let fw = vec("\(fcBase).weight")  // [2C, 128]
        let fb = vec("\(fcBase).bias")
        let sd = style.count
        var weight = [Float](repeating: 0, count: C), bias = [Float](repeating: 0, count: C)
        for o in 0 ..< C {
            var g: Float = fb.isEmpty ? 0 : fb[o], b: Float = fb.isEmpty ? 0 : fb[C + o]
            let gb = o * sd, bb = (C + o) * sd
            for i in 0 ..< sd { g += fw[gb + i] * style[i]; b += fw[bb + i] * style[i] }
            weight[o] = 1 + g
            bias[o] = b
        }
        let wT = Tensor.empty(shape: [C], dtype: .f32, device: device); wT.copyIn(from: weight)
        let bT = Tensor.empty(shape: [C], dtype: .f32, device: device); bT.copyIn(from: bias)
        return (wT, bT)
    }

    /// One F0/N curve: `[T,512]` shared-LSTM output → 3 AdaIN res-blocks
    /// (block 1 up-samples ×2) → 1×1 proj → `[2T]`. Channels-first throughout.
    private func curveGPU(
        _ tag: String, _ shFlat: [Float], rows T: Int, style: [Float],
        gw: KokoroGPUWeights, device: Device
    ) -> [Float] {
        // channels-last [T,512] → channels-first [512, T].
        var cf = [Float](repeating: 0, count: 512 * T)
        for t in 0 ..< T { for c in 0 ..< 512 { cf[c * T + t] = shFlat[t * 512 + c] } }
        let shCF = Tensor.empty(shape: [512, T], dtype: .f32, device: device); shCF.copyIn(from: cf)
        var x = adainResBlkGPU(shCF, base: "predictor.\(tag).0", inC: 512, outC: 512, upsample: false, length: T, style: style, gw: gw, device: device)
        x = adainResBlkGPU(x, base: "predictor.\(tag).1", inC: 512, outC: 256, upsample: true, length: T, style: style, gw: gw, device: device)
        let T2 = 2 * T
        x = adainResBlkGPU(x, base: "predictor.\(tag).2", inC: 256, outC: 256, upsample: false, length: T2, style: style, gw: gw, device: device)
        // proj: plain Conv1d(256→1, k1).
        let cmd = device.makeCommandBuffer()
        let projW = gw.vecT("predictor.\(tag)_proj.weight", shape: [1, 256, 1])
        let projB = gw.vecT("predictor.\(tag)_proj.bias", shape: [1])
        let proj = Ops.conv1dDilated(x, weight: projW, bias: projB, batch: 1, inCh: 256, inLen: T2,
            outCh: 1, k: 1, stride: 1, pad: 0, dilation: 1, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        return proj.toFloatArray()  // [2T]
    }
}
