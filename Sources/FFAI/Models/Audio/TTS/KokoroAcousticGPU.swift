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
// Kokoro / StyleTTS2 iSTFTNet generator — GPU-resident port of the
// conv-heavy tail that dominates synthesis cost (the CPU generator runs
// ~40 G MAC over ~7800 frames; on one debug CPU core that is ~18 minutes,
// the bulk of a full utterance). Everything here is channels-first
// `[C, T]` in `MTLBuffer`-backed `Tensor`s so the metaltile conv / adain /
// snake / leaky-relu kernels chain without per-op readback.
//
// Numerically validated against the CPU reference (`KokoroWeights.generate`,
// itself verified against mlx-audio) — see `KokoroGPUParityTests`.
//
// Scope: the generator (the bottleneck). The small decoder AdaIN stack and
// the front-end stay on the CPU reference for now (the decoder's grouped
// conv-transpose `pool` needs a grouped-convT kernel — a follow-up); the CPU
// decoder hands its `[512, 2T]` output to `generateGPU`.

import Foundation
import Metal

// ─── GPU weight cache ────────────────────────────────────────────────

/// Uploads + caches the generator's weights as GPU `Tensor`s, reconstructing
/// weight-normalized convs (`g·v/‖v‖`) on the CPU once at first use. Conv
/// weights keep their PyTorch layout: dilated convs are OIK `[Cout, Cin, k]`,
/// transposed convs are IOK `[Cin, Cout, k]` — exactly what `Ops.conv1dDilated`
/// / `Ops.conv1dTranspose` expect.
final class KokoroGPUWeights {
    let w: KokoroWeights
    let device: Device
    private var cache: [String: Tensor] = [:]

    init(_ w: KokoroWeights, device: Device = .shared) {
        self.w = w
        self.device = device
    }

    private func upload(_ key: String, _ data: [Float], shape: [Int]) -> Tensor {
        if let t = cache[key] { return t }
        let t = Tensor.empty(shape: shape, dtype: .f32, device: device)
        t.copyIn(from: data)
        cache[key] = t
        return t
    }

    /// Weight-norm dilated-conv weight (OIK) `[Cout, Cin, k]`.
    func convW(_ base: String) -> Tensor {
        let (wv, sh) = w.convWeight(base)
        return upload("\(base).W", wv, shape: sh)
    }

    /// Conv bias `[Cout]` (zeros if the module is bias-less).
    func convB(_ base: String, outC: Int) -> Tensor {
        let b = w.convBias(base)
        return upload("\(base).B", b.isEmpty ? [Float](repeating: 0, count: outC) : b, shape: [outC])
    }

    /// Weight-norm transposed-conv weight (IOK) `[Cin, Cout, k]`, normalized
    /// over dim 0 (`g` is `[Cin,1,1]`).
    func convTW(_ base: String) -> Tensor {
        let g = w.vec("\(base).weight_g"), v = w.vec("\(base).weight_v")
        let sh = w.shape("\(base).weight_v")
        let wv = WeightNorm.effectiveWeight(g: g, v: v, shape: sh, exceptDim: 0)
        return upload("\(base).TW", wv, shape: sh)
    }

    /// Plain (non-weight-norm) conv weight by key, with its stored shape.
    func plainW(_ name: String) -> Tensor {
        upload("\(name).P", w.vec(name), shape: w.shape(name))
    }

    /// Snake per-channel `alpha` `[C]` (stored `[1,C,1]`).
    func alpha(_ name: String, channels C: Int) -> Tensor {
        upload("\(name).A", w.vec(name), shape: [C])
    }

    /// Weight-norm depthwise transposed-conv weight (the `pool`): stored
    /// `[C, 1, k]`, normalized over dim 0 → flattened `[C, k]`.
    func poolW(_ base: String, channels C: Int) -> Tensor {
        let g = w.vec("\(base).weight_g"), v = w.vec("\(base).weight_v")
        let sh = w.shape("\(base).weight_v")  // [C, 1, k]
        let wv = WeightNorm.effectiveWeight(g: g, v: v, shape: sh, exceptDim: 0)
        return upload("\(base).PW", wv, shape: [C, sh[2]])
    }

    /// A cached all-zero bias `[C]` (the weight-norm `conv1x1` is bias-less,
    /// but `Ops.conv1dDilated` requires a bias).
    func zeroBias(_ C: Int) -> Tensor {
        upload("zeroBias.\(C)", [Float](repeating: 0, count: C), shape: [C])
    }

    /// Upload an arbitrary CPU `[rows, cols]` (row-major) as a GPU tensor.
    func uploadMatrix(_ data: [Float], rows: Int, cols: Int) -> Tensor {
        let t = Tensor.empty(shape: [rows, cols], dtype: .f32, device: device)
        t.copyIn(from: data)
        return t
    }

    /// AdaIN affine for a `[C]`-channel feature map: `gamma = 1 + fc(style)[:C]`,
    /// `beta = fc(style)[C:]`, computed on the CPU (the `fc` is tiny — `[2C,128]
    /// · [128]`) and uploaded as two `[C]` tensors. The kernel folds `(1+γ)`
    /// into `gamma`, so we add 1 here.
    func adainAffine(_ fcBase: String, style: [Float], channels C: Int) -> (gamma: Tensor, beta: Tensor) {
        let fw = w.vec("\(fcBase).weight")  // [2C, 128]
        let fb = w.vec("\(fcBase).bias")  // [2C]
        let sd = style.count
        var gamma = [Float](repeating: 0, count: C)
        var beta = [Float](repeating: 0, count: C)
        for o in 0 ..< C {
            var gAcc: Float = fb.isEmpty ? 0 : fb[o]
            var bAcc: Float = fb.isEmpty ? 0 : fb[C + o]
            let gBase = o * sd, bBase = (C + o) * sd
            for i in 0 ..< sd { gAcc += fw[gBase + i] * style[i]; bAcc += fw[bBase + i] * style[i] }
            gamma[o] = 1 + gAcc
            beta[o] = bAcc
        }
        let gT = Tensor.empty(shape: [C], dtype: .f32, device: device); gT.copyIn(from: gamma)
        let bT = Tensor.empty(shape: [C], dtype: .f32, device: device); bT.copyIn(from: beta)
        return (gT, bT)
    }
}

// ─── GPU generator ───────────────────────────────────────────────────

extension KokoroWeights {
    /// GPU iSTFTNet generator. `x0` is the CPU decoder output, channels-last
    /// `[T2, 512]`; `f0` the predictor F0 curve. Returns the `[frames, nFreq]`
    /// re/im planes (same contract as the CPU `generate`). Runs the whole
    /// conv/adain/snake stack on the GPU in one command buffer, with only the
    /// (small) NSF source, last-stage left-pad, and final magnitude/phase split
    /// on the CPU.
    func generateGPU(
        _ x0: [[Float]], style: [Float], f0: [Float], vocoder: KokoroVocoder,
        gw: KokoroGPUWeights, device: Device = .shared
    ) -> (re: [[Float]], im: [[Float]]) {
        let upRates = [10, 6], upPads = [5, 3], upChans = [256, 128]
        let nFFT = vocoder.nFFT, hop = vocoder.hopLength
        let nFreq = nFFT / 2 + 1
        let totalUp = (upRates[0] * upRates[1]) * hop

        // CPU NSF source [framesFull, 22] → channels-first [22, framesFull] GPU.
        let har = nsfSource(f0Curve: f0, totalUpsample: totalUp, nFFT: nFFT, hop: hop)
        let harFrames = har.count
        let harCh = 2 * nFreq  // 22
        var harCF = [Float](repeating: 0, count: harCh * harFrames)
        for t in 0 ..< harFrames { for c in 0 ..< harCh { harCF[c * harFrames + t] = har[t][c] } }
        let harT = Tensor.empty(shape: [harCh, harFrames], dtype: .f32, device: device)
        harT.copyIn(from: harCF)

        // Decoder output → channels-first [512, T2] GPU.
        let t2 = x0.count, cin0 = x0.first?.count ?? 512
        var xCF = [Float](repeating: 0, count: cin0 * t2)
        for t in 0 ..< t2 { for c in 0 ..< cin0 { xCF[c * t2 + t] = x0[t][c] } }
        var x = Tensor.empty(shape: [cin0, t2], dtype: .f32, device: device)
        x.copyIn(from: xCF)
        var xLen = t2
        var xCh = cin0

        var cmd = device.makeCommandBuffer()
        let numUp = upRates.count
        for stage in 0 ..< numUp {
            let C = upChans[stage]
            x = Ops.leakyReLU(x, slope: 0.1, on: cmd)
            // Source branch.
            let strideF0 = (stage == 0) ? upRates[1] : 1
            let padSrc = (stage == 0) ? (strideF0 + 1) / 2 : 0
            let ncBase = "decoder.generator.noise_convs.\(stage)"
            let ncW = gw.plainW("\(ncBase).weight")  // [C, 22, k]
            let kSrc = ncW.shape[2]
            let ncB = gw.convB(ncBase, outC: C)
            let srcFrames = (harFrames + 2 * padSrc - kSrc) / strideF0 + 1
            var src = Ops.conv1dDilated(
                harT, weight: ncW, bias: ncB, batch: 1, inCh: harCh, inLen: harFrames,
                outCh: C, k: kSrc, stride: strideF0, pad: padSrc, dilation: 1, on: cmd
            ).reshaped(to: [C, srcFrames])
            src = genResBlockGPU(src, base: "decoder.generator.noise_res.\(stage)",
                channels: C, length: srcFrames, style: style, gw: gw, on: cmd)
            // Main branch: transposed-conv up-sample.
            let upBase = "decoder.generator.ups.\(stage)"
            let upW = gw.convTW(upBase)  // [Cin, C, k]
            let kUp = upW.shape[2]
            let upB = gw.convB(upBase, outC: C)
            var upLen = (xLen - 1) * upRates[stage] - 2 * upPads[stage] + (kUp - 1) + 1
            x = Ops.conv1dTranspose(
                x, weight: upW, bias: upB, batch: 1, inCh: xCh, inLen: xLen, outCh: C,
                outLen: upLen, k: kUp, stride: upRates[stage], pad: upPads[stage], dilation: 1, on: cmd
            ).reshaped(to: [C, upLen])
            if stage == numUp - 1 {
                // reflection_pad((1,0)) — prepend one zero frame (channels-first).
                // A CPU round-trip (runs once): flush the chain, read x, pad,
                // re-upload, then continue on a fresh command buffer.
                cmd.commit()
                cmd.waitUntilCompleted()
                let flat = x.toFloatArray()
                var padded = [Float](repeating: 0, count: C * (upLen + 1))
                for c in 0 ..< C { for t in 0 ..< upLen { padded[c * (upLen + 1) + (t + 1)] = flat[c * upLen + t] } }
                x = Tensor.empty(shape: [C, upLen + 1], dtype: .f32, device: device)
                x.copyIn(from: padded)
                upLen += 1
                cmd = device.makeCommandBuffer()
            }
            // x + source (shapes match by construction).
            let T = min(upLen, srcFrames)
            x = Ops.add(x, src, on: cmd)
            // 3 res-blocks averaged.
            var acc: Tensor? = nil
            for j in 0 ..< 3 {
                let rb = genResBlockGPU(x, base: "decoder.generator.resblocks.\(stage * 3 + j)",
                    channels: C, length: T, style: style, gw: gw, on: cmd)
                acc = acc == nil ? rb : Ops.add(acc!, rb, on: cmd)
            }
            let third = Tensor.filled(1.0 / 3.0, shape: [C, T], dtype: .f32, device: device)
            x = Ops.mul(acc!, third, on: cmd)
            xCh = C
            xLen = T
        }
        x = Ops.leakyReLU(x, slope: 0.01, on: cmd)
        // conv_post → [22, frames].
        let cpBase = "decoder.generator.conv_post"
        let cpW = gw.convW(cpBase)
        let cpK = cpW.shape[2]
        let cpB = gw.convB(cpBase, outC: harCh)
        // conv_post k7 pad3 stride1 keeps the length: frames == xLen.
        let frames = xLen
        let post = Ops.conv1dDilated(
            x, weight: cpW, bias: cpB, batch: 1, inCh: xCh, inLen: xLen, outCh: harCh,
            k: cpK, stride: 1, pad: 3, dilation: 1, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // CPU magnitude/phase split: spec = exp(post[:nFreq]), phase = sin(post[nFreq:]),
        // re = spec·cos(phase), im = spec·sin(phase). post is channels-first [22, frames].
        let postFlat = post.toFloatArray()
        var re = [[Float]](repeating: [Float](repeating: 0, count: nFreq), count: frames)
        var im = re
        for t in 0 ..< frames {
            for k in 0 ..< nFreq {
                let mag = expf(postFlat[k * frames + t])
                let ph = sinf(postFlat[(nFreq + k) * frames + t])
                re[t][k] = mag * cosf(ph)
                im[t][k] = mag * sinf(ph)
            }
        }
        return (re, im)
    }

    /// GPU AdaIN-Snake res-block (`AdaINResBlock1`): for each dilation `d`,
    /// `x += convs2(snake(α2, convs1(snake(α1, adain1(x)))))`. Channels-first
    /// `[C, length]` throughout. Mirrors the CPU `genResBlock`.
    private func genResBlockGPU(
        _ x0: Tensor, base: String, channels C: Int, length T: Int,
        style: [Float], gw: KokoroGPUWeights, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let dils = [1, 3, 5]
        // Normalise to 2-D [C, T]: conv ops emit [1, C, T], adain/snake emit
        // [C, T], and Ops.add is shape-strict — keep everything [C, T].
        var h = x0.reshaped(to: [C, T])
        for (j, dil) in dils.enumerated() {
            let k1 = shape("\(base).convs1.\(j).weight_v")[2]
            let k2 = shape("\(base).convs2.\(j).weight_v")[2]
            let (g1, b1) = gw.adainAffine("\(base).adain1.\(j).fc", style: style, channels: C)
            var a = Ops.adain1d(h, gamma: g1, beta: b1, rows: C, length: T, on: cmd)
            a = Ops.snake(a, alpha: gw.alpha("\(base).alpha1.\(j)", channels: C), channels: C, length: T, on: cmd)
            a = Ops.conv1dDilated(
                a, weight: gw.convW("\(base).convs1.\(j)"), bias: gw.convB("\(base).convs1.\(j)", outC: C),
                batch: 1, inCh: C, inLen: T, outCh: C, k: k1, stride: 1,
                pad: dil * (k1 - 1) / 2, dilation: dil, on: cmd
            ).reshaped(to: [C, T])
            let (g2, b2) = gw.adainAffine("\(base).adain2.\(j).fc", style: style, channels: C)
            a = Ops.adain1d(a, gamma: g2, beta: b2, rows: C, length: T, on: cmd)
            a = Ops.snake(a, alpha: gw.alpha("\(base).alpha2.\(j)", channels: C), channels: C, length: T, on: cmd)
            a = Ops.conv1dDilated(
                a, weight: gw.convW("\(base).convs2.\(j)"), bias: gw.convB("\(base).convs2.\(j)", outC: C),
                batch: 1, inCh: C, inLen: T, outCh: C, k: k2, stride: 1,
                pad: (k2 - 1) / 2, dilation: 1, on: cmd
            ).reshaped(to: [C, T])
            h = Ops.add(h, a, on: cmd)
        }
        return h
    }
}

// ─── GPU decoder (AdaIN res-block stack) ─────────────────────────────

extension KokoroWeights {
    /// GPU decoder: `concat(asr, F0, N) → encode → concat(asr_res, F0, N) →
    /// decode ×4 (last up-samples ×2) → generator → re/im`. Mirrors the CPU
    /// `decode`. The small F0/N stride-2 convs run on the CPU (1-channel); the
    /// AdaIN res-block stack and generator run on the GPU. Each res-block
    /// self-manages its command buffer (it materialises its output for the
    /// next channels-first concat), so the orchestration stays simple.
    func decodeGPU(
        asr: [[Float]], f0: [Float], n: [Float], style: [Float], vocoder: KokoroVocoder,
        gw: KokoroGPUWeights, device: Device = .shared
    ) -> (re: [[Float]], im: [[Float]]) {
        let T = asr.count
        // F0/N stride-2 depthwise convs (tiny, 1-channel) on the CPU.
        func single(_ curve: [Float], base: String) -> [Float] {
            let (cw, _) = convWeight(base)  // [1,1,3]
            let b = convBias(base)
            let L = curve.count
            let lOut = (L + 2 * 1 - 3) / 2 + 1
            return (0 ..< lOut).map { t -> Float in
                var acc: Float = b.isEmpty ? 0 : b[0]
                let start = t * 2 - 1
                for k in 0 ..< 3 { let s = start + k; if s >= 0 && s < L { acc += cw[k] * curve[s] } }
                return acc
            }
        }
        let f0d = single(f0, base: "decoder.F0_conv")  // [T]
        let nd = single(n, base: "decoder.N_conv")  // [T]
        // F0/N as channels-first [1, T] GPU rows.
        let f0T = gw.uploadMatrix(f0d, rows: 1, cols: T)
        let ndT = gw.uploadMatrix(nd, rows: 1, cols: T)
        // asr channels-first [512, T].
        var asrCF = [Float](repeating: 0, count: 512 * T)
        for t in 0 ..< T { for c in 0 ..< 512 { asrCF[c * T + t] = asr[t][c] } }
        let asrT = gw.uploadMatrix(asrCF, rows: 512, cols: T)
        // asr_res.0 (512→64, k1) on the GPU.
        let arBase = "decoder.asr_res.0"
        let arK = shape("\(arBase).weight_v")[2]
        let cmdAR = device.makeCommandBuffer()
        let asrRes = Ops.conv1dDilated(
            asrT, weight: gw.convW(arBase), bias: gw.convB(arBase, outC: 64),
            batch: 1, inCh: 512, inLen: T, outCh: 64, k: arK, stride: 1, pad: 0, dilation: 1, on: cmdAR
        ).reshaped(to: [64, T])
        cmdAR.commit(); cmdAR.waitUntilCompleted()

        // encode: AdainResBlk on concat(asr, F0, N) = [514, T] → [1024, T].
        let encIn = concatRowsGPU([asrT, f0T, ndT], rows: [512, 1, 1], length: T, gw: gw, device: device)
        var x = adainResBlkGPU(encIn, base: "decoder.encode", inC: 514, outC: 1024,
            upsample: false, length: T, style: style, gw: gw, device: device)
        var xLen = T
        for blk in 0 ..< 4 {
            // concat(x, asr_res, F0, N) = [1090, T].
            let cat = concatRowsGPU([x, asrRes, f0T, ndT], rows: [x.shape[0], 64, 1, 1], length: T, gw: gw, device: device)
            let up = (blk == 3)
            let outC = up ? 512 : 1024
            x = adainResBlkGPU(cat, base: "decoder.decode.\(blk)", inC: 1090, outC: outC,
                upsample: up, length: T, style: style, gw: gw, device: device)
            xLen = up ? 2 * T : T
        }
        // Decoder output [512, 2T] → channels-last [2T, 512] for the generator.
        let xFlat = x.toFloatArray()
        let x0 = (0 ..< xLen).map { t in (0 ..< 512).map { c in xFlat[c * xLen + t] } }
        return generateGPU(x0, style: style, f0: f0, vocoder: vocoder, gw: gw, device: device)
    }

    /// GPU `AdainResBlk1d`: `norm1 → leaky → [pool] → conv1 → norm2 → leaky →
    /// conv2`, residual `(x + shortcut)/√2` with optional ×2 up-sample +
    /// 1×1 shortcut. Channels-first `[C, T]`. Self-managed command buffer
    /// (commits + materialises its result).
    private func adainResBlkGPU(
        _ x: Tensor, base: String, inC: Int, outC: Int, upsample: Bool, length T: Int,
        style: [Float], gw: KokoroGPUWeights, device: Device
    ) -> Tensor {
        var cmd = device.makeCommandBuffer()
        let (g1, b1) = gw.adainAffine("\(base).norm1.fc", style: style, channels: inC)
        var h = Ops.adain1d(x.reshaped(to: [inC, T]), gamma: g1, beta: b1, rows: inC, length: T, on: cmd)
        h = Ops.leakyReLU(h, slope: 0.2, on: cmd)
        var hLen = T
        if upsample {
            // pool: depthwise convT (k3 s2 p1) → [inC, 2T-1], then left-pad → [inC, 2T].
            let poolW = gw.poolW("\(base).pool", channels: inC)
            let poolB = gw.convB("\(base).pool", outC: inC)
            h = Ops.conv1dTransposeDepthwise(h, weight: poolW, bias: poolB,
                channels: inC, inLen: T, k: 3, stride: 2, pad: 1, on: cmd)
            let poolLen = (T - 1) * 2 - 2 + 2 + 1  // 2T - 1
            cmd.commit(); cmd.waitUntilCompleted()
            let flat = h.toFloatArray()
            var padded = [Float](repeating: 0, count: inC * (poolLen + 1))
            for c in 0 ..< inC { for t in 0 ..< poolLen { padded[c * (poolLen + 1) + (t + 1)] = flat[c * poolLen + t] } }
            h = Tensor.empty(shape: [inC, poolLen + 1], dtype: .f32, device: device)
            h.copyIn(from: padded)
            hLen = poolLen + 1  // 2T
            cmd = device.makeCommandBuffer()
        }
        let k1 = shape("\(base).conv1.weight_v")[2]
        h = Ops.conv1dDilated(
            h, weight: gw.convW("\(base).conv1"), bias: gw.convB("\(base).conv1", outC: outC),
            batch: 1, inCh: inC, inLen: hLen, outCh: outC, k: k1, stride: 1, pad: (k1 - 1) / 2, dilation: 1, on: cmd
        ).reshaped(to: [outC, hLen])
        let (g2, b2) = gw.adainAffine("\(base).norm2.fc", style: style, channels: outC)
        h = Ops.adain1d(h, gamma: g2, beta: b2, rows: outC, length: hLen, on: cmd)
        h = Ops.leakyReLU(h, slope: 0.2, on: cmd)
        let k2 = shape("\(base).conv2.weight_v")[2]
        h = Ops.conv1dDilated(
            h, weight: gw.convW("\(base).conv2"), bias: gw.convB("\(base).conv2", outC: outC),
            batch: 1, inCh: outC, inLen: hLen, outCh: outC, k: k2, stride: 1, pad: (k2 - 1) / 2, dilation: 1, on: cmd
        ).reshaped(to: [outC, hLen])
        // Shortcut: [×2 up-sample] → [1×1 conv].
        var sc = x.reshaped(to: [inC, T])
        var scLen = T
        if upsample {
            sc = Ops.upsampleNearest1d(sc, channels: inC, inLen: T, factor: 2, on: cmd)
            scLen = 2 * T
        }
        if has("\(base).conv1x1.weight_v") {
            sc = Ops.conv1dDilated(
                sc, weight: gw.convW("\(base).conv1x1"), bias: gw.zeroBias(outC),
                batch: 1, inCh: inC, inLen: scLen, outCh: outC, k: 1, stride: 1, pad: 0, dilation: 1, on: cmd
            ).reshaped(to: [outC, scLen])
        }
        // (h + sc) / √2.
        let T2 = min(hLen, scLen)
        var out = Ops.add(h.reshaped(to: [outC, hLen]), sc, on: cmd)
        let invSqrt2 = Tensor.filled(0.7071067811865476, shape: [outC, T2], dtype: .f32, device: device)
        out = Ops.mul(out, invSqrt2, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        return out
    }

    /// Concatenate channels-first `[Cᵢ, length]` tensors along the channel
    /// axis → `[ΣCᵢ, length]`. Channel-major layout makes each block a
    /// contiguous region, so this is a sequence of blits.
    private func concatRowsGPU(
        _ parts: [Tensor], rows: [Int], length: Int, gw: KokoroGPUWeights, device: Device
    ) -> Tensor {
        let total = rows.reduce(0, +)
        let out = Tensor.empty(shape: [total, length], dtype: .f32, device: device)
        let cmd = device.makeCommandBuffer()
        var rowOffset = 0
        for (i, part) in parts.enumerated() {
            let dst = out.slicedRows(start: rowOffset, count: rows[i])
            Ops.copy(part.reshaped(to: [rows[i], length]), into: dst, on: cmd)
            rowOffset += rows[i]
        }
        cmd.commit(); cmd.waitUntilCompleted()
        return out
    }
}
