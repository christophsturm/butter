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
// Gemma 4 audio tower (`gemma4_audio`) — a Conformer / USM speech encoder
// that turns a waveform into `[frames, output_proj_dims]` soft tokens
// spliced into the text stream at `audio_token_id` placeholders (between
// the `boa` / `eoa` span markers). This file is PHASE 1 of the multimodal
// build (see `planning/gemma4-multimodal-plan.md`): the config surface.
// The mel front-end, conv subsampling, the 12 Conformer layers (macaron
// FFN + local chunked attention + depthwise-conv module + clipped
// linears), and the cross-modal splice land in subsequent phases.

import Foundation
import Metal

/// Parsed `audio_config` block of a Gemma 4 multimodal checkpoint
/// (`google/gemma-4-*` / `mlx-community/gemma-4-*`). Mirrors the values the
/// reference `Gemma4AudioEncoder` consumes.
struct Gemma4AudioConfig {
    /// Encoder hidden width (1024 on E2B).
    let hidden: Int
    /// Number of Conformer layers (12 on E2B).
    let numLayers: Int
    /// Self-attention heads (8 on E2B).
    let numHeads: Int
    /// Width the encoder projects to before splicing into the text stream
    /// — must equal the text backbone's hidden size (1536 on E2B).
    let outputProjDims: Int
    let rmsNormEps: Float

    // ── Local chunked self-attention geometry ──
    /// Frames per attention chunk.
    let attentionChunkSize: Int
    /// Extra left-context frames each chunk may attend to.
    let attentionContextLeft: Int
    /// Extra right-context frames (0 = causal local window on E2B).
    let attentionContextRight: Int
    /// tanh logit soft-cap applied to attention scores.
    let attentionLogitCap: Float
    /// Fill value for masked-out (out-of-window) attention positions.
    let attentionInvalidLogitsValue: Float

    // ── Conv module + subsampling ──
    /// Depthwise conv kernel width in the conv module (5 on E2B).
    let convKernelSize: Int
    /// Channel widths of the two mel-subsampling conv layers ([128, 32]).
    let subsamplingConvChannels: [Int]

    /// Macaron-FFN residual scale (0.5 on E2B).
    let residualWeight: Float
    /// Whether the audio linears carry per-tensor input/output clipping
    /// ranges (the Gemma-4-audio "clipped linear" scheme).
    let useClippedLinears: Bool
    /// Hidden activation ("silu" on E2B).
    let hiddenActivation: String

    static func decode(_ c: ModelConfig) throws -> Gemma4AudioConfig {
        guard let hidden = c.int("hidden_size"),
            let numLayers = c.int("num_hidden_layers"),
            let numHeads = c.int("num_attention_heads")
        else {
            throw Gemma4Error.missingConfig("audio_config")
        }
        return Gemma4AudioConfig(
            hidden: hidden,
            numLayers: numLayers,
            numHeads: numHeads,
            outputProjDims: c.int("output_proj_dims") ?? hidden,
            rmsNormEps: Float(c.float("rms_norm_eps") ?? 1e-6),
            attentionChunkSize: c.int("attention_chunk_size") ?? 12,
            attentionContextLeft: c.int("attention_context_left") ?? 13,
            attentionContextRight: c.int("attention_context_right") ?? 0,
            attentionLogitCap: Float(c.float("attention_logit_cap") ?? 50.0),
            attentionInvalidLogitsValue: Float(
                c.float("attention_invalid_logits_value") ?? -1e9),
            convKernelSize: c.int("conv_kernel_size") ?? 5,
            subsamplingConvChannels: c.intArray("subsampling_conv_channels") ?? [128, 32],
            residualWeight: Float(c.float("residual_weight") ?? 0.5),
            useClippedLinears: c.bool("use_clipped_linears") ?? true,
            hiddenActivation: c.string("hidden_act") ?? "silu")
    }
}

// ─── Gemma 4 audio tower — Conformer / USM speech encoder ─────────────
//
// Same idioms as the Gemma 4 vision tower (`Gemma4Vision.swift`):
// token-major `[nFrames, hidden]` activations, GPU GEMMs through a
// `projectRows` helper, GPU element-wise Ops, and CPU attention via
// `DispatchQueue.concurrentPerform`.
//
// The attention is chunked-local with a sinusoidal relative-position
// embedding + a "relative-shift" reindex and a local-causal validity
// mask — there is no GPU kernel for it, so the whole attention block
// runs on the CPU (the requirement, and the only safe option for this
// pass). It is heavily commented because the relative-shift is subtle.
//
// The single-clip `B = 1` case is the entire `encode` contract: the
// public entry takes one mel spectrogram `[nFrames, 128]`, so every
// internal tensor carries an implicit leading batch of 1 and we work
// purely in the `[T, hidden]` token-major layout.

/// Linear with optional input/output clamping (`Gemma4ClippableLinear`).
/// Clamps the input to `[input_min, input_max]`, applies the (possibly
/// quantized) linear, then clamps the output to `[output_min,
/// output_max]`. The four bounds are scalar tensors loaded from the
/// checkpoint; they default to ±inf so clamping is a no-op until real
/// values are present.
final class Gemma4AudioClippableLinear {
    let linear: AnyLinear
    let inputMin: Float
    let inputMax: Float
    let outputMin: Float
    let outputMax: Float
    /// Whether any clamp bound is finite — when all four are ±inf the
    /// clamp passes are skipped entirely (the common case at load time
    /// for tensors without recorded ranges).
    let clampsInput: Bool
    let clampsOutput: Bool

    init(
        linear: AnyLinear,
        inputMin: Float, inputMax: Float,
        outputMin: Float, outputMax: Float
    ) {
        self.linear = linear
        self.inputMin = inputMin
        self.inputMax = inputMax
        self.outputMin = outputMin
        self.outputMax = outputMax
        self.clampsInput = inputMin.isFinite || inputMax.isFinite
        self.clampsOutput = outputMin.isFinite || outputMax.isFinite
    }

    /// Load `<base>.linear.{weight,…}` through `loadLinear` (so a 4-bit
    /// checkpoint yields a `QuantizedLinear`), plus the four optional
    /// scalar clamp bounds at `<base>.{input_min,input_max,output_min,
    /// output_max}`.
    static func load(
        base: String, in bundle: SafeTensorsBundle,
        quantization: ModelConfig.QuantizationConfig?
    ) throws -> Gemma4AudioClippableLinear {
        let linear = try loadLinear(
            base: "\(base).linear", in: bundle, quantization: quantization)
        // Each bound is a scalar tensor; default ±inf when absent.
        func bound(_ name: String, _ fallback: Float) -> Float {
            guard let t = try? bundle.tensor(named: "\(base).\(name)") else {
                return fallback
            }
            let v = t.toFloatArray()
            return v.isEmpty ? fallback : v[0]
        }
        return Gemma4AudioClippableLinear(
            linear: linear,
            inputMin: bound("input_min", -.infinity),
            inputMax: bound("input_max", .infinity),
            outputMin: bound("output_min", -.infinity),
            outputMax: bound("output_max", .infinity))
    }

    /// Apply to every row of a `[nRows, inDim]` token-major tensor:
    /// optional input clamp → GEMM → optional output clamp. Queues the
    /// GEMM on the supplied command buffer; the caller commits. Clamps
    /// (rare — bounds default to ±inf) run on the CPU, which forces a
    /// commit+wait around the GEMM only when a finite bound is present.
    /// `cmd` is `inout`: a CPU clamp forces a commit+wait (to read the
    /// pending GEMM/activations back), which retires the caller's command
    /// buffer. We replace it with a fresh one so the caller can keep
    /// chaining ops + its own final commit stays valid (encoding on a
    /// committed buffer traps). No clamp → the caller's `cmd` is reused
    /// unchanged (the efficient path; bounds default to ±inf).
    func projectRows(
        _ x: Tensor, nRows: Int, outDim: Int,
        on cmd: inout MTLCommandBuffer, device: Device
    ) -> Tensor {
        var input = x
        if clampsInput {
            // Materialise the input clamp before the GEMM. The caller's
            // `cmd` may carry pending work that produced `x`; commit it so
            // the CPU readback sees finished values, then refresh it.
            cmd.commit()
            cmd.waitUntilCompleted()
            input = gemma4AudioClampTensor(
                x, min: inputMin, max: inputMax, device: device)
            cmd = device.makeCommandBuffer()
        }
        let y = linear.callMany(input, t: nRows, on: cmd, device: device)
        if clampsOutput {
            cmd.commit()
            cmd.waitUntilCompleted()
            let clamped = gemma4AudioClampTensor(
                y, min: outputMin, max: outputMax, device: device)
            cmd = device.makeCommandBuffer()
            return clamped
        }
        return y
    }
}

/// CPU element-wise clamp of every value of `x` to `[min, max]`. Used
/// only when a `Gemma4AudioClippableLinear` carries a finite bound (the
/// uncommon case); there is no GPU clamp Op, and the clamp is cheap.
func gemma4AudioClampTensor(
    _ x: Tensor, min lo: Float, max hi: Float, device: Device
) -> Tensor {
    var flat = x.toFloatArray()
    for i in 0 ..< flat.count {
        if flat[i] < lo { flat[i] = lo }
        if flat[i] > hi { flat[i] = hi }
    }
    return MediaUtils.makeTensor(
        from: flat, shape: x.shape, dtype: x.dtype, device: device)
}

/// `AudioRMSNorm` — RMSNorm with the weight applied directly (no `+1`
/// zero-shift, unlike GemmaRMSNorm). Held as a plain weight tensor.
final class Gemma4AudioRMSNorm {
    let weight: Tensor
    let eps: Float

    init(weight: Tensor, eps: Float) {
        self.weight = weight
        self.eps = eps
    }

    static func load(
        named name: String, in bundle: SafeTensorsBundle, eps: Float
    ) throws -> Gemma4AudioRMSNorm {
        Gemma4AudioRMSNorm(
            weight: try bundle.tensor(named: "\(name).weight"), eps: eps)
    }

    /// Multi-row RMSNorm over `[nRows, rowSize]` token-major input.
    func normRows(
        _ x: Tensor, nRows: Int, rowSize: Int,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        Ops.rmsNormRows(
            x, weight: weight, eps: eps, nRows: nRows, rowSize: rowSize, on: cmd)
    }
}

/// One Gemma 4 macaron Conformer feed-forward sub-block. Mirrors
/// `ConformerFeedForward`: clip → pre-norm → ffw1 → SiLU → ffw2 → clip
/// → post-norm, then `residual + out * residual_weight`.
final class Gemma4ConformerFeedForward {
    let preNorm: Gemma4AudioRMSNorm
    let ffw1: Gemma4AudioClippableLinear
    let ffw2: Gemma4AudioClippableLinear
    let postNorm: Gemma4AudioRMSNorm
    let gradientClipping: Float
    let residualWeight: Float
    let hidden: Int
    /// `hidden * 4` — the FFN expansion width.
    let intermediate: Int

    init(
        preNorm: Gemma4AudioRMSNorm, ffw1: Gemma4AudioClippableLinear,
        ffw2: Gemma4AudioClippableLinear, postNorm: Gemma4AudioRMSNorm,
        gradientClipping: Float, residualWeight: Float,
        hidden: Int, intermediate: Int
    ) {
        self.preNorm = preNorm
        self.ffw1 = ffw1
        self.ffw2 = ffw2
        self.postNorm = postNorm
        self.gradientClipping = gradientClipping
        self.residualWeight = residualWeight
        self.hidden = hidden
        self.intermediate = intermediate
    }

    /// `x` is token-major `[nTokens, hidden]`. Returns the same shape.
    func forward(
        _ x: Tensor, nTokens: Int, on cmd: inout MTLCommandBuffer, device: Device
    ) -> Tensor {
        // residual = x; (clip x); pre_norm; ffw1; silu; ffw2; (clip);
        // post_norm; residual + out * residual_weight.
        //
        // The reference `clip(x, ±gradient_clipping)` uses
        // gradient_clipping = 1e10 — far outside any real activation
        // range, so it is a no-op on inference and there is no GPU clamp
        // Op; we skip it. (If a checkpoint ever ships a tighter bound this
        // would need a CPU clamp like the clippable-linear path.)
        _ = gradientClipping
        let normed = preNorm.normRows(x, nRows: nTokens, rowSize: hidden, on: cmd)
        let h1 = ffw1.projectRows(
            normed, nRows: nTokens, outDim: intermediate, on: &cmd, device: device)
        let act = Ops.silu(h1, on: cmd)
        let h2 = ffw2.projectRows(
            act, nRows: nTokens, outDim: hidden, on: &cmd, device: device)
        let post = postNorm.normRows(h2, nRows: nTokens, rowSize: hidden, on: cmd)
        // residual + out * residual_weight. Fold the scalar `residual_weight`
        // by multiplying with a tiled-scalar tensor (no GPU scalar-scale Op).
        let scaleT = gemma4AudioFilledTensor(
            value: residualWeight, count: nTokens * hidden,
            shape: [nTokens, hidden], dtype: x.dtype, device: device)
        let scaled = Ops.mul(post, scaleT, on: cmd)
        return Ops.add(x, scaled, on: cmd)
    }
}

/// Build a `[count]`-element tensor filled with a constant `value`,
/// shaped `shape`. Used to express a scalar multiply via `Ops.mul`
/// (there is no GPU scalar-scale Op).
func gemma4AudioFilledTensor(
    value: Float, count: Int, shape: [Int], dtype: DType, device: Device
) -> Tensor {
    MediaUtils.makeTensor(
        from: [Float](repeating: value, count: count),
        shape: shape, dtype: dtype, device: device)
}

/// Light-convolution module (`ConformerLightConv1d`): pre-norm →
/// linear_start(2×) → GLU → causal depthwise conv1d (kernel 5) → clip →
/// conv_norm → SiLU → linear_end, plus a residual.
final class Gemma4ConformerLightConv1d {
    let preNorm: Gemma4AudioRMSNorm
    let linearStart: Gemma4AudioClippableLinear
    /// Depthwise conv weight `[hidden, 1, kernel]` (one filter per
    /// channel; in_ch-per-group = 1).
    let depthwiseWeight: Tensor
    let convNorm: Gemma4AudioRMSNorm
    let linearEnd: Gemma4AudioClippableLinear
    let hidden: Int
    let kernelSize: Int
    let gradientClipping: Float
    /// Causal left-pad = kernel - 1.
    let causalPadding: Int

    init(
        preNorm: Gemma4AudioRMSNorm, linearStart: Gemma4AudioClippableLinear,
        depthwiseWeight: Tensor, convNorm: Gemma4AudioRMSNorm,
        linearEnd: Gemma4AudioClippableLinear, hidden: Int, kernelSize: Int,
        gradientClipping: Float
    ) {
        self.preNorm = preNorm
        self.linearStart = linearStart
        self.depthwiseWeight = depthwiseWeight
        self.convNorm = convNorm
        self.linearEnd = linearEnd
        self.hidden = hidden
        self.kernelSize = kernelSize
        self.gradientClipping = gradientClipping
        self.causalPadding = kernelSize - 1
    }

    /// `x` is token-major `[nTokens, hidden]`. Returns the same shape.
    /// Manages its own command buffers (the GLU + causal conv1d cross a
    /// CPU stage), so the caller doesn't pass one in.
    func forward(
        _ x: Tensor, nTokens: Int, device: Device
    ) -> Tensor {
        _ = gradientClipping  // gradient_clipping = 1e10 is an inference no-op.
        var cmd = device.makeCommandBuffer()
        let normed = preNorm.normRows(x, nRows: nTokens, rowSize: hidden, on: cmd)
        // linear_start projects hidden → 2·hidden; split + GLU gate.
        let proj = linearStart.projectRows(
            normed, nRows: nTokens, outDim: hidden * 2, on: &cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()

        // GLU: x1 * sigmoid(x2), where [x1 | x2] = split halves of the
        // last dim. Done on the CPU because we then need the causal
        // depthwise conv1d, which is most faithfully expressed on the
        // CPU at this small width (hidden ≤ 1024, kernel 5).
        let projFlat = proj.toFloatArray()  // [nTokens, 2*hidden]
        var gated = [Float](repeating: 0, count: nTokens * hidden)  // [T, hidden]
        for t in 0 ..< nTokens {
            let base = t * 2 * hidden
            for c in 0 ..< hidden {
                let a = projFlat[base + c]
                let b = projFlat[base + hidden + c]
                gated[t * hidden + c] = a * (1.0 / (1.0 + exp(-b)))
            }
        }

        // Causal depthwise conv1d: each channel `c` is convolved with its
        // own length-`kernel` filter over the time axis, left-padded by
        // `causalPadding` zeros so output[t] only depends on inputs ≤ t.
        // depthwiseWeight is `[hidden, 1, kernel]` row-major; tap[k] is
        // the weight for time offset (t - (kernel-1) + k).
        let w = depthwiseWeight.toFloatArray()  // [hidden, kernel]
        let k = kernelSize
        // VALIDATE: depthwise weight layout assumed [hidden, 1, kernel]
        // flattened to hidden*kernel, tap index `c*k + j` for channel c,
        // filter position j (j=0 is the oldest / left-most tap).
        var conv = [Float](repeating: 0, count: nTokens * hidden)
        for t in 0 ..< nTokens {
            for c in 0 ..< hidden {
                var acc: Float = 0
                for j in 0 ..< k {
                    // Position in the padded sequence: the left pad shifts
                    // the effective source index by `causalPadding`.
                    let src = t - causalPadding + j
                    if src >= 0 {
                        acc += gated[src * hidden + c] * w[c * k + j]
                    }
                }
                conv[t * hidden + c] = acc
            }
        }
        let convT = MediaUtils.makeTensor(
            from: conv, shape: [nTokens, hidden], dtype: x.dtype, device: device)

        // clip → conv_norm → SiLU → linear_end, + residual. The clip is
        // gradient_clipping (1e10) — a no-op on inference (see the FFN).
        var cmd2 = device.makeCommandBuffer()
        let cn = convNorm.normRows(convT, nRows: nTokens, rowSize: hidden, on: cmd2)
        let act = Ops.silu(cn, on: cmd2)
        let end = linearEnd.projectRows(
            act, nRows: nTokens, outDim: hidden, on: &cmd2, device: device)
        let out = Ops.add(end, x, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return out
    }
}

/// One full Gemma 4 Conformer block (`ConformerBlock`): macaron FFW →
/// pre/post-norm chunked-local attention → light-conv module → macaron
/// FFW → final norm. The attention runs on the CPU.
final class Gemma4ConformerBlock {
    let feedForward1: Gemma4ConformerFeedForward
    let attention: Gemma4AudioAttention
    let lightConv: Gemma4ConformerLightConv1d
    let feedForward2: Gemma4ConformerFeedForward
    let normPreAttn: Gemma4AudioRMSNorm
    let normPostAttn: Gemma4AudioRMSNorm
    let normOut: Gemma4AudioRMSNorm
    let hidden: Int
    let gradientClipping: Float

    init(
        feedForward1: Gemma4ConformerFeedForward, attention: Gemma4AudioAttention,
        lightConv: Gemma4ConformerLightConv1d,
        feedForward2: Gemma4ConformerFeedForward,
        normPreAttn: Gemma4AudioRMSNorm, normPostAttn: Gemma4AudioRMSNorm,
        normOut: Gemma4AudioRMSNorm, hidden: Int, gradientClipping: Float
    ) {
        self.feedForward1 = feedForward1
        self.attention = attention
        self.lightConv = lightConv
        self.feedForward2 = feedForward2
        self.normPreAttn = normPreAttn
        self.normPostAttn = normPostAttn
        self.normOut = normOut
        self.hidden = hidden
        self.gradientClipping = gradientClipping
    }

    /// `x` token-major `[nTokens, hidden]`; `valid[t] == true` for real
    /// (non-padding) frames. `causalValidMask` is the precomputed
    /// `[chunkSize, contextSize]` local-causal mask. Returns the block
    /// output `[nTokens, hidden]`.
    func forward(
        _ x: Tensor, nTokens: Int, valid: [Bool], causalValidMask: [Bool],
        device: Device
    ) -> Tensor {
        // The reference clamps with gradient_clipping (1e10) at several
        // points; all are inference no-ops (see the FFN sub-block), so
        // they are omitted here.
        _ = gradientClipping
        // ── feed_forward1 ──
        var cmd = device.makeCommandBuffer()
        var h = feedForward1.forward(x, nTokens: nTokens, on: &cmd, device: device)

        // ── attention sub-block (pre/post norm + residual) ──
        // residual = h; norm_pre_attn; attn; h = residual + norm_post_attn(attn).
        let residual = h
        let preNormed = normPreAttn.normRows(
            h, nRows: nTokens, rowSize: hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // CPU chunked-local attention (the relative-shift attention).
        let attnOut = attention.forward(
            preNormed, nTokens: nTokens, valid: valid,
            causalValidMask: causalValidMask, device: device)

        cmd = device.makeCommandBuffer()
        let postNormed = normPostAttn.normRows(
            attnOut, nRows: nTokens, rowSize: hidden, on: cmd)
        h = Ops.add(residual, postNormed, on: cmd)

        // Zero out invalid (padding) positions before the light-conv, so
        // the causal conv never mixes padding into a real frame.
        // (`validity_mask` multiply in the reference.)
        cmd.commit()
        cmd.waitUntilCompleted()
        h = zeroInvalidRows(h, nTokens: nTokens, valid: valid, device: device)

        // ── light-conv module ──
        h = lightConv.forward(h, nTokens: nTokens, device: device)

        // ── feed_forward2 → norm_out ──
        var cmd2 = device.makeCommandBuffer()
        let ff2 = feedForward2.forward(h, nTokens: nTokens, on: &cmd2, device: device)
        let out = normOut.normRows(ff2, nRows: nTokens, rowSize: hidden, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return out
    }

    /// Zero every row `t` whose `valid[t]` is false (the reference's
    /// `x * (~mask)` multiply). Cheap CPU pass — runs once per block.
    private func zeroInvalidRows(
        _ x: Tensor, nTokens: Int, valid: [Bool], device: Device
    ) -> Tensor {
        var flat = x.toFloatArray()
        for t in 0 ..< nTokens where !valid[t] {
            for c in 0 ..< hidden { flat[t * hidden + c] = 0 }
        }
        return MediaUtils.makeTensor(
            from: flat, shape: [nTokens, hidden], dtype: x.dtype, device: device)
    }
}

/// Chunked-local self-attention with a sinusoidal relative-position
/// embedding, a "relative-shift" reindex, per-dim query scaling, and
/// tanh logit soft-capping. Runs entirely on the CPU.
///
/// The reference (`AudioAttention` + `AudioRelativePositionEmbedding`)
/// splits the `T`-frame sequence into chunks of `chunkSize` frames
/// ("query blocks") and, for each chunk, gathers a `contextSize`-frame
/// "context block" of keys/values spanning `max_past_horizon` frames to
/// the left, the chunk itself, and `max_future_horizon` frames to the
/// right. Attention is local: each query in a chunk attends only to its
/// chunk's context block, gated by `causalValidMask`.
final class Gemma4AudioAttention {
    let qProj: Gemma4AudioClippableLinear
    let kProj: Gemma4AudioClippableLinear
    let vProj: Gemma4AudioClippableLinear
    let post: Gemma4AudioClippableLinear
    /// Relative-position key projection `[numHeads*headDim, hidden]`
    /// (a plain linear; `relative_k_proj.weight`). Used to project the
    /// sinusoidal position signal into per-head position keys.
    let relativeKProj: AnyLinear
    /// Per-dim learnable scale `[headDim]` (softplus-activated).
    let perDimScale: [Float]

    let numHeads: Int
    let headDim: Int
    let hidden: Int
    let chunkSize: Int
    /// `max(0, attention_context_left - 1)`.
    let maxPastHorizon: Int
    /// `attention_context_right`.
    let maxFutureHorizon: Int
    /// `chunkSize + maxPastHorizon + maxFutureHorizon`.
    let contextSize: Int
    let invalidLogitsValue: Float
    let softcap: Float
    /// `(headDim^-0.5) / ln 2`.
    let qScale: Float
    /// `ln(1 + e) / ln 2`.
    let kScale: Float
    /// `[maxSpanPlus1, numHeads, headDim]` — the projected sinusoidal
    /// relative-position keys, precomputed once at load. `maxSpanPlus1 =
    /// maxPastHorizon + maxFutureHorizon + 1` (the number of distinct
    /// relative offsets in `[maxPast … -maxFuture]`).
    let relPosKeys: [Float]
    let maxSpanPlus1: Int

    init(
        qProj: Gemma4AudioClippableLinear, kProj: Gemma4AudioClippableLinear,
        vProj: Gemma4AudioClippableLinear, post: Gemma4AudioClippableLinear,
        relativeKProj: AnyLinear, perDimScale: [Float],
        numHeads: Int, headDim: Int, hidden: Int, chunkSize: Int,
        maxPastHorizon: Int, maxFutureHorizon: Int, contextSize: Int,
        invalidLogitsValue: Float, softcap: Float, qScale: Float, kScale: Float,
        relPosKeys: [Float], maxSpanPlus1: Int
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.post = post
        self.relativeKProj = relativeKProj
        self.perDimScale = perDimScale
        self.numHeads = numHeads
        self.headDim = headDim
        self.hidden = hidden
        self.chunkSize = chunkSize
        self.maxPastHorizon = maxPastHorizon
        self.maxFutureHorizon = maxFutureHorizon
        self.contextSize = contextSize
        self.invalidLogitsValue = invalidLogitsValue
        self.softcap = softcap
        self.qScale = qScale
        self.kScale = kScale
        self.relPosKeys = relPosKeys
        self.maxSpanPlus1 = maxSpanPlus1
    }

    /// Build the projected sinusoidal relative-position keys at load
    /// time. Mirrors `AudioRelativePositionEmbedding.__call__`'s
    /// precomputable head: the `pos_indices = [maxPast … -maxFuture]`
    /// sinusoidal signal (channel `= hidden`) projected by
    /// `relative_k_proj` and reshaped to `[maxSpanPlus1, numHeads,
    /// headDim]`. Position-only; independent of the activations.
    static func buildRelPosKeys(
        relativeKProj: AnyLinear, dtype: DType,
        hidden: Int, numHeads: Int, headDim: Int,
        maxPastHorizon: Int, maxFutureHorizon: Int, device: Device
    ) -> (keys: [Float], maxSpanPlus1: Int) {
        // pos_indices = arange(maxPast, -maxFuture - 1, -1): descending.
        var positions: [Float] = []
        var p = maxPastHorizon
        while p >= -maxFutureHorizon {
            positions.append(Float(p))
            p -= 1
        }
        let maxSpanPlus1 = positions.count

        // Sinusoidal timing signal, channel dim = hidden. num_timescales =
        // hidden / 2; first half is sin, second half cos (concatenated).
        let numTimescales = hidden / 2
        let minTimescale: Float = 1.0
        let maxTimescale: Float = 10000.0
        let logInc =
            log(maxTimescale / minTimescale) / Float(max(numTimescales - 1, 1))
        var invTimescales = [Float](repeating: 0, count: numTimescales)
        for i in 0 ..< numTimescales {
            invTimescales[i] = minTimescale * exp(Float(i) * -logInc)
        }
        // signal[pos] = [sin(pos*inv[0..]) … | cos(pos*inv[0..]) …] (hidden).
        var signal = [Float](repeating: 0, count: maxSpanPlus1 * hidden)
        for r in 0 ..< maxSpanPlus1 {
            let pos = positions[r]
            for i in 0 ..< numTimescales {
                let scaled = pos * invTimescales[i]
                signal[r * hidden + i] = sin(scaled)
                signal[r * hidden + numTimescales + i] = cos(scaled)
            }
        }

        // Project the signal by relative_k_proj → [maxSpanPlus1,
        // numHeads*headDim], then keep as [maxSpanPlus1, numHeads, headDim].
        // `dtype` is the projection weight's storage dtype — the GEMM
        // input must match it.
        let cmd = device.makeCommandBuffer()
        let signalT = MediaUtils.makeTensor(
            from: signal, shape: [maxSpanPlus1, hidden], dtype: dtype, device: device)
        let projected = relativeKProj.callMany(
            signalT, t: maxSpanPlus1, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        // projected is [maxSpanPlus1, numHeads*headDim] row-major; the
        // reshape to [maxSpanPlus1, numHeads, headDim] is a no-op on the
        // flat array.
        return (projected.toFloatArray(), maxSpanPlus1)
    }

    /// CPU chunked-local attention. `hidden_states` token-major
    /// `[nTokens, hidden]`. Returns `[nTokens, hidden]`.
    func forward(
        _ hiddenStates: Tensor, nTokens: Int, valid: [Bool],
        causalValidMask: [Bool], device: Device
    ) -> Tensor {
        let nQH = numHeads
        let hd = headDim
        let qkvDim = numHeads * headDim

        // ── q/k/v projections (GPU GEMMs through the clippable linears) ──
        var cmd = device.makeCommandBuffer()
        let qT = qProj.projectRows(
            hiddenStates, nRows: nTokens, outDim: qkvDim, on: &cmd, device: device)
        let kT = kProj.projectRows(
            hiddenStates, nRows: nTokens, outDim: qkvDim, on: &cmd, device: device)
        let vT = vProj.projectRows(
            hiddenStates, nRows: nTokens, outDim: qkvDim, on: &cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()

        var qa = qT.toFloatArray()  // [nTokens, numHeads*headDim]
        var ka = kT.toFloatArray()
        let va = vT.toFloatArray()

        // Per-dim query scale: q *= q_scale * softplus(per_dim_scale);
        // k *= k_scale. softplus(x) = log(1 + e^x).
        var perDim = [Float](repeating: 0, count: hd)
        for d in 0 ..< hd {
            let x = perDimScale[d]
            // Numerically stable softplus.
            perDim[d] = x > 20 ? x : log1p(exp(x))
        }
        for t in 0 ..< nTokens {
            for h in 0 ..< nQH {
                let base = t * qkvDim + h * hd
                for d in 0 ..< hd {
                    qa[base + d] *= qScale * perDim[d]
                    ka[base + d] *= kScale
                }
            }
        }

        // ── Chunk geometry ──
        // numBlocks = ceil(T / chunkSize). Each query block holds
        // chunkSize frames (the last padded with zeros); each context
        // block holds contextSize key/value frames extracted with the
        // _extract_block_context windowing.
        let numBlocks = (nTokens + chunkSize - 1) / chunkSize

        // Output accumulator, token-major [nTokens, numHeads*headDim].
        let H = numHeads * headDim
        var outFlat = [Float](repeating: 0, count: nTokens * H)

        // Capture immutables for the @Sendable concurrent closure (rebind
        // the now-scaled q/k arrays to `let` copies so they cross the
        // closure boundary as value types, matching the vision tower's
        // `cpuAttention` pattern).
        let qScaled = qa
        let kScaled = ka
        let chunk = chunkSize
        let ctx = contextSize
        let maxSpan = maxSpanPlus1
        let cap = softcap
        let invalidVal = invalidLogitsValue
        let relKeys = relPosKeys  // [maxSpan, numHeads, headDim]
        let maxPast = maxPastHorizon

        // For context extraction we conceptually left-pad k/v/valid by
        // maxPastHorizon and right-pad by maxFutureHorizon + chunkSize - 1,
        // then block `u` reads padded indices [u*chunkSize ..
        // u*chunkSize + contextSize). A context slot `c` of block `u` maps
        // back to original frame index `u*chunk + c - maxPast`; out of
        // range → padding (zero key/value, invalid mask). Inlined inside
        // the closure below to avoid a non-Sendable local-func capture.

        // Parallelize over (block, head). Each (u, h) writes only to the
        // disjoint output rows of block `u`, head `h` — race-free.
        outFlat.withUnsafeMutableBufferPointer { outBuf in
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: numBlocks * nQH) { work in
                let u = work / nQH
                let h = work % nQH

                // For each query frame `w` (0..<chunkSize) in this block:
                for w in 0 ..< chunk {
                    let qFrame = u * chunk + w  // original frame index
                    if qFrame >= nTokens { continue }  // padded query → skip
                    let qBase = qFrame * qkvDim + h * hd

                    // Compute logits over the contextSize key slots.
                    var logits = [Float](repeating: 0, count: ctx)
                    for c in 0 ..< ctx {
                        // Context slot → original frame (see note above).
                        let kFrame = u * chunk + c - maxPast
                        // term_ac: q · k (key from the context frame).
                        var ac: Float = 0
                        if kFrame >= 0 && kFrame < nTokens {
                            let kBase = kFrame * qkvDim + h * hd
                            for d in 0 ..< hd {
                                ac += qScaled[qBase + d] * kScaled[kBase + d]
                            }
                        }
                        // term_bd: q · relPosKey[relOffset], where the
                        // relative offset maps the (query slot w, key slot c)
                        // pair to a row of the precomputed relative-position
                        // keys via the relative-shift. See _relative_shift:
                        // after the shift, query slot `w` attending key slot
                        // `c` indexes relative position
                        //   relIdx = (maxPast + w) - c
                        // clamped to [0, maxSpan); the descending pos_indices
                        // make row 0 the largest past offset (+maxPast) and
                        // row maxSpan-1 the largest future offset (-maxFuture).
                        // VALIDATE: relative-shift index mapping
                        //   relIdx = maxPast + w - c
                        // derived from _relative_shift's pad/reshape/slice;
                        // confirm against the mlx-vlm oracle.
                        var bd: Float = 0
                        let relIdx = maxPast + w - c
                        if relIdx >= 0 && relIdx < maxSpan {
                            let relBase = (relIdx * nQH + h) * hd
                            for d in 0 ..< hd {
                                bd += qScaled[qBase + d] * relKeys[relBase + d]
                            }
                        }
                        logits[c] = ac + bd
                    }

                    // tanh soft-cap: logits = tanh(logits / cap) * cap.
                    for c in 0 ..< ctx {
                        logits[c] = tanh(logits[c] / cap) * cap
                    }

                    // Validity mask: a context slot is attended iff its
                    // source frame is valid AND the local-causal mask
                    // (causalValidMask[w, c]) permits it. Invalid → set to
                    // invalidLogitsValue so it drops out of the softmax.
                    for c in 0 ..< ctx {
                        let kFrame = u * chunk + c - maxPast
                        let sourceValid =
                            kFrame >= 0 && kFrame < nTokens && valid[kFrame]
                        let maskOK = causalValidMask[w * ctx + c]
                        if !(sourceValid && maskOK) {
                            logits[c] = invalidVal
                        }
                    }

                    // Softmax over the contextSize logits.
                    var maxLogit = -Float.greatestFiniteMagnitude
                    for c in 0 ..< ctx where logits[c] > maxLogit { maxLogit = logits[c] }
                    var sumExp: Float = 0
                    for c in 0 ..< ctx {
                        let e = exp(logits[c] - maxLogit)
                        logits[c] = e
                        sumExp += e
                    }
                    let invSum = sumExp > 0 ? 1.0 / sumExp : 0

                    // Weighted sum of context values → output row for this
                    // (query frame, head).
                    let oBase = qFrame * H + h * hd
                    for d in 0 ..< hd { outPtr[oBase + d] = 0 }
                    for c in 0 ..< ctx {
                        let kFrame = u * chunk + c - maxPast
                        guard kFrame >= 0 && kFrame < nTokens else { continue }
                        let prob = logits[c] * invSum
                        if prob == 0 { continue }
                        let vBase = kFrame * qkvDim + h * hd
                        for d in 0 ..< hd {
                            outPtr[oBase + d] += prob * va[vBase + d]
                        }
                    }
                }
            }
        }

        // Reassemble [nTokens, hidden] context and post-project.
        let contextT = MediaUtils.makeTensor(
            from: outFlat, shape: [nTokens, H], dtype: hiddenStates.dtype,
            device: device)
        var cmd2 = device.makeCommandBuffer()
        let projected = post.projectRows(
            contextT, nRows: nTokens, outDim: hidden, on: &cmd2, device: device)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return projected
    }
}

/// The Gemma 4 Conformer audio encoder. Loads from the `audio_tower.*`
/// tensors and turns a mel spectrogram into audio soft tokens in the
/// text hidden dim. Public contract — other code depends on the
/// `load` / `encode` signatures, so don't change them.
// `@unchecked Sendable`: all stored state is immutable `let` (config,
// weight tensors, the Conformer block stack, the precomputed mask) set
// once at load and never mutated — the encoder is read-only during
// `encode`. Same rationale as `Gemma4VLVisionModel`. Needed so the audio
// encode hook (`@Sendable` closure on `VisionModel`) can capture it.
final class Gemma4AudioEncoder: @unchecked Sendable {
    let config: Gemma4AudioConfig
    /// SSCP subsampling: two Conv2d(3×3, stride 2) + LayerNorm + ReLU
    /// blocks, then a flatten + input-projection linear.
    let convChannels: [Int]
    /// Conv block 0/1 weights `[outCh, inCh, 3, 3]` (OIHW) and their
    /// per-channel LayerNorm scale `[outCh]` (no bias).
    let conv0Weight: Tensor
    let conv0NormWeight: Tensor
    let conv1Weight: Tensor
    let conv1NormWeight: Tensor
    /// `input_proj_linear` `[hidden, freq*lastChannels]` (no bias).
    let inputProj: AnyLinear
    let inputProjInDim: Int
    let blocks: [Gemma4ConformerBlock]
    /// `output_proj` `[outputProjDims, hidden]` + bias `[outputProjDims]`.
    let outputProj: AnyLinear
    let dtype: DType
    /// Precomputed local-causal validity mask `[chunkSize, contextSize]`.
    let causalValidMask: [Bool]

    init(
        config: Gemma4AudioConfig, convChannels: [Int],
        conv0Weight: Tensor, conv0NormWeight: Tensor,
        conv1Weight: Tensor, conv1NormWeight: Tensor,
        inputProj: AnyLinear, inputProjInDim: Int,
        blocks: [Gemma4ConformerBlock], outputProj: AnyLinear,
        dtype: DType, causalValidMask: [Bool]
    ) {
        self.config = config
        self.convChannels = convChannels
        self.conv0Weight = conv0Weight
        self.conv0NormWeight = conv0NormWeight
        self.conv1Weight = conv1Weight
        self.conv1NormWeight = conv1NormWeight
        self.inputProj = inputProj
        self.inputProjInDim = inputProjInDim
        self.blocks = blocks
        self.outputProj = outputProj
        self.dtype = dtype
        self.causalValidMask = causalValidMask
    }

    /// Mel front-end feature width — fixed at 128 mel bins (the SSCP
    /// `INPUT_FEAT_SIZE`).
    static let melFeatureSize = 128
    /// Conv kernel + stride for the SSCP subsampling.
    static let convKernel = 3
    static let convStride = 2
    /// Symmetric padding (1,1,1,1) on the SSCP convs.
    static let convPad = 1

    /// Load from the `audio_tower.*` tensors of a Gemma 4 checkpoint.
    static func load(
        weights: SafeTensorsBundle, config: Gemma4AudioConfig,
        quantization: ModelConfig.QuantizationConfig?, device: Device
    ) throws -> Gemma4AudioEncoder {
        // Probe the prefix — VLM conversions may namespace the tower
        // under `model.audio_tower.` rather than `audio_tower.`.
        let at =
            weights.has("audio_tower.output_proj.weight")
            ? weights.prefixed("audio_tower.")
            : weights.prefixed("model.audio_tower.")

        let eps = config.rmsNormEps

        // ── SSCP subsampling convs ──
        let sscp = "subsample_conv_projection"
        let conv0Weight = try at.tensor(named: "\(sscp).layer0.conv.weight")
        let conv0NormWeight = try at.tensor(named: "\(sscp).layer0.norm.weight")
        let conv1Weight = try at.tensor(named: "\(sscp).layer1.conv.weight")
        let conv1NormWeight = try at.tensor(named: "\(sscp).layer1.norm.weight")
        let dtype = conv0Weight.dtype

        // input_proj_linear: hidden ← freq * lastChannels.
        // freq after two (k=3, stride=2, pad=1) convs over 128 mel bins:
        //   freq = (128 + 2 - 3)/2 + 1 = 64, then (64+2-3)/2+1 = 32.
        var freq = melFeatureSize
        for _ in 0 ..< 2 { freq = (freq + 2 * convPad - convKernel) / convStride + 1 }
        let lastChannels = config.subsamplingConvChannels.last ?? 32
        let inputProjInDim = freq * lastChannels
        let inputProj = try loadLinear(
            base: "\(sscp).input_proj_linear", in: at, quantization: quantization)

        // ── Conformer blocks ──
        let hidden = config.hidden
        let numHeads = config.numHeads
        let headDim = hidden / numHeads
        var blocks: [Gemma4ConformerBlock] = []
        blocks.reserveCapacity(config.numLayers)
        for i in 0 ..< config.numLayers {
            let p = "layers.\(i)"

            func ffw(_ name: String) throws -> Gemma4ConformerFeedForward {
                Gemma4ConformerFeedForward(
                    preNorm: try Gemma4AudioRMSNorm.load(
                        named: "\(p).\(name).pre_layer_norm", in: at, eps: eps),
                    ffw1: try Gemma4AudioClippableLinear.load(
                        base: "\(p).\(name).ffw_layer_1", in: at,
                        quantization: quantization),
                    ffw2: try Gemma4AudioClippableLinear.load(
                        base: "\(p).\(name).ffw_layer_2", in: at,
                        quantization: quantization),
                    postNorm: try Gemma4AudioRMSNorm.load(
                        named: "\(p).\(name).post_layer_norm", in: at, eps: eps),
                    gradientClipping: gradientClippingValue,
                    residualWeight: config.residualWeight,
                    hidden: hidden, intermediate: hidden * 4)
            }

            // ── Attention ──
            let maxPast = max(0, config.attentionContextLeft - 1)
            let maxFuture = config.attentionContextRight
            let contextSize = config.attentionChunkSize + maxPast + maxFuture
            let relKProj = try at.tensor(named: "\(p).self_attn.relative_k_proj.weight")
            let relKLinear = AnyLinear(Linear(weight: relKProj))
            let (relKeys, maxSpan) = Gemma4AudioAttention.buildRelPosKeys(
                relativeKProj: relKLinear, dtype: relKProj.dtype,
                hidden: hidden, numHeads: numHeads,
                headDim: headDim, maxPastHorizon: maxPast,
                maxFutureHorizon: maxFuture, device: device)
            let perDimScaleT = try at.tensor(named: "\(p).self_attn.per_dim_scale")
            let attention = Gemma4AudioAttention(
                qProj: try Gemma4AudioClippableLinear.load(
                    base: "\(p).self_attn.q_proj", in: at, quantization: quantization),
                kProj: try Gemma4AudioClippableLinear.load(
                    base: "\(p).self_attn.k_proj", in: at, quantization: quantization),
                vProj: try Gemma4AudioClippableLinear.load(
                    base: "\(p).self_attn.v_proj", in: at, quantization: quantization),
                post: try Gemma4AudioClippableLinear.load(
                    base: "\(p).self_attn.post", in: at, quantization: quantization),
                relativeKProj: relKLinear,
                perDimScale: perDimScaleT.toFloatArray(),
                numHeads: numHeads, headDim: headDim, hidden: hidden,
                chunkSize: config.attentionChunkSize,
                maxPastHorizon: maxPast, maxFutureHorizon: maxFuture,
                contextSize: contextSize,
                invalidLogitsValue: config.attentionInvalidLogitsValue,
                softcap: config.attentionLogitCap,
                qScale: powf(Float(headDim), -0.5) / log(2.0),
                kScale: log(1.0 + Float(M_E)) / log(2.0),
                relPosKeys: relKeys, maxSpanPlus1: maxSpan)

            // ── Light conv ──
            let lightConv = Gemma4ConformerLightConv1d(
                preNorm: try Gemma4AudioRMSNorm.load(
                    named: "\(p).lconv1d.pre_layer_norm", in: at, eps: eps),
                linearStart: try Gemma4AudioClippableLinear.load(
                    base: "\(p).lconv1d.linear_start", in: at,
                    quantization: quantization),
                depthwiseWeight: try at.tensor(
                    named: "\(p).lconv1d.depthwise_conv1d.weight"),
                convNorm: try Gemma4AudioRMSNorm.load(
                    named: "\(p).lconv1d.conv_norm", in: at, eps: eps),
                linearEnd: try Gemma4AudioClippableLinear.load(
                    base: "\(p).lconv1d.linear_end", in: at,
                    quantization: quantization),
                hidden: hidden, kernelSize: config.convKernelSize,
                gradientClipping: gradientClippingValue)

            blocks.append(
                Gemma4ConformerBlock(
                    feedForward1: try ffw("feed_forward1"),
                    attention: attention,
                    lightConv: lightConv,
                    feedForward2: try ffw("feed_forward2"),
                    normPreAttn: try Gemma4AudioRMSNorm.load(
                        named: "\(p).norm_pre_attn", in: at, eps: eps),
                    normPostAttn: try Gemma4AudioRMSNorm.load(
                        named: "\(p).norm_post_attn", in: at, eps: eps),
                    normOut: try Gemma4AudioRMSNorm.load(
                        named: "\(p).norm_out", in: at, eps: eps),
                    hidden: hidden, gradientClipping: gradientClippingValue))
        }

        // ── Output projection ──
        let outputProj = try loadLinear(
            base: "output_proj", in: at, quantization: quantization)

        // ── Precompute the local-causal validity mask ──
        let causalValidMask = buildCausalValidMask(config: config)

        return Gemma4AudioEncoder(
            config: config, convChannels: config.subsamplingConvChannels,
            conv0Weight: conv0Weight, conv0NormWeight: conv0NormWeight,
            conv1Weight: conv1Weight, conv1NormWeight: conv1NormWeight,
            inputProj: inputProj, inputProjInDim: inputProjInDim,
            blocks: blocks, outputProj: outputProj, dtype: dtype,
            causalValidMask: causalValidMask)
    }

    /// Gradient-clipping bound from the reference config
    /// (`gradient_clipping: 1e10`). Not in `Gemma4AudioConfig`, so it's
    /// pinned to the reference default here.
    static let gradientClippingValue: Float = 10_000_000_000.0

    /// Build the `[chunkSize, contextSize]` local-causal validity mask.
    /// Mirrors `_build_causal_valid_mask`: a query at chunk-position `w`
    /// may attend a context slot `c` iff the slot is within
    /// `[w, w + maxPast + maxFuture]` — i.e. not before the query's own
    /// past window and not beyond its future window.
    static func buildCausalValidMask(config: Gemma4AudioConfig) -> [Bool] {
        let chunkSize = config.attentionChunkSize
        let maxFuture = config.attentionContextRight
        let maxPast = max(0, config.attentionContextLeft - 1)
        let upperDiagonal = maxPast + maxFuture
        let contextSize = chunkSize + maxPast + maxFuture

        // Reference:
        //   lower_causal = tril(ones(context, chunk)).T   → [chunk, context]
        //   upper_causal = tril(ones(chunk, context), k=upperDiagonal)
        //   mask = lower_causal & upper_causal
        //
        // lower_causal[w, c] = (c >= w)   (transpose of a lower-triangular)
        // upper_causal[w, c] = (c <= w + upperDiagonal)
        // VALIDATE: causal-mask band [w, w+upperDiagonal]; check against
        // the mlx-vlm oracle's `_build_causal_valid_mask` output.
        var mask = [Bool](repeating: false, count: chunkSize * contextSize)
        for w in 0 ..< chunkSize {
            for c in 0 ..< contextSize {
                let lower = c >= w
                let upper = c <= w + upperDiagonal
                mask[w * contextSize + c] = lower && upper
            }
        }
        return mask
    }

    /// Encode a mel spectrogram `[nFrames, 128]` (+ per-frame validity
    /// mask, true = valid) to audio soft tokens `[nOutFrames,
    /// outputProjDims (1536)]`.
    func encode(mel: Tensor, melMask: [Bool], device: Device) -> (encodings: Tensor, mask: [Bool]) {
        precondition(
            mel.shape.count == 2 && mel.shape[1] == Gemma4AudioEncoder.melFeatureSize,
            "Gemma4AudioEncoder.encode: mel must be [nFrames, 128], got \(mel.shape)")
        let nFrames = mel.shape[0]
        precondition(
            melMask.count == nFrames,
            "Gemma4AudioEncoder.encode: melMask count \(melMask.count) ≠ nFrames \(nFrames)")

        // `InspectTap` is a no-op unless BUTTER_INSPECT is set (single bool
        // compare per boundary); under `butter inspect --layer-trace` it
        // prints per-layer min/max/nan/inf + first values. Each Conformer
        // block commits its own command buffers internally, so `h` is
        // always GPU-resident at a boundary — the tap just reads it and
        // hands back a fresh (empty) buffer we let the next stage replace.
        let tap = InspectTap.fromEnvironment

        // ── SSCP subsampling → [nOut, hidden] + downsampled mask ──
        // The reference masks invalid frames to zero, pads (T,F) by 1,
        // runs the 3×3 stride-2 conv, LayerNorm(channels) + ReLU, twice,
        // then flattens (F·C) and projects to hidden. The convs are small
        // (channels ≤ 128, freq ≤ 64) so the subsampling runs on the CPU
        // for a faithful, easy-to-validate layout; the projection is a
        // GPU GEMM.
        let (subsampled, subMask, nOut) = subsample(
            mel: mel, melMask: melMask, device: device)

        // input_proj_linear: [nOut, freq*lastChannels] → [nOut, hidden].
        var workCmd = device.makeCommandBuffer()
        var h = inputProj.callMany(
            subsampled, t: nOut, on: workCmd, device: device)
        workCmd.commit()
        workCmd.waitUntilCompleted()
        workCmd = tap.dumpLayerBoundary(
            h, label: "audio_sscp", layer: -1,
            cmd: device.makeCommandBuffer(), device: device)

        // ── Conformer block stack ──
        for (i, block) in blocks.enumerated() {
            h = block.forward(
                h, nTokens: nOut, valid: subMask,
                causalValidMask: causalValidMask, device: device)
            workCmd = tap.dumpLayerBoundary(
                h, label: "audio_layer", layer: i,
                cmd: workCmd, device: device)
        }

        // ── Output projection (1024 → 1536, + bias) ──
        var encodings = outputProj.callMany(
            h, t: nOut, on: workCmd, device: device)
        workCmd.commit()
        workCmd.waitUntilCompleted()

        // Zero out invalid output frames (the reference's final
        // `mx.where(mask, 0, encodings)`).
        encodings = zeroInvalidOutputs(
            encodings, nOut: nOut, valid: subMask, device: device)
        workCmd = tap.dumpLayerBoundary(
            encodings, label: "audio_out", layer: -1,
            cmd: device.makeCommandBuffer(), device: device)

        return (encodings, subMask)
    }

    /// Run the two SSCP conv blocks on the CPU and return the flattened
    /// `[nOut, freq*lastChannels]` features, the downsampled validity
    /// mask, and `nOut`. Each block: zero invalid frames → pad (T,F) by 1
    /// → Conv2d(3×3, stride 2) → LayerNorm(channels) → ReLU; the time
    /// mask is downsampled by the stride.
    private func subsample(
        mel: Tensor, melMask: [Bool], device: Device
    ) -> (features: Tensor, mask: [Bool], nOut: Int) {
        let melFlat = mel.toFloatArray()  // [nFrames, 128]
        let nFrames = mel.shape[0]
        let inFeat = Gemma4AudioEncoder.melFeatureSize

        // Treat the mel as a [T, F, C=1] feature map (channel-last, like
        // the reference). The conv weight is OHWI — `[outCh, kH, kW, inCh]`
        // (mlx `Conv2d` layout, confirmed from the checkpoint).
        // Block 0: inCh=1, outCh=convChannels[0]. Block 1: inCh=
        // convChannels[0], outCh=convChannels[1].
        var curT = nFrames
        var curF = inFeat
        var curC = 1
        // Current activation, flat [T, F, C].
        var act = [Float](repeating: 0, count: curT * curF * curC)
        for t in 0 ..< curT {
            for f in 0 ..< inFeat {
                act[t * inFeat + f] = melFlat[t * inFeat + f]
            }
        }
        var mask = melMask

        let weights = [conv0Weight, conv1Weight]
        let normWeights = [conv0NormWeight, conv1NormWeight]
        let outChannels = convChannels

        for blockIdx in 0 ..< 2 {
            let outCh = outChannels[blockIdx]
            let inCh = curC
            let w = weights[blockIdx].toFloatArray()  // [outCh, inCh, 3, 3]
            let normW = normWeights[blockIdx].toFloatArray()  // [outCh]

            // Zero invalid (padding) time frames before the conv.
            for t in 0 ..< curT where !mask[t] {
                for f in 0 ..< curF {
                    for c in 0 ..< inCh { act[(t * curF + f) * inCh + c] = 0 }
                }
            }

            // Output spatial dims after pad=1, k=3, stride=2.
            let outT =
                (curT + 2 * Gemma4AudioEncoder.convPad - Gemma4AudioEncoder.convKernel)
                / Gemma4AudioEncoder.convStride + 1
            let outF =
                (curF + 2 * Gemma4AudioEncoder.convPad - Gemma4AudioEncoder.convKernel)
                / Gemma4AudioEncoder.convStride + 1
            // VALIDATE: SSCP conv output dims outT/outF for the chosen
            // padding/stride; confirm against mlx-vlm shapes.
            var outAct = [Float](repeating: 0, count: outT * outF * outCh)

            let pad = Gemma4AudioEncoder.convPad
            let stride = Gemma4AudioEncoder.convStride
            let k = Gemma4AudioEncoder.convKernel
            let inCurF = curF
            let inCurT = curT
            // Conv2d: out[ot, of, oc] = sum_{kh,kw,ic}
            //   in[ot*stride - pad + kh, of*stride - pad + kw, ic] *
            //   w[oc, ic, kh, kw].  Symmetric zero padding.
            for ot in 0 ..< outT {
                for of in 0 ..< outF {
                    for oc in 0 ..< outCh {
                        var acc: Float = 0
                        for kh in 0 ..< k {
                            let it = ot * stride - pad + kh
                            if it < 0 || it >= inCurT { continue }
                            for kw in 0 ..< k {
                                let iff = of * stride - pad + kw
                                if iff < 0 || iff >= inCurF { continue }
                                for ic in 0 ..< inCh {
                                    let inV = act[(it * inCurF + iff) * inCh + ic]
                                    // mlx `Conv2d` weight is OHWI:
                                    // [outCh, kH, kW, inCh] (confirmed from the
                                    // checkpoint — layer0 is [128,3,3,1]).
                                    let wV = w[((oc * k + kh) * k + kw) * inCh + ic]
                                    acc += inV * wV
                                }
                            }
                        }
                        outAct[(ot * outF + of) * outCh + oc] = acc
                    }
                }
            }

            // LayerNorm over the channel dim (last), no bias, then ReLU.
            // Reference uses LayerNorm(out_channels, eps=rms_norm_eps).
            let eps = config.rmsNormEps
            for ot in 0 ..< outT {
                for of in 0 ..< outF {
                    let base = (ot * outF + of) * outCh
                    var mean: Float = 0
                    for c in 0 ..< outCh { mean += outAct[base + c] }
                    mean /= Float(outCh)
                    var varr: Float = 0
                    for c in 0 ..< outCh {
                        let d = outAct[base + c] - mean
                        varr += d * d
                    }
                    varr /= Float(outCh)
                    let inv = 1.0 / (varr + eps).squareRoot()
                    for c in 0 ..< outCh {
                        var v = (outAct[base + c] - mean) * inv * normW[c]
                        if v < 0 { v = 0 }  // ReLU
                        outAct[base + c] = v
                    }
                }
            }

            // Downsample the time mask by the stride: take every
            // `stride`-th frame, capped to outT.
            var newMask = [Bool](repeating: false, count: outT)
            for t in 0 ..< outT {
                let src = t * stride
                newMask[t] = src < mask.count ? mask[src] : false
            }
            mask = newMask

            act = outAct
            curT = outT
            curF = outF
            curC = outCh
        }

        // Flatten F·C → [T, F*C] (channel-last flatten matches the
        // reference reshape(B, T, F*C)).
        let flatDim = curF * curC
        precondition(
            flatDim == inputProjInDim,
            "Gemma4AudioEncoder.subsample: flat dim \(flatDim) ≠ "
                + "input_proj in-dim \(inputProjInDim)")
        // `act` is already [T, F, C] row-major == [T, F*C] flat.
        let features = MediaUtils.makeTensor(
            from: act, shape: [curT, flatDim], dtype: dtype, device: device)
        return (features, mask, curT)
    }

    /// Zero every output row whose downsampled mask is invalid.
    private func zeroInvalidOutputs(
        _ x: Tensor, nOut: Int, valid: [Bool], device: Device
    ) -> Tensor {
        let outDim = config.outputProjDims
        var flat = x.toFloatArray()
        for t in 0 ..< nOut where t < valid.count && !valid[t] {
            for c in 0 ..< outDim { flat[t * outDim + c] = 0 }
        }
        return MediaUtils.makeTensor(
            from: flat, shape: [nOut, outDim], dtype: x.dtype, device: device)
    }
}

/// Semantic alias used by the Gemma 4 Omni encoder-free path, where the
/// same RMSNorm-no-scale + linear projector is the shared multimodal
/// embedder for BOTH the vision (`embed_vision`) and audio (`embed_audio`)
/// projections. The implementation lives on `Gemma4AudioEmbedder` (the
/// original VL audio-tower call site); the alias keeps the vision call site
/// readable.
typealias Gemma4MultimodalEmbedder = Gemma4AudioEmbedder

/// Multi-modal embedder for the audio path (`embed_audio`). Mirrors the
/// reference `MultimodalEmbedder`: an unweighted RMSNorm (`RMSNormNoScale`,
/// `embedding_pre_projection_norm`) over each audio soft token followed by
/// a linear projection (`embedding_projection`) from the encoder's output
/// width (`output_proj_dims`, 1536 on E2B) into the text hidden dim. The
/// projection may be quantized in 4-bit checkpoints, so it is held as an
/// `AnyLinear` loaded through `loadLinear` — exactly the same pattern as
/// the vision tower's `embedderProjection`.
///
/// `@unchecked Sendable`: the projection + scalar config are immutable
/// `let` set once at load; needed so the audio encode hook can capture it.
final class Gemma4AudioEmbedder: @unchecked Sendable {
    /// `embedding_projection` — `[textHidden, embeddingDim]` (quantized in
    /// 4-bit checkpoints, loaded via `loadLinear`).
    let projection: AnyLinear
    /// The no-scale pre-projection norm's epsilon (the norm is unweighted).
    let normEps: Float
    /// Encoder output width feeding the projection (`output_proj_dims`).
    let embeddingDim: Int
    let dtype: DType

    init(
        projection: AnyLinear, normEps: Float, embeddingDim: Int, dtype: DType
    ) {
        self.projection = projection
        self.normEps = normEps
        self.embeddingDim = embeddingDim
        self.dtype = dtype
    }

    /// Load `<prefix>.embedding_projection.{weight,scales,biases}` from the
    /// checkpoint. Probes both the bare and `model.`-namespaced prefixes
    /// (VLM conversions may carry either). `prefix` defaults to
    /// `embed_audio` (the encoder-based VL audio tower); the Gemma 4 Omni
    /// encoder-free path reuses this same module for its vision projector
    /// (`embed_vision`, embeddingDim 3840) and audio projector
    /// (`embed_audio`, embeddingDim 640) — the embedder is generic in
    /// `embeddingDim`, so only the prefix + dim differ.
    static func load(
        weights: SafeTensorsBundle, embeddingDim: Int, normEps: Float,
        dtype: DType, quantization: ModelConfig.QuantizationConfig?,
        prefix: String = "embed_audio"
    ) throws -> Gemma4AudioEmbedder {
        let ea =
            weights.has("\(prefix).embedding_projection.weight")
            ? weights.prefixed("\(prefix).")
            : weights.prefixed("model.\(prefix).")
        let projection = try loadLinear(
            base: "embedding_projection", in: ea, quantization: quantization)
        return Gemma4AudioEmbedder(
            projection: projection, normEps: normEps,
            embeddingDim: embeddingDim, dtype: dtype)
    }

    /// RMSNorm-no-scale each `[nTokens, embeddingDim]` audio soft token,
    /// then project into the text hidden dim. Returns `[nTokens, textHidden]`.
    /// Mirrors `Gemma4VLVisionModel.embed`.
    func embed(_ encodings: Tensor, device: Device) -> Tensor {
        let nTokens = encodings.shape[0]
        let dim = embeddingDim
        let src = encodings.toFloatArray()
        // Unweighted RMSNorm — the embedder's pre-projection norm has no
        // learned scale.
        var normed = [Float](repeating: 0, count: nTokens * dim)
        for t in 0 ..< nTokens {
            var ss: Float = 0
            for d in 0 ..< dim { ss += src[t * dim + d] * src[t * dim + d] }
            let inv = 1.0 / (ss / Float(dim) + normEps).squareRoot()
            for d in 0 ..< dim { normed[t * dim + d] = src[t * dim + d] * inv }
        }
        let normedT = MediaUtils.makeTensor(
            from: normed, shape: [nTokens, dim], dtype: dtype, device: device)

        let cmd = device.makeCommandBuffer()
        // `callMany` handles a QuantizedLinear from a 4-bit checkpoint via
        // the dequant-gemm path.
        let projected = projection.callMany(
            normedT, t: nTokens, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        return projected
    }
}

// ─── Gemma 4 Omni encoder-free audio framing ─────────────────────────

/// Gemma 4 Omni's encoder-free audio front-end. Unlike the encoder-based
/// Gemma 4 VL audio tower (mel front-end + Conformer, above), the Omni
/// model does the 40 ms / 640-sample framing in its *processor*: raw
/// 16 kHz samples ARE the features. So the Swift side just reshapes the
/// waveform into `[nFrames, samplesPerToken]` — no mel, no STFT, no
/// Conformer — and feeds it straight to `embed_audio` (the shared
/// `Gemma4MultimodalEmbedder`: RMSNorm-no-scale + Linear 640→3840).
enum Gemma4OmniAudio {
    /// `audio_seq_length` cap from the processor config — 30 s at
    /// 40 ms/token (`750 · 640 = 480000` samples at 16 kHz).
    static let maxFrames = 750

    /// Slice `waveform` into non-overlapping `samplesPerToken` (640 = 40 ms
    /// at 16 kHz) frames, right zero-padding the final partial frame, capped
    /// at `maxFrames`. Returns `[nFrames, samplesPerToken]` in `dtype`.
    /// Always ≥ 1 frame (empty / sub-frame input → one zero-padded frame,
    /// matching a single `<|audio|>` placeholder).
    static func frame(
        _ waveform: [Float], samplesPerToken: Int,
        dtype: DType, device: Device
    ) -> Tensor {
        let maxSamples = maxFrames * samplesPerToken
        let samples =
            waveform.count > maxSamples
            ? Array(waveform[0 ..< maxSamples]) : waveform
        let nFrames = max(
            1, (samples.count + samplesPerToken - 1) / samplesPerToken)
        var rows = [Float](repeating: 0, count: nFrames * samplesPerToken)
        for i in 0 ..< samples.count { rows[i] = samples[i] }
        return MediaUtils.makeTensor(
            from: rows, shape: [nFrames, samplesPerToken],
            dtype: dtype, device: device)
    }
}
