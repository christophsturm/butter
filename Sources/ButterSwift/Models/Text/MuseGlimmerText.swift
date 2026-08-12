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
// Muse Glimmer text — the 30B dense decoder. The family enum
// (`enum MuseGlimmer`), variant protocol, and error type live in
// `Models/MuseGlimmer.swift` (the family root — read its header
// comment first, it documents every architectural divergence from a
// stock Gemma 3/4 sandwich-norm decoder: NoPE full layers, the
// weightless-but-Q-scaled qk-norm, sigmoid attention gating, the
// dual-eps sandwich norm, the non-centered final norm, the
// RMSNorm-instead-of-sqrt(hidden) embed path, and the pre-multiplied
// logit soft-cap).
//
// TEXT-ONLY: the checkpoint is a VL wrapper
// (`MuseGlimmerForConditionalGeneration`) even when we only run text.
// We read decoder params from `text_config` and pull tensors from the
// `language_model.`-prefixed sub-tree only — vision-tower / adapter /
// projection weights are never touched (`SafeTensorsBundle.prefixed`
// hides everything outside the prefix by construction).

import Foundation
import Metal

// MARK: - MuseGlimmerDense — 30B text decoder

public struct MuseGlimmerDense: MuseGlimmerVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256, prefillStepSize: 1024,
        temperature: 1.0, topP: 0.95, topK: 64,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config rawConfig: ModelConfig,
        weights rawWeights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> MuseGlimmerModel {
        // The checkpoint's decoder params live under `text_config`;
        // fall back to the flat config for a hypothetical text-only
        // `config.json` pointed straight at the language model.
        let config = rawConfig.subConfig("text_config") ?? rawConfig
        // The checkpoint's decoder tensors live under `language_model.`
        // — `Idefics3`/`FastVLM`/`LFM2` establish this pattern for VL
        // wrappers. After stripping the prefix, base paths match the
        // plain-text convention (`model.embed_tokens`, `model.layers.N`,
        // `model.norm.weight`, `lm_head.weight`) exactly.
        let weights = rawWeights.prefixed("language_model.")

        guard let hidden = config.hiddenSize,
            let nLayers = config.numLayers,
            let nHeads = config.numAttentionHeads,
            let intermediate = config.intermediateSize,
            let eps = config.rmsNormEps,
            let vocab = config.vocabSize
        else {
            throw MuseGlimmerError.missingConfig("hidden_size/num_hidden_layers/num_attention_heads/intermediate_size/rms_norm_eps/vocab_size")
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let headDim = config.headDim ?? (hidden / nHeads)
        let maxSeq = config.int("max_position_embeddings") ?? 131072
        let slidingWindow = config.int("sliding_window") ?? 2048
        // `post_norm_eps` — the two post-* sandwich norms use a much
        // tighter eps (1e-8 on the shipped checkpoint) than the two
        // pre-* norms (`rms_norm_eps`, 1e-5). Fall back to `eps` if a
        // checkpoint omits the field entirely.
        let postNormEps = config.float("post_norm_eps") ?? eps
        // `qk_scale_factor` multiplies the (weightless-normed) Q only —
        // see the family-root header. Default 1.0 degrades to a plain
        // weightless norm if a checkpoint omits it.
        let qkScaleFactor = Float(config.float("qk_scale_factor") ?? 1.0)
        // RoPE base lives under `rope_parameters.rope_theta` on this
        // checkpoint (not the flat `rope_theta` key `ModelConfig.ropeTheta`
        // reads) — fall back through both spellings.
        let ropeTheta = Float(
            (config.nested("rope_parameters")?["rope_theta"] as? Double)
                ?? config.ropeTheta ?? 500_000.0)
        // `output_multiplier` / `final_logit_softcapping` — folded
        // together into one host-side tanh pass at the LM head; see
        // `MuseGlimmerOps.softcap` below. `finalLogitSoftcap == 0`
        // disables soft-capping entirely (mirrors Gemma2/4's optional
        // pattern) — shouldn't happen on this checkpoint, but a
        // future revision might ship without it.
        let outputMultiplier = Float(config.float("output_multiplier") ?? 1.0)
        let finalLogitSoftcap = Float(config.float("final_logit_softcapping") ?? 0)
        let tieEmbed = config.tieWordEmbeddings
        let quant = config.quantization

        // `layer_types[i] == "sliding_attention"` vs `"full_attention"`
        // — full layers are NoPE (no rotary embedding at all) and run
        // an unbounded KV cache; sliding layers get RoPE(ropeTheta) and
        // a `sliding_window`-capped cache. Falls back to the documented
        // (3× sliding + 1× full) × 13 pattern if `layer_types` is absent.
        let layerTypeNames = config.raw["layer_types"] as? [String]
        func isSlidingLayer(_ i: Int) -> Bool {
            if let names = layerTypeNames, i < names.count {
                return names[i] == "sliding_attention"
            }
            return (i + 1) % 4 != 0
        }

        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant
        )

        // Resolve BEFORE the layer loop: when `model.embed_tokens` is
        // quantized, `embedTokens.weight.dtype` is the packed uint32
        // storage dtype, NOT the dequantized activation dtype every
        // downstream `Ops.*` call runs in (matmul/norm outputs land in
        // `scales.dtype`, typically f16/bf16). The per-layer synthesized
        // qk-norm weights below must match the latter.
        let activationDtype: DType
        if weights.isQuantized("model.embed_tokens"),
            let scales = try? weights.tensor(named: "model.embed_tokens.scales")
        {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        var layers: [MuseGlimmerLayer] = []
        layers.reserveCapacity(nLayers)

        for i in 0 ..< nLayers {
            let p = "model.layers.\(i)"
            let isSliding = isSlidingLayer(i)

            let qProj = try loadLinear(base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
            let kProj = try loadLinear(base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
            let vProj = try loadLinear(base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
            let oProj = try loadLinear(base: "\(p).self_attn.o_proj", in: weights, quantization: quant)
            let gateProj = try loadLinear(base: "\(p).self_attn.gate_proj", in: weights, quantization: quant)
            let mlpGateProj = try loadLinear(base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
            let upProj = try loadLinear(base: "\(p).mlp.up_proj", in: weights, quantization: quant)
            let downProj = try loadLinear(base: "\(p).mlp.down_proj", in: weights, quantization: quant)

            // The four sandwich norms are the "centered" (`1 + weight`)
            // RMSNorm — fold the `+1` into the loaded weight up front via
            // Gemma3Text's `loadGemmaRMSNorm` (same math, reused as-is).
            // Pre-* norms use `rms_norm_eps`; post-* norms use the much
            // tighter `post_norm_eps`.
            let inputNorm = try loadGemmaRMSNorm(base: "\(p).input_layernorm.weight", in: weights, eps: eps)
            let postAttnNorm = try loadGemmaRMSNorm(
                base: "\(p).post_attention_layernorm.weight", in: weights, eps: postNormEps)
            let preFFNorm = try loadGemmaRMSNorm(base: "\(p).pre_feedforward_layernorm.weight", in: weights, eps: eps)
            let postFFNorm = try loadGemmaRMSNorm(
                base: "\(p).post_feedforward_layernorm.weight", in: weights, eps: postNormEps)

            // Q/K norm is WEIGHTLESS in the reference model
            // (`with_scale=False`) — no `.weight` tensor exists in the
            // checkpoint for it. Q is additionally scaled by the scalar
            // `qkScaleFactor` (K is not). We fold both facts into the
            // existing `RMSNorm(weight:eps:)` kernel by synthesizing a
            // constant `[headDim]` weight tensor — `qkScaleFactor`
            // broadcast for Q, `1.0` broadcast for K — rather than
            // adding a separate elementwise-multiply dispatch.
            let qNormWeight = Tensor.empty(shape: [headDim], dtype: activationDtype, device: device)
            fillScalar(qNormWeight, scalar: qkScaleFactor, dtype: activationDtype)
            let kNormWeight = Tensor.empty(shape: [headDim], dtype: activationDtype, device: device)
            fillScalar(kNormWeight, scalar: 1.0, dtype: activationDtype)
            let qNorm = RMSNorm(weight: qNormWeight, eps: Float(eps))
            let kNorm = RMSNorm(weight: kNormWeight, eps: Float(eps))

            layers.append(
                MuseGlimmerLayer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    gateProj: gateProj,
                    mlpGateProj: mlpGateProj, upProj: upProj, downProj: downProj,
                    inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                    preFFNorm: preFFNorm, postFFNorm: postFFNorm,
                    qNorm: qNorm, kNorm: kNorm,
                    hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                    headDim: headDim, intermediate: intermediate,
                    ropeTheta: ropeTheta,
                    isSliding: isSliding
                ))
        }

        // The FINAL norm is NOT the centered variant — the reference
        // model builds it via the plain `MuseGlimmerRMSNorm` class
        // (`with_scale=True`, weight used directly, no `+1`). Load the
        // raw weight straight into `RMSNorm`, bypassing
        // `loadGemmaRMSNorm`'s fold.
        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"), eps: Float(eps))

        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: weights, quantization: quant)
        } else if let q = quant, weights.isQuantized("model.embed_tokens") {
            let t = try weights.quantizedTriplet("model.embed_tokens")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: bits, groupSize: q.groupSize
                ))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        // Weightless RMSNorm applied to the embedded token row — Muse
        // Glimmer does NOT use Gemma's `sqrt(hidden_size)` embed-scale
        // multiply. Synthesize a constant-`1.0` `[hidden]` weight and
        // run it through the same `RMSNorm` kernel as a norm (not a
        // `Ops.mul`).
        let embedNormWeight = Tensor.empty(shape: [hidden], dtype: activationDtype, device: device)
        fillScalar(embedNormWeight, scalar: 1.0, dtype: activationDtype)
        let embedNorm = RMSNorm(weight: embedNormWeight, eps: Float(eps))

        return MuseGlimmerModel(
            embedTokens: embedTokens, embedNorm: embedNorm, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            outputMultiplier: outputMultiplier, finalLogitSoftcap: finalLogitSoftcap,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxSeq: maxSeq, dtype: activationDtype,
            slidingWindow: slidingWindow,
            kvCacheKind: options.kvCache,
            kvEviction: options.kvEviction
        )
    }
}

// MARK: - MuseGlimmerLayer

public final class MuseGlimmerLayer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    /// Attention-output sigmoid gate — `attn_out * sigmoid(gateProj(x))`,
    /// `x` being the SAME `input_layernorm`-normalized hidden state that
    /// feeds Q/K/V (not the raw residual, not the SDPA output).
    let gateProj: AnyLinear
    let mlpGateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm, preFFNorm, postFFNorm: RMSNorm
    /// Weightless-in-spirit Q/K norm — see the family-root header and
    /// `MuseGlimmerDense.loadModel` for how the constant weight tensors
    /// (`qkScaleFactor` for Q, `1.0` for K) are synthesized.
    let qNorm, kNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    /// Standard `1/sqrt(head_dim)` SDPA scale — the extra
    /// `qkScaleFactor` is already baked into Q via `qNorm`'s weight, so
    /// no non-standard scale needs to reach `Ops.sdpaDecode`.
    let scale: Float
    /// Sliding-window (RoPE, `.window(2048)` KV cache) vs full-attention
    /// (NoPE, unbounded KV cache) — `layer_idx % 4 == 3` is full in the
    /// documented (3 sliding + 1 full) × 13 pattern.
    public let isSliding: Bool

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        gateProj: AnyLinear,
        mlpGateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        preFFNorm: RMSNorm, postFFNorm: RMSNorm,
        qNorm: RMSNorm, kNorm: RMSNorm,
        hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        intermediate: Int, ropeTheta: Float, isSliding: Bool
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.gateProj = gateProj
        self.mlpGateProj = mlpGateProj
        self.upProj = upProj
        self.downProj = downProj
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
        self.preFFNorm = preFFNorm
        self.postFFNorm = postFFNorm
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.hidden = hidden
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.intermediate = intermediate
        self.ropeTheta = ropeTheta
        self.isSliding = isSliding
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        for (k, v) in gateProj.parameters() { out.append(("self_attn.gate_proj.\(k)", v)) }
        for (k, v) in mlpGateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_attention_layernorm.\(k)", v)) }
        for (k, v) in preFFNorm.parameters() { out.append(("pre_feedforward_layernorm.\(k)", v)) }
        for (k, v) in postFFNorm.parameters() { out.append(("post_feedforward_layernorm.\(k)", v)) }
        return out
    }

    /// Single-token forward. `position` is the absolute sequence index
    /// (used for RoPE on sliding layers only — full layers are NoPE).
    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let xNorm = inputNorm(h, on: cmd)

        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // Per-head Q/K norm — Q's synthesized weight already carries
        // `qkScaleFactor`; K's is a plain `1.0` broadcast.
        var qNorm2D = Ops.rmsNormRows(
            q, weight: qNorm.weight, eps: qNorm.eps,
            nRows: nHeads, rowSize: headDim, on: cmd
        ).reshaped(to: [nHeads, headDim])
        var kNorm2D = Ops.rmsNormRows(
            k, weight: kNorm.weight, eps: kNorm.eps,
            nRows: nKVHeads, rowSize: headDim, on: cmd
        ).reshaped(to: [nKVHeads, headDim])

        // NoPE on full-attention layers: skip RoPE entirely — do not
        // apply even a zero/identity rotation, matching the reference
        // model's `position_embeddings=None` short-circuit.
        if isSliding {
            qNorm2D = Ops.rope(
                qNorm2D, position: position, headDim: headDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd)
            kNorm2D = Ops.rope(
                kNorm2D, position: position, headDim: headDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd)
        }

        cache.appendOnGPU(kFlat: kNorm2D, vFlat: v.reshaped(to: [nKVHeads, headDim]), on: cmd)
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: cache.length)
        let attnOut = Ops.sdpaDecode(
            q: qNorm2D, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd,
            sinkEnd: sinkEnd, windowStart: windowStart)

        // Sigmoid gate on the SDPA output, computed from `xNorm` (the
        // same input Q/K/V read) — NOT from the attention output and
        // NOT from the raw residual `h`.
        let gateLogits = gateProj(xNorm, on: cmd)
        let gatedAttn = Ops.sigmoidMul(
            attnOut.reshaped(to: [nHeads * headDim]), gateLogits, on: cmd)
        let oOut = oProj(gatedAttn, on: cmd)

        // Gemma-style post-norm placement: normalize the branch output
        // BEFORE adding to the residual (post_attention_layernorm uses
        // the tight `post_norm_eps`, folded into its weight already).
        let normedAttn = postAttnNorm(oOut, on: cmd)

        let postAttn: Tensor
        let mlpNorm: Tensor
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                h, normedAttn, weight: preFFNorm.weight, eps: preFFNorm.eps,
                nRows: 1, rowSize: hidden, on: cmd)
            postAttn = fused.residual
            mlpNorm = fused.normed
        } else {
            postAttn = Ops.add(h, normedAttn, on: cmd)
            mlpNorm = preFFNorm(postAttn, on: cmd)
        }

        // SwiGLU MLP (SiLU gate, not Gemma's GELU).
        let gate = mlpGateProj(mlpNorm, on: cmd)
        let up = upProj(mlpNorm, on: cmd)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        let normedMLP = postFFNorm(mlpOut, on: cmd)
        return Ops.add(postAttn, normedMLP, on: cmd)
    }
}

// MARK: - MuseGlimmerModel

public final class MuseGlimmerModel: LanguageModel {
    public let embedTokens: AnyEmbedding
    /// Weightless-with-synthesized-constant-`1.0` RMSNorm applied to
    /// the embedded token row — Muse Glimmer's replacement for Gemma's
    /// `sqrt(hidden_size)` embed-scale multiply.
    public let embedNorm: RMSNorm
    public let layers: [MuseGlimmerLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    /// `output_multiplier` (pre-scale) and `final_logit_softcapping`
    /// (cap) for the LM head's `cap · tanh(logits · multiplier / cap)`
    /// pass. `finalLogitSoftcap == 0` disables soft-capping.
    public let outputMultiplier: Float
    public let finalLogitSoftcap: Float

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let dtype: DType

    /// Window size for sliding-attention layers (three of every four).
    public let slidingWindow: Int

    public let kvCacheKind: KVCacheKind
    public let kvEviction: KVEviction

    init(
        embedTokens: AnyEmbedding, embedNorm: RMSNorm, layers: [MuseGlimmerLayer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        outputMultiplier: Float, finalLogitSoftcap: Float,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        vocab: Int, maxSeq: Int, dtype: DType,
        slidingWindow: Int,
        kvCacheKind: KVCacheKind, kvEviction: KVEviction
    ) {
        self.embedTokens = embedTokens
        self.embedNorm = embedNorm
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.outputMultiplier = outputMultiplier
        self.finalLogitSoftcap = finalLogitSoftcap
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.vocab = vocab
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.slidingWindow = slidingWindow
        self.kvCacheKind = kvCacheKind
        self.kvEviction = kvEviction
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() { out.append(("model.embed_tokens.\(k)", v)) }
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("model.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("model.norm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    /// Per-layer KV cache. Sliding layers use `.window(slidingWindow)`;
    /// full (NoPE) layers stay unbounded. A user-supplied `.window`
    /// override applies uniformly (same override semantics as Gemma 3).
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        var caches: [any LayerCacheProtocol] = []
        caches.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let layerEviction: KVEviction
            switch kvEviction {
            case .window:
                layerEviction = kvEviction
            case .unbounded:
                layerEviction =
                    layers[i].isSliding
                    ? .window(maxSize: min(slidingWindow, cap), keep: 0)
                    : .unbounded
            }
            switch kvCacheKind {
            case .raw:
                caches.append(
                    KVCache(
                        nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, eviction: layerEviction, device: device
                    ))
            default:
                preconditionFailure(
                    "MuseGlimmer: only .raw KV cache supported today; got \(kvCacheKind)")
            }
        }
        return caches
    }

    /// Muse Glimmer's soft-capped LM head needs the logits on the host
    /// (see `MuseGlimmerOps.softcap`), so — unlike Gemma3's
    /// continuous-`cmd` model, which can queue embed→layers→head as one
    /// chain and let the CALLER's own eventual commit submit everything
    /// — this forward NEVER touches the caller's `cmd` for compute. It
    /// mirrors `Gemma4Text.forward(tokenId:...)`'s private-buffer
    /// strategy instead: every stage runs on a `device.makeCommandBuffer()`
    /// of its own, committed+waited internally, so `h` (and finally
    /// `logits`) is always host-visible before the NEXT private buffer
    /// reads it. `cmd` stays pristine throughout; the caller's later
    /// `cmd.commit()` (in `forwardSample` / the `LanguageModel` default
    /// `forward(...)`) is a true no-op over the already-resident return
    /// value — never a double-commit.
    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let tap = InspectTap.fromEnvironment

        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        var workCmd = device.makeCommandBuffer()
        let h0 = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        var h = embedNorm(h0, on: workCmd)
        if tap.shouldDump(layer: -1) {
            workCmd.commit()
            workCmd.waitUntilCompleted()
            _ = tap.dumpLayerBoundary(
                h, label: "embed_norm", layer: -1, cmd: device.makeCommandBuffer(), device: device)
            workCmd = device.makeCommandBuffer()
        }

        for (i, layer) in layers.enumerated() {
            h = layer.forward(
                h, position: position,
                cache: caches[i] as! any KVCacheProtocol,
                on: workCmd, device: device)
            if tap.shouldDump(layer: i) {
                workCmd.commit()
                workCmd.waitUntilCompleted()
                _ = tap.dumpLayerBoundary(
                    h, label: "layer_out", layer: i, cmd: device.makeCommandBuffer(), device: device)
                workCmd = device.makeCommandBuffer()
            }
        }

        let normed = finalNorm(h, on: workCmd)
        let rawLogits = lmHead(normed, on: workCmd)

        if finalLogitSoftcap > 0 {
            // Commits + waits `workCmd` internally, then does the
            // tanh-softcap pass host-side. Never touches `cmd`.
            return MuseGlimmerOps.softcap(
                rawLogits, multiplier: outputMultiplier, cap: finalLogitSoftcap, on: workCmd)
        }
        // No soft-cap (a hypothetical checkpoint revision that omits
        // `final_logit_softcapping`) — still resolve on our own private
        // buffer so the return value is resident either way.
        workCmd.commit()
        workCmd.waitUntilCompleted()
        return rawLogits
    }

    /// Multi-token forward — prefill fast path. Loops `forward(tokenId:)`
    /// per row, same as Gemma3Model's first-light override. Muse
    /// Glimmer mixes sliding-window vs full attention every 4th layer
    /// (like Gemma 3/4), so collapsing to one `sdpaMulti(causal: true)`
    /// dispatch per layer needs the same per-layer sliding/full split
    /// those families haven't done yet either. No perf work this wave —
    /// correctness + baseline numbers only.
    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(!tokenIds.isEmpty, "MuseGlimmerModel.forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, tok) in tokenIds.enumerated() {
            logits = forward(
                tokenId: tok, position: position + i,
                caches: caches, on: cmd, device: device)
        }
        return logits
    }
}

// MARK: - Soft-cap

enum MuseGlimmerOps {
    /// `cap · tanh(x · multiplier / cap)` — Gemma2-style logit
    /// soft-cap with an extra pre-multiplier folded into the `tanh`
    /// argument (`Gemma4Ops.softcap`'s formula generalized with
    /// `multiplier`; `multiplier == 1.0` reduces to the Gemma2/4 case
    /// exactly). Host-side after `cmd` completes — same rationale as
    /// `Gemma4Ops.softcap`: this runs once per token on the `[vocab]`
    /// logit vector the sampler reads back anyway, so there's no
    /// benefit to a GPU dispatch. Commits + waits `cmd`; the caller's
    /// own commit becomes a no-op over the now-resident tensor.
    static func softcap(
        _ logits: Tensor, multiplier: Float, cap: Float,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        cmd.commit()
        cmd.waitUntilCompleted()
        let n = logits.elementCount
        let ptr = logits.buffer.contents().advanced(by: logits.offset)
        let mOverCap = multiplier / cap
        switch logits.dtype {
        case .f32:
            let p = ptr.bindMemory(to: Float.self, capacity: n)
            for i in 0 ..< n { p[i] = cap * tanh(p[i] * mOverCap) }
        case .f16:
            let p = ptr.bindMemory(to: UInt16.self, capacity: n)
            for i in 0 ..< n {
                let f = halfBitsToFloatForTest(p[i])
                p[i] = floatToHalfBitsForTest(cap * tanh(f * mOverCap))
            }
        case .bf16:
            let p = ptr.bindMemory(to: UInt16.self, capacity: n)
            for i in 0 ..< n {
                let f = bf16BitsToFloatForTest(p[i])
                p[i] = floatToBf16BitsForTest(cap * tanh(f * mOverCap))
            }
        default:
            fatalError("MuseGlimmerOps.softcap: unsupported dtype \(logits.dtype)")
        }
        return logits
    }
}
