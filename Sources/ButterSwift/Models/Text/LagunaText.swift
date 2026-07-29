// Copyright 2026 Tom Turney (@TheTom)
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
// Laguna text backbone (S-2.1 / XS-2.1; config-driven).
//
// Compose existing primitives:
//   • Hybrid iSWA (full every 4th layer, else SWA-512) + per-layer Q heads
//   • Softplus per-head attention gate applied before o_proj
//   • QK-RMSNorm pre-RoPE
//   • Full layers: partial rotary 0.5 + YaRN-scale RoPE (theta 5e5)
//   • Sliding layers: full rotary, plain RoPE (theta 1e4)
//   • Layer 0 dense SwiGLU; remaining layers MoE via MoELayer
//     (sigmoidBiasedTopK, top-K + shared expert, routed_scaling from config)
//
// GPU MoE router fast path is topK=8 + softmax-only. Laguna uses
// sigmoid + bias (S top-10, XS top-8), so routing stays on the CPU path
// (same class as Hy3 when the GPU envelope does not match). Prefill is
// still per-token forwardMulti → MoELayer.decode; wire decodeMany after
// smoke.
//
// NVFP4 board quant: switch_mlp stacked experts via expert-indexed
// `iron_nvfp4_*` kernels; attention/router/lm_head stay BF16.

import Foundation
import Metal
import IronSwift

// ─── LagunaTextConfig ────────────────────────────────────────────────

/// Static text-decoder shape decoded from a Laguna `config.json`.
struct LagunaTextConfig {
    let nLayers: Int
    let hidden: Int
    let vocab: Int
    let maxSeq: Int
    let rmsNormEps: Float

    /// Default / full-attention Q head count (48 on S-2.1).
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int

    /// Per-layer Q head overrides (sliding layers → 72).
    let perLayerHeads: [Int: Int]
    /// Window size for sliding layers; `0` = full attention.
    let perLayerSlidingWindow: [Int: Int]
    /// Partial rotary factor per layer (full=0.5, swa=1.0).
    let perLayerPartialRotaryFactor: [Int: Float]
    /// Per-layer RoPE base θ (full=5e5, swa=1e4).
    let perLayerRopeTheta: [Int: Float]
    /// Layer types aligned 1-1 with the stack.
    let layerTypes: [String]

    // Full-attention YaRN (applied only when layer type is full).
    let fullRopeTheta: Float
    let fullPartialRotary: Float
    let yarnFactor: Float
    let yarnOriginalContext: Int
    let yarnBetaSlow: Float
    let yarnBetaFast: Float
    let yarnAttentionFactor: Float

    // Sliding-attention RoPE defaults.
    let slidingRopeTheta: Float
    let slidingPartialRotary: Float
    let slidingWindow: Int

    // MoE / dense FFN.
    let mlpOnlyLayers: Set<Int>
    let decoderSparseStep: Int
    let nExperts: Int
    let nExpertsPerToken: Int
    let moeIntermediate: Int
    let sharedExpertIntermediate: Int
    let denseIntermediate: Int
    let routerNormTopK: Bool
    let routerScalingFactor: Float
    /// Softplus headwise gate present on every attention layer.
    let useSoftplusHeadwiseAttnGate: Bool
    let tieWordEmbeddings: Bool
    let bosTokenId: Int?
    let eosTokenIds: [Int]

    static func decode(_ tc: ModelConfig) throws -> LagunaTextConfig {
        guard
            let nLayers = tc.int("num_hidden_layers"),
            let hidden = tc.int("hidden_size"),
            let vocab = tc.int("vocab_size"),
            let nHeads = tc.int("num_attention_heads")
        else {
            throw LagunaError.missingConfig("core shape (layers/hidden/vocab/heads)")
        }
        let nKVHeads = tc.int("num_key_value_heads") ?? nHeads
        let headDim = tc.int("head_dim") ?? (hidden / nHeads)
        let maxSeq =
            tc.int("max_position_embeddings")
            ?? tc.int("model_max_length") ?? 1_048_576

        guard let layerTypes = tc.raw["layer_types"] as? [String], layerTypes.count == nLayers
        else {
            throw LagunaError.missingConfig("layer_types (length \(nLayers))")
        }

        let headsPerLayer = tc.intArray("num_attention_heads_per_layer")
        if let hpl = headsPerLayer, hpl.count != nLayers {
            throw LagunaError.unsupportedConfig(
                "num_attention_heads_per_layer length \(hpl.count) ≠ nLayers \(nLayers)")
        }

        let swaSize = tc.int("sliding_window") ?? 512

        // Nested rope_parameters.{full,sliding}_attention (Gemma-4 style).
        let ropeParams = tc.nested("rope_parameters") ?? [:]
        let fullRP = ropeParams["full_attention"] as? [String: Any] ?? [:]
        let swaRP = ropeParams["sliding_attention"] as? [String: Any] ?? [:]

        func f64(_ d: [String: Any], _ k: String, _ def: Double) -> Float {
            if let v = d[k] as? Double { return Float(v) }
            if let v = d[k] as? Int { return Float(v) }
            return Float(def)
        }
        func i64(_ d: [String: Any], _ k: String, _ def: Int) -> Int {
            if let v = d[k] as? Int { return v }
            if let v = d[k] as? Double { return Int(v) }
            return def
        }

        let fullTheta = f64(fullRP, "rope_theta", 500_000)
        let fullPartial = f64(fullRP, "partial_rotary_factor", 0.5)
        let yarnFactor = f64(fullRP, "factor", 128)
        let yarnOrig = i64(fullRP, "original_max_position_embeddings", 8192)
        let yarnBetaSlow = f64(fullRP, "beta_slow", 1.0)
        let yarnBetaFast = f64(fullRP, "beta_fast", 32.0)
        let yarnAttnFactor = f64(fullRP, "attention_factor", 1.0)

        let swaTheta = f64(swaRP, "rope_theta", 10_000)
        let swaPartial = f64(swaRP, "partial_rotary_factor", 1.0)

        var perLayerHeads: [Int: Int] = [:]
        var perLayerSWA: [Int: Int] = [:]
        var perLayerPartial: [Int: Float] = [:]
        var perLayerTheta: [Int: Float] = [:]

        for (i, lt) in layerTypes.enumerated() {
            switch lt {
            case "full_attention":
                perLayerSWA[i] = 0
                perLayerPartial[i] = fullPartial
                perLayerTheta[i] = fullTheta
                if let hpl = headsPerLayer { perLayerHeads[i] = hpl[i] }
            case "sliding_attention":
                perLayerSWA[i] = swaSize
                perLayerPartial[i] = swaPartial
                perLayerTheta[i] = swaTheta
                if let hpl = headsPerLayer { perLayerHeads[i] = hpl[i] }
            default:
                throw LagunaError.unsupportedLayerType(lt)
            }
        }

        let mlpOnly = Set(tc.intArray("mlp_only_layers") ?? [0])
        let sparseStep = tc.int("decoder_sparse_step") ?? 1

        let nExperts = tc.int("num_experts") ?? 256
        let topK = tc.int("num_experts_per_tok") ?? 10
        let moeInter = tc.int("moe_intermediate_size") ?? 1024
        let sharedInter =
            tc.int("shared_expert_intermediate_size") ?? moeInter
        let denseInter = tc.int("intermediate_size") ?? (hidden * 4)

        // gating: "per-head" | true | false
        let useGate: Bool = {
            if let s = tc.string("gating") {
                return s == "per-head" || s == "per_head" || s == "true"
            }
            return tc.bool("gating") ?? true
        }()

        return LagunaTextConfig(
            nLayers: nLayers,
            hidden: hidden,
            vocab: vocab,
            maxSeq: maxSeq,
            rmsNormEps: Float(tc.float("rms_norm_eps") ?? 1e-6),
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim,
            perLayerHeads: perLayerHeads,
            perLayerSlidingWindow: perLayerSWA,
            perLayerPartialRotaryFactor: perLayerPartial,
            perLayerRopeTheta: perLayerTheta,
            layerTypes: layerTypes,
            fullRopeTheta: fullTheta,
            fullPartialRotary: fullPartial,
            yarnFactor: yarnFactor,
            yarnOriginalContext: yarnOrig,
            yarnBetaSlow: yarnBetaSlow,
            yarnBetaFast: yarnBetaFast,
            yarnAttentionFactor: yarnAttnFactor,
            slidingRopeTheta: swaTheta,
            slidingPartialRotary: swaPartial,
            slidingWindow: swaSize,
            mlpOnlyLayers: mlpOnly,
            decoderSparseStep: sparseStep,
            nExperts: nExperts,
            nExpertsPerToken: topK,
            moeIntermediate: moeInter,
            sharedExpertIntermediate: sharedInter,
            denseIntermediate: denseInter,
            routerNormTopK: tc.bool("norm_topk_prob") ?? true,
            routerScalingFactor: Float(
                tc.float("moe_routed_scaling_factor") ?? 2.5),
            useSoftplusHeadwiseAttnGate: useGate,
            tieWordEmbeddings: tc.bool("tie_word_embeddings") ?? false,
            bosTokenId: tc.bosTokenId,
            eosTokenIds: tc.eosTokenIds)
    }

    /// Whether layer `i` uses MoE (vs dense SwiGLU).
    func usesMoE(layer i: Int) -> Bool {
        guard !mlpOnlyLayers.contains(i) else { return false }
        return (i + 1) % decoderSparseStep == 0
    }

    func isSliding(layer i: Int) -> Bool {
        (perLayerSlidingWindow[i] ?? 0) > 0
    }

    func nHeads(layer i: Int) -> Int {
        perLayerHeads[i] ?? nHeads
    }
}

// ─── LagunaHybrid ────────────────────────────────────────────────────

public enum LagunaHybrid: LagunaVariant {
    public static var availableCapabilities: Set<Capability> { [.textIn, .textOut] }

    public static func loadModel(
        config: ModelConfig,
        weights inputWeights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LagunaModel {
        let tc = try LagunaTextConfig.decode(config)
        let quant = config.quantization
        let eps = tc.rmsNormEps
        let dtype: DType = quant != nil ? .bf16 : .f16

        // mlx-community Laguna conversions wrap the text tower under
        // `language_model.*` (same layout as a VL text backbone) even
        // though the model is text-only. Strip that prefix when present.
        let weights: SafeTensorsBundle = {
            if inputWeights.has("language_model.model.embed_tokens.weight")
                || inputWeights.has("language_model.model.embed_tokens")
                || inputWeights.isQuantized("language_model.model.embed_tokens")
            {
                return inputWeights.prefixed("language_model.")
            }
            return inputWeights
        }()

        // Cap context: 1M advertized is not practical for first load.
        // Override with LoadOptions.maxContextLength or BUTTER_LAGUNA_MAX_SEQ.
        let defaultMaxSeq: Int = {
            if let s = ProcessInfo.processInfo.environment["BUTTER_LAGUNA_MAX_SEQ"],
                let n = Int(s), n > 0
            {
                return n
            }
            return 8192
        }()
        let maxSeq =
            options.maxContextLength
            ?? min(tc.maxSeq, defaultMaxSeq)
        if maxSeq < tc.maxSeq {
            fputs(
                "[laguna] context clamped to \(maxSeq) (checkpoint advertises \(tc.maxSeq)); "
                    + "raise with LoadOptions.maxContextLength or BUTTER_LAGUNA_MAX_SEQ\n",
                stderr)
        }

        // NVFP4 board quant uses switch_mlp stacked weights (supported).
        // Reject only the compressed-tensors packed layout we don't load.
        if weights.has("model.layers.1.mlp.experts.0.gate_proj.weight_packed") {
            throw LagunaError.unsupportedConfig(
                "compressed-tensors weight_packed experts not supported; "
                    + "use mlx affine4 or Poolside NVFP4 switch_mlp layout")
        }

        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: tc.hidden, quantization: quant)

        var layers: [LagunaLayer] = []
        layers.reserveCapacity(tc.nLayers)
        for i in 0 ..< tc.nLayers {
            layers.append(
                try buildLagunaLayer(
                    index: i, tc: tc, weights: weights,
                    quant: quant, eps: eps, dtype: dtype, device: device))
        }
        // (device threaded into MoE bias convert)

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"),
            eps: eps)

        let lmHead: AnyLinear
        var lmHeadMxfp8: Mxfp8Linear? = nil
        // Prefer BF16/quantized stock head for quality. MXFP8 sidecar enables
        // certified decode prune (DEFAULT ON when present; escape with
        // BUTTER_LAGUNA_LM_HEAD_PRUNE=0) or full replace
        // (BUTTER_LAGUNA_LM_HEAD_MXFP8=1).
        if let mx = try? lagunaLoadMxfp8LmHead(
            modelDir: weights.directory, hidden: tc.hidden, vocab: tc.vocab,
            device: device)
        {
            lmHeadMxfp8 = mx
            fputs("[laguna] lm_head: MXFP8 sidecar present\n", stderr)
        }
        if ProcessInfo.processInfo.environment["BUTTER_LAGUNA_LM_HEAD_MXFP8"] == "1",
            let mx = lmHeadMxfp8
        {
            lmHead = AnyLinear(mx)
            fputs("[laguna] lm_head: using MXFP8 full replace\n", stderr)
        } else if weights.has("lm_head.weight") || weights.isQuantized("lm_head") {
            lmHead = try loadLinear(
                base: "lm_head", in: weights, quantization: quant)
        } else if tc.tieWordEmbeddings, !weights.isQuantized("model.embed_tokens") {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        } else {
            throw LagunaError.missingTensor(
                "lm_head.weight (tied quantized embed not supported)")
        }
        if LagunaLmHeadPrune.enabled, lmHeadMxfp8 != nil,
            (lmHead.inner as? Linear)?.weight.dtype == .bf16
        {
            fputs(
                "[laguna] lm_head: certified 2-pass prune DEFAULT ON "
                    + "(BUTTER_LAGUNA_LM_HEAD_PRUNE=0 to disable)\n",
                stderr)
        }
        // YaRN bounds from the *rotated* dim (partial_rotary 0.5 → 64 on
        // S-2.1). attn_factor comes from config.attention_factor (1.4852),
        // not the mscale/mscale_all_dim ratio (both 1.0 in the HF file).
        let rotaryDimFull = {
            var d = Int((Float(tc.headDim) * tc.fullPartialRotary).rounded())
            if d % 2 != 0 { d -= 1 }
            return max(2, min(tc.headDim, d))
        }()
        let yarnBounds = Ops.RoPEYaRN.from(
            headDim: rotaryDimFull,
            thetaBase: tc.fullRopeTheta,
            factor: tc.yarnFactor,
            betaFast: tc.yarnBetaFast,
            betaSlow: tc.yarnBetaSlow,
            originalMaxPosition: Float(tc.yarnOriginalContext),
            mscale: 1,
            mscaleAllDim: 1)
        let yarn = Ops.RoPEYaRN(
            factor: yarnBounds.factor,
            low: yarnBounds.low,
            high: yarnBounds.high,
            attnFactor: tc.yarnAttentionFactor)

        return LagunaModel(
            embedTokens: embedTokens,
            layers: layers,
            finalNorm: finalNorm,
            lmHead: lmHead,
            lmHeadMxfp8: lmHeadMxfp8,
            textConfig: tc,
            yarn: yarn,
            hidden: tc.hidden,
            nLayers: tc.nLayers,
            nHeads: tc.nHeads,
            nKVHeads: tc.nKVHeads,
            headDim: tc.headDim,
            vocab: tc.vocab,
            maxSeq: maxSeq,
            dtype: dtype,
            slidingWindow: tc.slidingWindow)
    }
}

// ─── Layer build ─────────────────────────────────────────────────────

private func buildLagunaLayer(
    index i: Int,
    tc: LagunaTextConfig,
    weights: SafeTensorsBundle,
    quant: ModelConfig.QuantizationConfig?,
    eps: Float,
    dtype: DType,
    device: Device
) throws -> LagunaLayer {
    let p = "model.layers.\(i)"
    let nHeads = tc.nHeads(layer: i)
    let isSliding = tc.isSliding(layer: i)
    let ropeTheta = tc.perLayerRopeTheta[i] ?? (isSliding ? tc.slidingRopeTheta : tc.fullRopeTheta)
    let partial = tc.perLayerPartialRotaryFactor[i]
        ?? (isSliding ? tc.slidingPartialRotary : tc.fullPartialRotary)
    var rotaryDim = Int((Float(tc.headDim) * partial).rounded())
    if rotaryDim % 2 != 0 { rotaryDim -= 1 }
    rotaryDim = max(2, min(tc.headDim, rotaryDim))

    let qProj = try loadLinear(
        base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
    let kProj = try loadLinear(
        base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
    let vProj = try loadLinear(
        base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
    let oProj = try loadLinear(
        base: "\(p).self_attn.o_proj", in: weights, quantization: quant)

    let gProj: AnyLinear?
    if tc.useSoftplusHeadwiseAttnGate {
        gProj = try loadLinear(
            base: "\(p).self_attn.g_proj", in: weights, quantization: quant)
    } else {
        gProj = nil
    }

    let qNorm = RMSNorm(
        weight: try weights.tensor(named: "\(p).self_attn.q_norm.weight"),
        eps: eps)
    let kNorm = RMSNorm(
        weight: try weights.tensor(named: "\(p).self_attn.k_norm.weight"),
        eps: eps)
    let inputNorm = RMSNorm(
        weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
        eps: eps)
    let postAttnNorm = RMSNorm(
        weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
        eps: eps)

    let ffn: LagunaFFN
    if tc.usesMoE(layer: i) {
        ffn = .moe(
            try buildLagunaMoE(
                prefix: "\(p).mlp", tc: tc, weights: weights, quant: quant,
                actDtype: dtype, device: device, layerIndex: i))
    } else {
        let gate = try loadLinear(
            base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
        let up = try loadLinear(
            base: "\(p).mlp.up_proj", in: weights, quantization: quant)
        let down = try loadLinear(
            base: "\(p).mlp.down_proj", in: weights, quantization: quant)
        ffn = .dense(LagunaDenseMLP(gate: gate, up: up, down: down, hidden: tc.hidden))
    }

    return LagunaLayer(
        qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
        gProj: gProj, qNorm: qNorm, kNorm: kNorm,
        inputNorm: inputNorm, postAttnNorm: postAttnNorm,
        ffn: ffn,
        hidden: tc.hidden, nHeads: nHeads, nKVHeads: tc.nKVHeads,
        headDim: tc.headDim, rotaryDim: rotaryDim,
        ropeTheta: ropeTheta, isSliding: isSliding,
        isFullYarn: !isSliding,
        dtype: dtype, device: device)
}

private func buildLagunaMoE(
    prefix p: String,
    tc: LagunaTextConfig,
    weights: SafeTensorsBundle,
    quant: ModelConfig.QuantizationConfig?,
    actDtype: DType,
    device: Device,
    layerIndex: Int = -1
) throws -> MoELayer {
    // Router: hidden → nExperts. HF/GGUF use `mlp.gate`; some mlx
    // conversions (Laguna XS 4-bit) nest the projection as `mlp.gate.proj`.
    let gateBase: String = {
        if weights.has("\(p).gate.weight") || weights.isQuantized("\(p).gate") {
            return "\(p).gate"
        }
        if weights.has("\(p).gate.proj.weight") || weights.isQuantized("\(p).gate.proj") {
            return "\(p).gate.proj"
        }
        return "\(p).gate"
    }()
    let gate = try loadLinear(
        base: gateBase, in: weights, quantization: quant)

    // Prefer stacked mlx `switch_mlp` (from GGUF→mlx convert). Fall back
    // to per-expert HF keys when present.
    let gateProj: [AnyLinear]
    let upProj: [AnyLinear]
    let downProj: [AnyLinear]
    let stackedInt4: MoELayer.StackedInt4Experts?
    let stackedNvfp4: MoELayer.StackedNvfp4Experts?
    let isNvfp4Mode = (quant?.mode == "nvfp4")
        || weights.isNvfp4("\(p).switch_mlp.gate_proj")
    if weights.has("\(p).switch_mlp.gate_proj.weight")
        || weights.isQuantized("\(p).switch_mlp.gate_proj")
        || weights.isNvfp4("\(p).switch_mlp.gate_proj")
    {
        if isNvfp4Mode {
            // NVFP4: expert-indexed path uses stacked tensors. MoELayer still
            // requires nExperts-long gate/up/down arrays, but Path B never
            // calls them when stackedNvfp4 is set.
            //
            // CRITICAL: do NOT allocate one empty [mid,hidden] per expert —
            // that was ~2 MB × 256 × 2 (gate+down) × 39 layers ≈ 40 GB of
            // dead unified memory and was thrashing decode to ~50 t/s.
            // Share one stub linear across all expert slots (and layers
            // via the static cache below).
            stackedNvfp4 = lagunaTryLoadStackedNvfp4(
                base: "\(p).switch_mlp", weights: weights,
                numExperts: tc.nExperts, moeIntermediate: tc.moeIntermediate,
                hidden: tc.hidden, actDtype: actDtype)
            stackedInt4 = nil
            let stubs = lagunaNvfp4ExpertStubs(
                moeIntermediate: tc.moeIntermediate, hidden: tc.hidden,
                dtype: actDtype, device: device)
            gateProj = Array(repeating: stubs.gateUp, count: tc.nExperts)
            upProj = gateProj
            downProj = Array(repeating: stubs.down, count: tc.nExperts)
            if stackedNvfp4 == nil {
                throw LagunaError.unsupportedConfig(
                    "NVFP4 switch_mlp present but stacked load failed for \(p)")
            }
        } else {
            gateProj = try lagunaSliceStackedExperts(
                base: "\(p).switch_mlp.gate_proj", in: weights,
                numExperts: tc.nExperts, outDim: tc.moeIntermediate, inDim: tc.hidden,
                quant: quant)
            upProj = try lagunaSliceStackedExperts(
                base: "\(p).switch_mlp.up_proj", in: weights,
                numExperts: tc.nExperts, outDim: tc.moeIntermediate, inDim: tc.hidden,
                quant: quant)
            downProj = try lagunaSliceStackedExperts(
                base: "\(p).switch_mlp.down_proj", in: weights,
                numExperts: tc.nExperts, outDim: tc.hidden, inDim: tc.moeIntermediate,
                quant: quant)
            stackedInt4 = lagunaTryLoadStackedInt4(
                base: "\(p).switch_mlp", weights: weights, quant: quant,
                numExperts: tc.nExperts, moeIntermediate: tc.moeIntermediate,
                hidden: tc.hidden)
            stackedNvfp4 = nil
        }
    } else {
        var g: [AnyLinear] = []
        var u: [AnyLinear] = []
        var d: [AnyLinear] = []
        g.reserveCapacity(tc.nExperts)
        u.reserveCapacity(tc.nExperts)
        d.reserveCapacity(tc.nExperts)
        for e in 0 ..< tc.nExperts {
            let eb = "\(p).experts.\(e)"
            g.append(
                try loadLinear(
                    base: "\(eb).gate_proj", in: weights, quantization: quant))
            u.append(
                try loadLinear(
                    base: "\(eb).up_proj", in: weights, quantization: quant))
            d.append(
                try loadLinear(
                    base: "\(eb).down_proj", in: weights, quantization: quant))
        }
        gateProj = g
        upProj = u
        downProj = d
        stackedInt4 = nil
        stackedNvfp4 = nil
    }

    // Bias for selection: HF key or convert alias.
    let biasCandidates = [
        "\(p).experts.e_score_correction_bias",
        "\(p).gate.e_score_correction_bias",
        "\(p).e_score_correction_bias",
    ]
    guard let biasName = biasCandidates.first(where: { weights.has($0) }) else {
        throw LagunaError.missingTensor(biasCandidates[0])
    }
    let biasTensor = try weights.tensor(named: biasName)
    let expertBias = biasTensor.toFloatArray()
    precondition(
        expertBias.count == tc.nExperts,
        "Laguna: e_score_correction_bias count \(expertBias.count) ≠ nExperts \(tc.nExperts)")

    let router = MoERouter(
        nExperts: tc.nExperts,
        topK: tc.nExpertsPerToken,
        gatingMode: .sigmoidBiasedTopK,
        normTopKProb: tc.routerNormTopK,
        expertBias: expertBias,
        routedScalingFactor: tc.routerScalingFactor)

    let sharedGate: AnyLinear
    let sharedUp: AnyLinear
    let sharedDown: AnyLinear
    if isNvfp4Mode || weights.isNvfp4("\(p).shared_expert.gate_proj") {
        sharedGate = try lagunaLoadNvfp4Linear(
            base: "\(p).shared_expert.gate_proj", in: weights)
        sharedUp = try lagunaLoadNvfp4Linear(
            base: "\(p).shared_expert.up_proj", in: weights)
        sharedDown = try lagunaLoadNvfp4Linear(
            base: "\(p).shared_expert.down_proj", in: weights)
    } else {
        sharedGate = try loadLinear(
            base: "\(p).shared_expert.gate_proj", in: weights, quantization: quant)
        sharedUp = try loadLinear(
            base: "\(p).shared_expert.up_proj", in: weights, quantization: quant)
        sharedDown = try loadLinear(
            base: "\(p).shared_expert.down_proj", in: weights, quantization: quant)
    }

    // GPU Path B needs expert bias in the activation dtype (bf16/f16).
    let expertBiasGPU: Tensor = {
        if biasTensor.dtype == actDtype { return biasTensor }
        let host = biasTensor.toFloatArray()
        let out = Tensor.empty(shape: [tc.nExperts], dtype: actDtype, device: device)
        switch actDtype {
        case .f16: out.copyIn(from: host.map { Float16($0) })
        case .bf16:
            out.copyIn(from: host.map { v -> UInt16 in
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            })
        case .f32: out.copyIn(from: host)
        default: fatalError("Laguna: unsupported act dtype \(actDtype) for expert bias")
        }
        return out
    }()

    // Path B: GPU top-k + host experts (default ON). Opt out with
    // BUTTER_LAGUNA_PATHB=0 to force fully-host routing.
    let pathBGPU =
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_PATHB"] == "0"
            ? nil : expertBiasGPU

    return MoELayer(
        gate: gate,
        gateProj: gateProj, upProj: upProj, downProj: downProj,
        sharedGateProj: sharedGate,
        sharedUpProj: sharedUp,
        sharedDownProj: sharedDown,
        router: router,
        hidden: tc.hidden,
        stackedInt4Experts: stackedInt4,
        stackedNvfp4Experts: stackedNvfp4,
        expertBiasGPU: pathBGPU,
        layerIndex: layerIndex)
}

private func lagunaLoadNvfp4Linear(
    base: String, in weights: SafeTensorsBundle,
    blockSize: Int = 16, global: Float = 1.0
) throws -> AnyLinear {
    let w = try weights.tensor(named: "\(base).weight")
    let s = try weights.tensor(named: "\(base).scales")
    precondition(w.dtype == .u32 && s.dtype == .u8)
    return AnyLinear(
        Nvfp4Linear(weight: w, scales: s, blockSize: blockSize, global: global))
}

/// Process-wide stub linears for NVFP4 Path B (stacked experts only).
/// One tiny `[1,1]` weight pair is enough — host/hybrid Path B must not
/// touch these when `stackedNvfp4` is live. Sharing across all 39×256
/// slots keeps dead GPU memory at ~bytes instead of ~40 GB.
private enum LagunaNvfp4ExpertStubCache {
    nonisolated(unsafe) static var gateUp: AnyLinear?
    nonisolated(unsafe) static var down: AnyLinear?
    static let lock = NSLock()
}

private func lagunaNvfp4ExpertStubs(
    moeIntermediate: Int, hidden: Int, dtype: DType, device: Device
) -> (gateUp: AnyLinear, down: AnyLinear) {
    LagunaNvfp4ExpertStubCache.lock.lock()
    defer { LagunaNvfp4ExpertStubCache.lock.unlock() }
    if let g = LagunaNvfp4ExpertStubCache.gateUp,
        let d = LagunaNvfp4ExpertStubCache.down
    {
        return (g, d)
    }
    // Minimal placeholders — never executed on the NVFP4 stacked path.
    // Shape is intentionally tiny; size is only for MoELayer count checks.
    _ = moeIntermediate
    _ = hidden
    let gu = AnyLinear(
        Linear(weight: Tensor.empty(shape: [1, 1], dtype: dtype, device: device)))
    let dn = AnyLinear(
        Linear(weight: Tensor.empty(shape: [1, 1], dtype: dtype, device: device)))
    LagunaNvfp4ExpertStubCache.gateUp = gu
    LagunaNvfp4ExpertStubCache.down = dn
    return (gu, dn)
}

private func lagunaTryLoadStackedNvfp4(
    base: String, weights: SafeTensorsBundle,
    numExperts: Int, moeIntermediate: Int, hidden: Int,
    actDtype: DType, blockSize: Int = 16, global: Float = 1.0
) -> MoELayer.StackedNvfp4Experts? {
    do {
        let gw = try weights.tensor(named: "\(base).gate_proj.weight")
        let gs = try weights.tensor(named: "\(base).gate_proj.scales")
        let uw = try weights.tensor(named: "\(base).up_proj.weight")
        let us = try weights.tensor(named: "\(base).up_proj.scales")
        let dw = try weights.tensor(named: "\(base).down_proj.weight")
        let ds = try weights.tensor(named: "\(base).down_proj.scales")
        guard gw.dtype == .u32, gs.dtype == .u8 else { return nil }
        // Expected: [E, mid, hidden/8] and [E, mid, hidden/16]
        return MoELayer.StackedNvfp4Experts(
            gateWeight: gw, gateScales: gs,
            upWeight: uw, upScales: us,
            downWeight: dw, downScales: ds,
            numExperts: numExperts, moeIntermediate: moeIntermediate,
            hidden: hidden, blockSize: blockSize, global: global,
            dtype: actDtype)
    } catch {
        return nil
    }
}

/// Slice stacked expert weights `[E, out, in]` (or quantized) into
/// per-expert `AnyLinear`s — same contract as the Qwen3.5 MoE path.
private func lagunaSliceStackedExperts(
    base: String, in weights: SafeTensorsBundle,
    numExperts: Int, outDim: Int, inDim: Int,
    quant: ModelConfig.QuantizationConfig?
) throws -> [AnyLinear] {
    var out: [AnyLinear] = []
    out.reserveCapacity(numExperts)
    if let q = quant, weights.isQuantized(base) {
        let stackedW = try weights.tensor(named: "\(base).weight")
        let stackedS = try weights.tensor(named: "\(base).scales")
        let stackedB = try weights.tensor(named: "\(base).biases")
        let packedCols = stackedW.shape[stackedW.shape.count - 1]
        let groupCols = stackedS.shape[stackedS.shape.count - 1]
        let bits = deriveAffineQuantBits(
            weightPackedCols: packedCols, scaleCols: groupCols,
            groupSize: q.groupSize)
        precondition(
            [3, 4, 5, 6, 8].contains(bits),
            "lagunaSliceStackedExperts: derived \(bits)-bit for \(base) "
                + "— unsupported quantization bit-width")
        for e in 0 ..< numExperts {
            let w = stackedW.slicedRows(start: e, count: 1)
                .reshaped(to: [outDim, packedCols])
            let s = stackedS.slicedRows(start: e, count: 1)
                .reshaped(to: [outDim, groupCols])
            let b = stackedB.slicedRows(start: e, count: 1)
                .reshaped(to: [outDim, groupCols])
            out.append(
                AnyLinear(
                    QuantizedLinear(
                        weight: w, scales: s, biases: b,
                        bits: bits, groupSize: q.groupSize)))
        }
    } else {
        let stacked = try weights.tensor(named: "\(base).weight")
        for e in 0 ..< numExperts {
            out.append(
                AnyLinear(
                    Linear(
                        weight: stacked.slicedRows(start: e, count: 1)
                            .reshaped(to: [outDim, inDim]))))
        }
    }
    return out
}

/// Stacked int4 gather path when shapes pass the bm16 tile contract.
private func lagunaTryLoadStackedInt4(
    base: String, weights: SafeTensorsBundle,
    quant: ModelConfig.QuantizationConfig?,
    numExperts: Int, moeIntermediate: Int, hidden: Int
) -> MoELayer.StackedInt4Experts? {
    guard let q = quant,
        weights.isQuantized("\(base).gate_proj"),
        weights.isQuantized("\(base).up_proj"),
        weights.isQuantized("\(base).down_proj"),
        hidden % 32 == 0,
        moeIntermediate % 32 == 0
    else { return nil }
    do {
        let gw = try weights.tensor(named: "\(base).gate_proj.weight")
        let gs = try weights.tensor(named: "\(base).gate_proj.scales")
        let gb = try weights.tensor(named: "\(base).gate_proj.biases")
        let uw = try weights.tensor(named: "\(base).up_proj.weight")
        let us = try weights.tensor(named: "\(base).up_proj.scales")
        let ub = try weights.tensor(named: "\(base).up_proj.biases")
        let dw = try weights.tensor(named: "\(base).down_proj.weight")
        let ds = try weights.tensor(named: "\(base).down_proj.scales")
        let db = try weights.tensor(named: "\(base).down_proj.biases")
        let gateBits = deriveAffineQuantBits(
            weightPackedCols: gw.shape[gw.shape.count - 1],
            scaleCols: gs.shape[gs.shape.count - 1],
            groupSize: q.groupSize)
        let upBits = deriveAffineQuantBits(
            weightPackedCols: uw.shape[uw.shape.count - 1],
            scaleCols: us.shape[us.shape.count - 1],
            groupSize: q.groupSize)
        let downBits = deriveAffineQuantBits(
            weightPackedCols: dw.shape[dw.shape.count - 1],
            scaleCols: ds.shape[ds.shape.count - 1],
            groupSize: q.groupSize)
        guard gateBits == 4, upBits == 4, downBits == 4 else { return nil }
        let dtype = gs.dtype
        guard dtype == .f16 || dtype == .bf16 || dtype == .f32 else { return nil }
        return MoELayer.StackedInt4Experts(
            gateWeight: gw, gateScales: gs, gateBiases: gb,
            upWeight: uw, upScales: us, upBiases: ub,
            downWeight: dw, downScales: ds, downBiases: db,
            numExperts: numExperts, moeIntermediate: moeIntermediate,
            hidden: hidden, groupSize: q.groupSize, dtype: dtype)
    } catch {
        return nil
    }
}

// ─── Dense MLP ───────────────────────────────────────────────────────

final class LagunaDenseMLP {
    let gate, up, down: AnyLinear
    let hidden: Int

    init(gate: AnyLinear, up: AnyLinear, down: AnyLinear, hidden: Int) {
        self.gate = gate
        self.up = up
        self.down = down
        self.hidden = hidden
    }

    func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in gate.parameters() { out.append(("gate_proj.\(k)", v)) }
        for (k, v) in up.parameters() { out.append(("up_proj.\(k)", v)) }
        for (k, v) in down.parameters() { out.append(("down_proj.\(k)", v)) }
        return out
    }

    func forward(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        let g = gate(x, on: cmd)
        let u = up(x, on: cmd)
        let inner = Ops.swiglu(gate: g, up: u, on: cmd)
        return down(inner, on: cmd)
    }

    func forwardMany(
        _ x: Tensor, t: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let g = gate.callMany(x, t: t, on: cmd, device: device)
        let u = up.callMany(x, t: t, on: cmd, device: device)
        _ = Ops.swiglu(gate: g, up: u, on: cmd, into: u)
        return down.callMany(u, t: t, on: cmd, device: device)
    }
}

enum LagunaFFN {
    case dense(LagunaDenseMLP)
    case moe(MoELayer)
}

// ─── LagunaLayer ─────────────────────────────────────────────────────

final class LagunaLayer {
    let qProj, kProj, vProj, oProj: AnyLinear
    let gProj: AnyLinear?
    /// Stacked dense [Q;K;V] weight for single-GEMV decode (BF16 NVFP4 path).
    let qkvFusedWeight: Tensor?
    let qOutDim: Int
    let kOutDim: Int
    let vOutDim: Int
    let qNorm, kNorm: RMSNorm
    let inputNorm, postAttnNorm: RMSNorm
    let ffn: LagunaFFN
    let hidden, nHeads, nKVHeads, headDim, rotaryDim: Int
    let ropeTheta: Float
    let isSliding: Bool
    /// Full-attention layers use YaRN-scaled partial RoPE.
    let isFullYarn: Bool
    let dtype: DType
    let scale: Float
    /// Gather indices for softplus head-gate expand: each head id repeated
    /// `headDim` times. Built once at first use (decode T=1).
    private var softplusExpandIds: Tensor?
    /// Prefill softplus expand ids for multi-token (cached by T·nHeads·headDim).
    private var softplusExpandIdsMany: Tensor?
    private var softplusExpandIdsManyT: Int = 0
    /// Persistent MoE residual temp (avoids per-token `Tensor.empty` +
    /// breaks Path B `accumulatorScratch` aliasing).
    /// With default residual path we write `h2 + ffnOut` *into* this buffer
    /// (no blit copy). `BUTTER_LAGUNA_FFN_COPY=1` restores the older
    /// copy-then-add path for triage.
    private var ffnOutScratch: Tensor?
    /// Fused QKV int4 decode scratch: `[qDim+kDim+vDim]`, views into regions.
    private var qkvFusedScratch: Tensor?
    private var qOutScratch: Tensor?
    private var kOutScratch: Tensor?
    private var vOutScratch: Tensor?
    /// GPU path may still commit on CPU-router MoE fallback.
    var commitsCommandBuffer: Bool { true }

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        gProj: AnyLinear?, qNorm: RMSNorm, kNorm: RMSNorm,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        ffn: LagunaFFN,
        hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int, rotaryDim: Int,
        ropeTheta: Float, isSliding: Bool, isFullYarn: Bool, dtype: DType,
        device: Device = .shared
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.gProj = gProj
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
        self.ffn = ffn
        self.hidden = hidden
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.rotaryDim = rotaryDim
        self.ropeTheta = ropeTheta
        self.isSliding = isSliding
        self.isFullYarn = isFullYarn
        self.dtype = dtype
        self.scale = 1.0 / Float(Double(headDim).squareRoot())

        // Build stacked dense QKV for single-GEMV decode when all three are
        // plain Linear (NVFP4 Laguna: BF16 attention).
        if let lq = qProj.inner as? Linear, let lk = kProj.inner as? Linear,
            let lv = vProj.inner as? Linear,
            lq.weight.dtype == dtype, lk.weight.dtype == dtype, lv.weight.dtype == dtype,
            lq.weight.shape[1] == hidden, lk.weight.shape[1] == hidden,
            lv.weight.shape[1] == hidden
        {
            let qd = lq.weight.shape[0]
            let kd = lk.weight.shape[0]
            let vd = lv.weight.shape[0]
            self.qOutDim = qd
            self.kOutDim = kd
            self.vOutDim = vd
            let total = qd + kd + vd
            let rowBytes = hidden * dtype.byteSize
            let buf = device.makeBuffer(length: total * rowBytes)
            let qBytes = qd * rowBytes
            let kBytes = kd * rowBytes
            memcpy(buf.contents(), lq.weight.buffer.contents().advanced(by: lq.weight.offset), qBytes)
            memcpy(
                buf.contents().advanced(by: qBytes),
                lk.weight.buffer.contents().advanced(by: lk.weight.offset), kBytes)
            memcpy(
                buf.contents().advanced(by: qBytes + kBytes),
                lv.weight.buffer.contents().advanced(by: lv.weight.offset),
                vd * rowBytes)
            self.qkvFusedWeight = Tensor(
                buffer: buf, offset: 0, shape: [total, hidden], dtype: dtype)
        } else {
            self.qkvFusedWeight = nil
            // Best-effort dims for non-fused path (unused).
            self.qOutDim = (qProj.inner as? Linear)?.weight.shape[0]
                ?? (qProj.inner as? QuantizedLinear)?.weight.shape[0] ?? nHeads * headDim
            self.kOutDim = (kProj.inner as? Linear)?.weight.shape[0]
                ?? (kProj.inner as? QuantizedLinear)?.weight.shape[0] ?? nKVHeads * headDim
            self.vOutDim = (vProj.inner as? Linear)?.weight.shape[0]
                ?? (vProj.inner as? QuantizedLinear)?.weight.shape[0] ?? nKVHeads * headDim
        }
    }

    func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        if let g = gProj {
            for (k, v) in g.parameters() { out.append(("self_attn.g_proj.\(k)", v)) }
        }
        for (k, v) in qNorm.parameters() { out.append(("self_attn.q_norm.\(k)", v)) }
        for (k, v) in kNorm.parameters() { out.append(("self_attn.k_norm.\(k)", v)) }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_attention_layernorm.\(k)", v)) }
        switch ffn {
        case .dense(let mlp):
            for (k, v) in mlp.parameters() { out.append(("mlp.\(k)", v)) }
        case .moe(let moe):
            for (k, v) in moe.parameters() { out.append(("mlp.\(k)", v)) }
        }
        return out
    }

    /// Single-token decode. MoE path and softplus gate may commit `cmd`;
    /// the caller must refresh after `commitsCommandBuffer` layers.
    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
        yarn: Ops.RoPEYaRN,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let work = cmd
        let residual = h
        let xNorm = inputNorm(h, on: work)

        // Fused int4 Q/K/V when all three are QuantizedLinear 4/gs64 and
        // shapes pass the fast-path contract (hidden % 512 == 0, outs % 8).
        let q: Tensor
        let k: Tensor
        let v: Tensor
        if let qq = qProj.inner as? QuantizedLinear, qq.bits == 4, qq.groupSize == 64,
            let qk = kProj.inner as? QuantizedLinear, qk.bits == 4, qk.groupSize == 64,
            let qv = vProj.inner as? QuantizedLinear, qv.bits == 4, qv.groupSize == 64,
            hidden % 512 == 0
        {
            let qDim = qq.weight.shape[0]
            let kDim = qk.weight.shape[0]
            let vDim = qv.weight.shape[0]
            if qDim % 8 == 0 && kDim % 8 == 0 && vDim % 8 == 0 {
                let total = qDim + kDim + vDim
                if qkvFusedScratch == nil
                    || qkvFusedScratch!.elementCount != total
                    || qkvFusedScratch!.dtype != xNorm.dtype
                {
                    qkvFusedScratch = Tensor.empty(
                        shape: [total], dtype: xNorm.dtype, device: device)
                    qOutScratch = Tensor(
                        buffer: qkvFusedScratch!.buffer,
                        offset: qkvFusedScratch!.offset,
                        shape: [qDim], dtype: xNorm.dtype)
                    kOutScratch = Tensor(
                        buffer: qkvFusedScratch!.buffer,
                        offset: qkvFusedScratch!.offset + qDim * xNorm.dtype.byteSize,
                        shape: [kDim], dtype: xNorm.dtype)
                    vOutScratch = Tensor(
                        buffer: qkvFusedScratch!.buffer,
                        offset: qkvFusedScratch!.offset
                            + (qDim + kDim) * xNorm.dtype.byteSize,
                        shape: [vDim], dtype: xNorm.dtype)
                }
                Ops.batchedQkvQgemvInt4Fast(
                    x: xNorm,
                    wQ: qq.weight, scalesQ: qq.scales, biasesQ: qq.biases,
                    wK: qk.weight, scalesK: qk.scales, biasesK: qk.biases,
                    wV: qv.weight, scalesV: qv.scales, biasesV: qv.biases,
                    outQ: qDim, outK: kDim, outV: vDim,
                    on: work, into: qkvFusedScratch!)
                q = qOutScratch!
                k = kOutScratch!
                v = vOutScratch!
            } else {
                q = qProj(xNorm, on: work)
                k = kProj(xNorm, on: work)
                v = vProj(xNorm, on: work)
            }
        } else if let qkvW = qkvFusedWeight,
            qkvW.dtype == xNorm.dtype,
            ProcessInfo.processInfo.environment["BUTTER_LAGUNA_SERIAL_QKV"] == nil
        {
            // Single dense GEMV over stacked [Q;K;V] (NVFP4 board: BF16 attn).
            let qDim = qOutDim
            let kDim = kOutDim
            let total = qDim + kDim + vOutDim
            if qkvFusedScratch == nil
                || qkvFusedScratch!.elementCount != total
                || qkvFusedScratch!.dtype != xNorm.dtype
            {
                qkvFusedScratch = Tensor.empty(
                    shape: [total], dtype: xNorm.dtype, device: device)
                qOutScratch = Tensor(
                    buffer: qkvFusedScratch!.buffer,
                    offset: qkvFusedScratch!.offset,
                    shape: [qDim], dtype: xNorm.dtype)
                kOutScratch = Tensor(
                    buffer: qkvFusedScratch!.buffer,
                    offset: qkvFusedScratch!.offset + qDim * xNorm.dtype.byteSize,
                    shape: [kDim], dtype: xNorm.dtype)
                vOutScratch = Tensor(
                    buffer: qkvFusedScratch!.buffer,
                    offset: qkvFusedScratch!.offset
                        + (qDim + kDim) * xNorm.dtype.byteSize,
                    shape: [vOutDim], dtype: xNorm.dtype)
            }
            _ = Ops.gemv(
                weight: qkvW, input: xNorm, on: work, into: qkvFusedScratch!)
            q = qOutScratch!
            k = kOutScratch!
            v = vOutScratch!
        } else {
            q = qProj(xNorm, on: work)
            k = kProj(xNorm, on: work)
            v = vProj(xNorm, on: work)
        }

        // QK-norm + RoPE. Sliding decode: fused one-dispatch path
        // (BUTTER_LAGUNA_FUSED_QK_NORM_ROPE, default ON) collapses four
        // stock kernels into one TG-per-head kernel — mlxfast
        // DARKBLOOM_FUSED_SLIDING_QK_NORM_ROPE. Full / partial / YaRN
        // keep the stock chain.
        let qN: Tensor
        let kN: Tensor
        if LagunaFusedQKNormRoPE.canFuse(
            isSliding: isSliding, rotaryDim: rotaryDim,
            headDim: headDim, dtype: q.dtype)
        {
            let fused = LagunaFusedQKNormRoPE.apply(
                rawQ: q, rawK: k,
                qWeight: qNorm.weight, kWeight: kNorm.weight,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                position: position, thetaBase: ropeTheta, eps: qNorm.eps,
                on: work, device: device)
            qN = fused.q
            kN = fused.k
        } else {
            var qTmp = Ops.rmsNormRows(
                q, weight: qNorm.weight, eps: qNorm.eps,
                nRows: nHeads, rowSize: headDim, on: work)
            var kTmp = Ops.rmsNormRows(
                k, weight: kNorm.weight, eps: kNorm.eps,
                nRows: nKVHeads, rowSize: headDim, on: work)
            applyRopeInPlace(
                q: &qTmp, k: &kTmp, position: position, yarn: yarn, on: work)
            qN = qTmp
            kN = kTmp
        }

        let q2d = qN.reshaped(to: [nHeads, headDim])
        let k2d = kN.reshaped(to: [nKVHeads, headDim])
        let v2d = v.reshaped(to: [nKVHeads, headDim])
        cache.appendOnGPU(kFlat: k2d, vFlat: v2d, on: work)

        let (cacheK, cacheV) = cache.prepareForAttention(on: work)
        let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: cache.length)
        var attnOut = Ops.sdpaDecode(
            q: q2d, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: work,
            sinkEnd: sinkEnd, windowStart: windowStart
        ).reshaped(to: [nHeads * headDim])

        // Softplus per-head gate before o_proj (GPU expand — no host sync).
        var workBuf = work
        if let gProj {
            attnOut = applySoftplusHeadGate(
                attnOut, xNorm: xNorm, gProj: gProj,
                cmd: &workBuf, device: device)
        }

        let oOut = oProj(attnOut, on: workBuf)
        // Fused residual + post-attn RMSNorm when hidden ≤ 4096 (Laguna = 2048).
        let h2: Tensor
        let ffnIn: Tensor
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                residual, oOut, weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: 1, rowSize: hidden, on: workBuf)
            h2 = fused.residual
            ffnIn = fused.normed
        } else {
            h2 = Ops.add(residual, oOut, on: workBuf)
            ffnIn = postAttnNorm(h2, on: workBuf)
        }

        switch ffn {
        case .dense(let mlp):
            let ffnOut = mlp.forward(ffnIn, on: workBuf)
            return Ops.add(h2, ffnOut, on: workBuf)
        case .moe(let moe):
            // Full Path B: residual on same CB, one commit+wait at layer end.
            // Host/hybrid commits mid-decode — residual on fresh CB + wait.
            let ffnOut = moe.decode(
                ffnIn, position: position,
                cache: StatelessLayerCache(),
                cmd: workBuf, device: device)
            if workBuf.status == .committed || workBuf.status == .completed
                || workBuf.status == .error
            {
                let addCmd = device.makeCommandBuffer()
                let out = Ops.add(h2, ffnOut, on: addCmd)
                addCmd.commit()
                addCmd.waitUntilCompleted()
                return out
            }
            // Residual dest is a dedicated persistent buffer so we never need
            // to blit-copy ffnOut away from accumulatorScratch. ffnOut may
            // alias acc; writing residual into ffnOutScratch is race-free
            // under per-layer (or packed) CB commit order.
            if ffnOutScratch == nil
                || ffnOutScratch!.dtype != ffnOut.dtype
                || ffnOutScratch!.elementCount != hidden
            {
                ffnOutScratch = Tensor.empty(
                    shape: [hidden], dtype: ffnOut.dtype, device: device)
            }
            // Escape: old blit materialize (debug / if residual quality drifts).
            if ProcessInfo.processInfo.environment["BUTTER_LAGUNA_FFN_COPY"] == "1" {
                Ops.copy(ffnOut, into: ffnOutScratch!, on: workBuf)
                return Ops.add(h2, ffnOutScratch!, on: workBuf)
            }
            return Ops.add(h2, ffnOut, on: workBuf, into: ffnOutScratch!)
        }
    }

    /// Multi-token layer forward for chunked prefill. `h` is `[nRows, hidden]`.
    /// Batches QKV / SDPA / MoE gather instead of the per-token `forward` loop.
    func forwardMulti(
        _ h: Tensor, startingAt position: Int, cache: any KVCacheProtocol,
        yarn: Ops.RoPEYaRN,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let nRows = h.shape[0]
        precondition(h.shape == [nRows, hidden], "LagunaLayer.forwardMulti: shape \(h.shape)")
        precondition(nRows > 0)

        let work = cmd
        let residual = h
        let xNormFlat = Ops.rmsNormRows(
            h.reshaped(to: [nRows * hidden]),
            weight: inputNorm.weight, eps: inputNorm.eps,
            nRows: nRows, rowSize: hidden, on: work)
        let xNorm = xNormFlat.reshaped(to: [nRows, hidden])

        // QKV projections (batched affine/dequant GEMM).
        var q = qProj.callMany(xNorm, t: nRows, on: work, device: device)
        var k = kProj.callMany(xNorm, t: nRows, on: work, device: device)
        let v = vProj.callMany(xNorm, t: nRows, on: work, device: device)

        // QK-norm over T·heads rows.
        q = Ops.rmsNormRows(
            q.reshaped(to: [nRows * nHeads * headDim]),
            weight: qNorm.weight, eps: qNorm.eps,
            nRows: nRows * nHeads, rowSize: headDim, on: work
        ).reshaped(to: [nRows, nHeads * headDim])
        k = Ops.rmsNormRows(
            k.reshaped(to: [nRows * nKVHeads * headDim]),
            weight: kNorm.weight, eps: kNorm.eps,
            nRows: nRows * nKVHeads, rowSize: headDim, on: work
        ).reshaped(to: [nRows, nKVHeads * headDim])

        // RoPE + KV append for the whole chunk.
        lagunaBatchedRopeAndKV(
            q: &q, k: &k, v: v,
            nRows: nRows, position: position,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            rotaryDim: rotaryDim, ropeTheta: ropeTheta,
            isFullYarn: isFullYarn, yarn: yarn,
            cache: cache, cmd: work, device: device)

        // Causal multi-query SDPA over the growing cache.
        let qForSdpa = q.reshaped(to: [nRows, nHeads, headDim])
        let (cacheK, cacheV) = cache.prepareForAttention(on: work)
        let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: cache.length)
        // Sliding-window layers: sdpaMulti is causal-only; when window
        // applies, fall back to per-row decode SDPA (still better than a
        // full per-token layer graph for the FFN). Full layers use sdpaMulti.
        let attnOut: Tensor
        if isSliding, windowStart > 0 || sinkEnd > 0 {
            // Concurrent per-query windowed SDPA (one encoder, all rows).
            // Sequential sdpaDecode was 30 SWA layers × T serial kernels.
            let flat = Tensor.empty(
                shape: [nRows, nHeads * headDim], dtype: q.dtype, device: device)
            let rowBytes = nHeads * headDim * q.dtype.byteSize
            let psoName: String?
            switch (headDim, q.dtype) {
            case (128, .bf16): psoName = "iron_sdpa_decode_bf16"
            case (128, .f16): psoName = "iron_sdpa_decode_f16"
            case (128, .f32): psoName = "iron_sdpa_decode_f32"
            default: psoName = nil
            }
            if let psoName,
                let enc = work.makeComputeCommandEncoder(dispatchType: .concurrent)
            {
                let pso = PSOCache.shared.pipelineState(for: psoName)
                let threadsPerGroup = 1024
                let headsPerGroup = nHeads / nKVHeads
                enc.setComputePipelineState(pso)
                enc.setBuffer(cacheK.buffer, offset: cacheK.offset, index: 1)
                enc.setBuffer(cacheV.buffer, offset: cacheV.offset, index: 2)
                var headDimV = UInt32(headDim)
                var kvStrideV = UInt32(cache.maxSeq)
                var hpgV = UInt32(headsPerGroup)
                var hasSink: UInt32 = 0
                var sinkLogit: Float = 0
                var scaleV = scale
                enc.setBytes(&headDimV, length: 4, index: 4)
                enc.setBytes(&kvStrideV, length: 4, index: 6)
                enc.setBytes(&hpgV, length: 4, index: 7)
                enc.setBytes(&hasSink, length: 4, index: 10)
                enc.setBytes(&sinkLogit, length: 4, index: 11)
                enc.setBytes(&scaleV, length: 4, index: 12)
                let grid = MTLSize(
                    width: nHeads * threadsPerGroup, height: 1, depth: 1)
                let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
                for i in 0 ..< nRows {
                    let nKV = min(cache.length, position + i + 1)
                    let (sEnd, wStart) = cache.sdpaSinkWindow(nKV: nKV)
                    enc.setBuffer(q.buffer, offset: q.offset + i * rowBytes, index: 0)
                    enc.setBuffer(
                        flat.buffer, offset: flat.offset + i * rowBytes, index: 3)
                    var nKVV = UInt32(nKV)
                    var sEndV = UInt32(sEnd)
                    var wStartV = UInt32(wStart)
                    enc.setBytes(&nKVV, length: 4, index: 5)
                    enc.setBytes(&sEndV, length: 4, index: 8)
                    enc.setBytes(&wStartV, length: 4, index: 9)
                    enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
                }
                enc.endEncoding()
            } else {
                // Uncommon head dim: sequential Ops.sdpaDecode.
                for i in 0 ..< nRows {
                    let qi = q.slicedRows(start: i, count: 1)
                        .reshaped(to: [nHeads, headDim])
                    let nKV = min(cache.length, position + i + 1)
                    let (sEnd, wStart) = cache.sdpaSinkWindow(nKV: nKV)
                    let row = Ops.sdpaDecode(
                        q: qi, k: cacheK, v: cacheV,
                        nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                        nKV: nKV, kvStride: cache.maxSeq,
                        scale: scale, on: work,
                        sinkEnd: sEnd, windowStart: wStart)
                    let blit = work.makeBlitCommandEncoder()!
                    blit.copy(
                        from: row.buffer, sourceOffset: row.offset,
                        to: flat.buffer,
                        destinationOffset: flat.offset + i * rowBytes,
                        size: rowBytes)
                    blit.endEncoding()
                }
            }
            attnOut = flat
        } else if let mmaOut = LagunaSdpa.prefillMmaBf16(
            qTHD: qForSdpa, cacheK: cacheK, cacheV: cacheV,
            nRows: nRows, position: position,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            maxSeq: cache.maxSeq, scale: scale,
            on: work, device: device)
        {
            // sdpaPrefillMma (f16 MMA via bf16 cast) — big prefill lever
            // when T%32==0. Escape: BUTTER_LAGUNA_SDPA_PREFILL_MMA=0.
            attnOut = mmaOut
        } else {
            attnOut = Ops.sdpaMulti(
                q: qForSdpa, k: cacheK, v: cacheV,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                baseKV: position, nQuery: nRows, kvStride: cache.maxSeq,
                causal: true, scale: scale, on: work
            ).reshaped(to: [nRows, nHeads * headDim])
        }

        // Softplus head gate for the chunk — host expand (same math as decode).
        var gated = attnOut
        var workBuf = work
        if let gProj {
            gated = applySoftplusHeadGateManyHost(
                attnOut, xNorm: xNorm, gProj: gProj, t: nRows,
                cmd: &workBuf, device: device)
        }
        let work2 = workBuf

        let oOut = oProj.callMany(gated, t: nRows, on: work2, device: device)
        let residualFlat = residual.reshaped(to: [nRows * hidden])
        let oFlat = oOut.reshaped(to: [nRows * hidden])
        let h2: Tensor
        let ffnIn: Tensor
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                residualFlat, oFlat, weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: nRows, rowSize: hidden, on: work2)
            h2 = fused.residual.reshaped(to: [nRows, hidden])
            ffnIn = fused.normed.reshaped(to: [nRows, hidden])
        } else {
            h2 = Ops.add(residualFlat, oFlat, on: work2).reshaped(to: [nRows, hidden])
            ffnIn = Ops.rmsNormRows(
                h2.reshaped(to: [nRows * hidden]),
                weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: nRows, rowSize: hidden, on: work2
            ).reshaped(to: [nRows, hidden])
        }

        switch ffn {
        case .dense(let mlp):
            let ffnOut = mlp.forwardMany(ffnIn, t: nRows, on: work2, device: device)
            let out = Ops.add(
                h2.reshaped(to: [nRows * hidden]),
                ffnOut.reshaped(to: [nRows * hidden]),
                on: work2
            ).reshaped(to: [nRows, hidden])
            if work2.status != .committed && work2.status != .completed
                && work2.status != .error
            {
                work2.commit()
            }
            work2.waitUntilCompleted()
            return out
        case .moe(let moe):
            // Batched gather MoE (decodeMany). Default ON — big prefill win.
            // MPP gather is quality-broken for Laguna affine4 on M5; force
            // simdgroup path. Escape: BUTTER_LAGUNA_MOE_MANY=0 → serial.
            // MPP re-enable: BUTTER_MOE_BGEMM_MPP=1 (and unset NO_MPP).
            if ProcessInfo.processInfo.environment["BUTTER_MOE_BGEMM_NO_MPP"] == nil
                && ProcessInfo.processInfo.environment["BUTTER_MOE_BGEMM_MPP"] == nil
            {
                setenv("BUTTER_MOE_BGEMM_NO_MPP", "1", 0)
            }
            if ProcessInfo.processInfo.environment["BUTTER_LAGUNA_MOE_MANY"] == "0" {
                let dtBytes = ffnIn.dtype.byteSize
                let outFlat = Tensor.empty(
                    shape: [nRows * hidden], dtype: ffnIn.dtype, device: device)
                var moeCmd = work2
                for r in 0 ..< nRows {
                    let hRow = Tensor(
                        buffer: ffnIn.buffer,
                        offset: ffnIn.offset + r * hidden * dtBytes,
                        shape: [hidden], dtype: ffnIn.dtype)
                    if moeCmd.status == .committed || moeCmd.status == .completed
                        || moeCmd.status == .error
                    {
                        moeCmd = device.makeCommandBuffer()
                    }
                    let rowOut = moe.decode(
                        hRow, position: position + r,
                        cache: StatelessLayerCache(),
                        cmd: moeCmd, device: device)
                    let dest = Tensor(
                        buffer: outFlat.buffer,
                        offset: outFlat.offset + r * hidden * dtBytes,
                        shape: [hidden], dtype: ffnIn.dtype)
                    let copyCmd = device.makeCommandBuffer()
                    Ops.copy(rowOut, into: dest, on: copyCmd)
                    copyCmd.commit()
                    copyCmd.waitUntilCompleted()
                    moeCmd = device.makeCommandBuffer()
                }
                let addCmd = device.makeCommandBuffer()
                let out = Ops.add(
                    h2.reshaped(to: [nRows * hidden]),
                    outFlat,
                    on: addCmd
                ).reshaped(to: [nRows, hidden])
                addCmd.commit()
                addCmd.waitUntilCompleted()
                return out
            }
            // Stay on work2 when device plan keeps cmd open; residual on
            // same CB then commit without wait (model prefill pipe fences).
            let ffnOutFlat = moe.decodeMany(
                ffnIn.reshaped(to: [nRows * hidden]),
                t: nRows, cmd: work2, device: device, commitAtEnd: false)
            let out: Tensor
            if work2.status == .committed || work2.status == .completed
                || work2.status == .error
            {
                // MoE committed the layer CB (e.g. isolated MPP path). Residual
                // must land on a fresh CB and be waited (or at least committed
                // into the prefill pipe) so the next layer does not race.
                let addCmd = device.makeCommandBuffer()
                out = Ops.add(
                    h2.reshaped(to: [nRows * hidden]),
                    ffnOutFlat,
                    on: addCmd
                ).reshaped(to: [nRows, hidden])
                addCmd.commit()
                // Wait here: prefill pipe only tracks work2, not this addCmd.
                addCmd.waitUntilCompleted()
                return out
            }
            out = Ops.add(
                h2.reshaped(to: [nRows * hidden]),
                ffnOutFlat,
                on: work2
            ).reshaped(to: [nRows, hidden])
            if work2.status != .committed && work2.status != .completed
                && work2.status != .error
            {
                work2.commit()
            }
            return out
        }
    }

    /// Softplus headwise gate over T tokens. GPU expand via gather
    /// (table `[T·nHeads, 1]`, ids head-within-table repeated headDim).
    private func applySoftplusHeadGateManyHost(
        _ attnOut: Tensor, xNorm: Tensor, gProj: AnyLinear, t: Int,
        cmd: inout MTLCommandBuffer, device: Device
    ) -> Tensor {
        let gRaw = gProj.callMany(xNorm, t: t, on: cmd, device: device)
        let gSoft = Ops.softplus(gRaw, on: cmd)
        let nGates = t * nHeads
        let table = gSoft.reshaped(to: [nGates, 1])
        // Expand ids: for each gate g, headDim repeats of g. Cache by T.
        let expandCount = nGates * headDim
        if softplusExpandIdsMany == nil || softplusExpandIdsManyT != t
            || softplusExpandIdsMany!.elementCount != expandCount
        {
            var ids = [UInt32](repeating: 0, count: expandCount)
            for g in 0 ..< nGates {
                let base = g * headDim
                for d in 0 ..< headDim { ids[base + d] = UInt32(g) }
            }
            let idTensor = Tensor.empty(
                shape: [expandCount], dtype: .u32, device: device)
            idTensor.copyIn(from: ids)
            softplusExpandIdsMany = idTensor
            softplusExpandIdsManyT = t
        }
        let gExp2d = Ops.gather(
            table: table, tokenIds: softplusExpandIdsMany!, on: cmd)
        let gExp = gExp2d.reshaped(to: [t, nHeads * headDim])
        let attn = attnOut.reshaped(to: [t, nHeads * headDim])
        return Ops.mul(attn, gExp, on: cmd)
    }

    private func applyRopeInPlace(
        q: inout Tensor, k: inout Tensor,
        position: Int, yarn: Ops.RoPEYaRN,
        on cmd: MTLCommandBuffer
    ) {
        // Full layers: true YaRN on the partial-rotary subset
        // (half_dim = rotaryDim/2, head stride = headDim). Cos/sin already
        // include config attention_factor via yarn.attnFactor.
        if isFullYarn {
            lagunaRopeYaRNPartial(
                q, position: position, nHeads: nHeads,
                headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: ropeTheta, yarn: yarn, on: cmd)
            lagunaRopeYaRNPartial(
                k, position: position, nHeads: nKVHeads,
                headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: ropeTheta, yarn: yarn, on: cmd)
            return
        }
        // Sliding layers: plain RoPE.
        if rotaryDim == headDim {
            _ = Ops.rope(
                q.reshaped(to: [nHeads, headDim]),
                position: position, headDim: headDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd, into: q)
            _ = Ops.rope(
                k.reshaped(to: [nKVHeads, headDim]),
                position: position, headDim: headDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd, into: k)
        } else {
            Ops.ropePartial(
                q, position: position,
                headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd)
            Ops.ropePartial(
                k, position: position,
                headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd)
        }
    }

    /// Softplus headwise gate: expand [nHeads] → [nHeads·headDim] via GPU
    /// gather (table = softplus gates as `[nHeads, 1]`, ids = head id
    /// repeated `headDim` times). No host sync — stays on `cmd`.
    private func applySoftplusHeadGate(
        _ attnOut: Tensor, xNorm: Tensor, gProj: AnyLinear,
        cmd: inout MTLCommandBuffer, device: Device
    ) -> Tensor {
        let gRaw = gProj(xNorm, on: cmd)
        let gSoft = Ops.softplus(gRaw, on: cmd)
        // Table [nHeads, 1] for gather expand.
        let table = gSoft.reshaped(to: [nHeads, 1])
        if softplusExpandIds == nil
            || softplusExpandIds!.elementCount != nHeads * headDim
        {
            var ids = [UInt32](repeating: 0, count: nHeads * headDim)
            for h in 0 ..< nHeads {
                let base = h * headDim
                for d in 0 ..< headDim { ids[base + d] = UInt32(h) }
            }
            let t = Tensor.empty(shape: [nHeads * headDim], dtype: .u32, device: device)
            t.copyIn(from: ids)
            softplusExpandIds = t
        }
        let gExp2d = Ops.gather(table: table, tokenIds: softplusExpandIds!, on: cmd)
        let gExp = gExp2d.reshaped(to: [nHeads * headDim])
        return Ops.mul(attnOut, gExp, on: cmd)
    }
}

/// Batched RoPE + KV append for a prefill chunk.
/// Sliding layers: one `ropePartialManyTwo` + one `appendRangeOnGPUMany`.
/// Full YaRN layers: per-row YaRN (kernel is single-position) then same
/// batched KV write.
private func lagunaBatchedRopeAndKV(
    q: inout Tensor, k: inout Tensor, v: Tensor,
    nRows: Int, position: Int,
    nHeads: Int, nKVHeads: Int, headDim: Int,
    rotaryDim: Int, ropeTheta: Float,
    isFullYarn: Bool, yarn: Ops.RoPEYaRN,
    cache: any KVCacheProtocol,
    cmd: MTLCommandBuffer, device: Device
) {
    let qDim = nHeads * headDim
    let kvDim = nKVHeads * headDim
    let positionsT = Tensor.empty(shape: [nRows], dtype: .u32, device: device)
    let positionsPtr = positionsT.buffer.contents()
        .advanced(by: positionsT.offset)
        .bindMemory(to: UInt32.self, capacity: nRows)
    for r in 0 ..< nRows { positionsPtr[r] = UInt32(position + r) }

    if isFullYarn {
        // Single-position YaRN kernel — apply per row (still far fewer
        // dispatches than the old full per-token layer graph).
        let qRowBytes = qDim * q.dtype.byteSize
        let kRowBytes = kvDim * k.dtype.byteSize
        for r in 0 ..< nRows {
            let qRow = Tensor(
                buffer: q.buffer, offset: q.offset + r * qRowBytes,
                shape: [qDim], dtype: q.dtype)
            let kRow = Tensor(
                buffer: k.buffer, offset: k.offset + r * kRowBytes,
                shape: [kvDim], dtype: k.dtype)
            lagunaRopeYaRNPartial(
                qRow, position: position + r, nHeads: nHeads,
                headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: ropeTheta, yarn: yarn, on: cmd)
            lagunaRopeYaRNPartial(
                kRow, position: position + r, nHeads: nKVHeads,
                headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: ropeTheta, yarn: yarn, on: cmd)
        }
    } else {
        Ops.ropePartialManyTwo(
            q: q, qNHeads: nHeads, qRowStride: qDim,
            k: k, kNHeads: nKVHeads, kRowStride: kvDim,
            positions: positionsT, t: nRows,
            headDim: headDim, rotaryDim: rotaryDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)
    }

    guard let kv = cache as? KVCache else {
        for i in 0 ..< nRows {
            let kRow = k.slicedRows(start: i, count: 1)
                .reshaped(to: [nKVHeads, headDim])
            let vRow = v.slicedRows(start: i, count: 1)
                .reshaped(to: [nKVHeads, headDim])
            cache.appendOnGPU(kFlat: kRow, vFlat: vRow, on: cmd)
        }
        return
    }
    let appendPositions = Tensor.empty(shape: [nRows], dtype: .u32, device: device)
    kv.reserveSlotsManyOnHost(t: nRows, into: appendPositions)
    // appendRangeOnGPUMany expects k/v as [T, nKVHeads, headDim] flat
    // with row stride nKVHeads*headDim — our k/v are [T, nKV*d].
    kv.appendRangeOnGPUMany(
        kFlat: k, vFlat: v, t: nRows,
        positions: appendPositions, on: cmd)
}

/// YaRN on a partial-rotary head: rotate pairs `(i, i + rotaryDim/2)` for
/// `i ∈ [0, rotaryDim/2)` inside each `headDim`-strided head. Matches HF
/// partial-rotary + YaRN (`cos/sin * attention_factor` inside the kernel).
private func lagunaRopeYaRNPartial(
    _ qk: Tensor, position: Int, nHeads: Int,
    headDim: Int, rotaryDim: Int,
    thetaBase: Float, yarn: Ops.RoPEYaRN,
    on cmd: MTLCommandBuffer
) {
    precondition(
        qk.elementCount == nHeads * headDim,
        "lagunaRopeYaRNPartial: qk size \(qk.elementCount) ≠ nHeads·headDim")
    precondition(
        rotaryDim > 0 && rotaryDim <= headDim && rotaryDim % 2 == 0,
        "lagunaRopeYaRNPartial: rotaryDim \(rotaryDim) invalid for headDim \(headDim)")
    let halfRotary = rotaryDim / 2
    let grid = MTLSize(width: nHeads, height: halfRotary, depth: 1)
    let tg = MTLSize(width: 1, height: 1, depth: 1)
    switch qk.dtype {
    case .f32:
        IronKernels.iron_rope_yarn_f32(
            qk: qk.buffer, qkOffset: qk.offset,
            out: qk.buffer, outOffset: qk.offset,
            head_dim: UInt32(headDim), half_dim: UInt32(halfRotary),
            position: UInt32(position), theta_base: thetaBase,
            factor: yarn.factor, low: yarn.low, high: yarn.high,
            attn_factor: yarn.attnFactor,
            gridSize: grid, threadgroupSize: tg, on: cmd)
    case .f16:
        IronKernels.iron_rope_yarn_f16(
            qk: qk.buffer, qkOffset: qk.offset,
            out: qk.buffer, outOffset: qk.offset,
            head_dim: UInt32(headDim), half_dim: UInt32(halfRotary),
            position: UInt32(position), theta_base: thetaBase,
            factor: yarn.factor, low: yarn.low, high: yarn.high,
            attn_factor: yarn.attnFactor,
            gridSize: grid, threadgroupSize: tg, on: cmd)
    case .bf16:
        IronKernels.iron_rope_yarn_bf16(
            qk: qk.buffer, qkOffset: qk.offset,
            out: qk.buffer, outOffset: qk.offset,
            head_dim: UInt32(headDim), half_dim: UInt32(halfRotary),
            position: UInt32(position), theta_base: thetaBase,
            factor: yarn.factor, low: yarn.low, high: yarn.high,
            attn_factor: yarn.attnFactor,
            gridSize: grid, threadgroupSize: tg, on: cmd)
    default:
        fatalError("lagunaRopeYaRNPartial: unsupported dtype \(qk.dtype)")
    }
}

// ─── LagunaModel ─────────────────────────────────────────────────────

public final class LagunaModel: LanguageModel {
    public let embedTokens: AnyEmbedding
    let layers: [LagunaLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear
    /// Optional MXFP8 coarse copy for decode prune.
    let lmHeadMxfp8: Mxfp8Linear?
    let textConfig: LagunaTextConfig
    let yarn: Ops.RoPEYaRN

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let dtype: DType
    public let slidingWindow: Int

    public var defaultPrefillStepSize: Int { 4096 }
    public var requiresLeadingBOS: Bool { false }

    init(
        embedTokens: AnyEmbedding,
        layers: [LagunaLayer],
        finalNorm: RMSNorm,
        lmHead: AnyLinear,
        lmHeadMxfp8: Mxfp8Linear? = nil,
        textConfig: LagunaTextConfig,
        yarn: Ops.RoPEYaRN,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int,
        headDim: Int, vocab: Int, maxSeq: Int, dtype: DType,
        slidingWindow: Int
    ) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.lmHeadMxfp8 = lmHeadMxfp8
        self.textConfig = textConfig
        self.yarn = yarn
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.vocab = vocab
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.slidingWindow = slidingWindow
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("model.embed_tokens.\(k)", v))
        }
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("model.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("model.norm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        return (0 ..< nLayers).map { i in
            let eviction: KVEviction =
                layers[i].isSliding
                ? .window(maxSize: min(slidingWindow, cap), keep: 0)
                : .unbounded
            return KVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                dtype: dtype, eviction: eviction, device: device)
        }
    }

    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        // Multi-CB layer pipeline (default). Residual dest buffer replaces
        // the old ffnCopy blit. Single-CB all layers: LAYER_PIPE=0.
        //
        // Per-token host/Metal overhead (40 layers, 39 MoE, pipe depth 4):
        //   ~41 main CBs + ~39 dual-queue B CBs + ~78 events + 39 blits
        // Packing layers/CB cuts main CB count; dual-queue opt-out cuts B/events;
        // residual-into-scratch skips the blit.
        let pipe =
            ProcessInfo.processInfo.environment["BUTTER_LAGUNA_LAYER_PIPE"] != "0"
        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])

        if pipe {
            // Depth: how many CBs in flight (default 4).
            let depth = max(
                1, Int(ProcessInfo.processInfo.environment["BUTTER_LAGUNA_PIPE_DEPTH"] ?? "") ?? 4)
            // Layers packed into one CB before commit (default 2 → ~half
            // the main-queue commits vs 1-layer/CB). Escape 1 for triage.
            let layersPerCB = max(
                1,
                Int(ProcessInfo.processInfo.environment["BUTTER_LAGUNA_LAYERS_PER_CB"] ?? "")
                    ?? 2)
            var inflight: [MTLCommandBuffer] = []
            if workCmd.status != .committed && workCmd.status != .completed
                && workCmd.status != .error
            {
                workCmd.commit()
            }
            inflight.append(workCmd)
            var i = 0
            while i < layers.count {
                workCmd = device.makeCommandBuffer()
                let end = min(i + layersPerCB, layers.count)
                while i < end {
                    h = layers[i].forward(
                        h, position: position,
                        cache: caches[i] as! any KVCacheProtocol,
                        yarn: yarn,
                        cmd: workCmd, device: device)
                    // MoE host/hybrid may have committed mid-layer; refresh.
                    if workCmd.status == .committed || workCmd.status == .completed
                        || workCmd.status == .error
                    {
                        workCmd = device.makeCommandBuffer()
                    }
                    i += 1
                }
                if workCmd.status != .committed && workCmd.status != .completed
                    && workCmd.status != .error
                {
                    workCmd.commit()
                }
                inflight.append(workCmd)
                while inflight.count > depth {
                    inflight.removeFirst().waitUntilCompleted()
                }
            }
            for cb in inflight { cb.waitUntilCompleted() }
        } else {
            for (i, layer) in layers.enumerated() {
                if workCmd.status == .committed || workCmd.status == .completed
                    || workCmd.status == .error
                {
                    workCmd = device.makeCommandBuffer()
                }
                h = layer.forward(
                    h, position: position,
                    cache: caches[i] as! any KVCacheProtocol,
                    yarn: yarn,
                    cmd: workCmd, device: device)
                _ = i
            }
            if workCmd.status != .committed && workCmd.status != .completed
                && workCmd.status != .error
            {
                workCmd.commit()
            }
            workCmd.waitUntilCompleted()
        }

        return lagunaFinalNormLmHead(
            h: h, finalNorm: finalNorm, lmHead: lmHead,
            lmHeadMxfp8: lmHeadMxfp8, on: cmd, device: device)
    }

    /// Chunked batched prefill: embed once, then per-layer `forwardMulti`
    /// (QKV/SDPA/MoE gather). Falls back to per-token only when T == 1.
    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(!tokenIds.isEmpty, "LagunaModel.forwardMulti: tokenIds must be non-empty")
        let n = tokenIds.count
        if n == 1 {
            return forward(
                tokenId: tokenIds[0], position: position,
                caches: caches, on: cmd, device: device)
        }
        // Full serial escape hatch (debug only).
        if ProcessInfo.processInfo.environment["BUTTER_LAGUNA_SERIAL_PREFILL"] == "1" {
            var logits: Tensor!
            for (i, tok) in tokenIds.enumerated() {
                logits = forward(
                    tokenId: tok, position: position + i,
                    caches: caches, on: cmd, device: device)
            }
            return logits
        }

        // Embed all tokens in one gather.
        let tokBuf = device.makeBuffer(length: n * 4)
        let tids = tokenIds.map { UInt32($0) }
        tids.withUnsafeBytes {
            _ = memcpy(tokBuf.contents(), $0.baseAddress!, n * 4)
        }
        let tokenTensor = Tensor(buffer: tokBuf, offset: 0, shape: [n], dtype: .u32)
        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd)
            .reshaped(to: [n, hidden])
        if workCmd.status != .committed && workCmd.status != .completed
            && workCmd.status != .error
        {
            workCmd.commit()
        }
        workCmd.waitUntilCompleted()

        // Depth-2 pipe for prefill layers (same as decode) when MoE
        // decodeMany commits without mid-layer host waits.
        let layerScan =
            ProcessInfo.processInfo.environment["BUTTER_LAGUNA_LAYER_SCAN"] == "1"
        let layerTiming =
            ProcessInfo.processInfo.environment["BUTTER_LAGUNA_TIMING"] == "1"
        // When scanning/timing, force pipe depth 1 so each layer is fully
        // complete before the host reads h (or times the CB).
        let prefDepth = (layerScan || layerTiming)
            ? 1
            : max(
                1,
                Int(ProcessInfo.processInfo.environment["BUTTER_LAGUNA_PREFILL_PIPE"] ?? "")
                    ?? 4)
        var inflight: [MTLCommandBuffer] = []
        var timingMs: [(Int, String, Double)] = []
        for (i, layer) in layers.enumerated() {
            workCmd = device.makeCommandBuffer()
            let t0 = layerTiming ? CFAbsoluteTimeGetCurrent() : 0
            h = layer.forwardMulti(
                h, startingAt: position,
                cache: caches[i] as! any KVCacheProtocol,
                yarn: yarn,
                cmd: workCmd, device: device)
            if workCmd.status != .committed && workCmd.status != .completed
                && workCmd.status != .error
            {
                workCmd.commit()
            }
            inflight.append(workCmd)
            while inflight.count > prefDepth {
                inflight.removeFirst().waitUntilCompleted()
            }
            if layerTiming {
                for cb in inflight { cb.waitUntilCompleted() }
                inflight.removeAll()
                let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                let kind: String
                switch layer.ffn {
                case .dense: kind = "dense"
                case .moe: kind = "moe"
                }
                let tag = "\(kind)\(layer.isSliding ? "+swa" : "+full")"
                timingMs.append((i, tag, ms))
            }
            if layerScan {
                // Ensure this layer's CB finished (depth may already have waited).
                for cb in inflight { cb.waitUntilCompleted() }
                inflight.removeAll()
                let vals = h.toFloatArray()
                let nan = vals.filter { $0.isNaN }.count
                let inf = vals.filter { $0.isInfinite }.count
                let mx = vals.filter { $0.isFinite }.map(abs).max() ?? -1
                let ffnKind: String
                switch layer.ffn {
                case .dense: ffnKind = "dense"
                case .moe: ffnKind = "moe"
                }
                fputs(
                    "[layer-scan] L\(i) nan=\(nan)/\(vals.count) inf=\(inf) "
                        + "maxAbs=\(mx) ffn=\(ffnKind) swa=\(layer.isSliding)\n",
                    stderr)
                if nan > 0 {
                    fputs("[layer-scan] FIRST_NAN at layer \(i)\n", stderr)
                    break
                }
            }
            _ = i
        }
        for cb in inflight { cb.waitUntilCompleted() }
        if layerTiming, !timingMs.isEmpty {
            let total = timingMs.reduce(0.0) { $0 + $1.2 }
            var moe = 0.0, dense = 0.0, swa = 0.0, full = 0.0
            for (_, tag, ms) in timingMs {
                if tag.contains("moe") { moe += ms } else { dense += ms }
                if tag.contains("swa") { swa += ms } else { full += ms }
            }
            fputs(
                "[laguna-timing] layers=\(String(format: "%.1f", total))ms "
                    + "moe=\(String(format: "%.1f", moe)) dense=\(String(format: "%.1f", dense)) "
                    + "swa=\(String(format: "%.1f", swa)) full=\(String(format: "%.1f", full))\n",
                stderr)
            // Top-5 slowest layers
            let top = timingMs.sorted { $0.2 > $1.2 }.prefix(5)
            for (i, tag, ms) in top {
                fputs(
                    "[laguna-timing]   L\(String(format: "%02d", i)) \(tag) "
                        + "\(String(format: "%.2f", ms))ms\n",
                    stderr)
            }
        }

        // Logits for the last prompt token (generate path contract).
        let last = h.slicedRows(start: n - 1, count: 1).reshaped(to: [hidden])
        return lagunaFinalNormLmHead(
            h: last, finalNorm: finalNorm, lmHead: lmHead,
            lmHeadMxfp8: lmHeadMxfp8, on: cmd, device: device)
    }
}

/// Fused finalNorm + int4 lm_head when shapes match `rmsNormQgemvInt4Fast`
/// (hidden % 512 == 0, vocab % 8 == 0, groupSize 64). Laguna XS hits this.
/// Certified MXFP8 2-pass prune for BF16 lm_head decode (DEFAULT ON when
/// sidecar present; `BUTTER_LAGUNA_LM_HEAD_PRUNE=0` disables).
private func lagunaFinalNormLmHead(
    h: Tensor, finalNorm: RMSNorm, lmHead: AnyLinear,
    lmHeadMxfp8: Mxfp8Linear? = nil,
    on cmd: MTLCommandBuffer,
    device: Device = .shared
) -> Tensor {
    if let qLm = lmHead.inner as? QuantizedLinear, qLm.bits == 4,
        qLm.groupSize == 64,
        h.elementCount % 512 == 0,
        qLm.weight.shape[0] % 8 == 0
    {
        let outDim = qLm.weight.shape[0]
        let logits = Tensor.empty(shape: [outDim], dtype: h.dtype)
        Ops.rmsNormQgemvInt4Fast(
            x: h, normWeight: finalNorm.weight, eps: finalNorm.eps,
            qWeight: qLm.weight, qScales: qLm.scales, qBiases: qLm.biases,
            on: cmd, into: logits)
        return logits
    }
    let normed = finalNorm(h, on: cmd)
    // Decode prune: only when env allows, MXFP8 sidecar loaded, and stock
    // BF16 lm_head shape matches the certified kernels (Laguna XS).
    if LagunaLmHeadPrune.enabled,
        let mx = lmHeadMxfp8,
        let lin = lmHead.inner as? Linear,
        lin.weight.dtype == .bf16,
        lin.weight.shape == [100_352, 2048]
    {
        return LagunaLmHeadPrune.logits(
            normed: normed, mx: mx, bf16Weight: lin.weight,
            on: cmd, device: device)
    }
    return lmHead(normed, on: cmd)
}

/// Load offline MXFP8 lm_head sidecar produced by mlx.quantize mode=mxfp8.
private func lagunaLoadMxfp8LmHead(
    modelDir: URL, hidden: Int, vocab: Int, device: Device
) throws -> Mxfp8Linear {
    let codesURL = modelDir.appendingPathComponent("lm_head_mxfp8_codes.u8")
    let scalesURL = modelDir.appendingPathComponent("lm_head_mxfp8_scales.u8")
    guard FileManager.default.fileExists(atPath: codesURL.path),
        FileManager.default.fileExists(atPath: scalesURL.path)
    else {
        throw LagunaError.missingTensor("lm_head_mxfp8_codes.u8 / scales")
    }
    let expectCodes = vocab * hidden
    let expectScales = vocab * (hidden / 32)
    let codesBuf = device.makeBuffer(length: expectCodes)
    let scalesBuf = device.makeBuffer(length: expectScales)
    // Stream file → buffer (avoid holding a second 200 MB Data copy).
    let cfd = open(codesURL.path, O_RDONLY)
    let sfd = open(scalesURL.path, O_RDONLY)
    guard cfd >= 0, sfd >= 0 else {
        throw LagunaError.missingTensor("open mxfp8 sidecar")
    }
    defer { close(cfd); close(sfd) }
    var got = 0
    while got < expectCodes {
        let n = read(cfd, codesBuf.contents().advanced(by: got), expectCodes - got)
        if n <= 0 { break }
        got += n
    }
    guard got == expectCodes else {
        throw LagunaError.unsupportedConfig("mxfp8 codes short read \(got)/\(expectCodes)")
    }
    got = 0
    while got < expectScales {
        let n = read(sfd, scalesBuf.contents().advanced(by: got), expectScales - got)
        if n <= 0 { break }
        got += n
    }
    guard got == expectScales else {
        throw LagunaError.unsupportedConfig("mxfp8 scales short read \(got)/\(expectScales)")
    }
    let codes = Tensor(
        buffer: codesBuf, offset: 0, shape: [vocab, hidden], dtype: .u8)
    let scales = Tensor(
        buffer: scalesBuf, offset: 0, shape: [vocab, hidden / 32], dtype: .u8)
    return Mxfp8Linear(weight: codes, scales: scales, blockSize: 32)
}
