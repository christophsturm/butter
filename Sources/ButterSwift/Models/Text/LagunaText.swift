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

// ─── LagunaLastRowPrefill ────────────────────────────────────────────

/// Terminal-layer last-row prefill specialization: at prefill only the
/// LAST token's logits matter (`LagunaModel.forwardMulti` slices to the
/// last row before finalNorm+lm_head; see the `let last = ...` call
/// site near the end of that function). Stock `forwardMulti` still runs
/// the FINAL transformer layer's attention output projection + full MoE
/// FFN over every row, then throws away T-1 of them.
///
/// When enabled, `LagunaModel.forwardMulti` calls
/// `LagunaLayer.forwardMultiLastRow` instead of `forwardMulti` for the
/// FINAL layer index only (and only when `nRows > 1`, i.e. a real
/// prefill chunk, T=1 already takes the decode `forward()` path). That
/// variant keeps K/V projection + RoPE + cache append full-width (the
/// cache needs every row for future decode continuation) but computes
/// Q / QK-norm+RoPE / SDPA / o_proj / MoE FFN for the LAST row only,
/// reusing the same T=1 decode kernels `forward()` already uses: a
/// "last row of a prefill chunk" is shape-identical to "one decode step"
/// once the chunk's K/V are resident in the cache.
///
/// Default ON (w2-integration cycle 3 verdict): end-to-end validated;
/// 512-token prompt + 20-token continuation gave byte-identical output
/// with the flag on vs off, and a same-window prefill A/B (3 runs/side,
/// 512-tok `hello world` prompt, 5s cooldowns) measured OFF median
/// 0.97s vs ON median 0.83s, a 14.4% win comfortably clearing the 1.5%
/// bar. Set `BUTTER_LAGUNA_LASTROW_PREFILL=0` to restore the full
/// per-row terminal-layer path.

/// Launch-time flags on the per-token / per-layer Laguna forward paths,
/// hoisted to cached statics. None toggled mid-process by tests
/// (verified against Tests/ setenv sites); live-toggled flags must stay
/// on `ButterEnv[...]`.
private enum LagunaEnvFlags {
    static let serialQkv = ButterEnv["BUTTER_LAGUNA_SERIAL_QKV"] != nil
    static let ffnCopy = ButterEnv["BUTTER_LAGUNA_FFN_COPY"] == "1"
    static let attnPhaseTiming = ButterEnv["BUTTER_LAGUNA_ATTN_PHASE_TIMING"] == "1"
    static let layerPipe = ButterEnv["BUTTER_LAGUNA_LAYER_PIPE"] != "0"
    static let pipeDepth = max(1, Int(ButterEnv["BUTTER_LAGUNA_PIPE_DEPTH"] ?? "") ?? 4)
    static let layersPerCBEnv = Int(ButterEnv["BUTTER_LAGUNA_LAYERS_PER_CB"] ?? "")
    static let serialPrefill = ButterEnv["BUTTER_LAGUNA_SERIAL_PREFILL"] == "1"
    static let layerScan = ButterEnv["BUTTER_LAGUNA_LAYER_SCAN"] == "1"
    static let timing = ButterEnv["BUTTER_LAGUNA_TIMING"] == "1"
    static let prefillPipeEnv = Int(ButterEnv["BUTTER_LAGUNA_PREFILL_PIPE"] ?? "")
    static let moeBgemmNoMpp = ButterEnv["BUTTER_MOE_BGEMM_NO_MPP"] != nil
    static let moeBgemmMpp = ButterEnv["BUTTER_MOE_BGEMM_MPP"] != nil
    static let moeManyOff = ButterEnv["BUTTER_LAGUNA_MOE_MANY"] == "0"
    // W-headgate: fuse the softplus head-gate's `gather(expand_ids) + mul`
    // scaffolding into one `iron_head_gate_mul` broadcast pass (no host
    // expand-id build, no gExp intermediate, one dispatch). Byte-identical
    // to the gather+mul it replaces (bit-exact gate copy + native-T mul),
    // so default ON; escape `BUTTER_LAGUNA_HEAD_GATE_FUSE=0`. The measured
    // `gate` bucket is the largest non-GEMM prefill block (~0.95 ms/layer,
    // 26% of attention) — see benchmarks/laguna-xs README head-gate note.
    static let headGateFuse = ButterEnv["BUTTER_LAGUNA_HEAD_GATE_FUSE"] != "0"
}

enum LagunaLastRowPrefill {
    static let enabled: Bool = ButterEnv["BUTTER_LAGUNA_LASTROW_PREFILL"] != "0"
}

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
            if let s = ButterEnv["BUTTER_LAGUNA_MAX_SEQ"],
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
        if ButterEnv["BUTTER_LAGUNA_LM_HEAD_MXFP8"] == "1",
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
        // w21: build the planar int5 coarse copy from the BF16 lm_head weight
        // when BUTTER_LAGUNA_LMHEAD_INT5=1. Untimed init; the range guard can
        // decline (returns nil) and the prune keeps the MXFP8 arm. Only worth
        // building when the prune can actually run (needs the BF16 head).
        var lmHeadInt5: LagunaLmHeadPrune.LagunaInt5Coarse? = nil
        if LagunaLmHeadPrune.int5Enabled,
            let lin = lmHead.inner as? Linear,
            lin.weight.dtype == .bf16,
            lin.weight.shape == [100_352, 2048]
        {
            lmHeadInt5 = LagunaLmHeadPrune.LagunaInt5Coarse.build(
                from: lin.weight, device: device)
            fputs(
                lmHeadInt5 != nil
                    ? "[laguna] lm_head: int5 coarse copy built (w21, ~-36% coarse bytes)\n"
                    : "[laguna] lm_head: int5 coarse copy declined; using MXFP8 arm\n",
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
            lmHeadInt5: lmHeadInt5,
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
        ffn = .dense(
            LagunaDenseMLP(
                gate: gate, up: up, down: down, hidden: tc.hidden,
                dtype: dtype, device: device))
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
        dtype: dtype, device: device, layerIndex: i)
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
        ButterEnv["BUTTER_LAGUNA_PATHB"] == "0"
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
        layerIndex: layerIndex,
        dtype: actDtype, device: device)
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
    /// T27 Stage 2: L00 is Laguna XS's one dense (non-MoE) FFN layer.
    /// Its gate/up/down GEMMs are plain BF16 (`hidden=2048`,
    /// `intermediate=8192`, both %64==0 for this board) and were still
    /// riding the naive `Ops.gemm` path after T26 only wired the
    /// attention Q/K/V/O projections. Pre-transposed once at load time,
    /// same pattern as `LagunaLayer.qWeightT` etc. `nil` when the
    /// corresponding proj isn't a plain bf16 Linear (falls back safely).
    /// See benchmarks/laguna-xs/README.md T27.
    ///
    /// F-85: `nil` ALSO when `LagunaMppGemm.staticEligible` holds for the
    /// projection -- MPP takes the raw untransposed weight directly, so
    /// every real (t%64==0) forward call routes through it and the steel
    /// weightT copy would be dead weight (a full [outDim,inDim] bf16
    /// buffer + a one-time transpose dispatch, never read). See
    /// `LagunaMppGemm.swift`'s header for the full "lazy build" reasoning.
    let gateWeightT, upWeightT, downWeightT: Tensor?
    private let intermediate: Int

    init(gate: AnyLinear, up: AnyLinear, down: AnyLinear, hidden: Int, dtype: DType = .bf16, device: Device = .shared) {
        self.gate = gate
        self.up = up
        self.down = down
        self.hidden = hidden
        self.gateWeightT = LagunaMppGemm.staticEligible(gate, dtype: dtype)
            ? nil : LagunaSteelGemm.weightT(gate, dtype: dtype, device: device)
        self.upWeightT = LagunaMppGemm.staticEligible(up, dtype: dtype)
            ? nil : LagunaSteelGemm.weightT(up, dtype: dtype, device: device)
        self.downWeightT = LagunaMppGemm.staticEligible(down, dtype: dtype)
            ? nil : LagunaSteelGemm.weightT(down, dtype: dtype, device: device)
        // intermediate = gate/up's outDim = down's inDim. Only needed for
        // the eligibility check below; derive from whichever proj is a
        // plain Linear (falls back to 0, ineligible, if none are).
        if let l = gate.inner as? Linear {
            self.intermediate = l.weight.shape[0]
        } else if let l = down.inner as? Linear {
            self.intermediate = l.weight.shape[1]
        } else {
            self.intermediate = 0
        }
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
        // F-85: MPP first, steel second, naive `AnyLinear.callMany` last.
        // See `LagunaMppGemm.tryMppGemm`'s header for why this is the one
        // hook every call site uses instead of re-deriving the tiering.
        func projOrMpp(
            _ linear: AnyLinear, weightT: Tensor?, xIn: Tensor, outDim: Int, inDim: Int
        ) -> Tensor {
            if let mpp = LagunaMppGemm.tryMppGemm(
                linear, x: xIn, t: t, outDim: outDim, inDim: inDim, on: cmd, device: device)
            {
                return mpp
            }
            if let weightT,
                LagunaSteelGemm.eligible(t: t, outDim: outDim, inDim: inDim, dtype: xIn.dtype)
            {
                return LagunaSteelGemm.callMany(
                    input: xIn, weightT: weightT, t: t, outDim: outDim, inDim: inDim,
                    on: cmd, device: device)
            }
            return linear.callMany(xIn, t: t, on: cmd, device: device)
        }
        let g = projOrMpp(gate, weightT: gateWeightT, xIn: x, outDim: intermediate, inDim: hidden)
        let u = projOrMpp(up, weightT: upWeightT, xIn: x, outDim: intermediate, inDim: hidden)
        _ = Ops.swiglu(gate: g, up: u, on: cmd, into: u)
        return projOrMpp(down, weightT: downWeightT, xIn: u, outDim: hidden, inDim: intermediate)
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
    /// Layer index for diagnostic logging only (T26 attn phase timing).
    /// `-1` when not supplied by the caller (tests etc.); logging just
    /// prints "L-1" in that case, no functional dependency on it.
    let layerIndex: Int
    /// T26 Stage 2: pre-transposed `[inDim,outDim]` bf16 weights for the
    /// `iron_steel_gemm_64x64x16_2x2_bf16` MMA GEMM fast path (see
    /// LagunaSteelGemm.swift). `nil` when the corresponding proj isn't a
    /// plain bf16 Linear (e.g. quantized). Computed once at load time.
    let qWeightT, kWeightT, vWeightT, oWeightT: Tensor?
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
    /// Attention-side group-32 affine INT8 requant (see
    /// `LagunaAttnInt8.swift`). Built lazily on first decode
    /// call, once, from the resident BF16 Q/K/V/gate/o_proj weights.
    /// `nil` after a declined attempt (shape/dtype mismatch or
    /// `BUTTER_LAGUNA_ATTN_INT8=0`); `attnInt8Attempted` distinguishes
    /// "not tried yet" from "tried and declined" so we only pay the
    /// one-time setup dispatch once either way.
    private var attnInt8Attempted = false
    private var attnInt8QKVBank: LagunaAttnInt8.Bank?
    private var attnInt8GateRows = 0
    private var attnInt8OProjBank: LagunaAttnInt8.Bank?
    private var attnInt8Scratch: Tensor?
    private var attnInt8QScratch: Tensor?
    private var attnInt8KScratch: Tensor?
    private var attnInt8VScratch: Tensor?
    private var attnInt8GateScratch: Tensor?
    private var attnInt8OProjScratch: Tensor?
    /// NVFP4 group-16 attention tail (see `LagunaAttnNvfp4Tail.swift`).
    /// Mutually exclusive with `attnInt8QKVBank`/`attnInt8OProjBank` per
    /// layer: `ensureAttnInt8` builds ONE of the two flavors for a
    /// given layer, never both (`LagunaAttnNvfp4Tail.appliesToLayer`
    /// decides which, gated behind the same `attnInt8Attempted` /
    /// `LagunaAttnInt8.enabled` one-time setup). Decode reuses the
    /// `attnInt8*Scratch` bf16 buffers above for either flavor's output
    /// (they're generic bf16 Q/K/V(+gate)/o_proj holding buffers, not
    /// int8-specific).
    private var attnNvfp4QKVBank: LagunaAttnNvfp4Tail.Bank?
    private var attnNvfp4GateRows = 0
    private var attnNvfp4OProjBank: LagunaAttnNvfp4Tail.Bank?
    /// T24 gated-o_proj fusion scratch (`LagunaGatedOProjFuse`). Kernel 1's
    /// gated-row output (feeds the INT8 o_proj QMV bank when active).
    private var fusedGatedScratch: Tensor?
    /// Kernel 2's final `o_proj` output (plain BF16 `o_proj` path only).
    private var fusedOProjScratch: Tensor?
    /// Kernel 2's small `[nHeads]` post-softplus gate-values scratch.
    private var fusedGateValuesScratch: Tensor?
    /// GPU path may still commit on CPU-router MoE fallback.
    var commitsCommandBuffer: Bool { true }

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        gProj: AnyLinear?, qNorm: RMSNorm, kNorm: RMSNorm,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        ffn: LagunaFFN,
        hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int, rotaryDim: Int,
        ropeTheta: Float, isSliding: Bool, isFullYarn: Bool, dtype: DType,
        device: Device = .shared, layerIndex: Int = -1
    ) {
        self.layerIndex = layerIndex
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

        // T26 Stage 2: pre-transpose plain-bf16 attention weights once at
        // load time for the steel MMA GEMM fast path. Cheap (one-off,
        // off the hot path) and independent of the qkvFusedWeight decode
        // path below; this feeds LagunaLayer.forwardMulti's prefill GEMMs.
        // T27: factored into `LagunaSteelGemm.weightT` so L00's dense FFN
        // and the MoE router GEMM can reuse the same guard.
        //
        // F-85: skip building the steel weightT entirely when
        // `LagunaMppGemm.staticEligible` already covers this projection's
        // shape -- MPP takes the raw `[outDim,inDim]` weight directly, so
        // steel's transposed copy would never be read. See
        // `LagunaMppGemm.swift`'s header.
        self.qWeightT = LagunaMppGemm.staticEligible(qProj, dtype: dtype)
            ? nil : LagunaSteelGemm.weightT(qProj, dtype: dtype, device: device)
        self.kWeightT = LagunaMppGemm.staticEligible(kProj, dtype: dtype)
            ? nil : LagunaSteelGemm.weightT(kProj, dtype: dtype, device: device)
        self.vWeightT = LagunaMppGemm.staticEligible(vProj, dtype: dtype)
            ? nil : LagunaSteelGemm.weightT(vProj, dtype: dtype, device: device)
        self.oWeightT = LagunaMppGemm.staticEligible(oProj, dtype: dtype)
            ? nil : LagunaSteelGemm.weightT(oProj, dtype: dtype, device: device)

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
            // Fused QKV weight replaces three separate BF16 Linears for
            // single-GEMV decode; it's read on every decode token for
            // the lifetime of the model, so pin it like any other
            // resident weight (same rationale as KVCache.markWeightsResident).
            device.markWeightsResident([buf])
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

    /// Lazy, once-only build of the attention side-requant layout:
    /// group-32 affine INT8 (`LagunaAttnInt8`, T20) for most layers, or
    /// group-16 NVFP4 (`LagunaAttnNvfp4Tail`, T41) for late layers when
    /// `LagunaAttnNvfp4Tail.appliesToLayer(layerIndex)` holds. Declines
    /// silently (leaves every bank `nil`) unless every projection is a
    /// plain bias-free BF16 `Linear` with `hidden` a multiple of 32; the
    /// stock BF16 decode path below is always the fallback. Runs on its
    /// own tiny setup command buffer, synchronously, once per layer on
    /// the layer's first decode call; the concatenation step is pure CPU
    /// memcpy so it must happen only after that buffer's GPU work
    /// completes. A layer builds exactly ONE of the two bank flavors,
    /// never both.
    private func ensureAttnInt8(device: Device) {
        guard !attnInt8Attempted else { return }
        attnInt8Attempted = true
        guard LagunaAttnInt8.enabled,
            let lq = qProj.inner as? Linear, let lk = kProj.inner as? Linear,
            let lv = vProj.inner as? Linear, let lo = oProj.inner as? Linear,
            lq.weight.dtype == .bf16, lk.weight.dtype == .bf16,
            lv.weight.dtype == .bf16, lo.weight.dtype == .bf16, lo.bias == nil,
            lq.weight.shape == [nHeads * headDim, hidden],
            lk.weight.shape == [nKVHeads * headDim, hidden],
            lv.weight.shape == [nKVHeads * headDim, hidden],
            lo.weight.shape == [hidden, nHeads * headDim],
            hidden.isMultiple(of: LagunaAttnInt8.groupSize),
            (nHeads * headDim).isMultiple(of: LagunaAttnInt8.groupSize)
        else { return }

        if LagunaAttnNvfp4Tail.appliesToLayer(layerIndex) {
            let setupCmd = device.makeCommandBuffer()
            guard
                let qBank = LagunaAttnNvfp4Tail.quantizeG16(
                    weight: lq.weight, device: device, cmd: setupCmd),
                let kBank = LagunaAttnNvfp4Tail.quantizeG16(
                    weight: lk.weight, device: device, cmd: setupCmd),
                let vBank = LagunaAttnNvfp4Tail.quantizeG16(
                    weight: lv.weight, device: device, cmd: setupCmd)
            else { return }
            var banks = [qBank, kBank, vBank]
            var gateRows = 0
            var gBank: LagunaAttnNvfp4Tail.Bank?
            if let gProj, let lg = gProj.inner as? Linear, lg.bias == nil,
                lg.weight.dtype == .bf16, lg.weight.shape == [nHeads, hidden]
            {
                gBank = LagunaAttnNvfp4Tail.quantizeG16(
                    weight: lg.weight, device: device, cmd: setupCmd)
            }
            let oBank = LagunaAttnNvfp4Tail.quantizeG16(
                weight: lo.weight, device: device, cmd: setupCmd)
            setupCmd.commit()
            setupCmd.waitUntilCompleted()

            if let gBank {
                banks.append(gBank)
                gateRows = nHeads
            }
            guard let fusedQKV = LagunaAttnNvfp4Tail.concat(banks, device: device), let oBank
            else { return }
            attnNvfp4QKVBank = fusedQKV
            attnNvfp4GateRows = gateRows
            attnNvfp4OProjBank = oBank
            return
        }

        let setupCmd = device.makeCommandBuffer()
        guard
            let qBank = LagunaAttnInt8.quantizeG32(weight: lq.weight, device: device, cmd: setupCmd),
            let kBank = LagunaAttnInt8.quantizeG32(weight: lk.weight, device: device, cmd: setupCmd),
            let vBank = LagunaAttnInt8.quantizeG32(weight: lv.weight, device: device, cmd: setupCmd)
        else { return }
        var banks = [qBank, kBank, vBank]
        var gateRows = 0
        var gBank: LagunaAttnInt8.Bank?
        if let gProj, let lg = gProj.inner as? Linear, lg.bias == nil,
            lg.weight.dtype == .bf16, lg.weight.shape == [nHeads, hidden]
        {
            gBank = LagunaAttnInt8.quantizeG32(weight: lg.weight, device: device, cmd: setupCmd)
        }
        let oBank = LagunaAttnInt8.quantizeG32(weight: lo.weight, device: device, cmd: setupCmd)
        setupCmd.commit()
        setupCmd.waitUntilCompleted()

        if let gBank {
            banks.append(gBank)
            gateRows = nHeads
        }
        guard let fusedQKV = LagunaAttnInt8.concat(banks, device: device), let oBank else { return }
        attnInt8QKVBank = fusedQKV
        attnInt8GateRows = gateRows
        attnInt8OProjBank = oBank
    }

    /// Stock softplus per-head gate applied to already-computed raw logits
    /// (either the INT8 QKV bank's tail rows, or a `gProj(xNorm)` call the
    /// caller already made): softplus, GPU gather-expand [nHeads] ->
    /// [nHeads·headDim], broadcast-multiply. Fallback path when the T24
    /// `LagunaGatedOProjFuse` kernels are disabled or decline (shape/dtype
    /// mismatch).
    private func applySoftplusHeadGateFromLogits(
        _ attnOut: Tensor, gateLogits: Tensor, cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let gSoft = Ops.softplus(gateLogits, on: cmd)
        // W-headgate fuse: one broadcast pass replaces the expand-id gather
        // + separate mul (byte-identical). Escape `BUTTER_LAGUNA_HEAD_GATE_FUSE=0`.
        if LagunaEnvFlags.headGateFuse {
            let gate = gSoft.reshaped(to: [nHeads])
            return Ops.headGateMul(
                attn: attnOut, gate: gate, headDim: headDim, on: cmd)
        }
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

    /// Single-token decode. MoE path and softplus gate may commit `cmd`;
    /// the caller must refresh after `commitsCommandBuffer` layers.
    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
        yarn: Ops.RoPEYaRN,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        var work = cmd
        // T39: per-kernel decode GPU-time probe (BUTTER_LAGUNA_DECODE_KERNEL_TIMING=1).
        // `dk`/`dkB` are no-ops (return immediately, `work`/`workBuf` never
        // reassigned) when the env var is unset: zero behavior/perf change
        // on the default hot path, same discipline as T26's
        // `attnPhaseCheckpoint` for prefill. See benchmarks/laguna-xs/README.md T39.
        let dkTiming = LagunaDecodeKernelTiming.enabled
        func dk(_ label: String) {
            guard dkTiming else { return }
            work = LagunaDecodeKernelTiming.checkpoint(label, cmd: work, device: device)
        }
        let residual = h
        ensureAttnInt8(device: device)

        // Computed lazily: the norm-folded INT8 QKV bank path below folds
        // `inputNorm` into its own dispatch, so the hot case (bank present,
        // gate rows folded into it too) never needs the standalone
        // normalized row at all. Every other branch (int4 batched QKV,
        // dense fused QKV, plain per-projection, and the gate fallback
        // further down when the bank didn't fold the gate in) still does.
        var xNormCache: Tensor?
        func xNorm() -> Tensor {
            if let cached = xNormCache { return cached }
            let computed = inputNorm(residual, on: work)
            xNormCache = computed
            return computed
        }

        // Native group-32 affine INT8 QKV(+gate) bank: one batched QMV
        // dispatch replaces the BF16 fused-GEMV path below. Takes priority
        // over both the int4 and dense BF16 fused-QKV branches when
        // available. Late layers (T41) build a group-16 NVFP4 bank
        // instead (`LagunaAttnNvfp4Tail.appliesToLayer`), checked first
        // since the two banks are mutually exclusive per layer (only one
        // is ever non-nil); both write into the same `attnInt8*Scratch`
        // bf16 buffers, which are generic Q/K/V(+gate) holding buffers
        // independent of which quantization scheme produced them.
        // `qmvNormFolded` (BUTTER_LAGUNA_QKV_NORMFOLD, default ON) folds
        // `inputNorm`'s reduction into the INT8 bank's own dispatch so
        // that block's head is one dispatch instead of two; on decline
        // (shape/alignment guard, or the escape hatch) falls back to
        // `inputNorm` + `qmv` unchanged. The NVFP4 tail bank has no
        // norm-fold variant yet, so it always materializes `xNorm()`.
        var attnInt8Gate: Tensor?
        let q: Tensor
        let k: Tensor
        let v: Tensor
        if let bank = attnNvfp4QKVBank, residual.dtype == .bf16 {
            let qDim = nHeads * headDim
            let kDim = nKVHeads * headDim
            let vDim = nKVHeads * headDim
            let total = bank.outDim
            if attnInt8Scratch == nil || attnInt8Scratch!.elementCount != total {
                attnInt8Scratch = Tensor.empty(shape: [total], dtype: .bf16, device: device)
                let bytes = attnInt8Scratch!.dtype.byteSize
                attnInt8QScratch = Tensor(
                    buffer: attnInt8Scratch!.buffer, offset: attnInt8Scratch!.offset,
                    shape: [qDim], dtype: .bf16)
                attnInt8KScratch = Tensor(
                    buffer: attnInt8Scratch!.buffer,
                    offset: attnInt8Scratch!.offset + qDim * bytes,
                    shape: [kDim], dtype: .bf16)
                attnInt8VScratch = Tensor(
                    buffer: attnInt8Scratch!.buffer,
                    offset: attnInt8Scratch!.offset + (qDim + kDim) * bytes,
                    shape: [vDim], dtype: .bf16)
                if attnNvfp4GateRows == nHeads {
                    attnInt8GateScratch = Tensor(
                        buffer: attnInt8Scratch!.buffer,
                        offset: attnInt8Scratch!.offset + (qDim + kDim + vDim) * bytes,
                        shape: [nHeads], dtype: .bf16)
                }
            }
            // DECODE_QMV_V2: norm-folded single-dispatch QKV first
            // (the NVFP4 twin of the INT8 branch's `qmvNormFolded`
            // below); on decline (flag off, misaligned hidden) the
            // pre-existing `xNorm()` + `qmv` chain runs unchanged.
            let normFolded = LagunaAttnNvfp4Tail.normQmvV2(
                residual: residual, normWeight: inputNorm.weight, eps: inputNorm.eps,
                bank: bank, on: work, device: device, into: attnInt8Scratch!)
            if !normFolded {
                LagunaAttnNvfp4Tail.qmv(
                    input: xNorm(), bank: bank, on: work, device: device, into: attnInt8Scratch!)
            }
            q = attnInt8QScratch!
            k = attnInt8KScratch!
            v = attnInt8VScratch!
            if attnNvfp4GateRows == nHeads, let gateScratch = attnInt8GateScratch {
                attnInt8Gate = gateScratch
            }
        } else if let bank = attnInt8QKVBank, residual.dtype == .bf16 {
            let qDim = nHeads * headDim
            let kDim = nKVHeads * headDim
            let vDim = nKVHeads * headDim
            let total = bank.outDim
            if attnInt8Scratch == nil || attnInt8Scratch!.elementCount != total {
                attnInt8Scratch = Tensor.empty(shape: [total], dtype: .bf16, device: device)
                let bytes = attnInt8Scratch!.dtype.byteSize
                attnInt8QScratch = Tensor(
                    buffer: attnInt8Scratch!.buffer, offset: attnInt8Scratch!.offset,
                    shape: [qDim], dtype: .bf16)
                attnInt8KScratch = Tensor(
                    buffer: attnInt8Scratch!.buffer,
                    offset: attnInt8Scratch!.offset + qDim * bytes,
                    shape: [kDim], dtype: .bf16)
                attnInt8VScratch = Tensor(
                    buffer: attnInt8Scratch!.buffer,
                    offset: attnInt8Scratch!.offset + (qDim + kDim) * bytes,
                    shape: [vDim], dtype: .bf16)
                if attnInt8GateRows == nHeads {
                    attnInt8GateScratch = Tensor(
                        buffer: attnInt8Scratch!.buffer,
                        offset: attnInt8Scratch!.offset + (qDim + kDim + vDim) * bytes,
                        shape: [nHeads], dtype: .bf16)
                }
            }
            let normFolded = LagunaAttnInt8.qmvNormFolded(
                residual: residual, normWeight: inputNorm.weight, eps: inputNorm.eps,
                bank: bank, on: work, device: device, into: attnInt8Scratch!)
            if !normFolded {
                LagunaAttnInt8.qmv(
                    input: xNorm(), bank: bank, on: work, device: device, into: attnInt8Scratch!)
            }
            q = attnInt8QScratch!
            k = attnInt8KScratch!
            v = attnInt8VScratch!
            if let gateScratch = attnInt8GateScratch {
                attnInt8Gate = gateScratch
            }
        } else if let qq = qProj.inner as? QuantizedLinear, qq.bits == 4, qq.groupSize == 64,
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
                    || qkvFusedScratch!.dtype != xNorm().dtype
                {
                    qkvFusedScratch = Tensor.empty(
                        shape: [total], dtype: xNorm().dtype, device: device)
                    qOutScratch = Tensor(
                        buffer: qkvFusedScratch!.buffer,
                        offset: qkvFusedScratch!.offset,
                        shape: [qDim], dtype: xNorm().dtype)
                    kOutScratch = Tensor(
                        buffer: qkvFusedScratch!.buffer,
                        offset: qkvFusedScratch!.offset + qDim * xNorm().dtype.byteSize,
                        shape: [kDim], dtype: xNorm().dtype)
                    vOutScratch = Tensor(
                        buffer: qkvFusedScratch!.buffer,
                        offset: qkvFusedScratch!.offset
                            + (qDim + kDim) * xNorm().dtype.byteSize,
                        shape: [vDim], dtype: xNorm().dtype)
                }
                Ops.batchedQkvQgemvInt4Fast(
                    x: xNorm(),
                    wQ: qq.weight, scalesQ: qq.scales, biasesQ: qq.biases,
                    wK: qk.weight, scalesK: qk.scales, biasesK: qk.biases,
                    wV: qv.weight, scalesV: qv.scales, biasesV: qv.biases,
                    outQ: qDim, outK: kDim, outV: vDim,
                    on: work, into: qkvFusedScratch!)
                q = qOutScratch!
                k = kOutScratch!
                v = vOutScratch!
            } else {
                q = qProj(xNorm(), on: work)
                k = kProj(xNorm(), on: work)
                v = vProj(xNorm(), on: work)
            }
        } else if let qkvW = qkvFusedWeight,
            qkvW.dtype == xNorm().dtype,
            !LagunaEnvFlags.serialQkv
        {
            // Single dense GEMV over stacked [Q;K;V] (NVFP4 board: BF16 attn).
            let qDim = qOutDim
            let kDim = kOutDim
            let total = qDim + kDim + vOutDim
            if qkvFusedScratch == nil
                || qkvFusedScratch!.elementCount != total
                || qkvFusedScratch!.dtype != xNorm().dtype
            {
                qkvFusedScratch = Tensor.empty(
                    shape: [total], dtype: xNorm().dtype, device: device)
                qOutScratch = Tensor(
                    buffer: qkvFusedScratch!.buffer,
                    offset: qkvFusedScratch!.offset,
                    shape: [qDim], dtype: xNorm().dtype)
                kOutScratch = Tensor(
                    buffer: qkvFusedScratch!.buffer,
                    offset: qkvFusedScratch!.offset + qDim * xNorm().dtype.byteSize,
                    shape: [kDim], dtype: xNorm().dtype)
                vOutScratch = Tensor(
                    buffer: qkvFusedScratch!.buffer,
                    offset: qkvFusedScratch!.offset
                        + (qDim + kDim) * xNorm().dtype.byteSize,
                    shape: [vOutDim], dtype: xNorm().dtype)
            }
            _ = Ops.gemv(
                weight: qkvW, input: xNorm(), on: work, into: qkvFusedScratch!)
            q = qOutScratch!
            k = kOutScratch!
            v = vOutScratch!
        } else {
            q = qProj(xNorm(), on: work)
            k = kProj(xNorm(), on: work)
            v = vProj(xNorm(), on: work)
        }
        dk("qkv_qmv")

        // F86: BUTTER_LAGUNA_FUSED_ATTN_STEP (default OFF, opt-in
        // until validated on the hardware gates) collapses the
        // sliding-layer decode attention chain [QK-norm+RoPE] -> [KV
        // append] -> [GQA2 SDPA] into ONE dispatch. See
        // LagunaFusedSlidingAttnStep.swift for the full design and the
        // bit-exactness argument. Declines (falls through to the
        // unchanged stock chain below) for full-attention layers,
        // non-bf16, a quantized cache, or any GQA-pairing shape
        // mismatch.
        let fusedAttnStepCache = cache as? KVCache
        let fusedAttnStepFuses =
            !dkTiming && fusedAttnStepCache != nil
            && LagunaFusedSlidingAttnStep.canFuse(
                isSliding: isSliding, rotaryDim: rotaryDim, headDim: headDim,
                nHeads: nHeads, nKVHeads: nKVHeads, dtype: q.dtype)

        // F-fullattn: BUTTER_LAGUNA_FUSED_FULL_ATTN_STEP (default ON,
        // opt-in) is the full-attention twin of the sliding fusion
        // above: collapses [RMSNorm + partial YaRN RoPE] -> [KV
        // append] -> [GQA2 SDPA] into ONE dispatch for full-attention
        // layers. See LagunaFusedFullAttnStep.swift for the full
        // design. `LagunaFusedSlidingAttnStep.canFuse` requires
        // `isSliding == true` and `LagunaFusedFullAttnStep.canFuse`
        // requires `isSliding == false`, so these two fused branches
        // are mutually exclusive by construction.
        let fusedFullAttnStepFuses =
            !dkTiming && fusedAttnStepCache != nil
            && LagunaFusedFullAttnStep.canFuse(
                isSliding: isSliding, rotaryDim: rotaryDim, headDim: headDim,
                nHeads: nHeads, nKVHeads: nKVHeads, dtype: q.dtype)

        var attnOut: Tensor
        if fusedAttnStepFuses, let fusedCache = fusedAttnStepCache {
            attnOut = LagunaFusedSlidingAttnStep.apply(
                rawQ: q, rawK: k, rawV: v,
                qWeight: qNorm.weight, kWeight: kNorm.weight,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                position: position, thetaBase: ropeTheta, eps: qNorm.eps,
                cache: fusedCache, scale: scale,
                on: work, device: device)
            dk("qknorm_rope")
            dk("kv_append")
            dk("sdpa_gqa2")
        } else if fusedFullAttnStepFuses, let fusedCache = fusedAttnStepCache {
            attnOut = LagunaFusedFullAttnStep.apply(
                rawQ: q, rawK: k, rawV: v,
                qWeight: qNorm.weight, kWeight: kNorm.weight,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, rotaryDim: rotaryDim,
                position: position, thetaBase: ropeTheta, eps: qNorm.eps,
                yarn: yarn,
                cache: fusedCache, scale: scale,
                on: work, device: device)
            dk("qknorm_rope")
            dk("kv_append")
            dk("sdpa_gqa2")
        } else {
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
            dk("qknorm_rope")

            let q2d = qN.reshaped(to: [nHeads, headDim])
            let k2d = kN.reshaped(to: [nKVHeads, headDim])
            let v2d = v.reshaped(to: [nKVHeads, headDim])
            cache.appendOnGPU(kFlat: k2d, vFlat: v2d, on: work)

            let (cacheK, cacheV) = cache.prepareForAttention(on: work)
            let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: cache.length)
            dk("kv_append")
            // T24 kernel 3: GQA head-pairing + ALPHASKIP decode SDPA (opt-out
            // BUTTER_LAGUNA_SDPA_GQA2=0). Declines outside its shape envelope
            // (headDim==128, bf16, heads_per_group even) and falls back to the
            // stock `Ops.sdpaDecode`.
            attnOut =
                (LagunaSdpaDecodeGqa2.apply(
                    q: q2d, k: cacheK, v: cacheV,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    nKV: cache.length, kvStride: cache.maxSeq, scale: scale,
                    on: work, device: device,
                    sinkEnd: sinkEnd, windowStart: windowStart)
                    ?? Ops.sdpaDecode(
                        q: q2d, k: cacheK, v: cacheV,
                        nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                        nKV: cache.length, kvStride: cache.maxSeq,
                        scale: scale, on: work,
                        sinkEnd: sinkEnd, windowStart: windowStart)
                ).reshaped(to: [nHeads * headDim])
            dk("sdpa_gqa2")
        }

        // Softplus per-head gate before o_proj (GPU expand — no host sync).
        // The INT8 QKV bank's gate tail (raw logits) takes priority when
        // present; otherwise fall back to the stock `gProj(xNorm)` call.
        // Both sources land at the same [nHeads] raw-logit shape, which is
        // what both T24 fused kernels below consume directly.
        var workBuf = work
        func dkB(_ label: String) {
            guard dkTiming else { return }
            workBuf = LagunaDecodeKernelTiming.checkpoint(label, cmd: workBuf, device: device)
        }
        let gateLogitsRaw: Tensor? = attnInt8Gate ?? gProj?(xNorm(), on: workBuf)

        let oOut: Tensor
        if let oBank = attnNvfp4OProjBank, attnOut.dtype == .bf16 {
            if attnInt8OProjScratch == nil {
                attnInt8OProjScratch = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
            }
            // T41 NVFP4 sibling of the T32-task-2 gate fold below: fold
            // the gate-product-softplus multiply into the NVFP4 QMV's own
            // activation prologue when possible, same fallback chain.
            var gatedDone = false
            if let gateLogits = gateLogitsRaw {
                gatedDone = LagunaAttnNvfp4Tail.qmvGated(
                    input: attnOut, gateLogits: gateLogits, bank: oBank,
                    heads: nHeads, headDim: headDim,
                    on: workBuf, device: device, into: attnInt8OProjScratch!)
            }
            if !gatedDone {
                if let gateLogits = gateLogitsRaw {
                    if fusedGatedScratch == nil
                        || fusedGatedScratch!.elementCount != nHeads * headDim
                    {
                        fusedGatedScratch = Tensor.empty(
                            shape: [nHeads * headDim], dtype: .bf16, device: device)
                    }
                    let fused = LagunaGatedOProjFuse.applyGateProductSoftplus(
                        attnOut: attnOut, gateLogits: gateLogits, heads: nHeads, headDim: headDim,
                        on: workBuf, device: device, into: fusedGatedScratch!)
                    attnOut =
                        fused
                        ? fusedGatedScratch!
                        : applySoftplusHeadGateFromLogits(
                            attnOut, gateLogits: gateLogits, cmd: workBuf, device: device)
                }
                LagunaAttnNvfp4Tail.qmv(
                    input: attnOut, bank: oBank, on: workBuf, device: device,
                    into: attnInt8OProjScratch!)
            }
            oOut = attnInt8OProjScratch!
        } else if let oBank = attnInt8OProjBank, attnOut.dtype == .bf16 {
            if attnInt8OProjScratch == nil {
                attnInt8OProjScratch = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
            }
            // T32 task 2: fold the gate-product-softplus multiply into the
            // INT8 QMV's own activation prologue: one dispatch replaces
            // the T24-kernel-1 gate fuse + this QMV (two dispatches).
            var gatedDone = false
            if let gateLogits = gateLogitsRaw {
                gatedDone = LagunaAttnInt8.qmvGated(
                    input: attnOut, gateLogits: gateLogits, bank: oBank,
                    heads: nHeads, headDim: headDim,
                    on: workBuf, device: device, into: attnInt8OProjScratch!)
            }
            if !gatedDone {
                // Fallback: T24 kernel 1 gate-product-softplus fuse feeds
                // the T20 INT8 o_proj QMV bank unchanged (stock chain when
                // `BUTTER_LAGUNA_INT8_OPROJ_GATED=0`, or no gate present).
                if let gateLogits = gateLogitsRaw {
                    if fusedGatedScratch == nil
                        || fusedGatedScratch!.elementCount != nHeads * headDim
                    {
                        fusedGatedScratch = Tensor.empty(
                            shape: [nHeads * headDim], dtype: .bf16, device: device)
                    }
                    let fused = LagunaGatedOProjFuse.applyGateProductSoftplus(
                        attnOut: attnOut, gateLogits: gateLogits, heads: nHeads, headDim: headDim,
                        on: workBuf, device: device, into: fusedGatedScratch!)
                    attnOut =
                        fused
                        ? fusedGatedScratch!
                        : applySoftplusHeadGateFromLogits(
                            attnOut, gateLogits: gateLogits, cmd: workBuf, device: device)
                }
                LagunaAttnInt8.qmv(
                    input: attnOut, bank: oBank, on: workBuf, device: device,
                    into: attnInt8OProjScratch!)
            }
            oOut = attnInt8OProjScratch!
        } else {
            // T24 kernel 2: plain BF16 o_proj path, fold the gate multiply
            // directly into the o_proj GEMV (unroll 2); the gated row never
            // materializes. Needs a post-softplus (activated) [nHeads] gate,
            // a cheap dispatch vs. the O(hidden)-wide gather+mul it replaces.
            var fusedOut: Tensor?
            if let gateLogits = gateLogitsRaw, let lo = oProj.inner as? Linear,
                lo.bias == nil, lo.weight.dtype == .bf16, attnOut.dtype == .bf16
            {
                if fusedGateValuesScratch == nil {
                    fusedGateValuesScratch = Tensor.empty(
                        shape: [nHeads], dtype: .bf16, device: device)
                }
                _ = Ops.softplus(gateLogits, on: workBuf, into: fusedGateValuesScratch!)
                if fusedOProjScratch == nil {
                    fusedOProjScratch = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
                }
                let ok = LagunaGatedOProjFuse.applyGatedOutputProjection(
                    attentionOutput: attnOut, gateValues: fusedGateValuesScratch!,
                    weight: lo.weight, heads: nHeads, headDim: headDim, hidden: hidden,
                    on: workBuf, device: device, into: fusedOProjScratch!)
                if ok { fusedOut = fusedOProjScratch! }
            }
            if let fusedOut {
                oOut = fusedOut
            } else {
                if let gateLogits = gateLogitsRaw {
                    attnOut = applySoftplusHeadGateFromLogits(
                        attnOut, gateLogits: gateLogits, cmd: workBuf, device: device)
                }
                oOut = oProj(attnOut, on: workBuf)
            }
        }
        dkB("oproj_gated")
        // Fused residual + post-attn RMSNorm when hidden ≤ 4096 (Laguna = 2048).
        // T28 lever 2: when the FFN is MoE with a plain-BF16-Linear router
        // gate at Laguna XS's exact shape, fold the router GEMV into the
        // SAME dispatch too (LagunaResidualRmsRouterFuse): the normed row
        // is already resident in the fused kernel's threadgroup memory, so
        // the router GEMV costs no extra launch. `moe.decode` then skips
        // its own `gate(h, on: cmd)` via `precomputedLogits`.
        let h2: Tensor
        let ffnIn: Tensor
        var precomputedRouterLogits: Tensor?
        if case .moe(let moe) = ffn,
            LagunaResidualRmsRouterFuse.enabled,
            hidden == 2048, moe.router.nExperts == 256,
            OpsValidation.validateAddRmsNorm(n: hidden) == nil,
            let gateLinear = moe.gate.inner as? Linear,
            gateLinear.bias == nil, gateLinear.weight.dtype == .bf16
        {
            let fused = LagunaResidualRmsRouterFuse.apply(
                residual: residual, attnOut: oOut,
                normWeight: postAttnNorm.weight, eps: postAttnNorm.eps,
                routerWeight: gateLinear.weight,
                hidden: hidden, nExperts: moe.router.nExperts,
                on: workBuf, device: device)
            h2 = fused.residual
            ffnIn = fused.normed
            precomputedRouterLogits = fused.logits
        } else if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                residual, oOut, weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: 1, rowSize: hidden, on: workBuf)
            h2 = fused.residual
            ffnIn = fused.normed
        } else {
            h2 = Ops.add(residual, oOut, on: workBuf)
            ffnIn = postAttnNorm(h2, on: workBuf)
        }
        dkB("residual_norm_pre_moe")

        switch ffn {
        case .dense(let mlp):
            let ffnOut = mlp.forward(ffnIn, on: workBuf)
            let out = Ops.add(h2, ffnOut, on: workBuf)
            dkB("l00_dense_ffn")
            return out
        case .moe(let moe):
            // Full Path B: residual on same CB, one commit+wait at layer end.
            // Host/hybrid commits mid-decode — residual on fresh CB + wait.
            //
            // T43: always call through `decodeFoldingResidual` and pass
            // `h2` (the pre-FFN residual). `moe` itself decides whether it
            // can fold (nine-slot-eligible shapes + `BUTTER_LAGUNA_DOWN_RESIDUAL_FOLD=1`)
            // — when it does, `residualFolded` comes back true and `ffnOut`
            // already IS `h2 + ffn(ffnIn)`, so the `Ops.add` below (and the
            // `workBuf.status` fresh-CB fallback branch) are skipped
            // entirely: one dispatch fewer per MoE decode layer. When it
            // declines (flag off, non-nine-slot decode branch, shape
            // mismatch), `ffnOut` is exactly what `decode` /
            // `decodeWithPrecomputedLogits` would have returned and every
            // path below behaves exactly as before this change.
            // T43 fix: pre-allocate the dedicated per-layer output tensor
            // and hand it to the fold as the down-reduce destination, so
            // the folded layer output has the same tensor identity as the
            // stock chain's Ops.add result (never accumulatorScratch --
            // see the aliasing note above the stock Ops.add below).
            if ffnOutScratch == nil
                || ffnOutScratch!.dtype != ffnIn.dtype
                || ffnOutScratch!.elementCount != hidden
            {
                ffnOutScratch = Tensor.empty(
                    shape: [hidden], dtype: ffnIn.dtype, device: device)
            }
            let ffnOut: Tensor
            let residualFolded: Bool
            if let logits = precomputedRouterLogits {
                (ffnOut, residualFolded) = moe.decodeFoldingResidual(
                    ffnIn, position: position,
                    cache: StatelessLayerCache(),
                    cmd: workBuf, device: device,
                    precomputedLogits: logits, residual: h2,
                    foldInto: ffnOutScratch!)
            } else {
                (ffnOut, residualFolded) = moe.decodeFoldingResidual(
                    ffnIn, position: position,
                    cache: StatelessLayerCache(),
                    cmd: workBuf, device: device,
                    precomputedLogits: nil, residual: h2,
                    foldInto: ffnOutScratch!)
            }
            if residualFolded {
                dkB("l00_moe_ffn_residual_folded")
                return ffnOut
            }
            if workBuf.status == .committed || workBuf.status == .completed
                || workBuf.status == .error
            {
                let addCmd = device.makeCommandBuffer()
                let out = Ops.add(h2, ffnOut, on: addCmd)
                if dkTiming {
                    _ = LagunaDecodeKernelTiming.checkpoint(
                        "residual_add_post_moe", cmd: addCmd, device: device)
                } else {
                    addCmd.commit()
                    addCmd.waitUntilCompleted()
                }
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
            if LagunaEnvFlags.ffnCopy {
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

        // T26: break the `pre` (attention) phase T25 found dominant
        // (74.9% of MoE-layer prefill wall-clock) into QKV-proj /
        // qk-norm / rope+kv / sdpa / gate / o_proj / add+postAttnNorm
        // via forced commit+wait checkpoints, same technique as T25's
        // BUTTER_LAGUNA_MOE_PHASE_TIMING. Diagnostic only: `work` only
        // gets reassigned inside `attnPhaseCheckpoint`, whose body is a
        // no-op (`guard attnPhaseTiming else { return }`) when the env
        // var is unset, so this is zero behavior change by default.
        // See benchmarks/laguna-xs/README.md T26.
        let attnPhaseTiming = LagunaEnvFlags.attnPhaseTiming
        var attnPhaseTimes: [(String, Double)] = []
        var attnPhaseT0 = CFAbsoluteTimeGetCurrent()
        var work = cmd
        // Buffer-passing variant so callees that own their own local `cmd`
        // (e.g. LagunaSdpa.prefillMmaBf16's internal pad/cast/transpose/
        // compact/mma/transpose-back stages) can checkpoint sub-phases and
        // get back the fresh buffer to keep encoding onto, without a
        // dangling reference to an already-committed one.
        func attnPhaseCheckpointBuf(_ label: String, _ buf: MTLCommandBuffer) -> MTLCommandBuffer {
            guard attnPhaseTiming else { return buf }
            let b = buf
            if b.status != .committed && b.status != .completed && b.status != .error {
                b.commit()
            }
            b.waitUntilCompleted()
            let ms = (CFAbsoluteTimeGetCurrent() - attnPhaseT0) * 1000
            attnPhaseTimes.append((label, ms))
            attnPhaseT0 = CFAbsoluteTimeGetCurrent()
            return device.makeCommandBuffer()
        }
        func attnPhaseCheckpoint(_ label: String) {
            guard attnPhaseTiming else { return }
            work = attnPhaseCheckpointBuf(label, work)
        }
        func attnPhaseFlush() {
            guard attnPhaseTiming, !attnPhaseTimes.isEmpty else { return }
            let total = attnPhaseTimes.reduce(0.0) { $0 + $1.1 }
            let parts = attnPhaseTimes.map {
                "\($0.0)=\(String(format: "%.2f", $0.1))ms"
            }.joined(separator: " ")
            fputs(
                "[laguna-attn-phase] L\(String(format: "%02d", layerIndex)) "
                    + "total=\(String(format: "%.2f", total))ms \(parts)\n",
                stderr)
        }

        let residual = h
        let xNormFlat = Ops.rmsNormRows(
            h.reshaped(to: [nRows * hidden]),
            weight: inputNorm.weight, eps: inputNorm.eps,
            nRows: nRows, rowSize: hidden, on: work)
        let xNorm = xNormFlat.reshaped(to: [nRows, hidden])
        attnPhaseCheckpoint("norm")

        // QKV projections (batched affine/dequant GEMM). F-85: MPP tensor-
        // ops GEMM first (see `LagunaMppGemm.tryMppGemm`), steel MMA GEMM
        // second when shapes are 64-aligned bf16 (prefill; decode T=1
        // always falls back to naive), naive `AnyLinear.callMany` last.
        // Escape: BUTTER_LAGUNA_MPP_GEMM=0 / BUTTER_LAGUNA_ATTN_STEEL_GEMM=0.
        func projOrMpp(
            _ linear: AnyLinear, weightT: Tensor?, x: Tensor, outDim: Int, inDim: Int
        ) -> Tensor {
            if let mpp = LagunaMppGemm.tryMppGemm(
                linear, x: x, t: nRows, outDim: outDim, inDim: inDim, on: work, device: device)
            {
                return mpp
            }
            if let weightT,
                LagunaSteelGemm.eligible(t: nRows, outDim: outDim, inDim: inDim, dtype: x.dtype)
            {
                return LagunaSteelGemm.callMany(
                    input: x, weightT: weightT, t: nRows, outDim: outDim, inDim: inDim,
                    on: work, device: device)
            }
            return linear.callMany(x, t: nRows, on: work, device: device)
        }
        var q: Tensor
        var k: Tensor
        let v: Tensor
        // F-85: when MPP statically covers all three of Q/K/V's shapes,
        // prefer three individual `projOrMpp` calls (each routes through
        // the faster MPP kernel) over T37's steel-concurrent-encoder
        // batch below -- per the w3/mpp-bench numbers, MPP's per-dispatch
        // win (2.2-3.3x vs steel) dwarfs the +1.3% T37 measured from
        // merging 3 steel dispatches into 1 encoder. Only falls through
        // to the steel-concurrent tier when MPP can't take at least one
        // of the three (quantized weight, odd shape, MPP disabled/
        // incapable) -- unchanged from T37 in that case.
        //
        // T37: candidate (c), Q/K/V as one concurrent-encoder Pair-of-3
        // instead of three separate serial encoders - only when all
        // three shapes are steel-eligible (same guard `projOrMpp` uses
        // per-projection). Default ON (measured +1.3% prefill@512,
        // byte-identical, decode flat); escape BUTTER_LAGUNA_QKV_CONCURRENT=0.
        // See LagunaSteelGemm.swift + README T37 for the full reasoning.
        let qkvAllMppEligible =
            LagunaMppGemm.eligible(t: nRows, outDim: qOutDim, inDim: hidden, dtype: xNorm.dtype)
            && LagunaMppGemm.eligible(t: nRows, outDim: kOutDim, inDim: hidden, dtype: xNorm.dtype)
            && LagunaMppGemm.eligible(t: nRows, outDim: vOutDim, inDim: hidden, dtype: xNorm.dtype)
        if !qkvAllMppEligible,
            LagunaSteelGemm.concurrentQkvEnabled,
            let qWeightT, let kWeightT, let vWeightT,
            LagunaSteelGemm.eligible(t: nRows, outDim: qOutDim, inDim: hidden, dtype: xNorm.dtype),
            LagunaSteelGemm.eligible(t: nRows, outDim: kOutDim, inDim: hidden, dtype: xNorm.dtype),
            LagunaSteelGemm.eligible(t: nRows, outDim: vOutDim, inDim: hidden, dtype: xNorm.dtype),
            let (qc, kc, vc) = LagunaSteelGemm.callManyQKVConcurrent(
                input: xNorm,
                qWeightT: qWeightT, qOutDim: qOutDim,
                kWeightT: kWeightT, kOutDim: kOutDim,
                vWeightT: vWeightT, vOutDim: vOutDim,
                t: nRows, inDim: hidden, on: work, device: device)
        {
            q = qc
            k = kc
            v = vc
        } else {
            q = projOrMpp(qProj, weightT: qWeightT, x: xNorm, outDim: qOutDim, inDim: hidden)
            k = projOrMpp(kProj, weightT: kWeightT, x: xNorm, outDim: kOutDim, inDim: hidden)
            v = projOrMpp(vProj, weightT: vWeightT, x: xNorm, outDim: vOutDim, inDim: hidden)
        }
        attnPhaseCheckpoint("qkv_proj")

        // T30: prefill sliding-layer fused QK-norm+RoPE. One dispatch
        // replaces `qknorm` + Q's share of `rope_kv` AND writes Q
        // directly into the padded `[nHeads,tPad,headDim]` f16 layout
        // `prefillMmaBf16` otherwise builds via its own pad/cast/transpose
        // dispatches (`sdpa_qpad` + `sdpa_qcast_transpose`), killing those
        // too, see benchmarks/laguna-xs/README.md T30. Sliding layers
        // only; full-attention (YaRN) layers keep the stock chain below.
        // Escape: BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE=0.
        let prefillFuseTPad = ((nRows + 31) / 32) * 32
        // Matches the old call site's `cache.length` value exactly: that
        // call ran AFTER this chunk's own K/V append (`cache.length` there
        // == `position + nRows`), so evaluating it here, before any
        // append, with the post-append length explicitly keeps the same
        // semantics if `sdpaSinkWindow` ever stops ignoring `nKV` (today
        // it unconditionally returns `(0, 0)`, per T26).
        let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: position + nRows)
        // T-blockmap: sliding-window prefill is real (not inert) once
        // BUTTER_LAGUNA_PREFILL_BLOCKMAP=1; see LagunaSdpa.swift's
        // `prefillBlockmapEnabled` header comment. `cache.effectiveMaxSize`
        // is the configured sliding-window size for a `.window` eviction
        // cache (Laguna sets `.window(maxSize: min(slidingWindow, cap),
        // keep: 0)` for sliding layers below), independent of the
        // currently-degenerate `cache.sdpaSinkWindow` indirection above.
        // The fused QK-norm/RoPE prefill path
        // (`prefillMmaTailFromHTD`) doesn't know about windowing at all,
        // so sliding layers must route around it (fall through to the
        // plain `prefillMmaBf16` branch below) whenever a real window
        // could apply, rather than silently keep treating them as full
        // causal.
        let blockmapWindow =
            (isSliding && LagunaSdpa.prefillBlockmapEnabled) ? cache.effectiveMaxSize : 0
        let prefillFuseEligible =
            LagunaSdpa.prefillQkRopeFuseEligible(
                isSliding: isSliding, rotaryDim: rotaryDim, headDim: headDim,
                dtype: q.dtype, nHeads: nHeads, nKVHeads: nKVHeads, nRows: nRows)
            && position + prefillFuseTPad <= cache.maxSeq
            && !(windowStart > 0 || sinkEnd > 0)
            && blockmapWindow == 0
        // T30 stretch: YaRN full-attention twin (partial rotary), same
        // fuse flag, same eligibility shape modulo isSliding/rotaryDim.
        let prefillYarnFuseEligible =
            LagunaSdpa.prefillQkYarnFuseEligible(
                isSliding: isSliding, rotaryDim: rotaryDim, headDim: headDim,
                dtype: q.dtype, nHeads: nHeads, nKVHeads: nKVHeads, nRows: nRows)
            && position + prefillFuseTPad <= cache.maxSeq
            && !(windowStart > 0 || sinkEnd > 0)
            && blockmapWindow == 0

        let attnOut: Tensor
        // T26 diagnosis: which branch this layer actually takes. Logged
        // alongside the phase breakdown so "sliding path dominates" can
        // be confirmed/refuted with real per-layer evidence instead of
        // assumed from `isSliding` alone (KVCacheProtocol's default
        // `sdpaSinkWindow` is `(0, 0)` (unoverridden today), so the
        // windowed branch below may never actually fire even on SWA
        // layers at T <= slidingWindow; see T26 README section).
        var attnPathLabel = "?"

        if prefillFuseEligible, let kv = cache as? KVCache {
            let (qHTD, kFused, slot) = LagunaSdpa.fusedQKNormRopeForPrefill(
                rawQ: q, rawK: k, qWeight: qNorm.weight, kWeight: kNorm.weight,
                nRows: nRows, tPad: prefillFuseTPad, position: position,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                thetaBase: ropeTheta, eps: qNorm.eps,
                on: work, device: device)
            attnPhaseCheckpoint("qknorm_rope_fused")

            let appendPositions = Tensor.empty(shape: [nRows], dtype: .u32, device: device)
            kv.reserveSlotsManyOnHost(t: nRows, into: appendPositions)
            kv.appendRangeOnGPUMany(
                kFlat: kFused, vFlat: v, t: nRows,
                positions: appendPositions, on: work)
            attnPhaseCheckpoint("rope_kv")

            let (cacheK, cacheV) = cache.prepareForAttention(on: work)
            attnPhaseCheckpoint("kv_prep")

            attnOut = LagunaSdpa.prefillMmaTailFromHTD(
                qHTD: qHTD, tPad: prefillFuseTPad, slot: slot,
                cacheK: cacheK, cacheV: cacheV,
                nRows: nRows, position: position,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                maxSeq: cache.maxSeq, scale: scale,
                on: &work, device: device,
                phaseCheckpoint: attnPhaseTiming ? attnPhaseCheckpointBuf : nil)
            attnPathLabel = "prefillMma+qkRopeFused"
            attnPhaseCheckpoint("sdpa[\(attnPathLabel)]")
        } else if prefillYarnFuseEligible, let kv = cache as? KVCache {
            let (qHTD, kFused, slot) = LagunaSdpa.fusedQKNormYarnForPrefill(
                rawQ: q, rawK: k, qWeight: qNorm.weight, kWeight: kNorm.weight,
                nRows: nRows, tPad: prefillFuseTPad, position: position,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: ropeTheta, eps: qNorm.eps, yarn: yarn,
                on: work, device: device)
            attnPhaseCheckpoint("qknorm_rope_fused")

            let appendPositions = Tensor.empty(shape: [nRows], dtype: .u32, device: device)
            kv.reserveSlotsManyOnHost(t: nRows, into: appendPositions)
            kv.appendRangeOnGPUMany(
                kFlat: kFused, vFlat: v, t: nRows,
                positions: appendPositions, on: work)
            attnPhaseCheckpoint("rope_kv")

            let (cacheK, cacheV) = cache.prepareForAttention(on: work)
            attnPhaseCheckpoint("kv_prep")

            attnOut = LagunaSdpa.prefillMmaTailFromHTD(
                qHTD: qHTD, tPad: prefillFuseTPad, slot: slot,
                cacheK: cacheK, cacheV: cacheV,
                nRows: nRows, position: position,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                maxSeq: cache.maxSeq, scale: scale,
                on: &work, device: device,
                phaseCheckpoint: attnPhaseTiming ? attnPhaseCheckpointBuf : nil)
            attnPathLabel = "prefillMma+qkYarnFused"
            attnPhaseCheckpoint("sdpa[\(attnPathLabel)]")
        } else {
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
        attnPhaseCheckpoint("qknorm")

        // RoPE + KV append for the whole chunk.
        lagunaBatchedRopeAndKV(
            q: &q, k: &k, v: v,
            nRows: nRows, position: position,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            rotaryDim: rotaryDim, ropeTheta: ropeTheta,
            isFullYarn: isFullYarn, yarn: yarn,
            cache: cache, cmd: work, device: device)
        attnPhaseCheckpoint("rope_kv")

        // Causal multi-query SDPA over the growing cache.
        let qForSdpa = q.reshaped(to: [nRows, nHeads, headDim])
        let (cacheK, cacheV) = cache.prepareForAttention(on: work)
        attnPhaseCheckpoint("kv_prep")
        // Sliding-window layers: sdpaMulti is causal-only; when window
        // applies, fall back to per-row decode SDPA (still better than a
        // full per-token layer graph for the FFN). Full layers use sdpaMulti.
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
            attnPathLabel = "windowed-decode-loop"
        } else if let mmaOut = LagunaSdpa.prefillMmaBf16(
            qTHD: qForSdpa, cacheK: cacheK, cacheV: cacheV,
            nRows: nRows, position: position,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            maxSeq: cache.maxSeq, scale: scale,
            on: &work, device: device,
            window: blockmapWindow, sinkEnd: 0,
            phaseCheckpoint: attnPhaseTiming ? attnPhaseCheckpointBuf : nil)
        {
            // sdpaPrefillMma (f16 MMA via bf16 cast) — big prefill lever
            // when T%32==0. Escape: BUTTER_LAGUNA_SDPA_PREFILL_MMA=0.
            // `window: blockmapWindow` is 0 (pure causal, unchanged
            // behavior) unless BUTTER_LAGUNA_PREFILL_BLOCKMAP=1 AND this
            // is a sliding layer, per the T-blockmap comment above.
            attnOut = mmaOut
            attnPathLabel = blockmapWindow > 0 ? "prefillMma+blockmapWindow" : "prefillMma"
        } else {
            attnOut = Ops.sdpaMulti(
                q: qForSdpa, k: cacheK, v: cacheV,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                baseKV: position, nQuery: nRows, kvStride: cache.maxSeq,
                causal: true, scale: scale, on: work
            ).reshaped(to: [nRows, nHeads * headDim])
            attnPathLabel = "sdpaMulti-fallback"
        }
        attnPhaseCheckpoint("sdpa[\(attnPathLabel)]")
        }

        // Softplus head gate for the chunk — host expand (same math as decode).
        var gated = attnOut
        if let gProj {
            gated = applySoftplusHeadGateManyHost(
                attnOut, xNorm: xNorm, gProj: gProj, t: nRows,
                cmd: &work, device: device)
        }
        attnPhaseCheckpoint("gate")

        let oOut = projOrMpp(
            oProj, weightT: oWeightT, x: gated, outDim: hidden, inDim: qOutDim)
        attnPhaseCheckpoint("oproj")
        let residualFlat = residual.reshaped(to: [nRows * hidden])
        let oFlat = oOut.reshaped(to: [nRows * hidden])
        let h2: Tensor
        let ffnIn: Tensor
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                residualFlat, oFlat, weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: nRows, rowSize: hidden, on: work)
            h2 = fused.residual.reshaped(to: [nRows, hidden])
            ffnIn = fused.normed.reshaped(to: [nRows, hidden])
        } else {
            h2 = Ops.add(residualFlat, oFlat, on: work).reshaped(to: [nRows, hidden])
            ffnIn = Ops.rmsNormRows(
                h2.reshaped(to: [nRows * hidden]),
                weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: nRows, rowSize: hidden, on: work
            ).reshaped(to: [nRows, hidden])
        }
        attnPhaseCheckpoint("addnorm")
        attnPhaseFlush()

        switch ffn {
        case .dense(let mlp):
            let ffnOut = mlp.forwardMany(ffnIn, t: nRows, on: work, device: device)
            let out = Ops.add(
                h2.reshaped(to: [nRows * hidden]),
                ffnOut.reshaped(to: [nRows * hidden]),
                on: work
            ).reshaped(to: [nRows, hidden])
            if work.status != .committed && work.status != .completed
                && work.status != .error
            {
                work.commit()
            }
            work.waitUntilCompleted()
            return out
        case .moe(let moe):
            // Batched gather MoE (decodeMany). Default ON — big prefill win.
            // MPP gather is quality-broken for Laguna affine4 on M5; force
            // simdgroup path. Escape: BUTTER_LAGUNA_MOE_MANY=0 → serial.
            // MPP re-enable: BUTTER_MOE_BGEMM_MPP=1 (and unset NO_MPP).
            if !LagunaEnvFlags.moeBgemmNoMpp
                && !LagunaEnvFlags.moeBgemmMpp
            {
                setenv("BUTTER_MOE_BGEMM_NO_MPP", "1", 0)
            }
            if LagunaEnvFlags.moeManyOff {
                let dtBytes = ffnIn.dtype.byteSize
                let outFlat = Tensor.empty(
                    shape: [nRows * hidden], dtype: ffnIn.dtype, device: device)
                var moeCmd = work
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
            // Stay on work when device plan keeps cmd open; residual on
            // same CB then commit without wait (model prefill pipe fences).
            //
            // W57: BUTTER_LAGUNA_FUSED_MOE_TAIL=1 folds this residual add
            // (+ the shared-expert add, inside MoELayer) into the MoE
            // combine-tail kernel itself (`Ops.moeUnpermuteTail`) — one
            // dispatch instead of combine + 2 separate `Ops.add` passes.
            // `moe.lastResidualFused` (set by `decodeMany` right before
            // returning) reports whether that actually happened for THIS
            // call; when false (flag off, token-order FMA path, or the
            // per-row swiGLU shared-expert fallback — see
            // `LagunaFusedMoeTail`'s file header), `ffnOutFlat` is the
            // plain unfused MoE output exactly as before and the residual
            // add below still runs, unchanged. Default OFF: passing
            // `fusedResidual` only changes behavior when the flag is set;
            // `decodeMany` itself gates on `LagunaFusedMoeTail.enabled`.
            let h2Flat = h2.reshaped(to: [nRows * hidden])
            let ffnOutFlat = moe.decodeMany(
                ffnIn.reshaped(to: [nRows * hidden]),
                t: nRows, cmd: work, device: device, commitAtEnd: false,
                fusedResidual: h2Flat)
            let residualFused = moe.lastResidualFused
            let out: Tensor
            if work.status == .committed || work.status == .completed
                || work.status == .error
            {
                // MoE committed the layer CB (e.g. isolated MPP path). Residual
                // must land on a fresh CB and be waited (or at least committed
                // into the prefill pipe) so the next layer does not race.
                if residualFused {
                    // Already added inside the fused tail kernel, on
                    // whatever (possibly rotated) command buffer produced
                    // `ffnOutFlat` — that buffer was necessarily committed
                    // ahead of `work` showing committed here (same queue,
                    // submission order), so waiting on `work` itself gives
                    // the same host-side completion guarantee the fresh
                    // `addCmd` below exists for, with no separate dispatch.
                    work.waitUntilCompleted()
                    return ffnOutFlat.reshaped(to: [nRows, hidden])
                }
                let addCmd = device.makeCommandBuffer()
                out = Ops.add(
                    h2Flat,
                    ffnOutFlat,
                    on: addCmd
                ).reshaped(to: [nRows, hidden])
                addCmd.commit()
                // Wait here: prefill pipe only tracks work, not this addCmd.
                addCmd.waitUntilCompleted()
                return out
            }
            if residualFused {
                out = ffnOutFlat.reshaped(to: [nRows, hidden])
            } else {
                out = Ops.add(
                    h2Flat,
                    ffnOutFlat,
                    on: work
                ).reshaped(to: [nRows, hidden])
            }
            if work.status != .committed && work.status != .completed
                && work.status != .error
            {
                work.commit()
            }
            return out
        }
    }

    /// Terminal-layer last-row prefill (see `LagunaLastRowPrefill`).
    /// Caller (`LagunaModel.forwardMulti`) invokes this INSTEAD of
    /// `forwardMulti` only for the model's final layer index, only when
    /// `nRows > 1` and `LagunaLastRowPrefill.enabled`.
    ///
    /// Phase A (full width, byte-identical to stock `forwardMulti`):
    /// input norm, K/V projection, K QK-norm+RoPE, batched KV-cache
    /// append. Every row's K/V is needed by the cache for future decode
    /// continuation, so this cannot be narrowed to one row.
    ///
    /// Phase B (last row only, reusing the T=1 decode kernels
    /// `forward()` already uses against the now-fully-populated cache):
    /// Q projection + QK-norm+RoPE, `LagunaSdpaDecodeGqa2`/`Ops.sdpaDecode`
    /// at q_len=1, the T24 gated-o_proj chain, and `MoELayer.decode` /
    /// `LagunaDenseMLP.forward`. Returns a flat `[hidden]` row (NOT
    /// `[1, hidden]`), same convention `forward()`'s return value uses,
    /// so the caller can feed it straight into `lagunaFinalNormLmHead`
    /// without an extra `slicedRows` step.
    ///
    /// Reuse map / non-reuse notes (see also the commit message + the
    /// F-lastrow report):
    ///   - Q projection: plain `qProj` GEMV, NOT the decode fused-QKV
    ///     banks (INT8 group-32 bank / int4 fast path / dense
    ///     `qkvFusedWeight`). Those banks compute Q+K+V TOGETHER in one
    ///     dispatch; splitting them to produce Q alone would either
    ///     recompute K/V that Phase A already wrote to the cache (waste)
    ///     or require a new kernel variant: out of scope for a plumbing
    ///     pass. This is the same plain-`qProj` branch decode's
    ///     `forward()` falls back to when none of its fused banks apply.
    ///   - QK-norm+RoPE: `ropeQOnly` mirrors `applyRopeInPlace`'s Q
    ///     branch exactly, but the fused JOINT Q+K decode kernel
    ///     (`LagunaFusedQKNormRoPE`) is not reusable here: it always
    ///     norms+ropes both a Q row AND a K row in one dispatch, and K
    ///     is already handled (batched) in Phase A; calling it would
    ///     mean uselessly recomputing K a second time just to discard
    ///     the result.
    ///   - SDPA: `LagunaSdpaDecodeGqa2.apply` / `Ops.sdpaDecode`, called
    ///     with the exact same contract `forward()` uses (`nKV: cache.length`
    ///     after Phase A's append, `cache.sdpaSinkWindow` for window/sink
    ///     bounds), fully reused, zero new kernels.
    ///   - o_proj + softplus gate: the exact `ensureAttnInt8` /
    ///     `attnInt8OProjBank` / `LagunaGatedOProjFuse` / plain-`oProj`
    ///     chain `forward()` uses, re-invoked here rather than factored
    ///     into a shared helper (keeps `forward()`'s tested control flow
    ///     untouched; this file has no build/test loop available to
    ///     verify a riskier refactor).
    ///   - Residual + post-attn norm: `Ops.addAndRmsNorm` /
    ///     `LagunaResidualRmsRouterFuse.apply`, same eligibility checks
    ///     `forward()` uses, `nRows: 1`.
    ///   - FFN: `MoELayer.decode` / `.decodeWithPrecomputedLogits` /
    ///     `LagunaDenseMLP.forward`: the same single-row (decode) FFN
    ///     entry points `forward()` calls per decode step, just invoked
    ///     once here instead of once per prefill row.
    func forwardMultiLastRow(
        _ h: Tensor, startingAt position: Int, cache: any KVCacheProtocol,
        yarn: Ops.RoPEYaRN,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let nRows = h.shape[0]
        precondition(h.shape == [nRows, hidden], "LagunaLayer.forwardMultiLastRow: shape \(h.shape)")
        precondition(nRows > 1, "forwardMultiLastRow: T=1 already takes the decode forward() path")
        let lastPos = position + nRows - 1

        var work = cmd
        let residualFull = h

        // ── Phase A: full width, byte-identical to stock forwardMulti ──
        let xNormFlat = Ops.rmsNormRows(
            h.reshaped(to: [nRows * hidden]),
            weight: inputNorm.weight, eps: inputNorm.eps,
            nRows: nRows, rowSize: hidden, on: work)
        let xNorm = xNormFlat.reshaped(to: [nRows, hidden])

        func projOrSteel(
            _ linear: AnyLinear, weightT: Tensor?, x: Tensor, outDim: Int, inDim: Int
        ) -> Tensor {
            if let weightT,
                LagunaSteelGemm.eligible(t: nRows, outDim: outDim, inDim: inDim, dtype: x.dtype)
            {
                return LagunaSteelGemm.callMany(
                    input: x, weightT: weightT, t: nRows, outDim: outDim, inDim: inDim,
                    on: work, device: device)
            }
            return linear.callMany(x, t: nRows, on: work, device: device)
        }

        var k = projOrSteel(kProj, weightT: kWeightT, x: xNorm, outDim: kOutDim, inDim: hidden)
        let v = projOrSteel(vProj, weightT: vWeightT, x: xNorm, outDim: vOutDim, inDim: hidden)

        k = Ops.rmsNormRows(
            k.reshaped(to: [nRows * kOutDim]),
            weight: kNorm.weight, eps: kNorm.eps,
            nRows: nRows * nKVHeads, rowSize: headDim, on: work
        ).reshaped(to: [nRows, kOutDim])

        lagunaBatchedRopeKVOnly(
            k: &k, v: v, nRows: nRows, position: position,
            nKVHeads: nKVHeads, headDim: headDim,
            rotaryDim: rotaryDim, ropeTheta: ropeTheta,
            isFullYarn: isFullYarn, yarn: yarn,
            cache: cache, cmd: work, device: device)

        // ── Phase B: last row only, decode-shaped ──
        let xNormLast = xNorm.slicedRows(start: nRows - 1, count: 1).reshaped(to: [hidden])
        let residualLast = residualFull.slicedRows(start: nRows - 1, count: 1).reshaped(to: [hidden])

        var qRaw = qProj(xNormLast, on: work)
        qRaw = Ops.rmsNormRows(
            qRaw, weight: qNorm.weight, eps: qNorm.eps,
            nRows: nHeads, rowSize: headDim, on: work)
        ropeQOnly(&qRaw, position: lastPos, yarn: yarn, on: work)

        let q2d = qRaw.reshaped(to: [nHeads, headDim])
        let (cacheK, cacheV) = cache.prepareForAttention(on: work)
        let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: cache.length)
        var attnOut =
            (LagunaSdpaDecodeGqa2.apply(
                q: q2d, k: cacheK, v: cacheV,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                nKV: cache.length, kvStride: cache.maxSeq, scale: scale,
                on: work, device: device,
                sinkEnd: sinkEnd, windowStart: windowStart)
                ?? Ops.sdpaDecode(
                    q: q2d, k: cacheK, v: cacheV,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    nKV: cache.length, kvStride: cache.maxSeq,
                    scale: scale, on: work,
                    sinkEnd: sinkEnd, windowStart: windowStart)
            ).reshaped(to: [nHeads * headDim])

        // Gate + o_proj: identical dispatch chain to decode `forward()`.
        ensureAttnInt8(device: device)
        let gateLogitsRaw: Tensor? = gProj?(xNormLast, on: work)

        let oOut: Tensor
        if let oBank = attnInt8OProjBank, attnOut.dtype == .bf16 {
            if attnInt8OProjScratch == nil {
                attnInt8OProjScratch = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
            }
            var gatedDone = false
            if let gateLogits = gateLogitsRaw {
                gatedDone = LagunaAttnInt8.qmvGated(
                    input: attnOut, gateLogits: gateLogits, bank: oBank,
                    heads: nHeads, headDim: headDim,
                    on: work, device: device, into: attnInt8OProjScratch!)
            }
            if !gatedDone {
                if let gateLogits = gateLogitsRaw {
                    if fusedGatedScratch == nil
                        || fusedGatedScratch!.elementCount != nHeads * headDim
                    {
                        fusedGatedScratch = Tensor.empty(
                            shape: [nHeads * headDim], dtype: .bf16, device: device)
                    }
                    let fused = LagunaGatedOProjFuse.applyGateProductSoftplus(
                        attnOut: attnOut, gateLogits: gateLogits, heads: nHeads, headDim: headDim,
                        on: work, device: device, into: fusedGatedScratch!)
                    attnOut =
                        fused
                        ? fusedGatedScratch!
                        : applySoftplusHeadGateFromLogits(
                            attnOut, gateLogits: gateLogits, cmd: work, device: device)
                }
                LagunaAttnInt8.qmv(
                    input: attnOut, bank: oBank, on: work, device: device,
                    into: attnInt8OProjScratch!)
            }
            oOut = attnInt8OProjScratch!
        } else {
            var fusedOut: Tensor?
            if let gateLogits = gateLogitsRaw, let lo = oProj.inner as? Linear,
                lo.bias == nil, lo.weight.dtype == .bf16, attnOut.dtype == .bf16
            {
                if fusedGateValuesScratch == nil {
                    fusedGateValuesScratch = Tensor.empty(shape: [nHeads], dtype: .bf16, device: device)
                }
                _ = Ops.softplus(gateLogits, on: work, into: fusedGateValuesScratch!)
                if fusedOProjScratch == nil {
                    fusedOProjScratch = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
                }
                let ok = LagunaGatedOProjFuse.applyGatedOutputProjection(
                    attentionOutput: attnOut, gateValues: fusedGateValuesScratch!,
                    weight: lo.weight, heads: nHeads, headDim: headDim, hidden: hidden,
                    on: work, device: device, into: fusedOProjScratch!)
                if ok { fusedOut = fusedOProjScratch! }
            }
            if let fusedOut {
                oOut = fusedOut
            } else {
                if let gateLogits = gateLogitsRaw {
                    attnOut = applySoftplusHeadGateFromLogits(
                        attnOut, gateLogits: gateLogits, cmd: work, device: device)
                }
                oOut = oProj(attnOut, on: work)
            }
        }

        // Residual + post-attn norm, single row.
        let h2: Tensor
        let ffnIn: Tensor
        var precomputedRouterLogits: Tensor?
        if case .moe(let moe) = ffn,
            LagunaResidualRmsRouterFuse.enabled,
            hidden == 2048, moe.router.nExperts == 256,
            OpsValidation.validateAddRmsNorm(n: hidden) == nil,
            let gateLinear = moe.gate.inner as? Linear,
            gateLinear.bias == nil, gateLinear.weight.dtype == .bf16
        {
            let fused = LagunaResidualRmsRouterFuse.apply(
                residual: residualLast, attnOut: oOut,
                normWeight: postAttnNorm.weight, eps: postAttnNorm.eps,
                routerWeight: gateLinear.weight,
                hidden: hidden, nExperts: moe.router.nExperts,
                on: work, device: device)
            h2 = fused.residual
            ffnIn = fused.normed
            precomputedRouterLogits = fused.logits
        } else if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                residualLast, oOut, weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: 1, rowSize: hidden, on: work)
            h2 = fused.residual
            ffnIn = fused.normed
        } else {
            h2 = Ops.add(residualLast, oOut, on: work)
            ffnIn = postAttnNorm(h2, on: work)
        }

        // Terminal FFN, single row: same decode entry points `forward()`
        // calls per decode step.
        switch ffn {
        case .dense(let mlp):
            let ffnOut = mlp.forward(ffnIn, on: work)
            let out = Ops.add(h2, ffnOut, on: work)
            if work.status != .committed && work.status != .completed
                && work.status != .error
            {
                work.commit()
            }
            work.waitUntilCompleted()
            return out
        case .moe(let moe):
            let ffnOut: Tensor
            if let logits = precomputedRouterLogits {
                ffnOut = moe.decodeWithPrecomputedLogits(
                    ffnIn, position: lastPos,
                    cache: StatelessLayerCache(),
                    cmd: work, device: device,
                    precomputedLogits: logits)
            } else {
                ffnOut = moe.decode(
                    ffnIn, position: lastPos,
                    cache: StatelessLayerCache(),
                    cmd: work, device: device)
            }
            if work.status == .committed || work.status == .completed
                || work.status == .error
            {
                let addCmd = device.makeCommandBuffer()
                let out = Ops.add(h2, ffnOut, on: addCmd)
                addCmd.commit()
                addCmd.waitUntilCompleted()
                return out
            }
            let out = Ops.add(h2, ffnOut, on: work)
            if work.status != .committed && work.status != .completed
                && work.status != .error
            {
                work.commit()
            }
            return out
        }
    }

    /// Q-only counterpart of `applyRopeInPlace`, used by
    /// `forwardMultiLastRow`: K has already been normed + RoPE'd +
    /// cache-appended for the full chunk in Phase A, so Phase B only
    /// needs to rotate the single last-row Q. Mirrors `applyRopeInPlace`'s
    /// Q branch exactly (same kernel calls, same theta/rotaryDim/YaRN
    /// selection), kept separate rather than making `applyRopeInPlace`
    /// take an optional K, to leave the existing (tested) decode path's
    /// source untouched.
    private func ropeQOnly(
        _ q: inout Tensor, position: Int, yarn: Ops.RoPEYaRN, on cmd: MTLCommandBuffer
    ) {
        if isFullYarn {
            lagunaRopeYaRNPartial(
                q, position: position, nHeads: nHeads,
                headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: ropeTheta, yarn: yarn, on: cmd)
            return
        }
        if rotaryDim == headDim {
            _ = Ops.rope(
                q.reshaped(to: [nHeads, headDim]),
                position: position, headDim: headDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd, into: q)
        } else {
            Ops.ropePartial(
                q, position: position,
                headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd)
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
        // W-headgate fuse: one broadcast pass replaces the host-built
        // expand-id gather + separate mul (byte-identical). Escape `=0`.
        if LagunaEnvFlags.headGateFuse {
            let attn = attnOut.reshaped(to: [t, nHeads * headDim])
            let gate = gSoft.reshaped(to: [nGates])
            return Ops.headGateMul(
                attn: attn, gate: gate, headDim: headDim, on: cmd)
        }
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

/// K/V-only counterpart of `lagunaBatchedRopeAndKV`, used by
/// `LagunaLayer.forwardMultiLastRow` (terminal-layer last-row prefill):
/// batched RoPE for K + batched KV-cache append, with no Q handling at
/// all (the caller computes Q separately, last-row only). Mirrors
/// `lagunaBatchedRopeAndKV` exactly minus every Q line, kept as its own
/// function rather than making `q` optional there, so the existing
/// (tested) stock-prefill path's source stays untouched.
private func lagunaBatchedRopeKVOnly(
    k: inout Tensor, v: Tensor,
    nRows: Int, position: Int,
    nKVHeads: Int, headDim: Int,
    rotaryDim: Int, ropeTheta: Float,
    isFullYarn: Bool, yarn: Ops.RoPEYaRN,
    cache: any KVCacheProtocol,
    cmd: MTLCommandBuffer, device: Device
) {
    let kvDim = nKVHeads * headDim
    let positionsT = Tensor.empty(shape: [nRows], dtype: .u32, device: device)
    let positionsPtr = positionsT.buffer.contents()
        .advanced(by: positionsT.offset)
        .bindMemory(to: UInt32.self, capacity: nRows)
    for r in 0 ..< nRows { positionsPtr[r] = UInt32(position + r) }

    if isFullYarn {
        let kRowBytes = kvDim * k.dtype.byteSize
        for r in 0 ..< nRows {
            let kRow = Tensor(
                buffer: k.buffer, offset: k.offset + r * kRowBytes,
                shape: [kvDim], dtype: k.dtype)
            lagunaRopeYaRNPartial(
                kRow, position: position + r, nHeads: nKVHeads,
                headDim: headDim, rotaryDim: rotaryDim,
                thetaBase: ropeTheta, yarn: yarn, on: cmd)
        }
    } else {
        Ops.ropePartialMany(
            k, positions: positionsT,
            t: nRows, nHeads: nKVHeads,
            headDim: headDim, rotaryDim: rotaryDim, rowStride: kvDim,
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
    /// Optional w21 planar int5 coarse copy (BUTTER_LAGUNA_LMHEAD_INT5=1);
    /// swaps the coarse GEMV's byte layout, MXFP8 stays the fallback.
    let lmHeadInt5: LagunaLmHeadPrune.LagunaInt5Coarse?
    let textConfig: LagunaTextConfig
    let yarn: Ops.RoPEYaRN

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let dtype: DType
    public let slidingWindow: Int

    public var defaultPrefillStepSize: Int { 4096 }
    public var requiresLeadingBOS: Bool { false }

    // ─── W35: non-blocking decode CB window ──────────────────────────
    // BUTTER_LAGUNA_NONBLOCK_DECODE (default ON; =0 disables) replaces the per-token blocking
    // depth window in `forwardFromTokenTensor` (removeFirst().
    // waitUntilCompleted() ~10x/token + trailing full drain) with a
    // NON-blocking window: every layer-group CB gets an
    // addCompletedHandler that signals a counting semaphore, and the
    // host only blocks when the total number of in-flight CBs would
    // exceed `nonblockInflightCBs` (BUTTER_LAGUNA_INFLIGHT_CBS,
    // default 24). The window is an *instance* semaphore, so it spans
    // MULTIPLE tokens: `forwardFromTokenTensor` returns as soon as the
    // stack is committed, and GpuTokenLoop (lag > 1) starts encoding
    // token N+1..N+3 while token N still executes — steady-state
    // throughput approaches max(T_enc, T_gpu) instead of the sum.
    //
    // Correctness: all decode CBs in this path come from
    // `device.makeCommandBuffer()` = the PRIMARY queue only (verified:
    // no `makeCommandBufferB`/`commandQueueB` use anywhere in the
    // Laguna decode path), so GPU-side sequencing of per-layer scratch
    // reuse across layers AND tokens follows commit order + Metal
    // hazard tracking; no fences needed. Host-side per-token state
    // (KVEvictionState.reserveNextSlot, `length` bumps) mutates at
    // ENCODE time on the single host encode thread, so its order is
    // unchanged. The queue caps in-flight CBs at 256
    // (IronLibrary.defaultMaxCommandBufferCount); the default window
    // (24) stays far below that so `makeCommandBuffer` never blocks
    // silently. Incompatible with the T39 dkTiming probe (which needs
    // per-kernel commit+wait): the flag is guarded off when
    // `LagunaDecodeKernelTiming.enabled`.
    //
    // NOTE: mid-layer commits by host/hybrid MoE fallbacks bypass the
    // window (uncounted); Laguna's default GPU-router Path B never
    // takes those, and they self-limit (they commit+wait on the host).
    static let nonblockDecodeEnabled: Bool =
        ButterEnv["BUTTER_LAGUNA_NONBLOCK_DECODE"] != "0"
    static let nonblockInflightCBs: Int = max(
        2,
        Int(ButterEnv["BUTTER_LAGUNA_INFLIGHT_CBS"] ?? "") ?? 24)
    /// Counting semaphore backing the cross-token in-flight window.
    /// Lazy is safe: decode host encode is single-threaded.
    private lazy var nonblockSem = DispatchSemaphore(value: Self.nonblockInflightCBs)

    init(
        embedTokens: AnyEmbedding,
        layers: [LagunaLayer],
        finalNorm: RMSNorm,
        lmHead: AnyLinear,
        lmHeadMxfp8: Mxfp8Linear? = nil,
        lmHeadInt5: LagunaLmHeadPrune.LagunaInt5Coarse? = nil,
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
        self.lmHeadInt5 = lmHeadInt5
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
        return forwardFromTokenTensor(
            tokenTensor, position: position,
            caches: caches, on: cmd, device: device)
    }

    /// GPU-resident decode step (F-token-loop): same computation as
    /// `forward(tokenId:...)` above, but skips the per-call host
    /// buffer-build: `tokenIdBuffer` is consumed as-is, typically the
    /// previous decode step's `Ops.argmax` output buffer, so the
    /// embedding gather below chains directly off GPU-resident state.
    /// See `GpuTokenLoop` (Generation/GpuTokenLoop.swift) for the
    /// decode-loop side of this; this method only removes the per-call
    /// host round trip for the *input* token id; the layer stack
    /// still runs its own internal command-buffer pipe and waits on it
    /// exactly as `forward(tokenId:...)` does.
    public var supportsGPUTokenFeedback: Bool { true }

    public func forward(
        tokenIdBuffer: Tensor, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            tokenIdBuffer.dtype == .u32,
            "LagunaModel.forward(tokenIdBuffer:): must be u32")
        precondition(
            tokenIdBuffer.elementCount == 1,
            "LagunaModel.forward(tokenIdBuffer:): must be a single-element tensor")
        return forwardFromTokenTensor(
            tokenIdBuffer, position: position,
            caches: caches, on: cmd, device: device)
    }

    /// Shared body for `forward(tokenId:...)` / `forward(tokenIdBuffer:...)`:
    /// everything after the token id has become a device `Tensor`.
    private func forwardFromTokenTensor(
        _ tokenTensor: Tensor, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // Multi-CB layer pipeline (default). Residual dest buffer replaces
        // the old ffnCopy blit. Single-CB all layers: LAYER_PIPE=0.
        //
        // Per-token host/Metal overhead (40 layers, 39 MoE, pipe depth 4):
        //   ~41 main CBs + ~39 dual-queue B CBs + ~78 events + 39 blits
        // Packing layers/CB cuts main CB count; dual-queue opt-out cuts B/events;
        // residual-into-scratch skips the blit.
        let pipe = LagunaEnvFlags.layerPipe
        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        // Embed-phase checkpoint, closing the last gap in the per-phase
        // decode timing taxonomy (embed/attn/moe/lm_head_sample/host_gap).
        // Reuses T39's decode-kernel-timing checkpoint (same forced
        // commit+wait methodology; `LagunaDecodeKernelTiming.enabled`
        // already forces `nonblock = false` below, so this stays
        // consistent with the rest of the decode step being measured
        // synchronously). No-op when BUTTER_LAGUNA_DECODE_KERNEL_TIMING is
        // unset.
        if LagunaDecodeKernelTiming.enabled {
            workCmd = LagunaDecodeKernelTiming.checkpoint("embed", cmd: workCmd, device: device)
        }

        // BUTTER_DEBUG_ROUTER_SETS (butter#108): debug-only fully
        // synchronous per-layer branch, taken instead of the pipe/nonblock
        // machinery below. `forwardFromTokenTensor` is the shared body for
        // BOTH `forward(tokenId:)` (host) and `forward(tokenIdBuffer:)`
        // (GPU-chained) — the whole point of the flag is comparing the
        // per-layer MoE router's top-K expert selection between those two
        // arms at the same prompt/position, so both must go through this
        // same instrumented path. One `RouterSetsDebug.nextToken()` per
        // call (i.e. once per generated token, not per layer) then a
        // commit+wait after EVERY layer: Laguna's decode scratches
        // (`routerIndicesScratch` et al.) are grow-only and reused in
        // place, so without forcing full completion here a later
        // layer/token's write could race the readback in
        // `MoELayer.debugRouterSnapshot()` under the default nonblock
        // decode window (see `LagunaModel.nonblockDecodeEnabled`'s header
        // for why that window is otherwise safe: it relies on single-queue
        // ordering, which this branch just collapses to synchronous
        // instead of relying on).
        if RouterSetsDebug.enabled {
            RouterSetsDebug.nextToken()
            if workCmd.status != .committed && workCmd.status != .completed
                && workCmd.status != .error
            {
                workCmd.commit()
            }
            workCmd.waitUntilCompleted()
            for (i, layer) in layers.enumerated() {
                let stepCmd = device.makeCommandBuffer()
                h = layer.forward(
                    h, position: position,
                    cache: caches[i] as! any KVCacheProtocol,
                    yarn: yarn,
                    cmd: stepCmd, device: device)
                if stepCmd.status != .committed && stepCmd.status != .completed
                    && stepCmd.status != .error
                {
                    stepCmd.commit()
                }
                stepCmd.waitUntilCompleted()
                if case .moe(let moe) = layer.ffn,
                    let snap = moe.debugRouterSnapshot()
                {
                    RouterSetsDebug.report(
                        layer: moe.layerIndex, indices: snap.indices, scores: snap.scores)
                }
            }
            return lagunaFinalNormLmHead(
                h: h, finalNorm: finalNorm, lmHead: lmHead,
                lmHeadMxfp8: lmHeadMxfp8, lmHeadInt5: lmHeadInt5,
                on: cmd, device: device)
        }

        if pipe {
            // Depth: how many CBs in flight (default 4). T28 lever 4 swept
            // 1..8 (fib prompt, --stats, 5s cooldowns): flat ~100.1-100.5
            // tok/s across the whole range: depth is already past its
            // saturation point at every value tested here, so the default
            // (4) is kept as-is.
            let depth = LagunaEnvFlags.pipeDepth
            // Layers packed into one CB before commit. T28 lever 4 swept
            // 1..20 (fib prompt, --stats, 3 runs/side, 5s cooldowns) and
            // found the old default (2) was already past its own sweet
            // spot: 1 measured +1.9% over 2 (102.1 vs 100.1 tok/s avg) and
            // every larger value regressed monotonically (3→98.3, 4→96.5,
            // ... 20→75.3 tok/s): packing MORE layers per CB trades away
            // overlap for bigger, less-parallel command buffers on this
            // per-layer-CB architecture. Flipped default to 1 (finest
            // granularity = fastest here); escape via the env var if a
            // future change shifts this balance again.
            // W35 non-blocking window (see the property block above the
            // initializer). dkTiming needs per-kernel commit+wait, so the
            // flag is forced off under it. Under nonblock the layers/CB
            // default rises 1 → 8: with layersPerCB=1 a token is ~41 CBs,
            // which would eat the whole 24-CB window in half a token; at
            // 8 a token is ~6 CBs so 3-4 tokens fit in flight.
            let nonblock = Self.nonblockDecodeEnabled && !LagunaDecodeKernelTiming.enabled
            let layersPerCB = max(
                1,
                LagunaEnvFlags.layersPerCBEnv
                    ?? (nonblock ? 8 : 1))
            // Non-blocking commit: block only when the cross-token
            // in-flight CB count would exceed the window. Handler is
            // registered BEFORE commit (commit-then-add can miss the
            // callback if the buffer finishes first).
            let sem = nonblock ? nonblockSem : nil
            func commitWindowed(_ cb: MTLCommandBuffer) {
                if let sem {
                    sem.wait()
                    cb.addCompletedHandler { _ in sem.signal() }
                }
                cb.commit()
            }
            var inflight: [MTLCommandBuffer] = []
            if workCmd.status != .committed && workCmd.status != .completed
                && workCmd.status != .error
            {
                commitWindowed(workCmd)
            }
            if !nonblock { inflight.append(workCmd) }
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
                    commitWindowed(workCmd)
                }
                if !nonblock {
                    inflight.append(workCmd)
                    while inflight.count > depth {
                        inflight.removeFirst().waitUntilCompleted()
                    }
                }
            }
            // W35: no trailing drain under nonblock — the caller's `cmd`
            // (final-norm + lm_head, committed by GpuTokenLoop with the
            // argmax) is on the same queue and therefore already
            // sequences after every layer CB committed above.
            if !nonblock {
                for cb in inflight { cb.waitUntilCompleted() }
            }
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
            lmHeadMxfp8: lmHeadMxfp8, lmHeadInt5: lmHeadInt5,
            on: cmd, device: device)
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
        if LagunaEnvFlags.serialPrefill {
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
        // Embed-phase checkpoint, closing the last gap in the per-phase
        // prefill timing taxonomy (embed/attn/moe/lm_head_sample/host_gap).
        // This commit+wait was already unconditional here (embed must land
        // before the first layer reads `h`), so timing it costs nothing
        // extra -- just a clock read on each side, gated by the same
        // BUTTER_LAGUNA_ATTN_PHASE_TIMING flag the rest of this function's
        // prefill phase timing (below) already uses.
        let embedPhaseTiming = LagunaEnvFlags.attnPhaseTiming
        let embedT0 = embedPhaseTiming ? CFAbsoluteTimeGetCurrent() : 0
        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd)
            .reshaped(to: [n, hidden])
        if workCmd.status != .committed && workCmd.status != .completed
            && workCmd.status != .error
        {
            workCmd.commit()
        }
        workCmd.waitUntilCompleted()
        if embedPhaseTiming {
            let ms = (CFAbsoluteTimeGetCurrent() - embedT0) * 1000
            fputs("[laguna-prefill-embed] total=\(String(format: "%.2f", ms))ms\n", stderr)
        }

        // Depth-2 pipe for prefill layers (same as decode) when MoE
        // decodeMany commits without mid-layer host waits.
        let layerScan =
            LagunaEnvFlags.layerScan
        let layerTiming =
            LagunaEnvFlags.timing
        // When scanning/timing, force pipe depth 1 so each layer is fully
        // complete before the host reads h (or times the CB).
        let prefDepth = (layerScan || layerTiming)
            ? 1
            : max(1, LagunaEnvFlags.prefillPipeEnv ?? 4)
        var inflight: [MTLCommandBuffer] = []
        var timingMs: [(Int, String, Double)] = []
        // F-lastrow: when set, the FINAL layer's already-computed last
        // row (`[hidden]` flat); see `LagunaLastRowPrefill`. Populated
        // only on the last loop iteration, only when the lastrow path
        // was actually taken (nRows>1 and the flag is on); `nil`
        // otherwise, in which case the post-loop code falls back to
        // slicing the last row out of `h` exactly like stock.
        var lastRowOut: Tensor?
        for (i, layer) in layers.enumerated() {
            workCmd = device.makeCommandBuffer()
            let t0 = layerTiming ? CFAbsoluteTimeGetCurrent() : 0
            let isFinalLayer = i == layers.count - 1
            if isFinalLayer, n > 1, LagunaLastRowPrefill.enabled {
                lastRowOut = layer.forwardMultiLastRow(
                    h, startingAt: position,
                    cache: caches[i] as! any KVCacheProtocol,
                    yarn: yarn,
                    cmd: workCmd, device: device)
            } else {
                h = layer.forwardMulti(
                    h, startingAt: position,
                    cache: caches[i] as! any KVCacheProtocol,
                    yarn: yarn,
                    cmd: workCmd, device: device)
            }
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
                // Lastrow path only ever produces `[hidden]` (one row);
                // the nan/inf/maxAbs scan below is still meaningful over
                // that narrower slice, just fewer values than the
                // full-width `h` other layers scan.
                let vals = (lastRowOut ?? h).toFloatArray()
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
        // F-lastrow: when the terminal layer took the last-row path,
        // `lastRowOut` already IS that row (flat `[hidden]`); no slice
        // needed. Otherwise fall back to the stock slice.
        let last = lastRowOut ?? h.slicedRows(start: n - 1, count: 1).reshaped(to: [hidden])
        return lagunaFinalNormLmHead(
            h: last, finalNorm: finalNorm, lmHead: lmHead,
            lmHeadMxfp8: lmHeadMxfp8, lmHeadInt5: lmHeadInt5,
            on: cmd, device: device)
    }
}

/// Fused finalNorm + int4 lm_head when shapes match `rmsNormQgemvInt4Fast`
/// (hidden % 512 == 0, vocab % 8 == 0, groupSize 64). Laguna XS hits this.
/// Certified MXFP8 2-pass prune for BF16 lm_head decode (DEFAULT ON when
/// sidecar present; `BUTTER_LAGUNA_LM_HEAD_PRUNE=0` disables).
private func lagunaFinalNormLmHead(
    h: Tensor, finalNorm: RMSNorm, lmHead: AnyLinear,
    lmHeadMxfp8: Mxfp8Linear? = nil,
    lmHeadInt5: LagunaLmHeadPrune.LagunaInt5Coarse? = nil,
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
    // Decode prune: only when env allows, MXFP8 sidecar loaded, stock
    // BF16 lm_head shape matches the certified kernels (Laguna XS), AND
    // the current token's sampling mode (T40 follow-up, `Generate.swift`'s
    // `sampleNext`) hasn't opted this token out. `currentSamplingMode`
    // defaults to `.pruneCertified`, so every call site that predates
    // this follow-up (tests, `butter bench`, greedy/gpu-categorical decode)
    // is unaffected -- this only changes behavior for CPU-sampled tokens
    // (temperature>0 or repetitionPenalty!=1.0), where the coarse
    // approximation on non-candidate rows is the wrong distribution to
    // sample from (see `LagunaLmHeadPrune.SamplingMode` doc). Non-Laguna
    // callers of this file never touch `currentSamplingMode`, so it's
    // always `.pruneCertified` for them and this is a pure no-op there.
    if LagunaLmHeadPrune.enabled,
        LagunaLmHeadPrune.currentSamplingMode != .exactFullVocab,
        let mx = lmHeadMxfp8,
        let lin = lmHead.inner as? Linear,
        lin.weight.dtype == .bf16,
        lin.weight.shape == [100_352, 2048]
    {
        // `.pruneCertified` and `.pruneCandidateRestricted` both run the
        // identical cheap two-pass kernels -- the candidate-restricted
        // opt-in only changes how `Generate.swift` CONSUMES the result
        // (masking non-candidate rows to -infinity before CPU sampling,
        // using the certificate `logits(...)` already stashes in
        // `LagunaLmHeadPrune.lastCandidateCertificate`), not how the GPU
        // computes it.
        return LagunaLmHeadPrune.logits(
            normed: normed, mx: mx, bf16Weight: lin.weight,
            int5: lmHeadInt5, on: cmd, device: device)
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
    // Sidecar lm_head bytes are loaded straight from disk (not through
    // the GGUF mmap-view path), so they never hit the residency set
    // that path wires automatically. It's read every decode-prune pass
    // (or every forward, in full-replace mode), so pin it explicitly.
    device.markWeightsResident([codesBuf, scalesBuf])
    return Mxfp8Linear(weight: codes, scales: scales, blockSize: 32)
}
