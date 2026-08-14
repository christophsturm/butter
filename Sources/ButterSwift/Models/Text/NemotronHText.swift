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
// NemotronH text — concrete variants + the stack-interleaved hybrid
// decoder. The family enum (`enum NemotronH`), variant protocol
// (`NemotronHVariant`), and unified error type (`NemotronHError`) live
// in `Models/Nemotron.swift` (the unified Nemotron family root, which
// also unions metadata across NemotronH / NemotronVL /
// NemotronDiffusion / NemotronDiffusionVL). This file holds the
// text-only impl:
//
//   • `NemotronHHybrid` — the single variant (dense / MoE FFN mix is
//     decided per checkpoint from `hybrid_override_pattern` +
//     `num_experts`),
//   • `NemotronHModel` — the full LanguageModel decoder.
//
// ─── Architecture ────────────────────────────────────────────────────
//
// NemotronH is a *stack-interleaved hybrid* model.
//
// Unlike FalconH1 (a *parallel* hybrid where every layer runs Mamba
// AND attention together), NemotronH is a **stack-interleaved** hybrid:
// a `hybrid_override_pattern` string assigns each decoder layer exactly
// ONE mixer kind, and the kinds genuinely vary down the stack:
//
//   "M"  → Mamba 2 selective-SSM mixer       (NemotronHMambaLayer)
//   "*"  → multi-head attention              (NemotronHAttentionLayer)
//   "-"  → dense squared-ReLU MLP            (NemotronHMLPLayer)
//   "E"  → mixture-of-experts feed-forward   (rejected — see below)
//
// e.g. NemotronH-4B-Base ships
//   "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
// — 52 layers, 4 of them attention, the rest split between Mamba and
// dense MLP. The layer array is therefore *heterogeneous*: it is held
// as `[any DecoderLayer]` and the decode loop walks it in lockstep
// with a per-index `[any LayerCacheProtocol]`.
//
// Per-layer dataflow — every layer kind shares the same pre-norm +
// residual skeleton (matches mlx-lm's `nemotron_h.py`):
//
//   residual = h
//   h        = norm(h)               [hidden]  — RMSNorm, one per layer
//   h        = mixer(h)              [hidden]  — M / * / - / E
//   out      = residual + h
//
// There is NO separate pre-FF norm and NO second residual: a NemotronH
// layer is "one norm, one mixer, one residual add", and the MLP / MoE
// layers are themselves a layer kind in the stack rather than a
// trailing block inside every layer.
//
// ─── Attention has no RoPE ───────────────────────────────────────────
//
// NemotronH attention uses *no* positional embedding — no RoPE, no ALiBi.
// The Mamba layers carry the sequence-order information; the sparse
// attention layers attend without rotation. `NemotronHAttentionLayer`
// therefore skips the `Ops.rope` call that every other attention family
// in Butter makes.
//
// ─── Mamba 2 with n_groups > 1 + gated mixer RMSNorm ─────────────────
//
// Every published NemotronH checkpoint ships `n_groups = 8` and a gated
// mixer RMSNorm (`mixer.norm`). The shared `Mamba2Layer` does not
// support either, so NemotronH carries its own `NemotronHMambaLayer`:
//
//   * **Grouped B/C** — the SSM `B`/`C` tensors are shared across
//     `heads_per_group = n_heads / n_groups` heads. We reuse the
//     shipped scalar `Ops.ssmStep` kernel by dispatching it once per
//     group over a contiguous head sub-slab — no kernel change needed.
//   * **Gated RMSNorm** — `y` is gated by `silu(z)` BEFORE a per-group
//     (unweighted) RMSNorm, then scaled by the learned `mixer.norm`
//     weight. Implemented with `Ops.rmsNormRows` (one row per group) +
//     `Ops.mul`.
//
// ─── MoE layers ("E") — Cascade-2 / Nemotron-3 lineage ───────────────
//
// The "E" layer kind ships with the Nemotron-Cascade-2-30B-A3B,
// Nemotron-3 Nano/Super/Ultra (all 30B-A3B+) checkpoints. The MoE
// block diverges enough from `Models/MoELayer.swift` (squared-ReLU
// experts with NO gate projection, sigmoid+bias router, raw-sigmoid
// post-K renormalisation + routed_scaling, always-on dense shared
// expert) that it lives next to the rest of the variants in this file
// as `NemotronHMoELayer` (below). Grouped expert routing (`n_group >
// 1`) is rejected at load — every shipped checkpoint sets `n_group:
// 1`, which collapses to flat top-K.

import Foundation
import Metal

// ─── Layer kind ──────────────────────────────────────────────────────

/// The four mixer kinds a `hybrid_override_pattern` character can name.
enum NemotronHLayerKind: Equatable {
    case mamba  // "M"
    case attention  // "*"
    case mlp  // "-"
    case moe  // "E"

    init(from char: Character) throws {
        switch char {
        case "M": self = .mamba
        case "*": self = .attention
        case "-": self = .mlp
        case "E": self = .moe
        default:
            throw NemotronHError.unsupportedConfig(
                "unknown hybrid_override_pattern character '\(char)'")
        }
    }

    /// Newer NemotronH checkpoints (e.g. Nemotron-3.5-Lightning-30B-A3B)
    /// ship the layer schedule as a JSON array of names —
    /// `layers_block_type: ["mamba", "moe", "attention", ...]` — rather
    /// than the packed `hybrid_override_pattern` character string.
    /// Same schedule, different config schema; mirrors Granite4's
    /// `layer_types` string-array parsing (`Granite4Text.swift`).
    ///
    /// Deliberately NOT `init(from: String)`: overloading `init(from:)`
    /// on `Character` vs `String` makes every existing bare-string-
    /// literal call site (`NemotronHLayerKind(from: "M")`, used
    /// throughout `NemotronHTextTests.swift`) resolve to THIS
    /// initializer instead of the `Character` one — Swift's default
    /// literal inference picks `String` over `Character` when both
    /// overloads are viable, silently breaking every existing
    /// single-character call site. A distinctly-labeled initializer
    /// sidesteps the ambiguity entirely.
    init(fromBlockTypeName name: String) throws {
        switch name {
        case "mamba": self = .mamba
        case "attention": self = .attention
        case "mlp": self = .mlp
        case "moe": self = .moe
        default:
            throw NemotronHError.unsupportedConfig(
                "unknown layers_block_type entry '\(name)'")
        }
    }
}

// ─── NemotronHHybrid — the single variant ────────────────────────────

public struct NemotronHHybrid: NemotronHVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// NemotronH ships both `-Base` and `-Instruct` checkpoints. Greedy
    /// by default keeps the integration suite deterministic.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 256,
        temperature: 0.0,
        topP: 1.0,
        topK: 0,
        minP: 0.0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options _: LoadOptions,
        device: Device
    ) throws -> NemotronHModel {
        guard let hidden = config.hiddenSize,
            let vocab = config.vocabSize,
            let nHeads = config.numAttentionHeads
        else { throw NemotronHError.missingConfig("hidden / vocab / num_attention_heads") }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        // NemotronH attention can carry an explicit head_dim
        // (attention_head_dim on the 4B config, head_dim elsewhere) that
        // is NOT hidden / n_heads.
        let headDim =
            config.int("attention_head_dim")
            ?? config.int("head_dim")
            ?? (hidden / nHeads)
        let intermediate = config.intermediateSize ?? (4 * hidden)
        let eps = Float(
            config.float("layer_norm_epsilon")
                ?? config.float("rms_norm_eps") ?? 1e-5)
        let tieEmbed = config.tieWordEmbeddings

        // ── Hybrid layer schedule ─────────────────────────────────────
        // Two config schemas carry the identical semantic schedule:
        // the older packed `hybrid_override_pattern` char-string
        // (NemotronH-4B-Base, Cascade-2), and the newer
        // `layers_block_type` JSON string-array (Nemotron-3.5-
        // Lightning). Try the array form first since it's unambiguous;
        // fall back to the char-string.
        let kinds: [NemotronHLayerKind]
        if let blockTypes = config.raw["layers_block_type"] as? [String], !blockTypes.isEmpty {
            kinds = try blockTypes.map { try NemotronHLayerKind(fromBlockTypeName: $0) }
        } else if let pattern = config.string("hybrid_override_pattern"), !pattern.isEmpty {
            kinds = try Array(pattern).map { try NemotronHLayerKind(from: $0) }
        } else {
            throw NemotronHError.missingConfig("hybrid_override_pattern / layers_block_type")
        }
        let nLayers = kinds.count
        // MoE ("E") layer support — Cascade-2 / Nemotron-3 Nano/Super/
        // Ultra. Built by `buildNemotronHMoELayer` below. Grouped
        // routing (`n_group > 1` / `topk_group > 1`) is NOT supported
        // yet because every shipped checkpoint sets both to 1, which
        // collapses to flat top-K — the loader rejects the grouped
        // case explicitly so a developer hitting it gets a clear error
        // rather than a silent wrong answer.
        let hasMoE = kinds.contains(.moe)
        let nGroupMoE = config.int("n_group") ?? 1
        let topKGroupMoE = config.int("topk_group") ?? 1
        if hasMoE {
            guard nGroupMoE == 1, topKGroupMoE == 1 else {
                throw NemotronHError.unsupportedConfig(
                    "MoE grouped routing (n_group=\(nGroupMoE), "
                        + "topk_group=\(topKGroupMoE)) not yet supported — "
                        + "every shipped Nemotron MoE checkpoint sets both to 1.")
            }
        }

        // ── Mamba 2 mixer geometry ────────────────────────────────────
        guard let mambaNHeads = config.int("mamba_num_heads")
        else { throw NemotronHError.missingConfig("mamba_num_heads") }
        guard let mambaHeadDim = config.int("mamba_head_dim")
        else { throw NemotronHError.missingConfig("mamba_head_dim") }
        guard let stateDim = config.int("ssm_state_size")
        else { throw NemotronHError.missingConfig("ssm_state_size") }
        let convKernel = config.int("conv_kernel") ?? 4
        let nGroups = config.int("n_groups") ?? 1
        let useConvBias = config.bool("use_conv_bias") ?? true

        // d_inner is taken directly from the Mamba head decomposition
        // (NemotronH does not use the `expand * hidden` relation).
        let dInner = mambaNHeads * mambaHeadDim
        guard mambaNHeads % nGroups == 0 else {
            throw NemotronHError.unsupportedConfig(
                "mamba_num_heads (\(mambaNHeads)) must be a multiple of "
                    + "n_groups (\(nGroups))")
        }
        // The gated mixer RMSNorm runs per group via Ops.rmsNormRows,
        // whose kernel requires the row size (group size) to be a
        // multiple of 128 and ≤ 4096. dInner / nGroups is that row size.
        let groupSize = dInner / nGroups
        guard groupSize % 128 == 0, groupSize <= 4096 else {
            throw NemotronHError.unsupportedConfig(
                "per-group mixer RMSNorm row size d_inner/n_groups = "
                    + "\(groupSize) must be a multiple of 128 and ≤ 4096 "
                    + "(rmsNormRows kernel invariant)")
        }
        let convDim = dInner + 2 * nGroups * stateDim

        // time_step_limit clamps softplus(dt). NemotronH-4B ships
        // [0.0, Infinity] (no clamp); honour it if a checkpoint sets one.
        let tsLimit = config.raw["time_step_limit"] as? [Double]
        let tsMin = Float(tsLimit?.first ?? 0.0)
        let tsMax: Float = {
            guard let hi = tsLimit?.dropFirst().first else { return .infinity }
            return hi.isFinite ? Float(hi) : .infinity
        }()

        // ── Activation dtype — taken from the embedding table ─────────
        // The quantization guard must run BEFORE the dtype precondition:
        // a quantized checkpoint's packed-u32 embedding used to trip the
        // precondition first, killing the process with a messageless
        // SIGTRAP in release builds instead of this thrown error.
        guard config.quantization == nil else {
            throw NemotronHError.unsupportedConfig(
                "quantized NemotronH checkpoints not yet supported — load a raw bf16/f16 variant")
        }
        let embedWRaw = try weights.tensor(named: "backbone.embeddings.weight")
        let activationDtype = embedWRaw.dtype
        precondition(
            activationDtype == .f32 || activationDtype == .bf16 || activationDtype == .f16,
            "NemotronH: unexpected activation dtype \(activationDtype)")
        let embedTokens = AnyEmbedding(Embedding(weight: embedWRaw))

        // ── MoE hyper-parameters (only read when an "E" layer exists) ─
        // Reading these unconditionally would force Mamba/MLP-only
        // checkpoints (e.g. Nemotron-H-4B-Base) to carry MoE fields they
        // don't have. Read once, here, so each MoE layer construction is
        // a pure pass-through.
        let moeIntermediate =
            hasMoE
            ? (config.int("moe_intermediate_size")
                ?? config.int("intermediate_size") ?? intermediate)
            : 0
        let moeTopK = hasMoE ? (config.int("num_experts_per_tok") ?? 6) : 0
        let nRoutedExperts = hasMoE ? (config.int("n_routed_experts") ?? 0) : 0
        if hasMoE {
            guard nRoutedExperts > 0 else {
                throw NemotronHError.missingConfig("n_routed_experts")
            }
        }
        let nSharedExperts = hasMoE ? (config.int("n_shared_experts") ?? 1) : 0
        let sharedExpertIntermediate =
            hasMoE
            ? (config.int("moe_shared_expert_intermediate_size") ?? moeIntermediate)
            : 0
        let routedScalingFactor: Float =
            hasMoE
            ? Float(config.float("routed_scaling_factor") ?? 1.0)
            : 1.0
        let normTopkProb = hasMoE ? (config.bool("norm_topk_prob") ?? true) : true

        // ── Per-layer construction ────────────────────────────────────
        var layers: [any DecoderLayer] = []
        layers.reserveCapacity(nLayers)
        for (i, kind) in kinds.enumerated() {
            let p = "backbone.layers.\(i)"
            // Every layer kind shares one pre-mixer RMSNorm.
            let norm = RMSNorm(
                weight: try weights.tensor(named: "\(p).norm.weight"), eps: eps)

            switch kind {
            case .mamba:
                layers.append(
                    try buildMambaLayer(
                        prefix: "\(p).mixer", norm: norm, weights: weights,
                        hidden: hidden, dInner: dInner, convDim: convDim,
                        nHeads: mambaNHeads, headDim: mambaHeadDim, stateDim: stateDim,
                        nGroups: nGroups, convKernel: convKernel,
                        useConvBias: useConvBias, eps: eps,
                        tsMin: tsMin, tsMax: tsMax,
                        dtype: activationDtype, device: device))

            case .attention:
                let qProj = AnyLinear(
                    Linear(
                        weight: try weights.tensor(named: "\(p).mixer.q_proj.weight")))
                let kProj = AnyLinear(
                    Linear(
                        weight: try weights.tensor(named: "\(p).mixer.k_proj.weight")))
                let vProj = AnyLinear(
                    Linear(
                        weight: try weights.tensor(named: "\(p).mixer.v_proj.weight")))
                let oProj = AnyLinear(
                    Linear(
                        weight: try weights.tensor(named: "\(p).mixer.o_proj.weight")))
                layers.append(
                    NemotronHAttentionLayer(
                        norm: norm,
                        qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                        nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim))

            case .mlp:
                let upProj = AnyLinear(
                    Linear(
                        weight: try weights.tensor(named: "\(p).mixer.up_proj.weight")))
                let downProj = AnyLinear(
                    Linear(
                        weight: try weights.tensor(named: "\(p).mixer.down_proj.weight")))
                layers.append(
                    NemotronHMLPLayer(
                        norm: norm, upProj: upProj, downProj: downProj,
                        hidden: hidden, intermediate: intermediate))

            case .moe:
                layers.append(
                    try buildNemotronHMoELayer(
                        prefix: "\(p).mixer", norm: norm, weights: weights,
                        hidden: hidden, moeIntermediate: moeIntermediate,
                        nRoutedExperts: nRoutedExperts, topK: moeTopK,
                        nSharedExperts: nSharedExperts,
                        sharedExpertIntermediate: sharedExpertIntermediate,
                        routedScalingFactor: routedScalingFactor,
                        normTopkProb: normTopkProb,
                        dtype: activationDtype, device: device))
            }
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "backbone.norm_f.weight"), eps: eps)

        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight_scale_2") {
            // NVFP4-quantized lm_head (Nemotron-3.5-Lightning) — huge
            // (vocab × hidden), quantized same as the MoE experts.
            lmHead = AnyLinear(
                try loadNemotronHNvfp4Linear(
                    base: "lm_head", outDim: vocab, inDim: hidden,
                    weights: weights, device: device))
        } else if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = AnyLinear(Linear(weight: try weights.tensor(named: "lm_head.weight")))
        } else {
            lmHead = AnyLinear(Linear(weight: embedWRaw))
        }

        let maxSeq = config.int("max_position_embeddings") ?? 8192
        let model = NemotronHModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            mambaNHeads: mambaNHeads, mambaHeadDim: mambaHeadDim,
            stateDim: stateDim, convDim: convDim, convKernel: convKernel,
            nGroups: nGroups, dInner: dInner,
            vocab: vocab, maxSeq: maxSeq, dtype: activationDtype)

        // F-131 MTP self-speculative decode — opt-in (BUTTER_NEMOTRON_MTP=1)
        // load of the checkpoint's optional 53rd "nextn" block. Gated on
        // BOTH the env var and the tensor actually being present, so a
        // checkpoint without an MTP block (every non-Lightning NemotronH
        // release) is completely unaffected regardless of the env var.
        // See `NemotronHMTPHead.swift` for the architecture + loader.
        if ButterEnv["BUTTER_NEMOTRON_MTP"] == "1", weights.has("mtp.layers.0.eh_proj.weight") {
            model.mtpHead = try NemotronHMTPHead.load(
                weights: weights, embedTokens: embedTokens, lmHead: lmHead,
                hidden: hidden, vocab: vocab,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                moeIntermediate: moeIntermediate, nRoutedExperts: nRoutedExperts,
                moeTopK: moeTopK, nSharedExperts: nSharedExperts,
                sharedExpertIntermediate: sharedExpertIntermediate,
                routedScalingFactor: routedScalingFactor, normTopkProb: normTopkProb,
                eps: eps, dtype: activationDtype, device: device)
        }
        return model
    }

    /// Build one Mamba 2 mixer layer (`"M"`). Reads + derives the
    /// per-head SSM parameters and transposes the conv1d weight, the
    /// same load-time arithmetic `Mamba2`/`FalconH1` do.
    private static func buildMambaLayer(
        prefix p: String, norm: RMSNorm, weights: SafeTensorsBundle,
        hidden: Int, dInner: Int, convDim: Int,
        nHeads: Int, headDim: Int, stateDim: Int, nGroups: Int,
        convKernel: Int, useConvBias: Bool, eps: Float,
        tsMin: Float, tsMax: Float,
        dtype: DType, device: Device
    ) throws -> NemotronHMambaLayer {
        // Nemotron-3.5-Lightning ships mamba mixer in_proj/out_proj as
        // static FP8 (W8A8, scalar weight_scale + input_scale) — see
        // `NemotronHNvfp4.swift` header. Butter has no FP8 GEMM/GEMV
        // kernel, so `loadMambaProjNH` dequantizes to the activation
        // dtype at load time and then (F-128 bandwidth wave, default
        // on) requantizes down to affine int4 — see that function's
        // doc for the oracle-precision justification and the
        // `BUTTER_NEMOTRON_MAMBA_PROJ_Q4=0` escape hatch. Non-quantized
        // checkpoints (NemotronH-4B-Base, Cascade-2) hit the
        // plain-tensor / plain-bf16-Linear branch unchanged.
        let inProjOutDim = dInner + convDim + nHeads
        let inProj = try loadMambaProjNH(
            base: "\(p).in_proj", outDim: inProjOutDim, inDim: hidden,
            weights: weights, dtype: dtype, device: device)
        let outProj = try loadMambaProjNH(
            base: "\(p).out_proj", outDim: hidden, inDim: dInner,
            weights: weights, dtype: dtype, device: device)

        // conv1d.weight ships [conv_dim, 1, kernel]; the iron kernel
        // wants [kernel, conv_dim].
        let convWSrc = try weights.tensor(named: "\(p).conv1d.weight")
        precondition(
            convWSrc.elementCount == convDim * convKernel,
            "NemotronH: conv1d.weight count mismatch: \(convWSrc.shape)")
        let convW = transposeConv1dWeightNH(
            src: convWSrc, kernel: convKernel, channels: convDim,
            dtype: dtype, device: device)
        let convB: Tensor = {
            if useConvBias, weights.has("\(p).conv1d.bias") {
                return castVectorNH(
                    (try? weights.tensor(named: "\(p).conv1d.bias"))
                        ?? zeroVectorNH(convDim, dtype: dtype, device: device),
                    count: convDim, dtype: dtype, device: device)
            }
            return zeroVectorNH(convDim, dtype: dtype, device: device)
        }()

        // A_eff = -exp(A_log); dt_bias per head; D tiled across head_dim.
        let aEff = computeAEffNH(
            aLog: try weights.tensor(named: "\(p).A_log"),
            nHeads: nHeads, dtype: dtype, device: device)
        let dtBias = castVectorNH(
            try weights.tensor(named: "\(p).dt_bias"),
            count: nHeads, dtype: dtype, device: device)
        let dTiled = tileDNH(
            d: try weights.tensor(named: "\(p).D"),
            nHeads: nHeads, headDim: headDim, dtype: dtype, device: device)

        // Gated mixer RMSNorm weight — full [d_inner].
        let mixerNorm = RMSNorm(
            weight: try weights.tensor(named: "\(p).norm.weight"), eps: eps)

        return NemotronHMambaLayer(
            norm: norm, inProj: inProj, outProj: outProj,
            convW: convW, convB: convB,
            aEff: aEff, dtBias: dtBias, dTiled: dTiled,
            mixerNorm: mixerNorm,
            hidden: hidden, dInner: dInner, convDim: convDim,
            nHeads: nHeads, headDim: headDim, stateDim: stateDim,
            nGroups: nGroups, convKernel: convKernel,
            tsMin: tsMin, tsMax: tsMax, dtype: dtype)
    }
}

// ─── NemotronHMambaLayer — "M" ───────────────────────────────────────
//
// A Mamba 2 selective-SSM mixer with `n_groups > 1` grouped B/C and a
// gated mixer RMSNorm. Conforms to `DecoderLayer`; its cache slot is a
// `Mamba2LayerCache` (SSM state + conv state).

public final class NemotronHMambaLayer: Module, DecoderLayer {
    let norm: RMSNorm
    let inProj, outProj: AnyLinear
    let convW: Tensor  // [kernel, conv_dim]
    let convB: Tensor  // [conv_dim]
    let aEff: Tensor  // [n_heads]   = -exp(A_log)
    let dtBias: Tensor  // [n_heads]
    let dTiled: Tensor  // [d_inner]   D[h] tiled across head_dim
    let mixerNorm: RMSNorm  // gated mixer RMSNorm weight [d_inner]
    let hidden, dInner, convDim, nHeads, headDim, stateDim, nGroups, convKernel: Int
    let tsMin, tsMax: Float
    let dtype: DType
    /// Heads sharing one B/C group.
    let headsPerGroup: Int
    /// Per-group gated-RMSNorm row size = d_inner / n_groups.
    let groupSize: Int
    /// `[groupSize]` ones vector — the "no learned scale" weight for the
    /// per-group unweighted RMSNorm pass (the learned `mixer.norm`
    /// weight is applied afterwards via `Ops.mul`). Built once at load.
    let onesWeight: Tensor

    init(
        norm: RMSNorm, inProj: AnyLinear, outProj: AnyLinear,
        convW: Tensor, convB: Tensor,
        aEff: Tensor, dtBias: Tensor, dTiled: Tensor,
        mixerNorm: RMSNorm,
        hidden: Int, dInner: Int, convDim: Int,
        nHeads: Int, headDim: Int, stateDim: Int, nGroups: Int,
        convKernel: Int, tsMin: Float, tsMax: Float, dtype: DType
    ) {
        self.norm = norm
        self.inProj = inProj
        self.outProj = outProj
        self.convW = convW
        self.convB = convB
        self.aEff = aEff
        self.dtBias = dtBias
        self.dTiled = dTiled
        self.mixerNorm = mixerNorm
        self.hidden = hidden
        self.dInner = dInner
        self.convDim = convDim
        self.nHeads = nHeads
        self.headDim = headDim
        self.stateDim = stateDim
        self.nGroups = nGroups
        self.convKernel = convKernel
        self.tsMin = tsMin
        self.tsMax = tsMax
        self.dtype = dtype
        self.headsPerGroup = nHeads / nGroups
        self.groupSize = dInner / nGroups
        self.onesWeight = Tensor.filled(1.0, shape: [dInner / nGroups], dtype: dtype)
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in norm.parameters() { out.append(("norm.\(k)", v)) }
        for (k, v) in inProj.parameters() { out.append(("mixer.in_proj.\(k)", v)) }
        for (k, v) in outProj.parameters() { out.append(("mixer.out_proj.\(k)", v)) }
        for (k, v) in mixerNorm.parameters() { out.append(("mixer.norm.\(k)", v)) }
        return out
    }

    /// `DecoderLayer` conformance. Cache slot is a `Mamba2LayerCache`.
    public func decode(
        _ h: Tensor, position _: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        guard let layerCache = cache as? Mamba2LayerCache else {
            fatalError("NemotronHMambaLayer: expected Mamba2LayerCache, got \(type(of: cache))")
        }

        // (1) pre-mixer RMSNorm.
        let xNorm = norm(h, on: cmd)

        // (2) in_proj → split into z (gate) / xBC / dt_raw.
        //     in_proj output layout: [d_inner | conv_dim | n_heads].
        let proj = inProj(xNorm, on: cmd)
        let z = proj.slicedRows(start: 0, count: dInner)
        let xBC = proj.slicedRows(start: dInner, count: convDim)
        let dtRaw = proj.slicedRows(start: dInner + convDim, count: nHeads)

        // (3) conv1d causal step (rolling state) + SiLU.
        let convOut = Tensor.empty(shape: [convDim], dtype: dtype, device: device)
        Ops.conv1dCausalStep(
            x: xBC, w: convW, b: convB,
            state: layerCache.conv.state, into: convOut,
            nChannels: convDim, kernelSize: convKernel, on: cmd)
        let convAct = Ops.silu(convOut, on: cmd)

        // (4) split conv output → x / B / C.
        //     conv layout: [d_inner | n_groups*state_dim | n_groups*state_dim].
        let x = convAct.slicedRows(start: 0, count: dInner)
        let bAll = convAct.slicedRows(start: dInner, count: nGroups * stateDim)
            .reshaped(to: [nGroups, stateDim])
        let cAll = convAct.slicedRows(
            start: dInner + nGroups * stateDim,
            count: nGroups * stateDim
        )
        .reshaped(to: [nGroups, stateDim])

        // (5) dt = softplus(dt_raw + dt_bias). F-129 mixer-fuse: one
        // dispatch instead of two — see NemotronHMambaMixerFuse for the
        // bit-exactness contract. Default OFF: bit-exact in local dev
        // but CI's GPU recorded 142/512 bf16 mismatches for the same
        // test (gh run 31712976150) — a transcendental-unit hazard, not
        // a formula bug. Opt in with `BUTTER_NEMOTRON_MIXER_FUSE_SOFTPLUS
        // =1` only where bit-exactness has been independently
        // reconfirmed.
        let dt: Tensor
        if NemotronHMambaMixerFuse.softplusEnabled {
            dt = NemotronHMambaMixerFuse.addSoftplus(dtRaw, dtBias, on: cmd, device: device)
        } else {
            let dtSum = Ops.add(dtRaw, dtBias, on: cmd)
            dt = Ops.softplus(dtSum, on: cmd)
        }

        // (6) selective scan. F-130 grouped-SSM fuse: default ON, one
        //     dispatch spanning every (head, channel) pair instead of
        //     `nGroups` separate per-group dispatches — see
        //     NemotronHMambaGroupedSSMFuse for the bit-exactness
        //     contract. `BUTTER_NEMOTRON_SSM_GROUPED=0` restores the
        //     original per-group loop (the shipped single-group
        //     `ssm_step` kernel dispatched once per group, each on a
        //     host-sliced sub-view).
        let y = Tensor.empty(shape: [nHeads, headDim], dtype: dtype, device: device)
        let xHeads = x.reshaped(to: [nHeads, headDim])
        let stateHeads = layerCache.ssm.h  // [nHeads, headDim, stateDim]
        let bFlat = bAll.reshaped(to: [nGroups * stateDim])
        let cFlat = cAll.reshaped(to: [nGroups * stateDim])
        if NemotronHMambaGroupedSSMFuse.enabled {
            NemotronHMambaGroupedSSMFuse.ssmStepGrouped(
                x: x, a: aEff, b: bFlat, c: cFlat, dt: dt,
                state: stateHeads, into: y,
                nHeads: nHeads, headDim: headDim, stateDim: stateDim,
                headsPerGroup: headsPerGroup,
                on: cmd, device: device)
        } else {
            for g in 0 ..< nGroups {
                let h0 = g * headsPerGroup
                let xg = xHeads.slicedRows(start: h0, count: headsPerGroup)
                    .reshaped(to: [headsPerGroup * headDim])
                let yg = y.slicedRows(start: h0, count: headsPerGroup)
                let stateG = stateHeads.slicedRows(start: h0, count: headsPerGroup)
                let aG = aEff.slicedRows(start: h0, count: headsPerGroup)
                let dtG = dt.slicedRows(start: h0, count: headsPerGroup)
                let bG = bAll.slicedRows(start: g, count: 1).reshaped(to: [stateDim])
                let cG = cAll.slicedRows(start: g, count: 1).reshaped(to: [stateDim])
                Ops.ssmStep(
                    x: xg, a: aG, b: bG, c: cG, dt: dtG,
                    state: stateG, into: yg,
                    nHeads: headsPerGroup, headDim: headDim, stateDim: stateDim,
                    on: cmd)
            }
        }
        let yFlat = y.reshaped(to: [dInner])

        // (7) skip: y += D_tiled * x. F-129 mixer-fuse: default ON —
        //     pure mul+add, no transcendentals, bit-exact on both local
        //     dev and CI (unlike addSoftplus above).
        let ySkip: Tensor
        if NemotronHMambaMixerFuse.enabled {
            ySkip = NemotronHMambaMixerFuse.addMul(base: yFlat, dTiled, x, on: cmd, device: device)
        } else {
            let dx = Ops.mul(dTiled, x, on: cmd)
            ySkip = Ops.add(yFlat, dx, on: cmd)
        }

        // (8) gated mixer RMSNorm: y *= silu(z), then per-group
        //     (unweighted) RMSNorm, then scale by the learned weight.
        //     Matches mlx-lm's NemotronHRMSNormGated exactly. F-129
        //     mixer-fuse: `mulSilu` is measured NOT bit-exact anywhere
        //     it's been tried (see NemotronHMambaMixerFuse's header) so
        //     it stays behind its own opt-in gate, default OFF — same
        //     transcendental-unit hazard class as addSoftplus above,
        //     just never bit-exact even locally rather than only on CI.
        let yGated: Tensor
        if NemotronHMambaMixerFuse.siluEnabled {
            yGated = NemotronHMambaMixerFuse.mulSilu(ySkip, z, on: cmd, device: device)
        } else {
            let zAct = Ops.silu(z, on: cmd)
            yGated = Ops.mul(ySkip, zAct, on: cmd)
        }
        let yNormed = Ops.rmsNormRows(
            yGated, weight: onesWeight,
            eps: mixerNorm.eps, nRows: nGroups, rowSize: groupSize, on: cmd)
        let yScaled = Ops.mul(yNormed, mixerNorm.weight, on: cmd)

        // (9) out_proj + residual.
        let yOut = outProj(yScaled, on: cmd)
        let result = Ops.add(h, yOut, on: cmd)

        layerCache.advance()
        return result
    }

    /// F-13x chunked-scan batched prefill: `h` is `[T, hidden]`; returns
    /// `[T, hidden]`. Runs the WHOLE mixer batched over T rows —
    /// RMSNorm, in_proj, conv1d+SiLU, dt=softplus, the selective scan,
    /// D-skip, gated mixer RMSNorm, and out_proj all become one (or a
    /// few, for the scan) dispatch spanning every row instead of T
    /// per-row dispatches. See `NemotronHMambaChunkedScan`'s file
    /// header for the full design (chunked-but-still-sequential v1
    /// scan + the byte-generic column-slice/row-broadcast-tile helpers
    /// that stitch the rest of the mixer together at batched-T shape).
    /// `BUTTER_NEMOTRON_CHUNKED_SCAN=0` falls back to the exact
    /// per-row default (`DecoderLayer`'s protocol-extension
    /// implementation, duplicated below since Swift can't dispatch to
    /// a protocol default from an overriding conformance).
    public func decodeMulti(
        _ h: Tensor, startingAt position: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        guard let layerCache = cache as? Mamba2LayerCache else {
            fatalError("NemotronHMambaLayer: expected Mamba2LayerCache, got \(type(of: cache))")
        }
        let t = h.shape[0]
        guard NemotronHMambaChunkedScan.enabled else {
            let result = Tensor.empty(shape: h.shape, dtype: h.dtype, device: device)
            result.zero()
            for i in 0 ..< t {
                let row = h.slicedRows(start: i, count: 1).reshaped(to: [hidden])
                let out = decode(
                    row, position: position + i, cache: cache, cmd: cmd, device: device)
                let dst = result.slicedRows(start: i, count: 1).reshaped(to: [hidden])
                _ = Ops.add(out, dst, on: cmd, into: dst)
            }
            return result
        }

        // (1) pre-mixer RMSNorm, batched over T rows.
        let xNorm = Ops.rmsNormRows(
            h, weight: norm.weight, eps: norm.eps, nRows: t, rowSize: hidden, on: cmd)

        // (2) in_proj is ONE combined GEMM (z | xBC | dt_raw) — batch
        // it, then column-slice the three pieces back out (no per-piece
        // Linear to call separately, unlike Qwen3.5 GDN's 4-projection
        // layout — see NemotronHMambaChunkedScan's header).
        let projDim = dInner + convDim + nHeads
        let proj = inProj.callMany(xNorm, t: t, on: cmd, device: device)
        let z = NemotronHMambaChunkedScan.sliceColumns(
            proj, colStart: 0, colCount: dInner, totalCols: projDim,
            t: t, device: device, on: cmd)
        let xBC = NemotronHMambaChunkedScan.sliceColumns(
            proj, colStart: dInner, colCount: convDim, totalCols: projDim,
            t: t, device: device, on: cmd)
        let dtRaw = NemotronHMambaChunkedScan.sliceColumns(
            proj, colStart: dInner + convDim, colCount: nHeads, totalCols: projDim,
            t: t, device: device, on: cmd)

        // (3) batched conv1d + SiLU: `conv1dMany` (native-dtype output,
        // rounds at the same point `iron_conv1d_causal_step_*` does)
        // followed by the SAME `Ops.silu` dispatch `decode`'s per-token
        // path calls — NOT `Ops.conv1dCausalStepSiluCastMany` (that
        // kernel fuses conv+silu behind a single f32 accumulator with
        // no intermediate native-dtype round, which measurably
        // shortened the chunked-vs-sequential greedy match; see
        // `NemotronHMambaChunkedScan.conv1dMany`'s header).
        let convRaw = NemotronHMambaChunkedScan.conv1dMany(
            src: xBC, w: convW, b: convB,
            stateIn: layerCache.conv.state, stateOut: layerCache.conv.state,
            t: t, convDim: convDim, convKernel: convKernel,
            device: device, on: cmd)
        let convAct = Ops.silu(convRaw, on: cmd)

        // (4) split conv output -> x / B / C (same layout as `decode`'s
        // step (4)).
        let bcWidth = nGroups * stateDim
        let x = NemotronHMambaChunkedScan.sliceColumns(
            convAct, colStart: 0, colCount: dInner, totalCols: convDim,
            t: t, device: device, on: cmd)
        let bAll = NemotronHMambaChunkedScan.sliceColumns(
            convAct, colStart: dInner, colCount: bcWidth, totalCols: convDim,
            t: t, device: device, on: cmd)
        let cAll = NemotronHMambaChunkedScan.sliceColumns(
            convAct, colStart: dInner + bcWidth, colCount: bcWidth, totalCols: convDim,
            t: t, device: device, on: cmd)

        // (5) dt = softplus(dt_raw + dt_bias), batched. Deliberately
        // NOT a fused add+softplus dispatch — reuses the exact same
        // `Ops.add` + `Ops.softplus` kernels `decode`'s default
        // (non-mixer-fuse) path calls, just over `[T, nHeads]` instead
        // of `[nHeads]`; see NemotronHMambaChunkedScan's header for why
        // a fresh transcendental fusion here would reopen the CI
        // portability hazard `NemotronHMambaMixerFuse.addSoftplus`
        // already hit.
        let dtBiasTiled = NemotronHMambaChunkedScan.rowBroadcastTile(
            dtBias, t: t, device: device, on: cmd)
        let dtSum = Ops.add(dtRaw, dtBiasTiled, on: cmd)
        let dt = Ops.softplus(dtSum, on: cmd)

        // (6) chunked selective scan — the actual dispatch-count win.
        // `ceil(T / chunkSize)` dispatches instead of T; state carries
        // across chunks via `layerCache.ssm.h` (Metal serializes the
        // read-after-write across encoders on this one `cmd`, no host
        // sync needed between chunks).
        let y = Tensor.empty(shape: [t, dInner], dtype: dtype, device: device)
        let stateHeads = layerCache.ssm.h
        let chunkSize = NemotronHMambaChunkedScan.chunkSize
        var chunkStart = 0
        while chunkStart < t {
            let chunkLen = min(chunkSize, t - chunkStart)
            autoreleasepool {
                NemotronHMambaChunkedScan.scanChunk(
                    x: x.slicedRows(start: chunkStart, count: chunkLen),
                    a: aEff,
                    b: bAll.slicedRows(start: chunkStart, count: chunkLen),
                    c: cAll.slicedRows(start: chunkStart, count: chunkLen),
                    dt: dt.slicedRows(start: chunkStart, count: chunkLen),
                    state: stateHeads,
                    into: y.slicedRows(start: chunkStart, count: chunkLen),
                    nHeads: nHeads, headDim: headDim, stateDim: stateDim,
                    headsPerGroup: headsPerGroup, chunkLen: chunkLen,
                    dInner: dInner, bcStride: bcWidth,
                    on: cmd, device: device)
            }
            chunkStart += chunkLen
        }

        // (7) skip: y += D_tiled * x, batched. F-129's `addMul` fusion
        // (pure mul+add, no transcendentals — the one class verified
        // bit-exact on both local dev and CI) is reused UNCHANGED on
        // the tiled operand; only the broadcast-tile itself is new,
        // and it does no floating-point math.
        let dTiledTiled = NemotronHMambaChunkedScan.rowBroadcastTile(
            dTiled, t: t, device: device, on: cmd)
        let ySkip: Tensor
        if NemotronHMambaMixerFuse.enabled {
            ySkip = NemotronHMambaMixerFuse.addMul(base: y, dTiledTiled, x, on: cmd, device: device)
        } else {
            let dx = Ops.mul(dTiledTiled, x, on: cmd)
            ySkip = Ops.add(y, dx, on: cmd)
        }

        // (8) gated mixer RMSNorm, batched: y *= silu(z), then per-group
        // (unweighted) RMSNorm over `t * nGroups` rows (`Ops.rmsNormRows`
        // already broadcasts one `[groupSize]` weight across every row —
        // no tiling needed there, matches `decode`'s single-token call
        // broadcasting the same `onesWeight` across `nGroups` rows).
        let zAct = Ops.silu(z, on: cmd)
        let yGated = Ops.mul(ySkip, zAct, on: cmd)
        let yNormed = Ops.rmsNormRows(
            yGated.reshaped(to: [t * nGroups, groupSize]), weight: onesWeight,
            eps: mixerNorm.eps, nRows: t * nGroups, rowSize: groupSize, on: cmd
        ).reshaped(to: [t, dInner])
        let mixerNormWeightTiled = NemotronHMambaChunkedScan.rowBroadcastTile(
            mixerNorm.weight, t: t, device: device, on: cmd)
        let yScaled = Ops.mul(yNormed, mixerNormWeightTiled, on: cmd)

        // (9) out_proj + residual, batched.
        let yOut = outProj.callMany(yScaled, t: t, on: cmd, device: device)
        let result = Ops.add(h, yOut, on: cmd)

        layerCache.advanceBy(t)
        return result
    }
}

// ─── NemotronHAttentionLayer — "*" ───────────────────────────────────
//
// Multi-head attention with NO positional embedding (no RoPE). Cache
// slot is a raw `KVCache`.

public final class NemotronHAttentionLayer: Module, DecoderLayer {
    let norm: RMSNorm
    let qProj, kProj, vProj, oProj: AnyLinear
    let nHeads, nKVHeads, headDim: Int
    let scale: Float

    init(
        norm: RMSNorm,
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        nHeads: Int, nKVHeads: Int, headDim: Int
    ) {
        self.norm = norm
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in norm.parameters() { out.append(("norm.\(k)", v)) }
        for (k, v) in qProj.parameters() { out.append(("mixer.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("mixer.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("mixer.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("mixer.o_proj.\(k)", v)) }
        return out
    }

    /// `DecoderLayer` conformance. Cache slot is a `KVCache`.
    public func decode(
        _ h: Tensor, position _: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device _: Device
    ) -> Tensor {
        guard let kv = cache as? KVCache else {
            fatalError("NemotronHAttentionLayer: expected KVCache, got \(type(of: cache))")
        }
        let xNorm = norm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // No RoPE — NemotronH attention attends without positional
        // rotation. K/V go straight into the cache unrotated.
        kv.appendOnGPU(
            kFlat: k.reshaped(to: [nKVHeads, headDim]),
            vFlat: v.reshaped(to: [nKVHeads, headDim]), on: cmd)

        let (cacheK, cacheV) = kv.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: q.reshaped(to: [nHeads, headDim]), k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: kv.length, kvStride: kv.maxSeq,
            scale: scale, on: cmd)

        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        return Ops.add(h, oOut, on: cmd)
    }

    /// F-132 batched prefill: T-batched attention. `h` is `[T, hidden]`;
    /// returns `[T, hidden]`. Collapses the T-token per-row loop into
    /// batched Q/K/V projections + one KV-cache range append + one
    /// `Ops.sdpaMulti(causal: true)` dispatch + a batched output
    /// projection — mirrors `Qwen35AttentionMixer.forwardMany`, minus
    /// RoPE and the per-head Q/K RMSNorm (NemotronH attention has
    /// neither — see the file header's "Attention has no RoPE" note).
    /// All work queues onto `cmd`; no commit inside (matches `decode`'s
    /// contract — `commitsMidDecode` is `false` for this layer).
    public func decodeMulti(
        _ h: Tensor, startingAt position: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        guard let kv = cache as? KVCache else {
            fatalError("NemotronHAttentionLayer: expected KVCache, got \(type(of: cache))")
        }
        let t = h.shape[0]
        let hiddenDim = h.shape.last!
        let qDim = nHeads * headDim
        let kvDim = nKVHeads * headDim

        let xNorm = Ops.rmsNormRows(
            h, weight: norm.weight, eps: norm.eps,
            nRows: t, rowSize: hiddenDim, on: cmd)

        // Batched Q/K/V — one gemm each instead of T gemvs.
        let qOut = qProj.callMany(xNorm, t: t, on: cmd, device: device)
        let kOut = kProj.callMany(xNorm, t: t, on: cmd, device: device)
        let vOut = vProj.callMany(xNorm, t: t, on: cmd, device: device)

        // No RoPE — straight into the cache, unrotated (same as `decode`).
        let appendPositions = Tensor.empty(shape: [t], dtype: .u32, device: device)
        kv.reserveSlotsManyOnHost(t: t, into: appendPositions)
        kv.appendRangeOnGPUMany(
            kFlat: kOut.reshaped(to: [t, nKVHeads, headDim]),
            vFlat: vOut.reshaped(to: [t, nKVHeads, headDim]),
            t: t, positions: appendPositions, on: cmd)

        let (cacheK, cacheV) = kv.prepareForAttention(on: cmd)
        let attnAll: Tensor
        if headDim == 128 || headDim == 256 {
            let qBlock = qOut.reshaped(to: [t, nHeads, headDim])
            attnAll = Ops.sdpaMulti(
                q: qBlock, k: cacheK, v: cacheV,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                baseKV: position, nQuery: t, kvStride: kv.maxSeq,
                causal: true, scale: scale, on: cmd
            ).reshaped(to: [t * qDim])
        } else {
            // Escape path for a hypothetical NemotronH checkpoint whose
            // head_dim isn't 128/256 (every shipped checkpoint, including
            // Nemotron-3.5-Lightning, uses 128) — per-row sdpaDecode loop,
            // still batched everywhere else in this method.
            let scratch = Tensor.empty(shape: [t * qDim], dtype: xNorm.dtype, device: device)
            let dtBytes = xNorm.dtype.byteSize
            for r in 0 ..< t {
                let qRow = Tensor(
                    buffer: qOut.buffer, offset: qOut.offset + r * qDim * dtBytes,
                    shape: [nHeads, headDim], dtype: xNorm.dtype)
                let outRow = Tensor(
                    buffer: scratch.buffer, offset: scratch.offset + r * qDim * dtBytes,
                    shape: [nHeads, headDim], dtype: xNorm.dtype)
                _ = Ops.sdpaDecode(
                    q: qRow, k: cacheK, v: cacheV,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    nKV: position + r + 1, kvStride: kv.maxSeq,
                    scale: scale, on: cmd, into: outRow)
            }
            attnAll = scratch
        }
        _ = kvDim  // silence unused-var warning on the RoPE-free path
        let attnRows = attnAll.reshaped(to: [t, qDim])
        let oOut = oProj.callMany(attnRows, t: t, on: cmd, device: device)
        return Ops.add(h, oOut, on: cmd)
    }
}

// ─── NemotronHMLPLayer — "-" ─────────────────────────────────────────
//
// Dense feed-forward block with a squared-ReLU activation and a single
// up-projection (no gate — unlike SwiGLU). Cache slot is a
// `StatelessLayerCache` (no per-token state).

public final class NemotronHMLPLayer: Module, DecoderLayer {
    let norm: RMSNorm
    let upProj, downProj: AnyLinear
    let hidden, intermediate: Int

    init(
        norm: RMSNorm, upProj: AnyLinear, downProj: AnyLinear,
        hidden: Int, intermediate: Int
    ) {
        self.norm = norm
        self.upProj = upProj
        self.downProj = downProj
        self.hidden = hidden
        self.intermediate = intermediate
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in norm.parameters() { out.append(("norm.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mixer.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mixer.down_proj.\(k)", v)) }
        return out
    }

    /// `DecoderLayer` conformance. Cache slot is a `StatelessLayerCache`
    /// and is ignored — a dense MLP holds no per-token state.
    public func decode(
        _ h: Tensor, position _: Int,
        cache _: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device _: Device
    ) -> Tensor {
        let xNorm = norm(h, on: cmd)
        // down( relu(up(x))^2 ) — squared-ReLU activation.
        let up = upProj(xNorm, on: cmd)
        let r = Ops.relu(up, on: cmd)
        let r2 = Ops.mul(r, r, on: cmd)
        let mlpOut = downProj(r2, on: cmd)
        return Ops.add(h, mlpOut, on: cmd)
    }

    /// F-132 batched prefill: T-batched dense MLP. `h` is `[T, hidden]`;
    /// returns `[T, hidden]`. A dense MLP has no per-token state, so this
    /// is a pure batched-GEMM win — one `up.callMany` + one `down.callMany`
    /// instead of T single-row GEMVs, with the squared-ReLU activation
    /// unchanged (elementwise, already shape-agnostic).
    public func decodeMulti(
        _ h: Tensor, startingAt _: Int,
        cache _: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let t = h.shape[0]
        let rowSize = h.shape.last!
        let xNorm = Ops.rmsNormRows(
            h, weight: norm.weight, eps: norm.eps,
            nRows: t, rowSize: rowSize, on: cmd)
        let up = upProj.callMany(xNorm, t: t, on: cmd, device: device)
        let r = Ops.relu(up, on: cmd)
        let r2 = Ops.mul(r, r, on: cmd)
        let mlpOut = downProj.callMany(r2, t: t, on: cmd, device: device)
        return Ops.add(h, mlpOut, on: cmd)
    }
}

// ─── NemotronHMoELayer — "E" ─────────────────────────────────────────
//
// Block-sparse MoE feed-forward used by Nemotron-Cascade-2 and the
// Nemotron-3 Nano/Super/Ultra line. Different enough from the reusable
// `MoELayer` (squared-ReLU experts, sigmoid+bias router, post-K
// renormalisation + routed-scaling, always-on dense shared expert) that
// it lives here next to the rest of the NemotronH variants.
//
// Routing — mirrors mlx-lm's `NemotronHMoEGate` exactly:
//   1. logits = gate(x)                                  [nRouted]
//   2. origScores = sigmoid(logits.toFloat32)            [nRouted]
//   3. scores = origScores + e_score_correction_bias     [nRouted]
//   4. (grouped routing if n_group > 1; n_group == 1 → flat top-K)
//   5. inds = top-K of scores                            [topK]
//   6. finalScores = origScores[inds]   (← raw sigmoid, *not* biased)
//   7. if topK > 1 && norm_topk_prob:
//        finalScores /= sum(finalScores) + 1e-20
//   8. finalScores *= routed_scaling_factor              [topK]
//
// Per-selected-expert (squared-ReLU 2-layer MLP — NO gate, unlike
// SwiGLU):
//   y_e = down_proj_e( relu(up_proj_e(xNorm)) ^ 2 )
//
// Output:
//   y      = Σ_k finalScores[k] · y_inds[k]
//   if shared_experts: y += shared_experts(xNorm)   (squared-ReLU dense
//                                                    MLP at moe_shared_
//                                                    expert_intermediate_size)
//   result = h + y                                  (residual outside)
//
// Grouped routing (n_group > 1, topk_group > 1): NOT implemented —
// every shipped Nemotron MoE checkpoint (Cascade-2-30B-A3B, Nemotron-3
// Nano 30B-A3B, Super, Ultra) sets `n_group: 1`, which collapses to
// flat top-K. The loader rejects checkpoints with n_group > 1 with a
// clear `unsupportedConfig` until a real checkpoint exercises the
// grouped path.
//
// Command-buffer contract: `decode` commits the passed `cmd` (router
// CPU readback), then runs the experts + shared expert + residual on a
// fresh internal buffer and returns a fully-resident tensor. The host
// `NemotronHModel.forward` therefore must refresh `cmd` after this
// layer — same pattern as `MoELayer`. Cache slot is a
// `StatelessLayerCache` (no per-token state).

/// A single NemotronH expert: squared-ReLU 2-layer MLP, no gate. The
/// shared expert reuses the same structure with a wider intermediate.
public final class NemotronHExpert: Module {
    let upProj, downProj: AnyLinear

    init(upProj: AnyLinear, downProj: AnyLinear) {
        self.upProj = upProj
        self.downProj = downProj
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in upProj.parameters() { out.append(("up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("down_proj.\(k)", v)) }
        return out
    }

    /// Run the expert: down( relu(up(x))^2 ). Caller owns `cmd`.
    fileprivate func forward(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        let up = upProj(x, on: cmd)
        let r = Ops.relu(up, on: cmd)
        let r2 = Ops.mul(r, r, on: cmd)
        return downProj(r2, on: cmd)
    }
}

public final class NemotronHMoELayer: Module, DecoderLayer {
    let norm: RMSNorm
    let gate: AnyLinear
    /// `[nRouted]` correction bias applied only to the *selection*
    /// score (the FINAL weight is `sigmoid(logits)`, unbiased).
    let eScoreCorrectionBias: Tensor
    /// `[nRouted]` host-resident copy of `eScoreCorrectionBias` — read
    /// once at init so each decode skips a CPU readback of the bias.
    let eScoreCorrectionBiasHost: [Float]
    let experts: [NemotronHExpert]
    /// Always-on dense shared MLP. `nil` if the checkpoint declared
    /// `n_shared_experts == 0`.
    let sharedExpert: NemotronHExpert?

    public let hidden, nRouted, topK: Int
    public let routedScalingFactor: Float
    public let normTopkProb: Bool
    let dtype: DType

    /// `DecoderLayer.commitsMidDecode` — always `true`: this layer's
    /// router is CPU-side (a `toFloatArray()` readback of the router
    /// logits), unlike `NemotronHNvfp4MoELayer`'s GPU-routed default
    /// path. See the perf-wave doc on that sibling class for why the
    /// NVFP4/Lightning checkpoint doesn't need this.
    public var commitsMidDecode: Bool { true }

    init(
        norm: RMSNorm, gate: AnyLinear, eScoreCorrectionBias: Tensor,
        experts: [NemotronHExpert], sharedExpert: NemotronHExpert?,
        hidden: Int, topK: Int,
        routedScalingFactor: Float, normTopkProb: Bool,
        dtype: DType
    ) {
        precondition(
            topK > 0 && topK <= experts.count,
            "NemotronHMoELayer: topK \(topK) out of range "
                + "1…\(experts.count)")
        self.norm = norm
        self.gate = gate
        self.eScoreCorrectionBias = eScoreCorrectionBias
        self.eScoreCorrectionBiasHost = readFloatsNH(eScoreCorrectionBias)
        self.experts = experts
        self.sharedExpert = sharedExpert
        self.hidden = hidden
        self.nRouted = experts.count
        self.topK = topK
        self.routedScalingFactor = routedScalingFactor
        self.normTopkProb = normTopkProb
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in norm.parameters() { out.append(("norm.\(k)", v)) }
        for (k, v) in gate.parameters() { out.append(("mixer.gate.\(k)", v)) }
        out.append(
            (
                "mixer.gate.e_score_correction_bias",
                eScoreCorrectionBias
            ))
        for (i, expert) in experts.enumerated() {
            for (k, v) in expert.parameters() {
                out.append(("mixer.experts.\(i).\(k)", v))
            }
        }
        if let sharedExpert {
            for (k, v) in sharedExpert.parameters() {
                out.append(("mixer.shared_experts.\(k)", v))
            }
        }
        return out
    }

    /// `DecoderLayer` conformance. Commits `cmd` and returns a
    /// fully-resident tensor produced on a fresh internal buffer.
    public func decode(
        _ h: Tensor, position _: Int,
        cache _: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // ── Pre-mixer RMSNorm + router GEMV on the caller's buffer ───
        let xNorm = norm(h, on: cmd)
        let logitsTensor = gate(xNorm, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── CPU routing — sigmoid + bias for selection, raw sigmoid
        //                  for weighting ─────────────────────────────
        let logits = logitsTensor.toFloatArray()
        var origScores = [Float](repeating: 0, count: nRouted)
        var selectionScores = [Float](repeating: 0, count: nRouted)
        for i in 0 ..< nRouted {
            let s = 1.0 / (1.0 + Foundation.exp(-logits[i]))
            origScores[i] = s
            selectionScores[i] = s + eScoreCorrectionBiasHost[i]
        }
        let order = (0 ..< nRouted).sorted { a, b in
            if selectionScores[a] != selectionScores[b] {
                return selectionScores[a] > selectionScores[b]
            }
            return a < b
        }
        let inds = Array(order.prefix(topK))
        var finalScores = inds.map { origScores[$0] }
        if topK > 1 && normTopkProb {
            let sum = finalScores.reduce(0, +) + 1e-20
            finalScores = finalScores.map { $0 / sum }
        }
        finalScores = finalScores.map { $0 * routedScalingFactor }

        // ── Per-expert squared-ReLU MLP + weighted sum, on a fresh
        //    buffer (xNorm is already resident from the committed cmd) ─
        let workCmd = device.makeCommandBuffer()
        var accumulator: Tensor?
        for (k, ei) in inds.enumerated() {
            let y = experts[ei].forward(xNorm, on: workCmd)
            let w = Tensor.filled(
                finalScores[k], shape: [hidden],
                dtype: dtype, device: device)
            let scaled = Ops.mul(y, w, on: workCmd)
            accumulator =
                accumulator.map { Ops.add($0, scaled, on: workCmd) }
                ?? scaled
        }
        // Shared expert — squared-ReLU MLP, always-on. Mirrors
        // mlx-lm's `if let sharedExperts { y = y + sharedExperts(x) }`.
        if let sharedExpert {
            let sharedOut = sharedExpert.forward(xNorm, on: workCmd)
            accumulator =
                accumulator.map { Ops.add($0, sharedOut, on: workCmd) }
                ?? sharedOut
        }
        // ── Residual ─────────────────────────────────────────────────
        // `accumulator` is non-nil — topK ≥ 1 always picks at least one
        // expert, and even if a checkpoint somehow ships topK == 0 the
        // shared expert is always present here.
        let routedOut = accumulator!
        let result = Ops.add(h, routedOut, on: workCmd)
        workCmd.commit()
        workCmd.waitUntilCompleted()
        return result
    }
}

/// Build one MoE layer for a NemotronH checkpoint. The shared expert
/// is constructed when `nSharedExperts > 0`.
///
/// Not `private`: `NemotronHMTPHead.swift` (F-131, MTP self-speculative
/// decode) reuses this unchanged to build the MTP block's own MoE
/// sublayer (`mtp.layers.1`, `mtp_layers_block_type: ["attention",
/// "moe"]`) — the MTP checkpoint tensors are always plain BF16 (see
/// that file's header), so this always takes the unquantized branch
/// below for that call site, but the NVFP4 branch is left reachable
/// unmodified for a future quantized-MTP checkpoint.
func buildNemotronHMoELayer(
    prefix p: String, norm: RMSNorm, weights: SafeTensorsBundle,
    hidden: Int, moeIntermediate: Int,
    nRoutedExperts: Int, topK: Int,
    nSharedExperts: Int, sharedExpertIntermediate: Int,
    routedScalingFactor: Float, normTopkProb: Bool,
    dtype: DType, device: Device
) throws -> any DecoderLayer {
    // Router: hidden → nRouted logits. The gate/router projection and
    // its correction bias are NEVER NVFP4/FP8-quantized on any shipped
    // NemotronH MoE checkpoint (including Nemotron-3.5-Lightning-
    // NVFP4 — verified against the checkpoint's safetensors index), so
    // this read is unconditional. It IS, however, stored in plain F32
    // on Nemotron-3.5-Lightning specifically (every other unquantized
    // weight — attention proj, norms, embeddings — is BF16) — cast to
    // the activation dtype so `Ops.gemv`'s `weight.dtype == input.dtype`
    // precondition doesn't crash the first forward pass.
    let gateWRaw = try weights.tensor(named: "\(p).gate.weight")
    let gate = AnyLinear(Linear(weight: castMatrixNH(gateWRaw, dtype: dtype, device: device)))
    // The correction bias is per-expert; mlx-lm keeps it in fp32 — we
    // do the same (the bias is the only thing the cast-predicate
    // protects from being cast down to bf16/f16).
    let eSCBRaw = try weights.tensor(named: "\(p).gate.e_score_correction_bias")
    precondition(
        eSCBRaw.elementCount == nRoutedExperts,
        "NemotronH MoE: e_score_correction_bias has "
            + "\(eSCBRaw.elementCount) entries, expected \(nRoutedExperts)")

    // NVFP4-quantized routed experts (Nemotron-3.5-Lightning) — detect
    // via expert 0's down_proj carrying a `weight_scale_2` (the NVFP4
    // per-tensor global scale; see `NemotronHNvfp4.swift`). Every other
    // shipped NemotronH MoE checkpoint (Cascade-2, Nemotron-3
    // Nano/Super/Ultra) ships plain bf16 experts and falls through to
    // the unquantized path below unchanged.
    if weights.has("\(p).experts.0.down_proj.weight_scale_2") {
        let eScoreCorrectionBiasHost = readFloatsNH(eSCBRaw)
        let stacked = try loadNemotronHStackedNvfp4Experts(
            prefix: p, nExperts: nRoutedExperts, moeIntermediate: moeIntermediate,
            hidden: hidden, weights: weights, dtype: dtype, device: device)
        var sharedUp: Nvfp4Linear?
        var sharedDown: Nvfp4Linear?
        if nSharedExperts > 0 {
            precondition(
                nSharedExperts == 1,
                "NemotronH MoE: n_shared_experts \(nSharedExperts) > 1 not supported.")
            let sharedIntermediate =
                sharedExpertIntermediate > 0 ? sharedExpertIntermediate : moeIntermediate
            sharedUp = try loadNemotronHNvfp4Linear(
                base: "\(p).shared_experts.up_proj", outDim: sharedIntermediate, inDim: hidden,
                weights: weights, device: device)
            sharedDown = try loadNemotronHNvfp4Linear(
                base: "\(p).shared_experts.down_proj", outDim: hidden, inDim: sharedIntermediate,
                weights: weights, device: device)
        }
        return NemotronHNvfp4MoELayer(
            norm: norm, gate: gate, eScoreCorrectionBias: eScoreCorrectionBiasHost,
            stacked: stacked, sharedUp: sharedUp, sharedDown: sharedDown,
            hidden: hidden, topK: topK,
            routedScalingFactor: routedScalingFactor, normTopkProb: normTopkProb,
            dtype: dtype, device: device)
    }

    // Routed experts: each is a squared-ReLU 2-layer MLP. The published
    // checkpoint ships them as `mixer.experts.<e>.{up_proj,down_proj}.weight`
    // (Python layout — the mlx-swift port stacks them into
    // `switch_mlp.fc1/fc2` at sanitize time, but Butter keeps them
    // per-expert because we dispatch one expert per top-K slot and
    // never need the stacked form).
    var experts: [NemotronHExpert] = []
    experts.reserveCapacity(nRoutedExperts)
    for e in 0 ..< nRoutedExperts {
        let upProj = AnyLinear(
            Linear(
                weight: try weights.tensor(
                    named: "\(p).experts.\(e).up_proj.weight")))
        let downProj = AnyLinear(
            Linear(
                weight: try weights.tensor(
                    named: "\(p).experts.\(e).down_proj.weight")))
        experts.append(NemotronHExpert(upProj: upProj, downProj: downProj))
    }

    // Shared expert (n_shared_experts >= 1). The shape is wider
    // (`moe_shared_expert_intermediate_size`, ≈ 2× the routed
    // intermediate on Cascade-2) but otherwise the same squared-ReLU
    // MLP. mlx-lm only supports nSharedExperts ∈ {0, 1}; we keep the
    // same restriction here — the `n_shared_experts > 1` case has no
    // shipped checkpoint exercising it.
    let sharedExpert: NemotronHExpert?
    if nSharedExperts > 0 {
        precondition(
            nSharedExperts == 1,
            "NemotronH MoE: n_shared_experts \(nSharedExperts) > 1 "
                + "not supported (no shipped checkpoint uses it).")
        let sharedUp = AnyLinear(
            Linear(
                weight: try weights.tensor(named: "\(p).shared_experts.up_proj.weight")))
        let sharedDown = AnyLinear(
            Linear(
                weight: try weights.tensor(named: "\(p).shared_experts.down_proj.weight")))
        sharedExpert = NemotronHExpert(upProj: sharedUp, downProj: sharedDown)
        _ = sharedExpertIntermediate  // Plumbed for future shape checks.
    } else {
        sharedExpert = nil
    }
    _ = moeIntermediate  // Plumbed for future shape checks.

    return NemotronHMoELayer(
        norm: norm, gate: gate, eScoreCorrectionBias: eSCBRaw,
        experts: experts, sharedExpert: sharedExpert,
        hidden: hidden, topK: topK,
        routedScalingFactor: routedScalingFactor,
        normTopkProb: normTopkProb,
        dtype: dtype)
}

// ─── F-132 batched prefill (M>1 forward) ──────────────────────────────
//
// Default ON. `BUTTER_NEMOTRON_BATCHED_PREFILL=0` reverts
// `NemotronHModel.forwardMulti` to the original per-token loop — the
// escape hatch for bisection / a regression that only reproduces on
// the batched path.
enum NemotronHBatchedPrefill {
    static let enabled = ButterEnv["BUTTER_NEMOTRON_BATCHED_PREFILL"] != "0"
}

// ─── NemotronHModel ──────────────────────────────────────────────────

public final class NemotronHModel: LanguageModel {
    public let embedTokens: AnyEmbedding
    /// Heterogeneous layer stack — each entry is a Mamba / attention /
    /// MLP layer, ordered by the `hybrid_override_pattern`.
    public let layers: [any DecoderLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let mambaNHeads, mambaHeadDim, stateDim, convDim, convKernel, nGroups, dInner: Int
    public let dtype: DType

    /// F-131 MTP self-speculative decode: the checkpoint's optional
    /// 53rd "nextn" block (`mtp.*` tensors), loaded when
    /// `BUTTER_NEMOTRON_MTP=1` AND the checkpoint ships one (see
    /// `NemotronHMTPHead.swift`). `nil` for every checkpoint without an
    /// MTP block, and for MTP-bearing checkpoints when the env var is
    /// unset — the extra ~2.6 GB of BF16 expert weights simply never
    /// gets loaded/resident in that case. Settable post-init (not an
    /// init parameter) so `NemotronHHybrid.loadModel` can attach it
    /// after the backbone (whose `embedTokens`/`lmHead` the head shares
    /// by reference) already exists.
    public var mtpHead: NemotronHMTPHead?

    /// Layer kinds, index-aligned with `layers` — drives `makeLayerCaches`.
    let layerKinds: [NemotronHLayerKind]

    init(
        embedTokens: AnyEmbedding, layers: [any DecoderLayer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        mambaNHeads: Int, mambaHeadDim: Int, stateDim: Int,
        convDim: Int, convKernel: Int, nGroups: Int, dInner: Int,
        vocab: Int, maxSeq: Int, dtype: DType
    ) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.mambaNHeads = mambaNHeads
        self.mambaHeadDim = mambaHeadDim
        self.stateDim = stateDim
        self.convDim = convDim
        self.convKernel = convKernel
        self.nGroups = nGroups
        self.dInner = dInner
        self.vocab = vocab
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.layerKinds = layers.map { layer in
            switch layer {
            case is NemotronHMambaLayer: return .mamba
            case is NemotronHAttentionLayer: return .attention
            case is NemotronHMLPLayer: return .mlp
            case is NemotronHMoELayer: return .moe
            case is NemotronHNvfp4MoELayer: return .moe
            default:
                fatalError(
                    "NemotronHModel: unknown DecoderLayer "
                        + "\(type(of: layer)) in heterogeneous stack")
            }
        }
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("backbone.embeddings.\(k)", v))
        }
        for (i, layer) in layers.enumerated() {
            let params: [(String, Tensor)]
            switch layer {
            case let l as NemotronHMambaLayer: params = l.parameters()
            case let l as NemotronHAttentionLayer: params = l.parameters()
            case let l as NemotronHMLPLayer: params = l.parameters()
            case let l as NemotronHMoELayer: params = l.parameters()
            case let l as NemotronHNvfp4MoELayer: params = l.parameters()
            default: params = []
            }
            for (k, v) in params { out.append(("backbone.layers.\(i).\(k)", v)) }
        }
        for (k, v) in finalNorm.parameters() { out.append(("backbone.norm_f.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    /// One cache per layer index, matching the layer kind:
    ///   M → Mamba2LayerCache, * → KVCache, - → StatelessLayerCache.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        return layerKinds.map { kind in
            switch kind {
            case .mamba:
                return Mamba2LayerCache(
                    nHeads: mambaNHeads, stateDim: stateDim, headDim: mambaHeadDim,
                    convChannels: convDim, convKernelSize: convKernel,
                    dtype: dtype, device: device)
            case .attention:
                return KVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, device: device)
            case .mlp, .moe:
                return StatelessLayerCache()
            }
        }
    }

    /// Queue a single-token forward pass. Walks the heterogeneous
    /// `[any DecoderLayer]` in lockstep with the per-layer caches.
    ///
    /// Layers are checked via `DecoderLayer.commitsMidDecode`, not
    /// `layerKinds` — CPU-routed NemotronH MoE layers (`NemotronHMoELayer`,
    /// the non-NVFP4 checkpoints) still commit the command buffer
    /// mid-decode (the router readback forces a sync), but the F-126
    /// perf wave's GPU-routed `NemotronHNvfp4MoELayer` (Lightning) does
    /// not — it queues onto `cmd` exactly like the Mamba / Attention /
    /// MLP layers, so a Lightning-family model with `usesGpuRouter ==
    /// true` on every MoE layer takes the single-`cmd` fast path below
    /// with ONE commit+wait for the whole 52-layer token, not 46 (see
    /// `NemotronHNvfp4MoELayer`'s file doc). A model that still has at
    /// least one committing layer (any CPU-routed MoE layer present)
    /// falls to the private-`workCmd` path, refreshed on every
    /// committing-layer transition — the Jamba command-buffer
    /// discipline.
    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let (_, logits) = forwardHiddenAndLogits(
            tokenId: tokenId, position: position, caches: caches, on: cmd, device: device)
        return logits
    }

    /// Same single-token forward as `forward(tokenId:)`, but also
    /// returns the PRE-final-norm backbone hidden state — the seed an
    /// MTP draft head's `hnorm`/first `draftStep` needs (F-131,
    /// `NemotronHMTPHead`/`NemotronHMTPSpecDecode`; mirrors Qwen3x's
    /// `forwardManyHiddenAndLogits` returning both for the same reason).
    /// Identical body to `forward(tokenId:)` otherwise — that method is
    /// now a thin wrapper discarding the hidden half.
    public func forwardHiddenAndLogits(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> (hidden: Tensor, logits: Tensor) {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        // Without any committing layer, the entire forward queues onto
        // `cmd` exactly like before (the workCmd path needs an
        // internal commit per committing-layer transition). Pick the
        // cheaper path per checkpoint.
        let hasCommittingLayer = layers.contains { $0.commitsMidDecode }
        if !hasCommittingLayer {
            var h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])
            for (i, layer) in layers.enumerated() {
                h = layer.decode(
                    h, position: position, cache: caches[i],
                    cmd: cmd, device: device)
            }
            let normed = finalNorm(h, on: cmd)
            return (hidden: h, logits: lmHead(normed, on: cmd))
        }

        // Checkpoint has ≥1 committing layer: run every layer on a
        // private buffer so those commits don't poison the caller's
        // `cmd`. Refresh the work buffer after each layer that commits.
        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        for (i, layer) in layers.enumerated() {
            h = layer.decode(
                h, position: position, cache: caches[i],
                cmd: workCmd, device: device)
            // A layer that commits `workCmd` hands back a resident
            // tensor; we obtain a fresh buffer so the next layer's
            // dispatches don't land on an already-committed buffer.
            if layer.commitsMidDecode {
                workCmd = device.makeCommandBuffer()
            }
        }
        // Final embedding → norm + lm_head on the caller's pristine
        // `cmd`. `h` is resident (the work buffer either committed via
        // a committing layer or is about to fall out of scope unused —
        // flush it so any pending non-committing dispatches reach the
        // GPU).
        workCmd.commit()
        workCmd.waitUntilCompleted()
        let normed = finalNorm(h, on: cmd)
        return (hidden: h, logits: lmHead(normed, on: cmd))
    }

    /// Multi-token forward — prefill fast path.
    ///
    /// F-132 batched prefill (default ON, `BUTTER_NEMOTRON_BATCHED_
    /// PREFILL=0` reverts to the per-token loop below): embeds all `T`
    /// prompt tokens in one gather, then walks the heterogeneous layer
    /// stack calling each layer's `decodeMulti(_:startingAt:cache:cmd:
    /// device:)` — attention (`NemotronHAttentionLayer`) and dense MLP
    /// (`NemotronHMLPLayer`) layers take a real batched-T
    /// implementation (batched projections + one `sdpaMulti` /
    /// `gemm` dispatch instead of T single-row ones); Mamba 2
    /// (`NemotronHMambaLayer`) keeps `DecoderLayer`'s default per-row
    /// loop — the selective-scan recurrence is inherently sequential,
    /// v1 does not attempt a chunked-scan kernel (see the perf-wave
    /// doc for the profiling call that decided this); the NVFP4 MoE
    /// layer (`NemotronHNvfp4MoELayer`) gets its own `decodeMulti`
    /// override (`NemotronHNvfp4.swift`) reusing the existing batched-
    /// prefill gather-GEMM / sort-route MoE primitives when its GPU
    /// router is active, falling back to the per-row default otherwise.
    ///
    /// Command-buffer discipline mirrors `forwardHiddenAndLogits`: a
    /// layer whose `commitsMidDecode` is `true` (a CPU-routed MoE
    /// layer's host readback) forces a fresh work buffer for the next
    /// layer; every other layer queues onto the same buffer across the
    /// whole T-token chunk. Only the LAST row's hidden state needs the
    /// final norm + lm_head (the protocol's documented contract: return
    /// the tail position's logits only), so that step, and it alone,
    /// runs on the caller's `cmd`.
    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            !tokenIds.isEmpty,
            "NemotronHModel.forwardMulti: tokenIds must be non-empty")
        let t = tokenIds.count
        guard t > 1, NemotronHBatchedPrefill.enabled else {
            // T==1, or the escape hatch — original per-token loop.
            var logits: Tensor!
            for (i, tok) in tokenIds.enumerated() {
                logits = forward(
                    tokenId: tok, position: position + i,
                    caches: caches, on: cmd, device: device)
            }
            return logits
        }

        let idsBuf = device.makeBuffer(length: t * 4)
        let idsHost = tokenIds.map { UInt32($0) }
        idsHost.withUnsafeBytes { raw in
            _ = memcpy(idsBuf.contents(), raw.baseAddress!, t * 4)
        }
        let idsTensor = Tensor(buffer: idsBuf, offset: 0, shape: [t], dtype: .u32)

        var workCmd = device.makeCommandBuffer()
        // `Ops.gather` (what `embedTokens` wraps) already returns
        // `[t, hidden]` — no reshape needed.
        var h = embedTokens(idsTensor, on: workCmd)
        for (i, layer) in layers.enumerated() {
            let out = autoreleasepool { () -> Tensor in
                layer.decodeMulti(
                    h, startingAt: position, cache: caches[i],
                    cmd: workCmd, device: device)
            }
            h = out
            if layer.commitsMidDecode {
                workCmd = device.makeCommandBuffer()
            }
        }
        // Flush every pending dispatch so the last row is resident
        // before the caller's `cmd` reads it.
        workCmd.commit()
        workCmd.waitUntilCompleted()

        let lastRow = h.slicedRows(start: t - 1, count: 1).reshaped(to: [hidden])
        let normed = finalNorm(lastRow, on: cmd)
        return lmHead(normed, on: cmd)
    }

    // ─── VLM embedding-input path ────────────────────────────────────
    //
    // NemotronH is a VL-target text backbone (Nemotron-VLM wraps it).
    // The splice supplies a `[hidden]` row directly — a vision-encoder
    // token or a text-token embedding the VL model looked up. Identical
    // command-buffer discipline to `forward(tokenId:)`: checkpoints
    // with no committing layer queue onto `cmd`; checkpoints with a
    // committing layer (CPU-routed MoE) run layers on a private buffer
    // refreshed after each commit.

    public var supportsEmbeddingInput: Bool { true }

    public func forward(
        inputEmbedding: Tensor, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            inputEmbedding.elementCount == hidden,
            "NemotronHModel.forward(inputEmbedding:): expected [\(hidden)], "
                + "got \(inputEmbedding.shape)")

        let hasCommittingLayer = layers.contains { $0.commitsMidDecode }
        if !hasCommittingLayer {
            var h = inputEmbedding.reshaped(to: [hidden])
            for (i, layer) in layers.enumerated() {
                h = layer.decode(
                    h, position: position, cache: caches[i],
                    cmd: cmd, device: device)
            }
            let normed = finalNorm(h, on: cmd)
            return lmHead(normed, on: cmd)
        }

        var workCmd = device.makeCommandBuffer()
        var h = inputEmbedding.reshaped(to: [hidden])
        for (i, layer) in layers.enumerated() {
            h = layer.decode(
                h, position: position, cache: caches[i],
                cmd: workCmd, device: device)
            if layer.commitsMidDecode {
                workCmd = device.makeCommandBuffer()
            }
        }
        workCmd.commit()
        workCmd.waitUntilCompleted()
        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    /// Raw embedding-table lookup for one text token.
    public func textEmbedding(tokenId: Int, device: Device) -> Tensor {
        let cmd = device.makeCommandBuffer()
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        let embed = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])
        cmd.commit()
        cmd.waitUntilCompleted()
        return embed
    }
}

// ─── Load-time host helpers ──────────────────────────────────────────
//
// Small CPU-side derivations done once at load — the cost is in the
// noise. Mirror the `Mamba2` / `FalconH1` helpers.

/// Read an f32 / bf16 / f16 tensor into `[Float]`.
private func readFloatsNH(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .bf16:
        return t.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
    case .f16:
        return t.toArray(as: Float16.self).map { Float($0) }
    default:
        fatalError("NemotronH: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Write a `[Float]` into a fresh tensor of the requested dtype.
private func writeFloatsNH(
    _ values: [Float], shape: [Int],
    dtype: DType, device: Device
) -> Tensor {
    let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
    switch dtype {
    case .f32:
        t.copyIn(from: values)
    case .bf16:
        t.copyIn(from: values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })
    case .f16:
        t.copyIn(from: values.map { Float16($0) })
    default:
        fatalError("NemotronH: unsupported dtype for host conversion: \(dtype)")
    }
    return t
}

/// A_eff = -exp(A_log), per head, in the activation dtype.
private func computeAEffNH(
    aLog: Tensor, nHeads: Int,
    dtype: DType, device: Device
) -> Tensor {
    let floats = readFloatsNH(aLog)
    precondition(floats.count == nHeads, "NemotronH: A_log expected [n_heads]")
    return writeFloatsNH(
        floats.map { -Foundation.exp($0) },
        shape: [nHeads], dtype: dtype, device: device)
}

/// Cast a per-head / per-channel vector to the activation dtype.
private func castVectorNH(
    _ src: Tensor, count: Int,
    dtype: DType, device: Device
) -> Tensor {
    if src.dtype == dtype { return src }
    let floats = readFloatsNH(src)
    precondition(floats.count == count, "NemotronH: vector size mismatch")
    return writeFloatsNH(floats, shape: [count], dtype: dtype, device: device)
}

/// Cast a tensor of any shape to `dtype`, preserving shape. No-op if
/// already `dtype`. Nemotron-3.5-Lightning ships the MoE router
/// `gate.weight` in F32 while every other unquantized weight (attn
/// proj, norms, embeddings) is BF16 — `Ops.gemv` hard-preconditions
/// `weight.dtype == input.dtype`, so a mismatched router weight
/// crashes the very first forward pass. Cast at load time so the
/// router always matches the activation dtype, regardless of what
/// dtype a given checkpoint happens to store it in.
private func castMatrixNH(
    _ src: Tensor, dtype: DType, device: Device
) -> Tensor {
    if src.dtype == dtype { return src }
    let floats = readFloatsNH(src)
    return writeFloatsNH(floats, shape: src.shape, dtype: dtype, device: device)
}

/// Tile `D[h]` across `head_dim` channels → `[n_heads * head_dim]`.
private func tileDNH(
    d: Tensor, nHeads: Int, headDim: Int,
    dtype: DType, device: Device
) -> Tensor {
    let floats = readFloatsNH(d)
    precondition(floats.count == nHeads, "NemotronH: D expected [n_heads]")
    var tiled: [Float] = []
    tiled.reserveCapacity(nHeads * headDim)
    for h in 0 ..< nHeads {
        for _ in 0 ..< headDim { tiled.append(floats[h]) }
    }
    return writeFloatsNH(tiled, shape: [nHeads * headDim], dtype: dtype, device: device)
}

/// Transpose HF conv1d.weight `[C, 1, K]` → `[K, C]` for the iron
/// conv kernel.
private func transposeConv1dWeightNH(
    src: Tensor, kernel K: Int, channels C: Int,
    dtype: DType, device: Device
) -> Tensor {
    let floats = readFloatsNH(src)
    precondition(floats.count == K * C, "NemotronH: conv1d.weight count mismatch")
    var dst = [Float](repeating: 0, count: K * C)
    for c in 0 ..< C {
        for k in 0 ..< K { dst[k * C + c] = floats[c * K + k] }
    }
    return writeFloatsNH(dst, shape: [K, C], dtype: dtype, device: device)
}

/// A zero-filled `[n]` vector in the requested dtype.
private func zeroVectorNH(_ n: Int, dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: [n], dtype: dtype, device: device)
    t.zero()
    return t
}
