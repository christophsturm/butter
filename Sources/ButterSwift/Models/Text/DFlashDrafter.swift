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
// DFlashDrafter — Meta's DFlash block-diffusion speculative drafter
// for Muse Glimmer 30B (spec: F-85 follow-up, "glimmer-dflash").
//
// Ported from the reference MLX implementation (`z-lab/dflash`'s
// `DFlashDraftModel` / `DFlashAttention`, as exercised by the
// `dflash-mlx` Apple-Silicon port) — NOT a from-scratch redesign. The
// checkpoint (`meta-models/Muse-Glimmer-30B-assistant`,
// `model.safetensors`, 5.11 GB bf16) ships alongside Muse Glimmer 30B
// and reuses the TARGET model's token embedding table and LM head —
// this file owns only the 5-layer draft transformer + the fc/norm
// that project target hidden states into the draft's own residual
// stream. See `DFlashSpecDecode.swift` for the decode-loop driver.
//
// ── Contract (from the checkpoint's config.json + reference source) ──
//
//   * `block_size = 16`: one draft call proposes 15 NEW tokens
//     following a real "anchor" token — the block is
//     `[anchor, mask, mask, ..., mask]` (16 slots), and the draft's
//     output hidden at slots `[1..<16]` (NOT slot 0) get pushed
//     through the TARGET's `lm_head` to produce 15 draft logits rows.
//   * `target_layer_ids = [1, 13, 25, 37, 49]` (5 of the target's 52
//     decoder layers, 0-indexed by layer OUTPUT). Their hidden states
//     at every input row are concatenated feature-wise
//     (`[rows, 5*hidden]`) and projected by `encoder.fc` +
//     `encoder.output_norm_enc` into the draft's own `[rows, hidden]`
//     "context" — this is NOT a single pooled vector; it is one
//     projected feature row per target input position, and it grows
//     across decode iterations (see `DFlashSpecDecode`).
//   * `mask_token_id = 201818`: a real id in the TARGET's vocabulary
//     (202048 entries) — the block's non-anchor slots are embedded
//     via the TARGET's OWN embedding table (`noise_embedding =
//     target.embed_tokens(block_tokens)`), not a draft-owned
//     embedding (the checkpoint carries none).
//   * Attention (`DFlashAttention`, per layer): keys/values are
//     projected from `concat([context, hiddenStates], axis=0)` (the
//     growing context THEN this call's block, in that row order);
//     queries are projected from `hiddenStates` (the block) only.
//     RoPE offsets both by the SAME running counter — context rows
//     first, block rows continuing from there — using the draft's OWN
//     internal position clock (`cache.length`), which is INDEPENDENT
//     of the target's absolute token position. The projected K/V rows
//     get pushed into the draft's own per-layer `KVCache`
//     (`.unbounded`); attention runs FULLY (non-causal, `causal:
//     false` — every block query attends every cached position,
//     including later block rows) over the whole cache, matching the
//     reference's `mask_mode == "none"` (never set to `"causal"` in
//     the shipped port). After every draft call the JUST-APPENDED
//     block rows (exactly `block_size` of them) are trimmed back off
//     the draft cache — only the context rows persist, becoming part
//     of the history the NEXT draft call's context gets appended
//     after.
//   * Q/K per-head RMSNorm here IS weighted (`self_attn.q_norm` /
//     `k_norm`, shape `[headDim]`) — unlike Muse Glimmer's own
//     weightless-with-synthesized-constant scheme. No `qk_scale_factor`
//     equivalent on this checkpoint.
//   * Decoder layer = plain pre-norm (`input_layernorm` → attn →
//     residual add → `post_attention_layernorm` → SwiGLU MLP →
//     residual add) — NOT Glimmer's 4-norm Gemma sandwich. Confirmed
//     against the checkpoint's tensor names (no
//     `pre_feedforward_layernorm` / `post_feedforward_layernorm`
//     keys).
//
// Losslessness does NOT depend on any of the above being pixel-exact
// to Meta's reference: `DFlashSpecDecode` verifies every proposed
// token against the TARGET's own greedy argmax and replays on any
// mismatch (see that file's header) — a drafter bug can only cost
// acceptance rate / speed, never correctness. This port is a best-
// effort faithful reproduction of the verbatim reference source
// fetched from the `dflash-mlx` project, not a guess.

import Foundation
import Metal

/// Static config parsed from the drafter checkpoint's `config.json`.
public struct DFlashConfig: Sendable {
    public let hidden: Int
    public let nLayers: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let intermediate: Int
    public let eps: Float
    public let ropeTheta: Float
    public let blockSize: Int
    public let maskTokenId: Int
    /// 0-indexed target decoder-layer OUTPUT indices whose hidden
    /// states feed the draft's context, in checkpoint order (also the
    /// order `encoder.fc`'s input columns are laid out in).
    public let targetLayerIDs: [Int]

    public init(
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        intermediate: Int, eps: Float, ropeTheta: Float, blockSize: Int,
        maskTokenId: Int, targetLayerIDs: [Int]
    ) {
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.intermediate = intermediate
        self.eps = eps
        self.ropeTheta = ropeTheta
        self.blockSize = blockSize
        self.maskTokenId = maskTokenId
        self.targetLayerIDs = targetLayerIDs
    }
}

public enum DFlashError: Error, CustomStringConvertible {
    case missingConfig(String)
    case badConfig(String)

    public var description: String {
        switch self {
        case .missingConfig(let f): return "DFlashDrafter: missing config field(s): \(f)"
        case .badConfig(let m): return "DFlashDrafter: \(m)"
        }
    }
}

/// One DFlash decoder layer. Plain pre-norm (no sandwich norm) —
/// `input_layernorm` → self-attn (context-conditioned, non-causal) →
/// residual add → `post_attention_layernorm` → SwiGLU MLP → residual
/// add.
public final class DFlashDecoderLayer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let mlpGateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm: RMSNorm
    /// Weighted per-head Q/K RMSNorm (unlike the target's weightless
    /// scheme) — real `[headDim]` weight tensors from the checkpoint.
    let qNorm, kNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim: Int
    let ropeTheta: Float
    let scale: Float

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        mlpGateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        qNorm: RMSNorm, kNorm: RMSNorm,
        hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int, ropeTheta: Float
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.mlpGateProj = mlpGateProj
        self.upProj = upProj
        self.downProj = downProj
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.hidden = hidden
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        for (k, v) in mlpGateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_attention_layernorm.\(k)", v)) }
        return out
    }

    /// `hiddenStates`: `[queryLen, hidden]` — the evolving block
    /// residual stream. `context`: `[contextLen, hidden]` — this
    /// round's fc+norm-projected target hidden states (SAME tensor
    /// passed to every layer; only `hiddenStates` evolves layer to
    /// layer, matching the reference's single `target_hidden` compute
    /// up front in `DFlashDraftModel.__call__`).
    func forward(
        _ hiddenStates: Tensor, context: Tensor,
        cache: KVCache, on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let queryLen = hiddenStates.shape[0]
        let contextLen = context.shape[0]
        precondition(hiddenStates.shape == [queryLen, hidden])
        precondition(context.shape == [contextLen, hidden])

        let xNorm = Ops.rmsNormRows(
            hiddenStates.reshaped(to: [queryLen * hidden]),
            weight: inputNorm.weight, eps: inputNorm.eps,
            nRows: queryLen, rowSize: hidden, on: cmd
        ).reshaped(to: [queryLen, hidden])

        // queries ← xNorm only.
        let q = qProj.callMany(xNorm, t: queryLen, on: cmd, device: device)
        let qNormAll = Ops.rmsNormRows(
            q.reshaped(to: [queryLen * nHeads, headDim]),
            weight: qNorm.weight, eps: qNorm.eps,
            nRows: queryLen * nHeads, rowSize: headDim, on: cmd
        ).reshaped(to: [queryLen, nHeads, headDim])

        // keys/values ← concat([context, xNorm], axis=0). Context is
        // NOT itself re-normalized here — the reference computes
        // `target_hidden = hidden_norm(fc(target_hidden))` ONCE in
        // `DFlashDraftModel.__call__` before the layer loop, and every
        // layer's attention reads that already-normalized tensor
        // directly (there is no per-layer input norm applied to
        // `context`, only to `hiddenStates` via `xNorm` above).
        let kvLen = contextLen + queryLen
        let kvStates = Tensor.empty(shape: [kvLen, hidden], dtype: hiddenStates.dtype, device: device)
        Ops.copy(context, into: kvStates.slicedRows(start: 0, count: contextLen), on: cmd)
        Ops.copy(xNorm, into: kvStates.slicedRows(start: contextLen, count: queryLen), on: cmd)

        let k = kProj.callMany(kvStates, t: kvLen, on: cmd, device: device)
        let v = vProj.callMany(kvStates, t: kvLen, on: cmd, device: device)
        let kNormAll = Ops.rmsNormRows(
            k.reshaped(to: [kvLen * nKVHeads, headDim]),
            weight: kNorm.weight, eps: kNorm.eps,
            nRows: kvLen * nKVHeads, rowSize: headDim, on: cmd
        ).reshaped(to: [kvLen, nKVHeads, headDim])

        // RoPE — one call per row, incrementing from the draft cache's
        // own running position clock. Keys start at `cache.length`
        // (context rows first, then this call's block rows); queries
        // start `contextLen` further along, at the position the FIRST
        // block row lands on once context is appended — matching the
        // reference's `rope(queries, offset=cache.offset+context_len)`
        // / `rope(keys, offset=cache.offset)`.
        let baseOffset = cache.length
        let qRot = Tensor.empty(
            shape: [queryLen * nHeads, headDim], dtype: qNormAll.dtype, device: device)
        for i in 0 ..< queryLen {
            let row = qNormAll.reshaped(to: [queryLen * nHeads, headDim])
                .slicedRows(start: i * nHeads, count: nHeads)
            let out = qRot.slicedRows(start: i * nHeads, count: nHeads)
            _ = Ops.rope(
                row, position: baseOffset + contextLen + i, headDim: headDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd, into: out)
        }
        let kRot = Tensor.empty(
            shape: [kvLen * nKVHeads, headDim], dtype: kNormAll.dtype, device: device)
        for i in 0 ..< kvLen {
            let row = kNormAll.reshaped(to: [kvLen * nKVHeads, headDim])
                .slicedRows(start: i * nKVHeads, count: nKVHeads)
            let out = kRot.slicedRows(start: i * nKVHeads, count: nKVHeads)
            _ = Ops.rope(
                row, position: baseOffset + i, headDim: headDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd, into: out)
        }

        // Append the FULL kv_states (context rows THEN block rows) to
        // the draft's own cache — the caller trims the block-sized
        // tail back off after this call returns (see
        // `DFlashDraftModel.forward`'s doc comment).
        let vFlat = v.reshaped(to: [kvLen, nKVHeads, headDim])
        for i in 0 ..< kvLen {
            cache.appendOnGPU(
                kFlat: kRot.slicedRows(start: i * nKVHeads, count: nKVHeads)
                    .reshaped(to: [nKVHeads, headDim]),
                vFlat: vFlat.slicedRows(start: i, count: 1).reshaped(to: [nKVHeads, headDim]),
                on: cmd)
        }

        // Full (non-causal) attention over the WHOLE cache — every
        // block query attends every cached position, including later
        // block rows. `mask_mode` never becomes `"causal"` on this
        // checkpoint's shipped port (see file header).
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaMulti(
            q: qRot.reshaped(to: [queryLen, nHeads, headDim]), k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            baseKV: baseOffset + contextLen, nQuery: queryLen, kvStride: cache.maxSeq,
            causal: false, scale: scale, on: cmd
        )
        let attnFlat = attnOut.reshaped(to: [queryLen, nHeads * headDim])
        let oOut = oProj.callMany(attnFlat, t: queryLen, on: cmd, device: device)

        let postAttn = Ops.add(
            hiddenStates.reshaped(to: [queryLen * hidden]),
            oOut.reshaped(to: [queryLen * hidden]),
            on: cmd
        ).reshaped(to: [queryLen, hidden])
        let mlpNorm = Ops.rmsNormRows(
            postAttn.reshaped(to: [queryLen * hidden]),
            weight: postAttnNorm.weight, eps: postAttnNorm.eps,
            nRows: queryLen, rowSize: hidden, on: cmd
        ).reshaped(to: [queryLen, hidden])

        let gate = mlpGateProj.callMany(mlpNorm, t: queryLen, on: cmd, device: device)
        let up = upProj.callMany(mlpNorm, t: queryLen, on: cmd, device: device)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj.callMany(mlpInner, t: queryLen, on: cmd, device: device)

        return Ops.add(
            postAttn.reshaped(to: [queryLen * hidden]),
            mlpOut.reshaped(to: [queryLen * hidden]),
            on: cmd
        ).reshaped(to: [queryLen, hidden])
    }
}

/// The full 5-layer DFlash draft model. Owns no embedding / LM head —
/// both are borrowed from the target (`MuseGlimmerModel.embedTokens` /
/// `.lmHead`); see `DFlashSpecDecode.swift`.
public final class DFlashDraftModel: Module {
    public let config: DFlashConfig
    let fc: AnyLinear
    let hiddenNorm: RMSNorm
    let layers: [DFlashDecoderLayer]
    let finalNorm: RMSNorm

    init(config: DFlashConfig, fc: AnyLinear, hiddenNorm: RMSNorm, layers: [DFlashDecoderLayer], finalNorm: RMSNorm) {
        self.config = config
        self.fc = fc
        self.hiddenNorm = hiddenNorm
        self.layers = layers
        self.finalNorm = finalNorm
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in fc.parameters() { out.append(("encoder.fc.\(k)", v)) }
        for (k, v) in hiddenNorm.parameters() { out.append(("encoder.output_norm_enc.\(k)", v)) }
        for (i, l) in layers.enumerated() {
            for (k, v) in l.parameters() { out.append(("layers.\(i).\(k)", v)) }
        }
        for (k, v) in finalNorm.parameters() { out.append(("norm.\(k)", v)) }
        return out
    }

    /// Fresh per-layer KV cache for one generation run. `.unbounded` —
    /// the draft's own context grows by roughly one row per emitted
    /// token over the whole run (see `DFlashSpecDecode`'s trim
    /// discipline), so `maxSeq` should cover the caller's
    /// `maxNewTokens` budget with headroom.
    public func makeCache(maxSeq: Int, device: Device = .shared) -> [KVCache] {
        (0 ..< config.nLayers).map { _ in
            KVCache(
                nKVHeads: config.nKVHeads, headDim: config.headDim, maxSeq: maxSeq,
                dtype: fc.parameters()[0].1.dtype, eviction: .unbounded, device: device)
        }
    }

    /// One draft call: propose a `[blockSize, hidden]` hidden-state
    /// block from `noiseEmbedding` (the target's raw embedding lookup
    /// of `[anchor, mask, mask, ...]`, `[blockSize, hidden]`)
    /// conditioned on `contextFeatures` (`[contextLen,
    /// targetLayerIDs.count * hidden]` — the RAW concatenated target
    /// hidden states, NOT yet fc/norm-projected). Every layer's own
    /// `KVCache` grows by `contextLen + blockSize` rows; the caller
    /// MUST trim each back down by exactly `blockSize` afterward (see
    /// `DFlashSpecDecode`) — this function does not do it itself so
    /// callers that want to inspect the pre-trim state (tests) can.
    public func forward(
        noiseEmbedding: Tensor, contextFeatures: Tensor,
        caches: [KVCache], on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(caches.count == layers.count, "DFlashDraftModel.forward: cache count mismatch")
        let contextLen = contextFeatures.shape[0]
        precondition(
            contextFeatures.shape == [contextLen, config.targetLayerIDs.count * config.hidden],
            "DFlashDraftModel.forward: contextFeatures shape \(contextFeatures.shape) "
                + "≠ [\(contextLen), \(config.targetLayerIDs.count * config.hidden)]")

        let projected = fc.callMany(contextFeatures, t: contextLen, on: cmd, device: device)
        let context = Ops.rmsNormRows(
            projected.reshaped(to: [contextLen * config.hidden]),
            weight: hiddenNorm.weight, eps: hiddenNorm.eps,
            nRows: contextLen, rowSize: config.hidden, on: cmd
        ).reshaped(to: [contextLen, config.hidden])

        var h = noiseEmbedding
        for (layer, cache) in zip(layers, caches) {
            h = layer.forward(h, context: context, cache: cache, on: cmd, device: device)
        }
        let queryLen = h.shape[0]
        return Ops.rmsNormRows(
            h.reshaped(to: [queryLen * config.hidden]),
            weight: finalNorm.weight, eps: finalNorm.eps,
            nRows: queryLen, rowSize: config.hidden, on: cmd
        ).reshaped(to: [queryLen, config.hidden])
    }

    // MARK: - Loading

    public static func loadModel(
        path: String, device: Device = .shared
    ) throws -> DFlashDraftModel {
        let url = URL(fileURLWithPath: path)
        let configURL = url.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        guard let raw = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw DFlashError.badConfig("config.json is not a JSON object")
        }
        func intField(_ key: String) throws -> Int {
            guard let v = raw[key] as? Int else { throw DFlashError.missingConfig(key) }
            return v
        }
        guard let hidden = raw["hidden_size"] as? Int,
            let nLayers = raw["num_hidden_layers"] as? Int,
            let nHeads = raw["num_attention_heads"] as? Int,
            let nKVHeads = raw["num_key_value_heads"] as? Int,
            let headDim = raw["head_dim"] as? Int,
            let intermediate = raw["intermediate_size"] as? Int,
            let eps = raw["rms_norm_eps"] as? Double,
            let blockSize = raw["block_size"] as? Int,
            let maskTokenId = raw["mask_token_id"] as? Int,
            let targetLayerIDs = raw["target_layer_ids"] as? [Int]
        else {
            throw DFlashError.missingConfig(
                "hidden_size/num_hidden_layers/num_attention_heads/num_key_value_heads/"
                    + "head_dim/intermediate_size/rms_norm_eps/block_size/mask_token_id/target_layer_ids")
        }
        let ropeTheta: Float
        if let ropeParams = raw["rope_parameters"] as? [String: Any],
            let theta = ropeParams["rope_theta"] as? Double
        {
            ropeTheta = Float(theta)
        } else if let theta = raw["rope_theta"] as? Double {
            ropeTheta = Float(theta)
        } else {
            ropeTheta = 500_000.0
        }

        let config = DFlashConfig(
            hidden: hidden, nLayers: nLayers, nHeads: nHeads, nKVHeads: nKVHeads,
            headDim: headDim, intermediate: intermediate, eps: Float(eps),
            ropeTheta: ropeTheta, blockSize: blockSize, maskTokenId: maskTokenId,
            targetLayerIDs: targetLayerIDs)

        let weights = try SafeTensorsBundle(directory: url, device: device)

        let fc = try loadLinear(base: "encoder.fc", in: weights, quantization: nil)
        let hiddenNormWeight = try weights.tensor(named: "encoder.output_norm_enc.weight")
        let hiddenNorm = RMSNorm(weight: hiddenNormWeight, eps: Float(eps))

        var layers: [DFlashDecoderLayer] = []
        layers.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let p = "layers.\(i)"
            let qProj = try loadLinear(base: "\(p).self_attn.q_proj", in: weights, quantization: nil)
            let kProj = try loadLinear(base: "\(p).self_attn.k_proj", in: weights, quantization: nil)
            let vProj = try loadLinear(base: "\(p).self_attn.v_proj", in: weights, quantization: nil)
            let oProj = try loadLinear(base: "\(p).self_attn.o_proj", in: weights, quantization: nil)
            let mlpGateProj = try loadLinear(base: "\(p).mlp.gate_proj", in: weights, quantization: nil)
            let upProj = try loadLinear(base: "\(p).mlp.up_proj", in: weights, quantization: nil)
            let downProj = try loadLinear(base: "\(p).mlp.down_proj", in: weights, quantization: nil)
            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"), eps: Float(eps))
            let postAttnNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"), eps: Float(eps))
            let qNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).self_attn.q_norm.weight"), eps: Float(eps))
            let kNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).self_attn.k_norm.weight"), eps: Float(eps))
            layers.append(
                DFlashDecoderLayer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    mlpGateProj: mlpGateProj, upProj: upProj, downProj: downProj,
                    inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                    qNorm: qNorm, kNorm: kNorm,
                    hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    ropeTheta: ropeTheta))
        }

        let finalNorm = RMSNorm(weight: try weights.tensor(named: "norm.weight"), eps: Float(eps))

        return DFlashDraftModel(
            config: config, fc: fc, hiddenNorm: hiddenNorm, layers: layers, finalNorm: finalNorm)
    }
}
