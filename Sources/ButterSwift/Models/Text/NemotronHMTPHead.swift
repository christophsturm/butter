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
// NemotronHMTPHead — the Multi-Token Prediction draft head for
// Nemotron-3.5-Lightning-30B-A3B (F-131, spec-decode follow-up to
// Qwen3x's `Qwen3xMTPHead`).
//
// ─── Archaeology (checkpoint tensor map) ──────────────────────────────
//
// `~/models/nemotron35-lightning-nvfp4` (30B-A3B, 52 backbone layers —
// 23 mamba / 23 moe / 6 attention, hidden 2688) carries a 53rd block
// under `mtp.*`, unwired since the initial port. `config.json`:
//
//   num_nextn_predict_layers: 1
//   mtp_layers_block_type:    ["attention", "moe"]
//
// Tensor map (`mtp.layers.0` = the fusion + attention half,
// `mtp.layers.1` = the MoE half — matches `mtp_layers_block_type`
// exactly):
//
//   mtp.layers.0.enorm.weight              [hidden]      RMSNorm(embed)
//   mtp.layers.0.hnorm.weight               [hidden]      RMSNorm(prevHidden)
//   mtp.layers.0.eh_proj.weight             [hidden, 2*hidden]  no bias
//   mtp.layers.0.norm.weight                [hidden]      pre-attn RMSNorm
//   mtp.layers.0.mixer.{q,k,v,o}_proj.weight            (no RoPE — matches
//                                                         every NemotronH
//                                                         attention layer)
//   mtp.layers.1.norm.weight                [hidden]      pre-MoE RMSNorm
//   mtp.layers.1.mixer.gate.{weight,e_score_correction_bias}
//   mtp.layers.1.mixer.experts.<0..127>.{up,down}_proj.weight
//   mtp.layers.1.mixer.shared_experts.{up,down}_proj.weight
//   mtp.layers.1.final_layernorm.weight     [hidden]      plays backbone's
//                                                          `norm_f` role
//
// No separate embedding table or lm_head under `mtp.*` — both are
// SHARED with the backbone (`backbone.embeddings.weight` /
// `lm_head.weight`), referenced here, never copied. Every `mtp.*`
// tensor is plain BF16 — the checkpoint's `quantization_config.ignore`
// list carries a literal `"mtp*"` entry, i.e. the MTP block was
// deliberately excluded from the NVFP4/FP8 quantization pass that
// covers the 52-layer backbone (confirmed against the safetensors
// header directly: no `weight_scale`/`weight_scale_2` sibling on any
// `mtp.*` tensor). The attention sublayer's shapes (q_proj
// `[4096, 2688]`, k_proj `[256, 2688]`) match the backbone's own
// attention layers exactly (`nHeads=32`, `nKVHeads=2`, `headDim=128`) —
// same hyperparameters, just unquantized. All 270 `mtp.*` tensors live
// in the checkpoint's LAST shard (`model-00052-of-00052.safetensors`),
// ~2.6 GB of BF16 (dominated by the 128 routed experts) — roughly 2%
// of the 52-layer target's weight footprint, matching the "cheap
// drafter" assumption this feature is built on.
//
// ─── Architecture ──────────────────────────────────────────────────────
//
// No reference forward pass for this checkpoint's MTP block ships in
// this repo's standard tooling, so the layout below was verified
// against the reference MTP implementation for this checkpoint:
//
//   fused        = eh_proj( concat[ enorm(embed(tok)), hnorm(prevHidden) ] )
//   afterAttn    = attnSublayer.decode(fused, ...)     // normal NemotronH
//                                                       // attention layer:
//                                                       // residual=fused,
//                                                       // norm+attn+add
//   afterMoe     = moeSublayer.decode(afterAttn, ...)  // normal NemotronH
//                                                       // MoE layer, same
//                                                       // residual pattern
//   finalHidden  = finalLayerNorm(afterMoe)
//   draftLogits  = sharedLmHead(finalHidden)
//
// CONCAT ORDER: embedding-first, hidden-second — same order
// Qwen3xMTPHead's fc uses; the fc-fusion unit test pins it. `eh_proj`'s
// OUTPUT REPLACES the block's hidden-state input entirely (it becomes
// the input to the attention sublayer's own forward) — it is NOT added
// to anything; the attention/MoE sublayers' own internal pre-norm +
// residual pattern (already implemented by `NemotronHAttentionLayer.
// decode` / `NemotronHMoELayer.decode` unchanged) takes over from
// there. `final_layernorm` applies AFTER the MoE sublayer's own
// residual add, mirroring the backbone's own `norm_f` — this head
// therefore composes ENTIRELY from the backbone's existing
// decoder-layer classes; zero new Metal.
//
// ─── KV semantics ──────────────────────────────────────────────────────
//
// The attention sublayer runs as a normal decoder layer with its OWN
// persistent KV cache (independent of every backbone layer's cache,
// mirroring `Qwen3xMTPHead.kvCache`) — full causal history, backbone
// position ids passed through unchanged. `NemotronHMTPSpecDecode` must
// advance this cache for every committed token, in lockstep with the
// backbone, and reuses the SAME single attention sublayer on every
// chained draft step (`num_nextn_predict_layers = 1`). The MoE
// sublayer is stateless (`StatelessLayerCache`) — no cache to manage.
//
// ─── Why NO Mamba/SSM state lives in this head ────────────────────────
//
// The MTP block's own two sublayers are attention + MoE — no Mamba 2
// mixer — so the head itself needs no `SSMStateCache`/`ConvStateCache`
// rollback machinery. The TARGET's 23 Mamba layers are a different
// story (see `NemotronHMTPSpecDecode.swift`'s design-choice doc for why
// the driver's answer is "never speculatively write target state in
// the first place" rather than snapshot+restore).
//
// ─── Command-buffer / threading contract ──────────────────────────────
//
// Mirrors `Qwen3xMTPHead.draftStep`: queues embedding gather, fuse, the
// attention sublayer, and (for `draftStep`'s trailing MoE + norm + lm_head)
// onto the caller's `cmd`. UNLIKE Qwen's dense-FFN-only head, this
// head's MoE sublayer (`NemotronHMoELayer.decode`) COMMITS the buffer
// internally (CPU router readback — `commitsMidDecode == true`, same as
// every CPU-routed NemotronH MoE layer in the backbone), so `draftStep`
// hands it a private work buffer and refreshes `cmd` afterward — same
// discipline `NemotronHModel.forward` already uses for a
// committing-layer transition. Callers therefore do NOT need to
// commit+wait between `draftStep` calls purely for this reason, but
// SHOULD anyway before feeding the returned hidden into a chained draft
// step, since `NemotronHAttentionLayer`/`NemotronHMoELayer` reuse no
// per-instance scratch across calls (unlike Qwen's mixer) — the
// no-commit-needed property here is a courtesy, not a correctness
// requirement either way.

import Foundation
import Metal

public final class NemotronHMTPHead: Module {
    // ── Fusion (mtp.layers.0's non-attention tensors) ─────────────────
    let ehProj: AnyLinear
    let enorm: RMSNorm
    let hnorm: RMSNorm

    // ── The two MTP sublayers — reuse the backbone's own classes ──────
    let attnSublayer: NemotronHAttentionLayer
    let moeSublayer: any DecoderLayer
    let finalLayerNorm: RMSNorm

    // ── Backbone references (shared, not copied) ──────────────────────
    let embedTokens: AnyEmbedding
    let lmHead: AnyLinear

    /// The attention sublayer's own KV cache — see the file header's KV
    /// semantics note.
    public let kvCache: KVCache
    /// Stateless cache slot for the MoE sublayer's `DecoderLayer.decode`
    /// call (no per-token state).
    private let moeCache = StatelessLayerCache()

    public let hidden: Int
    public let vocab: Int
    public let dtype: DType

    /// Draft-cache slot capacity — mirrors `Qwen3xMTPHead.defaultCacheCapacity`.
    static let defaultCacheCapacity = 8192

    init(
        ehProj: AnyLinear, enorm: RMSNorm, hnorm: RMSNorm,
        attnSublayer: NemotronHAttentionLayer, moeSublayer: any DecoderLayer,
        finalLayerNorm: RMSNorm,
        embedTokens: AnyEmbedding, lmHead: AnyLinear,
        kvCache: KVCache, hidden: Int, vocab: Int, dtype: DType
    ) {
        self.ehProj = ehProj
        self.enorm = enorm
        self.hnorm = hnorm
        self.attnSublayer = attnSublayer
        self.moeSublayer = moeSublayer
        self.finalLayerNorm = finalLayerNorm
        self.embedTokens = embedTokens
        self.lmHead = lmHead
        self.kvCache = kvCache
        self.hidden = hidden
        self.vocab = vocab
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in ehProj.parameters() { out.append(("mtp.layers.0.eh_proj.\(k)", v)) }
        for (k, v) in enorm.parameters() { out.append(("mtp.layers.0.enorm.\(k)", v)) }
        for (k, v) in hnorm.parameters() { out.append(("mtp.layers.0.hnorm.\(k)", v)) }
        for (k, v) in attnSublayer.parameters() { out.append(("mtp.layers.0.\(k)", v)) }
        for (k, v) in moeSublayer.parameters() { out.append(("mtp.layers.1.\(k)", v)) }
        for (k, v) in finalLayerNorm.parameters() {
            out.append(("mtp.layers.1.final_layernorm.\(k)", v))
        }
        return out
    }

    /// Reset the draft KV cache (start a fresh draft context).
    public func resetCache() { kvCache.reset() }

    // ── Forward ─────────────────────────────────────────────────────

    /// `eh_proj( concat( enorm(embedding), hnorm(prevHidden) ) )` —
    /// EMBEDDING half first, HIDDEN half second (see the file header;
    /// load-bearing for the eh_proj column layout, verified against the
    /// reference MTP implementation for this checkpoint). Both inputs
    /// `[hidden]`; returns `[hidden]`. Queued on `cmd`, no commit.
    func fuseInputs(
        prevHidden: Tensor, embedding: Tensor,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            prevHidden.elementCount == hidden && embedding.elementCount == hidden,
            "NemotronHMTPHead.fuseInputs: inputs must be [\(hidden)]")
        precondition(
            prevHidden.dtype == dtype && embedding.dtype == dtype,
            "NemotronHMTPHead.fuseInputs: dtype mismatch (want \(dtype))")
        let eNorm = enorm(embedding, on: cmd)
        let hNorm = hnorm(prevHidden, on: cmd)
        let dtBytes = dtype.byteSize
        let concat = Tensor.empty(shape: [2 * hidden], dtype: dtype, device: device)
        let firstHalf = Tensor(
            buffer: concat.buffer, offset: concat.offset,
            shape: [hidden], dtype: dtype)
        let secondHalf = Tensor(
            buffer: concat.buffer, offset: concat.offset + hidden * dtBytes,
            shape: [hidden], dtype: dtype)
        Ops.copy(eNorm, into: firstHalf, on: cmd)
        Ops.copy(hNorm, into: secondHalf, on: cmd)
        return ehProj(concat, on: cmd)
    }

    /// One draft decode step. `prevHidden` is the backbone's
    /// pre-final-norm hidden state at the previous position (from
    /// `NemotronHModel.forward`'s returned hidden, or a prior
    /// `draftStep`), `prevToken` the token sampled/accepted at that
    /// position, `position` the position of the token being drafted.
    ///
    /// Returns the MTP block's FULL output hidden state — post-attention
    /// AND post-MoE, pre-final-norm (the chained-draft hidden from the
    /// MTP formulation this block follows) — feedable into the next
    /// `draftStep` as `prevHidden` (the single
    /// MTP block is reused on every chained step, `num_nextn_predict_
    /// layers = 1`), and the draft logits `[vocab]`.
    ///
    /// COMMAND-BUFFER CONTRACT — UNLIKE `Qwen3xMTPHead.draftStep`: this
    /// method ALWAYS commits+completes `cmd` before returning (and a
    /// second, internal buffer). The MoE sublayer's CPU router readback
    /// forces a commit partway through regardless of what the caller
    /// wants (`commitsMidDecode == true` for every plain-bf16 NemotronH
    /// MoE layer, which every shipped MTP checkpoint's `mtp.layers.1`
    /// resolves to — see the file header); a caller-visible "never
    /// commits `cmd`" contract would be a lie for this head. Callers
    /// must therefore NOT commit `cmd` again afterward — doing so
    /// double-commits an already-submitted `MTLCommandBuffer`, which
    /// Metal does not allow. Pass a FRESH, never-committed buffer in.
    ///
    /// KV lockstep: each call appends one K/V row to `kvCache`, so the
    /// driver must run this head over EVERY committed token at its
    /// backbone position — the head's causal history mirrors the
    /// backbone's. Rejected draft rows are rolled back with
    /// `kvCache.truncate(toLength:)`.
    public func draftStep(
        prevHidden: Tensor, prevToken: Int, position: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> (hidden: Tensor, logits: Tensor) {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(prevToken)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        let embedding = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

        let fused = fuseInputs(
            prevHidden: prevHidden, embedding: embedding, on: cmd, device: device)
        // Attention sublayer never commits (dense proj + SDPA only).
        let afterAttn = attnSublayer.decode(
            fused, position: position, cache: kvCache, cmd: cmd, device: device)

        // MoE sublayer takes the SAME `cmd` — NOT a fresh buffer.
        // `NemotronHMoELayer.decode`'s contract (matching
        // `NemotronHModel.forward`'s committing-layer discipline) is:
        // it COMMITS the `cmd` it's handed as its FIRST action (to
        // flush the router GEMV + everything queued on it so far,
        // INCLUDING the attention sublayer's dispatches above — they
        // share this same buffer), then runs the experts on its own
        // internal buffer and returns a fully-resident tensor. Handing
        // it a FRESH empty buffer instead (F-131's first cut of this
        // file did exactly that) is a real bug, not just a missed
        // optimization: `afterAttn`'s producing dispatches would still
        // be sitting UNCOMMITTED on `cmd` while a disconnected `workCmd`
        // tried to read it — a genuine cross-command-buffer race with no
        // ordering guarantee (confirmed by the symptom: 0% draft
        // acceptance against the real checkpoint, fixed by this change).
        let afterMoe = moeSublayer.decode(
            afterAttn, position: position, cache: moeCache, cmd: cmd, device: device)
        // A future MTP checkpoint's MoE sublayer might NOT commit (e.g.
        // an NVFP4 GPU-routed variant, mirroring `NemotronHNvfp4MoELayer`)
        // — `cmd` would then still be open with `afterMoe`'s dispatches
        // queued on it. Flush before handing off to a fresh buffer below
        // so `finalLayerNorm`/`lmHead` never race a still-open `cmd`
        // either, same reasoning as above.
        if !moeSublayer.commitsMidDecode {
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let finalCmd = device.makeCommandBuffer()
        let normed = finalLayerNorm(afterMoe, on: finalCmd)
        let logits = lmHead(normed, on: finalCmd)
        finalCmd.commit()
        finalCmd.waitUntilCompleted()
        // Chain the BLOCK'S FULL output (post-attention AND post-MoE,
        // pre-final-norm — i.e. `afterMoe`, the chained-draft hidden
        // from the MTP formulation this block follows) as the next
        // chained draftStep's `prevHidden`, NOT the
        // attention-only intermediate `afterAttn`. Returning `afterAttn`
        // here was F-131's second bug: it corrupted every chained draft
        // step beyond the first in a burst (step 0 is seeded from the
        // REAL backbone hidden and is unaffected by this one).
        return (hidden: afterMoe, logits: logits)
    }

    // ── Loading ─────────────────────────────────────────────────────

    /// Load the head from the SAME `weights` bundle the backbone loads
    /// from — unlike Qwen3x's separate `mtp-sidecar/` directory,
    /// Nemotron-3.5-Lightning ships `mtp.*` tensors inline in the main
    /// checkpoint's safetensors shards (all in the last shard,
    /// `model-00052-of-00052.safetensors` on the published release).
    /// `embedTokens`/`lmHead` are the backbone's instances (shared by
    /// reference, matching `Qwen3xMTPHead.load`'s convention). The MoE
    /// hyperparameters are the backbone's own MoE config values —
    /// Nemotron-3.5-Lightning's `mtp.layers.1` uses the identical
    /// routing geometry (128 routed / top-6 / 1 shared /
    /// routed_scaling_factor 2.5 / norm_topk_prob) as every backbone MoE
    /// layer; nothing in `mtp_layers_block_type` suggests a
    /// checkpoint-specific override, and no shipped checkpoint carries
    /// one.
    static func load(
        weights: SafeTensorsBundle,
        embedTokens: AnyEmbedding, lmHead: AnyLinear,
        hidden: Int, vocab: Int,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        moeIntermediate: Int, nRoutedExperts: Int, moeTopK: Int,
        nSharedExperts: Int, sharedExpertIntermediate: Int,
        routedScalingFactor: Float, normTopkProb: Bool,
        eps: Float, dtype: DType, device: Device
    ) throws -> NemotronHMTPHead {
        precondition(
            nRoutedExperts > 0,
            "NemotronHMTPHead.load: nRoutedExperts must be > 0 — the "
                + "backbone MoE config (n_routed_experts) must be resolved "
                + "before loading an MTP block")

        let ehProj = AnyLinear(
            Linear(weight: try weights.tensor(named: "mtp.layers.0.eh_proj.weight")))
        let enorm = RMSNorm(
            weight: try weights.tensor(named: "mtp.layers.0.enorm.weight"), eps: eps)
        let hnorm = RMSNorm(
            weight: try weights.tensor(named: "mtp.layers.0.hnorm.weight"), eps: eps)

        // Attention sublayer — identical construction to the backbone's
        // `.attention` case in `NemotronHHybrid.loadModel` (no RoPE, no
        // qk-norm, plain bf16 dense proj — the MTP block is never
        // NVFP4/FP8 quantized, see the file header).
        let attnNorm = RMSNorm(
            weight: try weights.tensor(named: "mtp.layers.0.norm.weight"), eps: eps)
        let qProj = AnyLinear(
            Linear(weight: try weights.tensor(named: "mtp.layers.0.mixer.q_proj.weight")))
        let kProj = AnyLinear(
            Linear(weight: try weights.tensor(named: "mtp.layers.0.mixer.k_proj.weight")))
        let vProj = AnyLinear(
            Linear(weight: try weights.tensor(named: "mtp.layers.0.mixer.v_proj.weight")))
        let oProj = AnyLinear(
            Linear(weight: try weights.tensor(named: "mtp.layers.0.mixer.o_proj.weight")))
        let attnSublayer = NemotronHAttentionLayer(
            norm: attnNorm, qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim)

        // MoE sublayer — reuse the backbone's own builder unchanged
        // (auto-detects NVFP4 vs plain bf16 per-tensor; always resolves
        // to plain bf16 for this checkpoint's `mtp.*` tensors).
        let moeNorm = RMSNorm(
            weight: try weights.tensor(named: "mtp.layers.1.norm.weight"), eps: eps)
        let moeSublayer = try buildNemotronHMoELayer(
            prefix: "mtp.layers.1.mixer", norm: moeNorm, weights: weights,
            hidden: hidden, moeIntermediate: moeIntermediate,
            nRoutedExperts: nRoutedExperts, topK: moeTopK,
            nSharedExperts: nSharedExperts, sharedExpertIntermediate: sharedExpertIntermediate,
            routedScalingFactor: routedScalingFactor, normTopkProb: normTopkProb,
            dtype: dtype, device: device)

        let finalLayerNorm = RMSNorm(
            weight: try weights.tensor(named: "mtp.layers.1.final_layernorm.weight"), eps: eps)

        // Shape sanity — fail loudly on a malformed/future checkpoint
        // rather than producing silent garbage drafts.
        guard let ehProjLinear = ehProj.inner as? Linear,
            ehProjLinear.weight.shape == [hidden, 2 * hidden]
        else {
            throw NemotronHError.unsupportedConfig(
                "mtp.layers.0.eh_proj.weight shape mismatch — expected [\(hidden), \(2 * hidden)]"
            )
        }
        guard enorm.weight.elementCount == hidden, hnorm.weight.elementCount == hidden,
            finalLayerNorm.weight.elementCount == hidden
        else {
            throw NemotronHError.unsupportedConfig(
                "mtp.* norm shapes do not match hidden=\(hidden)")
        }

        // The head's own KV cache — sized like Qwen3xMTPHead's, lives
        // for the head's entire lifetime (built once at load, `.weight`
        // residency so `Device.releaseGenerationCaches()` never drops
        // it between generations — see `Qwen3xMTPHead.load`'s identical
        // comment for the residency-kind rationale).
        let cacheCapacity =
            ButterEnv["BUTTER_NEMOTRON_MTP_CACHE"]
            .flatMap(Int.init) ?? defaultCacheCapacity
        let kvCache = KVCache(
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: cacheCapacity,
            dtype: dtype, device: device, residency: .weight)

        return NemotronHMTPHead(
            ehProj: ehProj, enorm: enorm, hnorm: hnorm,
            attnSublayer: attnSublayer, moeSublayer: moeSublayer,
            finalLayerNorm: finalLayerNorm,
            embedTokens: embedTokens, lmHead: lmHead,
            kvCache: kvCache, hidden: hidden, vocab: vocab, dtype: dtype)
    }
}
