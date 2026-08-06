// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// Qwen3xMTPHead - the Multi-Token Prediction draft head for the
// Qwen 3.5 / 3.6 hybrid family (spec 030 phase 1).
//
// The published Qwen3.5/3.6 checkpoints ship a single-layer MTP
// ("nextn") head under `mtp.*` (`mtp_num_hidden_layers = 1`,
// `mtp_use_dedicated_embeddings = false`). Architecture:
//
//   h'           = layer( fc( concat( normEmbedding(embed(tok)),
//                                     normHidden(h_prev) ) ) )
//   draftLogits  = sharedLmHead( mtpNorm(h') )
//
// CONCAT ORDER (verified against the reference speculative-decode
// implementation of this head - the mainline transformers loader skips
// `mtp.*` entirely): the fc input is EMBEDDING FIRST (columns
// 0..hidden-1), HIDDEN SECOND (columns hidden..2*hidden-1), each
// through its own pre-fc norm. Swapping the halves produces silently
// garbage drafts, so the fc-fusion unit test pins the order against a
// CPU reference.
//
// `layer` is architecturally IDENTICAL to the backbone's
// full-attention layer class (gated q_proj at 2x heads, per-head
// q/k norm, partial rotary 0.25, dense SwiGLU FFN), `fc` is a
// bias-free `[hidden, 2*hidden]` fuse projection, and the embedding
// table + lm_head are SHARED with the backbone (referenced here, never
// copied).
//
// KV SEMANTICS (also verified against the reference implementation):
// the MTP layer runs as a NORMAL decoder layer - persistent KV cache,
// full causal history, backbone position ids passed through UNCHANGED.
// The phase-2 driver must therefore advance this head's cache for
// EVERY committed token, in lockstep with the backbone (not only
// during draft bursts), and chain-drafting reuses this SAME single
// layer on every draft step (`mtp_num_hidden_layers = 1`).
//
// The weights load from a `mtp-sidecar/` directory placed beside the
// backbone checkpoint (single `model.safetensors` + a `config.json`
// carrying the quantization block written by `butter convert`; keys
// are `mtp.*`). The sidecar tensors are extracted from the RAW
// published checkpoint: the standard sanitize pass DROPS every `mtp.*`
// tensor, so a sidecar by construction carries *centered* outer
// RMSNorm weights (the kernel-side `1.0 +` is implicit - the same
// fingerprint rule `Qwen35Hybrid.loadModel` keys its
// `needsCenteredNormFold` detection on). Every sidecar norm therefore
// gets the +1 fold at load time, mirroring the backbone loader's
// convention for raw checkpoints.
//
// Command-buffer contract: `draftStep` queues ALL work (embedding
// gather, fuse, attention layer, final norm, lm_head) onto the
// caller's `cmd` and never commits it. The attention layer here always
// carries a dense FFN, so `Qwen35AttentionLayer.decode` leaves `cmd`
// in-flight (its `commitsCommandBuffer` is false for dense FFN).
//
// Threading contract: NOT safe for concurrent calls. The wrapped
// `Qwen35AttentionMixer` reuses per-instance scratch tensors, so the
// caller must commit (and complete) the `cmd` of one `draftStep`
// before queueing the next.

import Foundation
import Metal

public final class Qwen3xMTPHead: Module {
    // ── Sidecar-owned modules ────────────────────────────────────────
    /// Fuse projection `[hidden, 2*hidden]`, no bias.
    let fc: AnyLinear
    /// Pre-fc RMSNorm over the previous backbone hidden state.
    let preFcNormHidden: RMSNorm
    /// Pre-fc RMSNorm over the token embedding.
    let preFcNormEmbedding: RMSNorm
    /// The single MTP decoder layer - the backbone's full-attention
    /// layer class, constructed from the `mtp.layers.0.*` tensors.
    let layer: Qwen35AttentionLayer
    /// Final `mtp.norm` RMSNorm applied before the shared lm_head.
    let mtpNorm: RMSNorm

    // ── Backbone references (shared, not copied) ─────────────────────
    let embedTokens: AnyEmbedding
    let lmHead: AnyLinear

    /// The head's OWN KV cache (the draft layer attends over its own
    /// key/value history, independent of every backbone layer cache).
    /// Unbounded eviction; capacity is `defaultCacheCapacity` slots
    /// (override with `BUTTER_QWEN_MTP_CACHE=<n>`).
    public let kvCache: KVCache

    public let hidden: Int
    public let vocab: Int
    public let dtype: DType

    /// Draft-cache slot capacity. The phase-1 driver only needs the
    /// drafting horizon plus the accepted-token history; 8192 slots
    /// costs ~34 MB at bf16 (4 KV heads x 256 dims, K+V). The spec
    /// decode engine (phase 2) can size this via the env override.
    /// TODO(spec 030 phase 2): plumb the generation cap through
    /// instead of a fixed default.
    static let defaultCacheCapacity = 8192

    init(
        fc: AnyLinear,
        preFcNormHidden: RMSNorm, preFcNormEmbedding: RMSNorm,
        layer: Qwen35AttentionLayer, mtpNorm: RMSNorm,
        embedTokens: AnyEmbedding, lmHead: AnyLinear,
        kvCache: KVCache,
        hidden: Int, vocab: Int, dtype: DType
    ) {
        self.fc = fc
        self.preFcNormHidden = preFcNormHidden
        self.preFcNormEmbedding = preFcNormEmbedding
        self.layer = layer
        self.mtpNorm = mtpNorm
        self.embedTokens = embedTokens
        self.lmHead = lmHead
        self.kvCache = kvCache
        self.hidden = hidden
        self.vocab = vocab
        self.dtype = dtype
    }

    /// Sidecar-owned parameters only - the shared embedding + lm_head
    /// belong to (and are reported by) the backbone model.
    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in fc.parameters() { out.append(("mtp.fc.\(k)", v)) }
        for (k, v) in preFcNormHidden.parameters() {
            out.append(("mtp.pre_fc_norm_hidden.\(k)", v))
        }
        for (k, v) in preFcNormEmbedding.parameters() {
            out.append(("mtp.pre_fc_norm_embedding.\(k)", v))
        }
        for (k, v) in layer.parameters() { out.append(("mtp.layers.0.\(k)", v)) }
        for (k, v) in mtpNorm.parameters() { out.append(("mtp.norm.\(k)", v)) }
        return out
    }

    /// Reset the draft KV cache (start a fresh draft context).
    public func resetCache() { kvCache.reset() }

    // ── Forward ──────────────────────────────────────────────────────

    /// The pre-layer input fusion:
    /// `fc( concat( normEmbedding(embedding), normHidden(prevHidden) ) )`
    /// - EMBEDDING half first, HIDDEN half second (see the file header;
    /// the order is load-bearing for the fc column layout). Both inputs
    /// are `[hidden]`; returns `[hidden]`. Queued on `cmd`, no commit.
    /// Internal (not private) so the fc-fusion unit test can exercise
    /// exactly this piece against a CPU reference.
    func fuseInputs(
        prevHidden: Tensor, embedding: Tensor,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            prevHidden.elementCount == hidden && embedding.elementCount == hidden,
            "Qwen3xMTPHead.fuseInputs: inputs must be [\(hidden)]")
        precondition(
            prevHidden.dtype == dtype && embedding.dtype == dtype,
            "Qwen3xMTPHead.fuseInputs: dtype mismatch (want \(dtype))")
        let hNorm = preFcNormHidden(prevHidden, on: cmd)
        let eNorm = preFcNormEmbedding(embedding, on: cmd)
        // Concat via two blits into one contiguous [2*hidden] buffer -
        // EMBEDDING half first, hidden-state half second (the fc column
        // order of the published checkpoint; see the file header).
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
        return fc(concat, on: cmd)
    }

    /// One draft decode step. `prevHidden` is the backbone's
    /// pre-final-norm hidden state at the previous position (from
    /// `Qwen35Model.forwardManyHiddenAndLogits` or a prior `draftStep`),
    /// `prevToken` the token sampled/accepted at that position, and
    /// `position` the RoPE position of the token being drafted.
    ///
    /// Returns the head's own hidden state (feedable into the next
    /// `draftStep` for multi-token drafting - the single MTP layer is
    /// reused on every chained step) and the draft logits `[vocab]`.
    /// All work queued on `cmd`; no commit inside - but the caller MUST
    /// commit + complete `cmd` before the next `draftStep` (mixer
    /// scratch reuse, see the threading contract above).
    ///
    /// KV lockstep (phase 2): each call appends one K/V row to
    /// `kvCache`, so the driver must run the head over EVERY committed
    /// token (accepted or freshly sampled) at its backbone position -
    /// the head's causal history mirrors the backbone's, with no
    /// position offset. Rejected draft rows are rolled back with
    /// `kvCache.truncate(toLength:)`.
    public func draftStep(
        prevHidden: Tensor, prevToken: Int, position: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> (hidden: Tensor, logits: Tensor) {
        // Token embedding via the shared backbone table.
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(prevToken)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(
            buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        let embedding = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

        // Fuse + the single decoder layer (dense FFN: does not commit
        // `cmd`; everything stays on the caller's buffer).
        let fused = fuseInputs(
            prevHidden: prevHidden, embedding: embedding,
            on: cmd, device: device)
        let hOut = layer.decode(
            fused, position: position, cache: kvCache,
            cmd: cmd, device: device)

        // mtp.norm + shared lm_head.
        let normed = mtpNorm(hOut, on: cmd)
        let logits = lmHead(normed, on: cmd)
        return (hidden: hOut, logits: logits)
    }

    /// T-batched fuse: same computation as `fuseInputs`, over `t` rows
    /// at once. `prevHiddenFlat` / `embeddingFlat` are `[t, hidden]`
    /// flat; returns `[t, 2*hidden]` flat (fc's batched output).
    /// EMBEDDING-FIRST / HIDDEN-SECOND column order preserved exactly
    /// (see `fuseInputs` / the file header) — each row is concatenated
    /// with its own pair of per-row blits rather than one bulk copy,
    /// since the two source tensors interleave into a strided
    /// destination. `t` is small (γ+1 at most), so T row-blits on one
    /// command buffer is still one command-buffer submission instead
    /// of `t` — the actual point of the batched pass.
    func fuseInputsMany(
        prevHiddenFlat: Tensor, embeddingFlat: Tensor, t: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            prevHiddenFlat.elementCount == t * hidden
                && embeddingFlat.elementCount == t * hidden,
            "Qwen3xMTPHead.fuseInputsMany: inputs must be [\(t), \(hidden)]")
        precondition(
            prevHiddenFlat.dtype == dtype && embeddingFlat.dtype == dtype,
            "Qwen3xMTPHead.fuseInputsMany: dtype mismatch (want \(dtype))")
        let hNormFlat = Ops.rmsNormRows(
            prevHiddenFlat, weight: preFcNormHidden.weight, eps: preFcNormHidden.eps,
            nRows: t, rowSize: hidden, on: cmd)
        let eNormFlat = Ops.rmsNormRows(
            embeddingFlat, weight: preFcNormEmbedding.weight, eps: preFcNormEmbedding.eps,
            nRows: t, rowSize: hidden, on: cmd)

        let dtBytes = dtype.byteSize
        let concat = Tensor.empty(shape: [t, 2 * hidden], dtype: dtype, device: device)
        for i in 0 ..< t {
            let eSrc = Tensor(
                buffer: eNormFlat.buffer, offset: eNormFlat.offset + i * hidden * dtBytes,
                shape: [hidden], dtype: dtype)
            let hSrc = Tensor(
                buffer: hNormFlat.buffer, offset: hNormFlat.offset + i * hidden * dtBytes,
                shape: [hidden], dtype: dtype)
            let eDst = Tensor(
                buffer: concat.buffer, offset: concat.offset + i * 2 * hidden * dtBytes,
                shape: [hidden], dtype: dtype)
            let hDst = Tensor(
                buffer: concat.buffer,
                offset: concat.offset + (i * 2 * hidden + hidden) * dtBytes,
                shape: [hidden], dtype: dtype)
            Ops.copy(eSrc, into: eDst, on: cmd)
            Ops.copy(hSrc, into: hDst, on: cmd)
        }
        // `AnyLinear.callMany` returns `[t, outDim]` (2D) — reshape to
        // the 1D `[t*hidden]` flat convention `decodeMany` requires (its
        // dense-FFN residual add is shape-strict, not just
        // elementCount-strict, and its own mixer output is always kept
        // 1D between layers).
        return fc.callMany(
            concat.reshaped(to: [t * 2 * hidden]), t: t, on: cmd, device: device
        ).reshaped(to: [t * hidden])
    }

    /// T-batched MTP KV lockstep pass — advances `kvCache` over `t`
    /// committed tokens in ONE T-row pass instead of `t` sequential
    /// `draftStep` calls (each of which pays its own command-buffer
    /// commit + `waitUntilCompleted`). `prevTokens[i]` is the token at
    /// row `i`'s own position; `prevHiddensFlat` is `[t, hidden]` flat,
    /// row `i` holding the backbone hidden at row `i`'s PREVIOUS
    /// position (exactly the feed single-token `draftStep` takes,
    /// stacked). `startPosition` is row 0's RoPE position; rows MUST be
    /// contiguous (`startPosition + i`) — the lockstep driver only ever
    /// batches a contiguous suffix of the committed block (see
    /// `MTPSpecDecode.generateGreedyMTP`).
    ///
    /// Unlike `draftStep`, this does NOT return a hidden/logits pair:
    /// the lockstep re-advance only needs the KV side effect (both
    /// values are discarded at every existing call site), so this skips
    /// `mtpNorm` + the shared `lmHead` entirely — a pure win, not just
    /// a host-sync one. Do not use this for draft-burst chaining (γ
    /// steps need the per-step hidden to feed the next step; this path
    /// has no chained dependency between rows by construction, which is
    /// exactly what makes it safe to batch without the per-call scratch
    /// aliasing that broke sharing ONE command buffer across chained
    /// `draftStep` calls).
    public func draftStepMany(
        prevHiddensFlat: Tensor, prevTokens: [Int], startPosition: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) {
        let t = prevTokens.count
        precondition(t > 0, "Qwen3xMTPHead.draftStepMany: t must be positive")
        precondition(
            prevHiddensFlat.elementCount == t * hidden,
            "Qwen3xMTPHead.draftStepMany: prevHiddensFlat size "
                + "\(prevHiddensFlat.elementCount) ≠ t*hidden = \(t * hidden)")

        // Batched token embedding — `Ops.gather` / `dequantGather`
        // already take an n-token index tensor (single-token
        // `draftStep` just passes n=1); no new gather primitive needed.
        let tokenBuf = device.makeBuffer(length: t * 4)
        let ids = prevTokens.map { UInt32($0) }
        _ = ids.withUnsafeBufferPointer { buf in
            memcpy(tokenBuf.contents(), buf.baseAddress, t * 4)
        }
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [t], dtype: .u32)
        let embeddingFlat = embedTokens(tokenTensor, on: cmd).reshaped(to: [t * hidden])

        let fused = fuseInputsMany(
            prevHiddenFlat: prevHiddensFlat, embeddingFlat: embeddingFlat,
            t: t, on: cmd, device: device)
        // Full layer forward (attention + dense FFN) for its KV side
        // effect only — mirrors single-token `draftStep`, whose
        // lockstep caller already discards both `hidden` and `logits`.
        _ = layer.decodeMany(
            fused, t: t, startPosition: startPosition,
            cache: kvCache, cmd: cmd, device: device)
    }

    // ── Loading ──────────────────────────────────────────────────────

    /// Load the head from a sidecar directory. `embedTokens` / `lmHead`
    /// are the backbone's instances (shared by reference). The
    /// remaining geometry arguments come from the backbone config (the
    /// MTP layer shares every attention hyperparameter with the
    /// backbone's full-attention layers).
    static func load(
        sidecarDirectory: URL,
        embedTokens: AnyEmbedding, lmHead: AnyLinear,
        hidden: Int, vocab: Int,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        rotaryDim: Int, ropeTheta: Float, attnOutputGate: Bool,
        eps: Float, dtype: DType, device: Device
    ) throws -> Qwen3xMTPHead {
        let weights = try SafeTensorsBundle(directory: sidecarDirectory, device: device)
        // Pin the sidecar weights in the residency set, matching the
        // main-bundle load path.
        var buffers: [MTLBuffer] = []
        for file in weights.files {
            for entry in file.entries.values { buffers.append(entry.buffer) }
        }
        device.markWeightsResident(buffers)

        // Quantization block from the sidecar's own config.json (written
        // by `butter convert` to match the backbone's quant settings).
        let config = try ModelConfig.load(from: sidecarDirectory)
        let quant = config.quantization
        if quant == nil, weights.isQuantized("mtp.fc") {
            throw Qwen35Error.missingConfig(
                "mtp-sidecar carries quantized tensors but its config.json "
                    + "has no quantization block")
        }

        // Sidecar norms are centered (raw-checkpoint convention, see the
        // file header) - fold the +1 at load, like the backbone loader
        // does for raw checkpoints.
        func loadFoldedNorm(_ name: String) throws -> RMSNorm {
            let raw = try weights.tensor(named: name)
            return RMSNorm(weight: foldGemmaRMSNormWeight(raw), eps: eps)
        }

        let fc = try loadLinear(base: "mtp.fc", in: weights, quantization: quant)
        let preFcNormHidden = try loadFoldedNorm("mtp.pre_fc_norm_hidden.weight")
        let preFcNormEmbedding = try loadFoldedNorm("mtp.pre_fc_norm_embedding.weight")
        let mtpNorm = try loadFoldedNorm("mtp.norm.weight")

        // The single MTP decoder layer - same construction as the
        // backbone's `.attention` case in `Qwen35Hybrid.loadModel`.
        let p = "mtp.layers.0"
        let inputNorm = try loadFoldedNorm("\(p).input_layernorm.weight")
        let postNorm = try loadFoldedNorm("\(p).post_attention_layernorm.weight")
        let qProj = try loadLinear(
            base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
        let kProj = try loadLinear(
            base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
        let vProj = try loadLinear(
            base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
        let oProj = try loadLinear(
            base: "\(p).self_attn.o_proj", in: weights, quantization: quant)
        let qNorm = try loadFoldedNorm("\(p).self_attn.q_norm.weight")
        let kNorm = try loadFoldedNorm("\(p).self_attn.k_norm.weight")
        let gateProj = try loadLinear(
            base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
        let upProj = try loadLinear(
            base: "\(p).mlp.up_proj", in: weights, quantization: quant)
        let downProj = try loadLinear(
            base: "\(p).mlp.down_proj", in: weights, quantization: quant)

        // Shape sanity - fail loudly on a malformed sidecar rather than
        // producing silent garbage drafts.
        let qRows = qwen3xMTPOutRows(qProj)
        let expectedQRows = (attnOutputGate ? 2 : 1) * nHeads * headDim
        guard qRows == expectedQRows else {
            throw Qwen35Error.unsupportedConfig(
                "mtp-sidecar q_proj rows \(qRows) != expected \(expectedQRows)")
        }
        let kRows = qwen3xMTPOutRows(kProj)
        guard kRows == nKVHeads * headDim else {
            throw Qwen35Error.unsupportedConfig(
                "mtp-sidecar k_proj rows \(kRows) != expected \(nKVHeads * headDim)")
        }
        let fcRows = qwen3xMTPOutRows(fc)
        guard fcRows == hidden else {
            throw Qwen35Error.unsupportedConfig(
                "mtp-sidecar fc rows \(fcRows) != hidden \(hidden)")
        }
        guard preFcNormHidden.weight.elementCount == hidden,
            preFcNormEmbedding.weight.elementCount == hidden,
            mtpNorm.weight.elementCount == hidden,
            qNorm.weight.elementCount == headDim,
            kNorm.weight.elementCount == headDim
        else {
            throw Qwen35Error.unsupportedConfig(
                "mtp-sidecar norm shapes do not match hidden=\(hidden) "
                    + "headDim=\(headDim)")
        }

        let mixer = Qwen35AttentionMixer(
            qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
            qNorm: qNorm, kNorm: kNorm,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            rotaryDim: rotaryDim, ropeTheta: ropeTheta,
            attnOutputGate: attnOutputGate, device: device)
        let layer = Qwen35AttentionLayer(
            inputNorm: inputNorm, postNorm: postNorm,
            mixer: mixer,
            ffn: .dense(
                Qwen35DenseMLP(
                    gateProj: gateProj, upProj: upProj, downProj: downProj)),
            hidden: hidden)

        let cacheCapacity =
            ProcessInfo.processInfo.environment["BUTTER_QWEN_MTP_CACHE"]
            .flatMap(Int.init) ?? defaultCacheCapacity
        let kvCache = KVCache(
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: cacheCapacity,
            dtype: dtype, device: device)

        return Qwen3xMTPHead(
            fc: fc,
            preFcNormHidden: preFcNormHidden,
            preFcNormEmbedding: preFcNormEmbedding,
            layer: layer, mtpNorm: mtpNorm,
            embedTokens: embedTokens, lmHead: lmHead,
            kvCache: kvCache,
            hidden: hidden, vocab: vocab, dtype: dtype)
    }
}

/// Output-row count of an `AnyLinear` regardless of quantization (the
/// packed quantized weight keeps its logical row count in shape[0]).
private func qwen3xMTPOutRows(_ linear: AnyLinear) -> Int {
    switch linear.inner {
    case let q as QuantizedLinear: return q.weight.shape[0]
    case let l as Linear: return l.weight.shape[0]
    default: return -1
    }
}
