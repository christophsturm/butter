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
// carrying the quantization block written by `butter convert`). Both
// historical `mtp.*` keys and the unprefixed standalone-head layout
// published for Qwen3.8 are accepted. Historical sidecars extracted
// `mtp.*` tensors from a raw checkpoint and therefore carry centered
// RMSNorm weights (the kernel-side `1.0 +` is implicit). The standalone
// Qwen3.8 head is already sanitized and must not receive that fold again.
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

/// Which draft steps (if any) consult the cheaper 3-bit draft-only
/// `lm_head` instead of the shared 4-bit one. Parsed once, at head-load
/// time, from `BUTTER_QWEN_MTP_DRAFT_LMHEAD_3BIT` — see
/// `Qwen3xMTPHead.load` and the `draftLmHead` doc comment. `draftStep`
/// is the ONLY consumer; verify/backbone logits always read the shared
/// 4-bit `lmHead` regardless of this mode.
///
///   - "1"       -> `.always`     every draft step in every burst
///   - "greedy"  -> `.greedyOnly` only when the run is greedy
///                  (temperature == 0); sampled runs fall back to 4-bit
///   - "depth2+" -> `.depth2Plus` 4-bit for draft-step 0 of a burst (the
///                  first speculative token — errors here "anchor" and
///                  poison the rest of the chain under sampling), 3-bit
///                  for every chained step after it (index ≥ 1). Measured
///                  NOT a reliable net win for sampled runs (2 A/B reps
///                  landed on opposite sides of the flag-off baseline,
///                  59.1%/42.2% accept vs 43.9%/46.4% off) — stays
///                  opt-in only, never auto-selected.
///   - unset     -> `.off`. Greedy 3-bit is a measured WASH on
///                  throughput, not a win: the draft bucket shrinks
///                  (~1.9 -> 1.7s / 512 tok) but acceptance drops
///                  ~1-2pp and the two cancel — paired A/B read
///                  +0.7% in one session and -0.7% in another, i.e.
///                  zero inside noise. A default flip also makes the
///                  regenerable sidecar artifact load-bearing for
///                  every greedy caller. Not worth it; digest-safety
///                  is proven, so flipping later is a one-line change
///                  if a real win ever shows up.
///   - any other value -> treated as unset (`.off`), with a
///                  stderr warning (typo guard)
enum DraftLmHead3BitMode: Equatable {
    case off
    case always
    case greedyOnly
    case depth2Plus

    static func parse(_ raw: String?) -> DraftLmHead3BitMode {
        switch raw {
        case nil: return .off
        case "1": return .always
        case "greedy": return .greedyOnly
        case "depth2+": return .depth2Plus
        default:
            fputs(
                "[qwen3x] WARNING: BUTTER_QWEN_MTP_DRAFT_LMHEAD_3BIT="
                    + "\(raw ?? "") not recognized (want 1|greedy|depth2+); "
                    + "treating as unset (off)\n",
                stderr)
            return .off
        }
    }
}

@inline(__always)
func qwen3xDraftCandidateBetter(
    lhsValue: Float, lhsID: Int,
    rhsValue: Float, rhsID: Int
) -> Bool {
    let lhsNaN = lhsValue.isNaN
    let rhsNaN = rhsValue.isNaN
    if lhsNaN != rhsNaN { return !lhsNaN }
    if lhsValue != rhsValue { return lhsValue > rhsValue }
    return lhsID < rhsID
}

/// Returns the strongest `count` IDs from `values[0..<limit]` in heap order.
/// Ordering is by descending value, then ascending ID, with NaNs last.
func qwen3xTopDraftCandidateIDs(
    _ values: [Float], count: Int, limit: Int
) -> [Int] {
    precondition(count > 0 && count <= limit && limit <= values.count)
    var heap: [Int] = []
    heap.reserveCapacity(count)

    @inline(__always)
    func worse(_ lhs: Int, than rhs: Int) -> Bool {
        qwen3xDraftCandidateBetter(
            lhsValue: values[rhs], lhsID: rhs,
            rhsValue: values[lhs], rhsID: lhs)
    }

    for id in 0 ..< limit {
        if heap.count < count {
            heap.append(id)
            var child = heap.count - 1
            while child > 0 {
                let parent = (child - 1) / 2
                guard worse(heap[child], than: heap[parent]) else { break }
                heap.swapAt(child, parent)
                child = parent
            }
            continue
        }
        guard
            qwen3xDraftCandidateBetter(
                lhsValue: values[id], lhsID: id,
                rhsValue: values[heap[0]], rhsID: heap[0])
        else { continue }
        heap[0] = id
        var parent = 0
        while true {
            let left = parent * 2 + 1
            guard left < heap.count else { break }
            let right = left + 1
            var worstChild = left
            if right < heap.count && worse(heap[right], than: heap[left]) {
                worstChild = right
            }
            guard worse(heap[worstChild], than: heap[parent]) else { break }
            heap.swapAt(parent, worstChild)
            parent = worstChild
        }
    }
    return heap
}

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

    /// Optional cheaper draft-only lm_head, loaded from
    /// `<model>/draft_lm_head_3bit.safetensors` when
    /// `BUTTER_QWEN_MTP_DRAFT_LMHEAD_3BIT=1` and the sidecar file is
    /// present (see `load(sidecarDirectory:...)`). MLX-format 3-bit
    /// affine g64, requantized offline from the shared 4-bit g64
    /// `lmHead` (`scratch/quantize_draft_lmhead_3bit.py`). Used ONLY by
    /// `draftStep` for the draft-burst logits that pick candidate
    /// tokens — `draftStepMany` (KV lockstep) never computes logits at
    /// all, and the verifier / backbone always read the shared 4-bit
    /// `lmHead`. A draft mistake here only costs acceptance rate (the
    /// verifier still argmaxes the TRUNK, never this head's logits), so
    /// it can never produce incorrect output — only a different
    /// draft/accept boundary. `nil` when the flag is unset or the
    /// sidecar is absent (falls back to the shared `lmHead`).
    let draftLmHead: AnyLinear?

    /// Optional compact 2-bit proposal readout declared by the sidecar.
    /// Its 32 best rows are reranked with the shared 4-bit head before use.
    let declaredDraftLmHead: QuantizedLinear?

    /// Resolved policy for which draft steps use `draftLmHead` (see
    /// `DraftLmHead3BitMode`'s doc comment). `.off` whenever
    /// `draftLmHead` is `nil` regardless of this value — `draftStep`
    /// still gates on `draftLmHead != nil` first.
    let draftLmHeadMode: DraftLmHead3BitMode

    /// The head's OWN KV cache (the draft layer attends over its own
    /// key/value history, independent of every backbone layer cache).
    /// Unbounded eviction; normal model loads use the backbone's configured
    /// context capacity (override with `BUTTER_QWEN_MTP_CACHE=<n>`).
    public let kvCache: KVCache

    public let hidden: Int
    public let vocab: Int
    public let dtype: DType

    static let compactDraftPrefixCount = 98_304
    static let compactDraftControlStart = 248_044
    static let compactDraftRealCount = 98_330
    static let compactDraftPaddedCount = 98_336
    static let draftRerankCandidateCount = 16

    static func mapCompactDraftID(_ id: Int) -> Int {
        precondition(id >= 0 && id < compactDraftRealCount)
        if id < compactDraftPrefixCount { return id }
        return id + compactDraftControlStart - compactDraftPrefixCount
    }

    /// Standalone draft-cache slot capacity. Normal model loads pass the
    /// backbone's configured context capacity. 8192 slots cost ~34 MB at
    /// bf16 (4 KV heads x 256 dims, K+V).
    static let defaultCacheCapacity = 8192

    static func resolvedCacheCapacity(configuredMaxSeq: Int, override: String?) -> Int {
        override.flatMap(Int.init) ?? configuredMaxSeq
    }

    init(
        fc: AnyLinear,
        preFcNormHidden: RMSNorm, preFcNormEmbedding: RMSNorm,
        layer: Qwen35AttentionLayer, mtpNorm: RMSNorm,
        embedTokens: AnyEmbedding, lmHead: AnyLinear,
        draftLmHead: AnyLinear? = nil,
        declaredDraftLmHead: QuantizedLinear? = nil,
        draftLmHeadMode: DraftLmHead3BitMode = .off,
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
        self.draftLmHead = draftLmHead
        self.declaredDraftLmHead = declaredDraftLmHead
        self.draftLmHeadMode = draftLmHeadMode
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
    /// - `draftStepIndex`: 0-based position of THIS call within its
    ///   speculative burst (0 = the first drafted token after the seed;
    ///   see `mtpDecodeLoop`'s `for j in 0 ..< effectiveGamma` — pass
    ///   its `j` straight through). Only consulted under
    ///   `.depth2Plus` mode; callers outside a burst (the KV-lockstep
    ///   re-advance over already-committed tokens, whose logits are
    ///   always discarded) may pass the default.
    /// - `isGreedy`: whether the CURRENT generation run is greedy
    ///   (temperature == 0). Only consulted under `.greedyOnly` mode.
    public func draftStep(
        prevHidden: Tensor, prevToken: Int, position: Int,
        draftStepIndex: Int = 0, isGreedy: Bool = true,
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

        // mtp.norm + lm_head. Draft logits only ever feed candidate
        // selection (see `draftLmHead`'s doc comment) — prefer the
        // cheaper 3-bit draft-only head when loaded AND this step's
        // policy (see `DraftLmHead3BitMode`) says to use it, else the
        // shared 4-bit head the backbone/verifier also use.
        let useDeclaredDraftHead = isGreedy && hasDeclaredDraftRerank
        let useDraftHead: Bool
        if draftLmHead != nil && !useDeclaredDraftHead {
            switch draftLmHeadMode {
            case .off: useDraftHead = false
            case .always: useDraftHead = true
            case .greedyOnly: useDraftHead = isGreedy
            case .depth2Plus: useDraftHead = draftStepIndex >= 1
            }
        } else {
            useDraftHead = false
        }
        let normed = mtpNorm(hOut, on: cmd)
        let logits: Tensor
        if useDeclaredDraftHead {
            logits = declaredDraftLmHead!(normed, on: cmd)
        } else {
            logits = (useDraftHead ? draftLmHead! : lmHead)(normed, on: cmd)
        }
        // Qwen3.8 chains the head's own post-`mtp.norm` hidden into the
        // next draft step. Returning `hOut` here silently degrades every
        // candidate after draft position zero.
        return (hidden: normed, logits: logits)
    }

    private var declaredDraftExactHead: QuantizedLinear? {
        guard let coarse = declaredDraftLmHead,
            coarse.bits == 2,
            coarse.groupSize == 64,
            coarse.weight.shape == [Self.compactDraftPaddedCount, hidden / 16],
            coarse.scales.shape == [Self.compactDraftPaddedCount, hidden / 64],
            coarse.biases.shape == coarse.scales.shape,
            let exact = lmHead.inner as? QuantizedLinear,
            exact.bits == 4,
            exact.groupSize == 64,
            exact.additiveBias == nil,
            exact.weight.shape.count == 2,
            exact.weight.shape[0] > Self.compactDraftControlStart + 25,
            exact.weight.shape[1] == hidden / 8,
            exact.scales.shape[1] == hidden / 64,
            exact.biases.shape == exact.scales.shape
        else { return nil }
        return exact
    }

    var hasDeclaredDraftRerank: Bool { declaredDraftExactHead != nil }

    /// Selects one exact draft token entirely on the GPU. The coarse top-k,
    /// compact-ID mapping, gathered exact head, and final indexed argmax are
    /// ordered on the caller's command buffer.
    func prepareDeclaredDraftSelection(
        coarseLogits: Tensor, normalizedHidden: Tensor,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor? {
        guard let exact = declaredDraftExactHead,
            coarseLogits.elementCount == Self.compactDraftPaddedCount,
            coarseLogits.dtype == .bf16,
            normalizedHidden.elementCount == hidden
        else { return nil }
        let tileCount = 64
        let candidateScratch = Tensor.empty(
            shape: [tileCount * Self.draftRerankCandidateCount],
            dtype: .u32, device: device)
        let compactIDs = Tensor.empty(
            shape: [Self.draftRerankCandidateCount], dtype: .u32, device: device)
        Ops.topKIndicesTiled(
            logits: coarseLogits, candidateScratch: candidateScratch,
            indicesOut: compactIDs, count: Self.compactDraftRealCount,
            tileCount: tileCount, topK: Self.draftRerankCandidateCount,
            on: cmd)
        let fullIDs = Tensor.empty(
            shape: [Self.draftRerankCandidateCount], dtype: .u32, device: device)
        Ops.piecewiseOffsetIndices(
            compactIDs, threshold: Self.compactDraftPrefixCount,
            offset: Self.compactDraftControlStart - Self.compactDraftPrefixCount,
            into: fullIDs, on: cmd)
        let logits = Ops.dequantGemvInt4Gathered(
            weight: exact.weight, scales: exact.scales, biases: exact.biases,
            rowIndices: fullIDs, input: normalizedHidden,
            groupSize: exact.groupSize, on: cmd)
        let selected = Tensor.empty(shape: [1], dtype: .u32, device: device)
        Ops.indexedArgmax(logits, indices: fullIDs, into: selected, on: cmd)
        return selected
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
        cacheCapacity: Int = defaultCacheCapacity,
        eps: Float, dtype: DType, device: Device
    ) throws -> Qwen3xMTPHead {
        let weights = try SafeTensorsBundle(directory: sidecarDirectory, device: device)
        let tensorPrefix = try tensorPrefix(for: weights.allKeys)
        func key(_ relative: String) -> String { tensorPrefix + relative }
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
        if quant == nil, weights.isQuantized(key("fc")) {
            throw Qwen35Error.missingConfig(
                "mtp-sidecar carries quantized tensors but its config.json "
                    + "has no quantization block")
        }

        // Historical namespaced sidecar norms are centered. The standalone
        // Qwen3.8 head's root-level norms are already conventional weights.
        func loadFoldedNorm(_ name: String) throws -> RMSNorm {
            let raw = try weights.tensor(named: name)
            let weight = tensorPrefix == "mtp." ? foldGemmaRMSNormWeight(raw) : raw
            return RMSNorm(weight: weight, eps: eps)
        }

        let fc = try loadLinear(base: key("fc"), in: weights, quantization: quant)
        let preFcNormHidden = try loadFoldedNorm(key("pre_fc_norm_hidden.weight"))
        let preFcNormEmbedding = try loadFoldedNorm(key("pre_fc_norm_embedding.weight"))
        let mtpNorm = try loadFoldedNorm(key("norm.weight"))

        // The single MTP decoder layer - same construction as the
        // backbone's `.attention` case in `Qwen35Hybrid.loadModel`.
        let p = key("layers.0")
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

        let projectionPrecisionIslands: Qwen35ProjectionPrecisionIslands?
        let hasProjectionPrecisionIslands = weights.allKeys.contains {
            $0.hasPrefix("precision_islands.")
        }
        if hasProjectionPrecisionIslands
            && ButterEnv["BUTTER_QWEN_MTP_PRECISION_ISLANDS"] != "0"
        {
            projectionPrecisionIslands = try qwen3xLoadProjectionPrecisionIslands(
                from: weights,
                hidden: hidden,
                qOutputWidth: expectedQRows,
                kvOutputWidth: nKVHeads * headDim,
                dtype: dtype,
                device: device)
        } else {
            projectionPrecisionIslands = nil
        }

        let mixer = Qwen35AttentionMixer(
            qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
            qNorm: qNorm, kNorm: kNorm,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            rotaryDim: rotaryDim, ropeTheta: ropeTheta,
            attnOutputGate: attnOutputGate,
            projectionPrecisionIslands: projectionPrecisionIslands,
            device: device)
        let layer = Qwen35AttentionLayer(
            inputNorm: inputNorm, postNorm: postNorm,
            mixer: mixer,
            ffn: .dense(
                Qwen35DenseMLP(
                    gateProj: gateProj, upProj: upProj, downProj: downProj)),
            hidden: hidden)

        let resolvedCacheCapacity = Self.resolvedCacheCapacity(
            configuredMaxSeq: cacheCapacity,
            override: ButterEnv["BUTTER_QWEN_MTP_CACHE"])
        // `.weight` residency: this cache is built ONCE here (at head
        // load) and lives for the head's entire lifetime — every
        // `generate()` call on the owning model calls `resetCache()`
        // (a logical position reset), never rebuilds this object. It
        // must NOT be in the per-generation residency set that
        // `Device.releaseGenerationCaches()` bulk-drops between arms/
        // generations, or it would silently lose its residency-set
        // fast path the first time any generation on this model ends.
        // See `Device.CacheResidencyKind`'s doc comment.
        let kvCache = KVCache(
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: resolvedCacheCapacity,
            dtype: dtype, device: device, residency: .weight)

        // Optional cheaper draft-only lm_head (see `draftLmHead`'s doc
        // comment). Lives beside the model checkpoint (a sibling of
        // `sidecarDirectory`, which is `<model>/mtp-sidecar`), not
        // inside the mtp-sidecar bundle itself — it's a standalone
        // single-file safetensors, produced offline by
        // `scratch/quantize_draft_lmhead_3bit.py`. Loaded whenever the
        // mode is anything but `.off` — `draftStep` decides per-call
        // whether to actually CONSULT it (see `DraftLmHead3BitMode`).
        // Note the default (env unset) now resolves to `.greedyOnly`, so
        // this load is attempted for EVERY model whose sidecar directory
        // exists, not just opt-in runs — the missing-file warning below
        // is therefore gated on an EXPLICIT env var so the common "no
        // sidecar built for this checkpoint" case stays silent.
        let draftLmHead3BitEnv =
            ButterEnv["BUTTER_QWEN_MTP_DRAFT_LMHEAD_3BIT"]
        let draftLmHeadMode = DraftLmHead3BitMode.parse(draftLmHead3BitEnv)
        var draftLmHead: AnyLinear?
        if draftLmHeadMode != .off {
            let modelRoot = sidecarDirectory.deletingLastPathComponent()
            let draftLmHeadURL = modelRoot.appendingPathComponent(
                "draft_lm_head_3bit.safetensors")
            if FileManager.default.fileExists(atPath: draftLmHeadURL.path) {
                let file = try SafeTensorsFile(url: draftLmHeadURL, device: device)
                guard let weightEntry = file.entries["weight"],
                    let scalesEntry = file.entries["scales"],
                    let biasesEntry = file.entries["biases"]
                else {
                    throw Qwen35Error.missingConfig(
                        "draft_lm_head_3bit.safetensors missing weight/scales/biases")
                }
                let draftGroupSize = 64
                let bits = deriveAffineQuantBits(
                    weightPackedCols: weightEntry.shape[weightEntry.shape.count - 1],
                    scaleCols: scalesEntry.shape[scalesEntry.shape.count - 1],
                    groupSize: draftGroupSize)
                guard bits == 3 else {
                    throw Qwen35Error.unsupportedConfig(
                        "draft_lm_head_3bit.safetensors derived bits=\(bits), expected 3")
                }
                device.markWeightsResident([
                    weightEntry.buffer, scalesEntry.buffer, biasesEntry.buffer,
                ])
                draftLmHead = AnyLinear(
                    QuantizedLinear(
                        weight: try file.tensor(named: "weight"),
                        scales: try file.tensor(named: "scales"),
                        biases: try file.tensor(named: "biases"),
                        bits: bits, groupSize: draftGroupSize))
            } else if let draftLmHead3BitEnv {
                // Only warn when the caller EXPLICITLY asked for the
                // 3-bit draft head (see the note above) — the silent
                // `.greedyOnly` default should never spam stderr for
                // checkpoints that simply never had a sidecar built.
                fputs(
                    "[qwen3x] WARNING: BUTTER_QWEN_MTP_DRAFT_LMHEAD_3BIT="
                        + "\(draftLmHead3BitEnv) but \(draftLmHeadURL.path) does not "
                        + "exist; draft head falls back to the shared 4-bit lm_head\n",
                    stderr)
            }
        }

        var declaredDraftLmHead: QuantizedLinear?
        let declaredDraftKeys = [
            "draft_lm_head.weight",
            "draft_lm_head.scales",
            "draft_lm_head.biases",
        ]
        let declaredDraftKeyCount = declaredDraftKeys.filter(weights.allKeys.contains).count
        if declaredDraftKeyCount != 0 && declaredDraftKeyCount != declaredDraftKeys.count {
            throw Qwen35Error.missingConfig(
                "draft_lm_head must provide weight, scales, and biases")
        }
        if declaredDraftKeyCount == declaredDraftKeys.count
            && ButterEnv["BUTTER_QWEN_MTP_Q2_RERANK"] != "0"
        {
            let weight = try weights.tensor(named: declaredDraftKeys[0])
            let scales = try weights.tensor(named: declaredDraftKeys[1])
            let biases = try weights.tensor(named: declaredDraftKeys[2])
            let groupSize = 64
            let bits = deriveAffineQuantBits(
                weightPackedCols: weight.shape[1],
                scaleCols: scales.shape[1],
                groupSize: groupSize)
            guard bits == 2,
                weight.shape == [compactDraftPaddedCount, hidden / 16],
                scales.shape == [compactDraftPaddedCount, hidden / groupSize],
                biases.shape == scales.shape
            else {
                throw Qwen35Error.unsupportedConfig(
                    "draft_lm_head does not match the compact 2-bit Qwen3.8 contract")
            }
            declaredDraftLmHead = QuantizedLinear(
                weight: weight, scales: scales, biases: biases,
                bits: bits, groupSize: groupSize)
        }

        return Qwen3xMTPHead(
            fc: fc,
            preFcNormHidden: preFcNormHidden,
            preFcNormEmbedding: preFcNormEmbedding,
            layer: layer, mtpNorm: mtpNorm,
            embedTokens: embedTokens, lmHead: lmHead,
            draftLmHead: draftLmHead,
            declaredDraftLmHead: declaredDraftLmHead,
            draftLmHeadMode: draftLmHeadMode,
            kvCache: kvCache,
            hidden: hidden, vocab: vocab, dtype: dtype)
    }

    /// Standalone Qwen3.8 MTP repositories store `fc.weight`,
    /// `layers.0.*`, and friends at the root. Older Butter sidecars retain
    /// the original checkpoint's `mtp.*` namespace. Prefer the namespaced
    /// layout when both are present so existing converted artifacts remain
    /// compatible.
    static func tensorPrefix(for keys: [String]) throws -> String {
        let names = Set(keys)
        if names.contains("mtp.fc.weight") { return "mtp." }
        if names.contains("fc.weight") { return "" }
        throw Qwen35Error.missingConfig(
            "MTP sidecar has neither mtp.fc.weight nor fc.weight")
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

private func qwen3xLoadProjectionPrecisionIslands(
    from weights: SafeTensorsBundle,
    hidden: Int,
    qOutputWidth: Int,
    kvOutputWidth: Int,
    dtype: DType,
    device: Device
) throws -> Qwen35ProjectionPrecisionIslands {
    let qIndexKey = "precision_islands.q.indices"
    let qWeightKey = "precision_islands.q.weight"
    let kIndexKey = "precision_islands.k.indices"
    let kWeightKey = "precision_islands.k.weight"
    let vIndexKey = "precision_islands.v.indices"
    let vWeightKey = "precision_islands.v.weight"
    let requiredKeys = [
        qIndexKey, qWeightKey,
        kIndexKey, kWeightKey,
        vIndexKey, vWeightKey,
    ]
    guard requiredKeys.allSatisfy(weights.allKeys.contains) else {
        throw Qwen35Error.missingConfig(
            "declared projection precision islands require q, k, and v weights and indices")
    }

    let qIndicesRaw = try weights.tensor(named: qIndexKey)
    let kIndicesRaw = try weights.tensor(named: kIndexKey)
    let vIndicesRaw = try weights.tensor(named: vIndexKey)
    let qWeight = try weights.tensor(named: qWeightKey)
    let kWeight = try weights.tensor(named: kWeightKey)
    let vWeight = try weights.tensor(named: vWeightKey)

    func validatedIndices(
        _ tensor: Tensor, count: Int, upperBound: Int,
        requirePermutation: Bool, label: String
    ) throws -> [Int] {
        guard tensor.shape == [count], tensor.dtype == .i32 || tensor.dtype == .u32 else {
            throw Qwen35Error.unsupportedConfig(
                "precision_islands.\(label).indices must be [\(count)] i32 or u32")
        }
        let values: [Int]
        if tensor.dtype == .i32 {
            values = tensor.toArray(as: Int32.self).map(Int.init)
        } else {
            values = tensor.toArray(as: UInt32.self).map(Int.init)
        }
        guard values.allSatisfy({ $0 >= 0 && $0 < upperBound }),
            Set(values).count == count,
            !requirePermutation || count == upperBound
        else {
            throw Qwen35Error.unsupportedConfig(
                "precision_islands.\(label).indices are not a valid unique mapping")
        }
        return values
    }

    let qCount = qIndicesRaw.elementCount
    let qIndices = try validatedIndices(
        qIndicesRaw, count: qCount, upperBound: qOutputWidth,
        requirePermutation: false, label: "q")
    let kIndices = try validatedIndices(
        kIndicesRaw, count: kvOutputWidth, upperBound: kvOutputWidth,
        requirePermutation: true, label: "k")
    let vIndices = try validatedIndices(
        vIndicesRaw, count: kvOutputWidth, upperBound: kvOutputWidth,
        requirePermutation: true, label: "v")

    guard qWeight.shape == [qCount, hidden],
        kWeight.shape == [kvOutputWidth, hidden],
        vWeight.shape == [kvOutputWidth, hidden],
        qWeight.dtype == dtype,
        kWeight.dtype == dtype,
        vWeight.dtype == dtype
    else {
        throw Qwen35Error.unsupportedConfig(
            "precision island projection shapes or dtypes do not match the draft layer")
    }

    let combinedRows = qCount + 2 * kvOutputWidth
    let combinedWeight = Tensor.empty(
        shape: [combinedRows, hidden], dtype: dtype, device: device)
    let rowBytes = hidden * dtype.byteSize
    let destination = combinedWeight.buffer.contents().advanced(by: combinedWeight.offset)
    _ = memcpy(
        destination,
        qWeight.buffer.contents().advanced(by: qWeight.offset),
        qWeight.byteCount)
    for sourceRow in 0 ..< kvOutputWidth {
        _ = memcpy(
            destination.advanced(by: (qCount + kIndices[sourceRow]) * rowBytes),
            kWeight.buffer.contents().advanced(by: kWeight.offset + sourceRow * rowBytes),
            rowBytes)
        _ = memcpy(
            destination.advanced(
                by: (qCount + kvOutputWidth + vIndices[sourceRow]) * rowBytes),
            vWeight.buffer.contents().advanced(by: vWeight.offset + sourceRow * rowBytes),
            rowBytes)
    }

    let qIndexTensor = Tensor.empty(shape: [qCount], dtype: .u32, device: device)
    qIndexTensor.copyIn(from: qIndices.map(UInt32.init))
    device.markWeightsResident([combinedWeight.buffer, qIndexTensor.buffer])
    return Qwen35ProjectionPrecisionIslands(
        projection: AnyLinear(Linear(weight: combinedWeight)),
        qIndices: qIndexTensor,
        qOutputWidth: qOutputWidth,
        kvOutputWidth: kvOutputWidth)
}
