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
// MoELayer — a reusable mixture-of-experts feed-forward block.
//
// An MoE FFN replaces the dense SwiGLU MLP of a transformer layer. The
// substitution is:
//
//   dense:   y = down(silu(gate(x)) * up(x))
//   MoE:     logits = gateLinear(x)                  // [nExperts]
//            (indices, weights) = route(logits)      // top-K
//            y = Σ_{e ∈ indices} weights[e] · expert_e(x)
//            (+ sharedExpert(x) if present)
//
// Each expert is itself a SwiGLU FFN. The router selects the top-K
// experts by gate logit and combines their outputs weighted by the
// (softmax-normalised) gate values. Some architectures additionally run
// an always-on "shared" expert whose output is added unconditionally.
//
// ─── Gating math — the part that breaks silently ─────────────────────
//
// Two checkpoint families disagree on the order of softmax vs top-K:
//
//   .softmaxThenTopK  — Qwen3 / Qwen3.5 MoE. softmax over ALL nExperts
//                       logits first, then pick the top-K of the
//                       softmax probabilities, then (optionally)
//                       re-normalise the K picked probabilities so they
//                       sum to 1.
//   .topKThenSoftmax  — Granite4. pick the top-K raw logits
//                       first, then softmax over just those K. The
//                       result is always normalised by construction.
//
// `normTopKProb` only applies to `.softmaxThenTopK` (Qwen's
// `norm_topk_prob` config flag). `.topKThenSoftmax` is intrinsically
// normalised so the flag is ignored.
//
// ─── Command-buffer contract ─────────────────────────────────────────
//
// The router needs the gate logits on the CPU to do top-K selection.
// Butter's decode path batches a whole token's work onto one command
// buffer, so `MoELayer.decode` commits the caller's `cmd`, waits, reads
// the gate logits back, then runs the selected experts on its own
// private command buffer and returns a fully-resident tensor.
//
// A host model that places `MoELayer` in its `[any DecoderLayer]` stack
// MUST therefore obtain a fresh command buffer after the MoE layer's
// `decode` returns — the passed `cmd` has been committed. This mirrors
// the established `InspectTap.dumpLayerBoundary` pattern (commit + wait
// + hand back a new command buffer). See the integration note at the
// bottom of this file.

import Foundation
import Metal

// ─── Gating mode ─────────────────────────────────────────────────────

/// Order in which the router applies softmax relative to top-K
/// selection. The two production checkpoint families disagree; pick the
/// one matching the architecture being ported.
public enum MoEGatingMode: Sendable {
    /// softmax over all experts, then top-K of the probabilities, then
    /// optional re-normalisation (Qwen3 / Qwen3.5 MoE, `norm_topk_prob`).
    case softmaxThenTopK
    /// top-K of the raw logits, then softmax over just those K
    /// (Granite4). Always normalised.
    case topKThenSoftmax
    /// DeepSeek-V3 / NemotronH / **Hy3** (`hy_v3`) style:
    ///   unbiased = sigmoid(logits)
    ///   biased   = unbiased + expert_bias          (selection only)
    ///   top-K by biased; weights = renorm(unbiased[chosen]) if
    ///   `normTopKProb`, then × `routedScalingFactor`.
    /// Requires non-nil `expertBias`. Confirmed against reference
    /// implementations of hy_v3 (sigmoid scoring + expert-bias selection).
    case sigmoidBiasedTopK
}

// ─── MoERouter — pure-CPU top-K + combine-weight computation ─────────

/// Stateless router: maps a `[nExperts]` gate-logit vector to the
/// top-K expert indices and their (normalised) combine weights.
///
/// Implemented entirely on the CPU — the gate vector is tiny (nExperts
/// 8–128) and the decode loop already has a sync point per token, so a
/// CPU readback of the gate gemv result is the simplest correct path.
public struct MoERouter: Sendable {
    public let nExperts: Int
    public let topK: Int
    public let gatingMode: MoEGatingMode
    /// Re-normalise the K combine weights to sum to 1. Honoured only for
    /// `.softmaxThenTopK` (`.topKThenSoftmax` is normalised by
    /// construction).
    public let normTopKProb: Bool

    /// Optional per-expert additive bias.
    /// - `.softmaxThenTopK` (LFM2-MoE): applied to post-softmax probs
    ///   *before* top-K; biased values also become combine weights.
    /// - `.sigmoidBiasedTopK` (Hy3 / DSv3 / NemotronH): applied to
    ///   sigmoid scores for **selection only**; combine weights stay
    ///   the raw sigmoid values. Required non-nil for that mode.
    public let expertBias: [Float]?

    /// Multiplier applied to final combine weights after renorm.
    /// Hy3 `router_scaling_factor` (2.826), NemotronH
    /// `routed_scaling_factor`, DSv3/v4 equivalents. Default 1.0 leaves
    /// other families unchanged.
    public let routedScalingFactor: Float

    public init(
        nExperts: Int, topK: Int, gatingMode: MoEGatingMode,
        normTopKProb: Bool = true, expertBias: [Float]? = nil,
        routedScalingFactor: Float = 1.0
    ) {
        precondition(nExperts > 0, "MoERouter: nExperts must be positive")
        precondition(
            topK > 0 && topK <= nExperts,
            "MoERouter: topK (\(topK)) must be in 1...nExperts (\(nExperts))")
        if let bias = expertBias {
            precondition(
                bias.count == nExperts,
                "MoERouter: expertBias has \(bias.count) entries, "
                    + "nExperts is \(nExperts)")
        }
        if gatingMode == .sigmoidBiasedTopK {
            precondition(
                expertBias != nil,
                "MoERouter: .sigmoidBiasedTopK requires expertBias "
                    + "(Hy3 / DSv3 e_score_correction_bias)")
        }
        self.nExperts = nExperts
        self.topK = topK
        self.gatingMode = gatingMode
        self.normTopKProb = normTopKProb
        self.expertBias = expertBias
        self.routedScalingFactor = routedScalingFactor
    }

    /// Routing result: `indices[i]` is the i-th selected expert,
    /// `weights[i]` its combine weight. Both arrays have `topK` entries
    /// and are index-aligned.
    public struct Routing: Equatable, Sendable {
        public let indices: [Int]
        public let weights: [Float]
    }

    /// Compute the top-K expert indices + combine weights for one gate
    /// vector. `logits.count` must equal `nExperts`.
    ///
    /// Tie-breaking on equal logits: the smaller expert index wins —
    /// matches NumPy / PyTorch / MLX `argpartition` + `argmax`
    /// semantics, so output stays deterministic across runs.
    public func route(logits: [Float]) -> Routing {
        precondition(
            logits.count == nExperts,
            "MoERouter.route: logits.count \(logits.count) ≠ nExperts \(nExperts)")

        switch gatingMode {
        case .softmaxThenTopK where normTopKProb && expertBias == nil:
            // Fast path: `softmaxThenTopK + normTopKProb=true` produces
            // the same weights as `.topKThenSoftmax`. Proof:
            //   softmax(L)[i] = exp(L[i] - M) / Z, M = max(L), Z = Σ
            //   top-K of softmax probs by value desc = top-K of raw
            //     logits (softmax is monotonic on selection)
            //   weights_pre[k] = exp(L[idx_k] - M) / Z
            //   renorm: weights[k] = weights_pre[k] / Σ_k weights_pre
            //     = exp(L[idx_k]) / Σ_k exp(L[idx_k])
            //     = softmax over just the K picked logits
            // Skips the full softmax over `nExperts=256` (~500 ops
            // including 256 exp + sum reduction + 256 div). Saves
            // ~40 K host ops / token on Qwen3.6-A3B (40 MoE layers ×
            // ~1000 ops per route saved). Qwen3 / Qwen3.5 / Qwen3.6 MoE
            // all hit this branch (`norm_topk_prob: true` in every
            // shipped config).
            //
            // REQUIRES `expertBias == nil`: the proof's "top-K of raw
            // logits == top-K of softmax probs" step breaks once an
            // additive per-expert bias is applied AFTER softmax (the
            // bias is not monotonic in the raw logit). LFM2-MoE supplies
            // an `expert_bias`, so it falls through to the general
            // `.softmaxThenTopK` path below, which selects on
            // `softmax(logits) + bias`.
            let idx = Self.topKIndices(logits, k: topK)
            let weights = Self.softmax(idx.map { logits[$0] })
            return Routing(
                indices: idx,
                weights: Self.applyScale(weights, scale: routedScalingFactor))

        case .softmaxThenTopK:
            // `normTopKProb=false` case — semantically distinct
            // (unnormalised softmax probs as weights). Must walk the
            // full softmax over all nExperts.
            let probs = Self.softmax(logits)
            // Optional per-expert additive bias (LFM2-MoE `expert_bias`).
            // With no bias `gated == probs`, so other softmaxThenTopK
            // families are byte-identical.
            let gated: [Float]
            if let bias = expertBias {
                gated = zip(probs, bias).map { $0 + $1 }
            } else {
                gated = probs
            }
            let idx = Self.topKIndices(gated, k: topK)
            var weights = idx.map { gated[$0] }
            // Optional re-normalisation of the K picked weights.
            if normTopKProb {
                let sum = weights.reduce(0, +)
                if sum > 0 { weights = weights.map { $0 / sum } }
            }
            return Routing(
                indices: idx,
                weights: Self.applyScale(weights, scale: routedScalingFactor))

        case .topKThenSoftmax:
            // Top-K of raw logits, softmax over just the K picked —
            // always normalised by construction so `normTopKProb`
            // doesn't apply.
            let idx = Self.topKIndices(logits, k: topK)
            let weights = Self.softmax(idx.map { logits[$0] })
            return Routing(
                indices: idx,
                weights: Self.applyScale(weights, scale: routedScalingFactor))

        case .sigmoidBiasedTopK:
            // Path B — Hy3 / NemotronH / DeepSeek-V3 noaux.
            // Selection on (sigmoid + bias); weights from raw sigmoid.
            let bias = expertBias!  // preconditioned non-nil at init
            var unbiased = [Float](repeating: 0, count: nExperts)
            var biased = [Float](repeating: 0, count: nExperts)
            for i in 0 ..< nExperts {
                let s = 1.0 / (1.0 + Foundation.exp(-logits[i]))
                unbiased[i] = s
                biased[i] = s + bias[i]
            }
            let idx = Self.topKIndices(biased, k: topK)
            var weights = idx.map { unbiased[$0] }
            if normTopKProb {
                let sum = weights.reduce(0, +) + 1e-20
                weights = weights.map { $0 / sum }
            }
            return Routing(
                indices: idx,
                weights: Self.applyScale(weights, scale: routedScalingFactor))
        }
    }

    @inline(__always)
    private static func applyScale(_ w: [Float], scale: Float) -> [Float] {
        if scale == 1.0 { return w }
        return w.map { $0 * scale }
    }

    // ─── CPU helpers ─────────────────────────────────────────────────

    /// Numerically-stable softmax over a small vector.
    static func softmax(_ x: [Float]) -> [Float] {
        guard let maxV = x.max() else { return [] }
        let exps = x.map { Foundation.exp($0 - maxV) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else {
            // Degenerate (all -inf): uniform fallback.
            return [Float](repeating: 1 / Float(x.count), count: x.count)
        }
        return exps.map { $0 / sum }
    }

    /// Indices of the K largest values, ordered largest-first. Ties
    /// resolved by smaller index (matches argpartition semantics).
    static func topKIndices(_ x: [Float], k: Int) -> [Int] {
        // Partial-sort: maintain a k-sized min-heap of (value, index)
        // pairs, ordered by (value asc, index desc) so the smallest
        // value with the largest index is at the top (i.e. easiest to
        // evict). At the end, drain the heap and reverse — gives the
        // top-K ordered by (value desc, index asc), matching the prior
        // full-sort semantics.
        //
        // Complexity: O(n log k) vs the prior `Array.sorted` O(n log n).
        // At Qwen3.6-A3B (nExperts=256, k=8) this is ~280 ops vs ~2048
        // ops per call → ~7× host CPU per route × 40 MoE layers per
        // decode token ≈ 5-7 ms / token saved. Stale comment claimed
        // "nExperts ≤ 128 so a full sort is cheaper" — at 256 it
        // already wasn't, and partial-sort is the right structure
        // regardless.
        precondition(k > 0 && k <= x.count, "topKIndices: k must be in 1...n")
        var heap: [(value: Float, index: Int)] = []
        heap.reserveCapacity(k)
        // Min-heap ordering: smallest value with largest index at root.
        // Returns true when `a` is "smaller" (closer to root / first to
        // evict). For tie on value, the entry with the LARGER index is
        // smaller (we want smaller indices to win ties on the final
        // sort, so the larger-index entry is the easier evict).
        @inline(__always) func less(_ a: (Float, Int), _ b: (Float, Int)) -> Bool {
            if a.0 != b.0 { return a.0 < b.0 }
            return a.1 > b.1
        }
        @inline(__always) func siftUp(_ start: Int) {
            var i = start
            while i > 0 {
                let parent = (i - 1) >> 1
                if less(heap[i], heap[parent]) {
                    heap.swapAt(i, parent)
                    i = parent
                } else {
                    return
                }
            }
        }
        @inline(__always) func siftDown(_ start: Int) {
            var i = start
            let n = heap.count
            while true {
                let l = 2 * i + 1
                let r = 2 * i + 2
                var smallest = i
                if l < n && less(heap[l], heap[smallest]) { smallest = l }
                if r < n && less(heap[r], heap[smallest]) { smallest = r }
                if smallest == i { return }
                heap.swapAt(i, smallest)
                i = smallest
            }
        }
        for i in 0 ..< x.count {
            let entry = (x[i], i)
            if heap.count < k {
                heap.append(entry)
                siftUp(heap.count - 1)
            } else if less(heap[0], entry) {
                // Heap root is the easiest evict; the new entry beats
                // it on (value desc, index asc).
                heap[0] = entry
                siftDown(0)
            }
        }
        // Drain — `heap` is min-ordered, but the public contract returns
        // largest-first (`value desc, index asc`). Sort the final k
        // entries by the same (value desc, index asc) criterion.
        heap.sort { a, b in
            if a.value != b.value { return a.value > b.value }
            return a.index < b.index
        }
        return heap.map { $0.index }
    }
}

// ─── MoELayer — DecoderLayer-conforming MoE feed-forward block ───────

/// A mixture-of-experts feed-forward layer. Holds the gate projection,
/// the per-expert SwiGLU weights, and an optional always-on shared
/// expert. Conforms to `DecoderLayer` so a hybrid model can place it in
/// a heterogeneous `[any DecoderLayer]` stack; it carries no per-token
/// state and is handed a `StatelessLayerCache`.
///
/// `decode` runs: gate gemv → CPU route → per-expert SwiGLU on the
/// selected experts → combine. See the command-buffer contract note in
/// the file header — `decode` commits the passed `cmd`.
///
/// Threading contract: instances are NOT safe for concurrent calls.
/// `decode` / `decodeMany` mutate per-instance scratch buffers
/// (`routerIndicesScratch`, `accumulatorScratch`, `expertGScratches`,
/// `dequantizedGateCache`, …) without internal synchronisation. Callers
/// must serialise inference on a given instance — typical model loops
/// already do this since a token can't fan out across MoE layers in
/// parallel. The class is `final` for closed-world dispatch, not for
/// concurrency.
public final class MoELayer: Module, DecoderLayer {
    /// Router projection: hidden → nExperts logits.
    public let gate: AnyLinear
    /// Per-expert SwiGLU projections. Each array has `nExperts` entries;
    /// `gateProj[e]` / `upProj[e]` / `downProj[e]` are expert `e`'s
    /// projections.
    public let gateProj: [AnyLinear]
    public let upProj: [AnyLinear]
    public let downProj: [AnyLinear]
    /// Optional always-on shared expert (Granite4-style). Its
    /// output is added to the routed combination unconditionally.
    public let sharedGateProj: AnyLinear?
    public let sharedUpProj: AnyLinear?
    public let sharedDownProj: AnyLinear?

    public let router: MoERouter
    public let hidden: Int

    /// Stacked-expert weight handles for the batched gather BGEMM fast
    /// path. When set, `decode` dispatches `iron_moe_gather_qmm_mma_int4_bm16`
    /// instead of running `topK` sequential per-expert SwiGLU triplets —
    /// 3 kernel launches per token instead of `3 * topK` (24 → 3 at
    /// `topK=8`). Populated by the host model's MoE builder when (a) the
    /// checkpoint is mlx int4-quantized, (b) the stacked weight tensors
    /// are intact (not sliced away by the per-expert wrapper path), and
    /// (c) the moeIntermediate / hidden shapes satisfy the bm16 kernel's
    /// `N % 32 == 0` and `K % 32 == 0` tile contract. Left nil for
    /// everyone else — the legacy serial-expert loop still runs.
    public struct StackedInt4Experts: Sendable {
        /// `[numExperts, moeIntermediate, hidden/8]` u32 packed.
        public let gateWeight: Tensor
        /// `[numExperts, moeIntermediate, hidden/groupSize]` in `dtype`.
        public let gateScales: Tensor
        public let gateBiases: Tensor
        /// `[numExperts, moeIntermediate, hidden/8]` u32 packed.
        public let upWeight: Tensor
        public let upScales: Tensor
        public let upBiases: Tensor
        /// `[numExperts, hidden, moeIntermediate/8]` u32 packed.
        public let downWeight: Tensor
        public let downScales: Tensor
        public let downBiases: Tensor
        public let numExperts: Int
        public let moeIntermediate: Int
        public let hidden: Int
        public let groupSize: Int
        /// Activation dtype the scales / biases / activations all use.
        public let dtype: DType

        public init(
            gateWeight: Tensor, gateScales: Tensor, gateBiases: Tensor,
            upWeight: Tensor, upScales: Tensor, upBiases: Tensor,
            downWeight: Tensor, downScales: Tensor, downBiases: Tensor,
            numExperts: Int, moeIntermediate: Int, hidden: Int,
            groupSize: Int, dtype: DType
        ) {
            self.gateWeight = gateWeight
            self.gateScales = gateScales
            self.gateBiases = gateBiases
            self.upWeight = upWeight
            self.upScales = upScales
            self.upBiases = upBiases
            self.downWeight = downWeight
            self.downScales = downScales
            self.downBiases = downBiases
            self.numExperts = numExperts
            self.moeIntermediate = moeIntermediate
            self.hidden = hidden
            self.groupSize = groupSize
            self.dtype = dtype
        }
    }
    public let stackedInt4Experts: StackedInt4Experts?

    /// Env-flag cache — read once at init so the decode loop doesn't
    /// pay 3 × `ProcessInfo.processInfo.environment[...]` dictionary
    /// lookups per MoE layer per token. At Qwen3.6-A3B that's 120
    /// lookups / token saved (40 layers × 3 envs). Each lookup is
    /// ~100 ns on Foundation, so the total is sub-noise (~12 µs / token)
    /// — shipped for cleanliness more than the µbench delta.
    public let enableBGEMM: Bool
    public let useBm8Env: Bool
    public let useM1Env: Bool
    /// Additional env-flag caches.
    let noDequantGate: Bool
    let gpuRouterEnabled: Bool
    let forceBGEMMAtT1: Bool
    let noBGEMMBm64: Bool
    let bm64Env: Bool
    /// F-85 expert-aligned tile-plan MoE gather GEMM gate. Raw env string
    /// cached at init (same reasoning as the other env-flag caches above);
    /// `tilePlanEnabled` below is the single place that interprets it.
    let tilePlanEnvRaw: String?
    /// F-85 coop-core MoE gather GEMM gate (BM=32 tile plan, `iron_qmm_coop`'s
    /// inner loop). Raw env string cached at init; `coopEnabled` below is
    /// the single place that interprets it.
    let coopEnvRaw: String?
    /// F-85 two-expert tile-pairing lever gate (`iron_moe_gather_qmm_coop_paired`,
    /// only takes effect where `coopEnabled` already holds). Raw env
    /// string cached at init; `pairedEnabled` below is the single place
    /// that interprets it.
    let pairedEnvRaw: String?
    /// F-85 tile-pairing mTotal threshold override. Raw env string cached
    /// at init; `pairedMaxMTotal` below is the single place that
    /// interprets it.
    let pairedMaxMTotalEnvRaw: String?
    /// F-85 paired-GEMM indirect-dispatch gate (only takes effect where
    /// `useCoopPaired` already holds). Raw env string cached at init;
    /// `pairedIndirectEnabled` below is the single place that interprets
    /// it.
    let pairedIndirectEnvRaw: String?
    /// F-85 idle-tile-cost indirect-dispatch gate for the PRODUCTION coop
    /// path (`iron_moe_gather_qmm_coop` itself, both the plain
    /// `useCoopTilePlan` dispatch and the paired lever's "own" dispatch -
    /// see `coopIndirectEnabled` below). Raw env string cached at init.
    let coopIndirectEnvRaw: String?
    /// Bucketed counting-sort gate for `iron_moe_sort_plan` (F-85 prefill
    /// gate). Raw env string cached at init; `sortPlanCountingEnabled`
    /// below is the single place that interprets it.
    let sortPlanCountingEnvRaw: String?
    /// Parallelized tile-plan builder gate (F-85 small-M follow-up). Raw
    /// env string cached at init; `parallelPlanEnabled` below is the
    /// single place that interprets it.
    let parallelPlanEnvRaw: String?

    /// Lazy dequant of the gate weight + cached output. Skipped when
    /// `gate.inner` is not a 8-bit QuantizedLinear or env
    /// `BUTTER_MOE_NO_DEQUANT_GATE=1` is set.
    private var dequantizedGateCache: Tensor?
    private var dequantizedGateCacheKN: Tensor?
    private var dequantizedGateAttempted = false
    private var gateLogitsScratch: Tensor?

    /// Cached scratches for the per-expert weighted-sum loop at
    /// decode T=1.
    private var accumulatorScratch: Tensor?
    private var topKScalarsBuf: MTLBuffer?
    /// Cached per-expert g/u/inner/down outputs.
    private var expertGScratches: [Tensor] = []
    private var expertUScratches: [Tensor] = []
    private var expertInnerScratches: [Tensor] = []
    private var expertOutScratches: [Tensor] = []
    /// GPU MoE router scratches — written once per layer per token by
    /// `iron_moe_router_topk`, then consumed by the per-slot indexed qmms
    /// and the scalarFMAChain8.
    private var routerIndicesScratch: Tensor?
    private var routerWeightsScratch: Tensor?

    /// F-85 tile-pairing plan buffers (see `useCoopPaired` in
    /// `decodeMany`) - allocated ONCE, lazily, on this instance's first
    /// paired-path call, then reused for every subsequent call. Fixed
    /// size: `moePairedTilePlanCapacity(nExperts:)` depends only on
    /// `nExperts` (a per-instance constant via `router.nExperts`), never
    /// on `mTotal`, so unlike the tile-plan-capacity scratches above
    /// there is no resize case to handle - first-alloc size is the only
    /// size this instance will ever need. Follows the same per-instance
    /// cached-scratch pattern as `routerIndicesScratch` / `accumulatorScratch`
    /// above; same threading contract (see the class doc comment) - a
    /// given instance's calls are serialised by the caller, so no lock
    /// is needed here either.
    ///
    /// Safety of reuse ACROSS forward calls on one instance: Metal
    /// enables automatic hazard tracking by default for plain
    /// `device.makeBuffer` allocations (not heap/untracked), and this
    /// codebase's whole per-layer command-buffer chaining already
    /// depends on command buffers committed to one `MTLCommandQueue`
    /// executing in commit order with no explicit inter-buffer fence
    /// (see `Qwen3xText._forwardManyBatched`'s `workCmd` rotation - a
    /// fresh command buffer is made and committed per MoE layer, with no
    /// `waitUntilCompleted` between layers outside of `BUTTER_PROFILE_LAYERS`).
    /// The paired-plan builder dispatch (write) for call N+1 and the
    /// paired GEMM dispatch (read) for call N therefore only ever touch
    /// this buffer from commands that Metal schedules and executes in
    /// the SAME order they were committed - the builder's write cannot
    /// run ahead of the previous call's GEMM read of the same buffer.
    /// This is the identical assumption the rest of the forward pass
    /// already relies on to get correct (not just fast) results without
    /// a host-side wait between every layer, so reusing these six
    /// buffers introduces no NEW cross-call hazard beyond what every
    /// other per-layer buffer in this file already accepts.
    ///
    /// Buffers are never shared BETWEEN MoELayer instances (each
    /// transformer layer owns its own `MoELayer`, hence its own copy of
    /// these six buffers) - so there is no cross-LAYER aliasing question
    /// to reason about at all, only same-layer call-N vs call-(N+1)
    /// reuse, per the above.
    private var pairedTileExpertABuf: MTLBuffer?
    private var pairedTileRowStartABuf: MTLBuffer?
    private var pairedTileRowCountABuf: MTLBuffer?
    private var pairedTileExpertBBuf: MTLBuffer?
    private var pairedTileRowStartBBuf: MTLBuffer?
    private var pairedTileRowCountBBuf: MTLBuffer?
    private var pairedTileBufCapacity = 0

    /// F-85 indirect-dispatch args for the paired GEMM (see
    /// `Ops.moeGatherDequantGemmInt4CoopTilePlanPairedIndirect`) - the
    /// root-cause fix for `pairedEnabled`'s worst-case-capacity regression
    /// (see that property's doc). Layout per buffer: 3 x u32 =
    /// `[n_out_tiles, pair_count, 1]`, the exact
    /// `MTLDispatchThreadgroupsIndirectArguments` triple. TWO buffers, not
    /// one, because the gate/up leg (`n_out = moeIntermediate`) and the
    /// down leg (`n_out = hidden`) need independent word-0 values; both
    /// receive the SAME device-computed `pair_count` (word 1) every call,
    /// since the real paired-tile count does not depend on which leg is
    /// dispatching. Same lazy-alloc-once, reuse-across-calls pattern and
    /// same cross-call GPU-ordering safety argument as
    /// `pairedTileExpertABuf` above - word 1 is the only word a GPU
    /// dispatch ever writes post-alloc, and that write (this call's
    /// builder) always precedes the read (this call's paired GEMM) in the
    /// same command-buffer-chain order. Words 0/2 are written ONCE, from
    /// the CPU, at allocation time (see `pairedIndirectArgsBuffers`) -
    /// `moeIntermediate`/`hidden` never change after init, so there is
    /// never a need to rewrite them.
    private var pairedIndirectArgsGateUpBuf: MTLBuffer?
    private var pairedIndirectArgsDownBuf: MTLBuffer?

    /// F-85 idle-tile-cost indirect-dispatch args for the PRODUCTION coop
    /// GEMM (`iron_moe_gather_qmm_coop` - see
    /// `Ops.moeGatherDequantGemmInt4CoopTilePlanIndirect`). Same layout
    /// and same lazy-alloc-once / cross-call GPU-ordering safety argument
    /// as `pairedIndirectArgsGateUpBuf` above: `[n_out_tiles, real_tile_count, 1]`,
    /// words 0/2 written once from the CPU, word 1 device-written every
    /// call before this call's GEMM read of the same buffer (same command-
    /// buffer-chain ordering the rest of this file already depends on).
    ///
    /// Shared between the two call sites that feed the SAME production
    /// kernel with a BM=32 tile plan: the plain `useCoopTilePlan` path
    /// (not paired) and the paired lever's "own" dispatch (`useCoopPaired`).
    /// Safe to share because exactly one of those two branches executes
    /// per `decodeMany` call on a given instance (mutually exclusive on
    /// `useCoopPaired`), so there is never a same-call double-write.
    private var coopIndirectArgsGateUpBuf: MTLBuffer?
    private var coopIndirectArgsDownBuf: MTLBuffer?

    /// F-85 tile-plan buffers for the `useTilePlan` gather GEMM (plain
    /// bm16 tileplan default AND the coop BM=32 default that supersedes
    /// it, see `useTilePlan`/`useCoopTilePlan` in `decodeMany`) - allocated
    /// lazily, per instance, then reused/grown across calls, unlike
    /// `pairedTileExpertABuf` above this was still a fresh
    /// `device.makeBuffer` triple every layer, every forward call on the
    /// DEFAULT path (tilePlanEnabled/coopEnabled both default ON) until
    /// this cache was added - the paired lever's own buffers got the
    /// reuse treatment in the same round that shipped this gate, but the
    /// plain/coop tile-plan buffers this gate always uses first were
    /// missed.
    ///
    /// Unlike `pairedTileBufCapacity` (constant per instance - depends
    /// only on `nExperts`), this capacity depends on `mTotal`, which
    /// varies per prefill chunk (every chunk but the last is the same
    /// configured step size, so in practice this grows once then holds
    /// steady). Grow-only, matching `Ops.identityRowsBuffer`'s pattern:
    /// a call needing less than the cached capacity just uses a
    /// `[requestedCapacity]`-shaped view into the larger buffer (the
    /// dispatch grid and the zero-fill below both use the PER-CALL
    /// capacity, never the buffer's full backing length), so holding
    /// onto the largest capacity seen so far is always safe and never
    /// wastes a downstream dispatch on stale tail data.
    ///
    /// Same cross-call GPU-ordering safety argument as
    /// `pairedTileExpertABuf` above (this file's whole per-layer
    /// command-buffer chaining already depends on committed-order
    /// execution with no inter-buffer fence), and the zero-fill every
    /// call is still required for the same reason: a previous call's
    /// tail past THIS call's real tile count would otherwise leak stale
    /// `tile_row_count` values the GEMM kernel needs to read as zero to
    /// treat that tile as inert padding.
    private var tilePlanTileExpertBuf: MTLBuffer?
    private var tilePlanTileRowStartBuf: MTLBuffer?
    private var tilePlanTileRowCountBuf: MTLBuffer?
    private var tilePlanBufCapacity = 0

    /// - gate: hidden → nExperts router projection.
    /// - gateProj/upProj/downProj: `nExperts`-long arrays of per-expert
    ///   SwiGLU projections, index-aligned with the expert id.
    /// - sharedGate/Up/DownProj: optional shared-expert SwiGLU; pass all
    ///   three or none.
    /// - router: the top-K + gating-math configuration.
    /// - stackedInt4Experts: optional batched-BGEMM fast path. Per-expert
    ///   arrays above remain the source of truth for `parameters()`
    ///   (checkpoint binding) and the fallback decode path.
    public init(
        gate: AnyLinear,
        gateProj: [AnyLinear], upProj: [AnyLinear], downProj: [AnyLinear],
        sharedGateProj: AnyLinear? = nil,
        sharedUpProj: AnyLinear? = nil,
        sharedDownProj: AnyLinear? = nil,
        router: MoERouter, hidden: Int,
        stackedInt4Experts: StackedInt4Experts? = nil
    ) {
        precondition(
            gateProj.count == router.nExperts,
            "MoELayer: gateProj has \(gateProj.count) experts, router expects \(router.nExperts)")
        precondition(
            upProj.count == router.nExperts,
            "MoELayer: upProj has \(upProj.count) experts, router expects \(router.nExperts)")
        precondition(
            downProj.count == router.nExperts,
            "MoELayer: downProj has \(downProj.count) experts, router expects \(router.nExperts)")
        let sharedCount = [sharedGateProj, sharedUpProj, sharedDownProj]
            .filter { $0 != nil }.count
        precondition(
            sharedCount == 0 || sharedCount == 3,
            "MoELayer: shared expert needs all three of gate/up/down or none")
        self.gate = gate
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
        self.sharedGateProj = sharedGateProj
        self.sharedUpProj = sharedUpProj
        self.sharedDownProj = sharedDownProj
        self.router = router
        self.hidden = hidden
        if let s = stackedInt4Experts {
            precondition(
                s.numExperts == router.nExperts,
                "MoELayer: stackedInt4Experts.numExperts \(s.numExperts) ≠ router.nExperts \(router.nExperts)"
            )
            precondition(
                s.hidden == hidden,
                "MoELayer: stackedInt4Experts.hidden \(s.hidden) ≠ MoELayer.hidden \(hidden)")
            precondition(
                s.moeIntermediate % 32 == 0 && hidden % 32 == 0,
                "MoELayer: stackedInt4Experts shape (moeIntermediate=\(s.moeIntermediate), hidden=\(hidden)) violates bm16 N%32 / K%32 tile contract"
            )
        }
        self.stackedInt4Experts = stackedInt4Experts
        let env = ProcessInfo.processInfo.environment
        self.enableBGEMM = env["BUTTER_MOE_BGEMM"] != nil
        self.useBm8Env = env["BUTTER_MOE_BGEMM_BM8"] != nil
        self.useM1Env = env["BUTTER_MOE_M1"] != nil
        self.noDequantGate = env["BUTTER_MOE_NO_DEQUANT_GATE"] != nil
        // Default GPU router ON — end-to-end correctness is pinned
        // across bf16/f16/f32 by the integration suite. Opt out with
        // `BUTTER_MOE_GPU_ROUTER=0`. The win: 40 ×
        // `waitUntilCompleted` per decode token → 0.
        self.gpuRouterEnabled = env["BUTTER_MOE_GPU_ROUTER"] != "0"
        self.forceBGEMMAtT1 = env["BUTTER_MOE_BGEMM_FORCE_T1"] != nil
        self.noBGEMMBm64 = env["BUTTER_MOE_BGEMM_NO_BM64"] != nil
        self.bm64Env = env["BUTTER_MOE_BGEMM_BM64"] != nil
        self.tilePlanEnvRaw = env["BUTTER_MOE_TILEPLAN"]
        self.coopEnvRaw = env["BUTTER_MOE_COOP"]
        self.pairedEnvRaw = env["BUTTER_MOE_PAIRED"]
        self.pairedMaxMTotalEnvRaw = env["BUTTER_MOE_PAIRED_MAX_MTOTAL"]
        self.pairedIndirectEnvRaw = env["BUTTER_MOE_PAIRED_INDIRECT"]
        self.coopIndirectEnvRaw = env["BUTTER_MOE_COOP_INDIRECT"]
        self.sortPlanCountingEnvRaw = env["BUTTER_MOE_COUNTING_SORT"]
        self.parallelPlanEnvRaw = env["BUTTER_MOE_PARALLEL_PLAN"]
    }

    /// F-85 expert-aligned tile-plan gate. Default ON as of the wiring
    /// pass's quality + speed gates: `--dump-prefill-logits` cosine 1.0 /
    /// top1 / top5 set+order match on 3 prompts each at T=512/1024/4096/
    /// 16384 (bit-exact, not just close - max_abs_dlogit 0.0 everywhere),
    /// greedy 32-token generation token-for-token identical to bm16 at
    /// every T tested, and an order-balanced interleaved e2e prefill
    /// speed gate showing a consistent 1.02-1.03x win at every T from 512
    /// through 32768 (median of 3-4 trials per config per T, no high-T
    /// regression). Escape hatch: `BUTTER_MOE_TILEPLAN=0` reverts to bm16.
    /// Only wired into the DEVPLAN2 device-sorted-plan path
    /// (`prefillChainsOnCallerCmd`); other routing paths keep bm16
    /// regardless of this flag (see `decodeMany`).
    public var tilePlanEnabled: Bool {
        tilePlanEnvRaw != "0"
    }

    /// F-85 bucketed counting-sort gate for `iron_moe_sort_plan`
    /// (`Ops.moeSortPlanCounting`, `moe_sort_plan_counting.rs`). Replaces
    /// the original kernel's O(mTotal^2) per-thread linear scan with a
    /// three-pass O(mTotal * nExperts) histogram -> prefix-sum -> scatter
    /// pipeline, same stable order (`--dump-prefill-logits` byte-identical
    /// at every T tested, deterministic across repeat dispatch, checked
    /// against the original kernel's own GPU output directly, not just a
    /// CPU oracle). Isolated kernel bench (Zipf-seeded routing draws,
    /// same shape as production) shows the win growing with mTotal as the
    /// asymptotics predict: ~6-9x at mTotal 4096/8192, ~10-15x at 16384,
    /// ~38x at 32768. Default ON. Escape hatch: `BUTTER_MOE_COUNTING_SORT=0`
    /// reverts to the original kernel. Only wired into the DEVPLAN2
    /// device-sorted-plan path (`useDevPlan2`) - the only path that calls
    /// `iron_moe_sort_plan` at all.
    public var sortPlanCountingEnabled: Bool {
        sortPlanCountingEnvRaw != "0"
    }

    /// F-85 parallelized tile-plan builder gate (small-M follow-up to the
    /// idle-tile-cost/counting-sort round). `iron_moe_build_tile_plan` /
    /// `_bm32` / `_bm32_own` each dispatch ONE threadgroup for the whole
    /// plan build - every expert's binary search, the prefix sum, and
    /// every tile emission all run on a single GPU core, a per-layer fixed
    /// cost that does not shrink with `mTotal` and was found
    /// disproportionately expensive at low prefill `T`. The parallel
    /// replacements (`Ops.moeBuildTilePlanParallel`/`Bm32Parallel`/
    /// `Bm32OwnParallel`, see `moe_tile_plan_expert_counts.rs`'s module
    /// doc) split the per-expert row-range search out to `nExperts`
    /// threadgroups (one per expert, running concurrently across the
    /// GPU's cores) followed by a small single-threadgroup prefix-sum +
    /// emission pass - same output format/contract, byte-exact against the
    /// original builders across the full fixture set (dual CPU-oracle +
    /// live-original-kernel checks). Default ON. Escape hatch:
    /// `BUTTER_MOE_PARALLEL_PLAN=0` reverts to the single-threadgroup
    /// builders. Applies wherever `useTilePlan` holds - the plain bm16
    /// path, the default `useCoopTilePlan` path, and the paired lever's
    /// "own" dispatch (see `decodeMany`).
    public var parallelPlanEnabled: Bool {
        parallelPlanEnvRaw != "0"
    }

    /// F-85 coop-core MoE gather GEMM gate (BM=32 tile plan over
    /// `iron_moe_gather_qmm_coop`, the gather counterpart of the dense
    /// `iron_qmm_coop` core). Default ON as of the wiring pass's quality +
    /// speed gates, same evidence bar as `tilePlanEnabled` above:
    /// isolated kernel bench vs the BM=16 tileplan incumbent shows +31-98%
    /// GB/s across mTotal 4096/16384/32768, both legs, 3 Zipf-seeded
    /// routing draws (see `moe_gather_qmm_coop.rs`'s bench sweep and its
    /// `moe_mpp_tileplan.rs` isolated-comparison counterpart);
    /// `--dump-prefill-logits` is bit-exact (cosine 1.0, max_abs_dlogit
    /// 0.0, top1/top5 set+order match) at T=512/1024/4096/16384; decode
    /// (T=1) tok/s unchanged (expected - this path only engages on the
    /// DEVPLAN2 batched-prefill route); and an order-balanced interleaved
    /// e2e prefill speed gate (base vs coop, 2 trials/T) shows a consistent
    /// 14-23% win at every T from 512 through 32768 with no regression at
    /// any point already at or above the reference bar. Escape hatch:
    /// `BUTTER_MOE_COOP=0` reverts to the BM=16 tileplan default. Only
    /// applies where `useTilePlan` already holds (DEVPLAN2 path) and the
    /// coop core's own shape gate passes (`nOut % 64 == 0` on both legs,
    /// `groupSize == 64`) - see `useCoopTilePlan` in `decodeMany`.
    ///
    /// Auto-default gates on `Ops.mppAutoCapable` (Apple GPU Family 9+ /
    /// macOS 26+) - the same hardware requirement `dispatchQmmMma`'s coop
    /// tier checks for the dense GEMM, since `iron_moe_gather_qmm_coop`
    /// wraps the identical `matmul2d` cooperative-tensor core. Unlike the
    /// dense path (which falls back to a cheaper simdgroup tier when
    /// incapable), this kernel has no such fallback once dispatched, so
    /// the check has to live here rather than inside the Ops wrapper.
    /// `BUTTER_MOE_COOP=1` force-enables past this check (A/B only, same
    /// "does not bypass the hardware requirement" caveat as
    /// `BUTTER_QMM_COOP=1` - forcing this on incapable hardware is
    /// untested).
    public var coopEnabled: Bool {
        switch coopEnvRaw {
        case "0": return false
        case "1": return true
        default: return Ops.mppAutoCapable
        }
    }

    /// F-85 two-expert tile-pairing lever gate (`iron_moe_gather_qmm_coop_paired`,
    /// splits the coop tile plan into an "own" plan for full/17-31-row
    /// remainders, fed the UNCHANGED `iron_moe_gather_qmm_coop` dispatch,
    /// plus a "paired" plan for `1..=16`-row remainders packed two experts
    /// to a tile, fed the new paired-GEMM dispatch - see
    /// `moe_tile_plan_builder_paired.rs`'s module doc). Isolated kernel
    /// bench GO/NO-GO verdict (own+paired dispatch time vs the unpaired
    /// coop core's single-dispatch time, same Zipf-seeded routing draws):
    /// paired-total beats unpaired by 7-12% at mTotal=4096, 4-9% at 8192,
    /// parity at 16384, and LOSES by 3-6% at 32768 (the 2x per-tile
    /// dequant cost of pairing stops paying for itself once padding is
    /// already near zero at high mTotal - see `paired_coverage`'s doc).
    ///
    /// DEFAULT OFF - opt in with `BUTTER_MOE_PAIRED=1`. A follow-up round
    /// preallocated the six paired-plan buffers this file used to
    /// `device.makeBuffer` fresh every layer, every forward call (see
    /// `pairedTilePlanBuffers`) and re-measured: the e2e regression this
    /// gate originally shipped with (15.6% at T=512, 10.6% at T=1024)
    /// barely moved (~14.7% at T=512 with the buffers preallocated), so
    /// the buffer-allocation hypothesis this doc used to carry was WRONG,
    /// or at best a minor contributor. An ablation (env-gated skip of the
    /// paired builder dispatch vs the paired GEMM dispatch, isolating
    /// each) pinned the actual cost on the paired GEMM dispatch itself,
    /// independent of whether it has any real rows to process: the
    /// dispatch grid's height is `moePairedTilePlanCapacity(nExperts:)`
    /// (`ceil(nExperts/2)`, a STATIC worst-case bound, 128 at this
    /// model's 256 experts) - not the actual number of paired tiles a
    /// given routing draw produces, which is typically far smaller at
    /// low mTotal. Every dispatch therefore launches ~128 threadgroups
    /// per leg per layer regardless of how many are real, and the
    /// kernel's masking (`x_in_run` et al.) only zeroes the loaded row
    /// data for an inert tile - it does not shrink the K-loop / coop-matmul
    /// work that threadgroup still runs. The isolated kernel bench that
    /// produced the 7-12%-win verdict above dispatches at the ACTUAL
    /// paired-tile count for its routing draw, not this worst-case
    /// capacity, which is why its verdict does not reproduce here: real
    /// wiring cannot know the actual count without a GPU-to-CPU
    /// readback (which DEVPLAN2 exists specifically to avoid), so it
    /// must always dispatch at capacity.
    ///
    /// A follow-up round built the indirect dispatch this doc called for
    /// (`dispatchThreadgroupsIndirect` fed the device-computed real
    /// paired-tile count - see `pairedIndirectEnabled`, default ON
    /// wherever this gate is opted into). Honest result: it is a REAL,
    /// bit-exact-verified improvement, not a full fix. Same-binary
    /// interleaved A/B/C (base / old-capacity-dispatch / new-indirect-
    /// dispatch, 7 trials each) measured the regression at T=512 going
    /// from -14.9% (old) to -10.6% (new indirect) vs base, and at T=1024
    /// from -9.7% (old) to -6.2% (new indirect) vs base - roughly a third
    /// to two-fifths of the regression recovered, but paired STILL loses
    /// to the unpaired coop path at both T. This means the fixed
    /// per-dispatch overhead (a second compute-encoder creation, PSO
    /// bind, and buffer sets for the paired GEMM, plus the paired
    /// builder's own dispatch, every layer, every forward call) is a
    /// real, separate cost from the capacity-padding this round fixed -
    /// not yet isolated or addressed. Given that, this gate STAYS DEFAULT
    /// OFF - the indirect-dispatch fix is real but not sufficient on its
    /// own to flip the default. Opt-in either way: `BUTTER_MOE_PAIRED=1`
    /// enables, unset or `=0` stays off. Only takes effect where
    /// `useCoopTilePlan` already holds - see `decodeMany`.
    public var pairedEnabled: Bool {
        pairedEnvRaw == "1"
    }

    /// mTotal threshold for the tile-pairing lever (see `pairedEnabled`):
    /// paired dispatch runs when `mTotal < pairedMaxMTotal`, the unpaired
    /// single-dispatch coop path otherwise. Default 16384 - the measured
    /// parity boundary, chosen over 32768 (where pairing measures a 3-6%
    /// LOSS) so the default never dispatches the extra pass where it
    /// cannot win. Override via `BUTTER_MOE_PAIRED_MAX_MTOTAL=<mTotal>`.
    public var pairedMaxMTotal: Int {
        if let raw = pairedMaxMTotalEnvRaw, let v = Int(raw) {
            return v
        }
        return 16384
    }

    /// F-85 paired-GEMM indirect-dispatch gate (root-cause fix for the
    /// worst-case-capacity regression `pairedEnabled`'s doc measured).
    /// Default ON wherever `useCoopPaired` already holds - the dispatch
    /// grid then comes from the device-computed real paired-tile count
    /// (`Ops.moeBuildTilePlanBm32Paired`'s `pairCountGateUp`/`pairCountDown`
    /// outputs, fed through `dispatchThreadgroupsIndirect`) instead of the
    /// static `moePairedTilePlanCapacity(nExperts:)` bound. Escape hatch
    /// `BUTTER_MOE_PAIRED_INDIRECT=0` reverts to the old direct/capacity
    /// dispatch, kept only for A/B measurement against this fix - not a
    /// production fallback (the old path is the one that regressed).
    public var pairedIndirectEnabled: Bool {
        pairedIndirectEnvRaw != "0"
    }

    /// F-85 idle-tile-cost indirect-dispatch gate for the PRODUCTION coop
    /// GEMM (`iron_moe_gather_qmm_coop`). An ablation on this kernel's
    /// guard structure found no early exit before the K-loop: every
    /// dispatched threadgroup, including worst-case-capacity padding whose
    /// `tileRowCount == 0`, pays the full W dequant and `coop_tile_run`
    /// matmul - same-binary capacity-vs-real-tile-count dispatch measured
    /// 10-21% idle-tile cost at the F-85 low-context mTotal targets, well
    /// past the round's 5% materiality bar. Default ON wherever
    /// `useCoopTilePlan` already holds (both the plain path and the
    /// paired lever's "own" dispatch) - the dispatch grid then comes from
    /// the device-computed real tile count
    /// (`Ops.moeBuildTilePlanBm32`/`Own`'s `tileCountGateUp`/`tileCountDown`
    /// outputs, fed through `dispatchThreadgroupsIndirect`) instead of the
    /// static `moeTilePlanCapacityBm32(mTotal:nExperts:)` bound. Escape
    /// hatch `BUTTER_MOE_COOP_INDIRECT=0` reverts to the old direct/capacity
    /// dispatch, for A/B measurement against this fix.
    public var coopIndirectEnabled: Bool {
        coopIndirectEnvRaw != "0"
    }

    /// DEVPLAN2 + stacked int4: expert path can stay on the caller's cmd.
    /// Single source of truth for the token-order device plan — `decodeMany`
    /// branches on this same predicate, so caller chaining (`commitAtEnd:
    /// false`) and the internal no-mid-commit path can never disagree.
    public var prefillChainsOnCallerCmd: Bool {
        ProcessInfo.processInfo.environment["DEVPLAN2"] != "0"
            && gpuRouterEnabled
            && router.gatingMode == .softmaxThenTopK
            && router.normTopKProb
            && router.expertBias == nil
            && router.topK == 8
            && stackedInt4Experts != nil
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in gate.parameters() { out.append(("gate.\(k)", v)) }
        for (e, proj) in gateProj.enumerated() {
            for (k, v) in proj.parameters() { out.append(("experts.\(e).gate_proj.\(k)", v)) }
        }
        for (e, proj) in upProj.enumerated() {
            for (k, v) in proj.parameters() { out.append(("experts.\(e).up_proj.\(k)", v)) }
        }
        for (e, proj) in downProj.enumerated() {
            for (k, v) in proj.parameters() { out.append(("experts.\(e).down_proj.\(k)", v)) }
        }
        if let sg = sharedGateProj {
            for (k, v) in sg.parameters() { out.append(("shared_expert.gate_proj.\(k)", v)) }
        }
        if let su = sharedUpProj {
            for (k, v) in su.parameters() { out.append(("shared_expert.up_proj.\(k)", v)) }
        }
        if let sd = sharedDownProj {
            for (k, v) in sd.parameters() { out.append(("shared_expert.down_proj.\(k)", v)) }
        }
        return out
    }

    /// Lazily allocates (grow-only, once-or-on-growth per instance) and
    /// returns the three F-85 tile-plan buffers backing `useTilePlan` /
    /// `useCoopTilePlan`, then zero-fills the requested `capacity` prefix
    /// of each before handing them back. See the property doc above
    /// `tilePlanTileExpertBuf` for the grow-only reuse design and why
    /// re-zeroing the requested prefix every call is still required for
    /// correctness (a previous, possibly larger, call's tail past THIS
    /// call's real tile count would otherwise leak stale
    /// expert/row_start/row_count into what the GEMM kernel expects to
    /// be an inert dead tile).
    private func tilePlanBuffers(
        capacity: Int, device: Device
    ) -> (tileExpert: MTLBuffer, tileRowStart: MTLBuffer, tileRowCount: MTLBuffer) {
        if tilePlanTileExpertBuf == nil || tilePlanBufCapacity < capacity {
            tilePlanTileExpertBuf = device.makeBuffer(length: capacity * 4)
            tilePlanTileRowStartBuf = device.makeBuffer(length: capacity * 4)
            tilePlanTileRowCountBuf = device.makeBuffer(length: capacity * 4)
            tilePlanBufCapacity = capacity
        }
        let teBuf = tilePlanTileExpertBuf!
        let trsBuf = tilePlanTileRowStartBuf!
        let trcBuf = tilePlanTileRowCountBuf!
        memset(teBuf.contents(), 0, capacity * 4)
        memset(trsBuf.contents(), 0, capacity * 4)
        memset(trcBuf.contents(), 0, capacity * 4)
        return (teBuf, trsBuf, trcBuf)
    }

    /// Lazily allocates (once per instance) and returns the six F-85
    /// paired-tile-plan buffers, then zero-fills all six before handing
    /// them back. See the property doc above `pairedTileExpertABuf` for
    /// the allocation-reuse design and why re-zeroing every call is still
    /// required for correctness (the real per-call paired-tile count is
    /// data-dependent, so a previous call's tail past THIS call's real
    /// count would otherwise leak stale expert/row_start/row_count into
    /// what the paired GEMM kernel expects to be an inert dead tile).
    /// The zero-fill itself is a few KB (`6 * pairedCap * 4` bytes) -
    /// negligible; the `device.makeBuffer` calls this replaces were the
    /// actual per-layer, per-forward-call cost.
    private func pairedTilePlanBuffers(
        pairedCap: Int, device: Device
    ) -> (
        expertA: MTLBuffer, rowStartA: MTLBuffer, rowCountA: MTLBuffer,
        expertB: MTLBuffer, rowStartB: MTLBuffer, rowCountB: MTLBuffer
    ) {
        if pairedTileExpertABuf == nil || pairedTileBufCapacity != pairedCap {
            pairedTileExpertABuf = device.makeBuffer(length: pairedCap * 4)
            pairedTileRowStartABuf = device.makeBuffer(length: pairedCap * 4)
            pairedTileRowCountABuf = device.makeBuffer(length: pairedCap * 4)
            pairedTileExpertBBuf = device.makeBuffer(length: pairedCap * 4)
            pairedTileRowStartBBuf = device.makeBuffer(length: pairedCap * 4)
            pairedTileRowCountBBuf = device.makeBuffer(length: pairedCap * 4)
            pairedTileBufCapacity = pairedCap
        }
        let teaBuf = pairedTileExpertABuf!
        let trsaBuf = pairedTileRowStartABuf!
        let trcaBuf = pairedTileRowCountABuf!
        let tebBuf = pairedTileExpertBBuf!
        let trsbBuf = pairedTileRowStartBBuf!
        let trcbBuf = pairedTileRowCountBBuf!
        memset(teaBuf.contents(), 0, pairedCap * 4)
        memset(trsaBuf.contents(), 0, pairedCap * 4)
        memset(trcaBuf.contents(), 0, pairedCap * 4)
        memset(tebBuf.contents(), 0, pairedCap * 4)
        memset(trsbBuf.contents(), 0, pairedCap * 4)
        memset(trcbBuf.contents(), 0, pairedCap * 4)
        return (teaBuf, trsaBuf, trcaBuf, tebBuf, trsbBuf, trcbBuf)
    }

    /// Lazily allocates (once per instance) the two paired-GEMM indirect-args
    /// buffers - see the property doc above `pairedIndirectArgsGateUpBuf`.
    /// Pre-fills `n_out_tiles` (word 0) and `z=1` (word 2) from the CPU;
    /// leaves `pair_count` (word 1) untouched for
    /// `Ops.moeBuildTilePlanBm32Paired` to write every call. Safe to only
    /// do this once: `moeIntermediate % 64 == 0` / `hidden % 64 == 0`
    /// already hold by the time this is called (gated on `useCoopTilePlan`
    /// upstream), so the division is exact.
    private func pairedIndirectArgsBuffers(
        moeIntermediate: Int, hidden: Int, device: Device
    ) -> (gateUp: MTLBuffer, down: MTLBuffer) {
        if pairedIndirectArgsGateUpBuf == nil {
            let gateUpBuf = device.makeBuffer(length: 3 * 4)
            let downBuf = device.makeBuffer(length: 3 * 4)
            let gateUpWords = gateUpBuf.contents().bindMemory(to: UInt32.self, capacity: 3)
            gateUpWords[0] = UInt32(moeIntermediate / 64)
            gateUpWords[1] = 0
            gateUpWords[2] = 1
            let downWords = downBuf.contents().bindMemory(to: UInt32.self, capacity: 3)
            downWords[0] = UInt32(hidden / 64)
            downWords[1] = 0
            downWords[2] = 1
            pairedIndirectArgsGateUpBuf = gateUpBuf
            pairedIndirectArgsDownBuf = downBuf
        }
        return (pairedIndirectArgsGateUpBuf!, pairedIndirectArgsDownBuf!)
    }

    /// Lazily allocates (once per instance) the two indirect-args buffers
    /// for the PRODUCTION coop GEMM - see the property doc above
    /// `coopIndirectArgsGateUpBuf` for the sharing argument between the
    /// plain `useCoopTilePlan` path and the paired lever's "own" dispatch.
    /// Same word layout and same CPU-fills-words-0/2-once contract as
    /// `pairedIndirectArgsBuffers`.
    private func coopIndirectArgsBuffers(
        moeIntermediate: Int, hidden: Int, device: Device
    ) -> (gateUp: MTLBuffer, down: MTLBuffer) {
        if coopIndirectArgsGateUpBuf == nil {
            let gateUpBuf = device.makeBuffer(length: 3 * 4)
            let downBuf = device.makeBuffer(length: 3 * 4)
            let gateUpWords = gateUpBuf.contents().bindMemory(to: UInt32.self, capacity: 3)
            gateUpWords[0] = UInt32(moeIntermediate / 64)
            gateUpWords[1] = 0
            gateUpWords[2] = 1
            let downWords = downBuf.contents().bindMemory(to: UInt32.self, capacity: 3)
            downWords[0] = UInt32(hidden / 64)
            downWords[1] = 0
            downWords[2] = 1
            coopIndirectArgsGateUpBuf = gateUpBuf
            coopIndirectArgsDownBuf = downBuf
        }
        return (coopIndirectArgsGateUpBuf!, coopIndirectArgsDownBuf!)
    }

    /// `DecoderLayer` conformance. `cache` is a `StatelessLayerCache`
    /// and is ignored — an MoE FFN holds no per-token state.
    ///
    /// IMPORTANT: this commits the passed `cmd` (the router needs the
    /// gate logits on the CPU). The host model must obtain a fresh
    /// command buffer afterwards. See the file header.
    public func decode(
        _ h: Tensor, position _: Int,
        cache _: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        return Profile.signpost("moe.decode") { () -> Tensor in
        precondition(
            h.elementCount == hidden,
            "MoELayer.decode: input has \(h.elementCount) elements, expected hidden \(hidden)")

        // ── 1. Gate gemv on the caller's command buffer ──────────────
        let logitsTensor = gate(h, on: cmd)

        // ── 2a. GPU MoE router fast path ──────────────────────────────
        // When `BUTTER_MOE_GPU_ROUTER=1` (default ON) AND the layer has
        // stacked int4 expert layout AND routing matches
        // `iron_moe_router_topk`'s assumptions (softmax-then-topK + Qwen-
        // MoE-style renorm, topK=8), skip the CPU sync entirely. The
        // router writes topK indices + weights to a GPU buffer; per-
        // expert qmms read their slot's expert id from that buffer via
        // `dequantGemvInt4ExpertIndexed`; the per-slot routing weight
        // feeds `scalarFMAChain8`.
        if self.gpuRouterEnabled,
            let stacked = stackedInt4Experts,
            stacked.dtype == h.dtype,
            router.gatingMode == .softmaxThenTopK,
            router.normTopKProb,
            router.expertBias == nil,
            router.topK == 8
        {
            return decodeGPURouter(
                h: h, logitsTensor: logitsTensor,
                stacked: stacked, cmd: cmd, device: device)
        }

        // ── 2. Commit + wait so the router can read the logits ───────
        // The decode path batches a token onto one command buffer; the
        // router needs a CPU sync point. Commit here, then run the
        // experts on a private buffer below.
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── 3. CPU routing — top-K + combine weights ─────────────────
        let logits = logitsTensor.toFloatArray()
        let routing = router.route(logits: logits)

        // ── 4. Per-expert SwiGLU on a private command buffer ─────────
        // Only the selected experts run (the non-selected ones would
        // contribute zero — skipping them is the same result, cheaper).
        let work = device.makeCommandBuffer()
        var accumulator: Tensor?
        // Batched gather-BGEMM fast path is opt-in via BUTTER_MOE_BGEMM=1.
        // At T=1 decode with topK=8 / m_total=8 it currently regresses
        // vs the sequential per-expert matvec path on M5 Max — the
        // bm16 tile pads to 16 rows but only 8 commit, so the kernel
        // pays full weight-reload bandwidth for 50% useful work. Real
        // win shape is prefill (m_total scales with T, more rows share
        // each expert's weight tile) or NAX (BUTTER_MOE_BGEMM_MPP=1).
        // Path is wired + correctness-verified — flip the default once
        // the m_total<16 regression is closed (likely via either a
        // dedicated bm8 kernel emit or by promoting prefill to use
        // this path first).
        if let stacked = stackedInt4Experts, stacked.dtype == h.dtype, enableBGEMM {
            // Fast path: one batched gather BGEMM per projection. The
            // kernel expects rows of activations sorted by expert id; we
            // sort the topK indices ascending and replicate `h` into the
            // gathered row order. The per-row expert assignment goes in
            // an int32 indices buffer.
            accumulator = batchedSwiGLU(
                h, stacked: stacked, routing: routing,
                on: work, device: device)
        } else {
            for (slot, expertId) in routing.indices.enumerated() {
                // Broadcast the CPU combine weight into a [hidden] constant
                // tensor so the element-wise `Ops.mul` can scale the expert
                // output — avoids a dedicated scalar-multiply kernel.
                let weightTensor = Tensor.filled(
                    routing.weights[slot],
                    shape: [hidden], dtype: h.dtype,
                    device: device)
                let expertOut = swiGLU(
                    h,
                    gateProj: gateProj[expertId],
                    upProj: upProj[expertId],
                    downProj: downProj[expertId],
                    on: work)
                let scaled = Ops.mul(expertOut, weightTensor, on: work)
                accumulator = accumulator.map { Ops.add($0, scaled, on: work) } ?? scaled
            }
        }

        // ── 5. Optional always-on shared expert ──────────────────────
        if let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj {
            let sharedOut = swiGLU(h, gateProj: sg, upProj: su, downProj: sd, on: work)
            accumulator = accumulator.map { Ops.add($0, sharedOut, on: work) } ?? sharedOut
        }

        // topK ≥ 1 so `accumulator` is always non-nil here.
        let result = accumulator!
        // Commit without wait: `result` is on the in-flight `work`
        // buffer; the caller's residual-add cmd will hazard-track the
        // read against this write. Saves ~0.5-1 ms host-stall per
        // MoE layer per token (Qwen3.6-A3B = 40 layers).
        work.commit()
        return result
        }  // Profile.signpost("moe.decode")
    }

    /// T-batched MoE forward. `hFlat` is `[T, hidden]` flat; returns
    /// `[T, hidden]` flat. The mTotal = T·topK rows fan into a single
    /// `Ops.moeGatherDequantGemmInt4` (or bm8 variant) per projection
    /// instead of T·topK sequential per-expert SwiGLU triplets. At
    /// Qwen3.6-A3B T=32, topK=8 → mTotal=256 fills the BM=16 tiles
    /// 16-deep — the regime where cooperative-tensor weight sharing
    /// PAYS (the same kernels regress at T=1 because every row is a
    /// unique expert with no shared weight).
    ///
    /// Requires `stackedInt4Experts` (the BGEMM weights layout). Falls
    /// back to per-token `decode` loop otherwise.
    ///
    /// Architecture:
    ///   1. Gate gemv batched (`gate.callMany`) → `[T, nExperts]`.
    ///   2. `cmd.commit() + wait` — router needs logits on CPU.
    ///   3. Per-token routing on host (T calls, each O(nExperts) + a
    ///      partial sort).
    ///   4. Build plan over T·topK rows: `sourceToken[m]`,
    ///      `expertId[m]`, `weight[m]`. Sort by `expertId` ascending so
    ///      consecutive rows share an expert (the kernel walks rows in
    ///      tile order; sorted = weight tile reuse across rows in a
    ///      tile).
    ///   5. `Ops.gather(h, gatherIdx)` → `xGathered [mTotal, hidden]`
    ///      in one dispatch (vs T·topK host memcpys).
    ///   6. `moeGatherDequantGemmInt4` × 3 (gate / up / down) on the
    ///      gathered batch — one dispatch each.
    ///   7. `Ops.swiglu` element-wise across `[mTotal, moeIntermediate]`.
    ///   8. Weighted scatter-sum: for each `m`, scale `downOut[m]` by
    ///      `weights[m]` and accumulate into `outFlat[sourceToken[m]]`.
    ///      Uses the same `Tensor.filled([hidden])` broadcast pattern as
    ///      the single-token `batchedSwiGLU`, mTotal times. Scatter-sum
    ///      is the dominant residual cost at large `mTotal`; a future
    ///      `iron_moe_scatter_scale_add` kernel collapses these mTotal
    ///      dispatches into one.
    ///   9. Shared expert (if any): one per-row SwiGLU call across T
    ///      tokens (also currently a T-loop because the single-token
    ///      `swiGLU` path is the standing API).
    ///
    /// All work after the gate-readback runs on a fresh `work` cmd that
    /// commits once at the end. The caller's `cmd` is consumed by the
    /// gate gemv and the commit + wait.
    public func decodeMany(
        _ hFlat: Tensor, t: Int,
        cmd: MTLCommandBuffer, device: Device,
        commitAtEnd: Bool = true
    ) -> Tensor {
        precondition(
            hFlat.elementCount == t * hidden,
            "MoELayer.decodeMany: hFlat size \(hFlat.elementCount) ≠ T·hidden = \(t * hidden)")
        precondition(t > 0, "MoELayer.decodeMany: T must be positive")

        let dt = hFlat.dtype

        // ── 1. Gate gemv batched on caller's cmd ─────────────────────────
        let hRows = hFlat.reshaped(to: [t, hidden])
        let gateLogitsAll = gate.callMany(hRows, t: t, on: cmd, device: device)
        // gateLogitsAll shape: [T, nExperts]

        // ── 2. Routing — GPU fast path or CPU sync ──────────────────────
        // When router config matches `iron_moe_router_topk`'s semantic
        // envelope (softmax → topK + Qwen-MoE renorm, no expert bias,
        // k=8), run the T-batched router on GPU via
        // `Ops.moeRouterTopKMany`. Eliminates the T-parallel host
        // `router.route` loop (~50 ms of CPU work per prefill at T=512 ×
        // 40 MoE layers). One commit+wait still needed to read indices +
        // weights for the host-side sort plan.
        let nExperts = router.nExperts
        let topK = router.topK
        // Token-order device plan (DEVPLAN2, default on): the GPU router
        // writes indices+weights; we run the expert GEMMs in token order
        // (no expert sort), so no readback / waitUntilCompleted / host sort.
        // Same predicate as `prefillChainsOnCallerCmd` — callers use it to
        // decide `commitAtEnd`, so the two must never drift.
        let useDevPlan2 = prefillChainsOnCallerCmd
        var d2Indices: Tensor? = nil
        var d2Source: Tensor? = nil
        var d2InvPerm: Tensor? = nil
        var d2Weights: Tensor? = nil
        var routings = [MoERouter.Routing](
            repeating: MoERouter.Routing(indices: [], weights: []),
            count: t)
        let useGPURouter =
            self.gpuRouterEnabled
            && router.gatingMode == .softmaxThenTopK
            && router.normTopKProb
            && router.expertBias == nil
            && router.topK == 8
        if useDevPlan2 {
            let mTot = t * topK
            let indicesBuf = Tensor.empty(shape: [t, topK], dtype: .u32, device: device)
            let weightsBuf = Tensor.empty(shape: [t, topK], dtype: gateLogitsAll.dtype, device: device)
            Ops.moeRouterTopKMany(
                logits: gateLogitsAll, indicesOut: indicesBuf, weightsOut: weightsBuf,
                t: t, nExperts: nExperts, k: topK, normTopkProb: router.normTopKProb, on: cmd)
            // Device sort: expert-grouped plan on GPU (no readback/wait/host sort).
            let idsFlat = Tensor(buffer: indicesBuf.buffer, offset: indicesBuf.offset, shape: [mTot], dtype: .u32)
            let se = Tensor.empty(shape: [mTot], dtype: .u32, device: device)
            let st = Tensor.empty(shape: [mTot], dtype: .u32, device: device)
            let ip = Tensor.empty(shape: [mTot], dtype: .u32, device: device)
            // F-85 bucketed counting-sort (default ON, see
            // `sortPlanCountingEnabled`; escape hatch
            // BUTTER_MOE_COUNTING_SORT=0 reverts to the O(mTotal^2) original).
            // Block size 256 - see `moe_sort_plan_counting.rs`'s isolated
            // bench sweep at production mTotal.
            if self.sortPlanCountingEnabled {
                Ops.moeSortPlanCounting(
                    topkIds: idsFlat, sortedExperts: se, sourceTokens: st, invPerm: ip,
                    mTotal: mTot, k: topK, nExperts: nExperts, blockSize: 256, on: cmd)
            } else {
                Ops.moeSortPlan(
                    topkIds: idsFlat, sortedExperts: se, sourceTokens: st, invPerm: ip,
                    mTotal: mTot, k: topK, on: cmd)
            }
            d2Source = st
            d2InvPerm = ip
            d2Indices = se
            d2Weights = Tensor(buffer: weightsBuf.buffer, offset: weightsBuf.offset, shape: [mTot], dtype: gateLogitsAll.dtype)
        } else if useGPURouter {
            let indicesBuf = Tensor.empty(
                shape: [t, topK], dtype: .u32, device: device)
            let weightsBuf = Tensor.empty(
                shape: [t, topK], dtype: gateLogitsAll.dtype, device: device)
            Ops.moeRouterTopKMany(
                logits: gateLogitsAll, indicesOut: indicesBuf,
                weightsOut: weightsBuf,
                t: t, nExperts: nExperts, k: topK,
                normTopkProb: router.normTopKProb, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            let indicesHost: [UInt32] = indicesBuf.toArray(as: UInt32.self)
            let weightsHost: [Float] = weightsBuf.toFloatArray()
            for r in 0 ..< t {
                let base = r * topK
                let idxs = (0 ..< topK).map { Int(indicesHost[base + $0]) }
                let wts = Array(weightsHost[base ..< (base + topK)])
                routings[r] = MoERouter.Routing(indices: idxs, weights: wts)
            }
        } else {
            cmd.commit()
            cmd.waitUntilCompleted()
            let logitsHost = gateLogitsAll.toFloatArray()
            routings.withUnsafeMutableBufferPointer { buf in
                // `nonisolated(unsafe)`: each iteration writes a disjoint
                // `buf[r]`. `router` is already Sendable (its state is
                // read-only for the duration of this call) so the local
                // capture needs no extra qualifier — only the raw pointer
                // does, since Swift can't reason about disjoint writes.
                nonisolated(unsafe) let bufPtr = buf.baseAddress!
                let routerLocal = router
                DispatchQueue.concurrentPerform(iterations: t) { r in
                    let start = r * nExperts
                    let rowLogits = Array(logitsHost[start ..< (start + nExperts)])
                    bufPtr[r] = routerLocal.route(logits: rowLogits)
                }
            }
        }

        // ── 4. Build sorted plan over T·topK rows (host) — skipped when
        //      the token-order device plan is active. ────────────────────
        let mTotal = t * topK
        var sortedExpertsHost = [UInt32]()
        var sourceTokensHost = [UInt32]()
        var invPermHost = [UInt32]()
        var weightsTokenOrderHost = [Float]()
        if d2Indices == nil {
        // Tuple: (sortKey expertId, sourceToken, originalSlot, weight)
        var planTuples: [(Int, Int, Int, Float)] = []
        planTuples.reserveCapacity(mTotal)
        for r in 0 ..< t {
            let routing = routings[r]
            for slot in 0 ..< topK {
                planTuples.append((routing.indices[slot], r, slot, routing.weights[slot]))
            }
        }
        planTuples.sort { $0.0 < $1.0 }
        sortedExpertsHost.reserveCapacity(mTotal)
        sourceTokensHost.reserveCapacity(mTotal)
        var sortedWeightsHost = [Float]()
        sortedWeightsHost.reserveCapacity(mTotal)
        for tuple in planTuples {
            sortedExpertsHost.append(UInt32(tuple.0))
            sourceTokensHost.append(UInt32(tuple.1))
            sortedWeightsHost.append(tuple.3)
        }
        // ── 4b. Build inv_perm + per-token-order weights for the
        // iron_moe_unpermute fused scatter-sum kernel. `invPermHost[r·k+s]`
        // = sortedIdx where (token r, slot s) landed in `downOut`.
        // `weightsTokenOrderHost[r·k+s]` = routings[r].weights[s]
        // (unsorted, matches the kernel's per-token lookup).
        invPermHost = [UInt32](repeating: 0, count: mTotal)
        for (sortedIdx, tuple) in planTuples.enumerated() {
            let r = tuple.1
            let slot = tuple.2
            invPermHost[r * topK + slot] = UInt32(sortedIdx)
        }
        weightsTokenOrderHost.reserveCapacity(mTotal)
        for r in 0 ..< t {
            for slot in 0 ..< topK {
                weightsTokenOrderHost.append(routings[r].weights[slot])
            }
        }
        }  // end if d2Indices == nil (host sort)

        // ── 5. Fall back to per-token decode if no stacked-int4 weights ──
        guard let stacked = stackedInt4Experts, stacked.dtype == dt else {
            let outFlat = Tensor.empty(shape: [t * hidden], dtype: dt, device: device)
            let dtBytes = dt.byteSize
            var workCmd = device.makeCommandBuffer()
            for r in 0 ..< t {
                let hRow = Tensor(
                    buffer: hFlat.buffer,
                    offset: hFlat.offset + r * hidden * dtBytes,
                    shape: [hidden], dtype: dt)
                let rowOut = decode(
                    hRow, position: 0,
                    cache: StatelessLayerCache(),
                    cmd: workCmd, device: device)
                let outRow = Tensor(
                    buffer: outFlat.buffer,
                    offset: outFlat.offset + r * hidden * dtBytes,
                    shape: [hidden], dtype: dt)
                let copyCmd = device.makeCommandBuffer()
                Ops.copy(rowOut, into: outRow, on: copyCmd)
                copyCmd.commit()
                workCmd = device.makeCommandBuffer()
            }
            return outFlat
        }

        // ── 6. Work cmd for the rest of the layer ────────────────────────
        // DEVPLAN2 reuses `cmd`; host-router paths already committed it.
        let work = useDevPlan2 ? cmd : device.makeCommandBuffer()
        let moeIntermediate = stacked.moeIntermediate
        let groupSize = stacked.groupSize

        // ── 7. Gather index + expert ids ─────────────────────────────────
        let gatherIdxTensor: Tensor
        let indices: Tensor
        if let di = d2Indices, let ds = d2Source {
            gatherIdxTensor = ds
            indices = di
        } else {
            let gatherIdxBuf = device.makeBuffer(length: mTotal * 4)
            sourceTokensHost.withUnsafeBytes {
                _ = memcpy(gatherIdxBuf.contents(), $0.baseAddress!, mTotal * 4)
            }
            gatherIdxTensor = Tensor(buffer: gatherIdxBuf, offset: 0, shape: [mTotal], dtype: .u32)
            let expertIdsBuf = device.makeBuffer(length: mTotal * 4)
            sortedExpertsHost.withUnsafeBytes {
                _ = memcpy(expertIdsBuf.contents(), $0.baseAddress!, mTotal * 4)
            }
            indices = Tensor(buffer: expertIdsBuf, offset: 0, shape: [mTotal], dtype: .u32)
        }

        // ── 8. Gather activations: [mTotal, hidden] ──────────────────────
        // Source rows are picked from h[sourceToken] — one dispatch.
        let xGathered = Ops.gather(table: hRows, tokenIds: gatherIdxTensor, on: work)
        precondition(
            xGathered.elementCount == mTotal * hidden,
            "MoELayer.decodeMany: gather output unexpected size")

        // ── 9. Gate / up BGEMM → [mTotal, moeIntermediate] ───────────────
        let gateOut = Tensor.empty(
            shape: [mTotal, moeIntermediate], dtype: dt,
            device: device)
        let upOut = Tensor.empty(
            shape: [mTotal, moeIntermediate], dtype: dt,
            device: device)
        // Tile selection for the batched-prefill regime:
        //   - mTotal ≥ 64 + `BUTTER_MOE_BGEMM_BM64=1` → bm64_mpp (NAX
        //     cooperative-tensor). Originally gated off after the
        //     equivalence canary tripped (argmax drift) at m1; the root
        //     cause was `Ops.moeGatherDequantGemmInt4Bm64Mpp` dispatching
        //     via `dispatchThreads` instead of `dispatchThreadgroups`,
        //     fixed in the commit titled "dispatch bm64 int4 MPP kernel
        //     as threadgroups, not threads". Post-fix quality gate is
        //     cosine 1.0 / top1 match / zero max-logit delta vs bm16 at
        //     mTotal 1024/2067/4096, so the wrapper itself is correct.
        //     It stays opt-in for the perf reason below, not a
        //     correctness gate.
        //   - 16 ≤ mTotal → bm16 (the default `moeGatherDequant
        //     GemmInt4`). 16-deep tile fill across consecutive
        //     sorted-by-expert rows. THIS is the path that drives the
        //     2.69× T=32 win.
        //   - mTotal ≤ 8 + `BUTTER_MOE_BGEMM_BM8=1` → bm8 (decode T=1
        //     fallback).
        // bm16 (simdgroup-MMA `moeGatherDequantGemmInt4`) is the default:
        // it is fastest across every measured prefill mTotal for this
        // model's narrow experts (moeIntermediate 512), +3-5% end-to-end
        // over bm64_mpp at mTotal 1024..16384 on an M5 Max, wikitext-2
        // perplexity unchanged. bm64_mpp (NAX cooperative-tensor) is
        // retained as opt-in via BUTTER_MOE_BGEMM_BM64=1 for wider-expert
        // models / other hardware where its larger tile may win.
        let useBm64 = self.bm64Env && !self.noBGEMMBm64 && mTotal >= 1024
        let useBm8 = !useBm64 && topK <= 8 && useBm8Env && mTotal <= 8
        let bgemm:
            (Tensor, Tensor, Tensor, Tensor, Tensor, Int, Int, Int, Int, MTLCommandBuffer, Tensor)
                -> Void
        if useBm64 {
            bgemm = Ops.moeGatherDequantGemmInt4Bm64Mpp
        } else if useBm8 {
            bgemm = Ops.moeGatherDequantGemmInt4Bm8
        } else {
            bgemm = Ops.moeGatherDequantGemmInt4
        }

        // F-85 expert-aligned tile-plan MoE gather GEMM (default ON, see
        // `tilePlanEnabled`; escape hatch BUTTER_MOE_TILEPLAN=0). Only
        // wired into the DEVPLAN2 device-sorted-plan path: `indices` there is already
        // expert-sorted ascending on device (iron_moe_sort_plan's own
        // output, no host readback) - exactly what
        // `iron_moe_build_tile_plan` needs, and preserving that no-sync
        // property is the whole point of building the plan on-device
        // instead of reading `indices` back to the host. The other
        // routing paths (host GPU-router sync, CPU router) keep bm16
        // regardless of this flag - they already pay a host round-trip
        // elsewhere in this function, and wiring tileplan there would
        // need a Swift port of `build_tile_plan` this pass didn't need
        // for the measured prefill path. bm64/bm8 opt-ins also take
        // priority when set - tileplan only replaces the bm16 default.
        let useTilePlan = useDevPlan2 && !useBm64 && !useBm8 && self.tilePlanEnabled
        // F-85 coop-core MoE gather GEMM (BM=32 tile plan over
        // `iron_moe_gather_qmm_coop`, see `coopEnabled`). Only takes over
        // from the bm16 tileplan default when the coop core's shape
        // requirements hold for BOTH gate/up (nOut=moeIntermediate) and
        // down (nOut=hidden) legs at this layer - `nOut % 64 == 0` (BN=64)
        // and `groupSize == 64` (the dense core's baked group assumption,
        // shared here). A layer that fails either check silently keeps the
        // bm16 tileplan path even with the env flag set, rather than
        // half-switching one leg.
        let useCoopTilePlan =
            useTilePlan && self.coopEnabled
            && moeIntermediate.isMultiple(of: 64) && hidden.isMultiple(of: 64)
            && groupSize == 64
        // F-85 two-expert tile-pairing lever (see `pairedEnabled` /
        // `pairedMaxMTotal` doc) - only takes over from the single-dispatch
        // coop path when the coop core is already active AND mTotal is
        // below the measured parity/loss boundary.
        let useCoopPaired =
            useCoopTilePlan && self.pairedEnabled && mTotal < self.pairedMaxMTotal
        var tilePlanTileExpert: Tensor?
        var tilePlanTileRowStart: Tensor?
        var tilePlanTileRowCount: Tensor?
        var tilePlanCapacity = 0
        var tilePlanXRows: Tensor?
        // Paired-tile plan halves (see `useCoopPaired`) - only built when
        // that gate holds; `tilePlanTileExpert`/etc above then hold the
        // "own" plan (full/17-31-row remainders) instead of the regular
        // combined plan.
        var pairedTileExpertA: Tensor?
        var pairedTileRowStartA: Tensor?
        var pairedTileRowCountA: Tensor?
        var pairedTileExpertB: Tensor?
        var pairedTileRowStartB: Tensor?
        var pairedTileRowCountB: Tensor?
        var pairedCapacity = 0
        // F-85 indirect-dispatch args (see `pairedIndirectArgsGateUpBuf`
        // doc) - only populated alongside the paired plan above.
        var pairedIndirectArgsGateUp: MTLBuffer?
        var pairedIndirectArgsDown: MTLBuffer?
        // F-85 idle-tile-cost indirect-dispatch args for the PRODUCTION
        // coop GEMM (see `coopIndirectArgsGateUpBuf` doc) - populated by
        // EITHER the plain `useCoopTilePlan` builder call or the paired
        // lever's "own" builder call below, whichever this call takes
        // (mutually exclusive on `useCoopPaired`).
        var coopIndirectArgsGateUp: MTLBuffer?
        var coopIndirectArgsDown: MTLBuffer?
        if useTilePlan {
            // Host-computable worst-case capacity - no GPU-to-CPU sync (see
            // `Ops.moeTilePlanCapacity` / `Ops.moeTilePlanCapacityBm32`).
            // Plan buffers must start zero-filled: the plan-builder kernel
            // only writes the real `[0, realTileCount)` entries; padding
            // out to `capacity` stays at this zero fill (`tile_expert = 0`
            // - a valid expert id - and `tile_row_count = 0`, which the
            // GEMM kernel's tail mask turns into an inert dispatched tile).
            let capacity =
                useCoopTilePlan
                ? Ops.moeTilePlanCapacityBm32(mTotal: mTotal, nExperts: nExperts)
                : Ops.moeTilePlanCapacity(mTotal: mTotal, nExperts: nExperts)
            // Preallocated, per-instance, grow-only reuse (see
            // `tilePlanBuffers` / the property doc above
            // `tilePlanTileExpertBuf`) - was three fresh `device.makeBuffer`
            // calls every layer, every forward call on this gate's default
            // (bm16-tileplan-or-coop) path, same allocation-churn class the
            // paired lever's buffers were already fixed for.
            let (teBuf, trsBuf, trcBuf) = tilePlanBuffers(capacity: capacity, device: device)
            let tileExpert = Tensor(buffer: teBuf, offset: 0, shape: [capacity], dtype: .u32)
            let tileRowStart = Tensor(buffer: trsBuf, offset: 0, shape: [capacity], dtype: .u32)
            let tileRowCount = Tensor(buffer: trcBuf, offset: 0, shape: [capacity], dtype: .u32)
            if useCoopPaired {
                // "own" plan: same buffers/capacity as the regular coop
                // tile plan, but built by the pairing-aware own builder -
                // it skips the 1..16-row remainders the paired dispatch
                // below covers instead, so the two dispatches never
                // double-write the same row.
                // F-85 idle-tile-cost indirect-dispatch args (see
                // `coopIndirectArgsGateUpBuf` doc) - `tileCountGateUp`/
                // `tileCountDown` below are 1-element views into word 1
                // (byte offset 4) of each buffer; the "own" builder writes
                // the SAME real own-tile count into both.
                let (ownIndirectGateUpBuf, ownIndirectDownBuf) = coopIndirectArgsBuffers(
                    moeIntermediate: moeIntermediate, hidden: hidden, device: device)
                let ownTileCountGateUp = Tensor(
                    buffer: ownIndirectGateUpBuf, offset: 4, shape: [1], dtype: .u32)
                let ownTileCountDown = Tensor(
                    buffer: ownIndirectDownBuf, offset: 4, shape: [1], dtype: .u32)
                // F-85 parallelized builder (see `parallelPlanEnabled`) -
                // default ON, escape hatch BUTTER_MOE_PARALLEL_PLAN=0.
                if self.parallelPlanEnabled {
                    Ops.moeBuildTilePlanBm32OwnParallel(
                        sortedExperts: indices,
                        tileExpert: tileExpert, tileRowStart: tileRowStart,
                        tileRowCount: tileRowCount,
                        tileCountGateUp: ownTileCountGateUp, tileCountDown: ownTileCountDown,
                        mTotal: mTotal, nExperts: nExperts, on: work)
                } else {
                    Ops.moeBuildTilePlanBm32Own(
                        sortedExperts: indices,
                        tileExpert: tileExpert, tileRowStart: tileRowStart,
                        tileRowCount: tileRowCount,
                        tileCountGateUp: ownTileCountGateUp, tileCountDown: ownTileCountDown,
                        mTotal: mTotal, nExperts: nExperts, on: work)
                }
                coopIndirectArgsGateUp = ownIndirectGateUpBuf
                coopIndirectArgsDown = ownIndirectDownBuf

                // Preallocated, per-instance, reused across layers/calls
                // (see `pairedTilePlanBuffers` / the property doc above
                // `pairedTileExpertABuf`) - was six fresh `device.makeBuffer`
                // calls every layer, every forward call (240+ allocs per
                // forward at Qwen3.6-35B-A3B's ~40 MoE layers). Removes
                // that allocation churn regardless, but see `pairedEnabled`'s
                // doc: re-measurement showed this was NOT the regression's
                // dominant cost - the paired GEMM dispatch's fixed
                // worst-case-capacity grid is.
                let pairedCap = Ops.moePairedTilePlanCapacity(nExperts: nExperts)
                let (teaBuf, trsaBuf, trcaBuf, tebBuf, trsbBuf, trcbBuf) =
                    pairedTilePlanBuffers(pairedCap: pairedCap, device: device)
                let tileExpertA = Tensor(buffer: teaBuf, offset: 0, shape: [pairedCap], dtype: .u32)
                let tileRowStartA = Tensor(
                    buffer: trsaBuf, offset: 0, shape: [pairedCap], dtype: .u32)
                let tileRowCountA = Tensor(
                    buffer: trcaBuf, offset: 0, shape: [pairedCap], dtype: .u32)
                let tileExpertB = Tensor(buffer: tebBuf, offset: 0, shape: [pairedCap], dtype: .u32)
                let tileRowStartB = Tensor(
                    buffer: trsbBuf, offset: 0, shape: [pairedCap], dtype: .u32)
                let tileRowCountB = Tensor(
                    buffer: trcbBuf, offset: 0, shape: [pairedCap], dtype: .u32)
                // F-85 indirect-dispatch args (see `pairedIndirectArgsGateUpBuf`
                // doc) - `pairCountGateUp`/`pairCountDown` below are 1-element
                // views into word 1 (byte offset 4) of each buffer; the
                // builder writes the SAME real paired-tile count into both.
                let (indirectGateUpBuf, indirectDownBuf) = pairedIndirectArgsBuffers(
                    moeIntermediate: moeIntermediate, hidden: hidden, device: device)
                let pairCountGateUp = Tensor(
                    buffer: indirectGateUpBuf, offset: 4, shape: [1], dtype: .u32)
                let pairCountDown = Tensor(
                    buffer: indirectDownBuf, offset: 4, shape: [1], dtype: .u32)
                Ops.moeBuildTilePlanBm32Paired(
                    sortedExperts: indices,
                    tileExpertA: tileExpertA, tileRowStartA: tileRowStartA,
                    tileRowCountA: tileRowCountA,
                    tileExpertB: tileExpertB, tileRowStartB: tileRowStartB,
                    tileRowCountB: tileRowCountB,
                    pairCountGateUp: pairCountGateUp, pairCountDown: pairCountDown,
                    mTotal: mTotal, nExperts: nExperts, on: work)
                pairedTileExpertA = tileExpertA
                pairedTileRowStartA = tileRowStartA
                pairedTileRowCountA = tileRowCountA
                pairedTileExpertB = tileExpertB
                pairedTileRowStartB = tileRowStartB
                pairedTileRowCountB = tileRowCountB
                pairedCapacity = pairedCap
                pairedIndirectArgsGateUp = indirectGateUpBuf
                pairedIndirectArgsDown = indirectDownBuf
            } else if useCoopTilePlan {
                // F-85 idle-tile-cost indirect-dispatch args - see
                // `coopIndirectArgsGateUpBuf` doc.
                let (indirectGateUpBuf, indirectDownBuf) = coopIndirectArgsBuffers(
                    moeIntermediate: moeIntermediate, hidden: hidden, device: device)
                let tileCountGateUp = Tensor(
                    buffer: indirectGateUpBuf, offset: 4, shape: [1], dtype: .u32)
                let tileCountDown = Tensor(
                    buffer: indirectDownBuf, offset: 4, shape: [1], dtype: .u32)
                // F-85 parallelized builder (see `parallelPlanEnabled`) -
                // default ON, escape hatch BUTTER_MOE_PARALLEL_PLAN=0. This is
                // the default production coop path's builder call.
                if self.parallelPlanEnabled {
                    Ops.moeBuildTilePlanBm32Parallel(
                        sortedExperts: indices,
                        tileExpert: tileExpert, tileRowStart: tileRowStart,
                        tileRowCount: tileRowCount,
                        tileCountGateUp: tileCountGateUp, tileCountDown: tileCountDown,
                        mTotal: mTotal, nExperts: nExperts, on: work)
                } else {
                    Ops.moeBuildTilePlanBm32(
                        sortedExperts: indices,
                        tileExpert: tileExpert, tileRowStart: tileRowStart,
                        tileRowCount: tileRowCount,
                        tileCountGateUp: tileCountGateUp, tileCountDown: tileCountDown,
                        mTotal: mTotal, nExperts: nExperts, on: work)
                }
                coopIndirectArgsGateUp = indirectGateUpBuf
                coopIndirectArgsDown = indirectDownBuf
            } else {
                // F-85 parallelized builder (see `parallelPlanEnabled`) -
                // bm16 fallback path (coop core off or shape doesn't
                // qualify - see `useCoopTilePlan`).
                if self.parallelPlanEnabled {
                    Ops.moeBuildTilePlanParallel(
                        sortedExperts: indices,
                        tileExpert: tileExpert, tileRowStart: tileRowStart,
                        tileRowCount: tileRowCount,
                        mTotal: mTotal, nExperts: nExperts, on: work)
                } else {
                    Ops.moeBuildTilePlan(
                        sortedExperts: indices,
                        tileExpert: tileExpert, tileRowStart: tileRowStart,
                        tileRowCount: tileRowCount,
                        mTotal: mTotal, nExperts: nExperts, on: work)
                }
            }
            // xGathered is already pre-gathered (step 8 above) - identity
            // row map, same helper the bm16 MPP path uses.
            let idRows = Ops.identityRowsBuffer(minCount: mTotal, device: device)
            tilePlanXRows = Tensor(buffer: idRows, offset: 0, shape: [mTotal], dtype: .u32)
            tilePlanTileExpert = tileExpert
            tilePlanTileRowStart = tileRowStart
            tilePlanTileRowCount = tileRowCount
            tilePlanCapacity = capacity
        }

        // Gate/up/down all route through this same tile-plan-or-bgemm
        // selection - factored into one closure rather than repeating the
        // useCoopPaired / useTilePlan branch at each of the three call
        // sites. Below `pairedMaxMTotal` this issues the own-tiles
        // dispatch (existing coop wrapper, fed the own plan) followed by
        // the pairs dispatch (paired-GEMM wrapper, fed the paired plan)
        // into the SAME output tensor - the two plans partition the
        // sorted-row space with no overlap, so no sync is needed between
        // them beyond Metal's ordinary same-buffer write ordering.
        func dispatchGatherGemm(
            _ input: Tensor, _ weight: Tensor, _ scales: Tensor, _ biases: Tensor,
            nOutLeg: Int, kInLeg: Int, into out: Tensor
        ) {
            if useCoopPaired,
                let tileExpert = tilePlanTileExpert, let tileRowStart = tilePlanTileRowStart,
                let tileRowCount = tilePlanTileRowCount, let xRows = tilePlanXRows,
                let teA = pairedTileExpertA, let trsA = pairedTileRowStartA,
                let trcA = pairedTileRowCountA,
                let teB = pairedTileExpertB, let trsB = pairedTileRowStartB,
                let trcB = pairedTileRowCountB,
                let indirectGateUpBuf = pairedIndirectArgsGateUp,
                let indirectDownBuf = pairedIndirectArgsDown
            {
                // F-85 idle-tile-cost indirect dispatch for the "own" leg
                // (see `coopIndirectEnabled` doc) - this leg's utilization
                // against the shared worst-case capacity is typically
                // WORSE than the plain path's own utilization at the same
                // mTotal (stripping the pairing candidates out shrinks the
                // real own-tile count while the capacity bound stays the
                // same), so this fix matters more here, not less.
                if self.coopIndirectEnabled, let ownIndirectGateUpBuf = coopIndirectArgsGateUp,
                    let ownIndirectDownBuf = coopIndirectArgsDown
                {
                    let ownIndirectArgs = nOutLeg == moeIntermediate ? ownIndirectGateUpBuf : ownIndirectDownBuf
                    Ops.moeGatherDequantGemmInt4CoopTilePlanIndirect(
                        input: input, weight: weight, scales: scales, biases: biases,
                        tileExpert: tileExpert, tileRowStart: tileRowStart, tileRowCount: tileRowCount,
                        xRows: xRows, indirectArgs: ownIndirectArgs, indirectArgsOffset: 0,
                        nOut: nOutLeg, kIn: kInLeg,
                        groupSize: groupSize, on: work, into: out)
                } else {
                    // Escape hatch to the old direct/worst-case-capacity
                    // dispatch, for A/B measurement against the indirect
                    // fix above - see `coopIndirectEnabled`.
                    Ops.moeGatherDequantGemmInt4CoopTilePlan(
                        input: input, weight: weight, scales: scales, biases: biases,
                        tileExpert: tileExpert, tileRowStart: tileRowStart, tileRowCount: tileRowCount,
                        xRows: xRows, capacity: tilePlanCapacity, nOut: nOutLeg, kIn: kInLeg,
                        groupSize: groupSize, on: work, into: out)
                }
                // F-85 indirect dispatch (see `pairedIndirectEnabled` doc) -
                // the root-cause fix for the worst-case-capacity padding
                // `pairedEnabled`'s doc measured. `nOutLeg == moeIntermediate`
                // picks the gate/up-shaped indirect-args triple; the down
                // leg (`nOutLeg == hidden`) uses the other - see
                // `pairedIndirectArgsBuffers`.
                if self.pairedIndirectEnabled {
                    let indirectArgs = nOutLeg == moeIntermediate ? indirectGateUpBuf : indirectDownBuf
                    Ops.moeGatherDequantGemmInt4CoopTilePlanPairedIndirect(
                        input: input, weight: weight, scales: scales, biases: biases,
                        tileExpertA: teA, tileRowStartA: trsA, tileRowCountA: trcA,
                        tileExpertB: teB, tileRowStartB: trsB, tileRowCountB: trcB,
                        xRows: xRows, indirectArgs: indirectArgs, indirectArgsOffset: 0,
                        nOut: nOutLeg, kIn: kInLeg,
                        groupSize: groupSize, on: work, into: out)
                } else {
                    // Escape hatch to the old direct/worst-case-capacity
                    // dispatch, for A/B measurement against the indirect
                    // fix above - see `pairedIndirectEnabled`.
                    Ops.moeGatherDequantGemmInt4CoopTilePlanPaired(
                        input: input, weight: weight, scales: scales, biases: biases,
                        tileExpertA: teA, tileRowStartA: trsA, tileRowCountA: trcA,
                        tileExpertB: teB, tileRowStartB: trsB, tileRowCountB: trcB,
                        xRows: xRows, capacity: pairedCapacity, nOut: nOutLeg, kIn: kInLeg,
                        groupSize: groupSize, on: work, into: out)
                }
            } else if useCoopTilePlan,
                let tileExpert = tilePlanTileExpert, let tileRowStart = tilePlanTileRowStart,
                let tileRowCount = tilePlanTileRowCount, let xRows = tilePlanXRows,
                self.coopIndirectEnabled,
                let indirectGateUpBuf = coopIndirectArgsGateUp,
                let indirectDownBuf = coopIndirectArgsDown
            {
                // F-85 idle-tile-cost indirect dispatch (see
                // `coopIndirectEnabled` doc) - the plain (non-paired) coop
                // path's default. `nOutLeg == moeIntermediate` picks the
                // gate/up-shaped indirect-args triple; the down leg
                // (`nOutLeg == hidden`) uses the other.
                let indirectArgs = nOutLeg == moeIntermediate ? indirectGateUpBuf : indirectDownBuf
                Ops.moeGatherDequantGemmInt4CoopTilePlanIndirect(
                    input: input, weight: weight, scales: scales, biases: biases,
                    tileExpert: tileExpert, tileRowStart: tileRowStart, tileRowCount: tileRowCount,
                    xRows: xRows, indirectArgs: indirectArgs, indirectArgsOffset: 0,
                    nOut: nOutLeg, kIn: kInLeg,
                    groupSize: groupSize, on: work, into: out)
            } else if useTilePlan,
                let tileExpert = tilePlanTileExpert, let tileRowStart = tilePlanTileRowStart,
                let tileRowCount = tilePlanTileRowCount, let xRows = tilePlanXRows
            {
                // Escape hatch (`BUTTER_MOE_COOP_INDIRECT=0`, or the bm16
                // tileplan default when the coop core's shape requirements
                // don't hold - see `useCoopTilePlan`) - old direct/
                // worst-case-capacity dispatch, unchanged.
                let tilePlanGemm =
                    useCoopTilePlan
                    ? Ops.moeGatherDequantGemmInt4CoopTilePlan : Ops.moeGatherDequantGemmInt4TilePlan
                tilePlanGemm(
                    input, weight, scales, biases,
                    tileExpert, tileRowStart, tileRowCount,
                    xRows,
                    tilePlanCapacity, nOutLeg, kInLeg,
                    groupSize, work, out)
            } else {
                bgemm(
                    input, weight, scales, biases,
                    indices, mTotal, nOutLeg, kInLeg, groupSize, work, out)
            }
        }

        dispatchGatherGemm(
            xGathered, stacked.gateWeight, stacked.gateScales, stacked.gateBiases,
            nOutLeg: moeIntermediate, kInLeg: hidden, into: gateOut)
        dispatchGatherGemm(
            xGathered, stacked.upWeight, stacked.upScales, stacked.upBiases,
            nOutLeg: moeIntermediate, kInLeg: hidden, into: upOut)

        // ── 10. SwiGLU fused: silu(gate) * up ────────────────────────────
        // Write into `gateOut` in-place so we don't allocate a third
        // [mTotal, inter] slab (gate + up already live). Down QMM then
        // reads `gateOut` as its activation matrix.
        let inner = Ops.swiglu(gate: gateOut, up: upOut, on: work, into: gateOut)

        // ── 11. Down BGEMM → [mTotal, hidden] ────────────────────────────
        let downOut = Tensor.empty(
            shape: [mTotal, hidden], dtype: dt,
            device: device)
        dispatchGatherGemm(
            inner, stacked.downWeight, stacked.downScales, stacked.downBiases,
            nOutLeg: hidden, kInLeg: moeIntermediate, into: downOut)

        // ── 12. Weighted scatter-sum back to [T, hidden] via fused
        // `iron_moe_unpermute` kernel — ONE dispatch over T·hidden
        // elements instead of mTotal·2 small `Tensor.filled([hidden])`
        // + mul + add per row. At Qwen3.6-A3B T=32 topK=8 mTotal=256
        // that's 512 dispatches → 1 per gate/up/down stage.
        let dtBytes = dt.byteSize
        let outFlat = Tensor.empty(shape: [t * hidden], dtype: dt, device: device)
        let invPermTensor: Tensor
        let weightsTensor: Tensor
        if let dip = d2InvPerm, let dw = d2Weights, dw.dtype == dt {
            invPermTensor = dip
            weightsTensor = dw
        } else {
            let invPermBuf = device.makeBuffer(length: mTotal * 4)
            invPermHost.withUnsafeBytes {
                _ = memcpy(invPermBuf.contents(), $0.baseAddress!, mTotal * 4)
            }
            invPermTensor = Tensor(buffer: invPermBuf, offset: 0, shape: [mTotal], dtype: .u32)
            let wt = Tensor.empty(shape: [mTotal], dtype: dt, device: device)
            switch dt {
            case .f32:
                wt.copyIn(from: weightsTokenOrderHost)
            case .bf16:
                wt.copyIn(
                    from: weightsTokenOrderHost.map { v -> UInt16 in
                        let bits = v.bitPattern
                        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                        return UInt16(rounded >> 16)
                    })
            case .f16:
                wt.copyIn(from: weightsTokenOrderHost.map { Float16($0) })
            default:
                fatalError("MoELayer.decodeMany: unsupported dtype \(dt) for unpermute weights")
            }
            weightsTensor = wt
        }
        Ops.moeUnpermute(
            expertOutputs: downOut, invPerm: invPermTensor,
            topKWeights: weightsTensor, into: outFlat,
            nRows: t, hidden: hidden, k: topK, on: work)
        let _ = dtBytes  // retained for the shared-expert fan-out below

        // ── 13. Optional shared expert — per-row T-loop ─────────────────
        // Shared expert is one always-on per-token SwiGLU. Looping T
        // calls here mirrors today's single-token flow; a batched
        // shared-expert SwiGLU is a small follow-up (gate/up/down all
        // accept callMany inputs).
        if let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj {
            for r in 0 ..< t {
                let hRow = Tensor(
                    buffer: hFlat.buffer,
                    offset: hFlat.offset + r * hidden * dtBytes,
                    shape: [hidden], dtype: dt)
                let sharedOut = swiGLU(
                    hRow, gateProj: sg, upProj: su, downProj: sd,
                    on: work)
                let outRow = Tensor(
                    buffer: outFlat.buffer,
                    offset: outFlat.offset + r * hidden * dtBytes,
                    shape: [hidden], dtype: dt)
                _ = Ops.add(outRow, sharedOut, on: work, into: outRow)
            }
        }

        // Commit without wait; skip when the caller will chain + commit.
        if commitAtEnd {
            work.commit()
        }
        return outFlat
    }

    /// Batched gather BGEMM fast path. T=1 decode: `mTotal = topK`. Each
    /// row of the gathered batch holds the same `[hidden]` activation
    /// vector but is paired with a different expert id via `indices`.
    /// The kernel walks rows in tile order — boundary masking lets it
    /// process the trailing partial tile when `mTotal < BM`. One
    /// dispatch per projection replaces `topK` sequential per-expert
    /// SwiGLU triplets:
    ///   24 (topK=8) dispatches → 3 dispatches (gate / up / down) plus
    ///   element-wise SiLU + mul + scaled scatter-sum (~6 small ops).
    /// Outputs are scaled by the router's combine weights on the CPU
    /// (the same `Tensor.filled([hidden])` trick the serial path uses,
    /// but only once per topK slot — still tiny vs the BGEMM cost).
    ///
    /// Tile selection:
    ///   - `BUTTER_MOE_BGEMM_BM8=1` + `topK ≤ 8` → bm8 MPP kernel
    ///     (BM=8 fills the tile exactly at Qwen3.6-A3B topK=8 decode;
    ///     bm16 would waste 50 % of the trailing tile rows).
    ///     Requires Apple10+ GPU (M5 Max) + macOS 26.2+.
    ///   - otherwise → bm16 path (Ops.moeGatherDequantGemmInt4),
    ///     which itself respects `BUTTER_MOE_BGEMM_MPP=1` for the
    ///     MPP/NAX variant at larger m_total.
    private func batchedSwiGLU(
        _ h: Tensor,
        stacked: StackedInt4Experts,
        routing: MoERouter.Routing,
        on cmd: MTLCommandBuffer,
        device: Device
    ) -> Tensor {
        let topK = routing.indices.count
        let moeIntermediate = stacked.moeIntermediate
        let groupSize = stacked.groupSize
        let dtype = stacked.dtype
        let useBm8 = topK <= 8 && useBm8Env
        // `BUTTER_MOE_M1=1` routes the gather through the scalar m1
        // kernel `iron_moe_gather_qmm_int4` — no MPP / cooperative-tensor
        // overhead. At decode T=1 the cooperative-tensor variants
        // (bm8 / bm16 / bm64) lose ~15 % to descriptor setup cost; the
        // m1 path does a one-output-cell-per-TG dot-product with
        // `simd_sum` reduction and is closer in shape to the production
        // per-expert dequant_gemv path.
        let useM1 = !useBm8 && useM1Env

        // ── Sort topK by expert id ascending (kernel contract) ──
        // Track the original slot so we can apply the right combine
        // weight after the BGEMM. The slot mapping never goes through
        // the GPU, so sorting on the host is cheap.
        let sorted = routing.indices.enumerated().sorted { $0.element < $1.element }
        let sortedExperts = sorted.map { UInt32($0.element) }
        let sortedSlots = sorted.map { $0.offset }

        // ── Materialise the [topK, hidden] activation gather + indices ──
        let xGathered = Tensor.empty(shape: [topK, hidden], dtype: dtype, device: device)
        let inputBytes = hidden * dtype.byteSize
        // h is the same vector for every row at T=1 decode. memcpy from
        // host backing — Tensors here are storage-shared.
        let src = h.buffer.contents().advanced(by: h.offset)
        let dst = xGathered.buffer.contents().advanced(by: xGathered.offset)
        for r in 0 ..< topK {
            dst.advanced(by: r * inputBytes)
                .copyMemory(from: src, byteCount: inputBytes)
        }
        let indices = Tensor.empty(shape: [topK], dtype: .u32, device: device)
        indices.copyIn(from: sortedExperts)

        // For the m1 path we also need CSR `expert_offsets`. Build on
        // host (cheap: nExperts ≤ 256). `expert_offsets[e]` = first row
        // assigned to expert e (or t_rows if expert e is unselected).
        // The kernel does a linear walk over this on each lane.
        var expertOffsetsT: Tensor? = nil
        if useM1 {
            let nExperts = stacked.numExperts
            var csrHost = [UInt32](repeating: UInt32(topK), count: nExperts + 1)
            var firstRowForExpert = [Int](repeating: topK, count: nExperts)
            for (row, e) in sortedExperts.enumerated() {
                let ei = Int(e)
                if firstRowForExpert[ei] == topK { firstRowForExpert[ei] = row }
            }
            var minOffset = topK
            for e in (0 ... nExperts).reversed() {
                if e < nExperts && firstRowForExpert[e] < minOffset {
                    minOffset = firstRowForExpert[e]
                }
                csrHost[e] = UInt32(minOffset)
            }
            let offsetsT = Tensor.empty(shape: [nExperts + 1], dtype: .u32, device: device)
            offsetsT.copyIn(from: csrHost)
            expertOffsetsT = offsetsT
        }

        // ── Gate / up projections: [topK, moeIntermediate] ──
        let gateOut = Tensor.empty(shape: [topK, moeIntermediate], dtype: dtype, device: device)
        let upOut = Tensor.empty(shape: [topK, moeIntermediate], dtype: dtype, device: device)

        if let offsetsT = expertOffsetsT {
            let nExperts = stacked.numExperts
            Ops.moeGatherDequantGemmInt4M1(
                xGathered,
                stacked.gateWeight, stacked.gateScales, stacked.gateBiases,
                offsetsT,
                topK, moeIntermediate, hidden,
                nExperts, groupSize, cmd, gateOut)
            Ops.moeGatherDequantGemmInt4M1(
                xGathered,
                stacked.upWeight, stacked.upScales, stacked.upBiases,
                offsetsT,
                topK, moeIntermediate, hidden,
                nExperts, groupSize, cmd, upOut)
        } else {
            let bgemm = useBm8 ? Ops.moeGatherDequantGemmInt4Bm8 : Ops.moeGatherDequantGemmInt4
            bgemm(
                xGathered,
                stacked.gateWeight, stacked.gateScales, stacked.gateBiases,
                indices,
                topK, moeIntermediate, hidden,
                groupSize, cmd, gateOut)
            bgemm(
                xGathered,
                stacked.upWeight, stacked.upScales, stacked.upBiases,
                indices,
                topK, moeIntermediate, hidden,
                groupSize, cmd, upOut)
        }

        // ── SwiGLU activation: silu(gate) * up — fused into one dispatch ──
        // In-place into `gateOut` — drop the third [topK, inter] slab.
        let inner = Ops.swiglu(gate: gateOut, up: upOut, on: cmd, into: gateOut)

        // ── Down projection: [topK, hidden] ──
        let downOut = Tensor.empty(shape: [topK, hidden], dtype: dtype, device: device)
        if let offsetsT = expertOffsetsT {
            let nExperts = stacked.numExperts
            Ops.moeGatherDequantGemmInt4M1(
                inner,
                stacked.downWeight, stacked.downScales, stacked.downBiases,
                offsetsT,
                topK, hidden, moeIntermediate,
                nExperts, groupSize, cmd, downOut)
        } else {
            let bgemm = useBm8 ? Ops.moeGatherDequantGemmInt4Bm8 : Ops.moeGatherDequantGemmInt4
            bgemm(
                inner,
                stacked.downWeight, stacked.downScales, stacked.downBiases,
                indices,
                topK, hidden, moeIntermediate,
                groupSize, cmd, downOut)
        }

        // ── Weighted scatter-sum back to a single [hidden] vector ──
        // For each routed slot, scale the corresponding row of downOut
        // by `routing.weights[slot]` and accumulate. The Tensor.filled
        // broadcast scalar trick mirrors the serial path's per-slot
        // weight application.
        var acc: Tensor?
        for (sortedIdx, originalSlot) in sortedSlots.enumerated() {
            // Row view into downOut at the sorted position.
            let row = Tensor(
                buffer: downOut.buffer,
                offset: downOut.offset + sortedIdx * hidden * dtype.byteSize,
                shape: [hidden], dtype: dtype)
            let w = Tensor.filled(
                routing.weights[originalSlot],
                shape: [hidden], dtype: dtype, device: device)
            let scaled = Ops.mul(row, w, on: cmd)
            acc = acc.map { Ops.add($0, scaled, on: cmd) } ?? scaled
        }
        return acc!
    }

    /// One SwiGLU FFN: down(silu(gate(x)) * up(x)).
    ///
    /// `silu(gate) * up` runs as a single fused `iron_swiglu` dispatch
    /// instead of two launches (silu → mul). Saves one elementwise
    /// kernel + one full-tensor RMW per call. At Qwen3.6-A3B decode T=1
    /// the per-expert intermediate is [moeIntermediate=768] so the win
    /// is small per dispatch, but the loop runs 8 experts × ≤40 MoE
    /// layers per token, so the per-token saving compounds.
    private func swiGLU(
        _ x: Tensor,
        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        let g = gateProj(x, on: cmd)
        let u = upProj(x, on: cmd)
        let inner = Ops.swiglu(gate: g, up: u, on: cmd)
        return downProj(inner, on: cmd)
    }

    /// GPU MoE router decode path. Replaces the
    /// `cmd.commit + waitUntilCompleted` sync with a full-GPU pipeline:
    /// router writes topK indices + weights to GPU buffers; per-slot
    /// indexed qmms (gate/up phase + down phase) batch onto shared
    /// encoders; final accumulation via `scalarFMAChain8` (or the fused
    /// `moeDownSwigluAccumInt4Chain8` kernel when constraints allow).
    ///
    /// Predicated by `decode` callers — the layer must have stackedInt4
    /// experts with matching dtype, router must be softmax-then-topK
    /// with normTopKProb=true / no expertBias / topK=8.
    private func decodeGPURouter(
        h: Tensor, logitsTensor: Tensor,
        stacked: StackedInt4Experts,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // Lazy-init per-instance scratches.
        if routerIndicesScratch == nil
            || routerIndicesScratch!.elementCount != router.topK
        {
            routerIndicesScratch = Tensor.empty(
                shape: [router.topK], dtype: .u32, device: device)
            routerWeightsScratch = Tensor.empty(
                shape: [router.topK], dtype: h.dtype, device: device)
        }
        if accumulatorScratch == nil
            || accumulatorScratch!.dtype != h.dtype
            || accumulatorScratch!.elementCount != hidden
        {
            accumulatorScratch = Tensor.empty(
                shape: [hidden], dtype: h.dtype, device: device)
        }
        let outDim = stacked.moeIntermediate
        for slot in 0 ..< router.topK {
            while expertGScratches.count <= slot {
                expertGScratches.append(
                    Tensor.empty(shape: [outDim], dtype: h.dtype, device: device))
                expertUScratches.append(
                    Tensor.empty(shape: [outDim], dtype: h.dtype, device: device))
                expertInnerScratches.append(
                    Tensor.empty(shape: [outDim], dtype: h.dtype, device: device))
                expertOutScratches.append(
                    Tensor.empty(shape: [hidden], dtype: h.dtype, device: device))
            }
        }

        // GPU router: logits → topK indices + weights (still on cmd).
        Ops.moeRouterTopK(
            logits: logitsTensor,
            indicesOut: routerIndicesScratch!,
            weightsOut: routerWeightsScratch!,
            nExperts: router.nExperts, k: router.topK,
            normTopkProb: router.normTopKProb,
            on: cmd)

        // Batched per-expert indexed qmms on ONE encoder per phase.
        // Gate (8 calls) + up (8 calls) share inDim/outDim/groupSize →
        // one Many encoder. Down (8 calls) has different out_dim →
        // second Many encoder.
        var expertIdxScratches: [Tensor] = []
        expertIdxScratches.reserveCapacity(router.topK)
        for slot in 0 ..< router.topK {
            expertIdxScratches.append(
                Tensor(
                    buffer: routerIndicesScratch!.buffer,
                    offset: routerIndicesScratch!.offset + slot * 4,
                    shape: [1], dtype: .u32))
        }

        // Phase 1a: 16 calls (8 gate + 8 up) on one encoder.
        var ws: [Tensor] = []
        var ss: [Tensor] = []
        var bs: [Tensor] = []
        var ins: [Tensor] = []
        var idxs: [Tensor] = []
        var outs0: [Tensor] = []
        ws.reserveCapacity(router.topK * 2)
        ss.reserveCapacity(router.topK * 2)
        bs.reserveCapacity(router.topK * 2)
        ins.reserveCapacity(router.topK * 2)
        idxs.reserveCapacity(router.topK * 2)
        outs0.reserveCapacity(router.topK * 2)
        for slot in 0 ..< router.topK {
            ws.append(stacked.gateWeight)
            ss.append(stacked.gateScales)
            bs.append(stacked.gateBiases)
            ins.append(h)
            idxs.append(expertIdxScratches[slot])
            outs0.append(expertGScratches[slot])
            ws.append(stacked.upWeight)
            ss.append(stacked.upScales)
            bs.append(stacked.upBiases)
            ins.append(h)
            idxs.append(expertIdxScratches[slot])
            outs0.append(expertUScratches[slot])
        }
        Ops.dequantGemvInt4ExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss, biasesStacked: bs,
            inputs: ins, expertIndices: idxs, outputs: outs0,
            groupSize: stacked.groupSize, on: cmd)

        // Fuse phase 1b (swigluMany), phase 2 (8 down indexed-gemvs),
        // and phase 3 (scalarFMAChain8) into ONE dispatch via
        // `iron_moe_down_swiglu_accum_int4_chain8`. Per-slot `inner` is
        // staged in 3 KiB threadgroup memory, never spilling to global
        // memory. Kernel constraint: moeIntermediate ≤ 768 + groupSize
        // == 64. Qwen3.6-A3B has exactly 768 so this matches; fall back
        // to the 3-dispatch chain for any larger configs.
        let gs = Array(expertGScratches.prefix(router.topK))
        let us = Array(expertUScratches.prefix(router.topK))
        let acc = accumulatorScratch!
        if router.topK == 8
            && stacked.moeIntermediate <= 768
            && stacked.groupSize == 64
        {
            Ops.moeDownSwigluAccumInt4Chain8(
                gates: gs, ups: us,
                expertIndices: routerIndicesScratch!,
                slotWeights: routerWeightsScratch!,
                weightsStacked: stacked.downWeight,
                scalesStacked: stacked.downScales,
                biasesStacked: stacked.downBiases,
                output: acc,
                inDim: stacked.moeIntermediate,
                outDim: hidden,
                groupSize: stacked.groupSize,
                on: cmd)
        } else {
            // Legacy 3-dispatch chain (Phase 1b + 2 + 3).
            let inners = Array(expertInnerScratches.prefix(router.topK))
            Ops.swigluMany(gates: gs, ups: us, outs: inners, on: cmd)
            var dws: [Tensor] = []
            var dss: [Tensor] = []
            var dbs: [Tensor] = []
            var dins: [Tensor] = []
            var didxs: [Tensor] = []
            var douts: [Tensor] = []
            dws.reserveCapacity(router.topK)
            dss.reserveCapacity(router.topK)
            dbs.reserveCapacity(router.topK)
            dins.reserveCapacity(router.topK)
            didxs.reserveCapacity(router.topK)
            douts.reserveCapacity(router.topK)
            for slot in 0 ..< router.topK {
                dws.append(stacked.downWeight)
                dss.append(stacked.downScales)
                dbs.append(stacked.downBiases)
                dins.append(expertInnerScratches[slot])
                didxs.append(expertIdxScratches[slot])
                douts.append(expertOutScratches[slot])
            }
            Ops.dequantGemvInt4ExpertIndexedMany(
                weightsStacked: dws, scalesStacked: dss, biasesStacked: dbs,
                inputs: dins, expertIndices: didxs, outputs: douts,
                groupSize: stacked.groupSize, on: cmd)
            var scalars: [Tensor] = []
            scalars.reserveCapacity(router.topK)
            for slot in 0 ..< router.topK {
                scalars.append(
                    Tensor(
                        buffer: routerWeightsScratch!.buffer,
                        offset: routerWeightsScratch!.offset
                            + slot * h.dtype.byteSize,
                        shape: [1], dtype: h.dtype))
            }
            let outs = Array(expertOutScratches.prefix(router.topK))
            Ops.scalarFMAChain8(
                scalars: scalars, values: outs, out: acc, on: cmd)
        }

        // Commit cmd without wait — preserves the caller contract
        // (caller starts a fresh cmd for residual-add / shared expert).
        // The win is the ABSENT `waitUntilCompleted` that the CPU-sync
        // path needed before this branch.
        //
        // Contract: the returned tensor aliases `accumulatorScratch`,
        // which the GPU is still writing through the in-flight `cmd`.
        // Callers MUST not re-invoke `decode` on this layer until that
        // cmd completes — a second call would re-enter `decodeGPURouter`
        // and start overwriting the scratch while the GPU is still
        // reading from it. Per-token serial dispatch in the model loop
        // satisfies this; concurrent decodes on a shared MoELayer
        // instance do not.
        cmd.commit()
        return acc
    }
}

// ─── Integration note for MoE-bearing families ───────────────────────
//
// A hybrid family (NemotronH, Granite4, Jamba, Qwen3.5 MoE,
// Gemma4 MoE) places an `MoELayer` in its `[any DecoderLayer]` stack
// exactly where a dense MLP layer would otherwise sit. To wire one up:
//
//  1. Build the router with the architecture's gating math:
//       let router = MoERouter(nExperts: cfg.numExperts,
//                               topK: cfg.numExpertsPerToken,
//                               gatingMode: .softmaxThenTopK,   // Qwen
//                               normTopKProb: cfg.normTopkProb)
//     Use `.topKThenSoftmax` for Granite4-style checkpoints.
//
//  2. Expert weights ship stacked as `[nExperts, outDim, inDim]`
//     tensors. Slice per-expert with `Tensor.slicedRows` and wrap:
//       let gateProj = (0..<nExperts).map { e in
//           AnyLinear(Linear(weight:
//               stackedGate.slicedRows(start: e, count: 1)
//                          .reshaped(to: [moeIntermediate, hidden])))
//       }
//     `up_proj` / `down_proj` follow the same shape. For
//     affine-quantised experts use `QuantizedLinear` per slice (the
//     stacked weight / scales / biases each slice along dim 0).
//
//  3. Optionally build the shared expert (Granite4). Pass all
//     three shared projections to the `MoELayer` initialiser, or none.
//
//  4. The MoE layer's residual add + pre-norm stay in the host layer
//     (same as the dense MLP): `h + moeLayer.decode(postAttnNorm(h)…)`.
//
//  5. CRITICAL — `MoELayer.decode` commits the command buffer it is
//     given. The host model's decode loop must refresh `workCmd` after
//     the call, e.g.:
//       h = moeLayer.decode(h, position: pos, cache: caches[i],
//                           cmd: workCmd, device: device)
//       workCmd = device.makeCommandBuffer()   // ← fresh buffer
//     This is the same commit-and-replace shape `InspectTap` uses for
//     layer-boundary dumps.
