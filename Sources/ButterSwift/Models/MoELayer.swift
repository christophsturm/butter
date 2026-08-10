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

/// Launch-time A/B flags read on the per-token / per-layer MoE decode
/// paths. Hoisted to cached statics: these are process-lifetime
/// switches, and even the getenv(3) form costs a libc environ scan per
/// read x ~30 reads x 40 layers per token. None of these names is
/// toggled mid-process by tests (verified against every setenv/unsetenv
/// site under Tests/); mid-process-toggled flags must stay on live
/// `ButterEnv[...]` reads.
private enum MoEEnvFlags {
    static let devplan2 = ButterEnv["DEVPLAN2"] != "0"
    static let nvfp4Gather = ButterEnv["BUTTER_NVFP4_GATHER"] == "1"
    static let moePhaseTiming = ButterEnv["BUTTER_LAGUNA_MOE_PHASE_TIMING"] == "1"
    static let serialPathB = ButterEnv["BUTTER_NVFP4_SERIAL_PATHB"] == "1"
    static let dualQueueOn = ButterEnv["BUTTER_LAGUNA_DUAL_QUEUE"] != "0"
    static let noShared = ButterEnv["BUTTER_NVFP4_NO_SHARED"] != nil
    static let sharedQmmOn = ButterEnv["BUTTER_NVFP4_SHARED_QMM"] != "0"
    static let useGatherQmm = ButterEnv["BUTTER_NVFP4_USE_GATHER_QMM"] == "1"
    static let gatherMppEnv = ButterEnv["BUTTER_NVFP4_USE_GATHER_MPP"]
    static let gatherNax =
        ButterEnv["BUTTER_NVFP4_GATHER_NAX"] == "1"
        || ButterEnv["BUTTER_NVFP4_GATHER_NAX"]?.lowercased() == "true"
    static let expertGroupQmm = ButterEnv["BUTTER_NVFP4_EXPERT_GROUP_QMM"] == "1"
    static let gatherSort = ButterEnv["BUTTER_NVFP4_GATHER_SORT"] == "1"
    static let expertGroupAsync = ButterEnv["BUTTER_NVFP4_EXPERT_GROUP_ASYNC"] != "0"
    static let mppForceAllBf16 = ButterEnv["BUTTER_NVFP4_MPP_BF16"] == "1"
    // W53: late-layer nvfp4 fast-path extension. Numerically measured
    // (BUTTER_LAGUNA_MOE_MAGNITUDE_SCAN, real laguna-xs-2.1-nvfp4
    // prefill, two unrelated prompts, BUTTER_LAGUNA_LASTROW_PREFILL=0
    // so every row of every late layer is actually exercised): L37 and
    // L39's swiglu-inner / down-GEMM activations top out ~1970 / ~1110
    // (33-59x under half's 65504 ceiling — same order of magnitude as
    // the f16Inner lever's existing "~800, 80x headroom" calibration).
    // L38 is a structural outlier: swiglu-inner ~446,362, down-output
    // ~461,888 — reproducibly IDENTICAL (bit-for-bit) across a 6-token
    // and a 461-token prompt of unrelated content, i.e. a fixed
    // "massive activation" feature, not routing noise — 6.8-7.0x OVER
    // the f16 ceiling. That's exactly the ~1e6 half-coop-tile overflow
    // class flagged at W50 (post-L38 residual maxAbs measured ~1.1M
    // here, matching W50's 1,130,496 reference point). So L38 alone
    // stays on the f32 fallback; L37 and L39 move to the fast
    // bf16/nvfp4 path. Default ON. Escape to the pre-W53 threshold
    // (bf16MaxLayer=36, no late layers fast, matches the shipped
    // behavior through #120) with BUTTER_NVFP4_MPP_LATE_LAYER_FAST=0.
    static let mppLateLayerFast = ButterEnv["BUTTER_NVFP4_MPP_LATE_LAYER_FAST"] != "0"
    static let mppBf16MaxLayer =
        Int(ButterEnv["BUTTER_NVFP4_MPP_BF16_MAX_LAYER"] ?? "")
        ?? (mppLateLayerFast ? 39 : 36)
    // Layers forced to stay on the f32 fallback even when they fall
    // within bf16MaxLayer's range — L38's measured f16 overflow (see
    // above). Comma-separated layer indices; empty string forces every
    // layer up to bf16MaxLayer fast (NOT recommended — reintroduces the
    // W50 failure mode at L38 on this model).
    static let mppBf16ExcludeLayers: Set<Int> = {
        if let raw = ButterEnv["BUTTER_NVFP4_MPP_BF16_EXCLUDE_LAYERS"] {
            return Set(raw.split(separator: ",").compactMap { Int($0) })
        }
        return mppLateLayerFast ? [38] : []
    }()
    // W54: for layers excluded from the fast bf16-MPP path above (L38 by
    // default — see mppBf16ExcludeLayers), route through the T33 "steel"
    // structure NVFP4 gather kernel (Nvfp4GatherSteel.swift /
    // butter_nvfp4_gather_steel_bgemm_*.metal) instead of falling all the
    // way back to the f32-promote chain. Steel stages BOTH the
    // dequantized weight and the activation as native `bfloat` in
    // per-lane MMA fragments (plain simdgroup_matrix, no
    // mpp::tensor_ops, no threadgroup half staging) — see that file's
    // header: "keeps the full f32-width exponent range, so that
    // overflow class structurally cannot occur". That is the correct
    // fix for L38's ~446K swiglu-inner / ~462K down-output (both far
    // past half's 65504 ceiling, trivially inside bf16's ~3.4e38
    // range) — NOT the half-coop-tile MPP kernels the fast path above
    // uses (bf16 MPP) nor the ExpertGather V2 path (explicit bf16→f16
    // cast of the A operand), both of which stage through half and
    // would reproduce the exact overflow this layer was excluded for.
    // Steel was previously opt-in only (BUTTER_NVFP4_GATHER_STEEL=1)
    // because a whole-model A/B lost (T35: -6.5%, see
    // LagunaNvfp4MppGather.swift's header) — but here it only has to
    // beat the f32 fallback for ONE layer's 3 GEMMs, not the shipped
    // bf16-MPP fast path model-wide, so that prior loss doesn't apply.
    // Default ON for exactly the excluded-layer set. Escape to the
    // pre-W54 W53 behavior (f32 fallback for excluded layers) with
    // BUTTER_NVFP4_MPP_STEEL_EXCLUDED=0.
    static let mppSteelForExcludedLayers =
        ButterEnv["BUTTER_NVFP4_MPP_STEEL_EXCLUDED"] != "0"
    static let mppNoSharedCast = ButterEnv["BUTTER_NVFP4_MPP_NO_SHARED_CAST"] == "1"
    static let mppFusedGateUp = ButterEnv["BUTTER_NVFP4_MPP_FUSED_GATEUP"] != "0"
    static let mppSerialGateUp = ButterEnv["BUTTER_NVFP4_MPP_SERIAL_GATEUP"] == "1"
    static let pathBMode = ButterEnv["BUTTER_LAGUNA_PATHB"] ?? "full"
    static let pathBFused = ButterEnv["BUTTER_LAGUNA_PATHB_FUSED"] != "0"
    // W53 numerics-first measurement: dump per-stage f32 max|activation|
    // for the f32-fallback (useF32Promote) MoE branch, gated to late
    // layers only. Diagnostic-only, off by default, forces sync
    // read-back (breaks pipelining) so never left on in perf runs.
    // See benchmarks/laguna-xs/README.md W53 (or the campaign compendium)
    // for the decision this measurement gates: whether layers 37-39 can
    // safely leave the f32 fallback for the fast nvfp4/f16-device-direct
    // expert path.
    static let moeMagnitudeScan = ButterEnv["BUTTER_LAGUNA_MOE_MAGNITUDE_SCAN"] == "1"
    static let moeMagnitudeScanMinLayer =
        Int(ButterEnv["BUTTER_LAGUNA_MOE_MAGNITUDE_SCAN_MIN_LAYER"] ?? "") ?? 34
}

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
    /// T27 Stage 2: pre-transposed router weight for the steel MMA GEMM
    /// fast path (`LagunaSteelGemm`). The router GEMM (`gate.callMany`
    /// in `decodeManyNvfp4`'s prefill route phase) is plain BF16
    /// (`[nExperts, hidden]`, e.g. `[256, 2048]` on Laguna XS, both
    /// %64==0) and rode the naive `Ops.gemm` path until now, same class
    /// of gap T26 found in attention's Q/K/V/O projections. `nil` when
    /// `gate` isn't a plain bf16 Linear (falls back automatically via
    /// `LagunaSteelGemm.eligible`). See benchmarks/laguna-xs/README.md T27.
    ///
    /// F-85: also `nil` when `LagunaMppGemm.staticEligible` already
    /// covers the router GEMM's shape -- MPP takes the raw untransposed
    /// weight directly, so the steel copy is skipped whenever every real
    /// (t%64==0) router call would route through MPP instead. See
    /// `LagunaMppGemm.swift`'s header.
    let gateWeightT: Tensor?

    /// W57: set by `decodeMany`/`decodeManyNvfp4` immediately after each
    /// call to record whether a `fusedResidual` passed to that call was
    /// actually folded into the MoE combine tail (`Ops.moeUnpermuteTail`)
    /// vs left for the caller to add itself. Side-channel rather than a
    /// tuple return so `decodeMany`'s signature/behavior stays unchanged
    /// for its other callers (Qwen3x*, MTP head) — only the one Laguna
    /// call site (`LagunaText.forwardMulti`'s `.moe` case) reads this,
    /// immediately after its `decodeMany` call, before any other call on
    /// this instance can race it (prefill is sequential per layer). See
    /// `LagunaFusedMoeTail` for the full fusion rationale/gate.
    public private(set) var lastResidualFused: Bool = false

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

    /// Stacked NVFP4 experts (Laguna board quant). Scales are u8 E4M3,
    /// blockSize 16, optional global scale (default 1.0).
    public struct StackedNvfp4Experts: Sendable {
        public let gateWeight: Tensor  // [E, mid, hidden/8] u32
        public let gateScales: Tensor  // [E, mid, hidden/16] u8
        public let upWeight: Tensor
        public let upScales: Tensor
        public let downWeight: Tensor  // [E, hidden, mid/8] u32
        public let downScales: Tensor
        public let numExperts: Int
        public let moeIntermediate: Int
        public let hidden: Int
        public let blockSize: Int
        public let global: Float
        public let dtype: DType  // activation dtype

        public init(
            gateWeight: Tensor, gateScales: Tensor,
            upWeight: Tensor, upScales: Tensor,
            downWeight: Tensor, downScales: Tensor,
            numExperts: Int, moeIntermediate: Int, hidden: Int,
            blockSize: Int = 16, global: Float = 1.0, dtype: DType
        ) {
            self.gateWeight = gateWeight
            self.gateScales = gateScales
            self.upWeight = upWeight
            self.upScales = upScales
            self.downWeight = downWeight
            self.downScales = downScales
            self.numExperts = numExperts
            self.moeIntermediate = moeIntermediate
            self.hidden = hidden
            self.blockSize = blockSize
            self.global = global
            self.dtype = dtype
        }
    }
    public let stackedNvfp4Experts: StackedNvfp4Experts?

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
    /// Single-dispatch fusion of the counting-sort three-pass chain (W61
    /// Stage-1 fusion #1 A/B gate). Raw env string cached at init;
    /// `sortPlanFusedEnabled` below is the single place that interprets
    /// it. Default OFF (opt-in via `BUTTER_MOE_FUSED_SORT_PLAN=1`) until
    /// the wired A/B gate lands.
    let sortPlanFusedEnvRaw: String?
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
    /// Path B (sigmoid+bias) score scratches — `[nExperts]`.
    private var unbiasedScoreScratch: Tensor?
    private var biasedScoreScratch: Tensor?
    private var pathBScaleVecScratch: Tensor?
    /// BUTTER_DEBUG_ROUTER_SETS (butter#108): raw (pre-sigmoid) router
    /// GEMV logits + expert bias tensors from this layer's most recent
    /// decode-step routing dispatch, stashed only when the flag is on.
    /// `LagunaRouterTop8Fuse`'s fused kernel (the default) computes
    /// sigmoid+bias inline and never writes `unbiasedScoreScratch` /
    /// `biasedScoreScratch`, so those can't be read back for a debug
    /// dump — these two are the raw ingredients instead, always valid
    /// regardless of which route-chain variant fired. See
    /// `debugRouterSnapshot()` below and `RouterSetsDebug.swift`.
    private var debugRouterLogits: Tensor?
    private var debugRouterBias: Tensor?

    /// BUTTER_DEBUG_ROUTER_SETS (butter#108): host readback of this
    /// layer's most recent decode-step top-K expert selection. Callers
    /// (`LagunaModel.forwardFromTokenTensor`'s debug-only synchronous
    /// per-layer branch) must have already ensured the command buffer
    /// that encoded the routing dispatch has completed — this performs
    /// no synchronization of its own, matching `LogitMarginDebug`'s
    /// "caller guarantees host-readable" precondition. Returns `nil`
    /// when the flag is off, this layer hasn't routed yet this call (a
    /// dense FFN layer, or a decode branch that predates the stash
    /// sites), or the scratch shapes don't line up.
    public func debugRouterSnapshot() -> (indices: [Int], scores: [Float])? {
        guard RouterSetsDebug.enabled,
            let idxT = routerIndicesScratch,
            let logitsT = debugRouterLogits,
            let biasT = debugRouterBias
        else { return nil }
        let indices = idxT.toArray(as: UInt32.self).map { Int($0) }
        let logits = logitsT.toFloatArray()
        let bias = biasT.toFloatArray()
        let scores: [Float] = indices.map { id in
            guard id >= 0, id < logits.count, id < bias.count else { return .nan }
            let sig: Float = 1.0 / (1.0 + exp(-logits[id]))
            return sig + bias[id]
        }
        return (indices, scores)
    }
    /// Optional row-concat [gate; up] NVFP4 banks for fused shared SwiGLU.
    private var fusedSharedGateUpWeight: Tensor?
    private var fusedSharedGateUpScales: Tensor?
    private var fusedSharedGateUpGlobal: Float = 1.0
    private var fusedSharedGateUpAttempted = false
    /// T28 nine-slot merged MoE dispatch (routed + shared expert in one
    /// threadgroup family). Cached eligibility + raw shared-down NVFP4
    /// fields (mirrors `ensureFusedSharedGateUp`'s lazy-attempt pattern).
    private var nineSlotAttempted = false
    private var nineSlotSharedDownWeight: Tensor?
    private var nineSlotSharedDownScales: Tensor?
    private var nineSlotSharedDownGlobal: Float = 1.0
    private var sharedMidScratch: Tensor?
    /// T43: set by `decodeNineSlot` when the nine-slot down-reduce dispatch
    /// folded the post-MoE residual add into its own lane-0 epilogue
    /// (`BUTTER_LAGUNA_DOWN_RESIDUAL_FOLD=1`, nine-slot bf16 path only).
    /// Reset at the top of every `decodeFoldingResidual` call so a decode
    /// that takes a different internal branch (GPU router, Path B int4,
    /// CPU fallback — none of which fold) reports `false` and the caller
    /// falls back to its own residual add. Single-threaded per layer
    /// instance, same assumption the other decode scratches above rely on.
    private var lastResidualFolded = false
    /// Layer index in the stack (for late-layer MPP f32 promote). `-1` = unknown.
    public let layerIndex: Int
    /// Prefill NVFP4 MoE matmul scratches (grow-only; avoids per-layer
    /// Tensor.empty of gate/up/down/out for every MoE layer × every chunk).
    private var prefGateOutScratch: Tensor?
    private var prefUpOutScratch: Tensor?
    private var prefDownOutScratch: Tensor?
    private var prefOutFlatScratch: Tensor?
    /// Prefill router / sort-plan scratches (grow-only).
    private var prefUnbiasedScratch: Tensor?
    private var prefBiasedScratch: Tensor?
    private var prefIndicesTokScratch: Tensor?
    private var prefWeightsTokScratch: Tensor?
    private var prefSortedExpertsScratch: Tensor?
    private var prefSortedSourceScratch: Tensor?
    private var prefInvPermScratch: Tensor?
    /// Token-order gather source ids: src[r·k+s] = r (cached by mTotal).
    private var prefTokenOrderSrcScratch: Tensor?
    private var prefTokenOrderSrcMTotal: Int = 0
    /// Prefill shared-expert scratches (grow-only): fused gate‖up, mid, down.
    private var prefSharedGuScratch: Tensor?
    private var prefSharedGateScratch: Tensor?
    private var prefSharedUpScratch: Tensor?
    private var prefSharedMidScratch: Tensor?
    private var prefSharedDownScratch: Tensor?
    /// Optional GPU-resident expert bias for Path B (same dtype as acts).
    public let expertBiasGPU: Tensor?

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

    /// T21 hygiene: host-sorted MoE routing transients for `decodeMany`
    /// (int4) / `decodeManyNvfp4`: gather-index (source token per sorted
    /// row) and expert-id (routed expert per sorted row) buffers, plus the
    /// unpermute-scatter inverse-permutation buffer. These were a fresh
    /// `device.makeBuffer` pair/single every forward call on EVERY MoE
    /// layer (i.e. every decode token and every prefill chunk) until this
    /// cache was added, same grow-only per-instance reuse pattern as
    /// `tilePlanTileExpertBuf` above, same cross-call GPU-ordering safety
    /// argument (this file's per-layer command-buffer chaining already
    /// depends on committed-order execution), and same reason growing is
    /// always safe: every call `memcpy`s its own full `mTotal` prefix
    /// before any dispatch reads it, so a larger backing buffer never
    /// leaks a previous call's stale tail into a smaller current call.
    /// `decodeMany` and `decodeManyNvfp4` are mutually exclusive per
    /// instance (an instance is either int4- or NVFP4-stacked, decided at
    /// load time), so sharing one set of buffers between them is safe,
    /// same sharing argument as `coopIndirectArgsGateUpBuf` above.
    ///
    /// Being per-instance stored properties, these are also independently
    /// kept alive by `MoELayer`'s own lifetime (not by any one command
    /// buffer's retain): a required precondition, not just a perf nicety,
    /// for `Device.unretainedCommandBuffersEnabled` to ever be safe to
    /// flip on for this file's `commitAtEnd`-chained (commit-without-wait)
    /// dispatches. See the audit in the T21 hygiene PR description.
    private var gatherIdxPoolBuf: MTLBuffer?
    private var expertIdsPoolBuf: MTLBuffer?
    private var gatherIdxPoolCapacity = 0
    private var invPermPoolBuf: MTLBuffer?
    private var invPermPoolCapacity = 0

    /// Lazily allocates (grow-only) and returns the gather-index +
    /// expert-id routing buffers. See the property doc above
    /// `gatherIdxPoolBuf` for the reuse design.
    private func gatherRoutingBuffers(
        mTotal: Int, device: Device
    ) -> (gatherIdx: MTLBuffer, expertIds: MTLBuffer) {
        if gatherIdxPoolBuf == nil || gatherIdxPoolCapacity < mTotal {
            gatherIdxPoolBuf = device.makeBuffer(length: mTotal * 4)
            expertIdsPoolBuf = device.makeBuffer(length: mTotal * 4)
            gatherIdxPoolCapacity = mTotal
        }
        return (gatherIdxPoolBuf!, expertIdsPoolBuf!)
    }

    /// Lazily allocates (grow-only) and returns the unpermute inverse-
    /// permutation buffer. See the property doc above `gatherIdxPoolBuf`
    /// for the reuse design.
    private func invPermRoutingBuffer(mTotal: Int, device: Device) -> MTLBuffer {
        if invPermPoolBuf == nil || invPermPoolCapacity < mTotal {
            invPermPoolBuf = device.makeBuffer(length: mTotal * 4)
            invPermPoolCapacity = mTotal
        }
        return invPermPoolBuf!
    }

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
        stackedInt4Experts: StackedInt4Experts? = nil,
        stackedNvfp4Experts: StackedNvfp4Experts? = nil,
        expertBiasGPU: Tensor? = nil,
        layerIndex: Int = -1,
        dtype: DType = .bf16,
        device: Device = .shared
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
        self.gateWeightT = LagunaMppGemm.staticEligible(gate, dtype: dtype)
            ? nil : LagunaSteelGemm.weightT(gate, dtype: dtype, device: device)
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
        if let s = stackedNvfp4Experts {
            precondition(s.numExperts == router.nExperts)
            precondition(s.hidden == hidden)
        }
        self.stackedNvfp4Experts = stackedNvfp4Experts
        self.layerIndex = layerIndex
        if let bias = expertBiasGPU {
            precondition(
                bias.elementCount == router.nExperts,
                "MoELayer: expertBiasGPU has \(bias.elementCount) entries, "
                    + "router expects \(router.nExperts)")
        }
        self.expertBiasGPU = expertBiasGPU
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
        self.sortPlanFusedEnvRaw = env["BUTTER_MOE_FUSED_SORT_PLAN"]
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

    /// W61 Stage-1 fusion #1: single-dispatch fusion of the counting-sort
    /// three-pass chain (`Ops.moeSortPlanFused`,
    /// `moe_route_sort_plan_fused.rs`). Same stable order as
    /// `moeSortPlanCounting` (byte-identical `--dump-prefill-logits`,
    /// oracle-checked against both a CPU stable sort and the three-pass
    /// chain's own GPU output across uniform/Zipf/empty-expert/production
    /// shapes — see `moe_route_sort_plan_fused_correctness.rs`), ONE
    /// dispatch instead of three. Default OFF pending the wired A/B gate;
    /// `BUTTER_MOE_FUSED_SORT_PLAN=1` opts in. Only takes effect where
    /// `sortPlanCountingEnabled` would otherwise fire (the DEVPLAN2
    /// device-sorted-plan path) — this flag selects between the fused
    /// kernel and the three-pass chain, not between counting-sort and the
    /// original O(mTotal^2) kernel.
    public var sortPlanFusedEnabled: Bool {
        sortPlanFusedEnvRaw == "1"
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
        MoEEnvFlags.devplan2
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
        // Prefer stacked banks when present (NVFP4 / int4 Path B) — the
        // per-expert arrays may be shared stubs and would under/over-count.
        if let s = stackedNvfp4Experts {
            out.append(("switch_mlp.gate_proj.weight", s.gateWeight))
            out.append(("switch_mlp.gate_proj.scales", s.gateScales))
            out.append(("switch_mlp.up_proj.weight", s.upWeight))
            out.append(("switch_mlp.up_proj.scales", s.upScales))
            out.append(("switch_mlp.down_proj.weight", s.downWeight))
            out.append(("switch_mlp.down_proj.scales", s.downScales))
        } else if let s = stackedInt4Experts {
            out.append(("switch_mlp.gate_proj.weight", s.gateWeight))
            out.append(("switch_mlp.gate_proj.scales", s.gateScales))
            out.append(("switch_mlp.gate_proj.biases", s.gateBiases))
            out.append(("switch_mlp.up_proj.weight", s.upWeight))
            out.append(("switch_mlp.up_proj.scales", s.upScales))
            out.append(("switch_mlp.up_proj.biases", s.upBiases))
            out.append(("switch_mlp.down_proj.weight", s.downWeight))
            out.append(("switch_mlp.down_proj.scales", s.downScales))
            out.append(("switch_mlp.down_proj.biases", s.downBiases))
        } else {
            for (e, proj) in gateProj.enumerated() {
                for (k, v) in proj.parameters() {
                    out.append(("experts.\(e).gate_proj.\(k)", v))
                }
            }
            for (e, proj) in upProj.enumerated() {
                for (k, v) in proj.parameters() {
                    out.append(("experts.\(e).up_proj.\(k)", v))
                }
            }
            for (e, proj) in downProj.enumerated() {
                for (k, v) in proj.parameters() {
                    out.append(("experts.\(e).down_proj.\(k)", v))
                }
            }
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
    /// `DecoderLayer` conformance: exact 4-arg signature required for
    /// protocol witness matching (a defaulted 5th parameter does NOT
    /// satisfy the protocol requirement, so `decodeWithPrecomputedLogits`
    /// below is a separate entry point, not an overload of this one).
    public func decode(
        _ h: Tensor, position: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        decodeImpl(
            h, position: position, cache: cache,
            cmd: cmd, device: device, precomputedLogits: nil)
    }

    /// T28 lever 2: when the caller already computed the router logits (the
    /// residual+RMSNorm+router fold in `LagunaLayer.forward` reuses the
    /// normed row it just staged in threadgroup memory to also run the
    /// router GEMV, see `LagunaResidualRmsRouterFuse`), call this instead
    /// of `decode` to skip the redundant `gate(h, on: cmd)` dispatch.
    public func decodeWithPrecomputedLogits(
        _ h: Tensor, position: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device,
        precomputedLogits: Tensor
    ) -> Tensor {
        decodeImpl(
            h, position: position, cache: cache,
            cmd: cmd, device: device, precomputedLogits: precomputedLogits, residual: nil)
    }

    /// T43: down-residual-fold entry point. Not part of `DecoderLayer`
    /// protocol conformance (unlike `decode`/`decodeWithPrecomputedLogits`
    /// above, so its signature is free to carry the extra `residual` arg
    /// and tuple return) — callers that know their `ffn` case is `.moe`
    /// concretely (e.g. `LagunaLayer.forward`) can call it directly.
    /// Attempts to fold `residual` into the nine-slot down-reduce kernel's
    /// epilogue (`BUTTER_LAGUNA_DOWN_RESIDUAL_FOLD=1`, nine-slot-eligible
    /// shapes only); when `residualFolded` comes back `false` (flag off,
    /// shapes don't qualify, or a non-nine-slot decode branch fired),
    /// `output` is the plain MoE ffn output exactly as `decode` would
    /// return, and the caller must still add `residual` itself.
    public func decodeFoldingResidual(
        _ h: Tensor, position: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device,
        precomputedLogits: Tensor?,
        residual: Tensor,
        foldInto: Tensor? = nil
    ) -> (output: Tensor, residualFolded: Bool) {
        lastResidualFolded = false
        let output = decodeImpl(
            h, position: position, cache: cache,
            cmd: cmd, device: device, precomputedLogits: precomputedLogits,
            residual: residual, foldInto: foldInto)
        return (output, lastResidualFolded)
    }

    private func decodeImpl(
        _ h: Tensor, position _: Int,
        cache _: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device,
        precomputedLogits: Tensor?,
        residual: Tensor? = nil,
        foldInto: Tensor? = nil
    ) -> Tensor {
        return Profile.signpost("moe.decode") { () -> Tensor in
        precondition(
            h.elementCount == hidden,
            "MoELayer.decode: input has \(h.elementCount) elements, expected hidden \(hidden)")

        // ── 1. Gate gemv on the caller's command buffer (skipped when the
        //      caller already fused it into the residual+RMSNorm dispatch).
        let logitsTensor = precomputedLogits ?? gate(h, on: cmd)

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

        // Path B (Laguna / Hy3-style): sigmoid + bias top-k on GPU.
        if self.gpuRouterEnabled,
            router.gatingMode == .sigmoidBiasedTopK,
            let biasGPU = expertBiasGPU,
            biasGPU.dtype == h.dtype,
            router.topK == 8
        {
            if let nv = stackedNvfp4Experts, nv.dtype == h.dtype {
                return decodeGPUPathBNvfp4(
                    h: h, logitsTensor: logitsTensor,
                    stacked: nv, expertBiasGPU: biasGPU,
                    cmd: cmd, device: device, residual: residual,
                    foldInto: foldInto)
            }
            if let stacked = stackedInt4Experts, stacked.dtype == h.dtype {
                return decodeGPUPathB(
                    h: h, logitsTensor: logitsTensor,
                    stacked: stacked, expertBiasGPU: biasGPU,
                    cmd: cmd, device: device)
            }
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

    /// T-batched MoE (`[T·hidden]` flat in/out). Stacked int4 → gather
    /// BGEMM ×3 + unpermute; else per-token `decode`. DEVPLAN2 keeps the
    /// path on `cmd` (no mid-commit). `commitAtEnd: false` lets the caller
    /// chain more work before commit.
    public func decodeMany(
        _ hFlat: Tensor, t: Int,
        cmd: MTLCommandBuffer, device: Device,
        commitAtEnd: Bool = true,
        fusedResidual: Tensor? = nil
    ) -> Tensor {
        precondition(
            hFlat.elementCount == t * hidden,
            "MoELayer.decodeMany: hFlat size \(hFlat.elementCount) ≠ T·hidden = \(t * hidden)")
        precondition(t > 0, "MoELayer.decodeMany: T must be positive")
        lastResidualFused = false

        let dt = hFlat.dtype

        // NVFP4 quality prefill: skip host/device sort plan entirely and
        // run token-order Path B (batched gate + per-token experts).
        // Gather remains behind BUTTER_NVFP4_GATHER=1.
        // NAX-class spike is Ops-level only (BUTTER_NVFP4_GATHER_NAX=1) and
        // rides the default Path-B MPP gather — do NOT force the experimental
        // useGather branch (that skips default MPP entirely).
        if let nv = stackedNvfp4Experts, nv.dtype == dt,
            !MoEEnvFlags.nvfp4Gather
        {
            return decodeManyNvfp4(
                hFlat: hFlat, t: t, stacked: nv,
                gatherIdx: nil, sortedExperts: nil,
                invPerm: nil, weightsTok: nil,
                routings: [], sortedExpertsHost: [],
                sourceTokensHost: [], invPermHost: [],
                weightsTokenOrderHost: [],
                fusedResidual: fusedResidual,
                cmd: cmd, device: device, commitAtEnd: commitAtEnd,
                useDevicePlan: true)
        }

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
        // Token-order device plan (DEVPLAN2): GPU softmax router + no host
        // sort. Only valid under the softmax topK=8 envelope — sigmoid+bias
        // (Laguna / Hy3 Path B) stays on the host concurrent route path.
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
        // Laguna / Hy3 Path B: GPU sigmoid+bias top-k (then host sort plan).
        let usePathBGPURouter =
            self.gpuRouterEnabled
            && router.gatingMode == .sigmoidBiasedTopK
            && expertBiasGPU != nil
            && expertBiasGPU!.dtype == gateLogitsAll.dtype
            && router.topK == 8
        let useDevPlan2 = prefillChainsOnCallerCmd && useGPURouter
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
            // W61: fused single-dispatch plan (opt-in, see
            // `sortPlanFusedEnabled`) takes precedence over the three-pass
            // chain when enabled.
            if self.sortPlanFusedEnabled {
                Ops.moeSortPlanFused(
                    topkIds: idsFlat, sortedExperts: se, sourceTokens: st, invPerm: ip,
                    mTotal: mTot, k: topK, nExperts: nExperts, blockSize: 256, on: cmd)
            } else if self.sortPlanCountingEnabled {
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
        } else if usePathBGPURouter, let biasGPU = expertBiasGPU {
            // Device Path B plan (Hy3-style): sigmoid+bias → top-k → sort plan
            // on GPU, no host logit/index readback. Sets d2* for gather.
            let mTot = t * topK
            let scoreDt = gateLogitsAll.dtype
            let flatN = t * nExperts
            let unbiased = Tensor.empty(shape: [flatN], dtype: scoreDt, device: device)
            let biased = Tensor.empty(shape: [flatN], dtype: scoreDt, device: device)
            Ops.moeSigmoidBiasRows(
                logits: gateLogitsAll.reshaped(to: [flatN]),
                bias: biasGPU,
                unbiased: unbiased, biased: biased,
                tRows: t, nExperts: nExperts, on: cmd)
            let indicesBuf = Tensor.empty(shape: [mTot], dtype: .u32, device: device)
            let weightsBuf = Tensor.empty(shape: [mTot], dtype: scoreDt, device: device)
            Ops.moeRouterTopKBiasedMany(
                scoreBiased: biased, scoreUnbiased: unbiased,
                indicesOut: indicesBuf, weightsOut: weightsBuf,
                t: t, nExperts: nExperts, k: topK, on: cmd)
            let scale = router.routedScalingFactor
            if scale != 1.0 {
                let scaleVec = Tensor.filled(
                    scale, shape: [mTot], dtype: scoreDt, device: device)
                _ = Ops.mul(weightsBuf, scaleVec, on: cmd, into: weightsBuf)
            }
            let se = Tensor.empty(shape: [mTot], dtype: .u32, device: device)
            let st = Tensor.empty(shape: [mTot], dtype: .u32, device: device)
            let ip = Tensor.empty(shape: [mTot], dtype: .u32, device: device)
            // W61: fused single-dispatch plan (opt-in, see
            // `sortPlanFusedEnabled`) takes precedence over the three-pass
            // chain when enabled. This is Laguna's Path B (sigmoid+bias)
            // route.
            if self.sortPlanFusedEnabled {
                Ops.moeSortPlanFused(
                    topkIds: indicesBuf, sortedExperts: se, sourceTokens: st, invPerm: ip,
                    mTotal: mTot, k: topK, nExperts: nExperts, blockSize: 256, on: cmd)
            } else if self.sortPlanCountingEnabled {
                Ops.moeSortPlanCounting(
                    topkIds: indicesBuf, sortedExperts: se, sourceTokens: st, invPerm: ip,
                    mTotal: mTot, k: topK, nExperts: nExperts, blockSize: 256, on: cmd)
            } else {
                Ops.moeSortPlan(
                    topkIds: indicesBuf, sortedExperts: se, sourceTokens: st, invPerm: ip,
                    mTotal: mTot, k: topK, on: cmd)
            }
            d2Source = st
            d2InvPerm = ip
            d2Indices = se
            d2Weights = weightsBuf  // token-order weights (pre-sort)
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

        // ── 5. NVFP4 stacked gather, else int4, else per-token decode ──
        if let nv = stackedNvfp4Experts, nv.dtype == dt {
            return decodeManyNvfp4(
                hFlat: hFlat, t: t, stacked: nv,
                gatherIdx: d2Source, sortedExperts: d2Indices,
                invPerm: d2InvPerm, weightsTok: d2Weights,
                routings: routings, sortedExpertsHost: sortedExpertsHost,
                sourceTokensHost: sourceTokensHost, invPermHost: invPermHost,
                weightsTokenOrderHost: weightsTokenOrderHost,
                fusedResidual: fusedResidual,
                cmd: cmd, device: device, commitAtEnd: commitAtEnd,
                useDevicePlan: d2Indices != nil)
        }
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
        // Device plans (softmax DEVPLAN2 or Path B d2*) reuse `cmd`;
        // host-router paths already committed it.
        let useDevicePlan = useDevPlan2 || d2Indices != nil
        let work = useDevicePlan ? cmd : device.makeCommandBuffer()
        let moeIntermediate = stacked.moeIntermediate
        let groupSize = stacked.groupSize

        // ── 7. Gather index + expert ids ─────────────────────────────────
        let gatherIdxTensor: Tensor
        let indices: Tensor
        if let di = d2Indices, let ds = d2Source {
            gatherIdxTensor = ds
            indices = di
        } else {
            let (gatherIdxBuf, expertIdsBuf) = gatherRoutingBuffers(mTotal: mTotal, device: device)
            sourceTokensHost.withUnsafeBytes {
                _ = memcpy(gatherIdxBuf.contents(), $0.baseAddress!, mTotal * 4)
            }
            gatherIdxTensor = Tensor(buffer: gatherIdxBuf, offset: 0, shape: [mTotal], dtype: .u32)
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
        // Tileplan needs device-sorted experts (d2Indices) + no host wait.
        // Path B (sigmoid) device plan is quality-broken with tileplan on
        // Laguna affine4 — keep bm16 gather for Path B; softmax DEVPLAN2
        // keeps tileplan when enabled.
        let pathBDevicePlan = usePathBGPURouter && d2Indices != nil
        let useTilePlan =
            useDevicePlan && !pathBDevicePlan && !useBm64 && !useBm8
            && self.tilePlanEnabled
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
            let invPermBuf = invPermRoutingBuffer(mTotal: mTotal, device: device)
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

    /// Prefill MoE for stacked NVFP4 (board quant).
    ///
    /// **Default (quality):** token-order Path B — one batched `gate.callMany`,
    /// then per-token `decodeGPUPathBNvfp4` (same math as decode). Shared
    /// Path B scratches are reused serially on one CB (Metal orders
    /// sequential encoders). This is the only multi-token path proven
    /// Paris-clean so far.
    ///
    /// **Experimental:** `BUTTER_NVFP4_GATHER=1` → sorted expert-indexed /
    /// MPP gather (still quality-broken; keep for A/B only).
    private func decodeManyNvfp4(
        hFlat: Tensor, t: Int, stacked: StackedNvfp4Experts,
        gatherIdx: Tensor?, sortedExperts: Tensor?,
        invPerm: Tensor?, weightsTok: Tensor?,
        routings: [MoERouter.Routing],
        sortedExpertsHost: [UInt32], sourceTokensHost: [UInt32],
        invPermHost: [UInt32], weightsTokenOrderHost: [Float],
        fusedResidual: Tensor? = nil,
        cmd: MTLCommandBuffer, device: Device,
        commitAtEnd: Bool, useDevicePlan: Bool
    ) -> Tensor {
        let dt = hFlat.dtype
        let topK = router.topK
        let mTotal = t * topK
        let moeIntermediate = stacked.moeIntermediate
        let dtBytes = dt.byteSize
        var work = useDevicePlan ? cmd : device.makeCommandBuffer()
        // T25 diagnosis: split one MoE layer's prefill wall-clock into
        // route / sort / gate+up-GEMM / down-GEMM / tail(unpermute+shared)
        // phases via forced commit+wait checkpoints. Diagnostic-only:
        // serializes the layer (like BUTTER_LAGUNA_TIMING's prefDepth=1)
        // and is off by default (zero behavior change when unset).
        // See benchmarks/laguna-xs/README.md T25.
        let moePhaseTiming =
            MoEEnvFlags.moePhaseTiming
        var moePhaseTimes: [(String, Double)] = []
        var moePhaseT0 = CFAbsoluteTimeGetCurrent()
        func moePhaseCheckpoint(_ label: String) {
            guard moePhaseTiming else { return }
            if work.status != .committed && work.status != .completed
                && work.status != .error
            {
                work.commit()
            }
            work.waitUntilCompleted()
            let ms = (CFAbsoluteTimeGetCurrent() - moePhaseT0) * 1000
            moePhaseTimes.append((label, ms))
            work = device.makeCommandBuffer()
            moePhaseT0 = CFAbsoluteTimeGetCurrent()
        }
        // W53: forced sync read-back of one f32 activation tensor's
        // max|value| (+ nan/inf census), stderr-logged. Off by default
        // (MoEEnvFlags.moeMagnitudeScan); when on, scoped to
        // layerIndex >= moeMagnitudeScanMinLayer so it's cheap even for
        // a full-model prefill run. Breaks command-buffer pipelining by
        // design — diagnostic only, never in a perf measurement.
        func moeMagScan(_ label: String, _ tensor: Tensor) {
            guard MoEEnvFlags.moeMagnitudeScan,
                layerIndex >= MoEEnvFlags.moeMagnitudeScanMinLayer
            else { return }
            if work.status != .committed && work.status != .completed
                && work.status != .error
            {
                work.commit()
            }
            work.waitUntilCompleted()
            let vals = tensor.toFloatArray()
            let nan = vals.filter { $0.isNaN }.count
            let inf = vals.filter { $0.isInfinite }.count
            let mx = vals.filter { $0.isFinite }.map(abs).max() ?? -1
            fputs(
                "[moe-mag-scan] L\(layerIndex) \(label) maxAbs=\(mx) "
                    + "nan=\(nan) inf=\(inf) n=\(vals.count)\n", stderr)
            work = device.makeCommandBuffer()
        }
        let hRows = hFlat.reshaped(to: [t, hidden])
        if prefOutFlatScratch == nil
            || prefOutFlatScratch!.dtype != dt
            || prefOutFlatScratch!.elementCount < t * hidden
        {
            prefOutFlatScratch = Tensor.empty(
                shape: [t * hidden], dtype: dt, device: device)
        }
        let outFlat = Tensor(
            buffer: prefOutFlatScratch!.buffer,
            offset: prefOutFlatScratch!.offset,
            shape: [t * hidden], dtype: dt)

        // ── Quality path: token-order concurrent Path B ──────────────
        // Same math as decodeGPUPathBNvfp4, but:
        //   • one gate GEMM for all T
        //   • GPU sigmoid+bias top-k many
        //   • concurrent expert-indexed GEMV over T·topK slots (token order)
        //   • per-token weighted sum (no sort / unpermute)
        //   • shared expert via batched QMM
        // Default: concurrent token-order Path B (quality matches serial
        // cos≈1.0 after FMA+per-token shared; QMM shared / unpermute were
        // the quality bugs). Escape serial: BUTTER_NVFP4_SERIAL_PATHB=1
        // Experimental sorted gather: BUTTER_NVFP4_GATHER=1
        // Note: default Path B (this `!useGather` branch) already hosts MPP
        // gather for T≥4. BUTTER_NVFP4_GATHER_NAX=1 only swaps the MPP body
        // via Ops.nvfp4MoeGatherPrefersNax — leave useGather alone.
        let useGather =
            MoEEnvFlags.nvfp4Gather
        let useSerialPathB =
            MoEEnvFlags.serialPathB
        // "pre" isolates whatever attention/norm work the caller already
        // encoded onto this same command buffer before MoE started (this
        // is the FIRST checkpoint of the layer's CB, so it is NOT free;
        // don't misread a later "route" bucket as router-only cost unless
        // this checkpoint exists to soak up the backlog).
        moePhaseCheckpoint("pre")
        if !useGather {
            // F-85: MPP tensor-ops GEMM fast path first for the router
            // projection (hidden -> nExperts, e.g. 2048 -> 256 on Laguna
            // XS -- `outDim%64==0`/`inDim%32==0` satisfies MPP's looser
            // tile contract), T27's steel MMA GEMM second, naive
            // `gate.callMany` last. Falls back automatically when
            // ineligible (non-bf16, unaligned shapes, T=1 decode, or
            // BUTTER_LAGUNA_MPP_GEMM=0 / BUTTER_LAGUNA_ATTN_STEEL_GEMM=0).
            let logitsAll: Tensor
            if let mpp = LagunaMppGemm.tryMppGemm(
                gate, x: hRows, t: t, outDim: router.nExperts, inDim: hidden,
                on: work, device: device)
            {
                logitsAll = mpp
            } else if let gateWeightT,
                LagunaSteelGemm.eligible(
                    t: t, outDim: router.nExperts, inDim: hidden, dtype: hRows.dtype)
            {
                logitsAll = LagunaSteelGemm.callMany(
                    input: hRows, weightT: gateWeightT, t: t, outDim: router.nExperts,
                    inDim: hidden, on: work, device: device)
            } else {
                logitsAll = gate.callMany(hRows, t: t, on: work, device: device)
            }
            if useSerialPathB || expertBiasGPU == nil || expertBiasGPU!.dtype != dt
                || topK != 8
            {
                if let biasGPU = expertBiasGPU, biasGPU.dtype == dt, topK == 8 {
                    for r in 0 ..< t {
                        let hRow = Tensor(
                            buffer: hFlat.buffer,
                            offset: hFlat.offset + r * hidden * dtBytes,
                            shape: [hidden], dtype: dt)
                        let logRow = Tensor(
                            buffer: logitsAll.buffer,
                            offset: logitsAll.offset + r * router.nExperts * dtBytes,
                            shape: [router.nExperts], dtype: dt)
                        let rowOut = decodeGPUPathBNvfp4(
                            h: hRow, logitsTensor: logRow,
                            stacked: stacked, expertBiasGPU: biasGPU,
                            cmd: work, device: device)
                        let outRow = Tensor(
                            buffer: outFlat.buffer,
                            offset: outFlat.offset + r * hidden * dtBytes,
                            shape: [hidden], dtype: dt)
                        Ops.copy(rowOut, into: outRow, on: work)
                    }
                } else {
                    for r in 0 ..< t {
                        let hRow = Tensor(
                            buffer: hFlat.buffer,
                            offset: hFlat.offset + r * hidden * dtBytes,
                            shape: [hidden], dtype: dt)
                        let rowOut = decode(
                            hRow, position: 0,
                            cache: StatelessLayerCache(),
                            cmd: work, device: device)
                        let outRow = Tensor(
                            buffer: outFlat.buffer,
                            offset: outFlat.offset + r * hidden * dtBytes,
                            shape: [hidden], dtype: dt)
                        Ops.copy(rowOut, into: outRow, on: work)
                    }
                }
            } else if let biasGPU = expertBiasGPU {
                // Token-order concurrent Path B (quality target = serial).
                // Kick dual-queue shared ASAP (depends only on hRows) so it
                // overlaps router + sort + MPP. Filled later into outFlat.
                var prefillSharedOut: Tensor?
                var prefillSharedDone: UInt64?
                let dualPrefillEarly =
                    MoEEnvFlags.dualQueueOn
                    && !MoEEnvFlags.noShared
                    && MoEEnvFlags.sharedQmmOn
                    && sharedGateProj != nil
                if dualPrefillEarly,
                    let sg = sharedGateProj, let su = sharedUpProj,
                    let sd = sharedDownProj
                {
                    let cmdB = device.makeCommandBufferB()
                    let vStart = device.nextSyncValue()
                    work.encodeSignalEvent(device.syncEvent, value: vStart)
                    cmdB.encodeWaitForEvent(device.syncEvent, value: vStart)
                    if let sOut = prefillSharedExpertNvfp4(
                        hRows: hRows, t: t, gate: sg, up: su, down: sd,
                        dtype: dt, on: cmdB, device: device)
                    {
                        prefillSharedOut = sOut
                        let vDone = device.nextSyncValue()
                        cmdB.encodeSignalEvent(device.syncEvent, value: vDone)
                        cmdB.commit()
                        prefillSharedDone = vDone
                    }
                }

                let nExperts = router.nExperts
                let flatN = t * nExperts
                if prefUnbiasedScratch == nil
                    || prefUnbiasedScratch!.dtype != dt
                    || prefUnbiasedScratch!.elementCount < flatN
                {
                    prefUnbiasedScratch = Tensor.empty(
                        shape: [flatN], dtype: dt, device: device)
                    prefBiasedScratch = Tensor.empty(
                        shape: [flatN], dtype: dt, device: device)
                }
                let unbiased = Tensor(
                    buffer: prefUnbiasedScratch!.buffer,
                    offset: prefUnbiasedScratch!.offset,
                    shape: [flatN], dtype: dt)
                let biased = Tensor(
                    buffer: prefBiasedScratch!.buffer,
                    offset: prefBiasedScratch!.offset,
                    shape: [flatN], dtype: dt)
                Ops.moeSigmoidBiasRows(
                    logits: logitsAll.reshaped(to: [flatN]),
                    bias: biasGPU,
                    unbiased: unbiased, biased: biased,
                    tRows: t, nExperts: nExperts, on: work)
                if prefIndicesTokScratch == nil
                    || prefIndicesTokScratch!.elementCount < mTotal
                {
                    prefIndicesTokScratch = Tensor.empty(
                        shape: [mTotal], dtype: .u32, device: device)
                }
                if prefWeightsTokScratch == nil
                    || prefWeightsTokScratch!.dtype != dt
                    || prefWeightsTokScratch!.elementCount < mTotal
                {
                    prefWeightsTokScratch = Tensor.empty(
                        shape: [mTotal], dtype: dt, device: device)
                }
                let indicesTok = Tensor(
                    buffer: prefIndicesTokScratch!.buffer,
                    offset: prefIndicesTokScratch!.offset,
                    shape: [mTotal], dtype: .u32)
                let weightsTok = Tensor(
                    buffer: prefWeightsTokScratch!.buffer,
                    offset: prefWeightsTokScratch!.offset,
                    shape: [mTotal], dtype: dt)
                Ops.moeRouterTopKBiasedMany(
                    scoreBiased: biased, scoreUnbiased: unbiased,
                    indicesOut: indicesTok, weightsOut: weightsTok,
                    t: t, nExperts: nExperts, k: topK, on: work)
                let scale = router.routedScalingFactor
                if scale != 1.0 {
                    let scaleVec = Tensor.filled(
                        scale, shape: [mTotal], dtype: dt, device: device)
                    _ = Ops.mul(weightsTok, scaleVec, on: work, into: weightsTok)
                }
                moePhaseCheckpoint("route")

                // Expert GEMVs / QMM:
                //   • Expert-group dense QMM (default T≥4): sort by expert, one
                //     nvfp4Qmm per expert run → shares weight loads across tokens.
                //     Big prefill BW win vs concurrent GEMV (reloads every slot).
                //   • Concurrent expert-indexed GEMV (default T<4 / decode).
                //   • Non-MPP gather QMM: BUTTER_NVFP4_USE_GATHER_QMM=1
                //   • MPP gather: BUTTER_NVFP4_USE_GATHER_MPP=1 (still all-NaN).
                // Escape group QMM: BUTTER_NVFP4_EXPERT_GROUP_QMM=0
                // Grow-only prefill matmul scratches (per MoELayer instance).
                let midNeed = mTotal * moeIntermediate
                let hidNeed = mTotal * hidden
                if prefGateOutScratch == nil
                    || prefGateOutScratch!.dtype != dt
                    || prefGateOutScratch!.elementCount < midNeed
                {
                    prefGateOutScratch = Tensor.empty(
                        shape: [mTotal, moeIntermediate], dtype: dt, device: device)
                    prefUpOutScratch = Tensor.empty(
                        shape: [mTotal, moeIntermediate], dtype: dt, device: device)
                }
                if prefDownOutScratch == nil
                    || prefDownOutScratch!.dtype != dt
                    || prefDownOutScratch!.elementCount < hidNeed
                {
                    prefDownOutScratch = Tensor.empty(
                        shape: [mTotal, hidden], dtype: dt, device: device)
                }
                let gateOut = Tensor(
                    buffer: prefGateOutScratch!.buffer,
                    offset: prefGateOutScratch!.offset,
                    shape: [mTotal, moeIntermediate], dtype: dt)
                let upOut = Tensor(
                    buffer: prefUpOutScratch!.buffer,
                    offset: prefUpOutScratch!.offset,
                    shape: [mTotal, moeIntermediate], dtype: dt)
                let downOut = Tensor(
                    buffer: prefDownOutScratch!.buffer,
                    offset: prefDownOutScratch!.offset,
                    shape: [mTotal, hidden], dtype: dt)
                // Prefill MoE matmul (quality-gated):
                //   • MPP gather default ON for T>=4 (f32-promoted bf16 path).
                //     Escape: BUTTER_NVFP4_USE_GATHER_MPP=0
                //   • Non-MPP gather QMM: BUTTER_NVFP4_USE_GATHER_QMM=1
                //   • Expert-group dense QMM: BUTTER_NVFP4_EXPERT_GROUP_QMM=1
                //   • Concurrent eidx: T<4 or MPP disabled
                let useGatherQmm =
                    MoEEnvFlags.useGatherQmm
                let mppEnv =
                    MoEEnvFlags.gatherMppEnv
                // GATHER_NAX rides default MPP; force MPP on even for T<4.
                let naxOn =
                    MoEEnvFlags.gatherNax
                let useGatherMpp =
                    !useGatherQmm
                    && mppEnv != "0"
                    && (naxOn || mppEnv == "1" || t >= 4)
                let useExpertGroupQmm =
                    !useGatherQmm && !useGatherMpp
                    && MoEEnvFlags.expertGroupQmm
                // Sorted plan for group QMM / MPP.
                var sortedExperts: Tensor? = nil
                var sortedSource: Tensor? = nil
                var sortedInvPerm: Tensor? = nil
                // W52 byte-hunt: when the f32Promote branch below skips
                // the castF32ToBf16->downOut materialise (see
                // LagunaF32PromoteCombineFuse), it stashes the raw f32
                // down-projection output here so the combine call site
                // can read it directly instead of the (unwritten)
                // downOut bf16 buffer.
                var pendingCombineF32: Tensor? = nil
                // Sort when MPP, expert-group, or gather-QMM with SORT=1.
                let sortGather =
                    MoEEnvFlags.gatherSort
                if useGatherMpp || useExpertGroupQmm
                    || (useGatherQmm && sortGather)
                {
                    if prefSortedExpertsScratch == nil
                        || prefSortedExpertsScratch!.elementCount < mTotal
                    {
                        prefSortedExpertsScratch = Tensor.empty(
                            shape: [mTotal], dtype: .u32, device: device)
                        prefSortedSourceScratch = Tensor.empty(
                            shape: [mTotal], dtype: .u32, device: device)
                        prefInvPermScratch = Tensor.empty(
                            shape: [mTotal], dtype: .u32, device: device)
                    }
                    let se = Tensor(
                        buffer: prefSortedExpertsScratch!.buffer,
                        offset: prefSortedExpertsScratch!.offset,
                        shape: [mTotal], dtype: .u32)
                    let st = Tensor(
                        buffer: prefSortedSourceScratch!.buffer,
                        offset: prefSortedSourceScratch!.offset,
                        shape: [mTotal], dtype: .u32)
                    let ip = Tensor(
                        buffer: prefInvPermScratch!.buffer,
                        offset: prefInvPermScratch!.offset,
                        shape: [mTotal], dtype: .u32)
                    // MPP path: default-ON counting-sort (F-85, same gate as
                    // DEVPLAN2). The prior classic-only forcing here dated
                    // from ccbbd6fd (#76, Jul 31) — counting-sort was
                    // correlated with full-model NaNs on the MPP gather-QMM
                    // consumer at that time. Root-caused: not a bug in the
                    // sort itself (its isolated GPU-correctness suite —
                    // deterministic repeat-dispatch, Zipf-skewed routing,
                    // production-scale uniform, block-size-non-multiple,
                    // zero-count-expert edge cases — all pass, and the two
                    // kernels write a byte-identical permutation for a given
                    // topk_ids input). The NaN traced back to the MPP NVFP4
                    // gather-QMM kernel it fed, since fixed by iron's
                    // expert-aligned A-device-direct gather QMM (register-
                    // direct masked C store, vectorized NVFP4 decode — all
                    // merged after ccbbd6fd). Reverified: `--dump-prefill-logits`
                    // cos=1.0 / maxdiff=0.0 vs classic sort at T=512 and
                    // T=2048, byte-identical 256-tok greedy "The capital of
                    // France is" generation. Escape hatch unchanged:
                    // BUTTER_MOE_COUNTING_SORT=0 reverts to classic sort
                    // everywhere (this path + DEVPLAN2).
                    // W61: fused single-dispatch plan (opt-in, see
                    // `sortPlanFusedEnabled`) takes precedence over the
                    // three-pass chain when enabled; same gate scope as
                    // `useCounting` above (this path + DEVPLAN2).
                    let useFused = self.sortPlanFusedEnabled
                    let useCounting = self.sortPlanCountingEnabled
                    if useFused {
                        Ops.moeSortPlanFused(
                            topkIds: indicesTok, sortedExperts: se, sourceTokens: st,
                            invPerm: ip, mTotal: mTotal, k: topK, nExperts: nExperts,
                            blockSize: 256, on: work)
                    } else if useCounting {
                        Ops.moeSortPlanCounting(
                            topkIds: indicesTok, sortedExperts: se, sourceTokens: st,
                            invPerm: ip, mTotal: mTotal, k: topK, nExperts: nExperts,
                            blockSize: 256, on: work)
                    } else {
                        Ops.moeSortPlan(
                            topkIds: indicesTok, sortedExperts: se, sourceTokens: st,
                            invPerm: ip, mTotal: mTotal, k: topK, on: work)
                    }
                    sortedExperts = se
                    sortedSource = st
                    sortedInvPerm = ip
                }
                moePhaseCheckpoint("sort")
                if useExpertGroupQmm,
                    let se = sortedExperts, let st = sortedSource
                {
                    // Expert-group dense QMM dual-CB pipe:
                    //   work  : router + sort (already encoded) — NO gather yet
                    //   wait  : host segment scan needs sorted expert ids only
                    //   qmmCmd: gather + concurrent gate/up/down NAX/reduction
                    //           QMM (nvfp4QmmMany) + unpermute + shared add
                    //
                    // Moving gather after the wait keeps sort-only latency on
                    // the host fence and pipelines gather→QMM on one CB with
                    // no mid-layer GPU bubble. Dual-queue shared (kicked
                    // earlier on queue B) overlaps the wait; we consume it
                    // via syncEvent instead of re-running shared on qmmCmd
                    // (avoids racing prefShared* scratches).
                    //
                    // Escape / A-B: BUTTER_NVFP4_EXPERT_GROUP_ASYNC=0 restores
                    // the old gather-before-wait ordering (same math).
                    // Default = async pipe ON whenever expert-group is on.
                    let asyncPipe =
                        MoEEnvFlags.expertGroupAsync

                    let xGathered: Tensor
                    if asyncPipe {
                        // Sort only on `work`; gather lands on qmmCmd below.
                        if work.status != .committed && work.status != .completed
                            && work.status != .error
                        {
                            work.commit()
                        }
                        work.waitUntilCompleted()
                        xGathered = Tensor.empty(
                            shape: [mTotal, hidden], dtype: dt, device: device)
                    } else {
                        // Legacy: gather before fence (extra GPU work under wait).
                        xGathered = Ops.gather(
                            table: hRows, tokenIds: st, on: work)
                        if work.status != .committed && work.status != .completed
                            && work.status != .error
                        {
                            work.commit()
                        }
                        work.waitUntilCompleted()
                    }

                    // Host run-length encode over sorted expert ids → segments.
                    let expertIdsHost = se.toArray(as: UInt32.self)
                    var segStarts: [Int] = []
                    var segMs: [Int] = []
                    var segExperts: [Int] = []
                    var segStart = 0
                    while segStart < mTotal {
                        let e = Int(expertIdsHost[segStart])
                        var segEnd = segStart + 1
                        while segEnd < mTotal && Int(expertIdsHost[segEnd]) == e {
                            segEnd += 1
                        }
                        segStarts.append(segStart)
                        segMs.append(segEnd - segStart)
                        segExperts.append(e)
                        segStart = segEnd
                    }
                    let nSeg = segStarts.count

                    // Fresh CB: gather (async pipe) + all expert QMMs concurrent.
                    let qmmCmd = device.makeCommandBuffer()
                    if asyncPipe {
                        _ = Ops.gather(
                            table: hRows, tokenIds: st, on: qmmCmd,
                            into: xGathered)
                    }

                    var gW: [Tensor] = [], gS: [Tensor] = [], gIn: [Tensor] = []
                    var gOut: [Tensor] = [], gM: [Int] = []
                    var uW: [Tensor] = [], uS: [Tensor] = [], uIn: [Tensor] = []
                    var uOut: [Tensor] = [], uM: [Int] = []
                    gW.reserveCapacity(nSeg); gS.reserveCapacity(nSeg)
                    gIn.reserveCapacity(nSeg); gOut.reserveCapacity(nSeg)
                    gM.reserveCapacity(nSeg)
                    uW.reserveCapacity(nSeg); uS.reserveCapacity(nSeg)
                    uIn.reserveCapacity(nSeg); uOut.reserveCapacity(nSeg)
                    uM.reserveCapacity(nSeg)
                    for i in 0 ..< nSeg {
                        let e = segExperts[i]
                        let mSeg = segMs[i]
                        let s0 = segStarts[i]
                        let xSeg = xGathered.slicedRows(start: s0, count: mSeg)
                        gW.append(
                            stacked.gateWeight.slicedRows(start: e, count: 1)
                                .reshaped(to: [moeIntermediate, hidden / 8]))
                        gS.append(
                            stacked.gateScales.slicedRows(start: e, count: 1)
                                .reshaped(to: [moeIntermediate, hidden / 16]))
                        gIn.append(xSeg)
                        gOut.append(gateOut.slicedRows(start: s0, count: mSeg))
                        gM.append(mSeg)
                        uW.append(
                            stacked.upWeight.slicedRows(start: e, count: 1)
                                .reshaped(to: [moeIntermediate, hidden / 8]))
                        uS.append(
                            stacked.upScales.slicedRows(start: e, count: 1)
                                .reshaped(to: [moeIntermediate, hidden / 16]))
                        uIn.append(xSeg)
                        uOut.append(upOut.slicedRows(start: s0, count: mSeg))
                        uM.append(mSeg)
                    }
                    // Gate + up concurrent; per-segment NAX when m%32==0.
                    Ops.nvfp4QmmMany(
                        weights: gW + uW, scales: gS + uS,
                        inputs: gIn + uIn, outputs: gOut + uOut,
                        ms: gM + uM,
                        blockSize: stacked.blockSize, global: stacked.global,
                        on: qmmCmd)
                    let inner = Ops.swiglu(
                        gate: gateOut, up: upOut, on: qmmCmd, into: gateOut)
                    var dW: [Tensor] = [], dS: [Tensor] = [], dIn: [Tensor] = []
                    var dOut: [Tensor] = [], dM: [Int] = []
                    dW.reserveCapacity(nSeg); dS.reserveCapacity(nSeg)
                    dIn.reserveCapacity(nSeg); dOut.reserveCapacity(nSeg)
                    dM.reserveCapacity(nSeg)
                    for i in 0 ..< nSeg {
                        let e = segExperts[i]
                        let mSeg = segMs[i]
                        let s0 = segStarts[i]
                        dW.append(
                            stacked.downWeight.slicedRows(start: e, count: 1)
                                .reshaped(to: [hidden, moeIntermediate / 8]))
                        dS.append(
                            stacked.downScales.slicedRows(start: e, count: 1)
                                .reshaped(to: [hidden, moeIntermediate / 16]))
                        dIn.append(inner.slicedRows(start: s0, count: mSeg))
                        dOut.append(downOut.slicedRows(start: s0, count: mSeg))
                        dM.append(mSeg)
                    }
                    Ops.nvfp4QmmMany(
                        weights: dW, scales: dS, inputs: dIn, outputs: dOut,
                        ms: dM,
                        blockSize: stacked.blockSize, global: stacked.global,
                        on: qmmCmd)
                    // Weighted sum stays on qmmCmd (sorted → unpermute).
                    if let inv = sortedInvPerm {
                        // Zero outFlat so any unpermute gap is 0 not NaN garbage.
                        if let blit = qmmCmd.makeBlitCommandEncoder() {
                            let o0 = outFlat.offset
                            let o1 = o0 + outFlat.elementCount * dtBytes
                            blit.fill(buffer: outFlat.buffer, range: o0 ..< o1, value: 0)
                            blit.endEncoding()
                        }
                        Ops.moeUnpermute(
                            expertOutputs: downOut, invPerm: inv,
                            topKWeights: weightsTok, into: outFlat,
                            nRows: t, hidden: hidden, k: topK, on: qmmCmd)
                    }
                    // Shared expert: prefer dual-queue result (overlaps wait).
                    if let done = prefillSharedDone, let sOut = prefillSharedOut {
                        qmmCmd.encodeWaitForEvent(device.syncEvent, value: done)
                        _ = Ops.add(outFlat, sOut, on: qmmCmd, into: outFlat)
                    } else if !MoEEnvFlags.noShared,
                        let sg = sharedGateProj, let su = sharedUpProj,
                        let sd = sharedDownProj
                    {
                        let useSharedQmm = MoEEnvFlags.sharedQmmOn
                        if useSharedQmm,
                            let shared = prefillSharedExpertNvfp4(
                                hRows: hRows, t: t, gate: sg, up: su, down: sd,
                                dtype: dt, on: qmmCmd, device: device)
                        {
                            _ = Ops.add(outFlat, shared, on: qmmCmd, into: outFlat)
                        } else if !useSharedQmm {
                            for r in 0 ..< t {
                                let hRow = Tensor(
                                    buffer: hFlat.buffer,
                                    offset: hFlat.offset + r * hidden * dtBytes,
                                    shape: [hidden], dtype: dt)
                                let shared = swiGLU(
                                    hRow, gateProj: sg, upProj: su, downProj: sd,
                                    on: qmmCmd)
                                let outRow = Tensor(
                                    buffer: outFlat.buffer,
                                    offset: outFlat.offset + r * hidden * dtBytes,
                                    shape: [hidden], dtype: dt)
                                _ = Ops.add(outRow, shared, on: qmmCmd, into: outRow)
                            }
                        }
                    }
                    if qmmCmd.status != .committed
                        && qmmCmd.status != .completed && qmmCmd.status != .error
                    {
                        qmmCmd.commit()
                    }
                    return outFlat
                }
                if useGatherQmm || useGatherMpp {
                    let srcTok: Tensor
                    let expIds: Tensor
                    if let st = sortedSource, let se = sortedExperts {
                        srcTok = st
                        expIds = se
                    } else {
                        // Token-order: x row = h[token]. Cache host fill by mTotal.
                        if prefTokenOrderSrcScratch == nil
                            || prefTokenOrderSrcMTotal != mTotal
                            || prefTokenOrderSrcScratch!.elementCount < mTotal
                        {
                            var srcHost = [UInt32](repeating: 0, count: mTotal)
                            for r in 0 ..< t {
                                for s in 0 ..< topK {
                                    srcHost[r * topK + s] = UInt32(r)
                                }
                            }
                            let src = Tensor.empty(
                                shape: [mTotal], dtype: .u32, device: device)
                            src.copyIn(from: srcHost)
                            prefTokenOrderSrcScratch = src
                            prefTokenOrderSrcMTotal = mTotal
                        }
                        srcTok = Tensor(
                            buffer: prefTokenOrderSrcScratch!.buffer,
                            offset: prefTokenOrderSrcScratch!.offset,
                            shape: [mTotal], dtype: .u32)
                        expIds = indicesTok
                    }
                    // Decide MPP dtype early so f32 path can fuse gather+cast
                    // (skip materializing bf16 xGathered).
                    let forceAllBf16 = MoEEnvFlags.mppForceAllBf16
                    let bf16MaxLayer = MoEEnvFlags.mppBf16MaxLayer
                    let bf16ExcludeLayers = MoEEnvFlags.mppBf16ExcludeLayers
                    let useNativeBf16Mpp =
                        forceAllBf16
                        || (layerIndex >= 0 && layerIndex <= bf16MaxLayer
                            && bf16MaxLayer >= 0
                            && !bf16ExcludeLayers.contains(layerIndex))
                    let noSharedCast = MoEEnvFlags.mppNoSharedCast
                    // W54: layers excluded from the fast bf16-MPP path take
                    // the T33 steel-structure kernel instead of the f32
                    // fallback below (see MoEEnvFlags.mppSteelForExcludedLayers'
                    // doc) — shape-eligible for both Laguna GEMM legs
                    // (nOut%64==0, kIn%16==0/%blockSize==0 hold for
                    // moeIntermediate=512/hidden=2048 either way round).
                    // Only bf16 input can use it; the f32-forced escapes
                    // (BUTTER_NVFP4_MPP_BF16_MAX_LAYER=-1 etc.) still fall
                    // through to useF32Promote below untouched.
                    let useSteelRoute =
                        useGatherMpp && dt == .bf16 && !useNativeBf16Mpp
                        && bf16ExcludeLayers.contains(layerIndex)
                        && MoEEnvFlags.mppSteelForExcludedLayers
                        && Ops.nvfp4GatherSteelGatherEligible(
                            nOut: moeIntermediate, kIn: hidden)
                        && Ops.nvfp4GatherSteelGatherEligible(
                            nOut: hidden, kIn: moeIntermediate)
                    let useF32Promote =
                        useGatherMpp && dt == .bf16 && !useNativeBf16Mpp && !noSharedCast
                        && !useSteelRoute

                    // Fused gate+up MPP (default BM16): x_rows + shared X load.
                    // Escape: BUTTER_NVFP4_MPP_FUSED_GATEUP=0
                    // When BM64 is opted in (`BUTTER_NVFP4_MOE_BM64=1` /
                    // `BUTTER_NVFP4_GATHER_BM=64`) skip fused BM16 gateup so
                    // gate+up also take the BM64 concurrent pair — otherwise
                    // only down used BM64 and the opt-in looked "slower"
                    // while barely covering 1/3 of the matmul traffic.
                    let wantBm64GateUp = Ops.nvfp4MoeGatherPrefersBm64(
                        mTotal: mTotal, nOut: moeIntermediate, kIn: hidden)
                    let useFusedGateUp =
                        !wantBm64GateUp
                        && MoEEnvFlags.mppFusedGateUp
                    let xGathered: Tensor
                    if useSteelRoute {
                        // Steel needs a real materialized [mTotal, kIn] bf16
                        // gather (no xRows/gatherCast fusion support) —
                        // always take the materialize branch below.
                        let xBytes = mTotal * hidden * dtBytes
                        let xBuf = device.intermediateScratch(
                            tag: "laguna_mpp_x_gathered", minBytes: xBytes)
                        let gathered = Tensor(
                            buffer: xBuf, offset: 0,
                            shape: [mTotal, hidden], dtype: dt)
                        _ = Ops.gather(
                            table: hRows, tokenIds: srcTok, on: work,
                            into: gathered)
                        xGathered = gathered
                    } else if useF32Promote || (useGatherMpp && useFusedGateUp) {
                        // Fused paths read hRows via x_rows / gatherCast.
                        xGathered = hRows
                    } else {
                        // Materialise [mTotal, kIn] once for gate+up pair
                        // (BM64 concurrent or non-fused BM16 / non-MPP).
                        // Grow-only scratch — bare Ops.gather was Tensor.empty
                        // every MoE layer (driver thrash under swap).
                        let xBytes = mTotal * hidden * dtBytes
                        let xBuf = device.intermediateScratch(
                            tag: "laguna_mpp_x_gathered", minBytes: xBytes)
                        let gathered = Tensor(
                            buffer: xBuf, offset: 0,
                            shape: [mTotal, hidden], dtype: dt)
                        _ = Ops.gather(
                            table: hRows, tokenIds: srcTok, on: work,
                            into: gathered)
                        xGathered = gathered
                    }
                    if useGatherMpp {
                        // Zero outs only when the active M-tile leaves a
                        // partial last row (kernel masks writes but defensive
                        // for partial-tile quality). Full tiles overwrite
                        // every output element — skip 3 blit fills × 39 layers.
                        // BM64 tile height 64; BM16 height 16.
                        let tileM = wantBm64GateUp ? 64 : 16
                        if mTotal % tileM != 0, let blit = work.makeBlitCommandEncoder() {
                            let g0 = gateOut.offset
                            let g1 = g0 + gateOut.elementCount * dtBytes
                            let u0 = upOut.offset
                            let u1 = u0 + upOut.elementCount * dtBytes
                            let d0 = downOut.offset
                            let d1 = d0 + downOut.elementCount * dtBytes
                            blit.fill(buffer: gateOut.buffer, range: g0 ..< g1, value: 0)
                            blit.fill(buffer: upOut.buffer, range: u0 ..< u1, value: 0)
                            blit.fill(buffer: downOut.buffer, range: d0 ..< d1, value: 0)
                            blit.endEncoding()
                        }
                        // Dual-queue shared already kicked earlier (overlaps
                        // router+sort+MPP).
                        // Late-layer half coop overflow: bf16 MPP NaNs from
                        // ~L38 when |h| grows. Early layers stay on native
                        // bf16 MPP (faster). Threshold:
                        //   BUTTER_NVFP4_MPP_BF16_MAX_LAYER (default 36)
                        // Force all-bf16: BUTTER_NVFP4_MPP_BF16=1
                        // Force all-f32:  BUTTER_NVFP4_MPP_BF16_MAX_LAYER=-1
                        // f32 promote path only when NOT using native bf16 MPP.
                        if useF32Promote {
                            let midBytes = mTotal * moeIntermediate * 4
                            let downBytes = mTotal * hidden * 4
                            let gBuf = device.intermediateScratch(
                                tag: "laguna_mpp_gate_f32", minBytes: midBytes)
                            let uBuf = device.intermediateScratch(
                                tag: "laguna_mpp_up_f32", minBytes: midBytes)
                            let dBuf = device.intermediateScratch(
                                tag: "laguna_mpp_down_f32", minBytes: downBytes)
                            let gateF = Tensor(
                                buffer: gBuf, offset: 0,
                                shape: [mTotal, moeIntermediate], dtype: .f32)
                            let upF = Tensor(
                                buffer: uBuf, offset: 0,
                                shape: [mTotal, moeIntermediate], dtype: .f32)
                            let downF = Tensor(
                                buffer: dBuf, offset: 0,
                                shape: [mTotal, hidden], dtype: .f32)
                            if useFusedGateUp {
                                // Cast only [T, H] then fused gate+up via x_rows.
                                let hFBytes = t * hidden * 4
                                let hFBuf = device.intermediateScratch(
                                    tag: "laguna_mpp_h_f32", minBytes: hFBytes)
                                let hF = Tensor(
                                    buffer: hFBuf, offset: 0,
                                    shape: [t, hidden], dtype: .f32)
                                Ops.castToF32(hRows, into: hF, on: work)
                                moeMagScan("gateup_input", hF)
                                Ops.nvfp4MoeGatherBm16GateUp(
                                    input: hF, xRows: srcTok,
                                    weightGate: stacked.gateWeight,
                                    scalesGate: stacked.gateScales, outGate: gateF,
                                    weightUp: stacked.upWeight,
                                    scalesUp: stacked.upScales, outUp: upF,
                                    indices: expIds,
                                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                                    blockSize: stacked.blockSize, global: stacked.global,
                                    on: work)
                            } else {
                                let xBytes = mTotal * hidden * 4
                                let xBuf = device.intermediateScratch(
                                    tag: "laguna_mpp_x_f32", minBytes: xBytes)
                                let xF = Tensor(
                                    buffer: xBuf, offset: 0,
                                    shape: [mTotal, hidden], dtype: .f32)
                                Ops.gatherCastBf16ToF32(
                                    table: hRows, tokenIds: srcTok,
                                    mTotal: mTotal, hidden: hidden,
                                    on: work, into: xF)
                                moeMagScan("gateup_input", xF)
                                // Concurrent gate+up; BM64 when env opts in
                                // (grid/TPG match iron BM64 contract).
                                Ops.nvfp4MoeGatherBm16Pair(
                                    input: xF,
                                    weight0: stacked.gateWeight,
                                    scales0: stacked.gateScales, out0: gateF,
                                    weight1: stacked.upWeight,
                                    scales1: stacked.upScales, out1: upF,
                                    indices: expIds,
                                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                                    blockSize: stacked.blockSize, global: stacked.global,
                                    on: work)
                            }
                            moeMagScan("gate_pre_swiglu", gateF)
                            moeMagScan("up_pre_swiglu", upF)
                            // In-place SwiGLU into gateF (skip separate mid buffer).
                            _ = Ops.swiglu(gate: gateF, up: upF, on: work, into: gateF)
                            moePhaseCheckpoint("gateup")
                            moeMagScan("swiglu_inner", gateF)
                            Ops.nvfp4MoeGatherBm16(
                                input: gateF, weight: stacked.downWeight,
                                scales: stacked.downScales, indices: expIds,
                                mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                                blockSize: stacked.blockSize, global: stacked.global,
                                on: work, into: downF)
                            moeMagScan("down_output", downF)
                            // W52 byte-hunt: skip the castF32ToBf16 ->
                            // downOut materialise (write-once-read-once
                            // bf16 round trip) and let the combine call
                            // site below read `downF` (f32) directly —
                            // see LagunaF32PromoteCombineFuse. Escape
                            // BUTTER_LAGUNA_F32_COMBINE_FUSE=0 restores
                            // the stock cast-then-combine chain (needed
                            // whenever sortedInvPerm is nil, since that
                            // path combines via scalarFMAChain8 on
                            // downOut directly, not moeUnpermute).
                            if LagunaF32PromoteCombineFuse.enabled, sortedInvPerm != nil {
                                pendingCombineF32 = downF
                            } else {
                                Ops.castF32ToBf16(downF, into: downOut, on: work)
                            }
                            moePhaseCheckpoint("down")
                        } else if useSteelRoute {
                            // W54: L38 (the excluded layer). Measured via
                            // BUTTER_LAGUNA_MOE_MAGNITUDE_SCAN at this
                            // layer: gate/up's OWN input (the residual
                            // gather, maxAbs ~3) and output (gate/up
                            // pre-swiglu, maxAbs ~556/1384) both sit
                            // comfortably under half's 65504 ceiling — only
                            // the SwiGLU-amplified product feeding `down`
                            // (~446K-770K depending on prompt) overflows.
                            // So gate/up stay on the shipped fast native-
                            // bf16 path (byte-identical to L37/L39's route,
                            // preferNativeBf16 forced true here since this
                            // layer's own useNativeBf16Mpp is false by
                            // construction) and ONLY `down` pays for the
                            // T33 steel-structure kernel (native bfloat MMA
                            // fragments, no MPP tensor-ops, no half staging
                            // anywhere — structurally overflow-safe here).
                            // See MoEEnvFlags.mppSteelForExcludedLayers'
                            // header comment for the full rationale +
                            // Nvfp4GatherSteel.swift / that file's .metal
                            // sibling for the kernel itself. `down`'s
                            // BM16-vs-BM64 run-length heuristic mirrors the
                            // shipped steel dispatcher directly (bypassing
                            // its automatic priority chain, which would
                            // otherwise hit ExpertGather's f16-cast A
                            // operand first).
                            moeMagScan("gateup_input", xGathered)
                            Ops.nvfp4MoeGatherBm16(
                                input: xGathered, weight: stacked.gateWeight,
                                scales: stacked.gateScales, indices: expIds,
                                mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                                blockSize: stacked.blockSize, global: stacked.global,
                                preferNativeBf16: true,
                                on: work, into: gateOut)
                            Ops.nvfp4MoeGatherBm16(
                                input: xGathered, weight: stacked.upWeight,
                                scales: stacked.upScales, indices: expIds,
                                mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                                blockSize: stacked.blockSize, global: stacked.global,
                                preferNativeBf16: true,
                                on: work, into: upOut)
                            let nExperts = stacked.gateWeight.shape[0]
                            let steelBm16 = Ops.nvfp4GatherSteelPrefersBm16(
                                mTotal: mTotal, nExperts: nExperts)
                            moeMagScan("gate_pre_swiglu", gateOut)
                            moeMagScan("up_pre_swiglu", upOut)
                            let steelInner = Ops.swiglu(
                                gate: gateOut, up: upOut, on: work, into: gateOut)
                            moePhaseCheckpoint("gateup")
                            moeMagScan("swiglu_inner", steelInner)
                            if steelBm16 {
                                Ops.nvfp4GatherSteelBm16(
                                    input: steelInner, weight: stacked.downWeight,
                                    scales: stacked.downScales, indices: expIds,
                                    mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                                    blockSize: stacked.blockSize, global: stacked.global,
                                    on: work, into: downOut)
                            } else {
                                Ops.nvfp4GatherSteel(
                                    input: steelInner, weight: stacked.downWeight,
                                    scales: stacked.downScales, indices: expIds,
                                    mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                                    blockSize: stacked.blockSize, global: stacked.global,
                                    on: work, into: downOut)
                            }
                            moeMagScan("down_output", downOut)
                            moePhaseCheckpoint("down")
                        } else if useFusedGateUp {
                            // Follow-on lever chain (skip identity
                            // re-sort; see LagunaNvfp4ExpertGather's
                            // lever header). Fires only with
                            // BUTTER_LAGUNA_GATHER_V2=1 + a lever flag
                            // + the sorted plan + both routed
                            // projections V2-eligible; the base
                            // default chain below is untouched
                            // otherwise (base A/B arm stays exact).
                            let v2Levers = LagunaNvfp4ExpertGather.levers
                            if v2Levers.any, sortedSource != nil,
                                sortedExperts != nil,
                                LagunaNvfp4ExpertGather.eligible(
                                    dtype: dt, nOut: moeIntermediate,
                                    kIn: hidden,
                                    blockSize: stacked.blockSize),
                                LagunaNvfp4ExpertGather.eligible(
                                    dtype: dt, nOut: hidden,
                                    kIn: moeIntermediate,
                                    blockSize: stacked.blockSize)
                            {
                                let inner = LagunaNvfp4ExpertGather.gateUpChain(
                                    hRows: hRows, srcTok: srcTok, indices: expIds,
                                    gateWeight: stacked.gateWeight,
                                    gateScales: stacked.gateScales,
                                    upWeight: stacked.upWeight,
                                    upScales: stacked.upScales,
                                    gateOut: gateOut, upOut: upOut,
                                    mTotal: mTotal, nOut: moeIntermediate,
                                    kIn: hidden,
                                    blockSize: stacked.blockSize,
                                    global: stacked.global,
                                    levers: v2Levers,
                                    on: work, device: device)
                                moePhaseCheckpoint("gateup")
                                LagunaNvfp4ExpertGather.downChain(
                                    inner: inner, indices: expIds,
                                    weight: stacked.downWeight,
                                    scales: stacked.downScales,
                                    mTotal: mTotal, nOut: hidden,
                                    kIn: moeIntermediate,
                                    blockSize: stacked.blockSize,
                                    global: stacked.global,
                                    levers: v2Levers,
                                    on: work, device: device, into: downOut)
                                moePhaseCheckpoint("down")
                            } else {
                            // Native bf16 fused gate+up (x_rows, no materialise).
                            Ops.nvfp4MoeGatherBm16GateUp(
                                input: hRows, xRows: srcTok,
                                weightGate: stacked.gateWeight,
                                scalesGate: stacked.gateScales, outGate: gateOut,
                                weightUp: stacked.upWeight,
                                scalesUp: stacked.upScales, outUp: upOut,
                                indices: expIds,
                                mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                                blockSize: stacked.blockSize, global: stacked.global,
                                on: work)
                            let inner = Ops.swiglu(
                                gate: gateOut, up: upOut, on: work, into: gateOut)
                            moePhaseCheckpoint("gateup")
                            Ops.nvfp4MoeGatherBm16(
                                input: inner, weight: stacked.downWeight,
                                scales: stacked.downScales, indices: expIds,
                                mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                                blockSize: stacked.blockSize, global: stacked.global,
                                preferNativeBf16: useNativeBf16Mpp,
                                on: work, into: downOut)
                            moePhaseCheckpoint("down")
                            }
                        } else {
                            // Materialise gather + concurrent pair (BM16 or BM64)
                            // / serial fallback.
                            if !MoEEnvFlags.mppSerialGateUp
                            {
                                Ops.nvfp4MoeGatherBm16Pair(
                                    input: xGathered,
                                    weight0: stacked.gateWeight,
                                    scales0: stacked.gateScales, out0: gateOut,
                                    weight1: stacked.upWeight,
                                    scales1: stacked.upScales, out1: upOut,
                                    indices: expIds,
                                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                                    blockSize: stacked.blockSize, global: stacked.global,
                                    on: work)
                            } else {
                                Ops.nvfp4MoeGatherBm16(
                                    input: xGathered, weight: stacked.gateWeight,
                                    scales: stacked.gateScales, indices: expIds,
                                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                                    blockSize: stacked.blockSize, global: stacked.global,
                                    preferNativeBf16: useNativeBf16Mpp,
                                    on: work, into: gateOut)
                                Ops.nvfp4MoeGatherBm16(
                                    input: xGathered, weight: stacked.upWeight,
                                    scales: stacked.upScales, indices: expIds,
                                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                                    blockSize: stacked.blockSize, global: stacked.global,
                                    preferNativeBf16: useNativeBf16Mpp,
                                    on: work, into: upOut)
                            }
                            let inner = Ops.swiglu(
                                gate: gateOut, up: upOut, on: work, into: gateOut)
                            moePhaseCheckpoint("gateup")
                            Ops.nvfp4MoeGatherBm16(
                                input: inner, weight: stacked.downWeight,
                                scales: stacked.downScales, indices: expIds,
                                mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                                blockSize: stacked.blockSize, global: stacked.global,
                                preferNativeBf16: useNativeBf16Mpp,
                                on: work, into: downOut)
                            moePhaseCheckpoint("down")
                        }
                    } else {
                        // Non-MPP gather QMM (token order).
                        Ops.nvfp4GatherQmm(
                            weight: stacked.gateWeight, scales: stacked.gateScales,
                            input: xGathered, expertIds: expIds,
                            mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                            blockSize: stacked.blockSize, global: stacked.global,
                            on: work, into: gateOut)
                        Ops.nvfp4GatherQmm(
                            weight: stacked.upWeight, scales: stacked.upScales,
                            input: xGathered, expertIds: expIds,
                            mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                            blockSize: stacked.blockSize, global: stacked.global,
                            on: work, into: upOut)
                        let inner = Ops.swiglu(
                            gate: gateOut, up: upOut, on: work, into: gateOut)
                        Ops.nvfp4GatherQmm(
                            weight: stacked.downWeight, scales: stacked.downScales,
                            input: inner, expertIds: expIds,
                            mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                            blockSize: stacked.blockSize, global: stacked.global,
                            on: work, into: downOut)
                    }
                } else {
                    var gW: [Tensor] = [], gS: [Tensor] = [], gIn: [Tensor] = []
                    var gIdx: [Tensor] = [], gOut: [Tensor] = []
                    var uW: [Tensor] = [], uS: [Tensor] = [], uIn: [Tensor] = []
                    var uIdx: [Tensor] = [], uOut: [Tensor] = []
                    gW.reserveCapacity(mTotal); gS.reserveCapacity(mTotal)
                    gIn.reserveCapacity(mTotal); gIdx.reserveCapacity(mTotal)
                    gOut.reserveCapacity(mTotal)
                    uW.reserveCapacity(mTotal); uS.reserveCapacity(mTotal)
                    uIn.reserveCapacity(mTotal); uIdx.reserveCapacity(mTotal)
                    uOut.reserveCapacity(mTotal)
                    for r in 0 ..< t {
                        let hRow = Tensor(
                            buffer: hFlat.buffer,
                            offset: hFlat.offset + r * hidden * dtBytes,
                            shape: [hidden], dtype: dt)
                        for s in 0 ..< topK {
                            let slot = r * topK + s
                            let eIdx = Tensor(
                                buffer: indicesTok.buffer,
                                offset: indicesTok.offset + slot * 4,
                                shape: [1], dtype: .u32)
                            let gRow = Tensor(
                                buffer: gateOut.buffer,
                                offset: gateOut.offset + slot * moeIntermediate * dtBytes,
                                shape: [moeIntermediate], dtype: dt)
                            let uRow = Tensor(
                                buffer: upOut.buffer,
                                offset: upOut.offset + slot * moeIntermediate * dtBytes,
                                shape: [moeIntermediate], dtype: dt)
                            gW.append(stacked.gateWeight); gS.append(stacked.gateScales)
                            gIn.append(hRow); gIdx.append(eIdx); gOut.append(gRow)
                            uW.append(stacked.upWeight); uS.append(stacked.upScales)
                            uIn.append(hRow); uIdx.append(eIdx); uOut.append(uRow)
                        }
                    }
                    Ops.nvfp4DequantGemvExpertIndexedMany(
                        weightsStacked: gW + uW, scalesStacked: gS + uS,
                        inputs: gIn + uIn, expertIndices: gIdx + uIdx,
                        outputs: gOut + uOut,
                        blockSize: stacked.blockSize, global: stacked.global, on: work)
                    let innerFlat = Ops.swiglu(
                        gate: gateOut.reshaped(to: [mTotal * moeIntermediate]),
                        up: upOut.reshaped(to: [mTotal * moeIntermediate]),
                        on: work)
                    var dW: [Tensor] = [], dS: [Tensor] = [], dIn: [Tensor] = []
                    var dIdx: [Tensor] = [], dOut: [Tensor] = []
                    for slot in 0 ..< mTotal {
                        let inRow = Tensor(
                            buffer: innerFlat.buffer,
                            offset: innerFlat.offset + slot * moeIntermediate * dtBytes,
                            shape: [moeIntermediate], dtype: dt)
                        let eIdx = Tensor(
                            buffer: indicesTok.buffer,
                            offset: indicesTok.offset + slot * 4,
                            shape: [1], dtype: .u32)
                        let oRow = Tensor(
                            buffer: downOut.buffer,
                            offset: downOut.offset + slot * hidden * dtBytes,
                            shape: [hidden], dtype: dt)
                        dW.append(stacked.downWeight); dS.append(stacked.downScales)
                        dIn.append(inRow); dIdx.append(eIdx); dOut.append(oRow)
                    }
                    Ops.nvfp4DequantGemvExpertIndexedMany(
                        weightsStacked: dW, scalesStacked: dS,
                        inputs: dIn, expertIndices: dIdx, outputs: dOut,
                        blockSize: stacked.blockSize, global: stacked.global, on: work)
                }

                // W57: fused combine+shared+residual tail eligibility.
                // Only the sorted/MPP `Ops.moeUnpermute` path (not the
                // token-order FMA `else` below, not the
                // `LagunaSortedCombine` numerics-audit alternative — a
                // separate, orthogonal experiment; that flag takes
                // priority when both happen to be set — see
                // `LagunaFusedMoeTail`'s file header for the full
                // rationale). `tailFused` records whether the fused
                // kernel actually ran this call, so the unfused
                // shared-expert block below (and `lastResidualFused`,
                // read by the caller) can key off it.
                let fuseTailWanted =
                    LagunaFusedMoeTail.enabled && fusedResidual != nil
                    && !(LagunaSortedCombine.enabled && topK <= 8)
                var tailFused = false

                // Weighted sum: FMA when outputs are token-order; unpermute
                // when expert-sorted (MPP path).
                if let inv = sortedInvPerm {
                    // Zero outFlat so any unpermute gap is 0 not NaN garbage.
                    // GPU blit (not host memset) — avoids CPU↔GPU sync on
                    // shared storage mid-cmdbuf under swap pressure. Not
                    // needed by the fused kernel (it writes every element
                    // unconditionally) but kept unconditional for
                    // simplicity — cheap relative to the tail's other
                    // dispatches and harmless when overwritten right after.
                    if let blit = work.makeBlitCommandEncoder() {
                        let o0 = outFlat.offset
                        let o1 = o0 + outFlat.elementCount * dtBytes
                        blit.fill(buffer: outFlat.buffer, range: o0 ..< o1, value: 0)
                        blit.endEncoding()
                    }
                    // F-85 prefill pack item 4: opt-in hand-authored
                    // combine kernel with explicit split multiply/add (no
                    // FMA contraction), see LagunaSortedCombine.swift's
                    // file header for why this exists alongside
                    // Ops.moeUnpermute (which already does the same fusion
                    // structurally, but its generated-kernel source isn't
                    // visible in this repo to audit for the numerics risk
                    // the mission flags). Default OFF, zero behavior
                    // change unless BUTTER_LAGUNA_SORTED_COMBINE=1.
                    if let f32Down = pendingCombineF32 {
                        // W52 byte-hunt: f32Promote layer, downOut was
                        // never materialised — combine straight off the
                        // f32 down-GEMM output (see
                        // LagunaF32PromoteCombineFuse's header for the
                        // byte-identical double-round argument).
                        LagunaF32PromoteCombineFuse.combine(
                            expertOutputsF32: f32Down, invPerm: inv,
                            topKWeights: weightsTok, into: outFlat,
                            nRows: t, hidden: hidden, k: topK,
                            on: work, device: device)
                    } else if LagunaSortedCombine.enabled, topK <= 8 {
                        LagunaSortedCombine.combine(
                            expertOutputs: downOut, invPerm: inv,
                            topKWeights: weightsTok, into: outFlat,
                            nRows: t, hidden: hidden, k: topK,
                            on: work, device: device)
                    } else if fuseTailWanted {
                        // Resolve the shared-expert output as ONE
                        // [T,hidden] tensor (or nil) first — same three
                        // sub-cases the unfused shared-add block below
                        // handles, just without doing the add yet. The
                        // per-row swiGLU fallback (no batched QMM) can't
                        // produce one [T,hidden] tensor up front without
                        // its own scratch-buffer restructuring — that
                        // sub-case is out of scope for this bounded lever
                        // (rare/slow path already), so it falls through to
                        // the unfused sequence below instead.
                        let sharedIsFallbackOnly =
                            !MoEEnvFlags.noShared && sharedGateProj != nil
                            && !MoEEnvFlags.sharedQmmOn && prefillSharedDone == nil
                        if sharedIsFallbackOnly {
                            Ops.moeUnpermute(
                                expertOutputs: downOut, invPerm: inv,
                                topKWeights: weightsTok, into: outFlat,
                                nRows: t, hidden: hidden, k: topK, on: work)
                        } else {
                            var sharedTensor: Tensor?
                            if let done = prefillSharedDone, let sOut = prefillSharedOut {
                                work.encodeWaitForEvent(device.syncEvent, value: done)
                                sharedTensor = sOut
                            } else if !MoEEnvFlags.noShared,
                                let sg = sharedGateProj, let su = sharedUpProj,
                                let sd = sharedDownProj, MoEEnvFlags.sharedQmmOn
                            {
                                sharedTensor = prefillSharedExpertNvfp4(
                                    hRows: hRows, t: t, gate: sg, up: su, down: sd,
                                    dtype: dt, on: work, device: device)
                            }
                            Ops.moeUnpermuteTail(
                                expertOutputs: downOut, invPerm: inv,
                                topKWeights: weightsTok, sharedExpert: sharedTensor,
                                residual: fusedResidual!, into: outFlat,
                                nRows: t, hidden: hidden, k: topK, on: work)
                            tailFused = true
                            lastResidualFused = true
                        }
                    } else {
                        Ops.moeUnpermute(
                            expertOutputs: downOut, invPerm: inv,
                            topKWeights: weightsTok, into: outFlat,
                            nRows: t, hidden: hidden, k: topK, on: work)
                    }
                } else {
                    for r in 0 ..< t {
                        var scalars: [Tensor] = []
                        var values: [Tensor] = []
                        scalars.reserveCapacity(topK)
                        values.reserveCapacity(topK)
                        for s in 0 ..< topK {
                            let slot = r * topK + s
                            scalars.append(
                                Tensor(
                                    buffer: weightsTok.buffer,
                                    offset: weightsTok.offset + slot * dtBytes,
                                    shape: [1], dtype: dt))
                            values.append(
                                Tensor(
                                    buffer: downOut.buffer,
                                    offset: downOut.offset + slot * hidden * dtBytes,
                                    shape: [hidden], dtype: dt))
                        }
                        let acc = Tensor(
                            buffer: outFlat.buffer,
                            offset: outFlat.offset + r * hidden * dtBytes,
                            shape: [hidden], dtype: dt)
                        Ops.scalarFMAChain8(
                            scalars: scalars, values: values, out: acc, on: work)
                    }
                }

                // Shared expert: dual-queue (MPP path) or batched QMM on
                // `work`. Skipped when the W57 fused kernel already folded
                // shared (+ residual) into the combine dispatch above.
                if !tailFused {
                    if let done = prefillSharedDone, let sOut = prefillSharedOut {
                        work.encodeWaitForEvent(device.syncEvent, value: done)
                        _ = Ops.add(outFlat, sOut, on: work, into: outFlat)
                    } else if !MoEEnvFlags.noShared,
                        let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj
                    {
                        let useSharedQmm =
                            MoEEnvFlags.sharedQmmOn
                        if useSharedQmm,
                            let shared = prefillSharedExpertNvfp4(
                                hRows: hRows, t: t, gate: sg, up: su, down: sd,
                                dtype: dt, on: work, device: device)
                        {
                            _ = Ops.add(outFlat, shared, on: work, into: outFlat)
                        } else if !useSharedQmm {
                            for r in 0 ..< t {
                                let hRow = Tensor(
                                    buffer: hFlat.buffer,
                                    offset: hFlat.offset + r * hidden * dtBytes,
                                    shape: [hidden], dtype: dt)
                                let shared = swiGLU(
                                    hRow, gateProj: sg, upProj: su, downProj: sd, on: work)
                                let outRow = Tensor(
                                    buffer: outFlat.buffer,
                                    offset: outFlat.offset + r * hidden * dtBytes,
                                    shape: [hidden], dtype: dt)
                                _ = Ops.add(outRow, shared, on: work, into: outRow)
                            }
                        }
                    }
                }
            }
            moePhaseCheckpoint("tail")
            if moePhaseTiming, !moePhaseTimes.isEmpty {
                let total = moePhaseTimes.reduce(0.0) { $0 + $1.1 }
                let parts = moePhaseTimes.map {
                    "\($0.0)=\(String(format: "%.2f", $0.1))ms"
                }.joined(separator: " ")
                fputs(
                    "[laguna-moe-phase] L\(String(format: "%02d", layerIndex)) "
                        + "total=\(String(format: "%.2f", total))ms \(parts)\n",
                    stderr)
            }
            if commitAtEnd, !useDevicePlan,
                work.status != .committed && work.status != .completed
                    && work.status != .error
            {
                work.commit()
            }
            return outFlat
        }

        // ── Experimental gather / sorted eidx (A/B only) ─────────────
        let useGatherQmm =
            MoEEnvFlags.useGatherQmm
        let gatherIdxTensor: Tensor
        let indices: Tensor
        if let se = sortedExperts, let gi = gatherIdx {
            gatherIdxTensor = gi
            indices = se
        } else {
            let (gatherIdxBuf, expertIdsBuf) = gatherRoutingBuffers(mTotal: mTotal, device: device)
            sourceTokensHost.withUnsafeBytes {
                _ = memcpy(gatherIdxBuf.contents(), $0.baseAddress!, mTotal * 4)
            }
            gatherIdxTensor = Tensor(
                buffer: gatherIdxBuf, offset: 0, shape: [mTotal], dtype: .u32)
            sortedExpertsHost.withUnsafeBytes {
                _ = memcpy(expertIdsBuf.contents(), $0.baseAddress!, mTotal * 4)
            }
            indices = Tensor(
                buffer: expertIdsBuf, offset: 0, shape: [mTotal], dtype: .u32)
        }

        let xGathered = Ops.gather(table: hRows, tokenIds: gatherIdxTensor, on: work)
        let gateOut = Tensor.empty(
            shape: [mTotal, moeIntermediate], dtype: dt, device: device)
        let upOut = Tensor.empty(
            shape: [mTotal, moeIntermediate], dtype: dt, device: device)
        let downOut = Tensor.empty(shape: [mTotal, hidden], dtype: dt, device: device)

        if useGatherQmm {
            Ops.nvfp4MoeGatherBm16(
                input: xGathered, weight: stacked.gateWeight, scales: stacked.gateScales,
                indices: indices, mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                blockSize: stacked.blockSize, global: stacked.global,
                on: work, into: gateOut)
            Ops.nvfp4MoeGatherBm16(
                input: xGathered, weight: stacked.upWeight, scales: stacked.upScales,
                indices: indices, mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                blockSize: stacked.blockSize, global: stacked.global,
                on: work, into: upOut)
            let inner = Ops.swiglu(gate: gateOut, up: upOut, on: work, into: gateOut)
            Ops.nvfp4MoeGatherBm16(
                input: inner, weight: stacked.downWeight, scales: stacked.downScales,
                indices: indices, mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                blockSize: stacked.blockSize, global: stacked.global,
                on: work, into: downOut)
        } else {
            var gW: [Tensor] = [], gS: [Tensor] = [], gIn: [Tensor] = []
            var gIdx: [Tensor] = [], gOut: [Tensor] = []
            var uW: [Tensor] = [], uS: [Tensor] = [], uIn: [Tensor] = []
            var uIdx: [Tensor] = [], uOut: [Tensor] = []
            for r in 0 ..< mTotal {
                let xRow = Tensor(
                    buffer: xGathered.buffer,
                    offset: xGathered.offset + r * hidden * dtBytes,
                    shape: [hidden], dtype: dt)
                let eIdx = Tensor(
                    buffer: indices.buffer, offset: indices.offset + r * 4,
                    shape: [1], dtype: .u32)
                let gRow = Tensor(
                    buffer: gateOut.buffer,
                    offset: gateOut.offset + r * moeIntermediate * dtBytes,
                    shape: [moeIntermediate], dtype: dt)
                let uRow = Tensor(
                    buffer: upOut.buffer,
                    offset: upOut.offset + r * moeIntermediate * dtBytes,
                    shape: [moeIntermediate], dtype: dt)
                gW.append(stacked.gateWeight); gS.append(stacked.gateScales)
                gIn.append(xRow); gIdx.append(eIdx); gOut.append(gRow)
                uW.append(stacked.upWeight); uS.append(stacked.upScales)
                uIn.append(xRow); uIdx.append(eIdx); uOut.append(uRow)
            }
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: gW + uW, scalesStacked: gS + uS,
                inputs: gIn + uIn, expertIndices: gIdx + uIdx,
                outputs: gOut + uOut,
                blockSize: stacked.blockSize, global: stacked.global, on: work)
            let innerFlat = Ops.swiglu(
                gate: gateOut.reshaped(to: [mTotal * moeIntermediate]),
                up: upOut.reshaped(to: [mTotal * moeIntermediate]),
                on: work)
            var dW: [Tensor] = [], dS: [Tensor] = [], dIn: [Tensor] = []
            var dIdx: [Tensor] = [], dOut: [Tensor] = []
            for r in 0 ..< mTotal {
                let inRow = Tensor(
                    buffer: innerFlat.buffer,
                    offset: innerFlat.offset + r * moeIntermediate * dtBytes,
                    shape: [moeIntermediate], dtype: dt)
                let eIdx = Tensor(
                    buffer: indices.buffer, offset: indices.offset + r * 4,
                    shape: [1], dtype: .u32)
                let oRow = Tensor(
                    buffer: downOut.buffer,
                    offset: downOut.offset + r * hidden * dtBytes,
                    shape: [hidden], dtype: dt)
                dW.append(stacked.downWeight); dS.append(stacked.downScales)
                dIn.append(inRow); dIdx.append(eIdx); dOut.append(oRow)
            }
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: dW, scalesStacked: dS,
                inputs: dIn, expertIndices: dIdx, outputs: dOut,
                blockSize: stacked.blockSize, global: stacked.global, on: work)
        }

        let invPermTensor: Tensor
        let weightsTensor: Tensor
        if let ip = invPerm, let wt = weightsTok, wt.dtype == dt {
            invPermTensor = ip
            weightsTensor = wt
        } else {
            let invPermBuf = invPermRoutingBuffer(mTotal: mTotal, device: device)
            invPermHost.withUnsafeBytes {
                _ = memcpy(invPermBuf.contents(), $0.baseAddress!, mTotal * 4)
            }
            invPermTensor = Tensor(
                buffer: invPermBuf, offset: 0, shape: [mTotal], dtype: .u32)
            let wt = Tensor.empty(shape: [mTotal], dtype: dt, device: device)
            switch dt {
            case .f32: wt.copyIn(from: weightsTokenOrderHost)
            case .bf16:
                wt.copyIn(from: weightsTokenOrderHost.map { v -> UInt16 in
                    let bits = v.bitPattern
                    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                    return UInt16(rounded >> 16)
                })
            case .f16: wt.copyIn(from: weightsTokenOrderHost.map { Float16($0) })
            default: fatalError("decodeManyNvfp4: bad dtype")
            }
            weightsTensor = wt
        }
        Ops.moeUnpermute(
            expertOutputs: downOut, invPerm: invPermTensor,
            topKWeights: weightsTensor, into: outFlat,
            nRows: t, hidden: hidden, k: topK, on: work)

        if !MoEEnvFlags.noShared,
            let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj,
            let shared = prefillSharedExpertNvfp4(
                hRows: hRows, t: t, gate: sg, up: su, down: sd,
                dtype: dt, on: work, device: device)
        {
            _ = Ops.add(outFlat, shared, on: work, into: outFlat)
        }
        if commitAtEnd { work.commit() }
        return outFlat
    }

    /// Prefill shared expert via NVFP4 QMM with grow-only scratches.
    /// Default: fused iron gate+up QMM on [gate;up] bank + interleaved SwiGLU
    /// + down QMM. Escape fused: `BUTTER_LAGUNA_SHARED_PREFILL_FUSED_GATEUP=0`
    /// (falls back to concurrent gate+up QMM). Returns `[t * hidden]`.
    private func prefillSharedExpertNvfp4(
        hRows: Tensor, t: Int,
        gate sg: AnyLinear, up su: AnyLinear, down sd: AnyLinear,
        dtype dt: DType,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor? {
        let gn = sg.inner as? Nvfp4Linear
        let un = su.inner as? Nvfp4Linear
        let dn = sd.inner as? Nvfp4Linear
        let mid: Int
        if let gn, let un, gn.outDim == un.outDim {
            mid = gn.outDim
        } else {
            // Unknown dims — allocate via callMany (no scratch).
            let g = sg.callMany(hRows, t: t, on: cmd, device: device)
            let u = su.callMany(hRows, t: t, on: cmd, device: device)
            let sharedInner = Ops.swiglu(
                gate: g.reshaped(to: [g.elementCount]),
                up: u.reshaped(to: [u.elementCount]),
                on: cmd, into: u.reshaped(to: [u.elementCount]))
            let m = sharedInner.elementCount / t
            return sd.callMany(
                sharedInner.reshaped(to: [t, m]), t: t, on: cmd, device: device
            ).reshaped(to: [t * hidden])
        }
        let guNeed = t * mid * 2
        let midNeed = t * mid
        let downNeed = t * hidden
        if prefSharedMidScratch == nil
            || prefSharedMidScratch!.dtype != dt
            || prefSharedMidScratch!.elementCount < midNeed
        {
            prefSharedMidScratch = Tensor.empty(
                shape: [t, mid], dtype: dt, device: device)
        }
        if prefSharedDownScratch == nil
            || prefSharedDownScratch!.dtype != dt
            || prefSharedDownScratch!.elementCount < downNeed
        {
            prefSharedDownScratch = Tensor.empty(
                shape: [t, hidden], dtype: dt, device: device)
        }
        let midOut = Tensor(
            buffer: prefSharedMidScratch!.buffer,
            offset: prefSharedMidScratch!.offset,
            shape: [t, mid], dtype: dt)
        let downOut = Tensor(
            buffer: prefSharedDownScratch!.buffer,
            offset: prefSharedDownScratch!.offset,
            shape: [t, hidden], dtype: dt)

        let useFused =
            LagunaNvfp4Fuse.sharedPrefillFusedGateUpEnabled
            && gn != nil && un != nil
            && gn!.inDim == un!.inDim
            && ensureFusedSharedGateUp(gate: sg, up: su, device: device) != nil

        if useFused,
            let (fw, fs, fg) = ensureFusedSharedGateUp(
                gate: sg, up: su, device: device),
            let dn
        {
            if prefSharedGuScratch == nil
                || prefSharedGuScratch!.dtype != dt
                || prefSharedGuScratch!.elementCount < guNeed
            {
                prefSharedGuScratch = Tensor.empty(
                    shape: [t, mid * 2], dtype: dt, device: device)
            }
            let gu = Tensor(
                buffer: prefSharedGuScratch!.buffer,
                offset: prefSharedGuScratch!.offset,
                shape: [t, mid * 2], dtype: dt)
            _ = Ops.nvfp4Qmm(
                weight: fw, scales: fs, input: hRows,
                m: t, blockSize: gn!.blockSize, global: fg,
                on: cmd, into: gu)
            LagunaNvfp4Fuse.swigluGateUpInterleaved(
                gu: gu, mid: mid, on: cmd, device: device, into: midOut)
            _ = Ops.nvfp4Qmm(
                weight: dn.weight, scales: dn.scales, input: midOut,
                m: t, blockSize: dn.blockSize, global: dn.global,
                on: cmd, into: downOut)
            return downOut.reshaped(to: [t * hidden])
        }

        // Concurrent (or serial callMany) gate+up into scratches.
        if prefSharedGateScratch == nil
            || prefSharedGateScratch!.dtype != dt
            || prefSharedGateScratch!.elementCount < midNeed
        {
            prefSharedGateScratch = Tensor.empty(
                shape: [t, mid], dtype: dt, device: device)
            prefSharedUpScratch = Tensor.empty(
                shape: [t, mid], dtype: dt, device: device)
        }
        let gOut = Tensor(
            buffer: prefSharedGateScratch!.buffer,
            offset: prefSharedGateScratch!.offset,
            shape: [t, mid], dtype: dt)
        let uOut = Tensor(
            buffer: prefSharedUpScratch!.buffer,
            offset: prefSharedUpScratch!.offset,
            shape: [t, mid], dtype: dt)
        if let gn, let un, gn.inDim == un.inDim, gn.outDim == un.outDim {
            Ops.nvfp4QmmMany(
                weights: [gn.weight, un.weight],
                scales: [gn.scales, un.scales],
                inputs: [hRows, hRows],
                outputs: [gOut, uOut],
                ms: [t, t],
                blockSize: gn.blockSize, global: gn.global,
                on: cmd)
        } else if let gn, let un {
            // Prefer nvfp4Qmm(+into) over callMany(+into) — AnyLinear path
            // has no `into:` and overload resolution can fail mid-rebase.
            _ = Ops.nvfp4Qmm(
                weight: gn.weight, scales: gn.scales, input: hRows,
                m: t, blockSize: gn.blockSize, global: gn.global,
                on: cmd, into: gOut)
            _ = Ops.nvfp4Qmm(
                weight: un.weight, scales: un.scales, input: hRows,
                m: t, blockSize: un.blockSize, global: un.global,
                on: cmd, into: uOut)
        } else {
            // Non-NVFP4 shared (shouldn't hit on Laguna board path).
            let g = sg.callMany(hRows, t: t, on: cmd, device: device)
            let u = su.callMany(hRows, t: t, on: cmd, device: device)
            let midFlat = midOut.reshaped(to: [midNeed])
            let sharedInner = Ops.swiglu(
                gate: g.reshaped(to: [g.elementCount]),
                up: u.reshaped(to: [u.elementCount]),
                on: cmd, into: midFlat)
            let d = sd.callMany(
                sharedInner.reshaped(to: [t, mid]), t: t, on: cmd, device: device)
            Ops.copy(d.reshaped(to: [t * hidden]), into: downOut.reshaped(to: [t * hidden]),
                on: cmd)
            return downOut.reshaped(to: [t * hidden])
        }
        let midFlat = midOut.reshaped(to: [midNeed])
        let sharedInner = Ops.swiglu(
            gate: gOut.reshaped(to: [midNeed]),
            up: uOut.reshaped(to: [midNeed]),
            on: cmd, into: midFlat)
        if let dn {
            _ = Ops.nvfp4Qmm(
                weight: dn.weight, scales: dn.scales,
                input: sharedInner.reshaped(to: [t, mid]),
                m: t, blockSize: dn.blockSize, global: dn.global,
                on: cmd, into: downOut)
        } else {
            let d = sd.callMany(
                sharedInner.reshaped(to: [t, mid]), t: t, on: cmd, device: device)
            Ops.copy(d.reshaped(to: [t * hidden]), into: downOut.reshaped(to: [t * hidden]),
                on: cmd)
        }
        return downOut.reshaped(to: [t * hidden])
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

    /// Path B decode: GPU sigmoid+bias top-k (indices/weights), then the
    /// **same** per-expert SwiGLU path as the host router (gateProj[id]).
    /// Expert-indexed stacked GEMV is not quality-locked for Laguna affine4
    /// yet — host experts keep tokens correct while we still skip the
    /// Path B decode (Laguna / Hy3 sigmoid+bias top-k).
    ///
    /// Modes via `BUTTER_LAGUNA_PATHB`:
    /// - unset / `1` / `full` (default): GPU router + int4 expert-indexed
    ///   GEMV (no host router sync). Requires model-loop CB fencing
    ///   (LagunaModel.forward pipelines depth-1 waits).
    /// - `hybrid`: GPU router → host sync → QuantizedLinear experts.
    /// - `0` handled by caller (does not enter this function).
    private func decodeGPUPathB(
        h: Tensor, logitsTensor: Tensor,
        stacked: StackedInt4Experts,
        expertBiasGPU: Tensor,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let mode = MoEEnvFlags.pathBMode
            .lowercased()
        if mode == "hybrid" {
            return decodeGPUPathBHybrid(
                h: h, logitsTensor: logitsTensor,
                expertBiasGPU: expertBiasGPU, cmd: cmd, device: device)
        }
        // `full`, `1`, or anything else → expert-indexed.
        return decodeGPUPathBFull(
            h: h, logitsTensor: logitsTensor, stacked: stacked,
            expertBiasGPU: expertBiasGPU, cmd: cmd, device: device)
    }

    /// Quality Path B: GPU sigmoid+bias top-k, one sync, host experts.
    private func decodeGPUPathBHybrid(
        h: Tensor, logitsTensor: Tensor,
        expertBiasGPU: Tensor,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let topK = router.topK
        let nExperts = router.nExperts
        let scoreDt = h.dtype

        if routerIndicesScratch == nil || routerIndicesScratch!.elementCount != topK {
            routerIndicesScratch = Tensor.empty(shape: [topK], dtype: .u32, device: device)
            routerWeightsScratch = Tensor.empty(shape: [topK], dtype: scoreDt, device: device)
        }
        if unbiasedScoreScratch == nil
            || unbiasedScoreScratch!.elementCount != nExperts
            || unbiasedScoreScratch!.dtype != scoreDt
        {
            unbiasedScoreScratch = Tensor.empty(shape: [nExperts], dtype: scoreDt, device: device)
            biasedScoreScratch = Tensor.empty(shape: [nExperts], dtype: scoreDt, device: device)
        }

        if RouterSetsDebug.enabled {
            debugRouterLogits = logitsTensor
            debugRouterBias = expertBiasGPU
        }
        let unbiased = Ops.sigmoid(logitsTensor, on: cmd, into: unbiasedScoreScratch!)
        let biased = Ops.add(unbiased, expertBiasGPU, on: cmd, into: biasedScoreScratch!)
        Ops.moeRouterTopKBiased(
            scoreBiased: biased, scoreUnbiased: unbiased,
            indicesOut: routerIndicesScratch!,
            weightsOut: routerWeightsScratch!,
            nExperts: nExperts, k: topK, on: cmd)

        let scale = router.routedScalingFactor
        if scale != 1.0 {
            if pathBScaleVecScratch == nil
                || pathBScaleVecScratch!.dtype != scoreDt
                || pathBScaleVecScratch!.elementCount != topK
            {
                pathBScaleVecScratch = Tensor.filled(
                    scale, shape: [topK], dtype: scoreDt, device: device)
            }
            _ = Ops.mul(
                routerWeightsScratch!, pathBScaleVecScratch!, on: cmd,
                into: routerWeightsScratch!)
        }

        cmd.commit()
        cmd.waitUntilCompleted()

        let indicesHost: [UInt32] = routerIndicesScratch!.toArray(as: UInt32.self)
        let weightsHost: [Float] = routerWeightsScratch!.toFloatArray()
        precondition(indicesHost.count == topK && weightsHost.count == topK)

        let work = device.makeCommandBuffer()
        var accumulator: Tensor?
        for slot in 0 ..< topK {
            let expertId = Int(indicesHost[slot])
            precondition(expertId >= 0 && expertId < gateProj.count)
            let weightTensor = Tensor.filled(
                weightsHost[slot], shape: [hidden], dtype: h.dtype, device: device)
            let expertOut = swiGLU(
                h,
                gateProj: gateProj[expertId],
                upProj: upProj[expertId],
                downProj: downProj[expertId],
                on: work)
            let scaled = Ops.mul(expertOut, weightTensor, on: work)
            accumulator = accumulator.map { Ops.add($0, scaled, on: work) } ?? scaled
        }
        if let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj {
            let sharedOut = swiGLU(h, gateProj: sg, upProj: su, downProj: sd, on: work)
            accumulator = accumulator.map { Ops.add($0, sharedOut, on: work) } ?? sharedOut
        }
        let result = accumulator!
        work.commit()
        return result
    }

    /// Full GPU Path B (expert-indexed). Experimental — quality not yet
    /// gated for Laguna affine4.
    private func decodeGPUPathBFull(
        h: Tensor, logitsTensor: Tensor,
        stacked: StackedInt4Experts,
        expertBiasGPU: Tensor,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let topK = router.topK
        let nExperts = router.nExperts
        let scoreDt = h.dtype
        precondition(topK == 8, "decodeGPUPathBFull: topK must be 8")

        if routerIndicesScratch == nil || routerIndicesScratch!.elementCount != topK {
            routerIndicesScratch = Tensor.empty(shape: [topK], dtype: .u32, device: device)
            routerWeightsScratch = Tensor.empty(shape: [topK], dtype: scoreDt, device: device)
        }
        if unbiasedScoreScratch == nil
            || unbiasedScoreScratch!.elementCount != nExperts
            || unbiasedScoreScratch!.dtype != scoreDt
        {
            unbiasedScoreScratch = Tensor.empty(shape: [nExperts], dtype: scoreDt, device: device)
            biasedScoreScratch = Tensor.empty(shape: [nExperts], dtype: scoreDt, device: device)
        }
        if accumulatorScratch == nil
            || accumulatorScratch!.dtype != h.dtype
            || accumulatorScratch!.elementCount != hidden
        {
            accumulatorScratch = Tensor.empty(shape: [hidden], dtype: h.dtype, device: device)
        }
        let outDim = stacked.moeIntermediate
        while expertGScratches.count < topK {
            expertGScratches.append(
                Tensor.empty(shape: [outDim], dtype: h.dtype, device: device))
            expertUScratches.append(
                Tensor.empty(shape: [outDim], dtype: h.dtype, device: device))
            expertInnerScratches.append(
                Tensor.empty(shape: [outDim], dtype: h.dtype, device: device))
            expertOutScratches.append(
                Tensor.empty(shape: [hidden], dtype: h.dtype, device: device))
        }

        if RouterSetsDebug.enabled {
            debugRouterLogits = logitsTensor
            debugRouterBias = expertBiasGPU
        }
        let unbiased = Ops.sigmoid(logitsTensor, on: cmd, into: unbiasedScoreScratch!)
        let biased = Ops.add(unbiased, expertBiasGPU, on: cmd, into: biasedScoreScratch!)
        Ops.moeRouterTopKBiased(
            scoreBiased: biased, scoreUnbiased: unbiased,
            indicesOut: routerIndicesScratch!,
            weightsOut: routerWeightsScratch!,
            nExperts: nExperts, k: topK, on: cmd)

        let scale = router.routedScalingFactor
        if scale != 1.0 {
            if pathBScaleVecScratch == nil
                || pathBScaleVecScratch!.dtype != scoreDt
                || pathBScaleVecScratch!.elementCount != topK
            {
                pathBScaleVecScratch = Tensor.filled(
                    scale, shape: [topK], dtype: scoreDt, device: device)
            }
            _ = Ops.mul(
                routerWeightsScratch!, pathBScaleVecScratch!, on: cmd,
                into: routerWeightsScratch!)
        }

        var expertIdxScratches: [Tensor] = []
        expertIdxScratches.reserveCapacity(topK)
        for slot in 0 ..< topK {
            expertIdxScratches.append(
                Tensor(
                    buffer: routerIndicesScratch!.buffer,
                    offset: routerIndicesScratch!.offset + slot * 4,
                    shape: [1], dtype: .u32))
        }

        var ws: [Tensor] = []
        var ss: [Tensor] = []
        var bs: [Tensor] = []
        var ins: [Tensor] = []
        var idxs: [Tensor] = []
        var outs0: [Tensor] = []
        ws.reserveCapacity(topK * 2)
        ss.reserveCapacity(topK * 2)
        bs.reserveCapacity(topK * 2)
        ins.reserveCapacity(topK * 2)
        idxs.reserveCapacity(topK * 2)
        outs0.reserveCapacity(topK * 2)
        for slot in 0 ..< topK {
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

        // Dual-queue shared expert (Hy3-style): run full shared SwiGLU on
        // queue B concurrent with routed experts on `cmd`. Escape:
        // BUTTER_LAGUNA_DUAL_QUEUE=0.
        let dualQueue =
            MoEEnvFlags.dualQueueOn
            && sharedGateProj != nil && sharedUpProj != nil && sharedDownProj != nil
        var sharedOut: Tensor?
        var sharedDone: UInt64?
        if dualQueue,
            let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj
        {
            let cmdB = device.makeCommandBufferB()
            let vStart = device.nextSyncValue()
            cmd.encodeSignalEvent(device.syncEvent, value: vStart)
            cmdB.encodeWaitForEvent(device.syncEvent, value: vStart)
            let sGate = sg(h, on: cmdB)
            let sUp = su(h, on: cmdB)
            let sInner = Ops.swiglu(gate: sGate, up: sUp, on: cmdB)
            sharedOut = sd(sInner, on: cmdB)
            let vDone = device.nextSyncValue()
            cmdB.encodeSignalEvent(device.syncEvent, value: vDone)
            cmdB.commit()
            sharedDone = vDone
        }

        let gs = Array(expertGScratches.prefix(topK))
        let us = Array(expertUScratches.prefix(topK))
        let acc = accumulatorScratch!

        // Fused swiglu+down+weighted-sum when mid≤768 and gs=64 (Laguna OK).
        // Escape: BUTTER_LAGUNA_PATHB_FUSED=0 → 3-dispatch chain.
        let useFused =
            MoEEnvFlags.pathBFused
            && stacked.moeIntermediate <= 768 && stacked.groupSize == 64
        if useFused {
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
            let inners = Array(expertInnerScratches.prefix(topK))
            Ops.swigluMany(gates: gs, ups: us, outs: inners, on: cmd)
            var dws: [Tensor] = []
            var dss: [Tensor] = []
            var dbs: [Tensor] = []
            var dins: [Tensor] = []
            var didxs: [Tensor] = []
            var douts: [Tensor] = []
            dws.reserveCapacity(topK)
            dss.reserveCapacity(topK)
            dbs.reserveCapacity(topK)
            dins.reserveCapacity(topK)
            didxs.reserveCapacity(topK)
            douts.reserveCapacity(topK)
            for slot in 0 ..< topK {
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
            scalars.reserveCapacity(topK)
            let wBytes = h.dtype.byteSize
            for slot in 0 ..< topK {
                scalars.append(
                    Tensor(
                        buffer: routerWeightsScratch!.buffer,
                        offset: routerWeightsScratch!.offset + slot * wBytes,
                        shape: [1], dtype: h.dtype))
            }
            Ops.scalarFMAChain8(
                scalars: scalars,
                values: Array(expertOutScratches.prefix(topK)),
                out: acc, on: cmd)
        }

        if let done = sharedDone, let sOut = sharedOut {
            cmd.encodeWaitForEvent(device.syncEvent, value: done)
            return Ops.add(acc, sOut, on: cmd)
        }
        if let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj {
            let shared = swiGLU(h, gateProj: sg, upProj: su, downProj: sd, on: cmd)
            return Ops.add(acc, shared, on: cmd)
        }
        return acc
    }

    /// T32 task 1: sigmoid + bias + top-k select + x-scale, one dispatch
    /// when eligible (`LagunaRouteFuse.fusedRoute`), else the stock
    /// four-dispatch chain (sigmoid, +bias, moeRouterTopKBiased, x scale).
    /// Shared by `decodeGPUPathBNvfp4` and `decodeNineSlot`: both run the
    /// identical post-router-GEMV elementwise tail on the identical
    /// scratch tensors. Writes `routerIndicesScratch!` /
    /// `routerWeightsScratch!` in place; does not touch `logitsTensor`
    /// (the router GEMV itself); see the file header of
    /// `LagunaRouteFuse.swift` for why that GEMV stays out of scope.
    private func runPathBRouteChain(
        logitsTensor: Tensor, expertBiasGPU: Tensor,
        nExperts: Int, topK: Int, scoreDt: DType,
        cmd: MTLCommandBuffer, device: Device
    ) {
        if RouterSetsDebug.enabled {
            debugRouterLogits = logitsTensor
            debugRouterBias = expertBiasGPU
        }
        let scale = router.routedScalingFactor
        // W31: bit-exact single-dispatch fusion, checked first — see
        // `LagunaRouterTop8Fuse.swift`. Takes priority over the older
        // `LagunaRouteFuse` (known to diverge past the first few tokens,
        // see its file header) so a user who has both env vars set gets
        // the correct one.
        if LagunaRouterTop8Fuse.fusedRoute(
            logits: logitsTensor, bias: expertBiasGPU,
            indicesOut: routerIndicesScratch!, weightsOut: routerWeightsScratch!,
            nExperts: nExperts, k: topK, scale: scale,
            on: cmd, device: device)
        {
            return
        }
        if LagunaRouteFuse.fusedRoute(
            logits: logitsTensor, bias: expertBiasGPU,
            indicesOut: routerIndicesScratch!, weightsOut: routerWeightsScratch!,
            nExperts: nExperts, k: topK, scale: scale,
            on: cmd, device: device)
        {
            return
        }
        // Fallback: stock four-dispatch chain.
        let unbiased = Ops.sigmoid(logitsTensor, on: cmd, into: unbiasedScoreScratch!)
        let biased = Ops.add(unbiased, expertBiasGPU, on: cmd, into: biasedScoreScratch!)
        Ops.moeRouterTopKBiased(
            scoreBiased: biased, scoreUnbiased: unbiased,
            indicesOut: routerIndicesScratch!,
            weightsOut: routerWeightsScratch!,
            nExperts: nExperts, k: topK, on: cmd)
        if scale != 1.0 {
            if pathBScaleVecScratch == nil
                || pathBScaleVecScratch!.dtype != scoreDt
                || pathBScaleVecScratch!.elementCount != topK
            {
                pathBScaleVecScratch = Tensor.filled(
                    scale, shape: [topK], dtype: scoreDt, device: device)
            }
            _ = Ops.mul(
                routerWeightsScratch!, pathBScaleVecScratch!, on: cmd,
                into: routerWeightsScratch!)
        }
    }

    /// Path B full with stacked NVFP4 experts (board quant).
    private func decodeGPUPathBNvfp4(
        h: Tensor, logitsTensor: Tensor,
        stacked: StackedNvfp4Experts,
        expertBiasGPU: Tensor,
        cmd: MTLCommandBuffer, device: Device,
        residual: Tensor? = nil,
        foldInto: Tensor? = nil
    ) -> Tensor {
        let topK = router.topK
        let nExperts = router.nExperts
        let scoreDt = h.dtype
        precondition(topK == 8)

        if routerIndicesScratch == nil || routerIndicesScratch!.elementCount != topK {
            routerIndicesScratch = Tensor.empty(shape: [topK], dtype: .u32, device: device)
            routerWeightsScratch = Tensor.empty(shape: [topK], dtype: scoreDt, device: device)
        }
        if unbiasedScoreScratch == nil
            || unbiasedScoreScratch!.elementCount != nExperts
            || unbiasedScoreScratch!.dtype != scoreDt
        {
            unbiasedScoreScratch = Tensor.empty(shape: [nExperts], dtype: scoreDt, device: device)
            biasedScoreScratch = Tensor.empty(shape: [nExperts], dtype: scoreDt, device: device)
        }
        if accumulatorScratch == nil
            || accumulatorScratch!.dtype != h.dtype
            || accumulatorScratch!.elementCount != hidden
        {
            accumulatorScratch = Tensor.empty(shape: [hidden], dtype: h.dtype, device: device)
        }
        let outDim = stacked.moeIntermediate
        while expertGScratches.count < topK {
            expertGScratches.append(
                Tensor.empty(shape: [outDim], dtype: h.dtype, device: device))
            expertUScratches.append(
                Tensor.empty(shape: [outDim], dtype: h.dtype, device: device))
            expertInnerScratches.append(
                Tensor.empty(shape: [outDim], dtype: h.dtype, device: device))
            expertOutScratches.append(
                Tensor.empty(shape: [hidden], dtype: h.dtype, device: device))
        }

        // T28: nine-slot merged dispatch (routed + shared in one threadgroup
        // family) replaces the whole shared-expert dual-queue/single-queue
        // chain below plus the routed swiglu/down-reduce branches further
        // down. Only engages for the exact Laguna XS Path B shape (same
        // guard as `lagunaHot` below) and when the shared expert is
        // NVFP4-backed with a fuseGateUp-compatible gate/up pair. Escape
        // `BUTTER_LAGUNA_NINE_SLOT=0`.
        if LagunaNvfp4Fuse.nineSlotEnabled,
            h.dtype == .bf16, stacked.moeIntermediate == 512, hidden == 2048,
            topK == 8, stacked.blockSize == 16,
            let nineSlot = ensureNineSlotShared(device: device)
        {
            return decodeNineSlot(
                h: h, logitsTensor: logitsTensor, stacked: stacked,
                expertBiasGPU: expertBiasGPU, shared: nineSlot,
                cmd: cmd, device: device, residual: residual,
                foldInto: foldInto)
        }

        // Shared expert depends only on h (not routing). Two schedules:
        //   dual-queue (BUTTER_LAGUNA_DUAL_QUEUE!=0, default ON historically):
        //     queue-B CB + event pair — overlaps router/experts, but costs
        //     ~1 CB + 2 events per MoE layer (~39/token). Can hurt under
        //     host thrash / deep pipe (already many main CBs in flight).
        //   single-queue early (default when DUAL_QUEUE=0): encode shared
        //     first on `cmd` so Metal can overlap independent dispatches
        //     with router+experts without cross-queue events.
        let dualQueue =
            MoEEnvFlags.dualQueueOn
            && sharedGateProj != nil
        var sharedOut: Tensor?
        var sharedDone: UInt64?
        if let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj {
            if dualQueue {
                let cmdB = device.makeCommandBufferB()
                let vStart = device.nextSyncValue()
                cmd.encodeSignalEvent(device.syncEvent, value: vStart)
                cmdB.encodeWaitForEvent(device.syncEvent, value: vStart)
                let sInner: Tensor
                if LagunaNvfp4Fuse.sharedSwigluEnabled,
                    let (fw, fs, fg) = ensureFusedSharedGateUp(
                        gate: sg, up: su, device: device)
                {
                    sInner = LagunaNvfp4Fuse.sharedSwigluQmv(
                        input: h, fusedWeight: fw, fusedScales: fs,
                        global: fg, on: cmdB, device: device)
                } else {
                    let sGate = sg(h, on: cmdB)
                    let sUp = su(h, on: cmdB)
                    sInner = Ops.swiglu(gate: sGate, up: sUp, on: cmdB)
                }
                sharedOut = sd(sInner, on: cmdB)
                let vDone = device.nextSyncValue()
                cmdB.encodeSignalEvent(device.syncEvent, value: vDone)
                cmdB.commit()
                sharedDone = vDone
            } else {
                // Early on main queue (before router) — no events.
                let sInner: Tensor
                if LagunaNvfp4Fuse.sharedSwigluEnabled,
                    let (fw, fs, fg) = ensureFusedSharedGateUp(
                        gate: sg, up: su, device: device)
                {
                    sInner = LagunaNvfp4Fuse.sharedSwigluQmv(
                        input: h, fusedWeight: fw, fusedScales: fs,
                        global: fg, on: cmd, device: device)
                } else {
                    let sGate = sg(h, on: cmd)
                    let sUp = su(h, on: cmd)
                    sInner = Ops.swiglu(gate: sGate, up: sUp, on: cmd)
                }
                sharedOut = sd(sInner, on: cmd)
            }
        }

        runPathBRouteChain(
            logitsTensor: logitsTensor, expertBiasGPU: expertBiasGPU,
            nExperts: nExperts, topK: topK, scoreDt: scoreDt,
            cmd: cmd, device: device)

        var expertIdxScratches: [Tensor] = []
        expertIdxScratches.reserveCapacity(topK)
        for slot in 0 ..< topK {
            expertIdxScratches.append(
                Tensor(
                    buffer: routerIndicesScratch!.buffer,
                    offset: routerIndicesScratch!.offset + slot * 4,
                    shape: [1], dtype: .u32))
        }

        let acc = accumulatorScratch!
        let canFuse =
            h.dtype == .bf16
            && stacked.moeIntermediate <= 768
            && stacked.blockSize == 16

        // Gate+up path (Laguna mid=512, hidden=2048, topK=8, bf16):
        //   ROUTED_SWIGLU (default ON) — one QMV+SwiGLU over 8 experts
        //   GATEUP_FUSE=1 — older per-row concurrent fuse
        //   else — concurrent iron eidx gate/up + swigluMany
        // Down path:
        //   DOWN_REDUCE (default ON) — concurrent 8-slot down+weighted reduce
        //   FUSED=1 serial chain8 (slow) | SLOT_FUSE=1 | stock eidx+FMA
        let lagunaHot =
            canFuse
            && h.dtype == .bf16
            && stacked.moeIntermediate == 512
            && hidden == 2048
            && topK == 8
        let useRoutedSwiglu =
            lagunaHot && LagunaNvfp4Fuse.pathBRoutedSwigluEnabled
        let useGateUpFuse =
            canFuse && !useRoutedSwiglu && LagunaNvfp4Fuse.pathBGateUpFuseEnabled
        let useDownReduce =
            lagunaHot && LagunaNvfp4Fuse.pathBDownReduceEnabled

        let mids: [Tensor]
        if useRoutedSwiglu {
            let inners = Array(expertInnerScratches.prefix(topK))
            LagunaNvfp4Fuse.moeRoutedSwigluQmv(
                input: h,
                gateWeight: stacked.gateWeight,
                gateScales: stacked.gateScales,
                upWeight: stacked.upWeight,
                upScales: stacked.upScales,
                expertIndices: routerIndicesScratch!,
                outputs: inners,
                global: stacked.global,
                on: cmd, device: device)
            mids = inners
        } else if useGateUpFuse {
            let inners = Array(expertInnerScratches.prefix(topK))
            LagunaNvfp4Fuse.moeGateUpSwigluMany(
                gateWeight: stacked.gateWeight,
                gateScales: stacked.gateScales,
                upWeight: stacked.upWeight,
                upScales: stacked.upScales,
                input: h,
                expertIndices: expertIdxScratches,
                outputs: inners,
                inDim: hidden,
                outDim: stacked.moeIntermediate,
                blockSize: stacked.blockSize,
                global: stacked.global,
                on: cmd, device: device)
            mids = inners
        } else if canFuse && LagunaNvfp4Fuse.pathBDownFusedEnabled {
            // Serial chain8 needs separate gate/up buffers; emit them now.
            var ws: [Tensor] = []
            var ss: [Tensor] = []
            var ins: [Tensor] = []
            var idxs: [Tensor] = []
            var outs0: [Tensor] = []
            for slot in 0 ..< topK {
                ws.append(stacked.gateWeight); ss.append(stacked.gateScales)
                ins.append(h); idxs.append(expertIdxScratches[slot])
                outs0.append(expertGScratches[slot])
                ws.append(stacked.upWeight); ss.append(stacked.upScales)
                ins.append(h); idxs.append(expertIdxScratches[slot])
                outs0.append(expertUScratches[slot])
            }
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: ws, scalesStacked: ss,
                inputs: ins, expertIndices: idxs, outputs: outs0,
                blockSize: stacked.blockSize, global: stacked.global, on: cmd)
            mids = []  // unused — chain8 consumes G/U directly
        } else if canFuse && LagunaNvfp4Fuse.pathBSlotFuseEnabled {
            var ws: [Tensor] = []
            var ss: [Tensor] = []
            var ins: [Tensor] = []
            var idxs: [Tensor] = []
            var outs0: [Tensor] = []
            for slot in 0 ..< topK {
                ws.append(stacked.gateWeight); ss.append(stacked.gateScales)
                ins.append(h); idxs.append(expertIdxScratches[slot])
                outs0.append(expertGScratches[slot])
                ws.append(stacked.upWeight); ss.append(stacked.upScales)
                ins.append(h); idxs.append(expertIdxScratches[slot])
                outs0.append(expertUScratches[slot])
            }
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: ws, scalesStacked: ss,
                inputs: ins, expertIndices: idxs, outputs: outs0,
                blockSize: stacked.blockSize, global: stacked.global, on: cmd)
            mids = []  // unused — slot fuse consumes G/U
        } else {
            var ws: [Tensor] = []
            var ss: [Tensor] = []
            var ins: [Tensor] = []
            var idxs: [Tensor] = []
            var outs0: [Tensor] = []
            for slot in 0 ..< topK {
                ws.append(stacked.gateWeight); ss.append(stacked.gateScales)
                ins.append(h); idxs.append(expertIdxScratches[slot])
                outs0.append(expertGScratches[slot])
                ws.append(stacked.upWeight); ss.append(stacked.upScales)
                ins.append(h); idxs.append(expertIdxScratches[slot])
                outs0.append(expertUScratches[slot])
            }
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: ws, scalesStacked: ss,
                inputs: ins, expertIndices: idxs, outputs: outs0,
                blockSize: stacked.blockSize, global: stacked.global, on: cmd)
            let gs = Array(expertGScratches.prefix(topK))
            let us = Array(expertUScratches.prefix(topK))
            let inners = Array(expertInnerScratches.prefix(topK))
            Ops.swigluMany(gates: gs, ups: us, outs: inners, on: cmd)
            mids = inners
        }

        if canFuse && LagunaNvfp4Fuse.pathBDownFusedEnabled
            && !useRoutedSwiglu && !useGateUpFuse
        {
            // Serial chain8 needs separate gate/up buffers.
            let gs = Array(expertGScratches.prefix(topK))
            let us = Array(expertUScratches.prefix(topK))
            LagunaNvfp4Fuse.moeDownSwigluAccumChain8(
                gates: gs, ups: us,
                expertIndices: routerIndicesScratch!,
                slotWeights: routerWeightsScratch!,
                downWeight: stacked.downWeight,
                downScales: stacked.downScales,
                output: acc,
                inDim: stacked.moeIntermediate,
                outDim: hidden,
                blockSize: stacked.blockSize,
                global: stacked.global,
                on: cmd, device: device)
        } else if canFuse && LagunaNvfp4Fuse.pathBSlotFuseEnabled
            && !useRoutedSwiglu && !useGateUpFuse
        {
            let gs = Array(expertGScratches.prefix(topK))
            let us = Array(expertUScratches.prefix(topK))
            var douts: [Tensor] = []
            for slot in 0 ..< topK { douts.append(expertOutScratches[slot]) }
            LagunaNvfp4Fuse.moeSlotSwigluDownMany(
                gates: gs, ups: us,
                expertIndices: expertIdxScratches,
                downWeight: stacked.downWeight,
                downScales: stacked.downScales,
                outputs: douts,
                inDim: stacked.moeIntermediate,
                outDim: hidden,
                blockSize: stacked.blockSize,
                global: stacked.global,
                on: cmd, device: device)
            var scalars: [Tensor] = []
            let wBytes = h.dtype.byteSize
            for slot in 0 ..< topK {
                scalars.append(
                    Tensor(
                        buffer: routerWeightsScratch!.buffer,
                        offset: routerWeightsScratch!.offset + slot * wBytes,
                        shape: [1], dtype: h.dtype))
            }
            Ops.scalarFMAChain8(
                scalars: scalars,
                values: Array(expertOutScratches.prefix(topK)),
                out: acc, on: cmd)
        } else if useDownReduce && !mids.isEmpty {
            // mlxfast FUSED_ROUTED_DOWN_REDUCE: one concurrent 8-slot
            // down QMV + BF16 weighted reduce → acc.
            LagunaNvfp4Fuse.moeRoutedDownReduce(
                mids: mids,
                expertIndices: routerIndicesScratch!,
                slotWeights: routerWeightsScratch!,
                downWeight: stacked.downWeight,
                downScales: stacked.downScales,
                output: acc,
                global: stacked.global,
                on: cmd, device: device)
        } else {
            // Stock: concurrent eidx down + scalarFMAChain8.
            precondition(!mids.isEmpty, "decodeGPUPathBNvfp4: missing mids")
            var dws: [Tensor] = []
            var dss: [Tensor] = []
            var dins: [Tensor] = []
            var didxs: [Tensor] = []
            var douts: [Tensor] = []
            for slot in 0 ..< topK {
                dws.append(stacked.downWeight)
                dss.append(stacked.downScales)
                dins.append(mids[slot])
                didxs.append(expertIdxScratches[slot])
                douts.append(expertOutScratches[slot])
            }
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: dws, scalesStacked: dss,
                inputs: dins, expertIndices: didxs, outputs: douts,
                blockSize: stacked.blockSize, global: stacked.global, on: cmd)
            var scalars: [Tensor] = []
            let wBytes = h.dtype.byteSize
            for slot in 0 ..< topK {
                scalars.append(
                    Tensor(
                        buffer: routerWeightsScratch!.buffer,
                        offset: routerWeightsScratch!.offset + slot * wBytes,
                        shape: [1], dtype: h.dtype))
            }
            Ops.scalarFMAChain8(
                scalars: scalars,
                values: Array(expertOutScratches.prefix(topK)),
                out: acc, on: cmd)
        }

        if let done = sharedDone, let sOut = sharedOut {
            cmd.encodeWaitForEvent(device.syncEvent, value: done)
            return Ops.add(acc, sOut, on: cmd)
        }
        if let sOut = sharedOut {
            // Single-queue early path: shared already encoded above.
            return Ops.add(acc, sOut, on: cmd)
        }
        return acc
    }

    /// T28 nine-slot dispatch: cached raw shared-expert NVFP4 fields (fused
    /// [gate;up] bank + standalone down bank).
    private struct NineSlotShared {
        let fusedWeight: Tensor
        let fusedScales: Tensor
        let fusedGlobal: Float
        let downWeight: Tensor
        let downScales: Tensor
        let downGlobal: Float
    }

    /// Lazy-attempt build of the nine-slot shared-expert fields (fused
    /// gate/up bank via `ensureFusedSharedGateUp` + raw down bank). Mirrors
    /// `ensureFusedSharedGateUp`'s single-attempt-then-cache pattern; returns
    /// nil (permanently, for this layer instance) if the shared expert isn't
    /// NVFP4-backed or its shapes don't match the fuseGateUp precondition.
    private func ensureNineSlotShared(device: Device) -> NineSlotShared? {
        if let fw = fusedSharedGateUpWeight, let fs = fusedSharedGateUpScales,
            let dw = nineSlotSharedDownWeight, let ds = nineSlotSharedDownScales
        {
            return NineSlotShared(
                fusedWeight: fw, fusedScales: fs, fusedGlobal: fusedSharedGateUpGlobal,
                downWeight: dw, downScales: ds, downGlobal: nineSlotSharedDownGlobal)
        }
        if nineSlotAttempted { return nil }
        nineSlotAttempted = true
        guard let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj,
            let (fw, fs, fg) = ensureFusedSharedGateUp(gate: sg, up: su, device: device),
            let dLinear = sd.inner as? Nvfp4Linear,
            dLinear.outDim == hidden, dLinear.inDim == 512
        else { return nil }
        nineSlotSharedDownWeight = dLinear.weight
        nineSlotSharedDownScales = dLinear.scales
        nineSlotSharedDownGlobal = dLinear.global
        return NineSlotShared(
            fusedWeight: fw, fusedScales: fs, fusedGlobal: fg,
            downWeight: dLinear.weight, downScales: dLinear.scales,
            downGlobal: dLinear.global)
    }

    /// T28 nine-slot merged MoE decode: router (unchanged) + one fused
    /// routed+shared SwiGLU dispatch + one fused routed+shared down-reduce
    /// dispatch. Returns the combined (routed + shared) MoE output directly,
    /// with no separate `Ops.add(acc, sharedOut)`, since the down-reduce
    /// kernel already folds the shared leg into its weighted-sum accumulator.
    ///
    /// T28 follow-up (`BUTTER_LAGUNA_GATEUP_SPLIT=1`,
    /// `LagunaNvfp4Fuse.gateUpSplitEnabled`): the gate-up half only can be
    /// split back into a routed-8 dispatch + a shared-1 dispatch (one
    /// concurrent-dispatch encoder, shared first) instead of the single
    /// merged nine-slot gate-up kernel; see
    /// `LagunaNvfp4Fuse.moeGateUpSplitSwigluQmv` for the dependency-overlap
    /// rationale. The down+residual leg (`moeNineSlotDownReduce`) is NOT
    /// split and runs identically either way.
    ///
    /// T43 (`BUTTER_LAGUNA_DOWN_RESIDUAL_FOLD=1`,
    /// `LagunaNvfp4Fuse.downResidualFoldEnabled`): when `residual` is
    /// non-nil and the flag is on, the down-reduce dispatch also folds in
    /// `+ residual` (sets `self.lastResidualFolded = true` so
    /// `decodeFoldingResidual` can tell the caller it no longer needs its
    /// own `Ops.add`). Otherwise behaves exactly as before this note.
    private func decodeNineSlot(
        h: Tensor, logitsTensor: Tensor,
        stacked: StackedNvfp4Experts, expertBiasGPU: Tensor,
        shared: NineSlotShared,
        cmd: MTLCommandBuffer, device: Device,
        residual: Tensor? = nil,
        foldInto: Tensor? = nil
    ) -> Tensor {
        let topK = router.topK
        let nExperts = router.nExperts
        let scoreDt = h.dtype

        // T39: per-kernel decode GPU-time probe. `localCmd` starts aliased
        // to the caller's `cmd` (== `workBuf` in `LagunaLayer.forward`); the
        // first checkpoint below commits that buffer (so the caller's own
        // `workBuf.status == .committed` check already in place for the
        // host/hybrid MoE path picks it up correctly and allocates its own
        // fresh residual-add buffer) and every checkpoint after that keeps
        // encoding on progressively fresher buffers. No-op when the env var
        // is unset: `localCmd` never diverges from `cmd`, dispatches land on
        // the caller's buffer exactly as before. See
        // benchmarks/laguna-xs/README.md T39.
        let dkTiming = LagunaDecodeKernelTiming.enabled
        var localCmd = cmd
        func dk(_ label: String) {
            guard dkTiming else { return }
            localCmd = LagunaDecodeKernelTiming.checkpoint(label, cmd: localCmd, device: device)
        }

        runPathBRouteChain(
            logitsTensor: logitsTensor, expertBiasGPU: expertBiasGPU,
            nExperts: nExperts, topK: topK, scoreDt: scoreDt,
            cmd: localCmd, device: device)
        dk("router_route")

        if sharedMidScratch == nil || sharedMidScratch!.dtype != h.dtype {
            sharedMidScratch = Tensor.empty(shape: [512], dtype: h.dtype, device: device)
        }
        let inners = Array(expertInnerScratches.prefix(topK))
        if LagunaNvfp4Fuse.gateUpSplitEnabled {
            // T28 follow-up: routed-8 + shared-1 on one concurrent-dispatch
            // encoder (shared first) instead of the merged nine-slot
            // gate-up kernel. See moeGateUpSplitSwigluQmv doc comment.
            LagunaNvfp4Fuse.moeGateUpSplitSwigluQmv(
                input: h,
                gateWeight: stacked.gateWeight, gateScales: stacked.gateScales,
                upWeight: stacked.upWeight, upScales: stacked.upScales,
                expertIndices: routerIndicesScratch!,
                routedOutputs: inners,
                sharedWeight: shared.fusedWeight, sharedScales: shared.fusedScales,
                sharedOutput: sharedMidScratch!,
                global: stacked.global, sharedGlobal: shared.fusedGlobal,
                on: localCmd, device: device)
        } else {
            LagunaNvfp4Fuse.moeNineSlotSwigluQmv(
                input: h,
                gateWeight: stacked.gateWeight, gateScales: stacked.gateScales,
                upWeight: stacked.upWeight, upScales: stacked.upScales,
                expertIndices: routerIndicesScratch!,
                routedOutputs: inners,
                sharedWeight: shared.fusedWeight, sharedScales: shared.fusedScales,
                sharedOutput: sharedMidScratch!,
                global: stacked.global, sharedGlobal: shared.fusedGlobal,
                on: localCmd, device: device)
        }
        dk("moe_swiglu")

        let acc = accumulatorScratch!
        // T43: fold `+ residual` into this dispatch when the caller passed
        // one, the flag is on, and `residual` is a matching [hidden] bf16
        // tensor (same guard shape the nine-slot branch above already
        // established: h.dtype == .bf16, hidden == 2048). Declines silently
        // (stock two-dispatch chain) on any mismatch instead of trapping,
        // since `residual` here is caller-supplied, not derived from the
        // same shape preconditions as the rest of this function's tensors.
        let foldResidual: Tensor? =
            if let residual, LagunaNvfp4Fuse.downResidualFoldEnabled,
                residual.dtype == acc.dtype, residual.elementCount == hidden
            {
                residual
            } else {
                nil
            }
        // T43 fix: when folding, the layer OUTPUT must not alias
        // accumulatorScratch (the stock chain's own comment in
        // LagunaText.forward documents that hazard; its Ops.add writes to
        // a dedicated ffnOutScratch for exactly this reason). Bind the
        // caller-provided destination as the kernel's output instead.
        let downOut = (foldResidual != nil ? foldInto : nil) ?? acc
        LagunaNvfp4Fuse.moeNineSlotDownReduce(
            mids: inners, sharedMid: sharedMidScratch!,
            expertIndices: routerIndicesScratch!, slotWeights: routerWeightsScratch!,
            downWeight: stacked.downWeight, downScales: stacked.downScales,
            sharedDownWeight: shared.downWeight, sharedDownScales: shared.downScales,
            output: downOut,
            global: stacked.global, sharedGlobal: shared.downGlobal,
            residual: foldResidual,
            on: localCmd, device: device)
        if foldResidual != nil {
            lastResidualFolded = true
        }
        dk("moe_down_reduce")
        return downOut
    }

    /// Lazy build of fused shared [gate; up] NVFP4 banks (once per layer).
    private func ensureFusedSharedGateUp(
        gate: AnyLinear, up: AnyLinear, device: Device
    ) -> (Tensor, Tensor, Float)? {
        if let w = fusedSharedGateUpWeight, let s = fusedSharedGateUpScales {
            return (w, s, fusedSharedGateUpGlobal)
        }
        if fusedSharedGateUpAttempted { return nil }
        fusedSharedGateUpAttempted = true
        guard let g = gate.inner as? Nvfp4Linear,
            let u = up.inner as? Nvfp4Linear,
            let fused = LagunaNvfp4Fuse.fuseGateUp(gate: g, up: u, device: device)
        else { return nil }
        fusedSharedGateUpWeight = fused.weight
        fusedSharedGateUpScales = fused.scales
        fusedSharedGateUpGlobal = fused.global
        return fused
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
