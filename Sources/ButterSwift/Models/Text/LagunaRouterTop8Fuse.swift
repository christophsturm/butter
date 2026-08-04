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
// W31: bit-exact fused replacement for `runPathBRouteChain`'s stock
// four-dispatch decode router chain (`Ops.sigmoid` -> `Ops.add`
// (+expertBias) -> `Ops.moeRouterTopKBiased` (top-k select) -> `Ops.mul`
// (routedScalingFactor)). One dispatch instead of four saves ~3 dispatch
// launches x 39 MoE layers/token on Laguna XS decode — see
// `LagunaRouteFuse.swift`'s file header for the general "merge tiny
// elementwise dispatches" rationale (T32 task 1); this file targets the
// SAME chain but is a from-scratch kernel, NOT a reuse of that one.
//
// `LagunaRouteFuse` (BUTTER_LAGUNA_FUSED_ROUTE, default OFF) already
// attempted this fusion and shipped disabled: its header documents that
// it "keeps everything in float until one final store" instead of
// reproducing the stock chain's bf16 round-trips at each stage, which
// both (a) measured flat (-2.6% wall, not a win) and (b) can flip the
// top-k tie-break on near-equal scores because comparisons run against
// un-rounded float sigmoid+bias instead of the bf16-rounded values the
// stock chain actually compares. This kernel exists to fix (b): every
// point where the stock chain round-trips through a bf16 store gets an
// explicit `bfloat(...)` cast here, so the VALUES being compared and
// accumulated are bit-identical to stock, not just "close":
//
//   1. `Ops.sigmoid(logits) -> unbiasedScoreScratch` (bf16 store)         — replicated as `bfloat(sigmoid_f)`
//   2. `Ops.add(unbiased, bias) -> biasedScoreScratch` (bf16 store)       — replicated as `bfloat(float(unbiased_bf) + float(bias))`
//   3. `Ops.moeRouterTopKBiased`: selection compares the bf16-rounded
//      biased scores (stage 2's output, not raw sigmoid+bias), same
//      per-lane strided scan + already-chosen mask + simd_max/simd_min
//      index-consensus tie-break (ties -> smallest index wins, because
//      the per-lane scan uses strict `>` so equal-valued earlier
//      candidates aren't displaced, and cross-lane ties resolve via
//      `simd_min` on index). Weights come from the UNBIASED bf16 score
//      at the chosen index (bias is selection-only, never enters the
//      weight), renormalized by the sum of the k chosen unbiased scores,
//      then bf16-rounded (`weightsOut` store) — this is stage 3's own
//      round.
//   4. `Ops.mul(weights, routedScalingFactor) -> weightsOut` (bf16
//      store, only when `scale != 1.0`) — replicated as a SEPARATE
//      `bfloat(float(weight_bf) * float(scale_bf))`, i.e. the stock
//      chain's double-rounding (round to bf16 at stage 3, then round
//      AGAIN after the scale multiply) is preserved rather than folded
//      into one rounding.
//
// Sigmoid/add/mul/top-k formulas themselves are copied verbatim from
// `Sources/IronSwift/Resources/kernels/iron_sigmoid_bf16.metal`,
// `vector_add_bf16.metal`, `iron_mul_bf16.metal`, and
// `iron_moe_router_topk_biased_bf16.metal` — see those files for the
// oracle math this kernel must stay byte-identical to. `unbiased`
// sigmoid values are recomputed inline wherever the stock chain would
// have re-read `unbiasedScoreScratch` from memory (during selection AND
// at the final weight lookup) rather than cached in threadgroup memory:
// the sigmoid formula is a pure function of the (unchanged) input logit,
// so recomputing it is bit-identical to caching it, just trades a `<=7`
// redundant `exp()` per lane per slot for skipping a threadgroup-memory
// round-trip — irrelevant here since this kernel's whole reason to
// exist is dispatch-count, not ALU or bandwidth (single 256-wide row,
// one threadgroup, 32 lanes).
//
// Escape: `BUTTER_LAGUNA_ROUTER_TOP8_FUSED=0` (default) restores the
// stock four-dispatch chain. Opt in with `=1` only after confirming
// byte-identical decode output against the stock chain on your model —
// see `benchmarks/laguna-xs/README.md` W31 entry for the validation
// this shipped with.
//
// Reachability gotcha (cost real time to discover): `runPathBRouteChain`
// is only called from `decodeGPUPathBNvfp4` / `decodeNineSlot`, which
// `MoELayer.decodeImpl` only reaches when `stackedNvfp4Experts` is
// populated — i.e. an NVFP4-quantized checkpoint
// (`~/models/laguna-xs-2.1-nvfp4`), NOT the affine-4bit checkpoint
// (`~/models/laguna-xs-2.1-4bit`), which instead runs `decodeGPUPathB`
// -> `decodeGPUPathBHybrid`/`decodeGPUPathBFull`'s OWN separately
// inlined stock four-dispatch chain and never touches this file. Test
// against the NVFP4 checkpoint, or this flag silently no-ops.

import Foundation
import Metal
import IronSwift

enum LagunaRouterTop8Fuse {
    /// Default ON (`=0` to disable). Byte-identical to the stock
    /// four-dispatch chain by construction (see file header for the
    /// stage-by-stage rounding replication) and digest-verified at
    /// temp 0 over 200-token runs on two prompts; paired ABBA measured
    /// +4.6% decode from the three saved dispatches per MoE layer.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_ROUTER_TOP8_FUSED"] != "0"
    }

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        let opts = MTLCompileOptions()
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    // One threadgroup (32 lanes = 1 simdgroup), one row (decode = T=1).
    // Selection loop structure mirrors `iron_moe_router_topk_biased_bf16`
    // exactly; the only difference from that kernel is that sigmoid+bias
    // are computed inline (with the same intermediate bf16 rounding the
    // stock chain performs via its separate sigmoid/add dispatches)
    // instead of being read from precomputed scratch tensors.
    private static let routeSrc = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_router_top8_fused_bf16(
            const device bfloat *logits [[buffer(0)]],
            const device bfloat *bias [[buffer(1)]],
            device uint *indices_out [[buffer(2)]],
            device bfloat *weights_out [[buffer(3)]],
            constant uint &n_experts [[buffer(4)]],
            constant uint &k [[buffer(5)]],
            constant float &scale [[buffer(6)]],
            constant uint &apply_scale [[buffer(7)]],
            uint lane [[thread_position_in_threadgroup]])
        {
            threadgroup float tg_chosen_idx[32];
            uint n_per_lane = (n_experts + 31u) / 32u;

            for (uint slot = 0; slot < k; ++slot) {
                float best_val = -INFINITY;
                uint best_idx = 0;
                for (uint i = 0; i < n_per_lane; ++i) {
                    uint j = i * 32u + lane;
                    if (j < n_experts) {
                        // Stage 1 (Ops.sigmoid store) + stage 2 (Ops.add
                        // store): round to bf16 at EACH stage, matching
                        // the stock chain's two separate dispatches.
                        float sig = 1.0f / (1.0f + exp(-float(logits[j])));
                        bfloat unbiased_bf = bfloat(sig);
                        bfloat biased_bf = bfloat(float(unbiased_bf) + float(bias[j]));
                        float score = float(biased_bf);

                        bool already_chosen = false;
                        for (uint p = 0; p < slot; ++p) {
                            if (j == uint(tg_chosen_idx[p])) {
                                already_chosen = true;
                                break;
                            }
                        }
                        float candidate = already_chosen ? -INFINITY : score;
                        if (candidate > best_val) {
                            best_val = candidate;
                            best_idx = j;
                        }
                    }
                }
                float global_best_val = simd_max(best_val);
                uint idx_or_sentinel =
                    (best_val == global_best_val) ? best_idx : 0xFFFFFFFFu;
                float global_best_idx = simd_min(float(idx_or_sentinel));
                if (lane == 0) {
                    tg_chosen_idx[slot] = global_best_idx;
                }
                simdgroup_barrier(mem_flags::mem_none);
            }

            bool active = lane < k;
            uint my_idx = active ? uint(tg_chosen_idx[lane]) : 0u;
            // Re-read (recompute) the unbiased sigmoid at the chosen
            // index — bit-identical to the stock kernel's
            // `score_unbiased[my_idx]` memory read since it's the same
            // deterministic formula over the same input logit.
            float my_sig = 1.0f / (1.0f + exp(-float(logits[my_idx])));
            bfloat my_unbiased_bf = bfloat(my_sig);
            float my_unbiased = active ? float(my_unbiased_bf) : 0.0f;
            float sum_chosen = simd_sum(my_unbiased);
            float weight_f = my_unbiased / sum_chosen;
            // Stage 3 (moeRouterTopKBiased's own weightsOut store).
            bfloat weight_bf = bfloat(weight_f);
            // Stage 4 (Ops.mul store) — only when the stock chain would
            // have run it (`scale != 1.0`); otherwise stage 3's value is
            // final, matching stock exactly (no extra rounding).
            if (apply_scale != 0u) {
                bfloat scale_bf = bfloat(scale);
                weight_bf = bfloat(float(weight_bf) * float(scale_bf));
            }
            if (active) {
                indices_out[lane] = my_idx;
                weights_out[lane] = weight_bf;
            }
        }
        """

    /// Fused sigmoid + bias + top-k select + x-scale in one dispatch,
    /// bit-exact against the stock four-dispatch chain (see file
    /// header). Declines (returns `false`, outputs left untouched)
    /// unless enabled and every shape/dtype precondition holds; caller
    /// falls back to the stock chain. `k` must be <= 32 (one simdgroup
    /// lane per selected slot) and `nExperts` must fit in the strided
    /// per-lane scan (no hard cap here — `n_per_lane = ceil(nExperts/32)`
    /// — but this is sized for Laguna's 256-expert router).
    static func fusedRoute(
        logits: Tensor, bias: Tensor,
        indicesOut: Tensor, weightsOut: Tensor,
        nExperts: Int, k: Int, scale: Float,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Bool {
        guard enabled,
            k <= 32,
            logits.dtype == .bf16, bias.dtype == .bf16, weightsOut.dtype == .bf16,
            indicesOut.dtype == .u32,
            logits.elementCount == nExperts, bias.elementCount == nExperts,
            indicesOut.elementCount == k, weightsOut.elementCount == k
        else { return false }

        let p = pso(
            "laguna_router_top8_fused_bf16", source: routeSrc, device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(logits.buffer, offset: logits.offset, index: 0)
        enc.setBuffer(bias.buffer, offset: bias.offset, index: 1)
        enc.setBuffer(indicesOut.buffer, offset: indicesOut.offset, index: 2)
        enc.setBuffer(weightsOut.buffer, offset: weightsOut.offset, index: 3)
        var ne = UInt32(nExperts)
        var kk = UInt32(k)
        var sc = scale
        var applyScale: UInt32 = scale != 1.0 ? 1 : 0
        enc.setBytes(&ne, length: 4, index: 4)
        enc.setBytes(&kk, length: 4, index: 5)
        enc.setBytes(&sc, length: 4, index: 6)
        enc.setBytes(&applyScale, length: 4, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }
}
