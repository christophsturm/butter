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
// T32 task 1: MoE route-side dispatch fusion for Laguna's Path B
// (sigmoid+bias top-k) decode router. Hand-written live-compiled Metal,
// same `pso()` + `makeLibrary(source:)` pattern as `LagunaNvfp4Fuse` /
// `LagunaAttnInt8` / `LagunaGatedOProjFuse`: butter's `iron` DSL has no
// primitive for this shape.
//
// The stock chain (`MoELayer.decodeNineSlot` / `decodeGPUPathBNvfp4`)
// runs four separate dispatches over the already-computed router GEMV
// logits: `Ops.sigmoid`, `Ops.add` (expert bias), `Ops.moeRouterTopKBiased`
// (top-k selection + weight), `Ops.mul` (routed_scaling_factor). Each of
// the first, second, and fourth is a tiny elementwise op over `nExperts`
// (256) or `topK` (8) elements: dispatch-launch overhead dominates their
// real cost on this host (see benchmarks/laguna-xs/README.md T31 §2/§3),
// the same "merge small elementwise dispatches" class of win T28 lever 1
// already proved for the FFN gate/up/down chain.
//
// This file folds all four into ONE dispatch: `laguna_fused_route_bf16`
// reads the raw router logits + raw expert bias directly, computes the
// sigmoid and bias-add inline (per candidate, not materialized to a
// scratch tensor), runs the identical 32-lane top-k selection algorithm
// as `iron_moe_router_topk_biased_{bf16,f16,f32}`
// (`Sources/IronSwift/Resources/kernels/iron_moe_router_topk_biased_bf16.metal`),
// same simd_max/simd_min index-consensus loop, same unconditional
// renorm-by-sum-of-chosen-unbiased weight), then multiplies by
// `routedScalingFactor` before the single bf16 store. Deliberately does
// NOT touch the router GEMV itself (`gate(h, on: cmd)` / `Ops.gemv`):
// T28 lever 2 found that folding the GEMV into a single-threadgroup
// dispatch serializes work `Ops.gemv` otherwise spreads across up to 256
// threadgroups, costing far more than the one dispatch it saves. Only the
// tiny post-GEMV elementwise tail is in scope here.
//
// Not claimed byte-identical: the stock chain rounds to bf16 three times
// (sigmoid store, biased-add store, then the topk kernel's own weight
// round) before an optional fourth round from the scale multiply; this
// kernel keeps everything in float until one final store. Same FP-reorder
// class as every other decode fuse in this file family (T24, T28):
// expect coherent-but-diverging generations past the first few tokens,
// not correctness breakage.
//
// Escape: `BUTTER_LAGUNA_FUSED_ROUTE=0` restores the stock four-dispatch
// chain byte-for-byte. Default ON.

import Foundation
import Metal
import IronSwift

enum LagunaRouteFuse {
    /// Default OFF (T32: structurally sound but measured flat, and expert
    /// tie-breaks can diverge from the stock chain). Opt in with
    /// `BUTTER_LAGUNA_FUSED_ROUTE=1`; the stock
    /// sigmoid / +bias / top-k / x-scale four-dispatch chain.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_FUSED_ROUTE"] == "1"
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

    // One threadgroup (32 lanes = 1 simdgroup), one row. Mirrors
    // `iron_moe_router_topk_biased_bf16`'s selection algorithm exactly
    // (same index-consensus via simd_max on value / simd_min on index),
    // but computes sigmoid(logit) + bias inline per candidate instead of
    // reading precomputed scratch tensors, and folds the final
    // `x routedScalingFactor` multiply into the same store.
    private static let routeSrc = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_fused_route_bf16(
            const device bfloat *logits [[buffer(0)]],
            const device bfloat *bias [[buffer(1)]],
            device uint *indices_out [[buffer(2)]],
            device bfloat *weights_out [[buffer(3)]],
            constant uint &n_experts [[buffer(4)]],
            constant uint &k [[buffer(5)]],
            constant float &scale [[buffer(6)]],
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
                        float s = 1.0f / (1.0f + exp(-float(logits[j])));
                        float biased = s + float(bias[j]);
                        bool already_chosen = false;
                        for (uint p = 0; p < slot; ++p) {
                            if (j == uint(tg_chosen_idx[p])) {
                                already_chosen = true;
                                break;
                            }
                        }
                        float candidate = already_chosen ? -INFINITY : biased;
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
            float my_unbiased =
                active ? (1.0f / (1.0f + exp(-float(logits[my_idx])))) : 0.0f;
            float sum_chosen = simd_sum(my_unbiased);
            float weight = (my_unbiased / sum_chosen) * scale;
            if (active) {
                indices_out[lane] = my_idx;
                weights_out[lane] = bfloat(weight);
            }
        }
        """

    /// Fused sigmoid + bias + top-k select + x-scale in one dispatch.
    /// Declines (returns `false`, outputs left untouched) unless enabled
    /// and every shape/dtype precondition holds; caller falls back to the
    /// stock four-dispatch chain. `k` must be <= 32 (one simdgroup lane
    /// per selected slot).
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

        let p = pso("laguna_fused_route_bf16", source: routeSrc, device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(logits.buffer, offset: logits.offset, index: 0)
        enc.setBuffer(bias.buffer, offset: bias.offset, index: 1)
        enc.setBuffer(indicesOut.buffer, offset: indicesOut.offset, index: 2)
        enc.setBuffer(weightsOut.buffer, offset: weightsOut.offset, index: 3)
        var ne = UInt32(nExperts)
        var kk = UInt32(k)
        var sc = scale
        enc.setBytes(&ne, length: 4, index: 4)
        enc.setBytes(&kk, length: 4, index: 5)
        enc.setBytes(&sc, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }
}
