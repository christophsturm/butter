// Copyright 2026 Tom Turney (@TheTom)
//
// Decode-path NVFP4 fusions for NemotronH-Lightning's gate-less
// squared-ReLU routed experts (`down(relu(up(x))^2)`), mirroring
// `LagunaNvfp4Fuse`'s routed SwiGLU Path B fuses but adapted for
// Lightning's shapes (hidden=2688, moeIntermediate=1856, topK=6) and
// activation (no gate projection — squared-ReLU instead of SwiGLU).
//
// `NemotronHNvfp4MoELayer.decodeGPURouted`'s per-slot loop issued 6x
// expert-indexed up GEMV + 6x relu + 6x square + 6x expert-indexed down
// GEMV + 6x headGateMul + 5x accumulate-add = 35 dispatches/layer x 23
// MoE layers just for the routed-expert math (see F-126 dispatch-count
// note in NemotronHNvfp4.swift). These two kernels collapse that to 2
// dispatches total:
//
//   moeRoutedRelu2Qmv    — up GEMV + relu + square for all 6 slots, 1 dispatch.
//   moeRoutedDownReduce  — down GEMV + weighted reduce for all 6 slots, 1 dispatch.
//
// Escape: `BUTTER_NEMOTRON_NVFP4_FUSED_EXPERTS=0` falls back to the
// original per-slot loop (kept byte-for-byte in
// `NemotronHNvfp4MoELayer.decodeGPURouted`).
//
// Kernel packaging: same scheme as LagunaNvfp4Fuse — the canonical
// source lives in Sources/IronSwift/Resources/kernel_overlays/*.metal,
// applied + linked into kernels_extra.metallib by
// scripts/link-extra-metals.sh (`make link-extra-metals`). `pso()`
// prefers the metallib PSO and falls back to live `makeLibrary(source:)`
// from the identical string embedded below when the overlay hasn't been
// linked into this build.

import Foundation
import Metal
import IronSwift

enum NemotronLightningNvfp4Fuse {
    /// Default ON — escape `BUTTER_NEMOTRON_NVFP4_FUSED_EXPERTS=0` to
    /// fall back to the original per-slot GEMV/relu/square/GEMV/
    /// headGateMul/add loop for A/B comparison or regression triage.
    static let fusedExpertsEnabled: Bool =
        ButterEnv["BUTTER_NEMOTRON_NVFP4_FUSED_EXPERTS"] != "0"

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static let header = """
    #include <metal_stdlib>
    using namespace metal;

    static inline float iron_nvfp4_e4m3_scale(uchar bits) {
        ushort h = ushort((uint(bits) + (uint(bits) & 128u)) << 7);
        return float(as_type<half>(h)) * 256.0f;
    }

    static inline float iron_nvfp4_pack_dot_fast(
        uint packed, float scale, const thread float* input, uint p_off
    ) {
        float accum = 0.0f;
        const uint xe = packed & 0x0F0F0F0Fu;
        const uint ge = xe | (xe << 3);
        const uint yo = packed & 0xF0F0F0F0u;
        const uint go = yo | (yo >> 3);
        const uint M = 0x8E008E00u;
        const uint p0 = (ge << 9) & M;
        const uint p1 = (go << 8) & M;
        const uint p2 = (ge << 1) & M;
        const uint p3 = go & M;
        const float2 v04 = float2(as_type<half2>(p0));
        const float2 v15 = float2(as_type<half2>(p1));
        const float2 v26 = float2(as_type<half2>(p2));
        const float2 v37 = float2(as_type<half2>(p3));
        accum += v04.x * scale * input[p_off + 0u];
        accum += v15.x * scale * input[p_off + 1u];
        accum += v26.x * scale * input[p_off + 2u];
        accum += v37.x * scale * input[p_off + 3u];
        accum += v04.y * scale * input[p_off + 4u];
        accum += v15.y * scale * input[p_off + 5u];
        accum += v26.y * scale * input[p_off + 6u];
        accum += v37.y * scale * input[p_off + 7u];
        return accum;
    }

    static inline float iron_nvfp4_qdot_16(
        const device uchar* weight_bytes,
        const thread float* input,
        float scale
    ) {
        float accum = 0.0f;
        const device uint* packs = (const device uint*)weight_bytes;
        const float scale16k = scale * 16384.0f;
        accum += iron_nvfp4_pack_dot_fast(packs[0], scale16k, input, 0u);
        accum += iron_nvfp4_pack_dot_fast(packs[1], scale16k, input, 8u);
        return accum;
    }
    """

    // MARK: - Fused routed up-projection + squared-ReLU

    private static let routedRelu2Src = header + """
    kernel void nemotron_moe_routed_relu2_qmv_nvfp4_bf16(
        device const bfloat *input [[buffer(0)]],
        device const uint *up_weight [[buffer(1)]],
        device const uchar *up_scales [[buffer(2)]],
        device const uint *indices [[buffer(3)]],
        device bfloat *activated_0 [[buffer(4)]],
        device bfloat *activated_1 [[buffer(5)]],
        device bfloat *activated_2 [[buffer(6)]],
        device bfloat *activated_3 [[buffer(7)]],
        device bfloat *activated_4 [[buffer(8)]],
        device bfloat *activated_5 [[buffer(9)]],
        constant float &global_scale [[buffer(10)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        constexpr uint input_width = 2688u;
        constexpr uint output_width = 1856u;
        constexpr uint packed_row_bytes = input_width / 2u;
        constexpr uint scale_row_bytes = input_width / 16u;
        constexpr uint packed_expert_bytes = output_width * packed_row_bytes;
        constexpr uint scale_expert_bytes = output_width * scale_row_bytes;
        constexpr uint n_blocks = input_width / 16u;
        constexpr uint n_iters = (n_blocks + 31u) / 32u;
        constexpr uint routed_experts = 6u;

        uint expert_slot = tgid % routed_experts;
        uint tile = tgid / routed_experts;
        uint expert = indices[expert_slot];
        uint first_row = tile * 4u + sgid * 2u;

        const device uchar* u_wbase =
            (const device uchar*)up_weight + expert * packed_expert_bytes;
        const device uchar* u_sbase =
            up_scales + expert * scale_expert_bytes;

        thread float up_result[2] = {0.f, 0.f};
        thread float input_values[16];

        for (uint iter = 0u; iter < n_iters; ++iter) {
            uint groupIdx = iter * 32u + lane;
            bool valid = groupIdx < n_blocks;
            if (valid) {
                const device vec<bfloat, 4>* input_vectors =
                    (const device vec<bfloat, 4>*)(input + groupIdx * 16u);
                for (uint i = 0; i < 4u; ++i) {
                    const vec<bfloat, 4> values = input_vectors[i];
                    input_values[4u * i + 0u] = float(values[0]);
                    input_values[4u * i + 1u] = float(values[1]);
                    input_values[4u * i + 2u] = float(values[2]);
                    input_values[4u * i + 3u] = float(values[3]);
                }
            }
            for (uint row = 0u; row < 2u; ++row) {
                if (!valid) continue;
                uint out_row = first_row + row;
                const device uchar* up_w =
                    u_wbase + out_row * packed_row_bytes + groupIdx * 8u;
                const device uchar* up_s =
                    u_sbase + out_row * scale_row_bytes + groupIdx;
                float us = iron_nvfp4_e4m3_scale(up_s[0]) * global_scale;
                up_result[row] += iron_nvfp4_qdot_16(up_w, input_values, us);
            }
        }

        device bfloat* activated;
        switch (expert_slot) {
            case 0u: activated = activated_0; break;
            case 1u: activated = activated_1; break;
            case 2u: activated = activated_2; break;
            case 3u: activated = activated_3; break;
            case 4u: activated = activated_4; break;
            default: activated = activated_5; break;
        }

        for (uint row = 0u; row < 2u; ++row) {
            up_result[row] = simd_sum(up_result[row]);
            if (lane == 0u) {
                bfloat up_bf = bfloat(up_result[row]);
                float r = (up_bf > bfloat(0.0f)) ? float(up_bf) : 0.0f;
                activated[first_row + row] = bfloat(r * r);
            }
        }
    }
    """

    // MARK: - Fused routed down-projection + weighted reduce

    private static let routedDownReduceSrc = header + """
    kernel void nemotron_moe_routed_down_reduce_nvfp4_bf16(
        device const bfloat *mid_0 [[buffer(0)]],
        device const bfloat *mid_1 [[buffer(1)]],
        device const bfloat *mid_2 [[buffer(2)]],
        device const bfloat *mid_3 [[buffer(3)]],
        device const bfloat *mid_4 [[buffer(4)]],
        device const bfloat *mid_5 [[buffer(5)]],
        device const uint *indices [[buffer(6)]],
        device const bfloat *slot_weights [[buffer(7)]],
        device const uint *down_weight [[buffer(8)]],
        device const uchar *down_scales [[buffer(9)]],
        device bfloat *output [[buffer(10)]],
        constant float &global_scale [[buffer(11)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]
    ) {
        constexpr uint input_width = 1856u;
        constexpr uint experts_per_token = 6u;
        constexpr uint outputs_per_simd = 4u;
        constexpr uint n_blocks = input_width / 16u;
        constexpr uint n_iters = (n_blocks + 31u) / 32u;
        constexpr uint out_dim = 2688u;
        constexpr uint n_packs_row = input_width / 8u;

        uint expert_slot = simd_group;
        uint lane = simd_lane;
        uint first_row = tgid * outputs_per_simd;
        if (expert_slot >= experts_per_token || first_row >= out_dim) return;

        const device bfloat* expert_input;
        switch (expert_slot) {
            case 0u: expert_input = mid_0; break;
            case 1u: expert_input = mid_1; break;
            case 2u: expert_input = mid_2; break;
            case 3u: expert_input = mid_3; break;
            case 4u: expert_input = mid_4; break;
            default: expert_input = mid_5; break;
        }

        uint expert = indices[expert_slot];
        uint exp_base_packs = expert * out_dim * n_packs_row;
        uint exp_base_blocks = expert * out_dim * n_blocks;

        thread float result[outputs_per_simd] = {0.f, 0.f, 0.f, 0.f};

        for (uint iter = 0u; iter < n_iters; ++iter) {
            uint groupIdx = iter * 32u + lane;
            bool valid = groupIdx < n_blocks;
            thread float input_values[16];
            if (valid) {
                const device vec<bfloat, 4>* input_vectors =
                    (const device vec<bfloat, 4>*)(
                        expert_input + groupIdx * 16u);
                for (uint i = 0; i < 4u; ++i) {
                    const vec<bfloat, 4> values = input_vectors[i];
                    input_values[4u * i + 0u] = float(values[0]);
                    input_values[4u * i + 1u] = float(values[1]);
                    input_values[4u * i + 2u] = float(values[2]);
                    input_values[4u * i + 3u] = float(values[3]);
                }
            }
            for (uint row = 0u; row < outputs_per_simd; ++row) {
                if (!valid) continue;
                uint output_row = first_row + row;
                uint row_pack_off = exp_base_packs + output_row * n_packs_row + groupIdx * 2u;
                uint row_block_off = exp_base_blocks + output_row * n_blocks + groupIdx;
                float scale =
                    iron_nvfp4_e4m3_scale(down_scales[row_block_off]) * global_scale;
                const device uchar* wbytes =
                    (const device uchar*)(down_weight + row_pack_off);
                result[row] += iron_nvfp4_qdot_16(wbytes, input_values, scale);
            }
        }
        for (uint row = 0u; row < outputs_per_simd; ++row) {
            result[row] = simd_sum(result[row]);
        }

        threadgroup bfloat expert_outputs[experts_per_token * outputs_per_simd];
        if (lane == 0u) {
            for (uint row = 0u; row < outputs_per_simd; ++row) {
                expert_outputs[expert_slot * outputs_per_simd + row] =
                    bfloat(result[row]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (expert_slot == 0u && lane < outputs_per_simd) {
            bfloat total = bfloat(0.0f);
            for (uint slot = 0u; slot < experts_per_token; ++slot) {
                bfloat route_weight = slot_weights[slot];
                bfloat product = bfloat(
                    float(expert_outputs[slot * outputs_per_simd + lane])
                    * float(route_weight));
                total = bfloat(float(product) + float(total));
            }
            output[first_row + lane] = total;
        }
    }
    """

    /// Prefer precompiled metallib PSO (kernels_extra.metallib, see the
    /// file-header doc). Falls back to live `makeLibrary(source:)` for
    /// builds that haven't re-run `make link-extra-metals` yet.
    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        if let p = try? PSOCache.shared.pipelineStateThrowing(for: name) {
            psoCache[name] = p
            return p
        }
        let opts = MTLCompileOptions()
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    /// Fused routed up-projection + squared-ReLU for NemotronH-Lightning
    /// (hidden=2688, moeIntermediate=1856, topK=6). One dispatch replaces
    /// 6x expert-indexed up GEMV (`Ops.nvfp4DequantGemvExpertIndexedMany`)
    /// + 6x `Ops.relu` + 6x `Ops.mul` (squaring).
    ///
    /// Epilogue rounding contract (must match the unfused chain bit for
    /// bit — see NemotronHNvfp4MoELayer.decodeGPURouted's per-slot loop):
    ///   1. up-GEMV float accumulator rounds to bf16 (mimics the unfused
    ///      GEMV kernel writing `upOuts[slot]` as a real bf16 tensor).
    ///   2. relu on that bf16 value in float, which is lossless (0 and
    ///      any positive bf16 value round-trip through float exactly) —
    ///      equivalent to `Ops.relu`'s own extra bf16 round being a
    ///      no-op given an already-bf16 input.
    ///   3. square in float, round to bf16 (mimics `Ops.mul(r, r)`).
    static func moeRoutedRelu2Qmv(
        input: Tensor,
        upWeight: Tensor,
        upScales: Tensor,
        expertIndices: Tensor,
        outputs: [Tensor],
        global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        device: Device
    ) {
        precondition(outputs.count == 6, "moeRoutedRelu2Qmv: expected 6 output slots (topK)")
        precondition(input.dtype == .bf16 && input.elementCount == 2688)
        precondition(expertIndices.dtype == .u32 && expertIndices.elementCount == 6)
        precondition(upWeight.dtype == .u32 && upScales.dtype == .u8)
        for i in 0 ..< 6 {
            precondition(outputs[i].dtype == .bf16 && outputs[i].elementCount == 1856)
        }

        let mtl = device.mtlDevice
        let p = pso(
            "nemotron_moe_routed_relu2_qmv_nvfp4_bf16",
            source: routedRelu2Src, device: mtl)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(upWeight.buffer, offset: upWeight.offset, index: 1)
        enc.setBuffer(upScales.buffer, offset: upScales.offset, index: 2)
        enc.setBuffer(expertIndices.buffer, offset: expertIndices.offset, index: 3)
        for i in 0 ..< 6 {
            enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 4 + i)
        }
        var g = global
        enc.setBytes(&g, length: 4, index: 10)
        // 6 experts x 464 tiles (moeIntermediate/4); TPG=64 (2 SGs x 32).
        enc.dispatchThreadgroups(
            MTLSize(width: 6 * 464, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Fused routed down-projection + weighted sum for NemotronH-Lightning
    /// (moeIntermediate=1856, hidden=2688, topK=6). One dispatch replaces
    /// 6x expert-indexed down GEMV + 6x `Ops.headGateMul` + 5x `Ops.add`.
    ///
    /// `slotWeights` must be the SAME buffer the unfused chain's
    /// `headGateMul` reads per slot (already includes `routedScalingFactor`
    /// and sum-to-1 renormalization) — the reduce chain here reproduces
    /// `Ops.headGateMul` + serial `Ops.add` bit for bit given identical
    /// per-slot down-GEMV outputs (see the kernel's in-file doc comment).
    static func moeRoutedDownReduce(
        mids: [Tensor],
        expertIndices: Tensor,
        slotWeights: Tensor,
        downWeight: Tensor,
        downScales: Tensor,
        output: Tensor,
        global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        device: Device
    ) {
        precondition(mids.count == 6, "moeRoutedDownReduce: expected 6 mid slots (topK)")
        precondition(expertIndices.dtype == .u32 && expertIndices.elementCount == 6)
        precondition(slotWeights.dtype == .bf16 && slotWeights.elementCount == 6)
        precondition(output.dtype == .bf16 && output.elementCount == 2688)
        precondition(downWeight.dtype == .u32 && downScales.dtype == .u8)
        for i in 0 ..< 6 {
            precondition(mids[i].dtype == .bf16 && mids[i].elementCount == 1856)
        }

        let mtl = device.mtlDevice
        let p = pso(
            "nemotron_moe_routed_down_reduce_nvfp4_bf16",
            source: routedDownReduceSrc, device: mtl)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        for i in 0 ..< 6 {
            enc.setBuffer(mids[i].buffer, offset: mids[i].offset, index: i)
        }
        enc.setBuffer(expertIndices.buffer, offset: expertIndices.offset, index: 6)
        enc.setBuffer(slotWeights.buffer, offset: slotWeights.offset, index: 7)
        enc.setBuffer(downWeight.buffer, offset: downWeight.offset, index: 8)
        enc.setBuffer(downScales.buffer, offset: downScales.offset, index: 9)
        enc.setBuffer(output.buffer, offset: output.offset, index: 10)
        var g = global
        enc.setBytes(&g, length: 4, index: 11)
        // 672 tiles (hidden/4) x 4 rows; TPG=192 (6 SGs x 32 lanes).
        enc.dispatchThreadgroups(
            MTLSize(width: 672, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1))
        enc.endEncoding()
    }
}
