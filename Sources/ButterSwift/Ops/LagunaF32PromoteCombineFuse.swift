// Copyright 2026 Tom Turney (@TheTom)
//
// W52 byte-hunt: fuse the f32Promote MoE fallback's down-output
// materialisation into the combine step.
//
// CONTEXT: `BUTTER_NVFP4_MPP_BF16_MAX_LAYER` (default 36) routes late
// Laguna layers (37-39 of 39 sparse layers — the "half coop overflow"
// layers, see MoELayer.swift's `useF32Promote`) through an f32-precision
// MoE MPP fallback instead of the W51-levered bf16/f16 expert-aligned
// gather chain (`LagunaNvfp4ExpertGather`). That fallback's down
// projection lands in an f32 scratch (`downF`), which the stock chain
// then casts to bf16 (`Ops.castF32ToBf16`) into `downOut` — purely so
// the shared combine step (`Ops.moeUnpermute`) can read a uniform-dtype
// `expertOutputs` buffer. That intermediate `downOut` bf16 buffer is
// written exactly once and read exactly once (by the very next
// dispatch, the combine/unpermute), with no other consumer — the same
// "write-then-read-once round trip" class W51 eliminated for the sort/
// gather/scatter machinery, just on the dtype-cast side instead.
//
// FIX: skip the cast entirely — read `downF` (f32) straight into the
// combine kernel, which does the bf16 rounding itself as part of the
// weighted-sum load (see `bf16(float(x))` below). Deletes one dispatch
// per f32Promote layer and the entire `downOut` bf16 write+read for
// that layer's down output (nets ~half the down-output-handling bytes
// for those layers — the f32 read replaces a 4B read; the round-trip
// through a 2B bf16 buffer is what disappears).
//
// BYTE-IDENTICAL: `iron_moe_unpermute_bf16` (the generated kernel the
// stock chain dispatches into) reads `expert_outputs` as bf16, casts to
// float, multiplies by the bf16 combine weight (also cast to float),
// and accumulates in float before the final bf16 store (see the
// extracted kernel source referenced in the W52 byte-census notes).
// Feeding it `downOut` post-cast means the value it reads is already
// `bfloat(float(downF_raw))` — a DOUBLE round (f32 -> bf16 -> float
// again inside the kernel). This fused kernel reproduces that exact
// double round in one pass: `float(bfloat(downF_raw))` before the same
// float accumulate + bf16 store math, so the output is bit-identical to
// the stock two-dispatch chain.
//
// SCOPE: only reachable from MoELayer's `useF32Promote` branch (3 of 39
// sparse layers by default). Small in isolation — the win is a fraction
// of a layer's bytes, times 3 layers — but it is the same pattern class
// as W51 and the campaign asked for an audit of exactly this shape.
// Default ON; escape `BUTTER_LAGUNA_F32_COMBINE_FUSE=0` restores the
// stock `castF32ToBf16` + `Ops.moeUnpermute` two-dispatch chain.

import Foundation
import Metal

enum LagunaF32PromoteCombineFuse {
    static var enabled: Bool {
        ButterEnv["BUTTER_LAGUNA_F32_COMBINE_FUSE"] != "0"
    }

    nonisolated(unsafe) private static var pso: MTLComputePipelineState?
    private static let lock = NSLock()

    // Direct Metal port of `iron_moe_unpermute_bf16` (see file header)
    // with `expert_outputs` widened to f32 and the double-round the
    // stock cast-then-combine chain would have paid folded into the
    // per-element load.
    private static let src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void laguna_moe_unpermute_f32in_bf16(
        const device float *expert_outputs [[buffer(0)]],
        const device uint *inv_perm [[buffer(1)]],
        const device bfloat *top_k_weights [[buffer(2)]],
        device bfloat *out [[buffer(3)]],
        constant uint &hidden [[buffer(4)]],
        constant uint &k [[buffer(5)]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 tgid3 [[threadgroup_position_in_grid]]
    ) {
        uint token = tgid3.x;
        uint lane = tid3.x;
        uint rowBaseInv = token * k;
        uint rowBaseOut = token * hidden;
        uint nPerLane = (hidden + 127u) / 128u;
        for (uint i = 0u; i < nPerLane; ++i) {
            uint h = i * 128u + lane;
            if (h >= hidden) continue;
            float acc = 0.0f;
            for (uint s = 0u; s < k; ++s) {
                uint slot = rowBaseInv + s;
                uint pos = inv_perm[slot];
                // Double-round: the stock chain's castF32ToBf16 rounds
                // the raw f32 down-GEMM output to bf16 before combine
                // ever sees it — reproduce that exact rounding here so
                // the accumulate operates on the same bf16-quantized
                // value, not the higher-precision raw f32.
                float v = float(bfloat(expert_outputs[pos * hidden + h]));
                float w = float(top_k_weights[slot]);
                acc += w * v;
            }
            out[rowBaseOut + h] = bfloat(acc);
        }
    }
    """

    private static func pipelineState(device: MTLDevice) -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let pso { return pso }
        let lib = try! device.makeLibrary(source: src, options: MTLCompileOptions())
        let fn = lib.makeFunction(name: "laguna_moe_unpermute_f32in_bf16")!
        let p = try! device.makeComputePipelineState(function: fn)
        pso = p
        return p
    }

    /// Combine an f32 expert-output table directly (no intermediate bf16
    /// materialise). `expertOutputsF32` = `[nRows*k, hidden]` f32 (plan
    /// row order); `invPerm`/`topKWeights` are the same tensors the stock
    /// `Ops.moeUnpermute` call takes. `out` is bf16, `[nRows, hidden]`.
    static func combine(
        expertOutputsF32: Tensor, invPerm: Tensor, topKWeights: Tensor,
        into out: Tensor, nRows: Int, hidden: Int, k: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(expertOutputsF32.dtype == .f32)
        precondition(invPerm.dtype == .u32)
        precondition(topKWeights.dtype == .bf16 && out.dtype == .bf16)
        precondition(expertOutputsF32.elementCount == nRows * k * hidden)
        precondition(invPerm.elementCount == nRows * k)
        precondition(topKWeights.elementCount == nRows * k)
        precondition(out.elementCount == nRows * hidden)
        let p = pipelineState(device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            fatalError("LagunaF32PromoteCombineFuse.combine: could not make compute encoder")
        }
        enc.setComputePipelineState(p)
        enc.setBuffer(expertOutputsF32.buffer, offset: expertOutputsF32.offset, index: 0)
        enc.setBuffer(invPerm.buffer, offset: invPerm.offset, index: 1)
        enc.setBuffer(topKWeights.buffer, offset: topKWeights.offset, index: 2)
        enc.setBuffer(out.buffer, offset: out.offset, index: 3)
        var hiddenU = UInt32(hidden)
        var kU = UInt32(k)
        enc.setBytes(&hiddenU, length: 4, index: 4)
        enc.setBytes(&kU, length: 4, index: 5)
        let tgWidth = 128
        enc.dispatchThreadgroups(
            MTLSize(width: nRows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        enc.endEncoding()
    }
}
