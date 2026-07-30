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
// Decode-only gated o_proj fusion (T24 "TASK 1"): hand-written
// live-compiled Metal via the `LagunaNvfp4Fuse`/`LagunaAttnInt8` `pso()`
// pattern. Butter's own DSL (`iron-std`'s `#[kernel]` macro) has no
// primitive for this shape, so this stays runtime Metal rather than a
// DSL port.
//
// Two independent fusions, routed to whichever `o_proj` path a layer's
// `ensureAttnInt8()` build landed on (see `LagunaText.swift` decode
// `forward`):
//
// 1. `laguna_gate_product_softplus_bf16_h{48,64}`: one dispatch computes
//    the numerically-stable softplus of the RAW per-head gate logit (same
//    `max(x,0) + log(1+exp(-|x|))` form as `iron_softplus_bf16`, i.e. the
//    op stream `Ops.softplus` already runs), rounds it to BF16 exactly
//    where `Ops.softplus`'s output cast rounds it, then broadcast-
//    multiplies against the raw SDPA row and rounds to BF16 exactly where
//    `Ops.mul` rounds it. Bit-identical to the stock
//    `softplus -> gather-expand -> mul` chain (3 dispatches), in 1. Feeds
//    the T20 INT8 o_proj QMV bank unchanged when that bank is active:
//    the bank's dequant already expects a plain BF16 gated row.
//
// 2. `laguna_gated_output_projection_bf16_h{H}_u2`: for the plain BF16
//    `o_proj` `Linear` path only (INT8 bank absent or declined), folds
//    the per-head gate multiply directly into the o_proj GEMV's column
//    loads (unroll depth 2), so the `[nHeads*headDim]` gated row never
//    materializes at all. Takes already-*activated* (post-softplus)
//    per-head gate values; the caller runs a `[nHeads]`-wide
//    `Ops.softplus` first (cheap: O(64) vs the O(8192) gather+mul it
//    replaces) and passes that in, satisfying this kernel's activated-gate
//    precondition.
//
// Escape: `BUTTER_LAGUNA_FUSED_GATED_OPROJ=0` restores the stock
// softplus+gather-expand+mul (+ separate GEMV) chain byte-for-byte.
// Default ON.

import Foundation
import Metal
import IronSwift

enum LagunaGatedOProjFuse {
    /// Default ON; set `BUTTER_LAGUNA_FUSED_GATED_OPROJ=0` to restore the
    /// stock per-dispatch softplus/gather/mul (+ separate o_proj) chain.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_FUSED_GATED_OPROJ"] != "0"
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

    // ─── Kernel 1: raw-logit softplus + broadcast gate product ─────────

    private static func gateProductSoftplusSource(
        heads: Int, headDim: Int, name: String
    ) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void \(name)(
            const device bfloat *attention_output [[buffer(0)]],
            const device bfloat *gate_logits [[buffer(1)]],
            device bfloat *gated [[buffer(2)]],
            uint gid [[thread_position_in_grid]])
        {
            constexpr uint HEAD_DIM = \(headDim);
            constexpr uint TOTAL = \(heads * headDim);
            if (gid >= TOTAL) return;
            uint head = gid / HEAD_DIM;
            // Numerically-stable softplus, same op stream/order as
            // `iron_softplus_bf16.metal`: max(x,0) + log(1 + exp(-|x|)).
            float x = float(gate_logits[head]);
            bool pos = x > 0.0f;
            float negx = 0.0f - x;
            float ax = pos ? x : negx;
            float e = exp(0.0f - ax);
            float lg = log(1.0f + e);
            bfloat gate_bf = bfloat((pos ? x : 0.0f) + lg);
            gated[gid] = bfloat(float(attention_output[gid]) * float(gate_bf));
        }
        """
    }

    /// Fused softplus(gate_logits) * attention_output in one dispatch.
    /// Declines (returns `false`, `into` left untouched) unless
    /// `BUTTER_LAGUNA_FUSED_GATED_OPROJ != "0"` and every shape/dtype
    /// precondition holds; caller falls back to the stock chain.
    static func applyGateProductSoftplus(
        attnOut: Tensor, gateLogits: Tensor, heads: Int, headDim: Int,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard enabled,
            attnOut.dtype == .bf16, gateLogits.dtype == .bf16, out.dtype == .bf16,
            attnOut.elementCount == heads * headDim,
            gateLogits.elementCount == heads,
            out.elementCount == heads * headDim
        else { return false }

        let name = "laguna_gate_product_softplus_bf16_h\(heads)"
        let src = gateProductSoftplusSource(heads: heads, headDim: headDim, name: name)
        let p = pso(name, source: src, device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(attnOut.buffer, offset: attnOut.offset, index: 0)
        enc.setBuffer(gateLogits.buffer, offset: gateLogits.offset, index: 1)
        enc.setBuffer(out.buffer, offset: out.offset, index: 2)
        let total = heads * headDim
        let tgWidth = 256
        let groups = (total + tgWidth - 1) / tgWidth
        enc.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

    // ─── Kernel 2: gate-folded o_proj GEMV (plain BF16 weight, unroll 2) ─

    /// Unroll-2 gate-folded o_proj GEMV. Non-palette shape only: butter's
    /// stock `o_proj` `Linear` weight is always plain dense BF16, never
    /// palette-quantized. One threadgroup (128 threads = 4 simdgroups x
    /// 32 lanes) emits `rows_per_group = 16` output rows; grid tiles =
    /// `hidden / 16`. `blocks = in_vec_size / 128 == heads`, and each
    /// 128-wide column block is exactly one attention head, so
    /// `gate_values[block]` is that head's (already-softplus'd) scalar.
    ///
    /// ## DISPATCH INVARIANTS
    /// - `heads` even (unroll depth 2 must divide `blocks == heads`).
    /// - `headDim == 128` (kernel hardcodes `block_width = head_dim`).
    /// - `hidden % 16 == 0` (grid tiling; Laguna hidden=2048).
    private static func gatedOutputProjectionSource(
        heads: Int, headDim: Int, name: String
    ) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void \(name)(
            const device bfloat *attention_output [[buffer(0)]],
            const device bfloat *gate_values [[buffer(1)]],
            const device bfloat *weight [[buffer(2)]],
            device bfloat *projected [[buffer(3)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint sg [[simdgroup_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]])
        {
            constexpr uint unroll = 2;
            constexpr uint in_vec_size = \(heads * headDim);
            constexpr uint rows_per_thread = 4;
            constexpr uint values_per_thread = 4;
            constexpr uint block_width = 128;
            constexpr uint blocks = in_vec_size / block_width;
            constexpr uint rows_per_group = 16;

            uint tile = tgid.x;
            uint out_row = tile * rows_per_group + sg * rows_per_thread;
            thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
            thread float coefficients[values_per_thread];

            uint column = lane * values_per_thread;
            for (uint block = 0; block < blocks; block += unroll) {
                vec<bfloat, 4> raw_values[unroll];
                vec<bfloat, 4> weight_values[unroll][rows_per_thread];
                for (uint u = 0; u < unroll; ++u) {
                    uint column_u = column + u * block_width;
                    const device vec<bfloat, 4>* av =
                        (const device vec<bfloat, 4>*)(attention_output + column_u);
                    raw_values[u] = av[0];
                    for (uint row = 0; row < rows_per_thread; ++row) {
                        const device vec<bfloat, 4>* row_values =
                            (const device vec<bfloat, 4>*)(
                                weight + (out_row + row) * in_vec_size + column_u);
                        weight_values[u][row] = row_values[0];
                    }
                }

                for (uint u = 0; u < unroll; ++u) {
                    // Column `4 * lane + 128 * (block + u)` is in head `block + u`.
                    float gate = float(gate_values[block + u]);
                    for (uint i = 0; i < values_per_thread; ++i) {
                        coefficients[i] = float(bfloat(float(raw_values[u][i]) * gate));
                    }
                    for (uint row = 0; row < rows_per_thread; ++row) {
                        for (uint i = 0; i < values_per_thread; ++i) {
                            result[row] += float(weight_values[u][row][i]) * coefficients[i];
                        }
                    }
                }

                column += unroll * block_width;
            }

            for (uint row = 0; row < rows_per_thread; ++row) {
                for (ushort delta = 16; delta >= 1; delta >>= 1) {
                    result[row] += metal::simd_shuffle_down(result[row], delta);
                }
            }
            if (lane == 0) {
                for (uint row = 0; row < rows_per_thread; ++row) {
                    projected[out_row + row] = bfloat(result[row]);
                }
            }
        }
        """
    }

    /// Fused gate-multiply + o_proj GEMV in one dispatch. `gateValues`
    /// must already be softplus-activated (caller runs a cheap
    /// `[heads]`-wide `Ops.softplus` first, see file header). Declines
    /// (returns `false`) unless enabled and every shape/dtype/divisibility
    /// precondition holds; caller falls back to the stock gate-expand +
    /// separate `o_proj` GEMV chain.
    static func applyGatedOutputProjection(
        attentionOutput: Tensor, gateValues: Tensor, weight: Tensor,
        heads: Int, headDim: Int, hidden: Int,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard enabled,
            headDim == 128, heads.isMultiple(of: 2), hidden.isMultiple(of: 16),
            attentionOutput.dtype == .bf16, gateValues.dtype == .bf16,
            weight.dtype == .bf16, out.dtype == .bf16,
            attentionOutput.elementCount == heads * headDim,
            gateValues.elementCount == heads,
            weight.shape == [hidden, heads * headDim],
            out.elementCount == hidden
        else { return false }

        let name = "laguna_gated_output_projection_bf16_h\(heads)_u2"
        let src = gatedOutputProjectionSource(heads: heads, headDim: headDim, name: name)
        let p = pso(name, source: src, device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(attentionOutput.buffer, offset: attentionOutput.offset, index: 0)
        enc.setBuffer(gateValues.buffer, offset: gateValues.offset, index: 1)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        enc.setBuffer(out.buffer, offset: out.offset, index: 3)
        let tiles = hidden / 16
        enc.dispatchThreadgroups(
            MTLSize(width: tiles, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }
}
