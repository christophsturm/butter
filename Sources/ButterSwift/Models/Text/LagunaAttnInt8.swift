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
// Attention-side group-32 affine INT8 requant: decode-only side layout
// built at load from the BF16 Q/K/V (+ per-head gate) and o_proj
// projections.
//
// At load time each attention block's Q/K/V (+ per-head gate) BF16
// weights are re-quantized to a group-32 affine INT8 side layout and
// row-concatenated into ONE bank; the serial decode token then reads
// that bank with a single batched QMV dispatch instead of three (or
// four) separate BF16 GEMVs. `o_proj` gets its own standalone group-32
// INT8 bank (different input width, it consumes the SDPA output, not
// the normalized residual). The original BF16 parameters stay resident
// and authoritative; prefill (`forwardMulti`) never touches this path.
//
// `iron_qmv_int8_fast_*` (Ops.swift) is a close cousin but hardcodes
// group_size=64 in its generated inner loop (see
// `QuantizedOpsValidation.validateAffineQuantize`, which hardcodes the
// *quantize* kernels to group_size=64 for the same reason: the iron
// codegen reduction shape is fixed at build time). Neither is reusable
// for group-32, so both the quantize step and the batched QMV here are
// hand-written live-compiled Metal, following the same `pso()` +
// `makeLibrary(source:)` pattern as `LagunaNvfp4Fuse`.
//
// Escape: `BUTTER_LAGUNA_ATTN_INT8=0` restores the stock BF16 path.

import Foundation
import Metal
import IronSwift

enum LagunaAttnInt8 {
    /// Default ON; set `BUTTER_LAGUNA_ATTN_INT8=0` to fall back to the
    /// stock BF16 QKV / o_proj decode path.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_ATTN_INT8"] != "0"
    }

    static let groupSize = 32

    /// One group-32 affine INT8 side layout: `packed` is `[outDim,
    /// inDim/4]` u32 (4 int8 codes per word), `scales`/`biases` are
    /// `[outDim, inDim/groupSize]` bf16. Dequant: `code * scale + bias`.
    struct Bank {
        let packed: Tensor
        let scales: Tensor
        let biases: Tensor
        let outDim: Int
        let inDim: Int
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

    // Group-32 affine INT8 quantize: one threadgroup (32 lanes = 1
    // simdgroup) per group. Mirrors the algorithm of
    // `iron_affine_quantize_int8_bf16` (min/max simd-reduce -> scale +
    // zero-point bias -> round-clamp-pack 4 codes/word) but with 1
    // element/lane over 32 lanes instead of 2 elements/lane over 32
    // lanes (that kernel's fixed 64-wide group).
    private static let quantizeSrc = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_attn_quantize_int8_g32_bf16(
            const device bfloat *w [[buffer(0)]],
            device uint *out [[buffer(1)]],
            device bfloat *scales [[buffer(2)]],
            device bfloat *biases [[buffer(3)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_threadgroup]])
        {
            constexpr uint GROUP = 32;
            uint g = tgid.x;
            uint base = g * GROUP;

            float v = float(w[base + lane]);
            float lo = simd_min(v);
            float hi = simd_max(v);
            float rawScale = (hi - lo) / 255.0f;
            float scale = rawScale < 1e-7f ? 1.0f : rawScale;
            float invScale = 1.0f / scale;

            if (lane == 0) {
                scales[g] = bfloat(scale);
                biases[g] = bfloat(lo);
            }

            // 8 words/group (32 elements / 4 per word); lanes 0..7 pack.
            if (lane < 8) {
                uint packed = 0;
                for (uint i = 0; i < 4; ++i) {
                    uint idx = lane * 4 + i;
                    float x = float(w[base + idx]);
                    float q = clamp((x - lo) * invScale + 0.5f, 0.0f, 255.0f);
                    packed |= (uint(q) << (8u * i));
                }
                out[g * 8 + lane] = packed;
            }
        }
        """

    // Batched group-32 affine INT8 GEMV over a (possibly row-concatenated)
    // bank. TPG = 64 (2 simdgroups x 32 lanes), 4 output rows per
    // simdgroup => 8 rows/threadgroup; grid = ceil(outDim/8). Row bounds
    // are checked so `outDim` need not be a multiple of 8 (e.g. the
    // per-head gate tail, or an odd head count).
    private static let qmvSrc = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_attn_qmv_int8_g32_bf16(
            const device uint *w [[buffer(0)]],
            const device bfloat *scales [[buffer(1)]],
            const device bfloat *biases [[buffer(2)]],
            const device bfloat *x [[buffer(3)]],
            device bfloat *out [[buffer(4)]],
            constant uint &inDim [[buffer(5)]],
            constant uint &outDim [[buffer(6)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 32;
            constexpr uint ROWS_PER_SG = 4;
            uint packsPerRow = inDim / 4;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * (2u * ROWS_PER_SG) + sg * ROWS_PER_SG;

            float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (uint block = 0; block < packsPerRow; block += 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 4u;
                    bfloat4 xv = *((const device bfloat4 *)(x + xBase));
                    float x0 = float(xv[0]);
                    float x1 = float(xv[1]);
                    float x2 = float(xv[2]);
                    float x3 = float(xv[3]);
                    float xs = x0 + x1 + x2 + x3;
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < ROWS_PER_SG; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float s = float(scales[row * groupsPerRow + g]);
                            float b = float(biases[row * groupsPerRow + g]);
                            float c0 = float(p & 0xFFu);
                            float c1 = float((p >> 8u) & 0xFFu);
                            float c2 = float((p >> 16u) & 0xFFu);
                            float c3 = float((p >> 24u) & 0xFFu);
                            float qd = c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3;
                            acc[r] += s * qd + b * xs;
                        }
                    }
                }
            }

            for (uint r = 0; r < ROWS_PER_SG; ++r) {
                float total = simd_sum(acc[r]);
                if (lane == 0) {
                    uint row = rowBase + r;
                    if (row < outDim) {
                        out[row] = bfloat(total);
                    }
                }
            }
        }
        """

    /// Quantize a `[outDim, inDim]` BF16 weight to a group-32 affine
    /// INT8 side layout. Declines (returns `nil`) unless `inDim` is a
    /// multiple of 32 and `weight` is 2D BF16, matching the group-32
    /// alignment the packed layout requires.
    static func quantizeG32(
        weight: Tensor, device: Device, cmd: MTLCommandBuffer
    ) -> Bank? {
        guard weight.dtype == .bf16, weight.shape.count == 2,
            weight.shape[1].isMultiple(of: groupSize)
        else { return nil }
        let outDim = weight.shape[0]
        let inDim = weight.shape[1]
        let packsPerRow = inDim / 4
        let groupsPerRow = inDim / groupSize
        let packed = Tensor.empty(shape: [outDim, packsPerRow], dtype: .u32, device: device)
        let scales = Tensor.empty(shape: [outDim, groupsPerRow], dtype: .bf16, device: device)
        let biases = Tensor.empty(shape: [outDim, groupsPerRow], dtype: .bf16, device: device)
        let nGroups = outDim * groupsPerRow
        let p = pso(
            "laguna_attn_quantize_int8_g32_bf16", source: quantizeSrc,
            device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(weight.buffer, offset: weight.offset, index: 0)
            enc.setBuffer(packed.buffer, offset: packed.offset, index: 1)
            enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
            enc.setBuffer(biases.buffer, offset: biases.offset, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: nGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
        }
        return Bank(packed: packed, scales: scales, biases: biases, outDim: outDim, inDim: inDim)
    }

    /// Row-concatenate banks that share `inDim` (e.g. Q/K/V + per-head
    /// gate) into one bank a single QMV can read. Pure CPU memcpy; the
    /// GPU quantize dispatches that produced `banks` must already have
    /// completed (caller commits + waits the setup command buffer
    /// first).
    static func concat(_ banks: [Bank], device: Device) -> Bank? {
        guard let first = banks.first,
            banks.allSatisfy({ $0.inDim == first.inDim })
        else { return nil }
        let totalOut = banks.reduce(0) { $0 + $1.outDim }
        let packsPerRow = first.inDim / 4
        let groupsPerRow = first.inDim / groupSize
        let packedBuf = device.makeBuffer(length: totalOut * packsPerRow * 4)
        let scalesBuf = device.makeBuffer(length: totalOut * groupsPerRow * 2)
        let biasesBuf = device.makeBuffer(length: totalOut * groupsPerRow * 2)
        var packedOff = 0
        var scalesOff = 0
        var biasesOff = 0
        for b in banks {
            let packedBytes = b.outDim * packsPerRow * 4
            let scaleBytes = b.outDim * groupsPerRow * 2
            memcpy(
                packedBuf.contents().advanced(by: packedOff),
                b.packed.buffer.contents().advanced(by: b.packed.offset), packedBytes)
            memcpy(
                scalesBuf.contents().advanced(by: scalesOff),
                b.scales.buffer.contents().advanced(by: b.scales.offset), scaleBytes)
            memcpy(
                biasesBuf.contents().advanced(by: biasesOff),
                b.biases.buffer.contents().advanced(by: b.biases.offset), scaleBytes)
            packedOff += packedBytes
            scalesOff += scaleBytes
            biasesOff += scaleBytes
        }
        // Row-concatenated bank replaces the stock BF16 Q/K/V(+gate) or
        // o_proj weights for every decode token from here on; pin it
        // in the residency set like any other long-lived weight buffer.
        device.markWeightsResident([packedBuf, scalesBuf, biasesBuf])
        return Bank(
            packed: Tensor(
                buffer: packedBuf, offset: 0, shape: [totalOut, packsPerRow], dtype: .u32),
            scales: Tensor(
                buffer: scalesBuf, offset: 0, shape: [totalOut, groupsPerRow], dtype: .bf16),
            biases: Tensor(
                buffer: biasesBuf, offset: 0, shape: [totalOut, groupsPerRow], dtype: .bf16),
            outDim: totalOut, inDim: first.inDim)
    }

    /// Single-token (`x = [inDim]` bf16) batched QMV over `bank`,
    /// writing `[bank.outDim]` bf16 into `out`.
    static func qmv(
        input: Tensor, bank: Bank, on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) {
        precondition(input.dtype == .bf16, "LagunaAttnInt8.qmv: input must be bf16")
        precondition(
            input.elementCount == bank.inDim,
            "LagunaAttnInt8.qmv: input.elementCount \(input.elementCount) != bank.inDim \(bank.inDim)"
        )
        precondition(
            out.elementCount == bank.outDim,
            "LagunaAttnInt8.qmv: out.elementCount \(out.elementCount) != bank.outDim \(bank.outDim)"
        )
        let p = pso(
            "laguna_attn_qmv_int8_g32_bf16", source: qmvSrc, device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
            enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
            enc.setBuffer(bank.biases.buffer, offset: bank.biases.offset, index: 2)
            enc.setBuffer(input.buffer, offset: input.offset, index: 3)
            enc.setBuffer(out.buffer, offset: out.offset, index: 4)
            var inDimV = UInt32(bank.inDim)
            var outDimV = UInt32(bank.outDim)
            enc.setBytes(&inDimV, length: 4, index: 5)
            enc.setBytes(&outDimV, length: 4, index: 6)
            let rowsPerTG = 8
            let tiles = (bank.outDim + rowsPerTG - 1) / rowsPerTG
            enc.dispatchThreadgroups(
                MTLSize(width: tiles, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
        }
    }
}
