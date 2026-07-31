// Copyright 2026 Tom Turney (@TheTom)
//
// Equivalence tests for the three NVFP4 dequant instruction-reduction
// transforms applied across butter's NVFP4 kernels (LagunaNvfp4Fuse.swift
// `header`, the kernel_overlays/*.metal qdot overlays, and the per-nibble
// `_scale`/`_scale_byte`/`_scale_b`/`_scale_f` decoders in the tracked
// butter_nvfp4_gather_*.metal / butter_nvfp4_moe_gateup_*.metal kernels):
//
//   1. NIBBLE-SPLIT word-parallel E2M1 pack decode (19 -> 13 int ops).
//   2. E4M3 scale CARRY SIGN-FOLD (~7 -> 5 int ops).
//   3. 2^22-class scale fold: the E2M1 `*16384.0f` half-domain restore is
//      folded into the group `scale` once (in `iron_nvfp4_qdot_16` /
//      `iron_nvfp4_pack_dot` / the expert-indexed dequant GEMV call
//      sites) instead of 4x per pack inside `iron_nvfp4_pack_dot_fast`.
//
// All three are shipped unconditional (no BUTTER_NVFP4_FAST_DEQUANT
// escape) because they are proved bit-exact here: #1 and #2 are checked
// exhaustively (every one of the 16 E2M1 nibble codes at every one of the
// 8 pack lanes; every one of the 256 E4M3 byte values), and #3 is a pure
// power-of-two float refactor (see the derivation comment on
// `iron_nvfp4_pack_dot_fast` in LagunaNvfp4Fuse.swift) validated here by
// a GPU kernel-integration test comparing the shipped (new) decode
// against a live-compiled copy of the pre-optimization (old) decode
// running through the identical dispatch/reduction structure.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("NVFP4 fast dequant equivalence")
struct Nvfp4FastDequantTests {
    // MARK: - Swift ports of the reference (pre-optimization) decoders

    /// `iron_nvfp4_e2m1` -- the slow-path per-nibble E2M1 LUT/switch decode
    /// every kernel family's fast path is checked against.
    private static func referenceE2M1(_ nibble: UInt32) -> Float {
        let magCode = nibble & 7
        let mag: Float
        switch magCode {
        case 0: mag = 0.0
        case 1: mag = 0.5
        case 2: mag = 1.0
        case 3: mag = 1.5
        case 4: mag = 2.0
        case 5: mag = 3.0
        case 6: mag = 4.0
        default: mag = 6.0
        }
        return (nibble & 8) != 0 ? -mag : mag
    }

    /// Pre-optimization `iron_nvfp4_e4m3_scale` / `nvfp4_decode_scale_byte`
    /// / `butter_na{x,x64}_nvfp4_scale{_f}` / `butter_nvfp4_scale{_b}`
    /// body (branch + exp2), ported verbatim to Swift.
    private static func referenceE4M3Scale(_ bits: UInt8) -> Float {
        let b = UInt32(bits)
        let exp = (b >> 3) & 15
        let mant = Float(b & 7)
        let mag: Float
        if exp == 0 {
            mag = mant * 0.001953125  // 2^-9
        } else {
            let p = Float(pow(2.0, Double(exp) - 7.0))
            mag = (1.0 + mant * 0.125) * p
        }
        return ((b >> 7) & 1) != 0 ? -mag : mag
    }

    // MARK: - Swift ports of the new (fast) decoders

    /// `iron_nvfp4_pack_dot_fast`'s NIBBLE-SPLIT bit-twiddle, ported to
    /// Swift: returns the 8 decoded (unscaled, pre-*16384) magnitudes for
    /// nibbles n0..n7 of `packed`, in that order.
    private static func fastPackDecodeRaw(_ packed: UInt32) -> [Float] {
        let xe = packed & 0x0F0F_0F0F
        let ge = xe | (xe << 3)
        let yo = packed & 0xF0F0_F0F0
        let go = yo | (yo >> 3)
        let M: UInt32 = 0x8E00_8E00
        let p0 = (ge << 9) & M
        let p1 = (go << 8) & M
        let p2 = (ge << 1) & M
        let p3 = go & M
        func half2(_ p: UInt32) -> (Float, Float) {
            let lo = UInt16(p & 0xFFFF)
            let hi = UInt16((p >> 16) & 0xFFFF)
            return (Float(Float16(bitPattern: lo)), Float(Float16(bitPattern: hi)))
        }
        let v04 = half2(p0), v15 = half2(p1), v26 = half2(p2), v37 = half2(p3)
        // half2 packs even/odd nibbles: (n0,n4), (n1,n5), (n2,n6), (n3,n7)
        return [v04.0, v15.0, v26.0, v37.0, v04.1, v15.1, v26.1, v37.1]
    }

    /// `iron_nvfp4_e4m3_scale`'s CARRY SIGN-FOLD, ported to Swift.
    private static func fastE4M3Scale(_ bits: UInt8) -> Float {
        let b = UInt32(bits)
        let h = UInt16((b + (b & 128)) << 7)
        return Float(Float16(bitPattern: h)) * 256.0
    }

    // MARK: - Test 1: exhaustive 16-code E2M1 nibble-split equivalence (CPU)

    @Test("NIBBLE-SPLIT decode matches iron_nvfp4_e2m1 for all 16 codes at every pack lane")
    func nibbleSplitExhaustive16Codes() throws {
        var mismatches: [String] = []
        for lane in 0 ..< 8 {
            for nibble: UInt32 in 0 ..< 16 {
                var packed: UInt32 = 0
                packed |= (nibble & 0xF) << (UInt32(lane) * 4)
                let decoded = Self.fastPackDecodeRaw(packed)
                let expected = Self.referenceE2M1(nibble)
                let got = decoded[lane] * 16384.0  // half-domain restore, as callers apply
                if got != expected {
                    mismatches.append(
                        "lane=\(lane) nibble=0x\(String(nibble, radix: 16)) "
                            + "got=\(got) expected=\(expected)")
                }
                // Every OTHER lane must decode to exactly 0 (neutral filler,
                // proves zero cross-lane contamination from the shared
                // ge/go spread).
                for other in 0 ..< 8 where other != lane {
                    #expect(
                        decoded[other] == 0,
                        Comment(rawValue: "lane=\(lane) nibble=0x\(String(nibble, radix: 16)) "
                            + "leaked into lane \(other): \(decoded[other])"))
                }
            }
        }
        #expect(mismatches.isEmpty, "NIBBLE-SPLIT mismatches:\n\(mismatches.joined(separator: "\n"))")
    }

    /// All 16 codes packed into a single word simultaneously (2 nibbles
    /// per lane can't fit 16 distinct codes in 8 lanes, so this covers
    /// codes 0..7 signed-positive in lanes 0..7 as a realistic "one full
    /// pack" sanity check on top of the per-lane isolation test above).
    @Test("NIBBLE-SPLIT decode matches iron_nvfp4_e2m1 for a full mixed pack")
    func nibbleSplitFullPack() throws {
        let nibbles: [UInt32] = [0x0, 0x1, 0x9, 0x3, 0xC, 0x5, 0xE, 0x7]  // mix of signs/mags
        var packed: UInt32 = 0
        for (lane, n) in nibbles.enumerated() {
            packed |= (n & 0xF) << (UInt32(lane) * 4)
        }
        let decoded = Self.fastPackDecodeRaw(packed).map { $0 * 16384.0 }
        for (lane, n) in nibbles.enumerated() {
            #expect(decoded[lane] == Self.referenceE2M1(n), "lane \(lane) nibble 0x\(String(n, radix: 16))")
        }
    }

    // MARK: - Test 2: exhaustive 256-value E4M3 fold equivalence (CPU)

    @Test("CARRY SIGN-FOLD E4M3 decode matches the branch/exp2 reference for all 256 byte values")
    func e4m3SignFoldExhaustive256() throws {
        var mismatches: [String] = []
        for raw in 0 ... 255 {
            let bits = UInt8(raw)
            let expected = Self.referenceE4M3Scale(bits)
            let got = Self.fastE4M3Scale(bits)
            if got != expected {
                mismatches.append("bits=0x\(String(bits, radix: 16)) got=\(got) expected=\(expected)")
            }
        }
        #expect(mismatches.isEmpty, "E4M3 sign-fold mismatches:\n\(mismatches.joined(separator: "\n"))")
    }

    // MARK: - Test 3: kernel-integration, bit-identical old vs new dequant path

    /// Live-compiled copy of `iron_nvfp4_dequant_gemv_expert_indexed_f32`
    /// using the PRE-OPTIMIZATION decode (branch/exp2 E4M3 scale + the
    /// original 19-op mask/shift E2M1 pack decode with the per-value
    /// `*16384.0f`), same dispatch/reduction geometry as the shipped
    /// kernel. Diffing this against the shipped (now fast-path) kernel
    /// isolates the decode-formula change: same buffers, same TG/SIMD
    /// reduction tree, only the per-element decode differs.
    private static let oldRefSource = """
    #include <metal_stdlib>
    using namespace metal;

    static inline float nvfp4_ref_old_e4m3_scale(uchar bits) {
        uint b = uint(bits);
        uint exp = (b >> 3) & 15u;
        float mant = float(b & 7u);
        float mag;
        if (exp == 0u) {
            mag = mant * 0.001953125f;
        } else {
            mag = (1.0f + mant * 0.125f) * exp2(float(exp) - 7.0f);
        }
        return ((b >> 7) & 1u) != 0u ? -mag : mag;
    }

    static inline float nvfp4_ref_old_pack_dot_fast(
        uint packed, float scale, const thread float* input, uint p_off
    ) {
        float accum = 0.0f;
        const uint p0 =
            ((packed & 0x00070007u) << 9) | ((packed & 0x00080008u) << 12);
        const uint p1 =
            ((packed & 0x00700070u) << 5) | ((packed & 0x00800080u) << 8);
        const uint p2 =
            ((packed & 0x07000700u) << 1) | ((packed & 0x08000800u) << 4);
        const uint p3 =
            ((packed & 0x70007000u) >> 3) | (packed & 0x80008000u);
        const float2 v04 = float2(as_type<half2>(p0)) * 16384.0f;
        const float2 v15 = float2(as_type<half2>(p1)) * 16384.0f;
        const float2 v26 = float2(as_type<half2>(p2)) * 16384.0f;
        const float2 v37 = float2(as_type<half2>(p3)) * 16384.0f;
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

    kernel void nvfp4_ref_old_dequant_gemv_expert_indexed_f32(
        const device uint *weights_stacked [[buffer(0)]],
        const device uchar *scales_stacked [[buffer(1)]],
        const device float *input [[buffer(2)]],
        const device uint *expert_index [[buffer(3)]],
        device float *output [[buffer(4)]],
        constant uint &in_dim [[buffer(5)]],
        constant uint &out_dim [[buffer(6)]],
        constant uint &block_size [[buffer(7)]],
        constant float &global [[buffer(8)]],
        uint3 _tid3  [[thread_position_in_threadgroup]],
        uint3 _tgid3 [[threadgroup_position_in_grid]],
        uint3 _lsize3 [[threads_per_threadgroup]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]
    ) {
        uint tid    = _tid3.x;
        uint tgid_x = _tgid3.x;
        uint lsize  = _lsize3.x;
        uint n_simd = lsize / 32u;
        if (n_simd == 0u) { return; }
        threadgroup float v_total_sg[32];

        uint row = tgid_x;
        uint n_blocks = in_dim / block_size;
        uint expert = expert_index[0];
        uint n_packs_per_row = in_dim / 8u;
        uint packs_per_block = block_size / 8u;
        uint row_pack_off = expert * out_dim * n_packs_per_row + row * n_packs_per_row;
        uint row_block_off = expert * out_dim * n_blocks + row * n_blocks;

        float acc = 0.0f;
        uint p_iters = (n_packs_per_row + lsize - 1u) / lsize;
        for (uint p_iter = 0u; p_iter < p_iters; p_iter += 1u) {
            uint pack_idx = p_iter * lsize + tid;
            if (pack_idx < n_packs_per_row) {
                uint blk = pack_idx / packs_per_block;
                float scale =
                    nvfp4_ref_old_e4m3_scale(scales_stacked[row_block_off + blk]) * global;
                uint packed = weights_stacked[row_pack_off + pack_idx];
                uint p_off = pack_idx * 8u;
                thread float xv[8];
                for (uint i = 0u; i < 8u; ++i) {
                    xv[i] = input[p_off + i];
                }
                acc += nvfp4_ref_old_pack_dot_fast(packed, scale, xv, 0u);
            }
        }

        float sg_sum = simd_sum(acc);
        if (simd_lane == 0u) {
            v_total_sg[simd_group] = sg_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        if (simd_group == 0u) {
            float lane_v = (simd_lane < n_simd) ? v_total_sg[simd_lane] : 0.0f;
            total = simd_sum(lane_v);
            if (simd_lane == 0u) {
                v_total_sg[0] = total;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            output[row] = v_total_sg[0];
        }
    }
    """

    @Test("Shipped (fast) expert-indexed GEMV kernel is bit-identical to the pre-optimization decode")
    func kernelIntegrationBitIdenticalOldVsNew() throws {
        let device = Device.shared
        let mtlDevice = device.mtlDevice

        // Live-compile the old-formula reference kernel.
        let lib = try mtlDevice.makeLibrary(source: Self.oldRefSource, options: MTLCompileOptions())
        guard let fn = lib.makeFunction(name: "nvfp4_ref_old_dequant_gemv_expert_indexed_f32") else {
            Issue.record("failed to make old-ref function")
            return
        }
        let oldPso = try mtlDevice.makeComputePipelineState(function: fn)

        // Synthetic single-expert weights: vary nibble codes and scale
        // bytes across the full pack/block range so both decode paths are
        // meaningfully exercised (test 1/2 above already prove exhaustive
        // per-value equivalence; this proves the end-to-end wiring/fold
        // composition is bit-exact once accumulated through the same
        // TG/SIMD reduction tree).
        let outDim = 8
        let inDim = 256
        let blockSize = 16
        let packs = inDim / 8   // 32
        let groups = inDim / blockSize  // 16
        let global: Float = 1.375

        var wHost = [UInt32](repeating: 0, count: outDim * packs)
        for r in 0 ..< outDim {
            for p in 0 ..< packs {
                var word: UInt32 = 0
                for nib in 0 ..< 8 {
                    let code = UInt32((r * 131 + p * 17 + nib * 7) % 16)
                    word |= code << UInt32(nib * 4)
                }
                wHost[r * packs + p] = word
            }
        }
        var sHost = [UInt8](repeating: 0, count: outDim * groups)
        for r in 0 ..< outDim {
            for g in 0 ..< groups {
                sHost[r * groups + g] = UInt8((r * 53 + g * 29) % 256)
            }
        }
        var xHost = [Float](repeating: 0, count: inDim)
        for i in 0 ..< inDim {
            xHost[i] = Float(i % 13) * 0.25 - 1.5
        }

        let wBuf = device.makeBuffer(length: wHost.count * 4)
        let sBuf = device.makeBuffer(length: sHost.count)
        let xBuf = device.makeBuffer(length: xHost.count * 4)
        let idxBuf = device.makeBuffer(length: 4)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 4) }
        memset(idxBuf.contents(), 0, 4)  // expert 0

        let weight = Tensor(buffer: wBuf, offset: 0, shape: [1, outDim, packs], dtype: .u32)
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [1, outDim, groups], dtype: .u8)
        let input = Tensor(buffer: xBuf, offset: 0, shape: [inDim], dtype: .f32)
        let expertIdx = Tensor(buffer: idxBuf, offset: 0, shape: [1], dtype: .u32)

        let outNew = Tensor.empty(shape: [outDim], dtype: .f32, device: device)
        let outOld = Tensor.empty(shape: [outDim], dtype: .f32, device: device)

        let cmd = device.makeCommandBuffer()

        // New (shipped, fast decode) path via the real Ops entry point.
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: [weight], scalesStacked: [scales],
            inputs: [input], expertIndices: [expertIdx], outputs: [outNew],
            blockSize: blockSize, global: global, on: cmd)

        // Old (pre-optimization decode) path via the live-compiled reference.
        let tpg = 64
        let grid = MTLSize(width: outDim, height: 1, depth: 1)
        let tgSize = MTLSize(width: tpg, height: 1, depth: 1)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            Issue.record("failed to make compute encoder")
            return
        }
        enc.setComputePipelineState(oldPso)
        enc.setBuffer(wBuf, offset: 0, index: 0)
        enc.setBuffer(sBuf, offset: 0, index: 1)
        enc.setBuffer(xBuf, offset: 0, index: 2)
        enc.setBuffer(idxBuf, offset: 0, index: 3)
        enc.setBuffer(outOld.buffer, offset: outOld.offset, index: 4)
        var inDimV = UInt32(inDim)
        var outDimV = UInt32(outDim)
        var blockV = UInt32(blockSize)
        var globalV = global
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&outDimV, length: 4, index: 6)
        enc.setBytes(&blockV, length: 4, index: 7)
        enc.setBytes(&globalV, length: 4, index: 8)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tgSize)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "cmd status \(cmd.status.rawValue)")

        func toF32(_ t: Tensor) -> [Float] {
            let n = t.elementCount
            let ptr = t.buffer.contents().advanced(by: t.offset)
                .bindMemory(to: Float.self, capacity: n)
            return (0 ..< n).map { ptr[$0] }
        }
        let newVals = toF32(outNew)
        let oldVals = toF32(outOld)
        print("[nvfp4-fast-dequant] new=\(newVals)")
        print("[nvfp4-fast-dequant] old=\(oldVals)")

        var maxAbsDiff: Float = 0
        for i in 0 ..< outDim {
            maxAbsDiff = max(maxAbsDiff, abs(newVals[i] - oldVals[i]))
            #expect(newVals[i] == oldVals[i], "row \(i): new=\(newVals[i]) old=\(oldVals[i])")
        }
        #expect(maxAbsDiff == 0, "max_abs_diff old vs new = \(maxAbsDiff)")
    }
}
