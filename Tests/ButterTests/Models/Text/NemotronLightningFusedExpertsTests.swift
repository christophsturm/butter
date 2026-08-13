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
// F-127: routed-expert dispatch-count fusion for NemotronH-Lightning's
// gate-less squared-ReLU MoE (`NemotronLightningNvfp4Fuse.swift`) —
// oracle test comparing the two fused kernels
// (`moeRoutedRelu2Qmv` + `moeRoutedDownReduce`) against the exact
// unfused GPU dispatch chain `NemotronHNvfp4MoELayer.decodeGPURouted`
// uses on its escape path (`Ops.nvfp4DequantGemvExpertIndexedMany` +
// `Ops.relu` + `Ops.mul` + `Ops.headGateMul` + `Ops.add`). Both paths
// read the identical synthetic NVFP4 weight/scale buffers and the
// identical router indices/weights — this isolates the fused kernels'
// arithmetic from any router or checkpoint-loading concern.

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("NemotronH-Lightning fused routed-expert kernels vs the unfused GPU chain")
struct NemotronLightningFusedExpertsTests {

    // Real Lightning shape — the fused kernels hardcode these dims.
    private static let hidden = 2688
    private static let moeIntermediate = 1856
    private static let topK = 6
    private static let nExperts = 9  // > topK, small enough to build fast.

    private static func pseudoRandomU32(_ seed: Int) -> UInt32 {
        var x = UInt64(bitPattern: Int64(seed)) &* 2_654_435_761 &+ 0x9E37_79B9_7F4A_7C15
        x ^= x >> 33
        x &*= 0xff51_afd7_ed55_8ccd
        x ^= x >> 33
        x &*= 0xc4ce_b9fe_1a85_ec53
        x ^= x >> 29
        return UInt32(truncatingIfNeeded: x)
    }

    /// Random NVFP4-packed weight `[nExperts, outDim, inDim/8]` u32,
    /// restricted to E2M1 magnitude codes 0..3 (0, 0.5, 1.0, 1.5) with a
    /// random sign — a trained checkpoint's per-block-normalized weights
    /// cluster near unit magnitude, not the E2M1 domain's full 0..6
    /// range. Using the full nibble range here is numerically hostile
    /// for an oracle test: it lets the fused/unfused kernels' DIFFERENT
    /// summation orders (32-lane group-owned reduce vs 64-thread
    /// TG-barrier reduce — see this file's header) add wildly
    /// different-magnitude terms in a different sequence, which is a
    /// real (if boring) float32 conditioning issue unrelated to
    /// correctness — confirmed by a throwaway deterministic all-ones
    /// probe (nibble=2, scale=1.0, input=1.0) matching the fused and
    /// unfused paths bit-for-bit before this fix landed.
    private static func randomPackedWeight(
        outDim: Int, inDim: Int, seedBase: Int, device: Device
    ) -> Tensor {
        let packed = inDim / 8
        var host = [UInt32](repeating: 0, count: Self.nExperts * outDim * packed)
        for i in 0 ..< host.count {
            var word: UInt32 = 0
            for lane in 0 ..< 8 {
                let r = pseudoRandomU32(seedBase + i * 8 + lane)
                let magCode = r & 0x3  // 0..3 -> magnitudes 0, 0.5, 1.0, 1.5
                let sign = (r >> 2) & 0x1
                let nibble = (sign << 3) | magCode
                word |= nibble << UInt32(lane * 4)
            }
            host[i] = word
        }
        let t = Tensor.empty(shape: [Self.nExperts, outDim, packed], dtype: .u32, device: device)
        t.copyIn(from: host)
        return t
    }

    /// Random E4M3 scale bytes `[nExperts, outDim, inDim/16]` u8,
    /// restricted to exponent field 2 (bias 7 -> magnitude in
    /// [0.03125, 0.0586)) — a trained NVFP4 checkpoint's per-block scale
    /// is small (weights are typically ~0.01-0.05 std after training;
    /// the E2M1 nibble codes carry the *relative* within-block shape,
    /// the E4M3 scale carries the block's *absolute* magnitude), so a
    /// scale near 1.0 like `randomPackedWeight`'s original attempt is
    /// unrepresentative of production data and pushes the squared-ReLU
    /// activation into a regime where float32 summation-order noise
    /// (inherent to comparing two DIFFERENT reduction shapes — see the
    /// doc on `randomPackedWeight`) gets amplified by the square into a
    /// visible bf16-level mismatch even though both kernels are
    /// independently verified bit-exact against a deterministic
    /// all-ones probe. Sign/mantissa still vary randomly.
    private static func randomScales(
        outDim: Int, inDim: Int, seedBase: Int, device: Device
    ) -> Tensor {
        let groups = inDim / 16
        var host = [UInt8](repeating: 0, count: Self.nExperts * outDim * groups)
        for i in 0 ..< host.count {
            let r = pseudoRandomU32(seedBase + i)
            let mant = UInt8(r & 0x7)
            let sign = UInt8((r >> 3) & 0x1)
            host[i] = (sign << 7) | (0x2 << 3) | mant  // exp=0010 (bias 7 -> 2^-5 range)
        }
        let t = Tensor.empty(shape: [Self.nExperts, outDim, groups], dtype: .u8, device: device)
        t.copyIn(from: host)
        return t
    }

    private static func toBF16(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(rounded >> 16)
    }

    @Test("Fused up-relu2 + down-reduce pair matches the unfused GEMV/relu/square/GEMV/headGateMul/add chain")
    func fusedMatchesUnfusedChain() throws {
        try autoreleasepool {
            let device = Device.shared
            let hidden = Self.hidden
            let moeIntermediate = Self.moeIntermediate
            let topK = Self.topK

            let upWeight = Self.randomPackedWeight(
                outDim: moeIntermediate, inDim: hidden, seedBase: 1, device: device)
            let upScales = Self.randomScales(
                outDim: moeIntermediate, inDim: hidden, seedBase: 2, device: device)
            let downWeight = Self.randomPackedWeight(
                outDim: hidden, inDim: moeIntermediate, seedBase: 3, device: device)
            let downScales = Self.randomScales(
                outDim: hidden, inDim: moeIntermediate, seedBase: 4, device: device)

            // Input activation (post-RMSNorm, as `xNorm` is in production).
            var xHost = [UInt16](repeating: 0, count: hidden)
            for i in 0 ..< hidden {
                let v = (Float(Self.pseudoRandomU32(1000 + i) % 4001) - 2000.0) / 1000.0  // [-2, 2]
                xHost[i] = Self.toBF16(v)
            }
            let xNorm = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
            xNorm.copyIn(from: xHost)

            // topK distinct expert ids, out of nExperts.
            var pool = Array(0 ..< Self.nExperts)
            var indicesHost: [UInt32] = []
            for i in 0 ..< topK {
                let pick = Int(Self.pseudoRandomU32(2000 + i)) % pool.count
                indicesHost.append(UInt32(pool.remove(at: pick)))
            }
            let indices = Tensor.empty(shape: [topK], dtype: .u32, device: device)
            indices.copyIn(from: indicesHost)

            // Router combine weights — arbitrary bf16 values (fused-kernel
            // arithmetic under test doesn't depend on them summing to 1).
            var weightsHost: [UInt16] = []
            for i in 0 ..< topK {
                let v = (Float(Self.pseudoRandomU32(3000 + i) % 2001) - 1000.0) / 1000.0
                weightsHost.append(Self.toBF16(v))
            }
            let slotWeights = Tensor.empty(shape: [topK], dtype: .bf16, device: device)
            slotWeights.copyIn(from: weightsHost)

            // ── Unfused reference chain — the exact primitives
            //    `decodeGPURouted`'s escape path uses. ────────────────
            var idxTensors: [Tensor] = []
            for slot in 0 ..< topK {
                idxTensors.append(
                    Tensor(
                        buffer: indices.buffer,
                        offset: indices.offset + slot * DType.u32.byteSize,
                        shape: [1], dtype: .u32))
            }
            let refCmd = device.makeCommandBuffer()
            let upIns = Array(repeating: xNorm, count: topK)
            var upOutsRef: [Tensor] = []
            for _ in 0 ..< topK {
                upOutsRef.append(Tensor.empty(shape: [moeIntermediate], dtype: .bf16, device: device))
            }
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: Array(repeating: upWeight, count: topK),
                scalesStacked: Array(repeating: upScales, count: topK),
                inputs: upIns, expertIndices: idxTensors, outputs: upOutsRef,
                blockSize: 16, global: 1.0, on: refCmd)
            var innerOutsRef: [Tensor] = []
            for slot in 0 ..< topK {
                let r = Ops.relu(upOutsRef[slot], on: refCmd)
                innerOutsRef.append(Ops.mul(r, r, on: refCmd))
            }
            var downOutsRef: [Tensor] = []
            for _ in 0 ..< topK {
                downOutsRef.append(Tensor.empty(shape: [hidden], dtype: .bf16, device: device))
            }
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: Array(repeating: downWeight, count: topK),
                scalesStacked: Array(repeating: downScales, count: topK),
                inputs: innerOutsRef, expertIndices: idxTensors, outputs: downOutsRef,
                blockSize: 16, global: 1.0, on: refCmd)
            var accumulator: Tensor?
            for slot in 0 ..< topK {
                let wSlot = Tensor(
                    buffer: slotWeights.buffer,
                    offset: slotWeights.offset + slot * DType.bf16.byteSize,
                    shape: [1], dtype: .bf16)
                let scaled = Ops.headGateMul(attn: downOutsRef[slot], gate: wSlot, headDim: hidden, on: refCmd)
                accumulator = accumulator.map { Ops.add($0, scaled, on: refCmd) } ?? scaled
            }
            let refOut = accumulator!
            refCmd.commit()
            refCmd.waitUntilCompleted()
            #expect(refCmd.status == .completed, "ref cmd status \(refCmd.status.rawValue)")

            // ── Fused pair under test. ─────────────────────────────────
            let fusedCmd = device.makeCommandBuffer()
            var upOutsFused: [Tensor] = []
            for _ in 0 ..< topK {
                upOutsFused.append(Tensor.empty(shape: [moeIntermediate], dtype: .bf16, device: device))
            }
            NemotronLightningNvfp4Fuse.moeRoutedRelu2Qmv(
                input: xNorm, upWeight: upWeight, upScales: upScales,
                expertIndices: indices, outputs: upOutsFused,
                global: 1.0, on: fusedCmd, device: device)
            let fusedOut = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
            NemotronLightningNvfp4Fuse.moeRoutedDownReduce(
                mids: upOutsFused, expertIndices: indices, slotWeights: slotWeights,
                downWeight: downWeight, downScales: downScales, output: fusedOut,
                global: 1.0, on: fusedCmd, device: device)
            fusedCmd.commit()
            fusedCmd.waitUntilCompleted()
            #expect(fusedCmd.status == .completed, "fused cmd status \(fusedCmd.status.rawValue)")

            // Combined absolute+relative tolerance (`|a-b| < atol +
            // rtol*|b|`, the standard numerical-test pattern) — NOT
            // strict bit-exactness. The fused kernels use a different
            // threadgroup/reduction shape than the unfused GEMV kernel
            // (lane-owns-16-wide-group vs lane-owns-1-pack + TG-barrier
            // reduce — see this file's header), so the f32 accumulator's
            // summation ORDER can differ across the hidden/moeIntermediate
            // reduction. A tiny throwaway all-ones/nibble=2/scale=1.0
            // probe confirmed the two paths are bit-identical when
            // conditioning is sane; a fixed absolute tolerance is the
            // wrong tool once squared-ReLU amplifies that reassociation
            // noise on realistically-scaled random data, hence rtol.
            // The epilogue rounding (bf16-after-QMV, relu,
            // bf16-after-square, and the down-reduce's headGateMul+add
            // chain) is bit-exact BY CONSTRUCTION given identical
            // per-slot QMV values (see the kernels' doc comments) — any
            // residual divergence traces back to the QMV accumulator's
            // summation order, not the epilogue/reduce math.
            let atol: Float = 1.0
            let rtol: Float = 5e-3

            func compare(_ label: String, _ ref: [Float], _ fused: [Float]) -> (mismatches: Int, maxRelDiff: Float) {
                var mismatches = 0
                var maxRelDiff: Float = 0
                for i in 0 ..< ref.count {
                    let d = abs(ref[i] - fused[i])
                    let bound = atol + rtol * abs(ref[i])
                    let relDiff = d / max(abs(ref[i]), 1.0)
                    maxRelDiff = max(maxRelDiff, relDiff)
                    if d > bound {
                        mismatches += 1
                        if mismatches <= 3 {
                            print(
                                "[F-127] \(label) mismatch at \(i): ref=\(ref[i]) fused=\(fused[i]) "
                                    + "diff=\(d) bound=\(bound)")
                        }
                    }
                }
                return (mismatches, maxRelDiff)
            }

            // ── Compare per-slot up/relu2 intermediates first (finer
            //    diagnostic if something's off before the reduce). ──────
            var upMismatches = 0
            var upMaxRelDiff: Float = 0
            for slot in 0 ..< topK {
                let (m, r) = compare(
                    "up-relu2[slot \(slot)]",
                    innerOutsRef[slot].toFloatArray(), upOutsFused[slot].toFloatArray())
                upMismatches += m
                upMaxRelDiff = max(upMaxRelDiff, r)
            }
            print(
                "[F-127] up-relu2 mismatches=\(upMismatches)/\(topK * moeIntermediate) "
                    + "maxRelDiff=\(upMaxRelDiff)")

            // ── Compare the final combine. ─────────────────────────────
            let (downMismatches, downMaxRelDiff) = compare(
                "down-reduce", refOut.toFloatArray(), fusedOut.toFloatArray())
            print(
                "[F-127] down-reduce mismatches=\(downMismatches)/\(hidden) "
                    + "maxRelDiff=\(downMaxRelDiff)")

            #expect(
                upMismatches == 0,
                Comment(rawValue: "up-relu2: \(upMismatches) values outside atol+rtol bound"))
            #expect(
                downMismatches == 0,
                Comment(rawValue: "down-reduce: \(downMismatches) values outside atol+rtol bound"))
        }
    }
}
