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
// F-132 batched prefill — `decodeMulti` overrides for the attention,
// dense MLP, and NVFP4 MoE NemotronH layer kinds. Unit-scale oracle:
// each override's `[T, hidden]` output must match T sequential calls to
// the same layer's per-token `decode(...)` — the existing, already-
// covered reference path. Small synthetic dims (not the real
// Nemotron-3.5-Lightning shapes) keep this GPU-but-no-checkpoint and
// fast; the real-checkpoint prefill-vs-per-token-loop parity gate lives
// in the integration suite (byte-identical greedy continuation).

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("NemotronH F-132 batched prefill decodeMulti vs per-token decode loop")
struct NemotronHBatchedPrefillTests {

    // ─── Shared helpers ────────────────────────────────────────────────

    private static func pseudoRandomU32(_ seed: Int) -> UInt32 {
        var x = UInt64(bitPattern: Int64(seed)) &* 2_654_435_761 &+ 0x9E37_79B9_7F4A_7C15
        x ^= x >> 33
        x &*= 0xff51_afd7_ed55_8ccd
        x ^= x >> 33
        x &*= 0xc4ce_b9fe_1a85_ec53
        x ^= x >> 29
        return UInt32(truncatingIfNeeded: x)
    }

    private static func toBF16(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(rounded >> 16)
    }

    private static func randomBF16Matrix(
        rows: Int, cols: Int, seedBase: Int, scale: Float = 0.1, device: Device
    ) -> Tensor {
        var host = [UInt16](repeating: 0, count: rows * cols)
        for i in 0 ..< host.count {
            let r = (Float(Self.pseudoRandomU32(seedBase + i) % 2001) - 1000.0) / 1000.0
            host[i] = Self.toBF16(r * scale)
        }
        let t = Tensor.empty(shape: [rows, cols], dtype: .bf16, device: device)
        t.copyIn(from: host)
        return t
    }

    private static func randomBF16Rows(
        t: Int, hidden: Int, seedBase: Int, scale: Float = 1.0, device: Device
    ) -> Tensor {
        randomBF16Matrix(rows: t, cols: hidden, seedBase: seedBase, scale: scale, device: device)
    }

    private static func compareRows(
        _ label: String, ref: [Float], got: [Float], atol: Float, rtol: Float
    ) -> Int {
        var mismatches = 0
        for i in 0 ..< ref.count {
            let d = abs(ref[i] - got[i])
            let bound = atol + rtol * abs(ref[i])
            if d > bound {
                mismatches += 1
                if mismatches <= 3 {
                    print("[\(label)] mismatch at \(i): ref=\(ref[i]) got=\(got[i]) diff=\(d) bound=\(bound)")
                }
            }
        }
        return mismatches
    }

    // ─── Dense MLP layer ─────────────────────────────────────────────

    @Test("dense MLP layer decodeMulti matches T sequential decode() calls")
    func mlpDecodeMultiMatchesPerRow() {
        autoreleasepool {
            let device = Device.shared
            let dim = 128
            let intermediate = 256
            let t = 5

            let normWeight = Tensor.empty(shape: [dim], dtype: .bf16, device: device)
            normWeight.copyIn(from: [UInt16](repeating: Self.toBF16(1.0), count: dim))
            let norm = RMSNorm(weight: normWeight, eps: 1e-6)
            let upProj = AnyLinear(
                Linear(weight: Self.randomBF16Matrix(rows: intermediate, cols: dim, seedBase: 10, device: device)))
            let downProj = AnyLinear(
                Linear(weight: Self.randomBF16Matrix(rows: dim, cols: intermediate, seedBase: 20, device: device)))
            let layer = NemotronHMLPLayer(
                norm: norm, upProj: upProj, downProj: downProj, hidden: dim, intermediate: intermediate)

            let hAll = Self.randomBF16Rows(t: t, hidden: dim, seedBase: 30, device: device)

            // Reference: per-row decode() loop on one shared cmd (matches
            // the protocol default decodeMulti's own command-buffer
            // discipline — NemotronHMLPLayer never commits mid-decode).
            let refCmd = device.makeCommandBuffer()
            var refRows: [Tensor] = []
            for i in 0 ..< t {
                let row = hAll.slicedRows(start: i, count: 1).reshaped(to: [dim])
                refRows.append(
                    layer.decode(row, position: i, cache: StatelessLayerCache(), cmd: refCmd, device: device))
            }
            refCmd.commit()
            refCmd.waitUntilCompleted()

            // Batched decodeMulti under test.
            let batchCmd = device.makeCommandBuffer()
            let out = layer.decodeMulti(
                hAll, startingAt: 0, cache: StatelessLayerCache(), cmd: batchCmd, device: device)
            batchCmd.commit()
            batchCmd.waitUntilCompleted()

            for i in 0 ..< t {
                let refFlat = refRows[i].toFloatArray()
                let gotFlat = out.slicedRows(start: i, count: 1).reshaped(to: [dim]).toFloatArray()
                let mismatches = Self.compareRows(
                    "mlp row \(i)", ref: refFlat, got: gotFlat, atol: 0.05, rtol: 1e-2)
                #expect(mismatches == 0, Comment(rawValue: "row \(i): \(mismatches)/\(dim) mismatches"))
            }
        }
    }

    // ─── Attention layer (no RoPE, GQA, real KVCache) ──────────────────

    @Test("attention layer decodeMulti matches T sequential decode() calls")
    func attentionDecodeMultiMatchesPerRow() {
        autoreleasepool {
            let device = Device.shared
            let headDim = 128  // sdpaMulti requires 128 or 256.
            let nHeads = 2
            let nKVHeads = 1
            let hidden = nHeads * headDim
            let t = 4

            let normWeight = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
            normWeight.copyIn(from: [UInt16](repeating: Self.toBF16(1.0), count: hidden))
            let norm = RMSNorm(weight: normWeight, eps: 1e-6)
            let qProj = AnyLinear(
                Linear(
                    weight: Self.randomBF16Matrix(
                        rows: nHeads * headDim, cols: hidden, seedBase: 40, device: device)))
            let kProj = AnyLinear(
                Linear(
                    weight: Self.randomBF16Matrix(
                        rows: nKVHeads * headDim, cols: hidden, seedBase: 50, device: device)))
            let vProj = AnyLinear(
                Linear(
                    weight: Self.randomBF16Matrix(
                        rows: nKVHeads * headDim, cols: hidden, seedBase: 60, device: device)))
            let oProj = AnyLinear(
                Linear(
                    weight: Self.randomBF16Matrix(
                        rows: hidden, cols: nHeads * headDim, seedBase: 70, device: device)))
            let layer = NemotronHAttentionLayer(
                norm: norm, qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim)

            let hAll = Self.randomBF16Rows(t: t, hidden: hidden, seedBase: 80, device: device)

            // Reference: per-row decode() loop against its own KVCache,
            // on one shared cmd (attention never commits mid-decode).
            let refCache = KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: 16, dtype: .bf16, device: device)
            let refCmd = device.makeCommandBuffer()
            var refRows: [Tensor] = []
            for i in 0 ..< t {
                let row = hAll.slicedRows(start: i, count: 1).reshaped(to: [hidden])
                refRows.append(
                    layer.decode(row, position: i, cache: refCache, cmd: refCmd, device: device))
            }
            refCmd.commit()
            refCmd.waitUntilCompleted()

            // Batched decodeMulti under test, against a fresh KVCache.
            let batchCache = KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: 16, dtype: .bf16, device: device)
            let batchCmd = device.makeCommandBuffer()
            let out = layer.decodeMulti(
                hAll, startingAt: 0, cache: batchCache, cmd: batchCmd, device: device)
            batchCmd.commit()
            batchCmd.waitUntilCompleted()

            #expect(refCache.length == batchCache.length)

            for i in 0 ..< t {
                let refFlat = refRows[i].toFloatArray()
                let gotFlat = out.slicedRows(start: i, count: 1).reshaped(to: [hidden]).toFloatArray()
                let mismatches = Self.compareRows(
                    "attn row \(i)", ref: refFlat, got: gotFlat, atol: 0.1, rtol: 2e-2)
                #expect(mismatches == 0, Comment(rawValue: "row \(i): \(mismatches)/\(hidden) mismatches"))
            }
        }
    }

    // ─── NVFP4 MoE layer (GPU router path) ─────────────────────────────

    private static func randomPackedWeight(
        nExperts: Int, outDim: Int, inDim: Int, seedBase: Int, device: Device
    ) -> Tensor {
        let packed = inDim / 8
        var host = [UInt32](repeating: 0, count: nExperts * outDim * packed)
        for i in 0 ..< host.count {
            var word: UInt32 = 0
            for lane in 0 ..< 8 {
                let r = Self.pseudoRandomU32(seedBase + i * 8 + lane)
                let magCode = r & 0x3
                let sign = (r >> 2) & 0x1
                let nibble = (sign << 3) | magCode
                word |= nibble << UInt32(lane * 4)
            }
            host[i] = word
        }
        let tt = Tensor.empty(shape: [nExperts, outDim, packed], dtype: .u32, device: device)
        tt.copyIn(from: host)
        return tt
    }

    private static func randomScales(
        nExperts: Int, outDim: Int, inDim: Int, seedBase: Int, device: Device
    ) -> Tensor {
        let groups = inDim / 16
        var host = [UInt8](repeating: 0, count: nExperts * outDim * groups)
        for i in 0 ..< host.count {
            let r = Self.pseudoRandomU32(seedBase + i)
            let mant = UInt8(r & 0x7)
            let sign = UInt8((r >> 3) & 0x1)
            host[i] = (sign << 7) | (0x2 << 3) | mant
        }
        let tt = Tensor.empty(shape: [nExperts, outDim, groups], dtype: .u8, device: device)
        tt.copyIn(from: host)
        return tt
    }

    @Test("NVFP4 MoE layer decodeMulti (GPU router) matches T sequential decode() calls")
    func moeDecodeMultiMatchesPerRow() throws {
        // Small synthetic shapes — deliberately NOT the real Lightning
        // dims (hidden=2688/moeIntermediate=1856), so `usesFusedExperts`
        // is false and `decode()`'s GPU-router reference takes the
        // generic per-slot GEMV/relu/square/GEMV/headGateMul/add escape
        // path (`Ops.nvfp4DequantGemvExpertIndexedMany`) — an
        // independent implementation from `decodeMulti`'s gather-GEMM +
        // sort-route path, so a match here is a real cross-check, not a
        // tautology.
        try autoreleasepool {
            let device = Device.shared
            let hidden = 128
            let moeIntermediate = 64
            let nExperts = 8
            let topK = 4
            let t = 5

            let upWeight = Self.randomPackedWeight(
                nExperts: nExperts, outDim: moeIntermediate, inDim: hidden, seedBase: 100, device: device)
            let upScales = Self.randomScales(
                nExperts: nExperts, outDim: moeIntermediate, inDim: hidden, seedBase: 200, device: device)
            let downWeight = Self.randomPackedWeight(
                nExperts: nExperts, outDim: hidden, inDim: moeIntermediate, seedBase: 300, device: device)
            let downScales = Self.randomScales(
                nExperts: nExperts, outDim: hidden, inDim: moeIntermediate, seedBase: 400, device: device)
            let stacked = NemotronHStackedNvfp4Experts(
                upWeight: upWeight, upScales: upScales,
                downWeight: downWeight, downScales: downScales,
                numExperts: nExperts, moeIntermediate: moeIntermediate, hidden: hidden,
                blockSize: 16, dtype: .bf16)

            let sharedUpW = Self.randomPackedWeight(
                nExperts: 1, outDim: moeIntermediate, inDim: hidden, seedBase: 500, device: device
            ).reshaped(to: [moeIntermediate, hidden / 8])
            let sharedUpS = Self.randomScales(
                nExperts: 1, outDim: moeIntermediate, inDim: hidden, seedBase: 600, device: device
            ).reshaped(to: [moeIntermediate, hidden / 16])
            let sharedDownW = Self.randomPackedWeight(
                nExperts: 1, outDim: hidden, inDim: moeIntermediate, seedBase: 700, device: device
            ).reshaped(to: [hidden, moeIntermediate / 8])
            let sharedDownS = Self.randomScales(
                nExperts: 1, outDim: hidden, inDim: moeIntermediate, seedBase: 800, device: device
            ).reshaped(to: [hidden, moeIntermediate / 16])
            let sharedUp = Nvfp4Linear(weight: sharedUpW, scales: sharedUpS, blockSize: 16, global: 1.0)
            let sharedDown = Nvfp4Linear(weight: sharedDownW, scales: sharedDownS, blockSize: 16, global: 1.0)

            let normWeight = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
            normWeight.copyIn(from: [UInt16](repeating: Self.toBF16(1.0), count: hidden))
            let norm = RMSNorm(weight: normWeight, eps: 1e-6)
            let gate = AnyLinear(
                Linear(weight: Self.randomBF16Matrix(rows: nExperts, cols: hidden, seedBase: 900, device: device)))
            // Small per-expert bias spread — enough to make selection
            // non-trivial without the extreme common-mode offset the
            // dedicated router-precision test already covers.
            var biasHost = [Float](repeating: 0, count: nExperts)
            for i in 0 ..< nExperts {
                biasHost[i] = (Float(Self.pseudoRandomU32(1000 + i) % 201) - 100.0) / 100.0
            }

            let layer = NemotronHNvfp4MoELayer(
                norm: norm, gate: gate, eScoreCorrectionBias: biasHost,
                stacked: stacked, sharedUp: sharedUp, sharedDown: sharedDown,
                hidden: hidden, topK: topK,
                routedScalingFactor: 1.5, normTopkProb: true, dtype: .bf16,
                device: device)
            #expect(layer.usesGpuRouter, "test requires the GPU-router path to be active")

            let hAll = Self.randomBF16Rows(t: t, hidden: hidden, seedBase: 1100, scale: 0.5, device: device)

            // Reference: per-row decode() loop on one shared cmd (GPU
            // router never commits mid-decode).
            let refCmd = device.makeCommandBuffer()
            var refRows: [Tensor] = []
            for i in 0 ..< t {
                let row = hAll.slicedRows(start: i, count: 1).reshaped(to: [hidden])
                refRows.append(
                    layer.decode(row, position: i, cache: StatelessLayerCache(), cmd: refCmd, device: device))
            }
            refCmd.commit()
            refCmd.waitUntilCompleted()
            #expect(refCmd.status == .completed)

            // Batched decodeMulti under test.
            let batchCmd = device.makeCommandBuffer()
            let out = layer.decodeMulti(
                hAll, startingAt: 0, cache: StatelessLayerCache(), cmd: batchCmd, device: device)
            batchCmd.commit()
            batchCmd.waitUntilCompleted()
            #expect(batchCmd.status == .completed)

            var totalMismatches = 0
            for i in 0 ..< t {
                let refFlat = refRows[i].toFloatArray()
                let gotFlat = out.slicedRows(start: i, count: 1).reshaped(to: [hidden]).toFloatArray()
                totalMismatches += Self.compareRows(
                    "moe row \(i)", ref: refFlat, got: gotFlat, atol: 1.0, rtol: 5e-2)
            }
            #expect(
                totalMismatches == 0,
                Comment(rawValue: "\(totalMismatches)/\(t * hidden) values outside atol+rtol bound"))
        }
    }
}
