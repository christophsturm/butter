// Copyright 2026 Eric Kryski (@ekryski)
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
// Regression test for the Qwen3.5/3.6 dense batched-prefill degenerate
// output (token-0 "!" loops): the batched quantized-embedding gather
// (`Ops.dequantGather` with nTokens > 1) at the production shape
// (bf16 scales, hidden 2560, groupSize 64) must match the CPU
// `q * scale + bias` reference for EVERY gathered row, not just row 0.

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("DequantGather batched (Qwen3x prefill shape)")
struct DequantGatherBatchedTests {

    static func pack8(_ nibbles: [UInt32]) -> UInt32 {
        precondition(nibbles.count == 8)
        var w: UInt32 = 0
        for (i, v) in nibbles.enumerated() { w |= (v & 0xF) << (UInt32(i) * 4) }
        return w
    }

    static func bf16Bits(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(rounded >> 16)
    }

    /// Run one (nTokens, hidden, groupSize) case in bf16 and compare
    /// against the CPU dequant reference.
    private func runCase(nTokens: Int, hidden: Int, groupSize: Int, vocab: Int) {
        autoreleasepool {
            let nGroups = hidden / groupSize

            var rng = SystemRandomNumberGenerator()
            var q = [[UInt32]](repeating: [], count: vocab)
            for r in 0 ..< vocab {
                q[r] = (0 ..< hidden).map { _ in UInt32.random(in: 0 ..< 16, using: &rng) }
            }
            let scales: [Float] = (0 ..< (vocab * nGroups)).map { _ in
                Float.random(in: 0.01 ..< 0.1, using: &rng)
            }
            let biases: [Float] = (0 ..< (vocab * nGroups)).map { _ in
                Float.random(in: -0.5 ..< 0.5, using: &rng)
            }

            var packed: [UInt32] = []
            packed.reserveCapacity(vocab * hidden / 8)
            for r in 0 ..< vocab {
                for i in stride(from: 0, to: hidden, by: 8) {
                    packed.append(Self.pack8(Array(q[r][i ..< i + 8])))
                }
            }

            let tokens: [UInt32] = (0 ..< nTokens).map { _ in
                UInt32.random(in: 0 ..< UInt32(vocab), using: &rng)
            }

            let weight = Tensor.empty(shape: [vocab, hidden / 8], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(shape: [vocab, nGroups], dtype: .bf16)
            scalesT.copyIn(from: scales.map(Self.bf16Bits))
            let biasesT = Tensor.empty(shape: [vocab, nGroups], dtype: .bf16)
            biasesT.copyIn(from: biases.map(Self.bf16Bits))
            let idsT = Tensor.empty(shape: [nTokens], dtype: .u32)
            idsT.copyIn(from: tokens)

            var out: Tensor!
            let cb = Device.shared.makeCommandBuffer()
            out = Ops.dequantGather(
                weight: weight, scales: scalesT, biases: biasesT,
                tokenIds: idsT, hidden: hidden, bits: 4, groupSize: groupSize,
                on: cb)
            cb.commit()
            cb.waitUntilCompleted()

            let got = out.toFloatArray()
            var mismatches = 0
            var l2 = 0.0
            for t in 0 ..< nTokens {
                let row = Int(tokens[t])
                for d in 0 ..< hidden {
                    let g = d / groupSize
                    // bf16-round the reference operands to match the GPU.
                    let s = Float(
                        bitPattern: UInt32(Self.bf16Bits(scales[row * nGroups + g])) << 16)
                    let b = Float(
                        bitPattern: UInt32(Self.bf16Bits(biases[row * nGroups + g])) << 16)
                    let expected = Float(q[row][d]) * s + b
                    let gotV = got[t * hidden + d]
                    l2 += Double(gotV) * Double(gotV)
                    // bf16 output rounding: allow ~1% of magnitude + abs floor.
                    if abs(gotV - expected) > max(0.02, abs(expected) * 0.02) {
                        mismatches += 1
                    }
                }
            }
            #expect(
                mismatches == 0,
                "nTokens=\(nTokens) hidden=\(hidden) gs=\(groupSize): \(mismatches) mismatched elements (out l2=\(l2.squareRoot()))"
            )
        }
    }

    @Test("bf16 single-token gather matches CPU (decode/prewarm shape)")
    func singleTokenMatches() {
        runCase(nTokens: 1, hidden: 2560, groupSize: 64, vocab: 512)
    }

    @Test("bf16 batched gather matches CPU (prefill shape, nTokens=5)")
    func batchedFiveMatches() {
        runCase(nTokens: 5, hidden: 2560, groupSize: 64, vocab: 512)
    }

    @Test("bf16 batched gather matches CPU (nTokens=32)")
    func batchedThirtyTwoMatches() {
        runCase(nTokens: 32, hidden: 2560, groupSize: 64, vocab: 512)
    }
}
