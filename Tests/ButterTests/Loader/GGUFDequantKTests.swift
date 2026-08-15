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
// Unit tests for `GGUFDequantK` — the host-CPU Q4_K / Q5_K / Q6_K
// dequant used by `Qwen3xGGUFLoader`.
//
// Two kinds of coverage:
//
//  1. Real-checkpoint regression — the exact first block of
//     `blk.0.attn_gate.weight` (Q4_K) / `blk.0.ssm_out.weight` (Q5_K) /
//     `blk.0.attn_qkv.weight` (Q6_K) from a real
//     `Qwen3.8-27B-Q4_K_M.gguf`, dumped as raw hex + cross-checked
//     value-by-value against Python's `gguf.quants.{Q4_K,Q5_K,Q6_K}
//     .dequantize` reference implementation (max abs error ~6e-9 —
//     float32 rounding noise, not an algorithm difference). This is
//     the same check that caught the block-layout logic being correct
//     before it was ever wired into the real loader.
//
//  2. Synthetic hand-verified blocks — degenerate scale/min patterns
//     chosen so the expected output is computable by hand (not just
//     "whatever the reference implementation says"), covering the
//     bit-unpacking machinery (nibble low/high split, the two-level
//     6-bit scale/min pack for Q4_K/Q5_K, the 6-bit ql/qh reconstruction
//     for Q6_K) independent of any external oracle.

import Foundation
import Testing

@testable import ButterSwift

@Suite("GGUFDequantK")
struct GGUFDequantKTests {

    @Test("Block-format constants match the GGUF spec")
    func blockConstants() {
        #expect(GGUFDequant.q4_KBlockBytes == 144)
        #expect(GGUFDequant.q4_KBlockValues == 256)
        #expect(GGUFDequant.q5_KBlockBytes == 176)
        #expect(GGUFDequant.q5_KBlockValues == 256)
        #expect(GGUFDequant.q6_KBlockBytes == 210)
        #expect(GGUFDequant.q6_KBlockValues == 256)
    }

    // ─── Real-checkpoint regression (see file header) ──────────────────

    @Test("Q4_K matches gguf-py reference on a real block (blk.0.attn_gate.weight)")
    func q4_KRealBlockMatchesReference() {
        let bytes = hexToBytes(
            "6d058811fff5f7f2eaf3f2e9f9f827848777946779e743868aa66455c69008a57630a59283b6759565975605267b4f995c87a929b9762886b8876b27c396c05b85a63ad738fc457a1a0e6743a7254687dccc41268604aac7d684a01bae87aaa8cc5f9bcb69c7ab65aaabf15d57cb45b6d0976c0c956ac3b4888248cf59c742589c6ed6fb71874074383a9877366af16d"
        )
        #expect(bytes.count == GGUFDequant.q4_KBlockBytes)
        let out = bytes.withUnsafeBufferPointer { raw in
            GGUFDequant.dequantQ4_K_cpu(rawBlocks: raw, nValues: 256)
        }
        #expect(out.count == 256)
        #expect(abs(out.reduce(0, +) - (-0.38275186)) < 1e-4, "block sum mismatch: \(out.reduce(0, +))")
        let expectedHead: [Float] = [
            0.00815231, 0.00815231, -0.00749516, 0.00815231, 0.01858395,
            0.00815231, -0.01271099, 0.00293648, 0.02379978, 0.00293648,
        ]
        for i in 0..<10 {
            #expect(abs(out[i] - expectedHead[i]) < 1e-4, "out[\(i)]=\(out[i]) expected \(expectedHead[i])")
        }
        let expectedTail: [Float] = [
            0.00093484, -0.00767541, -0.02489591, -0.01198053, 0.02676558, -0.01198053,
        ]
        for (j, i) in (250..<256).enumerated() {
            #expect(
                abs(out[i] - expectedTail[j]) < 1e-4, "out[\(i)]=\(out[i]) expected \(expectedTail[j])")
        }
    }

    @Test("Q5_K matches gguf-py reference on a real block (blk.0.ssm_out.weight)")
    func q5_KRealBlockMatchesReference() {
        let bytes = hexToBytes(
            "b502f611ffa7f1b4dfa0e96ec17ffef57670b8460ab23888da78c40e321cda121cd4f6593c54f658dec6205275717658b519823aabe7b73b0602f70cb92eec18dea40912ad13f9c5bcb70c603e8024459cb00f9ef5ab1a367ac6d6078a242e1f6980f92675e7b04d8100a93ee0dc90691552b4ed8a3f99deec6aceea73b1f4cef1e46ff64405f1e68eb0c3b11f2e1604d0023943dfabac9ff3eb53dd8c4f94ec7b868df660af015362c69f3497c3a0d5"
        )
        #expect(bytes.count == GGUFDequant.q5_KBlockBytes)
        let out = bytes.withUnsafeBufferPointer { raw in
            GGUFDequant.dequantQ5_K_cpu(rawBlocks: raw, nValues: 256)
        }
        #expect(out.count == 256)
        #expect(abs(out.reduce(0, +) - (-0.11392148)) < 1e-4, "block sum mismatch: \(out.reduce(0, +))")
        let expectedHead: [Float] = [
            -0.00954586, 0.00086325, -0.0173527, 0.00346553, 0.00606781,
            -0.0043413, -0.0043413, 0.00606781, -0.00694358, -0.0173527,
        ]
        for i in 0..<10 {
            #expect(abs(out[i] - expectedHead[i]) < 1e-4, "out[\(i)]=\(out[i]) expected \(expectedHead[i])")
        }
    }

    @Test("Q6_K matches gguf-py reference on a real block (blk.0.attn_qkv.weight)")
    func q6_KRealBlockMatchesReference() {
        let bytes = hexToBytes(
            "f3de257b07b793997fc71dc300344f77cf30650de15986c4a504dd3d501a1bf1523f05650554e6294b01430c4d86f41850febb1d51c6d9d53108195c3fde549df851904fcd075830a900d01abdc9965699f4078c10382cb18a5444812b9482d76fa431da6780129a00e4657c475e7da10bf53d1d5269acf681eb1776741e785383605e1c85d92fd975a2d3691cbb6a39f2f5eb0a8cf6a0a7bdf290fa0af8a51b765b937da507a0aa2867d0729555a5e9e25d87a56ea6a112526626549a5a5e699c5ab6c03dafacc05a9967477480a25bba00"
        )
        #expect(bytes.count == GGUFDequant.q6_KBlockBytes)
        let out = bytes.withUnsafeBufferPointer { raw in
            GGUFDequant.dequantQ6_K_cpu(rawBlocks: raw, nValues: 256)
        }
        #expect(out.count == 256)
        #expect(abs(out.reduce(0, +) - (-0.24817046)) < 1e-4, "block sum mismatch: \(out.reduce(0, +))")
        let expectedHead: [Float] = [
            -0.02106428, 0.01995564, -0.00554323, 0.02328157, 0.00997782,
            0.00997782, -0.02106428, 0.00776052, 0.00110865, -0.00776052,
        ]
        for i in 0..<10 {
            #expect(abs(out[i] - expectedHead[i]) < 1e-4, "out[\(i)]=\(out[i]) expected \(expectedHead[i])")
        }
    }

    // ─── Synthetic hand-verified blocks ─────────────────────────────────

    @Test("Q4_K synthetic block: degenerate scale=1/min=0 reduces to raw nibble value")
    func q4_KSyntheticDegenerate() {
        // d=1.0, dmin=0.0 (exact in fp16).
        var block = [UInt8](repeating: 0, count: GGUFDequant.q4_KBlockBytes)
        writeF16(1.0, into: &block, at: 0)  // d
        writeF16(0.0, into: &block, at: 2)  // dmin
        // scales[12] chosen (see Qwen3xGGUFLoader.swift file header /
        // GGUFDequantK.swift `scaleMinK4`) so every sub-block j=0..<8
        // decodes to (scale=1, min=0): bytes 0..3 = 1 (low-nibble
        // scales for j<4), bytes 4..7 = 0 (low-nibble mins for j<4),
        // bytes 8..11 = 1 (feeds both the high-nibble scale AND min
        // for j=4..7, since q[0..3]'s top 2 bits are 0).
        let scales: [UInt8] = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        for (i, b) in scales.enumerated() { block[4 + i] = b }
        // qs[128]: every byte = 0xAB → low nibble 0xB=11, high nibble 0xA=10.
        for i in 0..<128 { block[16 + i] = 0xAB }

        let out = block.withUnsafeBufferPointer { raw in
            GGUFDequant.dequantQ4_K_cpu(rawBlocks: raw, nValues: 256)
        }
        // Each 64-value macro-block is [32 low-nibble values][32
        // high-nibble values] — see `dequantQ4_KBlock`.
        for macro in 0..<4 {
            let base = macro * 64
            for l in 0..<32 {
                #expect(out[base + l] == 11.0, "low-nibble out[\(base + l)]=\(out[base + l])")
            }
            for l in 0..<32 {
                #expect(out[base + 32 + l] == 10.0, "high-nibble out[\(base + 32 + l)]=\(out[base + 32 + l])")
            }
        }
    }

    @Test("Q5_K synthetic block: qh all-zero reduces to the Q4_K nibble formula")
    func q5_KSyntheticDegenerate() {
        var block = [UInt8](repeating: 0, count: GGUFDequant.q5_KBlockBytes)
        writeF16(1.0, into: &block, at: 0)  // d
        writeF16(0.0, into: &block, at: 2)  // dmin
        let scales: [UInt8] = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        for (i, b) in scales.enumerated() { block[4 + i] = b }
        // qh[32] left zero — the extra high bit never fires, so this
        // degenerates to exactly the Q4_K nibble formula above.
        // ql[128]: every byte = 0xAB.
        for i in 0..<128 { block[48 + i] = 0xAB }

        let out = block.withUnsafeBufferPointer { raw in
            GGUFDequant.dequantQ5_K_cpu(rawBlocks: raw, nValues: 256)
        }
        for macro in 0..<4 {
            let base = macro * 64
            for l in 0..<32 {
                #expect(out[base + l] == 11.0, "low-nibble out[\(base + l)]=\(out[base + l])")
            }
            for l in 0..<32 {
                #expect(out[base + 32 + l] == 10.0, "high-nibble out[\(base + 32 + l)]=\(out[base + 32 + l])")
            }
        }
    }

    @Test("Q5_K synthetic block: qh high bit adds exactly 16 to the nibble value")
    func q5_KSyntheticHighBit() {
        var block = [UInt8](repeating: 0, count: GGUFDequant.q5_KBlockBytes)
        writeF16(1.0, into: &block, at: 0)
        writeF16(0.0, into: &block, at: 2)
        let scales: [UInt8] = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        for (i, b) in scales.enumerated() { block[4 + i] = b }
        // qh all-ones: every low-nibble slot's high bit (u1, bit 0 of
        // qh initially) is set for macro-block 0, so its value should
        // be 11 + 16 = 27 there (u1/u2 rotate by 2 bits per
        // macro-block — only macro 0 is asserted here for a single
        // unambiguous check).
        for i in 0..<32 { block[16 + i] = 0xFF }  // qh
        for i in 0..<128 { block[48 + i] = 0xAB }  // ql

        let out = block.withUnsafeBufferPointer { raw in
            GGUFDequant.dequantQ5_K_cpu(rawBlocks: raw, nValues: 256)
        }
        // Macro-block 0: u1 = 1 (bit 0 set in qh=0xFF) → low nibble + 16.
        //                u2 = 2 (bit 1 set in qh=0xFF) → high nibble + 16.
        for l in 0..<32 {
            #expect(out[l] == 27.0, "out[\(l)]=\(out[l]) expected 27.0 (11 + 16)")
        }
        for l in 0..<32 {
            #expect(out[32 + l] == 26.0, "out[\(32 + l)]=\(out[32 + l]) expected 26.0 (10 + 16)")
        }
    }

    @Test("Q6_K synthetic block: scale=1, ql/qh zero reduces to the (0 - 32) center point")
    func q6_KSyntheticDegenerate() {
        var block = [UInt8](repeating: 0, count: GGUFDequant.q6_KBlockBytes)
        // ql[128] / qh[64] already zero from the repeating-0 init.
        for i in 0..<16 { block[192 + i] = 1 }  // scales[16] (i8) = 1
        writeF16(1.0, into: &block, at: 208)  // d

        let out = block.withUnsafeBufferPointer { raw in
            GGUFDequant.dequantQ6_K_cpu(rawBlocks: raw, nValues: 256)
        }
        // q = (0 | (0<<4)) - 32 = -32 for every one of the 4 packed
        // lanes; d=1, sc=1 → out = -32 everywhere.
        for i in 0..<256 {
            #expect(out[i] == -32.0, "out[\(i)]=\(out[i]) expected -32.0")
        }
    }

    @Test("Q6_K synthetic block: non-zero ql/qh/scale reconstructs the expected 6-bit value")
    func q6_KSyntheticNonzero() {
        var block = [UInt8](repeating: 0, count: GGUFDequant.q6_KBlockBytes)
        // ql[l] = 0x0F for l in [0, 32) → low nibble 0xF=15 feeds both
        // q1 (l<32, low nibble) and q3 (l<32, high nibble of ql[l]=0,
        // since 0x0F>>4 = 0).
        for l in 0..<32 { block[l] = 0x0F }
        // qh all zero → no high-2-bit contribution.
        for i in 0..<16 { block[192 + i] = 2 }  // scales[16] = 2
        writeF16(1.0, into: &block, at: 208)  // d

        let out = block.withUnsafeBufferPointer { raw in
            GGUFDequant.dequantQ6_K_cpu(rawBlocks: raw, nValues: 256)
        }
        // q1 = (0xF | 0) - 32 = -17; out[l] = d * sc[is=0] * q1 = 1*2*(-17) = -34.
        for l in 0..<32 {
            #expect(out[l] == -34.0, "out[\(l)]=\(out[l]) expected -34.0")
        }
        // q3 = (0xF>>4 | 0) - 32 = (0 | 0) - 32 = -32; out[64+l] = 1*2*(-32) = -64.
        for l in 0..<32 {
            #expect(out[64 + l] == -64.0, "out[\(64 + l)]=\(out[64 + l]) expected -64.0")
        }
    }
}

// ─── Test helpers ────────────────────────────────────────────────────

private func hexToBytes(_ s: String) -> [UInt8] {
    var bytes = [UInt8]()
    bytes.reserveCapacity(s.count / 2)
    var idx = s.startIndex
    while idx < s.endIndex {
        let next = s.index(idx, offsetBy: 2)
        bytes.append(UInt8(s[idx..<next], radix: 16)!)
        idx = next
    }
    return bytes
}

/// Write an f16-encoded `Float` into `block` at byte offset `at` (2
/// bytes, little-endian) — mirrors how `d`/`dmin` are read back by
/// `GGUFDequant.dequantQ4_KBlock` et al.
private func writeF16(_ value: Float, into block: inout [UInt8], at offset: Int) {
    let bits = Float16(value).bitPattern
    block[offset] = UInt8(bits & 0xFF)
    block[offset + 1] = UInt8((bits >> 8) & 0xFF)
}
