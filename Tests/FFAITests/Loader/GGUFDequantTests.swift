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
// Unit tests for GGUFDequant — block-format constants + a deterministic
// Q8_0 round-trip (build a known block → dequant → verify d * qs). The
// Q2_K / IQ2_XXS bit-layout dequant is exercised exhaustively against a
// canonical oracle in metaltile's kernel correctness tests; here we
// verify the FFAI-side block constants and the Q8_0 CPU-split → GPU
// pipeline end to end.

import Foundation
import Metal
import Testing

@testable import FFAI

@Suite("GGUFDequant")
struct GGUFDequantTests {

    @Test("Block-format constants match the GGUF spec")
    func blockConstants() {
        // Q8_0: 2-byte fp16 scale + 32 × int8 = 34 B / 32 values.
        #expect(GGUFDequant.q8_0BlockBytes == 34)
        #expect(GGUFDequant.q8_0BlockValues == 32)
        // Q2_K: scales[16] + qs[64] + fp16 d + fp16 dmin = 84 B / 256.
        #expect(GGUFDequant.q2_KBlockBytes == 84)
        #expect(GGUFDequant.q2_KBlockValues == 256)
        // IQ2_XXS: 66 B / 256.
        #expect(GGUFDequant.iq2_xxsBlockBytes == 66)
        #expect(GGUFDequant.iq2_xxsBlockValues == 256)
    }

    @Test("Q8_0 round-trip: out[i] == d * qs[i]")
    func q8_0RoundTrip() {
        let device = Device.shared
        // One block: scale d = 0.5 (exact in fp16), quants -16…+15.
        let d: Float16 = 0.5
        let qs: [Int8] = (0..<32).map { Int8($0 - 16) }

        var block = Data()
        withUnsafeBytes(of: d.bitPattern.littleEndian) { block.append(contentsOf: $0) }
        for q in qs { block.append(UInt8(bitPattern: q)) }
        #expect(block.count == GGUFDequant.q8_0BlockBytes)

        let cmd = device.makeCommandBuffer()
        let out = GGUFDequant.dequantQ8_0(
            rawBlocks: block, nValues: 32, outDtype: .f32, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()

        let host = out.toFloatArray()
        #expect(host.count == 32)
        for i in 0..<32 {
            let expected = Float(d) * Float(qs[i])
            #expect(abs(host[i] - expected) < 1e-3, "out[\(i)]=\(host[i]) expected \(expected)")
        }
    }
}
