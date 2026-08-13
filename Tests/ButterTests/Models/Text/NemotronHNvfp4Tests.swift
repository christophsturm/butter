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
// Host-side codec coverage for the Nemotron-3.5-Lightning-30B-A3B-NVFP4
// port. These tests exercise the pure CPU math (E4M3 codec, weight
// repacking, global-scale folding) with synthetic data — no GPU
// checkpoint download required, unlike the full load/decode/coherence
// gate which needs the real ~17 GB checkpoint.

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("NemotronH NVFP4/FP8 mixed-precision loader helpers")
struct NemotronHNvfp4Tests {

    // ─── E4M3 (float8_e4m3fn) scalar codec ────────────────────────────

    @Test("E4M3 decode matches hand-computed reference values")
    func e4m3DecodeKnownValues() {
        // 0x00 = S0 E0000 M000 -> +0.0
        #expect(E4M3.decode(0x00) == 0.0)
        // 0x80 = S1 E0000 M000 -> -0.0
        #expect(E4M3.decode(0x80) == 0.0 || E4M3.decode(0x80) == -0.0)
        // 0x38 = 0b0_0111_000 -> sign 0, exp 7 (stored), mant 0
        //   -> 2^(7-7) * (1 + 0/8) = 1.0
        #expect(abs(E4M3.decode(0x38) - 1.0) < 1e-6)
        // 0xB8 = 0b1_0111_000 -> same magnitude, negative -> -1.0
        #expect(abs(E4M3.decode(0xB8) - (-1.0)) < 1e-6)
        // 0x40 = 0b0_1000_000 -> exp 8 (stored), mant 0 -> 2^1 * 1 = 2.0
        #expect(abs(E4M3.decode(0x40) - 2.0) < 1e-6)
        // 0x7E = 0b0_1111_110 -> exp 15 (stored), mant 6 -> max finite 448.
        #expect(abs(E4M3.decode(0x7E) - 448.0) < 1e-3)
        // 0x7F = 0b0_1111_111 -> NaN.
        #expect(E4M3.decode(0x7F).isNaN)
        #expect(E4M3.decode(0xFF).isNaN)
        // 0x01 = smallest positive subnormal = 2^-6 * (1/8) = 2^-9.
        #expect(abs(E4M3.decode(0x01) - Float(exactly: 1)! / 512.0) < 1e-9)
    }

    @Test("E4M3 encode(decode(byte)) round-trips for every representable byte")
    func e4m3RoundTripsAllBytes() {
        for raw in 0 ..< 256 {
            let byte = UInt8(raw)
            let exp = (byte >> 3) & 0x0F
            let mant = byte & 0x07
            if exp == 0x0F && mant == 0x07 { continue }  // NaN — no round trip.
            let value = E4M3.decode(byte)
            let reencoded = E4M3.encodeNearest(value)
            #expect(
                E4M3.decode(reencoded) == value,
                Comment(
                    rawValue: "byte \(String(byte, radix: 2)) -> \(value) -> re-encoded "
                        + "\(reencoded) -> \(E4M3.decode(reencoded))"))
        }
    }

    @Test("E4M3 encode saturates overflow to max finite instead of NaN")
    func e4m3EncodeSaturatesOverflow() {
        let byte = E4M3.encodeNearest(1_000_000.0)
        #expect(!E4M3.decode(byte).isNaN)
        #expect(abs(E4M3.decode(byte) - 448.0) < 1e-3)
        let negByte = E4M3.encodeNearest(-1_000_000.0)
        #expect(abs(E4M3.decode(negByte) - (-448.0)) < 1e-3)
    }

    // ─── NVFP4 weight repack (HF u8[out,in/2] -> Butter u32[out,in/8]) ─

    @Test("repackNvfp4Weight preserves bytes exactly (pure reinterpretation)")
    func repackPreservesBytes() {
        autoreleasepool {
            let device = Device.shared
            let outDim = 4
            let inDim = 32  // in/2 = 16 bytes/row, in/8 = 4 u32 lanes/row.
            var bytes = [UInt8](repeating: 0, count: outDim * (inDim / 2))
            for i in 0 ..< bytes.count { bytes[i] = UInt8((i * 37 + 5) % 256) }
            let hfWeight = Tensor.empty(shape: [outDim, inDim / 2], dtype: .u8, device: device)
            hfWeight.copyIn(from: bytes)

            let repacked = repackNvfp4Weight(hfWeight, outDim: outDim, inDim: inDim, device: device)
            #expect(repacked.dtype == .u32)
            #expect(repacked.shape == [outDim, inDim / 8])

            // Byte-for-byte identity: `Tensor.toArray(as:)` sizes its
            // read by `elementCount` in the TENSOR's own dtype (16 u32
            // lanes here, not 64 u8 bytes), so read as u32 and
            // decompose each lane back to little-endian bytes by hand
            // — matching how `repackNvfp4Weight`'s memcpy actually laid
            // them out on this (little-endian, Apple Silicon) platform.
            let repackedLanes = repacked.toArray(as: UInt32.self)
            var repackedBytes: [UInt8] = []
            repackedBytes.reserveCapacity(repackedLanes.count * 4)
            for lane in repackedLanes {
                repackedBytes.append(UInt8(lane & 0xFF))
                repackedBytes.append(UInt8((lane >> 8) & 0xFF))
                repackedBytes.append(UInt8((lane >> 16) & 0xFF))
                repackedBytes.append(UInt8((lane >> 24) & 0xFF))
            }
            #expect(repackedBytes == bytes)
        }
    }

    // ─── Global-scale folding into per-block E4M3 micro-scales ────────

    @Test("foldNvfp4GlobalIntoScales with global=1.0 is a byte-identical copy")
    func foldIdentityScale() {
        autoreleasepool {
            let device = Device.shared
            let bytes: [UInt8] = [0x38, 0x40, 0x00, 0x30, 0x2C]  // assorted E4M3 codes.
            let scaleT = Tensor.empty(shape: [bytes.count], dtype: .u8, device: device)
            scaleT.copyIn(from: bytes)
            let folded = foldNvfp4GlobalIntoScales(scaleT, global: 1.0, device: device)
            #expect(folded.toArray(as: UInt8.self) == bytes)
        }
    }

    @Test("foldNvfp4GlobalIntoScales(global: 2.0) doubles the decoded scale value")
    func foldDoublesScale() {
        autoreleasepool {
            let device = Device.shared
            // 0x38 decodes to 1.0; folding in a global of 2.0 should
            // land on (or very near) the E4M3 code for 2.0 (0x40).
            let scaleT = Tensor.empty(shape: [1], dtype: .u8, device: device)
            scaleT.copyIn(from: [UInt8(0x38)])
            let folded = foldNvfp4GlobalIntoScales(scaleT, global: 2.0, device: device)
            let decoded = E4M3.decode(folded.toArray(as: UInt8.self)[0])
            #expect(abs(decoded - 2.0) < 1e-3)
        }
    }

    // ─── FP8 -> bf16 dense dequantization (mamba in_proj/out_proj) ────

    @Test("dequantFp8ToDType reproduces scale * E4M3(byte) within bf16 rounding")
    func fp8DequantMatchesReference() {
        autoreleasepool {
            let device = Device.shared
            let outDim = 2
            let inDim = 4
            // 0x38 (=1.0), 0x40 (=2.0), 0x00 (=0.0), 0xB8 (=-1.0), ...
            let bytes: [UInt8] = [0x38, 0x40, 0x00, 0xB8, 0x38, 0x38, 0x40, 0x00]
            let f8 = Tensor.empty(shape: [outDim, inDim], dtype: .u8, device: device)
            f8.copyIn(from: bytes)
            let scale: Float = 0.5
            let out = dequantFp8ToDType(
                f8, scale: scale, outDim: outDim, inDim: inDim,
                dtype: .bf16, device: device)
            let result = out.toFloatArray()
            for i in 0 ..< bytes.count {
                let expected = E4M3.decode(bytes[i]) * scale
                #expect(
                    abs(result[i] - expected) < 5e-3,
                    Comment(rawValue: "index \(i): \(result[i]) vs \(expected)"))
            }
        }
    }
}
