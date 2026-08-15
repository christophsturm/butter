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
// Host-CPU dequant for the ggml "K-quant" family (Q4_K / Q5_K / Q6_K).
//
// Unlike `GGUFDequant`'s Q8_0 / Q2_K / IQ2_XXS paths (which split the
// on-disk block into a GPU-resident layout for an `iron_gguf_dequant_*`
// Metal kernel — DSv4 keeps those quant formats resident and inline-
// dequants them every decode step), the K-quants here are only ever
// needed ONCE per tensor at load time: the Qwen3.5/3.6 GGUF loader
// (`Qwen3xGGUFLoader.swift`) dequants straight to f32 on the CPU, then
// immediately re-quantizes to Butter's native affine int4 g64 format
// via `QuantizedOps.quantizeAffine` (the same primitive `butter convert`
// and `NemotronHNvfp4.swift`'s `requantizeDenseToInt4NH` use) — so
// there is no decode-time cost to paying for the dequant on the CPU
// once at load, and it avoids writing three new `#[kernel]` Metal
// dequant kernels purely to serve a one-shot load-time conversion.
//
// Block layouts (from the canonical ggml-quants.c reference, verified
// byte-for-byte against `gguf-py`'s `Q4_K` / `Q5_K` / `Q6_K` dequantize
// reference on real blocks from a Qwen3.8-27B-Q4_K_M checkpoint — see
// `Tests/ButterTests/Loader/GGUFDequantKTests.swift`):
//
//   Q4_K (144 B / 256 values):
//     [f16 d; f16 dmin; u8 scales[12]; u8 qs[128]]
//     8 sub-blocks of 32 values, each with its own 6-bit scale + 6-bit
//     min packed two-per-byte-pair across the 12-byte `scales` array
//     (`getScaleMinK4`). `qs` nibble-packs two sub-blocks (low nibble /
//     high nibble) per 32-byte half.
//
//   Q5_K (176 B / 256 values):
//     [f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 ql[128]]
//     Same two-level scale scheme as Q4_K, plus one extra high bit per
//     value from `qh` (bit position rotates by sub-block: `u1`/`u2`
//     shift left 2 every 64-value macro-block).
//
//   Q6_K (210 B / 256 values):
//     [u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d]
//     6-bit values (4 low bits from `ql`, 2 high bits from `qh`),
//     centered at 32, one i8 scale per 16-value sub-block, single f16
//     super-block scale (no separate min — Q6_K is symmetric, unlike
//     Q4_K/Q5_K's asymmetric min-max scheme).

import Foundation

extension GGUFDequant {
    // ─── Shared two-level scale/min unpack (Q4_K / Q5_K) ───────────────

    /// `get_scale_min_k4` — unpack the 6-bit scale + 6-bit min for
    /// sub-block `j` (0..<8) from the 12-byte packed `scales` array.
    @inline(__always)
    static func scaleMinK4(_ j: Int, _ q: UnsafePointer<UInt8>) -> (UInt8, UInt8) {
        if j < 4 {
            return (q[j] & 63, q[j + 4] & 63)
        } else {
            let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
            let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
            return (d, m)
        }
    }

    // ─── Q4_K ────────────────────────────────────────────────────────

    public static let q4_KBlockBytes = 144
    public static let q4_KBlockValues = 256

    /// Dequant one Q4_K block (144 bytes → 256 f32 values) into `out`
    /// starting at `outOffset`. Matches ggml's `dequantize_row_q4_K`.
    @inline(__always)
    static func dequantQ4_KBlock(
        _ block: UnsafePointer<UInt8>, into out: UnsafeMutablePointer<Float>, outOffset: Int
    ) {
        let dBits = UInt16(block[0]) | (UInt16(block[1]) << 8)
        let dminBits = UInt16(block[2]) | (UInt16(block[3]) << 8)
        let d = Float(Float16(bitPattern: dBits))
        let dmin = Float(Float16(bitPattern: dminBits))
        let scales = block + 4
        let qs = block + 16
        var isIdx = 0
        var qOff = 0
        var yOff = outOffset
        var j = 0
        while j < 256 {
            let (sc1, m1raw) = scaleMinK4(isIdx + 0, scales)
            let d1 = d * Float(sc1)
            let m1 = dmin * Float(m1raw)
            let (sc2, m2raw) = scaleMinK4(isIdx + 1, scales)
            let d2 = d * Float(sc2)
            let m2 = dmin * Float(m2raw)
            for l in 0..<32 {
                out[yOff + l] = d1 * Float(qs[qOff + l] & 0xF) - m1
            }
            for l in 0..<32 {
                out[yOff + 32 + l] = d2 * Float(qs[qOff + l] >> 4) - m2
            }
            yOff += 64
            qOff += 32
            isIdx += 2
            j += 64
        }
    }

    /// Dequant a whole Q4_K tensor (flat, `nValues` elements) into a
    /// host `Tensor` (f32). Parallelized across blocks; the caller
    /// (`Qwen3xGGUFLoader`) is expected to immediately re-quantize the
    /// result to int4 and drop this f32 intermediate — wrap the call in
    /// `autoreleasepool` at the call site to bound peak RSS.
    static func dequantQ4_K_cpu(
        rawBlocks: UnsafeBufferPointer<UInt8>, nValues: Int
    ) -> [Float] {
        precondition(nValues % q4_KBlockValues == 0, "dequantQ4_K_cpu: nValues must be a multiple of 256")
        let nBlocks = nValues / q4_KBlockValues
        precondition(
            rawBlocks.count >= nBlocks * q4_KBlockBytes,
            "dequantQ4_K_cpu: rawBlocks too short for \(nBlocks) blocks")
        var out = [Float](repeating: 0, count: nValues)
        out.withUnsafeMutableBufferPointer { outPtr in
            let outBase = outPtr.baseAddress!
            let rawBase = rawBlocks.baseAddress!
            let chunks = min(nBlocks, 32)
            let blocksPerChunk = (nBlocks + chunks - 1) / chunks
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                let start = c * blocksPerChunk
                let end = min(start + blocksPerChunk, nBlocks)
                guard start < end else { return }
                for b in start..<end {
                    let blockPtr = rawBase + b * q4_KBlockBytes
                    dequantQ4_KBlock(blockPtr, into: outBase, outOffset: b * q4_KBlockValues)
                }
            }
        }
        return out
    }

    // ─── Q5_K ────────────────────────────────────────────────────────

    public static let q5_KBlockBytes = 176
    public static let q5_KBlockValues = 256

    /// Dequant one Q5_K block (176 bytes → 256 f32 values). Matches
    /// ggml's `dequantize_row_q5_K`.
    @inline(__always)
    static func dequantQ5_KBlock(
        _ block: UnsafePointer<UInt8>, into out: UnsafeMutablePointer<Float>, outOffset: Int
    ) {
        let dBits = UInt16(block[0]) | (UInt16(block[1]) << 8)
        let dminBits = UInt16(block[2]) | (UInt16(block[3]) << 8)
        let d = Float(Float16(bitPattern: dBits))
        let dmin = Float(Float16(bitPattern: dminBits))
        let scales = block + 4
        let qh = block + 16
        let ql = block + 48
        var isIdx = 0
        var qlOff = 0
        var yOff = outOffset
        var u1: UInt8 = 1
        var u2: UInt8 = 2
        var j = 0
        while j < 256 {
            let (sc1, m1raw) = scaleMinK4(isIdx + 0, scales)
            let d1 = d * Float(sc1)
            let m1 = dmin * Float(m1raw)
            let (sc2, m2raw) = scaleMinK4(isIdx + 1, scales)
            let d2 = d * Float(sc2)
            let m2 = dmin * Float(m2raw)
            for l in 0..<32 {
                let hi: Float = (qh[l] & u1) != 0 ? 16 : 0
                out[yOff + l] = d1 * (Float(ql[qlOff + l] & 0xF) + hi) - m1
            }
            for l in 0..<32 {
                let hi: Float = (qh[l] & u2) != 0 ? 16 : 0
                out[yOff + 32 + l] = d2 * (Float(ql[qlOff + l] >> 4) + hi) - m2
            }
            yOff += 64
            qlOff += 32
            isIdx += 2
            u1 <<= 2
            u2 <<= 2
            j += 64
        }
    }

    static func dequantQ5_K_cpu(
        rawBlocks: UnsafeBufferPointer<UInt8>, nValues: Int
    ) -> [Float] {
        precondition(nValues % q5_KBlockValues == 0, "dequantQ5_K_cpu: nValues must be a multiple of 256")
        let nBlocks = nValues / q5_KBlockValues
        precondition(
            rawBlocks.count >= nBlocks * q5_KBlockBytes,
            "dequantQ5_K_cpu: rawBlocks too short for \(nBlocks) blocks")
        var out = [Float](repeating: 0, count: nValues)
        out.withUnsafeMutableBufferPointer { outPtr in
            let outBase = outPtr.baseAddress!
            let rawBase = rawBlocks.baseAddress!
            let chunks = min(nBlocks, 32)
            let blocksPerChunk = (nBlocks + chunks - 1) / chunks
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                let start = c * blocksPerChunk
                let end = min(start + blocksPerChunk, nBlocks)
                guard start < end else { return }
                for b in start..<end {
                    let blockPtr = rawBase + b * q5_KBlockBytes
                    dequantQ5_KBlock(blockPtr, into: outBase, outOffset: b * q5_KBlockValues)
                }
            }
        }
        return out
    }

    // ─── Q6_K ────────────────────────────────────────────────────────

    public static let q6_KBlockBytes = 210
    public static let q6_KBlockValues = 256

    /// Dequant one Q6_K block (210 bytes → 256 f32 values). Matches
    /// ggml's `dequantize_row_q6_K`. Symmetric (no min) — one i8 scale
    /// per 16-value sub-block, single f16 super-block scale.
    @inline(__always)
    static func dequantQ6_KBlock(
        _ block: UnsafePointer<UInt8>, into out: UnsafeMutablePointer<Float>, outOffset: Int
    ) {
        let ql = block
        let qh = block + 128
        let sc = UnsafeRawPointer(block + 192).assumingMemoryBound(to: Int8.self)
        let dBits = UInt16(block[208]) | (UInt16(block[209]) << 8)
        let d = Float(Float16(bitPattern: dBits))
        var n = 0
        var yOff = outOffset
        var qlOff = 0
        var qhOff = 0
        var scOff = 0
        while n < 256 {
            for l in 0..<32 {
                let isIdx = l / 16
                let q1 = Int32((ql[qlOff + l] & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32
                let q2 = Int32((ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32
                let q3 = Int32((ql[qlOff + l] >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32
                let q4 = Int32((ql[qlOff + l + 32] >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32
                out[yOff + l] = d * Float(sc[scOff + isIdx + 0]) * Float(q1)
                out[yOff + 32 + l] = d * Float(sc[scOff + isIdx + 2]) * Float(q2)
                out[yOff + 64 + l] = d * Float(sc[scOff + isIdx + 4]) * Float(q3)
                out[yOff + 96 + l] = d * Float(sc[scOff + isIdx + 6]) * Float(q4)
            }
            yOff += 128
            qlOff += 64
            qhOff += 32
            scOff += 8
            n += 128
        }
    }

    static func dequantQ6_K_cpu(
        rawBlocks: UnsafeBufferPointer<UInt8>, nValues: Int
    ) -> [Float] {
        precondition(nValues % q6_KBlockValues == 0, "dequantQ6_K_cpu: nValues must be a multiple of 256")
        let nBlocks = nValues / q6_KBlockValues
        precondition(
            rawBlocks.count >= nBlocks * q6_KBlockBytes,
            "dequantQ6_K_cpu: rawBlocks too short for \(nBlocks) blocks")
        var out = [Float](repeating: 0, count: nValues)
        out.withUnsafeMutableBufferPointer { outPtr in
            let outBase = outPtr.baseAddress!
            let rawBase = rawBlocks.baseAddress!
            let chunks = min(nBlocks, 32)
            let blocksPerChunk = (nBlocks + chunks - 1) / chunks
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                let start = c * blocksPerChunk
                let end = min(start + blocksPerChunk, nBlocks)
                guard start < end else { return }
                for b in start..<end {
                    let blockPtr = rawBase + b * q6_KBlockBytes
                    dequantQ6_KBlock(blockPtr, into: outBase, outOffset: b * q6_KBlockValues)
                }
            }
        }
        return out
    }
}
