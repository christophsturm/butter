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
// GGUF on-disk → GPU-resident dequant pipeline. For each supported
// quant format, splits the packed on-disk blocks into the GPU-resident
// tensor layout the metaltile dequant kernel expects, then dispatches
// the kernel and returns a host-readable `Tensor`.
//
// The CPU split is a one-pass scan over the raw GGUF bytes — fp16
// scales are converted to f32 by host code so the kernel doesn't have
// to bit-cast inside the DSL.

import Foundation
import QuartzCore
import Metal

public enum GGUFDequant {
    // ─── Q8_0 ──────────────────────────────────────────────────────────

    /// Block: `[fp16 d (2 B); int8 qs[32] (32 B)]` = 34 B per 32 values.
    static let q8_0BlockBytes = 34
    static let q8_0BlockValues = 32

    /// Split each on-disk Q8_0 block into the GPU-resident tensors the
    /// kernel expects: a contiguous `[n_blocks * 32]` byte buffer of
    /// int8 quants (kernel sign-reconstructs via `select`) and a
    /// `[n_blocks]` f32 buffer of fp16-converted block super-scales.
    static func dequantQ8_0(
        rawBlocks: Data, nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor? = nil, slot: String = "default"
    ) -> Tensor {
        precondition(nValues % q8_0BlockValues == 0)
        let nBlocks = nValues / q8_0BlockValues
        precondition(
            rawBlocks.count >= nBlocks * q8_0BlockBytes,
            "GGUFDequant.Q8_0: rawBlocks too short for \(nBlocks) blocks")

        // Write the CPU splits DIRECTLY into the intermediate
        // MTLBuffers (skip per-call Swift arrays).
        let qsBytesCount = nBlocks * q8_0BlockValues
        let scBytes = nBlocks * 4
        let qsBuf = device.intermediateScratch(tag: "gguf_dequant_u8_\(slot)", minBytes: qsBytesCount)
        let scBuf = device.intermediateScratch(tag: "gguf_dequant_f32_\(slot)", minBytes: scBytes)
        let qsPtr = qsBuf.contents().assumingMemoryBound(to: UInt8.self)
        let scPtr = scBuf.contents().assumingMemoryBound(to: Float.self)
        rawBlocks.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!
            for b in 0..<nBlocks {
                let blockBase = base.advanced(by: b * q8_0BlockBytes)
                let dBits = UInt16(blockBase[0]) | (UInt16(blockBase[1]) << 8)
                scPtr[b] = Float(Float16(bitPattern: dBits))
                qsPtr.advanced(by: b * q8_0BlockValues).update(
                    from: blockBase.advanced(by: 2), count: q8_0BlockValues)
            }
        }
        let qsTensor = Tensor(buffer: qsBuf, offset: 0, shape: [qsBytesCount], dtype: .u8)
        let scalesTensor = Tensor(buffer: scBuf, offset: 0, shape: [nBlocks], dtype: .f32)
        return Ops.ggufDequantQ8_0(
            qsSigned: qsTensor, scales: scalesTensor,
            nValues: nValues, outDtype: outDtype,
            on: cmd, into: out)
    }

    // ─── Q2_K ──────────────────────────────────────────────────────────

    /// Block: `[u8 scales[16]; u8 qs[64]; fp16 d; fp16 dmin]` = 84 B
    /// per 256 values.
    public static let q2_KBlockBytes = 84
    public static let q2_KBlockValues = 256

    static func dequantQ2_K(
        rawBlocks: Data, nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor? = nil, slot: String = "default"
    ) -> Tensor {
        precondition(nValues % q2_KBlockValues == 0)
        let nBlocks = nValues / q2_KBlockValues
        precondition(
            rawBlocks.count >= nBlocks * q2_KBlockBytes,
            "GGUFDequant.Q2_K: rawBlocks too short for \(nBlocks) blocks")

        // Write the 4 CPU splits DIRECTLY into the intermediate
        // MTLBuffers (skip the per-call 524 MB / 128 MB / 32 KB Swift
        // arrays that wasn't getting returned to the OS).
        let qsBytes = nBlocks * 16 * 4
        let scBytes = nBlocks * 16
        let dBytes = nBlocks * 4
        // Q2_K uses BOTH a u32 + a u8 + two f32 buffers; tag them
        // separately so the u32 / u8 / f32 slabs from IQ2_XXS /
        // Q8_0 (same tags) are reused too.
        let qsBuf = device.intermediateScratch(tag: "gguf_dequant_u32_\(slot)", minBytes: qsBytes)
        let scBuf = device.intermediateScratch(tag: "gguf_dequant_u8_\(slot)", minBytes: scBytes)
        // Q2_K needs TWO f32 buffers (d + dmin) — different tag for
        // the second so they don't overwrite each other.
        let dBuf = device.intermediateScratch(tag: "gguf_dequant_f32_\(slot)", minBytes: dBytes)
        let dminBuf = device.intermediateScratch(tag: "gguf_dequant_f32_dmin_\(slot)", minBytes: dBytes)
        let qsPtr = qsBuf.contents().assumingMemoryBound(to: UInt32.self)
        let scPtr = scBuf.contents().assumingMemoryBound(to: UInt8.self)
        let dPtr = dBuf.contents().assumingMemoryBound(to: Float.self)
        let dminPtr = dminBuf.contents().assumingMemoryBound(to: Float.self)
        rawBlocks.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!
            let qsBytes = UnsafeMutableRawPointer(qsPtr).assumingMemoryBound(to: UInt8.self)
            let chunks = 32
            let blocksPerChunk = (nBlocks + chunks - 1) / chunks
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                let start = c * blocksPerChunk
                let end = min(start + blocksPerChunk, nBlocks)
                guard start < end else { return }
                for b in start..<end {
                    let blockBase = base.advanced(by: b * q2_KBlockBytes)
                    memcpy(scPtr.advanced(by: b * 16), blockBase, 16)
                    memcpy(qsBytes.advanced(by: b * 64), blockBase.advanced(by: 16), 64)
                    let dBits = UnsafeRawPointer(blockBase + 80).load(as: UInt16.self)
                    let dminBits = UnsafeRawPointer(blockBase + 82).load(as: UInt16.self)
                    dPtr[b] = Float(Float16(bitPattern: dBits))
                    dminPtr[b] = Float(Float16(bitPattern: dminBits))
                }
            }
        }
        let qsTensor = Tensor(buffer: qsBuf, offset: 0, shape: [nBlocks * 16], dtype: .u32)
        let scalesTensor = Tensor(buffer: scBuf, offset: 0, shape: [nBlocks * 16], dtype: .u8)
        let dTensor = Tensor(buffer: dBuf, offset: 0, shape: [nBlocks], dtype: .f32)
        let dminTensor = Tensor(buffer: dminBuf, offset: 0, shape: [nBlocks], dtype: .f32)
        return Ops.ggufDequantQ2_K(
            qsPacked: qsTensor, scales: scalesTensor,
            dF32: dTensor, dminF32: dminTensor,
            nValues: nValues, outDtype: outDtype,
            on: cmd, into: out)
    }

    // ─── IQ2_XXS ───────────────────────────────────────────────────────

    /// Block: `[fp16 d (2 B); u16 qs[32] (64 B)]` = 66 B per 256 values.
    public static let iq2_xxsBlockBytes = 66
    public static let iq2_xxsBlockValues = 256

    static func dequantIQ2_XXS(
        rawBlocks: Data, nValues: Int, outDtype: DType,
        gridTensor: Tensor, signsTensor: Tensor,
        on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor? = nil, slot: String = "default"
    ) -> Tensor {
        precondition(nValues % iq2_xxsBlockValues == 0)
        let nBlocks = nValues / iq2_xxsBlockValues
        precondition(
            rawBlocks.count >= nBlocks * iq2_xxsBlockBytes,
            "GGUFDequant.IQ2_XXS: rawBlocks too short for \(nBlocks) blocks")

        let qsBytes = nBlocks * 16 * 4
        let dBytes = nBlocks * 4
        let qsBuf = device.intermediateScratch(tag: "gguf_dequant_u32_\(slot)", minBytes: qsBytes)
        let dBuf = device.intermediateScratch(tag: "gguf_dequant_f32_\(slot)", minBytes: dBytes)
        let qsPtr = qsBuf.contents().assumingMemoryBound(to: UInt32.self)
        let dPtr = dBuf.contents().assumingMemoryBound(to: Float.self)
        let _tStage0 = CACurrentMediaTime()
        rawBlocks.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!
            let qsBytes = UnsafeMutableRawPointer(qsPtr).assumingMemoryBound(to: UInt8.self)
            let chunks = 32
            let blocksPerChunk = (nBlocks + chunks - 1) / chunks
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                let start = c * blocksPerChunk
                let end = min(start + blocksPerChunk, nBlocks)
                guard start < end else { return }
                for b in start..<end {
                    let blockBase = base.advanced(by: b * iq2_xxsBlockBytes)
                    let dBits = UnsafeRawPointer(blockBase).load(as: UInt16.self)
                    dPtr[b] = Float(Float16(bitPattern: dBits))
                    memcpy(qsBytes.advanced(by: b * 64), blockBase.advanced(by: 2), 64)
                }
            }
        }
        let _tStage1 = CACurrentMediaTime()
        GGUFDequant.profStageIq2 += _tStage1 - _tStage0
        let qsTensor = Tensor(buffer: qsBuf, offset: 0, shape: [nBlocks * 16], dtype: .u32)
        let dTensor = Tensor(buffer: dBuf, offset: 0, shape: [nBlocks], dtype: .f32)
        let r = Ops.ggufDequantIQ2_XXS(
            qsU32: qsTensor, dF32: dTensor,
            grid: gridTensor, signs: signsTensor,
            nValues: nValues, outDtype: outDtype,
            on: cmd, into: out)
        GGUFDequant.profEncodeIq2 += CACurrentMediaTime() - _tStage1
        return r
    }

    // ─── Shared LUT cache ──────────────────────────────────────────────

    /// One-shot upload of the IQ2_XXS lookup tables. Cached so the
    /// 2048+128-byte upload happens at most once per process.
    /// `NSLock`-guarded for the rare cross-thread first-touch case;
    /// after first init the read is a fast pointer compare.
    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var iq2xxsTablesCache:
        (grid: Tensor, signs: Tensor, device: Device)? = nil

    public static func iq2xxsTables(device: Device) -> (grid: Tensor, signs: Tensor) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let cached = iq2xxsTablesCache, cached.device === device {
            return (cached.grid, cached.signs)
        }
        // MUST be dedicated, persistent buffers — NOT the shared
        // "gguf_dequant_u8" scratch slot that makeU8Tensor uses. grid
        // (2048 B) and ksigns (128 B) have different sizes but the same
        // scratch tag, so routing both through makeU8Tensor made the
        // ksigns copy overwrite the first 128 bytes of the grid buffer
        // (corrupting grid entries for keys 0..15 → every IQ2_XXS dequant
        // that hit a low grid key produced wrong weights). These tables
        // are static and live for the whole process, so give each its own
        // owned MTLBuffer.
        let grid = persistentU8Tensor(bytes: GGUFIQ2XXSTables.grid, device: device)
        let signs = persistentU8Tensor(bytes: GGUFIQ2XXSTables.ksigns, device: device)
        iq2xxsTablesCache = (grid, signs, device)
        return (grid, signs)
    }

    // ─── Buffer construction helpers ───────────────────────────────────

    /// Dedicated, owned u8 buffer for a static lookup table (IQ2_XXS
    /// grid / ksigns). Unlike `makeU8Tensor` this does NOT use the shared
    /// scratch slot, so two tables of different sizes can't alias.
    private static func persistentU8Tensor(bytes: [UInt8], device: Device) -> Tensor {
        let buf = device.makeBuffer(length: max(bytes.count, 1))
        bytes.withUnsafeBufferPointer { src in
            if let base = src.baseAddress {
                buf.contents().copyMemory(from: UnsafeRawPointer(base), byteCount: bytes.count)
            }
        }
        return Tensor(buffer: buf, offset: 0, shape: [bytes.count], dtype: .u8)
    }

    private static func makeU8Tensor(bytes: [UInt8], device: Device) -> Tensor {
        // The IQ2_XXS lookup tables (grid + ksigns) are static and
        // SHARED across every IQ2_XXS dequant — those go through the
        // long-lived `iq2xxsTablesCache` (see `iq2xxsTables`), not
        // through this helper. The TRANSIENT u8 buffers that DO flow
        // through here are the per-dequant `qs_signed` (Q8_0) and
        // `scales` (Q2_K) intermediates — both safely reusable per
        // call boundary (caller commits + waits the dequant cmd
        // buffer before returning).
        let need = max(bytes.count, 1)
        let buf = device.intermediateScratch(tag: "gguf_dequant_u8", minBytes: need)
        bytes.withUnsafeBufferPointer { src in
            buf.contents().copyMemory(
                from: UnsafeRawPointer(src.baseAddress!), byteCount: bytes.count)
        }
        return Tensor(buffer: buf, offset: 0, shape: [bytes.count], dtype: .u8)
    }

    private static func makeU32Tensor(values: [UInt32], device: Device) -> Tensor {
        // Use the persistent dequant intermediate buffer keyed by
        // "u32". Caller (GGUFTensorBundle.tensor) commits + waits the
        // dequant kernel before returning, so the same buffer is
        // safely reusable for the next dequant call. Saves ~524 MB
        // per IQ2_XXS expert tensor of fresh-MTLBuffer churn that
        // Metal's driver wasn't pooling efficiently.
        let bytes = max(values.count * 4, 4)
        let buf = device.intermediateScratch(tag: "gguf_dequant_u32", minBytes: bytes)
        values.withUnsafeBufferPointer { src in
            buf.contents().copyMemory(
                from: UnsafeRawPointer(src.baseAddress!), byteCount: values.count * 4)
        }
        return Tensor(buffer: buf, offset: 0, shape: [values.count], dtype: .u32)
    }

    private static func makeF32Tensor(values: [Float], device: Device) -> Tensor {
        let bytes = max(values.count * 4, 4)
        let buf = device.intermediateScratch(tag: "gguf_dequant_f32", minBytes: bytes)
        values.withUnsafeBufferPointer { src in
            buf.contents().copyMemory(
                from: UnsafeRawPointer(src.baseAddress!), byteCount: values.count * 4)
        }
        return Tensor(buffer: buf, offset: 0, shape: [values.count], dtype: .f32)
    }
    nonisolated(unsafe) public static var profStageIq2: Double = 0
    nonisolated(unsafe) public static var profEncodeIq2: Double = 0
}
