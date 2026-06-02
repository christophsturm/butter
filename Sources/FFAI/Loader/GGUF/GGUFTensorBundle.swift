// Copyright 2026 Tom Turney (@TheTom)
import QuartzCore
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
// `GGUFTensorBundle` — adapter that exposes a single .gguf file (or a
// directory containing one) as a tensor namespace for the DeepSeek-V4
// GGUF loader path.
//
// NOTE: this is a *parallel* loader for the DSv4 GGUF path, NOT a
// drop-in replacement for `SafeTensorsBundle`. It deliberately does
// not (yet) implement `SafeTensorsBundle`'s public surface
// (`has` / `allKeys` / `prefixed` / `withAddedPrefix` /
// `quantizedTriplet`), and the standard `Model.load` family dispatch
// only ever constructs `SafeTensorsBundle`. DSv4 reaches this bundle
// via the separate `DeepSeekV4Variant.loadModelFromGGUF` entry point.
// Unifying the two behind a shared `TensorBundle` protocol so the
// dispatcher can take either format is future work, deferred until the
// DSv4 forward path lands.
//
// **Status:** WIP scaffold. The reader (header + KV + tensor-info
// table) is fully implemented in `GGUFReader.swift` and works end-to-
// end on real DeepSeek-V4-Flash IQ2_XXS GGUFs. `tensor(named:)` decodes
// the on-disk bytes into the GPU-resident split that metaltile's GGUF
// dequant kernels expect (Q8_0 / Q2_K / IQ2_XXS) — for formats whose
// dequant kernels haven't landed yet (every i-quant other than
// IQ2_XXS, the FP4 / TQ / MXFP4 variants), it throws
// `GGUFError.unsupportedDequant` so the loader fails fast instead of
// returning garbage.

import Foundation
import Metal
import Tokenizers

/// A single GGUF file presented as a tensor namespace for the
/// DeepSeek-V4 GGUF loader path. NOT a drop-in `SafeTensorsBundle`
/// replacement — it exposes a GGUF-specific surface (staged / resident
/// expert-slice dequant) and is consumed only via
/// `DeepSeekV4Variant.loadModelFromGGUF`, not the standard family
/// dispatch. See the file-header note for the shared-protocol
/// unification that's deferred to future work.
/// Resident pre-split IQ2_XXS expert weights for one MoE tensor — the
/// full `[nExperts × nblkPerExpert]` split buffers, filled lazily per
/// expert on first route. Subsequent tokens routing the same expert pay
/// zero staging. macOS storage-mode-shared MTLBuffers commit physical
/// pages on first write, so only the touched experts cost memory.
/// Capacity (distinct experts) of a packed resident pool per tensor.
/// The full 256-expert split would be ~85 GB across the model and
/// thrash against the 84 GB mmap; the working set actually touched
/// during a decode is far smaller, so we pack only touched experts into
/// `RESIDENT_POOL_CAP` slots (≈15 GB total) keyed by expert id.
let RESIDENT_POOL_CAP = 64

public final class ResidentIQ2Split: @unchecked Sendable {
    public let qs: MTLBuffer
    public let d: MTLBuffer
    public let slotmap: MTLBuffer  // [nExperts] u32: expert id → packed slot (GPU-resident)
    public let nBlocksPerExpert: Int
    public let mOut: Int
    public let kIn: Int
    public var slotOf: [Int: Int] = [:]  // expert id → packed slot
    public var nextSlot: Int = 0
    init(qs: MTLBuffer, d: MTLBuffer, slotmap: MTLBuffer, nBlocksPerExpert: Int, mOut: Int, kIn: Int) {
        self.qs = qs; self.d = d; self.slotmap = slotmap; self.nBlocksPerExpert = nBlocksPerExpert
        self.mOut = mOut; self.kIn = kIn
    }
}

public final class ResidentQ2KSplit: @unchecked Sendable {
    public let qs: MTLBuffer
    public let scales: MTLBuffer
    public let d: MTLBuffer
    public let dmin: MTLBuffer
    public let slotmap: MTLBuffer
    public let nBlocksPerExpert: Int
    public let mOut: Int
    public let kIn: Int
    public var slotOf: [Int: Int] = [:]
    public var nextSlot: Int = 0
    init(
        qs: MTLBuffer, scales: MTLBuffer, d: MTLBuffer, dmin: MTLBuffer, slotmap: MTLBuffer,
        nBlocksPerExpert: Int, mOut: Int, kIn: Int
    ) {
        self.qs = qs; self.scales = scales; self.d = d; self.dmin = dmin; self.slotmap = slotmap
        self.nBlocksPerExpert = nBlocksPerExpert; self.mOut = mOut; self.kIn = kIn
    }
}

/// Resident Q8_0 weight split (whole tensor) for `Ops.gemvQ8` — kept at
/// 1 byte/weight (qs int8 + per-block f32 scale) rather than expanded to
/// f16, halving the bandwidth of the dense attn/shexp projections that
/// dominate decode GPU time.
public final class ResidentQ8: @unchecked Sendable {
    public let qs: MTLBuffer  // [nBlocks * 8] u32 (32 int8/block)
    public let d: MTLBuffer  // [nBlocks] f32
    public let mOut: Int
    public let kIn: Int
    init(qs: MTLBuffer, d: MTLBuffer, mOut: Int, kIn: Int) {
        self.qs = qs; self.d = d; self.mOut = mOut; self.kIn = kIn
    }
}

public final class GGUFTensorBundle: @unchecked Sendable {
    public let directory: URL
    public let reader: GGUFReader
    nonisolated(unsafe) var iq2SplitCache: [String: ResidentIQ2Split] = [:]
    nonisolated(unsafe) var q2kSplitCache: [String: ResidentQ2KSplit] = [:]
    // PERSISTENT prefill pools (cap = nExperts), built once at first prefill
    // and reused warm across chunks/layers — the slotOf map persists so an
    // expert is repacked only once (no per-chunk re-read). ~70GB when full;
    // single-copy (mmap pages MADV_FREE'd after repack). This is the
    // resident-weights path (opt-in via FFAI_PREFILL_RESIDENT).
    nonisolated(unsafe) var iq2PrefillCache: [String: ResidentIQ2Split] = [:]
    nonisolated(unsafe) var q2kPrefillCache: [String: ResidentQ2KSplit] = [:]
    nonisolated(unsafe) var q8Cache: [String: ResidentQ8] = [:]
    // RAW bulk-gather pools (interleaved blocks, no deinterleave) for the
    // view-u16 bgemm — reliable makeBuffer GPU memory, cheap bulk-memcpy fill.
    struct RawGatherEntry { var buffer: MTLBuffer; var slotOf: [Int: Int]; var nextSlot: Int }
    nonisolated(unsafe) var rawGatherCache: [String: RawGatherEntry] = [:]
    let gatherCacheLock = NSLock()

    public init(directory: URL) throws {
        self.directory = directory
        // Locate the .gguf file inside the directory. Conventional
        // single-file layout; sharded GGUF is rare in 2026 (the
        // 4-bit DSv4 file is 86 GB and ships as a single blob).
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        let ggufs = contents.filter { $0.pathExtension == "gguf" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard let url = ggufs.first else {
            throw GGUFError.missingMetadataKey("any .gguf file in \(directory.path)")
        }
        // If the user has both a main weights GGUF and a sibling MTP-only
        // GGUF (DSv4 ships this way), prefer the larger one for the
        // weights bundle; the MTP heads load separately via the same
        // reader once that path is wired.
        let preferred =
            ggufs.max(by: {
                let lhs =
                    (try? FileManager.default.attributesOfItem(atPath: $0.path))?[.size]
                    as? Int ?? 0
                let rhs =
                    (try? FileManager.default.attributesOfItem(atPath: $1.path))?[.size]
                    as? Int ?? 0
                return lhs < rhs
            }) ?? url
        self.reader = try GGUFReader(url: preferred)
    }

    /// Single-file convenience init when the caller already knows the
    /// GGUF URL exactly (tests, the MTP-side load, ...).
    public init(url: URL) throws {
        self.directory = url.deletingLastPathComponent()
        self.reader = try GGUFReader(url: url)
    }

    /// Materialize a tensor from the GGUF as a host-side `Tensor`.
    /// Supported on-disk formats: F32 / F16 / BF16 (direct copy) and
    /// Q8_0 / Q2_K / IQ2_XXS (GPU dequant via the metaltile
    /// `ffai_gguf_dequant_*` kernels). Other quant types raise
    /// `GGUFError.unsupportedDequant` — they land in follow-ups as
    /// the kernel surface grows.
    ///
    /// - Parameters:
    ///   - named: tensor name from the GGUF tensor info table
    ///   - outDtype: target activation dtype for the returned tensor.
    ///     `nil` defaults to f32 for quantized inputs and the on-disk
    ///     dtype for float inputs.
    ///   - device: the device whose command queue handles the dequant
    ///     dispatch. Defaults to `.shared`.
    ///   - persistent: when `true`, quantized-dequant output lands in a
    ///     per-tensor-name scratch slot (stable across calls) instead of
    ///     the shape-keyed shared "full" pool. REQUIRED for any weight
    ///     kept resident past the call (e.g. the DSv4 shared-expert
    ///     gate/up/down): otherwise two same-shape dequants (gate + up)
    ///     resolve to the SAME pooled buffer and the second overwrites
    ///     the first — silently aliasing every layer's shared expert to
    ///     the last-loaded tensor.
    public func tensor(
        named: String, outDtype: DType? = nil, device: Device = .shared,
        persistent: Bool = false
    ) throws -> Tensor {
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        let shape = info.dimensions.map { Int($0) }
        // NOTE: `reader.rawBytes(named:)` is NOT called at the top of
        // this function — that API uses `Data.subdata` which copies
        // for slices ≥16 KB. For DSv4 expert tensors (~half a GB raw
        // each, 3 per layer), pre-fetching the bytes here duplicates
        // ~1.5 GB / layer into anonymous RAM. Each case below reads
        // bytes lazily — the f16/f32/bf16 path uses `rawBytes` only
        // because the data is small and immediately copied to an
        // MTLBuffer; the quant paths use `withRawBytes` for zero-copy
        // access into the mmap.

        switch info.type {
        case .f32, .f16, .bf16:
            let srcDtype: DType = info.type == .f32 ? .f32 : (info.type == .f16 ? .f16 : .bf16)
            let dstDtype = outDtype ?? srcDtype
            if srcDtype == dstDtype {
                // Fast path — direct byte copy straight from the mmap
                // into the MTLBuffer. Uses `withRawBytes` (zero-copy
                // view of the mapped region) rather than `rawBytes`
                // (which `subdata`-copies the whole tensor into an
                // anonymous heap Data first). For token_embd / output
                // (each ~1 GB f16) that second copy was both wasteful
                // and crashed the release build at the function-return
                // boundary (1 GB temp Data churn).
                let buf = try reader.withRawBytes(named: named) { src -> MTLBuffer in
                    let b = device.makeBuffer(length: max(src.count, srcDtype.byteSize))
                    if let base = src.baseAddress, src.count > 0 {
                        b.contents().copyMemory(from: base, byteCount: src.count)
                    }
                    return b
                }
                return Tensor(buffer: buf, offset: 0, shape: shape, dtype: srcDtype)
            }
            let raw = try reader.rawBytes(named: named)
            // Cross-dtype convert path — go through f32, then narrow
            // to the destination dtype. Tensors hit here are small
            // (norms, sinks, biases) so CPU conversion is fine; the
            // bulk weights live on the q8_0 / q2_K / iq2_xxs paths
            // below where the dequant kernel handles the dtype
            // narrowing on-GPU.
            return try Self.convertHalfPrecisionTensor(
                raw: raw, srcDtype: srcDtype, dstDtype: dstDtype,
                shape: shape, device: device)

        case .q8_0:
            return try dequantWholeTensor(
                named: named, shape: shape, nValues: Int(info.numElements),
                outDtype: outDtype, persistent: persistent, device: device
            ) { raw, nValues, dtOut, cmd, out in
                _ = GGUFDequant.dequantQ8_0(
                    rawBlocks: raw, nValues: nValues, outDtype: dtOut,
                    on: cmd, device: device, into: out)
            }

        case .q2_K:
            return try dequantWholeTensor(
                named: named, shape: shape, nValues: Int(info.numElements),
                outDtype: outDtype, persistent: persistent, device: device
            ) { raw, nValues, dtOut, cmd, out in
                _ = GGUFDequant.dequantQ2_K(
                    rawBlocks: raw, nValues: nValues, outDtype: dtOut,
                    on: cmd, device: device, into: out)
            }

        case .iq2_xxs:
            let (grid, signs) = GGUFDequant.iq2xxsTables(device: device)
            return try dequantWholeTensor(
                named: named, shape: shape, nValues: Int(info.numElements),
                outDtype: outDtype, persistent: persistent, device: device
            ) { raw, nValues, dtOut, cmd, out in
                _ = GGUFDequant.dequantIQ2_XXS(
                    rawBlocks: raw, nValues: nValues, outDtype: dtOut,
                    gridTensor: grid, signsTensor: signs,
                    on: cmd, device: device, into: out)
            }

        case .i32, .i64, .i8:
            // Integer tables (e.g. DSv4 ffn_gate_tid2eid hash-routing
            // table) — carried through verbatim, no dequant. Bytes copy
            // straight from the mmap into an MTLBuffer; consumers read
            // them host-side via `toArray(as:)`.
            let srcDtype: DType = info.type == .i64 ? .i64 : (info.type == .i8 ? .i8 : .i32)
            let buf = try reader.withRawBytes(named: named) { src -> MTLBuffer in
                let b = device.makeBuffer(length: max(src.count, srcDtype.byteSize))
                if let base = src.baseAddress, src.count > 0 {
                    b.contents().copyMemory(from: base, byteCount: src.count)
                }
                return b
            }
            return Tensor(buffer: buf, offset: 0, shape: shape, dtype: srcDtype)

        default:
            throw GGUFError.unsupportedDequant(info.type, tensor: named)
        }
    }

    /// Shared driver for the whole-tensor block-quant cases of
    /// `tensor(named:)` (Q8_0 / Q2_K / IQ2_XXS). Allocates the pooled
    /// dequant output, wraps the tensor's mmap bytes zero-copy, runs the
    /// caller's dequant kernel into the output on a fresh cmd buffer,
    /// commits + waits, and reshapes. Only the kernel call differs
    /// between quant types, so it's the closure; everything else
    /// (pooling, zero-copy, sync) is identical and lives here.
    ///
    /// `dequant` receives `(rawBlocks, nValues, outDtype, cmd, out)` and
    /// must encode its kernel into `cmd` writing into `out` — it must not
    /// commit (this driver owns the cmd lifecycle).
    private func dequantWholeTensor(
        named: String, shape: [Int], nValues: Int,
        outDtype: DType?, persistent: Bool, device: Device,
        dequant: (
            _ rawBlocks: Data, _ nValues: Int, _ outDtype: DType,
            _ cmd: MTLCommandBuffer, _ out: Tensor
        ) -> Void
    ) throws -> Tensor {
        let dtOut = outDtype ?? .f32
        let out = Self.pooledDequantOutput(
            nValues: nValues, dtype: dtOut, device: device,
            tagSuffix: persistent ? named : "full")
        let cmd = device.makeCommandBuffer()
        try reader.withRawBytes(named: named) { ptr in
            let zeroCopy = Data(
                bytesNoCopy: UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                count: ptr.count, deallocator: .none)
            dequant(zeroCopy, nValues, dtOut, cmd, out)
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        return out.reshaped(to: shape)
    }

    // ─── Per-expert MoE dequant (staged / gathered) ──────────────────
    //
    // Per-expert dequant of a DSv4 IQ2_XXS expert ([4096, 2048] slice
    // of [4096, 2048, 256]) is ~2 MB raw read + ~16 MB dequanted output,
    // vs. ~530 MB raw + ~4 GB output for the full tensor —
    // `n_experts / top_k_per_token = 256 / 6 ≈ 43×` less work per token
    // when only the selected experts need materialization.

    /// Staging result for a parallel CPU pre-pass over one expert
    /// slice. The GPU encode step (`encodeStagedExpertSlice`) runs on
    /// the main thread later, reading these buffer refs.
    public struct StagedExpertSlice {
        public let outBuf: MTLBuffer
        public let qsBuf: MTLBuffer
        public let dBuf: MTLBuffer
        public let scBuf: MTLBuffer?
        public let dminBuf: MTLBuffer?
        public let nValuesPerExpert: Int
        public let nBlocks: Int
        public let outDtype: DType
        public let outShape: [Int]
        public let infoType: GGUFTensorType
    }

    /// CPU-only staging: do the block-byte unpack into intermediate
    /// MTLBuffers. Thread-safe across calls with distinct `slot` tags
    /// (each tag → its own pooled buffer set).
    public func stageExpertSlice(
        named: String, expertIdx: Int, nExperts: Int,
        slot: String, outDtype: DType? = nil, device: Device = .shared
    ) throws -> StagedExpertSlice {
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        let nValuesTotal = Int(info.numElements)
        let nValuesPerExpert = nValuesTotal / nExperts
        let byteStart = (Int(info.byteLength) / nExperts) * expertIdx
        let byteLen = Int(info.byteLength) / nExperts
        let dtOut = outDtype ?? .f32
        let outShape = info.dimensions.dropLast().map { Int($0) }
        switch info.type {
        case .iq2_xxs:
            let nBlocks = nValuesPerExpert / GGUFDequant.iq2_xxsBlockValues
            let qsBytes = nBlocks * 16 * 4
            let dBytes = nBlocks * 4
            let qsBuf = device.intermediateScratch(tag: "gguf_dequant_u32_\(slot)", minBytes: qsBytes)
            let dBuf = device.intermediateScratch(tag: "gguf_dequant_f32_\(slot)", minBytes: dBytes)
            let outBuf = device.intermediateScratch(
                tag: "dequant_out_\(dtOut)_\(nValuesPerExpert)_expert_\(slot)",
                minBytes: nValuesPerExpert * dtOut.byteSize)
            let qsPtr = qsBuf.contents().assumingMemoryBound(to: UInt32.self)
            let dPtr = dBuf.contents().assumingMemoryBound(to: Float.self)
            try reader.withRawBytesSlice(named: named, byteStart: byteStart, byteLength: byteLen) { ptr in
                let base = ptr.baseAddress!
                let qsU8 = qsPtr.withMemoryRebound(to: UInt8.self, capacity: nBlocks * 64) { $0 }
                // Inner serial (no nested concurrentPerform — outer
                // already parallel across calls).
                for b in 0 ..< nBlocks {
                    let blockBase = base.advanced(by: b * GGUFDequant.iq2_xxsBlockBytes)
                    let dBits = blockBase.withMemoryRebound(to: UInt16.self, capacity: 1) { $0.pointee }
                    dPtr[b] = Float(Float16(bitPattern: dBits))
                    memcpy(qsU8.advanced(by: b * 64), blockBase.advanced(by: 2), 64)
                }
            }
            return StagedExpertSlice(
                outBuf: outBuf, qsBuf: qsBuf, dBuf: dBuf, scBuf: nil, dminBuf: nil,
                nValuesPerExpert: nValuesPerExpert, nBlocks: nBlocks,
                outDtype: dtOut, outShape: outShape, infoType: .iq2_xxs)
        case .q2_K:
            let nBlocks = nValuesPerExpert / GGUFDequant.q2_KBlockValues
            let qsBytes = nBlocks * 16 * 4
            let scBytes = nBlocks * 16
            let dBytes = nBlocks * 4
            let qsBuf = device.intermediateScratch(tag: "gguf_dequant_u32_\(slot)", minBytes: qsBytes)
            let scBuf = device.intermediateScratch(tag: "gguf_dequant_u8_\(slot)", minBytes: scBytes)
            let dBuf = device.intermediateScratch(tag: "gguf_dequant_f32_\(slot)", minBytes: dBytes)
            let dminBuf = device.intermediateScratch(tag: "gguf_dequant_f32_dmin_\(slot)", minBytes: dBytes)
            let outBuf = device.intermediateScratch(
                tag: "dequant_out_\(dtOut)_\(nValuesPerExpert)_expert_\(slot)",
                minBytes: nValuesPerExpert * dtOut.byteSize)
            let qsPtr = qsBuf.contents().assumingMemoryBound(to: UInt32.self)
            let scPtr = scBuf.contents().assumingMemoryBound(to: UInt8.self)
            let dPtr = dBuf.contents().assumingMemoryBound(to: Float.self)
            let dminPtr = dminBuf.contents().assumingMemoryBound(to: Float.self)
            try reader.withRawBytesSlice(named: named, byteStart: byteStart, byteLength: byteLen) { ptr in
                let base = ptr.baseAddress!
                let qsU8 = qsPtr.withMemoryRebound(to: UInt8.self, capacity: nBlocks * 64) { $0 }
                for b in 0 ..< nBlocks {
                    let blockBase = base.advanced(by: b * GGUFDequant.q2_KBlockBytes)
                    memcpy(scPtr.advanced(by: b * 16), blockBase, 16)
                    memcpy(qsU8.advanced(by: b * 64), blockBase.advanced(by: 16), 64)
                    let dBits = blockBase.advanced(by: 80).withMemoryRebound(to: UInt16.self, capacity: 1) {
                        $0.pointee
                    }
                    let dminBits = blockBase.advanced(by: 82).withMemoryRebound(to: UInt16.self, capacity: 1) {
                        $0.pointee
                    }
                    dPtr[b] = Float(Float16(bitPattern: dBits))
                    dminPtr[b] = Float(Float16(bitPattern: dminBits))
                }
            }
            return StagedExpertSlice(
                outBuf: outBuf, qsBuf: qsBuf, dBuf: dBuf, scBuf: scBuf, dminBuf: dminBuf,
                nValuesPerExpert: nValuesPerExpert, nBlocks: nBlocks,
                outDtype: dtOut, outShape: outShape, infoType: .q2_K)
        default:
            throw GGUFError.unsupportedDequant(info.type, tensor: named)
        }
    }

    /// Result of staging N routed IQ2_XXS experts into contiguous
    /// slot-major split buffers for `Ops.moeGatherGemvIQ2XXS`.
    public struct GatheredIQ2XXS {
        public let qsAll: Tensor  // [nSlots * nblkPerExpert * 16] u32
        public let dAll: Tensor  // [nSlots * nblkPerExpert]      f32
        public let nSlots: Int
        public let mOut: Int  // output rows per expert
        public let kIn: Int  // input dim
    }

    /// CPU staging for the fused 6-expert IQ2_XXS gather GEMV. Reads the
    /// quant bytes for each routed expert straight from the (resident)
    /// mmap and lays the split (qs_u32 / d_f32) format down slot-major in
    /// ONE pair of pooled buffers — so the whole role (gate or up) is a
    /// single `moe_gather_gemv_iq2xxs` dispatch instead of 6×{dequant,gemv}.
    public func stageGatherIQ2XXS(
        named: String, expertIndices: [Int], nExperts: Int,
        slot: String, device: Device = .shared
    ) throws -> GatheredIQ2XXS {
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        precondition(info.type == .iq2_xxs, "stageGatherIQ2XXS: \(named) is \(info.type)")
        let nSlots = expertIndices.count
        let nValuesPerExpert = Int(info.numElements) / nExperts
        let nBlocksPerExpert = nValuesPerExpert / GGUFDequant.iq2_xxsBlockValues
        // outShape = [m_out, k_in] (n_experts dim already dropped).
        let dims = info.dimensions.map { Int($0) }  // [k_in, m_out, n_experts] fast-first
        let kIn = dims[0]
        let mOut = dims[1]
        let byteLenPerExpert = Int(info.byteLength) / nExperts

        let qsBytes = nSlots * nBlocksPerExpert * 16 * 4
        let dBytes = nSlots * nBlocksPerExpert * 4
        let qsBuf = device.intermediateScratch(tag: "gather_iq2_qs_\(slot)", minBytes: qsBytes)
        let dBuf = device.intermediateScratch(tag: "gather_iq2_d_\(slot)", minBytes: dBytes)
        let qsPtr = qsBuf.contents().assumingMemoryBound(to: UInt32.self)
        let dPtr = dBuf.contents().assumingMemoryBound(to: Float.self)
        let qsU8 = qsPtr.withMemoryRebound(
            to: UInt8.self, capacity: nSlots * nBlocksPerExpert * 64
        ) { $0 }

        for (s, e) in expertIndices.enumerated() {
            let byteStart = byteLenPerExpert * e
            let qsSlotBlock = s * nBlocksPerExpert  // first block index for this slot
            try reader.withRawBytesSlice(
                named: named, byteStart: byteStart, byteLength: byteLenPerExpert
            ) { ptr in
                let base = ptr.baseAddress!
                // Parallel block-split: distinct dst ranges per block →
                // thread-safe. The qs memcpy is the dominant decode-time
                // CPU cost, so fan it across P-cores.
                let blkBytes = GGUFDequant.iq2_xxsBlockBytes
                let chunks = 16
                let per = (nBlocksPerExpert + chunks - 1) / chunks
                DispatchQueue.concurrentPerform(iterations: chunks) { c in
                    let lo = c * per
                    let hi = min(lo + per, nBlocksPerExpert)
                    var b = lo
                    while b < hi {
                        let blockBase = base.advanced(by: b * blkBytes)
                        let dBits = blockBase.withMemoryRebound(to: UInt16.self, capacity: 1) { $0.pointee }
                        dPtr[qsSlotBlock + b] = Float(Float16(bitPattern: dBits))
                        memcpy(qsU8.advanced(by: (qsSlotBlock + b) * 64), blockBase.advanced(by: 2), 64)
                        b += 1
                    }
                }
            }
        }
        let qsAll = Tensor(buffer: qsBuf, offset: 0, shape: [nSlots * nBlocksPerExpert * 16], dtype: .u32)
        let dAll = Tensor(buffer: dBuf, offset: 0, shape: [nSlots * nBlocksPerExpert], dtype: .f32)
        return GatheredIQ2XXS(qsAll: qsAll, dAll: dAll, nSlots: nSlots, mOut: mOut, kIn: kIn)
    }

    /// Result of staging N routed Q2_K experts into contiguous
    /// slot-major split buffers for `Ops.moeGatherDownQ2K`.
    public struct GatheredQ2K {
        public let qsAll: Tensor  // [nSlots * nblkPerExpert * 16] u32
        public let scalesAll: Tensor  // [nSlots * nblkPerExpert * 16] u8
        public let dAll: Tensor  // [nSlots * nblkPerExpert]      f32
        public let dminAll: Tensor  // [nSlots * nblkPerExpert]      f32
        public let nSlots: Int
        public let mOut: Int
        public let kIn: Int
    }

    /// CPU staging for the fused 6-expert Q2_K gather down-projection.
    /// Lays the split (qs_u32 / scales_u8 / d_f32 / dmin_f32) format down
    /// slot-major in one set of pooled buffers.
    public func stageGatherQ2K(
        named: String, expertIndices: [Int], nExperts: Int,
        slot: String, device: Device = .shared
    ) throws -> GatheredQ2K {
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        precondition(info.type == .q2_K, "stageGatherQ2K: \(named) is \(info.type)")
        let nSlots = expertIndices.count
        let nValuesPerExpert = Int(info.numElements) / nExperts
        let nBlocksPerExpert = nValuesPerExpert / GGUFDequant.q2_KBlockValues
        let dims = info.dimensions.map { Int($0) }  // [k_in, m_out, n_experts]
        let kIn = dims[0]
        let mOut = dims[1]
        let byteLenPerExpert = Int(info.byteLength) / nExperts

        let qsBuf = device.intermediateScratch(
            tag: "gather_q2k_qs_\(slot)", minBytes: nSlots * nBlocksPerExpert * 16 * 4)
        let scBuf = device.intermediateScratch(tag: "gather_q2k_sc_\(slot)", minBytes: nSlots * nBlocksPerExpert * 16)
        let dBuf = device.intermediateScratch(tag: "gather_q2k_d_\(slot)", minBytes: nSlots * nBlocksPerExpert * 4)
        let dminBuf = device.intermediateScratch(
            tag: "gather_q2k_dmin_\(slot)", minBytes: nSlots * nBlocksPerExpert * 4)
        let qsPtr = qsBuf.contents().assumingMemoryBound(to: UInt32.self)
        let scPtr = scBuf.contents().assumingMemoryBound(to: UInt8.self)
        let dPtr = dBuf.contents().assumingMemoryBound(to: Float.self)
        let dminPtr = dminBuf.contents().assumingMemoryBound(to: Float.self)
        let qsU8 = qsPtr.withMemoryRebound(to: UInt8.self, capacity: nSlots * nBlocksPerExpert * 64) { $0 }

        for (s, e) in expertIndices.enumerated() {
            let byteStart = byteLenPerExpert * e
            let slotBlock = s * nBlocksPerExpert
            try reader.withRawBytesSlice(
                named: named, byteStart: byteStart, byteLength: byteLenPerExpert
            ) { ptr in
                let base = ptr.baseAddress!
                let blkBytes = GGUFDequant.q2_KBlockBytes
                let chunks = 16
                let per = (nBlocksPerExpert + chunks - 1) / chunks
                DispatchQueue.concurrentPerform(iterations: chunks) { c in
                    let lo = c * per
                    let hi = min(lo + per, nBlocksPerExpert)
                    var b = lo
                    while b < hi {
                        let blockBase = base.advanced(by: b * blkBytes)
                        memcpy(scPtr.advanced(by: (slotBlock + b) * 16), blockBase, 16)
                        memcpy(qsU8.advanced(by: (slotBlock + b) * 64), blockBase.advanced(by: 16), 64)
                        let dBits = blockBase.advanced(by: 80).withMemoryRebound(to: UInt16.self, capacity: 1) {
                            $0.pointee
                        }
                        let dminBits = blockBase.advanced(by: 82).withMemoryRebound(to: UInt16.self, capacity: 1) {
                            $0.pointee
                        }
                        dPtr[slotBlock + b] = Float(Float16(bitPattern: dBits))
                        dminPtr[slotBlock + b] = Float(Float16(bitPattern: dminBits))
                        b += 1
                    }
                }
            }
        }
        return GatheredQ2K(
            qsAll: Tensor(buffer: qsBuf, offset: 0, shape: [nSlots * nBlocksPerExpert * 16], dtype: .u32),
            scalesAll: Tensor(buffer: scBuf, offset: 0, shape: [nSlots * nBlocksPerExpert * 16], dtype: .u8),
            dAll: Tensor(buffer: dBuf, offset: 0, shape: [nSlots * nBlocksPerExpert], dtype: .f32),
            dminAll: Tensor(buffer: dminBuf, offset: 0, shape: [nSlots * nBlocksPerExpert], dtype: .f32),
            nSlots: nSlots, mOut: mOut, kIn: kIn)
    }

    /// Resident lazy-fill IQ2_XXS gather: returns the full
    /// `[nExperts × nblk]` split buffers for `named`, ensuring the
    /// requested `expertIndices` are filled. Experts fill once and are
    /// reused across tokens — eliminating per-token staging for the
    /// touched working set. Returns the buffers + dims; the caller
    /// passes `expertIndices` to the kernel as `expert_ids`.
    public func residentGatherIQ2XXS(
        named: String, expertIndices: [Int], nExperts: Int, device: Device = .shared,
        poolCap: Int? = nil, persist: Bool = false
    ) throws -> (split: ResidentIQ2Split, qsAll: Tensor, dAll: Tensor, slots: [Int])? {
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        let nValuesPerExpert = Int(info.numElements) / nExperts
        let nBlocksPerExpert = nValuesPerExpert / GGUFDequant.iq2_xxsBlockValues
        let dims = info.dimensions.map { Int($0) }
        let byteLenPerExpert = Int(info.byteLength) / nExperts
        // `poolCap` (prefill): use a FRESH, uncached pool sized to fit all
        // requested experts — the cached RESIDENT_POOL_CAP=64 pool can't hold
        // the >64 distinct experts a large prefill chunk routes to. Allocated
        // per call, freed after the layer (not stored in the cache).
        let effCap = poolCap ?? RESIDENT_POOL_CAP

        gatherCacheLock.lock()
        let s: ResidentIQ2Split
        if persist {
            // PERSISTENT prefill pool: build once at effCap, reuse warm across
            // chunks (slotOf persists → each expert repacked only once).
            var split = iq2PrefillCache[named]
            if split == nil {
                split = ResidentIQ2Split(
                    qs: device.makeBuffer(length: effCap * nBlocksPerExpert * 16 * 4),
                    d: device.makeBuffer(length: effCap * nBlocksPerExpert * 4),
                    slotmap: device.makeBuffer(length: nExperts * 4),
                    nBlocksPerExpert: nBlocksPerExpert, mOut: dims[1], kIn: dims[0])
                memset(split!.slotmap.contents(), 0, nExperts * 4)
                iq2PrefillCache[named] = split
            }
            s = split!
        } else if poolCap != nil {
            s = ResidentIQ2Split(
                qs: device.makeBuffer(length: effCap * nBlocksPerExpert * 16 * 4),
                d: device.makeBuffer(length: effCap * nBlocksPerExpert * 4),
                slotmap: device.makeBuffer(length: nExperts * 4),
                nBlocksPerExpert: nBlocksPerExpert, mOut: dims[1], kIn: dims[0])
            memset(s.slotmap.contents(), 0, nExperts * 4)
        } else {
            var split = iq2SplitCache[named]
            if split == nil {
                split = ResidentIQ2Split(
                    qs: device.makeBuffer(length: RESIDENT_POOL_CAP * nBlocksPerExpert * 16 * 4),
                    d: device.makeBuffer(length: RESIDENT_POOL_CAP * nBlocksPerExpert * 4),
                    slotmap: device.makeBuffer(length: nExperts * 4),
                    nBlocksPerExpert: nBlocksPerExpert, mOut: dims[1], kIn: dims[0])
                memset(split!.slotmap.contents(), 0, nExperts * 4)  // misses → in-bounds slot 0
                iq2SplitCache[named] = split
            }
            s = split!
        }
        let slotmapPtr = s.slotmap.contents().assumingMemoryBound(to: UInt32.self)
        // Resolve packed slots; signal a fall-back if the pool is full
        // and a new expert appears (caller uses the staging path).
        var slots: [Int] = []
        slots.reserveCapacity(expertIndices.count)
        var toFill: [(slot: Int, expert: Int)] = []
        for e in expertIndices {
            if let sl = s.slotOf[e] { slots.append(sl); continue }
            guard s.nextSlot < effCap else { gatherCacheLock.unlock(); return nil }
            let sl = s.nextSlot; s.nextSlot += 1; s.slotOf[e] = sl
            slotmapPtr[e] = UInt32(sl)  // mirror to GPU slotmap for the sync-free path
            slots.append(sl); toFill.append((sl, e))
        }
        gatherCacheLock.unlock()

        let qsPtr = s.qs.contents().assumingMemoryBound(to: UInt32.self)
        let dPtr = s.d.contents().assumingMemoryBound(to: Float.self)
        let qsU8 = qsPtr.withMemoryRebound(to: UInt8.self, capacity: effCap * nBlocksPerExpert * 64) { $0 }
        let blkBytes = GGUFDequant.iq2_xxsBlockBytes
        // Parallel OVER experts (not blocks-within-expert): each expert's
        // ~32k cold mmap blocks page-fault off the 86GB file; issuing many
        // experts' faults concurrently gives the NVMe real queue depth
        // (serial per-expert faults left the SSD idle between experts).
        DispatchQueue.concurrentPerform(iterations: toFill.count) { fi in
            let (slot, e) = toFill[fi]
            let byteStart = byteLenPerExpert * e
            let base0 = slot * nBlocksPerExpert
            try? reader.withRawBytesSlice(named: named, byteStart: byteStart, byteLength: byteLenPerExpert) { ptr in
                let base = ptr.baseAddress!
                var b = 0
                while b < nBlocksPerExpert {
                    let blockBase = base.advanced(by: b * blkBytes)
                    let dBits = blockBase.withMemoryRebound(to: UInt16.self, capacity: 1) { $0.pointee }
                    dPtr[base0 + b] = Float(Float16(bitPattern: dBits))
                    memcpy(qsU8.advanced(by: (base0 + b) * 64), blockBase.advanced(by: 2), 64)
                    b += 1
                }
            }
        }
        let qsAll = Tensor(buffer: s.qs, offset: 0, shape: [effCap * nBlocksPerExpert * 16], dtype: .u32)
        let dAll = Tensor(buffer: s.d, offset: 0, shape: [effCap * nBlocksPerExpert], dtype: .f32)
        return (s, qsAll, dAll, slots)
    }

    /// Resident lazy-fill Q2_K gather (down role). See `residentGatherIQ2XXS`.
    public func residentGatherQ2K(
        named: String, expertIndices: [Int], nExperts: Int, device: Device = .shared,
        poolCap: Int? = nil, persist: Bool = false
    ) throws -> (
        split: ResidentQ2KSplit, qsAll: Tensor, scalesAll: Tensor, dAll: Tensor, dminAll: Tensor, slots: [Int]
    )? {
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        let nValuesPerExpert = Int(info.numElements) / nExperts
        let nBlocksPerExpert = nValuesPerExpert / GGUFDequant.q2_KBlockValues
        let dims = info.dimensions.map { Int($0) }
        let byteLenPerExpert = Int(info.byteLength) / nExperts
        let effCap = poolCap ?? RESIDENT_POOL_CAP  // prefill: fresh uncached pool sized to fit all experts

        gatherCacheLock.lock()
        let s: ResidentQ2KSplit
        if persist {
            var split = q2kPrefillCache[named]
            if split == nil {
                split = ResidentQ2KSplit(
                    qs: device.makeBuffer(length: effCap * nBlocksPerExpert * 16 * 4),
                    scales: device.makeBuffer(length: effCap * nBlocksPerExpert * 16),
                    d: device.makeBuffer(length: effCap * nBlocksPerExpert * 4),
                    dmin: device.makeBuffer(length: effCap * nBlocksPerExpert * 4),
                    slotmap: device.makeBuffer(length: nExperts * 4),
                    nBlocksPerExpert: nBlocksPerExpert, mOut: dims[1], kIn: dims[0])
                memset(split!.slotmap.contents(), 0, nExperts * 4)
                q2kPrefillCache[named] = split
            }
            s = split!
        } else if poolCap != nil {
            s = ResidentQ2KSplit(
                qs: device.makeBuffer(length: effCap * nBlocksPerExpert * 16 * 4),
                scales: device.makeBuffer(length: effCap * nBlocksPerExpert * 16),
                d: device.makeBuffer(length: effCap * nBlocksPerExpert * 4),
                dmin: device.makeBuffer(length: effCap * nBlocksPerExpert * 4),
                slotmap: device.makeBuffer(length: nExperts * 4),
                nBlocksPerExpert: nBlocksPerExpert, mOut: dims[1], kIn: dims[0])
            memset(s.slotmap.contents(), 0, nExperts * 4)
        } else {
            var split = q2kSplitCache[named]
            if split == nil {
                split = ResidentQ2KSplit(
                    qs: device.makeBuffer(length: RESIDENT_POOL_CAP * nBlocksPerExpert * 16 * 4),
                    scales: device.makeBuffer(length: RESIDENT_POOL_CAP * nBlocksPerExpert * 16),
                    d: device.makeBuffer(length: RESIDENT_POOL_CAP * nBlocksPerExpert * 4),
                    dmin: device.makeBuffer(length: RESIDENT_POOL_CAP * nBlocksPerExpert * 4),
                    slotmap: device.makeBuffer(length: nExperts * 4),
                    nBlocksPerExpert: nBlocksPerExpert, mOut: dims[1], kIn: dims[0])
                memset(split!.slotmap.contents(), 0, nExperts * 4)
                q2kSplitCache[named] = split
            }
            s = split!
        }
        let slotmapPtr = s.slotmap.contents().assumingMemoryBound(to: UInt32.self)
        var slots: [Int] = []
        var toFill: [(slot: Int, expert: Int)] = []
        for e in expertIndices {
            if let sl = s.slotOf[e] { slots.append(sl); continue }
            guard s.nextSlot < effCap else { gatherCacheLock.unlock(); return nil }
            let sl = s.nextSlot; s.nextSlot += 1; s.slotOf[e] = sl
            slotmapPtr[e] = UInt32(sl)
            slots.append(sl); toFill.append((sl, e))
        }
        gatherCacheLock.unlock()

        let qsPtr = s.qs.contents().assumingMemoryBound(to: UInt32.self)
        let scPtr = s.scales.contents().assumingMemoryBound(to: UInt8.self)
        let dPtr = s.d.contents().assumingMemoryBound(to: Float.self)
        let dminPtr = s.dmin.contents().assumingMemoryBound(to: Float.self)
        let qsU8 = qsPtr.withMemoryRebound(to: UInt8.self, capacity: effCap * nBlocksPerExpert * 64) { $0 }
        let blkBytes = GGUFDequant.q2_KBlockBytes
        // Parallel OVER experts — see residentGatherIQ2XXS rationale (NVMe
        // queue depth for the cold mmap page-faults).
        DispatchQueue.concurrentPerform(iterations: toFill.count) { fi in
            let (slot, e) = toFill[fi]
            let byteStart = byteLenPerExpert * e
            let base0 = slot * nBlocksPerExpert
            try? reader.withRawBytesSlice(named: named, byteStart: byteStart, byteLength: byteLenPerExpert) { ptr in
                let base = ptr.baseAddress!
                var b = 0
                while b < nBlocksPerExpert {
                    let blockBase = base.advanced(by: b * blkBytes)
                    memcpy(scPtr.advanced(by: (base0 + b) * 16), blockBase, 16)
                    memcpy(qsU8.advanced(by: (base0 + b) * 64), blockBase.advanced(by: 16), 64)
                    let dBits = blockBase.advanced(by: 80).withMemoryRebound(to: UInt16.self, capacity: 1) {
                        $0.pointee
                    }
                    let dminBits = blockBase.advanced(by: 82).withMemoryRebound(to: UInt16.self, capacity: 1) {
                        $0.pointee
                    }
                    dPtr[base0 + b] = Float(Float16(bitPattern: dBits))
                    dminPtr[base0 + b] = Float(Float16(bitPattern: dminBits))
                    b += 1
                }
            }
        }
        return (
            s,
            Tensor(buffer: s.qs, offset: 0, shape: [effCap * nBlocksPerExpert * 16], dtype: .u32),
            Tensor(buffer: s.scales, offset: 0, shape: [effCap * nBlocksPerExpert * 16], dtype: .u8),
            Tensor(buffer: s.d, offset: 0, shape: [effCap * nBlocksPerExpert], dtype: .f32),
            Tensor(buffer: s.dmin, offset: 0, shape: [effCap * nBlocksPerExpert], dtype: .f32),
            slots
        )
    }

    /// Resident Q8_0 split for a whole weight tensor (cached). Built once
    /// on first use; the dense attn/shexp projections then gemv directly
    /// from 1-byte Q8 instead of 2-byte f16.
    public func residentQ8(_ named: String, device: Device = .shared) throws -> ResidentQ8 {
        gatherCacheLock.lock()
        if let c = q8Cache[named] { gatherCacheLock.unlock(); return c }
        gatherCacheLock.unlock()
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        precondition(info.type == .q8_0, "residentQ8: \(named) is \(info.type)")
        let nValues = Int(info.numElements)
        let nBlocks = nValues / GGUFDequant.q8_0BlockValues
        let dims = info.dimensions.map { Int($0) }  // [n_in, n_out]
        let qsBuf = device.makeBuffer(length: nBlocks * 8 * 4)
        let dBuf = device.makeBuffer(length: nBlocks * 4)
        let qsPtr = qsBuf.contents().assumingMemoryBound(to: UInt32.self)
        let dPtr = dBuf.contents().assumingMemoryBound(to: Float.self)
        let qsI8 = qsPtr.withMemoryRebound(to: Int8.self, capacity: nBlocks * 32) { $0 }
        let blk = GGUFDequant.q8_0BlockBytes
        try reader.withRawBytes(named: named) { ptr in
            let base = ptr.baseAddress!
            let chunks = 16
            let per = (nBlocks + chunks - 1) / chunks
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                let lo = c * per, hi = min(lo + per, nBlocks)
                var b = lo
                while b < hi {
                    let bb = base.advanced(by: b * blk)
                    let dBits = bb.withMemoryRebound(to: UInt16.self, capacity: 1) { $0.pointee }
                    dPtr[b] = Float(Float16(bitPattern: dBits))
                    memcpy(qsI8.advanced(by: b * 32), bb.advanced(by: 2), 32)
                    b += 1
                }
            }
        }
        let q8 = ResidentQ8(qs: qsBuf, d: dBuf, mOut: dims[1], kIn: dims[0])
        gatherCacheLock.lock(); q8Cache[named] = q8; gatherCacheLock.unlock()
        return q8
    }

    /// Fast-path accessors: the already-built resident pool for a
    /// tensor (nil if the warmup pass hasn't created it yet). The
    /// sync-free GPU router path reads these without filling.
    public func builtIQ2(_ named: String) -> ResidentIQ2Split? {
        gatherCacheLock.lock(); defer { gatherCacheLock.unlock() }
        return iq2SplitCache[named]
    }

    // ── Zero-copy GPU weight views (overlapping no-copy mmap views) ──
    // Lets kernels read raw quant bytes straight from the mmap by
    // (buffer, offset) — no CPU repack into a pool. Built once, lazily.
    private var _modelViews: GGUFModelViews??  // outer optional = "tried", inner = result
    private let modelViewsLock = NSLock()

    /// `(MTLBuffer, byteOffset)` for a tensor's raw bytes, residing in an
    /// overlapping no-copy mmap view. `nil` if views can't be built.
    public func gpuTensorView(named: String, device: Device = .shared) -> (buffer: MTLBuffer, offset: Int)? {
        modelViewsLock.lock()
        if _modelViews == nil {
            let maxTensor = reader.tensorInfos.map { $0.byteLength }.max() ?? 0
            if let base = reader.mmapBase {
                let mv = GGUFModelViews(
                    mmapBase: base, fileSize: reader.mmapByteCount,
                    dataStart: Int(reader.tensorDataOffset),
                    maxTensorBytes: maxTensor, device: device)
                // Register the no-copy windows with the device residency set.
                // The buffers wrap read-only mmap pages; the CPU faults them
                // in on access, but the GPU can only read pages Metal has made
                // resident. The mmap views must be pinned via an MTLResidency
                // Set for exactly this reason — without it the kernel reads
                // unfaulted pages → wrong weights (the FFAI_PREFILL_VIEW gateP
                // divergence). markWeightsResident is a no-op pre-macOS-15.
                if let mv { device.markWeightsResident(mv.views.map { $0.buffer }) }
                _modelViews = mv
            } else {
                _modelViews = .some(nil)  // not a contiguous mmap; no view path
            }
        }
        let mv = _modelViews ?? nil
        modelViewsLock.unlock()
        guard let mv = mv, let idx = reader.tensorIndex[named] else { return nil }
        let info = reader.tensorInfos[idx]
        let absStart = Int(reader.tensorDataOffset + info.dataOffset)
        return mv.view(absStart: absStart, length: info.byteLength)
    }

    /// Per-expert block count + byte stride for an MoE expert tensor, straight
    /// from metadata (NO pool build). IQ2_XXS and Q2_K both pack 256 values
    /// per block; byteLenPerExpert is the kernel's `expert_byte_stride` (the
    /// GGUF expert tensor is [n_experts, n_out, n_in] contiguous). Used by the
    /// zero-copy view bgemm path so it never repacks/MADV_FREEs experts.
    public func expertViewInfo(named: String, nExperts: Int) -> (nBlocksPerExpert: Int, byteLenPerExpert: Int)? {
        guard let idx = reader.tensorIndex[named] else { return nil }
        let info = reader.tensorInfos[idx]
        let nBlocksPerExpert = (Int(info.numElements) / nExperts) / 256
        let byteLenPerExpert = Int(info.byteLength) / nExperts
        return (nBlocksPerExpert, byteLenPerExpert)
    }
    public func builtQ2K(_ named: String) -> ResidentQ2KSplit? {
        gatherCacheLock.lock(); defer { gatherCacheLock.unlock() }
        return q2kSplitCache[named]
    }

    /// Fire-and-forget async readahead of a tensor's mmap pages — see
    /// `GGUFReader.prefetchTensor`. Used to overlap the cold expert-weight
    /// disk I/O with the previous layer's GPU compute (madvise WILLNEED, no
    /// memcpy → no unified-memory bandwidth contention).
    public func prefetchTensor(named: String) { reader.prefetchTensor(named: named) }

    /// RAW bulk gather: copy each routed expert's RAW bytes (interleaved
    /// blocks, no deinterleave) CONTIGUOUSLY into a reliable makeBuffer pool —
    /// ONE bulk memcpy/expert vs residentGatherIQ2XXS's per-block deinterleave
    /// (32768 tiny memcpys/expert). The view-u16 bgemm reads this directly
    /// (slot-indexed, stride = byteLenPerExpert). Reliable GPU memory (avoids
    /// the mmap-residency-zeros bug) + a much cheaper repack. Caches per-name,
    /// pool sized to poolCap. Returns (buffer, slotOf, nBlocksPerExpert, stride).
    /// `reuseKey` (e.g. "gate"/"up"): cache the gather buffer under a STABLE
    /// per-role key instead of the layer-specific tensor name, and REFILL it
    /// each call. In a 43-layer prefill the per-name cache would retain 43×
    /// buffers (~22 GB, never freed = the memory-pressure/freeze bomb); reusing
    /// one buffer per role keeps it at ~2 buffers. Safe because the caller
    /// commits+waits the layer's command buffer before the next layer refills.
    public func rawGatherBlocks(
        named: String, expertIndices: [Int], nExperts: Int, device: Device = .shared, poolCap: Int,
        reuseKey: String? = nil
    ) throws -> (buffer: MTLBuffer, slotOf: [Int: Int], nBlocksPerExpert: Int, byteStride: Int)? {
        guard let idx = reader.tensorIndex[named] else { return nil }
        let info = reader.tensorInfos[idx]
        let byteLenPerExpert = Int(info.byteLength) / nExperts
        let nBlocksPerExpert = (Int(info.numElements) / nExperts) / 256
        let cacheKey = reuseKey ?? named
        gatherCacheLock.lock()
        var ent = rawGatherCache[cacheKey]
        // (Re)allocate when missing or when a prior buffer is too small for this
        // layer's poolCap. With reuseKey we RESET slotOf/nextSlot every call so
        // the single buffer is refilled with THIS layer's experts.
        let needBytes = poolCap * byteLenPerExpert
        if ent == nil || ent!.buffer.length < needBytes {
            ent = RawGatherEntry(buffer: device.makeBuffer(length: needBytes), slotOf: [:], nextSlot: 0)
        } else if reuseKey != nil {
            ent!.slotOf = [:]; ent!.nextSlot = 0
        }
        var slotOf = ent!.slotOf
        var nextSlot = ent!.nextSlot
        var toFill: [(slot: Int, expert: Int)] = []
        for e in expertIndices {
            if slotOf[e] != nil { continue }
            guard nextSlot < poolCap else { gatherCacheLock.unlock(); return nil }
            slotOf[e] = nextSlot; toFill.append((nextSlot, e)); nextSlot += 1
        }
        ent!.slotOf = slotOf; ent!.nextSlot = nextSlot
        rawGatherCache[cacheKey] = ent
        let buf = ent!.buffer
        gatherCacheLock.unlock()
        let dst = buf.contents()
        // CONCURRENT memcpy from ONE whole-tensor map. withRawBytes maps the
        // tensor once and MADV_FREEs it ONCE at the end (single-threaded) — so
        // the per-thread memcpys race nothing (disjoint read-only src regions,
        // disjoint dst regions). At large N (~all experts) this copies the whole
        // tensor; sequential single-thread was ~110ms/tensor, concurrent is far
        // faster. (Per-slice withRawBytesSlice + concurrentPerform was unsafe:
        // each call MADV_FREEs its page-rounded range, evicting neighbors.)
        try reader.withRawBytes(named: named) { ptr in
            let src = ptr.baseAddress!
            DispatchQueue.concurrentPerform(iterations: toFill.count) { fi in
                let (slot, e) = toFill[fi]
                memcpy(
                    dst.advanced(by: slot * byteLenPerExpert),
                    src.advanced(by: byteLenPerExpert * e), byteLenPerExpert)
            }
        }
        return (buf, slotOf, nBlocksPerExpert, byteLenPerExpert)
    }

    /// Encode the GPU dequant kernel for a pre-staged slice. Must be
    /// called from the main thread (Metal encoder is not thread-safe).
    public func encodeStagedExpertSlice(_ s: StagedExpertSlice, device: Device = .shared, on cmd: MTLCommandBuffer)
        -> Tensor
    {
        let out = Tensor(buffer: s.outBuf, offset: 0, shape: [s.nValuesPerExpert], dtype: s.outDtype)
        switch s.infoType {
        case .iq2_xxs:
            let (grid, signs) = GGUFDequant.iq2xxsTables(device: device)
            let qsTensor = Tensor(buffer: s.qsBuf, offset: 0, shape: [s.nBlocks * 16], dtype: .u32)
            let dTensor = Tensor(buffer: s.dBuf, offset: 0, shape: [s.nBlocks], dtype: .f32)
            _ = Ops.ggufDequantIQ2_XXS(
                qsU32: qsTensor, dF32: dTensor,
                grid: grid, signs: signs,
                nValues: s.nValuesPerExpert, outDtype: s.outDtype,
                on: cmd, into: out)
        case .q2_K:
            let qsTensor = Tensor(buffer: s.qsBuf, offset: 0, shape: [s.nBlocks * 16], dtype: .u32)
            let scalesTensor = Tensor(buffer: s.scBuf!, offset: 0, shape: [s.nBlocks * 16], dtype: .u8)
            let dTensor = Tensor(buffer: s.dBuf, offset: 0, shape: [s.nBlocks], dtype: .f32)
            let dminTensor = Tensor(buffer: s.dminBuf!, offset: 0, shape: [s.nBlocks], dtype: .f32)
            _ = Ops.ggufDequantQ2_K(
                qsPacked: qsTensor, scales: scalesTensor,
                dF32: dTensor, dminF32: dminTensor,
                nValues: s.nValuesPerExpert, outDtype: s.outDtype,
                on: cmd, into: out)
        default:
            fatalError("encodeStagedExpertSlice: unsupported type \(s.infoType)")
        }
        return out.reshaped(to: s.outShape)
    }

    public func dequantExpertSliceOnto(
        named: String, expertIdx: Int, nExperts: Int,
        slot: String,
        outDtype: DType? = nil, device: Device = .shared,
        on cmd: MTLCommandBuffer
    ) throws -> Tensor {
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        let nValuesTotal = Int(info.numElements)
        let nValuesPerExpert = nValuesTotal / nExperts
        let byteStart = (Int(info.byteLength) / nExperts) * expertIdx
        let byteLen = Int(info.byteLength) / nExperts
        let dtOut = outDtype ?? .f32
        let outShape = info.dimensions.dropLast().map { Int($0) }
        GGUFTensorBundle.profSliceType[String(describing: info.type), default: 0] += 1
        switch info.type {
        case .q8_0:
            let _tq = CACurrentMediaTime()
            let out = Self.pooledDequantOutput(
                nValues: nValuesPerExpert, dtype: dtOut, device: device,
                tagSuffix: "expert_\(slot)")
            try reader.withRawBytesSlice(named: named, byteStart: byteStart, byteLength: byteLen) { ptr in
                let zeroCopy = Data(
                    bytesNoCopy: UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                    count: ptr.count, deallocator: .none)
                _ = GGUFDequant.dequantQ8_0(
                    rawBlocks: zeroCopy, nValues: nValuesPerExpert, outDtype: dtOut,
                    on: cmd, device: device, into: out, slot: slot)
            }
            GGUFTensorBundle.profSliceQ80 += CACurrentMediaTime() - _tq
            return out.reshaped(to: outShape)
        case .q2_K:
            let _tq = CACurrentMediaTime()
            let out = Self.pooledDequantOutput(
                nValues: nValuesPerExpert, dtype: dtOut, device: device,
                tagSuffix: "expert_\(slot)")
            try reader.withRawBytesSlice(named: named, byteStart: byteStart, byteLength: byteLen) { ptr in
                let zeroCopy = Data(
                    bytesNoCopy: UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                    count: ptr.count, deallocator: .none)
                _ = GGUFDequant.dequantQ2_K(
                    rawBlocks: zeroCopy, nValues: nValuesPerExpert, outDtype: dtOut,
                    on: cmd, device: device, into: out, slot: slot)
            }
            GGUFTensorBundle.profSliceQ2K += CACurrentMediaTime() - _tq
            return out.reshaped(to: outShape)
        case .iq2_xxs:
            let _ta = CACurrentMediaTime()
            let (grid, signs) = GGUFDequant.iq2xxsTables(device: device)
            let _tb = CACurrentMediaTime()
            let out = Self.pooledDequantOutput(
                nValues: nValuesPerExpert, dtype: dtOut, device: device,
                tagSuffix: "expert_\(slot)")
            let _tc = CACurrentMediaTime()
            try reader.withRawBytesSlice(named: named, byteStart: byteStart, byteLength: byteLen) { ptr in
                let _td = CACurrentMediaTime()
                let zeroCopy = Data(
                    bytesNoCopy: UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                    count: ptr.count, deallocator: .none)
                _ = GGUFDequant.dequantIQ2_XXS(
                    rawBlocks: zeroCopy, nValues: nValuesPerExpert, outDtype: dtOut,
                    gridTensor: grid, signsTensor: signs,
                    on: cmd, device: device, into: out, slot: slot)
                let _te = CACurrentMediaTime()
                GGUFTensorBundle.profSliceWrslice += _td - _tc
                GGUFTensorBundle.profSliceDequant += _te - _td
            }
            GGUFTensorBundle.profSlicePooled += _tc - _tb
            GGUFTensorBundle.profSliceTables += _tb - _ta
            return out.reshaped(to: outShape)
        default:
            throw GGUFError.unsupportedDequant(info.type, tensor: named)
        }
    }
    nonisolated(unsafe) public static var profSliceTables: Double = 0
    nonisolated(unsafe) public static var profSlicePooled: Double = 0
    nonisolated(unsafe) public static var profSliceWrslice: Double = 0
    nonisolated(unsafe) public static var profSliceDequant: Double = 0
    nonisolated(unsafe) public static var profSliceQ80: Double = 0
    nonisolated(unsafe) public static var profSliceQ2K: Double = 0
    nonisolated(unsafe) public static var profSliceType: [String: Int] = [:]

    public func dequantExpertSlice(
        named: String, expertIdx: Int, nExperts: Int,
        slot: String = "default",
        outDtype: DType? = nil, device: Device = .shared
    ) throws -> Tensor {
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        // Expert e's slice is the e-th out of `nExperts` equal-sized
        // chunks of the tensor (slowest GGUF dim = n_experts).
        let nValuesTotal = Int(info.numElements)
        let nValuesPerExpert = nValuesTotal / nExperts
        let byteStart = (Int(info.byteLength) / nExperts) * expertIdx
        let byteLen = Int(info.byteLength) / nExperts
        let dtOut = outDtype ?? .f32
        let outShape = info.dimensions.dropLast().map { Int($0) }  // shape minus n_experts axis
        switch info.type {
        case .q8_0:
            let out = Self.pooledDequantOutput(
                nValues: nValuesPerExpert, dtype: dtOut, device: device,
                tagSuffix: "expert_\(slot)")
            let cmd = device.makeCommandBuffer()
            try reader.withRawBytesSlice(named: named, byteStart: byteStart, byteLength: byteLen) { ptr in
                let zeroCopy = Data(
                    bytesNoCopy: UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                    count: ptr.count, deallocator: .none)
                _ = GGUFDequant.dequantQ8_0(
                    rawBlocks: zeroCopy, nValues: nValuesPerExpert, outDtype: dtOut,
                    on: cmd, device: device, into: out)
            }
            cmd.commit()
            cmd.waitUntilCompleted()
            return out.reshaped(to: outShape)
        case .q2_K:
            let out = Self.pooledDequantOutput(
                nValues: nValuesPerExpert, dtype: dtOut, device: device,
                tagSuffix: "expert_\(slot)")
            let cmd = device.makeCommandBuffer()
            try reader.withRawBytesSlice(named: named, byteStart: byteStart, byteLength: byteLen) { ptr in
                let zeroCopy = Data(
                    bytesNoCopy: UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                    count: ptr.count, deallocator: .none)
                _ = GGUFDequant.dequantQ2_K(
                    rawBlocks: zeroCopy, nValues: nValuesPerExpert, outDtype: dtOut,
                    on: cmd, device: device, into: out)
            }
            cmd.commit()
            cmd.waitUntilCompleted()
            return out.reshaped(to: outShape)
        case .iq2_xxs:
            let (grid, signs) = GGUFDequant.iq2xxsTables(device: device)
            let out = Self.pooledDequantOutput(
                nValues: nValuesPerExpert, dtype: dtOut, device: device,
                tagSuffix: "expert_\(slot)")
            let cmd = device.makeCommandBuffer()
            try reader.withRawBytesSlice(named: named, byteStart: byteStart, byteLength: byteLen) { ptr in
                let zeroCopy = Data(
                    bytesNoCopy: UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                    count: ptr.count, deallocator: .none)
                _ = GGUFDequant.dequantIQ2_XXS(
                    rawBlocks: zeroCopy, nValues: nValuesPerExpert, outDtype: dtOut,
                    gridTensor: grid, signsTensor: signs,
                    on: cmd, device: device, into: out)
            }
            cmd.commit()
            cmd.waitUntilCompleted()
            return out.reshaped(to: outShape)
        default:
            throw GGUFError.unsupportedDequant(info.type, tensor: named)
        }
    }

    /// Pre-allocated dequant output buffer keyed by (dtype, nValues).
    /// All layer-load calls with the SAME shape + dtype reuse the
    /// SAME MTLBuffer — Metal's driver pool wasn't recycling fresh
    /// 4 GB IQ2_XXS expert-tensor allocations efficiently (~1.5 GB
    /// stuck per layer). Caller must commit + wait the dequant cmd
    /// buffer before requesting the same (dtype, nValues) again
    /// (which the per-call `cmd.commit + waitUntilCompleted` here
    /// ensures), AND must finish any forward work that consumed the
    /// previous layer's buffer before loading the next layer.
    private static func pooledDequantOutput(
        nValues: Int, dtype: DType, device: Device,
        tagSuffix: String = "full"
    ) -> Tensor {
        let bytes = nValues * dtype.byteSize
        // Distinct tag suffix for per-expert slice outputs so they
        // don't collide with the full-tensor outputs at the same
        // shape — keeps `Layer.ffnGateExps` (full 3D pool) and the
        // transient per-expert slice in different slabs.
        let tag = "dequant_out_\(dtype)_\(nValues)_\(tagSuffix)"
        let buf = device.intermediateScratch(tag: tag, minBytes: bytes)
        return Tensor(buffer: buf, offset: 0, shape: [nValues], dtype: dtype)
    }

    /// CPU-side dtype conversion for the small float tensors
    /// (norms, sinks, biases) where the GGUF on-disk dtype differs
    /// from the caller's requested activation dtype. Goes via f32
    /// then narrows; bf16 isn't covered yet (only emerges as a
    /// destination once a model needs bf16 activations and a
    /// dedicated f32-bytes → bf16-bytes helper lands).
    private static func convertHalfPrecisionTensor(
        raw: Data, srcDtype: DType, dstDtype: DType,
        shape: [Int], device: Device
    ) throws -> Tensor {
        // Step 1: decode raw → [Float] in f32.
        var f32s: [Float]
        switch srcDtype {
        case .f32:
            f32s = raw.withUnsafeBytes { rawBuf in
                Array(rawBuf.bindMemory(to: Float.self))
            }
        case .f16:
            f32s = raw.withUnsafeBytes { rawBuf in
                rawBuf.bindMemory(to: Float16.self).map { Float($0) }
            }
        case .bf16:
            f32s = raw.withUnsafeBytes { rawBuf in
                rawBuf.bindMemory(to: UInt16.self).map { bits in
                    Float(bitPattern: UInt32(bits) << 16)
                }
            }
        default:
            throw GGUFError.unsupportedDequant(.f32, tensor: "convert src \(srcDtype)")
        }
        // Step 2: encode f32 → dst bytes.
        let outByteCount = f32s.count * dstDtype.byteSize
        let buf = device.makeBuffer(length: max(outByteCount, dstDtype.byteSize))
        switch dstDtype {
        case .f32:
            buf.contents().assumingMemoryBound(to: Float.self)
                .update(from: &f32s, count: f32s.count)
        case .f16:
            var f16s: [Float16] = f32s.map { Float16($0) }
            buf.contents().assumingMemoryBound(to: Float16.self)
                .update(from: &f16s, count: f16s.count)
        case .bf16:
            var bf16s: [UInt16] = f32s.map { v in
                let bits = v.bitPattern
                // Round-to-nearest-even truncation: add bias before
                // shifting so the round-half-to-even tie-break is
                // approximated (matches PyTorch's bf16 cast).
                let lsb = (bits >> 16) & 1
                let rounded = bits + 0x7FFF + lsb
                return UInt16(rounded >> 16)
            }
            buf.contents().assumingMemoryBound(to: UInt16.self)
                .update(from: &bf16s, count: bf16s.count)
        default:
            throw GGUFError.unsupportedDequant(.f32, tensor: "convert dst \(dstDtype)")
        }
        return Tensor(buffer: buf, offset: 0, shape: shape, dtype: dstDtype)
    }

    // ─── Architecture-introspection helpers ──────────────────────────

    /// `general.architecture` — what the loader's family dispatch
    /// switches on. Returns `nil` if the metadata key is missing
    /// (malformed GGUF).
    public var architecture: String? {
        reader.metadataString("general.architecture")
    }

    /// `general.name` — model display name (e.g.
    /// "DeepSeek V4 Flash"). Optional.
    public var modelName: String? {
        reader.metadataString("general.name")
    }

    /// Build a swift-transformers `Tokenizer` from the embedded
    /// `tokenizer.ggml.*` metadata. Throws when the embedded
    /// tokenizer kind isn't a BPE-family variant the adapter knows
    /// how to translate.
    public func tokenizer() throws -> any Tokenizers.Tokenizer {
        try GGUFTokenizerAdapter.build(reader: reader)
    }
}
