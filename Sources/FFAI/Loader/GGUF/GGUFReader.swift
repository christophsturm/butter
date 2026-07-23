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
// GGUF v3 file reader — header + metadata KV + tensor info table.
//
// All scalars are little-endian. Parses lazily where possible: the
// tensor info table is decoded eagerly (small + needed for the load
// dispatch), but tensor data stays mmap'd; the loader copies / dequants
// individual tensors on demand.
//
// Adapts the canonical GGUF v3 binary spec. Pure Swift; no FFI.

import Foundation

/// One opened GGUF file — header + metadata KV + tensor info, plus a
/// memory-mapped handle to the raw bytes for on-demand tensor reads.
public final class GGUFReader {
    /// Backing file URL.
    public let url: URL
    /// File version (must be 3 — earlier versions throw at parse).
    public let version: UInt32
    /// Tensor-data section alignment (default 32, may be overridden by
    /// the `general.alignment` metadata key).
    public let alignment: UInt64
    /// Tensor-data section absolute file offset.
    public let tensorDataOffset: UInt64
    /// Metadata KV block.
    public let metadata: [String: GGUFValue]
    /// Tensor info table (ordered as stored on disk).
    public let tensorInfos: [GGUFTensorInfo]
    /// Name → index into `tensorInfos` for O(1) lookup.
    public let tensorIndex: [String: Int]
    /// Memory-mapped backing data. Held to keep the mapping alive
    /// for the lifetime of any tensor read.
    private let mapped: Data

    /// Stable base pointer of the mmap'd file. Valid for the reader's
    /// lifetime (the `mapped` Data holds the mapping; mmap pages don't
    /// move). Used to wrap zero-copy GPU views over the EXISTING mapping
    /// — never a second mmap (that would double resident memory).
    /// Returns nil if the Data isn't a contiguous mmap (e.g. tiny test
    /// Data); callers fall back to the streaming read path.
    public lazy var mmapBase: UnsafeRawPointer? = {
        return mapped.withUnsafeBytes { $0.baseAddress }
    }()
    public var mmapByteCount: Int { mapped.count }

    /// When true, skip the post-read `MADV_FREE` that evicts mmap
    /// pages from RSS. For a model that fits resident (84 GB on a
    /// 128 GB Mac), evicting expert quant pages after each read forces
    /// every subsequent token to re-fault ~3 GB from disk, collapsing
    /// steady-state decode to ~2 tps. Keep pages resident instead.
    /// Default ON (resident) — set FFAI_MADV_FREE=1 to restore the
    /// streaming behaviour for memory-constrained runs.
    static let keepResident: Bool =
        ProcessInfo.processInfo.environment["FFAI_MADV_FREE"] != "1"

    // ─── Init ──────────────────────────────────────────────────────────

    public convenience init(url: URL) throws {
        // `.mappedIfSafe` lets Foundation fall back to a regular read
        // if the filesystem doesn't support mmap (network shares,
        // some FUSE mounts). Worst case is a slow first read; we
        // tolerate it.
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        try self.init(url: url, data: data)
    }

    /// In-memory init — useful for tests that synthesise a GGUF
    /// header in `Data` without writing a temp file.
    public init(url: URL, data: Data) throws {
        self.url = url
        self.mapped = data
        var cursor = GGUFCursor(data: data)

        // ── Header ──
        let magic = try cursor.readBytes(GGUFConstants.magic.count, at: "magic")
        guard magic == GGUFConstants.magic else {
            throw GGUFError.badMagic
        }
        let version: UInt32 = try cursor.readLE(at: "version")
        guard version == GGUFConstants.supportedVersion else {
            throw GGUFError.unsupportedVersion(version)
        }
        self.version = version
        let tensorCount: UInt64 = try cursor.readLE(at: "tensor_count")
        let metadataCount: UInt64 = try cursor.readLE(at: "metadata_kv_count")

        // ── Metadata KV block ──
        var metadata: [String: GGUFValue] = [:]
        metadata.reserveCapacity(Int(metadataCount))
        for _ in 0..<metadataCount {
            let key = try cursor.readString(at: "metadata key")
            if metadata[key] != nil {
                throw GGUFError.duplicateKey(key)
            }
            let value = try GGUFReader.readValue(cursor: &cursor, key: key)
            metadata[key] = value
        }
        self.metadata = metadata

        // Optional override of the tensor-data section alignment.
        var alignment = GGUFConstants.defaultAlignment
        if case .uint32(let v) = metadata["general.alignment"] {
            alignment = UInt64(v)
        }
        self.alignment = alignment

        // ── Tensor info table ──
        var infos: [GGUFTensorInfo] = []
        infos.reserveCapacity(Int(tensorCount))
        var seenNames = Set<String>()
        seenNames.reserveCapacity(Int(tensorCount))
        var index: [String: Int] = [:]
        index.reserveCapacity(Int(tensorCount))
        for i in 0..<tensorCount {
            let name = try cursor.readString(at: "tensor name")
            if !seenNames.insert(name).inserted {
                throw GGUFError.duplicateTensorName(name)
            }
            let nDims: UInt32 = try cursor.readLE(at: "tensor n_dims")
            var dims: [UInt64] = []
            dims.reserveCapacity(Int(nDims))
            for _ in 0..<nDims {
                dims.append(try cursor.readLE(at: "tensor dim"))
            }
            let typeTag: UInt32 = try cursor.readLE(at: "tensor type")
            guard let type = GGUFTensorType(rawValue: typeTag) else {
                throw GGUFError.unknownTensorType(typeTag, tensor: name)
            }
            let dataOffset: UInt64 = try cursor.readLE(at: "tensor data offset")
            infos.append(
                GGUFTensorInfo(name: name, dimensions: dims, type: type, dataOffset: dataOffset)
            )
            index[name] = Int(i)
        }
        self.tensorInfos = infos
        self.tensorIndex = index

        // ── Padding to alignment boundary ──
        // The tensor-data section starts at the next `alignment`-aligned
        // file offset after the tensor info table. Compute it from the
        // cursor's current position; the padding bytes themselves are
        // not validated (some writers leave garbage, others write
        // zeros — either is spec-conformant).
        let here = UInt64(cursor.offset)
        self.tensorDataOffset = ((here + alignment - 1) / alignment) * alignment
    }

    // ─── Tensor data read ─────────────────────────────────────────────

    /// Return the raw on-disk bytes for tensor `name`. The returned
    /// **Use `withRawBytes` instead for the heavy dequant path** — this
    /// API calls `Data.subdata` which COPIES for slices ≥16 KB, paging
    /// the entire mmap range into anonymous RAM. With 1.5 GB of raw
    /// quant blocks per DSv4 layer × 43 layers, the copies stack up to
    /// ~70 GB of duplicated RSS that the OS file-cache eviction
    /// can't reclaim. Kept for callers that need a standalone Data
    /// handle (e.g., metadata-string reads where the slice is tiny).
    public func rawBytes(named name: String) throws -> Data {
        guard let idx = tensorIndex[name] else {
            throw GGUFError.missingMetadataKey("tensor:\(name)")
        }
        let info = tensorInfos[idx]
        let start = Int(tensorDataOffset + info.dataOffset)
        let end = start + info.byteLength
        return mapped.subdata(in: start..<end)
    }

    /// Zero-copy access to a CUSTOM BYTE RANGE inside a tensor's
    /// region of the mmap. Used for per-expert slice dequant of
    /// `[hidden, intermediate, n_experts]` MoE expert tensors —
    /// each expert is a contiguous sub-range of the full tensor's
    /// byte block (slowest GGUF dim is the n_experts axis).
    /// Async readahead hint for a whole tensor's mmap pages
    /// (`MADV_WILLNEED`). Fire-and-forget: it schedules the kernel to
    /// fault the pages from disk into the page cache WITHOUT copying any
    /// bytes itself — so calling it on a background thread during GPU
    /// compute overlaps the cold SSD I/O (the dominant cost of the first
    /// expert gather, which runs at ~11 GB/s = latency-bound on cold page
    /// faults) WITHOUT contending for the unified-memory bandwidth the way
    /// a background memcpy does (that regressed 2×). The subsequent
    /// synchronous gather then hits warm pages. No-op if the pages are
    /// already resident. Safe to race the real read (advisory only).
    public func prefetchTensor(named name: String) {
        guard let idx = tensorIndex[name] else { return }
        let info = tensorInfos[idx]
        let start = Int(tensorDataOffset + info.dataOffset)
        let length = Int(info.byteLength)
        mapped.withUnsafeBytes { raw in
            guard let b = raw.baseAddress else { return }
            let base = b.advanced(by: start)
            let pageMask = Int(getpagesize()) - 1
            let aStart = Int(bitPattern: base) & ~pageMask
            let aEnd = (Int(bitPattern: base) + length + pageMask) & ~pageMask
            _ = madvise(UnsafeMutableRawPointer(bitPattern: aStart), aEnd - aStart, MADV_WILLNEED)
            // Force-fault: madvise(WILLNEED) is too passive on macOS — the pages
            // evict before the synchronous gather memcpy reuses them. Touch one
            // byte per page to actually pull them into the page cache now (on the
            // background prefetch thread, overlapping the current layer's NAX
            // compute which is disk-idle). Volatile sink so it isn't optimized out.
            let pg = pageMask + 1
            if let p = UnsafePointer<UInt8>(bitPattern: aStart) {
                let total = aEnd - aStart
                var acc: UInt8 = 0, off = 0
                while off < total { acc = acc &+ p[off]; off += pg }
                Self.prefetchSink = acc
            }
        }
    }
    nonisolated(unsafe) static var prefetchSink: UInt8 = 0

    public func withRawBytesSlice<T>(
        named name: String, byteStart relStart: Int, byteLength: Int,
        _ body: (UnsafeBufferPointer<UInt8>) throws -> T
    ) throws -> T {
        guard let idx = tensorIndex[name] else {
            throw GGUFError.missingMetadataKey("tensor:\(name)")
        }
        let info = tensorInfos[idx]
        precondition(
            relStart + byteLength <= info.byteLength,
            "withRawBytesSlice: range overflows tensor (start=\(relStart) + len=\(byteLength) > total=\(info.byteLength))")
        let start = Int(tensorDataOffset + info.dataOffset) + relStart
        let length = byteLength
        return try mapped.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!.advanced(by: start)
            let result = try body(UnsafeBufferPointer<UInt8>(start: base, count: length))
            if !Self.keepResident {
                let pageMask = Int(getpagesize()) - 1
                let pageAlignedStart = Int(bitPattern: UnsafeRawPointer(base)) & ~pageMask
                let endAddr = Int(bitPattern: UnsafeRawPointer(base)) + length
                let pageAlignedEnd = (endAddr + pageMask) & ~pageMask
                let advLen = pageAlignedEnd - pageAlignedStart
                _ = madvise(
                    UnsafeMutableRawPointer(bitPattern: pageAlignedStart),
                    advLen, MADV_FREE)
            }
            return result
        }
    }

    /// Zero-copy access to a tensor's raw bytes via a closure. The
    /// closure receives an `UnsafeBufferPointer<UInt8>` pointing INTO
    /// the original mmap — no copy, no anonymous RAM allocation. The
    /// pointer is valid only inside the closure scope.
    ///
    /// After the closure returns, the mmap pages are advised to the
    /// kernel as `MADV_DONTNEED` — for layer-streaming forward, each
    /// layer's ~1.5 GB of raw quant blocks is read once then never
    /// touched again, so keeping those pages resident is pure RSS
    /// pressure. The kernel reclaims them lazily under memory
    /// pressure even without the hint, but the explicit hint lets us
    /// stay well under the 128 GB unified-memory ceiling during the
    /// 43-layer forward.
    public func withRawBytes<T>(
        named name: String, _ body: (UnsafeBufferPointer<UInt8>) throws -> T
    ) throws -> T {
        guard let idx = tensorIndex[name] else {
            throw GGUFError.missingMetadataKey("tensor:\(name)")
        }
        let info = tensorInfos[idx]
        let start = Int(tensorDataOffset + info.dataOffset)
        let length = info.byteLength
        return try mapped.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!.advanced(by: start)
            let result = try body(UnsafeBufferPointer<UInt8>(start: base, count: length))
            // Darwin: MADV_FREE tells the kernel these pages are
            // safely reclaimable. Unlike POSIX_MADV_DONTNEED (which
            // is a no-op on Darwin), MADV_FREE actually evicts the
            // pages from RSS — important when streaming 43 layers
            // through a 86 GB GGUF on a 128 GB Mac, where holding
            // every layer's 1.5 GB of raw quant pages resident would
            // overflow before reaching the LM head.
            if !Self.keepResident {
                let pageMask = Int(getpagesize()) - 1
                let pageAlignedStart = Int(bitPattern: UnsafeRawPointer(base)) & ~pageMask
                let endAddr = Int(bitPattern: UnsafeRawPointer(base)) + length
                let pageAlignedEnd = (endAddr + pageMask) & ~pageMask
                let advLen = pageAlignedEnd - pageAlignedStart
                _ = madvise(
                    UnsafeMutableRawPointer(bitPattern: pageAlignedStart),
                    advLen, MADV_FREE)
            }
            return result
        }
    }

    /// Convenience: get a metadata value, casted to a specific type.
    /// Returns nil if absent or the type doesn't match.
    public func metadataString(_ key: String) -> String? {
        if case .string(let s) = metadata[key] { return s }
        return nil
    }

    public func metadataUInt32(_ key: String) -> UInt32? {
        switch metadata[key] {
        case .uint32(let v): return v
        case .int32(let v) where v >= 0: return UInt32(v)
        case .uint64(let v) where v <= UInt32.max: return UInt32(v)
        default: return nil
        }
    }

    public func metadataFloat(_ key: String) -> Float? {
        switch metadata[key] {
        case .float32(let v): return v
        case .float64(let v): return Float(v)
        default: return nil
        }
    }

    public func metadataBool(_ key: String) -> Bool? {
        if case .bool(let b) = metadata[key] { return b }
        return nil
    }

    public func metadataStringArray(_ key: String) -> [String]? {
        if case .array(.string(let arr)) = metadata[key] { return arr }
        return nil
    }

    /// Integer array accessor — coerces any of the integer-typed GGUF
    /// array kinds (i32 / u32 / i64 / u64 / i16 / u16 / i8 / u8) to
    /// `[Int]`. Used for per-layer parameter arrays like
    /// `deepseek4.attention.compress_ratios`.
    public func metadataIntArray(_ key: String) -> [Int]? {
        switch metadata[key] {
        case .array(.int32(let a)): return a.map { Int($0) }
        case .array(.uint32(let a)): return a.map { Int($0) }
        case .array(.int64(let a)): return a.map { Int($0) }
        case .array(.uint64(let a)): return a.map { Int($0) }
        case .array(.int16(let a)): return a.map { Int($0) }
        case .array(.uint16(let a)): return a.map { Int($0) }
        case .array(.int8(let a)): return a.map { Int($0) }
        case .array(.uint8(let a)): return a.map { Int($0) }
        default: return nil
        }
    }

    // ─── Value-type decoder (internal) ────────────────────────────────

    private static func readValue(cursor: inout GGUFCursor, key: String) throws -> GGUFValue {
        let tag: UInt32 = try cursor.readLE(at: "value-type tag (key=\(key))")
        guard let kind = GGUFValueType(rawValue: tag) else {
            throw GGUFError.unknownValueType(tag, key: key)
        }
        return try readScalarOrArray(cursor: &cursor, kind: kind, key: key)
    }

    private static func readScalarOrArray(
        cursor: inout GGUFCursor, kind: GGUFValueType, key: String
    ) throws -> GGUFValue {
        switch kind {
        case .uint8: return .uint8(try cursor.readLE(at: "u8 (\(key))"))
        case .int8: return .int8(Int8(bitPattern: try cursor.readLE(at: "i8 (\(key))")))
        case .uint16: return .uint16(try cursor.readLE(at: "u16 (\(key))"))
        case .int16:
            let raw: UInt16 = try cursor.readLE(at: "i16 (\(key))")
            return .int16(Int16(bitPattern: raw))
        case .uint32: return .uint32(try cursor.readLE(at: "u32 (\(key))"))
        case .int32:
            let raw: UInt32 = try cursor.readLE(at: "i32 (\(key))")
            return .int32(Int32(bitPattern: raw))
        case .uint64: return .uint64(try cursor.readLE(at: "u64 (\(key))"))
        case .int64:
            let raw: UInt64 = try cursor.readLE(at: "i64 (\(key))")
            return .int64(Int64(bitPattern: raw))
        case .float32:
            let raw: UInt32 = try cursor.readLE(at: "f32 (\(key))")
            return .float32(Float(bitPattern: raw))
        case .float64:
            let raw: UInt64 = try cursor.readLE(at: "f64 (\(key))")
            return .float64(Double(bitPattern: raw))
        case .bool:
            let b: UInt8 = try cursor.readLE(at: "bool (\(key))")
            return .bool(b != 0)
        case .string:
            return .string(try cursor.readString(at: "string (\(key))"))
        case .array:
            let elemTag: UInt32 = try cursor.readLE(at: "array-elem-type (\(key))")
            guard let elemKind = GGUFValueType(rawValue: elemTag) else {
                throw GGUFError.unknownValueType(elemTag, key: "\(key)[]")
            }
            let n: UInt64 = try cursor.readLE(at: "array-len (\(key))")
            return .array(try readArrayElements(cursor: &cursor, kind: elemKind, count: n, key: key))
        }
    }

    private static func readArrayElements(
        cursor: inout GGUFCursor, kind: GGUFValueType, count: UInt64, key: String
    ) throws -> GGUFArrayValue {
        let n = Int(count)
        switch kind {
        case .uint8:
            var out = [UInt8](); out.reserveCapacity(n)
            for _ in 0..<n { out.append(try cursor.readLE(at: "u8[] (\(key))")) }
            return .uint8(out)
        case .int8:
            var out = [Int8](); out.reserveCapacity(n)
            for _ in 0..<n {
                out.append(Int8(bitPattern: try cursor.readLE(at: "i8[] (\(key))")))
            }
            return .int8(out)
        case .uint16:
            var out = [UInt16](); out.reserveCapacity(n)
            for _ in 0..<n { out.append(try cursor.readLE(at: "u16[] (\(key))")) }
            return .uint16(out)
        case .int16:
            var out = [Int16](); out.reserveCapacity(n)
            for _ in 0..<n {
                let raw: UInt16 = try cursor.readLE(at: "i16[] (\(key))")
                out.append(Int16(bitPattern: raw))
            }
            return .int16(out)
        case .uint32:
            var out = [UInt32](); out.reserveCapacity(n)
            for _ in 0..<n { out.append(try cursor.readLE(at: "u32[] (\(key))")) }
            return .uint32(out)
        case .int32:
            var out = [Int32](); out.reserveCapacity(n)
            for _ in 0..<n {
                let raw: UInt32 = try cursor.readLE(at: "i32[] (\(key))")
                out.append(Int32(bitPattern: raw))
            }
            return .int32(out)
        case .uint64:
            var out = [UInt64](); out.reserveCapacity(n)
            for _ in 0..<n { out.append(try cursor.readLE(at: "u64[] (\(key))")) }
            return .uint64(out)
        case .int64:
            var out = [Int64](); out.reserveCapacity(n)
            for _ in 0..<n {
                let raw: UInt64 = try cursor.readLE(at: "i64[] (\(key))")
                out.append(Int64(bitPattern: raw))
            }
            return .int64(out)
        case .float32:
            var out = [Float](); out.reserveCapacity(n)
            for _ in 0..<n {
                let raw: UInt32 = try cursor.readLE(at: "f32[] (\(key))")
                out.append(Float(bitPattern: raw))
            }
            return .float32(out)
        case .float64:
            var out = [Double](); out.reserveCapacity(n)
            for _ in 0..<n {
                let raw: UInt64 = try cursor.readLE(at: "f64[] (\(key))")
                out.append(Double(bitPattern: raw))
            }
            return .float64(out)
        case .bool:
            var out = [Bool](); out.reserveCapacity(n)
            for _ in 0..<n {
                let b: UInt8 = try cursor.readLE(at: "bool[] (\(key))")
                out.append(b != 0)
            }
            return .bool(out)
        case .string:
            var out = [String](); out.reserveCapacity(n)
            for _ in 0..<n { out.append(try cursor.readString(at: "string[] (\(key))")) }
            return .string(out)
        case .array:
            // Nested arrays are not supported by GGUF v3. Future-proof
            // by throwing — the spec marks array-of-array as reserved.
            throw GGUFError.unknownValueType(
                GGUFValueType.array.rawValue, key: "\(key)[] (nested array)"
            )
        }
    }
}

// ─── Cursor — bounds-checked LE reader ───────────────────────────────

/// Forward-only byte cursor with bounds checking. Crashes-as-errors:
/// every off-the-end read throws `GGUFError.truncated`. Used only
/// during the parse phase (the read-back path uses direct slicing).
struct GGUFCursor {
    let data: Data
    var offset: Int

    init(data: Data) {
        self.data = data
        self.offset = 0
    }

    mutating func readBytes(_ count: Int, at where_: String) throws -> [UInt8] {
        guard offset + count <= data.count else {
            throw GGUFError.truncated(at: where_)
        }
        let slice = data[offset..<offset + count]
        offset += count
        return Array(slice)
    }

    mutating func readLE<T: FixedWidthInteger & UnsignedInteger>(at where_: String) throws -> T {
        let bytes = MemoryLayout<T>.size
        guard offset + bytes <= data.count else {
            throw GGUFError.truncated(at: where_)
        }
        var value: T = 0
        for i in 0..<bytes {
            value |= T(data[offset + i]) << (8 * i)
        }
        offset += bytes
        return value
    }

    mutating func readString(at where_: String) throws -> String {
        let length: UInt64 = try readLE(at: "\(where_) length")
        let bytes = try readBytes(Int(length), at: where_)
        guard let s = String(bytes: bytes, encoding: .utf8) else {
            throw GGUFError.stringNotUTF8(at: where_)
        }
        return s
    }
}
