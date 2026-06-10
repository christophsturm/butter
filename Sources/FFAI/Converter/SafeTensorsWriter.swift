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
// SafeTensorsWriter — pure-Swift writer for the safetensors v2 format.
//
// Mirrors exactly the format that SafeTensors.swift (the loader) reads:
//
//   [8 bytes: u64-LE header length N]
//   [N bytes: UTF-8 JSON header]
//   [tensor data, contiguous, in append order]
//
// JSON schema (written with keys sorted alphabetically, matching mlx-lm
// convention so diff tooling produces stable output):
//
//   {
//     "__metadata__": { "format": "pt", "ffai_version": "..." },
//     "tensor_name": {
//       "data_offsets": [start, end],
//       "dtype": "BF16" | "F16" | "F32" | "U32" | ...,
//       "shape": [d0, d1, ...]
//     }
//   }
//
// Usage:
//
//   let w = SafeTensorsWriter(url: outputURL)
//   w.append(name: "a.weight", dtype: .bf16, shape: [16, 64], bytes: rawData)
//   try w.finalize()

import Foundation
import Metal

/// Errors surfaced by SafeTensorsWriter.
public enum SafeTensorsWriterError: Error, CustomStringConvertible {
    case emptyTensor(String)
    case duplicateName(String)
    case jsonSerializationFailed
    case writeFailed(URL, Error)

    public var description: String {
        switch self {
        case .emptyTensor(let n):
            return "SafeTensorsWriter: tensor \"\(n)\" has zero bytes — cannot write empty tensor"
        case .duplicateName(let n):
            return "SafeTensorsWriter: tensor \"\(n)\" appended twice"
        case .jsonSerializationFailed:
            return "SafeTensorsWriter: JSON header serialization failed"
        case .writeFailed(let u, let e):
            return "SafeTensorsWriter: write to \(u.path) failed: \(e)"
        }
    }
}

/// Accumulates tensors in memory and writes them as a single safetensors file.
///
/// Not thread-safe — all `append` calls must come from a single thread
/// before `finalize` is called.
public final class SafeTensorsWriter {
    private let url: URL

    // Each entry records: name, dtype, shape, raw bytes. We buffer all
    // tensor data in-memory so we can compute data_offsets before writing
    // the header. For a 7B model at int4 this is ~3.5 GB RAM — acceptable
    // for a one-shot convert workflow; not suitable for streaming.
    private struct Pending {
        let name: String
        let dtype: DType
        let shape: [Int]
        let bytes: Data
    }

    private var pending: [Pending] = []
    private var usedNames: Set<String> = []

    /// Soft cap on a single output shard's tensor-data bytes. When the
    /// accumulated tensors exceed this, `finalize` splits them across
    /// `model-0000i-of-0000N.safetensors` shards + a `*.index.json` map
    /// (the HF / mlx-community convention). A single tensor larger than the
    /// cap still gets its own shard. Default 5 GB (matches `transformers`'
    /// `max_shard_size="5GB"`). Set to `.max` to force a single file.
    private let maxShardBytes: Int

    /// Default per-shard cap — 5 GB, the HF/transformers/mlx-community
    /// convention for sharded safetensors checkpoints.
    public static let defaultMaxShardBytes = 5_000_000_000

    public init(url: URL, maxShardBytes: Int = SafeTensorsWriter.defaultMaxShardBytes) {
        self.url = url
        self.maxShardBytes = maxShardBytes
    }

    /// Accumulate one tensor. `bytes` must be raw little-endian tensor data
    /// in `dtype` layout. The tensor is buffered in memory until `finalize`.
    public func append(name: String, dtype: DType, shape: [Int], bytes: Data) throws {
        guard !bytes.isEmpty else { throw SafeTensorsWriterError.emptyTensor(name) }
        guard !usedNames.contains(name) else { throw SafeTensorsWriterError.duplicateName(name) }
        usedNames.insert(name)
        pending.append(Pending(name: name, dtype: dtype, shape: shape, bytes: bytes))
    }

    /// Write the safetensors output. If the accumulated tensor data fits in
    /// one shard (≤ `maxShardBytes`) a single file is written at `url`
    /// (original behavior). Otherwise the tensors are split across
    /// `model-0000i-of-0000N.safetensors` shards plus a
    /// `model.safetensors.index.json` weight-map, the HF/mlx-community
    /// convention. Clears the pending queue after writing.
    public func finalize() throws {
        let stem = url.deletingPathExtension().lastPathComponent  // "model"
        let ext = url.pathExtension.isEmpty ? "safetensors" : url.pathExtension
        let dir = url.deletingLastPathComponent()

        // Pick the fewest shards that keep every shard ≤ maxShardBytes, then
        // split the tensors **evenly** across that many shards (by cumulative
        // byte position) so they come out ~uniform — rather than several full
        // shards plus a small tail. Each tensor is placed whole into the shard
        // its cumulative-start falls in, so a shard's size is ≈ total/count
        // plus at most one tensor's slack (well under maxShardBytes for any
        // LLM checkpoint, whose largest tensor is far smaller than a shard).
        let totalBytes = pending.reduce(0) { $0 + $1.bytes.count }
        let shardCount = max(1, (totalBytes + maxShardBytes - 1) / maxShardBytes)

        var shards: [[Pending]]
        if shardCount <= 1 || totalBytes == 0 {
            shards = [pending]
        } else {
            var buckets: [[Pending]] = Array(repeating: [], count: shardCount)
            var cumulative = 0
            for p in pending {
                // floor(cumulative / (totalBytes / shardCount)), integer math.
                var s = cumulative * shardCount / totalBytes
                if s >= shardCount { s = shardCount - 1 }
                buckets[s].append(p)
                cumulative += p.bytes.count
            }
            shards = buckets.filter { !$0.isEmpty }
        }

        if shards.count <= 1 {
            // Single file at `url` — no index needed (loaders treat a lone
            // `model.safetensors` as the whole checkpoint).
            try writeShardFile(url: url, tensors: pending)
        } else {
            let n = shards.count
            var weightMap: [String: String] = [:]
            for (i, shard) in shards.enumerated() {
                let name = String(format: "%@-%05d-of-%05d.%@", stem, i + 1, n, ext)
                try writeShardFile(url: dir.appendingPathComponent(name), tensors: shard)
                for p in shard { weightMap[p.name] = name }
            }
            try writeIndex(
                at: dir.appendingPathComponent("\(stem).\(ext).index.json"),
                weightMap: weightMap, totalSize: totalBytes)
        }

        pending.removeAll()
        usedNames.removeAll()
    }

    // ─── Helpers ────────────────────────────────────────────────────────

    /// Serialize one safetensors file (`[u64 header len][JSON header][data]`)
    /// for `tensors`, with data_offsets relative to this file. Atomic write.
    private func writeShardFile(url: URL, tensors: [Pending]) throws {
        var headerDict: [String: Any] = [:]
        // __metadata__ sorts to the top ("_" precedes letters in ASCII).
        headerDict["__metadata__"] = [
            "format": "pt",
            "ffai_version": FFAI.version,
        ]
        var offset = 0
        for p in tensors {
            let start = offset
            let end = offset + p.bytes.count
            headerDict[p.name] =
                [
                    // Keys sorted: data_offsets < dtype < shape
                    "data_offsets": [start, end],
                    "dtype": safeTensorsDTypeName(p.dtype),
                    "shape": p.shape,
                ] as [String: Any]
            offset = end
        }

        let jsonData: Data
        do {
            jsonData = try JSONSerialization.data(
                withJSONObject: headerDict, options: [.sortedKeys])
        } catch {
            throw SafeTensorsWriterError.jsonSerializationFailed
        }

        // [8 bytes u64-LE header length][JSON][tensor data, contiguous]
        var file = Data()
        var headerLen = UInt64(jsonData.count)
        withUnsafeBytes(of: &headerLen) { file.append(contentsOf: $0) }
        file.append(jsonData)
        for p in tensors { file.append(p.bytes) }

        do {
            try file.write(to: url, options: .atomic)
        } catch {
            throw SafeTensorsWriterError.writeFailed(url, error)
        }
    }

    /// Write the `model.safetensors.index.json` shard map: `{ "metadata":
    /// { "total_size": <data bytes> }, "weight_map": { tensor → shard } }`.
    private func writeIndex(at url: URL, weightMap: [String: String], totalSize: Int) throws {
        let index: [String: Any] = [
            "metadata": ["total_size": totalSize],
            "weight_map": weightMap,
        ]
        let data: Data
        do {
            data = try JSONSerialization.data(
                withJSONObject: index, options: [.sortedKeys, .prettyPrinted])
        } catch {
            throw SafeTensorsWriterError.jsonSerializationFailed
        }
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw SafeTensorsWriterError.writeFailed(url, error)
        }
    }

    /// Map DType to the uppercase safetensors dtype tag that the loader
    /// (SafeTensors.swift → DType.fromSafeTensors) reads back correctly.
    private func safeTensorsDTypeName(_ dtype: DType) -> String {
        switch dtype {
        case .f32: return "F32"
        case .f16: return "F16"
        case .bf16: return "BF16"
        case .i32: return "I32"
        case .u32: return "U32"
        case .i8: return "I8"
        case .u8: return "U8"
        case .i64: return "I64"
        case .u64: return "U64"
        }
    }
}
