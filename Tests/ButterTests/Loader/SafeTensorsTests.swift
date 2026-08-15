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
import Foundation
import Testing

@testable import ButterSwift

@Suite("SafeTensors")
struct SafeTensorsTests {
    /// Build a minimal valid safetensors file in `directory` containing one
    /// f32 tensor named `tensorName` with the given values + shape.
    static func writeSyntheticFile(
        directory: URL, tensorName: String,
        shape: [Int], values: [Float],
        filename: String = "model.safetensors"
    ) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let dataLen = values.count * MemoryLayout<Float>.size
        let header =
            #"{"\#(tensorName)":{"dtype":"F32","shape":\#(shape.description),"data_offsets":[0,\#(dataLen)]}}"#
        let headerBytes = Array(header.utf8)
        var headerLen = UInt64(headerBytes.count)

        var data = Data()
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(contentsOf: headerBytes)
        values.withUnsafeBufferPointer { buf in
            data.append(contentsOf: UnsafeRawBufferPointer(buf))
        }

        let url = directory.appendingPathComponent(filename)
        try data.write(to: url)
        return url
    }

    @Test("load synthetic single-tensor file")
    func loadSingleFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("butter-st-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let values: [Float] = [1, 2, 3, 4, 5, 6]
        let url = try Self.writeSyntheticFile(
            directory: dir, tensorName: "alpha.weight",
            shape: [2, 3], values: values
        )

        let f = try SafeTensorsFile(url: url)
        let t = try f.tensor(named: "alpha.weight")
        #expect(t.shape == [2, 3])
        #expect(t.dtype == .f32)
        #expect(t.toArray(as: Float.self) == values)
    }

    @Test("mmap is released at end of init (no VA leak)")
    func mmapReleasedAfterInit() throws {
        // Init copies each tensor's bytes into a new MTLBuffer; the
        // source mmap has no callers after the copy loop, so init
        // munmap's it eagerly. Previously the mapping was retained
        // until deinit — for a 10 GB quantized checkpoint that's
        // 10 GB of virtual address space held for the model's
        // entire lifetime even though nothing reads from it.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("butter-st-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = try Self.writeSyntheticFile(
            directory: dir, tensorName: "weights",
            shape: [4], values: [1, 2, 3, 4]
        )

        let f = try SafeTensorsFile(url: url)
        #expect(
            f.isMmapRetained == false,
            "init must munmap; otherwise we leak virtual address space")

        // Tensor access still works — the bytes were copied into the
        // MTLBuffer before munmap, so reads don't touch the mapping.
        let t = try f.tensor(named: "weights")
        #expect(t.toArray(as: Float.self) == [1, 2, 3, 4])
    }

    @Test("missing tensor throws")
    func missingTensor() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("butter-st-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = try Self.writeSyntheticFile(
            directory: dir, tensorName: "present", shape: [1], values: [1]
        )
        let f = try SafeTensorsFile(url: url)
        do {
            _ = try f.tensor(named: "absent")
            Issue.record("expected error")
        } catch let e as SafeTensorsError {
            if case .missingTensor = e { /* ok */
            } else {
                Issue.record("expected .missingTensor, got \(e)")
            }
        }
    }

    @Test("file-not-found throws")
    func fileNotFound() {
        do {
            _ = try SafeTensorsFile(url: URL(fileURLWithPath: "/__nope__/missing.safetensors"))
            Issue.record("expected error")
        } catch let e as SafeTensorsError {
            if case .fileNotFound = e { /* ok */
            } else {
                Issue.record("expected .fileNotFound, got \(e)")
            }
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test("bundle picks up single .safetensors via fallback glob")
    func bundleSingleFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("butter-st-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let values: [Float] = [42, 43, 44]
        _ = try Self.writeSyntheticFile(
            directory: dir, tensorName: "x", shape: [3], values: values
        )

        let bundle = try SafeTensorsBundle(directory: dir)
        #expect(bundle.has("x"))
        let t = try bundle.tensor(named: "x")
        #expect(t.toArray(as: Float.self) == values)
        #expect(bundle.allKeys == ["x"])
    }

    @Test("sharded bundle reads model.safetensors.index.json")
    func bundleSharded() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("butter-st-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Two shards
        _ = try Self.writeSyntheticFile(
            directory: dir, tensorName: "a", shape: [1], values: [10],
            filename: "model-00001-of-00002.safetensors"
        )
        _ = try Self.writeSyntheticFile(
            directory: dir, tensorName: "b", shape: [1], values: [20],
            filename: "model-00002-of-00002.safetensors"
        )
        // Index file
        let index = """
            {"weight_map": {"a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors"}}
            """
        try index.write(
            to: dir.appendingPathComponent("model.safetensors.index.json"),
            atomically: true, encoding: .utf8)

        let bundle = try SafeTensorsBundle(directory: dir)
        #expect(try bundle.tensor(named: "a").toArray(as: Float.self) == [10])
        #expect(try bundle.tensor(named: "b").toArray(as: Float.self) == [20])
        #expect(bundle.allKeys.sorted() == ["a", "b"])
    }
}

// ─── In-memory adapter (used by Qwen3xGGUFLoader) ─────────────────────

@Suite("SafeTensors in-memory adapter")
struct SafeTensorsInMemoryTests {
    /// `SafeTensorsFile.init(url:entries:)` / `SafeTensorsBundle
    /// .init(inMemory:directory:)` — no disk file — is how
    /// `Qwen3xGGUFLoader` reuses `loadLinear`/`isQuantized`/
    /// `quantizedTriplet` for a GGUF checkpoint. Verify the roundtrip
    /// (tensor lookup, quantized-triplet detection) works identically
    /// to the disk-backed path these tests exercise above.
    @Test("plain f32 tensor roundtrips through tensor(named:)")
    func plainTensorRoundtrip() throws {
        let device = Device.shared
        let t = Tensor.empty(shape: [4], dtype: .f32, device: device)
        t.copyIn(from: [Float](arrayLiteral: 1, 2, 3, 4))
        let entry = SafeTensorsFile.Entry(name: "x.weight", dtype: .f32, shape: [4], buffer: t.buffer)
        let file = SafeTensorsFile(url: URL(fileURLWithPath: "/synthetic.gguf"), entries: ["x.weight": entry])
        let bundle = SafeTensorsBundle(inMemory: file, directory: URL(fileURLWithPath: "/"))

        #expect(bundle.has("x.weight"))
        #expect(bundle.allKeys == ["x.weight"])
        let readBack = try bundle.tensor(named: "x.weight")
        #expect(readBack.toArray(as: Float.self) == [1, 2, 3, 4])
    }

    @Test("quantized triplet is detected via isQuantized/quantizedTriplet")
    func quantizedTripletRoundtrip() throws {
        let device = Device.shared
        func entry(_ name: String, dtype: DType, shape: [Int]) -> SafeTensorsFile.Entry {
            let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
            t.zero()
            return SafeTensorsFile.Entry(name: name, dtype: dtype, shape: shape, buffer: t.buffer)
        }
        let entries: [String: SafeTensorsFile.Entry] = [
            "w.weight": entry("w.weight", dtype: .u32, shape: [8, 2]),
            "w.scales": entry("w.scales", dtype: .bf16, shape: [8, 1]),
            "w.biases": entry("w.biases", dtype: .bf16, shape: [8, 1]),
            // A plain (non-quantized) tensor alongside it — isQuantized
            // must stay scoped to the exact base name.
            "n.weight": entry("n.weight", dtype: .f32, shape: [8]),
        ]
        let file = SafeTensorsFile(url: URL(fileURLWithPath: "/synthetic.gguf"), entries: entries)
        let bundle = SafeTensorsBundle(inMemory: file, directory: URL(fileURLWithPath: "/"))

        #expect(bundle.isQuantized("w"))
        #expect(!bundle.isQuantized("n"))
        let triplet = try bundle.quantizedTriplet("w")
        #expect(triplet.weight.dtype == .u32)
        #expect(triplet.scales.dtype == .bf16)
        #expect(triplet.biases.dtype == .bf16)
    }
}
