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
// `ConvertDriver` end-to-end smoke tests — exercise every supported
// `QuantSpec` against a tiny synthetic checkpoint built on disk.
//
// What each test verifies:
//   * `ConvertDriver.convert` writes a `(weight, scales, biases)`
//     triplet for an affine-quantize spec and the shapes match
//     `QuantizedOpsValidation.packedUInt32Count`.
//   * `ConvertDriver.convert` writes a single downcast tensor with
//     the target dtype for an `fp16` / `bf16` spec.
//   * Norm-like 1D weights are copied through unchanged.
//   * `config.json` carries the right `quantization` block (or omits
//     it entirely for a pure downcast).
//
// These DON'T test numerical correctness of the quantize kernels —
// `QuantizedOpsAffineTests.swift` and the metaltile-side
// `qmm_mma_int*_gpu_correctness.rs` tests already pin those. The
// convert-driver tests guard the *plumbing*: spec routing, storage
// shape calculation, config.json rewriting, dtype handling.

import Foundation
import Metal
import Testing

@testable import FFAI

@Suite("ConvertDriver")
struct ConvertDriverTests {

    // ─── Affine-quantize specs ─────────────────────────────────────

    @Test(
        "convert with .bits(N) produces an N-bit triplet for every supported width",
        arguments: [2, 3, 4, 5, 6, 8])
    func quantizesAtEveryBitWidth(bits: Int) throws {
        let dir = tempDir()
        defer { cleanUp(dir) }

        // Source: one 2D weight + one 1D norm. Inner dim 64 is the
        // smallest size that satisfies group_size=64 alignment for
        // every bit-width.
        let outDim = 4
        let inDim = 64
        try writeSyntheticSource(at: dir, outDim: outDim, inDim: inDim)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(bits)
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let outBundle = try SafeTensorsBundle(
            directory: destDir, device: .shared)

        // The 2D weight must have shipped as a triplet — packed u32
        // weight + scales + biases.
        let packed = try outBundle.tensor(named: "model.layers.0.q_proj.weight")
        let scales = try outBundle.tensor(named: "model.layers.0.q_proj.scales")
        let biases = try outBundle.tensor(named: "model.layers.0.q_proj.biases")
        #expect(packed.dtype == .u32, "packed weight must be u32 for bits=\(bits)")
        let expectedPacks = QuantizedOpsValidation.packedUInt32Count(
            numel: outDim * inDim, bits: bits)!
        #expect(
            packed.elementCount == expectedPacks,
            "packed count mismatch at bits=\(bits): got \(packed.elementCount), want \(expectedPacks)")
        // scales / biases are [outDim, inDim/groupSize] = [4, 1].
        #expect(scales.shape == [outDim, 1])
        #expect(biases.shape == [outDim, 1])

        // The 1D norm must be copied unchanged (still bf16, same shape).
        let norm = try outBundle.tensor(named: "model.layers.0.input_layernorm.weight")
        #expect(norm.dtype == .bf16)
        #expect(norm.shape == [outDim])

        // config.json must record the quantization block with bits=N.
        let cfg = try readConfig(at: destDir)
        let q = cfg["quantization"] as? [String: Any]
        #expect(q?["bits"] as? Int == bits)
        #expect(q?["group_size"] as? Int == 64)
        #expect(q?["mode"] as? String == "affine")
    }

    // ─── Downcast specs ────────────────────────────────────────────

    @Test("convert with .fp16 downcasts the 2D weight + norm to fp16 (no triplet)")
    func downcastsToFp16() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }

        try writeSyntheticSource(at: dir, outDim: 4, inDim: 64)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .fp16
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let outBundle = try SafeTensorsBundle(
            directory: destDir, device: .shared)

        // The 2D weight must have shipped as a single fp16 tensor (no
        // .scales / .biases siblings).
        let proj = try outBundle.tensor(named: "model.layers.0.q_proj.weight")
        #expect(proj.dtype == .f16)
        #expect(proj.shape == [4, 64])
        #expect(
            !outBundle.has("model.layers.0.q_proj.scales"),
            "downcast must not emit a quantization triplet")
        #expect(!outBundle.has("model.layers.0.q_proj.biases"))

        // Norms always pass through unchanged (still bf16, original
        // shape). The downcast spec on .bits doesn't touch norms — the
        // role-routing rule excludes them.
        let norm = try outBundle.tensor(named: "model.layers.0.input_layernorm.weight")
        #expect(norm.dtype == .bf16)
        #expect(norm.shape == [4])

        // No quantization block in config when there's no triplet.
        let cfg = try readConfig(at: destDir)
        #expect(cfg["quantization"] == nil)
        #expect(cfg["quantization_config"] == nil)
    }

    @Test("convert with .bf16 source = .bf16 spec passes weights through unchanged")
    func bf16SpecIsNoOpForBf16Source() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }

        try writeSyntheticSource(at: dir, outDim: 4, inDim: 64)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bf16
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let outBundle = try SafeTensorsBundle(
            directory: destDir, device: .shared)

        // Source is bf16 + .bf16 spec → effectiveSpec returns nil for
        // the source-already-target case, so the tensor flows through
        // the pass-through path. Shape + dtype preserved.
        let proj = try outBundle.tensor(named: "model.layers.0.q_proj.weight")
        #expect(proj.dtype == .bf16)
        #expect(proj.shape == [4, 64])
        #expect(!outBundle.has("model.layers.0.q_proj.scales"))
    }

    // ─── Mixed-spec conversion ─────────────────────────────────────

    @Test("mixed: --bits 4 + --embedding-bits 8 + --vision-bits fp16 routes each role correctly")
    func mixedPrecisionRouting() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }

        try writeSyntheticSource(
            at: dir, outDim: 4, inDim: 64,
            includeEmbedding: true,
            includeVisionTower: true)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.embeddingSpec = .bits(8)
        opts.visionSpec = .fp16
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let outBundle = try SafeTensorsBundle(
            directory: destDir, device: .shared)

        // Main weight: 4-bit triplet.
        let mainWeight = try outBundle.tensor(named: "model.layers.0.q_proj.weight")
        #expect(mainWeight.dtype == .u32)
        let mainExpected = QuantizedOpsValidation.packedUInt32Count(numel: 4 * 64, bits: 4)!
        #expect(mainWeight.elementCount == mainExpected)
        #expect(outBundle.has("model.layers.0.q_proj.scales"))

        // Embedding: 8-bit triplet (different bit-width than --bits).
        let embed = try outBundle.tensor(named: "model.embed_tokens.weight")
        #expect(embed.dtype == .u32)
        let embedExpected = QuantizedOpsValidation.packedUInt32Count(numel: 4 * 64, bits: 8)!
        #expect(
            embed.elementCount == embedExpected,
            "embedding should pack at 8-bit when --embedding-bits 8")
        #expect(outBundle.has("model.embed_tokens.scales"))

        // Vision tower: fp16 downcast (no triplet).
        let vision = try outBundle.tensor(named: "model.visual.blocks.0.attn.qkv.weight")
        #expect(vision.dtype == .f16)
        #expect(!outBundle.has("model.visual.blocks.0.attn.qkv.scales"))
    }

    // ─── MoE router handling ────────────────────────────────────────

    @Test("MoE router gate stays full-precision by default (routerSpec nil)")
    func moeRouterSkippedByDefault() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }

        try writeSyntheticSource(
            at: dir, outDim: 4, inDim: 64, includeMoERouter: true)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)  // main projections quantize…
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let outBundle = try SafeTensorsBundle(directory: destDir, device: .shared)

        // Main projection IS quantized (control).
        #expect(outBundle.has("model.layers.0.q_proj.scales"))

        // Router gate passes through unchanged — no triplet, source dtype.
        let gate = try outBundle.tensor(named: "model.layers.0.feed_forward.gate.weight")
        #expect(
            gate.dtype == .bf16,
            "router gate should keep its bf16 source dtype, not pack to u32")
        #expect(
            !outBundle.has("model.layers.0.feed_forward.gate.scales"),
            "router gate must not get a quantization triplet by default")
    }

    @Test("--router-bits opts the MoE router into quantization")
    func moeRouterQuantizedWhenOptedIn() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }

        try writeSyntheticSource(
            at: dir, outDim: 4, inDim: 64, includeMoERouter: true)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.routerSpec = .bits(8)  // mirror an upstream 8-bit router
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let outBundle = try SafeTensorsBundle(directory: destDir, device: .shared)

        // Router gate now packed to an 8-bit triplet.
        let gate = try outBundle.tensor(named: "model.layers.0.feed_forward.gate.weight")
        #expect(gate.dtype == .u32)
        let expected = QuantizedOpsValidation.packedUInt32Count(numel: 4 * 64, bits: 8)!
        #expect(
            gate.elementCount == expected,
            "router should pack at 8-bit when --router-bits 8")
        #expect(outBundle.has("model.layers.0.feed_forward.gate.scales"))
    }

    @Test("quantized router round-trips: loadLinear reads logical in_dim, not packed width")
    func quantizedRouterLoadsViaLoadLinear() throws {
        // Reproduces + guards the LFM2-MoE bug: a quantized router gate
        // (e.g. mlx-community/LFM2-8B-A1B-4bit's `feed_forward.gate`,
        // packed U32 whose column count is the *packed* width, not the
        // logical hidden) must load as a `QuantizedLinear` so the gemv
        // sees the logical in_dim. Loading it as a plain `Linear` read
        // in_dim from the packed width and crashed (`in_dim 512 vs 2048`).
        let dir = tempDir()
        defer { cleanUp(dir) }

        try writeSyntheticSource(
            at: dir, outDim: 4, inDim: 64, includeMoERouter: true)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.routerSpec = .bits(8)  // produce a real quantized gate triplet
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let outBundle = try SafeTensorsBundle(directory: destDir, device: .shared)
        let gate = try loadLinear(
            base: "model.layers.0.feed_forward.gate", in: outBundle,
            quantization: ModelConfig.QuantizationConfig(bits: 4, groupSize: 64))

        let q = try #require(
            gate.inner as? QuantizedLinear,
            "quantized router must load as QuantizedLinear, not plain Linear")
        #expect(q.bits == 8, "loadLinear should derive the router's 8-bit width from shapes")
        // Logical in_dim = packed columns × (32 / bits). For 8-bit that's
        // 16 packed cols × 4 = 64 — the original hidden, NOT the packed 16.
        let packedCols = q.weight.shape[1]
        let logicalInDim = packedCols * (32 / q.bits)
        #expect(
            logicalInDim == 64,
            "router gemv must see logical in_dim 64, not packed width \(packedCols)")
    }

    @Test("isMoERouterKey matches router gates but not dense gate_proj")
    func routerKeyMatcher() {
        // Positive — exact `.gate.weight` / `.router.weight` suffixes.
        #expect(ConvertDriver.isMoERouterKey("model.layers.0.feed_forward.gate.weight"))
        #expect(ConvertDriver.isMoERouterKey("model.layers.3.block_sparse_moe.gate.weight"))
        #expect(ConvertDriver.isMoERouterKey("model.layers.7.mlp.gate.weight"))
        #expect(ConvertDriver.isMoERouterKey("model.layers.0.mlp.router.weight"))
        // Negative — dense-MLP projections and attention weights.
        #expect(!ConvertDriver.isMoERouterKey("model.layers.0.mlp.gate_proj.weight"))
        #expect(!ConvertDriver.isMoERouterKey("model.layers.0.mlp.gate_up_proj.weight"))
        #expect(!ConvertDriver.isMoERouterKey("model.layers.0.q_proj.weight"))
        #expect(!ConvertDriver.isMoERouterKey("model.layers.0.feed_forward.gate.scales"))
    }

    // ─── Helpers ────────────────────────────────────────────────────

    /// Build a minimum-viable HuggingFace-style source directory with
    /// a single safetensors file + config.json. The synthetic model has
    /// just enough tensors to exercise role routing: one main linear,
    /// one norm, optionally an embedding and a vision-tower linear.
    private func writeSyntheticSource(
        at dir: URL, outDim: Int, inDim: Int,
        includeEmbedding: Bool = false,
        includeVisionTower: Bool = false,
        includeMoERouter: Bool = false
    ) throws {
        let safetensors = dir.appendingPathComponent("model.safetensors")
        let writer = SafeTensorsWriter(url: safetensors)

        // Main linear projection — 2D bf16 weight.
        let projBytes = bf16Bytes((0 ..< outDim * inDim).map { Float($0) * 0.01 - 0.5 })
        try writer.append(
            name: "model.layers.0.q_proj.weight",
            dtype: .bf16, shape: [outDim, inDim], bytes: projBytes)

        // Norm — 1D bf16, will always pass through.
        let normBytes = bf16Bytes((0 ..< outDim).map { _ in Float(1.0) })
        try writer.append(
            name: "model.layers.0.input_layernorm.weight",
            dtype: .bf16, shape: [outDim], bytes: normBytes)

        if includeEmbedding {
            let embedBytes = bf16Bytes((0 ..< outDim * inDim).map { Float($0) * 0.02 })
            try writer.append(
                name: "model.embed_tokens.weight",
                dtype: .bf16, shape: [outDim, inDim], bytes: embedBytes)
        }

        if includeVisionTower {
            let visionBytes = bf16Bytes((0 ..< outDim * inDim).map { Float($0) * 0.03 })
            try writer.append(
                name: "model.visual.blocks.0.attn.qkv.weight",
                dtype: .bf16, shape: [outDim, inDim], bytes: visionBytes)
        }

        if includeMoERouter {
            // MoE router gate `[numExperts, hidden]`. `outDim` doubles as
            // numExperts here; inDim=64 satisfies group_size alignment so
            // the tensor WOULD be quantized if it weren't router-skipped.
            let gateBytes = bf16Bytes((0 ..< outDim * inDim).map { Float($0) * 0.04 })
            try writer.append(
                name: "model.layers.0.feed_forward.gate.weight",
                dtype: .bf16, shape: [outDim, inDim], bytes: gateBytes)
        }

        try writer.finalize()

        // Minimal config.json — ConvertDriver requires one to exist
        // (it reads it to inject the quantization block).
        let cfg: [String: Any] = [
            "architectures": ["FakeForCausalLM"],
            "model_type": "fake",
            "hidden_size": inDim,
            "num_hidden_layers": 1,
        ]
        let cfgData = try JSONSerialization.data(
            withJSONObject: cfg, options: [.sortedKeys, .prettyPrinted])
        try cfgData.write(to: dir.appendingPathComponent("config.json"))
    }

    private func readConfig(at dir: URL) throws -> [String: Any] {
        let url = dir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        let obj = try JSONSerialization.jsonObject(with: data)
        guard let dict = obj as? [String: Any] else {
            throw NSError(
                domain: "ConvertDriverTests", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "config.json is not a dict"])
        }
        return dict
    }

    private func bf16Bytes(_ floats: [Float]) -> Data {
        let bits = floats.map { f -> UInt16 in
            let b = f.bitPattern
            let rounded = b &+ 0x7FFF &+ ((b >> 16) & 1)
            return UInt16(rounded >> 16)
        }
        return bits.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-convert-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func cleanUp(_ dir: URL) {
        try? FileManager.default.removeItem(at: dir)
    }
}
