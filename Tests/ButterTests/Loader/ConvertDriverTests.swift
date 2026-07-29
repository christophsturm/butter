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
// `QuantizedOpsAffineTests.swift` and the iron-side
// `qmm_mma_int*_gpu_correctness.rs` tests already pin those. The
// convert-driver tests guard the *plumbing*: spec routing, storage
// shape calculation, config.json rewriting, dtype handling.

import Foundation
import Metal
import Testing

@testable import ButterSwift

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
        // Positive — exact `.gate.weight` / `.router.weight` suffixes,
        // plus diffusion-gemma's `.router.proj.weight` scoring matrix.
        #expect(ConvertDriver.isMoERouterKey("model.layers.0.feed_forward.gate.weight"))
        #expect(ConvertDriver.isMoERouterKey("model.layers.3.block_sparse_moe.gate.weight"))
        #expect(ConvertDriver.isMoERouterKey("model.layers.7.mlp.gate.weight"))
        #expect(ConvertDriver.isMoERouterKey("model.layers.0.mlp.router.weight"))
        #expect(ConvertDriver.isMoERouterKey("model.decoder.layers.0.router.proj.weight"))
        // Negative — dense-MLP projections and attention weights.
        #expect(!ConvertDriver.isMoERouterKey("model.layers.0.mlp.gate_proj.weight"))
        #expect(!ConvertDriver.isMoERouterKey("model.layers.0.mlp.gate_up_proj.weight"))
        #expect(!ConvertDriver.isMoERouterKey("model.layers.0.q_proj.weight"))
        #expect(!ConvertDriver.isMoERouterKey("model.layers.0.feed_forward.gate.scales"))
    }

    // ─── MoE expert layout (stacked vs per-expert) ──────────────────

    @Test("parseExpertWeightKey maps w1/w3/w2 + *_proj to stacked switch_mlp base")
    func expertKeyParsing() {
        func base(_ k: String) -> String? { ConvertDriver.parseExpertWeightKey(k)?.stackedBase }
        func idx(_ k: String) -> Int? { ConvertDriver.parseExpertWeightKey(k)?.expert }
        let p = "model.layers.2.feed_forward"
        #expect(base("\(p).experts.0.w1.weight") == "\(p).switch_mlp.gate_proj")
        #expect(base("\(p).experts.5.w3.weight") == "\(p).switch_mlp.up_proj")
        #expect(base("\(p).experts.31.w2.weight") == "\(p).switch_mlp.down_proj")
        #expect(base("\(p).experts.3.gate_proj.weight") == "\(p).switch_mlp.gate_proj")
        #expect(idx("\(p).experts.7.w1.weight") == 7)
        // Non-expert keys → nil.
        #expect(ConvertDriver.parseExpertWeightKey("\(p).gate.weight") == nil)
        #expect(ConvertDriver.parseExpertWeightKey("model.layers.0.q_proj.weight") == nil)
        #expect(ConvertDriver.parseExpertWeightKey("\(p).experts.0.w1.scales") == nil)
    }

    @Test("default layout gathers per-expert weights into switch_mlp.<proj>")
    func stackedExpertConversion() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }
        let E = 4
        try writeSyntheticSource(at: dir, outDim: 4, inDim: 64, moeExperts: E)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)  // default expertLayout == .stacked
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let out = try SafeTensorsBundle(directory: destDir, device: .shared)
        let p = "model.layers.2.feed_forward"

        // Stacked quantized triplet [E, out, packed] / [E, out, groups].
        let w = try out.tensor(named: "\(p).switch_mlp.gate_proj.weight")
        #expect(w.dtype == .u32)
        let packedCols = QuantizedOpsValidation.packedUInt32Count(numel: 4 * 64, bits: 4)! / 4
        #expect(w.shape == [E, 4, packedCols])
        #expect(out.has("\(p).switch_mlp.gate_proj.scales"))
        #expect(out.has("\(p).switch_mlp.up_proj.weight"))   // from w3
        #expect(out.has("\(p).switch_mlp.down_proj.weight"))  // from w2
        // The per-expert source tensors must NOT survive in stacked mode.
        #expect(!out.has("\(p).experts.0.w1.weight"))
        #expect(!out.has("\(p).experts.3.w2.weight"))
    }

    @Test("per-expert layout keeps experts.<e>.<proj> tensors (quantized)")
    func perExpertLayoutConversion() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }
        try writeSyntheticSource(at: dir, outDim: 4, inDim: 64, moeExperts: 3)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.expertLayout = .perExpert
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let out = try SafeTensorsBundle(directory: destDir, device: .shared)
        let p = "model.layers.2.feed_forward"
        // Per-expert tensors preserved + quantized; no stacked tensor.
        #expect(out.has("\(p).experts.0.w1.weight"))
        #expect(out.has("\(p).experts.0.w1.scales"))
        #expect(!out.has("\(p).switch_mlp.gate_proj.weight"))
    }

    @Test("--expert-bits bf16 leaves stacked experts full-precision (no triplet)")
    func stackedExpertsFullPrecision() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }
        let E = 4
        try writeSyntheticSource(at: dir, outDim: 4, inDim: 64, moeExperts: E)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.expertSpec = .bf16  // leave experts raw even though body is 4-bit
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let out = try SafeTensorsBundle(directory: destDir, device: .shared)
        let p = "model.layers.2.feed_forward"
        let w = try out.tensor(named: "\(p).switch_mlp.gate_proj.weight")
        #expect(w.dtype == .bf16)
        #expect(w.shape == [E, 4, 64])  // raw stacked [E, out, in]
        #expect(!out.has("\(p).switch_mlp.gate_proj.scales"))
    }

    // ─── Fused 3D MoE experts (diffusion-gemma style) ───────────────

    @Test("parseFusedExpertKey maps fused experts.<proj> → switch_mlp base; rejects per-expert keys")
    func fusedExpertKeyParsing() {
        let p = "model.decoder.layers.0"
        // gate_up_proj → split kind under the switch_mlp base.
        let gu = ConvertDriver.parseFusedExpertKey("\(p).experts.gate_up_proj")
        #expect(gu?.outBasePrefix == "\(p).switch_mlp")
        #expect(gu?.kind == .gateUp)
        // Separate projections (+ w1/w3/w2 aliases) → single proj kind.
        #expect(ConvertDriver.parseFusedExpertKey("\(p).experts.down_proj")?.kind == .proj("down_proj"))
        #expect(ConvertDriver.parseFusedExpertKey("\(p).experts.gate_proj")?.kind == .proj("gate_proj"))
        #expect(ConvertDriver.parseFusedExpertKey("\(p).experts.w3")?.kind == .proj("up_proj"))
        // Tolerate an optional trailing `.weight`.
        #expect(
            ConvertDriver.parseFusedExpertKey("\(p).experts.down_proj.weight")?.kind
                == .proj("down_proj"))
        // Per-expert indexed keys are NOT fused (parseExpertWeightKey owns them).
        #expect(ConvertDriver.parseFusedExpertKey("\(p).experts.0.w1.weight") == nil)
        #expect(ConvertDriver.parseFusedExpertKey("\(p).experts.5.gate_proj.weight") == nil)
        // Non-expert keys (incl. a dense gate_up_proj that isn't under .experts.).
        #expect(ConvertDriver.parseFusedExpertKey("\(p).self_attn.q_proj.weight") == nil)
        #expect(ConvertDriver.parseFusedExpertKey("\(p).mlp.gate_up_proj.weight") == nil)
    }

    @Test("default (.fused) quantizes experts.gate_up_proj/down_proj IN PLACE (matches mlx layout)")
    func fusedExpertPreserveDefault() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }
        let E = 4
        try writeSyntheticSource(at: dir, outDim: 4, inDim: 64, fusedMoEExperts: E)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)  // default expertLayout == .fused
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let out = try SafeTensorsBundle(directory: destDir, device: .shared)
        let hidden = 64, moeOut = 128  // see writeSyntheticSource

        // gate_up stays FUSED + in place under its original name:
        // [E, 2*moeOut, hidden] → packed [E, 2*moeOut, hidden/8].
        let gu = try out.tensor(named: "model.layers.3.experts.gate_up_proj.weight")
        #expect(gu.dtype == .u32)
        #expect(gu.shape == [E, 2 * moeOut, hidden / 8])  // [4, 256, 8]
        #expect(
            try out.tensor(named: "model.layers.3.experts.gate_up_proj.scales").shape
                == [E, 2 * moeOut, hidden / 64])
        #expect(out.has("model.layers.3.experts.gate_up_proj.biases"))

        // down_proj in place: [E, hidden, moeOut] → packed [E, hidden, moeOut/8].
        let dn = try out.tensor(named: "model.layers.3.experts.down_proj.weight")
        #expect(dn.shape == [E, hidden, moeOut / 8])  // [4, 64, 16]
        #expect(
            try out.tensor(named: "model.layers.3.experts.down_proj.scales").shape
                == [E, hidden, moeOut / 64])

        // No split, no rename to switch_mlp under the default layout.
        #expect(!out.has("model.layers.3.switch_mlp.gate_proj.weight"))
        #expect(!out.has("model.layers.3.switch_mlp.up_proj.weight"))
    }

    @Test(".fused + --expert-bits bf16 keeps experts.gate_up_proj as a raw 3D stack")
    func fusedExpertPreserveFullPrecision() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }
        let E = 3
        try writeSyntheticSource(at: dir, outDim: 4, inDim: 64, fusedMoEExperts: E)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.expertSpec = .bf16  // experts stay full-precision
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let out = try SafeTensorsBundle(directory: destDir, device: .shared)
        let gu = try out.tensor(named: "model.layers.3.experts.gate_up_proj.weight")
        #expect(gu.dtype == .bf16)
        #expect(gu.shape == [E, 256, 64])  // raw [E, 2*moeOut, hidden], unsplit
        #expect(!out.has("model.layers.3.experts.gate_up_proj.scales"))
        #expect(try out.tensor(named: "model.layers.3.experts.down_proj.weight").shape == [E, 64, 128])
    }

    @Test("--expert-layout stacked splits fused gate_up into switch_mlp.{gate,up,down}")
    func fusedExpertStackedSplit() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }
        let E = 4
        try writeSyntheticSource(at: dir, outDim: 4, inDim: 64, fusedMoEExperts: E)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.expertLayout = .stacked  // opt into the Butter switch_mlp split
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let out = try SafeTensorsBundle(directory: destDir, device: .shared)
        let p = "model.layers.3.switch_mlp"
        let hidden = 64, moeOut = 128

        let gate = try out.tensor(named: "\(p).gate_proj.weight")
        #expect(gate.dtype == .u32)
        #expect(gate.shape == [E, moeOut, hidden / 8])  // [4, 128, 8]
        #expect(out.has("\(p).up_proj.weight"))
        #expect(try out.tensor(named: "\(p).down_proj.weight").shape == [E, hidden, moeOut / 8])
        // The fused tensor is gone in the split layout.
        #expect(!out.has("model.layers.3.experts.gate_up_proj.weight"))
    }

    @Test("--expert-layout stacked gate_up split selects the right rows (bf16 round-trip is exact)")
    func fusedGateUpSplitIsExact() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }
        let E = 2
        try writeSyntheticSource(at: dir, outDim: 4, inDim: 64, fusedMoEExperts: E)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.expertLayout = .stacked  // the split path is what we're pinning
        opts.expertSpec = .bf16  // raw copy → exact, isolates the slicing logic
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let srcB = try SafeTensorsBundle(directory: dir, device: .shared)
        let outB = try SafeTensorsBundle(directory: destDir, device: .shared)

        let hidden = 64, moeOut = 128
        let perExpert = (2 * moeOut) * hidden
        let half = moeOut * hidden
        let gateUp = try srcB.tensor(named: "model.layers.3.experts.gate_up_proj").toFloatArray()
        let gate = try outB.tensor(named: "model.layers.3.switch_mlp.gate_proj.weight").toFloatArray()
        let up = try outB.tensor(named: "model.layers.3.switch_mlp.up_proj.weight").toFloatArray()

        // Per expert: gate == first half of the gate_up rows, up == second.
        for e in 0 ..< E {
            let base = e * perExpert
            let expGate = Array(gateUp[base ..< base + half])
            let expUp = Array(gateUp[base + half ..< base + 2 * half])
            #expect(
                Array(gate[e * half ..< (e + 1) * half]) == expGate,
                "expert \(e): gate_proj must be the FIRST half of gate_up rows")
            #expect(
                Array(up[e * half ..< (e + 1) * half]) == expUp,
                "expert \(e): up_proj must be the SECOND half of gate_up rows")
        }

        // down_proj is a straight per-expert passthrough (no split).
        let srcDown = try srcB.tensor(named: "model.layers.3.experts.down_proj").toFloatArray()
        let outDown = try outB.tensor(named: "model.layers.3.switch_mlp.down_proj.weight").toFloatArray()
        #expect(outDown == srcDown, "down_proj must be copied through expert-for-expert, unchanged")
    }

    // ─── Sharded output ─────────────────────────────────────────────

    @Test("output above the shard cap splits into model-0000X + index.json (round-trips)")
    func shardsLargeOutput() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }
        try writeSyntheticSource(
            at: dir, outDim: 4, inDim: 64, includeEmbedding: true, moeExperts: 3)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.maxShardBytes = 256  // tiny cap → force several shards
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let fm = FileManager.default
        let files = (try? fm.contentsOfDirectory(atPath: destDir.path)) ?? []
        let shards = files.filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }
        #expect(shards.count >= 2, "tiny cap should produce multiple shards, got \(shards)")
        #expect(files.contains("model.safetensors.index.json"))
        #expect(
            !files.contains("model.safetensors"),
            "a sharded checkpoint must not also write a lone model.safetensors")

        // index.json: weight_map covers tensors, points only at real shards,
        // and metadata.total_size is positive.
        let idx =
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: destDir.appendingPathComponent("model.safetensors.index.json")))
            as! [String: Any]
        let weightMap = idx["weight_map"] as! [String: String]
        let meta = idx["metadata"] as! [String: Any]
        #expect((meta["total_size"] as! Int) > 0)
        #expect(Set(weightMap.values).isSubset(of: Set(shards)))

        // Round-trip: the loader reads the sharded checkpoint transparently.
        let outB = try SafeTensorsBundle(directory: destDir, device: .shared)
        #expect(try outB.tensor(named: "model.layers.0.q_proj.weight").dtype == .u32)
        #expect(outB.has("model.embed_tokens.weight"))
        // Per-expert source under the default .fused layout still stacks to switch_mlp.
        #expect(outB.has("model.layers.2.feed_forward.switch_mlp.gate_proj.weight"))
    }

    @Test("output below the shard cap stays a single model.safetensors (no index)")
    func singleFileBelowShardCap() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }
        try writeSyntheticSource(at: dir, outDim: 4, inDim: 64)

        let destDir = dir.appendingPathComponent("out")
        var opts = ConvertOptions()
        opts.bits = .bits(4)  // default 5 GB cap; synthetic model is tiny
        try ConvertDriver.convert(sourceDir: dir, destDir: destDir, options: opts)

        let files = (try? FileManager.default.contentsOfDirectory(atPath: destDir.path)) ?? []
        #expect(files.contains("model.safetensors"))
        #expect(!files.contains("model.safetensors.index.json"))
        #expect(!files.contains { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") })
    }

    // ─── Helpers ────────────────────────────────────────────────────

    /// Build a minimum-viable HuggingFace-style source directory with
    /// a single safetensors file + config.json. The synthetic model has
    /// just enough tensors to exercise role routing: one main linear,
    /// one norm, optionally an embedding and a vision-tower linear.
    /// Per-expert `out` (= moe_intermediate) and `in` (= hidden) dims for
    /// the fused-3D synthetic experts. Deliberately distinct + both
    /// group_size(64)-aligned so the gate/up split (rows) and the down
    /// in-dim are independently verifiable.
    private static let fusedMoEHidden = 64
    private static let fusedMoEIntermediate = 128

    private func writeSyntheticSource(
        at dir: URL, outDim: Int, inDim: Int,
        includeEmbedding: Bool = false,
        includeVisionTower: Bool = false,
        includeMoERouter: Bool = false,
        moeExperts: Int = 0,
        fusedMoEExperts: Int = 0
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

        // Per-expert MoE weights at layer 2: experts.<e>.{w1,w3,w2}.weight,
        // each [outDim, inDim] (kept quantizable: inDim divisible by 64).
        for e in 0 ..< moeExperts {
            for proj in ["w1", "w3", "w2"] {
                let bytes = bf16Bytes(
                    (0 ..< outDim * inDim).map { Float($0) * 0.001 + Float(e) * 0.01 })
                try writer.append(
                    name: "model.layers.2.feed_forward.experts.\(e).\(proj).weight",
                    dtype: .bf16, shape: [outDim, inDim], bytes: bytes)
            }
        }

        // Fused 3D MoE expert stacks at layer 3 (diffusion-gemma style):
        //   experts.gate_up_proj [E, 2*moeOut, hidden]  (gate ‖ up on rows)
        //   experts.down_proj    [E, hidden,   moeOut]
        // No per-expert index, no `.weight` suffix — the fused layout.
        if fusedMoEExperts > 0 {
            let E = fusedMoEExperts
            let hidden = Self.fusedMoEHidden
            let moeOut = Self.fusedMoEIntermediate
            let gateUp = bf16Bytes(
                (0 ..< E * (2 * moeOut) * hidden).map { Float($0) * 0.0001 - 0.3 })
            try writer.append(
                name: "model.layers.3.experts.gate_up_proj",
                dtype: .bf16, shape: [E, 2 * moeOut, hidden], bytes: gateUp)
            let down = bf16Bytes(
                (0 ..< E * hidden * moeOut).map { Float($0) * 0.0001 - 0.2 })
            try writer.append(
                name: "model.layers.3.experts.down_proj",
                dtype: .bf16, shape: [E, hidden, moeOut], bytes: down)
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
            .appendingPathComponent("butter-convert-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func cleanUp(_ dir: URL) {
        try? FileManager.default.removeItem(at: dir)
    }
}
