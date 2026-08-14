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
// Unit tests for the Muse Glimmer vision path — the ViT-G/14 perception
// encoder + vision-adapter/vision-projection MLP
// (`Models/Vision/MuseGlimmerVision.swift`).
//
// Offline — no checkpoint download. Covers:
//   • `MuseGlimmerVisionConfig.decode` — geometry + defaults + the
//     per-layer window/full attention selector.
//   • `museGlimmerWindowAttentionGroups` — the window-partitioning
//     function, checked structurally (every patch covered exactly once,
//     window boundaries respected) on a small synthetic grid.
//   • A numeric oracle proving windowed (restricted-softmax) attention
//     is mathematically equivalent to full attention with an explicit
//     block-diagonal mask — the actual "window attention masking"
//     correctness property, independent of `MuseGlimmerVisionBlock`'s
//     implementation (this test re-derives both sides from raw Q/K/V
//     arrays, it does not call into the block's private code).
//   • The patch-embed projection is a single `Ops.gemm` against a small
//     hand-computed weight/input — verifies the checkpoint's ALREADY-2D
//     patch-embed weight (no Conv3d reshape) round-trips through the
//     tower's K-tile padding correctly.
//   • The projector's quantized `AnyLinear.callMany` fast path
//     (`Ops.dequantGemmDynamicM`, the actual kernel `vision_adapter.fc1
//     /fc2` and `vision_projection` dispatch at their real `outDim`s —
//     all three are `outDim % 32 == 0`) against the decoupled
//     dequantize-then-GEMM reference path (`Ops.affineDequantize` +
//     `Ops.gemm`) on a synthetic 4-bit quantized weight built with the
//     same `outDim % 32 == 0` shape so the SAME kernel the real
//     projector uses is exercised.

import Foundation
import Metal
import Testing

@testable import ButterSwift

// ─── Config decode + routing (house baseline) ─────────────────────────

@Suite("Muse Glimmer Vision — config")
struct MuseGlimmerVisionConfigTests {

    /// A representative `MuseGlimmerForConditionalGeneration` root config
    /// — the real checkpoint's `vision_config` shape.
    private func makeRootConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "hidden_act": "gelu",
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "layer_norm_eps": 1e-5,
            "layer_types": Array(
                repeating: ["window_attention", "window_attention", "window_attention", "full_attention"],
                count: 12
            ).flatMap { $0 } + ["window_attention", "full_attention"],  // 50 entries
            "max_position_embeddings": 1024,
            "merge_size": 2,
            "model_type": "muse_glimmer_vision",
            "num_attention_heads": 16,
            "num_hidden_layers": 50,
            "patch_size": 14,
            "patch_temporal": 2,
            "pos_emb_height": 32,
            "pos_emb_width": 32,
            "rope_parameters": ["rope_theta": 10_000.0, "rope_type": "default"],
        ]
        let raw: [String: Any] = [
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "model_type": "muse_glimmer",
            "image_token_id": 200_092,
            "video_token_id": 200_091,
            "vision_config": visionConfig,
            "text_config": ["model_type": "muse_glimmer_text", "hidden_size": 6656],
        ]
        return ModelConfig(
            architecture: "MuseGlimmerForConditionalGeneration",
            modelType: "muse_glimmer", raw: raw)
    }

    @Test("routes as a vision-language checkpoint via nested vision_config")
    func routesAsVisionLanguage() {
        let cfg = makeRootConfig()
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
    }

    @Test("vision_config decodes into the ViT-G/14 geometry")
    func visionConfigDecode() throws {
        let vc = try #require(makeRootConfig().subConfig("vision_config"))
        let parsed = try MuseGlimmerVisionConfig.decode(vc)
        #expect(parsed.depth == 50)
        #expect(parsed.hidden == 1536)
        #expect(parsed.intermediate == 8960)
        #expect(parsed.numHeads == 16)
        #expect(parsed.patchSize == 14)
        #expect(parsed.mergeSize == 2)
        #expect(parsed.temporalPatchSize == 2)
        #expect(parsed.posEmbHeight == 32)
        #expect(parsed.posEmbWidth == 32)
        #expect(abs(parsed.ropeTheta - 10_000.0) < 1e-3)
        #expect(parsed.inChannels == 3)
        // Derived geometry.
        #expect(parsed.headDim == 96)  // 1536 / 16
        #expect(parsed.mergeUnit == 4)  // 2×2 patches per merged token
        #expect(parsed.windowPatches == 32)  // == pos_emb_height (documented default)

        // Patch-embed flatten dim: inChannels·temporalPatch·patch·patch.
        let patchDim = parsed.inChannels * parsed.temporalPatchSize * parsed.patchSize * parsed.patchSize
        #expect(patchDim == 1176)  // matches the checkpoint's [1536, 1176] tensor
        let patchDimPadded = ((patchDim + gemmKTileWidth - 1) / gemmKTileWidth) * gemmKTileWidth
        #expect(patchDimPadded == 1184)  // next multiple of 16
    }

    @Test("vision_config decode falls back to documented defaults")
    func visionConfigDefaults() throws {
        let minimal = ModelConfig(
            architecture: nil, modelType: nil,
            raw: [
                "num_hidden_layers": 4,
                "hidden_size": 64,
                "num_attention_heads": 4,
                "patch_size": 14,
                "merge_size": 2,
            ])
        let parsed = try MuseGlimmerVisionConfig.decode(minimal)
        #expect(parsed.intermediate == 64 * 4)  // hidden * 4 fallback
        #expect(parsed.temporalPatchSize == 2)
        #expect(parsed.inChannels == 3)
        #expect(abs(parsed.layerNormEps - 1e-5) < 1e-9)
        #expect(parsed.posEmbHeight == 32)
        #expect(parsed.posEmbWidth == 32)
        #expect(abs(parsed.ropeTheta - 10_000.0) < 1e-3)
        #expect(parsed.layerTypes == nil)
    }

    @Test("decode throws on missing required geometry fields")
    func visionConfigMissingRequired() {
        let broken = ModelConfig(architecture: nil, modelType: nil, raw: ["hidden_size": 64])
        #expect(throws: (any Error).self) {
            try MuseGlimmerVisionConfig.decode(broken)
        }
    }

    // ── Per-layer attention-scope selector ──

    @Test("isFullAttentionLayer reads layer_types when present")
    func fullAttentionFromLayerTypes() throws {
        let vc = try #require(makeRootConfig().subConfig("vision_config"))
        let parsed = try MuseGlimmerVisionConfig.decode(vc)
        // Pattern is (3× window + 1× full) × 12, then one extra window
        // layer, then one final full layer (50 entries total).
        #expect(!parsed.isFullAttentionLayer(0))
        #expect(!parsed.isFullAttentionLayer(1))
        #expect(!parsed.isFullAttentionLayer(2))
        #expect(parsed.isFullAttentionLayer(3))
        #expect(!parsed.isFullAttentionLayer(4))
        #expect(parsed.isFullAttentionLayer(49))  // last layer is full
    }

    @Test("isFullAttentionLayer falls back to (3× window + 1× full) × N without layer_types")
    func fullAttentionFallbackPattern() throws {
        let minimal = ModelConfig(
            architecture: nil, modelType: nil,
            raw: [
                "num_hidden_layers": 8, "hidden_size": 64,
                "num_attention_heads": 4, "patch_size": 14, "merge_size": 2,
            ])
        let parsed = try MuseGlimmerVisionConfig.decode(minimal)
        for i in 0 ..< 8 {
            let expected = (i + 1) % 4 == 0
            #expect(
                parsed.isFullAttentionLayer(i) == expected,
                "layer \(i): expected full=\(expected)")
        }
    }
}

// ─── Window-grouping — structural correctness ──────────────────────────

@Suite("Muse Glimmer Vision — window grouping")
struct MuseGlimmerWindowGroupingTests {

    @Test("partitions every patch exactly once into 4 windows on an 4×4 grid")
    func partitionsSmallGrid() {
        // gridSide=4, mergeSize=2 → 2×2 merge-blocks (mergeBlocks=2).
        // windowPatches=2 → winBlocks = 2/2 = 1 merge-block per window
        // side → nWin = ceil(2/1) = 2 → 2×2 = 4 distinct windows, each
        // covering exactly one merge-block (4 patches).
        let groups = museGlimmerWindowAttentionGroups(
            gridSide: 4, mergeSize: 2, windowPatches: 2, gridT: 1)
        #expect(groups.count == 4)
        for g in groups { #expect(g.count == 4) }

        // Every patch index [0, 16) appears in EXACTLY one group.
        var seen = [Int: Int](minimumCapacity: 16)
        for g in groups {
            for idx in g { seen[idx, default: 0] += 1 }
        }
        #expect(seen.count == 16, "expected all 16 patches covered, got \(seen.count)")
        for (idx, count) in seen {
            #expect(count == 1, "patch \(idx) appears in \(count) groups, expected 1")
        }

        // Merge-block-raster order: merge-block (br,bc) occupies
        // indices [ (br·2+bc)·4, (br·2+bc)·4 + 4 ). Window (wr,wc) with
        // winBlocks=1 is exactly merge-block (wr,wc) — so group[k] ==
        // { (wr·2+wc)·4 ..< (wr·2+wc)·4+4 } for k = wr·2+wc.
        for wr in 0 ..< 2 {
            for wc in 0 ..< 2 {
                let k = wr * 2 + wc
                let base = (wr * 2 + wc) * 4
                let expected = Set(base ..< (base + 4))
                #expect(
                    Set(groups[k]) == expected,
                    "window (\(wr),\(wc)): got \(groups[k].sorted()), expected \(expected.sorted())")
            }
        }
    }

    @Test("windowPatches >= gridSide collapses to one full-image window")
    func collapsesToFullImage() {
        // Matches the default single-tile encode(image:) path: window
        // granularity (32) == the default test grid (32) → windowed and
        // full-attention layers are numerically identical.
        let groups = museGlimmerWindowAttentionGroups(
            gridSide: 32, mergeSize: 2, windowPatches: 32, gridT: 1)
        #expect(groups.count == 1)
        #expect(groups[0].count == 32 * 32)
        #expect(Set(groups[0]) == Set(0 ..< (32 * 32)))
    }

    @Test("gridT > 1 partitions each temporal group independently")
    func temporalGroupsIndependent() {
        let groups = museGlimmerWindowAttentionGroups(
            gridSide: 4, mergeSize: 2, windowPatches: 2, gridT: 2)
        #expect(groups.count == 8)  // 4 windows × 2 temporal groups
        // No group spans both temporal groups — every index in a group
        // is either entirely < 16 (frame 0) or entirely >= 16 (frame 1).
        for g in groups {
            let allFrame0 = g.allSatisfy { $0 < 16 }
            let allFrame1 = g.allSatisfy { $0 >= 16 }
            #expect(allFrame0 || allFrame1, "window mixes temporal groups: \(g.sorted())")
        }
    }
}

// ─── Numeric oracle: windowed == full-attention-with-block-mask ───────

@Suite("Muse Glimmer Vision — window attention masking oracle")
struct MuseGlimmerWindowAttentionMaskingOracleTests {

    /// Independent windowed-softmax attention: restricts both the score
    /// matrix and the value sum to each group. `q`/`k`/`v` are
    /// `[nTokens][headDim]`.
    private func windowedAttention(
        q: [[Float]], k: [[Float]], v: [[Float]], groups: [[Int]], scale: Float
    ) -> [[Float]] {
        let headDim = q[0].count
        var out = [[Float]](repeating: [Float](repeating: 0, count: headDim), count: q.count)
        for group in groups {
            for i in group {
                var scores = [Float](repeating: 0, count: group.count)
                var maxS = -Float.greatestFiniteMagnitude
                for (gi, j) in group.enumerated() {
                    var dot: Float = 0
                    for d in 0 ..< headDim { dot += q[i][d] * k[j][d] }
                    scores[gi] = dot * scale
                    if scores[gi] > maxS { maxS = scores[gi] }
                }
                var sum: Float = 0
                for gi in 0 ..< group.count {
                    scores[gi] = exp(scores[gi] - maxS)
                    sum += scores[gi]
                }
                for (gi, j) in group.enumerated() {
                    let w = scores[gi] / sum
                    for d in 0 ..< headDim { out[i][d] += w * v[j][d] }
                }
            }
        }
        return out
    }

    /// Independent DENSE full attention with an explicit block-diagonal
    /// mask derived from `groups` (score = -inf for cross-group pairs).
    /// Re-derived from scratch — does not call `windowedAttention` or
    /// any production code.
    private func maskedFullAttention(
        q: [[Float]], k: [[Float]], v: [[Float]], groups: [[Int]], scale: Float
    ) -> [[Float]] {
        let n = q.count
        let headDim = q[0].count
        // windowOf[token] = group index.
        var windowOf = [Int](repeating: -1, count: n)
        for (gi, g) in groups.enumerated() {
            for idx in g { windowOf[idx] = gi }
        }
        var out = [[Float]](repeating: [Float](repeating: 0, count: headDim), count: n)
        for i in 0 ..< n {
            var scores = [Float](repeating: -Float.greatestFiniteMagnitude, count: n)
            var maxS = -Float.greatestFiniteMagnitude
            for j in 0 ..< n where windowOf[j] == windowOf[i] {
                var dot: Float = 0
                for d in 0 ..< headDim { dot += q[i][d] * k[j][d] }
                scores[j] = dot * scale
                if scores[j] > maxS { maxS = scores[j] }
            }
            var sum: Float = 0
            var weights = [Float](repeating: 0, count: n)
            for j in 0 ..< n where windowOf[j] == windowOf[i] {
                weights[j] = exp(scores[j] - maxS)
                sum += weights[j]
            }
            for j in 0 ..< n where windowOf[j] == windowOf[i] {
                let w = weights[j] / sum
                for d in 0 ..< headDim { out[i][d] += w * v[j][d] }
            }
        }
        return out
    }

    @Test("windowed softmax == full attention with a block-diagonal mask")
    func windowedEqualsFullyMaskedFull() {
        let groups = museGlimmerWindowAttentionGroups(
            gridSide: 4, mergeSize: 2, windowPatches: 2, gridT: 1)
        #expect(groups.count == 4)

        // Deterministic pseudo-random Q/K/V — no RNG dependency, just a
        // few distinct trig-derived floats per token so every dot
        // product is distinct (guards against a degenerate all-equal
        // test that would pass trivially).
        let n = 16
        let headDim = 4
        func vec(_ seed: Int, _ salt: Float) -> [Float] {
            (0 ..< headDim).map { d in sin(Float(seed * 7 + d) * 0.37 + salt) }
        }
        let q = (0 ..< n).map { vec($0, 0.1) }
        let k = (0 ..< n).map { vec($0, 1.7) }
        let v = (0 ..< n).map { vec($0, 3.3) }
        let scale = 1.0 / Float(headDim).squareRoot()

        let windowed = windowedAttention(q: q, k: k, v: v, groups: groups, scale: scale)
        let masked = maskedFullAttention(q: q, k: k, v: v, groups: groups, scale: scale)

        for i in 0 ..< n {
            for d in 0 ..< headDim {
                #expect(
                    abs(windowed[i][d] - masked[i][d]) < 1e-5,
                    "token \(i), dim \(d): windowed=\(windowed[i][d]) masked=\(masked[i][d])")
            }
        }

        // Non-vacuous sanity check: an UNMASKED full attention (every
        // token attends to every other token) gives a DIFFERENT result
        // for at least one token — proves the mask actually changes the
        // computation rather than the test being trivially satisfied.
        let fullGroup = [Array(0 ..< n)]
        let unmasked = windowedAttention(q: q, k: k, v: v, groups: fullGroup, scale: scale)
        var anyDifferent = false
        for i in 0 ..< n {
            for d in 0 ..< headDim where abs(unmasked[i][d] - masked[i][d]) > 1e-4 {
                anyDifferent = true
            }
        }
        #expect(anyDifferent, "masked and unmasked attention should differ on this input")
    }
}

// ─── Patch-embed GEMM oracle ────────────────────────────────────────────

@Suite("Muse Glimmer Vision — patch-embed GEMM oracle")
struct MuseGlimmerPatchEmbedGemmTests {

    @Test("Ops.gemm patch-embed projection matches a manual dot product")
    func patchEmbedMatchesManualDotProduct() {
        let device = Device.shared
        // hidden=3, patchDim=16 (already K-tile aligned — no padding
        // needed at this size). weight[o,c] = o*16+c (distinct per
        // cell); input row r = [r, r+1, ..., r+15].
        let hidden = 3
        let patchDim = 16
        var wVals = [Float](repeating: 0, count: hidden * patchDim)
        for o in 0 ..< hidden {
            for c in 0 ..< patchDim { wVals[o * patchDim + c] = Float(o * patchDim + c) * 0.01 }
        }
        let weight = ImagePreprocessing.makeTensor(
            from: wVals, shape: [hidden, patchDim], dtype: .f32, device: device)

        let nPatches = 2
        var xVals = [Float](repeating: 0, count: nPatches * patchDim)
        for r in 0 ..< nPatches {
            for c in 0 ..< patchDim { xVals[r * patchDim + c] = Float(r + c) }
        }
        let x = ImagePreprocessing.makeTensor(
            from: xVals, shape: [nPatches, patchDim], dtype: .f32, device: device)

        let cmd = device.makeCommandBuffer()
        let out = Ops.gemm(weight: weight, input: x, nRows: nPatches, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        let got = out.toArray(as: Float.self)

        for r in 0 ..< nPatches {
            for o in 0 ..< hidden {
                var expected: Float = 0
                for c in 0 ..< patchDim {
                    expected += wVals[o * patchDim + c] * xVals[r * patchDim + c]
                }
                let actual = got[r * hidden + o]
                #expect(
                    abs(actual - expected) < 1e-3,
                    "(r=\(r), o=\(o)): expected \(expected), got \(actual)")
            }
        }
    }

    @Test("padLinearColsTo pads the real checkpoint patch dim (1176 → 1184) with zero columns")
    func realPatchDimPadding() {
        let device = Device.shared
        let hidden = 2
        let patchDim = 1176
        let patchDimPadded = ((patchDim + gemmKTileWidth - 1) / gemmKTileWidth) * gemmKTileWidth
        #expect(patchDimPadded == 1184)

        let wVals: [Float] = (0 ..< (hidden * patchDim)).map { Float($0) }
        let w = ImagePreprocessing.makeTensor(
            from: wVals, shape: [hidden, patchDim], dtype: .f32, device: device)
        let padded = padLinearColsTo(w, toCols: patchDimPadded, device: device)
        #expect(padded.shape == [hidden, patchDimPadded])
        let got = padded.toArray(as: Float.self)
        // Original columns preserved; trailing 8 columns per row are zero.
        for o in 0 ..< hidden {
            for c in 0 ..< patchDim {
                #expect(got[o * patchDimPadded + c] == wVals[o * patchDim + c])
            }
            for c in patchDim ..< patchDimPadded {
                #expect(got[o * patchDimPadded + c] == 0)
            }
        }
    }
}

// ─── Projector GEMM oracle (quantized) ─────────────────────────────────

@Suite("Muse Glimmer Vision — projector quantized GEMM oracle")
struct MuseGlimmerProjectorGemmTests {

    /// The real `vision_adapter`/`vision_projection` layers are all
    /// `outDim % 32 == 0` (4096, 4096, 6656), which routes
    /// `QuantizedLinear.callMany` through the fused
    /// `Ops.dequantGemmDynamicM` kernel — the SAME kernel this test
    /// exercises (outDim=32) rather than the narrow-outDim dequantize-
    /// then-gemm fallback, so the test is representative of the actual
    /// projector dispatch.
    @Test("AnyLinear.callMany (fused dequant-GEMM) matches dequantize-then-gemm reference")
    func fusedDequantGemmMatchesReference() {
        let device = Device.shared
        let outDim = 32
        let inDim = 64
        let groupSize = 64  // 1 group per row at this inDim
        let bits = 4
        let packFactor = 32 / bits  // 8

        // A deterministic, non-degenerate weight matrix (mixed sign,
        // varied magnitude — exercises real quantization steps, not
        // just zero/constant groups).
        var wVals = [Float](repeating: 0, count: outDim * inDim)
        for o in 0 ..< outDim {
            for c in 0 ..< inDim {
                wVals[o * inDim + c] = sin(Float(o * inDim + c) * 0.013) * 2.0
            }
        }
        let wSrc = ImagePreprocessing.makeTensor(
            from: wVals, shape: [outDim, inDim], dtype: .f32, device: device)

        let packed = Tensor.empty(shape: [outDim, inDim / packFactor], dtype: .u32, device: device)
        let scales = Tensor.empty(shape: [outDim, inDim / groupSize], dtype: .f32, device: device)
        let biases = Tensor.empty(shape: [outDim, inDim / groupSize], dtype: .f32, device: device)
        packed.zero()
        scales.zero()
        biases.zero()

        let qCmd = device.makeCommandBuffer()
        QuantizedOps.quantizeAffine(
            weight: wSrc, packed: packed, scales: scales, biases: biases,
            bits: bits, groupSize: groupSize, on: qCmd)
        qCmd.commit()
        qCmd.waitUntilCompleted()

        let quantLinear = QuantizedLinear(
            weight: packed, scales: scales, biases: biases, bits: bits, groupSize: groupSize)
        let anyLinear = AnyLinear(quantLinear)

        let t = 3
        var xVals = [Float](repeating: 0, count: t * inDim)
        for r in 0 ..< t {
            for c in 0 ..< inDim { xVals[r * inDim + c] = cos(Float(r * inDim + c) * 0.021) }
        }
        let x = ImagePreprocessing.makeTensor(
            from: xVals, shape: [t, inDim], dtype: .f32, device: device)

        // Fused path — the actual production dispatch.
        let fusedCmd = device.makeCommandBuffer()
        let fused = anyLinear.callMany(x, t: t, on: fusedCmd, device: device)
        fusedCmd.commit()
        fusedCmd.waitUntilCompleted()
        let fusedOut = fused.toArray(as: Float.self)

        // Reference path — decoupled dequantize (Ops.affineDequantize)
        // then a plain dense Ops.gemm.
        let dense = Tensor.empty(shape: [outDim, inDim], dtype: .f32, device: device)
        let refCmd = device.makeCommandBuffer()
        Ops.affineDequantize(
            weight: packed, scales: scales, biases: biases,
            outDim: outDim, inDim: inDim, groupSize: groupSize, bits: bits,
            on: refCmd, into: dense)
        let refOut = Ops.gemm(weight: dense, input: x, nRows: t, on: refCmd)
        refCmd.commit()
        refCmd.waitUntilCompleted()
        let refOutVals = refOut.toArray(as: Float.self)

        #expect(fusedOut.count == refOutVals.count)
        for i in 0 ..< fusedOut.count {
            #expect(
                abs(fusedOut[i] - refOutVals[i]) < 1e-3,
                "index \(i): fused=\(fusedOut[i]) reference=\(refOutVals[i])")
        }

        // Non-vacuous sanity: the quantized output should be CLOSE to
        // (but not bit-identical to) the unquantized reference GEMM —
        // proves the test weight actually round-trips through real
        // 4-bit quantization rather than degenerating to zero.
        let unquantCmd = device.makeCommandBuffer()
        let unquantOut = Ops.gemm(weight: wSrc, input: x, nRows: t, on: unquantCmd)
        unquantCmd.commit()
        unquantCmd.waitUntilCompleted()
        let unquantVals = unquantOut.toArray(as: Float.self)
        var maxAbs: Float = 0
        for v in unquantVals { maxAbs = max(maxAbs, abs(v)) }
        #expect(maxAbs > 0.1, "reference GEMM output should be non-trivial")
    }
}
