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
// DIAGNOSTIC PROBE — not a permanent regression test. Investigates the
// color-fidelity bug tracked for Muse Glimmer vision (solid-color /
// simple-shape images caption the wrong color, but shapes and splicing
// are structurally correct). Probes the ONE undocumented fact in
// `MuseGlimmerVisionModel`'s patch-embed: the checkpoint ships the
// patch-embed weight as an ALREADY-2D `[1536, 1176]` matrix (not a 5-D
// Conv3d tensor the way the Qwen-VL family's `flattenPatchEmbed` can
// layout-detect), so the 1176-column flatten convention
// `(temporal, channel, patchRow, patchCol)` the port assumed is a
// documented GUESS, not a verified fact.
//
// Since a SOLID-COLOR patch has the same value at every (temporal,
// row, col) position within a channel, the only thing a flatten-order
// mismatch can scramble is WHICH weight columns each channel's value
// lands on — so red/green/blue patch-embed outputs computed under the
// WRONG convention still differ from each other (channels land on
// *some* columns), just not in the way the trained weight actually
// encodes color. Distinctness alone doesn't prove correctness, but a
// convention that collapses two colors toward each other (a wrong
// permutation projecting through near-duplicate weight-column groups)
// is falsified by this probe — and the real caption run over the
// remaining candidates confirms the winner. This avoids the ~2 min
// full vision-tower encode per guess the file header's real
// integration test pays.
//
// Run explicitly (opted-in, not part of the default suite):
//   BUTTER_RUN_MODEL_TESTS=1 swift test --filter MuseGlimmerPatchEmbedOrderingProbe

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite(
    "Muse Glimmer patch-embed flatten-order probe", .serialized,
    .enabled(if: modelTestsOptedIn))
struct MuseGlimmerPatchEmbedOrderingProbeTests {

    /// SigLIP-normalized (mean 0.5, std 0.5) per-channel value for a
    /// solid-color image: `(v - 0.5) / 0.5`.
    private func normalized(_ v: Float) -> Float { (v - 0.5) / 0.5 }

    /// One candidate 1176-column flatten convention. `index(t, ch, py, px)`
    /// returns the column a given `(temporal, channel, patchRow,
    /// patchCol)` element lands on.
    private struct Ordering {
        let name: String
        let index: (_ t: Int, _ ch: Int, _ py: Int, _ px: Int) -> Int
    }

    /// Build a raw (pre-padding) 1176-length patch vector for a solid
    /// `(r, g, b)` color under `ordering`, with the given per-temporal-
    /// slot channel values (`duplicate` — same color in both slots —
    /// unless `secondSlotZero` requests the alternate "zero second
    /// slot" hypothesis for suspect #2).
    private func buildPatchVector(
        r: Float, g: Float, b: Float,
        c: Int, tp: Int, p: Int,
        ordering: Ordering, secondSlotZero: Bool
    ) -> [Float] {
        let channelVals = [normalized(r), normalized(g), normalized(b)]
        var vec = [Float](repeating: 0, count: c * tp * p * p)
        for t in 0 ..< tp {
            for ch in 0 ..< c {
                let v = (secondSlotZero && t == 1) ? 0 : channelVals[ch]
                for py in 0 ..< p {
                    for px in 0 ..< p {
                        vec[ordering.index(t, ch, py, px)] = v
                    }
                }
            }
        }
        return vec
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        for i in 0 ..< a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = (na.squareRoot() * nb.squareRoot())
        return denom > 0 ? dot / denom : 0
    }

    private func l2(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0
        for i in 0 ..< a.count { s += (a[i] - b[i]) * (a[i] - b[i]) }
        return s.squareRoot()
    }

    @Test("patch-embed output distinctness across candidate flatten orderings")
    func flattenOrderingDistinctness() throws {
        let path = ("~/models/muse-glimmer-30b-4bit" as NSString).expandingTildeInPath
        let dir = URL(fileURLWithPath: path)
        let config = try ModelConfig.load(from: dir)
        let device = Device.shared
        let bundle = try SafeTensorsBundle(directory: dir, device: device)
        let towerWeights = bundle.prefixed("vision_tower.")
        let rawPatch = try towerWeights.tensor(named: "patch_embedder.patch_embedding.weight")

        let visionConfig = try #require(config.subConfig("vision_config"))
        let cfg = try MuseGlimmerVisionConfig.decode(visionConfig)
        let c = cfg.inChannels
        let tp = cfg.temporalPatchSize
        let p = cfg.patchSize
        let patchDim = c * tp * p * p
        #expect(rawPatch.shape == [cfg.hidden, patchDim])
        let patchDimPadded = ((patchDim + gemmKTileWidth - 1) / gemmKTileWidth) * gemmKTileWidth
        let patchEmbedWeight = padLinearColsTo(rawPatch, toCols: patchDimPadded, device: device)

        // ── Candidate flatten orderings ──
        // A: (t, ch, py, px) — CURRENT production `unfoldPatches` order,
        //    matches the Qwen-VL family's `flattenPatchEmbed` dst
        //    convention (temporal-major).
        // B: (ch, t, py, px) — channel-major, temporal next: what a
        //    naive `weight.reshape(hidden, -1)` on a channel-first
        //    PyTorch Conv3d `[hidden, inCh, tP, H, W]` tensor produces
        //    with NO axis permute.
        // C: (t, py, px, ch) — temporal-major, channels-LAST: what a
        //    naive reshape on an MLX/channels-last export
        //    `[hidden, tP, H, W, inCh]` produces with no permute.
        // D: (ch, py, px, t) — channel-major, temporal innermost.
        // E: (py, px, t, ch) — spatial-major, channels-last.
        // F: (py, px, ch, t) — spatial-major, temporal innermost.
        let orderings: [Ordering] = [
            Ordering(name: "A (t,ch,py,px) [current]") { t, ch, py, px in
                ((t * c + ch) * p + py) * p + px
            },
            Ordering(name: "B (ch,t,py,px)") { t, ch, py, px in
                ((ch * tp + t) * p + py) * p + px
            },
            Ordering(name: "C (t,py,px,ch)") { t, ch, py, px in
                ((t * p + py) * p + px) * c + ch
            },
            Ordering(name: "D (ch,py,px,t)") { t, ch, py, px in
                ((ch * p + py) * p + px) * tp + t
            },
            Ordering(name: "E (py,px,t,ch)") { t, ch, py, px in
                ((py * p + px) * tp + t) * c + ch
            },
            Ordering(name: "F (py,px,ch,t)") { t, ch, py, px in
                ((py * p + px) * c + ch) * tp + t
            },
        ]

        func embed(_ raw: [Float]) -> [Float] {
            var padded = raw
            padded.append(contentsOf: [Float](repeating: 0, count: patchDimPadded - raw.count))
            let x = ImagePreprocessing.makeTensor(
                from: padded, shape: [1, patchDimPadded],
                dtype: patchEmbedWeight.dtype, device: device)
            let cmd = device.makeCommandBuffer()
            let out = Ops.gemm(weight: patchEmbedWeight, input: x, nRows: 1, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            return out.toFloatArray()
        }

        print(
            "\n=== Muse Glimmer patch-embed flatten-order probe "
                + "(duplicate-temporal hypothesis) ===")
        func fmt(_ v: Float) -> String { String(format: "%8.4f", v) }
        print("ordering                      cos(R,G)  cos(R,B)  cos(G,B)  |  L2(R,G)  L2(R,B)  L2(G,B)")

        var results: [(name: String, meanCos: Float, meanL2: Float)] = []
        for ordering in orderings {
            let rVec = buildPatchVector(
                r: 1, g: 0, b: 0, c: c, tp: tp, p: p, ordering: ordering, secondSlotZero: false)
            let gVec = buildPatchVector(
                r: 0, g: 1, b: 0, c: c, tp: tp, p: p, ordering: ordering, secondSlotZero: false)
            let bVec = buildPatchVector(
                r: 0, g: 0, b: 1, c: c, tp: tp, p: p, ordering: ordering, secondSlotZero: false)
            let rE = embed(rVec)
            let gE = embed(gVec)
            let bE = embed(bVec)

            let cosRG = cosine(rE, gE)
            let cosRB = cosine(rE, bE)
            let cosGB = cosine(gE, bE)
            let l2RG = l2(rE, gE)
            let l2RB = l2(rE, bE)
            let l2GB = l2(gE, bE)
            let meanCos = (cosRG + cosRB + cosGB) / 3
            let meanL2 = (l2RG + l2RB + l2GB) / 3
            results.append((ordering.name, meanCos, meanL2))

            let padded = ordering.name.padding(toLength: 28, withPad: " ", startingAt: 0)
            print(
                "\(padded) \(fmt(cosRG)) \(fmt(cosRB)) \(fmt(cosGB))  |  "
                    + "\(fmt(l2RG)) \(fmt(l2RB)) \(fmt(l2GB))")
        }

        let winner = results.min { $0.meanCos < $1.meanCos }
        print("\nLowest mean cosine (most distinct): \(winner?.name ?? "?")")
        let winnerByL2 = results.max { $0.meanL2 < $1.meanL2 }
        print("Highest mean L2 (most distinct): \(winnerByL2?.name ?? "?")")

        // ── Suspect #2: temporal duplication vs zero-fill, on the
        //    CURRENT ordering (A) — probe whether zeroing the second
        //    temporal slot changes distinctness materially.
        print("\n=== Temporal duplication vs zero-fill (ordering A) ===")
        let orderingA = orderings[0]
        for secondSlotZero in [false, true] {
            let rVec = buildPatchVector(
                r: 1, g: 0, b: 0, c: c, tp: tp, p: p, ordering: orderingA,
                secondSlotZero: secondSlotZero)
            let gVec = buildPatchVector(
                r: 0, g: 1, b: 0, c: c, tp: tp, p: p, ordering: orderingA,
                secondSlotZero: secondSlotZero)
            let bVec = buildPatchVector(
                r: 0, g: 0, b: 1, c: c, tp: tp, p: p, ordering: orderingA,
                secondSlotZero: secondSlotZero)
            let rE = embed(rVec)
            let gE = embed(gVec)
            let bE = embed(bVec)
            let meanCos = (cosine(rE, gE) + cosine(rE, bE) + cosine(gE, bE)) / 3
            let meanL2 = (l2(rE, gE) + l2(rE, bE) + l2(gE, bE)) / 3
            print(
                "secondSlotZero=\(secondSlotZero): meanCos=\(meanCos) meanL2=\(meanL2)")
        }

        #expect(!results.isEmpty)
    }

    /// Suspect #4 follow-up: the full-tower probe found color-only
    /// image pairs (`cos ≈ 0.9993`) LESS distinct than same-color
    /// different-SHAPE pairs (`cos ≈ 0.998`) — real evidence of a
    /// color-specific signal loss, not just baseline ViT-embedding-
    /// space similarity. `ln_pre` (applied immediately after
    /// `patchEmbed + posEmb`, before any block) and `norm1`/`norm2`
    /// inside every one of the 50 blocks all re-center each row to
    /// zero mean. If the color-vs-color difference in the raw
    /// patch-embed vector lives mostly in the per-row MEAN (a uniform
    /// additive shift across all 1536 dims) rather than in the
    /// per-dimension VARIANCE pattern, LayerNorm's mean-subtraction
    /// would attenuate almost all of it on the very first pass — while
    /// shape differences (which show up as across-PATCH structure, not
    /// a per-row constant) survive LayerNorm relatively unscathed. This
    /// decomposes the current-ordering (A) patch-embed output for
    /// red/green/blue into mean-shift vs residual-after-mean-removal to
    /// check that hypothesis directly, with no tower forward needed.
    @Test("mean-shift vs residual decomposition of patch-embed color signal")
    func meanShiftVsResidualDecomposition() throws {
        let path = ("~/models/muse-glimmer-30b-4bit" as NSString).expandingTildeInPath
        let dir = URL(fileURLWithPath: path)
        let config = try ModelConfig.load(from: dir)
        let device = Device.shared
        let bundle = try SafeTensorsBundle(directory: dir, device: device)
        let towerWeights = bundle.prefixed("vision_tower.")
        let rawPatch = try towerWeights.tensor(named: "patch_embedder.patch_embedding.weight")

        let visionConfig = try #require(config.subConfig("vision_config"))
        let cfg = try MuseGlimmerVisionConfig.decode(visionConfig)
        let c = cfg.inChannels
        let tp = cfg.temporalPatchSize
        let p = cfg.patchSize
        let patchDim = c * tp * p * p
        let patchDimPadded = ((patchDim + gemmKTileWidth - 1) / gemmKTileWidth) * gemmKTileWidth
        let patchEmbedWeight = padLinearColsTo(rawPatch, toCols: patchDimPadded, device: device)

        let ordering = Ordering(name: "A (t,ch,py,px) [current]") { t, ch, py, px in
            ((t * c + ch) * p + py) * p + px
        }

        func embed(r: Float, g: Float, b: Float) -> [Float] {
            let raw = buildPatchVector(
                r: r, g: g, b: b, c: c, tp: tp, p: p, ordering: ordering, secondSlotZero: false)
            var padded = raw
            padded.append(contentsOf: [Float](repeating: 0, count: patchDimPadded - raw.count))
            let x = ImagePreprocessing.makeTensor(
                from: padded, shape: [1, patchDimPadded],
                dtype: patchEmbedWeight.dtype, device: device)
            let cmd = device.makeCommandBuffer()
            let out = Ops.gemm(weight: patchEmbedWeight, input: x, nRows: 1, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            return out.toFloatArray()
        }

        func meanAndResidualNorm(_ v: [Float]) -> (mean: Float, residualNorm: Float) {
            let mean = v.reduce(0, +) / Float(v.count)
            var resNorm: Float = 0
            for x in v { resNorm += (x - mean) * (x - mean) }
            return (mean, resNorm.squareRoot())
        }

        let rE = embed(r: 1, g: 0, b: 0)
        let gE = embed(r: 0, g: 1, b: 0)
        let bE = embed(r: 0, g: 0, b: 1)

        let (rMean, rRes) = meanAndResidualNorm(rE)
        let (gMean, gRes) = meanAndResidualNorm(gE)
        let (bMean, bRes) = meanAndResidualNorm(bE)

        // How much of the R-vs-G difference vector's total L2 norm is
        // explained by the (scalar, broadcast) mean-shift alone, vs the
        // residual (post-mean-removal, what LayerNorm keeps)?
        func decomposeDiff(_ a: [Float], _ aMean: Float, _ b: [Float], _ bMean: Float) -> (
            meanShiftNorm: Float, residualDiffNorm: Float, totalDiffNorm: Float
        ) {
            let meanDelta = aMean - bMean
            let meanShiftNorm = abs(meanDelta) * Float(a.count).squareRoot()
            var residualDiffSq: Float = 0
            var totalDiffSq: Float = 0
            for i in 0 ..< a.count {
                let aRes = a[i] - aMean
                let bRes = b[i] - bMean
                residualDiffSq += (aRes - bRes) * (aRes - bRes)
                totalDiffSq += (a[i] - b[i]) * (a[i] - b[i])
            }
            return (meanShiftNorm, residualDiffSq.squareRoot(), totalDiffSq.squareRoot())
        }

        let rg = decomposeDiff(rE, rMean, gE, gMean)
        let rb = decomposeDiff(rE, rMean, bE, bMean)
        let gb = decomposeDiff(gE, gMean, bE, bMean)

        print("\n=== patch-embed mean-shift vs residual decomposition (ordering A) ===")
        print("row means: R=\(rMean) G=\(gMean) B=\(bMean)")
        print("row residual (post-mean) L2 norms: R=\(rRes) G=\(gRes) B=\(bRes)")
        print(
            "R-G diff: total=\(rg.totalDiffNorm) meanShift=\(rg.meanShiftNorm) "
                + "residual=\(rg.residualDiffNorm) meanShift²/total²="
                + "\(Float(pow(rg.meanShiftNorm, 2) / pow(rg.totalDiffNorm, 2)))")
        print(
            "R-B diff: total=\(rb.totalDiffNorm) meanShift=\(rb.meanShiftNorm) "
                + "residual=\(rb.residualDiffNorm) meanShift²/total²="
                + "\(Float(pow(rb.meanShiftNorm, 2) / pow(rb.totalDiffNorm, 2)))")
        print(
            "G-B diff: total=\(gb.totalDiffNorm) meanShift=\(gb.meanShiftNorm) "
                + "residual=\(gb.residualDiffNorm) meanShift²/total²="
                + "\(Float(pow(gb.meanShiftNorm, 2) / pow(gb.totalDiffNorm, 2)))")

        #expect(rRes > 0)
    }
}
