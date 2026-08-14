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
// DIAGNOSTIC PROBE — not a permanent regression test. Runs the FULL
// production `MuseGlimmerVisionModel` (patch-embed → learned position
// embed → 50-layer transformer stack with windowed/full attention +
// 2-D RoPE → ln_post → merge → quantized adapter/projection) on solid
// red/green/blue images and reports how distinct the FINAL projected
// (text-hidden-dim) embeddings are.
//
// WHY THIS INSTEAD OF AN ACTUAL CAPTION: `vlm.generate(...)` /
// `m.generate(...)` on this checkpoint allocate a KV cache sized to
// `config.text_config.max_position_embeddings` (131072) — MuseGlimmer's
// loader (`MuseGlimmerText.swift`) hardcodes this and does not honor
// `LoadOptions.maxContextLength` the way Laguna/Hy3/NemotronDiffusion
// do (grep confirms no `maxContextLength` reference in
// `MuseGlimmerText.swift`) — so ANY `generate()` call needs ~18 GB
// weights + ~21 GB KV cache ≈ 40+ GB RSS, which blew straight through
// this session's 40 GB box-priority self-watch ceiling during a real
// attempt (observed 9.5 → 19.4 → 36.7 → 38.2 → 46.9 GB in under a
// minute, killed by the watchdog). `encodeImage`/`VisionModel.encode`
// needs NO KV cache — pure feedforward — so it's the deepest
// production code path this session can safely exercise. This is a
// REAL GAP worth fixing in `MuseGlimmerText.swift` (wire
// `options.maxContextLength` the way other families do) but is out of
// scope for a vision-only color-fidelity fix.
//
// Run explicitly (opted-in, not part of the default suite):
//   BUTTER_RUN_MODEL_TESTS=1 swift test --filter MuseGlimmerVisionTowerColorProbe

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite(
    "Muse Glimmer full-tower color-distinctness probe", .serialized,
    .enabled(if: modelTestsOptedIn))
struct MuseGlimmerVisionTowerColorProbeTests {

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        for i in 0 ..< a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = na.squareRoot() * nb.squareRoot()
        return denom > 0 ? dot / denom : 0
    }

    private func l2(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0
        for i in 0 ..< a.count { s += (a[i] - b[i]) * (a[i] - b[i]) }
        return s.squareRoot()
    }

    /// Mean-pool a `[nTokens, hidden]` embedding down to `[hidden]` —
    /// collapses the 256 merged tokens of a solid-color image (which
    /// should all be near-identical) into one comparable vector.
    private func meanPool(_ t: Tensor) -> [Float] {
        let vals = t.toFloatArray()
        let hidden = t.shape[1]
        let nRows = t.shape[0]
        var out = [Float](repeating: 0, count: hidden)
        for r in 0 ..< nRows {
            for d in 0 ..< hidden { out[d] += vals[r * hidden + d] }
        }
        for d in 0 ..< hidden { out[d] /= Float(nRows) }
        return out
    }

    @Test("full-tower projected-embedding distinctness for solid red/green/blue")
    func fullTowerDistinctness() throws {
        let path = ("~/models/muse-glimmer-30b-4bit" as NSString).expandingTildeInPath
        let dir = URL(fileURLWithPath: path)
        let config = try ModelConfig.load(from: dir)
        let device = Device.shared
        let bundle = try SafeTensorsBundle(directory: dir, device: device)
        let visionConfig = try #require(config.subConfig("vision_config"))
        let textHidden = config.subConfig("text_config")?.int("hidden_size") ?? 6656

        let vision = try MuseGlimmerVisionModel.load(
            visionConfig: visionConfig, textHidden: textHidden,
            weights: bundle, quantization: config.quantization,
            dtype: .bf16, device: device)
        let encoder = vision.asVisionEncoder()
        let side = vision.gridSide * vision.cfg.patchSize  // 448

        func embed(r: Float, g: Float, b: Float) -> [Float] {
            let image = RGBImage.solid(width: side, height: side, r: r, g: g, b: b)
            let pixels = ImagePreprocessing.preprocess(
                image, targetW: side, targetH: side,
                normalization: .siglip, dtype: .bf16, device: device)
            let out = encoder.encode(image: pixels, device: device)
            return meanPool(out)
        }

        let start = CFAbsoluteTimeGetCurrent()
        let rE = embed(r: 1, g: 0, b: 0)
        print("red encode: \((CFAbsoluteTimeGetCurrent() - start) * 1000) ms")
        let gE = embed(r: 0, g: 1, b: 0)
        let bE = embed(r: 0, g: 0, b: 1)

        let cosRG = cosine(rE, gE)
        let cosRB = cosine(rE, bE)
        let cosGB = cosine(gE, bE)
        let l2RG = l2(rE, gE)
        let l2RB = l2(rE, bE)
        let l2GB = l2(gE, bE)

        print("\n=== Muse Glimmer FULL-TOWER projected-embedding distinctness ===")
        print("cos(R,G)=\(cosRG) cos(R,B)=\(cosRB) cos(G,B)=\(cosGB)")
        print("L2(R,G)=\(l2RG) L2(R,B)=\(l2RB) L2(G,B)=\(l2GB)")

        var rNorm: Float = 0
        for v in rE { rNorm += v * v }
        rNorm = rNorm.squareRoot()
        print("‖R embedding‖ = \(rNorm), dim = \(rE.count)")

        #expect(rE.count == textHidden)
    }

    /// Suspect #4 groundwork: compare the MAGNITUDE of the raw
    /// patch-embed signal (which carries the color-distinguishing
    /// information, per the flatten-order probe) against the learned
    /// absolute position-embedding table it gets summed with
    /// (`h = Ops.add(patchEmbed, posEmb)`, pre-`ln_pre`). If posEmb's
    /// norm dominates patchEmbed's, the sum is posEmb-with-a-small-
    /// color-perturbation for EVERY color — which would explain the
    /// full-tower probe's near-identical (cos ≈ 0.999) final embeddings
    /// despite patch-embed itself being strongly color-distinct
    /// (cos ≈ -0.43 in the flatten-order probe).
    @Test("patch-embed vs position-embedding magnitude")
    func patchEmbedVsPosEmbedMagnitude() throws {
        let path = ("~/models/muse-glimmer-30b-4bit" as NSString).expandingTildeInPath
        let dir = URL(fileURLWithPath: path)
        let config = try ModelConfig.load(from: dir)
        let device = Device.shared
        let bundle = try SafeTensorsBundle(directory: dir, device: device)
        let visionConfig = try #require(config.subConfig("vision_config"))
        let textHidden = config.subConfig("text_config")?.int("hidden_size") ?? 6656

        let vision = try MuseGlimmerVisionModel.load(
            visionConfig: visionConfig, textHidden: textHidden,
            weights: bundle, quantization: config.quantization,
            dtype: .bf16, device: device)

        // Raw patch-embed output for a solid-red patch (reuses the
        // production `patchEmbedWeight`, current [t,ch,py,px] ordering).
        let cfg = vision.cfg
        let c = cfg.inChannels
        let tp = cfg.temporalPatchSize
        let p = cfg.patchSize
        let patchDim = c * tp * p * p
        var raw = [Float](repeating: 0, count: patchDim)
        for t in 0 ..< tp {
            for ch in 0 ..< c {
                let v: Float = ch == 0 ? 1.0 : -1.0  // normalized solid red
                for py in 0 ..< p {
                    for px in 0 ..< p {
                        raw[((t * c + ch) * p + py) * p + px] = v
                    }
                }
            }
        }
        var padded = raw
        padded.append(
            contentsOf: [Float](repeating: 0, count: vision.patchDimPadded - raw.count))
        let x = ImagePreprocessing.makeTensor(
            from: padded, shape: [1, vision.patchDimPadded],
            dtype: vision.patchEmbedWeight.dtype, device: device)
        let cmd = device.makeCommandBuffer()
        let out = Ops.gemm(weight: vision.patchEmbedWeight, input: x, nRows: 1, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        let patchEmbedVals = out.toFloatArray()
        var patchNorm: Float = 0
        for v in patchEmbedVals { patchNorm += v * v }
        patchNorm = patchNorm.squareRoot()

        let posVals = vision.posEmbedTable.toFloatArray()
        let hidden = cfg.hidden
        let nPosRows = vision.posEmbedTable.shape[0]
        var meanRowNorm: Float = 0
        var maxRowNorm: Float = 0
        for row in 0 ..< nPosRows {
            var n: Float = 0
            for d in 0 ..< hidden {
                let v = posVals[row * hidden + d]
                n += v * v
            }
            n = n.squareRoot()
            meanRowNorm += n
            maxRowNorm = max(maxRowNorm, n)
        }
        meanRowNorm /= Float(nPosRows)

        print("\n=== patch-embed vs position-embedding magnitude ===")
        print("‖patchEmbed(red)‖ = \(patchNorm)  (dim \(hidden))")
        print("mean ‖posEmbedTable row‖ = \(meanRowNorm), max = \(maxRowNorm)  (over \(nPosRows) rows)")
        print("ratio meanPos/patchEmbed = \(meanRowNorm / patchNorm)")

        #expect(patchNorm > 0)
    }

    /// Context for the `cos ≈ 0.999` full-tower result above: is that
    /// unusually high (color signal specifically getting lost) or just
    /// how this tower's embedding space normally looks (a large shared
    /// "generic image" direction with real content in a small residual
    /// — a known property of CLIP/SigLIP-family embedding spaces)?
    /// Compares solid-red-square vs solid-green-square (color-only
    /// difference) against solid-red-square vs the green-circle fixture
    /// (color AND shape difference). The bug report says SHAPE is
    /// recognized correctly but COLOR is not — if that's real, a
    /// structurally different image should be markedly LESS similar
    /// than a same-shape different-color image. If the two cosines are
    /// close, the tower isn't distinguishing shape much better than
    /// color either, and the bug is more likely downstream (splice /
    /// decode) than in the tower itself.
    @Test("color-only vs shape+color distinctness (context for the cos≈0.999 result)")
    func colorVsShapeDistinctness() throws {
        let path = ("~/models/muse-glimmer-30b-4bit" as NSString).expandingTildeInPath
        let dir = URL(fileURLWithPath: path)
        let config = try ModelConfig.load(from: dir)
        let device = Device.shared
        let bundle = try SafeTensorsBundle(directory: dir, device: device)
        let visionConfig = try #require(config.subConfig("vision_config"))
        let textHidden = config.subConfig("text_config")?.int("hidden_size") ?? 6656

        let vision = try MuseGlimmerVisionModel.load(
            visionConfig: visionConfig, textHidden: textHidden,
            weights: bundle, quantization: config.quantization,
            dtype: .bf16, device: device)
        let encoder = vision.asVisionEncoder()
        let side = vision.gridSide * vision.cfg.patchSize  // 448

        func embed(_ image: RGBImage) -> [Float] {
            let pixels = ImagePreprocessing.preprocess(
                image, targetW: side, targetH: side,
                normalization: .siglip, dtype: .bf16, device: device)
            let out = encoder.encode(image: pixels, device: device)
            return meanPool(out)
        }

        let redSquare = RGBImage.solid(width: side, height: side, r: 1, g: 0, b: 0)
        let greenSquare = RGBImage.solid(width: side, height: side, r: 0, g: 1, b: 0)
        let circleURL = resourceURL("muse_glimmer_green_circle.png")
        let greenCircle = try RGBImage.load(contentsOf: circleURL)

        let rE = embed(redSquare)
        let gSquareE = embed(greenSquare)
        let gCircleE = embed(greenCircle)

        let cosColorOnly = cosine(rE, gSquareE)  // red square vs green square
        let cosShapeAndColor = cosine(rE, gCircleE)  // red square vs green circle
        let cosSameColorDiffShape = cosine(gSquareE, gCircleE)  // green square vs green circle

        print("\n=== color-only vs shape+color distinctness ===")
        print("cos(red square, green square)  [color only]        = \(cosColorOnly)")
        print("cos(red square, green circle)  [color + shape]      = \(cosShapeAndColor)")
        print("cos(green square, green circle) [shape only]        = \(cosSameColorDiffShape)")

        #expect(rE.count == textHidden)
    }
}
