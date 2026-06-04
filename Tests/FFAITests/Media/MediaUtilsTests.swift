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
import TestHelpers
import Testing

@testable import FFAI

// Unit tests for `MediaUtils.smartResize` — the pure native-resolution
// geometry helper the Qwen-family / GLM-OCR image processors use. No GPU,
// no model load: validates the HF `smart_resize` algorithm (round to a
// multiple of `factor`, scale into the `[minPixels, maxPixels]` budget,
// throw on extreme aspect ratios) against hand-computed reference values.
@Suite("MediaUtils smartResize")
struct MediaUtilsSmartResizeTests {

    /// Standard Qwen2-VL bounds: patch-14, merge-2 → factor 28, min 56²
    /// (3136), max 28²·1280 (1003520).
    private let qwen2 = MediaUtils.SmartResizeBounds.qwen(
        patchSize: 14, mergeSize: 2)

    private func resize(_ h: Int, _ w: Int) throws -> (h: Int, w: Int) {
        try MediaUtils.smartResize(height: h, width: w, bounds: qwen2)
    }

    /// Every returned side is a multiple of `factor` — guarantees a whole
    /// patch grid that the 2×2 merger can pool cleanly.
    private func assertMultipleOfFactor(_ r: (h: Int, w: Int)) {
        #expect(r.h % 28 == 0)
        #expect(r.w % 28 == 0)
    }

    @Test("Qwen2 bounds derive factor + pixel budget from patch/merge")
    func boundsDerivation() {
        #expect(qwen2.factor == 28)  // 14 · 2
        #expect(qwen2.minPixels == 56 * 56)  // 4 · 28²
        #expect(qwen2.maxPixels == 28 * 28 * 1280)
    }

    @Test("square at the legacy default resolution is unchanged")
    func squareInBudget() throws {
        // 392×392 (28×28 patches) was the old fixed grid; it sits inside
        // the budget and rounds to itself → 28×28 grid → 196 merged tokens.
        let r = try resize(392, 392)
        #expect(r == (h: 392, w: 392))
        assertMultipleOfFactor(r)
        let mergedTokens = (r.h / 28) * (r.w / 28)  // (gridSide/merge)²
        #expect(mergedTokens == 14 * 14)
    }

    @Test("landscape preserves aspect, rounds each side to the factor")
    func landscapeRounds() throws {
        // 1024×768 → round(1024/28)=37·28=1036, round(768/28)=27·28=756.
        // 1036·756 = 783 216 ≤ max → no rescale.
        let r = try resize(768, 1024)
        #expect(r == (h: 756, w: 1036))
        assertMultipleOfFactor(r)
    }

    @Test("portrait preserves aspect (mirror of landscape)")
    func portraitRounds() throws {
        let r = try resize(1024, 768)
        #expect(r == (h: 1036, w: 756))
        assertMultipleOfFactor(r)
    }

    @Test("tiny image is up-scaled to the minimum pixel budget")
    func upscaleToMin() throws {
        // 28×28 rounds to 28×28 (784 px) < min 3136 → β=√(3136/784)=2 →
        // ceil(28·2/28)·28 = 56 → 56×56 = exactly minPixels.
        let r = try resize(28, 28)
        #expect(r == (h: 56, w: 56))
        #expect(r.h * r.w >= qwen2.minPixels)
        assertMultipleOfFactor(r)
    }

    @Test("a small-but-above-min image is not up-scaled")
    func smallInBudget() throws {
        // 100×100 → round(100/28)=4·28=112 → 12544 px ≥ min → kept.
        let r = try resize(100, 100)
        #expect(r == (h: 112, w: 112))
        assertMultipleOfFactor(r)
    }

    @Test("large image is down-scaled under the max pixel budget")
    func downscaleToMax() throws {
        // 4000×3000 rounds over max; β=√(12 000 000/1 003 520)≈3.458 →
        // floor(3000/β/28)·28 = 840, floor(4000/β/28)·28 = 1148.
        let r = try resize(3000, 4000)
        #expect(r == (h: 840, w: 1148))
        #expect(r.h * r.w <= qwen2.maxPixels)
        assertMultipleOfFactor(r)
    }

    @Test("extreme aspect ratio throws (unsupported size)")
    func extremeAspectThrows() {
        // 20×5000 → ratio 250 > 200 → unsupported.
        #expect(throws: MediaUtils.SmartResizeError.self) {
            _ = try resize(20, 5000)
        }
    }

    @Test("aspect ratio at the limit is accepted")
    func aspectAtLimitOK() throws {
        // ratio exactly 200 (≤ max) → no throw.
        let r = try resize(28, 5600)
        assertMultipleOfFactor(r)
    }

    @Test("patch-16 / merge-2 bounds scale the factor (Qwen3 family)")
    func qwen3Factor() throws {
        let bounds = MediaUtils.SmartResizeBounds.qwen(patchSize: 16, mergeSize: 2)
        #expect(bounds.factor == 32)
        let r = try MediaUtils.smartResize(height: 512, width: 512, bounds: bounds)
        #expect(r.h % 32 == 0)
        #expect(r.w % 32 == 0)
    }
}

// Unit tests for `MediaUtils.decodeVideoFrames` — the production video
// decode/sample utility (AVFoundation). Decodes the checked-in `cat.mp4`
// fixture; no model load. This is the path every VLM video integration
// test now flows through (via `VisionTestHelpers.catVideoFrames`).
@Suite("MediaUtils video decode")
struct MediaVideoDecodeTests {
    @Test("decodeVideoFrames samples cat.mp4 to RGB frames in [0,1]")
    func decodeCatVideo() throws {
        let frames = try MediaUtils.decodeVideoFrames(
            url: VisionTestHelpers.catVideoURL(), maxFrames: 4)
        #expect(!frames.isEmpty, "cat.mp4 should decode at least one frame")
        #expect(frames.count <= 4, "must not exceed maxFrames")
        for f in frames {
            #expect(f.width > 0 && f.height > 0, "frame dims must be positive")
            #expect(
                f.pixels.count == f.width * f.height * 3,
                "pixels must be row-major RGB (w·h·3)")
        }
        // Spot-check the pixel range on the first frame.
        let first = try #require(frames.first)
        #expect(
            first.pixels.allSatisfy { $0 >= 0 && $0 <= 1 },
            "decoded pixels must be normalized to [0, 1]")
    }

    @Test("maxFrames=1 yields exactly the t=0 thumbnail frame")
    func decodeSingleFrame() throws {
        let frames = try MediaUtils.decodeVideoFrames(
            url: VisionTestHelpers.catVideoURL(), maxFrames: 1)
        #expect(frames.count == 1)
    }
}

// Unit tests for the consolidated Qwen-family video / merged-token
// geometry helpers (`paddedFrameCount`, `mergedVideoTokenCount`,
// `smartResizeGrid`) that the Qwen2 / 2.5 / 3 VL towers now share. Pure
// integer geometry — no GPU, no model.
@Suite("MediaUtils video geometry")
struct MediaVideoGeometryTests {
    @Test("paddedFrameCount rounds up to a temporal-patch multiple")
    func paddedFrameCount() {
        #expect(MediaUtils.paddedFrameCount(4, temporalPatchSize: 2) == 4)
        #expect(MediaUtils.paddedFrameCount(5, temporalPatchSize: 2) == 6)
        #expect(MediaUtils.paddedFrameCount(1, temporalPatchSize: 2) == 2)
        #expect(MediaUtils.paddedFrameCount(3, temporalPatchSize: 1) == 3)
        #expect(MediaUtils.paddedFrameCount(7, temporalPatchSize: 4) == 8)
    }

    @Test("mergedVideoTokenCount = (T/tp)·(gridSide/m)²")
    func mergedVideoTokenCount() {
        // gridSide 32, merge 2 → llmSide 16. 4 frames, tp 2 → gridT 2.
        // → 2 · 16² = 512.
        #expect(
            MediaUtils.mergedVideoTokenCount(
                frameCount: 4, gridSide: 32,
                spatialMergeSize: 2, temporalPatchSize: 2) == 512)
        // 5 frames padded to 6 → gridT 3 → 3 · 256 = 768.
        #expect(
            MediaUtils.mergedVideoTokenCount(
                frameCount: 5, gridSide: 32,
                spatialMergeSize: 2, temporalPatchSize: 2) == 768)
    }

    @Test("smartResizeGrid returns an aligned grid + merged-token count")
    func smartResizeGrid() throws {
        // 512×512, patch 16, merge 2 → factor 32. The grid sides are
        // multiples of merge; mergedTokens = (gridH/m)·(gridW/m).
        let r = try MediaUtils.smartResizeGrid(
            width: 512, height: 512, patchSize: 16, spatialMergeSize: 2)
        #expect(r.gridH % 2 == 0)
        #expect(r.gridW % 2 == 0)
        #expect(r.mergedTokens == (r.gridH / 2) * (r.gridW / 2))
    }
}
