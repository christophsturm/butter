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
// Pure-geometry unit tests for the Gemma 4 variable-resolution image
// processor — `Gemma4VLVisionModel.aspectPreservingSize`. No model load /
// GPU: validates the aspect-ratio-preserving target dims (and therefore the
// soft-token count) across landscape / portrait / square / extreme-strip
// inputs against the transformers reference
// (`get_aspect_ratio_preserving_size`, `image_processing_pil_gemma4`).
// A single landscape integration image (the dog) only ever exercises one
// aspect ratio; these cover the non-square `gridH × gridW` paths in both
// orientations plus the degenerate clamps.

import Testing

@testable import FFAI

@Suite("Gemma4 Vision Geometry")
struct Gemma4VisionGeometryTests {

    /// Gemma 4 e2b vision config: patch 16, pooling kernel 3, 280 default
    /// soft tokens. `sideMult = 48`, `maxPatches = 280·9 = 2520`.
    private func size(_ w: Int, _ h: Int) -> (h: Int, w: Int) {
        Gemma4VLVisionModel.aspectPreservingSize(
            width: w, height: h,
            patchSize: 16, poolingKernelSize: 3, maxSoftTokens: 280)
    }

    /// Soft-token count for a target `(H, W)` — `(H/16)·(W/16) / 9`.
    private func tokens(_ s: (h: Int, w: Int)) -> Int {
        (s.h / 16) * (s.w / 16) / 9
    }

    @Test("landscape — matches the reference (and the dog oracle: 266 tokens)")
    func landscape() {
        let s = size(863, 625)
        #expect(s == (h: 672, w: 912))
        #expect(tokens(s) == 266)
    }

    @Test("portrait — mirror of landscape (H/W swapped), same token count")
    func portrait() {
        let s = size(625, 863)
        #expect(s == (h: 912, w: 672))
        #expect(tokens(s) == 266)
    }

    @Test("square — near-square grid, 256 tokens")
    func square() {
        let s = size(700, 700)
        #expect(s == (h: 768, w: 768))
        #expect(tokens(s) == 256)
    }

    @Test("wide and tall are mirror images")
    func wideTall() {
        let wide = size(1600, 400)
        let tall = size(400, 1600)
        #expect(wide == (h: 384, w: 1584))
        #expect(tall == (h: 1584, w: 384))
        #expect(tokens(wide) == 264)
        #expect(tokens(tall) == 264)
    }

    @Test("small image upsizes into the patch budget")
    func small() {
        let s = size(100, 50)
        #expect(s == (h: 528, w: 1104))
        #expect(tokens(s) == 253)
    }

    @Test("degenerate strip — extreme aspect ratio stays valid + divisible")
    func degenerateStrip() {
        let s = size(2000, 30)
        #expect(s == (h: 96, w: 6528))
        #expect(tokens(s) == 272)
        // Both dims divisible by sideMult (48) — required so the patch grid
        // and the 3×3 pooling tile both land cleanly.
        #expect(s.h % 48 == 0 && s.w % 48 == 0)
        // Never exceeds the patch budget (2520).
        #expect((s.h / 16) * (s.w / 16) <= 2520)
    }
}
