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

// Unit tests for the Qwen 2-VL family (the full-attention
// ViT + Qwen 2 text backbone `Qwen2VLForConditionalGeneration` checkpoint).
//
// Offline — covers VL routing and `Qwen2VLVisionConfig.decode`, which
// turns the nested `vision_config` into the ViT geometry (depth, head dim,
// spatial-merge unit) the vision tower is built from.
@Suite("Qwen2 Vision")
struct Qwen2VisionConfigTests {

    /// A representative `Qwen2VLForConditionalGeneration` config matching
    /// the mlx-community/Qwen2-VL-2B-Instruct-4bit checkpoint layout.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "depth": 32,
            "embed_dim": 1280,  // vision hidden (Qwen2-VL naming)
            "hidden_size": 1536,  // text backbone hidden (merger output)
            "mlp_ratio": 4.0,
            "num_heads": 16,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_chans": 3,
        ]
        let raw: [String: Any] = [
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "model_type": "qwen2_vl",
            "image_token_id": 151_655,
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "vision_config": visionConfig,
        ]
        return ModelConfig(
            architecture: "Qwen2VLForConditionalGeneration",
            modelType: "qwen2_vl", raw: raw)
    }

    @Test("routes as a vision-language checkpoint")
    func routesAsVisionLanguage() {
        let cfg = makeConfig()
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        #expect(
            VisionLanguageArchitectures.architectures
                .contains("Qwen2VLForConditionalGeneration"))
        #expect(Qwen2VL.defaultImageTokenId == 151_655)
        #expect(cfg.int("image_token_id") == 151_655)
    }

    @Test("vision_config decodes into ViT geometry")
    func visionConfigDecode() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try Qwen2VLVisionConfig.decode(vc)
        #expect(parsed.depth == 32)
        #expect(parsed.hidden == 1280)  // embed_dim
        #expect(parsed.outHidden == 1536)  // hidden_size (text dim)
        #expect(parsed.numHeads == 16)
        #expect(parsed.patchSize == 14)
        #expect(parsed.spatialMergeSize == 2)
        #expect(parsed.temporalPatchSize == 2)
        #expect(parsed.inChannels == 3)
        // Derived geometry.
        #expect(parsed.headDim == 80)  // 1280 / 16
        #expect(parsed.mergeUnit == 4)  // 2 × 2 patches per merged token
        // intermediate = mlp_ratio × hidden = 4 × 1280 = 5120
        #expect(parsed.intermediate == 5120)
    }

    @Test("vision_config decode falls back to documented defaults")
    func visionConfigDefaults() throws {
        // Minimal config — only the required fields. `decode` fills
        // mlp_ratio / temporal_patch_size / in_channels.
        let minimal = ModelConfig(
            architecture: nil, modelType: nil,
            raw: [
                "depth": 16,
                "embed_dim": 1024,
                "num_heads": 8,
                "patch_size": 14,
                "spatial_merge_size": 2,
            ])
        let parsed = try Qwen2VLVisionConfig.decode(minimal)
        // mlp_ratio defaults to 4.0
        #expect(parsed.intermediate == 4096)  // 4 × 1024
        // outHidden falls back to hidden when hidden_size absent
        #expect(parsed.outHidden == 1024)
        #expect(parsed.temporalPatchSize == 2)
        #expect(parsed.inChannels == 3)
        #expect(parsed.layerNormEps == 1e-6)
    }
}

// Offline geometry tests for the Qwen2-VL native-resolution (`smartResize`)
// path: the placeholder-count helper (`smartResizeGrid`) and the
// tensor-shape → patch-grid derivation (`gridDims`). No GPU / model load —
// validates the variable-resolution geometry that replaced the fixed 392²
// square grid (the merged-token count a caller uses to size the
// image-placeholder run, and the non-square grid the tower derives).
@Suite("Qwen2 Vision native resolution")
struct Qwen2VisionSmartResizeTests {

    /// Standard Qwen2-VL vision config (patch-14, merge-2 → factor 28).
    private func config() throws -> Qwen2VLVisionConfig {
        try Qwen2VLVisionConfig.decode(
            ModelConfig(
                architecture: nil, modelType: nil,
                raw: [
                    "depth": 32, "embed_dim": 1280, "hidden_size": 1536,
                    "num_heads": 16, "patch_size": 14,
                    "spatial_merge_size": 2, "temporal_patch_size": 2,
                ]))
    }

    @Test("legacy square resolution → 28×28 grid, 196 merged tokens")
    func squareGrid() throws {
        let cfg = try config()
        let g = try Qwen2VLVisionModel.smartResizeGrid(
            width: 392, height: 392, cfg: cfg)
        #expect(g.gridH == 28 && g.gridW == 28)
        #expect(g.mergedTokens == 196)  // (28/2)²
    }

    @Test("landscape image → non-square grid, aspect preserved")
    func landscapeGrid() throws {
        let cfg = try config()
        // 1024×768 → smartResize (756, 1036) → grid 54×74.
        let g = try Qwen2VLVisionModel.smartResizeGrid(
            width: 1024, height: 768, cfg: cfg)
        #expect(g.gridH == 54 && g.gridW == 74)
        #expect(g.mergedTokens == 27 * 37)  // (54/2)·(74/2) = 999
    }

    @Test("portrait image → transposed non-square grid")
    func portraitGrid() throws {
        let cfg = try config()
        let g = try Qwen2VLVisionModel.smartResizeGrid(
            width: 768, height: 1024, cfg: cfg)
        #expect(g.gridH == 74 && g.gridW == 54)
        #expect(g.mergedTokens == 37 * 27)  // 999, mirror of landscape
    }

    @Test("tiny image is up-scaled to the minimum grid")
    func tinyGrid() throws {
        let cfg = try config()
        // 28×28 → smartResize (56, 56) → grid 4×4 → 4 merged tokens.
        let g = try Qwen2VLVisionModel.smartResizeGrid(
            width: 28, height: 28, cfg: cfg)
        #expect(g.gridH == 4 && g.gridW == 4)
        #expect(g.mergedTokens == 4)
    }

    @Test("gridDims derives the patch grid from a preprocessed tensor")
    func gridDimsFromTensor() throws {
        let cfg = try config()
        // [1, 3, 56, 84] (H=56, W=84) → grid (56/14, 84/14) = (4, 6).
        let pixels = [Float](repeating: 0, count: 1 * 3 * 56 * 84)
        let t = MediaUtils.makeTensor(
            from: pixels, shape: [1, 3, 56, 84], dtype: .f32, device: .shared)
        let (gridH, gridW) = Qwen2VLVisionModel.gridDims(for: t, cfg: cfg)
        #expect(gridH == 4 && gridW == 6)
    }

    @Test("extreme aspect ratio is rejected (unsupported size)")
    func extremeAspectRejected() throws {
        let cfg = try config()
        #expect(throws: MediaUtils.SmartResizeError.self) {
            _ = try Qwen2VLVisionModel.smartResizeGrid(
                width: 6000, height: 28, cfg: cfg)
        }
    }
}
