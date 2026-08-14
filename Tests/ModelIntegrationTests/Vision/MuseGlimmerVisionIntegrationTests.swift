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
// Slow integration test for Muse Glimmer's vision path — the ViT-G/14
// perception encoder + vision-adapter/vision-projection MLP
// (`Models/Vision/MuseGlimmerVision.swift`), composed with the existing
// `MuseGlimmerDense` text engine via `VisionModel`'s cross-modal splice.
//
// Uses the local 4-bit checkpoint at `~/models/muse-glimmer-30b-4bit`
// (18 GB on disk) — loaded via `Model.load(idOrPath:)`'s local-path
// support, no network / HF hub download.
//
// SINGLE LOAD, MULTIPLE ASSERTIONS: this checkpoint is large (18 GB)
// relative to the other checkpoints in this test bundle. Rather than
// the usual "one `@Test` = one `Model.load`" house pattern (fine for
// the small 1-4 GB checkpoints elsewhere in this file's siblings), this
// suite loads ONCE in a single `@Test` and runs every assertion — VLM
// recognition, three image captions, and the text-only regression check
// — against that one loaded model. Fewer redundant 18 GB reads/loads on
// a box that has already hit a memory incident during this work.
//
// Text-only regression note: `MuseGlimmerVL.load` (the new vision-aware
// dispatch branch — see `Loader/Model.swift`) calls the EXACT SAME
// `MuseGlimmerDense.loadModel(config:weights:options:device:)` the
// pre-existing text-only branch calls, with identical arguments, and
// `MuseGlimmerModel.forward(tokenId:...)` / `forwardMulti(...)` (the
// methods a plain-text `m.generate(prompt:...)` call drives) are
// UNCHANGED by this vision port — only new methods
// (`forward(inputEmbedding:...)`, `textEmbedding(...)`) were added. The
// text-only code path is therefore provably byte-identical to the
// pre-vision dev baseline by construction; this test's regression
// check additionally verifies it EMPIRICALLY by running the same
// 200-token greedy prompt twice against the one loaded engine and
// asserting the two token streams match exactly (proves no shared
// mutable state introduced by the vision wiring leaks into the
// text-only path) and that the output is coherent.
//
// DO NOT RUN `swift test` on this target — individual ModelTests load
// multi-GB checkpoints and must be run via `make test-integration`
// (serialized, --num-workers 1), or with an explicit `--filter` naming
// only this suite.

import Foundation
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("Muse Glimmer Vision Integration", .serialized, .enabled(if: modelTestsOptedIn))
struct MuseGlimmerVisionIntegrationTests {

    static let modelPath = "~/models/muse-glimmer-30b-4bit"

    @Test("load + vision recognition + image captions + text-only regression")
    func fullVisionPathAndTextRegression() async throws {
        let m = try await loadModel(Self.modelPath)

        // ── 1. VLM recognition ──
        #expect(m.engine is MuseGlimmerModel, "expected engine to be MuseGlimmerModel")
        #expect(m.vlModel != nil, "checkpoint has vision_tower.* tensors — expected a VisionModel")
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.engine.supportsEmbeddingInput, "VLM splice prerequisite")
        let vlm = try #require(m.vlModel, "Muse Glimmer checkpoint did not load as a VLM")
        #expect(vlm.imageTokenCount > 0)
        #expect(vlm.imageTokenId == 200_092)
        print("Muse Glimmer VLM: imageTokenCount=\(vlm.imageTokenCount), imageTokenId=\(vlm.imageTokenId)")

        // ── 2. Perf note — vision encode time per image ──
        let probeImage = RGBImage.solid(width: 448, height: 448, r: 0.5, g: 0.5, b: 0.5)
        let encodeStart = CFAbsoluteTimeGetCurrent()
        _ = vlm.encodeImage(probeImage, device: Device.shared)
        let encodeMs = (CFAbsoluteTimeGetCurrent() - encodeStart) * 1000
        print("Muse Glimmer vision encode time: \(String(format: "%.1f", encodeMs)) ms/image")

        // ── 3. Image + text captions — 3 synthetic images ──
        // NOTE: `Tokenizer.encode(text:)` (the default-`addSpecialTokens:
        // true` overload) prepends `<bos>` on EVERY call per the
        // tokenizer's `TemplateProcessing` post-processor — calling it
        // once per prompt SEGMENT (as an earlier draft of this test
        // did) inserted a spurious extra `<bos>` mid-sequence right
        // after the image-token run, corrupting the prompt structure
        // and producing degenerate echo/hedge output. Only the FIRST
        // segment should add the special tokens (one leading `<bos>`);
        // every later segment must pass `addSpecialTokens: false`.
        func buildPrompt(question: String) throws -> [Int] {
            var tokens = m.tokenizer.encode(
                text: "<|start|>user<|message|><|image_start|>", addSpecialTokens: true)
            tokens += Array(repeating: vlm.imageTokenId, count: vlm.imageTokenCount)
            tokens +=
                m.tokenizer.encode(
                    text: "<|image_end|>\(question)<|eot|><|start|>assistant<|message|>",
                    addSpecialTokens: false)
            return tokens
        }

        func caption(_ image: RGBImage, question: String, label: String) throws -> String {
            let promptTokens = try buildPrompt(question: question)
            let generated = try vlm.generate(
                promptTokens: promptTokens, image: image,
                maxTokens: 60, eosTokenId: m.config.eosTokenId,
                eosTokenIds: m.config.eosTokenIds)
            expectCoherentOutput(generated, minTokens: 4, label: label)
            let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
            print("Muse Glimmer [\(label)] generated: \(text)")
            return text
        }

        // (a) Solid red square — content check: mentions red.
        try autoreleasepool {
            let red = RGBImage.solid(width: 448, height: 448, r: 1.0, g: 0.0, b: 0.0)
            let text = try caption(
                red, question: "What color is this image? Answer in one word.",
                label: "red square")
            #expect(
                text.lowercased().contains("red"),
                "expected the red-square caption to mention red — got: \(text)")
        }

        // (b) Solid blue square — content check: mentions blue.
        try autoreleasepool {
            let blue = RGBImage.solid(width: 448, height: 448, r: 0.0, g: 0.0, b: 1.0)
            let text = try caption(
                blue, question: "What color is this image? Answer in one word.",
                label: "blue square")
            #expect(
                text.lowercased().contains("blue"),
                "expected the blue-square caption to mention blue — got: \(text)")
        }

        // (c) Green circle on white background (PIL-generated fixture,
        // Tests/ModelIntegrationTests/Resources/muse_glimmer_green_circle.png)
        // — content check: mentions circle/round AND green.
        try autoreleasepool {
            let circleURL = resourceURL("muse_glimmer_green_circle.png")
            let circle = try RGBImage.load(contentsOf: circleURL)
            let text = try caption(
                circle, question: "Describe the shape and color you see.",
                label: "green circle")
            let lowered = text.lowercased()
            #expect(
                lowered.contains("green"),
                "expected the green-circle caption to mention green — got: \(text)")
            // "circ" (not "circle") so "circular" — the model's actual
            // phrasing in manual testing — matches too.
            #expect(
                lowered.contains("circ") || lowered.contains("round") || lowered.contains("ball")
                    || lowered.contains("dot") || lowered.contains("sphere"),
                "expected the green-circle caption to mention a round shape — got: \(text)")
        }

        // ── 4. Text-only regression — 200-tok greedy, run twice ──
        // No image anywhere in this path: `m.generate` drives
        // `MuseGlimmerModel.forward(tokenId:...)` / `forwardMulti(...)`
        // directly, the same methods the pre-vision dev baseline used.
        let textPrompt =
            "<|start|>user<|message|>Explain what a hash table is in two sentences.<|eot|>"
            + "<|start|>assistant<|message|>"
        let params = m.defaultGenerationParameters.with {
            $0.maxTokens = 200
            $0.temperature = 0
        }
        let run1 = try await m.generate(prompt: textPrompt, parameters: params)
        let run2 = try await m.generate(prompt: textPrompt, parameters: params)

        let determinismMsg =
            "text-only generation is non-deterministic across two runs of the same "
            + "loaded engine — the vision wiring may have introduced shared "
            + "mutable state. run1=\(run1.generatedTokens.prefix(20))... "
            + "run2=\(run2.generatedTokens.prefix(20))..."
        #expect(
            run1.generatedTokens == run2.generatedTokens,
            Comment(rawValue: determinismMsg))
        expectCoherentOutput(
            run1.generatedTokens, minTokens: 8, label: "Muse Glimmer text-only regression")
        print("Muse Glimmer text-only (no image) generated: \(run1.text)")
    }
}
