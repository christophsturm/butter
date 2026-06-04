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
// Vision-side test helpers — image / video fixtures + content-recognition
// assertions shared by every VLM integration suite.
//
// Every VLM family's "image + text" test feeds the same real photograph
// (`Tests/Resources/dog.jpeg`, a golden retriever) and asserts the decoded
// caption mentions a dog. Centralising the image load here keeps every
// family test in lock-step and gives a single precise failure message
// if the fixture goes missing.
//
// The video helpers below are thin conveniences over the PRODUCTION
// decoder `MediaUtils.decodeVideoFrames(url:maxFrames:)` (the AVFoundation
// decode/sample logic lives in the Media layer, not here). They resolve
// the checked-in `Tests/Resources/cat.mp4` fixture and decode it:
// `catVideoFirstFrame()` for single-frame VLM tests, `catVideoFrames(
// maxFrames:)` for multi-frame `encode(frames:)` integration tests.

import FFAI
import Foundation
import Testing

public enum VisionTestHelpers {

    // MARK: - Image fixtures (dog.jpeg)

    /// Load the shared golden-retriever test photo as an `RGBImage`.
    /// Fails the calling test (rather than skipping) if the fixture is
    /// missing or undecodable — the image is checked into the repo, so a
    /// miss is a real breakage.
    public static func dogImage() throws -> RGBImage {
        let url = resourceURL("dog.jpeg")
        do {
            return try RGBImage.load(contentsOf: url)
        } catch {
            let message = "VisionTestHelpers: failed to load dog.jpeg fixture: \(error)"
            Issue.record(Comment(rawValue: message))
            throw error
        }
    }

    /// Load the dog fixture, resize to `targetSize × targetSize`, and
    /// return a planar **CHW** float array — the layout PaliGemma /
    /// SigLIP-style towers consume directly. No mean/std normalisation
    /// is applied here (each caller picks its own); pixels are in `[0,1]`.
    ///
    /// `count = 3 * targetSize * targetSize`. CPU-only — the dog image
    /// is small, this helper is dwarfed by the rest of the vision
    /// encoder cost, so a simple bilinear resize is fine.
    public static func dogImageCHW(targetSize: Int) throws -> [Float] {
        let img = try dogImage()
        let resized = MediaUtils.resize(
            img, targetW: targetSize, targetH: targetSize)
        var planar = [Float](repeating: 0, count: 3 * targetSize * targetSize)
        let plane = targetSize * targetSize
        for y in 0 ..< targetSize {
            for x in 0 ..< targetSize {
                let srcBase = (y * targetSize + x) * 3
                let dstBase = y * targetSize + x
                planar[0 * plane + dstBase] = resized.pixels[srcBase]
                planar[1 * plane + dstBase] = resized.pixels[srcBase + 1]
                planar[2 * plane + dstBase] = resized.pixels[srcBase + 2]
            }
        }
        return planar
    }

    /// Same as `dogImageCHW(targetSize:)` but applies per-channel
    /// normalisation in the same pass. Use this when the family expects
    /// CLIP-mean (`[0.48145466, 0.4578275, 0.40821073]`) /
    /// CLIP-std (`[0.26862954, 0.26130258, 0.27577711]`) or its own
    /// `ImageNormalization` preset.
    public static func dogImageCHWNormalized(
        targetSize: Int, normalization: ImageNormalization
    ) throws -> [Float] {
        var pixels = try dogImageCHW(targetSize: targetSize)
        let plane = targetSize * targetSize
        let means = [normalization.mean.0, normalization.mean.1, normalization.mean.2]
        let stds = [normalization.std.0, normalization.std.1, normalization.std.2]
        for c in 0 ..< 3 {
            let m = means[c]
            let s = stds[c]
            for i in (c * plane) ..< ((c + 1) * plane) {
                pixels[i] = (pixels[i] - m) / s
            }
        }
        return pixels
    }

    // MARK: - Video fixtures (cat.mp4)

    /// Decode the first frame of `cat.mp4` as an `RGBImage`. Used by
    /// every VLM today (one frame fed through the existing single-image
    /// `encode(image:)` path); multi-frame callers use
    /// `catVideoFrames(maxFrames:)` instead.
    public static func catVideoFirstFrame() throws -> RGBImage {
        let frames = try catVideoFrames(maxFrames: 1)
        guard let first = frames.first else {
            throw MediaUtils.VideoError.decodeFailed(
                "cat.mp4 decoded to zero frames")
        }
        return first
    }

    /// Absolute URL to the checked-in `cat.mp4` fixture — for tests that
    /// drive `MediaUtils.decodeVideoFrames(url:)` directly.
    public static func catVideoURL() -> URL { resourceURL("cat.mp4") }

    /// Decode up to `maxFrames` evenly-spaced frames from the checked-in
    /// `cat.mp4` fixture. Thin convenience over the production decoder
    /// `MediaUtils.decodeVideoFrames(url:maxFrames:)` — the AVFoundation
    /// decode/sample logic lives in the Media layer, not here, so tests
    /// exercise the same path real callers use. Fails the calling test if
    /// the asset can't be decoded (the file is checked in, so a miss is a
    /// real breakage).
    public static func catVideoFrames(maxFrames: Int) throws -> [RGBImage] {
        try MediaUtils.decodeVideoFrames(
            url: resourceURL("cat.mp4"), maxFrames: maxFrames)
    }

    // MARK: - Content-recognition assertions

    /// Assert a decoded VLM caption actually describes the dog photo.
    /// Lowercased substring match on "dog" — wide enough to accept
    /// "dog", "Golden Retriever" (no) but "puppy"/"canine" callers add
    /// their own; the single-word "dog" anchor is the agreed contract.
    public static func expectMentionsDog(
        _ text: String, label: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let lowered = text.lowercased()
        let comment = Comment(
            rawValue: "\(label): caption should mention a dog — got: \(text)")
        #expect(
            lowered.contains("dog"),
            comment,
            sourceLocation: sourceLocation)
    }

    /// Assert a decoded VLM caption describes the cat video frame.
    /// Accepts either "cat" or "kitten" — model verbosity varies.
    public static func expectMentionsCat(
        _ text: String, label: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let lowered = text.lowercased()
        let comment = Comment(
            rawValue: "\(label): caption should mention a cat — got: \(text)")
        #expect(
            lowered.contains("cat") || lowered.contains("kitten"),
            comment,
            sourceLocation: sourceLocation)
    }

    // MARK: - OCR fixture (testocr.png)

    /// Load `Resources/testocr.png` as an `RGBImage`. The fixture is a
    /// 640×480 black-on-white scan that prints the canonical OCR test
    /// passage ("This is a lot of 12 point text … The quick brown dog
    /// jumped over the lazy fox."). Used by OCR-capable VLM tests
    /// (GLM-OCR, PaliGemma `ocr en` task, …) to assert the model
    /// actually recovers printed text, not just produces a plausible
    /// English caption.
    public static func ocrTestImage() throws -> RGBImage {
        let url = resourceURL("testocr.png")
        do {
            return try RGBImage.load(contentsOf: url)
        } catch {
            let message = "VisionTestHelpers: failed to load testocr.png fixture: \(error)"
            Issue.record(Comment(rawValue: message))
            throw error
        }
    }

    /// `ocrTestImage()` resized to `targetSize × targetSize` and packed
    /// CHW (3·H·W floats in [0,1]). No normalisation; each caller picks
    /// its own (CLIP for GLM-OCR, SigLIP for PaliGemma, …).
    public static func ocrTestImageCHW(targetSize: Int) throws -> [Float] {
        let img = try ocrTestImage()
        let resized = MediaUtils.resize(
            img, targetW: targetSize, targetH: targetSize)
        var planar = [Float](repeating: 0, count: 3 * targetSize * targetSize)
        let plane = targetSize * targetSize
        for y in 0 ..< targetSize {
            for x in 0 ..< targetSize {
                let srcBase = (y * targetSize + x) * 3
                let dstBase = y * targetSize + x
                planar[0 * plane + dstBase] = resized.pixels[srcBase]
                planar[1 * plane + dstBase] = resized.pixels[srcBase + 1]
                planar[2 * plane + dstBase] = resized.pixels[srcBase + 2]
            }
        }
        return planar
    }

    /// `ocrTestImageCHW(targetSize:)` plus per-channel normalisation
    /// applied in the same pass. Mirrors `dogImageCHWNormalized` but
    /// for the OCR fixture.
    public static func ocrTestImageCHWNormalized(
        targetSize: Int, normalization: ImageNormalization
    ) throws -> [Float] {
        var pixels = try ocrTestImageCHW(targetSize: targetSize)
        let plane = targetSize * targetSize
        let means = [normalization.mean.0, normalization.mean.1, normalization.mean.2]
        let stds = [normalization.std.0, normalization.std.1, normalization.std.2]
        for c in 0 ..< 3 {
            let m = means[c]
            let s = stds[c]
            for i in (c * plane) ..< ((c + 1) * plane) {
                pixels[i] = (pixels[i] - m) / s
            }
        }
        return pixels
    }

    /// `ocrTestImage()` resized to `targetSize × targetSize`, normalised
    /// with the supplied mean/std, and returned in interleaved HWC float
    /// layout (`[H, W, 3]`). This is what GLM-OCR's `GlmOcrRGBImage`
    /// expects — the model's `setImagePixels`-style entrypoints accept
    /// HWC, not CHW.
    public static func ocrTestImageHWCNormalized(
        targetSize: Int, normalization: ImageNormalization
    ) throws -> [Float] {
        let img = try ocrTestImage()
        let resized = MediaUtils.resize(
            img, targetW: targetSize, targetH: targetSize)
        let means = [normalization.mean.0, normalization.mean.1, normalization.mean.2]
        let stds = [normalization.std.0, normalization.std.1, normalization.std.2]
        var out = [Float](repeating: 0, count: resized.pixels.count)
        var i = 0
        while i < resized.pixels.count {
            out[i + 0] = (resized.pixels[i + 0] - means[0]) / stds[0]
            out[i + 1] = (resized.pixels[i + 1] - means[1]) / stds[1]
            out[i + 2] = (resized.pixels[i + 2] - means[2]) / stds[2]
            i += 3
        }
        return out
    }

    /// Phrases present in the canonical `testocr.png` passage. Used by
    /// `expectRecognizesOCRText` — a generated string need only contain
    /// *some* of these to count as having successfully read the image
    /// (no OCR model recovers every word at every quantisation).
    public static let ocrCandidatePhrases: [String] = [
        // High-signal words: the OCR test image's distinctive content.
        "quick brown",
        "lazy fox",
        "lazy dog",  // The classic typo-tolerant version.
        "12 point",
        "12-point",
        "ocr",
        "jumped over",
        "file format",
        "all types",
        "see if it works",
    ]

    /// Assert a decoded VLM transcription recovers enough of the
    /// `testocr.png` passage to count as having genuinely read the image.
    /// Requires at least `minMatches` candidate phrases (default 2) from
    /// `ocrCandidatePhrases` to appear in the lowercased output.
    ///
    /// This is intentionally lenient: any individual OCR model may
    /// transcribe imperfectly (skip lines, drop punctuation, hallucinate
    /// the trailing "lazy fox" as "lazy dog"). Requiring two phrases
    /// catches the "model wrote real text" signal without locking us
    /// into one verbatim output.
    public static func expectRecognizesOCRText(
        _ text: String, label: String, minMatches: Int = 2,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let lowered = text.lowercased()
        let matched = ocrCandidatePhrases.filter { lowered.contains($0) }
        let comment = Comment(
            rawValue:
                "\(label): expected ≥\(minMatches) OCR phrases recovered "
                + "from testocr.png; matched \(matched.count) (\(matched)) — got: \(text)"
        )
        #expect(
            matched.count >= minMatches,
            comment,
            sourceLocation: sourceLocation)
    }
}

