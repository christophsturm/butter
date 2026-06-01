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
// MediaUtils — the consolidated media I/O + preprocessing module
// (GPU/parallel-first). See planning/multi-modal-support.md. Over time this
// folds in / supersedes the scattered `ImagePreprocessing` +
// `AudioPreprocessing`; it is the single home for resize / normalize, the
// per-model resolution processors (smart_resize, tile-splitting, letterbox),
// AVFoundation decode (image/audio/video), and camera/mic capture.
//
// Every op here is GPU-kernel-backed (or parallelized) — never a scalar CPU
// loop (multi-modal-support.md §2.2 / §12).

import Foundation
import Metal

public enum MediaUtils {

    // ─── Resize + normalize (GPU) ────────────────────────────────────

    /// GPU bilinear resize + per-channel normalize → planar NCHW
    /// `[1, 3, targetH, targetW]`, via the fused `ffai_resize_normalize`
    /// kernel (`Ops.resizeNormalize`). Replaces the CPU triple-loop in
    /// `ImagePreprocessing.resize` + `preprocess`. `align_corners = false`
    /// half-pixel sampling (the HF / transformers convention).
    ///
    /// `RGBImage.pixels` are interleaved RGB in `[0, 1]`; `normalization`
    /// supplies the per-channel `(mean, std)` (use `ImageNormalization(
    /// mean: (0,0,0), std: (1,1,1))` for a rescale-only pass, e.g. Gemma 4,
    /// whose tower re-centres with `2·(v − 0.5)` itself).
    public static func resizeNormalize(
        _ image: RGBImage, targetW: Int, targetH: Int,
        normalization: ImageNormalization, dtype: DType = .f32,
        device: Device = .shared
    ) -> Tensor {
        precondition(
            targetW > 0 && targetH > 0,
            "MediaUtils.resizeNormalize: target dims must be positive")
        // Upload the interleaved source at the requested dtype (the kernel
        // is generic over T; `makeTensor` handles the f32→dtype conversion).
        let src = ImagePreprocessing.makeTensor(
            from: image.pixels,
            shape: [image.height, image.width, 3],
            dtype: dtype, device: device)
        let cmd = device.makeCommandBuffer()
        let out = Ops.resizeNormalize(
            src, srcW: image.width, srcH: image.height,
            targetW: targetW, targetH: targetH,
            mean: normalization.mean, std: normalization.std, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        // [3, targetH, targetW] → NCHW [1, 3, targetH, targetW].
        return out.reshaped(to: [1, 3, targetH, targetW])
    }

    // ─── smart_resize (native-resolution geometry) ───────────────────

    /// Pixel bounds + rounding factor for `smartResize`. The native-
    /// resolution image processors (Qwen2/2.5/3/3.5-VL, GLM-OCR) round the
    /// image to a grid whose side is a multiple of `factor = patchSize ·
    /// spatialMergeSize`, under a `[minPixels, maxPixels]` total-pixel
    /// budget. Expressing the bounds factor-relative (`minFactor·factor²`,
    /// `maxFactor·factor²`) reproduces the HF defaults across patch sizes:
    /// patch-14 / merge-2 → factor 28, min 56² (= 4·28²), max 28²·1280.
    public struct SmartResizeBounds: Sendable {
        /// Rounding multiple in pixels — `patchSize · spatialMergeSize`. A
        /// side that is a multiple of `factor` is divisible by both the
        /// patch size (so the grid is whole patches) and the merge size (so
        /// the 2×2 patch-merger pools cleanly).
        public let factor: Int
        /// Lower bound on total output pixels (`h·w`). Tiny images are
        /// up-scaled to at least this many pixels.
        public let minPixels: Int
        /// Upper bound on total output pixels (`h·w`). Large images are
        /// down-scaled to at most this many pixels — this is the vision
        /// token budget (`h·w / factor²` merged tokens at most).
        public let maxPixels: Int
        /// Max long/short side aspect ratio; beyond this `smartResize`
        /// throws (extreme strips can't be represented without huge token
        /// counts). HF uses 200.
        public let maxAspectRatio: Int

        public init(factor: Int, minPixels: Int, maxPixels: Int, maxAspectRatio: Int = 200) {
            self.factor = factor
            self.minPixels = minPixels
            self.maxPixels = maxPixels
            self.maxAspectRatio = maxAspectRatio
        }

        /// Standard Qwen-family bounds: `factor = patchSize · mergeSize`,
        /// `minPixels = 4·factor²` (HF `56²` at patch-14/merge-2),
        /// `maxPixels = maxTokens·factor²` (HF `28²·1280`). Reused by
        /// Qwen2/2.5/3/3.5-VL and GLM-OCR; each may override `maxTokens`.
        public static func qwen(
            patchSize: Int, mergeSize: Int,
            minTokenSides: Int = 2, maxTokens: Int = 1280
        ) -> SmartResizeBounds {
            let factor = patchSize * mergeSize
            return SmartResizeBounds(
                factor: factor,
                minPixels: (minTokenSides * minTokenSides) * factor * factor,
                maxPixels: maxTokens * factor * factor)
        }
    }

    /// Error thrown when an image's aspect ratio exceeds the processor's
    /// supported range (the audit's "throw on unsupported size" rule).
    public enum SmartResizeError: Error, CustomStringConvertible {
        case aspectRatioTooExtreme(width: Int, height: Int, max: Int)
        public var description: String {
            switch self {
            case .aspectRatioTooExtreme(let w, let h, let max):
                return "MediaUtils.smartResize: image \(w)×\(h) has aspect "
                    + "ratio \(Double(Swift.max(w, h)) / Double(Swift.min(w, h))) "
                    + "> \(max) — unsupported (would need an unbounded token count)"
            }
        }
    }

    /// HF `smart_resize`: the largest aspect-ratio-preserving `(h, w)` whose
    /// sides are multiples of `bounds.factor` and whose area lies in
    /// `[minPixels, maxPixels]`. This is the native-resolution geometry the
    /// Qwen-family / GLM-OCR image processors feed into the variable-grid
    /// vision tower (one merged token per `factor²` output pixels).
    ///
    /// Mirrors `transformers` `smart_resize`: round to nearest multiple of
    /// `factor`; if over `maxPixels`, scale down by `√(area/maxPixels)` and
    /// floor-round; if under `minPixels`, scale up by `√(minPixels/area)`
    /// and ceil-round. Pure geometry — `static`, no GPU — so it is unit-
    /// testable across aspect ratios. Throws on an extreme aspect ratio.
    public static func smartResize(
        height: Int, width: Int, bounds: SmartResizeBounds
    ) throws -> (h: Int, w: Int) {
        precondition(
            height > 0 && width > 0,
            "MediaUtils.smartResize: dims must be positive")
        let f = bounds.factor
        let long = Swift.max(height, width)
        let short = Swift.min(height, width)
        if Double(long) / Double(short) > Double(bounds.maxAspectRatio) {
            throw SmartResizeError.aspectRatioTooExtreme(
                width: width, height: height, max: bounds.maxAspectRatio)
        }
        // Round each side to the nearest multiple of `factor`, floored at
        // `factor` (a side can't be smaller than one merge block).
        func roundBy(_ x: Double) -> Int {
            Swift.max(f, Int((x / Double(f)).rounded()) * f)
        }
        func floorBy(_ x: Double) -> Int {
            Swift.max(f, Int((x / Double(f)).rounded(.down)) * f)
        }
        func ceilBy(_ x: Double) -> Int {
            Int((x / Double(f)).rounded(.up)) * f
        }
        var hBar = roundBy(Double(height))
        var wBar = roundBy(Double(width))
        let area = Double(height) * Double(width)
        if hBar * wBar > bounds.maxPixels {
            let beta = (area / Double(bounds.maxPixels)).squareRoot()
            hBar = floorBy(Double(height) / beta)
            wBar = floorBy(Double(width) / beta)
        } else if hBar * wBar < bounds.minPixels {
            let beta = (Double(bounds.minPixels) / area).squareRoot()
            hBar = ceilBy(Double(height) * beta)
            wBar = ceilBy(Double(width) * beta)
        }
        return (h: hBar, w: wBar)
    }
}
