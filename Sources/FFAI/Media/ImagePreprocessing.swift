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
// ImagePreprocessing — CPU-side image resize / normalize / patchify for
// vision-language models.
//
// Every VLM vision encoder consumes a fixed-layout pixel tensor: an
// image resized to the model's expected resolution, channel-normalized
// with the checkpoint's per-channel mean/std, and laid out NCHW. This
// file does that on the CPU — the cost is negligible next to the
// transformer forward, and keeping it CPU-side avoids a Metal round-trip
// for what is fundamentally a one-shot setup step (spec:
// "CPU initially; Metal later if it shows up in profiles").
//
// The output is a `Tensor` in the model's activation dtype, ready to
// hand straight to `Ops.conv2d` / `Ops.patchEmbed`.

import CoreGraphics
import Foundation
import ImageIO

/// Error decoding an image file into an `RGBImage`.
public enum RGBImageError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case decodeFailed(String)

    public var description: String {
        switch self {
        case .fileNotFound(let path):
            return "RGBImage: no image file at \(path)"
        case .decodeFailed(let path):
            return "RGBImage: could not decode image at \(path)"
        }
    }
}

/// A decoded RGB image in planar CPU memory: `pixels` is row-major
/// `[height, width, 3]` with values already in `[0, 1]`.
public struct RGBImage: Sendable {
    public let width: Int
    public let height: Int
    /// Interleaved RGB, `height * width * 3` floats in `[0, 1]`.
    public let pixels: [Float]

    public init(width: Int, height: Int, pixels: [Float]) {
        precondition(
            pixels.count == width * height * 3,
            "RGBImage: pixel count \(pixels.count) != w*h*3 \(width * height * 3)")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// A solid-color test image — handy for unit tests and as a
    /// deterministic stand-in when no real image is supplied.
    public static func solid(
        width: Int, height: Int,
        r: Float, g: Float, b: Float
    ) -> RGBImage {
        var px = [Float](repeating: 0, count: width * height * 3)
        for i in 0 ..< (width * height) {
            px[i * 3] = r
            px[i * 3 + 1] = g
            px[i * 3 + 2] = b
        }
        return RGBImage(width: width, height: height, pixels: px)
    }

    /// Decode an image file (JPEG / PNG / any ImageIO-supported format)
    /// from disk into an interleaved-RGB `[0,1]` `RGBImage`.
    ///
    /// ImageIO is the idiomatic Apple-platform decoder — it handles the
    /// container parsing, colour-space conversion, and orientation. The
    /// decoded pixels are drawn into a tightly-packed RGBA8 bitmap in
    /// sRGB, then the alpha channel is dropped and the 8-bit samples are
    /// scaled to `[0,1]`.
    public static func load(contentsOf url: URL) throws -> RGBImage {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw RGBImageError.fileNotFound(url.path)
        }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw RGBImageError.decodeFailed(url.path)
        }

        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4  // RGBA8
        let bytesPerRow = width * bytesPerPixel
        var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)

        // Draw the decoded image into a known-layout RGBA8 sRGB bitmap.
        // CGBitmapContext handles whatever the source colour space /
        // bit-depth was; we read back a uniform 8-bit RGBA buffer.
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw RGBImageError.decodeFailed(url.path)
        }
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        let ok: Bool = rgba.withUnsafeMutableBytes { raw -> Bool in
            guard
                let ctx = CGContext(
                    data: raw.baseAddress, width: width, height: height,
                    bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                    space: colorSpace, bitmapInfo: bitmapInfo)
            else { return false }
            ctx.draw(
                cgImage,
                in: CGRect(
                    x: 0, y: 0,
                    width: width, height: height))
            return true
        }
        guard ok else { throw RGBImageError.decodeFailed(url.path) }

        // RGBA8 → interleaved-RGB float [0,1], dropping alpha.
        var pixels = [Float](repeating: 0, count: width * height * 3)
        let inv255: Float = 1.0 / 255.0
        for i in 0 ..< (width * height) {
            pixels[i * 3] = Float(rgba[i * 4]) * inv255
            pixels[i * 3 + 1] = Float(rgba[i * 4 + 1]) * inv255
            pixels[i * 3 + 2] = Float(rgba[i * 4 + 2]) * inv255
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }
}

/// Normalization constants — per-channel mean / std. The defaults are
/// the CLIP / SigLIP values most VLM checkpoints ship.
public struct ImageNormalization: Sendable {
    public let mean: (Float, Float, Float)
    public let std: (Float, Float, Float)

    public init(mean: (Float, Float, Float), std: (Float, Float, Float)) {
        self.mean = mean
        self.std = std
    }

    /// SigLIP / Gemma-VL normalization — mean 0.5, std 0.5 per channel
    /// (maps `[0,1]` pixels to `[-1,1]`).
    public static let siglip = ImageNormalization(
        mean: (0.5, 0.5, 0.5), std: (0.5, 0.5, 0.5))

    /// CLIP / Qwen-VL normalization — ImageNet-derived per-channel
    /// statistics OpenAI's CLIP preprocessing established.
    public static let clip = ImageNormalization(
        mean: (0.48145466, 0.4578275, 0.40821073),
        std: (0.26862954, 0.26130258, 0.27577711))
}

// Image preprocessing lives on `MediaUtils` — the single facade models call.
// The data types above (`RGBImage`, `ImageNormalization`) stay top-level.
extension MediaUtils {

    /// Resize `image` to `targetW × targetH` with bilinear interpolation.
    /// Vision encoders expect a fixed input resolution (SigLIP 224 / 896,
    /// CLIP 336, …); this is the resampling step.
    public static func resize(
        _ image: RGBImage,
        targetW: Int, targetH: Int
    ) -> RGBImage {
        precondition(
            targetW > 0 && targetH > 0,
            "MediaUtils.resize: target dims must be positive")
        if image.width == targetW && image.height == targetH { return image }

        var out = [Float](repeating: 0, count: targetW * targetH * 3)
        // Map each output pixel center back into the input grid. The
        // half-pixel offset keeps the sampling centered, matching the
        // `align_corners = false` convention HF image processors use.
        let scaleX = Float(image.width) / Float(targetW)
        let scaleY = Float(image.height) / Float(targetH)
        for oy in 0 ..< targetH {
            let srcY = (Float(oy) + 0.5) * scaleY - 0.5
            let y0 = max(0, min(image.height - 1, Int(srcY.rounded(.down))))
            let y1 = min(image.height - 1, y0 + 1)
            let wy = max(0, min(1, srcY - Float(y0)))
            for ox in 0 ..< targetW {
                let srcX = (Float(ox) + 0.5) * scaleX - 0.5
                let x0 = max(0, min(image.width - 1, Int(srcX.rounded(.down))))
                let x1 = min(image.width - 1, x0 + 1)
                let wx = max(0, min(1, srcX - Float(x0)))
                for c in 0 ..< 3 {
                    let p00 = image.pixels[(y0 * image.width + x0) * 3 + c]
                    let p01 = image.pixels[(y0 * image.width + x1) * 3 + c]
                    let p10 = image.pixels[(y1 * image.width + x0) * 3 + c]
                    let p11 = image.pixels[(y1 * image.width + x1) * 3 + c]
                    let top = p00 * (1 - wx) + p01 * wx
                    let bot = p10 * (1 - wx) + p11 * wx
                    out[(oy * targetW + ox) * 3 + c] = top * (1 - wy) + bot * wy
                }
            }
        }
        return RGBImage(width: targetW, height: targetH, pixels: out)
    }

    /// Resize + per-channel normalize, producing an NCHW float tensor
    /// `[1, 3, targetH, targetW]` in `dtype` — the canonical layout
    /// `Ops.conv2d` consumes (and, reshaped to `[3, h, w]`, what
    /// `Ops.patchEmbed` consumes).
    ///
    /// `pixel = (pixel - mean[c]) / std[c]`, applied per channel after
    /// the resize.
    public static func preprocess(
        _ image: RGBImage,
        targetW: Int, targetH: Int,
        normalization: ImageNormalization,
        dtype: DType,
        device: Device = .shared
    ) -> Tensor {
        let resized = resize(image, targetW: targetW, targetH: targetH)
        let means = [normalization.mean.0, normalization.mean.1, normalization.mean.2]
        let stds = [normalization.std.0, normalization.std.1, normalization.std.2]

        // Interleaved RGB → planar NCHW, normalizing in the same pass.
        var planar = [Float](repeating: 0, count: 3 * targetH * targetW)
        let plane = targetH * targetW
        for y in 0 ..< targetH {
            for x in 0 ..< targetW {
                let srcBase = (y * targetW + x) * 3
                for c in 0 ..< 3 {
                    let v = (resized.pixels[srcBase + c] - means[c]) / stds[c]
                    planar[c * plane + y * targetW + x] = v
                }
            }
        }
        return makeTensor(
            from: planar, shape: [1, 3, targetH, targetW],
            dtype: dtype, device: device)
    }

    /// Patchify a normalized planar `[3, h, w]` float array into a flat
    /// `[num_patches, patch_dim]` tensor — each row is one
    /// `3 × patchH × patchW` patch flattened in `(c, py, px)` order,
    /// matching the `patch_embed` weight-column convention.
    ///
    /// This is the explicit-unfold path; the fused `Ops.patchEmbed`
    /// kernel does the same gather internally, so most callers go
    /// straight through `Ops.patchEmbed` and never need this. It exists
    /// for encoders that materialize patches (and for testing).
    public static func patchify(
        planar: [Float], channels: Int, height: Int, width: Int,
        patchH: Int, patchW: Int, dtype: DType, device: Device = .shared
    ) -> Tensor {
        precondition(
            planar.count == channels * height * width,
            "MediaUtils.patchify: array size mismatch")
        precondition(
            height % patchH == 0 && width % patchW == 0,
            "MediaUtils.patchify: image not divisible by patch")
        let patchesH = height / patchH
        let patchesW = width / patchW
        let numPatches = patchesH * patchesW
        let patchDim = channels * patchH * patchW
        let plane = height * width

        var out = [Float](repeating: 0, count: numPatches * patchDim)
        for ph in 0 ..< patchesH {
            for pw in 0 ..< patchesW {
                let patch = ph * patchesW + pw
                var col = 0
                for c in 0 ..< channels {
                    for py in 0 ..< patchH {
                        let row = (ph * patchH + py) * width + pw * patchW
                        for px in 0 ..< patchW {
                            out[patch * patchDim + col] = planar[c * plane + row + px]
                            col += 1
                        }
                    }
                }
            }
        }
        return makeTensor(
            from: out, shape: [numPatches, patchDim],
            dtype: dtype, device: device)
    }

    /// Build a `Tensor` of `shape` / `dtype` from a `[Float]` source,
    /// converting to the storage dtype. Centralizes the f32 / f16 / bf16
    /// host-side conversion every preprocessing path shares.
    static func makeTensor(
        from values: [Float], shape: [Int],
        dtype: DType, device: Device
    ) -> Tensor {
        let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
        switch dtype {
        case .f32:
            t.copyIn(from: values)
        case .f16:
            t.copyIn(from: values.map { Float16($0) })
        case .bf16:
            t.copyIn(
                from: values.map { v -> UInt16 in
                    let bits = v.bitPattern
                    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                    return UInt16(rounded >> 16)
                })
        default:
            fatalError("ImagePreprocessing: unsupported dtype \(dtype)")
        }
        return t
    }

    /// CPU bicubic resize of an `[srcH, srcW, channels]` row-major grid to
    /// `[dstH, dstW, channels]`. Catmull-Rom cubic kernel (`a = -0.75`) with
    /// half-pixel sample centres (PyTorch `align_corners=false`), edge-clamped
    /// taps. Used to resize a learned position-embedding grid to a target
    /// patch grid for native-resolution towers (SigLIP2 NaFlex
    /// `resize_positional_embeddings`). Cold path — one resize per image.
    static func bicubicResize(
        _ src: [Float], srcH: Int, srcW: Int, channels: Int,
        dstH: Int, dstW: Int
    ) -> [Float] {
        precondition(
            src.count == srcH * srcW * channels,
            "bicubicResize: src count \(src.count) != \(srcH * srcW * channels)")
        func cubic(_ t: Float) -> Float {
            let a: Float = -0.75
            let x = abs(t)
            if x <= 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
            if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
            return 0
        }
        func clampIdx(_ i: Int, _ n: Int) -> Int { Swift.max(0, Swift.min(n - 1, i)) }
        var dst = [Float](repeating: 0, count: dstH * dstW * channels)
        let scaleY = Float(srcH) / Float(dstH)
        let scaleX = Float(srcW) / Float(dstW)
        for oy in 0 ..< dstH {
            let sy = (Float(oy) + 0.5) * scaleY - 0.5
            let y0 = Int(sy.rounded(.down))
            let wy = (0 ..< 4).map { cubic(sy - Float(y0 - 1 + $0)) }
            for ox in 0 ..< dstW {
                let sx = (Float(ox) + 0.5) * scaleX - 0.5
                let x0 = Int(sx.rounded(.down))
                let wx = (0 ..< 4).map { cubic(sx - Float(x0 - 1 + $0)) }
                for c in 0 ..< channels {
                    var acc: Float = 0
                    for j in 0 ..< 4 {
                        let yy = clampIdx(y0 - 1 + j, srcH)
                        for i in 0 ..< 4 {
                            let xx = clampIdx(x0 - 1 + i, srcW)
                            acc += wy[j] * wx[i] * src[(yy * srcW + xx) * channels + c]
                        }
                    }
                    dst[(oy * dstW + ox) * channels + c] = acc
                }
            }
        }
        return dst
    }
}
