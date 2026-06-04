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
// Video file decode + frame sampling — the production counterpart to
// `AudioPreprocessing`'s `loadWaveform(url:)`. Loads a video container the
// OS can decode (h.264 / HEVC / ProRes mp4 / mov / …) and samples up to
// `maxFrames` evenly-spaced frames as `[RGBImage]`, ready for the
// per-frame `MediaUtils` resize/normalize + the vision tower's
// temporal-patch folding (`VisionEncoder.encode(frames:)`).
//
// FFAI has no raw-frame video API surface: callers pass a video URL and
// the Media layer owns the decode → sample → RGBImage path, exactly as
// audio callers pass a URL to `loadWaveform`. AVFoundation is Apple-only,
// so the whole decode is `#if canImport(AVFoundation)`-gated and isolated
// here (the rest of the Media layer is platform-portable).

import CoreGraphics
import Foundation

#if canImport(AVFoundation)
    import AVFoundation
#endif

extension MediaUtils {
    /// Errors raised by the video decode path.
    public enum VideoError: Error, CustomStringConvertible {
        case decodeFailed(String)

        public var description: String {
            switch self {
            case .decodeFailed(let msg):
                return "MediaUtils video decode failed: \(msg)"
            }
        }
    }

    #if canImport(AVFoundation)
        /// Decode up to `maxFrames` evenly-spaced frames from a video file
        /// as `[RGBImage]` in display order. Any container AVFoundation can
        /// read (h.264 / HEVC / ProRes mp4 / mov / …) works without an extra
        /// framework dependency.
        ///
        /// Frame `0` is sampled at `t = 0` (the natural thumbnail frame) and
        /// the rest are spaced over the asset's duration. The result is
        /// clipped to the actual frame count if the clip is shorter than
        /// requested. Tight timestamp tolerance lands the samples on real
        /// frames (relevant for the multi-frame `encode(frames:)` path),
        /// not interpolated approximations.
        ///
        /// Throws `VideoError.decodeFailed` if the asset can't be opened,
        /// has an invalid duration, or yields no decodable frames.
        public static func decodeVideoFrames(
            url: URL, maxFrames: Int
        ) throws -> [RGBImage] {
            precondition(maxFrames > 0, "maxFrames must be positive")
            let asset = AVURLAsset(url: url)
            let durationCM = loadAssetDurationSync(asset)
            let durationSecs = CMTimeGetSeconds(durationCM)
            guard durationSecs.isFinite, durationSecs > 0 else {
                throw VideoError.decodeFailed(
                    "\(url.lastPathComponent) has invalid duration \(durationSecs)")
            }

            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.requestedTimeToleranceBefore = .zero
            generator.requestedTimeToleranceAfter = .zero

            let stamps: [CMTime] = (0 ..< maxFrames).map { i in
                // Evenly spaced — frame 0 at t=0, frame N-1 just before
                // duration. For maxFrames=1 this picks t=0.
                let t =
                    maxFrames == 1
                    ? 0.0
                    : durationSecs * Double(i) / Double(maxFrames)
                return CMTime(seconds: t, preferredTimescale: 600)
            }

            var frames: [RGBImage] = []
            frames.reserveCapacity(stamps.count)
            for stamp in stamps {
                // Skip frames AVFoundation can't decode; only throw below if
                // we end up with none at all.
                guard let cg = try? generator.copyCGImage(at: stamp, actualTime: nil)
                else { continue }
                frames.append(try rgbImage(from: cg))
            }
            guard !frames.isEmpty else {
                throw VideoError.decodeFailed(
                    "\(url.lastPathComponent) — no decodable frames")
            }
            return frames
        }

        /// Convert a CoreGraphics image to FFAI's `RGBImage` (row-major RGB
        /// `Float` in `[0, 1]`). Renders into a fresh 32-bit RGBA buffer so
        /// the source colorspace / alpha layout doesn't have to be guessed;
        /// the alpha channel is dropped.
        private static func rgbImage(from cg: CGImage) throws -> RGBImage {
            let w = cg.width
            let h = cg.height
            var raw = [UInt8](repeating: 0, count: w * h * 4)
            let bytesPerRow = w * 4
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
            guard
                let ctx = raw.withUnsafeMutableBufferPointer({ ptr in
                    CGContext(
                        data: ptr.baseAddress,
                        width: w, height: h, bitsPerComponent: 8,
                        bytesPerRow: bytesPerRow,
                        space: colorSpace, bitmapInfo: bitmapInfo)
                })
            else {
                throw VideoError.decodeFailed("CGContext init failed for \(w)x\(h)")
            }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

            var pixels = [Float](repeating: 0, count: w * h * 3)
            for y in 0 ..< h {
                for x in 0 ..< w {
                    let srcBase = (y * w + x) * 4
                    let dstBase = (y * w + x) * 3
                    pixels[dstBase + 0] = Float(raw[srcBase + 0]) / 255.0
                    pixels[dstBase + 1] = Float(raw[srcBase + 1]) / 255.0
                    pixels[dstBase + 2] = Float(raw[srcBase + 2]) / 255.0
                }
            }
            return RGBImage(width: w, height: h, pixels: pixels)
        }

        /// Synchronously read `asset.duration` via the modern async loader,
        /// bridged onto a `DispatchSemaphore` (the decode API is sync).
        private static func loadAssetDurationSync(_ asset: AVURLAsset) -> CMTime {
            let box = CMTimeBox()
            let sem = DispatchSemaphore(value: 0)
            Task {
                if let value = try? await asset.load(.duration) {
                    box.value = value
                }
                sem.signal()
            }
            sem.wait()
            return box.value
        }
    #endif
}

#if canImport(AVFoundation)
    /// `final class` reference holder so the duration can be passed out of
    /// the `Task` closure without Swift 6 flagging a mutable local crossing
    /// an isolation boundary. `@unchecked Sendable` is safe: the sole reader
    /// runs only after `sem.wait()`, a strict happens-before via the
    /// semaphore.
    private final class CMTimeBox: @unchecked Sendable {
        var value: CMTime = .zero
    }
#endif
