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
}
