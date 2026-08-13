// Copyright 2026 Tom Turney (@TheTom)
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
// MuseGlimmerModel + DFlash support — the two extra forward variants
// the DFlash spec-decode verify step needs that `forwardMulti` doesn't
// provide:
//
//   * `forwardMulti` (the pre-existing prefill fast path) returns
//     logits for ONLY the tail row — every caller so far has only
//     needed the last position's prediction. DFlash verify needs
//     EVERY row's logits (to check each drafted token against the
//     target's own argmax) plus every row's `target_layer_ids` hidden
//     features (to seed the NEXT draft call's context — see
//     `DFlashDrafter.swift`'s header).
//   * `forwardOneWithFeatures` is the single-token sibling of
//     `forward(tokenId:position:...)`, used both by the fallback loop
//     below (when a verify block straddles the sliding-window fast-
//     path bound) and by `DFlashSpecDecode`'s exactness-replay path
//     (which must re-run the accepted prefix through the SAME
//     single-token kernels the baseline greedy loop uses — see that
//     file's header for why the replay can't reuse batched kernels).
//
// **Fast-path window bound — same caveat as `forwardMulti`.** Both
// `forwardManyLogitsAndFeatures` and the plain `forwardMulti` share
// the identical `position + n <= slidingWindow` eligibility check
// (`MuseGlimmerLayer.forwardMulti`'s doc comment has the full
// `KVEvictionState.reserveNextSlot` reasoning). Verify blocks here are
// small (`blockSize = 16` ≪ 2048) so the fast path always applies
// EARLY in a generation; once the running position passes
// `slidingWindow - blockSize` (~2032 tokens into the conversation)
// every verify call falls back to the per-token loop below — still
// numerically correct (each token takes the exact same code path as
// baseline decode), just no longer batched, so spec-decode's speed
// advantage tapers off past that point. Not fixed here — same
// pre-existing limitation `forwardMulti` documents; a real fix would
// need `MuseGlimmerLayer`'s batched `sdpaMulti` call to support a
// non-zero KV base ring-offset for the sliding layers, which is out
// of scope for this port.

import Foundation
import Metal

extension MuseGlimmerModel {

    /// Single-token forward that ALSO returns the `targetLayerIDs`
    /// hidden-state features (`[1, targetLayerIDs.count * hidden]`,
    /// concatenated in the order given) alongside the usual logits.
    /// Mirrors `forward(tokenId:position:...)` exactly (same private-
    /// command-buffer discipline, same cache append) plus the capture.
    public func forwardOneWithFeatures(
        tokenId: Int, position: Int, targetLayerIDs: [Int],
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> (logits: Tensor, features: Tensor) {
        let tap = InspectTap.fromEnvironment
        let targetSet = Set(targetLayerIDs)

        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        var workCmd = device.makeCommandBuffer()
        let h0 = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        var h = embedNorm(h0, on: workCmd)

        var captured: [Int: Tensor] = [:]
        for (i, layer) in layers.enumerated() {
            h = layer.forward(
                h, position: position,
                cache: caches[i] as! any KVCacheProtocol,
                on: workCmd, device: device)
            if targetSet.contains(i) {
                // `[1, hidden]` — matches `concatFeatures`'s `[rows, hidden]`
                // expectation (rows == 1 here) rather than a bare
                // `[hidden]` 1-D tensor, whose `slicedRows` would slice
                // into the hidden dimension itself.
                let snap = Tensor.empty(shape: [1, hidden], dtype: h.dtype, device: device)
                Ops.copy(h.reshaped(to: [1, hidden]), into: snap, on: workCmd)
                captured[i] = snap
            }
            if tap.shouldDump(layer: i) {
                workCmd.commit()
                workCmd.waitUntilCompleted()
                _ = tap.dumpLayerBoundary(
                    h, label: "layer_out", layer: i, cmd: device.makeCommandBuffer(), device: device)
                workCmd = device.makeCommandBuffer()
            }
        }

        let normed = finalNorm(h, on: workCmd)
        let rawLogits = lmHead(normed, on: workCmd)

        let logits: Tensor
        if finalLogitSoftcap > 0 {
            // Commits + waits `workCmd` internally.
            logits = MuseGlimmerOps.softcap(
                rawLogits, multiplier: outputMultiplier, cap: finalLogitSoftcap, on: workCmd)
        } else {
            workCmd.commit()
            workCmd.waitUntilCompleted()
            logits = rawLogits
        }

        let features = concatFeatures(captured, order: targetLayerIDs, rows: 1, device: device)
        return (logits, features)
    }

    /// Batched sibling of `forwardOneWithFeatures` — returns logits AND
    /// `targetLayerIDs` features for EVERY input row (not just the
    /// tail), needed by `DFlashSpecDecode`'s batched verify step. See
    /// this file's header for the fast-path window bound shared with
    /// `forwardMulti`.
    public func forwardManyLogitsAndFeatures(
        tokenIds: [Int], startingAt position: Int, targetLayerIDs: [Int],
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> (logits: Tensor, features: Tensor) {
        precondition(
            !tokenIds.isEmpty,
            "MuseGlimmerModel.forwardManyLogitsAndFeatures: tokenIds must be non-empty")
        let n = tokenIds.count
        let fastPathEligible = position + n <= slidingWindow
        guard fastPathEligible else {
            var logitsRows: [Tensor] = []
            var featureRows: [Tensor] = []
            logitsRows.reserveCapacity(n)
            featureRows.reserveCapacity(n)
            for (i, tok) in tokenIds.enumerated() {
                let fCmd = device.makeCommandBuffer()
                let (l, f) = forwardOneWithFeatures(
                    tokenId: tok, position: position + i, targetLayerIDs: targetLayerIDs,
                    caches: caches, on: fCmd, device: device)
                fCmd.commit()
                fCmd.waitUntilCompleted()
                logitsRows.append(l)
                featureRows.append(f)
            }
            let stackCmd = device.makeCommandBuffer()
            let logits = stackRows(logitsRows, device: device, on: stackCmd)
            let features = stackRows(featureRows, device: device, on: stackCmd)
            stackCmd.commit()
            stackCmd.waitUntilCompleted()
            return (logits, features)
        }

        let workCmd = device.makeCommandBuffer()
        let targetSet = Set(targetLayerIDs)

        let idsBuf = device.makeBuffer(length: n * 4)
        idsBuf.contents().withMemoryRebound(to: UInt32.self, capacity: n) { p in
            for (i, t) in tokenIds.enumerated() { p[i] = UInt32(t) }
        }
        let idsTensor = Tensor(buffer: idsBuf, offset: 0, shape: [n], dtype: .u32)
        let h0 = embedTokens(idsTensor, on: workCmd)
        var h = Ops.rmsNormRows(
            h0.reshaped(to: [n * hidden]),
            weight: embedNorm.weight, eps: embedNorm.eps,
            nRows: n, rowSize: hidden, on: workCmd
        ).reshaped(to: [n, hidden])

        var captured: [Int: Tensor] = [:]
        for (i, layer) in layers.enumerated() {
            h = layer.forwardMulti(
                h, startingAt: position,
                cache: caches[i] as! any KVCacheProtocol,
                on: workCmd, device: device)
            if targetSet.contains(i) {
                let snap = Tensor.empty(shape: [n, hidden], dtype: h.dtype, device: device)
                Ops.copy(h, into: snap, on: workCmd)
                captured[i] = snap
            }
        }

        // Final norm + LM head on EVERY row (the divergence from
        // `forwardMulti`, which slices to the tail before this point).
        let normed = Ops.rmsNormRows(
            h.reshaped(to: [n * hidden]),
            weight: finalNorm.weight, eps: finalNorm.eps,
            nRows: n, rowSize: hidden, on: workCmd
        ).reshaped(to: [n, hidden])
        let rawLogits = lmHead.callMany(normed, t: n, on: workCmd, device: device)

        let logits: Tensor
        if finalLogitSoftcap > 0 {
            logits = MuseGlimmerOps.softcap(
                rawLogits, multiplier: outputMultiplier, cap: finalLogitSoftcap, on: workCmd)
        } else {
            workCmd.commit()
            workCmd.waitUntilCompleted()
            logits = rawLogits
        }

        let features = concatFeatures(captured, order: targetLayerIDs, rows: n, device: device)
        return (logits, features)
    }

    /// Concatenate `rows`-row `[rows, hidden]` capture tensors
    /// (keyed by layer index) into one `[rows, order.count * hidden]`
    /// tensor, feature-major within each row in `order`'s order — the
    /// layout `DFlashDraftModel.forward`'s `fc` projection expects.
    /// One tiny blit per (row, feature) pair — `order.count` is 5 and
    /// `rows` is always small on every caller (≤ a prefill chunk, or a
    /// single verify block) so this is not perf-sensitive.
    private func concatFeatures(
        _ captured: [Int: Tensor], order: [Int], rows: Int, device: Device
    ) -> Tensor {
        precondition(
            order.allSatisfy { captured[$0] != nil },
            "MuseGlimmerModel.concatFeatures: missing capture for one or more targetLayerIDs "
                + "\(order) — captured keys \(captured.keys.sorted())")
        let dtype = captured[order[0]]!.dtype
        let featureCount = order.count
        let dest = Tensor.empty(shape: [rows, featureCount, hidden], dtype: dtype, device: device)
        let bytesPerRow = hidden * dtype.byteSize
        let cmd = device.makeCommandBuffer()
        for (k, layerIdx) in order.enumerated() {
            let src = captured[layerIdx]!
            for r in 0 ..< rows {
                let srcRow = src.slicedRows(start: r, count: 1)
                let dstOffset = dest.offset + (r * featureCount + k) * bytesPerRow
                let dst = Tensor(buffer: dest.buffer, offset: dstOffset, shape: [1, hidden], dtype: dtype)
                Ops.copy(srcRow, into: dst, on: cmd)
            }
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        return dest.reshaped(to: [rows, featureCount * hidden])
    }

    /// Stack `n` single-row `[1, width]` tensors (all same dtype/width)
    /// into one `[n, width]` tensor. Used only by the fallback (non-
    /// fast-path) loop above, where each row comes from a separate
    /// single-token forward call.
    private func stackRows(_ rows: [Tensor], device: Device, on cmd: MTLCommandBuffer) -> Tensor {
        precondition(!rows.isEmpty, "MuseGlimmerModel.stackRows: rows must be non-empty")
        let width = rows[0].elementCount
        let dtype = rows[0].dtype
        let dest = Tensor.empty(shape: [rows.count, width], dtype: dtype, device: device)
        for (i, row) in rows.enumerated() {
            Ops.copy(
                row.reshaped(to: [1, width]),
                into: dest.slicedRows(start: i, count: 1), on: cmd)
        }
        return dest
    }
}
