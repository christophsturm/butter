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
    ///
    /// OWNERSHIP: never encodes onto `cmd` — allocates and commits its
    /// own internal buffer(s), returning only once both results are
    /// host-resident. Callers must NOT `cmd.commit()` /
    /// `cmd.waitUntilCompleted()` the buffer they passed in (a
    /// diagnosed cause of a wedged Metal queue in `DFlashSpecDecode`'s
    /// prefill/verify/replay loop — see that file's header).
    public func forwardOneWithFeatures(
        tokenId: Int, position: Int, targetLayerIDs: [Int],
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> (logits: Tensor, features: Tensor) {
        precondition(
            cmd.status == .notEnqueued,
            "MuseGlimmerModel.forwardOneWithFeatures: cmd must be a fresh, uncommitted buffer — "
                + "this function never encodes onto it and commits its own internal buffer instead")
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

    /// Chained sibling of `forwardOneWithFeatures` for REPLAYING a
    /// sequence of single-token steps — same per-token single-token
    /// kernel path (`layer.forward`, NOT the batched `forwardMulti`
    /// path — see `DFlashSpecDecode.swift`'s EXACTNESS DISCIPLINE
    /// section for why replay specifically needs the single-token
    /// kernel), but all `tokens.count` steps' work is encoded onto ONE
    /// command buffer with ONE commit+wait at the end, instead of
    /// `tokens.count` separate self-committing calls.
    ///
    /// Correctness: each step's KV cache append only depends on
    /// PRECEDING steps' appends already being encoded earlier on the
    /// SAME command buffer — Metal executes one buffer's encoded
    /// commands in encode order, so this needs no inter-step host
    /// synchronization (unlike, say, the bail loop, where each step's
    /// INPUT token depends on the PREVIOUS step's own OUTPUT argmax —
    /// a genuine data dependency that can't be chained this way).
    /// `tokens` here must already be the exact resolved (tokenId,
    /// position) sequence to replay.
    ///
    /// Returns the LAST step's logits (softcap applied, matching
    /// `forwardOneWithFeatures`) and one features row per step, in
    /// input order.
    ///
    /// OWNERSHIP: allocates and commits its own internal buffer,
    /// exactly once for the whole chain — never touches the caller's
    /// `cmd`. Motivation: collapsing N self-committing round trips
    /// into 1 cuts this driver's total command-buffer submission count
    /// dramatically once every DFlash cycle replays unconditionally
    /// (see `DFlashSpecDecode.swift`'s EXACTNESS DISCIPLINE section) —
    /// a diagnosed contributor to a sustained-load Metal queue wedge
    /// (`IOGPUCommandQueueSubmitCommandBuffers` never returning) on
    /// long gate runs. Reducing round-trip COUNT is the lever; per-call
    /// `autoreleasepool` wrapping (also applied at every call site)
    /// only bounds ARC/ObjC object lifetime, not GPU-driver-side
    /// command-buffer tracking state, so it alone did not resolve the
    /// wedge — this does the actual work of cutting submission volume.
    public func forwardManySingleStepChained(
        tokens: [(tokenId: Int, position: Int)], targetLayerIDs: [Int],
        caches: [any LayerCacheProtocol],
        device: Device
    ) -> (lastLogits: Tensor, featureRows: [Tensor]) {
        precondition(
            !tokens.isEmpty,
            "MuseGlimmerModel.forwardManySingleStepChained: tokens must be non-empty")
        let tap = InspectTap.fromEnvironment
        let featureCount = targetLayerIDs.count
        // layer index -> its position within targetLayerIDs (matching
        // `concatFeatures`' `order.enumerated()` — needed since a
        // layer's OWN index and its rank among the target layers can
        // differ, e.g. targetLayerIDs = [1, 13, 25, 37, 49]).
        var layerToFeatureIndex: [Int: Int] = [:]
        for (k, layerIdx) in targetLayerIDs.enumerated() { layerToFeatureIndex[layerIdx] = k }

        // Pre-allocate ONE destination for every step's feature capture
        // — `[tokens.count, featureCount, hidden]`, written directly
        // (row = step, no per-step scratch tensor / no per-step
        // concatFeatures call) so the whole chain, including feature
        // bookkeeping, stays on the SAME command buffer.
        var featuresDestBuf: Tensor?

        var workCmd = device.makeCommandBuffer()
        var lastRawLogits: Tensor!

        for (stepIdx, step) in tokens.enumerated() {
            let tokenBuf = device.makeBuffer(length: 4)
            var tid = UInt32(step.tokenId)
            memcpy(tokenBuf.contents(), &tid, 4)
            let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

            let h0 = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
            var h = embedNorm(h0, on: workCmd)

            if featuresDestBuf == nil {
                featuresDestBuf = Tensor.empty(
                    shape: [tokens.count, featureCount, hidden], dtype: h.dtype, device: device)
            }
            let featuresDest = featuresDestBuf!
            let bytesPerRow = hidden * h.dtype.byteSize

            for (i, layer) in layers.enumerated() {
                h = layer.forward(
                    h, position: step.position,
                    cache: caches[i] as! any KVCacheProtocol,
                    on: workCmd, device: device)
                if let k = layerToFeatureIndex[i] {
                    let dstOffset =
                        featuresDest.offset
                        + (stepIdx * featureCount + k) * bytesPerRow
                    let dst = Tensor(
                        buffer: featuresDest.buffer, offset: dstOffset, shape: [1, hidden],
                        dtype: h.dtype)
                    Ops.copy(h.reshaped(to: [1, hidden]), into: dst, on: workCmd)
                }
                if tap.shouldDump(layer: i) {
                    workCmd.commit()
                    workCmd.waitUntilCompleted()
                    _ = tap.dumpLayerBoundary(
                        h, label: "layer_out", layer: i, cmd: device.makeCommandBuffer(),
                        device: device)
                    workCmd = device.makeCommandBuffer()
                }
            }

            let normed = finalNorm(h, on: workCmd)
            let rawLogits = lmHead(normed, on: workCmd)
            if stepIdx == tokens.count - 1 {
                lastRawLogits = rawLogits
            }
        }

        // `MuseGlimmerOps.softcap` ALWAYS commits+waits whatever `cmd`
        // it's given (see its doc comment) — committing `workCmd`
        // OURSELVES first and then handing it to `softcap` would be a
        // double-commit (`-[IOGPUMetalCommandBuffer validate]: commit
        // an already committed command buffer`, a hard Metal validation
        // crash, not a silent no-op — confirmed the hard way). Exactly
        // one of these two branches may commit `workCmd`.
        let lastLogits: Tensor
        if finalLogitSoftcap > 0 {
            lastLogits = MuseGlimmerOps.softcap(
                lastRawLogits, multiplier: outputMultiplier, cap: finalLogitSoftcap, on: workCmd)
        } else {
            workCmd.commit()
            workCmd.waitUntilCompleted()
            lastLogits = lastRawLogits
        }

        let featuresFlat = featuresDestBuf!.reshaped(to: [tokens.count, featureCount * hidden])
        var featureRows: [Tensor] = []
        featureRows.reserveCapacity(tokens.count)
        for r in 0 ..< tokens.count { featureRows.append(featuresFlat.slicedRows(start: r, count: 1)) }
        return (lastLogits, featureRows)
    }

    /// Batched sibling of `forwardOneWithFeatures` — returns logits AND
    /// `targetLayerIDs` features for EVERY input row (not just the
    /// tail), needed by `DFlashSpecDecode`'s batched verify step. See
    /// this file's header for the fast-path window bound shared with
    /// `forwardMulti`.
    ///
    /// OWNERSHIP: same contract as `forwardOneWithFeatures` above —
    /// never encodes onto `cmd`, commits its own internal buffer(s)
    /// instead. Callers must NOT commit/wait `cmd` (diagnosed
    /// queue-wedge risk — see that function's doc comment).
    public func forwardManyLogitsAndFeatures(
        tokenIds: [Int], startingAt position: Int, targetLayerIDs: [Int],
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> (logits: Tensor, features: Tensor) {
        precondition(
            !tokenIds.isEmpty,
            "MuseGlimmerModel.forwardManyLogitsAndFeatures: tokenIds must be non-empty")
        precondition(
            cmd.status == .notEnqueued,
            "MuseGlimmerModel.forwardManyLogitsAndFeatures: cmd must be a fresh, uncommitted "
                + "buffer — this function never encodes onto it and commits its own internal "
                + "buffer instead")
        let n = tokenIds.count
        let fastPathEligible = position + n <= slidingWindow
        guard fastPathEligible else {
            var logitsRows: [Tensor] = []
            var featureRows: [Tensor] = []
            logitsRows.reserveCapacity(n)
            featureRows.reserveCapacity(n)
            for (i, tok) in tokenIds.enumerated() {
                // `forwardOneWithFeatures` self-commits+waits its own
                // private buffer and never touches `fCmd` — do NOT
                // commit/wait `fCmd` here (see that function's doc
                // comment above; this was the same double-commit
                // pattern diagnosed in `DFlashSpecDecode`'s call sites).
                //
                // `autoreleasepool` per token — same cooperative-
                // executor rationale as `forwardMulti`'s sibling
                // fallback loop.
                let (l, f) = autoreleasepool { () -> (Tensor, Tensor) in
                    let fCmd = device.makeCommandBuffer()
                    return forwardOneWithFeatures(
                        tokenId: tok, position: position + i, targetLayerIDs: targetLayerIDs,
                        caches: caches, on: fCmd, device: device)
                }
                logitsRows.append(l)
                featureRows.append(f)
            }
            return autoreleasepool { () -> (Tensor, Tensor) in
                let stackCmd = device.makeCommandBuffer()
                let logits = stackRows(logitsRows, device: device, on: stackCmd)
                let features = stackRows(featureRows, device: device, on: stackCmd)
                stackCmd.commit()
                stackCmd.waitUntilCompleted()
                return (logits, features)
            }
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
