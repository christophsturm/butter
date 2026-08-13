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
// DFlashSpecDecode — the DFlash block-diffusion self-speculative decode
// loop for Muse Glimmer 30B (see `DFlashDrafter.swift`'s header for the
// drafter's architecture/contract). Same driver shape as
// `SpecDecode.generateGreedy` / `MTPSpecDecode`'s greedy loop, adapted
// for a BLOCK drafter instead of a chained-AR one:
//
//   1. Draft ONE block of `blockSize` slots — `[anchor, mask, ..., mask]`
//      — via `DFlashDraftModel.forward`, conditioned on `contextFeatures`
//      (the running `targetLayerIDs` hidden-state history the PREVIOUS
//      iteration's verify pass captured). Greedy-argmax the drafter's
//      own (target `lm_head`-backed) logits at slots `[1..<blockSize]`
//      to fill in `blockSize - 1` candidate tokens.
//   2. Trim the draft's own per-layer caches back down by exactly
//      `blockSize` (the block query rows just appended) — see
//      `DFlashDraftModel.forward`'s doc comment; this happens
//      regardless of what verification decides, since the draft cache
//      only ever needs to remember CONTEXT rows, never its own guesses.
//   3. Snapshot the TARGET's caches, then verify the WHOLE block in one
//      batched `forwardManyLogitsAndFeatures` pass — this both scores
//      every drafted token against the target's real argmax AND
//      captures the `targetLayerIDs` features every accepted row will
//      hand to the NEXT draft call.
//   4. Accept the longest matching prefix (`SpeculativeAccept.verify`,
//      the SAME greedy longest-prefix machinery `SpecDecode`/
//      `MTPSpecDecode` use).
//   5. EXACTNESS DISCIPLINE (inherited verbatim from `SpecDecode`/
//      `MTPSpecDecode` — see their headers): the batched verify pass's
//      logits decide ACCEPTANCE ONLY. On a partial/zero accept, the
//      target's caches are restored from the pre-verify snapshot and
//      the accepted prefix is REPLAYED through the single-token
//      `forwardOneWithFeatures` path — batched and single-token kernels
//      can drift by ULPs, so the token that gets EMITTED, and the
//      target-layer features that seed the NEXT draft block, both come
//      from whichever kernel path the baseline plain-greedy loop would
//      have used. On a full accept, the bonus token AND the next
//      context features come straight from the batched pass — no
//      replay needed, matching `SpecDecode`'s existing full-accept
//      handling.
//
// Losslessness: the emitted token stream is byte-identical to plain
// greedy `MuseGlimmerModel.forward` decode BY CONSTRUCTION — see (5)
// above — regardless of drafter quality. A drafter bug can only cost
// acceptance rate / speed, never correctness. `DFlashSpecDecodeGate`
// tests (`Tests/ModelIntegrationTests`) assert this directly.
//
// CACHE ROLLBACK SOUNDNESS (both of the target's cache kinds — sliding
// `.window`-evicting AND full `.unbounded`): this driver reuses the
// EXISTING `caches.snapshotAll()` / `.restoreAll()` composite
// (`CacheSnapshot.swift`), unmodified, the same machinery `SpecDecode`
// and `MTPSpecDecode` already run on Qwen3.5/3.6's mixed cache kinds.
// For a plain `KVCache` the snapshot payload is JUST the `length`
// integer (`.kv(length:absolutePosition:)`) and restore is
// `kv.truncate(toLength:)` — no byte-level K/V restore at all. This is
// sound for a window-evicting ring cache specifically BECAUSE physical
// slot placement is a pure function of absolute position
// (`KVEvictionState.reserveNextSlot`): restoring `length` alone rewinds
// what the cache considers "valid," and the replay's subsequent
// `appendOnGPU` calls recompute the SAME physical slots a plain-decode
// run would have written, byte-identically. There is no ring-wrap
// danger zone to gate around — Glimmer's sliding AND full layers are
// literally the same `KVCache` class (`makeLayerCaches`, differing
// only in `eviction:`), so `restoreAll` already handles both uniformly.
// (This was flagged as an open risk before reading `CacheSnapshot.swift`
// — it turned out to already be a solved problem in-tree.)
//
// Priming: unlike MTP's self-speculative loop (which can defer its
// first real draft by one iteration), DFlash's very first block needs
// REAL `targetLayerIDs` context from the start — there is no
// "draft-free" fallback shape for a block drafter. Callers MUST supply
// `seedFeatures`, the target's `targetLayerIDs` features at the
// position `lastToken` occupies (i.e. captured while producing
// `lastToken` itself during prefill) — see
// `MuseGlimmerModel.forwardOneWithFeatures`.
//
// Greedy-only, like `SpecDecode.generateGreedy` — argmax everywhere.

import Foundation
import Metal

public struct DFlashSpecDecodeStats {
    public let tokensGenerated: Int
    public let verifySteps: Int
    /// Draft candidates proposed (always `blockSize - 1` per verify
    /// step — the drafter always fills the whole block).
    public let candidatesProposed: Int
    public let candidatesAccepted: Int
    /// Single-token target forwards run to replay accepted prefixes on
    /// partial/zero accepts (the exactness discipline).
    public let replaySingleSteps: Int
    public let wallclockSeconds: Double
    /// Wall-clock attribution (BUTTER_GLIMMER_DFLASH_PROF=1 line at
    /// the call site prints these; always accumulated, cheap timers).
    public let draftSeconds: Double
    public let verifySeconds: Double
    public let replaySeconds: Double

    public var acceptanceRate: Double {
        guard candidatesProposed > 0 else { return 0 }
        return Double(candidatesAccepted) / Double(candidatesProposed)
    }
    public var tps: Double {
        guard wallclockSeconds > 0 else { return 0 }
        return Double(tokensGenerated) / wallclockSeconds
    }
}

public enum DFlashSpecDecode {

    /// Run the greedy DFlash block-speculative decode loop.
    ///
    /// - `draftCaches`: fresh per-layer caches from
    ///   `draft.makeCache(maxSeq:)` — persists (and grows) across the
    ///   whole call; pass a fresh set (or `.reset()`  — not exposed on
    ///   `KVCache` today, so callers should just allocate fresh caches
    ///   per generation) when starting a new conversation.
    /// - `seedFeatures`: `[1, targetLayerIDs.count * hidden]` — the
    ///   target's captured features at `lastToken`'s position (from
    ///   prefill's final single-token step). See the file header.
    /// - `caches`: the TARGET's real per-layer caches, already
    ///   reflecting the prompt (same contract as `SpecDecode.
    ///   generateGreedy`).
    public static func generateGreedy(
        model: MuseGlimmerModel,
        draft: DFlashDraftModel,
        draftCaches: [KVCache],
        lastToken: Int,
        position: Int,
        seedFeatures: Tensor,
        caches: [any LayerCacheProtocol],
        history: inout [Int],
        maxNewTokens: Int,
        stopTokens: Set<Int> = [],
        device: Device = .shared,
        emit: ((Int) -> Void)? = nil
    ) -> DFlashSpecDecodeStats {
        precondition(
            maxNewTokens >= 0, "DFlashSpecDecode.generateGreedy: maxNewTokens must be ≥ 0")
        let blockSize = draft.config.blockSize
        let featureWidth = draft.config.targetLayerIDs.count * model.hidden
        precondition(
            seedFeatures.shape == [1, featureWidth],
            "DFlashSpecDecode.generateGreedy: seedFeatures shape \(seedFeatures.shape) "
                + "≠ [1, \(featureWidth)]")

        var pos = position
        var prev = lastToken
        var contextFeatures = seedFeatures
        var tokensGenerated = 0
        var verifySteps = 0
        var candidatesProposed = 0
        var candidatesAccepted = 0
        var replaySingleSteps = 0
        var msDraft = 0.0, msVerify = 0.0, msReplay = 0.0
        let t0 = Date()

        loop: while tokensGenerated < maxNewTokens {
            // ── 1. Draft one block ────────────────────────────────────
            let dT = Date()
            var blockTokens = [prev] + Array(repeating: draft.config.maskTokenId, count: blockSize - 1)
            let idsBuf = device.makeBuffer(length: blockSize * 4)
            idsBuf.contents().withMemoryRebound(to: UInt32.self, capacity: blockSize) { p in
                for (i, t) in blockTokens.enumerated() { p[i] = UInt32(t) }
            }
            let idsTensor = Tensor(buffer: idsBuf, offset: 0, shape: [blockSize], dtype: .u32)

            let embedCmd = device.makeCommandBuffer()
            let noiseEmbedding = model.embedTokens(idsTensor, on: embedCmd)
            embedCmd.commit()
            embedCmd.waitUntilCompleted()

            let draftCmd = device.makeCommandBuffer()
            let draftHiddenAll = draft.forward(
                noiseEmbedding: noiseEmbedding, contextFeatures: contextFeatures,
                caches: draftCaches, on: draftCmd, device: device)
            draftCmd.commit()
            draftCmd.waitUntilCompleted()
            // Trim the block rows this call just appended — see
            // `DFlashDraftModel.forward`'s doc comment. Unconditional:
            // happens whether or not the target later accepts anything.
            for lc in draftCaches { lc.truncate(toLength: lc.length - blockSize) }

            let draftTail = draftHiddenAll.slicedRows(start: 1, count: blockSize - 1)
            let dLogitsCmd = device.makeCommandBuffer()
            let draftLogits = model.lmHead.callMany(
                draftTail, t: blockSize - 1, on: dLogitsCmd, device: device)
            dLogitsCmd.commit()
            dLogitsCmd.waitUntilCompleted()
            let dFlat = draftLogits.toFloatArray()
            let dVocab = dFlat.count / (blockSize - 1)
            var draftedSuffix: [Int] = []
            draftedSuffix.reserveCapacity(blockSize - 1)
            for j in 0 ..< (blockSize - 1) {
                draftedSuffix.append(argmax(dFlat, offset: j * dVocab, count: dVocab))
            }
            for j in 0 ..< (blockSize - 1) { blockTokens[1 + j] = draftedSuffix[j] }
            candidatesProposed += draftedSuffix.count
            msDraft += Date().timeIntervalSince(dT)

            // ── 2. Snapshot + batched verify ──────────────────────────
            let snap = caches.snapshotAll(device: device)
            let vT = Date()
            let vCmd = device.makeCommandBuffer()
            let (logitsAll, featuresAll) = model.forwardManyLogitsAndFeatures(
                tokenIds: blockTokens, startingAt: pos,
                targetLayerIDs: draft.config.targetLayerIDs,
                caches: caches, on: vCmd, device: device)
            vCmd.commit()
            vCmd.waitUntilCompleted()
            verifySteps += 1
            msVerify += Date().timeIntervalSince(vT)

            let flat = logitsAll.toFloatArray()
            let vocab = flat.count / blockSize
            precondition(
                flat.count == blockSize * vocab,
                "DFlashSpecDecode: verify logits have \(flat.count) elements; "
                    + "expected blockSize*vocab = \(blockSize * vocab)")
            var verifierTokens: [Int] = []
            verifierTokens.reserveCapacity(blockSize - 1)
            for j in 0 ..< (blockSize - 1) {
                verifierTokens.append(argmax(flat, offset: j * vocab, count: vocab))
            }
            let batchedBonus = argmax(flat, offset: (blockSize - 1) * vocab, count: vocab)

            let outcome = SpeculativeAccept.verify(
                draft: draftedSuffix, verifierTokens: verifierTokens, bonusToken: batchedBonus)
            let k = outcome.acceptedDraft.count
            candidatesAccepted += k
            let committed = [prev] + outcome.acceptedDraft

            var nextTok: Int
            var nextContext: Tensor
            if k < draftedSuffix.count {
                // ── Partial/zero accept: restore + replay ─────────────
                let rT = Date()
                caches.restoreAll(from: snap, device: device)
                var lastLogits: Tensor!
                var replayFeatureRows: [Tensor] = []
                replayFeatureRows.reserveCapacity(committed.count)
                for (i, tok) in committed.enumerated() {
                    let sCmd = device.makeCommandBuffer()
                    let (l, f) = model.forwardOneWithFeatures(
                        tokenId: tok, position: pos + i,
                        targetLayerIDs: draft.config.targetLayerIDs,
                        caches: caches, on: sCmd, device: device)
                    sCmd.commit()
                    sCmd.waitUntilCompleted()
                    lastLogits = l
                    replayFeatureRows.append(f)
                    replaySingleSteps += 1
                }
                let lastFlat = lastLogits.toFloatArray()
                nextTok = argmax(lastFlat, offset: 0, count: lastFlat.count)
                nextContext = concatVertically(replayFeatureRows, device: device)
                msReplay += Date().timeIntervalSince(rT)
            } else {
                // ── Full accept: caches already hold the verified block ─
                nextTok = batchedBonus
                nextContext = featuresAll
            }

            // ── Commit + emit ─────────────────────────────────────────
            var stopped = false
            var emitted = 0
            for tok in committed {
                if stopTokens.contains(tok) {
                    stopped = true
                    break
                }
                history.append(tok)
                emit?(tok)
                emitted += 1
                tokensGenerated += 1
                if tokensGenerated >= maxNewTokens { break }
            }
            if stopped || emitted < committed.count { break loop }

            pos += committed.count
            prev = nextTok
            contextFeatures = nextContext
        }

        if ButterEnv["BUTTER_GLIMMER_DFLASH_PROF"] == "1" {
            let msg =
                "[dflash-prof] draft=\(String(format: "%.3f", msDraft))s "
                + "verify=\(String(format: "%.3f", msVerify))s "
                + "replay=\(String(format: "%.3f", msReplay))s\n"
            FileHandle.standardError.write(msg.data(using: .utf8)!)
        }

        return DFlashSpecDecodeStats(
            tokensGenerated: tokensGenerated,
            verifySteps: verifySteps,
            candidatesProposed: candidatesProposed,
            candidatesAccepted: candidatesAccepted,
            replaySingleSteps: replaySingleSteps,
            wallclockSeconds: Date().timeIntervalSince(t0),
            draftSeconds: msDraft, verifySeconds: msVerify, replaySeconds: msReplay)
    }
}

/// Stack `n` `[1, width]` row tensors into one `[n, width]` tensor —
/// used to assemble the replay path's per-row feature captures into
/// the shape `contextFeatures` needs for the next draft call.
private func concatVertically(_ rows: [Tensor], device: Device) -> Tensor {
    precondition(!rows.isEmpty, "concatVertically: rows must be non-empty")
    let width = rows[0].elementCount
    let dtype = rows[0].dtype
    let dest = Tensor.empty(shape: [rows.count, width], dtype: dtype, device: device)
    let cmd = device.makeCommandBuffer()
    for (i, row) in rows.enumerated() {
        Ops.copy(row.reshaped(to: [1, width]), into: dest.slicedRows(start: i, count: 1), on: cmd)
    }
    cmd.commit()
    cmd.waitUntilCompleted()
    return dest
}

@inline(__always)
private func argmax(_ flat: [Float], offset: Int, count: Int) -> Int {
    precondition(count > 0)
    var bestIdx = 0
    var bestVal = flat[offset]
    for i in 1 ..< count {
        let v = flat[offset + i]
        if v > bestVal {
            bestVal = v
            bestIdx = i
        }
    }
    return bestIdx
}
