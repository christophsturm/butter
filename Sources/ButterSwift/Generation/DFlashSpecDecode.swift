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
//   3. Snapshot the TARGET's caches, then verify the WHOLE (possibly
//      confidence-truncated — see ADAPTIVE CONTROL below) block in one
//      batched `forwardManyLogitsAndFeatures` pass — this both scores
//      every drafted token against the target's real argmax AND
//      captures the `targetLayerIDs` features every accepted row will
//      hand to the NEXT draft call.
//   4. Accept the longest matching prefix (`SpeculativeAccept.verify`,
//      the SAME greedy longest-prefix machinery `SpecDecode`/
//      `MTPSpecDecode` use).
//   5. EXACTNESS DISCIPLINE — the batched verify pass's logits decide
//      ACCEPTANCE ONLY, NEVER emission or cache content. On EVERY
//      cycle — partial, zero, AND full accept alike — the target's
//      caches are restored from the pre-verify snapshot and the
//      committed prefix is REPLAYED through the single-token
//      `forwardOneWithFeatures` path; the token that gets EMITTED, and
//      the target-layer features that seed the NEXT draft block, both
//      come from that replay, never from the batched pass directly.
//
//      This is DELIBERATELY MORE CONSERVATIVE than `SpecDecode`'s/
//      `MTPSpecDecode`'s own "full accept trusts the batched pass"
//      shortcut (which this file used to mirror — see git history).
//      That shortcut's premise is: the batched kernel's own argmax
//      already agreed with every drafted token, so its cache write
//      must be fine to keep. That premise is true for TOKEN IDENTITY
//      but not for CACHE CONTENT — batched and single-token kernels
//      can drift by ULPs (documented risk, previously assumed
//      harmless on full accept), and unlike the chained-AR MTP/SpecDecode
//      drivers, DFlash's confidence-gated truncation (BUTTER_GLIMMER_
//      DFLASH_PMIN) plus adaptive block-size shrinkage routinely drive
//      `draftedSuffix.count` down to 2-3 candidates, making FULL ACCEPT
//      common (not rare, as it is at a fixed blockSize=16) — giving
//      that latent ULP drift many more chances to accumulate in the
//      cache and eventually flip an argmax. Confirmed by direct
//      repro: forcing single-token replay even on full accept turns a
//      reproducible byte-identical FAILURE (on high-entropy content,
//      right as the adaptive controller's bail path — which has no
//      accept/verify safety net of its own — starts reading that
//      drifted cache) into an exact match. See
//      `DFlashAdaptiveBlockBailIntegrationTests`/the forced-bail gate
//      in `GlimmerDFlashSpecDecodeIntegrationTests` for the regression
//      coverage. The extra replay costs some of full accept's speed
//      win — see ADAPTIVE CONTROL's retuned thresholds below, which
//      account for this.
//
// Losslessness: the emitted token stream is byte-identical to plain
// greedy `MuseGlimmerModel.forward` decode BY CONSTRUCTION — see (5)
// above — regardless of drafter quality OR how the ADAPTIVE CONTROL
// below chooses to draft. Neither lever below touches acceptance or
// emission semantics; both only decide how many candidates get
// PROPOSED (and whether a cycle drafts at all) — see each one's doc
// comment. `DFlashSpecDecodeGate` tests (`Tests/ModelIntegrationTests`)
// assert this directly.
//
// ── ADAPTIVE CONTROL (worst-case-slowdown fix) ─────────────────────────
//
// Measured on open-ended (non-templated) prompts: the fixed-γ=15 loop
// nets a SLOWDOWN (0.33-0.47x vs plain decode) once acceptance drops to
// 13-23% — verifying 15 mostly-wrong candidates through the 30B target
// costs more than the tokens it recovers, and every low-acceptance
// cycle also pays a full snapshot+restore+replay. Two independent,
// env-gated levers attack this, mirroring `MTPSpecDecode`'s adaptive
// machinery's "trust realized outcomes, not drafter self-confidence"
// design (see `SpecDecode.MTPAdaptiveGamma`'s doc comment) plus one
// DFlash-specific twist (a distribution-free confidence gate DOES help
// here, unlike the Qwen MTP head, because DFlash's draft logits are
// pushed through the TARGET's own `lm_head` — a low top-1 probability
// there is a direct, well-calibrated signal that THIS target would not
// have picked that token, not just an artifact of an undertrained draft
// head):
//
//   1. BUTTER_GLIMMER_DFLASH_PMIN=<float>: confidence-gated draft
//      truncation. The 5-layer drafter is cheap — it always computes
//      the FULL block's `blockSize - 1` slots (no early-exit inside the
//      joint non-causal attention pass). What this gate cuts is how
//      many of those slots get sent to the EXPENSIVE step — the target
//      verify batch — by truncating `draftedSuffix` (and the block sent
//      to `forwardManyLogitsAndFeatures`) at the first slot whose
//      target-`lm_head`-backed top-1 probability falls below `pMin`.
//      Verifying 4 confident tokens beats verifying 15 mostly-wrong
//      ones: verify cost scales with block size, and a shorter
//      truncated block also means a partial-accept replay (if any)
//      covers fewer positions. This requires ZERO new accept-plan
//      logic — `draftedSuffix`/`blockTokens` are just shorter arrays
//      feeding the SAME verify/accept/replay code every other cycle
//      already runs.
//
//      Floor of 2: the gate never truncates below 2 kept candidates
//      (verify batch of 3 rows), mirroring `MTPSpecDecode`'s own
//      BUTTER_QWEN_MTP_PMIN floor-of-2 — a confirmed, general
//      `Ops.dequantGemmDynamicM` small-M (BM=2, T=2) kernel bug lives
//      in the SAME quantized-GEMM path Muse Glimmer's 4-bit verify
//      pass routes through, so a 2-row verify batch is avoided
//      unconditionally rather than re-deriving model-specific proof
//      the bug doesn't also reach here.
//
//   2. BUTTER_GLIMMER_DFLASH_ADAPTIVE=1: accept-length EMA controller
//      (`DFlashAdaptiveBlock`) over a discrete block-size LADDER
//      (`[blockSize, blockSize/2, ..., 4]`, e.g. `[16, 8, 4]`) plus a
//      BAIL state below the smallest rung. Gated on REALIZED trailing
//      acceptance (a hysteresis band, same shape as
//      `SpecDecode.MTPAdaptiveGamma`): low trailing acceptance steps
//      the block size DOWN one rung; once even the smallest rung can't
//      clear the low bound, drafting stops entirely for
//      BUTTER_GLIMMER_DFLASH_BAIL_TOKENS plain single-token target
//      decode steps (see the loop's bail branch) rather than continuing
//      to pay per-cycle draft+verify+snapshot overhead on content the
//      drafter clearly can't predict. High trailing acceptance steps
//      the rung back UP (by at most one per cycle) — so a templated
//      stretch inside an otherwise-open-ended generation re-earns the
//      full block, same hysteresis-driven recovery MTP's controller
//      uses. See `DFlashAdaptiveBlock`'s doc comment for the controller
//      itself.
//
// Unset/empty ⇒ both levers OFF, behavior byte-identical to the
// original fixed-blockSize loop — matching every other env-gated knob's
// contract elsewhere in this codebase.
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
    /// Draft candidates PROPOSED TO VERIFY (post-truncation when
    /// BUTTER_GLIMMER_DFLASH_PMIN is set; post-rung-shrink when
    /// BUTTER_GLIMMER_DFLASH_ADAPTIVE is set — otherwise always
    /// `blockSize - 1` per verify step, matching the original
    /// fixed-block behavior).
    public let candidatesProposed: Int
    public let candidatesAccepted: Int
    /// Single-token target forwards run to replay accepted prefixes on
    /// partial/zero accepts (the exactness discipline).
    public let replaySingleSteps: Int
    /// Plain single-token target decode steps run while
    /// `BUTTER_GLIMMER_DFLASH_ADAPTIVE=1`'s controller is bailed (no
    /// drafting at all this cycle). Always 0 when the feature is off.
    public let bailSteps: Int
    /// Number of times the controller entered bail (0 when the feature
    /// is off or trailing acceptance never craters at the smallest
    /// rung).
    public let bailEntries: Int
    /// Verify cycles where BUTTER_GLIMMER_DFLASH_PMIN's confidence gate
    /// cut the block short before it reached verify. Always 0 when the
    /// feature is off.
    public let truncatedCycles: Int
    /// Mean effective block size across drafting cycles (excludes bail
    /// steps, which draft nothing). Equals the drafter's configured
    /// `block_size` when BUTTER_GLIMMER_DFLASH_ADAPTIVE is off.
    public let avgBlockSize: Double
    public let wallclockSeconds: Double
    /// Wall-clock attribution (BUTTER_GLIMMER_DFLASH_PROF=1 line at
    /// the call site prints these; always accumulated, cheap timers).
    public let draftSeconds: Double
    public let verifySeconds: Double
    public let replaySeconds: Double
    public let bailSeconds: Double

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
    ///
    /// See the file header's ADAPTIVE CONTROL section for
    /// BUTTER_GLIMMER_DFLASH_PMIN / BUTTER_GLIMMER_DFLASH_ADAPTIVE (plus
    /// _ADAPT_WINDOW / _ADAPT_LO / _ADAPT_HI / _BAIL_TOKENS) — all
    /// unset/empty by default, byte-identical to the original
    /// fixed-block loop.
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
        let blockSizeCap = draft.config.blockSize
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
        var bailSteps = 0
        var truncatedCycles = 0
        var msDraft = 0.0, msVerify = 0.0, msReplay = 0.0, msBail = 0.0
        let t0 = Date()

        // ── BUTTER_GLIMMER_DFLASH_PMIN=<float>: confidence-gated draft
        // truncation (lever 1 — see the file header). Unset/empty ⇒
        // feature OFF, every cycle sends the full (possibly
        // rung-shrunk) block to verify, matching the original loop.
        let pMinEnv = ButterEnv["BUTTER_GLIMMER_DFLASH_PMIN"]
        let pMin: Float? = (pMinEnv?.isEmpty == false) ? Float(pMinEnv!) : nil
        // Never truncate below this many kept candidates — see the file
        // header's floor-of-2 rationale (dodges a general small-M
        // quantized-GEMM kernel bug at a 2-row verify batch).
        let confidenceFloor = 2

        // ── BUTTER_GLIMMER_DFLASH_ADAPTIVE=1: accept-length EMA
        // block-size + bail controller (lever 2 — see the file header).
        // Unset/empty ⇒ feature OFF, every cycle drafts the full
        // `blockSizeCap`, matching the original loop.
        let adaptiveEnabled = ButterEnv["BUTTER_GLIMMER_DFLASH_ADAPTIVE"] == "1"
        let adaptWindowEnv = ButterEnv["BUTTER_GLIMMER_DFLASH_ADAPT_WINDOW"]
        let adaptWindow = (adaptWindowEnv?.isEmpty == false) ? (Int(adaptWindowEnv!) ?? 24) : 24
        let adaptLoEnv = ButterEnv["BUTTER_GLIMMER_DFLASH_ADAPT_LO"]
        // Retuned well above the original 0.45 — see the file header's
        // EXACTNESS DISCIPLINE section for why a drafting cycle's
        // break-even acceptance is much higher now that every cycle
        // unconditionally replays (no more "full accept is free").
        let adaptLo = (adaptLoEnv?.isEmpty == false) ? (Double(adaptLoEnv!) ?? 0.75) : 0.75
        let adaptHiEnv = ButterEnv["BUTTER_GLIMMER_DFLASH_ADAPT_HI"]
        let adaptHi = (adaptHiEnv?.isEmpty == false) ? (Double(adaptHiEnv!) ?? 0.90) : 0.90
        let bailTokensEnv = ButterEnv["BUTTER_GLIMMER_DFLASH_BAIL_TOKENS"]
        let bailTokens = (bailTokensEnv?.isEmpty == false) ? (Int(bailTokensEnv!) ?? 32) : 32
        // Descending rung ladder, e.g. [16, 8, 4] for the checkpoint's
        // default block_size=16 — halve down to a floor of 4 (below
        // that the block barely proposes any candidates at all; bail
        // takes over instead of shrinking further).
        var rungs: [Int] = [blockSizeCap]
        var r = blockSizeCap
        while r > 4 {
            r = max(4, r / 2)
            if r != rungs.last { rungs.append(r) }
        }
        var adaptiveBlock: DFlashAdaptiveBlock? =
            adaptiveEnabled
            ? DFlashAdaptiveBlock(
                rungs: rungs, window: adaptWindow, lo: adaptLo, hi: adaptHi,
                bailTokens: bailTokens)
            : nil

        // Accumulates target-layer features for every plain-decode
        // token emitted while bailed — folded into `contextFeatures` in
        // ONE shot when drafting resumes so the draft's own KV cache
        // (and its internal RoPE position clock, `cache.length`) sees
        // the WHOLE bailed span's context, not just the last token of
        // it. See the file header's ADAPTIVE CONTROL section.
        var bailFeatureRows: [Tensor] = []

        loop: while tokensGenerated < maxNewTokens {
            // ── Bail: plain single-token target decode, no drafting ──
            if adaptiveBlock?.isBailed == true {
                let bT = Date()
                // `forwardOneWithFeatures` self-commits+waits its own
                // private buffer and never touches `cmd` — do NOT
                // commit/wait `cmd` here (see that function's doc
                // comment). Doing so was a diagnosed double-commit that
                // wedged the Metal queue under load.
                //
                // `autoreleasepool`: this branch runs once per bailed
                // token — up to `bailTokens` (32+) times per bail span,
                // many spans per generation — each iteration creating a
                // fresh MTLCommandBuffer + several ObjC-bridged Metal
                // objects (encoders, etc.) inside `forwardOneWithFeatures`.
                // Without draining per-iteration, ARC only releases
                // those at the ENCLOSING autoreleasepool's drain point
                // (the outer test-runner loop, effectively never inside
                // a single generateGreedy call) — the same MTLBuffer/
                // command-buffer accumulation pattern diagnosed
                // elsewhere as an M5 Max GPU-queue wedge under sustained
                // load (cooperative-executor threads never drain autorelease pools).
                let (logits, features) = autoreleasepool { () -> (Tensor, Tensor) in
                    let cmd = device.makeCommandBuffer()
                    return model.forwardOneWithFeatures(
                        tokenId: prev, position: pos, targetLayerIDs: draft.config.targetLayerIDs,
                        caches: caches, on: cmd, device: device)
                }
                msBail += Date().timeIntervalSince(bT)

                if stopTokens.contains(prev) { break loop }
                history.append(prev)
                emit?(prev)
                tokensGenerated += 1
                bailSteps += 1
                bailFeatureRows.append(features)
                adaptiveBlock?.consumeBailToken()

                if tokensGenerated >= maxNewTokens { break loop }

                let flat = logits.toFloatArray()
                prev = argmax(flat, offset: 0, count: flat.count)
                pos += 1

                if adaptiveBlock?.isBailed == false {
                    // Bail just ended: fold the whole accumulated span
                    // into `contextFeatures` for the next draft call.
                    contextFeatures = concatVertically(bailFeatureRows, device: device)
                    bailFeatureRows.removeAll(keepingCapacity: true)
                }
                continue loop
            }

            let thisBlockSize = adaptiveBlock?.nextCycleBlockSize() ?? blockSizeCap

            // ── 1. Draft one block ────────────────────────────────────
            let dT = Date()
            var blockTokens =
                [prev] + Array(repeating: draft.config.maskTokenId, count: thisBlockSize - 1)
            let idsTensor = autoreleasepool { () -> Tensor in
                let idsBuf = device.makeBuffer(length: thisBlockSize * 4)
                idsBuf.contents().withMemoryRebound(to: UInt32.self, capacity: thisBlockSize) { p in
                    for (i, t) in blockTokens.enumerated() { p[i] = UInt32(t) }
                }
                return Tensor(buffer: idsBuf, offset: 0, shape: [thisBlockSize], dtype: .u32)
            }

            // `autoreleasepool` around each of this cycle's command
            // buffers — see the bail/verify/replay sites' comments for
            // why (draining per-cycle rather than accumulating for the
            // whole `generateGreedy` call).
            let noiseEmbedding = autoreleasepool { () -> Tensor in
                let embedCmd = device.makeCommandBuffer()
                let e = model.embedTokens(idsTensor, on: embedCmd)
                embedCmd.commit()
                embedCmd.waitUntilCompleted()
                return e
            }

            let draftHiddenAll = autoreleasepool { () -> Tensor in
                let draftCmd = device.makeCommandBuffer()
                let h = draft.forward(
                    noiseEmbedding: noiseEmbedding, contextFeatures: contextFeatures,
                    caches: draftCaches, on: draftCmd, device: device)
                draftCmd.commit()
                draftCmd.waitUntilCompleted()
                return h
            }
            // Trim the block rows this call just appended — see
            // `DFlashDraftModel.forward`'s doc comment. Unconditional
            // (uses `thisBlockSize`, the size actually drafted this
            // cycle — independent of any later confidence truncation
            // below, which only shortens what gets sent to VERIFY).
            for lc in draftCaches { lc.truncate(toLength: lc.length - thisBlockSize) }

            let draftTail = draftHiddenAll.slicedRows(start: 1, count: thisBlockSize - 1)
            let draftLogits = autoreleasepool { () -> Tensor in
                let dLogitsCmd = device.makeCommandBuffer()
                let l = model.lmHead.callMany(
                    draftTail, t: thisBlockSize - 1, on: dLogitsCmd, device: device)
                dLogitsCmd.commit()
                dLogitsCmd.waitUntilCompleted()
                return l
            }
            let dFlat = draftLogits.toFloatArray()
            let dVocab = dFlat.count / (thisBlockSize - 1)
            var draftedSuffix: [Int] = []
            draftedSuffix.reserveCapacity(thisBlockSize - 1)
            var truncatedHere = false
            for j in 0 ..< (thisBlockSize - 1) {
                if let pMin {
                    let (idx, pTop) = argmaxAndTopProb(dFlat, offset: j * dVocab, count: dVocab)
                    // Confidence gate: stop proposing once the drafter's
                    // OWN top-token probability (through the target's
                    // lm_head) drops below pMin — but never below
                    // `confidenceFloor` kept candidates (see the file
                    // header's small-M kernel-bug floor).
                    if draftedSuffix.count >= confidenceFloor, pTop < pMin {
                        truncatedHere = true
                        break
                    }
                    draftedSuffix.append(idx)
                } else {
                    draftedSuffix.append(argmax(dFlat, offset: j * dVocab, count: dVocab))
                }
            }
            for j in 0 ..< draftedSuffix.count { blockTokens[1 + j] = draftedSuffix[j] }
            blockTokens = Array(blockTokens.prefix(1 + draftedSuffix.count))
            if truncatedHere { truncatedCycles += 1 }
            candidatesProposed += draftedSuffix.count
            msDraft += Date().timeIntervalSince(dT)

            // ── 2. Snapshot + batched verify ──────────────────────────
            let snap = caches.snapshotAll(device: device)
            let vT = Date()
            // `forwardManyLogitsAndFeatures` self-commits+waits its own
            // private buffer and never touches `vCmd` — do NOT
            // commit/wait it here (see that function's doc comment;
            // this was the diagnosed double-commit that wedged the
            // Metal queue mid-prefill).
            // Only `logitsAll` is needed — `forwardManyLogitsAndFeatures`
            // also returns per-row `targetLayerIDs` features, but this
            // driver no longer trusts ANY batched-pass output besides
            // the accept/reject DECISION (see the file header's
            // EXACTNESS DISCIPLINE section) — the features that seed
            // the next draft call always come from the replay below
            // instead, never from here.
            //
            // `autoreleasepool` — see the bail/replay sites' comments.
            // This call alone builds a `[blockTokens.count, hidden]`-wide
            // internal graph (every layer, every row) per cycle; over a
            // full multi-prompt gate run that's still thousands of
            // ObjC-bridged Metal objects with no drain point between
            // cycles absent this.
            let logitsAll = autoreleasepool { () -> Tensor in
                let vCmd = device.makeCommandBuffer()
                let (l, _) = model.forwardManyLogitsAndFeatures(
                    tokenIds: blockTokens, startingAt: pos,
                    targetLayerIDs: draft.config.targetLayerIDs,
                    caches: caches, on: vCmd, device: device)
                return l
            }
            verifySteps += 1
            msVerify += Date().timeIntervalSince(vT)

            let flat = logitsAll.toFloatArray()
            let rows = blockTokens.count
            let vocab = flat.count / rows
            precondition(
                flat.count == rows * vocab,
                "DFlashSpecDecode: verify logits have \(flat.count) elements; "
                    + "expected rows*vocab = \(rows * vocab)")
            // `confidenceFloor` (above) is meant to guarantee
            // `blockTokens.count` (the verify GEMM's T) never hits the
            // confirmed BM=2/T=2 `dequantGemmDynamicM` small-M kernel
            // bug (see the file header). Assert it rather than trust
            // it — cheap, and a real regression here would silently
            // corrupt logits instead of crashing.
            precondition(
                rows != 2,
                "DFlashSpecDecode: verify block hit T=2 (blockTokens=\(blockTokens)) — "
                    + "the confidenceFloor=2 guard should make this unreachable")
            var verifierTokens: [Int] = []
            verifierTokens.reserveCapacity(draftedSuffix.count)
            for j in 0 ..< draftedSuffix.count {
                verifierTokens.append(argmax(flat, offset: j * vocab, count: vocab))
            }
            let batchedBonus = argmax(flat, offset: (rows - 1) * vocab, count: vocab)

            let outcome = SpeculativeAccept.verify(
                draft: draftedSuffix, verifierTokens: verifierTokens, bonusToken: batchedBonus)
            let k = outcome.acceptedDraft.count
            candidatesAccepted += k
            let committed = [prev] + outcome.acceptedDraft
            // Feed the realized (post-truncation) outcome back into the
            // controller for the NEXT cycle's block-size/bail decision.
            adaptiveBlock?.record(accepted: k, proposed: draftedSuffix.count)
            let willBail1 = adaptiveBlock?.isBailed ?? false

            // ── Restore + replay, UNCONDITIONALLY (see the file
            // header's EXACTNESS DISCIPLINE section for why this no
            // longer branches on full-vs-partial accept) ─────────────
            let rT = Date()
            caches.restoreAll(from: snap, device: device)
            // Chained replay — ONE command buffer for the whole
            // `committed.count`-token replay instead of `committed.count`
            // separate self-committing calls (see
            // `forwardManySingleStepChained`'s doc comment: reducing
            // total command-buffer submission COUNT is what actually
            // addresses the sustained-load Metal queue wedge; replay is
            // now unconditional every cycle, so this was the highest-
            // volume site in the whole driver).
            let (lastLogits, replayFeatureRows) = autoreleasepool {
                () -> (Tensor, [Tensor]) in
                model.forwardManySingleStepChained(
                    tokens: committed.enumerated().map { (tokenId: $1, position: pos + $0) },
                    targetLayerIDs: draft.config.targetLayerIDs,
                    caches: caches, device: device)
            }
            replaySingleSteps += committed.count
            let lastFlat = lastLogits.toFloatArray()
            let nextTok = argmax(lastFlat, offset: 0, count: lastFlat.count)
            let nextContext = concatVertically(replayFeatureRows, device: device)
            msReplay += Date().timeIntervalSince(rT)

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
            if willBail1 {
                // This cycle just triggered enterBail() (via record()
                // above) — the NEXT loop iteration takes the bail
                // branch, which doesn't consume `contextFeatures` at
                // all (it folds `bailFeatureRows` into it only once the
                // bail span ENDS). Without this, `nextContext` — this
                // cycle's own committed-token feature rows, exactly as
                // legitimate a slice of "context since the last draft()
                // call" as any other cycle's — would simply be
                // discarded when `contextFeatures` gets overwritten at
                // BAIL-END, permanently starving the draft's KV cache
                // of these rows (never re-added by any later call).
                // Seed `bailFeatureRows` with them instead so BAIL-END's
                // fold includes this cycle's rows too, ahead of the
                // bail span's own — preserving the "one context row per
                // emitted token, chronological order" invariant across
                // the transition. Doesn't affect correctness (a
                // drafter-quality-only lever, see the file header) —
                // this is purely restoring the context the resumed
                // drafter conditions on.
                let n = nextContext.shape[0]
                for r in 0 ..< n {
                    bailFeatureRows.append(nextContext.slicedRows(start: r, count: 1))
                }
            } else {
                contextFeatures = nextContext
            }
        }

        if ButterEnv["BUTTER_GLIMMER_DFLASH_PROF"] == "1" {
            let avgBlock = adaptiveBlock?.avgBlockSize ?? Double(blockSizeCap)
            let msg =
                "[dflash-prof] draft=\(String(format: "%.3f", msDraft))s "
                + "verify=\(String(format: "%.3f", msVerify))s "
                + "replay=\(String(format: "%.3f", msReplay))s "
                + "bail=\(String(format: "%.3f", msBail))s "
                + "avgBlock=\(String(format: "%.2f", avgBlock)) "
                + "truncated=\(truncatedCycles) bailSteps=\(bailSteps) "
                + "bailEntries=\(adaptiveBlock?.bailEntries ?? 0)\n"
            FileHandle.standardError.write(msg.data(using: .utf8)!)
        }

        return DFlashSpecDecodeStats(
            tokensGenerated: tokensGenerated,
            verifySteps: verifySteps,
            candidatesProposed: candidatesProposed,
            candidatesAccepted: candidatesAccepted,
            replaySingleSteps: replaySingleSteps,
            bailSteps: bailSteps,
            bailEntries: adaptiveBlock?.bailEntries ?? 0,
            truncatedCycles: truncatedCycles,
            avgBlockSize: adaptiveBlock?.avgBlockSize ?? Double(blockSizeCap),
            wallclockSeconds: Date().timeIntervalSince(t0),
            draftSeconds: msDraft, verifySeconds: msVerify, replaySeconds: msReplay,
            bailSeconds: msBail)
    }
}

extension DFlashSpecDecode {

    /// Adaptive block-size + bail controller for the DFlash draft loop,
    /// gated on REALIZED trailing acceptance — same "honest signal"
    /// rationale as `SpecDecode.MTPAdaptiveGamma` (see its doc comment):
    /// realized accept/reject outcomes are trusted over drafter
    /// self-confidence for deciding how MUCH to draft next.
    ///
    /// Unlike MTP's unit-step gamma (1...cap), DFlash's per-cycle cost
    /// is dominated by the TARGET verify pass (a 30B forward), not the
    /// 5-layer drafter, so the lever here is a discrete block-size
    /// LADDER (`rungs`, strictly descending, e.g. `[16, 8, 4]`) plus a
    /// BAIL state below the smallest rung: once trailing acceptance
    /// craters even at the smallest block, stop drafting entirely for
    /// `bailTokens` tokens (plain single-token target decode — see
    /// `generateGreedy`'s bail branch) rather than continuing to pay
    /// per-cycle draft+verify+snapshot overhead on content the drafter
    /// clearly can't predict.
    ///
    /// Pure logic, no model/tensor dependency — unit-testable directly
    /// (`DFlashAdaptiveBlockTests`), same spirit as `MTPAdaptiveGamma`.
    /// Never affects accept semantics: only how many candidates get
    /// PROPOSED per cycle (and whether a cycle drafts at all) — see
    /// `DFlashSpecDecode.swift`'s file header. A bad block-size/bail
    /// choice costs speed, never correctness.
    struct DFlashAdaptiveBlock {
        /// Descending block sizes, e.g. `[16, 8, 4]`. `rungs[0]` is the
        /// ceiling — callers pass the drafter checkpoint's configured
        /// `block_size` first.
        let rungs: [Int]
        /// Trailing window size, in per-candidate accept/reject
        /// outcomes (not cycles).
        let window: Int
        /// Step the rung DOWN by one (or, at the smallest rung, enter
        /// bail) once trailing acceptance falls strictly below this.
        let lo: Double
        /// Step the rung UP by one (ceiling `rungs[0]`) once trailing
        /// acceptance rises strictly above this. Requires `lo < hi`.
        let hi: Double
        /// Plain-decode tokens to run once bailed before resuming
        /// drafting at the smallest rung.
        let bailTokens: Int

        private(set) var rungIndex = 0
        private(set) var bailRemaining = 0

        // Trailing window, same ring-buffer shape as MTPAdaptiveGamma.
        private var ring: [Bool]
        private var ringNext = 0
        private var ringFilled = 0
        private var trueCount = 0

        // Cross-run stats for the `[dflash-prof]` line.
        private(set) var blockSizeCycleSum = 0
        private(set) var blockSizeCycles = 0
        private(set) var bailEntries = 0
        private(set) var bailedTokensTotal = 0

        init(rungs: [Int], window: Int, lo: Double, hi: Double, bailTokens: Int) {
            precondition(!rungs.isEmpty, "DFlashAdaptiveBlock: rungs must be non-empty")
            precondition(
                zip(rungs, rungs.dropFirst()).allSatisfy { $0 > $1 },
                "DFlashAdaptiveBlock: rungs must be strictly descending")
            precondition(window >= 1, "DFlashAdaptiveBlock: window must be >= 1")
            precondition(lo < hi, "DFlashAdaptiveBlock: lo (\(lo)) must be < hi (\(hi))")
            precondition(bailTokens >= 1, "DFlashAdaptiveBlock: bailTokens must be >= 1")
            self.rungs = rungs
            self.window = window
            self.lo = lo
            self.hi = hi
            self.bailTokens = bailTokens
            self.ring = Array(repeating: false, count: window)
        }

        /// Effective block size at the current rung.
        var blockSize: Int { rungs[rungIndex] }
        /// True while a bail span is in progress — the caller must skip
        /// drafting entirely and run plain single-token decode instead.
        var isBailed: Bool { bailRemaining > 0 }

        /// Enter bail: stop drafting for `bailTokens` plain-decode
        /// tokens, resuming at the SMALLEST rung once it ends. Clears
        /// the trailing window — content on the far side of a bail span
        /// may look nothing like what triggered it, so climbing back up
        /// re-warms on fresh evidence rather than stale rejects.
        mutating func enterBail() {
            bailRemaining = bailTokens
            bailEntries += 1
            rungIndex = rungs.count - 1
            ring = Array(repeating: false, count: window)
            ringNext = 0
            ringFilled = 0
            trueCount = 0
        }

        /// Call once per plain-decode token emitted while bailed.
        mutating func consumeBailToken() {
            precondition(bailRemaining > 0, "DFlashAdaptiveBlock.consumeBailToken: not bailed")
            bailRemaining -= 1
            bailedTokensTotal += 1
        }

        /// Effective block size for the cycle about to draft — call
        /// exactly once per drafting cycle (never during bail). Folds
        /// into the avg-block-size stat as a side effect.
        mutating func nextCycleBlockSize() -> Int {
            blockSizeCycleSum += blockSize
            blockSizeCycles += 1
            return blockSize
        }

        /// Record this cycle's realized (post-truncation) accept
        /// outcome and step the rung — or trigger bail — for the NEXT
        /// cycle. `proposed` should be however many candidates were
        /// ACTUALLY sent to verify this cycle (post
        /// BUTTER_GLIMMER_DFLASH_PMIN truncation, if any) — the honest
        /// signal is what verify actually saw. A draft-free cycle
        /// (`proposed == 0`) is a no-op.
        mutating func record(accepted: Int, proposed: Int) {
            guard proposed > 0 else { return }
            precondition(
                accepted >= 0 && accepted <= proposed,
                "DFlashAdaptiveBlock.record: accepted (\(accepted)) out of range for "
                    + "proposed (\(proposed))")
            for i in 0 ..< proposed { push(i < accepted) }
            // Warmup: hold until the window has observed a full
            // `window` candidates.
            guard ringFilled >= window else { return }
            let rate = Double(trueCount) / Double(ringFilled)
            if rate < lo {
                if rungIndex == rungs.count - 1 {
                    enterBail()
                } else {
                    rungIndex += 1
                }
            } else if rate > hi {
                rungIndex = max(0, rungIndex - 1)
            }
            // rate in [lo, hi]: hold — the hysteresis band.
        }

        private mutating func push(_ accepted: Bool) {
            if ringFilled == window {
                if ring[ringNext] { trueCount -= 1 }
            } else {
                ringFilled += 1
            }
            ring[ringNext] = accepted
            if accepted { trueCount += 1 }
            ringNext = (ringNext + 1) % window
        }

        var avgBlockSize: Double {
            blockSizeCycles > 0 ? Double(blockSizeCycleSum) / Double(blockSizeCycles) : 0
        }
    }
}

/// Stack `n` `[1, width]` row tensors into one `[n, width]` tensor —
/// used to assemble the replay path's (and, under
/// BUTTER_GLIMMER_DFLASH_ADAPTIVE, the bail path's) per-row feature
/// captures into the shape `contextFeatures` needs for the next draft
/// call.
private func concatVertically(_ rows: [Tensor], device: Device) -> Tensor {
    precondition(!rows.isEmpty, "concatVertically: rows must be non-empty")
    let width = rows[0].elementCount
    let dtype = rows[0].dtype
    return autoreleasepool { () -> Tensor in
        let dest = Tensor.empty(shape: [rows.count, width], dtype: dtype, device: device)
        let cmd = device.makeCommandBuffer()
        for (i, row) in rows.enumerated() {
            Ops.copy(row.reshaped(to: [1, width]), into: dest.slicedRows(start: i, count: 1), on: cmd)
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        return dest
    }
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

/// Argmax + exact softmax top-1 probability over one logits row, in the
/// same host readback (BUTTER_GLIMMER_DFLASH_PMIN's confidence gate).
/// Two passes over `count` floats: the first finds `bestIdx`/`bestVal`
/// (identical to `argmax` above), the second computes the numerically
/// stable normalizer `s = Σ exp(l_i - bestVal)`. Since
/// `exp(bestVal - bestVal) == 1`, the top probability is `1 / s` — no
/// separate `exp(l_max - logsumexp(l))` term needed. Mirrors
/// `MTPSpecDecode.swift`'s identically-named/shaped private helper
/// (kept file-local rather than shared, matching this codebase's
/// existing per-file duplication of small argmax helpers).
@inline(__always)
private func argmaxAndTopProb(_ flat: [Float], offset: Int, count: Int) -> (Int, Float) {
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
    var s: Float = 0
    for i in 0 ..< count {
        s += Foundation.exp(flat[offset + i] - bestVal)
    }
    return (bestIdx, 1 / s)
}
