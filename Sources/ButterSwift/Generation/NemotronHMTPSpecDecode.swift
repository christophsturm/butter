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
// NemotronHMTPSpecDecode — the MTP self-speculative decode driver for
// Nemotron-3.5-Lightning-30B-A3B (F-131). Greedy-only (argmax
// everywhere), mirroring `SpecDecode.generateGreedyMTP`'s (Qwen3x) and
// `DFlashSpecDecode`'s (Muse Glimmer) driver shape, adapted for
// NemotronH's architecture — read this header before touching the
// design, it documents WHY the shape differs from both precedents.
//
// ─── DESIGN CHOICE: (a), sequential-through-the-mixer — NOT batched ───
//
// The port brief asked which of two designs the recurrent-state
// constraint forces:
//   (a) process drafted tokens sequentially through the backbone
//       anyway, since Mamba 2's scan is inherently sequential and no
//       multi-token step kernel exists, OR
//   (b) verify via a genuinely batched pass, then replay accepted
//       tokens single-token from a snapshot (the DFlash pattern).
//
// NemotronHModel has NO genuinely batched (M>1 GEMM, multi-row SDPA,
// multi-token Mamba scan) forward path at all — `DecoderLayer.
// decodeMulti`'s DEFAULT implementation (which every NemotronH mamba/
// attention/MLP/MoE layer uses; none override it) is a documented
// per-row LOOP of the same single-token `decode(...)`, and
// `NemotronHModel.forwardMulti` is likewise a bare loop of `forward
// (tokenId:)` — see both files' doc comments. There is no separate
// "batched verify kernel" here to trust or distrust: "verify" and
// "single-token decode" are the SAME function, called the SAME way,
// differing only in whether intermediate results get committed to a
// command buffer individually or as one shared submission (see below).
// Design (a) is therefore not a choice so much as the only thing the
// codebase supports — this file follows it, and the (b)-flavored
// discipline the task brief separately mandates (see FULL-ACCEPT TRUST
// below) is layered on top regardless.
//
// ─── WHAT VERIFY ACTUALLY BUYS: command-buffer batching, not compute ──
//
// Even without a batched-GEMM kernel, encoding γ+1 sequential
// `NemotronHModel.forwardHiddenAndLogits` calls onto ONE command buffer
// (queued back-to-back, no intermediate `commit()`/`waitUntilCompleted
// ()`) and syncing ONCE at the end is cheaper than γ+1 separate
// single-token decode calls, each with its own submit+sync round trip
// — Metal command-buffer submission and CPU/GPU synchronization is
// real, non-zero overhead per call, independent of the FLOPs/bytes
// moved. This holds precisely because every layer in the Lightning
// checkpoint's decode path is non-committing (`DecoderLayer.
// commitsMidDecode == false` — the mamba/attention layers never commit,
// and F-126's GPU-routed `NemotronHNvfp4MoELayer` doesn't either, see
// its file doc); a hypothetical MTP-bearing checkpoint with any
// CPU-routed MoE layer (`commitsMidDecode == true`) would still be
// CORRECT here (each committing layer's own internal `workCmd` commit
// is unaffected by this driver's outer batching) but would lose most
// of this batching benefit.
//
// ─── FULL-ACCEPT TRUST: NOT taken, per the port brief's directive ─────
//
// Because verify and single-token decode share the literal same
// kernel dispatches, there is no ULP-drift hazard here of the kind the
// MTP/DFlash exactness audits found (that hazard is specifically
// between a numerically DIFFERENT batched-GEMM verify kernel and the
// M=1 GEMV decode kernel — no such second kernel exists in this
// codebase for NemotronH). A "trust the batch on full accept" shortcut
// would therefore likely be safe here on first-principles grounds.
// The port brief mandates the conservative discipline regardless
// ("do NOT introduce a new trust-the-batch full-accept path... snapshot
// + replay every cycle"), and this driver follows that literally: on
// EVERY cycle — partial, zero, AND full accept alike — the verify
// pass's cache writes are discarded (full snapshot restore, including
// the Mamba 2 recurrent state via the new `Mamba2LayerCache`/
// `SSMStateCache` content-snapshot machinery this port added — see
// `CacheSnapshot.swift`'s `.mamba2` case) and the committed prefix is
// REPLAYED through `forwardHiddenAndLogits`, ONE TOKEN PER COMMAND
// BUFFER, exactly like `DFlashSpecDecode`'s replay loop.
//
// ─── WHY THIS IS AN ARCHITECTURAL LOSS, NOT A TUNING PROBLEM ──────────
//
// The always-replay discipline cancels the ONLY lever available here.
// Per cycle, accepting k of γ drafted tokens costs:
//   γ   round trips — the draft burst (one per chained MTP head step,
//       same cost model as Qwen3x's `Qwen3xMTPHead.draftStep`)
//   1   round trip  — the batched verify pass (γ+1 rows, one commit)
//   k+1 round trips — the mandatory replay (`prev` + k accepted drafts,
//       one commit each, matching DFlash's literal per-token loop)
// vs. plain greedy decode emitting the same k+1 tokens: k+1 round
// trips, no draft/verify overhead. Extra round trips per cycle:
//   γ + 1 + (k+1) − (k+1) = γ + 1
// — a constant γ+1 round-trip tax EVERY cycle, independent of k (i.e.
// independent of how good the drafter is), PLUS real wasted backbone
// compute for the (γ−k) rejected verify rows (no early-exit — a batch
// pass always runs every row), PLUS the drafter's own compute. There is
// no acceptance rate at which this can net positive: the verify pass's
// command-buffer-batching win is paid right back by the mandatory
// replay, unconditionally, every cycle. See the port's gate results
// for the measured magnitude — this is reported as a loss and the
// feature ships opt-in-OFF per the brief's explicit allowance
// ("if it's a loss, it ships opt-in-OFF... still merges as wired
// foundation").
//
// ─── Losslessness ──────────────────────────────────────────────────────
//
// Byte-identical to plain greedy `NemotronHModel.forward` decode BY
// CONSTRUCTION: every emitted token comes from `forwardHiddenAndLogits`
// called with EXACTLY the committed token/position/cache state plain
// decode would use — the verify pass's logits decide ONLY the accept
// length, never emission or final cache content (see the FULL-ACCEPT
// TRUST section above).
//
// ─── Env gates ──────────────────────────────────────────────────────
//
//   BUTTER_NEMOTRON_MTP=1        — required to load the MTP block at
//                                   all (see `NemotronHMTPHead.swift`).
//   BUTTER_NEMOTRON_MTP_SPEC=1   — required to actually USE it for
//                                   decode (default OFF — this file's
//                                   own measured-loss result; see the
//                                   header above). Loading and driving
//                                   are separate gates so `butter bench`
//                                   can load the block without paying
//                                   the decode-path cost, for future
//                                   perplexity/quality checks on the
//                                   drafter alone.
//   BUTTER_NEMOTRON_MTP_GAMMA=<n> — draft chain length, default 2.

import Foundation
import Metal

/// Stats for one `NemotronHMTPSpecDecode.generateGreedy` run.
public struct NemotronHMTPSpecDecodeStats {
    public let tokensGenerated: Int
    /// Batched verify passes run (incl. the priming pass, which
    /// proposes zero candidates).
    public let verifySteps: Int
    public let candidatesProposed: Int
    public let candidatesAccepted: Int
    /// Single-token backbone forwards run to replay the committed
    /// prefix — ALWAYS runs (see the file header's FULL-ACCEPT TRUST
    /// section), so this is >= `tokensGenerated`, not just the
    /// partial-accept correction cost `MTPSpecDecodeStats.
    /// replaySingleSteps` tracks for Qwen3x.
    public let replaySingleSteps: Int
    /// Draft-head passes spent advancing the MTP KV cache over
    /// committed tokens (excludes draft-burst steps).
    public let lockstepSteps: Int
    public let wallclockSeconds: Double

    public var acceptanceRate: Double {
        guard candidatesProposed > 0 else { return 0 }
        return Double(candidatesAccepted) / Double(candidatesProposed)
    }
    public var tps: Double {
        guard wallclockSeconds > 0 else { return 0 }
        return Double(tokensGenerated) / wallclockSeconds
    }
}

public enum NemotronHMTPSpecDecode {

    /// Run the greedy MTP self-speculative decode loop — argmax
    /// everywhere, deterministic, byte-identical to plain greedy decode
    /// (see the file header).
    ///
    /// Caller contract matches `SpecDecode.generateGreedyMTP`: `caches`
    /// already reflect the prompt, `lastToken` is the most recent
    /// (already-sampled, not yet cache-committed) token, `position` is
    /// where `lastToken` lands on the next decode step.
    ///
    /// - `head`: the checkpoint's MTP draft head (`model.mtpHead`). Its
    ///   KV cache is truncated/advanced by this loop; callers starting
    ///   a fresh generation should `head.resetCache()` first.
    /// - `lastHidden`: backbone pre-final-norm hidden at `position - 1`.
    ///   Pass `nil` to let the loop prime itself (one draft-free cycle:
    ///   snapshot, a 1-token verify to capture the hidden, replay).
    /// - `emit`: called once per committed token, in stream order. Stop
    ///   tokens are NOT emitted.
    ///
    /// Never emits more than `maxNewTokens` tokens.
    public static func generateGreedy(
        model: NemotronHModel,
        head: NemotronHMTPHead,
        gamma: Int,
        lastToken: Int,
        position: Int,
        lastHidden: Tensor? = nil,
        caches: [any LayerCacheProtocol],
        history: inout [Int],
        maxNewTokens: Int,
        stopTokens: Set<Int> = [],
        device: Device = .shared,
        emit: ((Int) -> Void)? = nil
    ) -> NemotronHMTPSpecDecodeStats {
        precondition(gamma >= 1, "NemotronHMTPSpecDecode: gamma must be ≥ 1")
        precondition(maxNewTokens >= 0, "NemotronHMTPSpecDecode: maxNewTokens must be ≥ 0")

        var pos = position
        var prev = lastToken
        var prevHidden = lastHidden
        var tokensGenerated = 0
        var verifySteps = 0
        var candidatesProposed = 0
        var candidatesAccepted = 0
        var replaySingleSteps = 0
        var lockstepSteps = 0
        let t0 = Date()

        loop: while tokensGenerated < maxNewTokens {
            // Pre-burst MTP cache length — the truncation target for
            // the post-accept lockstep re-advance.
            let mtpLen = head.kvCache.length

            // ── Draft burst (skipped on the priming iteration) ────────
            var drafts: [Int] = []
            if let seedHidden = prevHidden {
                var chainHidden = seedHidden
                var chainTok = prev
                for j in 0 ..< gamma {
                    // `head.draftStep` commits+completes `dCmd` itself
                    // (its MoE sublayer's CPU router readback forces a
                    // commit partway through) — do NOT commit again here,
                    // that double-commits an already-submitted command
                    // buffer. See `NemotronHMTPHead.draftStep`'s doc.
                    let dCmd = device.makeCommandBuffer()
                    let (h, dLogits) = head.draftStep(
                        prevHidden: chainHidden, prevToken: chainTok,
                        position: pos + j, on: dCmd, device: device)
                    let c = Sampling.argmax(dLogits)
                    drafts.append(c)
                    chainHidden = h
                    chainTok = c
                }
                candidatesProposed += drafts.count
            }

            // ── Snapshot ALL target caches (incl. Mamba 2 recurrent
            //    state — see `CacheSnapshot.swift`'s `.mamba2` case)
            //    before the speculative verify pass ────────────────────
            let snap = caches.snapshotAll(device: device)

            // ── Verify: γ+1 sequential single-token forwards, ONE
            //    shared command buffer (see the file header — this is
            //    the ONLY lever available; no batched-GEMM kernel
            //    exists for this model) ──────────────────────────────
            let inputIds = [prev] + drafts
            let vCmd = device.makeCommandBuffer()
            var rowHiddens: [Tensor] = []
            var rowLogits: [Tensor] = []
            rowHiddens.reserveCapacity(inputIds.count)
            rowLogits.reserveCapacity(inputIds.count)
            for (i, tok) in inputIds.enumerated() {
                let (h, l) = model.forwardHiddenAndLogits(
                    tokenId: tok, position: pos + i, caches: caches,
                    on: vCmd, device: device)
                rowHiddens.append(h)
                rowLogits.append(l)
            }
            vCmd.commit()
            vCmd.waitUntilCompleted()
            verifySteps += 1

            // ── Accept plan — greedy longest-matching-prefix ──────────
            var verifierTokens: [Int] = []
            verifierTokens.reserveCapacity(drafts.count)
            for j in 0 ..< drafts.count {
                verifierTokens.append(Sampling.argmax(rowLogits[j]))
            }
            let batchedBonus = Sampling.argmax(rowLogits[inputIds.count - 1])
            let outcome = SpeculativeAccept.verify(
                draft: drafts, verifierTokens: verifierTokens, bonusToken: batchedBonus)
            let committed = [prev] + outcome.acceptedDraft
            candidatesAccepted += outcome.acceptedDraft.count

            // ── FULL-ACCEPT TRUST: not taken (see file header) — the
            //    verify pass's cache writes are discarded unconditionally
            //    and `committed` is replayed through the single-token
            //    path, one command buffer per token, matching
            //    `DFlashSpecDecode`'s conservative discipline. ─────────
            caches.restoreAll(from: snap, device: device)
            var nextTok = outcome.bonusToken
            // Per-row replay hidden, index-aligned with `committed` —
            // row `j` is the backbone hidden produced by feeding
            // `committed[j]`, i.e. exactly the "hidden at the PREVIOUS
            // position" the MTP lockstep re-advance below needs to seed
            // `committed[j+1]`'s draftStep call (same feed convention
            // `head.draftStep`'s doc comment describes).
            var replayHiddens: [Tensor] = []
            replayHiddens.reserveCapacity(committed.count)
            for (i, tok) in committed.enumerated() {
                let rCmd = device.makeCommandBuffer()
                let (h, l) = model.forwardHiddenAndLogits(
                    tokenId: tok, position: pos + i, caches: caches,
                    on: rCmd, device: device)
                rCmd.commit()
                rCmd.waitUntilCompleted()
                replayHiddens.append(h)
                replaySingleSteps += 1
                if i == committed.count - 1 {
                    // Re-derive the next seed via argmax on the exact
                    // replay-kernel logits — the baseline-matching
                    // exactness discipline (mirrors MTPSpecDecode /
                    // DFlashSpecDecode's own final-row re-derivation).
                    nextTok = Sampling.argmax(l)
                }
            }
            let lastReplayHidden = replayHiddens[replayHiddens.count - 1]

            // ── Commit + emit (stop tokens and budget respected) ──────
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

            // ── MTP KV lockstep over the committed block ──────────────
            // Roll back the burst's draft rows, then advance the head
            // once per committed token using the hidden the REPLAY pass
            // produced (never the discarded verify pass's) — the head's
            // own causal history must mirror the backbone's actual
            // (replayed) trajectory. `committed[0] == prev` pairs with
            // `prevHidden` (last cycle's carry — the hidden at pos - 1);
            // `committed[j >= 1]` pairs with `replayHiddens[j - 1]` (the
            // hidden produced by feeding `committed[j - 1]`, i.e. the
            // backbone hidden at position `pos + j - 1`).
            head.kvCache.truncate(toLength: mtpLen)
            for (j, tok) in committed.enumerated() {
                guard let ph = (j == 0) ? prevHidden : replayHiddens[j - 1] else {
                    // Priming iteration only: no hidden exists for
                    // `prev`'s position — the head's history skips this
                    // one row (matches MTPSpecDecode's identical note).
                    continue
                }
                // `draftStep` commits `lCmd` itself — see its doc.
                let lCmd = device.makeCommandBuffer()
                _ = head.draftStep(
                    prevHidden: ph, prevToken: tok, position: pos + j,
                    on: lCmd, device: device)
                lockstepSteps += 1
            }

            pos += committed.count
            prev = nextTok
            prevHidden = lastReplayHidden
        }

        return NemotronHMTPSpecDecodeStats(
            tokensGenerated: tokensGenerated, verifySteps: verifySteps,
            candidatesProposed: candidatesProposed, candidatesAccepted: candidatesAccepted,
            replaySingleSteps: replaySingleSteps, lockstepSteps: lockstepSteps,
            wallclockSeconds: Date().timeIntervalSince(t0))
    }
}
