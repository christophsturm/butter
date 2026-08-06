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
// MTPSpecDecode — the MTP self-speculative token iterator for
// Qwen 3.5 / 3.6 (spec 030 phase 2; `MTPSelfSpeculativeTokenIterator`
// in planning/architecture.md).
//
// Same driver shape as `SpecDecode.generateGreedy` (the n-gram
// draft-verify loop), with the external `Drafter` replaced by the
// checkpoint's own single-layer MTP head (`Qwen3xMTPHead`):
//
//   1. Chain-draft γ candidate tokens with `mtpHead.draftStep`: the
//      first step fuses the backbone's last pre-final-norm hidden with
//      the last committed/sampled token; each subsequent step chains
//      the head's OWN output hidden h' with its previous draft (the
//      DeepSeek-nextn recipe; `mtp_num_hidden_layers = 1`, so the one
//      layer is reused every step). Host commit between steps per the
//      head's threading contract.
//   2. Verify with ONE batched backbone pass —
//      `forwardManyHiddenAndLogits([prev] + drafts)` — which also
//      returns the per-position hidden states the next draft burst and
//      the head's KV lockstep need.
//   3. Accept via `SpeculativeAccept.verify` (longest matching prefix).
//   4. EXACTNESS DISCIPLINE (inherited verbatim from SpecDecode): the
//      batched verify logits are used ONLY for accept-length decisions.
//      On a partial/zero accept the emitted next token comes from
//      REPLAYING the accepted prefix through the single-token forward
//      path after a full cache restore — batched kernels can drift vs
//      decode kernels (SpecDecode.swift ~168-176). On a full accept the
//      bonus token comes from the trailing batched logits row, matching
//      `SpecDecode.generateGreedy`'s existing bonus handling.
//   5. MTP KV LOCKSTEP: the draft head runs as a normal decoder layer
//      with a persistent KV cache mirroring the backbone's committed
//      history (see Qwen3xMTPHead's header). After each iteration the
//      burst's draft rows are rolled back (`kvCache.truncate`) and the
//      head is re-advanced over every committed token using the hidden
//      states the verify pass returned — one draftStep-shaped pass per
//      committed token, logits discarded (the simple per-token version;
//      BUTTER_QWEN_MTP_BATCHED_LOCKSTEP=1 batches this into ONE
//      T-batched `Qwen3xMTPHead.draftStepMany` pass instead, off by
//      default).
//
// PRIMING: the loop needs the backbone hidden at `position - 1` to
// seed the first draft burst. When the caller can't supply it (the
// standard `Generate.swift` prefill path only returns logits), the
// first iteration runs draft-free: snapshot, a 1-token
// `forwardManyHiddenAndLogits([prev])` to capture the hidden, then the
// standard restore + single-token replay so the emitted token stays on
// the exact decode path. The head's KV history simply misses the
// pre-priming rows (prompt + the priming position) — positions are
// passed through unchanged so later rows still line up; this only
// (mildly) affects draft quality, never correctness.
//
// Greedy-only, like `generateGreedy` — argmax everywhere.

import Foundation
import Metal

/// Stats for one `generateGreedyMTP` run. Mirrors `SpecDecodeStats`
/// plus the MTP-specific replay / lockstep costs.
public struct MTPSpecDecodeStats {
    public let tokensGenerated: Int
    /// Batched verify passes run (incl. the priming pass, which
    /// proposes zero candidates).
    public let verifySteps: Int
    public let candidatesProposed: Int
    public let candidatesAccepted: Int
    /// Single-token backbone forwards run to replay accepted prefixes
    /// on partial/zero accepts (the exactness discipline).
    public let replaySingleSteps: Int
    /// Draft-head passes spent advancing the MTP KV cache over
    /// committed tokens (the lockstep cost; excludes draft-burst steps).
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

extension SpecDecode {

    /// Pure accept-arithmetic for one MTP spec iteration — split out of
    /// the loop so the commit/replay bookkeeping is unit-testable
    /// without a model. `drafts` may be empty (the priming iteration).
    struct MTPAcceptPlan: Equatable {
        /// Tokens committed this iteration, in order:
        /// `[prev] + acceptedDraftPrefix`.
        let committed: [Int]
        /// Next iteration's seed token. From the batched bonus row on a
        /// full accept; from the single-token replay otherwise (the
        /// value here is the *accept-time* candidate — the replay path
        /// recomputes it on the exact decode path before use).
        let nextToken: Int
        /// True on a partial/zero accept: restore every cache and
        /// replay `committed` through the single-token forward path.
        let needsReplay: Bool
        /// Row of the verify pass's `[T, hidden]` output that feeds the
        /// next draft burst (the hidden at the last committed position).
        let nextHiddenRow: Int
    }

    /// Compute the commit plan from the draft block and the verifier's
    /// own (argmax) tokens. `verifierTokens[j]` verifies `drafts[j]`;
    /// `batchedBonus` is the verifier's token one past the draft block.
    static func planMTPAccept(
        prev: Int, drafts: [Int], verifierTokens: [Int], batchedBonus: Int
    ) -> MTPAcceptPlan {
        if drafts.isEmpty {
            // Priming / draft-free iteration: commit `prev` alone and
            // force the replay path (nothing verified backs the batched
            // pass, so the emitted next token must come from the exact
            // single-token forward).
            return MTPAcceptPlan(
                committed: [prev], nextToken: batchedBonus,
                needsReplay: true, nextHiddenRow: 0)
        }
        let outcome = SpeculativeAccept.verify(
            draft: drafts, verifierTokens: verifierTokens,
            bonusToken: batchedBonus)
        let k = outcome.acceptedDraft.count
        let committed = [prev] + outcome.acceptedDraft
        // On a FULL accept the next token is `batchedBonus`, taken from
        // the batched verify logits and emitted without ever passing
        // through the single-token decode path. If the two kernels
        // disagree on an argmax tie the stream diverges from baseline
        // greedy. BUTTER_QWEN_MTP_ALWAYS_REPLAY=1 forces the replay path
        // on every iteration to isolate that as the divergence source.
        let forceReplay =
            ProcessInfo.processInfo.environment["BUTTER_QWEN_MTP_ALWAYS_REPLAY"] == "1"
        return MTPAcceptPlan(
            committed: committed,
            nextToken: outcome.bonusToken,
            needsReplay: forceReplay || k < drafts.count,
            nextHiddenRow: committed.count - 1)
    }

    /// Run the greedy MTP self-speculative decode loop. Caller
    /// contract matches `generateGreedy`: `caches` already reflect the
    /// prompt, `lastToken` is the most recent (already-sampled, not yet
    /// cache-committed) token, `position` is where `lastToken` lands on
    /// the next decode step.
    ///
    /// - `head`: the checkpoint's MTP draft head (`model.mtpHead`). Its
    ///   KV cache is truncated/advanced by this loop; callers starting
    ///   a fresh generation should `head.resetCache()` first.
    /// - `lastHidden`: backbone pre-final-norm hidden at `position - 1`
    ///   (e.g. the last row of a `forwardManyHiddenAndLogits` prefill).
    ///   Pass `nil` to let the loop prime itself (one extra batched +
    ///   single-token pass; see the file header).
    /// - `emit`: called once per committed token, in stream order —
    ///   the exact tokens the baseline greedy loop would emit. Stop
    ///   tokens are NOT emitted (matching `Generate.driveGeneration`).
    ///
    /// Never emits more than `maxNewTokens` tokens (a full-accept block
    /// is trimmed at the budget, unlike `generateGreedy`'s overshoot).
    public static func generateGreedyMTP(
        model: Qwen35Model,
        head: Qwen3xMTPHead,
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
    ) -> MTPSpecDecodeStats {
        precondition(gamma >= 1, "SpecDecode.generateGreedyMTP: gamma must be ≥ 1")
        precondition(
            maxNewTokens >= 0,
            "SpecDecode.generateGreedyMTP: maxNewTokens must be ≥ 0")

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
        // BUTTER_QWEN_MTP_PROF=1 attribution buckets (seconds).
        var msDraft = 0.0, msVerify = 0.0, msReplay = 0.0
        var msLockstep = 0.0, msSnap = 0.0
        let stateRewind =
            ProcessInfo.processInfo.environment["BUTTER_QWEN_MTP_STATE_REWIND"] == "1"
        var rewindSteps = 0
        let batchedReplay =
            ProcessInfo.processInfo.environment["BUTTER_QWEN_MTP_BATCHED_REPLAY"] == "1"
        // BUTTER_QWEN_MTP_BATCHED_LOCKSTEP=1: run the post-accept MTP
        // KV re-advance (below) as ONE T-batched `draftStepMany` pass
        // instead of k+1 sequential `draftStep` calls. Off by default;
        // read fresh on every call (no hoisted/cached global) so a
        // per-process env change always takes effect.
        let batchedLockstep =
            ProcessInfo.processInfo.environment["BUTTER_QWEN_MTP_BATCHED_LOCKSTEP"] == "1"
        let parityProbe =
            ProcessInfo.processInfo.environment["BUTTER_QWEN_MTP_PARITY"] == "1"
        var parityChecked = 0
        var parityMismatch = 0
        var parityMismatchMargin = 0.0
        var parityMatchMargin = 0.0
        var parityRowChecked: [Int] = []
        var parityRowMismatch: [Int] = []
        // BUTTER_QWEN_MTP_PMIN=<float>: confidence-gated dynamic draft
        // length. Unset/empty ⇒ feature OFF, behavior byte-identical to
        // the fixed-γ loop (every burst drafts exactly `gamma` tokens).
        // When set, `gamma` doubles as the burst's n_max: the burst
        // still stops at `gamma` tokens, but may stop EARLIER once the
        // drafter's own top-token probability drops below `pMin` — read
        // fresh every call, no static caching, matching the other
        // env-gated knobs above.
        let pMinEnv = ProcessInfo.processInfo.environment["BUTTER_QWEN_MTP_PMIN"]
        let pMin: Float? = (pMinEnv?.isEmpty == false) ? Float(pMinEnv!) : nil
        // avgdraft observability (BUTTER_QWEN_MTP_PROF=1 line below):
        // draftCycles counts bursts actually run (excludes the priming
        // iteration, which has no seed hidden and drafts nothing).
        var draftCycles = 0
        var minDraftLen = Int.max
        var maxDraftLen = 0

        if stateRewind { model.setSpecCapture(true, caches: caches) }
        defer { if stateRewind { model.setSpecCapture(false, caches: caches) } }
        loop: while tokensGenerated < maxNewTokens {
            // Pre-burst MTP cache length — the truncation target for
            // the post-accept lockstep (every burst row is rolled back
            // before the canonical re-advance). MUST be read before
            // the burst appends its rows.
            let mtpLen = head.kvCache.length

            // ── Draft burst (skipped on the priming iteration) ────────
            // Chain up to γ tokens through the single MTP layer. Each
            // step commits its own command buffer (the head's mixer
            // reuses per-instance scratch — see its threading contract)
            // and reads one argmax back to feed the chain.
            var drafts: [Int] = []
            if let seedHidden = prevHidden {
                var chainHidden = seedHidden
                var chainTok = prev
                for j in 0 ..< gamma {
                    let dT = Date()
                    let dCmd = device.makeCommandBuffer()
                    let (h, dLogits) = head.draftStep(
                        prevHidden: chainHidden, prevToken: chainTok,
                        position: pos + j, on: dCmd, device: device)
                    dCmd.commit()
                    dCmd.waitUntilCompleted()
                    // Single CPU readback of this step's logits row,
                    // shared by both the argmax (always) and the
                    // confidence check (only when BUTTER_QWEN_MTP_PMIN
                    // is set) — no extra GPU work either way.
                    let flatD = dLogits.toFloatArray()
                    let c: Int
                    var pTop: Float = 1
                    if pMin != nil {
                        let (idx, top) = argmaxAndTopProb(
                            flatD, offset: 0, count: flatD.count)
                        c = idx
                        pTop = top
                    } else {
                        c = argmax(flatD, offset: 0, count: flatD.count)
                    }
                    drafts.append(c)
                    chainHidden = h
                    chainTok = c
                    msDraft += Date().timeIntervalSince(dT)
                    // Confidence gate: stop the burst once the drafter's
                    // own top-token probability drops below pMin. Always
                    // draft at least 1 token first — the seed hidden's
                    // (pre-burst) confidence isn't available without an
                    // extra head pass, so the earliest possible stop is
                    // right after a draft, never before one.
                    //
                    // Floor is 2, not 1: `Ops.dequantGemmDynamicM`'s
                    // default-on small-M fast path (BUTTER_QMM_SMALL_M)
                    // routes verify batches of T=2 (drafts.count==1)
                    // through its BM=2 kernel, which has a pre-existing
                    // correctness bug — confirmed independent of this
                    // feature by reproducing the exact same corrupted
                    // digest via BUTTER_QWEN_MTP_GAMMA=1 (T=2, every
                    // cycle) on the unmodified tree, and confirming
                    // BUTTER_QMM_SMALL_M=0 fixes it. T=3/4 (BM=4) and
                    // T=8 verified clean. Rather than disabling the fast
                    // path for every pMin run (which would cost real
                    // throughput on the T≥3 cycles too, and stop this
                    // arm from being an apples-to-apples comparison
                    // against the fixed-γ baseline), the gate simply
                    // never stops at drafts.count == 1 — costing at most
                    // one extra (cheap, single-layer) draft step per
                    // early-stopped cycle while the underlying kernel
                    // bug gets its own fix.
                    if let pMin, drafts.count >= 2, pTop < pMin { break }
                }
                candidatesProposed += drafts.count
                draftCycles += 1
                minDraftLen = min(minDraftLen, drafts.count)
                maxDraftLen = max(maxDraftLen, drafts.count)
                // BUTTER_QWEN_MTP_TRACE=1: first few cycles only. Tells
                // apart a numerically broken head (constant/degenerate
                // draft ids) from a misaligned-but-live one (varied,
                // confident ids that simply do not match the verifier).
                if ProcessInfo.processInfo.environment["BUTTER_QWEN_MTP_TRACE"] == "1",
                    verifySteps < 6
                {
                    FileHandle.standardError.write(
                        "[mtp-trace] cycle=\(verifySteps) prev=\(prev) "
                            .data(using: .utf8)!)
                    FileHandle.standardError.write(
                        "drafts=\(drafts)\n".data(using: .utf8)!)
                }
            }

            // ── Snapshot ALL caches before the speculative forward ────
            // Backbone layer caches via the composite snapshot; the MTP
            // head's own KV cache alongside (`mtpLen` above — same
            // length-only rollback semantics as the composite's KVCache
            // case). Taking the backbone snapshot AFTER the draft burst
            // is safe: the burst only touches the head's own cache.
            let snapT = Date()
            let snap = caches.snapshotAll(device: device)
            msSnap += Date().timeIntervalSince(snapT)

            // ── Verify: ONE batched backbone pass ─────────────────────
            let inputIds = [prev] + drafts
            let vT = Date()
            let vCmd = device.makeCommandBuffer()
            var (hiddenAll, logitsAll) = model.forwardManyHiddenAndLogits(
                tokenIds: inputIds, startPosition: pos,
                caches: caches, on: vCmd, device: device)
            vCmd.commit()
            vCmd.waitUntilCompleted()
            msVerify += Date().timeIntervalSince(vT)
            verifySteps += 1

            let flat = logitsAll.toFloatArray()
            let rows = inputIds.count
            let vocab = flat.count / rows
            precondition(
                flat.count == rows * vocab,
                "generateGreedyMTP: logits have \(flat.count) elements; "
                    + "expected rows*vocab = \(rows * vocab)")

            // Verifier tokens: argmax per draft-aligned row. Row j
            // verifies drafts[j] (the token at position pos + 1 + j);
            // the trailing row is the full-accept bonus.
            var verifierTokens: [Int] = []
            verifierTokens.reserveCapacity(drafts.count)
            for j in 0 ..< drafts.count {
                verifierTokens.append(argmax(flat, offset: j * vocab, count: vocab))
            }
            let batchedBonus = argmax(
                flat, offset: (rows - 1) * vocab, count: vocab)

            // ── Accept plan (longest matching prefix) ─────────────────
            if ProcessInfo.processInfo.environment["BUTTER_QWEN_MTP_TRACE"] == "1",
                verifySteps <= 6
            {
                FileHandle.standardError.write(
                    "[mtp-trace] verifier=\(verifierTokens) bonus=\(batchedBonus)\n"
                        .data(using: .utf8)!)
            }
            let plan = planMTPAccept(
                prev: prev, drafts: drafts,
                verifierTokens: verifierTokens, batchedBonus: batchedBonus)
            candidatesAccepted += plan.committed.count - 1

            var nextTok = plan.nextToken
            var replayed = false
            if plan.needsReplay {
                // ── Partial/zero accept: restore + replay ─────────────
                // The batched pass's cache writes and logits are
                // discarded; the accepted prefix replays through the
                // single-token forward so the emitted stream (and the
                // cache contents backing it) stay bit-identical to the
                // baseline greedy loop. Next-prev comes from the LAST
                // single-step forward, not the batched row.
                let rT = Date()
                if stateRewind {
                    // ── Partial accept WITHOUT replay ─────────────────
                    // The batched pass is causal, so everything it
                    // produced for accepted rows is already correct: the
                    // KV entries, the hidden states, and `plan.nextToken`
                    // (the verifier row one past the accepted prefix).
                    // The ONLY thing wrong is the recurrent state, which
                    // advanced over the rejected tokens too.
                    //
                    // So restore just the recurrent state, re-run the
                    // conv + chunk scan over the accepted prefix (no
                    // weight traffic), and shorten KV to match. This
                    // replaces a full forward — all 14 GB of weights —
                    // with two small dispatches per GDN layer.
                    caches.restoreAll(
                        from: snap, device: device, recurrentOnly: true)
                    let wCmd = device.makeCommandBuffer()
                    let ok = model.rewindCaches(
                        keep: plan.committed.count, absolutePosition: pos,
                        caches: caches, on: wCmd, device: device)
                    wCmd.commit()
                    wCmd.waitUntilCompleted()
                    if ok {
                        msReplay += Date().timeIntervalSince(rT)
                        // nextTok stays plan.nextToken; hiddenAll stays
                        // the verify pass's rows.
                        rewindSteps += 1
                        replayed = true
                    } else {
                        // Shape not eligible — fall back to replaying,
                        // which needs the FULL snapshot restored first.
                        caches.restoreAll(from: snap, device: device)
                    }
                }
                if !replayed {
                caches.restoreAll(from: snap, device: device)
                if batchedReplay {
                    // The rejected rows polluted every cache, so the
                    // accepted prefix has to be re-run from the restored
                    // snapshot either way. Doing it as ONE batched pass
                    // costs a single T=k+1 forward instead of k+1
                    // separate T=1 forwards, and the weights are loaded
                    // once rather than k+1 times.
                    //
                    // This gives up the bit-exactness discipline: the
                    // next token now comes from batched logits on EVERY
                    // path, not just full accepts. That is the same
                    // contract llama.cpp and vLLM use (they never replay
                    // at all). BUTTER_QWEN_MTP_ALWAYS_REPLAY=1 with this
                    // flag unset remains the bit-exact mode.
                    let rCmd = device.makeCommandBuffer()
                    let (rHidden, rLogits) = model.forwardManyHiddenAndLogits(
                        tokenIds: plan.committed, startPosition: pos,
                        caches: caches, on: rCmd, device: device)
                    rCmd.commit()
                    rCmd.waitUntilCompleted()
                    replaySingleSteps += 1
                    let rf = rLogits.toFloatArray()
                    let rv = rf.count / plan.committed.count
                    nextTok = argmax(
                        rf, offset: (plan.committed.count - 1) * rv,
                        count: rv)
                    // Seed the next burst from THIS pass's hiddens: the
                    // verify pass's rows were computed before the
                    // restore and its trailing rows are backed by
                    // rejected tokens.
                    hiddenAll = rHidden
                } else {
                    var lastLogits: Tensor!
                    for (i, tok) in plan.committed.enumerated() {
                        let sCmd = device.makeCommandBuffer()
                        lastLogits = model.forward(
                            tokenId: tok, position: pos + i,
                            caches: caches, on: sCmd, device: device)
                        sCmd.commit()
                        sCmd.waitUntilCompleted()
                        replaySingleSteps += 1
                    }
                    nextTok = argmax(lastLogits)
                }
                msReplay += Date().timeIntervalSince(rT)
                }
            }
            // Full accept: backbone caches already hold the verified
            // block (all rows backed by accepted tokens); `nextTok` is
            // the batched bonus, per SpecDecode's existing handling.

            // ── Parity probe (diagnostic only) ────────────────────────
            // BUTTER_QWEN_MTP_PARITY=1 recomputes the bonus through the
            // single-token path on FULL accepts and reports how far the
            // batched logits drift from it. The emitted stream still
            // uses the fast path, so timings stay representative and
            // only the diagnostic cost is added.
            //
            // The margin is |top1 - top2| of the batched bonus row. A
            // mismatch at a tiny margin is an argmax tie-break between
            // near-equal logits; a mismatch at a wide margin means the
            // batched pass computed materially different logits, which
            // would also be silently costing acceptance on the verifier
            // rows.
            if parityProbe, !plan.needsReplay {
                let saved = caches.snapshotAll(device: device)
                caches.restoreAll(from: snap, device: device)
                var truthLogits: Tensor!
                for (i, tok) in plan.committed.enumerated() {
                    let pCmd = device.makeCommandBuffer()
                    truthLogits = model.forward(
                        tokenId: tok, position: pos + i,
                        caches: caches, on: pCmd, device: device)
                    pCmd.commit()
                    pCmd.waitUntilCompleted()
                    // On a full accept `committed == inputIds`, so replay
                    // step i is the ground truth for batched row i. Track
                    // per-row agreement to tell a last-row boundary bug
                    // apart from the batched pass being wrong throughout.
                    let rowTruth = argmax(truthLogits)
                    let rowBatched = argmax(
                        flat, offset: i * vocab, count: vocab)
                    while parityRowChecked.count <= i {
                        parityRowChecked.append(0)
                        parityRowMismatch.append(0)
                    }
                    parityRowChecked[i] += 1
                    if rowTruth != rowBatched { parityRowMismatch[i] += 1 }
                }
                let truth = argmax(truthLogits)
                parityChecked += 1
                let base = (rows - 1) * vocab
                var t1 = -Float.infinity
                var t2 = -Float.infinity
                for v in 0 ..< vocab {
                    let x = flat[base + v]
                    if x > t1 {
                        t2 = t1
                        t1 = x
                    } else if x > t2 {
                        t2 = x
                    }
                }
                let margin = t1 - t2
                if truth != batchedBonus {
                    parityMismatch += 1
                    parityMismatchMargin += Double(margin)
                    let msg =
                        "[mtp-parity] MISMATCH batched=\(batchedBonus) "
                        + "single=\(truth) margin=\(margin)\n"
                    FileHandle.standardError.write(msg.data(using: .utf8)!)
                } else {
                    parityMatchMargin += Double(margin)
                }
                caches.restoreAll(from: saved, device: device)
            }

            // ── Commit + emit (stop tokens and budget respected) ──────
            var stopped = false
            var emitted = 0
            for tok in plan.committed {
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
            if stopped || emitted < plan.committed.count { break loop }

            // ── MTP KV lockstep over the committed block ──────────────
            // Roll back the burst's draft rows (steps ≥ 1 were fed the
            // head's own h' chain, not backbone hiddens), then advance
            // the head once per committed token with the verify pass's
            // backbone hidden at the previous position — the canonical
            // lockstep feed. Logits are discarded; no sampling.
            head.kvCache.truncate(toLength: mtpLen)
            if batchedLockstep {
                // ── Batched re-advance — ONE T-batched head pass ──────
                // Only row 0 can be missing a hidden (the priming
                // iteration, where `prevHidden` is nil for `committed[0]
                // == prev`); when that happens the row is DROPPED from
                // the batch (never zero-filled — see the task note in
                // Qwen3xMTPHead.draftStepMany) and the remaining suffix,
                // which is always contiguous, is batched instead. Every
                // post-priming iteration has `prevHidden != nil`, so
                // `startIdx` is 0 from the second iteration on.
                let startIdx = (prevHidden == nil) ? 1 : 0
                if startIdx < plan.committed.count {
                    let rowsT = plan.committed.count - startIdx
                    let lT = Date()
                    // Assemble the `[rowsT, hidden]` hidden batch: row 0
                    // (when startIdx == 0) is `prevHidden` — a tensor
                    // from a DIFFERENT buffer than `hiddenAll` (last
                    // iteration's carry) — every other row is a
                    // `hiddenAll` view. Blit both sources into one
                    // contiguous buffer so the batched pass reads a
                    // single tensor.
                    let hiddenBytes = model.hidden * hiddenAll.dtype.byteSize
                    let hiddensFlat = Tensor.empty(
                        shape: [rowsT, model.hidden], dtype: hiddenAll.dtype,
                        device: device)
                    let lCmd = device.makeCommandBuffer()
                    for row in 0 ..< rowsT {
                        let j = startIdx + row
                        let src =
                            (j == 0)
                            ? prevHidden!
                            : hiddenRow(hiddenAll, j - 1, hidden: model.hidden)
                        let dst = Tensor(
                            buffer: hiddensFlat.buffer,
                            offset: hiddensFlat.offset + row * hiddenBytes,
                            shape: [model.hidden], dtype: hiddenAll.dtype)
                        Ops.copy(src, into: dst, on: lCmd)
                    }
                    let tokens = Array(plan.committed[startIdx...])
                    head.draftStepMany(
                        prevHiddensFlat: hiddensFlat.reshaped(
                            to: [rowsT * model.hidden]),
                        prevTokens: tokens, startPosition: pos + startIdx,
                        on: lCmd, device: device)
                    lCmd.commit()
                    lCmd.waitUntilCompleted()
                    msLockstep += Date().timeIntervalSince(lT)
                    lockstepSteps += 1
                }
            } else {
                for (j, tok) in plan.committed.enumerated() {
                    // committed[0] == prev at position pos pairs with the
                    // hidden at pos - 1 (last iteration's carry); committed
                    // [j ≥ 1] pairs with the verify hidden at pos + j - 1.
                    guard let ph = (j == 0) ? prevHidden : hiddenRow(hiddenAll, j - 1, hidden: model.hidden)
                    else {
                        // Priming iteration only: no hidden exists for
                        // `prev`'s position — the head's history skips this
                        // one row (see the file header).
                        continue
                    }
                    let lT = Date()
                    let lCmd = device.makeCommandBuffer()
                    _ = head.draftStep(
                        prevHidden: ph, prevToken: tok, position: pos + j,
                        on: lCmd, device: device)
                    lCmd.commit()
                    lCmd.waitUntilCompleted()
                    msLockstep += Date().timeIntervalSince(lT)
                    lockstepSteps += 1
                }
            }

            // ── Advance ───────────────────────────────────────────────
            // Next burst seeds from the verify hidden at the last
            // committed position. The verify pass blits `hiddenAll`
            // into a fresh tensor on its own command buffer, so the
            // row view stays valid across later forwards.
            prevHidden = hiddenRow(hiddenAll, plan.nextHiddenRow, hidden: model.hidden)
            pos += plan.committed.count
            prev = nextTok
            if stopTokens.contains(nextTok) { break loop }
        }
        // Like `generateGreedy`: `prev` is next-iteration input (its KV
        // slot may not be committed); it is NOT appended to `history`.

        if ProcessInfo.processInfo.environment["BUTTER_QWEN_MTP_PROF"] == "1" {
            let tot = Date().timeIntervalSince(t0)
            let avgDraft =
                draftCycles > 0
                ? Double(candidatesProposed) / Double(draftCycles) : 0.0
            let minD = draftCycles > 0 ? minDraftLen : 0
            let line = String(
                format:
                    "[mtp-prof] total=%.2fs draft=%.2f verify=%.2f replay=%.2f "
                    + "lockstep=%.2f snap=%.2f other=%.2f rewinds=\(rewindSteps) "
                    + "avgdraft=\(String(format: "%.2f", avgDraft)) "
                    + "mindraft=\(minD) maxdraft=\(maxDraftLen) draftcycles=\(draftCycles)\n",
                tot, msDraft, msVerify, msReplay, msLockstep, msSnap,
                tot - msDraft - msVerify - msReplay - msLockstep - msSnap)
            FileHandle.standardError.write(line.data(using: .utf8)!)
        }
        if parityProbe, parityChecked > 0 {
            let matched = parityChecked - parityMismatch
            let line = String(
                format:
                    "[mtp-parity] full-accepts=%d mismatch=%d (%.1f%%) "
                    + "avg-margin match=%.4f mismatch=%.4f\n",
                parityChecked, parityMismatch,
                100.0 * Double(parityMismatch) / Double(parityChecked),
                matched > 0 ? parityMatchMargin / Double(matched) : 0.0,
                parityMismatch > 0
                    ? parityMismatchMargin / Double(parityMismatch) : 0.0)
            FileHandle.standardError.write(line.data(using: .utf8)!)
            var rows = "[mtp-parity] per-row mismatch:"
            for i in 0 ..< parityRowChecked.count {
                rows += " row\(i)=\(parityRowMismatch[i])/\(parityRowChecked[i])"
            }
            FileHandle.standardError.write((rows + "\n").data(using: .utf8)!)
        }
        return MTPSpecDecodeStats(
            tokensGenerated: tokensGenerated,
            verifySteps: verifySteps,
            candidatesProposed: candidatesProposed,
            candidatesAccepted: candidatesAccepted,
            replaySingleSteps: replaySingleSteps,
            lockstepSteps: lockstepSteps,
            wallclockSeconds: Date().timeIntervalSince(t0))
    }
}

// ─── helpers ─────────────────────────────────────────────────────────

/// `[hidden]` view of one row of a `[T, hidden]` hidden-state tensor.
@inline(__always)
private func hiddenRow(_ hiddenAll: Tensor, _ row: Int, hidden: Int) -> Tensor {
    Tensor(
        buffer: hiddenAll.buffer,
        offset: hiddenAll.offset + row * hidden * hiddenAll.dtype.byteSize,
        shape: [hidden], dtype: hiddenAll.dtype)
}

@inline(__always)
private func argmax(_ logits: Tensor) -> Int {
    let host = logits.toFloatArray()
    return argmax(host, offset: 0, count: host.count)
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
/// same host readback (BUTTER_QWEN_MTP_PMIN's confidence gate). Two
/// passes over `count` floats: the first finds `bestIdx`/`bestVal`
/// (identical to `argmax` above), the second computes the numerically
/// stable normalizer `s = Σ exp(l_i - bestVal)`. Since
/// `exp(bestVal - bestVal) == 1`, the top probability is `1 / s` —
/// no separate `exp(l_max - logsumexp(l))` term needed.
@inline(__always)
private func argmaxAndTopProb(_ flat: [Float], offset: Int, count: Int) -> (
    Int, Float
) {
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
