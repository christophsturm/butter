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
// GpuTokenLoop: GPU-resident decode token feedback loop.
//
// The old greedy-argmax decode step (`LanguageModel.forwardSample`) reads
// the chosen token id back to the host (`cmd.commit()` +
// `waitUntilCompleted()`) before the *next* step can even start encoding,
// because the next step's embedding gather needs that token id as an
// input. That host round trip serializes every decode step behind a
// 4-byte readback that has nothing to do with the next step's actual GPU
// work.
//
// The fix: `LanguageModel.forward(tokenIdBuffer:...)` (implemented by
// families that set `supportsGPUTokenFeedback`) takes the token id as a
// device-resident `.u32` `Tensor` instead of a host `Int`. Feed it the
// *previous* step's `Ops.argmax` output buffer directly and the
// embedding gather chains on the GPU, same command queue, so ordering
// is already correct without a host wait (see `DeepSeekV4Text.swift`'s
// `forwardAllLayers` seed-copy comment: "No wait: ... reads ... on the
// same queue, which serializes after"). The host only needs the actual
// token value for two things: detokenization and the stop-token check,
// both read back **asynchronously** via `MTLCommandBuffer
// .addCompletedHandler` (registered before `commit()`, per the idiom in
// `LagunaLmHeadPrune.swift` / `DeepSeekV4Text.commitWithProfile`).
//
// `GpuTokenLoopSchedule` below owns the ring-buffer slot math and the
// lag-bounded lookahead: at most `lag` steps may be committed ahead of
// the last host-confirmed-safe (non-stop) step. Default `lag` is 1:
// step N+1 is in flight while step N's readback is still pending, but
// step N+2 doesn't launch until step N has resolved and been confirmed
// not to be a stop token. This keeps stop semantics identical to the
// old synchronous loop (the stop token and everything after it is never
// emitted); the only change is that up to `lag` steps of GPU work past
// an eventual stop token get computed speculatively and discarded
// (cheap: KV caches are per-`driveGeneration`-call and thrown away
// with the rest of the decode state, so the extra writes are inert).
//
// `GpuTokenLoop.run` is pure scheduling logic (no Metal), so it's
// unit-testable directly, see `Tests/ButterTests/Generation/
// GpuTokenLoopTests.swift`. `GpuTokenLoop.runGPU` is the real
// Metal-backed decode loop `Generate.swift` calls.

import Foundation
import Metal

/// Ring-buffer slot bookkeeping for the GPU token loop, kept separate
/// from Metal calls so the wraparound / lag-window math is directly
/// unit-testable.
public struct GpuTokenLoopSchedule {
    /// Number of persistent ring-buffer slots. Must be strictly greater
    /// than `lag`: a slot is only ever relaunched into once its
    /// previous occupant has definitely been drained, and the lag
    /// bound is what guarantees that (see `slot(forToken:)` doc).
    public let ringSize: Int
    /// Max steps allowed in flight (committed, unresolved) ahead of the
    /// last drained/confirmed-not-a-stop-token step. Default 1 in
    /// `GpuTokenLoop.runGPU`; see file header.
    public let lag: Int

    public init(ringSize: Int, lag: Int) {
        precondition(lag >= 0, "GpuTokenLoopSchedule: lag must be >= 0")
        precondition(
            ringSize > lag,
            "GpuTokenLoopSchedule: ringSize (\(ringSize)) must exceed lag (\(lag)) "
                + "or a slot could be reused before its previous occupant drains")
        self.ringSize = ringSize
        self.lag = lag
    }

    /// Ring slot holding token index `tokenIndex`'s value. Token index
    /// 0 is the seed (the already-known token fed into the first
    /// chained step); indices 1... are produced by the GPU loop.
    ///
    /// Reuse safety: slot `s` is reused for token index `t + ringSize`
    /// only once token `t` has been launched, and this scheduler never
    /// launches token `t + ringSize` until `resolvedUpTo >=
    /// t + ringSize - lag`. Since `ringSize > lag`,
    /// `t + ringSize - lag > t`, so token `t` (an earlier index that
    /// occupies the same slot) is guaranteed already resolved (and
    /// therefore already drained/read) before its slot is written
    /// again.
    public func slot(forToken tokenIndex: Int) -> Int {
        ((tokenIndex % ringSize) + ringSize) % ringSize
    }
}

/// Pure (Metal-free) driver for the lag-bounded, ring-buffered decode
/// chain. See the file header for the overall design; this type only
/// sequences calls into the three closures the caller supplies; it
/// has no Metal or model dependency of its own, which is what makes it
/// directly unit-testable with a CPU-only harness.
public enum GpuTokenLoop {
    /// Runs the scheduling loop.
    ///
    /// - Parameters:
    ///   - remainingBudget: how many more tokens to produce beyond the
    ///     already-known seed token (token index 0). `0` is a no-op.
    ///   - schedule: ring/lag bookkeeping.
    ///   - launch: encode + commit the GPU step that produces token
    ///     `tokenIndex` (1-based) from the value in ring slot
    ///     `schedule.slot(forToken: tokenIndex - 1)`, writing its
    ///     result into `schedule.slot(forToken: tokenIndex)` and
    ///     arranging for that slot's value to become available to
    ///     `resolve` once the GPU work completes. Must not block.
    ///     Called with strictly increasing `tokenIndex`, and never more
    ///     than `schedule.lag + 1` ahead of the most recently resolved
    ///     index.
    ///   - resolve: blocks (if necessary) until token `tokenIndex`'s
    ///     value is known, then returns it. Called with strictly
    ///     increasing `tokenIndex`, always for a token that has
    ///     already been `launch`ed.
    ///   - isStop: predicate on a resolved token id.
    ///   - onResolved: called once per resolved **non-stop** token, in
    ///     order, with its 1-based index and value. Not called for a
    ///     token where `isStop` returns `true`, nor for any token after
    ///     it (those may still be `launch`ed speculatively, see the
    ///     lag note in the file header, but are discarded, never
    ///     passed here).
    /// - Returns: `true` if generation stopped early on a stop token
    ///   before exhausting `remainingBudget`, `false` if it ran the
    ///   full budget.
    @discardableResult
    public static func run(
        remainingBudget: Int,
        schedule: GpuTokenLoopSchedule,
        launch: (_ tokenIndex: Int) -> Void,
        resolve: (_ tokenIndex: Int) -> UInt32,
        isStop: (_ value: UInt32) -> Bool,
        onResolved: (_ tokenIndex: Int, _ value: UInt32) throws -> Void
    ) rethrows -> Bool {
        guard remainingBudget > 0 else { return false }
        var launchedUpTo = 0
        var resolvedUpTo = 0
        var stopped = false
        while resolvedUpTo < remainingBudget && !stopped {
            // Keep up to `lag + 1` steps in flight ahead of the last
            // confirmed-safe token: this is the overlap, the readback
            // for the oldest in-flight step (below) happens while these
            // newer steps are still encoding/executing on the GPU.
            while launchedUpTo < remainingBudget,
                (launchedUpTo - resolvedUpTo) < (schedule.lag + 1)
            {
                launchedUpTo += 1
                launch(launchedUpTo)
            }
            let waitingFor = resolvedUpTo + 1
            let value = resolve(waitingFor)
            resolvedUpTo += 1
            if isStop(value) {
                stopped = true
            } else {
                try onResolved(waitingFor, value)
            }
        }
        return stopped
    }
}

// MARK: - Metal-backed decode loop

extension GpuTokenLoop {
    /// Real Metal-backed decode chain for engines with
    /// `supportsGPUTokenFeedback == true`. Produces up to
    /// `remainingBudget` more tokens starting from `seedToken` (already
    /// resolved/emitted by the caller) at `startPosition`, the
    /// position the *first* chained step's `forward(tokenIdBuffer:...)`
    /// call runs at.
    ///
    /// `onToken` is called once per resolved, non-stop token id, in
    /// order; the caller does the same per-token bookkeeping
    /// (detokenize, yield a `GenerationChunk`, append to history, etc.)
    /// it would otherwise do inline in its decode loop.
    ///
    /// - Returns: `true` if a stop token was hit before `remainingBudget`
    ///   tokens were produced.
    @discardableResult
    public static func runGPU(
        engine: any LanguageModel,
        seedToken: Int,
        startPosition: Int,
        caches: [any LayerCacheProtocol],
        remainingBudget: Int,
        stopSet: Set<Int>,
        device: Device,
        lag: Int = 1,
        onToken: (Int) throws -> Void
    ) rethrows -> Bool {
        guard remainingBudget > 0 else { return false }
        precondition(
            engine.supportsGPUTokenFeedback,
            "GpuTokenLoop.runGPU: engine does not support GPU token feedback; "
                + "check engine.supportsGPUTokenFeedback before calling")

        let ringSize = max(lag + 3, 4)
        let schedule = GpuTokenLoopSchedule(ringSize: ringSize, lag: lag)

        // Persistent ring of single-element u32 buffers, allocated once
        // up front (not per-step, and not from the per-token scratch
        // slab: these must stay valid across command-buffer
        // completions, same reasoning as the dedicated countBuf in
        // LagunaLmHeadPrune's diag path). Slot i's buffer is dual
        // purpose: it's the `Ops.argmax` *output* of the step that
        // produces token i, and the embedding-gather *input* of the
        // step that produces token i+1, fed straight back in with no
        // copy and no host touch.
        let ringBuffers: [Tensor] = (0 ..< ringSize).map { _ in
            Tensor(
                buffer: device.mtlDevice.makeBuffer(length: 4, options: .storageModeShared)!,
                offset: 0, shape: [1], dtype: .u32)
        }
        // Seed slot 0 with the already-known token: one host memcpy,
        // unavoidable (this value was already a CPU Int by the time we
        // get here) and done exactly once per `runGPU` call, not once
        // per decode step.
        var seed = UInt32(seedToken)
        withUnsafeBytes(of: &seed) { raw in
            ringBuffers[0].buffer.contents().copyMemory(
                from: raw.baseAddress!, byteCount: 4)
        }

        // One semaphore per ring slot: the completion handler for the
        // step that *writes* slot s signals it; `resolve` waits on it
        // before reading. This is what makes the async readback safe:
        // by the time `resolve` reads `ringBuffers[slot]`, the
        // completion handler (which only fires after the GPU write is
        // done) has already run.
        let resolvedSem = (0 ..< ringSize).map { _ in DispatchSemaphore(value: 0) }

        // `BUTTER_LAGUNA_DECODE_PROBE=N` — every Nth chained token runs as
        // a forced-sync, phase-instrumented token through the T39
        // `LagunaDecodeKernelTiming` checkpoint machinery, while every
        // other token keeps the normal pipelined path at full speed. This
        // is the decode analog of the prefill phase probes, which the
        // default GPU token loop bypasses entirely: sampling 1-in-N keeps
        // the run representative (steady-state pipelined
        // decode between probes) instead of collapsing the whole run to
        // forced-sync like BUTTER_LAGUNA_DECODE_KERNEL_TIMING=1 does.
        //
        // Mechanics per probe token:
        //   1. drain: wait on the previous token's outer CB — every CB of
        //      every earlier token was committed (single caller thread)
        //      before this launch and the queue is FIFO, so this drains
        //      ALL in-flight work. The probe token therefore soaks no
        //      prior-phase tail (the §6.4 drain-contamination trap is
        //      handled by construction, not by discarding first buckets).
        //   2. flip `BUTTER_LAGUNA_DECODE_KERNEL_TIMING` on via setenv for
        //      exactly this token — `LagunaDecodeKernelTiming.enabled` is
        //      a LIVE getenv read, so the per-layer attn/moe/embed
        //      checkpoints inside `forwardFromTokenTensor` fire for this
        //      token only; the final norm + lm_head + argmax on the outer
        //      CB is recorded here as `lmhead_argmax_chain`.
        //   3. restore the env and resume the pipelined loop.
        //
        // Zero-cost off: when the var is unset/0 the per-launch cost is
        // one Int compare. No-op when full-run KERNEL_TIMING=1 is already
        // on (that mode owns the flag). setenv mid-run is a diagnostics-
        // only concession (theoretical race with concurrent getenv on
        // other threads); never on unless the probe var is exported.
        let probeEveryN = max(0, Int(ButterEnv["BUTTER_LAGUNA_DECODE_PROBE"] ?? "") ?? 0)
        let fullRunTimingOn = LagunaDecodeKernelTiming.enabled
        var lastOuterCmd: MTLCommandBuffer?

        func launch(_ tokenIndex: Int) {
            let inputSlot = schedule.slot(forToken: tokenIndex - 1)
            let outputSlot = schedule.slot(forToken: tokenIndex)
            let position = startPosition + tokenIndex - 1
            let probing =
                probeEveryN > 0 && !fullRunTimingOn
                && tokenIndex % probeEveryN == 0
            if probing {
                lastOuterCmd?.waitUntilCompleted()
                setenv("BUTTER_LAGUNA_DECODE_KERNEL_TIMING", "1", 1)
                LagunaDecodeKernelTiming.beginToken()
            }
            let cmd = device.makeCommandBuffer()
            let logits = engine.forward(
                tokenIdBuffer: ringBuffers[inputSlot], position: position,
                caches: caches, on: cmd, device: device)
            Ops.argmax(logits, into: ringBuffers[outputSlot], on: cmd)
            // Register BEFORE commit: commit-then-addCompletedHandler
            // can miss the callback if the buffer finishes first.
            let sem = resolvedSem[outputSlot]
            cmd.addCompletedHandler { _ in sem.signal() }
            if probing {
                // Commits + waits `cmd` internally and records the bucket;
                // the completed handler above still fires, so `resolve`
                // is satisfied exactly as on the normal path.
                _ = LagunaDecodeKernelTiming.checkpoint(
                    "lmhead_argmax_chain", cmd: cmd, device: device)
                LagunaDecodeKernelTiming.endToken()
                unsetenv("BUTTER_LAGUNA_DECODE_KERNEL_TIMING")
                lastOuterCmd = cmd
                return
            }
            lastOuterCmd = cmd
            cmd.commit()
            // BUTTER_DEBUG_LOGIT_MARGIN: GPU path, chained-loop arm — the
            // default decode path for engines with GPU token feedback
            // (e.g. Laguna). This loop is normally fully async (the
            // whole point is never waiting on a single step here); when
            // the flag is on we force a synchronous readback of this
            // step's logits + chosen id so the margin can be reported in
            // strict emission order. Debug-only cost, and it collapses
            // the lag-pipelined schedule to effectively synchronous for
            // the run — expected, this is not a perf-preserving flag.
            if LogitMarginDebug.enabled {
                cmd.waitUntilCompleted()
                let outBuf = ringBuffers[outputSlot]
                let chosenId = Int(
                    outBuf.buffer.contents()
                        .advanced(by: outBuf.offset)
                        .assumingMemoryBound(to: UInt32.self).pointee)
                LogitMarginDebug.report(logits: logits, chosenId: chosenId)
            }
        }

        func resolve(_ tokenIndex: Int) -> UInt32 {
            let slot = schedule.slot(forToken: tokenIndex)
            resolvedSem[slot].wait()
            let buf = ringBuffers[slot]
            return buf.buffer.contents()
                .advanced(by: buf.offset)
                .assumingMemoryBound(to: UInt32.self).pointee
        }

        return try GpuTokenLoop.run(
            remainingBudget: remainingBudget,
            schedule: schedule,
            launch: launch,
            resolve: resolve,
            isStop: { stopSet.contains(Int($0)) },
            onResolved: { _, value in try onToken(Int(value)) }
        )
    }
}
