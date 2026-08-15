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
// T39: per-kernel/phase-group GPU-time probe for one decode token.
// Prefill already has forced-commit+wait phase timing (T26
// BUTTER_LAGUNA_ATTN_PHASE_TIMING, T25/T27 BUTTER_LAGUNA_MOE_PHASE_TIMING);
// decode never had an equivalent, because the default decode path pipelines
// many small command buffers and forcing a commit+wait at every internal
// boundary changes the very timing being measured. This file is the decode
// sibling using the SAME forced-checkpoint technique as prefill, restricted
// to BUTTER_LAGUNA_DECODE_KERNEL_TIMING=1: zero behavior/perf change on the
// default hot path (every checkpoint call site is a no-op guarded by
// `enabled`, mirroring T26's `attnPhaseCheckpoint`).
//
// Each checkpoint commits the current in-flight command buffer, waits for
// completion, and records:
//   wallUs    - host round-trip (commit -> waitUntilCompleted return)
//   gpuUs     - MTLCommandBuffer.gpuEndTime - gpuStartTime (actual GPU-busy
//               time for the dispatches on that buffer; populated by the
//               driver once the buffer completes, no extra setup needed)
//   gapUs     - wallUs - gpuUs (submission/scheduling/host overhead not
//               explained by GPU execution - the mission's "host idle
//               bubble" candidate)
// then returns a fresh command buffer for the caller to keep encoding onto.
// Buckets accumulate across the whole run (all tokens, all layers) under a
// name-keyed table; `report()` divides by the token count recorded via
// `beginToken`/`endToken` to print mean us/token per bucket.
//
// Absolute numbers here are inflated vs. the real pipelined decode - same
// caveat T26/T31 already made for prefill's forced-sync phase timing:
// forcing a GPU round-trip on every checkpoint adds submission latency that
// normally overlaps with adjacent work. The SHARE each bucket takes of the
// forced-sync total is what transfers to the real budget; `report()` prints
// both the raw forced-sync mean and a share-projected mean against a
// separately-measured real (unforced) decode us/token, same methodology as
// T31's prefill share-projection table.

import Foundation
import Metal

public enum LagunaDecodeKernelTiming {
    public static var enabled: Bool {
        ButterEnv["BUTTER_LAGUNA_DECODE_KERNEL_TIMING"] == "1"
    }

    private struct Bucket {
        var wallUs: Double = 0
        var gpuUs: Double = 0
        var count: Int = 0
    }

    nonisolated(unsafe) private static var buckets: [String: Bucket] = [:]
    nonisolated(unsafe) private static var order: [String] = []
    nonisolated(unsafe) private static var tokenCount: Int = 0
    nonisolated(unsafe) private static var tokenWallUs: Double = 0
    nonisolated(unsafe) private static var tokenT0: Double = 0
    private static let lock = NSLock()

    /// Forces a commit+wait checkpoint on `cmd`, records the elapsed
    /// wall/GPU time under `label`, and returns a fresh command buffer for
    /// the caller to keep encoding onto. Callers should still guard the
    /// call site with `enabled` (avoids the extra allocation when off);
    /// mirrors T26's `attnPhaseCheckpoint(Buf)` pattern in
    /// `LagunaText.forwardMulti`.
    public static func checkpoint(_ label: String, cmd: MTLCommandBuffer, device: Device)
        -> MTLCommandBuffer
    {
        guard enabled else { return cmd }
        let t0 = CFAbsoluteTimeGetCurrent()
        if cmd.status != .committed, cmd.status != .completed, cmd.status != .error {
            cmd.commit()
        }
        cmd.waitUntilCompleted()
        let t1 = CFAbsoluteTimeGetCurrent()
        let wallUs = (t1 - t0) * 1e6
        // gpuStartTime/gpuEndTime are populated automatically once the
        // buffer completes (no MTLCounterSampleBuffer/counterSet setup
        // needed) - guaranteed by the waitUntilCompleted above.
        let gpuUs = max(0, (cmd.gpuEndTime - cmd.gpuStartTime) * 1e6)
        record(label, wallUs: wallUs, gpuUs: gpuUs)
        return device.makeCommandBuffer()
    }

    private static func record(_ label: String, wallUs: Double, gpuUs: Double) {
        lock.lock(); defer { lock.unlock() }
        if buckets[label] == nil { order.append(label) }
        var b = buckets[label] ?? Bucket()
        b.wallUs += wallUs
        b.gpuUs += gpuUs
        b.count += 1
        buckets[label] = b
    }

    /// Marks the start of one decode token's work (wraps the generation
    /// driver's per-token sample call). Used only to compute the outer
    /// "sampling/host gap" residual in `report()` - the wall time inside
    /// one token that no checkpoint bucket accounts for.
    public static func beginToken() {
        guard enabled else { return }
        tokenT0 = CFAbsoluteTimeGetCurrent()
    }

    public static func endToken() {
        guard enabled else { return }
        let dt = (CFAbsoluteTimeGetCurrent() - tokenT0) * 1e6
        lock.lock()
        tokenCount += 1
        tokenWallUs += dt
        lock.unlock()
    }

    public static func reset() {
        lock.lock()
        buckets = [:]
        order = []
        tokenCount = 0
        tokenWallUs = 0
        lock.unlock()
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }

    private static func padNum(_ v: Double, _ width: Int, decimals: Int = 1) -> String {
        let s = String(format: "%.\(decimals)f", v)
        return s.count >= width ? s : String(repeating: " ", count: width - s.count) + s
    }

    /// Prints the aggregated table to stderr. `realUsPerToken`, when
    /// provided, is a separately-measured *unforced* decode us/token (a
    /// plain `--stats` run at the same prompt/config) used to project each
    /// bucket's forced-sync SHARE onto the real budget.
    public static func report(realUsPerToken: Double? = nil) {
        lock.lock()
        let localBuckets = buckets
        let localOrder = order
        let n = max(tokenCount, 1)
        let forcedWallPerTokenUs = tokenWallUs / Double(n)
        lock.unlock()

        guard !localOrder.isEmpty else {
            fputs("[decode-kernel-timing] no samples recorded (tokens=\(n))\n", stderr)
            return
        }

        let totalBucketWallUs = localOrder.reduce(0.0) { $0 + (localBuckets[$1]?.wallUs ?? 0) }
        fputs("\n[decode-kernel-timing] N=\(n) tokens\n", stderr)
        fputs(
            "[decode-kernel-timing] forced-sync total "
                + String(format: "%.2f", forcedWallPerTokenUs / 1000) + "ms/token "
                + "(inflated vs real pipelined decode - forced commit+wait on every "
                + "checkpoint removes overlap the real decode loop gets for free; "
                + "SHARE below is what transfers, not this absolute number)\n",
            stderr)
        if let realUsPerToken {
            fputs(
                "[decode-kernel-timing] real (unforced, --stats) decode: "
                    + String(format: "%.2f", realUsPerToken / 1000) + "ms/token\n",
                stderr)
        }

        let sorted = localOrder.sorted {
            (localBuckets[$0]?.wallUs ?? 0) > (localBuckets[$1]?.wallUs ?? 0)
        }

        var lines: [String] = []
        lines.append(
            pad("phase", 26) + pad("wall_us", 11) + pad("gpu_us", 11) + pad("gap_us", 11)
                + pad("share%", 9) + pad("real_us", 10))
        var checkedTotalShare = 0.0
        for label in sorted {
            guard let b = localBuckets[label] else { continue }
            let wallPerToken = b.wallUs / Double(n)
            let gpuPerToken = b.gpuUs / Double(n)
            let gapPerToken = max(0, wallPerToken - gpuPerToken)
            let share = totalBucketWallUs > 0 ? (b.wallUs / totalBucketWallUs) * 100 : 0
            checkedTotalShare += share
            let realStr: String
            if let realUsPerToken {
                realStr = String(format: "%.1f", realUsPerToken * share / 100.0)
            } else {
                realStr = "-"
            }
            lines.append(
                pad(label, 26) + padNum(wallPerToken, 11) + padNum(gpuPerToken, 11)
                    + padNum(gapPerToken, 11) + padNum(share, 8) + "% "
                    + pad(realStr, 9))
        }
        // Sampling/host-gap residual: whole-token wall time (beginToken /
        // endToken, covers everything inside `sampleNext` - forward +
        // argmax/sample) minus the sum of every measured checkpoint bucket.
        // Positive => real work (or forced-sync overhead) this probe did not
        // bucket by name; near zero => the checkpoints above account for
        // essentially all of one token.
        let residualUs = max(0, forcedWallPerTokenUs - localOrder.reduce(0.0) {
            $0 + (localBuckets[$1]?.wallUs ?? 0) / Double(n)
        })
        lines.append(
            pad("sampling/host_gap(residual)", 26) + padNum(residualUs, 11) + padNum(0, 11)
                + padNum(residualUs, 11) + padNum(0, 8) + "% " + pad("-", 9))
        lines.append(
            pad("TOTAL (forced-sync)", 26) + padNum(forcedWallPerTokenUs, 11))
        for line in lines { fputs(line + "\n", stderr) }
        fputs(
            String(format: "[decode-kernel-timing] bucket share sum: %.1f%%\n", checkedTotalShare),
            stderr)
    }
}
