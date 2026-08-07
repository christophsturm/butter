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
// LogitMarginDebug: per-token greedy-decode confidence probe.
//
// `BUTTER_DEBUG_LOGIT_MARGIN=1` prints one line per emitted token to
// stderr: the token index, the chosen (argmax) token id, and the top-2
// logit gap (chosen logit minus the best of everything else) — a cheap
// confidence signal for "was this a near-tie or a clear winner". `=2`
// additionally prints the top-5 (id, logit) pairs for that token.
//
// This exists to settle a specific drift question (see
// `research/retrieval_attention` / the W43 laguna-drift investigation):
// Laguna's greedy stream differs depending on which argmax path runs
// (`BUTTER_GREEDY_GPU_ARGMAX` / `BUTTER_GPU_TOKEN_LOOP`), and the open
// question is whether that's benign near-tie flipping or the two paths
// disagreeing on the logits themselves. Call sites feed this the exact
// `Tensor` each path already argmaxes, so the reported gap always
// reflects what that path actually saw — see the call sites in
// `Generate.swift` (prefill seed token + CPU-sample decode step),
// `LanguageModel.swift` (`forwardSample`, the GPU-argmax-but-synchronous
// fallback decode loop), and `GpuTokenLoop.swift` (the chained GPU
// decode loop's `launch` closure).
//
// Cost: guarded by `enabled` at every call site (one `ProcessInfo`
// dictionary lookup cached per env read), so near-zero when off. When on,
// the GPU call sites force a synchronous readback of the logits row that
// the hot path normally never pays for — expected and fine, this is a
// debug flag, not a perf-preserving one.

import Foundation

public enum LogitMarginDebug {
    /// 0 = off, 1 = margin line, 2 = margin line + top-5.
    public static var level: Int {
        switch ProcessInfo.processInfo.environment["BUTTER_DEBUG_LOGIT_MARGIN"] {
        case "1": return 1
        case "2": return 2
        default: return 0
        }
    }

    public static var enabled: Bool { level >= 1 }

    nonisolated(unsafe) private static var tokenCounter: Int = 0
    private static let lock = NSLock()

    /// Resets the per-generation token counter. Call once per
    /// `driveGeneration` invocation, before the first token (prefill's
    /// seed token, index 0) is reported.
    public static func reset() {
        lock.lock()
        tokenCounter = 0
        lock.unlock()
    }

    /// Reports margin (and, at level 2, top-5) for `logits` against the
    /// already-chosen `chosenId`, tagging it with the next sequential
    /// token index. No-op unless `enabled`.
    ///
    /// Precondition: `logits` must already be host-readable (its
    /// producing GPU work fully synced) — callers on the async GPU path
    /// must force a wait first; see `GpuTokenLoop.runGPU`'s `launch`.
    public static func report(logits: Tensor, chosenId: Int) {
        guard enabled else { return }
        report(values: Sampling.decodeF32(logits), chosenId: chosenId)
    }

    /// Same as `report(logits:chosenId:)` but takes already-decoded f32
    /// values directly — the masked-candidate cpuSample branch in
    /// `Generate.swift` has these on hand without a second Tensor
    /// round-trip.
    public static func report(values: [Float], chosenId: Int) {
        guard enabled else { return }
        guard !values.isEmpty, chosenId >= 0, chosenId < values.count else { return }

        let chosenVal = values[chosenId]
        var second = -Float.infinity
        for i in 0 ..< values.count where i != chosenId {
            if values[i] > second { second = values[i] }
        }
        let gap = chosenVal - second

        let t: Int
        lock.lock()
        t = tokenCounter
        tokenCounter += 1
        lock.unlock()

        fputs("[margin] t=\(t) id=\(chosenId) gap=\(gap)\n", stderr)

        if level >= 2 {
            let top5 = values.enumerated().sorted { $0.element > $1.element }.prefix(5)
            let s = top5.map { "(\($0.offset), \($0.element))" }.joined(separator: " ")
            fputs("[margin-top5] t=\(t) \(s)\n", stderr)
        }
    }
}
