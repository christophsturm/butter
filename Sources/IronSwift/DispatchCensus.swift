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
// DispatchCensus: env-gated (BUTTER_DISPATCH_CENSUS=1) global counter of
// Metal compute dispatches, keyed by kernel name. Diagnostic-only — the
// standing tool for per-token/per-pass dispatch censuses when deciding
// whether a decode/prefill path is submission-bound (dispatch-count cuts
// pay) or compute/bandwidth-bound (they don't). NOT a production feature;
// the counter is a no-op unless the env var is set.
//
// Design: `PSOCache.pipelineState(for:)` is the single funnel every
// IronKernels-generated dispatch AND most hand-written Ops.swift/
// OpsNVFP4.swift/OpsFused.swift kernels pass through (one PSO lookup per
// dispatch, 1:1, immediately followed by exactly one encoder+dispatch+
// endEncoding). Ticking there, keyed by kernel name, gives a REAL
// per-kernel-name histogram for that whole surface for free.
//
// The exception: a handful of hand-fused "mega-kernel" files under
// Models/Text/Laguna*.swift maintain their OWN local MTLComputePipelineState
// cache (bypassing PSOCache, presumably to dodge its NSLock on the hottest
// path) -- primarily LagunaSdpa.swift's attention MMA core. Those call
// `DispatchCensus.tick(_:)` directly at their own dispatch sites/function
// entries; see call sites for the accounting note at each.
//
// Zero-cost when disabled: every call site is a single branch on a
// lazily-cached Bool (`enabled`), no lock touched, no dictionary touched.

import Foundation

public enum DispatchCensus {
    public static let enabled: Bool = {
        ProcessInfo.processInfo.environment["BUTTER_DISPATCH_CENSUS"] == "1"
    }()

    private static let lock = NSLock()
    // `nonisolated(unsafe)`: mutation is guarded by `lock` below, not by
    // actor isolation -- same pattern PSOCache.swift uses for its instance
    // `cache` dict via NSLock (mirrors an existing project convention).
    nonisolated(unsafe) private static var counts: [String: Int] = [:]
    nonisolated(unsafe) private static var total: Int = 0

    /// Record one dispatch under `label` (kernel name, or a hand-chosen
    /// function-name label for the local-cache mega-kernel files where a
    /// single label often represents several raw dispatches internally --
    /// see the per-label multiplier note printed by `report()`).
    @inline(__always)
    public static func tick(_ label: String) {
        guard enabled else { return }
        lock.lock()
        counts[label, default: 0] += 1
        total += 1
        lock.unlock()
    }

    public static func reset() {
        lock.lock()
        counts.removeAll()
        total = 0
        lock.unlock()
    }

    /// Snapshot the current histogram without printing (for programmatic
    /// use, e.g. the CLI printing a compact one-line summary).
    public static func snapshot() -> (total: Int, counts: [String: Int]) {
        lock.lock()
        defer { lock.unlock() }
        return (total, counts)
    }

    public static func report(topN: Int = 25) {
        guard enabled else { return }
        let (t, snapshot) = snapshot()
        FileHandle.standardError.write("[dispatch-census] total ticks: \(t)\n".data(using: .utf8)!)
        let sorted = snapshot.sorted { $0.value > $1.value }
        for (name, c) in sorted.prefix(topN) {
            FileHandle.standardError.write(
                "[dispatch-census]   \(c)\t\(name)\n".data(using: .utf8)!)
        }
    }
}
