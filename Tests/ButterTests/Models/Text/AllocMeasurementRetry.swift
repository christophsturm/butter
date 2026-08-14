// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// stableMeasurement — interference-detector + bounded-retry for tests
// that read `Device.shared`'s alloc counters (`bufferAllocBytes` /
// `scratchAllocBytes` / their `*Count` siblings — see Device.swift).
// Those are plain, non-atomic `public var`s on the process-wide
// `Device.shared` singleton. `Gemma4MoEAllocTraceTests` is itself
// `.serialized`, but Swift Testing parallelizes UNRELATED suites
// within the same process by default — CI's full-suite
// `swift test --filter "ButterTests|IronSwiftTests"` run loads dozens
// of other suites that also allocate through `Device.shared`
// concurrently. When one of THOSE allocations happens to land between
// a reset and a read, its bytes silently land in our count too. A
// scoped `swift test --filter Gemma4MoEAllocTraceTests` run never
// loads any other suite into the process, so it never sees this — that
// mismatch (flaky in CI's full run, zero variance locally/scoped) is
// exactly what CI runs 31732003551, 31769728283, and 31779837853
// showed. See `DeviceAllocCounterInterferenceTests` for a synthetic,
// deterministic (non-flaky-in-itself) reproduction of the mechanism.
//
// The counters this guards are otherwise 100% deterministic: they are
// exact running sums of caller-REQUESTED byte lengths (see
// `Device.makeBuffer(length:)` / `Device.allocScratch(bytes:)`), never
// anything read back from the Metal driver's actual granted allocation
// size — so two independent, freshly-generated measurements of the
// SAME knobs on an otherwise-quiet process produce byte-IDENTICAL
// results. A foreign allocation is a one-off event: it corrupts at
// most one attempt and has no reason to repeat the exact same
// contamination on the very next, independent attempt. Requiring two
// CONSECUTIVE attempts to agree therefore squares an already-rare
// per-attempt hit rate — indistinguishable from zero in CI's history —
// while a REAL regression in the measured code path reproduces
// identically on every attempt and still fails, every time, once
// `maxAttempts` is exhausted. This is a test-only, process-local retry
// — it does not touch `Device.swift` or any production allocation
// path.
import Testing

/// Runs `body()` up to `maxAttempts` times, returning as soon as two
/// CONSECUTIVE calls produce an `==` result. Records an `Issue` (and
/// returns the last observed value) if `maxAttempts` is exhausted
/// without two consecutive agreeing reads — that shape means either
/// sustained/repeated interference (very unlikely, see the file doc
/// comment) or a genuine regression in the measured code path, and
/// either way should surface as a test failure rather than being
/// silently papered over.
func stableMeasurement<T: Equatable>(
    maxAttempts: Int = 4,
    _ body: () throws -> T
) rethrows -> T {
    precondition(maxAttempts >= 2, "stableMeasurement needs at least 2 attempts to compare")
    var previous: T?
    var history: [T] = []
    for _ in 0..<maxAttempts {
        let current = try body()
        history.append(current)
        if let previous, previous == current {
            return current
        }
        previous = current
    }
    let message =
        "stableMeasurement: \(maxAttempts) attempts never agreed twice in a row — either "
        + "sustained cross-suite interference or a real regression in the measured code "
        + "path: \(history)"
    Issue.record("\(message)")
    return history[history.count - 1]
}
