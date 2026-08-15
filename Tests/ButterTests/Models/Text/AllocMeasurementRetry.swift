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
// stableMeasurement / retryUntilAcceptable — interference-detector +
// bounded-retry for tests that read `Device.shared`'s alloc counters
// (`bufferAllocBytes` / `scratchAllocBytes` / their `*Count` siblings —
// see Device.swift). Those are plain, non-atomic `public var`s on the
// process-wide `Device.shared` singleton. `Gemma4MoEAllocTraceTests` is
// itself `.serialized`, but Swift Testing parallelizes UNRELATED
// suites within the same process by default — CI's full-suite
// `swift test --filter "ButterTests|IronSwiftTests"` run loads ~285
// other suites that also allocate through `Device.shared` concurrently.
// When one of THOSE allocations happens to land between a reset and a
// read, its bytes silently land in our count too. A scoped
// `swift test --filter Gemma4MoEAllocTraceTests` run never loads any
// other suite into the process, so it never sees this — that mismatch
// (flaky in CI's full run, zero variance locally/scoped) is exactly
// what CI runs 31732003551, 31769728283, 31779837853, and (post-first-
// fix, at N=285-suite scale) 31800387587 showed. See
// `DeviceAllocCounterInterferenceTests` for a synthetic, deterministic
// (non-flaky-in-itself) reproduction of the mechanism.
//
// PR #139's CI run (31800387587) falsified the original design's
// working assumption: with ~285 suites racing `Device.shared`
// concurrently, a foreign allocation landing inside a measurement
// window is not a rare one-off — it can hit MOST or ALL windows, so
// two CONSECUTIVE attempts agreeing essentially never happens. The
// counters are still 100% deterministic in isolation (exact sums of
// caller-REQUESTED byte lengths, see `Device.makeBuffer(length:)` /
// `Device.allocScratch(bytes:)` — never anything read back from the
// Metal driver's actual granted size), so the TRUE value still recurs
// every time a window happens to land quiet — it just doesn't recur on
// two ADJACENT attempts under sustained interference, because
// different attempts get hit by different foreign allocations of
// different sizes. `stableMeasurement` below now accepts agreement
// between ANY two attempts within a bounded window (not just
// consecutive ones): the true value's repeat count grows with N (it
// wins whenever ANY two of N windows are quiet), while any single
// contamination pattern would have to recur byte-for-byte by pure
// coincidence to produce a false positive.
//
// `retryUntilAcceptable` is the companion for tests whose own
// assertion already tolerates jitter (e.g. `perTokenStability`'s
// spread-vs-mean shape check) — there, requiring two RAW readings to
// match exactly (even non-consecutively) throws away attempts that are
// already "close enough" per the test's own criterion. It retries
// until ONE attempt satisfies that criterion directly.
//
// Both are test-only, process-local retries — neither touches
// `Device.swift` or any production allocation path.
import Testing

/// Runs `body()` up to `maxAttempts` times, returning the first value
/// that has been observed at least twice — agreement need NOT be
/// between consecutive attempts (see the file doc comment for why:
/// under CI's ~285-suite concurrent load, adjacent-attempt agreement
/// essentially never happens, but the true value still recurs
/// whenever any two of the N windows happen to be quiet). Records an
/// `Issue` with the FULL attempt history (and returns the last
/// observed value) if `maxAttempts` is exhausted without any value
/// repeating — that shape means either sustained interference severe
/// enough that no two of N windows were both quiet, or a genuine
/// regression in the measured code path, and either way should
/// surface as a test failure with the raw attempts visible rather than
/// a bare "Issue recorded".
func stableMeasurement<T: Equatable>(
    maxAttempts: Int = 8,
    _ body: () throws -> T
) rethrows -> T {
    precondition(maxAttempts >= 2, "stableMeasurement needs at least 2 attempts to compare")
    var history: [T] = []
    for _ in 0..<maxAttempts {
        let current = try body()
        if history.contains(current) {
            return current
        }
        history.append(current)
    }
    let message =
        "stableMeasurement: \(maxAttempts) attempts, no value repeated (no two-of-\(maxAttempts) "
        + "agreement) — either sustained cross-suite interference severe enough that no two "
        + "windows landed quiet, or a real regression in the measured code path. All attempts, "
        + "in order: \(history)"
    Issue.record("\(message)")
    return history[history.count - 1]
}

/// Runs `body()` up to `maxAttempts` times, returning the first result
/// for which `isAcceptable` returns `true`. Use where the caller's OWN
/// assertion already tolerates jitter (a shape/threshold check, not
/// exact equality) — retrying until a fresh attempt directly satisfies
/// that same check is strictly better than requiring two raw readings
/// to match each other, since it also accepts an attempt whose
/// interference-contaminated bytes still happen to fall inside the
/// tolerated shape. Records an `Issue` with the FULL attempt history
/// (and returns the last attempt, so the caller's own `#expect`s still
/// fire with a precise message) if no attempt satisfies `isAcceptable`
/// within `maxAttempts`.
func retryUntilAcceptable<T>(
    maxAttempts: Int = 8,
    isAcceptable: (T) -> Bool,
    _ body: () throws -> T
) rethrows -> T {
    precondition(maxAttempts >= 1, "retryUntilAcceptable needs at least 1 attempt")
    var history: [T] = []
    for _ in 0..<maxAttempts {
        let current = try body()
        history.append(current)
        if isAcceptable(current) {
            return current
        }
    }
    let message =
        "retryUntilAcceptable: \(maxAttempts) attempts, none satisfied the acceptance check — "
        + "either sustained cross-suite interference or a real regression in the measured code "
        + "path. All attempts, in order: \(history)"
    Issue.record("\(message)")
    return history[history.count - 1]
}
