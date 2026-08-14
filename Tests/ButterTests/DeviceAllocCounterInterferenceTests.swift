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
// Synthetic, deterministic confirmation of the Gemma4MoEAllocTraceTests
// flake mechanism (see AllocMeasurementRetry.swift for the full writeup):
// `Device.shared`'s `bufferAllocCount`/`bufferAllocBytes`/
// `scratchAllocCount`/`scratchAllocBytes` are plain, non-atomic
// `public var`s on a process-wide singleton. A reset/measure window
// around them only reflects "this suite's own allocations" if nothing
// ELSE in the process allocates through `Device.shared` while the
// window is open — which Swift Testing's default cross-suite
// parallelism does not guarantee.
//
// This is deliberately built to be non-flaky-in-itself: the "foreign"
// allocation is synchronized with semaphores, not a sleep, so it lands
// strictly inside the measurement window on every run, on every
// machine — no timing luck involved, and safe to run as a permanent
// regression test.
import Foundation
import Testing

@testable import ButterSwift

@Suite(
    "Device.shared alloc-counter cross-task interference (Gemma4 alloc-trace flake root-cause)",
    .serialized
)
struct DeviceAllocCounterInterferenceTests {

    /// Resets `Device.shared`'s alloc counters, runs `body`, and
    /// returns the raw total bytes attributed to the window —
    /// deliberately with NO interference protection, i.e. exactly what
    /// `Gemma4MoEAllocTraceTests` did before the `stableMeasurement`
    /// fix.
    private func rawWindow(_ body: () -> Void) -> Int {
        let device = Device.shared
        device.bufferAllocCount = 0
        device.bufferAllocBytes = 0
        device.scratchAllocCount = 0
        device.scratchAllocBytes = 0
        body()
        return device.bufferAllocBytes + device.scratchAllocBytes
    }

    @Test("a concurrent foreign allocation through Device.shared bleeds into an unrelated measurement window")
    func foreignAllocationBleedsIntoWindow() {
        let device = Device.shared
        let ownBytes = 256
        let foreignBytes = 12_345

        // Three semaphores hand off control so the foreign allocation
        // is GUARANTEED to fire strictly between the window's reset and
        // its read — deterministic, not a race against the scheduler.
        let interfererReady = DispatchSemaphore(value: 0)
        let openWindowNow = DispatchSemaphore(value: 0)
        let foreignAllocDone = DispatchSemaphore(value: 0)

        let interferer = Thread {
            interfererReady.signal()
            openWindowNow.wait()
            _ = device.makeBuffer(length: foreignBytes)
            foreignAllocDone.signal()
        }
        interferer.start()
        interfererReady.wait()  // interferer is parked, about to allocate

        let measured = rawWindow {
            _ = device.makeBuffer(length: ownBytes)  // "our" own known allocation
            openWindowNow.signal()  // let the foreign allocation fire NOW, inside the window
            foreignAllocDone.wait()  // block until it lands before reading the counters
        }

        let bleedMessage =
            "expected the foreign allocation to bleed into the window (measured \(measured), "
            + "own \(ownBytes) + foreign \(foreignBytes) = \(ownBytes + foreignBytes)) — if "
            + "this fails, Device.shared's counters are no longer shared, non-atomic "
            + "process-wide state and the interference hypothesis needs re-checking"
        #expect(measured == ownBytes + foreignBytes, "\(bleedMessage)")
        #expect(
            measured != ownBytes,
            "sanity: a corrupted window must NOT read back as just its own bytes")
    }

    @Test("stableMeasurement detects a one-shot interference hit and recovers on retry")
    func stableMeasurementRecoversFromOneShotInterference() {
        let device = Device.shared
        let ownBytes = 256
        let foreignBytes = 999

        // Interference fires on the FIRST attempt only — models a rare,
        // one-off foreign allocation landing in a single CI run, not
        // sustained hammering. (See `stableMeasurementConvergesNonConsecutively`
        // below for the sustained/varying-interference case that PR
        // #139's CI run 31800387587 actually showed.)
        var interferenceArmed = true
        var attempts: [Int] = []

        let result = stableMeasurement(maxAttempts: 4) { () -> Int in
            let v = rawWindow {
                _ = device.makeBuffer(length: ownBytes)
                if interferenceArmed {
                    interferenceArmed = false
                    _ = device.makeBuffer(length: foreignBytes)
                }
            }
            attempts.append(v)
            return v
        }

        #expect(
            attempts.first == ownBytes + foreignBytes,
            "first attempt should be the contaminated reading: \(attempts)")
        #expect(
            result == ownBytes,
            "stableMeasurement should converge on the clean, repeatable value: \(attempts)")
        let retryCountMessage =
            "should need the contaminated attempt #1, a clean attempt #2, and a clean "
            + "attempt #3 that repeats attempt #2's value before returning: \(attempts)"
        #expect(attempts.count >= 3, "\(retryCountMessage)")
    }

    @Test("stableMeasurement converges via non-consecutive agreement under persistent, varying interference")
    func stableMeasurementConvergesNonConsecutively() {
        let device = Device.shared
        let ownBytes = 256

        // A DIFFERENT foreign byte count on most attempts, so no two
        // CONTAMINATED readings ever coincidentally match each other —
        // only the clean reading (256, `nil` below) can repeat. Clean
        // windows land at attempts #3 and #5, separated by a
        // contaminated attempt #4. The OLD "two-CONSECUTIVE" rule would
        // never converge here (no two ADJACENT attempts ever agree:
        // #1≠#2, #2≠#3, #3≠#4, #4≠#5) and would exhaust maxAttempts —
        // exactly the CI run 31800387587 failure shape, where ~285
        // suites racing Device.shared meant most windows took a
        // DIFFERENT foreign hit rather than the same one twice in a
        // row. "Any two of N" converges the moment the second clean
        // reading appears, matching non-adjacent attempt #3.
        let foreignBytesSchedule: [Int?] = [111, 222, nil, 333, nil]
        var idx = 0
        var attempts: [Int] = []

        let result = stableMeasurement(maxAttempts: 8) { () -> Int in
            let foreign = foreignBytesSchedule[idx]
            idx += 1
            let v = rawWindow {
                _ = device.makeBuffer(length: ownBytes)
                if let foreign { _ = device.makeBuffer(length: foreign) }
            }
            attempts.append(v)
            return v
        }

        #expect(
            attempts == [367, 478, 256, 589, 256],
            "unexpected attempt sequence — the fixture schedule itself may have changed: \(attempts)"
        )
        #expect(
            result == 256,
            "should converge on the clean value via non-consecutive agreement: \(attempts)")
        let stopCountMessage =
            "should stop as soon as the second clean reading (attempt #5) repeats attempt #3's "
            + "value, without needing attempts #1/#2/#4 (all mutually distinct) to agree with "
            + "anything: \(attempts)"
        #expect(attempts.count == 5, "\(stopCountMessage)")
    }

    @Test("retryUntilAcceptable retries until an attempt's own check passes")
    func retryUntilAcceptableRetriesUntilOwnCheckPasses() {
        var attempts: [Int] = []
        var callCount = 0

        let result = retryUntilAcceptable(maxAttempts: 8, isAcceptable: { $0 >= 3 }) { () -> Int in
            callCount += 1
            attempts.append(callCount)
            return callCount
        }

        #expect(result == 3, "should stop at the first attempt satisfying isAcceptable: \(attempts)")
        #expect(attempts == [1, 2, 3], "should have made exactly 3 attempts: \(attempts)")
    }
}
