// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed as an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Test probe: count prefill chunks begun vs committed so a TCP drop
// can prove leftover chunks did not run. Default off.

import Foundation
import os

enum PrefillAbortProbe {
    private static let lock = OSAllocatedUnfairLock(
        initialState: (
            enabled: false, begun: 0, committed: 0, holdAfterFirst: false,
            cancellationObserved: false, targetPromptCount: nil as Int?,
            targetFirstToken: nil as Int?, targetChunkSize: nil as Int?
        ))
    private static let firstCommitGate = NSCondition()
    private nonisolated(unsafe) static var firstCommitReached = false

    static var enabled: Bool {
        get { lock.withLock { $0.enabled } }
        set { lock.withLock { $0.enabled = newValue } }
    }
    static var begun: Int { lock.withLock { $0.begun } }
    static var committed: Int { lock.withLock { $0.committed } }
    static var cancellationObserved: Bool { lock.withLock { $0.cancellationObserved } }
    static var holdAfterFirstCommit: Bool {
        get { lock.withLock { $0.holdAfterFirst } }
        set {
            lock.withLock { $0.holdAfterFirst = newValue }
            if !newValue { wakeFirstCommitGate() }
        }
    }

    static func target(promptCount: Int, firstToken: Int?, chunkSize: Int) {
        lock.withLock {
            $0.targetPromptCount = promptCount
            $0.targetFirstToken = firstToken
            $0.targetChunkSize = chunkSize
        }
    }

    static func reset() {
        lock.withLock {
            $0.begun = 0
            $0.committed = 0
            $0.holdAfterFirst = false
            $0.cancellationObserved = false
            $0.targetPromptCount = nil
            $0.targetFirstToken = nil
            $0.targetChunkSize = nil
        }
        firstCommitGate.lock()
        firstCommitReached = false
        firstCommitGate.broadcast()
        firstCommitGate.unlock()
    }

    static func markBegun(promptCount: Int, firstToken: Int?, chunkSize: Int) {
        lock.withLock { s in
            guard s.enabled,
                s.targetPromptCount == promptCount,
                s.targetFirstToken == firstToken,
                s.targetChunkSize == chunkSize
            else { return }
            s.begun += 1
        }
    }

    static func markCommitted(promptCount: Int, firstToken: Int?, chunkSize: Int) {
        let shouldHold = lock.withLock { s -> Bool in
            guard s.enabled,
                s.targetPromptCount == promptCount,
                s.targetFirstToken == firstToken,
                s.targetChunkSize == chunkSize
            else { return false }
            s.committed += 1
            return s.holdAfterFirst && s.committed == 1
        }
        guard shouldHold else { return }

        firstCommitGate.lock()
        firstCommitReached = true
        firstCommitGate.broadcast()
        while holdAfterFirstCommit && !Task.isCancelled {
            firstCommitGate.wait(until: Date().addingTimeInterval(0.01))
        }
        if Task.isCancelled {
            lock.withLock { $0.cancellationObserved = true }
        }
        firstCommitGate.unlock()
    }

    static func waitForFirstCommit(timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        firstCommitGate.lock()
        defer { firstCommitGate.unlock() }
        while !firstCommitReached {
            if !firstCommitGate.wait(until: deadline) { return firstCommitReached }
        }
        return true
    }

    private static func wakeFirstCommitGate() {
        firstCommitGate.lock()
        firstCommitGate.broadcast()
        firstCommitGate.unlock()
    }
}
