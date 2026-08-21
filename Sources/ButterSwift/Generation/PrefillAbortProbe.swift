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
        initialState: (enabled: false, begun: 0, committed: 0))

    static var enabled: Bool {
        get { lock.withLock { $0.enabled } }
        set { lock.withLock { $0.enabled = newValue } }
    }
    static var begun: Int { lock.withLock { $0.begun } }
    static var committed: Int { lock.withLock { $0.committed } }

    static func reset() {
        lock.withLock { $0.begun = 0; $0.committed = 0 }
    }

    static func markBegun() {
        lock.withLock { s in
            guard s.enabled else { return }
            s.begun += 1
        }
    }

    static func markCommitted() {
        lock.withLock { s in
            guard s.enabled else { return }
            s.committed += 1
        }
    }
}
