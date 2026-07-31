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
// GpuTokenLoop: ring/lag scheduling logic is Metal-free and exercised
// directly with an in-memory fake GPU harness (`GpuTokenLoop.run`'s
// closures don't touch Metal at all). The dual-purpose ring buffer
// claim ("a device tensor populated by Ops.argmax gathers identically
// to a host-built one") is checked against real Metal, guarded the same
// way sibling GPU tests are (e.g. Nvfp4MppSmokeTests).

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("GpuTokenLoopSchedule")
struct GpuTokenLoopScheduleTests {
    @Test("slot(forToken:) wraps around ringSize")
    func slotWraparound() {
        let schedule = GpuTokenLoopSchedule(ringSize: 4, lag: 1)
        let expected = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        for (tokenIndex, want) in expected.enumerated() {
            #expect(schedule.slot(forToken: tokenIndex) == want)
        }
    }

    @Test("ring reuse never collides with an unresolved slot under the lag bound")
    func ringReuseRespectsLagBound() {
        // For every (ringSize, lag) pair the loop actually uses
        // (ringSize = max(lag + 3, 4) in GpuTokenLoop.runGPU), confirm
        // the slot a token `t` occupies is never reused by a later
        // token until at least `ringSize` steps later, i.e. two
        // distinct token indices within one lag-window's reach of each
        // other never share a slot.
        for lag in 0 ... 4 {
            let ringSize = max(lag + 3, 4)
            let schedule = GpuTokenLoopSchedule(ringSize: ringSize, lag: lag)
            for t in 0 ..< 50 {
                let slotT = schedule.slot(forToken: t)
                for later in (t + 1) ... (t + lag + 1) {
                    #expect(schedule.slot(forToken: later) != slotT)
                }
            }
        }
    }
}

@Suite("GpuTokenLoop.run scheduling")
struct GpuTokenLoopRunTests {
    @Test("zero remaining budget is a no-op")
    func zeroBudgetNoOp() throws {
        let schedule = GpuTokenLoopSchedule(ringSize: 4, lag: 1)
        var launches = 0
        let stopped = try GpuTokenLoop.run(
            remainingBudget: 0, schedule: schedule,
            launch: { _ in launches += 1 },
            resolve: { _ in 0 },
            isStop: { _ in false },
            onResolved: { _, _ in }
        )
        #expect(stopped == false)
        #expect(launches == 0)
    }

    @Test("full budget with no stop token: every token launched, resolved, and emitted in order")
    func fullBudgetNoStop() throws {
        let lag = 1
        let schedule = GpuTokenLoopSchedule(ringSize: 6, lag: lag)
        let remainingBudget = 10

        var launchedOrder: [Int] = []
        var outstanding = 0
        var maxOutstanding = 0
        var emitted: [(index: Int, value: UInt32)] = []

        let stopped = try GpuTokenLoop.run(
            remainingBudget: remainingBudget,
            schedule: schedule,
            launch: { tokenIndex in
                launchedOrder.append(tokenIndex)
                outstanding += 1
                maxOutstanding = max(maxOutstanding, outstanding)
            },
            resolve: { tokenIndex in
                outstanding -= 1
                return UInt32(tokenIndex * 10)
            },
            isStop: { _ in false },
            onResolved: { tokenIndex, value in emitted.append((tokenIndex, value)) }
        )

        #expect(stopped == false)
        #expect(launchedOrder == Array(1 ... remainingBudget))
        #expect(emitted.map { $0.index } == Array(1 ... remainingBudget))
        #expect(emitted.map { $0.value } == (1 ... remainingBudget).map { UInt32($0 * 10) })
        // Never more than lag + 1 steps in flight (launched, not yet
        // resolved) at once; this is the whole point of `lag`.
        #expect(maxOutstanding <= lag + 1)
    }

    @Test("stop token halts emission and bounds wasted lookahead to lag", arguments: [0, 1, 2, 3])
    func stopTokenWithinLagBound(lag: Int) throws {
        let schedule = GpuTokenLoopSchedule(ringSize: max(lag + 3, 4), lag: lag)
        let remainingBudget = 20
        let stopAt = 7  // 1-based token index that is the stop token

        var launchedCount = 0
        var emitted: [Int] = []

        let stopped = try GpuTokenLoop.run(
            remainingBudget: remainingBudget,
            schedule: schedule,
            launch: { _ in launchedCount += 1 },
            resolve: { tokenIndex in UInt32(tokenIndex) },
            isStop: { $0 == UInt32(stopAt) },
            onResolved: { tokenIndex, _ in emitted.append(tokenIndex) }
        )

        #expect(stopped == true)
        // The stop token itself, and everything after it, must never
        // be emitted: same contract as the old synchronous loop's
        // `if stopSet.contains(nextToken) { break }` (checked before
        // appending).
        #expect(emitted == Array(1 ..< stopAt))
        // At least the tokens up to and including the stop token were
        // launched; at most `lag` more beyond it were launched
        // speculatively before the scheduler noticed and stopped.
        #expect(launchedCount >= stopAt)
        #expect(launchedCount <= stopAt + lag)
    }

    @Test("stop token on the very first resolved token emits nothing")
    func stopOnFirstToken() throws {
        let schedule = GpuTokenLoopSchedule(ringSize: 4, lag: 1)
        var emitted: [Int] = []
        let stopped = try GpuTokenLoop.run(
            remainingBudget: 5, schedule: schedule,
            launch: { _ in },
            resolve: { tokenIndex in UInt32(tokenIndex) },
            isStop: { _ in true },
            onResolved: { tokenIndex, _ in emitted.append(tokenIndex) }
        )
        #expect(stopped == true)
        #expect(emitted.isEmpty)
    }

    @Test("errors from onResolved propagate out of run")
    func propagatesErrors() {
        struct Boom: Error {}
        let schedule = GpuTokenLoopSchedule(ringSize: 4, lag: 1)
        #expect(throws: Boom.self) {
            try GpuTokenLoop.run(
                remainingBudget: 3, schedule: schedule,
                launch: { _ in },
                resolve: { UInt32($0) },
                isStop: { _ in false },
                onResolved: { _, _ in throw Boom() }
            )
        }
    }
}

@Suite("GpuTokenLoop Metal gather equivalence")
struct GpuTokenLoopMetalTests {
    /// Same GPU-availability guard sibling Metal tests use (e.g.
    /// Nvfp4MppSmokeTests): skip rather than fail when no GPU.
    private var gpuAvailable: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }

    @Test("a device tensor written by Ops.argmax gathers identically to a host-built index tensor, for a scripted token sequence")
    func deviceIndexEmbeddingMatchesHostIndex() throws {
        try #require(gpuAvailable, "Metal required for this smoke test")

        let device = Device.shared
        let hidden = 8
        let vocab = 5
        // A short scripted sequence of "chosen tokens", stands in for
        // consecutive decode steps' argmax picks.
        let script = [3, 1, 4, 0, 2, 2, 4]

        // Synthetic embedding table: row i is filled with the value i,
        // so gathering row `token` is trivially checkable.
        var tableHost = [Float](repeating: 0, count: vocab * hidden)
        for row in 0 ..< vocab {
            for col in 0 ..< hidden { tableHost[row * hidden + col] = Float(row) }
        }
        let tableBuf = device.makeBuffer(length: tableHost.count * 4)
        tableHost.withUnsafeBytes {
            _ = memcpy(tableBuf.contents(), $0.baseAddress!, tableHost.count * 4)
        }
        let table = Tensor(buffer: tableBuf, offset: 0, shape: [vocab, hidden], dtype: .f32)

        for chosenToken in script {
            // Host-built index tensor: mirrors LagunaModel.forward(tokenId:)'s
            // tokenBuf construction (the path GpuTokenLoop's seed slot uses).
            let hostBuf = device.makeBuffer(length: 4)
            var hostVal = UInt32(chosenToken)
            memcpy(hostBuf.contents(), &hostVal, 4)
            let hostIndex = Tensor(buffer: hostBuf, offset: 0, shape: [1], dtype: .u32)

            // Device-produced index tensor: Ops.argmax over a synthetic
            // logits vector whose max sits at `chosenToken`, the same
            // shape/dtype/kernel GpuTokenLoop.runGPU chains into the next
            // step's embedding gather as `tokenIdBuffer`.
            var logitsHost = [Float](repeating: 0, count: vocab)
            logitsHost[chosenToken] = 100
            let logitsBuf = device.makeBuffer(length: vocab * 4)
            logitsHost.withUnsafeBytes {
                _ = memcpy(logitsBuf.contents(), $0.baseAddress!, vocab * 4)
            }
            let logits = Tensor(buffer: logitsBuf, offset: 0, shape: [vocab], dtype: .f32)
            let deviceIndexBuf = device.makeBuffer(length: 4)
            let deviceIndex = Tensor(buffer: deviceIndexBuf, offset: 0, shape: [1], dtype: .u32)

            let argmaxCmd = device.makeCommandBuffer()
            Ops.argmax(logits, into: deviceIndex, on: argmaxCmd)
            argmaxCmd.commit()
            argmaxCmd.waitUntilCompleted()

            let gatherCmd = device.makeCommandBuffer()
            let outHost = Ops.gather(table: table, tokenIds: hostIndex, on: gatherCmd)
            let outDevice = Ops.gather(table: table, tokenIds: deviceIndex, on: gatherCmd)
            gatherCmd.commit()
            gatherCmd.waitUntilCompleted()

            #expect(outHost.toFloatArray() == outDevice.toFloatArray())
            #expect(outDevice.toFloatArray() == [Float](repeating: Float(chosenToken), count: hidden))
        }
    }
}
