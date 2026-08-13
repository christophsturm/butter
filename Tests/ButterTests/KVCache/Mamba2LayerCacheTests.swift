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
// Mamba2LayerCacheTests — F-131 (NemotronH MTP self-speculative decode)
// added content-level snapshot/restore to `SSMStateCache` and
// `Mamba2LayerCache` (previously only `ConvStateCache`/`GDNStateCache`/
// `Qwen35GDNLayerCache` had it — see `CacheSnapshot.swift`'s new
// `.mamba2` case). These tests are the gate for that addition: bitwise
// round-trip correctness, standalone AND through the composite
// `[any LayerCacheProtocol].snapshotAll`/`restoreAll` path a
// speculative-decode driver actually calls, across multiple
// snapshot/mutate/restore cycles (simulating repeated draft-reject
// rollbacks in a decode loop).

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("Mamba2LayerCache snapshot/restore round-trip")
struct Mamba2LayerCacheTests {

    // MARK: - SSMStateCache standalone

    @Test("SSMStateCache snapshotEncode/restoreEncode round-trips bitwise")
    func ssmSnapshotRestoreBitwise() {
        autoreleasepool {
            let cache = SSMStateCache(nHeads: 3, stateDim: 5, headDim: 7)
            let original = (0 ..< (3 * 5 * 7)).map { Float($0) * 0.037 - 1.5 }
            cache.h.copyIn(from: original)

            let cmd = Device.shared.makeCommandBuffer()
            let snap = cache.snapshotEncode(on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            // Pollute the live state (simulating a rejected speculative
            // step's scan writes).
            let polluted = (0 ..< (3 * 5 * 7)).map { Float($0) * -2.11 + 9.0 }
            cache.h.copyIn(from: polluted)
            #expect(cache.h.toArray(as: Float.self) == polluted)

            let rCmd = Device.shared.makeCommandBuffer()
            cache.restoreEncode(from: snap, on: rCmd)
            rCmd.commit()
            rCmd.waitUntilCompleted()

            // Bitwise — fp32 blit, no arithmetic, must match exactly.
            #expect(cache.h.toArray(as: Float.self) == original)
        }
    }

    @Test("SSMStateCache committing snapshot()/restore() convenience wrappers round-trip")
    func ssmSnapshotRestoreConvenience() {
        autoreleasepool {
            let cache = SSMStateCache(nHeads: 2, stateDim: 4, headDim: 8)
            let original = (0 ..< (2 * 4 * 8)).map { Float($0) * 0.5 }
            cache.h.copyIn(from: original)

            let snap = cache.snapshot()
            cache.h.copyIn(from: [Float](repeating: -1, count: 2 * 4 * 8))
            cache.restore(from: snap)

            #expect(cache.h.toArray(as: Float.self) == original)
        }
    }

    @Test("SSMStateCache snapshot/restore across 5 sequential cycles stays bitwise exact")
    func ssmMultiCycleRoundTrip() {
        autoreleasepool {
            let cache = SSMStateCache(nHeads: 4, stateDim: 8, headDim: 16)
            let count = 4 * 8 * 16

            for cycle in 0 ..< 5 {
                let base = (0 ..< count).map { Float($0 + cycle * 97) * 0.011 }
                cache.h.copyIn(from: base)

                let cmd = Device.shared.makeCommandBuffer()
                let snap = cache.snapshotEncode(on: cmd)
                cmd.commit()
                cmd.waitUntilCompleted()

                // Simulate a decode step's worth of pollution — several
                // in-place mutations, as consecutive drafted tokens would
                // each advance the scan before a rejection is detected.
                for step in 0 ..< 3 {
                    let poison = (0 ..< count).map {
                        Float(($0 + step) % 13) * 3.3 - Float(cycle)
                    }
                    cache.h.copyIn(from: poison)
                }

                let rCmd = Device.shared.makeCommandBuffer()
                cache.restoreEncode(from: snap, on: rCmd)
                rCmd.commit()
                rCmd.waitUntilCompleted()

                #expect(
                    cache.h.toArray(as: Float.self) == base,
                    "cycle=\(cycle): restored state must exactly match the pre-pollution snapshot"
                )
            }
        }
    }

    // MARK: - Mamba2LayerCache composite (ssm + conv + length)

    @Test("Mamba2LayerCache composite snapshotEncode/restoreEncode round-trips both tensors + length")
    func compositeSnapshotRestore() {
        autoreleasepool {
            let cache = Mamba2LayerCache(
                nHeads: 2, stateDim: 4, headDim: 8,
                convChannels: 12, convKernelSize: 4, dtype: .f32)

            let ssmOriginal = (0 ..< (2 * 4 * 8)).map { Float($0) * 0.25 }
            cache.ssm.h.copyIn(from: ssmOriginal)
            let convOriginal = (0 ..< (3 * 12)).map { Float($0) * -0.6 }
            cache.conv.state.copyIn(from: convOriginal)
            cache.advance()
            cache.advance()
            cache.advance()
            #expect(cache.length == 3)

            let cmd = Device.shared.makeCommandBuffer()
            let snap = cache.snapshotEncode(on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            // Pollute everything, including the step counter.
            cache.ssm.h.copyIn(from: [Float](repeating: 42, count: 2 * 4 * 8))
            cache.conv.state.copyIn(from: [Float](repeating: -42, count: 3 * 12))
            cache.advance()
            cache.advance()
            #expect(cache.length == 5)

            let rCmd = Device.shared.makeCommandBuffer()
            cache.restoreEncode(ssm: snap.ssm, conv: snap.conv, on: rCmd)
            rCmd.commit()
            rCmd.waitUntilCompleted()
            cache.setLength(3)

            #expect(cache.ssm.h.toArray(as: Float.self) == ssmOriginal)
            #expect(cache.conv.state.toArray(as: Float.self) == convOriginal)
            #expect(cache.length == 3)
        }
    }

    // MARK: - Through the composite CacheSnapshot.swift dispatch

    @Test(
        "CacheSnapshot.snapshotAll/restoreAll round-trips a Mamba2LayerCache mixed with a KVCache and a StatelessLayerCache"
    )
    func cacheSnapshotAllRestoreAllMamba2() {
        autoreleasepool {
            let mamba = Mamba2LayerCache(
                nHeads: 2, stateDim: 4, headDim: 8,
                convChannels: 12, convKernelSize: 4, dtype: .f32)
            let kv = KVCache(nKVHeads: 2, headDim: 8, maxSeq: 16, dtype: .f32)
            // A real NemotronH `caches` array interleaves stateless
            // MLP/MoE slots between the stateful mamba/attention ones —
            // include one here so this test also covers the
            // `.stateless` snapshot case (F-131 found this the hard
            // way: `snapshotAll`'s catch-all `preconditionFailure` fired
            // the first time this driver ran against a real checkpoint,
            // because `StatelessLayerCache` had no case at all).
            let stateless = StatelessLayerCache()
            let caches: [any LayerCacheProtocol] = [mamba, kv, stateless]

            let ssmOriginal = (0 ..< (2 * 4 * 8)).map { Float($0) * 0.1 + 1 }
            mamba.ssm.h.copyIn(from: ssmOriginal)
            let convOriginal = (0 ..< (3 * 12)).map { Float($0) * 0.2 - 1 }
            mamba.conv.state.copyIn(from: convOriginal)
            mamba.advance()

            let snaps = caches.snapshotAll()

            // Pollute the mamba cache (simulating a rejected speculative
            // token's forward pass through the target).
            mamba.ssm.h.copyIn(from: [Float](repeating: 99, count: 2 * 4 * 8))
            mamba.conv.state.copyIn(from: [Float](repeating: -99, count: 3 * 12))
            mamba.advance()
            mamba.advance()
            #expect(mamba.length == 3)

            caches.restoreAll(from: snaps)

            #expect(mamba.ssm.h.toArray(as: Float.self) == ssmOriginal)
            #expect(mamba.conv.state.toArray(as: Float.self) == convOriginal)
            #expect(mamba.length == 1)
        }
    }

    @Test("CacheSnapshot round-trip across 4 cycles matches a fresh snapshot each time (bitwise)")
    func cacheSnapshotAllMultiCycle() {
        autoreleasepool {
            let mamba = Mamba2LayerCache(
                nHeads: 3, stateDim: 6, headDim: 4,
                convChannels: 9, convKernelSize: 3, dtype: .f32)
            let caches: [any LayerCacheProtocol] = [mamba]
            let ssmCount = 3 * 6 * 4
            let convCount = 2 * 9

            for cycle in 0 ..< 4 {
                let ssmBase = (0 ..< ssmCount).map { Float($0 - cycle) * 0.03 }
                let convBase = (0 ..< convCount).map { Float($0 + cycle) * -0.04 }
                mamba.ssm.h.copyIn(from: ssmBase)
                mamba.conv.state.copyIn(from: convBase)
                mamba.setLength(cycle)

                let snaps = caches.snapshotAll()

                // Multiple polluting mutations within the cycle, as a
                // multi-token draft burst would produce before rejection.
                for step in 0 ..< 2 {
                    mamba.ssm.h.copyIn(
                        from: (0 ..< ssmCount).map { Float($0 * step + 7) })
                    mamba.conv.state.copyIn(
                        from: (0 ..< convCount).map { Float($0 - step - 3) })
                    mamba.advance()
                }

                caches.restoreAll(from: snaps)

                #expect(
                    mamba.ssm.h.toArray(as: Float.self) == ssmBase,
                    "cycle=\(cycle): ssm state must match pre-pollution snapshot")
                #expect(
                    mamba.conv.state.toArray(as: Float.self) == convBase,
                    "cycle=\(cycle): conv state must match pre-pollution snapshot")
                #expect(
                    mamba.length == cycle,
                    "cycle=\(cycle): length must be restored, not just tensor content")
            }
        }
    }
}
