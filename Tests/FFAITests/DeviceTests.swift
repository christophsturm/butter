// Copyright 2026 Eric Kryski (@ekryski)
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
import Testing

@testable import FFAI

@Suite("Device")
struct DeviceTests {
    @Test("shared device + queue available")
    func shared() {
        let d = Device.shared
        #expect(d.mtlDevice.name.count > 0)
        // commandQueue is non-optional after construction
        _ = d.commandQueue
    }

    @Test("makeBuffer allocates requested length")
    func makeBuffer() {
        let buf = Device.shared.makeBuffer(length: 1024)
        #expect(buf.length >= 1024)
    }

    @Test("makeCommandBuffer returns a usable buffer")
    func makeCommandBuffer() {
        let cb = Device.shared.makeCommandBuffer()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)
    }

    // ─── Scratch slab allocator ──────────────────────────────────────
    //
    // Use an isolated `Device` (same MTLDevice + queue as `.shared`) so
    // these tests never mutate `Device.shared`'s scratch state, which
    // other parallel suites allocate against.

    private func isolatedDevice() -> Device {
        Device(mtlDevice: Device.shared.mtlDevice, commandQueue: Device.shared.commandQueue)
    }

    @Test("allocScratch returns 16-byte-aligned bumping offsets")
    func allocScratchAlignsAndBumps() {
        let d = isolatedDevice()
        let (b0, o0) = d.allocScratch(bytes: 100)
        let (b1, o1) = d.allocScratch(bytes: 32)
        // Same backing slab for both slices.
        #expect(b0 === b1)
        #expect(o0 == 0)
        // First slice was 100 bytes → next offset rounds up to 112 (16-aligned).
        #expect(o1 == 112)
        #expect(o1 % 16 == 0)
    }

    @Test("allocScratch updates diagnostic counters")
    func allocScratchCounters() {
        let d = isolatedDevice()
        #expect(d.scratchAllocCount == 0)
        #expect(d.scratchAllocBytes == 0)
        _ = d.allocScratch(bytes: 64)
        _ = d.allocScratch(bytes: 128)
        #expect(d.scratchAllocCount == 2)
        #expect(d.scratchAllocBytes == 192)
    }

    @Test("resetScratch rewinds the slab offset to 0")
    func resetScratchRewinds() {
        let d = isolatedDevice()
        let (_, first) = d.allocScratch(bytes: 256)
        #expect(first == 0)
        _ = d.allocScratch(bytes: 256)
        d.resetScratch()
        // After reset the next allocation starts back at offset 0.
        let (_, afterReset) = d.allocScratch(bytes: 16)
        #expect(afterReset == 0)
    }

    @Test("withScratch activates scratch mode and resets on exit")
    func withScratchScope() {
        let d = isolatedDevice()
        #expect(d.scratchModeActive == false)
        let observed = d.withScratch { () -> Bool in
            _ = d.allocScratch(bytes: 64)
            return d.scratchModeActive
        }
        #expect(observed == true)
        // Scope exit restored the flag and rewound the slab.
        #expect(d.scratchModeActive == false)
        let (_, offset) = d.allocScratch(bytes: 16)
        #expect(offset == 0)
    }

    @Test("nested withScratch does not reset the outer scope's slab")
    func withScratchNested() {
        let d = isolatedDevice()
        d.withScratch {
            _ = d.allocScratch(bytes: 64)  // outer offset now 64
            d.withScratch {
                // Inner scope sees scratch already active; it must NOT
                // reset on exit (the outer scope still owns the slab).
                #expect(d.scratchModeActive == true)
                _ = d.allocScratch(bytes: 32)
            }
            // Outer slab still has the inner allocation accounted for —
            // the next slice lands after both (64 + 32 → 16-aligned 96).
            #expect(d.scratchModeActive == true)
            let (_, offset) = d.allocScratch(bytes: 16)
            #expect(offset == 96)
        }
        #expect(d.scratchModeActive == false)
    }

    @Test("ensureScratchSlab grows the slab when no slices are live")
    func ensureScratchSlabGrows() {
        let d = isolatedDevice()
        d.ensureScratchSlab(8 * 1024 * 1024)
        let (buf, _) = d.allocScratch(bytes: 16)
        #expect(buf.length >= 8 * 1024 * 1024)
    }

    @Test("scalarBuffer caches one buffer per value")
    func scalarBufferCaches() {
        let d = isolatedDevice()
        let a = d.scalarBuffer(1.5)
        let b = d.scalarBuffer(1.5)
        let c = d.scalarBuffer(2.5)
        // Same value → same cached buffer; different value → distinct.
        #expect(a === b)
        #expect(a !== c)
        // Stored value is the 4-byte little-endian Float.
        let stored = a.contents().bindMemory(to: Float.self, capacity: 1).pointee
        #expect(stored == 1.5)
    }
}
