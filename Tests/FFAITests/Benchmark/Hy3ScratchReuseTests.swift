// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Verifies tagged intermediateScratch reuses buffers (no fresh makeBuffer
// growth after warm-up) — the free-path pad pattern Hy3 multi uses.
//
//   swift test -c release --filter Hy3ScratchReuse

import Foundation
import Metal
import Testing

@testable import FFAI

@Suite("Hy3 intermediateScratch reuse")
struct Hy3ScratchReuseTests {
    @Test("tagged intermediateScratch reuses buffer after warm-up")
    func scratchReusesBuffer() throws {
        let device = Device.shared
        let tag = "hy3_test_scratch_reuse_\(UUID().uuidString.prefix(8))"
        let bytes = 4 * 1024 * 1024  // 4 MB

        let a = device.intermediateScratch(tag: tag, minBytes: bytes)
        let b = device.intermediateScratch(tag: tag, minBytes: bytes)
        #expect(a === b, "same tag+size must return identical MTLBuffer")

        // Larger request grows; subsequent same-or-smaller reuses.
        let c = device.intermediateScratch(tag: tag, minBytes: bytes * 2)
        #expect(c.length >= bytes * 2)
        let d = device.intermediateScratch(tag: tag, minBytes: bytes)
        #expect(c === d, "grown buffer must be reused for smaller request")
    }
}
