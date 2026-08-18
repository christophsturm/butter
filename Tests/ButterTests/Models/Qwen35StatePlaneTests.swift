// Copyright 2026 Eric Kryski (@ekryski)
// SPDX-License-Identifier: Apache-2.0

import Testing

@testable import ButterSwift

@Suite("Qwen35 sparse state planes")
struct Qwen35StatePlaneTests {
    @Test("tail capture retains exactly four rollback targets")
    func tailCaptureWindow() {
        for keep in 1 ... 4 {
            #expect(!Qwen35Model.tailStatePlaneCaptured(keep: keep, tokenCount: 9))
        }
        for keep in 5 ... 8 {
            #expect(Qwen35Model.tailStatePlaneCaptured(keep: keep, tokenCount: 9))
        }
        #expect(!Qwen35Model.tailStatePlaneCaptured(keep: 9, tokenCount: 9))
    }

    @Test("short verification blocks retain every rollback target")
    func shortCaptureWindow() {
        for tokenCount in 2 ... 5 {
            for keep in 1 ..< tokenCount {
                #expect(
                    Qwen35Model.tailStatePlaneCaptured(
                        keep: keep, tokenCount: tokenCount))
            }
        }
    }

    @Test("invalid rollback targets are rejected")
    func invalidTargets() {
        #expect(!Qwen35Model.tailStatePlaneCaptured(keep: 0, tokenCount: 9))
        #expect(!Qwen35Model.tailStatePlaneCaptured(keep: 1, tokenCount: 1))
        #expect(!Qwen35Model.tailStatePlaneCaptured(keep: 10, tokenCount: 9))
    }
}
