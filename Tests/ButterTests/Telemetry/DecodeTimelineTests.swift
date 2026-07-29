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
// DecodeTimelineTests — percentile + sustained-retention math over a
// per-token wall-clock series. Pure functions, hand-built inputs,
// asserted outputs.

import Foundation
import Testing

@testable import ButterSwift

@Suite("DecodeTimeline")
struct DecodeTimelineTests {

    // 10/20/30/40/50 ms.
    private let sample = [0.010, 0.020, 0.030, 0.040, 0.050]

    // MARK: - Percentiles

    @Test("Empty series → nil")
    func percentileEmpty() {
        #expect(DecodeTimeline.percentileMs([], 0.5) == nil)
    }

    @Test("Single element → that value in ms")
    func percentileSingle() {
        #expect(DecodeTimeline.percentileMs([0.042], 0.99) == 42)
    }

    @Test("p0 / p50 / p100 hit the exact order statistics")
    func percentileExact() {
        #expect(DecodeTimeline.percentileMs(sample, 0.0) == 10)
        #expect(DecodeTimeline.percentileMs(sample, 0.5) == 30)
        #expect(DecodeTimeline.percentileMs(sample, 1.0) == 50)
    }

    @Test("p95 interpolates linearly between neighbours")
    func percentileInterpolates() {
        // rank = 0.95 * (5-1) = 3.8 → 40 + (50-40)*0.8 = 48 ms.
        let p95 = DecodeTimeline.percentileMs(sample, 0.95)
        #expect(p95 != nil)
        #expect(abs((p95 ?? 0) - 48) < 1e-9)
    }

    @Test("Unsorted input is handled (sorts internally)")
    func percentileUnsorted() {
        let shuffled = [0.050, 0.010, 0.040, 0.020, 0.030]
        #expect(DecodeTimeline.percentileMs(shuffled, 0.5) == 30)
    }

    @Test("p out of range clamps to [0,1]")
    func percentileClamps() {
        #expect(DecodeTimeline.percentileMs(sample, -1) == 10)
        #expect(DecodeTimeline.percentileMs(sample, 2) == 50)
    }

    // MARK: - Sustained retention

    @Test("Too few tokens → nil")
    func retentionTooShort() {
        // One below the threshold (warmup 10 + 16 = 26).
        let series = Array(repeating: 0.010, count: 25)
        #expect(DecodeTimeline.sustainedRetention(series) == nil)
    }

    @Test("Steady decode → retention ≈ 1.0")
    func retentionSteady() {
        let series = Array(repeating: 0.010, count: 40)
        let r = DecodeTimeline.sustainedRetention(series)
        #expect(r != nil)
        #expect(abs((r ?? 0) - 1.0) < 1e-9)
    }

    @Test("Late window 2× slower → retention ≈ 0.5")
    func retentionDrift() {
        // 10 warmup (dropped) + 8 early @10ms + 8 late @20ms.
        var series = Array(repeating: 0.005, count: 10)  // warmup, ignored
        series += Array(repeating: 0.010, count: 8)  // early
        series += Array(repeating: 0.020, count: 8)  // late
        let r = DecodeTimeline.sustainedRetention(series)
        #expect(r != nil)
        // early tok/s ÷ late tok/s = (1/0.010)/(1/0.020)... late slower → <1.
        #expect(abs((r ?? 0) - 0.5) < 1e-9)
    }

    @Test("Late window faster → retention > 1.0")
    func retentionSpeedUp() {
        var series = Array(repeating: 0.005, count: 10)
        series += Array(repeating: 0.020, count: 8)  // early slow
        series += Array(repeating: 0.010, count: 8)  // late fast
        let r = DecodeTimeline.sustainedRetention(series)
        #expect((r ?? 0) > 1.5)
    }
}
