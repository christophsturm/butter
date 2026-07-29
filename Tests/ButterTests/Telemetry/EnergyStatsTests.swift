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
// EnergyStatsTests — the per-task energy counter, the thermal-state
// mapping, and the phase tracker that deltas them. Pure / near-pure;
// no model load required.

import Foundation
import Testing

@testable import ButterSwift

@Suite("EnergyStats")
struct EnergyStatsTests {

    // MARK: - ThermalState

    @Test("ProcessInfo mapping round-trips every case")
    func thermalMapping() {
        #expect(ThermalState(.nominal) == .nominal)
        #expect(ThermalState(.fair) == .fair)
        #expect(ThermalState(.serious) == .serious)
        #expect(ThermalState(.critical) == .critical)
    }

    @Test("Comparable orders by severity, unknown lowest")
    func thermalComparable() {
        #expect(ThermalState.unknown < .nominal)
        #expect(ThermalState.nominal < .fair)
        #expect(ThermalState.fair < .serious)
        #expect(ThermalState.serious < .critical)
        // max() picks the worst — the operation the tracker relies on.
        #expect(max(ThermalState.nominal, .serious) == .serious)
        #expect(max(ThermalState.unknown, .nominal) == .nominal)
    }

    @Test("current() returns a real (non-unknown) state")
    func thermalCurrent() {
        let s = ThermalState.current()
        #expect(s != .unknown)
        #expect(s >= .nominal)
    }

    @Test("ThermalState is Codable as its rawValue string")
    func thermalCodable() throws {
        let data = try JSONEncoder().encode(ThermalState.serious)
        #expect(String(data: data, encoding: .utf8) == "\"serious\"")
        let back = try JSONDecoder().decode(ThermalState.self, from: data)
        #expect(back == .serious)
    }

    // MARK: - EnergySnapshot

    @Test("capture() succeeds and the counter is monotonic")
    func energyCaptureMonotonic() {
        guard let first = EnergySnapshot.capture() else {
            Issue.record("EnergySnapshot.capture() returned nil on Apple Silicon")
            return
        }
        // Burn a little CPU so the kernel counter advances.
        var x = 0.0
        for i in 0..<5_000_000 { x += Double(i) * 0.5 }
        guard let second = EnergySnapshot.capture() else {
            Issue.record("second capture nil (x=\(x))")
            return
        }
        #expect(second.taskEnergyNanojoules >= first.taskEnergyNanojoules)
    }

    // MARK: - EnergyThermalTracker

    @Test("Tracker yields non-negative, non-nil joules across phases")
    func trackerDeltas() {
        let t = EnergyThermalTracker()
        var x = 0.0
        for i in 0..<2_000_000 { x += Double(i) * 0.25 }
        t.endPrefill()
        for i in 0..<2_000_000 { x += Double(i) * 0.25 }
        t.sampleThermal()
        t.endDecode()
        _ = x

        let prefillJ = t.prefillEnergyJoules
        let decodeJ = t.decodeEnergyJoules
        #expect(prefillJ != nil)
        #expect(decodeJ != nil)
        #expect((prefillJ ?? -1) >= 0)
        #expect((decodeJ ?? -1) >= 0)
    }

    @Test("Tracker thermal peak is never below start or end")
    func trackerThermalPeak() {
        let t = EnergyThermalTracker()
        t.sampleThermal()
        t.endPrefill()
        t.endDecode()
        #expect(t.thermalPeak >= t.thermalAtStart)
        #expect(t.thermalPeak >= t.thermalAtEnd)
        #expect(t.thermalAtStart != .unknown)
    }

    @Test("decode energy falls back to baseline when prefill never ran")
    func decodeEnergyWithoutPrefill() {
        // Single-token-prompt shape: endPrefill() is still called in the
        // real loop, but verify the fallback path is well-defined when a
        // postPrefill snapshot is present yet prefill did no work.
        let t = EnergyThermalTracker()
        t.endDecode()  // no endPrefill()
        // decodeEnergyJoules uses baseline when postPrefill is nil.
        #expect(t.decodeEnergyJoules != nil)
        #expect((t.decodeEnergyJoules ?? -1) >= 0)
    }
}
