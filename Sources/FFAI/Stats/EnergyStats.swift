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
// EnergyStats — energy + thermal accounting captured at phase
// boundaries, the two axes the cross-framework Apple-Silicon benchmarks
// (john-rocky/apple-silicon-llm-bench) report that raw tok/s hides.
//
// Energy is read from the kernel's per-task power accounting via
// `task_info(TASK_POWER_INFO_V2)`. On Apple Silicon the `task_energy`
// field is the cumulative energy (nanojoules) the kernel attributes to
// this process — CPU + GPU work folded into one counter. This is a plain
// Mach call. We sample the cumulative counter at each phase boundary and 
// take deltas, exactly like the GPU-memory tracker.
//
// Thermal is `ProcessInfo.thermalState` — a cached property the OS
// updates on notifications, so reading it per token is near-free. We
// keep start / peak / end so a run that throttles mid-decode is visible
// rather than averaged away.

import Foundation

// MARK: - Thermal state

/// Portable mirror of `ProcessInfo.ThermalState`, plus an `.unknown`
/// sentinel for stats built without a live capture (hand-constructed
/// rows, the empty-prompt path). `Comparable` by severity so a tracker
/// can keep the worst (peak) state seen across a run.
public enum ThermalState: String, Sendable, Codable, Comparable {
    case nominal
    case fair
    case serious
    case critical
    case unknown

    /// Severity ordering used for peak tracking. `.unknown` sorts below
    /// everything so any real reading replaces it as the peak.
    private var severity: Int {
        switch self {
        case .unknown: return -1
        case .nominal: return 0
        case .fair: return 1
        case .serious: return 2
        case .critical: return 3
        }
    }

    public static func < (lhs: ThermalState, rhs: ThermalState) -> Bool {
        lhs.severity < rhs.severity
    }

    /// Map the Foundation enum into our portable case.
    public init(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal: self = .nominal
        case .fair: self = .fair
        case .serious: self = .serious
        case .critical: self = .critical
        @unknown default: self = .unknown
        }
    }

    /// The live process thermal state right now.
    public static func current() -> ThermalState {
        ThermalState(ProcessInfo.processInfo.thermalState)
    }
}

// MARK: - Energy snapshot

/// One reading of the kernel's cumulative per-task energy counter.
public struct EnergySnapshot: Sendable, Equatable {
    /// `task_power_info_v2.task_energy` — cumulative nanojoules the
    /// kernel attributes to this process since launch.
    public let taskEnergyNanojoules: UInt64

    /// Capture the current counter, or `nil` if the Mach call fails
    /// (e.g. an unsupported platform). Callers treat `nil` as
    /// "energy unavailable" and surface no J/token rather than a zero.
    public static func capture() -> EnergySnapshot? {
        var info = task_power_info_v2()
        // TASK_POWER_INFO_V2_COUNT is a `sizeof`-based macro the Swift
        // importer drops, so compute the natural_t-unit count by hand.
        var count = mach_msg_type_number_t(
            MemoryLayout<task_power_info_v2>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_, task_flavor_t(TASK_POWER_INFO_V2), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return EnergySnapshot(taskEnergyNanojoules: info.task_energy)
    }
}

// MARK: - Phase tracker

/// Accumulates energy + thermal readings across one `generate(...)`
/// run, mirroring `PhaseMemoryTracker`'s prefill / decode split. Energy
/// is sampled only at phase boundaries (three Mach calls total — the
/// counter is cumulative, so per-token sampling buys nothing); thermal
/// is sampled per token via `sampleThermal()` to catch a mid-run peak.
public final class EnergyThermalTracker: @unchecked Sendable {
    private let baseline: EnergySnapshot?
    private var postPrefill: EnergySnapshot?
    private var postDecode: EnergySnapshot?

    public let thermalAtStart: ThermalState
    public private(set) var thermalPeak: ThermalState
    public private(set) var thermalAtEnd: ThermalState

    public init() {
        let start = ThermalState.current()
        self.baseline = EnergySnapshot.capture()
        self.thermalAtStart = start
        self.thermalPeak = start
        self.thermalAtEnd = start
    }

    /// Cheap thermal read; keeps the worst state seen so far. Safe to
    /// call at every token boundary.
    public func sampleThermal() {
        let now = ThermalState.current()
        if now > thermalPeak { thermalPeak = now }
    }

    /// Mark the prefill→decode transition.
    public func endPrefill() {
        postPrefill = EnergySnapshot.capture()
    }

    /// Mark the end of decode; also latches the final thermal reading.
    public func endDecode() {
        postDecode = EnergySnapshot.capture()
        let end = ThermalState.current()
        thermalAtEnd = end
        if end > thermalPeak { thermalPeak = end }
    }

    /// Joules consumed during prefill (`baseline → postPrefill`), or
    /// `nil` if either reading was unavailable.
    public var prefillEnergyJoules: Double? {
        Self.deltaJoules(from: baseline, to: postPrefill)
    }

    /// Joules consumed during decode (`postPrefill → postDecode`). Falls
    /// back to `baseline` when prefill never ran (single-token prompt).
    public var decodeEnergyJoules: Double? {
        Self.deltaJoules(from: postPrefill ?? baseline, to: postDecode)
    }

    /// nanojoule delta → joules, guarding against a missing snapshot or
    /// a non-monotonic counter (clamped at 0).
    private static func deltaJoules(
        from start: EnergySnapshot?, to end: EnergySnapshot?
    ) -> Double? {
        guard let start, let end else { return nil }
        let deltaNJ =
            end.taskEnergyNanojoules >= start.taskEnergyNanojoules
            ? end.taskEnergyNanojoules - start.taskEnergyNanojoules
            : 0
        return Double(deltaNJ) * 1e-9
    }
}
