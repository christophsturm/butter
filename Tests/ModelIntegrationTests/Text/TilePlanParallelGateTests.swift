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
// F-85 parallelized tile-plan builder e2e gate - sweeps the batched
// forwardMany prefill path across the T ladder (512/1024/2048/4096/
// 16384/32768), reporting median tok/s per T. Reads
// FFAI_MOE_PARALLEL_PLAN from the process environment ONCE (same as
// production `MoELayer` init) and runs ONE config per invocation - run
// this test TWICE, once with the default environment (parallel ON) and
// once with FFAI_MOE_PARALLEL_PLAN=0 set BEFORE `swift test` starts, to
// compare. A single process loading the SAME config twice (to A/B
// within one run) was tried and rejected: the second `Model.load` call
// started before the first model's Metal-resident buffers had actually
// been released, briefly holding two ~35B-parameter models resident and
// pushing this shared M5 Max to under 2GB free physical memory. Two
// separate process launches (this file's actual shape) guarantees full
// OS-level reclamation between configs - matches the established
// per-config-process convention already used for other F-85 gates
// (e.g. `FFAI_MOE_COUNTING_SORT=0 swift test --filter ...`).
//
// GATE_LADDER_MAX_T caps the ladder to `targetT <= GATE_LADDER_MAX_T`
// for incremental validation before committing wall-clock/memory to
// the largest sentinels (16384/32768).

import Foundation
import Metal
import TestHelpers
import Testing

@testable import FFAI

private let gateLocalPath = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
private let gateCheckpointAvailable =
    FileManager.default.fileExists(atPath: gateLocalPath)

@Suite(
    "F-85 Parallel Tile-Plan Builder Gate", .serialized,
    .enabled(
        if: gateCheckpointAvailable,
        "requires a local checkpoint at \(gateLocalPath)")
)
struct TilePlanParallelGateTests {

    /// One (T, trialCount) sweep point in the ladder.
    private struct Point {
        let targetT: Int
        let trials: Int
    }

    private static let fullLadder: [Point] = [
        Point(targetT: 512, trials: 7),
        Point(targetT: 1024, trials: 7),
        Point(targetT: 2048, trials: 3),
        Point(targetT: 4096, trials: 3),
        Point(targetT: 16384, trials: 3),
        Point(targetT: 32768, trials: 3),
    ]

    /// GATE_LADDER_MAX_T / GATE_LADDER_MIN_T bound the ladder to
    /// `minT <= targetT <= maxT`. Running one (T, trials) point per
    /// process (min == max) is the safe default after repeated ladders
    /// within one process were found to accumulate memory across T
    /// values (each T's KV cache / activation buffers were not fully
    /// released before the next T's allocation, independent of the
    /// per-iteration `autoreleasepool` already in the trial loop below) -
    /// a single loaded model swept multiple T values close to OOMed this
    /// shared M5 Max twice. One T per process guarantees a clean process
    /// exit reclaims everything before the next point runs.
    private static var ladder: [Point] {
        let env = ProcessInfo.processInfo.environment
        let maxT = env["GATE_LADDER_MAX_T"].flatMap { Int($0) } ?? Int.max
        let minT = env["GATE_LADDER_MIN_T"].flatMap { Int($0) } ?? 0
        // GATE_TRIALS_OVERRIDE caps trial count at the largest T values -
        // fewer large-buffer allocate/free cycles lowers peak memory risk
        // on this shared machine at the cost of a noisier median.
        let trialsOverride = env["GATE_TRIALS_OVERRIDE"].flatMap { Int($0) }
        return fullLadder.filter { $0.targetT >= minT && $0.targetT <= maxT }
            .map { point in
                guard let cap = trialsOverride, cap < point.trials else { return point }
                return Point(targetT: point.targetT, trials: cap)
            }
    }

    @Test("F-85 parallel tile-plan builder - e2e prefill ladder (one config per process)")
    func tilePlanParallelGateLadder() async throws {
        let parallelPlanEnvRaw = ProcessInfo.processInfo.environment["FFAI_MOE_PARALLEL_PLAN"]
        let label =
            parallelPlanEnvRaw == "0"
            ? "single-threadgroup(FFAI_MOE_PARALLEL_PLAN=0)" : "parallel(default)"
        print("=== \(label) ===")

        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(gateLocalPath, options: opts)
        }
        let qwen = try #require(m.qwen35, "expected Qwen35Model engine")

        let seed =
            "The history of the printing press began when European craftsmen of the 15th century combined movable metal type with oil based ink screw presses paper to mass produce printed books pamphlets and broadsheets revolutionising communication"
        let seedEncoded = m.tokenizer.encode(text: seed)

        for point in Self.ladder {
            var encoded = seedEncoded
            while encoded.count < point.targetT {
                encoded.append(contentsOf: seedEncoded)
            }
            encoded = Array(encoded.prefix(point.targetT))
            let T = encoded.count

            // Warm up Metal PSO + first-token JIT before timing, same
            // convention as `runForwardManyBench`. Each iteration's
            // cache/command-buffer allocation and encoding is wrapped in
            // `autoreleasepool` - a tight Swift-concurrency loop never
            // gets the per-runloop-tick autorelease drain a normal event
            // loop would, so the MTLBuffers `forwardMany` allocates while
            // encoding (tile-plan scratch, gathered activations, etc.)
            // accumulate across iterations instead of freeing each one -
            // the documented MTLBuffer/autoreleasepool leak class, worse
            // here than usual because this loop repeats at every T up to
            // 32768.
            for _ in 0 ..< 2 {
                let warmCmd: MTLCommandBuffer = autoreleasepool {
                    let warmCaches = qwen.makeLayerCaches()
                    let warmCmd = Device.shared.makeCommandBuffer()
                    _ = qwen.forwardMany(
                        tokenIds: encoded, startPosition: 0,
                        caches: warmCaches, on: warmCmd, device: Device.shared)
                    warmCmd.commit()
                    return warmCmd
                }
                await warmCmd.awaitCompletion()
            }

            var secs: [Double] = []
            for _ in 0 ..< point.trials {
                var t0 = Date()
                let cmd: MTLCommandBuffer = autoreleasepool {
                    let caches = qwen.makeLayerCaches()
                    let cmd = Device.shared.makeCommandBuffer()
                    t0 = Date()
                    _ = qwen.forwardMany(
                        tokenIds: encoded, startPosition: 0,
                        caches: caches, on: cmd, device: Device.shared)
                    cmd.commit()
                    return cmd
                }
                await cmd.awaitCompletion()
                secs.append(Date().timeIntervalSince(t0))
            }
            secs.sort()
            let median = secs[secs.count / 2]
            let tps = Double(T) / median
            print(
                "GATE \(label) T=\(T) trials=\(point.trials): runs=\(secs.map { String(format: "%.3f", $0) }) median=\(String(format: "%.3f", median))s tps=\(String(format: "%.2f", tps))"
            )
        }
    }
}
