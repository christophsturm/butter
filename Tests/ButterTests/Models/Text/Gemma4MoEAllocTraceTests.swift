// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// Gemma4MoEAllocTraceTests — #105 per-token transient-allocation
// investigation. Offline: builds a tiny synthetic Gemma4 MoE checkpoint
// via `scratch/make_gemma4_moe_synthetic_checkpoint.py` (random F16
// weights, real tensor names/shapes at knob-controlled tiny dims — never
// touches a real multi-GB checkpoint) and drives it through the real
// `Gemma4MoE.loadModel` / `Gemma4Model.forward(tokenId:)` path used by
// `forwardMulti`'s prefill loop, reading `Device.bufferAllocBytes` /
// `scratchAllocBytes` directly around each call (the same counters
// `Device.allocTraceReport` prints under `BUTTER_ALLOC_TRACE=1`).
//
// `perTokenBytes(...)` sweeps one knob at a time (layers, experts,
// top-k, hidden, moe_intermediate) so the scaling law each term implies
// can be read off directly, without guessing from a static read of
// `Gemma4MoEFFN.forward` (which shows only activation-sized vector math
// — the actual volume, if any, must come from further down the stack).
//
// KV-cache sizing is deliberately taken OUT of the picture here — every
// run passes an explicit small `maxSeq` to `makeLayerCaches`, so this
// suite isolates the strictly-per-token forward cost from the (separate,
// one-time, NOT part of #105's "per-token" framing) cache-setup cost
// `makeLayerCaches` pays once before generation starts.

import Foundation
import Testing

@testable import ButterSwift

// `.serialized`: every test in this suite reads `Device.shared`'s raw
// `bufferAllocBytes`/`scratchAllocBytes` counters around a reset/measure
// window. Those counters are plain (non-atomic) `public var`s on a
// process-wide singleton — Swift Testing parallelizes test functions
// within a suite by default, so two tests racing on the same counters
// would produce exactly the kind of garbage, non-reproducible readings
// this suite exists to avoid. Matches the `.serialized` convention
// `Tests/ModelIntegrationTests` already uses for shared-Device suites.
@Suite("Gemma4 MoE alloc-trace (#105, synthetic checkpoint, offline)", .serialized)
struct Gemma4MoEAllocTraceTests {

    struct Knobs {
        var hidden = 256
        var numLayers = 2
        var numExperts = 4
        var topK = 2
        var moeIntermediate = 512
        var vocab = 1024
        var maxPositionEmbeddings = 4096
        var slidingWindow = 128

        var args: [String] {
            [
                "--hidden", "\(hidden)",
                "--num-layers", "\(numLayers)",
                "--num-experts", "\(numExperts)",
                "--top-k", "\(topK)",
                "--moe-intermediate", "\(moeIntermediate)",
                "--vocab", "\(vocab)",
                "--max-position-embeddings", "\(maxPositionEmbeddings)",
                "--sliding-window", "\(slidingWindow)",
            ]
        }
    }

    struct Measurement: Equatable {
        /// Per-token total transient bytes (`bufferAllocBytes +
        /// scratchAllocBytes`), one entry per generated token.
        let perTokenBytes: [Int]
    }

    /// Absolute path to the committed generator script, resolved via
    /// `#filePath` so it works regardless of the runner's cwd.
    static var generatorScript: String {
        URL(fileURLWithPath: "\(#filePath)")
            .deletingLastPathComponent()  // Gemma4MoEAllocTraceTests.swift -> Text/
            .deletingLastPathComponent()  // Text/ -> Models/
            .deletingLastPathComponent()  // Models/ -> ButterTests/
            .deletingLastPathComponent()  // ButterTests/ -> Tests/
            .deletingLastPathComponent()  // Tests/ -> repo root
            .appendingPathComponent("scratch/make_gemma4_moe_synthetic_checkpoint.py")
            .path
    }

    /// Shells out to the python3 generator and loads the resulting
    /// checkpoint through Butter's real Gemma4MoE loader — no tokenizer,
    /// `Gemma4MoE.loadModel` directly (`Model.load` isn't needed; we
    /// drive `forward(tokenId:)` with raw integer ids < vocab_size).
    /// Caller owns the temp directory's cleanup via the returned closure
    /// isn't provided — callers that only need one model per test can
    /// just let the directory leak into `/tmp` (a few MB, fine for a
    /// unit test); this suite always removes it via `defer`.
    func loadSyntheticModel(_ knobs: Knobs) throws -> Gemma4Model {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-moe-synth-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = ["python3", Self.generatorScript, "--out-dir", dir.path] + knobs.args
        let stderrPipe = Pipe()
        proc.standardError = stderrPipe
        try proc.run()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else {
            let err =
                String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)
                ?? ""
            Issue.record("synthetic checkpoint generator failed: \(err)")
            throw CocoaError(.fileReadUnknown)
        }

        let device = Device.shared
        let config = try ModelConfig.load(from: dir)
        let weights = try SafeTensorsBundle(directory: dir, device: device)
        return try Gemma4MoE.loadModel(
            config: config, weights: weights, options: LoadOptions(), device: device)
    }

    /// Measures per-token transient allocation strictly WITHIN
    /// `forward(tokenId:)` — cache setup happens once beforehand (with
    /// an explicit small `maxSeq`, so it can't leak into the per-token
    /// numbers below; see `cacheSetupScalesWithMaxPositionEmbeddings`
    /// for that cost measured on its own).
    func measure(_ knobs: Knobs, promptLen: Int = 3) throws -> Measurement {
        let device = Device.shared
        let model = try loadSyntheticModel(knobs)
        let caches = model.makeLayerCaches(maxSeq: 64, device: device)

        var perTokenBytes: [Int] = []
        for i in 0 ..< promptLen {
            let tok = i % knobs.vocab
            device.bufferAllocCount = 0
            device.bufferAllocBytes = 0
            device.scratchAllocCount = 0
            device.scratchAllocBytes = 0
            autoreleasepool {
                let cmd = device.makeCommandBuffer()
                _ = model.forward(
                    tokenId: tok, position: i, caches: caches, on: cmd, device: device)
                cmd.commit()
                cmd.waitUntilCompleted()
            }
            perTokenBytes.append(device.bufferAllocBytes + device.scratchAllocBytes)
        }
        return Measurement(perTokenBytes: perTokenBytes)
    }

    // ─── Pre-flight arithmetic (hard safety rule) ─────────────────────
    //
    // Baseline knobs: hidden=256, 2 layers, 4 experts (top-2),
    // moe_intermediate=512, vocab=1024. Every weight tensor is at most
    // `numExperts * moeIntermediate * hidden * 2 bytes` = 4*512*256*2 =
    // ~1 MB; the whole checkpoint is ~9 MB (see the generator's own
    // printed total). Even a 1000x per-token amplification bug would
    // land in the tens-of-MB range — trivially under any free-memory
    // floor on this machine. The largest sweep knob below (numLayers=8,
    // numExperts=16) still keeps the full checkpoint under ~100 MB.
    // No `--stats`/vm_stat check is meaningful at this scale; this
    // comment documents that the arithmetic was done, not skipped.

    /// The shape check `perTokenStability` asserts, factored out so
    /// `retryUntilAcceptable` can retry until an attempt already
    /// satisfies it. Deliberately the SAME logic as the `#expect`s
    /// below (not a looser or stricter variant) — this isn't widening
    /// a tolerance, it's letting an already-tolerant check decide when
    /// to stop retrying, instead of requiring two raw measurement
    /// vectors to agree (see AllocMeasurementRetry.swift: under CI's
    /// ~285-suite concurrent load, distinct attempts get hit by
    /// distinct foreign allocations, so raw-vector agreement converges
    /// far more slowly than "is this attempt's shape already fine").
    private func isStableShape(_ m: Measurement) -> Bool {
        guard m.perTokenBytes.count == 8 else { return false }
        let counts = m.perTokenBytes
        let strictlyIncreasing = zip(counts, counts.dropFirst()).allSatisfy { $0 < $1 }
        guard !strictlyIncreasing else { return false }
        let tail = Array(counts.suffix(4)).map(Double.init)
        let mean = tail.reduce(0, +) / Double(tail.count)
        let spread = (tail.max()! - tail.min()!)
        return spread <= mean * 0.25
    }

    @Test("baseline: per-token bytes are stable across tokens (no intra-token leak)")
    func perTokenStability() throws {
        // `retryUntilAcceptable`: see AllocMeasurementRetry.swift and
        // `isStableShape` above. Retries until an attempt's OWN shape
        // already satisfies the assertions below, rather than
        // requiring two raw measurement vectors to match — under CI's
        // ~285-suite concurrent load, per-call foreign-allocation
        // bytes differ attempt to attempt, so raw-vector agreement is
        // a much harder bar to clear than "already within the
        // tolerated shape" and was observed to exhaust retries in CI
        // run 31800387587 even though the exact `stableMeasurement`
        // value-equality approach remains correct for the short,
        // single-window exact-slope test below.
        let m = try retryUntilAcceptable(maxAttempts: 8, isAcceptable: isStableShape) {
            try measure(Knobs(), promptLen: 8)
        }
        #expect(m.perTokenBytes.count == 8)
        for (i, b) in m.perTokenBytes.enumerated() {
            print("[alloc-trace] token\(i): \(b) bytes")
        }
        // The risk under test is a LEAK: per-token allocation growing as
        // generation proceeds. Byte-IDENTITY across tokens turned out to
        // be a dev-box-specific property — CI's Metal allocator jitters
        // per-call byte counts (bucket rounding) even in steady state,
        // in both directions. A leak is distinguishable from jitter by
        // its shape: it accumulates, so counts strictly increase token
        // over token, and the spread grows without bound. Assert the
        // negation of that shape instead of identity:
        //   1. counts are NOT strictly increasing across the whole run
        //   2. tail spread stays small relative to the tail mean
        let counts = m.perTokenBytes
        let strictlyIncreasing = zip(counts, counts.dropFirst()).allSatisfy { $0 < $1 }
        #expect(
            !strictlyIncreasing,
            "per-token bytes strictly increasing — leak shape: \(counts)")
        let tail = Array(counts.suffix(4)).map(Double.init)
        let mean = tail.reduce(0, +) / Double(tail.count)
        let spread = (tail.max()! - tail.min()!)
        #expect(
            spread <= mean * 0.25,
            "steady-state tail spread \(spread) exceeds 25% of mean \(mean): \(counts)")
    }

    @Test(
        "scaling law: sweep layers / experts / top-k / hidden / moe_intermediate",
        arguments: [
            ("baseline", Knobs()),
            ("2x layers", Knobs(numLayers: 4)),
            ("4x layers", Knobs(numLayers: 8)),
            ("2x hidden", Knobs(hidden: 512)),
            ("2x moe_intermediate", Knobs(moeIntermediate: 1024)),
            (
                "2x numExperts, same topK (tests total-experts vs activated-experts scaling)",
                Knobs(numExperts: 8, topK: 2)
            ),
            ("2x topK, matching numExperts", Knobs(numExperts: 8, topK: 4)),
        ]
    )
    func scalingLaw(_ label: String, _ knobs: Knobs) throws {
        let m = try measure(knobs, promptLen: 1)
        let bytes = m.perTokenBytes.first ?? -1
        print(
            "[alloc-trace] \(label): hidden=\(knobs.hidden) layers=\(knobs.numLayers) "
                + "experts=\(knobs.numExperts) topK=\(knobs.topK) "
                + "moeIntermediate=\(knobs.moeIntermediate) -> \(bytes) bytes/token")
        #expect(bytes >= 0)
    }

    // ─── KV-cache setup: the actual dominant term ──────────────────────
    //
    // `scalingLaw` above shows the naive MoE-forward math is small and
    // scales exactly the way a static read of `Gemma4MoEFFN.forward`
    // predicts (linear in layers / topK / hidden / moe_intermediate,
    // FLAT in total numExperts). Projected to the real 26B-A4B's
    // dimensions (30 layers, hidden 2816, moe_intermediate 704, topK 8)
    // that naive-forward term comes out to ~10 MB/token — four to five
    // orders of magnitude short of the ~75-90GB "per-token" figure
    // #105 set out to explain.
    //
    // `Gemma4Model.makeLayerCaches` (Gemma4Text.swift) is the actual
    // dominant term, and it isn't in the MoE forward path at all — it's
    // one-time, called once before the generation loop starts, which is
    // almost certainly why the earlier per-token RSS sampling attributed
    // its cost to "the first token" and the commit message called it
    // "per-token": for EVERY sliding-window layer it allocates the
    // KVCache's PHYSICAL buffer at the full `maxSeq` cap, even though
    // the `.window(maxSize: min(slidingWindow, cap))` eviction policy
    // computed on the very next line guarantees the layer never reads or
    // writes past `slidingWindow` slots. On the real checkpoint
    // (max_position_embeddings=262144, sliding_window=1024, 25 of 30
    // layers sliding) that one-line oversight allocates a K/V buffer
    // 256x larger than the layer will ever use.
    @Test("KV cache setup scales with max_position_embeddings, not sliding_window (the real #105 term)")
    func cacheSetupScalesWithMaxPositionEmbeddings() throws {
        let device = Device.shared
        let sweeps: [(label: String, mpe: Int)] = [
            ("mpe=512", 512), ("mpe=2048", 2048), ("mpe=8192", 8192),
        ]
        // `stableMeasurement`: see AllocMeasurementRetry.swift. This
        // test asserts an EXACT slope (a regression gate, not a shape
        // check) — widening its tolerance would defeat the point of
        // the assertion, so the fix for cross-suite Device.shared
        // interference has to happen around the measurement instead.
        // Retrying the WHOLE 3-point sweep as one unit (rather than
        // each point independently) until ANY two of up to 8 full
        // sweeps agree byte-for-byte on all three points catches
        // interference landing in any one of the three windows — each
        // sweep's own window is short, so any-two-of-8 converges even
        // under CI's ~285-suite concurrent load.
        let bytesByMpe: [Int] = try stableMeasurement(maxAttempts: 8) {
            try sweeps.map { (label, mpe) -> Int in
                let knobs = Knobs(maxPositionEmbeddings: mpe, slidingWindow: 128)
                let model = try loadSyntheticModel(knobs)
                device.bufferAllocCount = 0
                device.bufferAllocBytes = 0
                device.scratchAllocCount = 0
                device.scratchAllocBytes = 0
                // Default maxSeq (nil) -> Gemma4Model.maxSeq -> config's
                // max_position_embeddings, exactly what `GenerateCommand`'s
                // `m.engine.makeLayerCaches()` (no explicit maxSeq) does.
                _ = model.makeLayerCaches(maxSeq: nil, device: device)
                let bytes = device.bufferAllocBytes + device.scratchAllocBytes
                print("[alloc-trace] makeLayerCaches sliding_window=128, \(label) -> \(bytes) bytes")
                return bytes
            }
        }
        let results: [(String, Int)] = zip(sweeps.map(\.label), bytesByMpe).map { ($0, $1) }
        // Regression gate. The default 2-layer synthetic config is 1
        // sliding + 1 global layer, both with the same K/V-head shape
        // (nKVHeads=1, headDim=128). Only the GLOBAL layer's physical
        // buffer should scale with max_position_embeddings (`.unbounded`
        // genuinely needs the full cap); the sliding layer's buffer is
        // fixed at `sliding_window` regardless of `mpe`. So bytes(mpe)
        // should be an exactly LINEAR function of mpe with slope
        // `2 (K+V) * nGlobalKVHeads(1) * headDim(128) * 2 bytes (f16)` =
        // 512 bytes per +1 mpe unit. Before the fix, the sliding layer's
        // buffer scaled with mpe too, roughly DOUBLING that slope (both
        // layers share the same head shape here) — so a regression back
        // to the bug would show up as slope ≈ 1024, not 512.
        let bytesByLabel = Dictionary(uniqueKeysWithValues: results)
        let b512 = bytesByLabel["mpe=512"]!
        let b2048 = bytesByLabel["mpe=2048"]!
        let b8192 = bytesByLabel["mpe=8192"]!
        #expect(b512 > 0)
        let slope1 = Double(b2048 - b512) / Double(2048 - 512)
        let slope2 = Double(b8192 - b2048) / Double(8192 - 2048)
        print("[alloc-trace] slope 512->2048 = \(slope1) bytes/mpe-unit, 2048->8192 = \(slope2)")
        #expect(
            abs(slope1 - slope2) < 1.0,
            "slope should be constant (linear in mpe); got \(slope1) then \(slope2)")
        let expectedSlope = 2.0 * 1.0 * 128.0 * 2.0  // (K+V) * nGlobalKVHeads * headDim * f16-bytes
        let message =
            "slope \(slope1) != expected \(expectedSlope) bytes/mpe-unit (only the global/"
            + "unbounded layer should scale with max_position_embeddings) — the sliding "
            + "layer's physical KV buffer may be scaling with mpe again (#105 regression)"
        #expect(abs(slope1 - expectedSlope) < 1.0, "\(message)")
    }
}
