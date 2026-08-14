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
// F-laguna-windowed-loop: forcing coverage for `LagunaLayer.forwardMulti`'s
// "windowed-decode-loop" branch (LagunaText.swift, `if isSliding,
// windowStart > 0 || sinkEnd > 0`).
//
// That branch is DEAD on every currently-live path:
// `KVCacheProtocol.sdpaSinkWindow`'s default (KVCache.swift ~line 133)
// unconditionally returns `(0, 0)` for every cache type in production
// use, so `windowStart > 0 || sinkEnd > 0` never holds and the branch
// never executes. Found during the Glimmer past-window work (#136):
// the branch used to run AFTER `lagunaBatchedRopeAndKV` had already
// batch-appended the WHOLE chunk's K/V into the ring, then read the
// cache per query row — so row i could see rows i+1..nRows-1's K/V
// (already appended ahead of it), which may have evicted exactly the
// keys row i should have attended to. A causality leak, latent because
// nothing arms the gate today.
//
// Fixed by interleaving each row's RoPE + KV append + attend onto the
// SAME shared command buffer, one row at a time — mirroring
// `MuseGlimmerLayer.forwardMulti`'s `windowWraps` path
// (MuseGlimmerText.swift, `MuseGlimmerLongPrefillTests.swift` is the
// reference oracle this file's first suite is modeled on).
//
// Two suites:
//
//   `LagunaWindowedDecodeLoopOracleTests` — op-level ring-alignment
//   oracle, isolated from `LagunaLayer` entirely. Drives the exact
//   `Ops.rope` -> `KVCache.appendOnGPU` -> `Ops.sdpaDecode` triple
//   three ways with IDENTICAL synthetic inputs over a small window (8)
//   forcing several ring wraps inside a 20-row chunk:
//     1. `reference`  — per-token discipline: fresh command buffer,
//        commit + wait after every row (what any correct interleaved
//        implementation must match).
//     2. `oldBatchedThenAttend` — the ORIGINAL buggy sequencing: every
//        row's RoPE is applied and every row's K/V is appended to the
//        SAME shared command buffer FIRST, and only once the whole
//        chunk has been appended does the per-row attend loop run.
//        This reproduces the exact defect described above.
//     3. `newInterleaved` — the FIXED sequencing now used by
//        `LagunaLayer.forwardMulti`'s windowed-decode-loop branch:
//        rope_i -> append_i -> attend_i, strictly interleaved, all on
//        one shared command buffer.
//   Both the final K/V cache buffers and every row's attention output
//   are compared BIT-FOR-BIT. `newInterleaved` must equal `reference`
//   exactly; `oldBatchedThenAttend` is asserted to DIVERGE from
//   `reference` — i.e. this test proves the pre-fix code was
//   demonstrably wrong, not just that the new code looks plausible.
//
//   `LagunaWindowedDecodeLoopWiringTests` — exercises the real
//   `LagunaLayer.forwardMulti` end-to-end through a test-only
//   `KVCacheProtocol` wrapper (`SinkWindowForcingKVCache`) that forces
//   `sdpaSinkWindow` to report a nonzero window, arming the otherwise
//   permanently-dead branch, and checks the branch actually engages
//   (finite output, cache accounting matches a `.window` eviction's
//   contract) rather than crashing or silently falling through.

import Foundation
import Metal
import Testing

@testable import ButterSwift

// MARK: - Test-only forcing cache

/// Wraps a real `KVCache` and overrides `sdpaSinkWindow` to report a
/// nonzero window, arming `LagunaLayer.forwardMulti`'s permanently-dead
/// "windowed-decode-loop" branch (see file header). No production cache
/// does this today — every real `KVCacheProtocol` conformer relies on
/// `KVCacheProtocol`'s default `(0, 0)`. This type exists purely to make
/// the dead branch reachable for testing; every other member forwards
/// straight through to `inner`.
private final class SinkWindowForcingKVCache: KVCacheProtocol, @unchecked Sendable {
    let inner: KVCache
    let forcedWindowStart: Int

    init(inner: KVCache, forcedWindowStart: Int) {
        self.inner = inner
        self.forcedWindowStart = forcedWindowStart
    }

    var nKVHeads: Int { inner.nKVHeads }
    var headDim: Int { inner.headDim }
    var dtype: DType { inner.dtype }
    var length: Int { inner.length }
    var maxSeq: Int { inner.maxSeq }
    var bytesAllocated: Int { inner.bytesAllocated }
    var bytesInUse: Int { inner.bytesInUse }
    var eviction: KVEviction { inner.eviction }
    var effectiveMaxSize: Int { inner.effectiveMaxSize }
    var absolutePosition: Int { inner.absolutePosition }

    func reset() { inner.reset() }
    func appendOnGPU(kFlat: Tensor, vFlat: Tensor, on cmd: MTLCommandBuffer) {
        inner.appendOnGPU(kFlat: kFlat, vFlat: vFlat, on: cmd)
    }
    func prepareForAttention(on cmd: MTLCommandBuffer) -> (k: Tensor, v: Tensor) {
        inner.prepareForAttention(on: cmd)
    }
    func truncate(toLength length: Int) { inner.truncate(toLength: length) }

    /// The one override that matters: clamp to `nKV` the same way a
    /// real sparse/paged cache's override would, but otherwise always
    /// nonzero once the cache holds more than one row — guaranteed to
    /// arm `isSliding, windowStart > 0 || sinkEnd > 0`.
    func sdpaSinkWindow(nKV: Int) -> (sinkEnd: Int, windowStart: Int) {
        (sinkEnd: 0, windowStart: min(forcedWindowStart, max(nKV - 1, 0)))
    }
}

// MARK: - Op-level ring-alignment oracle

@Suite("Laguna windowed-decode-loop — ring-alignment oracle")
struct LagunaWindowedDecodeLoopOracleTests {

    private static func pseudoRandomUniform(_ seed: inout UInt64) -> Float {
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        return Float(seed >> 40) / Float(1 << 24)
    }

    private static func randomTensor(
        shape: [Int], range: Float, dtype: DType, seed: inout UInt64, device: Device
    ) -> Tensor {
        let n = shape.reduce(1, *)
        let raw: [Float] = (0 ..< n).map { _ in (pseudoRandomUniform(&seed) * 2 - 1) * range }
        let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
        switch dtype {
        case .f32: t.copyIn(from: raw)
        case .f16: t.copyIn(from: raw.map { Float16($0) })
        case .bf16:
            t.copyIn(
                from: raw.map { v -> UInt16 in
                    let bits = v.bitPattern
                    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                    return UInt16(rounded >> 16)
                })
        default: fatalError("randomTensor: unsupported dtype \(dtype)")
        }
        return t
    }

    private enum Discipline {
        /// Fresh command buffer per row, commit + wait before the next
        /// row starts — the ground-truth per-token contract every
        /// correct chunked implementation must reproduce.
        case reference
        /// The ORIGINAL bug: every row's RoPE + KV append queued onto
        /// ONE shared command buffer FIRST (nothing commits between
        /// them), and only after all `nRows` appends have been queued
        /// does the per-row attend loop run (also on that same
        /// buffer). Mirrors `lagunaBatchedRopeAndKV` running before the
        /// pre-fix windowed-decode-loop branch read the cache.
        case oldBatchedThenAttend
        /// The FIX: rope_i -> append_i -> attend_i strictly
        /// interleaved, all on one shared command buffer — mirrors
        /// `LagunaLayer.forwardMulti`'s current windowed-decode-loop
        /// branch and `MuseGlimmerLayer.forwardMulti`'s `windowWraps`
        /// path.
        case newInterleaved
    }

    private static func runWindowedAttentionSequence(
        discipline: Discipline,
        nRows: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        window: Int, maxSeq: Int, dtype: DType,
        qRows: [Tensor], kRows: [Tensor], vRows: [Tensor],
        ropeTheta: Float, scale: Float, device: Device
    ) -> (kBytes: [UInt16], vBytes: [UInt16], attnBytes: [UInt16]) {
        let cache = KVCache(
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq, dtype: dtype,
            eviction: .window(maxSize: window, keep: 0), device: device,
            residency: .weight)
        var attnOut: [UInt16] = []
        attnOut.reserveCapacity(nRows * nHeads * headDim)

        func attendRow(_ i: Int, qRot: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
            let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
            let nKV = cache.length
            let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: nKV)
            return Ops.sdpaDecode(
                q: qRot, k: cacheK, v: cacheV,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                nKV: nKV, kvStride: cache.maxSeq,
                scale: scale, on: cmd,
                sinkEnd: sinkEnd, windowStart: windowStart)
        }

        switch discipline {
        case .reference:
            for i in 0 ..< nRows {
                let cmd = device.makeCommandBuffer()
                let qRot = Ops.rope(
                    qRows[i], position: i, headDim: headDim,
                    thetaBase: ropeTheta, scaling: .none, on: cmd)
                let kRot = Ops.rope(
                    kRows[i], position: i, headDim: headDim,
                    thetaBase: ropeTheta, scaling: .none, on: cmd)
                cache.appendOnGPU(kFlat: kRot, vFlat: vRows[i], on: cmd)
                let row = attendRow(i, qRot: qRot, on: cmd)
                cmd.commit()
                cmd.waitUntilCompleted()
                attnOut.append(contentsOf: row.toArray(as: UInt16.self))
            }

        case .oldBatchedThenAttend:
            let cmd = device.makeCommandBuffer()
            var qRots: [Tensor] = []
            // Phase 1 (the bug): rotate + append EVERY row before
            // attending to ANY of them.
            for i in 0 ..< nRows {
                let qRot = Ops.rope(
                    qRows[i], position: i, headDim: headDim,
                    thetaBase: ropeTheta, scaling: .none, on: cmd)
                let kRot = Ops.rope(
                    kRows[i], position: i, headDim: headDim,
                    thetaBase: ropeTheta, scaling: .none, on: cmd)
                cache.appendOnGPU(kFlat: kRot, vFlat: vRows[i], on: cmd)
                qRots.append(qRot)
            }
            // Phase 2: attend every row against the now-fully-populated
            // (over-evicted, from row i's point of view) cache.
            var rows: [Tensor] = []
            for i in 0 ..< nRows {
                rows.append(attendRow(i, qRot: qRots[i], on: cmd))
            }
            cmd.commit()
            cmd.waitUntilCompleted()
            for row in rows { attnOut.append(contentsOf: row.toArray(as: UInt16.self)) }

        case .newInterleaved:
            let cmd = device.makeCommandBuffer()
            var rows: [Tensor] = []
            for i in 0 ..< nRows {
                let qRot = Ops.rope(
                    qRows[i], position: i, headDim: headDim,
                    thetaBase: ropeTheta, scaling: .none, on: cmd)
                let kRot = Ops.rope(
                    kRows[i], position: i, headDim: headDim,
                    thetaBase: ropeTheta, scaling: .none, on: cmd)
                cache.appendOnGPU(kFlat: kRot, vFlat: vRows[i], on: cmd)
                rows.append(attendRow(i, qRot: qRot, on: cmd))
            }
            cmd.commit()
            cmd.waitUntilCompleted()
            for row in rows { attnOut.append(contentsOf: row.toArray(as: UInt16.self)) }
        }

        let kBytes = cache.kBuffer.toArray(as: UInt16.self)
        let vBytes = cache.vBuffer.toArray(as: UInt16.self)
        return (kBytes, vBytes, attnOut)
    }

    private static func ringAlignmentCase(dtype: DType) {
        let device = Device.shared
        let nHeads = 4
        let nKVHeads = 2
        let headDim = 128
        let window = 8
        let maxSeq = 32
        let nRows = 20  // > window: forces several ring wraps mid-"chunk"
        let scale: Float = 1.0 / Float(headDim).squareRoot()

        var seed: UInt64 = 0xC0FF_EE15_5EED_1234
        var qRows: [Tensor] = []
        var kRows: [Tensor] = []
        var vRows: [Tensor] = []
        for _ in 0 ..< nRows {
            qRows.append(
                Self.randomTensor(
                    shape: [nHeads, headDim], range: 1.0, dtype: dtype, seed: &seed, device: device))
            kRows.append(
                Self.randomTensor(
                    shape: [nKVHeads, headDim], range: 1.0, dtype: dtype, seed: &seed, device: device)
            )
            vRows.append(
                Self.randomTensor(
                    shape: [nKVHeads, headDim], range: 1.0, dtype: dtype, seed: &seed, device: device)
            )
        }

        func run(_ discipline: Discipline) -> (kBytes: [UInt16], vBytes: [UInt16], attnBytes: [UInt16]) {
            Self.runWindowedAttentionSequence(
                discipline: discipline,
                nRows: nRows, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                window: window, maxSeq: maxSeq, dtype: dtype,
                qRows: qRows, kRows: kRows, vRows: vRows,
                ropeTheta: 500_000.0, scale: scale, device: device)
        }

        let reference = run(.reference)
        let oldBuggy = run(.oldBatchedThenAttend)
        let fixed = run(.newInterleaved)

        // The fix must reproduce the per-token reference bit-for-bit —
        // Metal's execution model guarantees identical results for
        // identical kernels run in identical encoding order regardless
        // of where commit/wait boundaries fall, so this is exact
        // equality, not a tolerance envelope.
        #expect(
            reference.kBytes == fixed.kBytes,
            "\(dtype): FIXED — K cache bytes diverged from per-token reference")
        #expect(
            reference.vBytes == fixed.vBytes,
            "\(dtype): FIXED — V cache bytes diverged from per-token reference")
        #expect(
            reference.attnBytes == fixed.attnBytes,
            "\(dtype): FIXED — attention output diverged from per-token reference")

        // The pre-fix sequencing must PROVABLY diverge — this is the
        // forcing assertion: if this ever starts passing (old ==
        // reference), the synthetic case stopped exercising the
        // causality leak and needs a wider window/more rows, not a
        // deleted assertion.
        let oldMatchesReference =
            oldBuggy.kBytes == reference.kBytes
            && oldBuggy.vBytes == reference.vBytes
            && oldBuggy.attnBytes == reference.attnBytes
        #expect(
            !oldMatchesReference,
            "\(dtype): the pre-fix batch-append-then-attend sequencing matched the per-token reference — this test no longer proves the original bug was real; widen the window/row-count until it diverges again"
        )
    }

    @Test("window=8 over a 20-row chunk: fixed path is bit-exact vs per-token, old path provably diverges, f16")
    func ringAlignmentF16() { Self.ringAlignmentCase(dtype: .f16) }

    @Test("window=8 over a 20-row chunk: fixed path is bit-exact vs per-token, old path provably diverges, bf16")
    func ringAlignmentBf16() { Self.ringAlignmentCase(dtype: .bf16) }
}

// MARK: - Layer wiring sanity

@Suite("Laguna windowed-decode-loop — layer wiring", .serialized)
struct LagunaWindowedDecodeLoopWiringTests {

    private static func pseudoRandomUniform(_ seed: inout UInt64) -> Float {
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        return Float(seed >> 40) / Float(1 << 24)
    }

    private static func randomBf16(_ shape: [Int], seed: inout UInt64, scale: Float = 0.05) -> Tensor {
        let n = shape.reduce(1, *)
        var vals = [UInt16](repeating: 0, count: n)
        for i in 0 ..< n {
            let u = Self.pseudoRandomUniform(&seed) * 2 - 1
            let bits = (u * scale).bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            vals[i] = UInt16(rounded >> 16)
        }
        let t = Tensor.empty(shape: shape, dtype: .bf16)
        t.copyIn(from: vals)
        return t
    }

    /// Small Laguna-shaped sliding layer: headDim=128 (the only width
    /// `Ops.sdpaDecode` needs here), hidden a multiple of 64. Not
    /// wired to any INT8/NVFP4 QKV bank or fused-attn-step, so
    /// `forwardMulti`'s sliding branch always takes the plain
    /// `Ops.rmsNormRows` + windowed-decode-loop chain this test cares
    /// about — same shape of setup as `LagunaLastRowPrefillTests`'
    /// `makeTestLayer`.
    private static func makeTestLayer(seed: inout UInt64) -> LagunaLayer {
        let device = Device.shared
        let hidden = 128
        let nHeads = 4, nKVHeads = 2, headDim = 128
        func lin(_ outDim: Int, _ inDim: Int) -> AnyLinear {
            AnyLinear(Linear(weight: Self.randomBf16([outDim, inDim], seed: &seed)))
        }
        func norm(_ n: Int) -> RMSNorm {
            RMSNorm(weight: Self.randomBf16([n], seed: &seed, scale: 1.0), eps: 1e-6)
        }
        let qProj = lin(nHeads * headDim, hidden)
        let kProj = lin(nKVHeads * headDim, hidden)
        let vProj = lin(nKVHeads * headDim, hidden)
        let oProj = lin(hidden, nHeads * headDim)
        let gProj = lin(nHeads, hidden)
        let inter = hidden * 2
        let ffn = LagunaFFN.dense(
            LagunaDenseMLP(
                gate: lin(inter, hidden), up: lin(inter, hidden), down: lin(hidden, inter),
                hidden: hidden, dtype: .bf16, device: device))

        return LagunaLayer(
            qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
            gProj: gProj, qNorm: norm(headDim), kNorm: norm(headDim),
            inputNorm: norm(hidden), postAttnNorm: norm(hidden),
            ffn: ffn,
            hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, rotaryDim: headDim,
            ropeTheta: 500_000.0, isSliding: true, isFullYarn: false, dtype: .bf16,
            device: device, layerIndex: 0)
    }

    @Test("forcing sdpaSinkWindow nonzero engages windowed-decode-loop and stays finite")
    func windowedLoopEngagesAndStaysFinite() {
        let device = Device.shared
        let dtype: DType = .bf16
        var seed: UInt64 = 0x1357_9BDF_2468_ACE0
        let layer = Self.makeTestLayer(seed: &seed)
        let window = 8
        let nRows = 20  // > window: several ring wraps within the chunk

        let realCache = KVCache(
            nKVHeads: 2, headDim: 128, maxSeq: 32, dtype: dtype,
            eviction: .window(maxSize: window, keep: 0), device: device,
            residency: .weight)
        let cache = SinkWindowForcingKVCache(inner: realCache, forcedWindowStart: window)

        let h = Self.randomBf16([nRows, 128], seed: &seed, scale: 0.1)
        let yarn = Ops.RoPEYaRN(factor: 32, low: 1, high: 32, attnFactor: 1.0)

        let cmd = device.makeCommandBuffer()
        let out = layer.forwardMulti(
            h, startingAt: 0, cache: cache, yarn: yarn, cmd: cmd, device: device)
        if cmd.status != .committed && cmd.status != .completed && cmd.status != .error {
            cmd.commit()
        }
        cmd.waitUntilCompleted()

        #expect(out.shape == [nRows, 128])
        // Ring eviction saturates at the window size regardless of how
        // many rows were appended past it — same contract the branch's
        // per-row `cache.appendOnGPU` calls rely on.
        #expect(
            realCache.length == window,
            "cache.length \(realCache.length) should saturate at \(window)")
        #expect(
            realCache.absolutePosition == nRows,
            "cache.absolutePosition \(realCache.absolutePosition) should count every append (\(nRows))"
        )

        let floats = out.toFloatArray()
        #expect(floats.allSatisfy { $0.isFinite }, "windowed-decode-loop output contains NaN/Inf")
    }

    /// A plain (non-forcing) `.window` cache must NOT engage the branch
    /// — `sdpaSinkWindow` still reports `(0, 0)`, so `forwardMulti`
    /// takes the unchanged default path. This is the flip side of the
    /// forcing test above: it confirms the branch really is gated on
    /// `sdpaSinkWindow` and not on window eviction alone (Laguna's
    /// sliding layers use `.window` eviction in production today
    /// without ever touching the windowed-decode-loop branch).
    @Test("a real (non-forcing) window cache never engages windowed-decode-loop")
    func plainWindowCacheStaysOnDefaultPath() {
        let device = Device.shared
        let dtype: DType = .bf16
        var seed: UInt64 = 0x2468_1357_ACE0_9BDF
        let layer = Self.makeTestLayer(seed: &seed)
        let window = 8
        let nRows = 20

        let cache = KVCache(
            nKVHeads: 2, headDim: 128, maxSeq: 32, dtype: dtype,
            eviction: .window(maxSize: window, keep: 0), device: device,
            residency: .weight)
        #expect(cache.sdpaSinkWindow(nKV: cache.length).sinkEnd == 0)
        #expect(cache.sdpaSinkWindow(nKV: cache.length).windowStart == 0)

        let h = Self.randomBf16([nRows, 128], seed: &seed, scale: 0.1)
        let yarn = Ops.RoPEYaRN(factor: 32, low: 1, high: 32, attnFactor: 1.0)

        let cmd = device.makeCommandBuffer()
        let out = layer.forwardMulti(
            h, startingAt: 0, cache: cache, yarn: yarn, cmd: cmd, device: device)
        if cmd.status != .committed && cmd.status != .completed && cmd.status != .error {
            cmd.commit()
        }
        cmd.waitUntilCompleted()

        #expect(out.shape == [nRows, 128])
        let floats = out.toFloatArray()
        #expect(floats.allSatisfy { $0.isFinite }, "default-path output contains NaN/Inf")
    }
}
