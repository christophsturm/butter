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
// F-past-window-prefill: unit coverage for Muse Glimmer's batched
// prefill past a sliding-attention layer's window
// (`MuseGlimmerText.swift`, `MuseGlimmerLayer.forwardMulti`'s
// `windowWraps` branch + `MuseGlimmerModel.forwardMulti`'s
// `BUTTER_GLIMMER_LONG_PREFILL` gate).
//
// Two suites:
//
//   `MuseGlimmerRingAlignmentOracleTests` — the ring-alignment oracle.
//   Isolates the ONE thing that's actually new here (interleaving a
//   whole chunk's RoPE+append+attend onto a SINGLE command buffer
//   instead of one command buffer PER TOKEN) from GEMM-batching
//   reassociation, which is a separate, already-accepted source of
//   divergence elsewhere in the codebase (see the module header comment
//   in `MuseGlimmerText.swift`'s past-window doc comments). Drives the
//   exact `Ops.rope` → `KVCache.appendOnGPU` → `Ops.sdpaDecode` triple
//   `MuseGlimmerLayer.forwardMulti`'s windowed path uses, twice, with
//   IDENTICAL inputs: once with a fresh commit+wait per row (today's
//   per-token discipline) and once with every row encoded onto ONE
//   shared command buffer (the new path). A small window (8) forces
//   several ring wraps well inside a 20-row chunk. Both the resulting
//   KV cache buffers AND every row's attention output are compared
//   BIT-FOR-BIT — Metal's execution model guarantees identical results
//   for identical kernels run in identical encoding order regardless of
//   where commit/wait boundaries fall, so this is held to exact
//   equality, not a tolerance envelope.
//
//   `MuseGlimmerLongPrefillWiringTests` — exercises the real
//   `MuseGlimmerLayer.forwardMulti` end-to-end on a tiny synthetic
//   layer (no 30B checkpoint needed) to check the control flow itself:
//   `windowWraps` actually engages past the window, the ring-eviction
//   state ends up where it should, output stays finite, and
//   `BUTTER_GLIMMER_LONG_PREFILL=0` actually disables the new branch
//   (verified indirectly — see that test's doc comment for why a
//   direct output diff isn't safe here).

import Foundation
import Metal
import Testing

@testable import ButterSwift

// MARK: - Ring-alignment oracle

@Suite("Muse Glimmer past-window prefill — ring-alignment oracle")
struct MuseGlimmerRingAlignmentOracleTests {

    /// Deterministic xorshift* generator — same reseed pattern as the
    /// NemotronH mixer-fuse oracle tests.
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

    /// Runs the exact windowed-attention triple (`Ops.rope` ×2 →
    /// `KVCache.appendOnGPU` → `Ops.sdpaDecode`) for `nRows` synthetic
    /// tokens against a `.window(maxSize: window, keep: 0)` cache, and
    /// returns (final K buffer bytes, final V buffer bytes, all `nRows`
    /// attention-output rows) as raw `UInt16` bit patterns (f16/bf16
    /// storage — exact, no float round-trip).
    ///
    /// `singleCommandBuffer: true` mimics the NEW path — every row's
    /// RoPE/append/attend is encoded onto one shared `cmd`, committed
    /// once at the very end. `false` mimics the pre-existing per-token
    /// discipline — commit + `waitUntilCompleted` after every row.
    private static func runWindowedAttentionSequence(
        nRows: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        window: Int, maxSeq: Int, dtype: DType,
        qRows: [Tensor], kRows: [Tensor], vRows: [Tensor],
        ropeTheta: Float, scale: Float,
        singleCommandBuffer: Bool, device: Device
    ) -> (kBytes: [UInt16], vBytes: [UInt16], attnBytes: [UInt16]) {
        let cache = KVCache(
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq, dtype: dtype,
            eviction: .window(maxSize: window, keep: 0), device: device,
            residency: .weight)
        var attnOut: [UInt16] = []
        attnOut.reserveCapacity(nRows * nHeads * headDim)

        /// Per-token reference discipline: fresh command buffer,
        /// commit + wait, THEN read the row back — mirrors
        /// `forward(tokenId:...)`'s per-token round trip exactly.
        func runRowCommittedImmediately(_ i: Int, on cmd: MTLCommandBuffer) {
            let qRot = Ops.rope(
                qRows[i], position: i, headDim: headDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd)
            let kRot = Ops.rope(
                kRows[i], position: i, headDim: headDim,
                thetaBase: ropeTheta, scaling: .none, on: cmd)
            cache.appendOnGPU(kFlat: kRot, vFlat: vRows[i], on: cmd)
            let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
            let nKV = cache.length
            let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: nKV)
            let row = Ops.sdpaDecode(
                q: qRot, k: cacheK, v: cacheV,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                nKV: nKV, kvStride: cache.maxSeq,
                scale: scale, on: cmd,
                sinkEnd: sinkEnd, windowStart: windowStart)
            cmd.commit()
            cmd.waitUntilCompleted()
            attnOut.append(contentsOf: row.toArray(as: UInt16.self))
        }

        if singleCommandBuffer {
            // Encode every row's RoPE + append + attend onto ONE shared
            // `cmd` (no intermediate commit/wait — the new path's
            // discipline), then commit once and read every row's output
            // back only after the whole chunk has completed. The
            // property under test (does the ring end up correct when
            // nothing commits between appends) depends only on GPU
            // encoding order, not on when the host reads results back.
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
                let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
                let nKV = cache.length
                let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: nKV)
                let row = Ops.sdpaDecode(
                    q: qRot, k: cacheK, v: cacheV,
                    nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    nKV: nKV, kvStride: cache.maxSeq,
                    scale: scale, on: cmd,
                    sinkEnd: sinkEnd, windowStart: windowStart)
                rows.append(row)
            }
            cmd.commit()
            cmd.waitUntilCompleted()
            for row in rows { attnOut.append(contentsOf: row.toArray(as: UInt16.self)) }
        } else {
            for i in 0 ..< nRows {
                let cmd = device.makeCommandBuffer()
                runRowCommittedImmediately(i, on: cmd)
            }
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

        let reference = Self.runWindowedAttentionSequence(
            nRows: nRows, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            window: window, maxSeq: maxSeq, dtype: dtype,
            qRows: qRows, kRows: kRows, vRows: vRows,
            ropeTheta: 500_000.0, scale: scale,
            singleCommandBuffer: false, device: device)
        let batched = Self.runWindowedAttentionSequence(
            nRows: nRows, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            window: window, maxSeq: maxSeq, dtype: dtype,
            qRows: qRows, kRows: kRows, vRows: vRows,
            ropeTheta: 500_000.0, scale: scale,
            singleCommandBuffer: true, device: device)

        #expect(
            reference.kBytes == batched.kBytes,
            "\(dtype): K cache bytes diverged between per-token-committed and single-command-buffer windowed attention — ring alignment bug"
        )
        #expect(
            reference.vBytes == batched.vBytes,
            "\(dtype): V cache bytes diverged between per-token-committed and single-command-buffer windowed attention — ring alignment bug"
        )
        #expect(
            reference.attnBytes == batched.attnBytes,
            "\(dtype): attention output diverged between per-token-committed and single-command-buffer windowed attention"
        )
    }

    @Test("window=8 over a 20-row chunk: KV cache + attention output are bit-exact vs per-token, f16")
    func ringAlignmentF16() { Self.ringAlignmentCase(dtype: .f16) }

    @Test("window=8 over a 20-row chunk: KV cache + attention output are bit-exact vs per-token, bf16")
    func ringAlignmentBf16() { Self.ringAlignmentCase(dtype: .bf16) }
}

// MARK: - Layer wiring sanity

@Suite("Muse Glimmer past-window prefill — layer wiring")
struct MuseGlimmerLongPrefillWiringTests {

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

    private static func makeLinear(
        outDim: Int, inDim: Int, range: Float, dtype: DType, seed: inout UInt64, device: Device
    ) -> AnyLinear {
        let w = Self.randomTensor(
            shape: [outDim, inDim], range: range, dtype: dtype, seed: &seed, device: device)
        return AnyLinear(Linear(weight: w))
    }

    private static func makeRMSNorm(
        n: Int, dtype: DType, eps: Float, device: Device
    ) -> RMSNorm {
        // Centered-norm convention (`1 + weight`, per `loadGemmaRMSNorm`)
        // isn't needed here — a plain unit weight exercises the same
        // kernels this test cares about (the attention windowing, not
        // norm-weight semantics).
        let w = Tensor.empty(shape: [n], dtype: dtype, device: device)
        switch dtype {
        case .f16: w.copyIn(from: [Float16](repeating: 1.0, count: n))
        case .bf16: w.copyIn(from: [UInt16](repeating: 0x3F80, count: n))  // bf16 1.0
        case .f32: w.copyIn(from: [Float](repeating: 1.0, count: n))
        default: fatalError("makeRMSNorm: unsupported dtype \(dtype)")
        }
        return RMSNorm(weight: w, eps: eps)
    }

    /// Small-but-representative synthetic sliding layer: hidden=64,
    /// nHeads=4, nKVHeads=2, headDim=128 (matches the real model's
    /// supported SDPA head_dim), intermediate=32.
    private static func makeSlidingLayer(dtype: DType, device: Device) -> MuseGlimmerLayer {
        var seed: UInt64 = 0x5EED_C0DE_0000_0001
        let hidden = 64
        let nHeads = 4
        let nKVHeads = 2
        let headDim = 128
        let intermediate = 32
        let qOutDim = nHeads * headDim
        let kvOutDim = nKVHeads * headDim

        return MuseGlimmerLayer(
            qProj: Self.makeLinear(
                outDim: qOutDim, inDim: hidden, range: 0.05, dtype: dtype, seed: &seed,
                device: device),
            kProj: Self.makeLinear(
                outDim: kvOutDim, inDim: hidden, range: 0.05, dtype: dtype, seed: &seed,
                device: device),
            vProj: Self.makeLinear(
                outDim: kvOutDim, inDim: hidden, range: 0.05, dtype: dtype, seed: &seed,
                device: device),
            oProj: Self.makeLinear(
                outDim: hidden, inDim: qOutDim, range: 0.05, dtype: dtype, seed: &seed,
                device: device),
            gateProj: Self.makeLinear(
                outDim: qOutDim, inDim: hidden, range: 0.05, dtype: dtype, seed: &seed,
                device: device),
            mlpGateProj: Self.makeLinear(
                outDim: intermediate, inDim: hidden, range: 0.05, dtype: dtype, seed: &seed,
                device: device),
            upProj: Self.makeLinear(
                outDim: intermediate, inDim: hidden, range: 0.05, dtype: dtype, seed: &seed,
                device: device),
            downProj: Self.makeLinear(
                outDim: hidden, inDim: intermediate, range: 0.05, dtype: dtype, seed: &seed,
                device: device),
            inputNorm: Self.makeRMSNorm(n: hidden, dtype: dtype, eps: 1e-5, device: device),
            postAttnNorm: Self.makeRMSNorm(n: hidden, dtype: dtype, eps: 1e-8, device: device),
            preFFNorm: Self.makeRMSNorm(n: hidden, dtype: dtype, eps: 1e-5, device: device),
            postFFNorm: Self.makeRMSNorm(n: hidden, dtype: dtype, eps: 1e-8, device: device),
            qNorm: Self.makeRMSNorm(n: headDim, dtype: dtype, eps: 1e-5, device: device),
            kNorm: Self.makeRMSNorm(n: headDim, dtype: dtype, eps: 1e-5, device: device),
            hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            intermediate: intermediate, ropeTheta: 500_000.0, isSliding: true
        )
    }

    @Test("windowWraps engages past the window and produces finite output")
    func windowedPathEngagesAndStaysFinite() {
        let device = Device.shared
        let dtype: DType = .f16
        let layer = Self.makeSlidingLayer(dtype: dtype, device: device)
        let window = 8
        let nRows = 20

        let cache = KVCache(
            nKVHeads: 2, headDim: 128, maxSeq: 32, dtype: dtype,
            eviction: .window(maxSize: window, keep: 0), device: device,
            residency: .weight)

        var seed: UInt64 = 0x1357_9BDF_2468_ACE0
        let h = Self.randomTensor(
            shape: [nRows, 64], range: 0.1, dtype: dtype, seed: &seed, device: device)

        setenv("BUTTER_GLIMMER_LONG_PREFILL", "1", 1)
        let cmd = device.makeCommandBuffer()
        let out = layer.forwardMulti(h, startingAt: 0, cache: cache, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        unsetenv("BUTTER_GLIMMER_LONG_PREFILL")

        #expect(out.shape == [nRows, 64])
        // Ring eviction saturates at the window size regardless of how
        // many rows were appended past it.
        #expect(cache.length == window, "cache.length \(cache.length) should saturate at \(window)")
        #expect(
            cache.absolutePosition == nRows,
            "cache.absolutePosition \(cache.absolutePosition) should count every append (\(nRows))"
        )

        let floats = out.toFloatArray()
        #expect(floats.allSatisfy { $0.isFinite }, "windowed-prefill layer output contains NaN/Inf")
    }

    /// `BUTTER_GLIMMER_LONG_PREFILL=0` must keep `windowWraps` from ever
    /// engaging — i.e. the layer must still route sliding attention
    /// through the single whole-chunk `sdpaMulti` even once the chunk
    /// exceeds the window. That call reads physical ring slots as if
    /// they were absolute positions (see the doc comment on
    /// `MuseGlimmerLayer.forwardMulti`), which past the window is
    /// numerically WRONG relative to `windowWraps=true`'s exact
    /// per-token-equivalent answer — so the flag-off and flag-on runs
    /// are EXPECTED to diverge here. Asserting they differ (rather than
    /// asserting an exact flag-off value, which would just re-encode
    /// today's bug as a fixture) is the safe, low-maintenance way to
    /// confirm the flag actually gates the new branch instead of being
    /// silently ignored.
    @Test("BUTTER_GLIMMER_LONG_PREFILL=0 takes a different (pre-existing) path once past the window")
    func envFlagGatesTheNewBranch() {
        let device = Device.shared
        let dtype: DType = .f16
        let window = 8
        let nRows = 20
        var seed: UInt64 = 0x2468_1357_ACE0_9BDF
        let h = Self.randomTensor(
            shape: [nRows, 64], range: 0.1, dtype: dtype, seed: &seed, device: device)

        func run(flagOn: Bool) -> [Float] {
            let layer = Self.makeSlidingLayer(dtype: dtype, device: device)
            let cache = KVCache(
                nKVHeads: 2, headDim: 128, maxSeq: 32, dtype: dtype,
                eviction: .window(maxSize: window, keep: 0), device: device,
                residency: .weight)
            setenv("BUTTER_GLIMMER_LONG_PREFILL", flagOn ? "1" : "0", 1)
            let cmd = device.makeCommandBuffer()
            let out = layer.forwardMulti(h, startingAt: 0, cache: cache, on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            unsetenv("BUTTER_GLIMMER_LONG_PREFILL")
            return out.toFloatArray()
        }

        // Same layer weights, same input rows — construct the layer
        // fresh each call (`makeSlidingLayer` reseeds from a fixed
        // constant) so both runs see IDENTICAL weights, isolating the
        // env flag as the only variable.
        let onOut = run(flagOn: true)
        let offOut = run(flagOn: false)
        #expect(onOut.allSatisfy { $0.isFinite })
        #expect(offOut.allSatisfy { $0.isFinite })
        let anyDiffer = zip(onOut, offOut).contains { $0 != $1 }
        #expect(
            anyDiffer,
            "flag on vs off produced IDENTICAL output past the window — the env flag does not appear to gate the new branch"
        )
    }
}
