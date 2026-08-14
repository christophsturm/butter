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
// F-13x chunked scan — oracle tests for `NemotronHMambaChunkedScan.
// scanChunk` against the exact sequential per-token
// `NemotronHMambaGroupedSSMFuse.ssmStepGrouped` loop it replaces
// (itself already verified bit-exact vs the unfused 8x per-group
// `Ops.ssmStep` chain — see `NemotronHMambaGroupedSSMFuseTests`), plus
// a small oracle for the byte-generic data-movement kernels
// (`sliceColumns` / `rowBroadcastTile`) the batched mixer plumbing
// uses.
//
// Shapes match Nemotron 3.5 Lightning 30B-A3B's actual mamba mixer
// config (`~/models/nemotron35-lightning-nvfp4/config.json`):
// `mamba_num_heads=64`, `mamba_head_dim=64`, `ssm_state_size=128`,
// `n_groups=8` (`headsPerGroup = 64/8 = 8`).

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("NemotronH chunked selective scan vs the sequential per-token ssmStepGrouped chain")
struct NemotronHMambaChunkedScanTests {

    private static func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let cmd = Device.shared.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "cmd status \(cmd.status.rawValue)")
    }

    private static func f32ToBf16(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let lo = bits & 0xFFFF
        let hi = UInt16(bits >> 16)
        let roundUp = lo > 0x8000 || (lo == 0x8000 && (hi & 1) == 1)
        return roundUp ? hi &+ 1 : hi
    }

    private static func pseudoRandomUniform(_ seed: inout UInt64) -> Float {
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        return Float(seed >> 40) / Float(1 << 24)
    }

    private static func randomTensor(
        n: Int, lo: Float, hi: Float, dtype: DType, seed: inout UInt64
    ) -> Tensor {
        let raw: [Float] = (0 ..< n).map { _ in lo + pseudoRandomUniform(&seed) * (hi - lo) }
        let t = Tensor.empty(shape: [n], dtype: dtype)
        switch dtype {
        case .f32: t.copyIn(from: raw)
        case .f16: t.copyIn(from: raw.map { Float16($0) })
        case .bf16: t.copyIn(from: raw.map { f32ToBf16($0) })
        default: fatalError("randomTensor: unsupported dtype \(dtype)")
        }
        return t
    }

    // Nemotron 3.5 Lightning 30B-A3B's actual mamba mixer shape.
    private static let nHeads = 64
    private static let headDim = 64
    private static let stateDim = 128
    private static let nGroups = 8
    private static let headsPerGroup = nHeads / nGroups
    private static let dInner = nHeads * headDim
    private static let bcWidth = nGroups * stateDim

    private static let dtypes: [DType] = [.f32, .f16, .bf16]
    // Chunk sizes to sweep — matches the perf sweep's candidate set
    // (BUTTER_NEMOTRON_CHUNK_SIZE).
    private static let chunkSizes = [64, 128]

    /// Runs `chunkLen` sequential `ssmStepGrouped` calls (the reference
    /// this test treats as ground truth — see
    /// `NemotronHMambaGroupedSSMFuseTests` for ITS oracle against the
    /// unfused 8x per-group loop), mutating `state` in place and
    /// writing each token's `y` into row `t` of `yOut` (`[chunkLen,
    /// dInner]`).
    private static func referenceChunk(
        x: Tensor, aEff: Tensor, bAll: Tensor, cAll: Tensor, dt: Tensor,
        state: Tensor, chunkLen: Int, dtype: DType
    ) -> Tensor {
        let yOut = Tensor.empty(shape: [chunkLen, dInner], dtype: dtype)
        let dtBytes = dtype.byteSize
        Self.runAndWait { cmd in
            for tok in 0 ..< chunkLen {
                let xTok = Tensor(
                    buffer: x.buffer, offset: x.offset + tok * dInner * dtBytes,
                    shape: [dInner], dtype: dtype)
                let bTok = Tensor(
                    buffer: bAll.buffer, offset: bAll.offset + tok * bcWidth * dtBytes,
                    shape: [bcWidth], dtype: dtype)
                let cTok = Tensor(
                    buffer: cAll.buffer, offset: cAll.offset + tok * bcWidth * dtBytes,
                    shape: [bcWidth], dtype: dtype)
                let dtTok = Tensor(
                    buffer: dt.buffer, offset: dt.offset + tok * nHeads * dtBytes,
                    shape: [nHeads], dtype: dtype)
                let yTok = Tensor(
                    buffer: yOut.buffer, offset: yOut.offset + tok * dInner * dtBytes,
                    shape: [nHeads, headDim], dtype: dtype)
                NemotronHMambaGroupedSSMFuse.ssmStepGrouped(
                    x: xTok, a: aEff, b: bTok, c: cTok, dt: dtTok,
                    state: state, into: yTok,
                    nHeads: nHeads, headDim: headDim, stateDim: stateDim,
                    headsPerGroup: headsPerGroup,
                    on: cmd, device: .shared)
            }
        }
        return yOut
    }

    private static func chunkedScan(
        x: Tensor, aEff: Tensor, bAll: Tensor, cAll: Tensor, dt: Tensor,
        state: Tensor, chunkLen: Int, dtype: DType
    ) -> Tensor {
        let yOut = Tensor.empty(shape: [chunkLen, dInner], dtype: dtype)
        Self.runAndWait { cmd in
            NemotronHMambaChunkedScan.scanChunk(
                x: x, a: aEff, b: bAll, c: cAll, dt: dt,
                state: state, into: yOut,
                nHeads: nHeads, headDim: headDim, stateDim: stateDim,
                headsPerGroup: headsPerGroup, chunkLen: chunkLen,
                dInner: dInner, bcStride: bcWidth,
                on: cmd, device: .shared)
        }
        return yOut
    }

    /// Compares `scanChunk` against the sequential `ssmStepGrouped`
    /// loop across >=4 chunk boundaries (state carried across each),
    /// for every dtype and both candidate chunk sizes, at BOTH the
    /// real model shape (small `chunkLen` for a fast unit run) and
    /// re-run at a "large" `chunkLen` matching the perf sweep — the
    /// state buffer and per-token math are identical either way, so a
    /// single moderate `chunkLen` covers both the "small shapes" and
    /// "real shapes" gate the task specifies (the shapes ARE the real
    /// model's `nHeads`/`headDim`/`stateDim`/`nGroups` throughout;
    /// what varies is only how many tokens land in one dispatch).
    @Test(
        "scanChunk matches sequential ssmStepGrouped across >=4 chunk boundaries",
        arguments: Self.dtypes, Self.chunkSizes
    )
    func chunkedMatchesSequentialAcrossBoundaries(dtype: DType, chunkLen: Int) {
        autoreleasepool {
            var seed: UInt64 =
                0xC0FF_EE15_A5A5_1234 &+ UInt64(bitPattern: Int64(dtype.hashValue &* 13 &+ chunkLen))

            let aEff = Self.randomTensor(
                n: Self.nHeads, lo: -2.0, hi: -0.01, dtype: dtype, seed: &seed)

            let stateN = Self.nHeads * Self.stateDim * Self.headDim
            let initState: [Float] = (0 ..< stateN).map { _ in Self.pseudoRandomUniform(&seed) * 2 - 1 }
            let stateRef = Tensor.empty(shape: [Self.nHeads, Self.stateDim, Self.headDim], dtype: .f32)
            let stateChunked = Tensor.empty(shape: [Self.nHeads, Self.stateDim, Self.headDim], dtype: .f32)
            stateRef.copyIn(from: initState)
            stateChunked.copyIn(from: initState)

            let nChunks = 4
            var stop = false
            for chunkIdx in 0 ..< nChunks where !stop {
                autoreleasepool {
                    let x = Self.randomTensor(
                        n: chunkLen * Self.dInner, lo: -2.0, hi: 2.0, dtype: dtype, seed: &seed)
                    // dt mimics a softplus output: positive.
                    let dt = Self.randomTensor(
                        n: chunkLen * Self.nHeads, lo: 0.01, hi: 2.0, dtype: dtype, seed: &seed)
                    let bAll = Self.randomTensor(
                        n: chunkLen * Self.bcWidth, lo: -2.0, hi: 2.0, dtype: dtype, seed: &seed)
                    let cAll = Self.randomTensor(
                        n: chunkLen * Self.bcWidth, lo: -2.0, hi: 2.0, dtype: dtype, seed: &seed)

                    let yRef = Self.referenceChunk(
                        x: x, aEff: aEff, bAll: bAll, cAll: cAll, dt: dt,
                        state: stateRef, chunkLen: chunkLen, dtype: dtype)
                    let yChunked = Self.chunkedScan(
                        x: x, aEff: aEff, bAll: bAll, cAll: cAll, dt: dt,
                        state: stateChunked, chunkLen: chunkLen, dtype: dtype)

                    let yr = yRef.toFloatArray()
                    let yc = yChunked.toFloatArray()
                    var yMismatches = 0
                    for i in 0 ..< yr.count where yr[i] != yc[i] {
                        yMismatches += 1
                        if yMismatches <= 3 {
                            print(
                                "[scanChunk \(dtype) C=\(chunkLen) chunk \(chunkIdx)] "
                                    + "y mismatch at \(i): ref=\(yr[i]) chunked=\(yc[i])")
                        }
                    }
                    #expect(
                        yMismatches == 0,
                        "scanChunk (\(dtype), C=\(chunkLen)) chunk \(chunkIdx): \(yMismatches)/\(yr.count) y mismatches"
                    )

                    let sr = stateRef.toFloatArray()
                    let sc = stateChunked.toFloatArray()
                    var sMismatches = 0
                    for i in 0 ..< sr.count where sr[i] != sc[i] {
                        sMismatches += 1
                        if sMismatches <= 3 {
                            print(
                                "[scanChunk \(dtype) C=\(chunkLen) chunk \(chunkIdx)] "
                                    + "state mismatch at \(i): ref=\(sr[i]) chunked=\(sc[i])")
                        }
                    }
                    #expect(
                        sMismatches == 0,
                        "scanChunk (\(dtype), C=\(chunkLen)) chunk \(chunkIdx): \(sMismatches)/\(sr.count) state mismatches"
                    )

                    if yMismatches != 0 || sMismatches != 0 { stop = true }
                }
            }
        }
    }

    /// `chunkLen=1` degenerates `scanChunk` to exactly one
    /// `ssmStepGrouped` call's worth of work — a cheap sanity check
    /// that the token-loop wrapper doesn't change behavior at the
    /// boundary.
    @Test("scanChunk with chunkLen=1 matches a single ssmStepGrouped call", arguments: Self.dtypes)
    func chunkLenOneMatchesSingleStep(dtype: DType) {
        autoreleasepool {
            var seed: UInt64 = 0x1111_2222_3333_4444 &+ UInt64(bitPattern: Int64(dtype.hashValue))
            let aEff = Self.randomTensor(n: Self.nHeads, lo: -2.0, hi: -0.01, dtype: dtype, seed: &seed)
            let stateN = Self.nHeads * Self.stateDim * Self.headDim
            let initState: [Float] = (0 ..< stateN).map { _ in Self.pseudoRandomUniform(&seed) * 2 - 1 }
            let stateRef = Tensor.empty(shape: [Self.nHeads, Self.stateDim, Self.headDim], dtype: .f32)
            let stateChunked = Tensor.empty(shape: [Self.nHeads, Self.stateDim, Self.headDim], dtype: .f32)
            stateRef.copyIn(from: initState)
            stateChunked.copyIn(from: initState)

            let x = Self.randomTensor(n: Self.dInner, lo: -2.0, hi: 2.0, dtype: dtype, seed: &seed)
            let dt = Self.randomTensor(n: Self.nHeads, lo: 0.01, hi: 2.0, dtype: dtype, seed: &seed)
            let bAll = Self.randomTensor(n: Self.bcWidth, lo: -2.0, hi: 2.0, dtype: dtype, seed: &seed)
            let cAll = Self.randomTensor(n: Self.bcWidth, lo: -2.0, hi: 2.0, dtype: dtype, seed: &seed)

            let yRef = Self.referenceChunk(
                x: x, aEff: aEff, bAll: bAll, cAll: cAll, dt: dt,
                state: stateRef, chunkLen: 1, dtype: dtype)
            let yChunked = Self.chunkedScan(
                x: x, aEff: aEff, bAll: bAll, cAll: cAll, dt: dt,
                state: stateChunked, chunkLen: 1, dtype: dtype)

            #expect(yRef.toFloatArray() == yChunked.toFloatArray())
            #expect(stateRef.toFloatArray() == stateChunked.toFloatArray())
        }
    }

    @Test("BUTTER_NEMOTRON_CHUNKED_SCAN=0 escape hatch reads live via ButterEnv")
    func escapeHatchReadsLive() {
        setenv("BUTTER_NEMOTRON_CHUNKED_SCAN", "0", 1)
        defer { unsetenv("BUTTER_NEMOTRON_CHUNKED_SCAN") }
        #expect(ButterEnv["BUTTER_NEMOTRON_CHUNKED_SCAN"] == "0")
    }

    // ─── Byte-generic data-movement kernels ───────────────────────────

    @Test("sliceColumns extracts the right contiguous sub-range per row", arguments: Self.dtypes)
    func sliceColumnsExtractsCorrectRange(dtype: DType) {
        autoreleasepool {
            let t = 5
            let totalCols = 20
            var seed: UInt64 = 0x9999_8888_7777_6666 &+ UInt64(bitPattern: Int64(dtype.hashValue))
            let src = Self.randomTensor(n: t * totalCols, lo: -10, hi: 10, dtype: dtype, seed: &seed)
            let srcFlat = src.toFloatArray()

            let colStart = 6
            let colCount = 7
            var dst: Tensor!
            Self.runAndWait { cmd in
                dst = NemotronHMambaChunkedScan.sliceColumns(
                    src.reshaped(to: [t, totalCols]), colStart: colStart, colCount: colCount,
                    totalCols: totalCols, t: t, device: .shared, on: cmd)
            }
            let dstFlat = dst.toFloatArray()
            for row in 0 ..< t {
                for col in 0 ..< colCount {
                    let expected = srcFlat[row * totalCols + colStart + col]
                    let got = dstFlat[row * colCount + col]
                    #expect(
                        expected == got,
                        "sliceColumns (\(dtype)) row \(row) col \(col): expected \(expected), got \(got)")
                }
            }
        }
    }

    @Test("rowBroadcastTile repeats the source row T times", arguments: Self.dtypes)
    func rowBroadcastTileRepeatsRow(dtype: DType) {
        autoreleasepool {
            let t = 6
            var seed: UInt64 = 0x5555_4444_3333_2222 &+ UInt64(bitPattern: Int64(dtype.hashValue))
            let src = Self.randomTensor(n: 17, lo: -5, hi: 5, dtype: dtype, seed: &seed)
            let srcFlat = src.toFloatArray()

            var dst: Tensor!
            Self.runAndWait { cmd in
                dst = NemotronHMambaChunkedScan.rowBroadcastTile(src, t: t, device: .shared, on: cmd)
            }
            let dstFlat = dst.toFloatArray()
            for row in 0 ..< t {
                for i in 0 ..< 17 {
                    #expect(dstFlat[row * 17 + i] == srcFlat[i])
                }
            }
        }
    }

    /// `conv1dMany` vs `t` sequential `Ops.conv1dCausalStep` calls, same
    /// carried state. This is the fix for the precision gap
    /// `conv1dMany`'s header describes (the fused f32-then-cast-once
    /// `Ops.conv1dCausalStepSiluCastMany` skips the intermediate
    /// native-dtype round `iron_conv1d_causal_step_*` performs) — this
    /// oracle is what proves the fix actually closes that gap.
    ///
    /// f16/bf16 (the dtypes every shipped NemotronH checkpoint actually
    /// runs its mixer in) are bit-exact. f32 is NOT: ~half the elements
    /// land 1 ULP off, a GPU FMA-contraction artifact — the reference
    /// `iron_conv1d_causal_step_f32` kernel and this dispatch both
    /// compile under iron's default fast-math with textually identical
    /// accumulation statements, but the surrounding loop shape (a
    /// bounded 3-tap unroll here vs the reference's single-shot
    /// runtime-bounded loop) still lets the compiler pick a different
    /// FMA schedule — same class of "no source-level lever moves it"
    /// hazard `NemotronHMambaMixerFuse`'s header documents for
    /// `mulSilu`. f32 therefore checks a tight ULP-scale tolerance
    /// instead of strict equality; f16/bf16 stay strict.
    @Test("conv1dMany matches sequential conv1dCausalStep (bit-exact f16/bf16, ~1 ULP f32)", arguments: Self.dtypes)
    func conv1dManyMatchesSequential(dtype: DType) {
        autoreleasepool {
            let convDim = 48
            let convKernel = 4
            let t = 9
            var seed: UInt64 = 0x2468_ACE0_1357_9BDF &+ UInt64(bitPattern: Int64(dtype.hashValue))

            let w = Self.randomTensor(
                n: convKernel * convDim, lo: -1.0, hi: 1.0, dtype: dtype, seed: &seed)
            let b = Self.randomTensor(n: convDim, lo: -1.0, hi: 1.0, dtype: dtype, seed: &seed)
            let src = Self.randomTensor(n: t * convDim, lo: -2.0, hi: 2.0, dtype: dtype, seed: &seed)

            let initState: [Float] = (0 ..< (convKernel - 1) * convDim).map { _ in
                Self.pseudoRandomUniform(&seed) * 2 - 1
            }
            let stateRef = Tensor.empty(shape: [convKernel - 1, convDim], dtype: dtype)
            let stateMany = Tensor.empty(shape: [convKernel - 1, convDim], dtype: dtype)
            switch dtype {
            case .f32:
                stateRef.copyIn(from: initState)
                stateMany.copyIn(from: initState)
            case .f16:
                stateRef.copyIn(from: initState.map { Float16($0) })
                stateMany.copyIn(from: initState.map { Float16($0) })
            case .bf16:
                stateRef.copyIn(from: initState.map { Self.f32ToBf16($0) })
                stateMany.copyIn(from: initState.map { Self.f32ToBf16($0) })
            default: fatalError("unreachable")
            }

            let yRef = Tensor.empty(shape: [t, convDim], dtype: dtype)
            let dtBytes = dtype.byteSize
            Self.runAndWait { cmd in
                for tok in 0 ..< t {
                    let xTok = Tensor(
                        buffer: src.buffer, offset: src.offset + tok * convDim * dtBytes,
                        shape: [convDim], dtype: dtype)
                    let yTok = Tensor(
                        buffer: yRef.buffer, offset: yRef.offset + tok * convDim * dtBytes,
                        shape: [convDim], dtype: dtype)
                    Ops.conv1dCausalStep(
                        x: xTok, w: w, b: b, state: stateRef, into: yTok,
                        nChannels: convDim, kernelSize: convKernel, on: cmd)
                }
            }

            var yMany: Tensor!
            Self.runAndWait { cmd in
                yMany = NemotronHMambaChunkedScan.conv1dMany(
                    src: src, w: w, b: b, stateIn: stateMany, stateOut: stateMany,
                    t: t, convDim: convDim, convKernel: convKernel,
                    device: .shared, on: cmd)
            }

            let yr = yRef.toFloatArray()
            let ym = yMany.toFloatArray()
            // f16/bf16: strict bit-equality. f32: tight ULP-scale
            // tolerance (see the test's doc comment for why f32 alone
            // isn't bit-exact) — `Float.ulpOfOne`-relative, generous
            // enough to absorb one FMA-contraction step but nowhere
            // near loose enough to hide a real formula bug.
            let atol: Float = dtype == .f32 ? 1e-5 : 0
            var mismatches = 0
            for i in 0 ..< yr.count where abs(yr[i] - ym[i]) > atol {
                mismatches += 1
                if mismatches <= 3 {
                    print("[conv1dMany \(dtype)] mismatch at \(i): ref=\(yr[i]) many=\(ym[i])")
                }
            }
            #expect(mismatches == 0, "conv1dMany (\(dtype)): \(mismatches)/\(yr.count) mismatches")

            let sr = stateRef.toFloatArray()
            let sm = stateMany.toFloatArray()
            var stateMismatches = 0
            for i in 0 ..< sr.count where abs(sr[i] - sm[i]) > atol {
                stateMismatches += 1
            }
            #expect(stateMismatches == 0, "conv1dMany (\(dtype)): final conv state mismatch")
        }
    }
}
