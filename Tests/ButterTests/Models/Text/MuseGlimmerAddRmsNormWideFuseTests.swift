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
// glimmer-decode-gap lever 4: oracle test comparing
// `MuseGlimmerAddRmsNormWideFuse.addAndRmsNormWide` against the exact
// unfused chain it replaces (`Ops.add` then `Ops.rmsNorm`, which
// `dispatchRmsNorm` routes to the `iron_rms_norm_wide_*` kernel for any
// rowSize > 4096 — see `OpsValidation.addRmsNormMaxRowSize`). Held to
// EXACT bit-for-bit equality, not a tolerance envelope — unlike
// `addSoftplus` / `mulSilu` in `NemotronHMambaMixerFuseTests.swift`,
// this fusion has no transcendental beyond the single `rsqrt` every
// RMSNorm kernel in this codebase already uses (including the
// reference it's compared against), and a live 200-token greedy decode
// A/B against the unfused baseline (3 prompts, full Muse Glimmer 30B)
// confirmed byte-identical output at `fastMathEnabled = true` — see
// `MuseGlimmerAddRmsNormWideFuse.swift`'s "Compile flags" header
// section for the (wrong first, then corrected) empirical story.
//
// rowSize = 6656 matches Muse Glimmer's actual hidden_size; 5120 is a
// second, unrelated width to confirm the kernel isn't accidentally
// tuned to one specific `n`.

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("Muse Glimmer wide add+RMSNorm fusion vs the unfused Ops chain")
struct MuseGlimmerAddRmsNormWideFuseTests {

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
        n: Int, range: Float, dtype: DType, seed: inout UInt64
    ) -> Tensor {
        let raw: [Float] = (0 ..< n).map { _ in
            (pseudoRandomUniform(&seed) * 2 - 1) * range
        }
        let t = Tensor.empty(shape: [n], dtype: dtype)
        switch dtype {
        case .f32: t.copyIn(from: raw)
        case .f16: t.copyIn(from: raw.map { Float16($0) })
        case .bf16: t.copyIn(from: raw.map { f32ToBf16($0) })
        default: fatalError("randomTensor: unsupported dtype \(dtype)")
        }
        return t
    }

    private static let dtypes: [DType] = [.f32, .f16, .bf16]
    private static let rowSizes = [6656, 5120]  // Muse Glimmer's hidden, + an unrelated width
    private static let eps: Float = 1e-5

    @Test("addAndRmsNormWide matches Ops.add + Ops.rmsNorm exactly, all dtypes, nRows=1 (decode)")
    func decodeRowExact() {
        for dtype in Self.dtypes {
            for n in Self.rowSizes {
                autoreleasepool {
                    var seed: UInt64 = 0xA5A5_1234_9E37_79B9
                    let a = Self.randomTensor(n: n, range: 3.0, dtype: dtype, seed: &seed)
                    let b = Self.randomTensor(n: n, range: 3.0, dtype: dtype, seed: &seed)
                    let w = Self.randomTensor(n: n, range: 1.5, dtype: dtype, seed: &seed)

                    var unfusedResid: Tensor!
                    var unfusedNormed: Tensor!
                    Self.runAndWait { cmd in
                        unfusedResid = Ops.add(a, b, on: cmd)
                        unfusedNormed = Ops.rmsNorm(unfusedResid, weight: w, eps: Self.eps, on: cmd)
                    }
                    var fusedResid: Tensor!
                    var fusedNormed: Tensor!
                    Self.runAndWait { cmd in
                        let fused = MuseGlimmerAddRmsNormWideFuse.addAndRmsNormWide(
                            a, b, weight: w, eps: Self.eps,
                            nRows: 1, rowSize: n, device: .shared, on: cmd)
                        fusedResid = fused.residual
                        fusedNormed = fused.normed
                    }

                    let ur = unfusedResid.toFloatArray()
                    let fr = fusedResid.toFloatArray()
                    var residMismatches = 0
                    for i in 0 ..< n where ur[i] != fr[i] {
                        residMismatches += 1
                        if residMismatches <= 3 {
                            print("[addAndRmsNormWide \(dtype) n=\(n)] residual mismatch at \(i): unfused=\(ur[i]) fused=\(fr[i])")
                        }
                    }
                    #expect(residMismatches == 0, "residual (\(dtype), n=\(n)): \(residMismatches)/\(n) mismatches")

                    let un = unfusedNormed.toFloatArray()
                    let fn = fusedNormed.toFloatArray()
                    var normedMismatches = 0
                    for i in 0 ..< n where un[i] != fn[i] {
                        normedMismatches += 1
                        if normedMismatches <= 3 {
                            print("[addAndRmsNormWide \(dtype) n=\(n)] normed mismatch at \(i): unfused=\(un[i]) fused=\(fn[i])")
                        }
                    }
                    #expect(normedMismatches == 0, "normed (\(dtype), n=\(n)): \(normedMismatches)/\(n) mismatches")
                }
            }
        }
    }

    @Test("addAndRmsNormWide matches Ops.add + Ops.rmsNormRows exactly, all dtypes, nRows=4 (prefill-shaped)")
    func multiRowExact() {
        let nRows = 4
        for dtype in Self.dtypes {
            autoreleasepool {
                let n = 6656
                var seed: UInt64 = 0x27D4_EB4F_1656_67B1
                let a = Self.randomTensor(n: n * nRows, range: 3.0, dtype: dtype, seed: &seed)
                let b = Self.randomTensor(n: n * nRows, range: 3.0, dtype: dtype, seed: &seed)
                let w = Self.randomTensor(n: n, range: 1.5, dtype: dtype, seed: &seed)

                var unfusedNormed: Tensor!
                Self.runAndWait { cmd in
                    let resid = Ops.add(a, b, on: cmd)
                    unfusedNormed = Ops.rmsNormRows(
                        resid, weight: w, eps: Self.eps, nRows: nRows, rowSize: n, on: cmd)
                }
                var fusedNormed: Tensor!
                Self.runAndWait { cmd in
                    let fused = MuseGlimmerAddRmsNormWideFuse.addAndRmsNormWide(
                        a, b, weight: w, eps: Self.eps,
                        nRows: nRows, rowSize: n, device: .shared, on: cmd)
                    fusedNormed = fused.normed
                }

                let un = unfusedNormed.toFloatArray()
                let fn = fusedNormed.toFloatArray()
                var mismatches = 0
                for i in 0 ..< (n * nRows) where un[i] != fn[i] {
                    mismatches += 1
                    if mismatches <= 3 {
                        print("[addAndRmsNormWide \(dtype) multi-row] mismatch at \(i): unfused=\(un[i]) fused=\(fn[i])")
                    }
                }
                #expect(mismatches == 0, "multi-row normed (\(dtype)): \(mismatches)/\(n * nRows) mismatches")
            }
        }
    }

    @Test("BUTTER_MUSEGLIMMER_ADD_RMS_NORM_WIDE=0 escape hatch reads live via ButterEnv")
    func escapeHatchReadsLive() {
        setenv("BUTTER_MUSEGLIMMER_ADD_RMS_NORM_WIDE", "0", 1)
        defer { unsetenv("BUTTER_MUSEGLIMMER_ADD_RMS_NORM_WIDE") }
        #expect(ButterEnv["BUTTER_MUSEGLIMMER_ADD_RMS_NORM_WIDE"] == "0")
        #expect(!MuseGlimmerAddRmsNormWideFuse.enabled)
    }
}
