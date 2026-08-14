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
// rowSize > 4096 — see `OpsValidation.addRmsNormMaxRowSize`).
//
// **NOT held to exact bit-for-bit equality — this is hardware-dependent,
// same class as `addSoftplus` / `mulSilu` in
// `NemotronHMambaMixerFuseTests.swift`.** The first cut of this test
// DID assert exact equality: it passed on the M5 Max this fusion was
// developed on, including a live 200-token greedy decode A/B against
// the unfused baseline (3 prompts, full Muse Glimmer 30B, byte-identical
// output). PR #137's CI run on a different GPU then recorded 46-195
// bitwise mismatches per dtype for that same exact-equality assertion
// (`gh run view 31775982914`) while every one of the other 2029 tests
// in that run passed clean. Same root cause as `addSoftplus`: this
// kernel's threadgroup reduction feeds a single `rsqrt`, and
// `rsqrt`/transcendental rounding is compile-context- and
// hardware-dependent, not a portable property of the source. This test
// now locks in a tolerance envelope as a regression guard (catches a
// real formula bug) rather than re-asserting a portability claim the
// kernel hasn't earned — matching the fusion's now opt-in-only gate
// (`BUTTER_GLIMMER_WIDE_ADDNORM=1`, default OFF; see
// `MuseGlimmerAddRmsNormWideFuse.swift`'s file header).
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

    /// Generous relative envelope — same magnitude as `addSoftplus`'s
    /// tolerance test in `NemotronHMambaMixerFuseTests.swift`. The
    /// point is catching a gross regression (a real formula bug), not
    /// re-litigating the known cross-GPU rsqrt/reduction noise CI
    /// already surfaced.
    private static let rtol: Float = 1e-2
    private static let atol: Float = 1e-3

    private static func assertWithinEnvelope(
        _ unfused: [Float], _ fused: [Float], label: String
    ) {
        var outOfBand = 0
        for i in 0 ..< unfused.count {
            let bound = Self.atol + Self.rtol * abs(unfused[i])
            if abs(unfused[i] - fused[i]) > bound {
                outOfBand += 1
                if outOfBand <= 3 {
                    print("[\(label)] out-of-band at \(i): unfused=\(unfused[i]) fused=\(fused[i])")
                }
            }
        }
        #expect(outOfBand == 0, "\(label): \(outOfBand)/\(unfused.count) outside atol+rtol envelope")
    }

    private static let dtypes: [DType] = [.f32, .f16, .bf16]
    private static let rowSizes = [6656, 5120]  // Muse Glimmer's hidden, + an unrelated width
    private static let eps: Float = 1e-5

    @Test("addAndRmsNormWide matches Ops.add + Ops.rmsNorm within the known cross-GPU envelope, all dtypes, nRows=1 (decode)")
    func decodeRowWithinKnownTolerance() {
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

                    Self.assertWithinEnvelope(
                        unfusedResid.toFloatArray(), fusedResid.toFloatArray(),
                        label: "addAndRmsNormWide \(dtype) n=\(n) residual")
                    Self.assertWithinEnvelope(
                        unfusedNormed.toFloatArray(), fusedNormed.toFloatArray(),
                        label: "addAndRmsNormWide \(dtype) n=\(n) normed")
                }
            }
        }
    }

    @Test("addAndRmsNormWide matches Ops.add + Ops.rmsNormRows within the known cross-GPU envelope, all dtypes, nRows=4 (prefill-shaped)")
    func multiRowWithinKnownTolerance() {
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

                Self.assertWithinEnvelope(
                    unfusedNormed.toFloatArray(), fusedNormed.toFloatArray(),
                    label: "addAndRmsNormWide \(dtype) multi-row normed")
            }
        }
    }

    /// Tests the LIVE `ButterEnv` read only, NOT
    /// `MuseGlimmerAddRmsNormWideFuse.enabled` — `.enabled` is a
    /// `static let`, so it lazily reads `ButterEnv` exactly ONCE on its
    /// first touch anywhere in the process and caches that value for
    /// the process's lifetime (see `ButterEnv.swift`'s file header: this
    /// is the documented, intentional shape for a per-token/per-layer
    /// hot-path flag). Asserting on `.enabled` from a test is therefore
    /// order-dependent on whichever test happens to touch it first —
    /// this flaked exactly that way once already. Same reasoning
    /// `NemotronHMambaMixerFuseTests.escapeHatchReadsLive` already
    /// applies: check the live env read, not the cached static.
    @Test("BUTTER_GLIMMER_WIDE_ADDNORM=1 opt-in reads live via ButterEnv")
    func optInReadsLive() {
        setenv("BUTTER_GLIMMER_WIDE_ADDNORM", "1", 1)
        defer { unsetenv("BUTTER_GLIMMER_WIDE_ADDNORM") }
        #expect(ButterEnv["BUTTER_GLIMMER_WIDE_ADDNORM"] == "1")
    }
}
