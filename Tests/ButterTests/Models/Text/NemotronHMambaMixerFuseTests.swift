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
// F-129: NemotronH mamba mixer elementwise dispatch fusions
// (`NemotronHMambaMixerFuse.swift`) — oracle tests comparing each
// fused kernel against the exact unfused `Ops.*` chain it replaces, at
// all three floating dtypes. These are pure one-thread-per-element ops
// with no reduction, so unlike the F-127 MoE fuse (whose GPU reduction
// order can legitimately differ between fused/unfused), `addSoftplus`
// and `addMul` are held to EXACT bit-for-bit equality — and hit it.
// `mulSilu` is the one exception: measured NOT bit-exact (a GPU
// transcendental-precision quirk, not a formula bug — see
// `NemotronHMambaMixerFuse`'s header) and tested to a tolerance
// envelope instead, matching its default-OFF gate.

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("NemotronH mamba mixer fused elementwise kernels vs the unfused Ops chain")
struct NemotronHMambaMixerFuseTests {

    private static func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let cmd = Device.shared.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "cmd status \(cmd.status.rawValue)")
    }

    /// Round-to-nearest-even f32 -> bf16, expressed as the top 16 bits
    /// of the f32 bit pattern (same convention as `OpsDtypeTests`).
    private static func f32ToBf16(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let lo = bits & 0xFFFF
        let hi = UInt16(bits >> 16)
        let roundUp = lo > 0x8000 || (lo == 0x8000 && (hi & 1) == 1)
        return roundUp ? hi &+ 1 : hi
    }

    /// Deterministic xorshift* generator — same reseed pattern as the
    /// F-127 fused-experts test, small and dependency-free.
    private static func pseudoRandomUniform(_ seed: inout UInt64) -> Float {
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        // [0, 1)
        return Float(seed >> 40) / Float(1 << 24)
    }

    /// Build a `[n]` tensor of `dtype` with values uniform in
    /// `[-range, range]`, and return the AS-STORED float values (i.e.
    /// after the dtype's own rounding) so a CPU-side oracle can use the
    /// exact same numbers the GPU kernel reads.
    private static func randomTensor(
        n: Int, range: Float, dtype: DType, seed: inout UInt64
    ) -> (tensor: Tensor, floats: [Float]) {
        let raw: [Float] = (0 ..< n).map { _ in
            (pseudoRandomUniform(&seed) * 2 - 1) * range
        }
        let t = Tensor.empty(shape: [n], dtype: dtype)
        switch dtype {
        case .f32:
            t.copyIn(from: raw)
        case .f16:
            t.copyIn(from: raw.map { Float16($0) })
        case .bf16:
            t.copyIn(from: raw.map { f32ToBf16($0) })
        default:
            fatalError("randomTensor: unsupported dtype \(dtype)")
        }
        return (t, t.toFloatArray())
    }

    private static let dtypes: [DType] = [.f32, .f16, .bf16]
    private static let n = 512  // covers dInner-shaped ops; nHeads (64) below n

    @Test("addSoftplus matches Ops.add + Ops.softplus exactly, all dtypes")
    func addSoftplusExact() {
        for dtype in Self.dtypes {
            autoreleasepool {
                var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
                let (dtRaw, _) = Self.randomTensor(n: Self.n, range: 4.0, dtype: dtype, seed: &seed)
                let (dtBias, _) = Self.randomTensor(n: Self.n, range: 4.0, dtype: dtype, seed: &seed)

                var unfused: Tensor!
                Self.runAndWait { cmd in
                    let dtSum = Ops.add(dtRaw, dtBias, on: cmd)
                    unfused = Ops.softplus(dtSum, on: cmd)
                }
                var fused: Tensor!
                Self.runAndWait { cmd in
                    fused = NemotronHMambaMixerFuse.addSoftplus(
                        dtRaw, dtBias, on: cmd, device: .shared)
                }

                let a = unfused.toFloatArray()
                let b = fused.toFloatArray()
                var mismatches = 0
                for i in 0 ..< Self.n where a[i] != b[i] {
                    mismatches += 1
                    if mismatches <= 3 {
                        print("[addSoftplus \(dtype)] mismatch at \(i): unfused=\(a[i]) fused=\(b[i])")
                    }
                }
                #expect(mismatches == 0, "addSoftplus (\(dtype)): \(mismatches)/\(Self.n) mismatches")
            }
        }
    }

    /// `mulSilu` is NOT bit-exact — see `NemotronHMambaMixerFuse`'s file
    /// header for the four independently-tried source/compile-flag
    /// combinations that all left the same ~20% of elements 1-2 ULP off
    /// the unfused chain (a GPU transcendental-unit context-sensitivity
    /// that didn't respond to anything controllable from Swift/MSL
    /// source). This test locks in that measured envelope — NOT
    /// bit-exact equality — as a regression guard: it fails loudly if a
    /// future change makes `mulSilu` meaningfully WORSE (e.g. a real
    /// formula bug) while tolerating the known small ULP noise. Given
    /// this residual, `mulSilu` stays gated OFF by default
    /// (`BUTTER_NEMOTRON_MIXER_FUSE_SILU=1` to opt in) — see
    /// `NemotronHText.swift`'s mamba mixer `decode` wiring.
    @Test("mulSilu matches Ops.silu + Ops.mul within the known ULP envelope, all dtypes")
    func mulSiluWithinKnownTolerance() {
        for dtype in Self.dtypes {
            autoreleasepool {
                var seed: UInt64 = 0xC2B2_AE3D_27D4_EB4F
                let (a, _) = Self.randomTensor(n: Self.n, range: 3.0, dtype: dtype, seed: &seed)
                let (z, _) = Self.randomTensor(n: Self.n, range: 3.0, dtype: dtype, seed: &seed)

                var unfused: Tensor!
                Self.runAndWait { cmd in
                    let zAct = Ops.silu(z, on: cmd)
                    unfused = Ops.mul(a, zAct, on: cmd)
                }
                var fused: Tensor!
                Self.runAndWait { cmd in
                    fused = NemotronHMambaMixerFuse.mulSilu(a, z, on: cmd, device: .shared)
                }

                let u = unfused.toFloatArray()
                let f = fused.toFloatArray()
                // Generous relative envelope (dtype's own ULP is far
                // tighter than this) — the point is catching a gross
                // regression, not re-litigating the known noise floor.
                let rtol: Float = 1e-2
                let atol: Float = 1e-3
                var outOfBand = 0
                for i in 0 ..< Self.n {
                    let bound = atol + rtol * abs(u[i])
                    if abs(u[i] - f[i]) > bound {
                        outOfBand += 1
                        if outOfBand <= 3 {
                            print("[mulSilu \(dtype)] out-of-band at \(i): unfused=\(u[i]) fused=\(f[i])")
                        }
                    }
                }
                #expect(outOfBand == 0, "mulSilu (\(dtype)): \(outOfBand)/\(Self.n) outside atol+rtol envelope")
            }
        }
    }

    @Test("addMul matches Ops.mul + Ops.add exactly, all dtypes")
    func addMulExact() {
        for dtype in Self.dtypes {
            autoreleasepool {
                var seed: UInt64 = 0x1656_67B1_9E37_79F9
                let (base, _) = Self.randomTensor(n: Self.n, range: 3.0, dtype: dtype, seed: &seed)
                let (a, _) = Self.randomTensor(n: Self.n, range: 3.0, dtype: dtype, seed: &seed)
                let (b, _) = Self.randomTensor(n: Self.n, range: 3.0, dtype: dtype, seed: &seed)

                var unfused: Tensor!
                Self.runAndWait { cmd in
                    let dx = Ops.mul(a, b, on: cmd)
                    unfused = Ops.add(base, dx, on: cmd)
                }
                var fused: Tensor!
                Self.runAndWait { cmd in
                    fused = NemotronHMambaMixerFuse.addMul(base: base, a, b, on: cmd, device: .shared)
                }

                let u = unfused.toFloatArray()
                let f = fused.toFloatArray()
                var mismatches = 0
                for i in 0 ..< Self.n where u[i] != f[i] {
                    mismatches += 1
                    if mismatches <= 3 {
                        print("[addMul \(dtype)] mismatch at \(i): unfused=\(u[i]) fused=\(f[i])")
                    }
                }
                #expect(mismatches == 0, "addMul (\(dtype)): \(mismatches)/\(Self.n) mismatches")
            }
        }
    }

    @Test("BUTTER_NEMOTRON_MIXER_FUSE=0 escape hatch reads live via ButterEnv")
    func escapeHatchReadsLive() {
        setenv("BUTTER_NEMOTRON_MIXER_FUSE", "0", 1)
        defer { unsetenv("BUTTER_NEMOTRON_MIXER_FUSE") }
        #expect(ButterEnv["BUTTER_NEMOTRON_MIXER_FUSE"] == "0")
    }
}
