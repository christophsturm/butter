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
// F-130: `NemotronHMambaGroupedSSMFuse` — oracle test comparing the
// single wide-grid `ssmStepGrouped` dispatch against the exact unfused
// 8x per-group `Ops.ssmStep` loop it replaces (`NemotronHMambaLayer.
// decode`'s original selective-scan dispatch pattern), at all three
// activation dtypes.
//
// The mamba2 SSM state `h` is RECURRENT — it carries across every
// decoded token, so any drift (even 1 ULP) compounds and eventually
// flips an argmax somewhere in a long decode. Unlike the F-129
// elementwise fusions (which touch one token's worth of data and get
// re-checked fresh each call), this fusion's correctness bar is
// therefore stricter: strict bit-for-bit equality is verified not just
// per-step but across a 32-STEP sequential recurrence, comparing both
// the output `y` and the state buffer `h` itself after every step —
// catching a fusion that matches on step 1 but silently diverges by
// step 20.
//
// Shapes match Nemotron 3.5 Lightning 30B-A3B's actual mamba mixer
// config (`~/models/nemotron35-lightning-nvfp4/config.json`):
// `mamba_num_heads=64`, `mamba_head_dim=64`, `ssm_state_size=128`,
// `n_groups=8` (`headsPerGroup = 64/8 = 8`).

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("NemotronH grouped-SSM fused decode step vs the unfused 8x per-group loop")
struct NemotronHMambaGroupedSSMFuseTests {

    private static func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let cmd = Device.shared.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "cmd status \(cmd.status.rawValue)")
    }

    /// Round-to-nearest-even f32 -> bf16, expressed as the top 16 bits
    /// of the f32 bit pattern (same convention as `OpsDtypeTests` /
    /// `NemotronHMambaMixerFuseTests`).
    private static func f32ToBf16(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let lo = bits & 0xFFFF
        let hi = UInt16(bits >> 16)
        let roundUp = lo > 0x8000 || (lo == 0x8000 && (hi & 1) == 1)
        return roundUp ? hi &+ 1 : hi
    }

    /// Deterministic xorshift* generator — same reseed pattern as the
    /// F-127/F-129 fused-kernel tests, small and dependency-free.
    private static func pseudoRandomUniform(_ seed: inout UInt64) -> Float {
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        // [0, 1)
        return Float(seed >> 40) / Float(1 << 24)
    }

    /// Build a `[n]` tensor of `dtype` with values uniform in
    /// `[lo, hi]`, returning the tensor (the AS-STORED, dtype-rounded
    /// values aren't needed here — both sides of every comparison read
    /// back through `toFloatArray()`, so only the tensor is needed).
    private static func randomTensor(
        n: Int, lo: Float, hi: Float, dtype: DType, seed: inout UInt64
    ) -> Tensor {
        let raw: [Float] = (0 ..< n).map { _ in
            lo + pseudoRandomUniform(&seed) * (hi - lo)
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
        return t
    }

    // Nemotron 3.5 Lightning 30B-A3B's actual mamba mixer shape.
    private static let nHeads = 64
    private static let headDim = 64
    private static let stateDim = 128
    private static let nGroups = 8
    private static let headsPerGroup = nHeads / nGroups
    private static let nSteps = 32

    private static let dtypes: [DType] = [.f32, .f16, .bf16]

    /// Runs the unfused 8x per-group `Ops.ssmStep` loop — an exact
    /// transcription of `NemotronHMambaLayer.decode`'s step (6) — for
    /// one decode step, mutating `state` in place.
    private static func referenceStep(
        x: Tensor, aEff: Tensor, bAll: Tensor, cAll: Tensor, dt: Tensor,
        state: Tensor, dtype: DType
    ) -> Tensor {
        var y: Tensor!
        Self.runAndWait { cmd in
            let out = Tensor.empty(shape: [nHeads, headDim], dtype: dtype)
            let xHeads = x.reshaped(to: [nHeads, headDim])
            let bAllR = bAll.reshaped(to: [nGroups, stateDim])
            let cAllR = cAll.reshaped(to: [nGroups, stateDim])
            for g in 0 ..< nGroups {
                let h0 = g * headsPerGroup
                let xg = xHeads.slicedRows(start: h0, count: headsPerGroup)
                    .reshaped(to: [headsPerGroup * headDim])
                let yg = out.slicedRows(start: h0, count: headsPerGroup)
                let stateG = state.slicedRows(start: h0, count: headsPerGroup)
                let aG = aEff.slicedRows(start: h0, count: headsPerGroup)
                let dtG = dt.slicedRows(start: h0, count: headsPerGroup)
                let bG = bAllR.slicedRows(start: g, count: 1).reshaped(to: [stateDim])
                let cG = cAllR.slicedRows(start: g, count: 1).reshaped(to: [stateDim])
                Ops.ssmStep(
                    x: xg, a: aG, b: bG, c: cG, dt: dtG,
                    state: stateG, into: yg,
                    nHeads: headsPerGroup, headDim: headDim, stateDim: stateDim,
                    on: cmd)
            }
            y = out
        }
        return y
    }

    /// Runs the fused single-dispatch `ssmStepGrouped` for one decode
    /// step, mutating `state` in place.
    private static func fusedStep(
        x: Tensor, aEff: Tensor, bAll: Tensor, cAll: Tensor, dt: Tensor,
        state: Tensor, dtype: DType
    ) -> Tensor {
        var y: Tensor!
        Self.runAndWait { cmd in
            let out = Tensor.empty(shape: [nHeads, headDim], dtype: dtype)
            NemotronHMambaGroupedSSMFuse.ssmStepGrouped(
                x: x, a: aEff, b: bAll, c: cAll, dt: dt,
                state: state, into: out,
                nHeads: nHeads, headDim: headDim, stateDim: stateDim,
                headsPerGroup: headsPerGroup,
                on: cmd, device: .shared)
            y = out
        }
        return y
    }

    @Test(
        "ssmStepGrouped matches the unfused 8x per-group loop exactly across 32 sequential steps, all dtypes",
        arguments: Self.dtypes
    )
    func groupedMatchesLoopedAcrossRecurrence(dtype: DType) {
        autoreleasepool {
            var seed: UInt64 = 0xA24B_AED4_963E_E407 &+ UInt64(bitPattern: Int64(dtype.hashValue &* 7))

            // `aEff` (`-exp(A_log)`) is a fixed per-head load-time
            // constant in the real model — negative, so decay < 1 and
            // the recurrence stays bounded over 32 steps (matches how
            // Mamba is actually parameterized; avoids a spurious
            // overflow-vs-not divergence between the two code paths
            // that would just be measuring float overflow semantics,
            // not the fusion's correctness).
            let aEff = Self.randomTensor(n: Self.nHeads, lo: -2.0, hi: -0.01, dtype: dtype, seed: &seed)

            // Two independently-tracked recurrent state buffers,
            // seeded identically, then advanced by the reference loop
            // and the fused dispatch respectively.
            let stateN = Self.nHeads * Self.stateDim * Self.headDim
            let initState: [Float] = (0 ..< stateN).map { _ in
                Self.pseudoRandomUniform(&seed) * 2 - 1
            }
            let stateRef = Tensor.empty(shape: [Self.nHeads, Self.stateDim, Self.headDim], dtype: .f32)
            let stateFused = Tensor.empty(shape: [Self.nHeads, Self.stateDim, Self.headDim], dtype: .f32)
            stateRef.copyIn(from: initState)
            stateFused.copyIn(from: initState)

            var stop = false
            for step in 0 ..< Self.nSteps where !stop {
                autoreleasepool {
                    let x = Self.randomTensor(n: Self.nHeads * Self.headDim, lo: -2.0, hi: 2.0, dtype: dtype, seed: &seed)
                    // dt mimics a softplus output: positive.
                    let dt = Self.randomTensor(n: Self.nHeads, lo: 0.01, hi: 2.0, dtype: dtype, seed: &seed)
                    let bAll = Self.randomTensor(n: Self.nGroups * Self.stateDim, lo: -2.0, hi: 2.0, dtype: dtype, seed: &seed)
                    let cAll = Self.randomTensor(n: Self.nGroups * Self.stateDim, lo: -2.0, hi: 2.0, dtype: dtype, seed: &seed)

                    let yRef = Self.referenceStep(
                        x: x, aEff: aEff, bAll: bAll, cAll: cAll, dt: dt,
                        state: stateRef, dtype: dtype)
                    let yFused = Self.fusedStep(
                        x: x, aEff: aEff, bAll: bAll, cAll: cAll, dt: dt,
                        state: stateFused, dtype: dtype)

                    let yr = yRef.toFloatArray()
                    let yf = yFused.toFloatArray()
                    var yMismatches = 0
                    for i in 0 ..< yr.count where yr[i] != yf[i] {
                        yMismatches += 1
                        if yMismatches <= 3 {
                            print("[ssmStepGrouped \(dtype) step \(step)] y mismatch at \(i): ref=\(yr[i]) fused=\(yf[i])")
                        }
                    }
                    #expect(yMismatches == 0, "ssmStepGrouped (\(dtype)) step \(step): \(yMismatches)/\(yr.count) y mismatches")

                    let sr = stateRef.toFloatArray()
                    let sf = stateFused.toFloatArray()
                    var sMismatches = 0
                    for i in 0 ..< sr.count where sr[i] != sf[i] {
                        sMismatches += 1
                        if sMismatches <= 3 {
                            print("[ssmStepGrouped \(dtype) step \(step)] state mismatch at \(i): ref=\(sr[i]) fused=\(sf[i])")
                        }
                    }
                    #expect(sMismatches == 0, "ssmStepGrouped (\(dtype)) step \(step): \(sMismatches)/\(sr.count) state mismatches")

                    // Stop early on the first divergent step rather
                    // than spamming 31 more failures once broken.
                    if yMismatches != 0 || sMismatches != 0 {
                        stop = true
                    }
                }
            }
        }
    }

    @Test("BUTTER_NEMOTRON_SSM_GROUPED=0 escape hatch reads live via ButterEnv")
    func escapeHatchReadsLive() {
        setenv("BUTTER_NEMOTRON_SSM_GROUPED", "0", 1)
        defer { unsetenv("BUTTER_NEMOTRON_SSM_GROUPED") }
        #expect(ButterEnv["BUTTER_NEMOTRON_SSM_GROUPED"] == "0")
    }
}
