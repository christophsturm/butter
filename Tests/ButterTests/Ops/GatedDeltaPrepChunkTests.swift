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
// CPU-oracle correctness tests for `Ops.gatedDeltaPrepChunk` — the fused
// chunked GDN prep + recurrence dispatch. The dispatch geometry (multi-SG
// reduction TG of 128 threads, grid X = ceil(Dv/4) TGs) previously had no
// CI coverage and was only verified manually; these tests pin the wrapper's
// grid math and the kernel's numerics against a sequential CPU reference.
//
// The oracle mirrors the Rust kernel-side oracle in
// `iron-std/src/kernels/ssm/gated_delta_prep_chunk.rs`
// (`kernel_tests::oracle`): per token t —
//   dt    = softplus(aRaw + dtBias)          (un-clamped: ln(e^x + 1))
//   g     = exp(-exp(aLog) · dt)
//   beta  = sigmoid(bRaw)
//   state = state · g                        (decay)
//   kvMem = Σ_dk state · kNormed             (memory readout)
//   delta = (v − kvMem) · beta
//   state = state + kNormed · delta          (rank-1 update)
//   y     = Σ_dk state · qNormed             (output projection)
// with the state threaded across the whole T-loop (register-resident on
// GPU; one load at entry, one store at exit).
//
// Buffer layouts match the kernel doc block exactly:
//   convOut  [B, T, 2·Hk·Dk + Hv·Dv]  (only the trailing v slab is read
//                                      here; q/k slabs feed the RMSNorm
//                                      pre-pass, not this kernel)
//   qNormed / kNormed  [B, T, Hk, Dk]  indexed as (bt·Hk + hkIdx)·Dk + d
//   stateIn / stateOut [B, Hv, Dv, Dk]
//   y        [B, T, Hv, Dv]
//
// qNormed/kNormed are synthesized directly (deterministic sin/cos, O(0.3)
// magnitude — the scale the RMSNorm pre-pass produces) rather than routed
// through `Ops.gatedDeltaQKNormPrepass`: this suite unit-tests the
// prep-chunk dispatch in isolation, and the kernel consumes those buffers
// as plain dense inputs.
//
// Input conditioning follows the Rust fixture: small conv/state scales and
// aLog ≤ −3 keep the per-step gain well under 1 so y stays O(1) and f32
// reduction-order error stays far inside tolerance (the recurrence
// amplifies hot inputs by ~Σ_t δ_t otherwise).
//
// NOTE — Dv % 4 != 0 (idle-SG partial-TG path): the kernel itself guards
// `dv_idx < dv` so its last TG may run partially idle when Dv is not a
// multiple of 4, but `Ops.gatedDeltaPrepChunk` preconditions
// `dv % 32 == 0` (strictly stronger than the kernel's Dv % 4 geometry).
// That makes the partial-TG path unreachable through the Ops layer, so it
// cannot be exercised from this suite; it remains covered only by the
// kernel-side Rust tests (Dv=8 there also fails the Ops precondition).

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("Ops.gatedDeltaPrepChunk — fused chunked GDN prep + recurrence")
struct GatedDeltaPrepChunkTests {

    /// Sequential CPU reference for the chunked GDN recurrence (state
    /// carried across the T-loop). Mirrors the Rust kernel-side oracle.
    static func oracle(
        convOut: [Float], aLog: [Float], dtBias: [Float],
        aRaw: [Float], bRaw: [Float],
        qNormed: [Float], kNormed: [Float], stateIn: [Float],
        b: Int, t: Int, hv: Int, hk: Int, dv: Int, dk: Int
    ) -> (y: [Float], state: [Float]) {
        let strideB = 2 * hk * dk + hv * dv
        let hkPerHv = hv / hk
        var y = [Float](repeating: 0, count: b * t * hv * dv)
        var state = stateIn  // carried across the T-loop
        for batch in 0 ..< b {
            for tt in 0 ..< t {
                let bt = batch * t + tt
                let vBase = bt * strideB + 2 * hk * dk
                for hvIdx in 0 ..< hv {
                    let n = batch * hv + hvIdx
                    let qkBase = (bt * hk + hvIdx / hkPerHv) * dk
                    let gbetaIdx = bt * hv + hvIdx
                    // Un-clamped softplus, matching the kernel exactly.
                    let dt = log(exp(aRaw[gbetaIdx] + dtBias[hvIdx]) + 1)
                    let g = exp(-exp(aLog[hvIdx]) * dt)
                    let beta = 1 / (1 + exp(-bRaw[gbetaIdx]))
                    for dvIdx in 0 ..< dv {
                        let vVal = convOut[vBase + hvIdx * dv + dvIdx]
                        let sBase = n * dv * dk + dvIdx * dk
                        // Phase 1: decay + kv_mem readout.
                        var kvMem: Float = 0
                        var decayed = [Float](repeating: 0, count: dk)
                        for d in 0 ..< dk {
                            let s = state[sBase + d] * g
                            decayed[d] = s
                            kvMem += s * kNormed[qkBase + d]
                        }
                        let delta = (vVal - kvMem) * beta
                        // Phase 2: rank-1 update + output projection.
                        var out: Float = 0
                        for d in 0 ..< dk {
                            let sNew = decayed[d] + kNormed[qkBase + d] * delta
                            state[sBase + d] = sNew
                            out += sNew * qNormed[qkBase + d]
                        }
                        y[(bt * hv + hvIdx) * dv + dvIdx] = out
                    }
                }
            }
        }
        return (y, state)
    }

    /// Build deterministic inputs (sin/cos patterns, no RNG), run the CPU
    /// oracle, dispatch `Ops.gatedDeltaPrepChunk`, and compare y AND
    /// stateOut element-wise within `tol` (max-abs).
    static func runCase(b: Int, t: Int, hv: Int, hk: Int, dv: Int, dk: Int, tol: Float) {
        autoreleasepool {
            let strideB = 2 * hk * dk + hv * dv
            // Deterministic pseudo-random inputs, conditioned like the Rust
            // fixture (conv 0.02 / state 0.01 / aLog −3) so y stays O(1).
            let convOutVals = (0 ..< b * t * strideB).map { Float(sin(Double($0) * 0.0131) * 0.02) }
            let aLogVals = (0 ..< hv).map { Float(-3.0 - Double($0) * 0.05) }
            let dtBiasVals = (0 ..< hv).map { Float(-0.5 + Double($0) * 0.05) }
            let aRawVals = (0 ..< b * t * hv).map { Float(-0.3 + Double($0) * 0.01) }
            let bRawVals = (0 ..< b * t * hv).map { Float(-0.2 + Double($0) * 0.008) }
            let qNormedVals = (0 ..< b * t * hk * dk).map {
                Float(sin(Double($0) * 0.017 + 0.3) * 0.3)
            }
            let kNormedVals = (0 ..< b * t * hk * dk).map {
                Float(cos(Double($0) * 0.019 + 0.1) * 0.3)
            }
            let stateInVals = (0 ..< b * hv * dv * dk).map {
                Float(cos(Double($0) * 0.0073) * 0.01)
            }

            let (yExp, stateExp) = oracle(
                convOut: convOutVals, aLog: aLogVals, dtBias: dtBiasVals,
                aRaw: aRawVals, bRaw: bRawVals,
                qNormed: qNormedVals, kNormed: kNormedVals, stateIn: stateInVals,
                b: b, t: t, hv: hv, hk: hk, dv: dv, dk: dk)

            let convOut = Tensor.empty(shape: [b, t, strideB], dtype: .f32)
            convOut.copyIn(from: convOutVals)
            let aLog = Tensor.empty(shape: [hv], dtype: .f32)
            aLog.copyIn(from: aLogVals)
            let dtBias = Tensor.empty(shape: [hv], dtype: .f32)
            dtBias.copyIn(from: dtBiasVals)
            let aRaw = Tensor.empty(shape: [b, t, hv], dtype: .f32)
            aRaw.copyIn(from: aRawVals)
            let bRaw = Tensor.empty(shape: [b, t, hv], dtype: .f32)
            bRaw.copyIn(from: bRawVals)
            let qNormed = Tensor.empty(shape: [b, t, hk, dk], dtype: .f32)
            qNormed.copyIn(from: qNormedVals)
            let kNormed = Tensor.empty(shape: [b, t, hk, dk], dtype: .f32)
            kNormed.copyIn(from: kNormedVals)
            let stateIn = Tensor.empty(shape: [b, hv, dv, dk], dtype: .f32)
            stateIn.copyIn(from: stateInVals)
            let stateOut = Tensor.empty(shape: [b, hv, dv, dk], dtype: .f32)
            stateOut.zero()
            let y = Tensor.empty(shape: [b, t, hv, dv], dtype: .f32)
            y.zero()
            // Runtime [1] u32 chunk-length scalar (GPU-resident, no CPU
            // readback needed between chained chunk dispatches).
            let tLen = Tensor.empty(shape: [1], dtype: .u32)
            tLen.copyIn(from: [UInt32(t)])

            let (statePlanes, planesEnabled) = Ops.disabledGdnPlanes(dtype: .f32, device: .shared)
            runAndWait { cb in
                Ops.gatedDeltaPrepChunk(
                    convOut: convOut, aLog: aLog, dtBias: dtBias,
                    aRaw: aRaw, bRaw: bRaw,
                    qNormed: qNormed, kNormed: kNormed,
                    stateIn: stateIn, stateOut: stateOut, y: y,
                    tLen: tLen,
                    statePlanes: statePlanes, planesEnabled: planesEnabled,
                    batchSize: b, dk: dk, dv: dv, hv: hv, hk: hk, on: cb)
            }

            let yGot = y.toArray(as: Float.self)
            #expect(yGot.count == yExp.count)
            var yMaxErr: Float = 0
            for i in 0 ..< yExp.count { yMaxErr = max(yMaxErr, abs(yGot[i] - yExp[i])) }
            #expect(yMaxErr < tol, "y max-abs err \(yMaxErr) ≥ tol \(tol)")

            let stateGot = stateOut.toArray(as: Float.self)
            #expect(stateGot.count == stateExp.count)
            var sMaxErr: Float = 0
            for i in 0 ..< stateExp.count { sMaxErr = max(sMaxErr, abs(stateGot[i] - stateExp[i])) }
            #expect(sMaxErr < tol, "stateOut max-abs err \(sMaxErr) ≥ tol \(tol)")
        }
    }

    // Dv=32 is the smallest shape the Ops precondition (dv % 32 == 0)
    // admits; it spans 8 TGs on grid X, exercising the multi-TG Dv split
    // (the brief's Dv=8 / Dv=6 partial-TG shapes are Ops-unreachable — see
    // the file header note). Tolerance 1e-4: y is O(1) by construction and
    // the only GPU-vs-CPU divergence is simd_sum reduction order, which the
    // kernel-side f32 tests bound well below 5e-3 at identical conditioning.
    @Test("GQA hv=2·hk — T=4, Hv=4, Hk=2, Dv=32, Dk=64: y + stateOut match CPU oracle")
    func gqaChunkMatchesOracle() {
        Self.runCase(b: 1, t: 4, hv: 4, hk: 2, dv: 32, dk: 64, tol: 1e-4)
    }

    // Hv == Hk (no key-sharing) at the minimum Dk=32 (one state slot per
    // lane), T=3 — mirrors the Rust `no_gqa` fixture's shape at the
    // Ops-admissible Dv.
    @Test("no GQA hv==hk — T=3, Hv=4, Hk=4, Dv=32, Dk=32: y + stateOut match CPU oracle")
    func noGqaChunkMatchesOracle() {
        Self.runCase(b: 1, t: 3, hv: 4, hk: 4, dv: 32, dk: 32, tol: 1e-4)
    }

    // F-85 follow-up audit gap: `git grep` found no XCTest/Swift-Testing
    // coverage at all for `Ops.gatedDeltaPrepChunkFast` (only the generic
    // `Ops.gatedDeltaPrepChunk` above was covered) despite it being the
    // default dispatch path for this model's production GDN shape, see
    // `Qwen35GDNMixer.forwardManyChunked`'s `fastShapeMatches` /
    // `BUTTER_GDN_NO_PREP_CHUNK_FAST` gate. Same oracle/math as
    // `gqaChunkMatchesOracle` above (the fast kernel is a pure shape
    // specialization, bit-identical formula), dispatched via
    // `Ops.gatedDeltaPrepChunkFast` at the fixed production shape
    // (Dk=Dv=128, Hv=32, Hk=16) that wrapper hardcodes internally.
    static func runCaseFast(t: Int, tol: Float) {
        let (b, hv, hk, dv, dk) = (1, 32, 16, 128, 128)
        autoreleasepool {
            let strideB = 2 * hk * dk + hv * dv
            let convOutVals = (0 ..< b * t * strideB).map { Float(sin(Double($0) * 0.0131) * 0.02) }
            let aLogVals = (0 ..< hv).map { Float(-3.0 - Double($0) * 0.05) }
            let dtBiasVals = (0 ..< hv).map { Float(-0.5 + Double($0) * 0.05) }
            let aRawVals = (0 ..< b * t * hv).map { Float(-0.3 + Double($0) * 0.01) }
            let bRawVals = (0 ..< b * t * hv).map { Float(-0.2 + Double($0) * 0.008) }
            let qNormedVals = (0 ..< b * t * hk * dk).map {
                Float(sin(Double($0) * 0.017 + 0.3) * 0.3)
            }
            let kNormedVals = (0 ..< b * t * hk * dk).map {
                Float(cos(Double($0) * 0.019 + 0.1) * 0.3)
            }
            let stateInVals = (0 ..< b * hv * dv * dk).map {
                Float(cos(Double($0) * 0.0073) * 0.01)
            }

            let (yExp, stateExp) = oracle(
                convOut: convOutVals, aLog: aLogVals, dtBias: dtBiasVals,
                aRaw: aRawVals, bRaw: bRawVals,
                qNormed: qNormedVals, kNormed: kNormedVals, stateIn: stateInVals,
                b: b, t: t, hv: hv, hk: hk, dv: dv, dk: dk)

            let convOut = Tensor.empty(shape: [b, t, strideB], dtype: .f32)
            convOut.copyIn(from: convOutVals)
            let aLog = Tensor.empty(shape: [hv], dtype: .f32)
            aLog.copyIn(from: aLogVals)
            let dtBias = Tensor.empty(shape: [hv], dtype: .f32)
            dtBias.copyIn(from: dtBiasVals)
            let aRaw = Tensor.empty(shape: [b, t, hv], dtype: .f32)
            aRaw.copyIn(from: aRawVals)
            let bRaw = Tensor.empty(shape: [b, t, hv], dtype: .f32)
            bRaw.copyIn(from: bRawVals)
            let qNormed = Tensor.empty(shape: [b, t, hk, dk], dtype: .f32)
            qNormed.copyIn(from: qNormedVals)
            let kNormed = Tensor.empty(shape: [b, t, hk, dk], dtype: .f32)
            kNormed.copyIn(from: kNormedVals)
            let stateIn = Tensor.empty(shape: [b, hv, dv, dk], dtype: .f32)
            stateIn.copyIn(from: stateInVals)
            let stateOut = Tensor.empty(shape: [b, hv, dv, dk], dtype: .f32)
            stateOut.zero()
            let y = Tensor.empty(shape: [b, t, hv, dv], dtype: .f32)
            y.zero()
            let tLen = Tensor.empty(shape: [1], dtype: .u32)
            tLen.copyIn(from: [UInt32(t)])

            let (statePlanes, planesEnabled) = Ops.disabledGdnPlanes(dtype: .f32, device: .shared)
            runAndWait { cb in
                Ops.gatedDeltaPrepChunkFast(
                    convOut: convOut, aLog: aLog, dtBias: dtBias,
                    aRaw: aRaw, bRaw: bRaw,
                    qNormed: qNormed, kNormed: kNormed,
                    stateIn: stateIn, stateOut: stateOut, y: y,
                    tLen: tLen,
                    statePlanes: statePlanes, planesEnabled: planesEnabled,
                    batchSize: b, on: cb)
            }

            let yGot = y.toArray(as: Float.self)
            #expect(yGot.count == yExp.count)
            var yMaxErr: Float = 0
            for i in 0 ..< yExp.count { yMaxErr = max(yMaxErr, abs(yGot[i] - yExp[i])) }
            #expect(yMaxErr < tol, "y max-abs err \(yMaxErr) ≥ tol \(tol)")

            let stateGot = stateOut.toArray(as: Float.self)
            #expect(stateGot.count == stateExp.count)
            var sMaxErr: Float = 0
            for i in 0 ..< stateExp.count { sMaxErr = max(sMaxErr, abs(stateGot[i] - stateExp[i])) }
            #expect(sMaxErr < tol, "stateOut max-abs err \(sMaxErr) ≥ tol \(tol)")
        }
    }

    // Qwen3.6-35B-A3B production shape (Dk=Dv=128, Hv=32, Hk=16), the
    // exact shape `Qwen35GDNMixer.forwardManyChunked`'s `fastShapeMatches`
    // guard requires before dispatching this kernel. T kept small (3
    // tokens) for a fast correctness check; isolated/end-to-end perf are
    // covered by the Rust kernel benches, not this suite. Tol matches the
    // Rust `test_butter_gated_delta_prep_chunk_fast_d128_128_32_16` cell.
    @Test("fast path production shape, Dk=Dv=128, Hv=32, Hk=16, T=3: y + stateOut match CPU oracle")
    func fastPathProductionShapeMatchesOracle() {
        Self.runCaseFast(t: 3, tol: 1e-2)
    }

    // Confirms the fast and generic kernels agree with EACH OTHER, not just
    // with the CPU oracle independently: a pure shape specialization
    // should be bit-identical math, so this is the same invariant the
    // Rust-side quality gate checks (cosine 1.0 expected), pinned here at
    // the Ops-dispatch layer too. Dv=128/Hv=32/Hk=16/Dk=128 is required by
    // `gatedDeltaPrepChunkFast`; the generic kernel accepts the identical
    // shape via its `dk`/`dv`/`hv`/`hk` params, so both sides run the exact
    // same inputs.
    @Test("fast path matches generic kernel bit-for-bit at the shared production shape")
    func fastPathMatchesGenericKernel() {
        let (b, t, hv, hk, dv, dk) = (1, 3, 32, 16, 128, 128)
        autoreleasepool {
            let strideB = 2 * hk * dk + hv * dv
            let convOutVals = (0 ..< b * t * strideB).map { Float(sin(Double($0) * 0.0131) * 0.02) }
            let aLogVals = (0 ..< hv).map { Float(-3.0 - Double($0) * 0.05) }
            let dtBiasVals = (0 ..< hv).map { Float(-0.5 + Double($0) * 0.05) }
            let aRawVals = (0 ..< b * t * hv).map { Float(-0.3 + Double($0) * 0.01) }
            let bRawVals = (0 ..< b * t * hv).map { Float(-0.2 + Double($0) * 0.008) }
            let qNormedVals = (0 ..< b * t * hk * dk).map {
                Float(sin(Double($0) * 0.017 + 0.3) * 0.3)
            }
            let kNormedVals = (0 ..< b * t * hk * dk).map {
                Float(cos(Double($0) * 0.019 + 0.1) * 0.3)
            }
            let stateInVals = (0 ..< b * hv * dv * dk).map {
                Float(cos(Double($0) * 0.0073) * 0.01)
            }

            func buffers() -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
                let convOut = Tensor.empty(shape: [b, t, strideB], dtype: .f32)
                convOut.copyIn(from: convOutVals)
                let aLog = Tensor.empty(shape: [hv], dtype: .f32)
                aLog.copyIn(from: aLogVals)
                let dtBias = Tensor.empty(shape: [hv], dtype: .f32)
                dtBias.copyIn(from: dtBiasVals)
                let aRaw = Tensor.empty(shape: [b, t, hv], dtype: .f32)
                aRaw.copyIn(from: aRawVals)
                let bRaw = Tensor.empty(shape: [b, t, hv], dtype: .f32)
                bRaw.copyIn(from: bRawVals)
                let qNormed = Tensor.empty(shape: [b, t, hk, dk], dtype: .f32)
                qNormed.copyIn(from: qNormedVals)
                let kNormed = Tensor.empty(shape: [b, t, hk, dk], dtype: .f32)
                kNormed.copyIn(from: kNormedVals)
                let stateIn = Tensor.empty(shape: [b, hv, dv, dk], dtype: .f32)
                stateIn.copyIn(from: stateInVals)
                let stateOut = Tensor.empty(shape: [b, hv, dv, dk], dtype: .f32)
                stateOut.zero()
                let y = Tensor.empty(shape: [b, t, hv, dv], dtype: .f32)
                y.zero()
                let tLen = Tensor.empty(shape: [1], dtype: .u32)
                tLen.copyIn(from: [UInt32(t)])
                return (convOut, aLog, dtBias, aRaw, bRaw, qNormed, kNormed, stateIn, stateOut, y, tLen)
            }

            let (convOutG, aLogG, dtBiasG, aRawG, bRawG, qNormedG, kNormedG, stateInG, stateOutG, yG, tLenG) =
                buffers()
            let (statePlanesG, planesEnabledG) = Ops.disabledGdnPlanes(dtype: .f32, device: .shared)
            runAndWait { cb in
                Ops.gatedDeltaPrepChunk(
                    convOut: convOutG, aLog: aLogG, dtBias: dtBiasG,
                    aRaw: aRawG, bRaw: bRawG,
                    qNormed: qNormedG, kNormed: kNormedG,
                    stateIn: stateInG, stateOut: stateOutG, y: yG,
                    tLen: tLenG,
                    statePlanes: statePlanesG, planesEnabled: planesEnabledG,
                    batchSize: b, dk: dk, dv: dv, hv: hv, hk: hk, on: cb)
            }

            let (convOutF, aLogF, dtBiasF, aRawF, bRawF, qNormedF, kNormedF, stateInF, stateOutF, yF, tLenF) =
                buffers()
            let (statePlanesF, planesEnabledF) = Ops.disabledGdnPlanes(dtype: .f32, device: .shared)
            runAndWait { cb in
                Ops.gatedDeltaPrepChunkFast(
                    convOut: convOutF, aLog: aLogF, dtBias: dtBiasF,
                    aRaw: aRawF, bRaw: bRawF,
                    qNormed: qNormedF, kNormed: kNormedF,
                    stateIn: stateInF, stateOut: stateOutF, y: yF,
                    tLen: tLenF,
                    statePlanes: statePlanesF, planesEnabled: planesEnabledF,
                    batchSize: b, on: cb)
            }

            let yGenVals = yG.toArray(as: Float.self)
            let yFastVals = yF.toArray(as: Float.self)
            #expect(yGenVals.count == yFastVals.count)
            for i in 0 ..< yGenVals.count {
                #expect(yGenVals[i] == yFastVals[i], "y[\(i)] generic=\(yGenVals[i]) fast=\(yFastVals[i])")
            }

            let stateGenVals = stateOutG.toArray(as: Float.self)
            let stateFastVals = stateOutF.toArray(as: Float.self)
            #expect(stateGenVals.count == stateFastVals.count)
            for i in 0 ..< stateGenVals.count {
                #expect(
                    stateGenVals[i] == stateFastVals[i],
                    "stateOut[\(i)] generic=\(stateGenVals[i]) fast=\(stateFastVals[i])")
            }
        }
    }
}
