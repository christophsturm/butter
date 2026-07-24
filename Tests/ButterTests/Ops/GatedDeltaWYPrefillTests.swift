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
// Equivalence tests for `Ops.gatedDeltaWYPrefill` — the chunked-WY GDN
// prefill pipeline (`iron_gdn_wy_plan` + `iron_gdn_wy_scan` + the
// `Ops.gdnGateBeta` gate pre-pass) — against `Ops.gatedDeltaPrepChunk`,
// the production recurrence it opts in as a replacement for
// (`BUTTER_GDN_WY=1`). Both paths consume identical raw inputs
// (convOut / aRaw / bRaw / qNormed / kNormed / state) and must produce
// the same `y` and `stateOut` up to reduction-order + f16-staging noise.
//
// The WY plan kernel stages its triangular-solve scratch in f16 and its
// gate products run through exp/log rather than a per-token multiplicative
// recurrence, so tolerances here are wider than the prep-chunk oracle
// suite's 1e-4 — 5e-3 max-abs on O(1) outputs, in line with the
// kernel-side Rust tolerance for the same kernels.
//
// Gate conditioning intentionally covers the REALISTIC Qwen3.6 magnitude
// (`exp(a_log)` in ~[1, 16], per the GDN family's A_log init): per-token
// decay g can be ~e⁻⁴, so a 64-token chunk's prefix gate product G
// underflows f32 (e⁻²⁵⁶). The plan kernel therefore accumulates gates in
// log space (see gated_delta_wy_plan.rs) — this suite is what catches a
// regression to naive prefix products (which NaN via 0/0 in Γ = G_t/G_j
// and emit garbage e2e).

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

@Suite("Ops.gatedDeltaWYPrefill — chunked-WY GDN prefill vs prep-chunk")
struct GatedDeltaWYPrefillTests {
    /// The WY kernels are MPP cooperative-tensor (live-compiled) — the
    /// MPP matmul2d body does not compile below Apple GPU Family 9, so
    /// gate exactly like the other MPP suites (CI runners skip).
    static let mppAvailable: Bool = {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple9)
    }()


    /// Runs both paths on identical inputs and compares y + stateOut.
    /// `aLog0` conditions the gate magnitude: −3 (benign, prep-chunk
    /// fixture conditioning) or +2.08 ≈ log(8) (realistic Qwen3.6 A_log —
    /// the underflow regime the log-space gate accumulation exists for).
    static func runCase(
        t: Int, hv: Int, hk: Int, dv: Int, dk: Int,
        aLog0: Float, tol: Float
    ) {
        autoreleasepool {
            let b = 1
            let strideB = 2 * hk * dk + hv * dv
            let convOutVals = (0 ..< b * t * strideB).map { Float(sin(Double($0) * 0.0131) * 0.02) }
            let aLogVals = (0 ..< hv).map { aLog0 - Float($0) * 0.05 }
            let dtBiasVals = (0 ..< hv).map { Float(-0.5 + Double($0) * 0.05) }
            let aRawVals = (0 ..< b * t * hv).map { Float(-0.3 + Double($0 % 977) * 0.001) }
            let bRawVals = (0 ..< b * t * hv).map { Float(-0.2 + Double($0 % 991) * 0.0008) }
            // sqrt(2/dk) amplitude ⇒ ||k_t||² ≈ 1, matching what the
            // RMSNorm pre-pass feeds the kernels in production. An
            // unnormalised amplitude (e.g. flat 0.3 at dk=128 → ||k||²≈5.8)
            // makes the plan kernel's (I+L) forward-substitution grow
            // exponentially over the C=64 chunk and overflow its f16
            // tg_solve staging — a conditioning regime real inputs
            // never enter (see the kernel's f16-staging rationale).
            let qkScale = (2.0 / Double(dk)).squareRoot()
            let qNormedVals = (0 ..< b * t * hk * dk).map {
                Float(sin(Double($0) * 0.017 + 0.3) * qkScale)
            }
            let kNormedVals = (0 ..< b * t * hk * dk).map {
                Float(cos(Double($0) * 0.019 + 0.1) * qkScale)
            }
            let stateInVals = (0 ..< b * hv * dv * dk).map {
                Float(cos(Double($0) * 0.0073) * 0.01)
            }

            func f32Tensor(_ vals: [Float], shape: [Int]) -> Tensor {
                let t = Tensor.empty(shape: shape, dtype: .f32)
                t.copyIn(from: vals)
                return t
            }

            let convOut = f32Tensor(convOutVals, shape: [b, t, strideB])
            let aLog = f32Tensor(aLogVals, shape: [hv])
            let dtBias = f32Tensor(dtBiasVals, shape: [hv])
            let negExpALog = f32Tensor(aLogVals.map { -exp($0) }, shape: [hv])
            let aRaw = f32Tensor(aRawVals, shape: [b, t, hv])
            let bRaw = f32Tensor(bRawVals, shape: [b, t, hv])
            let qNormed = f32Tensor(qNormedVals, shape: [b, t, hk, dk])
            let kNormed = f32Tensor(kNormedVals, shape: [b, t, hk, dk])
            let tLen = Tensor.empty(shape: [1], dtype: .u32)
            tLen.copyIn(from: [UInt32(t)])

            // ── Reference: prep-chunk path ──────────────────────────────
            let stateInA = f32Tensor(stateInVals, shape: [b, hv, dv, dk])
            let stateOutA = Tensor.empty(shape: [b, hv, dv, dk], dtype: .f32)
            stateOutA.zero()
            let yA = Tensor.empty(shape: [b, t, hv, dv], dtype: .f32)
            yA.zero()
            runAndWait { cb in
                Ops.gatedDeltaPrepChunk(
                    convOut: convOut, aLog: aLog, dtBias: dtBias,
                    aRaw: aRaw, bRaw: bRaw,
                    qNormed: qNormed, kNormed: kNormed,
                    stateIn: stateInA, stateOut: stateOutA, y: yA,
                    tLen: tLen,
                    batchSize: b, dk: dk, dv: dv, hv: hv, hk: hk, on: cb)
            }

            // ── Candidate: chunked-WY pipeline ──────────────────────────
            let stateInB = f32Tensor(stateInVals, shape: [b, hv, dv, dk])
            let stateOutB = Tensor.empty(shape: [b, hv, dv, dk], dtype: .f32)
            stateOutB.zero()
            let yB = Tensor.empty(shape: [b, t, hv, dv], dtype: .f32)
            yB.zero()
            runAndWait { cb in
                Ops.gatedDeltaWYPrefill(
                    convOut: convOut,
                    dtBias: dtBias, negExpALog: negExpALog,
                    aRaw: aRaw, bRaw: bRaw,
                    qNormed: qNormed, kNormed: kNormed,
                    stateIn: stateInB, stateOut: stateOutB, y: yB,
                    tLen: tLen, t: t,
                    dk: dk, dv: dv, hv: hv, hk: hk,
                    on: cb, device: .shared)
            }

            let yRef = yA.toArray(as: Float.self)
            let yGot = yB.toArray(as: Float.self)
            #expect(yGot.count == yRef.count)
            var yMaxErr: Float = 0
            var yNaN = 0
            for i in 0 ..< yRef.count {
                if yGot[i].isNaN { yNaN += 1 }
                yMaxErr = max(yMaxErr, abs(yGot[i] - yRef[i]))
            }
            #expect(yNaN == 0, "WY y contains \(yNaN) NaNs")
            #expect(yMaxErr < tol, "y max-abs err \(yMaxErr) ≥ tol \(tol)")

            let sRef = stateOutA.toArray(as: Float.self)
            let sGot = stateOutB.toArray(as: Float.self)
            #expect(sGot.count == sRef.count)
            var sMaxErr: Float = 0
            var sNaN = 0
            for i in 0 ..< sRef.count {
                if sGot[i].isNaN { sNaN += 1 }
                sMaxErr = max(sMaxErr, abs(sGot[i] - sRef[i]))
            }
            #expect(sNaN == 0, "WY stateOut contains \(sNaN) NaNs")
            #expect(sMaxErr < tol, "stateOut max-abs err \(sMaxErr) ≥ tol \(tol)")
        }
    }

    // Benign gates (prep-chunk fixture conditioning, aLog ≤ −3 → g ≈ 0.97):
    // validates the wiring (layouts, grids, g/beta math, v slicing) without
    // stressing the gate-product range. T=128 = 2 chunks of C=64, GQA 2:1.
    @Test(
        "benign gates — T=128, Hv=4, Hk=2, Dv=32, Dk=32: WY matches prep-chunk",
        .enabled(if: mppAvailable, "MPP cooperative tensor — Apple GPU Family 9+"))
    func benignGatesMatch() {
        Self.runCase(t: 128, hv: 4, hk: 2, dv: 32, dk: 32, aLog0: -3.0, tol: 5e-3)
    }

    // Realistic Qwen3.6 gate magnitude (exp(aLog) ≈ 8): per-token g ~e⁻⁴,
    // 64-token prefix product ~e⁻²⁵⁶ — f32-underflow regime. Guards the
    // plan kernel's log-space gate accumulation (naive prefix products NaN
    // here and emitted "!!!…" garbage e2e before the fix).
    @Test(
        "realistic gates — T=128, Hv=4, Hk=2, Dv=32, Dk=32: WY matches prep-chunk",
        .enabled(if: mppAvailable, "MPP cooperative tensor — Apple GPU Family 9+"))
    func realisticGatesMatch() {
        Self.runCase(t: 128, hv: 4, hk: 2, dv: 32, dk: 32, aLog0: 2.08, tol: 5e-3)
    }

    // Production geometry (Dk=Dv=128, GQA 2:1) at reduced head count for
    // test speed — exercises the full-width MMA tiles + 4-Dv-tile scan.
    @Test(
        "production dims — T=64, Hv=4, Hk=2, Dv=128, Dk=128: WY matches prep-chunk",
        .enabled(if: mppAvailable, "MPP cooperative tensor — Apple GPU Family 9+"))
    func productionDimsMatch() {
        Self.runCase(t: 64, hv: 4, hk: 2, dv: 128, dk: 128, aLog0: 2.08, tol: 5e-3)
    }
}
