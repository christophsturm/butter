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
// suite's 1e-4 — normally 5e-3 max-abs on O(1) outputs, in line with the
// kernel-side Rust tolerance for the same kernels. The moderate-decay
// Family-9 case allows 1e-2 for its measured cooperative-tensor reduction
// drift; Family 10 retains the original 5e-3 bar.
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
    /// Non-periodic deterministic source for Q/K fixtures. Repeating
    /// sinusoids collapse the 64-row Gram matrix toward very low rank,
    /// which intentionally exercises WY's documented ill-conditioned
    /// algorithmic edge case rather than the wrapper wiring tested here.
    struct RNG {
        var state: UInt64

        mutating func signedUnit() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 40) / Float(1 << 23) - 1.0
        }
    }

    /// The WY kernels are MPP cooperative-tensor (live-compiled) — the
    /// MPP matmul2d body does not compile below Apple GPU Family 9, so
    /// gate exactly like the other MPP suites (CI runners skip).
    static let mppAvailable: Bool = {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple9)
    }()

    static let supportsApple10: Bool = {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple10)
    }()

    /// Runs both paths on identical inputs and compares y + stateOut.
    /// `aLog0` conditions the gate magnitude: 0.5 for moderate decay or
    /// +2.08 ≈ log(8) for realistic Qwen3.6 A_log (the underflow regime
    /// the log-space gate accumulation exists for).
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
            // Uniform[-1,1] has E[x²]=1/3, so sqrt(3/dk) amplitude gives
            // E[||k_t||²]≈1, matching what the RMSNorm pre-pass feeds the
            // kernels in production. The non-periodic source is important:
            // repeated sinusoids make each chunk's K Gram matrix nearly
            // low-rank, which is WY's documented ill-conditioned edge case
            // and not a useful wrapper-layout fixture. An
            // unnormalised amplitude (e.g. flat 0.3 at dk=128 → ||k||²≈5.8)
            // makes the plan kernel's (I+L) forward-substitution grow
            // exponentially over the C=64 chunk and overflow its f16
            // tg_solve staging — a conditioning regime real inputs
            // never enter (see the kernel's f16-staging rationale).
            let qkScale = Float((3.0 / Double(dk)).squareRoot())
            var qkRNG = RNG(state: 0x6A09_E667_F3BC_C909)
            let qNormedVals = (0 ..< b * t * hk * dk).map {
                _ in qkRNG.signedUnit() * qkScale
            }
            let kNormedVals = (0 ..< b * t * hk * dk).map {
                _ in qkRNG.signedUnit() * qkScale
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
            let (statePlanesA, planesEnabledA) = Ops.disabledGdnPlanes(dtype: .f32, device: .shared)
            runAndWait { cb in
                Ops.gatedDeltaPrepChunk(
                    convOut: convOut, aLog: aLog, dtBias: dtBias,
                    aRaw: aRaw, bRaw: bRaw,
                    qNormed: qNormed, kNormed: kNormed,
                    stateIn: stateInA, stateOut: stateOutA, y: yA,
                    tLen: tLen,
                    statePlanes: statePlanesA, planesEnabled: planesEnabledA,
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

    // Moderate-decay gates (aLog ≤ 0.5 → g ≈ 0.4): validates the wiring
    // (layouts, grids, g/beta math, v slicing) without stressing either
    // the near-one long-memory conditioning limit or gate underflow.
    // T=128 = 2 chunks of C=64, GQA 2:1,
    // at the production Dk/Dv=128 geometry. The old Dk/Dv=32 shortcut is
    // deliberately omitted: Iron's documented D32 WY-vs-monolithic bound
    // is 4e-2, so it cannot serve this wrapper's tighter parity contract.
    @Test(
        "moderate gates — T=128, Hv=4, Hk=2, Dv=128, Dk=128: WY matches prep-chunk",
        .enabled(if: mppAvailable, "MPP cooperative tensor — Apple GPU Family 9+"))
    func moderateGatesMatch() {
        let tolerance: Float = Self.supportsApple10 ? 5e-3 : 1e-2
        Self.runCase(t: 128, hv: 4, hk: 2, dv: 128, dk: 128, aLog0: 0.5, tol: tolerance)
    }

    // Realistic Qwen3.6 gate magnitude (exp(aLog) ≈ 8): per-token g ~e⁻⁴,
    // 64-token prefix product ~e⁻²⁵⁶ — f32-underflow regime. Guards the
    // plan kernel's log-space gate accumulation (naive prefix products NaN
    // here and emitted "!!!…" garbage e2e before the fix).
    @Test(
        "realistic gates — T=128, Hv=4, Hk=2, Dv=128, Dk=128: WY matches prep-chunk",
        .enabled(if: mppAvailable, "MPP cooperative tensor — Apple GPU Family 9+"))
    func realisticGatesMatch() {
        Self.runCase(t: 128, hv: 4, hk: 2, dv: 128, dk: 128, aLog0: 2.08, tol: 5e-3)
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
