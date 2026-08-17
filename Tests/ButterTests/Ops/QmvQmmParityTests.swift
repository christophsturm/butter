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
// QMV-vs-QMM cross-parity.
//
// Every other int4 test in QuantizedOpsTests validates one kernel
// against a CPU reference with its own tolerance. That leaves a gap:
// the matrix-VECTOR kernel (`dequantGemvInt4`, taken at T=1 by the
// single-token decode path) and the matrix-MATRIX kernel
// (`dequantGemmDynamicM`, taken at T>1 by the batched verify path) can
// each sit inside their own tolerance while disagreeing with EACH OTHER
// by more than either. Nothing tests that.
//
// That disagreement is exactly what breaks MTP speculative decode. The
// loop accepts or rejects a draft by comparing the batched pass's
// argmax against the draft, and on a full accept emits the batched
// bonus token directly. If the two kernels do not agree, the emitted
// stream drifts from the single-token baseline. On the real int4
// Qwen3.6-27B checkpoint row 0 of the batched pass disagreed with the
// single-token path on 45% of full-accept cycles at gamma=3 (21% at
// gamma=1), at logit margins up to 7.0 — a confidently different token,
// not an argmax tie-break.
//
// The equivalent per-row test on a DENSE bf16 model passes
// (Qwen35BatchedPrefillTests.batchedMatchesPerTokenAllRows), and
// disabling the small-M routing does not change the divergence, so the
// quantized kernels are where this has to be settled.

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("int4 QMV vs QMM cross-parity", .serialized)
struct QmvQmmParityTests {

    static let kIn = 512
    static let nOut = 64
    static let groupSize = 64

    /// Deterministic LCG so failures reproduce exactly.
    struct RNG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state
        }
        mutating func unit() -> Float {
            Float(next() >> 40) / Float(1 << 24)
        }
    }

    static func bf16Bits(_ v: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: v.bitPattern >> 16)
    }

    static func bf16Tensor(_ values: [Float], shape: [Int]) -> Tensor {
        let t = Tensor.empty(shape: shape, dtype: .bf16)
        t.copyIn(from: values.map { bf16Bits($0) })
        return t
    }

    static func bf16Values(_ t: Tensor) -> [Float] {
        t.toArray(as: UInt16.self).map {
            Float(bitPattern: UInt32($0) << 16)
        }
    }

    /// Build one random int4 weight bank plus a random activation row.
    static func fixture(seed: UInt64 = 0x5DEE_CE66_D3A7_1B41) -> (
        weight: Tensor, scales: Tensor, biases: Tensor, x: [Float]
    ) {
        var rng = RNG(state: seed)
        let groupsPerRow = kIn / groupSize

        // Packed nibbles: 8 per u32.
        var packed: [UInt32] = []
        packed.reserveCapacity(nOut * kIn / 8)
        for _ in 0 ..< (nOut * kIn / 8) {
            var word: UInt32 = 0
            for n in 0 ..< 8 {
                word |= UInt32(rng.next() & 0xF) << UInt32(n * 4)
            }
            packed.append(word)
        }
        let weight = Tensor.empty(shape: [nOut, kIn / 8], dtype: .u32)
        weight.copyIn(from: packed)

        let scales = (0 ..< nOut * groupsPerRow).map { _ in
            (rng.unit() - 0.5) * 0.05
        }
        let biases = (0 ..< nOut * groupsPerRow).map { _ in
            (rng.unit() - 0.5) * 0.2
        }
        let x = (0 ..< kIn).map { _ in (rng.unit() - 0.5) * 2.0 }

        return (
            weight,
            bf16Tensor(scales, shape: [nOut, groupsPerRow]),
            bf16Tensor(biases, shape: [nOut, groupsPerRow]),
            x
        )
    }

    static func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let device = Device.shared
        let cmd = device.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    /// Run the same activation row through both kernels and compare.
    /// `t` replicates the row so row 0 of a T-row batch is measured
    /// against the vector path — the batched verify pass's row 0 is the
    /// row that diverges worst on the real checkpoint.
    static func gemmRow0(
        _ f: (weight: Tensor, scales: Tensor, biases: Tensor, x: [Float]),
        t: Int
    ) -> [Float] {
        let device = Device.shared
        var rows: [Float] = []
        for _ in 0 ..< t { rows.append(contentsOf: f.x) }
        let input = bf16Tensor(rows, shape: [t * kIn])
        let out = Tensor.empty(shape: [t * nOut], dtype: .bf16)
        runAndWait { cb in
            Ops.dequantGemmDynamicM(
                input: input,
                weight: f.weight, scales: f.scales, biases: f.biases,
                t: t, nOut: nOut, kIn: kIn, groupSize: groupSize,
                on: cb, device: device, into: out)
        }
        return Array(bf16Values(out).prefix(nOut))
    }

    static func gemv(
        _ f: (weight: Tensor, scales: Tensor, biases: Tensor, x: [Float])
    ) -> [Float] {
        let input = bf16Tensor(f.x, shape: [kIn])
        var out: Tensor!
        runAndWait { cb in
            out = Ops.dequantGemvInt4(
                weight: f.weight, scales: f.scales, biases: f.biases,
                input: input, groupSize: groupSize, on: cb)
        }
        return bf16Values(out)
    }

    static func report(_ a: [Float], _ b: [Float]) -> (Float, Float, Int) {
        var maxAbs: Float = 0
        var maxRel: Float = 0
        var worst = -1
        for i in 0 ..< min(a.count, b.count) {
            let d = abs(a[i] - b[i])
            if d > maxAbs {
                maxAbs = d
                worst = i
            }
            let denom = max(abs(a[i]), abs(b[i]), 1e-3)
            maxRel = max(maxRel, d / denom)
        }
        return (maxAbs, maxRel, worst)
    }

    @Test("QMV(x) matches QMM([x]) at T=1")
    func gemvMatchesGemmT1() {
        let f = Self.fixture()
        let v = Self.gemv(f)
        let m = Self.gemmRow0(f, t: 1)
        let (maxAbs, maxRel, worst) = Self.report(v, m)
        #expect(
            maxRel < 0.02,
            "QMV vs QMM(T=1) disagree: max_abs=\(maxAbs) max_rel=\(maxRel) at out[\(worst)] (qmv=\(v[max(worst, 0)]) qmm=\(m[max(worst, 0)]))")
    }

    /// The batched verify pass runs T = gamma + 1 rows. Row 0 carries
    /// the same activation as the single-token path, so it must land on
    /// the same value regardless of how many rows share the dispatch.
    @Test("QMM row 0 is independent of the batch height T")
    func gemmRow0StableAcrossT() {
        let f = Self.fixture()
        let v = Self.gemv(f)
        for t in [2, 3, 4, 8] {
            let m = Self.gemmRow0(f, t: t)
            let (maxAbs, maxRel, worst) = Self.report(v, m)
            #expect(
                maxRel < 0.02,
                "T=\(t): QMV vs QMM row 0 disagree: max_abs=\(maxAbs) max_rel=\(maxRel) at out[\(worst)]")
        }
    }

    @Test("split M5/M7 QMM matches padded path on every row")
    func splitM57MatchesPaddedAllRows() {
        let f = Self.fixture()
        let device = Device.shared
        for t in [5, 7] {
            var rows: [Float] = []
            for row in 0 ..< t {
                let scale = 1.0 + Float(row) * 0.03125
                rows.append(contentsOf: f.x.map { $0 * scale + Float(row) * 0.001 })
            }
            let input = Self.bf16Tensor(rows, shape: [t * Self.kIn])
            func run(split: Bool) -> [Float] {
                setenv("BUTTER_QMM_SPLIT_M57", split ? "1" : "0", 1)
                let out = Tensor.empty(shape: [t * Self.nOut], dtype: .bf16)
                Self.runAndWait { cb in
                    Ops.dequantGemmDynamicM(
                        input: input,
                        weight: f.weight, scales: f.scales, biases: f.biases,
                        t: t, nOut: Self.nOut, kIn: Self.kIn,
                        groupSize: Self.groupSize,
                        on: cb, device: device, into: out)
                }
                return Self.bf16Values(out)
            }
            let padded = run(split: false)
            let split = run(split: true)
            unsetenv("BUTTER_QMM_SPLIT_M57")
            #expect(split == padded, "T=\(t): exact row split differs from padded path")
        }
    }

    @Test("BM3 M9 QMM matches padded path on every row")
    func bm3M9MatchesPaddedAllRows() {
        let f = Self.fixture()
        let device = Device.shared
        let t = 9
        var rows: [Float] = []
        for row in 0 ..< t {
            let scale = 1.0 + Float(row) * 0.03125
            rows.append(contentsOf: f.x.map { $0 * scale + Float(row) * 0.001 })
        }
        let input = Self.bf16Tensor(rows, shape: [t * Self.kIn])
        func run(bm3: Bool) -> [Float] {
            setenv("BUTTER_QMM_BM3_M9", bm3 ? "1" : "0", 1)
            let out = Tensor.empty(shape: [t * Self.nOut], dtype: .bf16)
            Self.runAndWait { cb in
                Ops.dequantGemmDynamicM(
                    input: input,
                    weight: f.weight, scales: f.scales, biases: f.biases,
                    t: t, nOut: Self.nOut, kIn: Self.kIn,
                    groupSize: Self.groupSize,
                    on: cb, device: device, into: out)
            }
            return Self.bf16Values(out)
        }
        let padded = run(bm3: false)
        let bm3 = run(bm3: true)
        unsetenv("BUTTER_QMM_BM3_M9")
        let (maxAbs, maxRel, worst) = Self.report(padded, bm3)
        #expect(
            maxAbs < 0.032,
            "BM3 M9 deviation beyond ULP scale: max_abs=\(maxAbs) max_rel=\(maxRel) at out[\(worst)]")
    }

    /// `BUTTER_QMM_MMA_BM8` routes T<=16 straight through `iron_qmm_mma_bm8`
    /// (see `Ops.dequantGemmDynamicM`'s BM8 fast-path arm). Row 0 of that
    /// path must agree with the single-token QMV baseline exactly like the
    /// default small-M / padded-BM32 paths above do. `dequantGemmDynamicM`
    /// reads the env fresh on every call, so toggling it around the
    /// dispatch (rather than caching a `static let`) is sufficient — no
    /// process restart needed.
    /// Class-numerics control for the bm8 tolerance question: at T=32 the
    /// DEFAULT route dispatches the existing BM=32 `iron_qmm_mma` kernel,
    /// which stages dequantized weights through threadgroup memory in the
    /// activation dtype (bf16 here) exactly like bm8 does. If this control
    /// shows the same ~0.02+ deviation from QMV, the deviation is the MMA
    /// class's staging numerics, not a bm8 defect.
    @Test("CONTROL: QMV vs default MMA32 route row 0 at T=32")
    func gemvMatchesMma32GemmRow0() {
        let f = Self.fixture()
        let v = Self.gemv(f)
        let m = Self.gemmRow0(f, t: 32)
        let (maxAbs, maxRel, worst) = Self.report(v, m)
        // Diagnostic control for the MMA-class tolerance. The deviation is
        // HARDWARE-DEPENDENT: this route measured max_rel 0.0074 on M5
        // and 0.0226 on the CI runner's GPU — the same 1-2 bf16-ULP
        // staging rounding, landing on different elements per GPU. So
        // the control asserts the MMA-class bar (0.03 rel + absolute
        // ULP backstop), NOT the scalar kernels' 0.02: those never
        // round-trip dequantized weights through threadgroup memory in
        // the activation dtype, and this class does.
        print("[mma32-control] maxAbs=\(maxAbs) maxRel=\(maxRel) worst=\(worst) qmv=\(v[max(worst, 0)]) qmm=\(m[max(worst, 0)])")
        #expect(maxRel < 0.03, "MMA32 class control beyond MMA-class bar: max_rel=\(maxRel)")
        #expect(maxAbs < 0.032, "MMA32 class control abs deviation beyond ULP scale: max_abs=\(maxAbs)")
    }

    @Test("QMV(x) matches BM8-path QMM([x]) row 0 at T=1,2,4,8")
    func gemvMatchesBm8GemmRow0() {
        setenv("BUTTER_QMM_MMA_BM8", "1", 1)
        defer { unsetenv("BUTTER_QMM_MMA_BM8") }
        let f = Self.fixture()
        let v = Self.gemv(f)
        for t in [1, 2, 4, 8] {
            let m = Self.gemmRow0(f, t: t)
            let (maxAbs, maxRel, worst) = Self.report(v, m)
            // 0.03, not the scalar kernels' 0.02: MMA-class kernels stage
            // dequantized W through threadgroup memory in the activation
            // dtype (bf16 here), adding a rounding step the scalar
            // kernels' f32-register pipeline never takes. Measured on
            // this fixture: max_abs 0.015625 (1-2 bf16 ULP at |out|~3.0)
            // for BOTH bm8 and the BM=32 control, max_rel 0.0226 (bm8) /
            // 0.0074 (control) — same element, same magnitude class. A
            // real indexing defect produces garbage, not consistent
            // ULP-scale drift, and the T-sweep value is bit-identical
            // across T=1..8.
            #expect(
                maxRel < 0.03,
                "T=\(t): QMV vs BM8 QMM row 0 disagree: max_abs=\(maxAbs) max_rel=\(maxRel) at out[\(worst)] (qmv=\(v[max(worst, 0)]) qmm=\(m[max(worst, 0)]))"
            )
            // Absolute backstop so the rel bar can't hide large errors on
            // tiny outputs: 2 bf16 ULPs at the fixture's output scale.
            #expect(
                maxAbs < 0.032,
                "T=\(t): BM8 abs deviation beyond ULP scale: max_abs=\(maxAbs)")
        }
    }

    /// `BUTTER_QMM_BM8_SCALAR` routes T=5..8 through `iron_qmm_bm8`, the
    /// scalar (non-MMA) BM=8 extension of `iron_qmm_bm4`'s own geometry
    /// (see `Ops.dequantGemmDynamicM`'s scalar-BM8 fast-path arm, and
    /// `Ops.dispatchQmmBm8Scalar`). Unlike `iron_qmm_mma_bm8` above, this
    /// kernel never stages dequantized W through threadgroup memory in
    /// the activation dtype — same all-f32-register accumulation
    /// pipeline as bm2/bm4 — so the scalar-kernel bar (0.02 rel, same as
    /// `gemvMatchesGemmT1` / `gemmRow0StableAcrossT`) applies here, not
    /// the 0.03 MMA-class bar. `dequantGemmDynamicM` reads the env fresh
    /// on every call, so toggling it around the dispatch is sufficient.
    @Test("QMV(x) matches scalar-BM8-path QMM([x]) row 0 at T=5,6,7,8")
    func gemvMatchesBm8ScalarGemmRow0() {
        setenv("BUTTER_QMM_BM8_SCALAR", "1", 1)
        defer { unsetenv("BUTTER_QMM_BM8_SCALAR") }
        let f = Self.fixture()
        let v = Self.gemv(f)
        for t in [5, 6, 7, 8] {
            let m = Self.gemmRow0(f, t: t)
            let (maxAbs, maxRel, worst) = Self.report(v, m)
            #expect(
                maxRel < 0.02,
                "T=\(t): QMV vs scalar-BM8 QMM row 0 disagree: max_abs=\(maxAbs) max_rel=\(maxRel) at out[\(worst)] (qmv=\(v[max(worst, 0)]) qmm=\(m[max(worst, 0)]))"
            )
        }
    }

    /// The fused 4-projection fast path (`Ops.batched4QmmFast`, taken by
    /// the GDN input projections when `BUTTER_GDN_BATCHED4_MANY=1`) was
    /// the last quantized-matmul variant with NO cross-parity coverage.
    /// An exactness sweep showed the flag alone shifts a 512-token greedy
    /// stream through its prefill-path use — coherent output, so
    /// kernel-family ULP drift, but the bound was never pinned by a test.
    /// This pins it: each of the four fused banks must agree with the
    /// unfused `dequantGemmDynamicM` on identical inputs, every row,
    /// across the T range the draft/verify paths use. The fused kernel
    /// accumulates per-thread like the scalar family (no threadgroup
    /// staging of dequantized weights in activation dtype), so the
    /// scalar-kernel bar (0.02 rel) applies, not the 0.03 MMA-class bar.
    @Test("fused-4 QMM matches unfused QMM, all banks, all rows, T=1,2,4,5,8")
    func batched4FusedMatchesUnfused() {
        // Four distinct banks — same shapes, different weights, like the
        // GDN q/k/v/(z|b) projections the production call site fuses.
        let banks = [
            Self.fixture(seed: 0x5DEE_CE66_D3A7_1B41),
            Self.fixture(seed: 0xA24B_AED4_963E_E407),
            Self.fixture(seed: 0x9E37_79B9_7F4A_7C15),
            Self.fixture(seed: 0xD1B5_4A32_D192_ED03),
        ]
        let x = banks[0].x
        for t in [1, 2, 4, 5, 8] {
            var rows: [Float] = []
            for _ in 0 ..< t { rows.append(contentsOf: x) }
            let input = Self.bf16Tensor(rows, shape: [t * Self.kIn])
            let outs = (0 ..< 4).map { _ in
                Tensor.empty(shape: [t * Self.nOut], dtype: .bf16)
            }
            Self.runAndWait { cb in
                Ops.batched4QmmFast(
                    input: input, m: t,
                    wA: banks[0].weight, scalesA: banks[0].scales,
                    biasesA: banks[0].biases, outA: outs[0],
                    wB: banks[1].weight, scalesB: banks[1].scales,
                    biasesB: banks[1].biases, outB: outs[1],
                    wC: banks[2].weight, scalesC: banks[2].scales,
                    biasesC: banks[2].biases, outC: outs[2],
                    wD: banks[3].weight, scalesD: banks[3].scales,
                    biasesD: banks[3].biases, outD: outs[3],
                    groupSize: Self.groupSize, on: cb)
            }
            for (bi, bank) in banks.enumerated() {
                let device = Device.shared
                let ref = Tensor.empty(shape: [t * Self.nOut], dtype: .bf16)
                Self.runAndWait { cb in
                    Ops.dequantGemmDynamicM(
                        input: input,
                        weight: bank.weight, scales: bank.scales,
                        biases: bank.biases,
                        t: t, nOut: Self.nOut, kIn: Self.kIn,
                        groupSize: Self.groupSize,
                        on: cb, device: device, into: ref)
                }
                let a = Self.bf16Values(outs[bi])
                let b = Self.bf16Values(ref)
                let (maxAbs, maxRel, worst) = Self.report(a, b)
                #expect(
                    maxRel < 0.02,
                    "T=\(t) bank=\(bi): fused-4 vs unfused disagree: max_abs=\(maxAbs) max_rel=\(maxRel) at out[\(worst)] (fused=\(a[max(worst, 0)]) unfused=\(b[max(worst, 0)]))"
                )
            }
        }
    }
}
