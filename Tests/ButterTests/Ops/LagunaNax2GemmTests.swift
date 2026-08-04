// Copyright 2026 Tom Turney (@TheTom)
//
// W34 correctness gate for `butter_dense_gemm_nax2_mpp_bf16` (the
// register-direct NAX-structure dense GEMM behind BUTTER_LAGUNA_NAX2_GEMM
// -- see the kernel's header for the structure and `LagunaMppGemm`'s
// `nax2Eligible`/`dispatchNax2` for the wiring). Mirrors
// `LagunaMppGemmTests`' gates (CPU reference at the real Laguna XS bench
// shapes, magnitude ramp, decline paths) plus the NAX2-specific ones:
// equivalence against the staged MPP kernel it sits ahead of, the
// stricter N%128/K%256 alignment gate, and swizzle-remap invariance.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("LagunaNax2Gemm (W34 register-direct NAX dense GEMM)", .serialized)
struct LagunaNax2GemmTests {
    private var mppAvailable: Bool { LagunaMppGemm.capable }

    /// Every test that wants the NAX2 path must opt in explicitly (the
    /// production default is OFF); scoped via defer so a failing test
    /// cannot leak the flag into later suites (`.serialized` keeps
    /// ordering deterministic).
    private func withNax2<T>(_ body: () throws -> T) rethrows -> T {
        setenv("BUTTER_LAGUNA_NAX2_GEMM", "1", 1)
        defer { unsetenv("BUTTER_LAGUNA_NAX2_GEMM") }
        return try body()
    }

    // MARK: - Host helpers (same bf16-truncated generation scheme as
    // LagunaMppGemmTests so CPU and GPU see identical input bits).

    private func bf16Bits(_ v: Float) -> UInt16 {
        UInt16((v.bitPattern >> 16) & 0xFFFF)
    }

    private func bf16Decode(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    private func randomBf16(count n: Int, seed: inout UInt32, scale: Float) -> (bits: [UInt16], f32: [Float]) {
        var bits = [UInt16](repeating: 0, count: n)
        var f32 = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            let u = Float(Int(seed % 2000)) / 1000.0 - 1.0  // ~[-1, 1)
            let f = u * scale
            let b = bf16Bits(f)
            bits[i] = b
            f32[i] = bf16Decode(b)
        }
        return (bits, f32)
    }

    private func makeBf16Tensor(_ bits: [UInt16], shape: [Int], device: Device) -> Tensor {
        let t = Tensor.empty(shape: shape, dtype: .bf16, device: device)
        t.copyIn(from: bits)
        return t
    }

    private func cpuReference(a: [Float], b: [Float], m: Int, n: Int, k: Int) -> [Float] {
        var out = [Float](repeating: 0, count: m * n)
        for i in 0 ..< m {
            let aRow = i * k
            for j in 0 ..< n {
                let bRow = j * k
                var acc: Float = 0
                for kk in 0 ..< k { acc += a[aRow + kk] * b[bRow + kk] }
                out[i * n + j] = acc
            }
        }
        return out
    }

    private func cosineSimilarity(_ x: [Float], _ y: [Float]) -> Float {
        precondition(x.count == y.count)
        var dot: Float = 0, nx: Float = 0, ny: Float = 0
        for i in 0 ..< x.count where x[i].isFinite && y[i].isFinite {
            dot += x[i] * y[i]
            nx += x[i] * x[i]
            ny += y[i] * y[i]
        }
        return dot / (sqrt(nx) * sqrt(ny) + 1e-12)
    }

    private func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let cmd = Device.shared.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    /// The 7 real Laguna XS bench shapes plus the MoE router projection
    /// -- every one satisfies NAX2's stricter N%128==0 / K%256==0 gate,
    /// so with the flag on, all production prefill GEMM traffic routes
    /// through the new kernel.
    private static let benchShapes:
        [(label: String, n: Int, k: Int)] = [
            ("qproj48h", 6144, 2048),
            ("qproj64h", 8192, 2048),
            ("kvproj", 1024, 2048),
            ("oproj48h", 2048, 6144),
            ("oproj64h", 2048, 8192),
            ("l00gateup", 8192, 2048),
            ("l00down", 2048, 8192),
            ("router", 256, 2048),
        ]

    // MARK: - Eligibility gate

    @Test("NAX2 gate: default OFF, and stricter N/K alignment declines to the staged kernel")
    func eligibilityGate() throws {
        // Opt-out contract: =0 must decline, even for perfect shapes.
        setenv("BUTTER_LAGUNA_NAX2_GEMM", "0", 1)
        #expect(!LagunaMppGemm.nax2Eligible(m: 512, n: 8192, k: 2048), "BUTTER_LAGUNA_NAX2_GEMM=0 must disable")
        unsetenv("BUTTER_LAGUNA_NAX2_GEMM")

        try withNax2 {
            guard mppAvailable else { return }  // capability is part of the gate below
            #expect(LagunaMppGemm.nax2Eligible(m: 512, n: 8192, k: 2048))
            #expect(LagunaMppGemm.nax2Eligible(m: 64, n: 128, k: 256), "minimum aligned shape")
            #expect(!LagunaMppGemm.nax2Eligible(m: 100, n: 8192, k: 2048), "M%64 != 0 declines")
            #expect(!LagunaMppGemm.nax2Eligible(m: 512, n: 8256, k: 2048), "N%128 != 0 declines")
            #expect(!LagunaMppGemm.nax2Eligible(m: 512, n: 8192, k: 2080), "K%256 != 0 declines")
        }
    }

    // MARK: - Correctness vs CPU reference (m=64, one M-tile; same
    // rationale as LagunaMppGemmTests.correctnessAtBenchShapes for why
    // the CPU triple loop stays at m=64).

    @Test("NAX2 dense GEMM matches CPU reference at all real Laguna XS shapes (m=64)")
    func correctnessAtBenchShapes() throws {
        guard mppAvailable else { return }  // MPP incapable (e.g. CI paravirt GPU): skip, not fail
        let device = Device.shared
        let m = 64

        try withNax2 {
            for (label, n, k) in Self.benchShapes {
                try #require(
                    LagunaMppGemm.nax2Eligible(m: m, n: n, k: k),
                    "\(label): bench shape unexpectedly ineligible for NAX2")
                var seed: UInt32 = UInt32(label.utf8.reduce(0) { $0 &+ UInt32($1) }) &+ 7
                let (aBits, aF32) = randomBf16(count: m * k, seed: &seed, scale: 2.0)
                let (bBits, bF32) = randomBf16(count: n * k, seed: &seed, scale: 2.0)
                let a = makeBf16Tensor(aBits, shape: [m, k], device: device)
                let w = makeBf16Tensor(bBits, shape: [n, k], device: device)

                var out: Tensor!
                runAndWait { cmd in
                    out = LagunaMppGemm.callMany(a: a, w: w, m: m, n: n, k: k, on: cmd, device: device)
                }

                let gpu = out.toArray(as: UInt16.self).map(bf16Decode)
                let ref = cpuReference(a: aF32, b: bF32, m: m, n: n, k: k)
                let nan = gpu.filter(\.isNaN).count
                let cos = cosineSimilarity(gpu, ref)
                print("[laguna-nax2-gemm-correctness] \(label) m=\(m) n=\(n) k=\(k) nan=\(nan) cos=\(cos)")
                #expect(nan == 0, "\(label): NAX2 produced \(nan) NaNs")
                // Same bar the staged kernel's CPU-reference gate uses
                // (its header documents 0.99985+ measured, 0.999 asserted:
                // reduced-precision K-accumulation noise, with margin).
                #expect(cos > 0.999, "\(label): NAX2 cos \(cos) vs CPU reference")
            }
        }
    }

    // MARK: - Equivalence vs the staged MPP kernel (m=512, GPU-vs-GPU,
    // full production chunk size -- the exact kernel NAX2 sits ahead of,
    // so this is the direct no-regression gate for the flag flip).

    @Test("NAX2 matches the staged MPP kernel at all real Laguna XS shapes (m=512)")
    func stagedEquivalenceAtBenchShapes() throws {
        guard mppAvailable else { return }  // MPP incapable (e.g. CI paravirt GPU): skip, not fail
        let device = Device.shared
        let m = 512

        for (label, n, k) in Self.benchShapes {
            var seed: UInt32 = UInt32(label.utf8.reduce(0) { $0 &+ UInt32($1) }) &+ 1234
            let (aBits, _) = randomBf16(count: m * k, seed: &seed, scale: 1.5)
            let (bBits, _) = randomBf16(count: n * k, seed: &seed, scale: 1.5)
            let a = makeBf16Tensor(aBits, shape: [m, k], device: device)
            let w = makeBf16Tensor(bBits, shape: [n, k], device: device)

            var nax2Out: Tensor!
            withNax2 {
                runAndWait { cmd in
                    nax2Out = LagunaMppGemm.callMany(a: a, w: w, m: m, n: n, k: k, on: cmd, device: device)
                }
            }

            var stagedOut: Tensor!
            runAndWait { cmd in
                stagedOut = LagunaMppGemm.callMany(a: a, w: w, m: m, n: n, k: k, on: cmd, device: device)
            }

            let nax2Vals = nax2Out.toArray(as: UInt16.self).map(bf16Decode)
            let stagedVals = stagedOut.toArray(as: UInt16.self).map(bf16Decode)
            let nan = nax2Vals.filter(\.isNaN).count
            let cos = cosineSimilarity(nax2Vals, stagedVals)
            print("[laguna-nax2-vs-staged] \(label) m=\(m) n=\(n) k=\(k) nan=\(nan) cos=\(cos)")
            #expect(nan == 0, "\(label): NAX2 produced \(nan) NaNs")
            // Same FP-reorder-class bar `steelEquivalenceAtBenchShapes`
            // holds the staged kernel to against steel.
            #expect(cos > 0.99999, "\(label): NAX2 vs staged MPP cos \(cos)")
        }
    }

    // MARK: - Magnitude ramp. The staged kernel needs the T23 fold to
    // survive SwiGLU-amplified activations through half staging; NAX2's
    // native bfloat operands (e8 exponent, f32 accumulate) make the
    // whole half-overflow cliff class structurally impossible -- this
    // ramp (same shape/texture as the staged kernel's fold test, out to
    // 16x past the T23-observed cliff) is the evidence.

    @Test("NAX2 stays finite and accurate through activation magnitude 800 with no fold")
    func magnitudeRamp() throws {
        guard mppAvailable else { return }  // MPP incapable (e.g. CI paravirt GPU): skip, not fail
        let device = Device.shared
        let m = 64, n = 128, k = 2048

        try withNax2 {
            try #require(LagunaMppGemm.nax2Eligible(m: m, n: n, k: k))
            for magnitude: Float in [1, 10, 50, 100, 200, 400, 800] {
                var seed: UInt32 = UInt32(magnitude) &+ 5
                let (aBits, aF32) = randomBf16(count: m * k, seed: &seed, scale: magnitude)
                let (bBits, bF32) = randomBf16(count: n * k, seed: &seed, scale: 2.0)
                let a = makeBf16Tensor(aBits, shape: [m, k], device: device)
                let w = makeBf16Tensor(bBits, shape: [n, k], device: device)

                var out: Tensor!
                runAndWait { cmd in
                    out = LagunaMppGemm.callMany(a: a, w: w, m: m, n: n, k: k, on: cmd, device: device)
                }
                let gpu = out.toArray(as: UInt16.self).map(bf16Decode)
                let nan = gpu.filter(\.isNaN).count
                let inf = gpu.filter(\.isInfinite).count
                let ref = cpuReference(a: aF32, b: bF32, m: m, n: n, k: k)
                let cos = cosineSimilarity(gpu, ref)
                print("[laguna-nax2-gemm-magnitude] mag=\(magnitude) nan=\(nan) inf=\(inf) cos=\(cos)")
                #expect(nan == 0, "NAX2 produced \(nan) NaNs at activation magnitude \(magnitude)")
                #expect(inf == 0, "NAX2 produced \(inf) Infs at activation magnitude \(magnitude)")
                #expect(cos > 0.9999, "NAX2 vs CPU reference cos=\(cos) at activation magnitude \(magnitude)")
            }
        }
    }

    // MARK: - Multi-M-tile + swizzle invariance. m=192 (3 M-tiles, the
    // same multi-tile count the original raw-kernel gate used) computed
    // at swizzleLog 0 and 2 must agree bit-for-bit: the swizzle only
    // permutes which hardware threadgroup computes which tile, plus the
    // rounded-up grid exercises the kernel's tiles_m/tiles_n early-exit
    // guard.

    @Test("NAX2 swizzle remap is output-invariant (m=192, swizzleLog 0 vs 2)")
    func swizzleInvariance() throws {
        guard mppAvailable else { return }  // MPP incapable (e.g. CI paravirt GPU): skip, not fail
        let device = Device.shared
        let m = 192, n = 1024, k = 2048

        try withNax2 {
            try #require(LagunaMppGemm.nax2Eligible(m: m, n: n, k: k))
            var seed: UInt32 = 2026
            let (aBits, _) = randomBf16(count: m * k, seed: &seed, scale: 1.5)
            let (bBits, _) = randomBf16(count: n * k, seed: &seed, scale: 1.5)
            let a = makeBf16Tensor(aBits, shape: [m, k], device: device)
            let w = makeBf16Tensor(bBits, shape: [n, k], device: device)

            var out0: Tensor!
            runAndWait { cmd in
                out0 = LagunaMppGemm.callMany(a: a, w: w, m: m, n: n, k: k, on: cmd, device: device)
            }

            setenv("BUTTER_LAGUNA_NAX2_SWIZZLE", "2", 1)
            defer { unsetenv("BUTTER_LAGUNA_NAX2_SWIZZLE") }
            var out2: Tensor!
            runAndWait { cmd in
                out2 = LagunaMppGemm.callMany(a: a, w: w, m: m, n: n, k: k, on: cmd, device: device)
            }

            let bits0 = out0.toArray(as: UInt16.self)
            let bits2 = out2.toArray(as: UInt16.self)
            #expect(bits0 == bits2, "swizzle remap changed output bits (tile mapping bug)")
        }
    }
}
