// Copyright 2026 Tom Turney (@TheTom)
//
// W34 kernel-level perf probe: `butter_dense_gemm_nax2_mpp_bf16` (the
// register-direct NAX-structure dense GEMM) vs the staged
// `butter_dense_gemm_mpp_bf16` it sits ahead of, at the real Laguna XS
// prefill projection shapes. Same command-buffer-per-dispatch median-of-3
// methodology as `MppSteelGemmBenchTests` (which measured staged-MPP vs
// steel for F-85); this file measures the NEXT hop of the same ladder.
//
//   swift test -c release --filter LagunaNax2GemmBenchTests
//
// Wall-clock context for reading the ratios: end-to-end prefill@1351 on
// laguna-xs-2.1-nvfp4 measured +24% with the flag on (see the W34 ledger
// entry), because dense GEMM is only part of the prefill wall; the
// kernel-level ratios here isolate how much of the remaining GEMM time
// the structure change actually removed.

import Foundation
import Metal
import IronSwift
import Testing

@testable import ButterSwift

@Suite("LagunaNax2Gemm kernel bench (W34)", .serialized)
struct LagunaNax2GemmBenchTests {
    private static var mppAvailable: Bool { LagunaMppGemm.capable }

    static func medianOf3(_ body: @escaping (MTLCommandBuffer) -> Void) -> Double {
        func runAndWait(_ b: (MTLCommandBuffer) -> Void) {
            let cmd = Device.shared.makeCommandBuffer()
            b(cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        for _ in 0 ..< 2 { runAndWait(body) }  // warmup, discarded
        var samples: [Double] = []
        for _ in 0 ..< 3 {
            let t0 = CFAbsoluteTimeGetCurrent()
            runAndWait(body)
            samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        samples.sort()
        return samples[1]
    }

    static func gflops(m: Int, n: Int, k: Int, ms: Double) -> Double {
        2.0 * Double(m) * Double(n) * Double(k) / (ms / 1000.0) / 1e9
    }

    static func randomBf16(count n: Int, seed: inout UInt32, scale: Float) -> [UInt16] {
        var bits = [UInt16](repeating: 0, count: n)
        for i in 0 ..< n {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            let u = Float(Int(seed % 2000)) / 1000.0 - 1.0
            bits[i] = UInt16(((u * scale).bitPattern >> 16) & 0xFFFF)
        }
        return bits
    }

    static func makeBf16Tensor(_ bits: [UInt16], shape: [Int], device: Device) -> Tensor {
        let t = Tensor.empty(shape: shape, dtype: .bf16, device: device)
        t.copyIn(from: bits)
        return t
    }

    @Test(
        "NAX2 vs staged MPP at real Laguna XS shapes (m=512 and m=1408)",
        .enabled(if: mppAvailable))
    func nax2VsStaged() throws {
        let device = Device.shared
        let shapes: [(label: String, n: Int, k: Int)] = [
            ("qproj48h", 6144, 2048),
            ("qproj64h", 8192, 2048),
            ("kvproj", 1024, 2048),
            ("oproj48h", 2048, 6144),
            ("oproj64h", 2048, 8192),
            ("l00gateup", 8192, 2048),
            ("l00down", 2048, 8192),
            ("router", 256, 2048),
        ]
        // Global clock warmup: ~1s of large-GEMM work before any timed
        // sample, so the first shapes measured are not penalized by GPU
        // clock ramp (first run of this file showed the same shape
        // measuring 2.6x apart depending only on position in the run
        // order).
        do {
            var seed: UInt32 = 77
            let a = Self.makeBf16Tensor(
                Self.randomBf16(count: 1408 * 2048, seed: &seed, scale: 1.0),
                shape: [1408, 2048], device: device)
            let w = Self.makeBf16Tensor(
                Self.randomBf16(count: 8192 * 2048, seed: &seed, scale: 1.0),
                shape: [8192, 2048], device: device)
            let t0 = CFAbsoluteTimeGetCurrent()
            while CFAbsoluteTimeGetCurrent() - t0 < 1.0 {
                let cmd = Device.shared.makeCommandBuffer()
                _ = LagunaMppGemm.callMany(
                    a: a, w: w, m: 1408, n: 8192, k: 2048, on: cmd, device: device,
                    nax2Mode: .forceNax2)
                cmd.commit()
                cmd.waitUntilCompleted()
            }
        }

        // m=512 matches the F-85 bench convention; m=1408 is the padded
        // dispatch M of the W34 end-to-end 1351-token prompt.
        for m in [512, 1408] {
            for (label, n, k) in shapes {
                var seed = UInt32(label.utf8.reduce(0) { $0 &+ UInt32($1) }) &+ 3
                let aBits = Self.randomBf16(count: m * k, seed: &seed, scale: 1.5)
                let bBits = Self.randomBf16(count: n * k, seed: &seed, scale: 1.5)
                let a = Self.makeBf16Tensor(aBits, shape: [m, k], device: device)
                let w = Self.makeBf16Tensor(bBits, shape: [n, k], device: device)

                let nax2Ms = Self.medianOf3 { cmd in
                    _ = LagunaMppGemm.callMany(
                        a: a, w: w, m: m, n: n, k: k, on: cmd, device: device,
                        nax2Mode: .forceNax2)
                }
                let stagedMs = Self.medianOf3 { cmd in
                    _ = LagunaMppGemm.callMany(
                        a: a, w: w, m: m, n: n, k: k, on: cmd, device: device,
                        nax2Mode: .forceStaged)
                }

                let nax2Gf = Self.gflops(m: m, n: n, k: k, ms: nax2Ms)
                let stagedGf = Self.gflops(m: m, n: n, k: k, ms: stagedMs)
                let line = String(
                    format:
                        "[w34-nax2-vs-staged] %@ m=%d n=%d k=%d nax2=%.3fms (%.0f GF/s) staged=%.3fms (%.0f GF/s) ratio=%.3f",
                    label, m, n, k, nax2Ms, nax2Gf, stagedMs, stagedGf, nax2Ms / stagedMs)
                print(line)
            }
        }
    }
}
