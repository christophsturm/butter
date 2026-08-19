// Copyright 2026 Tom Turney (@TheTom)
//
// GPU-only head-to-head anchor: measure NAX2 (`butter_dense_gemm_nax2_mpp_bf16`)
// and the staged predecessor with the SAME GPU-only clock the iron `#[bench]`
// harness uses (`GPUEndTime - GPUStartTime`), so its numbers are directly
// comparable to iron's recorded staged dense-GEMM kernel (`iron_gemm_nax_dense_bm64`) GB/s.
//
// WHY THIS FILE EXISTS. `LagunaNax2GemmBenchTests` times WALL clock
// (`CFAbsoluteTimeGetCurrent` around commit+wait), which includes per-dispatch
// command-buffer/encode/commit overhead. iron's bench (`runner/gpu.rs::measure`)
// times ONLY the GPU execution window of the command buffer. A wall number and
// a GPU-only number for the SAME kernel differ by ~2x for a single ~0.5ms
// dispatch — so comparing the staged kernel's iron GB/s (GPU-only) to NAX2's wall GB/s is
// invalid. This file removes that mismatch: NAX2 measured GPU-only, same byte
// model iron uses (`(M*K + N*K + M*N) * dtypeBytes`), median of a warmed run.
//
//   swift test -c release --filter LagunaNax2GpuOnlyBenchTests

import Foundation
import Metal
import IronSwift
import Testing

@testable import ButterSwift

@Suite("LagunaNax2 GPU-only bench (head-to-head anchor)", .serialized)
struct LagunaNax2GpuOnlyBenchTests {
    private static var mppAvailable: Bool { LagunaMppGemm.capable }

    // GPU-only median, mirroring iron runner/gpu.rs: warmup passes discarded,
    // then `iters` timed passes reading (GPUEndTime - GPUStartTime).
    static func gpuOnlyMedianMs(warmup: Int, iters: Int, _ body: (MTLCommandBuffer) -> Void) -> Double {
        func runOnce() -> Double {
            let cmd = Device.shared.makeCommandBuffer()
            body(cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            return (cmd.gpuEndTime - cmd.gpuStartTime) * 1000.0  // ms
        }
        for _ in 0 ..< warmup { _ = runOnce() }
        var s: [Double] = []
        for _ in 0 ..< iters { s.append(runOnce()) }
        s.sort()
        return s[s.count / 2]
    }

    static func gflops(m: Int, n: Int, k: Int, ms: Double) -> Double {
        2.0 * Double(m) * Double(n) * Double(k) / (ms / 1000.0) / 1e9
    }

    // Iron's byte model: (M*K + N*K + M*N) * dtypeBytes. bf16 == 2 bytes,
    // matching the f16 iron staged-kernel bench (both 2-byte operands & output).
    static func gbps(m: Int, n: Int, k: Int, ms: Double, bytesPerElem: Int = 2) -> Double {
        let bytes = Double((m * k + n * k + m * n) * bytesPerElem)
        return bytes / (ms / 1000.0) / 1e9
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
        "NAX2 & staged GPU-only GB/s at Laguna dense-proj shapes (m=512)",
        .enabled(if: mppAvailable))
    func nax2GpuOnly() throws {
        let device = Device.shared
        // The three dense projection shapes from the head-to-head anchor.
        let shapes: [(label: String, n: Int, k: Int)] = [
            ("qkv", 6144, 2048),
            ("o", 2048, 6144),
            ("gate_up", 8192, 2048),
        ]
        let m = 512

        // Global clock warmup (~1s of large GEMM) so early shapes are not
        // penalized by GPU clock ramp — same guard the wall-clock file uses.
        do {
            var seed: UInt32 = 91
            let a = Self.makeBf16Tensor(
                Self.randomBf16(count: 1408 * 2048, seed: &seed, scale: 1.0),
                shape: [1408, 2048], device: device)
            let w = Self.makeBf16Tensor(
                Self.randomBf16(count: 8192 * 2048, seed: &seed, scale: 1.0),
                shape: [8192, 2048], device: device)
            let t0 = CFAbsoluteTimeGetCurrent()
            while CFAbsoluteTimeGetCurrent() - t0 < 1.0 {
                let cmd = device.makeCommandBuffer()
                _ = LagunaMppGemm.callMany(
                    a: a, w: w, m: 1408, n: 8192, k: 2048, on: cmd, device: device,
                    nax2Mode: .forceNax2)
                cmd.commit(); cmd.waitUntilCompleted()
            }
        }

        for (label, n, k) in shapes {
            var seed = UInt32(label.utf8.reduce(0) { $0 &+ UInt32($1) }) &+ 5
            let a = Self.makeBf16Tensor(
                Self.randomBf16(count: m * k, seed: &seed, scale: 1.5), shape: [m, k], device: device)
            let w = Self.makeBf16Tensor(
                Self.randomBf16(count: n * k, seed: &seed, scale: 1.5), shape: [n, k], device: device)

            let nax2Ms = Self.gpuOnlyMedianMs(warmup: 15, iters: 15) { cmd in
                _ = LagunaMppGemm.callMany(
                    a: a, w: w, m: m, n: n, k: k, on: cmd, device: device,
                    nax2Mode: .forceNax2)
            }
            let stagedMs = Self.gpuOnlyMedianMs(warmup: 15, iters: 15) { cmd in
                _ = LagunaMppGemm.callMany(
                    a: a, w: w, m: m, n: n, k: k, on: cmd, device: device,
                    nax2Mode: .forceStaged)
            }

            let line = String(
                format:
                    "[w13-gpuonly] %@ m=%d n=%d k=%d  NAX2 %.3fms %.1f GB/s (%.0f GF/s)  |  staged %.3fms %.1f GB/s (%.0f GF/s)",
                label, m, n, k,
                nax2Ms, Self.gbps(m: m, n: n, k: k, ms: nax2Ms), Self.gflops(m: m, n: n, k: k, ms: nax2Ms),
                stagedMs, Self.gbps(m: m, n: n, k: k, ms: stagedMs), Self.gflops(m: m, n: n, k: k, ms: stagedMs))
            print(line)
        }
    }
}
