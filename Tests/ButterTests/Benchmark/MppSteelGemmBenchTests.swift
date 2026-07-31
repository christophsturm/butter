// Copyright 2026 Tom Turney (@TheTom)
//
// F-85 DECISIVE PREFILL EXPERIMENT: MPP `mpp::tensor_ops::matmul2d` dense
// GEMM (`butter_dense_gemm_mpp_bf16.metal`, new for this session) vs the
// shipped steel simdgroup MMA GEMM (`iron_steel_gemm_64x64x16_2x2_bf16`,
// T26's "1.78x prefill" winner, `Sources/IronSwift/Resources/kernels/
// iron_steel_gemm_64x64x16_2x2_bf16.metal`), at real Laguna XS prefill
// projection shapes.
//
// CONTEXT (see mission brief / `benchmarks/laguna-xs/README.md`): prefill
// @512 on this M5 Max currently lands ~1060 tok/s vs a proven ~4316 tok/s
// figure reported elsewhere on the same box. Leading hypothesis: the
// faster stack routes dense GEMMs (and attention) through MPP tensor ops,
// which on M5 (Apple gen>=17, macOS 26.2+) can dispatch onto the SoC's
// neural accelerators, while butter's shipped steel kernel is classic
// simdgroup MMA (compute-unit ALUs only, no neural-accelerator routing).
// This file is the controlled kernel-level probe of that hypothesis: same
// shapes, same input data, same command-buffer-per-dispatch measurement
// methodology as the existing NVFP4 steel-vs-mpp bench
// (`Nvfp4GatherSteelKernelBenchTests.swift`), but dense (no gather, no
// quantization) so the MPP-vs-steel delta isn't confounded by dequant
// cost on either side.
//
// Pure experiment infrastructure -- wires into neither `LagunaText` nor
// any other model forward path. `butter_dense_gemm_mpp_bf16` is a
// standalone kernel (see its own header for the full design writeup:
// tile geometry, operand dtype choice, and the open bfloat-vs-half MPP
// question flagged in T25 of the README) added solely for this bench.
//
//   swift test -c release --filter MppSteelGemmBenchTests
//
// (This session did not and could not run that command -- see the
// mission's NO swift build/test hard rule. A separate validation lane
// compiles and runs this file; see this file's own header comment /
// the PR description for the exact command and what a pass looks like.)

import Foundation
import Metal
import IronSwift
import Testing

@testable import ButterSwift

// MARK: - Dispatch helper for the new MPP kernel

/// Thin PSOCache-based dispatch for `butter_dense_gemm_mpp_bf16`, mirroring
/// `LagunaSteelGemm.callManyQKVConcurrent`'s manual-encoder pattern (no
/// `Ops.*` wrapper added -- this kernel is bench-only, not production
/// infrastructure).
enum MppDenseGemmBench {
    static let kernelName = "butter_dense_gemm_mpp_bf16"

    /// `out[M,N] = a[M,K] @ b[N,K]^T` -- `b` is `[N,K]` row-major
    /// (Butter `Linear.weight` layout, K/inDim contiguous), NOT
    /// pre-transposed. See the kernel's own header for why MPP's
    /// `tb=true` descriptor flag makes this possible without a
    /// transpose pass (unlike the steel kernel used as the comparison
    /// baseline below).
    static func run(
        a: Tensor, b: Tensor, m: Int, n: Int, k: Int,
        on cmd: MTLCommandBuffer, into out: Tensor
    ) {
        precondition(n % 64 == 0, "MppDenseGemmBench.run: n=\(n) must be a multiple of 64")
        precondition(k % 32 == 0, "MppDenseGemmBench.run: k=\(k) must be a multiple of 32")
        let pso = PSOCache.shared.pipelineState(for: kernelName)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            fatalError("MppDenseGemmBench.run: could not make compute encoder")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(a.buffer, offset: a.offset, index: 0)
        enc.setBuffer(b.buffer, offset: b.offset, index: 1)
        enc.setBuffer(out.buffer, offset: out.offset, index: 2)
        var mU = UInt32(m)
        var nU = UInt32(n)
        var kU = UInt32(k)
        enc.setBytes(&mU, length: 4, index: 3)
        enc.setBytes(&nU, length: 4, index: 4)
        enc.setBytes(&kU, length: 4, index: 5)
        let bm = 64, bn = 64
        let grid = MTLSize(width: (n + bn - 1) / bn, height: (m + bm - 1) / bm, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

@Suite("MPP tensor-ops dense GEMM vs steel simdgroup GEMM (F-85 prefill decisive experiment)", .serialized)
struct MppSteelGemmBenchTests {
    /// MPP cooperative tensors need Apple GPU Family 9+; require it
    /// explicitly rather than only relying on the live-compile PSO path
    /// to fail loudly, matching `Nvfp4MppSmokeTests`'s own gate.
    static var mppAvailable: Bool {
        guard let dev = MTLCreateSystemDefaultDevice() else { return false }
        return dev.supportsFamily(.apple9)
    }

    static func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let cmd = Device.shared.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)
    }

    // MARK: - Host data generation

    /// Generates `n` bf16-truncated values in roughly `[-scale, scale]`
    /// and returns both the raw bf16 bit pattern (for the GPU buffer) and
    /// the exact f32 value that bit pattern decodes to (for the CPU
    /// reference -- using the SAME truncated value the kernel will
    /// actually read avoids any host/device rounding mismatch polluting
    /// the correctness gate).
    static func randomBf16(count n: Int, seed: inout UInt32, scale: Float) -> (bits: [UInt16], f32: [Float]) {
        var bits = [UInt16](repeating: 0, count: n)
        var f32 = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            let u = Float(Int(seed % 2000)) / 1000.0 - 1.0  // ~[-1, 1)
            let f = u * scale
            let b = UInt16((f.bitPattern >> 16) & 0xFFFF)
            bits[i] = b
            f32[i] = Float(bitPattern: UInt32(b) << 16)
        }
        return (bits, f32)
    }

    static func makeBf16Tensor(_ bits: [UInt16], shape: [Int], device: Device) -> Tensor {
        let buf = device.makeBuffer(length: bits.count * 2)
        bits.withUnsafeBytes { _ = memcpy(buf.contents(), $0.baseAddress!, bits.count * 2) }
        return Tensor(buffer: buf, offset: 0, shape: shape, dtype: .bf16)
    }

    static func toF32(_ t: Tensor) -> [Float] {
        let n = t.elementCount
        let ptr = t.buffer.contents().advanced(by: t.offset)
            .bindMemory(to: UInt16.self, capacity: n)
        return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
    }

    /// `out[m,n] = sum_k a[m,k] * b[n,k]` -- naive triple loop, f32
    /// accumulation. Only ever called at small (correctness-test) shapes;
    /// at production GEMM sizes this would take far too long on CPU.
    static func cpuReference(a: [Float], b: [Float], m: Int, n: Int, k: Int) -> [Float] {
        var out = [Float](repeating: 0, count: m * n)
        for i in 0 ..< m {
            let aRow = i * k
            for j in 0 ..< n {
                let bRow = j * k
                var acc: Float = 0
                for kk in 0 ..< k {
                    acc += a[aRow + kk] * b[bRow + kk]
                }
                out[i * n + j] = acc
            }
        }
        return out
    }

    static func cosineSimilarity(_ x: [Float], _ y: [Float]) -> Float {
        precondition(x.count == y.count)
        var dot: Float = 0, nx: Float = 0, ny: Float = 0
        for i in 0 ..< x.count where x[i].isFinite && y[i].isFinite {
            dot += x[i] * y[i]
            nx += x[i] * x[i]
            ny += y[i] * y[i]
        }
        return dot / (sqrt(nx) * sqrt(ny) + 1e-12)
    }

    // MARK: - Correctness: MPP kernel vs CPU reference (small shapes)

    @Test(
        "MPP dense GEMM matches CPU reference, single-tile shape (m=64,n=64,k=32)",
        .enabled(if: mppAvailable))
    func mppMatchesCpuSingleTile() throws {
        let device = Device.shared
        let m = 64, n = 64, k = 32
        var seed: UInt32 = 1
        let (aBits, aF32) = Self.randomBf16(count: m * k, seed: &seed, scale: 2.0)
        let (bBits, bF32) = Self.randomBf16(count: n * k, seed: &seed, scale: 2.0)
        let a = Self.makeBf16Tensor(aBits, shape: [m, k], device: device)
        let b = Self.makeBf16Tensor(bBits, shape: [n, k], device: device)
        let out = Tensor.empty(shape: [m, n], dtype: .bf16, device: device)

        Self.runAndWait { cmd in
            MppDenseGemmBench.run(a: a, b: b, m: m, n: n, k: k, on: cmd, into: out)
        }

        let gpu = Self.toF32(out)
        let ref = Self.cpuReference(a: aF32, b: bF32, m: m, n: n, k: k)
        let nan = gpu.filter(\.isNaN).count
        let cos = Self.cosineSimilarity(gpu, ref)
        print("[mpp-dense-correctness] single-tile m=\(m) n=\(n) k=\(k) nan=\(nan) cos=\(cos)")
        print("[mpp-dense-correctness] gpu sample \(gpu.prefix(4)) ref sample \(ref.prefix(4))")
        #expect(nan == 0, "MPP dense GEMM produced \(nan) NaNs")
        #expect(cos > 0.9999, "MPP dense GEMM cos \(cos) vs CPU reference")
    }

    @Test(
        "MPP dense GEMM matches CPU reference, multi-tile + K-multi-step shape (m=192,n=128,k=96)",
        .enabled(if: mppAvailable))
    func mppMatchesCpuMultiTile() throws {
        let device = Device.shared
        let m = 192, n = 128, k = 96  // 3x2 output tiles, 3 K-steps of 32
        var seed: UInt32 = 7
        let (aBits, aF32) = Self.randomBf16(count: m * k, seed: &seed, scale: 3.0)
        let (bBits, bF32) = Self.randomBf16(count: n * k, seed: &seed, scale: 3.0)
        let a = Self.makeBf16Tensor(aBits, shape: [m, k], device: device)
        let b = Self.makeBf16Tensor(bBits, shape: [n, k], device: device)
        let out = Tensor.empty(shape: [m, n], dtype: .bf16, device: device)

        Self.runAndWait { cmd in
            MppDenseGemmBench.run(a: a, b: b, m: m, n: n, k: k, on: cmd, into: out)
        }

        let gpu = Self.toF32(out)
        let ref = Self.cpuReference(a: aF32, b: bF32, m: m, n: n, k: k)
        let nan = gpu.filter(\.isNaN).count
        let cos = Self.cosineSimilarity(gpu, ref)
        print("[mpp-dense-correctness] multi-tile m=\(m) n=\(n) k=\(k) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "MPP dense GEMM produced \(nan) NaNs")
        #expect(cos > 0.9999, "MPP dense GEMM cos \(cos) vs CPU reference")
    }

    @Test(
        "MPP dense GEMM matches CPU reference, non-64-multiple M boundary (m=100,n=128,k=64)",
        .enabled(if: mppAvailable))
    func mppMatchesCpuBoundaryM() throws {
        let device = Device.shared
        let m = 100, n = 128, k = 64  // m not a multiple of 64 -> exercises the OOB M mask
        var seed: UInt32 = 13
        let (aBits, aF32) = Self.randomBf16(count: m * k, seed: &seed, scale: 2.5)
        let (bBits, bF32) = Self.randomBf16(count: n * k, seed: &seed, scale: 2.5)
        let a = Self.makeBf16Tensor(aBits, shape: [m, k], device: device)
        let b = Self.makeBf16Tensor(bBits, shape: [n, k], device: device)
        let out = Tensor.empty(shape: [m, n], dtype: .bf16, device: device)
        // Zero the output first: the kernel only writes valid (gr<M) rows,
        // but the last threadgroup tile spans m_tile_base..m_tile_base+64,
        // and this test wants to confirm nothing spurious lands past M
        // (an uninitialized buffer could hide an OOB write as easily as
        // hide correct masking).
        memset(out.buffer.contents().advanced(by: out.offset), 0, m * n * 2)

        Self.runAndWait { cmd in
            MppDenseGemmBench.run(a: a, b: b, m: m, n: n, k: k, on: cmd, into: out)
        }

        let gpu = Self.toF32(out)
        let ref = Self.cpuReference(a: aF32, b: bF32, m: m, n: n, k: k)
        let nan = gpu.filter(\.isNaN).count
        let cos = Self.cosineSimilarity(gpu, ref)
        print("[mpp-dense-correctness] boundary-M m=\(m) n=\(n) k=\(k) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "MPP dense GEMM produced \(nan) NaNs")
        #expect(cos > 0.9999, "MPP dense GEMM cos \(cos) vs CPU reference (boundary M)")
    }

    // MARK: - Correctness: steel kernel vs the SAME CPU reference
    //
    // Sanity-checks the bench's own comparison baseline (not a new claim
    // about `iron_steel_gemm_64x64x16_2x2_bf16` -- T26 already gated it
    // end-to-end) and confirms the `transposeWeightBf16` + `callMany`
    // wiring this file uses matches the `[M,K]@[N,K]^T` semantics the MPP
    // kernel above was just checked against, so the perf comparison below
    // is apples-to-apples on the SAME mathematical GEMM, not two kernels
    // computing subtly different things that happen to run at different
    // speeds.

    @Test(
        "steel GEMM (via LagunaSteelGemm) matches the same CPU reference, single-tile shape",
        .enabled(if: mppAvailable))
    func steelMatchesCpuSingleTile() throws {
        let device = Device.shared
        let m = 64, n = 64, k = 64  // steel needs k%16==0; 64 keeps it simple
        var seed: UInt32 = 99
        let (aBits, aF32) = Self.randomBf16(count: m * k, seed: &seed, scale: 2.0)
        let (bBits, bF32) = Self.randomBf16(count: n * k, seed: &seed, scale: 2.0)
        let a = Self.makeBf16Tensor(aBits, shape: [m, k], device: device)
        let bLinear = Self.makeBf16Tensor(bBits, shape: [n, k], device: device)  // [N,K], Linear.weight layout

        try #require(LagunaSteelGemm.eligible(t: m, outDim: n, inDim: k, dtype: .bf16))

        let cmd0 = device.makeCommandBuffer()
        let bT = LagunaSteelGemm.transposeWeightBf16(bLinear, outDim: n, inDim: k, device: device)
        cmd0.commit()
        cmd0.waitUntilCompleted()

        var out: Tensor!
        Self.runAndWait { cmd in
            out = LagunaSteelGemm.callMany(input: a, weightT: bT, t: m, outDim: n, inDim: k, on: cmd, device: device)
        }

        let gpu = Self.toF32(out)
        let ref = Self.cpuReference(a: aF32, b: bF32, m: m, n: n, k: k)
        let nan = gpu.filter(\.isNaN).count
        let cos = Self.cosineSimilarity(gpu, ref)
        print("[steel-dense-correctness] single-tile m=\(m) n=\(n) k=\(k) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "steel GEMM produced \(nan) NaNs")
        #expect(cos > 0.9999, "steel GEMM cos \(cos) vs CPU reference")
    }

    // MARK: - Perf: MPP vs steel at real Laguna XS prefill projection shapes
    //
    // Laguna XS config (`Tests/ButterTests/Fixtures/laguna-xs-2.1-config.json`):
    // hidden=2048, head_dim=128, num_key_value_heads=8,
    // num_attention_heads_per_layer alternates 48/64 across layers,
    // intermediate_size=8192 (L00's dense FFN). M=512 matches the
    // T33/T34/README convention of benching prefill@512.
    //
    // 3 timed reps per shape (2 discarded warmup dispatches first),
    // median reported, GFLOP/s and a naive operand+output GB/s figure
    // computed from wall-clock. GB/s here is `(M*K + N*K + M*N)*2 bytes /
    // time` -- the minimum unique bytes touched if every element were
    // read/written exactly once, NOT real device memory traffic (both
    // kernels reuse tiles heavily via threadgroup memory /
    // cooperative-tensor registers) -- reported as a normalized
    // cross-kernel comparison point, not a bandwidth-roofline claim.

    struct GemmBenchResult {
        let label: String
        let m: Int, n: Int, k: Int
        let mppMedianMs: Double
        let steelMedianMs: Double
    }

    static func medianOf3(_ body: @escaping (MTLCommandBuffer) -> Void) -> Double {
        // 2 warmup dispatches (PSO compile / shader-cache warm, cold-start
        // noise), discarded.
        for _ in 0 ..< 2 { runAndWait(body) }
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
        let flops = 2.0 * Double(m) * Double(n) * Double(k)
        return flops / (ms / 1000.0) / 1e9
    }

    static func gbPerSec(m: Int, n: Int, k: Int, ms: Double) -> Double {
        let bytes = Double(m * k + n * k + m * n) * 2.0  // bf16 = 2 bytes/elem
        return bytes / (ms / 1000.0) / 1e9
    }

    @discardableResult
    static func runShapeBench(label: String, m: Int, n: Int, k: Int) -> GemmBenchResult {
        let device = Device.shared
        var seed: UInt32 = UInt32(label.utf8.reduce(0) { $0 &+ UInt32($1) }) &+ 1
        let (aBits, _) = Self.randomBf16(count: m * k, seed: &seed, scale: 1.5)
        let (bBits, _) = Self.randomBf16(count: n * k, seed: &seed, scale: 1.5)
        let a = Self.makeBf16Tensor(aBits, shape: [m, k], device: device)
        let bLinear = Self.makeBf16Tensor(bBits, shape: [n, k], device: device)  // [N,K]
        let outMpp = Tensor.empty(shape: [m, n], dtype: .bf16, device: device)

        // One-time transpose to [K,N] for steel, mirroring
        // `LagunaLayer.init`'s real weight-load-time call (off the hot
        // path there, and off the timed section here).
        let cmd0 = device.makeCommandBuffer()
        let bT = LagunaSteelGemm.transposeWeightBf16(bLinear, outDim: n, inDim: k, device: device)
        cmd0.commit()
        cmd0.waitUntilCompleted()

        let mppMs = Self.medianOf3 { cmd in
            MppDenseGemmBench.run(a: a, b: bLinear, m: m, n: n, k: k, on: cmd, into: outMpp)
        }
        var steelOut: Tensor!
        let steelMs = Self.medianOf3 { cmd in
            steelOut = LagunaSteelGemm.callMany(input: a, weightT: bT, t: m, outDim: n, inDim: k, on: cmd, device: device)
        }
        _ = steelOut

        let mppGf = Self.gflops(m: m, n: n, k: k, ms: mppMs)
        let steelGf = Self.gflops(m: m, n: n, k: k, ms: steelMs)
        let mppGb = Self.gbPerSec(m: m, n: n, k: k, ms: mppMs)
        let steelGb = Self.gbPerSec(m: m, n: n, k: k, ms: steelMs)
        print("[F85-mpp-vs-steel-\(label)] shape m=\(m) n=\(n) k=\(k)")
        print("[F85-mpp-vs-steel-\(label)] mpp   = \(mppMs) ms (median of 3)  \(mppGf) GFLOP/s  \(mppGb) GB/s")
        print("[F85-mpp-vs-steel-\(label)] steel = \(steelMs) ms (median of 3)  \(steelGf) GFLOP/s  \(steelGb) GB/s")
        print("[F85-mpp-vs-steel-\(label)] ratio (mpp/steel) = \(mppMs / steelMs)  (< 1.0 means MPP wins)")
        return GemmBenchResult(label: label, m: m, n: n, k: k, mppMedianMs: mppMs, steelMedianMs: steelMs)
    }

    @Test(
        "F-85 MPP vs steel, Q-proj 48-head layer (m=512,k=2048,n=6144)",
        .enabled(if: mppAvailable))
    func mppVsSteelQProj48Head() throws {
        Self.runShapeBench(label: "qproj48h", m: 512, n: 6144, k: 2048)
    }

    @Test(
        "F-85 MPP vs steel, Q-proj 64-head layer (m=512,k=2048,n=8192)",
        .enabled(if: mppAvailable))
    func mppVsSteelQProj64Head() throws {
        Self.runShapeBench(label: "qproj64h", m: 512, n: 8192, k: 2048)
    }

    @Test(
        "F-85 MPP vs steel, K/V-proj (m=512,k=2048,n=1024, num_key_value_heads=8 * head_dim=128)",
        .enabled(if: mppAvailable))
    func mppVsSteelKVProj() throws {
        Self.runShapeBench(label: "kvproj", m: 512, n: 1024, k: 2048)
    }

    @Test(
        "F-85 MPP vs steel, O-proj 48-head layer (m=512,k=6144,n=2048)",
        .enabled(if: mppAvailable))
    func mppVsSteelOProj48Head() throws {
        Self.runShapeBench(label: "oproj48h", m: 512, n: 2048, k: 6144)
    }

    @Test(
        "F-85 MPP vs steel, O-proj 64-head layer (m=512,k=8192,n=2048)",
        .enabled(if: mppAvailable))
    func mppVsSteelOProj64Head() throws {
        Self.runShapeBench(label: "oproj64h", m: 512, n: 2048, k: 8192)
    }

    @Test(
        "F-85 MPP vs steel, L00 dense FFN gate/up (m=512,k=2048,n=8192, intermediate_size)",
        .enabled(if: mppAvailable))
    func mppVsSteelL00GateUp() throws {
        Self.runShapeBench(label: "l00gateup", m: 512, n: 8192, k: 2048)
    }

    @Test(
        "F-85 MPP vs steel, L00 dense FFN down (m=512,k=8192,n=2048, intermediate_size -> hidden)",
        .enabled(if: mppAvailable))
    func mppVsSteelL00Down() throws {
        Self.runShapeBench(label: "l00down", m: 512, n: 2048, k: 8192)
    }
}
