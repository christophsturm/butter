// Copyright 2026 Tom Turney (@TheTom)
//
// T26 scratch probe: `iron_steel_gemm_64x64x16_2x2_bf16` (Iron-generated,
// simdgroup_matrix MMA GEMM) exists in the kernel library but has ZERO
// callers and ZERO test coverage anywhere in this repo, confirmed via
// `grep -rln "_64x64x16_2x2" Sources/ButterSwift Sources/IronSwift` before
// writing this. Before wiring it into Laguna's hot QKV/O-proj path (T26
// Stage 2 candidate: the naive `iron_gemm_bf16` scalar-FMA kernel that
// `Ops.gemm` actually dispatches was found to dominate attention wall-clock,
// not SDPA), pin down its A/B layout convention empirically against a CPU
// reference. Delete or keep as regression coverage once LagunaSteelGemm
// lands; either way this is throwaway-safe (isolated file, no production
// code touches it).
import Foundation
import Metal
import IronSwift
import TestHelpers
import Testing

@testable import ButterSwift

private func bf16Bits(_ v: Float) -> UInt16 {
    let bits = v.bitPattern
    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
    return UInt16(rounded >> 16)
}

@Suite("LagunaSteelGemmProbe")
struct LagunaSteelGemmProbeTests {
    @Test("iron_steel_gemm_64x64x16_2x2_bf16: a=[M,K] row-major, b=[K,N] row-major, out=[M,N]")
    func steelGemmSemantics() {
        autoreleasepool {
            let device = Device.shared
            let M = 64, K = 64, N = 64
            let aData = (0 ..< M * K).map { Float(($0 % 13)) * 0.05 - 0.3 }
            let bData = (0 ..< K * N).map { Float(($0 % 17)) * 0.04 - 0.25 }

            let aT = Tensor.empty(shape: [M, K], dtype: .bf16, device: device)
            let bT = Tensor.empty(shape: [K, N], dtype: .bf16, device: device)
            aT.copyIn(from: aData.map(bf16Bits))
            bT.copyIn(from: bData.map(bf16Bits))
            let outT = Tensor.empty(shape: [M, N], dtype: .bf16, device: device)

            let cmd = device.makeCommandBuffer()
            let bm = 64, bn = 64
            let grid = MTLSize(width: (N + bn - 1) / bn, height: (M + bm - 1) / bm, depth: 1)
            let tg = MTLSize(width: 128, height: 1, depth: 1)  // 4 simdgroups (2x2) * 32 lanes
            IronKernels.iron_steel_gemm_64x64x16_2x2_bf16_threadgroups(
                a: aT.buffer, aOffset: aT.offset,
                b: bT.buffer, bOffset: bT.offset,
                out: outT.buffer, outOffset: outT.offset,
                m: UInt32(M), n: UInt32(N), k: UInt32(K),
                gridSize: grid, threadgroupSize: tg, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            // CPU reference assuming a=[M,K] row-major, b=[K,N] row-major:
            // out[m,n] = sum_k a[m,k]*b[k,n]. bf16 round-trip both operands
            // through bf16Bits first so the reference sees the same
            // truncated inputs the kernel actually consumed.
            let aBf = aData.map { Float(bitPattern: UInt32(bf16Bits($0)) << 16) }
            let bBf = bData.map { Float(bitPattern: UInt32(bf16Bits($0)) << 16) }
            var expected = [Float](repeating: 0, count: M * N)
            for m in 0 ..< M {
                for n in 0 ..< N {
                    var acc: Float = 0
                    for k in 0 ..< K { acc += aBf[m * K + k] * bBf[k * N + n] }
                    expected[m * N + n] = acc
                }
            }
            let got = outT.toFloatArray()
            var maxAbsErr: Float = 0
            var dot: Float = 0
            var normG: Float = 0
            var normE: Float = 0
            for i in 0 ..< got.count {
                maxAbsErr = max(maxAbsErr, abs(got[i] - expected[i]))
                dot += got[i] * expected[i]
                normG += got[i] * got[i]
                normE += expected[i] * expected[i]
            }
            let cos = dot / (normG.squareRoot() * normE.squareRoot() + 1e-9)
            #expect(cos > 0.999, "cos=\(cos) maxAbsErr=\(maxAbsErr)")
        }
    }
}
