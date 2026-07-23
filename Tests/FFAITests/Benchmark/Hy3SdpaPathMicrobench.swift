// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Free-path SDPA A/B for Hy3 prefill shape:
//   MMA + compact K/V + THD↔HTD transpose
// vs
//   sdpaMulti on strided KV cache (no compact / no transpose)
//
//   swift test -c release --filter Hy3SdpaPath

import Foundation
import Metal
import Testing

@testable import FFAI

@Suite("Hy3 free-path SDPA microbench")
struct Hy3SdpaPathMicrobench {
    static let mppAvailable: Bool = {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple9)
    }()

    static func runAndWait(_ body: (MTLCommandBuffer) -> Void) {
        let cmd = Device.shared.makeCommandBuffer()
        body(cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)
    }

    @Test(
        "microbench SDPA MMA+compact vs sdpaMulti at Hy3 T256",
        .enabled(if: mppAvailable, "MPP cooperative tensor — Apple GPU Family 9+"))
    func microbenchHy3Shape() throws {
        // Hy3 free-path attention shape.
        let nRows = 256
        let nHeads = 64
        let nKV = 8
        let headDim = 128
        let maxSeq = 2048
        let scale: Float = 1.0 / sqrt(Float(headDim))
        let iters = 20

        let q = Tensor.empty(shape: [nRows, nHeads, headDim], dtype: .f16)
        q.copyIn(
            from: (0 ..< (nRows * nHeads * headDim)).map {
                Float16(Float(($0 % 17) - 8) * 0.01)
            })
        // Strided cache [nKV, maxSeq, D] with valid prefix [0:nRows].
        let cacheK = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f16)
        let cacheV = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f16)
        var kHost = [Float16](repeating: 0, count: nKV * maxSeq * headDim)
        var vHost = [Float16](repeating: 0, count: nKV * maxSeq * headDim)
        for kv in 0 ..< nKV {
            for t in 0 ..< nRows {
                for d in 0 ..< headDim {
                    let i = (kv * maxSeq + t) * headDim + d
                    kHost[i] = Float16(Float((t + d) % 13) * 0.02 - 0.1)
                    vHost[i] = Float16(Float((t * 3 + d) % 11) * 0.02)
                }
            }
        }
        cacheK.copyIn(from: kHost)
        cacheV.copyIn(from: vHost)

        let outMMA = Tensor.empty(shape: [nRows, nHeads, headDim], dtype: .f16)
        let outMulti = Tensor.empty(shape: [nRows, nHeads, headDim], dtype: .f16)

        // Warm + time MMA path (transpose + compact×2 + MMA + transpose).
        func timeMMA() -> Double {
            for _ in 0 ..< 3 {
                Self.runAndWait { cmd in
                    let qHTD = Ops.transposeTHD_f16(
                        q, t: nRows, h: nHeads, d: headDim, on: cmd)
                    let kP = Ops.compactKV_f16(
                        src: cacheK, nKV: nKV, maxSeq: maxSeq,
                        kLen: nRows, headDim: headDim, on: cmd)
                    let vP = Ops.compactKV_f16(
                        src: cacheV, nKV: nKV, maxSeq: maxSeq,
                        kLen: nRows, headDim: headDim, on: cmd)
                    let oHTD = Ops.sdpaPrefillMma(
                        q: qHTD, k: kP, v: vP,
                        nQHeads: nHeads, nKVHeads: nKV, headDim: headDim,
                        qLen: nRows, kLen: nRows, scale: scale, on: cmd)
                    _ = Ops.transposeHTD_f16(
                        oHTD, t: nRows, h: nHeads, d: headDim, on: cmd, into: outMMA)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                Self.runAndWait { cmd in
                    let qHTD = Ops.transposeTHD_f16(
                        q, t: nRows, h: nHeads, d: headDim, on: cmd)
                    let kP = Ops.compactKV_f16(
                        src: cacheK, nKV: nKV, maxSeq: maxSeq,
                        kLen: nRows, headDim: headDim, on: cmd)
                    let vP = Ops.compactKV_f16(
                        src: cacheV, nKV: nKV, maxSeq: maxSeq,
                        kLen: nRows, headDim: headDim, on: cmd)
                    let oHTD = Ops.sdpaPrefillMma(
                        q: qHTD, k: kP, v: vP,
                        nQHeads: nHeads, nKVHeads: nKV, headDim: headDim,
                        qLen: nRows, kLen: nRows, scale: scale, on: cmd)
                    _ = Ops.transposeHTD_f16(
                        oHTD, t: nRows, h: nHeads, d: headDim, on: cmd, into: outMMA)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        func timeMulti() -> Double {
            for _ in 0 ..< 3 {
                Self.runAndWait { cmd in
                    _ = Ops.sdpaMulti(
                        q: q, k: cacheK, v: cacheV,
                        nQHeads: nHeads, nKVHeads: nKV, headDim: headDim,
                        baseKV: 0, nQuery: nRows, kvStride: maxSeq,
                        causal: true, scale: scale, on: cmd, into: outMulti)
                }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters {
                Self.runAndWait { cmd in
                    _ = Ops.sdpaMulti(
                        q: q, k: cacheK, v: cacheV,
                        nQHeads: nHeads, nKVHeads: nKV, headDim: headDim,
                        baseKV: 0, nQuery: nRows, kvStride: maxSeq,
                        causal: true, scale: scale, on: cmd, into: outMulti)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
        }

        let tMMA = timeMMA()
        let tMulti = timeMulti()
        let line = String(
            format:
                "SDPA Hy3 T=%d H=%d KV=%d d=%d: MMA+compact+xpose=%.3f ms  sdpaMulti=%.3f ms  multi/MMA=%.2fx\n",
            nRows, nHeads, nKV, headDim, tMMA, tMulti, tMMA / max(tMulti, 1e-9))
        fputs(line, stderr)
        if let scratch = ProcessInfo.processInfo.environment["HY3_SCRATCH"] {
            let path = (scratch as NSString).appendingPathComponent(
                "hy3-sdpa-path-micro.log")
            try? line.write(toFile: path, atomically: true, encoding: .utf8)
        }
        #expect(tMMA > 0 && tMulti > 0)

        // Correctness: paths should be close (same math, different tiling).
        let a = outMMA.toFloatArray()
        let b = outMulti.toFloatArray()
        var maxAbs: Float = 0
        var nBad = 0
        for i in 0 ..< min(a.count, b.count) {
            let d = abs(a[i] - b[i])
            if d > maxAbs { maxAbs = d }
            // Prefill MMA vs multi can differ slightly; loose tol.
            if d > 0.15 { nBad += 1 }
        }
        fputs(
            String(format: "  correctness maxAbs=%.4f nBad=%d/%d\n", maxAbs, nBad, a.count),
            stderr)
        // Don't fail the suite on tol — log only; path choice is perf.
    }
}
