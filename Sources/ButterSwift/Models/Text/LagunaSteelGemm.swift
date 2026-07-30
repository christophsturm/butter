// Copyright 2026 Tom Turney (@TheTom)
//
// T26 Stage 2: attention QKV/O-proj dense GEMM via the Iron-generated
// `iron_steel_gemm_64x64x16_2x2_bf16` (simdgroup_matrix MMA) kernel,
// replacing the naive scalar-FMA `iron_gemm_bf16` kernel `Ops.gemm`
// dispatches today.
//
// T26 root cause: the T25/mission-brief hypothesis ("sliding-window SDPA
// path ~27ms/layer, should be ~1-2ms") was WRONG. Per-layer phase
// timing (BUTTER_LAGUNA_ATTN_PHASE_TIMING=1, see LagunaText.swift
// forwardMulti + LagunaSdpa.prefillMmaBf16) shows the SDPA MMA kernel
// itself + all its pad/cast/transpose/compact glue sums to ~7% of the
// attention-phase wall-clock; the QKV + O projections (`qProj`/`kProj`/
// `vProj`/`oProj` .callMany → Ops.gemm) are ~62% combined. Reading
// `Ops.gemm`'s dispatch (Sources/ButterSwift/Ops/Ops.swift) shows why:
// `iron_gemm_bf16.metal` is a 32×32-output-tile kernel where each thread
// does a fully scalar per-K-element FMA loop (threadgroup-staged, but no
// simdgroup_matrix / tensor-core usage at all), while the Iron kernel
// library already ships `iron_steel_gemm_64x64x16_2x2_bf16.metal`, a
// simdgroup_matrix MMA GEMM with 2×2 8×8 fragment tiling. That steel
// kernel has ZERO callers and ZERO test coverage anywhere in this repo
// (confirmed via `grep -rln "_64x64x16_2x2" Sources/`): it compiles into
// the kernel library but nothing in ButterSwift dispatches it. This file
// wires it into Laguna's dense attention projections specifically (not
// the shared `Ops.gemm`, to keep blast radius contained), gated by an
// empirically-verified A/B layout probe (see
// Tests/ButterTests/Ops/LagunaSteelGemmProbeTests.swift): `a` is `[M,K]`
// row-major (K contiguous, matches activations as-is, no transpose
// needed), `b` is `[K,N]` row-major (N contiguous), the *transpose* of
// how Butter stores Linear weights (`[outDim, inDim]`, inDim contiguous).
// Weights are transposed ONCE at layer-build time (cheap, off the hot
// path, see `transposeWeightBf16`), not per forward call.
//
// Escape: BUTTER_LAGUNA_ATTN_STEEL_GEMM=0 (falls back to the existing
// `AnyLinear.callMany` / naive `Ops.gemm` path unconditionally).
//
// Kernel constraint (no bounds checking in `iron_steel_gemm_64x64x16_2x2_bf16`,
// unlike the naive kernel): output writes are NOT masked against M/N, so
// `t`, `outDim`, and `inDim` must all be exact multiples of 64 or the
// kernel will read/write past tensor bounds. `eligible(...)` enforces
// this; ineligible shapes (e.g. T=1 decode, non-64-aligned prefill
// chunks) fall back to the existing path automatically.
//
// See benchmarks/laguna-xs/README.md T26.

import Foundation
import Metal
import IronSwift

enum LagunaSteelGemm {
    /// Default ON (`!= "0"`): T26 gates passed: byte-identical generated
    /// text vs the naive path (Paris, fib, 512-tok filler+continuation,
    /// all --temperature 0), no decode-steady regression (~91 tok/s
    /// either way, decode is always T=1, never eligible), and a clean
    /// prefill@512 win (569 -> 1013 tok/s median of 3, <0.3% run-to-run
    /// variance). Escape: BUTTER_LAGUNA_ATTN_STEEL_GEMM=0. See T26 README.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_ATTN_STEEL_GEMM"] != "0"
    }

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        let lib = try! device.makeLibrary(source: source, options: MTLCompileOptions())
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    private static let transposeSrc = """
    #include <metal_stdlib>
    using namespace metal;
    // [outDim, inDim] row-major (inDim contiguous) -> [inDim, outDim]
    // row-major (outDim contiguous). Load-time only, never the hot path.
    kernel void laguna_transpose_bf16(
        device const bfloat *src [[buffer(0)]],
        device bfloat *dst [[buffer(1)]],
        constant uint &out_dim [[buffer(2)]],
        constant uint &in_dim [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint o = gid.x;
        uint i = gid.y;
        if (o >= out_dim || i >= in_dim) return;
        dst[i * out_dim + o] = src[o * in_dim + i];
    }
    """

    /// One-time `[outDim,inDim]` bf16 -> `[inDim,outDim]` bf16 transpose.
    /// Own command buffer, commits and waits synchronously; call only at
    /// model-load time (`LagunaLayer.init`), never per forward call.
    static func transposeWeightBf16(
        _ weight: Tensor, outDim: Int, inDim: Int, device: Device
    ) -> Tensor {
        precondition(weight.dtype == .bf16, "transposeWeightBf16: weight must be bf16")
        precondition(weight.elementCount == outDim * inDim)
        let dst = Tensor.empty(shape: [inDim, outDim], dtype: .bf16, device: device)
        let cmd = device.makeCommandBuffer()
        let p = pso("laguna_transpose_bf16", source: transposeSrc, device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return dst }
        enc.setComputePipelineState(p)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 0)
        enc.setBuffer(dst.buffer, offset: dst.offset, index: 1)
        var outDimV = UInt32(outDim)
        var inDimV = UInt32(inDim)
        enc.setBytes(&outDimV, length: 4, index: 2)
        enc.setBytes(&inDimV, length: 4, index: 3)
        enc.dispatchThreads(
            MTLSize(width: outDim, height: inDim, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 32, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return dst
    }

    /// Whether the steel MMA path is safe to use for this shape (no OOB
    /// risk) and enabled. `t`/`outDim`/`inDim` must all be exact
    /// multiples of 64; the kernel has no bounds masking.
    static func eligible(t: Int, outDim: Int, inDim: Int, dtype: DType) -> Bool {
        enabled && dtype == .bf16
            && t % 64 == 0 && outDim % 64 == 0 && inDim % 64 == 0
    }

    /// T27 Stage 2: shared helper for the `steelWeightT(_:)` pattern T26
    /// introduced inline in `LagunaLayer.init`, pulled out here so
    /// `LagunaDenseMLP` (L00's dense FFN) and `MoELayer` (router GEMM)
    /// can reuse it instead of re-deriving the same guard. Returns `nil`
    /// (falls back to the naive path automatically via `eligible(...)`)
    /// unless `linear` wraps a plain `Linear` whose weight is bf16 and
    /// matches the caller's activation `dtype`.
    static func weightT(_ linear: AnyLinear, dtype: DType, device: Device) -> Tensor? {
        guard let l = linear.inner as? Linear, l.weight.dtype == dtype, dtype == .bf16
        else { return nil }
        let outDim = l.weight.shape[0]
        let inDim = l.weight.shape[1]
        return transposeWeightBf16(l.weight, outDim: outDim, inDim: inDim, device: device)
    }

    /// `out[t,outDim] = input[t,inDim] @ weightT[inDim,outDim]` via
    /// `iron_steel_gemm_64x64x16_2x2_bf16`. Caller must have checked
    /// `eligible(...)` first; no bounds checking here.
    static func callMany(
        input: Tensor, weightT: Tensor, t: Int, outDim: Int, inDim: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let out = Tensor.empty(shape: [t, outDim], dtype: .bf16, device: device)
        let bm = 64, bn = 64
        let grid = MTLSize(
            width: (outDim + bn - 1) / bn, height: (t + bm - 1) / bm, depth: 1)
        // 2×2 simdgroup tiling = 4 simdgroups/threadgroup * 32 lanes.
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        IronKernels.iron_steel_gemm_64x64x16_2x2_bf16_threadgroups(
            a: input.buffer, aOffset: input.offset,
            b: weightT.buffer, bOffset: weightT.offset,
            out: out.buffer, outOffset: out.offset,
            m: UInt32(t), n: UInt32(outDim), k: UInt32(inDim),
            gridSize: grid, threadgroupSize: tg, on: cmd)
        return out
    }
}
