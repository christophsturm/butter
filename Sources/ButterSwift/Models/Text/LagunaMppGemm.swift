// Copyright 2026 Tom Turney (@TheTom)
//
// F-85 campaign pivot: route Laguna's dense prefill BF16 projection GEMMs
// (Q/K/V/O, L00's dense FFN, the MoE router) through the MPP tensor-ops
// kernel `butter_dense_gemm_mpp_bf16` instead of (or ahead of) the steel
// simdgroup MMA path `LagunaSteelGemm` already wired in. Motivated by the
// w3/mpp-bench decisive experiment: at real Laguna XS prefill projection
// shapes (m=512), the MPP kernel measured 2.2-3.3x faster than steel
// (mpp/steel ratio 0.30-0.45), with one flagged outlier (qproj48h,
// n=6144, ratio 0.88 -- still a win, just a much smaller one; worth a
// follow-up look at whether that specific N needs different N-tiling,
// not investigated here).
//
// SEMANTICS: `butter_dense_gemm_mpp_bf16` computes out[M,N] = X[M,K] @
// W^T directly against Butter's native `Linear.weight` layout ([N,K]
// row-major, K/inDim contiguous, `tb=true` in the kernel's
// `matmul2d_descriptor`) -- unlike the steel kernel, which needs the
// weight pre-transposed to [K,N] once at load time
// (`LagunaSteelGemm.transposeWeightBf16`). That means a plain-bf16
// `Linear` whose shape statically satisfies the MPP kernel's tile
// contract (`outDim % 64 == 0`, `inDim % 32 == 0`) never needs a steel
// `weightT` copy built at all: every real (`t % 64 == 0`) forward call
// for that projection will route through MPP, so the steel fallback can
// never fire. `staticEligible(_:dtype:)` below is what callers use to
// skip that transpose+copy at model-load time -- see the call sites in
// `LagunaText.swift` / `MoELayer.swift` (`LagunaLayer.init`,
// `LagunaDenseMLP.init`, `MoELayer.init`) for the "only build steel's
// weightT if MPP declines" wiring.
//
// ELIGIBILITY: mirrors `LagunaSteelGemm.eligible`'s shape gate but
// looser on K (MPP requires only `K % 32 == 0`, not `K % 64 == 0` --
// the kernel's own BK=32 staging tile) and adds a device-capability +
// macOS-version gate (`Ops.mppAutoCapable`: Apple GPU Family 9+ / macOS
// 26+, the same check every other MPP `matmul2d` kernel in this repo
// already requires before it's safe to live-compile
// `MetalPerformancePrimitives` against the running OS's header -- see
// `PSOCache.liveCompileMppFunction` / `PSOCache.isMppKernel`, which
// triggers on this kernel's name containing `_mpp_`). `t % 64 == 0` is
// enforced same as steel: decode (`t == 1`) and any non-64-aligned
// prefill remainder chunk fall back to steel or the naive
// `AnyLinear.callMany` path automatically, never this kernel -- the
// kernel's own M-boundary masking (see its header) exists for direct
// callers / correctness tests, not because production dispatch relies
// on it.
//
// Escape: BUTTER_LAGUNA_MPP_GEMM=0 (falls back to steel / naive
// unconditionally, matching `LagunaSteelGemm.enabled`'s own convention).
// Default ON.

import Foundation
import Metal
import IronSwift

enum LagunaMppGemm {
    static let kernelName = "butter_dense_gemm_mpp_bf16"

    /// Default ON (`!= "0"`): candidate for the F-85 campaign pivot.
    /// Escape: BUTTER_LAGUNA_MPP_GEMM=0.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_MPP_GEMM"] != "0"
    }

    /// Apple GPU Family 9+ / macOS 26+ -- the same gate every other MPP
    /// `matmul2d` kernel in this repo requires (see `Ops.mppAutoCapable`'s
    /// own header). Reused rather than re-derived: one source of truth
    /// for "is this hardware/OS combo safe to live-compile MPP kernels
    /// on" across the whole repo.
    static var capable: Bool { Ops.mppAutoCapable }

    /// Whether `outDim`/`inDim`/`dtype` alone (independent of any
    /// particular call's row count `t`, and independent of `enabled`/
    /// `capable`) satisfy the kernel's tile contract: `outDim % 64 == 0`,
    /// `inDim % 32 == 0`, `dtype == .bf16`. Exposed separately from
    /// `shapeEligible` so `staticEligible` below can compose it with a
    /// weight-derived shape lookup without duplicating the arithmetic.
    private static func tileContract(outDim: Int, inDim: Int, dtype: DType) -> Bool {
        dtype == .bf16 && outDim % 64 == 0 && inDim % 32 == 0
    }

    /// Full shape+capability eligibility, independent of `t`. `true` here
    /// means: on this device, with MPP enabled, every `t % 64 == 0`
    /// forward call against a weight of this shape will route through
    /// MPP -- used both by `eligible(t:...)` below and by
    /// `staticEligible(_:dtype:)` to decide whether a steel `weightT`
    /// copy is ever going to be needed.
    static func shapeEligible(outDim: Int, inDim: Int, dtype: DType) -> Bool {
        enabled && capable && tileContract(outDim: outDim, inDim: inDim, dtype: dtype)
    }

    /// Whether `linear` is a plain bf16 `Linear` whose weight shape
    /// statically satisfies `shapeEligible` -- i.e. whether it is safe
    /// to skip building `LagunaSteelGemm.weightT(linear:...)` for this
    /// projection at model-load time. `false` for anything quantized,
    /// wrong-dtype, MPP-disabled, or MPP-incapable, so the steel
    /// fallback's weightT still gets built in every case where it might
    /// actually be needed.
    static func staticEligible(_ linear: AnyLinear, dtype: DType) -> Bool {
        guard let l = linear.inner as? Linear, l.weight.dtype == dtype, dtype == .bf16
        else { return false }
        let outDim = l.weight.shape[0]
        let inDim = l.weight.shape[1]
        return shapeEligible(outDim: outDim, inDim: inDim, dtype: dtype)
    }

    /// Whether the MPP path is safe AND worthwhile to dispatch for this
    /// exact call: `shapeEligible` plus `t % 64 == 0` (decode T=1 and
    /// non-64-aligned prefill remainder chunks decline, same convention
    /// `LagunaSteelGemm.eligible` uses for its own M dimension).
    static func eligible(t: Int, outDim: Int, inDim: Int, dtype: DType) -> Bool {
        shapeEligible(outDim: outDim, inDim: inDim, dtype: dtype) && t % 64 == 0
    }

    /// `out[m,n] = a[m,k] @ w[n,k]^T` via `butter_dense_gemm_mpp_bf16`.
    /// `w` is the RAW (untransposed) `Linear.weight`, `[n,k]` row-major --
    /// no `transposeWeightBf16` pass needed, unlike
    /// `LagunaSteelGemm.callMany`. Caller must have already checked
    /// `eligible(...)`; no bounds checking here beyond the kernel's own
    /// N/K preconditions.
    static func callMany(
        a: Tensor, w: Tensor, m: Int, n: Int, k: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(n % 64 == 0, "LagunaMppGemm.callMany: n=\(n) must be a multiple of 64")
        precondition(k % 32 == 0, "LagunaMppGemm.callMany: k=\(k) must be a multiple of 32")
        let out = Tensor.empty(shape: [m, n], dtype: .bf16, device: device)
        dispatch(a: a, w: w, out: out, m: m, n: n, k: k, on: cmd)
        return out
    }

    private static func dispatch(
        a: Tensor, w: Tensor, out: Tensor, m: Int, n: Int, k: Int,
        on cmd: MTLCommandBuffer
    ) {
        let pso = PSOCache.shared.pipelineState(for: kernelName)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            fatalError("LagunaMppGemm.dispatch: could not make compute encoder")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(a.buffer, offset: a.offset, index: 0)
        enc.setBuffer(w.buffer, offset: w.offset, index: 1)
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

    /// Single call-site hook: try the MPP GEMM for `linear` against `x`;
    /// `nil` when ineligible (wrong dtype/shape, disabled, incapable
    /// hardware, or `t` not a multiple of 64) so the caller falls back to
    /// `LagunaSteelGemm` and then the naive `AnyLinear.callMany` path, in
    /// that order. This is the one hook every call site in
    /// `LagunaText.swift` / `MoELayer.swift` uses -- no per-call-site
    /// re-derivation of the eligibility/dispatch logic.
    static func tryMppGemm(
        _ linear: AnyLinear, x: Tensor, t: Int, outDim: Int, inDim: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor? {
        guard let l = linear.inner as? Linear, l.weight.dtype == x.dtype,
            eligible(t: t, outDim: outDim, inDim: inDim, dtype: x.dtype)
        else { return nil }
        return callMany(a: x, w: l.weight, m: t, n: outDim, k: inDim, on: cmd, device: device)
    }
}
