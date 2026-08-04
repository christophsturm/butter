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
// triggers on this kernel's name containing `_mpp_`). The kernel itself
// requires `t % 64 == 0` at dispatch (its own M-boundary masking exists
// for direct callers / correctness tests, not because production
// dispatch relies on it) -- decode (`t == 1`) never reaches this kernel
// at all (see `tryMppGemm`'s callers, both batched-prefill-only paths).
//
// M-PADDING (benchmarks/laguna-xs/README.md section 21): real prefill
// chunk lengths are essentially never 64-aligned, so gating dispatch on
// `t % 64 == 0` directly (the original cut of this file) meant MPP
// declined for almost every real prompt, silently paying the ~1.7x-
// slower steel tax for the entire prefill over a remainder of at most
// 63 rows. `tryMppGemm` now pads the activation with zero rows up to
// the next multiple of 64 when only `t` (not the weight shape) fails
// the gate, dispatches MPP at the padded M, and slices the real rows
// back off -- see `tryMppGemm`'s header for the per-row-independence
// correctness argument and `padMEnabled`'s header for the A/B escape
// (BUTTER_LAGUNA_MPP_PAD_M=0).
//
// Escape: BUTTER_LAGUNA_MPP_GEMM=0 (falls back to steel / naive
// unconditionally, matching `LagunaSteelGemm.enabled`'s own convention).
// Default ON.

import Foundation
import Metal
import IronSwift

enum LagunaMppGemm {
    static let kernelName = "butter_dense_gemm_mpp_bf16"

    /// W34: register-direct NAX-structure dense GEMM (see
    /// `butter_dense_gemm_nax2_mpp_bf16.metal`'s header for the four
    /// structural differences vs the staged kernel above -- no
    /// threadgroup staging, deep BK=256 barrier-free K-loop, paired
    /// (16,32,16) matmul2d fragment calls, native bfloat operands).
    static let nax2KernelName = "butter_dense_gemm_nax2_mpp_bf16"

    /// Default ON (`!= "0"`): candidate for the F-85 campaign pivot.
    /// Escape: BUTTER_LAGUNA_MPP_GEMM=0.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_MPP_GEMM"] != "0"
    }

    /// W34 NAX2 try-first path. Default ON (=0 to disable, restoring
    /// the staged kernel for every dispatch): qualified via the W34/W39
    /// gates -- kernel-level 2-2.4x vs the staged kernel, wall +23.8
    /// percent prefill at 1351 tokens, cos 0.984 end-to-end logits vs
    /// staged (the same accepted noise class as the shipped
    /// staged-vs-steel delta), 6/6 correctness suite vs f32 CPU
    /// reference, native bfloat operands with strict precision.
    static var nax2Enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_NAX2_GEMM"] != "0"
    }

    /// L2-locality threadgroup-ID swizzle exponent for the NAX2 kernel
    /// (0 = identity, the measured-best default at Laguna prefill
    /// shapes; exposed for A/B on longer prompts / bigger grids).
    static var nax2SwizzleLog: UInt32 {
        UInt32(ProcessInfo.processInfo.environment["BUTTER_LAGUNA_NAX2_SWIZZLE"] ?? "0") ?? 0
    }

    /// The NAX2 kernel implements ONLY the fully aligned fast path
    /// (M%64, N%128, K%256 -- no edge masking of any kind, see its
    /// header). `m` here is the DISPATCH row count, i.e. post-pad when
    /// the pad-M path is engaged, so real prompts qualify via padding.
    /// Every Laguna XS prefill projection this file routes (Q/K/V/O,
    /// L00 gate/up/down, router) satisfies the N/K conditions; anything
    /// that does not declines to the staged kernel, not to steel.
    static func nax2Eligible(m: Int, n: Int, k: Int) -> Bool {
        nax2Enabled && capable && m % 64 == 0 && n % 128 == 0 && k % 256 == 0
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
    ///
    /// NOTE: this exact-multiple gate is what `tryMppGemm` below no
    /// longer uses directly for the M dimension -- see its header for
    /// why (`t % 64 == 0` is a hard requirement of the underlying
    /// kernel's tiling, but not of `t` itself, since padding fixes it).
    /// `eligible` is kept as-is because `BUTTER_LAGUNA_MPP_PAD_M=0`
    /// still routes `tryMppGemm` through it unchanged, for A/B against
    /// the padded path.
    static func eligible(t: Int, outDim: Int, inDim: Int, dtype: DType) -> Bool {
        shapeEligible(outDim: outDim, inDim: inDim, dtype: dtype) && t % 64 == 0
    }

    /// F-85 follow-up, section 21 of `benchmarks/laguna-xs/README.md`:
    /// `eligible(t:...)`'s `t % 64 == 0` gate means MPP only ever engaged
    /// when the whole prefill chunk happened to be 64-row-aligned --
    /// vanishingly rare for real prompts (measured: at M=2688, a 64
    /// multiple, MPP hit 1916 tok/s vs steel's 1110; at M=2690, two rows
    /// off, MPP declined outright and both paths tied at 515). Every
    /// real prompt was silently paying the ~1.7x-slower steel tax for
    /// the entire prefill because of a two-row remainder.
    ///
    /// Default ON (`!= "0"`): gates the M-padding path in `tryMppGemm`
    /// below. Escape: BUTTER_LAGUNA_MPP_PAD_M=0 reverts `tryMppGemm` to
    /// the pre-padding behavior (`eligible(t:...)` as the sole gate, no
    /// padding) for A/B against this change.
    static var padMEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_MPP_PAD_M"] != "0"
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
        // W34 try-first: route fully aligned shapes through the
        // register-direct NAX2 structure when opted in; everything else
        // (and everything when the flag is off) takes the staged kernel
        // unchanged.
        if nax2Eligible(m: m, n: n, k: k) {
            dispatchNax2(a: a, w: w, out: out, m: m, n: n, k: k, on: cmd)
            return
        }
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

    /// NAX2 dispatch: BM=64, BN=128, 8 simdgroups (256 threads) per
    /// threadgroup. Caller (`dispatch` above) has already checked
    /// `nax2Eligible`, so the grid divides exactly; the swizzle
    /// remapping pads the M dimension up when `swizzleLog > 0` (the
    /// kernel's own tiles_m/tiles_n guard drops the overshoot).
    private static func dispatchNax2(
        a: Tensor, w: Tensor, out: Tensor, m: Int, n: Int, k: Int,
        on cmd: MTLCommandBuffer
    ) {
        let pso = PSOCache.shared.pipelineState(for: nax2KernelName)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            fatalError("LagunaMppGemm.dispatchNax2: could not make compute encoder")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(a.buffer, offset: a.offset, index: 0)
        enc.setBuffer(w.buffer, offset: w.offset, index: 1)
        enc.setBuffer(out.buffer, offset: out.offset, index: 2)
        var mU = UInt32(m)
        var nU = UInt32(n)
        var kU = UInt32(k)
        var swizzle = nax2SwizzleLog
        enc.setBytes(&mU, length: 4, index: 3)
        enc.setBytes(&nU, length: 4, index: 4)
        enc.setBytes(&kU, length: 4, index: 5)
        enc.setBytes(&swizzle, length: 4, index: 6)
        let tilesM = m / 64
        let tilesN = n / 128
        let swizzleTile = 1 << Int(swizzle)
        let grid = MTLSize(
            width: tilesN * swizzleTile,
            height: (tilesM + swizzleTile - 1) / swizzleTile,
            depth: 1)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// Single call-site hook: try the MPP GEMM for `linear` against `x`;
    /// `nil` when ineligible (wrong dtype/shape, disabled, or incapable
    /// hardware) so the caller falls back to `LagunaSteelGemm` and then
    /// the naive `AnyLinear.callMany` path, in that order. This is the
    /// one hook every call site in `LagunaText.swift` / `MoELayer.swift`
    /// uses -- no per-call-site re-derivation of the eligibility/
    /// dispatch logic.
    ///
    /// M-padding (see `padMEnabled`'s header for the section-21 finding
    /// motivating this): `t % 64 == 0` is a hard requirement of
    /// `butter_dense_gemm_mpp_bf16`'s 64-row tiling, but it is a
    /// requirement on the GEMM's row COUNT at dispatch time, not on the
    /// real prefill length `t` itself. So when `shapeEligible` passes
    /// (the weight shape/dtype/capability gate -- the part padding
    /// *can't* fix) but `t` isn't 64-aligned, this pads the activation
    /// with zero rows up to the next multiple of 64, dispatches MPP at
    /// that padded M, and slices the real `t` rows back off before
    /// returning.
    ///
    /// Correctness: for a GEMM `out[m,:] = a[m,:] @ w^T`, each output row
    /// depends ONLY on its own input row `a[m,:]` -- there is no
    /// cross-row reduction. Appending padding rows after row `t-1`
    /// therefore cannot change rows `[0, t)` of the output; they come
    /// out byte-identical to what an unpadded GEMM over just the real
    /// rows would have produced (which is exactly what `steel`/naive
    /// would have computed for those rows). The padding rows' own
    /// (garbage-inconsequential, but zeroed here defensively so the
    /// kernel never reads uninitialized/NaN memory) output rows are
    /// simply discarded by the trailing `slicedRows`.
    ///
    /// Escape: BUTTER_LAGUNA_MPP_PAD_M=0 disables the pad path and
    /// reverts to the pre-padding `eligible(t:...)`-gated behavior (MPP
    /// only ever engages on already-64-aligned `t`), for A/B against
    /// this change.
    static func tryMppGemm(
        _ linear: AnyLinear, x: Tensor, t: Int, outDim: Int, inDim: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor? {
        guard let l = linear.inner as? Linear, l.weight.dtype == x.dtype else { return nil }

        guard padMEnabled else {
            guard eligible(t: t, outDim: outDim, inDim: inDim, dtype: x.dtype) else { return nil }
            return callMany(a: x, w: l.weight, m: t, n: outDim, k: inDim, on: cmd, device: device)
        }

        guard shapeEligible(outDim: outDim, inDim: inDim, dtype: x.dtype) else { return nil }

        let t64 = ((t + 63) / 64) * 64
        guard t64 != t else {
            return callMany(a: x, w: l.weight, m: t, n: outDim, k: inDim, on: cmd, device: device)
        }

        // Pad `x` ([t, inDim]) up to `xPad` ([t64, inDim]): blit-copy the
        // real `t` rows into `xPad`'s head, blit-fill the remaining
        // `t64 - t` rows to zero. One blit encoder does both, ordered
        // correctly against other GPU work already encoded on `cmd` --
        // same idiom as the existing int8 M-pad-to-32 caller in
        // `Hy3Text.swift` (`hy3Int8CallMany`, which pads to 32 the same
        // way for MPP's int8 dequant-GEMM kernel). A CPU-side `Tensor.zero()`
        // memset was deliberately NOT used here: it's only safe
        // immediately after a fresh, GPU-untouched allocation (see
        // `DecoderLayer.decodeMulti`'s header for that exact caveat) --
        // unsafe in general mid-forward-pass where `xPad` may alias a
        // reused scratch slab with GPU work still queued against it on
        // this same `cmd`.
        let xPad = Tensor.empty(shape: [t64, inDim], dtype: x.dtype, device: device)
        let rowBytes = inDim * x.dtype.byteSize
        let validBytes = t * rowBytes
        let tailBytes = (t64 - t) * rowBytes
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(
            from: x.buffer, sourceOffset: x.offset,
            to: xPad.buffer, destinationOffset: xPad.offset, size: validBytes)
        blit.fill(
            buffer: xPad.buffer,
            range: (xPad.offset + validBytes) ..< (xPad.offset + validBytes + tailBytes),
            value: 0)
        blit.endEncoding()

        // outPad is [t64, outDim]; slicedRows gives a contiguous [t,
        // outDim] view (same buffer, offset 0, row-major -- no copy) of
        // exactly the real rows, which is what every caller expects
        // back.
        let outPad = callMany(
            a: xPad, w: l.weight, m: t64, n: outDim, k: inDim, on: cmd, device: device)
        return outPad.slicedRows(start: 0, count: t)
    }
}
