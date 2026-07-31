// Copyright 2026 Tom Turney (@TheTom)
//
// T33: steel-MMA-structure NVFP4 gather GEMM. See
// Sources/IronSwift/Resources/kernels/butter_nvfp4_gather_steel_bgemm_dense_bf16.metal
// and butter_nvfp4_gather_steel_bgemm_bf16.metal for the kernel-level
// rationale (T25's "bfloat weight stage, no activation stage, wide-K
// barrier-light reduction" structural note applied to the NVFP4 gather-QMM
// GEMMs). benchmarks/laguna-xs/README.md T33 has the staged correctness/
// perf writeup.
//
// Stage 1 (this file, dense entry point): exactly one logical expert
// across the whole `[m,k]` input, validated against the plain `nvfp4Qgemv`
// dequant-GEMV oracle. Stage 2 adds the sorted-gather multi-expert path
// (`nvfp4GatherSteel`) behind `BUTTER_NVFP4_GATHER_STEEL=1`.

import Foundation
import Metal
import IronSwift

extension Ops {
    /// Whether `t`/`nOut`/`kIn` are safe for the steel-structure kernels
    /// (no bounds masking, same contract as `LagunaSteelGemm.eligible`).
    public static func nvfp4GatherSteelEligible(
        mTotal: Int, nOut: Int, kIn: Int, blockSize: Int = 16
    ) -> Bool {
        mTotal % 64 == 0 && nOut % 64 == 0 && kIn % 16 == 0 && kIn % blockSize == 0
    }

    /// Stage 1: dense single-expert NVFP4 GEMM via the steel MMA kernel.
    /// `input` `[mTotal, kIn]` bf16, `weight` `[nOut, kIn/8]` u32,
    /// `scales` `[nOut, kIn/blockSize]` u8. No expert indirection at all --
    /// callers that need per-row expert selection use `nvfp4GatherSteel`.
    public static func nvfp4GatherSteelDense(
        input: Tensor, weight: Tensor, scales: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(input.dtype == .bf16 && out.dtype == .bf16)
        precondition(weight.dtype == .u32 && scales.dtype == .u8)
        precondition(input.elementCount == mTotal * kIn)
        precondition(out.elementCount == mTotal * nOut)
        precondition(nvfp4GatherSteelEligible(mTotal: mTotal, nOut: nOut, kIn: kIn, blockSize: blockSize),
            "nvfp4GatherSteelDense: shape m=\(mTotal) n=\(nOut) k=\(kIn) block=\(blockSize) ineligible (need m,n %64==0, k%16==0, k%block==0)")
        let pso = PSOCache.shared.pipelineState(for: "butter_nvfp4_gather_steel_bgemm_dense_bf16")
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        var mU = UInt32(mTotal), nU = UInt32(nOut), kU = UInt32(kIn)
        var blockU = UInt32(blockSize), globalV = global
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(out.buffer, offset: out.offset, index: 3)
        enc.setBytes(&mU, length: 4, index: 4)
        enc.setBytes(&nU, length: 4, index: 5)
        enc.setBytes(&kU, length: 4, index: 6)
        enc.setBytes(&blockU, length: 4, index: 7)
        enc.setBytes(&globalV, length: 4, index: 8)
        let bm = 64, bn = 64
        let grid = MTLSize(width: nOut / bn, height: mTotal / bm, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// Whether `nOut`/`kIn` are safe for the stage-2 sorted-gather steel
    /// kernel. Unlike the dense stage-1 kernel, `mTotal` is bounds-checked
    /// per-row inside the kernel (ragged/partial tiles are the point of
    /// the gather path), so no `mTotal %% 64` requirement here.
    public static func nvfp4GatherSteelGatherEligible(
        nOut: Int, kIn: Int, blockSize: Int = 16
    ) -> Bool {
        nOut % 64 == 0 && kIn % 16 == 0 && kIn % blockSize == 0
    }

    /// Stage 2: sorted-gather multi-expert NVFP4 GEMM via the steel MMA
    /// kernel, with per-tile expert-run masking for ragged segments
    /// (tile-boundary straddle, zero-count experts, partial tail tiles).
    /// `input` `[mTotal, kIn]` bf16 already gathered (same convention as
    /// `nvfp4MoeGatherNax64`), `weight` `[nExperts, nOut, kIn/8]` u32,
    /// `scales` `[nExperts, nOut, kIn/blockSize]` u8, `indices[mTotal]`
    /// expert id per sorted row.
    public static func nvfp4GatherSteel(
        input: Tensor, weight: Tensor, scales: Tensor, indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(input.dtype == .bf16 && out.dtype == .bf16)
        precondition(weight.dtype == .u32 && scales.dtype == .u8)
        precondition(indices.dtype == .u32 && indices.elementCount == mTotal)
        precondition(input.elementCount == mTotal * kIn)
        precondition(out.elementCount == mTotal * nOut)
        precondition(nvfp4GatherSteelGatherEligible(nOut: nOut, kIn: kIn, blockSize: blockSize),
            "nvfp4GatherSteel: shape n=\(nOut) k=\(kIn) block=\(blockSize) ineligible (need n%64==0, k%16==0, k%block==0)")
        let pso = PSOCache.shared.pipelineState(for: "butter_nvfp4_gather_steel_bgemm_bf16")
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        var mU = UInt32(mTotal), nU = UInt32(nOut), kU = UInt32(kIn)
        var blockU = UInt32(blockSize), globalV = global
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(indices.buffer, offset: indices.offset, index: 3)
        enc.setBuffer(out.buffer, offset: out.offset, index: 4)
        enc.setBytes(&mU, length: 4, index: 5)
        enc.setBytes(&nU, length: 4, index: 6)
        enc.setBytes(&kU, length: 4, index: 7)
        enc.setBytes(&blockU, length: 4, index: 8)
        enc.setBytes(&globalV, length: 4, index: 9)
        let bm = 64, bn = 64
        let grid = MTLSize(width: nOut / bn, height: (mTotal + bm - 1) / bm, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// T35: BM16 row-tile variant of the stage-2 sorted-gather steel
    /// kernel. T34 diagnosed the remaining gap at Laguna XS's real
    /// production shape (~16 rows/expert average run) as `BM64` tile
    /// granularity forcing every 64-row tile that straddles multiple
    /// experts to pay a full K-reduction pass per straddling run. This
    /// kernel shrinks the M-tile to 16 rows (same as the shipped mpp
    /// default's granularity) so a typical tile holds close to exactly
    /// one run. Same eligibility as `nvfp4GatherSteel` (`bn` stays 64).
    public static func nvfp4GatherSteelBm16(
        input: Tensor, weight: Tensor, scales: Tensor, indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(input.dtype == .bf16 && out.dtype == .bf16)
        precondition(weight.dtype == .u32 && scales.dtype == .u8)
        precondition(indices.dtype == .u32 && indices.elementCount == mTotal)
        precondition(input.elementCount == mTotal * kIn)
        precondition(out.elementCount == mTotal * nOut)
        precondition(nvfp4GatherSteelGatherEligible(nOut: nOut, kIn: kIn, blockSize: blockSize),
            "nvfp4GatherSteelBm16: shape n=\(nOut) k=\(kIn) block=\(blockSize) ineligible (need n%64==0, k%16==0, k%block==0)")
        let pso = PSOCache.shared.pipelineState(for: "butter_nvfp4_gather_steel_bm16_bgemm_bf16")
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        var mU = UInt32(mTotal), nU = UInt32(nOut), kU = UInt32(kIn)
        var blockU = UInt32(blockSize), globalV = global
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(indices.buffer, offset: indices.offset, index: 3)
        enc.setBuffer(out.buffer, offset: out.offset, index: 4)
        enc.setBytes(&mU, length: 4, index: 5)
        enc.setBytes(&nU, length: 4, index: 6)
        enc.setBytes(&kU, length: 4, index: 7)
        enc.setBytes(&blockU, length: 4, index: 8)
        enc.setBytes(&globalV, length: 4, index: 9)
        let bm = 16, bn = 64
        let grid = MTLSize(width: nOut / bn, height: (mTotal + bm - 1) / bm, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// T35: concurrent gate+up pair on the BM16 steel kernel (mirrors
    /// `nvfp4GatherSteelPair`).
    public static func nvfp4GatherSteelBm16Pair(
        input: Tensor,
        weight0: Tensor, scales0: Tensor, out0: Tensor,
        weight1: Tensor, scales1: Tensor, out1: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer
    ) {
        precondition(input.dtype == .bf16 && out0.dtype == .bf16 && out1.dtype == .bf16)
        precondition(indices.dtype == .u32 && indices.elementCount == mTotal)
        precondition(nvfp4GatherSteelGatherEligible(nOut: nOut, kIn: kIn, blockSize: blockSize),
            "nvfp4GatherSteelBm16Pair: shape n=\(nOut) k=\(kIn) block=\(blockSize) ineligible")
        let pso = PSOCache.shared.pipelineState(for: "butter_nvfp4_gather_steel_bm16_bgemm_bf16")
        guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) else { return }
        enc.setComputePipelineState(pso)
        var mU = UInt32(mTotal), nU = UInt32(nOut), kU = UInt32(kIn)
        var blockU = UInt32(blockSize), globalV = global
        let bm = 16, bn = 64
        let grid = MTLSize(width: nOut / bn, height: (mTotal + bm - 1) / bm, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        // Gate
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight0.buffer, offset: weight0.offset, index: 1)
        enc.setBuffer(scales0.buffer, offset: scales0.offset, index: 2)
        enc.setBuffer(indices.buffer, offset: indices.offset, index: 3)
        enc.setBuffer(out0.buffer, offset: out0.offset, index: 4)
        enc.setBytes(&mU, length: 4, index: 5)
        enc.setBytes(&nU, length: 4, index: 6)
        enc.setBytes(&kU, length: 4, index: 7)
        enc.setBytes(&blockU, length: 4, index: 8)
        enc.setBytes(&globalV, length: 4, index: 9)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        // Up
        enc.setBuffer(weight1.buffer, offset: weight1.offset, index: 1)
        enc.setBuffer(scales1.buffer, offset: scales1.offset, index: 2)
        enc.setBuffer(out1.buffer, offset: out1.offset, index: 4)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// T36: serial (non-concurrent-encoder) gate+up pair on the BM16 steel
    /// kernel, as a fix candidate for the gateup MoE-phase discrepancy
    /// this session diagnosed. `nvfp4GatherSteelBm16Pair` issues gate and
    /// up as two `dispatchThreadgroups` calls on ONE `.concurrent`
    /// `MTLComputeCommandEncoder`, same pattern as the mpp default's own
    /// Pair path. A dedicated kernel-level probe
    /// (`Nvfp4GatherSteelKernelBenchTests.bm16PairVsMppPairGateupShape`)
    /// found that pattern costs steel far more than mpp at the identical
    /// shape (1.77x steel/mpp under `.concurrent` Pair dispatch, vs 1.06x
    /// for the same two kernels run as isolated serial single dispatches)
    /// -- steel's BM16 kernel launches a 128-thread (4-simdgroup)
    /// threadgroup carrying 2 per-lane `simdgroup_matrix<float,8,8>`
    /// accumulators per lane, a much heavier per-threadgroup register/
    /// occupancy footprint than mpp's 32-thread (1-simdgroup) BM16
    /// threadgroup; dispatching two such heavy kernels concurrently (the
    /// whole point of `.concurrent`, encouraging the driver to interleave
    /// them on the GPU for latency-hiding) makes them compete for the same
    /// scarce per-core resource mpp barely uses, which a single isolated
    /// dispatch never exercises. This variant dispatches gate and up as
    /// two separate serial encoders instead, trading away the
    /// latency-hiding `.concurrent` normally buys in exchange for full
    /// per-dispatch occupancy, on the theory that a heavy-footprint kernel
    /// has less to gain from interleaving and more to lose from resource
    /// contention than a light one does.
    public static func nvfp4GatherSteelBm16PairSerial(
        input: Tensor,
        weight0: Tensor, scales0: Tensor, out0: Tensor,
        weight1: Tensor, scales1: Tensor, out1: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer
    ) {
        nvfp4GatherSteelBm16(
            input: input, weight: weight0, scales: scales0, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: blockSize, global: global, on: cmd, into: out0)
        nvfp4GatherSteelBm16(
            input: input, weight: weight1, scales: scales1, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: blockSize, global: global, on: cmd, into: out1)
    }

    /// T35 follow-up: BM16 kernel with `bn` doubled to 128 (subN 16->32),
    /// spending the register headroom BM16 freed on a wider per-simdgroup
    /// N-tile per the mission brief's suggestion, to amortize the ragged-
    /// run scan's fixed per-tile cost over twice the output columns.
    /// Requires `nOut %% 128 == 0` (Laguna XS gate/up nOut=512, down
    /// nOut=2048 both qualify).
    public static func nvfp4GatherSteelBm16N128Eligible(
        nOut: Int, kIn: Int, blockSize: Int = 16
    ) -> Bool {
        nOut % 128 == 0 && kIn % 16 == 0 && kIn % blockSize == 0
    }

    public static func nvfp4GatherSteelBm16N128(
        input: Tensor, weight: Tensor, scales: Tensor, indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(input.dtype == .bf16 && out.dtype == .bf16)
        precondition(weight.dtype == .u32 && scales.dtype == .u8)
        precondition(indices.dtype == .u32 && indices.elementCount == mTotal)
        precondition(input.elementCount == mTotal * kIn)
        precondition(out.elementCount == mTotal * nOut)
        precondition(nvfp4GatherSteelBm16N128Eligible(nOut: nOut, kIn: kIn, blockSize: blockSize),
            "nvfp4GatherSteelBm16N128: shape n=\(nOut) k=\(kIn) block=\(blockSize) ineligible (need n%128==0, k%16==0, k%block==0)")
        let pso = PSOCache.shared.pipelineState(for: "butter_nvfp4_gather_steel_bm16_bn128_bgemm_bf16")
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        var mU = UInt32(mTotal), nU = UInt32(nOut), kU = UInt32(kIn)
        var blockU = UInt32(blockSize), globalV = global
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(indices.buffer, offset: indices.offset, index: 3)
        enc.setBuffer(out.buffer, offset: out.offset, index: 4)
        enc.setBytes(&mU, length: 4, index: 5)
        enc.setBytes(&nU, length: 4, index: 6)
        enc.setBytes(&kU, length: 4, index: 7)
        enc.setBytes(&blockU, length: 4, index: 8)
        enc.setBytes(&globalV, length: 4, index: 9)
        let bm = 16, bn = 128
        let grid = MTLSize(width: nOut / bn, height: (mTotal + bm - 1) / bm, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// T35: pick the BM16 row-tile steel kernel over BM64 when the average
    /// per-expert run length (`mTotal / nExperts`, computable host-side
    /// from the sort plan's known dimensions -- no GPU readback needed)
    /// is below a threshold. `nExperts` should come from the weight
    /// tensor's leading dimension (`weight.shape[0]`), NOT a GPU-side
    /// count of experts actually touched -- using the static upper bound
    /// slightly UNDER-estimates the true average run length whenever some
    /// experts get zero rows, which biases toward BM16 (the safer choice:
    /// BM16 stays close to parity even when a run happens to fill a whole
    /// 16-row tile, per the kernel-level bench; BM64 is the one that loses
    /// badly when tiles fragment). Threshold hardcoded from T35's own
    /// kernel-level bench data (see benchmarks/laguna-xs/README.md T35):
    /// BM16 won/near-parity at both 16 rows/expert (production shape,
    /// ~1.03-1.11x vs mpp, down from BM64's 2.6-2.92x) and 64 rows/expert
    /// (0.87x, a clean win); BM64 was at or ahead of BM16 by 512
    /// rows/expert (BM64 0.98x vs mpp, BM16 1.09x). 128 sits in the
    /// untested gap between the 64-win and 512-loss data points -- a
    /// round-number placeholder pending a probe at an intermediate run
    /// length, not a measured crossover.
    /// `BUTTER_NVFP4_GATHER_STEEL_BM=16`/`=64` forces a side for triage,
    /// `BUTTER_NVFP4_GATHER_STEEL_BM_THRESHOLD=<rows>` overrides the cutoff.
    public static func nvfp4GatherSteelPrefersBm16(mTotal: Int, nExperts: Int) -> Bool {
        let env = ProcessInfo.processInfo.environment
        if env["BUTTER_NVFP4_GATHER_STEEL_BM"] == "16" { return true }
        if env["BUTTER_NVFP4_GATHER_STEEL_BM"] == "64" { return false }
        guard nExperts > 0 else { return false }
        let avgRun = Double(mTotal) / Double(nExperts)
        let threshold = env["BUTTER_NVFP4_GATHER_STEEL_BM_THRESHOLD"].flatMap(Double.init) ?? 128.0
        return avgRun < threshold
    }

    /// Concurrent gate+up pair on the stage-2 steel kernel (mirrors
    /// `nvfp4MoeGatherNax64Pair`'s one-concurrent-encoder pattern).
    public static func nvfp4GatherSteelPair(
        input: Tensor,
        weight0: Tensor, scales0: Tensor, out0: Tensor,
        weight1: Tensor, scales1: Tensor, out1: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer
    ) {
        precondition(input.dtype == .bf16 && out0.dtype == .bf16 && out1.dtype == .bf16)
        precondition(indices.dtype == .u32 && indices.elementCount == mTotal)
        precondition(nvfp4GatherSteelGatherEligible(nOut: nOut, kIn: kIn, blockSize: blockSize),
            "nvfp4GatherSteelPair: shape n=\(nOut) k=\(kIn) block=\(blockSize) ineligible")
        let pso = PSOCache.shared.pipelineState(for: "butter_nvfp4_gather_steel_bgemm_bf16")
        guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) else { return }
        enc.setComputePipelineState(pso)
        var mU = UInt32(mTotal), nU = UInt32(nOut), kU = UInt32(kIn)
        var blockU = UInt32(blockSize), globalV = global
        let bm = 64, bn = 64
        let grid = MTLSize(width: nOut / bn, height: (mTotal + bm - 1) / bm, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        // Gate
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight0.buffer, offset: weight0.offset, index: 1)
        enc.setBuffer(scales0.buffer, offset: scales0.offset, index: 2)
        enc.setBuffer(indices.buffer, offset: indices.offset, index: 3)
        enc.setBuffer(out0.buffer, offset: out0.offset, index: 4)
        enc.setBytes(&mU, length: 4, index: 5)
        enc.setBytes(&nU, length: 4, index: 6)
        enc.setBytes(&kU, length: 4, index: 7)
        enc.setBytes(&blockU, length: 4, index: 8)
        enc.setBytes(&globalV, length: 4, index: 9)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        // Up
        enc.setBuffer(weight1.buffer, offset: weight1.offset, index: 1)
        enc.setBuffer(scales1.buffer, offset: scales1.offset, index: 2)
        enc.setBuffer(out1.buffer, offset: out1.offset, index: 4)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// Opt-in T33 steel-structure gather path. `BUTTER_NVFP4_GATHER_STEEL=1`,
    /// bf16 only, nOut%64==0, kIn%16==0, kIn%blockSize==0. Checked before
    /// NAX/NAX64/BM64 in the dispatch switch since it targets the same
    /// gather-QMM kernel-efficiency gap those five prior attempts failed on
    /// (see benchmarks/laguna-xs/README.md T14-T27, T33).
    public static func nvfp4MoeGatherPrefersSteel(
        dtype: DType, mTotal: Int, nOut: Int, kIn: Int, blockSize: Int = 16
    ) -> Bool {
        guard dtype == .bf16 else { return false }
        let env = ProcessInfo.processInfo.environment["BUTTER_NVFP4_GATHER_STEEL"]
        guard env == "1" || env?.lowercased() == "true" else { return false }
        return nvfp4GatherSteelGatherEligible(nOut: nOut, kIn: kIn, blockSize: blockSize)
    }
}
