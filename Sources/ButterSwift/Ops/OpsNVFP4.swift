// Copyright 2026 Tom Turney (@TheTom)
//
// NVFP4 (E2M1 + E4M3 micro-scale × global) Ops for Laguna XS board chase.
// Kernels live in kernels.metallib as iron_nvfp4_*.

import Foundation
import Metal
import IronSwift

extension Ops {
    /// NVFP4 dequant GEMV: `y = W_nvfp4 · x`.
    /// `weight` u32 packed `[out, in/8]`, `scales` u8 `[out, in/blockSize]`.
    /// `blockSize` is 16 for Poolside Laguna; `global` multiplies E4M3 scales
    /// (MLX often uses 1.0 when scales already absorb the factor).
    public static func nvfp4Qgemv(
        weight: Tensor, scales: Tensor, input: Tensor,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.dtype == .u32, "nvfp4Qgemv: weight must be u32")
        precondition(scales.dtype == .u8, "nvfp4Qgemv: scales must be u8")
        precondition(weight.shape.count == 2, "nvfp4Qgemv: weight must be 2D")
        let outDim = weight.shape[0]
        let packed = weight.shape[1]
        let inDim = packed * 8
        precondition(input.elementCount == inDim, "nvfp4Qgemv: input \(input.elementCount) ≠ inDim \(inDim)")
        precondition(
            scales.shape.count == 2 && scales.shape[0] == outDim
                && scales.shape[1] == inDim / blockSize,
            "nvfp4Qgemv: scales shape \(scales.shape) expected [\(outDim), \(inDim / blockSize)]")
        let result = out ?? Tensor.empty(shape: [outDim], dtype: input.dtype)
        // Reduction: one TG per out row, TPG=64 (matches iron tests).
        let tpg = 64
        let grid = MTLSize(width: outDim * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch input.dtype {
        case .f32:
            IronKernels.iron_nvfp4_qgemv_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim), block_size: UInt32(blockSize),
                global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_nvfp4_qgemv_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim), block_size: UInt32(blockSize),
                global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_nvfp4_qgemv_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim), block_size: UInt32(blockSize),
                global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("nvfp4Qgemv: unsupported dtype \(input.dtype)")
        }
        return result
    }

    /// Expert-indexed NVFP4 GEMV batch (gate/up/down slots). Concurrent
    /// encoder by default (distinct outs).
    public static func nvfp4DequantGemvExpertIndexedMany(
        weightsStacked: [Tensor], scalesStacked: [Tensor],
        inputs: [Tensor], expertIndices: [Tensor], outputs: [Tensor],
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer
    ) {
        let n = weightsStacked.count
        precondition(n > 0 && scalesStacked.count == n && inputs.count == n
            && expertIndices.count == n && outputs.count == n)
        let dtype = inputs[0].dtype
        let outDim = weightsStacked[0].shape[1]
        let packed = weightsStacked[0].shape[2]
        let inDim = packed * 8
        let tpg = 64
        // Threadgroups form: grid = outDim TGs × TPG threads (iron tests).
        let gridTG = MTLSize(width: outDim, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        let useConcurrent =
            ButterEnv["BUTTER_MOE_SERIAL_EXPERT_GEMV"] == nil
        let enc: MTLComputeCommandEncoder?
        if useConcurrent {
            enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        } else {
            enc = cmd.makeComputeCommandEncoder()
        }
        guard let enc else { return }
        let psoName: String
        switch dtype {
        case .f32: psoName = "iron_nvfp4_dequant_gemv_expert_indexed_f32"
        case .f16: psoName = "iron_nvfp4_dequant_gemv_expert_indexed_f16"
        case .bf16: psoName = "iron_nvfp4_dequant_gemv_expert_indexed_bf16"
        default: fatalError("nvfp4DequantGemvExpertIndexedMany: bad dtype \(dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        enc.setComputePipelineState(pso)
        var inDimV = UInt32(inDim)
        var outDimV = UInt32(outDim)
        var blockV = UInt32(blockSize)
        var globalV = global
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&outDimV, length: 4, index: 6)
        enc.setBytes(&blockV, length: 4, index: 7)
        enc.setBytes(&globalV, length: 4, index: 8)
        for i in 0 ..< n {
            enc.setBuffer(weightsStacked[i].buffer, offset: weightsStacked[i].offset, index: 0)
            enc.setBuffer(scalesStacked[i].buffer, offset: scalesStacked[i].offset, index: 1)
            enc.setBuffer(inputs[i].buffer, offset: inputs[i].offset, index: 2)
            enc.setBuffer(expertIndices[i].buffer, offset: expertIndices[i].offset, index: 3)
            enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 4)
            enc.dispatchThreadgroups(gridTG, threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
    }

    /// Prefer BM64 MPP when shapes allow and env opts in.
    ///
    /// Iron contract (`moe_mpp_bm64_block_scaled.rs`):
    ///   Mode Reduction; grid `[n_out/64, ceil(m_total/64), 1]`; TPG `[128,1,1]`
    ///   (4 SGs, 2×2); `k_in % 32 == 0`, `n_out % 64 == 0`.
    ///
    /// Tile pick for NVFP4 MoE gather:
    /// - **Default all-BM16** (gate/up mid=512 and down n=2048). Measured
    ///   T=512 prefill on M5 Max: hybrid BM64-for-wide-N was **slower** than
    ///   all-BM16 (~466 → ~491 t/s when hybrid off; MoE wall −5–6%).
    /// - Opt-in hybrid BM64 when `nOut ≥ 1024`: `BUTTER_NVFP4_HYBRID_BM=1`
    /// - Force all-BM64: `BUTTER_NVFP4_MOE_BM64=1` / `BUTTER_NVFP4_GATHER_BM=64`
    /// - Force BM16 everywhere: `BUTTER_NVFP4_GATHER_BM=16` (or leave defaults)
    /// - GATHER_NAX / GATHER_BK64 also want 64-tile geometry (skip fused BM16 gateup).
    public static func nvfp4MoeGatherPrefersBm64(mTotal: Int, nOut: Int, kIn: Int) -> Bool {
        guard mTotal >= 64, nOut % 64 == 0, kIn % 32 == 0 else { return false }
        if ButterEnv["BUTTER_NVFP4_GATHER_BM"] == "16" { return false }
        // NAX-class path also needs fused BM16 skipped (64-tile outs).
        if ButterEnv["BUTTER_NVFP4_GATHER_NAX"] == "1"
            || ButterEnv["BUTTER_NVFP4_GATHER_NAX"]?.lowercased() == "true"
        {
            return true
        }
        // NAX64 (T22 true-BK64 stage) also needs the 64-tile materialise path.
        if ButterEnv["BUTTER_NVFP4_GATHER_NAX64"] == "1"
            || ButterEnv["BUTTER_NVFP4_GATHER_NAX64"]?.lowercased() == "true"
        {
            return kIn % 64 == 0
        }
        // Board-class BK=64 stage (BM=BN=64) — opt-in, needs k_in % 64.
        if ButterEnv["BUTTER_NVFP4_GATHER_BK64"] == "1"
            || ButterEnv["BUTTER_NVFP4_GATHER_BK64"]?.lowercased() == "true"
        {
            return kIn % 64 == 0
        }
        if ButterEnv["BUTTER_NVFP4_MOE_BM64"] == "1" { return true }
        if ButterEnv["BUTTER_NVFP4_GATHER_BM"] == "64" { return true }
        // Hybrid BM64 for wide-N: OPT-IN only. Default all-BM16 (see header).
        let hybridOn = ButterEnv["BUTTER_NVFP4_HYBRID_BM"] == "1"
            || ButterEnv["BUTTER_NVFP4_HYBRID_BM"]?.lowercased() == "true"
        if hybridOn, nOut >= 1024 { return true }
        return false
    }

    /// Prefer iron `bm64_bk64` (stage BK=64, dual SK=32 MMA) when env ON and shapes allow.
    public static func nvfp4MoeGatherPrefersBm64Bk64(
        mTotal: Int, nOut: Int, kIn: Int
    ) -> Bool {
        guard mTotal >= 64, nOut % 64 == 0, kIn % 64 == 0 else { return false }
        return ButterEnv["BUTTER_NVFP4_GATHER_BK64"] == "1"
            || ButterEnv["BUTTER_NVFP4_GATHER_BK64"]?.lowercased() == "true"
    }

    /// Prefill MoE gather BGEMM NVFP4.
    /// Default BM16; opt-in BM64 via `BUTTER_NVFP4_MOE_BM64=1` or
    /// `BUTTER_NVFP4_GATHER_BM=64` when shapes allow.
    /// `indices` expert ids per gathered row `[mTotal] u32`.
    /// `weight` `[nExperts, nOut, kIn/8] u32`, `scales` `[nExperts, nOut, kIn/16] u8`.
    public static func nvfp4MoeGatherBm16(
        input: Tensor, weight: Tensor, scales: Tensor, indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        preferNativeBf16: Bool = false,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(weight.dtype == .u32 && scales.dtype == .u8)
        precondition(indices.dtype == .u32 && indices.elementCount == mTotal)
        precondition(input.elementCount == mTotal * kIn)
        precondition(out.elementCount == mTotal * nOut)
        // MPP coop-tensor kernels MUST use dispatchThreadgroups (grid in
        // TG units). Plain dispatchThreads silently corrupts matmul2d
        // (see iron-codegen emit.rs + int4 BM64 fix). That was the root
        // cause of all-NaN prefill logits on NVFP4 gather.
        // Geometry (iron contracts):
        //   BM16: grid [nOut/32, ceil(mTotal/16), 1], TPG=32 (1 SG)
        //   BM64: grid [nOut/64, ceil(mTotal/64), 1], TPG=128 (4 SGs 2×2)
        //   NAX:  same 64-tile grid via live butter_nvfp4_gather_qmm_rhs_nax_mpp_*
        //         when BUTTER_NVFP4_GATHER_NAX=1 (default OFF).
        //   EG-MPP: fused-dequant MPP tensor-ops gather-GEMM with true
        //         1-expert/threadgroup ownership (device-side tile plan,
        //         no straddling, no indirect dispatch) via live
        //         butter_nvfp4_moe_gather_eg_mpp_bf16, default ON
        //         (escape BUTTER_NVFP4_MPP_GATHER=0). See
        //         `LagunaNvfp4MppGather.swift` for the full design.
        // BUTTER_LAGUNA_GATHER_V2 (default ON, escape =0):
        // expert-aligned A-device-direct iron kernel, checked ahead of
        // EG-MPP. See `LagunaNvfp4ExpertGather.swift`.
        if LagunaNvfp4ExpertGather.eligible(dtype: input.dtype, nOut: nOut, kIn: kIn, blockSize: blockSize) {
            LagunaNvfp4ExpertGather.callGather(
                input: input, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global,
                on: cmd, device: Device.shared, into: out)
            return
        }
        if LagunaNvfp4MppGather.eligible(dtype: input.dtype, nOut: nOut, kIn: kIn, blockSize: blockSize) {
            LagunaNvfp4MppGather.callGather(
                input: input, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global,
                on: cmd, device: Device.shared, into: out)
            return
        }
        if nvfp4MoeGatherPrefersSteel(dtype: input.dtype, mTotal: mTotal, nOut: nOut, kIn: kIn, blockSize: blockSize) {
            // T35: BM16 row-tile variant below a run-length threshold
            // (fixes T34's diagnosed tile-fragmentation loss at production
            // scale), BM64 otherwise (wins outright on long runs).
            let nExperts = weight.shape[0]
            if nvfp4GatherSteelPrefersBm16(mTotal: mTotal, nExperts: nExperts) {
                nvfp4GatherSteelBm16(
                    input: input, weight: weight, scales: scales, indices: indices,
                    mTotal: mTotal, nOut: nOut, kIn: kIn,
                    blockSize: blockSize, global: global, on: cmd, into: out)
            } else {
                nvfp4GatherSteel(
                    input: input, weight: weight, scales: scales, indices: indices,
                    mTotal: mTotal, nOut: nOut, kIn: kIn,
                    blockSize: blockSize, global: global, on: cmd, into: out)
            }
            return
        }
        if nvfp4MoeGatherPrefersNax64(mTotal: mTotal, nOut: nOut, kIn: kIn) {
            nvfp4MoeGatherNax64(
                input: input, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global,
                preferNativeBf16: preferNativeBf16,
                on: cmd, into: out)
            return
        }
        if nvfp4MoeGatherPrefersNax(mTotal: mTotal, nOut: nOut, kIn: kIn) {
            nvfp4MoeGatherNax(
                input: input, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global,
                preferNativeBf16: preferNativeBf16,
                on: cmd, into: out)
            return
        }
        // Every path above is either shape/env-gated (opt-in — the user who
        // sets BUTTER_NVFP4_GATHER_NAX/NAX64 owns the hardware requirement,
        // same convention as `moeGatherDequantGemmInt4`'s force-on) or
        // already checks `Ops.mppAutoCapable` itself (`LagunaNvfp4ExpertGather`
        // / `LagunaNvfp4MppGather`'s `.eligible`). What's left below is the
        // DEFAULT dispatch — reached with no opt-in at all — and it
        // unconditionally needs a `matmul2d` cooperative-tensor PSO
        // (`iron_nvfp4_moe_gather_qmm_bm{16,64}[_bk64]_mpp_*`), which
        // PSOCache.pipelineState fatalErrors compiling on Apple GPU Family
        // <9 / macOS <26 ("unsupported deferred-static-alloca-size function
        // body in metal::cooperative_tensor" — see butter PR #132's first CI
        // run). There is no batched non-MPP kernel for this gather-GEMM
        // family at arbitrary (mTotal, expert-run) shapes, so on incapable
        // hardware degrade to the per-row expert-indexed GEMV loop instead
        // of reaching that fatalError — the same primitive `MoELayer`'s own
        // non-MPP path and `NemotronHNvfp4MoELayer.decodeMulti`'s per-row
        // `decode()` fallback already use for exactly this situation.
        guard Ops.mppAutoCapable else {
            nvfp4MoeGatherBm16NonMppFallback(
                input: input, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global,
                on: cmd, into: out)
            return
        }
        let useBm64 = nvfp4MoeGatherPrefersBm64(mTotal: mTotal, nOut: nOut, kIn: kIn)
        let mTotalU = UInt32(mTotal)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let blockU = UInt32(blockSize)
        // bf16 MPP uses half coop tiles; late Laguna layers grow residual
        // |h| past ~400 and half A*B overflows → Inf/NaN (eidx float GEMV
        // stays finite at |h|~1e6). Default: promote bf16 through f32 MPP.
        // Escape: BUTTER_NVFP4_MPP_BF16=1 or preferNativeBf16=true (early layers).
        let forceBf16Mpp =
            preferNativeBf16
            || ButterEnv["BUTTER_NVFP4_MPP_BF16"] == "1"
        if input.dtype == .bf16 && !forceBf16Mpp {
            // Reuse persistent slabs — Tensor.empty every call was ~3×
            // (gate/up/down) × 39 layers of fresh Metal buffer allocs per
            // prefill and shredded the driver pool under swap pressure.
            // Distinct x/out tags per (nOut,kIn) pair so concurrent gate+up
            // (and recursive down) never alias the same scratch mid-cmdbuf.
            let xTag = "nvfp4_mpp_xf_\(kIn)x\(nOut)"
            let oTag = "nvfp4_mpp_outf_\(kIn)x\(nOut)"
            let xBytes = mTotal * kIn * 4
            let oBytes = mTotal * nOut * 4
            let xBuf = Device.shared.intermediateScratch(tag: xTag, minBytes: xBytes)
            let oBuf = Device.shared.intermediateScratch(tag: oTag, minBytes: oBytes)
            let xF = Tensor(buffer: xBuf, offset: 0, shape: input.shape, dtype: .f32)
            let outF = Tensor(buffer: oBuf, offset: 0, shape: out.shape, dtype: .f32)
            Ops.castToF32(input, into: xF, on: cmd)
            nvfp4MoeGatherBm16(
                input: xF, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global,
                on: cmd, into: outF)
            castF32ToBf16(outF, into: out, on: cmd)
            return
        }
        if useBm64 {
            // BM=BN=64, 4 SGs → TPG=128. Grid [N/64, ceil(M/64), 1] TGs.
            // BK64 stage (dual SK=32 MMA) when BUTTER_NVFP4_GATHER_BK64=1.
            // Else classic iron `moe_mpp_bm64_block_scaled` BK=32.
            precondition(nOut % 64 == 0, "nvfp4 gather BM64: nOut \(nOut) % 64 != 0")
            precondition(kIn % 32 == 0, "nvfp4 gather BM64: kIn \(kIn) % 32 != 0")
            let grid = MTLSize(
                width: nOut / 64,
                height: (mTotal + 63) / 64,
                depth: 1)
            let tg = MTLSize(width: 128, height: 1, depth: 1)
            let useBk64 = nvfp4MoeGatherPrefersBm64Bk64(
                mTotal: mTotal, nOut: nOut, kIn: kIn)
            if useBk64 {
                // Live-compile MPP metal (not yet in Generated IronKernels).
                let psoName: String
                switch input.dtype {
                case .f32: psoName = "iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp_f32"
                case .f16: psoName = "iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp_f16"
                case .bf16: psoName = "iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp_bf16"
                default: fatalError("nvfp4 gather BK64: bad dtype \(input.dtype)")
                }
                let pso = PSOCache.shared.pipelineState(for: psoName)
                guard let enc = cmd.makeComputeCommandEncoder() else { return }
                enc.setComputePipelineState(pso)
                var m = mTotalU, n = nOutU, k = kInU, b = blockU, g = global
                enc.setBuffer(input.buffer, offset: input.offset, index: 0)
                enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
                enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
                enc.setBuffer(indices.buffer, offset: indices.offset, index: 3)
                enc.setBuffer(out.buffer, offset: out.offset, index: 4)
                enc.setBytes(&m, length: 4, index: 5)
                enc.setBytes(&n, length: 4, index: 6)
                enc.setBytes(&k, length: 4, index: 7)
                enc.setBytes(&b, length: 4, index: 8)
                enc.setBytes(&g, length: 4, index: 9)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                enc.endEncoding()
            } else {
            switch input.dtype {
            case .f32:
                IronKernels.iron_nvfp4_moe_gather_qmm_bm64_mpp_f32_threadgroups(
                    x: input.buffer, xOffset: input.offset,
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    indices: indices.buffer, indicesOffset: indices.offset,
                    out: out.buffer, outOffset: out.offset,
                    m_total: mTotalU, n_out: nOutU, k_in: kInU,
                    block_size: blockU, global: global,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            case .f16:
                IronKernels.iron_nvfp4_moe_gather_qmm_bm64_mpp_f16_threadgroups(
                    x: input.buffer, xOffset: input.offset,
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    indices: indices.buffer, indicesOffset: indices.offset,
                    out: out.buffer, outOffset: out.offset,
                    m_total: mTotalU, n_out: nOutU, k_in: kInU,
                    block_size: blockU, global: global,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            case .bf16:
                IronKernels.iron_nvfp4_moe_gather_qmm_bm64_mpp_bf16_threadgroups(
                    x: input.buffer, xOffset: input.offset,
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    indices: indices.buffer, indicesOffset: indices.offset,
                    out: out.buffer, outOffset: out.offset,
                    m_total: mTotalU, n_out: nOutU, k_in: kInU,
                    block_size: blockU, global: global,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            default:
                fatalError("nvfp4MoeGatherBm16: bad dtype \(input.dtype)")
            }
            }
        } else {
            // BM=16, BN=32, 1 SG → TPG=32. Grid [N/32, ceil(M/16), 1] TGs.
            // Default BK=32 (`iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_*`) when
            // kIn%32==0 — half the K-loop dequant rounds vs classic BK=16.
            // Escape classic BK16: BUTTER_NVFP4_GATHER_BK=16.
            precondition(nOut % 32 == 0, "nvfp4 gather BM16: nOut \(nOut) % 32 != 0")
            let grid = MTLSize(
                width: nOut / 32,
                height: (mTotal + 15) / 16,
                depth: 1)
            let tg = MTLSize(width: 32, height: 1, depth: 1)
            let forceBk16 =
                ButterEnv["BUTTER_NVFP4_GATHER_BK"] == "16"
            let useBk32 = !forceBk16 && kIn % 32 == 0
            if useBk32 {
                switch input.dtype {
                case .f32:
                    IronKernels.iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_f32_threadgroups(
                        x: input.buffer, xOffset: input.offset,
                        w: weight.buffer, wOffset: weight.offset,
                        scales: scales.buffer, scalesOffset: scales.offset,
                        indices: indices.buffer, indicesOffset: indices.offset,
                        out: out.buffer, outOffset: out.offset,
                        m_total: mTotalU, n_out: nOutU, k_in: kInU,
                        block_size: blockU, global: global,
                        gridSize: grid, threadgroupSize: tg, on: cmd)
                case .f16:
                    IronKernels.iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_f16_threadgroups(
                        x: input.buffer, xOffset: input.offset,
                        w: weight.buffer, wOffset: weight.offset,
                        scales: scales.buffer, scalesOffset: scales.offset,
                        indices: indices.buffer, indicesOffset: indices.offset,
                        out: out.buffer, outOffset: out.offset,
                        m_total: mTotalU, n_out: nOutU, k_in: kInU,
                        block_size: blockU, global: global,
                        gridSize: grid, threadgroupSize: tg, on: cmd)
                case .bf16:
                    IronKernels.iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_bf16_threadgroups(
                        x: input.buffer, xOffset: input.offset,
                        w: weight.buffer, wOffset: weight.offset,
                        scales: scales.buffer, scalesOffset: scales.offset,
                        indices: indices.buffer, indicesOffset: indices.offset,
                        out: out.buffer, outOffset: out.offset,
                        m_total: mTotalU, n_out: nOutU, k_in: kInU,
                        block_size: blockU, global: global,
                        gridSize: grid, threadgroupSize: tg, on: cmd)
                default:
                    fatalError("nvfp4MoeGatherBm16: bad dtype \(input.dtype)")
                }
            } else {
                switch input.dtype {
                case .f32:
                    IronKernels.iron_nvfp4_moe_gather_qmm_bm16_mpp_f32_threadgroups(
                        x: input.buffer, xOffset: input.offset,
                        w: weight.buffer, wOffset: weight.offset,
                        scales: scales.buffer, scalesOffset: scales.offset,
                        indices: indices.buffer, indicesOffset: indices.offset,
                        out: out.buffer, outOffset: out.offset,
                        m_total: mTotalU, n_out: nOutU, k_in: kInU,
                        block_size: blockU, global: global,
                        gridSize: grid, threadgroupSize: tg, on: cmd)
                case .f16:
                    IronKernels.iron_nvfp4_moe_gather_qmm_bm16_mpp_f16_threadgroups(
                        x: input.buffer, xOffset: input.offset,
                        w: weight.buffer, wOffset: weight.offset,
                        scales: scales.buffer, scalesOffset: scales.offset,
                        indices: indices.buffer, indicesOffset: indices.offset,
                        out: out.buffer, outOffset: out.offset,
                        m_total: mTotalU, n_out: nOutU, k_in: kInU,
                        block_size: blockU, global: global,
                        gridSize: grid, threadgroupSize: tg, on: cmd)
                case .bf16:
                    IronKernels.iron_nvfp4_moe_gather_qmm_bm16_mpp_bf16_threadgroups(
                        x: input.buffer, xOffset: input.offset,
                        w: weight.buffer, wOffset: weight.offset,
                        scales: scales.buffer, scalesOffset: scales.offset,
                        indices: indices.buffer, indicesOffset: indices.offset,
                        out: out.buffer, outOffset: out.offset,
                        m_total: mTotalU, n_out: nOutU, k_in: kInU,
                        block_size: blockU, global: global,
                        gridSize: grid, threadgroupSize: tg, on: cmd)
                default:
                    fatalError("nvfp4MoeGatherBm16: bad dtype \(input.dtype)")
                }
            }
        }
    }

    /// Non-MPP correctness fallback for `nvfp4MoeGatherBm16` (and its
    /// `Pair`/`XRows`/`XRowsPair` siblings, which route here too) on
    /// hardware/OS that cannot compile the `matmul2d` cooperative-tensor
    /// kernels this gather-GEMM family needs (`Ops.mppAutoCapable ==
    /// false`). There is no batched non-MPP kernel for arbitrary
    /// (mTotal, expert-run) shapes in this family, so this degrades to a
    /// per-row dispatch of the already-shipped `nvfp4DequantGemvExpertIndexedMany`
    /// GEMV primitive (the same kernel `MoELayer`'s own `useGatherQmm ==
    /// false` path already loops per-row for its non-MPP case, and the
    /// device-side analogue of `NemotronHNvfp4MoELayer.decodeMulti`'s
    /// per-row `decode()` fallback) — `mTotal` dispatches on one
    /// concurrent encoder, correctness first, no host readback. Logged
    /// once so a full prefill doesn't spam stderr.
    nonisolated(unsafe) private static var nonMppFallbackLogged = false
    private static func nvfp4MoeGatherBm16NonMppFallback(
        input: Tensor, weight: Tensor, scales: Tensor, indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int, global: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            input.dtype == .f32 || input.dtype == .f16 || input.dtype == .bf16,
            "nvfp4MoeGatherBm16NonMppFallback: unsupported dtype \(input.dtype)")
        if !nonMppFallbackLogged {
            nonMppFallbackLogged = true
            FileHandle.standardError.write(
                ("[nvfp4MoeGatherBm16] Ops.mppAutoCapable == false — no batched "
                    + "non-MPP kernel exists for this gather-GEMM family; "
                    + "degrading to a per-row expert-indexed GEMV loop "
                    + "(mTotal dispatches/call, correctness over speed). The "
                    + "fast MPP path needs Apple GPU Family 9+ / macOS 26+.\n")
                    .data(using: .utf8)!)
        }
        let dtBytes = input.dtype.byteSize
        var weights: [Tensor] = []
        var scaleList: [Tensor] = []
        var inputs: [Tensor] = []
        var expertIdx: [Tensor] = []
        var outputs: [Tensor] = []
        weights.reserveCapacity(mTotal)
        scaleList.reserveCapacity(mTotal)
        inputs.reserveCapacity(mTotal)
        expertIdx.reserveCapacity(mTotal)
        outputs.reserveCapacity(mTotal)
        for r in 0 ..< mTotal {
            inputs.append(
                Tensor(
                    buffer: input.buffer, offset: input.offset + r * kIn * dtBytes,
                    shape: [kIn], dtype: input.dtype))
            expertIdx.append(
                Tensor(
                    buffer: indices.buffer, offset: indices.offset + r * 4,
                    shape: [1], dtype: .u32))
            outputs.append(
                Tensor(
                    buffer: out.buffer, offset: out.offset + r * nOut * dtBytes,
                    shape: [nOut], dtype: out.dtype))
            weights.append(weight)
            scaleList.append(scales)
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: weights, scalesStacked: scaleList,
            inputs: inputs, expertIndices: expertIdx, outputs: outputs,
            blockSize: blockSize, global: global, on: cmd)
    }

    /// Gate+up BM16 MPP with fused token gather via `xRows`.
    ///
    /// Default: concurrent pair of **iron-bitexact** `xrows` kernels (same math
    /// as `iron_nvfp4_moe_gather_qmm_bm16_mpp_*` + `x_rows` indirection). Skips
    /// materialising `[mTotal, kIn]` gather — `input` is unpermuted `[T, kIn]`.
    ///
    /// Experimental single-kernel fuse (shared X load): `BUTTER_NVFP4_MPP_FUSED_GATEUP=1`
    /// → hand-written `butter_nvfp4_moe_gateup_*` (quality not default-safe yet).
    public static func nvfp4MoeGatherBm16GateUp(
        input: Tensor,
        xRows: Tensor?,
        weightGate: Tensor, scalesGate: Tensor, outGate: Tensor,
        weightUp: Tensor, scalesUp: Tensor, outUp: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer
    ) {
        precondition(input.dtype == outGate.dtype && input.dtype == outUp.dtype)
        precondition(nOut % 32 == 0)
        precondition(indices.dtype == .u32 && indices.elementCount == mTotal)
        precondition(outGate.elementCount == mTotal * nOut)
        precondition(outUp.elementCount == mTotal * nOut)
        let rows: Tensor
        if let xRows {
            precondition(xRows.dtype == .u32 && xRows.elementCount == mTotal)
            precondition(input.elementCount % kIn == 0)
            rows = xRows
        } else {
            precondition(input.elementCount == mTotal * kIn)
            let idBuf = Ops.identityRowsBuffer(minCount: mTotal, device: Device.shared)
            rows = Tensor(buffer: idBuf, offset: 0, shape: [mTotal], dtype: .u32)
        }
        let wantFusedKernel =
            ButterEnv["BUTTER_NVFP4_MPP_FUSED_GATEUP"] == "1"
        // `butter_nvfp4_moe_gateup_bm16_mpp_*` is a `matmul2d` kernel —
        // unlike BUTTER_NVFP4_GATHER_NAX/STEEL (features that are
        // explicitly and only about MPP), BUTTER_NVFP4_MPP_FUSED_GATEUP
        // is a fusion toggle a caller could set without realizing it
        // forces an MPP-only kernel, so still gate on `Ops.mppAutoCapable`
        // — when incapable, fall through to the materialise+Pair path
        // below (self-guarded via `nvfp4MoeGatherBm16Pair`).
        if wantFusedKernel, Ops.mppAutoCapable, input.dtype == .f32 || input.dtype == .bf16 {
            let psoName =
                input.dtype == .f32
                ? "butter_nvfp4_moe_gateup_bm16_mpp_f32"
                : "butter_nvfp4_moe_gateup_bm16_mpp_bf16"
            let pso = PSOCache.shared.pipelineState(for: psoName)
            guard let enc = cmd.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pso)
            var mTotalU = UInt32(mTotal)
            var nOutU = UInt32(nOut)
            var kInU = UInt32(kIn)
            var blockU = UInt32(blockSize)
            var globalV = global
            enc.setBuffer(input.buffer, offset: input.offset, index: 0)
            enc.setBuffer(rows.buffer, offset: rows.offset, index: 1)
            enc.setBuffer(weightGate.buffer, offset: weightGate.offset, index: 2)
            enc.setBuffer(scalesGate.buffer, offset: scalesGate.offset, index: 3)
            enc.setBuffer(weightUp.buffer, offset: weightUp.offset, index: 4)
            enc.setBuffer(scalesUp.buffer, offset: scalesUp.offset, index: 5)
            enc.setBuffer(indices.buffer, offset: indices.offset, index: 6)
            enc.setBuffer(outGate.buffer, offset: outGate.offset, index: 7)
            enc.setBuffer(outUp.buffer, offset: outUp.offset, index: 8)
            enc.setBytes(&mTotalU, length: 4, index: 9)
            enc.setBytes(&nOutU, length: 4, index: 10)
            enc.setBytes(&kInU, length: 4, index: 11)
            enc.setBytes(&blockU, length: 4, index: 12)
            enc.setBytes(&globalV, length: 4, index: 13)
            let grid = MTLSize(
                width: nOut / 32, height: (mTotal + 15) / 16, depth: 1)
            enc.dispatchThreadgroups(
                grid, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            return
        }
        // Default: materialise gather once + concurrent iron BM16-BK32 pair
        // (half K-loop vs classic BK16; skips butter xrows path).
        // Escape xrows: BUTTER_NVFP4_MPP_XROWS=1 (prior default).
        let wantXrows =
            ButterEnv["BUTTER_NVFP4_MPP_XROWS"] == "1"
        if wantXrows {
            nvfp4MoeGatherBm16XRowsPair(
                input: input, xRows: rows,
                weight0: weightGate, scales0: scalesGate, out0: outGate,
                weight1: weightUp, scales1: scalesUp, out1: outUp,
                indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd)
            return
        }
        // Gather tokens once into [mTotal, kIn], then iron BK32 concurrent pair.
        // Grow-only intermediateScratch — bare Ops.gather allocates Tensor.empty
        // per MoE layer (T=512 mTotal=4096 kIn=2048 → 16–32 MB × 39 layers) and
        // shreds the driver pool under swap pressure (same class as shared-expert
        // scratch win T15 ~452→520).
        let xIn: Tensor
        if let xRows {
            let elemBytes: Int
            switch input.dtype {
            case .f32: elemBytes = 4
            case .f16, .bf16: elemBytes = 2
            default: elemBytes = 4
            }
            let xBytes = mTotal * kIn * elemBytes
            let xBuf = Device.shared.intermediateScratch(
                tag: "nvfp4_mpp_gateup_x_\(kIn)_\(input.dtype)",
                minBytes: xBytes)
            let gathered = Tensor(
                buffer: xBuf, offset: 0,
                shape: [mTotal, kIn], dtype: input.dtype)
            _ = Ops.gather(
                table: input, tokenIds: xRows, on: cmd, into: gathered)
            xIn = gathered
        } else {
            xIn = input
        }
        nvfp4MoeGatherBm16Pair(
            input: xIn,
            weight0: weightGate, scales0: scalesGate, out0: outGate,
            weight1: weightUp, scales1: scalesUp, out1: outUp,
            indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: blockSize, global: global, on: cmd)
    }

    /// Single BM16 MPP gather with `x_rows` (iron-bitexact clone).
    public static func nvfp4MoeGatherBm16XRows(
        input: Tensor, xRows: Tensor,
        weight: Tensor, scales: Tensor, indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(nOut % 32 == 0)
        precondition(xRows.dtype == .u32 && xRows.elementCount == mTotal)
        precondition(indices.dtype == .u32 && indices.elementCount == mTotal)
        precondition(out.elementCount == mTotal * nOut)
        // `butter_nvfp4_moe_gather_bm16_mpp_xrows_*` is MPP-only, same
        // family gap as `nvfp4MoeGatherBm16`'s default branch. Route
        // incapable hardware through the exact same materialise+gather
        // fallback the "other dtype" arm below already uses (that arm
        // itself now self-guards via `nvfp4MoeGatherBm16`).
        let psoName: String
        switch input.dtype {
        case .f32 where Ops.mppAutoCapable: psoName = "butter_nvfp4_moe_gather_bm16_mpp_xrows_f32"
        case .bf16 where Ops.mppAutoCapable: psoName = "butter_nvfp4_moe_gather_bm16_mpp_xrows_bf16"
        default:
            let xIn = Ops.gather(table: input, tokenIds: xRows, on: cmd)
            nvfp4MoeGatherBm16(
                input: xIn, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd, into: out)
            return
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        var mTotalU = UInt32(mTotal)
        var nOutU = UInt32(nOut)
        var kInU = UInt32(kIn)
        var blockU = UInt32(blockSize)
        var globalV = global
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(xRows.buffer, offset: xRows.offset, index: 1)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 3)
        enc.setBuffer(indices.buffer, offset: indices.offset, index: 4)
        enc.setBuffer(out.buffer, offset: out.offset, index: 5)
        enc.setBytes(&mTotalU, length: 4, index: 6)
        enc.setBytes(&nOutU, length: 4, index: 7)
        enc.setBytes(&kInU, length: 4, index: 8)
        enc.setBytes(&blockU, length: 4, index: 9)
        enc.setBytes(&globalV, length: 4, index: 10)
        let grid = MTLSize(
            width: nOut / 32, height: (mTotal + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(
            grid, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Concurrent pair of iron-bitexact xrows MPP gathers (gate+up).
    public static func nvfp4MoeGatherBm16XRowsPair(
        input: Tensor, xRows: Tensor,
        weight0: Tensor, scales0: Tensor, out0: Tensor,
        weight1: Tensor, scales1: Tensor, out1: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer
    ) {
        precondition(input.dtype == out0.dtype && input.dtype == out1.dtype)
        // Same MPP-only gap as the serial `nvfp4MoeGatherBm16XRows` above.
        let psoName: String
        switch input.dtype {
        case .f32 where Ops.mppAutoCapable: psoName = "butter_nvfp4_moe_gather_bm16_mpp_xrows_f32"
        case .bf16 where Ops.mppAutoCapable: psoName = "butter_nvfp4_moe_gather_bm16_mpp_xrows_bf16"
        default:
            nvfp4MoeGatherBm16XRows(
                input: input, xRows: xRows, weight: weight0, scales: scales0,
                indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd, into: out0)
            nvfp4MoeGatherBm16XRows(
                input: input, xRows: xRows, weight: weight1, scales: scales1,
                indices: indices, mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd, into: out1)
            return
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { return }
        enc.setComputePipelineState(pso)
        var mTotalU = UInt32(mTotal)
        var nOutU = UInt32(nOut)
        var kInU = UInt32(kIn)
        var blockU = UInt32(blockSize)
        var globalV = global
        let grid = MTLSize(
            width: nOut / 32, height: (mTotal + 15) / 16, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        // Gate
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(xRows.buffer, offset: xRows.offset, index: 1)
        enc.setBuffer(weight0.buffer, offset: weight0.offset, index: 2)
        enc.setBuffer(scales0.buffer, offset: scales0.offset, index: 3)
        enc.setBuffer(indices.buffer, offset: indices.offset, index: 4)
        enc.setBuffer(out0.buffer, offset: out0.offset, index: 5)
        enc.setBytes(&mTotalU, length: 4, index: 6)
        enc.setBytes(&nOutU, length: 4, index: 7)
        enc.setBytes(&kInU, length: 4, index: 8)
        enc.setBytes(&blockU, length: 4, index: 9)
        enc.setBytes(&globalV, length: 4, index: 10)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        // Up
        enc.setBuffer(weight1.buffer, offset: weight1.offset, index: 2)
        enc.setBuffer(scales1.buffer, offset: scales1.offset, index: 3)
        enc.setBuffer(out1.buffer, offset: out1.offset, index: 5)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// Concurrent pair of MPP gathers (gate+up) on one concurrent encoder.
    /// Prefer f32 `input` (shared-cast path). Both outs must match `input` dtype.
    /// Uses BM64 when `nvfp4MoeGatherPrefersBm64` (same grid/TPG as serial path).
    public static func nvfp4MoeGatherBm16Pair(
        input: Tensor,
        weight0: Tensor, scales0: Tensor, out0: Tensor,
        weight1: Tensor, scales1: Tensor, out1: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer
    ) {
        precondition(input.dtype == out0.dtype && input.dtype == out1.dtype)
        precondition(nOut % 32 == 0)
        // EG-MPP: see `nvfp4MoeGatherBm16`'s own header comment +
        // `LagunaNvfp4MppGather.swift`. Default ON (escape
        // BUTTER_NVFP4_MPP_GATHER=0), checked ahead of steel/NAX/NAX64/BM16
        // same as the serial path above.
        // BUTTER_LAGUNA_GATHER_V2 (default ON, escape =0):
        // expert-aligned A-device-direct iron kernel, checked ahead of
        // EG-MPP. See `LagunaNvfp4ExpertGather.swift`.
        if LagunaNvfp4ExpertGather.eligible(dtype: input.dtype, nOut: nOut, kIn: kIn, blockSize: blockSize) {
            LagunaNvfp4ExpertGather.callGatherPair(
                input: input,
                weight0: weight0, scales0: scales0, out0: out0,
                weight1: weight1, scales1: scales1, out1: out1,
                indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global,
                on: cmd, device: Device.shared)
            return
        }
        if LagunaNvfp4MppGather.eligible(dtype: input.dtype, nOut: nOut, kIn: kIn, blockSize: blockSize) {
            LagunaNvfp4MppGather.callGatherPair(
                input: input,
                weight0: weight0, scales0: scales0, out0: out0,
                weight1: weight1, scales1: scales1, out1: out1,
                indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global,
                on: cmd, device: Device.shared)
            return
        }
        if nvfp4MoeGatherPrefersSteel(dtype: input.dtype, mTotal: mTotal, nOut: nOut, kIn: kIn, blockSize: blockSize) {
            // T35: same BM16/BM64 run-length threshold as the serial path.
            let nExperts = weight0.shape[0]
            if nvfp4GatherSteelPrefersBm16(mTotal: mTotal, nExperts: nExperts) {
                nvfp4GatherSteelBm16Pair(
                    input: input,
                    weight0: weight0, scales0: scales0, out0: out0,
                    weight1: weight1, scales1: scales1, out1: out1,
                    indices: indices,
                    mTotal: mTotal, nOut: nOut, kIn: kIn,
                    blockSize: blockSize, global: global, on: cmd)
            } else {
                nvfp4GatherSteelPair(
                    input: input,
                    weight0: weight0, scales0: scales0, out0: out0,
                    weight1: weight1, scales1: scales1, out1: out1,
                    indices: indices,
                    mTotal: mTotal, nOut: nOut, kIn: kIn,
                    blockSize: blockSize, global: global, on: cmd)
            }
            return
        }
        if nvfp4MoeGatherPrefersNax64(mTotal: mTotal, nOut: nOut, kIn: kIn) {
            nvfp4MoeGatherNax64Pair(
                input: input,
                weight0: weight0, scales0: scales0, out0: out0,
                weight1: weight1, scales1: scales1, out1: out1,
                indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd)
            return
        }
        if nvfp4MoeGatherPrefersNax(mTotal: mTotal, nOut: nOut, kIn: kIn) {
            nvfp4MoeGatherNaxPair(
                input: input,
                weight0: weight0, scales0: scales0, out0: out0,
                weight1: weight1, scales1: scales1, out1: out1,
                indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd)
            return
        }
        // Same DEFAULT-dispatch / `Ops.mppAutoCapable` gap as the serial
        // `nvfp4MoeGatherBm16` above (see its own comment at the matching
        // point) — the raw switch below only ever builds `_mpp_` PSO
        // names. Route through the serial entry point, which now
        // self-guards and degrades to the per-row GEMV fallback.
        guard Ops.mppAutoCapable else {
            nvfp4MoeGatherBm16(
                input: input, weight: weight0, scales: scales0, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd, into: out0)
            nvfp4MoeGatherBm16(
                input: input, weight: weight1, scales: scales1, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd, into: out1)
            return
        }
        let useBm64 = nvfp4MoeGatherPrefersBm64(mTotal: mTotal, nOut: nOut, kIn: kIn)
        let useBk64 = useBm64
            && nvfp4MoeGatherPrefersBm64Bk64(mTotal: mTotal, nOut: nOut, kIn: kIn)
        let forceBk16 =
            ButterEnv["BUTTER_NVFP4_GATHER_BK"] == "16"
        let useBk32 = !useBm64 && !forceBk16 && kIn % 32 == 0
        let psoName: String
        switch (input.dtype, useBm64, useBk64, useBk32) {
        case (.f32, true, true, _):
            psoName = "iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp_f32"
        case (.f16, true, true, _):
            psoName = "iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp_f16"
        case (.bf16, true, true, _):
            psoName = "iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp_bf16"
        case (.f32, true, false, _): psoName = "iron_nvfp4_moe_gather_qmm_bm64_mpp_f32"
        case (.f16, true, false, _): psoName = "iron_nvfp4_moe_gather_qmm_bm64_mpp_f16"
        case (.bf16, true, false, _): psoName = "iron_nvfp4_moe_gather_qmm_bm64_mpp_bf16"
        case (.f32, false, _, true): psoName = "iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_f32"
        case (.f16, false, _, true): psoName = "iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_f16"
        case (.bf16, false, _, true): psoName = "iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_bf16"
        case (.f32, false, _, false): psoName = "iron_nvfp4_moe_gather_qmm_bm16_mpp_f32"
        case (.f16, false, _, false): psoName = "iron_nvfp4_moe_gather_qmm_bm16_mpp_f16"
        case (.bf16, false, _, false): psoName = "iron_nvfp4_moe_gather_qmm_bm16_mpp_bf16"
        default:
            // Unknown dtype or unsupported combo → serial path (handles BM + cast).
            nvfp4MoeGatherBm16(
                input: input, weight: weight0, scales: scales0, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd, into: out0)
            nvfp4MoeGatherBm16(
                input: input, weight: weight1, scales: scales1, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd, into: out1)
            return
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { return }
        enc.setComputePipelineState(pso)
        var mTotalU = UInt32(mTotal)
        var nOutU = UInt32(nOut)
        var kInU = UInt32(kIn)
        var blockU = UInt32(blockSize)
        var globalV = global
        // Iron contracts: BM16 [N/32, ceil(M/16)] TPG=32; BM64 [N/64, ceil(M/64)] TPG=128.
        let grid: MTLSize
        let tg: MTLSize
        if useBm64 {
            precondition(nOut % 64 == 0 && kIn % 32 == 0)
            grid = MTLSize(
                width: nOut / 64,
                height: (mTotal + 63) / 64,
                depth: 1)
            tg = MTLSize(width: 128, height: 1, depth: 1)
        } else {
            grid = MTLSize(
                width: nOut / 32,
                height: (mTotal + 15) / 16,
                depth: 1)
            tg = MTLSize(width: 32, height: 1, depth: 1)
        }
        // Dispatch 0: gate
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(weight0.buffer, offset: weight0.offset, index: 1)
        enc.setBuffer(scales0.buffer, offset: scales0.offset, index: 2)
        enc.setBuffer(indices.buffer, offset: indices.offset, index: 3)
        enc.setBuffer(out0.buffer, offset: out0.offset, index: 4)
        enc.setBytes(&mTotalU, length: 4, index: 5)
        enc.setBytes(&nOutU, length: 4, index: 6)
        enc.setBytes(&kInU, length: 4, index: 7)
        enc.setBytes(&blockU, length: 4, index: 8)
        enc.setBytes(&globalV, length: 4, index: 9)
        // coop_tile MUST be dispatchThreadgroups (grid in TG counts), never
        // dispatchThreads — same trap as the serial BM64 path.
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        // Dispatch 1: up (same x/indices; independent w/out)
        enc.setBuffer(weight1.buffer, offset: weight1.offset, index: 1)
        enc.setBuffer(scales1.buffer, offset: scales1.offset, index: 2)
        enc.setBuffer(out1.buffer, offset: out1.offset, index: 4)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    // Live-compiled f32→bf16 (no iron cast_to_bf16 emitted today).
    nonisolated(unsafe) private static var castF32ToBf16PSO: MTLComputePipelineState?
    private static let castPSOLock = NSLock()

    /// Public for MoELayer f32-MPP shared-cast path (single cast of down out).
    public static func castF32ToBf16(
        _ input: Tensor, into out: Tensor, on cmd: MTLCommandBuffer
    ) {
        precondition(input.dtype == .f32 && out.dtype == .bf16)
        precondition(input.elementCount == out.elementCount)
        let n = input.elementCount
        castPSOLock.lock()
        let pso: MTLComputePipelineState = {
            if let p = castF32ToBf16PSO { return p }
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void butter_cast_f32_to_bf16(
                device const float *inp [[buffer(0)]],
                device bfloat *out [[buffer(1)]],
                uint i [[thread_position_in_grid]]) {
                out[i] = bfloat(inp[i]);
            }
            """
            let dev = Device.shared.mtlDevice
            let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
            let fn = lib.makeFunction(name: "butter_cast_f32_to_bf16")!
            let p = try! dev.makeComputePipelineState(function: fn)
            castF32ToBf16PSO = p
            return p
        }()
        castPSOLock.unlock()
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(pso)
            enc.setBuffer(input.buffer, offset: input.offset, index: 0)
            enc.setBuffer(out.buffer, offset: out.offset, index: 1)
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(n, 256), height: 1, depth: 1))
            enc.endEncoding()
        }
    }

    // Fused gather + bf16→f32 for MPP promote path (skips intermediate bf16 gather buffer).
    nonisolated(unsafe) private static var gatherCastPSO: MTLComputePipelineState?
    private static let gatherCastLock = NSLock()

    /// `out[m, d] = float(table[tokenIds[m], d])` for bf16 table → f32 out.
    /// Replaces Ops.gather + castToF32 for NVFP4 MPP f32-promote prefill.
    public static func gatherCastBf16ToF32(
        table: Tensor,  // [T, H] bf16
        tokenIds: Tensor,  // [mTotal] u32
        mTotal: Int,
        hidden: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor  // [mTotal, H] f32
    ) {
        precondition(table.dtype == .bf16)
        precondition(tokenIds.dtype == .u32 && tokenIds.elementCount >= mTotal)
        precondition(out.dtype == .f32 && out.elementCount >= mTotal * hidden)
        precondition(table.shape.count == 2 && table.shape[1] == hidden)
        gatherCastLock.lock()
        let pso: MTLComputePipelineState = {
            if let p = gatherCastPSO { return p }
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void butter_gather_cast_bf16_f32(
                device const bfloat *table [[buffer(0)]],
                device const uint *ids [[buffer(1)]],
                device float *out [[buffer(2)]],
                constant uint &hidden [[buffer(3)]],
                constant uint &m_total [[buffer(4)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint m = gid.x;
                uint d = gid.y;
                if (m >= m_total || d >= hidden) return;
                uint row = ids[m];
                out[m * hidden + d] = float(table[row * hidden + d]);
            }
            """
            let dev = Device.shared.mtlDevice
            let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
            let fn = lib.makeFunction(name: "butter_gather_cast_bf16_f32")!
            let p = try! dev.makeComputePipelineState(function: fn)
            gatherCastPSO = p
            return p
        }()
        gatherCastLock.unlock()
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table.buffer, offset: table.offset, index: 0)
        enc.setBuffer(tokenIds.buffer, offset: tokenIds.offset, index: 1)
        enc.setBuffer(out.buffer, offset: out.offset, index: 2)
        var hU = UInt32(hidden)
        var mU = UInt32(mTotal)
        enc.setBytes(&hU, length: 4, index: 3)
        enc.setBytes(&mU, length: 4, index: 4)
        // 2D grid (m, hidden)
        let tgW = min(mTotal, 16)
        let tgH = min(hidden, 16)
        enc.dispatchThreads(
            MTLSize(width: mTotal, height: hidden, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgW, height: tgH, depth: 1))
        enc.endEncoding()
    }

    /// Shared NAX cooperative-tensor dispatch for `nvfp4Qmm`'s aligned
    /// and M-padded call paths -- BM=BN=32, TPG 128 (4 SG × 32), grid
    /// `[out/32, m/32, 1]`. `m` here is whatever row count the caller
    /// wants dispatched (the real `m` for the already-aligned path, the
    /// padded `m32` for the pad path); `input`/`result` must have `m`
    /// rows. Returns `false` for a dtype the NAX kernels don't cover
    /// (caller falls back to the reduction kernel), `true` once the
    /// kernel has been encoded on `cmd`.
    private static func dispatchNaxQmm(
        weight: Tensor, scales: Tensor, input: Tensor, result: Tensor,
        m: Int, outDim: Int, inDim: Int,
        blockSize: Int, global: Float,
        on cmd: MTLCommandBuffer
    ) -> Bool {
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let grid = MTLSize(width: outDim / 32, height: m / 32, depth: 1)
        switch input.dtype {
        case .f32:
            IronKernels.iron_nvfp4_qmm_nax_f32_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                x: input.buffer, xOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                k: UInt32(inDim), n: UInt32(outDim),
                block_size: UInt32(blockSize), global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
            return true
        case .f16:
            IronKernels.iron_nvfp4_qmm_nax_f16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                x: input.buffer, xOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                k: UInt32(inDim), n: UInt32(outDim),
                block_size: UInt32(blockSize), global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
            return true
        case .bf16:
            IronKernels.iron_nvfp4_qmm_nax_bf16_threadgroups(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                x: input.buffer, xOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                k: UInt32(inDim), n: UInt32(outDim),
                block_size: UInt32(blockSize), global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
            return true
        default:
            return false
        }
    }

    /// Dense NVFP4 QMM: `Y[M, out] = X[M, in] · dequant(W)[out, in]ᵀ`.
    ///
    /// Prefers NAX cooperative-tensor (`iron_nvfp4_qmm_nax_*`) when
    /// `m` and `outDim` are multiples of 32 (BM=BN=32, TPG=128, grid
    /// `[out/32, m/32, 1]`). Escape: `BUTTER_NVFP4_QMM_NAX=0`.
    ///
    /// M-padding: real prefill token counts are almost never a multiple
    /// of 32, so gating NAX on `m % 32 == 0` directly meant it declined
    /// for nearly every real prompt, paying the slower reduction-kernel
    /// tax for the whole call over a remainder of at most 31 rows
    /// (measured at aligned M=2688: NAX 1926 tok/s vs reduction 1315,
    /// 1.46x). When `m` is 32-eligible-otherwise (`outDim`/`inDim` both
    /// 32-aligned) but not itself 32-aligned, this pads the input with
    /// zero rows up to the next multiple of 32, dispatches NAX at that
    /// padded `m32`, and copies/slices just the real `m` rows back out
    /// -- same idiom as `LagunaMppGemm.tryMppGemm`'s M-pad-to-64 path.
    ///
    /// Correctness: `out[m,:] = dequant(W)[out,:] · x[m,:]` -- each
    /// output row depends only on its own input row, so appending zero
    /// padding rows after row `m-1` cannot change rows `[0, m)` of the
    /// output; they come out byte-identical to the unpadded reduction
    /// kernel's result for those rows. The padding rows' own output is
    /// simply discarded (sliced off, or never copied into `out`).
    ///
    /// Escape: `BUTTER_NVFP4_QMM_NAX_PAD=0` disables the pad and reverts
    /// to the pre-pad behavior (NAX only engages on already-32-aligned
    /// `m`), for A/B against this change.
    public static func nvfp4Qmm(
        weight: Tensor, scales: Tensor, input: Tensor,
        m: Int, blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.dtype == .u32 && scales.dtype == .u8)
        precondition(weight.shape.count == 2)
        let outDim = weight.shape[0]
        let inDim = weight.shape[1] * 8
        precondition(input.elementCount == m * inDim)
        let result = out ?? Tensor.empty(shape: [m, outDim], dtype: input.dtype)

        // NAX: BM=BN=32, TPG 128 (4 SG × 32). Huge win vs reduction at
        // prefill shapes (T=512, mid=512, hidden multiple of 32).
        let naxEnv = ButterEnv["BUTTER_NVFP4_QMM_NAX"]
        let naxOn = naxEnv != "0" && naxEnv != "false"
        let padEnv = ButterEnv["BUTTER_NVFP4_QMM_NAX_PAD"]
        let padEnabled = padEnv != "0"

        if padEnabled, naxOn, m >= 32, m % 32 != 0, outDim % 32 == 0, inDim % 32 == 0 {
            let m32 = ((m + 31) / 32) * 32
            let rowBytes = inDim * input.dtype.byteSize
            let validBytes = m * rowBytes
            let tailBytes = (m32 - m) * rowBytes
            let xPad = Tensor.empty(shape: [m32, inDim], dtype: input.dtype)
            let padBlit = cmd.makeBlitCommandEncoder()!
            padBlit.copy(
                from: input.buffer, sourceOffset: input.offset,
                to: xPad.buffer, destinationOffset: xPad.offset, size: validBytes)
            padBlit.fill(
                buffer: xPad.buffer,
                range: (xPad.offset + validBytes) ..< (xPad.offset + validBytes + tailBytes),
                value: 0)
            padBlit.endEncoding()

            let resultPad = Tensor.empty(shape: [m32, outDim], dtype: input.dtype)
            if dispatchNaxQmm(
                weight: weight, scales: scales, input: xPad, result: resultPad,
                m: m32, outDim: outDim, inDim: inDim,
                blockSize: blockSize, global: global, on: cmd)
            {
                if let out {
                    // `into out:` contract: callers pass a pre-allocated
                    // (often scratch) buffer and read straight out of it,
                    // ignoring the return value -- see MoELayer.swift's
                    // `_ = Ops.nvfp4Qmm(..., into: gu)` call sites. Copy
                    // just the real `m` rows into `out`'s own buffer
                    // rather than returning `resultPad`'s.
                    let outRowBytes = outDim * input.dtype.byteSize
                    let outBlit = cmd.makeBlitCommandEncoder()!
                    outBlit.copy(
                        from: resultPad.buffer, sourceOffset: resultPad.offset,
                        to: out.buffer, destinationOffset: out.offset,
                        size: m * outRowBytes)
                    outBlit.endEncoding()
                    return out
                }
                return resultPad.slicedRows(start: 0, count: m)
            }
            // Unsupported dtype for the NAX kernels: abandon the pad
            // attempt (harmless -- xPad/resultPad just go unused) and
            // fall through to the reduction kernel below at the
            // original `m`/`result`, same as the aligned path's own
            // `default: break` fallback.
        }

        if naxOn, m >= 32, m % 32 == 0, outDim % 32 == 0, inDim % 32 == 0 {
            if dispatchNaxQmm(
                weight: weight, scales: scales, input: input, result: result,
                m: m, outDim: outDim, inDim: inDim,
                blockSize: blockSize, global: global, on: cmd)
            {
                return result
            }
        }

        // Reduction kernel: one TG per (m, out_row). dispatchThreads total
        // threads must be (m*outDim)*tpg so Metal forms m*outDim TGs of tpg.
        // (Missing *tpg was the shared-QMM quality break — tgid collapsed.)
        let tpg = 64
        let grid = MTLSize(width: m * outDim * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch input.dtype {
        case .f32:
            IronKernels.iron_nvfp4_qmm_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                x: input.buffer, xOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim),
                block_size: UInt32(blockSize), global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_nvfp4_qmm_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                x: input.buffer, xOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim),
                block_size: UInt32(blockSize), global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_nvfp4_qmm_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                x: input.buffer, xOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim),
                block_size: UInt32(blockSize), global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("nvfp4Qmm: unsupported dtype \(input.dtype)")
        }
        return result
    }

    /// Concurrent multi-segment NVFP4 QMM (expert-group prefill / shared gate+up).
    /// One concurrent encoder (or two when NAX + reduction mix); each entry is
    /// an independent dense QMM with its own m / weight / x / out. All rows
    /// share the same outDim/inDim.
    ///
    /// **Per-segment NAX:** a segment uses NAX when `m >= 32 && m % 32 == 0`
    /// (and out/in 32-aligned), same gate as `nvfp4Qmm`. Mixed batches no
    /// longer force *all* segments onto the reduction kernel — one small
    /// expert run used to kill NAX for every peer segment.
    /// Escape: `BUTTER_NVFP4_QMM_NAX=0`.
    public static func nvfp4QmmMany(
        weights: [Tensor], scales: [Tensor], inputs: [Tensor],
        outputs: [Tensor], ms: [Int],
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer
    ) {
        let n = weights.count
        precondition(n > 0 && scales.count == n && inputs.count == n
            && outputs.count == n && ms.count == n)
        let dtype = inputs[0].dtype
        let outDim = weights[0].shape[0]
        let inDim = weights[0].shape[1] * 8
        let naxEnv = ButterEnv["BUTTER_NVFP4_QMM_NAX"]
        let naxOn = naxEnv != "0" && naxEnv != "false"
        let dimsNax = naxOn && outDim % 32 == 0 && inDim % 32 == 0
        var naxIdx: [Int] = []
        var redIdx: [Int] = []
        naxIdx.reserveCapacity(n)
        redIdx.reserveCapacity(n)
        for i in 0 ..< n {
            let m = ms[i]
            precondition(m > 0)
            precondition(weights[i].shape[0] == outDim)
            precondition(inputs[i].elementCount == m * inDim)
            precondition(outputs[i].elementCount == m * outDim)
            if dimsNax && m >= 32 && m % 32 == 0 {
                naxIdx.append(i)
            } else {
                redIdx.append(i)
            }
        }
        var inDimV = UInt32(inDim)
        var outDimV = UInt32(outDim)
        var blockV = UInt32(blockSize)
        var globalV = global

        if !naxIdx.isEmpty {
            let psoName: String
            switch dtype {
            case .f32: psoName = "iron_nvfp4_qmm_nax_f32"
            case .f16: psoName = "iron_nvfp4_qmm_nax_f16"
            case .bf16: psoName = "iron_nvfp4_qmm_nax_bf16"
            default: fatalError("nvfp4QmmMany: bad dtype \(dtype)")
            }
            let pso = PSOCache.shared.pipelineState(for: psoName)
            guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
            else { return }
            enc.setComputePipelineState(pso)
            // NAX buffers: k=in, n=out at indices 4/5 (same layout as reduction).
            enc.setBytes(&inDimV, length: 4, index: 4)
            enc.setBytes(&outDimV, length: 4, index: 5)
            enc.setBytes(&blockV, length: 4, index: 6)
            enc.setBytes(&globalV, length: 4, index: 7)
            let tg = MTLSize(width: 128, height: 1, depth: 1)
            for i in naxIdx {
                let m = ms[i]
                enc.setBuffer(weights[i].buffer, offset: weights[i].offset, index: 0)
                enc.setBuffer(scales[i].buffer, offset: scales[i].offset, index: 1)
                enc.setBuffer(inputs[i].buffer, offset: inputs[i].offset, index: 2)
                enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 3)
                let grid = MTLSize(width: outDim / 32, height: m / 32, depth: 1)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            }
            enc.endEncoding()
        }

        if !redIdx.isEmpty {
            let psoName: String
            switch dtype {
            case .f32: psoName = "iron_nvfp4_qmm_f32"
            case .f16: psoName = "iron_nvfp4_qmm_f16"
            case .bf16: psoName = "iron_nvfp4_qmm_bf16"
            default: fatalError("nvfp4QmmMany: bad dtype \(dtype)")
            }
            let pso = PSOCache.shared.pipelineState(for: psoName)
            guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
            else { return }
            enc.setComputePipelineState(pso)
            enc.setBytes(&inDimV, length: 4, index: 4)
            enc.setBytes(&outDimV, length: 4, index: 5)
            enc.setBytes(&blockV, length: 4, index: 6)
            enc.setBytes(&globalV, length: 4, index: 7)
            let tpg = 64
            let tg = MTLSize(width: tpg, height: 1, depth: 1)
            for i in redIdx {
                let m = ms[i]
                enc.setBuffer(weights[i].buffer, offset: weights[i].offset, index: 0)
                enc.setBuffer(scales[i].buffer, offset: scales[i].offset, index: 1)
                enc.setBuffer(inputs[i].buffer, offset: inputs[i].offset, index: 2)
                enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 3)
                let grid = MTLSize(width: m * outDim * tpg, height: 1, depth: 1)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            }
            enc.endEncoding()
        }
    }

    /// Non-MPP gather QMM: for each row m, `out[m] = W[expert_ids[m]] · x[m]`.
    /// Weight `[E, out, in/8]` u32 contiguous as expert-major rows.
    /// Grid: one TG per (m, out_row) — same reduction convention as qgemv.
    public static func nvfp4GatherQmm(
        weight: Tensor, scales: Tensor, input: Tensor, expertIds: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(weight.dtype == .u32 && scales.dtype == .u8)
        precondition(expertIds.dtype == .u32 && expertIds.elementCount == mTotal)
        precondition(input.elementCount == mTotal * kIn)
        precondition(out.elementCount == mTotal * nOut)
        let tpg = 64
        let grid = MTLSize(width: mTotal * nOut * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch input.dtype {
        case .f32:
            IronKernels.iron_nvfp4_gather_qmm_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                x: input.buffer, xOffset: input.offset,
                output: out.buffer, outputOffset: out.offset,
                in_dim: UInt32(kIn), out_dim: UInt32(nOut),
                block_size: UInt32(blockSize), global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_nvfp4_gather_qmm_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                x: input.buffer, xOffset: input.offset,
                output: out.buffer, outputOffset: out.offset,
                in_dim: UInt32(kIn), out_dim: UInt32(nOut),
                block_size: UInt32(blockSize), global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            IronKernels.iron_nvfp4_gather_qmm_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                expert_ids: expertIds.buffer, expert_idsOffset: expertIds.offset,
                x: input.buffer, xOffset: input.offset,
                output: out.buffer, outputOffset: out.offset,
                in_dim: UInt32(kIn), out_dim: UInt32(nOut),
                block_size: UInt32(blockSize), global: global,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("nvfp4GatherQmm: bad dtype \(input.dtype)")
        }
    }

    /// MXFP8-E4M3 dequant GEMV: `y = W_mxfp8 · x`.
    /// `weight` u8 `[out, in]`, `scales` u8 `[out, in/blockSize]` (E8M0).
    /// Used for fast lm_head (half BF16 BW; argmax-faithful on Laguna).
    public static func mxfp8Qgemv(
        weight: Tensor, scales: Tensor, input: Tensor,
        blockSize: Int = 32,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.dtype == .u8 && scales.dtype == .u8)
        precondition(weight.shape.count == 2)
        let outDim = weight.shape[0]
        let inDim = weight.shape[1]
        precondition(input.elementCount == inDim)
        precondition(
            scales.shape == [outDim, inDim / blockSize],
            "mxfp8Qgemv: scales \(scales.shape) expected [\(outDim), \(inDim / blockSize)]")
        let result = out ?? Tensor.empty(shape: [outDim], dtype: input.dtype)
        let tpg = 64
        let grid = MTLSize(width: outDim * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch input.dtype {
        case .bf16:
            IronKernels.iron_mxfp8_e4m3_qgemv_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim), block_size: UInt32(blockSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            IronKernels.iron_mxfp8_e4m3_qgemv_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim), block_size: UInt32(blockSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f32:
            IronKernels.iron_mxfp8_e4m3_qgemv_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim), block_size: UInt32(blockSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("mxfp8Qgemv: bad dtype \(input.dtype)")
        }
        return result
    }
}

// ─── Mxfp8Linear ─────────────────────────────────────────────────────

/// MXFP8-E4M3 linear (lm_head coarse / full replace for decode BW).
public final class Mxfp8Linear: Module {
    public let weight: Tensor  // u8 [out, in]
    public let scales: Tensor  // u8 [out, in/32]
    public let blockSize: Int
    public let outDim: Int
    public let inDim: Int

    public init(weight: Tensor, scales: Tensor, blockSize: Int = 32) {
        precondition(weight.dtype == .u8 && scales.dtype == .u8)
        self.weight = weight
        self.scales = scales
        self.blockSize = blockSize
        self.outDim = weight.shape[0]
        self.inDim = weight.shape[1]
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("scales", scales)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.mxfp8Qgemv(
            weight: weight, scales: scales, input: x,
            blockSize: blockSize, on: cmd)
    }
}

// ─── Nvfp4Linear ─────────────────────────────────────────────────────

/// Single-projection NVFP4 linear (shared expert / future dense NVFP4).
public final class Nvfp4Linear: Module {
    public let weight: Tensor  // u32 [out, in/8]
    public let scales: Tensor  // u8  [out, in/16]
    public let blockSize: Int
    public let global: Float
    public let outDim: Int
    public let inDim: Int

    public init(
        weight: Tensor, scales: Tensor,
        blockSize: Int = 16, global: Float = 1.0
    ) {
        precondition(weight.dtype == .u32 && scales.dtype == .u8)
        precondition(weight.shape.count == 2)
        self.weight = weight
        self.scales = scales
        self.blockSize = blockSize
        self.global = global
        self.outDim = weight.shape[0]
        self.inDim = weight.shape[1] * 8
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("scales", scales)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.nvfp4Qgemv(
            weight: weight, scales: scales, input: x,
            blockSize: blockSize, global: global, on: cmd)
    }

    public func callMany(
        _ x: Tensor, t: Int,
        on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor? = nil
    ) -> Tensor {
        if t == 1 {
            if let out, out.elementCount >= outDim {
                let row = Tensor(
                    buffer: out.buffer, offset: out.offset,
                    shape: [outDim], dtype: x.dtype)
                _ = Ops.nvfp4Qgemv(
                    weight: weight, scales: scales, input: x,
                    blockSize: blockSize, global: global, on: cmd, into: row)
                return row.reshaped(to: [1, outDim])
            }
            let y = callAsFunction(x, on: cmd)
            return y.reshaped(to: [1, outDim])
        }
        // Batched NVFP4 QMM (one TG per (m, out_row)) — board prefill path.
        let result: Tensor
        if let out {
            precondition(out.elementCount >= t * outDim)
            result = Tensor(
                buffer: out.buffer, offset: out.offset,
                shape: [t, outDim], dtype: x.dtype)
        } else {
            result = Tensor.empty(shape: [t, outDim], dtype: x.dtype, device: device)
        }
        _ = Ops.nvfp4Qmm(
            weight: weight, scales: scales, input: x,
            m: t, blockSize: blockSize, global: global,
            on: cmd, into: result)
        return result
    }
}

