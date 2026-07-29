// Copyright 2026 Tom Turney (@TheTom)
//
// Spike: live Metal NVFP4 MoE gather QMM with NAX-class geometry (BM=BN=64,
// TPG=128). Env-gated via BUTTER_NVFP4_GATHER_NAX=1 (default OFF).
//
// HONEST: kernels use MPP matmul2d 32×32×32 × 4 SGs — same public primitive
// MLX steel NAXTile wraps. Not a paste of Apple MLX headers. Weight layout +
// NVFP4 dequant match iron BM16/BM64 (u32 packs + E4M3×global). Escape hatch:
// unset env → existing BM16 MPP path unchanged.

import Foundation
import Metal
import IronSwift

extension Ops {
    /// Opt-in NAX-class gather when shapes align.
    /// Requires: `BUTTER_NVFP4_GATHER_NAX=1`, mTotal≥64, nOut%64==0, kIn%32==0.
    public static func nvfp4MoeGatherPrefersNax(
        mTotal: Int, nOut: Int, kIn: Int
    ) -> Bool {
        let env = ProcessInfo.processInfo.environment["BUTTER_NVFP4_GATHER_NAX"]
        guard env == "1" || env?.lowercased() == "true" else { return false }
        guard mTotal >= 64, nOut % 64 == 0, kIn % 32 == 0 else { return false }
        return true
    }

    /// Dispatch NAX-class NVFP4 gather QMM (live-compiled MPP body).
    /// Caller must ensure `nvfp4MoeGatherPrefersNax` (or equivalent shape checks).
    public static func nvfp4MoeGatherNax(
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
        precondition(nOut % 64 == 0 && kIn % 32 == 0)

        // Same late-layer half overflow escape as BM16: promote bf16 → f32 MPP
        // unless early-layer preferNativeBf16 / force flag.
        let forceBf16Mpp =
            preferNativeBf16
            || ProcessInfo.processInfo.environment["BUTTER_NVFP4_MPP_BF16"] == "1"
        if input.dtype == .bf16 && !forceBf16Mpp {
            let xTag = "nvfp4_nax_xf_\(kIn)x\(nOut)"
            let oTag = "nvfp4_nax_outf_\(kIn)x\(nOut)"
            let xBytes = mTotal * kIn * 4
            let oBytes = mTotal * nOut * 4
            let xBuf = Device.shared.intermediateScratch(tag: xTag, minBytes: xBytes)
            let oBuf = Device.shared.intermediateScratch(tag: oTag, minBytes: oBytes)
            let xF = Tensor(buffer: xBuf, offset: 0, shape: input.shape, dtype: .f32)
            let outF = Tensor(buffer: oBuf, offset: 0, shape: out.shape, dtype: .f32)
            Ops.castToF32(input, into: xF, on: cmd)
            nvfp4MoeGatherNax(
                input: xF, weight: weight, scales: scales, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global,
                on: cmd, into: outF)
            castF32ToBf16(outF, into: out, on: cmd)
            return
        }

        // Default: iron BM64 MPP (quality-proven, live-compiled via PSOCache).
        // Hand-written butter_*_nax_mpp_* is opt-in via GATHER_NAX_LIVE=1 —
        // it compiles and runs but currently fails full-model quality (empty
        // gen); kept for kernel iteration / microbench.
        let wantLive =
            ProcessInfo.processInfo.environment["BUTTER_NVFP4_GATHER_NAX_LIVE"] == "1"
        let grid = MTLSize(
            width: nOut / 64,
            height: (mTotal + 63) / 64,
            depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        let mTotalU = UInt32(mTotal)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let blockU = UInt32(blockSize)

        if wantLive, let psoName: String = {
            switch input.dtype {
            case .f32: return "butter_nvfp4_gather_qmm_rhs_nax_mpp_f32"
            case .bf16: return "butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16"
            default: return nil
            }
        }() {
            do {
                let pso = try PSOCache.shared.pipelineStateThrowing(for: psoName)
                if ProcessInfo.processInfo.environment["BUTTER_NVFP4_GATHER_NAX_VERBOSE"]
                    == "1"
                {
                    struct Once { nonisolated(unsafe) static var printed = false }
                    if !Once.printed {
                        Once.printed = true
                        fputs(
                            "nvfp4MoeGatherNax LIVE: \(psoName) "
                                + "m=\(mTotal) n=\(nOut) k=\(kIn)\n",
                            stderr)
                    }
                }
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
                return
            } catch {
                if ProcessInfo.processInfo.environment["BUTTER_NVFP4_GATHER_NAX_VERBOSE"]
                    == "1"
                {
                    fputs(
                        "nvfp4MoeGatherNax LIVE PSO failed (\(error)); iron BM64\n",
                        stderr)
                }
            }
        }

        if ProcessInfo.processInfo.environment["BUTTER_NVFP4_GATHER_NAX_VERBOSE"] == "1" {
            struct Once { nonisolated(unsafe) static var printed = false }
            if !Once.printed {
                Once.printed = true
                fputs(
                    "nvfp4MoeGatherNax: iron BM64 MPP (NAX-class geometry) "
                        + "m=\(mTotal) n=\(nOut) k=\(kIn) dtype=\(input.dtype)\n",
                    stderr)
            }
        }
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
        default:
            fatalError("nvfp4MoeGatherNax: unsupported dtype \(input.dtype)")
        }
    }

    /// Concurrent gate+up pair on the NAX-class gather kernel.
    public static func nvfp4MoeGatherNaxPair(
        input: Tensor,
        weight0: Tensor, scales0: Tensor, out0: Tensor,
        weight1: Tensor, scales1: Tensor, out1: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer
    ) {
        precondition(input.dtype == out0.dtype && input.dtype == out1.dtype)
        precondition(nOut % 64 == 0 && kIn % 32 == 0)
        // Prefer live-compiled butter NAX-class kernels (hand metal under
        // Resources/kernels/). Escape iron BM64 only: BUTTER_NVFP4_GATHER_NAX_LIVE=0.
        // Default when GATHER_NAX is on: try LIVE first.
        let liveEnv = ProcessInfo.processInfo.environment["BUTTER_NVFP4_GATHER_NAX_LIVE"]
        let wantLive = liveEnv != "0" && liveEnv?.lowercased() != "false"
        let livePsoName: String?
        switch input.dtype {
        case .f32: livePsoName = "butter_nvfp4_gather_qmm_rhs_nax_mpp_f32"
        case .bf16: livePsoName = "butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16"
        default: livePsoName = nil
        }
        if wantLive, let liveName = livePsoName {
            do {
                let pso = try PSOCache.shared.pipelineStateThrowing(for: liveName)
                if ProcessInfo.processInfo.environment["BUTTER_NVFP4_GATHER_NAX_VERBOSE"]
                    == "1"
                {
                    struct Once { nonisolated(unsafe) static var printed = false }
                    if !Once.printed {
                        Once.printed = true
                        fputs(
                            "nvfp4MoeGatherNaxPair LIVE: \(liveName) "
                                + "m=\(mTotal) n=\(nOut) k=\(kIn)\n",
                            stderr)
                    }
                }
                guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
                else { return }
                enc.setComputePipelineState(pso)
                var mTotalU = UInt32(mTotal)
                var nOutU = UInt32(nOut)
                var kInU = UInt32(kIn)
                var blockU = UInt32(blockSize)
                var globalV = global
                let grid = MTLSize(
                    width: nOut / 64,
                    height: (mTotal + 63) / 64,
                    depth: 1)
                let tg = MTLSize(width: 128, height: 1, depth: 1)
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
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                enc.setBuffer(weight1.buffer, offset: weight1.offset, index: 1)
                enc.setBuffer(scales1.buffer, offset: scales1.offset, index: 2)
                enc.setBuffer(out1.buffer, offset: out1.offset, index: 4)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                enc.endEncoding()
                return
            } catch {
                if ProcessInfo.processInfo.environment["BUTTER_NVFP4_GATHER_NAX_VERBOSE"]
                    == "1"
                {
                    fputs(
                        "nvfp4MoeGatherNaxPair LIVE failed (\(error)); iron BM64\n",
                        stderr)
                }
            }
        }
        let psoName: String
        switch input.dtype {
        case .f32: psoName = "iron_nvfp4_moe_gather_qmm_bm64_mpp_f32"
        case .bf16: psoName = "iron_nvfp4_moe_gather_qmm_bm64_mpp_bf16"
        case .f16: psoName = "iron_nvfp4_moe_gather_qmm_bm64_mpp_f16"
        default:
            nvfp4MoeGatherNax(
                input: input, weight: weight0, scales: scales0, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd, into: out0)
            nvfp4MoeGatherNax(
                input: input, weight: weight1, scales: scales1, indices: indices,
                mTotal: mTotal, nOut: nOut, kIn: kIn,
                blockSize: blockSize, global: global, on: cmd, into: out1)
            return
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        if ProcessInfo.processInfo.environment["BUTTER_NVFP4_GATHER_NAX_VERBOSE"] == "1" {
            struct Once { nonisolated(unsafe) static var printed = false }
            if !Once.printed {
                Once.printed = true
                fputs(
                    "nvfp4MoeGatherNaxPair: iron BM64 \(psoName) "
                        + "m=\(mTotal) n=\(nOut) k=\(kIn)\n",
                    stderr)
            }
        }
        guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { return }
        enc.setComputePipelineState(pso)
        var mTotalU = UInt32(mTotal)
        var nOutU = UInt32(nOut)
        var kInU = UInt32(kIn)
        var blockU = UInt32(blockSize)
        var globalV = global
        let grid = MTLSize(
            width: nOut / 64,
            height: (mTotal + 63) / 64,
            depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
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
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.setBuffer(weight1.buffer, offset: weight1.offset, index: 1)
        enc.setBuffer(scales1.buffer, offset: scales1.offset, index: 2)
        enc.setBuffer(out1.buffer, offset: out1.offset, index: 4)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}
