// Copyright 2026 Tom Turney (@TheTom)
//
// Expert-aligned, A-device-direct NVFP4 gather
// QMM (`iron_nvfp4_moe_gather_qmm_expert_mpp_k{2048,512}_bf16`, see
// `crates/wh-iron-std/src/kernels/moe/moe_gather_qmm_expert_mpp.rs` in
// the iron tree for the kernel-side design + the structural diff it
// implements). Opt-in A/B flag: `BUTTER_LAGUNA_GATHER_V2=1`; default
// OFF, the shipped EG-MPP chain (`LagunaNvfp4MppGather`) is untouched.
//
// WHAT'S DIFFERENT FROM EG-MPP (the shipped default):
//   * grid.y is the EXPERT id (one TG per expert per N-tile; in-kernel
//     binary search finds the run) — no tile-plan prepass, no
//     expertOffsets pass, no padding-tile slots;
//   * the A operand streams DEVICE-DIRECT as f16 (no X threadgroup
//     staging slab at all), which is why the input is gather-CAST
//     bf16→f16 here (same single pass as EG's sort-order gather, cast
//     folded in — no extra dispatch);
//   * W-only 2^-14 range fold (single ×2^14 at store), integer E2M1
//     decode, one scale decode per lane per K-tile, per-SG band skip.
//
// Kernel-level microbench (iron tree, M5 Max, 256-expert sorted
// routing, identical shapes): gateup T=512 +93%, gateup ragged +139%,
// down T=512 +254% vs the flat-tile iron BM64/BK64 MPP baseline.
//
// ROW-ORDER ABI: identical to `LagunaNvfp4MppGather` post wiring-fix —
// arbitrary row order in, same row order out. The local sort
// degenerates to an identity permutation on MoELayer's real
// sorted-plan path.
//
// PADDING CONTRACT: the kernel's device-direct A tiles read whole
// 32-row windows, so the sorted-f16 input scratch is allocated with 64
// padding rows past mTotal (garbage there only ever lands in masked
// output rows — each C row is the dot of exactly one A row).
//
// K IS BAKED into the kernel variant (`k2048` gate/up, `k512` down —
// the device-direct A stride must be a compile-time literal), so
// eligibility is exactly kIn ∈ {2048, 512}: precisely Laguna's two
// routed-expert projections. Every other shape falls through to the
// existing chain untouched.

import Foundation
import Metal
import IronSwift

enum LagunaNvfp4ExpertGather {
    /// Opt-in A/B flag (default OFF): `BUTTER_LAGUNA_GATHER_V2=1`.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_GATHER_V2"] == "1"
    }

    /// Same MPP capability gate as every other `matmul2d` kernel here.
    static var capable: Bool { Ops.mppAutoCapable }

    /// bf16 only; `kIn` must match a baked K variant; `nOut % 64 == 0`
    /// (BN=64, no N mask); `blockSize == 16` (the kernel's scale-pair
    /// amortization assumes 2 packs/group).
    static func eligible(dtype: DType, nOut: Int, kIn: Int, blockSize: Int) -> Bool {
        enabled && capable && dtype == .bf16 && nOut % 64 == 0
            && (kIn == 2048 || kIn == 512) && blockSize == 16
    }

    private static func kernelName(kIn: Int) -> String {
        kIn == 2048
            ? "iron_nvfp4_moe_gather_qmm_expert_mpp_k2048_bf16"
            : "iron_nvfp4_moe_gather_qmm_expert_mpp_k512_bf16"
    }

    // MARK: - Row sort (identical mechanism to LagunaNvfp4MppGather)

    private struct RowSort {
        let sortedExperts: Tensor
        let sourceRows: Tensor
        let invPerm: Tensor
    }

    private static func sortRowsByExpert(
        indices: Tensor, mTotal: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> RowSort {
        let seBuf = device.intermediateScratch(tag: "nvfp4_xg_se", minBytes: mTotal * 4)
        let stBuf = device.intermediateScratch(tag: "nvfp4_xg_st", minBytes: mTotal * 4)
        let ipBuf = device.intermediateScratch(tag: "nvfp4_xg_ip", minBytes: mTotal * 4)
        let se = Tensor(buffer: seBuf, offset: 0, shape: [mTotal], dtype: .u32)
        let st = Tensor(buffer: stBuf, offset: 0, shape: [mTotal], dtype: .u32)
        let ip = Tensor(buffer: ipBuf, offset: 0, shape: [mTotal], dtype: .u32)
        Ops.moeSortPlan(
            topkIds: indices, sortedExperts: se, sourceTokens: st, invPerm: ip,
            mTotal: mTotal, k: 1, on: cmd)
        return RowSort(sortedExperts: se, sourceRows: st, invPerm: ip)
    }

    // MARK: - Fused sort-order gather + bf16→f16 cast

    nonisolated(unsafe) private static var gatherCastF16PSO: MTLComputePipelineState?
    private static let gatherCastF16Lock = NSLock()

    /// One pass: `out_f16[m, :] = half(table_bf16[ids[m], :])` — the
    /// same gather EG runs to bring rows into sorted order, with the
    /// f16 cast the device-direct A operand needs folded in for free.
    private static func gatherCastBf16ToF16(
        table: Tensor, tokenIds: Tensor, mTotal: Int, hidden: Int,
        on cmd: MTLCommandBuffer, into out: Tensor
    ) {
        precondition(table.dtype == .bf16 && out.dtype == .f16)
        gatherCastF16Lock.lock()
        let pso: MTLComputePipelineState = {
            if let p = gatherCastF16PSO { return p }
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void butter_gather_cast_bf16_f16(
                device const bfloat *table [[buffer(0)]],
                device const uint *ids [[buffer(1)]],
                device half *out [[buffer(2)]],
                constant uint &hidden [[buffer(3)]],
                constant uint &m_total [[buffer(4)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint m = gid.x;
                uint d = gid.y;
                if (m >= m_total || d >= hidden) return;
                uint row = ids[m];
                // Saturating cast: bf16 exponent range exceeds half's;
                // real Laguna activations top out ~800 (T23 calibration,
                // 80x under half max) but a clamp keeps pathological
                // values finite instead of inf-poisoning a valid row.
                float v = float(table[row * hidden + d]);
                out[m * hidden + d] = half(clamp(v, -65504.0f, 65504.0f));
            }
            """
            let dev = Device.shared.mtlDevice
            let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
            let fn = lib.makeFunction(name: "butter_gather_cast_bf16_f16")!
            let p = try! dev.makeComputePipelineState(function: fn)
            gatherCastF16PSO = p
            return p
        }()
        gatherCastF16Lock.unlock()
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table.buffer, offset: table.offset, index: 0)
        enc.setBuffer(tokenIds.buffer, offset: tokenIds.offset, index: 1)
        enc.setBuffer(out.buffer, offset: out.offset, index: 2)
        var hU = UInt32(hidden)
        var mU = UInt32(mTotal)
        enc.setBytes(&hU, length: 4, index: 3)
        enc.setBytes(&mU, length: 4, index: 4)
        let tgW = min(mTotal, 16)
        let tgH = min(hidden, 16)
        enc.dispatchThreads(
            MTLSize(width: mTotal, height: hidden, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgW, height: tgH, depth: 1))
        enc.endEncoding()
    }

    // MARK: - GEMM dispatch helpers

    /// Encode one expert-aligned GEMM dispatch (buffer layout matches
    /// the generated IronKernels wrapper; hand-encoded so gate+up can
    /// share one concurrent encoder).
    private static func encode(
        _ enc: MTLComputeCommandEncoder, pso: MTLComputePipelineState,
        x: Tensor, weight: Tensor, scales: Tensor, sortedExperts: Tensor,
        out: Tensor, mTotal: Int, nOut: Int, nExperts: Int,
        blockSize: Int, global: Float
    ) {
        enc.setComputePipelineState(pso)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(sortedExperts.buffer, offset: sortedExperts.offset, index: 3)
        enc.setBuffer(out.buffer, offset: out.offset, index: 4)
        var mU = UInt32(mTotal)
        var nU = UInt32(nOut)
        var bU = UInt32(blockSize)
        var gV = global
        enc.setBytes(&mU, length: 4, index: 5)
        enc.setBytes(&nU, length: 4, index: 6)
        enc.setBytes(&bU, length: 4, index: 7)
        enc.setBytes(&gV, length: 4, index: 8)
        // coop_tile kernels MUST use dispatchThreadgroups (grid in TG
        // units) — same trap note as every other MPP kernel here.
        enc.dispatchThreadgroups(
            MTLSize(width: nOut / 64, height: nExperts, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    // MARK: - Public entry points (mirror LagunaNvfp4MppGather's ABI)

    static func callGather(
        input: Tensor, weight: Tensor, scales: Tensor, indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor
    ) {
        precondition(input.dtype == .bf16 && out.dtype == .bf16)
        precondition(weight.dtype == .u32 && scales.dtype == .u8)
        precondition(indices.dtype == .u32 && indices.elementCount == mTotal)
        precondition(input.elementCount == mTotal * kIn)
        precondition(out.elementCount == mTotal * nOut)
        precondition(kIn == 2048 || kIn == 512, "K is baked into the kernel variant")
        precondition(nOut % 64 == 0 && blockSize == 16)
        let nExperts = weight.shape[0]
        precondition(mTotal <= 65536, "16-step binary search bound")

        let sort = sortRowsByExpert(indices: indices, mTotal: mTotal, on: cmd, device: device)

        // Sorted f16 A with the 64-row padding contract (see header).
        let sortedInBuf = device.intermediateScratch(
            tag: "nvfp4_xg_in_\(kIn)", minBytes: (mTotal + 64) * kIn * 2)
        let sortedIn = Tensor(
            buffer: sortedInBuf, offset: 0, shape: [mTotal, kIn], dtype: .f16)
        gatherCastBf16ToF16(
            table: input, tokenIds: sort.sourceRows, mTotal: mTotal, hidden: kIn,
            on: cmd, into: sortedIn)

        let sortedOutBuf = device.intermediateScratch(
            tag: "nvfp4_xg_out_\(nOut)", minBytes: mTotal * nOut * 2)
        let sortedOut = Tensor(
            buffer: sortedOutBuf, offset: 0, shape: [mTotal, nOut], dtype: .bf16)

        let pso = PSOCache.shared.pipelineState(for: kernelName(kIn: kIn))
        guard let enc = cmd.makeComputeCommandEncoder() else {
            fatalError("LagunaNvfp4ExpertGather.callGather: could not make compute encoder")
        }
        encode(
            enc, pso: pso, x: sortedIn, weight: weight, scales: scales,
            sortedExperts: sort.sortedExperts, out: sortedOut,
            mTotal: mTotal, nOut: nOut, nExperts: nExperts,
            blockSize: blockSize, global: global)
        enc.endEncoding()

        _ = Ops.gather(table: sortedOut, tokenIds: sort.invPerm, on: cmd, into: out)
    }

    /// Concurrent gate+up pair sharing one sort + one sorted-f16 gather.
    static func callGatherPair(
        input: Tensor,
        weight0: Tensor, scales0: Tensor, out0: Tensor,
        weight1: Tensor, scales1: Tensor, out1: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int = 16, global: Float = 1.0,
        on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(input.dtype == .bf16 && out0.dtype == .bf16 && out1.dtype == .bf16)
        precondition(kIn == 2048 || kIn == 512, "K is baked into the kernel variant")
        precondition(nOut % 64 == 0 && blockSize == 16)
        let nExperts = weight0.shape[0]
        precondition(mTotal <= 65536, "16-step binary search bound")

        let sort = sortRowsByExpert(indices: indices, mTotal: mTotal, on: cmd, device: device)

        let sortedInBuf = device.intermediateScratch(
            tag: "nvfp4_xg_in_\(kIn)", minBytes: (mTotal + 64) * kIn * 2)
        let sortedIn = Tensor(
            buffer: sortedInBuf, offset: 0, shape: [mTotal, kIn], dtype: .f16)
        gatherCastBf16ToF16(
            table: input, tokenIds: sort.sourceRows, mTotal: mTotal, hidden: kIn,
            on: cmd, into: sortedIn)

        let sortedOut0Buf = device.intermediateScratch(
            tag: "nvfp4_xg_out0_\(nOut)", minBytes: mTotal * nOut * 2)
        let sortedOut1Buf = device.intermediateScratch(
            tag: "nvfp4_xg_out1_\(nOut)", minBytes: mTotal * nOut * 2)
        let sortedOut0 = Tensor(
            buffer: sortedOut0Buf, offset: 0, shape: [mTotal, nOut], dtype: .bf16)
        let sortedOut1 = Tensor(
            buffer: sortedOut1Buf, offset: 0, shape: [mTotal, nOut], dtype: .bf16)

        let pso = PSOCache.shared.pipelineState(for: kernelName(kIn: kIn))
        guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) else {
            fatalError("LagunaNvfp4ExpertGather.callGatherPair: could not make compute encoder")
        }
        encode(
            enc, pso: pso, x: sortedIn, weight: weight0, scales: scales0,
            sortedExperts: sort.sortedExperts, out: sortedOut0,
            mTotal: mTotal, nOut: nOut, nExperts: nExperts,
            blockSize: blockSize, global: global)
        encode(
            enc, pso: pso, x: sortedIn, weight: weight1, scales: scales1,
            sortedExperts: sort.sortedExperts, out: sortedOut1,
            mTotal: mTotal, nOut: nOut, nExperts: nExperts,
            blockSize: blockSize, global: global)
        enc.endEncoding()

        _ = Ops.gather(table: sortedOut0, tokenIds: sort.invPerm, on: cmd, into: out0)
        _ = Ops.gather(table: sortedOut1, tokenIds: sort.invPerm, on: cmd, into: out1)
    }
}
