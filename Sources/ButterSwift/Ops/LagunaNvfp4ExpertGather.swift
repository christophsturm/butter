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

    // MARK: - Follow-on levers (scaffolding removal)

    /// A/B testing showed the gather bucket is scaffolding-bound
    /// (sort + gather-cast + scatter-back ≈ the GEMM itself). Each lever
    /// removes one piece, each behind its own opt-in env flag so the A/B
    /// arms compose incrementally on top of `BUTTER_LAGUNA_GATHER_V2=1`
    /// (which stays the master switch — none of this runs without it):
    ///
    ///   * `BUTTER_LAGUNA_V2_SKIP_SORT=1` — on MoELayer's sorted-plan
    ///     path the rows arriving here are ALREADY in expert-sorted
    ///     order (`Ops.moeSortPlan` ran upstream), so the local k=1
    ///     re-sort is the identity permutation, the sort-order gather is
    ///     a copy, and the invPerm scatter-back is a copy. Skipping them
    ///     is bit-identical: the caller's plan ids are used directly as
    ///     `sortedExperts`, the gather-cast reads straight from the
    ///     token table via the caller's `srcTok` (one pass, same values)
    ///     and the GEMM writes its output rows directly into the caller
    ///     target ("every row written exactly once" per the kernel
    ///     contract, so no zero-fill is needed either).
    ///
    /// The lever chain entry points (`gateUpChain` / `downChain`) are
    /// called by MoELayer's fused-gateup prefill branch only when the
    /// sorted plan exists — passing unsorted indices with skipSort on
    /// violates the kernel's "indices sorted ascending" contract and is
    /// the caller's responsibility (MoELayer guarantees it by
    /// construction; the lever tests pin the equivalence).
    ///   * `BUTTER_LAGUNA_V2_FUSED_SILU=1` — SiLU·mul fused into the
    ///     scatter-back. The base chain runs [2 invPerm output copies
    ///     + a separate `Ops.swiglu` dispatch] between the gate/up GEMMs
    ///     and the down GEMM; all three are elementwise passes over the
    ///     same `[mTotal, 512]` pair, so one kernel does
    ///     `inner[m] = bf16(silu(g[ip[m]]) · u[ip[m]])` in a single
    ///     read-modify-write (fusing SiLU into the GEMM epilogue itself
    ///     was previously found to lose to occupancy pressure — but the
    ///     scatter is elementwise, so fusion there is free). The f32
    ///     math reproduces the generated `iron_swiglu_bf16` expression
    ///     exactly (`x / (1 + exp(-x))`), so the store is bit-identical.
    ///   * `BUTTER_LAGUNA_V2_F16_INNER=1` — store the SwiGLU inner
    ///     activation f16 instead of bf16, so the down GEMM (whose A
    ///     operand is f16 device-direct) reads it with NO cast pass.
    ///     The base chain stores inner bf16 and then runs a full
    ///     bf16→f16 gather-cast just to feed the k512 kernel. To stay
    ///     BIT-IDENTICAL the f16 store deliberately double-rounds
    ///     through bf16 first — `f16(clamp(bf16(silu·u)))` — which is
    ///     exactly the value the deleted cast pass would have produced
    ///     (bf16→f16 is exact in range; the saturating clamp matches
    ///     `gatherCastBf16ToF16`). Range: measured Laguna inner tops out
    ///     ~|800| (calibrated) vs half's 65504 ceiling — 80× headroom,
    ///     pinned by the range-safety lever test at 10× the measured
    ///     bound.
    struct Levers {
        var skipSort: Bool
        var fusedSilu: Bool = false
        var f16Inner: Bool = false
        var any: Bool { skipSort || fusedSilu || f16Inner }
    }

    static var levers: Levers {
        let env = ProcessInfo.processInfo.environment
        return Levers(
            skipSort: env["BUTTER_LAGUNA_V2_SKIP_SORT"] == "1",
            fusedSilu: env["BUTTER_LAGUNA_V2_FUSED_SILU"] == "1",
            f16Inner: env["BUTTER_LAGUNA_V2_F16_INNER"] == "1")
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

    // MARK: - Fused SiLU·mul scatter-back (lever 2)

    nonisolated(unsafe) private static var siluScatterBf16PSO: MTLComputePipelineState?
    nonisolated(unsafe) private static var siluScatterF16PSO: MTLComputePipelineState?
    private static let siluScatterLock = NSLock()

    /// One pass replacing [gather(ip)→gate, gather(ip)→up, swiglu]:
    /// `inner[m, :] = silu(gs[ip[m], :]) · us[ip[m], :]`.
    /// `ip` is the invPerm scatter order (identity buffer on the
    /// skipSort path). f32 math matches the generated
    /// `iron_swiglu_bf16` bit-for-bit (`x / (1 + exp(-x))`), pinned by
    /// the lever test suite's byte-identity bar.
    ///
    /// `inner.dtype` selects the store: bf16 (wave-6 value exactly) or
    /// f16 for the f16Inner lever — the f16 store DELIBERATELY rounds
    /// through bf16 first and saturates at ±65504, reproducing
    /// [bf16 store → `gatherCastBf16ToF16`] to the bit (see the lever
    /// header). Internal (not private) so the lever tests can drive it
    /// with crafted magnitudes.
    static func siluScatter(
        gs: Tensor, us: Tensor, ip: Tensor,
        mTotal: Int, dimN: Int,
        on cmd: MTLCommandBuffer, into inner: Tensor
    ) {
        precondition(gs.dtype == .bf16 && us.dtype == .bf16)
        precondition(inner.dtype == .bf16 || inner.dtype == .f16)
        precondition(ip.dtype == .u32)
        let wantF16 = inner.dtype == .f16
        siluScatterLock.lock()
        if siluScatterBf16PSO == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void butter_v2_silu_scatter_bf16(
                device const bfloat *gs [[buffer(0)]],
                device const bfloat *us [[buffer(1)]],
                device const uint *ip [[buffer(2)]],
                device bfloat *inner [[buffer(3)]],
                constant uint &dimN [[buffer(4)]],
                constant uint &m_total [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint m = gid.x;
                uint d = gid.y;
                if (m >= m_total || d >= dimN) return;
                uint s = ip[m];
                float g = float(gs[s * dimN + d]);
                float u = float(us[s * dimN + d]);
                // Bit-exact clone of generated iron_swiglu_bf16:
                // silu(x) = x / (1 + exp(-x)) in f32, bf16 store.
                float sv = g / (1.0f + exp(-g));
                inner[m * dimN + d] = bfloat(sv * u);
            }
            kernel void butter_v2_silu_scatter_f16(
                device const bfloat *gs [[buffer(0)]],
                device const bfloat *us [[buffer(1)]],
                device const uint *ip [[buffer(2)]],
                device half *inner [[buffer(3)]],
                constant uint &dimN [[buffer(4)]],
                constant uint &m_total [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint m = gid.x;
                uint d = gid.y;
                if (m >= m_total || d >= dimN) return;
                uint s = ip[m];
                float g = float(gs[s * dimN + d]);
                float u = float(us[s * dimN + d]);
                float sv = g / (1.0f + exp(-g));
                // DELIBERATE double rounding: bf16 first (the wave-6
                // store), then the saturating f16 cast that the deleted
                // gatherCastBf16ToF16 pass would have applied — keeps
                // the down GEMM's A operand bit-identical to the
                // f16Inner-off chain.
                float v = float(bfloat(sv * u));
                inner[m * dimN + d] = half(clamp(v, -65504.0f, 65504.0f));
            }
            """
            let dev = Device.shared.mtlDevice
            let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
            siluScatterBf16PSO = try! dev.makeComputePipelineState(
                function: lib.makeFunction(name: "butter_v2_silu_scatter_bf16")!)
            siluScatterF16PSO = try! dev.makeComputePipelineState(
                function: lib.makeFunction(name: "butter_v2_silu_scatter_f16")!)
        }
        let pso = wantF16 ? siluScatterF16PSO! : siluScatterBf16PSO!
        siluScatterLock.unlock()
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(gs.buffer, offset: gs.offset, index: 0)
        enc.setBuffer(us.buffer, offset: us.offset, index: 1)
        enc.setBuffer(ip.buffer, offset: ip.offset, index: 2)
        enc.setBuffer(inner.buffer, offset: inner.offset, index: 3)
        var dU = UInt32(dimN)
        var mU = UInt32(mTotal)
        enc.setBytes(&dU, length: 4, index: 4)
        enc.setBytes(&mU, length: 4, index: 5)
        let tgW = min(mTotal, 16)
        let tgH = min(dimN, 16)
        enc.dispatchThreads(
            MTLSize(width: mTotal, height: dimN, depth: 1),
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
        sortedIndices: Bool = false,
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

        // Sorted f16 A with the 64-row padding contract (see header).
        let sortedInBuf = device.intermediateScratch(
            tag: "nvfp4_xg_in_\(kIn)", minBytes: (mTotal + 64) * kIn * 2)
        let sortedIn = Tensor(
            buffer: sortedInBuf, offset: 0, shape: [mTotal, kIn], dtype: .f16)

        if sortedIndices {
            // SKIP_SORT lever: `indices` is already sorted ascending and
            // the rows of `input` are already in that order (MoELayer's
            // sorted-plan path). The local k=1 sort would be the identity
            // permutation — skip it, cast in place order, and let the
            // GEMM write `out` directly (every row written exactly once
            // per the kernel contract; no scatter-back copy).
            let idBuf = Ops.identityRowsBuffer(minCount: mTotal, device: device)
            let idRows = Tensor(buffer: idBuf, offset: 0, shape: [mTotal], dtype: .u32)
            gatherCastBf16ToF16(
                table: input, tokenIds: idRows, mTotal: mTotal, hidden: kIn,
                on: cmd, into: sortedIn)
            let pso = PSOCache.shared.pipelineState(for: kernelName(kIn: kIn))
            guard let enc = cmd.makeComputeCommandEncoder() else {
                fatalError("LagunaNvfp4ExpertGather.callGather: could not make compute encoder")
            }
            encode(
                enc, pso: pso, x: sortedIn, weight: weight, scales: scales,
                sortedExperts: indices, out: out,
                mTotal: mTotal, nOut: nOut, nExperts: nExperts,
                blockSize: blockSize, global: global)
            enc.endEncoding()
            return
        }

        let sort = sortRowsByExpert(indices: indices, mTotal: mTotal, on: cmd, device: device)

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

    // MARK: - W6 follow-on lever chain (MoELayer fused-gateup prefill path)

    /// Gate+up stage of the lever chain. Called by `MoELayer`'s
    /// fused-gateup prefill branch ONLY on the sorted-plan path (rows in
    /// expert-sorted order, `indices` sorted ascending, `srcTok` maps
    /// sorted slot → token row of `hRows`). Returns the SwiGLU inner
    /// activation `[mTotal, nOut]` (row order preserved).
    ///
    /// skipSort lever ON here means: ONE fused gather-cast straight from
    /// the `[t, hidden]` token table (deletes the wave-6 chain's separate
    /// bf16 materialise gather AND the identity k=1 re-sort) and the
    /// GEMM pair writes `gateOut`/`upOut` directly (deletes two identity
    /// scatter-back copies). Bit-identical by construction:
    /// f16(bf16 row) is the same value whether the row was copied first,
    /// and the kernel writes every output row exactly once.
    static func gateUpChain(
        hRows: Tensor, srcTok: Tensor, indices: Tensor,
        gateWeight: Tensor, gateScales: Tensor,
        upWeight: Tensor, upScales: Tensor,
        gateOut: Tensor, upOut: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int, global: Float,
        levers: Levers,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(levers.any, "lever chain entered with no lever active")
        precondition(hRows.dtype == .bf16 && gateOut.dtype == .bf16 && upOut.dtype == .bf16)
        precondition(srcTok.dtype == .u32 && srcTok.elementCount == mTotal)
        precondition(indices.dtype == .u32 && indices.elementCount == mTotal)
        precondition(gateOut.elementCount == mTotal * nOut)
        precondition(upOut.elementCount == mTotal * nOut)
        precondition(kIn == 2048 || kIn == 512, "K is baked into the kernel variant")
        precondition(nOut % 64 == 0 && blockSize == 16)
        let nExperts = gateWeight.shape[0]
        precondition(mTotal <= 65536, "16-step binary search bound")

        // ── Stage A: A-operand staging (f16, padded, plan order) ──
        let sortedInBuf = device.intermediateScratch(
            tag: "nvfp4_xg_in_\(kIn)", minBytes: (mTotal + 64) * kIn * 2)
        let sortedIn = Tensor(
            buffer: sortedInBuf, offset: 0, shape: [mTotal, kIn], dtype: .f16)
        var sortInfo: RowSort? = nil
        if levers.skipSort {
            // One pass: token-table fetch + bf16→f16 cast, in plan order.
            gatherCastBf16ToF16(
                table: hRows, tokenIds: srcTok, mTotal: mTotal, hidden: kIn,
                on: cmd, into: sortedIn)
        } else {
            // Replicate the wave-6 chain: bf16 materialise gather +
            // local k=1 sort (identity on this path) + sort-order cast.
            let gBuf = device.intermediateScratch(
                tag: "nvfp4_xg_gathered_\(kIn)", minBytes: mTotal * kIn * 2)
            let gathered = Tensor(
                buffer: gBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)
            _ = Ops.gather(table: hRows, tokenIds: srcTok, on: cmd, into: gathered)
            let s = sortRowsByExpert(indices: indices, mTotal: mTotal, on: cmd, device: device)
            sortInfo = s
            gatherCastBf16ToF16(
                table: gathered, tokenIds: s.sourceRows, mTotal: mTotal, hidden: kIn,
                on: cmd, into: sortedIn)
        }
        let se = sortInfo?.sortedExperts ?? indices

        // ── Stage B: concurrent GEMM pair ──
        // Direct writes are only valid when no scatter/fusion follows.
        let gemmDirect = levers.skipSort && !levers.fusedSilu
        let dest0: Tensor
        let dest1: Tensor
        if gemmDirect {
            dest0 = gateOut
            dest1 = upOut
        } else {
            let s0Buf = device.intermediateScratch(
                tag: "nvfp4_xg_out0_\(nOut)", minBytes: mTotal * nOut * 2)
            let s1Buf = device.intermediateScratch(
                tag: "nvfp4_xg_out1_\(nOut)", minBytes: mTotal * nOut * 2)
            dest0 = Tensor(buffer: s0Buf, offset: 0, shape: [mTotal, nOut], dtype: .bf16)
            dest1 = Tensor(buffer: s1Buf, offset: 0, shape: [mTotal, nOut], dtype: .bf16)
        }
        let pso = PSOCache.shared.pipelineState(for: kernelName(kIn: kIn))
        guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) else {
            fatalError("LagunaNvfp4ExpertGather.gateUpChain: could not make compute encoder")
        }
        encode(
            enc, pso: pso, x: sortedIn, weight: gateWeight, scales: gateScales,
            sortedExperts: se, out: dest0,
            mTotal: mTotal, nOut: nOut, nExperts: nExperts,
            blockSize: blockSize, global: global)
        encode(
            enc, pso: pso, x: sortedIn, weight: upWeight, scales: upScales,
            sortedExperts: se, out: dest1,
            mTotal: mTotal, nOut: nOut, nExperts: nExperts,
            blockSize: blockSize, global: global)
        enc.endEncoding()

        // ── Stage C: scatter-back + SiLU·mul ──
        // f16Inner target: padded per the down kernel's device-direct A
        // contract (64 rows past mTotal; garbage there only ever lands
        // in masked output rows).
        func f16InnerScratch() -> Tensor {
            let buf = device.intermediateScratch(
                tag: "nvfp4_xg_inner_f16", minBytes: (mTotal + 64) * nOut * 2)
            return Tensor(buffer: buf, offset: 0, shape: [mTotal, nOut], dtype: .f16)
        }
        if levers.fusedSilu {
            let ip: Tensor
            if let s = sortInfo {
                ip = s.invPerm
            } else {
                let idBuf = Ops.identityRowsBuffer(minCount: mTotal, device: device)
                ip = Tensor(buffer: idBuf, offset: 0, shape: [mTotal], dtype: .u32)
            }
            let inner = levers.f16Inner ? f16InnerScratch() : gateOut
            siluScatter(
                gs: dest0, us: dest1, ip: ip,
                mTotal: mTotal, dimN: nOut, on: cmd, into: inner)
            return inner
        }
        if !gemmDirect {
            // Replicate the wave-6 scatter-back (identity copies here).
            _ = Ops.gather(table: dest0, tokenIds: sortInfo!.invPerm, on: cmd, into: gateOut)
            _ = Ops.gather(table: dest1, tokenIds: sortInfo!.invPerm, on: cmd, into: upOut)
        }
        if levers.f16Inner {
            // f16Inner without fusedSilu: same math as Ops.swiglu +
            // gather-cast, one pass, identity ids (gate/up are already
            // in caller row order here).
            let idBuf = Ops.identityRowsBuffer(minCount: mTotal, device: device)
            let ip = Tensor(buffer: idBuf, offset: 0, shape: [mTotal], dtype: .u32)
            let inner = f16InnerScratch()
            siluScatter(
                gs: gateOut, us: upOut, ip: ip,
                mTotal: mTotal, dimN: nOut, on: cmd, into: inner)
            return inner
        }
        return Ops.swiglu(gate: gateOut, up: upOut, on: cmd, into: gateOut)
    }

    /// Down stage of the lever chain — same sorted-plan contract as
    /// `gateUpChain`. `inner` is the SwiGLU output in plan row order.
    static func downChain(
        inner: Tensor, indices: Tensor,
        weight: Tensor, scales: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        blockSize: Int, global: Float,
        levers: Levers,
        on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor
    ) {
        precondition(levers.any, "lever chain entered with no lever active")
        if inner.dtype == .f16 {
            // f16Inner lever: the A operand is ALREADY f16 — no cast
            // pass at all.
            precondition(levers.f16Inner)
            precondition(out.dtype == .bf16 && weight.dtype == .u32 && scales.dtype == .u8)
            precondition(kIn == 2048 || kIn == 512, "K is baked into the kernel variant")
            precondition(nOut % 64 == 0 && blockSize == 16)
            precondition(
                inner.buffer.length - inner.offset >= (mTotal + 64) * kIn * 2,
                "f16 inner must carry the 64-row device-direct padding")
            let nExperts = weight.shape[0]
            let pso = PSOCache.shared.pipelineState(for: kernelName(kIn: kIn))
            if levers.skipSort {
                // Plan order in, plan order out: single GEMM dispatch.
                guard let enc = cmd.makeComputeCommandEncoder() else {
                    fatalError("LagunaNvfp4ExpertGather.downChain: could not make compute encoder")
                }
                encode(
                    enc, pso: pso, x: inner, weight: weight, scales: scales,
                    sortedExperts: indices, out: out,
                    mTotal: mTotal, nOut: nOut, nExperts: nExperts,
                    blockSize: blockSize, global: global)
                enc.endEncoding()
                return
            }
            // Without skipSort: replicate the wave-6 shape — local k=1
            // sort (identity on this path), f16 row reorder (plain
            // gather, no cast — that's the lever's saving here), GEMM,
            // invPerm scatter-back.
            let sort = sortRowsByExpert(indices: indices, mTotal: mTotal, on: cmd, device: device)
            let sortedInBuf = device.intermediateScratch(
                tag: "nvfp4_xg_in_\(kIn)", minBytes: (mTotal + 64) * kIn * 2)
            let sortedIn = Tensor(
                buffer: sortedInBuf, offset: 0, shape: [mTotal, kIn], dtype: .f16)
            _ = Ops.gather(table: inner, tokenIds: sort.sourceRows, on: cmd, into: sortedIn)
            let sortedOutBuf = device.intermediateScratch(
                tag: "nvfp4_xg_out_\(nOut)", minBytes: mTotal * nOut * 2)
            let sortedOut = Tensor(
                buffer: sortedOutBuf, offset: 0, shape: [mTotal, nOut], dtype: .bf16)
            guard let enc = cmd.makeComputeCommandEncoder() else {
                fatalError("LagunaNvfp4ExpertGather.downChain: could not make compute encoder")
            }
            encode(
                enc, pso: pso, x: sortedIn, weight: weight, scales: scales,
                sortedExperts: sort.sortedExperts, out: sortedOut,
                mTotal: mTotal, nOut: nOut, nExperts: nExperts,
                blockSize: blockSize, global: global)
            enc.endEncoding()
            _ = Ops.gather(table: sortedOut, tokenIds: sort.invPerm, on: cmd, into: out)
            return
        }
        callGather(
            input: inner, weight: weight, scales: scales, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: blockSize, global: global,
            sortedIndices: levers.skipSort,
            on: cmd, device: device, into: out)
    }
}
