// Copyright 2026 Tom Turney (@TheTom)
//
// w4/mpp-moegather: the NVFP4 MoE gather-GEMM's own F-85 pivot, mirroring
// `LagunaMppGemm.swift`'s dense-GEMM pivot (butter_dense_gemm_mpp_bf16) but
// for the quantized, ragged-expert-run prefill gather path -- the last big
// prefill chunk (~28% of the MoE-layer wall) still dispatching through the
// OLDER MPP-staging kernels (`iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_*` /
// `butter_nvfp4_moe_gather_bm16_mpp_xrows_*`, see `OpsNVFP4.swift`'s
// `nvfp4MoeGatherBm16`/`Pair`) by default.
//
// WHAT THIS ADDS: `butter_nvfp4_moe_gather_eg_mpp_bf16` (the .metal file
// next to this one) fuses the NVFP4 dequant directly into an MPP
// `matmul2d` tensor-ops gather-GEMM -- same BM=BN=64/BK=32/TPG=128
// cooperative-tensor geometry as `LagunaMppGemm`'s dense kernel and the
// existing `butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16` (NAX) kernel already
// wired in via `OpsNVFP4.nvfp4MoeGatherPrefersNax` -- but with EG-style
// ("Expert-Grouped") ownership: every threadgroup owns exactly ONE
// expert's rows for exactly one sub-tile of that expert's run, never a
// flat M-tile that can straddle two or more experts' runs the way BM16/
// NAX/NAX64 all can (and handle via intra-tile run scanning or masking
// instead of true ownership repartition).
//
// WHY THIS WASN'T DONE BEFORE: a prior session
// (`Nvfp4GatherEG.swift`, `BUTTER_NVFP4_EG1`) scoped exactly this idea --
// true 1-expert/threadgroup ownership -- and explicitly declined to ship
// the GEMM repartition, reasoning that a genuinely dynamic per-expert grid
// needs GPU-side INDIRECT dispatch (since `sortedExpertsHost` is empty on
// the default no-host-readback DEVPLAN2 path this targets, so the actual
// per-expert row counts are never available on the Swift side to size a
// static dispatch), and that shipping unverified indirect-dispatch
// coop-tensor kernel code without the ability to build/test carried too
// much silent-correctness-regression risk for that session's constraints.
// That session also documented three prior dedicated attempts (T33-T35 in
// benchmarks/laguna-xs/README.md) at the closely related problem via a
// from-scratch STEEL (simdgroup, not MPP tensor-ops) gather-GEMM family
// (`Nvfp4GatherSteel.swift`), which got kernel-level near-parity but never
// cleared the whole-run win bar (T35: -6.5%), attributed to per-expert
// run-length sensitivity and concurrency overhead specific to that
// rewrite.
//
// HOW THIS AVOIDS INDIRECT DISPATCH: instead of a data-dependent grid
// size, this dispatches a FIXED, over-provisioned grid
// (`maxTiles(mTotal:nExperts:)` below) sized so it can never run out of
// tile slots for any real `(mTotal, nExperts)` pair, and fills the unused
// tail with sentinel "padding" tiles the main kernel exits from
// immediately (see the .metal file's `tile_expert == 0xFFFFFFFF` check).
// A small device-side prepass kernel (`butter_nvfp4_moe_gather_eg_tileplan_u32`
// below, NOT MPP/coop-tensor -- a single-thread sequential scan, cheap
// relative to the GEMM it feeds) walks `Nvfp4GatherEG.expertOffsets`'
// per-expert `[start, count)` row ranges (already a real, tested utility
// from that prior session, just never wired to a consumer) and emits one
// `(expert, rowStart, rowCount<=64)` triple per output tile, ascending by
// expert. No host readback anywhere in this path: `mTotal` and `nExperts`
// are ordinary Swift `Int`s the caller already has (weight shape / sort
// plan output length), never derived from GPU-computed per-expert counts.
//
// MAX-TILES BOUND: `sum_e ceil(count_e / bm) <= ceil(mTotal / bm) +
// nExperts` -- each expert can "waste" at most `bm-1` extra rows in its
// own final partial tile versus the flat count, and there are at most
// `nExperts` such partial tiles, so `maxTiles` below is a safe (if
// occasionally loose) static upper bound the prepass kernel never
// overruns; see its own bounds check as defense in depth.
//
// T23 STAGE FOLD: mandatory and unconditional in the .metal kernel (same
// `kEgGatherStageBias`/`kEgGatherStageComp`, 2^-14/2^28, as every other MPP
// gather kernel in this family) -- SwiGLU-amplified activations reach
// magnitude 40-800 in real Laguna late-layer down_proj rows; see that
// file's header and `butter_nvfp4_gather_qmm_rhs_nax_mpp_bf16.metal`'s for
// the full derivation. Not re-derived here.
//
// SCOPE: bf16 only (matches `LagunaMppGemm`'s dense kernel and the
// existing NAX bf16 variant -- f32/f16 gather already has multiple
// existing MPP fallbacks in `OpsNVFP4.swift`, not worth doubling this
// kernel's surface before the bf16 path is even benched).
//
// Escape hatch: BUTTER_NVFP4_MPP_GATHER=0 restores the existing
// `nvfp4MoeGatherBm16`/`Pair` dispatch chain (NAX/NAX64/steel/BM16, in
// that priority order) unconditionally. Default ON after the T44
// re-validation (+6.9% prefill @512, byte-coherent) cleared the flip bar.
//
// w4/mpp-moegather WIRING FIX: `callGather`/`callGatherPair` used to
// trust that `indices` (and `input`, in the same row order) arrived
// already ascending-sorted by expert with contiguous per-expert runs --
// true for the direct-call kernel tests in `LagunaMppGatherTests.swift`
// (which construct sorted synthetic/real-gate runs by hand), but NOT a
// guarantee the public entry point (`Ops.nvfp4MoeGatherBm16`/`Pair` in
// `OpsNVFP4.swift`) actually offers: `MoELayer`'s own token-order
// fallback (`expIds = indicesTok`, taken whenever the sort plan is
// skipped) hands that entry point arbitrary per-row expert ids, same as
// every sibling gather kernel (BM16/NAX/NAX64/steel) already tolerates
// via intra-tile straddle scanning instead of true 1-expert ownership.
// Feeding unsorted `indices` straight into `Nvfp4GatherEG.expertOffsets`
// (a binary search that assumes ascending order) silently produced
// garbage (~cos 0.14) the one time a test exercised that public entry
// point without pre-sorting. Fix: `callGather`/`callGatherPair` now sort
// locally (`sortRowsByExpert`, below -- `Ops.moeSortPlan` with `k=1` so
// "token" and "row" coincide), gather the activation rows into sorted
// order, dispatch the EG kernel there, then scatter the result back to
// the caller's original row order. Net effect: this function's own ABI
// now matches its siblings' (arbitrary row order in, same row order
// out) instead of leaking the tile plan's internal sortedness
// requirement to callers. For callers that already pass pre-sorted
// input (the direct kernel tests, and `MoELayer`'s real sorted-plan
// path), the extra sort/gather/scatter degenerates to an identity
// permutation -- correctness-preserving, not free, but this kernel is
// still opt-in/unbenched so correctness comes first.

import Foundation
import Metal
import IronSwift

enum LagunaNvfp4MppGather {
    static let kernelName = "butter_nvfp4_moe_gather_eg_mpp_bf16"
    static let bm = 64
    static let bn = 64

    /// T44 re-validation (w4/mpp-moegather wiring fix, commit
    /// `1584e53a`): `callGather`/`callGatherPair` now locally sort by
    /// expert before dispatch (see file header), fixing the ~cos 0.14
    /// public-entry-point corruption the pre-fix ABI mismatch caused.
    /// 8/8 `LagunaMppGatherTests` pass (was 7/8), coherence is
    /// byte-identical to baseline on a 512-tok+20 continuation, and the
    /// same-window prefill win is ~6.9% (>= the 3% flip bar). Flipped to
    /// default ON, escape-to-disable, matching `LagunaMppGemm.enabled`'s
    /// convention: `BUTTER_NVFP4_MPP_GATHER=0` restores the
    /// NAX/NAX64/steel/BM16 dispatch chain unconditionally.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_NVFP4_MPP_GATHER"] != "0"
    }

    /// Apple GPU Family 9+ / macOS 26+ -- same gate as every other MPP
    /// `matmul2d` kernel in this repo (`LagunaMppGemm.capable`,
    /// `Ops.mppAutoCapable`'s own header).
    static var capable: Bool { Ops.mppAutoCapable }

    /// Shape/dtype eligibility, independent of `enabled`/`capable`.
    /// `n_out % 64 == 0` (BN=64 tiling, no N mask in the kernel),
    /// `k_in % 32 == 0` (BK=32 staging), `block_size == 16` (Poolside
    /// Laguna's only NVFP4 layout; the per-pack scale lookup assumes 2
    /// packs/group), dtype bf16 (this kernel's only staged precision).
    static func shapeEligible(dtype: DType, nOut: Int, kIn: Int, blockSize: Int) -> Bool {
        dtype == .bf16 && nOut % 64 == 0 && kIn % 32 == 0 && blockSize == 16
    }

    /// Full eligibility: shape + capability + opt-in flag.
    static func eligible(dtype: DType, nOut: Int, kIn: Int, blockSize: Int) -> Bool {
        enabled && capable && shapeEligible(dtype: dtype, nOut: nOut, kIn: kIn, blockSize: blockSize)
    }

    /// Safe static upper bound on the number of output M-tiles the tile
    /// plan can ever need -- see file header "MAX-TILES BOUND". Exposed
    /// so tests can size host-side comparison buffers identically to the
    /// real dispatch.
    static func maxTiles(mTotal: Int, nExperts: Int) -> Int {
        (mTotal + bm - 1) / bm + nExperts
    }

    // MARK: - Tile-plan prepass (device-side, no host readback, no indirect dispatch)

    nonisolated(unsafe) private static var tilePlanPSO: MTLComputePipelineState?
    private static let tilePlanLock = NSLock()

    // Single-thread sequential scan over `offsets` (nExperts+1 slots,
    // ascending prefix from `Nvfp4GatherEG.expertOffsets`). Cheap: total
    // work is O(nExperts + tilesUsed), both small relative to the GEMM
    // this feeds, and correctness is trivial to verify by inspection
    // (no parallel prefix-sum, no data race) -- deliberately not
    // optimized for throughput, see the Swift header's rationale for why
    // a sequential prepass was chosen over a data-dependent indirect
    // dispatch.
    private static let tilePlanSrc = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void butter_nvfp4_moe_gather_eg_tileplan_u32(
        const device uint *offsets [[buffer(0)]],        // n_experts+1, ascending prefix
        device uint *tile_expert [[buffer(1)]],           // max_tiles
        device uint *tile_row_start [[buffer(2)]],         // max_tiles
        device uint *tile_row_count [[buffer(3)]],         // max_tiles
        constant uint &n_experts [[buffer(4)]],
        constant uint &bm [[buffer(5)]],
        constant uint &max_tiles [[buffer(6)]]
    ) {
        uint cursor = 0u;
        for (uint e = 0u; e < n_experts; e++) {
            uint start = offsets[e];
            uint end = offsets[e + 1u];
            uint count = end - start;
            uint n_tiles = (count + bm - 1u) / bm;
            for (uint t = 0u; t < n_tiles; t++) {
                if (cursor >= max_tiles) return;  // defense in depth; maxTiles() bounds this from the host side
                uint row_start = start + t * bm;
                uint row_count = min(bm, end - row_start);
                tile_expert[cursor] = e;
                tile_row_start[cursor] = row_start;
                tile_row_count[cursor] = row_count;
                cursor++;
            }
        }
        for (uint i = cursor; i < max_tiles; i++) {
            tile_expert[i] = 0xFFFFFFFFu;
            tile_row_start[i] = 0u;
            tile_row_count[i] = 0u;
        }
    }
    """

    private static func tilePlanPipeline(device: MTLDevice) -> MTLComputePipelineState {
        tilePlanLock.lock()
        defer { tilePlanLock.unlock() }
        if let p = tilePlanPSO { return p }
        let lib = try! device.makeLibrary(source: tilePlanSrc, options: MTLCompileOptions())
        let fn = lib.makeFunction(name: "butter_nvfp4_moe_gather_eg_tileplan_u32")!
        let p = try! device.makeComputePipelineState(function: fn)
        tilePlanPSO = p
        return p
    }

    struct TilePlan {
        let tileExpert: Tensor
        let tileRowStart: Tensor
        let tileRowCount: Tensor
        let maxTiles: Int
    }

    /// Builds the per-output-M-tile ownership plan for `indices`
    /// (ascending sorted-expert-id-per-row -- by the time `callGather`/
    /// `callGatherPair` call this, `indices` is always
    /// `sortRowsByExpert(...).sortedExperts`, guaranteed ascending
    /// regardless of what order the public dispatch entry point
    /// received, see the file header's "WIRING FIX"). Two small device
    /// dispatches
    /// (`Nvfp4GatherEG.expertOffsets` then the tileplan kernel above), no
    /// host readback, no indirect dispatch. Scratch buffers are grow-only
    /// (`Device.intermediateScratch`) and tagged by `nExperts` (fixed per
    /// model, so this reuses the same slab across every layer/leg/call).
    static func buildTilePlan(
        indices: Tensor, mTotal: Int, nExperts: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> TilePlan {
        let tiles = maxTiles(mTotal: mTotal, nExperts: nExperts)

        let offsetsBuf = device.intermediateScratch(
            tag: "nvfp4_eg_offsets_\(nExperts)", minBytes: (nExperts + 1) * 4)
        let offsets = Tensor(buffer: offsetsBuf, offset: 0, shape: [nExperts + 1], dtype: .u32)
        Nvfp4GatherEG.expertOffsets(
            sortedExperts: indices, mTotal: mTotal, nExperts: nExperts,
            into: offsets, on: cmd, device: device)

        let expertBuf = device.intermediateScratch(
            tag: "nvfp4_eg_tile_expert_\(nExperts)", minBytes: tiles * 4)
        let rowStartBuf = device.intermediateScratch(
            tag: "nvfp4_eg_tile_row_start_\(nExperts)", minBytes: tiles * 4)
        let rowCountBuf = device.intermediateScratch(
            tag: "nvfp4_eg_tile_row_count_\(nExperts)", minBytes: tiles * 4)
        let tileExpert = Tensor(buffer: expertBuf, offset: 0, shape: [tiles], dtype: .u32)
        let tileRowStart = Tensor(buffer: rowStartBuf, offset: 0, shape: [tiles], dtype: .u32)
        let tileRowCount = Tensor(buffer: rowCountBuf, offset: 0, shape: [tiles], dtype: .u32)

        let pso = tilePlanPipeline(device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(pso)
            enc.setBuffer(offsets.buffer, offset: offsets.offset, index: 0)
            enc.setBuffer(tileExpert.buffer, offset: tileExpert.offset, index: 1)
            enc.setBuffer(tileRowStart.buffer, offset: tileRowStart.offset, index: 2)
            enc.setBuffer(tileRowCount.buffer, offset: tileRowCount.offset, index: 3)
            var nExpertsU = UInt32(nExperts)
            var bmU = UInt32(bm)
            var tilesU = UInt32(tiles)
            enc.setBytes(&nExpertsU, length: 4, index: 4)
            enc.setBytes(&bmU, length: 4, index: 5)
            enc.setBytes(&tilesU, length: 4, index: 6)
            // Single-thread scan: 1x1x1 grid, 1x1x1 threadgroup.
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc.endEncoding()
        }

        return TilePlan(
            tileExpert: tileExpert, tileRowStart: tileRowStart, tileRowCount: tileRowCount,
            maxTiles: tiles)
    }

    // MARK: - Row-order ABI fix (arbitrary row order in, same row order out)

    /// Output of `sortRowsByExpert`: same field-for-field contract as
    /// `Ops.moeSortPlan`'s own `sortedExperts`/`sourceTokens`/`invPerm`
    /// (drop-in identical semantics, see that function's own doc comment
    /// in `Ops.swift`) -- `sourceRows[sortedPos]` is the ORIGINAL row
    /// that landed at `sortedPos` (forward-gather key to pull `input`
    /// into sorted order); `invPerm[originalRow]` is its inverse, the
    /// sorted position that row landed at (forward-gather key to scatter
    /// a sorted-order result back to `originalRow`, exactly the
    /// `out[r][d] = expertOutputs[invPerm[r]][d]` identity
    /// `Ops.moeUnpermute`/`LagunaSortedCombine` already rely on for the
    /// k>1 case -- here k=1, one pick per row, so it degenerates to a
    /// plain gather).
    private struct RowSort {
        let sortedExperts: Tensor
        let sourceRows: Tensor
        let invPerm: Tensor
    }

    /// See the file header's "WIRING FIX". `indices` is `[mTotal]` u32
    /// expert ids in ARBITRARY row order (not required to be pre-sorted
    /// by expert). Sorts locally via `Ops.moeSortPlan` called with
    /// `k=1` -- with one "expert pick" per "token", `moeSortPlan`'s
    /// `mTotal` "(token,k)" rows degenerate exactly to this function's
    /// `mTotal` plain rows, so `sourceTokens`/`invPerm` degenerate to
    /// per-row (not per-token-times-k) source-row / inverse-permutation
    /// indices -- precisely what a flat per-row sort needs, reusing the
    /// same device-side (no host readback) kernel `MoELayer`'s own
    /// production sorted-plan path already runs, instead of a new
    /// bespoke sort.
    private static func sortRowsByExpert(
        indices: Tensor, mTotal: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> RowSort {
        let seBuf = device.intermediateScratch(tag: "nvfp4_eg_anyorder_se", minBytes: mTotal * 4)
        let stBuf = device.intermediateScratch(tag: "nvfp4_eg_anyorder_st", minBytes: mTotal * 4)
        let ipBuf = device.intermediateScratch(tag: "nvfp4_eg_anyorder_ip", minBytes: mTotal * 4)
        let se = Tensor(buffer: seBuf, offset: 0, shape: [mTotal], dtype: .u32)
        let st = Tensor(buffer: stBuf, offset: 0, shape: [mTotal], dtype: .u32)
        let ip = Tensor(buffer: ipBuf, offset: 0, shape: [mTotal], dtype: .u32)
        Ops.moeSortPlan(
            topkIds: indices, sortedExperts: se, sourceTokens: st, invPerm: ip,
            mTotal: mTotal, k: 1, on: cmd)
        return RowSort(sortedExperts: se, sourceRows: st, invPerm: ip)
    }

    // MARK: - Main gather-GEMM dispatch

    /// `out[m, n] = dequant(W_nvfp4[expert(m)])[n, :] . input[m, :]`, fused
    /// dequant + MPP tensor-ops GEMM, EG-style ownership (see file
    /// header). `weight` `[nExperts, nOut, kIn/8] u32`, `scales`
    /// `[nExperts, nOut, kIn/blockSize] u8` (E4M3), `indices` expert id
    /// per row `[mTotal] u32` in ARBITRARY row order (== the contract
    /// every other `nvfp4MoeGatherBm16` dispatch member already honors --
    /// need not be pre-sorted by expert), `input`/`indices` sharing that
    /// same row order `[mTotal, kIn]` bf16. `out` is written back in
    /// that identical caller row order (see the file header's "WIRING
    /// FIX" -- sorted internally, scattered back before returning).
    /// Caller must have already checked `eligible`.
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
        precondition(nOut % 64 == 0, "LagunaNvfp4MppGather: nOut \(nOut) must be a multiple of 64")
        precondition(kIn % 32 == 0, "LagunaNvfp4MppGather: kIn \(kIn) must be a multiple of 32")
        precondition(blockSize == 16, "LagunaNvfp4MppGather: only blockSize=16 is supported")
        let nExperts = weight.shape[0]

        let sort = sortRowsByExpert(indices: indices, mTotal: mTotal, on: cmd, device: device)

        let sortedInBuf = device.intermediateScratch(
            tag: "nvfp4_eg_anyorder_in", minBytes: mTotal * kIn * 2)
        let sortedIn = Tensor(buffer: sortedInBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)
        _ = Ops.gather(table: input, tokenIds: sort.sourceRows, on: cmd, into: sortedIn)

        let sortedOutBuf = device.intermediateScratch(
            tag: "nvfp4_eg_anyorder_out", minBytes: mTotal * nOut * 2)
        let sortedOut = Tensor(buffer: sortedOutBuf, offset: 0, shape: [mTotal, nOut], dtype: .bf16)

        let plan = buildTilePlan(
            indices: sort.sortedExperts, mTotal: mTotal, nExperts: nExperts, on: cmd, device: device)

        let pso = PSOCache.shared.pipelineState(for: kernelName)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            fatalError("LagunaNvfp4MppGather.callGather: could not make compute encoder")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(sortedIn.buffer, offset: sortedIn.offset, index: 0)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
        enc.setBuffer(plan.tileExpert.buffer, offset: plan.tileExpert.offset, index: 3)
        enc.setBuffer(plan.tileRowStart.buffer, offset: plan.tileRowStart.offset, index: 4)
        enc.setBuffer(plan.tileRowCount.buffer, offset: plan.tileRowCount.offset, index: 5)
        enc.setBuffer(sortedOut.buffer, offset: sortedOut.offset, index: 6)
        var nOutU = UInt32(nOut)
        var kInU = UInt32(kIn)
        var blockU = UInt32(blockSize)
        var globalV = global
        enc.setBytes(&nOutU, length: 4, index: 7)
        enc.setBytes(&kInU, length: 4, index: 8)
        enc.setBytes(&blockU, length: 4, index: 9)
        enc.setBytes(&globalV, length: 4, index: 10)
        // coop_tile kernels MUST use dispatchThreadgroups (grid in TG
        // units) -- same convention/trap as every other MPP matmul2d
        // kernel in this repo (see OpsNVFP4.swift's own comment on this).
        let grid = MTLSize(width: nOut / bn, height: plan.maxTiles, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        _ = Ops.gather(table: sortedOut, tokenIds: sort.invPerm, on: cmd, into: out)
    }

    /// Concurrent gate+up pair sharing one tile plan (built once, reused
    /// for both dispatches -- gate and up share the same `indices`/
    /// `input`/`mTotal`/`nExperts`, so the sort + tile plan are identical
    /// for both). Same row-order ABI as `callGather` (see its doc + the
    /// file header's "WIRING FIX"): `indices`/`input` in arbitrary row
    /// order, `out0`/`out1` written back in that same order.
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
        precondition(nOut % 64 == 0 && kIn % 32 == 0 && blockSize == 16)
        let nExperts = weight0.shape[0]

        let sort = sortRowsByExpert(indices: indices, mTotal: mTotal, on: cmd, device: device)

        let sortedInBuf = device.intermediateScratch(
            tag: "nvfp4_eg_anyorder_in", minBytes: mTotal * kIn * 2)
        let sortedIn = Tensor(buffer: sortedInBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)
        _ = Ops.gather(table: input, tokenIds: sort.sourceRows, on: cmd, into: sortedIn)

        // Gate/up need separate sorted-order output buffers (different
        // weights) before each is scattered back into its own out0/out1.
        let sortedOut0Buf = device.intermediateScratch(
            tag: "nvfp4_eg_anyorder_out0", minBytes: mTotal * nOut * 2)
        let sortedOut1Buf = device.intermediateScratch(
            tag: "nvfp4_eg_anyorder_out1", minBytes: mTotal * nOut * 2)
        let sortedOut0 = Tensor(buffer: sortedOut0Buf, offset: 0, shape: [mTotal, nOut], dtype: .bf16)
        let sortedOut1 = Tensor(buffer: sortedOut1Buf, offset: 0, shape: [mTotal, nOut], dtype: .bf16)

        let plan = buildTilePlan(
            indices: sort.sortedExperts, mTotal: mTotal, nExperts: nExperts, on: cmd, device: device)

        let pso = PSOCache.shared.pipelineState(for: kernelName)
        guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) else {
            fatalError("LagunaNvfp4MppGather.callGatherPair: could not make compute encoder")
        }
        enc.setComputePipelineState(pso)
        var nOutU = UInt32(nOut)
        var kInU = UInt32(kIn)
        var blockU = UInt32(blockSize)
        var globalV = global
        let grid = MTLSize(width: nOut / bn, height: plan.maxTiles, depth: 1)
        let tg = MTLSize(width: 128, height: 1, depth: 1)

        enc.setBuffer(sortedIn.buffer, offset: sortedIn.offset, index: 0)
        enc.setBuffer(weight0.buffer, offset: weight0.offset, index: 1)
        enc.setBuffer(scales0.buffer, offset: scales0.offset, index: 2)
        enc.setBuffer(plan.tileExpert.buffer, offset: plan.tileExpert.offset, index: 3)
        enc.setBuffer(plan.tileRowStart.buffer, offset: plan.tileRowStart.offset, index: 4)
        enc.setBuffer(plan.tileRowCount.buffer, offset: plan.tileRowCount.offset, index: 5)
        enc.setBuffer(sortedOut0.buffer, offset: sortedOut0.offset, index: 6)
        enc.setBytes(&nOutU, length: 4, index: 7)
        enc.setBytes(&kInU, length: 4, index: 8)
        enc.setBytes(&blockU, length: 4, index: 9)
        enc.setBytes(&globalV, length: 4, index: 10)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)

        enc.setBuffer(weight1.buffer, offset: weight1.offset, index: 1)
        enc.setBuffer(scales1.buffer, offset: scales1.offset, index: 2)
        enc.setBuffer(sortedOut1.buffer, offset: sortedOut1.offset, index: 6)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        _ = Ops.gather(table: sortedOut0, tokenIds: sort.invPerm, on: cmd, into: out0)
        _ = Ops.gather(table: sortedOut1, tokenIds: sort.invPerm, on: cmd, into: out1)
    }
}
