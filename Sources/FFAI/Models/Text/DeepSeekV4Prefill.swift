// Copyright 2026 Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//
// Batched prefill for DSv4-Flash: processes a chunk of N tokens in one
// pass, so each weight is read once and reused across all N tokens
// (vs forwardAllLayers' per-token re-read at ~31 tok/s). Uses the four
// validated prefill kernels: ffai_gemm_q8 (dense/attn projections),
// ffai_moe_gather_bgemm_iq2xxs/q2k_mpp (MoE), ffai_sdpa_prefill_d512_sink
// (causal attention). Returns the LAST token's logits (next-token).
//
// NOTE (validation parity): the sequential forwardAllLayers leaves
// state.position = 0 for every token, so RoPE (angle = pos*theta) is the
// identity — prefill therefore skips rope and still matches. Fixing the
// position-advance + per-token rope pairs with CSA/HCA for real 126k.

import Foundation
import Metal
import MetalTileSwift

struct PrefillError: Error, CustomStringConvertible {
    let message: String
    var description: String { message }
}

/// System-wide free memory as a percentage (free+inactive vs hw.memsize), or
/// nil if the mach query fails. Used by the prefill freeze guard.
func ffaiSystemFreePercent() -> Double? {
    var total: UInt64 = 0
    var sz = MemoryLayout<UInt64>.size
    if sysctlbyname("hw.memsize", &total, &sz, nil, 0) != 0 || total == 0 { return nil }
    var stats = vm_statistics64_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &stats) { ptr -> kern_return_t in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
            host_statistics64(mach_host_self(), HOST_VM_INFO64, intPtr, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return nil }
    var pageSize: UInt64 = 16384
    var psz = MemoryLayout<UInt64>.size
    _ = sysctlbyname("hw.pagesize", &pageSize, &psz, nil, 0)
    let freeBytes = (UInt64(stats.free_count) + UInt64(stats.inactive_count)) * pageSize
    return Double(freeBytes) / Double(total) * 100.0
}

extension DeepSeekV4Model {
    /// One-time guard: have we pre-warmed the expert tensors into page cache?
    nonisolated(unsafe) static var dsv4Prewarmed = false

    /// Batched prefill over `tokens`; returns the last token's logits.
    /// Single chunk (N ≤ a few hundred); KV cache is the chunk's own K/V.
    public func forwardPrefillChunk(tokens: [Int], state decodeState: DecodeState? = nil) throws -> Tensor {
        let cfg = textConfig
        let dt = activationDtype
        let hidden = cfg.hidden
        let headDim = cfg.headDim
        let nHeads = cfg.nHeads
        let intermediate = cfg.moeIntermediate
        let topK = cfg.nExpertsPerToken
        let scaling = cfg.routerScalingFactor
        let window = cfg.slidingWindow
        let N = tokens.count
        let nExperts = 256
        let scale = 1.0 / Float(headDim).squareRoot()

        // Per-layer scratch transients scale ~linearly with N (xPerm [N*topK,
        // hidden], gate/up/inner [M,intermediate], attn [N,*]). Measured ~1.1
        // MB/token; size the slab to fit one layer with headroom (2 MB/token,
        // 256 MB floor) so large chunks don't overflow the 256 MB default.
        device.ensureScratchSlab(max(256 << 20, N * (2 << 20)))
        // ── Seed hcState [N, 4, hidden] from per-token embeddings ──
        let embW = tokenEmbd.asGgufMatmulWeight()  // [vocab, hidden]
        // 2D [N*4, hidden]: row (t*4+c) = token t, mHC channel c. slicedRows
        // slices dim0, so a 3D shape here would index the token dim (overflow).
        var hcState = Tensor.empty(shape: [N * 4, hidden], dtype: dt, device: device)
        let seedCmd = device.makeCommandBuffer()
        for t in 0 ..< N {
            let row = embW.slicedRows(start: tokens[t], count: 1).reshaped(to: [hidden])
            for c in 0 ..< 4 {
                let dst = hcState.slicedRows(start: t * 4 + c, count: 1).reshaped(to: [hidden])
                Ops.copy(row, into: dst, on: seedCmd)
            }
        }
        seedCmd.commit(); seedCmd.waitUntilCompleted()

        // hcState is the only state that crosses layer boundaries. It MUST
        // be a persistent (non-scratch) buffer: prefillLayer's newH is
        // allocated inside withScratch, so it would be invalidated when the
        // slab resets at scope exit. Copy newH into the persistent hcState
        // before the scope ends (mirrors forwardAllLayers' hcStatePersistent).
        //
        // COLD-I/O READAHEAD: the per-layer expert gather runs at ~11 GB/s on
        // the first chunk (latency-bound on cold SSD page faults — the #2 cost).
        // A background madvise(WILLNEED) of the NEXT layer's expert tensors,
        // fired while the GPU computes THIS layer, overlaps that disk I/O with
        // compute. It only HINTS readahead — copies nothing — so it does NOT
        // contend for the unified-memory bandwidth the way a background memcpy
        // does (that regressed 2×). Fire-and-forget (advisory): no join, no
        // buffers; if readahead isn't done by the gather, it just faults
        // normally (no worse than baseline).
        let raQueue = DispatchQueue(label: "ffai.prefill.readahead", qos: .utility)
        // PREWARM: touch-read ALL expert tensors into the page cache ONCE,
        // before the first prefill. The 80GB model fits in 128GB cache, but
        // mmap is lazy — macOS hasn't faulted it, so the per-layer gather hits
        // cold/evicted pages (disk-bound 7-11 GB/s) → ~232 t/s. Pre-faulting
        // the whole model (reclaimable CACHE, not wired → no freeze, the guard
        // sees inactive pages as available) lifts warm prefill to ~320+ (94%+
        // of parity). One-time ~7s (the cold weight read paid upfront, like
        // model load) — subsequent chunks/prefills stay warm.
        if !Self.dsv4Prewarmed {
            Self.dsv4Prewarmed = true
            let names = (0 ..< cfg.nLayers).flatMap { li in
                ["blk.\(li).ffn_gate_exps.weight", "blk.\(li).ffn_up_exps.weight", "blk.\(li).ffn_down_exps.weight"]
            }
            DispatchQueue.concurrentPerform(iterations: names.count) { i in bundle.prefetchTensor(named: names[i]) }
        }
        for layerIdx in 0 ..< cfg.nLayers {
            if layerIdx + 1 < cfg.nLayers {
                let nx = layerIdx + 1
                raQueue.async { [weak self] in
                    guard let self else { return }
                    self.bundle.prefetchTensor(named: "blk.\(nx).ffn_gate_exps.weight")
                    self.bundle.prefetchTensor(named: "blk.\(nx).ffn_up_exps.weight")
                    self.bundle.prefetchTensor(named: "blk.\(nx).ffn_down_exps.weight")
                }
            }
            // FREEZE GUARD: abort cleanly if system free memory drops below a
            // floor (default 12%) before a layer's allocations. The M5 Max
            // freezes near ~8-10% free; bailing with a throw lets the OS
            // reclaim instead of the app-quit-monitor freeze. Override the
            // floor via FFAI_MEM_FLOOR_PCT, disable with =0.
            if let freePct = ffaiSystemFreePercent() {
                let floor = Double(ProcessInfo.processInfo.environment["FFAI_MEM_FLOOR_PCT"].flatMap { Int($0) } ?? 12)
                if floor > 0 && freePct < floor {
                    throw PrefillError(
                        message:
                            "prefill aborted at layer \(layerIdx): system free memory \(String(format: "%.0f", freePct))% < floor \(Int(floor))% (freeze guard). Reduce chunk size / residency."
                    )
                }
            }
            let layer = try self.layer(layerIdx)
            // autoreleasepool is CRITICAL: prefillLayer allocates a FRESH
            // per-layer expert pool (~1.5 GB of MTLBuffers via makeBuffer).
            // Metal buffers are autoreleased ObjC objects — without draining
            // the pool each layer they'd accumulate (43 × 1.5 GB ≈ 64 GB) and
            // freeze the machine. Drain per layer → peak stays ~one layer.
            try autoreleasepool {
                try device.withScratch {
                    let newH = try prefillLayer(
                        layer: layer, hcState: hcState, N: N, hidden: hidden, headDim: headDim,
                        nHeads: nHeads, intermediate: intermediate, topK: topK, nExperts: nExperts,
                        scaling: scaling, window: window, scale: scale, dt: dt, tokens: tokens,
                        decodeState: decodeState)
                    let ccmd = device.makeCommandBuffer()
                    Ops.copy(newH.reshaped(to: [N * 4, hidden]), into: hcState, on: ccmd)
                    ccmd.commit(); ccmd.waitUntilCompleted()
                }
            }
            if [0, 1, 2, 5, 10, 20, 24, 28, 32, 36, 40, 42].contains(layerIdx) {
                if N > 1 {
                }
            }
        }

        // ── Tail: last token's output head + lmhead ──
        // dsv4MhcExpand returns [N,4,hidden]; flatten to [N*4,hidden] so
        // slicedRows indexes (token*4+channel) rows, not the token dim.
        let hcFlat = hcState.reshaped(to: [N * 4, hidden])
        let lastH = hcFlat.slicedRows(start: (N - 1) * 4, count: 4).reshaped(to: [4, hidden])
        if let decodeState {
            let scmd = device.makeCommandBuffer()
            Ops.copy(lastH, into: decodeState.hcState, on: scmd)
            scmd.commit(); scmd.waitUntilCompleted()
            decodeState.position = N
            decodeState.currentToken = tokens[N - 1]
        }
        let cmd = device.makeCommandBuffer()
        let flatH = lastH.reshaped(to: [4 * hidden])
        // mHC rms-norm before the output_hc_fn mix (same fix as decode/mhcMix).
        let pre4 = mhcMix(flatH: flatH, fnWeight: outputHcFn.asGgufMatmulWeight(), on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        let pre4Host = pre4.toArray(as: Float.self)
        let scaleHost = outputHcScale.toArray(as: Float.self)
        let baseHost = outputHcBase.toArray(as: Float.self)
        var preFinal = [Float](repeating: 0, count: 4)
        for c in 0 ..< 4 {
            let z = pre4Host[c] * scaleHost[0] + baseHost[c]
            preFinal[c] = 1.0 / (1.0 + Foundation.exp(-z)) + cfg.hcEpsilon
        }
        let preTensor = Tensor.empty(shape: [4], dtype: .f32, device: device)
        preTensor.copyIn(from: preFinal)
        let cmd2 = device.makeCommandBuffer()
        let x = Ops.dsv4MhcCollapse(
            state: lastH, pre: preTensor, hiddenDim: hidden, nHc: 4, nTokens: 1,
            outDtype: dt, on: cmd2
        ).reshaped(to: [hidden])
        let xNorm = Ops.rmsNorm(x, weight: outputNorm, eps: cfg.rmsNormEps, on: cmd2)
        let logits: Tensor
        if let lmQ8 = try? bundle.residentQ8("output.weight", device: device) {
            logits = Tensor.empty(shape: [lmQ8.mOut], dtype: dt, device: device)
            Ops.gemvQ8(q8: lmQ8, x: xNorm, on: cmd2, into: logits)
        } else {
            logits = Ops.gemv(weight: outputHead.asGgufMatmulWeight(), input: xNorm, on: cmd2)
        }
        cmd2.commit(); cmd2.waitUntilCompleted()
        return logits
    }

    /// One prefill layer over N tokens. Returns the new hcState [N,4,hidden].
    private func prefillLayer(
        layer: DeepSeekV4Layer, hcState: Tensor, N: Int, hidden: Int, headDim: Int,
        nHeads: Int, intermediate: Int, topK: Int, nExperts: Int, scaling: Float,
        window: Int, scale: Float, dt: DType, tokens: [Int], decodeState: DecodeState?
    ) throws -> Tensor {
        let li = layer.layerIndex
        func q8(_ name: String) -> ResidentQ8? { try? bundle.residentQ8(name, device: device) }
        // Q8 GEMM from a resident Q8 weight (wraps its buffers as tensors).
        func gq8(_ r: ResidentQ8, _ inp: Tensor, _ outT: Tensor, _ n: Int, _ c: MTLCommandBuffer) {
            let nb = r.mOut * r.kIn / 32
            let qsT = Tensor(buffer: r.qs, offset: 0, shape: [nb * 8], dtype: .u32)
            let dT = Tensor(buffer: r.d, offset: 0, shape: [nb], dtype: .f32)
            // Cooperative-tensor MMA Q8 GEMM (~6× the scalar path) for the
            // batched case; the scalar gemmQ8 for small n where MMA doesn't pay.
            if n >= 32 {
                Ops.gemmQ8Mpp(qs: qsT, dF32: dT, input: inp, out: outT, inDim: r.kIn, outDim: r.mOut, nRows: n, on: c)
            } else {
                Ops.gemmQ8(qs: qsT, dF32: dT, input: inp, out: outT, inDim: r.kIn, outDim: r.mOut, nRows: n, on: c)
            }
        }

        // ===== Attention sub-block (N tokens) =====
        let cmd = device.makeCommandBuffer()
        let flatH = hcState.reshaped(to: [N, 4 * hidden])
        // mHC: mix = hc_attn_fn @ rms_norm_no_weight(flat) — the RMSNorm
        // over the flattened 4-channel state per token was missing (same
        // bug as decode's mhcMix). Without it the sinkhorn split is wrong.
        let flatHNorm = Ops.rmsNormRows(
            flatH, weight: hcFlatOnes(dt), eps: textConfig.rmsNormEps, nRows: N, rowSize: 4 * hidden, on: cmd)
        let mixes = Ops.gemm(weight: layer.hcAttnFn.asGgufMatmulWeight(), input: flatHNorm, nRows: N, on: cmd)
        let (preA, postA, combA) = Ops.dsv4MhcSinkhornSplit(
            mixes: mixes, scale: layer.hcAttnScale, base: layer.hcAttnBase,
            nTokens: N, eps: textConfig.hcEpsilon, sinkhornIters: textConfig.hcSinkhornIterations, on: cmd)
        let x = Ops.dsv4MhcCollapse(
            state: hcState, pre: preA, hiddenDim: hidden, nHc: 4, nTokens: N, outDtype: dt, on: cmd)
        let xNorm = Ops.rmsNormRows(
            x, weight: layer.attnNorm, eps: textConfig.rmsNormEps, nRows: N, rowSize: hidden, on: cmd)  // [N,hidden] rows

        // Q chain (Q8 gemm).
        let qaQ8 = q8("blk.\(li).attn_q_a.weight")!
        let qA = Tensor.empty(shape: [N, qaQ8.mOut], dtype: dt)
        gq8(qaQ8, xNorm, qA, N, cmd)
        Ops.rmsNormRows(
            qA, weight: layer.attnQANorm, eps: textConfig.rmsNormEps, nRows: N, rowSize: qaQ8.mOut, on: cmd, into: qA)
        let qbQ8 = q8("blk.\(li).attn_q_b.weight")!
        let q = Tensor.empty(shape: [N, qbQ8.mOut], dtype: dt)  // [N, nHeads*headDim]
        gq8(qbQ8, qA, q, N, cmd)
        // Per-head unit RMS over N*nHeads rows. (No rope: position 0 = identity.)
        Ops.rmsNormRows(
            q, weight: Tensor.filled(1.0, shape: [headDim], dtype: dt, device: device), eps: textConfig.rmsNormEps,
            nRows: N * nHeads, rowSize: headDim, on: cmd, into: q)

        // KV (Q8 gemm) → [N, headDim] (MQA 1 kv head).
        let kvQ8 = q8("blk.\(li).attn_kv.weight")!
        let kv = Tensor.empty(shape: [N, kvQ8.mOut], dtype: dt)
        gq8(kvQ8, xNorm, kv, N, cmd)
        let kvNorm = Tensor.empty(shape: [N, headDim], dtype: dt)
        Ops.rmsNormRows(
            kv, weight: layer.attnKVANorm, eps: textConfig.rmsNormEps, nRows: N, rowSize: headDim, on: cmd, into: kvNorm
        )

        // ── Per-position partial RoPE (token t at position t). The old
        // code skipped rope ("position 0 = identity") which is only valid
        // for N=1; real prompts have tokens at 0..N-1. Compressed layers use
        // compress_rope_theta + YaRN, full layers rope_theta. ──
        let qkRopeDim = textConfig.qkRopeHeadDim
        let nNope = headDim - qkRopeDim
        let layerRatio = li < layerCompressRatios.count ? layerCompressRatios[li] : 0
        let layerTheta = layerRatio != 0 ? textConfig.compressRopeTheta : textConfig.ropeTheta
        let yp = yarnParams(ratio: layerRatio)
        // Batched per-position partial RoPE: token t at position t, ONE
        // dispatch each for q (nHeads) and kv (1 head) — was a per-token loop
        // (2N tiny dispatches/layer, a top warm-prefill cost).
        Ops.dsv4PartialRopeRows(
            qk: q, out: q, nHeads: nHeads, headDim: headDim, nNope: nNope,
            nTokens: N, basePosition: 0, thetaBase: layerTheta, inverse: false,
            freqScale: yp.freqScale, extFactor: yp.extFactor, corrLow: yp.corrLow, corrHigh: yp.corrHigh, on: cmd)
        Ops.dsv4PartialRopeRows(
            qk: kvNorm, out: kvNorm, nHeads: 1, headDim: headDim, nNope: nNope,
            nTokens: N, basePosition: 0, thetaBase: layerTheta, inverse: false,
            freqScale: yp.freqScale, extFactor: yp.extFactor, corrLow: yp.corrLow, corrHigh: yp.corrHigh, on: cmd)
        if let decodeState {
            let layerState = decodeState.layerStates[li]
            let winCount = min(N, layerState.nSWA)
            let start = N - winCount
            Ops.copy(
                kvNorm.slicedRows(start: start, count: winCount),
                into: layerState.swCache.slicedRows(start: 0, count: winCount),
                on: cmd)
            layerState.swCount = winCount
            layerState.compCount = 0
        }

        // Causal sliding-window SDPA over the chunk. q [N,nHeads,headDim], kv [N,headDim].
        let attnOut = Tensor.empty(shape: [N, nHeads * headDim], dtype: dt)
        Ops.sdpaPrefillD512Sink(
            q: q, k: kvNorm, v: kvNorm, sinkLogit: layer.attnSinks, out: attnOut,
            headDim: headDim, nQHeads: nHeads, kvStride: N, headsPerGroup: nHeads,
            window: window, kvBase: 0, scale: scale, nQuery: N, on: cmd)
        // Inverse partial RoPE on the attention output (V carries the K
        // rotation in MQA; undo it per token before the O-LoRA).
        Ops.dsv4PartialRopeRows(
            qk: attnOut, out: attnOut, nHeads: nHeads, headDim: headDim, nNope: nNope,
            nTokens: N, basePosition: 0, thetaBase: layerTheta, inverse: true,
            freqScale: yp.freqScale, extFactor: yp.extFactor, corrLow: yp.corrLow, corrHigh: yp.corrHigh, on: cmd)

        // O-LoRA: 8 groups. oLow [N, 8*oLoraRank]. (f16 path — copy each
        // group's [N,groupDim] contiguous, gemm, write.)
        // Grouped O-LoRA-A: 8 groups, each a contiguous row block of the
        // Q8 resident attn_output_a with its own [groupDim] input slice.
        // Uses the proven groupedGemvQ8 per token-row — the f16
        // asGgufMatmulWeight().slicedRows() path is WRONG (the swap is a
        // label-only view, so slicing its leading dim reads non-contiguous
        // bytes). attnOut row = [8 groups × groupDim] contiguous input.
        let oGroups = 8
        let oLoraRank = textConfig.oLoraRank  // 1024
        let oLow = Tensor.empty(shape: [N, oGroups * oLoraRank], dtype: dt)
        let oaQ8 = q8("blk.\(li).attn_output_a.weight")!
        // Batched O-LoRA-A: amortized grouped GEMM via cooperative-tensor MMA
        // for all N tokens (replaces the per-token gemv that was the #1 attn
        // hotspot). oaQ8 is [oGroups*oLoraRank, perGroupIn]; the scalar grouped
        // GEMM handles small n where MMA doesn't pay.
        let oaNb = oaQ8.mOut * oaQ8.kIn / 32
        let oaQs = Tensor(buffer: oaQ8.qs, offset: 0, shape: [oaNb * 8], dtype: .u32)
        let oaD = Tensor(buffer: oaQ8.d, offset: 0, shape: [oaNb], dtype: .f32)
        if N >= 32 {
            Ops.groupedGemmQ8Mpp(
                qs: oaQs, dF32: oaD, input: attnOut, out: oLow,
                inDim: oaQ8.kIn, outDim: oaQ8.mOut, nRows: N,
                nGroups: oGroups, rowsPerGroup: oLoraRank, on: cmd)
        } else {
            Ops.groupedGemmQ8(
                qs: oaQs, dF32: oaD, input: attnOut, out: oLow,
                inDim: oaQ8.kIn, outDim: oaQ8.mOut, nRows: N,
                nGroups: oGroups, rowsPerGroup: oLoraRank, on: cmd)
        }
        let obQ8 = q8("blk.\(li).attn_output_b.weight")!
        let attnBlock = Tensor.empty(shape: [N, hidden], dtype: dt)
        gq8(obQ8, oLow, attnBlock, N, cmd)
        let hAfterAttn = Ops.dsv4MhcExpand(
            blockOut: attnBlock, post: postA, comb: combA, residualState: hcState,
            hiddenDim: hidden, nHc: 4, nTokens: N, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        if li == 0 {
        }

        // ===== FFN sub-block (N tokens) =====
        let fcmd = device.makeCommandBuffer()
        let flatH2 = hAfterAttn.reshaped(to: [N, 4 * hidden])
        let flatH2Norm = Ops.rmsNormRows(
            flatH2, weight: hcFlatOnes(dt), eps: textConfig.rmsNormEps, nRows: N, rowSize: 4 * hidden, on: fcmd)
        let mixes2 = Ops.gemm(weight: layer.hcFfnFn.asGgufMatmulWeight(), input: flatH2Norm, nRows: N, on: fcmd)
        let (preF, postF, combF) = Ops.dsv4MhcSinkhornSplit(
            mixes: mixes2, scale: layer.hcFfnScale, base: layer.hcFfnBase,
            nTokens: N, eps: textConfig.hcEpsilon, sinkhornIters: textConfig.hcSinkhornIterations, on: fcmd)
        let x2 = Ops.dsv4MhcCollapse(
            state: hAfterAttn, pre: preF, hiddenDim: hidden, nHc: 4, nTokens: N, outDtype: dt, on: fcmd)
        let xNorm2 = Ops.rmsNormRows(
            x2, weight: layer.ffnNorm, eps: textConfig.rmsNormEps, nRows: N, rowSize: hidden, on: fcmd)  // [N,hidden]

        // Router: logits [N,256] → sqrtsoftplus → readback → per-token top-6.
        let rLogits = Ops.gemm(weight: layer.ffnGateInp.asGgufMatmulWeight(), input: xNorm2, nRows: N, on: fcmd)
        let rF32 = Tensor.empty(shape: [N, nExperts], dtype: .f32)
        Ops.castToF32(rLogits, into: rF32, on: fcmd)
        // sqrtsoftplus is elementwise (bias indexed by flat element id), so
        // tile the per-expert [256] bias across all N tokens → [N*256].
        let bias0 = layer.expProbsBias ?? Tensor.filled(0.0, shape: [nExperts], dtype: .f32, device: device)
        let biasHost0 = bias0.toArray(as: Float.self)
        var tiledBias = [Float](); tiledBias.reserveCapacity(N * nExperts)
        for _ in 0 ..< N { tiledBias.append(contentsOf: biasHost0) }
        let bias = Tensor.empty(shape: [N * nExperts], dtype: .f32, device: device)
        bias.copyIn(from: tiledBias)
        let (scU, scB) = Ops.dsv4MoeRouterSqrtsoftplus(logits: rF32, bias: bias, on: fcmd)
        fcmd.commit(); fcmd.waitUntilCompleted()
        let biasedH = scB.toArray(as: Float.self)
        let unbiasedH = scU.toArray(as: Float.self)

        // Build (token,slot) rows, then expert-sort via a permutation so we
        // can also produce invPerm (origIdx → sorted position) + per-(token,
        // slot) weights for the single-dispatch batched unpermute later.
        var rowsU: [(tok: Int, slot: Int, expert: Int, w: Float)] = []
        rowsU.reserveCapacity(N * topK)
        var wOrig = [Float](repeating: 0, count: N * topK)  // origIdx = tok*topK+slot
        // Hash-routed early layers (il < n_hash_layer=3): experts come from
        // the ffn_gate_tid2eid token→expert table, NOT router top-k. Weights
        // still = probs[selected]/sum*scaling. (Same fix as decode.)
        let tid2eidHostArr = layer.ffnGateTid2Eid.map { tid2eidHost(layerIndex: li, tensor: $0) }
        for t in 0 ..< N {
            let top: [Int]
            if let table = tid2eidHostArr {
                top = (0 ..< topK).map { Int(table[tokens[t] * topK + $0]) }
            } else {
                var idx = Array((0 ..< nExperts).map { ($0, biasedH[t * nExperts + $0]) })
                idx.sort { $0.1 > $1.1 }
                top = idx.prefix(topK).map { $0.0 }
            }
            // routed_scaling_factor is applied AFTER the sum-to-1 renorm
            // (per the DSv4 reference) — applying it before, as `unbiased*scaling`,
            // cancels in the division and drops the 1.5× entirely.
            let ws = top.map { unbiasedH[t * nExperts + $0] }
            let sum = ws.reduce(0, +)
            let nw = sum > 0 ? ws.map { ($0 / sum) * scaling } : Array(repeating: scaling / Float(topK), count: topK)
            for k in 0 ..< topK { rowsU.append((t, k, top[k], nw[k])); wOrig[t * topK + k] = nw[k] }
        }
        // Stable expert-sort (token order preserved within an expert run).
        let perm = Array(0 ..< rowsU.count).sorted {
            rowsU[$0].expert != rowsU[$1].expert ? rowsU[$0].expert < rowsU[$1].expert : $0 < $1
        }
        let rows = perm.map { (tok: rowsU[$0].tok, expert: rowsU[$0].expert, w: rowsU[$0].w) }
        let M = rows.count
        // invPerm[tok*topK+slot] = sorted row position of that (token,slot).
        var invPermArr = [UInt32](repeating: 0, count: M)
        for j in 0 ..< M { let o = rowsU[perm[j]]; invPermArr[o.tok * topK + o.slot] = UInt32(j) }
        // Ensure routed experts resident; map expert→packed slot.
        let routedExperts = Array(Set(rows.map { $0.expert }))
        let gName = "blk.\(li).ffn_gate_exps.weight"; let uName = "blk.\(li).ffn_up_exps.weight";
        let dName = "blk.\(li).ffn_down_exps.weight"
        // Prefill routes a whole chunk's tokens. The gate/up/down expert
        // weights are read via the zero-copy u16 view path (raw-gather +
        // view-bm64), which needs no repacked split pool — just a pool cap
        // sized to the routed-expert count. `cap` bounds the raw-gather pool.
        let prefillPoolCap: Int? =
            routedExperts.count > RESIDENT_POOL_CAP ? max(routedExperts.count, 1) : nil
        let cap = prefillPoolCap ?? RESIDENT_POOL_CAP

        // Permuted x and per-role packed-slot indices.
        let fcmd2 = device.makeCommandBuffer()
        // Permute-by-expert in ONE gather dispatch (was an M-row CPU copy
        // loop = M tiny GPU dispatches that scaled with N*topK). xPerm[r] =
        // xNorm2[rows[r].tok].
        let xPerm = Tensor.empty(shape: [M, hidden], dtype: dt)
        let permTok = Tensor.empty(shape: [M], dtype: .u32, device: device)
        permTok.copyIn(from: rows.map { UInt32($0.tok) })
        _ = Ops.gather(table: xNorm2, tokenIds: permTok, on: fcmd2, into: xPerm)
        // gate/up BGEMM → [M, intermediate].
        let (grid, signs) = GGUFDequant.iq2xxsTables(device: device)
        let gateP = Tensor.empty(shape: [M, intermediate], dtype: dt)
        let upP = Tensor.empty(shape: [M, intermediate], dtype: dt)
        // RAW-GATHER + u16 view-bm64: bulk-copy routed experts' raw blocks into
        // a reliable makeBuffer (cheap bulk memcpy, not the 32768-tiny-memcpy
        // deinterleave), then the amortized bm64 reads them via aligned u16 (no
        // split pool, no mmap-residency-zeros bug). SLOT-indexed (the raw pool
        // is compacted by routed expert).
        guard
            let rawG = try bundle.rawGatherBlocks(
                named: gName, expertIndices: routedExperts, nExperts: nExperts, device: device, poolCap: cap,
                reuseKey: "prefill_view_gate"),
            let rawU = try bundle.rawGatherBlocks(
                named: uName, expertIndices: routedExperts, nExperts: nExperts, device: device, poolCap: cap,
                reuseKey: "prefill_view_up")
        else {
            throw PrefillError(
                message: "prefill L\(li): gate/up expert tensors '\(gName)'/'\(uName)' not found in GGUF")
        }
        let iq2Stride = rawG.nBlocksPerExpert * 66
        let slotGIdx = Tensor.empty(shape: [M], dtype: .u32, device: device)
        slotGIdx.copyIn(from: rows.map { UInt32(rawG.slotOf[$0.expert] ?? 0) })
        let slotUIdx = Tensor.empty(shape: [M], dtype: .u32, device: device)
        slotUIdx.copyIn(from: rows.map { UInt32(rawU.slotOf[$0.expert] ?? 0) })
        Ops.moeBgemmIQ2XXSViewU16Bm64(
            x: xPerm, viewBuf: rawG.buffer, viewByteOffset: 0,
            grid: grid, signs: signs, indices: slotGIdx, out: gateP,
            mTotal: M, nOut: intermediate, kIn: hidden,
            tensorByteOff: 0, expertByteStride: iq2Stride, on: fcmd2)
        Ops.moeBgemmIQ2XXSViewU16Bm64(
            x: xPerm, viewBuf: rawU.buffer, viewByteOffset: 0,
            grid: grid, signs: signs, indices: slotUIdx, out: upP,
            mTotal: M, nOut: intermediate, kIn: hidden,
            tensorByteOff: 0, expertByteStride: iq2Stride, on: fcmd2)
        // swiglu — element-wise over the whole [M, intermediate] tile in ONE
        // dispatch (was an M-row Swift slice loop + M dispatches/layer = ~264k
        // dispatches at N=1024 × 43 layers, the per-token CPU/encode overhead
        // that made t/s DROP with N). gateP/upP/innerP are contiguous → flat.
        let innerP = Tensor.empty(shape: [M, intermediate], dtype: dt)
        Ops.swigluLimitMany(gates: [gateP], ups: [upP], outs: [innerP], limit: textConfig.swigluLimit, on: fcmd2)
        // down → [M, hidden]. ZERO-REPACK: read raw Q2_K blocks via the view
        // kernel (no deinterleave pool), slot-indexed like the IQ2 gate/up
        // view path.
        let downP = Tensor.empty(shape: [M, hidden], dtype: dt)
        guard
            let rawD = try? bundle.rawGatherBlocks(
                named: dName, expertIndices: routedExperts, nExperts: nExperts, device: device, poolCap: cap,
                reuseKey: "prefill_view_down")
        else {
            throw PrefillError(message: "prefill L\(li): down expert tensor '\(dName)' not found in GGUF")
        }
        let slotDIdx = Tensor.empty(shape: [M], dtype: .u32, device: device)
        slotDIdx.copyIn(from: rows.map { UInt32(rawD.slotOf[$0.expert] ?? 0) })
        Ops.moeBgemmQ2KViewU16Bm64(
            x: innerP, viewBuf: rawD.buffer, viewByteOffset: 0,
            indices: slotDIdx, out: downP, mTotal: M, nOut: hidden, kIn: intermediate,
            tensorByteOff: 0, expertByteStride: rawD.byteStride, on: fcmd2)
        fcmd2.commit(); fcmd2.waitUntilCompleted()

        // Unpermute + weighted combine → moeAccum [N,hidden] in ONE dispatch
        // (was an M-row CPU loop of Tensor.filled+mul+add — the last per-token
        // offender in the batched path). invPerm maps each (token,slot) to its
        // expert-sorted row in downP; the kernel gathers + scales by topK weights.
        let fcmd3 = device.makeCommandBuffer()
        let moeAccum = Tensor.filled(0.0, shape: [N, hidden], dtype: dt, device: device)
        let invPermT = Tensor.empty(shape: [N * topK], dtype: .u32, device: device)
        invPermT.copyIn(from: invPermArr)
        let wT = Tensor.empty(shape: [N * topK], dtype: dt, device: device)
        if dt == .f16 { wT.copyIn(from: wOrig.map { Float16($0) }) } else { wT.copyIn(from: wOrig) }
        Ops.moeUnpermute(
            expertOutputs: downP, invPerm: invPermT, topKWeights: wT, into: moeAccum,
            nRows: N, hidden: hidden, k: topK, on: fcmd3)
        // Shared expert (Q8 gemm, M=N).
        let sgQ8 = q8("blk.\(li).ffn_gate_shexp.weight")!; let suQ8 = q8("blk.\(li).ffn_up_shexp.weight")!;
        let sdQ8 = q8("blk.\(li).ffn_down_shexp.weight")!
        let sG = Tensor.empty(shape: [N, sgQ8.mOut], dtype: dt)
        gq8(sgQ8, xNorm2, sG, N, fcmd3)
        let sU = Tensor.empty(shape: [N, suQ8.mOut], dtype: dt)
        gq8(suQ8, xNorm2, sU, N, fcmd3)
        // shared-expert swiglu — one flat dispatch over [N, intermediate].
        let sInner = Tensor.empty(shape: [N, intermediate], dtype: dt)
        Ops.swigluLimitMany(gates: [sG], ups: [sU], outs: [sInner], limit: textConfig.swigluLimit, on: fcmd3)
        let shexpOut = Tensor.empty(shape: [N, sdQ8.mOut], dtype: dt)
        gq8(sdQ8, sInner, shexpOut, N, fcmd3)
        let blockOut = Ops.add(moeAccum, shexpOut, on: fcmd3)
        let newH = Ops.dsv4MhcExpand(
            blockOut: blockOut, post: postF, comb: combF, residualState: hAfterAttn,
            hiddenDim: hidden, nHc: 4, nTokens: N, on: fcmd3)
        fcmd3.commit(); fcmd3.waitUntilCompleted()
        return newH
    }
}
