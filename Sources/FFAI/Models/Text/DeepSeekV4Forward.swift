// Copyright 2026 Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// DeepSeek V4 single-token decode forward path — full-attention
// sub-block. Lands incrementally: this file currently scaffolds the
// attention path against the existing Ops surface; the FFN sub-block
// (MoE + shared expert + mHC), the CSA / HCA paths, and the
// end-to-end `forward(...)` driver land in follow-ups.

import Foundation
import Metal
import QuartzCore

// MARK: - Per-call decode state

extension DeepSeekV4Model {
    /// Sliding-window MQA KV cache for one layer. Holds up to
    /// `n_swa=128` 512-d entries; appends grow `swCount` until the
    /// cache wraps. Indexing within the window stays in slot order
    /// (the SDPA kernel walks `[0..n_visible)` directly).
    public final class LayerKVState: @unchecked Sendable {
        public var swCache: Tensor  // [n_swa, head_dim]
        public var swCount: Int
        public let nSWA: Int
        public let headDim: Int

        // ── CSA/HCA compressor state (compress_ratio != 0 layers) ──
        public let compressRatio: Int  // 0 / 4 / 8 / 128
        public var compCache: Tensor?  // [maxComp, head_dim] compressed long-range KV
        public var compCount: Int = 0
        // Rolling per-token projection window (host): coff*ratio rows × width.
        // width = coff*head_dim; coff = (ratio==4 ? 2 : 1).
        public var compKvWin: [Float] = []
        public var compScoreWin: [Float] = []

        public init(headDim: Int, nSWA: Int, dtype: DType, compressRatio: Int = 0, maxComp: Int = 0) {
            self.swCache = Tensor.empty(shape: [nSWA, headDim], dtype: dtype)
            self.swCount = 0
            self.nSWA = nSWA
            self.headDim = headDim
            self.compressRatio = compressRatio
            if compressRatio != 0 {
                let coff = compressRatio == 4 ? 2 : 1
                let width = coff * headDim
                let rows = coff * compressRatio
                self.compKvWin = [Float](repeating: 0, count: rows * width)
                // scores init to -inf so unfilled lane-p rows contribute 0.
                self.compScoreWin = [Float](repeating: -1e30, count: rows * width)
                self.compCache = Tensor.empty(shape: [max(maxComp, 1), headDim], dtype: dtype)
            }
        }
    }

    /// One forward-call decode state.
    public final class DecodeState: @unchecked Sendable {
        public var layerStates: [LayerKVState]
        /// 4-channel mHC residual state, `[n_hc=4, hidden]`.
        public var hcState: Tensor
        public var position: Int
        /// Current input token id — needed by the hash-routed early
        /// layers (DSv4 ffn_gate_tid2eid lookup keys on the token id).
        public var currentToken: Int = 0

        public init(layerStates: [LayerKVState], hcState: Tensor, position: Int = 0) {
            self.layerStates = layerStates
            self.hcState = hcState
            self.position = position
        }
    }

    public func makeDecodeState() -> DecodeState {
        let cfg = textConfig
        // maxComp: compressed entries we can accumulate. For decode/prefill
        // of up to ~a few thousand tokens this is ctx/ratio; size generously.
        let maxCtx = 8192
        let states = (0 ..< cfg.nLayers).map { (il: Int) -> LayerKVState in
            let ratio = il < layerCompressRatios.count ? layerCompressRatios[il] : 0
            let maxComp = ratio != 0 ? (maxCtx / ratio + 4) : 0
            return LayerKVState(
                headDim: cfg.headDim, nSWA: cfg.slidingWindow, dtype: activationDtype,
                compressRatio: ratio, maxComp: maxComp)
        }
        let hc = Tensor.empty(shape: [4, cfg.hidden], dtype: activationDtype)
        return DecodeState(layerStates: states, hcState: hc, position: 0)
    }
}

// MARK: - Errors

enum DeepSeekV4ForwardError: Error, CustomStringConvertible {
    case notImplementedForRegime(Int)
    var description: String {
        switch self {
        case .notImplementedForRegime(let r):
            return "DSv4 forward path not yet implemented for compress_ratio=\(r)"
        }
    }
}

// MARK: - Shape helpers

extension Tensor {
    /// GGUF stores matmul weights as `[n_in_fast, n_out_slow]` in
    /// dimensions order, but `Ops.gemv` expects `[n_out, n_in]`.
    /// Swap the two dim labels (no data movement — same byte layout,
    /// different interpretation).
    public func asGgufMatmulWeight() -> Tensor {
        precondition(shape.count == 2, "asGgufMatmulWeight: rank must be 2")
        return reshaped(to: [shape[1], shape[0]])
    }
}

// MARK: - Ones-tensor cache (for per-head unit-RMS Q-norm)

extension DeepSeekV4Model {
    /// `[head_dim]` ones tensor, cached lazily on first access so
    /// the per-head Q unit-RMS norm has a no-op weight to pass to
    /// `Ops.rmsNormRows`. Backed by the generic `Tensor.filled(...)`
    /// constructor — any model that needs a constant-valued weight
    /// for a no-learnable-weight norm can reuse the same primitive.
    fileprivate func qHeadNormOnes(_ dt: DType) -> Tensor {
        if let cached = qHeadNormOnesCache { return cached }
        // MUST be a persistent (non-scratch) buffer: this tensor is cached
        // and reused across every layer/token. Tensor.filled → Tensor.empty
        // routes to the scratch slab whenever scratchModeActive is true
        // (which it is inside forwardFullAttnSubblock's withScratch), so the
        // cached "ones" would alias recycled scratch memory (= whatever
        // transient reused that slot, e.g. kvNorm) on the next layer —
        // silently corrupting the per-head Q-norm weight. Force real memory.
        let wasActive = device.scratchModeActive
        device.scratchModeActive = false
        let t = Tensor.filled(1.0, shape: [textConfig.headDim], dtype: dt, device: device)
        device.scratchModeActive = wasActive
        qHeadNormOnesCache = t
        return t
    }

    /// Host copy of a layer's i32 hash-routing table (token-major,
    /// `topK` experts per token), cached on first use.
    func tid2eidHost(layerIndex: Int, tensor: Tensor) -> [Int32] {
        if let cached = tid2eidCache[layerIndex] { return cached }
        let host = tensor.toArray(as: Int32.self)
        tid2eidCache[layerIndex] = host
        return host
    }
}

// MARK: - Full-attention sub-block forward
//
// ## Infrastructure gaps blocking the runnable body
//
// 1. **Mixed-dtype rmsNorm**. `Ops.rmsNorm` enforces
//    `x.dtype == weight.dtype` but DSv4 ships norm weights as f32
//    while activations are f16. Either (a) cast f32 norm weights to
//    f16 at load time inside `GGUFTensorBundle.tensor(named:outDtype:)`
//    (currently f32/f16/bf16 sources ignore `outDtype` and pass
//    through with their on-disk dtype), or (b) widen Ops.rmsNorm to
//    accept f32 weight + f16 input.
//
// 2. **No-weight rmsNormRows**. The per-head Q-norm has no learnable
//    weight (just `eps`). `Ops.rmsNormRows` requires a `[rowSize]`
//    weight tensor. Either allocate a ones tensor once, or add a
//    `rmsNormRowsNoWeight` variant.
//
// 3. **Grouped O-LoRA `mul_mat_id`**. `attn_output_a` is a single
//    [4096 × 8192] tensor that must be applied as 8 distinct
//    [4096 × 1024] slices, each driven by a different [4096] slice
//    of the [n_heads × head_dim] attention output. No Ops surface
//    today does this without 8 sequential gemvs against
//    output-axis-strided weight views — and `slicedRows` only
//    slices the leading dim.
//
// 4. **GGUF matmul-weight layout swap**. GGUF dimensions list the
//    fast dim first: `[n_in, n_out]`. Ops.gemv expects `[n_out, n_in]`.
//    The `Tensor.asGgufMatmulWeight()` helper above swaps the dim
//    labels (no data movement). Verified correct for `Ops.gemv` by
//    inspection but not yet unit-tested.
//
// 5. **Sliding-window cache append**. `Ops.copy(_:into:)` writes the
//    [head_dim] kv_norm into `swCache.slicedRows(start: slot, count: 1)`
//    which is shape `[1, head_dim]` — element-count matches but
//    dtype precondition may need the slice to be the same dtype as
//    src (currently fine, both are activation dtype). Untested.
//
// The decode-state types below are correct as-is; the
// `forwardFullAttnSubblock` function body lives in a working
// branch until the 5 gaps are closed.

extension DeepSeekV4Model {

    /// Decode one full-attention layer's attention sub-block.
    /// Reads `state.hcState` (the 4-channel residual), runs the
    /// full-attn block, writes the new 4-channel state back into
    /// `state.hcState`, and returns the un-residualised
    /// `block_out [hidden]` for downstream introspection.
    ///
    /// Wired against the real Ops API (`gemv` with GGUF shape-swap,
    /// `rmsNorm` for the learnable norms, `dsv4MhcSinkhornSplit /
    /// Collapse / Expand` for the mHC dance, `dsv4PartialRope` for
    /// the K/Q tail rotation, `dsv4SdpaDecodeD512Sink` for the
    /// MQA attention with attn_sinks).
    ///
    /// Known-incorrect details (deferred to follow-ups) — these matter
    /// for numerical
    /// correctness but not for "does the dispatch chain compile and
    /// run without NaN?":
    /// - Per-head Q-norm (eps-only, no learnable weight) is **skipped**
    ///   — needs a `[head_dim]` ones tensor or a no-weight rms variant.
    /// - Grouped O-LoRA collapses to a single 32768 → 8192 → 4096
    ///   matmul that **does NOT** apply per-group LoRA-A slices.
    ///   Output dims match; values are wrong until the proper
    ///   per-group dispatch lands.
    /// YaRN RoPE params for a layer. Full layers (ratio 0): no YaRN.
    /// Compressed layers: freq_scale = 1/yarn_factor, ext_factor = 1, and
    /// the correction-dim ramp bounds (YaRN rope corr-dims). The
    /// YaRN magnitude scale (mscale) cancels to 1 in DSv4, so it's omitted.
    func yarnParams(ratio: Int) -> (freqScale: Float, extFactor: Float, corrLow: Float, corrHigh: Float) {
        if ratio == 0 { return (1.0, 0.0, 0.0, 0.0) }
        let nRot = Float(textConfig.qkRopeHeadDim)
        let nCtx = Float(textConfig.yarnOriginalContext)
        let base = textConfig.compressRopeTheta
        let betaFast: Float = 32, betaSlow: Float = 1
        func corrDim(_ beta: Float) -> Float {
            nRot * Foundation.log(nCtx / (beta * 2 * Float.pi)) / (2 * Foundation.log(base))
        }
        let low = max(0, Foundation.floor(corrDim(betaFast)))
        let high = min(nRot - 1, Foundation.ceil(corrDim(betaSlow)))
        return (1.0 / textConfig.yarnFactor, 1.0, low, high)
    }

    /// Cached ones tensor [n_hc*hidden] for the no-weight RMSNorm applied
    /// to the flattened mHC state BEFORE the hc_*_fn mix projection
    /// (`mix = fn @ rms_norm(flat)`).
    func hcFlatOnes(_ dt: DType) -> Tensor {
        if let c = hcFlatOnesCache { return c }
        let n = 4 * textConfig.hidden
        let wasActive = device.scratchModeActive
        device.scratchModeActive = false
        let t = Tensor.filled(1.0, shape: [n], dtype: dt, device: device)
        device.scratchModeActive = wasActive
        hcFlatOnesCache = t
        return t
    }

    /// mHC mix = fn @ rms_norm_no_weight(flatH). The RMSNorm over the full
    /// flattened mHC state is the step FFAI was missing (it fed raw flatH).
    func mhcMix(flatH: Tensor, fnWeight: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        let normed = Ops.rmsNorm(
            flatH, weight: hcFlatOnes(flatH.dtype),
            eps: textConfig.rmsNormEps, on: cmd)
        return Ops.gemv(weight: fnWeight, input: normed, on: cmd)
    }

    /// Upload host floats into a tensor of the given dtype.
    private func uploadFloats(_ f: [Float], into t: Tensor, dtype: DType) {
        switch dtype {
        case .f32: t.copyIn(from: f)
        case .f16: t.copyIn(from: f.map { Float16($0) })
        case .bf16:
            t.copyIn(
                from: f.map { (v: Float) -> UInt16 in
                    let b = v.bitPattern; return UInt16((b &+ 0x7FFF &+ ((b >> 16) & 1)) >> 16)
                })
        default: fatalError("uploadFloats: unsupported dtype \(dtype)")
        }
    }

    /// DSv4 CSA/HCA KV compressor — one streaming step.
    /// Projects attn_norm → kv/score rows, rolls
    /// the per-token window, and on a `compress_ratio` boundary emits one
    /// compressed KV row (per-dim softmax pool → RMSNorm → compressed-RoPE)
    /// into `ls.compCache`. Runs on its own committed command buffer and
    /// recomputes attn_norm from the (committed) hcState so it doesn't
    /// disturb the caller's shared attn/ffn command buffer.
    func compressorStepCPU(
        layer: DeepSeekV4Layer, state: DecodeState, ls: LayerKVState,
        layerTheta: Float, nNope: Int
    ) {
        let cfg = textConfig; let dt = activationDtype
        let hidden = cfg.hidden; let headDim = cfg.headDim
        let ratio = ls.compressRatio
        let coff = ratio == 4 ? 2 : 1
        let width = coff * headDim
        let pos = state.position
        let posMod = pos % ratio
        let row = ratio == 4 ? (ratio + posMod) : posMod

        // Recompute attn_norm from committed hcState, then project kv/score.
        let c = device.makeCommandBuffer()
        let flatH = state.hcState.reshaped(to: [4 * hidden])
        let mixes = mhcMix(flatH: flatH, fnWeight: layer.hcAttnFn.asGgufMatmulWeight(), on: c)
        let (preA, _, _) = Ops.dsv4MhcSinkhornSplit(
            mixes: mixes, scale: layer.hcAttnScale, base: layer.hcAttnBase,
            nTokens: 1, eps: cfg.hcEpsilon, sinkhornIters: cfg.hcSinkhornIterations, on: c)
        let x = Ops.dsv4MhcCollapse(
            state: state.hcState, pre: preA,
            hiddenDim: hidden, nHc: 4, nTokens: 1, outDtype: dt, on: c
        ).reshaped(to: [hidden])
        let xNorm = Ops.rmsNorm(x, weight: layer.attnNorm, eps: cfg.rmsNormEps, on: c)
        let kvCur = Ops.gemv(weight: layer.attnCompressorKV!.asGgufMatmulWeight(), input: xNorm, on: c)
        let scCur = Ops.gemv(weight: layer.attnCompressorGate!.asGgufMatmulWeight(), input: xNorm, on: c)
        c.commit(); c.waitUntilCompleted()
        let kvH = kvCur.toFloatArray(); let scH = scCur.toFloatArray()
        let apeH = layer.attnCompressorAPE!.toFloatArray()  // [ratio, width], j fast

        // Store projected rows into the rolling window (+ APE on score).
        for j in 0 ..< width {
            ls.compKvWin[row * width + j] = kvH[j]
            ls.compScoreWin[row * width + j] = scH[j] + apeH[posMod * width + j]
        }
        if (pos + 1) % ratio != 0 { return }

        // Per-dimension softmax pool.
        let negHalf: Float = -1e30 * 0.5
        var pooled = [Float](repeating: 0, count: headDim)
        for j in 0 ..< headDim {
            var mx = -Float.infinity
            if ratio == 4 {
                for r in 0 ..< ratio {
                    mx = max(mx, ls.compScoreWin[r * width + j])
                    mx = max(mx, ls.compScoreWin[(ratio + r) * width + headDim + j])
                }
            } else {
                for r in 0 ..< ratio { mx = max(mx, ls.compScoreWin[r * width + j]) }
            }
            if mx <= negHalf { continue }
            var denom: Float = 0, sum: Float = 0
            if ratio == 4 {
                for r in 0 ..< ratio {
                    let wp = expf(ls.compScoreWin[r * width + j] - mx)
                    let wc = expf(ls.compScoreWin[(ratio + r) * width + headDim + j] - mx)
                    denom += wp + wc
                    sum += wp * ls.compKvWin[r * width + j]
                    sum += wc * ls.compKvWin[(ratio + r) * width + headDim + j]
                }
            } else {
                for r in 0 ..< ratio {
                    let w = expf(ls.compScoreWin[r * width + j] - mx)
                    denom += w; sum += w * ls.compKvWin[r * width + j]
                }
            }
            pooled[j] = denom > 0 ? sum / denom : 0
        }
        // RMSNorm(compressor_norm).
        let normH = layer.attnCompressorNorm!.toFloatArray()
        var ss = 0.0; for v in pooled { ss += Double(v) * Double(v) }
        let rms = Float(1.0 / ((ss / Double(headDim)) + Double(cfg.rmsNormEps)).squareRoot())
        var outComp = [Float](repeating: 0, count: headDim)
        for i in 0 ..< headDim { outComp[i] = pooled[i] * rms * normH[i] }

        guard let cc = ls.compCache, ls.compCount < cc.shape[0] else { return }
        let dst = cc.slicedRows(start: ls.compCount, count: 1).reshaped(to: [headDim])
        uploadFloats(outComp, into: dst, dtype: dt)
        let compPos = pos + 1 - ratio
        if compPos > 0 {
            let yp = yarnParams(ratio: ratio)
            let rc = device.makeCommandBuffer()
            Ops.dsv4PartialRope(
                qk: dst, out: dst, nHeads: 1, headDim: headDim, nNope: nNope,
                position: compPos, thetaBase: layerTheta, inverse: false,
                freqScale: yp.freqScale, extFactor: yp.extFactor, corrLow: yp.corrLow, corrHigh: yp.corrHigh,
                on: rc)
            rc.commit(); rc.waitUntilCompleted()
        }
        ls.compCount += 1

        // Slide CSA two-lane window: lane-c → lane-p, then lane-p → lane-c.
        if ratio == 4 {
            for r in 0 ..< ratio {
                for j in 0 ..< width {
                    ls.compKvWin[r * width + j] = ls.compKvWin[(ratio + r) * width + j]
                    ls.compScoreWin[r * width + j] = ls.compScoreWin[(ratio + r) * width + j]
                }
            }
            for r in 0 ..< ratio {
                for j in 0 ..< width {
                    ls.compKvWin[(ratio + r) * width + j] = ls.compKvWin[r * width + j]
                    ls.compScoreWin[(ratio + r) * width + j] = ls.compScoreWin[r * width + j]
                }
            }
        }
    }

    public func forwardFullAttnSubblock(
        layer: DeepSeekV4Layer, state: DecodeState, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let cfg = textConfig
        let dt = activationDtype
        let hidden = cfg.hidden
        let headDim = cfg.headDim
        let qLoraRank = cfg.qLoraRank
        let nHeads = cfg.nHeads
        let qkRopeDim = cfg.qkRopeHeadDim
        let nNope = headDim - qkRopeDim
        // Per-layer RoPE base: compressed layers (compress_ratio != 0, i.e.
        // CSA ratio-4 and HCA ratio-128) use compress_rope_theta=160000;
        // full layers (0, 1, 42 → ratio 0) use rope_theta=10000.
        // ratio != 0 ⇒ 160000.
        let li = layer.layerIndex
        let layerRatio = li < layerCompressRatios.count ? layerCompressRatios[li] : 0
        let layerTheta = layerRatio != 0 ? cfg.compressRopeTheta : cfg.ropeTheta
        let yp = yarnParams(ratio: layerRatio)

        // ── mHC pre/post/comb split ──
        // mixes = hc_attn_fn @ flatten(H)  →  [24]
        let flatH = state.hcState.reshaped(to: [4 * hidden])
        let hcAttnFnW = layer.hcAttnFn.asGgufMatmulWeight()
        let mixes = mhcMix(flatH: flatH, fnWeight: hcAttnFnW, on: cmd)
        let (preAttn, postAttn, combAttn) = Ops.dsv4MhcSinkhornSplit(
            mixes: mixes, scale: layer.hcAttnScale, base: layer.hcAttnBase,
            nTokens: 1, eps: cfg.hcEpsilon, sinkhornIters: cfg.hcSinkhornIterations,
            on: cmd)

        // ── mHC collapse: H[4, hidden] → x[hidden] (drop n_tokens=1 dim) ──
        let xWithTokens = Ops.dsv4MhcCollapse(
            state: state.hcState, pre: preAttn,
            hiddenDim: hidden, nHc: 4, nTokens: 1, outDtype: dt, on: cmd)
        let x = xWithTokens.reshaped(to: [hidden])

        // ── attn_norm ──
        let xNorm = Ops.rmsNorm(x, weight: layer.attnNorm, eps: cfg.rmsNormEps, on: cmd)
        if let da = dbgAnorm, li * 4 + 4 <= da.elementCount {
            // capture x (pre-norm collapse) per layer
            Ops.copy(
                x.slicedRows(start: 0, count: 4),
                into: da.slicedRows(start: li * 4, count: 4), on: cmd)
        }

        // ── Q low-rank chain: x → q_a → q_a_norm → q_b ──
        // q_a/q_b/kv/output_* are Q8_0 on disk — gemv straight from
        // resident Q8 (1 byte/weight) instead of the f16-expanded copy,
        // halving read bandwidth (attn is bandwidth-bound).
        let useQ8Attn = ProcessInfo.processInfo.environment["FFAI_DSV4_Q8ATTN"] != "0"
        let qA: Tensor
        if useQ8Attn, let qaQ8 = try? bundle.residentQ8("blk.\(layer.layerIndex).attn_q_a.weight", device: device) {
            qA = Tensor.empty(shape: [qaQ8.mOut], dtype: dt)
            Ops.gemvQ8(q8: qaQ8, x: xNorm, on: cmd, into: qA)
        } else {
            qA = Ops.gemv(weight: layer.attnQA.asGgufMatmulWeight(), input: xNorm, on: cmd)
        }
        let qANorm = Ops.rmsNorm(qA, weight: layer.attnQANorm, eps: cfg.rmsNormEps, on: cmd)
        if let d0 = dbgL0, li == 0, d0.elementCount >= 16 {
            Ops.copy(qANorm.slicedRows(start: 0, count: 4), into: d0.slicedRows(start: 12, count: 4), on: cmd)
        }
        let q: Tensor
        if useQ8Attn, let qbQ8 = try? bundle.residentQ8("blk.\(layer.layerIndex).attn_q_b.weight", device: device) {
            q = Tensor.empty(shape: [qbQ8.mOut], dtype: dt)
            Ops.gemvQ8(q8: qbQ8, x: qANorm, on: cmd, into: q)
        } else {
            q = Ops.gemv(weight: layer.attnQB.asGgufMatmulWeight(), input: qANorm, on: cmd)
        }
        // Per-head unit-RMS Q-norm: normalize each [head_dim] row
        // independently with no learnable weight. Pass a ones-tensor
        // of shape [head_dim] cached on the model.
        Ops.rmsNormRows(
            q, weight: qHeadNormOnes(dt), eps: cfg.rmsNormEps,
            nRows: nHeads, rowSize: headDim, on: cmd, into: q)

        // ── Partial RoPE on Q tail ──
        let qRoped = Tensor.empty(shape: q.shape, dtype: dt)
        Ops.copy(q, into: qRoped, on: cmd)
        Ops.dsv4PartialRope(
            qk: qRoped, out: qRoped,
            nHeads: nHeads, headDim: headDim, nNope: nNope,
            position: state.position, thetaBase: layerTheta, inverse: false, freqScale: yp.freqScale,
            extFactor: yp.extFactor, corrLow: yp.corrLow, corrHigh: yp.corrHigh, on: cmd)

        // ── KV down-projection + norm + partial RoPE ──
        let kv: Tensor
        if useQ8Attn, let kvQ8 = try? bundle.residentQ8("blk.\(layer.layerIndex).attn_kv.weight", device: device) {
            kv = Tensor.empty(shape: [kvQ8.mOut], dtype: dt)
            Ops.gemvQ8(q8: kvQ8, x: xNorm, on: cmd, into: kv)
        } else {
            kv = Ops.gemv(weight: layer.attnKV.asGgufMatmulWeight(), input: xNorm, on: cmd)
        }
        let kvNorm = Ops.rmsNorm(kv, weight: layer.attnKVANorm, eps: cfg.rmsNormEps, on: cmd)
        Ops.dsv4PartialRope(
            qk: kvNorm, out: kvNorm,
            nHeads: 1, headDim: headDim, nNope: nNope,
            position: state.position, thetaBase: layerTheta, inverse: false, freqScale: yp.freqScale,
            extFactor: yp.extFactor, corrLow: yp.corrLow, corrHigh: yp.corrHigh, on: cmd)

        // ── Append to sliding-window cache ──
        let layerState = state.layerStates[layer.layerIndex]
        let slot = layerState.swCount % layerState.nSWA
        Ops.copy(kvNorm, into: layerState.swCache.slicedRows(start: slot, count: 1), on: cmd)
        layerState.swCount += 1
        let nVisible = min(layerState.swCount, layerState.nSWA)

        // ── CSA/HCA compressor: update the compressed long-range KV stream
        // (runs on its own committed cmd; recomputes attn_norm from the
        // committed hcState — see compressorStepCPU). ──
        if layerState.compressRatio != 0 {
            compressorStepCPU(
                layer: layer, state: state, ls: layerState,
                layerTheta: layerTheta, nNope: nNope)
        }

        // ── MQA SDPA with attn_sinks over [raw sliding window ∪ compressed
        // rows]. CSA/HCA: dense over all compressed entries (no indexer top-k
        // needed while n_comp ≤ index_topk=512). ──
        let scale = 1.0 / Float(headDim).squareRoot()
        let nComp = layerState.compressRatio != 0 ? layerState.compCount : 0
        let kvBuf: Tensor
        let nKvTotal: Int
        if nComp > 0, let cc = layerState.compCache {
            let total = nVisible + nComp
            let combined = Tensor.empty(shape: [total, headDim], dtype: dt)
            Ops.copy(
                layerState.swCache.slicedRows(start: 0, count: nVisible),
                into: combined.slicedRows(start: 0, count: nVisible), on: cmd)
            Ops.copy(
                cc.slicedRows(start: 0, count: nComp),
                into: combined.slicedRows(start: nVisible, count: nComp), on: cmd)
            kvBuf = combined; nKvTotal = total
        } else {
            kvBuf = layerState.swCache.slicedRows(start: 0, count: nVisible)
            nKvTotal = nVisible
        }
        let attnOut = Ops.dsv4SdpaDecodeD512Sink(
            q: qRoped, k: kvBuf, v: kvBuf, sinkLogit: layer.attnSinks,
            nQHeads: nHeads, nKvHeads: 1, headDim: headDim,
            nKv: nKvTotal, kvStride: nKvTotal,
            scale: scale, outDtype: dt, on: cmd)

        if let d0 = dbgL0, li == 0, d0.elementCount >= 16 {
            Ops.copy(qRoped.slicedRows(start: 0, count: 4), into: d0.slicedRows(start: 0, count: 4), on: cmd)
            Ops.copy(kvNorm.slicedRows(start: 0, count: 4), into: d0.slicedRows(start: 4, count: 4), on: cmd)
            Ops.copy(
                attnOut.reshaped(to: [nHeads * headDim]).slicedRows(start: 0, count: 4),
                into: d0.slicedRows(start: 8, count: 4), on: cmd)
        }
        // ── Inverse partial RoPE on attention output ──
        Ops.dsv4PartialRope(
            qk: attnOut, out: attnOut,
            nHeads: nHeads, headDim: headDim, nNope: nNope,
            position: state.position, thetaBase: layerTheta, inverse: true, freqScale: yp.freqScale,
            extFactor: yp.extFactor, corrLow: yp.corrLow, corrHigh: yp.corrHigh, on: cmd)

        // ── Grouped O-LoRA: 8 groups × [4096, 1024] then [8192, 4096] ──
        // Reshape attnOut [n_heads, head_dim] → [n_groups, group_dim]
        // = [8, 4096]. Each group consumes a different LoRA-A slice;
        // since attn_output_a is stored [n_in=4096, n_out=8192] in
        // GGUF (= [8192, 4096] after the as-weight swap), and the
        // 8192-output-dim is the **concatenation of 8 × 1024
        // per-group LoRA-A outputs**, the per-group dispatch is:
        //   oLow[g, :1024] = wA[g, :1024, :] @ attnOut_group[g, :]
        // 8 sequential gemvs, each [1024, 4096], with weight slice
        // taken as a row-range of the swapped weight tensor (axis-0
        // slicing — contiguous and supported by `slicedRows`).
        let oGroups = 8
        let groupDim = (nHeads * headDim) / oGroups  // 4096
        let oLoraRank = cfg.oLoraRank  // 1024
        let attnOutGrouped = attnOut.reshaped(to: [oGroups, groupDim])
        let oLow = Tensor.empty(shape: [oGroups * oLoraRank], dtype: dt)
        let outputAW = layer.attnOutputA.asGgufMatmulWeight()  // [8192, 4096]
        let oaQ8 =
            useQ8Attn ? try? bundle.residentQ8("blk.\(layer.layerIndex).attn_output_a.weight", device: device) : nil
        if let oaQ8 = oaQ8 {
            // One grouped Q8 gemv: row block g reads attnOutGrouped[g].
            // attnOut is already the contiguous [oGroups * groupDim]
            // input the grouped kernel expects.
            Ops.groupedGemvQ8(
                q8: oaQ8, x: attnOut, rowsPerGroup: oLoraRank, on: cmd, into: oLow)
        } else {
            for g in 0 ..< oGroups {
                let inputSlice = attnOutGrouped.slicedRows(start: g, count: 1).reshaped(to: [groupDim])
                let outSlice = oLow.slicedRows(start: g * oLoraRank, count: oLoraRank)
                let weightSlice = outputAW.slicedRows(start: g * oLoraRank, count: oLoraRank)
                _ = Ops.gemv(weight: weightSlice, input: inputSlice, on: cmd, into: outSlice)
            }
        }
        let blockOut: Tensor
        if useQ8Attn, let obQ8 = try? bundle.residentQ8("blk.\(layer.layerIndex).attn_output_b.weight", device: device)
        {
            blockOut = Tensor.empty(shape: [obQ8.mOut], dtype: dt)
            Ops.gemvQ8(q8: obQ8, x: oLow, on: cmd, into: blockOut)
        } else {
            blockOut = Ops.gemv(
                weight: layer.attnOutputB.asGgufMatmulWeight(), input: oLow, on: cmd)
        }

        // ── mHC expand: write new 4-channel state ──
        let newH = Ops.dsv4MhcExpand(
            blockOut: blockOut, post: postAttn, comb: combAttn,
            residualState: state.hcState,
            hiddenDim: hidden, nHc: 4, nTokens: 1, on: cmd)
        state.hcState = newH
        if let d0 = dbgL0, li == 0, d0.elementCount >= 24 {
            Ops.copy(
                blockOut.reshaped(to: [hidden]).slicedRows(start: 0, count: 4),
                into: d0.slicedRows(start: 16, count: 4), on: cmd)
            Ops.copy(
                newH.reshaped(to: [4 * hidden]).slicedRows(start: 0, count: 4),
                into: d0.slicedRows(start: 20, count: 4), on: cmd)
        }
        return blockOut
    }

    /// FFN sub-block — runs the mHC dance + RMS norm + MoE-top-6 +
    /// shared expert + mHC expand. Single token, decode mode.
    ///
    /// Selection path: full sqrtsoftplus router scoring → top-6 via
    /// CPU readback (no GPU `argpartition` Op yet, so this is the
    /// quick-correct path). Expert dispatch is 6 × 3 gemvs against
    /// per-expert slices of the [n_experts, intermediate, hidden]
    /// tensors. Combine = weighted sum of expert outputs by
    /// `score_unbiased * routed_scaling_factor`, plus the
    /// always-on shared expert.
    public static func resetFfnProf() {
        Self.profRouter = 0
        Self.profExpertRecord = 0
        Self.profSharedRecord = 0
        Self.profGpuWait = 0
        Self.resetGpuProf()
    }
    public func forwardFfnSubblock(
        layer: DeepSeekV4Layer, state: DecodeState, on cmd: MTLCommandBuffer,
        copyHcInto outHc: Tensor? = nil
    ) throws -> Tensor {
        let _tFfnStart = CACurrentMediaTime()
        let cfg = textConfig
        let dt = activationDtype
        let hidden = cfg.hidden
        let intermediate = cfg.moeIntermediate
        // ffnGateExps is a placeholder [1] in the lazy-load world.
        // ffnGateInp is shape [hidden, n_experts] — last dim is the
        // real expert count. Prefer it over cfg, which falls back to
        // a generic default (288) when the gguf metadata doesn't
        // expose `n_routed_experts` and so disagrees with this gguf
        // (256-expert variant of DSv4-Flash).
        let nExperts = layer.ffnGateInp.shape.last ?? cfg.nExperts
        let topK = cfg.nExpertsPerToken
        let scaling = cfg.routerScalingFactor

        // ── mHC pre/post/comb split ──
        let flatH = state.hcState.reshaped(to: [4 * hidden])
        let hcFnW = layer.hcFfnFn.asGgufMatmulWeight()
        let mixes = mhcMix(flatH: flatH, fnWeight: hcFnW, on: cmd)
        let (preFfn, postFfn, combFfn) = Ops.dsv4MhcSinkhornSplit(
            mixes: mixes, scale: layer.hcFfnScale, base: layer.hcFfnBase,
            nTokens: 1, eps: cfg.hcEpsilon, sinkhornIters: cfg.hcSinkhornIterations,
            on: cmd)

        // ── mHC collapse + ffn_norm ──
        let xWithTokens = Ops.dsv4MhcCollapse(
            state: state.hcState, pre: preFfn,
            hiddenDim: hidden, nHc: 4, nTokens: 1, outDtype: dt, on: cmd)
        let x = xWithTokens.reshaped(to: [hidden])
        let xNorm = Ops.rmsNorm(x, weight: layer.ffnNorm, eps: cfg.rmsNormEps, on: cmd)
        if let d0 = dbgL0, layer.layerIndex == 0, d0.elementCount >= 56 {
            // xNorm[2048..2051] right after rmsNorm (compare to expert-gemv-time value)
            Ops.copy(xNorm.slicedRows(start: 2048, count: 4), into: d0.slicedRows(start: 52, count: 4), on: cmd)
        }
        // ── Router scoring: logits = ffn_gate_inp @ xNorm ──
        let routerLogits = Ops.gemv(
            weight: layer.ffnGateInp.asGgufMatmulWeight(), input: xNorm, on: cmd)
        // The sqrtsoftplus router Op takes f32 logits + bias and writes
        // f32 score_unbiased + score_biased. Routerlogits is `dt`
        // (activation dtype). Cast to f32 first.
        let routerLogitsF32 = Tensor.empty(shape: routerLogits.shape, dtype: .f32)
        Ops.castToF32(routerLogits, into: routerLogitsF32, on: cmd)
        let bias: Tensor
        if let b = layer.expProbsBias {
            bias = b
        } else {
            bias = Tensor.filled(0.0, shape: [nExperts], dtype: .f32, device: device)
        }
        let (scoreUnbiased, scoreBiased) = Ops.dsv4MoeRouterSqrtsoftplus(
            logits: routerLogitsF32, bias: bias, on: cmd)

        // ── Sync-free GPU routing fast path ──
        // Once the resident expert pools for this layer are built (the
        // first token fills them via the CPU path below), route entirely
        // on the GPU: top-K + weights via mt_dsv4_router_topk, raw→slot
        // remap, then the gather dispatches — NO per-layer
        // waitUntilCompleted. Removes 43 CPU↔GPU round-trips/token.
        // Correct when the routed experts are all pool-resident (always,
        // post-warmup, for a fixed prompt); a pool miss reads slot 0
        // (same timing — see PLAN.md §9 caveat).
        let gateNameF = "blk.\(layer.layerIndex).ffn_gate_exps.weight"
        let upNameF = "blk.\(layer.layerIndex).ffn_up_exps.weight"
        let downNameF = "blk.\(layer.layerIndex).ffn_down_exps.weight"
        if ProcessInfo.processInfo.environment["FFAI_DSV4_GPUROUTER"] != "0", topK == 6,
            layer.ffnGateTid2Eid == nil,  // hash-routed layers select via token→expert table, not GPU top-k
            let rg = bundle.builtIQ2(gateNameF),
            let ru = bundle.builtIQ2(upNameF),
            let rd = bundle.builtQ2K(downNameF)
        {
            return try gpuRoutedFfnTail(
                layer: layer, state: state, xNorm: xNorm,
                scoreBiased: scoreBiased, scoreUnbiased: scoreUnbiased,
                rg: rg, ru: ru, rd: rd, nExperts: nExperts, topK: topK,
                hidden: hidden, intermediate: intermediate, dt: dt,
                postFfn: postFfn, combFfn: combFfn, attnCmd: cmd,
                copyHcInto: outHc)
        }

        // CPU-side top-K selection. Sync flush, readback, argpartition.
        // (Also the warmup pass that builds the resident pools.)
        DeepSeekV4Model.commitWithProfile(cmd, tag: "attn+router")
        cmd.waitUntilCompleted()
        // EXPERIMENT (FFAI_DSV4_Q8K_ACT=1): replicate the reference's Q8_K
        // activation quant on the expert input. The IQ2/Q2_K experts are
        // imatrix-calibrated assuming Q8_K activations (the reference
        // quantize x → Q8_K before the 2-bit dot); FFAI's f16 activation
        // deviates. Round-trip xNorm through Q8_K (per-256-block,
        // iscale=-128/max) in place so the expert gemvs see the same
        // rounded activation. Router already consumed the f16 xNorm above.
        if ProcessInfo.processInfo.environment["FFAI_DSV4_Q8K_ACT"] == "1" {
            var h = xNorm.toFloatArray()
            let n = h.count
            var i = 0
            while i < n {
                let end = Swift.min(i + 256, n)
                var maxv: Float = 0, amax: Float = 0
                for j in i ..< end { let a = Swift.abs(h[j]); if a > amax { amax = a; maxv = h[j] } }
                if amax > 0 {
                    let iscale = -128.0 / maxv
                    let d = 1.0 / iscale
                    for j in i ..< end {
                        var q = (iscale * h[j]).rounded()
                        if q > 127 { q = 127 }
                        h[j] = Float(q) * Float(d)
                    }
                }
                i = end
            }
            xNorm.copyIn(from: h.map { Float16($0) })
        }
        let biasedHost = scoreBiased.toArray(as: Float.self)
        let unbiasedHost = scoreUnbiased.toArray(as: Float.self)
        let topIndices: [Int]
        if let tid2eid = layer.ffnGateTid2Eid {
            // ── Hash routing (DSv4 early layers, il < n_hash_layer=3) ──
            // The selected experts come from a precomputed token→expert
            // table (ffn_gate_tid2eid, i32 [n_expert_used, vocab], stored
            // token-major so token t's experts are at [t*topK ..< t*topK+topK]).
            // The router logits are NOT used for selection here — only for
            // the combine weights (probs at the hash-selected experts).
            let table = self.tid2eidHost(layerIndex: layer.layerIndex, tensor: tid2eid)
            let tok = state.currentToken
            topIndices = (0 ..< topK).map { Int(table[tok * topK + $0]) }
        } else {
            var indexed = Array(biasedHost.enumerated())
            indexed.sort { $0.element > $1.element }
            topIndices = Array(indexed.prefix(topK)).map { $0.offset }
        }
        // routed_scaling_factor applied AFTER the sum-to-1 renorm (DSv4
        // reference order); applying it before (unbiased*scaling) cancels
        // in the division and drops the 1.5× entirely. Hash and top-k
        // layers share this weight formula: probs[selected]/sum * scale.
        let topWeights = topIndices.map { unbiasedHost[$0] }
        let weightSum = topWeights.reduce(0, +)
        let normWeights: [Float] =
            weightSum > 0
            ? topWeights.map { ($0 / weightSum) * scaling }
            : Array(repeating: scaling / Float(topK), count: topK)
        if dbgL0 != nil, layer.layerIndex == 0 {
            let xn = xNorm.toFloatArray()
            FileHandle.standardError.write(
                Data(
                    String(
                        format:
                            "[dsv4bench] FFL0ROUTER experts=\(topIndices) tw=\(topWeights.map{String(format:"%.5f",$0)}) ffn_norm=%.5f,%.5f,%.5f,%.5f\n",
                        xn[0], xn[1], xn[2], xn[3]
                    ).utf8))
        }

        // ── Expert dispatch ──
        // gate_exps / up_exps:  [hidden, intermediate, n_experts]
        //   → reshape [n_experts, intermediate, hidden] (no data move,
        //     fast/slow swap), slice expert e, get [intermediate, hidden]
        //     = [n_out, n_in] which Ops.gemv accepts directly.
        // down_exps: [intermediate, hidden, n_experts]
        //   → reshape [n_experts, hidden, intermediate], slice e,
        //     [hidden, intermediate] = [n_out, n_in].
        // GPU-side accumulator: moeOut += w_k * expert_out_k for each
        // of topK experts, then + shared-expert output. Uses Ops.add
        // (vector_add) to keep the chain on-GPU — no per-expert
        // CPU sync.
        // **Lazy per-expert dequant** — dequant only the top-K=6
        // routed experts here, not all 256 at layer load. 42× less
        // dequant work per token (≤2 GB / layer vs ~12 GB eager,
        // ≤16 MB per expert × 6 experts × 3 roles).
        //
        // Each (iter k, role) gets its own pool slot so a later
        // iter's dequant can't overwrite an earlier iter's already-
        // encoded gemv input before cmd2.commit. Slots reused across
        // layers (cmd2 commits + waits at end of FFN, slabs are free
        // when the next layer's FFN starts).
        let _tRouter = CACurrentMediaTime()
        DeepSeekV4Model.profRouter += _tRouter - _tFfnStart
        let gateName = "blk.\(layer.layerIndex).ffn_gate_exps.weight"
        let upName = "blk.\(layer.layerIndex).ffn_up_exps.weight"
        let downName = "blk.\(layer.layerIndex).ffn_down_exps.weight"
        _ = intermediate
        let moeAccum = Tensor.filled(0.0, shape: [hidden], dtype: dt, device: device)
        let env = ProcessInfo.processInfo.environment
        let useFusedDown = topK == 6 && env["FFAI_DSV4_FUSED_DOWN"] == "1"
        let useFusedEpilogue =
            topK == 6
            && !useFusedDown
            && env["FFAI_DSV4_FUSED_EPILOGUE"] == "1"
        // PER-EXPERT 3-stage pipeline: gate / up / down on separate cmd
        // buffers. gate+up are independent of each other so CPU staging
        // for up can run concurrent with GPU running gate's dequant +
        // gemv. swiglu fuses on cmdAB after both gate+up finish.
        var pipeGate: [MTLCommandBuffer] = []
        var pipeUp: [MTLCommandBuffer] = []
        var pipeB: [MTLCommandBuffer] = []
        var gateOuts: [Tensor?] = Array(repeating: nil, count: topK)
        var upOuts: [Tensor?] = Array(repeating: nil, count: topK)
        for _ in 0 ..< topK {
            pipeGate.append(device.makeCommandBuffer())
            pipeUp.append(device.makeCommandBuffer())
            pipeB.append(device.makeCommandBuffer())
        }
        func recordExpertGate(_ k: Int) throws {
            let e = topIndices[k]
            let cmd = pipeGate[k]
            let gateWp = try bundle.dequantExpertSliceOnto(
                named: gateName, expertIdx: e, nExperts: nExperts,
                slot: "k\(k)_gate", outDtype: dt, device: device, on: cmd)
            // TEST: copy the pooled dequant into a fresh contiguous tensor
            // before the gemv to rule out a stride/offset metadata issue.
            let gateW = gateWp
            gateOuts[k] = Ops.gemv(weight: gateW.asGgufMatmulWeight(), input: xNorm, on: cmd)
            if let d0 = dbgL0, layer.layerIndex == 0, k == 0, d0.elementCount >= 52 {
                Ops.copy(
                    gateOuts[k]!.reshaped(to: [gateOuts[k]!.elementCount]).slicedRows(start: 0, count: 4),
                    into: d0.slicedRows(start: 48, count: 4), on: cmd)
            }
        }
        func recordExpertUp(_ k: Int) throws {
            let e = topIndices[k]
            let cmd = pipeUp[k]
            let upW = try bundle.dequantExpertSliceOnto(
                named: upName, expertIdx: e, nExperts: nExperts,
                slot: "k\(k)_up", outDtype: dt, device: device, on: cmd)
            upOuts[k] = Ops.gemv(weight: upW.asGgufMatmulWeight(), input: xNorm, on: cmd)
            if let d0 = dbgL0, layer.layerIndex == 0, k == 0, d0.elementCount >= 56 {
                Ops.copy(
                    upOuts[k]!.reshaped(to: [upOuts[k]!.elementCount]).slicedRows(start: 0, count: 4),
                    into: d0.slicedRows(start: 52, count: 4), on: cmd)
            }
        }
        func recordExpertB(_ k: Int, on cmd: MTLCommandBuffer) throws {
            let e = topIndices[k]
            let w = normWeights[k]
            let inner = Ops.swigluLimit(gate: gateOuts[k]!, up: upOuts[k]!, limit: textConfig.swigluLimit, on: cmd)
            let downW = try bundle.dequantExpertSliceOnto(
                named: downName, expertIdx: e, nExperts: nExperts,
                slot: "k\(k)_down", outDtype: dt, device: device, on: cmd)
            let expertOut = Ops.gemv(
                weight: downW.asGgufMatmulWeight(), input: inner, on: cmd)
            if let d0 = dbgL0, layer.layerIndex == 0, k == 0, d0.elementCount >= 52 {
                Ops.copy(
                    expertOut.reshaped(to: [expertOut.elementCount]).slicedRows(start: 0, count: 4),
                    into: d0.slicedRows(start: 48, count: 4), on: cmd)
            }
            let wT = Tensor.filled(w, shape: [hidden], dtype: dt, device: device)
            let scaled = Ops.mul(expertOut, wT, on: cmd)
            _ = Ops.add(moeAccum, scaled, on: cmd, into: moeAccum)
        }
        func recordExpertDownOnly(_ k: Int, on cmd: MTLCommandBuffer) throws -> Tensor {
            let e = topIndices[k]
            let inner = Ops.swigluLimit(gate: gateOuts[k]!, up: upOuts[k]!, limit: textConfig.swigluLimit, on: cmd)
            let downW = try bundle.dequantExpertSliceOnto(
                named: downName, expertIdx: e, nExperts: nExperts,
                slot: "k\(k)_down", outDtype: dt, device: device, on: cmd)
            return Ops.gemv(weight: downW.asGgufMatmulWeight(), input: inner, on: cmd)
        }
        func recordFusedEpilogue(on cmd: MTLCommandBuffer) throws {
            var expertOuts: [Tensor?] = Array(repeating: nil, count: topK)
            for k in 0 ..< (topK - 1) {
                expertOuts[k] = try recordExpertDownOnly(k, on: pipeB[k])
                DeepSeekV4Model.commitWithProfile(pipeB[k], tag: "expert_down")
            }
            expertOuts[topK - 1] = try recordExpertDownOnly(topK - 1, on: cmd)

            var scalars: [Tensor] = []
            var values: [Tensor] = []
            scalars.reserveCapacity(8)
            values.reserveCapacity(8)
            for k in 0 ..< topK {
                scalars.append(Tensor.filled(normWeights[k], shape: [1], dtype: dt, device: device))
                values.append(expertOuts[k]!)
            }
            let zeroScalar = Tensor.filled(0.0, shape: [1], dtype: dt, device: device)
            let zeroValue = Tensor.filled(0.0, shape: [hidden], dtype: dt, device: device)
            while scalars.count < 8 {
                scalars.append(zeroScalar)
                values.append(zeroValue)
            }
            Ops.scalarFMAChain8(scalars: scalars, values: values, out: moeAccum, on: cmd)
        }
        func recordFusedRoutedDown(on cmd: MTLCommandBuffer) throws {
            var inners: [Tensor] = []
            var downs: [Tensor] = []
            inners.reserveCapacity(topK)
            downs.reserveCapacity(topK)
            for k in 0 ..< topK {
                let e = topIndices[k]
                let inner = Ops.swigluLimit(gate: gateOuts[k]!, up: upOuts[k]!, limit: textConfig.swigluLimit, on: cmd)
                let downW = try bundle.dequantExpertSliceOnto(
                    named: downName, expertIdx: e, nExperts: nExperts,
                    slot: "k\(k)_down", outDtype: dt, device: device, on: cmd)
                inners.append(inner)
                downs.append(downW.asGgufMatmulWeight())
            }
            let weights = Tensor.empty(shape: [topK], dtype: .f32, device: device)
            weights.copyIn(from: normWeights)
            Ops.moeDownWeightedSum6(
                downs: downs, inners: inners, weights: weights,
                accum: moeAccum, on: cmd)
        }
        // Shared expert on its own pipe cmd buffer, encoded + committed
        // FIRST so the GPU can start it while CPU stages routed
        // experts.
        let shexpCmd = device.makeCommandBuffer()
        let sGate = Ops.gemv(
            weight: layer.ffnGateShexp.asGgufMatmulWeight(), input: xNorm, on: shexpCmd)
        let sUp = Ops.gemv(
            weight: layer.ffnUpShexp.asGgufMatmulWeight(), input: xNorm, on: shexpCmd)
        let sInner = Ops.swigluLimit(gate: sGate, up: sUp, limit: textConfig.swigluLimit, on: shexpCmd)
        let shexpOut = Ops.gemv(
            weight: layer.ffnDownShexp.asGgufMatmulWeight(), input: sInner, on: shexpCmd)
        if let d0 = dbgL0, layer.layerIndex == 0, d0.elementCount >= 52 {
            Ops.copy(
                sGate.reshaped(to: [sGate.elementCount]).slicedRows(start: 0, count: 4),
                into: d0.slicedRows(start: 40, count: 4), on: shexpCmd)
            Ops.copy(
                sUp.reshaped(to: [sUp.elementCount]).slicedRows(start: 0, count: 4),
                into: d0.slicedRows(start: 44, count: 4), on: shexpCmd)
            Ops.copy(
                sInner.reshaped(to: [sInner.elementCount]).slicedRows(start: 0, count: 4),
                into: d0.slicedRows(start: 48, count: 4), on: shexpCmd)
        }
        DeepSeekV4Model.commitWithProfile(shexpCmd, tag: "shexp")
        // FUSED gather path: one inline-dequant gather GEMV per role
        // (gate, up) computing all topK experts' outputs in a single
        // dispatch each — replaces 2*topK per-expert {dequant,gemv} cmd
        // buffers (12/layer) with 2. gate/up share x=xNorm.
        let useGather = topK == 6 && env["FFAI_DSV4_GATHER"] != "0"
        let useResident = env["FFAI_DSV4_RESIDENT"] != "0"
        // Build a u32 expert_ids tensor for a role's gather dispatch.
        func eids(_ slots: [Int]) -> Tensor {
            let t = Tensor.empty(shape: [slots.count], dtype: .u32, device: device)
            t.copyIn(from: slots.map { UInt32($0) })
            return t
        }
        // gate/up: resident packed pool (slots) when it fits, else
        // per-token staging (contiguous → identity ids).
        func gatherIQ2(_ name: String, _ tag: String) throws
            -> (qs: Tensor, d: Tensor, mOut: Int, kIn: Int, ids: Tensor)
        {
            if useResident,
                let r = try bundle.residentGatherIQ2XXS(
                    named: name, expertIndices: topIndices, nExperts: nExperts, device: device)
            {
                return (r.qsAll, r.dAll, r.split.mOut, r.split.kIn, eids(r.slots))
            }
            let g = try bundle.stageGatherIQ2XXS(
                named: name, expertIndices: topIndices, nExperts: nExperts, slot: tag, device: device)
            return (g.qsAll, g.dAll, g.mOut, g.kIn, eids(Array(0 ..< topK)))
        }
        if useGather {
            let (grid, signs) = GGUFDequant.iq2xxsTables(device: device)
            // ── gate ──
            let gG = try gatherIQ2(gateName, "gather_gate_L\(layer.layerIndex)")
            let gateAll = Tensor.empty(shape: [topK * gG.mOut], dtype: dt)
            let cmdGate = device.makeCommandBuffer()
            Ops.moeGatherGemvIQ2XXS(
                x: xNorm, qsAll: gG.qs, dAll: gG.d, expertIds: gG.ids,
                grid: grid, signs: signs,
                nSlots: topK, mOut: gG.mOut, kIn: gG.kIn, on: cmdGate, into: gateAll)
            DeepSeekV4Model.commitWithProfile(cmdGate, tag: "expert_gate")
            // ── up ──
            let gU = try gatherIQ2(upName, "gather_up_L\(layer.layerIndex)")
            let upAll = Tensor.empty(shape: [topK * gU.mOut], dtype: dt)
            let cmdUp = device.makeCommandBuffer()
            Ops.moeGatherGemvIQ2XXS(
                x: xNorm, qsAll: gU.qs, dAll: gU.d, expertIds: gU.ids,
                grid: grid, signs: signs,
                nSlots: topK, mOut: gU.mOut, kIn: gU.kIn, on: cmdUp, into: upAll)
            DeepSeekV4Model.commitWithProfile(cmdUp, tag: "expert_up")
            for k in 0 ..< topK {
                gateOuts[k] = gateAll.slicedRows(start: k * gG.mOut, count: gG.mOut).reshaped(to: [gG.mOut])
                upOuts[k] = upAll.slicedRows(start: k * gU.mOut, count: gU.mOut).reshaped(to: [gU.mOut])
            }
        } else {
            // 3-stage pipeline: gate / up / swiglu / down for each expert
            // distributed across pipeGate / pipeUp / pipeAB / pipeB.
            for k in 0 ..< topK {
                try recordExpertGate(k)
                DeepSeekV4Model.commitWithProfile(pipeGate[k], tag: "expert_gate")
                try recordExpertUp(k)
                DeepSeekV4Model.commitWithProfile(pipeUp[k], tag: "expert_up")
            }
        }
        let cmd2 = device.makeCommandBuffer()
        if useGather {
            // SwiGLU all topK experts into one contiguous inner buffer,
            // then a single Q2_K gather down-projection + router-weighted
            // sum writes moeAccum directly — replaces topK×{swiglu, dequant,
            // gemv} + the topK-way weighted accumulate with 2 dispatches.
            let innerAll = Tensor.empty(shape: [topK * intermediate], dtype: dt)
            var innerSlices: [Tensor] = []
            innerSlices.reserveCapacity(topK)
            for k in 0 ..< topK {
                innerSlices.append(
                    innerAll.slicedRows(start: k * intermediate, count: intermediate)
                        .reshaped(to: [intermediate]))
            }
            let cmdSwiglu = device.makeCommandBuffer()
            Ops.swigluLimitMany(
                gates: (0 ..< topK).map { gateOuts[$0]! },
                ups: (0 ..< topK).map { upOuts[$0]! },
                outs: innerSlices, limit: textConfig.swigluLimit, on: cmdSwiglu)
            DeepSeekV4Model.commitWithProfile(cmdSwiglu, tag: "expert_swiglu")
            let wT = Tensor.empty(shape: [topK], dtype: .f32, device: device)
            wT.copyIn(from: normWeights)
            if useResident,
                let rD = try bundle.residentGatherQ2K(
                    named: downName, expertIndices: topIndices, nExperts: nExperts, device: device)
            {
                Ops.moeGatherDownQ2K(
                    innersAll: innerAll, qsAll: rD.qsAll, scalesAll: rD.scalesAll,
                    dAll: rD.dAll, dminAll: rD.dminAll, expertIds: eids(rD.slots), weights: wT,
                    nSlots: topK, mOut: rD.split.mOut, kIn: rD.split.kIn, on: cmd2, into: moeAccum)
            } else {
                let gD = try bundle.stageGatherQ2K(
                    named: downName, expertIndices: topIndices, nExperts: nExperts,
                    slot: "gather_down_L\(layer.layerIndex)", device: device)
                Ops.moeGatherDownQ2K(
                    innersAll: innerAll, qsAll: gD.qsAll, scalesAll: gD.scalesAll,
                    dAll: gD.dAll, dminAll: gD.dminAll, expertIds: eids(Array(0 ..< topK)), weights: wT,
                    nSlots: topK, mOut: gD.mOut, kIn: gD.kIn, on: cmd2, into: moeAccum)
            }
        } else if useFusedDown {
            try recordFusedRoutedDown(on: cmd2)
        } else if useFusedEpilogue {
            try recordFusedEpilogue(on: cmd2)
        } else {
            for k in 0 ..< (topK - 1) {
                try recordExpertB(k, on: pipeB[k])
                DeepSeekV4Model.commitWithProfile(pipeB[k], tag: "expert_down")
            }
            try recordExpertB(topK - 1, on: cmd2)
        }
        let _tExpertRecord = CACurrentMediaTime()
        DeepSeekV4Model.profExpertRecord += _tExpertRecord - _tRouter
        let blockOut = Ops.add(moeAccum, shexpOut, on: cmd2)
        // mHC expand
        let newH = Ops.dsv4MhcExpand(
            blockOut: blockOut, post: postFfn, comb: combFfn,
            residualState: state.hcState,
            hiddenDim: hidden, nHc: 4, nTokens: 1, on: cmd2)
        if let d0 = dbgL0, layer.layerIndex == 0, d0.elementCount >= 40 {
            Ops.copy(moeAccum.slicedRows(start: 0, count: 4), into: d0.slicedRows(start: 24, count: 4), on: cmd2)
            Ops.copy(
                shexpOut.reshaped(to: [hidden]).slicedRows(start: 0, count: 4),
                into: d0.slicedRows(start: 28, count: 4), on: cmd2)
            Ops.copy(
                blockOut.reshaped(to: [hidden]).slicedRows(start: 0, count: 4),
                into: d0.slicedRows(start: 32, count: 4), on: cmd2)
            Ops.copy(
                newH.reshaped(to: [4 * hidden]).slicedRows(start: 0, count: 4),
                into: d0.slicedRows(start: 36, count: 4), on: cmd2)
        }
        // Fold the hcState copy-to-persistent into cmd2 — saves a
        // commit+wait round-trip per layer (~1 ms each × 43 layers).
        if let outHc = outHc {
            Ops.copy(newH, into: outHc, on: cmd2)
        }
        let _tBeforeCommit = CACurrentMediaTime()
        DeepSeekV4Model.profSharedRecord += _tBeforeCommit - _tExpertRecord
        DeepSeekV4Model.commitWithProfile(cmd2, tag: "ffn_final")
        // Drop the per-layer waitUntilCompleted — next layer's attn
        // cmd buffer queues on the same queue and is serialized
        // automatically. The terminal lmHead wait synchronizes
        // everything before host readback. state.hcState is reassigned
        // to outHc (persistent buffer), not the scratch-resident newH,
        // so withScratch's resetScratch at the layer body exit doesn't
        // invalidate cross-layer carry-over state.
        let _tAfterWait = CACurrentMediaTime()
        DeepSeekV4Model.profGpuWait += _tAfterWait - _tBeforeCommit
        state.hcState = outHc ?? newH
        return blockOut
    }

    /// Sync-free GPU-routed FFN tail. Routing (top-K + weights), the
    /// raw→slot remaps, the IQ2 gate/up gathers, SwiGLU, the Q2_K down
    /// gather+weighted-sum, the shared expert, and the mHC expand all
    /// run on the queue with NO `waitUntilCompleted` — the only sync per
    /// token is the terminal lmHead readback. Requires the resident
    /// pools (`rg`/`ru`/`rd`) to already hold the routed experts (the
    /// first token's CPU path fills them).
    private func gpuRoutedFfnTail(
        layer: DeepSeekV4Layer, state: DecodeState, xNorm: Tensor,
        scoreBiased: Tensor, scoreUnbiased: Tensor,
        rg: ResidentIQ2Split, ru: ResidentIQ2Split, rd: ResidentQ2KSplit,
        nExperts: Int, topK: Int, hidden: Int, intermediate: Int, dt: DType,
        postFfn: Tensor, combFfn: Tensor, attnCmd: MTLCommandBuffer,
        copyHcInto outHc: Tensor?
    ) throws -> Tensor {
        let (grid, signs) = GGUFDequant.iq2xxsTables(device: device)
        let cap = RESIDENT_POOL_CAP
        func smap(_ b: MTLBuffer) -> Tensor { Tensor(buffer: b, offset: 0, shape: [nExperts], dtype: .u32) }

        // Router top-K + weights on the attn cmd buffer; commit, NO wait.
        let rawIdx = Tensor.empty(shape: [topK], dtype: .u32)
        let gpuW = Tensor.empty(shape: [topK], dtype: .f32)
        Ops.dsv4RouterTopK(
            scoreBiased: scoreBiased, scoreUnbiased: scoreUnbiased,
            indicesOut: rawIdx, weightsOut: gpuW, nExperts: nExperts, k: topK, on: attnCmd)
        // The router kernel renormalizes the chosen weights to sum-to-1 but
        // does NOT apply routed_scaling_factor (it has no scaling input). Apply
        // it here — final_weight_k = scaling * unbiased_k / Σ unbiased — to match
        // the CPU decode/prefill paths. It MUST multiply AFTER the sum-to-1
        // renorm; folding it in before would cancel in the division (the bug
        // that left the GPU router path 1/scaling× too small).
        let scaling = textConfig.routerScalingFactor
        let scaleVec = Tensor.filled(scaling, shape: [topK], dtype: .f32, device: device)
        let gpuWScaled = Ops.mul(gpuW, scaleVec, on: attnCmd)
        DeepSeekV4Model.commitWithProfile(attnCmd, tag: "attn+router")

        // Shared expert — independent of routing, commit first. Q8 gemv
        // straight from resident Q8 (shexp gate/up/down are Q8_0).
        let shexpCmd = device.makeCommandBuffer()
        let q8on = ProcessInfo.processInfo.environment["FFAI_DSV4_Q8ATTN"] != "0"
        let li = layer.layerIndex
        let sGate: Tensor; let sUp: Tensor
        if q8on, let g = try? bundle.residentQ8("blk.\(li).ffn_gate_shexp.weight", device: device),
            let u = try? bundle.residentQ8("blk.\(li).ffn_up_shexp.weight", device: device)
        {
            sGate = Tensor.empty(shape: [g.mOut], dtype: dt); Ops.gemvQ8(q8: g, x: xNorm, on: shexpCmd, into: sGate)
            sUp = Tensor.empty(shape: [u.mOut], dtype: dt); Ops.gemvQ8(q8: u, x: xNorm, on: shexpCmd, into: sUp)
        } else {
            sGate = Ops.gemv(weight: layer.ffnGateShexp.asGgufMatmulWeight(), input: xNorm, on: shexpCmd)
            sUp = Ops.gemv(weight: layer.ffnUpShexp.asGgufMatmulWeight(), input: xNorm, on: shexpCmd)
        }
        let sInner = Ops.swigluLimit(gate: sGate, up: sUp, limit: textConfig.swigluLimit, on: shexpCmd)
        let shexpOut: Tensor
        if q8on, let d = try? bundle.residentQ8("blk.\(li).ffn_down_shexp.weight", device: device) {
            shexpOut = Tensor.empty(shape: [d.mOut], dtype: dt);
            Ops.gemvQ8(q8: d, x: sInner, on: shexpCmd, into: shexpOut)
        } else {
            shexpOut = Ops.gemv(weight: layer.ffnDownShexp.asGgufMatmulWeight(), input: sInner, on: shexpCmd)
        }
        DeepSeekV4Model.commitWithProfile(shexpCmd, tag: "shexp")

        // All routed-expert work shares ONE cmd buffer (the chain
        // gate/up → swiglu → down is sequential anyway, so merging loses
        // no GPU parallelism but saves ~4 commits/layer of CPU overhead).
        let ffnCmd = device.makeCommandBuffer()
        // gate
        let gateIds = Tensor.empty(shape: [topK], dtype: .u32)
        let gateAll = Tensor.empty(shape: [topK * rg.mOut], dtype: dt)
        Ops.remapU32(table: smap(rg.slotmap), idx: rawIdx, out: gateIds, n: topK, on: ffnCmd)
        Ops.moeGatherGemvIQ2XXS(
            x: xNorm,
            qsAll: Tensor(buffer: rg.qs, offset: 0, shape: [cap * rg.nBlocksPerExpert * 16], dtype: .u32),
            dAll: Tensor(buffer: rg.d, offset: 0, shape: [cap * rg.nBlocksPerExpert], dtype: .f32),
            expertIds: gateIds, grid: grid, signs: signs,
            nSlots: topK, mOut: rg.mOut, kIn: rg.kIn, on: ffnCmd, into: gateAll)
        // up
        let upIds = Tensor.empty(shape: [topK], dtype: .u32)
        let upAll = Tensor.empty(shape: [topK * ru.mOut], dtype: dt)
        Ops.remapU32(table: smap(ru.slotmap), idx: rawIdx, out: upIds, n: topK, on: ffnCmd)
        Ops.moeGatherGemvIQ2XXS(
            x: xNorm,
            qsAll: Tensor(buffer: ru.qs, offset: 0, shape: [cap * ru.nBlocksPerExpert * 16], dtype: .u32),
            dAll: Tensor(buffer: ru.d, offset: 0, shape: [cap * ru.nBlocksPerExpert], dtype: .f32),
            expertIds: upIds, grid: grid, signs: signs,
            nSlots: topK, mOut: ru.mOut, kIn: ru.kIn, on: ffnCmd, into: upAll)

        // SwiGLU all experts → one contiguous inner buffer.
        let innerAll = Tensor.empty(shape: [topK * intermediate], dtype: dt)
        var gates: [Tensor] = []; var ups: [Tensor] = []; var inners: [Tensor] = []
        for k in 0 ..< topK {
            gates.append(gateAll.slicedRows(start: k * rg.mOut, count: rg.mOut).reshaped(to: [rg.mOut]))
            ups.append(upAll.slicedRows(start: k * ru.mOut, count: ru.mOut).reshaped(to: [ru.mOut]))
            inners.append(
                innerAll.slicedRows(start: k * intermediate, count: intermediate).reshaped(to: [intermediate]))
        }
        Ops.swigluLimitMany(gates: gates, ups: ups, outs: inners, limit: textConfig.swigluLimit, on: ffnCmd)

        // Down gather + router-weighted sum → moeAccum.
        let moeAccum = Tensor.filled(0.0, shape: [hidden], dtype: dt, device: device)
        let downIds = Tensor.empty(shape: [topK], dtype: .u32)
        let cmd2 = ffnCmd
        Ops.remapU32(table: smap(rd.slotmap), idx: rawIdx, out: downIds, n: topK, on: cmd2)
        Ops.moeGatherDownQ2K(
            innersAll: innerAll,
            qsAll: Tensor(buffer: rd.qs, offset: 0, shape: [cap * rd.nBlocksPerExpert * 16], dtype: .u32),
            scalesAll: Tensor(buffer: rd.scales, offset: 0, shape: [cap * rd.nBlocksPerExpert * 16], dtype: .u8),
            dAll: Tensor(buffer: rd.d, offset: 0, shape: [cap * rd.nBlocksPerExpert], dtype: .f32),
            dminAll: Tensor(buffer: rd.dmin, offset: 0, shape: [cap * rd.nBlocksPerExpert], dtype: .f32),
            expertIds: downIds, weights: gpuWScaled,
            nSlots: topK, mOut: rd.mOut, kIn: rd.kIn, on: cmd2, into: moeAccum)

        let blockOut = Ops.add(moeAccum, shexpOut, on: cmd2)
        let newH = Ops.dsv4MhcExpand(
            blockOut: blockOut, post: postFfn, comb: combFfn,
            residualState: state.hcState, hiddenDim: hidden, nHc: 4, nTokens: 1, on: cmd2)
        if let outHc = outHc { Ops.copy(newH, into: outHc, on: cmd2) }
        DeepSeekV4Model.commitWithProfile(cmd2, tag: "ffn_final")
        state.hcState = outHc ?? newH
        return blockOut
    }
    nonisolated(unsafe) public static var profRouter: Double = 0
    nonisolated(unsafe) public static var profExpertRecord: Double = 0
    nonisolated(unsafe) public static var profSharedRecord: Double = 0
    nonisolated(unsafe) public static var profGpuWait: Double = 0
    nonisolated(unsafe) public static var profDequant: Double = 0
    nonisolated(unsafe) public static var profExpertGemv: Double = 0
    nonisolated(unsafe) public static var profExpertEpilogue: Double = 0

    // Per-cmd-buffer GPU-time profile. Keyed by tag set on MTLCommandBuffer.label
    // before commit. Updated in the completion handler from gpuStartTime/
    // gpuEndTime so we get TRUE GPU runtime per stage (not CPU wait time).
    nonisolated(unsafe) public static var profGpuByTag: [String: Double] = [:]
    nonisolated(unsafe) public static var profCountByTag: [String: Int] = [:]
    nonisolated(unsafe) public static var profKernelEncodeCount: Int = 0
    nonisolated(unsafe) public static let profGpuLock = NSLock()
    public static let profileEnabled =
        ProcessInfo.processInfo.environment["FFAI_DSV4_PROFILE"] == "1"

    public static func resetGpuProf() {
        profGpuLock.lock()
        defer { profGpuLock.unlock() }
        profGpuByTag.removeAll(keepingCapacity: true)
        profCountByTag.removeAll(keepingCapacity: true)
        profKernelEncodeCount = 0
    }

    /// Helper: label cmd buffer + install completion handler that
    /// records true GPU runtime, then commit. Aggregates into
    /// `profGpuByTag[tag]`.
    public static func commitWithProfile(_ cmd: MTLCommandBuffer, tag: String) {
        cmd.label = tag
        guard profileEnabled else {
            cmd.commit()
            return
        }
        cmd.addCompletedHandler { cb in
            let gpu = cb.gpuEndTime - cb.gpuStartTime
            profGpuLock.lock()
            profGpuByTag[tag, default: 0] += gpu
            profCountByTag[tag, default: 0] += 1
            profGpuLock.unlock()
        }
        cmd.commit()
    }

    /// Full single-token decode forward through all `nLayers` layers
    /// + output mHC head + output norm + LM head. Returns the logits
    /// vector `[vocab]`.
    ///
    /// **Note:** CSA / HCA forward paths aren't implemented yet, so
    /// `forwardFullAttnSubblock` is used on ALL layers regardless of
    /// `compress_ratio`. The output is dispatch-correct but the
    /// numerics for CSA/HCA layers are wrong (they should run the
    /// indexer + compressed-cache attention). Quality of the
    /// generated token will be garbage until those paths land.
    public func forwardAllLayers(
        inputTokenId: Int, state: DecodeState
    ) throws -> Tensor {
        let cfg = textConfig
        let dt = activationDtype
        let hidden = cfg.hidden
        state.currentToken = inputTokenId  // hash-routed early layers key on this

        // Seed hcState with the input token's embedding broadcast
        // across all 4 mHC channels.
        let embedRow = tokenEmbd.asGgufMatmulWeight()
            .slicedRows(start: inputTokenId, count: 1).reshaped(to: [hidden])
        let cmdSeed = device.makeCommandBuffer()
        for c in 0 ..< 4 {
            let dst = state.hcState.slicedRows(start: c, count: 1).reshaped(to: [hidden])
            Ops.copy(embedRow, into: dst, on: cmdSeed)
        }
        DeepSeekV4Model.commitWithProfile(cmdSeed, tag: "seed")
        // No wait — the first layer's attn reads hcState on the same
        // queue, which serializes after the seed copy.

        // Iterate layers wrapped in withScratch so the per-layer
        // transient tensors all flow through the device scratch slab
        // and get reset between layers. Without this, ~100
        // Tensor.empty calls per layer × 43 layers hammered Metal's
        // driver pool and RSS grew ~3 GB/min until OOM.
        //
        // CARRY-OVER STATE NOTE: state.hcState is the only tensor
        // that must persist ACROSS layer boundaries. The mHC expand
        // step at the end of each sub-block writes a new hcState
        // tensor — inside withScratch that lands in the slab, then
        // we COPY it into a persistent buffer before the scope exit
        // so the next layer reads from real memory, not the
        // about-to-be-reset slab.
        let hcStatePersistent = Tensor.empty(
            shape: [4, hidden], dtype: dt, device: device)
        var tLoad = 0.0
        var tAttn = 0.0
        var tFfn = 0.0
        var tCopy = 0.0
        var tRelease = 0.0
        for layerIdx in 0 ..< cfg.nLayers {
            let t0 = CACurrentMediaTime()
            let layer = try self.layer(layerIdx)
            let t1 = CACurrentMediaTime()
            try autoreleasepool {
                try device.withScratch {
                    // Single cmd buffer for attn + ffn prefix — the FFN
                    // top-K readback inside forwardFfnSubblock commits+waits
                    // the shared buffer once, instead of attn doing a
                    // separate commit+wait of its own.
                    let cmdAttn = device.makeCommandBuffer()
                    _ = forwardFullAttnSubblock(layer: layer, state: state, on: cmdAttn)
                    let t2 = CACurrentMediaTime()
                    _ = try forwardFfnSubblock(
                        layer: layer, state: state, on: cmdAttn,
                        copyHcInto: hcStatePersistent)
                    let t3 = CACurrentMediaTime()
                    let t4 = CACurrentMediaTime()
                    tAttn += t2 - t1
                    tFfn += t3 - t2
                    tCopy += t4 - t3
                }
            }
            let t5 = CACurrentMediaTime()
            if !keepLayersResident {
                self.releaseLayer(layerIdx)
            }
            let t6 = CACurrentMediaTime()
            tLoad += t1 - t0
            tRelease += t6 - t5
        }
        if DeepSeekV4Model.profileEnabled {
            print(
                String(
                    format: "[prof] load=%.2fs attn=%.2fs ffn=%.2fs copy=%.2fs release=%.2fs",
                    tLoad, tAttn, tFfn, tCopy, tRelease))
            print(
                String(
                    format: "[prof-ffn] router-host-wait=%.2fs expert-record=%.2fs shared-record=%.2fs gpu-wait=%.2fs",
                    DeepSeekV4Model.profRouter,
                    DeepSeekV4Model.profExpertRecord,
                    DeepSeekV4Model.profSharedRecord,
                    DeepSeekV4Model.profGpuWait))
            print(
                String(
                    format: "[prof-expert] dequant=%.2fs expert-gemv=%.2fs epilogue=%.2fs",
                    DeepSeekV4Model.profDequant,
                    DeepSeekV4Model.profExpertGemv,
                    DeepSeekV4Model.profExpertEpilogue))
            print(
                String(
                    format: "[prof-slice] tables=%.2fs pooled=%.2fs wrslice=%.2fs dequant=%.2fs q80=%.2fs q2k=%.2fs",
                    GGUFTensorBundle.profSliceTables,
                    GGUFTensorBundle.profSlicePooled,
                    GGUFTensorBundle.profSliceWrslice,
                    GGUFTensorBundle.profSliceDequant,
                    GGUFTensorBundle.profSliceQ80,
                    GGUFTensorBundle.profSliceQ2K))
            print("[prof-slice-type] \(GGUFTensorBundle.profSliceType)")
            // Per-cmd-buffer GPU runtime (true GPU time, NOT CPU wait).
            // Populated via commitWithProfile completion handlers.
            DeepSeekV4Model.profGpuLock.lock()
            let gpuTags = DeepSeekV4Model.profGpuByTag.sorted { $0.value > $1.value }
            let counts = DeepSeekV4Model.profCountByTag
            DeepSeekV4Model.profGpuLock.unlock()
            let totalGpu = gpuTags.reduce(0.0) { $0 + $1.value }
            print(
                String(
                    format: "[prof-gpu] total=%.4fs across %d cmd buffers",
                    totalGpu, counts.values.reduce(0, +)))
            for (tag, gpu) in gpuTags {
                let n = counts[tag] ?? 0
                let avgMs = n > 0 ? (gpu / Double(n)) * 1000.0 : 0
                print(
                    String(
                        format: "[prof-gpu]   %@: %.4fs / %d cmds (avg %.3f ms)",
                        tag, gpu, n, avgMs))
            }
            print(
                String(
                    format: "[prof-stage] iq2_stage=%.3fs iq2_encode=%.3fs",
                    GGUFDequant.profStageIq2, GGUFDequant.profEncodeIq2))
        }
        // Output mHC head: pre = sigmoid(output_hc_fn^T @ flatten(H)
        // * scale + base) + eps → [4]
        let flatH = state.hcState.reshaped(to: [4 * hidden])
        let cmdHead = device.makeCommandBuffer()
        let outputHcFnW = outputHcFn.asGgufMatmulWeight()
        let pre4 = mhcMix(flatH: flatH, fnWeight: outputHcFnW, on: cmdHead)
        DeepSeekV4Model.commitWithProfile(cmdHead, tag: "output_head_pre4")
        cmdHead.waitUntilCompleted()
        let pre4Host = pre4.toArray(as: Float.self)
        let scaleHost = outputHcScale.toArray(as: Float.self)
        let baseHost = outputHcBase.toArray(as: Float.self)
        let eps = cfg.hcEpsilon
        var preFinal = [Float](repeating: 0, count: 4)
        for c in 0 ..< 4 {
            let z = pre4Host[c] * scaleHost[0] + baseHost[c]
            preFinal[c] = 1.0 / (1.0 + Foundation.exp(-z)) + eps
        }
        let preTensor = Tensor.empty(shape: [4], dtype: .f32)
        preTensor.copyIn(from: preFinal)

        // Collapse H → x using preFinal, then LM head gemv.
        let cmdCollapse = device.makeCommandBuffer()
        let xWithTokens = Ops.dsv4MhcCollapse(
            state: state.hcState, pre: preTensor,
            hiddenDim: hidden, nHc: 4, nTokens: 1, outDtype: dt, on: cmdCollapse)
        let x = xWithTokens.reshaped(to: [hidden])
        let xNorm = Ops.rmsNorm(x, weight: outputNorm, eps: cfg.rmsNormEps, on: cmdCollapse)
        // LM head (output.weight is Q8_0, ~1 GB as f16) — gemv from
        // resident Q8 to halve the per-token output-projection bandwidth.
        let logits: Tensor
        if ProcessInfo.processInfo.environment["FFAI_DSV4_Q8ATTN"] != "0",
            let lmQ8 = try? bundle.residentQ8("output.weight", device: device)
        {
            logits = Tensor.empty(shape: [lmQ8.mOut], dtype: dt)
            Ops.gemvQ8(q8: lmQ8, x: xNorm, on: cmdCollapse, into: logits)
        } else {
            logits = Ops.gemv(weight: outputHead.asGgufMatmulWeight(), input: xNorm, on: cmdCollapse)
        }
        DeepSeekV4Model.commitWithProfile(cmdCollapse, tag: "lmhead+collapse")
        cmdCollapse.waitUntilCompleted()
        return logits
    }
}
