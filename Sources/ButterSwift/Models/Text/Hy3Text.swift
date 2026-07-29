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
// Hy3 text decoder — Tencent Hunyuan Hy3 (`hy_v3`).
//
// Path B router (`MoEGatingMode.sigmoidBiasedTopK`), ungated shared SwiGLU
// expert, and QK-RMSNorm before RoPE. Experts are stacked affine-quantized
// tensors (`mlp.switch_mlp.{gate,up,down}_proj`, shape [E, M, packed_K]).
// Decode uses on-device Path B routing + int2 affine expert-indexed GEMV.
// MTP (`num_nextn_predict_layers`) is not loaded yet.
//
// ── BUTTER_HY3_* probe / tuning surface ──────────────────────────────────
//
// Every `BUTTER_HY3_*` env var this family (and the Hy3-specific dispatches
// in Ops.swift) reads, in one place so a probe doesn't require grepping
// the whole file. All are off/default unless noted.
//
//   BUTTER_HY3_MAX_SEQ=N           Override the default clamped context
//                                length (see Hy3Dense.loadModel).
//   BUTTER_HY3_BF16=1              Keep checkpoint bf16 scales/biases
//                                as-is (skip the in-place → f16 cast).
//   BUTTER_HY3_DENSIFY_INT8=1      Densify int8 attn/shared weights for
//                                decode gemv (~10 GB extra).
//   BUTTER_HY3_SERIAL_QKV=1        Serial QKV callMany (no concurrent
//                                encoder).
//   BUTTER_HY3_SDPA_PREFILL_MMA=1  Force the MMA-based SDPA prefill path.
//   BUTTER_HY3_NO_SDPA_PREFILL_MMA=1  Disable it (default: on for nRows≥32).
//   BUTTER_HY3_DUAL_QUEUE=1        Dual-queue shared-expert / MoE overlap.
//   BUTTER_HY3_NO_DUAL=1           Force serial shared-expert after MoE.
//   BUTTER_HY3_SKIP_ATTN=1         Zero QKV/SDPA/O (free-path ceiling probe).
//   BUTTER_HY3_SKIP_SDPA=1         Keep QKV/O but zero the SDPA body.
//   BUTTER_HY3_SKIP_MOE=1          Zero the routed MoE path (profile
//                                residual = attn + shared only).
//   BUTTER_HY3_SKIP_SHARED=1       Zero the shared-expert path.
//   BUTTER_HY3_MOE_M1=1            Force the reduction-style (M1) gather
//                                instead of the tiled-BM dispatch.
//   BUTTER_HY3_MOE_SERIAL_GU=1     Serial gate/up gather (no concurrent
//                                encoder).
//   BUTTER_HY3_DUAL_MOE_GU=1       Dual-queue gate/up (mutually exclusive
//                                with BUTTER_HY3_DUAL_QUEUE).
//   BUTTER_HY3_MOE_BM64=1          Opt-in BM64 MPP gather tile.
//   BUTTER_HY3_MOE_BM8=1           Opt-in BM8 MPP gather tile.
//   BUTTER_HY3_MOE_MMA=1           Opt-in simdgroup-MMA BM16 gather.
//   BUTTER_HY3_EXPERT_GRID=1       Force the expert-outer grid dispatch.
//   BUTTER_HY3_NO_EXPERT_GRID=1    Disable it (default: on for mTotal≥1536).
//   BUTTER_HY3_ACTIVE_EG=1         Active-only expert-outer gather via
//                                compact + indirect dispatch.
//   BUTTER_HY3_MOE_SKIP_GU=1       Zero the gate/up gather output (probe).
//   BUTTER_HY3_MOE_SKIP_DOWN=1     Zero the down-projection output (probe).
//   BUTTER_HY3_SCRATCH=1           Opt-in grow-only prefill scratch slab
//                                (default off — regressed e2e prefill).
//   BUTTER_HY3_MEGA_CB=1           One command buffer for all layers.
//   BUTTER_HY3_CB_STRIDE=N         Commit every N layers (default 4).
//   BUTTER_HY3_PROFILE=1           Emit `[hy3-profile]` per-call timing.
//   BUTTER_HY3_QMM_MMA=1           Use the MMA QMM path instead of the
//                                default MPP/NAX int8 live-compile path
//                                (Ops.swift shared/expert-indexed GEMV).
//   BUTTER_HY3_SERIAL_EXPERT_GEMV=1  Force a serial encoder for
//                                  expert-indexed GEMV (Ops.swift).

import Foundation
import Metal

// ─── Variant ─────────────────────────────────────────────────────────

public enum Hy3Dense: Hy3Variant {
    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Hy3Model {
        let tc = config
        guard let nLayers = tc.int("num_hidden_layers") else {
            throw Hy3Error.missingConfig("num_hidden_layers")
        }
        guard let hidden = tc.int("hidden_size") else {
            throw Hy3Error.missingConfig("hidden_size")
        }
        guard let vocab = tc.int("vocab_size") else {
            throw Hy3Error.missingConfig("vocab_size")
        }
        guard let nHeads = tc.int("num_attention_heads") else {
            throw Hy3Error.missingConfig("num_attention_heads")
        }
        let nKVHeads = tc.int("num_key_value_heads") ?? nHeads
        let headDim = tc.int("head_dim") ?? (hidden / nHeads)
        let intermediate = tc.int("intermediate_size") ?? (hidden * 4)
        let moeIntermediate =
            tc.int("moe_intermediate_size")
            ?? tc.int("expert_hidden_dim")
            ?? intermediate
        let nExperts = tc.int("num_experts") ?? 0
        let topK = tc.int("num_experts_per_tok") ?? 8
        let nShared = tc.int("num_shared_experts") ?? 1
        let firstKDense = tc.int("first_k_dense_replace") ?? 1
        // Checkpoint advertises a 262144 window. Full KV at that depth is
        // ~86 GB for Hy3 (80 layers, 8 KV heads, d=128, f16 K+V) — unusable
        // next to the 92 GB oQ2 weights on a 128 GB Mac. Default 2048
        // (~0.67 GB) keeps SDPA/KV strides tight for short prefill; override
        // with LoadOptions.maxContextLength or BUTTER_HY3_MAX_SEQ for longer.
        let defaultMaxSeq: Int = {
            if let s = ProcessInfo.processInfo.environment["BUTTER_HY3_MAX_SEQ"],
               let n = Int(s), n > 0
            {
                return n
            }
            return 2048
        }()
        let maxPos = tc.int("max_position_embeddings")
        let maxSeq =
            options.maxContextLength
            ?? min(maxPos ?? defaultMaxSeq, defaultMaxSeq)
        if let maxPos, maxSeq < maxPos {
            fputs(
                "[hy3] context clamped to \(maxSeq) (checkpoint advertises \(maxPos)); "
                    + "raise with LoadOptions.maxContextLength or BUTTER_HY3_MAX_SEQ\n",
                stderr)
        }
        let eps = Float(tc.float("rms_norm_eps") ?? 1e-5)
        let routeNorm = tc.bool("route_norm") ?? true
        let routerScale = Float(tc.float("router_scaling_factor") ?? 1.0)
        let useQkNorm = tc.bool("qk_norm") ?? true

        // rope_parameters.rope_theta (Hy3) or rope_theta fallback
        var ropeTheta: Float = 10_000
        if let rp = tc.raw["rope_parameters"] as? [String: Any],
            let t = rp["rope_theta"] as? Double
        {
            ropeTheta = Float(t)
        } else if let t = tc.float("rope_theta") {
            ropeTheta = Float(t)
        }

        let quant = config.quantization
        let groupSize = quant?.groupSize ?? 64

        // Default f16 activations for oQ2. Checkpoint scales/biases are BF16;
        // promote with in-place GPU cast (same width).
        let preferF16 =
            ProcessInfo.processInfo.environment["BUTTER_HY3_BF16"] == nil
        let dtype: DType = preferF16 ? .f16 : (quant != nil ? .bf16 : .f16)
        let device = Device.shared
        // One CB for all in-place bf16→f16 casts at load (committed once).
        let castCmd: MTLCommandBuffer? = preferF16 ? device.makeCommandBuffer() : nil

        let embedTokens = try hy3MaybeF16Embedding(
            loadEmbedding(
                base: "model.embed_tokens", in: weights,
                hidden: hidden, quantization: quant),
            preferF16: preferF16, cmd: castCmd, device: device)

        var layers: [any Hy3LayerProtocol] = []
        layers.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let p = "model.layers.\(i)"
            let qProj = try hy3MaybeF16Linear(
                loadLinear(
                    base: "\(p).self_attn.q_proj", in: weights, quantization: quant),
                preferF16: preferF16, cmd: castCmd, device: device)
            let kProj = try hy3MaybeF16Linear(
                loadLinear(
                    base: "\(p).self_attn.k_proj", in: weights, quantization: quant),
                preferF16: preferF16, cmd: castCmd, device: device)
            let vProj = try hy3MaybeF16Linear(
                loadLinear(
                    base: "\(p).self_attn.v_proj", in: weights, quantization: quant),
                preferF16: preferF16, cmd: castCmd, device: device)
            let oProj = try hy3MaybeF16Linear(
                loadLinear(
                    base: "\(p).self_attn.o_proj", in: weights, quantization: quant),
                preferF16: preferF16, cmd: castCmd, device: device)

            let qNorm: RMSNorm?
            let kNorm: RMSNorm?
            if useQkNorm {
                qNorm = RMSNorm(
                    weight: hy3MaybeF16Tensor(
                        try weights.tensor(named: "\(p).self_attn.q_norm.weight"),
                        preferF16: preferF16, cmd: castCmd),
                    eps: eps)
                kNorm = RMSNorm(
                    weight: hy3MaybeF16Tensor(
                        try weights.tensor(named: "\(p).self_attn.k_norm.weight"),
                        preferF16: preferF16, cmd: castCmd),
                    eps: eps)
            } else {
                qNorm = nil
                kNorm = nil
            }

            let inputNorm = RMSNorm(
                weight: hy3MaybeF16Tensor(
                    try weights.tensor(named: "\(p).input_layernorm.weight"),
                    preferF16: preferF16, cmd: castCmd),
                eps: eps)
            let postAttnNorm = RMSNorm(
                weight: hy3MaybeF16Tensor(
                    try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                    preferF16: preferF16, cmd: castCmd),
                eps: eps)

            if i < firstKDense || nExperts == 0 {
                // Dense SwiGLU MLP.
                let gateProj = try hy3MaybeF16Linear(
                    loadLinear(
                        base: "\(p).mlp.gate_proj", in: weights, quantization: quant),
                    preferF16: preferF16, cmd: castCmd, device: device)
                let upProj = try hy3MaybeF16Linear(
                    loadLinear(
                        base: "\(p).mlp.up_proj", in: weights, quantization: quant),
                    preferF16: preferF16, cmd: castCmd, device: device)
                let downProj = try hy3MaybeF16Linear(
                    loadLinear(
                        base: "\(p).mlp.down_proj", in: weights, quantization: quant),
                    preferF16: preferF16, cmd: castCmd, device: device)
                layers.append(
                    Hy3DenseLayer(
                        qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                        qNorm: qNorm, kNorm: kNorm,
                        gateProj: gateProj, upProj: upProj, downProj: downProj,
                        inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                        hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                        headDim: headDim, ropeTheta: ropeTheta, dtype: dtype))
            } else {
                // MoE: router + stacked switch_mlp + ungated shared_mlp.
                let gate = try hy3MaybeF16Linear(
                    loadLinear(
                        base: "\(p).mlp.router.gate", in: weights, quantization: quant),
                    preferF16: preferF16, cmd: castCmd, device: device)
                let biasTensor = try weights.tensor(named: "\(p).mlp.router.expert_bias")
                precondition(
                    biasTensor.elementCount == nExperts,
                    "Hy3: expert_bias count \(biasTensor.elementCount) ≠ nExperts \(nExperts)")
                // CPU Path B fallback + MoERouter init still need host floats.
                let expertBiasHost = biasTensor.toFloatArray()
                // GPU Path B scores in activation dtype — convert once at load.
                let expertBiasGPU = hy3BiasTensor(
                    expertBiasHost, dtype: dtype, device: device)

                let switchGate = hy3MaybeF16Triplet(
                    try weights.quantizedTriplet("\(p).mlp.switch_mlp.gate_proj"),
                    preferF16: preferF16, cmd: castCmd)
                let switchUp = hy3MaybeF16Triplet(
                    try weights.quantizedTriplet("\(p).mlp.switch_mlp.up_proj"),
                    preferF16: preferF16, cmd: castCmd)
                let switchDown = hy3MaybeF16Triplet(
                    try weights.quantizedTriplet("\(p).mlp.switch_mlp.down_proj"),
                    preferF16: preferF16, cmd: castCmd)
                let gateBits = deriveAffineQuantBits(
                    weightPackedCols: switchGate.weight.shape[2],
                    scaleCols: switchGate.scales.shape[2],
                    groupSize: groupSize)
                let upBits = deriveAffineQuantBits(
                    weightPackedCols: switchUp.weight.shape[2],
                    scaleCols: switchUp.scales.shape[2],
                    groupSize: groupSize)
                let downBits = deriveAffineQuantBits(
                    weightPackedCols: switchDown.weight.shape[2],
                    scaleCols: switchDown.scales.shape[2],
                    groupSize: groupSize)

                let sharedGate = try hy3MaybeF16Linear(
                    loadLinear(
                        base: "\(p).mlp.shared_mlp.gate_proj", in: weights, quantization: quant),
                    preferF16: preferF16, cmd: castCmd, device: device)
                let sharedUp = try hy3MaybeF16Linear(
                    loadLinear(
                        base: "\(p).mlp.shared_mlp.up_proj", in: weights, quantization: quant),
                    preferF16: preferF16, cmd: castCmd, device: device)
                let sharedDown = try hy3MaybeF16Linear(
                    loadLinear(
                        base: "\(p).mlp.shared_mlp.down_proj", in: weights, quantization: quant),
                    preferF16: preferF16, cmd: castCmd, device: device)

                let router = MoERouter(
                    nExperts: nExperts, topK: topK,
                    gatingMode: .sigmoidBiasedTopK,
                    normTopKProb: routeNorm,
                    expertBias: expertBiasHost,
                    routedScalingFactor: routerScale)

                layers.append(
                    Hy3MoELayer(
                        qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                        qNorm: qNorm, kNorm: kNorm,
                        inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                        routerGate: gate, router: router,
                        expertBiasGPU: expertBiasGPU,
                        switchGate: switchGate, switchUp: switchUp, switchDown: switchDown,
                        gateBits: gateBits, upBits: upBits, downBits: downBits,
                        groupSize: groupSize,
                        sharedGate: sharedGate, sharedUp: sharedUp, sharedDown: sharedDown,
                        hidden: hidden, moeIntermediate: moeIntermediate,
                        nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                        ropeTheta: ropeTheta, dtype: dtype, nShared: nShared))
            }
        }

        let finalNorm = RMSNorm(
            weight: hy3MaybeF16Tensor(
                try weights.tensor(named: "model.norm.weight"),
                preferF16: preferF16, cmd: castCmd),
            eps: eps)

        let lmHead: AnyLinear
        if weights.has("lm_head.weight") {
            lmHead = try hy3MaybeF16Linear(
                loadLinear(base: "lm_head", in: weights, quantization: quant),
                preferF16: preferF16, cmd: castCmd, device: device)
        } else if !weights.isQuantized("model.embed_tokens") {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        } else {
            throw Hy3Error.missingTensor("lm_head.weight (tied quantized embed not supported)")
        }

        // Flush all in-place bf16→f16 casts before first generate.
        if let castCmd {
            if castCmd.status != .committed && castCmd.status != .completed {
                castCmd.commit()
            }
            castCmd.waitUntilCompleted()
            fputs("[hy3] promoted checkpoint BF16 scales/biases → f16 (GPU in-place)\n", stderr)
        }

        let model = Hy3Model(
            embedTokens: embedTokens, layers: layers, finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads, nKVHeads: nKVHeads,
            headDim: headDim, vocab: vocab, maxSeq: maxSeq, dtype: dtype)
        // Optional densify of int8 attn/shared for decode gemv (~10 GB).
        // Default OFF — thrashing on 128 GB with 75 GB oQ2 weights.
        if ProcessInfo.processInfo.environment["BUTTER_HY3_DENSIFY_INT8"] != nil {
            model.densifyInt8Linears(device: Device.shared)
        }
        return model
    }
}

// ─── Prefill helpers ─────────────────────────────────────────────────

/// int8 affine callMany via MPP when possible (pad M to 32); else callMany.
private func hy3Int8CallMany(
    _ lin: AnyLinear, _ x: Tensor, t: Int,
    on cmd: MTLCommandBuffer, device: Device
) -> Tensor {
    guard let ql = lin.inner as? QuantizedLinear, ql.bits == 8,
          ql.weight.shape[0] % 32 == 0, t > 0
    else {
        return lin.callMany(x, t: t, on: cmd, device: device)
    }
    let n = ql.weight.shape[0]
    let kIn = ql.scales.shape[ql.scales.shape.count - 1] * ql.groupSize
    let mPad = ((t + 31) / 32) * 32
    let dt = x.dtype
    let xIn: Tensor
    if mPad == t {
        xIn = x
    } else {
        let xPadBuf = device.intermediateScratch(
            tag: "hy3_int8_xpad_\(dt)", minBytes: mPad * kIn * dt.byteSize)
        xIn = Tensor(buffer: xPadBuf, offset: 0, shape: [mPad, kIn], dtype: dt)
        let valid = t * kIn * dt.byteSize
        let tail = (mPad - t) * kIn * dt.byteSize
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(
            from: x.buffer, sourceOffset: x.offset,
            to: xIn.buffer, destinationOffset: 0, size: valid)
        if tail > 0 {
            blit.fill(buffer: xIn.buffer, range: valid ..< (valid + tail), value: 0)
        }
        blit.endEncoding()
    }
    let outPad = hy3ScratchTensor(
        shape: [mPad, n], dtype: dt, tag: "hy3_int8_outpad_\(dt)_\(n)", device: device)
    guard let enc = cmd.makeComputeCommandEncoder() else {
        return lin.callMany(x, t: t, on: cmd, device: device)
    }
    Ops.encodeDequantGemmInt8Mpp(
        input: xIn, weight: ql.weight, scales: ql.scales, biases: ql.biases,
        m: mPad, n: n, k: kIn, groupSize: ql.groupSize, into: outPad, encoder: enc)
    enc.endEncoding()
    return mPad == t ? outPad : outPad.slicedRows(start: 0, count: t)
}

/// Concurrent int8 gate‖up (pad M to 32). Used by dense FFN and shared MLP.
private func hy3ConcurrentGateUp(
    ffnIn: Tensor, t: Int,
    gate: AnyLinear, up: AnyLinear,
    cmd: MTLCommandBuffer, device: Device
) -> (Tensor, Tensor) {
    guard let gL = gate.inner as? QuantizedLinear,
          let uL = up.inner as? QuantizedLinear,
          gL.bits == 8, uL.bits == 8, t > 0,
          gL.weight.shape[0] % 32 == 0, uL.weight.shape[0] % 32 == 0
    else {
        return (
            gate.callMany(ffnIn, t: t, on: cmd, device: device),
            up.callMany(ffnIn, t: t, on: cmd, device: device)
        )
    }
    let mPad = ((t + 31) / 32) * 32
    let kIn = gL.scales.shape[gL.scales.shape.count - 1] * gL.groupSize
    let dt = ffnIn.dtype
    let xIn: Tensor
    if mPad == t {
        xIn = ffnIn
    } else {
        let xPadBuf = device.intermediateScratch(
            tag: "hy3_gu_xpad_\(dt)", minBytes: mPad * kIn * dt.byteSize)
        xIn = Tensor(buffer: xPadBuf, offset: 0, shape: [mPad, kIn], dtype: dt)
        let valid = t * kIn * dt.byteSize
        let tail = (mPad - t) * kIn * dt.byteSize
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(
            from: ffnIn.buffer, sourceOffset: ffnIn.offset,
            to: xIn.buffer, destinationOffset: 0, size: valid)
        if tail > 0 {
            blit.fill(buffer: xIn.buffer, range: valid ..< (valid + tail), value: 0)
        }
        blit.endEncoding()
    }
    let gPad = hy3ScratchTensor(
        shape: [mPad, gL.weight.shape[0]], dtype: dt,
        tag: "hy3_gu_gpad_\(dt)_\(gL.weight.shape[0])", device: device)
    let uPad = hy3ScratchTensor(
        shape: [mPad, uL.weight.shape[0]], dtype: dt,
        tag: "hy3_gu_upad_\(dt)_\(uL.weight.shape[0])", device: device)
    guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) else {
        return (
            gate.callMany(ffnIn, t: t, on: cmd, device: device),
            up.callMany(ffnIn, t: t, on: cmd, device: device)
        )
    }
    let kInU = uL.scales.shape[uL.scales.shape.count - 1] * uL.groupSize
    Ops.encodeDequantGemmInt8Mpp(
        input: xIn, weight: gL.weight, scales: gL.scales, biases: gL.biases,
        m: mPad, n: gL.weight.shape[0], k: kIn, groupSize: gL.groupSize,
        into: gPad, encoder: enc)
    Ops.encodeDequantGemmInt8Mpp(
        input: xIn, weight: uL.weight, scales: uL.scales, biases: uL.biases,
        m: mPad, n: uL.weight.shape[0], k: kInU, groupSize: uL.groupSize,
        into: uPad, encoder: enc)
    enc.endEncoding()
    if mPad == t { return (gPad, uPad) }
    return (gPad.slicedRows(start: 0, count: t), uPad.slicedRows(start: 0, count: t))
}

/// Causal SDPA for Hy3 prefill: prefer `sdpaPrefillMma` when T%32==0
/// (f16, d=128); else `sdpaMulti`.
private func hy3SdpaPrefill(
    q: Tensor,  // [nRows, nHeads, headDim]
    cache: any KVCacheProtocol,
    nRows: Int, position: Int,
    nHeads: Int, nKVHeads: Int, headDim: Int,
    scale: Float,
    on cmd: MTLCommandBuffer
) -> Tensor {
    let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
    let usePrefillMma: Bool = {
        guard q.dtype == .f16, headDim == 128, nRows >= 32, nRows % 32 == 0
        else { return false }
        if ProcessInfo.processInfo.environment["BUTTER_HY3_NO_SDPA_PREFILL_MMA"] != nil {
            return false
        }
        if ProcessInfo.processInfo.environment["BUTTER_HY3_SDPA_PREFILL_MMA"] != nil {
            return true
        }
        return nRows >= 32
    }()
    if usePrefillMma {
        let kLen = position + nRows
        let dev = Device.shared
        let qHTD = hy3ScratchTensor(
            shape: [nHeads, nRows, headDim], dtype: .f16,
            tag: "hy3_sdpa_qhtd_\(nHeads)_\(nRows)_\(headDim)", device: dev)
        _ = Ops.transposeTHD_f16(
            q, t: nRows, h: nHeads, d: headDim, on: cmd, into: qHTD)
        // Compact K‖V on a concurrent encoder (two independent packs of the
        // strided cache into [nKV, kLen, d] slabs). Serial compact was two
        // full encoder open/close cycles per layer.
        let kPacked = hy3ScratchTensor(
            shape: [nKVHeads, kLen, headDim], dtype: .f16,
            tag: "hy3_sdpa_kpack_\(nKVHeads)_\(kLen)_\(headDim)", device: dev)
        let vPacked = hy3ScratchTensor(
            shape: [nKVHeads, kLen, headDim], dtype: .f16,
            tag: "hy3_sdpa_vpack_\(nKVHeads)_\(kLen)_\(headDim)", device: dev)
        if let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) {
            _ = Ops.compactKV_f16(
                src: cacheK, nKV: nKVHeads, maxSeq: cache.capacity,
                kLen: kLen, headDim: headDim, on: cmd, into: kPacked, encoder: enc)
            _ = Ops.compactKV_f16(
                src: cacheV, nKV: nKVHeads, maxSeq: cache.capacity,
                kLen: kLen, headDim: headDim, on: cmd, into: vPacked, encoder: enc)
            enc.endEncoding()
        } else {
            _ = Ops.compactKV_f16(
                src: cacheK, nKV: nKVHeads, maxSeq: cache.capacity,
                kLen: kLen, headDim: headDim, on: cmd, into: kPacked)
            _ = Ops.compactKV_f16(
                src: cacheV, nKV: nKVHeads, maxSeq: cache.capacity,
                kLen: kLen, headDim: headDim, on: cmd, into: vPacked)
        }
        let outHTD = hy3ScratchTensor(
            shape: [nHeads, nRows, headDim], dtype: .f16,
            tag: "hy3_sdpa_outhtd_\(nHeads)_\(nRows)_\(headDim)", device: dev)
        _ = Ops.sdpaPrefillMma(
            q: qHTD, k: kPacked, v: vPacked,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            qLen: nRows, kLen: kLen, scale: scale, on: cmd, into: outHTD)
        let outTHD = hy3ScratchTensor(
            shape: [nRows, nHeads, headDim], dtype: .f16,
            tag: "hy3_sdpa_outthd_\(nRows)_\(nHeads)_\(headDim)", device: dev)
        return Ops.transposeHTD_f16(
            outHTD, t: nRows, h: nHeads, d: headDim, on: cmd, into: outTHD)
    }
    return Ops.sdpaMulti(
        q: q, k: cacheK, v: cacheV,
        nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
        baseKV: position, nQuery: nRows, kvStride: cache.capacity,
        causal: true, scale: scale, on: cmd)
}

private func hy3ConcurrentQKV(
    xNorm: Tensor, t: Int,
    qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear,
    cmd: MTLCommandBuffer, device: Device
) -> (Tensor, Tensor, Tensor) {
    if ProcessInfo.processInfo.environment["BUTTER_HY3_SERIAL_QKV"] != nil {
        return (
            hy3Int8CallMany(qProj, xNorm, t: t, on: cmd, device: device),
            hy3Int8CallMany(kProj, xNorm, t: t, on: cmd, device: device),
            hy3Int8CallMany(vProj, xNorm, t: t, on: cmd, device: device)
        )
    }
    // Pad M to a multiple of 32 so concurrent int8 MPP always fires —
    // ragged T (e.g. 152) used to fall back to serial callMany and lose
    // Q‖K‖V overlap for the entire stack.
    if t > 0,
       let qL = qProj.inner as? QuantizedLinear,
       let kL = kProj.inner as? QuantizedLinear,
       let vL = vProj.inner as? QuantizedLinear,
       qL.bits == 8, kL.bits == 8, vL.bits == 8
    {
        func outDim(_ ql: QuantizedLinear) -> Int { ql.weight.shape[0] }
        func inDim(_ ql: QuantizedLinear) -> Int {
            ql.scales.shape[ql.scales.shape.count - 1] * ql.groupSize
        }
        let mPad = ((t + 31) / 32) * 32
        let kIn = inDim(qL)
        let dt = xNorm.dtype
        let xIn: Tensor
        if mPad == t {
            xIn = xNorm
        } else {
            let xPadBuf = device.intermediateScratch(
                tag: "hy3_qkv_xpad_\(dt)", minBytes: mPad * kIn * dt.byteSize)
            xIn = Tensor(
                buffer: xPadBuf, offset: 0, shape: [mPad, kIn], dtype: dt)
            let valid = t * kIn * dt.byteSize
            let tail = (mPad - t) * kIn * dt.byteSize
            let blit = cmd.makeBlitCommandEncoder()!
            blit.copy(
                from: xNorm.buffer, sourceOffset: xNorm.offset,
                to: xIn.buffer, destinationOffset: 0, size: valid)
            if tail > 0 {
                blit.fill(
                    buffer: xIn.buffer,
                    range: valid ..< (valid + tail),
                    value: 0)
            }
            blit.endEncoding()
        }
        let qPad = hy3ScratchTensor(
            shape: [mPad, outDim(qL)], dtype: dt,
            tag: "hy3_qkv_qpad_\(dt)_\(outDim(qL))", device: device)
        let kPad = hy3ScratchTensor(
            shape: [mPad, outDim(kL)], dtype: dt,
            tag: "hy3_qkv_kpad_\(dt)_\(outDim(kL))", device: device)
        let vPad = hy3ScratchTensor(
            shape: [mPad, outDim(vL)], dtype: dt,
            tag: "hy3_qkv_vpad_\(dt)_\(outDim(vL))", device: device)
        guard let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
        else {
            return (
                qProj.callMany(xNorm, t: t, on: cmd, device: device),
                kProj.callMany(xNorm, t: t, on: cmd, device: device),
                vProj.callMany(xNorm, t: t, on: cmd, device: device)
            )
        }
        Ops.encodeDequantGemmInt8Mpp(
            input: xIn, weight: qL.weight, scales: qL.scales, biases: qL.biases,
            m: mPad, n: outDim(qL), k: inDim(qL), groupSize: qL.groupSize,
            into: qPad, encoder: enc)
        Ops.encodeDequantGemmInt8Mpp(
            input: xIn, weight: kL.weight, scales: kL.scales, biases: kL.biases,
            m: mPad, n: outDim(kL), k: inDim(kL), groupSize: kL.groupSize,
            into: kPad, encoder: enc)
        Ops.encodeDequantGemmInt8Mpp(
            input: xIn, weight: vL.weight, scales: vL.scales, biases: vL.biases,
            m: mPad, n: outDim(vL), k: inDim(vL), groupSize: vL.groupSize,
            into: vPad, encoder: enc)
        enc.endEncoding()
        if mPad == t {
            return (qPad, kPad, vPad)
        }
        return (
            qPad.slicedRows(start: 0, count: t),
            kPad.slicedRows(start: 0, count: t),
            vPad.slicedRows(start: 0, count: t)
        )
    }
    return (
        qProj.callMany(xNorm, t: t, on: cmd, device: device),
        kProj.callMany(xNorm, t: t, on: cmd, device: device),
        vProj.callMany(xNorm, t: t, on: cmd, device: device)
    )
}

/// Full-head RoPE over T tokens + batched KV cache write.
/// Mirrors Qwen3x prefill: one `ropePartialManyTwo` + one
/// `appendRangeOnGPUMany` instead of a per-row loop (saves ~2T encoder
/// begin/ends per layer at large T).
private func hy3BatchedRopeAndKV(
    q: inout Tensor, k: inout Tensor, v: Tensor,
    nRows: Int, position: Int,
    nHeads: Int, nKVHeads: Int, headDim: Int,
    ropeTheta: Float,
    cache: any KVCacheProtocol,
    cmd: MTLCommandBuffer, device: Device
) {
    let qDim = nHeads * headDim
    let kvDim = nKVHeads * headDim
    let positionsT = hy3ScratchTensor(
        shape: [nRows], dtype: .u32, tag: "hy3_rope_pos_\(nRows)", device: device)
    let positionsPtr = positionsT.buffer.contents()
        .advanced(by: positionsT.offset)
        .bindMemory(to: UInt32.self, capacity: nRows)
    for r in 0 ..< nRows { positionsPtr[r] = UInt32(position + r) }
    // Full rotary (rotaryDim == headDim). In-place on q/k.
    Ops.ropePartialManyTwo(
        q: q, qNHeads: nHeads, qRowStride: qDim,
        k: k, kNHeads: nKVHeads, kRowStride: kvDim,
        positions: positionsT, t: nRows,
        headDim: headDim, rotaryDim: headDim,
        thetaBase: ropeTheta, scaling: .none, on: cmd)

    guard let kv = cache as? KVCache else {
        // Fallback: per-row append if a non-raw cache is wired.
        for i in 0 ..< nRows {
            let kRow = k.slicedRows(start: i, count: 1)
                .reshaped(to: [nKVHeads, headDim])
            let vRow = v.slicedRows(start: i, count: 1)
                .reshaped(to: [nKVHeads, headDim])
            cache.appendOnGPU(kFlat: kRow, vFlat: vRow, on: cmd)
        }
        return
    }
    let appendPositions = hy3ScratchTensor(
        shape: [nRows], dtype: .u32, tag: "hy3_kv_append_pos_\(nRows)", device: device)
    kv.reserveSlotsManyOnHost(t: nRows, into: appendPositions)
    kv.appendRangeOnGPUMany(
        kFlat: k, vFlat: v, t: nRows,
        positions: appendPositions, on: cmd)
}

// ─── Layer protocol ──────────────────────────────────────────────────

protocol Hy3LayerProtocol: Module {
    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor

    /// Multi-token layer forward for chunked prefill. `h` is `[nRows, hidden]`.
    func forwardMulti(
        _ h: Tensor, startingAt position: Int, cache: any KVCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor
}

// ─── Dense layer ─────────────────────────────────────────────────────

final class Hy3DenseLayer: Hy3LayerProtocol {
    let qProj, kProj, vProj, oProj: AnyLinear
    let qNorm, kNorm: RMSNorm?
    let gateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim: Int
    let ropeTheta: Float
    let dtype: DType
    let scale: Float

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        qNorm: RMSNorm?, kNorm: RMSNorm?,
        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        ropeTheta: Float, dtype: DType
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
        self.hidden = hidden
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.dtype = dtype
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }

    func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        if let qn = qNorm {
            for (k, v) in qn.parameters() { out.append(("self_attn.q_norm.\(k)", v)) }
        }
        if let kn = kNorm {
            for (k, v) in kn.parameters() { out.append(("self_attn.k_norm.\(k)", v)) }
        }
        for (k, v) in gateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_attention_layernorm.\(k)", v)) }
        return out
    }

    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let residual = h
        let xNorm = inputNorm(h, on: cmd)
        var q = qProj(xNorm, on: cmd)
        var k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // QK-norm before RoPE.
        if let qn = qNorm {
            q = hy3ApplyPerHeadRMSNorm(
                q, weight: qn.weight, eps: qn.eps,
                nHeads: nHeads, headDim: headDim, on: cmd, device: device)
        }
        if let kn = kNorm {
            k = hy3ApplyPerHeadRMSNorm(
                k, weight: kn.weight, eps: kn.eps,
                nHeads: nKVHeads, headDim: headDim, on: cmd, device: device)
        }

        let qRotated = Ops.rope(
            q.reshaped(to: [nHeads, headDim]),
            position: position, headDim: headDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)
        let kRotated = Ops.rope(
            k.reshaped(to: [nKVHeads, headDim]),
            position: position, headDim: headDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)

        cache.appendOnGPU(
            kFlat: kRotated,
            vFlat: v.reshaped(to: [nKVHeads, headDim]),
            on: cmd)

        let (kCache, vCache) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qRotated,
            k: kCache, v: vCache,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.capacity,
            scale: scale, on: cmd)
        let attn = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        let h2 = Ops.add(residual, attn, on: cmd)

        let ffnIn = postAttnNorm(h2, on: cmd)
        let g = gateProj(ffnIn, on: cmd)
        let u = upProj(ffnIn, on: cmd)
        let inner = Ops.swiglu(gate: g, up: u, on: cmd)
        let ffn = downProj(inner, on: cmd)
        return Ops.add(h2, ffn, on: cmd)
    }

    func forwardMulti(
        _ h: Tensor, startingAt position: Int, cache: any KVCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let nRows = h.shape[0]
        precondition(h.shape == [nRows, hidden])
        let residual = h
        let xNormFlat = hy3ScratchTensor(
            shape: [nRows * hidden], dtype: residual.dtype,
            tag: "hy3_xnorm_\(residual.dtype)_\(nRows)_\(hidden)", device: device)
        _ = Ops.rmsNormRows(
            h.reshaped(to: [nRows * hidden]),
            weight: inputNorm.weight, eps: inputNorm.eps,
            nRows: nRows, rowSize: hidden, on: cmd, into: xNormFlat)
        let xNorm = xNormFlat.reshaped(to: [nRows, hidden])

        let (qRaw, kRaw, v) = hy3ConcurrentQKV(
            xNorm: xNorm, t: nRows,
            qProj: qProj, kProj: kProj, vProj: vProj,
            cmd: cmd, device: device)
        var q = qRaw
        var k = kRaw

        if let qn = qNorm {
            let qNormOut = hy3ScratchTensor(
                shape: [nRows * nHeads * headDim], dtype: q.dtype,
                tag: "hy3_qn_\(q.dtype)_\(nRows)_\(nHeads)_\(headDim)", device: device)
            q = Ops.rmsNormRows(
                q.reshaped(to: [nRows * nHeads * headDim]),
                weight: qn.weight, eps: qn.eps,
                nRows: nRows * nHeads, rowSize: headDim, on: cmd, into: qNormOut
            ).reshaped(to: [nRows, nHeads * headDim])
        }
        if let kn = kNorm {
            let kNormOut = hy3ScratchTensor(
                shape: [nRows * nKVHeads * headDim], dtype: k.dtype,
                tag: "hy3_kn_\(k.dtype)_\(nRows)_\(nKVHeads)_\(headDim)", device: device)
            k = Ops.rmsNormRows(
                k.reshaped(to: [nRows * nKVHeads * headDim]),
                weight: kn.weight, eps: kn.eps,
                nRows: nRows * nKVHeads, rowSize: headDim, on: cmd, into: kNormOut
            ).reshaped(to: [nRows, nKVHeads * headDim])
        }

        // Batched RoPE + KV append (one encoder each, not T loops).
        hy3BatchedRopeAndKV(
            q: &q, k: &k, v: v,
            nRows: nRows, position: position,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            ropeTheta: ropeTheta, cache: cache, cmd: cmd, device: device)

        let qForSdpa = q.reshaped(to: [nRows, nHeads, headDim])
        let attnOut = hy3SdpaPrefill(
            q: qForSdpa, cache: cache,
            nRows: nRows, position: position,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            scale: scale, on: cmd)
        let oOut = hy3Int8CallMany(
            oProj, attnOut.reshaped(to: [nRows, nHeads * headDim]),
            t: nRows, on: cmd, device: device)
        let h2Flat = hy3ScratchTensor(
            shape: [nRows * hidden], dtype: residual.dtype,
            tag: "hy3_h2_resid_\(residual.dtype)_\(nRows)_\(hidden)", device: device)
        let ffnInFlat = hy3ScratchTensor(
            shape: [nRows * hidden], dtype: residual.dtype,
            tag: "hy3_ffnin_\(residual.dtype)_\(nRows)_\(hidden)", device: device)
        _ = Ops.addAndRmsNorm(
            residual.reshaped(to: [nRows * hidden]),
            oOut.reshaped(to: [nRows * hidden]),
            weight: postAttnNorm.weight, eps: postAttnNorm.eps,
            nRows: nRows, rowSize: hidden, on: cmd,
            residualOut: h2Flat, normedOut: ffnInFlat)
        let h2 = h2Flat.reshaped(to: [nRows, hidden])
        let ffnIn = ffnInFlat.reshaped(to: [nRows, hidden])
        // Concurrent gate‖up (int8 MPP) — dense layer intermediate is huge
        // (13312); serial left half the GPU idle.
        let (g, u) = hy3ConcurrentGateUp(
            ffnIn: ffnIn, t: nRows,
            gate: gateProj, up: upProj,
            cmd: cmd, device: device)
        // SwiGLU into u (same shape); avoids fresh intermediate alloc.
        _ = Ops.swiglu(gate: g, up: u, on: cmd, into: u)
        let ffn = hy3Int8CallMany(
            downProj, u, t: nRows, on: cmd, device: device)
        // In-place into residual (hPersist) — see MoE multi residual note.
        return Ops.add(
            h2.reshaped(to: [nRows * hidden]),
            ffn.reshaped(to: [nRows * hidden]),
            on: cmd,
            into: residual.reshaped(to: [nRows * hidden])
        ).reshaped(to: [nRows, hidden])
    }
}

// ─── MoE layer ───────────────────────────────────────────────────────

/// Stacked affine-quantized expert weights: shape [nExperts, outDim, packedIn].
struct Hy3StackedQuant {
    let weight: Tensor
    let scales: Tensor
    let biases: Tensor
}

final class Hy3MoELayer: Hy3LayerProtocol {
    let qProj, kProj, vProj, oProj: AnyLinear
    let qNorm, kNorm: RMSNorm?
    let inputNorm, postAttnNorm: RMSNorm
    let routerGate: AnyLinear
    let router: MoERouter
    /// GPU-resident expert_bias [nExperts] matching activation dtype when
    /// possible (Path B selection). Host floats stay on `router.expertBias`.
    let expertBiasGPU: Tensor
    let switchGate, switchUp, switchDown: Hy3StackedQuant
    let gateBits, upBits, downBits, groupSize: Int
    let sharedGate, sharedUp, sharedDown: AnyLinear
    let hidden, moeIntermediate, nHeads, nKVHeads, headDim: Int
    let ropeTheta: Float
    let dtype: DType
    let scale: Float
    let nShared: Int

    // Lazy per-layer scratches for the no-host GPU Path B path.
    private var routerIndicesScratch: Tensor?
    private var routerWeightsScratch: Tensor?
    private var unbiasedScratch: Tensor?
    private var biasedScratch: Tensor?
    private var accumulatorScratch: Tensor?
    private var expertGScratches: [Tensor] = []
    private var expertUScratches: [Tensor] = []
    private var expertInnerScratches: [Tensor] = []
    private var expertOutScratches: [Tensor] = []
    private var scaleVecScratch: Tensor?
    /// Cached expert-index views + gemv batch lists (stable buffer offsets).
    private var expertIdxViews: [Tensor] = []
    private var gateUpW: [Tensor] = []
    private var gateUpS: [Tensor] = []
    private var gateUpB: [Tensor] = []
    private var gateUpIn: [Tensor] = []
    private var gateUpIdx: [Tensor] = []
    private var gateUpOut: [Tensor] = []
    private var downW: [Tensor] = []
    private var downS: [Tensor] = []
    private var downB: [Tensor] = []
    private var downIn: [Tensor] = []
    private var downIdx: [Tensor] = []
    private var downOut: [Tensor] = []
    private var weightScalarViews: [Tensor] = []

    // Prefill workspaces (persistent, reused every layer / chunk).
    // Avoids makeBuffer + dual 1 GB scratch slabs under memory pressure.
    private var prefillCapT: Int = 0
    private var pfIndices: Tensor?
    private var pfWeights: Tensor?
    private var pfUnbiased: Tensor?
    private var pfBiased: Tensor?
    private var pfSortedExperts: Tensor?
    private var pfSourceT: Tensor?
    private var pfInvPerm: Tensor?
    private var pfXGathered: Tensor?
    private var pfGateOut: Tensor?
    private var pfUpOut: Tensor?
    private var pfDownOut: Tensor?
    private var pfRouted: Tensor?
    private var pfScaleVec: Tensor?
    /// Active-only expert-outer plan (compact from sorted ids).
    private var pfActiveExperts: Tensor?
    private var pfRowLo: Tensor?
    private var pfRowHi: Tensor?
    private var pfNActive: Tensor?
    private var pfIndirectGU: Tensor?
    private var pfIndirectDown: Tensor?

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        qNorm: RMSNorm?, kNorm: RMSNorm?,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        routerGate: AnyLinear, router: MoERouter,
        expertBiasGPU: Tensor,
        switchGate: SafeTensorsBundle.QuantizedTriplet,
        switchUp: SafeTensorsBundle.QuantizedTriplet,
        switchDown: SafeTensorsBundle.QuantizedTriplet,
        gateBits: Int, upBits: Int, downBits: Int, groupSize: Int,
        sharedGate: AnyLinear, sharedUp: AnyLinear, sharedDown: AnyLinear,
        hidden: Int, moeIntermediate: Int,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        ropeTheta: Float, dtype: DType, nShared: Int
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
        self.routerGate = routerGate
        self.router = router
        self.expertBiasGPU = expertBiasGPU
        self.switchGate = Hy3StackedQuant(
            weight: switchGate.weight, scales: switchGate.scales, biases: switchGate.biases)
        self.switchUp = Hy3StackedQuant(
            weight: switchUp.weight, scales: switchUp.scales, biases: switchUp.biases)
        self.switchDown = Hy3StackedQuant(
            weight: switchDown.weight, scales: switchDown.scales, biases: switchDown.biases)
        self.gateBits = gateBits
        self.upBits = upBits
        self.downBits = downBits
        self.groupSize = groupSize
        self.sharedGate = sharedGate
        self.sharedUp = sharedUp
        self.sharedDown = sharedDown
        self.hidden = hidden
        self.moeIntermediate = moeIntermediate
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.dtype = dtype
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
        self.nShared = nShared
    }

    func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        if let qn = qNorm {
            for (k, v) in qn.parameters() { out.append(("self_attn.q_norm.\(k)", v)) }
        }
        if let kn = kNorm {
            for (k, v) in kn.parameters() { out.append(("self_attn.k_norm.\(k)", v)) }
        }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_attention_layernorm.\(k)", v)) }
        for (k, v) in routerGate.parameters() { out.append(("mlp.router.gate.\(k)", v)) }
        out.append(("mlp.switch_mlp.gate_proj.weight", switchGate.weight))
        out.append(("mlp.switch_mlp.up_proj.weight", switchUp.weight))
        out.append(("mlp.switch_mlp.down_proj.weight", switchDown.weight))
        for (k, v) in sharedGate.parameters() { out.append(("mlp.shared_mlp.gate_proj.\(k)", v)) }
        for (k, v) in sharedUp.parameters() { out.append(("mlp.shared_mlp.up_proj.\(k)", v)) }
        for (k, v) in sharedDown.parameters() { out.append(("mlp.shared_mlp.down_proj.\(k)", v)) }
        return out
    }

    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // ── Attention (same as dense: qk-norm before RoPE) ───────────
        let residual = h
        let xAttn = inputNorm(h, on: cmd)
        var q = qProj(xAttn, on: cmd)
        var k = kProj(xAttn, on: cmd)
        let v = vProj(xAttn, on: cmd)
        if let qn = qNorm {
            q = hy3ApplyPerHeadRMSNorm(
                q, weight: qn.weight, eps: qn.eps,
                nHeads: nHeads, headDim: headDim, on: cmd, device: device)
        }
        if let kn = kNorm {
            k = hy3ApplyPerHeadRMSNorm(
                k, weight: kn.weight, eps: kn.eps,
                nHeads: nKVHeads, headDim: headDim, on: cmd, device: device)
        }
        let qRotated = Ops.rope(
            q.reshaped(to: [nHeads, headDim]),
            position: position, headDim: headDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)
        let kRotated = Ops.rope(
            k.reshaped(to: [nKVHeads, headDim]),
            position: position, headDim: headDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)
        cache.appendOnGPU(
            kFlat: kRotated,
            vFlat: v.reshaped(to: [nKVHeads, headDim]),
            on: cmd)
        let (kCache, vCache) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qRotated,
            k: kCache, v: vCache,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.capacity,
            scale: scale, on: cmd)
        let attn = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        let h2 = Ops.add(residual, attn, on: cmd)

        // ── MoE FFN: GPU Path B (no host 192-logit sync) ─────────────
        let ffnIn = postAttnNorm(h2, on: cmd)
        let logitsTensor = routerGate(ffnIn, on: cmd)

        // int2 affine stacked experts → expert-indexed gemv (T=1).
        // BM16 gather is for prefill only — at mTotal=8 it is ~4× slower.
        if gateBits == 2 && upBits == 2 && downBits == 2 && router.topK == 8 {
            let routed = decodeGPUPathB(
                ffnIn: ffnIn, logits: logitsTensor,
                cmd: cmd, device: device)
            return Ops.add(h2, routed, on: cmd)
        }
        return decodeHostRouter(
            h2: h2, ffnIn: ffnIn, logitsTensor: logitsTensor,
            cmd: cmd, device: device)
    }

    /// Fully on-device Path B: sigmoid+bias scores → biased top-k /
    /// unbiased renorm weights × router_scale → int2 expert-indexed
    /// gate/up/down → scalarFMAChain8 + ungated shared expert.
    /// Does **not** commit `cmd` (caller contract).
    private func decodeGPUPathB(
        ffnIn: Tensor, logits: Tensor,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let nExperts = router.nExperts
        let topK = router.topK
        ensureScratches(device: device)

        // Scores in activation dtype (bf16 for oQ2). expertBiasGPU is
        // converted to the same dtype at load.
        precondition(
            expertBiasGPU.dtype == logits.dtype,
            "Hy3 Path B: expert_bias dtype \(expertBiasGPU.dtype) ≠ logits \(logits.dtype)")
        let scoreDt = logits.dtype
        let unbiased = Ops.sigmoid(logits, on: cmd, into: unbiasedScratch!)
        let biased = Ops.add(unbiased, expertBiasGPU, on: cmd, into: biasedScratch!)

        Ops.moeRouterTopKBiased(
            scoreBiased: biased, scoreUnbiased: unbiased,
            indicesOut: routerIndicesScratch!,
            weightsOut: routerWeightsScratch!,
            nExperts: nExperts, k: topK, on: cmd)

        // final_weight = routed_scaling_factor * renormed_unbiased
        let scale = router.routedScalingFactor
        if scale != 1.0 {
            if scaleVecScratch == nil
                || scaleVecScratch!.dtype != scoreDt
                || scaleVecScratch!.elementCount != topK
            {
                scaleVecScratch = Tensor.filled(
                    scale, shape: [topK], dtype: scoreDt, device: device)
            }
            _ = Ops.mul(
                routerWeightsScratch!, scaleVecScratch!, on: cmd,
                into: routerWeightsScratch!)
        }

        // Rebuild gemv batch lists only when ffnIn identity changes (always
        // for the input list) — expert index views and weight stacks are stable.
        if expertIdxViews.count != topK {
            expertIdxViews = (0 ..< topK).map { slot in
                Tensor(
                    buffer: routerIndicesScratch!.buffer,
                    offset: routerIndicesScratch!.offset + slot * 4,
                    shape: [1], dtype: .u32)
            }
            gateUpW = []; gateUpS = []; gateUpB = []
            gateUpIn = []; gateUpIdx = []; gateUpOut = []
            downW = []; downS = []; downB = []
            downIn = []; downIdx = []; downOut = []
            for slot in 0 ..< topK {
                gateUpW.append(switchGate.weight)
                gateUpS.append(switchGate.scales)
                gateUpB.append(switchGate.biases)
                gateUpIn.append(ffnIn)  // replaced below each call
                gateUpIdx.append(expertIdxViews[slot])
                gateUpOut.append(expertGScratches[slot])
                gateUpW.append(switchUp.weight)
                gateUpS.append(switchUp.scales)
                gateUpB.append(switchUp.biases)
                gateUpIn.append(ffnIn)
                gateUpIdx.append(expertIdxViews[slot])
                gateUpOut.append(expertUScratches[slot])
                downW.append(switchDown.weight)
                downS.append(switchDown.scales)
                downB.append(switchDown.biases)
                downIn.append(expertInnerScratches[slot])
                downIdx.append(expertIdxViews[slot])
                downOut.append(expertOutScratches[slot])
            }
            let wBytes = scoreDt.byteSize
            weightScalarViews = (0 ..< topK).map { slot in
                Tensor(
                    buffer: routerWeightsScratch!.buffer,
                    offset: routerWeightsScratch!.offset + slot * wBytes,
                    shape: [1], dtype: scoreDt)
            }
        }
        // Refresh ffnIn slots (new residual each token).
        for slot in 0 ..< topK {
            gateUpIn[slot * 2] = ffnIn
            gateUpIn[slot * 2 + 1] = ffnIn
        }
        Ops.dequantGemvInt2ExpertIndexedMany(
            weightsStacked: gateUpW, scalesStacked: gateUpS, biasesStacked: gateUpB,
            inputs: gateUpIn, expertIndices: gateUpIdx, outputs: gateUpOut,
            groupSize: groupSize, on: cmd)

        Ops.swigluMany(
            gates: Array(expertGScratches.prefix(topK)),
            ups: Array(expertUScratches.prefix(topK)),
            outs: Array(expertInnerScratches.prefix(topK)), on: cmd)

        Ops.dequantGemvInt2ExpertIndexedMany(
            weightsStacked: downW, scalesStacked: downS, biasesStacked: downB,
            inputs: downIn, expertIndices: downIdx, outputs: downOut,
            groupSize: groupSize, on: cmd)

        let acc = accumulatorScratch!
        Ops.scalarFMAChain8(
            scalars: weightScalarViews,
            values: Array(expertOutScratches.prefix(topK)),
            out: acc, on: cmd)

        // Ungated shared expert — always on.
        if nShared > 0 {
            let sg = sharedGate(ffnIn, on: cmd)
            let su = sharedUp(ffnIn, on: cmd)
            let inner = Ops.swiglu(gate: sg, up: su, on: cmd)
            let sharedOut = sharedDown(inner, on: cmd)
            return Ops.add(acc, sharedOut, on: cmd)
        }
        return acc
    }

    /// Public so `Hy3Model.forward` can pre-size persistent scratches
    /// while `scratchModeActive` is false (slab slices must not back
    /// these across tokens).
    func ensureScratchesForDecode(device: Device) {
        ensureScratches(device: device)
    }

    private func ensureScratches(device: Device) {
        let topK = router.topK
        let nExperts = router.nExperts
        if routerIndicesScratch == nil || routerIndicesScratch!.elementCount != topK {
            routerIndicesScratch = Tensor.empty(shape: [topK], dtype: .u32, device: device)
            routerWeightsScratch = Tensor.empty(shape: [topK], dtype: dtype, device: device)
        }
        if unbiasedScratch == nil || unbiasedScratch!.elementCount != nExperts
            || unbiasedScratch!.dtype != dtype
        {
            unbiasedScratch = Tensor.empty(shape: [nExperts], dtype: dtype, device: device)
            biasedScratch = Tensor.empty(shape: [nExperts], dtype: dtype, device: device)
        }
        if accumulatorScratch == nil || accumulatorScratch!.dtype != dtype
            || accumulatorScratch!.elementCount != hidden
        {
            accumulatorScratch = Tensor.empty(shape: [hidden], dtype: dtype, device: device)
        }
        let outDim = moeIntermediate
        while expertGScratches.count < topK {
            expertGScratches.append(
                Tensor.empty(shape: [outDim], dtype: dtype, device: device))
            expertUScratches.append(
                Tensor.empty(shape: [outDim], dtype: dtype, device: device))
            expertInnerScratches.append(
                Tensor.empty(shape: [outDim], dtype: dtype, device: device))
            expertOutScratches.append(
                Tensor.empty(shape: [hidden], dtype: dtype, device: device))
        }
    }

    /// Legacy host top-k + row-slice gemv (non-int2 or non-topK=8).
    private func decodeHostRouter(
        h2: Tensor, ffnIn: Tensor, logitsTensor: Tensor,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        cmd.commit()
        cmd.waitUntilCompleted()
        let logits = logitsTensor.toFloatArray()
        let routing = router.route(logits: logits)

        let workCmd = device.makeCommandBuffer()
        let nSlots = routing.indices.count
        let moeInter = switchGate.weight.shape[1]

        var gW = [Tensor](); gW.reserveCapacity(nSlots)
        var gS = [Tensor](); gS.reserveCapacity(nSlots)
        var gB = [Tensor](); gB.reserveCapacity(nSlots)
        var uW = [Tensor](); uW.reserveCapacity(nSlots)
        var uS = [Tensor](); uS.reserveCapacity(nSlots)
        var uB = [Tensor](); uB.reserveCapacity(nSlots)
        var dW = [Tensor](); dW.reserveCapacity(nSlots)
        var dS = [Tensor](); dS.reserveCapacity(nSlots)
        var dB = [Tensor](); dB.reserveCapacity(nSlots)
        var gateOut = [Tensor](); gateOut.reserveCapacity(nSlots)
        var upOut = [Tensor](); upOut.reserveCapacity(nSlots)
        var downOut = [Tensor](); downOut.reserveCapacity(nSlots)
        var xs = [Tensor](); xs.reserveCapacity(nSlots)
        for ei in routing.indices {
            gW.append(sliceExpert(switchGate.weight, ei))
            gS.append(sliceExpert(switchGate.scales, ei))
            gB.append(sliceExpert(switchGate.biases, ei))
            uW.append(sliceExpert(switchUp.weight, ei))
            uS.append(sliceExpert(switchUp.scales, ei))
            uB.append(sliceExpert(switchUp.biases, ei))
            dW.append(sliceExpert(switchDown.weight, ei))
            dS.append(sliceExpert(switchDown.scales, ei))
            dB.append(sliceExpert(switchDown.biases, ei))
            gateOut.append(Tensor.empty(shape: [moeInter], dtype: dtype, device: device))
            upOut.append(Tensor.empty(shape: [moeInter], dtype: dtype, device: device))
            downOut.append(Tensor.empty(shape: [hidden], dtype: dtype, device: device))
            xs.append(ffnIn)
        }

        Ops.dequantGemvMany(
            bits: gateBits, weights: gW + uW, scales: gS + uS, biases: gB + uB,
            inputs: xs + xs, outputs: gateOut + upOut,
            groupSize: groupSize, on: workCmd)
        var inners = [Tensor](); inners.reserveCapacity(nSlots)
        for i in 0 ..< nSlots {
            inners.append(Ops.swiglu(gate: gateOut[i], up: upOut[i], on: workCmd))
        }
        Ops.dequantGemvMany(
            bits: downBits, weights: dW, scales: dS, biases: dB,
            inputs: inners, outputs: downOut,
            groupSize: groupSize, on: workCmd)

        var accumulator: Tensor?
        for slot in 0 ..< nSlots {
            let w = Tensor.filled(
                routing.weights[slot], shape: [hidden],
                dtype: dtype, device: device)
            let scaled = Ops.mul(downOut[slot], w, on: workCmd)
            accumulator =
                accumulator.map { Ops.add($0, scaled, on: workCmd) } ?? scaled
        }

        if nShared > 0 {
            let sg = sharedGate(ffnIn, on: workCmd)
            let su = sharedUp(ffnIn, on: workCmd)
            let inner = Ops.swiglu(gate: sg, up: su, on: workCmd)
            let sharedOut = sharedDown(inner, on: workCmd)
            accumulator =
                accumulator.map { Ops.add($0, sharedOut, on: workCmd) } ?? sharedOut
        }

        let result = Ops.add(h2, accumulator!, on: workCmd)
        workCmd.commit()
        workCmd.waitUntilCompleted()
        return result
    }

    /// Multi-token prefill: batched attention + CSR gather int2 MoE.
    func forwardMulti(
        _ h: Tensor, startingAt position: Int, cache: any KVCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let nRows = h.shape[0]
        precondition(h.shape == [nRows, hidden])
        let residual = h
        let xNormFlat = hy3ScratchTensor(
            shape: [nRows * hidden], dtype: residual.dtype,
            tag: "hy3_xnorm_\(residual.dtype)_\(nRows)_\(hidden)", device: device)
        _ = Ops.rmsNormRows(
            h.reshaped(to: [nRows * hidden]),
            weight: inputNorm.weight, eps: inputNorm.eps,
            nRows: nRows, rowSize: hidden, on: cmd, into: xNormFlat)
        let xNorm = xNormFlat.reshaped(to: [nRows, hidden])

        let skipAttn =
            ProcessInfo.processInfo.environment["BUTTER_HY3_SKIP_ATTN"] != nil
        let skipSdpa =
            ProcessInfo.processInfo.environment["BUTTER_HY3_SKIP_SDPA"] != nil

        let oOut: Tensor
        if skipAttn {
            // GPU-zero (not host Tensor.filled) — probe must not thrash PCIe/host.
            oOut = hy3GpuZeros(
                shape: [nRows, hidden], dtype: residual.dtype,
                tag: "hy3_skip_attn_z", cmd: cmd, device: device)
        } else {
            // Concurrent Q/K/V (pads M to 32 for int8 MPP).
            let (qRaw, kRaw, v) = hy3ConcurrentQKV(
                xNorm: xNorm, t: nRows,
                qProj: qProj, kProj: kProj, vProj: vProj,
                cmd: cmd, device: device)
            var q = qRaw
            var k = kRaw

            if let qn = qNorm {
                let qNormOut = hy3ScratchTensor(
                    shape: [nRows * nHeads * headDim], dtype: q.dtype,
                    tag: "hy3_qn_\(q.dtype)_\(nRows)_\(nHeads)_\(headDim)", device: device)
                q = Ops.rmsNormRows(
                    q.reshaped(to: [nRows * nHeads * headDim]),
                    weight: qn.weight, eps: qn.eps,
                    nRows: nRows * nHeads, rowSize: headDim, on: cmd, into: qNormOut
                ).reshaped(to: [nRows, nHeads * headDim])
            }
            if let kn = kNorm {
                let kNormOut = hy3ScratchTensor(
                    shape: [nRows * nKVHeads * headDim], dtype: k.dtype,
                    tag: "hy3_kn_\(k.dtype)_\(nRows)_\(nKVHeads)_\(headDim)", device: device)
                k = Ops.rmsNormRows(
                    k.reshaped(to: [nRows * nKVHeads * headDim]),
                    weight: kn.weight, eps: kn.eps,
                    nRows: nRows * nKVHeads, rowSize: headDim, on: cmd, into: kNormOut
                ).reshaped(to: [nRows, nKVHeads * headDim])
            }

            // Batched RoPE + KV append (one encoder each, not T loops).
            hy3BatchedRopeAndKV(
                q: &q, k: &k, v: v,
                nRows: nRows, position: position,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                ropeTheta: ropeTheta, cache: cache, cmd: cmd, device: device)

            let attnOut: Tensor
            if skipSdpa {
                attnOut = hy3GpuZeros(
                    shape: [nRows, nHeads, headDim], dtype: q.dtype,
                    tag: "hy3_skip_sdpa_z", cmd: cmd, device: device)
            } else {
                let qForSdpa = q.reshaped(to: [nRows, nHeads, headDim])
                attnOut = hy3SdpaPrefill(
                    q: qForSdpa, cache: cache,
                    nRows: nRows, position: position,
                    nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    scale: scale, on: cmd)
            }
            oOut = hy3Int8CallMany(
                oProj, attnOut.reshaped(to: [nRows, nHeads * headDim]),
                t: nRows, on: cmd, device: device)
        }
        // Fused residual add + post-attn RMSNorm (one launch vs two).
        // Scratch into: avoids 2× makeBuffer per layer for resid/normed.
        let h2Flat = hy3ScratchTensor(
            shape: [nRows * hidden], dtype: residual.dtype,
            tag: "hy3_h2_resid_\(residual.dtype)_\(nRows)_\(hidden)", device: device)
        let ffnInFlat = hy3ScratchTensor(
            shape: [nRows * hidden], dtype: residual.dtype,
            tag: "hy3_ffnin_\(residual.dtype)_\(nRows)_\(hidden)", device: device)
        _ = Ops.addAndRmsNorm(
            residual.reshaped(to: [nRows * hidden]),
            oOut.reshaped(to: [nRows * hidden]),
            weight: postAttnNorm.weight, eps: postAttnNorm.eps,
            nRows: nRows, rowSize: hidden, on: cmd,
            residualOut: h2Flat, normedOut: ffnInFlat)
        let h2 = h2Flat.reshaped(to: [nRows, hidden])
        let ffnIn = ffnInFlat.reshaped(to: [nRows, hidden])

        precondition(
            gateBits == 2 && upBits == 2 && downBits == 2 && router.topK == 8,
            "Hy3 forwardMulti: only int2 topK=8 Path B gather is implemented")
        let skipMoE = ProcessInfo.processInfo.environment["BUTTER_HY3_SKIP_MOE"] != nil
        let skipShared = ProcessInfo.processInfo.environment["BUTTER_HY3_SKIP_SHARED"] != nil
        // Default: fuse shared gate‖up onto the same concurrent encoder as
        // MoE gate‖up after gather (no dual-queue events).
        let dualQueue =
            ProcessInfo.processInfo.environment["BUTTER_HY3_DUAL_QUEUE"] != nil
        let noDual =
            ProcessInfo.processInfo.environment["BUTTER_HY3_NO_DUAL"] != nil
        let fuseShared =
            nShared > 0 && !skipShared && !dualQueue && !noDual && !skipMoE

        func runSharedMLP(on sharedCmd: MTLCommandBuffer) -> Tensor {
            let sg: Tensor
            let su: Tensor
            if let g = sharedGate.inner as? QuantizedLinear,
               let u = sharedUp.inner as? QuantizedLinear,
               g.bits == 8, u.bits == 8, nRows > 0
            {
                let mPad = ((nRows + 31) / 32) * 32
                let dt = ffnIn.dtype
                let inDimG = g.scales.shape[g.scales.shape.count - 1] * g.groupSize
                let inDimU = u.scales.shape[u.scales.shape.count - 1] * u.groupSize
                let xIn: Tensor
                if mPad == nRows {
                    xIn = ffnIn
                } else {
                    let xPadBuf = device.intermediateScratch(
                        tag: "hy3_shared_xpad_\(dt)",
                        minBytes: mPad * inDimG * dt.byteSize)
                    xIn = Tensor(
                        buffer: xPadBuf, offset: 0, shape: [mPad, inDimG], dtype: dt)
                    let valid = nRows * inDimG * dt.byteSize
                    let tail = (mPad - nRows) * inDimG * dt.byteSize
                    let blit = sharedCmd.makeBlitCommandEncoder()!
                    blit.copy(
                        from: ffnIn.buffer, sourceOffset: ffnIn.offset,
                        to: xIn.buffer, destinationOffset: 0, size: valid)
                    if tail > 0 {
                        blit.fill(
                            buffer: xIn.buffer,
                            range: valid ..< (valid + tail),
                            value: 0)
                    }
                    blit.endEncoding()
                }
                let sgPad = hy3ScratchTensor(
                    shape: [mPad, g.weight.shape[0]], dtype: dt,
                    tag: "hy3_shared_gpad_\(dt)_\(g.weight.shape[0])", device: device)
                let suPad = hy3ScratchTensor(
                    shape: [mPad, u.weight.shape[0]], dtype: dt,
                    tag: "hy3_shared_upad_\(dt)_\(u.weight.shape[0])", device: device)
                if let enc = sharedCmd.makeComputeCommandEncoder(dispatchType: .concurrent) {
                    Ops.encodeDequantGemmInt8Mpp(
                        input: xIn, weight: g.weight, scales: g.scales, biases: g.biases,
                        m: mPad, n: g.weight.shape[0], k: inDimG, groupSize: g.groupSize,
                        into: sgPad, encoder: enc)
                    Ops.encodeDequantGemmInt8Mpp(
                        input: xIn, weight: u.weight, scales: u.scales, biases: u.biases,
                        m: mPad, n: u.weight.shape[0], k: inDimU, groupSize: u.groupSize,
                        into: suPad, encoder: enc)
                    enc.endEncoding()
                } else {
                    return sharedDown.callMany(
                        Ops.swiglu(
                            gate: sharedGate.callMany(
                                ffnIn, t: nRows, on: sharedCmd, device: device),
                            up: sharedUp.callMany(
                                ffnIn, t: nRows, on: sharedCmd, device: device),
                            on: sharedCmd),
                        t: nRows, on: sharedCmd, device: device)
                }
                sg = mPad == nRows ? sgPad : sgPad.slicedRows(start: 0, count: nRows)
                su = mPad == nRows ? suPad : suPad.slicedRows(start: 0, count: nRows)
            } else {
                sg = sharedGate.callMany(ffnIn, t: nRows, on: sharedCmd, device: device)
                su = sharedUp.callMany(ffnIn, t: nRows, on: sharedCmd, device: device)
            }
            let inner = Ops.swiglu(gate: sg, up: su, on: sharedCmd)
            return sharedDown.callMany(inner, t: nRows, on: sharedCmd, device: device)
        }

        var sharedOut: Tensor?
        var sharedDoneValue: UInt64?
        if nShared > 0 && !skipShared && dualQueue {
            let cmdB = device.makeCommandBufferB()
            let vFFN = device.nextSyncValue()
            cmd.encodeSignalEvent(device.syncEvent, value: vFFN)
            cmdB.encodeWaitForEvent(device.syncEvent, value: vFFN)
            sharedOut = runSharedMLP(on: cmdB)
            let vDone = device.nextSyncValue()
            cmdB.encodeSignalEvent(device.syncEvent, value: vDone)
            cmdB.commit()
            sharedDoneValue = vDone
        }

        let routed: Tensor
        if skipMoE {
            // GPU blit-zero into persistent workspace (host filled was thrashing
            // free-MoE ceiling measurements ~79× 1.2MB copyIn per prefill).
            ensurePrefillWorkspaces(t: nRows, device: device)
            routed = Tensor(
                buffer: pfRouted!.buffer, offset: pfRouted!.offset,
                shape: [nRows, hidden], dtype: ffnIn.dtype)
            let bytes = nRows * hidden * ffnIn.dtype.byteSize
            let blit = cmd.makeBlitCommandEncoder()!
            blit.fill(
                buffer: routed.buffer,
                range: routed.offset ..< (routed.offset + bytes),
                value: 0)
            blit.endEncoding()
            if nShared > 0 && !skipShared && !dualQueue {
                sharedOut = runSharedMLP(on: cmd)
            }
        } else {
            let (r, fusedShared) = decodeManyGPU(
                ffnIn: ffnIn, t: nRows, cmd: cmd, device: device,
                fuseShared: fuseShared)
            routed = r
            if let fs = fusedShared { sharedOut = fs }
        }

        // Combine routed + shared into a scratch buffer (avoid fresh empty).
        var combined = routed
        if nShared > 0 && !skipShared {
            let combFlat = hy3ScratchTensor(
                shape: [nRows * hidden], dtype: routed.dtype,
                tag: "hy3_moe_comb_\(routed.dtype)_\(nRows)_\(hidden)", device: device)
            if dualQueue, let sOut = sharedOut, let vDone = sharedDoneValue {
                cmd.encodeWaitForEvent(device.syncEvent, value: vDone)
                combined = Ops.add(
                    routed.reshaped(to: [nRows * hidden]),
                    sOut.reshaped(to: [nRows * hidden]),
                    on: cmd, into: combFlat
                ).reshaped(to: [nRows, hidden])
            } else if noDual {
                let sOut = runSharedMLP(on: cmd)
                combined = Ops.add(
                    routed.reshaped(to: [nRows * hidden]),
                    sOut.reshaped(to: [nRows * hidden]),
                    on: cmd, into: combFlat
                ).reshaped(to: [nRows, hidden])
            } else if let sOut = sharedOut {
                combined = Ops.add(
                    routed.reshaped(to: [nRows * hidden]),
                    sOut.reshaped(to: [nRows * hidden]),
                    on: cmd, into: combFlat
                ).reshaped(to: [nRows, hidden])
            }
        }

        // Write layer residual back into `residual` (caller’s hPersist).
        // After addAndRmsNorm, original residual is only needed as h2’s
        // source (already in scratch) — in-place here drops the model
        // loop’s per-layer Ops.copy of [T, hidden] × 80 layers.
        return Ops.add(
            h2.reshaped(to: [nRows * hidden]),
            combined.reshaped(to: [nRows * hidden]),
            on: cmd,
            into: residual.reshaped(to: [nRows * hidden])
        ).reshaped(to: [nRows, hidden])
    }

    /// Public entry so `Hy3Model.forwardMulti` can size workspaces once
    /// before the layer stack (scratchMode off).
    func ensurePrefillWorkspacesPublic(t: Int, device: Device) {
        ensurePrefillWorkspaces(t: t, device: device)
    }

    /// Alias another layer's prefill workspaces (sequential reuse).
    func sharePrefillWorkspaces(from other: Hy3MoELayer) {
        prefillCapT = other.prefillCapT
        pfIndices = other.pfIndices
        pfWeights = other.pfWeights
        pfUnbiased = other.pfUnbiased
        pfBiased = other.pfBiased
        pfSortedExperts = other.pfSortedExperts
        pfSourceT = other.pfSourceT
        pfInvPerm = other.pfInvPerm
        pfXGathered = other.pfXGathered
        pfGateOut = other.pfGateOut
        pfUpOut = other.pfUpOut
        pfDownOut = other.pfDownOut
        pfRouted = other.pfRouted
        pfScaleVec = other.pfScaleVec
        pfActiveExperts = other.pfActiveExperts
        pfRowLo = other.pfRowLo
        pfRowHi = other.pfRowHi
        pfNActive = other.pfNActive
        pfIndirectGU = other.pfIndirectGU
        pfIndirectDown = other.pfIndirectDown
    }

    /// Ensure prefill workspaces fit `t` tokens (persistent across layers).
    private func ensurePrefillWorkspaces(t: Int, device: Device) {
        if prefillCapT >= t, pfGateOut != nil { return }
        // Grow only (never shrink mid-session). Cap at max of prior/current.
        let cap = max(t, prefillCapT, 128)
        prefillCapT = cap
        let topK = router.topK
        let mCap = cap * topK
        let nExperts = router.nExperts
        let dt = dtype
        // Force persistent alloc even if scratchMode is on.
        let was = device.scratchModeActive
        device.scratchModeActive = false
        defer { device.scratchModeActive = was }
        pfIndices = Tensor.empty(shape: [cap, topK], dtype: .u32, device: device)
        pfWeights = Tensor.empty(shape: [cap, topK], dtype: dt, device: device)
        pfUnbiased = Tensor.empty(shape: [cap * nExperts], dtype: dt, device: device)
        pfBiased = Tensor.empty(shape: [cap * nExperts], dtype: dt, device: device)
        pfSortedExperts = Tensor.empty(shape: [mCap], dtype: .u32, device: device)
        pfSourceT = Tensor.empty(shape: [mCap], dtype: .u32, device: device)
        pfInvPerm = Tensor.empty(shape: [mCap], dtype: .u32, device: device)
        pfXGathered = Tensor.empty(shape: [mCap, hidden], dtype: dt, device: device)
        pfGateOut = Tensor.empty(shape: [mCap, moeIntermediate], dtype: dt, device: device)
        pfUpOut = Tensor.empty(shape: [mCap, moeIntermediate], dtype: dt, device: device)
        pfDownOut = Tensor.empty(shape: [mCap, hidden], dtype: dt, device: device)
        pfRouted = Tensor.empty(shape: [cap, hidden], dtype: dt, device: device)
        pfActiveExperts = Tensor.empty(shape: [nExperts], dtype: .u32, device: device)
        pfRowLo = Tensor.empty(shape: [nExperts], dtype: .u32, device: device)
        pfRowHi = Tensor.empty(shape: [nExperts], dtype: .u32, device: device)
        pfNActive = Tensor.empty(shape: [1], dtype: .u32, device: device)
        // MTLDispatchThreadgroupsIndirectArguments = 3×u32
        pfIndirectGU = Tensor.empty(shape: [3], dtype: .u32, device: device)
        pfIndirectDown = Tensor.empty(shape: [3], dtype: .u32, device: device)
        let scale = router.routedScalingFactor
        if scale != 1.0 {
            pfScaleVec = Tensor.filled(scale, shape: [mCap], dtype: dt, device: device)
        } else {
            pfScaleVec = nil
        }
    }

    /// Batched Path B + expert-sorted int2 MMA gather for `t` tokens.
    /// Router stays on-device; plan is `iron_moe_sort_plan` then
    /// `iron_moe_gather_qmm_mma_int2_bm16_*` (affine BM16). No host
    /// index readback. Returns routed sum `[t, hidden]`.
    /// Multi-token Path B MoE. When `fuseShared`, also runs shared gate‖up
    /// on the same concurrent encoder as MoE gate‖up (and shared down
    /// concurrent with MoE down). Returns `(routed, sharedOut?)`.
    private func decodeManyGPU(
        ffnIn: Tensor, t: Int,
        cmd: MTLCommandBuffer, device: Device,
        fuseShared: Bool = false
    ) -> (Tensor, Tensor?) {
        let nExperts = router.nExperts
        let topK = router.topK
        let mTotal = t * topK
        let dt = ffnIn.dtype
        ensurePrefillWorkspaces(t: t, device: device)

        // Views into persistent caps (first t / mTotal rows only).
        let indicesAll = Tensor(
            buffer: pfIndices!.buffer, offset: pfIndices!.offset,
            shape: [t, topK], dtype: .u32)
        let weightsAll = Tensor(
            buffer: pfWeights!.buffer, offset: pfWeights!.offset,
            shape: [t, topK], dtype: dt)
        let flatN = t * nExperts
        let unbiased = Tensor(
            buffer: pfUnbiased!.buffer, offset: pfUnbiased!.offset,
            shape: [flatN], dtype: dt)
        let biased = Tensor(
            buffer: pfBiased!.buffer, offset: pfBiased!.offset,
            shape: [flatN], dtype: dt)
        let sortedExperts = Tensor(
            buffer: pfSortedExperts!.buffer, offset: pfSortedExperts!.offset,
            shape: [mTotal], dtype: .u32)
        let sourceT = Tensor(
            buffer: pfSourceT!.buffer, offset: pfSourceT!.offset,
            shape: [mTotal], dtype: .u32)
        let invPermT = Tensor(
            buffer: pfInvPerm!.buffer, offset: pfInvPerm!.offset,
            shape: [mTotal], dtype: .u32)
        let xGathered = Tensor(
            buffer: pfXGathered!.buffer, offset: pfXGathered!.offset,
            shape: [mTotal, hidden], dtype: dt)
        let gateOut = Tensor(
            buffer: pfGateOut!.buffer, offset: pfGateOut!.offset,
            shape: [mTotal, moeIntermediate], dtype: dt)
        let upOut = Tensor(
            buffer: pfUpOut!.buffer, offset: pfUpOut!.offset,
            shape: [mTotal, moeIntermediate], dtype: dt)
        let downOut = Tensor(
            buffer: pfDownOut!.buffer, offset: pfDownOut!.offset,
            shape: [mTotal, hidden], dtype: dt)
        let outFlat = Tensor(
            buffer: pfRouted!.buffer, offset: pfRouted!.offset,
            shape: [t * hidden], dtype: dt)

        let logitsAll = routerGate.callMany(ffnIn, t: t, on: cmd, device: device)
        Ops.moeSigmoidBiasRows(
            logits: logitsAll.reshaped(to: [flatN]),
            bias: expertBiasGPU,
            unbiased: unbiased, biased: biased,
            tRows: t, nExperts: nExperts, on: cmd)
        Ops.moeRouterTopKBiasedMany(
            scoreBiased: biased, scoreUnbiased: unbiased,
            indicesOut: indicesAll.reshaped(to: [mTotal]),
            weightsOut: weightsAll.reshaped(to: [mTotal]),
            t: t, nExperts: nExperts, k: topK, on: cmd)

        if let scaleVec = pfScaleVec {
            let scaleView = Tensor(
                buffer: scaleVec.buffer, offset: scaleVec.offset,
                shape: [mTotal], dtype: dt)
            _ = Ops.mul(
                weightsAll.reshaped(to: [mTotal]), scaleView, on: cmd,
                into: weightsAll.reshaped(to: [mTotal]))
        }

        let idsFlat = Tensor(
            buffer: indicesAll.buffer, offset: indicesAll.offset,
            shape: [mTotal], dtype: .u32)
        Ops.moeSortPlan(
            topkIds: idsFlat, sortedExperts: sortedExperts,
            sourceTokens: sourceT, invPerm: invPermT,
            mTotal: mTotal, k: topK, on: cmd)
        let wTokenOrder = Tensor(
            buffer: weightsAll.buffer, offset: weightsAll.offset,
            shape: [mTotal], dtype: dt)

        _ = Ops.gather(table: ffnIn, tokenIds: sourceT, on: cmd, into: xGathered)
        let forceM1 = ProcessInfo.processInfo.environment["BUTTER_HY3_MOE_M1"] != nil
        let useBm =
            !forceM1 && (moeIntermediate % 32 == 0) && (hidden % 32 == 0)
        let skipGU =
            ProcessInfo.processInfo.environment["BUTTER_HY3_MOE_SKIP_GU"] != nil
        let skipDown =
            ProcessInfo.processInfo.environment["BUTTER_HY3_MOE_SKIP_DOWN"] != nil
        let serialGU =
            ProcessInfo.processInfo.environment["BUTTER_HY3_MOE_SERIAL_GU"] != nil
        let dualMoeGU =
            ProcessInfo.processInfo.environment["BUTTER_HY3_DUAL_MOE_GU"] != nil
            && ProcessInfo.processInfo.environment["BUTTER_HY3_DUAL_QUEUE"] == nil
        let useSerialGU = serialGU && !dualMoeGU
        let useConcurrentGU = !serialGU && !dualMoeGU

        // Optional shared-expert fuse: pad ffnIn once, concurrent gate‖up with MoE.
        var sharedX: Tensor?
        var sharedSgPad: Tensor?
        var sharedSuPad: Tensor?
        var sharedMPad = 0
        var sharedG: QuantizedLinear?
        var sharedU: QuantizedLinear?
        if fuseShared,
           let g = sharedGate.inner as? QuantizedLinear,
           let u = sharedUp.inner as? QuantizedLinear,
           g.bits == 8, u.bits == 8, t > 0
        {
            sharedG = g
            sharedU = u
            sharedMPad = ((t + 31) / 32) * 32
            let inDimG = g.scales.shape[g.scales.shape.count - 1] * g.groupSize
            if sharedMPad == t {
                sharedX = ffnIn
            } else {
                let xPadBuf = device.intermediateScratch(
                    tag: "hy3_shared_xpad_\(dt)",
                    minBytes: sharedMPad * inDimG * dt.byteSize)
                let xPad = Tensor(
                    buffer: xPadBuf, offset: 0, shape: [sharedMPad, inDimG], dtype: dt)
                let valid = t * inDimG * dt.byteSize
                let tail = (sharedMPad - t) * inDimG * dt.byteSize
                let blit = cmd.makeBlitCommandEncoder()!
                blit.copy(
                    from: ffnIn.buffer, sourceOffset: ffnIn.offset,
                    to: xPad.buffer, destinationOffset: 0, size: valid)
                if tail > 0 {
                    blit.fill(
                        buffer: xPad.buffer, range: valid ..< (valid + tail), value: 0)
                }
                blit.endEncoding()
                sharedX = xPad
            }
            sharedSgPad = hy3ScratchTensor(
                shape: [sharedMPad, g.weight.shape[0]], dtype: dt,
                tag: "hy3_shared_gpad_\(dt)_\(g.weight.shape[0])", device: device)
            sharedSuPad = hy3ScratchTensor(
                shape: [sharedMPad, u.weight.shape[0]], dtype: dt,
                tag: "hy3_shared_upad_\(dt)_\(u.weight.shape[0])", device: device)
        }

        // Opt-in — e2e ties/slight loss vs full EG with early-exit empty
        // TGs at current unique≈80–100.
        let useActiveEG: Bool = {
            guard dt == .f16,
                  pfActiveExperts != nil, pfIndirectGU != nil, pfIndirectDown != nil
            else { return false }
            return ProcessInfo.processInfo.environment["BUTTER_HY3_ACTIVE_EG"] != nil
        }()

        // Zero-fill helper for phase skips (GPU blit, no host thrash).
        func zeroTensor(_ tns: Tensor) {
            let bytes = tns.elementCount * tns.dtype.byteSize
            let blit = cmd.makeBlitCommandEncoder()!
            blit.fill(
                buffer: tns.buffer,
                range: tns.offset ..< (tns.offset + bytes),
                value: 0)
            blit.endEncoding()
        }

        // Shared-expert gate+up dequant-GEMM pair, identical across every
        // MoE dispatch branch below. If `encoder` is given, encodes onto it
        // (caller owns ending the encoder); otherwise opens + closes its
        // own `.concurrent` encoder.
        func encodeSharedGateUp(
            _ g: QuantizedLinear, _ u: QuantizedLinear, _ xS: Tensor,
            _ sgPad: Tensor, _ suPad: Tensor,
            encoder: MTLComputeCommandEncoder? = nil
        ) {
            let ownsEnc = encoder == nil
            guard let enc = encoder ?? cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
            else { return }
            let inDimG = g.scales.shape[g.scales.shape.count - 1] * g.groupSize
            let inDimU = u.scales.shape[u.scales.shape.count - 1] * u.groupSize
            Ops.encodeDequantGemmInt8Mpp(
                input: xS, weight: g.weight, scales: g.scales,
                biases: g.biases, m: sharedMPad, n: g.weight.shape[0],
                k: inDimG, groupSize: g.groupSize, into: sgPad, encoder: enc)
            Ops.encodeDequantGemmInt8Mpp(
                input: xS, weight: u.weight, scales: u.scales,
                biases: u.biases, m: sharedMPad, n: u.weight.shape[0],
                k: inDimU, groupSize: u.groupSize, into: suPad, encoder: enc)
            if ownsEnc { enc.endEncoding() }
        }

        if useBm && useActiveEG,
           let act = pfActiveExperts, let rLo = pfRowLo, let rHi = pfRowHi,
           let nAct = pfNActive, let indGU = pfIndirectGU, let indDown = pfIndirectDown
        {
            Ops.moeCompactActiveExperts(
                sortedExperts: sortedExperts,
                activeExperts: act, rowLo: rLo, rowHi: rHi,
                nActiveOut: nAct,
                indirectGU: indGU, indirectDown: indDown,
                mTotal: mTotal,
                nOutGU: moeIntermediate, nOutDown: hidden,
                nExpertsCap: nExperts,
                on: cmd)
            if skipGU {
                zeroTensor(gateOut); zeroTensor(upOut)
            } else if useConcurrentGU || (!useSerialGU && !dualMoeGU),
               let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
            {
                Ops.moeGatherDequantGemmInt2Active(
                    input: xGathered,
                    weight: switchGate.weight, scales: switchGate.scales,
                    biases: switchGate.biases,
                    activeExperts: act, rowLo: rLo, rowHi: rHi,
                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                    groupSize: groupSize, on: cmd, into: gateOut, encoder: enc,
                    indirectArgs: indGU)
                Ops.moeGatherDequantGemmInt2Active(
                    input: xGathered,
                    weight: switchUp.weight, scales: switchUp.scales,
                    biases: switchUp.biases,
                    activeExperts: act, rowLo: rLo, rowHi: rHi,
                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                    groupSize: groupSize, on: cmd, into: upOut, encoder: enc,
                    indirectArgs: indGU)
                if let g = sharedG, let u = sharedU, let xS = sharedX,
                   let sgPad = sharedSgPad, let suPad = sharedSuPad
                {
                    encodeSharedGateUp(g, u, xS, sgPad, suPad, encoder: enc)
                }
                enc.endEncoding()
            } else {
                Ops.moeGatherDequantGemmInt2Active(
                    input: xGathered,
                    weight: switchGate.weight, scales: switchGate.scales,
                    biases: switchGate.biases,
                    activeExperts: act, rowLo: rLo, rowHi: rHi,
                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                    groupSize: groupSize, on: cmd, into: gateOut,
                    indirectArgs: indGU)
                Ops.moeGatherDequantGemmInt2Active(
                    input: xGathered,
                    weight: switchUp.weight, scales: switchUp.scales,
                    biases: switchUp.biases,
                    activeExperts: act, rowLo: rLo, rowHi: rHi,
                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                    groupSize: groupSize, on: cmd, into: upOut,
                    indirectArgs: indGU)
                if let g = sharedG, let u = sharedU, let xS = sharedX,
                   let sgPad = sharedSgPad, let suPad = sharedSuPad
                {
                    encodeSharedGateUp(g, u, xS, sgPad, suPad)
                }
            }
            _ = Ops.swiglu(gate: gateOut, up: upOut, on: cmd, into: upOut)
            let innerActive = upOut
            var sharedInnerActive: Tensor?
            if let sgPad = sharedSgPad, let suPad = sharedSuPad {
                let sg = sharedMPad == t ? sgPad : sgPad.slicedRows(start: 0, count: t)
                let su = sharedMPad == t ? suPad : suPad.slicedRows(start: 0, count: t)
                sharedInnerActive = Ops.swiglu(gate: sg, up: su, on: cmd, into: su)
            }
            if skipDown {
                zeroTensor(downOut)
            } else {
                Ops.moeGatherDequantGemmInt2Active(
                    input: innerActive,
                    weight: switchDown.weight, scales: switchDown.scales,
                    biases: switchDown.biases,
                    activeExperts: act, rowLo: rLo, rowHi: rHi,
                    mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                    groupSize: groupSize, on: cmd, into: downOut,
                    indirectArgs: indDown)
            }
            let sharedDownActive: Tensor? = sharedInnerActive.map {
                sharedDown.callMany($0, t: t, on: cmd, device: device)
            }
            Ops.moeUnpermute(
                expertOutputs: downOut.reshaped(to: [mTotal * hidden]),
                invPerm: invPermT,
                topKWeights: wTokenOrder,
                into: outFlat,
                nRows: t, hidden: hidden, k: topK,
                on: cmd)
            return (outFlat.reshaped(to: [t, hidden]), sharedDownActive)
        } else if useBm {
            if skipGU {
                zeroTensor(gateOut); zeroTensor(upOut)
            } else if dualMoeGU {
                // Gate on primary CB; up on queue B after gather. Shared
                // expert (if fused) stays on primary concurrent with gate.
                let vGather = device.nextSyncValue()
                cmd.encodeSignalEvent(device.syncEvent, value: vGather)
                let cmdB = device.makeCommandBufferB()
                cmdB.encodeWaitForEvent(device.syncEvent, value: vGather)
                Ops.moeGatherDequantGemmInt2(
                    input: xGathered,
                    weight: switchUp.weight, scales: switchUp.scales,
                    biases: switchUp.biases, indices: sortedExperts,
                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                    groupSize: groupSize, on: cmdB, into: upOut,
                    nExperts: nExperts)
                let vUpDone = device.nextSyncValue()
                cmdB.encodeSignalEvent(device.syncEvent, value: vUpDone)
                cmdB.commit()
                if let g = sharedG, let u = sharedU, let xS = sharedX,
                   let sgPad = sharedSgPad, let suPad = sharedSuPad,
                   let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
                {
                    Ops.moeGatherDequantGemmInt2(
                        input: xGathered,
                        weight: switchGate.weight, scales: switchGate.scales,
                        biases: switchGate.biases, indices: sortedExperts,
                        mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                        groupSize: groupSize, on: cmd, into: gateOut, encoder: enc,
                        nExperts: nExperts)
                    encodeSharedGateUp(g, u, xS, sgPad, suPad, encoder: enc)
                    enc.endEncoding()
                } else {
                    Ops.moeGatherDequantGemmInt2(
                        input: xGathered,
                        weight: switchGate.weight, scales: switchGate.scales,
                        biases: switchGate.biases, indices: sortedExperts,
                        mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                        groupSize: groupSize, on: cmd, into: gateOut,
                        nExperts: nExperts)
                }
                cmd.encodeWaitForEvent(device.syncEvent, value: vUpDone)
            } else if useSerialGU {
                let gs = groupSize
                let mi = moeIntermediate
                let hid = hidden
                Ops.moeGatherDequantGemmInt2(
                    input: xGathered,
                    weight: switchGate.weight, scales: switchGate.scales,
                    biases: switchGate.biases, indices: sortedExperts,
                    mTotal: mTotal, nOut: mi, kIn: hid,
                    groupSize: gs, on: cmd, into: gateOut,
                    nExperts: nExperts)
                Ops.moeGatherDequantGemmInt2(
                    input: xGathered,
                    weight: switchUp.weight, scales: switchUp.scales,
                    biases: switchUp.biases, indices: sortedExperts,
                    mTotal: mTotal, nOut: mi, kIn: hid,
                    groupSize: gs, on: cmd, into: upOut,
                    nExperts: nExperts)
                // Shared gate‖up after serial MoE (no concurrent BW fight).
                if let g = sharedG, let u = sharedU, let xS = sharedX,
                   let sgPad = sharedSgPad, let suPad = sharedSuPad
                {
                    encodeSharedGateUp(g, u, xS, sgPad, suPad)
                }
            } else if useConcurrentGU,
                      let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
            {
                Ops.moeGatherDequantGemmInt2(
                    input: xGathered,
                    weight: switchGate.weight, scales: switchGate.scales,
                    biases: switchGate.biases, indices: sortedExperts,
                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                    groupSize: groupSize, on: cmd, into: gateOut, encoder: enc,
                    nExperts: nExperts)
                Ops.moeGatherDequantGemmInt2(
                    input: xGathered,
                    weight: switchUp.weight, scales: switchUp.scales,
                    biases: switchUp.biases, indices: sortedExperts,
                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                    groupSize: groupSize, on: cmd, into: upOut, encoder: enc,
                    nExperts: nExperts)
                if let g = sharedG, let u = sharedU, let xS = sharedX,
                   let sgPad = sharedSgPad, let suPad = sharedSuPad
                {
                    encodeSharedGateUp(g, u, xS, sgPad, suPad, encoder: enc)
                }
                enc.endEncoding()
            } else {
                Ops.moeGatherDequantGemmInt2(
                    input: xGathered,
                    weight: switchGate.weight, scales: switchGate.scales,
                    biases: switchGate.biases, indices: sortedExperts,
                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                    groupSize: groupSize, on: cmd, into: gateOut,
                    nExperts: nExperts)
                Ops.moeGatherDequantGemmInt2(
                    input: xGathered,
                    weight: switchUp.weight, scales: switchUp.scales,
                    biases: switchUp.biases, indices: sortedExperts,
                    mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                    groupSize: groupSize, on: cmd, into: upOut,
                    nExperts: nExperts)
            }
        } else {
            Ops.moeGatherDequantGemmInt2M1(
                xGathered, switchGate.weight, switchGate.scales, switchGate.biases,
                sortedExperts, mTotal, moeIntermediate, hidden, nExperts, groupSize,
                cmd, gateOut)
            Ops.moeGatherDequantGemmInt2M1(
                xGathered, switchUp.weight, switchUp.scales, switchUp.biases,
                sortedExperts, mTotal, moeIntermediate, hidden, nExperts, groupSize,
                cmd, upOut)
        }
        // SwiGLU into upOut (same shape as gate/up) — avoids a fresh
        // [mTotal, moeInter] alloc per layer. Kernel loads gate/up then
        // stores; up is only used as the multiplicand.
        _ = Ops.swiglu(gate: gateOut, up: upOut, on: cmd, into: upOut)
        let inner = upOut
        var sharedInner: Tensor?
        if let sgPad = sharedSgPad, let suPad = sharedSuPad {
            let sg = sharedMPad == t ? sgPad : sgPad.slicedRows(start: 0, count: t)
            let su = sharedMPad == t ? suPad : suPad.slicedRows(start: 0, count: t)
            // Reuse suPad for shared SwiGLU out (same shape).
            sharedInner = Ops.swiglu(gate: sg, up: su, on: cmd, into: su)
        }

        if skipDown {
            zeroTensor(downOut)
        } else if useBm {
            Ops.moeGatherDequantGemmInt2(
                input: inner,
                weight: switchDown.weight, scales: switchDown.scales,
                biases: switchDown.biases, indices: sortedExperts,
                mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                groupSize: groupSize, on: cmd, into: downOut,
                nExperts: nExperts)
        } else {
            Ops.moeGatherDequantGemmInt2M1(
                inner, switchDown.weight, switchDown.scales, switchDown.biases,
                sortedExperts, mTotal, hidden, moeIntermediate, nExperts, groupSize,
                cmd, downOut)
        }
        let sharedDownOut: Tensor? = sharedInner.map {
            sharedDown.callMany($0, t: t, on: cmd, device: device)
        }

        Ops.moeUnpermute(
            expertOutputs: downOut.reshaped(to: [mTotal * hidden]),
            invPerm: invPermT,
            topKWeights: wTokenOrder,
            into: outFlat,
            nRows: t, hidden: hidden, k: topK,
            on: cmd)
        return (outFlat.reshaped(to: [t, hidden]), sharedDownOut)
    }

    /// 2D view of expert row `ei` in a stacked [E, M, packed] slab.
    private func sliceExpert(_ t: Tensor, _ ei: Int) -> Tensor {
        t.slicedRows(start: ei, count: 1)
            .reshaped(to: [t.shape[1], t.shape[2]])
    }
}

// ─── Model ───────────────────────────────────────────────────────────

public final class Hy3Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    let layers: [any Hy3LayerProtocol]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let dtype: DType
    /// One-shot flag: MoE decode scratches allocated for all layers.
    private var decodeScratchesReady = false
    /// Persistent residual buffer for single-token decode (avoids
    /// makeBuffer every token).
    private var decodeHCarry: Tensor?
    /// Densify bits=8 QuantizedLinears for decode-speed gemv.
    func densifyInt8Linears(device: Device) {
        let cmd = device.makeCommandBuffer()
        func densify(_ lin: AnyLinear) {
            if let q = lin.inner as? QuantizedLinear {
                q.densify(on: cmd, device: device)
            }
        }
        densify(lmHead)
        for layer in layers {
            if let d = layer as? Hy3DenseLayer {
                densify(d.qProj); densify(d.kProj); densify(d.vProj); densify(d.oProj)
                densify(d.gateProj); densify(d.upProj); densify(d.downProj)
            } else if let m = layer as? Hy3MoELayer {
                densify(m.qProj); densify(m.kProj); densify(m.vProj); densify(m.oProj)
                densify(m.routerGate)
                densify(m.sharedGate); densify(m.sharedUp); densify(m.sharedDown)
            }
        }
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    public var defaultPrefillStepSize: Int { 4096 }
    public var requiresLeadingBOS: Bool { false }

    init(
        embedTokens: AnyEmbedding,
        layers: [any Hy3LayerProtocol],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int,
        headDim: Int, vocab: Int, maxSeq: Int, dtype: DType
    ) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.vocab = vocab
        self.maxSeq = maxSeq
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() { out.append(("model.embed_tokens.\(k)", v)) }
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("model.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("model.norm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        return (0 ..< nLayers).map { _ in
            KVCache(
                nKVHeads: nKVHeads, headDim: headDim, contextLength: cap,
                dtype: dtype, eviction: .unbounded, device: device)
        }
    }

    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        // Per-layer commit without wait so encode overlaps GPU. Need
        // command-buffer pool ≥ nLayers (80). Default 64 stalls the tail.
        if !decodeScratchesReady {
            for layer in layers {
                if let moe = layer as? Hy3MoELayer {
                    moe.ensureScratchesForDecode(device: device)
                }
            }
            decodeScratchesReady = true
        }

        var workCmd = device.makeCommandBuffer()
        var lastCommitted: MTLCommandBuffer?
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        for (i, layer) in layers.enumerated() {
            h = layer.forward(
                h, position: position,
                cache: caches[i] as! any KVCacheProtocol,
                cmd: workCmd, device: device)
            if workCmd.status != .committed && workCmd.status != .completed
                && workCmd.status != .error
            {
                workCmd.commit()
            }
            lastCommitted = workCmd
            workCmd = device.makeCommandBuffer()
            _ = i
        }
        lastCommitted?.waitUntilCompleted()

        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(!tokenIds.isEmpty, "Hy3Model.forwardMulti: tokenIds must be non-empty")
        let n = tokenIds.count
        // u32 token buffer
        let tokBuf = device.makeBuffer(length: n * 4)
        var ids = tokenIds.map { UInt32($0) }
        ids.withUnsafeBytes {
            _ = memcpy(tokBuf.contents(), $0.baseAddress!, n * 4)
        }
        let tokenTensor = Tensor(buffer: tokBuf, offset: 0, shape: [n], dtype: .u32)

        // Shared MoE prefill workspaces (one set for all layers) +
        // per-layer commit-without-wait so encode overlaps GPU.
        // Avoid dual 1 GB scratch slabs (swap thrash with 75 GB weights).
        if let firstMoE = layers.compactMap({ $0 as? Hy3MoELayer }).first {
            firstMoE.ensurePrefillWorkspacesPublic(t: n, device: device)
            for layer in layers {
                if let moe = layer as? Hy3MoELayer, moe !== firstMoE {
                    moe.sharePrefillWorkspaces(from: firstMoE)
                }
            }
        }
        let profile = ProcessInfo.processInfo.environment["BUTTER_HY3_PROFILE"] != nil
        // Grow-only prefill scratch slab (opt-in). Default OFF — e2e with
        // default ON added ~4 GB and regressed full prefill (~156 vs ~264)
        // under residual swap; MoE pf* workspaces already avoid per-layer
        // MoE alloc.
        let useScratch =
            ProcessInfo.processInfo.environment["BUTTER_HY3_SCRATCH"] != nil
        if useScratch {
            // ~160 KiB/token/layer + 10% headroom; MoE pf* is persistent.
            let bytesPerLayer = max(8 << 20, n * 160 * 1024)
            let slab = min(4 << 30, max(256 << 20, (bytesPerLayer * layers.count * 11) / 10))
            device.ensureScratchSlab(slab)
            device.scratchModeActive = true
            if profile {
                fputs(
                    "[hy3] prefill scratch slab \(slab / (1 << 20)) MB "
                        + "(T=\(n) L=\(layers.count))\n", stderr)
            }
        }
        defer {
            if useScratch {
                device.scratchModeActive = false
                device.resetScratch()
            }
        }

        var workCmd = device.makeCommandBuffer()
        var lastCommitted: MTLCommandBuffer?
        var h = embedTokens(tokenTensor, on: workCmd)
        if h.shape.count == 1 {
            h = h.reshaped(to: [n, hidden])
        }
        // Residual carry must outlive scratch (and survive layer loop).
        let wasScratch = device.scratchModeActive
        device.scratchModeActive = false
        let hPersist = Tensor.empty(shape: [n, hidden], dtype: dtype, device: device)
        device.scratchModeActive = wasScratch
        Ops.copy(
            h.reshaped(to: [n * hidden]), into: hPersist.reshaped(to: [n * hidden]),
            on: workCmd)
        h = hPersist

        let megaCB =
            ProcessInfo.processInfo.environment["BUTTER_HY3_MEGA_CB"] != nil
        let cbStride: Int = {
            if megaCB { return layers.count }
            if let s = ProcessInfo.processInfo.environment["BUTTER_HY3_CB_STRIDE"],
               let n = Int(s), n > 0
            {
                return n
            }
            return 4
        }()
        var tEncode: CFTimeInterval = 0
        let tStack0 = CFAbsoluteTimeGetCurrent()
        for (i, layer) in layers.enumerated() {
            let t0 = CFAbsoluteTimeGetCurrent()
            let hNext = layer.forwardMulti(
                h, startingAt: position,
                cache: caches[i] as! any KVCacheProtocol,
                cmd: workCmd, device: device)
            if hNext.buffer !== hPersist.buffer || hNext.offset != hPersist.offset {
                Ops.copy(
                    hNext.reshaped(to: [n * hidden]),
                    into: hPersist.reshaped(to: [n * hidden]),
                    on: workCmd)
            }
            h = hPersist
            let shouldCommit =
                !megaCB
                && ((i + 1) % cbStride == 0 || i + 1 == layers.count)
            if shouldCommit {
                if workCmd.status != .committed && workCmd.status != .completed
                    && workCmd.status != .error
                {
                    workCmd.commit()
                }
                lastCommitted = workCmd
                if i + 1 < layers.count {
                    workCmd = device.makeCommandBuffer()
                }
            }
            tEncode += CFAbsoluteTimeGetCurrent() - t0
            if profile && (i == 0 || i + 1 == layers.count || i % 20 == 0) {
                fputs(
                    String(
                        format: "[hy3-profile] encode layer %d: %.2f ms\n", i,
                        (CFAbsoluteTimeGetCurrent() - t0) * 1000),
                    stderr)
            }
        }
        if megaCB {
            if workCmd.status != .committed && workCmd.status != .completed
                && workCmd.status != .error
            {
                workCmd.commit()
            }
            lastCommitted = workCmd
        }
        let tWait0 = CFAbsoluteTimeGetCurrent()
        lastCommitted?.waitUntilCompleted()
        if profile {
            fputs(
                String(
                    format:
                        "[hy3-profile] encode_sum=%.1f ms wait=%.1f ms wall=%.1f ms T=%d L=%d\n",
                    tEncode * 1000, (CFAbsoluteTimeGetCurrent() - tWait0) * 1000,
                    (CFAbsoluteTimeGetCurrent() - tStack0) * 1000, n, layers.count),
                stderr)
        }
        // Logits at last position only.
        let last = h.slicedRows(start: n - 1, count: 1).reshaped(to: [hidden])
        let normed = finalNorm(last, on: cmd)
        return lmHead(normed, on: cmd)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Per-head weighted RMSNorm over the last `headDim` axis — QK-norm.
private func hy3ApplyPerHeadRMSNorm(
    _ x: Tensor, weight: Tensor, eps: Float,
    nHeads: Int, headDim: Int,
    on cmd: MTLCommandBuffer, device: Device
) -> Tensor {
    // x is [nHeads * headDim] flat at decode T=1.
    precondition(x.elementCount == nHeads * headDim)
    _ = device
    return Ops.rmsNormRows(
        x, weight: weight, eps: eps,
        nRows: nHeads, rowSize: headDim, on: cmd)
}

/// GPU blit-zero into a reusable intermediateScratch slab (probe/skip paths).
private func hy3GpuZeros(
    shape: [Int], dtype: DType, tag: String,
    cmd: MTLCommandBuffer, device: Device
) -> Tensor {
    let t = hy3ScratchTensor(shape: shape, dtype: dtype, tag: tag, device: device)
    let bytes = t.elementCount * t.dtype.byteSize
    let blit = cmd.makeBlitCommandEncoder()!
    blit.fill(buffer: t.buffer, range: t.offset ..< (t.offset + bytes), value: 0)
    blit.endEncoding()
    return t
}

/// Grow-once tagged intermediate for per-layer free-path pads (QKV/O/SDPA).
/// Avoids `Tensor.empty` → fresh `makeBuffer` every layer × 80 (driver thrash
/// and RSS spikes). Safe across CB stride: buffer stays alive for the process.
private func hy3ScratchTensor(
    shape: [Int], dtype: DType, tag: String, device: Device
) -> Tensor {
    let n = shape.reduce(1, *)
    let bytes = max(n * dtype.byteSize, 64)
    let buf = device.intermediateScratch(tag: tag, minBytes: bytes)
    return Tensor(buffer: buf, offset: 0, shape: shape, dtype: dtype)
}

/// Pack host f32 expert_bias into a GPU tensor of activation dtype.
private func hy3BiasTensor(
    _ values: [Float], dtype: DType, device: Device
) -> Tensor {
    let t = Tensor.empty(shape: [values.count], dtype: dtype, device: device)
    switch dtype {
    case .f32:
        t.copyIn(from: values)
    case .f16:
        t.copyIn(from: values.map { Float16($0) })
    case .bf16:
        // Round-to-nearest-even bf16 (matches Tensor.filled).
        let packed: [UInt16] = values.map { v in
            let bits = v.bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            return UInt16(rounded >> 16)
        }
        t.copyIn(from: packed)
    default:
        fatalError("hy3BiasTensor: unsupported dtype \(dtype)")
    }
    return t
}

/// In-place GPU bf16→f16 when `preferF16` and tensor is bf16; else identity.
private func hy3MaybeF16Tensor(
    _ t: Tensor, preferF16: Bool, cmd: MTLCommandBuffer?
) -> Tensor {
    guard preferF16, t.dtype == .bf16, let cmd else { return t }
    return Ops.castBf16ToF16(t, on: cmd)
}

private func hy3MaybeF16Linear(
    _ lin: AnyLinear, preferF16: Bool, cmd: MTLCommandBuffer?, device: Device
) throws -> AnyLinear {
    _ = device
    guard preferF16, let cmd else { return lin }
    if let q = lin.inner as? QuantizedLinear {
        guard q.scales.dtype == .bf16 || q.biases.dtype == .bf16 else { return lin }
        let scales =
            q.scales.dtype == .bf16 ? Ops.castBf16ToF16(q.scales, on: cmd) : q.scales
        let biases =
            q.biases.dtype == .bf16 ? Ops.castBf16ToF16(q.biases, on: cmd) : q.biases
        let ab: Tensor? = {
            guard let b = q.additiveBias else { return nil }
            return b.dtype == .bf16 ? Ops.castBf16ToF16(b, on: cmd) : b
        }()
        return AnyLinear(
            QuantizedLinear(
                weight: q.weight, scales: scales, biases: biases,
                bits: q.bits, groupSize: q.groupSize, additiveBias: ab))
    }
    if let dense = lin.inner as? Linear {
        guard dense.weight.dtype == .bf16 else { return lin }
        let w = Ops.castBf16ToF16(dense.weight, on: cmd)
        let b = dense.bias.map {
            $0.dtype == .bf16 ? Ops.castBf16ToF16($0, on: cmd) : $0
        }
        return AnyLinear(Linear(weight: w, bias: b))
    }
    return lin
}

private func hy3MaybeF16Triplet(
    _ t: SafeTensorsBundle.QuantizedTriplet,
    preferF16: Bool, cmd: MTLCommandBuffer?
) -> SafeTensorsBundle.QuantizedTriplet {
    guard preferF16, let cmd else { return t }
    guard t.scales.dtype == .bf16 || t.biases.dtype == .bf16 else { return t }
    let scales =
        t.scales.dtype == .bf16 ? Ops.castBf16ToF16(t.scales, on: cmd) : t.scales
    let biases =
        t.biases.dtype == .bf16 ? Ops.castBf16ToF16(t.biases, on: cmd) : t.biases
    return SafeTensorsBundle.QuantizedTriplet(
        weight: t.weight, scales: scales, biases: biases)
}

private func hy3MaybeF16Embedding(
    _ emb: AnyEmbedding, preferF16: Bool, cmd: MTLCommandBuffer?, device: Device
) throws -> AnyEmbedding {
    _ = device
    guard preferF16, let cmd else { return emb }
    if let q = emb.inner as? QuantizedEmbedding {
        guard q.scales.dtype == .bf16 || q.biases.dtype == .bf16 else { return emb }
        let scales =
            q.scales.dtype == .bf16 ? Ops.castBf16ToF16(q.scales, on: cmd) : q.scales
        let biases =
            q.biases.dtype == .bf16 ? Ops.castBf16ToF16(q.biases, on: cmd) : q.biases
        return AnyEmbedding(
            QuantizedEmbedding(
                weight: q.weight, scales: scales, biases: biases,
                hidden: q.hidden, bits: q.bits, groupSize: q.groupSize))
    }
    if let e = emb.inner as? Embedding {
        guard e.weight.dtype == .bf16 else { return emb }
        return AnyEmbedding(
            Embedding(weight: Ops.castBf16ToF16(e.weight, on: cmd)))
    }
    return emb
}


