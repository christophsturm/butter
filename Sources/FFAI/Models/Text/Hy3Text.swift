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
// Wires the gaps identified against vLLM / llama.cpp:
//   1. Router Path B via `MoEGatingMode.sigmoidBiasedTopK`
//   2. Ungated shared SwiGLU expert added to routed sum
//   3. QK-RMSNorm *before* RoPE (vLLM `NORM_THEN_ROPE`)
//
// Experts are stored as stacked mlx affine tensors
// (`mlp.switch_mlp.{gate,up,down}_proj`, shape [E, M, packed_K]). Decode
// slices one expert row and runs `Ops.dequantGemv` (int2/int4/int8).
// MTP (`num_nextn_predict_layers`) is intentionally not loaded yet.

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
        // next to the 92 GB oQ2 weights on a 128 GB Mac. Default to 8192
        // (~1.3 GB); override with LoadOptions.maxContextLength for longer.
        let defaultMaxSeq = 8192
        let maxSeq =
            options.maxContextLength
            ?? min(tc.int("max_position_embeddings") ?? defaultMaxSeq, defaultMaxSeq)
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

        // Prefer bf16 activations when the bundle is mlx-quantized (oQ2).
        let dtype: DType = quant != nil ? .bf16 : .f16

        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant)

        var layers: [any Hy3LayerProtocol] = []
        layers.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let p = "model.layers.\(i)"
            let qProj = try loadLinear(
                base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
            let kProj = try loadLinear(
                base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
            let vProj = try loadLinear(
                base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
            let oProj = try loadLinear(
                base: "\(p).self_attn.o_proj", in: weights, quantization: quant)

            let qNorm: RMSNorm?
            let kNorm: RMSNorm?
            if useQkNorm {
                qNorm = RMSNorm(
                    weight: try weights.tensor(named: "\(p).self_attn.q_norm.weight"),
                    eps: eps)
                kNorm = RMSNorm(
                    weight: try weights.tensor(named: "\(p).self_attn.k_norm.weight"),
                    eps: eps)
            } else {
                qNorm = nil
                kNorm = nil
            }

            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                eps: eps)
            let postAttnNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: eps)

            if i < firstKDense || nExperts == 0 {
                // Dense SwiGLU MLP.
                let gateProj = try loadLinear(
                    base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
                let upProj = try loadLinear(
                    base: "\(p).mlp.up_proj", in: weights, quantization: quant)
                let downProj = try loadLinear(
                    base: "\(p).mlp.down_proj", in: weights, quantization: quant)
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
                let gate = try loadLinear(
                    base: "\(p).mlp.router.gate", in: weights, quantization: quant)
                let biasTensor = try weights.tensor(named: "\(p).mlp.router.expert_bias")
                let expertBias = biasTensor.toFloatArray()
                precondition(
                    expertBias.count == nExperts,
                    "Hy3: expert_bias count \(expertBias.count) ≠ nExperts \(nExperts)")

                let switchGate = try weights.quantizedTriplet("\(p).mlp.switch_mlp.gate_proj")
                let switchUp = try weights.quantizedTriplet("\(p).mlp.switch_mlp.up_proj")
                let switchDown = try weights.quantizedTriplet("\(p).mlp.switch_mlp.down_proj")
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

                let sharedGate = try loadLinear(
                    base: "\(p).mlp.shared_mlp.gate_proj", in: weights, quantization: quant)
                let sharedUp = try loadLinear(
                    base: "\(p).mlp.shared_mlp.up_proj", in: weights, quantization: quant)
                let sharedDown = try loadLinear(
                    base: "\(p).mlp.shared_mlp.down_proj", in: weights, quantization: quant)

                let router = MoERouter(
                    nExperts: nExperts, topK: topK,
                    gatingMode: .sigmoidBiasedTopK,
                    normTopKProb: routeNorm,
                    expertBias: expertBias,
                    routedScalingFactor: routerScale)

                layers.append(
                    Hy3MoELayer(
                        qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                        qNorm: qNorm, kNorm: kNorm,
                        inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                        routerGate: gate, router: router,
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
            weight: try weights.tensor(named: "model.norm.weight"),
            eps: eps)

        let lmHead: AnyLinear
        if weights.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: weights, quantization: quant)
        } else if !weights.isQuantized("model.embed_tokens") {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        } else {
            throw Hy3Error.missingTensor("lm_head.weight (tied quantized embed not supported)")
        }

        _ = options
        return Hy3Model(
            embedTokens: embedTokens, layers: layers, finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads, nKVHeads: nKVHeads,
            headDim: headDim, vocab: vocab, maxSeq: maxSeq, dtype: dtype)
    }
}

// ─── Layer protocol ──────────────────────────────────────────────────

protocol Hy3LayerProtocol: Module {
    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
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

        // QK-norm *before* RoPE (vLLM NORM_THEN_ROPE / llama.cpp order).
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
            nKV: cache.length, kvStride: cache.maxSeq,
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
}

// ─── MoE layer ───────────────────────────────────────────────────────

/// Stacked mlx-affine expert weights: shape [nExperts, outDim, packedIn].
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
    let switchGate, switchUp, switchDown: Hy3StackedQuant
    let gateBits, upBits, downBits, groupSize: Int
    let sharedGate, sharedUp, sharedDown: AnyLinear
    let hidden, moeIntermediate, nHeads, nKVHeads, headDim: Int
    let ropeTheta: Float
    let dtype: DType
    let scale: Float
    let nShared: Int

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        qNorm: RMSNorm?, kNorm: RMSNorm?,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        routerGate: AnyLinear, router: MoERouter,
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
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd)
        let attn = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        let h2 = Ops.add(residual, attn, on: cmd)

        // ── MoE FFN: Path B router on CPU after gate gemv sync ───────
        let ffnIn = postAttnNorm(h2, on: cmd)
        let logitsTensor = routerGate(ffnIn, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // toFloatArray handles f16/bf16/f32 gate logits.
        let logits = logitsTensor.toFloatArray()
        let routing = router.route(logits: logits)

        let workCmd = device.makeCommandBuffer()
        let nSlots = routing.indices.count
        let moeInter = switchGate.weight.shape[1]

        // Slice all expert rows once (host views into stacked slabs).
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

        // One encoder for all gate+up gemvs (2·nSlots dispatches), one for downs.
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

        // Ungated shared expert — always on (vLLM / llama.cpp).
        if nShared > 0 {
            let sg = sharedGate(ffnIn, on: workCmd)
            let su = sharedUp(ffnIn, on: workCmd)
            let inner = Ops.swiglu(gate: sg, up: su, on: workCmd)
            let sharedOut = sharedDown(inner, on: workCmd)
            accumulator =
                accumulator.map { Ops.add($0, sharedOut, on: workCmd) } ?? sharedOut
        }

        let routed = accumulator!
        let result = Ops.add(h2, routed, on: workCmd)
        workCmd.commit()
        workCmd.waitUntilCompleted()
        return result
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
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
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

        // Private work buffer — Hy3MoELayer commits for router CPU
        // readback. Never touch the caller's `cmd` until final norm +
        // lm_head (LanguageModel.forward commits that once).
        // Mirrors NemotronH / Qwen3x MoE-bearing decode.
        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.forward(
                h, position: position,
                cache: caches[i] as! any KVCacheProtocol,
                cmd: workCmd, device: device)
            // MoE commits `workCmd` for the router readback — refresh so
            // the next layer does not encode onto a committed buffer.
            if layer is Hy3MoELayer {
                workCmd = device.makeCommandBuffer()
            }
        }

        // Flush any trailing non-MoE work so `h` is GPU-resident before
        // we queue final ops on the caller's pristine `cmd`.
        workCmd.commit()
        workCmd.waitUntilCompleted()

        let normed = finalNorm(h, on: cmd)
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
