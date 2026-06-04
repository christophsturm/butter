// Copyright 2026 Eric Kryski (@ekryski)
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
// Gemma 4 VL — vision tower internals + family orchestrator.
//
// This file contains `Gemma4VLVisionConfig`, `Gemma4VLVisionBlock`,
// `Gemma4VLVisionModel`, `Gemma4VLComposedEncoder`, and all supporting
// CPU / GPU helpers for the bespoke Gemma 4 ViT (flattened linear
// patch-embed, learned 2D position tables, multi-dimensional RoPE MHA
// with per-projection q/k/v RMSNorms, GemmaRMSNorm zero-shift norms,
// SwiGLU MLP, attention pooling, multi-modal embedder). The Gemma 4 VL
// vision-language orchestrator (`enum Gemma4VL`) — which ties the
// Gemma 4 text backbone to this vision tower — lives at the bottom of
// this file. The family entry-point (`enum Gemma4`, `Gemma4Variant`,
// `Gemma4Error`) lives in `Models/Gemma4.swift`.

import Foundation
import Metal

// ─── Gemma 4 vision tower ────────────────────────────────────────────

/// Static shape of the Gemma 4 vision tower, decoded from
/// `vision_config`.
struct Gemma4VLVisionConfig {
    let depth: Int
    let hidden: Int
    let intermediate: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let patchSize: Int
    let rmsNormEps: Float
    let defaultOutputLength: Int
    let positionEmbeddingSize: Int
    let poolingKernelSize: Int
    let standardize: Bool
    let ropeTheta: Float

    static func decode(_ c: ModelConfig) throws -> Gemma4VLVisionConfig {
        guard let depth = c.int("num_hidden_layers"),
            let hidden = c.int("hidden_size"),
            let numHeads = c.int("num_attention_heads"),
            let patchSize = c.int("patch_size")
        else {
            throw Gemma4Error.missingConfig("vision_config")
        }
        let numKVHeads = c.int("num_key_value_heads") ?? numHeads
        let headDim = c.int("head_dim") ?? (hidden / numHeads)
        let intermediate = c.int("intermediate_size") ?? hidden * 4
        // The vision RoPE base frequency lives under `rope_parameters`;
        // default 100.0 per the Gemma 4 vision config.
        var ropeTheta: Float = 100.0
        if let rp = c.nested("rope_parameters") {
            if let t = rp["rope_theta"] as? Double {
                ropeTheta = Float(t)
            } else if let t = rp["rope_theta"] as? Int {
                ropeTheta = Float(t)
            }
        }
        return Gemma4VLVisionConfig(
            depth: depth, hidden: hidden, intermediate: intermediate,
            numHeads: numHeads, numKVHeads: numKVHeads, headDim: headDim,
            patchSize: patchSize,
            rmsNormEps: Float(c.float("rms_norm_eps") ?? 1e-6),
            defaultOutputLength: c.int("default_output_length") ?? 280,
            positionEmbeddingSize: c.int("position_embedding_size") ?? 10_240,
            poolingKernelSize: c.int("pooling_kernel_size") ?? 3,
            standardize: c.bool("standardize") ?? false,
            ropeTheta: ropeTheta)
    }
}

/// Per-projection scalar clamp bounds for a Gemma 4 clipped linear
/// (`Gemma4ClippableLinear`, `use_clipped_linears=True`). Each projection
/// clamps its input to `[inputMin, inputMax]` and its output to
/// `[outputMin, outputMax]` — scalar bounds baked into the checkpoint
/// (`*.input_min` / `*.input_max` / `*.output_min` / `*.output_max`).
struct Gemma4VisionClip {
    let inputMin: Float
    let inputMax: Float
    let outputMin: Float
    let outputMax: Float
}

/// The seven clipped-linear bound sets for one vision block. `nil` entries
/// fall back to a plain (unclamped) projection — a checkpoint that doesn't
/// ship clip bounds therefore behaves exactly as before.
struct Gemma4VisionBlockClips {
    var q: Gemma4VisionClip?
    var k: Gemma4VisionClip?
    var v: Gemma4VisionClip?
    var o: Gemma4VisionClip?
    var gate: Gemma4VisionClip?
    var up: Gemma4VisionClip?
    var down: Gemma4VisionClip?
    static let none = Gemma4VisionBlockClips()
}

/// One Gemma 4 vision block: GemmaRMSNorm → RoPE MHA + q/k/v norms →
/// GemmaRMSNorm + residual, GemmaRMSNorm → SwiGLU MLP → GemmaRMSNorm +
/// residual. Held as plain weight tensors; the forward runs CPU
/// attention + GPU GEMMs.
final class Gemma4VLVisionBlock {
    /// The four per-block GemmaRMSNorms (the `+1` zero-shift is folded
    /// into the loaded weight).
    let inputNorm: RMSNorm
    let postAttnNorm: RMSNorm
    let preFFNorm: RMSNorm
    let postFFNorm: RMSNorm
    let qProj, kProj, vProj, oProj: Linear
    /// Per-head q / k RMSNorms (weighted) and the unweighted v-norm eps.
    let qNorm: RMSNorm
    let kNorm: RMSNorm
    let gate, up, down: Linear
    /// Optional per-projection clamp bounds (Gemma 4 clipped linears).
    let clips: Gemma4VisionBlockClips
    let cfg: Gemma4VLVisionConfig
    /// SwiGLU intermediate rounded up to the GEMM K-tile width.
    let paddedIntermediate: Int

    init(
        inputNorm: RMSNorm, postAttnNorm: RMSNorm, preFFNorm: RMSNorm,
        postFFNorm: RMSNorm, qProj: Linear, kProj: Linear, vProj: Linear,
        oProj: Linear, qNorm: RMSNorm, kNorm: RMSNorm,
        gate: Linear, up: Linear, down: Linear,
        clips: Gemma4VisionBlockClips = .none,
        paddedIntermediate: Int, cfg: Gemma4VLVisionConfig
    ) {
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
        self.preFFNorm = preFFNorm
        self.postFFNorm = postFFNorm
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.gate = gate
        self.up = up
        self.down = down
        self.clips = clips
        self.paddedIntermediate = paddedIntermediate
        self.cfg = cfg
    }

    /// Forward `[nTokens, hidden]` activations through one block.
    /// `xPos` / `yPos` are the per-token grid coordinates driving the
    /// multi-dimensional vision RoPE.
    func forward(
        _ h: Tensor, nTokens: Int, xPos: [Int], yPos: [Int],
        device: Device
    ) -> Tensor {
        let hidden = cfg.hidden
        // ── Attention sub-block ──
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(
            h, weight: inputNorm.weight,
            eps: inputNorm.eps, nRows: nTokens,
            rowSize: hidden, on: cmd)
        let q = projectClipped(
            qProj, normed, clip: clips.q, nTokens: nTokens,
            outDim: cfg.numHeads * cfg.headDim, on: cmd)
        let k = projectClipped(
            kProj, normed, clip: clips.k, nTokens: nTokens,
            outDim: cfg.numKVHeads * cfg.headDim, on: cmd)
        let v = projectClipped(
            vProj, normed, clip: clips.v, nTokens: nTokens,
            outDim: cfg.numKVHeads * cfg.headDim, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let attn = attention(
            q: q, k: k, v: v, nTokens: nTokens,
            xPos: xPos, yPos: yPos, device: device)

        // ── Residual + MLP sub-block ──
        let cmd2 = device.makeCommandBuffer()
        let attnProj = projectClipped(
            oProj, attn, clip: clips.o, nTokens: nTokens,
            outDim: hidden, on: cmd2)
        let postAttn = Ops.rmsNormRows(
            attnProj, weight: postAttnNorm.weight,
            eps: postAttnNorm.eps, nRows: nTokens,
            rowSize: hidden, on: cmd2)
        let afterAttn = Ops.add(h, postAttn, on: cmd2)
        let preFF = Ops.rmsNormRows(
            afterAttn, weight: preFFNorm.weight,
            eps: preFFNorm.eps, nRows: nTokens,
            rowSize: hidden, on: cmd2)
        let g = projectClipped(
            gate, preFF, clip: clips.gate, nTokens: nTokens,
            outDim: paddedIntermediate, on: cmd2)
        let u = projectClipped(
            up, preFF, clip: clips.up, nTokens: nTokens,
            outDim: paddedIntermediate, on: cmd2)
        let act = Ops.gelu(g, on: cmd2)
        let gated = Ops.mul(act, u, on: cmd2)
        let d = projectClipped(
            down, gated, clip: clips.down, nTokens: nTokens,
            outDim: hidden, on: cmd2)
        let postFF = Ops.rmsNormRows(
            d, weight: postFFNorm.weight,
            eps: postFFNorm.eps, nRows: nTokens,
            rowSize: hidden, on: cmd2)
        let result = Ops.add(afterAttn, postFF, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return result
    }

    /// Full bidirectional multi-head attention with per-head q / k RMSNorm
    /// + multi-dimensional vision RoPE — fully GPU-resident. `q` / `k` / `v`
    /// are token-major `[nTokens, nHeads·headDim]` (q) / `[nTokens,
    /// nKVHeads·headDim]` (k, v). Returns the context token-major
    /// `[nTokens, nHeads·headDim]`.
    ///
    /// Pipeline (one command buffer, no host bounce):
    ///   1. Per-head RMSNorm — `Ops.rmsNormRows` over `[nTokens·heads,
    ///      headDim]` rows: q / k weighted by their per-head norm, v
    ///      unweighted (a ones vector, eps = `rmsNormEps`).
    ///   2. 2D vision RoPE — `Ops.rope2D` on q and k (token-major); the
    ///      kernel's row/col split + rotate-half math is byte-identical to
    ///      the previous CPU `applyRoPE` (verified against `rope_2d.rs`).
    ///   3. Token-major → head-major transpose of k and v
    ///      (`Ops.transposeTokensToHeads`) — the K/V layout
    ///      `Ops.sdpaBidirectional` reads.
    ///   4. `Ops.sdpaBidirectional` — full bidirectional GQA attention,
    ///      scale 1.0 (the q_norm folds the 1/√d), baseKV 0, kvStride ==
    ///      nTokens.
    ///
    /// The `ffai_sdpa_bidirectional_d{64,72,80,96}` kernels are GPU-correct
    /// at this production geometry — see `sdpa_bidirectional_gpu_correctness
    /// .rs` (incl. the 576×16×64 case) — with TPG 1024 / n_simd 32, so the
    /// machine-freeze the earlier CPU `concurrentPerform` fallback worked
    /// around (the 2026-05-27 GlmOcr bisect) no longer applies.
    private func attention(
        q: Tensor, k: Tensor, v: Tensor, nTokens: Int,
        xPos: [Int], yPos: [Int],
        device: Device
    ) -> Tensor {
        let nHeads = cfg.numHeads
        let nKVHeads = cfg.numKVHeads
        let headDim = cfg.headDim
        let dtype = q.dtype

        // Packed `[nTokens, 2]` u32 (row, col) positions for `rope2D`.
        var posFlat = [UInt32](repeating: 0, count: nTokens * 2)
        for t in 0 ..< nTokens {
            posFlat[t * 2] = UInt32(xPos[t])
            posFlat[t * 2 + 1] = UInt32(yPos[t])
        }
        let positions = Tensor.empty(
            shape: [nTokens * 2], dtype: .u32, device: device)
        positions.copyIn(from: posFlat)

        // Unweighted v-norm: a ones `[headDim]` weight in the activation
        // dtype (the old CPU path passed `weight: nil`).
        let onesWeight = Tensor.filled(
            1.0, shape: [headDim], dtype: dtype, device: device)

        let cmd = device.makeCommandBuffer()
        // 1. Per-head RMSNorm (q / k weighted, v unweighted).
        let qNormed = Ops.rmsNormRows(
            q, weight: qNorm.weight, eps: qNorm.eps,
            nRows: nTokens * nHeads, rowSize: headDim, on: cmd)
        let kNormed = Ops.rmsNormRows(
            k, weight: kNorm.weight, eps: kNorm.eps,
            nRows: nTokens * nKVHeads, rowSize: headDim, on: cmd)
        let vNormed = Ops.rmsNormRows(
            v, weight: onesWeight, eps: cfg.rmsNormEps,
            nRows: nTokens * nKVHeads, rowSize: headDim, on: cmd)
        // 2. 2D vision RoPE on q and k (token-major).
        let qRoped = Ops.rope2D(
            qNormed, positions: positions,
            nTokens: nTokens, nHeads: nHeads, headDim: headDim,
            thetaBase: cfg.ropeTheta, on: cmd)
        let kRoped = Ops.rope2D(
            kNormed, positions: positions,
            nTokens: nTokens, nHeads: nKVHeads, headDim: headDim,
            thetaBase: cfg.ropeTheta, on: cmd)
        // 3. Transpose k, v token-major → head-major for the SDPA contract.
        let kHead = Ops.transposeTokensToHeads(
            kRoped, nTokens: nTokens, nHeads: nKVHeads, headDim: headDim,
            on: cmd)
        let vHead = Ops.transposeTokensToHeads(
            vNormed, nTokens: nTokens, nHeads: nKVHeads, headDim: headDim,
            on: cmd)
        // 4. Full bidirectional GQA attention. Vision uses scale 1.0 (the
        //    q_norm already folds the 1/√d); baseKV 0, kvStride == nTokens.
        let attnOut = Ops.sdpaBidirectional(
            q: qRoped, k: kHead, v: vHead,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            baseKV: 0, nQuery: nTokens, kvStride: nTokens,
            scale: 1.0, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        // attnOut is `[nTokens, nHeads, headDim]` = `[nTokens, nHeads·headDim]`
        // contiguous — byte-identical to what o_proj's GEMM consumes.
        return attnOut.reshaped(to: [nTokens, nHeads * headDim])
    }

    /// Apply a `Gemma4ClippableLinear`: clamp the input to its scalar
    /// `[inputMin, inputMax]`, project, then clamp the output to
    /// `[outputMin, outputMax]` — all GPU-resident on `cmd`. A `nil` clip
    /// falls back to the plain `projectRows` (unclamped). The reference's
    /// `use_clipped_linears` is on for the Gemma 4 vision tower; without
    /// the clamps the (large) pre-pool activations diverge.
    private func projectClipped(
        _ linear: Linear, _ x: Tensor, clip: Gemma4VisionClip?,
        nTokens: Int, outDim: Int, on cmd: MTLCommandBuffer
    ) -> Tensor {
        guard let clip else {
            return projectRows(
                linear, x, nTokens: nTokens, outDim: outDim, on: cmd)
        }
        let inClamped = Ops.clampScalar(
            x, lo: clip.inputMin, hi: clip.inputMax, on: cmd)
        let y = projectRows(
            linear, inClamped, nTokens: nTokens, outDim: outDim, on: cmd)
        return Ops.clampScalar(
            y, lo: clip.outputMin, hi: clip.outputMax, on: cmd)
    }

    /// Apply a `Linear` to every row of `[nTokens, *]` via `Ops.gemm`,
    /// then broadcast-add the bias if present.
    private func projectRows(
        _ linear: Linear, _ x: Tensor, nTokens: Int,
        outDim: Int, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let y = Ops.gemm(weight: linear.weight, input: x, nRows: nTokens, on: cmd)
        guard let bias = linear.bias else { return y }
        return addRowBias(
            y, bias: bias, nRows: nTokens,
            rowSize: outDim, on: cmd)
    }
}

/// The Gemma 4 vision tower + multi-modal embedder. `encode` runs the
/// full forward and returns the pooled, projected soft tokens in the
/// text hidden dim.
final class Gemma4VLVisionModel: @unchecked Sendable {
    let cfg: Gemma4VLVisionConfig
    /// Patch-embed linear `[hidden, patchDimPadded]` (`3·patch·patch`
    /// columns, K-tile-padded).
    let patchEmbedWeight: Tensor
    let patchDimPadded: Int
    /// The two learned 2D position tables `[positionEmbeddingSize,
    /// hidden]` — index 0 is the x-axis, index 1 the y-axis.
    let posTableX: Tensor
    let posTableY: Tensor
    let blocks: [Gemma4VLVisionBlock]
    /// Optional standardization affine over the pooled tokens.
    let stdBias: Tensor?
    let stdScale: Tensor?
    /// Multi-modal embedder: GemmaRMSNorm (no-scale) + linear projection.
    /// Wrapped as `AnyLinear` so quantized checkpoints (4-bit e2b-it) work
    /// transparently — the weight is uint32-packed in those conversions.
    let embedderProjection: AnyLinear
    /// The no-scale norm's epsilon (the norm itself is unweighted).
    let embedderNormEps: Float
    let textHidden: Int
    let dtype: DType
    /// Patch grid side (square test image at the encoder's fixed size).
    let gridSide: Int
    /// Soft tokens one image contributes after pooling.
    let tokensPerImage: Int

    init(
        cfg: Gemma4VLVisionConfig, patchEmbedWeight: Tensor,
        patchDimPadded: Int, posTableX: Tensor, posTableY: Tensor,
        blocks: [Gemma4VLVisionBlock], stdBias: Tensor?, stdScale: Tensor?,
        embedderProjection: AnyLinear, embedderNormEps: Float,
        textHidden: Int, dtype: DType, gridSide: Int, tokensPerImage: Int
    ) {
        self.cfg = cfg
        self.patchEmbedWeight = patchEmbedWeight
        self.patchDimPadded = patchDimPadded
        self.posTableX = posTableX
        self.posTableY = posTableY
        self.blocks = blocks
        self.stdBias = stdBias
        self.stdScale = stdScale
        self.embedderProjection = embedderProjection
        self.embedderNormEps = embedderNormEps
        self.textHidden = textHidden
        self.dtype = dtype
        self.gridSide = gridSide
        self.tokensPerImage = tokensPerImage
    }

    static func load(
        visionConfig: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundle, dtype: DType,
        quantization: ModelConfig.QuantizationConfig?,
        device: Device
    ) throws -> Gemma4VLVisionModel {
        let cfg = try Gemma4VLVisionConfig.decode(visionConfig)
        // The vision tower weights are namespaced under `vision_tower.`
        // (after `model.` stripping); the multi-modal embedder under
        // `embed_vision.`. Probe both possible prefixes.
        let vt =
            weights.has("vision_tower.patch_embedder.input_proj.weight")
            ? weights.prefixed("vision_tower.")
            : weights.prefixed("model.vision_tower.")
        let ev =
            weights.has("embed_vision.embedding_projection.weight")
            ? weights.prefixed("embed_vision.")
            : weights.prefixed("model.embed_vision.")

        // ── Patch-embed ──
        // `input_proj.weight` is `[hidden, 3·patch·patch]`; pad the
        // `inDim` to the GEMM K-tile width.
        let rawPatch = try vt.tensor(named: "patch_embedder.input_proj.weight")
        let patchDim = 3 * cfg.patchSize * cfg.patchSize
        let patchDimPadded =
            ((patchDim + gemmKTileWidth - 1) / gemmKTileWidth) * gemmKTileWidth
        let patchEmbedWeight = padLinearColsTo(
            rawPatch, toCols: patchDimPadded, device: device)

        // ── Position embedding tables ──
        // `position_embedding_table` is `[2, positionEmbeddingSize,
        // hidden]`; split into the per-axis tables.
        let posTableRaw = try vt.tensor(
            named: "patch_embedder.position_embedding_table")
        let (posX, posY) = splitPositionTable(posTableRaw, device: device)

        // ── Block stack ──
        let paddedIntermediate =
            ((cfg.intermediate + gemmKTileWidth - 1) / gemmKTileWidth)
            * gemmKTileWidth
        var blocks: [Gemma4VLVisionBlock] = []
        blocks.reserveCapacity(cfg.depth)
        for i in 0 ..< cfg.depth {
            let p = "encoder.layers.\(i)"
            func lin(_ name: String) throws -> Linear {
                // Gemma 4 vision linears are wrapped in a clippable
                // module — the projection weight is `.linear.weight`.
                // The leading `try` covers the whole `??` — the RHS
                // `vt.tensor(...)` throws, so the operator expression
                // must be marked `try` at its start.
                let w =
                    try (try? vt.tensor(named: "\(p).\(name).linear.weight"))
                    ?? vt.tensor(named: "\(p).\(name).weight")
                let b =
                    (try? vt.tensor(named: "\(p).\(name).linear.bias"))
                    ?? (try? vt.tensor(named: "\(p).\(name).bias"))
                return Linear(weight: w, bias: b)
            }
            // The four per-block norms apply the weight DIRECTLY — the
            // Gemma 4 VISION tower's RMSNorm is `normed * weight`, NOT the
            // text tower's `(1 + weight)` fold (reference `vision.py`
            // RMSNorm: "weight applied directly"; `VisionRMSNorm`:
            // `normed * weight`). Folding +1 here over-scaled every block
            // and drove the per-block drift vs the mlx-vlm oracle.
            func gemmaNorm(_ name: String) throws -> RMSNorm {
                let raw = try vt.tensor(named: "\(p).\(name).weight")
                return RMSNorm(weight: raw, eps: cfg.rmsNormEps)
            }
            // The q / k head norms are plain (weighted, no +1) RMSNorms.
            func headNorm(_ name: String) throws -> RMSNorm {
                RMSNorm(
                    weight: try vt.tensor(named: "\(p).\(name).weight"),
                    eps: cfg.rmsNormEps)
            }
            // Gemma 4 `Gemma4ClippableLinear` scalar clamp bounds. Presence
            // of `*.input_min` signals a clipped linear; absent → nil (plain
            // projection). All four bounds are 1-element tensors.
            func clipFor(_ name: String) -> Gemma4VisionClip? {
                guard
                    let inMin = try? vt.tensor(named: "\(p).\(name).input_min"),
                    let inMax = try? vt.tensor(named: "\(p).\(name).input_max"),
                    let outMin = try? vt.tensor(named: "\(p).\(name).output_min"),
                    let outMax = try? vt.tensor(named: "\(p).\(name).output_max")
                else { return nil }
                return Gemma4VisionClip(
                    inputMin: inMin.toFloatArray()[0],
                    inputMax: inMax.toFloatArray()[0],
                    outputMin: outMin.toFloatArray()[0],
                    outputMax: outMax.toFloatArray()[0])
            }
            let clips = Gemma4VisionBlockClips(
                q: clipFor("self_attn.q_proj"),
                k: clipFor("self_attn.k_proj"),
                v: clipFor("self_attn.v_proj"),
                o: clipFor("self_attn.o_proj"),
                gate: clipFor("mlp.gate_proj"),
                up: clipFor("mlp.up_proj"),
                down: clipFor("mlp.down_proj"))
            let gate = try lin("mlp.gate_proj")
            let up = try lin("mlp.up_proj")
            let down = try lin("mlp.down_proj")
            blocks.append(
                Gemma4VLVisionBlock(
                    inputNorm: try gemmaNorm("input_layernorm"),
                    postAttnNorm: try gemmaNorm("post_attention_layernorm"),
                    preFFNorm: try gemmaNorm("pre_feedforward_layernorm"),
                    postFFNorm: try gemmaNorm("post_feedforward_layernorm"),
                    qProj: try lin("self_attn.q_proj"),
                    kProj: try lin("self_attn.k_proj"),
                    vProj: try lin("self_attn.v_proj"),
                    oProj: try lin("self_attn.o_proj"),
                    qNorm: try headNorm("self_attn.q_norm"),
                    kNorm: try headNorm("self_attn.k_norm"),
                    gate: padLinearRows(gate, toRows: paddedIntermediate, device: device),
                    up: padLinearRows(up, toRows: paddedIntermediate, device: device),
                    down: Linear(
                        weight: padLinearColsTo(
                            down.weight, toCols: paddedIntermediate,
                            device: device),
                        bias: down.bias),
                    clips: clips,
                    paddedIntermediate: paddedIntermediate, cfg: cfg))
        }

        // ── Standardization affine (optional) ──
        let stdBias = try? vt.tensor(named: "std_bias")
        let stdScale = try? vt.tensor(named: "std_scale")

        // ── Multi-modal embedder ──
        // `embedding_projection` may be quantized in 4-bit checkpoints (e.g.
        // `gemma-4-e2b-it-4bit` — weight dtype is U32). Load through
        // `loadLinear` so a `QuantizedLinear` is returned for those cases
        // instead of a plain `Linear` with packed-uint32 data, which would
        // trip `Ops.gemm`'s element-count precondition (packed column count
        // 8× less than the actual input width).
        // `embedding_pre_projection_norm` is a GemmaRMSNorm-no-scale —
        // unweighted; only the eps matters.
        let embedderProjection = try loadLinear(
            base: "embedding_projection", in: ev, quantization: quantization)

        // Size the test grid so the patch count pools cleanly into the
        // soft-token grid. `defaultOutputLength` is the soft-token count;
        // pick a grid whose side, divided by the pooling kernel, gives a
        // near-square soft-token grid.
        let softSide = Int(Double(cfg.defaultOutputLength).squareRoot())
        let tokensPerImage = softSide * softSide
        let gridSide = softSide * cfg.poolingKernelSize

        return Gemma4VLVisionModel(
            cfg: cfg, patchEmbedWeight: patchEmbedWeight,
            patchDimPadded: patchDimPadded, posTableX: posX, posTableY: posY,
            blocks: blocks, stdBias: stdBias, stdScale: stdScale,
            embedderProjection: embedderProjection,
            embedderNormEps: cfg.rmsNormEps, textHidden: textHidden,
            dtype: dtype, gridSide: gridSide, tokensPerImage: tokensPerImage)
    }

    /// Run the full vision forward on a preprocessed image. `image` is a
    /// normalized NCHW tensor `[1, 3, side, side]` where `side =
    /// gridSide · patchSize`. Returns `[tokensPerImage, textHidden]`.
    func encode(image: Tensor, device: Device) -> Tensor {
        // The Gemma 4 image processor is variable-resolution (aspect-ratio
        // preserving): `image` is NCHW `[1, 3, H, W]` (or `[3, H, W]`) with
        // H, W each a multiple of `patchSize · poolingKernelSize`. Derive
        // the (possibly non-square) patch grid from the image dimensions
        // rather than assuming the fixed square `gridSide`.
        let shp = image.shape
        let imgH = shp[shp.count - 2]
        let imgW = shp[shp.count - 1]
        let p = cfg.patchSize
        let gridH = imgH / p
        let gridW = imgW / p
        let nPatches = gridH * gridW

        // ── Patch unfold + embed ── (unfold → GEMM → posEmb, one cmd buffer)
        let cmd = device.makeCommandBuffer()
        let unfolded = unfoldPatches(image: image, gridH: gridH, gridW: gridW, on: cmd)
        var h = Ops.gemm(
            weight: patchEmbedWeight, input: unfolded,
            nRows: nPatches, on: cmd)
        // Fuse the learned 2D position-embedding build + add into one GPU
        // dispatch: out = h + posX[col] + posY[row].
        let (posX, posY) = resolvedPositionTables(gridH: gridH, gridW: gridW)
        h = Ops.posEmb2dAdd(
            tokens: h, posX: posX, posY: posY,
            hidden: cfg.hidden, gridW: gridW, gridH: gridH, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── Per-patch grid coordinates (row-major over gridH × gridW) ──
        var xPos = [Int](repeating: 0, count: nPatches)
        var yPos = [Int](repeating: 0, count: nPatches)
        for y in 0 ..< gridH {
            for x in 0 ..< gridW {
                let idx = y * gridW + x
                xPos[idx] = x
                yPos[idx] = y
            }
        }

        // ── Block stack ──
        for block in blocks {
            h = block.forward(
                h, nTokens: nPatches, xPos: xPos, yPos: yPos,
                device: device)
        }

        // ── Attention pooling → soft tokens, then multi-modal embed ──
        let pooled = pool(h, gridH: gridH, gridW: gridW, device: device)
        return embed(pooled, device: device)
    }

    /// Aspect-ratio-preserving target `(height, width)` for the
    /// variable-resolution image processor (instance form — pulls the
    /// geometry from `cfg`).
    func aspectPreservingSize(width: Int, height: Int) -> (h: Int, w: Int) {
        Gemma4VLVisionModel.aspectPreservingSize(
            width: width, height: height,
            patchSize: cfg.patchSize, poolingKernelSize: cfg.poolingKernelSize,
            maxSoftTokens: cfg.defaultOutputLength)
    }

    /// Aspect-ratio-preserving target `(height, width)`: the largest dims
    /// that (1) produce at most `maxPatches = maxSoftTokens · pool²` patches
    /// at `patchSize`, and (2) are divisible by `pool · patchSize`. Mirrors
    /// the reference `get_aspect_ratio_preserving_size`
    /// (`image_processing_pil_gemma4`). Pure geometry — `static` so it is
    /// unit-testable across aspect ratios without a loaded tower.
    static func aspectPreservingSize(
        width: Int, height: Int,
        patchSize: Int, poolingKernelSize: Int, maxSoftTokens: Int
    ) -> (h: Int, w: Int) {
        let pool = poolingKernelSize
        let maxPatches = maxSoftTokens * pool * pool
        let sideMult = pool * patchSize
        let targetPx = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPx / Double(width * height)).squareRoot()
        var th =
            Int((factor * Double(height) / Double(sideMult)).rounded(.down))
            * sideMult
        var tw =
            Int((factor * Double(width) / Double(sideMult)).rounded(.down))
            * sideMult
        // Degenerate-aspect clamps (reference): a dim that rounds to 0 is
        // pinned to one `sideMult` block, the other capped at `maxSide`.
        let maxSide = (maxPatches / (pool * pool)) * sideMult
        if th == 0 {
            th = sideMult
            tw = min(
                Int((Double(width) / Double(height)).rounded(.down)) * sideMult,
                maxSide)
        } else if tw == 0 {
            tw = sideMult
            th = min(
                Int((Double(height) / Double(width)).rounded(.down)) * sideMult,
                maxSide)
        }
        return (h: th, w: tw)
    }

    /// Present the tower as a `VisionEncoder` so `VisionModel` accepts it.
    func asVisionEncoder() -> VisionEncoder {
        Gemma4VLComposedEncoder(tower: self)
    }

    // ── Patch unfold ──

    /// Unfold a normalized NCHW image into a `[nPatches, patchDimPadded]`
    /// tensor. Each row holds one `3 · patch · patch` patch after the
    /// reference's `2·(x − 0.5)` re-centring; row order is raster
    /// `(patchRow, patchCol)`.
    private func unfoldPatches(
        image: Tensor, gridH: Int, gridW: Int, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let p = cfg.patchSize
        let imgH = gridH * p
        let imgW = gridW * p
        // GPU patch unfold: channel-interleaved `(py, px, c)` columns, the
        // reference `2·(x − 0.5)` re-centre, zero-padded to `patchDimPadded`.
        // The kernel's output dtype follows its input and feeds the patch-
        // embed GEMM (weight `dtype`), so present a `dtype` image — a cheap
        // cold-path cast if the preprocessed pixels arrived in another
        // precision.
        let img: Tensor
        if image.dtype == dtype {
            img = image
        } else {
            img = MediaUtils.makeTensor(
                from: image.toFloatArray(), shape: [3, imgH, imgW],
                dtype: dtype, device: .shared)
        }
        return Ops.im2colPatchInterleaved(
            img, channels: 3, inH: imgH, inW: imgW, patch: p,
            patchDimPadded: patchDimPadded, scale: 2.0, bias: -1.0, on: cmd)
    }

    // ── Position embedding ──

    /// Resolve the learned 2D position tables to the current patch grid:
    /// `posX[col] = tableX[min(posSize-1, col)]` (`col ∈ [0, gridW)`) and
    /// `posY[row] = tableY[min(posSize-1, row)]` (`row ∈ [0, gridH)`).
    /// Returns the two f32 tables `Ops.posEmb2dAdd` consumes — it fuses the
    /// per-patch `tokens + posX[col] + posY[row]` build-and-add into one GPU
    /// dispatch. The `min(posSize-1, ·)` clamp mirrors the reference's
    /// bounded table lookup for grids larger than the learned table.
    private func resolvedPositionTables(
        gridH: Int, gridW: Int
    ) -> (posX: Tensor, posY: Tensor) {
        let hidden = cfg.hidden
        let posSize = cfg.positionEmbeddingSize
        let tableX = posTableX.toFloatArray()
        let tableY = posTableY.toFloatArray()
        var xFlat = [Float](repeating: 0, count: gridW * hidden)
        for col in 0 ..< gridW {
            let xi = min(posSize - 1, col)
            for d in 0 ..< hidden {
                xFlat[col * hidden + d] = tableX[xi * hidden + d]
            }
        }
        var yFlat = [Float](repeating: 0, count: gridH * hidden)
        for row in 0 ..< gridH {
            let yi = min(posSize - 1, row)
            for d in 0 ..< hidden {
                yFlat[row * hidden + d] = tableY[yi * hidden + d]
            }
        }
        // `Ops.posEmb2dAdd` requires the tables in f32.
        let posX = MediaUtils.makeTensor(
            from: xFlat, shape: [gridW, hidden], dtype: .f32, device: .shared)
        let posY = MediaUtils.makeTensor(
            from: yFlat, shape: [gridH, hidden], dtype: .f32, device: .shared)
        return (posX, posY)
    }

    // ── Attention pooling ──

    /// Pool the `gridSide × gridSide` patch tokens down to
    /// `tokensPerImage` soft tokens by averaging each
    /// `poolingKernelSize × poolingKernelSize` neighbourhood, then scale
    /// by `sqrt(hidden)` and apply the optional standardization affine.
    /// Returns `[tokensPerImage, hidden]`.
    private func pool(_ h: Tensor, gridH: Int, gridW: Int, device: Device) -> Tensor {
        let k = cfg.poolingKernelSize
        let softH = gridH / k
        let softW = gridW / k
        let hidden = cfg.hidden
        // ── k×k average pool over the patch grid on the GPU. `h` is the
        // `[gridH·gridW, hidden]` token map = NHWC `[1, gridH, gridW, hidden]`;
        // a stride-k, pad-0 avg pool collapses each k×k neighbourhood to one
        // soft token, in the same `(sy, sx)` raster the CPU path produced.
        let cmd = device.makeCommandBuffer()
        let pooledT = Ops.avgPool2dNHWC(
            h, batch: 1, inH: gridH, inW: gridW, channels: hidden,
            kH: k, kW: k, stride: k, pad: 0, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        // ── Trailing √hidden scale + optional standardization affine. Both
        // are per-channel and trivially cheap at soft-token resolution, so
        // they stay on the CPU (no per-channel-affine GPU op exists yet).
        var pooled = pooledT.toFloatArray()
        let rootHidden = Float(Double(hidden).squareRoot())
        let stdB = stdBias?.toFloatArray()
        let stdS = stdScale?.toFloatArray()
        for t in 0 ..< (softH * softW) {
            let base = t * hidden
            for c in 0 ..< hidden {
                var val = pooled[base + c] * rootHidden
                if let stdB, let stdS {
                    val = (val - stdB[c]) * stdS[c]
                }
                pooled[base + c] = val
            }
        }
        let out = Tensor.empty(
            shape: [softH * softW, hidden],
            dtype: dtype, device: device)
        MediaUtils.copyFloats(pooled, into: out)
        return out
    }

    // ── Multi-modal embedder ──

    /// GemmaRMSNorm (no-scale) each pooled soft token, then project into
    /// the text hidden dim. Returns `[tokensPerImage, textHidden]`.
    private func embed(_ pooled: Tensor, device: Device) -> Tensor {
        let nTokens = pooled.shape[0]
        let hidden = cfg.hidden
        let src = pooled.toFloatArray()
        // Unweighted RMSNorm — the embedder's pre-projection norm has no
        // learned scale.
        var normed = [Float](repeating: 0, count: nTokens * hidden)
        for t in 0 ..< nTokens {
            var ss: Float = 0
            for d in 0 ..< hidden { ss += src[t * hidden + d] * src[t * hidden + d] }
            let inv = 1.0 / (ss / Float(hidden) + embedderNormEps).squareRoot()
            for d in 0 ..< hidden { normed[t * hidden + d] = src[t * hidden + d] * inv }
        }
        let normedT = Tensor.empty(
            shape: [nTokens, hidden], dtype: dtype,
            device: device)
        MediaUtils.copyFloats(normed, into: normedT)

        let cmd = device.makeCommandBuffer()
        // Use `callMany` so quantized weights (QuantizedLinear from a 4-bit
        // checkpoint) are handled correctly via the dequant-gemm path.
        let projected = embedderProjection.callMany(
            normedT, t: nTokens, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        return projected
    }

    // ── Static helpers ──

    /// Split a `[2, posSize, hidden]` position table into the per-axis
    /// `[posSize, hidden]` tables.
    static func splitPositionTable(_ raw: Tensor, device: Device)
        -> (x: Tensor, y: Tensor)
    {
        precondition(
            raw.shape.count == 3 && raw.shape[0] == 2,
            "Gemma4VL: position table must be [2, posSize, hidden], "
                + "got \(raw.shape)")
        let posSize = raw.shape[1]
        let hidden = raw.shape[2]
        let src = raw.toFloatArray()
        var x = [Float](repeating: 0, count: posSize * hidden)
        var y = [Float](repeating: 0, count: posSize * hidden)
        let plane = posSize * hidden
        for i in 0 ..< plane {
            x[i] = src[i]
            y[i] = src[plane + i]
        }
        let xT = MediaUtils.makeTensor(
            from: x, shape: [posSize, hidden], dtype: raw.dtype, device: device)
        let yT = MediaUtils.makeTensor(
            from: y, shape: [posSize, hidden], dtype: raw.dtype, device: device)
        return (xT, yT)
    }

    // Shared `addRowBias`, `padLinearRows`, and `padLinearColsTo`
    // helpers live in `VisionTowerOps.swift`.
}

/// A `VisionEncoder` subclass whose `encode` runs the Gemma 4 vision
/// tower + multi-modal embedder — so `VisionModel` (which holds a
/// `VisionEncoder`) transparently gets the pooled, projected soft
/// tokens.
final class Gemma4VLComposedEncoder: VisionEncoder {
    let tower: Gemma4VLVisionModel

    init(tower: Gemma4VLVisionModel) {
        self.tower = tower
        let c = tower.cfg
        let side = tower.gridSide * c.patchSize
        let facadeConfig = VisionEncoderConfig(
            inChannels: 3, imageSize: side,
            patchSize: side / Int(Double(tower.tokensPerImage).squareRoot()),
            hidden: c.hidden, intermediate: c.intermediate,
            nLayers: c.depth, nHeads: c.numHeads,
            layerNormEps: c.rmsNormEps, textHidden: tower.textHidden)
        let placeholderW = tower.patchEmbedWeight
        let placeholderNorm = LayerNorm(
            weight: placeholderW, bias: placeholderW, eps: c.rmsNormEps)
        super.init(
            config: facadeConfig,
            patchEmbedWeight: placeholderW, patchEmbedBias: placeholderW,
            positionEmbedding: placeholderW, layers: [],
            postLayerNorm: placeholderNorm,
            projection: nil, dtype: tower.dtype)
    }

    /// Gemma 4's variable-resolution image processor: aspect-ratio-
    /// preserving resize to dims divisible by `patchSize · poolingKernel`,
    /// then rescale to `[0, 1]` ONLY (no mean/std — the tower's patch
    /// unfold re-centres via `2·(v − 0.5)`; `RGBImage` pixels are already
    /// `[0, 1]`). The resulting non-square `[1, 3, H, W]` tensor drives the
    /// tower's variable soft-token count.
    override func preprocessImage(
        _ image: RGBImage, device: Device = .shared
    ) -> Tensor? {
        let (th, tw) = tower.aspectPreservingSize(
            width: image.width, height: image.height)
        let identity = ImageNormalization(mean: (0, 0, 0), std: (1, 1, 1))
        return MediaUtils.preprocess(
            image, targetW: tw, targetH: th, normalization: identity,
            dtype: tower.dtype, device: device)
    }

    /// Run the Gemma 4 vision tower. Returns
    /// `[tokensPerImage, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        tower.encode(image: image, device: device)
    }

    /// Video path: run every frame through the same Gemma 4 vision tower
    /// and concatenate the per-frame soft tokens in frame order. Gemma 4 has
    /// no bespoke temporal-patch fold — a video is a temporally-ordered
    /// sequence of images, each contributing its (variable-resolution) soft
    /// tokens, spliced at the prompt's `video_token_id` placeholders.
    ///
    /// Each frame's soft-token count is VARIABLE (the aspect-ratio-
    /// preserving processor), so the total is summed from the actual
    /// per-frame encodes rather than assuming a fixed `tokensPerImage`. The
    /// prompt must carry exactly `Σ per-frame counts` placeholders (the
    /// caller learns this by encoding first).
    override func encode(
        frames: [Tensor], device: Device = .shared
    ) throws -> Tensor {
        precondition(
            !frames.isEmpty,
            "Gemma4VLComposedEncoder.encode(frames:): expected ≥ 1 frame")
        let textHidden = tower.textHidden
        var flat = [Float]()
        var totalTokens = 0
        for frame in frames {
            let soft = tower.encode(image: frame, device: device)
            totalTokens += soft.shape[0]
            flat.append(contentsOf: soft.toFloatArray())
        }
        return MediaUtils.makeTensor(
            from: flat,
            shape: [totalTokens, textHidden],
            dtype: tower.dtype, device: device)
    }
}

// ─── Gemma 4 VL family orchestrator ──────────────────────────────────
//
// Composes the Gemma 4 vision tower above with the Gemma 4 text
// backbone (`Gemma4Dense` / `Gemma4E` / `Gemma4MoE` in
// `Models/Text/Gemma4Text.swift`) for the
// `Gemma4ForConditionalGeneration` checkpoints.
//
// Composition:
//   • Gemma 4 vision tower — a bespoke ViT that differs substantially
//     from the shared SigLIP `VisionEncoder`:
//       – patch-embed via a flattened linear projection (each
//         `3 · patch · patch` patch row is projected by one GEMM, after
//         a `2·(x − 0.5)` re-centring),
//       – a learned 2D position embedding: two `[positionEmbeddingSize,
//         hidden]` tables (one per spatial axis) summed by the patch's
//         `(x, y)` grid coordinate,
//       – RoPE attention with multi-dimensional positions (the head dim
//         is split into an x-rotary half and a y-rotary half) and
//         per-projection q / k / v RMSNorms,
//       – four GemmaRMSNorm "zero-shift" norms per block (input /
//         post-attention / pre-feedforward / post-feedforward) and a
//         GELU-gated SwiGLU MLP,
//       – an attention-pooling head that pools the patch grid down to
//         `default_output_length` soft tokens, then a `sqrt(hidden)`
//         scale and an optional standardization affine.
//   • Multi-modal embedder — a GemmaRMSNorm (no-scale) over the pooled
//     vision tokens followed by a linear projection into the text
//     hidden dim (`embed_vision.*`).
//   • Gemma 4 text backbone — the existing `Gemma4Model` engine, loaded
//     from the `language_model.`-prefixed sub-tree (the `Gemma4Loader`
//     probes the prefix and reads `text_config` itself).
//
// The three are joined by `VisionModel`'s cross-modal token splice: each
// `<image_soft_token>` placeholder (`image_token_id`) in the prompt
// takes one of the pooled, projected vision tokens.

public enum Gemma4VL {
    /// `image_token_id` default for Gemma 4 VL checkpoints.
    public static let defaultImageTokenId = 262_144
    /// `audio_token_id` default — the `<audio_soft_token>` placeholder id
    /// (mlx-vlm `config.audio_token_id`). Only consulted when the
    /// checkpoint carries an audio tower.
    public static let defaultAudioTokenId = 258_881
    /// `video_token_id` default — the `<video_soft_token>` placeholder id.
    /// Gemma 4 has no bespoke temporal fold: a video is a sequence of
    /// frames, each run through the vision tower, spliced at these slots.
    public static let defaultVideoTokenId = 258_884

    /// Build a `VisionModel` from a `Gemma4ForConditionalGeneration`
    /// checkpoint: the Gemma 4 vision tower + multi-modal embedder +
    /// Gemma 4 text backbone, joined by the cross-modal splice. When the
    /// checkpoint also carries an `audio_config` + `audio_tower.*`, the
    /// audio tower + `embed_audio` projection are loaded and the audio
    /// path is wired; otherwise the audio members are left `nil` so
    /// text-only / vision-only Gemma 4 checkpoints load unchanged.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Gemma4Error.missingConfig("vision_config")
        }

        // ── Text backbone — Gemma 4 engine ──
        // `Gemma4Loader` probes the weight prefix (`language_model.model.`
        // for a VLM conversion) and reads the text hyper-parameters from
        // the `text_config` sub-tree itself, so the whole `config` and
        // `weights` bundle is forwarded unchanged; the variant (dense /
        // E / MoE) is config-driven.
        let textVariant = try Gemma4.variant(for: config)
        let textEngine = try textVariant.loadModel(
            config: config, weights: weights, options: options, device: device)

        // ── Vision tower + multi-modal embedder ──
        let vision = try Gemma4VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: weights, dtype: textEngine.dtype,
            quantization: config.quantization, device: device)

        let imageTokenId =
            config.int("image_token_id")
            ?? config.int("image_token_index") ?? defaultImageTokenId

        // ── Audio tower + multi-modal embedder (optional) ──
        // Only load when the checkpoint declares an `audio_config` AND
        // ships `audio_tower.*` tensors. Text-only / vision-only Gemma 4
        // checkpoints have neither, so the audio members stay `nil` and
        // those checkpoints load exactly as before.
        let hasAudioTower =
            weights.has("audio_tower.output_proj.weight")
            || weights.has("model.audio_tower.output_proj.weight")
        var audioTokenId: Int? = nil
        var audioEncode:
            (@Sendable (_ waveform: [Float], _ device: Device) -> Tensor)? = nil
        if let audioConfigRaw = config.subConfig("audio_config"), hasAudioTower {
            let audioConfig = try Gemma4AudioConfig.decode(audioConfigRaw)
            let audioEncoder = try Gemma4AudioEncoder.load(
                weights: weights, config: audioConfig,
                quantization: config.quantization, device: device)
            // `embed_audio` projects the encoder's `[nFrames, output_proj_dims]`
            // soft tokens (1536 on E2B) into the text hidden dim. The
            // RMSNormNoScale eps matches the audio config.
            let embedAudio = try Gemma4AudioEmbedder.load(
                weights: weights, embeddingDim: audioConfig.outputProjDims,
                normEps: audioConfig.rmsNormEps, dtype: textEngine.dtype,
                quantization: config.quantization)
            audioTokenId =
                config.int("audio_token_id")
                ?? config.int("audio_token_index") ?? defaultAudioTokenId
            // The audio encode hook: waveform → log-Mel front-end → audio
            // encoder → multi-modal embedder. Mirrors the reference
            // `audio_tower(...)` then `embed_audio(...)`.
            audioEncode = { waveform, dev in
                let (mel, mask) = Gemma4AudioFrontend.melSpectrogram(
                    waveform: waveform, device: dev)
                let (encodings, _) = audioEncoder.encode(
                    mel: mel, melMask: mask, device: dev)
                return embedAudio.embed(encodings, device: dev)
            }
        }

        // ── Video placeholder id ──
        // Wire a `videoTokenId` only when the checkpoint declares one. The
        // composed encoder's `encode(frames:)` reuses the vision tower, so
        // video works as soon as the prompt carries the placeholders.
        let videoTokenId =
            config.int("video_token_id") ?? config.int("video_token_index")

        return try VisionModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .siglip,
            imageTokenCount: vision.tokensPerImage,
            audioTokenId: audioTokenId,
            audioEncode: audioEncode)
    }
}

// ─── Gemma 4 Omni (encoder-free) ─────────────────────────────────────
//
// Google's Gemma 4 12B "Unified"/Omni (`Gemma4UnifiedForConditionalGeneration`,
// model_type `gemma4_unified`) drops the bespoke ViT + Conformer encoders of
// the Gemma 4 VL family above. Instead it projects raw inputs straight into
// the text embedding space:
//
//   • Vision (`model.vision_embedder.*` + `model.embed_vision.*`):
//       image → 48×48 raw-pixel patches (flattened to 3·48·48 = 6912)
//             → patch_ln1 (LayerNorm 6912)
//             → patch_dense (Linear 6912→3840, +bias)
//             → patch_ln2 (LayerNorm 3840)
//             → + factorized 2D position embedding `[1120, 2, 3840]`
//             → pos_norm (LayerNorm 3840)
//             → embed_vision (RMSNorm-no-scale + Linear 3840→3840)
//       One soft token per 48×48 patch; the aspect-preserving resize keeps
//       the patch count ≤ `num_soft_tokens` (default 280). There is NO ViT
//       block stack and NO separate avg-pool: the 3×3 pooling is folded into
//       the 48-px (= 16·3) patch unfold (resize divisibility = patch·pool).
//   • Audio (`model.embed_audio.*`): raw 16 kHz waveform → 640-sample/40 ms
//       frames → embed_audio (RMSNorm-no-scale + Linear 640→3840). See
//       `Gemma4OmniAudio.frame` in Gemma4Audio.swift.
//   • Video: a temporally-ordered sequence of frames, each run through the
//       vision path, spliced at the `<|video|>` placeholders.
//
// The text backbone is byte-for-byte a `Gemma4Dense` (the omni text_config
// resolves to it), loaded from the `model.language_model.`-prefixed sub-tree.
// The three are joined by `VisionModel`'s cross-modal splice on the
// `<|image|>` / `<|audio|>` / `<|video|>` placeholder ids.

/// Static shape of the Gemma 4 Omni encoder-free vision embedder, decoded
/// from the checkpoint's `vision_config`.
struct Gemma4OmniVisionConfig {
    /// Raw-pixel patch side (`model_patch_size`, 48) — `patch_dense`'s input
    /// is `3 · modelPatchSize²` (6912).
    let modelPatchSize: Int
    /// Processor sub-patch (`patch_size`, 16). Resize divisibility is
    /// `patchSize · poolingKernelSize` (48); the unfold uses `modelPatchSize`.
    let patchSize: Int
    /// Pooling kernel (`pooling_kernel_size`, 3) — folded into the 48-px unfold.
    let poolingKernelSize: Int
    /// Multi-modal embed dim (`mm_embed_dim`, 3840) — the patch projection
    /// output + the position-embedding width.
    let hidden: Int
    /// Soft-token budget (`num_soft_tokens`, default 280; budgets
    /// {70,140,280,560,1120}).
    let numSoftTokens: Int
    /// Position-embedding table length (`mm_posemb_size`, 1120).
    let posEmbSize: Int
    /// LayerNorm epsilon for patch_ln1 / patch_ln2 / pos_norm (Gemma
    /// convention, `rms_norm_eps`).
    let layerNormEps: Float

    static func decode(_ c: ModelConfig) throws -> Gemma4OmniVisionConfig {
        guard let hidden = c.int("mm_embed_dim"),
            let modelPatch = c.int("model_patch_size")
        else {
            throw Gemma4Error.missingConfig("vision_config")
        }
        return Gemma4OmniVisionConfig(
            modelPatchSize: modelPatch,
            patchSize: c.int("patch_size") ?? 16,
            poolingKernelSize: c.int("pooling_kernel_size") ?? 3,
            hidden: hidden,
            numSoftTokens: c.int("num_soft_tokens") ?? 280,
            posEmbSize: c.int("mm_posemb_size") ?? 1120,
            layerNormEps: Float(c.float("rms_norm_eps") ?? 1e-6))
    }
}

/// The Gemma 4 Omni encoder-free vision tower, presented as a
/// `VisionEncoder` subclass so `VisionModel` accepts it (it overrides
/// `preprocessImage` / `encode(image:)` / `encode(frames:)`; the base
/// ViT slots are unused placeholders). All weights stay bf16 (the convert
/// pipeline routes the multimodal projectors to `--vision-bits`, default
/// bf16). `@unchecked Sendable` via the `VisionEncoder` base — the held
/// weights are immutable `let`s set once at load.
final class Gemma4OmniVisionEncoder: VisionEncoder {
    let cfg: Gemma4OmniVisionConfig
    let textHidden: Int
    /// Activation dtype (the text engine's) — every intermediate tensor +
    /// the cast LayerNorm weights use this so `Ops.layerNorm`'s same-dtype
    /// precondition holds.
    let towerDtype: DType
    /// `patch_dense` — Linear `[3840, 6912]` (+ bias), kept bf16.
    let patchDense: AnyLinear
    /// LayerNorms (weight+bias cast to `towerDtype`).
    let patchLN1: LayerNorm  // over 6912
    let patchLN2: LayerNorm  // over 3840
    let posNorm: LayerNorm  // over 3840
    /// Per-axis position tables `[posEmbSize, hidden]`, split from the
    /// `[posEmbSize, 2, hidden]` `pos_embedding` parameter.
    let posTableX: Tensor
    let posTableY: Tensor
    /// `embed_vision` — RMSNorm-no-scale + Linear `[3840, 3840]`.
    let embedder: Gemma4MultimodalEmbedder

    init(
        cfg: Gemma4OmniVisionConfig, textHidden: Int, towerDtype: DType,
        patchDense: AnyLinear, patchLN1: LayerNorm, patchLN2: LayerNorm,
        posNorm: LayerNorm, posTableX: Tensor, posTableY: Tensor,
        embedder: Gemma4MultimodalEmbedder
    ) {
        self.cfg = cfg
        self.textHidden = textHidden
        self.towerDtype = towerDtype
        self.patchDense = patchDense
        self.patchLN1 = patchLN1
        self.patchLN2 = patchLN2
        self.posNorm = posNorm
        self.posTableX = posTableX
        self.posTableY = posTableY
        self.embedder = embedder
        // Facade config so the base `VisionEncoder` is validly constructed;
        // every consumed method is overridden, so the ViT slots below are
        // never read. `imageSize`/`patchSize` are nominal.
        let facade = VisionEncoderConfig(
            inChannels: 3, imageSize: cfg.modelPatchSize,
            patchSize: cfg.modelPatchSize,
            hidden: cfg.hidden, intermediate: cfg.hidden,
            nLayers: 0, nHeads: 1,
            layerNormEps: cfg.layerNormEps, textHidden: textHidden)
        let placeholder = Tensor.empty(shape: [1], dtype: towerDtype)
        let placeholderNorm = LayerNorm(
            weight: placeholder, bias: placeholder, eps: cfg.layerNormEps)
        super.init(
            config: facade,
            patchEmbedWeight: placeholder, patchEmbedBias: placeholder,
            positionEmbedding: placeholder, layers: [],
            postLayerNorm: placeholderNorm, projection: nil, dtype: towerDtype)
    }

    static func load(
        visionConfig: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundle, dtype: DType,
        quantization: ModelConfig.QuantizationConfig?, device: Device
    ) throws -> Gemma4OmniVisionEncoder {
        let cfg = try Gemma4OmniVisionConfig.decode(visionConfig)
        // The omni checkpoint nests the vision embedder under
        // `model.vision_embedder.` (the convert keeps it bf16). Probe the
        // bare prefix too for forward-compat with a re-hosted layout.
        let ve =
            weights.has("vision_embedder.patch_dense.weight")
            ? weights.prefixed("vision_embedder.")
            : weights.prefixed("model.vision_embedder.")
        let patchDense = try loadLinear(
            base: "patch_dense", in: ve, quantization: quantization)
        let patchLN1 = try loadLayerNormCast(
            base: "patch_ln1", in: ve, eps: cfg.layerNormEps,
            dtype: dtype, device: device)
        let patchLN2 = try loadLayerNormCast(
            base: "patch_ln2", in: ve, eps: cfg.layerNormEps,
            dtype: dtype, device: device)
        let posNorm = try loadLayerNormCast(
            base: "pos_norm", in: ve, eps: cfg.layerNormEps,
            dtype: dtype, device: device)
        // `pos_embedding` is `[posEmbSize, 2, hidden]` (axis-1 = x/y) — note
        // the axis order differs from the VL tower's `[2, posSize, hidden]`.
        let posRaw = try ve.tensor(named: "pos_embedding")
        let (posX, posY) = splitOmniPositionTable(
            posRaw, dtype: dtype, device: device)
        // `embed_vision` — the shared multimodal embedder, embeddingDim =
        // mm_embed_dim (3840).
        let embedder = try Gemma4MultimodalEmbedder.load(
            weights: weights, embeddingDim: cfg.hidden,
            normEps: cfg.layerNormEps, dtype: dtype,
            quantization: quantization, prefix: "embed_vision")
        return Gemma4OmniVisionEncoder(
            cfg: cfg, textHidden: textHidden, towerDtype: dtype,
            patchDense: patchDense, patchLN1: patchLN1, patchLN2: patchLN2,
            posNorm: posNorm, posTableX: posX, posTableY: posY,
            embedder: embedder)
    }

    /// Aspect-ratio-preserving resize to dims divisible by
    /// `patchSize · poolingKernelSize` (48), then rescale to `[0, 1]` ONLY
    /// (the omni image processor is `do_normalize=false`; `RGBImage` pixels
    /// are already `[0, 1]` and `patch_ln1` is affine-invariant to centring).
    override func preprocessImage(
        _ image: RGBImage, device: Device = .shared
    ) -> Tensor? {
        let (th, tw) = Gemma4VLVisionModel.aspectPreservingSize(
            width: image.width, height: image.height,
            patchSize: cfg.patchSize, poolingKernelSize: cfg.poolingKernelSize,
            maxSoftTokens: cfg.numSoftTokens)
        let identity = ImageNormalization(mean: (0, 0, 0), std: (1, 1, 1))
        return MediaUtils.preprocess(
            image, targetW: tw, targetH: th, normalization: identity,
            dtype: towerDtype, device: device)
    }

    /// Encode a preprocessed `[1, 3, H, W]` image (H, W multiples of 48)
    /// into `[nSoftTokens, textHidden]` soft tokens. One soft token per
    /// 48×48 patch.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        let shp = image.shape
        let imgH = shp[shp.count - 2]
        let imgW = shp[shp.count - 1]
        let p = cfg.modelPatchSize
        let gridH = imgH / p
        let gridW = imgW / p
        let nPatches = gridH * gridW
        let patchDim = 3 * p * p

        // 1. Unfold into [nPatches, 6912] (raster (row,col) order). Values
        //    stay [0,1]; patch_ln1 normalises per-row so centring is moot.
        let unfolded = unfoldPatches(
            image: image, gridH: gridH, gridW: gridW, patchDim: patchDim)

        // 2. patch_ln1 → 3. patch_dense → 4. patch_ln2 → 5. +pos → 6. pos_norm.
        let cmd = device.makeCommandBuffer()
        var h = Ops.layerNorm(
            unfolded, weight: patchLN1.weight, bias: patchLN1.bias,
            eps: patchLN1.eps, nRows: nPatches, rowSize: patchDim, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let cmd2 = device.makeCommandBuffer()
        h = patchDense.callMany(h, t: nPatches, on: cmd2, device: device)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        let cmd3 = device.makeCommandBuffer()
        h = Ops.layerNorm(
            h, weight: patchLN2.weight, bias: patchLN2.bias,
            eps: patchLN2.eps, nRows: nPatches, rowSize: cfg.hidden, on: cmd3)
        let posEmb = positionEmbedding(gridH: gridH, gridW: gridW)
        h = Ops.add(h, posEmb, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()

        let cmd4 = device.makeCommandBuffer()
        h = Ops.layerNorm(
            h, weight: posNorm.weight, bias: posNorm.bias,
            eps: posNorm.eps, nRows: nPatches, rowSize: cfg.hidden, on: cmd4)
        cmd4.commit()
        cmd4.waitUntilCompleted()

        // 7. embed_vision (RMSNorm-no-scale + Linear 3840→3840).
        return embedder.embed(h, device: device)
    }

    /// Video: run every frame through the vision path and concatenate the
    /// per-frame soft tokens in frame order (no temporal fold — same as the
    /// VL composed encoder).
    override func encode(
        frames: [Tensor], device: Device = .shared
    ) throws -> Tensor {
        precondition(
            !frames.isEmpty,
            "Gemma4OmniVisionEncoder.encode(frames:): expected ≥ 1 frame")
        var flat = [Float]()
        var totalTokens = 0
        for frame in frames {
            let soft = encode(image: frame, device: device)
            totalTokens += soft.shape[0]
            flat.append(contentsOf: soft.toFloatArray())
        }
        return MediaUtils.makeTensor(
            from: flat, shape: [totalTokens, textHidden],
            dtype: towerDtype, device: device)
    }

    // ── Helpers ──

    /// Unfold a normalized NCHW image into `[nPatches, patchDim]` where each
    /// row is one `3 · modelPatchSize²` patch in (channel-last per pixel)
    /// raster `(patchRow, patchCol)` order. No re-centring (LayerNorm-invariant).
    private func unfoldPatches(
        image: Tensor, gridH: Int, gridW: Int, patchDim: Int
    ) -> Tensor {
        let p = cfg.modelPatchSize
        let pix = image.toFloatArray()  // [3, H, W]
        let imgH = gridH * p
        let imgW = gridW * p
        let nPatches = gridH * gridW
        var rows = [Float](repeating: 0, count: nPatches * patchDim)
        for pr in 0 ..< gridH {
            for pc in 0 ..< gridW {
                let patch = pr * gridW + pc
                var col = 0
                for py in 0 ..< p {
                    let yy = pr * p + py
                    for px in 0 ..< p {
                        let xx = pc * p + px
                        for ch in 0 ..< 3 {
                            rows[patch * patchDim + col] =
                                pix[(ch * imgH + yy) * imgW + xx]
                            col += 1
                        }
                    }
                }
            }
        }
        return MediaUtils.makeTensor(
            from: rows, shape: [nPatches, patchDim],
            dtype: towerDtype, device: .shared)
    }

    /// Build the per-patch 2D position embedding `[nPatches, hidden]`:
    /// `tableX[col] + tableY[row]`, clamped to the table length.
    private func positionEmbedding(gridH: Int, gridW: Int) -> Tensor {
        let hidden = cfg.hidden
        let tableX = posTableX.toFloatArray()
        let tableY = posTableY.toFloatArray()
        let posSize = cfg.posEmbSize
        let nPatches = gridH * gridW
        var dst = [Float](repeating: 0, count: nPatches * hidden)
        for pr in 0 ..< gridH {
            for pc in 0 ..< gridW {
                let patch = pr * gridW + pc
                let xi = min(posSize - 1, pc)
                let yi = min(posSize - 1, pr)
                for d in 0 ..< hidden {
                    dst[patch * hidden + d] =
                        tableX[xi * hidden + d] + tableY[yi * hidden + d]
                }
            }
        }
        return MediaUtils.makeTensor(
            from: dst, shape: [nPatches, hidden], dtype: towerDtype,
            device: .shared)
    }

    /// Split a `[posSize, 2, hidden]` position table into the per-axis
    /// `[posSize, hidden]` tables (axis-1 index 0 = x, 1 = y). Cast to `dtype`.
    static func splitOmniPositionTable(
        _ raw: Tensor, dtype: DType, device: Device
    ) -> (x: Tensor, y: Tensor) {
        precondition(
            raw.shape.count == 3 && raw.shape[1] == 2,
            "Gemma4Omni: pos_embedding must be [posSize, 2, hidden], "
                + "got \(raw.shape)")
        let posSize = raw.shape[0]
        let hidden = raw.shape[2]
        let src = raw.toFloatArray()  // [pos][axis][d]
        var x = [Float](repeating: 0, count: posSize * hidden)
        var y = [Float](repeating: 0, count: posSize * hidden)
        for pos in 0 ..< posSize {
            for d in 0 ..< hidden {
                x[pos * hidden + d] = src[(pos * 2 + 0) * hidden + d]
                y[pos * hidden + d] = src[(pos * 2 + 1) * hidden + d]
            }
        }
        return (
            MediaUtils.makeTensor(
                from: x, shape: [posSize, hidden], dtype: dtype, device: device),
            MediaUtils.makeTensor(
                from: y, shape: [posSize, hidden], dtype: dtype, device: device)
        )
    }
}

/// Load a `<base>.{weight,bias}` LayerNorm, casting both to `dtype` so
/// `Ops.layerNorm`'s same-dtype precondition holds against `dtype`
/// activations (the omni norms ship bf16; the cast is a no-op when the
/// activation dtype already matches).
private func loadLayerNormCast(
    base: String, in bundle: SafeTensorsBundle, eps: Float,
    dtype: DType, device: Device
) throws -> LayerNorm {
    let w = try bundle.tensor(named: "\(base).weight")
    let b = try bundle.tensor(named: "\(base).bias")
    let wCast = MediaUtils.makeTensor(
        from: w.toFloatArray(), shape: w.shape, dtype: dtype, device: device)
    let bCast = MediaUtils.makeTensor(
        from: b.toFloatArray(), shape: b.shape, dtype: dtype, device: device)
    return LayerNorm(weight: wCast, bias: bCast, eps: eps)
}

/// Gemma 4 Omni family orchestrator — builds a `VisionModel` from a
/// `Gemma4UnifiedForConditionalGeneration` checkpoint: the `Gemma4Dense`
/// text backbone + the encoder-free vision tower + the raw-frame audio
/// embedder, joined by the cross-modal splice.
public enum Gemma4Omni {
    /// `<|image|>` placeholder id (config `image_token_id`).
    public static let defaultImageTokenId = 258_880
    /// `<|audio|>` placeholder id (config `audio_token_id`).
    public static let defaultAudioTokenId = 258_881
    /// `<|video|>` placeholder id (config `video_token_id`).
    public static let defaultVideoTokenId = 258_884
    /// `audio_samples_per_token` / `audio_embed_dim` default (640 = 40 ms @ 16 kHz).
    public static let defaultAudioSamplesPerToken = 640

    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Gemma4Error.missingConfig("vision_config")
        }

        // ── Text backbone (Gemma4Dense; the omni text_config resolves to it) ──
        let textVariant = try Gemma4.variant(for: config)
        let textEngine = try textVariant.loadModel(
            config: config, weights: weights, options: options, device: device)
        let dtype = textEngine.dtype

        // ── Encoder-free vision tower ──
        let vision = try Gemma4OmniVisionEncoder.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: weights, dtype: dtype,
            quantization: config.quantization, device: device)

        let imageTokenId =
            config.int("image_token_id")
            ?? config.int("image_token_index") ?? defaultImageTokenId

        // ── Encoder-free audio (optional — gate on `embed_audio.*`) ──
        var audioTokenId: Int? = nil
        var audioEncode:
            (@Sendable (_ waveform: [Float], _ device: Device) -> Tensor)? = nil
        let hasAudio =
            weights.has("embed_audio.embedding_projection.weight")
            || weights.has("model.embed_audio.embedding_projection.weight")
        if hasAudio {
            let audioCfg = config.subConfig("audio_config")
            let samplesPerToken =
                audioCfg?.int("audio_samples_per_token")
                ?? audioCfg?.int("audio_embed_dim")
                ?? defaultAudioSamplesPerToken
            let audioEps = Float(audioCfg?.float("rms_norm_eps") ?? 1e-6)
            let embedAudio = try Gemma4MultimodalEmbedder.load(
                weights: weights, embeddingDim: samplesPerToken,
                normEps: audioEps, dtype: dtype,
                quantization: config.quantization, prefix: "embed_audio")
            audioTokenId =
                config.int("audio_token_id")
                ?? config.int("audio_token_index") ?? defaultAudioTokenId
            // Raw-frame audio: waveform → [nFrames, 640] → embed_audio. No mel.
            audioEncode = { waveform, dev in
                let frames = Gemma4OmniAudio.frame(
                    waveform, samplesPerToken: samplesPerToken,
                    dtype: dtype, device: dev)
                return embedAudio.embed(frames, device: dev)
            }
        }

        let videoTokenId =
            config.int("video_token_id") ?? config.int("video_token_index")
            ?? defaultVideoTokenId

        // The omni image processor is rescale-only; the encoder overrides
        // `preprocessImage`, so this normalization is just the unused fallback.
        let identity = ImageNormalization(mean: (0, 0, 0), std: (1, 1, 1))
        return try VisionModel(
            visionEncoder: vision, engine: textEngine,
            imageTokenId: imageTokenId, videoTokenId: videoTokenId,
            normalization: identity,
            imageTokenCount: vision.cfg.numSoftTokens,
            audioTokenId: audioTokenId, audioEncode: audioEncode)
    }
}
