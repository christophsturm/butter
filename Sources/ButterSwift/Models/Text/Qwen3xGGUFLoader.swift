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
// Qwen 3.5 / Qwen 3.6 GGUF checkpoint loading.
//
// Unlike DeepSeekV4's GGUF path (`DeepSeekV4Model.loadFromGGUF`), which
// builds a brand-new lazy-per-layer model class because DSv4's
// safetensors forward path was never finished, `Qwen35Hybrid`'s
// safetensors path is mature and load-bearing (every published
// Qwen3.5/3.6 4-bit MLX checkpoint runs through it today). Re-deriving
// the hybrid GDN/attention decode logic a second time for a GGUF-native
// model class would be pure duplication risk. Instead this loader is a
// **translation layer**: it dequants each GGUF tensor on the CPU,
// re-quantizes the big matrices to Butter's native affine int4 g64
// format (`QuantizedOps.quantizeAffine` — the same primitive
// `NemotronHNvfp4.swift`'s `requantizeDenseToInt4NH` and `butter
// convert` use), and populates an in-memory `SafeTensorsBundle`
// (`SafeTensorsFile.init(url:entries:)` / `SafeTensorsBundle
// .init(inMemory:directory:)`) under the exact tensor names
// `Qwen35Hybrid.loadModel` already expects. The existing, tested
// safetensors forward path then runs completely unchanged.
//
// ─── GGUF → HF tensor-name / layout mapping ───────────────────────────
//
// Metadata namespace is `qwen35.*` (both the Qwen 3.5 and Qwen 3.6
// lines share this GGUF architecture string — Qwen 3.6 is a checkpoint-
// size variant of the same arch, matching Butter's own `Qwen3xText.swift`
// file-level comment).
//
//   qwen35.block_count            → num_hidden_layers + 1 (last block
//                                    is the MTP `nextn` head — NOT
//                                    counted in num_hidden_layers; see
//                                    "MTP" below)
//   qwen35.embedding_length       → hidden_size
//   qwen35.attention.head_count   → num_attention_heads
//   qwen35.attention.head_count_kv→ num_key_value_heads
//   qwen35.attention.key_length   → head_dim
//   qwen35.attention.layer_norm_rms_epsilon → rms_norm_eps
//   qwen35.feed_forward_length    → intermediate_size
//   qwen35.context_length         → max_position_embeddings
//   qwen35.full_attention_interval→ full_attention_interval
//   qwen35.rope.freq_base         → rope_theta
//   qwen35.rope.dimension_count / head_dim → partial_rotary_factor
//   qwen35.ssm.conv_kernel        → linear_conv_kernel_dim
//   qwen35.ssm.group_count        → linear_num_key_heads
//   qwen35.ssm.state_size         → linear_key_head_dim AND
//                                    linear_value_head_dim — CHECKPOINT-
//                                    VERIFIED assumption: GGUF's ssm.*
//                                    namespace (borrowed from the Mamba2
//                                    schema) doesn't carry an
//                                    independent value-head-dim key, and
//                                    on Qwen3.6-27B (cross-checked
//                                    against the real HF `config.json`
//                                    — `linear_key_head_dim` == 128 ==
//                                    `linear_value_head_dim` == `ssm.
//                                    state_size`) the two head dims are
//                                    equal, so this holds for every
//                                    checkpoint that shape-matches. A
//                                    future Qwen3.x GGUF with distinct
//                                    K/V head dims would need this
//                                    revisited.
//
// Per-layer weights: GGUF's `blk.N.*` tensors map directly onto the
// safetensors names `loadModel`/`buildGDNMixer` already read — no
// value permutation needed for the big matrices (see below), just a
// name + dtype translation:
//
//   attn_norm.weight            → input_layernorm.weight
//   post_attention_norm.weight  → post_attention_layernorm.weight
//   ffn_gate/up/down.weight     → mlp.gate_proj/up_proj/down_proj.weight
//   (GDN layers)
//   attn_qkv.weight             → linear_attn.in_proj_qkv.weight
//   attn_gate.weight            → linear_attn.in_proj_z.weight
//   ssm_alpha.weight            → linear_attn.in_proj_a.weight
//   ssm_beta.weight             → linear_attn.in_proj_b.weight
//   ssm_out.weight              → linear_attn.out_proj.weight
//   ssm_conv1d.weight           → linear_attn.conv1d.weight (SHAPE TRICK
//                                  — see below)
//   ssm_norm.weight             → linear_attn.norm.weight (read
//                                  directly, bypasses the centered-norm
//                                  fold entirely — never folded either
//                                  convention)
//   ssm_a / ssm_dt.bias         → linear_attn.A_log / dt_bias
//   (full-attention layers)
//   attn_q/k/v/output.weight    → self_attn.q/k/v/o_proj.weight
//   attn_q_norm/k_norm.weight   → self_attn.q_norm/k_norm.weight
//
// `attn_q.weight`'s packed `[query(headDim) | gate(headDim)]`-per-head
// layout (Qwen3.5's `attn_output_gate`) is preserved byte-for-byte: GGUF
// conversion for a modern (non-permuted-RoPE-era) architecture like this
// one copies HF's row order unchanged — row `r` of the on-disk matrix is
// output feature `r` both in HF and in GGUF, so reshaping the OUTPUT
// dimension into `[nHeads, 2, headDim]` after dequant gives the exact
// same query/gate split `sliceHeadHalves35` already implements. Verified
// via the oracle comparison (see the load report) rather than assumed
// blind — a wrong assumption here would show up as incoherent decode
// output with a token-1 mismatch against the llama.cpp oracle.
//
// ─── Centered-RMSNorm fold — GGUF does NOT need it ────────────────────
//
// `Qwen35Hybrid.loadModel`'s `needsCenteredNormFold` heuristic exists to
// distinguish raw HF norm weights (centered around 0, need `+1` folded
// in at load) from mlx-community's pre-folded convention (~1-centered,
// no fold). Empirically dumping this GGUF's norm tensors
// (`blk.0.attn_norm.weight` mean ≈ 0.97, `output_norm.weight` mean ≈
// 1.94, `blk.0.ssm_norm.weight` mean ≈ 0.87 — all clustered near 1, NOT
// near 0) shows llama.cpp's own Qwen3.5/3.6 GGUF conversion ships the
// ALREADY-FOLDED convention, unlike raw HF safetensors. So this loader
// must force `needsCenteredNormFold == false`, which the existing
// heuristic derives from `hasUnsanitizedConv1d = conv1d.weight.shape
// .last != 1`. We supply `linear_attn.conv1d.weight` with a synthetic
// trailing-1 shape `[C, K, 1]` (mlx-community's "sanitized" marker) —
// the flat byte content is unaffected (the transpose that consumes it
// only cares about element order, not the shape label), but the shape
// now reads as "already sanitized" and the fold is correctly skipped.
// Getting this wrong (double-folding an already-1-centered weight to
// ~2) would silently corrupt every norm in the model — this was caught
// by dumping real tensor values before writing the loader, not assumed.
//
// ─── MTP (multi-token prediction) — NOT wired ─────────────────────────
//
// GGUF's `blk.<block_count-1>.nextn.*` tensors (the MTP draft head) are
// skipped entirely: `num_hidden_layers` is set to `block_count - 1` so
// the main decoder never looks past the real transformer depth, and no
// `mtp-sidecar` directory exists for `Qwen3xMTPHead` to find, so the
// optional MTP attach cleanly no-ops. Speculative decode is out of
// scope for this pass — baseline greedy generation is the goal.

import Foundation
import Metal

public enum Qwen3xGGUFError: Error, CustomStringConvertible {
    case missingTensor(String)
    case missingMetadataKey(String)
    case shapeMismatch(String)
    case unsupportedQuantType(GGUFTensorType, tensor: String)

    public var description: String {
        switch self {
        case .missingTensor(let n): return "Qwen3xGGUF: checkpoint is missing tensor '\(n)'"
        case .missingMetadataKey(let k): return "Qwen3xGGUF: required metadata key '\(k)' is missing"
        case .shapeMismatch(let m): return "Qwen3xGGUF: shape mismatch — \(m)"
        case .unsupportedQuantType(let t, let tensor):
            return "Qwen3xGGUF: GGUF quant '\(t)' for tensor '\(tensor)' not supported by the CPU dequant path"
        }
    }
}

enum Qwen3xGGUFLoader {
    /// Butter's native affine-quantize width — matches
    /// `requantizeDenseToInt4NH` (`QuantizedOpsValidation
    /// .validateAffineQuantize`'s kernel invariant: group size 64 is
    /// the only width the `iron_affine_quantize_int4_*` kernels emit).
    private static let quantBits = 4
    private static let quantGroupSize = 64

    /// Top-level entry: parse `qwen35.*` GGUF metadata into a
    /// `ModelConfig`, translate every tensor into the safetensors
    /// namespace `Qwen35Hybrid.loadModel` expects (dequant + requant to
    /// int4 for the big matrices, straight copy for norms/small
    /// vectors), and delegate to the existing, tested safetensors
    /// forward path.
    static func loadModel(
        gguf: GGUFTensorBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Qwen35Model {
        let reader = gguf.reader

        func metaU32(_ key: String) throws -> Int {
            guard let v = reader.metadataUInt32(key) else {
                throw Qwen3xGGUFError.missingMetadataKey(key)
            }
            return Int(v)
        }
        func metaFloat(_ key: String) throws -> Float {
            guard let v = reader.metadataFloat(key) else {
                throw Qwen3xGGUFError.missingMetadataKey(key)
            }
            return v
        }

        let hidden = try metaU32("qwen35.embedding_length")
        let nHeads = try metaU32("qwen35.attention.head_count")
        let nKVHeads = try metaU32("qwen35.attention.head_count_kv")
        let headDim = try metaU32("qwen35.attention.key_length")
        let rmsEps = try metaFloat("qwen35.attention.layer_norm_rms_epsilon")
        let intermediate = try metaU32("qwen35.feed_forward_length")
        let maxSeq = try metaU32("qwen35.context_length")
        let fullAttnInterval = try metaU32("qwen35.full_attention_interval")
        let ropeTheta = try metaFloat("qwen35.rope.freq_base")
        let ropeDimCount = try metaU32("qwen35.rope.dimension_count")
        let partialRotaryFactor = Float(ropeDimCount) / Float(headDim)
        let convKernel = try metaU32("qwen35.ssm.conv_kernel")
        let linearNumKeyHeads = try metaU32("qwen35.ssm.group_count")
        let linearKeyHeadDim = try metaU32("qwen35.ssm.state_size")
        // See file header — GGUF's ssm.* namespace has no independent
        // value-head-dim key; verified equal to the key head dim for
        // this checkpoint family.
        let linearValueHeadDim = linearKeyHeadDim
        let blockCount = try metaU32("qwen35.block_count")
        let nextnLayers = reader.metadataUInt32("qwen35.nextn_predict_layers").map(Int.init) ?? 0
        let nLayers = blockCount - nextnLayers

        // valueDim (in_proj_z / linear_attn.norm's row count) is read
        // straight off the on-disk tensor shape rather than derived
        // from metadata a second time — this is the ground truth the
        // GDN mixer geometry validation (`OpsValidation
        // .validateGatedDeltaStep`) has to satisfy, so deriving it from
        // the actual weight avoids a metadata/tensor mismatch surfacing
        // as a silent shape bug three layers of indirection later.
        guard let gateInfo = reader.tensorIndex["blk.0.attn_gate.weight"] else {
            throw Qwen3xGGUFError.missingTensor("blk.0.attn_gate.weight")
        }
        let valueDim = Int(reader.tensorInfos[gateInfo].numElements) / hidden
        let linearNumValueHeads = valueDim / linearValueHeadDim
        let keyDim = linearKeyHeadDim * linearNumKeyHeads
        let convDim = keyDim * 2 + valueDim

        // vocab_size — derived from token_embd.weight's element count
        // (hidden is already known), not hardcoded, so this loader
        // works unchanged across Qwen3.5/3.6 checkpoints with different
        // vocab sizes.
        guard let embedInfo = reader.tensorIndex["token_embd.weight"] else {
            throw Qwen3xGGUFError.missingTensor("token_embd.weight")
        }
        let vocab = Int(reader.tensorInfos[embedInfo].numElements) / hidden

        // layer_types — derived from which tensors actually exist per
        // layer (not the interval rule) so a checkpoint with an
        // irregular schedule still loads correctly.
        var layerTypes: [String] = []
        layerTypes.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            if reader.tensorIndex["blk.\(i).attn_q.weight"] != nil {
                layerTypes.append("full_attention")
            } else if reader.tensorIndex["blk.\(i).attn_qkv.weight"] != nil {
                layerTypes.append("linear_attention")
            } else {
                throw Qwen3xGGUFError.missingTensor(
                    "blk.\(i).{attn_q,attn_qkv}.weight (layer kind undetermined)")
            }
        }
        // Sanity-check against the documented interval rule — surfaces
        // a metadata/tensor disagreement loudly instead of silently
        // decoding with a scrambled hybrid schedule.
        for i in 0..<nLayers {
            let expected = (i + 1) % fullAttnInterval == 0 ? "full_attention" : "linear_attention"
            if layerTypes[i] != expected {
                Debug.log(
                    .load,
                    "Qwen3xGGUF: layer \(i) kind '\(layerTypes[i])' from tensor "
                        + "presence disagrees with the full_attention_interval="
                        + "\(fullAttnInterval) rule (expected '\(expected)') — "
                        + "trusting the tensor-presence result")
            }
        }

        let raw: [String: Any] = [
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_attention_heads": nHeads,
            "num_hidden_layers": nLayers,
            "num_key_value_heads": nKVHeads,
            "head_dim": headDim,
            "rms_norm_eps": Double(rmsEps),
            "tie_word_embeddings": false,
            "intermediate_size": intermediate,
            "max_position_embeddings": maxSeq,
            "full_attention_interval": fullAttnInterval,
            "attn_output_gate": true,
            "rope_theta": Double(ropeTheta),
            "partial_rotary_factor": Double(partialRotaryFactor),
            "linear_num_key_heads": linearNumKeyHeads,
            "linear_num_value_heads": linearNumValueHeads,
            "linear_key_head_dim": linearKeyHeadDim,
            "linear_value_head_dim": linearValueHeadDim,
            "linear_conv_kernel_dim": convKernel,
            "num_experts": 0,
            "num_experts_per_tok": 0,
            "layer_types": layerTypes,
            "quantization": [
                "bits": quantBits, "group_size": quantGroupSize, "mode": "affine",
            ] as [String: Any],
        ]
        let config = ModelConfig(
            architecture: "Qwen3_5ForCausalLM", modelType: "qwen3_5", raw: raw)

        Debug.log(
            .load,
            "Qwen3xGGUF: hidden=\(hidden) vocab=\(vocab) nLayers=\(nLayers) "
                + "nHeads=\(nHeads)/\(nKVHeads) headDim=\(headDim) "
                + "GDN keyDim=\(keyDim)(\(linearNumKeyHeads)x\(linearKeyHeadDim)) "
                + "valueDim=\(valueDim)(\(linearNumValueHeads)x\(linearValueHeadDim)) "
                + "convDim=\(convDim) convKernel=\(convKernel) "
                + "fullAttnLayers=\(layerTypes.filter { $0 == "full_attention" }.count)")

        // ── Build the in-memory tensor namespace ───────────────────────
        var entries: [String: SafeTensorsFile.Entry] = [:]
        entries.reserveCapacity(nLayers * 16 + 8)

        // Q4_K_M is itself a MIXED-precision scheme: llama.cpp's
        // quantizer deliberately ships numerically-sensitive tensors
        // (attn_v/attn_qkv/ffn_down/output, per its per-tensor
        // heuristic) at Q5_K/Q6_K (5-6 bit) while the bulk of the
        // model is Q4_K (~4 bit) — see the quant-type census in this
        // file's header / the load report. Flattening EVERY tensor to
        // uniform int4 on requant discards exactly the precision the
        // original quantizer allocated to the tensors it judged most
        // sensitive, compounding: Q6_K→f32→bf16→int4 loses far more
        // than Q6_K's own ~0.1% error budget. Preserve that signal:
        // requant Q5_K/Q6_K-sourced tensors to int8 (clean pack-factor
        // width, no odd-bit-stream packing) and leave the Q4_K bulk at
        // int4. `deriveAffineQuantBits` on the consuming side already
        // auto-detects bits per-tensor from the packed shape, so mixed
        // precision needs no plumbing beyond picking `bits` here.
        func requantBits(for ggufName: String) throws -> Int {
            guard let idx = reader.tensorIndex[ggufName] else {
                throw Qwen3xGGUFError.missingTensor(ggufName)
            }
            switch reader.tensorInfos[idx].type {
            case .q4_K, .f32, .f16, .bf16: return quantBits
            default: return 8  // Q5_K / Q6_K / Q8_0 source
            }
        }

        func addDense2D(key: String, ggufName: String, outDim: Int, inDim: Int) throws {
            let bits = try requantBits(for: ggufName)
            let dense = try autoreleasepool {
                try materializeDense2D(
                    reader: reader, name: ggufName, outDim: outDim, inDim: inDim, device: device)
            }
            autoreleasepool {
                addQuantizedLinear(
                    entries: &entries, key: key, dense: dense,
                    outDim: outDim, inDim: inDim, device: device,
                    bits: bits, groupSize: quantGroupSize)
            }
        }

        // `Ops.rmsNorm`/`Ops.rmsNormRows` (`RMSNorm.callAsFunction`)
        // precondition `x.dtype == weight.dtype` — every norm weight
        // (outer AND the GDN gated `linear_attn.norm`) MUST match
        // `activationDtype` (bf16), not stay f32. Caught empirically:
        // an f32-norm first pass loaded and even got through the
        // dtype fix above, but crashed on the very first forward step
        // with `Ops.rmsNorm: dtype mismatch` inside
        // `Qwen35GDNLayer.decode`. `dtype` defaults to bf16 for this
        // reason; the two callers that read host-side only
        // (`A_log`/`dt_bias` — converted to `[Float]` via
        // `readFloats35` and never touch a dtype-checked GPU kernel)
        // pass `.f32` explicitly to keep full source precision.
        func addVector(key: String, ggufName: String, n: Int, dtype: DType = .bf16) throws {
            let floats = try materializeFlatF32Array(reader: reader, name: ggufName, nValues: n)
            let t = Tensor.empty(shape: [n], dtype: dtype, device: device)
            switch dtype {
            case .f32: t.copyIn(from: floats)
            case .bf16: t.copyIn(from: floatsToBF16(floats))
            default: preconditionFailure("addVector: unsupported dtype \(dtype)")
            }
            entries[key] = SafeTensorsFile.Entry(name: key, dtype: dtype, shape: [n], buffer: t.buffer)
        }

        // ── Non-block weights ──────────────────────────────────────────
        try addDense2D(
            key: "model.embed_tokens", ggufName: "token_embd.weight",
            outDim: vocab, inDim: hidden)
        try addVector(key: "model.norm.weight", ggufName: "output_norm.weight", n: hidden)
        try addDense2D(key: "lm_head", ggufName: "output.weight", outDim: vocab, inDim: hidden)

        // ── Per-layer weights ───────────────────────────────────────────
        for i in 0..<nLayers {
            let g = "blk.\(i)"
            let p = "model.layers.\(i)"

            try addVector(key: "\(p).input_layernorm.weight", ggufName: "\(g).attn_norm.weight", n: hidden)
            try addVector(
                key: "\(p).post_attention_layernorm.weight",
                ggufName: "\(g).post_attention_norm.weight", n: hidden)

            // Dense SwiGLU MLP — every Qwen3.5/3.6 layer on this
            // checkpoint family (num_experts == 0); MoE routing is out
            // of scope for this loader (no MoE tensors exist in a dense
            // GGUF to route to).
            try addDense2D(
                key: "\(p).mlp.gate_proj", ggufName: "\(g).ffn_gate.weight",
                outDim: intermediate, inDim: hidden)
            try addDense2D(
                key: "\(p).mlp.up_proj", ggufName: "\(g).ffn_up.weight",
                outDim: intermediate, inDim: hidden)
            try addDense2D(
                key: "\(p).mlp.down_proj", ggufName: "\(g).ffn_down.weight",
                outDim: hidden, inDim: intermediate)

            if layerTypes[i] == "linear_attention" {
                let lp = "\(p).linear_attn"
                try addDense2D(
                    key: "\(lp).in_proj_qkv", ggufName: "\(g).attn_qkv.weight",
                    outDim: convDim, inDim: hidden)
                try addDense2D(
                    key: "\(lp).in_proj_z", ggufName: "\(g).attn_gate.weight",
                    outDim: valueDim, inDim: hidden)
                try addDense2D(
                    key: "\(lp).in_proj_a", ggufName: "\(g).ssm_alpha.weight",
                    outDim: linearNumValueHeads, inDim: hidden)
                try addDense2D(
                    key: "\(lp).in_proj_b", ggufName: "\(g).ssm_beta.weight",
                    outDim: linearNumValueHeads, inDim: hidden)
                try addDense2D(
                    key: "\(lp).out_proj", ggufName: "\(g).ssm_out.weight",
                    outDim: hidden, inDim: valueDim)

                // conv1d.weight: raw on-disk layout is c-major / k-fastest
                // (row r = channel r's `convKernel` taps), matching what
                // `transposeConv1dWeight35` expects as its flat input —
                // see the file header for why the shape is `[C, K, 1]`
                // (trailing-1 "already sanitized" marker) rather than the
                // physically-accurate `[C, K]`.
                let convFloats = try materializeFlatF32Array(
                    reader: reader, name: "\(g).ssm_conv1d.weight",
                    nValues: convDim * convKernel)
                let convT = Tensor.empty(shape: [convDim, convKernel, 1], dtype: .f32, device: device)
                convT.copyIn(from: convFloats)
                entries["\(lp).conv1d.weight"] = SafeTensorsFile.Entry(
                    name: "\(lp).conv1d.weight", dtype: .f32,
                    shape: [convDim, convKernel, 1], buffer: convT.buffer)

                try addVector(key: "\(lp).norm.weight", ggufName: "\(g).ssm_norm.weight", n: linearValueHeadDim)

                // GGUF's `ssm_a` is llama.cpp's PRE-COMPUTED `-exp(A_log)`,
                // not the raw `A_log` parameter Butter's GDN math expects.
                // Ground truth: llama.cpp's own forward pass
                // (`models/qwen35.cpp build_layer_attn_linear`) computes
                // `gate = softplus(alpha + dt_bias) * ssm_a` — a plain
                // multiply, no further `exp`/negate applied to `ssm_a` at
                // forward time — so `ssm_a` must already BE the final
                // `-exp(A_log)` term. Confirmed empirically: every
                // `blk.N.ssm_a` value in this checkpoint is small and
                // strictly negative (range ~[-0.34, -0.004]), the exact
                // signature of `-exp(x)` for a real-valued `x`, not a raw
                // log-parameter. Butter's `Qwen35GDNMixer` (matching
                // mlx-lm) instead takes raw `A_log` and computes
                // `-exp(A_log)` itself internally (see
                // `negExpALogTF32 = aLog.map { -exp($0) }` in
                // `Qwen3xText.swift`), so feeding GGUF's `ssm_a` through
                // unchanged double-transforms it. Invert: `A_log_fed =
                // log(-ssm_a)` so Butter's own `-exp(A_log_fed))`
                // reproduces the GGUF value exactly.
                //
                // Caught by cross-checking greedy-generate output against
                // the existing MLX safetensors loader for the same
                // checkpoint family: MLX correctly predicted " Paris" for
                // "The capital of France is"; the GGUF path (before this
                // fix) degenerated into repeating prompt tokens — a
                // classic wrong-decay-gate symptom (the GDN state either
                // never forgets or forgets everything), not the
                // characteristic NaN/garbage of a shape or dtype bug.
                let ssmARaw = try materializeFlatF32Array(
                    reader: reader, name: "\(g).ssm_a", nValues: linearNumValueHeads)
                let aLogFed = ssmARaw.map { Foundation.log(-$0) }
                let aLogT = Tensor.empty(shape: [linearNumValueHeads], dtype: .f32, device: device)
                aLogT.copyIn(from: aLogFed)
                entries["\(lp).A_log"] = SafeTensorsFile.Entry(
                    name: "\(lp).A_log", dtype: .f32, shape: [linearNumValueHeads], buffer: aLogT.buffer)

                // dt_bias is read host-side only (`readFloats35` inside
                // `buildGDNMixer`, converted to `[Float]` and never bound
                // to a dtype-checked GPU kernel) — keep full f32 source
                // precision rather than following the norm vectors' bf16
                // default. No value transform needed (GGUF's `ssm_dt.bias`
                // is the same additive bias in both conventions — it
                // enters the softplus argument identically either way).
                try addVector(
                    key: "\(lp).dt_bias", ggufName: "\(g).ssm_dt.bias", n: linearNumValueHeads,
                    dtype: .f32)
            } else {
                let ap = "\(p).self_attn"
                let qOutDim = 2 * nHeads * headDim  // [query | gate] per head
                try addDense2D(
                    key: "\(ap).q_proj", ggufName: "\(g).attn_q.weight",
                    outDim: qOutDim, inDim: hidden)
                try addDense2D(
                    key: "\(ap).k_proj", ggufName: "\(g).attn_k.weight",
                    outDim: nKVHeads * headDim, inDim: hidden)
                try addDense2D(
                    key: "\(ap).v_proj", ggufName: "\(g).attn_v.weight",
                    outDim: nKVHeads * headDim, inDim: hidden)
                try addDense2D(
                    key: "\(ap).o_proj", ggufName: "\(g).attn_output.weight",
                    outDim: hidden, inDim: nHeads * headDim)
                try addVector(key: "\(ap).q_norm.weight", ggufName: "\(g).attn_q_norm.weight", n: headDim)
                try addVector(key: "\(ap).k_norm.weight", ggufName: "\(g).attn_k_norm.weight", n: headDim)
            }

            if i % 8 == 0 {
                Debug.log(.load, "Qwen3xGGUF: loaded layer \(i)/\(nLayers) (\(layerTypes[i]))")
            }
        }

        let file = SafeTensorsFile(url: reader.url, entries: entries)
        let bundle = SafeTensorsBundle(
            inMemory: file, directory: reader.url.deletingLastPathComponent())

        return try Qwen35Hybrid.loadModel(
            config: config, weights: bundle, options: options, device: device)
    }

    // ─── Per-tensor materialization ────────────────────────────────────

    /// Dequant a named GGUF 2D weight matrix (flat `outDim * inDim`
    /// block-quantized elements) into a dense `[outDim, inDim]` bf16
    /// `Tensor`. Row-major output — GGUF's block layout is already
    /// row-per-output-feature (blocks run along the contraction/input
    /// axis), so no transpose is needed regardless of the tensor's
    /// on-disk `dimensions` axis order (which the generic
    /// `GGUFTensorBundle.tensor(named:)` mislabels for 2D tensors — see
    /// this loader deliberately bypassing that API for weight matrices).
    ///
    /// bf16, not f32: `quantizeAffine`'s output `scales`/`biases` dtype
    /// must match this dense input's dtype (see `addQuantizedLinear`),
    /// and that dtype becomes the whole model's `activationDtype` (read
    /// off `model.embed_tokens.scales` in `Qwen35Hybrid.loadModel`) —
    /// several fused kernels (e.g. the GDN mixer's SiLU-conv fusion)
    /// have no f32 code path, only f16/bf16, matching every real
    /// checkpoint (safetensors or GGUF) which always ships bf16 or f16
    /// activations. This was caught empirically: an f32 first pass
    /// loaded cleanly but crashed on the first forward step with
    /// `Ops.siluCastF32PlusCastF32Two: unsupported dtype f32`.
    private static func materializeDense2D(
        reader: GGUFReader, name: String, outDim: Int, inDim: Int, device: Device
    ) throws -> Tensor {
        let floats = try materializeFlatF32Array(reader: reader, name: name, nValues: outDim * inDim)
        let t = Tensor.empty(shape: [outDim, inDim], dtype: .bf16, device: device)
        t.copyIn(from: floatsToBF16(floats))
        return t
    }

    /// Round-to-nearest f32 → bf16 (truncate the low 16 mantissa bits
    /// after rounding) — same algorithm as `Qwen3xText.swift`'s private
    /// `writeFloats35` bf16 branch, duplicated here rather than widening
    /// that file's access (this is the only dtype this loader needs it
    /// for).
    private static func floatsToBF16(_ values: [Float]) -> [UInt16] {
        values.map { v -> UInt16 in
            let bits = v.bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            return UInt16(rounded >> 16)
        }
    }

    /// Read + dequant (if needed) a named GGUF tensor to a flat `[Float]`
    /// of length `nValues`, dispatching on the on-disk quant type. F32
    /// is a straight little-endian reinterpret (Apple Silicon is
    /// little-endian, matching GGUF's on-disk byte order); Q4_K/Q5_K/
    /// Q6_K route through the verified CPU dequant in `GGUFDequantK.swift`.
    private static func materializeFlatF32Array(
        reader: GGUFReader, name: String, nValues: Int
    ) throws -> [Float] {
        guard let idx = reader.tensorIndex[name] else {
            throw Qwen3xGGUFError.missingTensor(name)
        }
        let info = reader.tensorInfos[idx]
        guard Int(info.numElements) == nValues else {
            throw Qwen3xGGUFError.shapeMismatch(
                "\(name): expected \(nValues) elements, GGUF tensor has \(info.numElements)")
        }
        return try reader.withRawBytes(named: name) { raw -> [Float] in
            switch info.type {
            case .f32:
                guard let base = raw.baseAddress else { return [Float](repeating: 0, count: nValues) }
                return base.withMemoryRebound(to: Float.self, capacity: nValues) { fp in
                    Array(UnsafeBufferPointer(start: fp, count: nValues))
                }
            case .q4_K:
                return GGUFDequant.dequantQ4_K_cpu(rawBlocks: raw, nValues: nValues)
            case .q5_K:
                return GGUFDequant.dequantQ5_K_cpu(rawBlocks: raw, nValues: nValues)
            case .q6_K:
                return GGUFDequant.dequantQ6_K_cpu(rawBlocks: raw, nValues: nValues)
            default:
                throw Qwen3xGGUFError.unsupportedQuantType(info.type, tensor: name)
            }
        }
    }

    /// Requantize a dense `[outDim, inDim]` f32 tensor to Butter's
    /// native affine int4 (group size 64) `<key>.weight/.scales/.biases`
    /// triplet — identical primitive to `requantizeDenseToInt4NH`, just
    /// writing directly into the in-memory bundle's entry map instead of
    /// constructing a `QuantizedLinear` (the bundle-populate step is
    /// shared code that then flows through `loadLinear` like every other
    /// family).
    private static func addQuantizedLinear(
        entries: inout [String: SafeTensorsFile.Entry], key: String, dense: Tensor,
        outDim: Int, inDim: Int, device: Device, bits: Int, groupSize: Int
    ) {
        precondition(
            inDim % groupSize == 0,
            "Qwen3xGGUF: \(key) inDim=\(inDim) must be a multiple of groupSize=\(groupSize)")
        let packedCols = inDim * bits / 32
        let scaleCols = inDim / groupSize
        // scales/biases MUST match `dense.dtype` — `QuantizedOps
        // .quantizeAffine` preconditions on it, and (more importantly)
        // that dtype becomes the model's `activationDtype`. See
        // `materializeDense2D`'s doc for why this has to be bf16/f16,
        // never f32.
        let packed = Tensor.empty(shape: [outDim, packedCols], dtype: .u32, device: device)
        let scales = Tensor.empty(shape: [outDim, scaleCols], dtype: dense.dtype, device: device)
        let biases = Tensor.empty(shape: [outDim, scaleCols], dtype: dense.dtype, device: device)
        let cmd = device.makeCommandBuffer()
        QuantizedOps.quantizeAffine(
            weight: dense, packed: packed, scales: scales, biases: biases,
            bits: bits, groupSize: groupSize, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        entries["\(key).weight"] = SafeTensorsFile.Entry(
            name: "\(key).weight", dtype: .u32, shape: packed.shape, buffer: packed.buffer)
        entries["\(key).scales"] = SafeTensorsFile.Entry(
            name: "\(key).scales", dtype: dense.dtype, shape: scales.shape, buffer: scales.buffer)
        entries["\(key).biases"] = SafeTensorsFile.Entry(
            name: "\(key).biases", dtype: dense.dtype, shape: biases.shape, buffer: biases.buffer)
    }
}
