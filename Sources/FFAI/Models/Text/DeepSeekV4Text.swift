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
// DeepSeek V4 text backbone — DSv4-Flash / DSv4-Pro decoder config +
// variants.
//
// **Status:** Scaffold. This file declares the static shape — the
// `DeepSeekV4TextConfig` decoder, the two variants (`DeepSeekV4Flash`,
// `DeepSeekV4Pro`), and the `DeepSeekV4Model` placeholder — so the
// loader can identify a DSv4 checkpoint (safetensors or GGUF) and
// dispatch into the family. The forward path land in follow-up PRs
// per the multi-week metaltile kernel sequence (MLA decode, CSA
// sparse-gather SDPA, HCA compressed-stream SDPA, Lightning Indexer,
// FP4 / block-FP8 dequant).
//
// ─── Architecture summary (from the upstream config) ─────────────────
//
// • 43 transformer layers + 1 MTP head, hidden=4096, vocab=129,280.
// • Interleaved attention pattern (`compress_ratios` aligned 1-1 with
//   the layer stack): layers 0-1 = full attention; layers 2-41 =
//   alternating CSA(4×) / HCA(128×) pairs; layer 42 = full; layer 43
//   = MTP next-N predictor.
// • **MLA (Multi-head Latent Attention)** carried over from DSv3.
//   `head_dim=512` (MLA latent dim), `kv_heads=1` (single MLA cache),
//   `qk_rope_head_dim=64` decoupled-RoPE concat tail. Q is low-rank-
//   compressed to `q_lora_rank=1024`. O-projection is now ALSO
//   low-rank (`o_lora_rank=1024`, `o_groups=8`) — new in V4.
// • **CSA (Compressed Sparse Attention)** — 4× compressed KV stream.
//   A 64-head × 128-dim **Lightning Indexer** sub-network scores all
//   compressed entries; per query, top-`index_topk=512` selected +
//   128-token local sliding window. Scoring: `sqrtsoftplus`.
// • **HCA (Heavily Compressed Attention)** — 128× compressed KV
//   stream. Dense attention (no top-k) over the small compressed
//   buffer. Separate `compress_rope_theta=160_000` (vs `rope_theta=
//   10_000` for full + CSA layers).
// • MoE on every non-MTP layer (288 experts, top-k=6, expert
//   intermediate=2048, 1 always-on shared expert at dim=2048).
//   `noaux_tc` aux-loss-free + per-expert bias routing.
// • **`sqrtsoftplus` routing scoring** — replaces DSv3's sigmoid+bias
//   gate (companion `ffai_moe_router_sigmoid_bias` kernel from
//   the Step-3 series is the closest cousin; sqrtsoftplus is its
//   own small variant kernel).
// • **mHC (Manifold-Constrained Hyper-Connections)** — residual
//   projection matrix constrained to the Birkhoff polytope (doubly
//   stochastic) via 20 Sinkhorn-Knopp iterations. Folded into
//   weights at LOAD time, not per token — no runtime kernel needed.
// • **Clamped SwiGLU** — `swiglu_limit=10.0` applied to every MoE
//   expert MLP. The Step-3 `mt_clamped_swiglu` kernel is the
//   drop-in here.
// • **MTP head** — `num_nextn_predict_layers=1`. Carries the same
//   per-layer block shape as the main stack but its output flows
//   into a separate logits head used by speculative decoding.
//
// ─── Mixed precision (from the upstream config) ──────────────────────
//
// • MoE expert weights stored as **fp4** (FP4 e2m1, block 32, with
//   per-block fp8 e4m3 scales). New kernel family — not in
//   metaltile-std today.
// • Attention / router weights stored as **block-FP8** (FP8 e4m3,
//   block 128×128). Distinct from per-channel int4 affine; another
//   net-new dequant kernel.
// • Activations: bf16 throughout. Output norms: f32.

import Foundation
import Metal
import MetalTileSwift
import QuartzCore

// ─── DeepSeekV4TextConfig ────────────────────────────────────────────

public struct DeepSeekV4TextConfig: Sendable {
    let nLayers: Int  // 43 (excludes MTP)
    let hidden: Int  // 4096
    let vocab: Int  // 129_280
    let maxSeq: Int  // 1_048_576 (1M)
    let rmsNormEps: Float

    // ── Attention shape ──
    let nHeads: Int  // 64
    let nKVHeads: Int  // 1 — MLA carries one logical KV head
    let headDim: Int  // 512 — MLA latent dim
    let qkRopeHeadDim: Int  // 64 — decoupled-RoPE concat tail
    let qLoraRank: Int  // 1024 — Q low-rank compression
    let oLoraRank: Int  // 1024 — O-projection low-rank (new in V4)
    let oGroups: Int  // 8 — O-projection group count

    // ── Per-layer compression schedule ──
    /// `compress_ratios` mirror of the layer stack (length =
    /// nLayers + MTP slot). Value semantics:
    ///   - 0 → full attention
    ///   - 4 → CSA (4× compression)
    ///   - 128 → HCA (128× compression)
    let layerCompressRatios: [Int]
    let slidingWindow: Int  // 128 — CSA local window

    // ── Lightning Indexer ──
    let indexerHeads: Int  // 64
    let indexerHeadDim: Int  // 128
    let indexerTopK: Int  // 512

    // ── mHC ──
    let hcMultiplier: Int  // 4
    let hcEpsilon: Float  // 1e-6
    let hcSinkhornIterations: Int  // 20

    // ── RoPE ──
    let ropeTheta: Float  // 10_000 (full + CSA)
    let compressRopeTheta: Float  // 160_000 (HCA compressed stream)
    let yarnFactor: Float  // 16
    let yarnOriginalContext: Int  // 65_536

    // ── MoE ──
    let nExperts: Int  // 288
    let nExpertsPerToken: Int  // 6
    let nSharedExperts: Int  // 1
    let moeIntermediate: Int  // 2048
    let sharedExpertIntermediate: Int  // 2048
    let routerBias: Bool  // true (noaux_tc)
    let routerScalingFactor: Float  // 1.5
    let routerScoringFunc: String  // "sqrtsoftplus"

    // ── Activation clip ──
    let swigluLimit: Float  // 10.0

    // ── MTP ──
    let nMTPLayers: Int  // 1

    static func decode(_ tc: ModelConfig) throws -> DeepSeekV4TextConfig {
        guard
            let nLayers = tc.int("num_hidden_layers"),
            let hidden = tc.int("hidden_size"),
            let vocab = tc.int("vocab_size"),
            let nHeads = tc.int("num_attention_heads")
        else {
            throw DeepSeekV4Error.missingConfig("text_config core attention shape")
        }
        let nKVHeads = tc.int("num_key_value_heads") ?? 1
        let headDim = tc.int("head_dim") ?? 512
        let qkRopeHeadDim = tc.int("qk_rope_head_dim") ?? 64
        let qLoraRank = tc.int("q_lora_rank") ?? 1024
        let oLoraRank = tc.int("o_lora_rank") ?? 1024
        let oGroups = tc.int("o_groups") ?? 8
        let maxSeq =
            tc.int("max_position_embeddings")
            ?? tc.int("model_max_length") ?? 1_048_576

        // `compress_ratios` schedule. If absent, default to a
        // uniformly-full attention stack (which is wrong for DSv4 but
        // safe — the loader will reject before forward is reached).
        let ratios: [Int] = tc.intArray("compress_ratios") ?? Array(repeating: 0, count: nLayers + 1)

        let slidingWindow = tc.int("sliding_window") ?? 128

        // Lightning indexer.
        let indexerHeads = tc.int("index_n_heads") ?? 64
        let indexerHeadDim = tc.int("index_head_dim") ?? 128
        let indexerTopK = tc.int("index_topk") ?? 512

        // mHC.
        let hcMult = tc.int("hc_mult") ?? 4
        let hcEps = Float(tc.float("hc_eps") ?? 1e-6)
        let hcIters = tc.int("hc_sinkhorn_iters") ?? 20

        // RoPE.
        let ropeTheta = Float(tc.float("rope_theta") ?? 10_000)
        let compressRopeTheta = Float(tc.float("compress_rope_theta") ?? 160_000)
        var yarnFactor: Float = 1.0
        var yarnOriginal = maxSeq
        if let rs = tc.nested("rope_scaling") {
            if let f = rs["factor"] as? Double { yarnFactor = Float(f) }
            if let o = rs["original_max_position_embeddings"] as? Int { yarnOriginal = o }
        }

        // MoE.
        let nExperts = tc.int("n_routed_experts") ?? tc.int("num_experts") ?? 288
        let nExpertsPerToken =
            tc.int("num_experts_per_tok")
            ?? tc.int("num_experts_per_token") ?? 6
        let nShared = tc.int("n_shared_experts") ?? 1
        let moeIntermediate =
            tc.int("moe_intermediate_size") ?? tc.int("expert_dim") ?? 2048
        let sharedIntermediate =
            tc.int("share_expert_dim")
            ?? tc.int("shared_expert_intermediate_size")
            ?? moeIntermediate
        let routerScoringFunc = tc.string("scoring_func") ?? "sqrtsoftplus"
        let routerScale = Float(tc.float("routed_scaling_factor") ?? 1.5)

        let swigluLimit = Float(tc.float("swiglu_limit") ?? 10.0)
        let nMTPLayers = tc.int("num_nextn_predict_layers") ?? 1

        return DeepSeekV4TextConfig(
            nLayers: nLayers,
            hidden: hidden,
            vocab: vocab,
            maxSeq: maxSeq,
            rmsNormEps: Float(tc.float("rms_norm_eps") ?? 1e-6),
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim,
            qkRopeHeadDim: qkRopeHeadDim,
            qLoraRank: qLoraRank,
            oLoraRank: oLoraRank,
            oGroups: oGroups,
            layerCompressRatios: ratios,
            slidingWindow: slidingWindow,
            indexerHeads: indexerHeads,
            indexerHeadDim: indexerHeadDim,
            indexerTopK: indexerTopK,
            hcMultiplier: hcMult,
            hcEpsilon: hcEps,
            hcSinkhornIterations: hcIters,
            ropeTheta: ropeTheta,
            compressRopeTheta: compressRopeTheta,
            yarnFactor: yarnFactor,
            yarnOriginalContext: yarnOriginal,
            nExperts: nExperts,
            nExpertsPerToken: nExpertsPerToken,
            nSharedExperts: nShared,
            moeIntermediate: moeIntermediate,
            sharedExpertIntermediate: sharedIntermediate,
            routerBias: tc.bool("router_bias") ?? true,
            routerScalingFactor: routerScale,
            routerScoringFunc: routerScoringFunc,
            swigluLimit: swigluLimit,
            nMTPLayers: nMTPLayers)
    }
}

// ─── Variants ────────────────────────────────────────────────────────

/// 284B total / 13B active. The user-runnable size on Apple Silicon
/// today (4-bit weights + dequant load fits in 128 GB unified memory
/// at ~32K context).
public enum DeepSeekV4Flash: DeepSeekV4Variant {
    public static var availableCapabilities: Set<Capability> { [.textIn, .textOut] }
    public static var defaultGenerationParameters: GenerationParameters {
        GenerationParameters(
            maxTokens: 256, prefillStepSize: 4096,
            temperature: 1.0, topP: 0.95, topK: 64,
            repetitionPenalty: 1.0)
    }

    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> DeepSeekV4Model {
        let tc = DeepSeekV4Config.textConfig(config)
        _ = try DeepSeekV4TextConfig.decode(tc)
        throw DeepSeekV4Error.notYetImplemented("DeepSeekV4Flash safetensors forward path")
    }

    public static func loadModelFromGGUF(
        config: ModelConfig, gguf: GGUFTensorBundle,
        options: LoadOptions, device: Device
    ) throws -> DeepSeekV4Model {
        let tc = DeepSeekV4Config.textConfig(config)
        let textConfig = try DeepSeekV4TextConfig.decode(tc)
        // Architecture-string sanity-check now that the reader is
        // open. Either form is accepted upstream.
        if let arch = gguf.architecture,
            !DeepSeekV4.architectures.contains(arch),
            arch != "deepseek4"
        {
            throw DeepSeekV4Error.missingConfig(
                "general.architecture='\(arch)' not in DeepSeekV4 known set")
        }
        return try DeepSeekV4Model.loadFromGGUF(
            textConfig: textConfig, gguf: gguf, device: device, options: options)
    }

    /// Convenience GGUF loader for benchmarks/CLI that don't have a
    /// `ModelConfig` handy. Builds the minimal DSv4-Flash config from
    /// known dims and delegates to `loadModelFromGGUF`.
    public static func loadFlashFromGGUF(
        gguf: GGUFTensorBundle, device: Device, options: LoadOptions = LoadOptions()
    ) throws -> DeepSeekV4Model {
        let raw: [String: Any] = [
            "hidden_size": 4096, "num_hidden_layers": 43,
            "vocab_size": 129_280, "num_attention_heads": 64,
        ]
        let config = ModelConfig(
            architecture: "DeepSeekV4ForCausalLM", modelType: "deepseek4", raw: raw)
        return try loadModelFromGGUF(
            config: config, gguf: gguf, options: options, device: device)
    }
}

/// ~1.6T / 49B active. Same architecture as Flash, deeper + wider
/// MoE. Not currently runnable on Apple Silicon (weights alone need
/// ~480 GB at 4-bit). Kept here for completeness — the variant
/// dispatch + config decode work today; load throws.
public enum DeepSeekV4Pro: DeepSeekV4Variant {
    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> DeepSeekV4Model {
        _ = config; _ = weights; _ = options; _ = device
        throw DeepSeekV4Error.notYetImplemented("DeepSeekV4Pro safetensors forward path")
    }
}

// ─── DeepSeekV4Model — weight slots ──────────────────────────────────
//
// Tensor inventory mirrors the DSv4-Flash GGUF (see
// `Tests/ModelIntegrationTests/GGUFDsv4TensorMapTest.swift` for the
// full dump). Field names follow the GGUF tensor-name convention so
// the loader is a direct `bundle.tensor(named:"blk.\(N).\(suffix)")`
// dispatch.
//
// Architecture summary:
//
// - Each layer holds an mHC 4-channel residual state H[hidden, 4, t].
// - Attention sub-block: rms_norm → q_a → q_a_norm → q_b → per-head
//   Q-norm (eps-only) → partial RoPE on tail 64 dims of each head;
//   kv (single 512-d MQA head) → kv_a_norm → partial RoPE on tail 64
//   dims → optional FP8 quantize on first 448 dims → store in cache.
//   Softmax-attention with `attn_sinks` (per-head learnable extra
//   logit). Inverse partial RoPE on output. Grouped O-LoRA: reshape
//   to [4096, 8 groups] × [4096, 1024] per group → [8192] → wo_b →
//   [4096].
// - FFN sub-block: rms_norm → MoE (256 experts top-6, sqrt-softplus
//   routing OR precomputed hash routing via `ffn_gate_tid2eid` on
//   the first `n_hash_layers`) + shared-expert SwiGLU.
// - mHC: at each sub-block boundary, `hc_*_fn @ flatten(H)` produces
//   a 24-dim mix that splits as 4 `pre` (sigmoid+eps) + 4 `post`
//   (2·sigmoid) + 16 (=4×4) `comb` matrix (softmax + Sinkhorn-Knopp
//   row/col-normalized). `pre` collapses H → sub-block input;
//   `post` + `comb` expand the sub-block output back into H.
// - CSA (compress_ratio=4): adds a Lightning Indexer that scores all
//   compressed K-entries against a 64-head × 128-dim Q (sharing the
//   same `qr` from the main attn-path) + an attn compressor that
//   builds a 4×-pooled (overlap-2) compressed KV stream. Top-512
//   compressed slots feed the sparse-gather attention.
// - HCA (compress_ratio=128): only the attn compressor (no indexer);
//   dense attention over the small compressed stream.
// - Layer pattern per the GGUF: 0,1 = full; 2,4,…,42 = CSA;
//   3,5,…,41 = HCA. The `compress_ratios` array on the GGUF metadata
//   is authoritative.

/// One transformer block's worth of weights. Allocated per-layer
/// regardless of compression regime — the regime-specific tensors
/// (compressor, indexer) are nil for layers that don't use them.
public final class DeepSeekV4Layer: @unchecked Sendable {
    let layerIndex: Int
    let compressRatio: Int  // 0 = full, 4 = CSA, 128 = HCA

    // ── Common attention path ──
    let attnNorm: Tensor  // f32 [hidden]
    let attnQA: Tensor  // q8_0 [hidden, q_lora_rank]
    let attnQANorm: Tensor  // f32 [q_lora_rank]
    let attnQB: Tensor  // q8_0 [q_lora_rank, n_heads * head_dim]
    let attnKV: Tensor  // q8_0 [hidden, head_dim] (MQA: 1 kv head)
    let attnKVANorm: Tensor  // f32 [head_dim]
    let attnSinks: Tensor  // f32 [n_heads]
    let attnOutputA: Tensor  // q8_0 [group_dim, n_groups * o_lora_rank]
    let attnOutputB: Tensor  // q8_0 [n_groups * o_lora_rank, hidden]

    // ── FFN path ──
    let ffnNorm: Tensor  // f32 [hidden]
    let ffnGateInp: Tensor  // f16 [hidden, n_experts]
    let ffnGateTid2Eid: Tensor?  // i32 [n_experts_per_token, vocab] — hash-route, nil past n_hash_layers
    let ffnGateExps: Tensor  // iq2_xxs [hidden, expert_intermediate, n_experts]
    let ffnUpExps: Tensor  // iq2_xxs [hidden, expert_intermediate, n_experts]
    let ffnDownExps: Tensor  // q2_K [expert_intermediate, hidden, n_experts]
    let ffnGateShexp: Tensor  // q8_0 [hidden, shared_expert_intermediate]
    let ffnUpShexp: Tensor  // q8_0 [hidden, shared_expert_intermediate]
    let ffnDownShexp: Tensor  // q8_0 [shared_expert_intermediate, hidden]
    let expProbsBias: Tensor?  // f32 [n_experts] — noaux_tc bias, only on non-hash layers

    // ── mHC weights (attn + ffn sub-blocks) ──
    let hcAttnBase: Tensor  // f32 [24]
    let hcAttnFn: Tensor  // f16 [hc_dim, 24]  where hc_dim = n_hc * hidden = 4*4096 = 16384
    let hcAttnScale: Tensor  // f32 [3]
    let hcFfnBase: Tensor  // f32 [24]
    let hcFfnFn: Tensor  // f16 [hc_dim, 24]
    let hcFfnScale: Tensor  // f32 [3]

    // ── CSA / HCA compressor (compress_ratio > 0) ──
    let attnCompressorAPE: Tensor?  // f16 [coff * head_dim, ratio]  (ratio=4 for CSA, =128 for HCA)
    let attnCompressorGate: Tensor?  // f16 [hidden, coff * head_dim]
    let attnCompressorKV: Tensor?  // f16 [hidden, coff * head_dim]
    let attnCompressorNorm: Tensor?  // f32 [head_dim]

    // ── CSA-only Lightning Indexer (compress_ratio == 4) ──
    let indexerAttnQB: Tensor?  // f16 [q_lora_rank, indexer_n_heads * indexer_head_size]
    let indexerProj: Tensor?  // f16 [hidden, indexer_n_heads]
    let indexerCompressorAPE: Tensor?  // f16 [coff * indexer_head_size, ratio]
    let indexerCompressorGate: Tensor?  // f16 [hidden, coff * indexer_head_size]
    let indexerCompressorKV: Tensor?  // f16 [hidden, coff * indexer_head_size]
    let indexerCompressorNorm: Tensor?  // f32 [indexer_head_size]

    init(
        layerIndex: Int, compressRatio: Int,
        attnNorm: Tensor, attnQA: Tensor, attnQANorm: Tensor, attnQB: Tensor,
        attnKV: Tensor, attnKVANorm: Tensor, attnSinks: Tensor,
        attnOutputA: Tensor, attnOutputB: Tensor,
        ffnNorm: Tensor, ffnGateInp: Tensor, ffnGateTid2Eid: Tensor?,
        ffnGateExps: Tensor, ffnUpExps: Tensor, ffnDownExps: Tensor,
        ffnGateShexp: Tensor, ffnUpShexp: Tensor, ffnDownShexp: Tensor,
        expProbsBias: Tensor?,
        hcAttnBase: Tensor, hcAttnFn: Tensor, hcAttnScale: Tensor,
        hcFfnBase: Tensor, hcFfnFn: Tensor, hcFfnScale: Tensor,
        attnCompressorAPE: Tensor? = nil, attnCompressorGate: Tensor? = nil,
        attnCompressorKV: Tensor? = nil, attnCompressorNorm: Tensor? = nil,
        indexerAttnQB: Tensor? = nil, indexerProj: Tensor? = nil,
        indexerCompressorAPE: Tensor? = nil, indexerCompressorGate: Tensor? = nil,
        indexerCompressorKV: Tensor? = nil, indexerCompressorNorm: Tensor? = nil
    ) {
        self.layerIndex = layerIndex
        self.compressRatio = compressRatio
        self.attnNorm = attnNorm; self.attnQA = attnQA; self.attnQANorm = attnQANorm
        self.attnQB = attnQB; self.attnKV = attnKV; self.attnKVANorm = attnKVANorm
        self.attnSinks = attnSinks
        self.attnOutputA = attnOutputA; self.attnOutputB = attnOutputB
        self.ffnNorm = ffnNorm; self.ffnGateInp = ffnGateInp; self.ffnGateTid2Eid = ffnGateTid2Eid
        self.ffnGateExps = ffnGateExps; self.ffnUpExps = ffnUpExps; self.ffnDownExps = ffnDownExps
        self.ffnGateShexp = ffnGateShexp; self.ffnUpShexp = ffnUpShexp; self.ffnDownShexp = ffnDownShexp
        self.expProbsBias = expProbsBias
        self.hcAttnBase = hcAttnBase; self.hcAttnFn = hcAttnFn; self.hcAttnScale = hcAttnScale
        self.hcFfnBase = hcFfnBase; self.hcFfnFn = hcFfnFn; self.hcFfnScale = hcFfnScale
        self.attnCompressorAPE = attnCompressorAPE; self.attnCompressorGate = attnCompressorGate
        self.attnCompressorKV = attnCompressorKV; self.attnCompressorNorm = attnCompressorNorm
        self.indexerAttnQB = indexerAttnQB; self.indexerProj = indexerProj
        self.indexerCompressorAPE = indexerCompressorAPE
        self.indexerCompressorGate = indexerCompressorGate
        self.indexerCompressorKV = indexerCompressorKV
        self.indexerCompressorNorm = indexerCompressorNorm
    }
}

/// DSv4 decoder. Holds the shared embedding / output-norm / LM-head /
/// output-mHC weights eagerly + a **lazy per-layer cache**. Layers are
/// loaded on first `layer(_:)` access and can be evicted via
/// `releaseLayer(_:)` so a streaming decoder pipeline keeps only a
/// handful of layers in RAM at a time — the only realistic shape for
/// the 86 GB DSv4-Flash GGUF on Apple Silicon.
public final class DeepSeekV4Model: @unchecked Sendable {
    public let textConfig: DeepSeekV4TextConfig
    public let layerCompressRatios: [Int]  // 0 / 4 / 128 per layer

    // ── Non-block weights, eagerly loaded ──
    public let tokenEmbd: Tensor  // f16 [hidden, vocab]
    public let outputNorm: Tensor  // f32 [hidden]
    public let outputHead: Tensor  // q8_0 [hidden, vocab]
    public let outputHcBase: Tensor  // f32 [n_hc=4]
    public let outputHcFn: Tensor  // f16 [hc_dim, n_hc]
    public let outputHcScale: Tensor  // f32 [1]

    // ── Lazy per-layer cache ──
    let bundle: GGUFTensorBundle
    let device: Device
    let activationDtype: DType
    private var layerCache: [Int: DeepSeekV4Layer] = [:]
    private let cacheLock = NSLock()
    /// Cached `[head_dim]` ones tensor for the per-head unit-RMS Q
    /// norm (no learnable weight, so we pass ones to `rmsNormRows`).
    var qHeadNormOnesCache: Tensor?
    var hcFlatOnesCache: Tensor?
    /// Host-side cache of the i32 token→expert hash-routing tables
    /// (ffn_gate_tid2eid), keyed by layer index. Read once per layer
    /// instead of a 3 MB GPU→CPU copy every token.
    var tid2eidCache: [Int: [Int32]] = [:]
    /// When `true`, layers loaded by `forwardAllLayers` stay resident.
    /// Trades ~8-20 GB RAM for near-zero per-token load cost on token2+.
    public var keepLayersResident: Bool = false
    /// DEBUG stashes (set by the CLI bench's prompt-IDs debug path; read via
    /// optional guards in the forward, no-op when nil).
    public var dbgAnorm: Tensor?  // [nLayers*4] last-token per-layer attn_norm[0..3]
    public var dbgL0: Tensor?  // [8] L0 last-token: hcState[0..3], x(pre-norm)[0..3]

    init(
        textConfig: DeepSeekV4TextConfig, layerCompressRatios: [Int],
        tokenEmbd: Tensor, outputNorm: Tensor, outputHead: Tensor,
        outputHcBase: Tensor, outputHcFn: Tensor, outputHcScale: Tensor,
        bundle: GGUFTensorBundle, device: Device, activationDtype: DType
    ) {
        self.textConfig = textConfig
        self.layerCompressRatios = layerCompressRatios
        self.tokenEmbd = tokenEmbd
        self.outputNorm = outputNorm
        self.outputHead = outputHead
        self.outputHcBase = outputHcBase
        self.outputHcFn = outputHcFn
        self.outputHcScale = outputHcScale
        self.bundle = bundle
        self.device = device
        self.activationDtype = activationDtype
    }

    /// Lazy per-layer loader. First access dequants all tensors for
    /// the layer; subsequent accesses return the cached bundle.
    /// Thread-safe.
    public func layer(_ index: Int) throws -> DeepSeekV4Layer {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let cached = layerCache[index] { return cached }
        let layer = try DeepSeekV4Model.loadLayer(
            index: index,
            compressRatio: layerCompressRatios[index],
            bundle: bundle, device: device, activationDtype: activationDtype,
            textConfig: textConfig)
        layerCache[index] = layer
        return layer
    }

    /// Drop a layer from the cache to free GPU memory. Caller drives
    /// the eviction policy (typically: keep the active layer + the
    /// next one prefetched, drop everything else).
    public func releaseLayer(_ index: Int) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        layerCache.removeValue(forKey: index)
    }

    /// Top-level GGUF → DeepSeekV4Model entry. Eagerly loads the
    /// non-block weights + compress_ratios array; layers stay lazy.
    static func loadFromGGUF(
        textConfig: DeepSeekV4TextConfig, gguf: GGUFTensorBundle,
        device: Device, options: LoadOptions
    ) throws -> DeepSeekV4Model {
        // Activation dtype = f16 to match the GGUF's f16 weights
        // (hc_*_fn, indexer.*, compressor.*, token_embd, ffn_gate_inp).
        // The q8_0 / q2_K / iq2_xxs dequant kernels narrow into f16
        // at the store boundary so the bulk matmul weights end up
        // at the same dtype as the activations.
        // DEFAULT f32: required for correct decode/prefill — under f16 the
        // 43-layer mHC residual accumulates enough error to flip the argmax
        // off the correct token (Tokyo lands #2 by ~0.08 logits). f32 is
        // essentially FREE here: decode is weight-bandwidth-bound (84 GB of
        // quantized weights), so f32 activations measured ~the same tps as
        // f16 (13.8 vs 13.2). Set FFAI_DSV4_ACT_F16=1 to force f16 for perf
        // experiments. (Follow-up: selective f32 — f32 residual only — if a
        // future kernel makes f16 activations cheaper than f32.)
        let activationDtype: DType =
            ProcessInfo.processInfo.environment["FFAI_DSV4_ACT_F16"] == "1" ? .f16 : .f32
        // Extract the compress_ratios array from GGUF metadata. The
        // key matches the canonical DSv4 GGUF naming convention used
        // by the converter. Falls back to inferring from layer-tensor
        // presence (CSA layers have indexer.*, HCA layers don't, full
        // layers have neither).
        let ratios = try resolveCompressRatios(gguf: gguf, nLayers: textConfig.nLayers)

        // ── Eager non-block load ──
        let tokenEmbd = try gguf.tensor(named: "token_embd.weight", outDtype: activationDtype, device: device)
        let outputNorm = try gguf.tensor(named: "output_norm.weight", outDtype: activationDtype, device: device)
        let outputHead = try gguf.tensor(named: "output.weight", outDtype: activationDtype, device: device)
        let outputHcBase = try gguf.tensor(named: "output_hc_base.weight", outDtype: .f32, device: device)
        let outputHcFn = try gguf.tensor(named: "output_hc_fn.weight", outDtype: activationDtype, device: device)
        let outputHcScale = try gguf.tensor(named: "output_hc_scale.weight", outDtype: .f32, device: device)

        return DeepSeekV4Model(
            textConfig: textConfig, layerCompressRatios: ratios,
            tokenEmbd: tokenEmbd, outputNorm: outputNorm, outputHead: outputHead,
            outputHcBase: outputHcBase, outputHcFn: outputHcFn, outputHcScale: outputHcScale,
            bundle: gguf, device: device, activationDtype: activationDtype)
    }

    /// Returns the per-layer `compress_ratios` array. Prefers the
    /// GGUF metadata key; falls back to the structural inference if
    /// the key is absent.
    private static func resolveCompressRatios(
        gguf: GGUFTensorBundle, nLayers: Int
    ) throws -> [Int] {
        // Try direct metadata keys first (different converters use
        // slightly different names).
        let keysToTry = [
            "deepseek4.attention.compress_ratios",
            "deepseek4.compress_ratios",
            "compress_ratios",
        ]
        for key in keysToTry {
            if let arr = gguf.reader.metadataIntArray(key) {
                return arr
            }
        }
        // Fallback: infer from per-layer tensor presence.
        //   CSA → has `blk.N.indexer.attn_q_b.weight`
        //   HCA → has `blk.N.attn_compressor_kv.weight` but no indexer
        //   full → has neither
        var ratios: [Int] = []
        let names = Set(gguf.reader.tensorInfos.map { $0.name })
        for n in 0 ..< nLayers {
            let hasIndexer = names.contains("blk.\(n).indexer.attn_q_b.weight")
            let hasCompressor = names.contains("blk.\(n).attn_compressor_kv.weight")
            let ratio = hasIndexer ? 4 : (hasCompressor ? 128 : 0)
            ratios.append(ratio)
        }
        return ratios
    }

    /// Loads all tensors for one layer. Called by `layer(_:)` on
    /// cache miss. The activation-dtype dequant target is fixed at
    /// `activationDtype` for the bulk weights, with `.f32` for the
    /// per-channel norm + sink + bias scalars (they're tiny and the
    /// downstream kernels expect f32).
    private static func loadLayer(
        index n: Int, compressRatio: Int,
        bundle: GGUFTensorBundle, device: Device, activationDtype dt: DType,
        textConfig: DeepSeekV4TextConfig
    ) throws -> DeepSeekV4Layer {
        let p = "blk.\(n)"
        // Common attention path.
        // RMSNorm weights load at the ACTIVATION dtype so Ops.rmsNorm
        // doesn't fail its `x.dtype == weight.dtype` precondition.
        // The sink / bias / mHC-base / mHC-scale tensors stay f32 —
        // their consumer kernels take f32 inputs regardless of T.
        let attnNorm = try bundle.tensor(named: "\(p).attn_norm.weight", outDtype: dt, device: device)
        let attnQA = try bundle.tensor(named: "\(p).attn_q_a.weight", outDtype: dt, device: device)
        let attnQANorm = try bundle.tensor(named: "\(p).attn_q_a_norm.weight", outDtype: dt, device: device)
        let attnQB = try bundle.tensor(named: "\(p).attn_q_b.weight", outDtype: dt, device: device)
        let attnKV = try bundle.tensor(named: "\(p).attn_kv.weight", outDtype: dt, device: device)
        let attnKVANorm = try bundle.tensor(named: "\(p).attn_kv_a_norm.weight", outDtype: dt, device: device)
        let attnSinks = try bundle.tensor(named: "\(p).attn_sinks.weight", outDtype: .f32, device: device)
        let attnOutputA = try bundle.tensor(named: "\(p).attn_output_a.weight", outDtype: dt, device: device)
        let attnOutputB = try bundle.tensor(named: "\(p).attn_output_b.weight", outDtype: dt, device: device)

        // FFN path.
        let ffnNorm = try bundle.tensor(named: "\(p).ffn_norm.weight", outDtype: dt, device: device)
        let ffnGateInp = try bundle.tensor(named: "\(p).ffn_gate_inp.weight", outDtype: dt, device: device)
        let ffnGateTid2Eid = try? bundle.tensor(named: "\(p).ffn_gate_tid2eid.weight", outDtype: .i32, device: device)
        // MoE expert tensors LOADED LAZILY per top-K routed expert in
        // forwardFfnSubblock via bundle.dequantExpertSlice. Skip eager
        // dequant of all 256 experts/layer (~12 GB wasted). Placeholder
        // [1] tensors keep Layer non-nil; forward never reads them.
        let ffnGateExps = Tensor.filled(0.0, shape: [1], dtype: dt, device: device)
        let ffnUpExps = Tensor.filled(0.0, shape: [1], dtype: dt, device: device)
        let ffnDownExps = Tensor.filled(0.0, shape: [1], dtype: dt, device: device)
        // persistent: these are kept resident on the Layer; without a
        // per-name scratch slot the same-shape gate/up dequants alias to
        // one pooled buffer (gate == up == last-loaded).
        let ffnGateShexp = try bundle.tensor(
            named: "\(p).ffn_gate_shexp.weight", outDtype: dt, device: device, persistent: true)
        let ffnUpShexp = try bundle.tensor(
            named: "\(p).ffn_up_shexp.weight", outDtype: dt, device: device, persistent: true)
        let ffnDownShexp = try bundle.tensor(
            named: "\(p).ffn_down_shexp.weight", outDtype: dt, device: device, persistent: true)
        let expProbsBias = try? bundle.tensor(named: "\(p).exp_probs_b.bias", outDtype: .f32, device: device)

        // mHC weights.
        let hcAttnBase = try bundle.tensor(named: "\(p).hc_attn_base.weight", outDtype: .f32, device: device)
        let hcAttnFn = try bundle.tensor(named: "\(p).hc_attn_fn.weight", outDtype: dt, device: device)
        let hcAttnScale = try bundle.tensor(named: "\(p).hc_attn_scale.weight", outDtype: .f32, device: device)
        let hcFfnBase = try bundle.tensor(named: "\(p).hc_ffn_base.weight", outDtype: .f32, device: device)
        let hcFfnFn = try bundle.tensor(named: "\(p).hc_ffn_fn.weight", outDtype: dt, device: device)
        let hcFfnScale = try bundle.tensor(named: "\(p).hc_ffn_scale.weight", outDtype: .f32, device: device)

        // CSA / HCA compressor.
        var attnCompressorAPE: Tensor?
        var attnCompressorGate: Tensor?
        var attnCompressorKV: Tensor?
        var attnCompressorNorm: Tensor?
        if compressRatio > 0 {
            attnCompressorAPE = try bundle.tensor(
                named: "\(p).attn_compressor_ape.weight", outDtype: dt, device: device)
            attnCompressorGate = try bundle.tensor(
                named: "\(p).attn_compressor_gate.weight", outDtype: dt, device: device)
            attnCompressorKV = try bundle.tensor(named: "\(p).attn_compressor_kv.weight", outDtype: dt, device: device)
            attnCompressorNorm = try bundle.tensor(
                named: "\(p).attn_compressor_norm.weight", outDtype: dt, device: device)
        }

        // CSA-only Lightning Indexer.
        var indexerAttnQB: Tensor?
        var indexerProj: Tensor?
        var indexerCompressorAPE: Tensor?
        var indexerCompressorGate: Tensor?
        var indexerCompressorKV: Tensor?
        var indexerCompressorNorm: Tensor?
        if compressRatio == 4 {
            indexerAttnQB = try bundle.tensor(named: "\(p).indexer.attn_q_b.weight", outDtype: dt, device: device)
            indexerProj = try bundle.tensor(named: "\(p).indexer.proj.weight", outDtype: dt, device: device)
            indexerCompressorAPE = try bundle.tensor(
                named: "\(p).indexer_compressor_ape.weight", outDtype: dt, device: device)
            indexerCompressorGate = try bundle.tensor(
                named: "\(p).indexer_compressor_gate.weight", outDtype: dt, device: device)
            indexerCompressorKV = try bundle.tensor(
                named: "\(p).indexer_compressor_kv.weight", outDtype: dt, device: device)
            indexerCompressorNorm = try bundle.tensor(
                named: "\(p).indexer_compressor_norm.weight", outDtype: dt, device: device)
        }

        _ = textConfig  // shape-checking against the config is a follow-up

        return DeepSeekV4Layer(
            layerIndex: n, compressRatio: compressRatio,
            attnNorm: attnNorm, attnQA: attnQA, attnQANorm: attnQANorm, attnQB: attnQB,
            attnKV: attnKV, attnKVANorm: attnKVANorm, attnSinks: attnSinks,
            attnOutputA: attnOutputA, attnOutputB: attnOutputB,
            ffnNorm: ffnNorm, ffnGateInp: ffnGateInp, ffnGateTid2Eid: ffnGateTid2Eid,
            ffnGateExps: ffnGateExps, ffnUpExps: ffnUpExps, ffnDownExps: ffnDownExps,
            ffnGateShexp: ffnGateShexp, ffnUpShexp: ffnUpShexp, ffnDownShexp: ffnDownShexp,
            expProbsBias: expProbsBias,
            hcAttnBase: hcAttnBase, hcAttnFn: hcAttnFn, hcAttnScale: hcAttnScale,
            hcFfnBase: hcFfnBase, hcFfnFn: hcFfnFn, hcFfnScale: hcFfnScale,
            attnCompressorAPE: attnCompressorAPE, attnCompressorGate: attnCompressorGate,
            attnCompressorKV: attnCompressorKV, attnCompressorNorm: attnCompressorNorm,
            indexerAttnQB: indexerAttnQB, indexerProj: indexerProj,
            indexerCompressorAPE: indexerCompressorAPE, indexerCompressorGate: indexerCompressorGate,
            indexerCompressorKV: indexerCompressorKV, indexerCompressorNorm: indexerCompressorNorm)
    }
}

// ─── Config shim ─────────────────────────────────────────────────────

/// Returns the `text_config` sub-tree on a multimodal V4 conversion
/// (none ship today, but the slot exists upstream); otherwise the
/// top-level config (text-only checkpoint).
enum DeepSeekV4Config {
    static func textConfig(_ c: ModelConfig) -> ModelConfig {
        c.subConfig("text_config") ?? c
    }
}

// ═══════════════════════════════════════════════════════════════════
// Decode forward path (was DeepSeekV4Forward.swift — consolidated per the
// one-file-per-model-family convention).
// ═══════════════════════════════════════════════════════════════════


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

// ═══════════════════════════════════════════════════════════════════
// Batched prefill path (was DeepSeekV4Prefill.swift — consolidated).
// ═══════════════════════════════════════════════════════════════════


struct PrefillError: Error, CustomStringConvertible {
    let message: String
    var description: String { message }
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
            if let freePct = MemorySnapshot.systemFreePercent() {
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
