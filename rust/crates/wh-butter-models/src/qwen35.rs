// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Qwen3.5 / Qwen3.6-27B (GGUF arch `qwen35`): a 64-layer DENSE (no MoE)
//! hybrid transformer — 48 Gated-DeltaNet (GDN) linear-attention layers +
//! 16 full (GQA) attention layers, repeating `[GDN, GDN, GDN, full]`
//! (`full_attention_interval = 4`, layer `i` is full-attn iff
//! `(i+1) % 4 == 0`, verified against `Libraries/MLXLLM/Models/Qwen35.swift`
//! in mlx-swift-lm and against the GGUF tensor-name census: 16*11 + 48*14 +
//! 3 = 851 tensors, exact match). Hidden 5120, dense SwiGLU FFN 17408,
//! untied lm_head, vocab 248320.
//!
//! Full-attn: GQA 24q/4kv, head_dim 256, partial-rotary (rope over only the
//! first 64 of 256 dims, `rope_yarn_partial` with YaRN disabled), per-head
//! Q/K RMSNorm (weighted), and a **sigmoid** output gate fused into Q's
//! projection (`attn_q` is `[hidden, 2*n_head*head_dim]`: per head,
//! `[q(256); gate(256)]` — NOT swish, see `sigmoid_mul`'s doc; the task
//! brief's "swish output gate" for full-attn was a misreading of
//! `Qwen3NextRMSNormGated`, which only applies to GDN's OWN output gate).
//!
//! GDN: 16 key heads x 128, 48 value heads x 128, conv kernel 4 (causal,
//! depthwise, no bias), single-token recurrent step
//! `state' = g*state + outer((v - g*state@k)*beta, k)`,
//! `y = state'@q` (see `iron_gated_delta_step` in wh-iron-std, which
//! name-checks Qwen3.6 in its own doc — this port is the first caller).
//! Output gate is `rmsNorm(y, ssm_norm_weight) * silu(z)` — gate-AFTER-norm
//! (`Qwen3NextRMSNormGated` convention). This is NOT what the already-wired
//! `ops::gated_group_rmsnorm[_batched]` kernels compute (they do
//! gate-BEFORE-norm: `rmsNorm(y*silu(z))`, the Mamba2/NemotronH/Zamba2
//! convention) — reusing those directly would silently produce wrong
//! output, so this file composes the norm from `ops::rms_norm` (per-head,
//! reused from the same op the full-attn q/k-norm uses) + `ops::silu` +
//! `ops::mul` instead.
//!
//! Weights load straight from GGUF: dequantize each tensor to f32, then
//! requantize the big GEMM weights (qkv/o/ffn projections) into this
//! engine's Q4 format (`wh_butter_ops::quantize_q4`), same as
//! [`crate::laguna`]. GDN's small/precision-sensitive tensors — conv1d,
//! the alpha/beta decay-gate projections, `A_log`, `dt_bias`, `ssm_norm` —
//! stay F32 resident (mirrors mlx-swift-lm's explicit fp32-upcast for
//! `in_proj_a`/`in_proj_b`, done there to dodge a bf16 NaN issue at that
//! small an output width; F32 here sidesteps the same class of problem and
//! those tensors are tiny — 48 or 128 wide — so the bandwidth cost is
//! negligible). Embeddings/norms stay F32; the LM head is Q8 (`Q8Head`,
//! same layout as [`crate::laguna::LagunaModel`]'s).
//!
//! DECODE-ONLY, single-batch (B=1) for this port: every prompt token is
//! walked through [`decode_step`] one at a time (a correctness-first
//! "prefill via repeated decode", not a batched prefill kernel — fine for
//! oracle-length prompts, not the throughput path). MTP, vision, and
//! CUDA-graph capture are explicitly out of scope.

use wh_butter_core::{DType, Device, Error, Result, Tensor};
use wh_butter_loader::gguf::{Gguf, GgufTensor};
use wh_butter_ops as ops;

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32s_to_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
/// The `iron_gemv_q4_coalesced*` family reads per-block scales as f16; see
/// the identical helper + its rationale in `laguna.rs`.
fn f32s_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    fn half_bits(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 112;
        if e <= 0 {
            return sign;
        }
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        let m = (x >> 13) & 0x3ff;
        let round = (x >> 12) & 1;
        let v = ((e as u32) << 10) | m;
        sign | ((v + round) as u16)
    }
    v.iter().flat_map(|&f| half_bits(f).to_le_bytes()).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub hidden: usize,
    pub n_layers: usize,
    pub head_dim: usize,               // full-attn head_dim (256)
    pub n_head: usize,                 // 24
    pub n_kv_heads: usize,             // 4
    pub rope_n_rot: u32,               // 64 (partial rotary out of 256)
    pub rope_theta: f32,               // 1e7
    pub full_attention_interval: usize, // 4
    pub eps: f32,
    pub vocab: usize,
    pub ffn_inter: usize,              // 17408
    // GDN (linear attention)
    pub gdn_num_k_heads: usize,        // 16
    pub gdn_num_v_heads: usize,        // 48
    pub gdn_key_head_dim: usize,       // 128
    pub gdn_value_head_dim: usize,     // 128
    pub gdn_conv_kernel: usize,        // 4
}

impl Qwen35Config {
    /// Layer `i` (0-indexed) is full-attention iff `(i+1) % interval == 0`
    /// — verified against `Qwen35DecoderLayer.isLinear` in mlx-swift-lm
    /// (`isLinear = (layerIdx+1) % fullAttentionInterval != 0`) and against
    /// the GGUF tensor census (layers 3,7,11,...,63 carry `attn_q`/`attn_k`/
    /// `attn_v`/`attn_output`; all others carry `ssm_*`).
    pub fn is_full_layer(&self, i: usize) -> bool {
        (i + 1) % self.full_attention_interval == 0
    }
    pub fn gdn_key_dim(&self) -> usize {
        self.gdn_num_k_heads * self.gdn_key_head_dim
    }
    pub fn gdn_value_dim(&self) -> usize {
        self.gdn_num_v_heads * self.gdn_value_head_dim
    }
    pub fn gdn_conv_dim(&self) -> usize {
        self.gdn_key_dim() * 2 + self.gdn_value_dim()
    }
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    pub fn from_gguf(g: &Gguf) -> Self {
        let u = |k: &str, def: usize| g.meta_u32(k).map(|v| v as usize).unwrap_or(def);
        let f = |k: &str, def: f32| g.meta_f32(k).unwrap_or(def);

        let hidden = u("qwen35.embedding_length", 5120);
        let n_layers = u("qwen35.block_count", 64);
        let n_head = u("qwen35.attention.head_count", 24);
        let n_kv_heads = u("qwen35.attention.head_count_kv", 4);
        let head_dim = u("qwen35.attention.key_length", 256);
        let rope_n_rot = u("qwen35.rope.dimension_count", 64) as u32;
        let rope_theta = f("qwen35.rope.freq_base", 1.0e7);
        let full_attention_interval = u("qwen35.full_attention_interval", 4);
        let eps = f("qwen35.attention.layer_norm_rms_epsilon", 1e-6);
        // No `qwen35.vocab_size` key in the GGUF (verified via gguf-dump) —
        // derive from `token_embd.weight`'s own dims instead (ne=[hidden,
        // vocab], i.e. dims[1] = vocab; see `dims_out_in`'s doc for the
        // fastest-first convention).
        let vocab = g
            .tensor("token_embd.weight")
            .map(|t| t.dims[1] as usize)
            .unwrap_or(248320);
        let ffn_inter = u("qwen35.feed_forward_length", 17408);
        let gdn_num_k_heads = u("qwen35.ssm.group_count", 16);
        let gdn_value_dim_total = u("qwen35.ssm.inner_size", 6144);
        let gdn_num_v_heads = u("qwen35.ssm.time_step_rank", 48);
        let gdn_key_head_dim = u("qwen35.ssm.state_size", 128);
        let gdn_value_head_dim = gdn_value_dim_total / gdn_num_v_heads.max(1);
        let gdn_conv_kernel = u("qwen35.ssm.conv_kernel", 4);

        Qwen35Config {
            hidden,
            n_layers,
            head_dim,
            n_head,
            n_kv_heads,
            rope_n_rot,
            rope_theta,
            full_attention_interval,
            eps,
            vocab,
            ffn_inter,
            gdn_num_k_heads,
            gdn_num_v_heads,
            gdn_key_head_dim,
            gdn_value_head_dim,
            gdn_conv_kernel,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Weight loading primitives
// ─────────────────────────────────────────────────────────────────────────

struct Q4Dense {
    qs: Tensor,
    scales: Tensor,
    m: usize,
    k: usize,
}
impl Q4Dense {
    // NOTE (round-2 perf campaign, 2026-08): `ops::gemv_q4_auto` (2-row-per-
    // warp for narrow m_out) looked like a solid win in isolated per-shape
    // microbench (+57-59% on wk/wv/wo/gdn_wz/gdn_wo) but a full oracle-check
    // run wired to it hit a hard crash on the very first decode_step:
    // `Dispatch("dispatch failed: cuModuleLoadData: an illegal memory access
    // was encountered")` — a CUDA context-corrupting fault (surfaced late,
    // at the next module load, per the usual async-error pattern), not
    // reproduced by any single isolated shape in the microbench harness.
    // Root cause NOT isolated within this campaign's time budget. DO NOT
    // wire `gemv_q4_auto` here (or anywhere in the real decode path) until
    // that's root-caused — stays on the safe, fully-plain `gemv_q4` path.
    fn gemv(&self, dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
        ops::gemv_q4(dev, &self.qs, &self.scales, x, self.m, self.k, self.m)
    }
}

/// GGUF dims are fastest-first: a 2-D `[out, in]` weight is stored with
/// `dims = [in, out]`, which is exactly the `(m=out, k=in)` `gemv_q4`/
/// `ops::gemv` convention with zero transposition (same convention
/// `laguna.rs::dims_out_in` documents and relies on).
fn dims_out_in(t: &GgufTensor, name: &str) -> Result<(usize, usize)> {
    if t.dims.len() != 2 {
        return Err(Error::Msg(format!("{name}: expected 2-D tensor, got dims {:?}", t.dims)));
    }
    Ok((t.dims[1] as usize, t.dims[0] as usize))
}

fn load_q4dense(dev: &dyn Device, g: &Gguf, name: &str) -> Result<Q4Dense> {
    let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("qwen35: missing tensor '{name}'")))?;
    let (m, k) = dims_out_in(t, name)?;
    let flat = g.dequant_f32(name)?;
    let (qs, scales_f32) = ops::quantize_q4(&flat, m, k);
    let scales_bytes = f32s_to_f16_bytes(&scales_f32);
    Ok(Q4Dense {
        qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs))?, vec![qs.len()], DType::U32),
        scales: Tensor::new(dev.upload(&scales_bytes)?, vec![scales_bytes.len() / 2], DType::F16),
        m,
        k,
    })
}

/// Plain F32 dense `[m, k]` weight for `ops::gemv` — used for GDN's tiny
/// (`Hv`-wide) alpha/beta decay-gate projections, kept unquantized (see the
/// module doc's numerical-safety rationale).
fn load_f32_dense(dev: &dyn Device, g: &Gguf, name: &str) -> Result<Tensor> {
    let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("qwen35: missing tensor '{name}'")))?;
    let (m, k) = dims_out_in(t, name)?;
    let flat = g.dequant_f32(name)?;
    Ok(Tensor::new(dev.upload(&f32s_to_bytes(&flat))?, vec![m, k], DType::F32))
}

fn load_f32_vec(dev: &dyn Device, g: &Gguf, name: &str) -> Result<Tensor> {
    let flat = g.dequant_f32(name)?;
    let n = flat.len();
    Ok(Tensor::new(dev.upload(&f32s_to_bytes(&flat))?, vec![n], DType::F32))
}

/// `ssm_conv1d.weight`'s GGUF-native flat layout is CHANNEL-major /
/// kernel-minor (`ne = [conv_kernel, conv_dim]`, fastest-first, so
/// `flat[d*conv_kernel + k]`), but `ops::conv1d_causal_silu_many` (mirroring
/// `iron_conv1d_causal_step_silu_cast_many`) wants KERNEL-major /
/// channel-minor (`w[k*conv_dim + d]`, one contiguous `[conv_dim]` row per
/// tap). Transpose at load time.
fn repack_conv1d_weight(flat: &[f32], conv_dim: usize, conv_kernel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; flat.len()];
    for d in 0..conv_dim {
        for k in 0..conv_kernel {
            out[k * conv_dim + d] = flat[d * conv_kernel + k];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────
// Per-layer weights
// ─────────────────────────────────────────────────────────────────────────

/// Full (GQA) attention layer weights. `wq` is `[hidden, 2*n_head*head_dim]`
/// — Q and its sigmoid output-gate fused per head (see the module doc);
/// split at decode time with `ops::strided_head_half`.
struct FullAttnWeights {
    attn_norm: Tensor, // f32 [hidden]
    wq: Q4Dense,       // [n_head*2*head_dim, hidden]
    wk: Q4Dense,       // [n_kv_heads*head_dim, hidden]
    wv: Q4Dense,       // [n_kv_heads*head_dim, hidden]
    wo: Q4Dense,       // [hidden, n_head*head_dim]
    q_norm: Tensor,    // f32 [head_dim] (weighted RMSNorm, full head_dim)
    k_norm: Tensor,    // f32 [head_dim]
}

/// GDN (linear attention) layer weights.
struct GdnWeights {
    attn_norm: Tensor,     // f32 [hidden]
    wqkv: Q4Dense,         // [key_dim*2+value_dim, hidden]  ("attn_qkv")
    wz: Q4Dense,           // [value_dim, hidden]             ("attn_gate" = in_proj_z)
    w_alpha: Tensor,       // f32 [Hv, hidden]                ("ssm_alpha" = in_proj_a)
    w_beta: Tensor,        // f32 [Hv, hidden]                ("ssm_beta"  = in_proj_b)
    dt_bias: Tensor,       // f32 [Hv]
    neg_exp_a_log: Tensor, // f32 [Hv]  == GGUF's ssm_a directly (already -exp(A_log), see load_gguf)
    conv_w: Tensor,        // f32 [conv_kernel*conv_dim], kernel-major (repacked)
    ssm_norm_w: Tensor,    // f32 [Dv]  gated-RMSNorm weight
    wo: Q4Dense,           // [hidden, value_dim]             ("ssm_out")
}

enum Mixer {
    Full(FullAttnWeights),
    Gdn(GdnWeights),
}

struct DenseFfn {
    ffn_norm: Tensor, // f32 [hidden]
    gate: Q4Dense,    // [ffn_inter, hidden]
    up: Q4Dense,      // [ffn_inter, hidden]
    down: Q4Dense,    // [hidden, ffn_inter]
}

struct Qwen35Layer {
    mixer: Mixer,
    ffn: DenseFfn,
}

/// LM head resident as Q8 — same layout `ops::gemv_q8` consumes as
/// [`crate::laguna::LagunaModel`]'s `Q8Head`.
struct Q8Head {
    qs: Tensor,
    scales: Tensor,
    m: usize,
    k: usize,
}

/// Perf-campaign lever (decode-perf-campaign, 2026-08): vocab 248320 x
/// hidden 5120 makes the untied lm_head a monster single-token GEMV
/// (~7.7% of total decode-step GPU time at Q8, measured via nsys on this
/// port — `iron_gemv_q8_coalesced` avg ~6.5ms/step vs `iron_gemv_q4_coalesced`
/// dominating at ~85%). Q4 roughly halves the bytes streamed for this one
/// GEMV. Kept SWITCHABLE (env `QWEN35_LM_HEAD_Q4=1`, default remains Q8)
/// rather than an outright replacement: Laguna deliberately keeps its lm_head
/// at Q8 for output-quality reasons (see `laguna.rs`), and this port's oracle
/// gate is sensitive to any argmax perturbation — Q4 must be measured for
/// speedup AND re-gated against the oracle before it can become the default.
enum LmHead {
    Q8(Q8Head),
    Q4(Q4Dense),
}
impl LmHead {
    fn gemv(&self, dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
        match self {
            LmHead::Q8(h) => ops::gemv_q8(dev, &h.qs, &h.scales, x, h.m, h.k, h.m),
            LmHead::Q4(h) => h.gemv(dev, x),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// MTP (multi-token-prediction) sidecar -- self-speculative decode drafter
// ─────────────────────────────────────────────────────────────────────────
//
// F-85-style speculative decode: `qwen35.nextn_predict_layers` (GGUF meta)
// = 1, i.e. exactly ONE extra decoder-layer-shaped block appended after the
// 64 trunk layers (`blk.64.*` + `blk.64.nextn.*`), consuming ~1.6% of the
// model's own weight bytes per draft step (verified against the GGUF tensor
// census in `load_mtp`'s doc). Architecture confirmed against the llama.cpp
// fork's `models/qwen35.cpp` `graph_mtp()` (the only available from-source
// reference for this arch's MTP -- mlx-swift-lm's `Qwen35.swift` explicitly
// STRIPS `mtp.*` weights rather than running them, see its
// `hasMTPWeights`/`weights.filter` comment, so it was not usable as the
// executable reference here): DeepSeek-style NextN -- concat the seed
// token's embedding with a "final normed hidden" from the previous stage,
// project down through `eh_proj`, run one full-attention decoder layer
// (own KV cache, own position axis matching the backbone's), then share
// the trunk's own final-norm-shaped head. See `MtpWeights`'s doc for the
// exact per-tensor contract and `mtp_draft_step`'s doc for the forward.
struct MtpWeights {
    // `eh_proj`'s GGUF shape is `[hidden, 2*hidden]` computing
    // `eh_proj · [e_norm; h_norm]` (concat on the input axis). This engine
    // has no device `concat` op, so the weight is split at LOAD time into
    // two `[hidden,hidden]` halves and the concat-then-project is computed
    // as `eh_proj_e·e_norm + eh_proj_h·h_norm` instead -- algebraically
    // identical (block matrix times block vector), reuses existing
    // `gemv`+`add` rather than adding a new device op.
    eh_proj_e: Q4Dense, // [hidden, hidden] (embedding half)
    eh_proj_h: Q4Dense, // [hidden, hidden] (hidden half)
    enorm: Tensor,      // f32 [hidden]  ("blk.N.nextn.enorm")
    hnorm: Tensor,      // f32 [hidden]  ("blk.N.nextn.hnorm")
    attn_norm: Tensor,  // f32 [hidden]  ("blk.N.attn_norm")
    wq: Q4Dense,        // [n_head*2*head_dim, hidden]  ([q;gate] fused, same as FullAttnWeights)
    wk: Q4Dense,        // [n_kv_heads*head_dim, hidden]
    wv: Q4Dense,        // [n_kv_heads*head_dim, hidden]
    wo: Q4Dense,        // [hidden, n_head*head_dim]
    q_norm: Tensor,     // f32 [head_dim]
    k_norm: Tensor,     // f32 [head_dim]
    ffn_norm: Tensor,   // f32 [hidden]  ("blk.N.post_attention_norm")
    ffn_gate: Q4Dense,
    ffn_up: Q4Dense,
    ffn_down: Q4Dense,
    // Shared head: the GGUF carries `blk.N.nextn.shared_head_norm` (a
    // DEDICATED final-norm weight for the MTP head, distinct from the
    // backbone's own `output_norm`) but NO `nextn.shared_head_head` and NO
    // `nextn.embed_tokens` -- confirmed absent via `gguf-dump` (only
    // eh_proj/enorm/hnorm/shared_head_norm present under `blk.64.nextn.*`),
    // consistent with the checkpoint's "shared embeddings" convention: the
    // draft head reuses `model.tok_embd` (already true for any lookup) and
    // `model.lm_head` (the trunk's own LM head) directly, just with its OWN
    // pre-head norm.
    shared_head_norm: Tensor, // f32 [hidden]
}

/// Splits `eh_proj`'s `[hidden, 2*hidden]` GGUF weight (GGUF dims
/// `[2*hidden, hidden]`, i.e. `dims_out_in` gives `m=hidden` (out),
/// `k=2*hidden` (in)) into two independently-quantized `[hidden,hidden]`
/// `Q4Dense` halves -- see `MtpWeights::eh_proj_e`/`eh_proj_h`'s doc for why.
/// Row-major flat layout from `dequant_f32`: `flat[o*k + i]`, `i < hidden`
/// is the embedding-half column range (matches `ggml_concat(e_norm, h_norm,
/// dim=0)`'s e_norm-first order in the reference), `i >= hidden` is the
/// hidden-half range.
fn load_q4dense_eh_proj_split(dev: &dyn Device, g: &Gguf, name: &str, hidden: usize) -> Result<(Q4Dense, Q4Dense)> {
    let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("qwen35: missing tensor '{name}'")))?;
    let (m, k) = dims_out_in(t, name)?;
    if m != hidden || k != 2 * hidden {
        return Err(Error::Msg(format!(
            "{name}: expected eh_proj shape out={hidden} in={} (2*hidden), got out={m} in={k}",
            2 * hidden
        )));
    }
    let flat = g.dequant_f32(name)?;
    let mut we = vec![0f32; m * hidden];
    let mut wh = vec![0f32; m * hidden];
    for o in 0..m {
        we[o * hidden..(o + 1) * hidden].copy_from_slice(&flat[o * k..o * k + hidden]);
        wh[o * hidden..(o + 1) * hidden].copy_from_slice(&flat[o * k + hidden..o * k + 2 * hidden]);
    }
    let mk = |flat: Vec<f32>| -> Result<Q4Dense> {
        let (qs, scales_f32) = ops::quantize_q4(&flat, hidden, hidden);
        let scales_bytes = f32s_to_f16_bytes(&scales_f32);
        Ok(Q4Dense {
            qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs))?, vec![qs.len()], DType::U32),
            scales: Tensor::new(dev.upload(&scales_bytes)?, vec![scales_bytes.len() / 2], DType::F16),
            m: hidden,
            k: hidden,
        })
    };
    Ok((mk(we)?, mk(wh)?))
}

/// Loads the MTP sidecar if present. `qwen35.block_count` (`cfg.n_layers`,
/// 64) is the trunk-layer count; `qwen35.nextn_predict_layers` (GGUF meta,
/// verified = 1 on the MTP-preserved checkpoint) says the MTP block is
/// stored as ONE extra "layer" immediately after the trunk, i.e.
/// `blk.<n_layers>.*` (verified via `gguf-dump`: `blk.64.attn_norm.weight`,
/// `blk.64.attn_q.weight` [5120,12288] Q4_K, `blk.64.attn_k.weight`
/// [5120,1024] Q4_K, `blk.64.attn_v.weight` [5120,1024] Q6_K,
/// `blk.64.attn_output.weight` [6144,5120] Q4_K, `blk.64.attn_q_norm`/
/// `attn_k_norm.weight` [256], `blk.64.post_attention_norm.weight` [5120],
/// `blk.64.ffn_{gate,up}.weight` [5120,17408] Q4_K, `blk.64.ffn_down.weight`
/// [17408,5120] Q6_K, `blk.64.nextn.eh_proj.weight` [5120,10240] Q8_0,
/// `blk.64.nextn.{enorm,hnorm,shared_head_norm}.weight` [5120] F32 --
/// `nextn.embed_tokens`/`nextn.shared_head_head` confirmed ABSENT). Returns
/// `Ok(None)` (not an error) for a non-MTP checkpoint so the rest of the
/// loader/decode path is unaffected.
fn load_mtp(dev: &dyn Device, g: &Gguf, cfg: &Qwen35Config) -> Result<Option<MtpWeights>> {
    let p = format!("blk.{}", cfg.n_layers);
    if g.tensor(&format!("{p}.nextn.eh_proj.weight")).is_none() {
        return Ok(None);
    }
    let (eh_proj_e, eh_proj_h) =
        load_q4dense_eh_proj_split(dev, g, &format!("{p}.nextn.eh_proj.weight"), cfg.hidden)?;
    Ok(Some(MtpWeights {
        eh_proj_e,
        eh_proj_h,
        enorm: load_f32_vec(dev, g, &format!("{p}.nextn.enorm.weight"))?,
        hnorm: load_f32_vec(dev, g, &format!("{p}.nextn.hnorm.weight"))?,
        attn_norm: load_f32_vec(dev, g, &format!("{p}.attn_norm.weight"))?,
        wq: load_q4dense(dev, g, &format!("{p}.attn_q.weight"))?,
        wk: load_q4dense(dev, g, &format!("{p}.attn_k.weight"))?,
        wv: load_q4dense(dev, g, &format!("{p}.attn_v.weight"))?,
        wo: load_q4dense(dev, g, &format!("{p}.attn_output.weight"))?,
        q_norm: load_f32_vec(dev, g, &format!("{p}.attn_q_norm.weight"))?,
        k_norm: load_f32_vec(dev, g, &format!("{p}.attn_k_norm.weight"))?,
        ffn_norm: load_f32_vec(dev, g, &format!("{p}.post_attention_norm.weight"))?,
        ffn_gate: load_q4dense(dev, g, &format!("{p}.ffn_gate.weight"))?,
        ffn_up: load_q4dense(dev, g, &format!("{p}.ffn_up.weight"))?,
        ffn_down: load_q4dense(dev, g, &format!("{p}.ffn_down.weight"))?,
        shared_head_norm: load_f32_vec(dev, g, &format!("{p}.nextn.shared_head_norm.weight"))?,
    }))
}

pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    tok_embd: Tensor, // f32 [vocab, hidden]
    layers: Vec<Qwen35Layer>,
    out_norm: Tensor, // f32 [hidden]
    lm_head: LmHead,
    mtp: Option<MtpWeights>,
    /// Constant (not loaded) per-head Q/K rescale buffers for GDN, shared
    /// across all GDN layers: Qwen3.5's GDN normalizes q/k with a WEIGHTLESS
    /// RMSNorm scaled by `invScale^2` (q) / `invScale` (k), `invScale =
    /// 1/sqrt(key_head_dim)` (see `Qwen35GatedDeltaNet.callAsFunction` in
    /// mlx-swift-lm). `ops::rms_norm` takes a per-element weight, so filling
    /// it with a constant reproduces the weightless-scaled form exactly.
    gdn_q_scale: Tensor, // f32 [gdn_key_head_dim], all invScale^2
    gdn_k_scale: Tensor, // f32 [gdn_key_head_dim], all invScale
    /// Zero bias for `ops::conv1d_causal_silu_many` — Qwen3.5's GDN conv1d
    /// has `bias: false`.
    gdn_conv_zero_bias: Tensor, // f32 [conv_dim]
    /// GQA head-expansion index for GDN's q/k: `iron_gated_delta_step`'s
    /// OWN internal GQA decomposition is `hk_idx = hv_idx / (Hv/Hk)`
    /// (contiguous blocks), but the actual reference kernel this checkpoint
    /// was validated against (llama.cpp's `gated_delta_net.cu`,
    /// `iq1 = fastmodulo(h_idx, neqk1)` i.e. `hv_idx % Hk`) uses the
    /// CYCLIC/tiled mapping instead — verified by oracle checksum diff
    /// (attn_output-0 aggregate matched to 0.01% only after switching to
    /// cyclic; the block mapping was off by ~20%). Rather than patch the
    /// pinned iron kernel, pre-expand q/k from `[Hk,Dk]` to `[Hv,Dk]` via
    /// `ops::gather` with this constant index buffer (`idx[hv] = hv % Hk`),
    /// then call `gated_delta_step` with `hk=Hv` so its internal grouping
    /// becomes a no-op identity map.
    gdn_gqa_cyclic_idx: Tensor, // u32 [gdn_num_v_heads], idx[hv] = hv % gdn_num_k_heads
}

// ─────────────────────────────────────────────────────────────────────────
// Loading
// ─────────────────────────────────────────────────────────────────────────

pub fn load_gguf(dev: &dyn Device, g: &Gguf) -> Result<Qwen35Model> {
    let load_t0 = std::time::Instant::now();
    let cfg = Qwen35Config::from_gguf(g);

    let tok_embd = {
        let flat = g.dequant_f32("token_embd.weight")?;
        Tensor::new(dev.upload(&f32s_to_bytes(&flat))?, vec![cfg.vocab, cfg.hidden], DType::F32)
    };
    let out_norm = load_f32_vec(dev, g, "output_norm.weight")?;
    let head_name = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    // QWEN35_LM_HEAD_Q4=1: perf-campaign A/B lever, see `LmHead`'s doc.
    // Default (unset/0) keeps the Laguna-established Q8 lm_head.
    let lm_head_q4 = std::env::var("QWEN35_LM_HEAD_Q4").map(|v| v == "1").unwrap_or(false);
    let lm_head = if lm_head_q4 {
        LmHead::Q4(load_q4dense(dev, g, head_name)?)
    } else {
        let (qs, scales, m, k) = g.q8_repack(head_name)?;
        LmHead::Q8(Q8Head {
            qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs))?, vec![qs.len()], DType::U32),
            scales: Tensor::new(dev.upload(&f32s_to_bytes(&scales))?, vec![scales.len()], DType::F32),
            m,
            k,
        })
    };

    let inv_scale = 1.0 / (cfg.gdn_key_head_dim as f32).sqrt();
    let gdn_q_scale = {
        let v = vec![inv_scale * inv_scale; cfg.gdn_key_head_dim];
        Tensor::new(dev.upload(&f32s_to_bytes(&v))?, vec![cfg.gdn_key_head_dim], DType::F32)
    };
    let gdn_k_scale = {
        let v = vec![inv_scale; cfg.gdn_key_head_dim];
        Tensor::new(dev.upload(&f32s_to_bytes(&v))?, vec![cfg.gdn_key_head_dim], DType::F32)
    };
    let gdn_conv_zero_bias = {
        let v = vec![0.0f32; cfg.gdn_conv_dim()];
        Tensor::new(dev.upload(&f32s_to_bytes(&v))?, vec![cfg.gdn_conv_dim()], DType::F32)
    };
    let gdn_gqa_cyclic_idx = {
        let idx: Vec<u32> = (0..cfg.gdn_num_v_heads as u32).map(|hv| hv % cfg.gdn_num_k_heads as u32).collect();
        Tensor::new(dev.upload(&u32s_to_bytes(&idx))?, vec![cfg.gdn_num_v_heads], DType::U32)
    };

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("blk.{i}");
        let mixer = if cfg.is_full_layer(i) {
            Mixer::Full(FullAttnWeights {
                attn_norm: load_f32_vec(dev, g, &format!("{p}.attn_norm.weight"))?,
                wq: load_q4dense(dev, g, &format!("{p}.attn_q.weight"))?,
                wk: load_q4dense(dev, g, &format!("{p}.attn_k.weight"))?,
                wv: load_q4dense(dev, g, &format!("{p}.attn_v.weight"))?,
                wo: load_q4dense(dev, g, &format!("{p}.attn_output.weight"))?,
                q_norm: load_f32_vec(dev, g, &format!("{p}.attn_q_norm.weight"))?,
                k_norm: load_f32_vec(dev, g, &format!("{p}.attn_k_norm.weight"))?,
            })
        } else {
            // NOTE: despite the GGUF tensor name mirroring HF's raw "A_log"
            // parameter, llama.cpp's GGUF conversion for this arch BAKES IN
            // the `-exp(A_log)` transform at conversion time — verified
            // against the reference ggml graph (`gate-0 = MUL(a_softplus-0,
            // blk.0.ssm_a)`, a plain elementwise multiply with NO exp/negate
            // node applied to `ssm_a`) and against the tensor's own raw
            // values (small negatives, O(1e-2), consistent with an
            // already-final decay coefficient — re-applying `-exp()` to
            // those pushes everything toward -1, which was the bug: it
            // over-decayed the GDN state on every step, killing context
            // retention). Use the tensor directly, no further transform.
            let neg_exp_a_log = g.dequant_f32(&format!("{p}.ssm_a"))?;
            let conv_flat = g.dequant_f32(&format!("{p}.ssm_conv1d.weight"))?;
            let conv_repacked = repack_conv1d_weight(&conv_flat, cfg.gdn_conv_dim(), cfg.gdn_conv_kernel);
            Mixer::Gdn(GdnWeights {
                attn_norm: load_f32_vec(dev, g, &format!("{p}.attn_norm.weight"))?,
                wqkv: load_q4dense(dev, g, &format!("{p}.attn_qkv.weight"))?,
                wz: load_q4dense(dev, g, &format!("{p}.attn_gate.weight"))?,
                w_alpha: load_f32_dense(dev, g, &format!("{p}.ssm_alpha.weight"))?,
                w_beta: load_f32_dense(dev, g, &format!("{p}.ssm_beta.weight"))?,
                dt_bias: load_f32_vec(dev, g, &format!("{p}.ssm_dt.bias"))?,
                neg_exp_a_log: Tensor::new(
                    dev.upload(&f32s_to_bytes(&neg_exp_a_log))?,
                    vec![neg_exp_a_log.len()],
                    DType::F32,
                ),
                conv_w: Tensor::new(
                    dev.upload(&f32s_to_bytes(&conv_repacked))?,
                    vec![conv_repacked.len()],
                    DType::F32,
                ),
                ssm_norm_w: load_f32_vec(dev, g, &format!("{p}.ssm_norm.weight"))?,
                wo: load_q4dense(dev, g, &format!("{p}.ssm_out.weight"))?,
            })
        };
        let ffn = DenseFfn {
            ffn_norm: load_f32_vec(dev, g, &format!("{p}.post_attention_norm.weight"))?,
            gate: load_q4dense(dev, g, &format!("{p}.ffn_gate.weight"))?,
            up: load_q4dense(dev, g, &format!("{p}.ffn_up.weight"))?,
            down: load_q4dense(dev, g, &format!("{p}.ffn_down.weight"))?,
        };
        layers.push(Qwen35Layer { mixer, ffn });
    }

    let mtp = load_mtp(dev, g, &cfg)?;
    eprintln!(
        "Qwen35: load done in {:.2}s ({} layers, mtp={})",
        load_t0.elapsed().as_secs_f64(),
        cfg.n_layers,
        mtp.is_some()
    );

    Ok(Qwen35Model {
        cfg,
        tok_embd,
        layers,
        out_norm,
        lm_head,
        mtp,
        gdn_q_scale,
        gdn_k_scale,
        gdn_conv_zero_bias,
        gdn_gqa_cyclic_idx,
    })
}

// ─────────────────────────────────────────────────────────────────────────
// KV / recurrent-state cache
// ─────────────────────────────────────────────────────────────────────────

struct FullAttnSlot {
    k: Tensor, // f32 [n_kv_heads, cap, head_dim]
    v: Tensor,
    cap: usize,
}

struct GdnSlot {
    state: Tensor,      // f32 [Hv, Dv, Dk], read+written in place
    conv_state: Tensor, // f32 [conv_kernel-1, conv_dim], read+written in place
}

enum LayerSlot {
    Full(FullAttnSlot),
    Gdn(GdnSlot),
}

pub struct Qwen35Cache {
    slots: Vec<LayerSlot>,
}

impl Qwen35Cache {
    pub fn new(dev: &dyn Device, cfg: &Qwen35Config, max_seq: usize) -> Result<Self> {
        let mut slots = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            if cfg.is_full_layer(i) {
                let n = cfg.n_kv_heads * max_seq * cfg.head_dim;
                let k = Tensor::new(
                    dev.alloc_zeroed(n * 4)?,
                    vec![cfg.n_kv_heads, max_seq, cfg.head_dim],
                    DType::F32,
                );
                let v = Tensor::new(
                    dev.alloc_zeroed(n * 4)?,
                    vec![cfg.n_kv_heads, max_seq, cfg.head_dim],
                    DType::F32,
                );
                slots.push(LayerSlot::Full(FullAttnSlot { k, v, cap: max_seq }));
            } else {
                let state_n = cfg.gdn_num_v_heads * cfg.gdn_value_head_dim * cfg.gdn_key_head_dim;
                let state = Tensor::new(
                    dev.alloc_zeroed(state_n * 4)?,
                    vec![cfg.gdn_num_v_heads, cfg.gdn_value_head_dim, cfg.gdn_key_head_dim],
                    DType::F32,
                );
                let conv_n = (cfg.gdn_conv_kernel - 1) * cfg.gdn_conv_dim();
                let conv_state =
                    Tensor::new(dev.alloc_zeroed(conv_n * 4)?, vec![cfg.gdn_conv_kernel - 1, cfg.gdn_conv_dim()], DType::F32);
                slots.push(LayerSlot::Gdn(GdnSlot { state, conv_state }));
            }
        }
        Ok(Qwen35Cache { slots })
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Decode
// ─────────────────────────────────────────────────────────────────────────

/// QWEN35_DEBUG=1: per-token, layer-0-only checksum dump for oracle diffing
/// against `llama-eval-callback`'s per-node `sum = ...` output. Mirrors
/// `laguna.rs`'s `BUTTER_LAGUNA_DEBUG`/`dbg_stats` pattern. TEMPORARY —
/// gated dead weight when unset (one env read per call site).
fn dbg_sum(dev: &dyn Device, tag: &str, pos: u32, t: &Tensor) {
    let n = t.elem_count();
    let mut b = vec![0u8; n * 4];
    if dev.synchronize().is_err() || dev.download(t.buffer.as_ref(), &mut b).is_err() {
        eprintln!("QWEN35_DBG pos={pos} tag={tag}: <download failed>");
        return;
    }
    let f: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let sum: f64 = f.iter().map(|&v| v as f64).sum();
    eprintln!("QWEN35_DBG pos={pos} tag={tag} n={n} sum={sum:.6} head={:?}", &f[..4.min(n)]);
}

fn full_attn_layer(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    w: &FullAttnWeights,
    slot: &mut FullAttnSlot,
    x: &Tensor,
    pos: u32,
) -> Result<Tensor> {
    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;

    let qg = w.wq.gemv(dev, &h)?; // [n_head*2*head_dim]
    let k = w.wk.gemv(dev, &h)?; // [n_kv_heads*head_dim]
    let v = w.wv.gemv(dev, &h)?; // [n_kv_heads*head_dim]

    // Per-head [q(256); gate(256)] de-interleave (see `strided_head_half`'s doc).
    let q = ops::strided_head_half(dev, &qg, cfg.n_head, cfg.head_dim, 0)?;
    let gate = ops::strided_head_half(dev, &qg, cfg.n_head, cfg.head_dim, 1)?;

    let q = q.reshaped(vec![cfg.n_head, cfg.head_dim]);
    let k = k.reshaped(vec![cfg.n_kv_heads, cfg.head_dim]);
    let v = v.reshaped(vec![cfg.n_kv_heads, cfg.head_dim]);

    // Weighted per-head QK RMSNorm (full head_dim), before RoPE.
    let q = ops::rms_norm(dev, &q, &w.q_norm, cfg.eps)?;
    let k = ops::rms_norm(dev, &k, &w.k_norm, cfg.eps)?;

    // Partial-rotary RoPE: rotate only the first `rope_n_rot` (64) of 256
    // dims, plain (YaRN disabled: freq_scale=1, ext_factor=0, attn_factor=1).
    let q = ops::rope_yarn_partial(dev, &q, pos, cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144)?;
    let k = ops::rope_yarn_partial(dev, &k, pos, cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144)?;

    let posbuf = Tensor::new(dev.upload(&pos.to_le_bytes())?, vec![1], DType::U32);
    ops::kv_append(dev, &k, &slot.k, &posbuf, cfg.head_dim, slot.cap, cfg.n_kv_heads * cfg.head_dim)?;
    ops::kv_append(dev, &v, &slot.v, &posbuf, cfg.head_dim, slot.cap, cfg.n_kv_heads * cfg.head_dim)?;

    let heads_per_group = (cfg.n_head / cfg.n_kv_heads) as u32;
    let n_kv_used = pos + 1;
    let attn = ops::sdpa_decode(
        dev, &q, &slot.k, &slot.v, cfg.head_dim, n_kv_used, slot.cap as u32, heads_per_group, cfg.attn_scale(),
    )?; // [n_head, head_dim]

    // Sigmoid output gate (NOT swish — see module doc).
    let attn_flat = attn.reshaped(vec![cfg.n_head * cfg.head_dim]);
    let gated = ops::sigmoid_mul(dev, &attn_flat, &gate)?;

    let o = w.wo.gemv(dev, &gated)?;
    ops::add(dev, x, &o)
}

/// Embed → 64 layers (GDN or full-attn per `cfg.is_full_layer`) → final
/// RMSNorm → Q8 lm_head → next-token logits `[vocab]`. `pos` is this
/// token's absolute sequence position (0-based); call once per token,
/// reusing `cache` so KV/recurrent state persists across calls (prompt
/// tokens included — see the module doc's "decode-only" note).
pub fn decode_step(dev: &dyn Device, model: &Qwen35Model, cache: &mut Qwen35Cache, token_id: u32, pos: u32) -> Result<Tensor> {
    Ok(decode_step_ex(dev, model, cache, token_id, pos)?.0)
}

/// Same as [`decode_step`] but also returns `xn` — the post-`output_norm`
/// hidden state that fed the lm_head (i.e. `xn` such that
/// `lm_head.gemv(xn) == logits`). Added for MTP self-speculative decode
/// (see `mtp_draft_step`/`speculative_round`): the MTP drafter's seed input
/// is exactly this tensor, and re-deriving it by re-running the trunk would
/// double the backbone cost. `decode_step` is unchanged (still the plain,
/// single-return entry point every existing caller uses) — it now just
/// forwards to this and drops `xn`, so this refactor is behavior-preserving
/// for all pre-existing call sites.
pub fn decode_step_ex(
    dev: &dyn Device, model: &Qwen35Model, cache: &mut Qwen35Cache, token_id: u32, pos: u32,
) -> Result<(Tensor, Tensor)> {
    let cfg = &model.cfg;
    let debug = std::env::var("QWEN35_DEBUG").map(|v| v == "1").unwrap_or(false);
    let tok_buf = Tensor::new(dev.upload(&token_id.to_le_bytes())?, vec![1], DType::U32);
    let mut x = ops::gather(dev, &model.tok_embd, &tok_buf)?.reshaped(vec![cfg.hidden]);
    if debug { dbg_sum(dev, "x_embed", pos, &x); }

    for (i, layer) in model.layers.iter().enumerate() {
        x = match (&layer.mixer, &mut cache.slots[i]) {
            (Mixer::Full(w), LayerSlot::Full(slot)) => full_attn_layer(dev, cfg, w, slot, &x, pos)?,
            (Mixer::Gdn(w), LayerSlot::Gdn(slot)) => gdn_layer_full(dev, cfg, model, w, slot, &x, i, pos, debug)?,
            _ => return Err(Error::Msg(format!("qwen35: layer {i} mixer/slot kind mismatch"))),
        };
        if debug && i == 0 { dbg_sum(dev, "attn_residual-0", pos, &x); }
        let h2 = ops::rms_norm(dev, &x, &layer.ffn.ffn_norm, cfg.eps)?;
        let gate = layer.ffn.gate.gemv(dev, &h2)?;
        let up = layer.ffn.up.gemv(dev, &h2)?;
        let act = ops::swiglu(dev, &gate, &up)?;
        let ffn_out = layer.ffn.down.gemv(dev, &act)?;
        x = ops::add(dev, &x, &ffn_out)?;
        if debug && i == 0 { dbg_sum(dev, "l_out-0", pos, &x); }
    }

    let xn = ops::rms_norm(dev, &x, &model.out_norm, cfg.eps)?;
    let logits = model.lm_head.gemv(dev, &xn)?;
    Ok((logits, xn))
}

/// One GDN (linear-attention) layer's forward. Takes `model` (rather than
/// bundling the shared constant buffers — `gdn_q_scale`/`gdn_k_scale`/
/// `gdn_conv_zero_bias` — onto the per-layer `GdnWeights`, since those
/// three are identical across all 48 GDN layers and don't belong on a
/// loaded-per-layer struct).
#[allow(clippy::too_many_arguments)]
fn gdn_layer_full(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    model: &Qwen35Model,
    w: &GdnWeights,
    slot: &mut GdnSlot,
    x: &Tensor,
    layer_idx: usize,
    pos: u32,
    debug: bool,
) -> Result<Tensor> {
    let dbg = debug && layer_idx == 0;
    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;
    if dbg { dbg_sum(dev, "attn_norm-0", pos, &h); }

    let qkv = w.wqkv.gemv(dev, &h)?;
    if dbg { dbg_sum(dev, "linear_attn_qkv_mixed-0", pos, &qkv); }
    let z = w.wz.gemv(dev, &h)?;

    let (conv_out, new_conv_state) = ops::conv1d_causal_silu_many(
        dev, &qkv, &w.conv_w, &model.gdn_conv_zero_bias, &slot.conv_state, 1, cfg.gdn_conv_dim(),
    )?;
    slot.conv_state = new_conv_state;

    let key_dim = cfg.gdn_key_dim();
    let value_dim = cfg.gdn_value_dim();
    let q_raw = ops::slice(dev, &conv_out, 0, key_dim)?.reshaped(vec![cfg.gdn_num_k_heads, cfg.gdn_key_head_dim]);
    let k_raw = ops::slice(dev, &conv_out, key_dim, key_dim)?.reshaped(vec![cfg.gdn_num_k_heads, cfg.gdn_key_head_dim]);
    let v = ops::slice(dev, &conv_out, 2 * key_dim, value_dim)?.reshaped(vec![cfg.gdn_num_v_heads, cfg.gdn_value_head_dim]);
    if dbg { dbg_sum(dev, "v_conv_predelta-0", pos, &v); }

    let q = ops::rms_norm(dev, &q_raw, &model.gdn_q_scale, cfg.eps)?;
    let k = ops::rms_norm(dev, &k_raw, &model.gdn_k_scale, cfg.eps)?;
    if dbg { dbg_sum(dev, "q_conv_predelta-0", pos, &q); }
    if dbg { dbg_sum(dev, "k_conv_predelta-0", pos, &k); }

    let a_raw = ops::gemv(dev, &w.w_alpha, &h)?;
    let b_raw = ops::gemv(dev, &w.w_beta, &h)?;
    let (g, beta) = ops::gdn_gate_beta(dev, &a_raw, &b_raw, &w.dt_bias, &w.neg_exp_a_log, cfg.gdn_num_v_heads)?;
    if dbg {
        // Oracle's `gate-0` is LOG-space (ln(g) = neg_exp_a_log * softplus(...));
        // our `g` is already exp'd, so take ln() host-side for an apples-to-apples sum.
        let n = g.elem_count();
        let mut b = vec![0u8; n * 4];
        dev.synchronize().ok();
        dev.download(g.buffer.as_ref(), &mut b).ok();
        let gv: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let ln_sum: f64 = gv.iter().map(|&x| (x as f64).ln()).sum();
        eprintln!("QWEN35_DBG pos={pos} tag=gate-0(ln_g) n={n} sum={ln_sum:.6}");
        dbg_sum(dev, "beta_sigmoid-0", pos, &beta);
    }

    // `iron_gated_delta_step`'s own internal GQA decomposition is
    // `hk_idx = hv_idx / (Hv/Hk)` (contiguous blocks) — but the reference
    // this checkpoint was actually validated against (llama.cpp's CUDA
    // `gated_delta_net.cu`, `iq1 = fastmodulo(h_idx, neqk1)`, i.e.
    // `hv_idx % Hk`) uses the CYCLIC/tiled mapping instead (verified by
    // oracle checksum diff: `attn_output` matched to 0.01% only after
    // switching to cyclic — the block mapping was ~20% off). Pre-expand
    // q/k from `[Hk,Dk]` to `[Hv,Dk]` on-device via `ops::gather` with the
    // constant `gdn_gqa_cyclic_idx` (`idx[hv] = hv % Hk`), then call the
    // kernel with `hk=Hv` so its internal grouping becomes a no-op
    // identity map — sidesteps patching the pinned iron kernel entirely.
    let q = ops::gather(dev, &q, &model.gdn_gqa_cyclic_idx)?;
    let k = ops::gather(dev, &k, &model.gdn_gqa_cyclic_idx)?;

    let y = ops::gated_delta_step(
        dev, &q, &k, &v, &g, &beta, &slot.state,
        cfg.gdn_key_head_dim, cfg.gdn_value_head_dim, cfg.gdn_num_v_heads, cfg.gdn_num_v_heads,
    )?;
    if dbg { dbg_sum(dev, "attn_output-0", pos, &y); }

    let y = y.reshaped(vec![cfg.gdn_num_v_heads, cfg.gdn_value_head_dim]);
    let normed = ops::rms_norm(dev, &y, &w.ssm_norm_w, cfg.eps)?.reshaped(vec![value_dim]);
    if dbg { dbg_sum(dev, "node_50", pos, &normed); }
    let silu_z = ops::silu(dev, &z)?;
    let gated = ops::mul(dev, &normed, &silu_z)?;
    if dbg { dbg_sum(dev, "final_output-0", pos, &gated); }

    let o = w.wo.gemv(dev, &gated)?;
    if dbg { dbg_sum(dev, "linear_attn_out-0", pos, &o); }
    ops::add(dev, x, &o)
}

// ─────────────────────────────────────────────────────────────────────────
// MTP self-speculative decode
// ─────────────────────────────────────────────────────────────────────────

/// MTP's own attention KV cache. Architecturally identical to a backbone
/// full-attention layer's cache (see `FullAttnSlot`) — the MTP block tracks
/// the SAME absolute position axis as the backbone (see `mtp_draft_step`'s
/// doc), just with its own weights/K/V. Reusing `FullAttnSlot` directly
/// (rather than a parallel type) so `mtp_draft_step` can reuse
/// `full_attn_layer`'s exact kv_append/sdpa_decode call pattern.
pub struct MtpCache {
    slot: FullAttnSlot,
}

impl MtpCache {
    pub fn new(dev: &dyn Device, cfg: &Qwen35Config, max_seq: usize) -> Result<Self> {
        let n = cfg.n_kv_heads * max_seq * cfg.head_dim;
        let k = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, max_seq, cfg.head_dim], DType::F32);
        let v = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, max_seq, cfg.head_dim], DType::F32);
        Ok(MtpCache { slot: FullAttnSlot { k, v, cap: max_seq } })
    }
}

/// One MTP draft step — see `MtpWeights`'s module doc for the full
/// architectural contract. `h` = seed hidden (a "final normed hidden": the
/// backbone's own `xn` from [`decode_step_ex`] for the FIRST step of a
/// speculative round, or this function's own returned `h_nextn` when
/// chaining γ>1 draft steps). `tok` = seed token (its embedding is mixed
/// in via `eh_proj`) — the just-accepted/confirmed token for the first
/// step, or the previous step's own draft token when chaining. `pos` = the
/// absolute sequence position `tok` occupies; MTP's RoPE + KV-cache write
/// use this position directly (MTP shares the backbone's position axis —
/// confirmed against `llama.cpp`'s `graph_mtp`, which threads the SAME
/// `inp_pos` the backbone's own decode call used). Returns
/// `(draft_token, h_nextn)`: `draft_token` predicts the token at `pos+1`;
/// `h_nextn` is this step's own "final normed hidden", ready to chain
/// directly into the next call's `h`.
fn mtp_draft_step(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    model: &Qwen35Model,
    w: &MtpWeights,
    slot: &mut FullAttnSlot,
    h: &Tensor,
    tok: u32,
    pos: u32,
) -> Result<(u32, Tensor)> {
    let tok_buf = Tensor::new(dev.upload(&tok.to_le_bytes())?, vec![1], DType::U32);
    let e = ops::gather(dev, &model.tok_embd, &tok_buf)?.reshaped(vec![cfg.hidden]);
    let e_norm = ops::rms_norm(dev, &e, &w.enorm, cfg.eps)?;
    let h_norm = ops::rms_norm(dev, h, &w.hnorm, cfg.eps)?;
    // eh_proj(concat(e_norm, h_norm)) == eh_proj_e·e_norm + eh_proj_h·h_norm
    // — see `MtpWeights::eh_proj_e`'s doc for why this split form is used.
    let xe = w.eh_proj_e.gemv(dev, &e_norm)?;
    let xh = w.eh_proj_h.gemv(dev, &h_norm)?;
    let x = ops::add(dev, &xe, &xh)?;
    let inp_sa = x.clone();

    // One full-attention decoder layer on `x` (own weights/KV), same shape
    // as `full_attn_layer`'s trunk full-attn layers.
    let hn = ops::rms_norm(dev, &x, &w.attn_norm, cfg.eps)?;
    let qg = w.wq.gemv(dev, &hn)?;
    let k = w.wk.gemv(dev, &hn)?;
    let v = w.wv.gemv(dev, &hn)?;
    let q = ops::strided_head_half(dev, &qg, cfg.n_head, cfg.head_dim, 0)?;
    let gate = ops::strided_head_half(dev, &qg, cfg.n_head, cfg.head_dim, 1)?;
    let q = q.reshaped(vec![cfg.n_head, cfg.head_dim]);
    let k = k.reshaped(vec![cfg.n_kv_heads, cfg.head_dim]);
    let v = v.reshaped(vec![cfg.n_kv_heads, cfg.head_dim]);
    let q = ops::rms_norm(dev, &q, &w.q_norm, cfg.eps)?;
    let k = ops::rms_norm(dev, &k, &w.k_norm, cfg.eps)?;
    let q = ops::rope_yarn_partial(dev, &q, pos, cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144)?;
    let k = ops::rope_yarn_partial(dev, &k, pos, cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144)?;

    let posbuf = Tensor::new(dev.upload(&pos.to_le_bytes())?, vec![1], DType::U32);
    ops::kv_append(dev, &k, &slot.k, &posbuf, cfg.head_dim, slot.cap, cfg.n_kv_heads * cfg.head_dim)?;
    ops::kv_append(dev, &v, &slot.v, &posbuf, cfg.head_dim, slot.cap, cfg.n_kv_heads * cfg.head_dim)?;

    let heads_per_group = (cfg.n_head / cfg.n_kv_heads) as u32;
    let n_kv_used = pos + 1;
    let attn = ops::sdpa_decode(
        dev, &q, &slot.k, &slot.v, cfg.head_dim, n_kv_used, slot.cap as u32, heads_per_group, cfg.attn_scale(),
    )?;
    let attn_flat = attn.reshaped(vec![cfg.n_head * cfg.head_dim]);
    let gated = ops::sigmoid_mul(dev, &attn_flat, &gate)?;
    let o = w.wo.gemv(dev, &gated)?;
    let cur = ops::add(dev, &o, &inp_sa)?;

    let ffn_residual = cur.clone();
    let post = ops::rms_norm(dev, &cur, &w.ffn_norm, cfg.eps)?;
    let g2 = w.ffn_gate.gemv(dev, &post)?;
    let u2 = w.ffn_up.gemv(dev, &post)?;
    let act = ops::swiglu(dev, &g2, &u2)?;
    let ffn_out = w.ffn_down.gemv(dev, &act)?;
    let cur = ops::add(dev, &ffn_out, &ffn_residual)?;

    let h_nextn = ops::rms_norm(dev, &cur, &w.shared_head_norm, cfg.eps)?;
    // Shared head: MTP has no `nextn.shared_head_head` (see `MtpWeights`'s
    // doc), so this reuses `model.lm_head` directly.
    let logits = model.lm_head.gemv(dev, &h_nextn)?;
    let am = ops::argmax_f32_device(dev, &logits)?;
    let mut ib = [0u8; 4];
    dev.synchronize()?;
    dev.download(am.buffer.as_ref(), &mut ib)?;
    let draft_token = u32::from_le_bytes(ib);
    Ok((draft_token, h_nextn))
}

/// Combined backbone + MTP cache for speculative decode.
pub struct Qwen35SpecCache {
    pub backbone: Qwen35Cache,
    mtp: MtpCache,
}

impl Qwen35SpecCache {
    pub fn new(dev: &dyn Device, cfg: &Qwen35Config, max_seq: usize) -> Result<Self> {
        Ok(Qwen35SpecCache { backbone: Qwen35Cache::new(dev, cfg, max_seq)?, mtp: MtpCache::new(dev, cfg, max_seq)? })
    }
}

/// Per-round speculative-decode bookkeeping (for the γ ladder / acceptance-
/// rate measurement — see the perf-campaign report, not correctness).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecDecodeStats {
    pub n_drafted: usize,
    pub n_accepted: usize,
}

/// One speculative-decode round: draft up to `gamma` tokens from
/// `(last_token, last_pos, last_hidden)` via the MTP sidecar, then verify
/// against the backbone and return the run of tokens that are CONFIRMED
/// correct (i.e. bit-identical to what plain greedy `decode_step` would
/// have produced one token at a time).
///
/// ## Verify strategy: sequential ("lazy"), not batched
///
/// This calls [`decode_step_ex`] (the ordinary, unmodified backbone path)
/// **at most once per drafted candidate, and never for a candidate past the
/// first mismatch**: the loop compares the backbone's own greedy argmax
/// (computed from the PREVIOUS confirmed step's logits, cheap — just a
/// `lm_head.gemv` off already-computed hidden, no new trunk pass) against
/// the next draft token BEFORE running the trunk on it. Only once a draft
/// matches does its trunk pass actually execute (to advance KV/GDN state
/// and get the FOLLOWING step's logits for the next comparison).
///
/// Consequence: **no rollback machinery is needed.** Full-attention KV
/// (both the backbone's and MTP's own) is safe under the trunk's existing
/// position-indexed overwrite + explicit `n_kv_used` bound — a position
/// speculatively written during the draft phase but not reached by the
/// (lazy) verify phase is simply overwritten next round before anything
/// ever reads it. GDN's recurrent state is only ever advanced (via
/// `decode_step_ex`, which mutates it in place) for a token that has
/// ALREADY been confirmed correct at the point of the call — the sequence
/// of `decode_step_ex` calls this function makes is therefore exactly the
/// sequence plain (non-speculative) decode would have made, just
/// interleaved with cheap drafter calls. This is why the correctness bar
/// (bit-identical vs spec-decode OFF) holds by construction here.
///
/// ## What this does NOT do: save backbone bandwidth
///
/// Every ACCEPTED token still costs one full backbone forward pass (same
/// as plain decode) — this form saves nothing on the dominant GEMV cost,
/// and actively COSTS MORE than plain decode (drafter overhead + the
/// sequential argmax/download round-trips), so it will not beat the 14
/// tok/s bar. Its purpose is to validate the drafter (weight loading,
/// `eh_proj` split-GEMV algebra, KV-cache/position bookkeeping) and to
/// measure acceptance rate — the number that determines whether the
/// bandwidth-saving batched-verify path (chunked GDN + small-M
/// `gemm_q4_mpp` projections, needing real GDN-state snapshot/restore
/// rollback since it commits state for γ+1 positions before knowing the
/// accept count) is worth building out.
pub fn speculative_round(
    dev: &dyn Device,
    model: &Qwen35Model,
    cache: &mut Qwen35SpecCache,
    last_token: u32,
    last_pos: u32,
    last_hidden: &Tensor,
    gamma: usize,
) -> Result<(Vec<u32>, u32, u32, Tensor, SpecDecodeStats)> {
    let Some(mtp) = &model.mtp else {
        return Err(Error::Msg("speculative_round: model has no MTP weights (non-MTP checkpoint)".into()));
    };
    let cfg = &model.cfg;

    // ---- Draft phase: chain `gamma` MTP steps off the confirmed seed. ----
    let mut drafts: Vec<u32> = Vec::with_capacity(gamma);
    let mut dh = last_hidden.clone();
    let mut dtok = last_token;
    let mut dpos = last_pos;
    for _ in 0..gamma {
        let (t, h_next) = mtp_draft_step(dev, cfg, model, mtp, &mut cache.mtp.slot, &dh, dtok, dpos)?;
        drafts.push(t);
        dh = h_next;
        dtok = t;
        dpos += 1;
    }

    // ---- Verify phase: lazy sequential accept/reject (see doc above). ----
    // `logits` here is the ALREADY-COMPUTED backbone prediction for
    // `last_pos+1` (== `lm_head(last_hidden)`), reusing the hidden the
    // caller already paid for — no redundant trunk pass.
    let mut logits = model.lm_head.gemv(dev, last_hidden)?;
    let mut cur_hidden = last_hidden.clone();
    let mut cur_tok = last_token;
    let mut cur_pos = last_pos;
    let mut n_accept = 0usize;
    let mut accepted: Vec<u32> = Vec::with_capacity(gamma);

    for &d in &drafts {
        let am = ops::argmax_f32_device(dev, &logits)?;
        let mut ib = [0u8; 4];
        dev.synchronize()?;
        dev.download(am.buffer.as_ref(), &mut ib)?;
        let true_next = u32::from_le_bytes(ib);

        if true_next != d {
            // Misprediction: `true_next` (not `d`) is the real next token.
            // It was never drafted, so nothing needs to be rolled back —
            // just run the trunk on it once to seed the NEXT round, then
            // stop (the rest of `drafts` is discarded).
            let (logits2, xn2) = decode_step_ex(dev, model, &mut cache.backbone, true_next, cur_pos + 1)?;
            accepted.push(true_next);
            cur_pos += 1;
            cur_tok = true_next;
            cur_hidden = xn2;
            let _ = logits2; // not needed past this point; next round recomputes via lm_head(cur_hidden)
            return Ok((accepted, cur_tok, cur_pos, cur_hidden, SpecDecodeStats { n_drafted: drafts.len(), n_accepted: n_accept }));
        }

        // Accepted: `d` matches the backbone's own greedy choice. Advance
        // the trunk through it to get the next comparison's logits/hidden.
        accepted.push(d);
        cur_pos += 1;
        cur_tok = d;
        n_accept += 1;
        let (logits2, xn2) = decode_step_ex(dev, model, &mut cache.backbone, cur_tok, cur_pos)?;
        logits = logits2;
        cur_hidden = xn2;
    }

    // All `gamma` drafts accepted — `cur_hidden`/`logits` already hold the
    // confirmed state at `cur_pos`, ready to seed the next round directly.
    let _ = logits;
    Ok((accepted, cur_tok, cur_pos, cur_hidden, SpecDecodeStats { n_drafted: drafts.len(), n_accepted: n_accept }))
}
