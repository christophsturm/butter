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

pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    tok_embd: Tensor, // f32 [vocab, hidden]
    layers: Vec<Qwen35Layer>,
    out_norm: Tensor, // f32 [hidden]
    lm_head: Q8Head,
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
    let lm_head = {
        let (qs, scales, m, k) = g.q8_repack(head_name)?;
        Q8Head {
            qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs))?, vec![qs.len()], DType::U32),
            scales: Tensor::new(dev.upload(&f32s_to_bytes(&scales))?, vec![scales.len()], DType::F32),
            m,
            k,
        }
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

    eprintln!("Qwen35: load done in {:.2}s ({} layers)", load_t0.elapsed().as_secs_f64(), cfg.n_layers);

    Ok(Qwen35Model { cfg, tok_embd, layers, out_norm, lm_head, gdn_q_scale, gdn_k_scale, gdn_conv_zero_bias, gdn_gqa_cyclic_idx })
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
    ops::gemv_q8(dev, &model.lm_head.qs, &model.lm_head.scales, &xn, model.lm_head.m, model.lm_head.k, model.lm_head.m)
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
