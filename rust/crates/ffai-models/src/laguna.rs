// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Laguna-S-2.1 (GGUF arch "laguna"): a 117.55B / 8.14B-active sigmoid-routed
//! MoE transformer with a per-head softplus attention-output gate, per-head
//! QK-RMSNorm, and hybrid full/sliding-window attention (period-4, dense-
//! first). Layer 0 is a dense SwiGLU MLP (`mlp_only_layers = [0]`); layers
//! 1..47 are MoE (256 experts, top-10, sigmoid router with a score-
//! correction bias, one always-on shared expert). Full-attention layers use
//! YaRN RoPE over the first half of each head's dims; sliding-window layers
//! (window 512) use plain RoPE over the full head dim.
//!
//! Decode-only: single-token forward with a persistent KV cache (growing for
//! full-attention layers, a fixed-512 ring buffer for sliding layers). No
//! prefill / batched-token path here.
//!
//! Weights load straight from GGUF (see [`ffai_loader::gguf::Gguf`]):
//! dequantize each tensor to f32 once, then requantize into this engine's Q4
//! format ([`ffai_ops::quantize_q4`]) and drop the f32 — the GGUF k-quant
//! (Q4_K/Q6_K/...) never needs to match our block layout. Embeddings, the LM
//! head, and all norms stay dense f32 (small relative to the expert stacks
//! and precision-sensitive for routing/final logits).

use ffai_core::{DType, Device, Error, Result, Tensor};
use ffai_loader::gguf::Gguf;
use ffai_ops as ops;

fn u32s_to_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
/// The `ffai_gemv_q4_coalesced*` kernels read per-block scales as f16 (the
/// `d_f32: Tensor<f16>` param, name notwithstanding); uploading raw f32 bits
/// there silently reinterprets them as garbage f16s. Convert on the host at
/// load, exactly like the validated `qload(.., f16=true)` path in modeltests.
fn f32s_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    fn half_bits(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 112;
        if e <= 0 { return sign; }
        if e >= 0x1f { return sign | 0x7c00; }
        let m = (x >> 13) & 0x3ff;
        let round = (x >> 12) & 1;
        let v = ((e as u32) << 10) | m;
        sign | ((v + round) as u16)
    }
    v.iter().flat_map(|&f| half_bits(f).to_le_bytes()).collect()
}

/// One `[m, k]` row-major f32 weight requantized to this engine's Q4 format
/// and uploaded. `m`/`k` match the `gemv_q4` convention (`k` must be a
/// multiple of 32).
struct Q4Dense {
    qs: Tensor,
    scales: Tensor,
    m: usize,
    k: usize,
}
impl Q4Dense {
    fn upload(dev: &dyn Device, flat: &[f32], m: usize, k: usize) -> Result<Self> {
        let (qs, scales) = ops::quantize_q4(flat, m, k);
        Ok(Q4Dense {
            qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs))?, vec![qs.len()], DType::U32),
            scales: Tensor::new(dev.upload(&f32s_to_f16_bytes(&scales))?, vec![scales.len()], DType::F16),
            m,
            k,
        })
    }
    fn gemv(&self, dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
        ops::gemv_q4(dev, &self.qs, &self.scales, x, self.m, self.k, self.m)
    }
    fn gemv_accum(&self, dev: &dyn Device, x: &Tensor, acc: &Tensor, scale: &Tensor) -> Result<()> {
        ops::gemv_q4_accum(dev, &self.qs, &self.scales, x, acc, scale, self.m, self.k, self.m)
    }
}

/// `n_expert` stacked `[m, k]` Q4 weights, requantized expert-by-expert and
/// concatenated into one device buffer per projection. A per-expert view is
/// a byte-offset slice — no host round-trip to select an expert's weights,
/// only the (tiny) expert-id lookup needs to be on the host.
struct Q4ExpertStack {
    qs: Tensor,
    scales: Tensor,
    m: usize,
    k: usize,
    qs_words_per_expert: usize,
    scale_elems_per_expert: usize,
}
impl Q4ExpertStack {
    /// `flat` is the dequantized fused GGUF tensor, expert-major contiguous
    /// (`flat[e*m*k .. (e+1)*m*k]` is expert `e`'s `[m, k]` row-major slab —
    /// true for both `ffn_{gate,up}_exps` `[n_embd, n_ff_exp, n_expert]` and
    /// `ffn_down_exps` `[n_ff_exp, n_embd, n_expert]` given GGUF's
    /// fastest-dim-first tensor layout).
    fn upload(dev: &dyn Device, flat: &[f32], n_expert: usize, m: usize, k: usize) -> Result<Self> {
        let per_expert = m * k;
        if flat.len() != per_expert * n_expert {
            return Err(Error::Msg(format!(
                "Q4ExpertStack: flat len {} != n_expert*m*k ({} * {} * {})",
                flat.len(), n_expert, m, k
            )));
        }
        let qw = per_expert / 32 * 4; // quantize_q4: m*k/32 blocks, 4 u32/block
        let sw = per_expert / 32; // 1 f32 scale/block
        let mut qs_all = Vec::with_capacity(qw * n_expert);
        let mut sc_all = Vec::with_capacity(sw * n_expert);
        for e in 0..n_expert {
            let slab = &flat[e * per_expert..(e + 1) * per_expert];
            let (qs, sc) = ops::quantize_q4(slab, m, k);
            qs_all.extend_from_slice(&qs);
            sc_all.extend_from_slice(&sc);
        }
        Ok(Q4ExpertStack {
            qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs_all))?, vec![qs_all.len()], DType::U32),
            scales: Tensor::new(dev.upload(&f32s_to_f16_bytes(&sc_all))?, vec![sc_all.len()], DType::F16),
            m,
            k,
            qs_words_per_expert: qw,
            scale_elems_per_expert: sw,
        })
    }
    /// Device-side view of expert `e`'s `(qs, scales)` — a byte-offset slice
    /// into the shared stack, no copy.
    fn expert_view(&self, e: usize) -> (Tensor, Tensor) {
        let qw = self.qs_words_per_expert;
        let sw = self.scale_elems_per_expert;
        let qs = Tensor {
            buffer: self.qs.buffer.clone(),
            offset: self.qs.offset + e * qw * 4,
            shape: vec![qw],
            dtype: DType::U32,
        };
        let sc = Tensor {
            buffer: self.scales.buffer.clone(),
            offset: self.scales.offset + e * sw * 2, // f16 scales: 2 bytes each
            shape: vec![sw],
            dtype: DType::F16,
        };
        (qs, sc)
    }
    fn gemv(&self, dev: &dyn Device, e: usize, x: &Tensor) -> Result<Tensor> {
        let (qs, sc) = self.expert_view(e);
        ops::gemv_q4(dev, &qs, &sc, x, self.m, self.k, self.m)
    }
    fn gemv_accum(&self, dev: &dyn Device, e: usize, x: &Tensor, acc: &Tensor, scale: &Tensor) -> Result<()> {
        let (qs, sc) = self.expert_view(e);
        ops::gemv_q4_accum(dev, &qs, &sc, x, acc, scale, self.m, self.k, self.m)
    }
}

/// YaRN RoPE parameters for one layer-type (full-attention or sliding).
#[derive(Debug, Clone, Copy)]
pub struct RopeParams {
    pub n_rot: u32,
    pub theta: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub n_orig_ctx: u32,
}

/// Architecture config for Laguna-S-2.1, read from GGUF `laguna.*` metadata
/// (arch spec fields fall back to the model's known-fixed values when a key
/// is absent, e.g. weightless test fixtures or a schema this loader hasn't
/// seen yet — always cross-check against a real GGUF's `laguna.*` keys when
/// wiring a new checkpoint).
#[derive(Debug, Clone)]
pub struct LagunaConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    /// Q head count on a full-attention layer (`i % swa_period == 0`).
    pub n_head_full: usize,
    /// Q head count on a sliding-window layer.
    pub n_head_sliding: usize,
    /// Hybrid attention period: layer `i` is full-attention iff `i % swa_period == 0`.
    pub swa_period: usize,
    pub sliding_window: usize,
    pub eps: f32,
    pub vocab: usize,
    /// Leading dense (non-MoE) block count (Laguna-S-2.1: 1, i.e. layer 0 only).
    pub n_layer_dense_lead: usize,
    pub n_ff_dense: usize, // layer-0 dense MLP intermediate size
    pub n_expert: usize,
    pub n_expert_used: usize, // top_k
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub routed_scaling: f32,
    pub rope_full: RopeParams,
    pub rope_sliding: RopeParams,
}

impl LagunaConfig {
    pub fn is_full_layer(&self, i: usize) -> bool {
        i % self.swa_period == 0
    }
    pub fn is_dense_layer(&self, i: usize) -> bool {
        i < self.n_layer_dense_lead
    }
    pub fn n_head(&self, i: usize) -> usize {
        if self.is_full_layer(i) { self.n_head_full } else { self.n_head_sliding }
    }
    pub fn heads_per_group(&self, i: usize) -> u32 {
        (self.n_head(i) / self.n_kv_heads) as u32
    }
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
    pub fn rope_for(&self, i: usize) -> RopeParams {
        if self.is_full_layer(i) { self.rope_full } else { self.rope_sliding }
    }

    /// The literal Laguna-S-2.1 spec values (48 layers), used as fallbacks
    /// when a GGUF key is missing.
    fn defaults() -> Self {
        LagunaConfig {
            hidden: 3072,
            n_layers: 48,
            head_dim: 128,
            n_kv_heads: 8,
            n_head_full: 48,
            n_head_sliding: 72,
            swa_period: 4,
            sliding_window: 512,
            eps: 1e-6,
            vocab: 100352,
            n_layer_dense_lead: 1,
            n_ff_dense: 12288,
            n_expert: 256,
            n_expert_used: 10,
            n_ff_exp: 1024,
            n_ff_shexp: 1024,
            routed_scaling: 2.5,
            rope_full: RopeParams {
                n_rot: 64,
                theta: 500000.0,
                freq_scale: 1.0 / 128.0,
                ext_factor: 1.0,
                attn_factor: 1.4852030263919618,
                beta_fast: 32.0,
                beta_slow: 1.0,
                n_orig_ctx: 8192,
            },
            rope_sliding: RopeParams {
                n_rot: 128,
                theta: 10000.0,
                freq_scale: 1.0,
                ext_factor: 0.0,
                attn_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                n_orig_ctx: 8192,
            },
        }
    }

    /// Read `laguna.*` GGUF metadata, falling back to [`Self::defaults`] for
    /// any key this loader doesn't find (see the module doc — verify key
    /// names against a real GGUF before trusting a mismatch silently).
    pub fn from_gguf(g: &Gguf) -> Self {
        let d = Self::defaults();
        let u = |k: &str, def: usize| g.meta_u32(k).map(|v| v as usize).unwrap_or(def);
        let f = |k: &str, def: f32| g.meta_f32(k).unwrap_or(def);

        let n_layers = u("laguna.block_count", d.n_layers);
        let sliding_window = u("laguna.attention.sliding_window", d.sliding_window);
        let n_rot_full = u("laguna.rope.dimension_count", d.rope_full.n_rot as usize) as u32;
        let n_rot_sliding = u("laguna.rope.dimension_count_swa", d.rope_sliding.n_rot as usize) as u32;

        LagunaConfig {
            hidden: u("laguna.embedding_length", d.hidden),
            n_layers,
            head_dim: u("laguna.attention.key_length", d.head_dim),
            n_kv_heads: u("laguna.attention.head_count_kv", d.n_kv_heads),
            // The per-layer-type Q head split isn't a single scalar GGUF key
            // in this loader (see the struct doc); keep the spec constants.
            n_head_full: d.n_head_full,
            n_head_sliding: d.n_head_sliding,
            swa_period: d.swa_period,
            sliding_window,
            eps: f("laguna.attention.layer_norm_rms_epsilon", d.eps),
            vocab: u("laguna.vocab_size", d.vocab),
            n_layer_dense_lead: u("laguna.leading_dense_block_count", d.n_layer_dense_lead),
            n_ff_dense: u("laguna.feed_forward_length", d.n_ff_dense),
            n_expert: u("laguna.expert_count", d.n_expert),
            n_expert_used: u("laguna.expert_used_count", d.n_expert_used),
            n_ff_exp: u("laguna.expert_feed_forward_length", d.n_ff_exp),
            n_ff_shexp: u("laguna.expert_shared_feed_forward_length", d.n_ff_shexp),
            routed_scaling: f("laguna.expert_weights_scale", d.routed_scaling),
            rope_full: {
                // The GGUF is authoritative over the HF config: this export is
                // 256K (factor 32, yarn_attn_factor 1 + 0.1 ln 32 = 1.34657),
                // NOT the HF card's 1M / factor 128. Read the stored scaling
                // keys; the spec constants are only a last-resort fallback.
                let factor = f("laguna.rope.scaling.factor", 1.0 / d.rope_full.freq_scale);
                let attn_factor = g.meta_f32("laguna.rope.scaling.yarn_attn_factor")
                    .unwrap_or_else(|| 1.0 + 0.1 * factor.ln());
                RopeParams {
                    n_rot: n_rot_full,
                    theta: f("laguna.rope.freq_base", d.rope_full.theta),
                    freq_scale: 1.0 / factor,
                    ext_factor: d.rope_full.ext_factor,
                    attn_factor,
                    beta_fast: f("laguna.rope.scaling.yarn_beta_fast", d.rope_full.beta_fast),
                    beta_slow: f("laguna.rope.scaling.yarn_beta_slow", d.rope_full.beta_slow),
                    n_orig_ctx: u("laguna.rope.scaling.original_context_length", d.rope_full.n_orig_ctx as usize) as u32,
                }
            },
            rope_sliding: RopeParams {
                n_rot: n_rot_sliding,
                theta: f("laguna.rope.freq_base_swa", d.rope_sliding.theta),
                freq_scale: d.rope_sliding.freq_scale,
                ext_factor: d.rope_sliding.ext_factor,
                attn_factor: d.rope_sliding.attn_factor,
                beta_fast: d.rope_sliding.beta_fast,
                beta_slow: d.rope_sliding.beta_slow,
                n_orig_ctx: d.rope_sliding.n_orig_ctx,
            },
        }
    }
}

/// FFAI_LAGUNA_MICRO=1 enables three decode-path micro-fusions (QKVG concat
/// gemv, shared-expert fused swiglu gather, o_proj residual accum). The env
/// is read once at load time (to decide which weight layout to build below —
/// [`AttnProj`] / [`SharedExpert`]) and once per [`decode_step`] (to decide
/// the runtime code path). Both reads see the same value: the env is fixed
/// for the process lifetime (required for CUDA-graph capture anyway — see
/// [`GraphBufs`]), so load-time and decode-time never disagree.
fn laguna_micro_enabled() -> bool {
    std::env::var("FFAI_LAGUNA_MICRO").map(|v| v == "1").unwrap_or(false)
}

/// QKV(+gate) projection weights. `Fused` concatenates all four projections'
/// rows into one `Q4Dense` at load time (built ONLY under
/// `FFAI_LAGUNA_MICRO=1`, so a layer never carries both layouts resident at
/// once — that would double the ~2.4GB of attention-projection weights).
/// Row order in `wqkvg` is `[wq rows; wk rows; wv rows; wgate rows]`, all
/// sharing `k = hidden`; concatenation happens on the dequantized f32 slabs
/// BEFORE `quantize_q4`, so each row's Q4 block boundaries are unaffected
/// (`quantize_q4` blocks are per-row, block size 32 along `k`).
enum AttnProj {
    Split { wq: Q4Dense, wk: Q4Dense, wv: Q4Dense, wgate: Q4Dense },
    Fused { wqkvg: Q4Dense, n_q_rows: usize, n_kv_rows: usize, n_g_rows: usize },
}

/// Per-layer attention weights (Q4, except the two `[head_dim]` per-head
/// norms which stay f32).
struct AttnWeights {
    /// This layer's Q head count, derived from the loaded Q projection's
    /// actual row count (`n_q_rows / head_dim`) rather than `LagunaConfig`'s
    /// hardcoded 48/72 split — so a real GGUF's per-layer head count always
    /// wins over the spec constants if the two ever disagree.
    n_head: usize,
    attn_norm: Tensor, // [hidden] f32
    proj: AttnProj,
    wo: Q4Dense,    // [hidden, n_head*head_dim]
    q_norm: Tensor, // [head_dim] f32
    k_norm: Tensor, // [head_dim] f32
}

/// Layer-0's dense SwiGLU MLP.
struct DenseFfn {
    ffn_norm: Tensor, // [hidden] f32
    gate: Q4Dense,    // [n_ff_dense, hidden]
    up: Q4Dense,      // [n_ff_dense, hidden]
    down: Q4Dense,    // [hidden, n_ff_dense]
}

/// The always-on shared expert's gate+up projections. `Fused` repacks both
/// as single-expert (`n_expert=1`) [`Q4ExpertStack`]s at load time (built
/// ONLY under `FFAI_LAGUNA_MICRO=1`, same one-layout-resident rule as
/// [`AttnProj::Fused`]) plus a persistent one-element `idx=[0]` device
/// buffer, so decode can drive both through the same
/// `ops::moe_gather_q4_swiglu` the routed experts use (`top_k=1`) instead of
/// two plain gemvs + a host-orchestrated swiglu. `down` stays a plain
/// `Q4Dense` either way — only gate+up fuse into the gather.
struct SharedExpertFused {
    gate: Q4ExpertStack, // n_expert=1, [n_ff_shexp, hidden]
    up: Q4ExpertStack,   // n_expert=1, [n_ff_shexp, hidden]
    down: Q4Dense,       // [hidden, n_ff_shexp]
    /// Persistent device buffer holding the constant `u32` `0` — the single
    /// "expert index" `moe_gather_q4_swiglu` needs for a 1-expert stack.
    zero_idx: Tensor,
}
enum SharedExpert {
    Split { gate: Q4Dense, up: Q4Dense, down: Q4Dense },
    Fused(SharedExpertFused),
}

/// A routed MoE FFN block (layers `n_layer_dense_lead..`).
struct MoeFfn {
    ffn_norm: Tensor,  // [hidden] f32
    router: Tensor,    // [n_expert, hidden] f32 — kept dense for routing precision
    bias: Tensor,      // [n_expert] f32 — e_score_correction_bias (selection only)
    gate: Q4ExpertStack, // per-expert [n_ff_exp, hidden]
    up: Q4ExpertStack,   // per-expert [n_ff_exp, hidden]
    down: Q4ExpertStack, // per-expert [hidden, n_ff_exp]
    shared_expert: SharedExpert,
}

enum LayerFfn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

struct LagunaLayer {
    attn: AttnWeights,
    ffn: LayerFfn,
}

/// LM head resident as Q8 (the f32 head was a 1.2GB read per token, ~10% of
/// the whole decode bandwidth budget; Q8 quarters it at negligible logit
/// error). Same layout `ffai_ops::gemv_q8` consumes: 4 codes/u32, f32 scale
/// per 32-block.
struct Q8Head {
    qs: Tensor,
    scales: Tensor,
    m: usize,
    k: usize,
}

/// Full Laguna-S-2.1 weights, resident on `dev`.
pub struct LagunaModel {
    pub cfg: LagunaConfig,
    tok_embd: Tensor, // [vocab, hidden] f32
    layers: Vec<LagunaLayer>,
    out_norm: Tensor, // [hidden] f32
    lm_head: Q8Head,  // [vocab, hidden] q8
    /// Cached device-resident `1.0f32` scalar, reused as the shared expert's
    /// `gemv_q4_accum` weight (unconditional, unweighted accumulate).
    one: Tensor,
}

/// Persistent decode-time KV cache: one entry per layer, growing (`cap =
/// max_seq`) for full-attention layers, a fixed 512-slot ring for sliding
/// layers. RoPE is applied (with the token's absolute position) before the
/// K row is cached, so ring order doesn't matter for attention — softmax
/// over the cached K/V set is order-invariant once every slot is filled.
struct KvSlot {
    k: Tensor, // [n_kv_heads, cap, head_dim]
    v: Tensor,
    cap: usize,
}
pub struct LagunaKvCache {
    slots: Vec<KvSlot>,
}
impl LagunaKvCache {
    pub fn new(dev: &dyn Device, cfg: &LagunaConfig, max_seq: usize) -> Result<Self> {
        let mut slots = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let cap = if cfg.is_full_layer(i) { max_seq } else { cfg.sliding_window };
            let n = cfg.n_kv_heads * cap * cfg.head_dim;
            let k = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, cap, cfg.head_dim], DType::F32);
            let v = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, cap, cfg.head_dim], DType::F32);
            slots.push(KvSlot { k, v, cap });
        }
        Ok(LagunaKvCache { slots })
    }
}

fn dims_out_in(t: &ffai_loader::gguf::GgufTensor, name: &str) -> Result<(usize, usize)> {
    // GGUF dims are fastest-first: a 2-D `[out, in]` weight is stored as
    // `[in, out]` (`ggml`'s `create_tensor({k, m}, ...)` convention), which
    // is exactly the `(m, k)` `gemv_q4` wants with zero transposition.
    if t.dims.len() != 2 {
        return Err(Error::Msg(format!("{name}: expected 2-D tensor, got dims {:?}", t.dims)));
    }
    Ok((t.dims[1] as usize, t.dims[0] as usize))
}

fn load_dense(dev: &dyn Device, g: &Gguf, name: &str) -> Result<Q4Dense> {
    let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("gguf: tensor '{name}' missing")))?;
    let (m, k) = dims_out_in(t, name)?;
    let flat = g.dequant_f32(name)?;
    Q4Dense::upload(dev, &flat, m, k)
}

/// Like [`load_dense`] but returns the dequantized f32 slab (not yet
/// requantized/uploaded) plus its `(m, k)` — used by the QKVG-concat fusion,
/// which needs to concatenate rows across four tensors BEFORE `quantize_q4`.
fn load_dense_raw(g: &Gguf, name: &str) -> Result<(Vec<f32>, usize, usize)> {
    let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("gguf: tensor '{name}' missing")))?;
    let (m, k) = dims_out_in(t, name)?;
    let flat = g.dequant_f32(name)?;
    Ok((flat, m, k))
}

fn load_vec_f32(dev: &dyn Device, g: &Gguf, name: &str) -> Result<Tensor> {
    let flat = g.dequant_f32(name)?;
    let n = flat.len();
    Ok(Tensor::new(dev.upload(&f32s_to_bytes(&flat))?, vec![n], DType::F32))
}

fn load_expert_stack(dev: &dyn Device, g: &Gguf, name: &str, n_expert: usize, m: usize, k: usize) -> Result<Q4ExpertStack> {
    let flat = g.dequant_f32(name)?;
    Q4ExpertStack::upload(dev, &flat, n_expert, m, k)
}

/// Load Laguna-S-2.1 straight from a GGUF file: parse `laguna.*` metadata for
/// the config, then stream every tensor through `dequant_f32` → requantize
/// (Q4 for projections/experts, dense f32 for embeddings/norms/router) →
/// upload, dropping the f32 immediately after each tensor.
pub fn load_gguf(dev: &dyn Device, g: &Gguf) -> Result<LagunaModel> {
    let cfg = LagunaConfig::from_gguf(g);

    // token_embd / output are [vocab, hidden] (GGUF fastest-first dims already
    // give the flat data in row-major [vocab, hidden] order — see `dims_out_in`).
    let load_embed_like = |name: &str| -> Result<Tensor> {
        let flat = g.dequant_f32(name)?;
        Ok(Tensor::new(dev.upload(&f32s_to_bytes(&flat))?, vec![cfg.vocab, cfg.hidden], DType::F32))
    };
    let tok_embd = load_embed_like("token_embd.weight")?;

    let out_norm = load_vec_f32(dev, g, "output_norm.weight")?;
    let head_name = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let lm_head = {
        let (qs, sc, m, k) = g.q8_repack(head_name)?;
        Q8Head {
            qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs))?, vec![qs.len()], DType::U32),
            // gemv_q8 reads f32 scales (unlike the q4 family's f16), matching
            // the proven resident-Q8 loader path.
            scales: Tensor::new(dev.upload(&f32s_to_bytes(&sc))?, vec![sc.len()], DType::F32),
            m,
            k,
        }
    };
    let one = Tensor::new(dev.upload(&1.0f32.to_le_bytes())?, vec![1], DType::F32);

    let micro = laguna_micro_enabled();

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("blk.{i}");

        // n_head is derived from wq's own GGUF dims (48 full-attention / 72
        // sliding for Laguna-S-2.1), not the LagunaConfig formula — see the
        // AttnWeights doc comment.
        let (wq_flat, n_q_rows, hidden_k) = load_dense_raw(g, &format!("{p}.attn_q.weight"))?;
        let (wk_flat, n_kv_rows, _) = load_dense_raw(g, &format!("{p}.attn_k.weight"))?;
        let (wv_flat, n_kv_rows_v, _) = load_dense_raw(g, &format!("{p}.attn_v.weight"))?;
        let (wg_flat, n_g_rows, _) = load_dense_raw(g, &format!("{p}.attn_gate.weight"))?;
        debug_assert_eq!(n_kv_rows, n_kv_rows_v, "wk/wv row count mismatch");
        let n_head = n_q_rows / cfg.head_dim;

        // FFAI_LAGUNA_MICRO=1: concatenate the four dequantized f32 slabs
        // BEFORE quantize_q4 (row order q,k,v,gate; all share k=hidden), so
        // decode can do one gemv_q4 instead of four. Only ONE of the two
        // layouts is ever built per layer — building both would double the
        // resident attention-projection weights (~2.4GB total).
        let proj = if micro {
            let n_rows = n_q_rows + n_kv_rows + n_kv_rows_v + n_g_rows;
            let mut concat = Vec::with_capacity(n_rows * hidden_k);
            concat.extend_from_slice(&wq_flat);
            concat.extend_from_slice(&wk_flat);
            concat.extend_from_slice(&wv_flat);
            concat.extend_from_slice(&wg_flat);
            let wqkvg = Q4Dense::upload(dev, &concat, n_rows, hidden_k)?;
            AttnProj::Fused { wqkvg, n_q_rows, n_kv_rows, n_g_rows }
        } else {
            AttnProj::Split {
                wq: Q4Dense::upload(dev, &wq_flat, n_q_rows, hidden_k)?,
                wk: Q4Dense::upload(dev, &wk_flat, n_kv_rows, hidden_k)?,
                wv: Q4Dense::upload(dev, &wv_flat, n_kv_rows_v, hidden_k)?,
                wgate: Q4Dense::upload(dev, &wg_flat, n_g_rows, hidden_k)?,
            }
        };

        let attn = AttnWeights {
            n_head,
            attn_norm: load_vec_f32(dev, g, &format!("{p}.attn_norm.weight"))?,
            proj,
            wo: load_dense(dev, g, &format!("{p}.attn_output.weight"))?,
            q_norm: load_vec_f32(dev, g, &format!("{p}.attn_q_norm.weight"))?,
            k_norm: load_vec_f32(dev, g, &format!("{p}.attn_k_norm.weight"))?,
        };

        let ffn = if cfg.is_dense_layer(i) {
            LayerFfn::Dense(DenseFfn {
                ffn_norm: load_vec_f32(dev, g, &format!("{p}.ffn_norm.weight"))?,
                gate: load_dense(dev, g, &format!("{p}.ffn_gate.weight"))?,
                up: load_dense(dev, g, &format!("{p}.ffn_up.weight"))?,
                down: load_dense(dev, g, &format!("{p}.ffn_down.weight"))?,
            })
        } else {
            let router_flat = g.dequant_f32(&format!("{p}.ffn_gate_inp.weight"))?;
            let router = Tensor::new(
                dev.upload(&f32s_to_bytes(&router_flat))?,
                vec![cfg.n_expert, cfg.hidden],
                DType::F32,
            );

            // FFAI_LAGUNA_MICRO=1: repack the shared expert's gate+up as
            // single-expert (n_expert=1) stacks so decode can drive them
            // through the same moe_gather_q4_swiglu the routed experts use
            // (top_k=1) instead of two plain gemvs + a host-side swiglu.
            // `down` stays a plain Q4Dense either way. Only one layout is
            // ever built per layer (same memory-doubling rationale as
            // AttnProj above).
            let shared_expert = if micro {
                let gate = load_expert_stack(dev, g, &format!("{p}.ffn_gate_shexp.weight"), 1, cfg.n_ff_shexp, cfg.hidden)?;
                let up = load_expert_stack(dev, g, &format!("{p}.ffn_up_shexp.weight"), 1, cfg.n_ff_shexp, cfg.hidden)?;
                let down = load_dense(dev, g, &format!("{p}.ffn_down_shexp.weight"))?;
                let zero_idx = Tensor::new(dev.upload(&0u32.to_le_bytes())?, vec![1], DType::U32);
                SharedExpert::Fused(SharedExpertFused { gate, up, down, zero_idx })
            } else {
                SharedExpert::Split {
                    gate: load_dense(dev, g, &format!("{p}.ffn_gate_shexp.weight"))?,
                    up: load_dense(dev, g, &format!("{p}.ffn_up_shexp.weight"))?,
                    down: load_dense(dev, g, &format!("{p}.ffn_down_shexp.weight"))?,
                }
            };

            LayerFfn::Moe(MoeFfn {
                ffn_norm: load_vec_f32(dev, g, &format!("{p}.ffn_norm.weight"))?,
                router,
                bias: load_vec_f32(dev, g, &format!("{p}.exp_probs_b.bias"))?,
                gate: load_expert_stack(dev, g, &format!("{p}.ffn_gate_exps.weight"), cfg.n_expert, cfg.n_ff_exp, cfg.hidden)?,
                up: load_expert_stack(dev, g, &format!("{p}.ffn_up_exps.weight"), cfg.n_expert, cfg.n_ff_exp, cfg.hidden)?,
                down: load_expert_stack(dev, g, &format!("{p}.ffn_down_exps.weight"), cfg.n_expert, cfg.hidden, cfg.n_ff_exp)?,
                shared_expert,
            })
        };

        layers.push(LagunaLayer { attn, ffn });
    }

    Ok(LagunaModel { cfg, tok_embd, layers, out_norm, lm_head, one })
}

/// One decode layer: RMSNorm → QKV proj (+ per-head gate proj on the same
/// normed input) → per-head QK-RMSNorm → RoPE → KV-cache append → SDPA →
/// per-head softplus gate → o_proj → residual → RMSNorm → FFN (dense or
/// MoE) → residual. `pos` is the token's absolute sequence position.
#[allow(clippy::too_many_arguments)]
/// FFAI_LAGUNA_DEBUG=2: dump a tensor's stats (first-bad-op bisection inside
/// a layer). No-op cost is one env read per call site when disabled upstream.
fn dbg_stats(dev: &dyn Device, tag: &str, t: &Tensor) {
    let n = t.elem_count();
    let mut b = vec![0u8; n * 4];
    if dev.synchronize().is_err() || dev.download(t.buffer.as_ref(), &mut b).is_err() {
        eprintln!("    {tag}: <download failed>");
        return;
    }
    let f: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let n_nan = f.iter().filter(|v| v.is_nan()).count();
    let l2 = f.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
    let amax = f.iter().filter(|v| v.is_finite()).fold(0f32, |a, &v| a.max(v.abs()));
    eprintln!("    {tag}: n={n} |x|2={l2:.3e} amax={amax:.3e} nan={n_nan} head={:?}", &f[..4.min(n)]);
}

/// Per-step device buffers that make `decode_layer` CUDA-graph-capturable:
/// a captured graph replays the exact same recorded kernel launches every
/// token, so any value that changes per token (RoPE position, KV length)
/// must be read from a buffer whose CONTENTS the caller refreshes between
/// replays, rather than passed as a scalar baked in at capture time. The
/// buffer POINTERS are fixed for the lifetime of a `LagunaDecodeCtx`; only
/// the u32 values inside them change. `n_kv_full`/`n_kv_sliding` hold
/// `pos+1` / `min(pos+1, sliding_window)` respectively — which one a given
/// layer reads is chosen at call time by `cfg.is_full_layer(layer_idx)`,
/// which never changes across replays, so the choice of POINTER is itself
/// capture-stable even though the VALUE behind it isn't.
struct GraphBufs<'a> {
    n_kv_full: &'a Tensor,
    n_kv_sliding: &'a Tensor,
}

#[allow(clippy::too_many_arguments)]
fn decode_layer(
    dev: &dyn Device,
    cfg: &LagunaConfig,
    layer_idx: usize,
    w: &LagunaLayer,
    kv: &KvSlot,
    x: &Tensor,
    pos: u32,
    full_posbuf: &Tensor,
    sliding_posbuf: &Tensor,
    one: &Tensor,
    graph_bufs: Option<&GraphBufs>,
    micro: bool,
) -> Result<Tensor> {
    let deep_dbg = layer_idx == 0
        && std::env::var("FFAI_LAGUNA_DEBUG").map(|v| v == "2").unwrap_or(false);
    let hd = cfg.head_dim;
    let n_head = w.attn.n_head;
    let n_kv_heads = cfg.n_kv_heads;
    let is_full = cfg.is_full_layer(layer_idx);
    let rope = cfg.rope_for(layer_idx);
    let heads_per_group = (n_head / n_kv_heads) as u32;

    // ── attention ────────────────────────────────────────────────────
    if deep_dbg { dbg_stats(dev, "x_in", x); }
    let h = ops::rms_norm(dev, x, &w.attn.attn_norm, cfg.eps)?;
    if deep_dbg { dbg_stats(dev, "attn_norm", &h); }
    // g_proj consumes the SAME pre-attention normed input as q/k/v (not the
    // attention output) — matches the reference engine's graph exactly.
    //
    // FFAI_LAGUNA_MICRO=1 (fusion 1): one gemv_q4 on the concatenated
    // [wq;wk;wv;wgate] weight instead of four separate gemvs, then four
    // device-side slices (cheap 36KB/4KB/4KB/0.2KB copies) split the result
    // back into q/k/v/gate_raw. `ops::slice` is used rather than zero-copy
    // Tensor offset views because the gemv/gather kernels bind buffer BASE
    // pointers and ignore Tensor.offset.
    let (q, k, v, gate_raw) = match &w.attn.proj {
        AttnProj::Fused { wqkvg, n_q_rows, n_kv_rows, n_g_rows } => {
            let qkvg = wqkvg.gemv(dev, &h)?;
            let q = ops::slice(dev, &qkvg, 0, *n_q_rows)?;
            let k = ops::slice(dev, &qkvg, *n_q_rows, *n_kv_rows)?;
            let v = ops::slice(dev, &qkvg, n_q_rows + n_kv_rows, *n_kv_rows)?;
            let gate_raw = ops::slice(dev, &qkvg, n_q_rows + 2 * n_kv_rows, *n_g_rows)?;
            (q, k, v, gate_raw)
        }
        AttnProj::Split { wq, wk, wv, wgate } => {
            let q = wq.gemv(dev, &h)?;
            let k = wk.gemv(dev, &h)?;
            let v = wv.gemv(dev, &h)?;
            let gate_raw = wgate.gemv(dev, &h)?; // [n_head]
            (q, k, v, gate_raw)
        }
    };
    if deep_dbg {
        dbg_stats(dev, "q_proj", &q);
        dbg_stats(dev, "k_proj", &k);
        dbg_stats(dev, "v_proj", &v);
        dbg_stats(dev, "g_proj", &gate_raw);
    }

    let q = q.reshaped(vec![n_head, hd]);
    let k = k.reshaped(vec![n_kv_heads, hd]);
    let v = v.reshaped(vec![n_kv_heads, hd]);

    // Per-head QK RMSNorm (head_dim-level), before RoPE.
    let q = ops::rms_norm(dev, &q, &w.attn.q_norm, cfg.eps)?;
    let k = ops::rms_norm(dev, &k, &w.attn.k_norm, cfg.eps)?;
    if deep_dbg {
        dbg_stats(dev, "q_norm", &q);
        dbg_stats(dev, "k_norm", &k);
    }

    // YaRN (full layers) / plain (sliding layers) partial-rotary RoPE.
    // Graph mode reads `position` from the (capture-stable-pointer) posbuf
    // instead of taking it as a scalar baked in at capture time — same math,
    // see `ops::rope_yarn_partial_posbuf`'s doc.
    let (q, k) = if graph_bufs.is_some() {
        let posbuf = if is_full { full_posbuf } else { sliding_posbuf };
        let q = ops::rope_yarn_partial_posbuf(
            dev, &q, posbuf, rope.n_rot, rope.theta, rope.freq_scale, rope.ext_factor,
            rope.attn_factor, rope.beta_fast, rope.beta_slow, rope.n_orig_ctx,
        )?;
        let k = ops::rope_yarn_partial_posbuf(
            dev, &k, posbuf, rope.n_rot, rope.theta, rope.freq_scale, rope.ext_factor,
            rope.attn_factor, rope.beta_fast, rope.beta_slow, rope.n_orig_ctx,
        )?;
        (q, k)
    } else {
        let q = ops::rope_yarn_partial(
            dev, &q, pos, rope.n_rot, rope.theta, rope.freq_scale, rope.ext_factor,
            rope.attn_factor, rope.beta_fast, rope.beta_slow, rope.n_orig_ctx,
        )?;
        let k = ops::rope_yarn_partial(
            dev, &k, pos, rope.n_rot, rope.theta, rope.freq_scale, rope.ext_factor,
            rope.attn_factor, rope.beta_fast, rope.beta_slow, rope.n_orig_ctx,
        )?;
        (q, k)
    };
    if deep_dbg {
        dbg_stats(dev, "q_rope", &q);
        dbg_stats(dev, "k_rope", &k);
    }

    // KV cache append: full layers grow (posbuf = pos), sliding layers write
    // a ring slot (posbuf = pos % sliding_window). RoPE already used the
    // ABSOLUTE position above, so ring order doesn't affect the attention
    // math — only which slot gets overwritten.
    let posbuf = if is_full { full_posbuf } else { sliding_posbuf };
    ops::kv_append(dev, &k, &kv.k, posbuf, hd, kv.cap, n_kv_heads * hd)?;
    ops::kv_append(dev, &v, &kv.v, posbuf, hd, kv.cap, n_kv_heads * hd)?;

    // Graph mode reads the filled KV length from a (capture-stable-pointer)
    // device buffer instead of a baked-in scalar — same dense-causal math,
    // see `ops::sdpa_decode_nbuf`'s doc.
    let attn = if let Some(gb) = graph_bufs {
        let n_kv_buf = if is_full { gb.n_kv_full } else { gb.n_kv_sliding };
        ops::sdpa_decode_nbuf(
            dev, &q, &kv.k, &kv.v, hd, n_kv_buf, kv.cap as u32, heads_per_group, cfg.attn_scale(),
        )?
    } else {
        let n_kv_used = if is_full { pos + 1 } else { (pos + 1).min(cfg.sliding_window as u32) };
        ops::sdpa_decode(
            dev, &q, &kv.k, &kv.v, hd, n_kv_used, kv.cap as u32, heads_per_group, cfg.attn_scale(),
        )?
    }; // [n_head, hd]
    if deep_dbg { dbg_stats(dev, "sdpa", &attn); }

    // Per-head softplus gate, before o_proj.
    let gated = ops::gate_softplus_mul_perhead(dev, &attn, &gate_raw)?;
    let gated_flat = gated.reshaped(vec![n_head * hd]);
    if deep_dbg { dbg_stats(dev, "gated", &gated); }
    // FFAI_LAGUNA_MICRO=1 (fusion 3): copy x into a fresh accumulator (there
    // is no tensor-clone op; `ops::slice(dev, x, 0, hidden)` is a device copy)
    // then `wo.gemv_accum` computes x1 = x + wo·gated in one kernel, instead
    // of materializing `o = wo.gemv(...)` and adding it separately. Same op
    // count (copy+accum vs gemv+add) — this fusion trades materializing `o`
    // for materializing the copy, so it wins only if the accum kernel avoids
    // enough bandwidth to offset that; benchmarking decides, not this diff.
    let x1 = if micro {
        let acc = ops::slice(dev, x, 0, cfg.hidden)?;
        w.attn.wo.gemv_accum(dev, &gated_flat, &acc, one)?;
        if deep_dbg { dbg_stats(dev, "x_attn_res", &acc); }
        acc
    } else {
        let o = w.attn.wo.gemv(dev, &gated_flat)?;
        let x1 = ops::add(dev, x, &o)?;
        if deep_dbg {
            dbg_stats(dev, "o_proj", &o);
            dbg_stats(dev, "x_attn_res", &x1);
        }
        x1
    };

    // ── FFN ──────────────────────────────────────────────────────────
    let x2 = match &w.ffn {
        LayerFfn::Dense(f) => {
            let h2 = ops::rms_norm(dev, &x1, &f.ffn_norm, cfg.eps)?;
            let gate = f.gate.gemv(dev, &h2)?;
            let up = f.up.gemv(dev, &h2)?;
            let act = ops::swiglu(dev, &gate, &up)?;
            let down = f.down.gemv(dev, &act)?;
            if deep_dbg {
                dbg_stats(dev, "ffn_norm", &h2);
                dbg_stats(dev, "ffn_act", &act);
                dbg_stats(dev, "ffn_down", &down);
            }
            ops::add(dev, &x1, &down)?
        }
        LayerFfn::Moe(f) => {
            let moe_dbg = (1..=2).contains(&layer_idx)
                && std::env::var("FFAI_LAGUNA_DEBUG").map(|v| v == "2").unwrap_or(false);
            let h2 = ops::rms_norm(dev, &x1, &f.ffn_norm, cfg.eps)?;
            let logits = ops::gemv(dev, &f.router, &h2)?; // [n_expert]
            if moe_dbg {
                dbg_stats(dev, &format!("moe{layer_idx}_x_in"), x);
                dbg_stats(dev, &format!("moe{layer_idx}_x_attn_res"), &x1);
                dbg_stats(dev, &format!("moe{layer_idx}_ffn_norm"), &h2);
                dbg_stats(dev, &format!("moe{layer_idx}_router_logits"), &logits);
            }
            let (idx, wts) = ops::moe_router_device(
                dev, &logits, &f.bias, cfg.n_expert, cfg.n_expert_used, cfg.routed_scaling,
            )?;
            if moe_dbg {
                let mut wb = vec![0u8; cfg.n_expert_used * 4];
                let _ = dev.synchronize();
                let _ = dev.download(wts.buffer.as_ref(), &mut wb);
                let w: Vec<f32> = wb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                let mut ib = vec![0u8; cfg.n_expert_used * 4];
                let _ = dev.download(idx.buffer.as_ref(), &mut ib);
                let iv: Vec<u32> = ib.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                eprintln!("    moe{layer_idx}_topk idx={iv:?} wts={w:?}");
            }

            // Fully device-side expert path: gather-GEMV the top_k gate and up
            // stacks (the gather kernels index the expert slab by `idx` inside
            // the kernel; a byte-offset Tensor view does NOT work here because
            // dispatch binds buffer BASE pointers and drops Tensor.offset),
            // SwiGLU elementwise on the [top_k*inter] pair, gather the down
            // stack, then router-weighted sum into the accumulator.
            //
            // FFAI_LAGUNA_FUSE=1 replaces the two gathers + host swiglu with
            // one fused gate+up+SwiGLU gather kernel: same `h2` read once
            // instead of twice, one launch instead of two gathers plus the
            // elementwise pass. Default (unset) keeps the unfused path.
            let act = if std::env::var("FFAI_LAGUNA_FUSE").map(|v| v == "1").unwrap_or(false) {
                ops::moe_gather_q4_swiglu(
                    dev, &f.gate.qs, &f.gate.scales, &f.up.qs, &f.up.scales, &h2, &idx,
                    cfg.n_expert_used, cfg.n_ff_exp, cfg.hidden,
                )?
            } else {
                let gates = ops::moe_gather_q4(dev, &f.gate.qs, &f.gate.scales, &h2, &idx, cfg.n_expert_used, cfg.n_ff_exp, cfg.hidden)?;
                let ups = ops::moe_gather_q4(dev, &f.up.qs, &f.up.scales, &h2, &idx, cfg.n_expert_used, cfg.n_ff_exp, cfg.hidden)?;
                ops::swiglu(dev, &gates, &ups)?
            };
            let downs = ops::moe_gather_down(dev, &f.down.qs, &f.down.scales, &act, &idx, cfg.n_expert_used, cfg.n_ff_exp, cfg.hidden)?;
            let acc = Tensor::new(dev.alloc_zeroed(cfg.hidden * 4)?, vec![cfg.hidden], DType::F32);
            ops::moe_weighted_sum(dev, &downs, &wts, &acc, cfg.hidden, cfg.n_expert_used)?;

            if moe_dbg { dbg_stats(dev, &format!("moe{layer_idx}_acc_routed"), &acc); }

            // Always-on shared expert: same accumulator, unweighted (scale = 1,
            // the model-wide cached `one` scalar — no gating/router weight).
            //
            // FFAI_LAGUNA_MICRO=1 (fusion 2): the gate+up projections were
            // repacked at load time as single-expert (n_expert=1) stacks, so
            // one moe_gather_q4_swiglu call (top_k=1, the persistent
            // zero_idx buffer selecting "expert 0") replaces the separate
            // gate gemv + up gemv + host-orchestrated swiglu. `down` is
            // unchanged either way (plain gemv_accum into the routed acc).
            match &f.shared_expert {
                SharedExpert::Fused(se) => {
                    let sact = ops::moe_gather_q4_swiglu(
                        dev, &se.gate.qs, &se.gate.scales, &se.up.qs, &se.up.scales,
                        &h2, &se.zero_idx, 1, cfg.n_ff_shexp, cfg.hidden,
                    )?;
                    se.down.gemv_accum(dev, &sact, &acc, one)?;
                }
                SharedExpert::Split { gate, up, down } => {
                    let sgate = gate.gemv(dev, &h2)?;
                    let sup = up.gemv(dev, &h2)?;
                    let sact = ops::swiglu(dev, &sgate, &sup)?;
                    down.gemv_accum(dev, &sact, &acc, one)?;
                }
            }

            if moe_dbg { dbg_stats(dev, &format!("moe{layer_idx}_acc_final"), &acc); }
            ops::add(dev, &x1, &acc)?
        }
    };

    Ok(x2)
}

/// Persistent per-step device scalars decode reads instead of uploading a
/// fresh 1-element buffer every call. Allocated ONCE (in [`DecodeWorkspace::new`])
/// and refreshed in place every step via [`DecodeWorkspace::update`] — the
/// refresh always happens BEFORE the (eager or captured-graph) decode work
/// that reads them, and never inside a graph-capture region, so it's safe
/// regardless of whether the caller is warming up, capturing, or replaying.
///
/// `full_pos`/`sliding_pos` are decode_layer's existing posbuf inputs (KV
/// cache write index — full layers grow, sliding layers ring); `n_kv_full`/
/// `n_kv_sliding` are the NEW graph-mode buffers `sdpa_decode_nbuf` reads
/// (`pos+1` / `min(pos+1, sliding_window)`). Eager (non-graph) decode only
/// ever reads `token_id`/`full_pos`/`sliding_pos`; `n_kv_full`/`n_kv_sliding`
/// are only consulted by `decode_layer` when `FFAI_LAGUNA_GRAPH=1`.
pub struct DecodeWorkspace {
    token_id: Tensor,
    full_pos: Tensor,
    sliding_pos: Tensor,
    n_kv_full: Tensor,
    n_kv_sliding: Tensor,
}

impl DecodeWorkspace {
    pub fn new(dev: &dyn Device) -> Result<Self> {
        let zero = || -> Result<Tensor> {
            Ok(Tensor::new(dev.upload(&0u32.to_le_bytes())?, vec![1], DType::U32))
        };
        Ok(DecodeWorkspace {
            token_id: zero()?,
            full_pos: zero()?,
            sliding_pos: zero()?,
            n_kv_full: zero()?,
            n_kv_sliding: zero()?,
        })
    }

    /// Refresh every workspace buffer for the token at `pos`. Five tiny
    /// (single-`u32`) raw-CUDA writes, in place — no reallocation, so the
    /// buffers' device pointers (what a captured graph binds) never move.
    /// MUST be called outside any graph-capture region.
    pub fn update(&self, dev: &dyn Device, cfg: &LagunaConfig, token_id: u32, pos: u32) -> Result<()> {
        // TODO(perf): 5 tiny kernel launches per step (one per scalar). Still
        // far cheaper than 5 cuMemAlloc+H2D uploads, but a single fused
        // "write 5 u32s" raw kernel (one launch, 5 scalar args) would shave a
        // bit more host-side launch overhead off the outside-the-graph cost
        // once this becomes the next bottleneck.
        ops::write_u32(dev, &self.token_id, token_id)?;
        ops::write_u32(dev, &self.full_pos, pos)?;
        let sliding_pos = pos % cfg.sliding_window as u32;
        ops::write_u32(dev, &self.sliding_pos, sliding_pos)?;
        ops::write_u32(dev, &self.n_kv_full, pos + 1)?;
        let n_kv_sliding = (pos + 1).min(cfg.sliding_window as u32);
        ops::write_u32(dev, &self.n_kv_sliding, n_kv_sliding)?;
        Ok(())
    }
}

/// Single-token decode: embed → every layer (self-attention + cache append +
/// FFN) → final RMSNorm → lm_head → next-token logits `[vocab]`. `pos` is
/// this token's absolute sequence position (0 for the first token); the
/// caller advances it and re-calls per generated token, reusing `kv` across
/// calls so the cache persists.
///
/// Reads `token_id`/RoPE-position/KV-length from `ws` rather than uploading
/// fresh 1-element buffers every call (the former per-step host upload this
/// function used to do) — the caller must have already called
/// `ws.update(dev, &model.cfg, token_id, pos)` for this step. This is both a
/// standalone eager-mode win (no per-step `cuMemAlloc`+H2D for those
/// scalars) and what makes the whole step CUDA-graph-capturable: a graph
/// replays the exact recorded kernel launches, so every per-token input has
/// to sit behind a buffer POINTER that stays fixed across replays.
///
/// This is the path used both by plain eager decode (no env set) and by
/// [`LagunaDecodeCtx`]'s warmup steps and one-time capture step; graph
/// REPLAY (after capture) does not call this function at all — it re-runs
/// the already-recorded launches via `Device::graph_launch`.
///
/// `FFAI_LAGUNA_GRAPH=1` switches `decode_layer`'s RoPE/SDPA to the
/// buffer-driven (capture-safe) kernels; without it, behavior is byte-for-byte
/// identical to before this refactor (same kernels, same scalar arguments,
/// just reading `token_id`/`pos` off a persistent buffer instead of a fresh
/// one — the values are the same either way).
pub fn decode_step(
    dev: &dyn Device,
    model: &LagunaModel,
    kv: &LagunaKvCache,
    ws: &DecodeWorkspace,
    pos: u32,
) -> Result<Tensor> {
    let cfg = &model.cfg;
    let mut x = ops::gather(dev, &model.tok_embd, &ws.token_id)?.reshaped(vec![cfg.hidden]);

    let graph_mode = std::env::var("FFAI_LAGUNA_GRAPH").is_ok();
    let graph_bufs = if graph_mode {
        Some(GraphBufs { n_kv_full: &ws.n_kv_full, n_kv_sliding: &ws.n_kv_sliding })
    } else {
        None
    };
    // Read once per decode_step, passed down as a bool: all three micro-
    // fusions (QKVG concat gemv, shared-expert fused swiglu gather, o_proj
    // residual accum) are gated on this single env var. Fixed for the
    // process lifetime (same requirement CUDA-graph capture already has —
    // see GraphBufs's doc), so this per-step read always agrees with the
    // load-time read that picked AttnProj/SharedExpert's Fused vs Split.
    let micro = laguna_micro_enabled();

    // FFAI_LAGUNA_DEBUG=1: per-layer residual stats (L2 norm, NaN count) to
    // bisect a divergence to the first bad layer. Hot path unaffected when
    // the env is unset. Downloads mid-step, so it is INCOMPATIBLE with graph
    // capture/replay — `LagunaDecodeCtx::step` refuses to run at all when
    // this is set (see its doc); a bare `decode_step` call with both env
    // vars set would otherwise try to `dev.download` from inside a capture
    // region, which is forbidden by the CUDA driver.
    let debug = std::env::var("FFAI_LAGUNA_DEBUG").map(|v| v == "1").unwrap_or(false);
    for (i, layer) in model.layers.iter().enumerate() {
        x = decode_layer(dev, cfg, i, layer, &kv.slots[i], &x, pos, &ws.full_pos, &ws.sliding_pos, &model.one, graph_bufs.as_ref(), micro)?;
        if debug {
            let mut xb = vec![0u8; cfg.hidden * 4];
            dev.synchronize()?;
            dev.download(x.buffer.as_ref(), &mut xb)?;
            let xf: Vec<f32> = xb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let n_nan = xf.iter().filter(|v| v.is_nan()).count();
            let l2 = xf.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
            let amax = xf.iter().fold(0f32, |a, &v| if v.abs() > a { v.abs() } else { a });
            eprintln!("  layer {i:2} ({}{}): |x|2={l2:.3e} amax={amax:.3e} nan={n_nan} head={:?}",
                if cfg.is_full_layer(i) { "full" } else { "swa " },
                if cfg.is_dense_layer(i) { ", dense" } else { "" },
                &xf[..4]);
        }
    }

    let xn = ops::rms_norm(dev, &x, &model.out_norm, cfg.eps)?;
    ops::gemv_q8(dev, &model.lm_head.qs, &model.lm_head.scales, &xn, model.lm_head.m, model.lm_head.k, model.lm_head.m)
}

/// CUDA-graph-capturing decode driver: collapses Laguna's ~1000+ per-token
/// kernel launches (48 layers x ~22 ops) into ONE `cuGraphLaunch` replay per
/// token, removing host-launch-gap overhead. Enabled by running with
/// `FFAI_LAGUNA_GRAPH=1` (also read by `decode_step`/`decode_layer` to pick
/// the buffer-driven RoPE/SDPA kernels — see their docs).
///
/// Protocol (mirrors the NemotronH `NEMOTRON_GRAPH` precedent in
/// `ffai-modeltests`): the first two `step()` calls run [`decode_step`]
/// eagerly ("warm up N=2 full steps normally" — the pool's bucket free-list
/// reaches a deterministic steady state after the first call, since every
/// step walks the identical fixed 48-layer op sequence regardless of
/// `token_id`/`pos`, so the second call already allocates nothing new). The
/// THIRD call captures: `Device::begin_capture` + one more `decode_step` +
/// `Device::end_capture`. Capture only RECORDS the stream's kernel launches
/// — it does not execute them — so that third call's `decode_step` result is
/// not yet real data; `step()` immediately follows the capture with one
/// `Device::graph_launch` to actually compute that token's logits. Every
/// call after that just refreshes the workspace and replays the graph.
///
/// The logits `Tensor` returned from the capture step is kept alive for the
/// lifetime of this `LagunaDecodeCtx` (never dropped) so its device buffer's
/// pointer can never be recycled by the allocator's pool — the same buffer
/// is what every subsequent `graph_launch` writes into, and what `step()`
/// returns (a cheap `Arc` clone) every time thereafter.
///
/// `FFAI_LAGUNA_DEBUG` downloads mid-step and is incompatible with capture;
/// `step()` returns an error immediately if it's set, rather than silently
/// producing a broken capture.
pub struct LagunaDecodeCtx {
    ws: DecodeWorkspace,
    /// Executable-graph handle from `Device::end_capture` (an opaque `u64`;
    /// see the `Device` trait doc — CUDA's `CUgraphExec` cast through
    /// `usize`). `None` until the capture step has run.
    graph: Option<u64>,
    /// The fixed output buffer graph replays write into (see the struct doc).
    logits: Option<Tensor>,
    /// Steps run so far via this ctx (warmup + capture; stops incrementing
    /// once capture has happened — replay doesn't touch it).
    step_count: usize,
}

impl LagunaDecodeCtx {
    /// Eager warmup steps before the capture step (brief: "warm up N=2 full
    /// steps normally"). Capture happens on call number `WARMUP_STEPS + 1`.
    const WARMUP_STEPS: usize = 2;

    /// `model`/`kv`/`max_seq` are accepted (matching the `LagunaKvCache::new`
    /// shape) for parity with the decode driver's usual construction
    /// arguments and to leave room for future workspace sizing; today's
    /// workspace is five fixed 1-element buffers, independent of `max_seq`.
    pub fn new(dev: &dyn Device, model: &LagunaModel, kv: &LagunaKvCache, max_seq: usize) -> Result<Self> {
        let _ = (model, kv, max_seq);
        Ok(LagunaDecodeCtx { ws: DecodeWorkspace::new(dev)?, graph: None, logits: None, step_count: 0 })
    }

    /// Decode one token. Identical call shape to [`decode_step`] plus `self`;
    /// internally dispatches to eager warmup, one-time capture, or graph
    /// replay depending on how many steps have run. See the struct doc for
    /// the full protocol.
    pub fn step(
        &mut self,
        dev: &dyn Device,
        model: &LagunaModel,
        kv: &LagunaKvCache,
        token_id: u32,
        pos: u32,
    ) -> Result<Tensor> {
        if std::env::var("FFAI_LAGUNA_DEBUG").is_ok() {
            return Err(Error::Msg(
                "LagunaDecodeCtx::step: FFAI_LAGUNA_DEBUG is incompatible with CUDA-graph \
                 capture/replay (the debug path downloads mid-step, which the CUDA driver \
                 forbids during capture). Unset FFAI_LAGUNA_DEBUG or use plain decode_step."
                    .into(),
            ));
        }

        // Refresh the workspace BEFORE any capture/replay — always outside
        // the captured region, whether we're warming up, about to capture,
        // or replaying an already-captured graph.
        self.ws.update(dev, &model.cfg, token_id, pos)?;

        if let Some(exec) = self.graph {
            dev.graph_launch(exec)?;
            return Ok(self
                .logits
                .clone()
                .expect("LagunaDecodeCtx: graph captured but no logits tensor saved"));
        }

        self.step_count += 1;
        if self.step_count <= Self::WARMUP_STEPS {
            // Eager warmup: reaches pool steady state (deterministic alloc
            // order — every step walks the same fixed layer sequence).
            return decode_step(dev, model, kv, &self.ws, pos);
        }

        // step_count == WARMUP_STEPS + 1: capture this step's op sequence.
        // Capture only RECORDS launches; it does not execute them, so
        // `logits` is not yet valid data — one `graph_launch` right after
        // `end_capture` actually computes it (against the workspace values
        // written above, so it's the correct result for THIS token).
        dev.begin_capture()?;
        let logits = decode_step(dev, model, kv, &self.ws, pos)?;
        let exec = dev.end_capture()?;
        self.graph = Some(exec);
        self.logits = Some(logits.clone());
        dev.graph_launch(exec)?;
        Ok(logits)
    }
}
