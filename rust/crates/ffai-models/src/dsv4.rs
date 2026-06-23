// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4 Multi-head Latent Attention (decode), assembled from the
//! verified DSv4 ops in [`ffai_ops`]. Single-token, single-position KV
//! (`n_kv = 1`). Covers the full-attention layers; the CSA/HCA sparse path
//! (compressor + Lightning indexer) is a separate track (WIP in the
//! reference too). The mHC residual wrapping is the model loop's job.

use ffai_core::{DType, Device, Result, Tensor};
use ffai_ops as ops;

#[derive(Debug, Clone, Copy)]
pub struct MlaConfig {
    pub hidden: usize,
    pub n_heads: usize,
    pub head_dim: usize, // 512 (the d512 sink-SDPA kernel)
    pub q_lora_rank: usize,
    pub n_nope: usize,  // 448
    pub half_rot: usize, // 32  (rope tail = 64)
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

/// MLA attention weights (one full-attention layer).
pub struct MlaWeights {
    pub attn_norm: Tensor,        // [hidden]
    pub q_a: Tensor,              // [q_lora_rank, hidden]
    pub q_a_norm: Tensor,         // [q_lora_rank]
    pub q_b: Tensor,              // [n_heads*head_dim, q_lora_rank]
    pub kv: Tensor,               // [head_dim, hidden]
    pub kv_a_norm: Tensor,        // [head_dim]
    pub sink: Tensor,             // [n_heads] f32
    pub output_a: Vec<Tensor>,    // o_groups × [o_lora_rank, gsize]
    pub output_b: Tensor,         // [hidden, o_groups*o_lora_rank]
}

fn fb(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn tb(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Run MLA attention for a single token: `x [hidden]` → `blockOut [hidden]`.
pub fn mla_attention(
    dev: &dyn Device,
    cfg: &MlaConfig,
    w: &MlaWeights,
    x: &Tensor,
    position: u32,
) -> Result<Tensor> {
    let hd = cfg.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();

    let xn = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;

    // Q low-rank path + per-head unit-RMS norm + partial RoPE.
    let qa = ops::gemv(dev, &w.q_a, &xn)?;
    let qan = ops::rms_norm(dev, &qa, &w.q_a_norm, cfg.eps)?;
    let q = ops::gemv(dev, &w.q_b, &qan)?; // [n_heads*head_dim]
    let ones = Tensor::new(dev.upload(&tb(&vec![1.0f32; hd]))?, vec![hd], DType::F32);
    let q = ops::rms_norm(dev, &q.reshaped(vec![cfg.n_heads, hd]), &ones, cfg.eps)?;
    let q = ops::dsv4_partial_rope(
        dev, &q, cfg.n_heads as u32, hd as u32, cfg.n_nope as u32, cfg.half_rot as u32,
        position, cfg.rope_theta, false,
    )?;

    // KV latent path + norm + partial RoPE (single kv head).
    let kv = ops::gemv(dev, &w.kv, &xn)?; // [head_dim]
    let kvn = ops::rms_norm(dev, &kv, &w.kv_a_norm, cfg.eps)?;
    let kvn = ops::dsv4_partial_rope(
        dev, &kvn.reshaped(vec![1, hd]), 1, hd as u32, cfg.n_nope as u32, cfg.half_rot as u32,
        position, cfg.rope_theta, false,
    )?;

    // MQA sink-SDPA over the single-position cache (n_kv=1), then inverse RoPE.
    let attn =
        ops::sdpa_decode_sink(dev, &q, &kvn, &kvn, &w.sink, 1, 1, cfg.n_heads as u32, scale)?;
    let attn = ops::dsv4_partial_rope(
        dev, &attn, cfg.n_heads as u32, hd as u32, cfg.n_nope as u32, cfg.half_rot as u32,
        position, cfg.rope_theta, true,
    )?;

    // Grouped O-LoRA: attn [n_heads*head_dim] → o_groups groups; per-group
    // low-rank, concat, then the dense up-projection. The group slicing is
    // host-side (single token) since gemv consumes whole buffers.
    let qd = cfg.n_heads * hd;
    let gsize = qd / cfg.o_groups;
    let mut ab = vec![0u8; qd * 4];
    dev.synchronize()?;
    dev.download(attn.buffer.as_ref(), &mut ab)?;
    let attn_h = fb(&ab);

    let mut o_low = vec![0.0f32; cfg.o_groups * cfg.o_lora_rank];
    for g in 0..cfg.o_groups {
        let slice = &attn_h[g * gsize..(g + 1) * gsize];
        let tslice = Tensor::new(dev.upload(&tb(slice))?, vec![gsize], DType::F32);
        let og = ops::gemv(dev, &w.output_a[g], &tslice)?; // [o_lora_rank]
        dev.synchronize()?;
        let mut ogb = vec![0u8; cfg.o_lora_rank * 4];
        dev.download(og.buffer.as_ref(), &mut ogb)?;
        o_low[g * cfg.o_lora_rank..(g + 1) * cfg.o_lora_rank].copy_from_slice(&fb(&ogb));
    }
    let t_olow = Tensor::new(dev.upload(&tb(&o_low))?, vec![o_low.len()], DType::F32);
    ops::gemv(dev, &w.output_b, &t_olow)
}

// ── DeepSeek-V4 MoE feed-forward ────────────────────────────────────────

/// One expert's SwiGLU MLP (gate/up/down).
pub struct Dsv4Expert {
    pub gate: Tensor, // [intermediate, hidden]
    pub up: Tensor,   // [intermediate, hidden]
    pub down: Tensor, // [hidden, intermediate]
}

/// DSv4 routed MoE: sqrt(softplus) router (+bias) → top-k clamped-SwiGLU
/// experts (weighted by normalized unbiased score × routed_scaling) + an
/// always-on shared expert.
pub struct Dsv4Moe {
    pub router: Tensor,   // [n_experts, hidden] (f32 here)
    pub bias: Vec<f32>,   // [n_experts]
    pub experts: Vec<Dsv4Expert>,
    pub shared: Dsv4Expert,
    pub top_k: usize,
    pub routed_scaling: f32, // 1.5 for DSv4
    pub swiglu_limit: f32,   // 10.0 for DSv4
}

/// Run the DSv4 MoE feed-forward for a single token `x [hidden]`.
pub fn dsv4_moe(dev: &dyn Device, w: &Dsv4Moe, x: &Tensor) -> Result<Tensor> {
    let hidden = x.elem_count();

    // Router logits → host → sqrt(softplus) scores → top-k by biased.
    let logits_t = ops::gemv(dev, &w.router, x)?;
    dev.synchronize()?;
    let mut lb = vec![0u8; w.experts.len() * 4];
    dev.download(logits_t.buffer.as_ref(), &mut lb)?;
    let logits = fb(&lb);
    let (unbiased, biased) = ops::sqrtsoftplus_route(&logits, &w.bias);

    let mut order: Vec<usize> = (0..w.experts.len()).collect();
    order.sort_by(|&a, &b| biased[b].total_cmp(&biased[a]));
    let top: Vec<usize> = order.into_iter().take(w.top_k).collect();
    let denom: f32 = top.iter().map(|&e| unbiased[e]).sum();
    let weights: Vec<f32> =
        top.iter().map(|&e| unbiased[e] / denom * w.routed_scaling).collect();

    let mut acc = vec![0.0f32; hidden];
    let run_expert = |dev: &dyn Device, ex: &Dsv4Expert| -> Result<Vec<f32>> {
        let gate = ops::gemv(dev, &ex.gate, x)?;
        let up = ops::gemv(dev, &ex.up, x)?;
        let inner = ops::swiglu_limit(dev, &gate, &up, w.swiglu_limit)?;
        let out = ops::gemv(dev, &ex.down, &inner)?;
        dev.synchronize()?;
        let mut ob = vec![0u8; hidden * 4];
        dev.download(out.buffer.as_ref(), &mut ob)?;
        Ok(fb(&ob))
    };
    for (&e, &gw) in top.iter().zip(&weights) {
        let out = run_expert(dev, &w.experts[e])?;
        for i in 0..hidden {
            acc[i] += gw * out[i];
        }
    }
    let shared = run_expert(dev, &w.shared)?;
    for i in 0..hidden {
        acc[i] += shared[i];
    }

    Ok(Tensor::new(dev.upload(&tb(&acc))?, vec![hidden], x.dtype))
}

// ── DeepSeek-V4 mHC layer wrapping ──────────────────────────────────────

/// mHC (hyper-connection) weights for one subblock.
pub struct MhcWeights {
    pub hc_fn: Tensor,    // [24, n_hc*hidden] — the mix projection
    pub hc_scale: [f32; 3], // pre / post / comb scales
    pub hc_base: Vec<f32>,  // [24]
}

/// Full DSv4 attention subblock with the 4-channel mHC residual:
/// `mixes = hc_fn · flatten(state)` → sinkhorn split → collapse(state, pre) →
/// MLA(x) → expand(blockOut, post, comb, state) → new state `[n_hc, hidden]`.
/// `n_hc = 4`, `eps`/`iters` are the Sinkhorn params.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_attn_subblock(
    dev: &dyn Device,
    cfg: &MlaConfig,
    mhc: &MhcWeights,
    mla: &MlaWeights,
    hc_state: &Tensor, // [n_hc, hidden]
    position: u32,
    eps: f32,
    iters: u32,
) -> Result<Tensor> {
    const N_HC: usize = 4;
    let hidden = cfg.hidden;

    // mHC mix → sinkhorn split (host).
    let flat = hc_state.reshaped(vec![N_HC * hidden]);
    let mixes_t = ops::gemv(dev, &mhc.hc_fn, &flat)?; // [24]
    dev.synchronize()?;
    let mut mb = vec![0u8; 24 * 4];
    dev.download(mixes_t.buffer.as_ref(), &mut mb)?;
    let (pre, post, comb) =
        ops::dsv4_mhc_sinkhorn_split(&fb(&mb), mhc.hc_scale, &mhc.hc_base, eps, iters);

    let pre_t = Tensor::new(dev.upload(&tb(&pre))?, vec![N_HC], DType::F32);
    let post_t = Tensor::new(dev.upload(&tb(&post))?, vec![N_HC], DType::F32);
    let comb_t = Tensor::new(dev.upload(&tb(&comb))?, vec![N_HC * N_HC], DType::F32);

    // collapse → MLA → expand.
    let x = ops::dsv4_mhc_collapse(dev, hc_state, &pre_t, hidden as u32, N_HC as u32)?;
    let block_out = mla_attention(dev, cfg, mla, &x, position)?;
    ops::dsv4_mhc_expand(dev, &block_out, &post_t, &comb_t, hc_state, hidden as u32, N_HC as u32)
}
