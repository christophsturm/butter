// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Mixture-of-Experts MLP builder (DeepSeek-V4, GPT-OSS, Granite4, Qwen-MoE).
//! The attention half is the dense transformer (see [`super::llama`]); only
//! the feed-forward is replaced by a routed expert mixture. Built on the
//! shared [`ffai_ops`], so it runs on any backend.
//!
//! Single-token routing is done host-side (download the `n_experts` router
//! logits, pick top-k, softmax) — exact and cheap for one token; a fused
//! GPU router/top-k is a later optimization. The per-expert gate/up/down and
//! SwiGLU run on the device through the same ops the dense MLP uses.

use ffai_core::{DType, Device, Error, Result, Tensor};
use ffai_ops as ops;

/// One expert's SwiGLU MLP weights.
pub struct ExpertWeights {
    pub gate: Tensor, // [intermediate, hidden]
    pub up: Tensor,   // [intermediate, hidden]
    pub down: Tensor, // [hidden, intermediate]
}

/// A routed MoE feed-forward block.
pub struct MoeMlp {
    pub router: Tensor, // [n_experts, hidden]
    pub experts: Vec<ExpertWeights>,
    pub top_k: usize,
    /// Normalize the top-k routing weights with a softmax (Qwen/DeepSeek do).
    pub norm_topk: bool,
}

fn to_f32(bytes: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        _ => vec![],
    }
}
fn from_f32(vals: &[f32], dt: DType) -> Vec<u8> {
    match dt {
        DType::F32 => vals.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::BF16 => vals
            .iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        _ => vec![],
    }
}

/// Run the routed MoE MLP for a single token's hidden vector `h` `[hidden]`.
/// Returns the mixed expert output `[hidden]` (same dtype as `h`).
pub fn moe_mlp(dev: &dyn Device, w: &MoeMlp, h: &Tensor) -> Result<Tensor> {
    let n_experts = w.experts.len();
    let hidden = h.elem_count();

    // Router → logits, read to host for top-k selection.
    let logits = ops::gemv(dev, &w.router, h)?;
    dev.synchronize()?;
    let mut lb = vec![0u8; n_experts * logits.dtype.size_bytes()];
    dev.download(logits.buffer.as_ref(), &mut lb)?;
    let lf = to_f32(&lb, logits.dtype);

    // Top-k experts.
    let mut order: Vec<usize> = (0..n_experts).collect();
    order.sort_by(|&a, &b| lf[b].total_cmp(&lf[a]));
    let top: Vec<usize> = order.into_iter().take(w.top_k).collect();

    // Routing weights: softmax over the selected logits (or raw).
    let weights: Vec<f32> = if w.norm_topk {
        let m = top.iter().map(|&i| lf[i]).fold(f32::MIN, f32::max);
        let e: Vec<f32> = top.iter().map(|&i| (lf[i] - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|x| x / s).collect()
    } else {
        top.iter().map(|&i| lf[i]).collect()
    };

    // Accumulate the weighted expert outputs.
    let mut acc = vec![0.0f32; hidden];
    for (&e, &gw) in top.iter().zip(&weights) {
        let ex = &w.experts[e];
        let gate = ops::gemv(dev, &ex.gate, h)?;
        let up = ops::gemv(dev, &ex.up, h)?;
        let act = ops::swiglu(dev, &gate, &up)?;
        let out = ops::gemv(dev, &ex.down, &act)?;
        dev.synchronize()?;
        let mut ob = vec![0u8; hidden * out.dtype.size_bytes()];
        dev.download(out.buffer.as_ref(), &mut ob)?;
        let of = to_f32(&ob, out.dtype);
        for i in 0..hidden {
            acc[i] += gw * of[i];
        }
    }

    if acc.is_empty() {
        return Err(Error::Msg("moe_mlp: no experts".into()));
    }
    Ok(Tensor::new(dev.upload(&from_f32(&acc, h.dtype))?, vec![hidden], h.dtype))
}
