// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Run a REAL model — Qwen3-0.6B, BF16 weights from disk — through the
//! shared Rust engine on CUDA, and print the next-token prediction. A
//! single-token forward (the token attends to itself at pos 0) is exactly
//! HF's 1-token forward, so the argmax here is directly comparable to
//! `transformers` for `input_ids=[token]`.
//!
//! Run:  QWEN3_PATH=~/Qwen3-0.6B-hf/model.safetensors \
//!       cargo test -p wh-butter-cuda --features cuda --test qwen3_real -- --nocapture
#![cfg(feature = "cuda")]

use wh_butter_cuda::CudaDevice;
use wh_butter_loader::SafeTensors;
use wh_butter_models::llama::{LlamaConfig, forward_single, load_qwen3};

/// BF16 little-endian bytes → f32.
fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

#[test]
fn qwen3_0_6b_real_forward_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };
    let path = std::env::var("QWEN3_PATH")
        .unwrap_or_else(|_| "/home/pidtom/Qwen3-0.6B-hf/model.safetensors".to_string());
    eprintln!("loading {path}");
    let Ok(st) = SafeTensors::open(&path) else {
        eprintln!("no model at {path} — skipping");
        return;
    };

    let cfg = LlamaConfig {
        hidden: 1024,
        n_q_heads: 16,
        n_kv_heads: 8,
        head_dim: 128,
        intermediate: 3072,
        rope_theta: 1_000_000.0,
        eps: 1e-6,
        qk_norm: true,
        attn_bias: false,
    };
    const N_LAYERS: usize = 28;
    const VOCAB: usize = 151936;

    let mw = load_qwen3(dev.as_ref(), &st, &cfg, N_LAYERS).expect("load qwen3");
    eprintln!("model loaded ({N_LAYERS} layers, vocab {VOCAB})");

    // Token to condition on (override with TOK=…). 9707 = "Hello" in Qwen.
    let token: u32 = std::env::var("TOK").ok().and_then(|s| s.parse().ok()).unwrap_or(9707);

    let logits = forward_single(dev.as_ref(), &cfg, &mw, token).expect("forward");
    dev.synchronize().unwrap();

    // Output dtype = lm_head dtype = BF16.
    let mut lb = vec![0u8; VOCAB * 2];
    dev.download(logits.buffer.as_ref(), &mut lb).unwrap();
    let l = bf16_to_f32(&lb);
    assert_eq!(l.len(), VOCAB);
    assert!(l.iter().all(|x| x.is_finite()), "non-finite logits");

    // Top-5 predicted tokens.
    let mut idx: Vec<usize> = (0..VOCAB).collect();
    idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
    eprintln!("input token {token} → top-5 next-token logits:");
    for &i in idx.iter().take(5) {
        eprintln!("  id {i:>6}  logit {:.4}", l[i]);
    }
    eprintln!("ARGMAX next token = {} (logit {:.4})", idx[0], l[idx[0]]);
    eprintln!("✅ Qwen3-0.6B (real BF16 weights) ran through the shared Rust engine on GB10.");
}
