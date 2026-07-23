// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Real Qwen3-0.6B (BF16) through the shared Rust engine on the **Apple
//! GPU** — the same model that matches HF on CUDA, now on Metal. Proves a
//! real model is verified on BOTH platforms (the argmax must equal HF's
//! 21806 for input "Hello"=9707, the same answer the CUDA path gave).
//!
//! Runs on macOS with a Metal GPU; skips elsewhere.

use ffai_metal::MetalDevice;
use ffai_loader::SafeTensors;
use ffai_models::llama::{LlamaConfig, forward_single, load_qwen3};

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

#[test]
fn qwen3_0_6b_real_forward_on_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let path = std::env::var("QWEN3_PATH")
        .unwrap_or_else(|_| "/Users/tom/models/Qwen3-0.6B-hf/model.safetensors".to_string());
    eprintln!("device: {} — loading {path}", dev.name());
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
    eprintln!("model loaded; running forward on Apple GPU…");

    let token: u32 = std::env::var("TOK").ok().and_then(|s| s.parse().ok()).unwrap_or(9707);
    let logits = forward_single(dev.as_ref(), &cfg, &mw, token).expect("forward");
    dev.synchronize().unwrap();

    let mut lb = vec![0u8; VOCAB * 2];
    dev.download(logits.buffer.as_ref(), &mut lb).unwrap();
    let l = bf16_to_f32(&lb);

    let mut idx: Vec<usize> = (0..VOCAB).collect();
    idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
    eprintln!("input token {token} → top-5 (Metal):");
    for &i in idx.iter().take(5) {
        eprintln!("  id {i:>6}  logit {:.4}", l[i]);
    }
    eprintln!("ARGMAX (Metal) = {}", idx[0]);
    // HF / CUDA both gave 21806 for token 9707.
    if token == 9707 {
        assert_eq!(idx[0], 21806, "Metal argmax disagrees with HF/CUDA (21806)");
        eprintln!("✅ Qwen3-0.6B on Apple GPU predicts 21806 — matches HF and the CUDA run.");
    }
}
