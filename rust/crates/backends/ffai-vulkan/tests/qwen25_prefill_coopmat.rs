// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Batched-prefill correctness + throughput on the Vulkan/RDNA4 backend.
//!
//! Drives `GgufModel::prefill` (the multi-token path that routes every
//! projection + FFN matmul through `ffai_ops::gemm_q8_mpp` — the SimdGroup
//! CoopTile GEMM) and:
//!   1. validates the next-token logits' argmax matches the sequential `step`
//!      path AND that a short greedy continuation still says "Paris", and
//!   2. measures prefill throughput (prompt tokens / second) at a few prompt
//!      lengths.
//!
//! Run it twice to A/B the gated `VK_KHR_cooperative_matrix` lever:
//!   # scalar GEMM (SoftwareLocalC):   cargo test ... -- --nocapture
//!   # coopmat GEMM (fragment MMA):    MT_VK_COOPMAT=1 cargo test ... -- --nocapture
//! Both must produce identical tokens (the gate is bit-for-bit transparent to
//! the model); only the prefill tok/s changes.
//!
//! Requires `--features vulkan`. Skips if no Vulkan device or model is absent.

use ffai_loader::gguf::Gguf;
use ffai_models::gguf_tokenizer::GgufTokenizer;
use ffai_models::llama::GgufModel;
use ffai_vulkan::VulkanDevice;
use std::time::Instant;

#[test]
fn qwen25_1_5b_prefill_coopmat_vulkan() {
    let Some(dev) = VulkanDevice::create().expect("vulkan init") else {
        eprintln!("no Vulkan device — skipping");
        return;
    };
    let path = std::env::var("QWEN25_GGUF")
        .unwrap_or_else(|_| r"C:\models\qwen2.5-1.5b-instruct-q8_0.gguf".to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("model not found at {path} — skipping");
        return;
    }
    let coopmat = std::env::var("MT_VK_COOPMAT").map(|v| v == "1").unwrap_or(false);
    eprintln!("=== Qwen2.5-1.5B prefill — MT_VK_COOPMAT={} ===", coopmat as u8);

    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");

    // KV cache must hold the longest prefill we time below + a short decode tail.
    let cap = 640usize;
    let model = GgufModel::load(dev.as_ref(), &path, cap).expect("load gguf model (resident-Q8)");
    eprintln!(
        "loaded: layers={} hidden={} q_heads={} kv_heads={} head_dim={} bias={}",
        model.n_layers, model.cfg.hidden, model.cfg.n_q_heads, model.cfg.n_kv_heads,
        model.cfg.head_dim, model.cfg.attn_bias,
    );

    // ── (1) Correctness: batched prefill vs sequential step ──────────────────
    let prompt_text = "The capital of France is";
    let prompt = tok.encode(prompt_text);
    assert!(prompt.len() > 1, "need a multi-token prompt to exercise the batched path");

    // Sequential reference next-token logits (the proven per-token path).
    let mut seq_logits = Vec::new();
    for (i, &t) in prompt.iter().enumerate() {
        seq_logits = model.step(dev.as_ref(), t, i).expect("step");
    }
    let seq_argmax = argmax(&seq_logits);

    // Fresh model so the KV cache for the batched run is clean.
    let model2 = GgufModel::load(dev.as_ref(), &path, cap).expect("reload");
    let pf_logits = model2.prefill(dev.as_ref(), &prompt, 0).expect("prefill");
    let pf_argmax = argmax(&pf_logits);

    eprintln!(
        "next-token argmax — sequential={} ({:?})  batched-prefill={} ({:?})",
        seq_argmax, tok.decode(&[seq_argmax as u32]),
        pf_argmax, tok.decode(&[pf_argmax as u32]),
    );
    assert_eq!(
        seq_argmax, pf_argmax,
        "batched prefill next-token must match the sequential step path"
    );
    let first_tok_text = tok.decode(&[pf_argmax as u32]);
    assert!(
        first_tok_text.contains("Paris"),
        "expected the prefill's first predicted token to be 'Paris', got {first_tok_text:?}"
    );

    // Continue greedily a few tokens from the prefilled cache (decode via step).
    let mut next = pf_argmax as u32;
    let mut produced = vec![next];
    for i in 0..5 {
        let logits = model2.step(dev.as_ref(), next, prompt.len() + i).expect("decode step");
        next = argmax(&logits) as u32;
        produced.push(next);
    }
    eprintln!("FULL: {:?}{}", prompt_text, tok.decode(&produced));

    // ── (2) Throughput: prefill tok/s at a few prompt lengths ────────────────
    // Synthetic prompts (repeat the real prompt) so we can scale S without
    // depending on a long corpus. Each timing reloads the model so the KV cache
    // starts empty and the measurement is pure prefill compute (incl. dispatch).
    eprintln!("--- prefill throughput (MT_VK_COOPMAT={}) ---", coopmat as u8);
    for &s in &[64usize, 128, 256, 512] {
        if s + 8 > cap { continue; }
        let toks: Vec<u32> = (0..s).map(|i| prompt[i % prompt.len()]).collect();
        let m = GgufModel::load(dev.as_ref(), &path, cap).expect("reload for timing");
        // Warm once (first call JIT-compiles the kernels), then time.
        let _ = m.prefill(dev.as_ref(), &toks[..prompt.len()], 0).expect("warm");
        let m2 = GgufModel::load(dev.as_ref(), &path, cap).expect("reload timed");
        let t = Instant::now();
        let _ = m2.prefill(dev.as_ref(), &toks, 0).expect("timed prefill");
        let secs = t.elapsed().as_secs_f32();
        eprintln!("  S={s:>4}  {secs:7.3}s  {:8.1} prompt-tok/s", s as f32 / secs);
    }
    eprintln!("Qwen2.5-1.5B batched prefill OK on Vulkan/RDNA4 (coopmat={}).", coopmat as u8);
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv { bv = x; bi = i; }
    }
    bi
}
