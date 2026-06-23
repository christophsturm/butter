// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! End-to-end GGUF small-model load + generate on the **Vulkan** GPU (RDNA4).
//! Mirror of `ffai-metal/tests/qwen25_gguf_metal.rs` — the only difference is
//! the device constructor (`VulkanDevice::create()` instead of Metal). Proves
//! the backend-agnostic GGUF path (GgufModel::open + ffai_runtime::generate
//! over GgufModel::step) runs on the portable Vulkan backend.
//!
//! Loads Qwen2.5-1.5B-Instruct Q8_0 GGUF (geometry + Q8_0 weights + BPE
//! tokenizer from the one file), runs the KV-cache decode loop, and checks the
//! continuation of "The capital of France is" contains "Paris".
//!
//! Requires `--features vulkan`. Skips if no Vulkan device or model is absent.

use ffai_loader::gguf::Gguf;
use ffai_models::gguf_tokenizer::GgufTokenizer;
use ffai_models::llama::GgufModel;
use ffai_runtime::{generate, Sampling, StopOn};
use ffai_vulkan::VulkanDevice;
use std::time::Instant;

#[test]
fn qwen25_1_5b_gguf_generate_on_vulkan() {
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
    eprintln!("device: {} — loading GGUF {path}", dev.name());

    // Parse once: tokenizer reads the same Gguf; model re-parses for weights.
    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");
    eprintln!(
        "arch={:?} vocab={}",
        g.meta_str("general.architecture"),
        tok.vocab_size()
    );

    let t_load = Instant::now();
    let model = GgufModel::open(dev.as_ref(), &path, 256).expect("load gguf model");
    eprintln!(
        "model loaded in {:.2}s — layers={} hidden={} q_heads={} kv_heads={} head_dim={} bias={} theta={}",
        t_load.elapsed().as_secs_f32(),
        model.n_layers,
        model.cfg.hidden,
        model.cfg.n_q_heads,
        model.cfg.n_kv_heads,
        model.cfg.head_dim,
        model.cfg.attn_bias,
        model.cfg.rope_theta,
    );

    // Plain completion prompt (no chat template) — base-model style continuation.
    let prompt_text = "The capital of France is";
    let prompt = tok.encode(prompt_text);
    eprintln!("prompt {:?} -> {} ids: {:?}", prompt_text, prompt.len(), prompt);
    assert!(!prompt.is_empty(), "tokenizer produced no ids");

    let eos = tok.token_id("<|endoftext|>");
    // Short run keeps the capability demo quick (a few decode tokens is enough
    // to prove the full prefill+decode loop is correct on the Vulkan backend).
    let stop = StopOn { max_new: 6, eos };

    let t_gen = Instant::now();
    let mut n_steps = 0usize;
    let out = generate(&prompt, &stop, &Sampling::Greedy, 0, |token, pos| {
        n_steps += 1;
        model.step(dev.as_ref(), token, pos).expect("step")
    });
    let secs = t_gen.elapsed().as_secs_f32();
    let total_steps = prompt.len() + out.len(); // prefill steps + decode steps
    let _ = n_steps;

    let text = tok.decode(&out);
    eprintln!(
        "generated {} tokens in {:.2}s ({:.2} tok/s incl. prefill)",
        out.len(),
        secs,
        total_steps as f32 / secs
    );
    eprintln!("OUTPUT IDS: {:?}", out);
    eprintln!("OUTPUT: {:?}", text);
    eprintln!("FULL:   {:?}{}", prompt_text, text);

    assert!(
        text.contains("Paris"),
        "expected 'Paris' in continuation, got {text:?}"
    );
    eprintln!("Qwen2.5-1.5B-Q8 GGUF loaded + generated coherent output on Vulkan/RDNA4.");
}
