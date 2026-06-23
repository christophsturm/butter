// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Diagnostic: dump the top-K logits + the rank of " Paris" at the first
//! decode step, to tell a precision near-tie apart from a real kernel bug on
//! the Vulkan/RDNA4 path. Not a CI test — gated on QWEN25_GGUF + vulkan.

use ffai_loader::gguf::Gguf;
use ffai_models::gguf_tokenizer::GgufTokenizer;
use ffai_models::llama::GgufModel;
use ffai_vulkan::VulkanDevice;

#[test]
fn qwen25_first_logit_diag_vulkan() {
    let Some(dev) = VulkanDevice::create().expect("vulkan init") else {
        eprintln!("no Vulkan device — skipping");
        return;
    };
    let path = std::env::var("QWEN25_GGUF")
        .unwrap_or_else(|_| r"C:\models\qwen2.5-1.5b-instruct-q8_0.gguf".to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("model not found — skipping");
        return;
    }
    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tok");
    let model = GgufModel::open(dev.as_ref(), &path, 256).expect("load");

    let prompt = tok.encode("The capital of France is");
    eprintln!("[TOK] encoded ids = {prompt:?}");
    for &t in &prompt {
        eprintln!("[TOK]   id={t} -> {:?}", tok.decode(&[t]));
    }
    eprintln!("[CFG] hidden={} nq={} nkv={} hd={} n_layers={} vocab={}",
        model.cfg.hidden, model.cfg.n_q_heads, model.cfg.n_kv_heads,
        model.cfg.head_dim, model.n_layers, model.vocab);
    // Run prefill, keep the logits from the LAST prompt token (the next-token dist).
    let mut logits = vec![];
    for (pos, &t) in prompt.iter().enumerate() {
        logits = model.step(dev.as_ref(), t, pos).expect("step");
    }
    // Argmax + top-8.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    eprintln!("== top-8 next-token logits after prompt ==");
    for &i in idx.iter().take(8) {
        eprintln!("  id={i:>6}  logit={:>10.4}  tok={:?}", logits[i], tok.decode(&[i as u32]));
    }
    for cand in [" Paris", "Paris"] {
        if let Some(pid) = tok.token_id(cand) {
            let rank = idx.iter().position(|&i| i == pid as usize).unwrap();
            eprintln!("  candidate {cand:?}: id={pid} logit={:.4} rank={rank}", logits[pid as usize]);
        } else {
            eprintln!("  candidate {cand:?}: NOT a single token");
        }
    }
}
