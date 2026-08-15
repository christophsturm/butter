// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Qwen3.5/3.6-27B decode smoke + oracle check on CUDA. Env-gated: set
//! BUTTER_QWEN35_GGUF to the model's GGUF path, otherwise defaults to
//! ~/models/qwen36-27b/Qwen3.6-27B-Q4_K_M.gguf. Mirrors laguna.rs's test
//! file location/pattern (`crates/backends/wh-butter-cuda/tests/`).
use wh_butter_cuda::CudaDevice;
use wh_butter_loader::gguf::Gguf;
use wh_butter_models::gguf_tokenizer::GgufTokenizer;
use wh_butter_models::qwen35;

fn gguf_path() -> String {
    std::env::var("BUTTER_QWEN35_GGUF")
        .unwrap_or_else(|_| "/home/pidtom/models/qwen36-27b/Qwen3.6-27B-Q4_K_M.gguf".to_string())
}

fn argmax(logits: &[f32]) -> (usize, f32) {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    (best_i, best_v)
}

/// Load + build a decode cache + walk one prompt through `decode_step`,
/// greedily generating `n_gen` more tokens. Returns (prompt_ids,
/// generated_ids). Every step is downloaded + checked for NaN so a bad
/// layer fails loudly instead of silently producing garbage tokens.
fn run_prompt(
    dev: &dyn wh_butter_core::Device,
    model: &qwen35::Qwen35Model,
    tok: &GgufTokenizer,
    prompt: &str,
    n_gen: usize,
    max_seq: usize,
) -> (Vec<u32>, Vec<u32>) {
    let prompt_ids = tok.encode(prompt);
    eprintln!("prompt {prompt:?} -> {} tokens: {:?}", prompt_ids.len(), prompt_ids);
    let mut cache = qwen35::Qwen35Cache::new(dev, &model.cfg, max_seq).expect("cache alloc");

    let mut generated = Vec::with_capacity(n_gen);
    let mut pos = 0u32;
    let mut last_logits: Vec<f32> = Vec::new();

    let download_logits = |dev: &dyn wh_butter_core::Device, t: &wh_butter_core::Tensor, vocab: usize| -> Vec<f32> {
        let mut b = vec![0u8; vocab * 4];
        dev.synchronize().expect("sync");
        dev.download(t.buffer.as_ref(), &mut b).expect("download");
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };

    for &tid in &prompt_ids {
        let logits = qwen35::decode_step(dev, model, &mut cache, tid, pos).expect("decode_step (prompt)");
        last_logits = download_logits(dev, &logits, model.cfg.vocab);
        let n_nan = last_logits.iter().filter(|v| v.is_nan()).count();
        assert_eq!(n_nan, 0, "NaN in logits at prompt pos {pos} (token {tid})");
        pos += 1;
    }

    let mut next_tid = argmax(&last_logits).0 as u32;
    for _ in 0..n_gen {
        generated.push(next_tid);
        let logits = qwen35::decode_step(dev, model, &mut cache, next_tid, pos).expect("decode_step (gen)");
        let lv = download_logits(dev, &logits, model.cfg.vocab);
        let n_nan = lv.iter().filter(|v| v.is_nan()).count();
        assert_eq!(n_nan, 0, "NaN in logits at gen pos {pos}");
        let (id, val) = argmax(&lv);
        eprintln!("  pos {pos}: argmax={id} logit={val:.3} tok={:?}", tok.decode(&[id as u32]));
        next_tid = id as u32;
        pos += 1;
    }

    (prompt_ids, generated)
}

#[test]
fn qwen35_load_and_decode_smoke() {
    let Some(dev) = CudaDevice::create().expect("cuda") else {
        eprintln!("no CUDA, skip");
        return;
    };
    let d = dev.as_ref();
    let path = gguf_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("[skipped] GGUF not found at {path} (set BUTTER_QWEN35_GGUF)");
        return;
    }
    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");
    eprintln!("vocab_size (tokenizer) = {}", tok.vocab_size());

    let model = qwen35::load_gguf(d, &g).expect("load_gguf");
    eprintln!(
        "cfg: hidden={} n_layers={} n_head={} n_kv_heads={} head_dim={} vocab={} gdn(k_heads={} v_heads={} kdim={} vdim={})",
        model.cfg.hidden, model.cfg.n_layers, model.cfg.n_head, model.cfg.n_kv_heads, model.cfg.head_dim, model.cfg.vocab,
        model.cfg.gdn_num_k_heads, model.cfg.gdn_num_v_heads, model.cfg.gdn_key_head_dim, model.cfg.gdn_value_head_dim,
    );

    let (prompt_ids, gen1) = run_prompt(d, &model, &tok, "The capital of France is", 32, 256);
    eprintln!("prompt1 ids={prompt_ids:?}\ngenerated1={gen1:?}\ndecoded1={:?}", tok.decode(&gen1));

    let (prompt_ids2, gen2) = run_prompt(d, &model, &tok, "def fibonacci(n):", 32, 256);
    eprintln!("prompt2 ids={prompt_ids2:?}\ngenerated2={gen2:?}\ndecoded2={:?}", tok.decode(&gen2));
}
