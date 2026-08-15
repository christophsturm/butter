// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Greedy-decode the 3 oracle prompts (~/qwen36-oracle/prompts) and write
//! plain continuation text to ~/qwen36-oracle/candidates/<id>.txt for
//! compare.sh. temp0/top-k1 (argmax) greedy, 64 tokens, EOS-aware.
use wh_butter_cuda::CudaDevice;
use wh_butter_loader::gguf::Gguf;
use wh_butter_models::gguf_tokenizer::GgufTokenizer;
use wh_butter_models::qwen35;

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v { best_v = v; best_i = i; }
    }
    best_i as u32
}

fn run_one(dev: &dyn wh_butter_core::Device, model: &qwen35::Qwen35Model, tok: &GgufTokenizer, eos: Option<u32>, prompt: &str, max_gen: usize) -> String {
    let ids = tok.encode(prompt);
    let mut cache = qwen35::Qwen35Cache::new(dev, &model.cfg, (ids.len() + max_gen + 8).max(256)).expect("cache");
    let mut pos = 0u32;
    let mut last_logits = Vec::new();
    let download = |dev: &dyn wh_butter_core::Device, t: &wh_butter_core::Tensor, vocab: usize| -> Vec<f32> {
        let mut b = vec![0u8; vocab * 4];
        dev.synchronize().unwrap();
        dev.download(t.buffer.as_ref(), &mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    for &tid in &ids {
        let logits = qwen35::decode_step(dev, model, &mut cache, tid, pos).expect("decode_step");
        last_logits = download(dev, &logits, model.cfg.vocab);
        pos += 1;
    }
    let mut next = argmax(&last_logits);
    let mut generated = Vec::new();
    for _ in 0..max_gen {
        if Some(next) == eos { break; }
        generated.push(next);
        let logits = qwen35::decode_step(dev, model, &mut cache, next, pos).expect("decode_step");
        let lv = download(dev, &logits, model.cfg.vocab);
        next = argmax(&lv);
        pos += 1;
    }
    tok.decode(&generated)
}

#[test]
fn qwen35_oracle_check() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let path = std::env::var("BUTTER_QWEN35_GGUF")
        .unwrap_or_else(|_| "/home/pidtom/models/qwen36-27b/Qwen3.6-27B-Q4_K_M.gguf".to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[skipped] GGUF not found at {path}");
        return;
    }
    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");
    let eos = g.meta_u32("tokenizer.ggml.eos_token_id");
    eprintln!("eos_token_id={eos:?}");
    let model = qwen35::load_gguf(d, &g).expect("load_gguf");

    std::fs::create_dir_all("/home/pidtom/qwen36-oracle/candidates").ok();

    let cases: &[(&str, &str, usize)] = &[
        ("01_factual", "/home/pidtom/qwen36-oracle/prompts/01_factual.txt", 64),
        ("02_code", "/home/pidtom/qwen36-oracle/prompts/02_code.txt", 24),
        ("03_longer", "/home/pidtom/qwen36-oracle/prompts/03_longer.txt", 64),
    ];
    for (id, path, max_gen) in cases {
        let prompt = std::fs::read_to_string(path).expect("read prompt");
        // Prompts are stored with a trailing newline; strip exactly one to
        // match how the oracle almost certainly fed them (no doubled NL).
        let prompt = prompt.strip_suffix('\n').unwrap_or(&prompt);
        let out = run_one(d, &model, &tok, eos, prompt, *max_gen);
        eprintln!("=== {id} ===\n{out:?}\n");
        let out_path = format!("/home/pidtom/qwen36-oracle/candidates/{id}.txt");
        std::fs::write(&out_path, &out).expect("write candidate");
    }
}
