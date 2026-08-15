// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Uninstrumented steady-state decode throughput (tok/s), single-batch,
//! post-fix sanity number. Not a perf-tuning target for this port (see
//! qwen35.rs module doc: decode-first, correctness-first).
use wh_butter_cuda::CudaDevice;
use wh_butter_loader::gguf::Gguf;
use wh_butter_models::gguf_tokenizer::GgufTokenizer;
use wh_butter_models::qwen35;

#[test]
fn qwen35_bench_decode_tok_s() {
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
    let model = qwen35::load_gguf(d, &g).expect("load_gguf");

    let ids = tok.encode("The capital of France is");
    let mut cache = qwen35::Qwen35Cache::new(d, &model.cfg, 256).expect("cache");
    let mut pos = 0u32;
    let mut last_logits = Vec::new();
    let download = |dev: &dyn wh_butter_core::Device, t: &wh_butter_core::Tensor, vocab: usize| -> Vec<f32> {
        let mut b = vec![0u8; vocab * 4];
        dev.synchronize().unwrap();
        dev.download(t.buffer.as_ref(), &mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    for &tid in &ids {
        let logits = qwen35::decode_step(d, &model, &mut cache, tid, pos).expect("decode_step");
        last_logits = download(d, &logits, model.cfg.vocab);
        pos += 1;
    }
    let mut next = last_logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0 as u32;

    let n_steps = 64usize;
    let t0 = std::time::Instant::now();
    for _ in 0..n_steps {
        let logits = qwen35::decode_step(d, &model, &mut cache, next, pos).expect("decode_step");
        let lv = download(d, &logits, model.cfg.vocab);
        next = lv.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0 as u32;
        pos += 1;
    }
    let dt = t0.elapsed();
    let tok_s = n_steps as f64 / dt.as_secs_f64();
    eprintln!("QWEN35_BENCH: {n_steps} decode steps in {:.3}s => {:.2} tok/s (uninstrumented, includes per-step host download+argmax)", dt.as_secs_f64(), tok_s);
}
