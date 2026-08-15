// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! F-85 round-12 quality gate (logit-fidelity probe): greedy-decode prompt
//! 01 for 20 steps, logging the top-1 token id + top-1/top-2 logit margin
//! at each step. Run once per `QWEN35_MARLIN_G` value and diff the printed
//! token-id sequences by hand (or `diff` the two `--nocapture` logs) --
//! this harness itself has no baseline to compare against (env-var-driven,
//! one process = one group size), matching how `qwen35_oracle_check`'s
//! candidates are compared out-of-process via `compare.sh`.
use wh_butter_cuda::CudaDevice;
use wh_butter_loader::gguf::Gguf;
use wh_butter_models::gguf_tokenizer::GgufTokenizer;
use wh_butter_models::qwen35;

fn top1_top2(logits: &[f32]) -> (u32, f32, f32) {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut second_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            second_v = best_v;
            best_v = v;
            best_i = i;
        } else if v > second_v {
            second_v = v;
        }
    }
    (best_i as u32, best_v, second_v)
}

#[test]
fn qwen35_marlin_g_logit_probe() {
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

    let gs = std::env::var("QWEN35_MARLIN_G").unwrap_or_else(|_| "32".to_string());
    let prompt_path = "/home/pidtom/qwen36-oracle/prompts/01_factual.txt";
    let prompt = std::fs::read_to_string(prompt_path).expect("read prompt");
    let prompt = prompt.strip_suffix('\n').unwrap_or(&prompt);
    let ids = tok.encode(prompt);

    let mut cache = qwen35::Qwen35Cache::new(d, &model.cfg, (ids.len() + 32).max(256)).expect("cache");
    let mut pos = 0u32;
    let download = |dev: &dyn wh_butter_core::Device, t: &wh_butter_core::Tensor, vocab: usize| -> Vec<f32> {
        let mut b = vec![0u8; vocab * 4];
        dev.synchronize().unwrap();
        dev.download(t.buffer.as_ref(), &mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let mut last_logits = Vec::new();
    for &tid in &ids {
        let logits = qwen35::decode_step(d, &model, &mut cache, tid, pos).expect("decode_step");
        last_logits = download(d, &logits, model.cfg.vocab);
        pos += 1;
    }
    let (mut next, mut v1, mut v2) = top1_top2(&last_logits);
    eprintln!("QWEN35_MARLIN_G_LOGIT_PROBE QWEN35_MARLIN_G={gs} prompt=01_factual");
    for step in 0..20usize {
        eprintln!(
            "  step={step:2} top1_id={next:6} margin={:.4}",
            v1 - v2
        );
        let logits = qwen35::decode_step(d, &model, &mut cache, next, pos).expect("decode_step");
        let lv = download(d, &logits, model.cfg.vocab);
        let (n2, a, b) = top1_top2(&lv);
        next = n2;
        v1 = a;
        v2 = b;
        pos += 1;
    }
}
