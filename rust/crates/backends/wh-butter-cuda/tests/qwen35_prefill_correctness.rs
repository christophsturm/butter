// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! F-85 prefill wave -- correctness bar for the batched chunked-prefill
//! path (`qwen35::prefill`), per the task brief:
//!
//!   1. IDENTITY: token-by-token decode-loop prefill vs `prefill()` +
//!      decode must produce TOKEN-IDENTICAL 64-token greedy continuations
//!      on all 3 oracle prompts (same prompts/harness as
//!      `qwen35_oracle_check.rs`).
//!   2. STATE-CARRY: `prefill()` in 1 chunk vs 2 chunks (forcing an
//!      external-chunk boundary through the GDN recurrence state AND the
//!      causal-conv state) must produce IDENTICAL continuations -- exact
//!      state handoff proof.
use wh_butter_cuda::CudaDevice;
use wh_butter_core::Device;
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

fn download(dev: &dyn Device, t: &wh_butter_core::Tensor, vocab: usize) -> Vec<f32> {
    let mut b = vec![0u8; vocab * 4];
    dev.synchronize().unwrap();
    dev.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Pure token-by-token: every prompt token walked through `decode_step`,
/// then `max_gen` greedy-argmax continuation tokens the same way. This is
/// the pre-existing "prefill via repeated decode" path (module doc), the
/// baseline this task's batched `prefill()` must match.
fn decode_loop_continuation(
    dev: &dyn Device, model: &qwen35::Qwen35Model, eos: Option<u32>, ids: &[u32], max_gen: usize, max_seq: usize,
) -> Vec<u32> {
    let mut cache = qwen35::Qwen35Cache::new(dev, &model.cfg, max_seq).expect("cache");
    let mut pos = 0u32;
    let mut last_logits = Vec::new();
    for &tid in ids {
        let logits = qwen35::decode_step(dev, model, &mut cache, tid, pos).expect("decode_step");
        last_logits = download(dev, &logits, model.cfg.vocab);
        pos += 1;
    }
    greedy_continue(dev, model, &mut cache, eos, &last_logits, pos, max_gen)
}

/// Batched `prefill()` (at the given chunk size) then greedy decode
/// continuation -- the path under test.
fn prefill_continuation(
    dev: &dyn Device, model: &qwen35::Qwen35Model, eos: Option<u32>, ids: &[u32], max_gen: usize, max_seq: usize, chunk: usize,
) -> Vec<u32> {
    let mut cache = qwen35::Qwen35Cache::new(dev, &model.cfg, max_seq).expect("cache");
    let logits = qwen35::prefill(dev, model, &mut cache, ids, chunk).expect("prefill");
    let last_logits = download(dev, &logits, model.cfg.vocab);
    let pos = ids.len() as u32;
    greedy_continue(dev, model, &mut cache, eos, &last_logits, pos, max_gen)
}

fn greedy_continue(
    dev: &dyn Device, model: &qwen35::Qwen35Model, cache: &mut qwen35::Qwen35Cache, eos: Option<u32>,
    first_logits: &[f32], mut pos: u32, max_gen: usize,
) -> Vec<u32> {
    let mut next = argmax(first_logits);
    let mut out = Vec::new();
    for _ in 0..max_gen {
        if Some(next) == eos { break; }
        out.push(next);
        let logits = qwen35::decode_step(dev, model, cache, next, pos).expect("decode_step");
        let lv = download(dev, &logits, model.cfg.vocab);
        next = argmax(&lv);
        pos += 1;
    }
    out
}

#[test]
fn qwen35_prefill_correctness() {
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
    let model = qwen35::load_gguf(d, &g).expect("load_gguf");

    let cases: &[(&str, &str, usize)] = &[
        ("01_factual", "/home/pidtom/qwen36-oracle/prompts/01_factual.txt", 64),
        ("02_code", "/home/pidtom/qwen36-oracle/prompts/02_code.txt", 64),
        ("03_longer", "/home/pidtom/qwen36-oracle/prompts/03_longer.txt", 64),
    ];

    let mut all_pass = true;
    for (id, path, max_gen) in cases {
        let prompt = std::fs::read_to_string(path).expect("read prompt");
        let prompt = prompt.strip_suffix('\n').unwrap_or(&prompt);
        let ids = tok.encode(prompt);
        let max_seq = (ids.len() + max_gen + 8).max(256);

        // ---- Check 1: decode-loop vs prefill(1 chunk == whole prompt) ----
        let baseline = decode_loop_continuation(d, &model, eos, &ids, *max_gen, max_seq);
        let via_prefill_1chunk = prefill_continuation(d, &model, eos, &ids, *max_gen, max_seq, ids.len());

        let identical_1 = baseline == via_prefill_1chunk;
        eprintln!(
            "[{id}] prompt_len={} decode-loop-vs-prefill(1chunk) IDENTICAL={identical_1} (n_tok baseline={} prefill={})",
            ids.len(), baseline.len(), via_prefill_1chunk.len()
        );
        if !identical_1 {
            eprintln!("  baseline : {baseline:?}");
            eprintln!("  prefill1 : {via_prefill_1chunk:?}");
            let first_diverge = baseline.iter().zip(via_prefill_1chunk.iter()).position(|(a, b)| a != b);
            eprintln!("  first divergence at index: {first_diverge:?}");
            all_pass = false;
        }

        // ---- Check 2: state-carry -- prefill in 2 chunks vs 1 chunk ----
        // Split roughly in half; also forces the chunk boundary through
        // GDN's conv1d state (conv_kernel-1=3 rows) and recurrence state,
        // not just the full-attn KV cache.
        let half = (ids.len() / 2).max(1);
        let via_prefill_2chunk = prefill_continuation(d, &model, eos, &ids, *max_gen, max_seq, half);
        let identical_2 = via_prefill_1chunk == via_prefill_2chunk;
        eprintln!("[{id}] prefill(1chunk)-vs-prefill(2chunk@{half}) IDENTICAL={identical_2}");
        if !identical_2 {
            eprintln!("  prefill1 : {via_prefill_1chunk:?}");
            eprintln!("  prefill2 : {via_prefill_2chunk:?}");
            let first_diverge = via_prefill_1chunk.iter().zip(via_prefill_2chunk.iter()).position(|(a, b)| a != b);
            eprintln!("  first divergence at index: {first_diverge:?}");
            all_pass = false;
        }

        // ---- Bonus: also check against the true decode-loop baseline ----
        let identical_2v_baseline = baseline == via_prefill_2chunk;
        eprintln!("[{id}] decode-loop-vs-prefill(2chunk@{half}) IDENTICAL={identical_2v_baseline}");

        eprintln!("[{id}] decoded continuation: {:?}\n", tok.decode(&via_prefill_1chunk));
    }

    assert!(all_pass, "qwen35 prefill correctness FAILED -- see stderr above for divergence details");
}
