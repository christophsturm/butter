// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! F-85 prefill wave -- LONG-CONTEXT validation for the batched
//! chunked-prefill path (`qwen35::prefill`), extending
//! `qwen35_prefill_correctness.rs` (which only covers short oracle
//! prompts) and `qwen35_bench_prefill.rs` (which only covers pp512/pp2048)
//! out to 8K/16K depth:
//!
//!   1. NEEDLE: a fact planted near the start of an ~8K / ~16K token
//!      prompt (real repo-doc text) must be retrievable by a 32-token
//!      greedy continuation through the DEFAULT (chunk=0 -> 512-row)
//!      chunked prefill path -- the same path `butter generate` uses.
//!   2. CONSISTENCY AT DEPTH: prefill(~8K tokens, chunk=2048) vs
//!      prefill(~8K tokens, chunk=1024) must produce IDENTICAL greedy
//!      continuations -- state-carry proof at depth (4 chunks vs 8
//!      chunks through both the GDN recurrence/conv state AND the
//!      full-attn KV cache).
//!   3. RAGGED CHUNK BOUNDARY: prefill(~10K tokens, chunk=2048) -- last
//!      chunk is NOT a full 2048 rows (10000 % 2048 != 0) -- vs
//!      prefill(~10K tokens, chunk=len) (single chunk) must be IDENTICAL.
//!   4. THROUGHPUT CURVE: single-chunk prefill (chunk == prompt length,
//!      mirroring `qwen35_bench_prefill.rs`'s pp512/pp2048 convention) at
//!      512/2048/4096/8192/16384 tokens, 2 runs each, to see where the
//!      O(T^2)-within-chunk attention term bends the curve away from the
//!      GDN layers' O(T) chunk cost.
//!   5. DECODE-AFTER-PREFILL: 32-token greedy decode tok/s measured right
//!      after each depth point's 2nd prefill run, to quantify the KV-
//!      length effect on the 16 full-attn layers' decode cost.
//!
//! Real GGUF, real forward, uninstrumented wall-clock (matches this
//! file's siblings' conventions). Loads the model ONCE (real-checkpoint
//! load is ~2min for the 27B Q4_K_M GGUF) and reuses it across every
//! check below, instead of paying the load cost per check.
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

/// Batched `prefill()` at the given chunk size, then greedy decode
/// continuation. Mirrors `qwen35_prefill_correctness.rs::prefill_continuation`.
fn prefill_continuation(
    dev: &dyn Device, model: &qwen35::Qwen35Model, eos: Option<u32>, ids: &[u32], max_gen: usize, max_seq: usize, chunk: usize,
) -> Vec<u32> {
    let mut cache = qwen35::Qwen35Cache::new(dev, &model.cfg, max_seq).expect("cache");
    let logits = qwen35::prefill(dev, model, &mut cache, ids, chunk).expect("prefill");
    let last_logits = download(dev, &logits, model.cfg.vocab);
    let pos = ids.len() as u32;
    greedy_continue(dev, model, &mut cache, eos, &last_logits, pos, max_gen)
}

fn free_report(tag: &str) {
    if let Ok(out) = std::process::Command::new("free").arg("-g").output() {
        eprintln!("[mem@{tag}]\n{}", String::from_utf8_lossy(&out.stdout));
    }
}

/// Take the first `n_chars` chars of `s` (char-safe, not byte-safe --
/// avoids splitting a multi-byte UTF-8 sequence in the box-drawing/arrow
/// glyphs this codebase's doc comments use).
fn take_chars(chars: &[char], n_chars: usize) -> String {
    chars[..n_chars.min(chars.len())].iter().collect()
}

#[test]
fn qwen35_prefill_depth_validation() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let path = std::env::var("BUTTER_QWEN35_GGUF")
        .unwrap_or_else(|_| "/home/pidtom/models/qwen36-27b/Qwen3.6-27B-Q4_K_M.gguf".to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[skipped] GGUF not found at {path}");
        return;
    }
    let corpus_path = std::env::var("QWEN35_DEPTH_CORPUS")
        .unwrap_or_else(|_| "/home/pidtom/qwen35-depth-test/corpus_with_needle.txt".to_string());
    if !std::path::Path::new(&corpus_path).exists() {
        eprintln!("[skipped] corpus not found at {corpus_path}");
        return;
    }

    free_report("start");
    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");
    let eos = g.meta_u32("tokenizer.ggml.eos_token_id");
    let t_load = std::time::Instant::now();
    let model = qwen35::load_gguf(d, &g).expect("load_gguf");
    eprintln!("[load] {:.3}s", t_load.elapsed().as_secs_f64());
    free_report("after-load");

    let corpus = std::fs::read_to_string(&corpus_path).expect("read corpus");
    let chars: Vec<char> = corpus.chars().collect();
    // Calibrated on this exact corpus: ~3.12 chars/token; over-sample a
    // touch (3.3) and let the tokenizer's actual count be authoritative
    // (logged per-point below) rather than forcing an exact token count.
    let chars_per_tok = 3.3;
    let question = "\n\nQuestion: What is the secret code mentioned above? Answer:";

    let mut all_pass = true;

    // ---------------- 1. NEEDLE, default (production) chunking ----------------
    for &target in &[8000usize, 16000usize] {
        let n_chars = (target as f64 * chars_per_tok) as usize;
        let slice = take_chars(&chars, n_chars);
        let prompt = format!("{slice}{question}");
        let ids = tok.encode(&prompt);
        let max_gen = 32usize;
        let max_seq = ids.len() + max_gen + 8;
        let t0 = std::time::Instant::now();
        let out = prefill_continuation(d, &model, eos, &ids, max_gen, max_seq, 0 /* default chunk */);
        let dt = t0.elapsed().as_secs_f64();
        let text = tok.decode(&out);
        let hit = text.contains("7734");
        eprintln!(
            "[needle target~{target} actual={} tok] {:.3}s continuation={:?} HIT={}",
            ids.len(), dt, text, hit
        );
        if !hit { all_pass = false; }
    }

    // ---------------- 2. CONSISTENCY AT DEPTH: chunk 2048 vs 1024 @ ~8K ----------------
    let n_chars_8k = (8192.0 * chars_per_tok) as usize;
    let slice_8k = take_chars(&chars, n_chars_8k);
    let ids_8k = tok.encode(&slice_8k);
    let max_gen = 32usize;
    let max_seq_8k = ids_8k.len() + max_gen + 8;
    let c2048 = prefill_continuation(d, &model, eos, &ids_8k, max_gen, max_seq_8k, 2048);
    let c1024 = prefill_continuation(d, &model, eos, &ids_8k, max_gen, max_seq_8k, 1024);
    let consistency_pass = c2048 == c1024;
    eprintln!(
        "[consistency @ {} tok] chunk2048(4x)-vs-chunk1024(8x) IDENTICAL={}",
        ids_8k.len(), consistency_pass
    );
    if !consistency_pass {
        eprintln!("  c2048: {c2048:?}");
        eprintln!("  c1024: {c1024:?}");
        let first_diverge = c2048.iter().zip(c1024.iter()).position(|(a, b)| a != b);
        eprintln!("  first divergence at index: {first_diverge:?}");
        all_pass = false;
    }

    // ---------------- 3. RAGGED CHUNK BOUNDARY @ ~10K ----------------
    let n_chars_10k = (10000.0 * chars_per_tok) as usize;
    let slice_10k = take_chars(&chars, n_chars_10k);
    let ids_10k = tok.encode(&slice_10k);
    let max_seq_10k = ids_10k.len() + max_gen + 8;
    let single = prefill_continuation(d, &model, eos, &ids_10k, max_gen, max_seq_10k, ids_10k.len());
    let ragged = prefill_continuation(d, &model, eos, &ids_10k, max_gen, max_seq_10k, 2048);
    let ragged_pass = single == ragged;
    eprintln!(
        "[ragged @ {} tok, tail={}] singlechunk-vs-chunk2048(ragged tail) IDENTICAL={}",
        ids_10k.len(), ids_10k.len() % 2048, ragged_pass
    );
    if !ragged_pass {
        eprintln!("  single: {single:?}");
        eprintln!("  ragged: {ragged:?}");
        let first_diverge = single.iter().zip(ragged.iter()).position(|(a, b)| a != b);
        eprintln!("  first divergence at index: {first_diverge:?}");
        all_pass = false;
    }

    // ---------------- 4+5. THROUGHPUT CURVE + DECODE-AFTER-PREFILL ----------------
    println!("depth_target,depth_actual,run,prefill_s,prefill_tok_s,decode32_s,decode_tok_s");
    for &n_target in &[512usize, 2048, 4096, 8192, 16384] {
        let n_chars_n = (n_target as f64 * chars_per_tok) as usize;
        let slice_n = take_chars(&chars, n_chars_n);
        let ids_n = tok.encode(&slice_n);
        let n_actual = ids_n.len();
        let max_seq_n = n_actual + 40;

        for run in 1..=2 {
            let mut cache = qwen35::Qwen35Cache::new(d, &model.cfg, max_seq_n).expect("cache");
            let t0 = std::time::Instant::now();
            let logits = qwen35::prefill(d, &model, &mut cache, &ids_n, n_actual).expect("prefill");
            d.synchronize().unwrap();
            let dt = t0.elapsed().as_secs_f64();
            let tok_s = n_actual as f64 / dt;
            eprintln!("[pp{n_target}(actual={n_actual})] run{run}: {dt:.3}s => {tok_s:.2} tok/s");

            // Decode-after-prefill: 32 greedy steps right after this
            // prefill, only on the 2nd (steady-state) run per point.
            let (mut decode_s, mut decode_tok_s) = (0.0f64, 0.0f64);
            if run == 2 {
                let last_logits = download(d, &logits, model.cfg.vocab);
                let mut next = argmax(&last_logits);
                let mut pos = n_actual as u32;
                let steps = 32usize;
                let t1 = std::time::Instant::now();
                for _ in 0..steps {
                    let lg = qwen35::decode_step(d, &model, &mut cache, next, pos).expect("decode_step");
                    let lv = download(d, &lg, model.cfg.vocab);
                    next = argmax(&lv);
                    pos += 1;
                }
                decode_s = t1.elapsed().as_secs_f64();
                decode_tok_s = steps as f64 / decode_s;
                eprintln!(
                    "[decode-after-pp{n_target}] {steps} steps in {decode_s:.3}s => {decode_tok_s:.2} tok/s"
                );
                if n_target == 16384 {
                    free_report("after-16K-prefill+decode");
                }
            }
            println!("{n_target},{n_actual},{run},{dt:.3},{tok_s:.3},{decode_s:.3},{decode_tok_s:.3}");
        }
    }

    free_report("end");

    assert!(all_pass, "qwen35 depth validation FAILED -- see stderr above (needle miss and/or consistency divergence)");
}
