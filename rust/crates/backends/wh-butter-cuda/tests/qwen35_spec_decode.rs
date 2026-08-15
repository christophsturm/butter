// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! MTP self-speculative decode: correctness bar + acceptance-rate / tok/s
//! ladder across gamma in {2,3,4} (F-85 lever, see `qwen35::speculative_round`'s
//! doc). Mirrors `qwen35_oracle_check.rs`'s 3 oracle prompts.
//!
//! Correctness bar: for each prompt, spec-decode ON output must be
//! token-for-token IDENTICAL to spec-decode OFF (plain `decode_step`) for
//! 64 generated tokens. `speculative_round`'s lazy sequential verify makes
//! this true by construction (see its doc) -- this test is the empirical
//! check that the construction is actually implemented correctly.
//!
//! Requires `BUTTER_QWEN35_GGUF` pointed at the MTP-preserved checkpoint
//! (has `blk.64.nextn.*` tensors) -- the plain (non-MTP) checkpoint this
//! port's other tests default to does NOT carry the drafter, so this test
//! uses its own default path + explicit existence check.
use std::time::Instant;
use wh_butter_core::{Device, Tensor};
use wh_butter_cuda::CudaDevice;
use wh_butter_loader::gguf::Gguf;
use wh_butter_models::gguf_tokenizer::GgufTokenizer;
use wh_butter_models::qwen35::{self, Qwen35Model, SpecDecodeStats};

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

fn download_f32(dev: &dyn Device, t: &Tensor, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    dev.synchronize().unwrap();
    dev.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Plain greedy decode (spec OFF) -- the golden reference. Returns
/// (generated token ids, decode-only wall time for the generated tokens).
fn baseline_generate(
    dev: &dyn Device, model: &Qwen35Model, prompt_ids: &[u32], eos: Option<u32>, max_gen: usize,
) -> (Vec<u32>, f64) {
    let mut cache = qwen35::Qwen35Cache::new(dev, &model.cfg, (prompt_ids.len() + max_gen + 8).max(256)).expect("cache");
    let mut pos = 0u32;
    let mut last_logits = Vec::new();
    for &tid in prompt_ids {
        let logits = qwen35::decode_step(dev, model, &mut cache, tid, pos).expect("decode_step");
        last_logits = download_f32(dev, &logits, model.cfg.vocab);
        pos += 1;
    }
    let mut next = argmax(&last_logits);
    let mut out = Vec::new();
    let t0 = Instant::now();
    for _ in 0..max_gen {
        if Some(next) == eos {
            break;
        }
        out.push(next);
        let logits = qwen35::decode_step(dev, model, &mut cache, next, pos).expect("decode_step");
        let lv = download_f32(dev, &logits, model.cfg.vocab);
        next = argmax(&lv);
        pos += 1;
    }
    (out, t0.elapsed().as_secs_f64())
}

/// Speculative decode (spec ON). Prefills BOTH the backbone and MTP's own
/// KV cache (see `Qwen35SpecCache::warm_mtp`'s doc), then rounds of
/// `speculative_round` until `max_gen` tokens are produced or EOS is
/// emitted within an accepted run. Returns (generated token ids, decode-only
/// wall time, aggregate accept/draft stats across all rounds).
fn spec_generate(
    dev: &dyn Device, model: &Qwen35Model, prompt_ids: &[u32], eos: Option<u32>, max_gen: usize, gamma: usize,
) -> (Vec<u32>, f64, SpecDecodeStats) {
    let mut cache =
        qwen35::Qwen35SpecCache::new(dev, &model.cfg, (prompt_ids.len() + max_gen + 8).max(256)).expect("speccache");
    let mut pos = 0u32;
    let mut xn_last = None;
    let mut last_tok = *prompt_ids.last().expect("nonempty prompt");
    for (i, &tid) in prompt_ids.iter().enumerate() {
        let (_logits, xn) = qwen35::decode_step_ex(dev, model, &mut cache.backbone, tid, pos).expect("decode_step_ex");
        cache.warm_mtp(dev, model, &xn, tid, pos).expect("warm_mtp");
        if i == prompt_ids.len() - 1 {
            xn_last = Some(xn);
        }
        pos += 1;
    }
    let mut xn = xn_last.expect("prompt nonempty");
    let mut cur_pos = pos - 1; // position of `last_tok` (0-based, matches decode_step_ex's `pos` arg for that call)

    let mut out = Vec::new();
    let mut agg = SpecDecodeStats::default();
    let t0 = Instant::now();
    'outer: while out.len() < max_gen {
        let (accepted, new_tok, new_pos, new_xn, stats) =
            qwen35::speculative_round(dev, model, &mut cache, last_tok, cur_pos, &xn, gamma).expect("speculative_round");
        agg.n_drafted += stats.n_drafted;
        agg.n_accepted += stats.n_accepted;
        for &t in &accepted {
            if Some(t) == eos {
                break 'outer;
            }
            out.push(t);
            if out.len() >= max_gen {
                break 'outer;
            }
        }
        last_tok = new_tok;
        cur_pos = new_pos;
        xn = new_xn;
    }
    (out, t0.elapsed().as_secs_f64(), agg)
}

#[test]
fn qwen35_spec_decode_correctness_and_perf() {
    let Some(dev) = CudaDevice::create().expect("cuda") else {
        eprintln!("no CUDA, skip");
        return;
    };
    let d = dev.as_ref();
    let path = std::env::var("BUTTER_QWEN35_MTP_GGUF")
        .unwrap_or_else(|_| "/home/pidtom/models/qwen36-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf".to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[skipped] MTP GGUF not found at {path}");
        return;
    }
    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");
    let eos = g.meta_u32("tokenizer.ggml.eos_token_id");
    let model = qwen35::load_gguf(d, &g).expect("load_gguf");
    assert!(model.has_mtp(), "checkpoint at {path} has no MTP sidecar (blk.N.nextn.* tensors missing)");
    eprintln!("eos_token_id={eos:?} has_mtp=true");

    let gammas: Vec<usize> = std::env::var("BUTTER_QWEN35_SPEC_GAMMA")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![2, 3, 4]);

    let cases: &[(&str, &str, usize)] = &[
        ("01_factual", "/home/pidtom/qwen36-oracle/prompts/01_factual.txt", 64),
        ("02_code", "/home/pidtom/qwen36-oracle/prompts/02_code.txt", 24),
        ("03_longer", "/home/pidtom/qwen36-oracle/prompts/03_longer.txt", 64),
    ];

    let mut all_correct = true;
    for (id, path, max_gen) in cases {
        let prompt = std::fs::read_to_string(path).expect("read prompt");
        let prompt = prompt.strip_suffix('\n').unwrap_or(&prompt);
        let ids = tok.encode(prompt);
        eprintln!("--- {id} (prompt_tokens={}, max_gen={max_gen}) ---", ids.len());

        // Interleaved baseline/spec pairs (>=3 total per the perf-campaign
        // convention: baseline once here + once per gamma below == 4 pairs
        // for gammas=[2,3,4]).
        let (base_out, base_dt) = baseline_generate(d, &model, &ids, eos, *max_gen);
        let base_tps = base_out.len() as f64 / base_dt;
        eprintln!("  baseline: {} tok in {:.3}s = {:.2} tok/s -- {:?}", base_out.len(), base_dt, base_tps, tok.decode(&base_out));

        for &gamma in &gammas {
            let (spec_out, spec_dt, stats) = spec_generate(d, &model, &ids, eos, *max_gen, gamma);
            let spec_tps = spec_out.len() as f64 / spec_dt;
            let accept_rate = if stats.n_drafted > 0 { stats.n_accepted as f64 / stats.n_drafted as f64 } else { 0.0 };
            let matches = spec_out == base_out;
            all_correct &= matches;
            eprintln!(
                "  gamma={gamma}: {} tok in {:.3}s = {:.2} tok/s | drafted={} accepted={} accept_rate={:.1}% | MATCHES_BASELINE={matches}",
                spec_out.len(), spec_dt, spec_tps, stats.n_drafted, stats.n_accepted, accept_rate * 100.0
            );
            if !matches {
                let n = spec_out.len().min(base_out.len());
                let first_diff = (0..n).find(|&i| spec_out[i] != base_out[i]);
                eprintln!(
                    "    MISMATCH at token {:?}: spec={:?} base={:?}",
                    first_diff,
                    &spec_out[..spec_out.len().min(12)],
                    &base_out[..base_out.len().min(12)]
                );
            }
        }
    }
    assert!(all_correct, "spec-decode output diverged from baseline on at least one (prompt, gamma) pair -- see stderr above");
}
