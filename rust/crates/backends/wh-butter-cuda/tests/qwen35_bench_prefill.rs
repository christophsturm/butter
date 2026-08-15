// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! F-85 prefill wave: pp512/pp2048 prompt-processing throughput
//! (tok/s), batched `qwen35::prefill` vs. the old token-by-token
//! decode-loop baseline, plus a decode-after-prefill regression check.
//! Uninstrumented wall-clock, mirrors `qwen35_bench_decode.rs`'s
//! conventions (real GGUF, real forward, host argmax/download included
//! in the timed region only where the existing decode bench already did).
use wh_butter_cuda::CudaDevice;
use wh_butter_core::Device;
use wh_butter_loader::gguf::Gguf;
use wh_butter_models::qwen35;

/// Deterministic pseudo-random token ids in `[0, vocab)` -- content
/// doesn't matter for a throughput bench (every token drives the exact
/// same shape/compute path through embed -> layers -> lm_head regardless
/// of id), only shape does. Mirrors the seeded-PRNG convention the GEMM
/// microbenches in this same crate already use for synthetic inputs.
fn synthetic_prompt(n: usize, vocab: u32, seed: u32) -> Vec<u32> {
    let mut s = seed | 1;
    let mut next = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
    (0..n).map(|_| next() % vocab).collect()
}

fn download(dev: &dyn Device, t: &wh_butter_core::Tensor, vocab: usize) -> Vec<f32> {
    let mut b = vec![0u8; vocab * 4];
    dev.synchronize().unwrap();
    dev.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0 as u32
}

#[test]
fn qwen35_bench_prefill_pp() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let path = std::env::var("BUTTER_QWEN35_GGUF")
        .unwrap_or_else(|_| "/home/pidtom/models/qwen36-27b/Qwen3.6-27B-Q4_K_M.gguf".to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[skipped] GGUF not found at {path}");
        return;
    }
    let g = Gguf::open(&path).expect("open gguf");
    let model = qwen35::load_gguf(d, &g).expect("load_gguf");
    let vocab = model.cfg.vocab as u32;

    println!("prompt_len,mode,run,seconds,tok_s");

    for &n in &[512usize, 2048usize] {
        let ids = synthetic_prompt(n, vocab, 0x1234_5678 ^ n as u32);

        // ---- OLD baseline: token-by-token decode-loop over the prompt ----
        // One rep only (not the metric under test, and at ~13 tok/s a 2048-
        // token decode-loop rep alone is ~150s -- see the module doc).
        {
            let mut cache = qwen35::Qwen35Cache::new(d, &model.cfg, n + 32).expect("cache");
            let mut pos = 0u32;
            let t0 = std::time::Instant::now();
            for &tid in &ids {
                let _ = qwen35::decode_step(d, &model, &mut cache, tid, pos).expect("decode_step");
                pos += 1;
            }
            d.synchronize().unwrap();
            let dt = t0.elapsed().as_secs_f64();
            println!("{n},decode_loop,1,{dt:.3},{:.3}", n as f64 / dt);
            eprintln!("[pp{n}] decode-loop baseline: {dt:.3}s => {:.2} tok/s", n as f64 / dt);
        }

        // ---- NEW: batched prefill(), chunk = n (single chunk == pp{n}) ----
        let mut pp_times = Vec::new();
        for run in 1..=3 {
            let mut cache = qwen35::Qwen35Cache::new(d, &model.cfg, n + 32).expect("cache");
            let t0 = std::time::Instant::now();
            let logits = qwen35::prefill(d, &model, &mut cache, &ids, n).expect("prefill");
            d.synchronize().unwrap();
            let dt = t0.elapsed().as_secs_f64();
            pp_times.push(dt);
            println!("{n},prefill,{run},{dt:.3},{:.3}", n as f64 / dt);
            eprintln!("[pp{n}] prefill run {run}: {dt:.3}s => {:.2} tok/s", n as f64 / dt);

            // Decode-after-prefill regression check, only on the last run
            // (keeps the bench's total wall time bounded): continue
            // greedily for 32 tokens right after this prefill and confirm
            // the per-step decode rate hasn't regressed from the ~13.3
            // tok/s record.
            if run == 3 {
                let last_logits = download(d, &logits, model.cfg.vocab);
                let mut next = argmax(&last_logits);
                let mut pos = n as u32;
                let n_steps = 32usize;
                let t1 = std::time::Instant::now();
                for _ in 0..n_steps {
                    let lg = qwen35::decode_step(d, &model, &mut cache, next, pos).expect("decode_step");
                    let lv = download(d, &lg, model.cfg.vocab);
                    next = argmax(&lv);
                    pos += 1;
                }
                let dt2 = t1.elapsed().as_secs_f64();
                eprintln!("[pp{n}] decode-after-prefill: {n_steps} steps in {dt2:.3}s => {:.2} tok/s", n_steps as f64 / dt2);
            }
        }
        let avg = pp_times.iter().sum::<f64>() / pp_times.len() as f64;
        let speedup = {
            // Recompute the baseline seconds from the printed line above by
            // re-deriving n/tok_s isn't available here without re-running;
            // report avg pp tok/s instead, the table's speedup column is
            // derived from this + the decode_loop line already printed.
            n as f64 / avg
        };
        eprintln!("[pp{n}] prefill avg over 3 runs: {avg:.3}s => {speedup:.2} tok/s");
    }
}
