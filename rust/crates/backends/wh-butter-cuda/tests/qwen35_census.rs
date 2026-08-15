// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Round-9 step-time phase census: buckets one decode step's GPU time by
//! subsystem (routed marlin GEMMs / small plain projections / GDN chain /
//! attention / elementwise+norms / lm_head / embed), plus host-side
//! sync+download+argmax measured OUTSIDE the instrumented call (same split
//! `qwen35_bench_decode.rs` already uses). Run with the round-8 baseline-
//! of-record config (`QWEN35_MARLIN=1 QWEN35_LM_HEAD_Q4=1`) to match the
//! 12.58 tok/s number the round-9 brief cites.
//!
//! `decode_step_census` syncs the device at every bucket boundary (~700
//! sync round trips/step) to get real per-bucket wall time instead of
//! host-side launch-return noise -- that instrumentation has its own tax,
//! so this harness also times an uninstrumented reference block the same
//! way `qwen35_bench_decode.rs` does and reports the delta, rather than
//! letting it hide inside the per-bucket numbers.
use wh_butter_cuda::CudaDevice;
use wh_butter_loader::gguf::Gguf;
use wh_butter_models::gguf_tokenizer::GgufTokenizer;
use wh_butter_models::qwen35::{self, CensusBuckets};

#[test]
fn qwen35_step_census() {
    let Some(dev) = CudaDevice::create().expect("cuda") else {
        eprintln!("no CUDA, skip");
        return;
    };
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
    let argmax = |lv: &[f32]| -> u32 {
        lv.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0 as u32
    };
    for &tid in &ids {
        let logits = qwen35::decode_step(d, &model, &mut cache, tid, pos).expect("decode_step");
        last_logits = download(d, &logits, model.cfg.vocab);
        pos += 1;
    }
    let mut next = argmax(&last_logits);

    // Warm-up (module JIT compile/cache, marlin scratch alloc) -- uninstrumented.
    for _ in 0..8 {
        let logits = qwen35::decode_step(d, &model, &mut cache, next, pos).expect("decode_step");
        next = argmax(&download(d, &logits, model.cfg.vocab));
        pos += 1;
    }

    // Uninstrumented reference block -- same measurement qwen35_bench_decode.rs
    // takes, to quantify the census's own sync-tax against it.
    let n_ref = 32usize;
    let t_ref0 = std::time::Instant::now();
    for _ in 0..n_ref {
        let logits = qwen35::decode_step(d, &model, &mut cache, next, pos).expect("decode_step");
        next = argmax(&download(d, &logits, model.cfg.vocab));
        pos += 1;
    }
    let ref_ms_per_step = t_ref0.elapsed().as_secs_f64() * 1000.0 / n_ref as f64;

    // Instrumented census.
    let n_steps = 32usize;
    let mut buckets = CensusBuckets::default();
    let mut host_ms = 0.0f64;
    let t0 = std::time::Instant::now();
    for _ in 0..n_steps {
        let (logits, _xn) =
            qwen35::decode_step_census(d, &model, &mut cache, next, pos, &mut buckets).expect("decode_step_census");
        let th0 = std::time::Instant::now();
        next = argmax(&download(d, &logits, model.cfg.vocab));
        host_ms += th0.elapsed().as_secs_f64() * 1000.0;
        pos += 1;
    }
    let inst_ms_per_step = t0.elapsed().as_secs_f64() * 1000.0 / n_steps as f64;
    let host_ms_per_step = host_ms / n_steps as f64;

    let b = buckets;
    let gpu_total = b.total_ms() / n_steps as f64;
    let row = |name: &str, ms: f64| {
        eprintln!(
            "QWEN35_CENSUS  {name:<18} {:>8.3} ms/step  {:>6.2}%",
            ms / n_steps as f64,
            ms / b.total_ms() * 100.0
        )
    };
    eprintln!("QWEN35_CENSUS ===== step-time census ({n_steps} steps, n_ref={n_ref}) =====");
    row("embed", b.embed_ms);
    row("routed_marlin", b.routed_marlin_ms);
    row("small_proj", b.small_proj_ms);
    row("gdn_chain", b.gdn_chain_ms);
    row("attention", b.attention_ms);
    row("elementwise/norms", b.elementwise_ms);
    row("lm_head", b.lm_head_ms);
    eprintln!("QWEN35_CENSUS  {:<18} {:>8.3} ms/step  (gpu bucket sum)", "TOTAL(buckets)", gpu_total);
    eprintln!(
        "QWEN35_CENSUS  {:<18} {:>8.3} ms/step  (download+argmax, outside instrumented call)",
        "host_sync", host_ms_per_step
    );
    eprintln!(
        "QWEN35_CENSUS  instrumented total:   {:.3} ms/step ({:.2} tok/s)",
        inst_ms_per_step,
        1000.0 / inst_ms_per_step
    );
    eprintln!(
        "QWEN35_CENSUS  uninstrumented ref:   {:.3} ms/step ({:.2} tok/s)  <- sync-tax = {:.3} ms/step ({:.1}%)",
        ref_ms_per_step,
        1000.0 / ref_ms_per_step,
        inst_ms_per_step - ref_ms_per_step,
        (inst_ms_per_step - ref_ms_per_step) / ref_ms_per_step * 100.0
    );
}
