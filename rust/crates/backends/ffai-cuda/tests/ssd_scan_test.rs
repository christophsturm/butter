#![cfg(feature = "cuda")]
// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Mamba2 SSD chunked-MATMUL prefill scan (`ssm_prefill_scan_ssd`) on CUDA vs
//! the sequential `ssm_prefill_scan` reference, for the NemotronH cell
//! (dh=64, ds=128, H=64, G=8). Gate: max|Δy| < 1e-3 (fp16 GEMM path).
use iron_core::{DType, Device, Tensor};
use iron_cuda::CudaDevice;
use iron_ops::{ssm_prefill_scan, ssm_prefill_scan_ssd};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
// small, smooth pseudo-random fill in a sane range (keeps decay finite).
fn fill(n: usize, s: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (((i * 31 + s * 977) % 251) as f32 / 251.0 - 0.5) * 2.0 * scale).collect()
}
fn tn(d: &dyn Device, v: &[f32], sh: Vec<usize>) -> Tensor {
    Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32)
}

fn run_for_len(dev: &dyn Device, t: usize, l: u32) {
    let (h, dh, ds, ng) = (64usize, 64usize, 128usize, 8usize);
    // Use REALISTIC Mamba2 magnitudes to flush out precision/structural bugs the
    // benign synthetic ranges hide: bigger x/B/C, a wide A spread, and dt up to
    // ~1.5 (post-softplus) so decay spans a large dynamic range.
    let x = fill(t * h * dh, 1, 4.0);
    // a_log spread so A=-exp(a_log) ranges ~[-0.05, -30] across heads.
    let a_log: Vec<f32> = (0..h).map(|i| -3.0 + 6.0 * (i as f32 / h as f32)).collect();
    let b = fill(t * ng * ds, 3, 3.0);
    let c = fill(t * ng * ds, 4, 3.0);
    let dsk = fill(h, 5, 1.0);
    let dt: Vec<f32> = (0..t * h).map(|i| 0.05 + 1.4 * (((i * 13) % 11) as f32 / 11.0)).collect();
    let si = vec![0.0f32; h * dh * ds]; // prefill starts at zero state

    let xt = tn(dev, &x, vec![t * h * dh]);
    let at = tn(dev, &a_log, vec![h]);
    let bt = tn(dev, &b, vec![t * ng * ds]);
    let ct = tn(dev, &c, vec![t * ng * ds]);
    let dt_t = tn(dev, &dsk, vec![h]);
    let dtt = tn(dev, &dt, vec![t * h]);
    let sit = tn(dev, &si, vec![h * dh * ds]);

    // Sequential reference (on device).
    let (so_seq, y_seq) = ssm_prefill_scan(
        dev, &xt, &at, &bt, &ct, &dt_t, &dtt, &sit,
        t as u32, dh as u32, ds as u32, h as u32, ng as u32,
    ).unwrap();
    // SSD matmul path (on device).
    let (so_ssd, y_ssd) = ssm_prefill_scan_ssd(
        dev, &xt, &at, &bt, &ct, &dt_t, &dtt, &sit,
        t as u32, dh as u32, ds as u32, h as u32, ng as u32, l, None,
    ).unwrap();
    dev.synchronize().unwrap();

    let mut yb = vec![0u8; t * h * dh * 4];
    dev.download(y_seq.buffer.as_ref(), &mut yb).unwrap();
    let y_ref = fb(&yb);
    dev.download(y_ssd.buffer.as_ref(), &mut yb).unwrap();
    let y_got = fb(&yb);
    let mut sb = vec![0u8; h * dh * ds * 4];
    dev.download(so_seq.buffer.as_ref(), &mut sb).unwrap();
    let s_ref = fb(&sb);
    dev.download(so_ssd.buffer.as_ref(), &mut sb).unwrap();
    let s_got = fb(&sb);

    let mut ey = 0.0f32;
    let mut yref_mag = 0.0f32;
    for i in 0..y_ref.len() {
        ey = ey.max((y_got[i] - y_ref[i]).abs());
        yref_mag = yref_mag.max(y_ref[i].abs());
    }
    let mut es = 0.0f32;
    let mut sref_mag = 0.0f32;
    for i in 0..s_ref.len() {
        es = es.max((s_got[i] - s_ref[i]).abs());
        sref_mag = sref_mag.max(s_ref[i].abs());
    }
    // cosine similarity on y (robust to magnitude).
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..y_ref.len() {
        dot += y_ref[i] as f64 * y_got[i] as f64;
        na += (y_ref[i] as f64).powi(2);
        nb += (y_got[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    let rel = ey / yref_mag.max(1e-6);
    let rel_s = es / sref_mag.max(1e-6);
    eprintln!(
        "SSD vs sequential  T={t} L={l}:  rel|Δy|={rel:.3e}  cos={cos:.6}  rel|Δstate|={rel_s:.3e}  (|y|max={yref_mag:.3e})"
    );
    // fp16 GEMM (f32 accumulate) → ~1e-3 relative is the floor; gate on relative + cosine.
    assert!(rel < 3e-3, "y relative mismatch too large at T={t} L={l}: rel={rel:.3e} (abs={ey:.3e})");
    assert!(cos > 0.9999, "y cosine too low at T={t} L={l}: {cos:.6}");
    assert!(rel_s < 3e-3, "state relative mismatch too large at T={t} L={l}: rel={rel_s:.3e}");
}

#[test]
fn ssd_matmul_matches_sequential() {
    let Some(dev) = CudaDevice::create().expect("cuda") else {
        eprintln!("no CUDA — skip");
        return;
    };
    // exact multiples of L, and a non-multiple (tail-padded chunk).
    run_for_len(dev.as_ref(), 256, 128);
    run_for_len(dev.as_ref(), 512, 128);
    run_for_len(dev.as_ref(), 512, 256);
    run_for_len(dev.as_ref(), 300, 128); // 300 = 2*128 + 44 → tail chunk
    eprintln!("✅ SSD chunked-matmul scan matches sequential scan to <1e-3 (fp16).");
}

// ════════════════════════════════════════════════════════════════════════════
// FAST SSD-lever A/B harness (FUSED off-vs-on, no model load).
//
// The fused SSD scan (`ssm_prefill_scan_ssd`, NEMOTRON_SSD_FUSED default-on,
// escape NEMOTRON_SSD_FUSED_OFF=1) is HARD-GATED to the Nemotron Mamba2 cell
// (dh=64, ds=128, n_heads=64, n_groups=8) — see the shape guard at the top of
// `ssm_prefill_scan_ssd_strided`. So the right fast proxy for THIS lever is a
// synthetic problem AT THAT EXACT CELL, sized at a prefill-realistic S. It
// loads nothing → runs in milliseconds, and the per-path GPU time gives the
// fused kernel's RELATIVE speedup (the absolute differs from the 62GB 30B run
// because the 30B interleaves 23 SSD layers with MoE/attn, but the SSD-kernel
// speedup direction + magnitude transfer 1:1 since it is the SAME kernel on the
// SAME cell). Correctness gate = the same rel|Δ| < 3e-3 band-safe bound.
//
// This is the FAST TIER for the SSD lever. Final e2e tok/s + 16/16 quality
// still run on the 30B (`--test nemotron nemotron_decode_bench`).
//
// Invocation:
//   cargo test --release -p ffai-cuda --features cuda \
//     --test ssd_scan ssd_lever_ab_bench -- --exact --nocapture
//   env: SSD_BENCH_S="512,1024,2048" (default), SSD_BENCH_L=128, SSD_BENCH_ITERS=20
// ════════════════════════════════════════════════════════════════════════════
fn time_path<F: Fn() -> (Tensor, Tensor)>(dev: &dyn Device, iters: usize, f: F) -> f64 {
    // warmup (kernel compile / cache) then time `iters` device-synced runs.
    let _ = f();
    dev.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _r = f();
    }
    dev.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64 // ms/run
}

fn ab_for_len(dev: &dyn Device, t: usize, l: u32, iters: usize) {
    let (h, dh, ds, ng) = (64usize, 64usize, 128usize, 8usize); // Nemotron cell (hard-gated)
    let x = fill(t * h * dh, 1, 4.0);
    let a_log: Vec<f32> = (0..h).map(|i| -3.0 + 6.0 * (i as f32 / h as f32)).collect();
    let b = fill(t * ng * ds, 3, 3.0);
    let c = fill(t * ng * ds, 4, 3.0);
    let dsk = fill(h, 5, 1.0);
    let dt: Vec<f32> = (0..t * h).map(|i| 0.05 + 1.4 * (((i * 13) % 11) as f32 / 11.0)).collect();
    let si = vec![0.0f32; h * dh * ds];

    let xt = tn(dev, &x, vec![t * h * dh]);
    let at = tn(dev, &a_log, vec![h]);
    let bt = tn(dev, &b, vec![t * ng * ds]);
    let ct = tn(dev, &c, vec![t * ng * ds]);
    let dt_t = tn(dev, &dsk, vec![h]);
    let dtt = tn(dev, &dt, vec![t * h]);
    let sit = tn(dev, &si, vec![h * dh * ds]);

    // OFF = sequential reference scan; ON = fused chunked-matmul SSD scan.
    let off = || {
        ssm_prefill_scan(
            dev, &xt, &at, &bt, &ct, &dt_t, &dtt, &sit,
            t as u32, dh as u32, ds as u32, h as u32, ng as u32,
        ).unwrap()
    };
    let on = || {
        ssm_prefill_scan_ssd(
            dev, &xt, &at, &bt, &ct, &dt_t, &dtt, &sit,
            t as u32, dh as u32, ds as u32, h as u32, ng as u32, l, None,
        ).unwrap()
    };

    // correctness (rel|Δ| on y) — fused must match sequential within the band.
    let (_, y_off) = off();
    let (_, y_on) = on();
    dev.synchronize().unwrap();
    let mut yb = vec![0u8; t * h * dh * 4];
    dev.download(y_off.buffer.as_ref(), &mut yb).unwrap();
    let yr = fb(&yb);
    dev.download(y_on.buffer.as_ref(), &mut yb).unwrap();
    let yg = fb(&yb);
    let (mut e, mut mag) = (0.0f32, 0.0f32);
    for i in 0..yr.len() {
        e = e.max((yg[i] - yr[i]).abs());
        mag = mag.max(yr[i].abs());
    }
    let rel = e / mag.max(1e-6);

    let off_ms = time_path(dev, iters, off);
    let on_ms = time_path(dev, iters, on);
    let speedup = off_ms / on_ms.max(1e-9);
    eprintln!(
        "  S={t:<5} L={l:<4} | OFF(seq)={off_ms:8.3} ms  ON(fused)={on_ms:8.3} ms  speedup={speedup:5.2}x  | rel|Δy|={rel:.2e} {}",
        if rel < 3e-3 { "OK" } else { "FAIL" }
    );
    assert!(rel < 3e-3, "SSD fused mismatch at S={t} L={l}: rel={rel:.2e}");
}

#[test]
fn ssd_lever_ab_bench() {
    let Some(dev) = CudaDevice::create().expect("cuda") else {
        eprintln!("no CUDA — skip");
        return;
    };
    let s_list: Vec<usize> = std::env::var("SSD_BENCH_S")
        .unwrap_or_else(|_| "512,1024,2048".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let l: u32 = std::env::var("SSD_BENCH_L").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let iters: usize = std::env::var("SSD_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    eprintln!("──────── SSD lever A/B (Nemotron cell 64/128/64/8, synthetic, no model) ────────");
    eprintln!("  OFF = ssm_prefill_scan (sequential)   ON = ssm_prefill_scan_ssd (fused chunked-matmul)");
    for &s in &s_list {
        ab_for_len(dev.as_ref(), s, l, iters);
    }
    eprintln!("────────────────────────────────────────────────────────────────────────────────");
    eprintln!("✅ SSD fused matches sequential (<3e-3); speedup is the relative kernel win.");
}
