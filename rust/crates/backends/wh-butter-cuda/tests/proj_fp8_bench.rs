// FP8 W8A8 dense-proj GEMM micro-bench (synthetic, GEMM-level).
// Times f16 cuBLAS (gemm_cublas_f32out) vs W8A8 e4m3 (per-tensor gemm_fp8,
// per-channel gemm_fp8_perchan) at the real Nemotron-30B dense-proj shapes.
// Isolates: does the W8A8 path realize the ~1.8x GB10 roofline at the GEMM
// level, and does it stay numerically close (cosine vs f16) per-tensor vs
// per-channel? Seconds per run — no 30B reload.
use wh_butter_cuda::CudaDevice;
use wh_butter_core::{DType, Device, Tensor};
use wh_butter_ops::{cast_f32_f16, fp8_quant, fp8_quant_perrow, gemm_cublas_f32out, gemm_fp8, gemm_fp8_perchan};
use std::time::Instant;

fn rng(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0f32; n];
    let mut s: u64 = seed;
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = (s >> 33) as f32 / (1u64 << 31) as f32 * 0.2 - 0.1; // [-0.1,0.1]
    }
    v
}
fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }

fn download_f32(dev: &dyn Device, t: &Tensor, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    dev.download(t.buffer.as_ref(), &mut bytes).unwrap();
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64; let mut na = 0f64; let mut nb = 0f64;
    for i in 0..a.len() { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

// m=rows(seq), n=out_features, k=in_features.  out[m,n] = x[m,k] · w[n,k]^T
fn bench_shape(dev: &dyn Device, label: &str, m: usize, n: usize, k: usize, n_warm: usize, n_iter: usize) {
    let xf = rng(m * k, 11);
    let wf = rng(n * k, 22);
    let x32 = Tensor::new(dev.upload(&tb(&xf)).unwrap(), vec![m * k], DType::F32).reshaped(vec![m, k]);
    let w32 = Tensor::new(dev.upload(&tb(&wf)).unwrap(), vec![n * k], DType::F32).reshaped(vec![n, k]);
    let xh = cast_f32_f16(dev, &x32).unwrap(); // [m,k] f16
    let wh = cast_f32_f16(dev, &w32).unwrap(); // [n,k] f16
    let flops = 2.0 * m as f64 * n as f64 * k as f64;

    // ---- f16 cuBLAS reference (the current proj path) ----
    let f16_ref = gemm_cublas_f32out(dev, &xh, &wh, m, n, k).unwrap(); // [m,n] f32
    for _ in 0..n_warm { let _ = gemm_cublas_f32out(dev, &xh, &wh, m, n, k).unwrap(); }
    dev.synchronize().ok();
    let t0 = Instant::now();
    for _ in 0..n_iter { let _ = gemm_cublas_f32out(dev, &xh, &wh, m, n, k).unwrap(); }
    dev.synchronize().ok();
    let t_f16 = t0.elapsed().as_secs_f64() / n_iter as f64;
    let ref_v = download_f32(dev, &f16_ref, m * n);

    // ---- W8A8 per-TENSOR e4m3 (gemm_fp8) — GEMM-only timing (quant excluded) ----
    // Pre-quantize once (weights fixed; per-tensor act scale recomputed per call
    // in the real engine, but here we time the GEMM core in isolation).
    let (wq_pt, wsc_pt) = fp8_quant(dev, &wh, n * k, 1.0).unwrap();
    let (xq_pt, xsc_pt) = fp8_quant(dev, &xh, m * k, 1.0).unwrap();
    let pt_ref = gemm_fp8(dev, &xq_pt, &xsc_pt, &wq_pt, &wsc_pt, m, n, k, true).unwrap();
    for _ in 0..n_warm { let _ = gemm_fp8(dev, &xq_pt, &xsc_pt, &wq_pt, &wsc_pt, m, n, k, true).unwrap(); }
    dev.synchronize().ok();
    let t1 = Instant::now();
    for _ in 0..n_iter { let _ = gemm_fp8(dev, &xq_pt, &xsc_pt, &wq_pt, &wsc_pt, m, n, k, true).unwrap(); }
    dev.synchronize().ok();
    let t_pt = t1.elapsed().as_secs_f64() / n_iter as f64;
    let cos_pt = cosine(&ref_v, &download_f32(dev, &pt_ref, m * n));

    // ---- W8A8 per-CHANNEL e4m3 (gemm_fp8_perchan) — GEMM + rowcol-scale post-pass ----
    let (wq_pc, wsc_pc) = fp8_quant_perrow(dev, &wh, n, k).unwrap();
    let (xq_pc, xsc_pc) = fp8_quant_perrow(dev, &xh, m, k).unwrap();
    let pc_ref = gemm_fp8_perchan(dev, &xq_pc, &xsc_pc, &wq_pc, &wsc_pc, m, n, k).unwrap();
    for _ in 0..n_warm { let _ = gemm_fp8_perchan(dev, &xq_pc, &xsc_pc, &wq_pc, &wsc_pc, m, n, k).unwrap(); }
    dev.synchronize().ok();
    let t2 = Instant::now();
    for _ in 0..n_iter { let _ = gemm_fp8_perchan(dev, &xq_pc, &xsc_pc, &wq_pc, &wsc_pc, m, n, k).unwrap(); }
    dev.synchronize().ok();
    let t_pc = t2.elapsed().as_secs_f64() / n_iter as f64;
    let cos_pc = cosine(&ref_v, &download_f32(dev, &pc_ref, m * n));

    let tf = |t: f64| flops / t / 1e12;
    println!(
        "{label:22} m={m:5} n={n:5} k={k:5} | f16 {f16ms:7.3}ms {f16tf:6.1}TF | W8A8-PT {ptms:7.3}ms {pttf:6.1}TF {ptx:4.2}x cos {cospt:.5} | W8A8-PC {pcms:7.3}ms {pctf:6.1}TF {pcx:4.2}x cos {cospc:.5}",
        f16ms = t_f16 * 1e3, f16tf = tf(t_f16),
        ptms = t_pt * 1e3, pttf = tf(t_pt), ptx = t_f16 / t_pt, cospt = cos_pt,
        pcms = t_pc * 1e3, pctf = tf(t_pc), pcx = t_f16 / t_pc, cospc = cos_pc,
    );
}

#[test]
fn proj_fp8_bench() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else { eprintln!("no CUDA -- skip"); return; };
    let s: usize = std::env::var("PROJ_S").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let (nw, ni) = (5usize, 30usize);
    println!("\n==== FP8 W8A8 dense-proj GEMM micro-bench (S={s}) — Nemotron-30B shapes ====");
    println!("(f16 = current cuBLAS proj path; W8A8-PT = per-tensor gemm_fp8; W8A8-PC = per-channel gemm_fp8_perchan)\n");
    // m = seq tokens (S); n = out_features; k = in_features.
    bench_shape(dev.as_ref(), "mamba.in_proj",  s, 10304, 2688, nw, ni);
    bench_shape(dev.as_ref(), "mamba.out_proj", s,  2688, 4096, nw, ni);
    bench_shape(dev.as_ref(), "attn.q_proj",    s,  4096, 2688, nw, ni);
    bench_shape(dev.as_ref(), "attn.o_proj",    s,  2688, 4096, nw, ni);
    bench_shape(dev.as_ref(), "shared.up",      s,  3712, 2688, nw, ni);
    bench_shape(dev.as_ref(), "shared.down",    s,  2688, 3712, nw, ni);
    println!("\n==== done ====\n");
}
