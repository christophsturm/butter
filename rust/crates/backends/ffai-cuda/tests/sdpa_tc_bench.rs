// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! sdpa_multi vs sdpa_multi_tc throughput benchmark.
//! S=512/2048/4096/8192, nq=32 nkv=2 hd=128 causal=true (Nemotron dims).
use ffai_cuda::CudaDevice;
use ffai_core::{DType, Device, Tensor};
use ffai_ops::{sdpa_multi, sdpa_multi_tc};
use std::time::Instant;

fn rng(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0f32; n];
    let mut s: u64 = seed;
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = (s >> 33) as f32 / (1u64 << 31) as f32 * 0.2 - 0.1;
    }
    v
}
fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }

fn bench_one(dev: &dyn Device, s: usize, n_warm: usize, n_iter: usize) -> (f64, f64) {
    let (nq, nkv, hd) = (32usize, 2usize, 128usize);
    let hpg = nq / nkv;
    let n_kv = s;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let up = |v: &[f32]| Tensor::new(dev.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32);
    let q = up(&rng(s * nq * hd, 1)).reshaped(vec![s, nq, hd]);
    let k = up(&rng(nkv * n_kv * hd, 2)).reshaped(vec![nkv, n_kv, hd]);
    let v = up(&rng(nkv * n_kv * hd, 3)).reshaped(vec![nkv, n_kv, hd]);

    // Scalar warm-up.
    for _ in 0..n_warm {
        sdpa_multi(dev, &q, &k, &v, hd, nq as u32, 0, s as u32, n_kv as u32, hpg as u32, true, scale).unwrap();
    }
    dev.synchronize().ok();
    let t0 = Instant::now();
    for _ in 0..n_iter {
        sdpa_multi(dev, &q, &k, &v, hd, nq as u32, 0, s as u32, n_kv as u32, hpg as u32, true, scale).unwrap();
    }
    dev.synchronize().ok();
    let elapsed_scalar = t0.elapsed().as_secs_f64() / n_iter as f64;

    // TC warm-up.
    for _ in 0..n_warm {
        sdpa_multi_tc(dev, &q, &k, &v, hd, nq as u32, 0, s as u32, n_kv as u32, hpg as u32, true, scale).unwrap();
    }
    dev.synchronize().ok();
    let t1 = Instant::now();
    for _ in 0..n_iter {
        sdpa_multi_tc(dev, &q, &k, &v, hd, nq as u32, 0, s as u32, n_kv as u32, hpg as u32, true, scale).unwrap();
    }
    dev.synchronize().ok();
    let elapsed_tc = t1.elapsed().as_secs_f64() / n_iter as f64;

    (elapsed_scalar, elapsed_tc)
}

#[test]
fn sdpa_tc_bench() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else { eprintln!("no CUDA -- skip"); return; };
    let (nq, hd) = (32usize, 128usize);
    let mut log = String::new();
    log.push_str("\n=== sdpa_multi vs sdpa_multi_tc (nq=32, nkv=2, hd=128, causal, base_kv=0) ===\n");

    // S -> (n_warm, n_iter)
    let sizes: &[(usize, usize, usize)] = &[
        (512,  3, 5),
        (2048, 2, 3),
        (4096, 1, 2),
        (8192, 1, 2),
    ];

    for &(s, n_warm, n_iter) in sizes {
        let (t_scalar, t_tc) = bench_one(dev.as_ref(), s, n_warm, n_iter);
        let avg_kv = s as f64 / 2.0;
        let flops = 4.0 * nq as f64 * hd as f64 * avg_kv * s as f64;
        let tflops_scalar = flops / t_scalar / 1e12;
        let tflops_tc     = flops / t_tc     / 1e12;
        let speedup = t_scalar / t_tc;
        let line = format!(
            "S={s:5}: scalar {:.1}ms / {:.3} TFLOP/s | TC {:.1}ms / {:.3} TFLOP/s | speedup {:.2}x",
            t_scalar * 1e3, tflops_scalar,
            t_tc * 1e3, tflops_tc,
            speedup,
        );
        eprintln!("{line}");
        log.push_str(&line);
        log.push('\n');
    }

    let _ = std::fs::write("/home/pidtom/prefill_overnight.log", &log);
    eprintln!("{log}");
}
