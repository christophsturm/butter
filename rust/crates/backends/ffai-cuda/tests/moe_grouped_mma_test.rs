// Copyright 2026 Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Correctness for the grouped MoE GEMM (mma.sync register-O):
//! out[t,n] = Σ_k A[t,k]·W[eid(t)][n,k], sorted tokens, contiguous expert slab.
//! Reference computed on host (f32 accumulate of the same f16 inputs).

use ffai_cuda::CudaDevice;
use ffai_core::{DType, Device, Tensor};
use ffai_ops::{moe_grouped_gemm_mma, cast_f32_f16, cast_f16_f32};

fn rng(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0f32; n];
    let mut s = seed;
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.2;
    }
    v
}
fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }
fn dl(d: &dyn Device, t: &Tensor) -> Vec<f32> {
    let n = t.elem_count(); let mut b = vec![0u8; n*4];
    d.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
}
fn run(d: &dyn Device, n_exp: usize, k: usize, n: usize, groups: &[(usize, usize)]) {
    // groups: (expert_id, rows)
    let mut g_starts = vec![0usize];
    let mut expert_ids = vec![];
    for &(e, r) in groups { expert_ids.push(e); g_starts.push(g_starts.last().unwrap() + r); }
    let mt = *g_starts.last().unwrap();

    let a_f = rng(mt*k, 11);
    let w_f = rng(n_exp*n*k, 22);
    let (a16, w16) = (&a_f, &w_f); // f16 rounding folded into the 3e-2 tol

    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let a_dev = cast_f32_f16(d, &up(&a_f, vec![mt, k])).unwrap().reshaped(vec![mt, k]);
    let w_dev = cast_f32_f16(d, &up(&w_f, vec![n_exp, n, k])).unwrap().reshaped(vec![n_exp, n, k]);

    let out = moe_grouped_gemm_mma(d, &a_dev, &w_dev, &g_starts, &expert_ids, n, k).unwrap();
    let got = dl(d, &cast_f16_f32(d, &out).unwrap());

    // host reference
    let mut exp = vec![0f32; mt*n];
    for g in 0..groups.len() {
        let (eid, _) = groups[g];
        for t in g_starts[g]..g_starts[g+1] {
            for nn in 0..n {
                let mut acc = 0f32;
                for kk in 0..k { acc += a16[t*k+kk] * w16[(eid*n+nn)*k+kk]; }
                exp[t*n+nn] = acc;
            }
        }
    }
    let (mut maxa, mut maxr) = (0f32, 0f32);
    for (r, t) in exp.iter().zip(got.iter()) {
        let a = (r-t).abs(); let rel = a / r.abs().max(1e-3);
        if a>maxa {maxa=a;} if rel>maxr {maxr=rel;}
    }
    eprintln!("moe_grouped_mma mt={mt} N={n} K={k}: max_abs={maxa:.2e} max_rel={maxr:.2e}");
    assert!(maxr < 3e-2, "moe_grouped_mma: max_rel {maxr:.2e} > 3e-2 (max_abs {maxa:.2e})");
    eprintln!("moe_grouped_mma mt={mt} N={n} K={k}: PASS ✓");
}

#[test]
fn moe_grouped_mma_small() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run(d.as_ref(), 4, 64, 128, &[(0,20),(2,16),(3,8)]);
}
#[test]
fn moe_grouped_mma_nemotron_shape() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    // Nemotron up-proj shape: K=hid=2688, N=inter=1856, a few experts w/ ~96 rows.
    run(d.as_ref(), 8, 2688, 1856, &[(0,96),(3,40),(5,128),(7,17)]);
}

#[test]
fn moe_grouped_mma_bench() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let dev = d.as_ref();
    // Full S=2048 MoE up-proj: 128 experts × 96 rows, N=inter=1856, K=hid=2688.
    let (n_exp, n, k, rows) = (128usize, 1856usize, 2688usize, 96usize);
    let mut g_starts = vec![0usize]; let mut eids = vec![];
    for e in 0..n_exp { eids.push(e); g_starts.push(g_starts.last().unwrap() + rows); }
    let mt = *g_starts.last().unwrap();
    let a = cast_f32_f16(dev, &Tensor::new(dev.upload(&tb(&rng(mt*k, 1))).unwrap(), vec![mt*k], DType::F32)).unwrap().reshaped(vec![mt, k]);
    let w = cast_f32_f16(dev, &Tensor::new(dev.upload(&tb(&rng(n_exp*n*k, 2))).unwrap(), vec![n_exp*n*k], DType::F32)).unwrap().reshaped(vec![n_exp, n, k]);
    // warm
    let _ = moe_grouped_gemm_mma(dev, &a, &w, &g_starts, &eids, n, k).unwrap();
    dev.synchronize().ok();
    let iters = 30;
    let t = std::time::Instant::now();
    for _ in 0..iters { let _ = moe_grouped_gemm_mma(dev, &a, &w, &g_starts, &eids, n, k).unwrap(); }
    dev.synchronize().ok();
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let tflops = 2.0 * mt as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12;
    eprintln!("moe_grouped_mma_bench: mt={mt} N={n} K={k} → {ms:.3} ms/call, {tflops:.1} TFLOP/s (cuBLAS per-expert ref: 15.7)");
}
