// Copyright 2026 Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Correctness test for the FUSED single-kernel FlashAttention `sdpa_flash_fused`
//! vs the scalar `sdpa_multi` ground truth (GQA: nq=32, nkv=2, hd=128, causal).
//! Threshold: max |rel_err| ≤ 1e-2 (f16-input vs f32 tolerance).

use ffai_cuda::CudaDevice;
use ffai_core::{DType, Device, Tensor};
use ffai_ops::{sdpa_multi, sdpa_flash_fused};

fn rng(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0f32; n];
    let mut s: u64 = seed;
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f = (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
        *x = f * 0.2;
    }
    v
}

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }
fn dl_f32(d: &dyn Device, t: &Tensor) -> Vec<f32> {
    let n = t.elem_count();
    let mut b = vec![0u8; n * 4];
    d.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn run(dev: &dyn Device, s: usize, in_f16: bool) {
    let (nq, nkv, hd) = (32usize, 2usize, 128usize);
    let hpg = nq / nkv;
    let base_kv = 0usize;
    let n_kv = base_kv + s;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let q_vals = rng(s * nq * hd, 1234);
    let k_vals = rng(nkv * n_kv * hd, 5678);
    let v_vals = rng(nkv * n_kv * hd, 9012);

    let up = |v: &[f32]| Tensor::new(dev.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32);
    let q = up(&q_vals).reshaped(vec![s, nq, hd]);
    let k = up(&k_vals).reshaped(vec![nkv, n_kv, hd]);
    let v = up(&v_vals).reshaped(vec![nkv, n_kv, hd]);

    // Ground truth = scalar sdpa_multi (always f32).
    let ref_out = sdpa_multi(dev, &q, &k, &v, hd, nq as u32, base_kv as u32,
                             s as u32, n_kv as u32, hpg as u32, true, scale).unwrap();
    // Optionally cast inputs to f16 to exercise the f16 path.
    let (qf, kf, vf) = if in_f16 {
        (ffai_ops::cast_f32_f16(dev, &q).unwrap().reshaped(vec![s, nq, hd]),
         ffai_ops::cast_f32_f16(dev, &k).unwrap().reshaped(vec![nkv, n_kv, hd]),
         ffai_ops::cast_f32_f16(dev, &v).unwrap().reshaped(vec![nkv, n_kv, hd]))
    } else { (q.clone(), k.clone(), v.clone()) };

    let fused = sdpa_flash_fused(dev, &qf, &kf, &vf, hd, nq as u32, base_kv as u32,
                                 s as u32, n_kv as u32, hpg as u32, true, scale).unwrap();
    // fused output matches input dtype; widen f16 → f32 for comparison/download.
    let fused = if in_f16 { ffai_ops::cast_f16_f32(dev, &fused).unwrap() } else { fused };

    let a = dl_f32(dev, &ref_out);
    let b = dl_f32(dev, &fused);
    assert_eq!(a.len(), b.len());
    let (mut max_abs, mut max_rel) = (0f32, 0f32);
    for (r, t) in a.iter().zip(b.iter()) {
        let abs = (r - t).abs();
        let rel = abs / r.abs().max(1e-6);
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; }
    }
    let tag = if in_f16 { "f16" } else { "f32" };
    eprintln!("sdpa_flash_fused S={s} [{tag}]: max_abs={max_abs:.2e} max_rel={max_rel:.2e}");
    let tol = if in_f16 { 2e-2 } else { 1e-3 };
    assert!(max_rel < tol, "sdpa_flash_fused S={s} [{tag}]: max_rel={max_rel:.2e} > {tol:.0e}");
    eprintln!("sdpa_flash_fused S={s} [{tag}]: PASS ✓");
}

#[test]
fn sdpa_flash_fused_s128_f32() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else { eprintln!("no CUDA — skip"); return; };
    run(dev.as_ref(), 128, false);
}
#[test]
fn sdpa_flash_fused_s512_f16() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else { eprintln!("no CUDA — skip"); return; };
    run(dev.as_ref(), 512, true);
}
#[test]
fn sdpa_flash_fused_s2048_f16() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else { eprintln!("no CUDA — skip"); return; };
    run(dev.as_ref(), 2048, true);
}
