// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Direct correctness test for `sdpa_multi_tc` vs `sdpa_multi`.
//! Validates the TC path against the scalar kernel on same Q/K/V inputs
//! at S=128, S=512, S=2048 (GQA: nq=32, nkv=2, hd=128, causal=true).
//! Threshold: max |rel_err| ≤ 1e-2 (1% — typical f16 vs f32 tolerance).

use ffai_cuda::CudaDevice;
use ffai_core::{DType, Device, Tensor};
use ffai_ops::{sdpa_multi, sdpa_multi_tc};

fn rng(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0f32; n];
    let mut s: u64 = seed;
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f = (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0; // [-1, 1)
        *x = f * 0.2; // small values to avoid f16 saturation
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

fn run_sdpa_tc_test(dev: &dyn Device, s: usize) {
    let (nq, nkv, hd) = (32usize, 2usize, 128usize);
    let hpg = nq / nkv;
    let base_kv = 0usize; // no prefix
    let n_kv = base_kv + s;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let q_vals = rng(s * nq * hd, 1234);
    let k_vals = rng(nkv * n_kv * hd, 5678);
    let v_vals = rng(nkv * n_kv * hd, 9012);

    let up = |v: &[f32]| Tensor::new(
        dev.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32);
    let q = up(&q_vals).reshaped(vec![s, nq, hd]);
    let k = up(&k_vals).reshaped(vec![nkv, n_kv, hd]);
    let v = up(&v_vals).reshaped(vec![nkv, n_kv, hd]);

    // Scalar reference.
    let ref_out = sdpa_multi(dev, &q, &k, &v, hd, nq as u32, base_kv as u32,
                              s as u32, n_kv as u32, hpg as u32, true, scale).unwrap();
    // TC path.
    let tc_out  = sdpa_multi_tc(dev, &q, &k, &v, hd, nq as u32, base_kv as u32,
                                 s as u32, n_kv as u32, hpg as u32, true, scale).unwrap();

    let ref_data = dl_f32(dev, &ref_out);
    let tc_data  = dl_f32(dev, &tc_out);
    assert_eq!(ref_data.len(), tc_data.len());

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (r, t) in ref_data.iter().zip(tc_data.iter()) {
        let abs = (r - t).abs();
        let rel = abs / (r.abs().max(1e-6));
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; }
    }
    eprintln!("sdpa_multi_tc S={s}: max_abs={max_abs:.2e} max_rel={max_rel:.2e}");
    assert!(max_rel < 1e-2,
        "sdpa_multi_tc S={s}: max_rel={max_rel:.2e} exceeds 1e-2 threshold (max_abs={max_abs:.2e})");
    eprintln!("sdpa_multi_tc S={s}: PASS ✓");
}

#[test]
fn sdpa_multi_tc_correctness_s128() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else { eprintln!("no CUDA — skip"); return; };
    run_sdpa_tc_test(dev.as_ref(), 128);
}

#[test]
fn sdpa_multi_tc_correctness_s512() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else { eprintln!("no CUDA — skip"); return; };
    run_sdpa_tc_test(dev.as_ref(), 512);
}

#[test]
fn sdpa_multi_tc_correctness_s2048() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else { eprintln!("no CUDA — skip"); return; };
    run_sdpa_tc_test(dev.as_ref(), 2048);
}

#[test]
fn sdpa_multi_tc_correctness_s4096() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else { eprintln!("no CUDA --- skip"); return; };
    run_sdpa_tc_test(dev.as_ref(), 4096);
}
