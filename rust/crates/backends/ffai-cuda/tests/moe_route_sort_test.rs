// Copyright 2026 Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! On-device MoE route+counting-sort (moe_route_sort_device) vs host reference.
//! Validates: per-expert group counts (offsets) + the (token,weight) multiset per
//! expert match the host sigmoid+bias+topk+stable-sort.

use iron_cuda::CudaDevice;
use iron_core::{DType, Device, Tensor};
use iron_ops::moe_route_sort_device;

fn rng(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0f32; n]; let mut s = seed;
    for x in v.iter_mut() { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 2.0; }
    v
}
fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }
fn dl_u32(d: &dyn Device, t: &Tensor) -> Vec<u32> {
    let n = t.elem_count(); let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks(4).map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
}
fn dl_f32(d: &dyn Device, t: &Tensor) -> Vec<f32> {
    let n = t.elem_count(); let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
}

fn run(d: &dyn Device, s: usize, n_exp: usize, top_k: usize) {
    let scale = 1.5f32;
    let logits = rng(s*n_exp, 7);
    let bias = rng(n_exp, 99).iter().map(|x| x*0.3).collect::<Vec<_>>();
    let up = |v: &[f32]| Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32);
    let lg = up(&logits).reshaped(vec![s, n_exp]);
    let bi = up(&bias);
    let (st, sw, off) = moe_route_sort_device(d, &lg, &bi, n_exp, top_k, scale).unwrap();
    let stok = dl_u32(d, &st); let swt = dl_f32(d, &sw); let offs = dl_u32(d, &off);

    // host reference
    let mut host: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_exp]; // per expert: (token, weight)
    for ti in 0..s {
        let l = &logits[ti*n_exp..(ti+1)*n_exp];
        let sig: Vec<f32> = l.iter().map(|&z| 1.0/(1.0+(-z).exp())).collect();
        let b: Vec<f32> = (0..n_exp).map(|i| sig[i]+bias[i]).collect();
        let mut idx: Vec<usize> = (0..n_exp).collect();
        idx.sort_by(|&a,&c| b[c].partial_cmp(&b[a]).unwrap());
        let chosen = &idx[..top_k];
        let wsum: f32 = chosen.iter().map(|&e| sig[e]).sum::<f32>() + 1e-20;
        for &e in chosen { host[e].push((ti as u32, sig[e]/wsum*scale)); }
    }
    // 1. counts/offsets match
    for e in 0..n_exp {
        let dev_cnt = (offs[e+1]-offs[e]) as usize;
        assert_eq!(dev_cnt, host[e].len(), "expert {e}: dev count {dev_cnt} != host {}", host[e].len());
    }
    assert_eq!(*offs.last().unwrap() as usize, s*top_k, "total mt mismatch");
    // 2. per-expert (token,weight) multiset matches (order within group may differ)
    let mut maxw = 0f32;
    for e in 0..n_exp {
        let (lo, hi) = (offs[e] as usize, offs[e+1] as usize);
        let mut dev_grp: Vec<(u32,f32)> = (lo..hi).map(|r| (stok[r], swt[r])).collect();
        let mut h = host[e].clone();
        dev_grp.sort_by_key(|x| x.0); h.sort_by_key(|x| x.0);
        for ((dt,dw),(ht,hw)) in dev_grp.iter().zip(h.iter()) {
            assert_eq!(dt, ht, "expert {e}: token mismatch");
            let we = (dw-hw).abs(); if we>maxw {maxw=we;}
        }
    }
    eprintln!("moe_route_sort s={s} n_exp={n_exp} top_k={top_k}: counts+tokens MATCH, max|Δweight|={maxw:.2e} PASS ✓");
    assert!(maxw < 1e-4, "weight mismatch {maxw:.2e}");
}

#[test]
fn moe_route_sort_small() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run(d.as_ref(), 64, 16, 4);
}
#[test]
fn moe_route_sort_nemotron() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run(d.as_ref(), 2048, 128, 6); // Nemotron: s=2048, 128 experts, top-6
}
