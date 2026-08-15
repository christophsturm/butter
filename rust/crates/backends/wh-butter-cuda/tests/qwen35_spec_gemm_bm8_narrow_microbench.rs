// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Round-5 F-85 batched-verify follow-up, ARM B: does narrowing
//! `gemm_q4_bm8`'s BN from 32 down to 16 or 8 restore enough threadgroup
//! count to approach (not necessarily clear) the `<=1.4x` batched-verify
//! gate? Same harness/shapes/gate as `qwen35_spec_gemm_bm8_microbench.rs`
//! for direct apples-to-apples comparison.
//!
//! See `gemm_q4_bm8_narrow.rs` (iron repo) for the kernel doc, including the
//! correction that `coop_tile` on this CUDA backend is a software-emulated
//! shared-memory GEMM (no real `wmma`/`mma.sync`), not a hardware
//! tensor-core path -- so BN is a free retiling knob here, not constrained
//! by a hardware fragment shape.
use std::time::Instant;
use wh_butter_core::{DType, Tensor};
use wh_butter_cuda::CudaDevice;
use wh_butter_ops::{gemm_q4_bm8n16, gemm_q4_bm8n8, gemv_q4, quantize_q4};

fn tbu(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn f16_bits(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32 - 112;
    if e <= 0 { return sign; }
    if e >= 0x1f { return sign | 0x7c00; }
    let m = (x >> 13) & 0x3ff;
    let round = (x >> 12) & 1;
    let v = ((e as u32) << 10) | m;
    sign | ((v + round) as u16)
}
fn tb16(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|&f| f16_bits(f).to_le_bytes()).collect() }

#[test]
fn qwen35_spec_gemm_bm8_narrow_small_m_microbench() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let shapes: &[(usize, usize, &str)] = &[
        (12288, 5120, "full_wq(gate-fused)"),
        (10240, 5120, "gdn_wqkv"),
        (17408, 5120, "ffn_gate_or_up"),
        (5120, 17408, "ffn_down"),
    ];
    let mut gate_fail: Vec<String> = Vec::new();
    for &(m, k, lbl) in shapes {
        for &cand in &[2usize, 3, 4, 5] {
            for (name, bn) in [("bm8n16", 16usize), ("bm8n8", 8usize)] {
                let (ratio, us_multi, us_seq) = run(d, m, k, cand, lbl, bn);
                if ratio > 1.4 {
                    gate_fail.push(format!(
                        "{lbl} {name} cand={cand}: ratio={ratio:.2}x ({name}={us_multi:.1}us vs 1x single baseline, seq{cand}x={us_seq:.1}us)"
                    ));
                }
            }
        }
    }
    if !gate_fail.is_empty() {
        eprintln!("\n=== GATE FAILURES (ratio > 1.4x single-M baseline) ===");
        for f in &gate_fail { eprintln!("  {f}"); }
    } else {
        eprintln!("\n=== ALL SHAPES/cand/BN CLEARED THE <=1.4x GATE ===");
    }
}

fn run(d: &dyn wh_butter_core::Device, m: usize, k: usize, cand: usize, lbl: &str, bn: usize) -> (f64, f64, f64) {
    let mut s = 0x1u32;
    let mut rng = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as f32 / u32::MAX as f32) - 0.5 };
    let w: Vec<f32> = (0..m * k).map(|_| rng()).collect();
    let (qs, sc) = quantize_q4(&w, m, k);
    let qt = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
    let st16 = Tensor::new(d.upload(&tb16(&sc)).unwrap(), vec![sc.len()], DType::F16);

    let weight_bytes = (m * k / 2) as f64 + (m * k / 32 * 2) as f64;
    let it = 50;

    let x1: Vec<f32> = (0..k).map(|_| rng()).collect();
    let xt1 = Tensor::new(d.upload(&tb16(&x1)).unwrap(), vec![k], DType::F16);
    for _ in 0..5 { let _ = gemv_q4(d, &qt, &st16, &xt1, m, k, m).unwrap(); }
    d.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..it { let _ = gemv_q4(d, &qt, &st16, &xt1, m, k, m).unwrap(); }
    d.synchronize().unwrap();
    let us_single = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    let gbs_single = weight_bytes / us_single / 1e3;

    let x_multi: Vec<f32> = (0..cand * k).map(|_| rng()).collect();
    let xt_multi = Tensor::new(d.upload(&tb16(&x_multi)).unwrap(), vec![cand, k], DType::F16);
    let call = |xt: &Tensor| {
        if bn == 16 { gemm_q4_bm8n16(d, xt, &qt, &st16, cand, m, k) } else { gemm_q4_bm8n8(d, xt, &qt, &st16, cand, m, k) }
    };
    for _ in 0..5 { let _ = call(&xt_multi).unwrap(); }
    d.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..it { let _ = call(&xt_multi).unwrap(); }
    d.synchronize().unwrap();
    let us_multi = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    let gbs_multi = weight_bytes / us_multi / 1e3;

    for _ in 0..5 { for _ in 0..cand { let _ = gemv_q4(d, &qt, &st16, &xt1, m, k, m).unwrap(); } }
    d.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..it { for _ in 0..cand { let _ = gemv_q4(d, &qt, &st16, &xt1, m, k, m).unwrap(); } }
    d.synchronize().unwrap();
    let us_seq = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    let gbs_seq = (weight_bytes * cand as f64) / us_seq / 1e3;

    let ratio_vs_single = us_multi / us_single;
    let speedup_vs_seq = us_seq / us_multi;
    let pct_of_single_bw = gbs_multi / gbs_single * 100.0;
    eprintln!(
        "{lbl:24} bn={bn:2} m={m:7} k={k:6} cand={cand}: single={us_single:7.1}us ({gbs_single:6.1} GB/s)  multi={us_multi:8.1}us ({gbs_multi:7.1} GB/s, {pct_of_single_bw:5.1}% of single-M BW)  seq{cand}x={us_seq:8.1}us ({gbs_seq:6.1} GB/s)  ratio(multi/single)={ratio_vs_single:.2}x  speedup(vs seq)={speedup_vs_seq:.2}x"
    );
    (ratio_vs_single, us_multi, us_seq)
}
