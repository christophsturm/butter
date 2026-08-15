// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! MTP spec-decode perf checkpoint: does `gemm_q4_mpp` (existing, already
//! wired batched-M Q4 GEMM -- see `wh_butter_ops::gemm_q4_mpp`'s doc, it
//! bounds-checks `n_rows` internally so small M needs no padding) actually
//! win bandwidth over M sequential `gemv_q4` calls at the M = gamma+1 in
//! {2,3,4,5} range this port's speculative-decode verify pass would use?
//! This is the explicit "measure before building the driver" checkpoint --
//! confirms the win exists (or doesn't) before wiring batched verify.
//!
//! x is F32 host-side in this port's decode path (see qwen35.rs's module
//! doc), but `gemm_q4_mpp`'s coop_tile MMA path is templated over T =
//! F16/BF16/F32 -- this bench runs at F32 (`x.dtype = DType::F32`) to match
//! what a real verify call would pass WITHOUT an extra cast, so the
//! reported numbers are the true zero-extra-risk comparison (an F16-cast
//! variant would need its own correctness re-gate against the oracle, not
//! attempted here).
use std::time::Instant;
use wh_butter_core::{DType, Tensor};
use wh_butter_cuda::CudaDevice;
use wh_butter_ops::{gemm_q4_mpp, gemv_q4, quantize_q4};

fn tbf(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tbu(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
// Manual f32->f16 narrowing (same bit trick as qwen35.rs's f32s_to_f16_bytes)
// -- avoids adding a `half` crate dev-dependency just for this microbench.
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
fn qwen35_spec_gemm_mpp_small_m_microbench() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    // The 4 biggest / most representative decode GEMV shapes (m_out, k_in)
    // from the round-2 per-shape census (qwen35_gemv_shapes.rs) -- these
    // dominate decode-step wall time, so they're what a batched verify pass
    // needs to win on.
    let shapes: &[(usize, usize, &str)] = &[
        (12288, 5120, "full_wq(gate-fused)"),
        (10240, 5120, "gdn_wqkv"),
        (17408, 5120, "ffn_gate_or_up"),
        (5120, 17408, "ffn_down"),
    ];
    for &(m, k, lbl) in shapes {
        for &rows in &[1usize, 2, 3, 4, 5] {
            run(d, m, k, rows, lbl);
        }
    }
}

fn run(d: &dyn wh_butter_core::Device, m: usize, k: usize, rows: usize, lbl: &str) {
    let mut s = 0x1u32;
    let mut rng = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as f32 / u32::MAX as f32) - 0.5 };
    let w: Vec<f32> = (0..m * k).map(|_| rng()).collect();
    let (qs, sc) = quantize_q4(&w, m, k);
    let qt = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
    // gemm_q4_mpp reads scales as f16 (coop_stage path); gemv_q4 also reads
    // f16 -- upload a REAL f32->f16 narrowing so both kernels see sane
    // (non-garbage) values. Doesn't affect timing, just avoids any NaN-
    // triggered slow-path surprises.
    let st16 = Tensor::new(d.upload(&tb16(&sc)).unwrap(), vec![sc.len()], DType::F16);

    // Bytes moved: Q4 nibbles (0.5 B/param) + f16 per-32-block scale
    // (2B/block = 0.0625 B/param), read ONCE regardless of `rows` for the
    // batched path (that's the entire point) vs `rows` TIMES for the
    // sequential-gemv path.
    let weight_bytes = (m * k / 2) as f64 + (m * k / 32 * 2) as f64;

    // ---- gemm_q4_mpp: ONE dispatch over `rows` rows. ----
    let x_mpp: Vec<f32> = (0..rows * k).map(|_| rng()).collect();
    let xt_mpp = Tensor::new(d.upload(&tbf(&x_mpp)).unwrap(), vec![rows, k], DType::F32);
    for _ in 0..5 { let _ = gemm_q4_mpp(d, &xt_mpp, &qt, &st16, rows, m, k).unwrap(); }
    d.synchronize().unwrap();
    let it = 50;
    let t = Instant::now();
    for _ in 0..it { let _ = gemm_q4_mpp(d, &xt_mpp, &qt, &st16, rows, m, k).unwrap(); }
    d.synchronize().unwrap();
    let us_mpp = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    let gbs_mpp = weight_bytes / us_mpp / 1e3;

    // ---- sequential gemv_q4: `rows` dispatches, one row of x each. ----
    let st32 = Tensor::new(d.upload(&tb16(&sc)).unwrap(), vec![sc.len()], DType::F16);
    let x1: Vec<f32> = (0..k).map(|_| rng()).collect();
    let xt1 = Tensor::new(d.upload(&tbf(&x1)).unwrap(), vec![k], DType::F32);
    for _ in 0..5 { for _ in 0..rows { let _ = gemv_q4(d, &qt, &st32, &xt1, m, k, m).unwrap(); } }
    d.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..it { for _ in 0..rows { let _ = gemv_q4(d, &qt, &st32, &xt1, m, k, m).unwrap(); } }
    d.synchronize().unwrap();
    let us_seq = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    let gbs_seq = (weight_bytes * rows as f64) / us_seq / 1e3;

    let speedup = us_seq / us_mpp;
    eprintln!(
        "{lbl:24} m={m:7} k={k:6} rows={rows}: mpp={us_mpp:8.1}us ({gbs_mpp:7.1} GB/s)  seq{rows}x={us_seq:8.1}us ({gbs_seq:7.1} GB/s)  speedup={speedup:.2}x"
    );
}
