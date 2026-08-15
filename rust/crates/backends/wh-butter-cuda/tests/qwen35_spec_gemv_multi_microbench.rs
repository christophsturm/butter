// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! MTP spec-decode perf checkpoint #2: does `gemv_q4_multi` (the small-M
//! batched Q4 GEMV built to replace the `gemm_q4_mpp` dead end -- see
//! `qwen35_spec_gemm_mpp_microbench.rs`'s 15-70x-slower negative result)
//! actually win wall time over `cand` sequential `gemv_q4` calls at the
//! `cand = gamma+1` in {2,3,4,5} range the batched-verify path needs?
//!
//! GATE (per the F-85 final-lever task): `T_multi(cand) <= ~1.4 x
//! T_single` (>= 70% of the M=1 kernel's GB/s), i.e. total time for `cand`
//! columns must NOT scale ~linearly with `cand` -- the weight-DRAM read
//! has to actually amortize.
//!
//! ## VERDICT: GATE FAILED. Task 2 (batched-verify wiring) NOT built.
//!
//! Measured ratio-vs-single-M across all 4 dominant decode shapes: 1.5x-1.9x
//! @ cand=2 up to 3.6x-4.7x @ cand=5 -- every cell misses the <=1.4x bar,
//! and it gets WORSE (not better) at the higher cand values that matter
//! most for a real gamma=3/4 verify round. `gemv_q4_multi` (see its doc in
//! `wh-butter-ops/src/lib.rs`) went through 3 independent iterations before
//! this was accepted as a genuine hardware limit rather than a fixable bug:
//! (1) runtime-`m` loop bound -- register-spill bound, ratio 1.9x-5.3x; (2)
//! `#[kernel(variants(...))]` compile-time-specialized `M` (register-
//! promoted, matches the GDN `d128_*` fast-kernel precedent) -- real but
//! modest improvement, still 1.5x-4.5x; (3) `rows_per_tg` sweep 1/2/4/8
//! (the documented latency-hiding lever for the M=1 kernel itself) -- ZERO
//! measurable effect. Conclusion: GB10's per-SM instruction-issue rate for
//! the per-candidate dequant+FMA loop is the actual bottleneck once
//! `cand>1`, not DRAM bandwidth -- there is no slack to amortize into. This
//! is a second, independent confirmation of the same class of dead end
//! `gemm_q4_mpp` hit (far less severe here -- 1.5-4.7x vs 15-70x -- but
//! still well short of the gate). Per the task's explicit instruction
//! ("if the gate fails after 2-3 iterations, STOP and report honestly --
//! do not wire a losing kernel"), `speculative_round` in `qwen35.rs` was
//! NOT touched; it stays on the lazy-sequential verify form.
//!
//! Same 4 shapes as the gemm_q4_mpp checkpoint (round-2 per-shape census,
//! `qwen35_gemv_shapes.rs`) for direct apples-to-apples comparison against
//! that documented result. x is F32 (matches qwen35.rs's decode path --
//! host activations are F32 end to end, see that module's doc), so this
//! bench is the true zero-extra-cast comparison a real verify call would see.
use std::time::Instant;
use wh_butter_core::{DType, Tensor};
use wh_butter_cuda::CudaDevice;
use wh_butter_ops::{gemv_q4, gemv_q4_multi, quantize_q4};

fn tbf(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tbu(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
// Manual f32->f16 narrowing (same bit trick as qwen35.rs's f32s_to_f16_bytes
// and the gemm_q4_mpp microbench's f16_bits) -- avoids a `half` dev-dep.
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
fn qwen35_spec_gemv_multi_small_m_microbench() {
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
            let (ratio, us_multi, us_seq) = run(d, m, k, cand, lbl);
            // Gate: multi(cand) time <= 1.4x single (cand=1) time, i.e.
            // ratio = us_multi / us_single_baseline <= 1.4.
            if ratio > 1.4 {
                gate_fail.push(format!("{lbl} cand={cand}: ratio={ratio:.2}x (multi={us_multi:.1}us vs 1x single baseline, seq{cand}x={us_seq:.1}us)"));
            }
        }
    }
    if !gate_fail.is_empty() {
        eprintln!("\n=== GATE FAILURES (ratio > 1.4x single-M baseline) ===");
        for f in &gate_fail { eprintln!("  {f}"); }
    } else {
        eprintln!("\n=== ALL SHAPES/cand CLEARED THE <=1.4x GATE ===");
    }
}

fn run(d: &dyn wh_butter_core::Device, m: usize, k: usize, cand: usize, lbl: &str) -> (f64, f64, f64) {
    let mut s = 0x1u32;
    let mut rng = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as f32 / u32::MAX as f32) - 0.5 };
    let w: Vec<f32> = (0..m * k).map(|_| rng()).collect();
    let (qs, sc) = quantize_q4(&w, m, k);
    let qt = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
    let st16 = Tensor::new(d.upload(&tb16(&sc)).unwrap(), vec![sc.len()], DType::F16);

    let weight_bytes = (m * k / 2) as f64 + (m * k / 32 * 2) as f64;
    let it = 50;

    // ---- baseline: ONE single-M gemv_q4 dispatch (cand=1 cost floor). ----
    let x1: Vec<f32> = (0..k).map(|_| rng()).collect();
    let xt1 = Tensor::new(d.upload(&tbf(&x1)).unwrap(), vec![k], DType::F32);
    for _ in 0..5 { let _ = gemv_q4(d, &qt, &st16, &xt1, m, k, m).unwrap(); }
    d.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..it { let _ = gemv_q4(d, &qt, &st16, &xt1, m, k, m).unwrap(); }
    d.synchronize().unwrap();
    let us_single = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    let gbs_single = weight_bytes / us_single / 1e3;

    // ---- gemv_q4_multi: ONE dispatch over `cand` candidate columns. ----
    let x_multi: Vec<f32> = (0..cand * k).map(|_| rng()).collect();
    let xt_multi = Tensor::new(d.upload(&tbf(&x_multi)).unwrap(), vec![cand, k], DType::F32);
    for _ in 0..5 { let _ = gemv_q4_multi(d, &qt, &st16, &xt_multi, m, k, cand).unwrap(); }
    d.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..it { let _ = gemv_q4_multi(d, &qt, &st16, &xt_multi, m, k, cand).unwrap(); }
    d.synchronize().unwrap();
    let us_multi = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    let gbs_multi = weight_bytes / us_multi / 1e3; // weight read ONCE regardless of cand

    // ---- sequential gemv_q4: `cand` dispatches, one row of x each. ----
    for _ in 0..5 { for _ in 0..cand { let _ = gemv_q4(d, &qt, &st16, &xt1, m, k, m).unwrap(); } }
    d.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..it { for _ in 0..cand { let _ = gemv_q4(d, &qt, &st16, &xt1, m, k, m).unwrap(); } }
    d.synchronize().unwrap();
    let us_seq = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    let gbs_seq = (weight_bytes * cand as f64) / us_seq / 1e3;

    let ratio_vs_single = us_multi / us_single; // the GATE metric
    let speedup_vs_seq = us_seq / us_multi;
    let pct_of_single_bw = gbs_multi / gbs_single * 100.0;
    eprintln!(
        "{lbl:24} m={m:7} k={k:6} cand={cand}: single={us_single:7.1}us ({gbs_single:6.1} GB/s)  multi={us_multi:8.1}us ({gbs_multi:7.1} GB/s, {pct_of_single_bw:5.1}% of single-M BW)  seq{cand}x={us_seq:8.1}us ({gbs_seq:6.1} GB/s)  ratio(multi/single)={ratio_vs_single:.2}x  speedup(vs seq)={speedup_vs_seq:.2}x"
    );
    (ratio_vs_single, us_multi, us_seq)
}
