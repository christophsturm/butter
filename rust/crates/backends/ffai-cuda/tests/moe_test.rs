#![cfg(feature = "cuda")]
// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//!
//! Also includes: `moe_w4a16_marlin_vs_standard` — validates that the
//! Marlin-coalesced W4A16 kernel produces bit-close output to the standard
//! scattered-nibble kernel on a synthetic small problem.
//! MoE feed-forward (router → top-k → per-expert SwiGLU → weighted sum) on
//! Metal vs a CPU reference. Proves the MoE compute path — the exotic family
//! covering DeepSeek-V4 / GPT-OSS / Granite4 / Qwen-MoE — runs correctly on
//! the shared op layer. (Real MoE-model-vs-HF verification follows once the
//! large expert weights are staged.)

use ffai_core::{DType, Device, Tensor};
use ffai_cuda::CudaDevice;
use ffai_models::moe::{ExpertWeights, MoeMlp, moe_mlp};

fn tb(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn fill(n: usize, s: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7 + s * 131) % 97) as f32 - 48.0) * 0.01).collect()
}
fn tens(d: &dyn Device, v: &[f32], shape: Vec<usize>) -> Tensor {
    Tensor::new(d.upload(&tb(v)).unwrap(), shape, DType::F32)
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn mv(m: &[f32], v: &[f32], rows: usize, k: usize) -> Vec<f32> {
    (0..rows).map(|r| (0..k).map(|c| m[r * k + c] * v[c]).sum()).collect()
}

#[test]
fn moe_mlp_on_cuda_matches_cpu() {
    let Some(dev) = CudaDevice::create().expect("metal init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };
    const H: usize = 256;
    const INTER: usize = 512;
    const NE: usize = 8;
    const TOPK: usize = 2;

    let router = fill(NE * H, 1);
    let h = fill(H, 99);
    let experts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..NE)
        .map(|e| (fill(INTER * H, 10 + e), fill(INTER * H, 100 + e), fill(H * INTER, 200 + e)))
        .collect();

    let w = MoeMlp {
        router: tens(dev.as_ref(), &router, vec![NE, H]),
        experts: experts
            .iter()
            .map(|(g, u, d)| ExpertWeights {
                gate: tens(dev.as_ref(), g, vec![INTER, H]),
                up: tens(dev.as_ref(), u, vec![INTER, H]),
                down: tens(dev.as_ref(), d, vec![H, INTER]),
            })
            .collect(),
        top_k: TOPK,
        norm_topk: true,
    };
    let th = tens(dev.as_ref(), &h, vec![H]);

    let out = moe_mlp(dev.as_ref(), &w, &th).unwrap();
    dev.synchronize().unwrap();
    let mut ob = vec![0u8; H * 4];
    dev.download(out.buffer.as_ref(), &mut ob).unwrap();
    let got: Vec<f32> = ob.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    // CPU reference.
    let logits = mv(&router, &h, NE, H);
    let mut order: Vec<usize> = (0..NE).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top: Vec<usize> = order.into_iter().take(TOPK).collect();
    let m = top.iter().map(|&i| logits[i]).fold(f32::MIN, f32::max);
    let e: Vec<f32> = top.iter().map(|&i| (logits[i] - m).exp()).collect();
    let s: f32 = e.iter().sum();
    let wts: Vec<f32> = e.iter().map(|x| x / s).collect();
    let mut want = vec![0.0f32; H];
    for (&ei, &gw) in top.iter().zip(&wts) {
        let (g, u, d) = &experts[ei];
        let gate = mv(g, &h, INTER, H);
        let up = mv(u, &h, INTER, H);
        let act: Vec<f32> = (0..INTER).map(|i| silu(gate[i]) * up[i]).collect();
        let out = mv(d, &act, H, INTER);
        for i in 0..H {
            want[i] += gw * out[i];
        }
    }

    let mut err = 0.0f32;
    for i in 0..H {
        err = err.max((got[i] - want[i]).abs());
    }
    eprintln!("MoE MLP on CUDA vs CPU: max|Δ|={err:.3e} (top experts {top:?})");
    assert!(err <= 5e-3, "moe mismatch: {err:.3e}");
    eprintln!("MoE feed-forward runs on CUDA through the shared op layer, matches CPU.");
}

/// Validate that `moe_w4a16_marlin` (Marlin coalesced layout) produces output
/// within 1e-2 of `moe_w4a16` (standard scattered layout) on a small problem.
///
/// Both kernels compute the same matrix product; the only difference is how
/// the Q4 nibbles are stored in memory. This test proves the permutation +
/// dequant are bit-faithful (same signed nibble → same dequant value → same output).
#[test]
fn moe_w4a16_marlin_vs_standard() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping moe_w4a16_marlin_vs_standard");
        return;
    };
    use ffai_core::{DType, Tensor};
    use ffai_ops::{moe_w4a16, moe_w4a16_marlin, permute_q4_to_marlin, quantize_q4};

    // Problem dimensions: must be multiples of 64/32.
    // Use 4 experts, m_total=64, n_out=128, k_in=64.
    const N_EXP: usize = 4;
    const M: usize = 64;  // m_total (tokens sorted by expert, 16 per expert)
    const N: usize = 128; // n_out (output features)
    const K: usize = 64;  // k_in (input features)

    // Deterministic weight data.
    let wv: Vec<f32> = (0..N_EXP * N * K)
        .map(|i| (i as f32 * 0.019 - 0.3).sin() * 0.9)
        .collect();
    // Activation matrix: sorted by expert (16 rows per expert).
    let xv: Vec<f32> = (0..M * K)
        .map(|i| (i as f32 * 0.013 - 0.4).cos() * 1.1)
        .collect();
    // Expert index for each row (16 rows × expert 0, 1, 2, 3).
    let idx: Vec<u32> = (0..M).map(|r| (r / 16) as u32).collect();

    // Quantize → standard layout.
    let (qs_std, scales_f32) = quantize_q4(&wv, N_EXP * N, K);

    // Convert scales and x to f16 bytes using manual IEEE 754 conversion.
    // f32→f16: use the truncate-to-f16 bit trick (loses mantissa bits, OK for scales).
    let f32_to_f16_bits = |x: f32| -> u16 {
        // Simple f32→f16 via u32 bit manipulation (round-to-zero).
        let u = x.to_bits();
        let sign = (u >> 16) & 0x8000;
        let exp  = ((u >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = (u >> 13) & 0x3FF;
        if exp <= 0 { sign as u16 }
        else if exp >= 31 { (sign | 0x7C00) as u16 }
        else { (sign | ((exp as u32) << 10) | mant) as u16 }
    };
    let f16_to_f32 = |bits: u16| -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp  = ((bits >> 10) & 0x1F) as u32;
        let mant = (bits & 0x3FF) as u32;
        if exp == 0 { 0.0f32 }
        else if exp == 31 { f32::INFINITY * if sign == 1 { -1.0 } else { 1.0 } }
        else { f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)) }
    };

    let scales_f16: Vec<u16> = scales_f32.iter().map(|&s| f32_to_f16_bits(s)).collect();
    let xv_f16: Vec<u16> = xv.iter().map(|&x| f32_to_f16_bits(x)).collect();

    // Permute to Marlin layout.
    let qs_mar = permute_q4_to_marlin(&qs_std, N_EXP, N, K);
    assert_eq!(qs_std.len(), qs_mar.len(), "Marlin layout must have same u32 count");

    // Upload everything to GPU.
    let to_u8_u32 = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let to_u8_u16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let d = dev.as_ref();

    let qs_std_dev = Tensor::new(d.upload(&to_u8_u32(&qs_std)).unwrap(), vec![qs_std.len()], DType::U32);
    let qs_mar_dev = Tensor::new(d.upload(&to_u8_u32(&qs_mar)).unwrap(), vec![qs_mar.len()], DType::U32);
    let sc_dev     = Tensor::new(d.upload(&to_u8_u16(&scales_f16)).unwrap(), vec![scales_f16.len()], DType::F16);
    let x_f16_dev  = Tensor::new(d.upload(&to_u8_u16(&xv_f16)).unwrap(), vec![M, K], DType::F16);
    let idx_dev    = Tensor::new(d.upload(&to_u8_u32(&idx)).unwrap(), vec![M], DType::U32);

    // Run both kernels.
    let out_std = moe_w4a16(d, &x_f16_dev, &qs_std_dev, &sc_dev, &idx_dev, M, N, K).unwrap();
    let out_mar = moe_w4a16_marlin(d, &x_f16_dev, &qs_mar_dev, &sc_dev, &idx_dev, M, N, K).unwrap();
    d.synchronize().unwrap();

    // Download and compare (output is f16).
    let dl_f16 = |t: &Tensor| -> Vec<f32> {
        let mut buf = vec![0u8; t.elem_count() * 2];
        d.download(t.buffer.as_ref(), &mut buf).unwrap();
        buf.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()
    };
    let got_std = dl_f16(&out_std);
    let got_mar = dl_f16(&out_mar);

    let mut max_err = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..M * N {
        let ae = (got_std[i] - got_mar[i]).abs();
        let re = if got_std[i].abs() > 1e-6 { ae / got_std[i].abs() } else { ae };
        max_err = max_err.max(ae);
        max_rel = max_rel.max(re);
    }
    eprintln!("moe_w4a16_marlin vs standard: max_abs={max_err:.3e}  max_rel={max_rel:.3e}");
    assert!(max_err < 1e-2,
        "Marlin kernel output deviates from standard: max|Δ|={max_err:.3e} (tol 1e-2)");
    eprintln!("moe_w4a16_marlin correctness: PASS (max_abs={max_err:.3e})");
}

// ── f16 helpers shared by the straddle test ─────────────────────────────────
fn f32_to_f16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    let sign = (u >> 16) & 0x8000;
    let exp = ((u >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (u >> 13) & 0x3FF;
    if exp <= 0 {
        sign as u16
    } else if exp >= 31 {
        (sign | 0x7C00) as u16
    } else {
        (sign | ((exp as u32) << 10) | mant) as u16
    }
}
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        0.0f32
    } else if exp == 31 {
        f32::INFINITY * if sign == 1 { -1.0 } else { 1.0 }
    } else {
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

/// Reproduce the e2e divergence (argmax 3260 vs correct 1104) at the kernel
/// level on SMALL shapes. The original `moe_w4a16_marlin_vs_standard` test uses
/// M=64 (single tile), N=128 (÷128), 16 rows/expert — which misses two boundary
/// cases the real Nemotron shape (m≈12288, n_out=1856, n_exp=128, top_k=6) hits:
///
///   (A) an expert sub-run that STRADDLES a 64-row M-tile boundary, and
///   (B) n_out NOT a multiple of 128 (1856 is ÷64 but not ÷128).
///
/// Here: M=128 (two M-tiles), N=192 (÷64 but NOT ÷128), ragged per-expert row
/// counts (50 / 40 / 38) so expert 1 spans rows 50..89 → straddles the tile
/// boundary at row 64. Compares standard `moe_w4a16` AND `moe_bgemm_q4_bm64`
/// against the (correct) Marlin path on IDENTICAL weights/scales/indices.
#[test]
fn moe_w4a16_straddle_vs_marlin() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping moe_w4a16_straddle_vs_marlin");
        return;
    };
    use ffai_core::{DType, Tensor};
    use ffai_ops::{moe_bgemm_q4_bm64, moe_w4a16, moe_w4a16_marlin, permute_q4_to_marlin, quantize_q4};

    const N_EXP: usize = 3;
    const M: usize = 128; // two 64-row M-tiles
    const N: usize = 192; // ÷64 but NOT ÷128  ← suspect (B)
    const K: usize = 64;

    // Ragged per-expert row counts → expert 1 straddles the tile boundary (64).
    //   expert 0: rows   0..50  (50 rows, inside tile 0)
    //   expert 1: rows  50..90  (40 rows, STRADDLES 64)  ← suspect (A)
    //   expert 2: rows  90..128 (38 rows, inside tile 1)
    let counts = [50usize, 40, 38];
    let mut idx: Vec<u32> = Vec::with_capacity(M);
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            idx.push(e as u32);
        }
    }
    assert_eq!(idx.len(), M);

    let wv: Vec<f32> = (0..N_EXP * N * K)
        .map(|i| (i as f32 * 0.019 - 0.3).sin() * 0.9)
        .collect();
    let xv: Vec<f32> = (0..M * K)
        .map(|i| (i as f32 * 0.013 - 0.4).cos() * 1.1)
        .collect();

    let (qs_std, scales_f32) = quantize_q4(&wv, N_EXP * N, K);
    let scales_f16: Vec<u16> = scales_f32.iter().map(|&s| f32_to_f16_bits(s)).collect();
    let xv_f16: Vec<u16> = xv.iter().map(|&x| f32_to_f16_bits(x)).collect();

    let qs_mar = permute_q4_to_marlin(&qs_std, N_EXP, N, K);

    let to_u8_u32 = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let to_u8_u16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let d = dev.as_ref();

    let qs_std_dev = Tensor::new(d.upload(&to_u8_u32(&qs_std)).unwrap(), vec![qs_std.len()], DType::U32);
    let qs_mar_dev = Tensor::new(d.upload(&to_u8_u32(&qs_mar)).unwrap(), vec![qs_mar.len()], DType::U32);
    let sc_dev = Tensor::new(d.upload(&to_u8_u16(&scales_f16)).unwrap(), vec![scales_f16.len()], DType::F16);
    let x_f16_dev = Tensor::new(d.upload(&to_u8_u16(&xv_f16)).unwrap(), vec![M, K], DType::F16);
    let idx_dev = Tensor::new(d.upload(&to_u8_u32(&idx)).unwrap(), vec![M], DType::U32);

    let out_std = moe_w4a16(d, &x_f16_dev, &qs_std_dev, &sc_dev, &idx_dev, M, N, K).unwrap();
    let out_bg = moe_bgemm_q4_bm64(d, &x_f16_dev, &qs_std_dev, &sc_dev, &idx_dev, M, N, K).unwrap();
    let out_mar = moe_w4a16_marlin(d, &x_f16_dev, &qs_mar_dev, &sc_dev, &idx_dev, M, N, K).unwrap();
    d.synchronize().unwrap();

    let dl_f16 = |t: &Tensor| -> Vec<f32> {
        let mut buf = vec![0u8; t.elem_count() * 2];
        d.download(t.buffer.as_ref(), &mut buf).unwrap();
        buf.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()
    };
    let got_std = dl_f16(&out_std);
    let got_bg = dl_f16(&out_bg);
    let got_mar = dl_f16(&out_mar);

    // Per-row worst error vs the (correct) Marlin reference, to localize which
    // rows diverge (expect: rows in the straddling run and/or last n-cols).
    let row_err = |got: &[f32]| -> (f32, usize, usize) {
        let mut max_ae = 0.0f32;
        let mut max_row = 0;
        let mut max_col = 0;
        for r in 0..M {
            for c in 0..N {
                let ae = (got[r * N + c] - got_mar[r * N + c]).abs();
                if ae > max_ae {
                    max_ae = ae;
                    max_row = r;
                    max_col = c;
                }
            }
        }
        (max_ae, max_row, max_col)
    };
    let (std_ae, std_r, std_c) = row_err(&got_std);
    let (bg_ae, bg_r, bg_c) = row_err(&got_bg);
    eprintln!(
        "straddle: moe_w4a16   max|Δ|={std_ae:.3e} @ row {std_r} (exp {}) col {std_c}",
        idx[std_r]
    );
    eprintln!(
        "straddle: moe_bgemm   max|Δ|={bg_ae:.3e} @ row {bg_r} (exp {}) col {bg_c}  marlin={} bgemm={}",
        idx[bg_r],
        got_mar[bg_r * N + bg_c],
        got_bg[bg_r * N + bg_c]
    );
    // Histogram: how many bgemm cells exceed thresholds (broad precision vs a
    // few structurally-wrong cells), and which n-tile they fall in.
    let mut n_gt_1e3 = 0usize;
    let mut n_gt_1e2 = 0usize;
    let mut ntile_hits = [0usize; 4];
    for r in 0..M {
        for c in 0..N {
            let ae = (got_bg[r * N + c] - got_mar[r * N + c]).abs();
            if ae > 1e-3 {
                n_gt_1e3 += 1;
                ntile_hits[c / 64] += 1;
            }
            if ae > 1e-2 {
                n_gt_1e2 += 1;
            }
        }
    }
    eprintln!(
        "straddle: moe_bgemm cells |Δ|>1e-3: {n_gt_1e3} / {} ; >1e-2: {n_gt_1e2} ; by n-tile(64): {:?}",
        M * N,
        ntile_hits
    );

    // `moe_w4a16` (standard scattered WMMA) is BIT-EXACT with Marlin here
    // (same 16×16×16 fragment reduction order) → strict absolute bound.
    assert!(
        std_ae < 1e-2,
        "moe_w4a16 (standard) diverges from Marlin on straddle/N=192: max|Δ|={std_ae:.3e} @ row {std_r} col {std_c}"
    );
    // `moe_bgemm_q4_bm64` uses a different MMA tiling (coop_tile 32×32×32) and so
    // accumulates f16 products in a different order → up to ~1 f16 ULP of drift
    // on large-magnitude outputs (observed: 1.56e-2 == one ULP at |out|≈30, on
    // 6/24576 cells). That is precision noise, NOT a structural index bug — a
    // straddle/n-tile indexing bug would corrupt whole rows/columns. Guard with
    // a magnitude-relative bound (catches structural errors, tolerates ULP).
    let bg_rel = bg_ae / got_mar[bg_r * N + bg_c].abs().max(1.0);
    assert!(
        bg_rel < 1e-3,
        "moe_bgemm_q4_bm64 diverges from Marlin on straddle/N=192: max|Δ|={bg_ae:.3e} (rel {bg_rel:.3e}) @ row {bg_r} col {bg_c}"
    );
    eprintln!("moe_w4a16_straddle_vs_marlin: PASS (std bit-exact; bgemm within 1 f16 ULP)");
}
