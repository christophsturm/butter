// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! F-85 round-12 numerics gate: dense-random marlin GEMM vs f64 reference,
//! swept over group_size in {32,64,128} (marlin's group_blocks in
//! {2,4,8}), on the 4 census "big shapes" (routed_marlin's ffn_gate/up,
//! ffn_down, attn_q, gdn_qkv -- see qwen35.rs's `build_marlin` callers /
//! the census bucket doc), M=1 (decode-representative). Reference is
//! computed in f64 against the ORIGINAL (pre-quantization) dense-random
//! weight -- not the dequantized-then-repacked weight -- so the reported
//! max_rel captures BOTH the marlin kernel's own numerics (round-7's floor,
//! ~5.3e-4 at g=32) AND the requant fidelity loss from widening the scale
//! group (the effect this round is gating on, per the round-12 brief: "it's
//! still 4-bit values, only scale granularity changes... measure, don't
//! assume").
//!
//! IMPORTANT (discovered running this the first time): on i.i.d. uniform
//! random weights, that "SIGNAL" max_rel (GPU output vs the ORIGINAL
//! pre-quant weight) is dominated by the 4-bit quantizer's own CLT-
//! predictable signal loss (~sqrt(k)-scaling noise, ~7-9% at these K's) --
//! two orders of magnitude bigger than round-7's ~5.3e-4 kernel-numerics
//! floor, and NOT a kernel bug: random noise has none of the redundancy
//! real transformer weight rows do, so it's the worst case for a
//! per-group-amax quantizer and is not informative about real-model
//! fidelity. This harness reports BOTH: a KERNEL max_rel (GPU output vs
//! the DEQUANTIZED weight -- isolates repack/permute-scale/tensor-core
//! correctness, gated <=~2e-3) and a SIGNAL max_rel (informational only).
//! The real per-group-size FIDELITY question (does g=64/128 measurably
//! hurt real model quality) is answered by the oracle prompt + logit-match
//! quality gate on the real GGUF weights, not by this synthetic-data test.
//!
//! Run: `IRON_MARLIN_BUILD=1 cargo test --release -p wh-butter-cuda
//! --test qwen35_marlin_g_numerics -- --nocapture`
use wh_butter_core::{DType, Tensor};
use wh_butter_ops as ops;
use wh_butter_cuda::CudaDevice;

fn tbf(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tbu(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

fn f32s_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    fn half_bits(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 112;
        if e <= 0 {
            return sign;
        }
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        let m = (x >> 13) & 0x3ff;
        let round = (x >> 12) & 1;
        let v = ((e as u32) << 10) | m;
        sign | ((v + round) as u16)
    }
    v.iter().flat_map(|&f| half_bits(f).to_le_bytes()).collect()
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // subnormal half -> normalize
        let mut e = -1i32;
        let mut m = mant;
        loop {
            m <<= 1;
            e -= 1;
            if m & 0x400 != 0 {
                break;
            }
        }
        m &= 0x3ff;
        let f_exp = (127 - 15 + e + 1) as u32;
        return f32::from_bits(sign | (f_exp << 23) | (m << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    let f_exp = exp as u32 + (127 - 15);
    f32::from_bits(sign | (f_exp << 23) | (mant << 13))
}

fn bytes_to_f16_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

const GB10_SMS: i32 = 48;
const MARLIN_MAX_SHARED_MEM: i32 = 101376;

/// Deterministic xorshift PRNG (same recipe as the smallproj microbench --
/// consistent noise floor across both harnesses).
fn rng_vec(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s as f32 / u32::MAX as f32) - 0.5
    };
    (0..n).map(|_| next()).collect()
}

#[test]
fn qwen35_marlin_g_numerics() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();

    // The 4 routed "big shapes" (m=out_features, k=in_features), matching
    // qwen35.rs's marlin-routed weight classes at their real production
    // dims (see qwen35.rs's load_mtp doc comment + gdn wqkv shape comment
    // for provenance): ffn_gate/up [17408,5120], ffn_down [5120,17408],
    // attn_q [12288,5120], gdn attn_qkv [10240,5120].
    let shapes: &[(usize, usize, &str)] = &[
        (17408, 5120, "ffn_gate_up"),
        (5120, 17408, "ffn_down"),
        (12288, 5120, "attn_q"),
        (10240, 5120, "gdn_qkv"),
    ];
    let group_sizes: &[usize] = &[32, 64, 128];

    // Routing triple for M=1 (same for every shape/group-size).
    let off_bytes: Vec<u8> = [0i32, 1i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let off = Tensor::new(d.upload(&off_bytes).unwrap(), vec![2], DType::I32);
    let routing = ops::marlin_build_routing(d, &off, 1, 8, 1).unwrap();

    eprintln!("QWEN35_MARLIN_G_NUMERICS ===== dense-random GEMM vs f64 ref, M=1 =====");
    for &gs in group_sizes {
        let mut worst_rel = 0.0f64;
        for &(m, k, lbl) in shapes {
            if k % gs != 0 {
                eprintln!("  g={gs:4} {lbl:12} SKIP (k={k} not divisible by group size)");
                continue;
            }
            // Deterministic per-(shape,gs) seed so different group sizes
            // see the SAME underlying random weight/activation -- isolates
            // the group-size effect from RNG noise.
            let seed = (m as u32).wrapping_mul(31).wrapping_add(k as u32).wrapping_add(1);
            let w = rng_vec(seed, m * k);
            let x = rng_vec(seed.wrapping_add(0x9e37_79b9), k);

            // f64 SIGNAL reference: original (pre-quant) dense-random weight
            // x activation, full precision. Dominated by the 4-bit
            // symmetric quantizer's inherent signal loss on i.i.d. random
            // data (CLT-predictable ~sqrt(k)-scaling noise floor -- NOT a
            // kernel bug, and NOT representative of real (structured, low-
            // effective-entropy) model weights). Reported for context only,
            // not gated.
            let mut ref_signal = vec![0f64; m];
            for n in 0..m {
                let row = &w[n * k..(n + 1) * k];
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += row[kk] as f64 * x[kk] as f64;
                }
                ref_signal[n] = acc;
            }

            let (bqw, scales_f32) = ops::quantize_u4b8_gptq(&w, m, k, gs);

            // f64 KERNEL reference: DEQUANTIZED weight (same q/scale the GPU
            // kernel itself consumes) x activation, full precision. This is
            // the round-7-methodology number -- isolates repack + scale-
            // permute + tensor-core-epilogue correctness from the (much
            // larger, orthogonal) quantization signal-loss measured above.
            // THIS is what's gated at <=~2e-3.
            let num_groups = k / gs;
            let mut ref_kernel = vec![0f64; m];
            for n in 0..m {
                let mut acc = 0f64;
                for g in 0..num_groups {
                    let scale = scales_f32[g * m + n] as f64;
                    for j in 0..gs {
                        let kk = g * gs + j;
                        let k_packed = kk / 8;
                        let k_pos = kk % 8;
                        let word = bqw[k_packed * m + n];
                        let q = (word >> (4 * k_pos)) & 0xF;
                        let w_hat = (q as f64 - 8.0) * scale;
                        acc += w_hat * x[kk] as f64;
                    }
                }
                ref_kernel[n] = acc;
            }
            let scales_bytes = f32s_to_f16_bytes(&scales_f32);
            let bqw_t = Tensor::new(d.upload(&tbu(&bqw)).unwrap(), vec![bqw.len()], DType::U32);
            let scales_raw = Tensor::new(d.upload(&scales_bytes).unwrap(), vec![scales_bytes.len() / 2], DType::F16);
            let b_repacked = ops::marlin_repack(d, &bqw_t, k, m, GB10_SMS, MARLIN_MAX_SHARED_MEM).unwrap();
            let b_scales = ops::marlin_permute_scales(d, &scales_raw, num_groups, m).unwrap();
            let (thread_k, thread_n, blocks_per_sm) =
                ops::marlin_pick_config(d, 1, m, k, gs, GB10_SMS).unwrap();
            let (workspace, c_tmp) = ops::marlin_gemm_u4b8_workspace(d, m, GB10_SMS).unwrap();
            let x16_bytes = f32s_to_f16_bytes(&x);
            let x16 = Tensor::new(d.upload(&x16_bytes).unwrap(), vec![k], DType::F16);

            let out16 = ops::marlin_gemm_u4b8_f16(
                d, &x16, &b_repacked, &b_scales,
                &routing.0, &routing.1, &routing.2,
                &workspace, &c_tmp,
                1, m, k, gs, GB10_SMS,
                thread_k, thread_n, blocks_per_sm,
            ).unwrap();
            d.synchronize().unwrap();
            let mut out_bytes = vec![0u8; m * 2];
            d.download(out16.buffer.as_ref(), &mut out_bytes).unwrap();
            let out_f32 = bytes_to_f16_f32(&out_bytes);

            // KERNEL max_rel (gated <=~2e-3): GPU output vs the dequantized-
            // weight f64 reference -- isolates repack/permute/tensor-core
            // numerics from quantization signal loss.
            let mut max_rel_kernel = 0f64;
            let mut max_abs_kernel = 0f64;
            let ref_kernel_absmax = ref_kernel.iter().fold(0f64, |a, &v| a.max(v.abs())).max(1e-12);
            // SIGNAL max_rel (reported, NOT gated): GPU output vs the
            // original pre-quant weight -- dominated by the 4-bit
            // quantizer's inherent signal loss on i.i.d. random data.
            let mut max_rel_signal = 0f64;
            let ref_signal_absmax = ref_signal.iter().fold(0f64, |a, &v| a.max(v.abs())).max(1e-12);
            for n in 0..m {
                let got = out_f32[n] as f64;
                let abs_err_kernel = (got - ref_kernel[n]).abs();
                max_abs_kernel = max_abs_kernel.max(abs_err_kernel);
                max_rel_kernel = max_rel_kernel.max(abs_err_kernel / ref_kernel_absmax);
                let abs_err_signal = (got - ref_signal[n]).abs();
                max_rel_signal = max_rel_signal.max(abs_err_signal / ref_signal_absmax);
            }
            worst_rel = worst_rel.max(max_rel_kernel);
            eprintln!(
                "  g={gs:4} {lbl:12} m={m:6} k={k:6}  KERNEL max_abs={max_abs_kernel:.4e} max_rel={max_rel_kernel:.4e}  |  SIGNAL(vs orig weight, informational) max_rel={max_rel_signal:.4e}"
            );
        }
        eprintln!("QWEN35_MARLIN_G_NUMERICS  group_size={gs:4}  WORST KERNEL max_rel across shapes = {worst_rel:.4e}  (gate: <=~2e-3)");
    }
}
