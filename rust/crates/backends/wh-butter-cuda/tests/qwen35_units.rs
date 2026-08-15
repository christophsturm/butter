// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Cheap unit-isolation tests for the qwen35-port decode-correctness bug.
//! Each test builds a tiny/real-shape synthetic fixture, computes an
//! independent HOST reference (no shared code with the GPU kernels), and
//! diffs against the actual `wh_butter_ops` wrapper calls qwen35.rs's
//! `gdn_layer_full`/`full_attn_layer` use. Debug-plan step 1.
use wh_butter_cuda::CudaDevice;

fn up(d: &dyn wh_butter_core::Device, v: &[f32]) -> wh_butter_core::Tensor {
    use wh_butter_core::{DType, Tensor};
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    Tensor::new(d.upload(&b).unwrap(), vec![v.len()], DType::F32)
}
fn dl(d: &dyn wh_butter_core::Device, t: &wh_butter_core::Tensor, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    d.synchronize().unwrap();
    d.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Suspect 1: GDN composition (external rms_norm q/k-scale + gdn_gate_beta +
/// bare gated_delta_step) — exactly what `qwen35::gdn_layer_full` calls —
/// at the REAL Qwen3.6-27B shape (Hv=48, Hk=16, Dk=Dv=128), across 3
/// sequential decode steps carrying `state` forward. Host oracle mirrors
/// the GDN_PREFILL_CONTRACT.md §2/§4 formulas independently (no shared
/// code with the kernels under test).
#[test]
fn qwen35_gdn_decode_composition_matches_host_oracle() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    use wh_butter_core::{DType, Tensor};

    const HV: usize = 48;
    const HK: usize = 16;
    const DK: usize = 128;
    const DV: usize = 128;
    const EPS: f32 = 1e-6;
    let hk_per_hv = HV / HK;
    let inv_scale = 1.0f32 / (DK as f32).sqrt();

    // Constant per-element scale buffers, exactly as `Qwen35Model::load_gguf`
    // builds `gdn_q_scale`/`gdn_k_scale`.
    let q_scale = up(d, &vec![inv_scale * inv_scale; DK]);
    let k_scale = up(d, &vec![inv_scale; DK]);

    // Per-Hv-head learned params (constant across tokens), deterministic.
    let dt_bias: Vec<f32> = (0..HV).map(|i| -0.3 + (i as f32) * 0.01).collect();
    let a_log: Vec<f32> = (0..HV).map(|i| -2.0 + (i as f32) * 0.02).collect();
    let neg_exp_a_log: Vec<f32> = a_log.iter().map(|&a| -a.exp()).collect();
    let dt_bias_t = up(d, &dt_bias);
    let neg_exp_a_log_t = up(d, &neg_exp_a_log);

    // Persistent state, zero-initialized (matches Qwen35Cache::new's alloc_zeroed).
    let state_len = HV * DV * DK;
    let state = Tensor::new(d.alloc_zeroed(state_len * 4).unwrap(), vec![state_len], DType::F32);
    let mut host_state = vec![0.0f32; state_len];

    let n_tokens = 3;
    for t in 0..n_tokens {
        let seed = (t as f32) * 97.0;
        let q_raw: Vec<f32> = (0..HK * DK).map(|i| ((i as f32 + seed) * 0.0173).sin() * 0.5).collect();
        let k_raw: Vec<f32> = (0..HK * DK).map(|i| ((i as f32 + seed) * 0.0211).cos() * 0.5).collect();
        let v: Vec<f32> = (0..HV * DV).map(|i| ((i as f32 + seed) * 0.029).sin() * 0.3).collect();
        let a_raw: Vec<f32> = (0..HV).map(|i| ((i as f32 + seed) * 0.037).sin() * 0.4).collect();
        let b_raw: Vec<f32> = (0..HV).map(|i| ((i as f32 + seed) * 0.041).cos() * 0.4).collect();

        // ── GPU path: exactly qwen35::gdn_layer_full's sequence ──────────
        let q_raw_t = up(d, &q_raw).reshaped(vec![HK, DK]);
        let k_raw_t = up(d, &k_raw).reshaped(vec![HK, DK]);
        let v_t = up(d, &v).reshaped(vec![HV, DV]);
        let a_raw_t = up(d, &a_raw);
        let b_raw_t = up(d, &b_raw);

        let q_n = wh_butter_ops::rms_norm(d, &q_raw_t, &q_scale, EPS).unwrap();
        let k_n = wh_butter_ops::rms_norm(d, &k_raw_t, &k_scale, EPS).unwrap();
        let (g, beta) = wh_butter_ops::gdn_gate_beta(d, &a_raw_t, &b_raw_t, &dt_bias_t, &neg_exp_a_log_t, HV).unwrap();
        let y = wh_butter_ops::gated_delta_step(d, &q_n, &k_n, &v_t, &g, &beta, &state, DK, DV, HV, HK).unwrap();
        let y_gpu = dl(d, &y, HV * DV);

        // ── Host oracle: independent re-derivation from GDN_PREFILL_CONTRACT.md §2/§4 ──
        let mut q_normed = vec![0.0f32; HK * DK];
        let mut k_normed = vec![0.0f32; HK * DK];
        for h in 0..HK {
            let qrow = &q_raw[h * DK..(h + 1) * DK];
            let krow = &k_raw[h * DK..(h + 1) * DK];
            let qms = qrow.iter().map(|x| x * x).sum::<f32>() / DK as f32;
            let kms = krow.iter().map(|x| x * x).sum::<f32>() / DK as f32;
            let qr = 1.0 / (qms + EPS).sqrt();
            let kr = 1.0 / (kms + EPS).sqrt();
            for d_ in 0..DK {
                q_normed[h * DK + d_] = qrow[d_] * qr * (inv_scale * inv_scale);
                k_normed[h * DK + d_] = krow[d_] * kr * inv_scale;
            }
        }
        let mut g_h = vec![0.0f32; HV];
        let mut beta_h = vec![0.0f32; HV];
        for hv in 0..HV {
            let dt = ((a_raw[hv] + dt_bias[hv]).exp() + 1.0).ln();
            g_h[hv] = (neg_exp_a_log[hv] * dt).exp();
            beta_h[hv] = 1.0 / (1.0 + (-b_raw[hv]).exp());
        }
        let mut y_h = vec![0.0f32; HV * DV];
        for hv in 0..HV {
            let hk = hv / hk_per_hv;
            let gv = g_h[hv];
            let bv = beta_h[hv];
            for dv in 0..DV {
                let sbase = hv * DV * DK + dv * DK;
                let mut kv_mem = 0.0f32;
                let mut decayed = vec![0.0f32; DK];
                for dk_ in 0..DK {
                    let sd = host_state[sbase + dk_] * gv;
                    decayed[dk_] = sd;
                    kv_mem += sd * k_normed[hk * DK + dk_];
                }
                let delta = (v[hv * DV + dv] - kv_mem) * bv;
                let mut out = 0.0f32;
                for dk_ in 0..DK {
                    let sn = decayed[dk_] + k_normed[hk * DK + dk_] * delta;
                    host_state[sbase + dk_] = sn;
                    out += sn * q_normed[hk * DK + dk_];
                }
                y_h[hv * DV + dv] = out;
            }
        }

        let mut bad = 0;
        let mut max_abs_diff = 0.0f32;
        for i in 0..HV * DV {
            let diff = (y_gpu[i] - y_h[i]).abs();
            max_abs_diff = max_abs_diff.max(diff);
            if diff > 1e-3 * y_h[i].abs().max(1.0) {
                if bad < 8 {
                    eprintln!("t={t} MISMATCH i={i}: gpu={} host={}", y_gpu[i], y_h[i]);
                }
                bad += 1;
            }
        }
        eprintln!("t={t}: bad={bad}/{} max_abs_diff={max_abs_diff}", HV * DV);
        assert_eq!(bad, 0, "GDN composition mismatch at token {t}");
    }
    eprintln!("qwen35 GDN decode composition matches host oracle across {n_tokens} sequential tokens: OK");
}

/// Suspect 3: `strided_head_half` de-interleave on a ramp pattern —
/// values 0..N so any transposition/stride bug shows as an out-of-place
/// value rather than a coincidentally-correct one.
#[test]
fn qwen35_strided_head_half_ramp() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let n_head = 6usize;
    let head_dim = 8usize;
    // x[h, 0..head_dim) = q values, x[h, head_dim..2*head_dim) = gate values.
    let x: Vec<f32> = (0..n_head * 2 * head_dim).map(|i| i as f32).collect();
    let xt = up(d, &x).reshaped(vec![n_head, 2 * head_dim]);

    let q = wh_butter_ops::strided_head_half(d, &xt, n_head, head_dim, 0).unwrap();
    let gate = wh_butter_ops::strided_head_half(d, &xt, n_head, head_dim, 1).unwrap();
    let qv = dl(d, &q, n_head * head_dim);
    let gv = dl(d, &gate, n_head * head_dim);

    let mut bad = 0;
    for h in 0..n_head {
        for di in 0..head_dim {
            let expect_q = (h * 2 * head_dim + di) as f32;
            let expect_g = (h * 2 * head_dim + head_dim + di) as f32;
            if (qv[h * head_dim + di] - expect_q).abs() > 1e-6 {
                eprintln!("Q MISMATCH h={h} d={di}: got {} expect {expect_q}", qv[h * head_dim + di]);
                bad += 1;
            }
            if (gv[h * head_dim + di] - expect_g).abs() > 1e-6 {
                eprintln!("GATE MISMATCH h={h} d={di}: got {} expect {expect_g}", gv[h * head_dim + di]);
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "strided_head_half de-interleave wrong ({bad} bad elems)");
    eprintln!("strided_head_half ramp de-interleave: OK");
}

/// Suspect 2: `sdpa_decode` at the qwen35 full-attn shape (head_dim=256,
/// GQA 24 q-heads / 4 kv-heads, heads_per_group=6) with 3 cached KV
/// positions, vs a naive host softmax-attention reference. This shape
/// (head_dim=256) has never been checked against an independent host
/// reference anywhere in the tree (only cross-checked between kernel
/// variants at head_dim=128 in laguna.rs).
#[test]
fn qwen35_sdpa_decode_d256_matches_host_oracle() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let hd = 256usize;
    let n_kv_heads = 4usize;
    let n_head = 24usize;
    let heads_per_group = (n_head / n_kv_heads) as u32;
    let kv_stride = 8u32; // cache capacity
    let n_kv = 3u32; // 3 filled positions
    let scale = 1.0 / (hd as f32).sqrt();

    let ramp = |n: usize, step: f32, start: f32| -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    };
    let q: Vec<f32> = ramp(n_head * hd, 0.0113, -0.4);
    let k: Vec<f32> = ramp(n_kv_heads * kv_stride as usize * hd, 0.0091, -0.5);
    let v: Vec<f32> = ramp(n_kv_heads * kv_stride as usize * hd, 0.0067, -0.3);

    let tq = up(d, &q).reshaped(vec![n_head, hd]);
    let tk = up(d, &k).reshaped(vec![n_kv_heads, kv_stride as usize, hd]);
    let tv = up(d, &v).reshaped(vec![n_kv_heads, kv_stride as usize, hd]);

    let got = wh_butter_ops::sdpa_decode(d, &tq, &tk, &tv, hd, n_kv, kv_stride, heads_per_group, scale).unwrap();
    let gv = dl(d, &got, n_head * hd);

    // Naive host softmax attention: kv_head = q_head / heads_per_group,
    // dense causal over the first n_kv positions (all "past", decode-time).
    let mut hv_out = vec![0.0f32; n_head * hd];
    for qh in 0..n_head {
        let kvh = qh / heads_per_group as usize;
        let mut scores = vec![0.0f32; n_kv as usize];
        for t in 0..n_kv as usize {
            let mut dot = 0.0f32;
            for i in 0..hd {
                dot += q[qh * hd + i] * k[(kvh * kv_stride as usize + t) * hd + i];
            }
            scores[t] = dot * scale;
        }
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        for i in 0..hd {
            let mut acc = 0.0f32;
            for t in 0..n_kv as usize {
                acc += (exps[t] / sum_e) * v[(kvh * kv_stride as usize + t) * hd + i];
            }
            hv_out[qh * hd + i] = acc;
        }
    }

    let mut bad = 0;
    let mut max_abs_diff = 0.0f32;
    for i in 0..n_head * hd {
        let diff = (gv[i] - hv_out[i]).abs();
        max_abs_diff = max_abs_diff.max(diff);
        if diff > 1e-3 * hv_out[i].abs().max(1.0) {
            if bad < 8 {
                eprintln!("MISMATCH i={i}: gpu={} host={}", gv[i], hv_out[i]);
            }
            bad += 1;
        }
    }
    eprintln!("sdpa_decode_d256: bad={bad}/{} max_abs_diff={max_abs_diff}", n_head * hd);
    assert_eq!(bad, 0, "sdpa_decode head_dim=256 mismatch vs host oracle");
    eprintln!("sdpa_decode d256 GQA 24/4 matches host oracle: OK");
}

/// Suspect 4: `rope_yarn_partial` at qwen35's exact config (n_rot=64 of
/// head_dim=256, theta=1e7, YaRN disabled) at NONZERO positions (0, 1, 50).
/// The only existing coverage (laguna.rs) checks pos=0 only, which is a
/// trivial identity*mscale case that can't catch a wrong rotation
/// direction/frequency/pairing bug. Host reference is the standard NEOX
/// split-half RoPE formula, independently written.
#[test]
fn qwen35_rope_partial_nonzero_pos_matches_host_oracle() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    use wh_butter_core::{DType, Tensor};

    let n_heads = 4usize;
    let head_dim = 256usize;
    let n_rot = 64u32;
    let theta_base = 1.0e7f32;
    let half_rot = (n_rot / 2) as usize;

    for &pos in &[0u32, 1u32, 50u32] {
        let x: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| ((i as f32 * 0.0037 + pos as f32 * 0.01).sin()) * 0.6)
            .collect();
        let xt = Tensor::new(d.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap(), vec![n_heads, head_dim], DType::F32);

        let out = wh_butter_ops::rope_yarn_partial(d, &xt, pos, n_rot, theta_base, 1.0, 0.0, 1.0, 32.0, 1.0, 262144).unwrap();
        let ov = dl(d, &out, n_heads * head_dim);

        // Host NEOX split-half reference, independently derived:
        // theta_scale = theta_base^(-2/n_rot); theta_j = pos * theta_scale^j
        // for j in [0, half_rot); pair (x[j], x[j+half_rot]) rotated by theta_j;
        // dims >= n_rot pass through unrotated.
        let mut expect = x.clone();
        let theta_scale = theta_base.powf(-2.0 / n_rot as f32);
        for h in 0..n_heads {
            let base = h * head_dim;
            for j in 0..half_rot {
                let theta = pos as f32 * theta_scale.powi(j as i32);
                let (s, c) = theta.sin_cos();
                let x0 = x[base + j];
                let x1 = x[base + j + half_rot];
                expect[base + j] = x0 * c - x1 * s;
                expect[base + j + half_rot] = x0 * s + x1 * c;
            }
            // dims [n_rot, head_dim) untouched — already copied via x.clone().
        }

        let mut bad = 0;
        let mut max_abs_diff = 0.0f32;
        for i in 0..n_heads * head_dim {
            let diff = (ov[i] - expect[i]).abs();
            max_abs_diff = max_abs_diff.max(diff);
            if diff > 1e-3 * expect[i].abs().max(1.0) {
                if bad < 8 {
                    eprintln!("pos={pos} MISMATCH i={i}: gpu={} host={}", ov[i], expect[i]);
                }
                bad += 1;
            }
        }
        eprintln!("pos={pos}: bad={bad}/{} max_abs_diff={max_abs_diff}", n_heads * head_dim);
        assert_eq!(bad, 0, "rope_yarn_partial mismatch at pos={pos}");
    }
    eprintln!("rope_yarn_partial nonzero-pos matches host oracle: OK");
}
