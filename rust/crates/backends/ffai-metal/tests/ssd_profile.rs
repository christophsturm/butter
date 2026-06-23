// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! SSD-scan PROFILE + L-sweep on Metal. Times `ssm_prefill_scan_ssd_portable`
//! (the portable/Metal chunked-matmul Mamba2 scan) at S∈{2048,8192} across
//! L∈{64,128,256,512}, plus the sequential reference, to find the optimal
//! chunk length and quantify the O(S·L) intra-chunk vs O(nc) inter-chunk
//! tradeoff. Run with:  cargo test --release -p ffai-metal --test ssd_profile -- --nocapture --test-threads=1
use ffai_core::{DType, Device, Tensor};
use ffai_metal::MetalDevice;
use ffai_ops::{ssm_prefill_scan, ssm_prefill_scan_ssd_portable};
use std::time::Instant;

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn fill(n: usize, s: usize) -> Vec<f32> { (0..n).map(|i| (((i * 7 + s * 131) % 89) as f32 - 44.0) * 0.02).collect() }

fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn ssd_lsweep_profile_metal() {
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    let d = dev.as_ref();
    let (dh, ds, h, ng) = (64usize, 128usize, 64usize, 8usize);
    let n_layers = 23usize; // Nemotron-Cascade has 23 Mamba layers
    let iters = 5usize;

    let s_list = [2048usize, 8192usize];
    let l_list = [64u32, 128u32, 256u32, 512u32];

    for &t in &s_list {
        // Synthetic well-conditioned inputs (small dt → stable decay).
        let x = fill(t * h * dh, 11);
        let a_log: Vec<f32> = (0..h).map(|i| -0.5 - 0.01 * (i % 7) as f32).collect();
        let b = fill(t * ng * ds, 13);
        let c = fill(t * ng * ds, 17);
        let dsk = fill(h, 19);
        let dt_v: Vec<f32> = (0..t * h).map(|i| 0.02 + 0.001 * ((i % 11) as f32)).collect();
        let si = vec![0.0f32; h * dh * ds];
        let mk = |v: &[f32], sh: Vec<usize>| Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32);
        let xt = mk(&x, vec![t * h * dh]);
        let at = mk(&a_log, vec![h]);
        let bt = mk(&b, vec![t * ng * ds]);
        let ct = mk(&c, vec![t * ng * ds]);
        let dskt = mk(&dsk, vec![h]);
        let dtt = mk(&dt_v, vec![t * h]);

        eprintln!("\n========== S={t}  (one Mamba layer; ×{n_layers} layers for e2e estimate) ==========");

        // Sequential reference (the current default scan).
        let (so_seq, y_seq) = {
            let sit = mk(&si, vec![h * dh * ds]);
            ssm_prefill_scan(d, &xt, &at, &bt, &ct, &dskt, &dtt, &sit,
                t as u32, dh as u32, ds as u32, h as u32, ng as u32).unwrap()
        };
        d.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            let sit = mk(&si, vec![h * dh * ds]);
            let _ = ssm_prefill_scan(d, &xt, &at, &bt, &ct, &dskt, &dtt, &sit,
                t as u32, dh as u32, ds as u32, h as u32, ng as u32).unwrap();
            d.synchronize().unwrap();
        }
        let seq_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        eprintln!("  sequential scan : {:>9.1} us/layer   {:>8.2} ms × {n_layers} layers", seq_us, seq_us * n_layers as f64 / 1e3);
        let ys = fb(&{ let mut bb = vec![0u8; t * h * dh * 4]; d.download(y_seq.buffer.as_ref(), &mut bb).unwrap(); bb });
        let ss = fb(&{ let mut bb = vec![0u8; h * dh * ds * 4]; d.download(so_seq.buffer.as_ref(), &mut bb).unwrap(); bb });

        for &l in &l_list {
            let nc = (t as u32).div_ceil(l);
            // warmup + correctness
            let (so_p, y_p) = {
                let sit = mk(&si, vec![h * dh * ds]);
                ssm_prefill_scan_ssd_portable(d, &xt, &at, &bt, &ct, &dskt, &dtt, &sit,
                    t as u32, dh as u32, ds as u32, h as u32, ng as u32, l).unwrap()
            };
            d.synchronize().unwrap();
            let yp = fb(&{ let mut bb = vec![0u8; t * h * dh * 4]; d.download(y_p.buffer.as_ref(), &mut bb).unwrap(); bb });
            let sp = fb(&{ let mut bb = vec![0u8; h * dh * ds * 4]; d.download(so_p.buffer.as_ref(), &mut bb).unwrap(); bb });
            let y_cos = cos_sim(&ys, &yp);
            let s_cos = cos_sim(&ss, &sp);

            let t1 = Instant::now();
            for _ in 0..iters {
                let sit = mk(&si, vec![h * dh * ds]);
                let _ = ssm_prefill_scan_ssd_portable(d, &xt, &at, &bt, &ct, &dskt, &dtt, &sit,
                    t as u32, dh as u32, ds as u32, h as u32, ng as u32, l).unwrap();
                d.synchronize().unwrap();
            }
            let us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let spd = seq_us / us;
            eprintln!("  SSD L={l:<4} nc={nc:<4}: {:>9.1} us/layer   {:>8.2} ms ×{n_layers}   ({spd:.2}× vs seq)   y_cos={y_cos:.6} s_cos={s_cos:.6}", us, us * n_layers as f64 / 1e3);
        }
    }
}
