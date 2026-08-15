// Round-2 lever #3: per-shape iron_gemv_q4_coalesced GB/s at qwen3.6-27B's
// actual decode matvec shapes (hidden=5120, ffn_inter=17408, GDN key_dim=
// 2048/value_dim=6144, full-attn q/k/v/o, dense FFN gate/up/down, Q4
// lm_head). Modeled on gemv_q4_prof.rs's pattern (random weights + x,
// warmup 3, timed 50, report GB/s from actual bytes moved: 0.5 B/param Q4
// nibbles + 0.125 B/param f16 block-32 scale = 0.625 B/param).
use wh_butter_core::{DType, Tensor};
use wh_butter_ops::{gemv_q4, quantize_q4};
use wh_butter_cuda::CudaDevice;
use std::time::Instant;
fn tbf(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tbu(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

#[test]
fn qwen35_gemv_shape_census() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let shapes: &[(usize, usize, &str)] = &[
        (12288, 5120, "full_wq(gate-fused)"),
        (1024,  5120, "full_wk_or_wv"),
        (5120,  6144, "full_wo"),
        (10240, 5120, "gdn_wqkv"),
        (6144,  5120, "gdn_wz"),
        (5120,  6144, "gdn_wo"),
        (17408, 5120, "ffn_gate_or_up"),
        (5120, 17408, "ffn_down"),
        (248320, 5120, "lm_head_q4"),
    ];
    for &(m, k, lbl) in shapes {
        run(d, m, k, lbl);
    }
}

fn run(d: &dyn wh_butter_core::Device, m: usize, k: usize, lbl: &str) {
    let mut s = 0x1u32;
    let mut rng = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as f32 / u32::MAX as f32) - 0.5 };
    let w: Vec<f32> = (0..m * k).map(|_| rng()).collect();
    let x: Vec<f32> = (0..k).map(|_| rng()).collect();
    let (qs, sc) = quantize_q4(&w, m, k);
    let qt = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
    let st = Tensor::new(d.upload(&tbf(&sc)).unwrap(), vec![sc.len()], DType::F32);
    let xt = Tensor::new(d.upload(&tbf(&x)).unwrap(), vec![k], DType::F32);
    for _ in 0..5 { let _ = gemv_q4(d, &qt, &st, &xt, m, k, m).unwrap(); }
    d.synchronize().unwrap();
    let it = 100;
    let t = Instant::now();
    for _ in 0..it { let _ = gemv_q4(d, &qt, &st, &xt, m, k, m).unwrap(); }
    d.synchronize().unwrap();
    let us = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    // Kernel reads scale as f16 (2 B/block of 32) regardless of what we upload
    // host-side (this test uploads raw f32 host bytes like gemv_q4_prof.rs,
    // but the kernel only ever *reads* the first k/32*2 bytes of that buffer
    // as f16 — see iron_gemv_q4_coalesced's `d_f32: Tensor<f16>` param doc).
    // So actual DRAM traffic is q4 nibbles (0.5 B/param) + f16 scale (2B per
    // 32-elem block = 0.0625 B/param) = 0.5625 B/param, NOT 0.625.
    let bytes = (m * k / 2) as f64 + (m * k / 32 * 2) as f64;
    eprintln!("q4 {lbl:24} m={m:7} k={k:6}: {us:8.1} us  {:7.1} GB/s  ({:.3} MB weights)", bytes / us / 1e3, bytes / 1e6);
}
