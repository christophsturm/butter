// Q8 GEMV throughput microbench — isolates kernel cost from decode orchestration.
use ffai_core::{DType, Tensor};
use ffai_ops::{gemv_q8, quantize_q8};
use ffai_cuda::CudaDevice;
use std::time::Instant;

fn tbf(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tbu(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

#[test]
fn gemv_q8_throughput() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let d = dev.as_ref();
    for (m, k, label) in [(10304usize, 2688usize, "in_proj"), (1856, 2688, "expert_up"), (2688, 1856, "expert_down"), (131072, 2688, "lm_head")] {
        let mut s = 0x9e37u32;
        let mut rng = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as f32 / u32::MAX as f32) - 0.5 };
        let w: Vec<f32> = (0..m * k).map(|_| rng()).collect();
        let x: Vec<f32> = (0..k).map(|_| rng()).collect();
        let (qs, sc) = quantize_q8(&w, m, k);
        let qt = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
        let st = Tensor::new(d.upload(&tbf(&sc)).unwrap(), vec![sc.len()], DType::F32);
        let xt = Tensor::new(d.upload(&tbf(&x)).unwrap(), vec![k], DType::F32);
        for _ in 0..5 { let _ = gemv_q8(d, &qt, &st, &xt, m, k, m).unwrap(); }
        d.synchronize().unwrap();
        let iters = 200;
        let t0 = Instant::now();
        for _ in 0..iters { let _ = gemv_q8(d, &qt, &st, &xt, m, k, m).unwrap(); }
        d.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let bytes = (m * k) as f64 + (m * k / 32 * 4) as f64; // qs 1B/param + scales
        eprintln!("gemv_q8 {label:12} [{m}x{k}]: {us:7.1} us/call  {:6.1} GB/s", bytes / us / 1e3);
    }
}
