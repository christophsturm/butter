// Q4 gemv profiling target — one big cold GEMV (lm_head shape) for ncu.
use ffai_core::{DType, Tensor};
use ffai_ops::{gemv_q4, quantize_q4};
use ffai_cuda::CudaDevice;
use std::time::Instant;
fn tbf(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tbu(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
#[test]
fn q4_lmhead_prof() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { return; };
    let d = dev.as_ref();
    for (m,k,lbl) in [(131072usize,2688usize,"lmhead"),(1856,2688,"exp_up"),(2688,1856,"exp_down"),(10304,2688,"in_proj")] { run(d,m,k,lbl); }
}
fn run(d: &dyn ffai_core::Device, m: usize, k: usize, lbl: &str) {
    let mut s = 0x1u32; let mut rng = || { s ^= s<<13; s^=s>>17; s^=s<<5; (s as f32/u32::MAX as f32)-0.5 };
    let w: Vec<f32> = (0..m*k).map(|_| rng()).collect();
    let x: Vec<f32> = (0..k).map(|_| rng()).collect();
    let (qs, sc) = quantize_q4(&w, m, k);
    let qt = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
    let st = Tensor::new(d.upload(&tbf(&sc)).unwrap(), vec![sc.len()], DType::F32);
    let xt = Tensor::new(d.upload(&tbf(&x)).unwrap(), vec![k], DType::F32);
    for _ in 0..3 { let _ = gemv_q4(d, &qt, &st, &xt, m, k, m).unwrap(); }
    d.synchronize().unwrap();
    let it = 50; let t = Instant::now();
    for _ in 0..it { let _ = gemv_q4(d, &qt, &st, &xt, m, k, m).unwrap(); }
    d.synchronize().unwrap();
    let us = t.elapsed().as_secs_f64()*1e6/it as f64;
    let bytes = (m*k/2) as f64 + (m*k/32*4) as f64;
    eprintln!("q4 {lbl}: {us:.1} us  {:.1} GB/s", bytes/us/1e3);
}
