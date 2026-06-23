// MoE gather bandwidth: contiguous vs scattered expert indices (isolate scatter cost).
use ffai_core::{DType, Tensor};
use ffai_ops::{moe_gather_up_relu2, quantize_q4};
use ffai_cuda::CudaDevice;
use std::time::Instant;
fn tbf(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tbu(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
#[test]
fn gather_bw() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { return; };
    let d = dev.as_ref();
    let (n_exp, inter, hid, top_k) = (128usize, 1856usize, 2688usize, 6usize);
    let mut s=0x33u32; let mut rng=||{s^=s<<13;s^=s>>17;s^=s<<5;(s as f32/u32::MAX as f32)-0.5};
    let w: Vec<f32> = (0..n_exp*inter*hid).map(|_| rng()).collect();
    let (qs,sc)=quantize_q4(&w, n_exp*inter, hid);
    let qt=Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
    let st=Tensor::new(d.upload(&tbf(&sc)).unwrap(), vec![sc.len()], DType::F32);
    let x: Vec<f32>=(0..hid).map(|_| rng()).collect();
    let xt=Tensor::new(d.upload(&tbf(&x)).unwrap(), vec![hid], DType::F32);
    for (lbl, idx) in [("contiguous", vec![0u32,1,2,3,4,5]), ("scattered", vec![0u32,21,42,63,84,105])] {
        let it=Tensor::new(d.upload(&tbu(&idx)).unwrap(), vec![top_k], DType::U32);
        for _ in 0..5 { let _=moe_gather_up_relu2(d,&qt,&st,&xt,&it,top_k,inter,hid).unwrap(); }
        d.synchronize().unwrap();
        let n=100; let t=Instant::now();
        for _ in 0..n { let _=moe_gather_up_relu2(d,&qt,&st,&xt,&it,top_k,inter,hid).unwrap(); }
        d.synchronize().unwrap();
        let us=t.elapsed().as_secs_f64()*1e6/n as f64;
        let bytes=(top_k*inter*hid/2) as f64 + (top_k*inter*hid/32*4) as f64;
        eprintln!("gather {lbl}: {us:.1} us  {:.1} GB/s", bytes/us/1e3);
    }
}
