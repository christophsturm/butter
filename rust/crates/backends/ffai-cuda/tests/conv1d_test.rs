#![cfg(feature = "cuda")]
// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Causal conv1d step (conv1d_causal_step) on CUDA vs CPU — the short-conv
//! used by Mamba2 + audio front-ends.
use ffai_core::{DType, Device, Tensor};
use ffai_cuda::CudaDevice;
use ffai_ops::conv1d_causal_step;

fn tb(v:&[f32])->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn fb(b:&[u8])->Vec<f32>{b.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect()}
fn tn(d:&dyn Device,v:&[f32],sh:Vec<usize>)->Tensor{Tensor::new(d.upload(&tb(v)).unwrap(),sh,DType::F32)}

#[test]
fn conv1d_causal_step_on_cuda_matches_cpu(){
    let Some(dev)=CudaDevice::create().expect("metal") else { eprintln!("no CUDA — skip"); return; };
    let (nc, ks) = (128usize, 4usize);
    let x: Vec<f32> = (0..nc).map(|i| ((i as f32)*0.013).sin()).collect();
    let w: Vec<f32> = (0..ks*nc).map(|i| 0.1+((i as f32)*0.019).cos()*0.2).collect();
    let b: Vec<f32> = (0..nc).map(|i| (i as f32)*0.001-0.05).collect();
    let st: Vec<f32> = (0..(ks-1)*nc).map(|i| ((i as f32)*0.007).sin()*0.5).collect();

    let st_t = tn(dev.as_ref(),&st,vec![(ks-1)*nc]);
    let y = conv1d_causal_step(dev.as_ref(), &tn(dev.as_ref(),&x,vec![nc]), &tn(dev.as_ref(),&w,vec![ks*nc]), &tn(dev.as_ref(),&b,vec![nc]), &st_t, nc as u32, ks as u32).unwrap();
    dev.synchronize().unwrap();
    let mut yb=vec![0u8;nc*4]; dev.download(y.buffer.as_ref(),&mut yb).unwrap(); let yv=fb(&yb);
    let mut sb=vec![0u8;(ks-1)*nc*4]; dev.download(st_t.buffer.as_ref(),&mut sb).unwrap(); let so=fb(&sb);

    // CPU ref
    let mut y_ref=vec![0.0f32;nc]; let mut s_ref=vec![0.0f32;(ks-1)*nc];
    for d in 0..nc {
        let mut acc=b[d]+w[(ks-1)*nc+d]*x[d];
        for k in 0..ks-1 { acc += w[k*nc+d]*st[k*nc+d]; }
        y_ref[d]=acc;
        for k in 0..ks.saturating_sub(2) { s_ref[k*nc+d]=st[(k+1)*nc+d]; }
        s_ref[(ks-2)*nc+d]=x[d];
    }
    let mut ey=0.0f32; for i in 0..nc { ey=ey.max((yv[i]-y_ref[i]).abs()); }
    let mut es=0.0f32; for i in 0..(ks-1)*nc { es=es.max((so[i]-s_ref[i]).abs()); }
    eprintln!("conv1d_causal_step on CUDA vs CPU: y max|Δ|={ey:.3e}  state max|Δ|={es:.3e}");
    assert!(ey<=1e-4 && es<=1e-4, "conv1d mismatch y={ey:.3e} state={es:.3e}");
    eprintln!("✅ Causal conv1d step runs on CUDA through the shared op layer, matches CPU.");
}
