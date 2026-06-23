#![cfg(feature = "cuda")]
// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Mamba2 SSD selective-scan decode step (mt_ssm_step) on CUDA vs CPU —
//! the core SSM-family op (Mamba2/Jamba/FalconH1/LFM2).
use ffai_core::{DType, Device, Tensor};
use ffai_cuda::CudaDevice;
use ffai_ops::ssm_step;

fn tb(v:&[f32])->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn fb(b:&[u8])->Vec<f32>{b.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect()}
fn fill(n:usize,s:usize)->Vec<f32>{(0..n).map(|i|(((i*7+s*131)%89) as f32-44.0)*0.02).collect()}
fn tn(d:&dyn Device,v:&[f32],sh:Vec<usize>)->Tensor{Tensor::new(d.upload(&tb(v)).unwrap(),sh,DType::F32)}

#[test]
fn mt_ssm_step_on_cuda_matches_cpu(){
    let Some(dev)=CudaDevice::create().expect("metal") else { eprintln!("no CUDA — skip"); return; };
    let (nh, dh, ds, hpg) = (4usize, 8usize, 32usize, 2usize);
    let ng = nh/hpg;
    let x=fill(nh*dh,1); let a_log=fill(nh,2); let b=fill(ng*ds,3); let c=fill(ng*ds,4); let dsk=fill(nh,5); let dt:Vec<f32>=(0..nh).map(|i|0.3+0.1*i as f32).collect(); let si=fill(nh*dh*ds,7);

    let (so_t, out_t) = ssm_step(dev.as_ref(),
        &tn(dev.as_ref(),&x,vec![nh*dh]), &tn(dev.as_ref(),&a_log,vec![nh]),
        &tn(dev.as_ref(),&b,vec![ng*ds]), &tn(dev.as_ref(),&c,vec![ng*ds]),
        &tn(dev.as_ref(),&dsk,vec![nh]), &tn(dev.as_ref(),&dt,vec![nh]),
        &tn(dev.as_ref(),&si,vec![nh*dh*ds]), dh as u32, ds as u32, nh as u32, hpg as u32).unwrap();
    dev.synchronize().unwrap();
    let mut sb=vec![0u8;nh*dh*ds*4]; dev.download(so_t.buffer.as_ref(),&mut sb).unwrap(); let so=fb(&sb);
    let mut ob=vec![0u8;nh*dh*4]; dev.download(out_t.buffer.as_ref(),&mut ob).unwrap(); let out=fb(&ob);

    // CPU ref
    let mut so_ref=vec![0.0f32;nh*dh*ds]; let mut out_ref=vec![0.0f32;nh*dh];
    for n in 0..nh {
        let g=n/hpg; let da=(-(a_log[n].exp())*dt[n]).exp();
        for d in 0..dh {
            let xv=x[n*dh+d]; let mut acc=0.0f32;
            for s in 0..ds {
                let ns=da*si[n*dh*ds+d*ds+s]+xv*dt[n]*b[g*ds+s];
                so_ref[n*dh*ds+d*ds+s]=ns; acc+=ns*c[g*ds+s];
            }
            out_ref[n*dh+d]=acc+xv*dsk[n];
        }
    }
    let mut es=0.0f32; for i in 0..nh*dh*ds { es=es.max((so[i]-so_ref[i]).abs()); }
    let mut eo=0.0f32; for i in 0..nh*dh { eo=eo.max((out[i]-out_ref[i]).abs()); }
    eprintln!("mt_ssm_step on CUDA vs CPU: state max|Δ|={es:.3e}  out max|Δ|={eo:.3e}");
    assert!(es<=1e-4 && eo<=1e-4, "ssm mismatch state={es:.3e} out={eo:.3e}");
    eprintln!("✅ Mamba2 SSD selective-scan step runs on CUDA through the shared op layer, matches CPU.");
}
