// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! DSv4 MoE feed-forward (sqrtsoftplus route + clamped-SwiGLU experts +
//! shared expert) on Metal vs CPU.
use ffai_core::{DType, Device, Tensor};
use ffai_metal::MetalDevice;
use ffai_models::dsv4::{dsv4_moe, Dsv4Expert, Dsv4Moe};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn fill(n: usize, s: usize) -> Vec<f32> { (0..n).map(|i| (((i*7+s*131)%89) as f32 - 44.0)*0.01).collect() }
fn tn(d: &dyn Device, v: &[f32], shape: Vec<usize>) -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), shape, DType::F32) }
fn mv(m:&[f32],v:&[f32],r:usize,k:usize)->Vec<f32>{(0..r).map(|i|(0..k).map(|c|m[i*k+c]*v[c]).sum()).collect()}
fn sl(g:&[f32],u:&[f32],lim:f32)->Vec<f32>{(0..g.len()).map(|i|{let gl=g[i].min(lim);let s=gl/(1.0+(-gl).exp());s*u[i].clamp(-lim,lim)}).collect()}

#[test]
fn dsv4_moe_on_metal_matches_cpu() {
    let Some(dev)=MetalDevice::create().expect("metal init") else { eprintln!("no Metal — skip"); return; };
    let (h, im, ne, tk) = (256usize, 512usize, 8usize, 2usize);
    let (rs, lim) = (1.5f32, 10.0f32);
    let router = fill(ne*h, 1);
    let bias: Vec<f32> = (0..ne).map(|i| (i as f32)*0.03 - 0.1).collect();
    let ex: Vec<(Vec<f32>,Vec<f32>,Vec<f32>)> = (0..ne).map(|e| (fill(im*h,10+e), fill(im*h,30+e), fill(h*im,50+e))).collect();
    let sh = (fill(im*h,200), fill(im*h,210), fill(h*im,220));
    let x = fill(h, 99);

    let mk = |g:&Vec<f32>,u:&Vec<f32>,d:&Vec<f32>| Dsv4Expert{gate:tn(dev.as_ref(),g,vec![im,h]),up:tn(dev.as_ref(),u,vec![im,h]),down:tn(dev.as_ref(),d,vec![h,im])};
    let w = Dsv4Moe {
        router: tn(dev.as_ref(),&router,vec![ne,h]), bias: bias.clone(),
        experts: ex.iter().map(|(g,u,d)| mk(g,u,d)).collect(),
        shared: mk(&sh.0,&sh.1,&sh.2), top_k: tk, routed_scaling: rs, swiglu_limit: lim,
    };
    let tx = tn(dev.as_ref(),&x,vec![h]);
    let out = dsv4_moe(dev.as_ref(), &w, &tx).unwrap();
    dev.synchronize().unwrap();
    let mut ob = vec![0u8; h*4]; dev.download(out.buffer.as_ref(), &mut ob).unwrap();
    let got = fb(&ob);

    // CPU ref
    let logits = mv(&router,&x,ne,h);
    let unb: Vec<f32> = logits.iter().map(|&v| (v.max(0.0)+(1.0+(-v.abs()).exp()).ln()).sqrt()).collect();
    let bia: Vec<f32> = unb.iter().zip(&bias).map(|(u,b)| u+b).collect();
    let mut ord: Vec<usize> = (0..ne).collect(); ord.sort_by(|&a,&b| bia[b].total_cmp(&bia[a]));
    let top: Vec<usize> = ord.into_iter().take(tk).collect();
    let den: f32 = top.iter().map(|&e| unb[e]).sum();
    let wts: Vec<f32> = top.iter().map(|&e| unb[e]/den*rs).collect();
    let mut acc = vec![0.0f32; h];
    for (&e,&gw) in top.iter().zip(&wts) {
        let (g,u,d)=&ex[e]; let inner=sl(&mv(g,&x,im,h),&mv(u,&x,im,h),lim); let o=mv(d,&inner,h,im);
        for i in 0..h { acc[i]+=gw*o[i]; }
    }
    let si=sl(&mv(&sh.0,&x,im,h),&mv(&sh.1,&x,im,h),lim); let so=mv(&sh.2,&si,h,im);
    for i in 0..h { acc[i]+=so[i]; }

    let mut e=0.0f32; for i in 0..h { e=e.max((got[i]-acc[i]).abs()); }
    eprintln!("DSv4 MoE on Metal vs CPU: max|Δ|={e:.3e} (top {top:?})");
    assert!(e <= 5e-3, "dsv4 moe mismatch: {e:.3e}");
    eprintln!("✅ DSv4 MoE feed-forward runs on Apple GPU through the shared op layer, matches CPU.");
}
