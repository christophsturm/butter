// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full DSv4 attention SUBBLOCK (mHC ⊗ MLA): mHC mix → sinkhorn split →
//! collapse → MLA → expand → new 4-channel state. On Metal vs CPU. This is
//! the complete DSv4 attention layer unit. pos=0 → RoPE identity (validates
//! the layer wiring; RoPE verified separately).
use ffai_core::{DType, Device, Tensor};
use ffai_metal::MetalDevice;
use ffai_models::dsv4::{dsv4_attn_subblock, MhcWeights, MlaConfig, MlaWeights};
use ffai_ops::dsv4_mhc_sinkhorn_split;

fn tb(v:&[f32])->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn fb(b:&[u8])->Vec<f32>{b.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect()}
fn fill(n:usize,s:usize)->Vec<f32>{(0..n).map(|i|(((i*7+s*131)%89) as f32-44.0)*0.008).collect()}
fn tn(d:&dyn Device,v:&[f32],sh:Vec<usize>)->Tensor{Tensor::new(d.upload(&tb(v)).unwrap(),sh,DType::F32)}
fn rms(x:&[f32],w:&[f32],eps:f32)->Vec<f32>{let n=x.len();let ms:f32=x.iter().map(|v|v*v).sum::<f32>()/n as f32;let s=1.0/(ms+eps).sqrt();(0..n).map(|i|x[i]*s*w[i]).collect()}
fn mv(m:&[f32],v:&[f32],r:usize,k:usize)->Vec<f32>{(0..r).map(|i|(0..k).map(|c|m[i*k+c]*v[c]).sum()).collect()}

#[allow(clippy::too_many_arguments)]
fn cpu_mla(x:&[f32], an:&[f32], qa:&[f32], qan:&[f32], qb:&[f32], kv:&[f32], kvan:&[f32], sink:&[f32], oa:&[Vec<f32>], ob:&[f32], cfg:&MlaConfig)->Vec<f32>{
    let (h,hd,ql,qd,ol,og)=(cfg.hidden,cfg.head_dim,cfg.q_lora_rank,cfg.n_heads*cfg.head_dim,cfg.o_lora_rank,cfg.o_groups);
    let gsize=qd/og; let xn=rms(x,an,cfg.eps);
    let q0=mv(qb,&rms(&mv(qa,&xn,ql,h),qan,cfg.eps),qd,ql);
    let mut q=q0.clone();
    for hh in 0..cfg.n_heads{let row=&q0[hh*hd..(hh+1)*hd];let ms:f32=row.iter().map(|v|v*v).sum::<f32>()/hd as f32;let s=1.0/(ms+cfg.eps).sqrt();for d in 0..hd{q[hh*hd+d]=row[d]*s;}}
    let kvn=rms(&mv(kv,&xn,hd,h),kvan,cfg.eps); let scale=1.0/(hd as f32).sqrt();
    let mut attn=vec![0.0f32;qd];
    for hh in 0..cfg.n_heads{let sc=scale*(0..hd).map(|d|q[hh*hd+d]*kvn[d]).sum::<f32>();let m=sc.max(sink[hh]);let p=(sc-m).exp()/((sc-m).exp()+(sink[hh]-m).exp());for d in 0..hd{attn[hh*hd+d]=p*kvn[d];}}
    let mut olow=vec![0.0f32;og*ol];
    for g in 0..og{let s=&attn[g*gsize..(g+1)*gsize];let r=mv(&oa[g],s,ol,gsize);olow[g*ol..(g+1)*ol].copy_from_slice(&r);}
    mv(ob,&olow,h,og*ol)
}

#[test]
fn dsv4_attn_subblock_on_metal_matches_cpu(){
    let Some(dev)=MetalDevice::create().expect("metal init") else { eprintln!("no Metal — skip"); return; };
    let cfg=MlaConfig{hidden:512,n_heads:2,head_dim:512,q_lora_rank:256,n_nope:448,half_rot:32,o_lora_rank:64,o_groups:8,rope_theta:10000.0,eps:1e-6};
    let (h,hd,ql,qd,ol,og)=(cfg.hidden,cfg.head_dim,cfg.q_lora_rank,cfg.n_heads*cfg.head_dim,cfg.o_lora_rank,cfg.o_groups);
    let nhc=4; let gsize=qd/og; let eps=1e-6f32; let iters=20u32;
    // weights
    let an=fill(h,1);let qa=fill(ql*h,2);let qan=fill(ql,3);let qb=fill(qd*ql,4);let kv=fill(hd*h,5);let kvan=fill(hd,6);let sink=vec![0.4f32,-0.2];
    let oa:Vec<Vec<f32>>=(0..og).map(|g|fill(ol*gsize,20+g)).collect();let ob=fill(h*(og*ol),40);
    let hc_fn=fill(24*nhc*h,7);let hc_scale=[0.5f32,0.7,0.9];let hc_base=fill(24,8);
    let hc_state=fill(nhc*h,99);

    let mla=MlaWeights{attn_norm:tn(dev.as_ref(),&an,vec![h]),q_a:tn(dev.as_ref(),&qa,vec![ql,h]),q_a_norm:tn(dev.as_ref(),&qan,vec![ql]),q_b:tn(dev.as_ref(),&qb,vec![qd,ql]),kv:tn(dev.as_ref(),&kv,vec![hd,h]),kv_a_norm:tn(dev.as_ref(),&kvan,vec![hd]),sink:tn(dev.as_ref(),&sink,vec![2]),output_a:oa.iter().map(|g|tn(dev.as_ref(),g,vec![ol,gsize])).collect(),output_b:tn(dev.as_ref(),&ob,vec![h,og*ol])};
    let mhc=MhcWeights{hc_fn:tn(dev.as_ref(),&hc_fn,vec![24,nhc*h]),hc_scale,hc_base:hc_base.clone()};
    let ts=tn(dev.as_ref(),&hc_state,vec![nhc,h]);

    let out=dsv4_attn_subblock(dev.as_ref(),&cfg,&mhc,&mla,&ts,0,eps,iters).unwrap();
    dev.synchronize().unwrap();
    let mut ob_b=vec![0u8;nhc*h*4];dev.download(out.buffer.as_ref(),&mut ob_b).unwrap();
    let got=fb(&ob_b);

    // CPU ref
    let mixes=mv(&hc_fn,&hc_state,24,nhc*h);
    let (pre,post,comb)=dsv4_mhc_sinkhorn_split(&mixes,hc_scale,&hc_base,eps,iters);
    let x:Vec<f32>=(0..h).map(|d|(0..nhc).map(|c|pre[c]*hc_state[c*h+d]).sum()).collect();
    let blk=cpu_mla(&x,&an,&qa,&qan,&qb,&kv,&kvan,&sink,&oa,&ob,&cfg);
    let mut want=vec![0.0f32;nhc*h];
    for dst in 0..nhc{for d in 0..h{let mut a=blk[d]*post[dst];for src in 0..nhc{a+=comb[dst*nhc+src]*hc_state[src*h+d];}want[dst*h+d]=a;}}

    let mut e=0.0f32;for i in 0..nhc*h{e=e.max((got[i]-want[i]).abs());}
    eprintln!("DSv4 attn subblock (mHC⊗MLA) on Metal vs CPU: max|Δ|={e:.3e}");
    assert!(e<=5e-3,"subblock mismatch: {e:.3e}");
    eprintln!("✅ Full DSv4 attention layer unit runs on Apple GPU through the shared op layer, matches CPU.");
}
