// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full DSv4 MLA attention composite on Metal vs CPU. pos=0 → RoPE is
//! identity, so this validates the COMPOSITION wiring (q low-rank →
//! per-head q-norm → sink-SDPA → grouped O-LoRA); RoPE itself is verified
//! separately in dsv4_test.
use ffai_core::{DType, Device, Tensor};
use ffai_metal::MetalDevice;
use ffai_models::dsv4::{mla_attention, MlaConfig, MlaWeights};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn fill(n: usize, s: usize) -> Vec<f32> { (0..n).map(|i| (((i * 7 + s * 131) % 89) as f32 - 44.0) * 0.01).collect() }
fn tn(d: &dyn Device, v: &[f32], shape: Vec<usize>) -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), shape, DType::F32) }
fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len(); let ms: f32 = x.iter().map(|v| v*v).sum::<f32>()/n as f32; let s = 1.0/(ms+eps).sqrt();
    (0..n).map(|i| x[i]*s*w[i]).collect()
}
fn mv(m: &[f32], v: &[f32], rows: usize, k: usize) -> Vec<f32> { (0..rows).map(|r| (0..k).map(|c| m[r*k+c]*v[c]).sum()).collect() }

#[test]
fn dsv4_mla_attention_on_metal_matches_cpu() {
    let Some(dev) = MetalDevice::create().expect("metal init") else { eprintln!("no Metal — skip"); return; };
    let cfg = MlaConfig { hidden:512, n_heads:2, head_dim:512, q_lora_rank:256, n_nope:448, half_rot:32, o_lora_rank:64, o_groups:8, rope_theta:10000.0, eps:1e-6 };
    let (h, hd, ql, qd, ol, og) = (cfg.hidden, cfg.head_dim, cfg.q_lora_rank, cfg.n_heads*cfg.head_dim, cfg.o_lora_rank, cfg.o_groups);
    let gsize = qd / og;

    let attn_norm = fill(h,1); let q_a = fill(ql*h,2); let q_a_norm = fill(ql,3); let q_b = fill(qd*ql,4);
    let kv = fill(hd*h,5); let kv_a_norm = fill(hd,6); let sink = vec![0.4f32,-0.2];
    let output_a: Vec<Vec<f32>> = (0..og).map(|g| fill(ol*gsize, 20+g)).collect();
    let output_b = fill(h*(og*ol), 40);
    let x = fill(h, 99);

    let w = MlaWeights {
        attn_norm: tn(dev.as_ref(),&attn_norm,vec![h]),
        q_a: tn(dev.as_ref(),&q_a,vec![ql,h]), q_a_norm: tn(dev.as_ref(),&q_a_norm,vec![ql]),
        q_b: tn(dev.as_ref(),&q_b,vec![qd,ql]),
        kv: tn(dev.as_ref(),&kv,vec![hd,h]), kv_a_norm: tn(dev.as_ref(),&kv_a_norm,vec![hd]),
        sink: tn(dev.as_ref(),&sink,vec![2]),
        output_a: output_a.iter().map(|g| tn(dev.as_ref(),g,vec![ol,gsize])).collect(),
        output_b: tn(dev.as_ref(),&output_b,vec![h,og*ol]),
    };
    let tx = tn(dev.as_ref(),&x,vec![h]);
    let out = mla_attention(dev.as_ref(), &cfg, &w, &tx, 0).unwrap();
    dev.synchronize().unwrap();
    let mut ob = vec![0u8; h*4]; dev.download(out.buffer.as_ref(), &mut ob).unwrap();
    let got = fb(&ob);

    // CPU ref (pos=0 → rope identity).
    let xn = rms(&x, &attn_norm, cfg.eps);
    let qa = mv(&q_a,&xn,ql,h); let qan = rms(&qa,&q_a_norm,cfg.eps); let qf = mv(&q_b,&qan,qd,ql);
    let mut q = qf.clone();
    for hh in 0..cfg.n_heads { // per-head unit RMS (ones weight)
        let row = &qf[hh*hd..(hh+1)*hd];
        let ms: f32 = row.iter().map(|v| v*v).sum::<f32>()/hd as f32; let s = 1.0/(ms+cfg.eps).sqrt();
        for d in 0..hd { q[hh*hd+d] = row[d]*s; }
    }
    let kvn = rms(&mv(&kv,&xn,hd,h), &kv_a_norm, cfg.eps);
    let scale = 1.0/(hd as f32).sqrt();
    let mut attn = vec![0.0f32; qd];
    for hh in 0..cfg.n_heads {
        let score = scale * (0..hd).map(|d| q[hh*hd+d]*kvn[d]).sum::<f32>();
        let m = score.max(sink[hh]);
        let p = (score-m).exp() / ((score-m).exp() + (sink[hh]-m).exp());
        for d in 0..hd { attn[hh*hd+d] = p*kvn[d]; }
    }
    // grouped O
    let mut o_low = vec![0.0f32; og*ol];
    for g in 0..og { let s = &attn[g*gsize..(g+1)*gsize]; let r = mv(&output_a[g], s, ol, gsize); o_low[g*ol..(g+1)*ol].copy_from_slice(&r); }
    let want = mv(&output_b, &o_low, h, og*ol);

    let mut e = 0.0f32; for i in 0..h { e = e.max((got[i]-want[i]).abs()); }
    eprintln!("DSv4 MLA attention on Metal vs CPU: max|Δ|={e:.3e}");
    assert!(e <= 5e-3, "mla mismatch: {e:.3e}");
    eprintln!("✅ DSv4 MLA attention composite runs on Apple GPU through the shared op layer, matches CPU.");
}
