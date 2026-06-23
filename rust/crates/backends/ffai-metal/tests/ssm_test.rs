// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Mamba2 SSD selective-scan decode step (mt_ssm_step) on Metal vs CPU —
//! the core SSM-family op (Mamba2/Jamba/FalconH1/LFM2).
use ffai_core::{DType, Device, Tensor};
use ffai_metal::MetalDevice;
use ffai_ops::{ssm_step, ssm_prefill_scan, ssm_prefill_scan_ssd_portable};

fn tb(v:&[f32])->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn fb(b:&[u8])->Vec<f32>{b.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect()}
fn fill(n:usize,s:usize)->Vec<f32>{(0..n).map(|i|(((i*7+s*131)%89) as f32-44.0)*0.02).collect()}
fn tn(d:&dyn Device,v:&[f32],sh:Vec<usize>)->Tensor{Tensor::new(d.upload(&tb(v)).unwrap(),sh,DType::F32)}

#[test]
fn mt_ssm_step_on_metal_matches_cpu(){
    let Some(dev)=MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
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
    eprintln!("mt_ssm_step on Metal vs CPU: state max|Δ|={es:.3e}  out max|Δ|={eo:.3e}");
    assert!(es<=1e-4 && eo<=1e-4, "ssm mismatch state={es:.3e} out={eo:.3e}");
    eprintln!("✅ Mamba2 SSD selective-scan step runs on Apple GPU through the shared op layer, matches CPU.");
}

// Direct scan-level gate: the PORTABLE chunked-matmul SSD prefill scan
// (ssm_prefill_scan_ssd_portable, all #[kernel] ops + ffai_gemm_batched) must
// match the sequential reference (ssm_prefill_scan) to tight tolerance — the
// whole point of the state-space-duality rewrite is bit-equivalence to the
// serial recurrence (modulo f32 accumulation order). Nemotron cell dims.
fn cos_sim(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn run_portable_vs_seq(t: usize, l: u32) {
    let Some(dev)=MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    let d = dev.as_ref();
    // Nemotron cell: dh=64, ds=128, H=64, G=8.
    let (dh, ds, h, ng) = (64usize, 128usize, 64usize, 8usize);
    // Synthetic but well-conditioned inputs (small dt so decays are stable).
    let x = fill(t * h * dh, 11);
    let a_log: Vec<f32> = (0..h).map(|i| -0.5 - 0.01 * (i % 7) as f32).collect();
    let b = fill(t * ng * ds, 13);
    let c = fill(t * ng * ds, 17);
    let dsk = fill(h, 19);
    let dt_v: Vec<f32> = (0..t * h).map(|i| 0.02 + 0.001 * ((i % 11) as f32)).collect();
    let si = vec![0.0f32; h * dh * ds]; // prefill starts from zero state
    let mk = |v: &[f32], sh: Vec<usize>| tn(d, v, sh);

    let (so_seq, y_seq) = ssm_prefill_scan(
        d, &mk(&x, vec![t * h * dh]), &mk(&a_log, vec![h]),
        &mk(&b, vec![t * ng * ds]), &mk(&c, vec![t * ng * ds]),
        &mk(&dsk, vec![h]), &mk(&dt_v, vec![t * h]),
        &mk(&si, vec![h * dh * ds]), t as u32, dh as u32, ds as u32, h as u32, ng as u32,
    ).unwrap();
    let (so_port, y_port) = ssm_prefill_scan_ssd_portable(
        d, &mk(&x, vec![t * h * dh]), &mk(&a_log, vec![h]),
        &mk(&b, vec![t * ng * ds]), &mk(&c, vec![t * ng * ds]),
        &mk(&dsk, vec![h]), &mk(&dt_v, vec![t * h]),
        &mk(&si, vec![h * dh * ds]), t as u32, dh as u32, ds as u32, h as u32, ng as u32, l,
    ).unwrap();
    d.synchronize().unwrap();

    let dl = |ts: &Tensor, n: usize| { let mut bb = vec![0u8; n * 4]; d.download(ts.buffer.as_ref(), &mut bb).unwrap(); fb(&bb) };
    let ys = dl(&y_seq, t * h * dh);
    let yp = dl(&y_port, t * h * dh);
    let ss = dl(&so_seq, h * dh * ds);
    let sp = dl(&so_port, h * dh * ds);

    let y_max = (0..ys.len()).map(|i| (ys[i] - yp[i]).abs()).fold(0.0f32, f32::max);
    let s_max = (0..ss.len()).map(|i| (ss[i] - sp[i]).abs()).fold(0.0f32, f32::max);
    let y_cos = cos_sim(&ys, &yp);
    let s_cos = cos_sim(&ss, &sp);
    let y_norm = (ys.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / ys.len() as f64).sqrt();
    eprintln!("portable-SSD vs sequential scan [T={t}, L={l}, nc={}]:", (t as u32).div_ceil(l));
    eprintln!("  y:     max|Δ|={y_max:.3e}  cosine={y_cos:.8}  (y RMS≈{y_norm:.4})");
    eprintln!("  state: max|Δ|={s_max:.3e}  cosine={s_cos:.8}");
    assert!(y_cos > 0.9999, "y cosine {y_cos:.8} too low (T={t}, L={l})");
    assert!(s_cos > 0.9999, "state cosine {s_cos:.8} too low (T={t}, L={l})");
    eprintln!("✅ portable chunked-matmul SSD scan matches sequential (cosine>0.9999) — portable kernels only.");
}

// nc=1 (single chunk: pure intra-chunk path, no inter-chunk recurrence).
#[test]
fn ssd_portable_scan_single_chunk_metal() { run_portable_vs_seq(128, 128); }

// nc>1 with a partial tail chunk (exercises serial inter-chunk carry + zero-pad).
#[test]
fn ssd_portable_scan_multi_chunk_metal() { run_portable_vs_seq(300, 128); }

// L=256, multiple full chunks (the default chunk length).
#[test]
fn ssd_portable_scan_l256_metal() { run_portable_vs_seq(512, 256); }
