// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Q8_0 resident-weight matvec (`ffai_ops::gemv_q8`) vs a CPU dequant-dot
//! reference. Same kernel IR codegens to CUDA — this is the cheap local proof
//! of the quantized-GEMV path that the resident NemotronH decode loop rides on.
use ffai_core::{DType, Tensor};
use ffai_ops::{gemv_q8, quantize_q8};
use ffai_metal::MetalDevice;

fn tb_f32(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tb_u32(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn gemv_q8_matches_cpu_dequant_dot() {
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    let d = dev.as_ref();
    let (m, k) = (128usize, 256usize);
    let mut s = 0x1234_5678u32;
    let mut rng = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as f32 / u32::MAX as f32) - 0.5 };
    let w: Vec<f32> = (0..m * k).map(|_| rng()).collect();
    let x: Vec<f32> = (0..k).map(|_| rng()).collect();

    let (qs, scales) = quantize_q8(&w, m, k);

    // CPU reference: the SAME dequant the kernel does (int8·scale), dotted with x.
    let bpr = k / 32;
    let mut want = vec![0f32; m];
    for r in 0..m {
        let mut acc = 0f32;
        for b in 0..bpr {
            let dscale = scales[r * bpr + b];
            for i in 0..32 {
                let packed = qs[r * bpr * 8 + b * 8 + i / 4];
                let by = ((packed >> ((i % 4) * 8)) & 0xff) as u8;
                acc += dscale * (by as i8 as f32) * x[b * 32 + i];
            }
        }
        want[r] = acc;
    }

    let qs_t = Tensor::new(d.upload(&tb_u32(&qs)).unwrap(), vec![qs.len()], DType::U32);
    let sc_t = Tensor::new(d.upload(&tb_f32(&scales)).unwrap(), vec![scales.len()], DType::F32);
    let x_t = Tensor::new(d.upload(&tb_f32(&x)).unwrap(), vec![k], DType::F32);
    let out_t = gemv_q8(d, &qs_t, &sc_t, &x_t, m, k, m).unwrap(); // rows_per_group=m ⇒ dense
    let mut ob = vec![0u8; m * 4];
    d.download(out_t.buffer.as_ref(), &mut ob).unwrap();
    let got = fb(&ob);

    let maxerr = (0..m).fold(0f32, |a, r| a.max((got[r] - want[r]).abs()));
    eprintln!("gemv_q8 max abs err vs CPU dequant dot = {maxerr:.6}");
    assert!(maxerr < 1e-3, "gemv_q8 mismatch: {maxerr}");
    eprintln!("✅ gemv_q8 (Q8_0 resident matvec) matches CPU dequant dot");
}

#[test]
fn relu2_matches_host() {
    let Some(dev) = MetalDevice::create().expect("metal") else { return; };
    let d = dev.as_ref();
    let x: Vec<f32> = (-16..16).map(|i| i as f32 * 0.3).collect();
    let xt = Tensor::new(d.upload(&tb_f32(&x)).unwrap(), vec![x.len()], DType::F32);
    let ot = ffai_ops::relu2(d, &xt).unwrap();
    let mut ob = vec![0u8; x.len() * 4];
    d.download(ot.buffer.as_ref(), &mut ob).unwrap();
    let got = fb(&ob);
    let want: Vec<f32> = x.iter().map(|&v| { let r = v.max(0.0); r * r }).collect();
    let e = (0..x.len()).fold(0f32, |a, i| a.max((got[i] - want[i]).abs()));
    eprintln!("relu2 max err = {e:.6}");
    assert!(e < 1e-5, "relu2 mismatch {e}");
}

#[test]
fn fma_inplace_matches_host() {
    let Some(dev) = MetalDevice::create().expect("metal") else { return; };
    let d = dev.as_ref();
    let n = 100usize;
    let acc0: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let x: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let s: Vec<f32> = vec![2.5; n];
    let acc = Tensor::new(d.upload(&tb_f32(&acc0)).unwrap(), vec![n], DType::F32);
    let xt = Tensor::new(d.upload(&tb_f32(&x)).unwrap(), vec![n], DType::F32);
    let st = Tensor::new(d.upload(&tb_f32(&s)).unwrap(), vec![n], DType::F32);
    ffai_ops::fma_inplace(d, &acc, &xt, &st).unwrap();
    let mut ob = vec![0u8; n * 4]; d.download(acc.buffer.as_ref(), &mut ob).unwrap();
    let got = fb(&ob);
    let e = (0..n).fold(0f32, |a, i| a.max((got[i] - (acc0[i] + x[i] * s[i])).abs()));
    eprintln!("fma_inplace max err = {e:.6}");
    assert!(e < 1e-5, "fma mismatch {e}");
}

#[test]
fn gemv_q4_matches_cpu_dequant() {
    let Some(dev) = MetalDevice::create().expect("metal") else { return; };
    let d = dev.as_ref();
    let (m, k) = (96usize, 256usize);
    let mut s = 0xBEEFu32;
    let mut rng = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as f32 / u32::MAX as f32) - 0.5 };
    let w: Vec<f32> = (0..m * k).map(|_| rng()).collect();
    let x: Vec<f32> = (0..k).map(|_| rng()).collect();
    let (qs, sc) = ffai_ops::quantize_q4(&w, m, k);
    let bpr = k / 32;
    let mut want = vec![0f32; m];
    for r in 0..m { let mut a = 0f32;
        for b in 0..bpr { let dd = sc[r*bpr+b];
            for i in 0..32 { let word = qs[r*bpr*4 + b*4 + i/8]; let nib = (word >> ((i%8)*4)) & 0xf;
                let q = nib as i32 - if nib > 7 { 16 } else { 0 }; a += dd * q as f32 * x[b*32+i]; } }
        want[r] = a; }
    let qt = Tensor::new(d.upload(&tb_u32(&qs)).unwrap(), vec![qs.len()], DType::U32);
    let st = Tensor::new(d.upload(&tb_f32(&sc)).unwrap(), vec![sc.len()], DType::F32);
    let xt = Tensor::new(d.upload(&tb_f32(&x)).unwrap(), vec![k], DType::F32);
    let ot = ffai_ops::gemv_q4(d, &qt, &st, &xt, m, k, m).unwrap();
    let mut ob = vec![0u8; m*4]; d.download(ot.buffer.as_ref(), &mut ob).unwrap();
    let got = fb(&ob);
    let e = (0..m).fold(0f32, |a, r| a.max((got[r]-want[r]).abs()));
    eprintln!("gemv_q4 max err = {e:.6}");
    assert!(e < 1e-3, "gemv_q4 mismatch {e}");
}

#[test]
fn slice_and_conv_roll() {
    let Some(dev) = MetalDevice::create().expect("metal") else { return; };
    let d = dev.as_ref();
    let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let st = Tensor::new(d.upload(&tb_f32(&src)).unwrap(), vec![100], DType::F32);
    let sl = ffai_ops::slice(d, &st, 10, 5).unwrap();
    let mut ob = vec![0u8; 5*4]; d.download(sl.buffer.as_ref(), &mut ob).unwrap();
    assert_eq!(fb(&ob), vec![10.,11.,12.,13.,14.], "slice");
    // conv_roll: conv_dim=2, kc=3 → state=4 (2 blocks), keep=2; new=[old[2..4], xbc[0..2]]
    let old = Tensor::new(d.upload(&tb_f32(&[1.,2.,3.,4.])).unwrap(), vec![4], DType::F32);
    let xbc = Tensor::new(d.upload(&tb_f32(&[9.,8.])).unwrap(), vec![2], DType::F32);
    let nr = ffai_ops::conv_roll(d, &old, &xbc, 2, 3).unwrap();
    let mut ob = vec![0u8; 4*4]; d.download(nr.buffer.as_ref(), &mut ob).unwrap();
    assert_eq!(fb(&ob), vec![3.,4.,9.,8.], "conv_roll");
    eprintln!("✅ slice + conv_roll correct");
}
