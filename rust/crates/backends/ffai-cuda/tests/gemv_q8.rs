// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Q8_0 resident-weight matvec (`ffai_ops::gemv_q8`) vs a CPU dequant-dot
//! reference. Same kernel IR codegens to CUDA — this is the cheap local proof
//! of the quantized-GEMV path that the resident NemotronH decode loop rides on.
use ffai_core::{DType, Tensor};
use ffai_ops::{gemv_q8, quantize_q8};
use ffai_cuda::CudaDevice;

fn tb_f32(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tb_u32(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn gemv_q8_matches_cpu_dequant_dot() {
    let Some(dev) = CudaDevice::create().expect("metal") else { eprintln!("no CUDA — skip"); return; };
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
