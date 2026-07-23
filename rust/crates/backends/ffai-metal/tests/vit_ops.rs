// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! LayerNorm + GELU(tanh) ops vs CPU reference — the two new primitives the
//! VLM vision tower (ViT / SigLIP / CLIP) needs beyond the LLM op set.
use ffai_core::{DType, Tensor};
use ffai_metal::MetalDevice;
use ffai_ops::{gelu, layer_norm, matmul};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn gelu_tanh(x: f32) -> f32 { 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh()) }

#[test]
fn layernorm_and_gelu_vs_cpu() {
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    let d = dev.as_ref();
    let up = |v: &[f32], sh: Vec<usize>| Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32);
    let dl = |t: &Tensor, n: usize| { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };

    // LayerNorm over n=768 (SigLIP-base width), 2 rows.
    let (rows, n, eps) = (2usize, 768usize, 1e-6f32);
    let x: Vec<f32> = (0..rows * n).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03).collect();
    let w: Vec<f32> = (0..n).map(|i| 1.0 + ((i % 11) as f32 - 5.0) * 0.02).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
    let got = dl(&layer_norm(d, &up(&x, vec![rows, n]), &up(&w, vec![n]), &up(&b, vec![n]), eps).unwrap(), rows * n);
    let mut e = 0.0f32;
    for r in 0..rows {
        let row = &x[r * n..(r + 1) * n];
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let is = 1.0 / (var + eps).sqrt();
        for i in 0..n {
            let want = (row[i] - mean) * is * w[i] + b[i];
            e = e.max((got[r * n + i] - want).abs());
        }
    }
    eprintln!("LayerNorm max|Δ| = {e:.2e}");
    assert!(e < 1e-4, "layer_norm mismatch {e}");

    // GELU(tanh)
    let xv: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.05).collect();
    let g = dl(&gelu(d, &up(&xv, vec![xv.len()])).unwrap(), xv.len());
    let mut eg = 0.0f32;
    for i in 0..xv.len() { eg = eg.max((g[i] - gelu_tanh(xv[i])).abs()); }
    eprintln!("GELU(tanh) max|Δ| = {eg:.2e}");
    assert!(eg < 1e-4, "gelu mismatch {eg}");

    // matmul (prefill linear): out[r,:] = weight · input[r,:], weight[out,in], input[rows,in]
    let (rows2, in_d, out_d) = (196usize, 768usize, 1024usize); // rows not mult of 32 (edge)
    let wt: Vec<f32> = (0..out_d * in_d).map(|i| ((i * 13 % 97) as f32 - 48.0) * 0.01).collect();
    let inp: Vec<f32> = (0..rows2 * in_d).map(|i| ((i * 7 % 89) as f32 - 44.0) * 0.01).collect();
    let mm = dl(&matmul(d, &up(&wt, vec![out_d, in_d]), &up(&inp, vec![rows2, in_d])).unwrap(), rows2 * out_d);
    let mut em = 0.0f32;
    for r in [0usize, 95, 195] {
        for o in [0usize, 500, 1023] {
            let mut acc = 0.0f32;
            for kk in 0..in_d { acc += wt[o * in_d + kk] * inp[r * in_d + kk]; }
            em = em.max((mm[r * out_d + o] - acc).abs());
        }
    }
    eprintln!("matmul max|Δ| = {em:.2e}");
    assert!(em < 2e-3, "matmul mismatch {em}");
    eprintln!("✅ LayerNorm + GELU(tanh) + matmul match CPU on Apple GPU — ViT primitives ready.");
}
