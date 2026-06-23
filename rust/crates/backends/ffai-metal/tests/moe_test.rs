// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! MoE feed-forward (router → top-k → per-expert SwiGLU → weighted sum) on
//! Metal vs a CPU reference. Proves the MoE compute path — the exotic family
//! covering DeepSeek-V4 / GPT-OSS / Granite4 / Qwen-MoE — runs correctly on
//! the shared op layer. (Real MoE-model-vs-HF verification follows once the
//! large expert weights are staged.)

use ffai_core::{DType, Device, Tensor};
use ffai_metal::MetalDevice;
use ffai_models::moe::{ExpertWeights, MoeMlp, moe_mlp};

fn tb(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn fill(n: usize, s: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7 + s * 131) % 97) as f32 - 48.0) * 0.01).collect()
}
fn tens(d: &dyn Device, v: &[f32], shape: Vec<usize>) -> Tensor {
    Tensor::new(d.upload(&tb(v)).unwrap(), shape, DType::F32)
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn mv(m: &[f32], v: &[f32], rows: usize, k: usize) -> Vec<f32> {
    (0..rows).map(|r| (0..k).map(|c| m[r * k + c] * v[c]).sum()).collect()
}

#[test]
fn moe_mlp_on_metal_matches_cpu() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    const H: usize = 256;
    const INTER: usize = 512;
    const NE: usize = 8;
    const TOPK: usize = 2;

    let router = fill(NE * H, 1);
    let h = fill(H, 99);
    let experts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..NE)
        .map(|e| (fill(INTER * H, 10 + e), fill(INTER * H, 100 + e), fill(H * INTER, 200 + e)))
        .collect();

    let w = MoeMlp {
        router: tens(dev.as_ref(), &router, vec![NE, H]),
        experts: experts
            .iter()
            .map(|(g, u, d)| ExpertWeights {
                gate: tens(dev.as_ref(), g, vec![INTER, H]),
                up: tens(dev.as_ref(), u, vec![INTER, H]),
                down: tens(dev.as_ref(), d, vec![H, INTER]),
            })
            .collect(),
        top_k: TOPK,
        norm_topk: true,
    };
    let th = tens(dev.as_ref(), &h, vec![H]);

    let out = moe_mlp(dev.as_ref(), &w, &th).unwrap();
    dev.synchronize().unwrap();
    let mut ob = vec![0u8; H * 4];
    dev.download(out.buffer.as_ref(), &mut ob).unwrap();
    let got: Vec<f32> = ob.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    // CPU reference.
    let logits = mv(&router, &h, NE, H);
    let mut order: Vec<usize> = (0..NE).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top: Vec<usize> = order.into_iter().take(TOPK).collect();
    let m = top.iter().map(|&i| logits[i]).fold(f32::MIN, f32::max);
    let e: Vec<f32> = top.iter().map(|&i| (logits[i] - m).exp()).collect();
    let s: f32 = e.iter().sum();
    let wts: Vec<f32> = e.iter().map(|x| x / s).collect();
    let mut want = vec![0.0f32; H];
    for (&ei, &gw) in top.iter().zip(&wts) {
        let (g, u, d) = &experts[ei];
        let gate = mv(g, &h, INTER, H);
        let up = mv(u, &h, INTER, H);
        let act: Vec<f32> = (0..INTER).map(|i| silu(gate[i]) * up[i]).collect();
        let out = mv(d, &act, H, INTER);
        for i in 0..H {
            want[i] += gw * out[i];
        }
    }

    let mut err = 0.0f32;
    for i in 0..H {
        err = err.max((got[i] - want[i]).abs());
    }
    eprintln!("MoE MLP on Metal vs CPU: max|Δ|={err:.3e} (top experts {top:?})");
    assert!(err <= 5e-3, "moe mismatch: {err:.3e}");
    eprintln!("✅ MoE feed-forward runs on Apple GPU through the shared op layer, matches CPU.");
}
