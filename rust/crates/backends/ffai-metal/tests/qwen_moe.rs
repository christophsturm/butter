// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Real MoE feed-forward vs HF: load a real Qwen2-MoE block's weights and
//! run the MoE forward (softmax→top-k routing, SwiGLU experts, sigmoid-gated
//! shared expert) on the shared op layer, comparing to HF transformers'
//! Qwen2MoeSparseMoeBlock output for the same input. Turns MoE from
//! "compute-verified-vs-CPU" into "real-weights-verified-vs-HF".
use ffai_core::{DType, Tensor};
use ffai_metal::MetalDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{gemv, swiglu};

fn fb(b:&[u8])->Vec<f32>{b.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect()}
fn tb(v:&[f32])->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()}

#[test]
fn qwen2_moe_block_on_metal_matches_hf() {
    let path = std::env::var("QWENMOE_DIR").ok()
        .map(|d| format!("{d}/model.safetensors"))
        .unwrap_or_else(|| std::fs::read_to_string("/tmp/qwenmoe_path.txt").map(|s| format!("{}/model.safetensors", s.trim())).unwrap_or_default());
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };

    let (h, _moe_i, ne, tk) = (32usize, 44usize, 8usize, 4usize);
    let t = |name: &str| -> Tensor {
        let (bytes, dt, shape) = st.tensor(name).unwrap();
        assert_eq!(dt, DType::F32);
        Tensor::new(dev.upload(bytes).unwrap(), shape.to_vec(), DType::F32)
    };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b=vec![0u8;n*4]; dev.download(t.buffer.as_ref(),&mut b).unwrap(); fb(&b) };

    let x: Vec<f32> = (0..h).map(|i| i as f32 * 0.03 - 0.5).collect();
    let tx = Tensor::new(dev.upload(&tb(&x)).unwrap(), vec![h], DType::F32);
    let p = "model.layers.0.mlp";

    // Router: softmax over all, top-k (norm_topk_prob=false → raw probs).
    let logits_t = gemv(dev.as_ref(), &t(&format!("{p}.gate.weight")), &tx).unwrap();
    dev.synchronize().unwrap();
    let logits = dl(&logits_t, ne);
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / s).collect();
    let mut order: Vec<usize> = (0..ne).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let top: Vec<usize> = order.into_iter().take(tk).collect();

    let mut acc = vec![0.0f32; h];
    for &e in &top {
        let g = gemv(dev.as_ref(), &t(&format!("{p}.experts.{e}.gate_proj.weight")), &tx).unwrap();
        let u = gemv(dev.as_ref(), &t(&format!("{p}.experts.{e}.up_proj.weight")), &tx).unwrap();
        let act = swiglu(dev.as_ref(), &g, &u).unwrap();
        let o = gemv(dev.as_ref(), &t(&format!("{p}.experts.{e}.down_proj.weight")), &act).unwrap();
        dev.synchronize().unwrap();
        let ov = dl(&o, h);
        for i in 0..h { acc[i] += probs[e] * ov[i]; }
    }
    // Shared expert, sigmoid-gated.
    let slog = gemv(dev.as_ref(), &t(&format!("{p}.shared_expert_gate.weight")), &tx).unwrap();
    dev.synchronize().unwrap();
    let sg_val = 1.0 / (1.0 + (-dl(&slog, 1)[0]).exp());
    let sg = gemv(dev.as_ref(), &t(&format!("{p}.shared_expert.gate_proj.weight")), &tx).unwrap();
    let su = gemv(dev.as_ref(), &t(&format!("{p}.shared_expert.up_proj.weight")), &tx).unwrap();
    let sact = swiglu(dev.as_ref(), &sg, &su).unwrap();
    let so = gemv(dev.as_ref(), &t(&format!("{p}.shared_expert.down_proj.weight")), &sact).unwrap();
    dev.synchronize().unwrap();
    let sov = dl(&so, h);
    for i in 0..h { acc[i] += sg_val * sov[i]; }

    let hf = [-2.6e-05f32,-2.5e-05,4e-06,-7.1e-05,1.3e-05,3.5e-05,2.5e-05,1e-06,-2.7e-05,-2e-05,3.2e-05,-2.2e-05,3.2e-05,2.8e-05,-2.5e-05,9e-06,-2.5e-05,0.0,6e-05,2e-06,-1.9e-05,1e-05,-1.6e-05,3.2e-05,3.1e-05,2.2e-05,-3.2e-05,2.2e-05,9e-06,2.5e-05,1e-05,-2.2e-05];
    let mut e = 0.0f32; for i in 0..h { e = e.max((acc[i]-hf[i]).abs()); }
    eprintln!("Qwen2-MoE block on Metal vs HF: max|Δ|={e:.2e}  (top experts {top:?})");
    eprintln!("rust[..6]={:?}", &acc[..6]);
    assert!(e <= 3e-6, "qwen moe vs HF mismatch: {e:.2e}");
    eprintln!("✅ Real Qwen2-MoE feed-forward matches HF on the shared op layer (Apple GPU).");
}
