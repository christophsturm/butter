// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real VLM **vision tower** (SigLIP-base-patch16-224) forward on the
//! shared engine, verified vs HF transformers. This is the VLM family's new
//! half — vision encoder = conv/patch-embed (as a matmul) + bidirectional
//! transformer (LayerNorm + full self-attention + GELU-MLP). The LLM half is
//! already covered (dense Llama family); a VLM is vision-tower → projector →
//! LLM. Heavy projections run on the device `matmul`; attention on the device
//! `sdpa_decode` (full/bidirectional = attend over all n_kv patches, looped
//! per query); LayerNorm + GELU(tanh) on device. Bias/residual/head-reorg are
//! trivial host elementwise. Input is a deterministic synthetic pixel tensor
//! (same `sin(0.01·i)` formula as the HF reference).
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{gelu, layer_norm, matmul, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn siglip_vision_tower_vs_hf() {
    let dir = std::env::var("SIGLIP_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    // config (SigLIP-base-patch16-224 vision)
    let (hid, n_layers, nh, hd, inter, img, patch, eps) =
        (768usize, 12usize, 12usize, 64usize, 3072usize, 224usize, 16usize, 1e-6f32);
    let grid = img / patch;            // 14
    let np = grid * grid;              // 196 patches
    let scale = 1.0 / (hd as f32).sqrt(); // 0.125

    let g = |name: &str| -> Vec<f32> { let (b, dt, _s) = st.tensor(name).unwrap(); assert_eq!(dt, DType::F32, "{name}"); fb(b) };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    // add a per-column bias to a [rows, cols] host buffer (broadcast over rows)
    let add_bias = |m: &mut [f32], bias: &[f32], rows: usize, cols: usize| {
        for r in 0..rows { for c in 0..cols { m[r * cols + c] += bias[c]; } }
    };

    // ── patch embedding (conv2d stride=patch ≡ matmul over flattened patches) ──
    let n_pix = 3 * img * img;
    let pv: Vec<f32> = (0..n_pix).map(|i| (0.01 * i as f32).sin()).collect();
    // patch matrix [np, 3*patch*patch], inner order (c, kh, kw) to match conv weight flatten
    let pdim = 3 * patch * patch; // 768
    let mut patches = vec![0.0f32; np * pdim];
    for gh in 0..grid { for gw in 0..grid {
        let p = gh * grid + gw;
        for c in 0..3 { for kh in 0..patch { for kw in 0..patch {
            let h = gh * patch + kh; let w = gw * patch + kw;
            patches[p * pdim + c * patch * patch + kh * patch + kw] = pv[c * img * img + h * img + w];
        }}}
    }}
    let cw = g("vision_model.embeddings.patch_embedding.weight"); // [768, 3,16,16] = [768, 768]
    let emb = dl(&matmul(d, &up(&cw, vec![hid, pdim]), &up(&patches, vec![np, pdim])).unwrap(), np * hid);
    let cb = g("vision_model.embeddings.patch_embedding.bias");
    let pos = g("vision_model.embeddings.position_embedding.weight"); // [196, 768]
    let mut x = emb.clone();
    add_bias(&mut x, &cb, np, hid);
    for i in 0..np * hid { x[i] += pos[i]; }

    // ── encoder ──
    for l in 0..n_layers {
        let p = format!("vision_model.encoder.layers.{l}");
        // self-attention
        let ln1 = layer_norm(d, &up(&x, vec![np, hid]),
            &up(&g(&format!("{p}.layer_norm1.weight")), vec![hid]),
            &up(&g(&format!("{p}.layer_norm1.bias")), vec![hid]), eps).unwrap();
        let mut q = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![hid, hid]), &ln1).unwrap(), np * hid);
        let mut k = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![hid, hid]), &ln1).unwrap(), np * hid);
        let mut v = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![hid, hid]), &ln1).unwrap(), np * hid);
        add_bias(&mut q, &g(&format!("{p}.self_attn.q_proj.bias")), np, hid);
        add_bias(&mut k, &g(&format!("{p}.self_attn.k_proj.bias")), np, hid);
        add_bias(&mut v, &g(&format!("{p}.self_attn.v_proj.bias")), np, hid);
        // reorg to per-head KV cache [nh, np, hd] for sdpa_decode (kv_stride = np)
        let mut kb = vec![0.0f32; nh * np * hd];
        let mut vb = vec![0.0f32; nh * np * hd];
        for t in 0..np { for h in 0..nh { for dd in 0..hd {
            kb[h * np * hd + t * hd + dd] = k[t * hid + h * hd + dd];
            vb[h * np * hd + t * hd + dd] = v[t * hid + h * hd + dd];
        }}}
        let kt = up(&kb, vec![nh, np, hd]);
        let vt = up(&vb, vec![nh, np, hd]);
        // full bidirectional attention: each query patch attends over all np patches
        let mut attn = vec![0.0f32; np * hid];
        for t in 0..np {
            let qt = up(&q[t * hid..(t + 1) * hid], vec![nh, hd]);
            let a = sdpa_decode(d, &qt, &kt, &vt, hd, np as u32, np as u32, 1, scale).unwrap();
            let ad = dl(&a, hid);
            attn[t * hid..(t + 1) * hid].copy_from_slice(&ad);
        }
        let mut o = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.out_proj.weight")), vec![hid, hid]), &up(&attn, vec![np, hid])).unwrap(), np * hid);
        add_bias(&mut o, &g(&format!("{p}.self_attn.out_proj.bias")), np, hid);
        for i in 0..np * hid { x[i] += o[i]; } // residual

        // GELU-MLP
        let ln2 = layer_norm(d, &up(&x, vec![np, hid]),
            &up(&g(&format!("{p}.layer_norm2.weight")), vec![hid]),
            &up(&g(&format!("{p}.layer_norm2.bias")), vec![hid]), eps).unwrap();
        let mut h1 = dl(&matmul(d, &up(&g(&format!("{p}.mlp.fc1.weight")), vec![inter, hid]), &ln2).unwrap(), np * inter);
        add_bias(&mut h1, &g(&format!("{p}.mlp.fc1.bias")), np, inter);
        let act = gelu(d, &up(&h1, vec![np, inter])).unwrap();
        let mut h2 = dl(&matmul(d, &up(&g(&format!("{p}.mlp.fc2.weight")), vec![hid, inter]), &act).unwrap(), np * hid);
        add_bias(&mut h2, &g(&format!("{p}.mlp.fc2.bias")), np, hid);
        for i in 0..np * hid { x[i] += h2[i]; } // residual
    }

    // ── post layernorm = last_hidden_state ──
    let lhs = dl(&layer_norm(d, &up(&x, vec![np, hid]),
        &up(&g("vision_model.post_layernorm.weight"), vec![hid]),
        &up(&g("vision_model.post_layernorm.bias"), vec![hid]), eps).unwrap(), np * hid);

    // HF reference (deterministic sin(0.01·i) pixel input)
    let want0 = [-0.05489f32, -0.43045, 0.37643, 0.09968, -1.22139];
    let want100 = [0.03598f32, 1.52808, 0.41827, -0.1857, -1.84941];
    let mut e = 0.0f32;
    for i in 0..5 { e = e.max((lhs[i] - want0[i]).abs()); }
    for i in 0..5 { e = e.max((lhs[100 * hid + i] - want100[i]).abs()); }
    let sum: f32 = lhs.iter().sum();
    eprintln!("SigLIP vision tower on CUDA: LHS[0,:5]={:?}", &lhs[..5]);
    eprintln!("  LHS[100,:5]={:?}  sum={sum:.3} (HF sum=-792.795)  max|Δ| first-rows={e:.3e}", &lhs[100 * hid..100 * hid + 5]);
    assert!(e < 2e-2, "SigLIP last_hidden_state mismatch vs HF: max|Δ|={e:.3e}");
    assert!((sum + 792.795).abs() < 5.0, "SigLIP LHS sum off: {sum}");
    eprintln!("✅ Full real SigLIP vision tower matches HF on the shared engine (GB10 sm_121) — VLM vision half verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--google--siglip-base-patch16-224/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
