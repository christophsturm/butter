// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! GPT-2 autoregressive **greedy decode** (prefill + 30-token generation loop)
//! verified vs HF `generate(do_sample=False)`. Proves the full generate path —
//! prefill → argmax → append → re-forward — and **coherence**: the engine must
//! emit the exact token sequence HF does. Prints the generated ids so they can
//! be detokenized to text. (Each step re-forwards the growing sequence — an
//! incremental KV cache is a runtime optimization, not needed for correctness.)
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{gelu, layer_norm, matmul, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn gpt2_greedy_generate_vs_hf() {
    let dir = std::env::var("GPT2_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    let (hid, nh, hd, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 12usize, 50257usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let add_bias = |m: &mut [f32], b: &[f32], rows: usize, cols: usize| { for r in 0..rows { for c in 0..cols { m[r * cols + c] += b[c]; } } };
    let conv_t = |w: &[f32], nin: usize, nout: usize| -> Vec<f32> { let mut o = vec![0.0f32; nin * nout]; for i in 0..nin { for j in 0..nout { o[j * nin + i] = w[i * nout + j]; } } o };
    let reorg = |m: &[f32], n: usize| -> Vec<f32> { let mut o = vec![0.0f32; nh * n * hd]; for t in 0..n { for h in 0..nh { for dd in 0..hd { o[h * n * hd + t * hd + dd] = m[t * hid + h * hd + dd]; } } } o };

    // preload weights once
    let wte = g("wte.weight"); let wpe = g("wpe.weight");
    let lw: Vec<_> = (0..n_layers).map(|l| {
        let p = format!("h.{l}");
        (g(&format!("{p}.ln_1.weight")), g(&format!("{p}.ln_1.bias")),
         conv_t(&g(&format!("{p}.attn.c_attn.weight")), hid, 3 * hid), g(&format!("{p}.attn.c_attn.bias")),
         conv_t(&g(&format!("{p}.attn.c_proj.weight")), hid, hid), g(&format!("{p}.attn.c_proj.bias")),
         g(&format!("{p}.ln_2.weight")), g(&format!("{p}.ln_2.bias")),
         conv_t(&g(&format!("{p}.mlp.c_fc.weight")), hid, 4 * hid), g(&format!("{p}.mlp.c_fc.bias")),
         conv_t(&g(&format!("{p}.mlp.c_proj.weight")), 4 * hid, hid), g(&format!("{p}.mlp.c_proj.bias")))
    }).collect();
    let lnf_w = g("ln_f.weight"); let lnf_b = g("ln_f.bias");

    // one full forward over `ids`, returns argmax of last position
    let forward = |ids: &[usize]| -> usize {
        let seq = ids.len();
        let mut x = vec![0.0f32; seq * hid];
        for (i, &tok) in ids.iter().enumerate() { for e in 0..hid { x[i * hid + e] = wte[tok * hid + e] + wpe[i * hid + e]; } }
        for w in &lw {
            let h = dl(&layer_norm(d, &up(&x, vec![seq, hid]), &up(&w.0, vec![hid]), &up(&w.1, vec![hid]), eps).unwrap(), seq * hid);
            let mut qkv = dl(&matmul(d, &up(&w.2, vec![3 * hid, hid]), &up(&h, vec![seq, hid])).unwrap(), seq * 3 * hid);
            add_bias(&mut qkv, &w.3, seq, 3 * hid);
            let (mut q, mut k, mut v) = (vec![0.0f32; seq * hid], vec![0.0f32; seq * hid], vec![0.0f32; seq * hid]);
            for t in 0..seq {
                q[t * hid..(t + 1) * hid].copy_from_slice(&qkv[t * 3 * hid..t * 3 * hid + hid]);
                k[t * hid..(t + 1) * hid].copy_from_slice(&qkv[t * 3 * hid + hid..t * 3 * hid + 2 * hid]);
                v[t * hid..(t + 1) * hid].copy_from_slice(&qkv[t * 3 * hid + 2 * hid..t * 3 * hid + 3 * hid]);
            }
            let kt = up(&reorg(&k, seq), vec![nh, seq, hd]); let vt = up(&reorg(&v, seq), vec![nh, seq, hd]);
            let mut attn = vec![0.0f32; seq * hid];
            for i in 0..seq {
                let a = sdpa_decode(d, &up(&q[i * hid..(i + 1) * hid], vec![nh, hd]), &kt, &vt, hd, (i + 1) as u32, seq as u32, 1, scale).unwrap();
                attn[i * hid..(i + 1) * hid].copy_from_slice(&dl(&a, hid));
            }
            let mut o = dl(&matmul(d, &up(&w.4, vec![hid, hid]), &up(&attn, vec![seq, hid])).unwrap(), seq * hid);
            add_bias(&mut o, &w.5, seq, hid);
            for i in 0..seq * hid { x[i] += o[i]; }
            let h2 = dl(&layer_norm(d, &up(&x, vec![seq, hid]), &up(&w.6, vec![hid]), &up(&w.7, vec![hid]), eps).unwrap(), seq * hid);
            let mut f = dl(&matmul(d, &up(&w.8, vec![4 * hid, hid]), &up(&h2, vec![seq, hid])).unwrap(), seq * 4 * hid);
            add_bias(&mut f, &w.9, seq, 4 * hid);
            let act = dl(&gelu(d, &up(&f, vec![seq * 4 * hid])).unwrap(), seq * 4 * hid);
            let mut m = dl(&matmul(d, &up(&w.10, vec![hid, 4 * hid]), &up(&act, vec![seq, 4 * hid])).unwrap(), seq * hid);
            add_bias(&mut m, &w.11, seq, hid);
            for i in 0..seq * hid { x[i] += m[i]; }
        }
        let xf = dl(&layer_norm(d, &up(&x, vec![seq, hid]), &up(&lnf_w, vec![hid]), &up(&lnf_b, vec![hid]), eps).unwrap(), seq * hid);
        let last = &xf[(seq - 1) * hid..seq * hid];
        let logits = dl(&matmul(d, &up(&wte, vec![vocab, hid]), &up(last, vec![1, hid])).unwrap(), vocab);
        (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap()
    };

    // greedy decode loop
    let mut ids: Vec<usize> = vec![464, 3139, 286, 4881, 318]; // "The capital of France is"
    let prompt_len = ids.len();
    for _ in 0..30 { let nxt = forward(&ids); ids.push(nxt); }
    let outv = &ids[prompt_len..];
    eprintln!("GPT-2 greedy gen on CUDA: {outv:?}");

    let hf = [262usize, 3139, 286, 262, 4141, 2066, 11, 290, 262, 3139, 286, 262, 4141, 2066, 318, 262, 3139, 286, 262, 4141, 2066, 13, 198, 198, 464, 4141, 2066, 318, 262, 3139];
    assert_eq!(outv, &hf, "GPT-2 greedy generation diverges from HF");
    eprintln!("✅ GPT-2 greedy decode (30 tokens) matches HF generate() exactly — prefill + decode + coherence on the shared engine.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--gpt2/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
