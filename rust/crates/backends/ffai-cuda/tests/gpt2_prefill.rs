// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! GPT-2 multi-token **causal prefill** verified vs HF — the one remaining
//! primitive (all other model tests are single-token at pos 0). Processes an
//! 8-token prompt through the full stack; each position attends causally over
//! [0, i] (via `sdpa_decode` with n_kv = i+1 against the per-head K/V cache,
//! kv_stride = seq_len). Proves causal masking + the prefill path on the shared
//! op layer. (GPT-2 uses learned position embeddings, isolating causal masking
//! from RoPE.) argmax of the last position vs HF.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{gelu, layer_norm, matmul, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn gpt2_causal_prefill_vs_hf() {
    let dir = std::env::var("GPT2_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    let (hid, nh, hd, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 12usize, 50257usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();
    let ids = [464usize, 2068, 7586, 21831, 11687, 625, 262, 16931];
    let seq = ids.len();

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let add_bias = |m: &mut [f32], b: &[f32], rows: usize, cols: usize| { for r in 0..rows { for c in 0..cols { m[r * cols + c] += b[c]; } } };
    let conv_t = |w: &[f32], nin: usize, nout: usize| -> Vec<f32> {
        let mut o = vec![0.0f32; nin * nout];
        for i in 0..nin { for j in 0..nout { o[j * nin + i] = w[i * nout + j]; } } o
    };
    let reorg = |m: &[f32], n: usize| -> Vec<f32> { // [n,hid] → [nh,n,hd]
        let mut o = vec![0.0f32; nh * n * hd];
        for t in 0..n { for h in 0..nh { for dd in 0..hd { o[h * n * hd + t * hd + dd] = m[t * hid + h * hd + dd]; } } } o
    };

    let wte = g("wte.weight"); let wpe = g("wpe.weight");
    // x[seq, hid] = token embed + learned position embed
    let mut x = vec![0.0f32; seq * hid];
    for (i, &tok) in ids.iter().enumerate() {
        for e in 0..hid { x[i * hid + e] = wte[tok * hid + e] + wpe[i * hid + e]; }
    }

    for l in 0..n_layers {
        let p = format!("h.{l}");
        // attention (pre-LN over all positions)
        let h = dl(&layer_norm(d, &up(&x, vec![seq, hid]), &up(&g(&format!("{p}.ln_1.weight")), vec![hid]), &up(&g(&format!("{p}.ln_1.bias")), vec![hid]), eps).unwrap(), seq * hid);
        let cattn_w = conv_t(&g(&format!("{p}.attn.c_attn.weight")), hid, 3 * hid);
        let mut qkv = dl(&matmul(d, &up(&cattn_w, vec![3 * hid, hid]), &up(&h, vec![seq, hid])).unwrap(), seq * 3 * hid);
        add_bias(&mut qkv, &g(&format!("{p}.attn.c_attn.bias")), seq, 3 * hid);
        // split q/k/v [seq,hid]
        let mut q = vec![0.0f32; seq * hid]; let mut k = vec![0.0f32; seq * hid]; let mut v = vec![0.0f32; seq * hid];
        for t in 0..seq {
            q[t * hid..(t + 1) * hid].copy_from_slice(&qkv[t * 3 * hid..t * 3 * hid + hid]);
            k[t * hid..(t + 1) * hid].copy_from_slice(&qkv[t * 3 * hid + hid..t * 3 * hid + 2 * hid]);
            v[t * hid..(t + 1) * hid].copy_from_slice(&qkv[t * 3 * hid + 2 * hid..t * 3 * hid + 3 * hid]);
        }
        let kt = up(&reorg(&k, seq), vec![nh, seq, hd]);
        let vt = up(&reorg(&v, seq), vec![nh, seq, hd]);
        // causal: position i attends over [0, i] → n_kv = i+1, kv_stride = seq
        let mut attn = vec![0.0f32; seq * hid];
        for i in 0..seq {
            let qi = up(&q[i * hid..(i + 1) * hid], vec![nh, hd]);
            let a = sdpa_decode(d, &qi, &kt, &vt, hd, (i + 1) as u32, seq as u32, 1, scale).unwrap();
            attn[i * hid..(i + 1) * hid].copy_from_slice(&dl(&a, hid));
        }
        let cproj_w = conv_t(&g(&format!("{p}.attn.c_proj.weight")), hid, hid);
        let mut o = dl(&matmul(d, &up(&cproj_w, vec![hid, hid]), &up(&attn, vec![seq, hid])).unwrap(), seq * hid);
        add_bias(&mut o, &g(&format!("{p}.attn.c_proj.bias")), seq, hid);
        for i in 0..seq * hid { x[i] += o[i]; }

        // MLP (pre-LN, gelu_new = tanh)
        let h2 = dl(&layer_norm(d, &up(&x, vec![seq, hid]), &up(&g(&format!("{p}.ln_2.weight")), vec![hid]), &up(&g(&format!("{p}.ln_2.bias")), vec![hid]), eps).unwrap(), seq * hid);
        let fc_w = conv_t(&g(&format!("{p}.mlp.c_fc.weight")), hid, 4 * hid);
        let mut f = dl(&matmul(d, &up(&fc_w, vec![4 * hid, hid]), &up(&h2, vec![seq, hid])).unwrap(), seq * 4 * hid);
        add_bias(&mut f, &g(&format!("{p}.mlp.c_fc.bias")), seq, 4 * hid);
        let act = dl(&gelu(d, &up(&f, vec![seq * 4 * hid])).unwrap(), seq * 4 * hid);
        let proj_w = conv_t(&g(&format!("{p}.mlp.c_proj.weight")), 4 * hid, hid);
        let mut m = dl(&matmul(d, &up(&proj_w, vec![hid, 4 * hid]), &up(&act, vec![seq, 4 * hid])).unwrap(), seq * hid);
        add_bias(&mut m, &g(&format!("{p}.mlp.c_proj.bias")), seq, hid);
        for i in 0..seq * hid { x[i] += m[i]; }
    }

    // final LN; logits at LAST position (tied lm_head)
    let xf = dl(&layer_norm(d, &up(&x, vec![seq, hid]), &up(&g("ln_f.weight"), vec![hid]), &up(&g("ln_f.bias"), vec![hid]), eps).unwrap(), seq * hid);
    let last = &xf[(seq - 1) * hid..seq * hid];
    let logits = dl(&matmul(d, &up(&wte, vec![vocab, hid]), &up(last, vec![1, hid])).unwrap(), vocab);
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    eprintln!("GPT-2 causal prefill (seq={seq}) on CUDA: top3 = {:?} (HF = [11, 21831, 7586])", &idx[..3]);
    assert_eq!(&idx[..3], &[11usize, 21831, 7586], "GPT-2 prefill top-3 != HF");
    eprintln!("✅ GPT-2 multi-token causal prefill matches HF on the shared engine (GB10 sm_121) — causal prefill path verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--gpt2/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
