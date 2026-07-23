// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Llama-family multi-token causal prefill WITH RoPE-at-position, verified vs
//! HF — completes the prefill primitive (GPT-2 prefill proved causal masking
//! with learned positions; this proves RoPE applied per real position + GQA +
//! SwiGLU in a causal prefill). Uses SmolVLM-256M's text model (a SmolLM2-style
//! Llama: hidden 576, GQA 9q/3kv, head_dim 64, rope θ=1e5, 30 layers), run
//! text-only. This is exactly the text-half prefill a full VLM forward runs
//! after splicing image embeds — every piece now on the shared op layer.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{matmul, rms_norm, rope_llama, sdpa_decode, swiglu};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn llama_causal_prefill_rope_vs_hf() {
    let dir = std::env::var("SMOLVLM_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    let (hid, nq, nkv, hd, inter, n_layers, vocab, eps, theta) =
        (576usize, 9usize, 3usize, 64usize, 1536usize, 30usize, 49280usize, 1e-5f32, 100000.0f32);
    let qd = nq * hd; let kvd = nkv * hd; // 576, 192
    let scale = 1.0 / (hd as f32).sqrt();
    let ids = [1usize, 2520, 1396, 253, 8137, 275, 253];
    let seq = ids.len();

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };

    let tp = "model.text_model";
    let embed = g(&format!("{tp}.embed_tokens.weight"));
    let mut x = vec![0.0f32; seq * hid];
    for (i, &tok) in ids.iter().enumerate() { x[i * hid..(i + 1) * hid].copy_from_slice(&embed[tok * hid..(tok + 1) * hid]); }

    for l in 0..n_layers {
        let p = format!("{tp}.layers.{l}");
        // attention (pre-RMSNorm)
        let h = dl(&rms_norm(d, &up(&x, vec![seq, hid]), &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]), eps).unwrap(), seq * hid);
        let q = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![qd, hid]), &up(&h, vec![seq, hid])).unwrap(), seq * qd);
        let k = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![kvd, hid]), &up(&h, vec![seq, hid])).unwrap(), seq * kvd);
        let v = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![kvd, hid]), &up(&h, vec![seq, hid])).unwrap(), seq * kvd);
        // RoPE at each real position
        let mut qr = vec![0.0f32; seq * qd]; let mut kr = vec![0.0f32; seq * kvd];
        for t in 0..seq {
            let qt = rope_llama(d, &up(&q[t * qd..(t + 1) * qd], vec![nq, hd]), t as u32, theta, 1.0, 1.0, 1.0, 1e9).unwrap();
            qr[t * qd..(t + 1) * qd].copy_from_slice(&dl(&qt, qd));
            let kt = rope_llama(d, &up(&k[t * kvd..(t + 1) * kvd], vec![nkv, hd]), t as u32, theta, 1.0, 1.0, 1.0, 1e9).unwrap();
            kr[t * kvd..(t + 1) * kvd].copy_from_slice(&dl(&kt, kvd));
        }
        // K/V cache [nkv, seq, hd]
        let mut kc = vec![0.0f32; nkv * seq * hd]; let mut vc = vec![0.0f32; nkv * seq * hd];
        for t in 0..seq { for h2 in 0..nkv { for dd in 0..hd {
            kc[h2 * seq * hd + t * hd + dd] = kr[t * kvd + h2 * hd + dd];
            vc[h2 * seq * hd + t * hd + dd] = v[t * kvd + h2 * hd + dd];
        }}}
        let kt = up(&kc, vec![nkv, seq, hd]); let vt = up(&vc, vec![nkv, seq, hd]);
        // causal GQA: position i attends [0,i]
        let mut attn = vec![0.0f32; seq * qd];
        for i in 0..seq {
            let qi = up(&qr[i * qd..(i + 1) * qd], vec![nq, hd]);
            let a = sdpa_decode(d, &qi, &kt, &vt, hd, (i + 1) as u32, seq as u32, (nq / nkv) as u32, scale).unwrap();
            attn[i * qd..(i + 1) * qd].copy_from_slice(&dl(&a, qd));
        }
        let o = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, qd]), &up(&attn, vec![seq, qd])).unwrap(), seq * hid);
        for i in 0..seq * hid { x[i] += o[i]; }

        // SwiGLU MLP (pre-RMSNorm)
        let h2 = dl(&rms_norm(d, &up(&x, vec![seq, hid]), &up(&g(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), eps).unwrap(), seq * hid);
        let gate = matmul(d, &up(&g(&format!("{p}.mlp.gate_proj.weight")), vec![inter, hid]), &up(&h2, vec![seq, hid])).unwrap();
        let upp = matmul(d, &up(&g(&format!("{p}.mlp.up_proj.weight")), vec![inter, hid]), &up(&h2, vec![seq, hid])).unwrap();
        let act = swiglu(d, &gate, &upp).unwrap();
        let down = dl(&matmul(d, &up(&g(&format!("{p}.mlp.down_proj.weight")), vec![hid, inter]), &act.reshaped(vec![seq, inter])).unwrap(), seq * hid);
        for i in 0..seq * hid { x[i] += down[i]; }
    }

    // final norm; logits at last position (untied lm_head)
    let xf = dl(&rms_norm(d, &up(&x, vec![seq, hid]), &up(&g(&format!("{tp}.norm.weight")), vec![hid]), eps).unwrap(), seq * hid);
    let last = &xf[(seq - 1) * hid..seq * hid];
    let logits = dl(&matmul(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &up(last, vec![1, hid])).unwrap(), vocab);
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    eprintln!("Llama (SmolVLM-text) RoPE causal prefill (seq={seq}) on CUDA: top3 = {:?} (HF = [12642, 4052, 216])", &idx[..3]);
    assert_eq!(&idx[..3], &[12642usize, 4052, 216], "Llama prefill top-3 != HF");
    eprintln!("✅ Llama RoPE-at-position causal prefill matches HF on the shared engine (GB10 sm_121) — full prefill path verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
