// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real Whisper-base **encoder→decoder** STT forward (first decoder step),
//! verified vs HF. Adds the **cross-attention** mechanism: the decoder's query
//! attends over the encoder's 1500 output states (K/V from the encoder, per
//! decoder layer) — the same pattern every encoder-decoder and many VLMs use.
//! Encoder is the same conv-front-end + bidirectional transformer verified for
//! whisper-tiny (here d512/6L/8H). Decoder layer = causal self-attn (n_kv=1 at
//! pos 0) + cross-attn (n_kv=1500) + exact-erf GELU MLP, pre-norm, tied lm_head.
//! All on the shared op layer; exact-erf GELU host-side.
use ffai_core::{DType, Tensor};
use ffai_metal::MetalDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{add, gemv, layer_norm, matmul, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn erf(x: f32) -> f32 {
    let s = x.signum(); let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y
}
fn gelu_erf(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)) }

#[test]
fn whisper_base_stt_vs_hf() {
    let dir = std::env::var("WHISPER_BASE_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    let d = dev.as_ref();

    let (mel, t_in, dm, nh, hd, ffn, eps) = (80usize, 3000usize, 512usize, 8usize, 64usize, 2048usize, 1e-5f32);
    let (enc_layers, dec_layers, vocab) = (6usize, 6usize, 51865usize);
    let t_out = t_in / 2; // 1500
    let scale = 1.0 / (hd as f32).sqrt();

    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let add_bias = |m: &mut [f32], b: &[f32], rows: usize, cols: usize| { for r in 0..rows { for c in 0..cols { m[r * cols + c] += b[c]; } } };
    // reorg [n, dm] → per-head KV cache [nh, n, hd]
    let reorg = |m: &[f32], n: usize| -> Vec<f32> {
        let mut o = vec![0.0f32; nh * n * hd];
        for t in 0..n { for h in 0..nh { for dd in 0..hd { o[h * n * hd + t * hd + dd] = m[t * dm + h * hd + dd]; } } }
        o
    };

    // ════ ENCODER ════
    let feat: Vec<f32> = (0..mel * t_in).map(|i| (0.01 * i as f32).sin()).collect();
    let k = 3usize;
    // conv1 (80→512, k3 pad1 stride1)
    let mut col1 = vec![0.0f32; t_in * (mel * k)];
    for t in 0..t_in { for c in 0..mel { for kk in 0..k {
        let pos = t as isize - 1 + kk as isize;
        if pos >= 0 && (pos as usize) < t_in { col1[t * (mel * k) + c * k + kk] = feat[c * t_in + pos as usize]; }
    }}}
    let mut c1 = dl(&matmul(d, &up(&g("model.encoder.conv1.weight"), vec![dm, mel * k]), &up(&col1, vec![t_in, mel * k])).unwrap(), t_in * dm);
    add_bias(&mut c1, &g("model.encoder.conv1.bias"), t_in, dm);
    for v in c1.iter_mut() { *v = gelu_erf(*v); }
    // conv2 (512→512, k3 pad1 stride2)
    let mut col2 = vec![0.0f32; t_out * (dm * k)];
    for j in 0..t_out { for c in 0..dm { for kk in 0..k {
        let pos = 2 * j as isize - 1 + kk as isize;
        if pos >= 0 && (pos as usize) < t_in { col2[j * (dm * k) + c * k + kk] = c1[pos as usize * dm + c]; }
    }}}
    let mut enc = dl(&matmul(d, &up(&g("model.encoder.conv2.weight"), vec![dm, dm * k]), &up(&col2, vec![t_out, dm * k])).unwrap(), t_out * dm);
    add_bias(&mut enc, &g("model.encoder.conv2.bias"), t_out, dm);
    for v in enc.iter_mut() { *v = gelu_erf(*v); }
    let epos = g("model.encoder.embed_positions.weight");
    for i in 0..t_out * dm { enc[i] += epos[i]; }
    // full self-attention over `src` [n,dm] (normed); returns attn output [n*dm]
    let self_attn = |normed: &[f32], pfx: &str, n: usize| -> Vec<f32> {
        let mut q = dl(&matmul(d, &up(&g(&format!("{pfx}.q_proj.weight")), vec![dm, dm]), &up(normed, vec![n, dm])).unwrap(), n * dm);
        add_bias(&mut q, &g(&format!("{pfx}.q_proj.bias")), n, dm);
        let kk = dl(&matmul(d, &up(&g(&format!("{pfx}.k_proj.weight")), vec![dm, dm]), &up(normed, vec![n, dm])).unwrap(), n * dm); // no bias
        let mut vv = dl(&matmul(d, &up(&g(&format!("{pfx}.v_proj.weight")), vec![dm, dm]), &up(normed, vec![n, dm])).unwrap(), n * dm);
        add_bias(&mut vv, &g(&format!("{pfx}.v_proj.bias")), n, dm);
        let kt = up(&reorg(&kk, n), vec![nh, n, hd]); let vt = up(&reorg(&vv, n), vec![nh, n, hd]);
        let mut out = vec![0.0f32; n * dm];
        for t in 0..n {
            let qt = up(&q[t * dm..(t + 1) * dm], vec![nh, hd]);
            let a = sdpa_decode(d, &qt, &kt, &vt, hd, n as u32, n as u32, 1, scale).unwrap();
            out[t * dm..(t + 1) * dm].copy_from_slice(&dl(&a, dm));
        }
        let mut o = dl(&matmul(d, &up(&g(&format!("{pfx}.out_proj.weight")), vec![dm, dm]), &up(&out, vec![n, dm])).unwrap(), n * dm);
        add_bias(&mut o, &g(&format!("{pfx}.out_proj.bias")), n, dm);
        o
    };
    for l in 0..enc_layers {
        let p = format!("model.encoder.layers.{l}");
        let ln = dl(&layer_norm(d, &up(&enc, vec![t_out, dm]), &up(&g(&format!("{p}.self_attn_layer_norm.weight")), vec![dm]), &up(&g(&format!("{p}.self_attn_layer_norm.bias")), vec![dm]), eps).unwrap(), t_out * dm);
        let o = self_attn(&ln, &format!("{p}.self_attn"), t_out);
        for i in 0..t_out * dm { enc[i] += o[i]; }
        let ln2 = dl(&layer_norm(d, &up(&enc, vec![t_out, dm]), &up(&g(&format!("{p}.final_layer_norm.weight")), vec![dm]), &up(&g(&format!("{p}.final_layer_norm.bias")), vec![dm]), eps).unwrap(), t_out * dm);
        let mut h1 = dl(&matmul(d, &up(&g(&format!("{p}.fc1.weight")), vec![ffn, dm]), &up(&ln2, vec![t_out, dm])).unwrap(), t_out * ffn);
        add_bias(&mut h1, &g(&format!("{p}.fc1.bias")), t_out, ffn);
        for v in h1.iter_mut() { *v = gelu_erf(*v); }
        let mut h2 = dl(&matmul(d, &up(&g(&format!("{p}.fc2.weight")), vec![dm, ffn]), &up(&h1, vec![t_out, ffn])).unwrap(), t_out * dm);
        add_bias(&mut h2, &g(&format!("{p}.fc2.bias")), t_out, dm);
        for i in 0..t_out * dm { enc[i] += h2[i]; }
    }
    let enc_out = dl(&layer_norm(d, &up(&enc, vec![t_out, dm]), &up(&g("model.encoder.layer_norm.weight"), vec![dm]), &up(&g("model.encoder.layer_norm.bias"), vec![dm]), eps).unwrap(), t_out * dm);

    // ════ DECODER (single SOT token at pos 0) ════
    let sot = 50258usize;
    let demb = g("model.decoder.embed_tokens.weight");
    let dpos = g("model.decoder.embed_positions.weight");
    let mut x: Vec<f32> = (0..dm).map(|i| demb[sot * dm + i] + dpos[i]).collect();
    let dproj = |w: &str, b: Option<&str>, x: &Tensor, m: usize, inn: usize| -> Vec<f32> {
        let mut o = dl(&gemv(d, &up(&g(w), vec![m, inn]), x).unwrap(), m);
        if let Some(bb) = b { let bv = g(bb); for i in 0..m { o[i] += bv[i]; } }
        o
    };
    for l in 0..dec_layers {
        let p = format!("model.decoder.layers.{l}");
        // causal self-attn (n_kv=1 at pos0)
        let h = layer_norm(d, &up(&x, vec![dm]), &up(&g(&format!("{p}.self_attn_layer_norm.weight")), vec![dm]), &up(&g(&format!("{p}.self_attn_layer_norm.bias")), vec![dm]), eps).unwrap();
        let q = dproj(&format!("{p}.self_attn.q_proj.weight"), Some(&format!("{p}.self_attn.q_proj.bias")), &h, dm, dm);
        let kk = dproj(&format!("{p}.self_attn.k_proj.weight"), None, &h, dm, dm);
        let vv = dproj(&format!("{p}.self_attn.v_proj.weight"), Some(&format!("{p}.self_attn.v_proj.bias")), &h, dm, dm);
        let sa = sdpa_decode(d, &up(&q, vec![nh, hd]), &up(&kk, vec![nh, hd]), &up(&vv, vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = dproj(&format!("{p}.self_attn.out_proj.weight"), Some(&format!("{p}.self_attn.out_proj.bias")), &sa.reshaped(vec![dm]), dm, dm);
        for i in 0..dm { x[i] += o[i]; }
        // cross-attn (q from decoder, K/V from encoder output)
        let hc = layer_norm(d, &up(&x, vec![dm]), &up(&g(&format!("{p}.encoder_attn_layer_norm.weight")), vec![dm]), &up(&g(&format!("{p}.encoder_attn_layer_norm.bias")), vec![dm]), eps).unwrap();
        let qc = dproj(&format!("{p}.encoder_attn.q_proj.weight"), Some(&format!("{p}.encoder_attn.q_proj.bias")), &hc, dm, dm);
        let kc = dl(&matmul(d, &up(&g(&format!("{p}.encoder_attn.k_proj.weight")), vec![dm, dm]), &up(&enc_out, vec![t_out, dm])).unwrap(), t_out * dm); // no bias
        let mut vc = dl(&matmul(d, &up(&g(&format!("{p}.encoder_attn.v_proj.weight")), vec![dm, dm]), &up(&enc_out, vec![t_out, dm])).unwrap(), t_out * dm);
        add_bias(&mut vc, &g(&format!("{p}.encoder_attn.v_proj.bias")), t_out, dm);
        let kt = up(&reorg(&kc, t_out), vec![nh, t_out, hd]); let vt = up(&reorg(&vc, t_out), vec![nh, t_out, hd]);
        let ca = sdpa_decode(d, &up(&qc, vec![nh, hd]), &kt, &vt, hd, t_out as u32, t_out as u32, 1, scale).unwrap();
        let oc = dproj(&format!("{p}.encoder_attn.out_proj.weight"), Some(&format!("{p}.encoder_attn.out_proj.bias")), &ca.reshaped(vec![dm]), dm, dm);
        for i in 0..dm { x[i] += oc[i]; }
        // MLP
        let hm = layer_norm(d, &up(&x, vec![dm]), &up(&g(&format!("{p}.final_layer_norm.weight")), vec![dm]), &up(&g(&format!("{p}.final_layer_norm.bias")), vec![dm]), eps).unwrap();
        let mut f = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.fc1.weight")), vec![ffn, dm]), &hm).unwrap(), &up(&g(&format!("{p}.fc1.bias")), vec![ffn])).unwrap(), ffn);
        for v in f.iter_mut() { *v = gelu_erf(*v); }
        let m2 = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.fc2.weight")), vec![dm, ffn]), &up(&f, vec![ffn])).unwrap(), &up(&g(&format!("{p}.fc2.bias")), vec![dm])).unwrap(), dm);
        for i in 0..dm { x[i] += m2[i]; }
    }
    let xf = layer_norm(d, &up(&x, vec![dm]), &up(&g("model.decoder.layer_norm.weight"), vec![dm]), &up(&g("model.decoder.layer_norm.bias"), vec![dm]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&demb, vec![vocab, dm]), &xf).unwrap(), vocab); // tied proj_out
    let argmax = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
    eprintln!("Whisper-base STT on Metal: argmax = {argmax} (HF = 50362)");
    assert_eq!(argmax, 50362, "Whisper-base STT argmax != HF 50362");
    eprintln!("✅ Full real Whisper-base encoder→decoder STT matches HF on the shared engine (Apple GPU) — cross-attention path verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--openai--whisper-base/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
