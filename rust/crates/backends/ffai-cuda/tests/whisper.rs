// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Full real **audio encoder** (Whisper-tiny) forward on the shared engine,
//! verified vs HF transformers. The audio family's encoder = a conv front-end
//! (two Conv1d, done as im2col + `matmul`) + sinusoidal position embed + a
//! bidirectional transformer (LayerNorm + full self-attention + GELU-MLP) —
//! the same shared op set as the VLM tower. Whisper uses exact-erf GELU
//! (`activation_function="gelu"`), applied host-side here (the device
//! GELU-tanh op is verified separately for the VLM tower). Heavy projections
//! on the device `matmul`, attention on `sdpa_decode` (full/bidirectional,
//! looped per query), LayerNorm on device. Deterministic synthetic mel input.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{layer_norm, matmul, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
// erf via Abramowitz-Stegun 7.1.26 (~1.5e-7), then exact GELU = 0.5x(1+erf(x/√2)).
fn erf(x: f32) -> f32 {
    let s = x.signum(); let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y
}
fn gelu_erf(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)) }

#[test]
fn whisper_tiny_encoder_vs_hf() {
    let dir = std::env::var("WHISPER_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    // config (whisper-tiny)
    let (mel, t_in, dm, n_layers, nh, hd, ffn, eps) =
        (80usize, 3000usize, 384usize, 4usize, 6usize, 64usize, 1536usize, 1e-5f32);
    let t_out = t_in / 2; // 1500 after conv2 stride-2
    let scale = 1.0 / (hd as f32).sqrt();

    let g = |name: &str| -> Vec<f32> { let (b, dt, _s) = st.tensor(name).unwrap(); assert_eq!(dt, DType::F32, "{name}"); fb(b) };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let add_bias = |m: &mut [f32], bias: &[f32], rows: usize, cols: usize| {
        for r in 0..rows { for c in 0..cols { m[r * cols + c] += bias[c]; } }
    };

    // mel input [mel, t_in], deterministic
    let feat: Vec<f32> = (0..mel * t_in).map(|i| (0.01 * i as f32).sin()).collect();

    // ── conv1: Conv1d(80,384,k=3,pad=1,stride=1) → [t_in, 384] (im2col + matmul) ──
    let k = 3usize;
    let mut col1 = vec![0.0f32; t_in * (mel * k)];
    for t in 0..t_in { for c in 0..mel { for kk in 0..k {
        let pos = t as isize - 1 + kk as isize; // pad=1
        if pos >= 0 && (pos as usize) < t_in { col1[t * (mel * k) + c * k + kk] = feat[c * t_in + pos as usize]; }
    }}}
    let w1 = g("model.encoder.conv1.weight"); // [384,80,3] = [384, 240]
    let mut c1 = dl(&matmul(d, &up(&w1, vec![dm, mel * k]), &up(&col1, vec![t_in, mel * k])).unwrap(), t_in * dm);
    add_bias(&mut c1, &g("model.encoder.conv1.bias"), t_in, dm);
    for v in c1.iter_mut() { *v = gelu_erf(*v); } // c1: [t_in, 384], access [t][c]

    // ── conv2: Conv1d(384,384,k=3,pad=1,stride=2) → [t_out, 384] ──
    let mut col2 = vec![0.0f32; t_out * (dm * k)];
    for j in 0..t_out { for c in 0..dm { for kk in 0..k {
        let pos = 2 * j as isize - 1 + kk as isize; // stride 2, pad 1
        if pos >= 0 && (pos as usize) < t_in { col2[j * (dm * k) + c * k + kk] = c1[pos as usize * dm + c]; }
    }}}
    let w2 = g("model.encoder.conv2.weight"); // [384,384,3] = [384, 1152]
    let mut c2 = dl(&matmul(d, &up(&w2, vec![dm, dm * k]), &up(&col2, vec![t_out, dm * k])).unwrap(), t_out * dm);
    add_bias(&mut c2, &g("model.encoder.conv2.bias"), t_out, dm);
    for v in c2.iter_mut() { *v = gelu_erf(*v); }

    // + sinusoidal position embedding
    let pos = g("model.encoder.embed_positions.weight"); // [1500,384]
    let mut x = c2;
    for i in 0..t_out * dm { x[i] += pos[i]; }

    // ── encoder layers (pre-norm) ──
    let np = t_out;
    for l in 0..n_layers {
        let p = format!("model.encoder.layers.{l}");
        // self-attention (k_proj has NO bias in Whisper)
        let ln1 = layer_norm(d, &up(&x, vec![np, dm]),
            &up(&g(&format!("{p}.self_attn_layer_norm.weight")), vec![dm]),
            &up(&g(&format!("{p}.self_attn_layer_norm.bias")), vec![dm]), eps).unwrap();
        let mut q = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![dm, dm]), &ln1).unwrap(), np * dm);
        let k_h = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![dm, dm]), &ln1).unwrap(), np * dm);
        let mut v = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![dm, dm]), &ln1).unwrap(), np * dm);
        add_bias(&mut q, &g(&format!("{p}.self_attn.q_proj.bias")), np, dm);
        add_bias(&mut v, &g(&format!("{p}.self_attn.v_proj.bias")), np, dm);
        let mut kb = vec![0.0f32; nh * np * hd];
        let mut vb = vec![0.0f32; nh * np * hd];
        for t in 0..np { for h in 0..nh { for dd in 0..hd {
            kb[h * np * hd + t * hd + dd] = k_h[t * dm + h * hd + dd];
            vb[h * np * hd + t * hd + dd] = v[t * dm + h * hd + dd];
        }}}
        let kt = up(&kb, vec![nh, np, hd]);
        let vt = up(&vb, vec![nh, np, hd]);
        let mut attn = vec![0.0f32; np * dm];
        for t in 0..np {
            let qt = up(&q[t * dm..(t + 1) * dm], vec![nh, hd]);
            let a = sdpa_decode(d, &qt, &kt, &vt, hd, np as u32, np as u32, 1, scale).unwrap();
            attn[t * dm..(t + 1) * dm].copy_from_slice(&dl(&a, dm));
        }
        let mut o = dl(&matmul(d, &up(&g(&format!("{p}.self_attn.out_proj.weight")), vec![dm, dm]), &up(&attn, vec![np, dm])).unwrap(), np * dm);
        add_bias(&mut o, &g(&format!("{p}.self_attn.out_proj.bias")), np, dm);
        for i in 0..np * dm { x[i] += o[i]; }

        // GELU-MLP (exact-erf)
        let ln2 = layer_norm(d, &up(&x, vec![np, dm]),
            &up(&g(&format!("{p}.final_layer_norm.weight")), vec![dm]),
            &up(&g(&format!("{p}.final_layer_norm.bias")), vec![dm]), eps).unwrap();
        let mut h1 = dl(&matmul(d, &up(&g(&format!("{p}.fc1.weight")), vec![ffn, dm]), &ln2).unwrap(), np * ffn);
        add_bias(&mut h1, &g(&format!("{p}.fc1.bias")), np, ffn);
        for vv in h1.iter_mut() { *vv = gelu_erf(*vv); }
        let mut h2 = dl(&matmul(d, &up(&g(&format!("{p}.fc2.weight")), vec![dm, ffn]), &up(&h1, vec![np, ffn])).unwrap(), np * dm);
        add_bias(&mut h2, &g(&format!("{p}.fc2.bias")), np, dm);
        for i in 0..np * dm { x[i] += h2[i]; }
    }

    // final encoder layer_norm = last_hidden_state
    let lhs = dl(&layer_norm(d, &up(&x, vec![np, dm]),
        &up(&g("model.encoder.layer_norm.weight"), vec![dm]),
        &up(&g("model.encoder.layer_norm.bias"), vec![dm]), eps).unwrap(), np * dm);

    let want0 = [0.07884f32, 0.0372, 0.39875, 0.4143, -1.18014];
    let want750 = [1.36551f32, -1.5299, 1.39363, 1.85176, -1.30956];
    let mut e = 0.0f32;
    for i in 0..5 { e = e.max((lhs[i] - want0[i]).abs()); }
    for i in 0..5 { e = e.max((lhs[750 * dm + i] - want750[i]).abs()); }
    let sum: f32 = lhs.iter().sum();
    eprintln!("Whisper-tiny encoder on CUDA: ENC[0,:5]={:?}", &lhs[..5]);
    eprintln!("  ENC[750,:5]={:?}  sum={sum:.2} (HF=12390.46)  max|Δ| first-rows={e:.3e}", &lhs[750 * dm..750 * dm + 5]);
    assert!(e < 3e-2, "Whisper encoder mismatch vs HF: max|Δ|={e:.3e}");
    assert!((sum - 12390.46).abs() < 50.0, "Whisper encoder sum off: {sum}");
    eprintln!("✅ Full real Whisper-tiny audio encoder matches HF on the shared engine (GB10 sm_121) — audio half verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--openai--whisper-tiny/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
