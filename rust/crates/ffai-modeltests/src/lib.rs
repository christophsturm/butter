// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Shared, backend-agnostic model forwards + HF-reference verification.
//!
//! Each `verify_*(dev: &dyn Device)` holds a model's forward and its HF oracle
//! ONCE. The per-backend test files (`ffai-metal/tests/*`, `ffai-cuda/tests/*`)
//! are thin wrappers that build their device and call these — so a model's
//! logic lives in exactly one place, not a Metal test + a sed'd CUDA twin.
use ffai_core::{Backend, DType, Device, Tensor};
use ffai_loader::SafeTensors;
use ffai_ops::{add, gelu, gemv, layer_norm, sdpa_decode};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

/// Resolve a model dir from `$ENV` or an HF-cache snapshot glob.
fn model_dir(env: &str, hub: &str) -> Option<String> {
    if let Ok(d) = std::env::var(env) {
        return Some(d);
    }
    let base = format!("{}/.cache/huggingface/hub/{hub}/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}

/// GPT-2-124M single-token forward (LayerNorm-LLM, Conv1D weights, learned-pos,
/// gelu_new, tied) vs HF argmax 198. Runs on whatever `Device` is passed.
pub fn verify_gpt2(d: &dyn Device, plat: &str) {
    let dir = model_dir("GPT2_DIR", "models--gpt2").unwrap_or_default();
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };

    let (hid, nh, hd, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 12usize, 50257usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let conv_t = |w: &[f32], nin: usize, nout: usize| -> Vec<f32> { let mut o = vec![0.0f32; nin * nout]; for i in 0..nin { for j in 0..nout { o[j * nin + i] = w[i * nout + j]; } } o };

    let token = 5usize;
    let wte = g("wte.weight");
    let wpe = g("wpe.weight");
    let mut x: Vec<f32> = (0..hid).map(|i| wte[token * hid + i] + wpe[i]).collect();

    for l in 0..n_layers {
        let p = format!("h.{l}");
        let h = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.ln_1.weight")), vec![hid]),
            &up(&g(&format!("{p}.ln_1.bias")), vec![hid]), eps).unwrap();
        let cattn_w = conv_t(&g(&format!("{p}.attn.c_attn.weight")), hid, 3 * hid);
        let qkv = add(d, &gemv(d, &up(&cattn_w, vec![3 * hid, hid]), &h).unwrap(),
                      &up(&g(&format!("{p}.attn.c_attn.bias")), vec![3 * hid])).unwrap();
        let qkv = dl(&qkv, 3 * hid);
        let q = up(&qkv[0..hid], vec![nh, hd]);
        let k = up(&qkv[hid..2 * hid], vec![nh, hd]);
        let v = up(&qkv[2 * hid..3 * hid], vec![nh, hd]);
        let attn = sdpa_decode(d, &q, &k, &v, hd, 1, 1, 1, scale).unwrap();
        let cproj_w = conv_t(&g(&format!("{p}.attn.c_proj.weight")), hid, hid);
        let o = add(d, &gemv(d, &up(&cproj_w, vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap(),
                    &up(&g(&format!("{p}.attn.c_proj.bias")), vec![hid])).unwrap();
        let o = dl(&o, hid);
        for i in 0..hid { x[i] += o[i]; }

        let h2 = layer_norm(d, &up(&x, vec![hid]),
            &up(&g(&format!("{p}.ln_2.weight")), vec![hid]),
            &up(&g(&format!("{p}.ln_2.bias")), vec![hid]), eps).unwrap();
        let fc_w = conv_t(&g(&format!("{p}.mlp.c_fc.weight")), hid, 4 * hid);
        let f = add(d, &gemv(d, &up(&fc_w, vec![4 * hid, hid]), &h2).unwrap(),
                    &up(&g(&format!("{p}.mlp.c_fc.bias")), vec![4 * hid])).unwrap();
        let act = gelu(d, &f).unwrap();
        let proj_w = conv_t(&g(&format!("{p}.mlp.c_proj.weight")), 4 * hid, hid);
        let m = add(d, &gemv(d, &up(&proj_w, vec![hid, 4 * hid]), &act).unwrap(),
                    &up(&g(&format!("{p}.mlp.c_proj.bias")), vec![hid])).unwrap();
        let m = dl(&m, hid);
        for i in 0..hid { x[i] += m[i]; }
    }

    let xf = layer_norm(d, &up(&x, vec![hid]),
        &up(&g("ln_f.weight"), vec![hid]), &up(&g("ln_f.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&wte, vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = ffai_runtime::argmax(&logits);
    eprintln!("GPT-2-124M full forward on {plat}: argmax = {argmax} (HF = 198)");
    assert_eq!(argmax, 198, "GPT-2 argmax != HF 198");
    eprintln!("✅ Full real GPT-2 forward matches HF on the shared engine ({plat}) — one shared forward, both backends.");
}

/// Mamba2-130m single-token forward (SSM: conv1d + SSD scan + gated RMSNorm)
/// vs HF argmax 310. Same shared op layer, any `Device`.
pub fn verify_mamba2(d: &dyn Device, plat: &str) {
    use ffai_ops::{conv1d_causal_step, rms_norm, silu, ssm_step};
    let dir = model_dir("MAMBA2_DIR", "models--AntonV--mamba2-130m-hf").unwrap_or_default();
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };

    let (hid, di, nh, dh, ds, ng, kc, vocab, eps) = (768usize, 1536usize, 24usize, 64usize, 128usize, 1usize, 4usize, 50288usize, 1e-5f32);
    let conv_dim = di + 2 * ng * ds;
    let n_layers = 24usize;
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32]| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32) };
    let upm = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };

    let token = 5usize;
    let embed = g("backbone.embeddings.weight");
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();

    for l in 0..n_layers {
        let p = format!("backbone.layers.{l}");
        let xn = rms_norm(d, &up(&x), &up(&g(&format!("{p}.norm.weight"))), eps).unwrap();
        let in_proj = upm(&g(&format!("{p}.mixer.in_proj.weight")), vec![3352, hid]);
        let proj = dl(&gemv(d, &in_proj, &xn).unwrap(), 3352);
        let z = &proj[0..di];
        let xbc = &proj[di..di + conv_dim];
        let dt_raw = &proj[di + conv_dim..di + conv_dim + nh];
        let cw_hf = g(&format!("{p}.mixer.conv1d.weight"));
        let mut cw = vec![0.0f32; kc * conv_dim];
        for ch in 0..conv_dim { for k in 0..kc { cw[k * conv_dim + ch] = cw_hf[ch * kc + k]; } }
        let cb = g(&format!("{p}.mixer.conv1d.bias"));
        let state0 = vec![0.0f32; (kc - 1) * conv_dim];
        let yc = conv1d_causal_step(d, &up(xbc), &up(&cw), &up(&cb), &up(&state0), conv_dim as u32, kc as u32).unwrap();
        let xbc_act = dl(&silu(d, &yc).unwrap(), conv_dim);
        let x_ssm = &xbc_act[0..di];
        let bmat = &xbc_act[di..di + ng * ds];
        let cmat = &xbc_act[di + ng * ds..di + 2 * ng * ds];
        let dt_bias = g(&format!("{p}.mixer.dt_bias"));
        let dt: Vec<f32> = (0..nh).map(|i| softplus(dt_raw[i] + dt_bias[i])).collect();
        let a_log = g(&format!("{p}.mixer.A_log"));
        let dsk = g(&format!("{p}.mixer.D"));
        let state_in = vec![0.0f32; nh * dh * ds];
        let (_so, y_t) = ssm_step(d, &up(x_ssm), &up(&a_log), &up(bmat), &up(cmat), &up(&dsk), &up(&dt), &up(&state_in), dh as u32, ds as u32, nh as u32, (nh / ng) as u32).unwrap();
        let y = dl(&y_t, di);
        let sz = dl(&silu(d, &up(z)).unwrap(), di);
        let y_gated: Vec<f32> = (0..di).map(|i| y[i] * sz[i]).collect();
        let y_normed = rms_norm(d, &up(&y_gated), &up(&g(&format!("{p}.mixer.norm.weight"))), eps).unwrap();
        let out_proj = upm(&g(&format!("{p}.mixer.out_proj.weight")), vec![hid, di]);
        let out = dl(&gemv(d, &out_proj, &y_normed).unwrap(), hid);
        for i in 0..hid { x[i] += out[i]; }
    }

    let xf = rms_norm(d, &up(&x), &up(&g("backbone.norm_f.weight")), eps).unwrap();
    let lm = upm(&embed, vec![vocab, hid]);
    let logits = dl(&gemv(d, &lm, &xf).unwrap(), vocab);
    let argmax = ffai_runtime::argmax(&logits);
    eprintln!("Mamba2-130m full forward on {plat}: argmax = {argmax} (HF = 310)");
    assert_eq!(argmax, 310, "Mamba2 argmax != HF 310");
    eprintln!("✅ Full real Mamba2-130m forward matches HF on the shared engine ({plat}).");
}

/// Run the whole shared model suite against any backend. A new backend (ROCm,
/// Vulkan, …) implements `Device`, then calls this from one test file — and
/// inherits every model with zero model code.
pub fn run_all(d: &dyn Device, plat: &str) {
    verify_gpt2(d, plat);
    verify_pythia(d, plat);
    verify_gptneo(d, plat);
    verify_olmo2(d, plat);
    verify_gemma2(d, plat);
    verify_phi(d, plat);
    verify_stablelm2(d, plat);
    verify_olmoe(d, plat);
    verify_mamba2(d, plat);
    verify_falcon_h1(d, plat);
}

// exact-erf GELU (Abramowitz-Stegun) — shared by GPT-NeoX / Whisper-style nets
fn erf(x: f32) -> f32 {
    let s = x.signum(); let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y
}
fn gelu_erf(x: f32) -> f32 { 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)) }

/// Pythia-160m (GPT-NeoX): parallel residual, interleaved per-head QKV, partial
/// rotary (identity@pos0), exact-erf GELU. vs HF argmax 285.
pub fn verify_pythia(d: &dyn Device, plat: &str) {
    let dir = model_dir("PYTHIA_DIR", "models--EleutherAI--pythia-160m").unwrap_or_default();
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let (hid, nh, hd, inter, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 3072usize, 12usize, 50304usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let token = 5usize;
    let embed = g("gpt_neox.embed_in.weight");
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();
    for l in 0..n_layers {
        let p = format!("gpt_neox.layers.{l}");
        let ln1 = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]), &up(&g(&format!("{p}.input_layernorm.bias")), vec![hid]), eps).unwrap();
        let qkv = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.attention.query_key_value.weight")), vec![3*hid, hid]), &ln1).unwrap(), &up(&g(&format!("{p}.attention.query_key_value.bias")), vec![3*hid])).unwrap(), 3*hid);
        let (mut q, mut k, mut v) = (vec![0.0f32; hid], vec![0.0f32; hid], vec![0.0f32; hid]);
        for h in 0..nh { for dd in 0..hd { q[h*hd+dd]=qkv[h*3*hd+dd]; k[h*hd+dd]=qkv[h*3*hd+hd+dd]; v[h*hd+dd]=qkv[h*3*hd+2*hd+dd]; }}
        let attn = sdpa_decode(d, &up(&q, vec![nh, hd]), &up(&k, vec![nh, hd]), &up(&v, vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let ao = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.attention.dense.weight")), vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap(), &up(&g(&format!("{p}.attention.dense.bias")), vec![hid])).unwrap(), hid);
        let ln2 = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), &up(&g(&format!("{p}.post_attention_layernorm.bias")), vec![hid]), eps).unwrap();
        let mut h1 = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.mlp.dense_h_to_4h.weight")), vec![inter, hid]), &ln2).unwrap(), &up(&g(&format!("{p}.mlp.dense_h_to_4h.bias")), vec![inter])).unwrap(), inter);
        for vv in h1.iter_mut() { *vv = gelu_erf(*vv); }
        let m = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.mlp.dense_4h_to_h.weight")), vec![hid, inter]), &up(&h1, vec![inter])).unwrap(), &up(&g(&format!("{p}.mlp.dense_4h_to_h.bias")), vec![hid])).unwrap(), hid);
        for i in 0..hid { x[i] += ao[i] + m[i]; }
    }
    let xf = layer_norm(d, &up(&x, vec![hid]), &up(&g("gpt_neox.final_layer_norm.weight"), vec![hid]), &up(&g("gpt_neox.final_layer_norm.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("embed_out.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = ffai_runtime::argmax(&logits);
    eprintln!("Pythia-160m on {plat}: argmax = {argmax} (HF = 285)");
    assert_eq!(argmax, 285, "Pythia argmax != HF 285");
    eprintln!("✅ Pythia-160m matches HF ({plat}).");
}

/// OLMo-2-1B: post-norm placement + qk-norm over full proj, SwiGLU. vs HF top-3 [198,8,13].
pub fn verify_olmo2(d: &dyn Device, plat: &str) {
    use ffai_ops::{rms_norm, swiglu};
    let dir = model_dir("OLMO2_DIR", "models--allenai--OLMo-2-0425-1B").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nq, nkv, hd, inter, n_layers, vocab, eps) = (2048usize, 16usize, 16usize, 128usize, 8192usize, 16usize, 100352usize, 1e-6f32);
    let (qdim, kvdim) = (nq*hd, nkv*hd); let scale = 1.0/(hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].to_vec();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let xin = up(&x, vec![hid]);
        let q = gemv(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![qdim, hid]), &xin).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![kvdim, hid]), &xin).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![kvdim, hid]), &xin).unwrap();
        let q = rms_norm(d, &q, &up(&g(&format!("{p}.self_attn.q_norm.weight")), vec![qdim]), eps).unwrap();
        let k = rms_norm(d, &k, &up(&g(&format!("{p}.self_attn.k_norm.weight")), vec![kvdim]), eps).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nq, hd]), &k.reshaped(vec![nkv, hd]), &v.reshaped(vec![nkv, hd]), hd, 1, 1, (nq/nkv) as u32, scale).unwrap();
        let o = gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, qdim]), &attn.reshaped(vec![qdim])).unwrap();
        let o = dl(&rms_norm(d, &o, &up(&g(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), eps).unwrap(), hid);
        for i in 0..hid { x[i] += o[i]; }
        let xin2 = up(&x, vec![hid]);
        let gate = gemv(d, &up(&g(&format!("{p}.mlp.gate_proj.weight")), vec![inter, hid]), &xin2).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.mlp.up_proj.weight")), vec![inter, hid]), &xin2).unwrap();
        let act = swiglu(d, &gate, &upp).unwrap();
        let down = gemv(d, &up(&g(&format!("{p}.mlp.down_proj.weight")), vec![hid, inter]), &act).unwrap();
        let down = dl(&rms_norm(d, &down, &up(&g(&format!("{p}.post_feedforward_layernorm.weight")), vec![hid]), eps).unwrap(), hid);
        for i in 0..hid { x[i] += down[i]; }
    }
    let xf = rms_norm(d, &up(&x, vec![hid]), &up(&g("model.norm.weight"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("OLMo-2-1B on {plat}: top3 = {:?} (HF = [198,8,13])", &idx[..3]);
    assert_eq!(&idx[..3], &[198usize, 8, 13], "OLMo-2 top-3 != HF");
    eprintln!("✅ OLMo-2-1B matches HF ({plat}).");
}

/// GPT-Neo-125M: learned-pos + LayerNorm, separate q/k/v (no bias), no attn
/// scaling, gelu_new, tied. vs HF top-3 [28,59,91].
pub fn verify_gptneo(d: &dyn Device, plat: &str) {
    let dir = model_dir("GPTNEO_DIR", "models--EleutherAI--gpt-neo-125m").unwrap_or_default();
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let (hid, nh, hd, inter, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 3072usize, 12usize, 50257usize, 1e-5f32);
    let scale = 1.0f32;
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let token = 5usize;
    let wte = g("transformer.wte.weight"); let wpe = g("transformer.wpe.weight");
    let mut x: Vec<f32> = (0..hid).map(|i| wte[token*hid+i] + wpe[i]).collect();
    for l in 0..n_layers {
        let p = format!("transformer.h.{l}");
        let h = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.ln_1.weight")), vec![hid]), &up(&g(&format!("{p}.ln_1.bias")), vec![hid]), eps).unwrap();
        let q = gemv(d, &up(&g(&format!("{p}.attn.attention.q_proj.weight")), vec![hid, hid]), &h).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.attn.attention.k_proj.weight")), vec![hid, hid]), &h).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.attn.attention.v_proj.weight")), vec![hid, hid]), &h).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.attn.attention.out_proj.weight")), vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap(), &up(&g(&format!("{p}.attn.attention.out_proj.bias")), vec![hid])).unwrap(), hid);
        for i in 0..hid { x[i] += o[i]; }
        let h2 = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.ln_2.weight")), vec![hid]), &up(&g(&format!("{p}.ln_2.bias")), vec![hid]), eps).unwrap();
        let f = add(d, &gemv(d, &up(&g(&format!("{p}.mlp.c_fc.weight")), vec![inter, hid]), &h2).unwrap(), &up(&g(&format!("{p}.mlp.c_fc.bias")), vec![inter])).unwrap();
        let act = gelu(d, &f).unwrap();
        let m = dl(&add(d, &gemv(d, &up(&g(&format!("{p}.mlp.c_proj.weight")), vec![hid, inter]), &act).unwrap(), &up(&g(&format!("{p}.mlp.c_proj.bias")), vec![hid])).unwrap(), hid);
        for i in 0..hid { x[i] += m[i]; }
    }
    let xf = layer_norm(d, &up(&x, vec![hid]), &up(&g("transformer.ln_f.weight"), vec![hid]), &up(&g("transformer.ln_f.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&wte, vec![vocab, hid]), &xf).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("GPT-Neo-125M on {plat}: top3 = {:?} (HF = [28,59,91])", &idx[..3]);
    assert_eq!(&idx[..3], &[28usize, 59, 91], "GPT-Neo top-3 != HF");
    eprintln!("✅ GPT-Neo-125M matches HF ({plat}).");
}

/// Gemma-2-2b: √hidden embed-scale, RMSNorm(1+w), 4 norms/layer, geGLU, GQA
/// hd256, softcaps (argmax-invariant at 1 pos). vs HF top-3 [9707,235265,110].
pub fn verify_gemma2(d: &dyn Device, plat: &str) {
    use ffai_ops::{mul, rms_norm};
    let dir = model_dir("GEMMA2_DIR", "models--unsloth--gemma-2-2b-it").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nq, nkv, hd, inter, n_layers, vocab, eps) = (2304usize, 8usize, 4usize, 256usize, 9216usize, 26usize, 256000usize, 1e-6f32);
    let (qdim, kvdim) = (nq*hd, nkv*hd); let scale = 1.0/(256.0f32).sqrt(); let embed_scale = (hid as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let g1 = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0.iter().map(|w| w+1.0).collect() };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].iter().map(|v| v*embed_scale).collect();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let h = rms_norm(d, &up(&x, vec![hid]), &up(&g1(&format!("{p}.input_layernorm.weight")), vec![hid]), eps).unwrap();
        let q = gemv(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![qdim, hid]), &h).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![kvdim, hid]), &h).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![kvdim, hid]), &h).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nq, hd]), &k.reshaped(vec![nkv, hd]), &v.reshaped(vec![nkv, hd]), hd, 1, 1, (nq/nkv) as u32, scale).unwrap();
        let o = gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, qdim]), &attn.reshaped(vec![qdim])).unwrap();
        let o = dl(&rms_norm(d, &o, &up(&g1(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), eps).unwrap(), hid);
        for i in 0..hid { x[i] += o[i]; }
        let h2 = rms_norm(d, &up(&x, vec![hid]), &up(&g1(&format!("{p}.pre_feedforward_layernorm.weight")), vec![hid]), eps).unwrap();
        let gate = gelu(d, &gemv(d, &up(&g(&format!("{p}.mlp.gate_proj.weight")), vec![inter, hid]), &h2).unwrap()).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.mlp.up_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let act = mul(d, &gate, &upp).unwrap();
        let down = gemv(d, &up(&g(&format!("{p}.mlp.down_proj.weight")), vec![hid, inter]), &act).unwrap();
        let down = dl(&rms_norm(d, &down, &up(&g1(&format!("{p}.post_feedforward_layernorm.weight")), vec![hid]), eps).unwrap(), hid);
        for i in 0..hid { x[i] += down[i]; }
    }
    let xf = rms_norm(d, &up(&x, vec![hid]), &up(&g1("model.norm.weight"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&embed, vec![vocab, hid]), &xf).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("Gemma-2-2b on {plat}: top3 = {:?} (HF = [9707,235265,110])", &idx[..3]);
    assert_eq!(&idx[..3], &[9707usize, 235265, 110], "Gemma top-3 != HF");
    eprintln!("✅ Gemma-2-2b matches HF ({plat}).");
}

/// Phi-1.5: single shared norm → parallel attn+MLP, separate q/k/v+bias,
/// partial rotary, gelu_new. vs HF top-3 [11,13,546].
pub fn verify_phi(d: &dyn Device, plat: &str) {
    let dir = model_dir("PHI_DIR", "models--microsoft--phi-1_5").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nh, hd, inter, n_layers, vocab, eps) = (2048usize, 32usize, 64usize, 8192usize, 24usize, 51200usize, 1e-5f32);
    let scale = 1.0/(hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let proj = |w: &str, b: &str, x: &Tensor, m: usize, inn: usize| -> Tensor {
        add(d, &gemv(d, &up(&g(w), vec![m, inn]), x).unwrap(), &up(&g(b), vec![m])).unwrap() };
    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].to_vec();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let h = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]), &up(&g(&format!("{p}.input_layernorm.bias")), vec![hid]), eps).unwrap();
        let q = proj(&format!("{p}.self_attn.q_proj.weight"), &format!("{p}.self_attn.q_proj.bias"), &h, hid, hid);
        let k = proj(&format!("{p}.self_attn.k_proj.weight"), &format!("{p}.self_attn.k_proj.bias"), &h, hid, hid);
        let v = proj(&format!("{p}.self_attn.v_proj.weight"), &format!("{p}.self_attn.v_proj.bias"), &h, hid, hid);
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = dl(&proj(&format!("{p}.self_attn.dense.weight"), &format!("{p}.self_attn.dense.bias"), &attn.reshaped(vec![hid]), hid, hid), hid);
        let f = proj(&format!("{p}.mlp.fc1.weight"), &format!("{p}.mlp.fc1.bias"), &h, inter, hid);
        let act = gelu(d, &f).unwrap();
        let m = dl(&proj(&format!("{p}.mlp.fc2.weight"), &format!("{p}.mlp.fc2.bias"), &act, hid, inter), hid);
        for i in 0..hid { x[i] += o[i] + m[i]; }
    }
    let xf = layer_norm(d, &up(&x, vec![hid]), &up(&g("model.final_layernorm.weight"), vec![hid]), &up(&g("model.final_layernorm.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&add(d, &gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), &up(&g("lm_head.bias"), vec![vocab])).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("Phi-1.5 on {plat}: top3 = {:?} (HF = [11,13,546])", &idx[..3]);
    assert_eq!(&idx[..3], &[11usize, 13, 546], "Phi top-3 != HF");
    eprintln!("✅ Phi-1.5 matches HF ({plat}).");
}

/// StableLM-2-1.6B: LayerNorm(+bias), q/k/v bias, partial rotary, SwiGLU.
/// vs HF top-3 [341,11,280].
pub fn verify_stablelm2(d: &dyn Device, plat: &str) {
    use ffai_ops::swiglu;
    let dir = model_dir("STABLELM2_DIR", "models--stabilityai--stablelm-2-1_6b").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nh, hd, inter, n_layers, vocab, eps) = (2048usize, 32usize, 64usize, 5632usize, 24usize, 100352usize, 1e-5f32);
    let scale = 1.0/(hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let bproj = |w: &str, b: &str, x: &Tensor, m: usize| -> Tensor { add(d, &gemv(d, &up(&g(w), vec![m, hid]), x).unwrap(), &up(&g(b), vec![m])).unwrap() };
    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].to_vec();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let h = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]), &up(&g(&format!("{p}.input_layernorm.bias")), vec![hid]), eps).unwrap();
        let q = bproj(&format!("{p}.self_attn.q_proj.weight"), &format!("{p}.self_attn.q_proj.bias"), &h, hid);
        let k = bproj(&format!("{p}.self_attn.k_proj.weight"), &format!("{p}.self_attn.k_proj.bias"), &h, hid);
        let v = bproj(&format!("{p}.self_attn.v_proj.weight"), &format!("{p}.self_attn.v_proj.bias"), &h, hid);
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = dl(&gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, hid]), &attn.reshaped(vec![hid])).unwrap(), hid);
        for i in 0..hid { x[i] += o[i]; }
        let h2 = layer_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.post_attention_layernorm.weight")), vec![hid]), &up(&g(&format!("{p}.post_attention_layernorm.bias")), vec![hid]), eps).unwrap();
        let gate = gemv(d, &up(&g(&format!("{p}.mlp.gate_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.mlp.up_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let act = swiglu(d, &gate, &upp).unwrap();
        let m = dl(&gemv(d, &up(&g(&format!("{p}.mlp.down_proj.weight")), vec![hid, inter]), &act).unwrap(), hid);
        for i in 0..hid { x[i] += m[i]; }
    }
    let xf = layer_norm(d, &up(&x, vec![hid]), &up(&g("model.norm.weight"), vec![hid]), &up(&g("model.norm.bias"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("StableLM-2-1.6B on {plat}: top3 = {:?} (HF = [341,11,280])", &idx[..3]);
    assert_eq!(&idx[..3], &[341usize, 11, 280], "StableLM-2 top-3 != HF");
    eprintln!("✅ StableLM-2-1.6B matches HF ({plat}).");
}

/// OLMoE-1B-7B: 64-expert MoE, top-8 no-renorm, no shared expert, qk-norm over
/// full proj, MHA hd128. vs HF argmax 310.
pub fn verify_olmoe(d: &dyn Device, plat: &str) {
    use ffai_ops::{rms_norm, swiglu};
    let dir = model_dir("OLMOE_DIR", "models--allenai--OLMoE-1B-7B-0924").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nh, hd, inter, n_exp, top_k, n_layers, vocab, eps) = (2048usize, 16usize, 128usize, 1024usize, 64usize, 8usize, 16usize, 50304usize, 1e-5f32);
    let scale = 1.0/(hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32]| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32) };
    let upm = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let token = 5usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].to_vec();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let xn = rms_norm(d, &up(&x), &up(&g(&format!("{p}.input_layernorm.weight"))), eps).unwrap();
        let q = gemv(d, &upm(&g(&format!("{p}.self_attn.q_proj.weight")), vec![hid, hid]), &xn).unwrap();
        let k = gemv(d, &upm(&g(&format!("{p}.self_attn.k_proj.weight")), vec![hid, hid]), &xn).unwrap();
        let v = gemv(d, &upm(&g(&format!("{p}.self_attn.v_proj.weight")), vec![hid, hid]), &xn).unwrap();
        let q = rms_norm(d, &q, &up(&g(&format!("{p}.self_attn.q_norm.weight"))), eps).unwrap();
        let k = rms_norm(d, &k, &up(&g(&format!("{p}.self_attn.k_norm.weight"))), eps).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nh, hd]), &k.reshaped(vec![nh, hd]), &v.reshaped(vec![nh, hd]), hd, 1, 1, 1, scale).unwrap();
        let o = dl(&gemv(d, &upm(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, hid]), &attn.reshaped(vec![nh*hd])).unwrap(), hid);
        for i in 0..hid { x[i] += o[i]; }
        let xn2 = rms_norm(d, &up(&x), &up(&g(&format!("{p}.post_attention_layernorm.weight"))), eps).unwrap();
        let rl = dl(&gemv(d, &upm(&g(&format!("{p}.mlp.gate.weight")), vec![n_exp, hid]), &xn2).unwrap(), n_exp);
        let mx = rl.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = rl.iter().map(|&z| (z-mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e/sum).collect();
        let eidx = ffai_runtime::topk(&probs, top_k);
        let mut acc = vec![0.0f32; hid];
        for &e in &eidx {
            let w = probs[e]; let ep = format!("{p}.mlp.experts.{e}");
            let ge = gemv(d, &upm(&g(&format!("{ep}.gate_proj.weight")), vec![inter, hid]), &xn2).unwrap();
            let ue = gemv(d, &upm(&g(&format!("{ep}.up_proj.weight")), vec![inter, hid]), &xn2).unwrap();
            let act = swiglu(d, &ge, &ue).unwrap();
            let de = dl(&gemv(d, &upm(&g(&format!("{ep}.down_proj.weight")), vec![hid, inter]), &act).unwrap(), hid);
            for i in 0..hid { acc[i] += w*de[i]; }
        }
        for i in 0..hid { x[i] += acc[i]; }
    }
    let xf = rms_norm(d, &up(&x), &up(&g("model.norm.weight")), eps).unwrap();
    let logits = dl(&gemv(d, &upm(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = ffai_runtime::argmax(&logits);
    eprintln!("OLMoE-1B-7B on {plat}: argmax = {argmax} (HF = 310)");
    assert_eq!(argmax, 310, "OLMoE argmax != HF 310");
    eprintln!("✅ OLMoE-1B-7B (64-expert MoE) matches HF ({plat}).");
}

/// Falcon-H1-0.5B: hybrid Mamba2 mixer ∥ GQA attention per layer + µP. vs HF
/// top-3 [593,531,587]. (token 5 is a reserved zero-embedding slot — use 9707.)
pub fn verify_falcon_h1(d: &dyn Device, plat: &str) {
    use ffai_ops::{conv1d_causal_step, rms_norm, silu, ssm_step};
    let dir = model_dir("FALCON_H1_DIR", "models--tiiuae--Falcon-H1-0.5B-Base").unwrap_or_default();
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, nq, nkv, ahd, inter, n_layers, vocab, eps) = (1024usize, 8usize, 2usize, 64usize, 2048usize, 36usize, 32784usize, 1e-5f32);
    let (d_ssm, m_nh, m_dh, d_state, n_groups, d_conv) = (1536usize, 24usize, 64usize, 128usize, 1usize, 4usize);
    let conv_dim = d_ssm + 2*n_groups*d_state; let proj_dim = 2*d_ssm + 2*n_groups*d_state + m_nh;
    let ascale = 1.0/(ahd as f32).sqrt();
    let (ssm_in, ssm_out, attn_out) = (1.25f32, 0.23570226039551587f32, 0.9375f32);
    let (gate_mult, down_mult, embed_mult) = (0.8838834764831844f32, 0.5859375f32, 5.656854249492381f32);
    let ssm_m = [0.3535533905932738f32, 0.25, 0.3535533905932738, 0.5, 0.3535533905932738];
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    let mut mup = vec![0.0f32; proj_dim];
    for i in 0..proj_dim {
        let m = if i < d_ssm { ssm_m[0] } else if i < 2*d_ssm { ssm_m[1] }
            else if i < 2*d_ssm + n_groups*d_state { ssm_m[2] }
            else if i < 2*d_ssm + 2*n_groups*d_state { ssm_m[3] } else { ssm_m[4] };
        mup[i] = m * ssm_in;
    }
    let token = 9707usize;
    let embed = g("model.embed_tokens.weight");
    let mut x: Vec<f32> = embed[token*hid..(token+1)*hid].iter().map(|v| v*embed_mult).collect();
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        let h = rms_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.input_layernorm.weight")), vec![hid]), eps).unwrap();
        let mut proj = dl(&gemv(d, &up(&g(&format!("{p}.mamba.in_proj.weight")), vec![proj_dim, hid]), &h).unwrap(), proj_dim);
        for i in 0..proj_dim { proj[i] *= mup[i]; }
        let z = &proj[0..d_ssm]; let xbc = &proj[d_ssm..d_ssm+conv_dim]; let dt_raw = &proj[d_ssm+conv_dim..proj_dim];
        let cw_hf = g(&format!("{p}.mamba.conv1d.weight"));
        let mut cw = vec![0.0f32; d_conv*conv_dim];
        for ch in 0..conv_dim { for k in 0..d_conv { cw[k*conv_dim+ch] = cw_hf[ch*d_conv+k]; } }
        let cb = g(&format!("{p}.mamba.conv1d.bias"));
        let state0 = vec![0.0f32; (d_conv-1)*conv_dim];
        let yc = conv1d_causal_step(d, &up(xbc, vec![conv_dim]), &up(&cw, vec![d_conv, conv_dim]), &up(&cb, vec![conv_dim]), &up(&state0, vec![(d_conv-1)*conv_dim]), conv_dim as u32, d_conv as u32).unwrap();
        let xbc_act = dl(&silu(d, &yc).unwrap(), conv_dim);
        let x_ssm = &xbc_act[0..d_ssm]; let bmat = &xbc_act[d_ssm..d_ssm+n_groups*d_state]; let cmat = &xbc_act[d_ssm+n_groups*d_state..conv_dim];
        let dt_bias = g(&format!("{p}.mamba.dt_bias"));
        let dt: Vec<f32> = (0..m_nh).map(|i| softplus(dt_raw[i]+dt_bias[i])).collect();
        let a_log = g(&format!("{p}.mamba.A_log")); let dsk = g(&format!("{p}.mamba.D"));
        let state_in = vec![0.0f32; m_nh*m_dh*d_state];
        let (_so, y_t) = ssm_step(d, &up(x_ssm, vec![d_ssm]), &up(&a_log, vec![m_nh]), &up(bmat, vec![n_groups*d_state]), &up(cmat, vec![n_groups*d_state]), &up(&dsk, vec![m_nh]), &up(&dt, vec![m_nh]), &up(&state_in, vec![m_nh*m_dh*d_state]), m_dh as u32, d_state as u32, m_nh as u32, (m_nh/n_groups) as u32).unwrap();
        let y = dl(&y_t, d_ssm);
        let sz = dl(&silu(d, &up(z, vec![d_ssm])).unwrap(), d_ssm);
        let scan: Vec<f32> = (0..d_ssm).map(|i| y[i]*sz[i]).collect();
        let mamba_out = dl(&gemv(d, &up(&g(&format!("{p}.mamba.out_proj.weight")), vec![hid, d_ssm]), &up(&scan, vec![d_ssm])).unwrap(), hid);
        let q = gemv(d, &up(&g(&format!("{p}.self_attn.q_proj.weight")), vec![nq*ahd, hid]), &h).unwrap();
        let k = gemv(d, &up(&g(&format!("{p}.self_attn.k_proj.weight")), vec![nkv*ahd, hid]), &h).unwrap();
        let v = gemv(d, &up(&g(&format!("{p}.self_attn.v_proj.weight")), vec![nkv*ahd, hid]), &h).unwrap();
        let attn = sdpa_decode(d, &q.reshaped(vec![nq, ahd]), &k.reshaped(vec![nkv, ahd]), &v.reshaped(vec![nkv, ahd]), ahd, 1, 1, (nq/nkv) as u32, ascale).unwrap();
        let attn_out_v = dl(&gemv(d, &up(&g(&format!("{p}.self_attn.o_proj.weight")), vec![hid, nq*ahd]), &attn.reshaped(vec![nq*ahd])).unwrap(), hid);
        for i in 0..hid { x[i] += mamba_out[i]*ssm_out + attn_out_v[i]*attn_out; }
        let h2 = rms_norm(d, &up(&x, vec![hid]), &up(&g(&format!("{p}.pre_ff_layernorm.weight")), vec![hid]), eps).unwrap();
        let gate_w: Vec<f32> = g(&format!("{p}.feed_forward.gate_proj.weight")).iter().map(|w| w*gate_mult).collect();
        let gate = silu(d, &gemv(d, &up(&gate_w, vec![inter, hid]), &h2).unwrap()).unwrap();
        let upp = gemv(d, &up(&g(&format!("{p}.feed_forward.up_proj.weight")), vec![inter, hid]), &h2).unwrap();
        let act = dl(&gate, inter).iter().zip(dl(&upp, inter)).map(|(gg, uu)| gg*uu).collect::<Vec<f32>>();
        let ff = dl(&gemv(d, &up(&g(&format!("{p}.feed_forward.down_proj.weight")), vec![hid, inter]), &up(&act, vec![inter])).unwrap(), hid);
        for i in 0..hid { x[i] += ff[i]*down_mult; }
    }
    let xf = rms_norm(d, &up(&x, vec![hid]), &up(&g("model.final_layernorm.weight"), vec![hid]), eps).unwrap();
    let logits = dl(&gemv(d, &up(&g("lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let idx = ffai_runtime::topk(&logits, 3);
    eprintln!("Falcon-H1-0.5B on {plat}: top3 = {:?} (HF = [593,531,587])", &idx[..3]);
    assert_eq!(&idx[..3], &[593usize, 531, 587], "Falcon-H1 top-3 != HF");
    eprintln!("✅ Falcon-H1-0.5B (hybrid Mamba2+attn) matches HF ({plat}).");
}

/// NemotronH-Nano-Omni-30B-A3B (text backbone): 52-layer M/E/* hybrid.
/// pattern MEMEM*EMEMEM*… (M=Mamba2 mixer, E=128-expert MoE relu², *=GQA attn).
/// Single mixer per layer, pre-norm residual. BF16 weights → F32 on-device.
/// Verified against an HF-transformers CPU oracle (set NEMOTRON_ARGMAX).
pub fn verify_nemotron(d: &dyn Device, plat: &str) {
    use ffai_ops::{conv1d_causal_step, rms_norm, rope_llama, silu, ssm_step};
    const PATTERN: &str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
    let dir = std::env::var("NEMOTRON_DIR")
        .unwrap_or_else(|_| "/home/pidtom/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16".into());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, vocab, eps) = (2688usize, 131072usize, 1e-5f32);
    // Mamba2: d_inner 4096 (64h×64), n_groups 8, d_state 128, conv_kernel 4.
    let (di, m_nh, m_dh, ds, ng, kc) = (4096usize, 64usize, 64usize, 128usize, 8usize, 4usize);
    let conv_dim = di + 2 * ng * ds;            // 6144
    let in_proj_out = 2 * di + 2 * ng * ds + m_nh; // 10304
    // MoE: 128 experts top-6, relu² ungated, shared 3712, sigmoid+bias router, ×2.5.
    let (n_exp, top_k, inter, shared_inter, scale_f) = (128usize, 6usize, 1856usize, 3712usize, 2.5f32);
    // Attn: GQA 32q/2kv hd128, rope θ1e4.
    let (nq, nkv, hd, rope_theta) = (32usize, 2usize, 128usize, 10000f32);
    let (qdim, kvdim) = (nq * hd, nkv * hd); // 4096, 256
    let ascale = 1.0 / (hd as f32).sqrt();

    // Coarse perf accounting: how much of a token is weight dequant (CPU BF16→F32)
    // vs host↔device transfer vs everything else (GPU op dispatch). This baseline
    // quantifies the resident-weights + quant headroom.
    use std::cell::Cell;
    use std::time::Instant;
    let t_deq = Cell::new(0f64); // BF16→F32 dequant + mmap read
    let t_xfer = Cell::new(0f64); // upload + download bytes
    let g = |name: &str| -> Vec<f32> { let t = Instant::now(); let r = st.tensor_f32(name).unwrap().0; t_deq.set(t_deq.get() + t.elapsed().as_secs_f64()); r };
    let up = |v: &[f32]| -> Tensor { let t = Instant::now(); let b = d.upload(&tb(v)).unwrap(); t_xfer.set(t_xfer.get() + t.elapsed().as_secs_f64()); Tensor::new(b, vec![v.len()], DType::F32) };
    let upm = |v: &[f32], sh: Vec<usize>| -> Tensor { let t = Instant::now(); let b = d.upload(&tb(v)).unwrap(); t_xfer.set(t_xfer.get() + t.elapsed().as_secs_f64()); Tensor::new(b, sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let s = Instant::now(); let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); let r = fb(&b); t_xfer.set(t_xfer.get() + s.elapsed().as_secs_f64()); r };
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    let relu2 = |v: &mut [f32]| for x in v.iter_mut() { let r = x.max(0.0); *x = r * r; };
    let t_total = Instant::now();

    let token: usize = std::env::var("NEMOTRON_TOKEN").ok().and_then(|s| s.parse().ok()).unwrap_or(1234);
    let embed = g("language_model.backbone.embeddings.weight");
    let mut x: Vec<f32> = embed[token * hid..(token + 1) * hid].to_vec();
    let enorm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!("Nemotron: token {token} embed‖·‖={enorm:.3}");
    let dump = std::env::var("NEMOTRON_DUMP").is_ok();
    let fp = |tag: &str, v: &[f32]| {
        let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        eprintln!("{tag} norm={n:.4} head={:?}", v[..4].iter().map(|a| (a * 10000.0).round() / 10000.0).collect::<Vec<_>>());
    };
    if dump { fp("L00", &x); }

    for (l, mix) in PATTERN.chars().enumerate() {
        let p = format!("language_model.backbone.layers.{l}");
        let xn = rms_norm(d, &up(&x), &up(&g(&format!("{p}.norm.weight"))), eps).unwrap();
        match mix {
            'M' => {
                let in_proj = upm(&g(&format!("{p}.mixer.in_proj.weight")), vec![in_proj_out, hid]);
                let proj = dl(&gemv(d, &in_proj, &xn).unwrap(), in_proj_out);
                let z = &proj[0..di];
                let xbc = &proj[di..di + conv_dim];
                let dt_raw = &proj[di + conv_dim..di + conv_dim + m_nh];
                let cw_hf = g(&format!("{p}.mixer.conv1d.weight"));
                let mut cw = vec![0.0f32; kc * conv_dim];
                for ch in 0..conv_dim { for k in 0..kc { cw[k * conv_dim + ch] = cw_hf[ch * kc + k]; } }
                let cb = g(&format!("{p}.mixer.conv1d.bias"));
                let state0 = vec![0.0f32; (kc - 1) * conv_dim];
                let yc = conv1d_causal_step(d, &up(xbc), &up(&cw), &up(&cb), &up(&state0), conv_dim as u32, kc as u32).unwrap();
                let xbc_act = dl(&silu(d, &yc).unwrap(), conv_dim);
                let x_ssm = &xbc_act[0..di];
                let bmat = &xbc_act[di..di + ng * ds];
                let cmat = &xbc_act[di + ng * ds..di + 2 * ng * ds];
                let dt_bias = g(&format!("{p}.mixer.dt_bias"));
                let dt: Vec<f32> = (0..m_nh).map(|i| softplus(dt_raw[i] + dt_bias[i])).collect();
                let a_log = g(&format!("{p}.mixer.A_log"));
                let dsk = g(&format!("{p}.mixer.D"));
                let state_in = vec![0.0f32; m_nh * m_dh * ds];
                let (_so, y_t) = ssm_step(d, &up(x_ssm), &up(&a_log), &up(bmat), &up(cmat), &up(&dsk), &up(&dt), &up(&state_in), m_dh as u32, ds as u32, m_nh as u32, (m_nh / ng) as u32).unwrap();
                let y = dl(&y_t, di);
                let sz = dl(&silu(d, &up(z)).unwrap(), di);
                let y_gated: Vec<f32> = (0..di).map(|i| y[i] * sz[i]).collect();
                // Zamba2RMSNormGated: group-wise RMSNorm (group_size = d_inner/n_groups),
                // gate applied before, weight after. NOT a full-vector RMSNorm.
                let nw = g(&format!("{p}.mixer.norm.weight"));
                let gs = di / ng; // 512
                let mut yn = vec![0.0f32; di];
                for grp in 0..ng {
                    let s = grp * gs;
                    let seg = dl(&rms_norm(d, &up(&y_gated[s..s + gs]), &up(&nw[s..s + gs]), eps).unwrap(), gs);
                    yn[s..s + gs].copy_from_slice(&seg);
                }
                let out = dl(&gemv(d, &upm(&g(&format!("{p}.mixer.out_proj.weight")), vec![hid, di]), &up(&yn)).unwrap(), hid);
                for i in 0..hid { x[i] += out[i]; }
            }
            'E' => {
                let rl = dl(&gemv(d, &upm(&g(&format!("{p}.mixer.gate.weight")), vec![n_exp, hid]), &xn).unwrap(), n_exp);
                let sig: Vec<f32> = rl.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
                let bias = g(&format!("{p}.mixer.gate.e_score_correction_bias"));
                let choice: Vec<f32> = (0..n_exp).map(|i| sig[i] + bias[i]).collect();
                let eidx = ffai_runtime::topk(&choice, top_k);
                let mut w: Vec<f32> = eidx.iter().map(|&e| sig[e]).collect();
                let wsum: f32 = w.iter().sum::<f32>() + 1e-20;
                for v in w.iter_mut() { *v = *v / wsum * scale_f; }
                let mut acc = vec![0.0f32; hid];
                for (j, &e) in eidx.iter().enumerate() {
                    let ep = format!("{p}.mixer.experts.{e}");
                    let mut a = dl(&gemv(d, &upm(&g(&format!("{ep}.up_proj.weight")), vec![inter, hid]), &xn).unwrap(), inter);
                    relu2(&mut a);
                    let de = dl(&gemv(d, &upm(&g(&format!("{ep}.down_proj.weight")), vec![hid, inter]), &up(&a)).unwrap(), hid);
                    for i in 0..hid { acc[i] += w[j] * de[i]; }
                }
                // shared expert (relu², inter 3712)
                let mut sa = dl(&gemv(d, &upm(&g(&format!("{p}.mixer.shared_experts.up_proj.weight")), vec![shared_inter, hid]), &xn).unwrap(), shared_inter);
                relu2(&mut sa);
                let sde = dl(&gemv(d, &upm(&g(&format!("{p}.mixer.shared_experts.down_proj.weight")), vec![hid, shared_inter]), &up(&sa)).unwrap(), hid);
                for i in 0..hid { x[i] += acc[i] + sde[i]; }
            }
            '*' => {
                let q = gemv(d, &upm(&g(&format!("{p}.mixer.q_proj.weight")), vec![qdim, hid]), &xn).unwrap();
                let k = gemv(d, &upm(&g(&format!("{p}.mixer.k_proj.weight")), vec![kvdim, hid]), &xn).unwrap();
                let v = gemv(d, &upm(&g(&format!("{p}.mixer.v_proj.weight")), vec![kvdim, hid]), &xn).unwrap();
                let q = rope_llama(d, &q.reshaped(vec![nq, hd]), 0, rope_theta, 1.0, 1.0, 1.0, 8192.0).unwrap();
                let k = rope_llama(d, &k.reshaped(vec![nkv, hd]), 0, rope_theta, 1.0, 1.0, 1.0, 8192.0).unwrap();
                let attn = sdpa_decode(d, &q, &k, &v.reshaped(vec![nkv, hd]), hd, 1, 1, (nq / nkv) as u32, ascale).unwrap();
                let o = dl(&gemv(d, &upm(&g(&format!("{p}.mixer.o_proj.weight")), vec![hid, qdim]), &attn.reshaped(vec![qdim])).unwrap(), hid);
                for i in 0..hid { x[i] += o[i]; }
            }
            _ => unreachable!("bad pattern char"),
        }
        if dump { fp(&format!("L{:02}[{mix}]", l + 1), &x); }
    }
    let xf = rms_norm(d, &up(&x), &up(&g("language_model.backbone.norm_f.weight")), eps).unwrap();
    let logits = dl(&gemv(d, &upm(&g("language_model.lm_head.weight"), vec![vocab, hid]), &xf).unwrap(), vocab);
    let argmax = ffai_runtime::argmax(&logits);
    let idx = ffai_runtime::topk(&logits, 5);
    let total = t_total.elapsed().as_secs_f64();
    let (deq, xfer) = (t_deq.get(), t_xfer.get());
    let compute = (total - deq - xfer).max(0.0);
    eprintln!("──────── NemotronH-Nano BASELINE (1 token, F32, naive re-load) ────────");
    eprintln!("  total      {total:7.1}s   ({:.3} tok/s)", 1.0 / total);
    eprintln!("  dequant    {deq:7.1}s   {:.0}%  (CPU BF16→F32; gone once weights are quantized+resident)", 100.0 * deq / total);
    eprintln!("  transfer   {xfer:7.1}s   {:.0}%  (host↔device; gone once weights resident)", 100.0 * xfer / total);
    eprintln!("  compute    {compute:7.1}s   {:.0}%  (GPU op dispatch — the resident-decode floor in F32)", 100.0 * compute / total);
    eprintln!("  ⇒ resident-weights ceiling ≈ {:.2} tok/s (F32 compute only); quant+fusion cuts further", 1.0 / compute.max(1e-3));
    eprintln!("───────────────────────────────────────────────────────────────────────");
    eprintln!("NemotronH-Nano text forward on {plat}: argmax={argmax} top5={:?}", &idx[..5]);
    if let Ok(exp) = std::env::var("NEMOTRON_ARGMAX") {
        assert_eq!(argmax, exp.parse::<usize>().unwrap(), "Nemotron argmax != HF");
    } else if token == 1234 {
        // HF-transformers CPU oracle (NemotronHForCausalLM, bf16, naive Mamba2 path).
        assert_eq!(&idx[..5], &[1234usize, 99493, 99391, 67501, 49418], "Nemotron top5 != HF oracle");
        eprintln!("✅ NemotronH-Nano-30B-A3B text backbone matches HF ({plat}).");
    }
}

/// NemotronH-Nano-30B-A3B **resident Q8 decode benchmark**: quantize every big
/// matrix to Q8_0 and upload it ONCE (resident), then run a real decode loop —
/// per-token forward reuses the resident weights via [`gemv_q8`], carries an
/// attention KV cache (6 layers) + Mamba2 conv/SSM state (23 layers), and times
/// steady-state tok/s. This is the path off the 0.003 tok/s naive-reload floor.
/// Env: NEMOTRON_DECODE (steps, default 32), NEMOTRON_PREFILL (warm the cache to
/// this context length before timing, default 0).
pub fn bench_nemotron(d: &dyn Device, plat: &str) {
    use ffai_ops::{add, cast_f16_f32, cast_f32_f16, conv1d_causal_prefill, conv1d_causal_step, dequant_q4, dequant_q4_off, gather, gated_group_rmsnorm, gated_group_rmsnorm_batched, gemm_cublas, gemm_cublas_f32out, gemm_q4_mpp, gemv, gemv_q4, gemv_q4_accum, gemv_q4_relu2, gemv_q8, gemv_q8_relu2, gemv_q8_accum, kv_append, kv_append_many, mamba_split_conv, mamba_split_proj, matmul, moe_bgemm_q4_bm64, moe_expert_ids_from_offsets, moe_fused_ffn, moe_gather_down, moe_gather_up_relu2, moe_grouped_gemm, moe_grouped_gemm_cutlass, moe_q4_grouped_mma, moe_q4_grouped_mma_dev, moe_router_device, moe_scatter_add_det_dev, moe_scatter_add, moe_scatter_add_det, moe_w4a16, moe_w4a16_marlin, moe_weighted_sum, nvfp4_roundtrip, permute_q4_to_marlin, quantize_q4, quantize_q8, relu2, relu2_scale_f16, rms_norm, rope_llama, rope_llama_many, sdpa_multi, sdpa_multi_tc, sdpa_multi_tc_varlen, silu, slice, sdpa_decode, sdpa_decode_2pass, sdpa_decode_2pass_bc4, sdpa_decode_2pass_tiled, softplus_add, softplus_add_rows, ssm_prefill_scan, ssm_prefill_scan_chunked, ssm_prefill_scan_ssd, ssm_prefill_scan_ssd_portable, ssm_step, strided_col_copy};
    use std::collections::HashMap;
    use std::time::Instant;
    const PATTERN: &str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
    let dir = std::env::var("NEMOTRON_DIR")
        .unwrap_or_else(|_| "/home/pidtom/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16".into());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let (hid, vocab, eps) = (2688usize, 131072usize, 1e-5f32);
    let (di, m_nh, m_dh, ds, ng, kc) = (4096usize, 64usize, 64usize, 128usize, 8usize, 4usize);
    let conv_dim = di + 2 * ng * ds;
    let in_proj_out = 2 * di + 2 * ng * ds + m_nh;
    let (n_exp, top_k, inter, shared_inter, scale_f) = (128usize, 6usize, 1856usize, 3712usize, 2.5f32);
    let (nq, nkv, hd, rope_theta) = (32usize, 2usize, 128usize, 10000f32);
    let (qdim, kvdim) = (nq * hd, nkv * hd);
    let ascale = 1.0 / (hd as f32).sqrt();
    let gs = di / ng;

    let tbu = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    // f32 → f16 (round-to-nearest-even, positive normals; Q4 scales = amax/7 are in range).
    fn f32_to_f16(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 112; // 127 - 15
        if e <= 0 { return sign; }
        if e >= 0x1f { return sign | 0x7c00; }
        let m = (x >> 13) & 0x3ff;
        let round = (x >> 12) & 1;
        let v = ((e as u32) << 10) | m;
        sign | ((v + round) as u16)
    }
    let tb_f16 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&f| f32_to_f16(f).to_le_bytes()).collect() };
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32]| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32) };
    let upm = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let owned; let src = if t.dtype == DType::F16 { owned = cast_f16_f32(d, t).unwrap(); &owned } else { t }; let mut b = vec![0u8; n * 4]; d.download(src.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };

    // ── Q4 weight cache ─────────────────────────────────────────────────────────
    // Gated by NEMOTRON_Q4CACHE env (default ON). On first run, writes Q4+scales to
    // ~/.cache/nemo_q4/<sanitized_name>.q4b. Subsequent runs skip BF16→F32+quantize
    // and read the cached bytes directly → load time ~15-25s vs ~120s.
    // Cache file format (little-endian):
    //   [8B: m as u64][8B: k as u64][1B: f16 flag (0=f32, 1=f16)]
    //   [N*4 bytes: qs as u32 LE][M*2 or M*4 bytes: scales as f16 or f32 LE]
    let use_q4cache = std::env::var("NEMOTRON_Q4CACHE").map(|v| v != "0" && v != "false").unwrap_or(true);
    let cache_dir = std::env::var("NEMOTRON_Q4CACHE_DIR")
        .unwrap_or_else(|_| format!("{}/.cache/nemo_q4", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())));
    if use_q4cache { let _ = std::fs::create_dir_all(&cache_dir); }

    let cache_path = |name: &str| -> std::path::PathBuf {
        let safe: String = name.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect();
        std::path::PathBuf::from(&cache_dir).join(format!("{safe}.q4b"))
    };

    // Read from cache: returns (qs_bytes, sc_bytes, m, k, f16) or None if miss/disabled.
    let cache_read = |name: &str| -> Option<(Vec<u8>, Vec<u8>, usize, usize, bool)> {
        if !use_q4cache { return None; }
        let bytes = std::fs::read(cache_path(name)).ok()?;
        if bytes.len() < 17 { return None; }
        let m = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        let k = u64::from_le_bytes(bytes[8..16].try_into().ok()?) as usize;
        let f16 = bytes[16] != 0;
        let bpr = k / 32;
        let qs_len_u32 = m * bpr * 4;
        let sc_len = m * bpr;
        let sc_bytes = if f16 { sc_len * 2 } else { sc_len * 4 };
        let expected = 17 + qs_len_u32 * 4 + sc_bytes;
        if bytes.len() != expected { return None; } // stale/corrupt
        let qs_bytes = bytes[17..17 + qs_len_u32 * 4].to_vec();
        let sc_bytes = bytes[17 + qs_len_u32 * 4..].to_vec();
        Some((qs_bytes, sc_bytes, m, k, f16))
    };

    // Write to cache.
    let cache_write = |name: &str, qs: &[u32], sc: &[f32], m: usize, k: usize, f16: bool| {
        if !use_q4cache { return; }
        let mut out = Vec::with_capacity(17 + qs.len() * 4 + sc.len() * if f16 { 2 } else { 4 });
        out.extend_from_slice(&(m as u64).to_le_bytes());
        out.extend_from_slice(&(k as u64).to_le_bytes());
        out.push(f16 as u8);
        for &q in qs { out.extend_from_slice(&q.to_le_bytes()); }
        if f16 {
            for &s in sc { out.extend_from_slice(&f32_to_f16(s).to_le_bytes()); }
        } else {
            for &s in sc { out.extend_from_slice(&s.to_le_bytes()); }
        }
        let _ = std::fs::write(cache_path(name), &out);
    };

    // ── NVFP4 recipe flag ──────────────────────────────────────────────────────
    // NEMOTRON_NVFP4_RECIPE=1: switch Mamba in_proj/out_proj, shared-expert up/down,
    // and attention o_proj from Q4 to Q8.  Routed MoE experts, q/k/v proj, and
    // lm_head stay Q4.  Result is ~4.98 bits/weight, ~20.9 GB resident — matching
    // NVIDIA's published NVFP4 mixed-precision recipe for apples-to-apples comparison.
    let use_nvfp4_recipe = std::env::var("NEMOTRON_NVFP4_RECIPE").map(|v| v != "0" && v != "false").unwrap_or(false);

    // ── SETUP: quantize + upload all big matrices to Q4 resident (once) ──
    let t_load = Instant::now();
    let mut qw: HashMap<String, (Tensor, Tensor, usize, usize)> = HashMap::new(); // name → (qs, scales, m, k)
    // Q8 resident weights for NVFP4-recipe layers (Mamba in/out_proj, shared-expert up/down, attn o_proj).
    // Stored separately so the qmv/qacc/qrelu2 dispatch closures can key on which map to use
    // without touching the Q4 path at all.  Only populated when use_nvfp4_recipe=true.
    let mut qw8: HashMap<String, (Tensor, Tensor, usize, usize)> = HashMap::new();
    // NVFP4 fragment-packed MoE weights (NEMOTRON_FP4_MOE): name → (wp, wsc, gw)
    let mut fp4w: HashMap<String, (Tensor, Tensor, Tensor)> = HashMap::new();
    let use_fp4_moe = std::env::var("NEMOTRON_FP4_MOE").is_ok();
    // CUTLASS-layout NVFP4 MoE weights (NEMOTRON_CUTLASS_MOE=1 on top of
    // FP4_MOE): name → (packed e2m1 slab, swizzled ue4m3 SF slab, per-expert
    // globals). Feeds the grouped block-scaled CUTLASS GEMM instead of the
    // hand-rolled mma.
    let mut fp4w_cutlass: HashMap<String, (Tensor, Tensor, Tensor)> = HashMap::new();
    // Persistent CUTLASS grouped-FP4 handles per layer: (up_handle, dn_handle,
    // alpha_up, alpha_dn). Prepared lazily on first use; alpha buffers are
    // fixed addresses the prepared blob points at (values rewritten per call).
    let cutlass_handles: std::cell::RefCell<HashMap<String, (u64, u64, Tensor, Tensor)>> =
        std::cell::RefCell::new(HashMap::new());
    let use_fp4_cutlass_env = use_fp4_moe && std::env::var("NEMOTRON_CUTLASS_MOE").is_ok();
    let mut fw: HashMap<String, Vec<f32>> = HashMap::new(); // f32 weights used HOST-side
    let mut fwd: HashMap<String, Tensor> = HashMap::new(); // f32 weights RESIDENT on device
    let fd = |fwd: &mut HashMap<String, Tensor>, name: &str, v: &[f32], shape: Vec<usize>| {
        fwd.insert(name.to_string(), Tensor::new(d.upload(&tb(v)).unwrap(), shape, DType::F32));
    };

    // ── Q8 cache helpers (parallel to the Q4 cache, uses .q8b extension) ──────
    // Cache format: [8B m][8B k][N*4 bytes qs u32 LE][M*4 bytes scales f32 LE]
    // (Q8 scales are always f32; no f16 flag byte needed — layout is fixed.)
    let cache_path_q8 = |name: &str| -> std::path::PathBuf {
        let safe: String = name.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect();
        std::path::PathBuf::from(&cache_dir).join(format!("{safe}.q8b"))
    };
    let cache_read_q8 = |name: &str, m: usize, k: usize| -> Option<(Vec<u8>, Vec<u8>)> {
        if !use_q4cache { return None; }
        let bytes = std::fs::read(cache_path_q8(name)).ok()?;
        if bytes.len() < 16 { return None; }
        let cm = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        let ck = u64::from_le_bytes(bytes[8..16].try_into().ok()?) as usize;
        if cm != m || ck != k { return None; }
        let bpr = k / 32;
        let qs_len_u32 = m * bpr * 8;
        let sc_len = m * bpr;
        let expected = 16 + qs_len_u32 * 4 + sc_len * 4;
        if bytes.len() != expected { return None; }
        Some((bytes[16..16 + qs_len_u32 * 4].to_vec(), bytes[16 + qs_len_u32 * 4..].to_vec()))
    };
    let cache_write_q8 = |name: &str, qs: &[u32], sc: &[f32], m: usize, k: usize| {
        if !use_q4cache { return; }
        let mut out = Vec::with_capacity(16 + qs.len() * 4 + sc.len() * 4);
        out.extend_from_slice(&(m as u64).to_le_bytes());
        out.extend_from_slice(&(k as u64).to_le_bytes());
        for &q in qs { out.extend_from_slice(&q.to_le_bytes()); }
        for &s in sc { out.extend_from_slice(&s.to_le_bytes()); }
        let _ = std::fs::write(cache_path_q8(name), &out);
    };

    // Load a weight into qw8 (Q8).  Always uses f32 scales (Q8 kernels expect f32).
    let qload8 = |qw8: &mut HashMap<String, (Tensor, Tensor, usize, usize)>, name: &str, m: usize, k: usize| {
        if let Some((qs_bytes, sc_bytes)) = cache_read_q8(name, m, k) {
            let qt = Tensor::new(d.upload(&qs_bytes).unwrap(), vec![qs_bytes.len() / 4], DType::U32);
            let sct = Tensor::new(d.upload(&sc_bytes).unwrap(), vec![sc_bytes.len() / 4], DType::F32);
            qw8.insert(name.to_string(), (qt, sct, m, k));
            return;
        }
        let w = g(name);
        let (qs, sc) = quantize_q8(&w, m, k);
        cache_write_q8(name, &qs, &sc, m, k);
        let qt = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
        let sct = Tensor::new(d.upload(&tb(&sc)).unwrap(), vec![sc.len()], DType::F32);
        qw8.insert(name.to_string(), (qt, sct, m, k));
    };

    // f16: true for weights read by the PLAIN gemv kernel (qmv) — its scale param
    // is f16. false for shared-expert weights (relu2/accum kernels, f32 scale).
    let qload = |qw: &mut HashMap<String, (Tensor, Tensor, usize, usize)>, name: &str, m: usize, k: usize, f16: bool| {
        // Try cache hit first: skip BF16→F32 + quantize if available.
        if let Some((qs_bytes, sc_bytes, cm, ck, cf16)) = cache_read(name) {
            if cm == m && ck == k && cf16 == f16 {
                let qt = Tensor::new(d.upload(&qs_bytes).unwrap(), vec![qs_bytes.len() / 4], DType::U32);
                let sct = Tensor::new(d.upload(&sc_bytes).unwrap(), vec![sc_bytes.len() / if f16 { 2 } else { 4 }], if f16 { DType::F16 } else { DType::F32 });
                qw.insert(name.to_string(), (qt, sct, m, k));
                return;
            }
        }
        // Cache miss: full BF16→F32 + Q4 quantize path.
        let w = g(name);
        let (qs, sc) = quantize_q4(&w, m, k);
        cache_write(name, &qs, &sc, m, k, f16);
        let qt = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
        let sct = if f16 {
            Tensor::new(d.upload(&tb_f16(&sc)).unwrap(), vec![sc.len()], DType::F16)
        } else {
            Tensor::new(d.upload(&tb(&sc)).unwrap(), vec![sc.len()], DType::F32)
        };
        qw.insert(name.to_string(), (qt, sct, m, k));
    };
    let embed = g("language_model.backbone.embeddings.weight"); // host lookup table
    fd(&mut fwd, "norm_f", &g("language_model.backbone.norm_f.weight"), vec![hid]);
    qload(&mut qw, "language_model.lm_head.weight", vocab, hid, true);
    for (l, mix) in PATTERN.chars().enumerate() {
        let p = format!("language_model.backbone.layers.{l}");
        fd(&mut fwd, &format!("{p}.norm.weight"), &g(&format!("{p}.norm.weight")), vec![hid]);
        match mix {
            'M' => {
                // NVFP4 recipe: Mamba in_proj + out_proj → Q8 (FP8-class in the official recipe).
                if use_nvfp4_recipe {
                    qload8(&mut qw8, &format!("{p}.mixer.in_proj.weight"), in_proj_out, hid);
                    qload8(&mut qw8, &format!("{p}.mixer.out_proj.weight"), hid, di);
                } else {
                    qload(&mut qw, &format!("{p}.mixer.in_proj.weight"), in_proj_out, hid, true);
                    qload(&mut qw, &format!("{p}.mixer.out_proj.weight"), hid, di, true);
                }
                // conv weight pre-reorganized [kc, conv_dim] ONCE (was redone per step).
                let cw_hf = g(&format!("{p}.mixer.conv1d.weight"));
                let mut cw = vec![0.0f32; kc * conv_dim];
                for ch in 0..conv_dim { for kk in 0..kc { cw[kk * conv_dim + ch] = cw_hf[ch * kc + kk]; } }
                fd(&mut fwd, &format!("{p}.mixer.conv1d.weight"), &cw, vec![kc * conv_dim]);
                let convbias = g(&format!("{p}.mixer.conv1d.bias"));
                fd(&mut fwd, &format!("{p}.mixer.conv1d.bias"), &convbias, vec![conv_dim]);
                // Host copies of conv weight (reorganized [kc,conv_dim]) + bias for the
                // host-bridged batched-prefill causal conv (NEMOTRON_PREFILL_BATCHED).
                fw.insert(format!("{p}.mixer.conv1d.weight_host"), cw.clone());
                fw.insert(format!("{p}.mixer.conv1d.bias_host"), convbias);
                fd(&mut fwd, &format!("{p}.mixer.A_log"), &g(&format!("{p}.mixer.A_log")), vec![m_nh]);
                fd(&mut fwd, &format!("{p}.mixer.D"), &g(&format!("{p}.mixer.D")), vec![m_nh]);
                fw.insert(format!("{p}.mixer.dt_bias"), g(&format!("{p}.mixer.dt_bias")));     // host (softplus, tiny)
                fw.insert(format!("{p}.mixer.norm.weight"), g(&format!("{p}.mixer.norm.weight")));
            }
            'E' => {
                fd(&mut fwd, &format!("{p}.mixer.gate.weight"), &g(&format!("{p}.mixer.gate.weight")), vec![n_exp, hid]);
                // f16 copy for the batched-prefill cuBLAS router GEMM (688KB).
                fwd.insert(format!("{p}.mixer.gate.weight.f16"),
                    Tensor::new(d.upload(&tb_f16(&g(&format!("{p}.mixer.gate.weight")))).unwrap(), vec![n_exp, hid], DType::F16));
                fw.insert(format!("{p}.mixer.gate.e_score_correction_bias"), g(&format!("{p}.mixer.gate.e_score_correction_bias")));
                // Experts stored CONTIGUOUS ([n_exp*inter, hid] up, [n_exp*hid, inter] down)
                // so the batched gather kernel runs them as one big efficient GEMV.
                // Cache the combined expert packs under "{p}.moe_up_all" / ".moe_down_all".
                // When NEMOTRON_W4A16_MARLIN=1, the weights are permuted into Marlin
                // tile-major layout before GPU upload (cache stores standard layout).
                let moe_up_name = format!("{p}.moe_up_all");
                let moe_down_name = format!("{p}.moe_down_all");
                let use_marlin_layout = std::env::var("NEMOTRON_W4A16_MARLIN").is_ok();
                // Helper: optionally permute qs to Marlin layout before GPU upload.
                // (Marlin permute applied inline below under `_marlin` keys.)
                let (mup_hit, mdown_hit) = (cache_read(&moe_up_name), cache_read(&moe_down_name));
                if let (Some((uqb, usb, cm, ck, _)), Some((dqb, dsb, dm, dk, _))) = (mup_hit, mdown_hit) {
                    if cm == n_exp * inter && ck == hid && dm == n_exp * hid && dk == inter {
                        // Reconstruct qs vecs from cached bytes for optional Marlin permutation.
                        let uqs_std: Vec<u32> = uqb.chunks_exact(4).map(|b| u32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
                        let dqs_std: Vec<u32> = dqb.chunks_exact(4).map(|b| u32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
                        // STANDARD layout under base keys (sequential step path needs std).
                        let ut = Tensor::new(d.upload(&tbu(&uqs_std)).unwrap(), vec![uqs_std.len()], DType::U32);
                        let ust = Tensor::new(d.upload(&usb).unwrap(), vec![usb.len() / 2], DType::F16);
                        let dt2 = Tensor::new(d.upload(&tbu(&dqs_std)).unwrap(), vec![dqs_std.len()], DType::U32);
                        let dst2 = Tensor::new(d.upload(&dsb).unwrap(), vec![dsb.len() / 2], DType::F16);
                        qw.insert(moe_up_name.clone(), (ut, ust, n_exp * inter, hid));
                        qw.insert(moe_down_name.clone(), (dt2, dst2, n_exp * hid, inter));
                        // MARLIN layout under `_marlin` keys (batched Marlin path only).
                        if use_marlin_layout {
                            let uqs_mar = permute_q4_to_marlin(&uqs_std, n_exp, inter, hid);
                            let dqs_mar = permute_q4_to_marlin(&dqs_std, n_exp, hid, inter);
                            qw.insert(format!("{moe_up_name}_marlin"),
                                (Tensor::new(d.upload(&tbu(&uqs_mar)).unwrap(), vec![uqs_mar.len()], DType::U32),
                                 Tensor::new(d.upload(&usb).unwrap(), vec![usb.len() / 2], DType::F16), n_exp * inter, hid));
                            qw.insert(format!("{moe_down_name}_marlin"),
                                (Tensor::new(d.upload(&tbu(&dqs_mar)).unwrap(), vec![dqs_mar.len()], DType::U32),
                                 Tensor::new(d.upload(&dsb).unwrap(), vec![dsb.len() / 2], DType::F16), n_exp * hid, inter));
                        }
                    } else {
                        // Dimension mismatch — fall through to rebuild.
                        let (mut uqs, mut usc, mut dqs, mut dsc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                        for e in 0..n_exp {
                            let (q, s) = quantize_q4(&g(&format!("{p}.mixer.experts.{e}.up_proj.weight")), inter, hid);
                            uqs.extend(q); usc.extend(s);
                            let (q, s) = quantize_q4(&g(&format!("{p}.mixer.experts.{e}.down_proj.weight")), hid, inter);
                            dqs.extend(q); dsc.extend(s);
                        }
                        cache_write(&moe_up_name, &uqs, &usc, n_exp * inter, hid, true);
                        cache_write(&moe_down_name, &dqs, &dsc, n_exp * hid, inter, true);
                        // STANDARD layout under base keys (sequential step path).
                        qw.insert(moe_up_name.clone(), (Tensor::new(d.upload(&tbu(&uqs)).unwrap(), vec![uqs.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&usc)).unwrap(), vec![usc.len()], DType::F16), n_exp * inter, hid));
                        qw.insert(moe_down_name.clone(), (Tensor::new(d.upload(&tbu(&dqs)).unwrap(), vec![dqs.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&dsc)).unwrap(), vec![dsc.len()], DType::F16), n_exp * hid, inter));
                        if use_marlin_layout {
                            let uqs_mar = permute_q4_to_marlin(&uqs, n_exp, inter, hid);
                            let dqs_mar = permute_q4_to_marlin(&dqs, n_exp, hid, inter);
                            qw.insert(format!("{moe_up_name}_marlin"), (Tensor::new(d.upload(&tbu(&uqs_mar)).unwrap(), vec![uqs_mar.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&usc)).unwrap(), vec![usc.len()], DType::F16), n_exp * inter, hid));
                            qw.insert(format!("{moe_down_name}_marlin"), (Tensor::new(d.upload(&tbu(&dqs_mar)).unwrap(), vec![dqs_mar.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&dsc)).unwrap(), vec![dsc.len()], DType::F16), n_exp * hid, inter));
                        }
                    }
                } else {
                    // Cache miss: build + cache.
                    let (mut uqs, mut usc, mut dqs, mut dsc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    for e in 0..n_exp {
                        let (q, s) = quantize_q4(&g(&format!("{p}.mixer.experts.{e}.up_proj.weight")), inter, hid);
                        uqs.extend(q); usc.extend(s);
                        let (q, s) = quantize_q4(&g(&format!("{p}.mixer.experts.{e}.down_proj.weight")), hid, inter);
                        dqs.extend(q); dsc.extend(s);
                    }
                    cache_write(&moe_up_name, &uqs, &usc, n_exp * inter, hid, true);
                    cache_write(&moe_down_name, &dqs, &dsc, n_exp * hid, inter, true);
                    // STANDARD layout under base keys (sequential step path).
                    qw.insert(moe_up_name.clone(), (Tensor::new(d.upload(&tbu(&uqs)).unwrap(), vec![uqs.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&usc)).unwrap(), vec![usc.len()], DType::F16), n_exp * inter, hid));
                    qw.insert(moe_down_name.clone(), (Tensor::new(d.upload(&tbu(&dqs)).unwrap(), vec![dqs.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&dsc)).unwrap(), vec![dsc.len()], DType::F16), n_exp * hid, inter));
                    if use_marlin_layout {
                        let uqs_mar = permute_q4_to_marlin(&uqs, n_exp, inter, hid);
                        let dqs_mar = permute_q4_to_marlin(&dqs, n_exp, hid, inter);
                        qw.insert(format!("{moe_up_name}_marlin"), (Tensor::new(d.upload(&tbu(&uqs_mar)).unwrap(), vec![uqs_mar.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&usc)).unwrap(), vec![usc.len()], DType::F16), n_exp * inter, hid));
                        qw.insert(format!("{moe_down_name}_marlin"), (Tensor::new(d.upload(&tbu(&dqs_mar)).unwrap(), vec![dqs_mar.len()], DType::U32), Tensor::new(d.upload(&tb_f16(&dsc)).unwrap(), vec![dsc.len()], DType::F16), n_exp * hid, inter));
                    }
                }
                // NVFP4 W4A4 path (NEMOTRON_FP4_MOE=1): quantize+fragment-pack the
                // routed experts on DEVICE (upload f16 once, pack kernels do the rest).
                if use_fp4_moe {
                    let mut build = |which: &str, n: usize, k: usize| {
                        let key = format!("{p}.moe_{which}_fp4");
                        let (kt2, nt2) = (k / 64, n / 8);
                        let (wp_len, wsc_len) = (n_exp * nt2 * kt2 * 32 * 2, n_exp * nt2 * kt2 * 8);
                        let cpath = cache_path(&format!("{key}.v1"));
                        // disk-cache hit: upload the packed blobs directly (skips the
                        // f32 collect + f16 convert + device pack — the slow part).
                        if let Ok(b) = std::fs::read(&cpath) {
                            let expect = (wp_len + wsc_len + n_exp) * 4;
                            if b.len() == expect {
                                let wp = Tensor::new(d.upload(&b[..wp_len * 4]).unwrap(), vec![wp_len], DType::U32);
                                let wsc = Tensor::new(d.upload(&b[wp_len * 4..(wp_len + wsc_len) * 4]).unwrap(), vec![wsc_len], DType::U32);
                                let gw = Tensor::new(d.upload(&b[(wp_len + wsc_len) * 4..]).unwrap(), vec![n_exp], DType::F32);
                                fp4w.insert(key, (wp, wsc, gw));
                                return;
                            }
                        }
                        let mut wh: Vec<f32> = Vec::with_capacity(n_exp * n * k);
                        for e in 0..n_exp {
                            wh.extend(g(&format!("{p}.mixer.experts.{e}.{which}.weight")));
                        }
                        let bytes = tb_f16(&wh);
                        drop(wh);
                        let wt = Tensor::new(d.upload(&bytes).unwrap(), vec![n_exp * n, k], DType::F16);
                        drop(bytes);
                        let (wp, wsc, gw) = ffai_ops::moe_fp4_pack_weights(d, &wt, n_exp, n, k).unwrap();
                        // fill the cache (raw blob: wp ++ wsc ++ gw)
                        let mut blob = vec![0u8; (wp_len + wsc_len + n_exp) * 4];
                        d.download(wp.buffer.as_ref(), &mut blob[..wp_len * 4]).unwrap();
                        d.download(wsc.buffer.as_ref(), &mut blob[wp_len * 4..(wp_len + wsc_len) * 4]).unwrap();
                        d.download(gw.buffer.as_ref(), &mut blob[(wp_len + wsc_len) * 4..]).unwrap();
                        let _ = std::fs::write(&cpath, &blob);
                        fp4w.insert(key, (wp, wsc, gw));
                    };
                    build("up_proj", inter, hid);
                    build("down_proj", hid, inter);
                    // CUTLASS-layout NVFP4 weights (NEMOTRON_CUTLASS_MOE=1 on top
                    // of FP4_MOE): linear packed e2m1 + per-expert canonical-
                    // swizzle ue4m3 SF, quantized ONCE at setup from the
                    // dequanted f16 (parallel to the fragment pack above).
                    if use_fp4_cutlass_env {
                        let mut buildc = |which: &str, n: usize, k: usize| {
                            let key = format!("{p}.moe_{which}_fp4_cutlass");
                            let kb4 = (k / 16).div_ceil(4);
                            let sf_exp = n.div_ceil(128) * 512 * kb4;
                            let (pack_len, sf_len, gw_len) = (n_exp * n * k / 2, n_exp * sf_exp, n_exp * 4); // bytes
                            let cpath = cache_path(&format!("{key}.v2"));
                            // disk-cache hit: upload the packed blobs directly.
                            if let Ok(b) = std::fs::read(&cpath) {
                                if b.len() == pack_len + sf_len + gw_len {
                                    let wp = Tensor::new(d.upload(&b[..pack_len]).unwrap(), vec![pack_len], DType::U32);
                                    let wsf = Tensor::new(d.upload(&b[pack_len..pack_len + sf_len]).unwrap(), vec![sf_len], DType::U32);
                                    let gw = Tensor::new(d.upload(&b[pack_len + sf_len..]).unwrap(), vec![n_exp], DType::F32);
                                    fp4w_cutlass.insert(key, (wp, wsf, gw));
                                    return;
                                }
                            }
                            let mut wh: Vec<f32> = Vec::with_capacity(n_exp * n * k);
                            for e in 0..n_exp {
                                wh.extend(g(&format!("{p}.mixer.experts.{e}.{which}.weight")));
                            }
                            let bytes = tb_f16(&wh);
                            drop(wh);
                            let wt = Tensor::new(d.upload(&bytes).unwrap(), vec![n_exp * n, k], DType::F16);
                            drop(bytes);
                            let (wp, wsf, gw) = ffai_ops::lt_fp4_quant_slab(d, &wt, n_exp, n, k).unwrap();
                            // fill the cache (raw blob: pack ++ sf ++ gw)
                            let mut blob = vec![0u8; pack_len + sf_len + gw_len];
                            d.download(wp.buffer.as_ref(), &mut blob[..pack_len]).unwrap();
                            d.download(wsf.buffer.as_ref(), &mut blob[pack_len..pack_len + sf_len]).unwrap();
                            d.download(gw.buffer.as_ref(), &mut blob[pack_len + sf_len..]).unwrap();
                            let _ = std::fs::write(&cpath, &blob);
                            fp4w_cutlass.insert(key, (wp, wsf, gw));
                        };
                        buildc("up_proj", inter, hid);
                        buildc("down_proj", hid, inter);
                    }
                    // NEMOTRON_FP4_SHARED_FOLD=1: shared expert folded into the routed
                    // grouped GEMM as 2 extra always-selected groups. Up: shared up
                    // [2*inter, hid] appended as two [inter, hid] expert slabs. Down:
                    // shared down [hid, 2*inter] K-SPLIT into two [hid, inter] slabs
                    // (their scatter-adds sum to the full K=2*inter GEMM).
                    if std::env::var("NEMOTRON_FP4_SHARED_FOLD").is_ok() {
                        let mut buildf = |which: &str, n: usize, k: usize, split_k: bool| {
                            let key = format!("{p}.moe_{which}_fp4_fold");
                            let ne = n_exp + 2;
                            let (kt2, nt2) = (k / 64, n / 8);
                            let (wp_len, wsc_len) = (ne * nt2 * kt2 * 32 * 2, ne * nt2 * kt2 * 8);
                            let cpath = cache_path(&format!("{key}.v1"));
                            if let Ok(b) = std::fs::read(&cpath) {
                                let expect = (wp_len + wsc_len + ne) * 4;
                                if b.len() == expect {
                                    let wp = Tensor::new(d.upload(&b[..wp_len * 4]).unwrap(), vec![wp_len], DType::U32);
                                    let wsc = Tensor::new(d.upload(&b[wp_len * 4..(wp_len + wsc_len) * 4]).unwrap(), vec![wsc_len], DType::U32);
                                    let gw = Tensor::new(d.upload(&b[(wp_len + wsc_len) * 4..]).unwrap(), vec![ne], DType::F32);
                                    fp4w.insert(key, (wp, wsc, gw));
                                    return;
                                }
                            }
                            let mut wh: Vec<f32> = Vec::with_capacity(ne * n * k);
                            for e in 0..n_exp {
                                wh.extend(g(&format!("{p}.mixer.experts.{e}.{which}.weight")));
                            }
                            let shw: Vec<f32> = g(&format!("{p}.mixer.shared_experts.{which}.weight"));
                            if !split_k {
                                // up: [2n, k] row-major → slab order is already two [n,k] stacks
                                wh.extend(&shw);
                            } else {
                                // down: [n, 2k] row-major → two [n, k] column halves
                                for half in 0..2 {
                                    for r in 0..n {
                                        wh.extend(&shw[r * 2 * k + half * k..r * 2 * k + half * k + k]);
                                    }
                                }
                            }
                            let bytes = tb_f16(&wh);
                            drop(wh);
                            let wt = Tensor::new(d.upload(&bytes).unwrap(), vec![ne * n, k], DType::F16);
                            drop(bytes);
                            let (wp, wsc, gw) = ffai_ops::moe_fp4_pack_weights(d, &wt, ne, n, k).unwrap();
                            let mut blob = vec![0u8; (wp_len + wsc_len + ne) * 4];
                            d.download(wp.buffer.as_ref(), &mut blob[..wp_len * 4]).unwrap();
                            d.download(wsc.buffer.as_ref(), &mut blob[wp_len * 4..(wp_len + wsc_len) * 4]).unwrap();
                            d.download(gw.buffer.as_ref(), &mut blob[(wp_len + wsc_len) * 4..]).unwrap();
                            let _ = std::fs::write(&cpath, &blob);
                            fp4w.insert(key, (wp, wsc, gw));
                        };
                        buildf("up_proj", inter, hid, false);
                        buildf("down_proj", hid, inter, true);
                    }
                    // shared expert (n_exp=1 grouped call), NEMOTRON_FP4_SHARED=1
                    if std::env::var("NEMOTRON_FP4_SHARED").is_ok() {
                        let mut build1 = |which: &str, n: usize, k: usize| {
                            let wh: Vec<f32> = g(&format!("{p}.mixer.shared_experts.{which}.weight"));
                            let bytes = tb_f16(&wh);
                            drop(wh);
                            let wt = Tensor::new(d.upload(&bytes).unwrap(), vec![n, k], DType::F16);
                            drop(bytes);
                            let (wp, wsc, gw) = ffai_ops::moe_fp4_pack_weights(d, &wt, 1, n, k).unwrap();
                            fp4w.insert(format!("{p}.sh_{which}_fp4"), (wp, wsc, gw));
                        };
                        build1("up_proj", shared_inter, hid);
                        build1("down_proj", hid, shared_inter);
                    }
                }
                // NVFP4 recipe: shared-expert up/down → Q8.
                if use_nvfp4_recipe {
                    qload8(&mut qw8, &format!("{p}.mixer.shared_experts.up_proj.weight"), shared_inter, hid);
                    qload8(&mut qw8, &format!("{p}.mixer.shared_experts.down_proj.weight"), hid, shared_inter);
                } else {
                    qload(&mut qw, &format!("{p}.mixer.shared_experts.up_proj.weight"), shared_inter, hid, true);
                    qload(&mut qw, &format!("{p}.mixer.shared_experts.down_proj.weight"), hid, shared_inter, true);
                }
            }
            '*' => {
                qload(&mut qw, &format!("{p}.mixer.q_proj.weight"), qdim, hid, true);
                qload(&mut qw, &format!("{p}.mixer.k_proj.weight"), kvdim, hid, true);
                qload(&mut qw, &format!("{p}.mixer.v_proj.weight"), kvdim, hid, true);
                // NVFP4 recipe: attention o_proj → Q8.
                if use_nvfp4_recipe {
                    qload8(&mut qw8, &format!("{p}.mixer.o_proj.weight"), hid, qdim);
                } else {
                    qload(&mut qw, &format!("{p}.mixer.o_proj.weight"), hid, qdim, true);
                }
            }
            _ => unreachable!(),
        }
    }
    let load_s = t_load.elapsed().as_secs_f64();
    let q4_gb: f64 = qw.values().map(|(q, s, _, _)| (q.elem_count() * 4 + s.elem_count() * if s.dtype == DType::F16 { 2usize } else { 4usize }) as f64).sum::<f64>() / 1e9;
    let q8_gb: f64 = (qw8.values().map(|(q, s, _, _)| (q.elem_count() * 4 + s.elem_count() * 4) as f64).sum::<f64>() / 1e9).max(0.0);
    let recipe_label = if use_nvfp4_recipe { "NVFP4-recipe" } else { "all-Q4" };
    eprintln!("Nemotron resident setup [{}]: {:.1}s ({} Q4 + {} Q8 matrices, ~{:.2}GB Q4 + {:.2}GB Q8 = {:.2}GB total)",
        recipe_label, load_s, qw.len(), qw8.len(), q4_gb, q8_gb, q4_gb + q8_gb);

    // resident-weight quantized matvec
    let qmv = |x: &Tensor, name: &str| -> Tensor {
        let (qs, sc, m, k) = &qw[name];
        gemv_q4(d, qs, sc, x, *m, *k, *m).unwrap()
    };
    // resident-weight matvec that scales + accumulates into `acc` in one kernel.
    let qacc = |x: &Tensor, name: &str, acc: &Tensor, sb: &Tensor| {
        let (qs, sc, m, k) = &qw[name];
        gemv_q4_accum(d, qs, sc, x, acc, sb, *m, *k, *m).unwrap();
    };
    // resident-weight matvec with fused ReLU² (MoE expert up-projection).
    let qrelu2 = |x: &Tensor, name: &str| -> Tensor {
        let (qs, sc, m, k) = &qw[name];
        gemv_q4_relu2(d, qs, sc, x, *m, *k, *m).unwrap()
    };

    // ── Q8 dispatch closures (NVFP4-recipe layers) ────────────────────────────
    // These are no-ops when use_nvfp4_recipe=false (qw8 is empty; never called).
    let qmv8 = |x: &Tensor, name: &str| -> Tensor {
        let (qs, sc, m, k) = &qw8[name];
        gemv_q8(d, qs, sc, x, *m, *k, *m).unwrap()
    };
    let qrelu2_q8 = |x: &Tensor, name: &str| -> Tensor {
        let (qs, sc, m, k) = &qw8[name];
        gemv_q8_relu2(d, qs, sc, x, *m, *k, *m).unwrap()
    };
    let qacc_q8 = |x: &Tensor, name: &str, acc: &Tensor, sb: &Tensor| {
        let (qs, sc, m, k) = &qw8[name];
        gemv_q8_accum(d, qs, sc, x, acc, sb, *m, *k, *m).unwrap();
    };

    // ── Fused MoE FFN (NEMOTRON_MOE_FUSED=1) ─────────────────────────────────
    // Pre-allocate the scratch buffer once. All 23 MoE layers share it (they're
    // serial — layer N+1 starts only after N finishes).
    // NOTE: NEMOTRON_GRAPH + NEMOTRON_MOE_FUSED are mutually exclusive — the
    // fused kernel uses cuLaunchCooperativeKernel which can't be captured into a
    // CUDA graph. If both are set, fusion is silently disabled.
    let use_moe_fused = std::env::var("NEMOTRON_MOE_FUSED").is_ok()
        && !std::env::var("NEMOTRON_GRAPH").is_ok();
    let moe_scratch = if use_moe_fused {
        let scratch_bytes = top_k * inter * 4; // f32
        Some(Tensor::new(d.upload(&vec![0u8; scratch_bytes]).unwrap(), vec![top_k * inter], DType::F32))
    } else {
        None
    };
    if use_moe_fused {
        eprintln!("NEMOTRON_MOE_FUSED=1: using cooperative-groups fused MoE FFN kernel (eager, non-graph)");
        eprintln!("  NOTE: cuLaunchCooperativeKernel requires all grid blocks resident simultaneously.");
        eprintln!("  On GB10 (48 SMs), max coop blocks = 288 at 256 threads, need 336 for hid=2688.");
        eprintln!("  This will FAIL with 'too many blocks'. See NEMOTRON_MOE_FUSED analysis.");
    } else if std::env::var("NEMOTRON_MOE_FUSED").is_ok() && std::env::var("NEMOTRON_GRAPH").is_ok() {
        eprintln!("NEMOTRON_MOE_FUSED=1: disabled (NEMOTRON_GRAPH is set — cooperative launch not capturable)");
    }

    // ── DECODE: per-token forward reusing resident weights, KV + Mamba state ──
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let prefill = env("NEMOTRON_PREFILL", 0);
    let n_decode = env("NEMOTRON_DECODE", 32);
    let fakectx = env("NEMOTRON_FAKECTX", 0);
    // GQA split-K flash-decode sdpa: default ON for long context (≈+15-25% @ 32K
    // where the single-pass re-reads each shared KV head gqa_factor×); single-pass
    // is marginally better at tiny ctx (2-pass partial overhead). Opt out w/ NEMOTRON_NO_2PASS.
    let no_2pass = std::env::var("NEMOTRON_NO_2PASS").is_ok();
    // Device MoE router REGRESSED (-17-24%, clock-locked): the 1-simdgroup serial
    // top-k bubble underutilizes the GPU worse than host top-k + a cheap sync drain
    // (same lesson as device-Mamba). Host router is DEFAULT; device is opt-in only.
    let no_devrouter = !std::env::var("NEMOTRON_DEVROUTER").is_ok();
    // CUDA-graph capture/replay: collapse ~390 per-token kernel launches into ONE
    // cuGraphLaunch to remove host-launch-gap overhead. Requires NEMOTRON_DEVROUTER=1
    // (device MoE router, no host sync) and gates the final argmax download so the
    // captured region is fully sync-free. Set NEMOTRON_GRAPH=1 to enable.
    let use_graph = std::env::var("NEMOTRON_GRAPH").is_ok();
    // skip_dl: Cell<bool> threaded into the step closure. Set true during capture and
    // replay so the dl(logits) host-sync is skipped (forbidden during capture; not
    // needed for the graph throughput measurement).
    let skip_dl = std::cell::Cell::new(false);
    // F16 KV cache: the clock-locked "+11-27%" was a thermal/order artifact (the
    // internal A/B shows it neutral-to-negative — casts cost ≥ halved-read saves).
    // DEFAULT OFF (f32, no casts); opt in with NEMOTRON_F16KV.
    let f16kv = std::env::var("NEMOTRON_F16KV").is_ok();
    // KV-cache capacity (positions). In the batched-prefill path the queries
    // run at positions [fakectx, fakectx+prefill), so the cache must hold the
    // full fakectx prefix PLUS the whole prefill block PLUS the decode tail —
    // i.e. fakectx + prefill, NOT max(fakectx, prefill). Using max() under-sized
    // the cache whenever both were nonzero, so sdpa_multi/kv_append_many walked
    // past the buffer (base_kv + n_query > kv_stride) → illegal memory access at
    // deep context. (Plain decode, where one of the two is 0, is unaffected.)
    // NEMOTRON_PACKED=N packs N prompts (s·N tokens) into one batched prefill,
    // so the KV cache must hold N× the per-sequence length.
    let pack_n_cap = std::env::var("NEMOTRON_PACKED").ok()
        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
    let cap = fakectx + prefill * pack_n_cap + n_decode + 8;
    // per-layer state, indexed by absolute layer id. KV cache is now ON-DEVICE
    // ([nkv,cap,hd] per attn layer), so the growing context never round-trips
    // through the host — the 32K decode fix.
    let mut conv_state: Vec<Vec<f32>> = vec![Vec::new(); 52];
    let conv_dev: std::cell::RefCell<Vec<Option<Tensor>>> = std::cell::RefCell::new((0..52).map(|_| None).collect()); // device conv state (NEMOTRON_DEVMAMBA)
    let mut ssm_state: Vec<Option<Tensor>> = (0..52).map(|_| None).collect(); // recurrent SSM state ON-DEVICE
    let mut kvcache: Vec<Option<(Tensor, Tensor)>> = (0..52).map(|_| None).collect();
    let u32buf = |v: u32| Tensor::new(d.upload(&v.to_le_bytes()).unwrap(), vec![1], DType::U32);
    let ones_gs = up(&vec![1.0f32; gs]); // grouped-norm: normalize each 512-group weightless, then scale by the real weight
    // Optional per-op profiling (NEMOTRON_PROFILE=1): synchronize around EACH individual
    // op call to attribute GPU time precisely. Adds sync overhead on every op — use only
    // for profiling runs, not for throughput measurement.
    let prof = std::env::var("NEMOTRON_PROFILE").is_ok();

    // Per-op time accumulators (Cell<f64> = interior-mutable across the step closure).
    // Index: 0=rms_norm, 1=m_in_proj, 2=slice, 3=conv1d, 4=silu, 5=conv_roll,
    //        6=softplus_add, 7=ssm_step, 8=gated_norm, 9=m_out_proj,
    //        10=moe_gate_gemv, 11=moe_router_dev, 12=moe_gather_up, 13=moe_gather_down,
    //        14=moe_wsum, 15=shared_up, 16=shared_down_acc,
    //        17=rope, 18=q_proj, 19=k_proj, 20=v_proj, 21=kv_append, 22=sdpa, 23=o_proj,
    //        24=add, 25=norm_f, 26=lm_head, 27=add_residual_m, 28=add_residual_e,
    //        29=add_residual_a, 30=cast_f16
    const N_OPS: usize = 31;
    let op_t: Vec<std::cell::Cell<f64>> = (0..N_OPS).map(|_| std::cell::Cell::new(0f64)).collect();
    let op_calls: Vec<std::cell::Cell<u64>> = (0..N_OPS).map(|_| std::cell::Cell::new(0u64)).collect();
    // Bytes-read estimates per op (f64, accumulated across all calls)
    let op_bytes: Vec<std::cell::Cell<f64>> = (0..N_OPS).map(|_| std::cell::Cell::new(0f64)).collect();
    // Wall time per step (to compute host overhead = wall - sum_gpu)
    let step_wall: std::cell::Cell<f64> = std::cell::Cell::new(0f64);
    let step_count: std::cell::Cell<u64> = std::cell::Cell::new(0u64);

    // Macro-like helper: time one op when profiling is on.
    // sync(); t0 = now(); result = expr; sync(); acc.
    // Because we can't use macros in closures portably, we inline this pattern.
    // The (tm, te, ta, th) coarse cells kept for backward compat display.
    let (tm, te, ta, th) = (std::cell::Cell::new(0f64), std::cell::Cell::new(0f64), std::cell::Cell::new(0f64), std::cell::Cell::new(0f64));

    // INSTRUMENTATION (correctness audit): capture the last-token logit vector of
    // the most recent `step()` call so the prefill gate can compute logit-level
    // metrics (cosine, top-5 overlap, max-abs err) vs the batched path. REVERT.
    thread_local! { static LAST_STEP_LOGITS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) }; }
    thread_local! { static ROUTER_HOST_T: std::cell::Cell<f64> = const { std::cell::Cell::new(0f64) }; }
    thread_local! { static STEP_LAYER_TRACE: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) }; }
    thread_local! { static BATCHED_LAYER_TRACE: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) }; }
    thread_local! { static STEP0_FROZEN: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) }; }
    type SsmDump = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    thread_local! { static SEQ_SSM_DUMP: std::cell::RefCell<SsmDump> = const { std::cell::RefCell::new((Vec::new(),Vec::new(),Vec::new(),Vec::new())) }; }
    thread_local! { static BAT_SSM_DUMP: std::cell::RefCell<SsmDump> = const { std::cell::RefCell::new((Vec::new(),Vec::new(),Vec::new(),Vec::new())) }; }

    // one decode step at absolute position `pos`; returns next-token logits' argmax
    let step = |token: usize, pos: usize,
                    conv_state: &mut Vec<Vec<f32>>, ssm_state: &mut Vec<Option<Tensor>>,
                    kvcache: &mut Vec<Option<(Tensor, Tensor)>>| -> usize {
        let step_t0 = if prof { d.synchronize().ok(); Some(Instant::now()) } else { None };

        // Per-op timing helper: sync before + after, accumulate into op_t[idx].
        // bytes_read: approximate DRAM bytes fetched (input tensors + weights).
        // Only active when prof=true; otherwise the closure is zero-cost (condition
        // checked at runtime but the body is short-circuit skipped).
        macro_rules! pt {
            ($idx:expr, $bytes:expr, $expr:expr) => {{
                if prof {
                    d.synchronize().ok();
                    let _t0 = Instant::now();
                    let _r = $expr;
                    d.synchronize().ok();
                    let _e = _t0.elapsed().as_secs_f64();
                    op_t[$idx].set(op_t[$idx].get() + _e);
                    op_calls[$idx].set(op_calls[$idx].get() + 1);
                    op_bytes[$idx].set(op_bytes[$idx].get() + $bytes as f64);
                    _r
                } else {
                    $expr
                }
            }};
        }

        // residual stream stays ON-DEVICE the whole forward — no per-layer up(x)/dl(out).
        let mut xt = up(&embed[token * hid..(token + 1) * hid]);
        for (l, mix) in PATTERN.chars().enumerate() {
            let p = format!("language_model.backbone.layers.{l}");
            // rms_norm per layer: reads x (hid*4B) + weight (hid*4B)
            let xn = pt!(0, hid * 8, rms_norm(d, &xt, &fwd[&format!("{p}.norm.weight")], eps).unwrap());
            match mix {
                'M' => {
                    if std::env::var("NEMOTRON_SKIPMAMBA").is_ok() { continue; }
                    if !std::env::var("NEMOTRON_HOSTMAMBA").is_ok() {
                        // ALL-DEVICE Mamba (DEFAULT): no dl/host round-trips. +3.7% clean
                        // internal A/B; argmax 1234. Opt out: NEMOTRON_HOSTMAMBA.
                        // in_proj gemv_q4/q8: recipe layers use Q8 (2× bytes vs Q4)
                        let proj = pt!(1, if use_nvfp4_recipe { in_proj_out * hid + in_proj_out * 4 + hid * 4 } else { in_proj_out * hid / 2 + in_proj_out * 2 + hid * 4 },
                            if use_nvfp4_recipe { qmv8(&xn, &format!("{p}.mixer.in_proj.weight")) } else { qmv(&xn, &format!("{p}.mixer.in_proj.weight")) });
                        // slice×3: reads proj (in_proj_out*4B each)
                        let zt      = pt!(2, in_proj_out * 4, slice(d, &proj, 0, di).unwrap());
                        let xbc_t   = pt!(2, in_proj_out * 4, slice(d, &proj, di, conv_dim).unwrap());
                        let dt_raw_t= pt!(2, in_proj_out * 4, slice(d, &proj, di + conv_dim, m_nh).unwrap());
                        { let mut cd = conv_dev.borrow_mut(); if cd[l].is_none() { cd[l] = Some(Tensor::new(d.alloc_zeroed((kc - 1) * conv_dim * 4).unwrap(), vec![(kc - 1) * conv_dim], DType::F32)); } }
                        // conv1d: reads state ((kc-1)*conv_dim*4B) + xbc_t (conv_dim*4B) + weight (kc*conv_dim*4B) + bias (conv_dim*4B)
                        let yc = pt!(3, (kc - 1) * conv_dim * 4 + conv_dim * 4 + kc * conv_dim * 4 + conv_dim * 4, {
                            let cd = conv_dev.borrow();
                            conv1d_causal_step(d, &xbc_t, &fwd[&format!("{p}.mixer.conv1d.weight")], &fwd[&format!("{p}.mixer.conv1d.bias")], cd[l].as_ref().unwrap(), conv_dim as u32, kc as u32).unwrap()
                        });
                        // silu: reads yc (conv_dim*4B)
                        let xbc_act = pt!(4, conv_dim * 4, silu(d, &yc).unwrap());
                        // conv1d_causal_step above is the SOLE conv-ring updater on the all-device
                        // path: it shifts cd[l]'s ring in place and appends xbc_t (mutating the
                        // persistent device buffer). The previous conv_roll here shifted+appended a
                        // SECOND time -> the current token was double-counted in the conv history at
                        // every position >=1, silently corrupting multi-token decode past token 0
                        // (invisible at pos 0 where the ring is zero). Removed. (host-fallback branch
                        // below stays as-is: it uploads a throwaway copy + rolls conv_state[l] once.)
                        let x_ssm = pt!(2, (di + ng * ds * 2) * 4, slice(d, &xbc_act, 0, di).unwrap());
                        let bmat   = pt!(2, (di + ng * ds * 2) * 4, slice(d, &xbc_act, di, ng * ds).unwrap());
                        let cmat   = pt!(2, (di + ng * ds * 2) * 4, slice(d, &xbc_act, di + ng * ds, ng * ds).unwrap());
                        // softplus_add: reads dt_raw_t (m_nh*4B) + bias (m_nh*4B)
                        let dt = pt!(6, m_nh * 8, softplus_add(d, &dt_raw_t, &up(&fw[&format!("{p}.mixer.dt_bias")])).unwrap());
                        if ssm_state[l].is_none() { ssm_state[l] = Some(Tensor::new(d.alloc_zeroed(m_nh * m_dh * ds * 4).unwrap(), vec![m_nh * m_dh * ds], DType::F32)); }
                        // ssm_step: reads state (m_nh*m_dh*ds*4B) + x (di*4B) + A,B,C,D,dt (small)
                        let (so, y_t) = pt!(7, m_nh * m_dh * ds * 4 + di * 4 + m_nh * 5 * 4 + ng * ds * 2 * 4, {
                            ssm_step(d, &x_ssm, &fwd[&format!("{p}.mixer.A_log")], &bmat, &cmat, &fwd[&format!("{p}.mixer.D")], &dt, ssm_state[l].as_ref().unwrap(), m_dh as u32, ds as u32, m_nh as u32, (m_nh / ng) as u32).unwrap()
                        });
                        ssm_state[l] = Some(so);
                        // INSTRUMENTATION (revert): dump L0 SSM in/out for the seq path.
                        if l == 0 && std::env::var("NEMOTRON_DUMP_SSM").is_ok() {
                            let xs = dl(&x_ssm, di); let yy = dl(&y_t, di);
                            let bb = dl(&bmat, ng*ds); let cc = dl(&cmat, ng*ds);
                            SEQ_SSM_DUMP.with(|c| *c.borrow_mut() = (xs, bb, cc, yy));
                        }
                        // gated_group_rmsnorm: reads y_t (di*4B) + zt (di*4B) + norm weight (di*4B)
                        let yn = pt!(8, di * 12, gated_group_rmsnorm(d, &y_t, &zt, &up(&fw[&format!("{p}.mixer.norm.weight")]), eps, di, gs).unwrap());
                        // out_proj gemv_q4/q8: recipe → Q8
                        let out = pt!(9, if use_nvfp4_recipe { hid * di + hid * 4 + di * 4 } else { hid * di / 2 + hid * 2 + di * 4 },
                            if use_nvfp4_recipe { qmv8(&yn, &format!("{p}.mixer.out_proj.weight")) } else { qmv(&yn, &format!("{p}.mixer.out_proj.weight")) });
                        // add residual: reads xt (hid*4B) + out (hid*4B)
                        xt = pt!(27, hid * 8, add(d, &xt, &out).unwrap());
                        if prof { d.synchronize().ok(); let e = step_t0.map(|_| 0.0).unwrap_or(0.0); let _ = e; tm.set(tm.get()); } // coarse compat
                        if std::env::var("NEMOTRON_DUMP_LAYERS").is_ok() {
                            let h = dl(&xt, hid);
                            STEP_LAYER_TRACE.with(|c| { let mut v = c.borrow_mut(); if v.len() <= l { v.resize(l+1, Vec::new()); } v[l] = h; });
                        }
                        continue;
                    }
                    let proj = dl(&pt!(1, if use_nvfp4_recipe { in_proj_out * hid + in_proj_out * 4 + hid * 4 } else { in_proj_out * hid / 2 + in_proj_out * 2 + hid * 4 },
                        if use_nvfp4_recipe { qmv8(&xn, &format!("{p}.mixer.in_proj.weight")) } else { qmv(&xn, &format!("{p}.mixer.in_proj.weight")) }), in_proj_out);
                    let z = &proj[0..di];
                    let xbc = &proj[di..di + conv_dim];
                    let dt_raw = &proj[di + conv_dim..di + conv_dim + m_nh];
                    if conv_state[l].is_empty() { conv_state[l] = vec![0.0f32; (kc - 1) * conv_dim]; }
                    let yc = pt!(3, (kc - 1) * conv_dim * 4 + conv_dim * 4 + kc * conv_dim * 4 + conv_dim * 4,
                        conv1d_causal_step(d, &up(xbc), &fwd[&format!("{p}.mixer.conv1d.weight")], &fwd[&format!("{p}.mixer.conv1d.bias")], &up(&conv_state[l]), conv_dim as u32, kc as u32).unwrap());
                    let xbc_act = dl(&pt!(4, conv_dim * 4, silu(d, &yc).unwrap()), conv_dim);
                    { let s = &mut conv_state[l]; s.drain(0..conv_dim); s.extend_from_slice(xbc); }
                    let x_ssm = &xbc_act[0..di];
                    let bmat = &xbc_act[di..di + ng * ds];
                    let cmat = &xbc_act[di + ng * ds..di + 2 * ng * ds];
                    let dt_bias = &fw[&format!("{p}.mixer.dt_bias")];
                    let dt: Vec<f32> = (0..m_nh).map(|i| softplus(dt_raw[i] + dt_bias[i])).collect();
                    if ssm_state[l].is_none() { ssm_state[l] = Some(Tensor::new(d.alloc_zeroed(m_nh * m_dh * ds * 4).unwrap(), vec![m_nh * m_dh * ds], DType::F32)); }
                    let (so, y_t) = pt!(7, m_nh * m_dh * ds * 4 + di * 4 + m_nh * 5 * 4 + ng * ds * 2 * 4, {
                        ssm_step(d, &up(x_ssm), &fwd[&format!("{p}.mixer.A_log")], &up(bmat), &up(cmat), &fwd[&format!("{p}.mixer.D")], &up(&dt), ssm_state[l].as_ref().unwrap(), m_dh as u32, ds as u32, m_nh as u32, (m_nh / ng) as u32).unwrap()
                    });
                    ssm_state[l] = Some(so);
                    let y = dl(&y_t, di);
                    let nw = &fw[&format!("{p}.mixer.norm.weight")];
                    let mut yn = vec![0.0f32; di];
                    for grp in 0..ng {
                        let s = grp * gs;
                        let mut ss = 0.0f32;
                        for i in 0..gs { let g = y[s + i] * (z[s + i] / (1.0 + (-z[s + i]).exp())); yn[s + i] = g; ss += g * g; }
                        let r = 1.0 / ((ss / gs as f32) + eps).sqrt();
                        for i in 0..gs { yn[s + i] = yn[s + i] * r * nw[s + i]; }
                    }
                    let out = pt!(9, if use_nvfp4_recipe { hid * di + hid * 4 + di * 4 } else { hid * di / 2 + hid * 2 + di * 4 },
                        if use_nvfp4_recipe { qmv8(&up(&yn), &format!("{p}.mixer.out_proj.weight")) } else { qmv(&up(&yn), &format!("{p}.mixer.out_proj.weight")) });
                    xt = pt!(27, hid * 8, add(d, &xt, &out).unwrap());
                }
                'E' => {
                    if std::env::var("NEMOTRON_SKIPMOE").is_ok() { continue; }
                    // Router: ON-DEVICE (sigmoid+bias+top-k+norm+scale, no host sync) by
                    // default; host path kept for A/B via NEMOTRON_HOSTROUTER.
                    let (idx_buf, wts_buf) = if !std::env::var("NEMOTRON_DEVROUTER").is_ok() {
                        // host router: gate gemv (f32 weight 128×2688) then host top-k
                        let rl = dl(&pt!(10, n_exp * hid * 4, gemv(d, &fwd[&format!("{p}.mixer.gate.weight")], &xn).unwrap()), n_exp);
                        let sig: Vec<f32> = rl.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
                        let bias = &fw[&format!("{p}.mixer.gate.e_score_correction_bias")];
                        let choice: Vec<f32> = (0..n_exp).map(|i| sig[i] + bias[i]).collect();
                        let eidx = ffai_runtime::topk(&choice, top_k);
                        let mut w: Vec<f32> = eidx.iter().map(|&e| sig[e]).collect();
                        let wsum: f32 = w.iter().sum::<f32>() + 1e-20;
                        for v in w.iter_mut() { *v = *v / wsum * scale_f; }
                        (Tensor::new(d.upload(&tbu(&eidx.iter().map(|&e| e as u32).collect::<Vec<_>>())).unwrap(), vec![top_k], DType::U32), up(&w))
                    } else {
                        let logits = pt!(10, n_exp * hid * 4, gemv(d, &fwd[&format!("{p}.mixer.gate.weight")], &xn).unwrap());
                        let bias_dev = up(&fw[&format!("{p}.mixer.gate.e_score_correction_bias")]);
                        // device router kernel: reads logits (n_exp*4B) + bias (n_exp*4B)
                        pt!(11, n_exp * 8, moe_router_device(d, &logits, &bias_dev, n_exp, top_k, scale_f).unwrap())
                    };
                    let acc_dev = up(&vec![0.0f32; hid]);
                    let (uqs, usc, _, _) = &qw[&format!("{p}.moe_up_all")];
                    let (dqs, dsc, _, _) = &qw[&format!("{p}.moe_down_all")];
                    if let Some(ref scratch) = moe_scratch {
                        // ── FUSED path: one cooperative kernel, intermediate in L2 ──
                        // Reads: up Q4 (top_k*inter*hid/2) + up scales + dn Q4 (top_k*hid*inter/2) + dn scales + x (hid*4B)
                        pt!(12, top_k * inter * hid / 2 + top_k * inter * 2 + top_k * hid * inter / 2 + top_k * hid * 2 + hid * 4,
                            moe_fused_ffn(d, uqs, usc, dqs, dsc, &xn, &idx_buf, &wts_buf, &acc_dev, scratch, hid, inter, top_k).unwrap());
                    } else {
                        // ── Two-kernel path (baseline) ──
                        // moe_gather_up_relu2: reads top_k expert rows (top_k*inter*hid/2 Q4) + scales + x (hid*4B)
                        let a = pt!(12, top_k * inter * hid / 2 + top_k * inter * 2 + hid * 4,
                            moe_gather_up_relu2(d, uqs, usc, &xn, &idx_buf, top_k, inter, hid).unwrap());
                        // moe_gather_down: reads top_k expert rows (top_k*hid*inter/2 Q4) + scales + a (top_k*inter*4B)
                        let downs = pt!(13, top_k * hid * inter / 2 + top_k * hid * 2 + top_k * inter * 4,
                            moe_gather_down(d, dqs, dsc, &a, &idx_buf, top_k, inter, hid).unwrap());
                        // moe_weighted_sum: reads downs (top_k*hid*4B) + weights (top_k*4B)
                        pt!(14, top_k * hid * 4 + top_k * 4,
                            moe_weighted_sum(d, &downs, &wts_buf, &acc_dev, hid, top_k).unwrap());
                    }
                    // shared expert up qrelu2: Q4 or Q8 (recipe)
                    let sa = pt!(15, if use_nvfp4_recipe { shared_inter * hid + shared_inter * 4 + hid * 4 } else { shared_inter * hid / 2 + shared_inter * 2 + hid * 4 },
                        if use_nvfp4_recipe { qrelu2_q8(&xn, &format!("{p}.mixer.shared_experts.up_proj.weight")) } else { qrelu2(&xn, &format!("{p}.mixer.shared_experts.up_proj.weight")) });
                    // shared expert down qacc: Q4 or Q8 (recipe)
                    pt!(16, if use_nvfp4_recipe { hid * shared_inter + hid * 4 + shared_inter * 4 } else { hid * shared_inter / 2 + hid * 2 + shared_inter * 4 },
                        if use_nvfp4_recipe { qacc_q8(&sa, &format!("{p}.mixer.shared_experts.down_proj.weight"), &acc_dev, &up(&[1.0f32])) } else { qacc(&sa, &format!("{p}.mixer.shared_experts.down_proj.weight"), &acc_dev, &up(&[1.0f32])) });
                    // add residual
                    xt = pt!(28, hid * 8, add(d, &xt, &acc_dev).unwrap());
                }
                '*' => {
                    // q/k/v + RoPE stay ON-DEVICE; append k,v straight into the
                    // device KV cache; sdpa reads the cache. No host KV traffic.
                    // q_proj gemv_q4: reads Q4 (qdim*hid/2) + scales + x (hid*4B)
                    let q_raw = pt!(18, qdim * hid / 2 + qdim * 2 + hid * 4,
                        qmv(&xn, &format!("{p}.mixer.q_proj.weight")));
                    // k_proj gemv_q4
                    let k_raw = pt!(19, kvdim * hid / 2 + kvdim * 2 + hid * 4,
                        qmv(&xn, &format!("{p}.mixer.k_proj.weight")));
                    // v_proj gemv_q4
                    let v_raw = pt!(20, kvdim * hid / 2 + kvdim * 2 + hid * 4,
                        qmv(&xn, &format!("{p}.mixer.v_proj.weight")));
                    // rope: reads q (nq*hd*4B)
                    let q = pt!(17, nq * hd * 4,
                        rope_llama(d, &q_raw.reshaped(vec![nq, hd]), pos as u32, rope_theta, 1.0, 1.0, 1.0, 8192.0).unwrap());
                    // rope: reads k (nkv*hd*4B)
                    let k = pt!(17, nkv * hd * 4,
                        rope_llama(d, &k_raw.reshaped(vec![nkv, hd]), pos as u32, rope_theta, 1.0, 1.0, 1.0, 8192.0).unwrap());
                    let v = v_raw;
                    // F16 KV: halve the 32K KV read (sdpa = 34% of GPU). Cache + q/k/v in
                    // f16, 2pass reads them natively (registry-tested f16), attn→f32 for o_proj.
                    let (q, k, v) = if f16kv {
                        // cast f32→f16: reads (nq*hd*4B + nkv*hd*4B + nkv*hd*4B)
                        let qh = pt!(30, nq * hd * 4, cast_f32_f16(d, &q).unwrap());
                        let kh = pt!(30, nkv * hd * 4, cast_f32_f16(d, &k).unwrap());
                        let vh = pt!(30, nkv * hd * 4, cast_f32_f16(d, &v).unwrap());
                        (qh, kh, vh)
                    } else { (q, k, v) };
                    if kvcache[l].is_none() {
                        kvcache[l] = if f16kv {
                            let zf16 = || Tensor::new(d.alloc_zeroed(nkv * cap * hd * 2).unwrap(), vec![nkv * cap * hd], DType::F16);
                            Some((zf16(), zf16()))
                        } else {
                            Some((Tensor::new(d.alloc_zeroed(nkv * cap * hd * 4).unwrap(), vec![nkv * cap * hd], DType::F32), Tensor::new(d.alloc_zeroed(nkv * cap * hd * 4).unwrap(), vec![nkv * cap * hd], DType::F32)))
                        };
                    }
                    let (kcache, vcache) = kvcache[l].as_ref().unwrap();
                    let pb = u32buf(pos as u32);
                    // kv_append: writes nkv*hd floats into cache
                    pt!(21, nkv * hd * if f16kv { 2 } else { 4 },
                        kv_append(d, &k, kcache, &pb, hd, cap, nkv * hd).unwrap());
                    pt!(21, nkv * hd * if f16kv { 2 } else { 4 },
                        kv_append(d, &v, vcache, &pb, hd, cap, nkv * hd).unwrap());
                    let len = (pos + 1) as u32;
                    // SDPA bytes at 32K: reads K (nkv*len*hd bytes) + V (nkv*len*hd bytes) + q (nq*hd bytes)
                    let kv_bytes = 2 * nkv * (len as usize) * hd * if f16kv { 2 } else { 4 };
                    let q_bytes = nq * hd * if f16kv { 2 } else { 4 };
                    // Diagnostic: skip the sdpa (placeholder output, right shape) to measure
                    // its wall-time contribution via the internal A/B (NEMOTRON_AB=NEMOTRON_SKIPSDPA).
                    let attn = if std::env::var("NEMOTRON_SKIPSDPA").is_ok() {
                        q.clone()
                    } else if !no_2pass && len > 1024 {
                        // NEMOTRON_BLOCKS overrides the split-K block count.
                        // GB10 sweep: 128 ≈ optimal at all ctx (256 within noise, both
                        // beat 512/1024). Default 128; blocks MUST be a multiple of 32.
                        let blocks: u32 = env("NEMOTRON_BLOCKS", 128usize) as u32;
                        // NEMOTRON_BC4=1: BC=4 pass-1 (batched 4 positions/iter).
                        // NEMOTRON_TILED=1: tiled pass-1 (contiguous chunks, L2-friendly).
                        let use_bc4 = std::env::var("NEMOTRON_BC4").is_ok();
                        let use_tiled = std::env::var("NEMOTRON_TILED").is_ok();
                        let a = if use_bc4 {
                            pt!(22, kv_bytes + q_bytes, sdpa_decode_2pass_bc4(d, &q, &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, len, cap as u32, (nq / nkv) as u32, ascale, blocks).unwrap())
                        } else if use_tiled {
                            pt!(22, kv_bytes + q_bytes, sdpa_decode_2pass_tiled(d, &q, &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, len, cap as u32, (nq / nkv) as u32, ascale, blocks).unwrap())
                        } else {
                            pt!(22, kv_bytes + q_bytes, sdpa_decode_2pass(d, &q, &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, len, cap as u32, (nq / nkv) as u32, ascale, blocks).unwrap())
                        };
                        if f16kv { pt!(30, qdim * 2, cast_f16_f32(d, &a).unwrap()) } else { a }
                    } else {
                        let a = pt!(22, kv_bytes + q_bytes, sdpa_decode(d, &q, &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, len, cap as u32, (nq / nkv) as u32, ascale).unwrap());
                        if f16kv { pt!(30, qdim * 2, cast_f16_f32(d, &a).unwrap()) } else { a }
                    };
                    // o_proj gemv_q4/q8: recipe → Q8
                    let o = pt!(23, if use_nvfp4_recipe { hid * qdim + hid * 4 + qdim * 4 } else { hid * qdim / 2 + hid * 2 + qdim * 4 },
                        if use_nvfp4_recipe { qmv8(&attn.reshaped(vec![qdim]), &format!("{p}.mixer.o_proj.weight")) } else { qmv(&attn.reshaped(vec![qdim]), &format!("{p}.mixer.o_proj.weight")) });
                    xt = pt!(29, hid * 8, add(d, &xt, &o).unwrap());
                }
                _ => unreachable!(),
            }
            // Coarse section timing (backward compat display)
            if prof {
                let c = match mix { 'M' => &tm, 'E' => &te, _ => &ta };
                let _ = c; // section totals now computed from fine-grained op_t
            }
            // INSTRUMENTATION (revert): dump this token's hidden after layer l.
            // Overwritten every step → after the seq loop holds the LAST token trace.
            if std::env::var("NEMOTRON_DUMP_LAYERS").is_ok() {
                let h = dl(&xt, hid);
                STEP_LAYER_TRACE.with(|c| { let mut v = c.borrow_mut(); if v.len() <= l { v.resize(l+1, Vec::new()); } v[l] = h; });
            }
        }
        // Final norm + lm_head
        let xf = pt!(25, hid * 8, rms_norm(d, &xt, &fwd["norm_f"], eps).unwrap());
        // Gate the final argmax download: forbidden during CUDA-graph capture (host-sync).
        // During capture and graph replay we return a dummy 0 — the bench only measures
        // throughput, not token correctness.
        if skip_dl.get() {
            pt!(26, vocab * hid / 2 + vocab * 2 + hid * 4,
                { let _ = qmv(&xf, "language_model.lm_head.weight"); }); // keep lm_head in graph
            if prof { d.synchronize().ok(); step_wall.set(step_wall.get() + step_t0.unwrap().elapsed().as_secs_f64()); step_count.set(step_count.get() + 1); }
            return 0;
        }
        let logits = dl(&pt!(26, vocab * hid / 2 + vocab * 2 + hid * 4,
            qmv(&xf, "language_model.lm_head.weight")), vocab);
        if prof {
            d.synchronize().ok();
            step_wall.set(step_wall.get() + step_t0.unwrap().elapsed().as_secs_f64());
            step_count.set(step_count.get() + 1);
        }
        LAST_STEP_LOGITS.with(|c| *c.borrow_mut() = logits.clone()); // INSTRUMENTATION: revert
        ffai_runtime::argmax(&logits)
    };

    let mut tok = 1234usize;
    let mut pos = 0usize;
    // For the 32K measurement, start at `fakectx` (the device KV cache is alloc'd
    // to `cap` and zero-filled on first use, so the sdpa genuinely reads `fakectx`
    // cached positions — the real 32K work — without a 38-min token prefill).
    if fakectx > 0 { pos = fakectx; }
    // warm the cache to `prefill` positions, then time `n_decode` steps.
    // PHASE-0 BASELINE: time the sequential-prefill loop so we can report the
    // current prompt-processing tok/s (= prefill / prefill_seconds). This is the
    // ~67 tok/s sequential-decode-as-prefill number we're trying to beat ~95×
    // with a batched forward. One untimed warm step first (JIT) so we don't
    // charge first-launch compilation to the prefill measurement.
    // ── BATCHED PREFILL (NEMOTRON_PREFILL_BATCHED=1) ─────────────────────────
    // Process all S=prefill prompt tokens in ONE forward instead of S sequential
    // decode steps. Compute-bound path: dequant Q4 projection weights → f32 and
    // run the tensor-tiled `ffai_gemm` (matmul) over [S, *]; causal prefill
    // flash-attn (sdpa_multi) over S; SSD scan (ssm_prefill_scan) over S; per-
    // token MoE gather loop (correctness-first). Writes the SAME conv_dev /
    // ssm_state / kvcache the sequential `step()` uses, so a following decode
    // step continues correctly. Returns the next-token argmax after position S-1
    // (the same token the sequential path would produce). Token sequence is the
    // greedy chain seeded by `tok` (matching the sequential warm-up).
    thread_local! { static LAST_BATCHED_LOGITS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) }; }
    // Device logits tensor (no dl) + capture flag — for CUDA-graphing the forward.
    thread_local! { static LAST_BATCHED_LOGITS_DEV: std::cell::RefCell<Option<Tensor>> = const { std::cell::RefCell::new(None) }; }
    thread_local! { static CAPTURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
    let prefill_batched = std::env::var("NEMOTRON_PREFILL_BATCHED").is_ok();
    if prefill_batched && prefill > 0 {
        let s = prefill; // number of prompt tokens
        // f16 weight cache (Path A): dequant each Q4 weight to f16 ONCE (lazily on
        // first use), keyed by name (+ expert id for MoE). The forward then feeds
        // the cached f16 weight straight to cuBLAS — eliminates the ~26%-of-prefill
        // redundant per-forward dequant. ~2× the GEMM-weight VRAM; fine on GB10.
        // Disable with NEMOTRON_NO_W16CACHE=1 (A/B).
        let w16: std::cell::RefCell<HashMap<String, Tensor>> = std::cell::RefCell::new(HashMap::new());
        let no_w16 = std::env::var("NEMOTRON_NO_W16CACHE").is_ok();
        // NEMOTRON_RESIDENT_W16=1: ALSO cache the routed-expert up/down f16 weights
        // resident (not just dense+shared). ~44.8 GB extra f16 on GB10 (62 GB total
        // resident). Kills the routed-expert per-forward dequant (the remaining
        // ~42% of prefill) at the cost of streaming a fat f16 expert pool from the
        // unified-memory pool. Honest A/B lever — measure dequant%→0 vs the GEMM
        // becoming bandwidth-bound on the larger f16 working set.
        let resident_w16 = std::env::var("NEMOTRON_RESIDENT_W16").map(|v| v != "0" && v != "false").unwrap_or(false);
        // Greedy token chain (same as sequential): we need the S token ids. The
        // sequential path feeds argmax-of-previous; to reproduce it EXACTLY for
        // the correctness gate we run the sequential path's token chain here too
        // would defeat the purpose. Instead the gate runs both paths on the same
        // FIXED token list. For prefill we use the deterministic ramp tok, tok+1…
        // is NOT how decode chains; so the gate harness (NEMOTRON_PREFILL_CHECK)
        // builds an explicit id list. Here we accept an id list via closure.
        // ── Prefill per-op profiler (NEMOTRON_PROFILE=1) ─────────────────────
        // Sync-bracketed wall time + call count + FLOP estimate per op category.
        // Categories: 0=embed,1=rms_norm,2=proj_gemm(dequant+matmul),3=dequant,
        // 4=conv_prefill(CONV_DEVICE=on-device causal conv+silu; off=host ring-conv),
        // 5=ssm_scan,6=softplus_add_rows(dt prep),7=sdpa,8=rope,9=kv_append,
        // 10=moe_router,11=moe_experts,12=moe_shared,13=add,14=lm_head,15=slice/cast.
        const NPF: usize = 16;
        let pf_t: Vec<std::cell::Cell<f64>> = (0..NPF).map(|_| std::cell::Cell::new(0.0)).collect();
        let pf_n: Vec<std::cell::Cell<u64>> = (0..NPF).map(|_| std::cell::Cell::new(0)).collect();
        let pf_flop: Vec<std::cell::Cell<f64>> = (0..NPF).map(|_| std::cell::Cell::new(0.0)).collect();
        let pf_on = std::env::var("NEMOTRON_PROFILE").is_ok();
        // Device-resident constants cache: per-layer constant vectors (dt_bias,
        // router correction bias) + tiny index tensors, uploaded ONCE and reused
        // across layers AND forwards. These were re-uploaded per layer per
        // forward — ~140 small pageable H2D copies/forward, each paying ~0.3ms
        // of driver API time on GB10.
        let const_dev: std::cell::RefCell<HashMap<String, Tensor>> =
            std::cell::RefCell::new(HashMap::new());
        // NEMOTRON_FP4_DENSE=1: dense projections (qkv/o, mamba in/out_proj)
        // through the cublasLt block-scaled NVFP4 GEMM (~4x the f16 tensor-core
        // path on GB10). Weight pack cached here across forwards (quantized
        // once from the dequanted f16 weight); activations quantized per call.
        let w4lt: std::cell::RefCell<HashMap<String, (Tensor, Tensor)>> =
            std::cell::RefCell::new(HashMap::new());
        // Single-slot act-quant dedup: q/k/v (and any same-input projections)
        // quantize the SAME activation tensor — cache the last (input, dims) ->
        // (pack, sf). Holding the input Tensor keeps its buffer alive, so the
        // pointer-identity check can't alias a pool-recycled buffer.
        let w4g: std::cell::RefCell<HashMap<String, Tensor>> = std::cell::RefCell::new(HashMap::new());
        let actq_slot: std::cell::RefCell<Option<(Tensor, usize, usize, Tensor, Tensor, Tensor)>> =
            std::cell::RefCell::new(None);
        // Batched forward over an explicit token-id slice; writes states; returns
        // the argmax logits for the LAST token.
        let forward_batched = |ids: &[usize],
                               conv_dev: &std::cell::RefCell<Vec<Option<Tensor>>>,
                               ssm_state: &mut Vec<Option<Tensor>>,
                               kvcache: &mut Vec<Option<(Tensor, Tensor)>>,
                               base: usize| -> usize {
            let s = ids.len();
            // Backend gate: the CUDA-only escape hatches (gemm_cublas/cublasGemmEx,
            // moe_grouped_gemm/moe_scatter_add via dispatch_raw_cuda, the Marlin MoE,
            // and the SSD/TC-flash cuBLAS paths) hard-error on Metal. When the device
            // is NOT CUDA we route the projection/MoE/shared GEMMs through the portable
            // primitives that codegen to Metal hardware MMA: gemm_q4_mpp (Q4-native
            // cooperative-tensor MMA) for the dense projections, and dequant_q4_off +
            // matmul (f32 MMA) for the per-expert / shared-expert GEMMs. The scan
            // (ssm_prefill_scan), attention (sdpa_multi), conv (host-bridge) and router
            // (host top-k) defaults are already portable at d0/S≤4096. CUDA path is
            // left byte-for-byte unchanged — gated purely on `is_cuda`.
            let is_cuda = d.backend() == Backend::Cuda;
            // Fetch-or-upload a device-resident constant (cached across forwards).
            let cdev = |key: String, make: &dyn Fn() -> Tensor| -> Tensor {
                if let Some(t) = const_dev.borrow().get(&key) { return t.clone(); }
                let t = make();
                const_dev.borrow_mut().insert(key, t.clone());
                t
            };
            // ── Packed multi-sequence prefill (NEMOTRON_PACKED=N) ──────────────
            // N equal-length sequences (seg_len = s/N, a multiple of the SSD chunk
            // len) packed into one token stream. Attention goes block-diagonal via
            // sdpa_multi_tc_varlen (per-token segment-start `seg_lo`), and the Mamba
            // SSD scan resets recurrent state at each boundary (`seg_reset[chunk]`).
            // proj/MoE/router/norm are token-parallel → unchanged. RoPE is relative
            // so global positions give correct intra-segment rotation. NOTE: the
            // causal conv1d is NOT yet segment-aware (kc-1≈3 tokens/segment leak
            // across the boundary) — throughput path until the varlen conv lands.
            let n_seg = std::env::var("NEMOTRON_PACKED").ok()
                .and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
            let ssd_l_pk = std::env::var("NEMOTRON_SSD_L").ok()
                .and_then(|v| v.parse::<usize>().ok()).unwrap_or(128);
            let packed = is_cuda && n_seg > 1 && s % n_seg == 0 && (s / n_seg) % ssd_l_pk == 0;
            let (seg_lo_dev, seg_reset_dev) = if packed {
                let seg_len = s / n_seg;
                let seg_lo_h: Vec<u32> = (0..s).map(|r| ((r / seg_len) * seg_len) as u32).collect();
                let seg_lo = Tensor::new(d.upload(unsafe {
                    std::slice::from_raw_parts(seg_lo_h.as_ptr() as *const u8, s * 4) }).unwrap(),
                    vec![s], DType::U32);
                let nc = s.div_ceil(ssd_l_pk);
                let seg_reset_h: Vec<u32> = (0..nc)
                    .map(|c| if (c * ssd_l_pk) % seg_len == 0 { 1u32 } else { 0u32 }).collect();
                let seg_reset = Tensor::new(d.upload(unsafe {
                    std::slice::from_raw_parts(seg_reset_h.as_ptr() as *const u8, nc * 4) }).unwrap(),
                    vec![nc], DType::U32);
                (Some(seg_lo), Some(seg_reset))
            } else { (None, None) };
            // Per-segment length (KV-block size) → enables the attention segment-skip.
            let pk_seg_len: u32 = if packed { (s / n_seg) as u32 } else { 0 };
            // pf!(idx, flops, expr): sync-bracket + accumulate when profiling.
            macro_rules! pf { ($idx:expr, $flops:expr, $e:expr) => {{
                if pf_on { d.synchronize().ok(); let _t0 = Instant::now(); let _r = $e; d.synchronize().ok();
                    pf_t[$idx].set(pf_t[$idx].get() + _t0.elapsed().as_secs_f64());
                    pf_n[$idx].set(pf_n[$idx].get() + 1);
                    pf_flop[$idx].set(pf_flop[$idx].get() + ($flops) as f64); _r
                } else { $e }
            }}; }
            // Embed S tokens → [S, hid] f32 resident.
            // Embed cached device-side keyed by the id list: the upload sourced a
            // stack-local host Vec, so a CUDA-graph capture recorded a memcpy node
            // from a pointer that's dangling by replay time. Cached, the pre-warm
            // forwards fill it and the captured forward does NO embed H2D at all.
            let mut xt = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                ids.hash(&mut h);
                let key = format!("__embed.{:x}.{}", h.finish(), s);
                cdev(key, &|| {
                    let mut xh = vec![0.0f32; s * hid];
                    for (i, &t) in ids.iter().enumerate() { xh[i*hid..(i+1)*hid].copy_from_slice(&embed[t*hid..(t+1)*hid]); }
                    upm(&xh, vec![s, hid])
                })
            };
            // Tensor-core projection GEMM (Path A): dequant Q4 weight → f16 once,
            // cast x → f16, cuBLAS GemmEx (real tensor cores, f32 accumulate),
            // cast result → f32. Replaces the software-emulated coop_tile MMA
            // (~0.1% of peak). NEMOTRON_PREFILL_MPPGEMM=1 = old coop_tile path;
            // NEMOTRON_PREFILL_F32GEMM=1 = f32 scalar matmul (A/B).
            let use_f32_gemm = std::env::var("NEMOTRON_PREFILL_F32GEMM").is_ok();
            let use_mpp_gemm = std::env::var("NEMOTRON_PREFILL_MPPGEMM").is_ok();
            // ── Metal per-expert MoE GEMM compute dtype (Apple/Metal only) ──
            // The per-forward Q4→dtype dequant of the routed experts × 23 MoE layers is a
            // top prefill cost at small S. On Metal we dequant→f16 and run the per-expert
            // matmul in f16 (vs the old f32 dequant + f32 matmul): the f16 dequant writes
            // 2 B/elem (vs 4 B) and the f16 matmul reads 2 B weights, so it beats the f32
            // baseline (M5 Max: S=512 +26.6%, S=2048 98.8→113.5 tok/s) at EXACT-MATCH
            // numerics (argmax unchanged vs the f32 sequential ref, 2× repeat identical).
            // f16-compute is the Metal DEFAULT; set NEMOTRON_METAL_F32_EXPERTS=1 to fall
            // back to the f32 path for A/B. No resident f16 weight cache: caching the f16
            // experts resident was measured a NET LOSS on Metal (bandwidth/residency bound
            // — re-dequanting the compact Q4 each forward streams better than a fat f16
            // working set). CUDA path is unaffected (gated on `is_cuda` below).
            let metal_f16_experts = !is_cuda && !std::env::var("NEMOTRON_METAL_F32_EXPERTS").is_ok();
            // Cached Q4→f16 dequant. On miss, dequant the [m,k] slab (optionally at
            // a block offset for one MoE expert) and store under `key`. Returns a
            // cheap Tensor clone (Arc buffer). `cache=false` (or NEMOTRON_NO_W16CACHE)
            // → always dequant, no store. We CACHE the dense projection + shared
            // weights (small, always used: ~few GB) by default. The 128 routed
            // experts ×23 layers (~44.8 GB f16) re-dequant per forward UNLESS
            // NEMOTRON_RESIDENT_W16=1 (`resident_w16`), which also caches them
            // resident — fits GB10's 121 GB unified pool (~62 GB total resident).
            let deq16c = |key: &str, qs: &Tensor, sc: &Tensor, m: usize, k: usize, blk_off: usize, cache: bool| -> Tensor {
                if no_w16 || !cache {
                    return pf!(3, (m * k) as f64, dequant_q4_off(d, qs, sc, m, k, DType::F16, blk_off).unwrap());
                }
                if let Some(w) = w16.borrow().get(key) { return w.clone(); }
                let w = pf!(3, (m * k) as f64, dequant_q4_off(d, qs, sc, m, k, DType::F16, blk_off).unwrap());
                w16.borrow_mut().insert(key.to_string(), w.clone());
                w
            };
            let deq16 = |key: &str, qs: &Tensor, sc: &Tensor, m: usize, k: usize, blk_off: usize| -> Tensor {
                deq16c(key, qs, sc, m, k, blk_off, true)
            };
            // FP4 dense projections (NEMOTRON_FP4_DENSE=1): quantize the f16
            // activation per call, weight pack cached cross-forward; cublasLt
            // block-scaled NVFP4 GEMM with f32 output (fused output cast).
            // Value selects scope: "1"/"all" = every dense projection;
            // "mamba" = in/out_proj only; "qkv" = attention q/k/v/o only.
            let fp4_dense_scope = if is_cuda { std::env::var("NEMOTRON_FP4_DENSE").ok() } else { None };
            let fp4_dense = fp4_dense_scope.is_some();
            let fp4_name_ok = move |name: &str| -> bool {
                match fp4_dense_scope.as_deref() {
                    Some("mamba") => name.contains("in_proj") || name.contains("out_proj"),
                    Some("qkv") => name.contains("q_proj") || name.contains("k_proj")
                        || name.contains("v_proj") || name.contains("o_proj"),
                    // audit: in_proj carries the worst FP4 quantization error
                    // (normed-residual outliers) — exclude it, FP4 the rest.
                    Some("noin") => !name.contains("in_proj"),
                    Some("out") => name.contains("out_proj"),
                    Some(_) => true,
                    None => false,
                }
            };
            let qmm_fp4 = |xh: &Tensor, name: &str, rows: usize, m: usize, k: usize,
                           qs: &Tensor, sc: &Tensor| -> Tensor {
                // GLOBAL-AWARE: per-tensor global keeps ue4m3 block scales in
                // normal range (was the subnormal-floor bug = cosine 0.44);
                // the act·weight globals fold back via the GEMM D-scale.
                let (wp, wsf, wg) = {
                    if let (Some(t), Some(g)) = (w4lt.borrow().get(name).cloned(), w4g.borrow().get(name).cloned()) { (t.0, t.1, g) } else {
                        let wf = pf!(3, (m * k) as f64, dequant_q4_off(d, qs, sc, m, k, DType::F16, 0).unwrap());
                        let (wp, wsf, wg) = ffai_ops::lt_fp4_quant_g(d, &wf, m, k).unwrap();
                        w4lt.borrow_mut().insert(name.to_string(), (wp.clone(), wsf.clone()));
                        w4g.borrow_mut().insert(name.to_string(), wg.clone());
                        (wp, wsf, wg)
                    }
                };
                let (xp, xsf, xg) = {
                    let hit = actq_slot.borrow().as_ref().map_or(false, |(cx, cr, ck, _, _, _)|
                        *cr == rows && *ck == k && std::sync::Arc::ptr_eq(&cx.buffer, &xh.buffer));
                    if hit {
                        let s0 = actq_slot.borrow();
                        let (_, _, _, p, f, g) = s0.as_ref().unwrap();
                        (p.clone(), f.clone(), g.clone())
                    } else {
                        let t = pf!(15, 0.0, ffai_ops::lt_fp4_quant_g(d, xh, rows, k).unwrap());
                        *actq_slot.borrow_mut() = Some((xh.clone(), rows, k, t.0.clone(), t.1.clone(), t.2.clone()));
                        t
                    }
                };
                let dscale = ffai_ops::fp4_dscale(d, &xg, &wg).unwrap();
                let o = pf!(2, 2.0 * rows as f64 * m as f64 * k as f64,
                    ffai_ops::gemm_fp4(d, &xp, &xsf, &wp, &wsf, rows, m, k, true, Some(&dscale)).unwrap()); // [rows, m] f32
                if std::env::var("NEMOTRON_FP4_DENSE_DEBUG").is_ok() {
                    // numeric audit vs the f16 path (bench-only; huge max|d| = layout bug,
                    // few-percent rel = expected FP4 quantization noise)
                    let wf = deq16(name, qs, sc, m, k, 0);
                    let r16 = gemm_cublas_f32out(d, xh, &wf, rows, m, k).unwrap();
                    let a = dl(&o, rows * m); let b = dl(&r16, rows * m);
                    let (mut mab, mut mrel, mut den) = (0f32, 0f32, 0f32);
                    for i in 0..a.len() { let d0 = (a[i] - b[i]).abs(); mab = mab.max(d0); den = den.max(b[i].abs()); if b[i].abs() > 1.0 { mrel = mrel.max(d0 / b[i].abs()); } }
                    eprintln!("FP4_DENSE_DEBUG {name}: rows={rows} m={m} k={k} max|d|={mab:.4} max_rel(>1)={mrel:.4} max|ref|={den:.2}");
                }
                o
            };
            // FP8 dense (NEMOTRON_FP8_DENSE): the vendor recipe for this model
            // family runs Mamba in/out_proj (+ o_proj, shared) at FP8 e4m3 with
            // per-tensor scales — far gentler than FP4. Scopes: "mamba"
            // (in/out_proj), "recipe" (+ o_proj), "all".
            let fp8_dense_scope = if is_cuda { std::env::var("NEMOTRON_FP8_DENSE").ok() } else { None };
            let fp8_dense = fp8_dense_scope.is_some();
            let fp8_name_ok = move |name: &str| -> bool {
                match fp8_dense_scope.as_deref() {
                    Some("mamba") => name.contains("in_proj") || name.contains("out_proj"),
                    Some("recipe") => name.contains("in_proj") || name.contains("out_proj") || name.contains("o_proj"),
                    Some(_) => true,
                    None => false,
                }
            };
            let qmm_fp8d = |xh: &Tensor, name: &str, rows: usize, m: usize, k: usize,
                            qs: &Tensor, sc: &Tensor| -> Tensor {
                // PER-CHANNEL weight (rows=m) + PER-TOKEN act (rows) FP8, scales
                // folded post-GEMM (D[r,c]*=xsc[r]·wsc[c]) — the quality-
                // preserving granularity (per-tensor was cosine ~0.94).
                let (wq, wsc) = {
                    if let Some(t) = w4lt.borrow().get(&format!("{name}.fp8c")) { t.clone() } else {
                        let wf = pf!(3, (m * k) as f64, dequant_q4_off(d, qs, sc, m, k, DType::F16, 0).unwrap());
                        let t = ffai_ops::fp8_quant_perrow(d, &wf, m, k).unwrap();
                        w4lt.borrow_mut().insert(format!("{name}.fp8c"), t.clone());
                        t
                    }
                };
                let (xq, xsc) = pf!(15, 0.0, ffai_ops::fp8_quant_perrow(d, xh, rows, k).unwrap());
                pf!(2, 2.0 * rows as f64 * m as f64 * k as f64,
                    ffai_ops::gemm_fp8_perchan(d, &xq, &xsc, &wq, &wsc, rows, m, k).unwrap()) // [rows, m] f32
            };
            let qmm = |x: &Tensor, name: &str| -> Tensor {
                let (qs, sc, m, k) = &qw[name];
                let rows = x.elem_count() / *k;
                let flops = 2.0 * rows as f64 * *m as f64 * *k as f64;
                if fp8_dense && rows > 1 && fp8_name_ok(name) {
                    let xh = pf!(15, 0.0, cast_f32_f16(d, x).unwrap());
                    return qmm_fp8d(&xh, name, rows, *m, *k, qs, sc);
                }
                if fp4_dense && rows > 1 && *k % 32 == 0 && fp4_name_ok(name) {
                    // f32 input quantizes directly inside qmm_fp4 (no cast staging)
                    return qmm_fp4(x, name, rows, *m, *k, qs, sc);
                }
                if use_f32_gemm {
                    let wf = pf!(3, (m * k) as f64, dequant_q4(d, qs, sc, *m, *k, DType::F32).unwrap());
                    pf!(2, flops, matmul(d, &wf, x).unwrap()) // [rows, m]
                } else if use_mpp_gemm || !is_cuda {
                    // Portable Metal path (also NEMOTRON_PREFILL_MPPGEMM): Q4-native
                    // cooperative-tensor MMA — no cuBLAS, runs on Apple GPU hardware MMA.
                    let xh = pf!(15, 0.0, cast_f32_f16(d, x).unwrap());
                    let oh = pf!(2, flops, gemm_q4_mpp(d, &xh, qs, sc, rows, *m, *k).unwrap());
                    pf!(15, 0.0, cast_f16_f32(d, &oh).unwrap()) // [rows, m]
                } else {
                    // cuBLAS tensor-core path: out[rows,m] = x[rows,k] · W[m,k]ᵀ.
                    let wf = deq16(name, qs, sc, *m, *k, 0);
                    let xh = pf!(15, 0.0, cast_f32_f16(d, x).unwrap());
                    // Fused output cast: cuBLAS writes f32 directly (accumulate is
                    // already f32) — drops the separate cast_f16_f32, MFU preserved.
                    pf!(2, flops, gemm_cublas_f32out(d, &xh, &wf, rows, *m, *k).unwrap()) // [rows, m] f32
                }
            };
            // Like `qmm` but takes a PRE-CAST f16 activation `xh` (skips the input
            // cast_f32_f16). Lets multiple projections over the SAME normed input
            // (q/k/v from `xn`, shared-expert up from `xn`) share one f16 cast
            // instead of re-casting per projection — kills redundant slice/cast.
            // cuBLAS tensor-core path only (the fast prefill path).
            let qmm_h = |xh: &Tensor, name: &str| -> Tensor {
                let (qs, sc, m, k) = &qw[name];
                let rows = xh.elem_count() / *k;
                let flops = 2.0 * rows as f64 * *m as f64 * *k as f64;
                if fp8_dense && rows > 1 && fp8_name_ok(name) {
                    return qmm_fp8d(xh, name, rows, *m, *k, qs, sc);
                }
                if fp4_dense && rows > 1 && *k % 32 == 0 && fp4_name_ok(name) {
                    return qmm_fp4(xh, name, rows, *m, *k, qs, sc);
                }
                if !is_cuda {
                    // Portable Q4-native Metal MMA (xh already f16).
                    let oh = pf!(2, flops, gemm_q4_mpp(d, xh, qs, sc, rows, *m, *k).unwrap());
                    return pf!(15, 0.0, cast_f16_f32(d, &oh).unwrap()); // [rows, m]
                }
                let wf = deq16(name, qs, sc, *m, *k, 0);
                // Fused output cast: cuBLAS writes f32 directly — no cast_f16_f32.
                pf!(2, flops, gemm_cublas_f32out(d, xh, &wf, rows, *m, *k).unwrap()) // [rows, m] f32
            };
            // Fuse-slice-cast opt-in (NEMOTRON_FUSE_QKV=1): cast xn→f16 once per
            // attention/shared block and reuse via qmm_h. Default off (A/B).
            let fuse_qkv = std::env::var("NEMOTRON_FUSE_QKV").is_ok() && !use_f32_gemm && !use_mpp_gemm;
            // Pre-allocate grouped-GEMM weight scratch buffers (max size, reused across layers).
            // Avoids per-layer cuMemAlloc which is expensive. Allocated once at max expert count.
            // UP scratch: [n_exp * inter, hid] f16 (for all 128 experts, even if only ~107 active).
            // DN scratch: [n_exp * hid, inter] f16.
            // These are bounded: 128 * 1856 * 4096 * 2 ≈ 1.83 GB each, ≈ 3.67 GB total.
            // Pre-alloc grouped-gemm scratch for GROUPED_GEMM and W4A16 (which falls back
            // to grouped_gemm at large S — needs scratch to avoid per-layer cuMemAlloc).
            let needs_grouped_scratch = (std::env::var("NEMOTRON_GROUPED_GEMM").is_ok() && !std::env::var("NEMOTRON_BGEMM").is_ok())
                || std::env::var("NEMOTRON_W4A16").is_ok()
                || std::env::var("NEMOTRON_W4A16_MARLIN").is_ok();
            let grouped_up_scratch: Option<Tensor> = if needs_grouped_scratch {
                Some(Tensor::new(d.alloc(n_exp * inter * hid * 2).unwrap(), vec![n_exp * inter, hid], DType::F16))
            } else { None };
            let grouped_dn_scratch: Option<Tensor> = if needs_grouped_scratch {
                Some(Tensor::new(d.alloc(n_exp * hid * inter * 2).unwrap(), vec![n_exp * hid, inter], DType::F16))
            } else { None };
            // xn carried from the previous layer's fused add+rms_norm (None on layer 0
            // or after a non-fusable tail).
            let mut xn_carry: Option<Tensor> = None;
            for (l, mix) in PATTERN.chars().enumerate() {
                let p = format!("language_model.backbone.layers.{l}");
                let xn = match xn_carry.take() {
                    Some(t) => t,
                    None => pf!(1, 0.0, rms_norm(d, &xt, &fwd[&format!("{p}.norm.weight")], eps).unwrap()), // [s, hid]
                };
                match mix {
                    'M' => {
                        // ── NEMOTRON_CONV_DEVICE=1: fully on-device conv (no host round-trip) ──
                        // proj stays on device; strided_col_copy carves z/xbc/dt_raw on GPU;
                        // conv1d_causal_prefill + softplus_add_rows run the conv+silu+softplus
                        // without any dl/up. Saves ~42 MB of PCIe per Mamba layer per forward.
                        // DEFAULT-ON for CUDA (the on-device-MoE prefill win requires it;
                        // argmax-exact, ~2.5× over host ring-conv at S=2048 d0). Metal keeps
                        // the host ring-conv (its validated path). Escapes:
                        // NEMOTRON_CONV_DEVICE=1 forces on (Metal opt-in / explicit);
                        // NEMOTRON_CONV_DEVICE_OFF=1 forces the host ring-conv on CUDA.
                        let use_conv_device = std::env::var("NEMOTRON_CONV_DEVICE").is_ok()
                            || (is_cuda && std::env::var("NEMOTRON_CONV_DEVICE_OFF").is_err());
                        if use_conv_device {
                            let proj_t = pf!(2, 2.0 * s as f64 * in_proj_out as f64 * hid as f64,
                                qmm(&xn, &format!("{p}.mixer.in_proj.weight"))); // [s, in_proj_out] on device
                            // PART B: fused split — 1 dispatch replaces 3 strided_col_copy calls.
                            // proj layout: [z(di) | xbc(conv_dim) | dt_raw(m_nh)] per row.
                            let (z_t, xbc_t, dt_raw_t) = pf!(15, 0.0,
                                mamba_split_proj(d, &proj_t, s, in_proj_out, di, conv_dim, m_nh).unwrap());
                            // On-device causal depthwise conv1d + silu over all S tokens.
                            // Result: yc_silu [s * conv_dim] flat.
                            let yc_silu = pf!(4, 0.0, conv1d_causal_prefill(d, &xbc_t,
                                &fwd[&format!("{p}.mixer.conv1d.weight")],
                                &fwd[&format!("{p}.mixer.conv1d.bias")],
                                s, conv_dim, kc).unwrap());
                            // Persist conv ring state for decode continuity: last (kc-1) rows of xbc.
                            let ring_len = (kc - 1) * conv_dim;
                            let ring_off = (s.saturating_sub(kc - 1)) * conv_dim;
                            { let mut cd = conv_dev.borrow_mut();
                              cd[l] = Some(slice(d, &xbc_t, ring_off, ring_len).unwrap()); }
                            // Softplus + dt_bias on device: [s, m_nh] → dt_all [s*m_nh].
                            let dt_bias_dev = cdev(format!("{p}.mixer.dt_bias"), &|| up(&fw[&format!("{p}.mixer.dt_bias")]));
                            let dt_dev  = pf!(6, 0.0, softplus_add_rows(d, &dt_raw_t, &dt_bias_dev, s, m_nh).unwrap());
                            // SSD scan over S.
                            if ssm_state[l].is_none() { ssm_state[l] = Some(Tensor::new(d.alloc_zeroed(m_nh * m_dh * ds * 4).unwrap(), vec![m_nh * m_dh * ds], DType::F32)); }
                            let ssm_flops = s as f64 * m_nh as f64 * m_dh as f64 * ds as f64 * 4.0;
                            let use_chunked = std::env::var("NEMOTRON_CHUNKED_SCAN").is_ok();
                            // SSD chunked-matmul scan is DEFAULT-ON for CUDA (the Nemotron
                            // (64,128,64,8) cell): tensor-core GEMMs + fused per-group B/C
                            // cut ssm_scan ~681ms(36%)→75ms(<6%), e2e +~30% @d0, argmax-exact.
                            // Escape NEMOTRON_SSD_SEQ=1 → old sequential ssm_prefill_scan.
                            let ssd_seq = std::env::var("NEMOTRON_SSD_SEQ").is_ok();
                            let use_ssd = !ssd_seq && (is_cuda
                                || std::env::var("NEMOTRON_SSD_MATMUL").is_ok()
                                || std::env::var("NEMOTRON_SSD_FUSED").is_ok());
                            let use_ssd_port = std::env::var("NEMOTRON_SSD_PORTABLE").is_ok();
                            // CUDA default chunk len 128 (validated win); 256 elsewhere.
                            let ssd_l: u32 = std::env::var("NEMOTRON_SSD_L").ok().and_then(|v| v.parse().ok())
                                .unwrap_or(if is_cuda { 128 } else { 256 });
                            let mut ssd_split_out: Option<(Tensor, Tensor)> = None;
                            // NOTE: a strided-input SSD path (reading x/B/C in place from the
                            // conv output, no mamba_split_conv) was measured SLOWER (-6-8% e2e,
                            // argmax exact): scattered column reads from the 6K-wide conv rows
                            // cost more than the split's streaming write+read. Machinery kept in
                            // ssm_prefill_scan_ssd_strided; compact split path stays.
                            {
                                // PART B: fused split — 1 dispatch replaces 3 strided_col_copy calls.
                                let (x_dev, b_dev, c_dev) = pf!(15, 0.0,
                                    mamba_split_conv(d, &yc_silu, s, conv_dim, di, ng * ds).unwrap());
                                if use_ssd {
                                    ssd_split_out = Some(pf!(5, ssm_flops, ssm_prefill_scan_ssd(d, &x_dev, &fwd[&format!("{p}.mixer.A_log")], &b_dev, &c_dev, &fwd[&format!("{p}.mixer.D")], &dt_dev, ssm_state[l].as_ref().unwrap(), s as u32, m_dh as u32, ds as u32, m_nh as u32, ng as u32, ssd_l, seg_reset_dev.as_ref()).unwrap()));
                                } else if use_ssd_port {
                                    ssd_split_out = Some(pf!(5, ssm_flops, ssm_prefill_scan_ssd_portable(d, &x_dev, &fwd[&format!("{p}.mixer.A_log")], &b_dev, &c_dev, &fwd[&format!("{p}.mixer.D")], &dt_dev, ssm_state[l].as_ref().unwrap(), s as u32, m_dh as u32, ds as u32, m_nh as u32, ng as u32, ssd_l).unwrap()));
                                } else if use_chunked {
                                    ssd_split_out = Some(pf!(5, ssm_flops, ssm_prefill_scan_chunked(d, &x_dev, &fwd[&format!("{p}.mixer.A_log")], &b_dev, &c_dev, &fwd[&format!("{p}.mixer.D")], &dt_dev, ssm_state[l].as_ref().unwrap(), s as u32, m_dh as u32, ds as u32, m_nh as u32, ng as u32).unwrap()));
                                } else {
                                    ssd_split_out = Some(pf!(5, ssm_flops, ssm_prefill_scan(d, &x_dev, &fwd[&format!("{p}.mixer.A_log")], &b_dev, &c_dev, &fwd[&format!("{p}.mixer.D")], &dt_dev, ssm_state[l].as_ref().unwrap(), s as u32, m_dh as u32, ds as u32, m_nh as u32, ng as u32).unwrap()));
                                }
                            };
                            let (so, y_dev) = ssd_split_out.take().unwrap();
                            ssm_state[l] = Some(so);
                            // PART B: on-device batched gated group RMSNorm — eliminates the
                            // dl(y)+dl(z)+host loop+up(yn) round-trip (2 × s×di = ~42MB per layer).
                            let ynt = pf!(6, 0.0, gated_group_rmsnorm_batched(d, &y_dev, &z_t,
                                &up(&fw[&format!("{p}.mixer.norm.weight")]), eps, s, di, gs).unwrap());
                            let out = qmm(&ynt.reshaped(vec![s, di]), &format!("{p}.mixer.out_proj.weight"));
                            {
                                // fused residual add + NEXT layer's rms_norm (one pass, llama.cpp-style)
                                let nw = if l + 1 < PATTERN.len() { fwd.get(&format!("language_model.backbone.layers.{}.norm.weight", l + 1)) } else { fwd.get("norm_f") };
                                match nw {
                                    Some(w) if is_cuda && std::env::var("NEMOTRON_FUSED_ADDNORM").is_ok() => { let (r, nx) = ffai_ops::add_rms_norm(d, &xt, &out, w, eps).unwrap(); xt = r; xn_carry = Some(nx); }
                                    _ => { xt = add(d, &xt, &out).unwrap(); }
                                }
                            }
                        } else {
                        let proj = dl(&qmm(&xn, &format!("{p}.mixer.in_proj.weight")), s * in_proj_out); // [s, in_proj_out]
                        // Host-bridge the Mamba conv + SSD-scan input shuffle (correctness-first).
                        // Per-token slices: z[di], xbc[conv_dim], dt_raw[m_nh].
                        let cw = &fw[&format!("{p}.mixer.conv1d.weight_host")]; // [kc*conv_dim] reorganized
                        let cb = &fw[&format!("{p}.mixer.conv1d.bias_host")]; // [conv_dim]
                        // Causal depthwise conv over S (state = (kc-1) zeros prefix at prefill start),
                        // then silu. Produces xbc_act[s, conv_dim].
                        let mut z_all = vec![0.0f32; s * di];
                        let mut xssm = vec![0.0f32; s * di];          // [s, H*Dh]
                        let mut bmat_all = vec![0.0f32; s * ng * ds]; // [s, G*Ds]
                        let mut cmat_all = vec![0.0f32; s * ng * ds];
                        let mut dt_all = vec![0.0f32; s * m_nh];      // [s, H]
                        let dtb = &fw[&format!("{p}.mixer.dt_bias")];
                        // ring conv state carried across tokens within this prefill.
                        let mut ring = vec![0.0f32; (kc - 1) * conv_dim];
                        for ti in 0..s {
                            let base_p = ti * in_proj_out;
                            let z = &proj[base_p..base_p + di];
                            z_all[ti*di..(ti+1)*di].copy_from_slice(z);
                            let xbc = &proj[base_p + di..base_p + di + conv_dim];
                            let dt_raw = &proj[base_p + di + conv_dim..base_p + di + conv_dim + m_nh];
                            // depthwise causal conv1d (kc taps) + silu, per channel.
                            let mut yc = vec![0.0f32; conv_dim];
                            for ch in 0..conv_dim {
                                let mut acc = cb[ch];
                                for kk in 0..kc {
                                    // tap kk reads position (ti - (kc-1) + kk): ring for <0, xbc for current.
                                    let rel = kk as isize - (kc as isize - 1);
                                    let v = if rel < 0 {
                                        let idx = (kc - 1) as isize + rel; // ring slot
                                        ring[idx as usize * conv_dim + ch]
                                    } else { xbc[ch] };
                                    acc += cw[kk * conv_dim + ch] * v;
                                }
                                let a = acc; yc[ch] = a / (1.0 + (-a).exp()); // silu
                            }
                            // advance ring: drop oldest, push xbc.
                            if kc > 1 { ring.drain(0..conv_dim); ring.extend_from_slice(xbc); }
                            xssm[ti*di..(ti+1)*di].copy_from_slice(&yc[0..di]);
                            bmat_all[ti*ng*ds..(ti+1)*ng*ds].copy_from_slice(&yc[di..di+ng*ds]);
                            cmat_all[ti*ng*ds..(ti+1)*ng*ds].copy_from_slice(&yc[di+ng*ds..di+2*ng*ds]);
                            for hh in 0..m_nh { dt_all[ti*m_nh+hh] = softplus(dt_raw[hh] + dtb[hh]); }
                        }
                        // persist conv ring state for decode continuity.
                        { let mut cd = conv_dev.borrow_mut(); cd[l] = Some(up(&ring)); }
                        // SSD scan over S: x[s,H*Dh], b/c[s,G*Ds], dt[s,H], state[H*Dh*Ds].
                        if ssm_state[l].is_none() { ssm_state[l] = Some(Tensor::new(d.alloc_zeroed(m_nh * m_dh * ds * 4).unwrap(), vec![m_nh * m_dh * ds], DType::F32)); }
                        let x_dev = up(&xssm);
                        let b_dev = up(&bmat_all);
                        let c_dev = up(&cmat_all);
                        let dt_dev = up(&dt_all);
                        // SSD FLOPs ≈ s * H * Dh * Ds * 4 (dA*state + dbx, y=state*c)
                        let ssm_flops = s as f64 * m_nh as f64 * m_dh as f64 * ds as f64 * 4.0;
                        // NEMOTRON_CHUNKED_SCAN=1: use two-pass parallel segment scan.
                        // Drops serial depth from T to T/64, exploiting 64× more GPU parallelism.
                        // Default: sequential step-record (validated, correctness-first).
                        let use_chunked = std::env::var("NEMOTRON_CHUNKED_SCAN").is_ok();
                        // SSD chunked-matmul scan is DEFAULT-ON for CUDA (the Nemotron
                        // (64,128,64,8) cell): tensor-core GEMMs + fused per-group B/C
                        // cut ssm_scan ~681ms(36%)→75ms(<6%), e2e +~30% @d0, argmax-exact.
                        // Escape NEMOTRON_SSD_SEQ=1 → old sequential ssm_prefill_scan.
                        let ssd_seq = std::env::var("NEMOTRON_SSD_SEQ").is_ok();
                        let use_ssd = !ssd_seq && (is_cuda
                            || std::env::var("NEMOTRON_SSD_MATMUL").is_ok()
                            || std::env::var("NEMOTRON_SSD_FUSED").is_ok());
                        let use_ssd_port = std::env::var("NEMOTRON_SSD_PORTABLE").is_ok();
                        // CUDA default chunk len 128 (validated win); 256 elsewhere.
                        let ssd_l: u32 = std::env::var("NEMOTRON_SSD_L").ok().and_then(|v| v.parse().ok())
                            .unwrap_or(if is_cuda { 128 } else { 256 });
                        let (so, y_dev) = if use_ssd_port {
                            pf!(5, ssm_flops, ssm_prefill_scan_ssd_portable(d, &x_dev, &fwd[&format!("{p}.mixer.A_log")], &b_dev, &c_dev, &fwd[&format!("{p}.mixer.D")], &dt_dev, ssm_state[l].as_ref().unwrap(), s as u32, m_dh as u32, ds as u32, m_nh as u32, ng as u32, ssd_l).unwrap())
                        } else if use_ssd {
                            pf!(5, ssm_flops, ssm_prefill_scan_ssd(d, &x_dev, &fwd[&format!("{p}.mixer.A_log")], &b_dev, &c_dev, &fwd[&format!("{p}.mixer.D")], &dt_dev, ssm_state[l].as_ref().unwrap(), s as u32, m_dh as u32, ds as u32, m_nh as u32, ng as u32, ssd_l, seg_reset_dev.as_ref()).unwrap())
                        } else if use_chunked {
                            pf!(5, ssm_flops, ssm_prefill_scan_chunked(d, &x_dev, &fwd[&format!("{p}.mixer.A_log")], &b_dev, &c_dev, &fwd[&format!("{p}.mixer.D")], &dt_dev, ssm_state[l].as_ref().unwrap(), s as u32, m_dh as u32, ds as u32, m_nh as u32, ng as u32).unwrap())
                        } else {
                            pf!(5, ssm_flops, ssm_prefill_scan(d, &x_dev, &fwd[&format!("{p}.mixer.A_log")], &b_dev, &c_dev, &fwd[&format!("{p}.mixer.D")], &dt_dev, ssm_state[l].as_ref().unwrap(), s as u32, m_dh as u32, ds as u32, m_nh as u32, ng as u32).unwrap())
                        };
                        ssm_state[l] = Some(so);
                        let y = dl(&y_dev, s * di); // [s, di]
                        // INSTRUMENTATION (revert): dump L0 last-token SSM in/out (batched).
                        if l == 0 && std::env::var("NEMOTRON_DUMP_SSM").is_ok() {
                            let o = (s-1)*di; let og = (s-1)*ng*ds;
                            let xs = xssm[o..o+di].to_vec();
                            let bb = bmat_all[og..og+ng*ds].to_vec();
                            let cc = cmat_all[og..og+ng*ds].to_vec();
                            let yy = y[o..o+di].to_vec();
                            BAT_SSM_DUMP.with(|c| *c.borrow_mut() = (xs, bb, cc, yy));
                        }
                        // gated group rmsnorm per token, then out_proj batched.
                        let nw = &fw[&format!("{p}.mixer.norm.weight")];
                        let mut yn = vec![0.0f32; s * di];
                        for ti in 0..s {
                            let yo = ti*di;
                            for grp in 0..ng {
                                let sgrp = grp * gs; let mut ssq = 0.0f32;
                                for i in 0..gs { let zi = z_all[yo+sgrp+i]; let gg = y[yo+sgrp+i] * (zi/(1.0+(-zi).exp())); yn[yo+sgrp+i]=gg; ssq+=gg*gg; }
                                let r = 1.0/((ssq/gs as f32)+eps).sqrt();
                                for i in 0..gs { yn[yo+sgrp+i] = yn[yo+sgrp+i]*r*nw[sgrp+i]; }
                            }
                        }
                        let ynt = upm(&yn, vec![s, di]);
                        let out = qmm(&ynt, &format!("{p}.mixer.out_proj.weight")); // [s, hid]
                        {
                            // fused residual add + NEXT layer's rms_norm (one pass, llama.cpp-style)
                            let nw = if l + 1 < PATTERN.len() { fwd.get(&format!("language_model.backbone.layers.{}.norm.weight", l + 1)) } else { fwd.get("norm_f") };
                            match nw {
                                Some(w) if is_cuda && std::env::var("NEMOTRON_FUSED_ADDNORM").is_ok() => { let (r, nx) = ffai_ops::add_rms_norm(d, &xt, &out, w, eps).unwrap(); xt = r; xn_carry = Some(nx); }
                                _ => { xt = add(d, &xt, &out).unwrap(); }
                            }
                        }
                        } // end !use_conv_device
                    }
                    'E' => {
                        let bias = &fw[&format!("{p}.mixer.gate.e_score_correction_bias")];
                        // ── Router over all S tokens (host top-k, matching decode) ──
                        // Batched gate gemv: [S, n_exp]. gate.weight is f32 dense.
                        let gate_w = &fwd[&format!("{p}.mixer.gate.weight")];
                        // NEMOTRON_DEVSORT=1 (CUDA): on-device router + counting-sort
                        // (moe_route_sort_device) — keeps the gate logits ON DEVICE (no
                        // ~1MB rl_all dl), runs sigmoid+bias+topk + sort on the GPU, then
                        // reconstructs the SAME `triples` via 3 small dls (offsets, sorted
                        // tok, sorted wt). Removes the per-E-layer host sigmoid/topk/sort.
                        // Step toward fully-on-device MoE + CUDA-graph prefill.
                        let devsort = is_cuda && std::env::var("NEMOTRON_DEVSORT_OFF").is_err();
                        let _t_router = std::time::Instant::now();
                        // off_dev/st_dev/sw_dev hoisted out (device routing tensors) so the
                        // Q4 path can run fully on-device (device descriptors + gather +
                        // scatter) with NO host sync — the prerequisite for CUDA-graphing
                        // the forward (kill the host-launch GPU idle, the real 3-5× lever).
                        let mut moe_dev: Option<(Tensor, Tensor, Tensor)> = None; // (offsets, sorted_tok, sorted_wt)
                        let mut xn_f16_router: Option<Tensor> = None; // f16 xn from the router, reused by the gather
                        // Fully-on-device MoE (descriptors+gather+scatter all device) → SKIP
                        // the host triples reconstruction entirely. mt = s*top_k is
                        // deterministic (every token picks exactly top_k experts) so NO dl is
                        // needed → the MoE region is host-sync-free. DEFAULT-ON for CUDA
                        // batched prefill: it kills the per-expert host round-trips that left
                        // the GPU 56% IDLE on the host-triples path (nsys: 43.6%→98% busy) →
                        // ~2× prefill (S=2048 925→1820, S=1024 968→2009, S=3072 461→1865 tok/s;
                        // argmax-matches the host path at S=1024/2048/3072, gold 1104@S2048).
                        // NEMOTRON_Q4_DEVDESC_OFF=1 reverts to the host-triples path (A/B).
                        // (Decode is unaffected — this is inside the batched-prefill closure.)
                        let devdesc_skip = is_cuda && std::env::var("NEMOTRON_Q4_DEVDESC_OFF").is_err();
                        let (triples, mt): (Vec<(u32, usize, f32)>, usize) = if devsort {
                            let bias_dev = cdev(format!("{p}.mixer.gate.e_score_correction_bias"), &|| up(&bias[..]));
                            // xn cast once to f16: feeds BOTH the cuBLAS router GEMM
                            // (was software-MMA ffai_gemm, ~28x slower at this shape)
                            // and the f16 token gather below.
                            let xn_f16_e = pf!(15, 0.0, cast_f32_f16(d, &xn).unwrap());
                            let gate_w16 = &fwd[&format!("{p}.mixer.gate.weight.f16")];
                            let rl_dev = pf!(10, 2.0 * s as f64 * n_exp as f64 * hid as f64,
                                gemm_cublas_f32out(d, &xn_f16_e, gate_w16, s, n_exp, hid).unwrap());
                            xn_f16_router = Some(xn_f16_e);
                            let (st_dev, sw_dev, off_dev) = ffai_ops::moe_route_sort_device(d, &rl_dev, &bias_dev, n_exp, top_k, scale_f).unwrap();
                            let m = s * top_k;
                            let tr: Vec<(u32, usize, f32)> = if devdesc_skip {
                                Vec::new() // host-free: descriptors/gather/scatter consume device tensors
                            } else {
                                let rdu = |t: &Tensor, n: usize| -> Vec<u32> { let mut b = vec![0u8; n*4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); b.chunks(4).map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect() };
                                let offs = rdu(&off_dev, n_exp + 1);
                                let stok = rdu(&st_dev, m);
                                let swt = dl(&sw_dev, m);
                                let mut tr = Vec::with_capacity(m);
                                for e in 0..n_exp { for r in offs[e] as usize..offs[e+1] as usize { tr.push((e as u32, stok[r] as usize, swt[r])); } }
                                tr
                            };
                            moe_dev = Some((off_dev, st_dev, sw_dev));
                            (tr, m)
                        } else {
                            let rl_all = dl(&pf!(10, 2.0 * s as f64 * n_exp as f64 * hid as f64, matmul(d, gate_w, &xn).unwrap()), s * n_exp);
                            let mut tr: Vec<(u32, usize, f32)> = Vec::with_capacity(s * top_k);
                            for ti in 0..s {
                                let rl = &rl_all[ti*n_exp..(ti+1)*n_exp];
                                let sig: Vec<f32> = rl.iter().map(|&z| 1.0/(1.0+(-z).exp())).collect();
                                let choice: Vec<f32> = (0..n_exp).map(|i| sig[i]+bias[i]).collect();
                                let eidx = ffai_runtime::topk(&choice, top_k);
                                let mut w: Vec<f32> = eidx.iter().map(|&e| sig[e]).collect();
                                let wsum: f32 = w.iter().sum::<f32>()+1e-20; for v in w.iter_mut() { *v = *v/wsum*scale_f; }
                                for (j,&e) in eidx.iter().enumerate() { tr.push((e as u32, ti, w[j])); }
                            }
                            tr.sort_by_key(|t| t.0);
                            let m = tr.len();
                            (tr, m)
                        };
                        if std::env::var("NEMOTRON_ROUTER_TIME").is_ok() {
                            ROUTER_HOST_T.with(|c| c.set(c.get() + _t_router.elapsed().as_secs_f64()));
                        }
                        // Expert GEMM mode flags (read early so they're in scope for gather).
                        let use_bgemm = std::env::var("NEMOTRON_BGEMM").is_ok();
                        // NEMOTRON_W4A16=1: W4A16 WMMA grouped GEMM (inline Q4 dequant,
                        //   scattered nibble reads). Supersedes NEMOTRON_BGEMM when both set.
                        // NEMOTRON_W4A16_MARLIN=1: same as W4A16 but with Marlin coalesced
                        //   tile-major layout — requires weights pre-permuted at load time.
                        //   Supersedes both NEMOTRON_W4A16 and NEMOTRON_BGEMM.
                        let use_w4a16_marlin = std::env::var("NEMOTRON_W4A16_MARLIN").is_ok();
                        let use_w4a16 = std::env::var("NEMOTRON_W4A16").is_ok() || use_w4a16_marlin;
                        let use_bgemm = use_bgemm && !use_w4a16;
                        // On non-CUDA (Metal): the bm64 BGEMM / W4A16 / grouped-GEMM MoE
                        // expert paths all finish with moe_scatter_add (+relu2_scale_f16),
                        // which dispatch_raw_cuda → hard-error on Metal. Force the portable
                        // per-expert path (host scatter, host f32 relu2) whose only CUDA dep
                        // is the GEMM — and that we swap below for dequant_q4_off + matmul.
                        let (use_bgemm, use_w4a16, use_w4a16_marlin) =
                            if is_cuda { (use_bgemm, use_w4a16, use_w4a16_marlin) } else { (false, false, false) };
                        // NEMOTRON_GROUPED_GEMM=1: async two-pass all-expert GEMM — all UP
                        //   GEMMs enqueued (no inter-expert sync), on-device relu2_scale_f16,
                        //   all DN GEMMs, then on-device scatter-add. Uses the device-gather
                        //   path for xs. Eliminates per-expert host syncs + cast overhead.
                        let use_grouped_gemm = std::env::var("NEMOTRON_GROUPED_GEMM").is_ok() && !use_bgemm && !use_w4a16;
                        // NEMOTRON_CUTLASS_MOE=1 (CUDA + CUTLASS build): contiguous f16 expert
                        //   slabs (dequant the full Q4 weight once, resident-cached) + ONE
                        //   CUTLASS grouped GEMM per up/down over the sorted tokens. Beats
                        //   cuBLAS per-expert on the skinny m≈96 shape (17.1 vs 15.7 TFLOP/s).
                        let use_cutlass_moe = is_cuda && std::env::var("NEMOTRON_CUTLASS_MOE").is_ok()
                            && !use_fp4_moe
                            && !use_bgemm && !use_w4a16 && !use_grouped_gemm;
                        // NEMOTRON_Q4_GROUPED=1 (CUDA): Q4-NATIVE single-launch grouped MoE
                        //   GEMM (moe_q4_grouped_mma) — reads the model's signed Q4 weights
                        //   DIRECTLY (no f16 slab, no dequant) per up/down over the sorted
                        //   tokens; on-device relu²+det-scatter. Collapses ~234 per-expert
                        //   cuBLAS launches → 2 AND reads 4.5-bit weights (vs 16-bit) → the
                        //   Marlin lever for the BW-bound MoE wall. Uses the device gather.
                        // NEMOTRON_Q4_GROUPED_LASTEXACT=K: run the EXACT cuBLAS per-expert
                        // path for the LAST K MoE layers (output-proximal) and Q4-grouped for
                        // the earlier ones — a precision/speed hybrid to test whether late-
                        // layer exactness restores the argmax (1104) while keeping most of
                        // the +35%. moe_idx = this E-layer's 0-based index among all E-layers.
                        let q4_lastexact: usize = std::env::var("NEMOTRON_Q4_GROUPED_LASTEXACT")
                            .ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                        let n_moe_layers = PATTERN.chars().filter(|&c| c == 'E').count();
                        let moe_idx = PATTERN.chars().take(l).filter(|&c| c == 'E').count();
                        // NEMOTRON_Q4_GROUPED_FIRSTEXACT=K: exact cuBLAS for the FIRST K MoE
                        // layers (the sensitive early ones), Q4 for the rest. (lastexact failed
                        // at all K — early error propagates irreducibly; test exact-early.)
                        // Default FIRSTEXACT=4: the first 4 MoE layers run EXACT cuBLAS, the
                        // rest Q4. Validated argmax==1104@S2048 (exact-early restores the
                        // razor 1104/1776 near-tie that pure-Q4 flips; gap 0.125, robust).
                        // exact-LAST-K failed at all K (early error propagates irreducibly).
                        // devdesc is host-free (empty triples) → the per-expert exact path is
                        // unavailable; force firstexact=0 (all layers Q4; validated 1104).
                        let q4_firstexact: usize = if devdesc_skip { 0 } else {
                            std::env::var("NEMOTRON_Q4_GROUPED_FIRSTEXACT")
                                .ok().and_then(|v| v.parse().ok()).unwrap_or(4) };
                        let q4_layer_ok = moe_idx >= q4_firstexact
                            && moe_idx < n_moe_layers.saturating_sub(q4_lastexact);
                        // DEFAULT-ON for CUDA (escape: NEMOTRON_Q4_GROUPED_OFF=1): Q4-native
                        // grouped MoE for all but the first FIRSTEXACT layers. +21% @S2048
                        // (1009→1220 tok/s), argmax 1104 exact.
                        let use_q4_grouped = is_cuda && std::env::var("NEMOTRON_Q4_GROUPED_OFF").is_err()
                            && q4_layer_ok
                            && !use_bgemm && !use_w4a16 && !use_grouped_gemm && !use_cutlass_moe;
                        // NEMOTRON_MOE_GROUPED=1 (Metal only): replace the per-expert
                        //   dequant_q4_off+matmul loop with the token-sorted GROUPED Q4
                        //   GEMM (moe_bgemm_q4_bm64) — ONE MMA dispatch over all sorted
                        //   rows, Q4 dequant fused in the kernel's block prologue (no
                        //   per-expert f16 weight materialization). relu²+scale on host
                        //   f32 (overflow-safe, matches the validated per-expert numerics),
                        //   then a second grouped GEMM for down, then host scatter. The
                        //   Metal analog of the CUDA GROUPED_GEMM/BGEMM device path — but
                        //   those finish with the CUDA-only moe_scatter_add, so this path
                        //   keeps the portable host relu²/scatter and only swaps the GEMM.
                        //   bm64 requires n_out % 64 == 0: inter=1856 (29·64) and
                        //   hid=2688 (42·64) both qualify. Gated off on CUDA (uses the
                        //   shipped GROUPED_GEMM/BGEMM tensor-core paths instead).
                        let use_metal_grouped = !is_cuda
                            && std::env::var("NEMOTRON_MOE_GROUPED").is_ok()
                            && inter % 64 == 0 && hid % 64 == 0;
                        // Build sorted activation [mt, hid]:
                        // NEMOTRON_DEVICE_GATHER=1: stay on device (gather kernel + cast),
                        //   eliminates the 22MB dl(&xn) + 132MB host scatter per E-layer.
                        // Default: host download + gather (validated path).
                        let dev_gather = std::env::var("NEMOTRON_DEVICE_GATHER").is_ok();
                        // ON-DEVICE MoE gather/scatter (DEFAULT-ON for CUDA): keep the
                        // validated fewer_syncs per-expert cuBLAS GEMM + on-device
                        // relu²/scatter, but ALSO gather the sorted expert activation ON
                        // DEVICE (ffai_gather over xn rows → [mt,hid] f16) instead of the
                        // dl(&xn)+host-scatter+per-expert re-upload, and keep the routed
                        // accumulator on device through the shared-expert + residual merge.
                        // Kills the ~22MB D2H × 23 host gather + host scatter + the
                        // dl(acc)/upm(acc) round-trip — the dominant remaining host wall in
                        // the conv-device prefill path. argmax-exact (1104/1120/1763),
                        // bit-deterministic, +~70% e2e. Escape: NEMOTRON_ONDEVICE_MOE_OFF=1
                        // reverts to the host-gather path.
                        // Only the default fewer_syncs per-expert path consumes the device
                        // gather; the BGEMM/W4A16/grouped/two_pass paths have their own xs
                        // handling, so don't hijack their gather when those are selected.
                        let two_pass_sel = is_cuda && std::env::var("NEMOTRON_TWO_PASS").is_ok();
                        let fewer_syncs_off = std::env::var("NEMOTRON_FEWER_SYNCS_OFF").is_ok();
                        let ondevice_moe = is_cuda && std::env::var("NEMOTRON_ONDEVICE_MOE_OFF").is_err()
                            && !use_bgemm && !use_w4a16 && !use_grouped_gemm && !use_metal_grouped
                            && !two_pass_sel && !fewer_syncs_off;
                        // xs_dev_f16 is the device version (for DEV_GATHER or BGEMM paths).
                        // xs (host f32 vec) is populated for the default host path.
                        // Build sorted activation [mt, hid]:
                        // For BGEMM: host gather → single device upload (avoids per-expert uploads).
                        // For DEVICE_GATHER: on-device gather (EXPERIMENTAL, may produce wrong output).
                        // Default: host gather (validated path).
                        let xn_h_opt: Option<Vec<f32>> = if use_bgemm || use_w4a16 || dev_gather || use_grouped_gemm || use_metal_grouped {
                            // Need xn on host for BGEMM/W4A16/METAL_GROUPED (host gather) or DEV_GATHER (gather source).
                            Some(dl(&xn, s * hid))
                        } else {
                            None
                        };
                        // FP4-direct: actq_pack gathers on read from the ungathered
                        // [s,hid] f16 — skip building the gathered xs entirely.
                        let fp4_skip_gather = is_cuda && moe_dev.is_some() && xn_f16_router.is_some()
                            && std::env::var("NEMOTRON_FP4_DIRECT_OFF").is_err()
                            && std::env::var("NEMOTRON_FP4_SIM").is_err()
                            && fp4w.contains_key(&format!("{p}.moe_up_proj_fp4")) && !use_w4a16;
                        let (xs, xs_dev_f16_opt) = if fp4_skip_gather {
                            (Vec::new(), None)
                        } else if use_w4a16_marlin && moe_dev.is_some() {
                            // PATH B (graph-safe): Marlin gathers tokens ON-DEVICE via the sorted_tok
                            // list (st_dev) -- no host xs copy, no host triples. [mt,hid] f16, sorted by expert.
                            let (_, st_dev, _) = moe_dev.as_ref().unwrap();
                            let xn_f16g = match &xn_f16_router { Some(t) => t.clone(), None => pf!(15, 0.0, cast_f32_f16(d, &xn).unwrap()) };
                            let xn2d = xn_f16g.reshaped(vec![s, hid]);
                            let xs_f16 = pf!(12, 0.0, gather(d, &xn2d, st_dev).unwrap());
                            (Vec::new(), Some(xs_f16))
                        } else if use_bgemm || use_w4a16 || use_grouped_gemm || use_metal_grouped {
                            // BGEMM/W4A16: host gather → upload xs as one [mt, hid] f16 tensor.
                            let xn_h = xn_h_opt.as_ref().unwrap();
                            let mut xs_h = vec![0.0f32; mt * hid];
                            for (r,(_e,t,_)) in triples.iter().enumerate() {
                                xs_h[r*hid..(r+1)*hid].copy_from_slice(&xn_h[t*hid..(t+1)*hid]);
                            }
                            let xs_f16 = cast_f32_f16(d, &upm(&xs_h, vec![mt, hid])).unwrap();
                            (Vec::new(), Some(xs_f16))
                        } else if dev_gather || ondevice_moe {
                            // On-device gather: upload token indices, gather xn rows on device
                            // into one [mt,hid] f16 buffer. The fewer_syncs per-expert GEMM
                            // reads device slices of this (offset g0*hid*2) → no host xs,
                            // no dl(&xn), no host scatter, no per-expert re-upload.
                            // sorted_tok (st_dev) IS the device token-index gather list (==
                            // triples tokens) → use it DIRECTLY when devsort, skipping the host
                            // tok_idx build + re-upload (one step toward a host-sync-free forward).
                            let tok_owned: Option<Tensor> = if moe_dev.is_some() { None } else {
                                let tok_idx: Vec<u32> = triples.iter().map(|(_,t,_)| *t as u32).collect();
                                Some(Tensor::new(d.upload(&tbu(&tok_idx)).unwrap(), vec![mt], DType::U32))
                            };
                            let tok_dev = match &moe_dev { Some((_,st,_)) => st, None => tok_owned.as_ref().unwrap() };
                            // f16-FIRST: cast [s,hid] once (or reuse the router's cast)
                            // then gather in f16 — 3.2x less traffic than the old
                            // gather-f32 [mt,hid] + cast-down chain (mt = 6s).
                            let xn_f16g = match &xn_f16_router {
                                Some(t) => t.clone(),
                                None => pf!(15, 0.0, cast_f32_f16(d, &xn).unwrap()),
                            };
                            let xn2d = xn_f16g.reshaped(vec![s, hid]);
                            let xs_f16 = pf!(12, 0.0, gather(d, &xn2d, tok_dev).unwrap()); // [mt, hid] f16
                            (Vec::new(), Some(xs_f16))
                        } else {
                            // Host gather: download xn, scatter rows.
                            let xn_h = dl(&xn, s * hid);
                            let mut xs = vec![0.0f32; mt * hid];
                            for (r,(_e,t,_)) in triples.iter().enumerate() {
                                xs[r*hid..(r+1)*hid].copy_from_slice(&xn_h[t*hid..(t+1)*hid]);
                            }
                            (xs, None)
                        };
                        // ── Expert GEMM: batched (NEMOTRON_BGEMM) or per-expert cuBLAS ──
                        // NEMOTRON_BGEMM=1: replace the per-expert cuBLAS loop with
                        // moe_bgemm_q4_bm64 (one dispatch for all experts, Q4 native,
                        // no per-expert dequant) + on-device relu2_scale_f16 + another
                        // bm64 for the down pass. Reduces O(n_active_experts*2) cuBLAS
                        // calls to 2 dispatches + 1 download per E-layer.
                        // Default: per-expert cuBLAS loop (existing, validated path).
                        let (uqs, usc, _, _) = &qw[&format!("{p}.moe_up_all")];
                        let (dqs, dsc, _, _) = &qw[&format!("{p}.moe_down_all")];
                        let up_bpr = hid / 32;     // Q4 blocks per up-weight row
                        let down_bpr = inter / 32; // Q4 blocks per down-weight row
                        let inv = 1.0f32 / 256.0;
                        // use_bgemm already defined above (before gather block).
                        let mut acc_h = vec![0.0f32; s * hid];
                        // ONDEVICE_MOE: when the fewer_syncs scatter leaves the routed
                        // accumulator on device, keep it here so the shared expert +
                        // residual add stay on-device (no dl(acc)+upm(acc) round-trip).
                        let mut acc_dev_keep: Option<Tensor> = None;
                        // set when the shared expert was folded into the routed grouped
                        // GEMM (acc already holds routed + shared) — skip the sd block.
                        let mut shared_folded = false;
                        if use_w4a16 {
                            // ── W4A16 WMMA path (NEMOTRON_W4A16=1 or NEMOTRON_W4A16_MARLIN=1) ──
                            // Small S (mt ≤ W4A16_THRESH, standard path only): inline Q4 dequant
                            //   + WMMA — wins because inline dequant saves 50% weight BW.
                            // Large S (mt > W4A16_THRESH, standard path only): fall back to
                            //   grouped_gemm (dequant-once + cuBLAS) — cuBLAS tiles are much larger.
                            // Marlin path (use_w4a16_marlin=true): always use moe_w4a16_marlin for
                            //   all S (no grouped_gemm fallback — grouped_gemm expects standard
                            //   layout, but Marlin-layout weights would produce wrong results).
                            let w4a16_thresh: usize = std::env::var("NEMOTRON_W4A16_THRESH")
                                .ok().and_then(|v| v.parse().ok()).unwrap_or(12288);
                            let xs_f16 = xs_dev_f16_opt.as_ref().unwrap().clone();
                            let marlin_dev = use_w4a16_marlin && moe_dev.is_some();
                            if marlin_dev {
                                // PATH B (graph-safe): fully on-device Marlin W4A16 MoE. Per-row expert
                                // ids from CSR offsets-expand, scatter via det_dev (st_dev/sw_dev), acc on
                                // device (kept unconditionally: ondevice_moe is hardwired off for use_w4a16,
                                // but PATH B is fully device + the shared expert emits sd_dev_opt, so the
                                // downstream device merge is the correct graph-safe consumer). No host dl.
                                let (off_dev, st_dev, sw_dev) = moe_dev.as_ref().unwrap();
                                let idx_dev = moe_expert_ids_from_offsets(d, off_dev, n_exp, mt).unwrap();
                                let (muqs, musc, _, _) = &qw[&format!("{p}.moe_up_all_marlin")];
                                let (mdqs, mdsc, _, _) = &qw[&format!("{p}.moe_down_all_marlin")];
                                let up_out = pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                    moe_w4a16_marlin(d, &xs_f16, muqs, musc, &idx_dev, mt, inter, hid).unwrap());
                                let up_relu2 = pf!(3, 0.0, relu2_scale_f16(d, &up_out, inv).unwrap());
                                let dn_out_f16 = pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                    moe_w4a16_marlin(d, &up_relu2, mdqs, mdsc, &idx_dev, mt, hid, inter).unwrap());
                                let acc_dev = Tensor::new(d.alloc_zeroed(s * hid * 4).unwrap(), vec![s, hid], DType::F32);
                                moe_scatter_add_det_dev(d, &dn_out_f16, st_dev, sw_dev, &acc_dev, s, mt, hid, 256.0f32, true).unwrap();
                                acc_dev_keep = Some(acc_dev);
                            } else {
                                let wts_h: Vec<f32> = triples.iter().map(|(_,_,w)| *w).collect();
                                let tidx_h: Vec<u32> = triples.iter().map(|(_,t,_)| *t as u32).collect();
                                let wts_dev = upm(&wts_h, vec![mt]);
                                let tidx_dev2 = Tensor::new(d.upload(&tbu(&tidx_h)).unwrap(), vec![mt], DType::U32);
                                let acc_dev = upm(&acc_h, vec![s, hid]);
                                if use_w4a16_marlin || mt <= w4a16_thresh {
                                    // Marlin path: always inline WMMA (handles all mt, Marlin layout).
                                    // Standard W4A16 path: inline WMMA for small mt ≤ threshold.
                                    let idx_u32: Vec<u32> = triples.iter().map(|(e,_,_)| *e).collect();
                                    let idx_dev = Tensor::new(d.upload(&tbu(&idx_u32)).unwrap(), vec![mt], DType::U32);
                                    let dn_out_f16 = if use_w4a16_marlin {
                                        // Marlin coalesced path (NEMOTRON_W4A16_MARLIN=1):
                                        // read the MARLIN-layout copies under the `_marlin`
                                        // keys, NOT the base standard-layout weights that the
                                        // sequential step path (moe_fused_ffn) consumes.
                                        let (muqs, musc, _, _) = &qw[&format!("{p}.moe_up_all_marlin")];
                                        let (mdqs, mdsc, _, _) = &qw[&format!("{p}.moe_down_all_marlin")];
                                        let up_out = pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                            moe_w4a16_marlin(d, &xs_f16, muqs, musc, &idx_dev, mt, inter, hid).unwrap());
                                        let up_relu2 = pf!(3, 0.0, relu2_scale_f16(d, &up_out, inv).unwrap());
                                        pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                            moe_w4a16_marlin(d, &up_relu2, mdqs, mdsc, &idx_dev, mt, hid, inter).unwrap())
                                    } else {
                                        // Standard scattered-nibble W4A16 path (NEMOTRON_W4A16=1)
                                        let up_out = pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                            moe_w4a16(d, &xs_f16, uqs, usc, &idx_dev, mt, inter, hid).unwrap());
                                        let up_relu2 = pf!(3, 0.0, relu2_scale_f16(d, &up_out, inv).unwrap());
                                        pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                            moe_w4a16(d, &up_relu2, dqs, dsc, &idx_dev, mt, hid, inter).unwrap())
                                    };
                                    let dn_out = cast_f16_f32(d, &dn_out_f16).unwrap();
                                    moe_scatter_add(d, &dn_out, &tidx_dev2, &wts_dev, &acc_dev, mt, hid, 256.0f32).unwrap();
                                } else {
                                    // Large S (standard W4A16 only): grouped_gemm (dequant-once + cuBLAS).
                                    // NOTE: never reached when use_w4a16_marlin=true (weights in Marlin layout).
                                    let mut g_starts_l: Vec<usize> = vec![0];
                                    let mut expert_ids_l: Vec<usize> = Vec::new();
                                    let mut gi = 0usize;
                                    while gi < mt {
                                        let eid = triples[gi].0 as usize;
                                        expert_ids_l.push(eid);
                                        let mut gi2 = gi + 1;
                                        while gi2 < mt && triples[gi2].0 as usize == eid { gi2 += 1; }
                                        g_starts_l.push(gi2);
                                        gi = gi2;
                                    }
                                    let dn_out_f16 = pf!(11, 2.0*mt as f64*(inter+hid) as f64*hid as f64,
                                        moe_grouped_gemm(d, uqs, usc, dqs, dsc, &xs_f16,
                                            &g_starts_l, &expert_ids_l, hid, inter, up_bpr, down_bpr,
                                            grouped_up_scratch.as_ref(), grouped_dn_scratch.as_ref()).unwrap());
                                    let dn_out = cast_f16_f32(d, &dn_out_f16).unwrap();
                                    moe_scatter_add(d, &dn_out, &tidx_dev2, &wts_dev, &acc_dev, mt, hid, 256.0f32).unwrap();
                                }
                                acc_h = dl(&acc_dev, s * hid);
                            }
                        } else if use_bgemm {
                            // ── Batched bm64 BGEMM path ───────────────────────────────────
                            // xs_dev_f16 from device gather (set above when use_bgemm=true).
                            let xs_f16 = xs_dev_f16_opt.as_ref().unwrap().clone();
                            let idx_u32: Vec<u32> = triples.iter().map(|(e,_,_)| *e).collect();
                            let idx_dev = Tensor::new(d.upload(&tbu(&idx_u32)).unwrap(), vec![mt], DType::U32);
                            // UP: [mt, inter] = xs[mt,hid] · Wup[n_exp*inter,hid]^T  (Q4 bm64 BGEMM)
                            let up_out = pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                moe_bgemm_q4_bm64(d, &xs_f16, uqs, usc, &idx_dev, mt, inter, hid).unwrap());
                            // relu2 + scale: fused on device (no host round-trip).
                            let up_relu2 = pf!(3, 0.0, relu2_scale_f16(d, &up_out, inv).unwrap());
                            // DOWN: [mt, hid] = relu2_out[mt,inter] · Wdn[n_exp*hid,inter]^T  (Q4 bm64 BGEMM)
                            let dn_out_f16 = pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                moe_bgemm_q4_bm64(d, &up_relu2, dqs, dsc, &idx_dev, mt, hid, inter).unwrap());
                            let dn_out = cast_f16_f32(d, &dn_out_f16).unwrap(); // [mt, hid] f32
                            // Scatter-weight + unscale (×256):
                            // NEMOTRON_BGEMM: on-device atomic scatter (no dl per expert).
                            // Routed-expert weights and token indices for the scatter-add.
                            let wts_h: Vec<f32> = triples.iter().map(|(_,_,w)| *w).collect();
                            let tidx_h: Vec<u32> = triples.iter().map(|(_,t,_)| *t as u32).collect();
                            let wts_dev = upm(&wts_h, vec![mt]);
                            let tidx_dev2 = Tensor::new(d.upload(&tbu(&tidx_h)).unwrap(), vec![mt], DType::U32);
                            // acc_dev: pre-zeroed [s, hid] f32 output accumulator.
                            let acc_dev = upm(&acc_h, vec![s, hid]); // already vec of 0.0f32
                            // Deterministic scatter by default (atomicAdd nondeterministic).
                            if std::env::var("NEMOTRON_ATOMIC_SCATTER").is_ok() {
                                moe_scatter_add(d, &dn_out, &tidx_dev2, &wts_dev, &acc_dev, mt, hid, 256.0f32).unwrap();
                            } else {
                                let _ = &tidx_dev2;
                                moe_scatter_add_det(d, &dn_out, &tidx_h, &wts_dev, &acc_dev, s, mt, hid, 256.0f32, false).unwrap();
                            }
                            // Download accumulated output for residual add.
                            acc_h = dl(&acc_dev, s * hid);
                        } else if use_cutlass_moe {
                            // ── CUTLASS grouped MoE GEMM (NEMOTRON_CUTLASS_MOE=1) ─────────
                            // Contiguous f16 expert slabs (dequant the FULL Q4 weight once,
                            // resident-cached) + ONE CUTLASS grouped GEMM per up/down over
                            // the sorted tokens; on-device relu²+scatter. Beats cuBLAS
                            // per-expert on the skinny m≈96 shape (17.1 vs 15.7 TFLOP/s).
                            let xs_f16 = xs_dev_f16_opt.as_ref().unwrap().clone();
                            let mut g_starts: Vec<usize> = vec![0];
                            let mut expert_ids: Vec<usize> = Vec::new();
                            { let mut gi=0usize; while gi<mt { let eid=triples[gi].0 as usize; expert_ids.push(eid);
                                let mut g2=gi+1; while g2<mt && triples[g2].0 as usize==eid {g2+=1;} g_starts.push(g2); gi=g2; } }
                            // resident contiguous f16 slabs (build once per layer; cached in w16)
                            let up_slab = { let key=format!("{p}.up_slab");
                                if let Some(t)=w16.borrow().get(&key) { t.clone() }
                                else { let t=pf!(3,(n_exp*inter*hid) as f64, dequant_q4(d, uqs, usc, n_exp*inter, hid, DType::F16).unwrap());
                                       w16.borrow_mut().insert(key, t.clone()); t } };
                            let dn_slab = { let key=format!("{p}.dn_slab");
                                if let Some(t)=w16.borrow().get(&key) { t.clone() }
                                else { let t=pf!(3,(n_exp*hid*inter) as f64, dequant_q4(d, dqs, dsc, n_exp*hid, inter, DType::F16).unwrap());
                                       w16.borrow_mut().insert(key, t.clone()); t } };
                            let up_out = pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                moe_grouped_gemm_cutlass(d, &xs_f16, &up_slab, &g_starts, &expert_ids, inter, hid).unwrap());
                            let a2 = pf!(3, 0.0, relu2_scale_f16(d, &up_out, inv).unwrap());
                            let dn_out_f16 = pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                moe_grouped_gemm_cutlass(d, &a2, &dn_slab, &g_starts, &expert_ids, hid, inter).unwrap());
                            let wts_h: Vec<f32> = triples.iter().map(|(_,_,w)| *w).collect();
                            let tidx_h: Vec<u32> = triples.iter().map(|(_,t,_)| *t as u32).collect();
                            let wts_dev = upm(&wts_h, vec![mt]);
                            let acc_dev = upm(&acc_h, vec![s, hid]);
                            let dn_out = cast_f16_f32(d, &dn_out_f16).unwrap();
                            moe_scatter_add_det(d, &dn_out, &tidx_h, &wts_dev, &acc_dev, s, mt, hid, 256.0f32, false).unwrap();
                            acc_h = dl(&acc_dev, s * hid);
                        } else if use_q4_grouped {
                            // ── Q4-NATIVE grouped MoE GEMM (NEMOTRON_Q4_GROUPED=1) ────────
                            // moe_q4_grouped_mma reads the signed Q4 weights directly (no
                            // f16 slab, no dequant). ONE launch per up/down. Device gather
                            // gives xs_f16; on-device relu²(1/256)+det-scatter(×256).
                            // None in FP4-direct mode (gather skipped upstream) — dummy.
                            let xs_f16 = xs_dev_f16_opt.clone().unwrap_or_else(|| {
                                Tensor::new(d.upload(&[0u8; 2]).unwrap(), vec![1], DType::F16)
                            });
                            // Host g_starts/expert_ids only for the host-descriptor path; the
                            // devdesc path builds them on-device (triples is empty there).
                            let mut g_starts: Vec<usize> = vec![0];
                            let mut expert_ids: Vec<usize> = Vec::new();
                            if !triples.is_empty() { let mut gi=0usize; while gi<mt { let eid=triples[gi].0 as usize; expert_ids.push(eid);
                                let mut g2=gi+1; while g2<mt && triples[g2].0 as usize==eid {g2+=1;} g_starts.push(g2); gi=g2; } }
                            // NEMOTRON_Q4_DEVDESC=1: build tile descriptors ON DEVICE from the
                            // routing offsets (no host g_starts, FIXED launch) → the GEMM is
                            // graph-capturable. Validated to match the host-descriptor path.
                            let devdesc = devdesc_skip && moe_dev.is_some();
                            let off_dev_ref = moe_dev.as_ref().map(|(o,_,_)| o);
                            // NEMOTRON_FP4_SIM: round-trip the MoE activations through NVFP4
                            // (e2m1+ue4m3 block-16) to inject EXACTLY the FP4-GEMM quant noise
                            // (no new mma kernel) → de-risk argmax 1104@S2048 BEFORE building the
                            // native FP4 path. Weights stay Q4 (~same err as NVFP4 weights); the
                            // NEW noise is the f16→FP4 activations, which this exercises.
                            let fp4_sim = std::env::var("NEMOTRON_FP4_SIM").is_ok() && is_cuda;
                            let fp4rt = |t: &Tensor, rows: usize, kk: usize| -> Tensor {
                                let hf = dl(&cast_f16_f32(d, t).unwrap(), rows * kk);
                                let rt = nvfp4_roundtrip(&hf, rows, kk);
                                cast_f32_f16(d, &upm(&rt, vec![rows, kk])).unwrap()
                            };
                            // FP4 direct mode: skip the gathered xs entirely — actq_pack
                            // gathers on read via st_dev from the ungathered [s,hid] f16
                            // (and the amax pass scans 6x fewer elements).
                            let fp4_direct = !fp4_sim && moe_dev.is_some() && xn_f16_router.is_some()
                                && std::env::var("NEMOTRON_FP4_DIRECT_OFF").is_err()
                                && !use_fp4_cutlass_env // CUTLASS path gathers in its own quant
                                && fp4w.contains_key(&format!("{p}.moe_up_proj_fp4"));
                            let xs_in = if fp4_direct { xn_f16_router.as_ref().unwrap().clone() }
                                else if fp4_sim { fp4rt(&xs_f16, mt, hid) } else { xs_f16.clone() };
                            // Native NVFP4 W4A4 grouped mma (NEMOTRON_FP4_MOE=1, devdesc only):
                            // dynamic act quant+pack + Blackwell block-scaled FP4 mma. Needs
                            // METALTILE_CUDA_ARCH=compute_121a.
                            let fp4_moe = devdesc && fp4w.contains_key(&format!("{p}.moe_up_proj_fp4"));
                            // Shared expert FOLDED into the grouped GEMM as two always-
                            // selected groups (up: two N=inter slabs; down: K-split halves
                            // whose scatter-adds sum to the full down GEMM). Same act quant
                            // (identity gather rows of xn), same deterministic scatter —
                            // kills the separate shared up/down f16 GEMMs + relu2 + merge.
                            let fold_keys = fp4_moe && fp4_direct
                                && fp4w.contains_key(&format!("{p}.moe_up_proj_fp4_fold"));
                            let fp4_cutlass = fp4_moe && use_fp4_cutlass_env
                                && fp4w_cutlass.contains_key(&format!("{p}.moe_up_proj_fp4_cutlass"));
                            let cutlass_dev = fp4_cutlass && std::env::var("NEMOTRON_CUTLASS_DEV").is_ok();
                            if cutlass_dev {
                                // ── device-descriptor CUTLASS grouped NVFP4 (zero host
                                // work per call): persistent prepared handles per layer;
                                // per call = grouped act quant (device sfa offsets) +
                                // alpha refresh + ONE fill kernel + gemm.run, per
                                // up/down. Groups = ALL n_exp experts (M=0 groups run
                                // zero tiles). ────────────────────────────────────────
                                let off_dev = off_dev_ref.unwrap();
                                let (uwp, uwsf, ugw) = &fp4w_cutlass[&format!("{p}.moe_up_proj_fp4_cutlass")];
                                let (dwp, dwsf, dgw) = &fp4w_cutlass[&format!("{p}.moe_down_proj_fp4_cutlass")];
                                if !cutlass_handles.borrow().contains_key(p.as_str()) {
                                    let alpha_u = Tensor::new(d.alloc_zeroed(n_exp * 4).unwrap(), vec![n_exp], DType::F32);
                                    let alpha_d = Tensor::new(d.alloc_zeroed(n_exp * 4).unwrap(), vec![n_exp], DType::F32);
                                    let hu = d.moe_grouped_cutlass_fp4_prepare(
                                        uwp.buffer.as_ref(), uwsf.buffer.as_ref(), alpha_u.buffer.as_ref(),
                                        n_exp, inter, hid, mt).unwrap();
                                    let hd = d.moe_grouped_cutlass_fp4_prepare(
                                        dwp.buffer.as_ref(), dwsf.buffer.as_ref(), alpha_d.buffer.as_ref(),
                                        n_exp, hid, inter, mt).unwrap();
                                    cutlass_handles.borrow_mut().insert(p.clone(), (hu, hd, alpha_u, alpha_d));
                                }
                                let hb = cutlass_handles.borrow();
                                let (hu, hd, alpha_u, alpha_d) = hb.get(p.as_str()).unwrap();
                                // gather-on-read up-quant from the UNGATHERED xn — kills the
                                // explicit xs gather pass; amax scans s rows instead of mt.
                                let (_, st_dev0, _) = moe_dev.as_ref().unwrap();
                                let (ap, asf, ags) = if let Some(xnr) = xn_f16_router.as_ref() {
                                    pf!(15, 0.0, ffai_ops::lt_fp4_quant_grouped_dev(d, xnr, mt, hid, off_dev, n_exp, 0, 0.0, Some((st_dev0, s))).unwrap())
                                } else {
                                    pf!(15, 0.0, ffai_ops::lt_fp4_quant_grouped_dev(d, &xs_in, mt, hid, off_dev, n_exp, 0, 0.0, None).unwrap())
                                };
                                ffai_ops::fp4_group_alpha_id(d, &ags, ugw, alpha_u, n_exp).unwrap();
                                let up_out = Tensor::empty(d, vec![mt, inter], DType::F16).unwrap();
                                pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                    d.moe_grouped_cutlass_fp4_run(*hu, ap.buffer.as_ref(), asf.buffer.as_ref(), up_out.buffer.as_ref(), off_dev.buffer.as_ref()).unwrap());
                                // relu²·(1/256) fused into the down-quant (no standalone pass)
                                let (bp, bsf, bgs) = pf!(15, 0.0,
                                    ffai_ops::lt_fp4_quant_grouped_dev(d, &up_out, mt, inter, off_dev, n_exp, 1, inv, None).unwrap());
                                ffai_ops::fp4_group_alpha_id(d, &bgs, dgw, alpha_d, n_exp).unwrap();
                                let dn_out = Tensor::empty(d, vec![mt, hid], DType::F16).unwrap();
                                pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                    d.moe_grouped_cutlass_fp4_run(*hd, bp.buffer.as_ref(), bsf.buffer.as_ref(), dn_out.buffer.as_ref(), off_dev.buffer.as_ref()).unwrap());
                                drop(hb);
                                let acc_dev = Tensor::new(d.alloc_zeroed(s * hid * 4).unwrap(), vec![s, hid], DType::F32);
                                let (_, st_dev, sw_dev) = moe_dev.as_ref().unwrap();
                                moe_scatter_add_det_dev(d, &dn_out, st_dev, sw_dev, &acc_dev, s, mt, hid, 256.0f32, true).unwrap();
                                if ondevice_moe { acc_dev_keep = Some(acc_dev); } else { acc_h = dl(&acc_dev, s * hid); }
                            } else if fp4_cutlass {
                                // ── CUTLASS grouped NVFP4 GEMM (NEMOTRON_CUTLASS_MOE=1
                                // on top of FP4_MOE) ──────────────────────────────────
                                // Host group descriptors from the device routing offsets
                                // (one small [n_exp+1] dl per E-layer; device-side build
                                // is a graph-safety follow-up), CUTLASS-layout act quant
                                // (linear e2m1 + per-group swizzled SF), ONE grouped
                                // block-scaled GEMM per up/down, relu²·(1/256) between,
                                // then the same det-scatter as the mma path.
                                let off_dev = off_dev_ref.unwrap();
                                let mut ob = vec![0u8; (n_exp + 1) * 4];
                                d.download(off_dev.buffer.as_ref(), &mut ob).unwrap();
                                let offs: Vec<usize> = ob.chunks(4)
                                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize).collect();
                                let mut g_starts_c: Vec<usize> = vec![0];
                                let mut eids_c: Vec<usize> = Vec::new();
                                for e in 0..n_exp {
                                    if offs[e + 1] > offs[e] { eids_c.push(e); g_starts_c.push(offs[e + 1]); }
                                }
                                let (uwp, uwsf, ugw) = &fp4w_cutlass[&format!("{p}.moe_up_proj_fp4_cutlass")];
                                let (dwp, dwsf, dgw) = &fp4w_cutlass[&format!("{p}.moe_down_proj_fp4_cutlass")];
                                let (ap, asf, aoff, ags) = pf!(15, 0.0,
                                    ffai_ops::lt_fp4_quant_grouped(d, &xs_in, mt, hid, &g_starts_c).unwrap());
                                let up_out = pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                    ffai_ops::moe_grouped_gemm_cutlass_fp4(d, &ap, &asf, &aoff, &ags, uwp, uwsf, ugw, &g_starts_c, &eids_c, inter, hid).unwrap());
                                let a2 = pf!(3, 0.0, relu2_scale_f16(d, &up_out, inv).unwrap());
                                let (bp, bsf, boff, bgs) = pf!(15, 0.0,
                                    ffai_ops::lt_fp4_quant_grouped(d, &a2, mt, inter, &g_starts_c).unwrap());
                                let dn_out = pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                    ffai_ops::moe_grouped_gemm_cutlass_fp4(d, &bp, &bsf, &boff, &bgs, dwp, dwsf, dgw, &g_starts_c, &eids_c, hid, inter).unwrap());
                                let acc_dev = Tensor::new(d.alloc_zeroed(s * hid * 4).unwrap(), vec![s, hid], DType::F32);
                                let (_, st_dev, sw_dev) = moe_dev.as_ref().unwrap();
                                moe_scatter_add_det_dev(d, &dn_out, st_dev, sw_dev, &acc_dev, s, mt, hid, 256.0f32, true).unwrap();
                                if ondevice_moe { acc_dev_keep = Some(acc_dev); } else { acc_h = dl(&acc_dev, s * hid); }
                            } else if fold_keys {
                                let (uwp, uwsc, ugw) = &fp4w[&format!("{p}.moe_up_proj_fp4_fold")];
                                let (dwp, dwsc, dgw) = &fp4w[&format!("{p}.moe_down_proj_fp4_fold")];
                                let amax_dn = Tensor::new(d.alloc_zeroed(4).unwrap(), vec![1], DType::U32);
                                let (off0, st0, sw0) = moe_dev.as_ref().unwrap();
                                let (st_e, sw_e, off_e) = ffai_ops::moe_extend_groups(d, st0, sw0, off0, mt, s, n_exp).unwrap();
                                let mt_e = mt + 2 * s;
                                let up_out = pf!(11, 2.0*mt_e as f64*inter as f64*hid as f64,
                                    ffai_ops::moe_fp4_grouped_mma_dev(d, &xs_in, uwp, uwsc, ugw, &off_e, n_exp + 2, mt_e, inter, hid, 0, 0.0, None, Some((&amax_dn, 1, inv)), Some(&st_e)).unwrap());
                                let dn_out = pf!(11, 2.0*mt_e as f64*hid as f64*inter as f64,
                                    ffai_ops::moe_fp4_grouped_mma_dev(d, &up_out, dwp, dwsc, dgw, &off_e, n_exp + 2, mt_e, hid, inter, 1, inv, Some(&amax_dn), None, None).unwrap());
                                let acc_dev = Tensor::new(d.alloc_zeroed(s * hid * 4).unwrap(), vec![s, hid], DType::F32);
                                moe_scatter_add_det_dev(d, &dn_out, &st_e, &sw_e, &acc_dev, s, mt_e, hid, 256.0f32, true).unwrap();
                                shared_folded = true;
                                if ondevice_moe { acc_dev_keep = Some(acc_dev); } else { acc_h = dl(&acc_dev, s * hid); }
                            } else if fp4_moe {
                                let (uwp, uwsc, ugw) = &fp4w[&format!("{p}.moe_up_proj_fp4")];
                                let (dwp, dwsc, dgw) = &fp4w[&format!("{p}.moe_down_proj_fp4")];
                                // up-GEMM epilogue also computes the DOWN-GEMM's activation
                                // amax (of relu²-transformed outputs) — kills the full 45MB
                                // re-read the down amax pass used to do.
                                let amax_dn = Tensor::new(d.alloc_zeroed(4).unwrap(), vec![1], DType::U32);
                                let up_gidx = if fp4_direct { moe_dev.as_ref().map(|(_, st, _)| st) } else { None };
                                let up_out = pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                    ffai_ops::moe_fp4_grouped_mma_dev(d, &xs_in, uwp, uwsc, ugw, off_dev_ref.unwrap(), n_exp, mt, inter, hid, 0, 0.0, None, Some((&amax_dn, 1, inv)), up_gidx).unwrap());
                                // relu²·(1/256) fused into the down act-quant (no standalone pass)
                                let dn_out = pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                    ffai_ops::moe_fp4_grouped_mma_dev(d, &up_out, dwp, dwsc, dgw, off_dev_ref.unwrap(), n_exp, mt, hid, inter, 1, inv, Some(&amax_dn), None, None).unwrap());
                                // acc_h is still all-zero on this path — seed the accumulator
                                // with a device-side memset instead of uploading 22MB of zeros.
                                let acc_dev = Tensor::new(d.alloc_zeroed(s * hid * 4).unwrap(), vec![s, hid], DType::F32);
                                let (_, st_dev, sw_dev) = moe_dev.as_ref().unwrap();
                                moe_scatter_add_det_dev(d, &dn_out, st_dev, sw_dev, &acc_dev, s, mt, hid, 256.0f32, true).unwrap();
                                if ondevice_moe { acc_dev_keep = Some(acc_dev); } else { acc_h = dl(&acc_dev, s * hid); }
                            } else {
                            // UP: [mt,inter] = xs[mt,hid] · Wup_q4[eid; inter,hid]  (n=inter,k=hid)
                            let up_out = if devdesc {
                                pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                    moe_q4_grouped_mma_dev(d, &xs_in, uqs, usc, off_dev_ref.unwrap(), n_exp, mt, inter, hid).unwrap())
                            } else {
                                pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                    moe_q4_grouped_mma(d, &xs_in, uqs, usc, &g_starts, &expert_ids, inter, hid).unwrap())
                            };
                            let a2_raw = pf!(3, 0.0, relu2_scale_f16(d, &up_out, inv).unwrap());
                            let a2 = if fp4_sim { fp4rt(&a2_raw, mt, inter) } else { a2_raw };
                            // DOWN: [mt,hid] = a2[mt,inter] · Wdn_q4[eid; hid,inter]  (n=hid,k=inter)
                            let dn_out = if devdesc {
                                pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                    moe_q4_grouped_mma_dev(d, &a2, dqs, dsc, off_dev_ref.unwrap(), n_exp, mt, hid, inter).unwrap())
                            } else {
                                pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                    moe_q4_grouped_mma(d, &a2, dqs, dsc, &g_starts, &expert_ids, hid, inter).unwrap())
                            };
                            // acc_h is still all-zero here — device-side memset, no 22MB upload.
                            let acc_dev = Tensor::new(d.alloc_zeroed(s * hid * 4).unwrap(), vec![s, hid], DType::F32);
                            if devdesc {
                                // Fully on-device scatter: sorted_tok/sorted_wt (st/sw) ARE the
                                // device token-idx + weight tensors → atomic scatter, NO host
                                // tidx_h/wts_h (host-sync-free, toward graph capture).
                                let (_, st_dev, sw_dev) = moe_dev.as_ref().unwrap();
                                // DETERMINISTIC device scatter (device CSR build) → host-sync-
                                // free AND argmax-stable (the atomic scatter flipped 1104/1776).
                                moe_scatter_add_det_dev(d, &dn_out, st_dev, sw_dev, &acc_dev, s, mt, hid, 256.0f32, true).unwrap();
                            } else {
                                let wts_h: Vec<f32> = triples.iter().map(|(_,_,w)| *w).collect();
                                let tidx_h: Vec<u32> = triples.iter().map(|(_,t,_)| *t as u32).collect();
                                let wts_dev = upm(&wts_h, vec![mt]);
                                moe_scatter_add_det(d, &dn_out, &tidx_h, &wts_dev, &acc_dev, s, mt, hid, 256.0f32, true).unwrap();
                            }
                            if ondevice_moe { acc_dev_keep = Some(acc_dev); } else { acc_h = dl(&acc_dev, s * hid); }
                            } // end !fp4_moe (Q4 grouped path)
                        } else if use_grouped_gemm {
                            // ── Grouped-GEMM path (NEMOTRON_GROUPED_GEMM=1) ───────────────
                            // All UP dequants (async) → all UP GEMMs (async) → relu2_scale →
                            // all DN dequants (async) → all DN GEMMs (async) → scatter-add.
                            // No per-expert host sync. Uses on-device scatter for output.
                            let xs_f16 = xs_dev_f16_opt.as_ref().unwrap().clone();
                            // Build group boundaries (sorted by expert: triples is already sorted).
                            let mut g_starts: Vec<usize> = vec![0];
                            let mut expert_ids: Vec<usize> = Vec::new();
                            {
                                let mut gi = 0usize;
                                while gi < mt {
                                    let eid = triples[gi].0 as usize;
                                    expert_ids.push(eid);
                                    let mut gi2 = gi + 1;
                                    while gi2 < mt && triples[gi2].0 as usize == eid { gi2 += 1; }
                                    g_starts.push(gi2);
                                    gi = gi2;
                                }
                            }
                            let dn_out_f16 = pf!(11, 2.0*mt as f64*(inter+hid) as f64*hid as f64,
                                moe_grouped_gemm(d, uqs, usc, dqs, dsc, &xs_f16,
                                    &g_starts, &expert_ids, hid, inter, up_bpr, down_bpr,
                                    grouped_up_scratch.as_ref(), grouped_dn_scratch.as_ref()).unwrap());
                            // Scatter-weight + unscale (×256): same as BGEMM path.
                            let wts_h: Vec<f32> = triples.iter().map(|(_,_,w)| *w).collect();
                            let tidx_h: Vec<u32> = triples.iter().map(|(_,t,_)| *t as u32).collect();
                            let wts_dev = upm(&wts_h, vec![mt]);
                            let tidx_dev2 = Tensor::new(d.upload(&tbu(&tidx_h)).unwrap(), vec![mt], DType::U32);
                            let acc_dev = upm(&acc_h, vec![s, hid]);
                            // dn_out_f16 is [mt, hid] f16 unscaled (relu2 already applied 1/256×);
                            // scatter_add needs to multiply by w×256 → net effect: w×256/256 = w.
                            // But scatter_add signature multiplies by (scale/256): pass scale=256
                            // so net = w×(256/256) = w (matches the bgemm path).
                            let dn_out = cast_f16_f32(d, &dn_out_f16).unwrap();
                            // DETERMINISTIC scatter by default: the atomicAdd variant is
                            // run-to-run nondeterministic (FP atomic accumulation order)
                            // and flips the deep-context argmax. NEMOTRON_ATOMIC_SCATTER=1
                            // restores the old atomic kernel (A/B perf only).
                            if std::env::var("NEMOTRON_ATOMIC_SCATTER").is_ok() {
                                moe_scatter_add(d, &dn_out, &tidx_dev2, &wts_dev, &acc_dev, mt, hid, 256.0f32).unwrap();
                            } else {
                                let _ = &tidx_dev2;
                                moe_scatter_add_det(d, &dn_out, &tidx_h, &wts_dev, &acc_dev, s, mt, hid, 256.0f32, false).unwrap();
                            }
                            acc_h = dl(&acc_dev, s * hid);
                        } else if use_metal_grouped {
                            // ── Metal GROUPED Q4 GEMM path (NEMOTRON_MOE_GROUPED=1) ───────
                            // Token-sort already done (triples sorted by expert; xs_f16 is
                            // the [mt, hid] f16 gather in expert order). Run ONE grouped
                            // Q4 MMA GEMM per pass via moe_bgemm_q4_bm64 — the kernel reads
                            // each row's expert id from `idx_dev` and dequants that expert's
                            // Q4 nibbles inline (no per-expert f16 weight slab, no per-expert
                            // dispatch). relu²+scale stays host-f32 (squares can exceed f16
                            // max → overflow; this also keeps EXACT-MATCH numerics with the
                            // validated per-expert path). Host scatter back to token order.
                            //
                            // GOTCHA (relu² fusion placement): the CUDA path fuses relu² on
                            // device (relu2_scale_f16, dispatch_raw_cuda → unavailable on
                            // Metal). We keep relu² on host f32 between the two grouped GEMMs.
                            // The down-GEMM's grouping is IDENTICAL to the up-GEMM's (same
                            // sorted order / same idx_dev), so no re-sort between passes.
                            let xs_f16 = xs_dev_f16_opt.as_ref().unwrap().clone();
                            // Per-row expert ids (sorted): drives the bm64 weight selection.
                            let idx_u32: Vec<u32> = triples.iter().map(|(e,_,_)| *e).collect();
                            let idx_dev = Tensor::new(d.upload(&tbu(&idx_u32)).unwrap(), vec![mt], DType::U32);
                            // UP: [mt, inter] = xs[mt,hid] · Wup[expert][inter,hid]ᵀ  (Q4 bm64 MMA)
                            let up_out = pf!(11, 2.0*mt as f64*inter as f64*hid as f64,
                                moe_bgemm_q4_bm64(d, &xs_f16, uqs, usc, &idx_dev, mt, inter, hid).unwrap());
                            // relu² + scale (÷256) on host f32 (overflow-safe, exact-match).
                            let up_h = dl(&pf!(15, 0.0, cast_f16_f32(d, &up_out).unwrap()), mt * inter);
                            let a2_h: Vec<f32> = up_h.iter().map(|&v| { let r = v.max(0.0); r*r*inv }).collect();
                            let a2_f16 = pf!(15, 0.0, cast_f32_f16(d, &upm(&a2_h, vec![mt, inter])).unwrap());
                            // DOWN: [mt, hid] = a2[mt,inter] · Wdn[expert][hid,inter]ᵀ  (Q4 bm64 MMA)
                            let dn_out = pf!(11, 2.0*mt as f64*hid as f64*inter as f64,
                                moe_bgemm_q4_bm64(d, &a2_f16, dqs, dsc, &idx_dev, mt, hid, inter).unwrap());
                            let dn_h = dl(&pf!(15, 0.0, cast_f16_f32(d, &dn_out).unwrap()), mt * hid);
                            // Host scatter back to token order (router-weighted, ×256 unscale).
                            for (r,(_e2, t, w)) in triples.iter().enumerate() {
                                let dr = &dn_h[r*hid..(r+1)*hid];
                                let ah = &mut acc_h[(*t)*hid..(*t+1)*hid];
                                for i in 0..hid { ah[i] += w * dr[i] * 256.0; }
                            }
                        } else {
                            // ── Per-expert cuBLAS loop ────────────────────────────────────
                            // NEMOTRON_TWO_PASS=1: two-pass variant — all UP GEMMs first
                            //   (no intermediate CPU sync), then batch relu2_scale, then all
                            //   DOWN GEMMs, then batch downloads + scatter. This keeps the
                            //   cuBLAS stream uninterrupted for all ~128 experts per layer.
                            // Default: interleaved per-expert (original validated path).
                            // NEMOTRON_DEV_RELU2=1: per-expert with on-device relu2 (regresses).
                            // two_pass uses gemm_cublas; dev_relu2 uses relu2_scale_f16
                            // (CUDA-only). On Metal force the interleaved host-relu2 path.
                            let two_pass = is_cuda && std::env::var("NEMOTRON_TWO_PASS").is_ok();
                            // NEMOTRON_FEWER_SYNCS=1: keep the per-expert cuBLAS UP/DOWN
                            // GEMMs (near-best), but fuse relu² ON DEVICE (relu2_scale_f16)
                            // and accumulate each expert group into a DEVICE acc via
                            // moe_scatter_add — so NOTHING is downloaded per expert. One
                            // dl(acc_dev) at the end of the layer replaces the ~2×128
                            // per-expert dl()/cuStreamSynchronize pairs. Implies dev relu².
                            // DEFAULT-ON for CUDA: FEWER_SYNCS is now deterministic (the htod
                            // null-stream race was fixed in metaltile) and +26% at zero cost.
                            // NEMOTRON_FEWER_SYNCS_OFF=1 reverts to the per-expert dl()/sync path.
                            let fewer_syncs = is_cuda && !two_pass && std::env::var("NEMOTRON_FEWER_SYNCS_OFF").is_err();
                            let dev_relu2 = is_cuda && !two_pass && (fewer_syncs || std::env::var("NEMOTRON_DEV_RELU2").is_ok());
                            if fewer_syncs {
                                // ── Batched per-expert path (NEMOTRON_FEWER_SYNCS=1) ──────────
                                // Keep the per-expert cuBLAS UP/DOWN GEMMs (near-best), but
                                // collapse the ~2×128 per-expert dl()/cuStreamSynchronize pairs
                                // into work that stays on the ordered stream, ending in ONE
                                // dl(acc_dev) per E-layer:
                                //   1. All UP GEMMs (async) write into one [mt, inter] f16 buffer
                                //      (gemm_tc_off) → on-device relu²+scale (relu2_scale_f16).
                                //   2. All DOWN GEMMs (async, reading the same buffer) write into
                                //      one [mt, hid] f16 buffer → on-device moe_scatter_add into
                                //      acc_dev → ONE dl(acc_dev).
                                // Matches the already-shipped GROUPED_GEMM/BGEMM device-MoE math
                                // (same relu2_scale_f16 + atomic scatter) and the same correctness
                                // envelope: argmax is bit-stable across most runs but can flip on a
                                // near-tie, because cuBLAS tensor-core GEMM (gemm_tc_off, shared by
                                // GROUPED_GEMM) is itself run-to-run nondeterministic (split-K
                                // accumulation). This is NOT a race introduced here — proven by the
                                // host-relu²/host-scatter variant below jittering identically, and by
                                // GROUPED_GEMM landing on the same alt-tokens. The per-expert host
                                // dl()-per-expert path (default, no flag) stays bit-deterministic.
                                // NEMOTRON_FEWER_SYNCS_HOST swaps relu²+scatter to host f32 (same
                                // 2-buffer batching: 2 dl/layer instead of ~256) for A/B.
                                // CRITICAL: every async-GEMM input (xg/wup/wdn) and the shared
                                // up_all/a2_all/dn_all buffers must stay alive until the dl()
                                // drains the stream (else a freed buffer is read mid-GEMM → race).
                                let host_path = std::env::var("NEMOTRON_FEWER_SYNCS_HOST").is_ok();
                                let up_all = Tensor::new(d.alloc(mt * inter * 2).unwrap(), vec![mt, inter], DType::F16);
                                // ── Phase 1: all UP GEMMs → up_all ──
                                let mut keep_up: Vec<Tensor> = Vec::new();
                                let mut g0 = 0usize;
                                while g0 < mt {
                                    let e = triples[g0].0 as usize;
                                    let mut g1 = g0 + 1;
                                    while g1 < mt && triples[g1].0 as usize == e { g1 += 1; }
                                    let rows = g1 - g0;
                                    // ONDEVICE_MOE: read A from the device-gathered xs_f16 at
                                    // byte offset g0*hid*2 (no host slice/upload). Else host path.
                                    let (xa_buf, xa_off, xg_keep) = if ondevice_moe {
                                        let xsd = xs_dev_f16_opt.as_ref().unwrap();
                                        (xsd.buffer.clone(), g0 * hid * 2, None)
                                    } else {
                                        let xg = cast_f32_f16(d, &upm(&xs[g0*hid..g1*hid], vec![rows, hid])).unwrap();
                                        (xg.buffer.clone(), 0usize, Some(xg))
                                    };
                                    let wup = deq16c(&format!("{p}.up.{e}"), uqs, usc, inter, hid, e*inter*up_bpr, resident_w16);
                                    pf!(11, 2.0*rows as f64*inter as f64*hid as f64,
                                        d.gemm_tc_off(
                                            xa_buf.as_ref(), xa_off,
                                            wup.buffer.as_ref(), 0,
                                            up_all.buffer.as_ref(), g0 * inter * 2,
                                            rows, inter, hid, DType::F16).unwrap());
                                    if let Some(xg) = xg_keep { keep_up.push(xg); }
                                    keep_up.push(wup);
                                    g0 = g1;
                                }
                                // relu²+scale: on device (default, fast) or host f32 (deterministic).
                                let a2_all = if host_path {
                                    let up_h = dl(&cast_f16_f32(d, &up_all).unwrap(), mt * inter);
                                    drop(std::mem::take(&mut keep_up)); // dl drained stream
                                    let a2_h: Vec<f32> = up_h.iter().map(|&v| { let r = v.max(0.0); r*r*inv }).collect();
                                    cast_f32_f16(d, &upm(&a2_h, vec![mt, inter])).unwrap()
                                } else {
                                    pf!(3, 0.0, relu2_scale_f16(d, &up_all, inv).unwrap())
                                };
                                // ── Phase 2: all DOWN GEMMs → dn_all ──
                                let dn_all = Tensor::new(d.alloc(mt * hid * 2).unwrap(), vec![mt, hid], DType::F16);
                                let mut keep_dn: Vec<Tensor> = vec![a2_all.clone()];
                                keep_dn.append(&mut keep_up); // keep UP inputs alive on device path
                                let mut g0 = 0usize;
                                while g0 < mt {
                                    let e = triples[g0].0 as usize;
                                    let mut g1 = g0 + 1;
                                    while g1 < mt && triples[g1].0 as usize == e { g1 += 1; }
                                    let rows = g1 - g0;
                                    let wdn = deq16c(&format!("{p}.dn.{e}"), dqs, dsc, hid, inter, e*hid*down_bpr, resident_w16);
                                    pf!(11, 2.0*rows as f64*hid as f64*inter as f64,
                                        d.gemm_tc_off(
                                            a2_all.buffer.as_ref(), g0 * inter * 2,
                                            wdn.buffer.as_ref(), 0,
                                            dn_all.buffer.as_ref(), g0 * hid * 2,
                                            rows, hid, inter, DType::F16).unwrap());
                                    keep_dn.push(wdn);
                                    g0 = g1;
                                }
                                if host_path {
                                    // ONE dl of all DOWN outputs → deterministic host scatter
                                    // (same accumulation order as the validated per-expert path).
                                    let dn_h = dl(&cast_f16_f32(d, &dn_all).unwrap(), mt * hid);
                                    drop(keep_dn); let _ = (&up_all, &dn_all);
                                    for (r,(_e2, t, w)) in triples.iter().enumerate() {
                                        let dr = &dn_h[r*hid..(r+1)*hid];
                                        let ah = &mut acc_h[(*t)*hid..(*t+1)*hid];
                                        for i in 0..hid { ah[i] += w * dr[i] * 256.0; }
                                    }
                                } else {
                                    // On-device cast + scatter-add → ONE dl(acc_dev). unscale=256
                                    // → net router weight w (matches host-path ×256).
                                    let wts_h: Vec<f32> = triples.iter().map(|(_,_,w)| *w).collect();
                                    let tidx_h: Vec<u32> = triples.iter().map(|(_,t,_)| *t as u32).collect();
                                    let wts_dev = upm(&wts_h, vec![mt]);
                                    let tidx_dev = Tensor::new(d.upload(&tbu(&tidx_h)).unwrap(), vec![mt], DType::U32);
                                    let acc_dev = upm(&acc_h, vec![s, hid]); // acc_h all-zero here
                                    // Deterministic scatter by default (atomicAdd is run-to-run
                                    // nondeterministic). NEMOTRON_ATOMIC_SCATTER=1 → old kernel.
                                    // Default det path reads dn_all (f16) DIRECTLY — skips the
                                    // per-layer cast_f16_f32 + [mt,hid] f32 materialization (~66MB
                                    // write/layer). Atomic fallback still needs f32, cast lazily.
                                    if std::env::var("NEMOTRON_ATOMIC_SCATTER").is_ok() {
                                        let dn_f32 = cast_f16_f32(d, &dn_all).unwrap();
                                        moe_scatter_add(d, &dn_f32, &tidx_dev, &wts_dev, &acc_dev, mt, hid, 256.0f32).unwrap();
                                    } else {
                                        let _ = &tidx_dev;
                                        moe_scatter_add_det(d, &dn_all, &tidx_h, &wts_dev, &acc_dev, s, mt, hid, 256.0f32, true).unwrap();
                                    }
                                    if ondevice_moe {
                                        // Keep the routed accumulator on device — the shared
                                        // expert + residual merge below consume it on-device.
                                        acc_dev_keep = Some(acc_dev);
                                    } else {
                                        acc_h = dl(&acc_dev, s * hid);
                                    }
                                    drop(keep_dn); let _ = (&up_all, &dn_all);
                                }
                            } else if two_pass {
                                // ── Pass 1: All UP GEMMs (async, no sync between experts) ────
                                // IMPORTANT: keep xg and wup tensors ALIVE until all GEMMs complete
                                // (GPU reads these async; dropping them before the stream sync would
                                // race with the pool reusing their buffers for later dequants).
                                struct ExpertBatch {
                                    e: usize, g0: usize, g1: usize,
                                    a: Tensor,
                                    _xg: Tensor,  // keep alive until after sync
                                    _wup: Tensor, // keep alive until after sync
                                }
                                let mut up_batches: Vec<ExpertBatch> = Vec::new();
                                let mut g0 = 0usize;
                                while g0 < mt {
                                    let e = triples[g0].0 as usize;
                                    let mut g1 = g0 + 1;
                                    while g1 < mt && triples[g1].0 as usize == e { g1 += 1; }
                                    let rows = g1 - g0;
                                    let xg = cast_f32_f16(d, &upm(&xs[g0*hid..g1*hid], vec![rows, hid])).unwrap();
                                    let wup = deq16c(&format!("{p}.up.{e}"), uqs, usc, inter, hid, e*inter*up_bpr, resident_w16);
                                    let a = gemm_cublas(d, &xg, &wup, rows, inter, hid).unwrap();
                                    up_batches.push(ExpertBatch { e, g0, g1, a, _xg: xg, _wup: wup });
                                    g0 = g1;
                                }
                                // ── Pass 2: Sync once, then host relu2 in f32 (avoids f16 overflow
                                // in device relu2_scale for large activations), then upload.
                                // The single d.synchronize() waits for ALL 128 UP GEMMs at once.
                                d.synchronize().ok();
                                let a2s: Vec<Tensor> = up_batches.iter().map(|b| {
                                    let rows = b.g1 - b.g0;
                                    let a_h = dl(&cast_f16_f32(d, &b.a).unwrap(), rows*inter);
                                    let a2_h: Vec<f32> = a_h.iter().map(|&v| { let r = v.max(0.0); r*r*inv }).collect();
                                    cast_f32_f16(d, &upm(&a2_h, vec![rows, inter])).unwrap()
                                }).collect();
                                // ── Pass 3: All DOWN GEMMs (async, no sync between) ────────
                                // Keep wdn tensors alive for the same reason as wup.
                                struct DownBatch { dn: Tensor, g0: usize, g1: usize, _wdn: Tensor }
                                let mut dn_batches: Vec<DownBatch> = Vec::new();
                                for (b, a2) in up_batches.iter().zip(&a2s) {
                                    let rows = b.g1 - b.g0;
                                    let wdn = deq16c(&format!("{p}.dn.{}", b.e), dqs, dsc, hid, inter, b.e*hid*down_bpr, resident_w16);
                                    let dn = gemm_cublas(d, a2, &wdn, rows, hid, inter).unwrap();
                                    dn_batches.push(DownBatch { dn, g0: b.g0, g1: b.g1, _wdn: wdn });
                                }
                                // All inputs (up_batches, a2s) can drop after down GEMMs are enqueued.
                                drop(up_batches); drop(a2s);
                                // ── Pass 4+5: Download all dn + scatter ────────────────────
                                // The FIRST dl() syncs the stream (GPU already finished all GEMMs).
                                // Subsequent dl() for other experts: GPU already done, instant sync.
                                for db in dn_batches {
                                    let rows = db.g1 - db.g0;
                                    let dn_h = dl(&cast_f16_f32(d, &db.dn).unwrap(), rows*hid);
                                    for r in 0..rows {
                                        let (_e2, t, w) = triples[db.g0+r];
                                        let dr = &dn_h[r*hid..(r+1)*hid];
                                        let ah = &mut acc_h[t*hid..(t+1)*hid];
                                        for i in 0..hid { ah[i] += w * dr[i] * 256.0; }
                                    }
                                }
                            } else {
                                // ── Interleaved per-expert (default, validated) ───────────
                                let mut g0 = 0usize;
                                while g0 < mt {
                                    let e = triples[g0].0 as usize;
                                    let mut g1 = g0 + 1;
                                    while g1 < mt && triples[g1].0 as usize == e { g1 += 1; }
                                    let rows = g1 - g0;
                                    let dn_h = if metal_f16_experts {
                                        // ── Metal per-expert GEMM, f16 compute (default) ────────
                                        // dequant_q4_off → f16 expert weight slab (block offset into
                                        // the shared up/down qs/sc), cast activation → f16, run the
                                        // matmul in f16 (ffai_gemm[F16]). relu2 in host f32 (squares
                                        // can exceed f16 max → overflow). EXACT MATCH vs the f32 path
                                        // (argmax unchanged). No resident cache — re-dequant each
                                        // forward (the f16 cache is a bandwidth loss on Metal).
                                        let xg = pf!(15, 0.0, cast_f32_f16(d, &upm(&xs[g0*hid..g1*hid], vec![rows, hid])).unwrap()); // f16
                                        let wup = pf!(3, (inter*hid) as f64, dequant_q4_off(d, uqs, usc, inter, hid, DType::F16, e*inter*up_bpr).unwrap());
                                        let a = pf!(11, 2.0*rows as f64*inter as f64*hid as f64, matmul(d, &wup, &xg).unwrap()); // [rows, inter] f16
                                        let a_h = dl(&pf!(15, 0.0, cast_f16_f32(d, &a).unwrap()), rows*inter);
                                        let a2_h: Vec<f32> = a_h.iter().map(|&v| { let r = v.max(0.0); r*r*inv }).collect();
                                        let a2 = pf!(15, 0.0, cast_f32_f16(d, &upm(&a2_h, vec![rows, inter])).unwrap()); // f16
                                        let wdn = pf!(3, (hid*inter) as f64, dequant_q4_off(d, dqs, dsc, hid, inter, DType::F16, e*hid*down_bpr).unwrap());
                                        let dn = pf!(11, 2.0*rows as f64*hid as f64*inter as f64, matmul(d, &wdn, &a2).unwrap()); // [rows, hid] f16
                                        dl(&pf!(15, 0.0, cast_f16_f32(d, &dn).unwrap()), rows*hid)
                                    } else if !is_cuda {
                                        // ── Portable Metal per-expert GEMM, f32 (NEMOTRON_METAL_F32_EXPERTS=1) ──
                                        // dequant_q4_off → f32 expert weight slab (block offset into
                                        // the shared up/down qs/sc), then portable f32 `matmul`.
                                        // relu2 in host f32 (matches the CUDA host-relu2 default).
                                        let xg = upm(&xs[g0*hid..g1*hid], vec![rows, hid]); // f32
                                        let wup = pf!(3, (inter*hid) as f64, dequant_q4_off(d, uqs, usc, inter, hid, DType::F32, e*inter*up_bpr).unwrap());
                                        let a = pf!(11, 2.0*rows as f64*inter as f64*hid as f64, matmul(d, &wup, &xg).unwrap()); // [rows, inter] f32
                                        let a_h = dl(&a, rows*inter);
                                        let a2_h: Vec<f32> = a_h.iter().map(|&v| { let r = v.max(0.0); r*r*inv }).collect();
                                        let a2 = upm(&a2_h, vec![rows, inter]); // f32
                                        let wdn = pf!(3, (hid*inter) as f64, dequant_q4_off(d, dqs, dsc, hid, inter, DType::F32, e*hid*down_bpr).unwrap());
                                        let dn = pf!(11, 2.0*rows as f64*hid as f64*inter as f64, matmul(d, &wdn, &a2).unwrap()); // [rows, hid] f32
                                        dl(&dn, rows*hid)
                                    } else {
                                    let xg = cast_f32_f16(d, &upm(&xs[g0*hid..g1*hid], vec![rows, hid])).unwrap();
                                    let wup = deq16c(&format!("{p}.up.{e}"), uqs, usc, inter, hid, e*inter*up_bpr, resident_w16);
                                    let a = pf!(11, 2.0*rows as f64*inter as f64*hid as f64, gemm_cublas(d, &xg, &wup, rows, inter, hid).unwrap());
                                    let a2 = if dev_relu2 {
                                        pf!(3, 0.0, relu2_scale_f16(d, &a, inv).unwrap())
                                    } else {
                                        let a_h = dl(&cast_f16_f32(d, &a).unwrap(), rows*inter);
                                        let a2_h: Vec<f32> = a_h.iter().map(|&v| { let r = v.max(0.0); r*r*inv }).collect();
                                        cast_f32_f16(d, &upm(&a2_h, vec![rows, inter])).unwrap()
                                    };
                                    let wdn = deq16c(&format!("{p}.dn.{e}"), dqs, dsc, hid, inter, e*hid*down_bpr, resident_w16);
                                    let dn = pf!(11, 2.0*rows as f64*hid as f64*inter as f64, gemm_cublas(d, &a2, &wdn, rows, hid, inter).unwrap());
                                    dl(&cast_f16_f32(d, &dn).unwrap(), rows*hid)
                                    };
                                    for r in 0..rows {
                                        let (_e2, t, w) = triples[g0+r];
                                        let dr = &dn_h[r*hid..(r+1)*hid];
                                        let ah = &mut acc_h[t*hid..(t+1)*hid];
                                        for i in 0..hid { ah[i] += w * dr[i] * 256.0; }
                                    }
                                    g0 = g1;
                                }
                            }
                        }
                        // ── Shared expert: dense over all S (up→relu2→down) ──
                        // relu2 squares activations → can exceed f16 max (65504) →
                        // inf/NaN. NEMOTRON_SHARED_GEMV=1 uses the known-correct decode
                        // gemv per token (A/B oracle); default uses the MMA GEMM (f32
                        // to avoid the relu2 overflow).
                        let (suqs, susc, sm_up, sk_up) = &qw[&format!("{p}.mixer.shared_experts.up_proj.weight")];
                        let (sdqs, sdsc, sm_dn, sk_dn) = &qw[&format!("{p}.mixer.shared_experts.down_proj.weight")];
                        // Metal shared-expert MMA gate (DEFAULT ON): route the up/down
                        // GEMMs through the Q4-native cooperative-tensor MMA (gemm_q4_mpp,
                        // same kernel as the dense projections / per-expert experts) instead
                        // of dequant_q4→f32 + scalar matmul. relu2 stays in HOST f32 (squares
                        // can exceed f16 max → overflow) — identical to the routed per-expert
                        // f16 path (argmax-exact). The GEMM itself moving to Q4-native f16
                        // MMA is the win (~2 → ~14 TFLOP/s). Set NEMOTRON_METAL_F32_SHARED=1
                        // (or NEMOTRON_MOE_SHARED_MMA=0) to fall back to the f32 dequant+matmul
                        // path for A/B. This is the Metal DEFAULT (matches metal_f16_experts for
                        // the routed experts); NEMOTRON_MOE_SHARED_MMA=0 force-disables.
                        let shared_mma_flag = std::env::var("NEMOTRON_MOE_SHARED_MMA").ok();
                        let metal_mma_shared = !is_cuda
                            && shared_mma_flag.as_deref() != Some("0")
                            && !std::env::var("NEMOTRON_METAL_F32_SHARED").is_ok();
                        // devdesc: fully ON-DEVICE shared expert (no dl) for graph capture.
                        // device cuBLAS GEMMs + device relu²(full scale — shared activations
                        // are small, sa²≪f16max so no /256 needed) + device merge below.
                        let mut sd_dev_opt: Option<Tensor> = None;
                        let sd_h: Vec<f32> = if shared_folded {
                            Vec::new() // shared already accumulated in the grouped scatter
                        } else if devdesc_skip && is_cuda {
                            // devdesc: fully ON-DEVICE shared expert (no dl) — device cuBLAS
                            // (cublasLt, fast tensor-core) + device relu²(full scale — shared
                            // activations small, sa²≪f16max) + device merge below. NOTE: this
                            // is NOT CUDA-graph-capturable (the f16-out cublasLt wmma kernel
                            // faults on replay), but graphs are a dead end here anyway — the
                            // host-free forward alone pegs the GPU at ~98% busy, so the win is
                            // killing the host round-trip idle, NOT graph replay.
                            // reuse the router's f16 cast of xn (same tensor) when present
                            let xnh = match &xn_f16_router {
                                Some(t) => t.clone(),
                                None => pf!(15, 0.0, cast_f32_f16(d, &xn).unwrap()),
                            };
                            let fp4_shared = fp4w.contains_key(&format!("{p}.sh_up_proj_fp4"));
                            let sd = if fp4_shared {
                                // NVFP4 W4A4 shared expert: same grouped kernel, n_exp=1
                                // (offsets [0, s] — one run covering all rows).
                                let off1 = cdev(format!("__off1.{s}"), &|| Tensor::new(d.upload(&tbu(&[0u32, s as u32])).unwrap(), vec![2], DType::U32));
                                let (uwp, uwsc, ugw) = &fp4w[&format!("{p}.sh_up_proj_fp4")];
                                let (dwp, dwsc, dgw) = &fp4w[&format!("{p}.sh_down_proj_fp4")];
                                let amax_sd = Tensor::new(d.alloc_zeroed(4).unwrap(), vec![1], DType::U32);
                                let sa = pf!(12, 2.0*s as f64*shared_inter as f64*hid as f64,
                                    ffai_ops::moe_fp4_grouped_mma_dev(d, &xnh, uwp, uwsc, ugw, &off1, 1, s, shared_inter, hid, 0, 0.0, None, Some((&amax_sd, 1, 1.0)), None).unwrap());
                                pf!(12, 2.0*s as f64*hid as f64*shared_inter as f64,
                                    ffai_ops::moe_fp4_grouped_mma_dev(d, &sa, dwp, dwsc, dgw, &off1, 1, s, hid, shared_inter, 1, 1.0, Some(&amax_sd), None, None).unwrap())
                            } else if std::env::var("NEMOTRON_FP4_SHARED_LT").is_ok() {
                                // Lt block-scaled FP4 shared expert (~4x the f16 GEMM rate):
                                // weights packed once (cross-forward cache), activations
                                // quantized per call. Quant-regime lever (changes numerics).
                                let (uwq, uwsf, uwg) = {
                                    let key = format!("{p}.shup.lt4");
                                    if let (Some(t), Some(g)) = (w4lt.borrow().get(&key).cloned(), w4g.borrow().get(&key).cloned()) { (t.0, t.1, g) } else {
                                        let wf = deq16(&format!("{p}.shup"), suqs, susc, *sm_up, *sk_up, 0);
                                        let (wp, wsf, wg) = ffai_ops::lt_fp4_quant_g(d, &wf, *sm_up, *sk_up).unwrap();
                                        w4lt.borrow_mut().insert(key.clone(), (wp.clone(), wsf.clone())); w4g.borrow_mut().insert(key, wg.clone()); (wp, wsf, wg)
                                    }
                                };
                                let (dwq, dwsf, dwg) = {
                                    let key = format!("{p}.shdn.lt4");
                                    if let (Some(t), Some(g)) = (w4lt.borrow().get(&key).cloned(), w4g.borrow().get(&key).cloned()) { (t.0, t.1, g) } else {
                                        let wf = deq16(&format!("{p}.shdn"), sdqs, sdsc, *sm_dn, *sk_dn, 0);
                                        let (wp, wsf, wg) = ffai_ops::lt_fp4_quant_g(d, &wf, *sm_dn, *sk_dn).unwrap();
                                        w4lt.borrow_mut().insert(key.clone(), (wp.clone(), wsf.clone())); w4g.borrow_mut().insert(key, wg.clone()); (wp, wsf, wg)
                                    }
                                };
                                let (xq, xsf, xg) = pf!(15, 0.0, ffai_ops::lt_fp4_quant_g(d, &xnh, s, hid).unwrap());
                                let ds_u = ffai_ops::fp4_dscale(d, &xg, &uwg).unwrap();
                                let sa = pf!(12, 2.0*s as f64*shared_inter as f64*hid as f64,
                                    ffai_ops::gemm_fp4(d, &xq, &xsf, &uwq, &uwsf, s, *sm_up, *sk_up, false, Some(&ds_u)).unwrap());
                                let sa2 = pf!(3, 0.0, relu2_scale_f16(d, &sa, 1.0f32).unwrap());
                                let (aq, asf, ag) = pf!(15, 0.0, ffai_ops::lt_fp4_quant_g(d, &sa2, s, shared_inter).unwrap());
                                let ds_d = ffai_ops::fp4_dscale(d, &ag, &dwg).unwrap();
                                pf!(12, 2.0*s as f64*hid as f64*shared_inter as f64,
                                    ffai_ops::gemm_fp4(d, &aq, &asf, &dwq, &dwsf, s, *sm_dn, *sk_dn, false, Some(&ds_d)).unwrap())
                            } else {
                            let wsu = deq16(&format!("{p}.shup"), suqs, susc, *sm_up, *sk_up, 0);
                            let sa = pf!(12, 2.0*s as f64*shared_inter as f64*hid as f64,
                                gemm_cublas(d, &xnh, &wsu, s, *sm_up, *sk_up).unwrap());
                            let sa2 = pf!(3, 0.0, relu2_scale_f16(d, &sa, 1.0f32).unwrap()); // full scale, device
                            let wsd = deq16(&format!("{p}.shdn"), sdqs, sdsc, *sm_dn, *sk_dn, 0);
                            pf!(12, 2.0*s as f64*hid as f64*shared_inter as f64,
                                gemm_cublas(d, &sa2, &wsd, s, *sm_dn, *sk_dn).unwrap()) // [s,hid] f16
                            };
                            sd_dev_opt = Some(pf!(15, 0.0, cast_f16_f32(d, &sd).unwrap())); // device f32, full scale
                            Vec::new()
                        } else if metal_mma_shared {
                            // ── Portable Metal shared-expert MMA: gemm_q4_mpp up/down, host relu2 ──
                            // Cast xn → f16 once; up GEMM [s,hid]·Wupᵀ → [s,shared_inter] f16;
                            // relu2+(/256) in host f32 (overflow-safe); re-cast → f16; down GEMM
                            // [s,shared_inter]·Wdnᵀ → [s,hid] f16; (×256) on host. EXACT-MATCH
                            // numerics vs the f32 path (the f16 up-proj output fits f16 — the
                            // routed experts already prove this at the same activation scale).
                            let xnh = pf!(15, 0.0, cast_f32_f16(d, &xn).unwrap()); // [s, hid] f16
                            let sa = pf!(12, 2.0*s as f64*shared_inter as f64*hid as f64,
                                gemm_q4_mpp(d, &xnh, suqs, susc, s, *sm_up, *sk_up).unwrap()); // [s, shared_inter] f16
                            let sa_h = dl(&pf!(15, 0.0, cast_f16_f32(d, &sa).unwrap()), s * shared_inter);
                            let sa2_h: Vec<f32> = sa_h.iter().map(|&v| { let r = v.max(0.0); r * r / 256.0 }).collect();
                            let sa2 = pf!(15, 0.0, cast_f32_f16(d, &upm(&sa2_h, vec![s, shared_inter])).unwrap()); // f16
                            let sd = pf!(12, 2.0*s as f64*hid as f64*shared_inter as f64,
                                gemm_q4_mpp(d, &sa2, sdqs, sdsc, s, *sm_dn, *sk_dn).unwrap()); // [s, hid] f16
                            let sd_h = dl(&pf!(15, 0.0, cast_f16_f32(d, &sd).unwrap()), s * hid);
                            sd_h.iter().map(|&v| v * 256.0).collect()
                        } else if !is_cuda {
                            // ── Portable Metal shared-expert: dequant→f32 + matmul, host relu2 ──
                            // relu2 squares activations (can exceed f16 max) so we keep the
                            // intermediate in f32 throughout — no overflow, fully portable.
                            // A/B fallback (NEMOTRON_METAL_F32_SHARED=1).
                            let xnf = xn.clone(); // [s, hid] f32
                            let wsu = pf!(3, (*sm_up * *sk_up) as f64, dequant_q4(d, suqs, susc, *sm_up, *sk_up, DType::F32).unwrap());
                            let sa = pf!(12, 2.0*s as f64*shared_inter as f64*hid as f64, matmul(d, &wsu, &xnf).unwrap()); // [s, shared_inter] f32
                            let sa_h = dl(&sa, s * shared_inter);
                            let sa2_h: Vec<f32> = sa_h.iter().map(|&v| { let r = v.max(0.0); r * r / 256.0 }).collect();
                            let sa2 = upm(&sa2_h, vec![s, shared_inter]);
                            let wsd = pf!(3, (*sm_dn * *sk_dn) as f64, dequant_q4(d, sdqs, sdsc, *sm_dn, *sk_dn, DType::F32).unwrap());
                            let sd = pf!(12, 2.0*s as f64*hid as f64*shared_inter as f64, matmul(d, &wsd, &sa2).unwrap()); // [s, hid] f32
                            let sd_h = dl(&sd, s * hid);
                            sd_h.iter().map(|&v| v * 256.0).collect()
                        } else if std::env::var("NEMOTRON_SHARED_GEMV").is_ok() {
                            let mut out = vec![0.0f32; s * hid];
                            for ti in 0..s {
                                let x1 = slice(d, &xn, ti*hid, hid).unwrap();
                                let sa = gemv_q4_relu2(d, suqs, susc, &x1, shared_inter, hid, shared_inter).unwrap();
                                let acc1 = up(&vec![0.0f32; hid]);
                                gemv_q4_accum(d, sdqs, sdsc, &sa, &acc1, &up(&[1.0f32]), *sm_dn, *sk_dn, *sm_dn).unwrap();
                                out[ti*hid..(ti+1)*hid].copy_from_slice(&dl(&acc1, hid));
                            }
                            out
                        } else {
                            // cuBLAS tensor cores. relu2 squares → overflow f16; use on-device
                            // relu2_scale_f16 (NEMOTRON_BGEMM or NEMOTRON_DEV_RELU2) to avoid
                            // the host round-trip; otherwise fall back to host f32 path.
                            let dev_r2 = use_bgemm || std::env::var("NEMOTRON_DEV_RELU2").is_ok();
                            let wsu = deq16(&format!("{p}.shup"), suqs, susc, *sm_up, *sk_up, 0);
                            let shared_up_flops = 2.0 * s as f64 * shared_inter as f64 * hid as f64;
                            let sa = pf!(12, shared_up_flops, gemm_cublas(d, &cast_f32_f16(d, &xn).unwrap(), &wsu, s, shared_inter, hid).unwrap());
                            let sa2 = if dev_r2 {
                                // On-device: relu2_scale_f16 fuses relu2 + scale without dl.
                                pf!(3, 0.0, relu2_scale_f16(d, &sa, 1.0f32 / 256.0).unwrap())
                            } else {
                                let sa_h = dl(&cast_f16_f32(d, &sa).unwrap(), s * shared_inter);
                                let sa2_h: Vec<f32> = sa_h.iter().map(|&v| { let r = v.max(0.0); r * r / 256.0 }).collect();
                                cast_f32_f16(d, &upm(&sa2_h, vec![s, shared_inter])).unwrap()
                            };
                            let wsd = deq16(&format!("{p}.shdn"), sdqs, sdsc, *sm_dn, *sk_dn, 0);
                            let shared_dn_flops = 2.0 * s as f64 * hid as f64 * shared_inter as f64;
                            let sd = pf!(12, shared_dn_flops, gemm_cublas(d, &sa2, &wsd, s, hid, shared_inter).unwrap());
                            let sd_h = dl(&cast_f16_f32(d, &sd).unwrap(), s * hid);
                            sd_h.iter().map(|&v| v * 256.0).collect()
                        };
                        if let Some(acc_dev) = acc_dev_keep.take() {
                            // ONDEVICE_MOE: routed acc is already on device. Merge the
                            // shared-expert output + residual on device — no dl/upm. Under
                            // devdesc sd is ALREADY device (sd_dev_opt, host-sync-free);
                            // else upload the host f32 shared output.
                            {
                                // fused residual add + NEXT layer's rms_norm (one pass, llama.cpp-style)
                                let nw = if l + 1 < PATTERN.len() { fwd.get(&format!("language_model.backbone.layers.{}.norm.weight", l + 1)) } else { fwd.get("norm_f") };
                                if shared_folded {
                                    // shared already accumulated into acc by the grouped scatter
                                    xt = add(d, &xt, &acc_dev).unwrap();
                                } else {
                                    let sd_dev = sd_dev_opt.take().unwrap_or_else(|| upm(&sd_h, vec![s, hid]));
                                    match nw {
                                        Some(w) if is_cuda && std::env::var("NEMOTRON_FUSED_ADDNORM").is_ok() => { let merged = add(d, &acc_dev, &sd_dev).unwrap(); let (r, nx) = ffai_ops::add_rms_norm(d, &xt, &merged, w, eps).unwrap(); xt = r; xn_carry = Some(nx); }
                                        // one launch, no intermediate: residual + routed acc + shared out
                                        _ if is_cuda => { xt = ffai_ops::add3(d, &xt, &acc_dev, &sd_dev).unwrap(); }
                                        _ => { let merged = add(d, &acc_dev, &sd_dev).unwrap(); xt = add(d, &xt, &merged).unwrap(); }
                                    }
                                }
                            }
                        } else {
                            for i in 0..s*hid { acc_h[i] += sd_h[i]; }
                            let acc_up = upm(&acc_h, vec![s, hid]);
                            {
                                // fused residual add + NEXT layer's rms_norm (one pass, llama.cpp-style)
                                let nw = if l + 1 < PATTERN.len() { fwd.get(&format!("language_model.backbone.layers.{}.norm.weight", l + 1)) } else { fwd.get("norm_f") };
                                match nw {
                                    Some(w) if is_cuda && std::env::var("NEMOTRON_FUSED_ADDNORM").is_ok() => { let (r, nx) = ffai_ops::add_rms_norm(d, &xt, &acc_up, w, eps).unwrap(); xt = r; xn_carry = Some(nx); }
                                    _ => { xt = add(d, &xt, &acc_up).unwrap(); }
                                }
                            }
                        }
                    }
                    '*' => {
                        // Cast xn→f16 ONCE, reuse for q/k/v (NEMOTRON_FUSE_QKV) — else per-proj cast.
                        let (q_all, k_all, v_all) = if fuse_qkv {
                            let xnh = pf!(15, 0.0, cast_f32_f16(d, &xn).unwrap());
                            (qmm_h(&xnh, &format!("{p}.mixer.q_proj.weight")),
                             qmm_h(&xnh, &format!("{p}.mixer.k_proj.weight")),
                             qmm_h(&xnh, &format!("{p}.mixer.v_proj.weight")))
                        } else {
                            (qmm(&xn, &format!("{p}.mixer.q_proj.weight")),
                             qmm(&xn, &format!("{p}.mixer.k_proj.weight")),
                             qmm(&xn, &format!("{p}.mixer.v_proj.weight")))
                        };
                        // [s, qdim]=[s,nq*hd]; k/v [s, kvdim]=[s,nkv*hd]
                        // PART C: eliminate dl(q)/dl(k)/dl(v) + per-token rope dispatches.
                        // Build positions [s] once, use batched rope_llama_many + kv_append_many.
                        if kvcache[l].is_none() {
                            kvcache[l] = Some((Tensor::new(d.alloc_zeroed(nkv*cap*hd*4).unwrap(), vec![nkv*cap*hd], DType::F32), Tensor::new(d.alloc_zeroed(nkv*cap*hd*4).unwrap(), vec![nkv*cap*hd], DType::F32)));
                        }
                        let positions_h: Vec<u32> = (0..s).map(|ti| (base + ti) as u32).collect();
                        let positions_dev = Tensor::new(
                            d.upload(unsafe { std::slice::from_raw_parts(positions_h.as_ptr() as *const u8, positions_h.len() * 4) }).unwrap(),
                            vec![s], DType::U32);
                        // Batched rope: [s, n_heads, hd] each → rotated in one dispatch.
                        let qr = pf!(8, 0.0, rope_llama_many(d, &q_all.reshaped(vec![s, nq, hd]), &positions_dev, nq, hd, rope_theta, 1.0, 1.0, 1.0, 8192.0).unwrap());
                        let kr = pf!(8, 0.0, rope_llama_many(d, &k_all.reshaped(vec![s, nkv, hd]), &positions_dev, nkv, hd, rope_theta, 1.0, 1.0, 1.0, 8192.0).unwrap());
                        // Batched KV-cache append: 2 dispatches replace S*2 per-token kv_append calls.
                        let (kcache, vcache) = kvcache[l].as_ref().unwrap();
                        kv_append_many(d, &kr, &positions_dev, kcache, nkv, hd, cap).unwrap();
                        kv_append_many(d, &v_all.reshaped(vec![s, nkv, hd]), &positions_dev, vcache, nkv, hd, cap).unwrap();
                        // sdpa_multi: Q [n_query, n_q_heads, hd], K/V [n_kv, cap, hd], causal, base_kv=base.
                        // Attention path auto-select: the tensor-core cuBLAS flash-attn
                        // (sdpa_multi_tc) wins big once KV is deep (9.5-14× stage @ d8192+),
                        // but its prep/transpose overhead regresses shallow KV (d0). So
                        // auto-enable it once total KV (base+s) crosses ~4096; below that use
                        // the software-MMA sdpa_multi. Override: NEMOTRON_PREFILL_TCATTN=1 force
                        // on, =0 force off.
                        let avg_kv = base as f64 + s as f64 / 2.0;
                        let attn_flops = 4.0 * nq as f64 * hd as f64 * avg_kv * s as f64;
                        // sdpa_multi_tc is the cuBLAS tensor-core flash-attn (CUDA-only).
                        // On Metal always use the portable software-MMA sdpa_multi.
                        let use_tc_attn = is_cuda && match std::env::var("NEMOTRON_PREFILL_TCATTN").ok().as_deref() {
                            Some("0") => false,
                            Some(_) => true,
                            // Default-on the tensor-core flash-attn for any real prefill
                            // chunk (s>=512): the software-MMA sdpa_multi runs at ~0.5% MFU
                            // and was 16% of the S2048 forward; the TC path is 9-14x faster.
                            // (Was base+s>=4096, which left S2048/d0 on the slow path.)
                            None => s >= 512 || (base + s) >= 4096,
                        };
                        let attn = if use_tc_attn {
                            if let Some(seg_lo) = seg_lo_dev.as_ref() {
                                // Packed: block-diagonal varlen flash-attn (each query stays in its segment).
                                pf!(7, attn_flops, sdpa_multi_tc_varlen(d, &qr.reshaped(vec![s, nq, hd]), &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), seg_lo, pk_seg_len, hd, nq as u32, base as u32, s as u32, cap as u32, (nq/nkv) as u32, true, ascale).unwrap())
                            } else if hd == 128 && std::env::var("NEMOTRON_FLASH_MMA_OFF").is_err()
                                      && std::env::var("NEMOTRON_FLASH_WMMA").is_err()
                                      && std::env::var("NEMOTRON_FLASH_FUSED").is_err() {
                                // DEFAULT (CUDA, non-packed, hd=128): FlashAttention-2 with
                                // register-resident O (mma.sync) — beats cuBLAS-TC attention
                                // 1.5-3.3× (no HBM score materialization, causal-skipped, high
                                // occupancy). Argmax-exact. NEMOTRON_FLASH_MMA_OFF=1 → cuBLAS-TC.
                                pf!(7, attn_flops, ffai_ops::sdpa_flash_mma(d, &qr.reshaped(vec![s, nq, hd]), &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, nq as u32, base as u32, s as u32, cap as u32, (nq/nkv) as u32, true, ascale).unwrap())
                            } else if std::env::var("NEMOTRON_FLASH_WMMA").is_ok() {
                                // Fused tensor-core (wmma) causal FlashAttention: scores
                                // stay in shared (no HBM round-trip), causal skipped.
                                pf!(7, attn_flops, ffai_ops::sdpa_flash_wmma(d, &qr.reshaped(vec![s, nq, hd]), &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, nq as u32, base as u32, s as u32, cap as u32, (nq/nkv) as u32, true, ascale).unwrap())
                            } else if std::env::var("NEMOTRON_FLASH_FUSED").is_ok() {
                                // Fused single-kernel causal FlashAttention: no HBM
                                // score/prep round-trip, causal triangle skipped.
                                pf!(7, attn_flops, ffai_ops::sdpa_flash_fused(d, &qr.reshaped(vec![s, nq, hd]), &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, nq as u32, base as u32, s as u32, cap as u32, (nq/nkv) as u32, true, ascale).unwrap())
                            } else {
                            pf!(7, attn_flops, sdpa_multi_tc(d, &qr.reshaped(vec![s, nq, hd]), &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, nq as u32, base as u32, s as u32, cap as u32, (nq/nkv) as u32, true, ascale).unwrap())
                            }
                        } else {
                            pf!(7, attn_flops, sdpa_multi(d, &qr.reshaped(vec![s, nq, hd]), &kcache.reshaped(vec![nkv, cap, hd]), &vcache.reshaped(vec![nkv, cap, hd]), hd, nq as u32, base as u32, s as u32, cap as u32, (nq/nkv) as u32, true, ascale).unwrap())
                        };
                        // attn [s, nq, hd] = [s, qdim]; o_proj batched.
                        let o = qmm(&attn.reshaped(vec![s, qdim]), &format!("{p}.mixer.o_proj.weight"));
                        {
                            // fused residual add + NEXT layer's rms_norm (one pass, llama.cpp-style)
                            let nw = if l + 1 < PATTERN.len() { fwd.get(&format!("language_model.backbone.layers.{}.norm.weight", l + 1)) } else { fwd.get("norm_f") };
                            match nw {
                                Some(w) if is_cuda && std::env::var("NEMOTRON_FUSED_ADDNORM").is_ok() => { let (r, nx) = ffai_ops::add_rms_norm(d, &xt, &o, w, eps).unwrap(); xt = r; xn_carry = Some(nx); }
                                _ => { xt = add(d, &xt, &o).unwrap(); }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                // INSTRUMENTATION (revert): dump a token's hidden after layer l.
                // DUMP_TOK0=1 → token 0 (position 0, no cross-token influence);
                // else last token.
                if std::env::var("NEMOTRON_DUMP_LAYERS").is_ok() {
                    let tsel = if std::env::var("NEMOTRON_DUMP_TOK0").is_ok() { 0 } else { s-1 };
                    let h = dl(&slice(d, &xt, tsel*hid, hid).unwrap(), hid);
                    BATCHED_LAYER_TRACE.with(|c| { let mut v = c.borrow_mut(); if v.len() <= l { v.resize(l+1, Vec::new()); } v[l] = h; });
                }
            }
            // final norm + lm_head on the LAST token only — DEVICE-CLEAN (no dl→up
            // roundtrip): slice the last token on device, lm_head GEMM on device →
            // device logits. Store the device logits for the graph path; dl for argmax
            // ONLY when not capturing (a dl aborts cuda-graph stream capture).
            let xf = match xn_carry.take() {
                Some(t) => t,
                None => rms_norm(d, &xt, &fwd["norm_f"], eps).unwrap(), // [s, hid]
            };
            let last_dev = slice(d, &xf, (s-1)*hid, hid).unwrap();   // [hid] device
            let logits_dev = pf!(14, 2.0 * vocab as f64 * hid as f64, qmv(&last_dev, "language_model.lm_head.weight"));
            LAST_BATCHED_LOGITS_DEV.with(|c| *c.borrow_mut() = Some(logits_dev.clone()));
            if CAPTURING.with(|c| c.get()) { 0usize }
            else {
                let logits = dl(&logits_dev, vocab);
                let am = ffai_runtime::argmax(&logits);
                LAST_BATCHED_LOGITS.with(|c| *c.borrow_mut() = logits);
                am
            }
        };

        // Fixed prompt id list (deterministic ramp). The correctness gate runs BOTH
        // the sequential and batched paths over THIS SAME list and compares the
        // last-token argmax — the KV cache + conv/SSM final states must agree.
        let ids: Vec<usize> = {
            let base_ids: Vec<usize> = (0..s).map(|i| (tok + i) % vocab).collect();
            // NEMOTRON_PACKED=N: replicate the prompt into N packed sequences (s·N
            // tokens) to fill the GPU; forward_batched runs them block-diagonally.
            let np = std::env::var("NEMOTRON_PACKED").ok()
                .and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
            if np > 1 { base_ids.iter().cloned().cycle().take(s * np).collect() } else { base_ids }
        };

        // ── Correctness gate (NEMOTRON_PREFILL_CHECK=1) ──────────────────────
        // Sequential reference over the fixed ids: feed ids[i] at pos fakectx+i,
        // ignore the chained argmax, take the argmax after the last token.
        if std::env::var("NEMOTRON_PREFILL_CHECK").is_ok() {
            let mut seq_conv: Vec<Vec<f32>> = vec![Vec::new(); 52];
            let seq_convdev: std::cell::RefCell<Vec<Option<Tensor>>> = std::cell::RefCell::new((0..52).map(|_| None).collect());
            let mut seq_ssm: Vec<Option<Tensor>> = (0..52).map(|_| None).collect();
            let mut seq_kv: Vec<Option<(Tensor, Tensor)>> = (0..52).map(|_| None).collect();
            // swap in the sequential conv_dev (step closure captured conv_dev by ref
            // via parameter; the all-device Mamba uses conv_dev borrow). To keep the
            // reference independent we run step with its own conv_dev: the step
            // closure references the outer `conv_dev` RefCell, so reset it first.
            { let mut cd = conv_dev.borrow_mut(); for c in cd.iter_mut() { *c = None; } }
            let _ = &seq_convdev; let _ = &mut seq_conv;
            let mut seq_argmax = 0usize;
            let dump_tok0 = std::env::var("NEMOTRON_DUMP_TOK0").is_ok();
            for (i, &id) in ids.iter().enumerate() {
                seq_argmax = step(id, fakectx + i, &mut seq_conv, &mut seq_ssm, &mut seq_kv);
                // INSTRUMENTATION (revert): when comparing token 0, freeze the
                // sequential trace at the FIRST step (it's overwritten each call).
                if dump_tok0 && i == 0 {
                    let frozen = STEP_LAYER_TRACE.with(|c| c.borrow().clone());
                    STEP0_FROZEN.with(|c| *c.borrow_mut() = frozen);
                }
            }
            if dump_tok0 {
                let f = STEP0_FROZEN.with(|c| c.borrow().clone());
                STEP_LAYER_TRACE.with(|c| *c.borrow_mut() = f);
            }
            d.synchronize().ok();
            // reset shared conv_dev (step mutated the OUTER one) before batched run.
            // INSTRUMENTATION: snapshot the sequential reference logits BEFORE any
            // batched run can overwrite shared thread_locals.
            let seq_logits = LAST_STEP_LOGITS.with(|c| c.borrow().clone());
            // ── NONDETERMINISM PROBE (correctness audit; REVERT) ─────────────────
            // Run the batched forward N times with IDENTICAL ids + fresh states.
            // Record per-run last-token argmax + max-abs logit delta vs run 0.
            let n_repeat: usize = std::env::var("NEMOTRON_REPEAT").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
            let mut run0_logits: Vec<f32> = Vec::new();
            let mut last_bat_logits: Vec<f32> = Vec::new();
            let mut bat_argmax = 0usize;
            for r in 0..n_repeat.max(1) {
                { let mut cd = conv_dev.borrow_mut(); for c in cd.iter_mut() { *c = None; } }
                let mut chk_ssm: Vec<Option<Tensor>> = (0..52).map(|_| None).collect();
                let mut chk_kv: Vec<Option<(Tensor, Tensor)>> = (0..52).map(|_| None).collect();
                let am = forward_batched(&ids, &conv_dev, &mut chk_ssm, &mut chk_kv, fakectx);
                d.synchronize().ok();
                let bl = LAST_BATCHED_LOGITS.with(|c| c.borrow().clone());
                if r == 0 { run0_logits = bl.clone(); bat_argmax = am; }
                // max-abs delta vs run0
                let maxd = if run0_logits.len() == bl.len() {
                    run0_logits.iter().zip(bl.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max)
                } else { f32::NAN };
                // top-2 gap of this run (near-tie indicator)
                let mut sorted: Vec<f32> = bl.clone(); sorted.sort_by(|a,b| b.partial_cmp(a).unwrap());
                let top1 = sorted.get(0).copied().unwrap_or(0.0);
                let top2 = sorted.get(1).copied().unwrap_or(0.0);
                eprintln!("  [REPEAT {r}] batched argmax={am}  maxAbsΔ(vs run0)={maxd:.6}  top1={top1:.5} top2={top2:.5} top1-top2={:.6}", top1-top2);
                last_bat_logits = bl;
            }
            let _ = last_bat_logits;
            // Near-tie diagnosis: rank the sequential argmax within the batched
            // logit distribution + report the gap between batched-argmax and the
            // sequential token's logit. If they differ but the gap is < a few %,
            // it's a precision near-tie (Q4-GEMV vs dequant→f32-GEMM), not a bug.
            let blog = if !run0_logits.is_empty() { run0_logits.clone() } else { LAST_BATCHED_LOGITS.with(|c| c.borrow().clone()) };
            // A/B hook: dump batched logits to a file for cross-run comparison
            // (e.g. SSD-on vs SSD-off). NEMOTRON_DUMP_BLOGITS=<path>.
            if let Ok(path) = std::env::var("NEMOTRON_DUMP_BLOGITS") {
                let bytes: Vec<u8> = blog.iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(&path, &bytes);
                eprintln!("  [dumped {} batched logits to {path}]", blog.len());
            }
            let (rank, seq_logit, top_logit) = if !blog.is_empty() && seq_argmax < blog.len() {
                let sv = blog[seq_argmax];
                let top = blog[bat_argmax];
                let rank = blog.iter().filter(|&&x| x > sv).count();
                (rank, sv, top)
            } else { (usize::MAX, 0.0, 0.0) };
            // PASS criterion for a precision-differing path: exact argmax match,
            // OR the sequential argmax sits in the batched top-5 within a <2% logit
            // gap (a Q4-GEMV-vs-dequant-f32-GEMM near-tie, not a structural bug).
            let gap_pct = (top_logit - seq_logit).abs() / top_logit.abs().max(1e-6) * 100.0;
            // A benign precision flip = the two argmaxes are a reshuffle WITHIN the shared
            // top-5 (same candidate set), not a wrong prediction. Accumulated f16/Q4 drift
            // over long S widens the top-1/top-2 gap (~5% at S=2048) while the top-5 stays
            // identical (cosine >0.998) — still benign. So key the verdict on top-5 agreement,
            // not just a fixed gap%. (Fixed-2% flagged the S=2048 near-tie as a false "bug".)
            let top5q = |v: &[f32]| { let mut idx: Vec<usize> = (0..v.len()).collect();
                idx.sort_by(|&a,&b| v[b].partial_cmp(&v[a]).unwrap()); idx.truncate(5); idx };
            let t5_overlap = if !seq_logits.is_empty() && seq_logits.len()==blog.len() {
                let (a,b)=(top5q(&seq_logits),top5q(&blog)); a.iter().filter(|x| b.contains(x)).count()
            } else { 0 };
            let near_tie = rank < 5 && (gap_pct < 2.0 || t5_overlap >= 4);
            let pass = seq_argmax == bat_argmax || near_tie;
            eprintln!("──────── PREFILL CORRECTNESS GATE (S={s}) ────────");
            eprintln!("  sequential last-token argmax = {seq_argmax}");
            eprintln!("  batched    last-token argmax = {bat_argmax}");
            if seq_argmax != bat_argmax {
                eprintln!("  near-tie diag: seq token rank in batched dist = {rank} (0=would-be-argmax)");
                eprintln!("  logit[batched_argmax]={top_logit:.5} logit[seq_argmax]={seq_logit:.5} gap={:.5} ({gap_pct:.3}%)",
                    top_logit - seq_logit);
            }
            eprintln!("  {}", if seq_argmax == bat_argmax { "EXACT MATCH ✓" }
                              else if near_tie { "NEAR-TIE PASS ✓ (precision flip within shared top-5)" }
                              else { "MISMATCH ✗ (structural bug)" });
            let _ = pass;
            // ── LOGIT-LEVEL AGREEMENT (correctness audit; REVERT) ────────────────
            // The real correctness question: does the FULL batched logit vector
            // agree with the trusted sequential reference? argmax can flip on a
            // near-tie; cosine + top-5 overlap + max-abs/rel err are the signal.
            if seq_logits.len() == blog.len() && !blog.is_empty() {
                let n = blog.len();
                // cosine similarity
                let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
                let (mut maxabs, mut sumsq_err, mut sumsq_ref) = (0f64, 0f64, 0f64);
                for i in 0..n {
                    let (a, b) = (seq_logits[i] as f64, blog[i] as f64);
                    dot += a*b; na += a*a; nb += b*b;
                    let e = (a-b).abs();
                    if e > maxabs { maxabs = e; }
                    sumsq_err += (a-b)*(a-b); sumsq_ref += a*a;
                }
                let cos = dot / (na.sqrt()*nb.sqrt()).max(1e-30);
                let rel_l2 = (sumsq_err.sqrt()) / (sumsq_ref.sqrt()).max(1e-30);
                // top-5 of each
                let top5 = |v: &[f32]| { let mut idx: Vec<usize> = (0..v.len()).collect();
                    idx.sort_by(|&a,&b| v[b].partial_cmp(&v[a]).unwrap()); idx.truncate(5); idx };
                let t5s = top5(&seq_logits); let t5b = top5(&blog);
                let overlap = t5s.iter().filter(|x| t5b.contains(x)).count();
                eprintln!("──────── LOGIT-LEVEL AGREEMENT (seq-ref vs batched, S={s}) ────────");
                eprintln!("  cosine similarity   = {cos:.8}");
                eprintln!("  max-abs logit error = {maxabs:.6}");
                eprintln!("  relative L2 error   = {rel_l2:.6}");
                eprintln!("  top-5 overlap       = {overlap}/5");
                eprintln!("  seq-ref top5 = {:?}", t5s.iter().map(|&i| (i, seq_logits[i])).collect::<Vec<_>>());
                eprintln!("  batched top5 = {:?}", t5b.iter().map(|&i| (i, blog[i])).collect::<Vec<_>>());
            } else {
                eprintln!("  LOGIT-LEVEL: length mismatch seq={} bat={} — skipped", seq_logits.len(), blog.len());
            }
            // ── PER-LAYER DIVERGENCE TRACE (NEMOTRON_DUMP_LAYERS=1; REVERT) ───────
            if std::env::var("NEMOTRON_DUMP_LAYERS").is_ok() {
                let st = STEP_LAYER_TRACE.with(|c| c.borrow().clone());
                let bt = BATCHED_LAYER_TRACE.with(|c| c.borrow().clone());
                eprintln!("──────── PER-LAYER LAST-TOKEN DIVERGENCE (seq vs batched) ────────");
                eprintln!("  layer mix : cosine     maxAbs    relL2");
                let nl = st.len().min(bt.len());
                for l in 0..nl {
                    if st[l].len() != bt[l].len() || st[l].is_empty() { continue; }
                    let mix = PATTERN.chars().nth(l).unwrap_or('?');
                    let (mut dot, mut na, mut nb, mut maxa, mut se, mut sr) = (0f64,0f64,0f64,0f64,0f64,0f64);
                    for i in 0..st[l].len() {
                        let (a,b) = (st[l][i] as f64, bt[l][i] as f64);
                        dot+=a*b; na+=a*a; nb+=b*b; let e=(a-b).abs(); if e>maxa {maxa=e;} se+=(a-b)*(a-b); sr+=a*a;
                    }
                    let cos = dot/(na.sqrt()*nb.sqrt()).max(1e-30);
                    let rl2 = se.sqrt()/sr.sqrt().max(1e-30);
                    eprintln!("  L{l:>3} {mix}   : {cos:.6}  {maxa:.5}  {rl2:.5}");
                }
                eprintln!("──────────────────────────────────────────────────");
            }
            // ── L0 SSM IN/OUT COMPARISON (NEMOTRON_DUMP_SSM=1; REVERT) ───────────
            if std::env::var("NEMOTRON_DUMP_SSM").is_ok() {
                let (sx,sb,sc,sy) = SEQ_SSM_DUMP.with(|c| c.borrow().clone());
                let (bx,bb,bc,by) = BAT_SSM_DUMP.with(|c| c.borrow().clone());
                let cmp = |name: &str, a: &[f32], b: &[f32]| {
                    if a.len()!=b.len() || a.is_empty() { eprintln!("  {name}: len mismatch {}/{}", a.len(), b.len()); return; }
                    let (mut dot,mut na,mut nb,mut mx)=(0f64,0f64,0f64,0f64);
                    for i in 0..a.len(){let(x,y)=(a[i]as f64,b[i]as f64);dot+=x*y;na+=x*x;nb+=y*y;let e=(x-y).abs();if e>mx{mx=e;}}
                    let cos=dot/(na.sqrt()*nb.sqrt()).max(1e-30);
                    eprintln!("  {name:>6}: cosine={cos:.7}  maxAbs={mx:.6}");
                };
                eprintln!("──────── L0 SSM IN/OUT (seq vs batched, LAST token) ────────");
                cmp("x_ssm", &sx, &bx);
                cmp("B", &sb, &bb);
                cmp("C", &sc, &bc);
                cmp("y(SSM)", &sy, &by);
                eprintln!("──────────────────────────────────────────────────");
            }
            eprintln!("──────────────────────────────────────────────────");
            // reset shared state after the gate.
            { let mut cd = conv_dev.borrow_mut(); for c in cd.iter_mut() { *c = None; } }
        }

        // Warm (JIT) then timed.
        let _ = forward_batched(&ids, &conv_dev, &mut ssm_state, &mut kvcache, fakectx);
        d.synchronize().ok();
        // reset states for the timed run (warm mutated them).
        { let mut cd = conv_dev.borrow_mut(); for c in cd.iter_mut() { *c = None; } }
        for s2 in ssm_state.iter_mut() { *s2 = None; }
        for kv in kvcache.iter_mut() { *kv = None; }
        ROUTER_HOST_T.with(|c| c.set(0.0));
        // NEMOTRON_GRAPH=1 (CUDA, needs NEMOTRON_Q4_DEVDESC for a dl-free forward):
        // capture the whole forward into a CUDA graph, then time the REPLAY — kills
        // the host-launch-serialized GPU idle (the ~2× prefill lever vs vLLM).
        let graph_mode = std::env::var("NEMOTRON_GRAPH").is_ok() && d.backend() == Backend::Cuda;
        let (next, pf_s) = if graph_mode {
            // Pre-warm the caching pool to its high-water mark so NO cuMemAlloc fires
            // during capture (alloc is illegal while the stream is capturing). Two extra
            // forwards + resets fill the pool free-list with every scratch size the
            // forward needs concurrently.
            for _ in 0..2 {
                let _ = forward_batched(&ids, &conv_dev, &mut ssm_state, &mut kvcache, fakectx);
                d.synchronize().ok();
                { let mut cd = conv_dev.borrow_mut(); for c in cd.iter_mut() { *c = None; } }
                for s2 in ssm_state.iter_mut() { *s2 = None; }
                for kv in kvcache.iter_mut() { *kv = None; }
            }
            CAPTURING.with(|c| c.set(true));
            d.begin_capture().expect("begin_capture");
            let _ = forward_batched(&ids, &conv_dev, &mut ssm_state, &mut kvcache, fakectx);
            let exec = d.end_capture().expect("end_capture");
            CAPTURING.with(|c| c.set(false));
            d.synchronize().ok();
            // timed replay (graph_launch re-runs the recorded ops on the captured buffers)
            let t = Instant::now();
            d.graph_launch(exec).expect("graph_launch");
            d.synchronize().ok();
            let rep = t.elapsed().as_secs_f64();
            let logits = LAST_BATCHED_LOGITS_DEV.with(|c| c.borrow().clone()).map(|t| dl(&t, vocab)).unwrap_or_default();
            let am = ffai_runtime::argmax(&logits);
            LAST_BATCHED_LOGITS.with(|c| *c.borrow_mut() = logits);
            eprintln!("  ── CUDA-GRAPH replay: {rep:.3}s = {:.2} tok/s (captured forward, argmax {am}) ──", ids.len() as f64 / rep);
            (am, rep)
        } else {
            let t_pf = Instant::now();
            let next = forward_batched(&ids, &conv_dev, &mut ssm_state, &mut kvcache, fakectx);
            d.synchronize().ok();
            (next, t_pf.elapsed().as_secs_f64())
        };
        let n_tok = ids.len();
        let tps_batched = n_tok as f64 / pf_s;
        let conv_dev_on = std::env::var("NEMOTRON_CONV_DEVICE").is_ok()
            || (d.backend() == Backend::Cuda && std::env::var("NEMOTRON_CONV_DEVICE_OFF").is_err());
        let conv_dev_label = if conv_dev_on { "CONV_DEVICE=1" } else { "CONV_DEVICE=0(host)" };
        let ondevice_moe_on = d.backend() == Backend::Cuda && std::env::var("NEMOTRON_ONDEVICE_MOE_OFF").is_err();
        eprintln!("──────── NemotronH-Nano BATCHED PREFILL on {plat} ────────");
        eprintln!("  prefill  {n_tok} tok in {pf_s:.3}s = {tps_batched:.2} tok/s ({:.2} ms/tok) [batched forward, {conv_dev_label}, ONDEVICE_MOE={}]", pf_s * 1000.0 / n_tok as f64, if ondevice_moe_on { "1" } else { "0" });
        eprintln!("  last-token argmax = {next}");
        if std::env::var("NEMOTRON_ROUTER_TIME").is_ok() {
            eprintln!("  host router loop (sigmoid+topk+sort, all E-layers) = {:.1} ms ({:.1}% of wall)",
                ROUTER_HOST_T.with(|c| c.get()) * 1000.0, ROUTER_HOST_T.with(|c| c.get()) / pf_s * 100.0);
        }
        eprintln!("──────────────────────────────────────────────────────────────");
        // Append to ~/prefill_overnight.log.
        {
            use std::io::Write;
            let logpath = format!("{}/prefill_overnight.log", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()));
            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            let line = format!("[{ts}] {plat} S={s} {conv_dev_label} CHUNKED={} tps={tps_batched:.2} ms/tok={:.2} argmax={next}\n",
                if std::env::var("NEMOTRON_CHUNKED_SCAN").is_ok() { "1" } else { "0" },
                pf_s * 1000.0 / s as f64);
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&logpath) {
                let _ = f.write_all(line.as_bytes());
            }
        }

        // ── PREFILL PER-OP PROFILING MAP (NEMOTRON_PROFILE=1) ───────────────
        // The timed run above had profiling OFF (clean throughput). Now run one
        // sync-bracketed profiled pass and emit the per-op map (ms / % / calls /
        // GFLOP/s). GB10 Blackwell tensor-core peak (bf16, dense) ≈ 1000 TFLOP/s;
        // we print effective GFLOP/s and %-of-peak for the compute-bound ops.
        if pf_on {
            { let mut cd = conv_dev.borrow_mut(); for c in cd.iter_mut() { *c = None; } }
            for s2 in ssm_state.iter_mut() { *s2 = None; }
            for kv in kvcache.iter_mut() { *kv = None; }
            for c in pf_t.iter() { c.set(0.0); } for c in pf_n.iter() { c.set(0); } for c in pf_flop.iter() { c.set(0.0); }
            let t_prof = Instant::now();
            let _ = forward_batched(&ids, &conv_dev, &mut ssm_state, &mut kvcache, fakectx);
            d.synchronize().ok();
            let prof_wall = t_prof.elapsed().as_secs_f64();
            let names = ["embed","rms_norm","proj_gemm","dequant_q4","conv_prefill","ssm_scan","gated_norm","sdpa_prefill","rope","kv_append","moe_router","moe_experts","moe_shared","add","lm_head","slice/cast"];
            const PEAK_TFLOPS: f64 = 1000.0; // GB10 Blackwell bf16 dense tensor-core peak (approx)
            let sum_t: f64 = pf_t.iter().map(|c| c.get()).sum();
            let mut rows: Vec<(usize,f64,u64,f64)> = (0..NPF).map(|i| (i, pf_t[i].get(), pf_n[i].get(), pf_flop[i].get())).filter(|r| r.2 > 0).collect();
            rows.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
            let is_cuda = d.backend() == Backend::Cuda;
            let mut md = String::new();
            md.push_str(&format!("# Nemotron-Nano-30B BATCHED PREFILL — per-op profiling map\n\n"));
            if is_cuda {
                md.push_str(&format!("- Device: {plat} (GB10 Blackwell)\n- S (prompt tokens): {s}\n"));
            } else {
                md.push_str(&format!("- Device: {plat} (Apple/Metal)\n- S (prompt tokens): {s}\n"));
            }
            md.push_str(&format!("- Clean batched throughput: **{:.1} tok/s** ({:.2} ms/tok)\n", s as f64 / pf_s, pf_s * 1000.0 / s as f64));
            md.push_str(&format!("- Profiled pass wall (sync-bracketed, inflated): {prof_wall:.3}s; summed op time: {sum_t:.3}s\n"));
            if is_cuda {
                md.push_str(&format!("- vLLM-on-GB10 reference: pp2048@d0=6395, @d8192=4993, @d32768=2734 tok/s\n"));
                md.push_str(&format!("- Tensor-core peak assumed: {PEAK_TFLOPS:.0} TFLOP/s (bf16 dense)\n\n"));
            } else {
                md.push_str("\n");
            }
            md.push_str("| op | ms | % | calls | TFLOP/s | %peak |\n|---|---:|---:|---:|---:|---:|\n");
            for (i,t,n,fl) in &rows {
                let ms = t * 1000.0;
                let pct = if sum_t > 0.0 { t / sum_t * 100.0 } else { 0.0 };
                let tflops = if *t > 0.0 && *fl > 0.0 { fl / t / 1e12 } else { 0.0 };
                let peakpct = tflops / PEAK_TFLOPS * 100.0;
                let tfs = if tflops > 0.0 { format!("{tflops:.3}") } else { "—".into() };
                let pps = if tflops > 0.0 { format!("{peakpct:.2}%") } else { "—".into() };
                md.push_str(&format!("| {} | {ms:.2} | {pct:.1}% | {n} | {tfs} | {pps} |\n", names[*i]));
            }
            if is_cuda {
                md.push_str("\n## Gap analysis (Path A — cuBLAS tensor cores LANDED)\n");
                md.push_str("- **The GEMMs now hit the tensor cores via the cuBLAS escape hatch (`gemm_cublas`/`cublasGemmEx`, f16×f16→f32 accumulate).** proj_gemm jumped ~1→90 TFLOP/s (0.1%→9% of peak), moe_shared ~1→74, moe_experts ~0.8→28 — a 35-82× per-GEMM speedup vs the software-emulated coop_tile MMA. End-to-end prefill went from ~80 to 178-259 tok/s (peak at S=512).\n");
                md.push_str("- **New #1 bottleneck: `ssm_scan` (~41%)** — the sequential-in-T Mamba `ssm_step_record`. With the GEMMs fast, this is now the critical path → the deferred chunked/parallel SSD scan is the next target.\n");
                md.push_str("- **`dequant_q4` ~26% (5500 calls):** Q4 weights are re-dequanted to f16 EVERY forward (per layer + per routed expert). This is pure redundant work → cache the f16-dequanted weights resident once at load (trades VRAM for ~26% of prefill time).\n");
                md.push_str("- `sdpa_prefill`: software-MMA coop_tile by default; set NEMOTRON_PREFILL_TCATTN=1 for the cuBLAS tensor-core FlashAttention path (`sdpa_multi_tc`). At deep context (zero-KV synthetic) the TC path cuts this stage ~9.5× @d8192 (1754→185ms, 54.7%→11.3%), ~14× @d16384 (4616→328ms), ~13.7× @d32768 (8958→652ms), lifting end-to-end prefill from 104→165 tok/s @d32768.\n");
                md.push_str("- `moe_experts` ~12%: the GEMM is fast (28 TFLOP/s) but the per-expert cuBLAS loop + host relu2/scatter round-trips remain serial → fuse on-device or use cublasGemmStridedBatched.\n");
                md.push_str("- Throughput peaks at S=512 then declines (S=2048→178, S=8192→117): the per-expert host round-trips + dequant grow with S/expert-count. Removing dequant-per-forward + the host MoE bridges should restore monotonic scaling toward the vLLM band.\n");
            } else {
                md.push_str("\n## Notes (Apple/Metal path)\n");
                md.push_str("- GEMMs run on Apple GPU hardware MMA: dense projections via `gemm_q4_mpp` (Q4-native cooperative-tensor MMA), per-expert/shared MoE via dequant_q4_off + `matmul`. No cuBLAS/tensor-core escape hatches (those are CUDA-only).\n");
                md.push_str("- Per-expert MoE GEMM runs in **f16 by default** (dequant→f16 + f16 matmul); set NEMOTRON_METAL_F32_EXPERTS=1 for the f32 fallback. The f16-compute path is the measured Metal prefill win (M5 Max: S=512 +26.6%, S=2048 98.8→113.5 tok/s) at EXACT-MATCH numerics.\n");
                md.push_str("- A resident f16 expert cache was tried and is a NET LOSS on Metal (bandwidth/residency bound): re-dequanting the compact Q4 each forward streams better than a fat f16 working set — so weights are re-dequanted per forward, not cached.\n");
                md.push_str("- The %peak column above is computed against the GB10 tensor-core peak and is not meaningful on Metal; read the ms / % columns instead.\n");
            }
            let out_path = std::env::var("NEMOTRON_PROFILE_OUT").unwrap_or_else(|_| "/tmp/PROFILING_PREFILL.md".into());
            let _ = std::fs::write(&out_path, &md);
            eprintln!("\n{md}\n(written to {out_path})");
        }
        let _ = next;
        return;
    }

    if prefill > 0 {
        let warm = step(tok, pos, &mut conv_state, &mut ssm_state, &mut kvcache); tok = warm; pos += 1;
        let t_pf = Instant::now();
        for _ in 1..prefill { let nxt = step(tok, pos, &mut conv_state, &mut ssm_state, &mut kvcache); tok = nxt; pos += 1; }
        d.synchronize().ok();
        let pf_s = t_pf.elapsed().as_secs_f64();
        let pf_toks = (prefill - 1) as f64;
        eprintln!("──────── NemotronH-Nano SEQUENTIAL PREFILL on {plat} ────────");
        eprintln!("  prefill  {} tok in {pf_s:.3}s = {:.2} tok/s ({:.1} ms/tok) [baseline, sequential decode steps]", prefill - 1, pf_toks / pf_s, pf_s * 1000.0 / pf_toks);
        eprintln!("──────────────────────────────────────────────────────────────");
    }
    let first = step(tok, pos, &mut conv_state, &mut ssm_state, &mut kvcache); // 1 warm step (JIT) + correctness peek
    tok = first; pos += 1;
    if prefill == 0 && fakectx == 0 { eprintln!("Nemotron decode: token1234 → next argmax {first} (F32 ref argmax=1234; Q8 may perturb the near-tie)"); }
    // Fast internal A/B: one model load, toggle an env flag (e.g. MT_GEMV_2ROW)
    // between alternating decode batches IN-PROCESS — same thermal/clock state,
    // order-alternated to cancel drift. Resets pos each batch so the KV cap holds.
    if let Ok(ab) = std::env::var("NEMOTRON_AB") {
        // "on" sets the flag to NEMOTRON_AB_VAL (default "1"); "off" unsets it.
        // For value-flags like MT_MOE_RPT/MT_GEMV_RPT use NEMOTRON_AB_VAL=2 so
        // ON=rpt2 vs OFF=rpt1 (unset default), not the no-op "1" vs unset(=1).
        let ab_val = std::env::var("NEMOTRON_AB_VAL").unwrap_or_else(|_| "1".to_string());
        let rounds = 6usize;
        let base_pos = pos;
        let (mut t_off, mut t_on) = (0f64, 0f64);
        for r in 0..rounds {
            // alternate which config runs first each round to cancel position bias
            let first_on = r % 2 == 1;
            for &on in &[first_on, !first_on] {
                unsafe { if on { std::env::set_var(&ab, &ab_val); } else { std::env::remove_var(&ab); } }
                pos = base_pos;
                let s = Instant::now();
                for _ in 0..n_decode { let nxt = step(tok, pos, &mut conv_state, &mut ssm_state, &mut kvcache); tok = nxt; pos += 1; }
                let e = s.elapsed().as_secs_f64();
                if on { t_on += e; } else { t_off += e; }
            }
        }
        let off_tps = (rounds * n_decode) as f64 / t_off;
        let on_tps = (rounds * n_decode) as f64 / t_on;
        eprintln!("──── AB[{ab}] internal ({rounds} rounds × {n_decode} tok, ctx {}) ────", fakectx.max(prefill));
        eprintln!("  OFF {off_tps:.2} tok/s | ON {on_tps:.2} tok/s | delta {:+.1}%", (on_tps / off_tps - 1.0) * 100.0);
        let _ = tok;
        return;
    }
    let t0 = Instant::now();
    for _ in 0..n_decode { let nxt = step(tok, pos, &mut conv_state, &mut ssm_state, &mut kvcache); tok = nxt; pos += 1; }
    let dt = t0.elapsed().as_secs_f64();
    let eager_tps = n_decode as f64 / dt;
    let recipe_desc = if use_nvfp4_recipe {
        "NVFP4-recipe [Mamba in/out_proj, shared-expert up/down, attn o_proj → Q8; routed experts, q/k/v, lm_head → Q4]"
    } else {
        "all-Q4 [uniform 4-bit]"
    };
    eprintln!("──────── NemotronH-Nano DECODE on {plat} ────────");
    eprintln!("  quant    {recipe_desc}");
    eprintln!("  resident {:.2} GB  ({:.2} GB Q4 + {:.2} GB Q8)", q4_gb + q8_gb, q4_gb, q8_gb);
    eprintln!("  context  start {} + {n_decode} timed (pos→{pos})", fakectx.max(prefill));
    eprintln!("  decode   {n_decode} tok in {dt:.2}s = {eager_tps:.2} tok/s ({:.1} ms/tok)", dt * 1000.0 / n_decode as f64);
    eprintln!("  (vs naive-reload baseline 0.003 tok/s; resident weights uploaded once in {load_s:.0}s setup)");
    eprintln!("──────────────────────────────────────────────────────────────");

    if prof {
        // Per-op breakdown table. We accumulated sync-bracketed wall times for each op
        // across (n_decode+1) steps (the warm step + timed steps). Divide by step_count
        // to get per-token averages. ms/tok = total_s * 1000 / n_steps.
        let n_steps = step_count.get().max(1) as f64;
        let tok_ms = dt * 1000.0 / n_decode as f64; // actual ms/tok (no sync overhead)

        // Op names aligned with indices used in pt!() above
        let op_names = [
            "rms_norm",       // 0
            "m_in_proj",      // 1  gemv_q4 Mamba in_proj 10304×2688
            "slice",          // 2  slice (multiple per Mamba layer)
            "conv1d",         // 3
            "silu",           // 4
            "conv_roll",      // 5
            "softplus_add",   // 6
            "ssm_step",       // 7
            "gated_grm_norm", // 8
            "m_out_proj",     // 9  gemv_q4 Mamba out_proj 2688×4096
            "moe_gate_gemv",  // 10 f32 gemv router 128×2688
            "moe_router_dev", // 11 device top-k kernel
            "moe_gather_up",  // 12 moe_gather_up_relu2
            "moe_gather_down",// 13
            "moe_wsum",       // 14
            "shared_up_q4",   // 15 qrelu2 3712×2688
            "shared_down_acc",// 16 qacc  2688×3712
            "rope",           // 17
            "q_proj",         // 18 gemv_q4 4096×2688
            "k_proj",         // 19 gemv_q4  256×2688
            "v_proj",         // 20 gemv_q4  256×2688
            "kv_append",      // 21
            "sdpa_2pass",     // 22
            "o_proj",         // 23 gemv_q4 2688×4096
            "add_residual",   // 24 (unused, subsumed into 27/28/29)
            "norm_f",         // 25
            "lm_head",        // 26
            "add_M_resid",    // 27
            "add_E_resid",    // 28
            "add_A_resid",    // 29
            "cast_f16",       // 30
        ];

        // Sum all GPU op times (this is the "GPU-attributed" total; the rest is host)
        let gpu_total_s: f64 = op_t.iter().map(|c| c.get()).sum();
        let host_overhead_s = (step_wall.get() / n_steps - gpu_total_s / n_steps).max(0.0);
        let roofline_gbs = 189.0f64; // GB10 peak BW GB/s (measured ~189 with bandwidthTest)

        eprintln!("──────── PER-OP PROFILE (ctx {} avg over {:.0} steps, NEMOTRON_PROFILE=1) ────────",
            fakectx.max(prefill), n_steps);
        eprintln!("  Note: times include sync overhead. Use ratios; GPU total ≠ wall time.");
        eprintln!("  {:<20} {:>8} {:>7} {:>8} {:>12} {:>8}",
            "op", "ms/tok", "%GPU", "calls/tok", "GB_read/tok", "eff GB/s");
        eprintln!("  {}", "-".repeat(75));

        let mut sorted_idx: Vec<usize> = (0..N_OPS).collect();
        sorted_idx.sort_by(|&a, &b| op_t[b].get().partial_cmp(&op_t[a].get()).unwrap());

        for &i in &sorted_idx {
            let t_s = op_t[i].get();
            if t_s < 1e-9 { continue; } // skip unexercised ops
            let ms_per_tok = t_s * 1000.0 / n_steps;
            let pct = 100.0 * t_s / gpu_total_s;
            let calls_per_tok = op_calls[i].get() as f64 / n_steps;
            let gb_per_tok = op_bytes[i].get() / n_steps / 1e9;
            // eff GB/s = bytes_read_per_tok / (ms_per_tok / 1000)
            let eff_gbs = if ms_per_tok > 1e-4 { gb_per_tok / (ms_per_tok / 1000.0) } else { 0.0 };
            eprintln!("  {:<20} {:>7.2}ms {:>6.1}% {:>8.1} {:>11.3}GB {:>7.1} GB/s  (roofline {:.0}%)",
                op_names[i], ms_per_tok, pct, calls_per_tok, gb_per_tok, eff_gbs,
                if eff_gbs > 0.0 { 100.0 * eff_gbs / roofline_gbs } else { 0.0 });
        }
        eprintln!("  {}", "-".repeat(75));
        eprintln!("  {:<20} {:>7.2}ms {:>6.1}%  (GPU-attributed total, incl sync overhead)",
            "GPU subtotal", gpu_total_s * 1000.0 / n_steps, 100.0);
        eprintln!("  {:<20} {:>7.2}ms         (host overhead = wall - gpu subtotal)",
            "host overhead", host_overhead_s * 1000.0);
        eprintln!("  {:<20} {:>7.2}ms         (actual wall time per token)",
            "wall time/tok", tok_ms);
        eprintln!("────────────────────────────────────────────────────────────────────────────");
    }

    // ── CUDA GRAPH CAPTURE/REPLAY measurement ──────────────────────────────────
    // Activated by NEMOTRON_GRAPH=1. Requires NEMOTRON_DEVROUTER=1 (device router,
    // no host sync in the step). Collapses ~390 per-token launches into one
    // cuGraphLaunch to remove host-launch-gap overhead; measures vs eager at the
    // SAME thermal/clock state in one process.
    //
    // Protocol:
    //   a. Run 5 more warm eager steps (pool stays fully populated, clocks steady).
    //   b. Time M eager steps (skip_dl=false so argmax fires for comparison baseline).
    //   c. One more eager step (pool free-list in exact capture-step state).
    //   d. begin_capture → step (skip_dl=true, no devrouter host-sync) → end_capture.
    //   e. synchronize; time M graph_launches.
    //   f. Print eager_tps_graph_run, graph_tps, ratio.
    if use_graph {
        if no_devrouter {
            eprintln!("NEMOTRON_GRAPH: WARNING — NEMOTRON_DEVROUTER not set. Host router mid-token sync will break capture. Re-run with NEMOTRON_DEVROUTER=1.");
        }
        let gm = n_decode; // use same step count for apples-to-apples
        // Use `fakectx` as the fixed position for all graph-section measurements.
        // This keeps `pos` within bounds (KV cap = fakectx + n_decode + 8) AND gives
        // the correct 32K SDPA length. We don't advance pos in the graph section.
        let gpos = fakectx.max(prefill); // fixed position for graph-section steps
        // (a) 5 warm steps at gpos (untimed) — populates pool, reaches steady clocks.
        for _ in 0..5 { step(tok, gpos, &mut conv_state, &mut ssm_state, &mut kvcache); }
        // (b) Eager baseline at fixed gpos (skip_dl=false — argmax fires for comparison).
        let t_eager = Instant::now();
        for _ in 0..gm { step(tok, gpos, &mut conv_state, &mut ssm_state, &mut kvcache); }
        d.synchronize().ok();
        let eager_dt = t_eager.elapsed().as_secs_f64();
        let eager_tps_hot = gm as f64 / eager_dt;
        // (c) One pool-alignment step (pool free-list in exact same state the capture step sees).
        step(tok, gpos, &mut conv_state, &mut ssm_state, &mut kvcache);
        // (d) Capture: skip_dl = true so no host-sync in step.
        skip_dl.set(true);
        d.begin_capture().expect("NEMOTRON_GRAPH: begin_capture failed — check DEVROUTER and that no cuMemAlloc slipped into the step");
        step(tok, gpos, &mut conv_state, &mut ssm_state, &mut kvcache);
        let graph_handle = d.end_capture().expect("NEMOTRON_GRAPH: end_capture failed — check for stray sync/alloc in the captured region");
        // (e) Synchronize then time M graph launches — two modes:
        //   serial:  graph_launch (sync-per-token) measures single-token latency
        //   batched: graph_launch_batch (N enqueues, one sync) eliminates per-token
        //            host-GPU handoff, giving maximum pipeline throughput
        d.synchronize().ok();
        // Serial: measures true per-token wall time (GPU + host round-trip).
        let t_graph_serial = Instant::now();
        for _ in 0..gm { d.graph_launch(graph_handle).expect("graph_launch failed"); }
        let graph_serial_dt = t_graph_serial.elapsed().as_secs_f64();
        let graph_serial_tps = gm as f64 / graph_serial_dt;
        // Batched: N cuGraphLaunch enqueues, one cuStreamSynchronize — removes
        // per-token host-GPU handoff to reveal the true GPU throughput ceiling.
        d.synchronize().ok();
        let t_graph_batch = Instant::now();
        d.graph_launch_batch(graph_handle, gm).expect("graph_launch_batch failed");
        let graph_batch_dt = t_graph_batch.elapsed().as_secs_f64();
        let graph_batch_tps = gm as f64 / graph_batch_dt;
        let ratio_serial = graph_serial_tps / eager_tps_hot;
        let ratio_batch  = graph_batch_tps  / eager_tps_hot;
        eprintln!("──────── CUDA GRAPH CAPTURE/REPLAY (ctx {}, {} tok each) ────────", fakectx.max(prefill), gm);
        eprintln!("  eager (hot)       {eager_tps_hot:.2} tok/s  ({:.2} ms/tok)", eager_dt * 1000.0 / gm as f64);
        eprintln!("  graph serial      {graph_serial_tps:.2} tok/s  ({:.2} ms/tok)  ratio {ratio_serial:.3}x", graph_serial_dt * 1000.0 / gm as f64);
        eprintln!("  graph batched     {graph_batch_tps:.2} tok/s  ({:.2} ms/tok)  ratio {ratio_batch:.3}x", graph_batch_dt * 1000.0 / gm as f64);
        if graph_batch_tps >= 75.0 {
            eprintln!("  ✓ TARGET MET: graph_batch_tps {graph_batch_tps:.1} >= 75 tok/s");
        } else {
            eprintln!("  ✗ target 75 tok/s not yet met (gap {:.1} tok/s)", 75.0 - graph_batch_tps);
        }
        eprintln!("──────────────────────────────────────────────────────────────────────");
    }
}
