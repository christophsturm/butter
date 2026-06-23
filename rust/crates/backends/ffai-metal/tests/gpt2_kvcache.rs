// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! GPT-2 generation with a real **incremental KV cache** + **device-resident
//! weights** + throughput numbers. Weights are uploaded to the device ONCE;
//! prefill fills a per-layer K/V cache; each decode step forwards only the new
//! token and attends over the cache (n_kv = current length) — O(seq)/step, not
//! O(seq²). Only tiny per-step activations + the growing cache cross the bus.
//! Asserts generated tokens still match HF and prints prefill/decode tok/s.
use ffai_core::{DType, Tensor};
use ffai_metal::MetalDevice;
use ffai_loader::SafeTensors;
use ffai_ops::{add, gelu, gemv, layer_norm, matmul, sdpa_decode};
use std::time::Instant;

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

struct LW { ln1w: Tensor, ln1b: Tensor, cattn: Tensor, cattn_b: Tensor, cproj: Tensor, cproj_b: Tensor,
            ln2w: Tensor, ln2b: Tensor, fc: Tensor, fc_b: Tensor, proj: Tensor, proj_b: Tensor }

#[test]
fn gpt2_kvcache_decode_throughput() {
    let dir = std::env::var("GPT2_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let path = format!("{dir}/model.safetensors");
    let Ok(st) = SafeTensors::open(&path) else { eprintln!("no model at {path} — skipping"); return; };
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    let d = dev.as_ref();
    let plat = "Metal";

    let (hid, nh, hd, n_layers, vocab, eps) = (768usize, 12usize, 64usize, 12usize, 50257usize, 1e-5f32);
    let scale = 1.0 / (hd as f32).sqrt();
    let g = |name: &str| -> Vec<f32> { st.tensor_f32(name).unwrap().0 };
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };
    let conv_t = |w: &[f32], nin: usize, nout: usize| -> Vec<f32> { let mut o = vec![0.0f32; nin * nout]; for i in 0..nin { for j in 0..nout { o[j * nin + i] = w[i * nout + j]; } } o };
    let reorg = |m: &[f32], n: usize| -> Vec<f32> { let mut o = vec![0.0f32; nh * n * hd]; for t in 0..n { for h in 0..nh { for dd in 0..hd { o[h * n * hd + t * hd + dd] = m[t * hid + h * hd + dd]; } } } o };

    // ── upload all weights to the device ONCE (resident) ──
    let t_load = Instant::now();
    let wte = g("wte.weight"); let wpe = g("wpe.weight");
    let wte_t = up(&wte, vec![vocab, hid]); // lm_head (tied)
    let lwt: Vec<LW> = (0..n_layers).map(|l| { let p = format!("h.{l}"); LW {
        ln1w: up(&g(&format!("{p}.ln_1.weight")), vec![hid]), ln1b: up(&g(&format!("{p}.ln_1.bias")), vec![hid]),
        cattn: up(&conv_t(&g(&format!("{p}.attn.c_attn.weight")), hid, 3*hid), vec![3*hid, hid]), cattn_b: up(&g(&format!("{p}.attn.c_attn.bias")), vec![3*hid]),
        cproj: up(&conv_t(&g(&format!("{p}.attn.c_proj.weight")), hid, hid), vec![hid, hid]), cproj_b: up(&g(&format!("{p}.attn.c_proj.bias")), vec![hid]),
        ln2w: up(&g(&format!("{p}.ln_2.weight")), vec![hid]), ln2b: up(&g(&format!("{p}.ln_2.bias")), vec![hid]),
        fc: up(&conv_t(&g(&format!("{p}.mlp.c_fc.weight")), hid, 4*hid), vec![4*hid, hid]), fc_b: up(&g(&format!("{p}.mlp.c_fc.bias")), vec![4*hid]),
        proj: up(&conv_t(&g(&format!("{p}.mlp.c_proj.weight")), 4*hid, hid), vec![hid, 4*hid]), proj_b: up(&g(&format!("{p}.mlp.c_proj.bias")), vec![hid]),
    }}).collect();
    let lnf_w = up(&g("ln_f.weight"), vec![hid]); let lnf_b = up(&g("ln_f.bias"), vec![hid]);
    let load_s = t_load.elapsed().as_secs_f64();

    let mut kc: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
    let mut vc: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
    let argmax = |logits: &[f32]| (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();

    // forward ONE token at `pos`, extending the cache; resident weights, returns next argmax
    let step = |tok: usize, pos: usize, kc: &mut Vec<Vec<f32>>, vc: &mut Vec<Vec<f32>>| -> usize {
        let x0: Vec<f32> = (0..hid).map(|e| wte[tok*hid+e] + wpe[pos*hid+e]).collect();
        let mut xt = up(&x0, vec![hid]); // residual stream stays device-resident
        for (l, w) in lwt.iter().enumerate() {
            let h = layer_norm(d, &xt, &w.ln1w, &w.ln1b, eps).unwrap();
            let qkv_t = add(d, &gemv(d, &w.cattn, &h).unwrap(), &w.cattn_b).unwrap();
            let qkv = dl(&qkv_t, 3*hid); // the one unavoidable download — feeds the host KV cache
            kc[l].extend_from_slice(&qkv[hid..2*hid]); vc[l].extend_from_slice(&qkv[2*hid..3*hid]);
            let len = kc[l].len() / hid;
            let kt = up(&reorg(&kc[l], len), vec![nh, len, hd]);
            let vt = up(&reorg(&vc[l], len), vec![nh, len, hd]);
            let a = sdpa_decode(d, &up(&qkv[0..hid], vec![nh, hd]), &kt, &vt, hd, len as u32, len as u32, 1, scale).unwrap();
            let o = add(d, &gemv(d, &w.cproj, &a.reshaped(vec![hid])).unwrap(), &w.cproj_b).unwrap();
            xt = add(d, &xt, &o).unwrap(); // residual on-device (no round-trip)
            let h2 = layer_norm(d, &xt, &w.ln2w, &w.ln2b, eps).unwrap();
            let f = add(d, &gemv(d, &w.fc, &h2).unwrap(), &w.fc_b).unwrap();
            let act = gelu(d, &f).unwrap();
            let m = add(d, &gemv(d, &w.proj, &act).unwrap(), &w.proj_b).unwrap();
            xt = add(d, &xt, &m).unwrap(); // residual on-device
        }
        let xf = layer_norm(d, &xt, &lnf_w, &lnf_b, eps).unwrap();
        let logits = dl(&matmul(d, &wte_t, &xf.reshaped(vec![1, hid])).unwrap(), vocab);
        argmax(&logits)
    };

    // warmup: first dispatch of each kernel JIT-compiles (MSL) — exclude from timings
    let t_warm = Instant::now();
    { let mut wk = vec![Vec::new(); n_layers]; let mut wv = vec![Vec::new(); n_layers]; step(464, 0, &mut wk, &mut wv); }
    let warm_s = t_warm.elapsed().as_secs_f64();

    let prompt: Vec<usize> = vec![464, 3139, 286, 4881, 318]; // "The capital of France is"
    let t0 = Instant::now();
    let mut next = 0usize;
    for (pos, &tok) in prompt.iter().enumerate() { next = step(tok, pos, &mut kc, &mut vc); }
    let prefill_s = t0.elapsed().as_secs_f64();

    let mut outv = vec![next];
    let t1 = Instant::now();
    let mut pos = prompt.len();
    for _ in 0..29 { let tok = *outv.last().unwrap(); let nxt = step(tok, pos, &mut kc, &mut vc); outv.push(nxt); pos += 1; }
    let decode_s = t1.elapsed().as_secs_f64();

    let hf = [262usize, 3139, 286, 262, 4141, 2066, 11, 290, 262, 3139, 286, 262, 4141, 2066, 318, 262, 3139, 286, 262, 4141, 2066, 13, 198, 198, 464, 4141, 2066, 318, 262, 3139];
    eprintln!("GPT-2 KV-cache decode on {plat} (resident weights):");
    eprintln!("  weight upload (1x): {load_s:.3}s   kernel JIT warmup (1x): {warm_s:.3}s");
    eprintln!("  prefill {} tok in {:.3}s = {:.1} tok/s", prompt.len(), prefill_s, prompt.len() as f64 / prefill_s);
    eprintln!("  decode  {} tok in {:.3}s = {:.1} tok/s ({:.1} ms/tok)", outv.len(), decode_s, outv.len() as f64 / decode_s, decode_s * 1000.0 / outv.len() as f64);
    assert_eq!(&outv[..], &hf, "KV-cache generation diverges from HF");
    eprintln!("✅ Incremental KV-cache decode (resident weights) matches HF generate() exactly on {plat}.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--gpt2/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}
