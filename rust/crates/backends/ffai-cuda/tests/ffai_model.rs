// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! A full transformer decode layer (Qwen2-0.5B geometry) assembled from the
//! shared ffai-ops, run on CUDA through the Device trait, and checked against
//! an independent CPU reference. This is the proof that a model — not just an
//! op — runs on the shared layer: one builder (ffai_models::llama) + the op
//! seam produce a correct forward on real GB10 hardware.
#![cfg(feature = "cuda")]

use ffai_core::{DType, Device, Tensor};
use ffai_cuda::CudaDevice;
use ffai_models::llama::{LayerWeights, LlamaConfig, ModelWeights, decode_layer_self, forward_single};

fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
/// Deterministic small values in ~[-0.04, 0.04], varied by salt.
fn fill(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7 + salt * 131) % 97) as f32 - 48.0) * 0.0008).collect()
}
fn tens(dev: &dyn Device, data: &[f32], shape: Vec<usize>) -> Tensor {
    Tensor::new(dev.upload(&to_bytes(data)).unwrap(), shape, DType::F32)
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
/// out[m] = Σ_k mat[m*K + k] * vec[k]
fn matvec(mat: &[f32], vec: &[f32], m: usize, k: usize) -> Vec<f32> {
    (0..m).map(|r| (0..k).map(|c| mat[r * k + c] * vec[c]).sum()).collect()
}
fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let s = 1.0 / (ms + eps).sqrt();
    (0..n).map(|i| x[i] * s * w[i]).collect()
}

#[test]
fn qwen2_decode_layer_on_cuda_matches_cpu() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    // Qwen2-0.5B geometry.
    let cfg = LlamaConfig {
        hidden: 896,
        n_q_heads: 14,
        n_kv_heads: 2,
        head_dim: 64,
        intermediate: 4864,
        rope_theta: 1_000_000.0,
        eps: 1e-6,
        qk_norm: false,
        attn_bias: false,
    };
    let h = cfg.hidden;
    let qd = cfg.n_q_heads * cfg.head_dim; // 896
    let kd = cfg.n_kv_heads * cfg.head_dim; // 128
    let im = cfg.intermediate;

    // Weights (deterministic).
    let attn_norm = fill(h, 1);
    let wq = fill(qd * h, 2);
    let wk = fill(kd * h, 3);
    let wv = fill(kd * h, 4);
    let wo = fill(h * qd, 5);
    let mlp_norm = fill(h, 6);
    let w_gate = fill(im * h, 7);
    let w_up = fill(im * h, 8);
    let w_down = fill(h * im, 9);
    let x = fill(h, 10);

    let weights = LayerWeights {
        attn_norm: tens(dev.as_ref(), &attn_norm, vec![h]),
        wq: tens(dev.as_ref(), &wq, vec![qd, h]),
        wk: tens(dev.as_ref(), &wk, vec![kd, h]),
        wv: tens(dev.as_ref(), &wv, vec![kd, h]),
        wo: tens(dev.as_ref(), &wo, vec![h, qd]),
        bias_q: None,
        bias_k: None,
        bias_v: None,
        q_norm: None,
        k_norm: None,
        mlp_norm: tens(dev.as_ref(), &mlp_norm, vec![h]),
        w_gate: tens(dev.as_ref(), &w_gate, vec![im, h]),
        w_up: tens(dev.as_ref(), &w_up, vec![im, h]),
        w_down: tens(dev.as_ref(), &w_down, vec![h, im]),
    };
    let tx = tens(dev.as_ref(), &x, vec![h]);

    // ── GPU: the shared-layer forward ────────────────────────────────
    let out = decode_layer_self(dev.as_ref(), &cfg, &weights, &tx, 0).unwrap();
    dev.synchronize().unwrap();
    let mut ob = vec![0u8; h * 4];
    dev.download(out.buffer.as_ref(), &mut ob).unwrap();
    let got = from_bytes(&ob);

    // ── CPU reference (pos=0 → RoPE identity, n_kv=1 → attn=v per group) ─
    let hh = rmsnorm(&x, &attn_norm, cfg.eps);
    let q = matvec(&wq, &hh, qd, h);
    let _k = matvec(&wk, &hh, kd, h);
    let v = matvec(&wv, &hh, kd, h);
    let _ = &q; // rope identity at pos 0; q/k unchanged, attn ignores q for n_kv=1
    // attn[q_head, d] = v[kv_head, d], kv_head = q_head / heads_per_group
    let hpg = cfg.n_q_heads / cfg.n_kv_heads;
    let mut attn = vec![0.0f32; qd];
    for qh in 0..cfg.n_q_heads {
        let kvh = qh / hpg;
        for d in 0..cfg.head_dim {
            attn[qh * cfg.head_dim + d] = v[kvh * cfg.head_dim + d];
        }
    }
    let o = matvec(&wo, &attn, h, qd);
    let x1: Vec<f32> = (0..h).map(|i| x[i] + o[i]).collect();
    let h2 = rmsnorm(&x1, &mlp_norm, cfg.eps);
    let gate = matvec(&w_gate, &h2, im, h);
    let up = matvec(&w_up, &h2, im, h);
    let act: Vec<f32> = (0..im).map(|i| silu(gate[i]) * up[i]).collect();
    let down = matvec(&w_down, &act, h, im);
    let want: Vec<f32> = (0..h).map(|i| x1[i] + down[i]).collect();

    let mut err = 0.0f32;
    let mut worst = 0usize;
    for i in 0..h {
        let d = (got[i] - want[i]).abs();
        if d > err {
            err = d;
            worst = i;
        }
    }
    eprintln!(
        "transformer decode layer on CUDA vs CPU: max|Δ|={err:.3e} at [{worst}] (got {:.5}, want {:.5})",
        got[worst], want[worst]
    );
    assert!(err <= 5e-3, "decode layer mismatch: max|Δ|={err:.3e}");
    eprintln!("✅ Qwen2-0.5B-shaped decode layer runs on GB10 through the shared op layer, matches CPU.");
}

/// CPU-side weights for one layer (kept in sync with the GPU LayerWeights).
struct LW {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    mlp_norm: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}
fn gen_layer(cfg: &LlamaConfig, s: usize) -> LW {
    let (h, qd, kd, im) =
        (cfg.hidden, cfg.n_q_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim, cfg.intermediate);
    LW {
        attn_norm: fill(h, s),
        wq: fill(qd * h, s + 1),
        wk: fill(kd * h, s + 2),
        wv: fill(kd * h, s + 3),
        wo: fill(h * qd, s + 4),
        mlp_norm: fill(h, s + 5),
        w_gate: fill(im * h, s + 6),
        w_up: fill(im * h, s + 7),
        w_down: fill(h * im, s + 8),
    }
}
fn gpu_layer(dev: &dyn Device, cfg: &LlamaConfig, lw: &LW) -> LayerWeights {
    let (h, qd, kd, im) =
        (cfg.hidden, cfg.n_q_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim, cfg.intermediate);
    LayerWeights {
        attn_norm: tens(dev, &lw.attn_norm, vec![h]),
        wq: tens(dev, &lw.wq, vec![qd, h]),
        wk: tens(dev, &lw.wk, vec![kd, h]),
        wv: tens(dev, &lw.wv, vec![kd, h]),
        wo: tens(dev, &lw.wo, vec![h, qd]),
        bias_q: None,
        bias_k: None,
        bias_v: None,
        q_norm: None,
        k_norm: None,
        mlp_norm: tens(dev, &lw.mlp_norm, vec![h]),
        w_gate: tens(dev, &lw.w_gate, vec![im, h]),
        w_up: tens(dev, &lw.w_up, vec![im, h]),
        w_down: tens(dev, &lw.w_down, vec![h, im]),
    }
}
/// CPU reference for one decode layer (pos=0, n_kv=1).
fn cpu_layer(cfg: &LlamaConfig, x: &[f32], lw: &LW) -> Vec<f32> {
    let (h, qd, kd, im) =
        (cfg.hidden, cfg.n_q_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim, cfg.intermediate);
    let hh = rmsnorm(x, &lw.attn_norm, cfg.eps);
    let v = matvec(&lw.wv, &hh, kd, h);
    let hpg = cfg.n_q_heads / cfg.n_kv_heads;
    let mut attn = vec![0.0f32; qd];
    for qh in 0..cfg.n_q_heads {
        let kvh = qh / hpg;
        for d in 0..cfg.head_dim {
            attn[qh * cfg.head_dim + d] = v[kvh * cfg.head_dim + d];
        }
    }
    let o = matvec(&lw.wo, &attn, h, qd);
    let x1: Vec<f32> = (0..h).map(|i| x[i] + o[i]).collect();
    let h2 = rmsnorm(&x1, &lw.mlp_norm, cfg.eps);
    let gate = matvec(&lw.w_gate, &h2, im, h);
    let up = matvec(&lw.w_up, &h2, im, h);
    let act: Vec<f32> = (0..im).map(|i| silu(gate[i]) * up[i]).collect();
    let down = matvec(&lw.w_down, &act, h, im);
    (0..h).map(|i| x1[i] + down[i]).collect()
}

/// The FULL model graph — embedding → N layers → final norm → lm_head →
/// logits — run on CUDA via forward_single and checked against CPU. Proves
/// the whole transformer (not just a layer) runs on the shared op layer.
#[test]
fn qwen2_full_forward_logits_on_cuda_matches_cpu() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };
    let cfg = LlamaConfig {
        hidden: 896,
        n_q_heads: 14,
        n_kv_heads: 2,
        head_dim: 64,
        intermediate: 4864,
        rope_theta: 1_000_000.0,
        eps: 1e-6,
        qk_norm: false,
        attn_bias: false,
    };
    const VOCAB: usize = 2048;
    const N_LAYERS: usize = 2;
    let h = cfg.hidden;
    let token: u32 = 7;

    let embed = fill(VOCAB * h, 500);
    let final_norm = fill(h, 501);
    let lm_head = fill(VOCAB * h, 502);
    let layers: Vec<LW> = (0..N_LAYERS).map(|l| gen_layer(&cfg, 600 + l * 20)).collect();

    let mw = ModelWeights {
        embed: tens(dev.as_ref(), &embed, vec![VOCAB, h]),
        layers: layers.iter().map(|lw| gpu_layer(dev.as_ref(), &cfg, lw)).collect(),
        final_norm: tens(dev.as_ref(), &final_norm, vec![h]),
        lm_head: tens(dev.as_ref(), &lm_head, vec![VOCAB, h]),
    };

    // GPU
    let logits = forward_single(dev.as_ref(), &cfg, &mw, token).unwrap();
    dev.synchronize().unwrap();
    let mut lb = vec![0u8; VOCAB * 4];
    dev.download(logits.buffer.as_ref(), &mut lb).unwrap();
    let got = from_bytes(&lb);

    // CPU
    let mut x = embed[token as usize * h..(token as usize + 1) * h].to_vec();
    for lw in &layers {
        x = cpu_layer(&cfg, &x, lw);
    }
    let xn = rmsnorm(&x, &final_norm, cfg.eps);
    let want = matvec(&lm_head, &xn, VOCAB, h);

    let mut err = 0.0f32;
    for i in 0..VOCAB {
        err = err.max((got[i] - want[i]).abs());
    }
    // argmax (the predicted token) must agree exactly.
    let amax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    let (ga, ca) = (amax(&got), amax(&want));
    eprintln!("full forward logits on CUDA vs CPU: max|Δ|={err:.3e}, argmax gpu={ga} cpu={ca}");
    assert!(err <= 1e-2, "full forward mismatch: max|Δ|={err:.3e}");
    assert_eq!(ga, ca, "argmax (predicted token) disagrees");
    eprintln!("✅ Full {N_LAYERS}-layer Qwen2 forward → logits on GB10 through the shared op layer, matches CPU (same predicted token).");
}
