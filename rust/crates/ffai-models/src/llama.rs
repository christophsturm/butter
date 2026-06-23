// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Transformer-LLM builder (Llama / Qwen / Mistral / Yi / Phi / SmolLM …).
//! One decode layer assembled entirely from [`ffai_ops`] — so it runs on any
//! backend that implements [`ffai_core::Device`]. Config-parameterized: the
//! same code serves every model in this family by varying [`LlamaConfig`].

use ffai_core::{Device, Error, Result, Tensor};
use ffai_ops as ops;

/// Architecture config shared by the transformer-LLM family.
#[derive(Debug, Clone, Copy)]
pub struct LlamaConfig {
    pub hidden: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rope_theta: f32,
    pub eps: f32,
    /// Qwen3-style per-head RMSNorm on Q and K before RoPE.
    pub qk_norm: bool,
    /// QKV projection bias (Qwen2/Qwen2.5).
    pub attn_bias: bool,
}

impl LlamaConfig {
    pub fn heads_per_group(&self) -> u32 {
        (self.n_q_heads / self.n_kv_heads) as u32
    }
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// Per-layer weights (dense). Row-major: a projection `[out, in]` is applied
/// as `gemv(W, x)`.
pub struct LayerWeights {
    pub attn_norm: Tensor, // [hidden]
    pub wq: Tensor,        // [n_q_heads*head_dim, hidden]
    pub wk: Tensor,        // [n_kv_heads*head_dim, hidden]
    pub wv: Tensor,        // [n_kv_heads*head_dim, hidden]
    pub wo: Tensor,        // [hidden, n_q_heads*head_dim]
    /// QKV biases `[*_heads*head_dim]` (None unless attn_bias).
    pub bias_q: Option<Tensor>,
    pub bias_k: Option<Tensor>,
    pub bias_v: Option<Tensor>,
    /// Qwen3 per-head Q/K RMSNorm weights `[head_dim]` (None unless qk_norm).
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    pub mlp_norm: Tensor,  // [hidden]
    pub w_gate: Tensor,    // [intermediate, hidden]
    pub w_up: Tensor,      // [intermediate, hidden]
    pub w_down: Tensor,    // [hidden, intermediate]
}

/// One transformer decode step for a single token, attending only to itself
/// (single-position KV — `n_kv = 1`). This exercises the full layer pipeline
/// — RMSNorm → QKV proj → RoPE → SDPA → O proj → residual → RMSNorm →
/// SwiGLU MLP → residual — through the shared op layer. The multi-position
/// KV cache (write new k/v at `pos`, attend over `[0, pos]`) is the model
/// loop's responsibility; this is the per-layer compute.
///
/// `x` is the `[hidden]` residual stream for the current token.
pub fn decode_layer_self(
    dev: &dyn Device,
    cfg: &LlamaConfig,
    w: &LayerWeights,
    x: &Tensor,
    pos: u32,
) -> Result<Tensor> {
    let hd = cfg.head_dim;
    let theta = cfg.rope_theta;

    // ── attention ────────────────────────────────────────────────────
    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;
    let mut q = ops::gemv(dev, &w.wq, &h)?;
    let mut k = ops::gemv(dev, &w.wk, &h)?;
    let mut v = ops::gemv(dev, &w.wv, &h)?;
    if cfg.attn_bias {
        q = ops::add(dev, &q, w.bias_q.as_ref().ok_or_else(|| Error::Msg("attn_bias set but bias_q missing".into()))?)?;
        k = ops::add(dev, &k, w.bias_k.as_ref().ok_or_else(|| Error::Msg("attn_bias set but bias_k missing".into()))?)?;
        v = ops::add(dev, &v, w.bias_v.as_ref().ok_or_else(|| Error::Msg("attn_bias set but bias_v missing".into()))?)?;
    }

    // Reshape to heads; optional Qwen3 per-head Q/K RMSNorm before RoPE.
    let q = q.reshaped(vec![cfg.n_q_heads, hd]);
    let k = k.reshaped(vec![cfg.n_kv_heads, hd]);
    let (q, k) = if cfg.qk_norm {
        let qn = w.q_norm.as_ref().ok_or_else(|| Error::Msg("qk_norm set but q_norm missing".into()))?;
        let kn = w.k_norm.as_ref().ok_or_else(|| Error::Msg("qk_norm set but k_norm missing".into()))?;
        (ops::rms_norm(dev, &q, qn, cfg.eps)?, ops::rms_norm(dev, &k, kn, cfg.eps)?)
    } else {
        (q, k)
    };

    // RoPE on Q and K (vanilla — freq-band scaling disabled).
    let q = ops::rope_llama(dev, &q, pos, theta, 1.0, 1.0, 1.0, 1e9)?;
    let k = ops::rope_llama(dev, &k, pos, theta, 1.0, 1.0, 1.0, 1e9)?;

    // Single-position attention: KV cache is just this token (n_kv=1, stride=1).
    let attn = ops::sdpa_decode(
        dev,
        &q,
        &k,
        &v.reshaped(vec![cfg.n_kv_heads, hd]),
        hd,
        1,
        1,
        cfg.heads_per_group(),
        cfg.attn_scale(),
    )?;

    let o = ops::gemv(dev, &w.wo, &attn.reshaped(vec![cfg.n_q_heads * hd]))?;
    let x1 = ops::add(dev, x, &o)?;

    // ── MLP (SwiGLU) ─────────────────────────────────────────────────
    let h2 = ops::rms_norm(dev, &x1, &w.mlp_norm, cfg.eps)?;
    let gate = ops::gemv(dev, &w.w_gate, &h2)?;
    let up = ops::gemv(dev, &w.w_up, &h2)?;
    let act = ops::swiglu(dev, &gate, &up)?;
    let down = ops::gemv(dev, &w.w_down, &act)?;
    let x2 = ops::add(dev, &x1, &down)?;

    Ok(x2)
}

/// Full model weights for the transformer-LLM family.
pub struct ModelWeights {
    pub embed: Tensor,           // [vocab, hidden]
    pub layers: Vec<LayerWeights>,
    pub final_norm: Tensor,      // [hidden]
    pub lm_head: Tensor,         // [vocab, hidden]  (untied; tie = embed)
}

/// Single-token forward: embed → every decode layer (self-attention) →
/// final RMSNorm → lm_head → next-token logits `[vocab]`. The whole model
/// graph on the shared op layer — embedding, all layers, and the head. (The
/// multi-position KV cache for sequences ≥ 2 is the decode loop's job; this
/// is the per-token compute that proves the full graph.)
pub fn forward_single(
    dev: &dyn Device,
    cfg: &LlamaConfig,
    w: &ModelWeights,
    token_id: u32,
) -> Result<Tensor> {
    let ids = Tensor::new(
        dev.upload(&token_id.to_le_bytes())?,
        vec![1],
        ffai_core::DType::U32,
    );
    let mut x = ops::gather(dev, &w.embed, &ids)?.reshaped(vec![cfg.hidden]);
    for layer in &w.layers {
        x = decode_layer_self(dev, cfg, layer, &x, 0)?;
    }
    let xn = ops::rms_norm(dev, &x, &w.final_norm, cfg.eps)?;
    ops::gemv(dev, &w.lm_head, &xn)
}

/// Load a Qwen3 (or any HF-named transformer-LLM) checkpoint from a parsed
/// SafeTensors blob, uploading every weight to `dev`. Maps the HF names
/// (`model.layers.N.self_attn.q_proj.weight`, …) to [`ModelWeights`]. Falls
/// back to tied embeddings when `lm_head.weight` is absent.
pub fn load_qwen3(
    dev: &dyn Device,
    st: &ffai_loader::SafeTensors,
    cfg: &LlamaConfig,
    n_layers: usize,
) -> Result<ModelWeights> {
    let up = |name: &str| -> Result<Tensor> {
        let (bytes, dt, shape) = st.tensor(name)?;
        Ok(Tensor::new(dev.upload(bytes)?, shape.to_vec(), dt))
    };

    let embed = up("model.embed_tokens.weight")?;
    let final_norm = up("model.norm.weight")?;
    let lm_head = match up("lm_head.weight") {
        Ok(t) => t,
        Err(_) => up("model.embed_tokens.weight")?, // tied
    };

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let p = format!("model.layers.{l}");
        layers.push(LayerWeights {
            attn_norm: up(&format!("{p}.input_layernorm.weight"))?,
            wq: up(&format!("{p}.self_attn.q_proj.weight"))?,
            wk: up(&format!("{p}.self_attn.k_proj.weight"))?,
            wv: up(&format!("{p}.self_attn.v_proj.weight"))?,
            wo: up(&format!("{p}.self_attn.o_proj.weight"))?,
            bias_q: if cfg.attn_bias { Some(up(&format!("{p}.self_attn.q_proj.bias"))?) } else { None },
            bias_k: if cfg.attn_bias { Some(up(&format!("{p}.self_attn.k_proj.bias"))?) } else { None },
            bias_v: if cfg.attn_bias { Some(up(&format!("{p}.self_attn.v_proj.bias"))?) } else { None },
            q_norm: if cfg.qk_norm {
                Some(up(&format!("{p}.self_attn.q_norm.weight"))?)
            } else {
                None
            },
            k_norm: if cfg.qk_norm {
                Some(up(&format!("{p}.self_attn.k_norm.weight"))?)
            } else {
                None
            },
            mlp_norm: up(&format!("{p}.post_attention_layernorm.weight"))?,
            w_gate: up(&format!("{p}.mlp.gate_proj.weight"))?,
            w_up: up(&format!("{p}.mlp.up_proj.weight"))?,
            w_down: up(&format!("{p}.mlp.down_proj.weight"))?,
        });
    }

    Ok(ModelWeights { embed, layers, final_norm, lm_head })
}

/// A model loaded from an HF directory, with its derived config.
pub struct LoadedModel {
    pub cfg: LlamaConfig,
    pub weights: ModelWeights,
    pub n_layers: usize,
    pub vocab: usize,
}

/// Load any dense transformer-LLM straight from an HF directory: parse
/// `config.json` for the geometry, detect arch flags from the tensor names
/// (qk-norm by `q_norm.weight`, QKV bias by `q_proj.bias`), and upload the
/// weights. This is what makes the whole Llama/Qwen/Mistral/Yi/Phi/SmolLM
/// family load with one code path — no per-model hardcoding.
pub fn load_hf(dev: &dyn Device, dir: &str) -> Result<LoadedModel> {
    let cfg_txt = std::fs::read_to_string(format!("{dir}/config.json"))
        .map_err(|e| Error::Msg(format!("read config.json: {e}")))?;
    let j: serde_json::Value =
        serde_json::from_str(&cfg_txt).map_err(|e| Error::Msg(format!("config.json: {e}")))?;
    let u = |k: &str| j[k].as_u64().map(|x| x as usize);
    let hidden = u("hidden_size").ok_or_else(|| Error::Msg("config: hidden_size".into()))?;
    let n_q_heads = u("num_attention_heads").ok_or_else(|| Error::Msg("config: n_heads".into()))?;
    let n_kv_heads = u("num_key_value_heads").unwrap_or(n_q_heads);
    let head_dim = u("head_dim").unwrap_or(hidden / n_q_heads);
    let intermediate = u("intermediate_size").ok_or_else(|| Error::Msg("config: inter".into()))?;
    let n_layers = u("num_hidden_layers").ok_or_else(|| Error::Msg("config: n_layers".into()))?;
    let vocab = u("vocab_size").ok_or_else(|| Error::Msg("config: vocab".into()))?;
    let rope_theta = j["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let eps = j["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;

    let st = ffai_loader::SafeTensors::open(&format!("{dir}/model.safetensors"))?;
    let qk_norm = st.info("model.layers.0.self_attn.q_norm.weight").is_some();
    let attn_bias = st.info("model.layers.0.self_attn.q_proj.bias").is_some();

    let cfg = LlamaConfig {
        hidden,
        n_q_heads,
        n_kv_heads,
        head_dim,
        intermediate,
        rope_theta,
        eps,
        qk_norm,
        attn_bias,
    };
    let weights = load_qwen3(dev, &st, &cfg, n_layers)?;
    Ok(LoadedModel { cfg, weights, n_layers, vocab })
}

// ════════════════════════════════════════════════════════════════════════
// GGUF loader + KV-cache decode (Qwen2.5 / llama.cpp tensor naming)
// ════════════════════════════════════════════════════════════════════════
//
// The SafeTensors path above expects HF-named, float-on-disk tensors. GGUF
// ships quantized (Q8_0 here) under llama.cpp names (`blk.N.attn_q.weight`…).
// `load_qwen_gguf` dequantizes every weight to f32 on the host and uploads it,
// so the whole forward graph runs in f32 — backend-agnostic (the f32 gemv /
// rms_norm / sdpa kernels are registry-tested on Metal, CUDA, and Vulkan).
//
// Geometry is read from GGUF metadata (`qwen2.*` / `general.*`), so the same
// builder serves any GGUF dense LLM in the Llama/Qwen family with this layout.

/// Read geometry from a GGUF's metadata and derive the [`LlamaConfig`].
/// Probes for the `qwen2`/`llama` arch prefix and detects the QKV-bias flag
/// from the presence of `blk.0.attn_q.bias`.
pub fn gguf_config(g: &ffai_loader::gguf::Gguf) -> Result<(LlamaConfig, usize, usize)> {
    // Architecture prefix for the geometry keys (e.g. "qwen2", "llama").
    let arch = g.meta_str("general.architecture").unwrap_or("qwen2").to_string();
    let mu = |suffix: &str| -> Option<u32> { g.meta_u32(&format!("{arch}.{suffix}")) };
    let need = |suffix: &str| -> Result<u32> {
        mu(suffix).ok_or_else(|| Error::Msg(format!("gguf: missing metadata {arch}.{suffix}")))
    };

    let hidden = need("embedding_length")? as usize;
    let n_layers = need("block_count")? as usize;
    let n_q_heads = need("attention.head_count")? as usize;
    let n_kv_heads = mu("attention.head_count_kv").unwrap_or(n_q_heads as u32) as usize;
    let intermediate = need("feed_forward_length")? as usize;
    let head_dim = mu("attention.key_length").map(|v| v as usize).unwrap_or(hidden / n_q_heads);
    let rope_theta = g.meta_f32(&format!("{arch}.rope.freq_base")).unwrap_or(10000.0);
    let eps = g
        .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    // GGUF stores vocab as the token-list length; fall back to the embedding rows.
    let vocab = g
        .metadata_arr_str
        .get("tokenizer.ggml.tokens")
        .map(|v| v.len())
        .or_else(|| g.tensor("token_embd.weight").map(|t| t.dims[t.dims.len() - 1] as usize))
        .ok_or_else(|| Error::Msg("gguf: cannot determine vocab size".into()))?;

    // Qwen2.5 has QKV projection bias and no q/k-norm (vs Qwen3).
    let attn_bias = g.tensor("blk.0.attn_q.bias").is_some();
    let qk_norm = g.tensor("blk.0.attn_q_norm.weight").is_some();

    let cfg = LlamaConfig {
        hidden,
        n_q_heads,
        n_kv_heads,
        head_dim,
        intermediate,
        rope_theta,
        eps,
        qk_norm,
        attn_bias,
    };
    Ok((cfg, n_layers, vocab))
}

/// Build [`ModelWeights`] from a GGUF file (Q8_0/F16/F32 dequantized to f32 and
/// uploaded). Maps the llama.cpp tensor names to the builder's slots and wires
/// the Qwen2.5 QKV biases when `cfg.attn_bias`. Parallel to [`load_qwen3`].
pub fn load_qwen_gguf(
    dev: &dyn Device,
    g: &ffai_loader::gguf::Gguf,
    cfg: &LlamaConfig,
    n_layers: usize,
) -> Result<ModelWeights> {
    // Dequant a GGUF tensor → f32, upload, tag with its on-disk dims.
    let up = |name: &str| -> Result<Tensor> {
        let t = g
            .tensor(name)
            .ok_or_else(|| Error::Msg(format!("gguf: tensor '{name}' not found")))?;
        let data = g.dequant_f32(name)?;
        // GGUF dims are stored fastest-first (col-major-ish): a [out,in] matrix
        // is listed as dims=[in, out]. The f32 buffer is already row-major
        // [out, in] (block r·in .. (r+1)·in), so report shape [out, in].
        let shape: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();
        let bytes: &[u8] = bytemuck_cast(&data);
        Ok(Tensor::new(dev.upload(bytes)?, shape, ffai_core::DType::F32))
    };

    let embed = up("token_embd.weight")?;
    let final_norm = up("output_norm.weight")?;
    let lm_head = match up("output.weight") {
        Ok(t) => t,
        Err(_) => up("token_embd.weight")?, // tied embeddings
    };

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let p = format!("blk.{l}");
        layers.push(LayerWeights {
            attn_norm: up(&format!("{p}.attn_norm.weight"))?,
            wq: up(&format!("{p}.attn_q.weight"))?,
            wk: up(&format!("{p}.attn_k.weight"))?,
            wv: up(&format!("{p}.attn_v.weight"))?,
            wo: up(&format!("{p}.attn_output.weight"))?,
            bias_q: if cfg.attn_bias { Some(up(&format!("{p}.attn_q.bias"))?) } else { None },
            bias_k: if cfg.attn_bias { Some(up(&format!("{p}.attn_k.bias"))?) } else { None },
            bias_v: if cfg.attn_bias { Some(up(&format!("{p}.attn_v.bias"))?) } else { None },
            q_norm: if cfg.qk_norm { Some(up(&format!("{p}.attn_q_norm.weight"))?) } else { None },
            k_norm: if cfg.qk_norm { Some(up(&format!("{p}.attn_k_norm.weight"))?) } else { None },
            mlp_norm: up(&format!("{p}.ffn_norm.weight"))?,
            w_gate: up(&format!("{p}.ffn_gate.weight"))?,
            w_up: up(&format!("{p}.ffn_up.weight"))?,
            w_down: up(&format!("{p}.ffn_down.weight"))?,
        });
    }

    Ok(ModelWeights { embed, layers, final_norm, lm_head })
}

// ════════════════════════════════════════════════════════════════════════
// Phi-3 GGUF path: FUSED attention/MLP tensors split at load.
// ════════════════════════════════════════════════════════════════════════
//
// Phi-3 (arch="phi3") is a standard RMSNorm + RoPE + GQA + SwiGLU transformer —
// the same forward graph as Qwen — but llama.cpp ships two of its projections
// FUSED into single tensors:
//   * `blk.N.attn_qkv.weight`  — Q‖K‖V stacked  [out=(nq+2·nkv)·hd, in=hidden]
//   * `blk.N.ffn_up.weight`    — gate‖up stacked [out=2·intermediate, in=hidden]
// (no separate `attn_q/attn_k/attn_v` or `ffn_gate`). Phi-3 has NO QKV bias and
// NO q/k-norm, so once the fused tensors are split into the builder's q/k/v +
// gate/up slots, the existing Qwen f32 forward path (`GgufModel::step`) runs it
// unchanged. The split is a pure row-range slice of the row-major [out,in] f32
// buffer (fused weights are simple row concatenations), so it is exact — no
// re-quant, backend-agnostic (host-side slice → f32 upload → registry ops).

/// True when the GGUF declares the Phi-3 architecture (fused qkv/ffn_up).
pub fn gguf_is_phi3(g: &ffai_loader::gguf::Gguf) -> bool {
    g.meta_str("general.architecture") == Some("phi3")
        || g.tensor("blk.0.attn_qkv.weight").is_some()
}

/// Dequant a 2-D GGUF weight to a row-major f32 `[out, in]` buffer plus its
/// `(out, in)` dims. GGUF stores dims fastest-first, so a logical `[out, in]`
/// matrix is listed as `dims=[in, out]`; the dequantized buffer is row-major
/// `[out, in]`. Shared by the fused-tensor split below.
fn dequant_2d(g: &ffai_loader::gguf::Gguf, name: &str) -> Result<(Vec<f32>, usize, usize)> {
    let t = g
        .tensor(name)
        .ok_or_else(|| Error::Msg(format!("gguf: tensor '{name}' not found")))?;
    if t.dims.len() != 2 {
        return Err(Error::Msg(format!("gguf: '{name}' expected 2-D, got {:?}", t.dims)));
    }
    let in_dim = t.dims[0] as usize; // fastest (row stride)
    let out_dim = t.dims[1] as usize; // rows
    let data = g.dequant_f32(name)?;
    Ok((data, out_dim, in_dim))
}

/// Upload a contiguous row slice `[r0, r1)` of a row-major `[out, in]` f32 buffer
/// as a `[r1-r0, in]` weight tensor — the per-projection slab carved from a
/// fused GGUF tensor.
fn upload_rows(
    dev: &dyn Device,
    data: &[f32],
    in_dim: usize,
    r0: usize,
    r1: usize,
) -> Result<Tensor> {
    let slab = &data[r0 * in_dim..r1 * in_dim];
    let bytes: &[u8] = bytemuck_cast(slab);
    Ok(Tensor::new(dev.upload(bytes)?, vec![r1 - r0, in_dim], ffai_core::DType::F32))
}

/// Build [`ModelWeights`] from a **Phi-3** GGUF (f32 dequant-to-upload),
/// splitting the fused `attn_qkv` → q/k/v and `ffn_up` → gate/up at load.
/// Parallel to [`load_qwen_gguf`]; the rest of the graph is identical (Phi-3 has
/// no QKV bias, no q/k-norm).
pub fn load_phi3_gguf(
    dev: &dyn Device,
    g: &ffai_loader::gguf::Gguf,
    cfg: &LlamaConfig,
    n_layers: usize,
) -> Result<ModelWeights> {
    // Generic upload (handles 1-D norms and 2-D weights), matching
    // [`load_qwen_gguf`]'s `up`: GGUF dims are fastest-first, so reverse them.
    let up = |name: &str| -> Result<Tensor> {
        let t = g
            .tensor(name)
            .ok_or_else(|| Error::Msg(format!("gguf: tensor '{name}' not found")))?;
        let data = g.dequant_f32(name)?;
        let shape: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();
        let bytes: &[u8] = bytemuck_cast(&data);
        Ok(Tensor::new(dev.upload(bytes)?, shape, ffai_core::DType::F32))
    };

    let embed = up("token_embd.weight")?;
    let final_norm = up("output_norm.weight")?;
    let lm_head = match up("output.weight") {
        Ok(t) => t,
        Err(_) => up("token_embd.weight")?, // tied embeddings
    };

    let q_out = cfg.n_q_heads * cfg.head_dim;
    let kv_out = cfg.n_kv_heads * cfg.head_dim;

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let p = format!("blk.{l}");

        // Fused QKV: rows [0,q_out) = Q, [q_out,q_out+kv_out) = K, then V.
        let (qkv, qkv_out, qkv_in) = dequant_2d(g, &format!("{p}.attn_qkv.weight"))?;
        if qkv_out != q_out + 2 * kv_out {
            return Err(Error::Msg(format!(
                "phi3 '{p}.attn_qkv': out {qkv_out} != q({q_out})+2·kv({kv_out})"
            )));
        }
        let wq = upload_rows(dev, &qkv, qkv_in, 0, q_out)?;
        let wk = upload_rows(dev, &qkv, qkv_in, q_out, q_out + kv_out)?;
        let wv = upload_rows(dev, &qkv, qkv_in, q_out + kv_out, q_out + 2 * kv_out)?;

        // Fused gate/up: rows [0,inter) = gate, [inter,2·inter) = up.
        let (gu, gu_out, gu_in) = dequant_2d(g, &format!("{p}.ffn_up.weight"))?;
        if gu_out != 2 * cfg.intermediate {
            return Err(Error::Msg(format!(
                "phi3 '{p}.ffn_up': out {gu_out} != 2·intermediate({})",
                cfg.intermediate
            )));
        }
        let w_gate = upload_rows(dev, &gu, gu_in, 0, cfg.intermediate)?;
        let w_up = upload_rows(dev, &gu, gu_in, cfg.intermediate, 2 * cfg.intermediate)?;

        layers.push(LayerWeights {
            attn_norm: up(&format!("{p}.attn_norm.weight"))?,
            wq,
            wk,
            wv,
            wo: up(&format!("{p}.attn_output.weight"))?,
            bias_q: None,
            bias_k: None,
            bias_v: None,
            q_norm: None,
            k_norm: None,
            mlp_norm: up(&format!("{p}.ffn_norm.weight"))?,
            w_gate,
            w_up,
            w_down: up(&format!("{p}.ffn_down.weight"))?,
        });
    }

    Ok(ModelWeights { embed, layers, final_norm, lm_head })
}

// ════════════════════════════════════════════════════════════════════════
// Resident-Q8 GGUF path: keep the big matmul weights QUANTIZED (Q8_0) on the
// device and decode through `ffai_ops::gemv_q8` instead of dequant-to-f32.
// ════════════════════════════════════════════════════════════════════════
//
// This roughly halves the weight upload (int8 + f32 scale/32 ≈ 1.03 B/weight vs
// 4 B/weight for f32) AND cuts decode DRAM bandwidth (~4× less weight traffic
// per matvec — the resident-decode win). Norms/RoPE/attention stay f32 (small).
// Backend-agnostic: gemv_q8 is the registry-tested Metal/CUDA/Vulkan kernel.

/// One resident-Q8 weight matrix in the layout `ffai_ops::gemv_q8` consumes:
/// `qs` = int8 codes packed 4-per-u32 (8 u32 per 32-block), `scales` = one f32
/// per 32-block (`[m*k/32]`). Dense matvec ⇒ `rows_per_group = m`.
pub struct Q8Mat {
    pub qs: Tensor,     // [m * k/32 * 8] u32
    pub scales: Tensor, // [m * k/32]     f32
    pub m: usize,       // out-dim (rows)
    pub k: usize,       // in-dim  (cols, multiple of 32)
}

impl Q8Mat {
    /// `out[m] = Wq · x[k]` via the resident-Q8 grouped matvec (dense: one group).
    fn matvec(&self, dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
        ops::gemv_q8(dev, &self.qs, &self.scales, x, self.m, self.k, self.m)
    }

    /// Batched `out[n_rows, m] = x[n_rows, k] · Wqᵀ` via the resident-Q8
    /// cooperative-matrix GEMM (`gemm_q8_mpp`). This is the **prefill** path:
    /// all `n_rows` prompt tokens projected in one dispatch (vs `n_rows`×
    /// `matvec`), and on Vulkan/RDNA4 it lights up the coopmat fragment ops when
    /// `MT_VK_COOPMAT=1`. Returns `[n_rows, m]`.
    ///
    /// Debug: `FFAI_PREFILL_GEMV=1` routes the batched projection through a
    /// per-row `gemv_q8` loop instead (the proven decode kernel), to A/B the
    /// GEMM path against attention/rope when isolating a prefill correctness bug.
    fn gemm(&self, dev: &dyn Device, x: &Tensor, n_rows: usize) -> Result<Tensor> {
        if std::env::var("FFAI_PREFILL_GEMV").map(|v| v == "1").unwrap_or(false) {
            // Per-row gemv_q8: slice each token's [k] activation, matvec, stack.
            let mut rows: Vec<u8> = Vec::with_capacity(n_rows * self.m * 4);
            for r in 0..n_rows {
                let xr = ops::slice(dev, &x.reshaped(vec![n_rows * self.k]), r * self.k, self.k)?;
                let yr = ops::gemv_q8(dev, &self.qs, &self.scales, &xr, self.m, self.k, self.m)?;
                dev.synchronize()?;
                let mut yb = vec![0u8; self.m * 4];
                dev.download(yr.buffer.as_ref(), &mut yb)?;
                rows.extend_from_slice(&yb);
            }
            let buf = dev.upload(&rows)?;
            return Ok(Tensor::new(buf, vec![n_rows, self.m], ffai_core::DType::F32));
        }
        ops::gemm_q8_mpp(dev, &self.qs, &self.scales, x, n_rows, self.m, self.k)
    }
}

/// Per-layer resident-Q8 matmul weights (the 7 big projections). Norms/biases
/// stay f32 in the parallel [`LayerWeights`].
pub struct Q8Layer {
    pub wq: Q8Mat,
    pub wk: Q8Mat,
    pub wv: Q8Mat,
    pub wo: Q8Mat,
    pub w_gate: Q8Mat,
    pub w_up: Q8Mat,
    pub w_down: Q8Mat,
}

/// Resident-Q8 mirror of [`ModelWeights`]'s big matmuls (`embed`/norms stay f32).
pub struct Q8Weights {
    pub lm_head: Q8Mat,
    pub layers: Vec<Q8Layer>,
}

/// Upload a 2-D GGUF weight as a resident-Q8 [`Q8Mat`] — Q8_0 tensors are
/// repacked losslessly, F16/F32 are quantized per-32-block (see
/// [`ffai_loader::gguf::Gguf::q8_repack`]). No f32 weight upload.
fn up_q8(dev: &dyn Device, g: &ffai_loader::gguf::Gguf, name: &str) -> Result<Q8Mat> {
    let (qs, scales, m, k) = g.q8_repack(name)?;
    let qs_bytes: &[u8] = u32_bytes(&qs);
    let sc_bytes: &[u8] = bytemuck_cast(&scales);
    Ok(Q8Mat {
        qs: Tensor::new(dev.upload(qs_bytes)?, vec![qs.len()], ffai_core::DType::U32),
        scales: Tensor::new(dev.upload(sc_bytes)?, vec![scales.len()], ffai_core::DType::F32),
        m,
        k,
    })
}

/// Build the resident-Q8 big-matmul weights from a GGUF. Parallel to
/// [`load_qwen_gguf`], but keeps q/k/v/o + gate/up/down + lm_head QUANTIZED.
/// `output.weight` falls back to the (tied) `token_embd.weight`.
pub fn load_qwen_gguf_q8(
    dev: &dyn Device,
    g: &ffai_loader::gguf::Gguf,
    n_layers: usize,
) -> Result<Q8Weights> {
    let lm_head = match up_q8(dev, g, "output.weight") {
        Ok(t) => t,
        Err(_) => up_q8(dev, g, "token_embd.weight")?, // tied embeddings
    };
    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let p = format!("blk.{l}");
        layers.push(Q8Layer {
            wq: up_q8(dev, g, &format!("{p}.attn_q.weight"))?,
            wk: up_q8(dev, g, &format!("{p}.attn_k.weight"))?,
            wv: up_q8(dev, g, &format!("{p}.attn_v.weight"))?,
            wo: up_q8(dev, g, &format!("{p}.attn_output.weight"))?,
            w_gate: up_q8(dev, g, &format!("{p}.ffn_gate.weight"))?,
            w_up: up_q8(dev, g, &format!("{p}.ffn_up.weight"))?,
            w_down: up_q8(dev, g, &format!("{p}.ffn_down.weight"))?,
        });
    }
    Ok(Q8Weights { lm_head, layers })
}

/// Quantize a row-major `[m, k]` f32 slab into a resident-Q8 [`Q8Mat`] using the
/// identical per-32-block amax/127 scheme as [`ffai_loader::gguf::Gguf::q8_repack`].
/// Used to carve Q8 sub-matrices from Phi-3's fused (already-dequantized) tensors.
fn q8mat_from_f32(dev: &dyn Device, w: &[f32], m: usize, k: usize) -> Result<Q8Mat> {
    if k % 32 != 0 {
        return Err(Error::Msg(format!("q8mat_from_f32: in-dim {k} not a multiple of 32")));
    }
    let bpr = k / 32;
    let mut qs = vec![0u32; m * bpr * 8];
    let mut scales = vec![0f32; m * bpr];
    for r in 0..m {
        for b in 0..bpr {
            let base = r * k + b * 32;
            let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
            let d = amax / 127.0;
            scales[r * bpr + b] = d;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for w_i in 0..8 {
                let mut packed = 0u32;
                for i in 0..4 {
                    let q = (w[base + w_i * 4 + i] * inv).round().clamp(-127.0, 127.0) as i32;
                    packed |= ((q as u8) as u32) << (i * 8);
                }
                qs[r * bpr * 8 + b * 8 + w_i] = packed;
            }
        }
    }
    let qs_bytes: &[u8] = u32_bytes(&qs);
    let sc_bytes: &[u8] = bytemuck_cast(&scales);
    Ok(Q8Mat {
        qs: Tensor::new(dev.upload(qs_bytes)?, vec![qs.len()], ffai_core::DType::U32),
        scales: Tensor::new(dev.upload(sc_bytes)?, vec![scales.len()], ffai_core::DType::F32),
        m,
        k,
    })
}

/// Resident-Q8 mirror of [`load_phi3_gguf`]: keeps q/k/v/o + gate/up/down +
/// lm_head QUANTIZED, splitting the fused `attn_qkv`/`ffn_up` row-ranges first
/// (re-quantized per-32-block from the exact f32 slabs).
pub fn load_phi3_gguf_q8(
    dev: &dyn Device,
    g: &ffai_loader::gguf::Gguf,
    cfg: &LlamaConfig,
    n_layers: usize,
) -> Result<Q8Weights> {
    let lm_head = match up_q8(dev, g, "output.weight") {
        Ok(t) => t,
        Err(_) => up_q8(dev, g, "token_embd.weight")?, // tied embeddings
    };
    let q_out = cfg.n_q_heads * cfg.head_dim;
    let kv_out = cfg.n_kv_heads * cfg.head_dim;
    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let p = format!("blk.{l}");

        let (qkv, qkv_out, qkv_in) = dequant_2d(g, &format!("{p}.attn_qkv.weight"))?;
        if qkv_out != q_out + 2 * kv_out {
            return Err(Error::Msg(format!(
                "phi3 '{p}.attn_qkv': out {qkv_out} != q({q_out})+2·kv({kv_out})"
            )));
        }
        let wq = q8mat_from_f32(dev, &qkv[0..q_out * qkv_in], q_out, qkv_in)?;
        let wk = q8mat_from_f32(dev, &qkv[q_out * qkv_in..(q_out + kv_out) * qkv_in], kv_out, qkv_in)?;
        let wv = q8mat_from_f32(
            dev,
            &qkv[(q_out + kv_out) * qkv_in..(q_out + 2 * kv_out) * qkv_in],
            kv_out,
            qkv_in,
        )?;

        let (gu, gu_out, gu_in) = dequant_2d(g, &format!("{p}.ffn_up.weight"))?;
        if gu_out != 2 * cfg.intermediate {
            return Err(Error::Msg(format!(
                "phi3 '{p}.ffn_up': out {gu_out} != 2·intermediate({})",
                cfg.intermediate
            )));
        }
        let inter = cfg.intermediate;
        let w_gate = q8mat_from_f32(dev, &gu[0..inter * gu_in], inter, gu_in)?;
        let w_up = q8mat_from_f32(dev, &gu[inter * gu_in..2 * inter * gu_in], inter, gu_in)?;

        layers.push(Q8Layer {
            wq,
            wk,
            wv,
            wo: up_q8(dev, g, &format!("{p}.attn_output.weight"))?,
            w_gate,
            w_up,
            w_down: up_q8(dev, g, &format!("{p}.ffn_down.weight"))?,
        });
    }
    Ok(Q8Weights { lm_head, layers })
}

/// Phi-3 mirror of [`load_qwen_gguf_small_f32`]: f32 embed/norms/final-norm (the
/// big matmuls are resident-Q8 in [`load_phi3_gguf_q8`], so their slots are
/// dummies). Phi-3 has no QKV bias / q-k-norm.
pub fn load_phi3_gguf_small_f32(
    dev: &dyn Device,
    g: &ffai_loader::gguf::Gguf,
    _cfg: &LlamaConfig,
    n_layers: usize,
) -> Result<ModelWeights> {
    let up = |name: &str| -> Result<Tensor> {
        let (data, out_dim, in_dim) = dequant_2d(g, name)?;
        let bytes: &[u8] = bytemuck_cast(&data);
        Ok(Tensor::new(dev.upload(bytes)?, vec![out_dim, in_dim], ffai_core::DType::F32))
    };
    let up1 = |name: &str| -> Result<Tensor> {
        // 1-D norm vectors (dequant_2d requires 2-D; norms are 1-D).
        let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("gguf: '{name}' not found")))?;
        let data = g.dequant_f32(name)?;
        let shape: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();
        let bytes: &[u8] = bytemuck_cast(&data);
        Ok(Tensor::new(dev.upload(bytes)?, shape, ffai_core::DType::F32))
    };
    let dummy = |dev: &dyn Device| -> Result<Tensor> {
        Ok(Tensor::new(dev.upload(&0.0f32.to_le_bytes())?, vec![1, 1], ffai_core::DType::F32))
    };

    let embed = up("token_embd.weight")?;
    let final_norm = up1("output_norm.weight")?;
    let lm_head = dummy(dev)?; // resident-Q8 in Q8Weights

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let p = format!("blk.{l}");
        layers.push(LayerWeights {
            attn_norm: up1(&format!("{p}.attn_norm.weight"))?,
            wq: dummy(dev)?,
            wk: dummy(dev)?,
            wv: dummy(dev)?,
            wo: dummy(dev)?,
            bias_q: None,
            bias_k: None,
            bias_v: None,
            q_norm: None,
            k_norm: None,
            mlp_norm: up1(&format!("{p}.ffn_norm.weight"))?,
            w_gate: dummy(dev)?,
            w_up: dummy(dev)?,
            w_down: dummy(dev)?,
        });
    }
    Ok(ModelWeights { embed, layers, final_norm, lm_head })
}

/// Build the f32 *small* weights for the resident-Q8 path: embed (gather needs
/// f32), the RMSNorm weights, QKV biases, and final norm. The big matmul slots
/// in [`ModelWeights`] get a 1-element dummy (the Q8 path never reads them) so
/// no f32 matmul weight is uploaded. Mirrors [`load_qwen_gguf`]'s f32 dequant
/// for these small tensors only.
pub fn load_qwen_gguf_small_f32(
    dev: &dyn Device,
    g: &ffai_loader::gguf::Gguf,
    cfg: &LlamaConfig,
    n_layers: usize,
) -> Result<ModelWeights> {
    let up = |name: &str| -> Result<Tensor> {
        let t = g
            .tensor(name)
            .ok_or_else(|| Error::Msg(format!("gguf: tensor '{name}' not found")))?;
        let data = g.dequant_f32(name)?;
        let shape: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();
        let bytes: &[u8] = bytemuck_cast(&data);
        Ok(Tensor::new(dev.upload(bytes)?, shape, ffai_core::DType::F32))
    };
    // 1-element f32 dummy for the unused big-matmul slots (Q8 path ignores them).
    let dummy = |dev: &dyn Device| -> Result<Tensor> {
        Ok(Tensor::new(dev.upload(&0.0f32.to_le_bytes())?, vec![1, 1], ffai_core::DType::F32))
    };

    let embed = up("token_embd.weight")?;
    let final_norm = up("output_norm.weight")?;
    let lm_head = dummy(dev)?; // resident-Q8 in Q8Weights

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let p = format!("blk.{l}");
        layers.push(LayerWeights {
            attn_norm: up(&format!("{p}.attn_norm.weight"))?,
            wq: dummy(dev)?,
            wk: dummy(dev)?,
            wv: dummy(dev)?,
            wo: dummy(dev)?,
            bias_q: if cfg.attn_bias { Some(up(&format!("{p}.attn_q.bias"))?) } else { None },
            bias_k: if cfg.attn_bias { Some(up(&format!("{p}.attn_k.bias"))?) } else { None },
            bias_v: if cfg.attn_bias { Some(up(&format!("{p}.attn_v.bias"))?) } else { None },
            q_norm: if cfg.qk_norm { Some(up(&format!("{p}.attn_q_norm.weight"))?) } else { None },
            k_norm: if cfg.qk_norm { Some(up(&format!("{p}.attn_k_norm.weight"))?) } else { None },
            mlp_norm: up(&format!("{p}.ffn_norm.weight"))?,
            w_gate: dummy(dev)?,
            w_up: dummy(dev)?,
            w_down: dummy(dev)?,
        });
    }
    Ok(ModelWeights { embed, layers, final_norm, lm_head })
}

/// `&[u32]` → `&[u8]` for upload (little-endian host). Plain reinterpret.
fn u32_bytes(v: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding/invalid bit patterns; read-only byte view for
    // the duration of the upload call.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// `&[f32]` → `&[u8]` for upload (little-endian host; the device consumes raw
/// f32 bytes). Plain reinterpret — no external dep.
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid bit patterns; we expose its bytes
    // read-only for the duration of the upload call.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A GGUF-loaded model with a resident multi-position KV cache, exposing a
/// `step(token, pos) -> logits` suitable for `ffai_runtime::generate`. The KV
/// cache is `[n_kv_heads, cap, head_dim]` per layer; each step writes k/v at
/// `pos` and attends over `[0, pos]`. Everything runs in f32 on `dev`.
pub struct GgufModel {
    pub cfg: LlamaConfig,
    pub weights: ModelWeights,
    pub n_layers: usize,
    pub vocab: usize,
    cap: usize,
    /// Resident-Q8 big-matmul weights. When `Some`, `step` decodes through
    /// `gemv_q8` (weights kept quantized) instead of the f32 `gemv` path; the
    /// f32 big-matmul slots in `weights` are then unused dummy placeholders.
    q8: Option<Q8Weights>,
    // Per-layer (k_cache, v_cache), each [n_kv_heads * cap * head_dim] f32.
    kcache: Vec<Tensor>,
    vcache: Vec<Tensor>,
}

impl GgufModel {
    /// Load from a GGUF path, deriving config from metadata and allocating a
    /// KV cache of `cap` positions. Uses the f32 dequant-to-upload weight path.
    pub fn open(dev: &dyn Device, path: &str, cap: usize) -> Result<Self> {
        let g = ffai_loader::gguf::Gguf::open(path)?;
        let (cfg, n_layers, vocab) = gguf_config(&g)?;
        // Phi-3 ships FUSED attn_qkv / ffn_up tensors; split them at load. Once
        // split, the Qwen f32 forward path serves Phi-3 unchanged.
        let weights = if gguf_is_phi3(&g) {
            load_phi3_gguf(dev, &g, &cfg, n_layers)?
        } else {
            load_qwen_gguf(dev, &g, &cfg, n_layers)?
        };
        Self::with_weights(dev, cfg, weights, n_layers, vocab, cap)
    }

    /// Load from a GGUF path keeping the big matmul weights QUANTIZED (Q8) and
    /// decoding through `gemv_q8`. Only embed/norms/biases are uploaded as f32
    /// (small), so the weight upload is ~halved and decode reads ~4× less weight
    /// DRAM. Same geometry/KV-cache as [`open`]; backend-agnostic.
    pub fn open_q8(dev: &dyn Device, path: &str, cap: usize) -> Result<Self> {
        let g = ffai_loader::gguf::Gguf::open(path)?;
        let (cfg, n_layers, vocab) = gguf_config(&g)?;
        let (weights, q8) = if gguf_is_phi3(&g) {
            (
                load_phi3_gguf_small_f32(dev, &g, &cfg, n_layers)?,
                load_phi3_gguf_q8(dev, &g, &cfg, n_layers)?,
            )
        } else {
            (
                load_qwen_gguf_small_f32(dev, &g, &cfg, n_layers)?,
                load_qwen_gguf_q8(dev, &g, n_layers)?,
            )
        };
        let mut m = Self::with_weights(dev, cfg, weights, n_layers, vocab, cap)?;
        m.q8 = Some(q8);
        Ok(m)
    }

    /// **Recommended GGUF entrypoint.** Loads `path` preferring the resident-Q8
    /// decode path ([`open_q8`] — weights kept Q8, decode via `gemv_q8`, 20-57×
    /// faster than f32 and bit-identical in practice) and transparently FALLS
    /// BACK to the f32 dequant-to-upload path ([`open`]) if the Q8 repack can't
    /// handle the file (e.g. a quant type `q8_repack` doesn't support). Same
    /// geometry/KV-cache as both; backend-agnostic.
    ///
    /// Use this unless you specifically need the f32 reference path (e.g. a
    /// numerical baseline) — then call [`open`] directly.
    ///
    /// [`open`]: GgufModel::open
    /// [`open_q8`]: GgufModel::open_q8
    pub fn load(dev: &dyn Device, path: &str, cap: usize) -> Result<Self> {
        match Self::open_q8(dev, path, cap) {
            Ok(m) => Ok(m),
            Err(e) => {
                eprintln!(
                    "GgufModel::load: resident-Q8 path unavailable ({e}); \
                     falling back to f32 dequant-to-upload"
                );
                Self::open(dev, path, cap)
            }
        }
    }

    /// Construct from already-loaded weights (lets a caller reuse a parsed Gguf
    /// for the tokenizer without re-loading).
    pub fn with_weights(
        dev: &dyn Device,
        cfg: LlamaConfig,
        weights: ModelWeights,
        n_layers: usize,
        vocab: usize,
        cap: usize,
    ) -> Result<Self> {
        let nkv_hd = cfg.n_kv_heads * cfg.head_dim;
        let zero = vec![0.0f32; nkv_hd * cap];
        let zb = bytemuck_cast(&zero);
        let mut kcache = Vec::with_capacity(n_layers);
        let mut vcache = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            kcache.push(Tensor::new(dev.upload(zb)?, vec![nkv_hd * cap], ffai_core::DType::F32));
            vcache.push(Tensor::new(dev.upload(zb)?, vec![nkv_hd * cap], ffai_core::DType::F32));
        }
        Ok(GgufModel { cfg, weights, n_layers, vocab, cap, q8: None, kcache, vcache })
    }

    /// One decode step: embed `token`, run every layer with KV-cache attention
    /// over `[0, pos]`, final-norm + lm_head, and download the `[vocab]` f32
    /// logits to the host (so the pure-Rust sampler can pick the next token).
    pub fn step(&self, dev: &dyn Device, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let hd = cfg.head_dim;
        let cap = self.cap;
        let nkv = cfg.n_kv_heads;
        let nq = cfg.n_q_heads;

        let ids = Tensor::new(
            dev.upload(&token.to_le_bytes())?,
            vec![1],
            ffai_core::DType::U32,
        );
        let mut x = ops::gather(dev, &self.weights.embed, &ids)?.reshaped(vec![cfg.hidden]);

        let pb = Tensor::new(dev.upload(&(pos as u32).to_le_bytes())?, vec![1], ffai_core::DType::U32);

        for l in 0..self.n_layers {
            let w = &self.weights.layers[l];
            let q8l = self.q8.as_ref().map(|q| &q.layers[l]);
            // ── attention ──────────────────────────────────────────────
            let h = ops::rms_norm(dev, &x, &w.attn_norm, cfg.eps)?;
            let (mut q, mut k, mut v) = match q8l {
                Some(q8) => (q8.wq.matvec(dev, &h)?, q8.wk.matvec(dev, &h)?, q8.wv.matvec(dev, &h)?),
                None => (ops::gemv(dev, &w.wq, &h)?, ops::gemv(dev, &w.wk, &h)?, ops::gemv(dev, &w.wv, &h)?),
            };
            if cfg.attn_bias {
                q = ops::add(dev, &q, w.bias_q.as_ref().unwrap())?;
                k = ops::add(dev, &k, w.bias_k.as_ref().unwrap())?;
                v = ops::add(dev, &v, w.bias_v.as_ref().unwrap())?;
            }
            let q = q.reshaped(vec![nq, hd]);
            let k = k.reshaped(vec![nkv, hd]);
            let (q, k) = if cfg.qk_norm {
                (
                    ops::rms_norm(dev, &q, w.q_norm.as_ref().unwrap(), cfg.eps)?,
                    ops::rms_norm(dev, &k, w.k_norm.as_ref().unwrap(), cfg.eps)?,
                )
            } else {
                (q, k)
            };
            let q = ops::rope_llama(dev, &q, pos as u32, cfg.rope_theta, 1.0, 1.0, 1.0, 1e9)?;
            let k = ops::rope_llama(dev, &k, pos as u32, cfg.rope_theta, 1.0, 1.0, 1.0, 1e9)?;
            let v = v.reshaped(vec![nkv, hd]);

            // Write this token's K/V into the cache at `pos`, then attend [0,pos].
            ops::kv_append(dev, &k, &self.kcache[l], &pb, hd, cap, nkv * hd)?;
            ops::kv_append(dev, &v, &self.vcache[l], &pb, hd, cap, nkv * hd)?;
            let len = (pos + 1) as u32;
            let attn = ops::sdpa_decode(
                dev,
                &q,
                &self.kcache[l].reshaped(vec![nkv, cap, hd]),
                &self.vcache[l].reshaped(vec![nkv, cap, hd]),
                hd,
                len,
                cap as u32,
                cfg.heads_per_group(),
                cfg.attn_scale(),
            )?;
            let attn_o = attn.reshaped(vec![nq * hd]);
            let o = match q8l {
                Some(q8) => q8.wo.matvec(dev, &attn_o)?,
                None => ops::gemv(dev, &w.wo, &attn_o)?,
            };
            x = ops::add(dev, &x, &o)?;

            // ── MLP (SwiGLU) ───────────────────────────────────────────
            let h2 = ops::rms_norm(dev, &x, &w.mlp_norm, cfg.eps)?;
            let (gate, up) = match q8l {
                Some(q8) => (q8.w_gate.matvec(dev, &h2)?, q8.w_up.matvec(dev, &h2)?),
                None => (ops::gemv(dev, &w.w_gate, &h2)?, ops::gemv(dev, &w.w_up, &h2)?),
            };
            let act = ops::swiglu(dev, &gate, &up)?;
            let down = match q8l {
                Some(q8) => q8.w_down.matvec(dev, &act)?,
                None => ops::gemv(dev, &w.w_down, &act)?,
            };
            x = ops::add(dev, &x, &down)?;
        }

        let xn = ops::rms_norm(dev, &x, &self.weights.final_norm, cfg.eps)?;
        let logits = match self.q8.as_ref() {
            Some(q8) => q8.lm_head.matvec(dev, &xn)?,
            None => ops::gemv(dev, &self.weights.lm_head, &xn)?,
        };
        dev.synchronize()?;
        let mut bytes = vec![0u8; self.vocab * 4];
        dev.download(logits.buffer.as_ref(), &mut bytes)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// Batched **prefill**: process all `tokens` at positions `[start, start+S)`
    /// in ONE forward, writing their K/V into the cache, and return the next-token
    /// logits (the last row's `[vocab]`). This is the prefill counterpart of the
    /// per-token [`step`]: instead of `S` sequential `step()` calls (each doing
    /// `gemv_q8` matvecs), the seven projections + lm_head run as batched
    /// `[S, k]·Wᵀ` GEMMs through [`ffai_ops::gemm_q8_mpp`] — the SimdGroup CoopTile
    /// kernel that picks up `VK_KHR_cooperative_matrix` on Vulkan/RDNA4 when
    /// `MT_VK_COOPMAT=1`. Only the resident-Q8 path is GEMM-routed; without Q8
    /// weights it falls back to the per-token loop (f32 dense decode).
    ///
    /// Requires `head_dim == 128` (the `sdpa_multi` prefill-attention kernel).
    /// Positions must fit the KV-cache capacity (`start + S <= cap`).
    pub fn prefill(&self, dev: &dyn Device, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let s = tokens.len();
        // Fall back to the proven per-token path when the GEMM prereqs aren't met
        // (no resident-Q8 weights, single token, or non-128 head_dim).
        if self.q8.is_none() || s <= 1 || cfg.head_dim != 128 {
            let mut last = Vec::new();
            for (i, &t) in tokens.iter().enumerate() {
                last = self.step(dev, t, start + i)?;
            }
            return Ok(last);
        }
        let hd = cfg.head_dim;
        let cap = self.cap;
        let nkv = cfg.n_kv_heads;
        let nq = cfg.n_q_heads;
        let q8w = self.q8.as_ref().unwrap();

        // Embed all S tokens → x [S, hidden].
        let ids = Tensor::new(
            dev.upload(u32_bytes(tokens))?,
            vec![s],
            ffai_core::DType::U32,
        );
        let mut x = ops::gather(dev, &self.weights.embed, &ids)?.reshaped(vec![s, cfg.hidden]);

        // Positions [start, start+S) for batched RoPE + KV append.
        let pos_vec: Vec<u32> = (0..s).map(|i| (start + i) as u32).collect();
        let positions = Tensor::new(
            dev.upload(u32_bytes(&pos_vec))?,
            vec![s],
            ffai_core::DType::U32,
        );

        for l in 0..self.n_layers {
            let w = &self.weights.layers[l];
            let q8 = &q8w.layers[l];
            // ── attention ──────────────────────────────────────────────
            let h = ops::rms_norm(dev, &x, &w.attn_norm, cfg.eps)?; // [S, hidden]
            let mut q = q8.wq.gemm(dev, &h, s)?; // [S, nq*hd]  — coopmat GEMM
            let mut k = q8.wk.gemm(dev, &h, s)?; // [S, nkv*hd]
            let mut v = q8.wv.gemm(dev, &h, s)?; // [S, nkv*hd]
            if cfg.attn_bias {
                q = ops::add_bias_rows(dev, &q, w.bias_q.as_ref().unwrap(), s, nq * hd)?;
                k = ops::add_bias_rows(dev, &k, w.bias_k.as_ref().unwrap(), s, nkv * hd)?;
                v = ops::add_bias_rows(dev, &v, w.bias_v.as_ref().unwrap(), s, nkv * hd)?;
            }
            // Optional Qwen3 per-head Q/K RMSNorm (rms_norm normalizes the last
            // dim, so a [S*heads, head_dim] view is correct per (token,head)).
            let (q, k) = if cfg.qk_norm {
                let qv = q.reshaped(vec![s * nq, hd]);
                let kv = k.reshaped(vec![s * nkv, hd]);
                (
                    ops::rms_norm(dev, &qv, w.q_norm.as_ref().unwrap(), cfg.eps)?,
                    ops::rms_norm(dev, &kv, w.k_norm.as_ref().unwrap(), cfg.eps)?,
                )
            } else {
                (q, k)
            };
            // Batched RoPE over [S, heads, hd] (one dispatch each for Q and K).
            let q = ops::rope_llama_many(dev, &q.reshaped(vec![s, nq, hd]), &positions, nq, hd, cfg.rope_theta, 1.0, 1.0, 1.0, 1e9)?;
            let k = ops::rope_llama_many(dev, &k.reshaped(vec![s, nkv, hd]), &positions, nkv, hd, cfg.rope_theta, 1.0, 1.0, 1.0, 1e9)?;
            let v = v.reshaped(vec![s, nkv, hd]);

            // Append all S tokens' K/V into the cache in two batched dispatches,
            // then attend the whole block in ONE `sdpa_multi` flash dispatch.
            // `sdpa_multi` (causal) makes query `r` attend `[0, base_kv+r+1)`;
            // with `base_kv = start` that is the causal prefix `[0, start+r]` —
            // exactly the per-query `sdpa_decode` loop this replaced, but in a
            // single grid (`[n_q_heads*n_query]` workgroups) with no per-row
            // host download/upload round-trip. Validated bit-accurate on the
            // Vulkan backend by `metaltile-std/tests/vulkan_sdpa_multi.rs`
            // (prefill + GQA shapes) and the auto corpus.
            ops::kv_append_many(dev, &k, &positions, &self.kcache[l], nkv, hd, cap)?;
            ops::kv_append_many(dev, &v, &positions, &self.vcache[l], nkv, hd, cap)?;
            let attn = ops::sdpa_multi(
                dev,
                &q.reshaped(vec![s * nq * hd]),
                &self.kcache[l].reshaped(vec![nkv * cap * hd]),
                &self.vcache[l].reshaped(vec![nkv * cap * hd]),
                hd,
                nq as u32,
                start as u32, // base_kv
                s as u32,     // n_query
                cap as u32,   // kv_stride
                cfg.heads_per_group(),
                true, // causal
                cfg.attn_scale(),
            )?; // [s, nq, hd]
            let attn_o = attn.reshaped(vec![s, nq * hd]);
            let o = q8.wo.gemm(dev, &attn_o, s)?; // [S, hidden]  — coopmat GEMM
            x = ops::add(dev, &x.reshaped(vec![s * cfg.hidden]), &o.reshaped(vec![s * cfg.hidden]))?
                .reshaped(vec![s, cfg.hidden]);

            // ── MLP (SwiGLU) ───────────────────────────────────────────
            let h2 = ops::rms_norm(dev, &x, &w.mlp_norm, cfg.eps)?; // [S, hidden]
            let gate = q8.w_gate.gemm(dev, &h2, s)?; // [S, ffn]
            let up = q8.w_up.gemm(dev, &h2, s)?;     // [S, ffn]
            let act = ops::swiglu(dev, &gate, &up)?; // [S, ffn]
            let down = q8.w_down.gemm(dev, &act, s)?; // [S, hidden]
            x = ops::add(dev, &x.reshaped(vec![s * cfg.hidden]), &down.reshaped(vec![s * cfg.hidden]))?
                .reshaped(vec![s, cfg.hidden]);
        }

        // Final norm over all rows, then lm_head on the LAST token only (the
        // next-token logits). Slice the last hidden row on-device so the lm_head
        // stays a single matvec, matching the per-token decode path's final step.
        let xn = ops::rms_norm(dev, &x, &self.weights.final_norm, cfg.eps)?; // [S, hidden]
        let last_xn = ops::slice(dev, &xn.reshaped(vec![s * cfg.hidden]), (s - 1) * cfg.hidden, cfg.hidden)?;
        let logits = q8w.lm_head.matvec(dev, &last_xn)?;
        dev.synchronize()?;
        let mut bytes = vec![0u8; self.vocab * 4];
        dev.download(logits.buffer.as_ref(), &mut bytes)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}
