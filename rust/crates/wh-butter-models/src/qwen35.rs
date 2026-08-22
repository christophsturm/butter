// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Qwen3.5 / Qwen3.6-27B (GGUF arch `qwen35`): a 64-layer DENSE (no MoE)
//! hybrid transformer — 48 Gated-DeltaNet (GDN) linear-attention layers +
//! 16 full (GQA) attention layers, repeating `[GDN, GDN, GDN, full]`
//! (`full_attention_interval = 4`, layer `i` is full-attn iff
//! `(i+1) % 4 == 0`, verified against `Libraries/MLXLLM/Models/Qwen35.swift`
//! in mlx-swift-lm and against the GGUF tensor-name census: 16*11 + 48*14 +
//! 3 = 851 tensors, exact match). Hidden 5120, dense SwiGLU FFN 17408,
//! untied lm_head, vocab 248320.
//!
//! Full-attn: GQA 24q/4kv, head_dim 256, partial-rotary (rope over only the
//! first 64 of 256 dims, `rope_yarn_partial` with YaRN disabled), per-head
//! Q/K RMSNorm (weighted), and a **sigmoid** output gate fused into Q's
//! projection (`attn_q` is `[hidden, 2*n_head*head_dim]`: per head,
//! `[q(256); gate(256)]` — NOT swish, see `sigmoid_mul`'s doc; the task
//! brief's "swish output gate" for full-attn was a misreading of
//! `Qwen3NextRMSNormGated`, which only applies to GDN's OWN output gate).
//!
//! GDN: 16 key heads x 128, 48 value heads x 128, conv kernel 4 (causal,
//! depthwise, no bias), single-token recurrent step
//! `state' = g*state + outer((v - g*state@k)*beta, k)`,
//! `y = state'@q` (see `iron_gated_delta_step` in wh-iron-std, which
//! name-checks Qwen3.6 in its own doc — this port is the first caller).
//! Output gate is `rmsNorm(y, ssm_norm_weight) * silu(z)` — gate-AFTER-norm
//! (`Qwen3NextRMSNormGated` convention). This is NOT what the already-wired
//! `ops::gated_group_rmsnorm[_batched]` kernels compute (they do
//! gate-BEFORE-norm: `rmsNorm(y*silu(z))`, the Mamba2/NemotronH/Zamba2
//! convention) — reusing those directly would silently produce wrong
//! output, so this file composes the norm from `ops::rms_norm` (per-head,
//! reused from the same op the full-attn q/k-norm uses) + `ops::silu` +
//! `ops::mul` instead.
//!
//! Weights load straight from GGUF: dequantize each tensor to f32, then
//! requantize the big GEMM weights (qkv/o/ffn projections) into this
//! engine's Q4 format (`wh_butter_ops::quantize_q4`), same as
//! [`crate::laguna`]. GDN's small/precision-sensitive tensors — conv1d,
//! the alpha/beta decay-gate projections, `A_log`, `dt_bias`, `ssm_norm` —
//! stay F32 resident (mirrors mlx-swift-lm's explicit fp32-upcast for
//! `in_proj_a`/`in_proj_b`, done there to dodge a bf16 NaN issue at that
//! small an output width; F32 here sidesteps the same class of problem and
//! those tensors are tiny — 48 or 128 wide — so the bandwidth cost is
//! negligible). Embeddings/norms stay F32; the LM head is Q8 (`Q8Head`,
//! same layout as [`crate::laguna::LagunaModel`]'s).
//!
//! DECODE-ONLY, single-batch (B=1) for this port: every prompt token is
//! walked through [`decode_step`] one at a time (a correctness-first
//! "prefill via repeated decode", not a batched prefill kernel — fine for
//! oracle-length prompts, not the throughput path). MTP, vision, and
//! CUDA-graph capture are explicitly out of scope.

use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use wh_butter_core::{DType, Device, Error, Result, Tensor};
use wh_butter_loader::{
    SafeTensors,
    gguf::{Gguf, GgufTensor},
};
use wh_butter_ops as ops;

/// Prefill chunks that finished their GPU work. Tests use this to
/// observe leftover chunks after a TCP drop.
pub static PREFILL_CHUNKS_DONE: AtomicUsize = AtomicUsize::new(0);

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32s_to_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
/// The `iron_gemv_q4_coalesced*` family reads per-block scales as f16; see
/// the identical helper + its rationale in `laguna.rs`.
fn f32s_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    fn half_bits(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 112;
        if e <= 0 {
            return sign;
        }
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        let m = (x >> 13) & 0x3ff;
        let round = (x >> 12) & 1;
        let v = ((e as u32) << 10) | m;
        sign | ((v + round) as u16)
    }
    v.iter().flat_map(|&f| half_bits(f).to_le_bytes()).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub max_seq: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub head_dim: usize,               // full-attn head_dim (256)
    pub n_head: usize,                 // 24
    pub n_kv_heads: usize,             // 4
    pub rope_n_rot: u32,               // 64 (partial rotary out of 256)
    pub rope_theta: f32,               // 1e7
    pub full_attention_interval: usize, // 4
    pub eps: f32,
    pub vocab: usize,
    pub ffn_inter: usize,              // 17408
    // GDN (linear attention)
    pub gdn_num_k_heads: usize,        // 16
    pub gdn_num_v_heads: usize,        // 48
    pub gdn_key_head_dim: usize,       // 128
    pub gdn_value_head_dim: usize,     // 128
    pub gdn_conv_kernel: usize,        // 4
    /// Use a resident E4M3 cache for full-attention K/V.
    pub fp8_kv: bool,
}

impl Qwen35Config {
    /// Layer `i` (0-indexed) is full-attention iff `(i+1) % interval == 0`
    /// — verified against `Qwen35DecoderLayer.isLinear` in mlx-swift-lm
    /// (`isLinear = (layerIdx+1) % fullAttentionInterval != 0`) and against
    /// the GGUF tensor census (layers 3,7,11,...,63 carry `attn_q`/`attn_k`/
    /// `attn_v`/`attn_output`; all others carry `ssm_*`).
    pub fn is_full_layer(&self, i: usize) -> bool {
        (i + 1) % self.full_attention_interval == 0
    }
    pub fn gdn_key_dim(&self) -> usize {
        self.gdn_num_k_heads * self.gdn_key_head_dim
    }
    pub fn gdn_value_dim(&self) -> usize {
        self.gdn_num_v_heads * self.gdn_value_head_dim
    }
    pub fn gdn_conv_dim(&self) -> usize {
        self.gdn_key_dim() * 2 + self.gdn_value_dim()
    }
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    pub fn from_gguf(g: &Gguf) -> Self {
        let u = |k: &str, def: usize| g.meta_u32(k).map(|v| v as usize).unwrap_or(def);
        let f = |k: &str, def: f32| g.meta_f32(k).unwrap_or(def);

        let hidden = u("qwen35.embedding_length", 5120);
        let max_seq = u("qwen35.context_length", 262_144);
        // `qwen35.block_count` on an MTP-preserved checkpoint counts the
        // MTP sidecar block TOO (verified via `gguf-dump`: block_count=65,
        // nextn_predict_layers=1, but only 64 trunk layers carry
        // attn_norm/ssm_a-style mixer tensors -- layer 64 is `blk.64.nextn.*`
        // + a full-attn-shaped block, loaded separately by `load_mtp`, NOT
        // part of the `for i in 0..cfg.n_layers` trunk loop). Subtract
        // `nextn_predict_layers` (0 on a non-MTP checkpoint, where this is a
        // no-op) so `cfg.n_layers` stays "trunk layer count" everywhere else
        // in this file, and `load_mtp`'s `blk.{cfg.n_layers}.*` indexing
        // lands on the right block (64, not 65).
        let raw_block_count = u("qwen35.block_count", 64);
        let n_layer_nextn = u("qwen35.nextn_predict_layers", 0);
        let n_layers = raw_block_count - n_layer_nextn;
        let n_head = u("qwen35.attention.head_count", 24);
        let n_kv_heads = u("qwen35.attention.head_count_kv", 4);
        let head_dim = u("qwen35.attention.key_length", 256);
        let rope_n_rot = u("qwen35.rope.dimension_count", 64) as u32;
        let rope_theta = f("qwen35.rope.freq_base", 1.0e7);
        let full_attention_interval = u("qwen35.full_attention_interval", 4);
        let eps = f("qwen35.attention.layer_norm_rms_epsilon", 1e-6);
        // No `qwen35.vocab_size` key in the GGUF (verified via gguf-dump) —
        // derive from `token_embd.weight`'s own dims instead (ne=[hidden,
        // vocab], i.e. dims[1] = vocab; see `dims_out_in`'s doc for the
        // fastest-first convention).
        let vocab = g
            .tensor("token_embd.weight")
            .map(|t| t.dims[1] as usize)
            .unwrap_or(248320);
        let ffn_inter = u("qwen35.feed_forward_length", 17408);
        let gdn_num_k_heads = u("qwen35.ssm.group_count", 16);
        let gdn_value_dim_total = u("qwen35.ssm.inner_size", 6144);
        let gdn_num_v_heads = u("qwen35.ssm.time_step_rank", 48);
        let gdn_key_head_dim = u("qwen35.ssm.state_size", 128);
        let gdn_value_head_dim = gdn_value_dim_total / gdn_num_v_heads.max(1);
        let gdn_conv_kernel = u("qwen35.ssm.conv_kernel", 4);

        Qwen35Config {
            max_seq,
            hidden,
            n_layers,
            head_dim,
            n_head,
            n_kv_heads,
            rope_n_rot,
            rope_theta,
            full_attention_interval,
            eps,
            vocab,
            ffn_inter,
            gdn_num_k_heads,
            gdn_num_v_heads,
            gdn_key_head_dim,
            gdn_value_head_dim,
            gdn_conv_kernel,
            fp8_kv: false,
        }
    }

    /// Read the text-model geometry from a Hugging Face checkpoint directory.
    /// Multimodal wrapper configs keep these fields under `text_config`; a
    /// text-only config may place them at the root.
    pub fn from_hf_dir(dir: &str) -> Result<Self> {
        let path = std::path::Path::new(dir).join("config.json");
        let bytes = std::fs::read(&path)
            .map_err(|e| Error::Msg(format!("read {}: {e}", path.display())))?;
        let root: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Msg(format!("parse {}: {e}", path.display())))?;
        let cfg = root.get("text_config").unwrap_or(&root);
        let u = |key: &str| -> Result<usize> {
            cfg.get(key)
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
                .ok_or_else(|| Error::Msg(format!("config.json: {key} missing")))
        };
        let f = |key: &str| -> Result<f32> {
            cfg.get(key)
                .and_then(|value| value.as_f64())
                .map(|value| value as f32)
                .ok_or_else(|| Error::Msg(format!("config.json: {key} missing")))
        };
        let head_dim = u("head_dim")?;
        let rope = cfg.get("rope_parameters").unwrap_or(cfg);
        let rope_theta = rope
            .get("rope_theta")
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
            .unwrap_or(10_000_000.0);
        let partial_rotary = cfg
            .get("partial_rotary_factor")
            .and_then(|value| value.as_f64())
            .unwrap_or(0.25);
        let gdn_num_v_heads = u("linear_num_value_heads")?;
        let gdn_value_dim_total = gdn_num_v_heads * u("linear_value_head_dim")?;
        let advertised_max_seq = u("max_position_embeddings")?;
        // The current Rust RoPE path is native-context only. A checkpoint may
        // advertise a longer YaRN window, but reporting or allocating that
        // window before the YaRN math is wired would be a false capability.
        let max_seq = rope
            .get("original_max_position_embeddings")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .unwrap_or(advertised_max_seq);
        Ok(Self {
            max_seq,
            hidden: u("hidden_size")?,
            n_layers: u("num_hidden_layers")?,
            head_dim,
            n_head: u("num_attention_heads")?,
            n_kv_heads: u("num_key_value_heads")?,
            rope_n_rot: (head_dim as f64 * partial_rotary).round() as u32,
            rope_theta,
            full_attention_interval: u("full_attention_interval")?,
            eps: f("rms_norm_eps")?,
            vocab: u("vocab_size")?,
            ffn_inter: u("intermediate_size")?,
            gdn_num_k_heads: u("linear_num_key_heads")?,
            gdn_num_v_heads,
            gdn_key_head_dim: u("linear_key_head_dim")?,
            gdn_value_head_dim: gdn_value_dim_total / gdn_num_v_heads.max(1),
            gdn_conv_kernel: u("linear_conv_kernel_dim")?,
            fp8_kv: false,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Weight loading primitives
// ─────────────────────────────────────────────────────────────────────────

/// F-85 round-8: the marlin U4B8 repacked-weight + permuted-scale device
/// buffers for one [`Q4Dense`], built once at load time (see
/// `build_marlin`/`load_q4dense_routed`) for the 5 dense-projection weight
/// classes (ffn_gate/up/down, attn_q, gdn attn_qkv) that round-7 measured a
/// decode-M=1 win on. Kept ALONGSIDE the plain `qs`/`scales` (not instead
/// of) so `QWEN35_MARLIN=0` is a same-process A/B toggle, not a reload.
struct MarlinWeight {
    b_repacked: Tensor, // u32 [k/16, m*2] -- ops::marlin_repack output
    b_scales: Tensor,   // f16 [k/32, m], permuted -- ops::marlin_permute_scales output
    // F-85 round-10 (lever 1): `marlin_mm`'s auto-config search
    // (`thread_k=thread_n=blocks_per_sm=-1`) recomputes the same result on
    // EVERY call via a `cudaFuncGetAttributes` probe loop -- deterministic
    // for a fixed (m,k) shape, so resolved ONCE here (`build_marlin`, via
    // `ops::marlin_pick_config`) and reused for every decode step's call.
    thread_k: i32,
    thread_n: i32,
    blocks_per_sm: i32,
}

struct Q4Dense {
    qs: Tensor,
    scales: Tensor,
    m: usize,
    k: usize,
    marlin: Option<MarlinWeight>,
}

struct Dense16 {
    weight: Tensor,
    m: usize,
    k: usize,
}

struct Nvfp4Dense {
    weight: Tensor,
    scales: Tensor,
    scales_interleaved: Tensor,
    m: usize,
    k: usize,
    block_size: usize,
    input_multiplier: f32,
    weight_multiplier: f32,
}

enum DenseWeight {
    Q4(Q4Dense),
    Dense16(Dense16),
    Nvfp4(Nvfp4Dense),
}

impl DenseWeight {
    fn gemv(&self, dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Q4(weight) => weight.gemv(dev, x),
            Self::Dense16(weight) => {
                let input = match weight.weight.dtype {
                    DType::F32 => x.clone(),
                    DType::F16 => ops::cast_f32_f16(dev, x)?,
                    DType::BF16 => ops::cast_f32_bf16(dev, x)?,
                    dtype => {
                        return Err(Error::Msg(format!(
                            "qwen35: unsupported dense dtype {dtype}"
                        )));
                    }
                };
                let out = ops::gemv(dev, &weight.weight, &input)?;
                match out.dtype {
                    DType::F32 => Ok(out),
                    DType::F16 => ops::cast_f16_f32(dev, &out),
                    DType::BF16 => ops::cast_bf16_f32(dev, &out),
                    dtype => Err(Error::Msg(format!(
                        "qwen35: unsupported dense output {dtype}"
                    ))),
                }
            }
            Self::Nvfp4(weight) => {
                let use_w4a4 = std::env::var("BUTTER_QWEN35_NVFP4_W4A4")
                    .map(|value| value != "0" && value != "false")
                    .unwrap_or(true);
                if !use_w4a4 {
                    return ops::gemv_nvfp4(
                        dev,
                        &weight.weight,
                        &weight.scales,
                        x,
                        weight.m,
                        weight.k,
                        weight.block_size,
                        weight.weight_multiplier,
                    );
                }
                let (x_pack, x_scales, _) = ops::lt_fp4_quant_with_global(
                    dev,
                    x,
                    1,
                    weight.k,
                    weight.input_multiplier,
                )?;
                let alpha = weight.input_multiplier * weight.weight_multiplier;
                let alpha_tensor = Tensor::new(
                    dev.upload(&alpha.to_le_bytes())?,
                    vec![1],
                    DType::F32,
                );
                Ok(ops::gemm_fp4(
                    dev,
                    &x_pack,
                    &x_scales,
                    &weight.weight,
                    &weight.scales_interleaved,
                    1,
                    weight.m,
                    weight.k,
                    true,
                    Some(&alpha_tensor),
                )?
                .reshaped(vec![weight.m]))
            }
        }
    }

    fn gemm(&self, dev: &dyn Device, x: &Tensor, rows: usize) -> Result<Tensor> {
        match self {
            Self::Q4(weight) => weight.gemm(dev, x, rows),
            Self::Dense16(weight) => {
                let input = match weight.weight.dtype {
                    DType::F16 => ops::cast_f32_f16(dev, x)?,
                    DType::BF16 => ops::cast_f32_bf16(dev, x)?,
                    DType::F32 => return ops::matmul(dev, &weight.weight, x),
                    dtype => {
                        return Err(Error::Msg(format!(
                            "qwen35: unsupported dense dtype {dtype}"
                        )));
                    }
                };
                ops::gemm_cublas_f32out(dev, &input, &weight.weight, rows, weight.m, weight.k)
            }
            Self::Nvfp4(weight) => {
                let use_w4a4 = std::env::var("BUTTER_QWEN35_NVFP4_W4A4")
                    .map(|value| value != "0" && value != "false")
                    .unwrap_or(true);
                if use_w4a4 {
                    let (x_pack, x_scales, _) = ops::lt_fp4_quant_with_global(
                        dev,
                        x,
                        rows,
                        weight.k,
                        weight.input_multiplier,
                    )?;
                    let alpha = weight.input_multiplier * weight.weight_multiplier;
                    let alpha_tensor = Tensor::new(
                        dev.upload(&alpha.to_le_bytes())?,
                        vec![1],
                        DType::F32,
                    );
                    return Ok(ops::gemm_fp4(
                        dev,
                        &x_pack,
                        &x_scales,
                        &weight.weight,
                        &weight.scales_interleaved,
                        rows,
                        weight.m,
                        weight.k,
                        true,
                        Some(&alpha_tensor),
                    )?
                    .reshaped(vec![rows, weight.m]));
                }

                // Explicit W4A16 diagnostic fallback.
                let mut joined: Option<Tensor> = None;
                for row in 0..rows {
                    let input = ops::slice(dev, x, row * weight.k, weight.k)?;
                    let output = ops::gemv_nvfp4(
                        dev,
                        &weight.weight,
                        &weight.scales,
                        &input,
                        weight.m,
                        weight.k,
                        weight.block_size,
                        weight.weight_multiplier,
                    )?;
                    joined = Some(match joined {
                        None => output,
                        Some(previous) => {
                            ops::concat_rows_f32(dev, &previous, row, &output, 1, weight.m)?
                        }
                    });
                }
                joined
                    .map(|output| output.reshaped(vec![rows, weight.m]))
                    .ok_or_else(|| Error::Msg("qwen35: NVFP4 GEMM needs at least one row".into()))
            }
        }
    }
}
impl Q4Dense {
    // NOTE (round-2 perf campaign, 2026-08): `ops::gemv_q4_auto` (2-row-per-
    // warp for narrow m_out) looked like a solid win in isolated per-shape
    // microbench (+57-59% on wk/wv/wo/gdn_wz/gdn_wo) but a full oracle-check
    // run wired to it hit a hard crash on the very first decode_step:
    // `Dispatch("dispatch failed: cuModuleLoadData: an illegal memory access
    // was encountered")` — a CUDA context-corrupting fault (surfaced late,
    // at the next module load, per the usual async-error pattern), not
    // reproduced by any single isolated shape in the microbench harness.
    // Root cause NOT isolated within this campaign's time budget. DO NOT
    // wire `gemv_q4_auto` here (or anywhere in the real decode path) until
    // that's root-caused — stays on the safe, fully-plain `gemv_q4` path.
    //
    // F-85 round-8: the marlin U4B8 path (below, `if let Some(mw)`) is the
    // decode-M=1 GEMM for the 5 routed weight classes -- verified correct
    // (round-7, max_rel <=5.3e-4) and verified faster than gemv_q4 at M=1
    // (0.69x-0.79x). Unlike gemv_q4_auto this is a *different kernel family*
    // (tensor-core marlin, not a gemv retile), built + oracle-checked fresh
    // this round -- see the round-8 report for the A/B + oracle numbers.
    fn gemv(&self, dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
        if let Some(mw) = &self.marlin {
            // This decode path's hidden state is F32-resident end to end
            // (`tok_embd` is f32; `gemv_q4` is dtype-polymorphic and just
            // follows whatever `x.dtype` is, per `gemv_q4_dispatch`'s
            // `Tensor::empty(dev, ..., x.dtype)`). marlin's C ABI hardwires
            // f16 activations (`iron_marlin_gemm_u4b8_f16`'s `A`), so cast
            // at the boundary and cast the (small, m-wide) output back to
            // keep the rest of the pipeline's dtype contract unchanged.
            let routing = marlin_m1_routing(dev)?;
            let (workspace, c_tmp) = marlin_shared_scratch(dev)?;
            let need_cast = x.dtype != DType::F16;
            let x16 = if need_cast { ops::cast_f32_f16(dev, x)? } else { x.clone() };
            let out16 = ops::marlin_gemm_u4b8_f16(
                dev, &x16, &mw.b_repacked, &mw.b_scales,
                &routing.0, &routing.1, &routing.2,
                workspace, c_tmp,
                1, self.m, self.k, marlin_group_size(), GB10_SMS,
                mw.thread_k, mw.thread_n, mw.blocks_per_sm,
            )?;
            // marlin's GEMM returns [prob_m,prob_n] = [1,m]; gemv_q4's
            // convention (and every downstream call site) is flat [m].
            let out16 = out16.reshaped(vec![self.m]);
            return if need_cast { ops::cast_f16_f32(dev, &out16) } else { Ok(out16) };
        }
        ops::gemv_q4(dev, &self.qs, &self.scales, x, self.m, self.k, self.m)
    }

    /// Batched (`T`-row) GEMM for the chunked-prefill path (F-85 prefill
    /// wave): `out[t,:] = weight * x[t,:]` for all `t` rows in one
    /// dispatch. `x` is `[t, k]` f32, returns `[t, m]` f32.
    ///
    /// A per-shape microbench (`qwen35_prefill_gemm_microbench.rs`) at
    /// M=512/2048 across every routed projection shape found the batched
    /// Marlin U4B8 GEMM (this weight's EXISTING decode-M=1 `MarlinWeight`
    /// copy, just re-routed/re-configured for `t` instead of `1` -- see
    /// `marlin_routing_for`/`marlin_config_for`) faster than
    /// `gemm_q4_mpp` by 6.6x-35x at every shape tested, not a close call --
    /// `gemm_q4_mpp`'s coop_tile software-GEMM path (see
    /// `GDN_PREFILL_CONTRACT.md` §7.1's writeup on this same CUDA
    /// CoopTile-lowering cost class) does not scale onto GB10's tensor
    /// cores the way the real WMMA-class Marlin kernel does. So: use
    /// Marlin whenever this weight has one (every routed weight class
    /// does, per the `IRON_MARLIN_BUILD=1 QWEN35_MARLIN=1` build config
    /// this port runs under), falling back to `gemm_q4_mpp` only for a
    /// weight that never got a marlin build (dimension not 64/32-aligned,
    /// or `QWEN35_MARLIN=0`).
    fn gemm(&self, dev: &dyn Device, x: &Tensor, t: usize) -> Result<Tensor> {
        if let Some(mw) = &self.marlin {
            let routing = marlin_routing_for(dev, t)?;
            let (workspace, c_tmp) = marlin_shared_scratch(dev)?;
            let (thread_k, thread_n, blocks_per_sm) = marlin_config_for(dev, t, self.m, self.k)?;
            let need_cast = x.dtype != DType::F16;
            let x16 = if need_cast { ops::cast_f32_f16(dev, x)? } else { x.clone() };
            let out16 = ops::marlin_gemm_u4b8_f16(
                dev, &x16, &mw.b_repacked, &mw.b_scales,
                &routing.0, &routing.1, &routing.2,
                workspace, c_tmp,
                t, self.m, self.k, marlin_group_size(), GB10_SMS,
                thread_k, thread_n, blocks_per_sm,
            )?;
            let out16 = out16.reshaped(vec![t, self.m]);
            return if need_cast { ops::cast_f16_f32(dev, &out16) } else { Ok(out16) };
        }
        ops::gemm_q4_mpp(dev, x, &self.qs, &self.scales, t, self.m, self.k)
    }
}

/// GB10 (sm_121a) SM count -- matches the round-7/8 marlin probe harness
/// and the existing NEMOTRON marlin precedent (`wh-butter-modeltests`).
const GB10_SMS: i32 = 48;
/// Matches the round-7 probe harness's `iron_marlin_repack` call exactly
/// (a repack-kernel occupancy hint, not correctness-critical, but kept
/// bit-identical to the verified-working recipe).
const MARLIN_MAX_SHARED_MEM: i32 = 101376;
/// K-dimension group size for the u4b8 requant -- round-7's fixed GS=32,
/// now round-12-overridable via `QWEN35_MARLIN_G` (32/64/128; anything else
/// falls back to 32) so the g=64/g=128 bytes-per-token levers are a same-
/// process env toggle, not a reload -- mirrors `QWEN35_MARLIN`'s pattern.
/// Read ONCE (weight-load and decode-hot-path both call `marlin_group_size()`
/// many times/step; env var lookups are not free).
static MARLIN_GROUP_SIZE_CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
fn marlin_group_size() -> usize {
    *MARLIN_GROUP_SIZE_CELL.get_or_init(|| match std::env::var("QWEN35_MARLIN_G").ok().as_deref() {
        Some("64") => 64,
        Some("128") => 128,
        Some("32") | None => 32,
        Some(other) => {
            eprintln!("qwen35: QWEN35_MARLIN_G={other:?} unrecognized (want 32/64/128), defaulting to 32");
            32
        }
    })
}

/// F-85 round-8: the marlin routing triple (`sorted_token_ids`/`expert_ids`/
/// `num_tokens_past_padded`) for the trivial single-"expert" M=1 dense
/// decode-GEMM case is IDENTICAL for every routed weight (it depends only
/// on M and the block/expert-count constants, not the weight shape) --
/// built once per process and cached, not rebuilt on every decode step.
static MARLIN_M1_ROUTING: std::sync::OnceLock<(Tensor, Tensor, Tensor)> = std::sync::OnceLock::new();

fn marlin_m1_routing(dev: &dyn Device) -> Result<&'static (Tensor, Tensor, Tensor)> {
    if let Some(r) = MARLIN_M1_ROUTING.get() {
        return Ok(r);
    }
    let off_bytes: Vec<u8> = [0i32, 1i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let off = Tensor::new(dev.upload(&off_bytes)?, vec![2], DType::I32);
    let routing = ops::marlin_build_routing(dev, &off, 1, 8, 1)?;
    Ok(MARLIN_M1_ROUTING.get_or_init(|| routing))
}

/// F-85 round-8 (memory-gate fix): `ops::marlin_gemm_u4b8_workspace`'s
/// sizing formula clamps BOTH `workspace` (`min(N/64*64, sms*4)`) and
/// `c_tmp` (`min(N*512, sms*4*8*256)*2`) to an SM-count-bound ceiling that
/// every one of the 4 routed shapes (N in {17408,12288,10240,5120}) hits
/// -- i.e. the two buffers are the SAME size (192 i32 words / 786432 f32
/// words = ~3.1MB) for every routed weight on this GPU. Allocating one pair
/// PER WEIGHT (320 of them: 5 classes x 64 layers) would waste ~1GB on
/// `c_tmp` alone for zero benefit (decode calls are sequential, never
/// concurrent) -- one shared, cached pair is correct and ~320x smaller.
static MARLIN_SCRATCH: std::sync::OnceLock<(Tensor, Tensor)> = std::sync::OnceLock::new();

fn marlin_shared_scratch(dev: &dyn Device) -> Result<(&'static Tensor, &'static Tensor)> {
    if let Some((w, c)) = MARLIN_SCRATCH.get() {
        return Ok((w, c));
    }
    // Sized off the largest routed N (ffn_gate/up, 17408) -- safe for the
    // smaller routed N's too since the clamp formula only ever shrinks
    // toward the same sms-bound ceiling as N grows, never the reverse.
    let pair = ops::marlin_gemm_u4b8_workspace(dev, 17408, GB10_SMS)?;
    let (w, c) = MARLIN_SCRATCH.get_or_init(|| pair);
    Ok((w, c))
}

/// F-85 prefill wave: per-chunk-size cache of the marlin routing triple
/// (`sorted_token_ids`/`expert_ids`/`num_tokens_past_padded`) for the
/// trivial single-"expert" `off={0,t}` case, generalizing
/// [`marlin_m1_routing`] from the fixed `t=1` decode case to whatever
/// chunk size(s) `prefill` actually runs with (typically one entry, the
/// chunk size, plus a second for a shorter final chunk) -- built once per
/// distinct `t` and reused for every weight/layer/chunk of that size,
/// mirroring `marlin_m1_routing`'s "independent of the weight" rationale.
static MARLIN_ROUTING_BY_T: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<usize, (Tensor, Tensor, Tensor)>>> =
    std::sync::OnceLock::new();

fn marlin_routing_for(dev: &dyn Device, t: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let cache = MARLIN_ROUTING_BY_T.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(r) = cache.lock().unwrap().get(&t) {
        return Ok(r.clone());
    }
    let off_bytes: Vec<u8> = [0i32, t as i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let off = Tensor::new(dev.upload(&off_bytes)?, vec![2], DType::I32);
    let routing = ops::marlin_build_routing(dev, &off, 1, 8, t)?;
    cache.lock().unwrap().insert(t, routing.clone());
    Ok(routing)
}

/// F-85 prefill wave: per-`(m, k, t)`-shape cache of `marlin_mm`'s
/// auto-config triple, generalizing [`build_marlin`]'s load-time `t=1`
/// resolution (round-10, lever 1) to prefill's batched `t`. Same
/// rationale: `marlin_mm`'s own auto-config search is a
/// `cudaFuncGetAttributes` probe loop, deterministic for a fixed
/// `(t, m, k)` -- resolve once per distinct shape/chunk-size combination
/// (a handful, not once per layer) and reuse.
static MARLIN_CONFIG_BY_SHAPE_T: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(usize, usize, usize), (i32, i32, i32)>>> =
    std::sync::OnceLock::new();

fn marlin_config_for(dev: &dyn Device, t: usize, m: usize, k: usize) -> Result<(i32, i32, i32)> {
    let cache = MARLIN_CONFIG_BY_SHAPE_T.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key = (t, m, k);
    if let Some(c) = cache.lock().unwrap().get(&key) {
        return Ok(*c);
    }
    let cfg = ops::marlin_pick_config(dev, t, m, k, marlin_group_size(), GB10_SMS)?;
    cache.lock().unwrap().insert(key, cfg);
    Ok(cfg)
}

/// F-85 round-8: builds the marlin U4B8 repacked-weight + permuted-scale
/// buffers for one weight DIRECTLY from its GGUF f32 dequant (`flat`,
/// `[m,k]` row-major) -- not from our own already-lossy `Q4Dense`, so this
/// doesn't stack a second quantization error on top of the first (the
/// plan's original "dequant our Q4 -> requant" framing was written before
/// noticing the loader already holds the pre-Q4 f32 at this call site).
fn build_marlin(dev: &dyn Device, flat: &[f32], m: usize, k: usize) -> Result<MarlinWeight> {
    let gs = marlin_group_size();
    let (bqw, scales_f32) = ops::quantize_u4b8_gptq(flat, m, k, gs);
    let scales_bytes = f32s_to_f16_bytes(&scales_f32);
    let bqw_t = Tensor::new(dev.upload(&u32s_to_bytes(&bqw))?, vec![bqw.len()], DType::U32);
    let scales_raw = Tensor::new(dev.upload(&scales_bytes)?, vec![scales_bytes.len() / 2], DType::F16);
    let b_repacked = ops::marlin_repack(dev, &bqw_t, k, m, GB10_SMS, MARLIN_MAX_SHARED_MEM)?;
    let num_groups = k / gs;
    let b_scales = ops::marlin_permute_scales(dev, &scales_raw, num_groups, m)?;
    // F-85 round-10 (lever 1): resolve the auto-config ONCE per weight
    // shape at load time instead of paying marlin_mm's per-call
    // cudaFuncGetAttributes probe loop on every one of the ~320 decode
    // calls/step that share this (m,k).
    let (thread_k, thread_n, blocks_per_sm) =
        ops::marlin_pick_config(dev, 1, m, k, gs, GB10_SMS)?;
    Ok(MarlinWeight { b_repacked, b_scales, thread_k, thread_n, blocks_per_sm })
}

/// GGUF dims are fastest-first: a 2-D `[out, in]` weight is stored with
/// `dims = [in, out]`, which is exactly the `(m=out, k=in)` `gemv_q4`/
/// `ops::gemv` convention with zero transposition (same convention
/// `laguna.rs::dims_out_in` documents and relies on).
fn dims_out_in(t: &GgufTensor, name: &str) -> Result<(usize, usize)> {
    if t.dims.len() != 2 {
        return Err(Error::Msg(format!("{name}: expected 2-D tensor, got dims {:?}", t.dims)));
    }
    Ok((t.dims[1] as usize, t.dims[0] as usize))
}

fn load_q4dense(dev: &dyn Device, g: &Gguf, name: &str) -> Result<Q4Dense> {
    let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("qwen35: missing tensor '{name}'")))?;
    let (m, k) = dims_out_in(t, name)?;
    let flat = g.dequant_f32(name)?;
    let (qs, scales_f32) = ops::quantize_q4(&flat, m, k);
    let scales_bytes = f32s_to_f16_bytes(&scales_f32);
    Ok(Q4Dense {
        qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs))?, vec![qs.len()], DType::U32),
        scales: Tensor::new(dev.upload(&scales_bytes)?, vec![scales_bytes.len() / 2], DType::F16),
        m,
        k,
        marlin: None,
    })
}

/// F-85 round-8: like [`load_q4dense`], but additionally builds the marlin
/// U4B8 path ([`build_marlin`]) for the 5 dense-projection weight classes
/// round-7 measured a decode-M=1 win on (ffn_gate/up/down, attn_q, gdn
/// attn_qkv -- callers pick this vs. plain `load_q4dense` per weight name,
/// see the trunk-layer loop). `QWEN35_MARLIN=0` opts out (plain Q4Dense
/// only, same as `load_q4dense`) for A/B benchmarking without a reload-free
/// toggle. A marlin build failure is NON-fatal -- logged once, falls back
/// to the plain gemv_q4 path for that weight -- so a partial/older
/// wh-iron-runtime (built without IRON_MARLIN_BUILD=1) degrades cleanly
/// instead of failing the whole model load.
fn load_q4dense_routed(dev: &dyn Device, g: &Gguf, name: &str) -> Result<Q4Dense> {
    let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("qwen35: missing tensor '{name}'")))?;
    let (m, k) = dims_out_in(t, name)?;
    let flat = g.dequant_f32(name)?;
    let (qs, scales_f32) = ops::quantize_q4(&flat, m, k);
    let scales_bytes = f32s_to_f16_bytes(&scales_f32);
    let mut qd = Q4Dense {
        qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs))?, vec![qs.len()], DType::U32),
        scales: Tensor::new(dev.upload(&scales_bytes)?, vec![scales_bytes.len() / 2], DType::F16),
        m,
        k,
        marlin: None,
    };
    if std::env::var("QWEN35_MARLIN").as_deref() != Ok("0") {
        match build_marlin(dev, &flat, m, k) {
            Ok(mw) => qd.marlin = Some(mw),
            Err(e) => eprintln!("qwen35: marlin repack skipped for {name} ({m}x{k}): {e}"),
        }
    }
    Ok(qd)
}

fn load_q4weight_routed(dev: &dyn Device, g: &Gguf, name: &str) -> Result<DenseWeight> {
    load_q4dense_routed(dev, g, name).map(DenseWeight::Q4)
}

/// Plain F32 dense `[m, k]` weight for `ops::gemv` — used for GDN's tiny
/// (`Hv`-wide) alpha/beta decay-gate projections, kept unquantized (see the
/// module doc's numerical-safety rationale).
fn load_f32_dense(dev: &dyn Device, g: &Gguf, name: &str) -> Result<Tensor> {
    let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("qwen35: missing tensor '{name}'")))?;
    let (m, k) = dims_out_in(t, name)?;
    let flat = g.dequant_f32(name)?;
    Ok(Tensor::new(dev.upload(&f32s_to_bytes(&flat))?, vec![m, k], DType::F32))
}

fn load_f32_vec(dev: &dyn Device, g: &Gguf, name: &str) -> Result<Tensor> {
    let flat = g.dequant_f32(name)?;
    let n = flat.len();
    Ok(Tensor::new(dev.upload(&f32s_to_bytes(&flat))?, vec![n], DType::F32))
}

/// `ssm_conv1d.weight`'s GGUF-native flat layout is CHANNEL-major /
/// kernel-minor (`ne = [conv_kernel, conv_dim]`, fastest-first, so
/// `flat[d*conv_kernel + k]`), but `ops::conv1d_causal_silu_many` (mirroring
/// `iron_conv1d_causal_step_silu_cast_many`) wants KERNEL-major /
/// channel-minor (`w[k*conv_dim + d]`, one contiguous `[conv_dim]` row per
/// tap). Transpose at load time.
fn repack_conv1d_weight(flat: &[f32], conv_dim: usize, conv_kernel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; flat.len()];
    for d in 0..conv_dim {
        for k in 0..conv_kernel {
            out[k * conv_dim + d] = flat[d * conv_kernel + k];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────
// Per-layer weights
// ─────────────────────────────────────────────────────────────────────────

/// Full (GQA) attention layer weights. `wq` is `[hidden, 2*n_head*head_dim]`
/// — Q and its sigmoid output-gate fused per head (see the module doc);
/// split at decode time with `ops::strided_head_half`.
struct FullAttnWeights {
    attn_norm: Tensor, // f32 [hidden]
    wq: DenseWeight,   // [n_head*2*head_dim, hidden]
    wk: DenseWeight,   // [n_kv_heads*head_dim, hidden]
    wv: DenseWeight,   // [n_kv_heads*head_dim, hidden]
    wo: DenseWeight,   // [hidden, n_head*head_dim]
    q_norm: Tensor,    // f32 [head_dim] (weighted RMSNorm, full head_dim)
    k_norm: Tensor,    // f32 [head_dim]
}

/// GDN (linear attention) layer weights.
struct GdnWeights {
    attn_norm: Tensor,     // f32 [hidden]
    wqkv: DenseWeight,     // [key_dim*2+value_dim, hidden]  ("attn_qkv")
    wz: DenseWeight,       // [value_dim, hidden]             ("attn_gate" = in_proj_z)
    w_alpha: Tensor,       // f32 [Hv, hidden]                ("ssm_alpha" = in_proj_a)
    w_beta: Tensor,        // f32 [Hv, hidden]                ("ssm_beta"  = in_proj_b)
    dt_bias: Tensor,       // f32 [Hv]
    neg_exp_a_log: Tensor, // f32 [Hv]  == GGUF's ssm_a directly (already -exp(A_log), see load_gguf)
    /// `ln(-neg_exp_a_log)` — the RAW `a_log` the GENERIC
    /// `iron_gated_delta_prep_chunk[_fast]` kernel family expects (it
    /// computes `g = exp(-exp(a_log)*dt)` in-kernel, unlike the qwen35-
    /// specific decode kernel `iron_gdn_decode_fused`, which takes the
    /// checkpoint's already-transformed `neg_exp_a_log` directly and
    /// computes `g = exp(neg_exp_a_log*dt)` — see
    /// `ops::gdn_prep_chunk_fast_27b`'s doc for why re-deriving this once
    /// at load time, instead of feeding `neg_exp_a_log` straight through,
    /// is required for prefill-then-decode continuation to match:
    /// `neg_exp_a_log` is always < 0 (it's `-exp(x)` for real `x`), so
    /// `ln(-neg_exp_a_log)` is always finite. Computed once, host-side, at
    /// load time (tiny, `[Hv]`-wide) — NOT a hot-path cost.
    a_log_raw: Tensor,     // f32 [Hv]
    conv_w: Tensor,        // f32 [conv_kernel*conv_dim], kernel-major (repacked)
    ssm_norm_w: Tensor,    // f32 [Dv]  gated-RMSNorm weight
    wo: DenseWeight,       // [hidden, value_dim]             ("ssm_out")
}

enum Mixer {
    Full(FullAttnWeights),
    Gdn(GdnWeights),
}

struct DenseFfn {
    ffn_norm: Tensor,  // f32 [hidden]
    gate: DenseWeight, // [ffn_inter, hidden]
    up: DenseWeight,   // [ffn_inter, hidden]
    down: DenseWeight, // [hidden, ffn_inter]
}

struct Qwen35Layer {
    mixer: Mixer,
    ffn: DenseFfn,
}

/// LM head resident as Q8 — same layout `ops::gemv_q8` consumes as
/// [`crate::laguna::LagunaModel`]'s `Q8Head`.
struct Q8Head {
    qs: Tensor,
    scales: Tensor,
    m: usize,
    k: usize,
}

/// Perf-campaign lever (decode-perf-campaign, 2026-08): vocab 248320 x
/// hidden 5120 makes the untied lm_head a monster single-token GEMV
/// (~7.7% of total decode-step GPU time at Q8, measured via nsys on this
/// port — `iron_gemv_q8_coalesced` avg ~6.5ms/step vs `iron_gemv_q4_coalesced`
/// dominating at ~85%). Q4 roughly halves the bytes streamed for this one
/// GEMV. Kept SWITCHABLE (env `QWEN35_LM_HEAD_Q4=1`, default remains Q8)
/// rather than an outright replacement: Laguna deliberately keeps its lm_head
/// at Q8 for output-quality reasons (see `laguna.rs`), and this port's oracle
/// gate is sensitive to any argmax perturbation — Q4 must be measured for
/// speedup AND re-gated against the oracle before it can become the default.
enum LmHead {
    Q8(Q8Head),
    Q4(Q4Dense),
    Dense16(Dense16),
}
impl LmHead {
    fn gemv(&self, dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
        match self {
            LmHead::Q8(h) => ops::gemv_q8(dev, &h.qs, &h.scales, x, h.m, h.k, h.m),
            LmHead::Q4(h) => h.gemv(dev, x),
            LmHead::Dense16(h) => DenseWeight::Dense16(Dense16 {
                weight: h.weight.clone(),
                m: h.m,
                k: h.k,
            })
            .gemv(dev, x),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// MTP (multi-token-prediction) sidecar -- self-speculative decode drafter
// ─────────────────────────────────────────────────────────────────────────
//
// F-85-style speculative decode: `qwen35.nextn_predict_layers` (GGUF meta)
// = 1, i.e. exactly ONE extra decoder-layer-shaped block appended after the
// 64 trunk layers (`blk.64.*` + `blk.64.nextn.*`), consuming ~1.6% of the
// model's own weight bytes per draft step (verified against the GGUF tensor
// census in `load_mtp`'s doc). Architecture confirmed against the llama.cpp
// fork's `models/qwen35.cpp` `graph_mtp()` (the only available from-source
// reference for this arch's MTP -- mlx-swift-lm's `Qwen35.swift` explicitly
// STRIPS `mtp.*` weights rather than running them, see its
// `hasMTPWeights`/`weights.filter` comment, so it was not usable as the
// executable reference here): DeepSeek-style NextN -- concat the seed
// token's embedding with a "final normed hidden" from the previous stage,
// project down through `eh_proj`, run one full-attention decoder layer
// (own KV cache, own position axis matching the backbone's), then share
// the trunk's own final-norm-shaped head. See `MtpWeights`'s doc for the
// exact per-tensor contract and `mtp_draft_step`'s doc for the forward.
struct MtpWeights {
    // `eh_proj`'s GGUF shape is `[hidden, 2*hidden]` computing
    // `eh_proj · [e_norm; h_norm]` (concat on the input axis). This engine
    // has no device `concat` op, so the weight is split at LOAD time into
    // two `[hidden,hidden]` halves and the concat-then-project is computed
    // as `eh_proj_e·e_norm + eh_proj_h·h_norm` instead -- algebraically
    // identical (block matrix times block vector), reuses existing
    // `gemv`+`add` rather than adding a new device op.
    eh_proj_e: Q4Dense, // [hidden, hidden] (embedding half)
    eh_proj_h: Q4Dense, // [hidden, hidden] (hidden half)
    enorm: Tensor,      // f32 [hidden]  ("blk.N.nextn.enorm")
    hnorm: Tensor,      // f32 [hidden]  ("blk.N.nextn.hnorm")
    attn_norm: Tensor,  // f32 [hidden]  ("blk.N.attn_norm")
    wq: Q4Dense,        // [n_head*2*head_dim, hidden]  ([q;gate] fused, same as FullAttnWeights)
    wk: Q4Dense,        // [n_kv_heads*head_dim, hidden]
    wv: Q4Dense,        // [n_kv_heads*head_dim, hidden]
    wo: Q4Dense,        // [hidden, n_head*head_dim]
    q_norm: Tensor,     // f32 [head_dim]
    k_norm: Tensor,     // f32 [head_dim]
    ffn_norm: Tensor,   // f32 [hidden]  ("blk.N.post_attention_norm")
    ffn_gate: Q4Dense,
    ffn_up: Q4Dense,
    ffn_down: Q4Dense,
    // Shared head: the GGUF carries `blk.N.nextn.shared_head_norm` (a
    // DEDICATED final-norm weight for the MTP head, distinct from the
    // backbone's own `output_norm`) but NO `nextn.shared_head_head` and NO
    // `nextn.embed_tokens` -- confirmed absent via `gguf-dump` (only
    // eh_proj/enorm/hnorm/shared_head_norm present under `blk.64.nextn.*`),
    // consistent with the checkpoint's "shared embeddings" convention: the
    // draft head reuses `model.tok_embd` (already true for any lookup) and
    // `model.lm_head` (the trunk's own LM head) directly, just with its OWN
    // pre-head norm.
    shared_head_norm: Tensor, // f32 [hidden]
}

/// Splits `eh_proj`'s `[hidden, 2*hidden]` GGUF weight (GGUF dims
/// `[2*hidden, hidden]`, i.e. `dims_out_in` gives `m=hidden` (out),
/// `k=2*hidden` (in)) into two independently-quantized `[hidden,hidden]`
/// `Q4Dense` halves -- see `MtpWeights::eh_proj_e`/`eh_proj_h`'s doc for why.
/// Row-major flat layout from `dequant_f32`: `flat[o*k + i]`, `i < hidden`
/// is the embedding-half column range (matches `ggml_concat(e_norm, h_norm,
/// dim=0)`'s e_norm-first order in the reference), `i >= hidden` is the
/// hidden-half range.
fn load_q4dense_eh_proj_split(dev: &dyn Device, g: &Gguf, name: &str, hidden: usize) -> Result<(Q4Dense, Q4Dense)> {
    let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("qwen35: missing tensor '{name}'")))?;
    let (m, k) = dims_out_in(t, name)?;
    if m != hidden || k != 2 * hidden {
        return Err(Error::Msg(format!(
            "{name}: expected eh_proj shape out={hidden} in={} (2*hidden), got out={m} in={k}",
            2 * hidden
        )));
    }
    let flat = g.dequant_f32(name)?;
    let mut we = vec![0f32; m * hidden];
    let mut wh = vec![0f32; m * hidden];
    for o in 0..m {
        we[o * hidden..(o + 1) * hidden].copy_from_slice(&flat[o * k..o * k + hidden]);
        wh[o * hidden..(o + 1) * hidden].copy_from_slice(&flat[o * k + hidden..o * k + 2 * hidden]);
    }
    let mk = |flat: Vec<f32>| -> Result<Q4Dense> {
        let (qs, scales_f32) = ops::quantize_q4(&flat, hidden, hidden);
        let scales_bytes = f32s_to_f16_bytes(&scales_f32);
        Ok(Q4Dense {
            qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs))?, vec![qs.len()], DType::U32),
            scales: Tensor::new(dev.upload(&scales_bytes)?, vec![scales_bytes.len() / 2], DType::F16),
            m: hidden,
            k: hidden,
            marlin: None,
        })
    };
    Ok((mk(we)?, mk(wh)?))
}

/// Loads the MTP sidecar if present. `qwen35.block_count` (`cfg.n_layers`,
/// 64) is the trunk-layer count; `qwen35.nextn_predict_layers` (GGUF meta,
/// verified = 1 on the MTP-preserved checkpoint) says the MTP block is
/// stored as ONE extra "layer" immediately after the trunk, i.e.
/// `blk.<n_layers>.*` (verified via `gguf-dump`: `blk.64.attn_norm.weight`,
/// `blk.64.attn_q.weight` [5120,12288] Q4_K, `blk.64.attn_k.weight`
/// [5120,1024] Q4_K, `blk.64.attn_v.weight` [5120,1024] Q6_K,
/// `blk.64.attn_output.weight` [6144,5120] Q4_K, `blk.64.attn_q_norm`/
/// `attn_k_norm.weight` [256], `blk.64.post_attention_norm.weight` [5120],
/// `blk.64.ffn_{gate,up}.weight` [5120,17408] Q4_K, `blk.64.ffn_down.weight`
/// [17408,5120] Q6_K, `blk.64.nextn.eh_proj.weight` [5120,10240] Q8_0,
/// `blk.64.nextn.{enorm,hnorm,shared_head_norm}.weight` [5120] F32 --
/// `nextn.embed_tokens`/`nextn.shared_head_head` confirmed ABSENT). Returns
/// `Ok(None)` (not an error) for a non-MTP checkpoint so the rest of the
/// loader/decode path is unaffected.
fn load_mtp(dev: &dyn Device, g: &Gguf, cfg: &Qwen35Config) -> Result<Option<MtpWeights>> {
    let p = format!("blk.{}", cfg.n_layers);
    if g.tensor(&format!("{p}.nextn.eh_proj.weight")).is_none() {
        return Ok(None);
    }
    let (eh_proj_e, eh_proj_h) =
        load_q4dense_eh_proj_split(dev, g, &format!("{p}.nextn.eh_proj.weight"), cfg.hidden)?;
    Ok(Some(MtpWeights {
        eh_proj_e,
        eh_proj_h,
        enorm: load_f32_vec(dev, g, &format!("{p}.nextn.enorm.weight"))?,
        hnorm: load_f32_vec(dev, g, &format!("{p}.nextn.hnorm.weight"))?,
        attn_norm: load_f32_vec(dev, g, &format!("{p}.attn_norm.weight"))?,
        wq: load_q4dense(dev, g, &format!("{p}.attn_q.weight"))?,
        wk: load_q4dense(dev, g, &format!("{p}.attn_k.weight"))?,
        wv: load_q4dense(dev, g, &format!("{p}.attn_v.weight"))?,
        wo: load_q4dense(dev, g, &format!("{p}.attn_output.weight"))?,
        q_norm: load_f32_vec(dev, g, &format!("{p}.attn_q_norm.weight"))?,
        k_norm: load_f32_vec(dev, g, &format!("{p}.attn_k_norm.weight"))?,
        ffn_norm: load_f32_vec(dev, g, &format!("{p}.post_attention_norm.weight"))?,
        ffn_gate: load_q4dense(dev, g, &format!("{p}.ffn_gate.weight"))?,
        ffn_up: load_q4dense(dev, g, &format!("{p}.ffn_up.weight"))?,
        ffn_down: load_q4dense(dev, g, &format!("{p}.ffn_down.weight"))?,
        shared_head_norm: load_f32_vec(dev, g, &format!("{p}.nextn.shared_head_norm.weight"))?,
    }))
}

pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    tok_embd: Tensor, // f32 [vocab, hidden]
    layers: Vec<Qwen35Layer>,
    out_norm: Tensor, // f32 [hidden]
    lm_head: LmHead,
    mtp: Option<MtpWeights>,
    /// Constant (not loaded) per-head Q/K rescale buffers for GDN, shared
    /// across all GDN layers: Qwen3.5's GDN normalizes q/k with a WEIGHTLESS
    /// RMSNorm scaled by `invScale^2` (q) / `invScale` (k), `invScale =
    /// 1/sqrt(key_head_dim)` (see `Qwen35GatedDeltaNet.callAsFunction` in
    /// mlx-swift-lm). `ops::rms_norm` takes a per-element weight, so filling
    /// it with a constant reproduces the weightless-scaled form exactly.
    gdn_q_scale: Tensor, // f32 [gdn_key_head_dim], all invScale^2
    gdn_k_scale: Tensor, // f32 [gdn_key_head_dim], all invScale
    /// F-85 prefill wave: `[Hk*Dk]`-tiled siblings of `gdn_q_scale`/
    /// `gdn_k_scale`, for `ops::gdn_qknorm_prepass_chunk`'s per-head-shaped
    /// `q_norm_weight`/`k_norm_weight` params — see `gdn_q_scale_tiled`'s
    /// build-site doc (`load_gguf`) for why the untiled `[Dk]` buffer is
    /// wrong there (an out-of-bounds read, not just a shape mismatch).
    gdn_q_scale_tiled: Tensor, // f32 [Hk*Dk], invScale^2 repeated per head
    gdn_k_scale_tiled: Tensor, // f32 [Hk*Dk], invScale repeated per head
    /// Zero bias for `ops::conv1d_causal_silu_many` — Qwen3.5's GDN conv1d
    /// has `bias: false`.
    gdn_conv_zero_bias: Tensor, // f32 [conv_dim]
    /// GQA head-expansion index for GDN's q/k: `iron_gated_delta_step`'s
    /// OWN internal GQA decomposition is `hk_idx = hv_idx / (Hv/Hk)`
    /// (contiguous blocks), but the actual reference kernel this checkpoint
    /// was validated against (llama.cpp's `gated_delta_net.cu`,
    /// `iq1 = fastmodulo(h_idx, neqk1)` i.e. `hv_idx % Hk`) uses the
    /// CYCLIC/tiled mapping instead — verified by oracle checksum diff
    /// (attn_output-0 aggregate matched to 0.01% only after switching to
    /// cyclic; the block mapping was off by ~20%). Rather than patch the
    /// pinned iron kernel, pre-expand q/k from `[Hk,Dk]` to `[Hv,Dk]` via
    /// `ops::gather` with this constant index buffer (`idx[hv] = hv % Hk`),
    /// then call `gated_delta_step` with `hk=Hv` so its internal grouping
    /// becomes a no-op identity map.
    // F-85 round-11: no longer read on the decode path — the CYCLIC GQA
    // mapping it encoded is now baked directly into iron_gdn_decode_fused
    // (see ops::gdn_decode_fused doc). Kept (not deleted) since a future
    // chunked-prefill GDN kernel may want the same precomputed index.
    #[allow(dead_code)]
    gdn_gqa_cyclic_idx: Tensor, // u32 [gdn_num_v_heads], idx[hv] = hv % gdn_num_k_heads
    /// GGUF conversion uses the validated cyclic GDN head layout. Raw HF
    /// checkpoints retain the upstream contiguous repeat-interleave layout.
    gdn_cyclic_gqa: bool,
}

impl Qwen35Model {
    /// Whether this checkpoint carries the MTP sidecar (see `load_mtp`'s
    /// doc) — callers must check this before calling `speculative_round`.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Loading
// ─────────────────────────────────────────────────────────────────────────

pub fn load_gguf(dev: &dyn Device, g: &Gguf) -> Result<Qwen35Model> {
    let load_t0 = std::time::Instant::now();
    let cfg = Qwen35Config::from_gguf(g);

    let tok_embd = {
        let flat = g.dequant_f32("token_embd.weight")?;
        Tensor::new(dev.upload(&f32s_to_bytes(&flat))?, vec![cfg.vocab, cfg.hidden], DType::F32)
    };
    let out_norm = load_f32_vec(dev, g, "output_norm.weight")?;
    let head_name = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    // QWEN35_LM_HEAD_Q4=1: perf-campaign A/B lever, see `LmHead`'s doc.
    // Default (unset/0) keeps the Laguna-established Q8 lm_head.
    let lm_head_q4 = std::env::var("QWEN35_LM_HEAD_Q4").map(|v| v == "1").unwrap_or(false);
    let lm_head = if lm_head_q4 {
        LmHead::Q4(load_q4dense(dev, g, head_name)?)
    } else {
        let (qs, scales, m, k) = g.q8_repack(head_name)?;
        LmHead::Q8(Q8Head {
            qs: Tensor::new(dev.upload(&u32s_to_bytes(&qs))?, vec![qs.len()], DType::U32),
            scales: Tensor::new(dev.upload(&f32s_to_bytes(&scales))?, vec![scales.len()], DType::F32),
            m,
            k,
        })
    };

    let inv_scale = 1.0 / (cfg.gdn_key_head_dim as f32).sqrt();
    let gdn_q_scale = {
        let v = vec![inv_scale * inv_scale; cfg.gdn_key_head_dim];
        Tensor::new(dev.upload(&f32s_to_bytes(&v))?, vec![cfg.gdn_key_head_dim], DType::F32)
    };
    let gdn_k_scale = {
        let v = vec![inv_scale; cfg.gdn_key_head_dim];
        Tensor::new(dev.upload(&f32s_to_bytes(&v))?, vec![cfg.gdn_key_head_dim], DType::F32)
    };
    // F-85 prefill wave: `iron_gated_delta_qknorm_prepass` (batched chunked
    // prefill's q/k RMSNorm pre-pass) indexes its `q_norm_weight`/
    // `k_norm_weight` params as `[Hk, Dk]` (a genuine PER-HEAD learned
    // weight, in the generic form this kernel family was designed for) --
    // unlike `gdn_q_scale`/`gdn_k_scale` above, which are `[Dk]`-only
    // CONSTANT buffers (Qwen3.5's GDN q/k RMSNorm is weightless-scaled,
    // broadcasting the SAME `invScale`/`invScale^2` pattern across every
    // head, see those fields' doc). Feeding the `[Dk]` buffer directly
    // into a kernel that reads `w[hk_idx*Dk + s_idx]` for `hk_idx` up to
    // `Hk-1=15` is a 16x out-of-bounds read for every head but hk_idx=0 --
    // caught by this port's prefill-vs-decode-loop identity test (silently
    // wrong/erratic GDN layer-0 output, no crash since the OOB read still
    // lands in-bounds of an adjacent device allocation). Fix: tile the
    // constant `[Dk]` pattern `Hk` times into a real `[Hk*Dk]` buffer once
    // at load time -- mathematically a no-op (every head gets the
    // identical scale, exactly reproducing the broadcast this checkpoint's
    // weightless RMSNorm wants), just in the shape this kernel's generic
    // per-head-weight contract requires.
    let gdn_q_scale_tiled = {
        let mut v = vec![0.0f32; cfg.gdn_num_k_heads * cfg.gdn_key_head_dim];
        for h in 0..cfg.gdn_num_k_heads {
            v[h * cfg.gdn_key_head_dim..(h + 1) * cfg.gdn_key_head_dim].fill(inv_scale * inv_scale);
        }
        Tensor::new(dev.upload(&f32s_to_bytes(&v))?, vec![v.len()], DType::F32)
    };
    let gdn_k_scale_tiled = {
        let mut v = vec![0.0f32; cfg.gdn_num_k_heads * cfg.gdn_key_head_dim];
        for h in 0..cfg.gdn_num_k_heads {
            v[h * cfg.gdn_key_head_dim..(h + 1) * cfg.gdn_key_head_dim].fill(inv_scale);
        }
        Tensor::new(dev.upload(&f32s_to_bytes(&v))?, vec![v.len()], DType::F32)
    };
    let gdn_conv_zero_bias = {
        let v = vec![0.0f32; cfg.gdn_conv_dim()];
        Tensor::new(dev.upload(&f32s_to_bytes(&v))?, vec![cfg.gdn_conv_dim()], DType::F32)
    };
    let gdn_gqa_cyclic_idx = {
        let idx: Vec<u32> = (0..cfg.gdn_num_v_heads as u32).map(|hv| hv % cfg.gdn_num_k_heads as u32).collect();
        Tensor::new(dev.upload(&u32s_to_bytes(&idx))?, vec![cfg.gdn_num_v_heads], DType::U32)
    };

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("blk.{i}");
        let mixer = if cfg.is_full_layer(i) {
            Mixer::Full(FullAttnWeights {
                attn_norm: load_f32_vec(dev, g, &format!("{p}.attn_norm.weight"))?,
                // F-85 round-10 (lever 2): wk/wv/wo isolated-microbench
                // marlin-vs-gemv_q4 at these exact shapes (m=1024,k=5120
                // and m=5120,k=6144) showed marlin 2.7x-4.4x faster
                // (10.3-16.4us vs 16.4-71.8us, see
                // qwen35_marlin_smallproj_microbench.rs) -- routed like the
                // round-7/8 big shapes instead of plain gemv_q4.
                wq: load_q4weight_routed(dev, g, &format!("{p}.attn_q.weight"))?,
                wk: load_q4weight_routed(dev, g, &format!("{p}.attn_k.weight"))?,
                wv: load_q4weight_routed(dev, g, &format!("{p}.attn_v.weight"))?,
                wo: load_q4weight_routed(dev, g, &format!("{p}.attn_output.weight"))?,
                q_norm: load_f32_vec(dev, g, &format!("{p}.attn_q_norm.weight"))?,
                k_norm: load_f32_vec(dev, g, &format!("{p}.attn_k_norm.weight"))?,
            })
        } else {
            // NOTE: despite the GGUF tensor name mirroring HF's raw "A_log"
            // parameter, llama.cpp's GGUF conversion for this arch BAKES IN
            // the `-exp(A_log)` transform at conversion time — verified
            // against the reference ggml graph (`gate-0 = MUL(a_softplus-0,
            // blk.0.ssm_a)`, a plain elementwise multiply with NO exp/negate
            // node applied to `ssm_a`) and against the tensor's own raw
            // values (small negatives, O(1e-2), consistent with an
            // already-final decay coefficient — re-applying `-exp()` to
            // those pushes everything toward -1, which was the bug: it
            // over-decayed the GDN state on every step, killing context
            // retention). Use the tensor directly, no further transform.
            let neg_exp_a_log = g.dequant_f32(&format!("{p}.ssm_a"))?;
            // See `GdnWeights::a_log_raw`'s doc: the GENERIC prep-chunk
            // kernel family (batched prefill) re-applies `-exp()` in-kernel,
            // so it needs the RAW a_log this inverts back out of the
            // checkpoint's already-transformed `ssm_a` tensor.
            let a_log_raw_host: Vec<f32> = neg_exp_a_log.iter().map(|&v| (-v).ln()).collect();
            if i == 0 && std::env::var("QWEN35_DEBUG").is_ok() {
                let n_nan = a_log_raw_host.iter().filter(|v| v.is_nan()).count();
                let n_pos_input = neg_exp_a_log.iter().filter(|&&v| v >= 0.0).count();
                eprintln!(
                    "QWEN35_DBG layer0 neg_exp_a_log min={:?} max={:?} a_log_raw min={:?} max={:?} n_nan={n_nan} n_pos_input={n_pos_input}",
                    neg_exp_a_log.iter().cloned().fold(f32::INFINITY, f32::min),
                    neg_exp_a_log
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max),
                    a_log_raw_host
                        .iter()
                        .cloned()
                        .filter(|v| !v.is_nan())
                        .fold(f32::INFINITY, f32::min),
                    a_log_raw_host
                        .iter()
                        .cloned()
                        .filter(|v| !v.is_nan())
                        .fold(f32::NEG_INFINITY, f32::max),
                );
            }
            let conv_flat = g.dequant_f32(&format!("{p}.ssm_conv1d.weight"))?;
            let conv_repacked = repack_conv1d_weight(&conv_flat, cfg.gdn_conv_dim(), cfg.gdn_conv_kernel);
            Mixer::Gdn(GdnWeights {
                attn_norm: load_f32_vec(dev, g, &format!("{p}.attn_norm.weight"))?,
                wqkv: load_q4weight_routed(dev, g, &format!("{p}.attn_qkv.weight"))?,
                // F-85 round-10 (lever 2): see the wk/wv/wo comment above --
                // gdn_wz is the same (m=6144,k=5120) shape microbenched
                // there (marlin 16.4us vs gemv 71.7us).
                wz: load_q4weight_routed(dev, g, &format!("{p}.attn_gate.weight"))?,
                w_alpha: load_f32_dense(dev, g, &format!("{p}.ssm_alpha.weight"))?,
                w_beta: load_f32_dense(dev, g, &format!("{p}.ssm_beta.weight"))?,
                dt_bias: load_f32_vec(dev, g, &format!("{p}.ssm_dt.bias"))?,
                neg_exp_a_log: Tensor::new(
                    dev.upload(&f32s_to_bytes(&neg_exp_a_log))?,
                    vec![neg_exp_a_log.len()],
                    DType::F32,
                ),
                a_log_raw: Tensor::new(
                    dev.upload(&f32s_to_bytes(&a_log_raw_host))?,
                    vec![a_log_raw_host.len()],
                    DType::F32,
                ),
                conv_w: Tensor::new(
                    dev.upload(&f32s_to_bytes(&conv_repacked))?,
                    vec![conv_repacked.len()],
                    DType::F32,
                ),
                ssm_norm_w: load_f32_vec(dev, g, &format!("{p}.ssm_norm.weight"))?,
                // F-85 round-10 (lever 2): gdn_wo is the wo(5120,6144)
                // shape microbenched above (marlin 16.4us vs gemv 71.8us).
                wo: load_q4weight_routed(dev, g, &format!("{p}.ssm_out.weight"))?,
            })
        };
        let ffn = DenseFfn {
            ffn_norm: load_f32_vec(dev, g, &format!("{p}.post_attention_norm.weight"))?,
            gate: load_q4weight_routed(dev, g, &format!("{p}.ffn_gate.weight"))?,
            up: load_q4weight_routed(dev, g, &format!("{p}.ffn_up.weight"))?,
            down: load_q4weight_routed(dev, g, &format!("{p}.ffn_down.weight"))?,
        };
        layers.push(Qwen35Layer { mixer, ffn });
    }

    let mtp = load_mtp(dev, g, &cfg)?;
    eprintln!(
        "Qwen35: load done in {:.2}s ({} layers, mtp={})",
        load_t0.elapsed().as_secs_f64(),
        cfg.n_layers,
        mtp.is_some()
    );

    Ok(Qwen35Model {
        cfg,
        tok_embd,
        layers,
        out_norm,
        lm_head,
        mtp,
        gdn_q_scale,
        gdn_k_scale,
        gdn_q_scale_tiled,
        gdn_k_scale_tiled,
        gdn_conv_zero_bias,
        gdn_gqa_cyclic_idx,
        gdn_cyclic_gqa: true,
    })
}

fn load_hf_vec_f32(dev: &dyn Device, st: &SafeTensors, name: &str) -> Result<Tensor> {
    let (flat, shape) = st.tensor_f32(name)?;
    if shape.len() != 1 {
        return Err(Error::Msg(format!(
            "qwen35: {name} must be 1-D, got {shape:?}"
        )));
    }
    Ok(Tensor::new(
        dev.upload(&f32s_to_bytes(&flat))?,
        shape,
        DType::F32,
    ))
}

fn load_hf_gemma_norm(dev: &dyn Device, st: &SafeTensors, name: &str) -> Result<Tensor> {
    let (mut flat, shape) = st.tensor_f32(name)?;
    if shape.len() != 1 {
        return Err(Error::Msg(format!(
            "qwen35: {name} must be 1-D, got {shape:?}"
        )));
    }
    for value in &mut flat {
        *value += 1.0;
    }
    Ok(Tensor::new(
        dev.upload(&f32s_to_bytes(&flat))?,
        shape,
        DType::F32,
    ))
}

fn load_hf_dense16(dev: &dyn Device, st: &SafeTensors, name: &str) -> Result<Dense16> {
    let (bytes, dtype, shape) = st.tensor(name)?;
    if shape.len() != 2 || !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
        return Err(Error::Msg(format!(
            "qwen35: {name} must be a 2-D F16/BF16/F32 tensor, got {dtype} {shape:?}"
        )));
    }
    Ok(Dense16 {
        weight: Tensor::new(dev.upload(bytes)?, shape.to_vec(), dtype),
        m: shape[0],
        k: shape[1],
    })
}

fn load_hf_dense_f32(dev: &dyn Device, st: &SafeTensors, name: &str) -> Result<Tensor> {
    let (flat, shape) = st.tensor_f32(name)?;
    if shape.len() != 2 {
        return Err(Error::Msg(format!(
            "qwen35: {name} must be 2-D, got {shape:?}"
        )));
    }
    Ok(Tensor::new(
        dev.upload(&f32s_to_bytes(&flat))?,
        shape,
        DType::F32,
    ))
}

fn load_hf_nvfp4(dev: &dyn Device, st: &SafeTensors, base: &str) -> Result<Nvfp4Dense> {
    let matrix = st.load_nvfp4_tensor(base)?;
    let scale_rows = matrix.rows;
    let scale_cols = matrix.cols / matrix.block_size;
    let mut scales_interleaved = vec![0u8; matrix.scales.len()];
    let scale_col_blocks = scale_cols / 4;
    for row in 0..scale_rows {
        for col in 0..scale_cols {
            let dst = (row / 128) * (512 * scale_col_blocks)
                + (col / 4) * 512
                + (row % 32) * 16
                + ((row / 32) % 4) * 4
                + col % 4;
            scales_interleaved[dst] = matrix.scales[row * scale_cols + col];
        }
    }
    Ok(Nvfp4Dense {
        weight: Tensor::new(
            dev.upload(matrix.packed)?,
            vec![matrix.packed.len()],
            DType::U8,
        ),
        scales: Tensor::new(
            dev.upload(matrix.scales)?,
            vec![matrix.scales.len()],
            DType::U8,
        ),
        scales_interleaved: Tensor::new(
            dev.upload(&scales_interleaved)?,
            vec![scales_interleaved.len()],
            DType::U8,
        ),
        m: matrix.rows,
        k: matrix.cols,
        block_size: matrix.block_size,
        input_multiplier: matrix.input_multiplier,
        weight_multiplier: matrix.weight_multiplier,
    })
}

fn is_hf_nvfp4(st: &SafeTensors, base: &str) -> bool {
    st.info(&format!("{base}.weight_packed")).is_some()
        || (st
            .info(&format!("{base}.weight"))
            .map(|info| info.dtype == DType::U8)
            .unwrap_or(false)
            && st.info(&format!("{base}.weight_scale")).is_some())
}

fn load_hf_unquantized_weight(
    dev: &dyn Device,
    st: &SafeTensors,
    base: &str,
) -> Result<DenseWeight> {
    if is_hf_nvfp4(st, base) {
        return Err(Error::Msg(format!(
            "qwen35: {base} must remain unquantized for this checkpoint layout"
        )));
    }
    load_hf_dense16(dev, st, &format!("{base}.weight")).map(DenseWeight::Dense16)
}

fn load_hf_required_nvfp4(dev: &dyn Device, st: &SafeTensors, base: &str) -> Result<DenseWeight> {
    if !is_hf_nvfp4(st, base) {
        return Err(Error::Msg(format!(
            "qwen35: {base} is not packed NVFP4; refusing host dequantization"
        )));
    }
    load_hf_nvfp4(dev, st, base).map(DenseWeight::Nvfp4)
}

fn validate_hf_weight_kind(st: &SafeTensors, base: &str, nvfp4: bool) -> Result<()> {
    if nvfp4 {
        st.load_nvfp4_tensor(base)?;
        return Ok(());
    }
    if is_hf_nvfp4(st, base) {
        return Err(Error::Msg(format!(
            "qwen35: {base} must remain unquantized for this checkpoint layout"
        )));
    }
    let (_, dtype, shape) = st.tensor(&format!("{base}.weight"))?;
    if shape.len() != 2 || !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
        return Err(Error::Msg(format!(
            "qwen35: {base}.weight must be a 2-D F16/BF16/F32 tensor, got {dtype} {shape:?}"
        )));
    }
    Ok(())
}

/// Validate the checkpoint's quantization policy before the first upload.
/// This avoids discovering a late-layer mismatch after most of the model is
/// already resident and keeps every high-precision attention tensor explicit.
fn validate_hf_quant_policy(st: &SafeTensors, cfg: &Qwen35Config) -> Result<()> {
    for i in 0..cfg.n_layers {
        let p = format!("model.language_model.layers.{i}");
        if cfg.is_full_layer(i) {
            let a = format!("{p}.self_attn");
            for name in ["q_proj", "k_proj", "v_proj"] {
                validate_hf_weight_kind(st, &format!("{a}.{name}"), false)?;
            }
            validate_hf_weight_kind(st, &format!("{a}.o_proj"), true)?;
        } else {
            let a = format!("{p}.linear_attn");
            for name in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                validate_hf_weight_kind(st, &format!("{a}.{name}"), false)?;
            }
        }
        for name in ["gate_proj", "up_proj", "down_proj"] {
            validate_hf_weight_kind(st, &format!("{p}.mlp.{name}"), true)?;
        }
    }
    Ok(())
}

/// Load the language model from a ModelOpt NVFP4 SafeTensors directory.
/// Only explicitly named language tensors are uploaded, so multimodal
/// `visual*` tensors remain mmap-only and never become device residents.
pub fn load_nvfp4_dir(dev: &dyn Device, dir: &str) -> Result<Qwen35Model> {
    let load_t0 = std::time::Instant::now();
    let mut cfg = Qwen35Config::from_hf_dir(dir)?;
    cfg.fp8_kv = true;
    let st = SafeTensors::open_dir(dir)?;
    validate_hf_quant_policy(&st, &cfg)?;

    let (embed_bytes, embed_dtype, embed_shape) =
        st.tensor("model.language_model.embed_tokens.weight")?;
    if embed_shape != [cfg.vocab, cfg.hidden]
        || !matches!(embed_dtype, DType::F16 | DType::BF16 | DType::F32)
    {
        return Err(Error::Msg(format!(
            "qwen35: unexpected embedding {embed_dtype} {embed_shape:?}"
        )));
    }
    let tok_embd = Tensor::new(dev.upload(embed_bytes)?, embed_shape.to_vec(), embed_dtype);
    let out_norm = load_hf_gemma_norm(dev, &st, "model.language_model.norm.weight")?;
    let lm_head = LmHead::Dense16(load_hf_dense16(dev, &st, "lm_head.weight")?);

    let inv_scale = 1.0 / (cfg.gdn_key_head_dim as f32).sqrt();
    let gdn_q_scale = Tensor::new(
        dev.upload(&f32s_to_bytes(&vec![
            inv_scale * inv_scale;
            cfg.gdn_key_head_dim
        ]))?,
        vec![cfg.gdn_key_head_dim],
        DType::F32,
    );
    let gdn_k_scale = Tensor::new(
        dev.upload(&f32s_to_bytes(&vec![inv_scale; cfg.gdn_key_head_dim]))?,
        vec![cfg.gdn_key_head_dim],
        DType::F32,
    );
    let gdn_q_scale_tiled_values =
        vec![inv_scale * inv_scale; cfg.gdn_num_k_heads * cfg.gdn_key_head_dim];
    let gdn_q_scale_tiled = Tensor::new(
        dev.upload(&f32s_to_bytes(&gdn_q_scale_tiled_values))?,
        vec![gdn_q_scale_tiled_values.len()],
        DType::F32,
    );
    let gdn_k_scale_tiled_values = vec![inv_scale; cfg.gdn_num_k_heads * cfg.gdn_key_head_dim];
    let gdn_k_scale_tiled = Tensor::new(
        dev.upload(&f32s_to_bytes(&gdn_k_scale_tiled_values))?,
        vec![gdn_k_scale_tiled_values.len()],
        DType::F32,
    );
    let gdn_conv_zero_bias_values = vec![0.0f32; cfg.gdn_conv_dim()];
    let gdn_conv_zero_bias = Tensor::new(
        dev.upload(&f32s_to_bytes(&gdn_conv_zero_bias_values))?,
        vec![gdn_conv_zero_bias_values.len()],
        DType::F32,
    );
    let gdn_gqa_cyclic_idx_values: Vec<u32> = (0..cfg.gdn_num_v_heads as u32)
        .map(|head| head % cfg.gdn_num_k_heads as u32)
        .collect();
    let gdn_gqa_cyclic_idx = Tensor::new(
        dev.upload(&u32s_to_bytes(&gdn_gqa_cyclic_idx_values))?,
        vec![gdn_gqa_cyclic_idx_values.len()],
        DType::U32,
    );

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("model.language_model.layers.{i}");
        let mixer = if cfg.is_full_layer(i) {
            let a = format!("{p}.self_attn");
            Mixer::Full(FullAttnWeights {
                attn_norm: load_hf_gemma_norm(dev, &st, &format!("{p}.input_layernorm.weight"))?,
                wq: load_hf_unquantized_weight(dev, &st, &format!("{a}.q_proj"))?,
                wk: load_hf_unquantized_weight(dev, &st, &format!("{a}.k_proj"))?,
                wv: load_hf_unquantized_weight(dev, &st, &format!("{a}.v_proj"))?,
                wo: load_hf_required_nvfp4(dev, &st, &format!("{a}.o_proj"))?,
                q_norm: load_hf_gemma_norm(dev, &st, &format!("{a}.q_norm.weight"))?,
                k_norm: load_hf_gemma_norm(dev, &st, &format!("{a}.k_norm.weight"))?,
            })
        } else {
            let a = format!("{p}.linear_attn");
            let (a_log_raw_host, _) = st.tensor_f32(&format!("{a}.A_log"))?;
            let neg_exp_a_log_host: Vec<f32> =
                a_log_raw_host.iter().map(|value| -value.exp()).collect();
            let (conv_flat, _) = st.tensor_f32(&format!("{a}.conv1d.weight"))?;
            let conv_repacked =
                repack_conv1d_weight(&conv_flat, cfg.gdn_conv_dim(), cfg.gdn_conv_kernel);
            Mixer::Gdn(GdnWeights {
                attn_norm: load_hf_gemma_norm(dev, &st, &format!("{p}.input_layernorm.weight"))?,
                wqkv: load_hf_unquantized_weight(dev, &st, &format!("{a}.in_proj_qkv"))?,
                wz: load_hf_unquantized_weight(dev, &st, &format!("{a}.in_proj_z"))?,
                w_alpha: load_hf_dense_f32(dev, &st, &format!("{a}.in_proj_a.weight"))?,
                w_beta: load_hf_dense_f32(dev, &st, &format!("{a}.in_proj_b.weight"))?,
                dt_bias: load_hf_vec_f32(dev, &st, &format!("{a}.dt_bias"))?,
                neg_exp_a_log: Tensor::new(
                    dev.upload(&f32s_to_bytes(&neg_exp_a_log_host))?,
                    vec![neg_exp_a_log_host.len()],
                    DType::F32,
                ),
                a_log_raw: Tensor::new(
                    dev.upload(&f32s_to_bytes(&a_log_raw_host))?,
                    vec![a_log_raw_host.len()],
                    DType::F32,
                ),
                conv_w: Tensor::new(
                    dev.upload(&f32s_to_bytes(&conv_repacked))?,
                    vec![conv_repacked.len()],
                    DType::F32,
                ),
                ssm_norm_w: load_hf_vec_f32(dev, &st, &format!("{a}.norm.weight"))?,
                wo: load_hf_unquantized_weight(dev, &st, &format!("{a}.out_proj"))?,
            })
        };
        let ffn = DenseFfn {
            ffn_norm: load_hf_gemma_norm(dev, &st, &format!("{p}.post_attention_layernorm.weight"))?,
            gate: load_hf_required_nvfp4(dev, &st, &format!("{p}.mlp.gate_proj"))?,
            up: load_hf_required_nvfp4(dev, &st, &format!("{p}.mlp.up_proj"))?,
            down: load_hf_required_nvfp4(dev, &st, &format!("{p}.mlp.down_proj"))?,
        };
        layers.push(Qwen35Layer { mixer, ffn });
    }

    eprintln!(
        "Qwen35: NVFP4 language load done in {:.2}s ({} layers, vision skipped)",
        load_t0.elapsed().as_secs_f64(),
        cfg.n_layers,
    );
    Ok(Qwen35Model {
        cfg,
        tok_embd,
        layers,
        out_norm,
        lm_head,
        mtp: None,
        gdn_q_scale,
        gdn_k_scale,
        gdn_q_scale_tiled,
        gdn_k_scale_tiled,
        gdn_conv_zero_bias,
        gdn_gqa_cyclic_idx,
        gdn_cyclic_gqa: false,
    })
}

fn gather_tokens_f32(dev: &dyn Device, model: &Qwen35Model, ids: &Tensor) -> Result<Tensor> {
    let gathered = ops::gather(dev, &model.tok_embd, ids)?;
    match gathered.dtype {
        DType::F32 => Ok(gathered),
        DType::F16 => ops::cast_f16_f32(dev, &gathered),
        DType::BF16 => ops::cast_bf16_f32(dev, &gathered),
        dtype => Err(Error::Msg(format!(
            "qwen35: unsupported embedding dtype {dtype}"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// KV / recurrent-state cache
// ─────────────────────────────────────────────────────────────────────────

enum FullKvCache {
    Dense { k: Tensor, v: Tensor },
    Fp8 {
        k_codes: Tensor,
        k_scales: Tensor,
        v_codes: Tensor,
        v_scales: Tensor,
        partial_o: Tensor,
        partial_m: Tensor,
        partial_l: Tensor,
        prefill: Option<(Tensor, Tensor, usize)>,
    },
}

struct FullAttnSlot {
    kv: FullKvCache,
    cap: usize,
}

struct GdnSlot {
    state: Tensor,      // f32 [Hv, Dv, Dk], read+written in place
    conv_state: Tensor, // f32 [conv_kernel-1, conv_dim], read+written in place
}

enum LayerSlot {
    Full(FullAttnSlot),
    Gdn(GdnSlot),
}

pub struct Qwen35Cache {
    slots: Vec<LayerSlot>,
}

/// Recurrent state at a completed prompt boundary. Full-attention KV is not
/// copied: earlier positions are immutable and a later suffix overwrites any
/// speculative/generated positions beyond the boundary. GDN state is mutable
/// in place, so it must be preserved before decode advances it.
pub struct Qwen35PromptSnapshot {
    gdn: Vec<Option<(Tensor, Tensor)>>,
}

impl Qwen35Cache {
    pub fn new(dev: &dyn Device, cfg: &Qwen35Config, max_seq: usize) -> Result<Self> {
        let mut slots = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            if cfg.is_full_layer(i) {
                let n = cfg.n_kv_heads * max_seq * cfg.head_dim;
                let kv = if cfg.fp8_kv {
                    let groups = n / 16;
                    FullKvCache::Fp8 {
                        k_codes: Tensor::new(dev.alloc_zeroed(n)?, vec![cfg.n_kv_heads, max_seq, cfg.head_dim], DType::U8),
                        k_scales: Tensor::new(dev.alloc_zeroed(groups * 4)?, vec![cfg.n_kv_heads, max_seq, cfg.head_dim / 16], DType::F32),
                        v_codes: Tensor::new(dev.alloc_zeroed(n)?, vec![cfg.n_kv_heads, max_seq, cfg.head_dim], DType::U8),
                        v_scales: Tensor::new(dev.alloc_zeroed(groups * 4)?, vec![cfg.n_kv_heads, max_seq, cfg.head_dim / 16], DType::F32),
                        partial_o: Tensor::new(dev.alloc_zeroed(cfg.n_head * 256 * cfg.head_dim * 4)?, vec![cfg.n_head, 256, cfg.head_dim], DType::F32),
                        partial_m: Tensor::new(dev.alloc_zeroed(cfg.n_head * 256 * 4)?, vec![cfg.n_head, 256], DType::F32),
                        partial_l: Tensor::new(dev.alloc_zeroed(cfg.n_head * 256 * 4)?, vec![cfg.n_head, 256], DType::F32),
                        prefill: None,
                    }
                } else {
                    FullKvCache::Dense {
                        k: Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, max_seq, cfg.head_dim], DType::F32),
                        v: Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, max_seq, cfg.head_dim], DType::F32),
                    }
                };
                slots.push(LayerSlot::Full(FullAttnSlot { kv, cap: max_seq }));
            } else {
                let state_n = cfg.gdn_num_v_heads * cfg.gdn_value_head_dim * cfg.gdn_key_head_dim;
                let state = Tensor::new(
                    dev.alloc_zeroed(state_n * 4)?,
                    vec![cfg.gdn_num_v_heads, cfg.gdn_value_head_dim, cfg.gdn_key_head_dim],
                    DType::F32,
                );
                let conv_n = (cfg.gdn_conv_kernel - 1) * cfg.gdn_conv_dim();
                let conv_state =
                    Tensor::new(dev.alloc_zeroed(conv_n * 4)?, vec![cfg.gdn_conv_kernel - 1, cfg.gdn_conv_dim()], DType::F32);
                slots.push(LayerSlot::Gdn(GdnSlot { state, conv_state }));
            }
        }
        Ok(Qwen35Cache { slots })
    }

    /// KV capacity of the first full-attn slot, or 0 if this model has none.
    pub fn cap(&self) -> usize {
        self.slots
            .iter()
            .find_map(|s| match s {
                LayerSlot::Full(f) => Some(f.cap),
                LayerSlot::Gdn(_) => None,
            })
            .unwrap_or(0)
    }

    pub fn uses_fp8_kv(&self) -> bool {
        self.slots.iter().any(|slot| matches!(slot, LayerSlot::Full(FullAttnSlot { kv: FullKvCache::Fp8 { .. }, .. })))
    }

    fn begin_prefill(&mut self, dev: &dyn Device, cfg: &Qwen35Config, start_pos: usize, total: usize) -> Result<()> {
        if !self.uses_fp8_kv() { return Ok(()); }
        for slot in &mut self.slots {
            if let LayerSlot::Full(FullAttnSlot {
                kv: FullKvCache::Fp8 { k_codes, k_scales, v_codes, v_scales, prefill, .. }, cap,
            }) = slot {
                let prefill_cap = if start_pos == 0 { total } else { *cap };
                let n = cfg.n_kv_heads * prefill_cap * cfg.head_dim;
                let k = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, prefill_cap, cfg.head_dim], DType::F32);
                let v = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, prefill_cap, cfg.head_dim], DType::F32);
                if start_pos > 0 {
                    ops::kv_dequant_fp8_e4m3_prefix(dev, k_codes, k_scales, &k, cfg.n_kv_heads, cfg.head_dim, *cap, 16, start_pos)?;
                    ops::kv_dequant_fp8_e4m3_prefix(dev, v_codes, v_scales, &v, cfg.n_kv_heads, cfg.head_dim, *cap, 16, start_pos)?;
                }
                *prefill = Some((k, v, prefill_cap));
            }
        }
        Ok(())
    }

    fn finish_prefill(&mut self) {
        for slot in &mut self.slots {
            if let LayerSlot::Full(FullAttnSlot { kv: FullKvCache::Fp8 { prefill, .. }, .. }) = slot {
                *prefill = None;
            }
        }
    }

    pub fn snapshot_prompt(&self, dev: &dyn Device) -> Result<Qwen35PromptSnapshot> {
        let mut gdn = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            match slot {
                LayerSlot::Full(_) => gdn.push(None),
                LayerSlot::Gdn(slot) => {
                    let state = ops::slice(dev, &slot.state, 0, slot.state.elem_count())?;
                    let conv_state = ops::slice(dev, &slot.conv_state, 0, slot.conv_state.elem_count())?;
                    gdn.push(Some((state, conv_state)));
                }
            }
        }
        Ok(Qwen35PromptSnapshot { gdn })
    }

    pub fn restore_prompt(&mut self, snapshot: Qwen35PromptSnapshot) -> Result<()> {
        if snapshot.gdn.len() != self.slots.len() {
            return Err(Error::Msg("qwen35: prompt snapshot layer count mismatch".into()));
        }
        for (slot, saved) in self.slots.iter_mut().zip(snapshot.gdn) {
            match (slot, saved) {
                (LayerSlot::Full(_), None) => {}
                (LayerSlot::Gdn(slot), Some((state, conv_state))) => {
                    slot.state = state;
                    slot.conv_state = conv_state;
                }
                _ => return Err(Error::Msg("qwen35: prompt snapshot layer kind mismatch".into())),
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Decode
// ─────────────────────────────────────────────────────────────────────────

/// QWEN35_DEBUG=1: per-token, layer-0-only checksum dump for oracle diffing
/// against `llama-eval-callback`'s per-node `sum = ...` output. Mirrors
/// `laguna.rs`'s `BUTTER_LAGUNA_DEBUG`/`dbg_stats` pattern. TEMPORARY —
/// gated dead weight when unset (one env read per call site).
fn dbg_sum(dev: &dyn Device, tag: &str, pos: u32, t: &Tensor) {
    let n = t.elem_count();
    let mut b = vec![0u8; n * 4];
    if dev.synchronize().is_err() || dev.download(t.buffer.as_ref(), &mut b).is_err() {
        eprintln!("QWEN35_DBG pos={pos} tag={tag}: <download failed>");
        return;
    }
    let f: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let sum: f64 = f.iter().map(|&v| v as f64).sum();
    eprintln!("QWEN35_DBG pos={pos} tag={tag} n={n} sum={sum:.6} head={:?}", &f[..4.min(n)]);
}

/// F-85 prefill wave debug aid: like `dbg_sum`, but for a batched `[T,
/// hidden]` prefill tensor -- prints ONE line PER ROW, tagged with that
/// row's absolute position, in the SAME `QWEN35_DBG pos=N tag=... sum=...`
/// format `dbg_sum` (decode) uses, so a prefill run's per-position
/// checksums can be grepped/diffed directly against a decode-loop run's
/// over the same prompt (both gated by `QWEN35_DEBUG=1`).
fn dbg_sum_rows(dev: &dyn Device, tag: &str, positions: &[u32], t: &Tensor, hidden: usize) {
    let n = t.elem_count();
    let mut b = vec![0u8; n * 4];
    if dev.synchronize().is_err() || dev.download(t.buffer.as_ref(), &mut b).is_err() {
        eprintln!("QWEN35_DBG(rows) tag={tag}: <download failed>");
        return;
    }
    let f: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    for (row, &pos) in positions.iter().enumerate() {
        let row_slice = &f[row * hidden..(row + 1) * hidden];
        let sum: f64 = row_slice.iter().map(|&v| v as f64).sum();
        eprintln!("QWEN35_DBG pos={pos} tag={tag} n={hidden} sum={sum:.6} head={:?}", &row_slice[..4.min(hidden)]);
    }
}

fn append_decode_full_kv(
    dev: &dyn Device, cfg: &Qwen35Config, slot: &FullAttnSlot,
    q: &Tensor, k: &Tensor, v: &Tensor, posbuf: &Tensor, n_kv: u32,
) -> Result<Tensor> {
    let heads_per_group = (cfg.n_head / cfg.n_kv_heads) as u32;
    match &slot.kv {
        FullKvCache::Dense { k: cache_k, v: cache_v } => {
            ops::kv_append(dev, k, cache_k, posbuf, cfg.head_dim, slot.cap, cfg.n_kv_heads * cfg.head_dim)?;
            ops::kv_append(dev, v, cache_v, posbuf, cfg.head_dim, slot.cap, cfg.n_kv_heads * cfg.head_dim)?;
            ops::sdpa_decode(dev, q, cache_k, cache_v, cfg.head_dim, n_kv, slot.cap as u32, heads_per_group, cfg.attn_scale())
        }
        FullKvCache::Fp8 { k_codes, k_scales, v_codes, v_scales, partial_o, partial_m, partial_l, .. } => {
            ops::kv_append_fp8_e4m3_many(dev, k, posbuf, k_codes, k_scales, cfg.n_kv_heads, cfg.head_dim, slot.cap, 16)?;
            ops::kv_append_fp8_e4m3_many(dev, v, posbuf, v_codes, v_scales, cfg.n_kv_heads, cfg.head_dim, slot.cap, 16)?;
            let blocks = if n_kv <= 1024 { 64 } else if n_kv <= 8192 { 128 } else { 256 };
            ops::sdpa_decode_fp8_e4m3_2pass(
                dev, q, k_codes, k_scales, v_codes, v_scales, partial_o, partial_m, partial_l,
                cfg.head_dim, n_kv, slot.cap as u32, heads_per_group, 16, blocks, cfg.attn_scale(),
            )
        }
    }
}

fn full_attn_layer(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    w: &FullAttnWeights,
    slot: &mut FullAttnSlot,
    x: &Tensor,
    pos: u32,
) -> Result<Tensor> {
    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;

    let qg = w.wq.gemv(dev, &h)?; // [n_head*2*head_dim]
    let k = w.wk.gemv(dev, &h)?; // [n_kv_heads*head_dim]
    let v = w.wv.gemv(dev, &h)?; // [n_kv_heads*head_dim]

    // Per-head [q(256); gate(256)] de-interleave (see `strided_head_half`'s doc).
    let q = ops::strided_head_half(dev, &qg, cfg.n_head, cfg.head_dim, 0)?;
    let gate = ops::strided_head_half(dev, &qg, cfg.n_head, cfg.head_dim, 1)?;

    let q = q.reshaped(vec![cfg.n_head, cfg.head_dim]);
    let k = k.reshaped(vec![cfg.n_kv_heads, cfg.head_dim]);
    let v = v.reshaped(vec![cfg.n_kv_heads, cfg.head_dim]);

    // Weighted per-head QK RMSNorm (full head_dim), before RoPE.
    let q = ops::rms_norm(dev, &q, &w.q_norm, cfg.eps)?;
    let k = ops::rms_norm(dev, &k, &w.k_norm, cfg.eps)?;

    // Partial-rotary RoPE: rotate only the first `rope_n_rot` (64) of 256
    // dims, plain (YaRN disabled: freq_scale=1, ext_factor=0, attn_factor=1).
    let q = ops::rope_yarn_partial(dev, &q, pos, cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144)?;
    let k = ops::rope_yarn_partial(dev, &k, pos, cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144)?;

    let posbuf = Tensor::new(dev.upload(&pos.to_le_bytes())?, vec![1], DType::U32);
    let n_kv_used = pos + 1;
    let attn = append_decode_full_kv(dev, cfg, slot, &q, &k, &v, &posbuf, n_kv_used)?; // [n_head, head_dim]

    // Sigmoid output gate (NOT swish — see module doc).
    let attn_flat = attn.reshaped(vec![cfg.n_head * cfg.head_dim]);
    let gated = ops::sigmoid_mul(dev, &attn_flat, &gate)?;

    let o = w.wo.gemv(dev, &gated)?;
    ops::add(dev, x, &o)
}

/// Embed → 64 layers (GDN or full-attn per `cfg.is_full_layer`) → final
/// RMSNorm → Q8 lm_head → next-token logits `[vocab]`. `pos` is this
/// token's absolute sequence position (0-based); call once per token,
/// reusing `cache` so KV/recurrent state persists across calls (prompt
/// tokens included — see the module doc's "decode-only" note).
pub fn decode_step(dev: &dyn Device, model: &Qwen35Model, cache: &mut Qwen35Cache, token_id: u32, pos: u32) -> Result<Tensor> {
    Ok(decode_step_ex(dev, model, cache, token_id, pos)?.0)
}

/// Same as [`decode_step`] but also returns `xn` — the post-`output_norm`
/// hidden state that fed the lm_head (i.e. `xn` such that
/// `lm_head.gemv(xn) == logits`). Added for MTP self-speculative decode
/// (see `mtp_draft_step`/`speculative_round`): the MTP drafter's seed input
/// is exactly this tensor, and re-deriving it by re-running the trunk would
/// double the backbone cost. `decode_step` is unchanged (still the plain,
/// single-return entry point every existing caller uses) — it now just
/// forwards to this and drops `xn`, so this refactor is behavior-preserving
/// for all pre-existing call sites.
pub fn decode_step_ex(
    dev: &dyn Device, model: &Qwen35Model, cache: &mut Qwen35Cache, token_id: u32, pos: u32,
) -> Result<(Tensor, Tensor)> {
    let cfg = &model.cfg;
    let debug = std::env::var("QWEN35_DEBUG").map(|v| v == "1").unwrap_or(false);
    let tok_buf = Tensor::new(dev.upload(&token_id.to_le_bytes())?, vec![1], DType::U32);
    let mut x = gather_tokens_f32(dev, model, &tok_buf)?.reshaped(vec![cfg.hidden]);
    if debug { dbg_sum(dev, "x_embed", pos, &x); }

    for (i, layer) in model.layers.iter().enumerate() {
        x = match (&layer.mixer, &mut cache.slots[i]) {
            (Mixer::Full(w), LayerSlot::Full(slot)) => full_attn_layer(dev, cfg, w, slot, &x, pos)?,
            (Mixer::Gdn(w), LayerSlot::Gdn(slot)) => gdn_layer_full(dev, cfg, model, w, slot, &x, i, pos, debug)?,
            _ => return Err(Error::Msg(format!("qwen35: layer {i} mixer/slot kind mismatch"))),
        };
        if debug && i == 0 { dbg_sum(dev, "attn_residual-0", pos, &x); }
        let h2 = ops::rms_norm(dev, &x, &layer.ffn.ffn_norm, cfg.eps)?;
        let gate = layer.ffn.gate.gemv(dev, &h2)?;
        let up = layer.ffn.up.gemv(dev, &h2)?;
        let act = ops::swiglu(dev, &gate, &up)?;
        let ffn_out = layer.ffn.down.gemv(dev, &act)?;
        x = ops::add(dev, &x, &ffn_out)?;
        if debug && i == 0 { dbg_sum(dev, "l_out-0", pos, &x); }
    }

    let xn = ops::rms_norm(dev, &x, &model.out_norm, cfg.eps)?;
    let logits = model.lm_head.gemv(dev, &xn)?;
    Ok((logits, xn))
}

// ─────────────────────────────────────────────────────────────────────────
// F-85 prefill wave: BATCHED chunked prefill.
//
// Replaces "prefill via repeated decode_step" (the module doc's documented
// v1 limitation) with a real batched forward: `T` prompt-token rows
// processed per layer in one dispatch chain instead of `T` sequential
// per-token calls. Mirrors `laguna.rs`'s `prefill`/`prefill_layer`
// structure and conventions (chunked outer loop, `start`/`positions`
// threading, `t_len`-scalar dispatch, last-row-only lm_head) -- see that
// file's doc for the established pattern this follows.
//
// GDN layers: `iron_conv1d_causal_prefill` (state-continued via
// `ops::concat_rows_f32` prepending the carried `conv_state` history, see
// `prefill_layer_gdn`'s doc) + `ops::gdn_qknorm_prepass_chunk` +
// `ops::gdn_prep_chunk_fast_27b` (the validated Hv=48 GDN chunk kernel,
// see `GDN_PREFILL_CONTRACT.md`) -- state stays register-resident across
// the whole chunk, one load/store per chunk per layer instead of per token.
//
// Full-attn layers: batched QKV projection + per-head RMSNorm + batched
// partial-rotary RoPE (`ops::rope_yarn_partial_many`) + batched KV-cache
// write (`ops::kv_append_many`) + batched causal SDPA
// (`ops::sdpa_multi_tc_varlen`) + batched sigmoid output gate
// (`ops::strided_head_half_many` + `ops::sigmoid_mul`).
//
// GEMM: every dense projection routes through `Q4Dense::gemm` ->
// `ops::gemm_q4_mpp` (see that method's doc for the marlin-vs-mpp
// microbench finding). Decode's separate M=1 U4B8 Marlin path
// (`Q4Dense::gemv`) is untouched.
// ─────────────────────────────────────────────────────────────────────────

/// One BATCHED GDN prefill layer: `T` prompt-token rows processed in one
/// forward instead of `T` sequential per-token recurrence steps. `x` is
/// `[T, hidden]` f32.
///
/// State-carry across external chunks: `iron_conv1d_causal_prefill`
/// (unlike the GDN recurrence kernel) always starts from ZERO initial
/// conv state -- it has no `state_in` parameter, see its doc. To thread
/// this layer's carried `conv_state` (the `conv_kernel-1` most recent RAW
/// qkv rows) across chunks, this function PREPENDS `conv_state` onto the
/// chunk's `T` new raw qkv rows (`ops::concat_rows_f32`) before calling
/// the zero-state kernel over all `conv_kernel-1+T` rows, then slices off
/// the first `conv_kernel-1` (warm-up, already-known) output rows to get
/// this chunk's real `conv_out`, and re-derives the new `conv_state` as
/// the LAST `conv_kernel-1` rows of the same concatenated input buffer
/// (correct for any `T`, including `T < conv_kernel-1`, since it's just
/// "the last `conv_kernel-1` rows of the [history; new] sequence" either
/// way). At the very start of a fresh sequence `conv_state` is the
/// `Qwen35Cache::new` zero-init, giving the same zero-state behavior
/// `conv1d_causal_silu_many`'s decode-step path starts from.
///
/// The recurrent GDN state itself (`slot.state`, `[Hv,Dv,Dk]`) needs no
/// such trick: `ops::gdn_prep_chunk_fast_27b` takes `state_in` directly
/// and IS the state-carrying kernel (see its doc / `GDN_PREFILL_CONTRACT.md`).
#[allow(clippy::too_many_arguments)]
fn prefill_layer_gdn(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    model: &Qwen35Model,
    w: &GdnWeights,
    slot: &mut GdnSlot,
    x: &Tensor, // [T, hidden]
    t: usize,
    t_len: &Tensor, // [1] u32, == t
) -> Result<Tensor> {
    let hidden = cfg.hidden;
    let conv_dim = cfg.gdn_conv_dim();
    let kc = cfg.gdn_conv_kernel;
    let value_dim = cfg.gdn_value_dim();
    let hv = cfg.gdn_num_v_heads;
    let hk = cfg.gdn_num_k_heads;
    let dk = cfg.gdn_key_head_dim;
    let dv = cfg.gdn_value_head_dim;

    let trace = std::env::var("QWEN35_PREFILL_TRACE").is_ok();
    macro_rules! tr {
        ($tag:expr) => {
            if trace { dev.synchronize().map_err(|e| Error::Msg(format!("prefill_layer_gdn trace fault at {}: {e:?}", $tag)))?; eprintln!("QWEN35_PREFILL_TRACE gdn t={t} ok: {}", $tag); }
        };
    }

    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?; // [T, hidden]
    tr!("rms_norm(attn_norm)");
    let qkv_raw = w.wqkv.gemm(dev, &h, t)?; // [T, conv_dim]  (PRE-conv, matches decode's w.wqkv.gemv feed)
    tr!("wqkv.gemm");
    let z = w.wz.gemm(dev, &h, t)?; // [T, value_dim]  (output gate, never conv'd)
    tr!("wz.gemm");

    // conv1d+SiLU, state-continued (see this fn's doc).
    let concatenated = ops::concat_rows_f32(dev, &slot.conv_state, kc - 1, &qkv_raw.reshaped(vec![t * conv_dim]), t, conv_dim)?;
    tr!("concat_rows_f32");
    let conv_full = ops::conv1d_causal_prefill(dev, &concatenated, &w.conv_w, &model.gdn_conv_zero_bias, kc - 1 + t, conv_dim, kc)?;
    tr!("conv1d_causal_prefill");
    let conv_out = ops::slice(dev, &conv_full, (kc - 1) * conv_dim, t * conv_dim)?; // [T, conv_dim]
    slot.conv_state = ops::slice(dev, &concatenated, t * conv_dim, (kc - 1) * conv_dim)?.reshaped(vec![kc - 1, conv_dim]);
    tr!("slice x2");

    // Batched alpha/beta raw projections: [T, hidden] -> [T, Hv].
    let a_raw = ops::matmul(dev, &w.w_alpha, &h)?; // [T, Hv]
    let b_raw = ops::matmul(dev, &w.w_beta, &h)?; // [T, Hv]
    tr!("matmul(alpha/beta)");

    // Hoisted q/k RMSNorm pre-pass, then the chunked GDN recurrence.
    // NOTE: the TILED `[Hk*Dk]` scale buffers, not `gdn_q_scale`/
    // `gdn_k_scale` (`[Dk]` only) -- see `Qwen35Model::gdn_q_scale_tiled`'s
    // doc for why the untiled buffer is an out-of-bounds read here.
    let (q_normed, k_normed) = ops::gdn_qknorm_prepass_chunk(
        dev, &conv_out, &model.gdn_q_scale_tiled, &model.gdn_k_scale_tiled, t_len, t, dk, dv, hv, hk,
    )?;
    tr!("gdn_qknorm_prepass_chunk");
    let y = ops::gdn_prep_chunk_fast_27b(
        dev, &conv_out, &w.a_log_raw, &w.dt_bias, &a_raw, &b_raw, &q_normed, &k_normed, &slot.state,
        &model.gdn_conv_zero_bias, t_len, t,
        model.gdn_cyclic_gqa,
    )?; // [T, Hv, Dv]
    tr!("gdn_prep_chunk_fast_27b");

    // Gate-AFTER-norm tail (unfused, batched): rmsNorm(y, ssm_norm_w) *
    // silu(z), per-`(t,hv)` row over `Dv` -- see `gdn_tail_gate`'s doc for
    // why this is NOT `gated_group_rmsnorm`'s gate-BEFORE-norm convention.
    // `ops::rms_norm` already normalizes per-ROW over the last axis with a
    // per-element weight, so `[T*Hv, Dv]` batches this exactly like the
    // full-attn q/k-norm batching above.
    let y2 = ops::rms_norm(dev, &y.reshaped(vec![t * hv, dv]), &w.ssm_norm_w, cfg.eps)?;
    let z_silu = ops::silu(dev, &z.reshaped(vec![t * value_dim]))?;
    let gated = ops::mul(dev, &y2.reshaped(vec![t * value_dim]), &z_silu)?; // [T, value_dim]
    tr!("tail gate");

    let o = w.wo.gemm(dev, &gated.reshaped(vec![t, value_dim]), t)?; // [T, hidden]
    tr!("wo.gemm");
    let out = ops::add(dev, &x.reshaped(vec![t * hidden]), &o.reshaped(vec![t * hidden]))?;
    tr!("add residual");
    Ok(out.reshaped(vec![t, hidden]))
}

/// One BATCHED full-attention prefill layer: `T` prompt-token rows
/// processed in one forward instead of `T` sequential `full_attn_layer`
/// calls. Same op sequence as `full_attn_layer` (RMSNorm -> fused Q+gate
/// proj -> de-interleave -> per-head QK-RMSNorm -> partial-rotary RoPE ->
/// KV-cache write -> causal SDPA -> sigmoid gate -> o_proj -> residual),
/// batched over `[T, ...]` tensors. No sliding window in this
/// architecture (unlike Laguna) -- every full-attn layer attends the full
/// causal prefix, so KV writes go straight into `cache`'s persistent
/// growing slot (matches decode's `full_attn_layer`; no separate linear
/// prefill scratch needed).
#[allow(clippy::too_many_arguments)]
fn prefill_layer_full(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    w: &FullAttnWeights,
    slot: &mut FullAttnSlot,
    x: &Tensor, // [T, hidden]
    t: usize,
    start: usize,
    positions: &Tensor, // [T] u32
) -> Result<Tensor> {
    let hd = cfg.head_dim;
    let hidden = cfg.hidden;
    let n_head = cfg.n_head;
    let n_kv_heads = cfg.n_kv_heads;
    let heads_per_group = (n_head / n_kv_heads) as u32;
    let trace = std::env::var("QWEN35_PREFILL_TRACE").is_ok();
    macro_rules! tr {
        ($tag:expr) => {
            if trace { dev.synchronize().map_err(|e| Error::Msg(format!("prefill_layer_full trace fault at {}: {e:?}", $tag)))?; eprintln!("QWEN35_PREFILL_TRACE full t={t} ok: {}", $tag); }
        };
    }

    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?; // [T, hidden]
    tr!("rms_norm(attn_norm)");
    let qg = w.wq.gemm(dev, &h, t)?; // [T, n_head*2*head_dim]
    tr!("wq.gemm");
    let k = w.wk.gemm(dev, &h, t)?; // [T, n_kv_heads*head_dim]
    tr!("wk.gemm");
    let v = w.wv.gemm(dev, &h, t)?; // [T, n_kv_heads*head_dim]
    tr!("wv.gemm");

    let q = ops::strided_head_half_many(dev, &qg.reshaped(vec![t * n_head * 2 * hd]), t, n_head, hd, 0)?; // [T, n_head*hd]
    let gate = ops::strided_head_half_many(dev, &qg.reshaped(vec![t * n_head * 2 * hd]), t, n_head, hd, 1)?; // [T, n_head*hd]
    tr!("strided_head_half_many x2");

    let q2 = ops::rms_norm(dev, &q.reshaped(vec![t * n_head, hd]), &w.q_norm, cfg.eps)?;
    let k2 = ops::rms_norm(dev, &k.reshaped(vec![t * n_kv_heads, hd]), &w.k_norm, cfg.eps)?;
    tr!("q/k rms_norm");
    // Partial-rotary RoPE, plain (YaRN disabled) -- same fixed params as
    // `full_attn_layer`'s per-token `ops::rope_yarn_partial` call.
    let q3 = ops::rope_yarn_partial_many(
        dev, &q2.reshaped(vec![t, n_head, hd]), positions, n_head,
        cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144,
    )?;
    let k3 = ops::rope_yarn_partial_many(
        dev, &k2.reshaped(vec![t, n_kv_heads, hd]), positions, n_kv_heads,
        cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144,
    )?;
    tr!("rope_yarn_partial_many x2");
    let v3 = v.reshaped(vec![t, n_kv_heads, hd]);

    let (cache_k, cache_v, dense_cap) = match &slot.kv {
        FullKvCache::Dense { k, v } => (k, v, slot.cap),
        FullKvCache::Fp8 { k_codes, k_scales, v_codes, v_scales, prefill, .. } => {
            ops::kv_append_fp8_e4m3_many(dev, &k3, positions, k_codes, k_scales, n_kv_heads, hd, slot.cap, 16)?;
            ops::kv_append_fp8_e4m3_many(dev, &v3, positions, v_codes, v_scales, n_kv_heads, hd, slot.cap, 16)?;
            let (k, v, cap) = prefill.as_ref().ok_or_else(|| Error::Msg("qwen35: packed KV prefill scratch is not prepared".into()))?;
            (k, v, *cap)
        }
    };
    ops::kv_append_many(dev, &k3, positions, cache_k, n_kv_heads, hd, dense_cap)?;
    ops::kv_append_many(dev, &v3, positions, cache_v, n_kv_heads, hd, dense_cap)?;
    tr!("kv_append_many x2");

    // Every query attends the full causal prefix: seg_lo = 0 for all rows,
    // no sliding-window skip (win=0) -- matches Laguna's full-attn-layer
    // branch (`sdpa_varlen_1`), this architecture has no sliding layers.
    let seg_lo = Tensor::new(dev.alloc_zeroed(t * 4)?, vec![t], DType::U32);
    let attn = ops::sdpa_multi_tc_varlen(
        dev, &q3, cache_k, cache_v, &seg_lo, 0, hd, n_head as u32,
        start as u32, t as u32, dense_cap as u32, heads_per_group, true, cfg.attn_scale(), 0,
    )?; // [T, n_head, hd]
    tr!("sdpa_multi_tc_varlen");

    // Sigmoid output gate (NOT swish -- see module doc), batched.
    let attn_flat = attn.reshaped(vec![t * n_head * hd]);
    let gated = ops::sigmoid_mul(dev, &attn_flat, &gate)?;
    tr!("sigmoid_mul");

    let o = w.wo.gemm(dev, &gated.reshaped(vec![t, n_head * hd]), t)?;
    tr!("wo.gemm");
    let out = ops::add(dev, &x.reshaped(vec![t * hidden]), &o.reshaped(vec![t * hidden]));
    tr!("add residual");
    out
}

/// One BATCHED transformer layer (GDN or full-attn per `cfg.is_full_layer`)
/// dispatcher + the (mixer-independent) dense FFN, mirroring
/// `laguna.rs::prefill_layer`'s structure. `x` is `[T, hidden]`.
#[allow(clippy::too_many_arguments)]
fn prefill_layer(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    model: &Qwen35Model,
    layer_idx: usize,
    layer: &Qwen35Layer,
    slot: &mut LayerSlot,
    x: &Tensor,
    t: usize,
    start: usize,
    positions: &Tensor,
    t_len: &Tensor,
) -> Result<Tensor> {
    let x1 = match (&layer.mixer, slot) {
        (Mixer::Full(w), LayerSlot::Full(s)) => prefill_layer_full(dev, cfg, w, s, x, t, start, positions)?,
        (Mixer::Gdn(w), LayerSlot::Gdn(s)) => prefill_layer_gdn(dev, cfg, model, w, s, x, t, t_len)?,
        _ => return Err(Error::Msg(format!("qwen35: prefill layer {layer_idx} mixer/slot kind mismatch"))),
    };
    let x1 = x1.reshaped(vec![t, cfg.hidden]);
    let h2 = ops::rms_norm(dev, &x1, &layer.ffn.ffn_norm, cfg.eps)?;
    let gate = layer.ffn.gate.gemm(dev, &h2, t)?;
    let up = layer.ffn.up.gemm(dev, &h2, t)?;
    let act = ops::swiglu(dev, &gate, &up)?;
    let ffn_out = layer.ffn.down.gemm(dev, &act, t)?;
    let out = ops::add(dev, &x1.reshaped(vec![t * cfg.hidden]), &ffn_out.reshaped(vec![t * cfg.hidden]))?;
    Ok(out.reshaped(vec![t, cfg.hidden]))
}

/// Default prefill chunk size, mirroring `laguna.rs::default_prefill_chunk`.
/// 512 is the microbenched pp512 operating point the task's bench table
/// reports against; kept as a plain constant here (not env-tunable) since
/// the caller can always pass an explicit `chunk` to `prefill`.
fn default_prefill_chunk(total: usize) -> usize {
    total.min(512).max(1)
}

/// BATCHED chunked prefill: embed all `tokens` -> chunks of up to `chunk`
/// rows -> every layer (GDN or full-attn) -> final RMSNorm + lm_head on
/// the LAST token of the LAST chunk only (next-token logits). `cache` is
/// reused unmodified in shape/convention from the decode path -- after
/// `prefill` returns, `decode_step`/`decode_step_ex` can continue from
/// `pos = tokens.len()` exactly as if every token had been walked through
/// `decode_step` one at a time (see the correctness tests' identity gate).
///
/// `chunk == 0` picks [`default_prefill_chunk`]. Mirrors
/// `laguna.rs::prefill`'s structure (see that fn's doc) -- no separate
/// prefill scratch struct is needed here (no sliding-window layers in this
/// architecture), so `cache` alone (the same `Qwen35Cache` decode uses) is
/// threaded through both the chunk loop and the final compaction step
/// laguna needs and this architecture doesn't.
///
/// Equivalent to [`prefill_at`] with `start_pos = 0`.
pub fn prefill(dev: &dyn Device, model: &Qwen35Model, cache: &mut Qwen35Cache, tokens: &[u32], chunk: usize) -> Result<Tensor> {
    prefill_at(dev, model, cache, tokens, 0, chunk, None)
}

/// Like [`prefill`], but RoPE / KV / SDPA absolute positions start at
/// `start_pos` instead of 0. Used for append-only suffix prefill when the
/// prompt strictly extends a live cache (`start_pos == n_past`). Passing
/// the suffix alone with `start_pos = 0` would overwrite KV slot 0.
pub fn prefill_at(
    dev: &dyn Device,
    model: &Qwen35Model,
    cache: &mut Qwen35Cache,
    tokens: &[u32],
    start_pos: usize,
    chunk: usize,
    abort: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Tensor> {
    let cfg = &model.cfg;
    let total = tokens.len();
    if total == 0 {
        return Err(Error::Msg("prefill: empty prompt".into()));
    }
    cache.begin_prefill(dev, cfg, start_pos, total)?;
    let result = (|| {
    let chunk = if chunk == 0 { default_prefill_chunk(total) } else { chunk.max(1) };
    PREFILL_CHUNKS_DONE.store(0, AtomicOrdering::Relaxed);

    let mut x_last: Option<Tensor> = None;
    let mut t_last = 0usize;
    let mut start = 0usize;
    while start < total {
        if abort.map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).unwrap_or(false) {
            return Err(Error::Msg("prefill: aborted".into()));
        }
        let t = chunk.min(total - start);
        let chunk_tokens = &tokens[start..start + t];
        let abs_start = start_pos + start;

        let ids = Tensor::new(dev.upload(&u32s_to_bytes(chunk_tokens))?, vec![t], DType::U32);
        let mut x = gather_tokens_f32(dev, model, &ids)?.reshaped(vec![t, cfg.hidden]);

        let pos_vec: Vec<u32> = (0..t).map(|i| (abs_start + i) as u32).collect();
        let positions = Tensor::new(dev.upload(&u32s_to_bytes(&pos_vec))?, vec![t], DType::U32);
        let t_len = Tensor::new(dev.upload(&(t as u32).to_le_bytes())?, vec![1], DType::U32);

        let debug = std::env::var("QWEN35_DEBUG").map(|v| v == "1").unwrap_or(false);
        if debug {
            dbg_sum_rows(dev, "x_embed", &pos_vec, &x, cfg.hidden);
        }
        for (i, layer) in model.layers.iter().enumerate() {
            x = prefill_layer(dev, cfg, model, i, layer, &mut cache.slots[i], &x, t, abs_start, &positions, &t_len)?;
            if std::env::var("QWEN35_PREFILL_SYNC").is_ok() {
                dev.synchronize().map_err(|e| Error::Msg(format!("prefill fault after layer {i}: {e:?}")))?;
            }
            if debug && i == 0 {
                dbg_sum_rows(dev, "l_out-0", &pos_vec, &x, cfg.hidden);
            }
        }

        x_last = Some(x);
        t_last = t;
        start += t;
        PREFILL_CHUNKS_DONE.fetch_add(1, AtomicOrdering::Relaxed);
    }
    let x = x_last.expect("prefill: total > 0 was checked above, so at least one chunk always runs");

    let last_row = ops::slice(dev, &x.reshaped(vec![t_last * cfg.hidden]), (t_last - 1) * cfg.hidden, cfg.hidden)?;
    let xn = ops::rms_norm(dev, &last_row, &model.out_norm, cfg.eps)?;
    let logits = model.lm_head.gemv(dev, &xn)?;
    Ok(logits)
    })();
    cache.finish_prefill();
    result
}

// ─────────────────────────────────────────────────────────────────────────
// Round-9 step-time census (temporary diagnostic -- see
// wh-butter-cuda/tests/qwen35_census.rs for the harness). Not on any hot
// path: `decode_step`/`decode_step_ex`/`speculative_round` are untouched;
// this is a parallel instrumented copy of the same forward, opted into
// explicitly by calling `decode_step_census` instead.
// ─────────────────────────────────────────────────────────────────────────

/// Millisecond accumulators for one phase-bucketed decode step. Buckets:
///   - `routed_marlin_ms`: the 5 F-85 round-7/8 marlin-routed weight
///     classes' GEMV/GEMM (ffn_gate/up/down, full-attn wq, gdn wqkv).
///   - `small_proj_ms`: every OTHER dense projection, still plain
///     `gemv_q4` (full-attn wk/wv/wo, gdn wz/wo).
///   - `gdn_chain_ms`: the GDN-specific nonlinear chain across all 48 GDN
///     layers -- conv1d, q/k GDN-RMSNorm, alpha/beta gate projections,
///     gate/beta activation, the `gated_delta_step` recurrence, ssm_norm,
///     silu, mul.
///   - `attention_ms`: the full-attn-specific chain across all 16 full-
///     attn layers -- q/k RMSNorm, RoPE, KV-cache append, `sdpa_decode`,
///     sigmoid output gate.
///   - `elementwise_ms`: the glue between the buckets above -- per-layer
///     `attn_norm`/`ffn_norm` RMSNorm, `swiglu`, residual adds.
///   - `lm_head_ms`: final `output_norm` + lm_head GEMV.
///   - `embed_ms`: the token-embedding gather.
/// Host-side sync/download/argmax is deliberately NOT tracked here -- the
/// harness measures that itself around the call (matching how the plain
/// decode loop already spends it, see `qwen35_bench_decode.rs`), since it
/// happens strictly after this function returns.
#[derive(Debug, Clone, Copy, Default)]
pub struct CensusBuckets {
    pub embed_ms: f64,
    pub routed_marlin_ms: f64,
    pub small_proj_ms: f64,
    pub gdn_chain_ms: f64,
    pub attention_ms: f64,
    pub elementwise_ms: f64,
    pub lm_head_ms: f64,
}

impl CensusBuckets {
    pub fn total_ms(&self) -> f64 {
        self.embed_ms
            + self.routed_marlin_ms
            + self.small_proj_ms
            + self.gdn_chain_ms
            + self.attention_ms
            + self.elementwise_ms
            + self.lm_head_ms
    }
}

/// Sync the device, add the elapsed time since `*t0` to `*bucket`, and
/// reset `*t0` to now. Every call is a `dev.synchronize()` round trip
/// (dozens per layer, ~700/step) so the ABSOLUTE per-bucket numbers carry
/// a real instrumentation tax -- the harness reports the census total
/// against an uninstrumented reference measurement so the tax is visible
/// rather than silently baked in. Bucket PROPORTIONS (relative to the
/// instrumented total) are the load-bearing number, not the absolute ms.
fn mark(dev: &dyn Device, t0: &mut std::time::Instant, bucket: &mut f64) -> Result<()> {
    dev.synchronize()?;
    let now = std::time::Instant::now();
    *bucket += now.duration_since(*t0).as_secs_f64() * 1000.0;
    *t0 = now;
    Ok(())
}

/// Phase-bucketed twin of [`decode_step_ex`] -- same forward, same
/// call/dependency order, with `mark` calls dropped in at bucket
/// boundaries. See `CensusBuckets`' doc for what's in each bucket.
pub fn decode_step_census(
    dev: &dyn Device,
    model: &Qwen35Model,
    cache: &mut Qwen35Cache,
    token_id: u32,
    pos: u32,
    b: &mut CensusBuckets,
) -> Result<(Tensor, Tensor)> {
    let cfg = &model.cfg;
    dev.synchronize()?; // drain anything in flight from the caller before starting the clock
    let mut t0 = std::time::Instant::now();

    let tok_buf = Tensor::new(dev.upload(&token_id.to_le_bytes())?, vec![1], DType::U32);
    let mut x = gather_tokens_f32(dev, model, &tok_buf)?.reshaped(vec![cfg.hidden]);
    mark(dev, &mut t0, &mut b.embed_ms)?;

    for (i, layer) in model.layers.iter().enumerate() {
        x = match (&layer.mixer, &mut cache.slots[i]) {
            (Mixer::Full(w), LayerSlot::Full(slot)) => {
                full_attn_layer_census(dev, cfg, w, slot, &x, pos, &mut t0, b)?
            }
            (Mixer::Gdn(w), LayerSlot::Gdn(slot)) => {
                gdn_layer_full_census(dev, cfg, model, w, slot, &x, pos, &mut t0, b)?
            }
            _ => return Err(Error::Msg(format!("qwen35: layer {i} mixer/slot kind mismatch"))),
        };
        let h2 = ops::rms_norm(dev, &x, &layer.ffn.ffn_norm, cfg.eps)?;
        mark(dev, &mut t0, &mut b.elementwise_ms)?;
        let gate = layer.ffn.gate.gemv(dev, &h2)?;
        let up = layer.ffn.up.gemv(dev, &h2)?;
        mark(dev, &mut t0, &mut b.routed_marlin_ms)?;
        let act = ops::swiglu(dev, &gate, &up)?;
        mark(dev, &mut t0, &mut b.elementwise_ms)?;
        let ffn_out = layer.ffn.down.gemv(dev, &act)?;
        mark(dev, &mut t0, &mut b.routed_marlin_ms)?;
        x = ops::add(dev, &x, &ffn_out)?;
        mark(dev, &mut t0, &mut b.elementwise_ms)?;
    }

    let xn = ops::rms_norm(dev, &x, &model.out_norm, cfg.eps)?;
    let logits = model.lm_head.gemv(dev, &xn)?;
    mark(dev, &mut t0, &mut b.lm_head_ms)?;
    Ok((logits, xn))
}

#[allow(clippy::too_many_arguments)]
fn full_attn_layer_census(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    w: &FullAttnWeights,
    slot: &mut FullAttnSlot,
    x: &Tensor,
    pos: u32,
    t0: &mut std::time::Instant,
    b: &mut CensusBuckets,
) -> Result<Tensor> {
    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;
    mark(dev, t0, &mut b.elementwise_ms)?;

    let qg = w.wq.gemv(dev, &h)?;
    mark(dev, t0, &mut b.routed_marlin_ms)?;
    let k = w.wk.gemv(dev, &h)?;
    let v = w.wv.gemv(dev, &h)?;
    mark(dev, t0, &mut b.small_proj_ms)?;

    let q = ops::strided_head_half(dev, &qg, cfg.n_head, cfg.head_dim, 0)?;
    let gate = ops::strided_head_half(dev, &qg, cfg.n_head, cfg.head_dim, 1)?;
    let q = q.reshaped(vec![cfg.n_head, cfg.head_dim]);
    let k = k.reshaped(vec![cfg.n_kv_heads, cfg.head_dim]);
    let v = v.reshaped(vec![cfg.n_kv_heads, cfg.head_dim]);
    let q = ops::rms_norm(dev, &q, &w.q_norm, cfg.eps)?;
    let k = ops::rms_norm(dev, &k, &w.k_norm, cfg.eps)?;
    let q = ops::rope_yarn_partial(dev, &q, pos, cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144)?;
    let k = ops::rope_yarn_partial(dev, &k, pos, cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144)?;
    let posbuf = Tensor::new(dev.upload(&pos.to_le_bytes())?, vec![1], DType::U32);
    let n_kv_used = pos + 1;
    let attn = append_decode_full_kv(dev, cfg, slot, &q, &k, &v, &posbuf, n_kv_used)?;
    let attn_flat = attn.reshaped(vec![cfg.n_head * cfg.head_dim]);
    let gated = ops::sigmoid_mul(dev, &attn_flat, &gate)?;
    mark(dev, t0, &mut b.attention_ms)?;

    let o = w.wo.gemv(dev, &gated)?;
    mark(dev, t0, &mut b.small_proj_ms)?;
    let out = ops::add(dev, x, &o)?;
    mark(dev, t0, &mut b.elementwise_ms)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn gdn_layer_full_census(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    model: &Qwen35Model,
    w: &GdnWeights,
    slot: &mut GdnSlot,
    x: &Tensor,
    pos: u32,
    t0: &mut std::time::Instant,
    b: &mut CensusBuckets,
) -> Result<Tensor> {
    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;
    mark(dev, t0, &mut b.elementwise_ms)?;

    let qkv = w.wqkv.gemv(dev, &h)?;
    mark(dev, t0, &mut b.routed_marlin_ms)?;
    let z = w.wz.gemv(dev, &h)?;
    mark(dev, t0, &mut b.small_proj_ms)?;

    let (conv_out, new_conv_state) = ops::conv1d_causal_silu_many(
        dev, &qkv, &w.conv_w, &model.gdn_conv_zero_bias, &slot.conv_state, 1, cfg.gdn_conv_dim(),
    )?;
    slot.conv_state = new_conv_state;
    let key_dim = cfg.gdn_key_dim();
    let value_dim = cfg.gdn_value_dim();
    let q_raw = ops::slice(dev, &conv_out, 0, key_dim)?.reshaped(vec![cfg.gdn_num_k_heads, cfg.gdn_key_head_dim]);
    let k_raw = ops::slice(dev, &conv_out, key_dim, key_dim)?.reshaped(vec![cfg.gdn_num_k_heads, cfg.gdn_key_head_dim]);
    let v = ops::slice(dev, &conv_out, 2 * key_dim, value_dim)?.reshaped(vec![cfg.gdn_num_v_heads, cfg.gdn_value_head_dim]);
    let a_raw = ops::gemv(dev, &w.w_alpha, &h)?;
    let b_raw = ops::gemv(dev, &w.w_beta, &h)?;
    // F-85 round-11 (fusion round): fused q/k-RMSNorm + gate/beta +
    // gated-delta-net recurrence (`ops::gdn_decode_fused`) + fused
    // rmsNorm(ssm_norm)+silu(z)+mul tail (`ops::gdn_tail_gate`) — see
    // their docs in wh-butter-ops. Both stay inside this function's single
    // `gdn_chain_ms` bucket, same as the unfused chain they replace.
    let y = if model.gdn_cyclic_gqa {
        ops::gdn_decode_fused(
            dev, &q_raw, &k_raw, &v, &model.gdn_q_scale, &model.gdn_k_scale, &a_raw, &b_raw, &w.dt_bias,
            &w.neg_exp_a_log, &slot.state, cfg.eps,
            cfg.gdn_key_head_dim, cfg.gdn_value_head_dim, cfg.gdn_num_v_heads, cfg.gdn_num_k_heads,
        )?
    } else {
        let q = ops::rms_norm(dev, &q_raw, &model.gdn_q_scale, cfg.eps)?;
        let k = ops::rms_norm(dev, &k_raw, &model.gdn_k_scale, cfg.eps)?;
        let (g, beta) = ops::gdn_gate_beta(
            dev,
            &a_raw,
            &b_raw,
            &w.dt_bias,
            &w.neg_exp_a_log,
            cfg.gdn_num_v_heads,
        )?;
        ops::gated_delta_step(
            dev,
            &q,
            &k,
            &v,
            &g,
            &beta,
            &slot.state,
            cfg.gdn_key_head_dim,
            cfg.gdn_value_head_dim,
            cfg.gdn_num_v_heads,
            cfg.gdn_num_k_heads,
        )?
    };
    let gated = ops::gdn_tail_gate(dev, &y, &z, &w.ssm_norm_w, cfg.eps, cfg.gdn_num_v_heads, cfg.gdn_value_head_dim)?;
    mark(dev, t0, &mut b.gdn_chain_ms)?;

    let o = w.wo.gemv(dev, &gated)?;
    mark(dev, t0, &mut b.small_proj_ms)?;
    let out = ops::add(dev, x, &o)?;
    mark(dev, t0, &mut b.elementwise_ms)?;
    Ok(out)
}

/// One GDN (linear-attention) layer's forward. Takes `model` (rather than
/// bundling the shared constant buffers — `gdn_q_scale`/`gdn_k_scale`/
/// `gdn_conv_zero_bias` — onto the per-layer `GdnWeights`, since those
/// three are identical across all 48 GDN layers and don't belong on a
/// loaded-per-layer struct).
#[allow(clippy::too_many_arguments)]
fn gdn_layer_full(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    model: &Qwen35Model,
    w: &GdnWeights,
    slot: &mut GdnSlot,
    x: &Tensor,
    layer_idx: usize,
    pos: u32,
    debug: bool,
) -> Result<Tensor> {
    let dbg = debug && layer_idx == 0;
    let h = ops::rms_norm(dev, x, &w.attn_norm, cfg.eps)?;
    if dbg { dbg_sum(dev, "attn_norm-0", pos, &h); }

    let qkv = w.wqkv.gemv(dev, &h)?;
    if dbg { dbg_sum(dev, "linear_attn_qkv_mixed-0", pos, &qkv); }
    let z = w.wz.gemv(dev, &h)?;

    let (conv_out, new_conv_state) = ops::conv1d_causal_silu_many(
        dev, &qkv, &w.conv_w, &model.gdn_conv_zero_bias, &slot.conv_state, 1, cfg.gdn_conv_dim(),
    )?;
    slot.conv_state = new_conv_state;

    let key_dim = cfg.gdn_key_dim();
    let value_dim = cfg.gdn_value_dim();
    let q_raw = ops::slice(dev, &conv_out, 0, key_dim)?.reshaped(vec![cfg.gdn_num_k_heads, cfg.gdn_key_head_dim]);
    let k_raw = ops::slice(dev, &conv_out, key_dim, key_dim)?.reshaped(vec![cfg.gdn_num_k_heads, cfg.gdn_key_head_dim]);
    let v = ops::slice(dev, &conv_out, 2 * key_dim, value_dim)?.reshaped(vec![cfg.gdn_num_v_heads, cfg.gdn_value_head_dim]);
    if dbg { dbg_sum(dev, "v_conv_predelta-0", pos, &v); }

    let a_raw = ops::gemv(dev, &w.w_alpha, &h)?;
    let b_raw = ops::gemv(dev, &w.w_beta, &h)?;

    // F-85 round-11 (fusion round, DEBUG-ONLY branch): the fused decode
    // kernel below doesn't materialize q/k-normed or gate/beta as separate
    // tensors, so `QWEN35_DEBUG=1`'s per-node oracle dump (layer 0 only)
    // redundantly recomputes them via the OLD unfused ops here. Zero cost
    // when `dbg` is false — this never runs on the hot decode path.
    if dbg {
        let q = ops::rms_norm(dev, &q_raw, &model.gdn_q_scale, cfg.eps)?;
        let k = ops::rms_norm(dev, &k_raw, &model.gdn_k_scale, cfg.eps)?;
        dbg_sum(dev, "q_conv_predelta-0", pos, &q);
        dbg_sum(dev, "k_conv_predelta-0", pos, &k);
        let (g, beta) = ops::gdn_gate_beta(dev, &a_raw, &b_raw, &w.dt_bias, &w.neg_exp_a_log, cfg.gdn_num_v_heads)?;
        // Oracle's `gate-0` is LOG-space (ln(g) = neg_exp_a_log * softplus(...));
        // our `g` is already exp'd, so take ln() host-side for an apples-to-apples sum.
        let n = g.elem_count();
        let mut buf = vec![0u8; n * 4];
        dev.synchronize().ok();
        dev.download(g.buffer.as_ref(), &mut buf).ok();
        let gv: Vec<f32> = buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let ln_sum: f64 = gv.iter().map(|&x| (x as f64).ln()).sum();
        eprintln!("QWEN35_DBG pos={pos} tag=gate-0(ln_g) n={n} sum={ln_sum:.6}");
        dbg_sum(dev, "beta_sigmoid-0", pos, &beta);
    }

    // F-85 round-11: fused q/k-RMSNorm + gate/beta + gated-delta-net
    // recurrence, ONE dispatch (`ops::gdn_decode_fused`) instead of the
    // 5-dispatch chain (2x rms_norm + gdn_gate_beta + 2x gather) above it
    // — see that op's doc for the numerical-match rationale. The CYCLIC
    // GQA mapping this checkpoint requires (`hv_idx % Hk`, verified by
    // oracle checksum diff against the kernel-default contiguous-block
    // mapping — `attn_output` matched to 0.01% only after switching to
    // cyclic, the block mapping was ~20% off) is baked into the fused
    // kernel directly, so `q_raw`/`k_raw` are passed at their native
    // `[Hk,Dk]` shape with no pre-expand gather needed.
    let y = if model.gdn_cyclic_gqa {
        ops::gdn_decode_fused(
            dev, &q_raw, &k_raw, &v, &model.gdn_q_scale, &model.gdn_k_scale, &a_raw, &b_raw, &w.dt_bias,
            &w.neg_exp_a_log, &slot.state, cfg.eps,
            cfg.gdn_key_head_dim, cfg.gdn_value_head_dim, cfg.gdn_num_v_heads, cfg.gdn_num_k_heads,
        )?
    } else {
        let q = ops::rms_norm(dev, &q_raw, &model.gdn_q_scale, cfg.eps)?;
        let k = ops::rms_norm(dev, &k_raw, &model.gdn_k_scale, cfg.eps)?;
        let (g, beta) = ops::gdn_gate_beta(
            dev,
            &a_raw,
            &b_raw,
            &w.dt_bias,
            &w.neg_exp_a_log,
            cfg.gdn_num_v_heads,
        )?;
        ops::gated_delta_step(
            dev,
            &q,
            &k,
            &v,
            &g,
            &beta,
            &slot.state,
            cfg.gdn_key_head_dim,
            cfg.gdn_value_head_dim,
            cfg.gdn_num_v_heads,
            cfg.gdn_num_k_heads,
        )?
    };
    if dbg { dbg_sum(dev, "attn_output-0", pos, &y); }

    // F-85 round-11: fused rmsNorm(ssm_norm) + silu(z) + mul, ONE dispatch
    // (`ops::gdn_tail_gate`) instead of 3 — see that op's doc. NOTE: the
    // pre-mul "node_50" debug tap (the old code's `normed` intermediate)
    // is no longer separately available post-fusion and is dropped; the
    // pre-fusion "attn_output-0" (y) and post-fusion "final_output-0"
    // (gated) taps still bracket it.
    let gated = ops::gdn_tail_gate(dev, &y, &z, &w.ssm_norm_w, cfg.eps, cfg.gdn_num_v_heads, cfg.gdn_value_head_dim)?;
    if dbg { dbg_sum(dev, "final_output-0", pos, &gated); }

    let o = w.wo.gemv(dev, &gated)?;
    if dbg { dbg_sum(dev, "linear_attn_out-0", pos, &o); }
    ops::add(dev, x, &o)
}

// ─────────────────────────────────────────────────────────────────────────
// MTP self-speculative decode
// ─────────────────────────────────────────────────────────────────────────

/// MTP's own attention KV cache. Architecturally identical to a backbone
/// full-attention layer's cache (see `FullAttnSlot`) — the MTP block tracks
/// the SAME absolute position axis as the backbone (see `mtp_draft_step`'s
/// doc), just with its own weights/K/V. Reusing `FullAttnSlot` directly
/// (rather than a parallel type) so `mtp_draft_step` can reuse
/// `full_attn_layer`'s exact kv_append/sdpa_decode call pattern.
pub struct MtpCache {
    slot: FullAttnSlot,
}

impl MtpCache {
    pub fn new(dev: &dyn Device, cfg: &Qwen35Config, max_seq: usize) -> Result<Self> {
        let n = cfg.n_kv_heads * max_seq * cfg.head_dim;
        let k = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, max_seq, cfg.head_dim], DType::F32);
        let v = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, max_seq, cfg.head_dim], DType::F32);
        Ok(MtpCache { slot: FullAttnSlot { kv: FullKvCache::Dense { k, v }, cap: max_seq } })
    }
}

/// One MTP draft step — see `MtpWeights`'s module doc for the full
/// architectural contract. `h` = seed hidden (a "final normed hidden": the
/// backbone's own `xn` from [`decode_step_ex`] for the FIRST step of a
/// speculative round, or this function's own returned `h_nextn` when
/// chaining γ>1 draft steps). `tok` = seed token (its embedding is mixed
/// in via `eh_proj`) — the just-accepted/confirmed token for the first
/// step, or the previous step's own draft token when chaining. `pos` = the
/// absolute sequence position `tok` occupies; MTP's RoPE + KV-cache write
/// use this position directly (MTP shares the backbone's position axis —
/// confirmed against `llama.cpp`'s `graph_mtp`, which threads the SAME
/// `inp_pos` the backbone's own decode call used). Returns
/// `(draft_token, h_nextn)`: `draft_token` predicts the token at `pos+1`;
/// `h_nextn` is this step's own "final normed hidden", ready to chain
/// directly into the next call's `h`.
fn mtp_draft_step(
    dev: &dyn Device,
    cfg: &Qwen35Config,
    model: &Qwen35Model,
    w: &MtpWeights,
    slot: &mut FullAttnSlot,
    h: &Tensor,
    tok: u32,
    pos: u32,
) -> Result<(u32, Tensor)> {
    let tok_buf = Tensor::new(dev.upload(&tok.to_le_bytes())?, vec![1], DType::U32);
    let e = gather_tokens_f32(dev, model, &tok_buf)?.reshaped(vec![cfg.hidden]);
    let e_norm = ops::rms_norm(dev, &e, &w.enorm, cfg.eps)?;
    let h_norm = ops::rms_norm(dev, h, &w.hnorm, cfg.eps)?;
    // eh_proj(concat(e_norm, h_norm)) == eh_proj_e·e_norm + eh_proj_h·h_norm
    // — see `MtpWeights::eh_proj_e`'s doc for why this split form is used.
    let xe = w.eh_proj_e.gemv(dev, &e_norm)?;
    let xh = w.eh_proj_h.gemv(dev, &h_norm)?;
    let x = ops::add(dev, &xe, &xh)?;
    let inp_sa = x.clone();

    // One full-attention decoder layer on `x` (own weights/KV), same shape
    // as `full_attn_layer`'s trunk full-attn layers.
    let hn = ops::rms_norm(dev, &x, &w.attn_norm, cfg.eps)?;
    let qg = w.wq.gemv(dev, &hn)?;
    let k = w.wk.gemv(dev, &hn)?;
    let v = w.wv.gemv(dev, &hn)?;
    let q = ops::strided_head_half(dev, &qg, cfg.n_head, cfg.head_dim, 0)?;
    let gate = ops::strided_head_half(dev, &qg, cfg.n_head, cfg.head_dim, 1)?;
    let q = q.reshaped(vec![cfg.n_head, cfg.head_dim]);
    let k = k.reshaped(vec![cfg.n_kv_heads, cfg.head_dim]);
    let v = v.reshaped(vec![cfg.n_kv_heads, cfg.head_dim]);
    let q = ops::rms_norm(dev, &q, &w.q_norm, cfg.eps)?;
    let k = ops::rms_norm(dev, &k, &w.k_norm, cfg.eps)?;
    let q = ops::rope_yarn_partial(dev, &q, pos, cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144)?;
    let k = ops::rope_yarn_partial(dev, &k, pos, cfg.rope_n_rot, cfg.rope_theta, 1.0, 0.0, 1.0, 32.0, 1.0, 262144)?;

    let posbuf = Tensor::new(dev.upload(&pos.to_le_bytes())?, vec![1], DType::U32);
    let n_kv_used = pos + 1;
    let attn = append_decode_full_kv(dev, cfg, slot, &q, &k, &v, &posbuf, n_kv_used)?;
    let attn_flat = attn.reshaped(vec![cfg.n_head * cfg.head_dim]);
    let gated = ops::sigmoid_mul(dev, &attn_flat, &gate)?;
    let o = w.wo.gemv(dev, &gated)?;
    let cur = ops::add(dev, &o, &inp_sa)?;

    let ffn_residual = cur.clone();
    let post = ops::rms_norm(dev, &cur, &w.ffn_norm, cfg.eps)?;
    let g2 = w.ffn_gate.gemv(dev, &post)?;
    let u2 = w.ffn_up.gemv(dev, &post)?;
    let act = ops::swiglu(dev, &g2, &u2)?;
    let ffn_out = w.ffn_down.gemv(dev, &act)?;
    let cur = ops::add(dev, &ffn_out, &ffn_residual)?;

    let h_nextn = ops::rms_norm(dev, &cur, &w.shared_head_norm, cfg.eps)?;
    // Shared head: MTP has no `nextn.shared_head_head` (see `MtpWeights`'s
    // doc), so this reuses `model.lm_head` directly.
    let logits = model.lm_head.gemv(dev, &h_nextn)?;
    let am = ops::argmax_f32_device(dev, &logits)?;
    let mut ib = [0u8; 4];
    dev.synchronize()?;
    dev.download(am.buffer.as_ref(), &mut ib)?;
    let draft_token = u32::from_le_bytes(ib);
    Ok((draft_token, h_nextn))
}

/// Combined backbone + MTP cache for speculative decode.
pub struct Qwen35SpecCache {
    pub backbone: Qwen35Cache,
    mtp: MtpCache,
}

impl Qwen35SpecCache {
    pub fn new(dev: &dyn Device, cfg: &Qwen35Config, max_seq: usize) -> Result<Self> {
        Ok(Qwen35SpecCache { backbone: Qwen35Cache::new(dev, cfg, max_seq)?, mtp: MtpCache::new(dev, cfg, max_seq)? })
    }

    /// Runs one MTP forward at `(xn, token_id, pos)` purely to populate
    /// MTP's own KV cache and discards the draft prediction. Needed during
    /// PROMPT PREFILL (before any speculative round runs): `speculative_round`
    /// only ever calls the drafter for genuinely speculative continuations,
    /// so without this, MTP's attention would start every generation with
    /// an empty KV cache — no context from the prompt at all, tanking
    /// acceptance rate for reasons that have nothing to do with the
    /// drafter's own correctness. Mirrors the invariant `speculative_round`
    /// relies on for accepted-but-not-drafted tokens too (see its doc):
    /// every confirmed token gets exactly one MTP forward, seeded from the
    /// backbone's own `xn` at that position, keyed at that position.
    pub fn warm_mtp(&mut self, dev: &dyn Device, model: &Qwen35Model, xn: &Tensor, token_id: u32, pos: u32) -> Result<()> {
        let Some(mtp) = &model.mtp else {
            return Err(Error::Msg("warm_mtp: model has no MTP weights (non-MTP checkpoint)".into()));
        };
        mtp_draft_step(dev, &model.cfg, model, mtp, &mut self.mtp.slot, xn, token_id, pos)?;
        Ok(())
    }
}

/// Per-round speculative-decode bookkeeping (for the γ ladder / acceptance-
/// rate measurement — see the perf-campaign report, not correctness).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecDecodeStats {
    pub n_drafted: usize,
    pub n_accepted: usize,
}

/// One speculative-decode round: draft up to `gamma` tokens from
/// `(last_token, last_pos, last_hidden)` via the MTP sidecar, then verify
/// against the backbone and return the run of tokens that are CONFIRMED
/// correct (i.e. bit-identical to what plain greedy `decode_step` would
/// have produced one token at a time).
///
/// ## Verify strategy: sequential ("lazy"), not batched
///
/// This calls [`decode_step_ex`] (the ordinary, unmodified backbone path)
/// **at most once per drafted candidate, and never for a candidate past the
/// first mismatch**: the loop compares the backbone's own greedy argmax
/// (computed from the PREVIOUS confirmed step's logits, cheap — just a
/// `lm_head.gemv` off already-computed hidden, no new trunk pass) against
/// the next draft token BEFORE running the trunk on it. Only once a draft
/// matches does its trunk pass actually execute (to advance KV/GDN state
/// and get the FOLLOWING step's logits for the next comparison).
///
/// Consequence: **no rollback machinery is needed.** Full-attention KV
/// (both the backbone's and MTP's own) is safe under the trunk's existing
/// position-indexed overwrite + explicit `n_kv_used` bound — a position
/// speculatively written during the draft phase but not reached by the
/// (lazy) verify phase is simply overwritten next round before anything
/// ever reads it. GDN's recurrent state is only ever advanced (via
/// `decode_step_ex`, which mutates it in place) for a token that has
/// ALREADY been confirmed correct at the point of the call — the sequence
/// of `decode_step_ex` calls this function makes is therefore exactly the
/// sequence plain (non-speculative) decode would have made, just
/// interleaved with cheap drafter calls. This is why the correctness bar
/// (bit-identical vs spec-decode OFF) holds by construction here.
///
/// ## What this does NOT do: save backbone bandwidth
///
/// Every ACCEPTED token still costs one full backbone forward pass (same
/// as plain decode) — this form saves nothing on the dominant GEMV cost,
/// and actively COSTS MORE than plain decode (drafter overhead + the
/// sequential argmax/download round-trips), so it will not beat the 14
/// tok/s bar. Its purpose is to validate the drafter (weight loading,
/// `eh_proj` split-GEMV algebra, KV-cache/position bookkeeping) and to
/// measure acceptance rate — the number that determines whether the
/// bandwidth-saving batched-verify path (chunked GDN + small-M
/// `gemm_q4_mpp` projections, needing real GDN-state snapshot/restore
/// rollback since it commits state for γ+1 positions before knowing the
/// accept count) is worth building out.
pub fn speculative_round(
    dev: &dyn Device,
    model: &Qwen35Model,
    cache: &mut Qwen35SpecCache,
    last_token: u32,
    last_pos: u32,
    last_hidden: &Tensor,
    gamma: usize,
) -> Result<(Vec<u32>, u32, u32, Tensor, SpecDecodeStats)> {
    let Some(mtp) = &model.mtp else {
        return Err(Error::Msg("speculative_round: model has no MTP weights (non-MTP checkpoint)".into()));
    };
    let cfg = &model.cfg;

    // ---- Draft phase: chain `gamma` MTP steps off the confirmed seed. ----
    let mut drafts: Vec<u32> = Vec::with_capacity(gamma);
    let mut dh = last_hidden.clone();
    let mut dtok = last_token;
    let mut dpos = last_pos;
    for _ in 0..gamma {
        let (t, h_next) = mtp_draft_step(dev, cfg, model, mtp, &mut cache.mtp.slot, &dh, dtok, dpos)?;
        drafts.push(t);
        dh = h_next;
        dtok = t;
        dpos += 1;
    }

    // ---- Verify phase: lazy sequential accept/reject (see doc above). ----
    // `logits` here is the ALREADY-COMPUTED backbone prediction for
    // `last_pos+1` (== `lm_head(last_hidden)`), reusing the hidden the
    // caller already paid for — no redundant trunk pass.
    let mut logits = model.lm_head.gemv(dev, last_hidden)?;
    let mut cur_hidden = last_hidden.clone();
    let mut cur_tok = last_token;
    let mut cur_pos = last_pos;
    let mut n_accept = 0usize;
    let mut accepted: Vec<u32> = Vec::with_capacity(gamma);

    for &d in &drafts {
        let am = ops::argmax_f32_device(dev, &logits)?;
        let mut ib = [0u8; 4];
        dev.synchronize()?;
        dev.download(am.buffer.as_ref(), &mut ib)?;
        let true_next = u32::from_le_bytes(ib);

        if true_next != d {
            // Misprediction: `true_next` (not `d`) is the real next token.
            // It was never drafted, so nothing needs to be rolled back —
            // just run the trunk on it once to seed the NEXT round, then
            // stop (the rest of `drafts` is discarded).
            let (logits2, xn2) = decode_step_ex(dev, model, &mut cache.backbone, true_next, cur_pos + 1)?;
            accepted.push(true_next);
            cur_pos += 1;
            cur_tok = true_next;
            cur_hidden = xn2;
            let _ = logits2; // not needed past this point; next round recomputes via lm_head(cur_hidden)
            return Ok((accepted, cur_tok, cur_pos, cur_hidden, SpecDecodeStats { n_drafted: drafts.len(), n_accepted: n_accept }));
        }

        // Accepted: `d` matches the backbone's own greedy choice. Advance
        // the trunk through it to get the next comparison's logits/hidden.
        accepted.push(d);
        cur_pos += 1;
        cur_tok = d;
        n_accept += 1;
        let (logits2, xn2) = decode_step_ex(dev, model, &mut cache.backbone, cur_tok, cur_pos)?;
        logits = logits2;
        cur_hidden = xn2;
    }

    // All `gamma` drafts accepted — `cur_hidden`/`logits` already hold the
    // confirmed state at `cur_pos`, ready to seed the next round directly.
    let _ = logits;
    Ok((accepted, cur_tok, cur_pos, cur_hidden, SpecDecodeStats { n_drafted: drafts.len(), n_accepted: n_accept }))
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn hf_multimodal_wrapper_reads_text_geometry() {
        let root = std::env::temp_dir().join(format!("butter-qwen35-config-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "max_position_embeddings": 1_000_000,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "head_dim": 256,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {
                    "rope_theta": 10_000_000.0,
                    "original_max_position_embeddings": 262144
                },
                "full_attention_interval": 4,
                "rms_norm_eps": 0.000001,
                "vocab_size": 248320,
                "intermediate_size": 17408,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4
            }
        });
        std::fs::write(root.join("config.json"), config.to_string()).unwrap();
        let cfg = Qwen35Config::from_hf_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(cfg.max_seq, 262_144);
        assert_eq!((cfg.n_layers, cfg.hidden, cfg.ffn_inter), (64, 5120, 17408));
        assert_eq!((cfg.n_head, cfg.n_kv_heads, cfg.head_dim, cfg.rope_n_rot), (24, 4, 256, 64));
        assert_eq!((cfg.gdn_num_k_heads, cfg.gdn_num_v_heads), (16, 48));
        std::fs::remove_dir_all(root).unwrap();
    }
}
