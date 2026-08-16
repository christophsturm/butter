// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Real generation entrypoint: GGUF `general.architecture` dispatch ->
//! load -> batched chunked prefill (F-85) -> per-token decode loop.
//!
//! This lives in the engine crate (not `wh-butter-cli`) because `wh-butter`
//! already depends on `wh-butter-models`/`wh-butter-runtime`/
//! `wh-butter-loader` unconditionally (see this crate's `Cargo.toml`), so
//! the CLI, the FFI surface, and any future host can all call the same
//! `load`/`generate` without each re-deriving the load->prefill->decode
//! wiring.
//!
//! `load` and `generate` are deliberately SEPARATE calls (not one
//! `run(path, prompt)`): loading a real checkpoint dominates wall time
//! (~2min for the 27B Q4_K_M GGUF used in this port) and is completely
//! independent of the prompt, so a caller that wants to warm up CUDA's
//! NVRTC/JIT kernel cache with a throwaway generate before a timed one (see
//! `wh-butter-cli`'s `--warmup`) must be able to reuse the SAME loaded
//! weights for both calls rather than paying the load cost twice.
//! [`run`] is a thin load+generate convenience wrapper for callers that
//! only need one shot.
//!
//! Architecture registry: [`LoadedModel`] gets a new variant, `load`'s
//! `match` gets a new arm, and `generate`'s `match` gets the matching arm
//! per newly-ported model (the `qwen35` arm below is the reference shape:
//! tokenizer from GGUF metadata, `<arch>::load_gguf` at load time,
//! `<arch>::Cache::new` + `<arch>::prefill` for the prompt + `<arch>::
//! decode_step` per generated token at generate time).

use std::io::Write;
use std::time::Instant;

use wh_butter_core::{Device, Error, Result, Tensor};
use wh_butter_loader::gguf::Gguf;
use wh_butter_models::{gguf_tokenizer::GgufTokenizer, qwen35};
use wh_butter_runtime::{sample, Rng};
pub use wh_butter_runtime::Sampling;

/// Options for one [`generate`] call. `chunk == 0` is the "auto" sentinel
/// each model's own `prefill` already understands (see e.g.
/// `qwen35::default_prefill_chunk` / `laguna::default_prefill_chunk`) —
/// passed straight through rather than re-picking a default here, so this
/// module can't drift from the bench-validated per-model operating point.
#[derive(Debug, Clone)]
pub struct GenerateOpts {
    pub max_new: usize,
    pub chunk: usize,
    pub sampling: Sampling,
    pub seed: u64,
    /// Print each decoded token to stdout as it's produced.
    pub stream: bool,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        GenerateOpts { max_new: 128, chunk: 0, sampling: Sampling::Greedy, seed: 0, stream: true }
    }
}

/// Prompt-processing (prefill) and decode throughput + generated text for
/// one [`generate`] call. Prefill/decode are timed and reported
/// SEPARATELY (the CLI's UX-numbers requirement) since they run through
/// entirely different paths (batched chunked forward vs. per-token
/// `decode_step`) with very different tok/s.
#[derive(Debug, Clone)]
pub struct GenerateReport {
    pub architecture: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub token_ids: Vec<u32>,
    pub text: String,
}

/// A GGUF checkpoint loaded onto a [`Device`], dispatched by
/// `general.architecture`. Holds everything [`generate`] needs
/// (tokenizer, eos id, model weights) but no per-call state (no KV
/// cache/positions) — a single `LoadedModel` can drive any number of
/// independent [`generate`] calls (each builds its own fresh cache).
pub enum LoadedModel {
    Qwen35 { tok: GgufTokenizer, eos: Option<u32>, model: qwen35::Qwen35Model },
}

impl LoadedModel {
    pub fn architecture(&self) -> &'static str {
        match self {
            LoadedModel::Qwen35 { .. } => "qwen35",
        }
    }
}

/// GGUF architecture string that selects the generation path — read
/// straight from `general.architecture` metadata, same key `llama.rs`'s
/// `load_hf` already keys its own dispatch off of.
pub fn detect_architecture(g: &Gguf) -> String {
    g.meta_str("general.architecture").unwrap_or("unknown").to_string()
}

/// Open `gguf_path`, detect its architecture, and load weights onto `dev`.
/// The dominant-cost, prompt-independent half of "load a GGUF, generate
/// text" — call once, then drive as many [`generate`] calls as you like.
pub fn load(dev: &dyn Device, gguf_path: &str) -> Result<LoadedModel> {
    let g = Gguf::open(gguf_path)?;
    let arch = detect_architecture(&g);
    match arch.as_str() {
        "qwen35" => {
            let tok = GgufTokenizer::from_gguf(&g)?;
            let eos = g.meta_u32("tokenizer.ggml.eos_token_id");
            let model = qwen35::load_gguf(dev, &g)?;
            Ok(LoadedModel::Qwen35 { tok, eos, model })
        }
        other => Err(Error::Msg(format!(
            "butter generate: no registered generation path for GGUF architecture {other:?} \
             (registered: \"qwen35\"). To add one: teach `wh_butter::generate::{{load,generate}}` \
             a new match arm following the `qwen35` arm's shape."
        ))),
    }
}

/// Batched chunked prefill (F-85) over `prompt`, then per-token
/// `decode_step` for up to `opts.max_new` tokens. This is the "real
/// generation entrypoint" — prompt processing routes through the same
/// `<arch>::prefill` the bench/correctness harnesses validated
/// (token-identical to the old decode-loop-as-prefill baseline, see
/// `qwen35_prefill_correctness.rs`), not a decode-loop re-implementation.
pub fn generate(dev: &dyn Device, loaded: &LoadedModel, prompt: &str, opts: &GenerateOpts) -> Result<GenerateReport> {
    match loaded {
        LoadedModel::Qwen35 { tok, eos, model } => generate_qwen35(dev, tok, *eos, model, prompt, opts),
    }
}

/// Convenience one-shot wrapper: [`load`] + [`generate`]. Fine for a
/// single request; a caller issuing more than one (e.g. a warmup pass
/// before a timed bench run) should call `load` once and reuse the
/// `LoadedModel` instead, to avoid paying the load cost per call.
pub fn run(dev: &dyn Device, gguf_path: &str, prompt: &str, opts: &GenerateOpts) -> Result<GenerateReport> {
    let loaded = load(dev, gguf_path)?;
    generate(dev, &loaded, prompt, opts)
}

fn generate_qwen35(
    dev: &dyn Device,
    tok: &GgufTokenizer,
    eos: Option<u32>,
    model: &qwen35::Qwen35Model,
    prompt: &str,
    opts: &GenerateOpts,
) -> Result<GenerateReport> {
    let ids = tok.encode(prompt);
    if ids.is_empty() {
        return Err(Error::Msg("generate: empty prompt after tokenization".into()));
    }
    let max_seq = ids.len() + opts.max_new + 8;
    let mut cache = qwen35::Qwen35Cache::new(dev, &model.cfg, max_seq)?;

    // ---- Prompt processing: batched chunked prefill (F-85) ----
    let t_prefill = Instant::now();
    let logits = qwen35::prefill(dev, model, &mut cache, &ids, opts.chunk)?;
    dev.synchronize()?;
    let prefill_secs = t_prefill.elapsed().as_secs_f64();

    let mut last_logits = download_logits(dev, &logits, model.cfg.vocab)?;
    let mut pos = ids.len() as u32;

    // ---- Decode: the existing per-token decode_step path, unchanged ----
    let mut rng = Rng::new(opts.seed);
    let mut out_ids = Vec::with_capacity(opts.max_new);
    let t_decode = Instant::now();
    for _ in 0..opts.max_new {
        let next = sample(&last_logits, &opts.sampling, &mut rng) as u32;
        if Some(next) == eos {
            break;
        }
        out_ids.push(next);
        if opts.stream {
            print!("{}", tok.decode(&[next]));
            let _ = std::io::stdout().flush();
        }
        let logits = qwen35::decode_step(dev, model, &mut cache, next, pos)?;
        last_logits = download_logits(dev, &logits, model.cfg.vocab)?;
        pos += 1;
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();
    if opts.stream {
        println!();
    }

    let generated_tokens = out_ids.len();
    let text = tok.decode(&out_ids);
    Ok(GenerateReport {
        architecture: "qwen35".to_string(),
        prompt_tokens: ids.len(),
        generated_tokens,
        prefill_secs,
        decode_secs,
        prefill_tok_s: ids.len() as f64 / prefill_secs,
        decode_tok_s: if decode_secs > 0.0 { generated_tokens as f64 / decode_secs } else { 0.0 },
        token_ids: out_ids,
        text,
    })
}

fn download_logits(dev: &dyn Device, t: &Tensor, vocab: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; vocab * 4];
    dev.synchronize()?;
    dev.download(t.buffer.as_ref(), &mut b)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}
