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
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use wh_butter_core::{Device, Error, Result, Tensor};
use wh_butter_loader::gguf::Gguf;
use wh_butter_models::{gguf_tokenizer::GgufTokenizer, qwen35};
pub use wh_butter_runtime::Sampling;
use wh_butter_runtime::{Rng, sample};

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
    /// Cap KV allocation (`--max-context`). `None` sizes to
    /// `prompt + max_new + 8` as before.
    pub max_context: Option<usize>,
    /// When set and true, prefill/decode stops. Serve sets this on a
    /// client close so a dropped socket does not finish prefill.
    pub abort: Option<Arc<AtomicBool>>,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        GenerateOpts {
            max_new: 128,
            chunk: 0,
            sampling: Sampling::Greedy,
            seed: 0,
            stream: true,
            max_context: None,
            abort: None,
        }
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
    pub cached_tokens: usize,
}

/// A GGUF checkpoint loaded onto a [`Device`], dispatched by
/// `general.architecture`. Holds everything [`generate`] needs
/// (tokenizer, eos id, model weights) but no per-call state (no KV
/// cache/positions) — a single `LoadedModel` can drive any number of
/// independent [`generate`] calls (each builds its own fresh cache).
pub enum LoadedModel {
    Qwen35 {
        tok: GgufTokenizer,
        eos: Option<u32>,
        model: qwen35::Qwen35Model,
    },
}

impl LoadedModel {
    pub fn architecture(&self) -> &'static str {
        match self {
            LoadedModel::Qwen35 { .. } => "qwen35",
        }
    }

    pub fn max_model_len(&self) -> usize {
        match self {
            LoadedModel::Qwen35 { model, .. } => model.cfg.max_seq,
        }
    }
}

/// GGUF architecture string that selects the generation path — read
/// straight from `general.architecture` metadata, same key `llama.rs`'s
/// `load_hf` already keys its own dispatch off of.
pub fn detect_architecture(g: &Gguf) -> String {
    g.meta_str("general.architecture")
        .unwrap_or("unknown")
        .to_string()
}

/// Open `gguf_path`, detect its architecture, and load weights onto `dev`.
/// The dominant-cost, prompt-independent half of "load a GGUF, generate
/// text" — call once, then drive as many [`generate`] calls as you like.
pub fn load(dev: &dyn Device, gguf_path: &str) -> Result<LoadedModel> {
    if std::path::Path::new(gguf_path).is_dir() {
        let tok = GgufTokenizer::from_hf_dir(gguf_path)?;
        let eos = read_hf_eos(gguf_path)?;
        let model = qwen35::load_nvfp4_dir(dev, gguf_path)?;
        return Ok(LoadedModel::Qwen35 { tok, eos, model });
    }
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

fn read_hf_eos(dir: &str) -> Result<Option<u32>> {
    for file in ["generation_config.json", "tokenizer_config.json"] {
        let path = std::path::Path::new(dir).join(file);
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let json: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Msg(format!("parse {}: {e}", path.display())))?;
        let value = json.get("eos_token_id");
        let eos = value
            .and_then(|value| value.as_u64())
            .or_else(|| {
                value
                    .and_then(|value| value.as_array())
                    .and_then(|values| values.first())
                    .and_then(|value| value.as_u64())
            })
            .map(|value| value as u32);
        if eos.is_some() {
            return Ok(eos);
        }
    }
    Ok(None)
}

/// Batched chunked prefill (F-85) over `prompt`, then per-token
/// `decode_step` for up to `opts.max_new` tokens. This is the "real
/// generation entrypoint" — prompt processing routes through the same
/// `<arch>::prefill` the bench/correctness harnesses validated
/// (token-identical to the old decode-loop-as-prefill baseline, see
/// `qwen35_prefill_correctness.rs`), not a decode-loop re-implementation.
pub fn generate(
    dev: &dyn Device,
    loaded: &LoadedModel,
    prompt: &str,
    opts: &GenerateOpts,
) -> Result<GenerateReport> {
    let report = generate_with_sink(dev, loaded, prompt, opts, |text, _id| {
        if opts.stream {
            print!("{text}");
            let _ = std::io::stdout().flush();
        }
        true
    })?;
    if opts.stream {
        println!();
    }
    Ok(report)
}

/// Like [`generate`], but each decoded token is handed to `on_token`.
/// Return `false` from the callback to abort decode (client gone).
pub fn generate_with_sink<F>(
    dev: &dyn Device,
    loaded: &LoadedModel,
    prompt: &str,
    opts: &GenerateOpts,
    on_token: F,
) -> Result<GenerateReport>
where
    F: FnMut(&str, u32) -> bool,
{
    match loaded {
        LoadedModel::Qwen35 { tok, eos, model } => {
            let mut live = None;
            generate_qwen35(dev, tok, *eos, model, prompt, opts, &mut live, on_token)
        }
    }
}

/// Live KV / GDN cache carried across serve requests at the completed prompt
/// boundary. Decode may write later full-attention KV positions, but recurrent
/// state is restored before this value is published again.
pub struct LiveGenerateCache {
    pub cache: qwen35::Qwen35Cache,
    pub tokens: Vec<u32>,
    prompt_logits: Vec<f32>,
}

/// Longest common prefix length of two token sequences.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Prefix length safe to keep for hybrid Qwen3.5. If the new prompt is
/// fully covered by the cache (`n == new.len()`), decrement by 1. That
/// case is a rewind (need at least one new token to prefill) and
/// recurrent GDN/conv state cannot rewind.
pub fn n_past(cached: &[u32], new: &[u32]) -> usize {
    if new.is_empty() {
        return 0;
    }
    let mut n = common_prefix(cached, new);
    if n == new.len() {
        n -= 1;
    }
    n
}

/// Pure append-vs-rebuild decision for a live Qwen3.5 cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReusePlan {
    Exact,
    Append { n_past: usize },
    Rebuild,
}

/// An exact prompt reuses its saved prompt logits. Suffix-only prefill is
/// valid when `new` strictly appends to `cached`, capacity fits `need`, and
/// neither side is empty. Any rewind, fork, or undersize rebuilds.
pub fn reuse_plan(cached: &[u32], new: &[u32], cap: usize, need: usize) -> ReusePlan {
    if cached.is_empty() || new.is_empty() || cap < need {
        return ReusePlan::Rebuild;
    }
    if cached == new {
        return ReusePlan::Exact;
    }
    let n = common_prefix(cached, new);
    if n == cached.len() {
        ReusePlan::Append { n_past: n }
    } else {
        ReusePlan::Rebuild
    }
}

/// Like [`generate_with_sink`], but reuses `live` across calls when the
/// new prompt strictly appends to the previous prompt+generated tokens.
/// Rewinds rebuild because hybrid recurrent state cannot rewind.
pub fn generate_with_sink_cached<F>(
    dev: &dyn Device,
    loaded: &LoadedModel,
    prompt: &str,
    opts: &GenerateOpts,
    live: &mut Option<LiveGenerateCache>,
    on_token: F,
) -> Result<GenerateReport>
where
    F: FnMut(&str, u32) -> bool,
{
    match loaded {
        LoadedModel::Qwen35 { tok, eos, model } => {
            generate_qwen35(dev, tok, *eos, model, prompt, opts, live, on_token)
        }
    }
}

/// Convenience one-shot wrapper: [`load`] + [`generate`]. Fine for a
/// single request; a caller issuing more than one (e.g. a warmup pass
/// before a timed bench run) should call `load` once and reuse the
/// `LoadedModel` instead, to avoid paying the load cost per call.
pub fn run(
    dev: &dyn Device,
    gguf_path: &str,
    prompt: &str,
    opts: &GenerateOpts,
) -> Result<GenerateReport> {
    let loaded = load(dev, gguf_path)?;
    generate(dev, &loaded, prompt, opts)
}

fn generate_qwen35<F>(
    dev: &dyn Device,
    tok: &GgufTokenizer,
    eos: Option<u32>,
    model: &qwen35::Qwen35Model,
    prompt: &str,
    opts: &GenerateOpts,
    live: &mut Option<LiveGenerateCache>,
    mut on_token: F,
) -> Result<GenerateReport>
where
    F: FnMut(&str, u32) -> bool,
{
    let ids = tok.encode(prompt);
    if ids.is_empty() {
        return Err(Error::Msg(
            "generate: empty prompt after tokenization".into(),
        ));
    }
    let need = ids.len() + opts.max_new + 8;
    let cap = match opts.max_context {
        Some(m) if m > 0 => m,
        _ => need,
    };
    if cap > model.cfg.max_seq {
        return Err(Error::Msg(format!(
            "generate: requested context {cap} exceeds the wired model limit {}",
            model.cfg.max_seq
        )));
    }
    if cap < need {
        return Err(Error::Msg(format!(
            "generate: prompt+max_new ({need}) exceeds --max-context ({cap})"
        )));
    }
    if aborted(opts) {
        return Err(Error::Msg("generate: aborted".into()));
    }
    let plan = match live.as_ref() {
        Some(l) => reuse_plan(&l.tokens, &ids, l.cache.cap(), need),
        None => ReusePlan::Rebuild,
    };
    let cached_tokens = match plan {
        ReusePlan::Exact => ids.len(),
        ReusePlan::Append { n_past } => n_past,
        ReusePlan::Rebuild => 0,
    };

    let (mut cache, cached_prompt_logits) = match plan {
        ReusePlan::Exact | ReusePlan::Append { .. } => {
            let cached = live.take().expect("cache reuse plan implies live cache");
            (cached.cache, Some(cached.prompt_logits))
        }
        ReusePlan::Rebuild => {
            let _ = live.take();
            (qwen35::Qwen35Cache::new(dev, &model.cfg, cap)?, None)
        }
    };

    // ---- Prompt processing: batched chunked prefill (F-85) ----
    let t_prefill = Instant::now();
    let mut last_logits = match plan {
        ReusePlan::Exact => cached_prompt_logits.unwrap_or_default(),
        ReusePlan::Append { n_past } => qwen35::prefill_at(
            dev,
            model,
            &mut cache,
            &ids[n_past..],
            n_past,
            opts.chunk,
            opts.abort.as_deref(),
        )
        .and_then(|logits| download_logits(dev, &logits, model.cfg.vocab))?,
        ReusePlan::Rebuild => qwen35::prefill_at(
            dev,
            model,
            &mut cache,
            &ids,
            0,
            opts.chunk,
            opts.abort.as_deref(),
        )
        .and_then(|logits| download_logits(dev, &logits, model.cfg.vocab))?,
    };
    if matches!(plan, ReusePlan::Exact) && last_logits.is_empty() {
        return Err(Error::Msg(
            "generate: exact cache hit missing prompt logits".into(),
        ));
    }
    dev.synchronize()?;
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    let prompt_logits = last_logits.clone();
    let prompt_snapshot = cache.snapshot_prompt(dev)?;
    let mut pos = ids.len() as u32;

    // ---- Decode: the existing per-token decode_step path, unchanged ----
    let mut rng = Rng::new(opts.seed);
    let mut out_ids = Vec::with_capacity(opts.max_new);
    let t_decode = Instant::now();
    for _ in 0..opts.max_new {
        if aborted(opts) {
            break;
        }
        let next = sample(&last_logits, &opts.sampling, &mut rng) as u32;
        if Some(next) == eos {
            break;
        }
        out_ids.push(next);
        let piece = tok.decode(&[next]);
        if !on_token(&piece, next) {
            break;
        }
        let logits = qwen35::decode_step(dev, model, &mut cache, next, pos)?;
        last_logits = download_logits(dev, &logits, model.cfg.vocab)?;
        pos += 1;
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();

    let generated_tokens = out_ids.len();
    let text = tok.decode(&out_ids);
    cache.restore_prompt(prompt_snapshot)?;
    *live = Some(LiveGenerateCache {
        cache,
        tokens: ids.clone(),
        prompt_logits,
    });

    Ok(GenerateReport {
        architecture: "qwen35".to_string(),
        prompt_tokens: ids.len(),
        generated_tokens,
        prefill_secs,
        decode_secs,
        prefill_tok_s: ids.len() as f64 / prefill_secs,
        decode_tok_s: if decode_secs > 0.0 {
            generated_tokens as f64 / decode_secs
        } else {
            0.0
        },
        token_ids: out_ids,
        text,
        cached_tokens,
    })
}

pub fn aborted(opts: &GenerateOpts) -> bool {
    opts.abort
        .as_ref()
        .map(|a| a.load(Ordering::Relaxed))
        .unwrap_or(false)
}

/// Set abort when the peer closes `stream`. Serve polls this during
/// `prefill_at` so a TCP drop stops leftover chunks.
pub fn watch_disconnect(stream: &TcpStream) -> Arc<AtomicBool> {
    let abort = Arc::new(AtomicBool::new(false));
    if let Ok(peek) = stream.try_clone() {
        let abort2 = abort.clone();
        std::thread::spawn(move || {
            let _ = peek.set_nonblocking(true);
            let mut buf = [0u8; 1];
            while !abort2.load(Ordering::Relaxed) {
                match peek.peek(&mut buf) {
                    Ok(0) => {
                        abort2.store(true, Ordering::Relaxed);
                        break;
                    }
                    Ok(_) => {}
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            || e.kind() == std::io::ErrorKind::Interrupted
                            || e.kind() == std::io::ErrorKind::TimedOut => {}
                    Err(_) => {
                        abort2.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
    }
    abort
}

fn download_logits(dev: &dyn Device, t: &Tensor, vocab: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; vocab * 4];
    dev.synchronize()?;
    dev.download(t.buffer.as_ref(), &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Product: exact prompts reuse saved logits; append only on a strict
    /// extension that fits. Shorter prompts and forks rebuild.
    #[test]
    fn reuse_plan_table() {
        assert_eq!(
            reuse_plan(&[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6], 64, 16),
            ReusePlan::Append { n_past: 4 }
        );
        assert_eq!(
            reuse_plan(&[1, 2, 3, 4], &[1, 2, 3, 4], 64, 16),
            ReusePlan::Exact
        );
        assert_eq!(
            reuse_plan(&[1, 2, 3, 4], &[1, 2], 64, 16),
            ReusePlan::Rebuild
        );
        assert_eq!(
            reuse_plan(&[1, 2, 3, 4], &[9, 9], 64, 16),
            ReusePlan::Rebuild
        );
        assert_eq!(
            reuse_plan(&[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6], 8, 16),
            ReusePlan::Rebuild
        );
        assert_eq!(reuse_plan(&[], &[1, 2], 64, 16), ReusePlan::Rebuild);
        assert_eq!(n_past(&[1, 2, 3], &[1, 2, 3]), 2);
        assert_eq!(n_past(&[1, 2, 3, 4], &[1, 2, 3, 4, 5]), 4);
    }

    #[test]
    fn abort_flag_is_honored() {
        let mut opts = GenerateOpts::default();
        assert!(!aborted(&opts));
        let flag = Arc::new(AtomicBool::new(true));
        opts.abort = Some(flag);
        assert!(aborted(&opts));
    }

    #[test]
    fn tcp_drop_stops_prefill_at_remaining_chunks() {
        let path = match std::env::var("BUTTER_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip tcp_drop_stops_prefill_at_remaining_chunks: no BUTTER_QWEN35_GGUF");
                return;
            }
        };
        let devices = crate::devices();
        let Some(dev) = devices.first() else {
            eprintln!("skip tcp_drop_stops_prefill_at_remaining_chunks: no device");
            return;
        };
        let LoadedModel::Qwen35 { model, .. } = load(dev.as_ref(), &path).expect("load");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::net::TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        let abort = watch_disconnect(&server);
        let tokens: Vec<u32> = (0..64).map(|i| (i % 16) as u32).collect();
        let mut cache = qwen35::Qwen35Cache::new(dev.as_ref(), &model.cfg, 128).unwrap();
        let abort_c = abort.clone();
        let dropper = std::thread::spawn(move || {
            while qwen35::PREFILL_CHUNKS_DONE.load(Ordering::Relaxed) < 1 {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            drop(client);
        });
        let r = qwen35::prefill_at(
            dev.as_ref(),
            &model,
            &mut cache,
            &tokens,
            0,
            1,
            Some(abort_c.as_ref()),
        );
        dropper.join().unwrap();
        abort.store(true, Ordering::Relaxed);
        let done = qwen35::PREFILL_CHUNKS_DONE.load(Ordering::Relaxed);
        assert!(
            r.is_err(),
            "prefill_at finished after TCP drop, chunks={done}"
        );
        assert!(done >= 1, "no GPU chunk ran");
        assert!(done < 64, "leftover prefill completed: chunks={done}");
    }

    #[test]
    fn fp8_suffix_prefill_reuses_real_prefix() {
        let Ok(path) = std::env::var("BUTTER_QWEN35_NVFP4") else {
            eprintln!("skip fp8_suffix_prefill_reuses_real_prefix: no checkpoint");
            return;
        };
        let devices = crate::devices();
        let Some(dev) = devices.first() else { return };
        let loaded = load(dev.as_ref(), &path).expect("load");
        let opts = GenerateOpts { max_new: 1, max_context: Some(256), ..GenerateOpts::default() };
        let stem = "The capital of France is";
        let extended = "The capital of France is Paris. The capital of Germany is";
        let mut live = None;
        let first = generate_with_sink_cached(dev.as_ref(), &loaded, stem, &opts, &mut live, |_, _| true).expect("stem");
        let reused = generate_with_sink_cached(dev.as_ref(), &loaded, extended, &opts, &mut live, |_, _| true).expect("suffix");
        let mut fresh = None;
        let rebuilt = generate_with_sink_cached(dev.as_ref(), &loaded, extended, &opts, &mut fresh, |_, _| true).expect("rebuild");
        assert_eq!(reused.cached_tokens, first.prompt_tokens);
        assert_eq!(reused.token_ids, rebuilt.token_ids, "suffix prefill diverged from a rebuild");
    }
}
