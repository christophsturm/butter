// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Laguna-S-2.1 (GGUF arch "laguna"): a 117.55B / 8.14B-active sigmoid-routed
//! MoE transformer with a per-head softplus attention-output gate, per-head
//! QK-RMSNorm, and hybrid full/sliding-window attention (period-4, dense-
//! first). Layer 0 is a dense SwiGLU MLP (`mlp_only_layers = [0]`); layers
//! 1..47 are MoE (256 experts, top-10, sigmoid router with a score-
//! correction bias, one always-on shared expert). Full-attention layers use
//! YaRN RoPE over the first half of each head's dims; sliding-window layers
//! (window 512) use plain RoPE over the full head dim.
//!
//! Decode: single-token forward with a persistent KV cache (growing for
//! full-attention layers, a fixed-512 ring buffer for sliding layers).
//! Prefill: chunked batched forward (see [`prefill`]) with a linear sliding
//! window scratch compacted into the decode ring afterward.
//!
//! Weights load straight from GGUF (see [`ffai_loader::gguf::Gguf`]):
//! dequantize each tensor to f32 once, then requantize into this engine's Q4
//! format ([`ffai_ops::quantize_q4`]) and drop the f32, the GGUF k-quant
//! (Q4_K/Q6_K/...) never needs to match our block layout. Embeddings, the LM
//! head, and all norms stay dense f32 (small relative to the expert stacks
//! and precision-sensitive for routing/final logits).

use ffai_core::{DType, Device, Error, Result, Tensor};
use ffai_loader::gguf::Gguf;
use ffai_ops as ops;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

fn u32s_to_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
/// The `ffai_gemv_q4_coalesced*` kernels read per-block scales as f16 (the
/// `d_f32: Tensor<f16>` param, name notwithstanding); uploading raw f32 bits
/// there silently reinterprets them as garbage f16s. Convert on the host at
/// load, exactly like the validated `qload(.., f16=true)` path in modeltests.
fn f32s_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    fn half_bits(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 112;
        if e <= 0 { return sign; }
        if e >= 0x1f { return sign | 0x7c00; }
        let m = (x >> 13) & 0x3ff;
        let round = (x >> 12) & 1;
        let v = ((e as u32) << 10) | m;
        sign | ((v + round) as u16)
    }
    v.iter().flat_map(|&f| half_bits(f).to_le_bytes()).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// On-disk cache of converted engine-format weight blobs.
//
// `load_gguf` is the ~7.5min-cold-load path this exists to shortcut: per
// tensor it dequantizes GGUF (Q4_K/Q6_K/F16) to f32 on one CPU thread,
// requantizes to this engine's Q4 format (`ops::quantize_q4`), optionally
// permutes/interleaves/packs for FFAI_LAGUNA_MARLIN / _MOEFUSE / _W4A8, then
// uploads. Every one of those UPLOAD-READY byte blobs is cached here, one
// file per blob, so a warm reload skips straight to `read file -> upload`.
//
// LOUD REMINDER: bump `FORMAT_VERSION` whenever ANY of that byte-shaping
// logic changes (quantize_q4's block layout, permute_q4_to_marlin, the
// moefuse interleave order, w4a8_packw's packed layout, the f16 scale
// conversion, ...). Bumping it changes every blob's cache key, so stale
// pre-bump blobs are simply never read back (treated as a miss and rebuilt)
// instead of silently reinterpreted.
const FORMAT_VERSION: u32 = 1;

const CACHE_MAGIC: [u8; 4] = *b"FLC1"; // "FFAI Laguna Cache v1"

// `dtype_tag` is purely descriptive (a debugging aid if someone inspects a
// cache dir by hand), the cache never interprets a blob's bytes itself, it
// only ever hands them back verbatim to `dev.upload`.
const DTYPE_U32: u8 = 0;
const DTYPE_F32: u8 = 1;
const DTYPE_F16: u8 = 2;
/// Opaque packed bytes (Marlin-tile-permuted qs, W4A8 mxfp4 packed weights /
/// scale factors, ...): anything that isn't a flat array of one of the
/// above elementary types.
const DTYPE_RAW: u8 = 3;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Header: magic (4B) + FORMAT_VERSION (u32 LE) + dtype tag (1B) + payload
/// len (u64 LE), then the payload. Written atomically (tmp file + rename) so
/// a crash mid-write never leaves a corrupt file at the final path, readers
/// only ever see a complete blob or no file at all.
fn write_cache_blob(path: &std::path::Path, dtype_tag: u8, payload: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension(format!("tmp-{}-{:?}", std::process::id(), std::thread::current().id()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&CACHE_MAGIC)?;
        f.write_all(&FORMAT_VERSION.to_le_bytes())?;
        f.write_all(&[dtype_tag])?;
        f.write_all(&(payload.len() as u64).to_le_bytes())?;
        f.write_all(payload)?;
    }
    std::fs::rename(&tmp, path)
}

/// Reads back a blob written by [`write_cache_blob`]. Returns `None` for any
/// validation failure (missing file, bad magic, mismatched FORMAT_VERSION,
/// truncated payload), every failure mode is treated as a plain cache miss,
/// never an error, so a corrupt or foreign-version cache dir just gets
/// silently rebuilt.
fn read_cache_blob(path: &std::path::Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 17 || bytes[0..4] != CACHE_MAGIC {
        return None;
    }
    let fv = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if fv != FORMAT_VERSION {
        return None;
    }
    let len = u64::from_le_bytes(bytes[9..17].try_into().unwrap()) as usize;
    let payload = &bytes[17..];
    if payload.len() != len {
        return None;
    }
    Some(payload.to_vec())
}

/// On-disk cache of converted engine-format weight blobs, keyed by a
/// caller-chosen artifact key (conventionally the GGUF tensor name, possibly
/// suffixed for composite/variant artifacts) folded together with
/// [`FORMAT_VERSION`], the source GGUF's identity (size + mtime), and every
/// `FFAI_LAGUNA_*` load-shaping env flag, so a changed checkpoint, a changed
/// requant flag combo, or a bumped `FORMAT_VERSION` each land in their own
/// namespace within the same cache directory rather than colliding.
///
/// `Send + Sync` (every field is): built once per [`load_gguf`] call and
/// shared (via `&CacheCtx`) across the rayon worker threads that build the
/// per-layer byte blobs in parallel on a cache miss.
pub struct CacheCtx {
    /// `None` when the cache directory couldn't be created, every lookup
    /// then unconditionally misses (rebuild every artifact, every load), so
    /// a read-only or unwritable filesystem degrades to "cache disabled"
    /// instead of failing the load.
    dir: Option<std::path::PathBuf>,
    version_tag: u64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CacheCtx {
    /// `FFAI_LAGUNA_CACHE` overrides the cache directory; otherwise it lives
    /// alongside the GGUF at `<gguf_path>.ffai-cache/`.
    pub fn new(g: &Gguf) -> Self {
        let path = g.path();
        let dir = match std::env::var("FFAI_LAGUNA_CACHE") {
            Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
            _ => std::path::PathBuf::from(format!("{path}.ffai-cache")),
        };
        let dir = match std::fs::create_dir_all(&dir) {
            Ok(()) => Some(dir),
            Err(e) => {
                eprintln!(
                    "Laguna cache: could not create cache dir {} ({e}), caching disabled, every load will fully rebuild",
                    dir.display()
                );
                None
            }
        };
        // Source-GGUF identity: (size, mtime) rather than a content hash,
        // hashing an 80GB+ checkpoint on every load would itself dominate
        // load time. A changed file (new export, re-quant, ...) almost
        // always changes at least one of these; a false-negative (same
        // size+mtime, different content) is the same class of risk `make`
        // and every mtime-based build cache accepts.
        let (size, mtime) = std::fs::metadata(path)
            .map(|m| {
                let mtime = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                (m.len(), mtime)
            })
            .unwrap_or((0, 0));
        let flags = ["FFAI_LAGUNA_MICRO", "FFAI_LAGUNA_MOEFUSE", "FFAI_LAGUNA_MARLIN", "FFAI_LAGUNA_W4A8", "FFAI_LAGUNA_W4A8_ONLY"]
            .iter()
            .map(|k| format!("{k}={}", std::env::var(k).unwrap_or_default()))
            .collect::<Vec<_>>()
            .join("|");
        let version_tag = fnv1a64(format!("v{FORMAT_VERSION}|{size}|{mtime}|{flags}").as_bytes());
        CacheCtx { dir, version_tag, hits: AtomicU64::new(0), misses: AtomicU64::new(0) }
    }

    fn blob_path(&self, key: &str) -> Option<std::path::PathBuf> {
        let dir = self.dir.as_ref()?;
        let mut sanitized = String::with_capacity(key.len());
        for c in key.chars() {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                sanitized.push(c);
            } else {
                sanitized.push('_');
            }
        }
        let h = fnv1a64(format!("{key}\u{0}{}", self.version_tag).as_bytes());
        Some(dir.join(format!("{sanitized}-{h:016x}.bin")))
    }

    /// One cached byte blob. `build` runs (and its result gets written to
    /// disk) only on a miss.
    pub fn get_or_build(&self, key: &str, dtype_tag: u8, build: impl FnOnce() -> Result<Vec<u8>>) -> Result<Vec<u8>> {
        if let Some(path) = self.blob_path(key) {
            if let Some(bytes) = read_cache_blob(&path) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(bytes);
            }
        }
        let bytes = build()?;
        self.misses.fetch_add(1, Ordering::Relaxed);
        if let Some(path) = self.blob_path(key) {
            if let Err(e) = write_cache_blob(&path, dtype_tag, &bytes) {
                eprintln!("Laguna cache: failed writing '{key}': {e}");
            }
        }
        Ok(bytes)
    }

    /// Same contract as [`Self::get_or_build`] but for an artifact whose
    /// bytes are naturally co-produced by ONE expensive build step (e.g. a
    /// `Q4Dense`'s `qs` + `scales`, both products of a single
    /// `quantize_q4` call, or a `Q4Dense`+`MarlinDense` pair sharing one
    /// `quantize_q4` pass). Checks every key first; `build` runs at most
    /// once, only if at least one key is missing: so a full hit never
    /// redoes the shared work, and a full miss never redoes it either.
    pub fn get_or_build_multi(&self, keys: &[(&str, u8)], build: impl FnOnce() -> Result<Vec<Vec<u8>>>) -> Result<Vec<Vec<u8>>> {
        let paths: Vec<Option<std::path::PathBuf>> = keys.iter().map(|(k, _)| self.blob_path(k)).collect();
        let cached: Option<Vec<Vec<u8>>> = paths.iter().map(|p| p.as_ref().and_then(|p| read_cache_blob(p))).collect();
        if let Some(all) = cached {
            self.hits.fetch_add(keys.len() as u64, Ordering::Relaxed);
            return Ok(all);
        }
        let blobs = build()?;
        if blobs.len() != keys.len() {
            return Err(Error::Msg(format!("Laguna cache: build produced {} blobs, expected {} for keys {:?}", blobs.len(), keys.len(), keys.iter().map(|(k, _)| *k).collect::<Vec<_>>())));
        }
        self.misses.fetch_add(keys.len() as u64, Ordering::Relaxed);
        for ((key, dtype), (path, bytes)) in keys.iter().zip(paths.iter().zip(blobs.iter())) {
            if let Some(path) = path {
                if let Err(e) = write_cache_blob(path, *dtype, bytes) {
                    eprintln!("Laguna cache: failed writing '{key}': {e}");
                }
            }
        }
        Ok(blobs)
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits.load(Ordering::Relaxed), self.misses.load(Ordering::Relaxed))
    }

    pub fn dir_display(&self) -> String {
        match &self.dir {
            Some(d) => d.display().to_string(),
            None => "<disabled>".to_string(),
        }
    }
}

/// One `[m, k]` row-major f32 weight requantized to this engine's Q4 format
/// and uploaded. `m`/`k` match the `gemv_q4` convention (`k` must be a
/// multiple of 32).
struct Q4Dense {
    qs: Tensor,
    scales: Tensor,
    m: usize,
    k: usize,
}
impl Q4Dense {
    // Construction (dequant -> quantize_q4 -> upload, or the cache-hit
    // shortcut straight from disk) now lives in the free functions around
    // `q4dense_bytes_from_flat_cached` / `upload_q4dense_bytes` below, so
    // every construction site (plain, Marlin-shared, cached, parallel-phase)
    // goes through the same on-disk-cache-aware path instead of each having
    // its own uncached `dev.upload` call.
    fn gemv(&self, dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
        ops::gemv_q4(dev, &self.qs, &self.scales, x, self.m, self.k, self.m)
    }
    fn gemv_accum(&self, dev: &dyn Device, x: &Tensor, acc: &Tensor, scale: &Tensor) -> Result<()> {
        ops::gemv_q4_accum(dev, &self.qs, &self.scales, x, acc, scale, self.m, self.k, self.m)
    }
}

/// One `[n_out, k_in]` row-major f32 weight, requantized to Q4 and permuted
/// into the tensor-core Marlin tile-major layout ([`ops::permute_q4_to_marlin`])
/// for [`ops::moe_w4a16_marlin`]. Built ONLY under `FFAI_LAGUNA_MARLIN=1`, as
/// an ADDITIONAL resident copy alongside the plain [`Q4Dense`] decode uses,
/// this struct is prefill-only (see `prefill_layer`'s marlin branches).
///
/// `n_out` must be a multiple of 64 (`permute_q4_to_marlin`'s tile height);
/// `k_in` a multiple of 32 (Q4 block size). A dense `[n_out, k_in]` matrix is
/// fed to the (expert-batched) Marlin GEMM as a single "expert" (`n_exp=1`):
/// every row of `x` carries expert-id 0 via a persistent all-zero `indices`
/// buffer (see `prefill`'s `marlin_zero_idx`).
struct MarlinDense {
    qs: Tensor,     // Marlin tile-major Q4: [(n_out/64)*(k_in/32)*256] u32
    scales: Tensor, // f16, STANDARD (non-permuted) per-block layout: [n_out*(k_in/32)].
    // `moe_w4a16_marlin`'s kernel doc is explicit that scales are read in the
    // SAME layout `quantize_q4` produces; only `qs` gets tile-permuted.
    n_out: usize,
    k_in: usize,
}
impl MarlinDense {
    // Construction lives in `marlindense_bytes_from_flat_cached` /
    // `upload_marlindense_bytes` below (same rationale as `Q4Dense`'s impl
    // block doc above).
    /// `x` is `[t, k_in]` f16; `zero_idx` is a persistent all-zero `u32`
    /// buffer with at least `t` elements (the 1-expert "every row is expert
    /// 0" trick). Returns `[t, n_out]` f16, cast back with `ops::cast_f16_f32`
    /// wherever the caller needs f32 (see `prefill_layer`).
    fn call(&self, dev: &dyn Device, x_f16: &Tensor, zero_idx: &Tensor, t: usize) -> Result<Tensor> {
        ops::moe_w4a16_marlin(dev, x_f16, &self.qs, &self.scales, zero_idx, t, self.n_out, self.k_in)
    }
}

/// FFAI_LAGUNA_MARLIN=1: route dense prefill projections (QKV concat, o_proj,
/// layer-0 dense FFN, shared-expert FFN) through the tensor-core
/// [`MarlinDense`] GEMM instead of `gemm_q4_mpp`. Read once at load (to build
/// the extra Marlin-layout weight copies below) and once per `prefill_layer`
/// call (mirrors `laguna_micro_enabled`'s load/call split). Decode is
/// entirely unaffected, it never reads this env.
fn laguna_marlin_enabled() -> bool {
    std::env::var("FFAI_LAGUNA_MARLIN").map(|v| v == "1").unwrap_or(false)
}

/// Marlin-packed `[wq;wk;wv]` concat (gate excluded, its row count, 48 or 72,
/// is never 64-aligned, so it stays on `gemm_q4_mpp` via `AttnWeights::g_proj`).
/// `n_q_rows + 2*n_kv_rows` is 64-aligned for both Laguna-S-2.1 layer types
/// (full: 6144+2*1024=8192, sliding: 9216+2*1024=11264), verified at build
/// time rather than assumed, see `load_gguf`.
struct QkvMarlin {
    mat: MarlinDense, // [n_q_rows + 2*n_kv_rows, hidden]
    n_q_rows: usize,
    n_kv_rows: usize,
}

/// Marlin-packed `[gate;up]` concat (both halves share `n_ff` rows and
/// `hidden` columns), used for the dense layer-0 FFN and the always-on
/// shared expert, both of whose `2*n_ff` is 64-aligned for Laguna-S-2.1
/// (24576 and 2048 respectively).
struct GateUpMarlin {
    mat: MarlinDense, // [2*n_ff, hidden]
    n_ff: usize,
}

/// The always-on shared expert's Marlin copies (gate/up concat + down),
/// built regardless of whether decode's [`SharedExpert`] resolved to
/// `Split` or `Fused`, this struct reads the shexp GGUF tensors directly.
struct SharedMarlin {
    gateup: GateUpMarlin,
    down: MarlinDense, // [hidden, n_ff]
}

/// `n_expert` stacked `[m, k]` Q4 weights, requantized expert-by-expert and
/// concatenated into one device buffer per projection. A per-expert view is
/// a byte-offset slice, no host round-trip to select an expert's weights,
/// only the (tiny) expert-id lookup needs to be on the host.
struct Q4ExpertStack {
    qs: Tensor,
    scales: Tensor,
    m: usize,
    k: usize,
    qs_words_per_expert: usize,
    scale_elems_per_expert: usize,
}
impl Q4ExpertStack {
    // Construction lives in `q4_expert_stack_bytes_cached` /
    // `upload_expert_stack_bytes` below (same rationale as `Q4Dense`'s impl
    // block doc above); the per-expert `quantize_q4` loop that used to live
    // here is now inside that cached builder's closure.
    /// Device-side view of expert `e`'s `(qs, scales)`, a byte-offset slice
    /// into the shared stack, no copy.
    fn expert_view(&self, e: usize) -> (Tensor, Tensor) {
        let qw = self.qs_words_per_expert;
        let sw = self.scale_elems_per_expert;
        let qs = Tensor {
            buffer: self.qs.buffer.clone(),
            offset: self.qs.offset + e * qw * 4,
            shape: vec![qw],
            dtype: DType::U32,
        };
        let sc = Tensor {
            buffer: self.scales.buffer.clone(),
            offset: self.scales.offset + e * sw * 2, // f16 scales: 2 bytes each
            shape: vec![sw],
            dtype: DType::F16,
        };
        (qs, sc)
    }
    fn gemv(&self, dev: &dyn Device, e: usize, x: &Tensor) -> Result<Tensor> {
        let (qs, sc) = self.expert_view(e);
        ops::gemv_q4(dev, &qs, &sc, x, self.m, self.k, self.m)
    }
    fn gemv_accum(&self, dev: &dyn Device, e: usize, x: &Tensor, acc: &Tensor, scale: &Tensor) -> Result<()> {
        let (qs, sc) = self.expert_view(e);
        ops::gemv_q4_accum(dev, &qs, &sc, x, acc, scale, self.m, self.k, self.m)
    }
}

/// FFAI_LAGUNA_W4A8=1: one `[n_expert, n, k]` dequantized f32 expert-major
/// slab packed into the mxfp4 (e2m1 nibbles + per-32-block ue8m0 scale
/// factor) layout `ops::moe_grouped_gemm_w4a8` consumes as its weight
/// operand. Mirrors the NEMOTRON_W4A8_MOE loader's `buildw` closure
/// (`ffai-modeltests/src/lib.rs`, around `NEMOTRON_W4A8_MOE`) exactly: f32 →
/// f16 upload, zero-pad `k` up to the mxf8f6f4 SF atom width (128) via
/// `ops::pad_rows_f16` (a no-op clone when `k` is already 128-aligned, true
/// for both Laguna-S-2.1 K dims, `hidden=3072` and `n_ff_exp=1024`, so this
/// is a documented no-op on the shipped checkpoint, not dead code for a
/// hypothetical one), then `ops::w4a8_packw`. Neither this struct nor the
/// nemotron recipe it mirrors applies any outlier clipping or per-channel
/// rescale before packing, the packer's own per-32 ue8m0 block scale is the
/// only quantization-error mitigation on the weight side (mxfp4 is a
/// coarser format than the engine's own Q4: e2m1 has only 16 representable
/// magnitudes per block vs Q4's calibrated int4 codebook, so this is
/// expected to be lossier, the box's greedy-continuation gate decides
/// acceptance, per this file's module doc).
struct W4a8ExpertStack {
    packed: Tensor, // e2m1 nibble-packed, `ops::w4a8_packw`'s `packed` output
    sf: Tensor,     // per-32-block ue8m0 scale factors, atom-padded SFB layout
    /// Per-expert SFB stride in bytes (`ops::w4a8_packw`'s `sfb_exp_elems`
    /// return, fed straight through as `moe_grouped_gemm_w4a8`'s
    /// `sfb_exp_bytes`, see that fn's doc, the name is elem-count but the
    /// GEMM wants it as a byte stride, matching the nemotron call site).
    sfb_exp_bytes: i64,
    /// Original (unpadded) input width, `hidden` for gate/up (incl. the
    /// fused `2*n_ff_exp`-row stack), `n_ff_exp` for down. Not read at
    /// prefill time (only `kpad` drives the padding/quant calls below),
    /// kept for debugging/asserts, mirroring `Q4Dense`/`Q4ExpertStack`'s own
    /// `k` fields.
    #[allow(dead_code)]
    k: usize,
    /// `k` rounded up to the mxf8f6f4 SF atom width (128), the width
    /// actually packed and what every `w4a8_sfa_offsets`/
    /// `w4a8_actquant_grouped` call at prefill time must pad activations to
    /// before quantizing against this stack.
    kpad: usize,
}

/// FFAI_LAGUNA_W4A8=1: pack a dequantized `[n_expert, n, k]` f32 expert-major
/// slab (same `flat` a caller would otherwise only feed to `Q4ExpertStack`'s
/// builder, see the call sites in `build_moe_w4a8_sequential`, which
/// dequantize ONCE and hand the same slab to both) into a [`W4a8ExpertStack`],
/// threading the packed output through `cache`. See that struct's doc for the
/// exact recipe (f32→f16, pad-to-128, `ops::w4a8_packw`) and why it applies
/// no extra outlier handling.
///
/// UNLIKE every other cached artifact in this file, the pack itself
/// (`ops::pad_rows_f16` + `ops::w4a8_packw`) runs ON THE DEVICE, not on the
/// host, it needs `dev`, so it can never be moved into the rayon-parallel
/// per-layer CPU pass (`build_layer_bytes`); it always runs sequentially, on
/// whichever thread calls it (see `build_moe_w4a8_sequential`'s doc). On a
/// cache MISS this does exactly what the pre-cache `pack_w4a8_expert_stack`
/// did (upload raw f16, pad, pack) plus a download-back-to-host of the
/// result to write the cache blob. On a HIT, all of that: including the
/// f16 upload and both GPU kernels, is skipped: the packed bytes are
/// re-uploaded straight from disk.
fn pack_w4a8_expert_stack_cached(dev: &dyn Device, cache: &CacheCtx, key: &str, flat: &[f32], n_expert: usize, n: usize, k: usize) -> Result<W4a8ExpertStack> {
    let kpad = k.div_ceil(128) * 128;
    let packed_key = format!("{key}.w4a8.packed");
    let sf_key = format!("{key}.w4a8.sf");
    let meta_key = format!("{key}.w4a8.meta");
    let blobs = cache.get_or_build_multi(
        &[(packed_key.as_str(), DTYPE_RAW), (sf_key.as_str(), DTYPE_RAW), (meta_key.as_str(), DTYPE_RAW)],
        || -> Result<Vec<Vec<u8>>> {
            let wt = Tensor::new(dev.upload(&f32s_to_f16_bytes(flat))?, vec![n_expert * n, k], DType::F16);
            let wt_pad = ops::pad_rows_f16(dev, &wt, n_expert * n, k, kpad)?;
            let (packed, sf, sfb_exp_bytes) = ops::w4a8_packw(dev, &wt_pad, n_expert, n, kpad)?;
            dev.synchronize()?;
            let mut packed_bytes = vec![0u8; packed.elem_count() * 4];
            dev.download(packed.buffer.as_ref(), &mut packed_bytes)?;
            let mut sf_bytes = vec![0u8; sf.elem_count() * 4];
            dev.download(sf.buffer.as_ref(), &mut sf_bytes)?;
            Ok(vec![packed_bytes, sf_bytes, sfb_exp_bytes.to_le_bytes().to_vec()])
        },
    )?;
    let mut it = blobs.into_iter();
    let packed_bytes = it.next().unwrap();
    let sf_bytes = it.next().unwrap();
    let meta_bytes = it.next().unwrap();
    let packed = Tensor::new(dev.upload(&packed_bytes)?, vec![packed_bytes.len() / 4], DType::U32);
    let sf = Tensor::new(dev.upload(&sf_bytes)?, vec![sf_bytes.len() / 4], DType::U32);
    let sfb_exp_bytes = i64::from_le_bytes(meta_bytes[..8].try_into().map_err(|_| Error::Msg("Laguna cache: truncated w4a8 meta blob".into()))?);
    Ok(W4a8ExpertStack { packed, sf, sfb_exp_bytes, k, kpad })
}

/// YaRN RoPE parameters for one layer-type (full-attention or sliding).
#[derive(Debug, Clone, Copy)]
pub struct RopeParams {
    pub n_rot: u32,
    pub theta: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub n_orig_ctx: u32,
}

/// Architecture config for Laguna-S-2.1, read from GGUF `laguna.*` metadata
/// (arch spec fields fall back to the model's known-fixed values when a key
/// is absent, e.g. weightless test fixtures or a schema this loader hasn't
/// seen yet, always cross-check against a real GGUF's `laguna.*` keys when
/// wiring a new checkpoint).
#[derive(Debug, Clone)]
pub struct LagunaConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    /// Q head count on a full-attention layer (`i % swa_period == 0`).
    pub n_head_full: usize,
    /// Q head count on a sliding-window layer.
    pub n_head_sliding: usize,
    /// Hybrid attention period: layer `i` is full-attention iff `i % swa_period == 0`.
    pub swa_period: usize,
    pub sliding_window: usize,
    pub eps: f32,
    pub vocab: usize,
    /// Leading dense (non-MoE) block count (Laguna-S-2.1: 1, i.e. layer 0 only).
    pub n_layer_dense_lead: usize,
    pub n_ff_dense: usize, // layer-0 dense MLP intermediate size
    pub n_expert: usize,
    pub n_expert_used: usize, // top_k
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub routed_scaling: f32,
    pub rope_full: RopeParams,
    pub rope_sliding: RopeParams,
}

impl LagunaConfig {
    pub fn is_full_layer(&self, i: usize) -> bool {
        i % self.swa_period == 0
    }
    pub fn is_dense_layer(&self, i: usize) -> bool {
        i < self.n_layer_dense_lead
    }
    pub fn n_head(&self, i: usize) -> usize {
        if self.is_full_layer(i) { self.n_head_full } else { self.n_head_sliding }
    }
    pub fn heads_per_group(&self, i: usize) -> u32 {
        (self.n_head(i) / self.n_kv_heads) as u32
    }
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
    pub fn rope_for(&self, i: usize) -> RopeParams {
        if self.is_full_layer(i) { self.rope_full } else { self.rope_sliding }
    }

    /// The literal Laguna-S-2.1 spec values (48 layers), used as fallbacks
    /// when a GGUF key is missing.
    fn defaults() -> Self {
        LagunaConfig {
            hidden: 3072,
            n_layers: 48,
            head_dim: 128,
            n_kv_heads: 8,
            n_head_full: 48,
            n_head_sliding: 72,
            swa_period: 4,
            sliding_window: 512,
            eps: 1e-6,
            vocab: 100352,
            n_layer_dense_lead: 1,
            n_ff_dense: 12288,
            n_expert: 256,
            n_expert_used: 10,
            n_ff_exp: 1024,
            n_ff_shexp: 1024,
            routed_scaling: 2.5,
            rope_full: RopeParams {
                n_rot: 64,
                theta: 500000.0,
                freq_scale: 1.0 / 128.0,
                ext_factor: 1.0,
                attn_factor: 1.4852030263919618,
                beta_fast: 32.0,
                beta_slow: 1.0,
                n_orig_ctx: 8192,
            },
            rope_sliding: RopeParams {
                n_rot: 128,
                theta: 10000.0,
                freq_scale: 1.0,
                ext_factor: 0.0,
                attn_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                n_orig_ctx: 8192,
            },
        }
    }

    /// Read `laguna.*` GGUF metadata, falling back to [`Self::defaults`] for
    /// any key this loader doesn't find (see the module doc, verify key
    /// names against a real GGUF before trusting a mismatch silently).
    pub fn from_gguf(g: &Gguf) -> Self {
        let d = Self::defaults();
        let u = |k: &str, def: usize| g.meta_u32(k).map(|v| v as usize).unwrap_or(def);
        let f = |k: &str, def: f32| g.meta_f32(k).unwrap_or(def);

        let n_layers = u("laguna.block_count", d.n_layers);
        let sliding_window = u("laguna.attention.sliding_window", d.sliding_window);
        let n_rot_full = u("laguna.rope.dimension_count", d.rope_full.n_rot as usize) as u32;
        let n_rot_sliding = u("laguna.rope.dimension_count_swa", d.rope_sliding.n_rot as usize) as u32;

        LagunaConfig {
            hidden: u("laguna.embedding_length", d.hidden),
            n_layers,
            head_dim: u("laguna.attention.key_length", d.head_dim),
            n_kv_heads: u("laguna.attention.head_count_kv", d.n_kv_heads),
            // The per-layer-type Q head split isn't a single scalar GGUF key
            // in this loader (see the struct doc); keep the spec constants.
            n_head_full: d.n_head_full,
            n_head_sliding: d.n_head_sliding,
            swa_period: d.swa_period,
            sliding_window,
            eps: f("laguna.attention.layer_norm_rms_epsilon", d.eps),
            vocab: u("laguna.vocab_size", d.vocab),
            n_layer_dense_lead: u("laguna.leading_dense_block_count", d.n_layer_dense_lead),
            n_ff_dense: u("laguna.feed_forward_length", d.n_ff_dense),
            n_expert: u("laguna.expert_count", d.n_expert),
            n_expert_used: u("laguna.expert_used_count", d.n_expert_used),
            n_ff_exp: u("laguna.expert_feed_forward_length", d.n_ff_exp),
            n_ff_shexp: u("laguna.expert_shared_feed_forward_length", d.n_ff_shexp),
            routed_scaling: f("laguna.expert_weights_scale", d.routed_scaling),
            rope_full: {
                // The GGUF is authoritative over the HF config: this export is
                // 256K (factor 32, yarn_attn_factor 1 + 0.1 ln 32 = 1.34657),
                // NOT the HF card's 1M / factor 128. Read the stored scaling
                // keys; the spec constants are only a last-resort fallback.
                let factor = f("laguna.rope.scaling.factor", 1.0 / d.rope_full.freq_scale);
                let attn_factor = g.meta_f32("laguna.rope.scaling.yarn_attn_factor")
                    .unwrap_or_else(|| 1.0 + 0.1 * factor.ln());
                RopeParams {
                    n_rot: n_rot_full,
                    theta: f("laguna.rope.freq_base", d.rope_full.theta),
                    freq_scale: 1.0 / factor,
                    ext_factor: d.rope_full.ext_factor,
                    attn_factor,
                    beta_fast: f("laguna.rope.scaling.yarn_beta_fast", d.rope_full.beta_fast),
                    beta_slow: f("laguna.rope.scaling.yarn_beta_slow", d.rope_full.beta_slow),
                    n_orig_ctx: u("laguna.rope.scaling.original_context_length", d.rope_full.n_orig_ctx as usize) as u32,
                }
            },
            rope_sliding: RopeParams {
                n_rot: n_rot_sliding,
                theta: f("laguna.rope.freq_base_swa", d.rope_sliding.theta),
                freq_scale: d.rope_sliding.freq_scale,
                ext_factor: d.rope_sliding.ext_factor,
                attn_factor: d.rope_sliding.attn_factor,
                beta_fast: d.rope_sliding.beta_fast,
                beta_slow: d.rope_sliding.beta_slow,
                n_orig_ctx: d.rope_sliding.n_orig_ctx,
            },
        }
    }
}

/// FFAI_LAGUNA_MICRO=1 enables three decode-path micro-fusions (QKVG concat
/// gemv, shared-expert fused swiglu gather, o_proj residual accum). The env
/// is read once at load time (to decide which weight layout to build below,
/// [`AttnProj`] / [`SharedExpert`]) and once per [`decode_step`] (to decide
/// the runtime code path). Both reads see the same value: the env is fixed
/// for the process lifetime (required for CUDA-graph capture anyway, see
/// [`GraphBufs`]), so load-time and decode-time never disagree.
fn laguna_micro_enabled() -> bool {
    std::env::var("FFAI_LAGUNA_MICRO").map(|v| v == "1").unwrap_or(false)
}

/// FFAI_LAGUNA_MOEFUSE=1 fuses the routed MoE gate and up expert stacks
/// (`ffn_gate_exps`, `ffn_up_exps`) into ONE per-layer Q4 stack at load time,
/// so `prefill_layer`'s two `ops::moe_q4_grouped_mma_dev` calls over the gate
/// stack and the up stack (same sorted activation `a`, same `offsets`)
/// collapse into a single call with `n_out = 2*n_ff_exp`. See
/// [`RoutedGateUp::Fused`] for the exact per-expert row layout and the
/// decode-side tradeoff this implies. Read once at load time (to decide which
/// weight layout to build, same one-layout-resident rule as
/// [`laguna_micro_enabled`]); `decode_layer`/`prefill_layer` never re-read the
/// env, they just match on the [`RoutedGateUp`] variant `load_gguf` already
/// picked.
fn laguna_moefuse_enabled() -> bool {
    std::env::var("FFAI_LAGUNA_MOEFUSE").map(|v| v == "1").unwrap_or(false)
}

/// FFAI_LAGUNA_W4A8=1: route prefill's routed-MoE grouped GEMMs (gate/up,
/// down, the `moe_gateup_gemm`/`moe_down_gemm` stages that dominate the pp
/// profile, ~73% combined) through the W4A8 grouped-MoE GEMM (fp8 e4m3
/// activations x mxfp4 weights, cutlass mxf8f6f4) instead of the W4A16
/// `ops::moe_q4_grouped_mma_dev` path, mirroring the NEMOTRON_W4A8_MOE
/// recipe (`ffai-modeltests`'s `NEMOTRON_W4A8_MOE` build closure, and its
/// `w4a8_moe` prefill branch) exactly. Read once at load time (to build the
/// extra [`W4a8ExpertStack`]-packed weight copies below, ADDITIONAL to the
/// resident Q4 stacks decode still needs, see [`MoeFfnW4a8`]'s doc for the
/// resident-memory cost) and once per `prefill_layer` call is NOT needed:
/// the load-time-built `Option<MoeFfnW4a8>` on `MoeFfn` already encodes
/// whether this layer has a W4A8 path, so `prefill_layer` just matches on
/// it (same convention as `shared_marlin`/`qkv_marlin`). Decode is entirely
/// unaffected, it never reads this env or the `w4a8` field.
fn laguna_w4a8_enabled() -> bool {
    std::env::var("FFAI_LAGUNA_W4A8").map(|v| v == "1").unwrap_or(false)
}

/// QKV(+gate) projection weights. `Fused` concatenates all four projections'
/// rows into one `Q4Dense` at load time (built ONLY under
/// `FFAI_LAGUNA_MICRO=1`, so a layer never carries both layouts resident at
/// once, that would double the ~2.4GB of attention-projection weights).
/// Row order in `wqkvg` is `[wq rows; wk rows; wv rows; wgate rows]`, all
/// sharing `k = hidden`; concatenation happens on the dequantized f32 slabs
/// BEFORE `quantize_q4`, so each row's Q4 block boundaries are unaffected
/// (`quantize_q4` blocks are per-row, block size 32 along `k`).
enum AttnProj {
    Split { wq: Q4Dense, wk: Q4Dense, wv: Q4Dense, wgate: Q4Dense },
    Fused { wqkvg: Q4Dense, n_q_rows: usize, n_kv_rows: usize, n_g_rows: usize },
}

/// Per-layer attention weights (Q4, except the two `[head_dim]` per-head
/// norms which stay f32).
struct AttnWeights {
    /// This layer's Q head count, derived from the loaded Q projection's
    /// actual row count (`n_q_rows / head_dim`) rather than `LagunaConfig`'s
    /// hardcoded 48/72 split, so a real GGUF's per-layer head count always
    /// wins over the spec constants if the two ever disagree.
    n_head: usize,
    attn_norm: Tensor, // [hidden] f32
    proj: AttnProj,
    wo: Q4Dense,    // [hidden, n_head*head_dim]
    q_norm: Tensor, // [head_dim] f32
    k_norm: Tensor, // [head_dim] f32
    /// FFAI_LAGUNA_MARLIN=1 prefill-only extras (see [`QkvMarlin`]/
    /// [`MarlinDense`]'s docs); `None` when the env is unset OR (defensively)
    /// when a dimension isn't 64-row-aligned. `g_proj` is a plain standalone
    /// `Q4Dense` (NOT Marlin, gate's 48/72 rows are never 64-aligned) built
    /// only under the same env, so the marlin prefill branch has gate access
    /// independent of whether `proj` above resolved to `Split` or `Fused`.
    qkv_marlin: Option<QkvMarlin>,
    g_proj: Option<Q4Dense>,
    wo_marlin: Option<MarlinDense>,
}

/// Layer-0's dense SwiGLU MLP.
struct DenseFfn {
    ffn_norm: Tensor, // [hidden] f32
    gate: Q4Dense,    // [n_ff_dense, hidden]
    up: Q4Dense,      // [n_ff_dense, hidden]
    down: Q4Dense,    // [hidden, n_ff_dense]
    /// FFAI_LAGUNA_MARLIN=1 prefill-only extras, see [`GateUpMarlin`]/
    /// [`MarlinDense`]'s docs. `None` when the env is unset.
    gateup_marlin: Option<GateUpMarlin>,
    down_marlin: Option<MarlinDense>,
}

/// The always-on shared expert's gate+up projections. `Fused` repacks both
/// as single-expert (`n_expert=1`) [`Q4ExpertStack`]s at load time (built
/// ONLY under `FFAI_LAGUNA_MICRO=1`, same one-layout-resident rule as
/// [`AttnProj::Fused`]) plus a persistent one-element `idx=[0]` device
/// buffer, so decode can drive both through the same
/// `ops::moe_gather_q4_swiglu` the routed experts use (`top_k=1`) instead of
/// two plain gemvs + a host-orchestrated swiglu. `down` stays a plain
/// `Q4Dense` either way, only gate+up fuse into the gather.
struct SharedExpertFused {
    gate: Q4ExpertStack, // n_expert=1, [n_ff_shexp, hidden]
    up: Q4ExpertStack,   // n_expert=1, [n_ff_shexp, hidden]
    down: Q4Dense,       // [hidden, n_ff_shexp]
    /// Persistent device buffer holding the constant `u32` `0`, the single
    /// "expert index" `moe_gather_q4_swiglu` needs for a 1-expert stack.
    zero_idx: Tensor,
}
enum SharedExpert {
    Split { gate: Q4Dense, up: Q4Dense, down: Q4Dense },
    Fused(SharedExpertFused),
}

/// Routed-expert gate+up weight layout. `Split` is the plain per-projection
/// layout ([`Q4ExpertStack`] each). `Fused` is built ONLY under
/// `FFAI_LAGUNA_MOEFUSE=1` and REPLACES `Split` (see
/// [`laguna_moefuse_enabled`]'s doc, same one-layout-resident rule as
/// [`AttnProj`]/[`SharedExpert`]): expert `e`'s slab is `[2*n_ff_exp, hidden]`
/// with gate rows at local offset `[0, n_ff_exp)` and up rows at local offset
/// `[n_ff_exp, 2*n_ff_exp)`, i.e. per-expert interleaved (gate_0, up_0,
/// gate_1, up_1, ...), NOT all gate experts followed by all up experts. This
/// is the SAME single stack that `prefill_layer` and `decode_layer` both
/// read, so fusing gate+up does not add a second resident copy of the expert
/// weights.
///
/// Decode tradeoff: the unfused layout lets decode drive gate+up through one
/// `ops::moe_gather_q4_swiglu` call under `FFAI_LAGUNA_FUSE=1` (fused inline
/// SwiGLU, reads the activation once). The fused stack's per-expert row
/// layout does not fit that kernel's `row = e*inter + local` indexing for two
/// SEPARATE logical projections sharing one physical stack, so decode falls
/// back to one `ops::moe_gather_q4` (inter=2*n_ff_exp) plus a host-orchestrated
/// column split and `ops::swiglu` instead (see `decode_layer`'s MoE branch).
/// That is roughly the pre-FFAI_LAGUNA_FUSE decode path (about 1 tok/s slower
/// than the fused gather), traded for prefill's halved grouped-GEMM call
/// count. Default OFF.
enum RoutedGateUp {
    Split { gate: Q4ExpertStack, up: Q4ExpertStack },
    Fused(Q4ExpertStack), // per-expert [2*n_ff_exp, hidden]
}

/// FFAI_LAGUNA_W4A8=1 mirror of [`RoutedGateUp`], packed as
/// [`W4a8ExpertStack`]s instead of [`Q4ExpertStack`]s. Which variant is
/// built follows the SAME `FFAI_LAGUNA_MOEFUSE` choice `RoutedGateUp` made
/// for this layer (`load_gguf` builds both from the identical dequantized
/// f32 slab, so `Fused` here always pairs with `RoutedGateUp::Fused` and
/// vice versa), `prefill_layer` never needs to check the two enums agree,
/// they're constructed together.
enum RoutedGateUpW4a8 {
    Split { gate: W4a8ExpertStack, up: W4a8ExpertStack },
    Fused(W4a8ExpertStack), // per-expert [2*n_ff_exp, hidden]
}

/// FFAI_LAGUNA_W4A8=1 prefill-only routed-expert extras: mxfp4-packed
/// [`RoutedGateUpW4a8`] + down [`W4a8ExpertStack`], built ADDITIONALLY to
/// (never replacing) `MoeFfn`'s existing `gateup`/`down` [`Q4ExpertStack`]s
///, decode always drives the Q4 stacks, this is prefill-only, mirroring
/// every other `FFAI_LAGUNA_*` weight-layout env in this file. Unlike those,
/// this ADDS roughly 30GB of resident memory across all MoE layers (the
/// routed expert stacks are the bulk of the model, Q4 stays resident for
/// decode, so this genuinely doubles the routed-expert footprint rather
/// than trading one layout for another); acceptable for a prefill-throughput
/// win on hardware with the headroom, but unlike `FFAI_LAGUNA_MARLIN`
/// (which is roughly memory-neutral, permuting the SAME quantized weight)
/// this is NOT a "just flip it on" env on a memory-constrained box.
struct MoeFfnW4a8 {
    gateup: RoutedGateUpW4a8,
    down: W4a8ExpertStack, // per-expert [hidden, n_ff_exp]
}

/// A routed MoE FFN block (layers `n_layer_dense_lead..`).
struct MoeFfn {
    ffn_norm: Tensor,  // [hidden] f32
    router: Tensor,    // [n_expert, hidden] f32: kept dense for routing precision
    bias: Tensor,      // [n_expert] f32: e_score_correction_bias (selection only)
    gateup: RoutedGateUp,
    down: Q4ExpertStack, // per-expert [hidden, n_ff_exp]
    shared_expert: SharedExpert,
    /// FFAI_LAGUNA_MARLIN=1 prefill-only shared-expert extras, see
    /// [`SharedMarlin`]'s doc. `None` when the env is unset. The routed
    /// per-expert `gate`/`up`/`down` stacks above stay on the existing
    /// grouped-MMA prefill path (`ops::moe_q4_grouped_mma_dev`), that's
    /// already a tensor-core grouped GEMM, not the GEMV-class kernel this
    /// Marlin work targets.
    shared_marlin: Option<SharedMarlin>,
    /// FFAI_LAGUNA_W4A8=1 prefill-only routed-expert extras, see
    /// [`MoeFfnW4a8`]'s doc. `None` when the env is unset; when `Some`,
    /// `prefill_layer` routes the routed gate/up/down grouped GEMMs through
    /// `ops::moe_grouped_gemm_w4a8` instead of `ops::moe_q4_grouped_mma_dev`
    /// (the shared expert and everything else in the layer is untouched).
    w4a8: Option<MoeFfnW4a8>,
}

enum LayerFfn {
    Dense(DenseFfn),
    Moe(MoeFfn),
}

struct LagunaLayer {
    attn: AttnWeights,
    ffn: LayerFfn,
}

/// LM head resident as Q8 (the f32 head was a 1.2GB read per token, ~10% of
/// the whole decode bandwidth budget; Q8 quarters it at negligible logit
/// error). Same layout `ffai_ops::gemv_q8` consumes: 4 codes/u32, f32 scale
/// per 32-block.
struct Q8Head {
    qs: Tensor,
    scales: Tensor,
    m: usize,
    k: usize,
}

/// Full Laguna-S-2.1 weights, resident on `dev`.
pub struct LagunaModel {
    pub cfg: LagunaConfig,
    tok_embd: Tensor, // [vocab, hidden] f32
    layers: Vec<LagunaLayer>,
    out_norm: Tensor, // [hidden] f32
    lm_head: Q8Head,  // [vocab, hidden] q8
    /// Cached device-resident `1.0f32` scalar, reused as the shared expert's
    /// `gemv_q4_accum` weight (unconditional, unweighted accumulate).
    one: Tensor,
}

/// Persistent decode-time KV cache: one entry per layer, growing (`cap =
/// max_seq`) for full-attention layers, a fixed 512-slot ring for sliding
/// layers. RoPE is applied (with the token's absolute position) before the
/// K row is cached, so ring order doesn't matter for attention: softmax
/// over the cached K/V set is order-invariant once every slot is filled.
struct KvSlot {
    k: Tensor, // [n_kv_heads, cap, head_dim]
    v: Tensor,
    cap: usize,
}
pub struct LagunaKvCache {
    slots: Vec<KvSlot>,
}
impl LagunaKvCache {
    pub fn new(dev: &dyn Device, cfg: &LagunaConfig, max_seq: usize) -> Result<Self> {
        let mut slots = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let cap = if cfg.is_full_layer(i) { max_seq } else { cfg.sliding_window };
            let n = cfg.n_kv_heads * cap * cfg.head_dim;
            let k = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, cap, cfg.head_dim], DType::F32);
            let v = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, cap, cfg.head_dim], DType::F32);
            slots.push(KvSlot { k, v, cap });
        }
        Ok(LagunaKvCache { slots })
    }
}

/// Per-sliding-layer LINEAR K/V scratch used ONLY during [`prefill`]. The
/// existing sliding-layer decode cache (a `KvSlot` inside [`LagunaKvCache`])
/// is a fixed `sliding_window`-slot RING, safe for single-token decode
/// (write one slot, evict the oldest), but NOT safe to batch-write during
/// prefill: a `kv_append_many` writing a whole T-token chunk into the ring
/// would evict entries that EARLIER queries in that SAME chunk still need to
/// attend to (their window hasn't slid that far yet). Prefill instead writes
/// every sliding-layer token into a LINEAR buffer sized to the whole prompt
/// (`[n_kv_heads, prompt_cap, head_dim]`, absolute positions, no
/// wraparound), attends with `seg_lo[i] = max(0, pos_i - (sliding_window -
/// 1))` via `ops::sdpa_multi_tc_varlen`, and once the whole prompt has been
/// processed, [`prefill`] compacts the last `sliding_window` positions into
/// the decode ring (`ops::kv_compact_ring`) so `decode_step` /
/// `LagunaDecodeCtx` can continue from `pos = tokens.len()` exactly as if
/// every prompt token had gone through the per-token decode path.
///
/// Full-attention layers need no scratch, they reuse `LagunaKvCache`'s
/// already-growing per-layer buffer directly (a batched write there is safe:
/// nothing is ever evicted).
///
/// Memory: `n_kv_heads * prompt_cap * head_dim * 4 bytes * 2 (K+V)` per
/// sliding layer. At `prompt_cap = 32768` (this v1's documented supported
/// prompt-length ceiling) that is `8 * 32768 * 128 * 4 * 2 ≈ 256MiB` per
/// sliding layer; Laguna-S-2.1 has 36 sliding layers (48 total, minus 12
/// full-attention layers every 4th), so a full-size scratch is `≈9.7GB`.
/// Allocate ONE scratch sized to the largest prompt the caller expects to
/// prefill, not per call.
pub struct LagunaPrefillScratch {
    /// Indexed by layer; `None` for full-attention layers (`i % swa_period == 0`).
    slots: Vec<Option<KvSlot>>,
    prompt_cap: usize,
}

impl LagunaPrefillScratch {
    pub fn new(dev: &dyn Device, cfg: &LagunaConfig, prompt_cap: usize) -> Result<Self> {
        let mut slots = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            if cfg.is_full_layer(i) {
                slots.push(None);
                continue;
            }
            let n = cfg.n_kv_heads * prompt_cap * cfg.head_dim;
            let k = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, prompt_cap, cfg.head_dim], DType::F32);
            let v = Tensor::new(dev.alloc_zeroed(n * 4)?, vec![cfg.n_kv_heads, prompt_cap, cfg.head_dim], DType::F32);
            slots.push(Some(KvSlot { k, v, cap: prompt_cap }));
        }
        Ok(LagunaPrefillScratch { slots, prompt_cap })
    }
}

fn dims_out_in(t: &ffai_loader::gguf::GgufTensor, name: &str) -> Result<(usize, usize)> {
    // GGUF dims are fastest-first: a 2-D `[out, in]` weight is stored as
    // `[in, out]` (the reference C++ implementation's `create_tensor({k, m}, ...)` convention), which
    // is exactly the `(m, k)` `gemv_q4` wants with zero transposition.
    if t.dims.len() != 2 {
        return Err(Error::Msg(format!("{name}: expected 2-D tensor, got dims {:?}", t.dims)));
    }
    Ok((t.dims[1] as usize, t.dims[0] as usize))
}

/// Read `name`'s `(m, k)` from its GGUF header dims alone, cheap (metadata
/// only, no mmap data touched), so this is always safe to call eagerly, even
/// when the tensor's actual bytes end up not being dequantized at all (a
/// full cache hit for that tensor's requantized artifact).
fn tensor_dims(g: &Gguf, name: &str) -> Result<(usize, usize)> {
    let t = g.tensor(name).ok_or_else(|| Error::Msg(format!("gguf: tensor '{name}' missing")))?;
    dims_out_in(t, name)
}

// ── Bytes-mirror types ──────────────────────────────────────────────────
//
// Host-side (no `Tensor`, no `dev`) mirrors of the device-resident weight
// structs above, holding UPLOAD-READY byte blobs instead of `Tensor`s. This
// is the payload of `build_layer_bytes` (the rayon-parallel, cache-checked,
// pure-CPU half of loading one layer), `upload_layer` below turns one of
// these into the real `LagunaLayer` by doing nothing but `dev.upload` +
// `Tensor::new` calls, sequentially.

struct Q4DenseBytes { qs: Vec<u8>, scales: Vec<u8>, m: usize, k: usize }
struct MarlinDenseBytes { qs: Vec<u8>, scales: Vec<u8>, n_out: usize, k_in: usize }
struct Q4ExpertStackBytes { qs: Vec<u8>, scales: Vec<u8>, m: usize, k: usize }

enum AttnProjBytes {
    Fused { wqkvg: Q4DenseBytes, n_q_rows: usize, n_kv_rows: usize, n_g_rows: usize },
    Split { wq: Q4DenseBytes, wk: Q4DenseBytes, wv: Q4DenseBytes, wgate: Q4DenseBytes },
}
struct QkvMarlinBytes { mat: MarlinDenseBytes, n_q_rows: usize, n_kv_rows: usize }
struct GateUpMarlinBytes { mat: MarlinDenseBytes, n_ff: usize }
struct SharedMarlinBytes { gateup: GateUpMarlinBytes, down: MarlinDenseBytes }

struct AttnWeightsBytes {
    n_head: usize,
    attn_norm: Vec<u8>,
    proj: AttnProjBytes,
    wo: Q4DenseBytes,
    q_norm: Vec<u8>,
    k_norm: Vec<u8>,
    qkv_marlin: Option<QkvMarlinBytes>,
    g_proj: Option<Q4DenseBytes>,
    wo_marlin: Option<MarlinDenseBytes>,
}

struct DenseFfnBytes {
    ffn_norm: Vec<u8>,
    gate: Q4DenseBytes,
    up: Q4DenseBytes,
    down: Q4DenseBytes,
    gateup_marlin: Option<GateUpMarlinBytes>,
    down_marlin: Option<MarlinDenseBytes>,
}

struct SharedExpertFusedBytes { gate: Q4ExpertStackBytes, up: Q4ExpertStackBytes, down: Q4DenseBytes }
enum SharedExpertBytes {
    Split { gate: Q4DenseBytes, up: Q4DenseBytes, down: Q4DenseBytes },
    Fused(SharedExpertFusedBytes),
}
enum RoutedGateUpBytes {
    Split { gate: Q4ExpertStackBytes, up: Q4ExpertStackBytes },
    Fused(Q4ExpertStackBytes),
}

struct MoeFfnBytes {
    ffn_norm: Vec<u8>,
    router: Vec<u8>,
    bias: Vec<u8>,
    /// `None` under FFAI_LAGUNA_W4A8=1: the routed gate/up/down stacks are
    /// built afterward, sequentially, in `upload_layer` /
    /// `build_moe_w4a8_sequential` instead of here, W4A8 packing
    /// (`ops::pad_rows_f16` / `ops::w4a8_packw`) is a GPU op and can't run on
    /// a rayon worker thread, and it needs the SAME dequantized slab the Q4
    /// stack does, so keeping the two together (both sequential) avoids
    /// dequantizing the routed tensor twice.
    gateup: Option<RoutedGateUpBytes>,
    down: Option<Q4ExpertStackBytes>,
    shared_expert: SharedExpertBytes,
    shared_marlin: Option<SharedMarlinBytes>,
}

enum LayerFfnBytes { Dense(DenseFfnBytes), Moe(MoeFfnBytes) }
struct LayerBytes { attn: AttnWeightsBytes, ffn: LayerFfnBytes }

// ── Cache-aware builders (host bytes in, cache-checked, no `dev`) ──────────

/// The core cached-requant primitive: `build_flat` (a DEFERRED dequant, it
/// only runs on a miss, so a cache hit never touches the GGUF mmap) produces
/// the `[m, k]` f32 slab, which gets `quantize_q4`'d and cached as its
/// `qs`/`scales` byte pair under `key`.
fn q4dense_bytes_from_flat_cached(cache: &CacheCtx, key: &str, m: usize, k: usize, build_flat: impl FnOnce() -> Result<Vec<f32>>) -> Result<Q4DenseBytes> {
    let qs_key = format!("{key}.q4.qs");
    let sc_key = format!("{key}.q4.scales");
    let blobs = cache.get_or_build_multi(&[(qs_key.as_str(), DTYPE_U32), (sc_key.as_str(), DTYPE_F16)], || {
        let flat = build_flat()?;
        let (qs, scales) = ops::quantize_q4(&flat, m, k);
        Ok(vec![u32s_to_bytes(&qs), f32s_to_f16_bytes(&scales)])
    })?;
    let mut it = blobs.into_iter();
    Ok(Q4DenseBytes { qs: it.next().unwrap(), scales: it.next().unwrap(), m, k })
}

/// Marlin-only counterpart of [`q4dense_bytes_from_flat_cached`]: quantizes
/// AND Marlin-permutes inside the (deferred, miss-only) build closure.
fn marlindense_bytes_from_flat_cached(cache: &CacheCtx, key: &str, n_out: usize, k_in: usize, build_flat: impl FnOnce() -> Result<Vec<f32>>) -> Result<MarlinDenseBytes> {
    let qs_key = format!("{key}.marlin.qs");
    let sc_key = format!("{key}.marlin.scales");
    let blobs = cache.get_or_build_multi(&[(qs_key.as_str(), DTYPE_U32), (sc_key.as_str(), DTYPE_F16)], || {
        let flat = build_flat()?;
        let (qs_std, scales) = ops::quantize_q4(&flat, n_out, k_in);
        let qs_mar = ops::permute_q4_to_marlin(&qs_std, 1, n_out, k_in);
        Ok(vec![u32s_to_bytes(&qs_mar), f32s_to_f16_bytes(&scales)])
    })?;
    let mut it = blobs.into_iter();
    Ok(MarlinDenseBytes { qs: it.next().unwrap(), scales: it.next().unwrap(), n_out, k_in })
}

/// Cached counterpart of the old `load_dense_and_marlin`: one deferred
/// `quantize_q4` pass feeds BOTH the plain `Q4DenseBytes` (`.q4.*` keys,
/// always built) and, when `want_marlin && m % 64 == 0`, an additional
/// `MarlinDenseBytes` (`.marlin.*` keys) permuted from the SAME `qs_std`. A
/// full cache hit (all keys present) skips `build_flat`, and therefore the
/// GGUF dequant, entirely.
fn q4dense_and_marlin_bytes_cached(cache: &CacheCtx, key: &str, m: usize, k: usize, want_marlin: bool, build_flat: impl FnOnce() -> Result<Vec<f32>>) -> Result<(Q4DenseBytes, Option<MarlinDenseBytes>)> {
    let do_marlin = want_marlin && m % 64 == 0;
    let qs_key = format!("{key}.q4.qs");
    let sc_key = format!("{key}.q4.scales");
    let mqs_key = format!("{key}.marlin.qs");
    let msc_key = format!("{key}.marlin.scales");
    let mut keys: Vec<(&str, u8)> = vec![(qs_key.as_str(), DTYPE_U32), (sc_key.as_str(), DTYPE_F16)];
    if do_marlin {
        keys.push((mqs_key.as_str(), DTYPE_U32));
        keys.push((msc_key.as_str(), DTYPE_F16));
    }
    let blobs = cache.get_or_build_multi(&keys, || {
        let flat = build_flat()?;
        let (qs_std, scales) = ops::quantize_q4(&flat, m, k);
        let mut out = vec![u32s_to_bytes(&qs_std), f32s_to_f16_bytes(&scales)];
        if do_marlin {
            let qs_mar = ops::permute_q4_to_marlin(&qs_std, 1, m, k);
            out.push(u32s_to_bytes(&qs_mar));
            out.push(f32s_to_f16_bytes(&scales));
        }
        Ok(out)
    })?;
    let mut it = blobs.into_iter();
    let dense = Q4DenseBytes { qs: it.next().unwrap(), scales: it.next().unwrap(), m, k };
    let marlin = if do_marlin {
        Some(MarlinDenseBytes { qs: it.next().unwrap(), scales: it.next().unwrap(), n_out: m, k_in: k })
    } else {
        None
    };
    Ok((dense, marlin))
}

/// Cached dense f32 vector (norms, router, bias, embeddings, ...), the
/// upload bytes are literally `f32s_to_bytes(flat)`, so caching mostly buys
/// skipping `build_flat`'s dequant on a hit (which matters a lot for the
/// two big ones, `token_embd.weight` / `output.weight`, and is free for the
/// tiny per-layer norms).
fn f32vec_bytes_cached(cache: &CacheCtx, key: &str, build_flat: impl FnOnce() -> Result<Vec<f32>>) -> Result<Vec<u8>> {
    cache.get_or_build(&format!("{key}.f32vec"), DTYPE_F32, || Ok(f32s_to_bytes(&build_flat()?)))
}

/// Cached [`Q4ExpertStack`] bytes: `build_flat` (deferred) produces the
/// `[n_expert, m, k]` expert-major f32 slab; the per-expert `quantize_q4`
/// loop (the heavy part) only runs on a miss.
fn q4_expert_stack_bytes_cached(cache: &CacheCtx, key: &str, n_expert: usize, m: usize, k: usize, build_flat: impl FnOnce() -> Result<Vec<f32>>) -> Result<Q4ExpertStackBytes> {
    let qs_key = format!("{key}.q4x.qs");
    let sc_key = format!("{key}.q4x.scales");
    let blobs = cache.get_or_build_multi(&[(qs_key.as_str(), DTYPE_U32), (sc_key.as_str(), DTYPE_F16)], || {
        let flat = build_flat()?;
        let per_expert = m * k;
        if flat.len() != per_expert * n_expert {
            return Err(Error::Msg(format!(
                "Q4ExpertStack '{key}': flat len {} != n_expert*m*k ({} * {} * {})",
                flat.len(), n_expert, m, k
            )));
        }
        let qw = per_expert / 32 * 4; // quantize_q4: m*k/32 blocks, 4 u32/block
        let sw = per_expert / 32; // 1 f32 scale/block
        let mut qs_all = Vec::with_capacity(qw * n_expert);
        let mut sc_all = Vec::with_capacity(sw * n_expert);
        for e in 0..n_expert {
            let slab = &flat[e * per_expert..(e + 1) * per_expert];
            let (qs, sc) = ops::quantize_q4(slab, m, k);
            qs_all.extend_from_slice(&qs);
            sc_all.extend_from_slice(&sc);
        }
        Ok(vec![u32s_to_bytes(&qs_all), f32s_to_f16_bytes(&sc_all)])
    })?;
    let mut it = blobs.into_iter();
    Ok(Q4ExpertStackBytes { qs: it.next().unwrap(), scales: it.next().unwrap(), m, k })
}

/// Cached [`Q8Head`] bytes via [`Gguf::q8_repack`] (deferred, only runs on a
/// miss). `(m, k)` come from the tensor's own GGUF dims (matches
/// `q8_repack`'s internal derivation exactly, both read `dims[1]`/`dims[0]`
/// the same way `dims_out_in` does), so they're known without calling
/// `q8_repack` and therefore available even on a full cache hit.
fn q8head_bytes_cached(cache: &CacheCtx, g: &Gguf, name: &str, key: &str) -> Result<(Vec<u8>, Vec<u8>, usize, usize)> {
    let (m, k) = tensor_dims(g, name)?;
    let qs_key = format!("{key}.q8.qs");
    let sc_key = format!("{key}.q8.scales");
    let blobs = cache.get_or_build_multi(&[(qs_key.as_str(), DTYPE_U32), (sc_key.as_str(), DTYPE_F32)], || {
        let (qs, sc, _m, _k) = g.q8_repack(name)?;
        Ok(vec![u32s_to_bytes(&qs), f32s_to_bytes(&sc)])
    })?;
    let mut it = blobs.into_iter();
    Ok((it.next().unwrap(), it.next().unwrap(), m, k))
}

// ── Upload-phase helpers (sequential, `dev`-only, no CPU-heavy work) ───────

fn upload_q4dense_bytes(dev: &dyn Device, b: Q4DenseBytes) -> Result<Q4Dense> {
    Ok(Q4Dense {
        qs: Tensor::new(dev.upload(&b.qs)?, vec![b.qs.len() / 4], DType::U32),
        scales: Tensor::new(dev.upload(&b.scales)?, vec![b.scales.len() / 2], DType::F16),
        m: b.m,
        k: b.k,
    })
}
fn upload_marlindense_bytes(dev: &dyn Device, b: MarlinDenseBytes) -> Result<MarlinDense> {
    Ok(MarlinDense {
        qs: Tensor::new(dev.upload(&b.qs)?, vec![b.qs.len() / 4], DType::U32),
        scales: Tensor::new(dev.upload(&b.scales)?, vec![b.scales.len() / 2], DType::F16),
        n_out: b.n_out,
        k_in: b.k_in,
    })
}
fn upload_expert_stack_bytes(dev: &dyn Device, b: Q4ExpertStackBytes) -> Result<Q4ExpertStack> {
    let per_expert = b.m * b.k;
    Ok(Q4ExpertStack {
        qs: Tensor::new(dev.upload(&b.qs)?, vec![b.qs.len() / 4], DType::U32),
        scales: Tensor::new(dev.upload(&b.scales)?, vec![b.scales.len() / 2], DType::F16),
        m: b.m,
        k: b.k,
        qs_words_per_expert: per_expert / 32 * 4,
        scale_elems_per_expert: per_expert / 32,
    })
}
fn upload_f32_bytes(dev: &dyn Device, bytes: Vec<u8>) -> Result<Tensor> {
    let n = bytes.len() / 4;
    Ok(Tensor::new(dev.upload(&bytes)?, vec![n], DType::F32))
}

/// Pure-CPU half of loading one layer: dequantize (deferred inside every
/// cache closure, a cache HIT never touches the GGUF mmap) + requantize /
/// permute / interleave every weight this layer needs, each artifact
/// individually cache-checked. Touches neither `dev` nor any device buffer,
/// so this is exactly the per-layer unit `load_gguf` runs across a rayon
/// thread pool on a cache miss (`into_par_iter` over layer indices, see
/// that fn's doc). Under FFAI_LAGUNA_W4A8=1 the routed gate/up/down stacks
/// are left `None` here and built afterward, sequentially, by
/// `build_moe_w4a8_sequential` (see [`MoeFfnBytes::gateup`]'s doc for why).
#[allow(clippy::too_many_arguments)]
fn build_layer_bytes(g: &Gguf, cache: &CacheCtx, cfg: &LagunaConfig, i: usize, micro: bool, marlin: bool, moefuse: bool, w4a8: bool) -> Result<LayerBytes> {
    let p = format!("blk.{i}");
    let q_name = format!("{p}.attn_q.weight");
    let k_name = format!("{p}.attn_k.weight");
    let v_name = format!("{p}.attn_v.weight");
    let gate_name = format!("{p}.attn_gate.weight");

    // n_head is derived from wq's own GGUF dims (48 full-attention / 72
    // sliding for Laguna-S-2.1), not the LagunaConfig formula: see the
    // AttnWeights doc comment. Dims-only reads (no dequant) so this is cheap
    // regardless of cache state.
    let (n_q_rows, hidden_k) = tensor_dims(g, &q_name)?;
    let (n_kv_rows, _) = tensor_dims(g, &k_name)?;
    let (n_kv_rows_v, _) = tensor_dims(g, &v_name)?;
    let (n_g_rows, _) = tensor_dims(g, &gate_name)?;
    debug_assert_eq!(n_kv_rows, n_kv_rows_v, "wk/wv row count mismatch");
    let n_head = n_q_rows / cfg.head_dim;

    // FFAI_LAGUNA_MICRO=1: concatenate the four dequantized f32 slabs BEFORE
    // quantize_q4 (row order q,k,v,gate; all share k=hidden), so decode can
    // do one gemv_q4 instead of four. Only ONE of the two layouts is ever
    // built per layer, building both would double the resident
    // attention-projection weights (~2.4GB total).
    let proj = if micro {
        let n_rows = n_q_rows + n_kv_rows + n_kv_rows_v + n_g_rows;
        let wqkvg = q4dense_bytes_from_flat_cached(cache, &format!("{p}.attn_qkvg_fused"), n_rows, hidden_k, || {
            let mut concat = Vec::with_capacity(n_rows * hidden_k);
            concat.extend_from_slice(&g.dequant_f32(&q_name)?);
            concat.extend_from_slice(&g.dequant_f32(&k_name)?);
            concat.extend_from_slice(&g.dequant_f32(&v_name)?);
            concat.extend_from_slice(&g.dequant_f32(&gate_name)?);
            Ok(concat)
        })?;
        AttnProjBytes::Fused { wqkvg, n_q_rows, n_kv_rows, n_g_rows }
    } else {
        AttnProjBytes::Split {
            wq: q4dense_bytes_from_flat_cached(cache, &q_name, n_q_rows, hidden_k, || g.dequant_f32(&q_name))?,
            wk: q4dense_bytes_from_flat_cached(cache, &k_name, n_kv_rows, hidden_k, || g.dequant_f32(&k_name))?,
            wv: q4dense_bytes_from_flat_cached(cache, &v_name, n_kv_rows_v, hidden_k, || g.dequant_f32(&v_name))?,
            wgate: q4dense_bytes_from_flat_cached(cache, &gate_name, n_g_rows, hidden_k, || g.dequant_f32(&gate_name))?,
        }
    };

    // FFAI_LAGUNA_MARLIN=1: pack [wq;wk;wv] (gate excluded, see QkvMarlin's
    // doc) into a Marlin-tile-major MarlinDense for the tensor-core
    // batched-prefill GEMM, built ALONGSIDE (never instead of) the AttnProj
    // decode copy above. `g_proj` is a standalone plain Q4Dense so the
    // marlin prefill branch has gate access independent of whether `proj`
    // above resolved to Split or Fused.
    let qkv_marlin = if marlin {
        let n_rows_qkv = n_q_rows + n_kv_rows + n_kv_rows_v;
        if n_rows_qkv % 64 == 0 {
            let mat = marlindense_bytes_from_flat_cached(cache, &format!("{p}.attn_qkv"), n_rows_qkv, hidden_k, || {
                let mut concat = Vec::with_capacity(n_rows_qkv * hidden_k);
                concat.extend_from_slice(&g.dequant_f32(&q_name)?);
                concat.extend_from_slice(&g.dequant_f32(&k_name)?);
                concat.extend_from_slice(&g.dequant_f32(&v_name)?);
                Ok(concat)
            })?;
            Some(QkvMarlinBytes { mat, n_q_rows, n_kv_rows })
        } else {
            None
        }
    } else {
        None
    };
    let g_proj = if marlin {
        Some(q4dense_bytes_from_flat_cached(cache, &format!("{gate_name}.standalone"), n_g_rows, hidden_k, || g.dequant_f32(&gate_name))?)
    } else {
        None
    };

    let wo_name = format!("{p}.attn_output.weight");
    let (wo_m, wo_k) = tensor_dims(g, &wo_name)?;
    let (wo, wo_marlin) = q4dense_and_marlin_bytes_cached(cache, &wo_name, wo_m, wo_k, marlin, || g.dequant_f32(&wo_name))?;

    let attn_norm_name = format!("{p}.attn_norm.weight");
    let q_norm_name = format!("{p}.attn_q_norm.weight");
    let k_norm_name = format!("{p}.attn_k_norm.weight");
    let attn = AttnWeightsBytes {
        n_head,
        attn_norm: f32vec_bytes_cached(cache, &attn_norm_name, || g.dequant_f32(&attn_norm_name))?,
        proj,
        wo,
        q_norm: f32vec_bytes_cached(cache, &q_norm_name, || g.dequant_f32(&q_norm_name))?,
        k_norm: f32vec_bytes_cached(cache, &k_norm_name, || g.dequant_f32(&k_norm_name))?,
        qkv_marlin,
        g_proj,
        wo_marlin,
    };

    let ffn = if cfg.is_dense_layer(i) {
        // FFAI_LAGUNA_MARLIN=1: [gate;up] concat (2*n_ff_dense=24576,
        // 64-aligned) plus down, both Marlin-packed alongside the plain
        // per-projection Q4Dense copies decode/gemm_q4_mpp use.
        let gate_name2 = format!("{p}.ffn_gate.weight");
        let up_name2 = format!("{p}.ffn_up.weight");
        let down_name2 = format!("{p}.ffn_down.weight");
        let (ff_m, ff_k) = tensor_dims(g, &gate_name2)?;
        let gate = q4dense_bytes_from_flat_cached(cache, &gate_name2, ff_m, ff_k, || g.dequant_f32(&gate_name2))?;
        let up = q4dense_bytes_from_flat_cached(cache, &up_name2, ff_m, ff_k, || g.dequant_f32(&up_name2))?;
        let (down_m, down_k) = tensor_dims(g, &down_name2)?;
        let (down, down_marlin) = q4dense_and_marlin_bytes_cached(cache, &down_name2, down_m, down_k, marlin, || g.dequant_f32(&down_name2))?;
        let gateup_marlin = if marlin && (2 * ff_m) % 64 == 0 {
            let mat = marlindense_bytes_from_flat_cached(cache, &format!("{p}.ffn_gateup"), 2 * ff_m, ff_k, || {
                let mut concat = Vec::with_capacity(2 * ff_m * ff_k);
                concat.extend_from_slice(&g.dequant_f32(&gate_name2)?);
                concat.extend_from_slice(&g.dequant_f32(&up_name2)?);
                Ok(concat)
            })?;
            Some(GateUpMarlinBytes { mat, n_ff: ff_m })
        } else {
            None
        };
        let ffn_norm_name = format!("{p}.ffn_norm.weight");
        LayerFfnBytes::Dense(DenseFfnBytes {
            ffn_norm: f32vec_bytes_cached(cache, &ffn_norm_name, || g.dequant_f32(&ffn_norm_name))?,
            gate,
            up,
            down,
            gateup_marlin,
            down_marlin,
        })
    } else {
        let router_name = format!("{p}.ffn_gate_inp.weight");
        let router = f32vec_bytes_cached(cache, &router_name, || g.dequant_f32(&router_name))?;

        let gate_shexp = format!("{p}.ffn_gate_shexp.weight");
        let up_shexp = format!("{p}.ffn_up_shexp.weight");
        let down_shexp = format!("{p}.ffn_down_shexp.weight");

        // FFAI_LAGUNA_MICRO=1: repack the shared expert's gate+up as
        // single-expert (n_expert=1) stacks so decode can drive them through
        // the same moe_gather_q4_swiglu the routed experts use (top_k=1)
        // instead of two plain gemvs + a host-side swiglu. `down` stays a
        // plain Q4Dense either way. Only one layout is ever built per layer
        // (same memory-doubling rationale as AttnProj above).
        let shared_expert = if micro {
            let gate = q4_expert_stack_bytes_cached(cache, &format!("{gate_shexp}.se1"), 1, cfg.n_ff_shexp, cfg.hidden, || g.dequant_f32(&gate_shexp))?;
            let up = q4_expert_stack_bytes_cached(cache, &format!("{up_shexp}.se1"), 1, cfg.n_ff_shexp, cfg.hidden, || g.dequant_f32(&up_shexp))?;
            let (dm, dk) = tensor_dims(g, &down_shexp)?;
            let down = q4dense_bytes_from_flat_cached(cache, &down_shexp, dm, dk, || g.dequant_f32(&down_shexp))?;
            SharedExpertBytes::Fused(SharedExpertFusedBytes { gate, up, down })
        } else {
            let (gm, gk) = tensor_dims(g, &gate_shexp)?;
            let (um, uk) = tensor_dims(g, &up_shexp)?;
            let (dm, dk) = tensor_dims(g, &down_shexp)?;
            SharedExpertBytes::Split {
                gate: q4dense_bytes_from_flat_cached(cache, &gate_shexp, gm, gk, || g.dequant_f32(&gate_shexp))?,
                up: q4dense_bytes_from_flat_cached(cache, &up_shexp, um, uk, || g.dequant_f32(&up_shexp))?,
                down: q4dense_bytes_from_flat_cached(cache, &down_shexp, dm, dk, || g.dequant_f32(&down_shexp))?,
            }
        };

        // FFAI_LAGUNA_MARLIN=1: the shared expert's [gate;up] concat
        // (2*n_ff_shexp=2048, 64-aligned) plus down, Marlin-packed straight
        // from the shexp GGUF tensors, independent of whether
        // `shared_expert` above resolved to Split or Fused. The routed
        // per-expert stacks (gate/up/down just below) are NOT Marlin-packed:
        // prefill already drives them through the grouped-MMA GEMM
        // (`ops::moe_q4_grouped_mma_dev`), a different tensor-core kernel
        // this Marlin work doesn't target.
        let shared_marlin = if marlin {
            let (sm, sk) = tensor_dims(g, &gate_shexp)?;
            let (sdm, sdk) = tensor_dims(g, &down_shexp)?;
            if (2 * sm) % 64 == 0 && sdm % 64 == 0 {
                let mat = marlindense_bytes_from_flat_cached(cache, &format!("{p}.shexp_gateup"), 2 * sm, sk, || {
                    let mut concat = Vec::with_capacity(2 * sm * sk);
                    concat.extend_from_slice(&g.dequant_f32(&gate_shexp)?);
                    concat.extend_from_slice(&g.dequant_f32(&up_shexp)?);
                    Ok(concat)
                })?;
                let down = marlindense_bytes_from_flat_cached(cache, &format!("{down_shexp}.standalone"), sdm, sdk, || g.dequant_f32(&down_shexp))?;
                Some(SharedMarlinBytes { gateup: GateUpMarlinBytes { mat, n_ff: sm }, down })
            } else {
                None
            }
        } else {
            None
        };

        let gate_exps_name = format!("{p}.ffn_gate_exps.weight");
        let up_exps_name = format!("{p}.ffn_up_exps.weight");
        let down_exps_name = format!("{p}.ffn_down_exps.weight");

        // FFAI_LAGUNA_W4A8=1: the routed gate/up/down stacks are built
        // sequentially afterward (`build_moe_w4a8_sequential`), see
        // `MoeFfnBytes::gateup`'s doc.
        let (gateup, down) = if w4a8 {
            (None, None)
        } else if moefuse {
            // FFAI_LAGUNA_MOEFUSE=1: build ONLY the fused [gate;up] stack
            // (see RoutedGateUp::Fused's doc for the row layout), never both
            // layouts at once, the routed expert stacks are the bulk of
            // the model's resident memory and a second copy isn't
            // affordable.
            let per_expert = cfg.n_ff_exp * cfg.hidden;
            let n_expert = cfg.n_expert;
            let (n_ff_exp, hidden) = (cfg.n_ff_exp, cfg.hidden);
            let fused = q4_expert_stack_bytes_cached(cache, &format!("{p}.ffn_gateup_exps_fused"), n_expert, 2 * n_ff_exp, hidden, || {
                let gate_flat = g.dequant_f32(&gate_exps_name)?;
                let up_flat = g.dequant_f32(&up_exps_name)?;
                if gate_flat.len() != per_expert * n_expert || up_flat.len() != per_expert * n_expert {
                    return Err(Error::Msg(format!(
                        "ffn_gate/up_exps len mismatch gate={} up={} expected {} each (n_expert={n_expert} n_ff_exp={n_ff_exp} hidden={hidden})",
                        gate_flat.len(), up_flat.len(), per_expert * n_expert,
                    )));
                }
                let mut fused = Vec::with_capacity(2 * per_expert * n_expert);
                for e in 0..n_expert {
                    fused.extend_from_slice(&gate_flat[e * per_expert..(e + 1) * per_expert]);
                    fused.extend_from_slice(&up_flat[e * per_expert..(e + 1) * per_expert]);
                }
                Ok(fused)
            })?;
            let down = q4_expert_stack_bytes_cached(cache, &down_exps_name, cfg.n_expert, cfg.hidden, cfg.n_ff_exp, || g.dequant_f32(&down_exps_name))?;
            (Some(RoutedGateUpBytes::Fused(fused)), Some(down))
        } else {
            let gate = q4_expert_stack_bytes_cached(cache, &gate_exps_name, cfg.n_expert, cfg.n_ff_exp, cfg.hidden, || g.dequant_f32(&gate_exps_name))?;
            let up = q4_expert_stack_bytes_cached(cache, &up_exps_name, cfg.n_expert, cfg.n_ff_exp, cfg.hidden, || g.dequant_f32(&up_exps_name))?;
            let down = q4_expert_stack_bytes_cached(cache, &down_exps_name, cfg.n_expert, cfg.hidden, cfg.n_ff_exp, || g.dequant_f32(&down_exps_name))?;
            (Some(RoutedGateUpBytes::Split { gate, up }), Some(down))
        };

        let ffn_norm_name = format!("{p}.ffn_norm.weight");
        let bias_name = format!("{p}.exp_probs_b.bias");
        LayerFfnBytes::Moe(MoeFfnBytes {
            ffn_norm: f32vec_bytes_cached(cache, &ffn_norm_name, || g.dequant_f32(&ffn_norm_name))?,
            router,
            bias: f32vec_bytes_cached(cache, &bias_name, || g.dequant_f32(&bias_name))?,
            gateup,
            down,
            shared_expert,
            shared_marlin,
        })
    };

    Ok(LayerBytes { attn, ffn })
}

/// FFAI_LAGUNA_W4A8=1 sequential (non-rayon-parallel) build of one MoE
/// layer's routed gate/up/down. Mirrors the original inline `load_gguf`
/// logic exactly: dequantize each routed GGUF tensor ONCE and feed BOTH the
/// resident Q4 stack decode needs and the ADDITIONAL W4A8-packed stack
/// prefill needs (see [`MoeFfnW4a8`]'s doc), the Q4 side is routed through
/// [`CacheCtx`] (pure CPU; kept here rather than in the parallel
/// `build_layer_bytes` pass purely so it shares that one dequant with the
/// W4A8 pack below instead of paying for it twice) and the W4A8 pack itself
/// through [`pack_w4a8_expert_stack_cached`] (a GPU op, must run on this
/// thread, see that fn's doc). Not parallelized across layers: acceptable
/// since FFAI_LAGUNA_W4A8 is an opt-in, default-off, prefill-throughput
/// experiment, not the path the "fast load" goal targets.
fn build_moe_w4a8_sequential(dev: &dyn Device, g: &Gguf, cache: &CacheCtx, cfg: &LagunaConfig, i: usize) -> Result<(RoutedGateUp, Q4ExpertStack, Option<MoeFfnW4a8>)> {
    let p = format!("blk.{i}");
    let gate_exps_name = format!("{p}.ffn_gate_exps.weight");
    let up_exps_name = format!("{p}.ffn_up_exps.weight");
    let down_exps_name = format!("{p}.ffn_down_exps.weight");
    let moefuse = laguna_moefuse_enabled();
    // FFAI_LAGUNA_W4A8_ONLY=1: PREFILL BENCH MODE. Dual expert residency (Q4
    // for decode + mxfp4 for W4A8 prefill) is ~112GB and cannot fit; this
    // mode uploads a 1-expert DUMMY Q4 stack (decode would be wrong, never
    // run it in this mode) so the W4A8 prefill speed can be measured. The
    // real fix is a single shared expert format both paths read.
    let w4a8_only = std::env::var("FFAI_LAGUNA_W4A8_ONLY").map(|v| v == "1").unwrap_or(false);

    let gate_flat = g.dequant_f32(&gate_exps_name)?;
    let up_flat = g.dequant_f32(&up_exps_name)?;

    let (gateup, gateup_w4a8) = if moefuse {
        let per_expert = cfg.n_ff_exp * cfg.hidden;
        if gate_flat.len() != per_expert * cfg.n_expert || up_flat.len() != per_expert * cfg.n_expert {
            return Err(Error::Msg(format!(
                "ffn_gate/up_exps len mismatch gate={} up={} expected {} each (n_expert={} n_ff_exp={} hidden={})",
                gate_flat.len(), up_flat.len(), per_expert * cfg.n_expert, cfg.n_expert, cfg.n_ff_exp, cfg.hidden
            )));
        }
        let mut fused = Vec::with_capacity(2 * per_expert * cfg.n_expert);
        for e in 0..cfg.n_expert {
            fused.extend_from_slice(&gate_flat[e * per_expert..(e + 1) * per_expert]);
            fused.extend_from_slice(&up_flat[e * per_expert..(e + 1) * per_expert]);
        }
        let w4a8s = pack_w4a8_expert_stack_cached(dev, cache, &format!("{p}.ffn_gateup_exps_fused"), &fused, cfg.n_expert, 2 * cfg.n_ff_exp, cfg.hidden)?;
        let q4 = if w4a8_only {
            let b = q4_expert_stack_bytes_cached(cache, &format!("{p}.ffn_gateup_exps_fused.w4a8only1"), 1, 2 * cfg.n_ff_exp, cfg.hidden, || Ok(fused[..2 * per_expert].to_vec()))?;
            upload_expert_stack_bytes(dev, b)?
        } else {
            let b = q4_expert_stack_bytes_cached(cache, &format!("{p}.ffn_gateup_exps_fused"), cfg.n_expert, 2 * cfg.n_ff_exp, cfg.hidden, || Ok(fused))?;
            upload_expert_stack_bytes(dev, b)?
        };
        (RoutedGateUp::Fused(q4), Some(RoutedGateUpW4a8::Fused(w4a8s)))
    } else {
        let gate_b = q4_expert_stack_bytes_cached(cache, &gate_exps_name, cfg.n_expert, cfg.n_ff_exp, cfg.hidden, || Ok(gate_flat.clone()))?;
        let up_b = q4_expert_stack_bytes_cached(cache, &up_exps_name, cfg.n_expert, cfg.n_ff_exp, cfg.hidden, || Ok(up_flat.clone()))?;
        let gate_q4 = upload_expert_stack_bytes(dev, gate_b)?;
        let up_q4 = upload_expert_stack_bytes(dev, up_b)?;
        let gate_w4a8 = pack_w4a8_expert_stack_cached(dev, cache, &gate_exps_name, &gate_flat, cfg.n_expert, cfg.n_ff_exp, cfg.hidden)?;
        let up_w4a8 = pack_w4a8_expert_stack_cached(dev, cache, &up_exps_name, &up_flat, cfg.n_expert, cfg.n_ff_exp, cfg.hidden)?;
        (RoutedGateUp::Split { gate: gate_q4, up: up_q4 }, Some(RoutedGateUpW4a8::Split { gate: gate_w4a8, up: up_w4a8 }))
    };

    let down_flat = g.dequant_f32(&down_exps_name)?;
    let down_w4a8 = pack_w4a8_expert_stack_cached(dev, cache, &down_exps_name, &down_flat, cfg.n_expert, cfg.hidden, cfg.n_ff_exp)?;
    let down = if w4a8_only {
        let b = q4_expert_stack_bytes_cached(cache, &format!("{down_exps_name}.w4a8only1"), 1, cfg.hidden, cfg.n_ff_exp, || Ok(down_flat[..cfg.hidden * cfg.n_ff_exp].to_vec()))?;
        upload_expert_stack_bytes(dev, b)?
    } else {
        let b = q4_expert_stack_bytes_cached(cache, &down_exps_name, cfg.n_expert, cfg.hidden, cfg.n_ff_exp, || Ok(down_flat))?;
        upload_expert_stack_bytes(dev, b)?
    };

    // Both halves are always built together in this fn (it only ever runs
    // under FFAI_LAGUNA_W4A8=1), so `gateup_w4a8` is always `Some` here:
    // `.expect` just makes that invariant explicit instead of a silent
    // `.unwrap()`.
    let w4a8_moe = Some(MoeFfnW4a8 { gateup: gateup_w4a8.expect("w4a8 branch always builds a gateup pack"), down: down_w4a8 });
    Ok((gateup, down, w4a8_moe))
}

/// Sequential half of loading one layer: upload every byte blob
/// `build_layer_bytes` produced (or, under FFAI_LAGUNA_W4A8=1, that this fn
/// itself builds via `build_moe_w4a8_sequential`, see [`MoeFfnBytes::gateup`]'s
/// doc) and assemble the `Tensor`-based [`LagunaLayer`]. Never parallelized
/// across layers, `dev.upload`/device dispatch calls only ever happen on
/// the caller's thread (the loop in `load_gguf`), since backends aren't
/// guaranteed safe to drive concurrently from multiple threads.
fn upload_layer(dev: &dyn Device, g: &Gguf, cache: &CacheCtx, cfg: &LagunaConfig, i: usize, w4a8: bool, lb: LayerBytes) -> Result<LagunaLayer> {
    let attn = {
        let a = lb.attn;
        let proj = match a.proj {
            AttnProjBytes::Fused { wqkvg, n_q_rows, n_kv_rows, n_g_rows } => {
                AttnProj::Fused { wqkvg: upload_q4dense_bytes(dev, wqkvg)?, n_q_rows, n_kv_rows, n_g_rows }
            }
            AttnProjBytes::Split { wq, wk, wv, wgate } => AttnProj::Split {
                wq: upload_q4dense_bytes(dev, wq)?,
                wk: upload_q4dense_bytes(dev, wk)?,
                wv: upload_q4dense_bytes(dev, wv)?,
                wgate: upload_q4dense_bytes(dev, wgate)?,
            },
        };
        let wo = upload_q4dense_bytes(dev, a.wo)?;
        let qkv_marlin = a.qkv_marlin.map(|qm| -> Result<QkvMarlin> {
            Ok(QkvMarlin { mat: upload_marlindense_bytes(dev, qm.mat)?, n_q_rows: qm.n_q_rows, n_kv_rows: qm.n_kv_rows })
        }).transpose()?;
        let g_proj = a.g_proj.map(|b| upload_q4dense_bytes(dev, b)).transpose()?;
        let wo_marlin = a.wo_marlin.map(|b| upload_marlindense_bytes(dev, b)).transpose()?;
        AttnWeights {
            n_head: a.n_head,
            attn_norm: upload_f32_bytes(dev, a.attn_norm)?,
            proj,
            wo,
            q_norm: upload_f32_bytes(dev, a.q_norm)?,
            k_norm: upload_f32_bytes(dev, a.k_norm)?,
            qkv_marlin,
            g_proj,
            wo_marlin,
        }
    };

    let ffn = match lb.ffn {
        LayerFfnBytes::Dense(d) => {
            let gateup_marlin = d.gateup_marlin.map(|gm| -> Result<GateUpMarlin> {
                Ok(GateUpMarlin { mat: upload_marlindense_bytes(dev, gm.mat)?, n_ff: gm.n_ff })
            }).transpose()?;
            LayerFfn::Dense(DenseFfn {
                ffn_norm: upload_f32_bytes(dev, d.ffn_norm)?,
                gate: upload_q4dense_bytes(dev, d.gate)?,
                up: upload_q4dense_bytes(dev, d.up)?,
                down: upload_q4dense_bytes(dev, d.down)?,
                gateup_marlin,
                down_marlin: d.down_marlin.map(|b| upload_marlindense_bytes(dev, b)).transpose()?,
            })
        }
        LayerFfnBytes::Moe(m) => {
            let shared_expert = match m.shared_expert {
                SharedExpertBytes::Split { gate, up, down } => SharedExpert::Split {
                    gate: upload_q4dense_bytes(dev, gate)?,
                    up: upload_q4dense_bytes(dev, up)?,
                    down: upload_q4dense_bytes(dev, down)?,
                },
                SharedExpertBytes::Fused(f) => {
                    let zero_idx = Tensor::new(dev.upload(&0u32.to_le_bytes())?, vec![1], DType::U32);
                    SharedExpert::Fused(SharedExpertFused {
                        gate: upload_expert_stack_bytes(dev, f.gate)?,
                        up: upload_expert_stack_bytes(dev, f.up)?,
                        down: upload_q4dense_bytes(dev, f.down)?,
                        zero_idx,
                    })
                }
            };
            let shared_marlin = m.shared_marlin.map(|sm| -> Result<SharedMarlin> {
                Ok(SharedMarlin {
                    gateup: GateUpMarlin { mat: upload_marlindense_bytes(dev, sm.gateup.mat)?, n_ff: sm.gateup.n_ff },
                    down: upload_marlindense_bytes(dev, sm.down)?,
                })
            }).transpose()?;

            let (gateup, down, w4a8_moe) = if w4a8 {
                build_moe_w4a8_sequential(dev, g, cache, cfg, i)?
            } else {
                let gateup = match m.gateup.expect("non-W4A8 build_layer_bytes always sets gateup") {
                    RoutedGateUpBytes::Fused(q) => RoutedGateUp::Fused(upload_expert_stack_bytes(dev, q)?),
                    RoutedGateUpBytes::Split { gate, up } => {
                        RoutedGateUp::Split { gate: upload_expert_stack_bytes(dev, gate)?, up: upload_expert_stack_bytes(dev, up)? }
                    }
                };
                let down = upload_expert_stack_bytes(dev, m.down.expect("non-W4A8 build_layer_bytes always sets down"))?;
                (gateup, down, None)
            };

            LayerFfn::Moe(MoeFfn {
                ffn_norm: upload_f32_bytes(dev, m.ffn_norm)?,
                router: Tensor::new(dev.upload(&m.router)?, vec![cfg.n_expert, cfg.hidden], DType::F32),
                bias: upload_f32_bytes(dev, m.bias)?,
                gateup,
                down,
                shared_expert,
                shared_marlin,
                w4a8: w4a8_moe,
            })
        }
    };

    Ok(LagunaLayer { attn, ffn })
}

/// Load Laguna-S-2.1 straight from a GGUF file: parse `laguna.*` metadata for
/// the config, then stream every tensor through `dequant_f32` → requantize
/// (Q4 for projections/experts, dense f32 for embeddings/norms/router) →
/// upload, dropping the f32 immediately after each tensor.
///
/// Every uploaded byte blob goes through an on-disk [`CacheCtx`] first (see
/// that struct's doc): a warm reload of an unchanged GGUF under the same
/// `FFAI_LAGUNA_*` flags skips straight to `read file -> upload` for every
/// weight, instead of redoing the dequant+requant work. On a cache MISS, the
/// pure-CPU half of that work (`build_layer_bytes`) runs across a rayon
/// thread pool, one task per layer: layers are fully independent (no shared
/// mutable state besides the cache's own atomic hit/miss counters and the
/// read-only `g`/`cfg`), so this is an embarrassingly parallel fan-out.
/// Every `dev.upload` call happens afterward, sequentially, in layer order
/// (`upload_layer`), backends aren't guaranteed safe to drive from multiple
/// threads at once, so nothing in the parallel phase ever touches `dev`.
pub fn load_gguf(dev: &dyn Device, g: &Gguf) -> Result<LagunaModel> {
    let load_t0 = std::time::Instant::now();
    let cfg = LagunaConfig::from_gguf(g);
    let cache = CacheCtx::new(g);

    // token_embd / output are [vocab, hidden] (GGUF fastest-first dims
    // already give the flat data in row-major [vocab, hidden] order, see
    // `dims_out_in`).
    let tok_embd = {
        let bytes = f32vec_bytes_cached(&cache, "token_embd.weight", || g.dequant_f32("token_embd.weight"))?;
        Tensor::new(dev.upload(&bytes)?, vec![cfg.vocab, cfg.hidden], DType::F32)
    };

    let out_norm = {
        let bytes = f32vec_bytes_cached(&cache, "output_norm.weight", || g.dequant_f32("output_norm.weight"))?;
        upload_f32_bytes(dev, bytes)?
    };
    let head_name = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
    let lm_head = {
        let (qs_bytes, sc_bytes, m, k) = q8head_bytes_cached(&cache, g, head_name, head_name)?;
        Q8Head {
            qs: Tensor::new(dev.upload(&qs_bytes)?, vec![qs_bytes.len() / 4], DType::U32),
            // gemv_q8 reads f32 scales (unlike the q4 family's f16), matching
            // the proven resident-Q8 loader path.
            scales: Tensor::new(dev.upload(&sc_bytes)?, vec![sc_bytes.len() / 4], DType::F32),
            m,
            k,
        }
    };
    // Tiny process-lifetime constant, not worth routing through the cache.
    let one = Tensor::new(dev.upload(&1.0f32.to_le_bytes())?, vec![1], DType::F32);

    let micro = laguna_micro_enabled();
    let marlin = laguna_marlin_enabled();
    let moefuse = laguna_moefuse_enabled();
    let w4a8 = laguna_w4a8_enabled();

    // Pure-CPU half, rayon-parallel WITHIN a bounded window of layers.
    // Building all 48 layers before uploading holds tens of GB of blob bytes
    // plus each worker's multi-GB f32 dequant transients at once, which
    // OOM-kills the process on the shared 128GB. Window of 4: at most 4
    // layers' transients + blobs in flight, uploaded and dropped before the
    // next window starts. FFAI_LAGUNA_LOAD_WINDOW overrides.
    let window: usize = std::env::var("FFAI_LAGUNA_LOAD_WINDOW")
        .ok().and_then(|v| v.parse().ok()).filter(|&w| w >= 1).unwrap_or(4);
    let mut layers = Vec::with_capacity(cfg.n_layers);
    let mut start_l = 0usize;
    while start_l < cfg.n_layers {
        let end_l = (start_l + window).min(cfg.n_layers);
        let batch: Vec<LayerBytes> = (start_l..end_l)
            .into_par_iter()
            .map(|i| build_layer_bytes(g, &cache, &cfg, i, micro, marlin, moefuse, w4a8))
            .collect::<Result<Vec<_>>>()?;
        // Sequential half: every dev.upload / device dispatch call, in order.
        for (off, lb) in batch.into_iter().enumerate() {
            layers.push(upload_layer(dev, g, &cache, &cfg, start_l + off, w4a8, lb)?);
        }
        start_l = end_l;
    }

    let (hits, misses) = cache.stats();
    eprintln!(
        "Laguna: load done in {:.2}s ({hits} cache hits, {misses} misses, cache dir {})",
        load_t0.elapsed().as_secs_f64(),
        cache.dir_display(),
    );

    Ok(LagunaModel { cfg, tok_embd, layers, out_norm, lm_head, one })
}

/// One decode layer: RMSNorm → QKV proj (+ per-head gate proj on the same
/// normed input) → per-head QK-RMSNorm → RoPE → KV-cache append → SDPA →
/// per-head softplus gate → o_proj → residual → RMSNorm → FFN (dense or
/// MoE) → residual. `pos` is the token's absolute sequence position.
#[allow(clippy::too_many_arguments)]
/// FFAI_LAGUNA_DEBUG=2: dump a tensor's stats (first-bad-op bisection inside
/// a layer). No-op cost is one env read per call site when disabled upstream.
fn dbg_stats(dev: &dyn Device, tag: &str, t: &Tensor) {
    let n = t.elem_count();
    let mut b = vec![0u8; n * 4];
    if dev.synchronize().is_err() || dev.download(t.buffer.as_ref(), &mut b).is_err() {
        eprintln!("    {tag}: <download failed>");
        return;
    }
    let f: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let n_nan = f.iter().filter(|v| v.is_nan()).count();
    let l2 = f.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
    let amax = f.iter().filter(|v| v.is_finite()).fold(0f32, |a, &v| a.max(v.abs()));
    eprintln!("    {tag}: n={n} |x|2={l2:.3e} amax={amax:.3e} nan={n_nan} head={:?}", &f[..4.min(n)]);
}

/// Per-step device buffers that make `decode_layer` CUDA-graph-capturable:
/// a captured graph replays the exact same recorded kernel launches every
/// token, so any value that changes per token (RoPE position, KV length)
/// must be read from a buffer whose CONTENTS the caller refreshes between
/// replays, rather than passed as a scalar baked in at capture time. The
/// buffer POINTERS are fixed for the lifetime of a `LagunaDecodeCtx`; only
/// the u32 values inside them change. `n_kv_full`/`n_kv_sliding` hold
/// `pos+1` / `min(pos+1, sliding_window)` respectively, which one a given
/// layer reads is chosen at call time by `cfg.is_full_layer(layer_idx)`,
/// which never changes across replays, so the choice of POINTER is itself
/// capture-stable even though the VALUE behind it isn't.
struct GraphBufs<'a> {
    n_kv_full: &'a Tensor,
    n_kv_sliding: &'a Tensor,
}

#[allow(clippy::too_many_arguments)]
fn decode_layer(
    dev: &dyn Device,
    cfg: &LagunaConfig,
    layer_idx: usize,
    w: &LagunaLayer,
    kv: &KvSlot,
    x: &Tensor,
    pos: u32,
    full_posbuf: &Tensor,
    sliding_posbuf: &Tensor,
    one: &Tensor,
    graph_bufs: Option<&GraphBufs>,
    micro: bool,
) -> Result<Tensor> {
    let deep_dbg = layer_idx == 0
        && std::env::var("FFAI_LAGUNA_DEBUG").map(|v| v == "2").unwrap_or(false);
    let hd = cfg.head_dim;
    let n_head = w.attn.n_head;
    let n_kv_heads = cfg.n_kv_heads;
    let is_full = cfg.is_full_layer(layer_idx);
    let rope = cfg.rope_for(layer_idx);
    let heads_per_group = (n_head / n_kv_heads) as u32;

    // ── attention ────────────────────────────────────────────────────
    if deep_dbg { dbg_stats(dev, "x_in", x); }
    let h = ops::rms_norm(dev, x, &w.attn.attn_norm, cfg.eps)?;
    if deep_dbg { dbg_stats(dev, "attn_norm", &h); }
    // g_proj consumes the SAME pre-attention normed input as q/k/v (not the
    // attention output), matches the reference engine's graph exactly.
    //
    // FFAI_LAGUNA_MICRO=1 (fusion 1): one gemv_q4 on the concatenated
    // [wq;wk;wv;wgate] weight instead of four separate gemvs, then four
    // device-side slices (cheap 36KB/4KB/4KB/0.2KB copies) split the result
    // back into q/k/v/gate_raw. `ops::slice` is used rather than zero-copy
    // Tensor offset views because the gemv/gather kernels bind buffer BASE
    // pointers and ignore Tensor.offset.
    let (q, k, v, gate_raw) = match &w.attn.proj {
        AttnProj::Fused { wqkvg, n_q_rows, n_kv_rows, n_g_rows } => {
            let qkvg = wqkvg.gemv(dev, &h)?;
            let q = ops::slice(dev, &qkvg, 0, *n_q_rows)?;
            let k = ops::slice(dev, &qkvg, *n_q_rows, *n_kv_rows)?;
            let v = ops::slice(dev, &qkvg, n_q_rows + n_kv_rows, *n_kv_rows)?;
            let gate_raw = ops::slice(dev, &qkvg, n_q_rows + 2 * n_kv_rows, *n_g_rows)?;
            (q, k, v, gate_raw)
        }
        AttnProj::Split { wq, wk, wv, wgate } => {
            let q = wq.gemv(dev, &h)?;
            let k = wk.gemv(dev, &h)?;
            let v = wv.gemv(dev, &h)?;
            let gate_raw = wgate.gemv(dev, &h)?; // [n_head]
            (q, k, v, gate_raw)
        }
    };
    if deep_dbg {
        dbg_stats(dev, "q_proj", &q);
        dbg_stats(dev, "k_proj", &k);
        dbg_stats(dev, "v_proj", &v);
        dbg_stats(dev, "g_proj", &gate_raw);
    }

    let q = q.reshaped(vec![n_head, hd]);
    let k = k.reshaped(vec![n_kv_heads, hd]);
    let v = v.reshaped(vec![n_kv_heads, hd]);

    // Per-head QK RMSNorm (head_dim-level), before RoPE.
    let q = ops::rms_norm(dev, &q, &w.attn.q_norm, cfg.eps)?;
    let k = ops::rms_norm(dev, &k, &w.attn.k_norm, cfg.eps)?;
    if deep_dbg {
        dbg_stats(dev, "q_norm", &q);
        dbg_stats(dev, "k_norm", &k);
    }

    // YaRN (full layers) / plain (sliding layers) partial-rotary RoPE.
    // Graph mode reads `position` from the (capture-stable-pointer) posbuf
    // instead of taking it as a scalar baked in at capture time, same math,
    // see `ops::rope_yarn_partial_posbuf`'s doc.
    let (q, k) = if graph_bufs.is_some() {
        let posbuf = if is_full { full_posbuf } else { sliding_posbuf };
        let q = ops::rope_yarn_partial_posbuf(
            dev, &q, posbuf, rope.n_rot, rope.theta, rope.freq_scale, rope.ext_factor,
            rope.attn_factor, rope.beta_fast, rope.beta_slow, rope.n_orig_ctx,
        )?;
        let k = ops::rope_yarn_partial_posbuf(
            dev, &k, posbuf, rope.n_rot, rope.theta, rope.freq_scale, rope.ext_factor,
            rope.attn_factor, rope.beta_fast, rope.beta_slow, rope.n_orig_ctx,
        )?;
        (q, k)
    } else {
        let q = ops::rope_yarn_partial(
            dev, &q, pos, rope.n_rot, rope.theta, rope.freq_scale, rope.ext_factor,
            rope.attn_factor, rope.beta_fast, rope.beta_slow, rope.n_orig_ctx,
        )?;
        let k = ops::rope_yarn_partial(
            dev, &k, pos, rope.n_rot, rope.theta, rope.freq_scale, rope.ext_factor,
            rope.attn_factor, rope.beta_fast, rope.beta_slow, rope.n_orig_ctx,
        )?;
        (q, k)
    };
    if deep_dbg {
        dbg_stats(dev, "q_rope", &q);
        dbg_stats(dev, "k_rope", &k);
    }

    // KV cache append: full layers grow (posbuf = pos), sliding layers write
    // a ring slot (posbuf = pos % sliding_window). RoPE already used the
    // ABSOLUTE position above, so ring order doesn't affect the attention
    // math, only which slot gets overwritten.
    let posbuf = if is_full { full_posbuf } else { sliding_posbuf };
    ops::kv_append(dev, &k, &kv.k, posbuf, hd, kv.cap, n_kv_heads * hd)?;
    ops::kv_append(dev, &v, &kv.v, posbuf, hd, kv.cap, n_kv_heads * hd)?;

    // Graph mode reads the filled KV length from a (capture-stable-pointer)
    // device buffer instead of a baked-in scalar, same dense-causal math,
    // see `ops::sdpa_decode_nbuf`'s doc.
    let attn = if let Some(gb) = graph_bufs {
        let n_kv_buf = if is_full { gb.n_kv_full } else { gb.n_kv_sliding };
        ops::sdpa_decode_nbuf(
            dev, &q, &kv.k, &kv.v, hd, n_kv_buf, kv.cap as u32, heads_per_group, cfg.attn_scale(),
        )?
    } else {
        let n_kv_used = if is_full { pos + 1 } else { (pos + 1).min(cfg.sliding_window as u32) };
        ops::sdpa_decode(
            dev, &q, &kv.k, &kv.v, hd, n_kv_used, kv.cap as u32, heads_per_group, cfg.attn_scale(),
        )?
    }; // [n_head, hd]
    if deep_dbg { dbg_stats(dev, "sdpa", &attn); }

    // Per-head softplus gate, before o_proj.
    let gated = ops::gate_softplus_mul_perhead(dev, &attn, &gate_raw)?;
    let gated_flat = gated.reshaped(vec![n_head * hd]);
    if deep_dbg { dbg_stats(dev, "gated", &gated); }
    // FFAI_LAGUNA_MICRO=1 (fusion 3): copy x into a fresh accumulator (there
    // is no tensor-clone op; `ops::slice(dev, x, 0, hidden)` is a device copy)
    // then `wo.gemv_accum` computes x1 = x + wo·gated in one kernel, instead
    // of materializing `o = wo.gemv(...)` and adding it separately. Same op
    // count (copy+accum vs gemv+add), this fusion trades materializing `o`
    // for materializing the copy, so it wins only if the accum kernel avoids
    // enough bandwidth to offset that; benchmarking decides, not this diff.
    let x1 = if micro {
        let acc = ops::slice(dev, x, 0, cfg.hidden)?;
        w.attn.wo.gemv_accum(dev, &gated_flat, &acc, one)?;
        if deep_dbg { dbg_stats(dev, "x_attn_res", &acc); }
        acc
    } else {
        let o = w.attn.wo.gemv(dev, &gated_flat)?;
        let x1 = ops::add(dev, x, &o)?;
        if deep_dbg {
            dbg_stats(dev, "o_proj", &o);
            dbg_stats(dev, "x_attn_res", &x1);
        }
        x1
    };

    // ── FFN ──────────────────────────────────────────────────────────
    let x2 = match &w.ffn {
        LayerFfn::Dense(f) => {
            let h2 = ops::rms_norm(dev, &x1, &f.ffn_norm, cfg.eps)?;
            let gate = f.gate.gemv(dev, &h2)?;
            let up = f.up.gemv(dev, &h2)?;
            let act = ops::swiglu(dev, &gate, &up)?;
            let down = f.down.gemv(dev, &act)?;
            if deep_dbg {
                dbg_stats(dev, "ffn_norm", &h2);
                dbg_stats(dev, "ffn_act", &act);
                dbg_stats(dev, "ffn_down", &down);
            }
            ops::add(dev, &x1, &down)?
        }
        LayerFfn::Moe(f) => {
            let moe_dbg = (1..=2).contains(&layer_idx)
                && std::env::var("FFAI_LAGUNA_DEBUG").map(|v| v == "2").unwrap_or(false);
            let h2 = ops::rms_norm(dev, &x1, &f.ffn_norm, cfg.eps)?;
            let logits = ops::gemv(dev, &f.router, &h2)?; // [n_expert]
            if moe_dbg {
                dbg_stats(dev, &format!("moe{layer_idx}_x_in"), x);
                dbg_stats(dev, &format!("moe{layer_idx}_x_attn_res"), &x1);
                dbg_stats(dev, &format!("moe{layer_idx}_ffn_norm"), &h2);
                dbg_stats(dev, &format!("moe{layer_idx}_router_logits"), &logits);
            }
            let (idx, wts) = ops::moe_router_device(
                dev, &logits, &f.bias, cfg.n_expert, cfg.n_expert_used, cfg.routed_scaling,
            )?;
            if moe_dbg {
                let mut wb = vec![0u8; cfg.n_expert_used * 4];
                let _ = dev.synchronize();
                let _ = dev.download(wts.buffer.as_ref(), &mut wb);
                let w: Vec<f32> = wb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                let mut ib = vec![0u8; cfg.n_expert_used * 4];
                let _ = dev.download(idx.buffer.as_ref(), &mut ib);
                let iv: Vec<u32> = ib.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                eprintln!("    moe{layer_idx}_topk idx={iv:?} wts={w:?}");
            }

            // Fully device-side expert path: gather-GEMV the top_k gate and up
            // stacks (the gather kernels index the expert slab by `idx` inside
            // the kernel; a byte-offset Tensor view does NOT work here because
            // dispatch binds buffer BASE pointers and drops Tensor.offset),
            // SwiGLU elementwise on the [top_k*inter] pair, gather the down
            // stack, then router-weighted sum into the accumulator.
            //
            // FFAI_LAGUNA_FUSE=1 replaces the two gathers + host swiglu with
            // one fused gate+up+SwiGLU gather kernel: same `h2` read once
            // instead of twice, one launch instead of two gathers plus the
            // elementwise pass. Default (unset) keeps the unfused path. Only
            // available for RoutedGateUp::Split: `moe_gather_q4_swiglu` reads
            // gate and up as two SEPARATE `[n_exp*inter, hid]` stacks with
            // `row = e*inter + local` indexing, which the fused stack's
            // per-expert-interleaved rows do not satisfy.
            //
            // RoutedGateUp::Fused (FFAI_LAGUNA_MOEFUSE=1) instead reads the
            // ONE fused stack with a single moe_gather_q4 call at
            // inter=2*n_ff_exp (gathers both gate and up rows for the
            // selected experts in one launch), then splits the result's
            // columns into gate/up with extract_cols and runs the same
            // host-orchestrated swiglu as the pre-FFAI_LAGUNA_FUSE path
            // above, see RoutedGateUp::Fused's doc for the tradeoff this
            // implies (loses the fused-gather micro-win, about 1 tok/s).
            let act = match &f.gateup {
                RoutedGateUp::Fused(gu) => {
                    let inter = 2 * cfg.n_ff_exp;
                    let both = ops::moe_gather_q4(dev, &gu.qs, &gu.scales, &h2, &idx, cfg.n_expert_used, inter, cfg.hidden)?;
                    let gates = ops::extract_cols(dev, &both, cfg.n_expert_used, inter, 0, cfg.n_ff_exp)?;
                    let ups = ops::extract_cols(dev, &both, cfg.n_expert_used, inter, cfg.n_ff_exp, cfg.n_ff_exp)?;
                    ops::swiglu(dev, &gates, &ups)?
                }
                RoutedGateUp::Split { gate, up } => {
                    if std::env::var("FFAI_LAGUNA_FUSE").map(|v| v == "1").unwrap_or(false) {
                        ops::moe_gather_q4_swiglu(
                            dev, &gate.qs, &gate.scales, &up.qs, &up.scales, &h2, &idx,
                            cfg.n_expert_used, cfg.n_ff_exp, cfg.hidden,
                        )?
                    } else {
                        let gates = ops::moe_gather_q4(dev, &gate.qs, &gate.scales, &h2, &idx, cfg.n_expert_used, cfg.n_ff_exp, cfg.hidden)?;
                        let ups = ops::moe_gather_q4(dev, &up.qs, &up.scales, &h2, &idx, cfg.n_expert_used, cfg.n_ff_exp, cfg.hidden)?;
                        ops::swiglu(dev, &gates, &ups)?
                    }
                }
            };
            let downs = ops::moe_gather_down(dev, &f.down.qs, &f.down.scales, &act, &idx, cfg.n_expert_used, cfg.n_ff_exp, cfg.hidden)?;
            let acc = Tensor::new(dev.alloc_zeroed(cfg.hidden * 4)?, vec![cfg.hidden], DType::F32);
            ops::moe_weighted_sum(dev, &downs, &wts, &acc, cfg.hidden, cfg.n_expert_used)?;

            if moe_dbg { dbg_stats(dev, &format!("moe{layer_idx}_acc_routed"), &acc); }

            // Always-on shared expert: same accumulator, unweighted (scale = 1,
            // the model-wide cached `one` scalar, no gating/router weight).
            //
            // FFAI_LAGUNA_MICRO=1 (fusion 2): the gate+up projections were
            // repacked at load time as single-expert (n_expert=1) stacks, so
            // one moe_gather_q4_swiglu call (top_k=1, the persistent
            // zero_idx buffer selecting "expert 0") replaces the separate
            // gate gemv + up gemv + host-orchestrated swiglu. `down` is
            // unchanged either way (plain gemv_accum into the routed acc).
            match &f.shared_expert {
                SharedExpert::Fused(se) => {
                    let sact = ops::moe_gather_q4_swiglu(
                        dev, &se.gate.qs, &se.gate.scales, &se.up.qs, &se.up.scales,
                        &h2, &se.zero_idx, 1, cfg.n_ff_shexp, cfg.hidden,
                    )?;
                    se.down.gemv_accum(dev, &sact, &acc, one)?;
                }
                SharedExpert::Split { gate, up, down } => {
                    let sgate = gate.gemv(dev, &h2)?;
                    let sup = up.gemv(dev, &h2)?;
                    let sact = ops::swiglu(dev, &sgate, &sup)?;
                    down.gemv_accum(dev, &sact, &acc, one)?;
                }
            }

            if moe_dbg { dbg_stats(dev, &format!("moe{layer_idx}_acc_final"), &acc); }
            ops::add(dev, &x1, &acc)?
        }
    };

    Ok(x2)
}

/// One BATCHED prefill layer: `T` prompt-token rows processed in one forward
/// instead of `T` sequential [`decode_layer`] calls. Same op sequence as
/// `decode_layer` (RMSNorm → QKV(+gate) proj → per-head QK-RMSNorm → RoPE →
/// KV-cache write → SDPA → per-head softplus gate → o_proj → residual →
/// RMSNorm → FFN (dense or MoE) → residual), just every op runs on `[T, ...]`
/// batched tensors through the batched-prefill ops (`gemm_q4_mpp`,
/// `rope_yarn_partial_many`, `kv_append_many`, `sdpa_multi_tc_varlen`,
/// `gate_softplus_mul_perhead_many`, the on-device MoE
/// route+sort+grouped-GEMM+scatter pipeline) instead of the per-token gemv
/// family. `x` is `[T, hidden]`; `start` is the ABSOLUTE sequence position of
/// row 0 in this chunk (`positions[i] == start + i`, uploaded once by the
/// caller and shared across every layer of this chunk).
///
/// Full-attention layers write straight into `kv`'s persistent growing cache
/// (batched writes there are safe, nothing is evicted) and attend the full
/// causal prefix (`seg_lo` all-zero). Sliding-window layers write into
/// `scratch`'s LINEAR per-layer buffer instead (see [`LagunaPrefillScratch`]'s
/// doc for why the ring can't be batch-written) and attend a per-row window
/// `seg_lo[i] = max(0, (start+i) - (sliding_window-1))`.
#[allow(clippy::too_many_arguments)]

/// FFAI_LAGUNA_PP_PROF=1: coarse stage profile of the prefill path. psync
/// points synchronize, so the elapsed time of each sync attributes the queued
/// work since the previous point to the CURRENT tag. Coarse but names the wall.
static PROF: std::sync::Mutex<Option<std::collections::BTreeMap<String, std::time::Duration>>> =
    std::sync::Mutex::new(None);
fn prof_add(tag: &str, d: std::time::Duration) {
    let mut g = PROF.lock().unwrap();
    let m = g.get_or_insert_with(Default::default);
    *m.entry(tag.to_string()).or_default() += d;
}
/// Print and reset the accumulated stage profile.
pub fn prof_report() {
    if let Some(m) = PROF.lock().unwrap().take() {
        let total: f64 = m.values().map(|d| d.as_secs_f64()).sum();
        eprintln!("  prefill stage profile (sync-attributed, total {total:.3}s):");
        let mut rows: Vec<_> = m.into_iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        for (tag, d) in rows {
            eprintln!("    {tag:<20} {:>8.3}s  {:>5.1}%", d.as_secs_f64(), 100.0 * d.as_secs_f64() / total.max(1e-9));
        }
    }
}

fn prefill_layer(
    dev: &dyn Device,
    cfg: &LagunaConfig,
    layer_idx: usize,
    w: &LagunaLayer,
    kv: &LagunaKvCache,
    scratch: &LagunaPrefillScratch,
    x: &Tensor, // [T, hidden] f32
    t: usize,
    start: usize,
    positions: &Tensor, // [T] u32, positions[i] = start + i
    // FFAI_LAGUNA_MARLIN=1: a persistent all-zero `u32` buffer with at least
    // `t` elements, shared by every Marlin call in this layer (and every
    // other layer/chunk of the same `prefill()` call, see that fn's doc).
    // `None` when the env is unset; `Some` implies "check each per-weight
    // `Option<MarlinDense>` before using it" since an individual weight may
    // still be `None` (dimension not 64-aligned).
    marlin_zero_idx: Option<&Tensor>,
) -> Result<Tensor> {
    let hd = cfg.head_dim;
    let hidden = cfg.hidden;
    let n_head = w.attn.n_head;
    let n_kv_heads = cfg.n_kv_heads;
    let is_full = cfg.is_full_layer(layer_idx);
    let rope = cfg.rope_for(layer_idx);
    let heads_per_group = (n_head / n_kv_heads) as u32;
    // FFAI_LAGUNA_PREFILL_SYNC=1: synchronize after each stage so an async
    // device fault surfaces AT the guilty op instead of a later module load.
    let prof = std::env::var("FFAI_LAGUNA_PP_PROF").is_ok();
    let psync = |tag: &str| -> Result<()> {
        if prof || std::env::var("FFAI_LAGUNA_PREFILL_SYNC").is_ok() {
            let t0 = std::time::Instant::now();
            dev.synchronize().map_err(|e| {
                Error::Msg(format!("prefill fault layer {layer_idx} stage {tag}: {e:?}"))
            })?;
            if prof {
                prof_add(tag, t0.elapsed());
            }
        }
        Ok(())
    };

    // ── attention ────────────────────────────────────────────────────
    let h = ops::rms_norm(dev, x, &w.attn.attn_norm, cfg.eps)
        .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} attn_norm: {e:?}")))?; // [T, hidden]
    psync("attn_norm")?;

    // g_proj consumes the SAME pre-attention normed input as q/k/v, exactly
    // like decode_layer. FFAI_LAGUNA_MICRO=1 (Fused): one gemm_q4_mpp on the
    // concatenated [wq;wk;wv;wgate] weight, then COLUMN-range extraction
    // (`ops::extract_cols`, NOT `ops::slice`, which only slices a flat 1-D
    // buffer; the fused output here is `[T, n_rows_total]` row-major, so each
    // sub-projection is a strided column range, not a contiguous byte range).
    let (q, k, v, gate_raw) = if let (Some(zero_idx), Some(qkvm)) = (marlin_zero_idx, w.attn.qkv_marlin.as_ref()) {
        // FFAI_LAGUNA_MARLIN=1: tensor-core Marlin GEMM on the [wq;wk;wv]
        // concat instead of gemm_q4_mpp's scattered-nibble-read path. Marlin
        // consumes/emits f16. `extract_cols` is now f16-native, so extract
        // straight out of `qkv_f16` instead of casting the whole [T,
        // n_rows_qkv] result to f32 first, the extraction itself then moves
        // half the bytes (f16 in/out vs f32 in/out). Every one of q/k/v
        // still ends up cast to f32 individually right after, though:
        // q/k RMSNorm reads `w.attn.q_norm`/`k_norm`, which are loaded
        // f32-only (no f16 weight copy exists, see `AttnWeights`),
        // `rope_yarn_partial_many`'s raw-CUDA kernel is hardcoded `float*`
        // (f32 only, see its doc), and the persistent KV cache
        // (`LagunaKvCache`/`LagunaPrefillScratch`) is allocated f32-only,
        // so kv_append_many's destination forces v to f32 too. Net effect:
        // same total cast bytes as before (one f32-only chain either way),
        // but the extract itself is now cheaper, and it's one whole-buffer
        // cast fewer to schedule.
        // g_proj stays on gemm_q4_mpp, its 48/72 rows are never 64-aligned
        //, reading the standalone `g_proj` Q4Dense built for exactly this
        // (independent of whether `proj` above resolved to Split or Fused).
        let h_f16 = ops::cast_f32_f16(dev, &h)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} marlin qkv cast_f16: {e:?}")))?;
        let qkv_f16 = qkvm.mat.call(dev, &h_f16, zero_idx, t)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} marlin qkv gemm: {e:?}")))?;
        psync("qkv_proj_marlin")?;
        let n_rows_qkv = qkvm.n_q_rows + 2 * qkvm.n_kv_rows;
        let q_f16 = ops::extract_cols(dev, &qkv_f16, t, n_rows_qkv, 0, qkvm.n_q_rows)?;
        let k_f16 = ops::extract_cols(dev, &qkv_f16, t, n_rows_qkv, qkvm.n_q_rows, qkvm.n_kv_rows)?;
        let v_f16 = ops::extract_cols(dev, &qkv_f16, t, n_rows_qkv, qkvm.n_q_rows + qkvm.n_kv_rows, qkvm.n_kv_rows)?;
        let q = ops::cast_f16_f32(dev, &q_f16)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} marlin q cast_f32: {e:?}")))?;
        let k = ops::cast_f16_f32(dev, &k_f16)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} marlin k cast_f32: {e:?}")))?;
        let v = ops::cast_f16_f32(dev, &v_f16)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} marlin v cast_f32: {e:?}")))?;
        let g_proj = w.attn.g_proj.as_ref().expect(
            "AttnWeights::qkv_marlin and ::g_proj are both built together, gated on the same \
             FFAI_LAGUNA_MARLIN load-time read, see load_gguf",
        );
        let gate_raw = ops::gemm_q4_mpp(dev, &h, &g_proj.qs, &g_proj.scales, t, g_proj.m, g_proj.k)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} g_proj gemm: {e:?}")))?;
        (q, k, v, gate_raw)
    } else {
        match &w.attn.proj {
            AttnProj::Fused { wqkvg, n_q_rows, n_kv_rows, n_g_rows } => {
                let n_rows_total = n_q_rows + 2 * n_kv_rows + n_g_rows;
                let qkvg = ops::gemm_q4_mpp(dev, &h, &wqkvg.qs, &wqkvg.scales, t, n_rows_total, hidden)?; // [T, n_rows_total]
                let q = ops::extract_cols(dev, &qkvg, t, n_rows_total, 0, *n_q_rows)?;
                let k = ops::extract_cols(dev, &qkvg, t, n_rows_total, *n_q_rows, *n_kv_rows)?;
                let v = ops::extract_cols(dev, &qkvg, t, n_rows_total, n_q_rows + n_kv_rows, *n_kv_rows)?;
                let gate_raw = ops::extract_cols(dev, &qkvg, t, n_rows_total, n_q_rows + 2 * n_kv_rows, *n_g_rows)?;
                (q, k, v, gate_raw)
            }
            AttnProj::Split { wq, wk, wv, wgate } => {
                let q = ops::gemm_q4_mpp(dev, &h, &wq.qs, &wq.scales, t, wq.m, wq.k)
                    .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} q_proj gemm: {e:?}")))?;
                psync("q_proj")?;
                let k = ops::gemm_q4_mpp(dev, &h, &wk.qs, &wk.scales, t, wk.m, wk.k)
                    .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} k_proj gemm: {e:?}")))?;
                let v = ops::gemm_q4_mpp(dev, &h, &wv.qs, &wv.scales, t, wv.m, wv.k)
                    .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} v_proj gemm: {e:?}")))?;
                let gate_raw = ops::gemm_q4_mpp(dev, &h, &wgate.qs, &wgate.scales, t, wgate.m, wgate.k)
                    .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} g_proj gemm: {e:?}")))?; // [T, n_head]
                (q, k, v, gate_raw)
            }
        }
    };

    // Per-head QK RMSNorm (rms_norm normalizes the last dim, so a
    // [T*heads, head_dim] view is correct per (token,head)), then batched
    // RoPE, YaRN on full layers, plain on sliding (`cfg.rope_for` already
    // picked the right params for each, same as decode_layer).
    psync("qkvg_proj")?;
    let q2 = ops::rms_norm(dev, &q.reshaped(vec![t * n_head, hd]), &w.attn.q_norm, cfg.eps)
        .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} q_norm: {e:?}")))?;
    let k2 = ops::rms_norm(dev, &k.reshaped(vec![t * n_kv_heads, hd]), &w.attn.k_norm, cfg.eps)
        .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} k_norm: {e:?}")))?;
    psync("qk_norm")?;
    let q3 = ops::rope_yarn_partial_many(
        dev, &q2.reshaped(vec![t, n_head, hd]), positions, n_head,
        rope.n_rot, rope.theta, rope.freq_scale, rope.ext_factor, rope.attn_factor,
        rope.beta_fast, rope.beta_slow, rope.n_orig_ctx,
    ).map_err(|e| Error::Msg(format!("prefill layer {layer_idx} q_rope: {e:?}")))?;
    psync("q_rope")?;
    let k3 = ops::rope_yarn_partial_many(
        dev, &k2.reshaped(vec![t, n_kv_heads, hd]), positions, n_kv_heads,
        rope.n_rot, rope.theta, rope.freq_scale, rope.ext_factor, rope.attn_factor,
        rope.beta_fast, rope.beta_slow, rope.n_orig_ctx,
    ).map_err(|e| Error::Msg(format!("prefill layer {layer_idx} k_rope: {e:?}")))?;
    psync("k_rope")?;
    let v3 = v.reshaped(vec![t, n_kv_heads, hd]);

    // KV cache write + attention. Full layers grow the persistent cache
    // directly (safe: nothing is ever evicted). Sliding layers write the
    // LINEAR prefill scratch instead, see `LagunaPrefillScratch`'s doc for
    // why the decode ring cannot be batch-written mid-prompt.
    let attn = if is_full {
        let slot = &kv.slots[layer_idx];
        ops::kv_append_many(dev, &k3, positions, &slot.k, n_kv_heads, hd, slot.cap)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} kv_append k: {e:?}")))?;
        ops::kv_append_many(dev, &v3, positions, &slot.v, n_kv_heads, hd, slot.cap)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} kv_append v: {e:?}")))?;
        psync("kv_write")?;
        // Every query attends the full causal prefix: seg_lo = 0 for all rows.
        let seg_lo = Tensor::new(dev.alloc_zeroed(t * 4)?, vec![t], DType::U32);
        // Full-attention layer: no sliding window, so the block skip stays
        // off (win=0), every KV block is reachable from every query row.
        ops::sdpa_multi_tc_varlen(
            dev, &q3, &slot.k, &slot.v, &seg_lo, 0, hd, n_head as u32,
            start as u32, t as u32, slot.cap as u32, heads_per_group, true, cfg.attn_scale(), 0,
        ).map_err(|e| Error::Msg(format!("prefill layer {layer_idx} sdpa_varlen_1: {e:?}")))?
    } else {
        let slot = scratch.slots[layer_idx].as_ref().ok_or_else(|| {
            Error::Msg(format!("prefill_layer: layer {layer_idx} is sliding-window but scratch has no slot for it"))
        })?;
        ops::kv_append_many(dev, &k3, positions, &slot.k, n_kv_heads, hd, slot.cap)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} kv_append k: {e:?}")))?;
        ops::kv_append_many(dev, &v3, positions, &slot.v, n_kv_heads, hd, slot.cap)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} kv_append v: {e:?}")))?;
        psync("kv_write")?;
        // Per-row sliding window: query at absolute position p attends
        // [max(0, p - (window-1)), p]. All values are >= 0, so a u32 tensor
        // (the convention every other seg_lo caller in this codebase uses)
        // carries them losslessly even though the raw-CUDA kernel reads them
        // as `int`.
        let window = cfg.sliding_window as i64;
        let seg_lo_h: Vec<u32> = (0..t)
            .map(|i| ((start + i) as i64 - (window - 1)).max(0) as u32)
            .collect();
        let seg_lo = Tensor::new(dev.upload(&u32s_to_bytes(&seg_lo_h))?, vec![t], DType::U32);
        // Sliding-window layer: `seg_lo` above was built with exactly the
        // `seg_lo[i] = max(0, (start+i) - (window-1))` convention the
        // window-mode block skip expects, so pass the same window width as
        // `win`, it prunes the KV blocks each chunk of query rows can't
        // reach, on top of the (already-applied) per-element mask.
        ops::sdpa_multi_tc_varlen(
            dev, &q3, &slot.k, &slot.v, &seg_lo, 0, hd, n_head as u32,
            start as u32, t as u32, slot.cap as u32, heads_per_group, true, cfg.attn_scale(),
            cfg.sliding_window as u32,
        ).map_err(|e| Error::Msg(format!("prefill layer {layer_idx} sdpa_varlen_0: {e:?}")))?
    }; // [T, n_head, hd]

    // Per-head softplus gate, o_proj, residual.
    psync("sdpa")?;
    let gated = ops::gate_softplus_mul_perhead_many(dev, &attn, &gate_raw, n_head)
        .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} gate_many: {e:?}")))?; // [T, n_head, hd]
    psync("gate")?;
    let gated_flat = gated.reshaped(vec![t, n_head * hd]);
    let o = if let (Some(zero_idx), Some(wo_m)) = (marlin_zero_idx, w.attn.wo_marlin.as_ref()) {
        let gated_f16 = ops::cast_f32_f16(dev, &gated_flat)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} marlin o_proj cast_f16: {e:?}")))?;
        let o_f16 = wo_m.call(dev, &gated_f16, zero_idx, t)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} marlin o_proj gemm: {e:?}")))?;
        ops::cast_f16_f32(dev, &o_f16)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} marlin o_proj cast_f32: {e:?}")))?
    } else {
        ops::gemm_q4_mpp(dev, &gated_flat, &w.attn.wo.qs, &w.attn.wo.scales, t, w.attn.wo.m, w.attn.wo.k)
            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} o_gemm: {e:?}")))? // [T, hidden]
    };
    let x1 = ops::add(dev, &x.reshaped(vec![t * hidden]), &o.reshaped(vec![t * hidden]))?.reshaped(vec![t, hidden]);
    psync("o_proj_res")?;

    // ── FFN ──────────────────────────────────────────────────────────
    let x2 = match &w.ffn {
        LayerFfn::Dense(f) => {
            let h2 = ops::rms_norm(dev, &x1, &f.ffn_norm, cfg.eps)?; // [T, hidden]
            let down = if let (Some(zero_idx), Some(gum), Some(down_m)) =
                (marlin_zero_idx, f.gateup_marlin.as_ref(), f.down_marlin.as_ref())
            {
                // FFAI_LAGUNA_MARLIN=1: [gate;up] concat as one Marlin GEMM,
                // split + SwiGLU straight in f16 (extract_cols is f16-native
                // and mt_swiglu is registered for f16 in metaltile-std), then
                // a second Marlin GEMM for down, `down_m.call` wants f16
                // input directly. Unlike the attention QKV split above, this
                // chain has no f32-only consumer in the middle (no RMSNorm
                // weight, no rope, no f32 KV cache write), so the old
                // cast_f16_f32(gateup)/cast_f32_f16(act) pair is pure
                // overhead and is removed entirely; only `h2`'s initial
                // f32->f16 cast (Marlin's input requirement) and the final
                // f16->f32 cast (the f32 residual stream) remain.
                let h2_f16 = ops::cast_f32_f16(dev, &h2)?;
                let gateup_f16 = gum.mat.call(dev, &h2_f16, zero_idx, t)?; // [T, 2*n_ff] f16
                let gate = ops::extract_cols(dev, &gateup_f16, t, 2 * gum.n_ff, 0, gum.n_ff)?;
                let up = ops::extract_cols(dev, &gateup_f16, t, 2 * gum.n_ff, gum.n_ff, gum.n_ff)?;
                let act_f16 = ops::swiglu(dev, &gate, &up)?; // [T, n_ff] f16
                let down_f16 = down_m.call(dev, &act_f16, zero_idx, t)?; // [T, hidden] f16
                ops::cast_f16_f32(dev, &down_f16)?
            } else {
                let gate = ops::gemm_q4_mpp(dev, &h2, &f.gate.qs, &f.gate.scales, t, f.gate.m, f.gate.k)?;
                let up = ops::gemm_q4_mpp(dev, &h2, &f.up.qs, &f.up.scales, t, f.up.m, f.up.k)?;
                let act = ops::swiglu(dev, &gate, &up)?;
                ops::gemm_q4_mpp(dev, &act, &f.down.qs, &f.down.scales, t, f.down.m, f.down.k)?
            };
            ops::add(dev, &x1.reshaped(vec![t * hidden]), &down.reshaped(vec![t * hidden]))?.reshaped(vec![t, hidden])
        }
        LayerFfn::Moe(f) => {
            let h2 = ops::rms_norm(dev, &x1, &f.ffn_norm, cfg.eps)?; // [T, hidden]

            // Batched sigmoid router + on-device counting-sort: the SAME
            // sigmoid+bias+top_k math as decode's `moe_router_device`, over
            // all T rows in one pass. `logits` is the small dense f32 router
            // GEMM (router stays f32, precision-sensitive); route+sort is
            // fully on device (no host sync, mirrors the NemotronH
            // NEMOTRON_DEVSORT/devdesc prefill pattern).
            let logits = ops::matmul(dev, &f.router, &h2)
                .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} router_matmul: {e:?}")))?; // [T, n_expert]
            psync("moe_router_logits")?;
            let (sorted_tok, sorted_wt, offsets) = ops::moe_route_sort_device(
                dev, &logits, &f.bias, cfg.n_expert, cfg.n_expert_used, cfg.routed_scaling,
            )?;
            let mt = t * cfg.n_expert_used;
            if std::env::var("FFAI_LAGUNA_PREFILL_SYNC").is_ok() {
                let _ = dev.synchronize();
                let mut stb = vec![0u8; mt * 4];
                let _ = dev.download(sorted_tok.buffer.as_ref(), &mut stb);
                let st: Vec<u32> = stb.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                let mut ofb = vec![0u8; (cfg.n_expert + 1) * 4];
                let _ = dev.download(offsets.buffer.as_ref(), &mut ofb);
                let of: Vec<u32> = ofb.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                eprintln!("    prefill moe layer {layer_idx}: sorted_tok={st:?} offsets_last={} (expect {mt})", of[cfg.n_expert]);
            }

            // Gather the sorted-order activation (f16, the grouped-MMA
            // input dtype) once, then TWO grouped Q4 GEMMs (gate stack, up
            // stack) over the sorted rows, elementwise SwiGLU, one grouped
            // Q4 GEMM down. `moe_q4_grouped_mma_dev` builds its tile
            // descriptors ON DEVICE from `offsets`, no host g_starts/
            // expert_ids vectors, no host sync.
            //
            // NOTE ON DTYPE (RoutedGateUp::Fused branch below): the down
            // grouped-GEMM's raw CUDA kernel reads its activation input as
            // `const __half*` unconditionally (see `MOE_Q4_GROUPED_MMA_SRC`
            // in ffai-ops), f16 in, no f32 path exists. Combined with
            // `moe_q4_grouped_mma_dev` always emitting f16 and `extract_cols`
            // now being f16-native, the gate/up split + SwiGLU chain has NO
            // f32-only consumer anywhere in it, so it runs start to finish in
            // f16 with zero casts (the old cast_f16_f32/cast_f32_f16 pair
            // existed only because `extract_cols`/`ops::swiglu` used to
            // require f32 input; both now take f16 directly).
            psync("moe_route_sort")?;
            let h2_f16 = ops::cast_f32_f16(dev, &h2)
                .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} cast_f16: {e:?}")))?; // [T, hidden]
            let a = ops::gather(dev, &h2_f16, &sorted_tok)
                .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} moe_gather_rows: {e:?}")))?; // [mt, hidden] f16, sorted-token order

            // FFAI_LAGUNA_W4A8=1: route the routed gate/up/down grouped
            // GEMMs through the W4A8 grouped-MoE GEMM (fp8 e4m3 activations
            // x mxfp4 weights, cutlass mxf8f6f4) instead of
            // `moe_q4_grouped_mma_dev`'s W4A16 path, see `MoeFfnW4a8`'s doc
            // for the weight side. `f.w4a8` was decided once at load time
            // (`laguna_w4a8_enabled`'s doc), so this just matches the
            // `Option` rather than re-reading the env.
            //
            // Unlike `moe_q4_grouped_mma_dev` (device-built tile
            // descriptors, zero host sync: see the NOTE ON DTYPE below),
            // `ops::w4a8_sfa_offsets`/`w4a8_actquant_grouped`/
            // `moe_grouped_gemm_w4a8` all take HOST `g_starts`/`expert_ids`
            // (mirroring their ffai-modeltests NEMOTRON_W4A8_MOE call site
            // exactly, no device-side descriptor-building variant of this
            // GEMM exists in ffai-ops today). So this branch pays ONE small
            // device->host download per layer (`offsets` is `n_expert+1`
            // u32s, 1028 bytes for Laguna-S-2.1's 256 experts) to turn the
            // on-device routing offsets into host group-starts, the one
            // host sync this batched-prefill path otherwise avoids
            // entirely (decode-era work eliminated exactly this kind of
            // per-layer host round-trip for the W4A16 grouped path; W4A8
            // reintroduces a tiny one because `ops::w4a8_*` has no
            // on-device-descriptor sibling yet). `Device::download` blocks
            // until prior queued work completes, so no separate
            // `dev.synchronize()` call is needed first (same convention as
            // the NEMOTRON_W4A8_MOE call site).
            let down_out = if let Some(w4a8) = f.w4a8.as_ref() {
                let mut ofb = vec![0u8; (cfg.n_expert + 1) * 4];
                dev.download(offsets.buffer.as_ref(), &mut ofb)
                    .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 offsets download: {e:?}")))?;
                let offs: Vec<usize> = ofb.chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
                let mut g_starts: Vec<usize> = vec![0];
                let mut expert_ids: Vec<usize> = Vec::new();
                for e in 0..cfg.n_expert {
                    if offs[e + 1] > offs[e] {
                        expert_ids.push(e);
                        g_starts.push(offs[e + 1]);
                    }
                }
                let group_rows: Vec<i32> = g_starts.windows(2).map(|w| (w[1] - w[0]) as i32).collect();
                psync("w4a8_offsets_dl")?;

                // gate/up both read the same pre-FFN-norm activation, so
                // they share `k = hidden` and therefore the same `kpad`,
                // grab it off whichever stack this layer built
                // (`RoutedGateUpW4a8::Fused`'s single stack, or `::Split`'s
                // `gate` stack; both give the identical padded width).
                let hid_pad = match &w4a8.gateup {
                    RoutedGateUpW4a8::Fused(s) => s.kpad,
                    RoutedGateUpW4a8::Split { gate, .. } => gate.kpad,
                };
                let a_pad = ops::pad_rows_f16(dev, &a, mt, cfg.hidden, hid_pad)
                    .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 gateup pad: {e:?}")))?;
                let (uoff, ubytes) = ops::w4a8_sfa_offsets(&g_starts, hid_pad);
                let (a_q, a_sf) = ops::w4a8_actquant_grouped(dev, &a_pad, &group_rows, &uoff, ubytes, mt, hid_pad)
                    .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 gateup actquant: {e:?}")))?;
                psync("w4a8_gateup_actquant")?;

                // Same fused-vs-split shape as the W4A16 branch below: ONE
                // grouped GEMM to `2*n_ff_exp` for `Fused`, TWO separate
                // grouped GEMMs (sharing the same quantized activation) for
                // `Split`, then the identical SwiGLU either way.
                let act = match &w4a8.gateup {
                    RoutedGateUpW4a8::Fused(gu) => {
                        let n_ff = cfg.n_ff_exp;
                        let gateup_out = ops::moe_grouped_gemm_w4a8(
                            dev, &a_q, &a_sf, &uoff, &gu.packed, &gu.sf, gu.sfb_exp_bytes,
                            &g_starts, &expert_ids, 2 * n_ff, hid_pad,
                        ).map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 gateup gemm: {e:?}")))?; // [mt, 2*n_ff_exp] f16
                        psync("w4a8_gateup_gemm")?;
                        let gate_out = ops::extract_cols(dev, &gateup_out, mt, 2 * n_ff, 0, n_ff)?;
                        let up_out = ops::extract_cols(dev, &gateup_out, mt, 2 * n_ff, n_ff, n_ff)?;
                        ops::swiglu(dev, &gate_out, &up_out)
                            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 gateup swiglu: {e:?}")))? // [mt, n_ff_exp] f16
                    }
                    RoutedGateUpW4a8::Split { gate, up } => {
                        let gate_out = ops::moe_grouped_gemm_w4a8(
                            dev, &a_q, &a_sf, &uoff, &gate.packed, &gate.sf, gate.sfb_exp_bytes,
                            &g_starts, &expert_ids, cfg.n_ff_exp, hid_pad,
                        ).map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 gate gemm: {e:?}")))?; // [mt, n_ff_exp] f16
                        let up_out = ops::moe_grouped_gemm_w4a8(
                            dev, &a_q, &a_sf, &uoff, &up.packed, &up.sf, up.sfb_exp_bytes,
                            &g_starts, &expert_ids, cfg.n_ff_exp, hid_pad,
                        ).map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 up gemm: {e:?}")))?; // [mt, n_ff_exp] f16
                        psync("w4a8_gateup_gemm")?;
                        ops::swiglu(dev, &gate_out, &up_out)
                            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 gateup swiglu: {e:?}")))? // [mt, n_ff_exp] f16
                    }
                };

                let inter_pad = w4a8.down.kpad;
                let act_pad = ops::pad_rows_f16(dev, &act, mt, cfg.n_ff_exp, inter_pad)
                    .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 down pad: {e:?}")))?;
                let (doff, dbytes) = ops::w4a8_sfa_offsets(&g_starts, inter_pad);
                let (d_q, d_sf) = ops::w4a8_actquant_grouped(dev, &act_pad, &group_rows, &doff, dbytes, mt, inter_pad)
                    .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 down actquant: {e:?}")))?;
                psync("w4a8_down_actquant")?;
                let down_out = ops::moe_grouped_gemm_w4a8(
                    dev, &d_q, &d_sf, &doff, &w4a8.down.packed, &w4a8.down.sf, w4a8.down.sfb_exp_bytes,
                    &g_starts, &expert_ids, cfg.hidden, inter_pad,
                ).map_err(|e| Error::Msg(format!("prefill layer {layer_idx} w4a8 down gemm: {e:?}")))?; // [mt, hidden] f16
                down_out
            } else {
                // FFAI_LAGUNA_MOEFUSE=1 (RoutedGateUp::Fused): ONE grouped Q4 GEMM
                // with n_out = 2*n_ff_exp over the fused [gate;up] stack instead
                // of two separate grouped GEMMs, halving the grouped-GEMM call
                // count (each call is itself two kernel launches, an on-device
                // tile-descriptor build plus the GEMM, so this also halves the
                // launch count). See the NOTE ON DTYPE below: the whole
                // split+SwiGLU chain runs in f16 with no cast, mirroring the
                // shared-expert Marlin [gate;up] split above.
                let act = match &f.gateup {
                    RoutedGateUp::Fused(gu) => {
                        let n_ff = cfg.n_ff_exp;
                        let gateup_out = ops::moe_q4_grouped_mma_dev(
                            dev, &a, &gu.qs, &gu.scales, &offsets, cfg.n_expert, mt, 2 * n_ff, cfg.hidden,
                        )
                        .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} moe_gateup_gemm_fused: {e:?}")))?; // [mt, 2*n_ff_exp] f16
                        psync("moe_gateup_gemm")?;
                        // f16 straight through, see the NOTE ON DTYPE below.
                        let gate_out = ops::extract_cols(dev, &gateup_out, mt, 2 * n_ff, 0, n_ff)?;
                        let up_out = ops::extract_cols(dev, &gateup_out, mt, 2 * n_ff, n_ff, n_ff)?;
                        ops::swiglu(dev, &gate_out, &up_out)
                            .map_err(|e| Error::Msg(format!("prefill layer {layer_idx} moe_act_swiglu: {e:?}")))? // [mt, n_ff_exp] f16
                    }
                    RoutedGateUp::Split { gate, up } => {
                        let gate_out = ops::moe_q4_grouped_mma_dev(
                            dev, &a, &gate.qs, &gate.scales, &offsets, cfg.n_expert, mt, cfg.n_ff_exp, cfg.hidden,
                        )?; // [mt, n_ff_exp] f16
                        let up_out = ops::moe_q4_grouped_mma_dev(
                            dev, &a, &up.qs, &up.scales, &offsets, cfg.n_expert, mt, cfg.n_ff_exp, cfg.hidden,
                        )?; // [mt, n_ff_exp] f16
                        psync("moe_gateup_gemm")?;
                        ops::swiglu(dev, &gate_out, &up_out)? // [mt, n_ff_exp] f16
                    }
                };
                let down_out = ops::moe_q4_grouped_mma_dev(
                    dev, &act, &f.down.qs, &f.down.scales, &offsets, cfg.n_expert, mt, cfg.hidden, cfg.n_ff_exp,
                )?; // [mt, hidden] f16
                down_out
            };

            // Weighted scatter-add back to token order: acc[t,h] = sum over
            // this token's top_k sorted rows of down_out[r,h] * sorted_wt[r]
            // (router weight already includes routed_scaling, no extra
            // `unscale` factor needed here, unlike Nemotron's ReLU²/256 trick).
            let acc = Tensor::new(dev.alloc_zeroed(t * cfg.hidden * 4)?, vec![t, cfg.hidden], DType::F32);
            psync("moe_down_gemm")?;
            ops::moe_scatter_add_det_dev(
                dev, &down_out, &sorted_tok, &sorted_wt, &acc, t, mt, cfg.hidden, 1.0, true,
            )?;

            // Always-on shared expert: same batched gate→up→SwiGLU→down as a
            // routed expert, just over ALL T rows unweighted (decode's
            // unweighted `gemv_accum(scale=1)` equivalent). Handles both
            // weight layouts (FFAI_LAGUNA_MICRO=1's single-expert
            // Q4ExpertStack fusion, or the plain Q4Dense split) uniformly by
            // pulling a (qs, scales, m, k) view for gate/up either way,
            // `expert_view(0)`'s byte offset is always zero for a 1-expert
            // stack, so it's safe to feed into `gemm_q4_mpp` even though that
            // op's `Binding::Buffer` ignores a non-zero `Tensor.offset`.
            psync("moe_scatter")?;
            let sdown_out = if let (Some(zero_idx), Some(sm)) = (marlin_zero_idx, f.shared_marlin.as_ref()) {
                // FFAI_LAGUNA_MARLIN=1: same [gate;up]-concat-then-split
                // Marlin pattern as the dense-layer FFN above. `h2_f16` was
                // already computed for the routed-expert gather just above,
                // reuse it instead of re-casting. Split + SwiGLU stay in f16
                // end to end (extract_cols f16-native, mt_swiglu f16-native,
                // `sm.down.call` wants f16 in), the old cast_f16_f32(gateup)
                // / cast_f32_f16(sact) pair is removed; only the final
                // f16->f32 cast remains, needed because `sdown_out` is added
                // into the f32 MoE accumulator (`acc2`) below.
                let gateup_f16 = sm.gateup.mat.call(dev, &h2_f16, zero_idx, t)?; // [T, 2*n_ff_shexp] f16
                let sgate = ops::extract_cols(dev, &gateup_f16, t, 2 * sm.gateup.n_ff, 0, sm.gateup.n_ff)?;
                let sup = ops::extract_cols(dev, &gateup_f16, t, 2 * sm.gateup.n_ff, sm.gateup.n_ff, sm.gateup.n_ff)?;
                let sact_f16 = ops::swiglu(dev, &sgate, &sup)?; // [T, n_ff_shexp] f16
                let sdown_f16 = sm.down.call(dev, &sact_f16, zero_idx, t)?; // [T, hidden] f16
                ops::cast_f16_f32(dev, &sdown_f16)? // [T, hidden]
            } else {
                let (sg_qs, sg_sc, sg_m, sg_k, su_qs, su_sc, su_m, su_k, sdown) = match &f.shared_expert {
                    SharedExpert::Split { gate, up, down } => (
                        gate.qs.clone(), gate.scales.clone(), gate.m, gate.k,
                        up.qs.clone(), up.scales.clone(), up.m, up.k,
                        down,
                    ),
                    SharedExpert::Fused(se) => {
                        let (gqs, gsc) = se.gate.expert_view(0);
                        let (uqs, usc) = se.up.expert_view(0);
                        (gqs, gsc, se.gate.m, se.gate.k, uqs, usc, se.up.m, se.up.k, &se.down)
                    }
                };
                let sgate = ops::gemm_q4_mpp(dev, &h2, &sg_qs, &sg_sc, t, sg_m, sg_k)?;
                let sup = ops::gemm_q4_mpp(dev, &h2, &su_qs, &su_sc, t, su_m, su_k)?;
                let sact = ops::swiglu(dev, &sgate, &sup)?;
                ops::gemm_q4_mpp(dev, &sact, &sdown.qs, &sdown.scales, t, sdown.m, sdown.k)? // [T, hidden]
            };
            let acc2 = ops::add(dev, &acc.reshaped(vec![t * cfg.hidden]), &sdown_out.reshaped(vec![t * cfg.hidden]))?;

            ops::add(dev, &x1.reshaped(vec![t * hidden]), &acc2)?.reshaped(vec![t, hidden])
        }
    };

    Ok(x2)
}

/// Batched (multi-token) **prefill**: process the whole `tokens` prompt
/// `chunk` tokens at a time (default recommendation: 1024), writing every
/// token's K/V into `kv`/`scratch`, and return the LAST token's next-token
/// logits, the prefill counterpart of [`decode_step`]'s per-token forward.
/// Each chunk's `T` rows run through every layer as batched GEMMs
/// ([`ops::gemm_q4_mpp`], the grouped-MMA MoE pipeline, `sdpa_multi_tc_varlen`)
/// instead of `T` sequential [`decode_step`] calls.
///
/// `scratch` must have been sized (via [`LagunaPrefillScratch::new`]) for AT
/// LEAST `tokens.len()`. `kv` must have been sized (via
/// [`LagunaKvCache::new`]) for at least `tokens.len()` plus however many
/// decode tokens the caller intends to generate afterward, same contract as
/// feeding the same prompt through `decode_step` one token at a time.
///
/// After the LAST chunk, this compacts each sliding-window layer's linear
/// prefill scratch (the last `min(sliding_window, tokens.len())` positions)
/// into `kv`'s decode ring, so `decode_step` / `LagunaDecodeCtx` can continue
/// from `pos = tokens.len()` and produce the IDENTICAL greedy continuation a
/// pure per-token `decode_step` loop over the same prompt would.
///
/// `chunk` only bounds how many prompt tokens are embedded/projected/attended
/// in one forward pass (memory for the batched intermediates scales with
/// it); correctness is independent of the choice.
pub fn prefill(
    dev: &dyn Device,
    model: &LagunaModel,
    kv: &LagunaKvCache,
    scratch: &LagunaPrefillScratch,
    tokens: &[u32],
    chunk: usize,
) -> Result<Tensor> {
    let cfg = &model.cfg;
    let total = tokens.len();
    if total == 0 {
        return Err(Error::Msg("prefill: empty prompt".into()));
    }
    if total > scratch.prompt_cap {
        return Err(Error::Msg(format!(
            "prefill: prompt len {total} exceeds scratch prompt_cap {}", scratch.prompt_cap
        )));
    }
    let chunk = chunk.max(1);

    // FFAI_LAGUNA_MARLIN=1: ONE persistent all-zero `u32` indices buffer,
    // sized to `chunk` (every chunk's row count `t = chunk.min(total-start)`
    // is `<= chunk`), reused by every `MarlinDense::call` in every layer of
    // every chunk this `prefill()` invocation processes. The Marlin GEMM's
    // grouped-MoE ABI wants a per-row expert id; a dense `[n_out, k_in]`
    // matrix is fed as a single "expert 0", so every row's id is 0: see
    // `MarlinDense`'s doc. Built even if a given layer's per-weight
    // `Option<MarlinDense>` ends up `None` (dimension not 64-aligned); the
    // cost is one small zeroed alloc, and `prefill_layer` only reads through
    // this when a `Some(MarlinDense)` is also present.
    let marlin_zero_idx = if laguna_marlin_enabled() {
        Some(Tensor::new(dev.alloc_zeroed(chunk * 4)?, vec![chunk], DType::U32))
    } else {
        None
    };

    let mut x_last: Option<Tensor> = None;
    let mut t_last = 0usize;
    let mut start = 0usize;
    while start < total {
        let t = chunk.min(total - start);
        let chunk_tokens = &tokens[start..start + t];

        let ids = Tensor::new(dev.upload(&u32s_to_bytes(chunk_tokens))?, vec![t], DType::U32);
        let mut x = ops::gather(dev, &model.tok_embd, &ids)
            .map_err(|e| Error::Msg(format!("prefill embed gather: {e:?}")))?
            .reshaped(vec![t, cfg.hidden]);

        let pos_vec: Vec<u32> = (0..t).map(|i| (start + i) as u32).collect();
        let positions = Tensor::new(dev.upload(&u32s_to_bytes(&pos_vec))?, vec![t], DType::U32);

        if std::env::var("FFAI_LAGUNA_PREFILL_SYNC").is_ok() {
            dev.synchronize().map_err(|e| Error::Msg(format!("prefill fault pre-layers (embed/positions): {e:?}")))?;
        }
        for (i, layer) in model.layers.iter().enumerate() {
            x = prefill_layer(dev, cfg, i, layer, kv, scratch, &x, t, start, &positions, marlin_zero_idx.as_ref())?
                .into();
            if std::env::var("FFAI_LAGUNA_PREFILL_SYNC").is_ok() {
                dev.synchronize().map_err(|e| Error::Msg(format!("prefill fault after layer {i}: {e:?}")))?;
            }
        }

        x_last = Some(x);
        t_last = t;
        start += t;
    }
    let x = x_last.expect("prefill: total > 0 was checked above, so at least one chunk always runs");

    // Final norm + q8 lm_head on the LAST token of the LAST chunk only (the
    // next-token logits), cheaper than decode's reference-engine-style "norm all rows
    // then slice" since RMSNorm is per-row-independent anyway.
    let last_row = ops::slice(dev, &x.reshaped(vec![t_last * cfg.hidden]), (t_last - 1) * cfg.hidden, cfg.hidden)?;
    let xn = ops::rms_norm(dev, &last_row, &model.out_norm, cfg.eps)?;
    let logits = ops::gemv_q8(
        dev, &model.lm_head.qs, &model.lm_head.scales, &xn, model.lm_head.m, model.lm_head.k, model.lm_head.m,
    )?;

    // Compact each sliding-window layer's linear prefill scratch into the
    // persistent decode ring, one-time cost, so decode can continue from
    // `pos = total` exactly as it does today.
    let copy_len = total.min(cfg.sliding_window);
    let copy_start = total - copy_len;
    for i in 0..cfg.n_layers {
        if cfg.is_full_layer(i) {
            continue;
        }
        let sslot = scratch.slots[i].as_ref().expect("sliding layer always has a scratch slot");
        let dslot = &kv.slots[i];
        ops::kv_compact_ring(dev, &sslot.k, &dslot.k, cfg.n_kv_heads, sslot.cap, cfg.head_dim, dslot.cap, copy_start, copy_len)?;
        ops::kv_compact_ring(dev, &sslot.v, &dslot.v, cfg.n_kv_heads, sslot.cap, cfg.head_dim, dslot.cap, copy_start, copy_len)?;
    }

    Ok(logits)
}

/// Persistent per-step device scalars decode reads instead of uploading a
/// fresh 1-element buffer every call. Allocated ONCE (in [`DecodeWorkspace::new`])
/// and refreshed in place every step via [`DecodeWorkspace::update`], the
/// refresh always happens BEFORE the (eager or captured-graph) decode work
/// that reads them, and never inside a graph-capture region, so it's safe
/// regardless of whether the caller is warming up, capturing, or replaying.
///
/// `full_pos`/`sliding_pos` are decode_layer's existing posbuf inputs (KV
/// cache write index, full layers grow, sliding layers ring); `n_kv_full`/
/// `n_kv_sliding` are the NEW graph-mode buffers `sdpa_decode_nbuf` reads
/// (`pos+1` / `min(pos+1, sliding_window)`). Eager (non-graph) decode only
/// ever reads `token_id`/`full_pos`/`sliding_pos`; `n_kv_full`/`n_kv_sliding`
/// are only consulted by `decode_layer` when `FFAI_LAGUNA_GRAPH=1`.
pub struct DecodeWorkspace {
    token_id: Tensor,
    full_pos: Tensor,
    sliding_pos: Tensor,
    n_kv_full: Tensor,
    n_kv_sliding: Tensor,
}

impl DecodeWorkspace {
    pub fn new(dev: &dyn Device) -> Result<Self> {
        let zero = || -> Result<Tensor> {
            Ok(Tensor::new(dev.upload(&0u32.to_le_bytes())?, vec![1], DType::U32))
        };
        Ok(DecodeWorkspace {
            token_id: zero()?,
            full_pos: zero()?,
            sliding_pos: zero()?,
            n_kv_full: zero()?,
            n_kv_sliding: zero()?,
        })
    }

    /// Refresh every workspace buffer for the token at `pos`. Five tiny
    /// (single-`u32`) raw-CUDA writes, in place: no reallocation, so the
    /// buffers' device pointers (what a captured graph binds) never move.
    /// MUST be called outside any graph-capture region.
    pub fn update(&self, dev: &dyn Device, cfg: &LagunaConfig, token_id: u32, pos: u32) -> Result<()> {
        // TODO(perf): 5 tiny kernel launches per step (one per scalar). Still
        // far cheaper than 5 cuMemAlloc+H2D uploads, but a single fused
        // "write 5 u32s" raw kernel (one launch, 5 scalar args) would shave a
        // bit more host-side launch overhead off the outside-the-graph cost
        // once this becomes the next bottleneck.
        ops::write_u32(dev, &self.token_id, token_id)?;
        ops::write_u32(dev, &self.full_pos, pos)?;
        let sliding_pos = pos % cfg.sliding_window as u32;
        ops::write_u32(dev, &self.sliding_pos, sliding_pos)?;
        ops::write_u32(dev, &self.n_kv_full, pos + 1)?;
        let n_kv_sliding = (pos + 1).min(cfg.sliding_window as u32);
        ops::write_u32(dev, &self.n_kv_sliding, n_kv_sliding)?;
        Ok(())
    }
}

/// Single-token decode: embed → every layer (self-attention + cache append +
/// FFN) → final RMSNorm → lm_head → next-token logits `[vocab]`. `pos` is
/// this token's absolute sequence position (0 for the first token); the
/// caller advances it and re-calls per generated token, reusing `kv` across
/// calls so the cache persists.
///
/// Reads `token_id`/RoPE-position/KV-length from `ws` rather than uploading
/// fresh 1-element buffers every call (the former per-step host upload this
/// function used to do), the caller must have already called
/// `ws.update(dev, &model.cfg, token_id, pos)` for this step. This is both a
/// standalone eager-mode win (no per-step `cuMemAlloc`+H2D for those
/// scalars) and what makes the whole step CUDA-graph-capturable: a graph
/// replays the exact recorded kernel launches, so every per-token input has
/// to sit behind a buffer POINTER that stays fixed across replays.
///
/// This is the path used both by plain eager decode (no env set) and by
/// [`LagunaDecodeCtx`]'s warmup steps and one-time capture step; graph
/// REPLAY (after capture) does not call this function at all, it re-runs
/// the already-recorded launches via `Device::graph_launch`.
///
/// `FFAI_LAGUNA_GRAPH=1` switches `decode_layer`'s RoPE/SDPA to the
/// buffer-driven (capture-safe) kernels; without it, behavior is byte-for-byte
/// identical to before this refactor (same kernels, same scalar arguments,
/// just reading `token_id`/`pos` off a persistent buffer instead of a fresh
/// one, the values are the same either way).
pub fn decode_step(
    dev: &dyn Device,
    model: &LagunaModel,
    kv: &LagunaKvCache,
    ws: &DecodeWorkspace,
    pos: u32,
) -> Result<Tensor> {
    let cfg = &model.cfg;
    let mut x = ops::gather(dev, &model.tok_embd, &ws.token_id)?.reshaped(vec![cfg.hidden]);

    let graph_mode = std::env::var("FFAI_LAGUNA_GRAPH").is_ok();
    let graph_bufs = if graph_mode {
        Some(GraphBufs { n_kv_full: &ws.n_kv_full, n_kv_sliding: &ws.n_kv_sliding })
    } else {
        None
    };
    // Read once per decode_step, passed down as a bool: all three micro-
    // fusions (QKVG concat gemv, shared-expert fused swiglu gather, o_proj
    // residual accum) are gated on this single env var. Fixed for the
    // process lifetime (same requirement CUDA-graph capture already has,
    // see GraphBufs's doc), so this per-step read always agrees with the
    // load-time read that picked AttnProj/SharedExpert's Fused vs Split.
    let micro = laguna_micro_enabled();

    // FFAI_LAGUNA_DEBUG=1: per-layer residual stats (L2 norm, NaN count) to
    // bisect a divergence to the first bad layer. Hot path unaffected when
    // the env is unset. Downloads mid-step, so it is INCOMPATIBLE with graph
    // capture/replay, `LagunaDecodeCtx::step` refuses to run at all when
    // this is set (see its doc); a bare `decode_step` call with both env
    // vars set would otherwise try to `dev.download` from inside a capture
    // region, which is forbidden by the CUDA driver.
    let debug = std::env::var("FFAI_LAGUNA_DEBUG").map(|v| v == "1").unwrap_or(false);
    for (i, layer) in model.layers.iter().enumerate() {
        x = decode_layer(dev, cfg, i, layer, &kv.slots[i], &x, pos, &ws.full_pos, &ws.sliding_pos, &model.one, graph_bufs.as_ref(), micro)?;
        if debug {
            let mut xb = vec![0u8; cfg.hidden * 4];
            dev.synchronize()?;
            dev.download(x.buffer.as_ref(), &mut xb)?;
            let xf: Vec<f32> = xb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let n_nan = xf.iter().filter(|v| v.is_nan()).count();
            let l2 = xf.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
            let amax = xf.iter().fold(0f32, |a, &v| if v.abs() > a { v.abs() } else { a });
            eprintln!("  layer {i:2} ({}{}): |x|2={l2:.3e} amax={amax:.3e} nan={n_nan} head={:?}",
                if cfg.is_full_layer(i) { "full" } else { "swa " },
                if cfg.is_dense_layer(i) { ", dense" } else { "" },
                &xf[..4]);
        }
    }

    let xn = ops::rms_norm(dev, &x, &model.out_norm, cfg.eps)?;
    ops::gemv_q8(dev, &model.lm_head.qs, &model.lm_head.scales, &xn, model.lm_head.m, model.lm_head.k, model.lm_head.m)
}

/// CUDA-graph-capturing decode driver: collapses Laguna's ~1000+ per-token
/// kernel launches (48 layers x ~22 ops) into ONE `cuGraphLaunch` replay per
/// token, removing host-launch-gap overhead. Enabled by running with
/// `FFAI_LAGUNA_GRAPH=1` (also read by `decode_step`/`decode_layer` to pick
/// the buffer-driven RoPE/SDPA kernels, see their docs).
///
/// Protocol (mirrors the NemotronH `NEMOTRON_GRAPH` precedent in
/// `ffai-modeltests`): the first two `step()` calls run [`decode_step`]
/// eagerly ("warm up N=2 full steps normally", the pool's bucket free-list
/// reaches a deterministic steady state after the first call, since every
/// step walks the identical fixed 48-layer op sequence regardless of
/// `token_id`/`pos`, so the second call already allocates nothing new). The
/// THIRD call captures: `Device::begin_capture` + one more `decode_step` +
/// `Device::end_capture`. Capture only RECORDS the stream's kernel launches
/// (it does not execute them), so that third call's `decode_step` result is
/// not yet real data; `step()` immediately follows the capture with one
/// `Device::graph_launch` to actually compute that token's logits. Every
/// call after that just refreshes the workspace and replays the graph.
///
/// The logits `Tensor` returned from the capture step is kept alive for the
/// lifetime of this `LagunaDecodeCtx` (never dropped) so its device buffer's
/// pointer can never be recycled by the allocator's pool, the same buffer
/// is what every subsequent `graph_launch` writes into, and what `step()`
/// returns (a cheap `Arc` clone) every time thereafter.
///
/// `FFAI_LAGUNA_DEBUG` downloads mid-step and is incompatible with capture;
/// `step()` returns an error immediately if it's set, rather than silently
/// producing a broken capture.
pub struct LagunaDecodeCtx {
    ws: DecodeWorkspace,
    /// Executable-graph handle from `Device::end_capture` (an opaque `u64`;
    /// see the `Device` trait doc, CUDA's `CUgraphExec` cast through
    /// `usize`). `None` until the capture step has run.
    graph: Option<u64>,
    /// The fixed output buffer graph replays write into (see the struct doc).
    logits: Option<Tensor>,
    /// Steps run so far via this ctx (warmup + capture; stops incrementing
    /// once capture has happened, replay doesn't touch it).
    step_count: usize,
}

impl LagunaDecodeCtx {
    /// Eager warmup steps before the capture step (brief: "warm up N=2 full
    /// steps normally"). Capture happens on call number `WARMUP_STEPS + 1`.
    const WARMUP_STEPS: usize = 2;

    /// `model`/`kv`/`max_seq` are accepted (matching the `LagunaKvCache::new`
    /// shape) for parity with the decode driver's usual construction
    /// arguments and to leave room for future workspace sizing; today's
    /// workspace is five fixed 1-element buffers, independent of `max_seq`.
    pub fn new(dev: &dyn Device, model: &LagunaModel, kv: &LagunaKvCache, max_seq: usize) -> Result<Self> {
        let _ = (model, kv, max_seq);
        Ok(LagunaDecodeCtx { ws: DecodeWorkspace::new(dev)?, graph: None, logits: None, step_count: 0 })
    }

    /// Decode one token. Identical call shape to [`decode_step`] plus `self`;
    /// internally dispatches to eager warmup, one-time capture, or graph
    /// replay depending on how many steps have run. See the struct doc for
    /// the full protocol.
    pub fn step(
        &mut self,
        dev: &dyn Device,
        model: &LagunaModel,
        kv: &LagunaKvCache,
        token_id: u32,
        pos: u32,
    ) -> Result<Tensor> {
        if std::env::var("FFAI_LAGUNA_DEBUG").is_ok() {
            return Err(Error::Msg(
                "LagunaDecodeCtx::step: FFAI_LAGUNA_DEBUG is incompatible with CUDA-graph \
                 capture/replay (the debug path downloads mid-step, which the CUDA driver \
                 forbids during capture). Unset FFAI_LAGUNA_DEBUG or use plain decode_step."
                    .into(),
            ));
        }

        // Refresh the workspace BEFORE any capture/replay, always outside
        // the captured region, whether we're warming up, about to capture,
        // or replaying an already-captured graph.
        self.ws.update(dev, &model.cfg, token_id, pos)?;

        if let Some(exec) = self.graph {
            dev.graph_launch(exec)?;
            return Ok(self
                .logits
                .clone()
                .expect("LagunaDecodeCtx: graph captured but no logits tensor saved"));
        }

        self.step_count += 1;
        if self.step_count <= Self::WARMUP_STEPS {
            // Eager warmup: reaches pool steady state (deterministic alloc
            // order, every step walks the same fixed layer sequence).
            return decode_step(dev, model, kv, &self.ws, pos);
        }

        // step_count == WARMUP_STEPS + 1: capture this step's op sequence.
        // Capture only RECORDS launches; it does not execute them, so
        // `logits` is not yet valid data, one `graph_launch` right after
        // `end_capture` actually computes it (against the workspace values
        // written above, so it's the correct result for THIS token).
        dev.begin_capture()?;
        let logits = decode_step(dev, model, kv, &self.ws, pos)?;
        let exec = dev.end_capture()?;
        self.graph = Some(exec);
        self.logits = Some(logits.clone());
        dev.graph_launch(exec)?;
        Ok(logits)
    }
}
