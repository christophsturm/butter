// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-loader
//!
//! Weight loaders. Pure CPU byte-parsing + upload through the
//! [`Device`](ffai_core::Device) trait — no GPU API, fully shared across
//! backends. SafeTensors is implemented; GGUF / HF follow.

use ffai_core::{DType, Error, Result};
use std::collections::BTreeMap;

/// One tensor's location + metadata inside a SafeTensors blob.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    shard: usize,
    begin: usize,
    end: usize,
}

/// One mmap'd `.safetensors` shard: the map + where its data section starts.
struct Shard {
    map: memmap2::Mmap,
    data_start: usize,
}

/// One or more mmap'd `.safetensors` files (single or sharded). `tensor()`
/// returns a zero-copy slice routed to the right shard. mmap keeps the 14GB+
/// sharded checkpoints off the heap.
pub struct SafeTensors {
    shards: Vec<Shard>,
    index: BTreeMap<String, TensorInfo>,
}

/// IEEE half → f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if mant == 0 { sign << 31 } else {
            let mut e = -1i32; let mut m = mant;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            (sign << 31) | (((e + 127 - 15) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "I32" => DType::I32,
        "U32" => DType::U32,
        "I8" => DType::I8,
        "U8" => DType::U8,
        other => return Err(Error::Msg(format!("safetensors: unsupported dtype {other}"))),
    })
}

impl SafeTensors {
    /// mmap + parse one `.safetensors` shard, merging its tensors into `index`
    /// tagged with `shard_idx`. Returns the constructed [`Shard`].
    fn open_shard(path: &str, shard_idx: usize, index: &mut BTreeMap<String, TensorInfo>) -> Result<Shard> {
        let file = std::fs::File::open(path).map_err(|e| Error::Msg(format!("open {path}: {e}")))?;
        // SAFETY: read-only file outlives the mapping; treated as immutable bytes.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Msg(format!("mmap {path}: {e}")))?;
        if map.len() < 8 {
            return Err(Error::Msg("safetensors: file too small".into()));
        }
        let header_len = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        let header: serde_json::Value = serde_json::from_slice(&map[8..header_end])
            .map_err(|e| Error::Msg(format!("safetensors header JSON: {e}")))?;
        let obj = header
            .as_object()
            .ok_or_else(|| Error::Msg("safetensors: header not an object".into()))?;
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            // Omni/multimodal checkpoints mix in dtypes we don't decode (I64
            // position buffers, F8 vision scales, …). Skip those tensors rather
            // than failing the whole load — callers only ask for the ones they need.
            let dtype = match parse_dtype(v["dtype"].as_str().ok_or_else(|| Error::Msg("missing dtype".into()))?) {
                Ok(dt) => dt,
                Err(_) => continue,
            };
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or_else(|| Error::Msg("missing shape".into()))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v["data_offsets"]
                .as_array()
                .ok_or_else(|| Error::Msg("missing data_offsets".into()))?;
            let begin = off[0].as_u64().unwrap() as usize;
            let end = off[1].as_u64().unwrap() as usize;
            index.insert(name.clone(), TensorInfo { dtype, shape, shard: shard_idx, begin, end });
        }
        Ok(Shard { map, data_start: header_end })
    }

    /// Open + parse a single `.safetensors` file.
    pub fn open(path: &str) -> Result<Self> {
        let mut index = BTreeMap::new();
        let shard = Self::open_shard(path, 0, &mut index)?;
        Ok(SafeTensors { shards: vec![shard], index })
    }

    /// Open a model directory: sharded (`model-XXXXX-of-YYYYY.safetensors` per
    /// `model.safetensors.index.json`) or single (`model.safetensors`). All
    /// shards are mmap'd and merged into one tensor index.
    pub fn open_dir(dir: &str) -> Result<Self> {
        let idx_path = format!("{dir}/model.safetensors.index.json");
        let files: Vec<String> = if std::path::Path::new(&idx_path).exists() {
            let txt = std::fs::read_to_string(&idx_path)
                .map_err(|e| Error::Msg(format!("read {idx_path}: {e}")))?;
            let j: serde_json::Value = serde_json::from_str(&txt)
                .map_err(|e| Error::Msg(format!("index json: {e}")))?;
            let wm = j["weight_map"].as_object()
                .ok_or_else(|| Error::Msg("index: no weight_map".into()))?;
            let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for v in wm.values() {
                if let Some(f) = v.as_str() { set.insert(format!("{dir}/{f}")); }
            }
            set.into_iter().collect()
        } else {
            vec![format!("{dir}/model.safetensors")]
        };
        let mut index = BTreeMap::new();
        let mut shards = Vec::with_capacity(files.len());
        for (i, f) in files.iter().enumerate() {
            shards.push(Self::open_shard(f, i, &mut index)?);
        }
        Ok(SafeTensors { shards, index })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name)
    }

    /// Tensor decoded to `f32` (handles F32 / F16 / BF16 on disk) + its shape.
    /// The convenience every host-orchestrated model test wants — checkpoints
    /// ship in any of the three float widths.
    /// BF16 tensors are decoded in parallel across all available CPU cores to
    /// reduce the 62GB Nemotron load from ~110s to ~15-20s (stdlib threads only).
    // ── MLX 4-bit affine quant support ───────────────────────────────────────
    // MLX stores a quantized matrix as three tensors:
    //   <name>.weight U32  [rows, in/8]   — 8 nibbles packed per u32, element j at
    //                                        bits 4*j (low nibble = element 0)
    //   <name>.scales BF16 [rows, in/gs]  — per-group scale
    //   <name>.biases BF16 [rows, in/gs]  — per-group bias
    // Dequant (mode="affine"): w = q * scale + bias, group_size gs along `in`.
    // This lets the BF16-expecting model tests consume an MLX 4-bit checkpoint
    // transparently — `tensor_f32` returns the dense [rows, in] f32 weight.

    /// Decode a BF16/F16/F32 tensor's raw bytes to an f32 vec (small aux tensors).
    fn bytes_to_f32(b: &[u8], dt: DType) -> Vec<f32> {
        match dt {
            DType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            DType::F16 => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            DType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            _ => Vec::new(),
        }
    }

    /// True if `<base>.scales` exists — i.e. `<base>.weight` is MLX-quantized.
    fn is_mlx_quant(&self, base: &str) -> bool {
        self.index.contains_key(&format!("{base}.scales"))
    }

    /// MLX-affine-dequantize `<base>.weight` (+ `.scales`/`.biases`) to a dense
    /// f32 matrix `[rows, in]` where `in = packed_cols * 8`. Handles both 2-D
    /// `[rows, in/8]` and 3-D `[E, rows, in/8]` (flattened to `[E*rows, in]`).
    fn mlx_dequant_f32(&self, base: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (wb, wdt, wsh) = self.tensor(&format!("{base}.weight"))?;
        if wdt != DType::U32 {
            return Err(Error::Msg(format!("mlx_dequant '{base}': weight dtype {wdt} not U32")));
        }
        let (sb, sdt, ssh) = self.tensor(&format!("{base}.scales"))?;
        let scales = Self::bytes_to_f32(sb, sdt);
        // biases optional (affine without bias is rare here, but guard anyway).
        let biases = match self.tensor(&format!("{base}.biases")) {
            Ok((bb, bdt, _)) => Self::bytes_to_f32(bb, bdt),
            Err(_) => vec![0.0f32; scales.len()],
        };
        // Collapse leading expert dim if 3-D: [E, rows, packed] → rows' = E*rows.
        let packed_cols = *wsh.last().unwrap();
        let rows: usize = wsh[..wsh.len() - 1].iter().product();
        let in_dim = packed_cols * 8;
        let groups = *ssh.last().unwrap(); // scales last dim = in_dim / group_size
        let group_size = in_dim / groups;
        let words: Vec<u32> = wb.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        // Parallel dequant across rows (rows are independent; ~52×128 expert rows).
        let mut out = vec![0f32; rows * in_dim];
        let n_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1).min(rows.max(1));
        let chunk_rows = rows.div_ceil(n_threads);
        let out_base = out.as_mut_ptr() as usize;
        let words_ptr = words.as_ptr() as usize;
        let scales_ptr = scales.as_ptr() as usize;
        let biases_ptr = biases.as_ptr() as usize;
        let words_len = words.len();
        std::thread::scope(|scope| {
            for t in 0..n_threads {
                let r0 = t * chunk_rows;
                let r1 = (r0 + chunk_rows).min(rows);
                if r0 >= r1 { continue; }
                scope.spawn(move || {
                    // SAFETY: disjoint row ranges into `out`; read-only shared inputs.
                    let out_s = unsafe { std::slice::from_raw_parts_mut(out_base as *mut f32, rows * in_dim) };
                    let words_s = unsafe { std::slice::from_raw_parts(words_ptr as *const u32, words_len) };
                    let scales_s = unsafe { std::slice::from_raw_parts(scales_ptr as *const f32, rows * groups) };
                    let biases_s = unsafe { std::slice::from_raw_parts(biases_ptr as *const f32, rows * groups) };
                    for r in r0..r1 {
                        let wrow = r * packed_cols;
                        let grow = r * groups;
                        let orow = r * in_dim;
                        for col in 0..in_dim {
                            let word = words_s[wrow + col / 8];
                            let q = (word >> (4 * (col % 8))) & 0xf;
                            let g = col / group_size;
                            out_s[orow + col] = q as f32 * scales_s[grow + g] + biases_s[grow + g];
                        }
                    }
                });
            }
        });
        Ok((out, vec![rows, in_dim]))
    }

    /// Resolve a model-test tensor NAME against this checkpoint, transparently
    /// handling two real-world skews so the dense BF16-oriented model tests can
    /// load an MLX 4-bit Nemotron-Cascade checkpoint unchanged:
    ///   1. `language_model.` prefix present in the Omni naming but absent here.
    ///   2. routed experts requested per-expert (`...experts.{e}.up_proj.weight`)
    ///      but packed here as `...switch_mlp.fc1.weight[e]` (up) / `.fc2`(down).
    /// Returns the dense f32 weight + shape, MLX-dequantizing on the fly when the
    /// resolved tensor is MLX-quantized. Returns None if no remap applies (caller
    /// falls back to the plain path).
    fn resolve_f32(&self, name: &str) -> Option<Result<(Vec<f32>, Vec<usize>)>> {
        // Candidate names: as-is, then with the `language_model.` prefix stripped.
        let stripped = name.strip_prefix("language_model.").map(|s| s.to_string());
        for cand in [Some(name.to_string()), stripped].into_iter().flatten() {
            // (a) per-expert → packed switch_mlp slice.
            if let Some(rest) = cand.strip_suffix(".weight") {
                // ...mixer.experts.{e}.up_proj  /  .down_proj
                if let Some(idx) = rest.find(".experts.") {
                    let prefix = &rest[..idx]; // ...mixer
                    let tail = &rest[idx + ".experts.".len()..]; // {e}.up_proj or {e}.down_proj
                    if let Some((e_str, proj)) = tail.split_once('.') {
                        if let Ok(e) = e_str.parse::<usize>() {
                            let fc = if proj == "up_proj" { "fc1" } else { "fc2" };
                            let packed = format!("{prefix}.switch_mlp.{fc}");
                            if self.is_mlx_quant(&packed) {
                                return Some(self.mlx_expert_slice_f32(&packed, e));
                            }
                        }
                    }
                }
                // (b) plain MLX-quantized weight.
                if self.is_mlx_quant(rest) {
                    return Some(self.mlx_dequant_f32(rest));
                }
            }
            // (c) plain present tensor under the candidate name (e.g. prefix strip
            // for non-quantized tensors: norms, conv1d, A_log, D, gate bias…).
            if cand != name && self.index.contains_key(&cand) {
                return Some(self.tensor_f32_raw(&cand));
            }
        }
        None
    }

    /// Dequant ONLY expert `e`'s slab from a packed MLX `switch_mlp.fcN` tensor
    /// (3-D `[E, rows, packed]`) → dense `[rows, in]` f32. Touches only expert e's
    /// bytes (not all E experts) so the per-expert MoE setup loop stays O(E), not
    /// O(E²).
    fn mlx_expert_slice_f32(&self, packed: &str, e: usize) -> Result<(Vec<f32>, Vec<usize>)> {
        let (wb, wdt, wsh) = self.tensor(&format!("{packed}.weight"))?; // [E, rows, packed]
        if wdt != DType::U32 {
            return Err(Error::Msg(format!("mlx_expert_slice '{packed}': weight dtype {wdt} not U32")));
        }
        if wsh.len() != 3 {
            return Err(Error::Msg(format!("mlx_expert_slice '{packed}': expected 3-D weight, got {wsh:?}")));
        }
        let (n_exp, rows, packed_cols) = (wsh[0], wsh[1], wsh[2]);
        let in_dim = packed_cols * 8;
        let (sb, sdt, ssh) = self.tensor(&format!("{packed}.scales"))?; // [E, rows, groups]
        let groups = *ssh.last().unwrap();
        let group_size = in_dim / groups;
        let scales_all = Self::bytes_to_f32(sb, sdt);
        let biases_all = match self.tensor(&format!("{packed}.biases")) {
            Ok((bb, bdt, _)) => Self::bytes_to_f32(bb, bdt),
            Err(_) => vec![0.0f32; scales_all.len()],
        };
        if e >= n_exp {
            return Err(Error::Msg(format!("mlx_expert_slice '{packed}': expert {e} >= {n_exp}")));
        }
        // Byte/element offsets for expert e.
        let words_per_exp = rows * packed_cols;
        let w_off = e * words_per_exp; // in u32 words
        let g_off = e * rows * groups; // in scale/bias elements
        let w_bytes = &wb[w_off * 4..(w_off + words_per_exp) * 4];
        let words: Vec<u32> = w_bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let scales = &scales_all[g_off..g_off + rows * groups];
        let biases = &biases_all[g_off..g_off + rows * groups];
        let mut out = vec![0f32; rows * in_dim];
        for r in 0..rows {
            let wrow = r * packed_cols;
            let grow = r * groups;
            let orow = r * in_dim;
            for col in 0..in_dim {
                let word = words[wrow + col / 8];
                let q = (word >> (4 * (col % 8))) & 0xf;
                let g = col / group_size;
                out[orow + col] = q as f32 * scales[grow + g] + biases[grow + g];
            }
        }
        Ok((out, vec![rows, in_dim]))
    }

    /// The original plain decode path (F32/F16/BF16 only), no MLX/remap.
    fn tensor_f32_raw(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (b, dt, shape) = self.tensor(name)?;
        Self::decode_f32(b, dt, shape)
    }

    fn decode_f32(b: &[u8], dt: DType, shape: &[usize]) -> Result<(Vec<f32>, Vec<usize>)> {
        let v = match dt {
            DType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            DType::F16 => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            DType::BF16 => {
                let n_elems = b.len() / 2;
                let n_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1).min(n_elems.max(1));
                if n_threads <= 1 || n_elems < 4096 {
                    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
                } else {
                    let chunk_elems = (n_elems + n_threads - 1) / n_threads;
                    let mut out = vec![0f32; n_elems];
                    let out_base = out.as_mut_ptr();
                    let b_base = b.as_ptr();
                    let jobs: Vec<(usize, usize, usize)> = (0..n_threads).filter_map(|t| {
                        let start = t * chunk_elems;
                        let len = chunk_elems.min(n_elems.saturating_sub(start));
                        if len == 0 { return None; }
                        Some((out_base as usize + start * 4, b_base as usize + start * 2, len))
                    }).collect();
                    std::thread::scope(|scope| {
                        for (dst_addr, src_addr, len) in &jobs {
                            let (dst_addr, src_addr, len) = (*dst_addr, *src_addr, *len);
                            scope.spawn(move || {
                                let d: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(dst_addr as *mut f32, len) };
                                let s: &[u8] = unsafe { std::slice::from_raw_parts(src_addr as *const u8, len * 2) };
                                for (di, c) in d.iter_mut().zip(s.chunks_exact(2)) {
                                    *di = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                                }
                            });
                        }
                    });
                    out
                }
            }
            other => return Err(Error::Msg(format!("decode_f32: dtype {other} unsupported"))),
        };
        Ok((v, shape.to_vec()))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        // Fast path: tensor present verbatim AND not MLX-quantized.
        if let Some(info) = self.index.get(name) {
            if info.dtype != DType::U32 || !name.ends_with(".weight")
                || !self.is_mlx_quant(name.strip_suffix(".weight").unwrap_or(name))
            {
                let (b, dt, shape) = self.tensor(name)?;
                return Self::decode_f32(b, dt, shape);
            }
        }
        // Remap / MLX-dequant path (name skew or 4-bit checkpoint).
        if let Some(r) = self.resolve_f32(name) {
            return r;
        }
        let (b, dt, shape) = self.tensor(name)?;
        Self::decode_f32(b, dt, shape)
    }

    /// Raw bytes + dtype + shape of a tensor, or an error if absent.
    pub fn tensor(&self, name: &str) -> Result<(&[u8], DType, &[usize])> {
        let info = self
            .index
            .get(name)
            .ok_or_else(|| Error::Msg(format!("safetensors: tensor '{name}' not found")))?;
        let sh = &self.shards[info.shard];
        let s = sh.data_start + info.begin;
        let e = sh.data_start + info.end;
        Ok((&sh.map[s..e], info.dtype, &info.shape))
    }
}
mod iq2xxs_tables;
pub mod gguf;

// ── Capability probe ────────────────────────────────────────────────────
//
// Derive what a checkpoint can ACTUALLY do from the tensors present, not from
// the model card / config (which lie — quants silently strip vision/audio
// towers, repos reuse llama/qwen backbones + bolt on towers, names are a mess).
// Cross-checks the config's *claims* against the tensors and flags the gap.

/// What a checkpoint can do (each flag = the relevant tensors are present).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Caps {
    pub text: bool,
    pub vision: bool,
    pub audio: bool,
}

/// Grounded inspection of a model directory.
#[derive(Debug, Clone)]
pub struct Probe {
    /// `model_type` from config.json (the card's claim).
    pub config_model_type: Option<String>,
    pub config_architectures: Vec<String>,
    /// Capabilities **detected from the tensors** — the truth.
    pub detected: Caps,
    /// Capabilities the config **claims** (vision_config / audio_config keys).
    pub declared: Caps,
    pub n_tensors: usize,
    pub n_params: u64,
    /// Mismatches between claim and reality — e.g. a quant that stripped vision.
    pub warnings: Vec<String>,
}

fn any_name(names: &[String], pats: &[&str]) -> bool {
    names.iter().any(|n| pats.iter().any(|p| n.contains(p)))
}

/// Probe a model dir: read config.json (if any) + scan the SafeTensors tensor
/// names, and report what it can really do — independent of the card.
pub fn probe(dir: &str) -> Result<Probe> {
    let st = SafeTensors::open_dir(dir)?;
    let names: Vec<String> = st.names().cloned().collect();
    let n_params: u64 = names.iter().filter_map(|n| st.info(n)).map(|i| i.shape.iter().product::<usize>() as u64).sum();

    // detected from tensors (the truth)
    let detected = Caps {
        text: any_name(&names, &["embed_tokens", "wte", "embed_in", "token_embedding"]),
        vision: any_name(&names, &["vision_model", "vision_tower", "visual.", "patch_embedding", "patch_embed", "vision_encoder"]),
        audio: any_name(&names, &["audio_tower", "audio_encoder", "encoder.conv1", "feature_extractor", "audio_model"]),
    };

    // config.json claims (the card)
    let mut config_model_type = None;
    let mut config_architectures = Vec::new();
    let mut declared = Caps::default();
    if let Ok(txt) = std::fs::read_to_string(format!("{dir}/config.json")) {
        if let Ok(j) = serde_json::from_str::<serde_json::Value>(&txt) {
            config_model_type = j["model_type"].as_str().map(String::from);
            if let Some(a) = j["architectures"].as_array() {
                config_architectures = a.iter().filter_map(|x| x.as_str().map(String::from)).collect();
            }
            declared.vision = !j["vision_config"].is_null();
            declared.audio = !j["audio_config"].is_null() || !j["audio_encoder_config"].is_null();
            declared.text = !j["text_config"].is_null() || config_model_type.is_some();
        }
    }

    // claim-vs-reality gaps (Eric's "the quant stripped vision but the card didn't say so")
    let mut warnings = Vec::new();
    if declared.vision && !detected.vision {
        warnings.push("config declares a vision tower but NO vision tensors are present — this checkpoint/quant stripped it (text-only despite the card)".into());
    }
    if declared.audio && !detected.audio {
        warnings.push("config declares audio but NO audio tensors are present — stripped from this checkpoint".into());
    }
    if detected.vision && !declared.vision {
        warnings.push("vision tensors present but config has no vision_config — a tower was bolted on / the card under-reports".into());
    }

    Ok(Probe { config_model_type, config_architectures, detected, declared, n_tensors: names.len(), n_params, warnings })
}
