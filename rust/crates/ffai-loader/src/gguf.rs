// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! GGUF v3 reader + dequant. Parses the header (metadata + tensor infos) and
//! the tightly-packed data section, and dequantizes tensors to f32. Supported
//! formats: F32, F16, Q8_0, the k-quant super-blocks (Q2_K, Q4_K, Q5_K, Q6_K),
//! and IQ2_XXS. Q4_K_M GGUFs (the common k-quant most real models ship) load
//! through the Q4_K/Q6_K arms. Split/multi-part GGUFs (`*-NNNNN-of-MMMMM.gguf`)
//! are opened transparently: pass any one part and all parts are mmapped and
//! their tensor tables merged.

use ffai_core::{Error, Result};
use std::collections::BTreeMap;

use crate::iq2xxs_tables::{IQ2XXS_GRID, KSIGNS};

/// GGML tensor type (subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    Q2K,
    Q4K,
    Q5K,
    Q6K,
    Iq2Xxs,
    Other(u32),
}
impl GgmlType {
    fn from_u32(t: u32) -> Self {
        match t {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            8 => GgmlType::Q8_0,
            10 => GgmlType::Q2K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            16 => GgmlType::Iq2Xxs,
            o => GgmlType::Other(o),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64, // within the data section
    /// Which file part this tensor's data lives in (0 for single-file GGUF;
    /// the split index for llama.cpp multi-part `*-NNNNN-of-MMMMM.gguf`).
    pub part: usize,
}

/// A memory-mapped GGUF file (handles the 81GB DSv4 checkpoint without
/// reading it into RAM). For split GGUFs (`*-00001-of-00002.gguf`), `parts`
/// holds one mmap per file and each tensor records its part index; metadata
/// is taken from part 0 (the split header carries the full KV set).
pub struct Gguf {
    parts: Vec<memmap2::Mmap>,
    /// Per-part data-section start offset (after that part's header).
    data_starts: Vec<usize>,
    tensors: BTreeMap<String, GgufTensor>,
    pub metadata_u32: BTreeMap<String, u32>,
    pub metadata_str: BTreeMap<String, String>,
    /// Scalar f32 metadata (e.g. `qwen2.rope.freq_base`, `*.rms_norm_eps`).
    pub metadata_f32: BTreeMap<String, f32>,
    /// String-array metadata (e.g. `tokenizer.ggml.tokens`, `.merges`).
    pub metadata_arr_str: BTreeMap<String, Vec<String>>,
    /// Int-array metadata (e.g. `tokenizer.ggml.token_type`).
    pub metadata_arr_i32: BTreeMap<String, Vec<i32>>,
    /// f32-array metadata (e.g. `tokenizer.ggml.scores` for SentencePiece).
    pub metadata_arr_f32: BTreeMap<String, Vec<f32>>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        v
    }
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn gstr(&mut self) -> String {
        let n = self.u64() as usize;
        let s = String::from_utf8_lossy(&self.b[self.p..self.p + n]).into_owned();
        self.p += n;
        s
    }
    /// Read a scalar metadata value of `vtype`, returning whichever of
    /// {u32 (any scalar int/bool), f32, String} it decodes to. (Arrays are
    /// handled separately by the caller so the elements can be collected.)
    fn read_scalar(&mut self, vtype: u32) -> (Option<u32>, Option<f32>, Option<String>) {
        match vtype {
            0 | 1 => { self.p += 1; (Some(self.b[self.p - 1] as u32), None, None) }
            2 | 3 => { let v = u16::from_le_bytes(self.b[self.p..self.p + 2].try_into().unwrap()); self.p += 2; (Some(v as u32), None, None) }
            4 | 5 => (Some(self.u32()), None, None),
            6 => { let v = self.f32(); (None, Some(v), None) } // f32
            7 => { self.p += 1; (Some(self.b[self.p - 1] as u32), None, None) } // bool
            8 => (None, None, Some(self.gstr())),
            10 | 11 => { let v = self.u64(); (Some(v as u32), None, None) }
            12 => { self.p += 8; (None, None, None) } // f64
            _ => (None, None, None),
        }
    }
    /// Consume one metadata value of `vtype`, skipping its bytes (used to walk
    /// past array elements we don't keep).
    fn skip_one(&mut self, vtype: u32) {
        match vtype {
            0 | 1 | 7 => self.p += 1,
            2 | 3 => self.p += 2,
            4 | 5 | 6 => self.p += 4,
            10 | 11 | 12 => self.p += 8,
            8 => { let _ = self.gstr(); }
            9 => { let et = self.u32(); let len = self.u64(); for _ in 0..len { self.skip_one(et); } }
            _ => {}
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal half: value = mant * 2^-24. Normalize the mantissa so the
            // implicit leading 1 lands in bit 10, tracking the resulting f32
            // exponent. A subnormal half has unbiased exponent -14; each left
            // shift to reach the leading 1 lowers it by one more.
            let mut m = mant;
            let mut e: i32 = -14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff; // drop the now-implicit leading 1
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

/// Unpack the j-th 6-bit (scale, min) pair from a Q4_K/Q5_K block's packed
/// 12-byte `scales` array. This is the canonical GGML `get_scale_min_k4`:
/// the 8 sub-block scales/mins are stored 6 bits each, split across the first
/// 8 bytes (low 6 bits) and the last 4 bytes (high 2 bits + extra 4-bit field).
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let sc = q[j] & 63;
        let m = q[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (sc, m)
    }
}

/// One parsed GGUF header (metadata + tensor infos + aligned data start).
struct ParsedHeader {
    metadata_u32: BTreeMap<String, u32>,
    metadata_str: BTreeMap<String, String>,
    metadata_f32: BTreeMap<String, f32>,
    metadata_arr_str: BTreeMap<String, Vec<String>>,
    metadata_arr_i32: BTreeMap<String, Vec<i32>>,
    metadata_arr_f32: BTreeMap<String, Vec<f32>>,
    tensors: Vec<GgufTensor>,
    data_start: usize,
}

/// Parse a single GGUF v3 file's header (KV metadata + tensor infos) from its
/// mmapped bytes. Shared by the single-file and split-file open paths.
fn parse_header(bytes: &[u8], path: &str) -> Result<ParsedHeader> {
    let mut c = Cursor { b: bytes, p: 0 };
    let magic = c.u32();
    if magic != 0x4655_4747 {
        return Err(Error::Msg(format!("{path}: not a GGUF file (magic {magic:#x})")));
    }
    let version = c.u32();
    if version != 3 {
        return Err(Error::Msg(format!("{path}: GGUF version {version} unsupported (need 3)")));
    }
    let n_tensors = c.u64();
    let n_kv = c.u64();

    let mut metadata_u32 = BTreeMap::new();
    let mut metadata_str = BTreeMap::new();
    let mut metadata_f32 = BTreeMap::new();
    let mut metadata_arr_str: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut metadata_arr_i32: BTreeMap<String, Vec<i32>> = BTreeMap::new();
    let mut metadata_arr_f32: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut alignment: u64 = 32;
    for _ in 0..n_kv {
        let key = c.gstr();
        let vtype = c.u32();
        if vtype == 9 {
            // array: elem_type u32, len u64, then len elems. We collect
            // string arrays (tokenizer vocab/merges) and int arrays
            // (token_type/scores indices); everything else is skipped.
            let et = c.u32();
            let len = c.u64() as usize;
            match et {
                8 => {
                    let mut v = Vec::with_capacity(len);
                    for _ in 0..len { v.push(c.gstr()); }
                    metadata_arr_str.insert(key.clone(), v);
                }
                0 | 1 | 2 | 3 | 4 | 5 | 7 | 10 | 11 => {
                    let mut v = Vec::with_capacity(len);
                    for _ in 0..len {
                        let (u, _, _) = c.read_scalar(et);
                        v.push(u.unwrap_or(0) as i32);
                    }
                    metadata_arr_i32.insert(key.clone(), v);
                }
                6 => {
                    // f32 array (e.g. SentencePiece `tokenizer.ggml.scores`).
                    let mut v = Vec::with_capacity(len);
                    for _ in 0..len {
                        let (_, fl, _) = c.read_scalar(et);
                        v.push(fl.unwrap_or(0.0));
                    }
                    metadata_arr_f32.insert(key.clone(), v);
                }
                _ => { for _ in 0..len { c.skip_one(et); } }
            }
            continue;
        }
        let (u, fl, s) = c.read_scalar(vtype);
        if let Some(u) = u {
            if key == "general.alignment" {
                alignment = u as u64;
            }
            metadata_u32.insert(key.clone(), u);
        }
        if let Some(fl) = fl {
            metadata_f32.insert(key.clone(), fl);
        }
        if let Some(s) = s {
            metadata_str.insert(key.clone(), s);
        }
    }

    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.gstr();
        let n_dims = c.u32() as usize;
        let dims: Vec<u64> = (0..n_dims).map(|_| c.u64()).collect();
        let ggml_type = GgmlType::from_u32(c.u32());
        let offset = c.u64();
        tensors.push(GgufTensor { name, dims, ggml_type, offset, part: 0 });
    }

    // Align the data section start.
    let data_start = c.p.next_multiple_of(alignment as usize);
    Ok(ParsedHeader {
        metadata_u32,
        metadata_str,
        metadata_f32,
        metadata_arr_str,
        metadata_arr_i32,
        metadata_arr_f32,
        tensors,
        data_start,
    })
}

/// If `path` matches the llama.cpp split pattern `*-NNNNN-of-MMMMM.gguf`,
/// return the list of all part paths in order (1..=MMMMM). Otherwise return
/// `None` (single-file GGUF). The given `path` may be any one of the parts.
fn split_parts(path: &str) -> Option<Vec<String>> {
    // Match the trailing "-NNNNN-of-MMMMM.gguf".
    let stripped = path.strip_suffix(".gguf")?;
    let dash = stripped.rfind("-of-")?;
    let (left, mmmmm) = stripped.split_at(dash);
    let mmmmm = &mmmmm["-of-".len()..];
    let nnnnn_dash = left.rfind('-')?;
    let (prefix, nnnnn) = left.split_at(nnnnn_dash);
    let nnnnn = &nnnnn[1..];
    // Both index fields must be all-digits and the same width.
    if nnnnn.is_empty() || mmmmm.is_empty()
        || !nnnnn.bytes().all(|b| b.is_ascii_digit())
        || !mmmmm.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let total: usize = mmmmm.parse().ok()?;
    let width = mmmmm.len();
    if total <= 1 {
        return None;
    }
    Some(
        (1..=total)
            .map(|i| format!("{prefix}-{i:0width$}-of-{mmmmm}.gguf"))
            .collect(),
    )
}

impl Gguf {
    pub fn open(path: &str) -> Result<Self> {
        // Detect a split GGUF (`*-00001-of-00002.gguf`) and gather all parts;
        // a single-file GGUF is just a one-element list.
        let part_paths = split_parts(path).unwrap_or_else(|| vec![path.to_string()]);

        let mut parts: Vec<memmap2::Mmap> = Vec::with_capacity(part_paths.len());
        let mut data_starts: Vec<usize> = Vec::with_capacity(part_paths.len());
        let mut tensors: BTreeMap<String, GgufTensor> = BTreeMap::new();
        let mut meta: Option<ParsedHeader> = None;

        for (pi, pp) in part_paths.iter().enumerate() {
            let file = std::fs::File::open(pp)
                .map_err(|e| Error::Msg(format!("open {pp}: {e}")))?;
            // SAFETY: the file is read-only and outlives the mapping; we treat
            // it as an immutable byte slice.
            let bytes = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| Error::Msg(format!("mmap {pp}: {e}")))?;
            let mut hdr = parse_header(&bytes, pp)?;
            data_starts.push(hdr.data_start);
            let part_tensors = std::mem::take(&mut hdr.tensors);
            for mut t in part_tensors {
                t.part = pi;
                tensors.insert(t.name.clone(), t);
            }
            // Keep the full metadata from part 0 (the split header carries it).
            if pi == 0 {
                meta = Some(hdr);
            }
            parts.push(bytes);
        }

        let m = meta.expect("at least one part");
        Ok(Gguf {
            parts,
            data_starts,
            tensors,
            metadata_u32: m.metadata_u32,
            metadata_str: m.metadata_str,
            metadata_f32: m.metadata_f32,
            metadata_arr_str: m.metadata_arr_str,
            metadata_arr_i32: m.metadata_arr_i32,
            metadata_arr_f32: m.metadata_arr_f32,
        })
    }

    /// Read a `u32`/int scalar from metadata under `key`.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata_u32.get(key).copied()
    }
    /// Read an `f32` scalar from metadata under `key` (e.g. rope freq base, eps).
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata_f32.get(key).copied()
    }
    /// Read a string scalar from metadata under `key`.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata_str.get(key).map(|s| s.as_str())
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.get(name)
    }

    fn raw(&self, t: &GgufTensor, n_bytes: usize) -> &[u8] {
        let s = self.data_starts[t.part] + t.offset as usize;
        &self.parts[t.part][s..s + n_bytes]
    }

    /// Dequantize a tensor to f32. Supports F32, F16, Q8_0, the k-quants
    /// (Q2_K, Q4_K, Q5_K, Q6_K) and IQ2_XXS.
    pub fn dequant_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name).ok_or_else(|| Error::Msg(format!("tensor '{name}' not found")))?;
        let n: usize = t.dims.iter().product::<u64>() as usize;
        match t.ggml_type {
            GgmlType::F32 => {
                let b = self.raw(t, n * 4);
                Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
            }
            GgmlType::F16 => {
                let b = self.raw(t, n * 2);
                Ok(b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
            }
            GgmlType::Q8_0 => {
                // block of 32: f16 scale (2 bytes) + 32 int8.
                let nblocks = n / 32;
                let b = self.raw(t, nblocks * 34);
                let mut out = Vec::with_capacity(n);
                for blk in 0..nblocks {
                    let base = blk * 34;
                    let scale = f16_to_f32(u16::from_le_bytes([b[base], b[base + 1]]));
                    for i in 0..32 {
                        let q = b[base + 2 + i] as i8;
                        out.push(scale * q as f32);
                    }
                }
                Ok(out)
            }
            GgmlType::Q2K => {
                // block_q2_K (QK_K=256, 84 bytes): scales[16] + qs[64] + d(f16) + dmin(f16).
                const QK: usize = 256;
                let nblocks = n / QK;
                let b = self.raw(t, nblocks * 84);
                let mut out = Vec::with_capacity(n);
                for blk in 0..nblocks {
                    let base = blk * 84;
                    let scales = &b[base..base + 16];
                    let qs = &b[base + 16..base + 80];
                    let d = f16_to_f32(u16::from_le_bytes([b[base + 80], b[base + 81]]));
                    let dmin = f16_to_f32(u16::from_le_bytes([b[base + 82], b[base + 83]]));
                    let mut is = 0usize;
                    let mut q_off = 0usize;
                    for _n in (0..QK).step_by(128) {
                        let mut shift = 0u8;
                        for _j in 0..4 {
                            for half in 0..2 {
                                let sc = scales[is];
                                is += 1;
                                let dl = d * (sc & 0xF) as f32;
                                let ml = dmin * (sc >> 4) as f32;
                                for l in 0..16 {
                                    let q = (qs[q_off + half * 16 + l] >> shift) & 3;
                                    out.push(dl * q as f32 - ml);
                                }
                            }
                            shift += 2;
                        }
                        q_off += 32;
                    }
                }
                Ok(out)
            }
            GgmlType::Q4K => {
                // block_q4_K (QK_K=256, 144 bytes):
                //   d(f16) + dmin(f16) + scales[12] (6-bit packed, 8 sub-blocks)
                //   + qs[128] (4-bit codes; low nibbles then high nibbles).
                // 8 sub-blocks of 32; sub-blocks 2j/2j+1 share the 64-byte qs
                // chunk q (low nibble = even, high nibble = odd sub-block).
                const QK: usize = 256;
                let nblocks = n / QK;
                let b = self.raw(t, nblocks * 144);
                let mut out = Vec::with_capacity(n);
                for blk in 0..nblocks {
                    let base = blk * 144;
                    let d = f16_to_f32(u16::from_le_bytes([b[base], b[base + 1]]));
                    let dmin = f16_to_f32(u16::from_le_bytes([b[base + 2], b[base + 3]]));
                    let scales = &b[base + 4..base + 16]; // 12 bytes
                    let qs = &b[base + 16..base + 144]; // 128 bytes
                    for j in 0..(QK / 64) {
                        // pair of sub-blocks (low/high nibble) over a 32-byte qs chunk
                        let q = &qs[j * 32..j * 32 + 32];
                        let (sc1, m1) = get_scale_min_k4(2 * j, scales);
                        let d1 = d * sc1 as f32;
                        let min1 = dmin * m1 as f32;
                        let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
                        let d2 = d * sc2 as f32;
                        let min2 = dmin * m2 as f32;
                        for l in 0..32 {
                            out.push(d1 * (q[l] & 0xF) as f32 - min1);
                        }
                        for l in 0..32 {
                            out.push(d2 * (q[l] >> 4) as f32 - min2);
                        }
                    }
                }
                Ok(out)
            }
            GgmlType::Q5K => {
                // block_q5_K (QK_K=256, 176 bytes):
                //   d(f16) + dmin(f16) + scales[12] + qh[32] (1 high bit/elem)
                //   + qs[128] (4-bit low codes). Value = 4-bit low | (high bit << 4).
                const QK: usize = 256;
                let nblocks = n / QK;
                let b = self.raw(t, nblocks * 176);
                let mut out = Vec::with_capacity(n);
                for blk in 0..nblocks {
                    let base = blk * 176;
                    let d = f16_to_f32(u16::from_le_bytes([b[base], b[base + 1]]));
                    let dmin = f16_to_f32(u16::from_le_bytes([b[base + 2], b[base + 3]]));
                    let scales = &b[base + 4..base + 16]; // 12 bytes
                    let qh = &b[base + 16..base + 48]; // 32 bytes
                    let qs = &b[base + 48..base + 176]; // 128 bytes
                    for j in 0..(QK / 64) {
                        let q = &qs[j * 32..j * 32 + 32];
                        let (sc1, m1) = get_scale_min_k4(2 * j, scales);
                        let d1 = d * sc1 as f32;
                        let min1 = dmin * m1 as f32;
                        let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
                        let d2 = d * sc2 as f32;
                        let min2 = dmin * m2 as f32;
                        // The two high-bit planes for this 64-elem chunk select
                        // bit 2j (low sub-block) and 2j+1 (high sub-block).
                        let bit_lo = 1u8 << (2 * j);
                        let bit_hi = 1u8 << (2 * j + 1);
                        for l in 0..32 {
                            let hi = if qh[l] & bit_lo != 0 { 16 } else { 0 };
                            out.push(d1 * ((q[l] & 0xF) as i32 + hi) as f32 - min1);
                        }
                        for l in 0..32 {
                            let hi = if qh[l] & bit_hi != 0 { 16 } else { 0 };
                            out.push(d2 * ((q[l] >> 4) as i32 + hi) as f32 - min2);
                        }
                    }
                }
                Ok(out)
            }
            GgmlType::Q6K => {
                // block_q6_K (QK_K=256, 210 bytes):
                //   ql[128] (low 4 bits) + qh[64] (high 2 bits) + scales[16]
                //   (int8) + d(f16). 6-bit signed codes (q - 32) × per-16 scale.
                const QK: usize = 256;
                let nblocks = n / QK;
                let b = self.raw(t, nblocks * 210);
                let mut out = vec![0.0f32; n];
                for blk in 0..nblocks {
                    let base = blk * 210;
                    let ql = &b[base..base + 128];
                    let qh = &b[base + 128..base + 192];
                    let scales = &b[base + 192..base + 208]; // 16 int8
                    let d = f16_to_f32(u16::from_le_bytes([b[base + 208], b[base + 209]]));
                    let out_blk = &mut out[blk * QK..blk * QK + QK];
                    // Two 128-elem halves; each processes 32 lanes over 4 groups.
                    for n2 in 0..(QK / 128) {
                        let ql_h = &ql[n2 * 64..n2 * 64 + 64];
                        let qh_h = &qh[n2 * 32..n2 * 32 + 32];
                        let sc = &scales[n2 * 8..n2 * 8 + 8];
                        let dst = &mut out_blk[n2 * 128..n2 * 128 + 128];
                        for l in 0..32 {
                            let is = l / 16;
                            let q1 = ((ql_h[l] & 0xF) as i32 | (((qh_h[l] >> 0) & 3) as i32) << 4) - 32;
                            let q2 = ((ql_h[l + 32] & 0xF) as i32 | (((qh_h[l] >> 2) & 3) as i32) << 4) - 32;
                            let q3 = ((ql_h[l] >> 4) as i32 | (((qh_h[l] >> 4) & 3) as i32) << 4) - 32;
                            let q4 = ((ql_h[l + 32] >> 4) as i32 | (((qh_h[l] >> 6) & 3) as i32) << 4) - 32;
                            dst[l] = d * sc[is] as i8 as f32 * q1 as f32;
                            dst[l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                            dst[l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                            dst[l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                        }
                    }
                }
                Ok(out)
            }
            GgmlType::Iq2Xxs => {
                // block_iq2_xxs (256-elem, 66 bytes): d(f16) + qs[64]. qs is
                // 8 groups of 8 bytes: 4 grid indices + a u32 of scale(>>28)
                // and 4×7-bit sign indices into KSIGNS.
                const QK: usize = 256;
                let nblocks = n / QK;
                let b = self.raw(t, nblocks * 66);
                let mut out = vec![0.0f32; n];
                for blk in 0..nblocks {
                    let base = blk * 66;
                    let d = f16_to_f32(u16::from_le_bytes([b[base], b[base + 1]]));
                    let qs = &b[base + 2..base + 66]; // 64 bytes
                    for ib32 in 0..8 {
                        let g = ib32 * 8;
                        let aux8 = [qs[g], qs[g + 1], qs[g + 2], qs[g + 3]];
                        let aux32 = u32::from_le_bytes([qs[g + 4], qs[g + 5], qs[g + 6], qs[g + 7]]);
                        let db = d * 0.25 * (0.5 + (aux32 >> 28) as f32);
                        for l in 0..4 {
                            let grid = &IQ2XXS_GRID[aux8[l] as usize];
                            let signs = KSIGNS[((aux32 >> (7 * l)) & 127) as usize];
                            for j in 0..8 {
                                let s = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                                out[blk * QK + ib32 * 32 + l * 8 + j] = db * grid[j] as f32 * s;
                            }
                        }
                    }
                }
                Ok(out)
            }
            other => Err(Error::Msg(format!("dequant '{name}': {other:?} not supported"))),
        }
    }

    /// Repack a 2-D weight tensor into the resident-Q8 layout `ffai_ops::gemv_q8`
    /// consumes: `(qs, scales, m, k)` where `m`/`k` are the [out, in] matrix dims,
    /// `scales` is `[m * k/32]` f32 (one per 32-block), and `qs` is the int8
    /// codes packed 4-per-u32 (8 u32/block) at `qs[r*(k/32)*8 + b*8 + i/4]`.
    ///
    /// For a Q8_0 tensor this is a *lossless* repack of the on-disk blocks (the
    /// f16 block scale is widened to f32, the 32 int8 are re-packed) — no second
    /// quantization. F16/F32 tensors are quantized per-32-block via amax/127, the
    /// identical scheme to `ffai_ops::quantize_q8`. `k` (the in-dim, fastest GGUF
    /// dim) must be a multiple of 32.
    pub fn q8_repack(&self, name: &str) -> Result<(Vec<u32>, Vec<f32>, usize, usize)> {
        let t = self.tensor(name).ok_or_else(|| Error::Msg(format!("tensor '{name}' not found")))?;
        if t.dims.len() != 2 {
            return Err(Error::Msg(format!("q8_repack '{name}': expected 2-D, got {:?}", t.dims)));
        }
        // GGUF dims are fastest-first: a [out, in] matrix is listed as [in, out].
        let k = t.dims[0] as usize; // in  (fastest, row stride)
        let m = t.dims[1] as usize; // out (rows)
        if k % 32 != 0 {
            return Err(Error::Msg(format!("q8_repack '{name}': in-dim {k} not a multiple of 32")));
        }
        let bpr = k / 32;
        let mut qs = vec![0u32; m * bpr * 8];
        let mut scales = vec![0f32; m * bpr];
        match t.ggml_type {
            GgmlType::Q8_0 => {
                // On-disk block of 32: f16 scale (2 bytes) + 32 int8. The block
                // order is row-major over [out, in], so block (r,b) is at linear
                // block index r*bpr + b — exactly the gemv_q8 ordering. Lossless.
                let nblocks = m * bpr;
                let b = self.raw(t, nblocks * 34);
                for blk in 0..nblocks {
                    let base = blk * 34;
                    scales[blk] = f16_to_f32(u16::from_le_bytes([b[base], b[base + 1]]));
                    for w_i in 0..8 {
                        let mut packed = 0u32;
                        for i in 0..4 {
                            let byte = b[base + 2 + w_i * 4 + i];
                            packed |= (byte as u32) << (i * 8);
                        }
                        qs[blk * 8 + w_i] = packed;
                    }
                }
            }
            // F16/F32 and the k-quants (Q2_K/Q4_K/Q5_K/Q6_K/IQ2_XXS) are first
            // dequantized to f32, then re-quantized per-32-block to Q8 (amax/127).
            GgmlType::F32
            | GgmlType::F16
            | GgmlType::Q2K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Iq2Xxs => {
                let w = self.dequant_f32(name)?; // row-major [out, in]
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
            }
            other => return Err(Error::Msg(format!("q8_repack '{name}': {other:?} not supported"))),
        }
        Ok((qs, scales, m, k))
    }
}
