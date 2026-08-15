// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! NVFP4 (modelopt convention) checkpoint reader.
//!
//! Reads `<base>.weight` / `<base>.weight_scale` / `<base>.weight_scale_2`
//! straight out of a mmap'd [`SafeTensors`] into the flat layout `iron`'s
//! block-scaled kernel family (`iron_nvfp4_qgemv`, `iron_nvfp4_qmm`,
//! `iron_nvfp4_qmm_mma`, `iron_nvfp4_batched_4_qgemv`,
//! `iron_nvfp4_rms_norm_qgemv`, `iron_nvfp4_gated_rms_norm_qgemv`) expects —
//! see `NVFP4_LAYOUT.md` at the repo root for the full derivation. No
//! re-swizzle, no copy for `packed`/`scales`: both are borrowed straight from
//! the mmap.
//!
//! Layout (per tensor, `rows` output features × `cols` input features,
//! `block_size` = modelopt `group_size`, 16 in the Qwen3.6-27B checkpoint):
//!
//! ```text
//! <base>.weight        U8   [rows, cols/2]          2× E2M1 nibbles/byte
//! <base>.weight_scale  U8   [rows, cols/block_size]  raw E4M3 byte code
//! <base>.weight_scale_2 F32 scalar                    one global per tensor
//! ```
//!
//! `packed`/`scales` are exposed as raw bytes rather than pre-cast to `u32`:
//! the byte buffer *is* a valid `[rows, cols/8]` u32 buffer already (4
//! packed bytes = 1 u32, little-endian, matching iron's own
//! `packed >> (i*4) & 0xF` nibble-order — see `NVFP4_LAYOUT.md` §3) so
//! callers can reinterpret in place when uploading to a device buffer,
//! with zero host-side transform.

use crate::{DType, Error, Result, SafeTensors};

/// One NVFP4-quantized linear weight, borrowed zero-copy from the checkpoint
/// mmap. Kernel-ready per `iron`'s block-scaled `nvfp4` format variant
/// (`WT=u32` reinterpret of `packed`, `ST=u8` for `scales`, `SKIND=1`).
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4Tensor<'a> {
    /// `[rows, cols/2]` raw bytes — 2× E2M1 nibbles/byte (low nibble = even
    /// column). Reinterpret as `[rows, cols/8]` u32 for kernel dispatch.
    pub packed: &'a [u8],
    /// `[rows, cols/block_size]` raw E4M3 byte codes (one per block).
    pub scales: &'a [u8],
    /// Per-tensor global scale (modelopt `weight_scale_2`); the kernel's
    /// `global: f32` constexpr dispatch parameter. `w = e2m1(nibble) *
    /// e4m3(scale_byte) * scale2`.
    pub scale2: f32,
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
}

/// Decode a 4-bit E2M1 code to f32. Bit 3 = sign; bits [0,3) = magnitude over
/// the OCP MX/NVFP4 codebook `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. Mirrors iron's
/// `iron_decode_e2m1` (`kernels/primitives.rs`) exactly — kept in lockstep so
/// this host reference and the GPU kernel can't silently drift.
pub fn e2m1_decode(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let m = (code & 0x7) as usize;
    let mag = MAG[m];
    if code & 0x8 != 0 { -mag } else { mag }
}

/// Decode an 8-bit E4M3 (fp8, no inf) code to f32. 1 sign · 4 exp (bias 7) ·
/// 3 mantissa. Mirrors iron's `iron_decode_e4m3` exactly.
pub fn e4m3_decode(code: u8) -> f32 {
    let e = ((code >> 3) & 0xf) as i32;
    let m = (code & 0x7) as f32;
    let mag = if e == 0 {
        m * 0.001_953_125 // m/8 * 2^-6 (subnormal)
    } else {
        (1.0 + m * 0.125) * 2f32.powi(e - 7)
    };
    if code & 0x80 != 0 { -mag } else { mag }
}

/// Pack a row-major `[rows, cols]` f32 weight to modelopt-convention NVFP4
/// bytes for host-side test/bench fixtures — the inverse of
/// [`SafeTensors::dequant_nvfp4_tensor_f32`], useful for constructing
/// synthetic checkpoints that round-trip through the real loader path.
/// `cols` must be a multiple of `block_size`; `block_size` a multiple of 8.
pub fn pack_nvfp4_host(w: &[f32], rows: usize, cols: usize, block_size: usize) -> (Vec<u8>, Vec<u8>, f32) {
    assert_eq!(w.len(), rows * cols, "pack_nvfp4_host: shape mismatch");
    assert_eq!(cols % block_size, 0, "pack_nvfp4_host: cols not a multiple of block_size");
    assert_eq!(block_size % 8, 0, "pack_nvfp4_host: block_size not a multiple of 8");

    let amax = w.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale2 = if amax > 0.0 { amax / (6.0 * 448.0) } else { 1.0 };
    let inv_scale2 = if scale2 > 0.0 { 1.0 / scale2 } else { 0.0 };

    let n_blocks = cols / block_size;
    let mut scales = vec![0u8; rows * n_blocks];
    let mut packed = vec![0u8; rows * cols / 2];

    for r in 0..rows {
        for b in 0..n_blocks {
            let off = r * cols + b * block_size;
            let block_amax = w[off..off + block_size].iter().fold(0f32, |a, &v| a.max(v.abs()));
            // Quantize the block scale itself to E4M3 (round-trip through the
            // decode so the packed value matches what a real produce/consume
            // pass would see).
            let raw_scale = if block_amax > 0.0 { (block_amax * inv_scale2 / 6.0).max(1e-12) } else { 1.0 / 448.0 };
            let scale_byte = f32_to_e4m3_nearest(raw_scale);
            scales[r * n_blocks + b] = scale_byte;
            let block_scale = e4m3_decode(scale_byte) * scale2;
            for i in 0..block_size {
                let col = b * block_size + i;
                let v = w[r * cols + col];
                let q = if block_scale > 0.0 { v / block_scale } else { 0.0 };
                let code = f32_to_e2m1_nearest(q);
                let byte_idx = r * (cols / 2) + col / 2;
                if col % 2 == 0 {
                    packed[byte_idx] = (packed[byte_idx] & 0xf0) | code;
                } else {
                    packed[byte_idx] = (packed[byte_idx] & 0x0f) | (code << 4);
                }
            }
        }
    }
    (packed, scales, scale2)
}

fn f32_to_e2m1_nearest(v: f32) -> u8 {
    const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = v < 0.0;
    let av = v.abs();
    let mut best = 0usize;
    let mut best_d = f32::MAX;
    for (i, &g) in GRID.iter().enumerate() {
        let d = (av - g).abs();
        if d < best_d { best_d = d; best = i; }
    }
    (best as u8) | if sign && best != 0 { 0x8 } else { 0 }
}

fn f32_to_e4m3_nearest(v: f32) -> u8 {
    // Coarse nearest-code search over all 256 codes — fine for host-side
    // test-fixture packing (not a hot path).
    let mut best = 0u8;
    let mut best_d = f32::MAX;
    for c in 0u16..256 {
        let d = (e4m3_decode(c as u8) - v).abs();
        if d < best_d { best_d = d; best = c as u8; }
    }
    best
}

impl SafeTensors {
    /// Read `<base>.weight` / `.weight_scale` / `.weight_scale_2` as a
    /// kernel-ready, zero-copy [`Nvfp4Tensor`]. Fails if any of the three
    /// tensors is missing or has an unexpected dtype/shape.
    pub fn load_nvfp4_tensor(&self, base: &str) -> Result<Nvfp4Tensor<'_>> {
        let (wb, wdt, wsh) = self.tensor(&format!("{base}.weight"))?;
        if wdt != DType::U8 {
            return Err(Error::Msg(format!("load_nvfp4_tensor '{base}': weight dtype {wdt} != U8")));
        }
        let &[rows, packed_cols] = wsh else {
            return Err(Error::Msg(format!("load_nvfp4_tensor '{base}': weight shape {wsh:?} not 2-D")));
        };
        let cols = packed_cols * 2;

        let (sb, sdt, ssh) = self.tensor(&format!("{base}.weight_scale"))?;
        if sdt != DType::U8 {
            return Err(Error::Msg(format!("load_nvfp4_tensor '{base}': weight_scale dtype {sdt} != U8")));
        }
        let &[srows, n_blocks] = ssh else {
            return Err(Error::Msg(format!("load_nvfp4_tensor '{base}': weight_scale shape {ssh:?} not 2-D")));
        };
        if srows != rows {
            return Err(Error::Msg(format!("load_nvfp4_tensor '{base}': weight rows {rows} != weight_scale rows {srows}")));
        }
        if n_blocks == 0 || cols % n_blocks != 0 {
            return Err(Error::Msg(format!("load_nvfp4_tensor '{base}': cols {cols} not a multiple of n_blocks {n_blocks}")));
        }
        let block_size = cols / n_blocks;

        let (gb, gdt, _gsh) = self.tensor(&format!("{base}.weight_scale_2"))?;
        if gdt != DType::F32 || gb.len() < 4 {
            return Err(Error::Msg(format!("load_nvfp4_tensor '{base}': weight_scale_2 dtype {gdt} (want F32 scalar)")));
        }
        let scale2 = f32::from_le_bytes(gb[..4].try_into().unwrap());

        Ok(Nvfp4Tensor { packed: wb, scales: sb, scale2, rows, cols, block_size })
    }

    /// Host-dequantize an NVFP4 tensor to a dense `[rows, cols]` f32 matrix.
    /// Not a hot path (no threading, unlike [`Self::mlx_dequant_f32`]) — for
    /// checkpoint sanity checks and small-tensor test fixtures. Mirrors
    /// `iron`'s `iron_nvfp4_qgemv` decode math exactly (`e2m1_decode` ×
    /// `e4m3_decode` × `scale2`).
    pub fn dequant_nvfp4_tensor_f32(&self, base: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let t = self.load_nvfp4_tensor(base)?;
        let mut out = vec![0f32; t.rows * t.cols];
        let n_blocks = t.cols / t.block_size;
        for r in 0..t.rows {
            for col in 0..t.cols {
                let byte = t.packed[r * (t.cols / 2) + col / 2];
                let nibble = if col % 2 == 0 { byte & 0xf } else { byte >> 4 };
                let block = col / t.block_size;
                let scale_byte = t.scales[r * n_blocks + block];
                let w = e2m1_decode(nibble) * e4m3_decode(scale_byte) * t.scale2;
                out[r * t.cols + col] = w;
            }
        }
        Ok((out, vec![t.rows, t.cols]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_grid_matches_ocp_codebook() {
        let expect = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (m, &v) in expect.iter().enumerate() {
            assert_eq!(e2m1_decode(m as u8), v);
            assert_eq!(e2m1_decode(m as u8 | 0x8), -v);
        }
    }

    #[test]
    fn e4m3_decode_known_values() {
        // 0 01111 000 = 1.0 (bias 7, exp=7 -> 2^0, mantissa 0 -> 1.0)
        assert_eq!(e4m3_decode(0b0_0111_000), 1.0);
        // 1 01111 000 = -1.0
        assert_eq!(e4m3_decode(0b1_0111_000), -1.0);
        // 0 00000 000 = 0.0
        assert_eq!(e4m3_decode(0), 0.0);
    }

    #[test]
    fn pack_unpack_roundtrip_on_grid_values() {
        // Values already on the E2M1 grid should round-trip exactly through
        // pack -> byte layout -> dequant (this is the layout self-test: it
        // does NOT depend on the missing checkpoint, only on iron's own
        // decode conventions transcribed above).
        let rows = 4;
        let cols = 32; // 2 blocks of 16
        let block_size = 16;
        let grid = [0.0f32, 0.5, -1.0, 1.5, -2.0, 3.0, -4.0, 6.0];
        let w: Vec<f32> = (0..rows * cols).map(|i| grid[i % grid.len()]).collect();
        let (packed, scales, scale2) = pack_nvfp4_host(&w, rows, cols, block_size);
        assert_eq!(packed.len(), rows * cols / 2);
        assert_eq!(scales.len(), rows * (cols / block_size));

        // Re-derive via the same decode path dequant_nvfp4_tensor_f32 uses.
        let n_blocks = cols / block_size;
        let mut max_abs_err = 0f32;
        for r in 0..rows {
            for col in 0..cols {
                let byte = packed[r * (cols / 2) + col / 2];
                let nibble = if col % 2 == 0 { byte & 0xf } else { byte >> 4 };
                let block = col / block_size;
                let scale_byte = scales[r * n_blocks + block];
                let got = e2m1_decode(nibble) * e4m3_decode(scale_byte) * scale2;
                let want = w[r * cols + col];
                max_abs_err = max_abs_err.max((got - want).abs());
            }
        }
        // Values are exactly on the E2M1 grid; the only error source is the
        // E4M3 block-scale quantization (global amax-derived scale2 combined
        // with a nearest-E4M3 per-block scale) — should be tiny for this
        // uniform-magnitude fixture.
        assert!(max_abs_err < 1e-2, "max_abs_err {max_abs_err} too large for on-grid roundtrip");
    }
}
