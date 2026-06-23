// Copyright 2026 Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Metal-validation cross-check for the CUDA `dequant_q4_off` MMU-fault.
//!
//! The Nemotron batched-prefill CUDA path deterministically MMU-faults (Xid 31,
//! VIRT_READ out-of-bounds) and the prime suspect is `ffai_ops::dequant_q4_off`
//! — the 1-thread-per-u32-word Q4 dequant used for the dense projections
//! (blk_off=0) and the per-MoE-expert sub-slabs (blk_off != 0). This test calls
//! that op directly at both shapes, under Metal's GPU/shader validation layer
//! (enabled via env in the test harness; see comment at the bottom), and
//! bit-compares the output against a CPU reference dequant to catch a silent
//! over-read that lands in valid memory on Metal but faults at the buffer
//! boundary on CUDA.
//!
//! Run with validation ON:
//!   MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 METAL_DEVICE_WRAPPER_TYPE=1 \
//!     cargo test -p ffai-metal --test dequant_q4_oob -- --nocapture
//!
//! Nemotron-Cascade-2-30B-A3B dims: hid=2688, inter=1856, n_exp=128.

use ffai_core::{DType, Device, Tensor};
use ffai_metal::MetalDevice;
use ffai_ops::{dequant_q4_off, quantize_q4};

// ── f16 round-trip (mirrors the loader's tb_f16 + the kernel's f16 read) ──────
fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32 - 112; // 127 - 15
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

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h as u32) >> 10) & 0x1f;
    let mant = (h as u32) & 0x3ff;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((e + 127 - 15) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn tb_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn tb_f16(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&f| f32_to_f16(f).to_le_bytes()).collect()
}
fn dl_f16(d: &dyn Device, t: &Tensor, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 2];
    d.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// CPU reference dequant of the `[m,k]` slab at block offset `blk_off` inside a
/// `qs`/`scales` pool. Scales already rounded through f16. Mirrors the kernel.
fn cpu_dequant_off(
    qs: &[u32],
    scales_f16: &[f32],
    m: usize,
    k: usize,
    blk_off: usize,
) -> Vec<f32> {
    let bpr = k / 32;
    let mut out = vec![0f32; m * k];
    for r in 0..m {
        for b in 0..bpr {
            let blk = blk_off + r * bpr + b;
            let d = scales_f16[blk];
            for word in 0..4 {
                let packed = qs[blk * 4 + word];
                for i in 0..8 {
                    let nib = (packed >> (i * 4)) & 0xf;
                    let q = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                    out[r * k + b * 32 + word * 8 + i] = q as f32 * d;
                }
            }
        }
    }
    out
}

/// Build a Q4 pool of `n_rows` rows × `k` cols, scales rounded through f16 to
/// match the GPU's f16 scale read. Returns (qs, scales_f16_rounded).
fn build_pool(n_rows: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let n = n_rows * k;
    let vals: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.013 - 0.4).sin() * 1.7)
        .collect();
    let (qs, scales) = quantize_q4(&vals, n_rows, k);
    let scales_f16: Vec<f32> = scales.iter().map(|&s| f16_to_f32(f32_to_f16(s))).collect();
    (qs, scales_f16)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// (a) Dense projection: blk_off=0, full pool. m=k=2688 (hid).
#[test]
fn dequant_q4_dense_projection_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let d = dev.as_ref();
    let (m, k) = (2688usize, 2688usize);
    let (qs, sc16) = build_pool(m, k);
    let qt = Tensor::new(d.upload(&tb_u32(&qs)).unwrap(), vec![qs.len()], DType::U32);
    let st = Tensor::new(d.upload(&tb_f16(&sc16)).unwrap(), vec![sc16.len()], DType::F16);

    let out = dequant_q4_off(d, &qt, &st, m, k, DType::F16, 0).unwrap();
    let got = dl_f16(d, &out, m * k);
    let want = cpu_dequant_off(&qs, &sc16, m, k, 0);
    let diff = max_abs_diff(&got, &want);
    eprintln!("[dense blk_off=0 m={m} k={k}] max|GPU-CPU| = {diff:e}");
    // Tolerance = a few f16 ULP at this magnitude (our hand-rolled f16 oracle is
    // not bit-identical to Metal's storage rounding; >1e-2 would mean real
    // corruption / an over-read pulling in garbage).
    assert!(diff < 1e-2, "dense projection dequant mismatch: {diff:e}");
}

/// (b) MoE expert sub-slab: blk_off = e*inter*(hid/32), full pool present.
/// Exercises every expert offset including the LAST (e=n_exp-1), whose top
/// block is the final block of the pool — the CUDA boundary-fault candidate.
#[test]
fn dequant_q4_expert_subslab_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let d = dev.as_ref();
    let (n_exp, inter, hid) = (128usize, 1856usize, 2688usize);
    let up_bpr = hid / 32; // Q4 blocks per up-weight row
    // up pool: [n_exp*inter, hid]
    let (qs, sc16) = build_pool(n_exp * inter, hid);
    let qt = Tensor::new(d.upload(&tb_u32(&qs)).unwrap(), vec![qs.len()], DType::U32);
    let st = Tensor::new(d.upload(&tb_f16(&sc16)).unwrap(), vec![sc16.len()], DType::F16);

    // Spot-check the first, a middle, and the LAST expert (boundary case).
    for &e in &[0usize, n_exp / 2, n_exp - 1] {
        let blk_off = e * inter * up_bpr;
        let out = dequant_q4_off(d, &qt, &st, inter, hid, DType::F16, blk_off).unwrap();
        let got = dl_f16(d, &out, inter * hid);
        let want = cpu_dequant_off(&qs, &sc16, inter, hid, blk_off);
        let diff = max_abs_diff(&got, &want);
        eprintln!("[expert e={e} blk_off={blk_off} m={inter} k={hid}] max|GPU-CPU| = {diff:e}");
        assert!(diff < 1e-2, "expert {e} dequant mismatch: {diff:e}");
    }
}

/// (c) TIGHT-POOL boundary probe: a pool sized to EXACTLY the last expert's
/// slab (blk_off points at offset 0 of a buffer that holds only ONE expert's
/// worth of blocks). The kernel's top read is `qs[(blk_off + (m-1)*bpr +
/// (bpr-1))*4 + 3]`. If blk_off is non-zero but the bound buffer does NOT
/// actually contain `blk_off + m*bpr` blocks, the kernel over-reads. This is
/// what the CUDA MMU faults on. We bind a buffer that is exactly `m*bpr*4`
/// u32 words but pass a non-zero blk_off → guaranteed over-read of the qs
/// (and scales) buffer. Under Metal shader validation this MUST flag an OOB.
#[test]
fn dequant_q4_undersized_pool_overread_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let d = dev.as_ref();
    let (inter, hid) = (1856usize, 2688usize);
    let up_bpr = hid / 32;
    // Pool holds exactly ONE expert's blocks, but we ask for expert index 1.
    let (qs, sc16) = build_pool(inter, hid);
    let qt = Tensor::new(d.upload(&tb_u32(&qs)).unwrap(), vec![qs.len()], DType::U32);
    let st = Tensor::new(d.upload(&tb_f16(&sc16)).unwrap(), vec![sc16.len()], DType::F16);
    let blk_off = 1 * inter * up_bpr; // points PAST the end of this 1-expert pool

    eprintln!(
        "[overread] pool words={} but reading from blk_off={} (word {}) — expect Metal OOB flag",
        qs.len(),
        blk_off,
        blk_off * 4
    );
    // We do NOT assert numeric correctness here — the point is whether Metal's
    // validation layer reports the OOB read. If validation is off, Metal reads
    // garbage/zeros from past the buffer (lenient) and this returns Ok.
    let res = dequant_q4_off(d, &qt, &st, inter, hid, DType::F16, blk_off);
    eprintln!("[overread] dispatch result = {:?}", res.map(|_| "Ok"));
}
