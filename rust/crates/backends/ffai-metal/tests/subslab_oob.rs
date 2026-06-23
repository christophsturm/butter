// Copyright 2026 Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Metal-validation harness for the sub-slab / offset OOB class in `ffai-ops`.
//!
//! Companion to `dequant_q4_oob.rs`. Several ops address a sub-slab of a RAW
//! bound buffer purely via a host-supplied offset/stride (the tensor.offset is
//! ignored by design): `slice` (`src[off + i]`) and `strided_col_copy`
//! (`src[ti*stride + col_off + ci]`). A caller binding a buffer that does not
//! actually hold `offset + slab` elements over-reads past the end — silent
//! garbage on Metal (lenient past-the-end reads), a deterministic MMU-fault on
//! CUDA. Each op now has a defensive host-side bounds check that converts the
//! mis-size into a loud, shape-named `Err` BEFORE the GPU dispatch.
//!
//! These tests prove, per op:
//!   (a) a CORRECTLY sized call dispatches clean and bit-matches a CPU oracle,
//!   (b) an UNDERSIZED bound buffer trips the guard with a clear message
//!       (an `Err`, NOT a GPU fault / silent garbage),
//!   (c) the EXACT-fit boundary call (need == have) does NOT false-trip.
//!
//! Run with Metal GPU/shader validation ON:
//!   MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 METAL_DEVICE_WRAPPER_TYPE=1 \
//!     cargo test -p ffai-metal --test subslab_oob -- --nocapture

use ffai_core::{DType, Device, Tensor};
use ffai_metal::MetalDevice;
use ffai_ops::{slice, strided_col_copy};

fn tb_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn dl_f32(d: &dyn Device, t: &Tensor, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    d.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

// ── slice: out[i] = src[off + i] ────────────────────────────────────────────

/// (a) Correct-size slice: bind a buffer that holds off+len elems → clean +
/// bit-exact vs the CPU oracle.
#[test]
fn slice_correct_size_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let d = dev.as_ref();
    let total = 4096usize;
    let (off, len) = (1000usize, 1500usize); // off+len = 2500 < 4096
    let src: Vec<f32> = (0..total).map(|i| (i as f32 * 0.011).sin() * 3.0).collect();
    let st = Tensor::new(d.upload(&tb_f32(&src)).unwrap(), vec![total], DType::F32);

    let out = slice(d, &st, off, len).expect("correct-size slice must dispatch");
    let got = dl_f32(d, &out, len);
    let want = &src[off..off + len];
    let diff = max_abs_diff(&got, want);
    eprintln!("[slice off={off} len={len} have={total}] max|GPU-CPU| = {diff:e}");
    assert!(diff < 1e-6, "slice mismatch: {diff:e}");
}

/// (b) Undersized buffer: bind a buffer that holds FEWER than off+len elems →
/// the guard must return Err (NOT dispatch an over-read).
#[test]
fn slice_undersized_buffer_trips_guard_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let d = dev.as_ref();
    let total = 1024usize;
    let (off, len) = (900usize, 200usize); // off+len = 1100 > 1024 → OOB
    let src: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let st = Tensor::new(d.upload(&tb_f32(&src)).unwrap(), vec![total], DType::F32);

    let res = slice(d, &st, off, len);
    eprintln!(
        "[slice undersized off={off} len={len} have={total}] result = {:?}",
        res.as_ref().map(|_| "Ok")
    );
    let err = res.err().expect("undersized slice MUST return Err, not over-read");
    let msg = format!("{err:?}");
    assert!(msg.contains("slice OOB"), "guard message should name the op: {msg}");
}

/// (c) Exact-fit boundary: off+len == have. Must NOT false-trip (need == have
/// is in-bounds: the top read is src[off+len-1], the last valid element).
#[test]
fn slice_exact_fit_no_false_trip_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let d = dev.as_ref();
    let total = 2048usize;
    let (off, len) = (48usize, 2000usize); // off+len = 2048 == total
    let src: Vec<f32> = (0..total).map(|i| (i as f32 * 0.03).cos()).collect();
    let st = Tensor::new(d.upload(&tb_f32(&src)).unwrap(), vec![total], DType::F32);

    let out = slice(d, &st, off, len).expect("exact-fit slice must NOT false-trip");
    let got = dl_f32(d, &out, len);
    let want = &src[off..off + len];
    let diff = max_abs_diff(&got, want);
    eprintln!("[slice EXACT off={off} len={len} have={total}] max|GPU-CPU| = {diff:e}");
    assert!(diff < 1e-6, "exact-fit slice mismatch: {diff:e}");
}

// ── strided_col_copy: dst[ti*width+ci] = src[ti*stride + col_off + ci] ───────

fn scc_oracle(src: &[f32], s: usize, stride: usize, col_off: usize, width: usize) -> Vec<f32> {
    (0..s)
        .flat_map(|ti| (0..width).map(move |ci| src[ti * stride + col_off + ci]))
        .collect()
}

/// (a) Correct-size strided_col_copy: src holds (s-1)*stride+col_off+width
/// elems → clean + bit-exact vs CPU oracle.
#[test]
fn strided_col_copy_correct_size_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let d = dev.as_ref();
    let (s, stride, col_off, width) = (16usize, 128usize, 32usize, 64usize);
    let total = s * stride; // 2048; max read = 15*128+32+63 = 1015 < 2048
    let src: Vec<f32> = (0..total).map(|i| (i as f32 * 0.007).sin()).collect();
    let st = Tensor::new(d.upload(&tb_f32(&src)).unwrap(), vec![total], DType::F32);

    let out = strided_col_copy(d, &st, s, stride, col_off, width)
        .expect("correct-size strided_col_copy must dispatch");
    let got = dl_f32(d, &out, s * width);
    let want = scc_oracle(&src, s, stride, col_off, width);
    let diff = max_abs_diff(&got, &want);
    eprintln!("[scc s={s} stride={stride} col_off={col_off} width={width}] max|GPU-CPU| = {diff:e}");
    assert!(diff < 1e-6, "strided_col_copy mismatch: {diff:e}");
}

/// (b) Undersized buffer: bind a src whose last row's column window runs past
/// the end → guard must return Err (NOT dispatch an over-read).
#[test]
fn strided_col_copy_undersized_buffer_trips_guard_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let d = dev.as_ref();
    // need = (s-1)*stride + col_off + width = 7*100 + 90 + 50 = 840. Bind 800.
    let (s, stride, col_off, width) = (8usize, 100usize, 90usize, 50usize);
    let have = 800usize; // < 840 → the last row's window over-reads
    let src: Vec<f32> = (0..have).map(|i| i as f32).collect();
    let st = Tensor::new(d.upload(&tb_f32(&src)).unwrap(), vec![have], DType::F32);

    let res = strided_col_copy(d, &st, s, stride, col_off, width);
    eprintln!(
        "[scc undersized s={s} stride={stride} col_off={col_off} width={width} have={have}] result = {:?}",
        res.as_ref().map(|_| "Ok")
    );
    let err = res
        .err()
        .expect("undersized strided_col_copy MUST return Err, not over-read");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("strided_col_copy OOB"),
        "guard message should name the op: {msg}"
    );
}

/// (c) Exact-fit boundary: need == have. Must NOT false-trip.
#[test]
fn strided_col_copy_exact_fit_no_false_trip_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let d = dev.as_ref();
    // width is a multiple of 64 (the op's documented grid assumption: s*width
    // must be divisible by the 64-thread block). need = (s-1)*stride + col_off +
    // width = 3*200 + 30 + 64 = 694. Bind EXACTLY that many → need == have.
    let (s, stride, col_off, width) = (4usize, 200usize, 30usize, 64usize);
    let have = (s - 1) * stride + col_off + width; // 694
    let src: Vec<f32> = (0..have).map(|i| (i as f32 * 0.05).cos()).collect();
    let st = Tensor::new(d.upload(&tb_f32(&src)).unwrap(), vec![have], DType::F32);

    let out = strided_col_copy(d, &st, s, stride, col_off, width)
        .expect("exact-fit strided_col_copy must NOT false-trip");
    let got = dl_f32(d, &out, s * width);
    let want = scc_oracle(&src, s, stride, col_off, width);
    let diff = max_abs_diff(&got, &want);
    eprintln!("[scc EXACT s={s} stride={stride} col_off={col_off} width={width} have={have}] max|GPU-CPU| = {diff:e}");
    assert!(diff < 1e-6, "exact-fit strided_col_copy mismatch: {diff:e}");
}
