#![cfg(feature = "cuda")]
//! strided_col_copy launch-geometry regression: the op dispatches with a
//! fixed 64-thread block. Before the fix, the grid was `n / 64` (integer
//! division, no ceiling), so whenever `s * width` wasn't a multiple of 64
//! the tail elements were silently dropped (never written into dst, left
//! as zeroed garbage). Covers a deliberately non-64-aligned shape (the
//! Laguna-style head-count case, e.g. width=72) plus a 64-aligned control
//! shape to confirm the fix didn't regress the common case.
use wh_butter_core::{DType, Device, Tensor};
use wh_butter_cuda::CudaDevice;
use wh_butter_ops::strided_col_copy;

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
fn tn(d: &dyn Device, v: &[f32]) -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32) }

// Host-computed expectation: dst[ti*width + ci] = src[ti*stride + col_off + ci].
fn host_expected(src: &[f32], s: usize, stride: usize, col_off: usize, width: usize) -> Vec<f32> {
    (0..s)
        .flat_map(|ti| (0..width).map(move |ci| src[ti * stride + col_off + ci]))
        .collect()
}

fn run_case(name: &str, s: usize, stride: usize, col_off: usize, width: usize) {
    let Some(dev) = CudaDevice::create().expect("cuda") else {
        eprintln!("no CUDA, skipping {name}");
        return;
    };
    let d = dev.as_ref();
    let src: Vec<f32> = (0..s * stride).map(|i| ((i as f32) * 0.019).sin()).collect();
    let src_t = tn(d, &src);

    let out = strided_col_copy(d, &src_t, s, stride, col_off, width)
        .unwrap_or_else(|e| panic!("{name}: strided_col_copy dispatch failed: {e:?}"));
    d.synchronize().unwrap();

    let mut bytes = vec![0u8; s * width * 4];
    d.download(out.buffer.as_ref(), &mut bytes).unwrap();
    let got = fb(&bytes);
    let want = host_expected(&src, s, stride, col_off, width);

    assert_eq!(got.len(), want.len(), "{name}: output length mismatch");
    let mut max_diff = 0f32;
    let mut first_bad: Option<usize> = None;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        if diff > max_diff { max_diff = diff; }
        if diff > 1e-6 && first_bad.is_none() { first_bad = Some(i); }
    }
    eprintln!("[{name}] s={s} stride={stride} col_off={col_off} width={width} n={} max|GPU-CPU|={max_diff:e}", s * width);
    if let Some(i) = first_bad {
        panic!(
            "{name}: mismatch at flat index {i} (ti={}, ci={}): got={} want={}",
            i / width, i % width, got[i], want[i]
        );
    }
}

/// Deliberately non-64-aligned shape: s*width = 3*72 = 216, not a multiple of
/// 64. This is the Laguna-style case (head counts 48/72) that exposed the
/// launch-geometry bug: with the old `n / 64` grid, only 192 of 216 elements
/// were ever dispatched, so the last 24 (rows 2's tail) came back as
/// zero-initialized garbage instead of the real column values.
#[test]
fn strided_col_copy_non_64_aligned_shape() {
    run_case("non_64_aligned", 3, 100, 4, 72);
}

/// Control: a 64-aligned shape (s*width divisible by 64) exercising the
/// previously-working path, to confirm the div_ceil + in-kernel guard fix
/// didn't regress the common (aligned) case.
#[test]
fn strided_col_copy_64_aligned_control() {
    run_case("64_aligned_control", 4, 200, 8, 64);
}

/// Micro-repro for the prefill layer-1 fault: f16 table gather at the exact
/// failing shapes (table [2, 3072] f16 from a cast, indices [20] u32 of 0/1).
#[test]
fn gather_f16_prefill_shapes() {
    use wh_butter_core::{DType, Tensor};
    use wh_butter_cuda::CudaDevice;
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let t = 2usize;
    let hidden = 3072usize;
    let x: Vec<f32> = (0..t * hidden).map(|i| (i as f32) * 0.001 - 3.0).collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xf32 = Tensor::new(d.upload(&xb).unwrap(), vec![t, hidden], DType::F32);
    let xf16 = wh_butter_ops::cast_f32_f16(d, &xf32).unwrap();
    let idx: Vec<u32> = vec![0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0];
    let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ti = Tensor::new(d.upload(&ib).unwrap(), vec![idx.len()], DType::U32);
    let out = wh_butter_ops::gather(d, &xf16, &ti).expect("gather f16 dispatch");
    d.synchronize().expect("post-gather sync");
    let mut ob = vec![0u8; idx.len() * hidden * 2];
    d.download(out.buffer.as_ref(), &mut ob).expect("download");
    eprintln!("gather f16 prefill shapes: OK (first half-word {:02x}{:02x})", ob[0], ob[1]);
}
