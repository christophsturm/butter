// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Per-dispatch overhead micro-benchmark (Rust → Metal). Isolates the cost of
//! ONE kernel dispatch on the Apple GPU — the quantity that decides whether the
//! Rust host adds overhead vs native Swift. It does not: a dispatch is an
//! `MTLCommandBuffer.commit()` + wait, a Metal/driver cost paid identically
//! whether the encoder is driven from Swift (MetalKit) or Rust (objc2). The
//! decode bottleneck is the dispatch COUNT (~120/token), not the host language.
use ffai_core::{DType, Tensor};
use ffai_metal::MetalDevice;
use ffai_ops::{add, gemv};
use std::time::Instant;

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

#[test]
fn metal_per_dispatch_overhead() {
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    let d = dev.as_ref();
    let up = |v: &[f32], sh: Vec<usize>| Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32);

    // tiny elementwise add — work is ~0, so timing = pure dispatch overhead
    let n = 4096usize;
    let a = up(&vec![1.0f32; n], vec![n]);
    let b = up(&vec![2.0f32; n], vec![n]);
    // realistic decode op: a [2048x2048] gemv (the projection shape in a 2B model)
    let m = 2048usize;
    let w = up(&vec![0.01f32; m * m], vec![m, m]);
    let x = up(&vec![0.1f32; m], vec![m]);

    // warmup (JIT + PSO)
    for _ in 0..20 { let _ = add(d, &a, &b).unwrap(); let _ = gemv(d, &w, &x).unwrap(); }

    let iters = 3000;
    let t0 = Instant::now();
    for _ in 0..iters { let _ = add(d, &a, &b).unwrap(); }
    let add_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters { let _ = gemv(d, &w, &x).unwrap(); }
    let gemv_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

    eprintln!("Rust → Metal per-dispatch overhead (Apple GPU), {iters} iters each:");
    eprintln!("  tiny add (4096):      {add_us:.1} µs/dispatch");
    eprintln!("  gemv (2048x2048):     {gemv_us:.1} µs/dispatch");
    eprintln!("  → a ~120-dispatch decode step ≈ {:.2} ms just in dispatch overhead", add_us * 120.0 / 1000.0);
    eprintln!("  This commit()+wait cost is the same Metal driver call Swift makes; the host language is not the variable.");
}
