//! Decisive GB10 experiment: does CUDA-graph capture (the megakernel form on this
//! stack) cut per-token overhead? Times N back-to-back kernel dispatches launched
//! SEQUENTIALLY (async, one ordered stream) vs the same N captured as ONE graph and
//! REPLAYED. If graph << sequential, the inter-kernel host/launch bubble is real and
//! the all-device-token + capture build (phases 2-3) is worth it. The research could
//! not answer this for GB10 (all megakernel data was B200).
//!
//! Run: cargo test --release -p ffai-cuda --features cuda --test graph_bench -- --nocapture

#![cfg(feature = "cuda")]

use ffai_core::{DType, Tensor};
use ffai_ops::add;
use std::time::Instant;

#[test]
fn graph_vs_sequential_overhead() {
    let dev = match ffai_cuda::CudaDevice::create().unwrap() {
        Some(d) => d,
        None => {
            eprintln!("no CUDA device; skipping");
            return;
        }
    };
    let d = dev.as_ref();
    // Small buffers so the kernel is launch/overhead-bound, not bandwidth-bound —
    // that isolates the per-dispatch bubble (decode has ~390 such dispatches/token).
    let n = 2688usize;
    let a = Tensor::new(d.upload(&vec![0u8; n * 4]).unwrap(), vec![n], DType::F32);
    let b = Tensor::new(d.upload(&vec![0u8; n * 4]).unwrap(), vec![n], DType::F32);

    let launches = 390usize; // ~one decode token's dispatch count
    // Warm up: compile the kernel + fill the buffer pool so neither sequential nor
    // captured paths pay a one-time cuMemAlloc.
    for _ in 0..50 {
        let _ = add(d, &a, &b).unwrap();
    }
    d.synchronize().unwrap();

    // ── Sequential: a DEPENDENT chain (each add reads the prior output) — this is
    // the realistic decode shape; the RAW dependency exposes the inter-kernel bubble
    // (GPU idle between dependent kernels) that graph replay eliminates. ──
    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut x = add(d, &a, &b).unwrap();
        for _ in 0..launches - 1 {
            x = add(d, &x, &b).unwrap();
        }
        d.synchronize().unwrap();
        let _ = x;
    }
    let seq_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // ── Graph: capture the same dependent chain once, then replay ──
    d.begin_capture().unwrap();
    {
        let mut x = add(d, &a, &b).unwrap();
        for _ in 0..launches - 1 {
            x = add(d, &x, &b).unwrap();
        }
        let _ = x;
    }
    let exec = d.end_capture().unwrap();

    let t1 = Instant::now();
    for _ in 0..iters {
        d.graph_launch(exec).unwrap();
    }
    let graph_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    eprintln!("──── CUDA-graph vs sequential ({launches} dispatches) ────");
    eprintln!("  sequential: {seq_ms:.3} ms/token  ({:.2} us/launch)", seq_ms * 1000.0 / launches as f64);
    eprintln!("  graph replay: {graph_ms:.3} ms/token  ({:.2} us/launch)", graph_ms * 1000.0 / launches as f64);
    eprintln!("  speedup: {:.2}x  (overhead removed: {:.3} ms/token)", seq_ms / graph_ms, seq_ms - graph_ms);
    eprintln!("  ⇒ at 32K decode (~22 ms/token, ~3.4 ms host gap), graph could recover ~{:.1} ms ⇒ verdict on megakernel.", (seq_ms - graph_ms).min(3.4));
}
