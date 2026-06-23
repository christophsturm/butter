// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! NemotronH / Nemotron-Cascade-2-30B-A3B BATCHED PREFILL on Metal (Apple GPU).
//!
//! Parallels the CUDA `nemotron.rs` test but exercises the backend-gated portable
//! path in `bench_nemotron`: when the device is not CUDA the projection GEMMs run
//! via `gemm_q4_mpp` (Q4-native Apple hardware MMA), the MoE / shared experts via
//! `dequant_q4_off` + `matmul`, the scan via `ssm_prefill_scan`, attention via
//! `sdpa_multi` — no cuBLAS / Marlin / CUDA-raw kernels.
//!
//! Weights: an MLX 4-bit Nemotron-Cascade checkpoint, dequantized on the fly by
//! the loader's MLX-affine path. Set NEMOTRON_DIR to the checkpoint dir.
//! Not in run_all (large model + needs local weights). NEMOTRON_PREFILL_BATCHED=1
//! and NEMOTRON_PREFILL_S=<S> drive the batched prefill inside bench_nemotron.
use ffai_metal::MetalDevice;

#[test]
fn nemotron_batched_prefill_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    ffai_modeltests::bench_nemotron(dev.as_ref(), "Apple M5 Max (Metal)");
}
