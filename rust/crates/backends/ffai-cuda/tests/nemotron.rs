// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! NemotronH-Nano-Omni-30B-A3B text backbone on CUDA (spark-only; 62GB BF16).
//! Not in run_all — Metal can't hold this model. NEMOTRON_DIR overrides path,
//! NEMOTRON_ARGMAX=<oracle> turns the print into an assert.
use ffai_cuda::CudaDevice;
#[test]
fn nemotron_text_forward() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    ffai_modeltests::verify_nemotron(dev.as_ref(), "GB10 sm_121");
}

/// Resident-Q8 decode throughput (quantize+upload once, time steady-state tok/s).
#[test]
fn nemotron_decode_bench() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    ffai_modeltests::bench_nemotron(dev.as_ref(), "GB10 sm_121");
}
