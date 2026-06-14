// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! The entire shared model suite on CUDA — ONE file (mirror of the Metal one).
use ffai_cuda::CudaDevice;
#[test]
fn all_models_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    ffai_modeltests::run_all(dev.as_ref(), "GB10 sm_121");
}

/// FAST FUSE_SLICECAST lever A/B on a small same-family model (loads in seconds).
/// See `ffai_modeltests::bench_smallmodel_fuse_slicecast`. Default mamba2-130m;
/// FUSE_SC_MODEL=falcon for Falcon-H1-0.5B. Off-vs-on bit-exact + timing.
#[test]
fn smallmodel_fuse_slicecast_ab() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    ffai_modeltests::bench_smallmodel_fuse_slicecast(dev.as_ref(), "GB10 sm_121");
}
