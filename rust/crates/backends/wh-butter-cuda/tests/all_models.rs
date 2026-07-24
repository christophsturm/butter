// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! The entire shared model suite on CUDA — ONE file (mirror of the Metal one).
use wh_butter_cuda::CudaDevice;
#[test]
fn all_models_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    wh_butter_modeltests::run_all(dev.as_ref(), "GB10 sm_121");
}

// NOTE(rebrand-merge): this file used to also have a `smallmodel_fuse_slicecast_ab`
// test calling `wh_butter_modeltests::bench_smallmodel_fuse_slicecast`, but that
// function doesn't exist anywhere in the modeltests crate on the original
// PR #70 (a27773ec) commit either (pre-existing dead reference, not
// something this cherry-pick introduced — the FUSE_SLICECAST bench was
// apparently never actually landed in wh-butter-modeltests). Dropped rather
// than stubbed since there's no reference implementation to port.
