// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! The entire shared model suite on Metal — ONE file. The forwards live once in
//! ffai-modeltests; this just builds the device and runs them. A new backend
//! (ROCm/Vulkan/…) copies this file with its own Device. See docs/BACKENDS.md.
use ffai_metal::MetalDevice;
#[test]
fn all_models_on_metal() {
    let Some(dev) = MetalDevice::create().expect("metal") else { eprintln!("no Metal — skip"); return; };
    ffai_modeltests::run_all(dev.as_ref(), "Apple GPU");
}
