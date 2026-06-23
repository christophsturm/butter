// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # FFAI — F*cking Fast AI
//!
//! Modular, multi-backend inference engine. One Rust core behind the
//! [`Device`] trait; backends (Metal, CUDA, Vulkan, ROCm) are independent
//! crates selected by cargo feature. Kernels are shared with the Swift
//! engine via the metaltile IR.
//!
//! ## Two engines, one product
//!
//! - **Swift FFAI** (`swift/`) — the primary Apple path. Native Metal,
//!   ships to iPhone/iPad/Mac, fast. Maintained as a first-class engine.
//! - **Rust FFAI** (this) — the cross-platform path: CUDA / Vulkan / ROCm
//!   (+ Metal for cross-validation). Same kernels, same model parity.
//!
//! Use [`devices`] to enumerate every backend compiled into this build.

pub use ffai_core::{
    Backend, Binding, Device, DeviceBuffer, DType, Error, Grid, Kernel, Result, Tensor,
};
pub use ffai_ops as ops;

use std::sync::Arc;

/// Enumerate every device from the backends compiled into this build.
/// Backends that are present-but-unimplemented (current skeleton) probe to
/// `None`, so this returns empty until a backend's `create()` goes live.
pub fn devices() -> Vec<Arc<dyn Device>> {
    let mut out: Vec<Arc<dyn Device>> = Vec::new();

    #[cfg(feature = "cuda")]
    if let Ok(Some(d)) = ffai_cuda::CudaDevice::create() {
        out.push(d);
    }
    #[cfg(feature = "metal")]
    if let Ok(Some(d)) = ffai_metal::MetalDevice::create() {
        out.push(d);
    }
    #[cfg(feature = "vulkan")]
    if let Ok(Some(d)) = ffai_vulkan::VulkanDevice::create() {
        out.push(d);
    }

    out
}

/// Names of the backends compiled into this build (regardless of whether a
/// matching device was found).
pub fn compiled_backends() -> &'static [&'static str] {
    &[
        #[cfg(feature = "metal")]
        "metal",
        #[cfg(feature = "cuda")]
        "cuda",
        #[cfg(feature = "vulkan")]
        "vulkan",
    ]
}
