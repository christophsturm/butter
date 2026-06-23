// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-cuda
//!
//! CUDA backend for the FFAI engine. Under `--features cuda` it wraps
//! `metaltile_runtime::CudaDevice` (NVRTC → PTX → driver) behind the shared
//! [`ffai_core::Device`] trait — the same seam the Metal/Vulkan backends
//! implement, so everything above it is backend-agnostic. Without the
//! feature it is a stub so non-CUDA hosts still build the workspace.

#[cfg(feature = "cuda")]
mod imp;
#[cfg(feature = "cuda")]
pub use imp::{CudaBuffer, CudaDevice};

#[cfg(not(feature = "cuda"))]
mod stub {
    use ffai_core::{Device, Result};
    use std::sync::Arc;

    /// Stub: the crate builds on non-CUDA hosts, but no device is available
    /// until compiled with `--features cuda`.
    pub struct CudaDevice;

    impl CudaDevice {
        pub fn create() -> Result<Option<Arc<dyn Device>>> {
            Ok(None)
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use stub::CudaDevice;
