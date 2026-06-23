// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-vulkan
//!
//! Vulkan / SPIR-V backend — the portable GPU path (AMD / Intel / Android /
//! anything with a Vulkan 1.2+ driver). Under `--features vulkan` it wraps
//! `metaltile_runtime::VulkanDevice` (IR → GlslGenerator → shaderc → SPIR-V →
//! VkPipeline → dispatch) behind the shared [`ffai_core::Device`] trait — the
//! same seam the Metal / CUDA backends implement, so everything above it is
//! backend-agnostic. Without the feature it is a stub so non-Vulkan hosts
//! still build the workspace.

#[cfg(feature = "vulkan")]
mod imp;
#[cfg(feature = "vulkan")]
pub use imp::{VulkanBuffer, VulkanDevice};

#[cfg(not(feature = "vulkan"))]
mod stub {
    use ffai_core::{Device, Result};
    use std::sync::Arc;

    /// Stub: the crate builds on non-Vulkan hosts, but no device is available
    /// until compiled with `--features vulkan`.
    pub struct VulkanDevice;

    impl VulkanDevice {
        pub fn create() -> Result<Option<Arc<dyn Device>>> {
            Ok(None)
        }
    }
}

#[cfg(not(feature = "vulkan"))]
pub use stub::VulkanDevice;
