// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

/// Engine-wide error. Backends map their native failures (CUDA driver,
/// Metal, Vulkan) into these variants so the engine layer stays
/// backend-agnostic.
#[derive(Debug, Error)]
pub enum Error {
    #[error("alloc failed: {0}")]
    Alloc(String),
    #[error("device dispatch failed: {0}")]
    Dispatch(String),
    #[error("codegen failed: {0}")]
    Codegen(String),
    /// A backend crate exists but was not compiled in (feature off) or no
    /// matching hardware was found.
    #[error("backend unavailable: {0}")]
    BackendUnavailable(&'static str),
    /// A path that is scaffolded but not yet implemented.
    #[error("not yet implemented: {0}")]
    Unimplemented(&'static str),
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;
