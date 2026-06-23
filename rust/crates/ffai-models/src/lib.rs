// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-models
//!
//! Model forward passes as plain Rust over [`ffai_ops`]. This is the big
//! surface ported from FFAI-Swift `Models/` (~35 families) — written ONCE
//! and run on every backend via the shared [`ffai_core::Device`] trait. The
//! 35 families collapse to a handful of *builders* parameterized by config;
//! this module starts with the transformer-LLM builder (Llama / Qwen /
//! Gemma / Mistral / Yi / Phi / SmolLM / …).

use ffai_core::{Device, Result, Tensor};

/// A loaded model: weights resident on some [`Device`], able to run a
/// forward pass. Backend-neutral — the same impl runs on Metal/CUDA/Vulkan.
pub trait Model: Send + Sync {
    fn name(&self) -> &str;
    /// Run one forward pass over `tokens`, returning next-token logits.
    fn forward(&self, dev: &dyn Device, tokens: &[u32]) -> Result<Tensor>;
}

pub mod dsv4;
pub mod gguf_tokenizer;
pub mod llama;
pub mod moe;
