// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-metal
//!
//! Metal backend for the FFAI engine. Wraps `metaltile_runtime::Context`
//! (the public Metal face — MSL JIT, PSO cache, dispatch) behind the shared
//! [`ffai_core::Device`] trait, so the **same Rust models run on Apple GPUs
//! (and iOS)** as on CUDA. This is the half that makes "models shared across
//! CUDA *and* Metal" real.
//!
//! Each [`MetalBuffer`] keeps a host shadow (`Vec<u8>`, the `download` source of
//! truth) plus a lazily-cached GPU-resident `ResidentBuffer`, so pure inputs
//! (weights) upload **once** and bind resident across every dispatch instead of
//! being re-staged from host bytes each call. Outputs flow through host bytes
//! (preserves in-place reads + readback) and invalidate the stale resident copy.

use ffai_core::{Backend, Binding, Device, DeviceBuffer, Error, Grid, Kernel, Result};
use metaltile_runtime::{Context, DispatchSpec, MetalTileError, ResidentBuffer};
use parking_lot::{Mutex, RwLock};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

fn err(e: MetalTileError) -> Error {
    Error::Dispatch(e.to_string())
}

/// A device buffer. `data` is the host shadow (source of truth for `download`);
/// `resident` lazily caches a GPU-resident [`ResidentBuffer`] so inputs that
/// don't change between dispatches (weights!) upload **once** instead of every
/// dispatch. Writing the buffer (it being a kernel output) invalidates the
/// resident copy so the next read re-uploads the new bytes.
pub struct MetalBuffer {
    data: RwLock<Vec<u8>>,
    resident: Mutex<Option<ResidentBuffer>>,
}
// ResidentBuffer wraps an Rc<MTLBuffer> (!Send on macOS); all access is
// serialized through the MetalDevice Context mutex from one logical owner, so
// the manual Send/Sync are sound for this usage (mirrors MetalDevice).
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}
impl DeviceBuffer for MetalBuffer {
    fn len(&self) -> usize {
        self.data.read().len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Metal device: a metaltile `Context` behind the shared `Device` trait.
pub struct MetalDevice {
    ctx: Mutex<Context>,
    name: String,
}
// Context holds objc2 (`!Send`) types on macOS; we serialize all access
// through the Mutex and only ever submit from one logical owner, so the
// manual Send/Sync are sound for this usage.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    /// Probe for a Metal GPU; `Ok(None)` if none (e.g. off Apple silicon).
    pub fn create() -> Result<Option<Arc<dyn Device>>> {
        let ctx = Context::new().map_err(err)?;
        if !ctx.has_gpu() {
            return Ok(None);
        }
        let name = format!("Apple GPU (family {:?})", ctx.gpu_family());
        Ok(Some(Arc::new(MetalDevice { ctx: Mutex::new(ctx), name })))
    }

    fn shadow(b: &Arc<dyn DeviceBuffer>) -> Result<&MetalBuffer> {
        b.as_any()
            .downcast_ref::<MetalBuffer>()
            .ok_or_else(|| Error::Msg("metal: binding is not a MetalBuffer".into()))
    }
}

impl Device for MetalDevice {
    fn backend(&self) -> Backend {
        Backend::Metal
    }
    fn name(&self) -> &str {
        &self.name
    }

    fn alloc(&self, len: usize) -> Result<Arc<dyn DeviceBuffer>> {
        Ok(Arc::new(MetalBuffer { data: RwLock::new(vec![0u8; len]), resident: Mutex::new(None) }))
    }

    fn upload(&self, bytes: &[u8]) -> Result<Arc<dyn DeviceBuffer>> {
        Ok(Arc::new(MetalBuffer { data: RwLock::new(bytes.to_vec()), resident: Mutex::new(None) }))
    }

    fn download(&self, buf: &dyn DeviceBuffer, out: &mut [u8]) -> Result<()> {
        let mb = buf
            .as_any()
            .downcast_ref::<MetalBuffer>()
            .ok_or_else(|| Error::Msg("metal: download buffer is not a MetalBuffer".into()))?;
        let src = mb.data.read();
        let n = out.len().min(src.len());
        out[..n].copy_from_slice(&src[..n]);
        Ok(())
    }

    fn dispatch(&self, kernel: &Kernel, bindings: &[Binding], grid: Grid) -> Result<()> {
        let ctx = self.ctx.lock();
        let n_params = kernel.params.len();
        // Bind every buffer as GPU-resident (uploaded once, cached on the
        // MetalBuffer) and scalars as host bytes. Weights — pure inputs that
        // never change — thus upload a single time instead of every dispatch,
        // which was the dominant decode cost. Keys are param/constexpr NAMEs:
        // tensor params first, then constexprs (same contract as CUDA).
        let mut resident: BTreeMap<String, ResidentBuffer> = BTreeMap::new();
        let mut host_buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let fn_consts: BTreeMap<String, u32> = BTreeMap::new();
        for (i, b) in bindings.iter().enumerate() {
            let name = if i < n_params {
                kernel.params[i].name.clone()
            } else {
                kernel.constexprs[i - n_params].name.name().to_string()
            };
            match b {
                Binding::Scalar(s) => {
                    host_buffers.insert(name, s.clone());
                }
                Binding::Buffer(buf) => {
                    let mb = Self::shadow(buf)?;
                    // Output params go through host bytes (dispatch_chain
                    // allocates + returns them, and any in-place read sees the
                    // current value). Pure inputs bind resident + cache, so
                    // weights upload once across all dispatches.
                    if i < n_params && kernel.params[i].is_output {
                        host_buffers.insert(name, mb.data.read().clone());
                    } else {
                        let mut slot = mb.resident.lock();
                        if slot.is_none() {
                            let bytes = mb.data.read();
                            *slot = Some(ctx.upload_resident(&bytes).map_err(err)?);
                        }
                        resident.insert(name, slot.as_ref().unwrap().clone());
                    }
                }
            }
        }

        let g = [grid.grid[0] as usize, grid.grid[1] as usize, grid.grid[2] as usize];
        let t = [grid.block[0] as usize, grid.block[1] as usize, grid.block[2] as usize];
        let spec = DispatchSpec {
            kernel,
            buffers: &host_buffers,
            fn_consts: &fn_consts,
            grid_groups: g,
            threads_per_group: t,
            resident: &resident,
        };
        let results = ctx.dispatch_chain(&[spec]).map_err(err)?;
        let result = results.into_iter().next().ok_or_else(|| Error::Msg("metal: empty dispatch result".into()))?;

        // Copy each output param's result into its shadow + invalidate its
        // resident copy (it's now stale; next read re-uploads the new bytes).
        for (i, p) in kernel.params.iter().enumerate() {
            if p.is_output {
                if let Some(out_bytes) = result.output(&p.name) {
                    if let Binding::Buffer(buf) = &bindings[i] {
                        let mb = Self::shadow(buf)?;
                        {
                            let mut w = mb.data.write();
                            let n = w.len().min(out_bytes.len());
                            w[..n].copy_from_slice(&out_bytes[..n]);
                        }
                        *mb.resident.lock() = None;
                    }
                }
            }
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        // dispatch_chain commits + waits, so nothing is outstanding.
        Ok(())
    }
}
