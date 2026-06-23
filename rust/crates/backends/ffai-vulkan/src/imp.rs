// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Real Vulkan backend (compiled under `--features vulkan`). Wraps
//! `metaltile_runtime::VulkanDevice` — the GlslGenerator → shaderc → SPIR-V →
//! VkPipeline path that runs the kernel corpus bit-accurately on any Vulkan
//! 1.2+ device (validated on AMD RX 9070 XT / RDNA4) — behind the shared
//! [`ffai_core::Device`] trait.
//!
//! ## Two buffer models: resident (default, fast) + host-shadow (fallback)
//!
//! The CUDA backend keeps tensors resident in VRAM: `alloc`/`upload` return a
//! raw device handle, and `dispatch` binds those pre-uploaded buffers to a
//! cached pipeline. This backend now mirrors that, using the metaltile Vulkan
//! resident seam (`alloc_raw` / `htod_raw` / `dtoh_raw` / `compile_kernel` /
//! `run_pipeline_bound`):
//!
//!   * **Resident path (default).** `upload`/`alloc` allocate a persistent
//!     `VulkanRawBuffer` in VRAM and copy the host bytes in ONCE. `dispatch`
//!     looks up a per-(kernel,dims) cached `VulkanPipeline` (compiled the
//!     first time only), binds the already-resident buffers, and dispatches
//!     via `run_pipeline_bound` — no per-op weight re-upload, no per-op
//!     pipeline recompile. This is the CUDA-style residency model.
//!
//!   * **Host-shadow path (fallback, `FFAI_VULKAN_HOST_SHADOW=1`).** The
//!     original correctness path: buffers are `Vec<u8>` in RAM, and each
//!     `dispatch` streams every bound buffer to VRAM, recompiles the pipeline,
//!     runs `run_kernel`, and reads the outputs back. Correct but slow
//!     (re-uploads weights + recompiles every dispatch). Kept as a safety net
//!     / A-B reference.
//!
//! Both paths produce identical results; the resident path just eliminates the
//! per-dispatch host↔device round-trips that dominate decode latency.
//!
//! ## Batched submission (resident path only, default on)
//!
//! On top of residency, the resident path BATCHES dispatches: instead of one
//! `vkQueueSubmit` + `vkQueueWaitIdle` per op, `dispatch` records each op into a
//! pending queue and a single command buffer carries many dispatches (with a
//! shader-write→read barrier between them) to ONE submit + wait. The queue is
//! flushed at a size cap (`FFAI_VULKAN_BATCH_CAP`, default 48 — ~two
//! transformer layers, bounded by the metaltile descriptor pool), at
//! `synchronize`/`download` (a read must see the results), and on device drop.
//! This removes the per-op CPU↔GPU round-trip that dominates decode latency —
//! the single biggest decode lever once weights are resident. Disable with
//! `FFAI_VULKAN_BATCH=0` to fall back to per-op `run_pipeline_bound`.

use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use ffai_core::{Backend, Binding, Device, DeviceBuffer, Error, Grid, Kernel, Result};
use metaltile_core::ir::ParamKind;
use metaltile_runtime::{
    BatchDispatch, MetalTileError, VulkanDevice as MtVulkanDevice, VulkanPipeline, VulkanRawBuffer,
};

fn dispatch_err(e: MetalTileError) -> Error {
    Error::Dispatch(e.to_string())
}

/// Whether to batch many dispatches into ONE `vkQueueSubmit` + `vkQueueWaitIdle`
/// (the default) instead of submitting + waiting per op. On by default; set
/// `FFAI_VULKAN_BATCH=0` to force the per-op `run_pipeline_bound` path (an A-B
/// reference / debugging escape hatch).
fn batch_enabled() -> bool {
    std::env::var("FFAI_VULKAN_BATCH").map(|v| v == "0" || v.is_empty()).map(|off| !off).unwrap_or(true)
}

/// Max dispatches accumulated before an automatic flush. Bounds the live
/// descriptor sets (the metaltile pool holds 1024) and the transient companion
/// VRAM held until submit, so a long forward pass can't exhaust either. A
/// transformer layer is ~15-25 dispatches, so this batches roughly two layers
/// per submit while staying well inside the pool. Override with
/// `FFAI_VULKAN_BATCH_CAP`.
fn batch_cap() -> usize {
    std::env::var("FFAI_VULKAN_BATCH_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0 && *n <= 512)
        .unwrap_or(48)
}

/// One dispatch deferred for batched submission. Owns everything the batch
/// needs alive until the single submit completes: the cached pipeline, the
/// resident buffer handles (plan order), push bytes, grid, the transient
/// strided-companion buffers to free post-flush, and `Arc`s to the bound
/// device buffers so their VRAM is NOT released by `Drop` mid-batch.
struct QueuedDispatch {
    pipeline: Arc<VulkanPipeline>,
    bufs: Vec<VulkanRawBuffer>,
    push: Vec<u8>,
    grid: [u32; 3],
    transient: Vec<VulkanRawBuffer>,
    _keepalive: Vec<Arc<dyn DeviceBuffer>>,
}

/// Whether to use the slow host-shadow path instead of the resident path.
/// Off by default (resident); set `FFAI_VULKAN_HOST_SHADOW=1` to force the
/// original streaming path for A-B comparison / debugging.
fn host_shadow_enabled() -> bool {
    std::env::var("FFAI_VULKAN_HOST_SHADOW").map(|v| v != "0" && !v.is_empty()).unwrap_or(false)
}

/// Synthesize a Strided param's `_shape` / `_strides` companion buffer
/// (row-major) from the param's static shape — mirrors metaltile's own
/// `synth_strided_meta` so the resident binding layout matches the emitter.
fn synth_strided_meta(shape: &metaltile_core::shape::Shape, strides: bool) -> Vec<u8> {
    use metaltile_core::shape::Dim;
    let dims: Vec<u32> = (0..shape.rank())
        .map(|i| match shape.dim(i) {
            Some(Dim::Known(n)) => *n as u32,
            _ => 1,
        })
        .collect();
    let vals: Vec<u32> = if strides {
        let mut s = vec![1u32; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * dims[i + 1];
        }
        s
    } else {
        dims
    };
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Vulkan implementation of the shared [`Device`] trait. Holds the metaltile
/// runtime device in an `Arc` so it outlives every buffer/dispatch, plus the
/// resident pipeline cache.
pub struct VulkanDevice {
    dev: Arc<MtVulkanDevice>,
    name: String,
    host_shadow: bool,
    /// Cache of compiled pipelines keyed by a structural signature of the
    /// kernel (name + params + constexprs + mode + block). Identical signature
    /// ⟺ identical generated GLSL ⟺ reusable pipeline.
    pipelines: Mutex<HashMap<String, Arc<VulkanPipeline>>>,
    /// Whether to defer dispatches into a single batched submit.
    batch: bool,
    /// Auto-flush threshold for the pending queue.
    batch_cap: usize,
    /// Dispatches recorded but not yet submitted. Flushed at the cap, at
    /// `synchronize`/`download` (a read needs the results), and on device drop.
    pending: Mutex<Vec<QueuedDispatch>>,
}

impl VulkanDevice {
    /// Probe for a Vulkan device; `Ok(None)` if no loader / device is present.
    pub fn create() -> Result<Option<Arc<dyn Device>>> {
        match MtVulkanDevice::create().map_err(dispatch_err)? {
            Some(d) => {
                let host_shadow = host_shadow_enabled();
                let mode = if host_shadow { "host-shadow" } else { "resident" };
                let name = format!("Vulkan device (qfam={}, {mode})", d.queue_family());
                let batch = !host_shadow && batch_enabled();
                Ok(Some(Arc::new(VulkanDevice {
                    dev: Arc::new(d),
                    name,
                    host_shadow,
                    pipelines: Mutex::new(HashMap::new()),
                    batch,
                    batch_cap: batch_cap(),
                    pending: Mutex::new(Vec::new()),
                })))
            }
            None => Ok(None),
        }
    }

    /// Structural cache key for a (kernel, block) pair. Built from the exact
    /// data GLSL codegen consumes, so equal keys guarantee identical shaders.
    fn pipeline_key(kernel: &Kernel, block: [u32; 3]) -> String {
        let mut k = format!("{}|blk={:?}|mode={:?}|", kernel.name, block, kernel.mode);
        for p in &kernel.params {
            k.push_str(&format!(
                "P({},{:?},{:?},{},{:?});",
                p.name, p.dtype, p.kind, p.is_output, p.shape
            ));
        }
        for ce in &kernel.constexprs {
            k.push_str(&format!("C({},{:?});", ce.name.name(), ce.dtype));
        }
        k
    }

    /// Get-or-compile the cached pipeline for this kernel + block.
    fn get_pipeline(&self, kernel: &Kernel, block: [u32; 3]) -> Result<Arc<VulkanPipeline>> {
        let key = Self::pipeline_key(kernel, block);
        if let Some(p) = self.pipelines.lock().unwrap().get(&key) {
            return Ok(p.clone());
        }
        let pipeline = self.dev.compile_kernel(kernel, block).map_err(dispatch_err)?;
        let arc = Arc::new(pipeline);
        self.pipelines.lock().unwrap().insert(key, arc.clone());
        Ok(arc)
    }

    /// Resident dispatch: bind pre-uploaded VRAM buffers to the cached
    /// pipeline and launch — no weight re-upload, no recompile. When batching
    /// is enabled the dispatch is recorded into the pending queue and submitted
    /// later as part of a single multi-dispatch command buffer; otherwise it is
    /// submitted immediately via `run_pipeline_bound`.
    fn dispatch_resident(&self, kernel: &Kernel, bindings: &[Binding], grid: Grid) -> Result<()> {
        let pipeline = self.get_pipeline(kernel, grid.block)?;

        // Build the binding list in plan order. Param data buffers come from
        // the resident bindings; Strided companions are synthesized + uploaded
        // as transient resident buffers (tiny, freed after the dispatch).
        let mut bufs: Vec<VulkanRawBuffer> = Vec::new();
        // Transient companion buffers, owned so they live until the dispatch
        // (or the whole batch) is submitted, then freed.
        let mut transient: Vec<VulkanRawBuffer> = Vec::new();
        // Bound device buffers kept alive until submit so their `Drop` cannot
        // free the underlying VRAM out from under an in-flight batch.
        let mut keepalive: Vec<Arc<dyn DeviceBuffer>> = Vec::new();

        // Push-constant payload: constexprs in order, then `_n_elems` if the
        // plan needs it (Elementwise) — mirrors metaltile's `run_kernel`.
        let mut push: Vec<u8> = Vec::new();

        let mut param_i = 0usize;
        let mut const_i = 0usize;
        let mut n_elems: u32 = 0;
        for b in bindings {
            match b {
                Binding::Buffer(buf) => {
                    let p = kernel.params.get(param_i).ok_or_else(|| {
                        Error::Msg(format!(
                            "dispatch: more buffer bindings than kernel params ({})",
                            kernel.params.len()
                        ))
                    })?;
                    let vb = buf
                        .as_any()
                        .downcast_ref::<VulkanBuffer>()
                        .ok_or_else(|| {
                            Error::Msg("dispatch: binding is not a VulkanBuffer".into())
                        })?;
                    let raw = vb.resident().ok_or_else(|| {
                        Error::Msg("dispatch(resident): buffer has no resident handle".into())
                    })?;
                    // Track _n_elems from the output param (element count).
                    if p.is_output {
                        let esz = p.dtype.size_bytes().max(1);
                        n_elems = (vb.len() / esz) as u32;
                    }
                    bufs.push(raw);
                    keepalive.push(buf.clone());

                    // Strided params carry shape/strides companion SSBOs.
                    if matches!(p.kind, ParamKind::Strided) {
                        for is_strides in [false, true] {
                            let meta = synth_strided_meta(&p.shape, is_strides);
                            let t = self.dev.alloc_raw(meta.len()).map_err(dispatch_err)?;
                            self.dev.htod_raw(&t, &meta).map_err(dispatch_err)?;
                            transient.push(t);
                            bufs.push(t);
                        }
                    }
                    param_i += 1;
                }
                Binding::Scalar(scalar_bytes) => {
                    // Scalars map to constexprs → push constants, in order.
                    let _ce = kernel.constexprs.get(const_i).ok_or_else(|| {
                        Error::Msg(format!(
                            "dispatch: more scalar bindings than kernel constexprs ({})",
                            kernel.constexprs.len()
                        ))
                    })?;
                    push.extend_from_slice(scalar_bytes);
                    const_i += 1;
                }
            }
        }

        // Append synthetic `_n_elems` for Elementwise kernels (plan.has_n_elems).
        if pipeline.plan.has_n_elems {
            push.extend_from_slice(&n_elems.to_le_bytes());
        }

        if self.batch {
            // Defer: queue the dispatch. Flush when the cap is hit so live
            // descriptor sets + transient VRAM stay bounded.
            let mut q = self.pending.lock().unwrap();
            q.push(QueuedDispatch {
                pipeline,
                bufs,
                push,
                grid: grid.grid,
                transient,
                _keepalive: keepalive,
            });
            let full = q.len() >= self.batch_cap;
            drop(q);
            if full {
                self.flush()?;
            }
            return Ok(());
        }

        // Per-op (unbatched) path.
        let buf_refs: Vec<&VulkanRawBuffer> = bufs.iter().collect();
        let res = self
            .dev
            .run_pipeline_bound(&pipeline, &buf_refs, &push, grid.grid)
            .map_err(dispatch_err);
        for t in &transient {
            self.dev.free_raw(t);
        }
        res
    }

    /// Submit every queued dispatch in ONE command buffer (single
    /// `vkQueueSubmit` + `vkQueueWaitIdle`), then free the batch's transient
    /// companion buffers and release the kept-alive bound buffers. A no-op when
    /// the queue is empty.
    fn flush(&self) -> Result<()> {
        let queued: Vec<QueuedDispatch> = {
            let mut q = self.pending.lock().unwrap();
            if q.is_empty() {
                return Ok(());
            }
            std::mem::take(&mut *q)
        };

        let items: Vec<BatchDispatch> = queued
            .iter()
            .map(|qd| BatchDispatch {
                pipeline: &qd.pipeline,
                bufs: qd.bufs.clone(),
                push: qd.push.clone(),
                grid: qd.grid,
            })
            .collect();

        let res = self.dev.run_pipeline_batch(&items).map_err(dispatch_err);

        // Free transient companions and drop keepalive Arcs after submit
        // completed (run_pipeline_batch waits idle), regardless of outcome.
        for qd in &queued {
            for t in &qd.transient {
                self.dev.free_raw(t);
            }
        }
        drop(items);
        drop(queued);
        res
    }

    /// Host-shadow dispatch (fallback): stream every bound buffer to VRAM,
    /// recompile, run, read back. The original correctness path.
    fn dispatch_host_shadow(
        &self,
        kernel: &Kernel,
        bindings: &[Binding],
        grid: Grid,
    ) -> Result<()> {
        let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut output_shadows: Vec<(String, Arc<dyn DeviceBuffer>)> = Vec::new();

        let mut param_i = 0usize;
        let mut const_i = 0usize;
        for b in bindings {
            match b {
                Binding::Buffer(buf) => {
                    let p = kernel.params.get(param_i).ok_or_else(|| {
                        Error::Msg(format!(
                            "dispatch: more buffer bindings than kernel params ({})",
                            kernel.params.len()
                        ))
                    })?;
                    let vb = buf
                        .as_any()
                        .downcast_ref::<VulkanBuffer>()
                        .ok_or_else(|| {
                            Error::Msg("dispatch: binding is not a VulkanBuffer".into())
                        })?;
                    buffers.insert(p.name.clone(), vb.snapshot(self.dev.as_ref()));
                    if p.is_output {
                        output_shadows.push((p.name.clone(), buf.clone()));
                    }
                    debug_assert!(p.kind != ParamKind::Scalar, "scalar param bound as a buffer");
                    param_i += 1;
                }
                Binding::Scalar(scalar_bytes) => {
                    let ce = kernel.constexprs.get(const_i).ok_or_else(|| {
                        Error::Msg(format!(
                            "dispatch: more scalar bindings than kernel constexprs ({})",
                            kernel.constexprs.len()
                        ))
                    })?;
                    buffers.insert(ce.name.name().to_string(), scalar_bytes.clone());
                    const_i += 1;
                }
            }
        }

        let out = self
            .dev
            .run_kernel(kernel, &buffers, grid.grid, grid.block)
            .map_err(dispatch_err)?;

        for (name, buf) in output_shadows {
            if let Some(result) = out.get(&name) {
                let vb = buf.as_any().downcast_ref::<VulkanBuffer>().unwrap();
                vb.store_host(self.dev.as_ref(), result);
            }
        }
        Ok(())
    }
}

/// A device buffer. In resident mode it wraps a persistent `VulkanRawBuffer`
/// in VRAM (uploaded once); in host-shadow mode it is a `Vec<u8>` in RAM that
/// is streamed per-dispatch. The two are mutually exclusive per device.
pub struct VulkanBuffer {
    /// Resident VRAM handle (Some in resident mode).
    raw: Option<VulkanRawBuffer>,
    /// Host shadow bytes (Some in host-shadow mode).
    shadow: Option<Mutex<Vec<u8>>>,
    len: usize,
    /// Device kept alive so `Drop` can free the resident VRAM allocation.
    /// `None` in host-shadow mode (nothing to free). Cloning the `Arc` here
    /// guarantees the `VkDevice` outlives every buffer that owns a handle.
    dev: Option<Arc<MtVulkanDevice>>,
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        // Free the resident VRAM allocation so transient per-token activation
        // tensors don't leak (a 16 GB card would OOM in a long decode loop
        // otherwise). Host-shadow buffers just drop their `Vec<u8>`.
        if let (Some(raw), Some(dev)) = (&self.raw, &self.dev) {
            dev.free_raw(raw);
        }
    }
}

impl VulkanBuffer {
    fn resident(&self) -> Option<VulkanRawBuffer> {
        self.raw
    }

    /// Host-shadow: snapshot current bytes. In resident mode, read them back
    /// from VRAM (only used by the host-shadow dispatch, which never holds
    /// resident buffers — this branch is defensive).
    fn snapshot(&self, dev: &MtVulkanDevice) -> Vec<u8> {
        if let Some(s) = &self.shadow {
            return s.lock().unwrap().clone();
        }
        let mut out = vec![0u8; self.len];
        if let Some(raw) = &self.raw {
            let _ = dev.dtoh_raw(raw, &mut out);
        }
        out
    }

    /// Host-shadow: write output bytes back into the shadow (or VRAM).
    fn store_host(&self, dev: &MtVulkanDevice, data: &[u8]) {
        if let Some(s) = &self.shadow {
            let mut g = s.lock().unwrap();
            let n = g.len().min(data.len());
            g[..n].copy_from_slice(&data[..n]);
            return;
        }
        if let Some(raw) = &self.raw {
            let _ = dev.htod_raw(raw, data);
        }
    }
}

impl DeviceBuffer for VulkanBuffer {
    fn len(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Device for VulkanDevice {
    fn backend(&self) -> Backend {
        Backend::Vulkan
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn alloc(&self, len: usize) -> Result<Arc<dyn DeviceBuffer>> {
        if self.host_shadow {
            return Ok(Arc::new(VulkanBuffer {
                raw: None,
                shadow: Some(Mutex::new(vec![0u8; len])),
                len,
                dev: None,
            }));
        }
        let raw = self.dev.alloc_raw(len).map_err(dispatch_err)?;
        // Zero-initialize so an unwritten alloc reads as zeros (matches the
        // host-shadow `vec![0u8; len]` semantics some kernels rely on).
        if len > 0 {
            let zeros = vec![0u8; len];
            self.dev.htod_raw(&raw, &zeros).map_err(dispatch_err)?;
        }
        Ok(Arc::new(VulkanBuffer {
            raw: Some(raw),
            shadow: None,
            len,
            dev: Some(self.dev.clone()),
        }))
    }

    fn upload(&self, bytes: &[u8]) -> Result<Arc<dyn DeviceBuffer>> {
        if self.host_shadow {
            return Ok(Arc::new(VulkanBuffer {
                raw: None,
                shadow: Some(Mutex::new(bytes.to_vec())),
                len: bytes.len(),
                dev: None,
            }));
        }
        // Uploaded buffers are model inputs (weights, embeddings, token ids):
        // written ONCE here, then read by many dispatches and never read back to
        // the host. Put them in fast DEVICE_LOCAL VRAM (staged) rather than the
        // host-visible heap `alloc_raw` uses — on a discrete GPU host-visible
        // memory is system RAM / a slow PCIe window (~12 GB/s for shader reads),
        // and the decode GEMVs are weight-bandwidth-bound, so this is the single
        // biggest decode lever (~39x faster weight reads, ~470 GB/s on RDNA4).
        let raw = self
            .dev
            .alloc_raw_device_local(bytes)
            .map_err(dispatch_err)?;
        Ok(Arc::new(VulkanBuffer {
            raw: Some(raw),
            shadow: None,
            len: bytes.len(),
            dev: Some(self.dev.clone()),
        }))
    }

    fn download(&self, buf: &dyn DeviceBuffer, out: &mut [u8]) -> Result<()> {
        // A read must observe all prior dispatches, so flush any pending batch
        // (its single submit waits idle) before touching VRAM.
        if self.batch {
            self.flush()?;
        }
        let vb = buf
            .as_any()
            .downcast_ref::<VulkanBuffer>()
            .ok_or_else(|| Error::Msg("download: buffer is not a VulkanBuffer".into()))?;
        if let Some(s) = &vb.shadow {
            let g = s.lock().unwrap();
            let n = out.len().min(g.len());
            out[..n].copy_from_slice(&g[..n]);
            return Ok(());
        }
        if let Some(raw) = &vb.raw {
            self.dev.dtoh_raw(raw, out).map_err(dispatch_err)?;
        }
        Ok(())
    }

    fn dispatch(&self, kernel: &Kernel, bindings: &[Binding], grid: Grid) -> Result<()> {
        if self.host_shadow {
            self.dispatch_host_shadow(kernel, bindings, grid)
        } else {
            self.dispatch_resident(kernel, bindings, grid)
        }
    }

    fn synchronize(&self) -> Result<()> {
        // In per-op mode every dispatch already waited idle. In batch mode the
        // queued dispatches haven't been submitted yet — flush so the barrier
        // semantics of `synchronize` hold (all prior work complete on return).
        if self.batch {
            self.flush()?;
        }
        Ok(())
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // Flush any dispatches still queued so their submit completes and their
        // transient buffers are freed before the metaltile device tears down.
        if self.batch {
            let _ = self.flush();
        }
        // Resident buffers (VulkanRawBuffer) are owned by the Tensors/engine
        // above and are not tracked here, so they are not freed on device drop
        // — the metaltile VulkanDevice's own Drop tears down the VkDevice,
        // which releases all child objects (buffers + cached pipelines).
    }
}
