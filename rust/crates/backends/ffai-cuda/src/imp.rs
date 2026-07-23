// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Real CUDA backend (compiled under `--features cuda`). Wraps
//! `metaltile_runtime::CudaDevice`, the NVRTC → PTX → driver path that
//! runs the kernel corpus bit-accurately on GB10, behind the shared
//! [`ffai_core::Device`] trait, so the engine layer above is identical to
//! the Metal/Vulkan paths.

use std::any::Any;
use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::{Arc, RwLock};

use ffai_core::{Backend, Binding, Device, DeviceBuffer, Error, Grid, Kernel, Result};
use metaltile_codegen::{CodegenBackend, CudaGenerator};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::{CudaDevice as MtCudaDevice, CudaModule, MetalTileError};

fn dispatch_err(e: MetalTileError) -> Error {
    Error::Dispatch(e.to_string())
}

/// A compiled module, cached for reuse. The raw `CUmodule` is single-
/// context; we only ever touch it through its owning device, so the manual
/// `Send`/`Sync` are sound for our serialized-submission usage.
///
/// Holds an `Arc<MtCudaDevice>` so the CUDA CONTEXT outlives this module. A
/// `CudaModule`'s `Drop` calls `cuModuleUnload`, which faults
/// (STATUS_ACCESS_VIOLATION, 0xc0000005) if the context was already destroyed,
/// which is exactly what happened at process teardown: this device's `dev` field
/// (its only context-keepalive) dropped before the `modules` cache, destroying
/// the context, then the cached modules unloaded into the dead context. Pinning
/// the context here (and dropping `module` before `_dev` via field order below)
/// guarantees the module always unloads while its context is still live, the
/// same invariant `CudaBuffer` already upholds for device allocations.
struct CachedModule {
    // Field DECLARATION ORDER is DROP ORDER: `module` (cuModuleUnload) must run
    // BEFORE `_dev` (which may drop the last Arc and destroy the context).
    module: CudaModule,
    _dev: Arc<MtCudaDevice>,
}
unsafe impl Send for CachedModule {}
unsafe impl Sync for CachedModule {}

/// CUDA implementation of the shared [`Device`] trait. Holds the metaltile
/// runtime device in an `Arc` so persistent tensors (which free on drop)
/// keep the CUDA context alive as long as any buffer is live.
pub struct CudaDevice {
    dev: Arc<MtCudaDevice>,
    name: String,
    /// Compile-once cache: kernel name → loaded module.
    modules: RwLock<HashMap<String, Arc<CachedModule>>>,
    /// Memoized shared-mem bytes per (kernel, block_x), avoids re-walking the
    /// kernel IR on every dispatch (hot in decode: ~hundreds of dispatches/token).
    shared: RwLock<HashMap<(String, u32), u32>>,
}

impl CudaDevice {
    /// Probe for a CUDA device; `Ok(None)` if none is present.
    pub fn create() -> Result<Option<Arc<dyn Device>>> {
        match MtCudaDevice::create().map_err(dispatch_err)? {
            Some(d) => {
                let (maj, min) = d.compute_capability();
                let dev = CudaDevice {
                    dev: Arc::new(d),
                    name: format!("CUDA device (sm_{maj}{min})"),
                    modules: RwLock::new(HashMap::new()),
                    shared: RwLock::new(HashMap::new()),
                };
                Ok(Some(Arc::new(dev)))
            }
            None => Ok(None),
        }
    }

    /// Cache key for a compiled module. The bare kernel name is NOT unique:
    /// registry kernels share one name across their dtype variants (the ops
    /// layer keys its IR cache by (name, dtype), but the same name reaches
    /// this backend for every variant). Keying compiled modules by name alone
    /// silently reused the first-compiled dtype's binary for every other
    /// dtype, which reads with the wrong element stride and faults (or worse,
    /// corrupts silently). Fold each param's dtype into the key.
    fn module_key(kernel: &Kernel) -> String {
        use std::fmt::Write as _;
        let mut key = kernel.name.clone();
        key.push('|');
        for p in &kernel.params {
            let _ = write!(key, "{:?},", p.dtype);
        }
        key
    }

    fn module_for(&self, kernel: &Kernel) -> Result<Arc<CachedModule>> {
        let mkey = Self::module_key(kernel);
        if let Some(m) = self.modules.read().unwrap().get(&mkey) {
            return Ok(m.clone());
        }
        let cg = CudaGenerator::new();
        let src = cg
            .generate(kernel)
            .map_err(|e| Error::Codegen(format!("{e:?}")))?;
        // MT_DUMP_CUDA_SRC=<dir>: dump generated CUDA C++ for kernel inspection.
        if let Ok(dir) = std::env::var("MT_DUMP_CUDA_SRC") {
            let path = format!("{}/{}.cu", dir, kernel.name);
            let _ = std::fs::write(&path, &src);
        }
        let module = self
            .dev
            .compile(&src, &format!("{}.cu", kernel.name))
            .map_err(dispatch_err)?;
        let cached = Arc::new(CachedModule { module, _dev: self.dev.clone() });
        self.modules
            .write()
            .unwrap()
            .insert(mkey, cached.clone());
        Ok(cached)
    }
}

/// A persistent CUDA allocation. Frees on drop; holds an `Arc` of the
/// device so the context outlives the buffer.
pub struct CudaBuffer {
    ptr: u64,
    len: usize,
    dev: Arc<MtCudaDevice>,
}
// ptr is a plain integer handle; dev is an Arc. Sound to move/share for our
// serialized usage.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl DeviceBuffer for CudaBuffer {
    fn len(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        // Return to the device's size-bucketed pool instead of a synchronous
        // cuMemFree, the hot decode path reuses these every token.
        self.dev.free_raw_pooled(self.ptr, self.len);
    }
}

/// Element count of the kernel's first output param, derived from the
/// binding at that index. Used for the synthetic `_n_elems` Elementwise arg.
fn first_output_elems(kernel: &Kernel, bindings: &[Binding]) -> u32 {
    if let Some(i) = kernel.params.iter().position(|p| p.is_output) {
        let dt = kernel.params[i].dtype.size_bytes().max(1);
        if let Some(Binding::Buffer(b)) = bindings.get(i) {
            return (b.len() / dt) as u32;
        }
    }
    0
}

impl Device for CudaDevice {
    fn backend(&self) -> Backend {
        Backend::Cuda
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn alloc(&self, len: usize) -> Result<Arc<dyn DeviceBuffer>> {
        let ptr = self.dev.alloc_raw(len).map_err(dispatch_err)?;
        Ok(Arc::new(CudaBuffer { ptr, len, dev: self.dev.clone() }))
    }

    fn upload(&self, bytes: &[u8]) -> Result<Arc<dyn DeviceBuffer>> {
        let ptr = self.dev.alloc_raw(bytes.len()).map_err(dispatch_err)?;
        self.dev.htod(ptr, bytes).map_err(dispatch_err)?;
        Ok(Arc::new(CudaBuffer { ptr, len: bytes.len(), dev: self.dev.clone() }))
    }

    fn alloc_zeroed(&self, len: usize) -> Result<Arc<dyn DeviceBuffer>> {
        // Stream-ordered device memset, no host zero buffer, no pageable
        // H2D copy. A [s,hid] f32 MoE accumulator is ~22 MB; uploading zeros
        // for it cost ~4-6 ms of host-blocking copy per E-layer.
        let ptr = self.dev.alloc_raw(len).map_err(dispatch_err)?;
        self.dev.memset_zero_raw(ptr, len).map_err(dispatch_err)?;
        Ok(Arc::new(CudaBuffer { ptr, len, dev: self.dev.clone() }))
    }

    fn download(&self, buf: &dyn DeviceBuffer, out: &mut [u8]) -> Result<()> {
        let cb = buf
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| Error::Msg("download: buffer is not a CudaBuffer".into()))?;
        self.dev.dtoh(cb.ptr, out).map_err(dispatch_err)
    }

    fn dispatch(&self, kernel: &Kernel, bindings: &[Binding], grid: Grid) -> Result<()> {
        let module = self.module_for(kernel)?;
        let func = module.module.function(&kernel.name).map_err(dispatch_err)?;
        // Same dtype-collision hazard as the module cache: shared-memory sizing
        // depends on element widths, so fold the param dtypes in.
        let skey = (Self::module_key(kernel), grid.block[0]);
        let shared = if let Some(&s) = self.shared.read().unwrap().get(&skey) {
            s
        } else {
            let s = CudaGenerator::new().shared_bytes(kernel, grid.block[0]) as u32;
            self.shared.write().unwrap().insert(skey, s);
            s
        };

        // Marshal kernel args: device ptrs / scalar bytes in binding order,
        // then the synthetic _n_elems for Elementwise kernels. `ptr_store`
        // and `scalar_store` back the raw pointers, so they outlive `args`.
        let mut ptr_store: Vec<u64> = Vec::new();
        let mut scalar_store: Vec<Vec<u8>> = Vec::new();
        enum Slot {
            Ptr(usize),
            Scalar(usize),
        }
        let mut slots: Vec<Slot> = Vec::new();
        for b in bindings {
            match b {
                Binding::Buffer(buf) => {
                    let cb = buf
                        .as_any()
                        .downcast_ref::<CudaBuffer>()
                        .ok_or_else(|| Error::Msg("dispatch: binding is not a CudaBuffer".into()))?;
                    slots.push(Slot::Ptr(ptr_store.len()));
                    ptr_store.push(cb.ptr);
                }
                Binding::Scalar(bytes) => {
                    slots.push(Slot::Scalar(scalar_store.len()));
                    scalar_store.push(bytes.clone());
                }
            }
        }
        if kernel.mode == KernelMode::Elementwise {
            let n = first_output_elems(kernel, bindings);
            slots.push(Slot::Scalar(scalar_store.len()));
            scalar_store.push(n.to_le_bytes().to_vec());
        }

        let mut args: Vec<*mut c_void> = Vec::with_capacity(slots.len());
        for s in &slots {
            match s {
                Slot::Ptr(i) => args.push(&ptr_store[*i] as *const u64 as *mut c_void),
                Slot::Scalar(i) => args.push(scalar_store[*i].as_ptr() as *mut c_void),
            }
        }

        // Async launch (no per-dispatch cuCtxSynchronize), kernels pipeline on the
        // ordered default stream; `download`/`synchronize` sync when results are read.
        self.dev
            .launch_async(func, grid.grid, grid.block, shared, &mut args)
            .map_err(dispatch_err)
    }

    fn synchronize(&self) -> Result<()> {
        self.dev.synchronize().map_err(dispatch_err)
    }

    fn begin_capture(&self) -> Result<()> {
        self.dev.begin_capture().map_err(dispatch_err)
    }
    fn end_capture(&self) -> Result<u64> {
        self.dev.end_capture().map(|exec| exec as usize as u64).map_err(dispatch_err)
    }
    fn graph_launch(&self, exec: u64) -> Result<()> {
        self.dev.graph_launch(exec as usize as *mut std::ffi::c_void).map_err(dispatch_err)
    }
    fn graph_launch_batch(&self, exec: u64, n: usize) -> Result<()> {
        self.dev.graph_launch_batch(exec as usize as *mut std::ffi::c_void, n).map_err(dispatch_err)
    }

    fn dispatch_raw_cuda(
        &self,
        src: &str,
        prog_name: &str,
        fn_name: &str,
        ptrs: &[(&dyn ffai_core::DeviceBuffer, usize)],
        scalars: &[Vec<u8>],
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        cooperative: bool,
    ) -> ffai_core::Result<()> {
        // Compile (or reuse cached) module.
        let module = {
            let r = self.modules.read().unwrap();
            r.get(fn_name).cloned()
        };
        let module = if let Some(m) = module {
            m
        } else {
            let m = self.dev.compile(src, prog_name).map_err(dispatch_err)?;
            let cached = Arc::new(CachedModule { module: m, _dev: self.dev.clone() });
            self.modules.write().unwrap().insert(fn_name.to_string(), cached.clone());
            cached
        };
        let func = module.module.function(fn_name).map_err(dispatch_err)?;

        // Build args: device pointers (with offset applied) then scalar bytes.
        let mut ptr_store: Vec<u64> = Vec::new();
        let mut scalar_store: Vec<Vec<u8>> = Vec::new();
        enum Slot { Ptr(usize), Scalar(usize) }
        let mut slots: Vec<Slot> = Vec::new();

        for (buf, offset) in ptrs {
            let cb = buf.as_any().downcast_ref::<CudaBuffer>()
                .ok_or_else(|| ffai_core::Error::Msg("dispatch_raw_cuda: buffer is not CudaBuffer".into()))?;
            slots.push(Slot::Ptr(ptr_store.len()));
            ptr_store.push(cb.ptr + *offset as u64);
        }
        for s in scalars {
            slots.push(Slot::Scalar(scalar_store.len()));
            scalar_store.push(s.clone());
        }
        if fn_name.contains("marlin") && std::env::var("NEMOTRON_MARLIN_DBG").is_ok() {
            eprintln!("[MARLINPTR] fn={} grid={:?} block={:?} smem={} ptrs={:?} scalars_lens={:?}",
                fn_name, grid, block, shared_bytes,
                ptr_store.iter().map(|p| format!("0x{:x}", p)).collect::<Vec<_>>(),
                scalars.iter().map(|s| s.len()).collect::<Vec<_>>());
        }

        let mut args: Vec<*mut c_void> = slots.iter().map(|s| match s {
            Slot::Ptr(i)    => &ptr_store[*i]    as *const u64 as *mut c_void,
            Slot::Scalar(i) => scalar_store[*i].as_ptr() as *mut c_void,
        }).collect();

        // Use cooperative launch only when NOT inside a CUDA graph capture
        // (cuLaunchCooperativeKernel is not capturable). Fall back to regular
        // launch during capture, the caller (moe_fused_ffn) must not use
        // grid.sync() in that code path (handled by NEMOTRON_GRAPH exclusion).
        if cooperative && !self.dev.is_capturing() {
            self.dev.launch_async_coop(func, grid, block, shared_bytes, &mut args).map_err(dispatch_err)
        } else {
            self.dev.launch_async(func, grid, block, shared_bytes, &mut args).map_err(dispatch_err)
        }
    }

    fn gemm_tc(
        &self,
        x: &dyn DeviceBuffer,
        w: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
        dtype: ffai_core::DType,
    ) -> Result<()> {
        let xb = x.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_tc: x not CudaBuffer".into()))?;
        let wb = w.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_tc: w not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_tc: out not CudaBuffer".into()))?;
        self.dev.gemm_cublas(xb.ptr, wb.ptr, ob.ptr, m, n, k, dtype).map_err(dispatch_err)
    }

    fn gemm_tc_out_f32(
        &self,
        x: &dyn DeviceBuffer,
        w: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
        ab_dtype: ffai_core::DType,
    ) -> Result<()> {
        let xb = x.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_tc_out_f32: x not CudaBuffer".into()))?;
        let wb = w.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_tc_out_f32: w not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_tc_out_f32: out not CudaBuffer".into()))?;
        self.dev.gemm_cublas_f32out(xb.ptr, wb.ptr, ob.ptr, m, n, k, ab_dtype).map_err(dispatch_err)
    }

    fn gemm_fp4(
        &self,
        x_pack: &dyn DeviceBuffer,
        x_sf: &dyn DeviceBuffer,
        w_pack: &dyn DeviceBuffer,
        w_sf: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
        out_f32: bool,
        d_scale: Option<&dyn DeviceBuffer>,
    ) -> Result<()> {
        let xb = x_pack.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp4: x_pack not CudaBuffer".into()))?;
        let xs = x_sf.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp4: x_sf not CudaBuffer".into()))?;
        let wb = w_pack.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp4: w_pack not CudaBuffer".into()))?;
        let ws = w_sf.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp4: w_sf not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp4: out not CudaBuffer".into()))?;
        let ds = match d_scale {
            Some(t) => t.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp4: d_scale not CudaBuffer".into()))?.ptr,
            None => 0u64,
        };
        self.dev.gemm_cublaslt_fp4(xb.ptr, xs.ptr, wb.ptr, ws.ptr, ob.ptr, m, n, k, out_f32, ds).map_err(dispatch_err)
    }

    fn gemm_fp8(
        &self,
        x: &dyn DeviceBuffer,
        x_scale: &dyn DeviceBuffer,
        w: &dyn DeviceBuffer,
        w_scale: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
        out_f32: bool,
    ) -> Result<()> {
        let xb = x.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp8: x not CudaBuffer".into()))?;
        let xs = x_scale.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp8: x_scale not CudaBuffer".into()))?;
        let wb = w.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp8: w not CudaBuffer".into()))?;
        let ws = w_scale.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp8: w_scale not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_fp8: out not CudaBuffer".into()))?;
        self.dev.gemm_cublaslt_fp8(xb.ptr, xs.ptr, wb.ptr, ws.ptr, ob.ptr, m, n, k, out_f32).map_err(dispatch_err)
    }

    fn moe_grouped_cutlass(
        &self,
        a: &dyn DeviceBuffer,
        w: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        group_rows: &[i32],
        expert_ids: &[i32],
        n: usize,
        k: usize,
    ) -> Result<()> {
        let ab = a.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("moe_grouped_cutlass: a not CudaBuffer".into()))?;
        let wb = w.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("moe_grouped_cutlass: w not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("moe_grouped_cutlass: out not CudaBuffer".into()))?;
        self.dev.moe_grouped_cutlass(ab.ptr, wb.ptr, ob.ptr, group_rows, expert_ids, n, k).map_err(dispatch_err)
    }
    #[allow(clippy::too_many_arguments)]
    fn moe_marlin_gemm(
        &self,
        a: &dyn DeviceBuffer, b: &dyn DeviceBuffer, c: &dyn DeviceBuffer, c_tmp: &dyn DeviceBuffer,
        b_scales: &dyn DeviceBuffer, global_scale: &dyn DeviceBuffer, topk_weights: &dyn DeviceBuffer,
        sorted_token_ids: &dyn DeviceBuffer, expert_ids: &dyn DeviceBuffer, num_tokens_past_padded: &dyn DeviceBuffer,
        workspace: &dyn DeviceBuffer,
        moe_block: usize, num_experts: usize, top_k: usize, mul_topk_weights: bool,
        prob_m: usize, prob_n: usize, prob_k: usize, num_groups: usize, group_size: usize,
        sms: i32, use_fp32_reduce: bool,
    ) -> Result<()> {
        let g = |x: &dyn DeviceBuffer, nm: &str| x.as_any().downcast_ref::<CudaBuffer>().map(|b| b.ptr)
            .ok_or_else(|| Error::Msg(format!("moe_marlin_gemm: {nm} not CudaBuffer")));
        self.dev.moe_marlin_gemm(
            g(a,"a")?, g(b,"b")?, g(c,"c")?, g(c_tmp,"c_tmp")?,
            g(b_scales,"b_scales")?, g(global_scale,"global_scale")?, g(topk_weights,"topk_weights")?,
            g(sorted_token_ids,"sorted_token_ids")?, g(expert_ids,"expert_ids")?, g(num_tokens_past_padded,"ntpp")?,
            g(workspace,"workspace")?,
            moe_block, num_experts, top_k, mul_topk_weights, prob_m, prob_n, prob_k, num_groups, group_size, sms, use_fp32_reduce,
        ).map_err(dispatch_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn marlin_repack(
        &self, b_q: &dyn DeviceBuffer, out: &dyn DeviceBuffer,
        size_k: usize, size_n: usize, sms: i32, max_shared_mem: i32,
    ) -> Result<()> {
        let g = |x: &dyn DeviceBuffer, nm: &str| x.as_any().downcast_ref::<CudaBuffer>().map(|b| b.ptr)
            .ok_or_else(|| Error::Msg(format!("marlin_repack: {nm} not CudaBuffer")));
        self.dev.marlin_repack(g(b_q,"b_q")?, g(out,"out")?, size_k, size_n, sms, max_shared_mem).map_err(dispatch_err)
    }
    #[allow(clippy::too_many_arguments)]
    fn marlin_build_routing(&self,off:&dyn DeviceBuffer,n_exp:usize,blk:usize,mt:usize,stid:&dyn DeviceBuffer,eid:&dyn DeviceBuffer,ntpp:&dyn DeviceBuffer)->Result<()>{
        let gg=|x:&dyn DeviceBuffer,nm:&str| x.as_any().downcast_ref::<CudaBuffer>().map(|b|b.ptr).ok_or_else(||Error::Msg(format!("marlin_build_routing: {nm} not CudaBuffer")));
        self.dev.marlin_build_routing(gg(off,"off")?,n_exp,blk,mt,gg(stid,"stid")?,gg(eid,"eid")?,gg(ntpp,"ntpp")?).map_err(dispatch_err)
    }

    fn moe_grouped_cutlass_fp4(
        &self,
        a: &dyn DeviceBuffer,
        sfa: &dyn DeviceBuffer,
        w: &dyn DeviceBuffer,
        sfw: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        group_rows: &[i32],
        expert_ids: &[i32],
        sfa_off: &[i64],
        alpha_vec: Option<&dyn DeviceBuffer>,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let dc = |b: &dyn DeviceBuffer, what: &str| -> Result<u64> {
            Ok(b.as_any()
                .downcast_ref::<CudaBuffer>()
                .ok_or_else(|| Error::Msg(format!("moe_grouped_cutlass_fp4: {what} not CudaBuffer")))?
                .ptr)
        };
        let (ap, sfap, wp, sfwp, op) =
            (dc(a, "a")?, dc(sfa, "sfa")?, dc(w, "w")?, dc(sfw, "sfw")?, dc(out, "out")?);
        let alp = match alpha_vec {
            Some(b) => dc(b, "alpha_vec")?,
            None => 0,
        };
        self.dev
            .moe_grouped_cutlass_fp4(ap, sfap, wp, sfwp, op, group_rows, expert_ids, sfa_off, alp, n, k)
            .map_err(dispatch_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_cutlass_w4a8(
        &self,
        a: &dyn DeviceBuffer,
        sfa: &dyn DeviceBuffer,
        w: &dyn DeviceBuffer,
        sfw: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        group_rows: &[i32],
        expert_ids: &[i32],
        sfa_off: &[i64],
        sfb_exp_bytes: i64,
        alpha_vec: Option<&dyn DeviceBuffer>,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let dc = |b: &dyn DeviceBuffer, what: &str| -> Result<u64> {
            Ok(b.as_any()
                .downcast_ref::<CudaBuffer>()
                .ok_or_else(|| Error::Msg(format!("moe_grouped_cutlass_w4a8: {what} not CudaBuffer")))?
                .ptr)
        };
        let (ap, sfap, wp, sfwp, op) =
            (dc(a, "a")?, dc(sfa, "sfa")?, dc(w, "w")?, dc(sfw, "sfw")?, dc(out, "out")?);
        let alp = match alpha_vec {
            Some(b) => dc(b, "alpha_vec")?,
            None => 0,
        };
        self.dev
            .moe_grouped_cutlass_w4a8(ap, sfap, wp, sfwp, op, group_rows, expert_ids, sfa_off, sfb_exp_bytes, alp, n, k)
            .map_err(dispatch_err)
    }

    fn w4a8_actquant(
        &self,
        x: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        sf: &dyn DeviceBuffer,
        group_rows: &[i32],
        sfa_off: &[i64],
        n: usize,
        k: usize,
    ) -> Result<()> {
        let dc = |b: &dyn DeviceBuffer, what: &str| -> Result<u64> {
            Ok(b.as_any()
                .downcast_ref::<CudaBuffer>()
                .ok_or_else(|| Error::Msg(format!("w4a8_actquant: {what} not CudaBuffer")))?
                .ptr)
        };
        let (xp, op, sfp) = (dc(x, "x")?, dc(out, "out")?, dc(sf, "sf")?);
        self.dev
            .w4a8_actquant_run(xp, op, sfp, group_rows, sfa_off, n, k)
            .map_err(dispatch_err)
    }

    fn w4a8_packw(
        &self,
        w: &dyn DeviceBuffer,
        outp: &dyn DeviceBuffer,
        sf: &dyn DeviceBuffer,
        n_exp: usize,
        n: usize,
        k: usize,
    ) -> Result<i64> {
        let dc = |b: &dyn DeviceBuffer, what: &str| -> Result<u64> {
            Ok(b.as_any()
                .downcast_ref::<CudaBuffer>()
                .ok_or_else(|| Error::Msg(format!("w4a8_packw: {what} not CudaBuffer")))?
                .ptr)
        };
        let (wp, op, sfp) = (dc(w, "w")?, dc(outp, "outp")?, dc(sf, "sf")?);
        self.dev
            .w4a8_packw_run(wp, op, sfp, n_exp, n, k)
            .map_err(dispatch_err)
    }

    fn moe_grouped_cutlass_w8a8(
        &self,
        a: &dyn DeviceBuffer,
        sfa: &dyn DeviceBuffer,
        w: &dyn DeviceBuffer,
        sfw: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        group_rows: &[i32],
        expert_ids: &[i32],
        sfa_off: &[i64],
        sfb_exp_bytes: i64,
        alpha_vec: Option<&dyn DeviceBuffer>,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let dc = |b: &dyn DeviceBuffer, what: &str| -> Result<u64> {
            Ok(b.as_any()
                .downcast_ref::<CudaBuffer>()
                .ok_or_else(|| Error::Msg(format!("moe_grouped_cutlass_w8a8: {what} not CudaBuffer")))?
                .ptr)
        };
        let (ap, sfap, wp, sfwp, op) =
            (dc(a, "a")?, dc(sfa, "sfa")?, dc(w, "w")?, dc(sfw, "sfw")?, dc(out, "out")?);
        let alp = match alpha_vec {
            Some(b) => dc(b, "alpha_vec")?,
            None => 0,
        };
        self.dev
            .moe_grouped_cutlass_w8a8(ap, sfap, wp, sfwp, op, group_rows, expert_ids, sfa_off, sfb_exp_bytes, alp, n, k)
            .map_err(dispatch_err)
    }

    fn w8a8_actquant(
        &self,
        x: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        sf: &dyn DeviceBuffer,
        group_rows: &[i32],
        sfa_off: &[i64],
        n: usize,
        k: usize,
    ) -> Result<()> {
        let dc = |b: &dyn DeviceBuffer, what: &str| -> Result<u64> {
            Ok(b.as_any()
                .downcast_ref::<CudaBuffer>()
                .ok_or_else(|| Error::Msg(format!("w8a8_actquant: {what} not CudaBuffer")))?
                .ptr)
        };
        let (xp, op, sfp) = (dc(x, "x")?, dc(out, "out")?, dc(sf, "sf")?);
        self.dev
            .w8a8_actquant_run(xp, op, sfp, group_rows, sfa_off, n, k)
            .map_err(dispatch_err)
    }

    fn w8a8_packw(
        &self,
        w: &dyn DeviceBuffer,
        outp: &dyn DeviceBuffer,
        sf: &dyn DeviceBuffer,
        n_exp: usize,
        n: usize,
        k: usize,
    ) -> Result<i64> {
        let dc = |b: &dyn DeviceBuffer, what: &str| -> Result<u64> {
            Ok(b.as_any()
                .downcast_ref::<CudaBuffer>()
                .ok_or_else(|| Error::Msg(format!("w8a8_packw: {what} not CudaBuffer")))?
                .ptr)
        };
        let (wp, op, sfp) = (dc(w, "w")?, dc(outp, "outp")?, dc(sf, "sf")?);
        self.dev
            .w8a8_packw_run(wp, op, sfp, n_exp, n, k)
            .map_err(dispatch_err)
    }

    fn moe_grouped_cutlass_fp4_prepare(
        &self,
        w: &dyn DeviceBuffer,
        sfw: &dyn DeviceBuffer,
        alpha: &dyn DeviceBuffer,
        n_groups: usize,
        n: usize,
        k: usize,
        max_m_total: usize,
    ) -> Result<u64> {
        let wb = w.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fp4_prepare: w not CudaBuffer".into()))?;
        let sb = sfw.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fp4_prepare: sfw not CudaBuffer".into()))?;
        let al = alpha.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fp4_prepare: alpha not CudaBuffer".into()))?;
        self.dev.moe_grouped_cutlass_fp4_prepare(wb.ptr, sb.ptr, al.ptr, n_groups, n, k, max_m_total).map_err(dispatch_err)
    }

    fn moe_grouped_cutlass_fp4_run(
        &self,
        handle: u64,
        a: &dyn DeviceBuffer,
        sfa: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        off_dev: &dyn DeviceBuffer,
    ) -> Result<()> {
        let ab = a.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fp4_run: a not CudaBuffer".into()))?;
        let sb = sfa.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fp4_run: sfa not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fp4_run: out not CudaBuffer".into()))?;
        let fb = off_dev.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fp4_run: off not CudaBuffer".into()))?;
        self.dev.moe_grouped_cutlass_fp4_run(handle, ab.ptr, sb.ptr, ob.ptr, fb.ptr).map_err(dispatch_err)
    }

    fn moe_grouped_cutlass_fp4_FUSEDACT_prepare(
        &self,
        w: &dyn DeviceBuffer,
        sfw: &dyn DeviceBuffer,
        alpha: &dyn DeviceBuffer,
        norm_constant: &dyn DeviceBuffer,
        n_groups: usize,
        n: usize,
        k: usize,
        max_m_total: usize,
    ) -> Result<u64> {
        let wb = w.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fused_prepare: w not CudaBuffer".into()))?;
        let sb = sfw.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fused_prepare: sfw not CudaBuffer".into()))?;
        let al = alpha.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fused_prepare: alpha not CudaBuffer".into()))?;
        let nc = norm_constant.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fused_prepare: norm_constant not CudaBuffer".into()))?;
        self.dev.moe_grouped_cutlass_fp4_FUSEDACT_prepare(wb.ptr, sb.ptr, al.ptr, nc.ptr, n_groups, n, k, max_m_total).map_err(dispatch_err)
    }

    fn moe_grouped_cutlass_fp4_FUSEDACT_run(
        &self,
        handle: u64,
        a: &dyn DeviceBuffer,
        sfa: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        sfd: &dyn DeviceBuffer,
        off_dev: &dyn DeviceBuffer,
    ) -> Result<()> {
        let ab = a.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fused_run: a not CudaBuffer".into()))?;
        let sb = sfa.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fused_run: sfa not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fused_run: out not CudaBuffer".into()))?;
        let db = sfd.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fused_run: sfd not CudaBuffer".into()))?;
        let fb = off_dev.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("fused_run: off not CudaBuffer".into()))?;
        self.dev.moe_grouped_cutlass_fp4_FUSEDACT_run(handle, ab.ptr, sb.ptr, ob.ptr, db.ptr, fb.ptr).map_err(dispatch_err)
    }

    fn moe_grouped_cutlass_fp4_AMAX_prepare(
        &self,
        w: &dyn DeviceBuffer,
        sfw: &dyn DeviceBuffer,
        d_amax: &dyn DeviceBuffer,
        n_groups: usize,
        n: usize,
        k: usize,
        max_m_total: usize,
    ) -> Result<u64> {
        let wb = w.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("amax_prepare: w not CudaBuffer".into()))?;
        let sb = sfw.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("amax_prepare: sfw not CudaBuffer".into()))?;
        let am = d_amax.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("amax_prepare: d_amax not CudaBuffer".into()))?;
        self.dev.moe_grouped_cutlass_fp4_AMAX_prepare(wb.ptr, sb.ptr, am.ptr, n_groups, n, k, max_m_total).map_err(dispatch_err)
    }

    fn moe_grouped_cutlass_fp4_AMAX_run(
        &self,
        handle: u64,
        a: &dyn DeviceBuffer,
        sfa: &dyn DeviceBuffer,
        out: &dyn DeviceBuffer,
        off_dev: &dyn DeviceBuffer,
    ) -> Result<()> {
        let ab = a.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("amax_run: a not CudaBuffer".into()))?;
        let sb = sfa.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("amax_run: sfa not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("amax_run: out not CudaBuffer".into()))?;
        let fb = off_dev.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("amax_run: off not CudaBuffer".into()))?;
        self.dev.moe_grouped_cutlass_fp4_AMAX_run(handle, ab.ptr, sb.ptr, ob.ptr, fb.ptr).map_err(dispatch_err)
    }

    fn gemm_tc_off(
        &self,
        x: &dyn DeviceBuffer, x_off: usize,
        w: &dyn DeviceBuffer, w_off: usize,
        out: &dyn DeviceBuffer, out_off: usize,
        m: usize, n: usize, k: usize, dtype: ffai_core::DType,
    ) -> Result<()> {
        let xb = x.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_tc_off: x not CudaBuffer".into()))?;
        let wb = w.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_tc_off: w not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_tc_off: out not CudaBuffer".into()))?;
        self.dev.gemm_cublas(
            xb.ptr + x_off as u64,
            wb.ptr + w_off as u64,
            ob.ptr + out_off as u64,
            m, n, k, dtype,
        ).map_err(dispatch_err)
    }

    fn gemm_strided_batched(
        &self,
        x: &dyn DeviceBuffer, stride_x: i64,
        w: &dyn DeviceBuffer, stride_w: i64,
        out: &dyn DeviceBuffer, stride_out: i64,
        m: usize, n: usize, k: usize,
        batch_count: usize,
        dtype: ffai_core::DType,
    ) -> Result<()> {
        let xb = x.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_strided_batched: x not CudaBuffer".into()))?;
        let wb = w.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_strided_batched: w not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_strided_batched: out not CudaBuffer".into()))?;
        self.dev.gemm_cublas_strided_batched(
            xb.ptr, stride_x,
            wb.ptr, stride_w,
            ob.ptr, stride_out,
            m, n, k, batch_count, dtype,
        ).map_err(dispatch_err)
    }

    fn gemm_strided_batched_off(
        &self,
        x: &dyn DeviceBuffer, x_off: usize, stride_x: i64,
        w: &dyn DeviceBuffer, w_off: usize, stride_w: i64,
        out: &dyn DeviceBuffer, out_off: usize, stride_out: i64,
        m: usize, n: usize, k: usize,
        batch_count: usize,
        dtype: ffai_core::DType,
    ) -> Result<()> {
        let xb = x.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_strided_batched_off: x not CudaBuffer".into()))?;
        let wb = w.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_strided_batched_off: w not CudaBuffer".into()))?;
        let ob = out.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_strided_batched_off: out not CudaBuffer".into()))?;
        self.dev.gemm_cublas_strided_batched(
            xb.ptr + x_off as u64, stride_x,
            wb.ptr + w_off as u64, stride_w,
            ob.ptr + out_off as u64, stride_out,
            m, n, k, batch_count, dtype,
        ).map_err(dispatch_err)
    }

    fn gemm_batched(
        &self,
        x_buf: &dyn ffai_core::DeviceBuffer,
        x_offsets: &[usize],
        w_buf: &dyn ffai_core::DeviceBuffer,
        w_offsets: &[usize],
        out_buf: &dyn ffai_core::DeviceBuffer,
        out_offsets: &[usize],
        m: usize,
        n: usize,
        k: usize,
        dtype: ffai_core::DType,
    ) -> Result<()> {
        let xb = x_buf.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_batched: x_buf not CudaBuffer".into()))?;
        let wb = w_buf.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_batched: w_buf not CudaBuffer".into()))?;
        let ob = out_buf.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_batched: out_buf not CudaBuffer".into()))?;
        let x_ptrs:   Vec<u64> = x_offsets.iter().map(|&off| xb.ptr + off as u64).collect();
        let w_ptrs:   Vec<u64> = w_offsets.iter().map(|&off| wb.ptr + off as u64).collect();
        let out_ptrs: Vec<u64> = out_offsets.iter().map(|&off| ob.ptr + off as u64).collect();
        self.dev.gemm_cublas_batched(&x_ptrs, &w_ptrs, &out_ptrs, m, n, k, dtype)
            .map_err(dispatch_err)
    }

    fn gemm_grouped(
        &self,
        x_buf: &dyn ffai_core::DeviceBuffer,
        x_offsets: &[usize],
        w_buf: &dyn ffai_core::DeviceBuffer,
        w_offsets: &[usize],
        out_buf: &dyn ffai_core::DeviceBuffer,
        out_offsets: &[usize],
        m_per_group: &[i32],
        n: usize,
        k: usize,
        dtype: ffai_core::DType,
    ) -> Result<()> {
        let xb  = x_buf.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_grouped: x_buf not CudaBuffer".into()))?;
        let wb  = w_buf.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_grouped: w_buf not CudaBuffer".into()))?;
        let ob  = out_buf.as_any().downcast_ref::<CudaBuffer>().ok_or_else(|| Error::Msg("gemm_grouped: out_buf not CudaBuffer".into()))?;
        let x_ptrs:   Vec<u64> = x_offsets.iter().map(|&off| xb.ptr + off as u64).collect();
        let w_ptrs:   Vec<u64> = w_offsets.iter().map(|&off| wb.ptr + off as u64).collect();
        let out_ptrs: Vec<u64> = out_offsets.iter().map(|&off| ob.ptr + off as u64).collect();
        self.dev.gemm_cublas_grouped(&x_ptrs, &w_ptrs, &out_ptrs, m_per_group, n, k, dtype)
            .map_err(dispatch_err)
    }
}
