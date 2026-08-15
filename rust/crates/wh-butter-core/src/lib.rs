// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # wh-butter-core
//!
//! Backend-neutral primitives for the Butter inference engine. This crate
//! defines the single seam every backend implements — the [`Device`]
//! trait — plus the [`Tensor`] handle and the dtype/buffer/binding types
//! that flow through it unchanged on Metal, CUDA, Vulkan, or ROCm.
//!
//! ## How code is shared
//!
//! Kernels are shared via the **iron IR** ([`Kernel`], re-exported
//! from `wh-iron-core`). A model op builds or looks up a `Kernel` and
//! hands it to [`Device::dispatch`]; the backend lowers that IR to its
//! target language (MSL / CUDA C++ / SPIR-V) and launches it. So:
//!
//! - **Above** this trait (models, ops, loaders, KV cache, sampler) is
//!   plain Rust that never names a GPU API — written once, runs everywhere.
//! - **Below** it, each backend is a thin [`Device`] impl.
//! - The kernels themselves are generated once by iron.

use std::any::Any;
use std::sync::Arc;

mod error;
pub use error::{Error, Result};

// Re-export the shared kernel IR + dtype so the whole engine speaks one
// vocabulary and nothing downstream depends on wh-iron-core directly.
pub use wh_iron_core::dtype::DType;
pub use wh_iron_core::ir::Kernel;

/// Which accelerator family a [`Device`] targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Metal,
    Cuda,
    Vulkan,
    Rocm,
    Cpu,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::Rocm => "rocm",
            Backend::Cpu => "cpu",
        }
    }
}

/// An opaque device-side allocation. Each backend returns its own concrete
/// type (an `MTLBuffer` wrapper, a `CUdeviceptr` wrapper, a `VkBuffer`
/// wrapper) behind this trait, so [`Tensor`] stays backend-agnostic.
pub trait DeviceBuffer: Send + Sync {
    /// Length in bytes.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Escape hatch so backend code can downcast to its concrete buffer
    /// type when it needs the native handle for a launch.
    fn as_any(&self) -> &dyn Any;
}

/// A single kernel argument in signature order: either a device buffer or a
/// small by-value scalar/constexpr (little-endian bytes).
#[derive(Clone)]
pub enum Binding {
    Buffer(Arc<dyn DeviceBuffer>),
    Scalar(Vec<u8>),
}

/// Launch geometry: grid (blocks) × block (threads-per-block), 3-D. Maps
/// onto both CUDA's grid/block and Metal's threadgroups / threads-per-tg.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

impl Grid {
    /// 1-D launch: `blocks` threadgroups of `threads` lanes each.
    pub fn d1(blocks: u32, threads: u32) -> Self {
        Grid { grid: [blocks, 1, 1], block: [threads, 1, 1] }
    }
}

/// The one interface every backend implements. Object-safe so the engine
/// holds `Arc<dyn Device>` and dispatches without knowing the hardware.
pub trait Device: Send + Sync {
    fn backend(&self) -> Backend;
    /// Human-readable device name (e.g. `"Apple M5 Max"`, `"NVIDIA GB10"`).
    fn name(&self) -> &str;

    /// Allocate `len` bytes of uninitialized device memory.
    fn alloc(&self, len: usize) -> Result<Arc<dyn DeviceBuffer>>;

    /// Allocate + upload host bytes in one shot.
    fn upload(&self, bytes: &[u8]) -> Result<Arc<dyn DeviceBuffer>>;

    /// Allocate `len` bytes of zero-initialized device memory. Backends
    /// should override with a device-side fill — the default uploads a host
    /// zero buffer, which for multi-MB accumulators is a pageable H2D copy.
    fn alloc_zeroed(&self, len: usize) -> Result<Arc<dyn DeviceBuffer>> {
        self.upload(&vec![0u8; len])
    }

    /// Copy device memory back into a host slice.
    fn download(&self, buf: &dyn DeviceBuffer, out: &mut [u8]) -> Result<()>;

    /// Lower `kernel` (shared iron IR) for this backend and launch it
    /// over `grid` with `bindings` in signature order.
    fn dispatch(&self, kernel: &Kernel, bindings: &[Binding], grid: Grid) -> Result<()>;

    /// Block until all submitted work has completed.
    fn synchronize(&self) -> Result<()>;

    /// CUDA-graph capture (megakernel). `begin_capture` starts recording stream
    /// work; run an all-device (no host-sync) sequence; `end_capture` returns an
    /// opaque executable-graph handle; `graph_launch` replays it as ONE launch.
    /// Default impls error (backend without graph support). Returns handle as u64.
    fn begin_capture(&self) -> Result<()> {
        Err(Error::Msg("graph capture unsupported on this backend".into()))
    }
    fn end_capture(&self) -> Result<u64> {
        Err(Error::Msg("graph capture unsupported on this backend".into()))
    }
    fn graph_launch(&self, _exec: u64) -> Result<()> {
        Err(Error::Msg("graph capture unsupported on this backend".into()))
    }
    /// Issue `n` sequential graph launches on the GPU stream without syncing
    /// between them; sync once at the end. Eliminates per-token host-GPU
    /// handoff for the throughput benchmark. Default impl falls back to
    /// n individual `graph_launch` calls (correct but not pipelined).
    fn graph_launch_batch(&self, exec: u64, n: usize) -> Result<()> {
        for _ in 0..n { self.graph_launch(exec)?; }
        Ok(())
    }

    /// Escape hatch: compile raw CUDA C++ via NVRTC and launch a single
    /// function with the provided device-buffer pointer list (as `u64`
    /// CUdeviceptr values) plus a small scalar-bytes trailer.
    ///
    /// - `src`: full CUDA C++ source
    /// - `prog_name`: source file name hint (e.g. `"fused_moe.cu"`)
    /// - `fn_name`: `__global__` function to call
    /// - `ptrs`: ordered list of (CUdeviceptr, byte-offset) pairs
    /// - `scalars`: packed scalar bytes appended after ptrs in kernel arg order
    /// - `grid`/`block`/`shared_bytes`: launch geometry
    /// - `cooperative`: use `cuLaunchCooperativeKernel` for global grid-sync.
    ///   Falls back to regular launch if a CUDA graph capture is in progress
    ///   (cooperative launches are not capturable). Caller must handle the
    ///   graph-mode fallback externally if correctness requires the barrier.
    ///
    /// Returns `Err(unsupported)` on backends that don't implement this.
    fn dispatch_raw_cuda(
        &self,
        _src: &str,
        _prog_name: &str,
        _fn_name: &str,
        _ptrs: &[(&dyn DeviceBuffer, usize)],  // (buf, byte_offset)
        _scalars: &[Vec<u8>],
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_bytes: u32,
        _cooperative: bool,
    ) -> Result<()> {
        Err(Error::Msg("dispatch_raw_cuda unsupported on this backend".into()))
    }

    /// Tensor-core GEMM escape hatch (Path A). Computes the ROW-MAJOR product
    /// `out[m,n] = x[m,k] · w[n,k]ᵀ` (the projection `out[r,o]=Σ_k w[o,k]·x[r,k]`)
    /// via the backend's hardware matmul (cuBLAS tensor cores on CUDA). Inputs +
    /// output are device buffers of `dtype` (f16/bf16); accumulation is f32.
    /// Default impl errors on backends without a tensor-core matmul.
    #[allow(clippy::too_many_arguments)]
    fn gemm_tc(
        &self,
        _x: &dyn DeviceBuffer,
        _w: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
        _dtype: DType,
    ) -> Result<()> {
        Err(Error::Msg("gemm_tc (tensor-core GEMM) unsupported on this backend".into()))
    }

    /// f16/bf16 inputs, **f32 output** — fuses the post-GEMM `cast_f16_f32` into
    /// the matmul (cuBLAS writes the f32 result directly, accumulate is already
    /// f32). `out` must be an f32 buffer of `[m,n]`. Default errors; CUDA
    /// implements it via cublasLt with an f32 D layout.
    #[allow(clippy::too_many_arguments)]
    fn gemm_tc_out_f32(
        &self,
        _x: &dyn DeviceBuffer,
        _w: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
        _ab_dtype: DType,
    ) -> Result<()> {
        Err(Error::Msg("gemm_tc_out_f32 unsupported on this backend".into()))
    }

    /// Block-scaled NVFP4 GEMM (Blackwell): `out[m,n](f16) = X·Wᵀ` with both
    /// operands packed e2m1 (2/byte) + per-16-element ue4m3 scale tensors in
    /// the hardware's swizzled block layout. Default errors; CUDA implements
    /// it via cublasLt (measured ~4x the f16 tensor-core path on GB10).
    #[allow(clippy::too_many_arguments)]
    fn gemm_fp4(
        &self,
        _x_pack: &dyn DeviceBuffer,
        _x_sf: &dyn DeviceBuffer,
        _w_pack: &dyn DeviceBuffer,
        _w_sf: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
        _out_f32: bool,
        _d_scale: Option<&dyn DeviceBuffer>,
    ) -> Result<()> {
        Err(Error::Msg("gemm_fp4 (block-scaled NVFP4 GEMM) unsupported on this backend".into()))
    }

    /// FP8 (e4m3) GEMM with per-tensor f32 device scale pointers:
    /// `out[m,n] = sx*sw * X·Wᵀ`. Default errors; CUDA implements it via
    /// cublasLt (~2x the f16 tensor-core rate, far gentler quantization
    /// than FP4).
    #[allow(clippy::too_many_arguments)]
    fn gemm_fp8(
        &self,
        _x: &dyn DeviceBuffer,
        _x_scale: &dyn DeviceBuffer,
        _w: &dyn DeviceBuffer,
        _w_scale: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
        _out_f32: bool,
    ) -> Result<()> {
        Err(Error::Msg("gemm_fp8 unsupported on this backend".into()))
    }

    /// CUTLASS grouped MoE GEMM: `out[t,n] = Σ_k a[t,k]·w[eid(t)][n,k]` over
    /// sorted-token groups. `a` f16 `[mt,K]`, `w` contiguous f16 expert slab
    /// `[n_exp,N,K]`, `out` f16 `[mt,N]`; `group_rows`/`expert_ids` are host
    /// slices. Default errors; CUDA implements it via the AOT CUTLASS lib (when
    /// the runtime was built with CUTLASS_DIR).
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_cutlass(
        &self,
        _a: &dyn DeviceBuffer,
        _w: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _group_rows: &[i32],
        _expert_ids: &[i32],
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(Error::Msg("moe_grouped_cutlass unsupported on this backend".into()))
    }
    /// Validated NVFP4 grouped-MoE GEMM (marlin). Device buffers; routing arrays
    /// (`sorted_token_ids`/`expert_ids`/`num_tokens_past_padded`) are DEVICE i32.
    #[allow(clippy::too_many_arguments)]
    fn moe_marlin_gemm(
        &self,
        _a: &dyn DeviceBuffer, _b: &dyn DeviceBuffer, _c: &dyn DeviceBuffer, _c_tmp: &dyn DeviceBuffer,
        _b_scales: &dyn DeviceBuffer, _global_scale: &dyn DeviceBuffer, _topk_weights: &dyn DeviceBuffer,
        _sorted_token_ids: &dyn DeviceBuffer, _expert_ids: &dyn DeviceBuffer, _num_tokens_past_padded: &dyn DeviceBuffer,
        _workspace: &dyn DeviceBuffer,
        _moe_block: usize, _num_experts: usize, _top_k: usize, _mul_topk_weights: bool,
        _prob_m: usize, _prob_n: usize, _prob_k: usize, _num_groups: usize, _group_size: usize,
        _sms: i32, _use_fp32_reduce: bool,
    ) -> Result<()> {
        Err(Error::Msg("moe_marlin_gemm unsupported on this backend".into()))
    }

    /// Repack one expert's GPTQ-packed `[K/8,N]` u32 weights into marlin tile layout.
    #[allow(clippy::too_many_arguments)]
    fn marlin_repack(
        &self, _b_q: &dyn DeviceBuffer, _out: &dyn DeviceBuffer,
        _size_k: usize, _size_n: usize, _sms: i32, _max_shared_mem: i32,
    ) -> Result<()> {
        Err(Error::Msg("marlin_repack unsupported on this backend".into()))
    }
    #[allow(clippy::too_many_arguments)]
    fn marlin_build_routing(&self, _off:&dyn DeviceBuffer,_n_exp:usize,_blk:usize,_mt:usize,_stid:&dyn DeviceBuffer,_eid:&dyn DeviceBuffer,_ntpp:&dyn DeviceBuffer)->Result<()>{
        Err(Error::Msg("marlin_build_routing unsupported on this backend".into()))
    }

    /// F-85 round-8: scale-column permutation companion to [`marlin_repack`]
    /// for the symmetric U4B8 (bias-8) small-M decode GEMM below -- required
    /// once per weight after repack (round-7's correctness fix; un-permuted
    /// scales silently mis-select columns, 5x-500x relative error). `in_`/
    /// `out` are `[num_groups, size_n]` f16 device buffers; `out` may not
    /// alias `in_`.
    fn marlin_permute_scales(
        &self, _in_: &dyn DeviceBuffer, _out: &dyn DeviceBuffer,
        _num_groups: usize, _size_n: usize,
    ) -> Result<()> {
        Err(Error::Msg("marlin_permute_scales unsupported on this backend".into()))
    }

    /// F-85 round-8: small-M symmetric-u4 (kU4B8) W4A16 decode GEMM. `a` f16
    /// `[m,k]`, `b_repacked` = [`marlin_repack`] output of a `[k/8,n]` u32
    /// GPTQ-packed u4b8 weight, `b_scales` = [`marlin_permute_scales`]
    /// output, `c` f16 `[m,n]` out, `c_tmp` f32 scratch. Routing arrays
    /// (`sorted_token_ids`/`expert_ids`/`num_tokens_past_padded`) come from
    /// [`marlin_build_routing`] with a trivial single-"expert" `[0,m)` range
    /// -- this is marlin's generic grouped-GEMM driver run dense (no MoE
    /// routing of its own). Narrower than [`moe_marlin_gemm`] (no
    /// global_scale/topk_weights -- the u4b8 C shim hardwires those null;
    /// see round-7/8's marlin_shim.cu). Default errors; CUDA implements it
    /// via the AOT marlin lib (IRON_MARLIN_BUILD=1 builds).
    #[allow(clippy::too_many_arguments)]
    fn marlin_gemm_u4b8_f16(
        &self,
        _a: &dyn DeviceBuffer, _b_repacked: &dyn DeviceBuffer, _c: &dyn DeviceBuffer, _c_tmp: &dyn DeviceBuffer,
        _b_scales: &dyn DeviceBuffer,
        _sorted_token_ids: &dyn DeviceBuffer, _expert_ids: &dyn DeviceBuffer, _num_tokens_past_padded: &dyn DeviceBuffer,
        _workspace: &dyn DeviceBuffer,
        _prob_m: usize, _prob_n: usize, _prob_k: usize, _num_groups: usize, _group_size: usize,
        _sms: i32, _use_fp32_reduce: bool,
        _thread_k: i32, _thread_n: i32, _blocks_per_sm: i32,
    ) -> Result<()> {
        Err(Error::Msg("marlin_gemm_u4b8_f16 unsupported on this backend".into()))
    }

    /// F-85 round-10 (lever 1): host-side-only resolution of
    /// `marlin_mm`'s auto thread-config search for a fixed decode-M=1
    /// shape, so [`marlin_gemm_u4b8_f16`] can be called with an explicit
    /// `(thread_k, thread_n, blocks_per_sm)` instead of `(-1,-1,-1)` and
    /// skip the per-call `cudaFuncGetAttributes` probe loop. Callers should
    /// resolve this ONCE per weight class at load time (every layer of a
    /// class shares the same `prob_n`/`prob_k`) and cache the result.
    /// Returns `(-1,-1,-1)` semantics preserved by backends that don't
    /// implement this (falls back to marlin_mm's own auto path).
    fn marlin_pick_config(
        &self, _prob_m: usize, _prob_n: usize, _prob_k: usize, _group_size: usize, _sms: i32,
    ) -> Result<(i32, i32, i32)> {
        Err(Error::Msg("marlin_pick_config unsupported on this backend".into()))
    }

    /// CUTLASS grouped block-scaled NVFP4 MoE GEMM: `out[t,n] = Σ_k a[t,k]·
    /// w[eid(t)][n,k]` over sorted-token groups, both operands packed e2m1 +
    /// ue4m3 block-16 scales in the canonical 512B-block swizzle. `a` packed
    /// `[mt,K/2]` bytes, `sfa` per-group scale pool (group g's blob at byte
    /// `sfa_off[g]`, group-LOCAL rows), `w`/`sfw` per-expert packed/scale
    /// slabs, `out` f16 `[mt,N]`. `group_rows`/`expert_ids`/`sfa_off` are host
    /// slices. Default errors; CUDA implements it via the AOT CUTLASS lib
    /// (sm_120a/121a builds with CUTLASS_DIR).
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_cutlass_fp4(
        &self,
        _a: &dyn DeviceBuffer,
        _sfa: &dyn DeviceBuffer,
        _w: &dyn DeviceBuffer,
        _sfw: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _group_rows: &[i32],
        _expert_ids: &[i32],
        _sfa_off: &[i64],
        _alpha_vec: Option<&dyn DeviceBuffer>, // device f32[n_groups] per-group scales
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(Error::Msg("moe_grouped_cutlass_fp4 unsupported on this backend".into()))
    }

    /// W4A8 grouped MoE GEMM: fp8 e4m3 acts (a + ue8m0 sfa) x mxfp4 weights
    /// (w + ue8m0 sfw), cutlass mxf8f6f4 mixed block-scaled MMA. Mirrors
    /// `moe_grouped_cutlass_fp4` but holds near-bf16 quality (256 act levels).
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_cutlass_w4a8(
        &self,
        _a: &dyn DeviceBuffer,
        _sfa: &dyn DeviceBuffer,
        _w: &dyn DeviceBuffer,
        _sfw: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _group_rows: &[i32],
        _expert_ids: &[i32],
        _sfa_off: &[i64],
        _sfb_exp_bytes: i64,
        _alpha_vec: Option<&dyn DeviceBuffer>,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(Error::Msg("moe_grouped_cutlass_w4a8 unsupported on this backend".into()))
    }

    /// W4A8 fp8 act-quant: x[mt,K] half -> out e4m3 + sf ue8m0 (per-32,
    /// group-aware SFA; group_rows/sfa_off are HOST arrays).
    fn w4a8_actquant(
        &self,
        _x: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _sf: &dyn DeviceBuffer,
        _group_rows: &[i32],
        _sfa_off: &[i64],
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(Error::Msg("w4a8_actquant unsupported on this backend".into()))
    }

    /// W4A8 mxfp4 weight pack: w[n_exp,N,K] half -> outp e2m1 + sf ue8m0
    /// (per-32, SFB). Returns per-expert SFB element count.
    fn w4a8_packw(
        &self,
        _w: &dyn DeviceBuffer,
        _outp: &dyn DeviceBuffer,
        _sf: &dyn DeviceBuffer,
        _n_exp: usize,
        _n: usize,
        _k: usize,
    ) -> Result<i64> {
        Err(Error::Msg("w4a8_packw unsupported on this backend".into()))
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_cutlass_w8a8(
        &self,
        _a: &dyn DeviceBuffer,
        _sfa: &dyn DeviceBuffer,
        _w: &dyn DeviceBuffer,
        _sfw: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _group_rows: &[i32],
        _expert_ids: &[i32],
        _sfa_off: &[i64],
        _sfb_exp_bytes: i64,
        _alpha_vec: Option<&dyn DeviceBuffer>,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(Error::Msg("moe_grouped_cutlass_w8a8 unsupported on this backend".into()))
    }

    /// W4A8 fp8 act-quant: x[mt,K] half -> out e4m3 + sf ue8m0 (per-32,
    /// group-aware SFA; group_rows/sfa_off are HOST arrays).
    fn w8a8_actquant(
        &self,
        _x: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _sf: &dyn DeviceBuffer,
        _group_rows: &[i32],
        _sfa_off: &[i64],
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(Error::Msg("w8a8_actquant unsupported on this backend".into()))
    }

    /// W4A8 mxfp4 weight pack: w[n_exp,N,K] half -> outp e2m1 + sf ue8m0
    /// (per-32, SFB). Returns per-expert SFB element count.
    fn w8a8_packw(
        &self,
        _w: &dyn DeviceBuffer,
        _outp: &dyn DeviceBuffer,
        _sf: &dyn DeviceBuffer,
        _n_exp: usize,
        _n: usize,
        _k: usize,
    ) -> Result<i64> {
        Err(Error::Msg("w8a8_packw unsupported on this backend".into()))
    }

    /// Prepare a persistent CUTLASS grouped-FP4 handle (device blob +
    /// workspace + initialized GEMM; descriptors filled per call ON DEVICE).
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_cutlass_fp4_prepare(
        &self,
        _w: &dyn DeviceBuffer,
        _sfw: &dyn DeviceBuffer,
        _alpha: &dyn DeviceBuffer,
        _n_groups: usize,
        _n: usize,
        _k: usize,
        _max_m_total: usize,
    ) -> Result<u64> {
        Err(Error::Msg("moe_grouped_cutlass_fp4_prepare unsupported on this backend".into()))
    }

    /// Run a prepared CUTLASS grouped-FP4 handle: one device fill kernel from
    /// the router offsets + gemm.run. No host work, no per-call allocs.
    fn moe_grouped_cutlass_fp4_run(
        &self,
        _handle: u64,
        _a: &dyn DeviceBuffer,
        _sfa: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _off_dev: &dyn DeviceBuffer,
    ) -> Result<()> {
        Err(Error::Msg("moe_grouped_cutlass_fp4_run unsupported on this backend".into()))
    }

    /// Fused-activation prepare: GEMM1 folds relu²·(1/256) + NVFP4 block-quant
    /// into its epilogue, emitting e2m1 D + ue4m3 SFD. `norm_constant` = device
    /// f32 ptr to the static gs (1/256). Gated behind NEMOTRON_FUSE_UPACT.
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_cutlass_fp4_FUSEDACT_prepare(
        &self,
        _w: &dyn DeviceBuffer,
        _sfw: &dyn DeviceBuffer,
        _alpha: &dyn DeviceBuffer,
        _norm_constant: &dyn DeviceBuffer,
        _n_groups: usize,
        _n: usize,
        _k: usize,
        _max_m_total: usize,
    ) -> Result<u64> {
        Err(Error::Msg("moe_grouped_cutlass_fp4_FUSEDACT_prepare unsupported on this backend".into()))
    }

    /// Run a fused-activation handle: device fill kernel + gemm.run, emitting
    /// `out` (e2m1) + `sfd` (ue4m3) for the down-GEMM input. No host work.
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_cutlass_fp4_FUSEDACT_run(
        &self,
        _handle: u64,
        _a: &dyn DeviceBuffer,
        _sfa: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _sfd: &dyn DeviceBuffer,
        _off_dev: &dyn DeviceBuffer,
    ) -> Result<()> {
        Err(Error::Msg("moe_grouped_cutlass_fp4_FUSEDACT_run unsupported on this backend".into()))
    }

    /// Amax-in-epilogue prepare: GEMM1 emits f16 D = relu²·(1/256) AND a
    /// per-group amax (`d_amax`, device f32[n_groups], persistent) of those
    /// activated values. The down-quant reads the epilogue amax instead of
    /// re-scanning. Gated behind NEMOTRON_AMAX_EPI.
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_cutlass_fp4_AMAX_prepare(
        &self,
        _w: &dyn DeviceBuffer,
        _sfw: &dyn DeviceBuffer,
        _d_amax: &dyn DeviceBuffer,
        _n_groups: usize,
        _n: usize,
        _k: usize,
        _max_m_total: usize,
    ) -> Result<u64> {
        Err(Error::Msg("moe_grouped_cutlass_fp4_AMAX_prepare unsupported on this backend".into()))
    }

    /// Run an amax-in-epilogue handle: device fill kernel + gemm.run, emitting
    /// `out` (f16 a2) and writing the per-group amax into the handle's d_amax
    /// (zeroed on-stream inside the run). No host work.
    fn moe_grouped_cutlass_fp4_AMAX_run(
        &self,
        _handle: u64,
        _a: &dyn DeviceBuffer,
        _sfa: &dyn DeviceBuffer,
        _out: &dyn DeviceBuffer,
        _off_dev: &dyn DeviceBuffer,
    ) -> Result<()> {
        Err(Error::Msg("moe_grouped_cutlass_fp4_AMAX_run unsupported on this backend".into()))
    }

    /// Byte-offset variant of []: same semantics but each buffer
    /// pointer is advanced by its byte offset before the GEMM call. Used to
    /// write expert GEMM outputs into pre-allocated slabs without extra copies.
    #[allow(clippy::too_many_arguments)]
    fn gemm_tc_off(
        &self,
        _x: &dyn DeviceBuffer, _x_off: usize,
        _w: &dyn DeviceBuffer, _w_off: usize,
        _out: &dyn DeviceBuffer, _out_off: usize,
        _m: usize, _n: usize, _k: usize, _dtype: DType,
    ) -> Result<()> {
        Err(Error::Msg("gemm_tc_off unsupported on this backend".into()))
    }

    /// Strided-batched tensor-core GEMM: `batch_count` independent GEMMs in ONE cuBLAS call.
    /// `C_i[m,n] = X_i[m,k] · W_i[n,k]^T`, each matrix at byte offsets `i*stride_{x,w,out}`.
    /// x/w/out are device buffers of `dtype` (f16/bf16); strides in BYTES.
    #[allow(clippy::too_many_arguments)]
    fn gemm_strided_batched(
        &self,
        _x: &dyn DeviceBuffer, _stride_x: i64,
        _w: &dyn DeviceBuffer, _stride_w: i64,
        _out: &dyn DeviceBuffer, _stride_out: i64,
        _m: usize, _n: usize, _k: usize,
        _batch_count: usize,
        _dtype: DType,
    ) -> Result<()> {
        Err(Error::Msg("gemm_strided_batched unsupported on this backend".into()))
    }

    /// Like [gemm_strided_batched] but with explicit byte offsets into x/w/out.
    /// Base pointers are advanced by the respective byte offset before the GEMM call.
    /// Enables sliced-batch calls (e.g. broadcast K for GQA fan-out).
    #[allow(clippy::too_many_arguments)]
    fn gemm_strided_batched_off(
        &self,
        _x: &dyn DeviceBuffer, _x_off: usize, _stride_x: i64,
        _w: &dyn DeviceBuffer, _w_off: usize, _stride_w: i64,
        _out: &dyn DeviceBuffer, _out_off: usize, _stride_out: i64,
        _m: usize, _n: usize, _k: usize,
        _batch_count: usize,
        _dtype: DType,
    ) -> Result<()> {
        Err(Error::Msg("gemm_strided_batched_off unsupported on this backend".into()))
    }

    /// Pointer-array batched GEMM: `batch_count` independent GEMMs sharing
    /// `(m,n,k)`, reading X/W/Out from per-batch byte offsets into the three
    /// backing buffers. `C_i[m,n] = X_i[m,k] · W_i[n,k]^T`. Unlike
    /// [gemm_strided_batched] the per-batch offsets are arbitrary, so one
    /// operand can BROADCAST (multiple batches reuse the same slice) — used by
    /// the SSD scan to fan a per-group B/C slice into every head's batch slot
    /// without an 8× redundant gather. All operands f16/bf16; accumulate f32.
    #[allow(clippy::too_many_arguments)]
    fn gemm_batched(
        &self,
        _x_buf: &dyn DeviceBuffer,
        _x_offsets: &[usize],   // byte offset per batch into x_buf
        _w_buf: &dyn DeviceBuffer,
        _w_offsets: &[usize],   // byte offset per batch into w_buf
        _out_buf: &dyn DeviceBuffer,
        _out_offsets: &[usize], // byte offset per batch into out_buf
        _m: usize, _n: usize, _k: usize,
        _dtype: DType,
    ) -> Result<()> {
        Err(Error::Msg("gemm_batched unsupported on this backend".into()))
    }

    /// Grouped-batched GEMM (CUDA 13+ only): `group_count` independent GEMMs
    /// where group `i` computes `C_i[m_i, n] = X_i[m_i, k] · W_i[n, k]^T`.
    /// All inputs/outputs are f16; accumulate f32.
    ///
    /// `x_ptrs[i]`, `w_ptrs[i]`, `out_ptrs[i]` are raw device-pointer offsets
    /// into the backing allocation (bytes from the start of `x_buf` / `w_buf` /
    /// `out_buf` respectively). This avoids sub-allocating separate `DeviceBuffer`
    /// handles for every expert — the caller passes one large contiguous buffer
    /// for X and one for W, with per-expert byte offsets.
    #[allow(clippy::too_many_arguments)]
    fn gemm_grouped(
        &self,
        _x_buf: &dyn DeviceBuffer,
        _x_offsets: &[usize],   // byte offset per group into x_buf
        _w_buf: &dyn DeviceBuffer,
        _w_offsets: &[usize],   // byte offset per group into w_buf
        _out_buf: &dyn DeviceBuffer,
        _out_offsets: &[usize], // byte offset per group into out_buf
        _m_per_group: &[i32],
        _n: usize,
        _k: usize,
        _dtype: DType,
    ) -> Result<()> {
        Err(Error::Msg("gemm_grouped unsupported on this backend".into()))
    }
}

/// A handle to a region of device memory + shape + dtype. Backend-neutral:
/// the buffer is an `Arc<dyn DeviceBuffer>`, so one `Tensor` type flows
/// through every backend's code path unchanged.
#[derive(Clone)]
pub struct Tensor {
    pub buffer: Arc<dyn DeviceBuffer>,
    /// Byte offset into `buffer` where this tensor begins (slices share the
    /// parent allocation).
    pub offset: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    pub fn new(buffer: Arc<dyn DeviceBuffer>, shape: Vec<usize>, dtype: DType) -> Self {
        Tensor { buffer, offset: 0, shape, dtype }
    }

    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_count(&self) -> usize {
        self.elem_count() * self.dtype.size_bytes()
    }

    /// Allocate a fresh contiguous tensor on `dev`.
    pub fn empty(dev: &dyn Device, shape: Vec<usize>, dtype: DType) -> Result<Self> {
        let bytes = shape.iter().product::<usize>() * dtype.size_bytes();
        Ok(Tensor::new(dev.alloc(bytes)?, shape, dtype))
    }

    /// Reshape without copying. Element count must be preserved.
    pub fn reshaped(&self, new_shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            new_shape.iter().product::<usize>(),
            self.elem_count(),
            "reshape changes element count"
        );
        Tensor {
            buffer: self.buffer.clone(),
            offset: self.offset,
            shape: new_shape,
            dtype: self.dtype,
        }
    }
}
