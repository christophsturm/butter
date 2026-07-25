// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # wh-butter-ops
//!
//! The **seam** between model code and kernels — the Rust analog of
//! Butter-Swift's `Ops/`. Each function takes Tensors, builds (or looks up)
//! the corresponding iron [`Kernel`](wh_butter_core::Kernel), and dispatches
//! it through the [`Device`](wh_butter_core::Device) trait. Model code calls
//! these; it never touches a GPU API or a kernel directly. Re-implementing
//! this layer once is what lets the whole model surface above it run on
//! every backend.
//!
//! Every op here is real and runs on any backend implementing [`Device`]
//! (verified vs HF on CUDA + Metal). Elementwise ops hand-build their IR;
//! the heavier ones (matmul / rms_norm / layer_norm / sdpa / rope / conv1d /
//! ssm …) resolve a registered iron kernel via `lookup`, which carries
//! the correct dispatch mode.

use wh_butter_core::{Binding, DType, Device, Error, Grid, Kernel, Result, Tensor};
use wh_iron_core::ir::{ActKind, BinOpKind, IndexExpr, Op, Param, ParamKind, ValueId};
use wh_iron_core::shape::Shape;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::sync::OnceLock;

/// Device-resident scalar-constant cache: tiny buffers (norm `eps` values)
/// uploaded once per (device, bit-pattern) instead of per call. The norms run
/// hundreds of times per prefill forward, and each 4-byte upload pays a full
/// pageable H2D + driver round-trip.
fn scalar_buf(dev: &dyn Device, bits: u32) -> Result<std::sync::Arc<dyn wh_butter_core::DeviceBuffer>> {
    use std::cell::RefCell;
    thread_local! {
        static CACHE: RefCell<FxHashMap<(usize, u32), std::sync::Arc<dyn wh_butter_core::DeviceBuffer>>> =
            RefCell::new(FxHashMap::default());
    }
    let key = (dev as *const dyn Device as *const () as usize, bits);
    if let Some(b) = CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return Ok(b);
    }
    let b = dev.upload(&bits.to_le_bytes())?;
    CACHE.with(|c| c.borrow_mut().insert(key, b.clone()));
    Ok(b)
}

mod ssd_scan;
pub use ssd_scan::ssm_prefill_scan_ssd;

mod ssd_scan_portable;
pub use ssd_scan_portable::ssm_prefill_scan_ssd_portable;

/// Cache of resolved kernel IR keyed by (name, dtype). Resolving walks the
/// whole test registry building setups, which is expensive: a forward pass
/// dispatches the same handful of kernels hundreds of times, so cache them.
/// FxHashMap (single-cycle hash) + parking_lot Mutex (unpoisoned, inlinable).
fn kernel_cache() -> &'static Mutex<FxHashMap<(String, DType), Kernel>> {
    static C: OnceLock<Mutex<FxHashMap<(String, DType), Kernel>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(FxHashMap::default()))
}

/// Cache an inline-built kernel IR by (name, dtype). `kernel_ir_for` rebuilds the
/// full IR (~30+ ops) on every call, on the decode hot path that's ~250
/// rebuilds/token. Build once, clone thereafter.
fn cached_ir(name: &str, dtype: DType, build: impl FnOnce() -> Kernel) -> Kernel {
    let key = (name.to_string(), dtype);
    if let Some(k) = kernel_cache().lock().get(&key) {
        return k.clone();
    }
    let k = build();
    kernel_cache().lock().insert(key, k.clone());
    k
}

/// Look up a registered iron kernel by name and instantiate its IR for
/// `dtype`, **with the correct dispatch mode already set**. We pull it from
/// the test registry (`wh_iron_std::all_tests`) rather than the raw kernel
/// registry: the test/bench setup is where each kernel's `KernelMode`
/// (Elementwise / Reduction / Grid3D …) is configured, and that mode drives
/// codegen (e.g. whether `tid` vs `gid` is defined, the `_n_elems` arg). This
/// is the same path the CUDA corpus dispatches, so we inherit its correctness.
fn lookup(name: &str, dtype: DType) -> Result<Kernel> {
    let key = (name.to_string(), dtype);
    if let Some(k) = kernel_cache().lock().get(&key) {
        return Ok(k.clone());
    }
    for entry in wh_iron_std::all_tests() {
        let t = entry.test();
        if !t.dtypes().contains(&dtype) {
            continue;
        }
        let setup = t.setup(dtype);
        let k = setup.kernel();
        if k.name == name {
            kernel_cache().lock().insert(key, k.clone());
            return Ok(k.clone());
        }
    }
    Err(Error::Msg(format!(
        "kernel '{name}' [{dtype}] not found in the iron test registry"
    )))
}

/// Build an Elementwise `out[i] = a[i] <op> b[i]` kernel for `dtype`.
fn binop_kernel(name: &str, dtype: DType, op: BinOpKind) -> Kernel {
    let mut k = Kernel::new(name);
    for (pname, is_out) in [("a", false), ("b", false), ("c", true)] {
        k.params.push(Param {
            name: pname.into(),
            dtype,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(
        Op::Load {
            src: "a".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            mask: None,
            other: None,
        },
        ValueId::new(1),
    );
    k.body.push_op(
        Op::Load {
            src: "b".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            mask: None,
            other: None,
        },
        ValueId::new(2),
    );
    k.body.push_op(
        Op::BinOp { op, lhs: ValueId::new(1), rhs: ValueId::new(2) },
        ValueId::new(3),
    );
    k.body.push_op_no_result(Op::Store {
        dst: "c".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(3),
        mask: None,
    });
    k
}

/// Shared implementation for elementwise binary ops over matching-shape
/// tensors. Allocates a fresh output on `dev` and dispatches through the
/// backend-neutral [`Device`] trait.
fn elementwise(
    dev: &dyn Device,
    a: &Tensor,
    b: &Tensor,
    op: BinOpKind,
    name: &str,
) -> Result<Tensor> {
    if a.shape != b.shape {
        return Err(Error::Msg(format!(
            "{name}: shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    if a.dtype != b.dtype {
        return Err(Error::Msg(format!("{name}: dtype mismatch")));
    }
    let out = Tensor::empty(dev, a.shape.clone(), a.dtype)?;
    let k = binop_kernel(name, a.dtype, op);
    let n = a.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(a.buffer.clone()),
            Binding::Buffer(b.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Elementwise sum `a + b` (e.g. residual connections).
const ADD16_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
extern "C" __global__ void butter_add_f16(const __half* __restrict__ a, const __half* __restrict__ b, __half* __restrict__ out, long n){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; long stride=(long)gridDim.x*blockDim.x;
    for(;i<n;i+=stride) out[i]=__hadd(a[i],b[i]);
}
extern "C" __global__ void butter_add_bf16(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, long n){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; long stride=(long)gridDim.x*blockDim.x;
    for(;i<n;i+=stride) out[i]=__float2bfloat16(__bfloat162float(a[i])+__bfloat162float(b[i]));
}
"#;

const MULSCALAR16_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void butter_mul_scalar_f16(const __half* __restrict__ a, __half* __restrict__ out, float sc, long n){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; long stride=(long)gridDim.x*blockDim.x;
    for(;i<n;i+=stride) out[i]=__float2half(__half2float(a[i])*sc);
}
"#;

/// Elementwise `out = a * scale` for an f16 tensor (raw CUDA). Used to undo a
/// /256 pre-scale that keeps an f16 intermediate inside the f16 range.
pub fn mul_scalar_f16(dev: &dyn Device, a: &Tensor, scale: f32) -> Result<Tensor> {
    let out = Tensor::empty(dev, a.shape.clone(), DType::F16)?;
    let n = a.elem_count();
    dev.dispatch_raw_cuda(MULSCALAR16_SRC, "mul_scalar_f16.cu", "butter_mul_scalar_f16",
        &[(a.buffer.as_ref(), a.offset), (out.buffer.as_ref(), 0)],
        &[scale.to_le_bytes().to_vec(), (n as i64).to_le_bytes().to_vec()],
        [(n as u32).div_ceil(256).max(1), 1, 1], [256, 1, 1], 0, false)?;
    Ok(out)
}

pub fn add(dev: &dyn Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // 16-bit DSL elementwise vec4 codegen IMAs; route f16/bf16 through a raw kernel.
    if a.dtype == DType::F16 || a.dtype == DType::BF16 {
        let out = Tensor::empty(dev, a.shape.clone(), a.dtype)?;
        let n = a.elem_count();
        let fname = if a.dtype == DType::F16 { "butter_add_f16" } else { "butter_add_bf16" };
        dev.dispatch_raw_cuda(ADD16_SRC, "add16.cu", fname,
            &[(a.buffer.as_ref(), a.offset), (b.buffer.as_ref(), b.offset), (out.buffer.as_ref(), 0)],
            &[(n as i64).to_le_bytes().to_vec()],
            [(n as u32).div_ceil(256).max(1), 1, 1], [256, 1, 1], 0, false)?;
        return Ok(out);
    }
    elementwise(dev, a, b, BinOpKind::Add, "iron_add")
}

/// Elementwise product `a * b` (e.g. gating).
pub fn mul(dev: &dyn Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    elementwise(dev, a, b, BinOpKind::Mul, "iron_mul")
}

const ADD3_SRC: &str = r#"
extern "C" __global__ void butter_add3(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, float* __restrict__ out, long n)
{
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    long stride=(long)gridDim.x*blockDim.x;
    // association matches the unfused path exactly: xt + (acc + sd), a
    // left-assoc a+b+c flipped the d32768 argmax via last-ulp drift.
    for(;i<n;i+=stride) out[i]=a[i]+(b[i]+c[i]);
}
#include <cuda_fp16.h>
#include <cuda_bf16.h>
extern "C" __global__ void butter_add3_f16(const __half* __restrict__ a, const __half* __restrict__ b, const __half* __restrict__ c, __half* __restrict__ out, long n){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; long stride=(long)gridDim.x*blockDim.x;
    for(;i<n;i+=stride){ float r=__half2float(a[i])+(__half2float(b[i])+__half2float(c[i])); out[i]=__float2half(r);} }
extern "C" __global__ void butter_add3_bf16(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, const __nv_bfloat16* __restrict__ c, __nv_bfloat16* __restrict__ out, long n){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; long stride=(long)gridDim.x*blockDim.x;
    for(;i<n;i+=stride){ float r=__bfloat162float(a[i])+(__bfloat162float(b[i])+__bfloat162float(c[i])); out[i]=__float2bfloat16(r);} }
"#;

/// Fused three-way elementwise sum `a + b + c` (f32, CUDA raw path), one
/// launch + no intermediate tensor for the per-E-layer
/// `residual + routed_acc + shared_out` merge (was two `iron_add` launches).
pub fn add3(dev: &dyn Device, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    let out = Tensor::empty(dev, a.shape.clone(), a.dtype)?;
    let n = a.elem_count();
    let l = |v: i64| v.to_le_bytes().to_vec();
    let fname = match a.dtype { DType::F16 => "butter_add3_f16", DType::BF16 => "butter_add3_bf16", _ => "butter_add3" };
    let per = if a.dtype == DType::F32 { 256 * 4 } else { 256 };
    dev.dispatch_raw_cuda(ADD3_SRC, "add3.cu", fname,
        &[(a.buffer.as_ref(), a.offset), (b.buffer.as_ref(), b.offset),
          (c.buffer.as_ref(), c.offset), (out.buffer.as_ref(), 0)],
        &[l(n as i64)],
        [(n as u32).div_ceil(per).max(1), 1, 1], [256, 1, 1], 0, false)?;
    Ok(out)
}

// ── DeepSeek-V4 MoE ops ─────────────────────────────────────────────────

/// DSv4 clamped SwiGLU: `out = silu(min(gate, limit)) * clip(up, ±limit)`.
/// Dispatches `iron_dsv4_swiglu_limit` (limit = 10 for DSv4).
pub fn swiglu_limit(dev: &dyn Device, gate: &Tensor, up: &Tensor, limit: f32) -> Result<Tensor> {
    if gate.shape != up.shape {
        return Err(Error::Msg("swiglu_limit: gate/up shape mismatch".into()));
    }
    let k = lookup("iron_swiglu_limit", gate.dtype)?;
    let out = Tensor::empty(dev, gate.shape.clone(), gate.dtype)?;
    let n = gate.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(gate.buffer.clone()),
            Binding::Buffer(up.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar(limit.to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// DSv4 MoE router scoring: `score_unbiased[e] = sqrt(softplus(logit[e]))`,
/// `score_biased[e] = score_unbiased[e] + bias[e]`. Computed **host-side**,
/// the router operates on the tiny `[n_experts]` logit vector that the MoE
/// already downloads for top-k selection, so a GPU kernel buys nothing and
/// avoids the multi-output dispatch path. `biased` selects the top-k experts;
/// `unbiased` weights the combine.
pub fn sqrtsoftplus_route(logits: &[f32], bias: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let unbiased: Vec<f32> = logits
        .iter()
        .map(|&x| (x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()).sqrt())
        .collect();
    let biased: Vec<f32> = unbiased.iter().zip(bias).map(|(u, b)| u + b).collect();
    (unbiased, biased)
}

// ── Causal conv1d step (Mamba2 short-conv / audio front-ends) ───────────

/// Causal depthwise conv1d decode step (one thread per channel):
/// `y[d] = b[d] + w[K-1,d]·x[d] + Σ_{k<K-1} w[k,d]·state[k,d]`, then shift the
/// ring `state` and insert `x`. `w` is `[kernel_size·n_channels]`, `state` is
/// `[(kernel_size-1)·n_channels]` (updated in place). Dispatches
/// `conv1d_causal_step`. Returns `y [n_channels]`; `state` buffer is mutated.
pub fn conv1d_causal_step(
    dev: &dyn Device,
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    state: &Tensor,
    n_channels: u32,
    kernel_size: u32,
) -> Result<Tensor> {
    let k = lookup("iron_conv1d_causal_step", x.dtype)?;
    let y = Tensor::empty(dev, vec![n_channels as usize], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let bindings = vec![
        Binding::Buffer(x.buffer.clone()),
        Binding::Buffer(w.buffer.clone()),
        Binding::Buffer(b.buffer.clone()),
        Binding::Buffer(state.buffer.clone()), // in-place ring state
        Binding::Buffer(y.buffer.clone()),
        u(n_channels),
        u(kernel_size),
    ];
    // TIER-1 launch-geometry fix: move the n_channels extent from grid.x into
    // block.x (product unchanged → byte-identical program_id under Grid3D
    // codegen, where gid_x = blockIdx.x*blockDim.x + threadIdx.x). b is the
    // largest power-friendly block that divides n_channels (gcd capped at 256).
    let b = gcd_u32(n_channels, 256).max(1);
    let grid = Grid { grid: [n_channels / b, 1, 1], block: [b, 1, 1] };
    dev.dispatch(&k, &bindings, grid)?;
    Ok(y)
}

/// gcd for launch-geometry block sizing (Tier-1 grid→block extent moves).
#[inline]
fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ── SSM (Mamba2 SSD selective scan) ─────────────────────────────────────

/// Mamba2 SSD selective-scan **decode step** (single token, batch=1):
/// `da = exp(-exp(a_log)·dt)`; `state' = da·state + x·dt·B`;
/// `out = Σ_s C·state' + x·D`. Dispatches `iron_ssm_step`. Returns
/// `(state_out [n_heads·dh·ds], out [n_heads·dh])`. `ds` must be a multiple
/// of 32. Shapes: x `[n_heads·dh]`, a_log/d_skip/dt `[n_heads]`,
/// b_mat/c_mat `[n_groups·ds]`, state_in `[n_heads·dh·ds]`.
#[allow(clippy::too_many_arguments)]
pub fn ssm_step(
    dev: &dyn Device,
    x: &Tensor,
    a_log: &Tensor,
    b_mat: &Tensor,
    c_mat: &Tensor,
    d_skip: &Tensor,
    dt: &Tensor,
    state_in: &Tensor,
    dh: u32,
    ds: u32,
    n_heads: u32,
    heads_per_group: u32,
) -> Result<(Tensor, Tensor)> {
    let k = lookup("iron_ssm_step_grouped", x.dtype)?;
    let state_out = Tensor::empty(dev, state_in.shape.clone(), x.dtype)?;
    let out = Tensor::empty(dev, vec![(n_heads * dh) as usize], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let bindings = vec![
        Binding::Buffer(x.buffer.clone()),
        Binding::Buffer(a_log.buffer.clone()),
        Binding::Buffer(b_mat.buffer.clone()),
        Binding::Buffer(c_mat.buffer.clone()),
        Binding::Buffer(d_skip.buffer.clone()),
        Binding::Buffer(dt.buffer.clone()),
        Binding::Buffer(state_in.buffer.clone()),
        Binding::Buffer(state_out.buffer.clone()),
        Binding::Buffer(out.buffer.clone()),
        u(dh),
        u(ds),
        u(n_heads),
        u(heads_per_group),
    ];
    let grid = Grid { grid: [dh, n_heads, 1], block: [32, 1, 1] };
    dev.dispatch(&k, &bindings, grid)?;
    Ok((state_out, out))
}

// ── DeepSeek-V4 mHC (hyper-connection 4-channel residual) ───────────────

/// mHC sinkhorn split (single token, host-side): the 24-value `mixes`
/// vector → `pre[4]`, `post[4]`, `comb[4×4]`. Faithful transcription of
/// `iron_dsv4_mhc_sinkhorn_split` (a 3-output kernel; trivial at one token,
/// so it runs on the host). `scale` is `[pre, post, comb]`, `base` is `[24]`.
/// - pre[c]  = sigmoid(mixes[c]·pre_scale + base[c]) + eps
/// - post[c] = 2·sigmoid(mixes[4+c]·post_scale + base[4+c])
/// - comb    = per-row softmax of (mixes·comb_scale + base), then
///   `iters` Sinkhorn steps (column-normalize, then row-normalize).
pub fn dsv4_mhc_sinkhorn_split(
    mixes: &[f32],
    scale: [f32; 3],
    base: &[f32],
    eps: f32,
    iters: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let pre: Vec<f32> = (0..4).map(|c| sig(mixes[c] * scale[0] + base[c]) + eps).collect();
    let post: Vec<f32> = (0..4).map(|c| 2.0 * sig(mixes[4 + c] * scale[1] + base[4 + c])).collect();

    let mut c = [[0.0f32; 4]; 4];
    for i in 0..4 {
        let r: [f32; 4] = std::array::from_fn(|j| mixes[8 + i * 4 + j] * scale[2] + base[8 + i * 4 + j]);
        let m = r.iter().cloned().fold(f32::MIN, f32::max);
        let e: [f32; 4] = std::array::from_fn(|j| (r[j] - m).exp());
        let s: f32 = e.iter().sum();
        for j in 0..4 {
            c[i][j] = e[j] / s + eps;
        }
    }
    for _ in 0..iters {
        for j in 0..4 {
            let cs: f32 = (0..4).map(|i| c[i][j]).sum::<f32>() + eps;
            for row in c.iter_mut() {
                row[j] /= cs;
            }
        }
        for row in c.iter_mut() {
            let rs: f32 = row.iter().sum::<f32>() + eps;
            for v in row.iter_mut() {
                *v /= rs;
            }
        }
    }
    let comb: Vec<f32> = (0..4).flat_map(|i| (0..4).map(move |j| c[i][j])).collect();
    (pre, post, comb)
}


/// mHC collapse: `out[d] = Σ_c pre[c] · state[c, d]`, mix the 4-channel
/// residual state `[n_hc, hidden]` down to `[hidden]` using the per-channel
/// `pre` weights. Single token. Dispatches `iron_dsv4_mhc_collapse`.
pub fn dsv4_mhc_collapse(
    dev: &dyn Device,
    state: &Tensor,
    pre: &Tensor,
    hidden_dim: u32,
    n_hc: u32,
) -> Result<Tensor> {
    let k = lookup("iron_mhc_collapse", state.dtype)?;
    let out = Tensor::empty(dev, vec![hidden_dim as usize], state.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid::d1((hidden_dim).div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(state.buffer.clone()),
            Binding::Buffer(pre.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(hidden_dim),
            u(n_hc),
            u(1),
        ],
        grid,
    )?;
    Ok(out)
}

/// mHC expand: write the new 4-channel residual state
/// `state[dst, d] = block_out[d]·post[dst] + Σ_src comb[dst, src]·residual[src, d]`.
/// Single token. Returns `[n_hc, hidden]`. Dispatches `iron_dsv4_mhc_expand`.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_mhc_expand(
    dev: &dyn Device,
    block_out: &Tensor,
    post: &Tensor,
    comb: &Tensor,
    residual_state: &Tensor,
    hidden_dim: u32,
    n_hc: u32,
) -> Result<Tensor> {
    let k = lookup("iron_mhc_expand", block_out.dtype)?;
    let state = Tensor::empty(dev, vec![(n_hc * hidden_dim) as usize], block_out.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid::d1((hidden_dim).div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(block_out.buffer.clone()),
            Binding::Buffer(post.buffer.clone()),
            Binding::Buffer(comb.buffer.clone()),
            Binding::Buffer(residual_state.buffer.clone()),
            Binding::Buffer(state.buffer.clone()),
            u(hidden_dim),
            u(n_hc),
            u(1),
        ],
        grid,
    )?;
    Ok(state)
}

// ── Heavier ops — dispatch the registered iron kernels ─────────────

/// Row-wise RMS norm: `out[r] = x[r] * rsqrt(mean(x[r]²) + eps) * weight`.
/// Dispatches the registered `iron_rms_norm` reduction kernel — the same one
/// the Swift side runs. The last dim is the row width `n`; the kernel owns 4
/// elements per thread, so `n` must be a multiple of 128 and ≤ 4096 (the
/// `iron_rms_norm_wide` variant lifts this — wired later).
pub fn rms_norm(dev: &dyn Device, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let n = *x.shape.last().ok_or_else(|| Error::Msg("rms_norm: scalar input".into()))?;
    let rows = x.elem_count() / n;
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let eps_buf = scalar_buf(dev, eps.to_bits())?;

    // Fast path: 4 elems/thread, needs n a multiple of 128 and ≤ 4096.
    // Otherwise the strided wide variant handles any row width.
    // 16-bit (f16/bf16) DSL vec4 fast-path codegen is broken (IMAs); route them
    // through the scalar wide variant. f32 keeps the fast path.
    let (kname, block) = if x.dtype == DType::F32 && n % 128 == 0 && n <= 4096 {
        ("iron_rms_norm", (n / 4) as u32)
    } else {
        ("iron_rms_norm_wide", 256u32)
    };
    let k = lookup(kname, x.dtype)?;
    let grid = Grid { grid: [rows as u32, 1, 1], block: [block, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(weight.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Buffer(eps_buf),
            Binding::Scalar((n as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Matrix-vector product `mat @ vec`: `mat` is `[m, k]` row-major, `vec` is
/// `[k]`, result is `[m]`. Dispatches the registered `iron_gemv` kernel (one
/// threadgroup per output row). This is the decode-time projection path; the
/// batched/prefill cooperative matmul is a separate kernel, wired later.
pub fn gemv(dev: &dyn Device, mat: &Tensor, vec: &Tensor) -> Result<Tensor> {
    if mat.shape.len() != 2 {
        return Err(Error::Msg(format!("gemv: mat must be 2-D, got {:?}", mat.shape)));
    }
    let (m, kdim) = (mat.shape[0], mat.shape[1]);
    if vec.elem_count() != kdim {
        return Err(Error::Msg(format!(
            "gemv: vec len {} != mat K {kdim}",
            vec.elem_count()
        )));
    }
    let k = lookup("iron_gemv", mat.dtype)?;
    let out = Tensor::empty(dev, vec![m], mat.dtype)?;

    let grid = Grid { grid: [m as u32, 1, 1], block: [256, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(mat.buffer.clone()),
            Binding::Buffer(vec.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((kdim as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Q8_0 grouped matvec, weights stay QUANTIZED resident (int8, block 32, one
/// f32 scale/block): `out[r] = Σ_k dequant(qs[r,k], scales[r, k/32]) ·
/// x[(r/rows_per_group)·k_in + k]`. `qs` is u32-packed (4 int8/u32, 8 u32/block),
/// `scales` is `[m_out · k_in/32]` f32. Dense gemv ⇒ `rows_per_group = m_out`
/// (n_groups=1, `x = [k_in]`); MoE ⇒ per-expert grouping. Dispatches the
/// Reduction kernel `iron_grouped_gemv_q8_rows` (8-bit ⇒ ~4× less weight DRAM
/// than F32, the resident-decode bandwidth win). `k_in` must be a multiple of 32.
pub fn gemv_q8(
    dev: &dyn Device,
    qs: &Tensor,
    scales: &Tensor,
    x: &Tensor,
    m_out: usize,
    k_in: usize,
    rows_per_group: usize,
) -> Result<Tensor> {
    if k_in % 32 != 0 {
        return Err(Error::Msg(format!("gemv_q8: k_in {k_in} must be a multiple of 32")));
    }
    // Not in the test registry, so build the kernel IR directly (constexpr dims
    // are JIT-specialized by the runtime from the scalar bindings below).
    // Coalesced variant: consecutive lanes read consecutive qs words (~2× the
    // strided original's DRAM bandwidth on GB10).
    let k = cached_ir("iron_gemv_q8_coalesced", x.dtype, || { let mut k = wh_iron_std::kernels::gemm::gemv_quantized::iron_gemv_q8_coalesced::kernel_ir_for(x.dtype); k.mode = wh_iron_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![m_out], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid { grid: [m_out as u32, 1, 1], block: [32, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(k_in as u32),
            u(m_out as u32),
            u(rows_per_group as u32),
        ],
        grid,
    )?;
    Ok(out)
}

/// Batched Q8 GEMM — the **prefill** projection path. `out[r,o] = Σ_k W[o,k]·x[r,k]`
/// for `n_rows` tokens at once, dispatching iron's `iron_gemm_q8_mpp`
/// (SimdGroup-scope CoopTile 64×64×32 tile). On the Vulkan/RDNA4 backend this
/// kernel picks up the gated `VK_KHR_cooperative_matrix` fragment ops when
/// `BUTTER_VK_COOPMAT=1` (≈6× the scalar GEMM) — the realized prefill win. With the
/// gate unset it runs the bit-exact scalar `SoftwareLocalC` tiles. Either way it
/// replaces the `n_rows`× sequential `gemv_q8` matvecs of the per-token path.
///
/// The coopmat staging tiles are fp16, so the activation/output are carried as
/// f16 across the kernel: `x` (f32 `[n_rows, k_in]`) is cast to f16 on entry and
/// the f16 result is widened back to f32 `[n_rows, m_out]` so the rest of the
/// f32 forward graph is unchanged. Weight layout (`qs`/`scales`) is identical to
/// [`gemv_q8`] (dense, `rows_per_group = m_out`). `k_in` must be a multiple of 32.
pub fn gemm_q8_mpp(
    dev: &dyn Device,
    qs: &Tensor,
    scales: &Tensor,
    x: &Tensor,
    n_rows: usize,
    m_out: usize,
    k_in: usize,
) -> Result<Tensor> {
    if k_in % 32 != 0 {
        return Err(Error::Msg(format!("gemm_q8_mpp: k_in {k_in} must be a multiple of 32")));
    }
    // Coopmat fragments stage through fp16 → drive the kernel at f16. Cast the
    // f32 residual-stream activations in, widen the result back out.
    let xf16 = if x.dtype == DType::F16 { x.clone() } else { cast_f32_f16(dev, x)? };
    // `iron_gemm_q8_mpp` in Reduction mode (matches the iron coopmat test
    // + bench). Buffer order: x, qs, d_f32, out; constexpr n_rows, out_dim, k_in.
    let k = cached_ir("iron_gemm_q8_mpp", DType::F16, || {
        let mut k = wh_iron_std::kernels::gemm::gemm_q8_mpp::iron_gemm_q8_mpp::kernel_ir_for(DType::F16);
        k.mode = wh_iron_core::ir::KernelMode::Reduction;
        k
    });
    let out = Tensor::empty(dev, vec![n_rows * m_out], DType::F16)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    // grid (threadgroups) = [ceil(out_dim/64), ceil(n_rows/64), 1], tg [128,1,1].
    let grid = Grid {
        grid: [(m_out as u32).div_ceil(64), (n_rows as u32).div_ceil(64), 1],
        block: [128, 1, 1],
    };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(xf16.buffer.clone()),
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(n_rows as u32),
            u(m_out as u32),
            u(k_in as u32),
        ],
        grid,
    )?;
    let out_f32 = cast_f16_f32(dev, &out)?;
    Ok(out_f32.reshaped(vec![n_rows, m_out]))
}

/// Broadcast-add a per-feature bias `[n]` across every row of `x [n_rows, n]`:
/// `out[r,o] = x[r,o] + bias[o]`. The decode path adds a `[n]` bias to a single
/// `[n]` projection with plain [`add`]; the batched prefill path needs the same
/// bias on every token row, but [`add`] is shape-strict (no broadcast). This is
/// a fully on-device elementwise kernel (`out[i] = x[i] + bias[i % n]`), no
/// host round-trip, so it works on device-local (non-mappable) weight buffers.
/// Returns `[n_rows, n]` f32.
pub fn add_bias_rows(dev: &dyn Device, x: &Tensor, bias: &Tensor, n_rows: usize, n: usize) -> Result<Tensor> {
    // Fully on-device per-feature bias broadcast: `i = ProgramId(0)` (global
    // flat id), row-local column `col = i % n`, `out[i] = x[i] + bias[col]`.
    // `n` is a CONSTEXPR push-constant (bound as a `Binding::Scalar`) — NOT a
    // baked `Op::Const` literal — so the one cached `iron_add_bias_rows`
    // pipeline serves every projection width (q: nq*hd, k/v: nkv*hd) instead
    // of silently reusing the first call's `n`. The DSL `%` lowers to
    // `BinOpKind::Mod` → GLSL `uint(i) % uint(n)` (integer modulo), validated
    // bit-exact on Vulkan by `wh-iron-std/tests/vulkan_add_bias_rows.rs`.
    // No host round-trip — runs straight on the device-local weight buffer.
    use wh_iron_core::ir::ConstExprDecl;
    use wh_iron_core::ConstExpr;
    let total = n_rows * n;
    let out = Tensor::empty(dev, vec![total], DType::F32)?;
    let mut k = Kernel::new("iron_add_bias_rows");
    for (pname, is_out) in [("x", false), ("bias", false), ("out", true)] {
        k.params.push(Param {
            name: pname.into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.constexprs.push(ConstExprDecl {
        name: ConstExpr::new("n"),
        dtype: DType::U32,
        value: None,
    });
    // SSA: 0=i, 1=n, 2=col, 3=x[i], 4=bias[col], 5=sum.
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(
        Op::Load { src: "n".into(), indices: vec![], mask: None, other: None },
        ValueId::new(1),
    );
    k.body.push_op(
        Op::BinOp { op: BinOpKind::Mod, lhs: ValueId::new(0), rhs: ValueId::new(1) },
        ValueId::new(2),
    );
    k.body.push_op(
        Op::Load {
            src: "x".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            mask: None,
            other: None,
        },
        ValueId::new(3),
    );
    k.body.push_op(
        Op::Load {
            src: "bias".into(),
            indices: vec![IndexExpr::Value(ValueId::new(2))],
            mask: None,
            other: None,
        },
        ValueId::new(4),
    );
    k.body.push_op(
        Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(3), rhs: ValueId::new(4) },
        ValueId::new(5),
    );
    k.body.push_op_no_result(Op::Store {
        dst: "out".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(5),
        mask: None,
    });
    let grid = Grid::d1((total as u32).div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(bias.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((n as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out.reshaped(vec![n_rows, n]))
}

/// Q8 gemv with fused ReLU²: `out[r] = max(0, (Wq·x)[r])²` — a MoE expert's
/// `up` projection + activation in one dispatch. Dispatches `iron_gemv_q8_coalesced_relu2`.
pub fn gemv_q8_relu2(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, m_out: usize, k_in: usize, rows_per_group: usize) -> Result<Tensor> {
    let k = cached_ir("iron_gemv_q8_coalesced_relu2", x.dtype, || { let mut k = wh_iron_std::kernels::gemm::gemv_quantized::iron_gemv_q8_coalesced_relu2::kernel_ir_for(x.dtype); k.mode = wh_iron_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![m_out], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid { grid: [m_out as u32, 1, 1], block: [32, 1, 1] };
    dev.dispatch(&k, &[Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(k_in as u32), u(m_out as u32), u(rows_per_group as u32)], grid)?;
    Ok(out)
}

/// Q8 gemv that scales + accumulates in place: `acc[r] += scale · (Wq · x)[r]`.
/// Fuses an MoE expert's `down` projection with its router-weighted sum into the
/// layer accumulator — no per-expert scalar-broadcast upload or separate add.
/// `scale_buf` is a 1-element device buffer. Dispatches `iron_gemv_q8_coalesced_accum`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q8_accum(
    dev: &dyn Device,
    qs: &Tensor,
    scales: &Tensor,
    x: &Tensor,
    acc: &Tensor,
    scale_buf: &Tensor,
    m_out: usize,
    k_in: usize,
    rows_per_group: usize,
) -> Result<()> {
    if k_in % 32 != 0 {
        return Err(Error::Msg(format!("gemv_q8_accum: k_in {k_in} must be a multiple of 32")));
    }
    let k = cached_ir("iron_gemv_q8_coalesced_accum", acc.dtype, || { let mut k = wh_iron_std::kernels::gemm::gemv_quantized::iron_gemv_q8_coalesced_accum::kernel_ir_for(acc.dtype); k.mode = wh_iron_core::ir::KernelMode::Reduction; k });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid { grid: [m_out as u32, 1, 1], block: [32, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(acc.buffer.clone()),
            Binding::Buffer(scale_buf.buffer.clone()),
            u(k_in as u32),
            u(m_out as u32),
            u(rows_per_group as u32),
        ],
        grid,
    )?;
    Ok(())
}

/// Append one decode step's K (or V) into an in-place device KV cache:
/// `dst[h, pos, :] = src[h, :]`, where `dst` is `[nkv, cap, hd]`, `src` is
/// `[nkv*hd]`, and `posbuf` is a 1-element u32 device buffer. Keeps the growing
/// context entirely on-device — no host reorg/reupload per step (the 32K-context
/// fix). Dispatches `iron_kv_append` (runtime `pos` ⇒ compiled once, not per step).
pub fn kv_append(dev: &dyn Device, src: &Tensor, dst: &Tensor, posbuf: &Tensor, hd: usize, cap: usize, n: usize) -> Result<()> {
    let k = cached_ir("iron_kv_append", src.dtype, || { let mut k = wh_iron_std::kernels::kv_cache::cache::iron_kv_append::kernel_ir_for(src.dtype); k.mode = wh_iron_core::ir::KernelMode::Grid3D; k });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid { grid: [(n as u32).div_ceil(64), 1, 1], block: [64, 1, 1] };
    dev.dispatch(&k, &[Binding::Buffer(src.buffer.clone()), Binding::Buffer(dst.buffer.clone()), Binding::Buffer(posbuf.buffer.clone()), u(hd as u32), u(cap as u32)], grid)?;
    Ok(())
}

/// Device slice `out[i] = src[off + i]` for `len` elements (no host round-trip).
pub fn slice(dev: &dyn Device, src: &Tensor, off: usize, len: usize) -> Result<Tensor> {
    let k = cached_ir("iron_slice", src.dtype, || { let mut k = wh_iron_std::kernels::ops::copy::iron_slice::kernel_ir_for(src.dtype); k.mode = wh_iron_core::ir::KernelMode::Grid3D; k });
    // Defensive bounds check. The kernel reads `src[off + i]` for i in [0, len),
    // addressing a sub-slab purely via `off` against the RAW bound buffer (the
    // tensor.offset is ignored by design). A caller binding a buffer that does
    // not hold `off + len` elements triggers an out-of-bounds READ, silent
    // garbage on Metal, a deterministic MMU-fault on CUDA. Fail loudly, naming
    // the shapes, so the mis-sized caller is obvious on every backend.
    let need = off + len; // max element read is src[off + len - 1]
    let have = src.buffer.len() / src.dtype.size_bytes();
    if need > have {
        return Err(Error::Msg(format!(
            "slice OOB: off={off} len={len} needs {need} {:?} elems but bound buffer holds {have}",
            src.dtype
        )));
    }
    let out = Tensor::empty(dev, vec![len], src.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(src.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(off as u32), u(len as u32)], Grid { grid: [(len as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// Cast f32 → f16 (KV-cache compaction: store attention K/V at half precision
/// to halve the 32K-context sdpa read). Returns a fresh F16 tensor.
pub fn cast_f32_f16(dev: &dyn Device, src: &Tensor) -> Result<Tensor> {
    let k = cached_ir("iron_cast_f32_f16", DType::F16, || { let mut k = wh_iron_std::kernels::ops::unary::iron_cast_f32_f16::kernel_ir(); k.mode = wh_iron_core::ir::KernelMode::Grid3D; k });
    let n = src.elem_count();
    let out = Tensor::empty(dev, src.shape.clone(), DType::F16)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(src.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(n as u32)], Grid { grid: [(n as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// Cast f16 → f32 (reverse: widen the sdpa f16 output back to f32 for the
/// downstream o_proj Q4 GEMV, which consumes f32 activations). Fresh F32 tensor.
pub fn cast_f16_f32(dev: &dyn Device, src: &Tensor) -> Result<Tensor> {
    let k = cached_ir("iron_cast_f16_f32", DType::F32, || { let mut k = wh_iron_std::kernels::ops::unary::iron_cast_f16_f32::kernel_ir(); k.mode = wh_iron_core::ir::KernelMode::Grid3D; k });
    let n = src.elem_count();
    let out = Tensor::empty(dev, src.shape.clone(), DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(src.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(n as u32)], Grid { grid: [(n as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// Cast f32 → bf16. BF16 keeps f32's exponent range, so the residual stream
/// (NEMOTRON_BF16_STREAM) doesn't overflow on NemotronH massive-activation
/// channels the way f16 (max 65504) does.
pub fn cast_f32_bf16(dev: &dyn Device, src: &Tensor) -> Result<Tensor> {
    let k = cached_ir("iron_cast_f32_bf16", DType::BF16, || { let mut k = wh_iron_std::kernels::ops::unary::iron_cast_f32_bf16::kernel_ir(); k.mode = wh_iron_core::ir::KernelMode::Grid3D; k });
    let n = src.elem_count();
    let out = Tensor::empty(dev, src.shape.clone(), DType::BF16)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(src.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(n as u32)], Grid { grid: [(n as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// Cast bf16 → f32 (widen the residual back for ops that need f32 input).
pub fn cast_bf16_f32(dev: &dyn Device, src: &Tensor) -> Result<Tensor> {
    let k = cached_ir("iron_cast_bf16_f32", DType::F32, || { let mut k = wh_iron_std::kernels::ops::unary::iron_cast_to_f32::kernel_ir_for(DType::BF16); k.mode = wh_iron_core::ir::KernelMode::Grid3D; k });
    let n = src.elem_count();
    let out = Tensor::empty(dev, src.shape.clone(), DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(src.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(n as u32)], Grid { grid: [(n as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// Device Mamba dt: `softplus(dt_raw + dt_bias)`, no host round-trip.
pub fn softplus_add(dev: &dyn Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let k = cached_ir("iron_softplus_add", DType::F32, || { let mut k = wh_iron_std::kernels::ssm::scan::iron_softplus_add::kernel_ir(); k.mode = wh_iron_core::ir::KernelMode::Grid3D; k });
    let n = a.elem_count();
    let out = Tensor::empty(dev, a.shape.clone(), DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(a.buffer.clone()), Binding::Buffer(b.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(n as u32)], Grid { grid: [(n as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// NemotronH/Zamba2 gated grouped RMSNorm ON-DEVICE: out = (y·silu(z)) normalized
/// per `gs`-group, ×w. Removes the per-Mamba-layer dl→host-norm→up sync. `y` is f32.
pub fn gated_group_rmsnorm(dev: &dyn Device, y: &Tensor, z: &Tensor, w: &Tensor, eps: f32, di: usize, gs: usize) -> Result<Tensor> {
    let k = cached_ir("iron_gated_group_rmsnorm", z.dtype, || { let mut k = wh_iron_std::kernels::ssm::scan::iron_gated_group_rmsnorm::kernel_ir_for(z.dtype); k.mode = wh_iron_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![di], z.dtype)?;
    let eps_buf = Tensor::new(scalar_buf(dev, eps.to_bits())?, vec![1], DType::F32);
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let ng = (di / gs) as u32;
    dev.dispatch(&k, &[Binding::Buffer(y.buffer.clone()), Binding::Buffer(z.buffer.clone()), Binding::Buffer(w.buffer.clone()), Binding::Buffer(out.buffer.clone()), Binding::Buffer(eps_buf.buffer.clone()), u(gs as u32)], Grid { grid: [ng, 1, 1], block: [(gs / 4) as u32, 1, 1] })?;
    Ok(out)
}

/// Roll a causal-conv state on-device: `new = [old[conv_dim..], xbc]`.
pub fn conv_roll(dev: &dyn Device, old: &Tensor, xbc: &Tensor, conv_dim: usize, kc: usize) -> Result<Tensor> {
    let n = (kc - 1) * conv_dim;
    let keep = (kc - 2) * conv_dim;
    let k = cached_ir("iron_conv_roll", old.dtype, || { let mut k = wh_iron_std::kernels::convolution::conv1d_causal::iron_conv_roll::kernel_ir_for(old.dtype); k.mode = wh_iron_core::ir::KernelMode::Grid3D; k });
    let out = Tensor::empty(dev, vec![n], old.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(old.buffer.clone()), Binding::Buffer(xbc.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(conv_dim as u32), u(keep as u32), u(n as u32)], Grid { grid: [(n as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(out)
}

/// Mamba2 batched-prefill causal depthwise conv1d with inline SiLU (NEMOTRON_CONV_DEVICE path).
///
/// Processes all S prompt tokens in one dispatch with zero initial state (prefill
/// starts from scratch). Each thread computes one element of `y[s, conv_dim]` =
/// silu(bias[ch] + Σ_{k<kc} w[k,ch] * xbc_in[ti-(kc-1-k), ch]) where out-of-bounds
/// reads are zero. No host round-trip.
///
/// - `xbc_in`: `[s * conv_dim]` flat row-major
/// - `w`: `[kc * conv_dim]` reorganized (same layout as decode step: w[k, ch])
/// - `bias`: `[conv_dim]`
/// - returns `y [s * conv_dim]` with SiLU applied
pub fn conv1d_causal_prefill(
    dev: &dyn Device,
    xbc_in: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    s: usize,
    conv_dim: usize,
    kc: usize,
) -> Result<Tensor> {
    let kernel = lookup("iron_conv1d_causal_prefill", DType::F32)?;
    let y = Tensor::empty(dev, vec![s * conv_dim], DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(xbc_in.buffer.clone()),
            Binding::Buffer(w.buffer.clone()),
            Binding::Buffer(bias.buffer.clone()),
            Binding::Buffer(y.buffer.clone()),
            u(conv_dim as u32),
            u(kc as u32),
        ],
        // PERF: pack 256 threads/block (was block:[1,1,1] = 1 thread/block = 1/32
        // warp occupancy). The kernel uses gid_x = blockIdx*blockDim+threadIdx as a
        // global linear element id, so block size is transparent as long as the grid
        // covers exactly s*conv_dim threads. conv_dim (6144) is a multiple of 256, so
        // s*conv_dim is always divisible by 256 → no over-launch / OOB.
        { let n = (s * conv_dim) as u32; let b = 256u32; Grid { grid: [n / b, 1, 1], block: [b, 1, 1] } },
    )?;
    Ok(y)
}

const CONV1D_PREFILL_V4_SRC: &str = r#"
#include <cuda_runtime.h>

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

extern "C" __global__ void conv1d_causal_prefill_v4(
    const float* __restrict__ xbc,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int conv_dim4, int kc, int total4)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total4) return;
    int ti = idx / conv_dim4;
    int ci4 = idx - ti * conv_dim4;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xbc);
    const float4* __restrict__ w4 = reinterpret_cast<const float4*>(w);
    const float4* __restrict__ b4 = reinterpret_cast<const float4*>(bias);
    float4 acc = b4[ci4];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= kc) break;
        int src_t = ti - (kc - 1 - k);
        if (src_t >= 0) {
            float4 xv = x4[(size_t)src_t * conv_dim4 + ci4];
            float4 wv = w4[(size_t)k * conv_dim4 + ci4];
            acc.x += xv.x * wv.x;
            acc.y += xv.y * wv.y;
            acc.z += xv.z * wv.z;
            acc.w += xv.w * wv.w;
        }
    }
    acc.x = silu_f32(acc.x);
    acc.y = silu_f32(acc.y);
    acc.z = silu_f32(acc.z);
    acc.w = silu_f32(acc.w);
    reinterpret_cast<float4*>(y)[idx] = acc;
}

extern "C" __global__ void conv1d_causal_prefill_split_v4(
    const float* __restrict__ xbc,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ x_out,
    float* __restrict__ b_out,
    float* __restrict__ c_out,
    int conv_dim4, int di4, int ng_ds4, int kc, int total4)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total4) return;
    int ti = idx / conv_dim4;
    int ci4 = idx - ti * conv_dim4;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(xbc);
    const float4* __restrict__ w4 = reinterpret_cast<const float4*>(w);
    const float4* __restrict__ b4 = reinterpret_cast<const float4*>(bias);
    float4 acc = b4[ci4];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= kc) break;
        int src_t = ti - (kc - 1 - k);
        if (src_t >= 0) {
            float4 xv = x4[(size_t)src_t * conv_dim4 + ci4];
            float4 wv = w4[(size_t)k * conv_dim4 + ci4];
            acc.x += xv.x * wv.x;
            acc.y += xv.y * wv.y;
            acc.z += xv.z * wv.z;
            acc.w += xv.w * wv.w;
        }
    }
    acc.x = silu_f32(acc.x);
    acc.y = silu_f32(acc.y);
    acc.z = silu_f32(acc.z);
    acc.w = silu_f32(acc.w);
    if (ci4 < di4) {
        reinterpret_cast<float4*>(x_out)[(size_t)ti * di4 + ci4] = acc;
    } else if (ci4 < di4 + ng_ds4) {
        int j = ci4 - di4;
        reinterpret_cast<float4*>(b_out)[(size_t)ti * ng_ds4 + j] = acc;
    } else {
        int j = ci4 - di4 - ng_ds4;
        reinterpret_cast<float4*>(c_out)[(size_t)ti * ng_ds4 + j] = acc;
    }
}
"#;

/// Fused raw-float4 causal conv1d+SiLU that writes the post-conv x/B/C split
/// surfaces directly, deleting the temp [s,conv_dim] conv output + the split
/// launch. Equivalent to conv1d_causal_prefill + mamba_split_conv. Escape with
/// NEMOTRON_CONV_PREFILL_SPLIT_OFF=1 (falls back to that pair).
pub fn conv1d_causal_prefill_split(
    dev: &dyn Device,
    xbc_in: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    s: usize,
    conv_dim: usize,
    di: usize,
    ng_ds: usize,
    kc: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let raw_ok = std::env::var("NEMOTRON_CONV_PREFILL_RAW_OFF").is_err()
        && std::env::var("NEMOTRON_CONV_PREFILL_SPLIT_OFF").is_err()
        && xbc_in.dtype == DType::F32 && w.dtype == DType::F32 && bias.dtype == DType::F32
        && xbc_in.offset % 16 == 0 && w.offset % 16 == 0 && bias.offset % 16 == 0
        && di + 2 * ng_ds == conv_dim
        && di % 4 == 0 && ng_ds % 4 == 0 && conv_dim % 4 == 0;
    if raw_ok {
        let x_out = Tensor::empty(dev, vec![s * di], DType::F32)?;
        let b_out = Tensor::empty(dev, vec![s * ng_ds], DType::F32)?;
        let c_out = Tensor::empty(dev, vec![s * ng_ds], DType::F32)?;
        let i = |v: i32| v.to_le_bytes().to_vec();
        let total4 = (s * conv_dim / 4) as u32;
        if dev.dispatch_raw_cuda(
            CONV1D_PREFILL_V4_SRC,
            "conv1d_prefill_v4.cu",
            "conv1d_causal_prefill_split_v4",
            &[
                (xbc_in.buffer.as_ref(), xbc_in.offset),
                (w.buffer.as_ref(), w.offset),
                (bias.buffer.as_ref(), bias.offset),
                (x_out.buffer.as_ref(), 0),
                (b_out.buffer.as_ref(), 0),
                (c_out.buffer.as_ref(), 0),
            ],
            &[
                i((conv_dim / 4) as i32),
                i((di / 4) as i32),
                i((ng_ds / 4) as i32),
                i(kc as i32),
                i(total4 as i32),
            ],
            [total4.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            false,
        ).is_ok() {
            return Ok((x_out, b_out, c_out));
        }
    }
    let y = conv1d_causal_prefill(dev, xbc_in, w, bias, s, conv_dim, kc)?;
    mamba_split_conv(dev, &y, s, conv_dim, di, ng_ds)
}

/// Extract `[s, width]` columns from row-major `src [s, stride]` at column offset `col_off`.
/// Returns `dst [s * width]`, a contiguous slab. Used to carve z, xbc, dt_raw out of
/// the [s, in_proj_out] projection matrix on device (NEMOTRON_CONV_DEVICE path).
pub fn strided_col_copy(
    dev: &dyn Device,
    src: &Tensor,
    s: usize,
    stride: usize,
    col_off: usize,
    width: usize,
) -> Result<Tensor> {
    let kernel = lookup("iron_strided_col_copy", DType::F32)?;
    // Defensive bounds check. The kernel reads `src[ti*stride + col_off + ci]`
    // for ti in [0, s), ci in [0, width): a sub-slab addressed purely via the
    // host stride/col_off against the RAW bound buffer (tensor.offset ignored by
    // design). A caller binding a row-stride/offset/width that overruns the
    // bound buffer triggers an out-of-bounds READ, silent garbage on Metal, a
    // deterministic MMU-fault on CUDA. Fail loudly, naming the shapes.
    if s != 0 && width != 0 {
        let need = (s - 1) * stride + col_off + width; // max read is src[need - 1]
        let have = src.buffer.len() / DType::F32.size_bytes();
        if need > have {
            return Err(Error::Msg(format!(
                "strided_col_copy OOB: s={s} stride={stride} col_off={col_off} width={width} \
                 needs {need} f32 elems but bound buffer holds {have}"
            )));
        }
    }
    let dst = Tensor::empty(dev, vec![s * width], DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let total = (s * width) as u32;
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(src.buffer.clone()),
            Binding::Buffer(dst.buffer.clone()),
            u(stride as u32),
            u(col_off as u32),
            u(width as u32),
        ],
        // The kernel has no internal idx guard, so the grid must cover exactly
        // s*width threads. Laguna's gate-column extraction is not always
        // 64-aligned (head counts 48/72): use the largest power-of-two block
        // (<= 64) that divides the total — 64 for the common aligned shapes.
        {
            let b = 1u32 << total.trailing_zeros().min(6);
            Grid { grid: [total / b, 1, 1], block: [b, 1, 1] }
        },
    )?;
    Ok(dst)
}

const EXTRACT_COLS_SRC: &str = r#"
extern "C" __global__ void butter_extract_cols(
    const float* __restrict__ src, float* __restrict__ dst,
    int rows, int stride, int col_off, int width)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * width;
    if (i >= total) return;
    int c = (int)(i % width);
    long r = i / width;
    dst[i] = src[r * stride + col_off + c];
}
"#;

// f16 mirror of EXTRACT_COLS_SRC: same indexing, 16-bit loads/stores. Added
// so prefill's Marlin/grouped-MoE-GEMM column splits (both f16-native
// producers) can extract without a whole-buffer cast_f16_f32 first, see
// `extract_cols`'s doc.
const EXTRACT_COLS_F16_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void butter_extract_cols_f16(
    const __half* __restrict__ src, __half* __restrict__ dst,
    int rows, int stride, int col_off, int width)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * width;
    if (i >= total) return;
    int c = (int)(i % width);
    long r = i / width;
    dst[i] = src[r * stride + col_off + c];
}
"#;

/// Extract a `[rows, width]` column range starting at `col_off` from every
/// row of a `[rows, stride]` row-major buffer (`stride >= col_off + width`).
/// Used to split Laguna's fused `[T, n_q_rows+2*n_kv_rows+n_g_rows]`
/// QKVG-concat batched-prefill projection back into its Q/K/V/gate column
/// ranges (and, in f16, to split a Marlin/grouped-MoE-GEMM's fused
/// `[gate;up]` output without a cast first). Originally added as a
/// workaround for a `strided_col_copy` launch-geometry bug (fixed grid
/// dropped the tail whenever `rows*width` wasn't a multiple of 64, true for
/// Laguna's gate column range, whose width is the per-layer head count of 48
/// or 72, neither 64-aligned, times an arbitrary chunk length).
/// `strided_col_copy` now div_ceils its grid and guards in-kernel, so this
/// raw CUDA path is kept as-is only to avoid touching a call site mid-flight
/// elsewhere; this kernel's grid is sized to cover every element regardless
/// of `rows*width`'s divisibility. f32 and f16 (dispatched on `src.dtype`);
/// any other dtype errors.
pub fn extract_cols(dev: &dyn Device, src: &Tensor, rows: usize, stride: usize, col_off: usize, width: usize) -> Result<Tensor> {
    let (elem_bytes, cuda_src, prog_name, fn_name) = match src.dtype {
        DType::F32 => (DType::F32.size_bytes(), EXTRACT_COLS_SRC, "extract_cols.cu", "butter_extract_cols"),
        DType::F16 => (DType::F16.size_bytes(), EXTRACT_COLS_F16_SRC, "extract_cols_f16.cu", "butter_extract_cols_f16"),
        other => return Err(Error::Msg(format!("extract_cols: unsupported dtype {other:?} (f32/f16 only)"))),
    };
    if rows != 0 && width != 0 {
        let need = (rows - 1) * stride + col_off + width; // max read is src[need - 1]
        let have = src.buffer.len() / elem_bytes;
        if need > have {
            return Err(Error::Msg(format!(
                "extract_cols OOB: rows={rows} stride={stride} col_off={col_off} width={width} \
                 needs {need} {:?} elems but bound buffer holds {have}", src.dtype
            )));
        }
    }
    let dst = Tensor::empty(dev, vec![rows * width], src.dtype)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    let total = (rows * width) as u32;
    dev.dispatch_raw_cuda(
        cuda_src, prog_name, fn_name,
        &[(src.buffer.as_ref(), src.offset), (dst.buffer.as_ref(), 0)],
        &[i(rows as i32), i(stride as i32), i(col_off as i32), i(width as i32)],
        [total.div_ceil(256).max(1), 1, 1], [256, 1, 1], 0, false,
    )?;
    Ok(dst)
}

/// Batched softplus + tiled bias: `dst[ti*n + hi] = softplus(src[ti*n + hi] + bias[hi])`.
/// Converts the raw `[s, m_nh]` dt_raw tensor + dt_bias to `dt_all [s, m_nh]` on device.
/// (NEMOTRON_CONV_DEVICE path, replaces the CPU softplus loop.)
pub fn softplus_add_rows(
    dev: &dyn Device,
    src: &Tensor,
    bias: &Tensor,
    s: usize,
    n: usize,
) -> Result<Tensor> {
    let kernel = lookup("iron_softplus_add_rows", DType::F32)?;
    let dst = Tensor::empty(dev, vec![s * n], DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(src.buffer.clone()),
            Binding::Buffer(bias.buffer.clone()),
            Binding::Buffer(dst.buffer.clone()),
            u(n as u32),
        ],
        // PERF: pack threads/block (was 1). Kernel uses a global linear element id,
        // so any block size works provided the grid covers exactly s*n threads.
        // Pick the largest of {64,32,1} that divides s*n to avoid OOB over-launch.
        {
            let nt = (s * n) as u32;
            let b = if nt % 64 == 0 { 64u32 } else if nt % 32 == 0 { 32u32 } else { 1u32 };
            Grid { grid: [nt / b, 1, 1], block: [b, 1, 1] }
        },
    )?;
    Ok(dst)
}

/// Batched MoE up+ReLU²: gather `top_k` experts (indices `idx`) from the
/// contiguous `[n_exp*inter, hid]` Q4 weight into one `[top_k*inter]` GEMV.
pub fn moe_gather_up_relu2(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, idx: &Tensor, top_k: usize, inter: usize, hid: usize) -> Result<Tensor> {
    let k = cached_ir("iron_moe_gather_q4_relu2", x.dtype, || { let mut k = wh_iron_std::kernels::moe::moe_gather_q4::iron_moe_gather_q4_relu2::kernel_ir_for(x.dtype); k.mode = wh_iron_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![top_k * inter], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    // Default rpt=2: MoE-gather kernels are big + latency-bound; 2 warps/row hides
    // global-load latency (clean internal-A/B: +1.8% @ctx4096, MoE cost is ctx-independent).
    let rpt: u32 = std::env::var("BUTTER_MOE_RPT").ok().and_then(|v| v.parse().ok()).filter(|&r| r >= 1).unwrap_or(2);
    dev.dispatch(&k, &[Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(idx.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(hid as u32), u(inter as u32), u(rpt)], Grid { grid: [(inter as u32).div_ceil(rpt), top_k as u32, 1], block: [32 * rpt, 1, 1] })?;
    Ok(out)
}
// Single-block strided argmax over an f32 vector (the 100K-vocab logit row).
// Ties resolve to the LOWEST index, matching the host-side scan. One block of
// 1024 threads striding n, then a shared-memory tree reduce: microseconds at
// vocab scale, and it shrinks the per-step D2H from the full logit row to 4
// bytes for greedy decode.
const ARGMAX_F32_SRC: &str = r#"
extern "C" __global__ void butter_argmax_f32(
    const float* __restrict__ x, unsigned* __restrict__ out, int n)
{
    __shared__ float sv[1024];
    __shared__ unsigned si[1024];
    int tid = threadIdx.x;
    float best = -3.4028235e38f;
    unsigned bi = 0u;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[i];
        if (v > best || (v == best && (unsigned)i < bi)) { best = v; bi = (unsigned)i; }
    }
    sv[tid] = best; si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) {
                sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = si[0];
}
"#;

/// Device argmax of an f32 vector: returns a 1-element u32 tensor holding the
/// index of the maximum (ties: lowest index). Greedy decode downloads these 4
/// bytes instead of the whole logit row.
pub fn argmax_f32_device(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    let n = x.elem_count();
    let out = Tensor::empty(dev, vec![1], DType::U32)?;
    let i = |v: i32| v.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(
        ARGMAX_F32_SRC, "argmax_f32.cu", "butter_argmax_f32",
        &[(x.buffer.as_ref(), x.offset), (out.buffer.as_ref(), 0)],
        &[i(n as i32)],
        [1, 1, 1], [1024, 1, 1], 0, false,
    )?;
    Ok(out)
}

/// Batched MoE expert gather GEMV, plain (no fused activation): one
/// `[top_k*inter]` output from the contiguous `[n_exp*inter, hid]` Q4 stack.
/// The SwiGLU MoE path calls this for the gate and up stacks, then applies
/// the activation elementwise before the down gather. F16 per-block scales.
///
/// TODO(rebrand-merge): blocked — dispatches `iron_moe_gather_q4` via
/// `wh_iron_std`'s kernel IR, but the pinned `thewafflehaus/iron@dev` kernel
/// library only ships `iron_moe_gather_q4_relu2/_down/_down_accum` — the
/// plain (non-relu2) variant this needs is not present upstream yet. Kept as
/// an always-error stub until that kernel lands in the iron repo.
pub fn moe_gather_q4(_dev: &dyn Device, _qs: &Tensor, _scales: &Tensor, _x: &Tensor, _idx: &Tensor, _top_k: usize, _inter: usize, _hid: usize) -> Result<Tensor> {
    Err(Error::Msg("moe_gather_q4: blocked — iron_moe_gather_q4 kernel not yet in thewafflehaus/iron@dev (only _relu2/_down/_down_accum variants exist)".into()))
}

/// Batched MoE GATE+UP gather with FUSED inline SwiGLU: reads the SAME
/// activation `x` once and computes both the gate and up dot products for
/// each (slot, inter-row), writing `silu(gate)·up` straight to a single
/// `[top_k*inter]` output. Replaces two `moe_gather_q4` calls (gate stack,
/// up stack) plus a host-side elementwise SwiGLU pass with one dispatch.
/// F16 per-block scales on both `gate_scales` and `up_scales`.
#[allow(clippy::too_many_arguments)]
/// TODO(rebrand-merge): blocked — same iron-kernel gap as `moe_gather_q4`
/// above (`iron_moe_gather_q4_swiglu` not in `thewafflehaus/iron@dev` yet).
#[allow(clippy::too_many_arguments)]
pub fn moe_gather_q4_swiglu(_dev: &dyn Device, _gate_qs: &Tensor, _gate_scales: &Tensor, _up_qs: &Tensor, _up_scales: &Tensor, _x: &Tensor, _idx: &Tensor, _top_k: usize, _inter: usize, _hid: usize) -> Result<Tensor> {
    Err(Error::Msg("moe_gather_q4_swiglu: blocked — iron_moe_gather_q4_swiglu kernel not yet in thewafflehaus/iron@dev".into()))
}

/// Batched MoE down + router-weighted accumulate into `acc[hid]`. `qs` is the
/// contiguous `[n_exp*hid, inter]` Q4 weight; `x` is the `[top_k*inter]` up output.
#[allow(clippy::too_many_arguments)]
pub fn moe_gather_down_accum(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, idx: &Tensor, wts: &Tensor, acc: &Tensor, top_k: usize, inter: usize, hid: usize) -> Result<()> {
    let k = cached_ir("iron_moe_gather_q4_down_accum", x.dtype, || { let mut k = wh_iron_std::kernels::moe::moe_gather_q4::iron_moe_gather_q4_down_accum::kernel_ir_for(x.dtype); k.mode = wh_iron_core::ir::KernelMode::Reduction; k });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(idx.buffer.clone()), Binding::Buffer(wts.buffer.clone()), Binding::Buffer(acc.buffer.clone()), u(inter as u32), u(hid as u32), u(top_k as u32)], Grid { grid: [hid as u32, 1, 1], block: [32, 1, 1] })?;
    Ok(())
}

/// Batched MoE down gather → `[top_k*hid]` (one big GEMV, no accumulate).
pub fn moe_gather_down(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, idx: &Tensor, top_k: usize, inter: usize, hid: usize) -> Result<Tensor> {
    let k = cached_ir("iron_moe_gather_q4_down", x.dtype, || { let mut k = wh_iron_std::kernels::moe::moe_gather_q4::iron_moe_gather_q4_down::kernel_ir_for(x.dtype); k.mode = wh_iron_core::ir::KernelMode::Reduction; k });
    let out = Tensor::empty(dev, vec![top_k * hid], x.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    // Default rpt=2: MoE-gather kernels are big + latency-bound; 2 warps/row hides
    // global-load latency (clean internal-A/B: +1.8% @ctx4096, MoE cost is ctx-independent).
    let rpt: u32 = std::env::var("BUTTER_MOE_RPT").ok().and_then(|v| v.parse().ok()).filter(|&r| r >= 1).unwrap_or(2);
    dev.dispatch(&k, &[Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(idx.buffer.clone()), Binding::Buffer(out.buffer.clone()), u(inter as u32), u(hid as u32), u(rpt)], Grid { grid: [(hid as u32).div_ceil(rpt), top_k as u32, 1], block: [32 * rpt, 1, 1] })?;
    Ok(out)
}
// ── Fused MoE FFN (NEMOTRON_MOE_FUSED=1) ─────────────────────────────────────
// Raw CUDA C++ source for the cooperative-groups two-phase fused kernel.
//
// Grid design for cooperative launch feasibility on GB10 (148 SMs, 2048 threads/SM):
//   total_warps = hid = 2688  (one warp per acc[h] output element in phase 2)
//   block = 256 (8 warps), grid = ceil(hid/8) = 336 blocks
//   336 blocks / 148 SMs ≈ 2.3 blocks/SM < 8 max → cooperative launch WORKS
//
// Phase 1: each warp serializes over its assigned slice of top_k*inter up-proj rows.
//   rows_per_warp ≈ (top_k * inter) / hid ≈ 4.14 (integer: 4 or 5 rows per warp)
//   → warp `w` computes scratch[row] for row in [w*n_up/hid .. (w+1)*n_up/hid)
// Phase 2: after grid.sync(), warp `w` computes acc[w] (one output element) by
//   reading ALL of scratch and the down Q4 weights (down-proj GEMV over inter).
//
// The 44KB scratch (6×1856×4B) lives in L2 (~2TB/s GB10) throughout, not HBM.
const MOE_FUSED_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Q4 (signed 4-bit, nibble-packed) + f16 scale per 32 → f32 warp dot product.
// lane 0 holds the result after return.
__device__ __forceinline__ float q4f16_dot_warp(
    const unsigned* __restrict__ qs,
    const __half*   __restrict__ sc,
    const float*    __restrict__ x,
    int row, int k_in, int lane)
{
    int bpr = k_in >> 5;
    int nw  = bpr * 4;
    const unsigned* qrow = qs + (size_t)row * nw;
    const __half*   drow = sc + (size_t)row * bpr;
    float dot = 0.f;
    for (int j = lane; j < nw; j += 32) {
        int blk = j >> 2, sub = j & 3;
        unsigned p = qrow[j];
        float sc_f = __half2float(drow[blk]);
        const float* xb = x + (blk << 5) + (sub << 3);
        float a = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int nb = (p >> (i * 4)) & 0xf;
            a += (float)(nb > 7 ? nb - 16 : nb) * xb[i];
        }
        dot += sc_f * a;
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
    return dot;
}

__device__ __forceinline__ float relu2(float v) {
    float r = v > 0.f ? v : 0.f;
    return r * r;
}

extern "C" __global__ void moe_fused_ffn(
    const unsigned* __restrict__ up_qs,   // [n_exp*inter, hid] Q4 up weights
    const __half*   __restrict__ up_sc,   // [n_exp*inter, hid/32] f16 scales
    const unsigned* __restrict__ dn_qs,   // [n_exp*hid, inter] Q4 down weights
    const __half*   __restrict__ dn_sc,   // [n_exp*hid, inter/32] f16 scales
    const float*    __restrict__ x,       // [hid] input activation
    const unsigned* __restrict__ idx,     // [top_k] expert indices (u32)
    const float*    __restrict__ wts,     // [top_k] router weights (f32)
    float*          __restrict__ acc,     // [hid] output (pre-zeroed by caller)
    float*          __restrict__ scratch, // [top_k * inter] temp (pre-allocated)
    int hid, int inter, int top_k)
{
    cg::grid_group gg = cg::this_grid();

    // Global warp id, one warp per acc[h] for phase 2.
    int tid     = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane    = threadIdx.x & 31;
    int warp_id = tid >> 5;
    int n_warps = (int)gridDim.x * ((int)blockDim.x >> 5);

    // ── Phase 1: each warp serializes over its slice of up-proj rows ──────
    // Linearised row space: row_lin = k * inter + i  (k=expert index 0..top_k-1)
    int n_up = top_k * inter;
    // Use 64-bit division to avoid overflow; compiler optimises for small top_k.
    int row_start = (int)((long long)warp_id * n_up / n_warps);
    int row_end   = (int)((long long)(warp_id + 1) * n_up / n_warps);
    for (int row_lin = row_start; row_lin < row_end; row_lin++) {
        int k   = row_lin / inter;
        int i   = row_lin % inter;
        int exp = (int)idx[k];
        int row = exp * inter + i;
        float v = q4f16_dot_warp(up_qs, up_sc, x, row, hid, lane);
        if (lane == 0) scratch[k * inter + i] = relu2(v);
    }

    // ── Global grid barrier: all up-proj results are in scratch ──────────
    gg.sync();

    // ── Phase 2: each warp computes one acc[h] via down-proj ─────────────
    if (warp_id < hid) {
        int h = warp_id;
        float acc_h = 0.f;
        for (int k = 0; k < top_k; k++) {
            int exp = (int)idx[k];
            int row = exp * hid + h;
            int bpr = inter >> 5;
            int nw  = bpr * 4;
            const unsigned* qrow  = dn_qs + (size_t)row * nw;
            const __half*   drow  = dn_sc + (size_t)row * bpr;
            const float*    xb_base = scratch + k * inter;
            float dot = 0.f;
            for (int j = lane; j < nw; j += 32) {
                int blk = j >> 2, sub = j & 3;
                unsigned p = qrow[j];
                float sc_f = __half2float(drow[blk]);
                const float* xb = xb_base + (blk << 5) + (sub << 3);
                float a = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int nb = (p >> (i * 4)) & 0xf;
                    a += (float)(nb > 7 ? nb - 16 : nb) * xb[i];
                }
                dot += sc_f * a;
            }
            #pragma unroll
            for (int o = 16; o; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
            if (lane == 0) acc_h += dot * wts[k];
        }
        if (lane == 0) acc[h] += acc_h;
    }
}
"#;

/// Fused MoE FFN: up-proj + relu² + down-proj + router-weighted accumulate,
/// all in ONE cooperative-groups kernel. The `[top_k×inter]` intermediate lives
/// in L2 cache instead of HBM (scratch pre-allocated once by caller).
///
/// Replaces `moe_gather_up_relu2` + `moe_gather_down` + `moe_weighted_sum` with
/// a single launch. Enabled by `NEMOTRON_MOE_FUSED=1`.
///
/// `scratch` must be a device buffer of at least `top_k * inter * 4` bytes.
/// `acc` must be zero-initialised by the caller before the first expert's
/// contribution (matches `moe_gather_down_accum` contract).
#[allow(clippy::too_many_arguments)]
pub fn moe_fused_ffn(
    dev: &dyn Device,
    up_qs: &Tensor, up_sc: &Tensor,
    dn_qs: &Tensor, dn_sc: &Tensor,
    x: &Tensor,
    idx: &Tensor,
    wts: &Tensor,
    acc: &Tensor,
    scratch: &Tensor,
    hid: usize, inter: usize, top_k: usize,
) -> Result<()> {
    // n_warps = hid: one warp per acc[h] output element (phase 2).
    // Phase 1 distributes top_k*inter up-proj rows across the same n_warps warps
    // (each warp handles ≈4 rows = top_k*inter/hid ≈ 4.14 for NemotronH dims).
    // Grid: ceil(hid/16) = 168 blocks. GB10: 48 SMs × 4 blocks/SM (at 512 threads)
    // = 192 max cooperative blocks → 168 ≤ 192 ✓
    let block = 512u32;  // 16 warps/block
    let warps_per_block = block / 32;
    let grid = ((hid as u32) + warps_per_block - 1) / warps_per_block;
    let scalars: &[Vec<u8>] = &[
        (hid   as i32).to_le_bytes().to_vec(),
        (inter as i32).to_le_bytes().to_vec(),
        (top_k as i32).to_le_bytes().to_vec(),
    ];
    dev.dispatch_raw_cuda(
        MOE_FUSED_SRC,
        "moe_fused_ffn.cu",
        "moe_fused_ffn",
        &[
            (up_qs.buffer.as_ref(),   0),
            (up_sc.buffer.as_ref(),   0),
            (dn_qs.buffer.as_ref(),   0),
            (dn_sc.buffer.as_ref(),   0),
            (x.buffer.as_ref(),       0),
            (idx.buffer.as_ref(),     0),
            (wts.buffer.as_ref(),     0),
            (acc.buffer.as_ref(),     0),
            (scratch.buffer.as_ref(), 0),
        ],
        scalars,
        [grid, 1, 1],
        [block, 1, 1],
        0,
        true,  // cooperative = use cuLaunchCooperativeKernel for grid.sync()
    )
}

/// Router-weighted sum of per-expert down outputs into `acc`: `acc[h] += Σ wts·downs[·,h]`.
/// Fully ON-DEVICE MoE router (NemotronH sigmoid + e_score_correction_bias, top-k by
/// biased score, weights from unbiased renormalized to sum-1, ×routed_scaling_factor).
/// Returns (idx[top_k] U32, wts[top_k] F32) WITHOUT a host round-trip, replaces the
/// per-layer dl(gate)+host-topk+up(idx) sync that drains the async pipeline 23×/token.
pub fn moe_router_device(dev: &dyn Device, gate_logits: &Tensor, bias: &Tensor, n_exp: usize, top_k: usize, scale: f32) -> Result<(Tensor, Tensor)> {
    use wh_iron_core::ir::KernelMode;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let f = |v: f32| Binding::Scalar(v.to_le_bytes().to_vec());
    let unbiased = Tensor::empty(dev, vec![n_exp], DType::F32)?;
    let biased = Tensor::empty(dev, vec![n_exp], DType::F32)?;
    let ks = cached_ir("iron_moe_sigmoid_bias", DType::F32, || { let mut k = wh_iron_std::kernels::moe::moe_sigmoid_bias::iron_moe_sigmoid_bias::kernel_ir(); k.mode = KernelMode::Grid3D; k });
    dev.dispatch(&ks, &[Binding::Buffer(gate_logits.buffer.clone()), Binding::Buffer(bias.buffer.clone()), Binding::Buffer(unbiased.buffer.clone()), Binding::Buffer(biased.buffer.clone()), u(n_exp as u32)], Grid { grid: [(n_exp as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    let idx = Tensor::empty(dev, vec![top_k], DType::U32)?;
    let wts = Tensor::empty(dev, vec![top_k], DType::F32)?;
    let kr = cached_ir("iron_dsv4_router_topk", DType::F32, || { let mut k = wh_iron_std::kernels::moe::moe_router_topk_biased::iron_moe_router_topk_biased::kernel_ir_for(DType::F32); k.mode = KernelMode::Reduction; k });
    dev.dispatch(&kr, &[Binding::Buffer(biased.buffer.clone()), Binding::Buffer(unbiased.buffer.clone()), Binding::Buffer(idx.buffer.clone()), Binding::Buffer(wts.buffer.clone()), u(n_exp as u32), u(top_k as u32)], Grid { grid: [1, 1, 1], block: [32, 1, 1] })?;
    let kv = cached_ir("iron_vscale", DType::F32, || { let mut k = wh_iron_std::kernels::ops::unary::iron_vscale::kernel_ir(); k.mode = KernelMode::Grid3D; k });
    dev.dispatch(&kv, &[Binding::Buffer(wts.buffer.clone()), f(scale), u(top_k as u32)], Grid { grid: [(top_k as u32).div_ceil(64), 1, 1], block: [64, 1, 1] })?;
    Ok((idx, wts))
}

/// On-device MoE route + counting-sort. Replaces the per-E-layer HOST `triples`
/// round-trip (router logits → host → sigmoid+topk+sort → upload) that stalls the
/// GPU every MoE layer and blocks CUDA-graph capture of prefill. Given gate logits
/// `[s, n_exp]` (f32) + bias `[n_exp]`, produces (ALL on device):
///   sorted_tok `[mt]` u32, token index per sorted slot (mt = s*top_k)
///   sorted_wt  `[mt]` f32, router weight per slot (sigmoid, normalized, ·scale)
///   offsets    `[n_exp+1]` u32, group boundaries (expert e's rows = [off[e],off[e+1]))
/// Tokens grouped contiguously by expert. Order within an expert is atomic-cursor
/// (not stable), same near-tie nondeterminism as the existing GROUPED_GEMM path.
const MOE_BATCHED_ROUTER_SRC: &str = r#"
extern "C" __global__ void moe_batched_router(
    const float* __restrict__ logits, const float* __restrict__ bias,
    unsigned* __restrict__ idx_out, float* __restrict__ wt_out,
    int s, int n_exp, int top_k, float scale)
{
    __shared__ float u[256];
    __shared__ float b[256];
    __shared__ int chosen[32];
    int ti = blockIdx.x;
    int e = threadIdx.x;
    if (ti < s) {
        for (int j = e; j < n_exp; j += blockDim.x) {
            float z = logits[(long)ti*n_exp + j];
            float sg = 1.f/(1.f+expf(-z));   // precise expf (not __expf) to match host topk exactly
            u[j] = sg; b[j] = sg + bias[j];
        }
    }
    __syncthreads();
    if (ti < s && e == 0) {
        for (int r = 0; r < top_k; r++) {
            int bi = 0; float bv = -1e30f;
            for (int j = 0; j < n_exp; j++) { if (b[j] > bv) { bv = b[j]; bi = j; } }
            chosen[r] = bi; b[bi] = -1e30f;
        }
        float ws = 1e-20f;
        for (int r = 0; r < top_k; r++) ws += u[chosen[r]];
        for (int r = 0; r < top_k; r++) {
            idx_out[(long)ti*top_k + r] = (unsigned)chosen[r];
            wt_out[(long)ti*top_k + r] = u[chosen[r]] / ws * scale;
        }
    }
}
"#;
const MOE_HIST_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void moe_hist(const unsigned* __restrict__ idx, unsigned* counts, int mt) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= mt) return;
    atomicAdd(counts + idx[i], 1u);
}
"#;
const MOE_PREFIX_SRC: &str = r#"
extern "C" __global__ void moe_prefix(const unsigned* __restrict__ counts,
                                      unsigned* __restrict__ offsets, unsigned* __restrict__ cursor, int n_exp) {
    if (threadIdx.x == 0) {
        unsigned acc = 0; offsets[0] = 0;
        for (int j = 0; j < n_exp; j++) { cursor[j] = acc; acc += counts[j]; offsets[j+1] = acc; }
    }
}
"#;
const MOE_SCATTER_SORT_SRC: &str = r#"
extern "C" __global__ void moe_scatter_sort(
    const unsigned* __restrict__ idx, const float* __restrict__ wt,
    const unsigned* __restrict__ offsets, unsigned* __restrict__ sorted_tok, float* __restrict__ sorted_wt,
    int mt, int top_k, int n_exp)
{
    // STABLE counting-sort scatter, warp-per-expert: lanes sweep 32 consecutive
    // pairs per step (coalesced); __ballot gives each match its stable rank
    // (popc of lower lanes) on top of a running per-warp base. Deterministic,
    // no atomics, same output order as the old 1-thread-per-expert scan but
    // 32-way parallel per expert. O(n_exp*mt/32) per warp.
    int e = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; if (e >= n_exp) return;
    int lane = threadIdx.x & 31;
    unsigned base = offsets[e];
    for (int w = 0; w < mt; w += 32) {
        int i = w + lane;
        bool match = (i < mt) && (idx[i] == (unsigned)e);
        unsigned mask = __ballot_sync(0xffffffffu, match);
        if (match) {
            unsigned rank = __popc(mask & ((1u << lane) - 1u));
            sorted_tok[base + rank] = (unsigned)(i / top_k);
            sorted_wt[base + rank] = wt[i];
        }
        base += __popc(mask);
    }
}

extern "C" __global__ void moe_scatter_sort_rowpos(
    const unsigned* __restrict__ idx, const float* __restrict__ wt,
    const unsigned* __restrict__ offsets, unsigned* __restrict__ sorted_tok,
    float* __restrict__ sorted_wt, unsigned* __restrict__ rowpos,
    int mt, int top_k, int n_exp)
{
    // Same stable expert-major ordering as moe_scatter_sort, plus a token-major
    // slot -> sorted-row map for deterministic fixed-top_k output scatter.
    int e = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; if (e >= n_exp) return;
    int lane = threadIdx.x & 31;
    unsigned base = offsets[e];
    for (int w = 0; w < mt; w += 32) {
        int i = w + lane;
        bool match = (i < mt) && (idx[i] == (unsigned)e);
        unsigned mask = __ballot_sync(0xffffffffu, match);
        if (match) {
            unsigned rank = __popc(mask & ((1u << lane) - 1u));
            unsigned pos = base + rank;
            sorted_tok[pos] = (unsigned)(i / top_k);
            sorted_wt[pos] = wt[i];
            rowpos[i] = pos;
        }
        base += __popc(mask);
    }
}
"#;

#[allow(clippy::type_complexity)]
pub fn moe_route_sort_device(
    dev: &dyn Device, gate_logits: &Tensor, bias: &Tensor, n_exp: usize, top_k: usize, scale: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let s = gate_logits.elem_count() / n_exp;
    let mt = s * top_k;
    let idx = Tensor::empty(dev, vec![mt], DType::U32)?;
    let wt = Tensor::empty(dev, vec![mt], DType::F32)?;
    let counts = Tensor::new(dev.alloc_zeroed(n_exp * 4).map_err(|e| Error::Msg(format!("{e:?}")))?, vec![n_exp], DType::U32);
    let offsets = Tensor::empty(dev, vec![n_exp + 1], DType::U32)?;
    let cursor = Tensor::empty(dev, vec![n_exp], DType::U32)?;
    let sorted_tok = Tensor::empty(dev, vec![mt], DType::U32)?;
    let sorted_wt = Tensor::empty(dev, vec![mt], DType::F32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    let blk = if n_exp >= 256 { 256 } else { n_exp.max(32) } as u32;
    // 1. batched router → idx/wt
    dev.dispatch_raw_cuda(MOE_BATCHED_ROUTER_SRC, "moe_route_sort.cu", "moe_batched_router",
        &[(gate_logits.buffer.as_ref(), gate_logits.offset), (bias.buffer.as_ref(), bias.offset),
          (idx.buffer.as_ref(), 0), (wt.buffer.as_ref(), 0)],
        &[i(s as i32), i(n_exp as i32), i(top_k as i32), f(scale)],
        [s as u32, 1, 1], [blk, 1, 1], 0, false)?;
    // 2. histogram
    dev.dispatch_raw_cuda(MOE_HIST_SRC, "moe_hist.cu", "moe_hist",
        &[(idx.buffer.as_ref(), 0), (counts.buffer.as_ref(), 0)],
        &[i(mt as i32)], [(mt as u32).div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
    // 3. exclusive prefix → offsets + cursor (cursor[e] = offsets[e])
    dev.dispatch_raw_cuda(MOE_PREFIX_SRC, "moe_prefix.cu", "moe_prefix",
        &[(counts.buffer.as_ref(), 0), (offsets.buffer.as_ref(), 0), (cursor.buffer.as_ref(), 0)],
        &[i(n_exp as i32)], [1, 1, 1], [32, 1, 1], 0, false)?;
    // 4. STABLE scatter (one thread/expert, scans in index order) → sorted output
    let _ = &cursor; // cursor no longer used (stable scatter reads offsets directly)
    dev.dispatch_raw_cuda(MOE_SCATTER_SORT_SRC, "moe_scatter_sort.cu", "moe_scatter_sort",
        &[(idx.buffer.as_ref(), 0), (wt.buffer.as_ref(), 0), (offsets.buffer.as_ref(), 0),
          (sorted_tok.buffer.as_ref(), 0), (sorted_wt.buffer.as_ref(), 0)],
        &[i(mt as i32), i(top_k as i32), i(n_exp as i32)], [((n_exp * 32) as u32).div_ceil(128), 1, 1], [128, 1, 1], 0, false)?;
    Ok((sorted_tok, sorted_wt, offsets))
}

#[allow(clippy::type_complexity)]
pub fn moe_route_sort_device_rowpos(
    dev: &dyn Device, gate_logits: &Tensor, bias: &Tensor, n_exp: usize, top_k: usize, scale: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let s = gate_logits.elem_count() / n_exp;
    let mt = s * top_k;
    let idx = Tensor::empty(dev, vec![mt], DType::U32)?;
    let wt = Tensor::empty(dev, vec![mt], DType::F32)?;
    let counts = Tensor::new(dev.alloc_zeroed(n_exp * 4).map_err(|e| Error::Msg(format!("{e:?}")))?, vec![n_exp], DType::U32);
    let offsets = Tensor::empty(dev, vec![n_exp + 1], DType::U32)?;
    let cursor = Tensor::empty(dev, vec![n_exp], DType::U32)?;
    let sorted_tok = Tensor::empty(dev, vec![mt], DType::U32)?;
    let sorted_wt = Tensor::empty(dev, vec![mt], DType::F32)?;
    let rowpos = Tensor::empty(dev, vec![mt], DType::U32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    let blk = if n_exp >= 256 { 256 } else { n_exp.max(32) } as u32;
    dev.dispatch_raw_cuda(MOE_BATCHED_ROUTER_SRC, "moe_route_sort.cu", "moe_batched_router",
        &[(gate_logits.buffer.as_ref(), gate_logits.offset), (bias.buffer.as_ref(), bias.offset),
          (idx.buffer.as_ref(), 0), (wt.buffer.as_ref(), 0)],
        &[i(s as i32), i(n_exp as i32), i(top_k as i32), f(scale)],
        [s as u32, 1, 1], [blk, 1, 1], 0, false)?;
    dev.dispatch_raw_cuda(MOE_HIST_SRC, "moe_hist.cu", "moe_hist",
        &[(idx.buffer.as_ref(), 0), (counts.buffer.as_ref(), 0)],
        &[i(mt as i32)], [(mt as u32).div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
    dev.dispatch_raw_cuda(MOE_PREFIX_SRC, "moe_prefix.cu", "moe_prefix",
        &[(counts.buffer.as_ref(), 0), (offsets.buffer.as_ref(), 0), (cursor.buffer.as_ref(), 0)],
        &[i(n_exp as i32)], [1, 1, 1], [32, 1, 1], 0, false)?;
    let _ = &cursor;
    dev.dispatch_raw_cuda(MOE_SCATTER_SORT_SRC, "moe_scatter_sort.cu", "moe_scatter_sort_rowpos",
        &[(idx.buffer.as_ref(), 0), (wt.buffer.as_ref(), 0), (offsets.buffer.as_ref(), 0),
          (sorted_tok.buffer.as_ref(), 0), (sorted_wt.buffer.as_ref(), 0), (rowpos.buffer.as_ref(), 0)],
        &[i(mt as i32), i(top_k as i32), i(n_exp as i32)], [((n_exp * 32) as u32).div_ceil(128), 1, 1], [128, 1, 1], 0, false)?;
    Ok((sorted_tok, sorted_wt, offsets, rowpos, wt))
}

pub fn moe_weighted_sum(dev: &dyn Device, downs: &Tensor, wts: &Tensor, acc: &Tensor, hid: usize, top_k: usize) -> Result<()> {
    let k = cached_ir("iron_moe_weighted_sum", acc.dtype, || { let mut k = wh_iron_std::kernels::moe::moe_gather_q4::iron_moe_weighted_sum::kernel_ir_for(acc.dtype); k.mode = wh_iron_core::ir::KernelMode::Grid3D; k });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(&k, &[Binding::Buffer(downs.buffer.clone()), Binding::Buffer(wts.buffer.clone()), Binding::Buffer(acc.buffer.clone()), u(hid as u32), u(top_k as u32)], Grid { grid: [(hid as u32).div_ceil(256), 1, 1], block: [256, 1, 1] })?;
    Ok(())
}

// ── Q4 (4-bit) family, half the weight DRAM of Q8 (the decode lever). ──
fn gemv_q4_dispatch(dev: &dyn Device, kernel: &str, qs: &Tensor, scales: &Tensor, x: &Tensor, acc: Option<&Tensor>, scale_buf: Option<&Tensor>, m_out: usize, k_in: usize, rows_per_group: usize) -> Result<Tensor> {
    // 2-row tiling: 2 output rows per warp, shared activation read, 2 weight
    // streams in flight (more memory-level-parallelism on the latency-bound Q4 read).
    let vec = kernel == "plain" && std::env::var("BUTTER_GEMV_VEC").is_ok();
    let two_row = !vec && kernel == "plain" && std::env::var("BUTTER_GEMV_2ROW").is_ok();
    let name = if vec { "iron_gemv_q4_vec" } else if two_row { "iron_gemv_q4_coalesced_2row" } else { match kernel { "plain" => "iron_gemv_q4_coalesced", "relu2" => "iron_gemv_q4_coalesced_relu2", _ => "iron_gemv_q4_coalesced_accum" } };
    let k = cached_ir(name, x.dtype, || {
        let mut k = if vec {
            wh_iron_std::kernels::gemm::gemv_quantized::iron_gemv_q4_vec::kernel_ir_for(x.dtype)
        } else if two_row {
            wh_iron_std::kernels::gemm::gemv_quantized::iron_gemv_q4_coalesced_2row::kernel_ir_for(x.dtype)
        } else { match kernel {
            "plain" => wh_iron_std::kernels::gemm::gemv_quantized::iron_gemv_q4_coalesced::kernel_ir_for(x.dtype),
            "relu2" => wh_iron_std::kernels::gemm::gemv_quantized::iron_gemv_q4_coalesced_relu2::kernel_ir_for(x.dtype),
            _ => wh_iron_std::kernels::gemm::gemv_quantized::iron_gemv_q4_coalesced_accum::kernel_ir_for(x.dtype),
        }};
        k.mode = wh_iron_core::ir::KernelMode::Reduction;
        k
    });
    let out = acc.cloned().unwrap_or(Tensor::empty(dev, vec![m_out], x.dtype)?);
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let mut b = vec![Binding::Buffer(qs.buffer.clone()), Binding::Buffer(scales.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(out.buffer.clone())];
    if let Some(sb) = scale_buf { b.push(Binding::Buffer(sb.buffer.clone())); }
    b.extend([u(k_in as u32), u(m_out as u32), u(rows_per_group as u32)]);
    // Plain gemv is multi-warp capable (rows_per_tg warps/TG to hide the
    // ncu-measured global-load latency). BUTTER_GEMV_RPT=1 → original single-warp.
    let grid = if vec {
        // single-warp/row, vectorized loads (no rpt binding)
        Grid { grid: [m_out as u32, 1, 1], block: [32, 1, 1] }
    } else if kernel == "plain" {
        let rpt: u32 = std::env::var("BUTTER_GEMV_RPT").ok().and_then(|v| v.parse().ok()).filter(|&r| r >= 1).unwrap_or(1);
        b.push(u(rpt));
        let rows_per_block = if two_row { 2 * rpt } else { rpt };
        Grid { grid: [(m_out as u32).div_ceil(rows_per_block), 1, 1], block: [32 * rpt, 1, 1] }
    } else {
        // relu2 / accum: multi-warp capable (rows_per_tg warps/TG). The shared-
        // expert matrices (3712×2688, 2688×3712) are small; rpt>1 gives no
        // measured gain (these kernels are NOT latency-bound, the ~50% BW the
        // sync-bracketed profiler showed is a measurement artifact, the kernel
        // time is small). Default rpt=1 (bit-identical to the original single-
        // warp form); BUTTER_GEMV_SHARED_RPT overrides for experimentation.
        let rpt: u32 = std::env::var("BUTTER_GEMV_SHARED_RPT").ok().and_then(|v| v.parse().ok()).filter(|&r| r >= 1).unwrap_or(1);
        b.push(u(rpt));
        Grid { grid: [(m_out as u32).div_ceil(rpt), 1, 1], block: [32 * rpt, 1, 1] }
    };
    dev.dispatch(&k, &b, grid)?;
    Ok(out)
}
/// Q4 matvec `out = Wq4·x`. Block 32, int4 [-7,7], ONE F16 SCALE PER BLOCK:
/// the coalesced/relu2/accum kernels read `scales` as f16, so convert the f32
/// scales from [`quantize_q4`] with an f32-to-f16 pass before upload (raw f32
/// bytes get silently reinterpreted as garbage f16 and inflate every output).
pub fn gemv_q4(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, m_out: usize, k_in: usize, rpg: usize) -> Result<Tensor> {
    gemv_q4_dispatch(dev, "plain", qs, scales, x, None, None, m_out, k_in, rpg)
}
/// Q4 matvec with fused ReLU² (MoE expert up).
pub fn gemv_q4_relu2(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, m_out: usize, k_in: usize, rpg: usize) -> Result<Tensor> {
    gemv_q4_dispatch(dev, "relu2", qs, scales, x, None, None, m_out, k_in, rpg)
}
/// Q4 matvec, scale + accumulate into `acc` in place (MoE expert down).
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_accum(dev: &dyn Device, qs: &Tensor, scales: &Tensor, x: &Tensor, acc: &Tensor, scale_buf: &Tensor, m_out: usize, k_in: usize, rpg: usize) -> Result<()> {
    gemv_q4_dispatch(dev, "accum", qs, scales, x, Some(acc), Some(scale_buf), m_out, k_in, rpg)?;
    Ok(())
}
/// Quantize a row-major `[m,k]` F32 weight to Q4 blocks (block 32, symmetric int4
/// in [-7,7], f32 scale `amax/7`). Returns `(qs_u32, scales)`: qs `[m·k/32·4]`
/// (8 nibbles/u32, 4 u32/block), scales `[m·k/32]`. `k` must be a multiple of 32.
/// NOTE: the returned scales are host f32; the `gemv_q4`/`gemv_q4_relu2`/
/// `gemv_q4_accum` device kernels expect them uploaded as F16 (see the
/// `gemv_q4` doc). Only `MT_GEMV_VEC`'s vectorized kernel takes f32 scales.
pub fn quantize_q4(w: &[f32], m: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(k % 32, 0, "quantize_q4: k must be a multiple of 32");
    let bpr = k / 32;
    let mut qs = vec![0u32; m * bpr * 4];
    let mut scales = vec![0f32; m * bpr];
    for r in 0..m {
        for b in 0..bpr {
            let base = r * k + b * 32;
            let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
            let d = amax / 7.0;
            scales[r * bpr + b] = d;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for word in 0..4 {
                let mut packed = 0u32;
                for i in 0..8 {
                    let q = (w[base + word * 8 + i] * inv).round().clamp(-7.0, 7.0) as i32;
                    packed |= ((q as u32) & 0xf) << (i * 4);
                }
                qs[r * bpr * 4 + b * 4 + word] = packed;
            }
        }
    }
    (qs, scales)
}

/// Quantize a row-major `[m, k]` F32 weight to Q8_0 blocks (block 32, symmetric
/// int8, per-block f32 scale `amax/127`). Returns `(qs_u32_packed, scales_f32)`
/// laid out exactly as [`gemv_q8`] expects: qs `[m · k/32 · 8]` u32, scales
/// `[m · k/32]` f32. CPU, one-time at load. `k` must be a multiple of 32.
pub fn quantize_q8(w: &[f32], m: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(k % 32, 0, "quantize_q8: k must be a multiple of 32");
    let bpr = k / 32;
    let mut qs = vec![0u32; m * bpr * 8];
    let mut scales = vec![0f32; m * bpr];
    for r in 0..m {
        for b in 0..bpr {
            let base = r * k + b * 32;
            let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
            let d = amax / 127.0;
            scales[r * bpr + b] = d;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for w_i in 0..8 {
                let mut packed = 0u32;
                for i in 0..4 {
                    let q = (w[base + w_i * 4 + i] * inv).round().clamp(-127.0, 127.0) as i32;
                    packed |= ((q as u8) as u32) << (i * 8);
                }
                qs[r * bpr * 8 + b * 8 + w_i] = packed;
            }
        }
    }
    (qs, scales)
}

/// e2m1 (FP4) codebook magnitudes (index 0..7); sign bit is bit 3.
const E2M1_LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Encode a value `x` (already divided by its block scale) to a 4-bit e2m1 code
/// (sign<<3 | nearest-magnitude-index). Round-to-nearest over the 8-entry codebook.
#[inline]
fn enc_e2m1(v: f32) -> u8 {
    let sign = if v < 0.0 { 8u8 } else { 0u8 };
    let a = v.abs();
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for (i, &m) in E2M1_LUT.iter().enumerate() {
        let d = (a - m).abs();
        if d < bd { bd = d; best = i; }
    }
    sign | best as u8
}

/// Encode a non-negative scale `s` to a UE4M3 byte (4 exp bits bias-7, 3 mantissa,
/// no sign, NVFP4's micro-scale type). Round-to-nearest over representable values.
#[inline]
fn enc_ue4m3(s: f32) -> u8 {
    if s <= 0.0 { return 0; }
    let (mut be, mut bm, mut bd) = (0i32, 0i32, f32::INFINITY);
    for e in 0..16i32 {
        for m in 0..8i32 {
            let v = if e == 0 { (m as f32 / 8.0) * 0.007_812_5 }
                    else { (1.0 + m as f32 / 8.0) * (e as f32 - 7.0).exp2() };
            let d = (s - v).abs();
            if d < bd { bd = d; be = e; bm = m; }
        }
    }
    ((be << 3) | bm) as u8
}

/// Decode a UE4M3 byte back to f32 (for reconstruction / host reference).
#[inline]
pub fn dec_ue4m3(b: u8) -> f32 {
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as i32;
    if e == 0 { (m as f32 / 8.0) * 0.007_812_5 } else { (1.0 + m as f32 / 8.0) * (e as f32 - 7.0).exp2() }
}

/// Quantize a row-major `[m, k]` F32 weight to **NVFP4** (NVIDIA FP4): E2M1 4-bit
/// codes + per-**block-16** UE4M3 micro-scale + one per-tensor FP32 global. This is
/// the format the native Blackwell `mma.sync.kind::mxf4nvf4` tensor-core path
/// consumes (validated bit-exact on GB10/sm_121a). Single quantization from the
/// original full-precision weights (no Q4→FP4 double-quant). `k % 16 == 0`.
///
/// Returns `(codes, scales, global)`:
/// - `codes`: `[m · k/8]` u32, 8 e2m1 nibbles per u32 (nibble j of word w = element
///   `w*8 + j` of the row), row-major: same packing convention as [`quantize_q4`].
/// - `scales`: `[m · k/16]` UE4M3 bytes (one per 16-element K-block, row-major).
/// - `global`: per-tensor f32; the true block scale is `dec_ue4m3(scale) * global`.
///   Chosen so each block's `amax/6` lands in UE4M3's representable range.
///
/// Example: `let (codes, scales, g) = quantize_nvfp4(&w, n_rows, k_in);`
#[inline]
fn marlin_f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32 - 112; // 127 - 15
    if e <= 0 { return sign; }
    if e >= 0x1f { return sign | 0x7c00; }
    let m = (x >> 13) & 0x3ff;
    let round = (x >> 12) & 1;
    let v = ((e as u32) << 10) | m;
    sign | ((v + round) as u16)
}

/// Convert per-16 NVFP4 block-scale VALUES `[ngroup, n]` (row-major; value =
/// dec_ue4m3(byte)*global) into the marlin S0E5M3 fp8 byte layout the GEMM reads.
/// Mirrors marlin_permute_scales(group<K) + nvfp4_marlin_process_scales (f16,
/// scale_factor=1): scale_perm 64-permute, [0,2,1,3] 4-interleave, x2^7, zero-if<2,
/// reinterpret f16 bits `<<1`, take the high byte (= S0E5M3). Returns `[ngroup*n]` u8.
pub fn marlin_s0e5m3_scales(vals: &[f32], ngroup: usize, n: usize) -> Vec<u8> {
    let total = ngroup * n;
    assert_eq!(vals.len(), total);
    assert_eq!(total % 64, 0, "marlin_s0e5m3_scales: ngroup*n must be a multiple of 64");
    // scale_perm (64): [i + 8*j for i in 0..8 for j in 0..8]
    let mut sp = [0usize; 64];
    let mut idx = 0;
    for i in 0..8 { for j in 0..8 { sp[idx] = i + 8 * j; idx += 1; } }
    // A: 64-chunk column permute (marlin_permute_scales)
    let mut s1 = vec![0f32; total];
    for r in 0..(total / 64) {
        for j in 0..64 { s1[r * 64 + j] = vals[r * 64 + sp[j]]; }
    }
    // B: 4-chunk interleave [0,2,1,3]
    let il = [0usize, 2, 1, 3];
    let mut s2 = vec![0f32; total];
    for r in 0..(total / 4) {
        for j in 0..4 { s2[r * 4 + j] = s1[r * 4 + il[j]]; }
    }
    // C: x2^7, zero-if<2, f16 bits <<1, high byte = S0E5M3
    let mut out = vec![0u8; total];
    for p in 0..total {
        let mut v = s2[p] * 128.0;
        if v < 2.0 { v = 0.0; }
        let shifted = marlin_f32_to_f16(v) << 1;
        out[p] = (shifted >> 8) as u8;
    }
    out
}

pub fn quantize_nvfp4(w: &[f32], m: usize, k: usize) -> (Vec<u32>, Vec<u8>, f32) {
    assert_eq!(k % 16, 0, "quantize_nvfp4: k must be a multiple of 16");
    let nblk = k / 16;
    // Per-tensor global: makes per-block (amax/6) fit UE4M3's ~448 max.
    let gmax = w.iter().fold(0f32, |a, &x| a.max(x.abs()));
    let global = if gmax > 0.0 { gmax / 6.0 / 448.0 } else { 1.0 };
    let mut codes = vec![0u32; m * (k / 8)];
    let mut scales = vec![0u8; m * nblk];
    for r in 0..m {
        for b in 0..nblk {
            let base = r * k + b * 16;
            let amax = (0..16).fold(0f32, |a, i| a.max(w[base + i].abs()));
            let sb = enc_ue4m3((amax / 6.0) / global);
            scales[r * nblk + b] = sb;
            let bs = dec_ue4m3(sb) * global; // reconstructed block scale
            let inv = if bs > 0.0 { 1.0 / bs } else { 0.0 };
            for i in 0..16 {
                let code = enc_e2m1(w[base + i] * inv) as u32;
                let elem = b * 16 + i; // index within the row
                let word = elem / 8;
                let nib = elem % 8;
                codes[r * (k / 8) + word] |= code << (nib * 4);
            }
        }
    }
    (codes, scales, global)
}

/// Dequantize a [`quantize_nvfp4`] tensor back to row-major `[m, k]` f32 (for
/// host reference / FP4-sim round-trips). `codes` `[m·k/8]` u32, `scales`
/// `[m·k/16]` UE4M3 bytes, `global` f32.
pub fn dequantize_nvfp4(codes: &[u32], scales: &[u8], global: f32, m: usize, k: usize) -> Vec<f32> {
    let nblk = k / 16;
    let mut out = vec![0f32; m * k];
    for r in 0..m {
        for b in 0..nblk {
            let bs = dec_ue4m3(scales[r * nblk + b]) * global;
            for i in 0..16 {
                let elem = b * 16 + i;
                let word = elem / 8;
                let nib = elem % 8;
                let code = (codes[r * (k / 8) + word] >> (nib * 4)) & 0xf;
                let sign = if code & 8 != 0 { -1.0 } else { 1.0 };
                out[r * k + elem] = sign * E2M1_LUT[(code & 7) as usize] * bs;
            }
        }
    }
    out
}

/// Round-trip a row-major `[m, k]` f32 buffer through NVFP4 quantize→dequantize,
/// i.e. inject exactly the E2M1+UE4M3 block-16 quantization noise the native FP4
/// GEMM would see, without any kernel. Used to de-risk argmax before building the
/// real FP4 path (`NEMOTRON_FP4_SIM`).
pub fn nvfp4_roundtrip(w: &[f32], m: usize, k: usize) -> Vec<f32> {
    let (codes, scales, global) = quantize_nvfp4(w, m, k);
    dequantize_nvfp4(&codes, &scales, global, m, k)
}

/// Prefill linear: `out[r, :] = weight · input[r, :]` for a block of rows in
/// one dispatch, `weight` is `[out_dim, in_dim]`, `input` is `[n_rows, in_dim]`,
/// result `[n_rows, out_dim]`. Dispatches the 32×32-tiled Reduction kernel
/// `iron_gemm` (weight read once per tile, reused across the row block — the
/// many-token path that a `gemv`-per-row would re-stream). Requires
/// `in_dim % 16 == 0` (the K-tile contract); row/out-dim edges are handled
/// in-kernel. This is the matmul the prefill + VLM/ViT towers run on.
pub fn matmul(dev: &dyn Device, weight: &Tensor, input: &Tensor) -> Result<Tensor> {
    if weight.shape.len() != 2 {
        return Err(Error::Msg(format!("matmul: weight must be 2-D, got {:?}", weight.shape)));
    }
    let (out_dim, in_dim) = (weight.shape[0], weight.shape[1]);
    let rows = input.elem_count() / in_dim;
    if input.elem_count() != rows * in_dim {
        return Err(Error::Msg(format!("matmul: input {:?} not a multiple of in_dim {in_dim}", input.shape)));
    }
    if in_dim % 16 != 0 {
        return Err(Error::Msg(format!("matmul: in_dim {in_dim} must be a multiple of 16")));
    }
    let k = lookup("iron_gemm", weight.dtype)?;
    let out = Tensor::empty(dev, vec![rows, out_dim], weight.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid {
        grid: [(out_dim as u32).div_ceil(32), (rows as u32).div_ceil(32), 1],
        block: [1024, 1, 1],
    };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(weight.buffer.clone()),
            Binding::Buffer(input.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(in_dim as u32),
            u(out_dim as u32),
            u(rows as u32),
        ],
        grid,
    )?;
    Ok(out)
}

/// Tensor-core GEMM via cuBLAS (Path A escape hatch). Computes the row-major
/// `out[m,n] = x[m,k] · w[n,k]ᵀ`, the projection `out[r,o] = Σ_k w[o,k]·x[r,k]`
/// where `w` is the DENSE `[n,k]` weight (f16/bf16, NOT quantized) and `x` is
/// `[m,k]`. Runs on the GB10 tensor cores (f32 accumulate). The caller dequants
/// the Q4 weight to f16/bf16 once, then this hits real TFLOP/s (vs the
/// coop_tile software emulation at ~0.1% of peak).
pub fn gemm_cublas(
    dev: &dyn Device,
    x: &Tensor,
    w: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor> {
    if x.dtype != w.dtype {
        return Err(Error::Msg(format!("gemm_cublas: x/w dtype mismatch {:?} vs {:?}", x.dtype, w.dtype)));
    }
    let out = Tensor::empty(dev, vec![m, n], x.dtype)?;
    dev.gemm_tc(x.buffer.as_ref(), w.buffer.as_ref(), out.buffer.as_ref(), m, n, k, x.dtype)?;
    Ok(out)
}

/// Same as [], but writes an **f32** output, fusing the post-GEMM
/// `cast_f16_f32` into the matmul. `x`/`w` are f16/bf16; cuBLAS already
/// accumulates in f32, so the only change is the D-layout type → no extra
/// kernel, no MFU loss. Lets the model keep the residual stream f32 (required
///, f16/bf16 residual overflows/loses argmax) without a separate cast pass.
/// CUDA-only; other backends error (model gates this on `is_cuda`).
pub fn gemm_cublas_f32out(
    dev: &dyn Device,
    x: &Tensor,
    w: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor> {
    if x.dtype != w.dtype {
        return Err(Error::Msg(format!("gemm_cublas_f32out: x/w dtype mismatch {:?} vs {:?}", x.dtype, w.dtype)));
    }
    // A/B lever: BUTTER_F32OUT_FALLBACK=1 reverts to the unfused path (f16-out GEMM
    // + a separate `cast_f16_f32` kernel) so the fused vs unfused cast can be timed
    // back-to-back under the same clock. Default = fused (no extra kernel).
    if std::env::var("BUTTER_F32OUT_FALLBACK").is_ok() {
        let oh = gemm_cublas(dev, x, w, m, n, k)?;
        return cast_f16_f32(dev, &oh);
    }
    let out = Tensor::empty(dev, vec![m, n], DType::F32)?;
    dev.gemm_tc_out_f32(x.buffer.as_ref(), w.buffer.as_ref(), out.buffer.as_ref(), m, n, k, x.dtype)?;
    Ok(out)
}

/// Dense Q4 GEMM via cooperative-tensor MMA (`iron_gemm_q4_mpp`). Weight
/// `[out_dim, k_in]` is the bench's Q4 layout (qs u32 4-words/block, scales
/// f16 amax/7); `x` is `[n_rows, k_in]` of dtype T; out is `[n_rows, out_dim]`
/// of T. The tensor-core projection GEMM for prefill (replaces the f32 scalar
/// `matmul` that sat at ~0.1% of peak). Reduction-mode, grid
/// `[ceil(out/64), ceil(rows/64), 1]`, block 128. `k_in % 32 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_mpp(
    dev: &dyn Device,
    x: &Tensor,
    qs: &Tensor,
    scales: &Tensor,
    n_rows: usize,
    out_dim: usize,
    k_in: usize,
) -> Result<Tensor> {
    use wh_iron_core::ir::KernelMode;
    if k_in % 32 != 0 {
        return Err(Error::Msg(format!("gemm_q4_mpp: k_in {k_in} must be a multiple of 32")));
    }
    let out = Tensor::empty(dev, vec![n_rows, out_dim], x.dtype)?;
    let kern = cached_ir("iron_gemm_q4_mpp", x.dtype, || {
        let mut kk = wh_iron_std::kernels::gemm::gemm_q4_mpp::iron_gemm_q4_mpp::kernel_ir_for(x.dtype);
        kk.mode = KernelMode::Reduction;
        kk
    });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid {
        grid: [(out_dim as u32).div_ceil(64), (n_rows as u32).div_ceil(64), 1],
        block: [128, 1, 1],
    };
    dev.dispatch(
        &kern,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(n_rows as u32),
            u(out_dim as u32),
            u(k_in as u32),
        ],
        grid,
    )?;
    Ok(out)
}

/// Routed-expert MoE Q4 grouped BGEMM via MMA (`iron_moe_bgemm_q4_bm64`). The
/// token rows of `x` `[m_total, k_in]` MUST be pre-sorted by expert, with
/// `indices` `[m_total]` giving each row's expert id. Expert weights are the
/// contiguous Q4 pool `[n_experts*n_out, k_in]` (qs u32 + scales f16). Output
/// `[m_total, n_out]` (sorted order; host scatters back to token order). The
/// batched-over-S replacement for the per-token MoE gather loop. Reduction-mode,
/// grid `[n_out/64, ceil(m_total/64), 1]`, block 128. `n_out % 64 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn moe_bgemm_q4_bm64(
    dev: &dyn Device,
    x: &Tensor,
    qs: &Tensor,
    scales: &Tensor,
    indices: &Tensor,
    m_total: usize,
    n_out: usize,
    k_in: usize,
) -> Result<Tensor> {
    use wh_iron_core::ir::KernelMode;
    if n_out % 64 != 0 {
        return Err(Error::Msg(format!("moe_bgemm_q4_bm64: n_out {n_out} must be a multiple of 64")));
    }
    let out = Tensor::empty(dev, vec![m_total, n_out], x.dtype)?;
    let kern = cached_ir("iron_moe_bgemm_q4_bm64", x.dtype, || {
        let mut kk = wh_iron_std::kernels::moe::moe_bgemm_q4_bm64::iron_moe_bgemm_q4_bm64::kernel_ir_for(x.dtype);
        kk.mode = KernelMode::Reduction;
        kk
    });
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let grid = Grid {
        grid: [(n_out as u32) / 64, (m_total as u32).div_ceil(64), 1],
        block: [128, 1, 1],
    };
    dev.dispatch(
        &kern,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(indices.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(m_total as u32),
            u(n_out as u32),
            u(k_in as u32),
        ],
        grid,
    )?;
    Ok(out)
}

/// Q4 → dense `[m, k]` dequant. Expands a resident Q4 weight (the bench's
/// `quantize_q4` layout: `qs [m·(k/32)·4]` u32, signed nibbles; `scales
/// [m·(k/32)]` f32, amax/7) into a dense slab of `out_dtype` for the
/// compute-bound prefill GEMM (dequant-once → tensor-core `iron_gemm`).
/// 1-D grid, one thread per output value.
pub fn dequant_q4(
    dev: &dyn Device,
    qs: &Tensor,
    scales: &Tensor,
    m: usize,
    k: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    dequant_q4_off(dev, qs, scales, m, k, out_dtype, 0)
}

/// Like [`dequant_q4`] but dequants the `[m,k]` slab starting at block offset
/// `blk_off` (in 32-value Q4 blocks) inside the qs/scales pool, used to peel
/// one MoE expert's rows out of the contiguous `[n_exp*out, k]` Q4 pool without
/// a tensor view. `blk_off = expert * m * (k/32)`.
#[allow(clippy::too_many_arguments)]
pub fn dequant_q4_off(
    dev: &dyn Device,
    qs: &Tensor,
    scales: &Tensor,
    m: usize,
    k: usize,
    out_dtype: DType,
    blk_off: usize,
) -> Result<Tensor> {
    use wh_iron_core::ir::KernelMode;
    let n = m * k;
    // Defensive bounds check. The kernel addresses sub-slabs purely via blk_off
    // against the RAW bound buffers, reading up to qs[(blk_off + m*bpr - 1)*4 + 3]
    // and scales[blk_off + m*bpr - 1]. If a caller binds a buffer that doesn't
    // actually hold blk_off+slab blocks (an under-sized scratch / per-expert pool),
    // that is an out-of-bounds READ, silent garbage on Metal, a deterministic
    // MMU-fault (Xid 31 VIRT_READ) on CUDA. Fail loudly here, naming the shapes,
    // so the mis-sized caller is obvious on every backend.
    let bpr = k / 32;
    let need_blocks = blk_off + m * bpr;
    let have_qs_blocks = qs.buffer.len() / 16; // 4 u32 words/block * 4 B/word
    let have_sc_blocks = scales.buffer.len() / 2; // f16 scale/block
    if need_blocks > have_qs_blocks || need_blocks > have_sc_blocks {
        return Err(Error::Msg(format!(
            "dequant_q4_off OOB: m={m} k={k} blk_off={blk_off} needs {need_blocks} Q4 blocks \
             but bound buffers hold qs={have_qs_blocks} scales={have_sc_blocks} blocks"
        )));
    }
    let out = Tensor::empty(dev, vec![m, k], out_dtype)?;
    let kern = cached_ir("iron_dequant_q4", out_dtype, || {
        let mut kk = wh_iron_std::kernels::quant::dequant_q4::iron_dequant_q4::kernel_ir_for(out_dtype);
        kk.mode = KernelMode::Grid3D;
        kk
    });
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    // 1 thread per u32 word → 8 outputs. Grid is over words (n/8), not values:
    // amortises the qs/scale global traffic 8×. n is always a multiple of 8
    // (k is a multiple of 32 → 4 words/block, 8 values/word).
    let n_words = (n / 8) as u32;
    let grid = Grid::d1(n_words.div_ceil(256), 256);
    dev.dispatch(
        &kern,
        &[
            Binding::Buffer(qs.buffer.clone()),
            Binding::Buffer(scales.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(k as u32),
            u(n_words),
            u(blk_off as u32),
        ],
        grid,
    )?;
    Ok(out)
}

/// Multi-query (prefill) SDPA, attends `n_query` query rows against a shared
/// K/V cache in one dispatch. `causal=1` → query `r` attends `[0, base_kv+r+1)`
/// (causal within the block, prefix fully visible); `causal=0` → bidirectional
/// over `[0, base_kv+n_query)`. Q/out `[n_query, n_q_heads, head_dim]`, K/V
/// `[n_kv_heads, kv_stride, head_dim]`. head_dim hard-128. This is the prefill
/// flash-attn (`iron_sdpa_multi`), Reduction-mode, grid `[n_q_heads*n_query]`,
/// block 1024.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_multi(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_q_heads: u32,
    base_kv: u32,
    n_query: u32,
    kv_stride: u32,
    heads_per_group: u32,
    causal: bool,
    scale: f32,
) -> Result<Tensor> {
    use wh_iron_core::ir::KernelMode;
    if head_dim != 128 {
        return Err(Error::Msg(format!("sdpa_multi: head_dim must be 128, got {head_dim}")));
    }
    let out = Tensor::empty(dev, vec![n_query as usize, n_q_heads as usize, head_dim], q.dtype)?;
    let kern = cached_ir("iron_sdpa_multi", q.dtype, || {
        let mut kk = wh_iron_std::kernels::sdpa::sdpa_multi::iron_sdpa_multi::kernel_ir_for(q.dtype);
        kk.mode = KernelMode::Reduction;
        kk
    });
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid { grid: [n_q_heads * n_query, 1, 1], block: [1024, 1, 1] };
    dev.dispatch(
        &kern,
        &[
            Binding::Buffer(q.buffer.clone()),
            Binding::Buffer(k.buffer.clone()),
            Binding::Buffer(v.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(head_dim as u32),
            u(n_q_heads),
            u(base_kv),
            u(n_query),
            u(kv_stride),
            u(heads_per_group),
            u(u32::from(causal)),
            f(scale),
        ],
        grid,
    )?;
    Ok(out)
}

// ─── Tensor-core prefill SDPA (`sdpa_multi_tc`) ──────────────────────────────
//
// Drop-in replacement for [`sdpa_multi`] that runs the two big matmuls (QKᵀ and
// P·V) on the GB10 **tensor cores via cuBLAS** instead of the software-MMA
// `iron_sdpa_multi` kernel (which sits at ~0.1% of peak and dominated the
// long-context prefill). It is a FlashAttention-style online-softmax loop tiled
// over KV blocks so the score matrix is never materialised in full:
//
//   for each KV block:
//     S_blk = Qh · Khᵀ            (cublasGemmStridedBatchedEx, fp16→fp32→fp16)
//     P_blk = exp(S_blk - m_blk)  + causal mask  (custom kernel; emits m_blk,l_blk)
//     O_blk = P_blk · Vt          (cublasGemmStridedBatchedEx)
//     merge O_blk into running O with per-row rescale; update m_run, l_run
//   O = O_run / l_run, transpose head-major → row-major, cast back to caller dtype
//
// GQA fan-out (16 query heads : 1 KV head) is handled by issuing the batched
// GEMM **once per KV group** with the K/V stride set to 0 (the 16 query heads in
// a group all read the same KV slab, no duplicated KV is materialised).
//
// Q is pre-scaled by `scale` in the transpose kernel, so the GEMM/softmax run on
// already-scaled logits (matches `iron_sdpa_multi`, which folds `scale` into Q).

/// Grouped MoE GEMM via mma.sync (register-O), ONE launch for ALL experts.
/// `out[t,n] = Σ_k A[t,k]·W[grp(t)][n,k]` over sorted tokens (each contiguous
/// token range = one expert). Reads the RESIDENT f16 expert weights directly via
/// a device pointer array, no f16 scratch (that's what sank cuBLAS-grouped),
/// and runs every expert's output tiles concurrently to FILL the GPU (the skinny
/// m≈96 per-expert cuBLAS GEMMs hit only ~15 CTAs → 1.5% MFU). Same m16n8k16
/// register-O technique + fragment packing as `sdpa_flash_mma`'s QKᵀ. 1 warp /
/// (16-token × 64-col) tile; grid (n_mtiles, ceil(N/64)). f16 in/out.
const MOE_GROUPED_GEMM_MMA_SRC: &str = r#"
#include <cuda_fp16.h>
__device__ __forceinline__ unsigned pk2(__half a, __half b){
    return (unsigned)__half_as_ushort(a) | ((unsigned)__half_as_ushort(b) << 16);
}
__device__ __forceinline__ unsigned ld32(const __half* p){ return *reinterpret_cast<const unsigned*>(p); }
#define CPA(SD, SP, V) asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n" \
    :: "r"(SD), "l"(SP), "r"((V)?16:0))
// BM=64, BN=128: each block computes 128 N-cols so the loaded A tile is reused
// across 2x the weights → halves redundant A reads (the real bottleneck: A was
// re-read once per 64-wide N-tile, N/64=29 times/expert > weight traffic).
extern "C" __global__ void moe_grouped_gemm_mma(
    const __half* __restrict__ A,
    const __half* __restrict__ W_base,
    const int* __restrict__ tok0_arr,
    const int* __restrict__ eid_arr,
    const int* __restrict__ gend_arr,
    __half* __restrict__ out,
    int N, int K)
{
    const int BM=64, BN=128, BK=64, NS=16, NKS=4, U=BK/8;
    int bx = blockIdx.x, by = blockIdx.y;
    int tok0 = tok0_arr[bx], gend = gend_arr[bx], n0 = by * BN;
    const __half* W = W_base + (long)eid_arr[bx] * N * K;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tid4 = lane & 3;
    int wrow = warp * 16;

    __shared__ __half As[2][BM][BK];
    __shared__ __half Ws[2][BN][BK];
    float oc[NS][4];
    for (int s=0;s<NS;s++){ oc[s][0]=oc[s][1]=oc[s][2]=oc[s][3]=0.f; }
    int nchunk = K / BK;

#define LOAD_CHUNK(C, BUF) do { int kc=(C)*BK; \
    for (int e=tid; e<BM*U; e+=128){ int r=e/U, k8=(e%U)*8; int t=tok0+r; int v=(t<gend); \
        unsigned sd=(unsigned)__cvta_generic_to_shared(&As[BUF][r][k8]); \
        const __half* sp=&A[(long)(v?t:tok0)*K + kc + k8]; CPA(sd, sp, v); } \
    for (int e=tid; e<BN*U; e+=128){ int r=e/U, k8=(e%U)*8; int nn=n0+r; int v=(nn<N); \
        unsigned sd=(unsigned)__cvta_generic_to_shared(&Ws[BUF][r][k8]); \
        const __half* sp=&W[(long)(v?nn:n0)*K + kc + k8]; CPA(sd, sp, v); } \
    } while(0)

    LOAD_CHUNK(0, 0); asm volatile("cp.async.commit_group;\n");
    for (int c=0;c<nchunk;c++){
        if (c+1<nchunk){ LOAD_CHUNK(c+1, (c+1)&1); asm volatile("cp.async.commit_group;\n");
                         asm volatile("cp.async.wait_group 1;\n"); }
        else { asm volatile("cp.async.wait_group 0;\n"); }
        __syncthreads();
        int buf = c&1;
        for (int ks=0; ks<NKS; ks++){ int ko=ks*16;
            unsigned a0=ld32(&As[buf][wrow+gid][ko+2*tid4]);
            unsigned a1=ld32(&As[buf][wrow+gid+8][ko+2*tid4]);
            unsigned a2=ld32(&As[buf][wrow+gid][ko+2*tid4+8]);
            unsigned a3=ld32(&As[buf][wrow+gid+8][ko+2*tid4+8]);
            #pragma unroll
            for (int s=0;s<NS;s++){ int nb=s*8;
                unsigned b0=ld32(&Ws[buf][nb+gid][ko+2*tid4]);
                unsigned b1=ld32(&Ws[buf][nb+gid][ko+2*tid4+8]);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    :"+f"(oc[s][0]),"+f"(oc[s][1]),"+f"(oc[s][2]),"+f"(oc[s][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        __syncthreads();
    }
    int r0=tok0+wrow+gid, r8=tok0+wrow+gid+8;
    for (int s=0;s<NS;s++){ int col=n0+s*8+2*tid4;
        if (r0<gend && col<N){ out[(long)r0*N+col]=__float2half(oc[s][0]); out[(long)r0*N+col+1]=__float2half(oc[s][1]); }
        if (r8<gend && col<N){ out[(long)r8*N+col]=__float2half(oc[s][2]); out[(long)r8*N+col+1]=__float2half(oc[s][3]); }
    }
}
"#;

/// CUTLASS grouped MoE GEMM, `out[t,n] = Σ_k A[t,k]·W[eid][n,k]` over sorted
/// token groups. `a` f16 `[mt,K]`; `w_base` contiguous f16 expert slab
/// `[n_exp,N,K]`; `g_starts[g]..g_starts[g+1]` = group g's token range;
/// `expert_ids[g]` selects the slab. Beats cuBLAS/the hand-rolled kernel on the
/// skinny MoE shape. Returns `out[mt,N]` f16. CUDA + CUTLASS_DIR build only.
pub fn moe_grouped_gemm_cutlass(
    dev: &dyn Device,
    a: &Tensor,
    w_base: &Tensor,
    g_starts: &[usize],
    expert_ids: &[usize],
    n: usize,
    k: usize,
) -> Result<Tensor> {
    let n_active = expert_ids.len();
    let mt = g_starts.last().copied().unwrap_or(0);
    let out = Tensor::empty(dev, vec![mt.max(1), n], DType::F16)?;
    if n_active == 0 || mt == 0 { return Ok(out); }
    let group_rows: Vec<i32> = (0..n_active).map(|g| (g_starts[g + 1] - g_starts[g]) as i32).collect();
    let eids: Vec<i32> = expert_ids.iter().map(|&e| e as i32).collect();
    dev.moe_grouped_cutlass(a.buffer.as_ref(), w_base.buffer.as_ref(), out.buffer.as_ref(), &group_rows, &eids, n, k)?;
    Ok(out)
}

/// Grouped MoE GEMM (mma.sync register-O), `out[t,n] = Σ_k A[t,k]·W[eid][n,k]`
/// for sorted tokens. `a` is `[mt,K]` f16; `w_base` is the contiguous f16 expert
/// weight slab `[n_exp,N,K]`; `g_starts[g]..g_starts[g+1]` is group g's token
/// range; `expert_ids[g]` selects the weight slab. ONE launch, all experts'
/// tiles concurrent (fills the GPU), reading resident f16 weights directly (no
/// scratch). Returns `out[mt,N]` f16. K%16==0, N%8==0.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gemm_mma(
    dev: &dyn Device,
    a: &Tensor,
    w_base: &Tensor,
    g_starts: &[usize],
    expert_ids: &[usize],
    n: usize,
    k: usize,
) -> Result<Tensor> {
    let n_active = expert_ids.len();
    let mt = g_starts.last().copied().unwrap_or(0);
    if k % 64 != 0 { return Err(Error::Msg(format!("moe_grouped_gemm_mma: K {k} %64!=0"))); }
    let out = Tensor::empty(dev, vec![mt.max(1), n], DType::F16)?;
    if n_active == 0 || mt == 0 { return Ok(out); }
    // Build per-m-tile descriptors (BM=16 token rows each).
    let (mut tok0, mut eid, mut gend) = (Vec::new(), Vec::new(), Vec::new());
    for g in 0..n_active {
        let (s, e) = (g_starts[g], g_starts[g + 1]);
        let mut t = s;
        while t < e { tok0.push(t as i32); eid.push(expert_ids[g] as i32); gend.push(e as i32); t += 64; }
    }
    let n_mtiles = tok0.len();
    let up_i32 = |v: &[i32]| -> Result<Tensor> {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Ok(Tensor::new(dev.upload(&bytes).map_err(|e| Error::Msg(format!("{e:?}")))?, vec![v.len()], DType::U32))
    };
    let (t0d, eidd, ged) = (up_i32(&tok0)?, up_i32(&eid)?, up_i32(&gend)?);
    let i = |x: i32| x.to_le_bytes().to_vec();
    let grid = [n_mtiles as u32, (n as u32).div_ceil(128), 1];
    dev.dispatch_raw_cuda(
        MOE_GROUPED_GEMM_MMA_SRC, "moe_grouped_gemm_mma.cu", "moe_grouped_gemm_mma",
        &[(a.buffer.as_ref(), a.offset), (w_base.buffer.as_ref(), w_base.offset),
          (t0d.buffer.as_ref(), 0), (eidd.buffer.as_ref(), 0), (ged.buffer.as_ref(), 0),
          (out.buffer.as_ref(), 0)],
        &[i(n as i32), i(k as i32)],
        grid, [128, 1, 1], 0, false)?;
    Ok(out)
}

// Q4-NATIVE grouped MoE GEMM (Marlin-style). Like moe_grouped_gemm_mma but reads
// the model's SIGNED Q4 weights directly (NO f16 slab): cp.async packs int4 into
// shared u32 staging + f16 scales, register-dequants per b-fragment in the mma
// m16n8k16 loop. dqn = sign_extend((nib^8)-8) * f16scale (matches quantize_q4).
// Qs [n_exp*N · K/32 · 4] u32, Sc [n_exp*N · K/32] f16 (= moe_up_all/down_all).
const MOE_Q4_GROUPED_MMA_SRC: &str = r#"
#include <cuda_fp16.h>
__device__ __forceinline__ unsigned pk2(__half a, __half b){
    return (unsigned)__half_as_ushort(a) | ((unsigned)__half_as_ushort(b) << 16);
}
__device__ __forceinline__ __half dqn(unsigned w, int nib, __half sc){
    unsigned short t = (unsigned short)((((w>>(nib*4))&0xf) ^ 0x8) | 0x6400u);
    return __hmul(__hsub(__ushort_as_half(t), __float2half(1032.f)), sc);
}
// Vectorized dequant: two adjacent nibbles (low,high of the byte at bit-offset sh)
// -> packed f16x2, SIMD affine. Bit-exact to pk2(dqn(n0),dqn(n1)). sh=(nib_even)*4.
__device__ __forceinline__ unsigned dq2(unsigned w, int sh, __half sc){
    unsigned two = (w >> sh) & 0xFFu;
    unsigned packed = ((((two & 0xf0u) << 12) | (two & 0xfu)) ^ 0x00080008u) | 0x64006400u;
    __half2 h = *reinterpret_cast<__half2*>(&packed);
    h = __hmul2(__hsub2(h, __float2half2_rn(1032.0f)), __half2half2(sc));
    return *reinterpret_cast<unsigned*>(&h);
}
#define CPA(SD,SP) asm volatile("cp.async.cg.shared.global [%0],[%1],16,16;\n" :: "r"(SD), "l"(SP))
// BN=128 (NS=16): each block does 128 N-cols → the loaded A tile is reused over
// 2x the weights, halving redundant A reads (the bottleneck). Weight/scale loads
// CLAMP rows at n0+r>=N to the expert's row 0 (no OOB; garbage masked on store).
extern "C" __global__ void moe_q4_grouped_mma(
    const __half* __restrict__ A,
    const unsigned* __restrict__ Qs,
    const __half* __restrict__ Sc,
    const int* __restrict__ tok0_arr, const int* __restrict__ eid_arr, const int* __restrict__ gend_arr,
    __half* __restrict__ Out,
    int N, int K, int band_n)
{
    const int BM=64,BN=128,BK=32,NS=16,NKS=2,GS=32,WPR=BK/8,GPR=BK/GS; // BK32 occ-tune (smem 25->12.5KB)
    // band_n=0 (default): grid=[tile,nblk] with tile fast-varying -> a full
    // tile sweep (crossing experts) happens before the N-block advances.
    // band_n=1 (BUTTER_LAGUNA_SCHED=1): grid=[nblk,tile] with nblk fast-varying ->
    // a BAND of consecutive CTAs sweeps every N-block of the SAME tile (same
    // expert weight rows) before moving to the next tile, so the immediately
    // prior tile of that same expert (if any, from a >64-row group) just
    // touched the identical weight footprint -> L2 reuse across the band.
    int bx = band_n ? blockIdx.y : blockIdx.x;
    int by = band_n ? blockIdx.x : blockIdx.y;
    int tok0=tok0_arr[bx], gend=gend_arr[bx], eid=eid_arr[bx], n0=by*BN;
    int tid=threadIdx.x,warp=tid>>5,lane=tid&31,gid=lane>>2,tid4=lane&3,wrow=warp*16;
    __shared__ __half   As[2][BM][BK];
    __shared__ unsigned Wqs[2][BN][WPR];
    __shared__ __half   Scs[2][BN][GPR];
    float oc[NS][4]; for(int s=0;s<NS;s++){oc[s][0]=oc[s][1]=oc[s][2]=oc[s][3]=0.f;}
    int nchunk=K/BK, WPB=K/32;
    long ebase=(long)eid*N;
#define LOADC(C,BUF) do{ int kc=(C)*BK; \
    for(int e=tid;e<BM*(BK/8);e+=128){ int r=e/(BK/8),k8=(e%(BK/8))*8; int t=tok0+r; int v=(t<gend); \
        unsigned sd=(unsigned)__cvta_generic_to_shared(&As[BUF][r][k8]); \
        const __half* sp=&A[(long)(v?t:tok0)*K + kc + k8]; CPA(sd,sp); } \
    for(int e=tid;e<BN*WPR/4;e+=128){ int f=e*4,r=f/WPR,w0=f%WPR; long wr=ebase+((n0+r<N)?(n0+r):0); \
        unsigned sd=(unsigned)__cvta_generic_to_shared(&Wqs[BUF][r][w0]); \
        const unsigned* sp=&Qs[wr*WPB*4 + kc/8 + w0]; CPA(sd,sp); } \
    for(int e=tid;e<BN*GPR;e+=128){ int r=e/GPR,g=e%GPR; long sr=ebase+((n0+r<N)?(n0+r):0); \
        Scs[BUF][r][g]=Sc[sr*WPB + kc/32 + g]; } \
    }while(0)
    LOADC(0,0); asm volatile("cp.async.commit_group;\n"); __syncthreads();
    for(int c=0;c<nchunk;c++){
        if(c+1<nchunk){ LOADC(c+1,(c+1)&1); asm volatile("cp.async.commit_group;\n"); asm volatile("cp.async.wait_group 1;\n"); }
        else { asm volatile("cp.async.wait_group 0;\n"); }
        __syncthreads();
        int buf=c&1;
        for(int ks=0;ks<NKS;ks++){ int ko=ks*16; int c0=ko+2*tid4;
            int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
            int acol = (lane >> 4) * 8;
            unsigned aaddr = (unsigned)__cvta_generic_to_shared(&As[buf][wrow+arow][ko+acol]);
            unsigned a0,a1,a2,a3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(aaddr));
            #pragma unroll
            for(int s=0;s<NS;s++){ int nrow=s*8+gid;
                unsigned wA=Wqs[buf][nrow][c0/8], wB=Wqs[buf][nrow][(c0+8)/8];
                __half sA=Scs[buf][nrow][c0/GS], sB=Scs[buf][nrow][(c0+8)/GS];
                unsigned b0=dq2(wA,(c0&7)*4,sA);
                unsigned b1=dq2(wB,(c0&7)*4,sB);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    :"+f"(oc[s][0]),"+f"(oc[s][1]),"+f"(oc[s][2]),"+f"(oc[s][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        __syncthreads();
    }
    int r0=tok0+wrow+gid, r8=tok0+wrow+gid+8;
    for(int s=0;s<NS;s++){ int col=n0+s*8+2*tid4;
        if(r0<gend&&col<N){ Out[(long)r0*N+col]=__float2half(oc[s][0]); Out[(long)r0*N+col+1]=__float2half(oc[s][1]); }
        if(r8<gend&&col<N){ Out[(long)r8*N+col]=__float2half(oc[s][2]); Out[(long)r8*N+col+1]=__float2half(oc[s][3]); }
    }
}
"#;

// BUTTER_LAGUNA_SCHED=1: A/B switch for the two Laguna prefill CTA-scheduling
// levers on the Q4 grouped GEMM (see moe_q4_grouped_mma / _dev below). Default
// off = byte-identical launch geometry to before this change.
#[inline]
fn laguna_sched_on() -> bool {
    std::env::var("BUTTER_LAGUNA_SCHED").as_deref() != Ok("0")
}

/// Q4-native grouped MoE GEMM: `out[t,n] = Σ_k A[t,k]·dequant(Qs[eid,n,k])` over
/// sorted token groups. `a` f16 `[mt,K]`; `qs` u32 + `sc` f16 = the model's
/// contiguous `[n_exp*N, K]` Q4 pool (moe_up_all/down_all); `g_starts[g]..[g+1]`
/// = group g's token range; `expert_ids[g]` selects the expert. Single launch,
/// no dequant/slab. Returns `out[mt,N]` f16. `K%64==0`, `N%64==0`.
///
/// `BUTTER_LAGUNA_SCHED=1` turns on two scheduling levers (both host-side data
/// here already, so no extra device sync): (1) tiles are emitted in
/// DESCENDING group-row order instead of expert order, so the biggest groups'
/// CTAs launch first and small trailing groups' stragglers overlap with them
/// instead of forming a ragged tail; (2) the launch grid is transposed so a
/// band of consecutive CTAs sweeps all N-blocks of one tile (weight-tile L2
/// reuse) before moving to the next tile, see `band_n` in the kernel source.
#[allow(clippy::too_many_arguments)]
pub fn moe_q4_grouped_mma(
    dev: &dyn Device,
    a: &Tensor,
    qs: &Tensor,
    sc: &Tensor,
    g_starts: &[usize],
    expert_ids: &[usize],
    n: usize,
    k: usize,
) -> Result<Tensor> {
    let n_active = expert_ids.len();
    let mt = g_starts.last().copied().unwrap_or(0);
    if k % 64 != 0 { return Err(Error::Msg(format!("moe_q4_grouped_mma: K {k} %64!=0"))); }
    if n % 64 != 0 { return Err(Error::Msg(format!("moe_q4_grouped_mma: N {n} %64!=0"))); }
    let out = Tensor::empty(dev, vec![mt.max(1), n], DType::F16)?;
    if n_active == 0 || mt == 0 { return Ok(out); }
    let band_n = laguna_sched_on();
    // Default: expert order 0..n_active. BUTTER_LAGUNA_SCHED=1: descending group
    // row-count, so the largest groups' tiles are emitted (and thus launched)
    // first. Purely a host-side Vec reorder, g_starts/expert_ids are already
    // host slices, so this adds no new device sync.
    let order: Vec<usize> = if band_n {
        let mut idx: Vec<usize> = (0..n_active).collect();
        idx.sort_by_key(|&g| std::cmp::Reverse(g_starts[g + 1] - g_starts[g]));
        idx
    } else {
        (0..n_active).collect()
    };
    // Per-m-tile descriptors (BM=64 token rows each, never crossing a group).
    let (mut tok0, mut eid, mut gend) = (Vec::new(), Vec::new(), Vec::new());
    for &g in &order {
        let (s, e) = (g_starts[g], g_starts[g + 1]);
        let mut t = s;
        while t < e { tok0.push(t as i32); eid.push(expert_ids[g] as i32); gend.push(e as i32); t += 64; }
    }
    let n_mtiles = tok0.len();
    let up_i32 = |v: &[i32]| -> Result<Tensor> {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Ok(Tensor::new(dev.upload(&bytes).map_err(|e| Error::Msg(format!("{e:?}")))?, vec![v.len()], DType::U32))
    };
    let (t0d, eidd, ged) = (up_i32(&tok0)?, up_i32(&eid)?, up_i32(&gend)?);
    let i = |x: i32| x.to_le_bytes().to_vec();
    let grid = if band_n {
        [(n as u32).div_ceil(128), n_mtiles as u32, 1]
    } else {
        [n_mtiles as u32, (n as u32).div_ceil(128), 1]
    };
    dev.dispatch_raw_cuda(
        MOE_Q4_GROUPED_MMA_SRC, "moe_q4_grouped_mma.cu", "moe_q4_grouped_mma",
        &[(a.buffer.as_ref(), a.offset), (qs.buffer.as_ref(), qs.offset), (sc.buffer.as_ref(), sc.offset),
          (t0d.buffer.as_ref(), 0), (eidd.buffer.as_ref(), 0), (ged.buffer.as_ref(), 0),
          (out.buffer.as_ref(), 0)],
        &[i(n as i32), i(k as i32), i(band_n as i32)],
        grid, [128, 1, 1], 0, false)?;
    Ok(out)
}

// Build fixed-size Q4 grouped-GEMM tile descriptors ON DEVICE from the routing
// `offsets[n_exp+1]` (no host sync) → graph-safe. One thread walks the experts,
// emitting one descriptor per BM=64-row tile; padding tiles get gend=0 (the GEMM
// early-exits on them). maxt = mt/BM + n_exp is a per-S constant → fixed launch.
//
// `desc!=0` (BUTTER_LAGUNA_SCHED=1): before walking, the single thread does an
// O(n_exp^2) descending selection-sort of the experts by row-count (`offsets`
// deltas) into `order`, using dynamic shared mem (2*n_exp ints) as scratch, and
// walks experts in that order instead of 0..n_exp. n_exp is at most a few
// hundred (one routing step of one layer/chunk) so this is negligible next to
// the GEMM launch it precedes; no host readback of `offsets` is added.
const MOE_BUILD_TILES_SRC: &str = r#"
extern "C" __global__ void moe_build_tiles(
    const unsigned* __restrict__ offsets, int* __restrict__ tok0,
    int* __restrict__ eid, int* __restrict__ gend, int n_exp, int BM, int maxt, int desc)
{
    if (blockIdx.x || threadIdx.x) return;
    extern __shared__ int smem[];
    int* cnt = smem;          // [n_exp], only used when desc!=0
    int* order = smem + n_exp; // [n_exp]
    if (desc) {
        for (int e = 0; e < n_exp; e++) { cnt[e] = (int)(offsets[e+1] - offsets[e]); order[e] = e; }
        // round7: O(n log n) heapsort replacing the prior O(n^2) selection sort
        // (moe_build_tiles measured ~1.34ms/call via ncu at n_exp=256, single
        // thread -- the O(n^2)=65536-compare sort was the dominant cost).
        // Correctness is insensitive to exact tie-break order here: any
        // permutation of `order[]` yields identical GEMM output since tiles
        // write disjoint output regions regardless of emission order (see
        // moe_q4_grouped_laguna_sched_ab, which A/B-tests exactly this
        // desc-mode path against the default order on skewed/zero-count
        // groups and requires bit-exact agreement). Build a max-heap over
        // order[] keyed by cnt[order[i]], then extract-max repeatedly
        // (ascending), then reverse in place for descending final order.
        for (int start = n_exp / 2 - 1; start >= 0; start--) {
            int root = start;
            while (true) {
                int child = 2 * root + 1;
                if (child >= n_exp) break;
                if (child + 1 < n_exp && cnt[order[child + 1]] > cnt[order[child]]) child++;
                if (cnt[order[root]] >= cnt[order[child]]) break;
                int tmp = order[root]; order[root] = order[child]; order[child] = tmp;
                root = child;
            }
        }
        for (int end = n_exp - 1; end > 0; end--) {
            int tmp = order[0]; order[0] = order[end]; order[end] = tmp;
            int root = 0;
            while (true) {
                int child = 2 * root + 1;
                if (child >= end) break;
                if (child + 1 < end && cnt[order[child + 1]] > cnt[order[child]]) child++;
                if (cnt[order[root]] >= cnt[order[child]]) break;
                tmp = order[root]; order[root] = order[child]; order[child] = tmp;
                root = child;
            }
        }
        for (int i = 0, j = n_exp - 1; i < j; i++, j--) {
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }
    }
    int t = 0;
    for (int ii = 0; ii < n_exp; ii++) {
        int e = desc ? order[ii] : ii;
        unsigned s = offsets[e], en = offsets[e+1];
        for (unsigned x = s; x < en; x += (unsigned)BM) {
            if (t < maxt) { tok0[t]=(int)x; eid[t]=e; gend[t]=(int)en; t++; }
        }
    }
    for (; t < maxt; t++) { tok0[t]=0; eid[t]=0; gend[t]=0; }
}
"#;

/// Fully-on-device Q4 grouped MoE GEMM: builds tile descriptors on-device from
/// `offsets` (device `[n_exp+1]` u32) with a FIXED launch (grid=[maxt,N/64],
/// maxt = mt/64 + n_exp) → NO host sync, graph-capturable. Returns `out[mt,N]` f16.
///
/// `BUTTER_LAGUNA_SCHED=1` turns on the same two scheduling levers as
/// `moe_q4_grouped_mma`: descending group-row tile order (computed ON DEVICE
/// in `moe_build_tiles`, since this path only has a device `offsets` buffer:
/// no new device->host sync is added) and the transposed/banded launch grid.
#[allow(clippy::too_many_arguments)]
pub fn moe_q4_grouped_mma_dev(
    dev: &dyn Device,
    a: &Tensor,
    qs: &Tensor,
    sc: &Tensor,
    offsets: &Tensor,
    n_exp: usize,
    mt: usize,
    n: usize,
    k: usize,
) -> Result<Tensor> {
    if k % 64 != 0 { return Err(Error::Msg(format!("moe_q4_grouped_mma_dev: K {k} %64!=0"))); }
    if n % 64 != 0 { return Err(Error::Msg(format!("moe_q4_grouped_mma_dev: N {n} %64!=0"))); }
    let out = Tensor::empty(dev, vec![mt.max(1), n], DType::F16)?;
    if mt == 0 { return Ok(out); }
    let band_n = laguna_sched_on();
    let maxt = mt.div_ceil(64) + n_exp;
    let t0d = Tensor::empty(dev, vec![maxt], DType::U32)?;
    let eidd = Tensor::empty(dev, vec![maxt], DType::U32)?;
    let ged = Tensor::empty(dev, vec![maxt], DType::U32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    // 2 int arrays of n_exp each, only needed (and only touched by the
    // kernel) when desc!=0. Kept at 0 in the default/off path so that path's
    // launch geometry is byte-identical to before this change.
    let build_shared = if band_n { (2 * n_exp * std::mem::size_of::<i32>()) as u32 } else { 0 };
    dev.dispatch_raw_cuda(
        MOE_BUILD_TILES_SRC, "moe_build_tiles.cu", "moe_build_tiles",
        &[(offsets.buffer.as_ref(), offsets.offset), (t0d.buffer.as_ref(), 0),
          (eidd.buffer.as_ref(), 0), (ged.buffer.as_ref(), 0)],
        &[i(n_exp as i32), i(64), i(maxt as i32), i(band_n as i32)],
        [1, 1, 1], [1, 1, 1], build_shared, false)?;
    let grid = if band_n {
        [(n as u32).div_ceil(128), maxt as u32, 1]
    } else {
        [maxt as u32, (n as u32).div_ceil(128), 1]
    };
    dev.dispatch_raw_cuda(
        MOE_Q4_GROUPED_MMA_SRC, "moe_q4_grouped_mma.cu", "moe_q4_grouped_mma",
        &[(a.buffer.as_ref(), a.offset), (qs.buffer.as_ref(), qs.offset), (sc.buffer.as_ref(), sc.offset),
          (t0d.buffer.as_ref(), 0), (eidd.buffer.as_ref(), 0), (ged.buffer.as_ref(), 0),
          (out.buffer.as_ref(), 0)],
        &[i(n as i32), i(k as i32), i(band_n as i32)],
        grid, [128, 1, 1], 0, false)?;
    Ok(out)
}

/// Transpose+cast+scale Q `[n_query, n_q_heads, hd]` (caller dtype) into the
/// head-major fp16 `[n_q_heads, n_query, hd]` layout the batched GEMM wants, with
/// `scale` folded in. `IN_F32` selects f32 vs f16 input.
const SDPA_TC_QPREP_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void sdpa_tc_qprep(
    const void* __restrict__ q_in,   // [Sq, nq, hd], dtype = f32 (in_f32=1) or f16
    __half*     __restrict__ q_out,  // [nq, Sq, hd] f16, scaled
    int sq, int nq, int hd, float scale, int in_f32)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)sq * nq * hd;
    if (i >= total) return;
    int d  = (int)(i % hd);
    long t = i / hd;
    int h  = (int)(t % nq);
    int r  = (int)(t / nq);          // query row
    float val;
    if (in_f32) val = ((const float*)q_in)[i];
    else        val = __half2float(((const __half*)q_in)[i]);
    long o = ((long)h * sq + r) * hd + d;
    q_out[o] = __float2half(val * scale);
}
"#;

/// Cast K `[nkv, kv_stride, hd]` → fp16 `[nkv, n_kv, hd]` (drops cache padding).
const SDPA_TC_KPREP_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void sdpa_tc_kprep(
    const void* __restrict__ k_in,   // [nkv, kv_stride, hd]
    __half*     __restrict__ k_out,  // [nkv, blk, hd] f16, THIS BLOCK ONLY
    int nkv, int kv_stride, int kb0, int blk, int hd, int in_f32)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)nkv * blk * hd;
    if (i >= total) return;
    int d  = (int)(i % hd);
    long t = i / hd;
    int p  = (int)(t % blk);         // position within this block
    int kh = (int)(t / blk);         // kv head
    long si = ((long)kh * kv_stride + (kb0 + p)) * hd + d;
    float val = in_f32 ? ((const float*)k_in)[si]
                       : __half2float(((const __half*)k_in)[si]);
    k_out[i] = __float2half(val);
}
"#;

/// Cast+transpose ONE KV block of V `[nkv, kv_stride, hd]` → fp16
/// `[nkv, hd, blk]` (V^Tᵀ per head, block-contiguous so the P·V GEMM's leading
/// dim equals `blk = k`). Reads source rows `[kb0, kb0+blk)`.
const SDPA_TC_VPREP_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void sdpa_tc_vprep(
    const void* __restrict__ v_in,   // [nkv, kv_stride, hd]
    __half*     __restrict__ v_out,  // [nkv, hd, blk] f16
    int nkv, int kv_stride, int hd, int kb0, int blk, int in_f32)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)nkv * blk * hd;
    if (i >= total) return;
    int d  = (int)(i % hd);
    long t = i / hd;
    int j  = (int)(t % blk);          // position within block
    int kh = (int)(t / blk);          // kv head
    long si = ((long)kh * kv_stride + (kb0 + j)) * hd + d;
    float val = in_f32 ? ((const float*)v_in)[si]
                       : __half2float(((const __half*)v_in)[si]);
    long oi = ((long)kh * hd + d) * blk + j;   // [kh][d][j], leading dim = blk
    v_out[oi] = __float2half(val);
}
"#;

/// Online-softmax exp + causal mask for one KV block. Reads the (already
/// `scale`-folded) logits `S_blk[nq, Sq, blk]`, writes the unnormalised weights
/// `P_blk[nq, Sq, blk]` (fp16) plus the per-(head,row) block max `bm[nq,Sq]` and
/// block sum-exp `bl[nq,Sq]`. Causal: query row `r` (absolute pos `base_kv+r`)
/// attends absolute KV pos `kb0 + j` iff `kb0 + j <= base_kv + r`.
/// One thread block per (head,row); blockDim.x threads stride over `blk`.
const SDPA_TC_SOFTMAX_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void sdpa_tc_softmax(
    const __half* __restrict__ s_blk,  // [nq, Sq, blk] logits (scaled)
    __half*       __restrict__ p_blk,  // [nq, Sq, blk] weights out
    float*        __restrict__ bm,     // [nq, Sq] block max
    float*        __restrict__ bl,     // [nq, Sq] block sum-exp
    int nq, int sq, int blk, int kb0, int base_kv)
{
    long row = (long)blockIdx.x;       // global (head*Sq + r)
    if (row >= (long)nq * sq) return;
    int r = (int)(row % sq);
    int max_kv = base_kv + r;          // inclusive absolute KV position
    const __half* s = s_blk + row * blk;
    __half* p = p_blk + row * blk;
    int tid = threadIdx.x;
    int nt  = blockDim.x;
    // pass 1: row max over valid columns
    float lm = -3.4e38f;
    for (int j = tid; j < blk; j += nt) {
        int abspos = kb0 + j;
        float v = (abspos <= max_kv) ? __half2float(s[j]) : -3.4e38f;
        if (v > lm) lm = v;
    }
    __shared__ float red[256];
    red[tid] = lm; __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) { float o = red[tid+off]; if (o > red[tid]) red[tid] = o; }
        __syncthreads();
    }
    float m = red[0];
    __syncthreads();
    // pass 2: exp + sum
    float ls = 0.0f;
    for (int j = tid; j < blk; j += nt) {
        int abspos = kb0 + j;
        float w;
        if (abspos <= max_kv && m > -3.0e38f) { w = __expf(__half2float(s[j]) - m); }
        else { w = 0.0f; }
        p[j] = __float2half(w);
        ls += w;
    }
    red[tid] = ls; __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid+off];
        __syncthreads();
    }
    if (tid == 0) {
        bm[row] = (m > -3.0e38f) ? m : -3.4e38f;
        bl[row] = red[0];
    }
}
"#;

/// Merge one block's partial output into the running FlashAttention state.
/// `o_blk[nq,Sq,hd]` = P_blk·V_blk (unnormalised). Updates running max `m_run`,
/// running sum `l_run`, and running (unnormalised) output `o_run` with the
/// standard online-softmax rescale. One block per (head,row); threads over hd.
/// Varlen (packed multi-sequence) softmax + causal mask. Identical to
/// `sdpa_tc_softmax`, but bounds each query row `r` BELOW by its packed
/// segment's start `seg_lo[r]` (absolute KV position) in addition to the
/// causal upper bound. A query thus attends only WITHIN its own packed
/// sequence: `seg_lo[r] <= abspos <= base_kv + r`. With `seg_lo[r] == 0` for
/// all rows this is bit-identical to the dense kernel.
const SDPA_TC_SOFTMAX_VARLEN_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void sdpa_tc_softmax_varlen(
    const __half* __restrict__ s_blk,  // [nq, Sq, blk] logits (scaled)
    __half*       __restrict__ p_blk,  // [nq, Sq, blk] weights out
    float*        __restrict__ bm,     // [nq, Sq] block max
    float*        __restrict__ bl,     // [nq, Sq] block sum-exp
    const int*    __restrict__ seg_lo, // [Sq] per-row segment start (abs KV pos)
    int nq, int sq, int blk, int kb0, int base_kv)
{
    long row = (long)blockIdx.x;       // global (head*Sq + r)
    if (row >= (long)nq * sq) return;
    int r = (int)(row % sq);
    int max_kv = base_kv + r;          // inclusive causal upper bound
    int lo_kv  = seg_lo[r];            // inclusive segment lower bound
    const __half* s = s_blk + row * blk;
    __half* p = p_blk + row * blk;
    int tid = threadIdx.x;
    int nt  = blockDim.x;
    // pass 1: row max over valid columns
    float lm = -3.4e38f;
    for (int j = tid; j < blk; j += nt) {
        int abspos = kb0 + j;
        float v = (abspos >= lo_kv && abspos <= max_kv) ? __half2float(s[j]) : -3.4e38f;
        if (v > lm) lm = v;
    }
    __shared__ float red[256];
    red[tid] = lm; __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) { float o = red[tid+off]; if (o > red[tid]) red[tid] = o; }
        __syncthreads();
    }
    float m = red[0];
    __syncthreads();
    // pass 2: exp + sum
    float ls = 0.0f;
    for (int j = tid; j < blk; j += nt) {
        int abspos = kb0 + j;
        float w;
        if (abspos >= lo_kv && abspos <= max_kv && m > -3.0e38f) { w = __expf(__half2float(s[j]) - m); }
        else { w = 0.0f; }
        p[j] = __float2half(w);
        ls += w;
    }
    red[tid] = ls; __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid+off];
        __syncthreads();
    }
    if (tid == 0) {
        bm[row] = (m > -3.0e38f) ? m : -3.4e38f;
        bl[row] = red[0];
    }
}
"#;

const SDPA_TC_MERGE_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void sdpa_tc_merge(
    const __half* __restrict__ o_blk,  // [nq, Sq, hd] this block's P·V
    const float*  __restrict__ bm,     // [nq, Sq] block max
    const float*  __restrict__ bl,     // [nq, Sq] block sum
    float*        __restrict__ o_run,  // [nq, Sq, hd] running output (f32)
    float*        __restrict__ m_run,  // [nq, Sq]
    float*        __restrict__ l_run,  // [nq, Sq]
    int nq, int sq, int hd)
{
    long row = (long)blockIdx.x;
    if (row >= (long)nq * sq) return;
    float m_old = m_run[row];
    float l_old = l_run[row];
    float m_b   = bm[row];
    float l_b   = bl[row];
    if (l_b <= 0.0f) return;           // block fully masked → nothing to add
    float m_new = (m_b > m_old) ? m_b : m_old;
    float a = (m_old > -3.0e38f) ? __expf(m_old - m_new) : 0.0f; // rescale old
    float b = __expf(m_b   - m_new);                              // scale block
    const __half* ob = o_blk + row * hd;
    float* orun = o_run + row * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        orun[d] = orun[d] * a + __half2float(ob[d]) * b;
    }
    if (threadIdx.x == 0) {
        m_run[row] = m_new;
        l_run[row] = l_old * a + l_b * b;
    }
}
"#;

/// Finalise: normalise the running output by `l_run` and write it back in the
/// caller's `[n_query, n_q_heads, hd]` row-major layout (transpose from the
/// head-major `[nq, Sq, hd]` accumulator). `OUT_F32` selects output dtype.
const SDPA_TC_FINALIZE_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void sdpa_tc_finalize(
    const float* __restrict__ o_run,  // [nq, Sq, hd]
    const float* __restrict__ l_run,  // [nq, Sq]
    void*        __restrict__ out,    // [Sq, nq, hd] caller dtype
    int nq, int sq, int hd, int out_f32)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)nq * sq * hd;
    if (i >= total) return;
    int d  = (int)(i % hd);
    long t = i / hd;
    int r  = (int)(t % sq);
    int h  = (int)(t / sq);
    long row = (long)h * sq + r;
    float l = l_run[row];
    float val = (l > 0.0f) ? (o_run[i] / l) : 0.0f;
    long oi = ((long)r * nq + h) * hd + d;   // [r][h][d] row-major
    if (out_f32) ((float*)out)[oi] = val;
    else         ((__half*)out)[oi] = __float2half(val);
}
"#;

/// FUSED single-kernel causal FlashAttention (no HBM score/prob materialization,
/// no qprep/kprep/vprep passes). One warp per (head, query row): the warp holds
/// `Q[r]` in registers, streams K/V rows from HBM, and runs the online softmax +
/// `O` accumulation entirely in registers, scores never touch global memory.
/// Causal is exploited by looping `p` only to `base_kv + r` (the masked upper
/// triangle is never computed, ~2× fewer FLOPs than the materialized path).
/// hd must be a multiple of 32 (warp lanes own `hd/32` dims each). Q/K/V F32 or
/// F16; output matches `q.dtype`. This is v1 (scalar FMA, no tensor cores), it
/// wins by removing the 6-dispatch HBM round-trip + the causal waste; the wmma
/// upgrade is the follow-up. NEMOTRON_FLASH_FUSED=1 in the model.
const SDPA_FLASH_FUSED_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void sdpa_flash_fused(
    const void* __restrict__ q_in,   // [Sq, nq, hd]
    const void* __restrict__ k_in,   // [nkv, kv_stride, hd]
    const void* __restrict__ v_in,   // [nkv, kv_stride, hd]
    void*       __restrict__ out,    // [Sq, nq, hd]
    int sq, int nq, int nkv, int hd, int kv_stride,
    int base_kv, float scale, int in_f32, int out_f32)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int rpb  = blockDim.x >> 5;            // query rows per block (= warps)
    int r    = blockIdx.y * rpb + warp;    // query row
    int h    = blockIdx.x;                 // q head
    if (r >= sq) return;
    int hpg  = nq / nkv;
    int kh   = h / hpg;                    // kv head
    int maxp = base_kv + r;                // inclusive causal bound
    int dpl  = hd >> 5;                    // dims per lane (hd=128 -> 4)

    const float* qf = (const float*)q_in; const __half* qhp = (const __half*)q_in;
    const float* kf = (const float*)k_in; const __half* khp = (const __half*)k_in;
    const float* vf = (const float*)v_in; const __half* vhp = (const __half*)v_in;

    float qreg[8], oreg[8];
    long qbase = ((long)r * nq + h) * (long)hd;
    #pragma unroll
    for (int t = 0; t < dpl; t++) {
        int d = lane + (t << 5);
        float qv = in_f32 ? qf[qbase + d] : __half2float(qhp[qbase + d]);
        qreg[t] = qv * scale;
        oreg[t] = 0.0f;
    }
    float m = -3.4e38f, l = 0.0f;

    for (int p = 0; p <= maxp; p++) {
        long kbase = ((long)kh * kv_stride + p) * (long)hd;
        float part = 0.0f;
        #pragma unroll
        for (int t = 0; t < dpl; t++) {
            int d = lane + (t << 5);
            float kv = in_f32 ? kf[kbase + d] : __half2float(khp[kbase + d]);
            part += qreg[t] * kv;
        }
        // full-warp reduce (sum) then broadcast the dot to every lane
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            part += __shfl_down_sync(0xffffffffu, part, off);
        float s = __shfl_sync(0xffffffffu, part, 0);
        // online softmax (identical math on every lane → consistent m/l)
        float m_new = fmaxf(m, s);
        float corr  = __expf(m - m_new);    // first iter: exp(-inf)=0
        float pij   = __expf(s - m_new);
        l = l * corr + pij;
        #pragma unroll
        for (int t = 0; t < dpl; t++) {
            int d = lane + (t << 5);
            float vv = in_f32 ? vf[kbase + d] : __half2float(vhp[kbase + d]);
            oreg[t] = oreg[t] * corr + pij * vv;
        }
        m = m_new;
    }

    long obase = ((long)r * nq + h) * (long)hd;
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    #pragma unroll
    for (int t = 0; t < dpl; t++) {
        int d = lane + (t << 5);
        float ov = oreg[t] * inv;
        if (out_f32) ((float*)out)[obase + d] = ov;
        else         ((__half*)out)[obase + d] = __float2half(ov);
    }
}
"#;

/// FUSED tensor-core (wmma) causal FlashAttention. v2 of [`SDPA_FLASH_FUSED_SRC`]:
/// 1 warp / 16-query-row tile, hd=128. QKᵀ and P·V run on tensor cores (wmma
/// 16×16×16 f16→f32); scores round-trip through SHARED (never HBM) so the online
/// softmax + causal mask are plain per-row shared reductions (no fragment-layout
/// reduction). K is transposed on load (`Kt[d][j]`) so QKᵀ = Q·Kt is a direct
/// A·B. O accumulator stays in shared f32, rescaled per-row each KV tile. Causal
/// tiles beyond the query block are skipped. f16 in/out; ~30KB static shared.
const SDPA_FLASH_WMMA_SRC: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void sdpa_flash_wmma(
    const __half* __restrict__ q_in,   // [Sq, nq, 128] (already scaled? no, scaled here)
    const __half* __restrict__ k_in,   // [nkv, kv_stride, 128]
    const __half* __restrict__ v_in,   // [nkv, kv_stride, 128]
    __half*       __restrict__ out,    // [Sq, nq, 128]
    int sq, int nq, int nkv, int kv_stride, int base_kv, float scale)
{
    const int BR=16, BC=16, NC=8;          // 16 rows, 16 keys/tile, hd=128 -> 8 chunks
    int h  = blockIdx.x;
    int R0 = blockIdx.y * BR;
    int hpg = nq / nkv, kh = h / hpg;
    int lane = threadIdx.x;                 // 0..31 (one warp)

    __shared__ __half Qs[BR][128];
    __shared__ __half Kt[128][BC];          // transposed K: [d][j]
    __shared__ __half Vs[BC][128];
    __shared__ float  Os[BR][128];
    __shared__ float  Ss[BR][BC];
    __shared__ __half Ps[BR][BC];
    __shared__ float  scratch[BR][128];
    __shared__ float  m_sh[BR], l_sh[BR];

    for (int e = lane; e < BR*128; e += 32) {
        int i = e >> 7, d = e & 127, gr = R0 + i;
        float qv = (gr < sq) ? __half2float(q_in[((long)gr*nq + h)*128 + d]) : 0.0f;
        Qs[i][d] = __float2half(qv * scale);
        Os[i][d] = 0.0f;
    }
    for (int i = lane; i < BR; i += 32) { m_sh[i] = -3.4e38f; l_sh[i] = 0.0f; }
    __syncwarp();

    int maxp_block = base_kv + R0 + BR - 1;
    int last = base_kv + sq - 1;
    if (maxp_block > last) maxp_block = last;

    wmma::fragment<wmma::accumulator,16,16,16,float> s_acc;
    for (int kj0 = 0; kj0 <= maxp_block; kj0 += BC) {
        for (int e = lane; e < BC*128; e += 32) {
            int j = e >> 7, d = e & 127, p = kj0 + j;
            int ok = (p <= last);
            Kt[d][j] = ok ? k_in[((long)kh*kv_stride + p)*128 + d] : __float2half(0.0f);
            Vs[j][d] = ok ? v_in[((long)kh*kv_stride + p)*128 + d] : __float2half(0.0f);
        }
        __syncwarp();

        wmma::fill_fragment(s_acc, 0.0f);
        for (int c = 0; c < NC; c++) {
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
            wmma::load_matrix_sync(a, &Qs[0][c*16], 128);
            wmma::load_matrix_sync(b, &Kt[c*16][0], BC);
            wmma::mma_sync(s_acc, a, b, s_acc);
        }
        wmma::store_matrix_sync(&Ss[0][0], s_acc, BC, wmma::mem_row_major);
        __syncwarp();

        if (lane < BR) {
            int i = lane, gr = R0 + i;
            if (gr < sq) {
                int maxp = base_kv + R0 + i;
                float rmax = -3.4e38f;
                for (int j = 0; j < BC; j++) {
                    float sv = (kj0 + j <= maxp) ? Ss[i][j] : -3.4e38f;
                    rmax = fmaxf(rmax, sv);
                }
                float m_old = m_sh[i];
                float m_new = fmaxf(m_old, rmax);
                float corr  = (m_old > -3.0e38f) ? __expf(m_old - m_new) : 0.0f;
                float rsum  = 0.0f;
                for (int j = 0; j < BC; j++) {
                    float w = (kj0 + j <= maxp && m_new > -3.0e38f) ? __expf(Ss[i][j] - m_new) : 0.0f;
                    Ps[i][j] = __float2half(w);
                    rsum += w;
                }
                for (int d = 0; d < 128; d++) Os[i][d] *= corr;
                m_sh[i] = m_new;
                l_sh[i] = l_sh[i]*corr + rsum;
            } else {
                for (int j = 0; j < BC; j++) Ps[i][j] = __float2half(0.0f);
            }
        }
        __syncwarp();

        for (int dc = 0; dc < NC; dc++) {
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> pa;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> vb;
            wmma::fragment<wmma::accumulator,16,16,16,float> pv;
            wmma::fill_fragment(pv, 0.0f);
            wmma::load_matrix_sync(pa, &Ps[0][0], BC);
            wmma::load_matrix_sync(vb, &Vs[0][dc*16], 128);
            wmma::mma_sync(pv, pa, vb, pv);
            wmma::store_matrix_sync(&scratch[0][dc*16], pv, 128, wmma::mem_row_major);
        }
        __syncwarp();
        for (int e = lane; e < BR*128; e += 32) { int i = e >> 7, d = e & 127; Os[i][d] += scratch[i][d]; }
        __syncwarp();
    }

    if (lane < BR) {
        int i = lane, gr = R0 + i;
        if (gr < sq) {
            float l = l_sh[i], inv = (l > 0.0f) ? 1.0f/l : 0.0f;
            for (int d = 0; d < 128; d++) out[((long)gr*nq + h)*128 + d] = __float2half(Os[i][d]*inv);
        }
    }
}
"#;

/// FlashAttention-2 with register-resident O (mma.sync m16n8k16). v3 of the fused
/// path: O lives in registers (acc fragments), not shared: freeing ~16KB so the
/// 1-warp/16-row block hits high occupancy (v2's 30KB shared capped it at ~11%).
/// QKᵀ + P·V via `mma.sync` (manual fragment packing, documented layout); the
/// causal mask + online softmax round-trip the small S/P tiles through shared
/// (cheap), and O is rescaled per-row IN registers each KV tile (lane t holds
/// rows {gid, gid+8} → multiply by corr[gid], corr[gid+8]). f16 in/out, hd=128,
/// ~3KB shared. NEMOTRON_FLASH_MMA=1.
const SDPA_FLASH_MMA_SRC: &str = r#"
#include <cuda_fp16.h>
__device__ __forceinline__ unsigned pk(__half a, __half b){
    return (unsigned)__half_as_ushort(a) | ((unsigned)__half_as_ushort(b) << 16);
}
extern "C" __global__ void sdpa_flash_mma(
    const void* __restrict__ q_in, const __half* __restrict__ k_in,
    const __half* __restrict__ v_in, void* __restrict__ out,
    int sq, int nq, int nkv, int kv_stride, int base_kv, float scale, int in_f32, int out_f32)
{
    const int BR=16, BC=16, HD=128, NC=8, ND=16;  // NC=HD/16 (k-chunks), ND=HD/8 (O d-tiles)
    int h = blockIdx.x, R0 = blockIdx.y * BR;
    int hpg = nq / nkv, kh = h / hpg;
    int lane = threadIdx.x, gid = lane >> 2, tid4 = lane & 3;
    const float* qf=(const float*)q_in; const __half* qh=(const __half*)q_in;
    // K/V are pre-cast to f16 by the wrapper (cheap, repeatedly-read in the KV loop).

    __shared__ __half Qs[BR][HD];
    __shared__ __half Ks[BC][HD];
    __shared__ __half Vs[BC][HD];
    __shared__ float  Ss[BR][BC];
    __shared__ __half Ps[BR][BC];
    __shared__ float  cv[BR], mv[BR], lv[BR];

    for (int e = lane; e < BR*HD; e += 32) { int i=e>>7,d=e&127,gr=R0+i; long idx=((long)gr*nq+h)*HD+d;
        float qv = (gr<sq) ? (in_f32 ? qf[idx] : __half2float(qh[idx])) : 0.f;
        Qs[i][d] = __float2half(qv*scale); }
    float oc[ND][4];
    for (int t=0;t<ND;t++){ oc[t][0]=oc[t][1]=oc[t][2]=oc[t][3]=0.f; }
    if (lane<BR){ mv[lane]=-3.4e38f; lv[lane]=0.f; }
    __syncwarp();

    int last = base_kv + sq - 1;
    int maxpb = base_kv + R0 + BR - 1; if (maxpb>last) maxpb=last;

    for (int kj0=0; kj0<=maxpb; kj0+=BC) {
        for (int e=lane;e<BC*HD;e+=32){ int j=e>>7,d=e&127,p=kj0+j,ok=(p<=last); long idx=((long)kh*kv_stride+p)*HD+d;
            Ks[j][d]= ok? k_in[idx] : __float2half(0.f);
            Vs[j][d]= ok? v_in[idx] : __float2half(0.f); }
        __syncwarp();

        // QKᵀ -> 2 S-tiles (n8: keys 0-7, 8-15), accumulate over 8 k-chunks
        float sc[2][4] = {{0,0,0,0},{0,0,0,0}};
        for (int c=0;c<NC;c++){
            int kcol = c*16;
            // A = Q[16x16] chunk (row-major m16n8k16 A layout)
            unsigned a0=pk(Qs[gid][kcol+2*tid4],   Qs[gid][kcol+2*tid4+1]);
            unsigned a1=pk(Qs[gid+8][kcol+2*tid4], Qs[gid+8][kcol+2*tid4+1]);
            unsigned a2=pk(Qs[gid][kcol+2*tid4+8], Qs[gid][kcol+2*tid4+9]);
            unsigned a3=pk(Qs[gid+8][kcol+2*tid4+8],Qs[gid+8][kcol+2*tid4+9]);
            for (int s=0;s<2;s++){
                int kbase = s*8;  // key offset for this n8 tile; n_local = gid
                // B[kdim][n] = K[n][kdim]; col-major frag
                unsigned b0=pk(Ks[kbase+gid][kcol+2*tid4],   Ks[kbase+gid][kcol+2*tid4+1]);
                unsigned b1=pk(Ks[kbase+gid][kcol+2*tid4+8], Ks[kbase+gid][kcol+2*tid4+9]);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    :"+f"(sc[s][0]),"+f"(sc[s][1]),"+f"(sc[s][2]),"+f"(sc[s][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        // store S tiles -> shared (acc layout: c0,c1 row gid; c2,c3 row gid+8)
        for (int s=0;s<2;s++){
            Ss[gid][s*8+2*tid4]   = sc[s][0]; Ss[gid][s*8+2*tid4+1]   = sc[s][1];
            Ss[gid+8][s*8+2*tid4] = sc[s][2]; Ss[gid+8][s*8+2*tid4+1] = sc[s][3];
        }
        __syncwarp();

        if (lane<BR){ int i=lane,gr=R0+i;
            if (gr<sq){ int maxp=base_kv+R0+i; float rmax=-3.4e38f;
                for (int j=0;j<BC;j++){ float sv=(kj0+j<=maxp)?Ss[i][j]:-3.4e38f; rmax=fmaxf(rmax,sv); }
                float mo=mv[i], mn=fmaxf(mo,rmax); float cor=(mo>-3.0e38f)?__expf(mo-mn):0.f; float rs=0.f;
                for (int j=0;j<BC;j++){ float w=(kj0+j<=maxp&&mn>-3.0e38f)?__expf(Ss[i][j]-mn):0.f; Ps[i][j]=__float2half(w); rs+=w; }
                cv[i]=cor; mv[i]=mn; lv[i]=lv[i]*cor+rs;
            } else { cv[i]=1.f; for(int j=0;j<BC;j++) Ps[i][j]=__float2half(0.f); }
        }
        __syncwarp();

        // rescale O in registers (lane holds rows gid, gid+8)
        float cg=cv[gid], cg8=cv[gid+8];
        for (int t=0;t<ND;t++){ oc[t][0]*=cg; oc[t][1]*=cg; oc[t][2]*=cg8; oc[t][3]*=cg8; }

        // PV: O += P·V. A=P[16x16] (row-major), B=V[16(k)x8(d)] per d-tile (col-major B[k][d])
        unsigned pa0=pk(Ps[gid][2*tid4],   Ps[gid][2*tid4+1]);
        unsigned pa1=pk(Ps[gid+8][2*tid4], Ps[gid+8][2*tid4+1]);
        unsigned pa2=pk(Ps[gid][2*tid4+8], Ps[gid][2*tid4+9]);
        unsigned pa3=pk(Ps[gid+8][2*tid4+8],Ps[gid+8][2*tid4+9]);
        for (int t=0;t<ND;t++){
            int dcol=t*8;  // n = d in [dcol,dcol+8), n_local=gid
            unsigned vb0=pk(Vs[2*tid4][dcol+gid],   Vs[2*tid4+1][dcol+gid]);
            unsigned vb1=pk(Vs[2*tid4+8][dcol+gid], Vs[2*tid4+9][dcol+gid]);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                :"+f"(oc[t][0]),"+f"(oc[t][1]),"+f"(oc[t][2]),"+f"(oc[t][3])
                :"r"(pa0),"r"(pa1),"r"(pa2),"r"(pa3),"r"(vb0),"r"(vb1));
        }
        __syncwarp();
    }

    // finalize: O[row][dcol] = oc / l ; lane holds rows gid (c0,c1) and gid+8 (c2,c3)
    float l0 = lv[gid], l8 = lv[gid+8];
    float inv0=(l0>0.f)?1.f/l0:0.f, inv8=(l8>0.f)?1.f/l8:0.f;
    int r0=R0+gid, r8=R0+gid+8;
    float* of=(float*)out; __half* oh=(__half*)out;
    for (int t=0;t<ND;t++){ int dcol=t*8;
        if (r0<sq){ long i0=((long)r0*nq+h)*HD+dcol+2*tid4;
            if(out_f32){ of[i0]=oc[t][0]*inv0; of[i0+1]=oc[t][1]*inv0; }
            else       { oh[i0]=__float2half(oc[t][0]*inv0); oh[i0+1]=__float2half(oc[t][1]*inv0); } }
        if (r8<sq){ long i8=((long)r8*nq+h)*HD+dcol+2*tid4;
            if(out_f32){ of[i8]=oc[t][2]*inv8; of[i8+1]=oc[t][3]*inv8; }
            else       { oh[i8]=__float2half(oc[t][2]*inv8); oh[i8+1]=__float2half(oc[t][3]*inv8); } }
    }
}


#include <cuda_fp16.h>
extern "C" __global__ void sdpa_flash_mma_w4(
    const void* __restrict__ q_in, const __half* __restrict__ k_in,
    const __half* __restrict__ v_in, void* __restrict__ out,
    int sq, int nq, int nkv, int kv_stride, int base_kv, float scale, int in_f32, int out_f32)
{
    const int BR=16, BC=16, HD=128, NC=8, ND=16, NW=4;  // 4 warps/block, 64 rows
    int h = blockIdx.x;
    int wid = threadIdx.x >> 5;
    int R0 = blockIdx.y * (BR*NW) + wid * BR;
    int hpg = nq / nkv, kh = h / hpg;
    int lane = threadIdx.x & 31, gid = lane >> 2, tid4 = lane & 3;
    const float* qf=(const float*)q_in; const __half* qh=(const __half*)q_in;
    // K/V are pre-cast to f16 by the wrapper (cheap, repeatedly-read in the KV loop).

    __shared__ __half Qs_a[4][BR][HD];
    __shared__ __half Ks[BC][HD];
    __shared__ __half Vs[BC][HD];
    __shared__ float  Ss_a[4][BR][BC];
    __shared__ __half Ps_a[4][BR][BC];
    __shared__ float  cv_a[4][BR], mv_a[4][BR], lv_a[4][BR];
    __half (*Qs)[HD] = Qs_a[wid];
    float (*Ss)[BC] = Ss_a[wid];
    __half (*Ps)[BC] = Ps_a[wid];
    float *cv = cv_a[wid], *mv = mv_a[wid], *lv = lv_a[wid];

    for (int e = lane; e < BR*HD; e += 32) { int i=e>>7,d=e&127,gr=R0+i; long idx=((long)gr*nq+h)*HD+d;
        float qv = (gr<sq) ? (in_f32 ? qf[idx] : __half2float(qh[idx])) : 0.f;
        Qs[i][d] = __float2half(qv*scale); }
    float oc[ND][4];
    for (int t=0;t<ND;t++){ oc[t][0]=oc[t][1]=oc[t][2]=oc[t][3]=0.f; }
    if (lane<BR){ mv[lane]=-3.4e38f; lv[lane]=0.f; }
    __syncwarp();

    int last = base_kv + sq - 1;
    int maxpb = base_kv + R0 + BR - 1; if (maxpb>last) maxpb=last;          // this warp's causal max
    int blk_r_hi = blockIdx.y * (BR*NW) + (BR*NW) - 1;
    int maxpb_blk = base_kv + blk_r_hi; if (maxpb_blk>last) maxpb_blk=last;  // block max

    for (int kj0=0; kj0<=maxpb_blk; kj0+=BC) {
        __syncthreads();
        for (int e=threadIdx.x;e<BC*HD;e+=NW*32){ int j=e>>7,d=e&127,p=kj0+j,ok=(p<=last); long idx=((long)kh*kv_stride+p)*HD+d;
            Ks[j][d]= ok? k_in[idx] : __float2half(0.f);
            Vs[j][d]= ok? v_in[idx] : __float2half(0.f); }
        __syncthreads();
        if (kj0 > maxpb) continue;   // beyond this warp's causal window (barriers above are block-uniform)
        __syncwarp();

        // QKᵀ -> 2 S-tiles (n8: keys 0-7, 8-15), accumulate over 8 k-chunks
        float sc[2][4] = {{0,0,0,0},{0,0,0,0}};
        for (int c=0;c<NC;c++){
            int kcol = c*16;
            // A = Q[16x16] chunk (row-major m16n8k16 A layout)
            unsigned a0=pk(Qs[gid][kcol+2*tid4],   Qs[gid][kcol+2*tid4+1]);
            unsigned a1=pk(Qs[gid+8][kcol+2*tid4], Qs[gid+8][kcol+2*tid4+1]);
            unsigned a2=pk(Qs[gid][kcol+2*tid4+8], Qs[gid][kcol+2*tid4+9]);
            unsigned a3=pk(Qs[gid+8][kcol+2*tid4+8],Qs[gid+8][kcol+2*tid4+9]);
            for (int s=0;s<2;s++){
                int kbase = s*8;  // key offset for this n8 tile; n_local = gid
                // B[kdim][n] = K[n][kdim]; col-major frag
                unsigned b0=pk(Ks[kbase+gid][kcol+2*tid4],   Ks[kbase+gid][kcol+2*tid4+1]);
                unsigned b1=pk(Ks[kbase+gid][kcol+2*tid4+8], Ks[kbase+gid][kcol+2*tid4+9]);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    :"+f"(sc[s][0]),"+f"(sc[s][1]),"+f"(sc[s][2]),"+f"(sc[s][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        // store S tiles -> shared (acc layout: c0,c1 row gid; c2,c3 row gid+8)
        for (int s=0;s<2;s++){
            Ss[gid][s*8+2*tid4]   = sc[s][0]; Ss[gid][s*8+2*tid4+1]   = sc[s][1];
            Ss[gid+8][s*8+2*tid4] = sc[s][2]; Ss[gid+8][s*8+2*tid4+1] = sc[s][3];
        }
        __syncwarp();

        if (lane<BR){ int i=lane,gr=R0+i;
            if (gr<sq){ int maxp=base_kv+R0+i; float rmax=-3.4e38f;
                for (int j=0;j<BC;j++){ float sv=(kj0+j<=maxp)?Ss[i][j]:-3.4e38f; rmax=fmaxf(rmax,sv); }
                float mo=mv[i], mn=fmaxf(mo,rmax); float cor=(mo>-3.0e38f)?__expf(mo-mn):0.f; float rs=0.f;
                for (int j=0;j<BC;j++){ float w=(kj0+j<=maxp&&mn>-3.0e38f)?__expf(Ss[i][j]-mn):0.f; Ps[i][j]=__float2half(w); rs+=w; }
                cv[i]=cor; mv[i]=mn; lv[i]=lv[i]*cor+rs;
            } else { cv[i]=1.f; for(int j=0;j<BC;j++) Ps[i][j]=__float2half(0.f); }
        }
        __syncwarp();

        // rescale O in registers (lane holds rows gid, gid+8)
        float cg=cv[gid], cg8=cv[gid+8];
        for (int t=0;t<ND;t++){ oc[t][0]*=cg; oc[t][1]*=cg; oc[t][2]*=cg8; oc[t][3]*=cg8; }

        // PV: O += P·V. A=P[16x16] (row-major), B=V[16(k)x8(d)] per d-tile (col-major B[k][d])
        unsigned pa0=pk(Ps[gid][2*tid4],   Ps[gid][2*tid4+1]);
        unsigned pa1=pk(Ps[gid+8][2*tid4], Ps[gid+8][2*tid4+1]);
        unsigned pa2=pk(Ps[gid][2*tid4+8], Ps[gid][2*tid4+9]);
        unsigned pa3=pk(Ps[gid+8][2*tid4+8],Ps[gid+8][2*tid4+9]);
        for (int t=0;t<ND;t++){
            int dcol=t*8;  // n = d in [dcol,dcol+8), n_local=gid
            unsigned vb0=pk(Vs[2*tid4][dcol+gid],   Vs[2*tid4+1][dcol+gid]);
            unsigned vb1=pk(Vs[2*tid4+8][dcol+gid], Vs[2*tid4+9][dcol+gid]);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                :"+f"(oc[t][0]),"+f"(oc[t][1]),"+f"(oc[t][2]),"+f"(oc[t][3])
                :"r"(pa0),"r"(pa1),"r"(pa2),"r"(pa3),"r"(vb0),"r"(vb1));
        }
        __syncwarp();
    }

    // finalize: O[row][dcol] = oc / l ; lane holds rows gid (c0,c1) and gid+8 (c2,c3)
    float l0 = lv[gid], l8 = lv[gid+8];
    float inv0=(l0>0.f)?1.f/l0:0.f, inv8=(l8>0.f)?1.f/l8:0.f;
    int r0=R0+gid, r8=R0+gid+8;
    float* of=(float*)out; __half* oh=(__half*)out;
    for (int t=0;t<ND;t++){ int dcol=t*8;
        if (r0<sq){ long i0=((long)r0*nq+h)*HD+dcol+2*tid4;
            if(out_f32){ of[i0]=oc[t][0]*inv0; of[i0+1]=oc[t][1]*inv0; }
            else       { oh[i0]=__float2half(oc[t][0]*inv0); oh[i0+1]=__float2half(oc[t][1]*inv0); } }
        if (r8<sq){ long i8=((long)r8*nq+h)*HD+dcol+2*tid4;
            if(out_f32){ of[i8]=oc[t][2]*inv8; of[i8+1]=oc[t][3]*inv8; }
            else       { oh[i8]=__float2half(oc[t][2]*inv8); oh[i8+1]=__float2half(oc[t][3]*inv8); } }
    }
}

// Split-KV variant: grid.z partitions the KV range; each partition computes
// an UNNORMALIZED running-softmax partial (o*l, m, l) over its KV slice with
// identical math/masking, and a merge kernel combines partials. Parallelizes
// the serial per-row-block KV walk that dominates deep-context prefill.
extern "C" __global__ void sdpa_flash_mma_w4s(
    const void* __restrict__ q_in, const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    float* __restrict__ o_part,   // [P, sq, nq, HD] unnormalized
    float* __restrict__ ml_part,  // [P, sq, nq, 2] (m, l)
    int sq, int nq, int nkv, int kv_stride, int base_kv, float scale, int in_f32,
    int kv_chunk)
{
    const int BR=16, BC=16, HD=128, NC=8, ND=16, NW=4;
    int h = blockIdx.x;
    int wid = threadIdx.x >> 5;
    int R0 = blockIdx.y * (BR*NW) + wid * BR;
    int part = blockIdx.z;
    int kv0 = part * kv_chunk;
    int hpg = nq / nkv, kh = h / hpg;
    int lane = threadIdx.x & 31, gid = lane >> 2, tid4 = lane & 3;
    const float* qf=(const float*)q_in; const __half* qh=(const __half*)q_in;

    __shared__ __half Qs_a[4][BR][HD];
    __shared__ __half Ks2[2][BC][HD];   // cp.async double-buffered KV stages
    __shared__ __half Vs2[2][BC][HD];
    __shared__ float  Ss_a[4][BR][BC];
    __shared__ __half Ps_a[4][BR][BC];
    __shared__ float  cv_a[4][BR], mv_a[4][BR], lv_a[4][BR];
    __half (*Qs)[HD] = Qs_a[wid];
    float (*Ss)[BC] = Ss_a[wid];
    __half (*Ps)[BC] = Ps_a[wid];
    float *cv = cv_a[wid], *mv = mv_a[wid], *lv = lv_a[wid];

    for (int e = lane; e < BR*HD; e += 32) { int i=e>>7,d=e&127,gr=R0+i; long idx=((long)gr*nq+h)*HD+d;
        float qv = (gr<sq) ? (in_f32 ? qf[idx] : __half2float(qh[idx])) : 0.f;
        Qs[i][d] = __float2half(qv*scale); }
    float oc[ND][4];
    for (int t=0;t<ND;t++){ oc[t][0]=oc[t][1]=oc[t][2]=oc[t][3]=0.f; }
    if (lane<BR){ mv[lane]=-3.4e38f; lv[lane]=0.f; }
    __syncwarp();

    int last = base_kv + sq - 1;
    int maxpb = base_kv + R0 + BR - 1; if (maxpb>last) maxpb=last;
    int blk_r_hi = blockIdx.y * (BR*NW) + (BR*NW) - 1;
    int maxpb_blk = base_kv + blk_r_hi; if (maxpb_blk>last) maxpb_blk=last;
    int kv_hi = kv0 + kv_chunk - 1; if (kv_hi > maxpb_blk) kv_hi = maxpb_blk;

    // cp.async 16B chunk staging: BC*HD f16 = 4KB per matrix = 256 chunks;
    // 128 threads x 2 chunks each. Tail tiles (p > last) fall back to a tiny
    // zero-fill after the wait (rare: only the last tile of the sequence).
    #define KV_ISSUE(kj_,st_) do{                                                     for (int c2=threadIdx.x; c2<2*BC*HD/8; c2+=NW*32){                                 int which=c2>=(BC*HD/8);                                                       int e8=(c2%(BC*HD/8))*8;                                                       int j=e8>>7, d=e8&127, p=(kj_)+j;                                              const __half* src = which? &v_in[((long)kh*kv_stride+p)*HD+d]                                           : &k_in[((long)kh*kv_stride+p)*HD+d];                 __half* dst = which? &Vs2[st_][j][d] : &Ks2[st_][j][d];                        if (p<=last) {                                                                     unsigned sa=(unsigned)__cvta_generic_to_shared(dst);                           asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(sa),"l"(src));             }                                                                          }                                                                              asm volatile("cp.async.commit_group;");                                    }while(0)
    KV_ISSUE(kv0,0);
    for (int kj0=kv0; kj0<=kv_hi; kj0+=BC) {
        int st=((kj0-kv0)/BC)&1;
        if (kj0+BC<=kv_hi) {
            KV_ISSUE(kj0+BC,st^1);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        // zero-fill the (rare) beyond-last lanes of this stage
        if (kj0+BC-1 > last) {
            for (int e=threadIdx.x;e<BC*HD;e+=NW*32){ int j=e>>7,d=e&127;
                if (kj0+j>last){ Ks2[st][j][d]=__float2half(0.f); Vs2[st][j][d]=__float2half(0.f); } }
            __syncthreads();
        }
        __half (*Ks)[HD] = Ks2[st];
        __half (*Vs)[HD] = Vs2[st];
        if (kj0 > maxpb) continue;
        __syncwarp();

        float sc[2][4] = {{0,0,0,0},{0,0,0,0}};
        for (int c=0;c<NC;c++){
            int kcol = c*16;
            unsigned a0=pk(Qs[gid][kcol+2*tid4],   Qs[gid][kcol+2*tid4+1]);
            unsigned a1=pk(Qs[gid+8][kcol+2*tid4], Qs[gid+8][kcol+2*tid4+1]);
            unsigned a2=pk(Qs[gid][kcol+2*tid4+8], Qs[gid][kcol+2*tid4+9]);
            unsigned a3=pk(Qs[gid+8][kcol+2*tid4+8],Qs[gid+8][kcol+2*tid4+9]);
            for (int s2=0;s2<2;s2++){
                int kbase = s2*8;
                unsigned b0=pk(Ks[kbase+gid][kcol+2*tid4],   Ks[kbase+gid][kcol+2*tid4+1]);
                unsigned b1=pk(Ks[kbase+gid][kcol+2*tid4+8], Ks[kbase+gid][kcol+2*tid4+9]);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    :"+f"(sc[s2][0]),"+f"(sc[s2][1]),"+f"(sc[s2][2]),"+f"(sc[s2][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        for (int s2=0;s2<2;s2++){
            Ss[gid][s2*8+2*tid4]   = sc[s2][0]; Ss[gid][s2*8+2*tid4+1]   = sc[s2][1];
            Ss[gid+8][s2*8+2*tid4] = sc[s2][2]; Ss[gid+8][s2*8+2*tid4+1] = sc[s2][3];
        }
        __syncwarp();

        if (lane<BR){ int i=lane,gr=R0+i;
            if (gr<sq){ int maxp=base_kv+R0+i; float rmax=-3.4e38f;
                for (int j=0;j<BC;j++){ float sv=(kj0+j<=maxp)?Ss[i][j]:-3.4e38f; rmax=fmaxf(rmax,sv); }
                float mo=mv[i], mn=fmaxf(mo,rmax); float cor=(mo>-3.0e38f)?__expf(mo-mn):0.f; float rs=0.f;
                for (int j=0;j<BC;j++){ float w=(kj0+j<=maxp&&mn>-3.0e38f)?__expf(Ss[i][j]-mn):0.f; Ps[i][j]=__float2half(w); rs+=w; }
                cv[i]=cor; mv[i]=mn; lv[i]=lv[i]*cor+rs;
            } else { cv[i]=1.f; for(int j=0;j<BC;j++) Ps[i][j]=__float2half(0.f); }
        }
        __syncwarp();

        float cg=cv[gid], cg8=cv[gid+8];
        for (int t=0;t<ND;t++){ oc[t][0]*=cg; oc[t][1]*=cg; oc[t][2]*=cg8; oc[t][3]*=cg8; }

        unsigned pa0=pk(Ps[gid][2*tid4],   Ps[gid][2*tid4+1]);
        unsigned pa1=pk(Ps[gid+8][2*tid4], Ps[gid+8][2*tid4+1]);
        unsigned pa2=pk(Ps[gid][2*tid4+8], Ps[gid][2*tid4+9]);
        unsigned pa3=pk(Ps[gid+8][2*tid4+8],Ps[gid+8][2*tid4+9]);
        for (int t=0;t<ND;t++){
            int dcol=t*8;
            unsigned vb0=pk(Vs[2*tid4][dcol+gid],   Vs[2*tid4+1][dcol+gid]);
            unsigned vb1=pk(Vs[2*tid4+8][dcol+gid], Vs[2*tid4+9][dcol+gid]);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                :"+f"(oc[t][0]),"+f"(oc[t][1]),"+f"(oc[t][2]),"+f"(oc[t][3])
                :"r"(pa0),"r"(pa1),"r"(pa2),"r"(pa3),"r"(vb0),"r"(vb1));
        }
        __syncwarp();
    }

    #undef KV_ISSUE
    // emit UNNORMALIZED partial + (m,l); rows beyond sq skipped
    int r0=R0+gid, r8=R0+gid+8;
    long pbase = (long)part * sq;
    for (int t=0;t<ND;t++){ int dcol=t*8;
        if (r0<sq){ long i0=((pbase+r0)*nq+h)*(long)HD+dcol+2*tid4;
            o_part[i0]=oc[t][0]; o_part[i0+1]=oc[t][1]; }
        if (r8<sq){ long i8=((pbase+r8)*nq+h)*(long)HD+dcol+2*tid4;
            o_part[i8]=oc[t][2]; o_part[i8+1]=oc[t][3]; }
    }
    if (lane<BR){ int gr=R0+lane;
        if (gr<sq){ long mi=((pbase+gr)*nq+h)*2;
            ml_part[mi]=mv[lane]; ml_part[mi+1]=lv[lane]; } }
}

// FP8 (e4m3) attention v2: QK and PV via m16n8k32 e4m3 mma (2x f16 rate).
// K/V are PRE-QUANTIZED to e4m3 buffers once per call (kv_to_e4m3 below),
// v1 quantized in-smem on every KV re-read (frexp per element x 32
// row-blocks) which ate the mma savings; v2 staging is byte copies and the
// KV re-read traffic halves. BC=32 keys/tile so the PV k32 mma is fully
// utilized (v1's BC=16 zero-padded half of every PV mma).
__device__ __forceinline__ unsigned char enc_e4m3a(float v){
    unsigned s=(v<0.f)?0x80u:0u; float a=fabsf(v);
    if(!(a>0.f)) return (unsigned char)s;
    if(a>=448.f) return (unsigned char)(s|0x7Eu);
    int e; float fr=frexpf(a,&e); int E=e+6;
    unsigned c;
    if(E<1){ int m=(int)rintf(a*512.f); if(m>7)m=7; c=(unsigned)m; }
    else { int m=(int)rintf((2.f*fr-1.f)*8.f); if(m>7){m=0;E+=1;} if(E>15){E=15;m=6;} c=(unsigned)((E<<3)|m); }
    return (unsigned char)(s|c);
}
// One pass: f16 KV cache -> e4m3 (scaled). n = total elements.
extern "C" __global__ void kv_to_e4m3(
    const __half* __restrict__ k_in, const __half* __restrict__ v_in,
    unsigned char* __restrict__ k8, unsigned char* __restrict__ v8,
    const float* __restrict__ scales, long n)
{
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    long stride=(long)gridDim.x*blockDim.x;
    float ki=scales[1], vi=scales[2];
    for(;i<n;i+=stride){
        k8[i]=enc_e4m3a(__half2float(k_in[i])*ki);
        v8[i]=enc_e4m3a(__half2float(v_in[i])*vi);
    }
}
extern "C" __global__ void sdpa_flash_mma_w4s_fp8(
    const void* __restrict__ q_in, const unsigned char* __restrict__ k8,
    const unsigned char* __restrict__ v8,
    float* __restrict__ o_part, float* __restrict__ ml_part,
    const float* __restrict__ scales,
    int sq, int nq, int nkv, int kv_stride, int base_kv, float scale, int in_f32,
    int kv_chunk)
{
    const int BR=16, BC=32, HD=128, NC=4, ND=16, NW=4;
    int h = blockIdx.x;
    int wid = threadIdx.x >> 5;
    int R0 = blockIdx.y * (BR*NW) + wid * BR;
    int part = blockIdx.z;
    int kv0 = part * kv_chunk;
    int hpg = nq / nkv, kh = h / hpg;
    int lane = threadIdx.x & 31, gid = lane >> 2, tid4 = lane & 3;
    const float* qf=(const float*)q_in; const __half* qh=(const __half*)q_in;
    float qi=scales[0], ki=scales[1], vi=scales[2];
    float qk_de = 1.f/(qi*ki);
    float v_de  = 1.f/vi;

    __shared__ unsigned char Qs_a[4][BR][HD];
    __shared__ unsigned char Ks[BC][HD];
    __shared__ unsigned char Vs[HD][BC];   // [d][j] for PV B frags
    __shared__ float  Ss_a[4][BR][BC];
    __shared__ unsigned char Ps_a[4][BR][BC];
    __shared__ float  cv_a[4][BR], mv_a[4][BR], lv_a[4][BR];
    unsigned char (*Qs)[HD] = Qs_a[wid];
    float (*Ss)[BC] = Ss_a[wid];
    unsigned char (*Ps)[BC] = Ps_a[wid];
    float *cv = cv_a[wid], *mv = mv_a[wid], *lv = lv_a[wid];

    for (int e = lane; e < BR*HD; e += 32) { int i=e>>7,d=e&127,gr=R0+i; long idx=((long)gr*nq+h)*HD+d;
        float qv = (gr<sq) ? (in_f32 ? qf[idx] : __half2float(qh[idx])) : 0.f;
        Qs[i][d] = enc_e4m3a(qv*scale*qi); }
    float oc[ND][4];
    for (int t=0;t<ND;t++){ oc[t][0]=oc[t][1]=oc[t][2]=oc[t][3]=0.f; }
    if (lane<BR){ mv[lane]=-3.4e38f; lv[lane]=0.f; }
    __syncwarp();

    int last = base_kv + sq - 1;
    int maxpb = base_kv + R0 + BR - 1; if (maxpb>last) maxpb=last;
    int blk_r_hi = blockIdx.y * (BR*NW) + (BR*NW) - 1;
    int maxpb_blk = base_kv + blk_r_hi; if (maxpb_blk>last) maxpb_blk=last;
    int kv_hi = kv0 + kv_chunk - 1; if (kv_hi > maxpb_blk) kv_hi = maxpb_blk;

    #define PK4(p0,p1,p2,p3) ((unsigned)(p0)|((unsigned)(p1)<<8)|((unsigned)(p2)<<16)|((unsigned)(p3)<<24))
    for (int kj0=kv0; kj0<=kv_hi; kj0+=BC) {
        __syncthreads();
        for (int e=threadIdx.x;e<BC*HD;e+=NW*32){ int j=e>>7,d=e&127,p=kj0+j,ok=(p<=last); long idx=((long)kh*kv_stride+p)*HD+d;
            unsigned char kb = ok? k8[idx] : 0;
            unsigned char vb = ok? v8[idx] : 0;
            Ks[j][d]=kb; Vs[d][j]=vb; }
        __syncthreads();
        if (kj0 > maxpb) continue;
        __syncwarp();

        // QK: 4 n8 key tiles (BC=32), 4 k32 chunks over HD
        float sc[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        for (int c=0;c<NC;c++){
            int kcol = c*32;
            unsigned a0=PK4(Qs[gid][kcol+4*tid4],Qs[gid][kcol+4*tid4+1],Qs[gid][kcol+4*tid4+2],Qs[gid][kcol+4*tid4+3]);
            unsigned a1=PK4(Qs[gid+8][kcol+4*tid4],Qs[gid+8][kcol+4*tid4+1],Qs[gid+8][kcol+4*tid4+2],Qs[gid+8][kcol+4*tid4+3]);
            unsigned a2=PK4(Qs[gid][kcol+4*tid4+16],Qs[gid][kcol+4*tid4+17],Qs[gid][kcol+4*tid4+18],Qs[gid][kcol+4*tid4+19]);
            unsigned a3=PK4(Qs[gid+8][kcol+4*tid4+16],Qs[gid+8][kcol+4*tid4+17],Qs[gid+8][kcol+4*tid4+18],Qs[gid+8][kcol+4*tid4+19]);
            for (int s2=0;s2<4;s2++){
                int kbase = s2*8;
                unsigned b0=PK4(Ks[kbase+gid][kcol+4*tid4],Ks[kbase+gid][kcol+4*tid4+1],Ks[kbase+gid][kcol+4*tid4+2],Ks[kbase+gid][kcol+4*tid4+3]);
                unsigned b1=PK4(Ks[kbase+gid][kcol+4*tid4+16],Ks[kbase+gid][kcol+4*tid4+17],Ks[kbase+gid][kcol+4*tid4+18],Ks[kbase+gid][kcol+4*tid4+19]);
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    :"+f"(sc[s2][0]),"+f"(sc[s2][1]),"+f"(sc[s2][2]),"+f"(sc[s2][3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        for (int s2=0;s2<4;s2++){
            Ss[gid][s2*8+2*tid4]   = sc[s2][0]*qk_de; Ss[gid][s2*8+2*tid4+1]   = sc[s2][1]*qk_de;
            Ss[gid+8][s2*8+2*tid4] = sc[s2][2]*qk_de; Ss[gid+8][s2*8+2*tid4+1] = sc[s2][3]*qk_de;
        }
        __syncwarp();

        if (lane<BR){ int i=lane,gr=R0+i;
            if (gr<sq){ int maxp=base_kv+R0+i; float rmax=-3.4e38f;
                for (int j=0;j<BC;j++){ float sv=(kj0+j<=maxp)?Ss[i][j]:-3.4e38f; rmax=fmaxf(rmax,sv); }
                float mo=mv[i], mn=fmaxf(mo,rmax); float cor=(mo>-3.0e38f)?__expf(mo-mn):0.f; float rs=0.f;
                for (int j=0;j<BC;j++){ float w=(kj0+j<=maxp&&mn>-3.0e38f)?__expf(Ss[i][j]-mn):0.f;
                    Ps[i][j]=enc_e4m3a(w*448.f); rs+=w; }
                cv[i]=cor; mv[i]=mn; lv[i]=lv[i]*cor+rs;
            } else { cv[i]=1.f; for(int j=0;j<BC;j++) Ps[i][j]=0; }
        }
        __syncwarp();

        float cg=cv[gid], cg8=cv[gid+8];
        for (int t=0;t<ND;t++){ oc[t][0]*=cg; oc[t][1]*=cg; oc[t][2]*=cg8; oc[t][3]*=cg8; }

        // PV: one full k32 e4m3 mma per d-tile (BC=32 keys = k dim)
        unsigned pa0=PK4(Ps[gid][4*tid4],Ps[gid][4*tid4+1],Ps[gid][4*tid4+2],Ps[gid][4*tid4+3]);
        unsigned pa1=PK4(Ps[gid+8][4*tid4],Ps[gid+8][4*tid4+1],Ps[gid+8][4*tid4+2],Ps[gid+8][4*tid4+3]);
        unsigned pa2=PK4(Ps[gid][4*tid4+16],Ps[gid][4*tid4+17],Ps[gid][4*tid4+18],Ps[gid][4*tid4+19]);
        unsigned pa3=PK4(Ps[gid+8][4*tid4+16],Ps[gid+8][4*tid4+17],Ps[gid+8][4*tid4+18],Ps[gid+8][4*tid4+19]);
        for (int t=0;t<ND;t++){
            int dcol=t*8;
            unsigned vb0=PK4(Vs[dcol+gid][4*tid4],Vs[dcol+gid][4*tid4+1],Vs[dcol+gid][4*tid4+2],Vs[dcol+gid][4*tid4+3]);
            unsigned vb1=PK4(Vs[dcol+gid][4*tid4+16],Vs[dcol+gid][4*tid4+17],Vs[dcol+gid][4*tid4+18],Vs[dcol+gid][4*tid4+19]);
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                :"+f"(oc[t][0]),"+f"(oc[t][1]),"+f"(oc[t][2]),"+f"(oc[t][3])
                :"r"(pa0),"r"(pa1),"r"(pa2),"r"(pa3),"r"(vb0),"r"(vb1));
        }
        __syncwarp();
    }
    #undef PK4

    // P was quantized scaled by 448 (e4m3 precision for w in [0,1]); fold out.
    int r0=R0+gid, r8=R0+gid+8;
    long pbase = (long)part * sq;
    float ode = v_de / 448.f;
    for (int t=0;t<ND;t++){ int dcol=t*8;
        if (r0<sq){ long i0=((pbase+r0)*nq+h)*(long)HD+dcol+2*tid4;
            o_part[i0]=oc[t][0]*ode; o_part[i0+1]=oc[t][1]*ode; }
        if (r8<sq){ long i8=((pbase+r8)*nq+h)*(long)HD+dcol+2*tid4;
            o_part[i8]=oc[t][2]*ode; o_part[i8+1]=oc[t][3]*ode; }
    }
    if (lane<BR){ int gr=R0+lane;
        if (gr<sq){ long mi=((pbase+gr)*nq+h)*2;
            ml_part[mi]=mv[lane]; ml_part[mi+1]=lv[lane]; } }
}

// Merge split-KV partials: out[r,h,d] = sum_p o_p e^{m_p-m*} / sum_p l_p e^{m_p-m*}.
extern "C" __global__ void sdpa_flash_merge(
    const float* __restrict__ o_part, const float* __restrict__ ml_part,
    void* __restrict__ out, int sq, int nq, int n_part, int out_f32)
{
    const int HD=128;
    long n = (long)sq*nq*HD;
    long e = (long)blockIdx.x*blockDim.x+threadIdx.x;
    if (e>=n) return;
    long rh = e / HD;            // r*nq + h
    float mstar=-3.4e38f;
    for (int p=0;p<n_part;p++){ float m=ml_part[((long)p*sq*nq+rh)*2]; mstar=fmaxf(mstar,m); }
    float lsum=0.f, osum=0.f;
    for (int p=0;p<n_part;p++){
        float m=ml_part[((long)p*sq*nq+rh)*2], l=ml_part[((long)p*sq*nq+rh)*2+1];
        float w=(m>-3.0e38f)?__expf(m-mstar):0.f;
        lsum += l*w;
        osum += o_part[(long)p*sq*nq*HD+e]*w;
    }
    float r = (lsum>0.f)? osum/lsum : 0.f;
    if (out_f32) ((float*)out)[e]=r; else ((__half*)out)[e]=__float2half(r);
}

"#;

/// FlashAttention-2, register-resident O via mma.sync (see [`SDPA_FLASH_MMA_SRC`]).
/// Same contract as [`sdpa_multi_tc`]; casts Q/K/V → f16, runs the single kernel,
/// returns `q.dtype`. head_dim must be 128. CUDA-only.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_flash_mma(
    dev: &dyn Device,
    q: &Tensor, k: &Tensor, v: &Tensor,
    head_dim: usize, n_q_heads: u32, base_kv: u32, n_query: u32,
    kv_stride: u32, heads_per_group: u32, causal: bool, scale: f32,
) -> Result<Tensor> {
    let hd = head_dim;
    let nq = n_q_heads as usize;
    let sq = n_query as usize;
    let hpg = heads_per_group as usize;
    let nkv = nq / hpg;
    let base = base_kv as usize;
    if hd != 128 { return Err(Error::Msg(format!("sdpa_flash_mma: head_dim {hd} != 128"))); }
    if !causal { return Err(Error::Msg("sdpa_flash_mma: only causal wired".into())); }
    let n_kv = base + sq;
    if n_kv > kv_stride as usize {
        return Err(Error::Msg(format!("sdpa_flash_mma: n_kv {n_kv} exceeds kv_stride {kv_stride}")));
    }
    // Pre-cast K/V → f16 (cheap: [nkv,cap,hd]≈2M elems; they're re-read every KV
    // tile so f16 halves that HBM traffic). Q is read once → pass directly (no big
    // [Sq,nq,hd] cast). O written in native dtype (no output cast).
    let in_f32 = match q.dtype { DType::F32 => 1i32, DType::F16 => 0,
        other => return Err(Error::Msg(format!("sdpa_flash_mma: unsupported dtype {other:?}"))) };
    let out_f32 = if q.dtype == DType::F32 { 1i32 } else { 0 };
    let kf16 = match k.dtype { DType::F16 => k.clone(), _ => cast_f32_f16(dev, k)? };
    let vf16 = match v.dtype { DType::F16 => v.clone(), _ => cast_f32_f16(dev, v)? };
    let out = Tensor::empty(dev, vec![sq, nq, hd], q.dtype)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    // Split-KV at depth (NEMOTRON_FLASH_SPLITKV_OFF=1 disables): each block's
    // serial KV walk dominates deep-context prefill; partition the KV range
    // across grid.z, emit unnormalized running-softmax partials, merge. The
    // partial-softmax merge is the standard exact decomposition.
    let kv_chunk_target = 8192usize;
    let n_part = n_kv.div_ceil(kv_chunk_target).min(8);
    // FP8 attention (NEMOTRON_FLASH_FP8=1): e4m3 QK and PV mma at 2x the f16
    // tensor-core rate, attention at depth is mma-compute-bound. Static
    // conservative scales (16x: |x|<28 covers the post-softmax-scale Q and the
    // K/V ranges with e4m3's 448 ceiling). Quant-regime lever.
    if std::env::var("NEMOTRON_FLASH_FP8").is_ok() {
        let np = n_part.max(1);
        let kv_chunk = n_kv.div_ceil(np).div_ceil(32) * 32;
        let scl: [f32; 3] = [16.0, 16.0, 16.0];
        let mut sb = Vec::with_capacity(12);
        for v in scl { sb.extend_from_slice(&v.to_le_bytes()); }
        let scales = Tensor::new(dev.upload(&sb)?, vec![3], DType::F32);
        // pre-quantize the K/V cache to e4m3 ONCE per call (one elementwise
        // pass; also halves the per-tile KV re-read traffic).
        let kv_elems = nkv * kv_stride as usize * hd;
        let k8 = Tensor::new(dev.alloc(kv_elems)?, vec![kv_elems], DType::U32);
        let v8 = Tensor::new(dev.alloc(kv_elems)?, vec![kv_elems], DType::U32);
        let l = |x: i64| x.to_le_bytes().to_vec();
        dev.dispatch_raw_cuda(
            SDPA_FLASH_MMA_SRC, "sdpa_flash_mma.cu", "kv_to_e4m3",
            &[(kf16.buffer.as_ref(), kf16.offset), (vf16.buffer.as_ref(), vf16.offset),
              (k8.buffer.as_ref(), 0), (v8.buffer.as_ref(), 0), (scales.buffer.as_ref(), 0)],
            &[l(kv_elems as i64)],
            [((kv_elems as u32).div_ceil(256 * 4)).max(1), 1, 1], [256, 1, 1], 0, false)?;
        let o_part = Tensor::new(dev.alloc(np * sq * nq * hd * 4)?, vec![np * sq * nq * hd], DType::F32);
        let ml_part = Tensor::new(dev.alloc(np * sq * nq * 2 * 4)?, vec![np * sq * nq * 2], DType::F32);
        let grid = [nq as u32, (sq as u32).div_ceil(64), np as u32];
        dev.dispatch_raw_cuda(
            SDPA_FLASH_MMA_SRC, "sdpa_flash_mma.cu", "sdpa_flash_mma_w4s_fp8",
            &[(q.buffer.as_ref(), q.offset), (k8.buffer.as_ref(), 0),
              (v8.buffer.as_ref(), 0), (o_part.buffer.as_ref(), 0), (ml_part.buffer.as_ref(), 0),
              (scales.buffer.as_ref(), 0)],
            &[i(sq as i32), i(nq as i32), i(nkv as i32), i(kv_stride as i32), i(base as i32), f(scale), i(in_f32), i(kv_chunk as i32)],
            grid, [128, 1, 1], 0, false)?;
        let n_el = (sq * nq * hd) as u32;
        dev.dispatch_raw_cuda(
            SDPA_FLASH_MMA_SRC, "sdpa_flash_mma.cu", "sdpa_flash_merge",
            &[(o_part.buffer.as_ref(), 0), (ml_part.buffer.as_ref(), 0), (out.buffer.as_ref(), 0)],
            &[i(sq as i32), i(nq as i32), i(np as i32), i(out_f32)],
            [n_el.div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
        return Ok(out);
    }
    if n_part > 1 && std::env::var("NEMOTRON_FLASH_SPLITKV_OFF").is_err() {
        let kv_chunk = n_kv.div_ceil(n_part).div_ceil(16) * 16; // BC-aligned
        let o_part = Tensor::new(dev.alloc(n_part * sq * nq * hd * 4)?, vec![n_part * sq * nq * hd], DType::F32);
        let ml_part = Tensor::new(dev.alloc(n_part * sq * nq * 2 * 4)?, vec![n_part * sq * nq * 2], DType::F32);
        let grid = [nq as u32, (sq as u32).div_ceil(64), n_part as u32];
        dev.dispatch_raw_cuda(
            SDPA_FLASH_MMA_SRC, "sdpa_flash_mma.cu", "sdpa_flash_mma_w4s",
            &[(q.buffer.as_ref(), q.offset), (kf16.buffer.as_ref(), kf16.offset),
              (vf16.buffer.as_ref(), vf16.offset), (o_part.buffer.as_ref(), 0), (ml_part.buffer.as_ref(), 0)],
            &[i(sq as i32), i(nq as i32), i(nkv as i32), i(kv_stride as i32), i(base as i32), f(scale), i(in_f32), i(kv_chunk as i32)],
            grid, [128, 1, 1], 0, false)?;
        let n_el = (sq * nq * hd) as u32;
        dev.dispatch_raw_cuda(
            SDPA_FLASH_MMA_SRC, "sdpa_flash_mma.cu", "sdpa_flash_merge",
            &[(o_part.buffer.as_ref(), 0), (ml_part.buffer.as_ref(), 0), (out.buffer.as_ref(), 0)],
            &[i(sq as i32), i(nq as i32), i(n_part as i32), i(out_f32)],
            [n_el.div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
        return Ok(out);
    }
    // 4-warp blocks: 64 query rows/block sharing the K/V smem stage,
    // 2.7x the 1-warp kernel at S=2048, bit-identical output.
    let grid = [nq as u32, (sq as u32).div_ceil(64), 1];
    dev.dispatch_raw_cuda(
        SDPA_FLASH_MMA_SRC, "sdpa_flash_mma.cu", "sdpa_flash_mma_w4",
        &[(q.buffer.as_ref(), q.offset), (kf16.buffer.as_ref(), kf16.offset),
          (vf16.buffer.as_ref(), vf16.offset), (out.buffer.as_ref(), 0)],
        &[i(sq as i32), i(nq as i32), i(nkv as i32), i(kv_stride as i32), i(base as i32), f(scale), i(in_f32), i(out_f32)],
        grid, [128, 1, 1], 0, false)?;
    Ok(out)
}

/// Fused tensor-core causal FlashAttention (see [`SDPA_FLASH_WMMA_SRC`]). Same
/// numerics/contract as [`sdpa_multi_tc`]; casts Q/K/V → f16, runs the single
/// wmma kernel, returns `q.dtype`. head_dim must be 128. CUDA-only.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_flash_wmma(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_q_heads: u32,
    base_kv: u32,
    n_query: u32,
    kv_stride: u32,
    heads_per_group: u32,
    causal: bool,
    scale: f32,
) -> Result<Tensor> {
    let hd = head_dim;
    let nq = n_q_heads as usize;
    let sq = n_query as usize;
    let hpg = heads_per_group as usize;
    let nkv = nq / hpg;
    let base = base_kv as usize;
    if hd != 128 { return Err(Error::Msg(format!("sdpa_flash_wmma: head_dim {hd} != 128"))); }
    if !causal { return Err(Error::Msg("sdpa_flash_wmma: only causal wired".into())); }
    let n_kv = base + sq;
    if n_kv > kv_stride as usize {
        return Err(Error::Msg(format!("sdpa_flash_wmma: n_kv {n_kv} exceeds kv_stride {kv_stride}")));
    }
    // tensor cores need f16 inputs; cast if the caller passed f32 (the KV cache).
    let to_f16 = |t: &Tensor| -> Result<Tensor> {
        match t.dtype { DType::F16 => Ok(t.clone()), DType::F32 => cast_f32_f16(dev, t),
            other => Err(Error::Msg(format!("sdpa_flash_wmma: unsupported dtype {other:?}"))) }
    };
    let qh = to_f16(q)?;
    let kh = to_f16(k)?;
    let vh = to_f16(v)?;
    let out_h = Tensor::empty(dev, vec![sq, nq, hd], DType::F16)?;

    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    let grid = [nq as u32, (sq as u32).div_ceil(16), 1];
    dev.dispatch_raw_cuda(
        SDPA_FLASH_WMMA_SRC, "sdpa_flash_wmma.cu", "sdpa_flash_wmma",
        &[(qh.buffer.as_ref(), qh.offset), (kh.buffer.as_ref(), kh.offset),
          (vh.buffer.as_ref(), vh.offset), (out_h.buffer.as_ref(), 0)],
        &[i(sq as i32), i(nq as i32), i(nkv as i32), i(kv_stride as i32),
          i(base as i32), f(scale)],
        grid, [32, 1, 1], 0, false)?;
    match q.dtype { DType::F16 => Ok(out_h), _ => cast_f16_f32(dev, &out_h) }
}

/// Fused causal FlashAttention (see [`SDPA_FLASH_FUSED_SRC`]). Same signature +
/// numerics as [`sdpa_multi_tc`] (its correctness oracle): Q pre-scaled by
/// `scale`, GQA via `heads_per_group`, causal online softmax, output `[Sq,nq,hd]`
/// in `q.dtype`. CUDA-only. `head_dim % 32 == 0` required.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_flash_fused(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_q_heads: u32,
    base_kv: u32,
    n_query: u32,
    kv_stride: u32,
    heads_per_group: u32,
    causal: bool,
    scale: f32,
) -> Result<Tensor> {
    let hd = head_dim;
    let nq = n_q_heads as usize;
    let sq = n_query as usize;
    let hpg = heads_per_group as usize;
    let nkv = nq / hpg;
    let base = base_kv as usize;
    if hd % 32 != 0 {
        return Err(Error::Msg(format!("sdpa_flash_fused: head_dim {hd} not a multiple of 32")));
    }
    if !causal {
        return Err(Error::Msg("sdpa_flash_fused: only causal wired".into()));
    }
    let n_kv = base + sq;
    if n_kv > kv_stride as usize {
        return Err(Error::Msg(format!("sdpa_flash_fused: n_kv {n_kv} exceeds kv_stride {kv_stride}")));
    }
    let in_f32 = match q.dtype { DType::F32 => 1i32, DType::F16 => 0,
        other => return Err(Error::Msg(format!("sdpa_flash_fused: unsupported dtype {other:?}"))) };
    let out_f32 = if q.dtype == DType::F32 { 1i32 } else { 0 };
    let out = Tensor::empty(dev, vec![sq, nq, hd], q.dtype)?;

    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    let rpb: u32 = 4;                      // 4 warps/block, one query row each
    let block = [rpb * 32, 1, 1];
    let grid = [nq as u32, (sq as u32).div_ceil(rpb), 1];
    dev.dispatch_raw_cuda(
        SDPA_FLASH_FUSED_SRC, "sdpa_flash_fused.cu", "sdpa_flash_fused",
        &[(q.buffer.as_ref(), q.offset), (k.buffer.as_ref(), k.offset),
          (v.buffer.as_ref(), v.offset), (out.buffer.as_ref(), 0)],
        &[i(sq as i32), i(nq as i32), i(nkv as i32), i(hd as i32), i(kv_stride as i32),
          i(base as i32), f(scale), i(in_f32), i(out_f32)],
        grid, block, 0, false)?;
    Ok(out)
}

/// Tensor-core prefill SDPA, drop-in faster replacement for [`sdpa_multi`].
/// Same signature/semantics (causal/full, GQA, head_dim=128, Q pre-scaled by
/// `scale`), but runs QKᵀ and P·V on the cuBLAS tensor cores with a
/// FlashAttention online-softmax loop tiled over KV. Q/K/V may be F32 or F16;
/// output matches `q.dtype`. CUDA-only (falls back via the GEMM/dispatch FFI).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_multi_tc(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_q_heads: u32,
    base_kv: u32,
    n_query: u32,
    kv_stride: u32,
    heads_per_group: u32,
    causal: bool,
    scale: f32,
) -> Result<Tensor> {
    let hd = head_dim;
    let nq = n_q_heads as usize;
    let sq = n_query as usize;
    let hpg = heads_per_group as usize;
    let nkv = nq / hpg;
    let base = base_kv as usize;
    // Total KV positions the deepest query row attends.
    let n_kv = if causal { base + sq } else { base + sq };
    if n_kv > kv_stride as usize {
        return Err(Error::Msg(format!(
            "sdpa_multi_tc: n_kv {n_kv} exceeds kv_stride {kv_stride}")));
    }
    let in_f32 = match q.dtype { DType::F32 => 1i32, DType::F16 => 0,
        other => return Err(Error::Msg(format!("sdpa_multi_tc: unsupported dtype {other:?}"))) };

    // ── prep buffers (all f16) ──────────────────────────────────────────────
    let qh = Tensor::empty(dev, vec![nq, sq, hd], DType::F16)?;          // [nq,Sq,hd] scaled

    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    let blk256 = |n: usize| -> [u32; 3] { [((n as u32).div_ceil(256)).max(1), 1, 1] };

    dev.dispatch_raw_cuda(SDPA_TC_QPREP_SRC, "sdpa_tc_qprep.cu", "sdpa_tc_qprep",
        &[(q.buffer.as_ref(), q.offset), (qh.buffer.as_ref(), 0)],
        &[i(sq as i32), i(nq as i32), i(hd as i32), f(scale), i(in_f32)],
        blk256(nq*sq*hd), [256,1,1], 0, false)?;

    // ── running FlashAttention state (init via upload: o=0, l=0, m=-inf) ─────
    let o_run = Tensor::new(dev.alloc_zeroed(nq*sq*hd*4)?, vec![nq, sq, hd], DType::F32);
    let l_run = Tensor::new(dev.alloc_zeroed(nq*sq*4)?, vec![nq, sq], DType::F32);
    let neg: Vec<u8> = (0..nq*sq).flat_map(|_| (-3.4e38f32).to_le_bytes()).collect();
    let m_run = Tensor::new(dev.upload(&neg)?, vec![nq, sq], DType::F32);

    // KV block size: bound scores buffer ≈ nq*Sq*BK*2 bytes. 2048 keeps it ≤ ~512MB
    // at Sq≤4096; smaller Sq could go larger but 2048 is a safe universal default.
    let bk = 2048usize.min(n_kv);
    let scores = Tensor::empty(dev, vec![nq, sq, bk], DType::F16)?;
    let p_blk  = Tensor::empty(dev, vec![nq, sq, bk], DType::F16)?;
    let bm     = Tensor::empty(dev, vec![nq, sq], DType::F32)?;
    let bl     = Tensor::empty(dev, vec![nq, sq], DType::F32)?;
    let o_blk  = Tensor::empty(dev, vec![nq, sq, hd], DType::F16)?;
    let vt     = Tensor::empty(dev, vec![nkv, hd, bk], DType::F16)?;     // per-block V^T
    let kh     = Tensor::empty(dev, vec![nkv, bk, hd], DType::F16)?;     // per-block K, f16 (was whole-prefix)

    let el = 2i64; // f16 bytes
    let mut kb0 = 0usize;
    while kb0 < n_kv {
        let blk = bk.min(n_kv - kb0);

        // ── K prep: cast+copy THIS KV block only → kh[nkv,blk,hd] ───────────
        // (was one whole-causal-prefix conversion before the loop; see the
        // PERF FIX note in `sdpa_multi_tc_varlen` below for the rationale.)
        dev.dispatch_raw_cuda(SDPA_TC_KPREP_SRC, "sdpa_tc_kprep.cu", "sdpa_tc_kprep",
            &[(k.buffer.as_ref(), k.offset), (kh.buffer.as_ref(), 0)],
            &[i(nkv as i32), i(kv_stride as i32), i(kb0 as i32), i(blk as i32), i(hd as i32), i(in_f32)],
            blk256(nkv*blk*hd), [256,1,1], 0, false)?;

        // ── QKᵀ: per KV group (16 q-heads share one KV head) ────────────────
        // C[Sq, blk] = Qh[Sq,hd] · Kh_blk[blk,hd]ᵀ ; batch over the hpg q-heads.
        for g in 0..nkv {
            let q_off = g * hpg * sq * hd * 2;             // bytes into qh
            let k_off = g * blk * hd * 2;                  // bytes into kh (buffer-relative now)
            let s_off = g * hpg * sq * blk * 2;            // bytes into scores
            dev.gemm_strided_batched_off(
                qh.buffer.as_ref(),     q_off, (sq*hd) as i64 * el,   // X stride = one q-head
                kh.buffer.as_ref(),     k_off, 0,                     // W stride = 0 (broadcast KV)
                scores.buffer.as_ref(), s_off, (sq*blk) as i64 * el,  // out stride
                sq, blk, hd, hpg, DType::F16)?;
        }

        // ── softmax + causal mask for this block ────────────────────────────
        // scores/p_blk are PACKED at pitch `blk` (head stride sq*blk, row pitch
        // blk), only the first nq*sq*blk f16 of the bk-sized allocation are used.
        // This keeps the GEMM's lda == k == blk valid for the final partial block.
        dev.dispatch_raw_cuda(SDPA_TC_SOFTMAX_SRC, "sdpa_tc_softmax.cu", "sdpa_tc_softmax",
            &[(scores.buffer.as_ref(), 0), (p_blk.buffer.as_ref(), 0),
              (bm.buffer.as_ref(), 0), (bl.buffer.as_ref(), 0)],
            &[i(nq as i32), i(sq as i32), i(blk as i32), i(kb0 as i32), i(base as i32)],
            [(nq*sq) as u32, 1, 1], [256,1,1], 0, false)?;

        // ── transpose this block of V → vt[nkv, hd, blk] (lda = blk = k) ────
        dev.dispatch_raw_cuda(SDPA_TC_VPREP_SRC, "sdpa_tc_vprep.cu", "sdpa_tc_vprep",
            &[(v.buffer.as_ref(), v.offset), (vt.buffer.as_ref(), 0)],
            &[i(nkv as i32), i(kv_stride as i32), i(hd as i32), i(kb0 as i32), i(blk as i32), i(in_f32)],
            blk256(nkv*blk*hd), [256,1,1], 0, false)?;

        // ── P·V: O_blk[Sq,hd] = P[Sq,blk] · Vt[hd,blk]ᵀ ; batch over q-heads ─
        for g in 0..nkv {
            let p_off = g * hpg * sq * blk * 2;            // bytes into p_blk (pitch blk)
            let v_off = g * hd * blk * 2;                  // bytes into vt (this group's slab)
            let o_off = g * hpg * sq * hd * 2;             // bytes into o_blk
            dev.gemm_strided_batched_off(
                p_blk.buffer.as_ref(), p_off, (sq*blk) as i64 * el,
                vt.buffer.as_ref(),    v_off, 0,                       // Vt stride = 0 (broadcast)
                o_blk.buffer.as_ref(), o_off, (sq*hd) as i64 * el,
                sq, hd, blk, hpg, DType::F16)?;
        }

        // ── merge into running state ────────────────────────────────────────
        dev.dispatch_raw_cuda(SDPA_TC_MERGE_SRC, "sdpa_tc_merge.cu", "sdpa_tc_merge",
            &[(o_blk.buffer.as_ref(), 0), (bm.buffer.as_ref(), 0), (bl.buffer.as_ref(), 0),
              (o_run.buffer.as_ref(), 0), (m_run.buffer.as_ref(), 0), (l_run.buffer.as_ref(), 0)],
            &[i(nq as i32), i(sq as i32), i(hd as i32)],
            [(nq*sq) as u32, 1, 1], [128,1,1], 0, false)?;

        kb0 += blk;
    }

    // ── finalise: normalise + transpose head-major → row-major ──────────────
    let out = Tensor::empty(dev, vec![sq, nq, hd], q.dtype)?;
    let out_f32 = if q.dtype == DType::F32 { 1i32 } else { 0 };
    dev.dispatch_raw_cuda(SDPA_TC_FINALIZE_SRC, "sdpa_tc_finalize.cu", "sdpa_tc_finalize",
        &[(o_run.buffer.as_ref(), 0), (l_run.buffer.as_ref(), 0), (out.buffer.as_ref(), 0)],
        &[i(nq as i32), i(sq as i32), i(hd as i32), i(out_f32)],
        blk256(nq*sq*hd), [256,1,1], 0, false)?;
    Ok(out)
}

/// Varlen (packed multi-sequence) tensor-core FlashAttention. Identical to
/// [`sdpa_multi_tc`] except each query attends only within its own packed
/// segment: the softmax masks KV positions below `seg_lo[r]` (the absolute
/// start of query row `r`'s sequence). `seg_lo` is an `[n_query]` i32 device
/// buffer. Used by the `NEMOTRON_PACKED` batched-prefill path so N sequences
/// share one set of GEMMs while attention stays block-diagonal (correct).
///
/// NOTE: this computes the full `[Sq, n_kv]` QKᵀ and masks off-segment to
/// −inf, bit-correct, but still O((ΣL)²) QKᵀ for the packed-segment case.
/// The `seg_len == KV-block-size` branch below skips that; `win` (below)
/// is the analogous skip for a per-row SLIDING WINDOW instead of a packed
/// segment boundary.
pub fn sdpa_multi_tc_varlen(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seg_lo: &Tensor,        // [n_query] i32: abs KV start of each query's segment
    seg_len: u32,           // packed segment length; if == KV-block size, enables
                            // the per-block query-range SKIP (O((NL)^2)→O(NL^2)).
                            // 0 (or != block size) → full-range (correct, no skip).
    head_dim: usize,
    n_q_heads: u32,
    base_kv: u32,
    n_query: u32,
    kv_stride: u32,
    heads_per_group: u32,
    causal: bool,
    scale: f32,
    // Sliding-window block skip. 0 = off (no effect on any existing caller).
    // When `win > 0`, `seg_lo` is expected to encode a sliding window of
    // width `win` (`seg_lo[r] = max(0, (base_kv+r) - (win-1))`, exactly what
    // Laguna's sliding-layer prefill already uploads) and the per-KV-block
    // query-row range fed to the QKᵀ/PV GEMMs is restricted to the rows
    // whose window can possibly reach that block, see the derivation at
    // the `q_lo`/`q_cnt` computation below. Independent of `seg_len`/the
    // packed-segment skip: a caller should pass exactly one of the two
    // (Laguna always passes `seg_len = 0` and this `win`).
    win: u32,
) -> Result<Tensor> {
    let hd = head_dim;
    let nq = n_q_heads as usize;
    let sq = n_query as usize;
    let hpg = heads_per_group as usize;
    let nkv = nq / hpg;
    let base = base_kv as usize;
    let n_kv = if causal { base + sq } else { base + sq };
    if n_kv > kv_stride as usize {
        return Err(Error::Msg(format!(
            "sdpa_multi_tc_varlen: n_kv {n_kv} exceeds kv_stride {kv_stride}")));
    }
    if seg_lo.elem_count() != sq {
        return Err(Error::Msg(format!(
            "sdpa_multi_tc_varlen: seg_lo len {} != n_query {sq}", seg_lo.elem_count())));
    }
    let in_f32 = match q.dtype { DType::F32 => 1i32, DType::F16 => 0,
        other => return Err(Error::Msg(format!("sdpa_multi_tc_varlen: unsupported dtype {other:?}"))) };

    let qh = Tensor::empty(dev, vec![nq, sq, hd], DType::F16)?;

    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    let blk256 = |n: usize| -> [u32; 3] { [((n as u32).div_ceil(256)).max(1), 1, 1] };

    dev.dispatch_raw_cuda(SDPA_TC_QPREP_SRC, "sdpa_tc_qprep.cu", "sdpa_tc_qprep",
        &[(q.buffer.as_ref(), q.offset), (qh.buffer.as_ref(), 0)],
        &[i(sq as i32), i(nq as i32), i(hd as i32), f(scale), i(in_f32)],
        blk256(nq*sq*hd), [256,1,1], 0, false)?;

    let o_run = Tensor::new(dev.alloc_zeroed(nq*sq*hd*4)?, vec![nq, sq, hd], DType::F32);
    let l_run = Tensor::new(dev.alloc_zeroed(nq*sq*4)?, vec![nq, sq], DType::F32);
    let neg: Vec<u8> = (0..nq*sq).flat_map(|_| (-3.4e38f32).to_le_bytes()).collect();
    let m_run = Tensor::new(dev.upload(&neg)?, vec![nq, sq], DType::F32);

    // KV block size. Full-attention layers (win==0) keep 2048 (bounds the
    // scores buffer, see sibling `sdpa_multi_tc`'s doc). Sliding-window
    // layers (win!=0): the block-skip above only omits blocks with
    // q_cnt==0 ENTIRELY -- a block that IS touched still pays a full
    // bk-wide QK^T/PV GEMM + softmax/merge even though the window (512 for
    // Laguna, far smaller than 2048) only needs a fraction of that width.
    // Shrinking bk for win!=0 tightens the per-block reachable-range union
    // (`[kb0, kb0+blk+win)`, see the q_lo/q_cnt derivation below) at the
    // cost of more loop iterations (more kprep/QK/softmax/merge launches).
    // BUTTER_LAGUNA_SDPA_WIN_BK overrides the windowed-path block size for
    // A/B sweeps. Default 1024, NOT the 512 window width itself: measured
    // bk=512 vs bk=1024 vs bk=2048(orig) on Laguna-S-2.1/GB10 (chunk=1024,
    // MARLIN+MOEFUSE) -- bk=512 regresses n=1024 prefill -4% (extra
    // loop/launch overhead from doubling block count outweighs the reduced
    // overcompute there), while bk=1024 nets +1-1.5% at n=8192/32768 with
    // no regression anywhere tested. Full-attention layers (win==0) are
    // never affected by this env var.
    let win_bk: usize = std::env::var("BUTTER_LAGUNA_SDPA_WIN_BK")
        .ok().and_then(|v| v.parse().ok()).filter(|&b: &usize| b >= 1).unwrap_or(1024);
    let bk = if win != 0 { win_bk.min(n_kv) } else { 2048usize.min(n_kv) };
    let scores = Tensor::empty(dev, vec![nq, sq, bk], DType::F16)?;
    let p_blk  = Tensor::empty(dev, vec![nq, sq, bk], DType::F16)?;
    let bm     = Tensor::empty(dev, vec![nq, sq], DType::F32)?;
    let bl     = Tensor::empty(dev, vec![nq, sq], DType::F32)?;
    let o_blk  = Tensor::empty(dev, vec![nq, sq, hd], DType::F16)?;
    let vt     = Tensor::empty(dev, vec![nkv, hd, bk], DType::F16)?;
    // PERF FIX (was `[nkv, n_kv, hd]`, one whole-causal-prefix f32->f16
    // conversion upfront every call): for sliding-window layers (`win != 0`)
    // the vast majority of KV blocks have `q_cnt == 0` below (no row in this
    // chunk can reach them) -- the QKᵀ/PV GEMMs were already correctly
    // skipped for those blocks, but `kh` was still sized (and fully
    // converted) to the ENTIRE causal prefix on every prefill chunk, and the
    // softmax/merge dispatches ran unconditionally per block regardless of
    // `q_cnt`. For a windowed layer deep into a long prefill (`base` large)
    // that is an O(base) cost paid by EVERY chunk on every windowed layer
    // (Laguna: 36 of 48 layers are sliding-window) -- the root cause of this
    // engine's prefill throughput decaying with sequence length instead of
    // staying roughly flat. Fix: convert K per-block into a `bk`-sized
    // rolling buffer (mirroring `vt`/`SDPA_TC_VPREP_SRC`, which already
    // worked this way), and gate kprep+softmax+merge on the same `q_cnt>0`
    // check as the GEMMs -- a block no row can reach contributes nothing
    // either way (softmax would derive `bm=-inf`/`bl=0` for every row via
    // its own positional mask; `merge`'s `l_b <= 0` early-return already
    // no-ops on that combination), so skipping all four dispatches together
    // is bit-identical to running them, just without the wasted work.
    let kh     = Tensor::empty(dev, vec![nkv, bk, hd], DType::F16)?;

    let el = 2i64;
    let mut kb0 = 0usize;
    while kb0 < n_kv {
        let blk = bk.min(n_kv - kb0);
        // Query-range skip: restrict the expensive QKᵀ/PV tensor-core GEMMs
        // to the rows that can possibly attend anywhere in KV block [kb0,
        // kb0+blk), the cheap full-range softmax/merge below still apply
        // the real per-element mask, so a range that's a SUPERSET of the
        // true set is always safe (the extra rows just mask to p=0); a
        // range narrower than the true set would silently drop attention
        // mass. Only ever shrink, never tighten past that guarantee.
        //
        // Packed-segment mode (seg_len == bk): each KV-block is exactly one
        // packed segment, so only that segment's own rows [kb0, kb0+blk)
        // attend to it, row index and absolute KV position coincide in
        // this mode (packed sequences are laid out from base_kv=0).
        //
        // Sliding-window mode (win != 0): causal means query absolute
        // position i_abs can reach KV position kb0 (the block's first
        // index) only if i_abs >= kb0. The window means i_abs can reach KV
        // position kb0+blk-1 (the block's last index) only if that index is
        // still inside the window, i.e. seg_lo[i] = i_abs - win + 1 <=
        // kb0+blk-1, i.e. i_abs < kb0 + blk + win. So the absolute-position
        // range that can touch this block is [kb0, kb0+blk+win). Row `r` of
        // this call sits at absolute position `base + r` (base_kv is this
        // chunk's starting position), so convert that absolute range to a
        // row-index range and clamp into [0, sq).
        let (q_lo, q_cnt) = if win != 0 {
            let lo_abs = kb0 as i64;
            let hi_abs_excl = kb0 as i64 + blk as i64 + win as i64;
            let lo_r = (lo_abs - base as i64).max(0);
            let hi_r = (hi_abs_excl - base as i64).min(sq as i64);
            if hi_r > lo_r { (lo_r as usize, (hi_r - lo_r) as usize) } else { (0, 0) }
        } else if seg_len != 0 && seg_len as usize == bk {
            (kb0, blk)
        } else {
            (0, sq)
        };
        // q_cnt == 0: no row in this chunk can reach this block at all (the
        // derivation above is a superset of the true reachable set, so an
        // empty computed range proves the true set is also empty). Skip
        // kprep, the QKᵀ/PV GEMMs, vprep, softmax AND merge entirely (all
        // four moved inside this same guard -- see the PERF FIX note above
        // `kh`'s allocation for why this is bit-identical, not just faster).
        if q_cnt > 0 {
            dev.dispatch_raw_cuda(SDPA_TC_KPREP_SRC, "sdpa_tc_kprep.cu", "sdpa_tc_kprep",
                &[(k.buffer.as_ref(), k.offset), (kh.buffer.as_ref(), 0)],
                &[i(nkv as i32), i(kv_stride as i32), i(kb0 as i32), i(blk as i32), i(hd as i32), i(in_f32)],
                blk256(nkv*blk*hd), [256,1,1], 0, false)?;
            for g in 0..nkv {
                let q_off = g * hpg * sq * hd * 2 + q_lo * hd * 2;
                let k_off = g * blk * hd * 2;
                let s_off = g * hpg * sq * blk * 2 + q_lo * blk * 2;
                dev.gemm_strided_batched_off(
                    qh.buffer.as_ref(),     q_off, (sq*hd) as i64 * el,
                    kh.buffer.as_ref(),     k_off, 0,
                    scores.buffer.as_ref(), s_off, (sq*blk) as i64 * el,
                    q_cnt, blk, hd, hpg, DType::F16)?;
            }
            // varlen softmax: causal upper (base+r) AND segment lower (seg_lo[r]).
            dev.dispatch_raw_cuda(SDPA_TC_SOFTMAX_VARLEN_SRC, "sdpa_tc_softmax_varlen.cu", "sdpa_tc_softmax_varlen",
                &[(scores.buffer.as_ref(), 0), (p_blk.buffer.as_ref(), 0),
                  (bm.buffer.as_ref(), 0), (bl.buffer.as_ref(), 0),
                  (seg_lo.buffer.as_ref(), seg_lo.offset)],
                &[i(nq as i32), i(sq as i32), i(blk as i32), i(kb0 as i32), i(base as i32)],
                [(nq*sq) as u32, 1, 1], [256,1,1], 0, false)?;
            dev.dispatch_raw_cuda(SDPA_TC_VPREP_SRC, "sdpa_tc_vprep.cu", "sdpa_tc_vprep",
                &[(v.buffer.as_ref(), v.offset), (vt.buffer.as_ref(), 0)],
                &[i(nkv as i32), i(kv_stride as i32), i(hd as i32), i(kb0 as i32), i(blk as i32), i(in_f32)],
                blk256(nkv*blk*hd), [256,1,1], 0, false)?;
            for g in 0..nkv {
                let p_off = g * hpg * sq * blk * 2 + q_lo * blk * 2;
                let v_off = g * hd * blk * 2;
                let o_off = g * hpg * sq * hd * 2 + q_lo * hd * 2;
                dev.gemm_strided_batched_off(
                    p_blk.buffer.as_ref(), p_off, (sq*blk) as i64 * el,
                    vt.buffer.as_ref(),    v_off, 0,
                    o_blk.buffer.as_ref(), o_off, (sq*hd) as i64 * el,
                    q_cnt, hd, blk, hpg, DType::F16)?;
            }
            dev.dispatch_raw_cuda(SDPA_TC_MERGE_SRC, "sdpa_tc_merge.cu", "sdpa_tc_merge",
                &[(o_blk.buffer.as_ref(), 0), (bm.buffer.as_ref(), 0), (bl.buffer.as_ref(), 0),
                  (o_run.buffer.as_ref(), 0), (m_run.buffer.as_ref(), 0), (l_run.buffer.as_ref(), 0)],
                &[i(nq as i32), i(sq as i32), i(hd as i32)],
                [(nq*sq) as u32, 1, 1], [128,1,1], 0, false)?;
        }
        kb0 += blk;
    }
    let out = Tensor::empty(dev, vec![sq, nq, hd], q.dtype)?;
    let out_f32 = if q.dtype == DType::F32 { 1i32 } else { 0 };
    dev.dispatch_raw_cuda(SDPA_TC_FINALIZE_SRC, "sdpa_tc_finalize.cu", "sdpa_tc_finalize",
        &[(o_run.buffer.as_ref(), 0), (l_run.buffer.as_ref(), 0), (out.buffer.as_ref(), 0)],
        &[i(nq as i32), i(sq as i32), i(hd as i32), i(out_f32)],
        blk256(nq*sq*hd), [256,1,1], 0, false)?;
    Ok(out)
}

/// Raw CUDA source for the **chunked parallel SSD prefill scan** for the
/// NemotronH-Nano-30B Mamba2 cell (Dh=64, Ds=128, H=64, G=8, n_per_t=4).
///
/// The existing `ssm_step_record_d64_128_64_8` runs a serial loop over T
/// positions per (h, dh) cell, the serial chain of length T is the bottleneck.
/// This kernel replaces it with a **two-pass parallel segment scan**:
///
/// **Pass 1 (parallel)**: `N_SEG` warp-segments each process `T/N_SEG` positions
///   independently (assuming initial state = 0), computing a "partial state"
///   contribution and the segment's decay product `alpha_seg`.
///
/// **Pass 2 (sequential, warp 0 only)**: Stitch the true initial state for each
///   segment by walking the T/N_SEG chain: `s_in[k+1] = alpha_seg[k] * s_in[k] + partial[k]`.
///
/// **Pass 3 (parallel)**: Each segment re-computes its positions with the true
///   `s_in[seg]` and emits the correct `y[t]` values.
///
/// **Complexity**: 2T work (passes 1+3) + T/N_SEG serial (pass 2). At N_SEG=32,
/// depth drops from T=2048 to T/32=64. Speedup ≈ 32× for pure serial bottleneck.
///
/// **Memory**: Shared: `N_SEG × n_per_t × WARP` = 32×4×32 = 4096 f32 per block ≈ 16KB.
/// Block size: `[WARP × N_SEG, 1, 1]`; Grid3D: `[1, Dh, H]` (same as sequential).
///
/// **Dimensions fixed for NemotronH**: Dh=64, Ds=128, H=64, G=8, n_per_t=4.
/// The Ds/warp = 128/32 = 4 state values per lane (= n_per_t). G/H ratio = 8/64 = 1/8.
const SSM_CHUNKED_SCAN_SRC: &str = r#"
#include <cuda_fp16.h>
// Note: math.h / cmath not needed for NVRTC device code.
// expf, __shfl_xor_sync etc. are CUDA intrinsics, available without includes.

// NemotronH-Nano Mamba2 cell constants (hardcoded for performance).
#define DH      64u   // head_dim
#define DS      128u  // state_dim
#define NH      64u   // n_heads
#define NG      8u    // n_groups (G = H/ratio, ratio = NH/NG = 8)
#define NPT     4u    // n_per_t = DS / WARP (128/32 = 4 state slots per lane)
#define WARP    32u

// N_SEG segments per (h, dh) cell. Must divide T cleanly.
// 32 segments → at T=2048 each segment handles 64 steps; T=512 → 16 steps.
// Shared mem = N_SEG * NPT * WARP * 4 = 32*4*32*4 = 16KB + alpha (128B) + s_in (512B) per block.
// Block size = WARP * N_SEG = 32*32 = 1024 (CUDA max threads/block limit).
#define N_SEG   32u

// Total threads per block: WARP * N_SEG = 32 * 32 = 1024.
// Grid: [1, DH, NH] = [1, 64, 64] = 4096 blocks (same as sequential kernel).

extern "C" __global__ __launch_bounds__(WARP * N_SEG)
void ssm_chunked_prefill(
    const float*  __restrict__ x,        // [T, NH, DH]
    const float*  __restrict__ a_log,    // [NH]
    const float*  __restrict__ b_mat,    // [T, NG, DS]
    const float*  __restrict__ c_mat,    // [T, NG, DS]
    const float*  __restrict__ d_skip,   // [NH]
    const float*  __restrict__ dt,       // [T, NH]
    const float*  __restrict__ state_in, // [NH, DH, DS]
    float*        __restrict__ y,        // [T, NH, DH]
    float*        __restrict__ state_out,// [NH, DH, DS]
    unsigned int  t_total)
{
    // Each block owns one (d_idx, h_idx) cell.
    const unsigned int d_idx  = blockIdx.y;           // ∈ [0, DH)
    const unsigned int h_idx  = blockIdx.z;           // ∈ [0, NH)
    const unsigned int g_idx  = h_idx / (NH / NG);   // group index ∈ [0, NG)
    const unsigned int lane   = threadIdx.x & (WARP - 1u);  // ∈ [0, 32)
    const unsigned int seg_id = threadIdx.x / WARP;         // ∈ [0, N_SEG)

    // Shared memory: partial state contribution for each segment × state slots.
    // Layout: [N_SEG, NPT] where NPT lanes × each 1 float = 128 floats/segment.
    __shared__ float sh_partial[N_SEG * NPT * WARP];  // seg → [NPT, WARP] colmajor
    __shared__ float sh_alpha[N_SEG];                  // seg decay product
    __shared__ float sh_s_in[NPT * WARP];              // true initial state for each seg (broadcast)

    // Chunk parameters.
    const unsigned int T = t_total;
    const unsigned int seg_len = (T + N_SEG - 1u) / N_SEG;  // positions per segment (~T/N_SEG)
    const unsigned int t_start = seg_id * seg_len;
    const unsigned int t_end   = (t_start + seg_len < T) ? (t_start + seg_len) : T;

    // A = -exp(A_log[h]) (constant for this cell).
    const float a_neg = -expf(a_log[h_idx]);

    // State-slice offsets for this (h, d_idx) cell.
    const unsigned int state_base = (h_idx * DH + d_idx) * DS;

    // ── Pass 1: Compute partial state and alpha for this segment ─────────────
    // Assume state_in = 0. Walk t_start..t_end updating local state in registers.
    // Each lane owns NPT=4 consecutive Ds slots: lane*NPT .. lane*NPT+3.
    float local_s[NPT] = {0.f, 0.f, 0.f, 0.f};  // partial state (assuming s_in=0)
    float seg_alpha = 1.f;   // product of dA over [t_start, t_end)

    for (unsigned int t = t_start; t < t_end; ++t) {
        const unsigned int bt_h = t * NH + h_idx;
        const unsigned int bt_g = t * NG + g_idx;
        const float dt_v   = dt[bt_h];
        const float d_a    = expf(a_neg * dt_v);
        const float x_v    = x[bt_h * DH + d_idx];

        seg_alpha *= d_a;

        // Update local state and accumulate y later (will redo in pass 3).
        for (unsigned int i = 0u; i < NPT; ++i) {
            const unsigned int s_idx = lane * NPT + i;
            const float b_v = b_mat[bt_g * DS + s_idx];
            local_s[i] = d_a * local_s[i] + dt_v * b_v * x_v;
        }
    }

    // Write partial state to shared memory: sh_partial[seg_id, lane, i].
    for (unsigned int i = 0u; i < NPT; ++i) {
        sh_partial[(seg_id * NPT + i) * WARP + lane] = local_s[i];
    }
    if (lane == 0u) sh_alpha[seg_id] = seg_alpha;

    __syncthreads();

    // ── Pass 2: Sequential stitch (only seg 0 of each block = warp 0) ─────
    // Walk segments 0..N_SEG sequentially to compute true s_in for each segment.
    // Seg 0's s_in is the model's real state_in (loaded from global memory).
    if (seg_id == 0u) {
        // Load true initial state into sh_s_in (seg 0's start = state_in).
        for (unsigned int i = 0u; i < NPT; ++i) {
            sh_s_in[i * WARP + lane] = state_in[state_base + (lane * NPT + i)];
        }
        // Seg 0: its partial result already assumed s_in=0; add the real s_in×alpha.
        // (We store the TRUE s_in for seg k, then update to seg k+1's s_in.)
        for (unsigned int seg = 0u; seg < N_SEG; ++seg) {
            const float alpha_s = sh_alpha[seg];
            // True s_in for next seg = alpha_s * true_s_in[seg] + partial[seg].
            for (unsigned int i = 0u; i < NPT; ++i) {
                float s_in_val = sh_s_in[i * WARP + lane];
                float partial  = sh_partial[(seg * NPT + i) * WARP + lane];
                // The corrected partial for this seg: partial + alpha_s * s_in_val
                // We store the NEXT seg's s_in (= alpha_s * s_in_val + partial).
                sh_s_in[i * WARP + lane] = alpha_s * s_in_val + partial;
                // Overwrite partial with corrected carry for pass 3 to read.
                sh_partial[(seg * NPT + i) * WARP + lane] = s_in_val;
            }
        }
        // sh_s_in now holds the true final state after all T positions.
    }

    __syncthreads();

    // ── Pass 3: Recompute with true s_in, emit y ────────────────────────
    // sh_partial[seg_id * NPT + i, lane] now holds the TRUE s_in for this seg.
    for (unsigned int i = 0u; i < NPT; ++i) {
        local_s[i] = sh_partial[(seg_id * NPT + i) * WARP + lane];
    }

    for (unsigned int t = t_start; t < t_end; ++t) {
        const unsigned int bt_h = t * NH + h_idx;
        const unsigned int bt_g = t * NG + g_idx;
        const float dt_v   = dt[bt_h];
        const float d_a    = expf(a_neg * dt_v);
        const float x_v    = x[bt_h * DH + d_idx];

        float y_acc = 0.f;
        for (unsigned int i = 0u; i < NPT; ++i) {
            const unsigned int s_idx = lane * NPT + i;
            const float b_v = b_mat[bt_g * DS + s_idx];
            const float c_v = c_mat[bt_g * DS + s_idx];
            local_s[i] = d_a * local_s[i] + dt_v * b_v * x_v;
            y_acc += c_v * local_s[i];
        }
        // Warp-reduce y_acc across all 32 lanes.
        for (unsigned int mask = WARP >> 1u; mask > 0u; mask >>= 1u)
            y_acc += __shfl_xor_sync(0xffffffffu, y_acc, mask);

        if (lane == 0u) {
            const float d_v = d_skip[h_idx];
            y[bt_h * DH + d_idx] = y_acc + x_v * d_v;
        }
    }

    // ── Write state_out ───────────────────────────────────────────────────
    // Only warp 0 (seg_id=0, lane=all) holds the FINAL state in sh_s_in.
    if (seg_id == 0u) {
        for (unsigned int i = 0u; i < NPT; ++i) {
            state_out[state_base + (lane * NPT + i)] = sh_s_in[i * WARP + lane];
        }
    }
}
"#;

/// Mamba2 SSD **chunked parallel prefill scan** for the NemotronH cell
/// (Dh=64, Ds=128, H=64, G=8). Replaces the sequential-in-T `ssm_prefill_scan`
/// with a two-pass parallel segment scan that drops the serial depth from T to
/// T/N_SEG=T/64. Gated by `NEMOTRON_CHUNKED_SCAN=1`; correctness gate is the
/// same `NEMOTRON_PREFILL_CHECK` comparison against sequential.
///
/// Grid3D: `[1, Dh=64, H=64]`, block `[WARP × N_SEG]` = `[32 × 32 = 1024]`.
/// Shared mem per block: ~17KB (partial states + alpha + s_in scratch).
///
/// Returns `(state_out, y)`, same signature as `ssm_prefill_scan`.
#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_scan_chunked(
    dev: &dyn Device,
    x: &Tensor,
    a_log: &Tensor,
    b_mat: &Tensor,
    c_mat: &Tensor,
    d_skip: &Tensor,
    dt: &Tensor,
    state_in: &Tensor,
    t_total: u32,
    dh: u32,
    ds: u32,
    n_heads: u32,
    n_groups: u32,
) -> Result<(Tensor, Tensor)> {
    if (dh, ds, n_heads, n_groups) != (64, 128, 64, 8) {
        return Err(Error::Msg(format!(
            "ssm_prefill_scan_chunked: only Nemotron cell (64,128,64,8) wired, got ({dh},{ds},{n_heads},{n_groups})"
        )));
    }
    let t = t_total as usize;
    let (nh, dhu, _dsu) = (n_heads as usize, dh as usize, ds as usize);
    let y = Tensor::empty(dev, vec![t * nh * dhu], x.dtype)?;
    let state_out = Tensor::empty(dev, state_in.shape.clone(), x.dtype)?;
    // N_SEG must divide t_total (or kernel pads). 64 divides 512,2048,8192.
    // For t_total not divisible by 64, pass as-is: the kernel uses a guard.
    let t_bytes = t_total.to_le_bytes().to_vec();
    // Grid: [1, Dh, H]. Block: [WARP * N_SEG] = [32 * 64] = [2048].
    let grid = [1u32, dh, n_heads];
    let block = [32u32 * 32, 1, 1]; // WARP * N_SEG = 32*32 = 1024 (CUDA max threads/block)
    // Shared mem: N_SEG * NPT * WARP * 4 + N_SEG * 4 + NPT * WARP * 4
    //           = 32*4*32*4 + 32*4 + 4*32*4 = 16384 + 128 + 512 = 17024 bytes
    let shared_bytes: u32 = 32 * 4 * 32 * 4 + 32 * 4 + 4 * 32 * 4;
    // Cast inputs to f32 for the raw CUDA kernel (which reads f32).
    // The existing sequential kernel also accumulates in f32.
    // NOTE: if x/state are f16/bf16 we need casts. For now, only f32 is wired.
    if x.dtype != DType::F32 {
        return Err(Error::Msg("ssm_prefill_scan_chunked: only F32 input wired (cast first)".into()));
    }
    dev.dispatch_raw_cuda(
        SSM_CHUNKED_SCAN_SRC,
        "ssm_chunked_prefill.cu",
        "ssm_chunked_prefill",
        &[
            (x.buffer.as_ref(), 0),
            (a_log.buffer.as_ref(), 0),
            (b_mat.buffer.as_ref(), 0),
            (c_mat.buffer.as_ref(), 0),
            (d_skip.buffer.as_ref(), 0),
            (dt.buffer.as_ref(), 0),
            (state_in.buffer.as_ref(), 0),
            (y.buffer.as_ref(), 0),
            (state_out.buffer.as_ref(), 0),
        ],
        &[t_bytes],
        grid,
        block,
        shared_bytes,
        false,
    )?;
    Ok((state_out, y))
}

/// Mamba2 SSD **batched-prefill scan**, runs the sequential SSD forward over
/// all `t_total` prompt tokens in ONE dispatch, emitting every per-token
/// `y[t]` plus the final recurrent `state_out` (for decode continuity).
/// Dispatches `ssm_step_record_d64_128_64_8` (Nemotron cell: Dh=64, Ds=128,
/// H=64, G=8). The `(da_log, dbx_log)` tapes are written but ignored here
/// (they exist for the speculative-rollback replay path). Layouts:
/// `x [t·H·Dh]`, `a_log/d [H]`, `dt [t·H]`, `b/c [t·G·Ds]`,
/// `state_in/out [H·Dh·Ds]`, `y [t·H·Dh]`. Grid3D `[1, Dh, H]`, tg `[32,1,1]`.
#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_scan(
    dev: &dyn Device,
    x: &Tensor,
    a_log: &Tensor,
    b_mat: &Tensor,
    c_mat: &Tensor,
    d_skip: &Tensor,
    dt: &Tensor,
    state_in: &Tensor,
    t_total: u32,
    dh: u32,
    ds: u32,
    n_heads: u32,
    n_groups: u32,
) -> Result<(Tensor, Tensor)> {
    use wh_iron_core::ir::KernelMode;
    if (dh, ds, n_heads, n_groups) != (64, 128, 64, 8) {
        return Err(Error::Msg(format!(
            "ssm_prefill_scan: only Nemotron cell (64,128,64,8) wired, got ({dh},{ds},{n_heads},{n_groups})"
        )));
    }
    let t = t_total as usize;
    let (nh, dhu, dsu) = (n_heads as usize, dh as usize, ds as usize);
    let y = Tensor::empty(dev, vec![t * nh * dhu], x.dtype)?;
    let state_out = Tensor::empty(dev, state_in.shape.clone(), x.dtype)?;
    // Tape outputs, written by the kernel, unused by prefill.
    let da_log = Tensor::empty(dev, vec![t * nh * dsu], x.dtype)?;
    let dbx_log = Tensor::empty(dev, vec![t * nh * dhu * dsu], x.dtype)?;
    // mask buffer: has_mask=0 → kernel ignores contents, but the binding must exist.
    let mask = Tensor::new(dev.alloc_zeroed(t * 4).unwrap(), vec![t], DType::U32);
    let kern = cached_ir("ssm_step_record_d64_128_64_8", x.dtype, || {
        let mut kk = wh_iron_std::kernels::ssm::ssm_replay::iron_ssm_step_record_d64_128_64_8::kernel_ir_for(x.dtype);
        kk.mode = KernelMode::Grid3D;
        kk
    });
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let grid = Grid { grid: [1, dh, n_heads], block: [32, 1, 1] };
    dev.dispatch(
        &kern,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(a_log.buffer.clone()),
            Binding::Buffer(b_mat.buffer.clone()),
            Binding::Buffer(c_mat.buffer.clone()),
            Binding::Buffer(d_skip.buffer.clone()),
            Binding::Buffer(dt.buffer.clone()),
            Binding::Buffer(state_in.buffer.clone()),
            Binding::Buffer(mask.buffer.clone()),
            Binding::Buffer(y.buffer.clone()),
            Binding::Buffer(state_out.buffer.clone()),
            Binding::Buffer(da_log.buffer.clone()),
            Binding::Buffer(dbx_log.buffer.clone()),
            u(t_total),
            u(0), // has_mask
        ],
        grid,
    )?;
    Ok((state_out, y))
}

/// DeepSeek-V4 MLA decode attention: d512 SDPA with a per-head learnable
/// **attention sink** (a virtual logit that extends the softmax denominator).
/// `q` is `[n_q_heads, 512]`; `k`/`v` are the latent KV cache
/// `[n_kv_heads, kv_stride, 512]`; `sink_logit` is `[n_q_heads]` f32.
/// Dispatches `iron_sdpa_decode_d512_sink` (tg 512).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_sink(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sink_logit: &Tensor,
    n_kv: u32,
    kv_stride: u32,
    heads_per_group: u32,
    scale: f32,
) -> Result<Tensor> {
    const HD: usize = 512;
    let n_q_heads = q.elem_count() / HD;
    let out = Tensor::empty(dev, vec![n_q_heads, HD], q.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let kern = lookup("iron_sdpa_decode_d512_sink", q.dtype)?;
    let bindings = vec![
        Binding::Buffer(q.buffer.clone()),
        Binding::Buffer(k.buffer.clone()),
        Binding::Buffer(v.buffer.clone()),
        Binding::Buffer(sink_logit.buffer.clone()),
        Binding::Buffer(out.buffer.clone()),
        u(HD as u32),
        u(n_kv),
        u(kv_stride),
        u(heads_per_group),
        Binding::Scalar(scale.to_le_bytes().to_vec()),
    ];
    let grid = Grid { grid: [n_q_heads as u32, 1, 1], block: [512, 1, 1] };
    dev.dispatch(&kern, &bindings, grid)?;
    Ok(out)
}

/// Decode-time scaled-dot-product attention for a single query token.
///
/// Layout (matching the kernel): `q` is `[n_q_heads, head_dim]`; `k`/`v` are
/// `[n_kv_heads, kv_stride, head_dim]` (kv cache, `kv_stride` = capacity,
/// `n_kv` = filled length); `kv_head = q_head / heads_per_group` (GQA). No
/// attention sink / sliding window (dense causal). Picks the head-dim
/// specialized kernel variant. Output is `[n_q_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_kv: u32,
    kv_stride: u32,
    heads_per_group: u32,
    scale: f32,
) -> Result<Tensor> {
    let n_q_heads = q.elem_count() / head_dim;
    let out = Tensor::empty(dev, vec![n_q_heads, head_dim], q.dtype)?;

    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());

    // Per-variant trailing constexprs (everything after head_dim, n_kv,
    // kv_stride, heads_per_group). Dense path → sink/window disabled.
    let (name, block, trailing): (&str, u32, Vec<Binding>) = match head_dim {
        64 => ("iron_sdpa_decode_d64", 1024, vec![u(0), f(0.0), f(scale)]), // has_sink, sink_logit, scale
        96 => ("iron_sdpa_decode_d96", 1024, vec![f(scale)]),
        128 => ("iron_sdpa_decode", 1024, vec![u(0), u(0), u(0), f(0.0), f(scale)]), // sink_end, window_start, has_sink, sink_logit, scale
        256 => ("iron_sdpa_decode_d256", 1024, vec![u(0), f(0.0), f(scale)]),
        512 => ("iron_sdpa_decode_d512", 512, vec![f(scale)]),
        _ => return Err(Error::Msg(format!("sdpa_decode: unsupported head_dim {head_dim}"))),
    };

    let kern = lookup(name, q.dtype)?;
    let mut bindings = vec![
        Binding::Buffer(q.buffer.clone()),
        Binding::Buffer(k.buffer.clone()),
        Binding::Buffer(v.buffer.clone()),
        Binding::Buffer(out.buffer.clone()),
        u(head_dim as u32),
        u(n_kv),
        u(kv_stride),
        u(heads_per_group),
    ];
    bindings.extend(trailing);
    let grid = Grid { grid: [n_q_heads as u32, 1, 1], block: [block, 1, 1] };
    dev.dispatch(&kern, &bindings, grid)?;
    Ok(out)
}

// ── CUDA-graph-capturable dense decode SDPA (raw CUDA, head_dim 128) ────
//
// The registry `iron_sdpa_decode` kernel above takes `n_kv` as a launch-time
// constexpr, fine for eager decode (a fresh launch every token) but wrong
// for a captured graph, which replays the exact same recorded launch (same
// baked `n_kv`) every token. Rather than plumb a buffer-read `n_kv` through
// the iron registry kernel (invasive: `n_kv` there is a `#[constexpr]`
// used to size the unroll-friendly loop bound), this is a hand-written raw
// CUDA fallback that reads `n_kv` from a 1-element `u32` device buffer at
// kernel runtime instead. Same single-pass online-softmax algorithm as the
// registry kernel's dense path (`iron_sdpa_decode` with `sink_end=0,
// window_start=0, has_sink=0`): block per Q-head, 1024 threads (32 warps x
// 32 lanes), each lane owns 4 of the 128 head dims, `__shfl_xor_sync`
// butterfly reduction (broadcasts to every lane, matching MSL's
// all-to-all `simd_sum`/`simd_max`), f32 accumulation throughout. Verified
// against `sdpa_decode`'s output in `crates/backends/wh-butter-cuda/tests/laguna.rs`.
const SDPA_DECODE_NBUF_SRC: &str = r#"
extern "C" __global__ void butter_sdpa_decode_nbuf(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out, const unsigned int* __restrict__ n_kv_buf,
    int head_dim, int kv_stride, int heads_per_group, float scale)
{
    int q_head = blockIdx.x;
    int kv_head = q_head / heads_per_group;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int sg = tid >> 5;
    int ns = blockDim.x >> 5;
    unsigned int n_kv = n_kv_buf[0];

    __shared__ float tg_max[32];
    __shared__ float tg_sum[32];
    __shared__ float tg_out0[1056];
    __shared__ float tg_out1[1056];
    __shared__ float tg_out2[1056];
    __shared__ float tg_out3[1056];

    int q_off = q_head * head_dim;
    long kv_head_base = (long)kv_head * kv_stride * head_dim;
    int d0 = lane * 4;

    float qs[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) qs[i] = q[q_off + d0 + i] * scale;

    float run_max = -3.4028235e38f;
    float run_sum = 0.0f;
    float os[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned int t = sg; t < n_kv; t += ns) {
        long base = kv_head_base + (long)t * head_dim;
        long kv0 = base + d0;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) partial += qs[i] * k[kv0 + i];
        for (int mask = 16; mask > 0; mask >>= 1) partial += __shfl_xor_sync(0xffffffffu, partial, mask);
        float score = partial;
        float new_max = fmaxf(score, run_max);
        float factor = expf(run_max - new_max);
        float weight = expf(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        #pragma unroll
        for (int i = 0; i < 4; i++) os[i] = os[i] * factor + weight * v[kv0 + i];
    }

    if (lane == 0) { tg_max[sg] = run_max; tg_sum[sg] = run_sum; }
    __syncthreads();
    if (sg == 0) {
        float g_max_raw = (lane < ns) ? tg_max[lane] : -3.4028235e38f;
        float g_max = g_max_raw;
        for (int mask = 16; mask > 0; mask >>= 1) g_max = fmaxf(g_max, __shfl_xor_sync(0xffffffffu, g_max, mask));
        float g_sum_in = (lane < ns) ? tg_sum[lane] * expf(g_max_raw - g_max) : 0.0f;
        float g_sum = g_sum_in;
        for (int mask = 16; mask > 0; mask >>= 1) g_sum += __shfl_xor_sync(0xffffffffu, g_sum, mask);
        if (lane == 0) { tg_max[0] = g_max; tg_sum[0] = g_sum; }
    }
    __syncthreads();
    float g_max = tg_max[0];
    float g_sum = tg_sum[0];
    float rescale = (g_sum > 0.0f) ? expf(run_max - g_max) / g_sum : 0.0f;
    int stride = ns + 1;
    int idx = lane * stride + sg;
    tg_out0[idx] = os[0] * rescale;
    tg_out1[idx] = os[1] * rescale;
    tg_out2[idx] = os[2] * rescale;
    tg_out3[idx] = os[3] * rescale;
    __syncthreads();
    if (sg == 0) {
        float so0 = 0.0f, so1 = 0.0f, so2 = 0.0f, so3 = 0.0f;
        for (int g = 0; g < ns; g++) {
            int ri = lane * stride + g;
            so0 += tg_out0[ri];
            so1 += tg_out1[ri];
            so2 += tg_out2[ri];
            so3 += tg_out3[ri];
        }
        int out_off = q_off + d0;
        out[out_off]     = so0;
        out[out_off + 1] = so1;
        out[out_off + 2] = so2;
        out[out_off + 3] = so3;
    }
}
"#;

/// CUDA-graph-capturable fallback of [`sdpa_decode`] for `head_dim == 128`
/// (Laguna's only head_dim): reads the filled KV length `n_kv` from a
/// 1-element `u32` device buffer instead of a baked-in launch scalar. Dense
/// causal only (no sink / sliding-window mask), Laguna's sliding layers
/// bound the KV walk via the buffer's VALUE (the ring's current filled
/// length), so a separate window mask isn't needed. f32 only. See the
/// kernel-source doc above for the algorithm.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_nbuf(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_kv_buf: &Tensor,
    kv_stride: u32,
    heads_per_group: u32,
    scale: f32,
) -> Result<Tensor> {
    if head_dim != 128 {
        return Err(Error::Msg(format!("sdpa_decode_nbuf: only head_dim 128 is implemented, got {head_dim}")));
    }
    let n_q_heads = q.elem_count() / head_dim;
    let out = Tensor::empty(dev, vec![n_q_heads, head_dim], DType::F32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(
        SDPA_DECODE_NBUF_SRC, "sdpa_decode_nbuf.cu", "butter_sdpa_decode_nbuf",
        &[
            (q.buffer.as_ref(), q.offset),
            (k.buffer.as_ref(), k.offset),
            (v.buffer.as_ref(), v.offset),
            (out.buffer.as_ref(), 0),
            (n_kv_buf.buffer.as_ref(), n_kv_buf.offset),
        ],
        &[i(head_dim as i32), i(kv_stride as i32), i(heads_per_group as i32), f(scale)],
        [n_q_heads as u32, 1, 1], [1024, 1, 1], 0, false,
    )?;
    Ok(out)
}

/// GQA-aware split-K flash-decode SDPA (ported from a reference ML framework's `sdpa_vector_2pass`, head_dim
/// 128). Pass 1: grid `(n_kv_heads, blocks)`, block `(32, gqa_factor)`: one
/// simdgroup per Q-head of the group, so the `gqa_factor` heads SHARE each K/V
/// load (the single-pass kernel re-reads the shared KV head `gqa_factor`×, at
/// 32K that's the dominant cost). Each block-row strides a `1/blocks` slice of
/// the KV positions, emitting per-block (max, sum, partial_o). Pass 2: grid
/// `(n_q_heads)`, block 1024: online-softmax merge of the `blocks` partials.
/// `blocks` MUST be a multiple of 32 (pass-2 reducer constraint). Kernel math
/// is registry-validated (`test_butter_sdpa_decode_2pass_combined`, f32/f16/bf16).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_2pass(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_kv: u32,
    kv_stride: u32,
    gqa_factor: u32,
    scale: f32,
    blocks: u32,
) -> Result<Tensor> {
    use wh_iron_core::ir::KernelMode;
    let n_q_heads = q.elem_count() / head_dim;
    let n_kv_heads = n_q_heads / gqa_factor as usize;
    let nb = blocks as usize;
    let partial_o = Tensor::empty(dev, vec![n_q_heads * nb * head_dim], q.dtype)?;
    let partial_m = Tensor::empty(dev, vec![n_q_heads * nb], DType::F32)?;
    let partial_l = Tensor::empty(dev, vec![n_q_heads * nb], DType::F32)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());

    let k1 = cached_ir("sdpa_decode_2pass_pass1", q.dtype, || {
        let mut k = wh_iron_std::kernels::sdpa::sdpa_decode_2pass::iron_sdpa_decode_2pass_pass1::kernel_ir_for(q.dtype);
        k.mode = KernelMode::Reduction;
        k
    });
    dev.dispatch(&k1, &[
        Binding::Buffer(q.buffer.clone()), Binding::Buffer(k.buffer.clone()), Binding::Buffer(v.buffer.clone()),
        Binding::Buffer(partial_o.buffer.clone()), Binding::Buffer(partial_m.buffer.clone()), Binding::Buffer(partial_l.buffer.clone()),
        u(head_dim as u32), u(n_kv), u(kv_stride), u(gqa_factor), u(blocks), f(scale),
    ], Grid { grid: [n_kv_heads as u32, blocks, 1], block: [32 * gqa_factor, 1, 1] })?;

    let out = Tensor::empty(dev, vec![n_q_heads, head_dim], q.dtype)?;
    let k2 = cached_ir("sdpa_decode_2pass_pass2", q.dtype, || {
        let mut k = wh_iron_std::kernels::sdpa::sdpa_decode_2pass::iron_sdpa_decode_2pass_pass2::kernel_ir_for(q.dtype);
        k.mode = KernelMode::Reduction;
        k
    });
    dev.dispatch(&k2, &[
        Binding::Buffer(partial_o.buffer.clone()), Binding::Buffer(partial_m.buffer.clone()), Binding::Buffer(partial_l.buffer.clone()),
        Binding::Buffer(out.buffer.clone()), u(head_dim as u32), u(blocks),
    ], Grid { grid: [n_q_heads as u32, 1, 1], block: [1024, 1, 1] })?;
    Ok(out)
}

/// Like `sdpa_decode_2pass` but uses the TILED pass-1 variant: assigns each
/// block a contiguous chunk of KV positions for L2-cache-friendly sequential
/// access (vs the strided pattern which thrashes L2). Activated by NEMOTRON_TILED=1.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_2pass_tiled(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_kv: u32,
    kv_stride: u32,
    gqa_factor: u32,
    scale: f32,
    blocks: u32,
) -> Result<Tensor> {
    use wh_iron_core::ir::KernelMode;
    let n_q_heads = q.elem_count() / head_dim;
    let n_kv_heads = n_q_heads / gqa_factor as usize;
    let nb = blocks as usize;
    let partial_o = Tensor::empty(dev, vec![n_q_heads * nb * head_dim], q.dtype)?;
    let partial_m = Tensor::empty(dev, vec![n_q_heads * nb], DType::F32)?;
    let partial_l = Tensor::empty(dev, vec![n_q_heads * nb], DType::F32)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());

    let k1 = cached_ir("sdpa_decode_2pass_pass1_tiled", q.dtype, || {
        let mut k = wh_iron_std::kernels::sdpa::sdpa_decode_2pass::iron_sdpa_decode_2pass_pass1_tiled::kernel_ir_for(q.dtype);
        k.mode = KernelMode::Reduction;
        k
    });
    dev.dispatch(&k1, &[
        Binding::Buffer(q.buffer.clone()), Binding::Buffer(k.buffer.clone()), Binding::Buffer(v.buffer.clone()),
        Binding::Buffer(partial_o.buffer.clone()), Binding::Buffer(partial_m.buffer.clone()), Binding::Buffer(partial_l.buffer.clone()),
        u(head_dim as u32), u(n_kv), u(kv_stride), u(gqa_factor), u(blocks), f(scale),
    ], Grid { grid: [n_kv_heads as u32, blocks, 1], block: [32 * gqa_factor, 1, 1] })?;

    let out = Tensor::empty(dev, vec![n_q_heads, head_dim], q.dtype)?;
    let k2 = cached_ir("sdpa_decode_2pass_pass2", q.dtype, || {
        let mut k = wh_iron_std::kernels::sdpa::sdpa_decode_2pass::iron_sdpa_decode_2pass_pass2::kernel_ir_for(q.dtype);
        k.mode = KernelMode::Reduction;
        k
    });
    dev.dispatch(&k2, &[
        Binding::Buffer(partial_o.buffer.clone()), Binding::Buffer(partial_m.buffer.clone()), Binding::Buffer(partial_l.buffer.clone()),
        Binding::Buffer(out.buffer.clone()), u(head_dim as u32), u(blocks),
    ], Grid { grid: [n_q_heads as u32, 1, 1], block: [1024, 1, 1] })?;
    Ok(out)
}

/// Like `sdpa_decode_2pass` but uses the BC=4 pass-1 variant: processes 4 KV
/// positions per loop iteration to expose memory-level parallelism and hide
/// load latency. Same pass-2 reduction. Same correctness guarantees.
/// Activated by NEMOTRON_BC4=1 in the bench harness.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode_2pass_bc4(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_kv: u32,
    kv_stride: u32,
    gqa_factor: u32,
    scale: f32,
    blocks: u32,
) -> Result<Tensor> {
    use wh_iron_core::ir::KernelMode;
    let n_q_heads = q.elem_count() / head_dim;
    let n_kv_heads = n_q_heads / gqa_factor as usize;
    let nb = blocks as usize;
    let partial_o = Tensor::empty(dev, vec![n_q_heads * nb * head_dim], q.dtype)?;
    let partial_m = Tensor::empty(dev, vec![n_q_heads * nb], DType::F32)?;
    let partial_l = Tensor::empty(dev, vec![n_q_heads * nb], DType::F32)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());

    // BC=4 pass 1: 4 positions per loop iter for MLP / load-latency hiding.
    let k1 = cached_ir("sdpa_decode_2pass_pass1_bc4", q.dtype, || {
        let mut k = wh_iron_std::kernels::sdpa::sdpa_decode_2pass::iron_sdpa_decode_2pass_pass1_bc4::kernel_ir_for(q.dtype);
        k.mode = KernelMode::Reduction;
        k
    });
    dev.dispatch(&k1, &[
        Binding::Buffer(q.buffer.clone()), Binding::Buffer(k.buffer.clone()), Binding::Buffer(v.buffer.clone()),
        Binding::Buffer(partial_o.buffer.clone()), Binding::Buffer(partial_m.buffer.clone()), Binding::Buffer(partial_l.buffer.clone()),
        u(head_dim as u32), u(n_kv), u(kv_stride), u(gqa_factor), u(blocks), f(scale),
    ], Grid { grid: [n_kv_heads as u32, blocks, 1], block: [32 * gqa_factor, 1, 1] })?;

    // Pass 2 unchanged: same partial buffer layout, same reduction.
    let out = Tensor::empty(dev, vec![n_q_heads, head_dim], q.dtype)?;
    let k2 = cached_ir("sdpa_decode_2pass_pass2", q.dtype, || {
        let mut k = wh_iron_std::kernels::sdpa::sdpa_decode_2pass::iron_sdpa_decode_2pass_pass2::kernel_ir_for(q.dtype);
        k.mode = KernelMode::Reduction;
        k
    });
    dev.dispatch(&k2, &[
        Binding::Buffer(partial_o.buffer.clone()), Binding::Buffer(partial_m.buffer.clone()), Binding::Buffer(partial_l.buffer.clone()),
        Binding::Buffer(out.buffer.clone()), u(head_dim as u32), u(blocks),
    ], Grid { grid: [n_q_heads as u32, 1, 1], block: [1024, 1, 1] })?;
    Ok(out)
}

/// Run a registered elementwise kernel (Grid3D, one thread per element) with
/// the given ordered bindings, producing a fresh `out`-shaped output.
fn elementwise_kernel(
    dev: &dyn Device,
    name: &str,
    dtype: DType,
    out_shape: Vec<usize>,
    mut bindings: Vec<Binding>,
) -> Result<Tensor> {
    let k = lookup(name, dtype)?;
    let out = Tensor::empty(dev, out_shape, dtype)?;
    bindings.push(Binding::Buffer(out.buffer.clone()));
    let n = out.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    // output binding goes in its signature position, callers pass inputs,
    // we append the single output last (matches a,…,out param order).
    dev.dispatch(&k, &bindings, grid)?;
    Ok(out)
}

/// SiLU activation `out = x * sigmoid(x)`, elementwise. Dispatches `iron_silu`.
pub fn silu(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    elementwise_kernel(dev, "iron_silu", x.dtype, x.shape.clone(), vec![Binding::Buffer(x.buffer.clone())])
}

/// ReLU² activation `out = max(x,0)²`, elementwise: NemotronH MoE expert act.
/// Built inline (Relu → square) so the expert `up → relu² → down` chain stays
/// ON-DEVICE (no host round-trip between the two Q8 GEMVs). Dispatches `iron_relu2`.
pub fn relu2(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    let mut k = Kernel::new("iron_relu2");
    for (pname, is_out) in [("a", false), ("c", true)] {
        k.params.push(Param { name: pname.into(), dtype: x.dtype, shape: Shape::scalar(), is_output: is_out, kind: ParamKind::Tensor });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None }, ValueId::new(1));
    k.body.push_op(Op::Activation { kind: ActKind::Relu, value: ValueId::new(1) }, ValueId::new(2));
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(2) }, ValueId::new(3));
    k.body.push_op_no_result(Op::Store { dst: "c".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], value: ValueId::new(3), mask: None });
    k.mode = wh_iron_core::ir::KernelMode::Elementwise;
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let n = out.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(&k, &[Binding::Buffer(x.buffer.clone()), Binding::Buffer(out.buffer.clone())], grid)?;
    Ok(out)
}

/// Raw CUDA source for the **MoE scatter-add** kernel.
///
/// After the batched down-GEMM produces `dn_out [mt, hid]` (sorted by expert),
/// this kernel scatters each row back to the corresponding token slot in the
/// output accumulator, weighted by the router weight:
///
///   `acc[token_indices[r], h] += dn_out[r, h] * weights[r] * unscale`
///
/// Uses `atomicAdd(float*, float)` (available since CUDA 2.0/GPGPU) so multiple
/// experts for the same token add correctly. Block: `[BLOCK_H, 1]`; Grid:
/// `[mt, ceil(hid/BLOCK_H)]`. For NemotronH hid=2688, BLOCK_H=128: 21 col-blocks.
const MOE_SCATTER_ADD_SRC: &str = r#"
#include <cuda_fp16.h>
#define BLOCK_H 128

extern "C" __global__ void moe_scatter_add_f32(
    const float*   __restrict__ dn,      // [mt, hid] sorted-row expert outputs
    const unsigned* __restrict__ tidx,   // [mt] token index per row
    const float*   __restrict__ wts,     // [mt] router weights
    float*                      acc,     // [s, hid] output accumulator (pre-zeroed)
    int mt, int hid, float unscale)
{
    int r = (int)blockIdx.x;  // sorted row ∈ [0, mt)
    int h = (int)(blockIdx.y * BLOCK_H + threadIdx.x);  // hidden dim
    if (r < mt && h < hid) {
        float w = wts[r] * unscale;
        atomicAdd(&acc[(int)tidx[r] * hid + h], dn[r * hid + h] * w);
    }
}
"#;

/// Device-side MoE scatter-add: write `dn_out[r, h] * weights[r] * unscale`
/// into `acc[token_indices[r], h]` for each row `r` in the sorted batch.
/// `acc` must be pre-zeroed (use `Tensor::zeros`). Returns nothing (in-place).
///
/// Replaces the host scatter loop in the prefill MoE path, enabling fully
/// on-device processing when combined with device gather + batched bm64 GEMMs.
pub fn moe_scatter_add(
    dev: &dyn Device,
    dn: &Tensor,   // [mt, hid] f32 down-GEMM output (sorted)
    tidx: &Tensor, // [mt] u32 token indices
    wts: &Tensor,  // [mt] f32 router weights
    acc: &Tensor,  // [s, hid] f32 accumulator
    mt: usize,
    hid: usize,
    unscale: f32,
) -> Result<()> {
    let butter_i = (mt as i32).to_le_bytes().to_vec();
    let hid_i = (hid as i32).to_le_bytes().to_vec();
    let uscale_f = unscale.to_le_bytes().to_vec();
    const BLOCK_H: u32 = 128;
    let grid = [mt as u32, (hid as u32).div_ceil(BLOCK_H), 1];
    let block = [BLOCK_H, 1, 1];
    dev.dispatch_raw_cuda(
        MOE_SCATTER_ADD_SRC,
        "moe_scatter_add.cu",
        "moe_scatter_add_f32",
        &[
            (dn.buffer.as_ref(),   0),
            (tidx.buffer.as_ref(), 0),
            (wts.buffer.as_ref(),  0),
            (acc.buffer.as_ref(),  0),
        ],
        &[butter_i, hid_i, uscale_f],
        grid,
        block,
        0,
        false,
    )
}

/// Raw CUDA source for the **DETERMINISTIC MoE scatter-add** kernel.
///
/// The atomicAdd variant (`moe_scatter_add_f32`) is run-to-run NONdeterministic:
/// floating-point `atomicAdd` completes in hardware-arbitrated (nondeterministic)
/// order, so when several routed experts for the same token accumulate into the
/// same `acc[token,h]`, the summation order: and thus the low bits of the result
///, varies between identical runs. That jitter propagates through 52 layers and
/// can flip the final argmax at deep context.
///
/// This variant is bit-exact reproducible: one block per `(token, h-block)`, and
/// each thread sums *its* token's contributing rows in a FIXED order, read from a
/// pre-built CSR-style index (`tok_starts[s+1]`, `tok_rows[mt]`). No atomics.
///   `acc[t,h] = Σ_{r ∈ rows(t)}  dn[r,h] * wts[r] * unscale`   (rows in row order)
/// Grid: `[s, ceil(hid/BLOCK_H)]`; block `[BLOCK_H,1]`.
const MOE_SCATTER_ADD_DET_SRC: &str = r#"
#include <cuda_fp16.h>
#define BLOCK_H 128

extern "C" __global__ void moe_scatter_add_det_f32(
    const float*    __restrict__ dn,         // [mt, hid] sorted-row expert outputs
    const int*      __restrict__ tok_starts, // [s+1] CSR row offsets per token
    const int*      __restrict__ tok_rows,   // [mt] row ids grouped by token (fixed order)
    const float*    __restrict__ wts,        // [mt] router weights
    float*                       acc,        // [s, hid] output accumulator (pre-zeroed)
    int s, int hid, float unscale)
{
    int t = (int)blockIdx.x;                              // token ∈ [0, s)
    int h = (int)(blockIdx.y * BLOCK_H + threadIdx.x);    // hidden dim
    if (t < s && h < hid) {
        float sum = 0.f;
        int r0 = tok_starts[t];
        int r1 = tok_starts[t + 1];
        // Fixed-order reduction over this token's rows → deterministic.
        for (int j = r0; j < r1; ++j) {
            int r = tok_rows[j];
            sum += dn[r * hid + h] * (wts[r] * unscale);
        }
        acc[t * hid + h] = sum;
    }
}

// f16-input variant: reads the down-GEMM output directly as __half (no separate
// cast_f16_f32 + [mt,hid] f32 materialization). Same deterministic CSR order.
extern "C" __global__ void moe_scatter_add_det_f16(
    const __half*   __restrict__ dn,
    const int*      __restrict__ tok_starts,
    const int*      __restrict__ tok_rows,
    const float*    __restrict__ wts,
    float*                       acc,
    int s, int hid, float unscale)
{
    int t = (int)blockIdx.x;
    int h = (int)(blockIdx.y * BLOCK_H + threadIdx.x);
    if (t < s && h < hid) {
        float sum = 0.f;
        int r0 = tok_starts[t];
        int r1 = tok_starts[t + 1];
        for (int j = r0; j < r1; ++j) {
            int r = tok_rows[j];
            sum += __half2float(dn[r * hid + h]) * (wts[r] * unscale);
        }
        acc[t * hid + h] = sum;
    }
}

// f16-input fixed-top_k variant: rowpos maps token-major router slots
// (t*top_k+j) to expert-major sorted rows. This skips the token CSR rebuild.
extern "C" __global__ void moe_scatter_add_topk_det_f16(
    const __half*   __restrict__ dn,
    const unsigned* __restrict__ rowpos,
    const float*    __restrict__ wt_token_major,
    float*                       acc,
    int s, int top_k, int hid, float unscale)
{
    int t = (int)blockIdx.x;
    int h = (int)(blockIdx.y * BLOCK_H + threadIdx.x);
    if (t < s && h < hid) {
        float sum = 0.f;
        int base = t * top_k;
        for (int j = 0; j < top_k; ++j) {
            unsigned r = rowpos[base + j];
            sum += __half2float(dn[(int)r * hid + h]) * (wt_token_major[base + j] * unscale);
        }
        acc[t * hid + h] = sum;
    }
}

extern "C" __global__ void moe_scatter_add_topk_det_f16_x2(
    const __half*   __restrict__ dn,
    const unsigned* __restrict__ rowpos,
    const float*    __restrict__ wts_tok,
    float*                       acc,
    int s, int top_k, int hid, float unscale)
{
    int t = (int)blockIdx.x;
    int h = ((int)blockIdx.y * BLOCK_H + threadIdx.x) * 2;
    if (t < s && h < hid) {
        float sum0 = 0.f;
        float sum1 = 0.f;
        int base = t * top_k;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < top_k) {
                int r = (int)rowpos[base + j];
                float w = wts_tok[base + j] * unscale;
                long idx = (long)r * hid + h;
                __half2 hv = *reinterpret_cast<const __half2*>(dn + idx);
                float2 fv = __half22float2(hv);
                sum0 += fv.x * w;
                sum1 += fv.y * w;
            }
        }
        long out = (long)t * hid + h;
        acc[out] = sum0;
        acc[out + 1] = sum1;
    }
}
"#;

/// Deterministic MoE scatter-add (drop-in for [`moe_scatter_add`] on the prefill
/// path). `tidx_host` is the per-row token index (callers already build it on the
/// host before uploading). Builds a CSR index grouping rows by token in ascending
/// row order, uploads it, and runs the atomic-free deterministic kernel.
///
/// `acc` is WRITTEN (not accumulated), every output token is fully summed here,
/// so it need not be pre-zeroed, but tokens with no routed rows are set to 0.
pub fn moe_scatter_add_det(
    dev: &dyn Device,
    dn: &Tensor,        // [mt, hid] f32 down-GEMM output (sorted)
    tidx_host: &[u32],  // [mt] token index per row (host)
    wts: &Tensor,       // [mt] f32 router weights
    acc: &Tensor,       // [s, hid] f32 accumulator
    s: usize,
    mt: usize,
    hid: usize,
    unscale: f32,
    in_f16: bool,       // true → `dn` is f16 (read directly, no cast_f16_f32)
) -> Result<()> {
    // Build CSR (tok_starts[s+1], tok_rows[mt]) grouping rows by token. Rows are
    // appended in ascending row order → fixed, deterministic accumulation order.
    let mut counts = vec![0i32; s + 1];
    for &t in tidx_host { counts[t as usize + 1] += 1; }
    for i in 0..s { counts[i + 1] += counts[i]; } // prefix sum → tok_starts
    let tok_starts = counts; // [s+1]
    let mut cursor: Vec<i32> = tok_starts[..s].to_vec();
    let mut tok_rows = vec![0i32; mt];
    for (r, &t) in tidx_host.iter().enumerate() {
        let ti = t as usize;
        tok_rows[cursor[ti] as usize] = r as i32;
        cursor[ti] += 1;
    }
    let ts_bytes: Vec<u8> = tok_starts.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tr_bytes: Vec<u8> = tok_rows.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ts_dev = dev.upload(&ts_bytes).map_err(|e| Error::Msg(format!("moe_scatter_add_det: upload tok_starts: {e:?}")))?;
    let tr_dev = dev.upload(&tr_bytes).map_err(|e| Error::Msg(format!("moe_scatter_add_det: upload tok_rows: {e:?}")))?;

    let s_i = (s as i32).to_le_bytes().to_vec();
    let hid_i = (hid as i32).to_le_bytes().to_vec();
    let uscale_f = unscale.to_le_bytes().to_vec();
    const BLOCK_H: u32 = 128;
    let grid = [s as u32, (hid as u32).div_ceil(BLOCK_H), 1];
    let block = [BLOCK_H, 1, 1];
    dev.dispatch_raw_cuda(
        MOE_SCATTER_ADD_DET_SRC,
        "moe_scatter_add_det.cu",
        if in_f16 { "moe_scatter_add_det_f16" } else { "moe_scatter_add_det_f32" },
        &[
            (dn.buffer.as_ref(), 0),
            (ts_dev.as_ref(),    0),
            (tr_dev.as_ref(),    0),
            (wts.buffer.as_ref(), 0),
            (acc.buffer.as_ref(), 0),
        ],
        &[s_i, hid_i, uscale_f],
        grid,
        block,
        0,
        false,
    )
}

// Stable token-CSR scan: one thread per output token scans ALL mt sorted rows in
// ASCENDING row order, appending matching rows into tok_rows[toff[t]..]. No atomics
// → deterministic row order per token (== the host CSR moe_scatter_add_det builds).
const MOE_TOKEN_CSR_SRC: &str = r#"
extern "C" __global__ void moe_token_csr(
    const unsigned* __restrict__ sorted_tok, const unsigned* __restrict__ toff,
    unsigned* __restrict__ tok_rows, int mt, int s)
{
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= s) return;
    unsigned pos = toff[t];
    for (int r = 0; r < mt; r++) { if (sorted_tok[r] == (unsigned)t) { tok_rows[pos] = (unsigned)r; pos++; } }
}
"#;

/// DETERMINISTIC fully-on-device scatter-add (no host sync): builds the per-token
/// CSR (toff, tok_rows) ON DEVICE from `sorted_tok` (hist→prefix→stable scan) then
/// runs the deterministic moe_scatter_add_det kernel. Same result as the host
/// `moe_scatter_add_det` but host-sync-free → graph-capturable.
#[allow(clippy::too_many_arguments)]
/// x2 variant of moe_scatter_add_topk_det_f16: 2 hidden-dims/thread via __half2
/// vectorized loads. Exact; falls back to the x1 kernel on odd hid.
pub fn moe_scatter_add_topk_det_f16_x2(
    dev: &dyn Device,
    dn: &Tensor,
    rowpos: &Tensor,
    wts_tok: &Tensor,
    acc: &Tensor,
    s: usize,
    top_k: usize,
    hid: usize,
    unscale: f32,
) -> Result<()> {
    if hid % 2 != 0 {
        return moe_scatter_add_topk_det_f16(dev, dn, rowpos, wts_tok, acc, s, top_k, hid, unscale);
    }
    let i = |x: i32| x.to_le_bytes().to_vec();
    let pairs = (hid as u32).div_ceil(2);
    dev.dispatch_raw_cuda(
        MOE_SCATTER_ADD_DET_SRC,
        "moe_scatter_add_det.cu",
        "moe_scatter_add_topk_det_f16_x2",
        &[(dn.buffer.as_ref(), 0), (rowpos.buffer.as_ref(), rowpos.offset),
          (wts_tok.buffer.as_ref(), wts_tok.offset), (acc.buffer.as_ref(), 0)],
        &[i(s as i32), i(top_k as i32), i(hid as i32), unscale.to_le_bytes().to_vec()],
        [s as u32, pairs.div_ceil(128), 1], [128, 1, 1], 0, false)
}

pub fn moe_scatter_add_det_dev(
    dev: &dyn Device,
    dn: &Tensor,
    sorted_tok: &Tensor,
    wts: &Tensor,
    acc: &Tensor,
    s: usize,
    mt: usize,
    hid: usize,
    unscale: f32,
    in_f16: bool,
) -> Result<()> {
    let i = |x: i32| x.to_le_bytes().to_vec();
    let counts = Tensor::new(dev.alloc_zeroed(s * 4).map_err(|e| Error::Msg(format!("{e:?}")))?, vec![s], DType::U32);
    dev.dispatch_raw_cuda(MOE_HIST_SRC, "moe_hist.cu", "moe_hist",
        &[(sorted_tok.buffer.as_ref(), sorted_tok.offset), (counts.buffer.as_ref(), 0)],
        &[i(mt as i32)], [(mt as u32).div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
    let toff = Tensor::empty(dev, vec![s + 1], DType::U32)?;
    let cursor = Tensor::empty(dev, vec![s], DType::U32)?;
    dev.dispatch_raw_cuda(MOE_PREFIX_SRC, "moe_prefix.cu", "moe_prefix",
        &[(counts.buffer.as_ref(), 0), (toff.buffer.as_ref(), 0), (cursor.buffer.as_ref(), 0)],
        &[i(s as i32)], [1, 1, 1], [32, 1, 1], 0, false)?;
    let tok_rows = Tensor::empty(dev, vec![mt], DType::U32)?;
    dev.dispatch_raw_cuda(MOE_TOKEN_CSR_SRC, "moe_token_csr.cu", "moe_token_csr",
        &[(sorted_tok.buffer.as_ref(), sorted_tok.offset), (toff.buffer.as_ref(), 0), (tok_rows.buffer.as_ref(), 0)],
        &[i(mt as i32), i(s as i32)], [(s as u32).div_ceil(128), 1, 1], [128, 1, 1], 0, false)?;
    let s_i = (s as i32).to_le_bytes().to_vec();
    let hid_i = (hid as i32).to_le_bytes().to_vec();
    let uscale_f = unscale.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(
        MOE_SCATTER_ADD_DET_SRC, "moe_scatter_add_det.cu",
        if in_f16 { "moe_scatter_add_det_f16" } else { "moe_scatter_add_det_f32" },
        &[(dn.buffer.as_ref(), 0), (toff.buffer.as_ref(), 0), (tok_rows.buffer.as_ref(), 0),
          (wts.buffer.as_ref(), 0), (acc.buffer.as_ref(), 0)],
        &[s_i, hid_i, uscale_f],
        [s as u32, (hid as u32).div_ceil(128), 1], [128, 1, 1], 0, false)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn moe_scatter_add_topk_det_f16(
    dev: &dyn Device,
    dn: &Tensor,
    rowpos: &Tensor,
    wt_token_major: &Tensor,
    acc: &Tensor,
    s: usize,
    top_k: usize,
    hid: usize,
    unscale: f32,
) -> Result<()> {
    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(
        MOE_SCATTER_ADD_DET_SRC,
        "moe_scatter_add_det.cu",
        "moe_scatter_add_topk_det_f16",
        &[
            (dn.buffer.as_ref(), 0),
            (rowpos.buffer.as_ref(), rowpos.offset),
            (wt_token_major.buffer.as_ref(), wt_token_major.offset),
            (acc.buffer.as_ref(), 0),
        ],
        &[i(s as i32), i(top_k as i32), i(hid as i32), f(unscale)],
        [s as u32, (hid as u32).div_ceil(128), 1],
        [128, 1, 1],
        0,
        false,
    )?;
    Ok(())
}

/// Expand per-expert CSR offsets `[n_exp+1]` into a per-row expert-id vector
/// `[mt]` on device: `idx[r] = e` for every `r` in `[off[e], off[e+1])`.
/// Graph-safe replacement for the host `triples.map(|(e,_,_)| e)` build -- lets
/// the on-device Marlin W4A16 MoE path get per-row expert ids without a host dl.
const MOE_EXPERT_IDS_SRC: &str = r#"
extern "C" __global__ void moe_expert_ids_from_offsets(
    const unsigned int* __restrict__ off,  // [n_exp+1]
    unsigned int*       __restrict__ idx,  // [mt]
    int n_exp)
{
    int e = blockIdx.x;
    if (e >= n_exp) return;
    unsigned int lo = off[e];
    unsigned int hi = off[e + 1];
    for (unsigned int r = lo + threadIdx.x; r < hi; r += blockDim.x)
        idx[r] = (unsigned int)e;
}
"#;

/// Build a per-row expert-id tensor `[mt]` (u32) from CSR offsets `[n_exp+1]`.
/// One block per expert fills its contiguous row range. Fully device-side.
pub fn moe_expert_ids_from_offsets(dev: &dyn Device, off: &Tensor, n_exp: usize, mt: usize) -> Result<Tensor> {
    let idx = Tensor::empty(dev, vec![mt], DType::U32)?;
    dev.dispatch_raw_cuda(MOE_EXPERT_IDS_SRC, "moe_expert_ids.cu", "moe_expert_ids_from_offsets",
        &[(off.buffer.as_ref(), off.offset), (idx.buffer.as_ref(), 0)],
        &[(n_exp as i32).to_le_bytes().to_vec()],
        [n_exp as u32, 1, 1], [256, 1, 1], 0, false)?;
    Ok(idx)
}

/// Raw CUDA source for `relu2_scale_f16`: `out[i] = max(0, (float)in[i])^2 * scale`
/// where `in` and `out` are __half (f16). Fuses the cast-to-f32, relu, square,
/// scale, and cast-back-to-f16 that the MoE prefill loop previously did by
/// downloading to host, computing on CPU, and re-uploading. One kernel replaces
/// the full host-round-trip between the up and down GEMMs.
const RELU2_SCALE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void relu2_scale_bf16(
    const __nv_bfloat16* __restrict__ inp,
    __nv_bfloat16*       __restrict__ out,
    int n, float scale)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float v = __bfloat162float(inp[i]);
        if (v < 0.f) v = 0.f;
        // bf16 range (~3e38) holds relu^2 for this model; no f16-style clamp needed.
        out[i] = __float2bfloat16(v * v * scale);
    }
}
"#;

/// relu(x)^2 * scale for a BF16 tensor: reads BF16, squares in FP32, writes BF16.
/// Unlike the f16 variant there is no square-safe clamp (bf16 exponent range holds
/// the squared activations), so the FFN intermediate is not range-distorted.
pub fn relu2_scale_bf16(dev: &dyn Device, x: &Tensor, scale: f32) -> Result<Tensor> {
    let n = x.elem_count();
    let out = Tensor::empty(dev, x.shape.clone(), DType::BF16)?;
    let block = 256u32;
    let grid = (n as u32).div_ceil(block);
    dev.dispatch_raw_cuda(RELU2_SCALE_BF16_SRC, "relu2_scale_bf16.cu", "relu2_scale_bf16",
        &[(x.buffer.as_ref(), x.offset), (out.buffer.as_ref(), 0)],
        &[(n as i32).to_le_bytes().to_vec(), scale.to_le_bytes().to_vec()],
        [grid, 1, 1], [block, 1, 1], 0, false)?;
    Ok(out)
}

const RELU2_SCALE_F16_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void relu2_scale_f16(
    const __half* __restrict__ inp,
    __half*       __restrict__ out,
    int n, float scale)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float v = __half2float(inp[i]);
        if (v < 0.f) v = 0.f;
        float lim = sqrtf(65504.0f / scale);
        v = fminf(v, lim);   // clamp to f16-square-safe range (Inf/NaN -> lim, no overflow)
        out[i] = __float2half(v * v * scale);
    }
}
"#;

/// Fused relu² + scalar scale for f16 tensors, all on device.
/// Computes `out[i] = max(0, x[i])^2 * scale` (x and out are f16).
/// Used to replace the host round-trip between the MoE expert up and down GEMMs:
///   old: dl(a_f16) → relu2+scale on host → ul(a2_f16)  [forces GPU sync]
///   new: relu2_scale_f16(a_f16, scale)                  [stays on device]
pub fn relu2_scale_f16(dev: &dyn Device, x: &Tensor, scale: f32) -> Result<Tensor> {
    let n = x.elem_count();
    let out = Tensor::empty(dev, x.shape.clone(), DType::F16)?;
    let scale_bytes = scale.to_le_bytes().to_vec();
    let n_bytes = (n as i32).to_le_bytes().to_vec();
    let block = 256u32;
    let grid = (n as u32).div_ceil(block);
    dev.dispatch_raw_cuda(
        RELU2_SCALE_F16_SRC,
        "relu2_scale_f16.cu",
        "relu2_scale_f16",
        &[(x.buffer.as_ref(), 0), (out.buffer.as_ref(), 0)],
        &[n_bytes, scale_bytes],
        [grid, 1, 1],
        [block, 1, 1],
        0,
        false,
    )?;
    Ok(out)
}

/// Build an Elementwise `out[i] = act(a[i])` kernel for `dtype`. (`iron_gelu`
/// is bench-only in the iron corpus — no registered correctness test — so
/// we hand-build the IR the same way `binop_kernel` does for add/mul.)
fn unary_act_kernel(name: &str, dtype: DType, kind: ActKind) -> Kernel {
    let mut k = Kernel::new(name);
    for (pname, is_out) in [("a", false), ("c", true)] {
        k.params.push(Param {
            name: pname.into(),
            dtype,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(
        Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None },
        ValueId::new(1),
    );
    k.body.push_op(Op::Activation { kind, value: ValueId::new(1) }, ValueId::new(2));
    k.body.push_op_no_result(Op::Store {
        dst: "c".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(2),
        mask: None,
    });
    k
}

/// Fused multiply-add into an accumulator, elementwise IN-PLACE:
/// `acc[i] += x[i] · s[i]`. Lets the MoE expert sum stay ON-DEVICE, each
/// expert's `down` output is folded into `acc` on the GPU (one final download
/// per layer instead of one per expert). `s` is the per-expert weight broadcast
/// to `[len]`. Dispatches `iron_fma_inplace`.
pub fn fma_inplace(dev: &dyn Device, acc: &Tensor, x: &Tensor, s: &Tensor) -> Result<()> {
    let mut k = Kernel::new("iron_fma_inplace");
    for (pname, is_out) in [("acc", true), ("x", false), ("s", false)] {
        k.params.push(Param { name: pname.into(), dtype: acc.dtype, shape: Shape::scalar(), is_output: is_out, kind: ParamKind::Tensor });
    }
    let idx = || vec![IndexExpr::Value(ValueId::new(0))];
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(Op::Load { src: "acc".into(), indices: idx(), mask: None, other: None }, ValueId::new(1));
    k.body.push_op(Op::Load { src: "x".into(), indices: idx(), mask: None, other: None }, ValueId::new(2));
    k.body.push_op(Op::Load { src: "s".into(), indices: idx(), mask: None, other: None }, ValueId::new(3));
    k.body.push_op(Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(3) }, ValueId::new(4));
    k.body.push_op(Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(4) }, ValueId::new(5));
    k.body.push_op_no_result(Op::Store { dst: "acc".into(), indices: idx(), value: ValueId::new(5), mask: None });
    k.mode = wh_iron_core::ir::KernelMode::Elementwise;
    let n = acc.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(&k, &[Binding::Buffer(acc.buffer.clone()), Binding::Buffer(x.buffer.clone()), Binding::Buffer(s.buffer.clone())], grid)?;
    Ok(())
}

/// GELU activation (PyTorch tanh approximation, = `gelu_pytorch_tanh`),
/// elementwise. Hand-built `Activation(Gelu)` kernel. Used by ViT / SigLIP /
/// CLIP towers and the GELU-MLP LLM families.
pub fn gelu(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let k = unary_act_kernel("iron_gelu", x.dtype, ActKind::Gelu);
    let n = x.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[Binding::Buffer(x.buffer.clone()), Binding::Buffer(out.buffer.clone())],
        grid,
    )?;
    Ok(out)
}

/// LayerNorm `out = (x - mean) / sqrt(var + eps) * w + b`, normalized over the
/// last dim. Mean-subtracting + bias (unlike RMSNorm). Dispatches the
/// Reduction-mode `iron_layer_norm` (one threadgroup per row, block = n/4 — needs
/// the row width `n` divisible by `4·lsize`; ViT/SigLIP widths are). Used by
/// every transformer with LayerNorm (vision towers, BERT-style, GPT-2, …).
pub fn layer_norm(dev: &dyn Device, x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    let n = *x.shape.last().ok_or_else(|| Error::Msg("layer_norm: scalar input".into()))?;
    let rows = x.elem_count() / n;
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let eps_buf = scalar_buf(dev, eps.to_bits())?;
    let k = lookup("iron_layer_norm", x.dtype)?;
    let grid = Grid { grid: [rows as u32, 1, 1], block: [(n / 4) as u32, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(weight.buffer.clone()),
            Binding::Buffer(bias.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Buffer(eps_buf),
            Binding::Scalar((n as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Fused SwiGLU `out = silu(gate) * up`, elementwise. Dispatches `iron_swiglu`.
pub fn swiglu(dev: &dyn Device, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.shape != up.shape {
        return Err(Error::Msg("swiglu: gate/up shape mismatch".into()));
    }
    elementwise_kernel(
        dev,
        "iron_swiglu",
        gate.dtype,
        gate.shape.clone(),
        vec![Binding::Buffer(gate.buffer.clone()), Binding::Buffer(up.buffer.clone())],
    )
}

/// Embedding gather: `out[t, :] = table[indices[t], :]`. `table` is
/// `[vocab, dim]`, `indices` is `[n_tokens]` (u32). Dispatches `iron_gather`.
pub fn gather(dev: &dyn Device, table: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if table.shape.len() != 2 {
        return Err(Error::Msg(format!("gather: table must be 2-D, got {:?}", table.shape)));
    }
    let dim = table.shape[1];
    let n_tokens = indices.elem_count();
    let k = lookup("iron_gather", table.dtype)?;
    let out = Tensor::empty(dev, vec![n_tokens, dim], table.dtype)?;
    let n = (n_tokens * dim) as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(table.buffer.clone()),
            Binding::Buffer(indices.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((dim as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// DeepSeek-V4 partial RoPE: rotate only the rope tail of each head
/// (`[n_nope .. head_dim]`, adjacent-pair / GPT-J convention) in place; the
/// nope dims pass through. `qk` is `[n_heads, head_dim]`. Full-attention
/// layers: YaRN disabled (freq_scale=1, ext_factor=0). Dispatches
/// `iron_dsv4_partial_rope` (grid `[n_heads, half_rot, 1]`). Mutates and
/// returns a view of `qk`'s buffer (matches the Swift in-place RoPE).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_partial_rope(
    dev: &dyn Device,
    qk: &Tensor,
    n_heads: u32,
    head_dim: u32,
    n_nope: u32,
    half_rot: u32,
    position: u32,
    theta_base: f32,
    inverse: bool,
) -> Result<Tensor> {
    let k = lookup("iron_partial_rope", qk.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());
    // Bind qk as both input and output → in-place (nope dims preserved).
    let bindings = vec![
        Binding::Buffer(qk.buffer.clone()),
        Binding::Buffer(qk.buffer.clone()),
        u(head_dim),
        u(n_nope),
        u(half_rot),
        u(n_heads),
        u(position),
        f(theta_base),
        u(if inverse { 1 } else { 0 }),
        f(1.0), // freq_scale (YaRN off)
        f(0.0), // ext_factor
        f(0.0), // corr_low
        f(1.0), // corr_high
    ];
    // TIER-1 launch-geometry fix: move half_rot from grid.y into block.y
    // (product unchanged → byte-identical gid_y = blockIdx.y*blockDim.y +
    // threadIdx.y under Grid3D codegen).
    let grid = Grid { grid: [n_heads, 1, 1], block: [1, half_rot, 1] };
    dev.dispatch(&k, &bindings, grid)?;
    Ok(Tensor::new(qk.buffer.clone(), qk.shape.clone(), qk.dtype))
}

// ── Laguna YaRN partial-rotary RoPE (raw CUDA) ───────────────────────────
//
// Laguna's full-attention layers rotate only the first `n_rot` (partial-
// rotary) dims of each head with NEOX pairing (`j`, `j+n_rot/2`) and a YaRN
// frequency/magnitude correction; the tail dims pass through unrotated. The
// sliding-window layers reuse the same kernel with `n_rot == head_dim`,
// `freq_scale = 1.0`, `ext_factor = 0.0`, `attn_factor = 1.0` (plain RoPE,
// no passthrough, no magnitude scaling).
//
// Ports the corr_dims + ramp + mscale math from the reference C++ engine's
// YaRN rope kernel (`ggml_rope_yarn_corr_dims` + `rope_yarn` in that
// engine's CUDA rope source) bit-for-bit:
//   corr_dim(n_rotations) = n_rot * ln(n_orig_ctx / (n_rotations * 2π)) / (2 * ln(theta_base))
//   corr_low  = max(0, floor(corr_dim(beta_fast)))
//   corr_high = min(n_rot - 1, ceil(corr_dim(beta_slow)))
//   ramp(j)   = 1 - clamp((j - corr_low) / max(0.001, corr_high - corr_low), 0, 1)
//   theta     = lerp(freq_scale * theta_extrap, theta_extrap, ramp(j) * ext_factor)
//   mscale    = attn_factor * (ext_factor != 0 ? 1 + 0.1 * ln(1 / freq_scale) : 1)
//
// IMPORTANT: `attn_factor` here is the model's stored `attention_factor`
// (e.g. 1.4852030263919618 for Laguna-S-2.1's factor-128 YaRN), passed
// straight through, this kernel re-applies the `1 + 0.1*ln(1/freq_scale)`
// correction internally exactly as the reference engine does (the engine's
// the reference engine's context handling pre-cancels its OWN internally-derived default mscale
// against this same term before folding in the config's `attn_factor`; the
// net effect is that `attn_factor` reaches the rope kernel unmodified and
// gets this multiplier applied once, INSIDE the kernel). Do not pre-apply
// the correction a second time in caller code.
const ROPE_YARN_PARTIAL_SRC: &str = r#"
extern "C" __global__ void butter_rope_yarn_partial(
    const float* __restrict__ qk, float* __restrict__ out,
    int head_dim, int n_rot, int position,
    float theta_base, float freq_scale, float ext_factor, float attn_factor,
    float corr_low, float corr_high)
{
    int h = blockIdx.x;
    int tid = threadIdx.y;
    int half_rot = n_rot / 2;
    const float* row_in  = qk  + (long)h * head_dim;
    float*       row_out = out + (long)h * head_dim;

    if (tid < half_rot) {
        int j = tid;
        float theta_scale  = powf(theta_base, -2.0f / (float)n_rot);
        float theta_extrap = (float)position * powf(theta_scale, (float)j);
        float theta_interp = freq_scale * theta_extrap;
        float theta        = theta_interp;
        // attn_factor IS the final magnitude scale (mscale). The reference
        // stack cancels its kernel's internal 1 + 0.1*ln(1/freq_scale)
        // re-multiply by pre-dividing on the host, so the net scale equals the
        // model's stored attention_factor; we just apply it directly.
        float mscale       = attn_factor;
        if (ext_factor != 0.0f) {
            float y        = ((float)j - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp     = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
            float ramp_mix = ramp * ext_factor;
            theta  = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        }
        float cos_t = cosf(theta) * mscale;
        float sin_t = sinf(theta) * mscale;
        float x0 = row_in[j];
        float x1 = row_in[j + half_rot];
        row_out[j]            = x0 * cos_t - x1 * sin_t;
        row_out[j + half_rot] = x0 * sin_t + x1 * cos_t;
    } else {
        // Passthrough tail: copy the two contiguous elements at [n_rot + 2p,
        // n_rot + 2p + 1) unrotated (matches the reference NEOX kernel's
        // untouched-tail convention).
        int p   = tid - half_rot;
        int idx = n_rot + 2 * p;
        if (idx < head_dim) {
            row_out[idx]     = row_in[idx];
            row_out[idx + 1] = row_in[idx + 1];
        }
    }
}
"#;

/// YaRN-scaled partial-rotary RoPE (NEOX pairing) for a `[n_heads, head_dim]`
/// Q or K tensor at sequence `position`. `n_rot` is the number of rotated
/// dims (`<= head_dim`; the rest pass through unrotated). Pass `n_rot ==
/// head_dim`, `freq_scale = 1.0`, `ext_factor = 0.0`, `attn_factor = 1.0` for
/// plain (non-YaRN) RoPE, `beta_fast`/`beta_slow`/`n_orig_ctx` are then
/// unused (the ramp contributes zero once `ext_factor` is zero). f32 only.
#[allow(clippy::too_many_arguments)]
pub fn rope_yarn_partial(
    dev: &dyn Device,
    qk: &Tensor,
    position: u32,
    n_rot: u32,
    theta_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    n_orig_ctx: u32,
) -> Result<Tensor> {
    let head_dim = *qk.shape.last().ok_or_else(|| Error::Msg("rope_yarn_partial: scalar input".into()))?;
    let n_heads = qk.elem_count() / head_dim;
    if n_rot == 0 || n_rot as usize > head_dim || n_rot % 2 != 0 {
        return Err(Error::Msg(format!(
            "rope_yarn_partial: n_rot {n_rot} invalid for head_dim {head_dim}"
        )));
    }
    // ggml_rope_yarn_corr_dims: the [low, high] band (in rotated-pair-index
    // units) where the rotary frequency ramps from interpolated (YaRN-scaled)
    // to extrapolated (raw). Degenerate but harmless when ext_factor == 0.
    let corr_dim = |n_rotations: f32| -> f32 {
        n_rot as f32 * (n_orig_ctx as f32 / (n_rotations * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * theta_base.ln())
    };
    let corr_low = corr_dim(beta_fast).floor().max(0.0);
    let corr_high = corr_dim(beta_slow).ceil().min(n_rot as f32 - 1.0);

    let out = Tensor::empty(dev, qk.shape.clone(), DType::F32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    let half = (head_dim / 2) as u32;
    dev.dispatch_raw_cuda(
        ROPE_YARN_PARTIAL_SRC, "rope_yarn_partial.cu", "butter_rope_yarn_partial",
        &[(qk.buffer.as_ref(), qk.offset), (out.buffer.as_ref(), 0)],
        &[
            i(head_dim as i32), i(n_rot as i32), i(position as i32),
            f(theta_base), f(freq_scale), f(ext_factor), f(attn_factor),
            f(corr_low), f(corr_high),
        ],
        [n_heads as u32, 1, 1], [1, half, 1], 0, false,
    )?;
    Ok(out)
}

const ROPE_YARN_PARTIAL_POSBUF_SRC: &str = r#"
extern "C" __global__ void butter_rope_yarn_partial_posbuf(
    const float* __restrict__ qk, float* __restrict__ out,
    const unsigned int* __restrict__ pos_buf,
    int head_dim, int n_rot,
    float theta_base, float freq_scale, float ext_factor, float attn_factor,
    float corr_low, float corr_high)
{
    int position = (int)pos_buf[0];
    int h = blockIdx.x;
    int tid = threadIdx.y;
    int half_rot = n_rot / 2;
    const float* row_in  = qk  + (long)h * head_dim;
    float*       row_out = out + (long)h * head_dim;

    if (tid < half_rot) {
        int j = tid;
        float theta_scale  = powf(theta_base, -2.0f / (float)n_rot);
        float theta_extrap = (float)position * powf(theta_scale, (float)j);
        float theta_interp = freq_scale * theta_extrap;
        float theta        = theta_interp;
        float mscale       = attn_factor;
        if (ext_factor != 0.0f) {
            float y        = ((float)j - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp     = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
            float ramp_mix = ramp * ext_factor;
            theta  = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        }
        float cos_t = cosf(theta) * mscale;
        float sin_t = sinf(theta) * mscale;
        float x0 = row_in[j];
        float x1 = row_in[j + half_rot];
        row_out[j]            = x0 * cos_t - x1 * sin_t;
        row_out[j + half_rot] = x0 * sin_t + x1 * cos_t;
    } else {
        int p   = tid - half_rot;
        int idx = n_rot + 2 * p;
        if (idx < head_dim) {
            row_out[idx]     = row_in[idx];
            row_out[idx + 1] = row_in[idx + 1];
        }
    }
}
"#;

/// CUDA-graph-capturable variant of [`rope_yarn_partial`]: identical math,
/// but `position` is read from a 1-element `u32` device buffer at kernel
/// runtime instead of being baked in as a launch-time scalar. A captured
/// graph replays the exact same recorded launch every token, so any value
/// that changes per token (the sequence position) has to live behind a
/// stable pointer whose CONTENTS the caller updates before each replay,
/// rather than as an argument frozen at capture time. `corr_low`/`corr_high`
/// depend only on the (per-layer-type, capture-time-constant) RoPE constants,
/// so those stay ordinary scalars. Used by Laguna's decode path under
/// `BUTTER_LAGUNA_GRAPH=1`; the plain scalar-position `rope_yarn_partial` above
/// remains the default (eager, non-graph) path.
#[allow(clippy::too_many_arguments)]
pub fn rope_yarn_partial_posbuf(
    dev: &dyn Device,
    qk: &Tensor,
    pos_buf: &Tensor,
    n_rot: u32,
    theta_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    n_orig_ctx: u32,
) -> Result<Tensor> {
    let head_dim = *qk.shape.last().ok_or_else(|| Error::Msg("rope_yarn_partial_posbuf: scalar input".into()))?;
    let n_heads = qk.elem_count() / head_dim;
    if n_rot == 0 || n_rot as usize > head_dim || n_rot % 2 != 0 {
        return Err(Error::Msg(format!(
            "rope_yarn_partial_posbuf: n_rot {n_rot} invalid for head_dim {head_dim}"
        )));
    }
    // Same YaRN corr-dims derivation as `rope_yarn_partial`, a function of
    // the RoPE constants only, never the runtime position.
    let corr_dim = |n_rotations: f32| -> f32 {
        n_rot as f32 * (n_orig_ctx as f32 / (n_rotations * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * theta_base.ln())
    };
    let corr_low = corr_dim(beta_fast).floor().max(0.0);
    let corr_high = corr_dim(beta_slow).ceil().min(n_rot as f32 - 1.0);

    let out = Tensor::empty(dev, qk.shape.clone(), DType::F32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    let half = (head_dim / 2) as u32;
    dev.dispatch_raw_cuda(
        ROPE_YARN_PARTIAL_POSBUF_SRC, "rope_yarn_partial_posbuf.cu", "butter_rope_yarn_partial_posbuf",
        &[
            (qk.buffer.as_ref(), qk.offset),
            (out.buffer.as_ref(), 0),
            (pos_buf.buffer.as_ref(), pos_buf.offset),
        ],
        &[
            i(head_dim as i32), i(n_rot as i32),
            f(theta_base), f(freq_scale), f(ext_factor), f(attn_factor),
            f(corr_low), f(corr_high),
        ],
        [n_heads as u32, 1, 1], [1, half, 1], 0, false,
    )?;
    Ok(out)
}

const ROPE_YARN_PARTIAL_MANY_SRC: &str = r#"
extern "C" __global__ void butter_rope_yarn_partial_many(
    const float* __restrict__ qk, float* __restrict__ out,
    const unsigned int* __restrict__ positions,
    int head_dim, int n_rot, int n_heads,
    float theta_base, float freq_scale, float ext_factor, float attn_factor,
    float corr_low, float corr_high)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.z;
    int half_rot = n_rot / 2;
    int position = (int)positions[t];
    long row_off = ((long)t * n_heads + h) * head_dim;
    const float* row_in  = qk  + row_off;
    float*       row_out = out + row_off;

    if (tid < half_rot) {
        int j = tid;
        float theta_scale  = powf(theta_base, -2.0f / (float)n_rot);
        float theta_extrap = (float)position * powf(theta_scale, (float)j);
        float theta_interp = freq_scale * theta_extrap;
        float theta        = theta_interp;
        float mscale       = attn_factor;
        if (ext_factor != 0.0f) {
            float y        = ((float)j - corr_low) / fmaxf(0.001f, corr_high - corr_low);
            float ramp     = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
            float ramp_mix = ramp * ext_factor;
            theta  = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        }
        float cos_t = cosf(theta) * mscale;
        float sin_t = sinf(theta) * mscale;
        float x0 = row_in[j];
        float x1 = row_in[j + half_rot];
        row_out[j]            = x0 * cos_t - x1 * sin_t;
        row_out[j + half_rot] = x0 * sin_t + x1 * cos_t;
    } else {
        int p   = tid - half_rot;
        int idx = n_rot + 2 * p;
        if (idx < head_dim) {
            row_out[idx]     = row_in[idx];
            row_out[idx + 1] = row_in[idx + 1];
        }
    }
}
"#;

/// Batched [`rope_yarn_partial`]: applies YaRN-scaled partial-rotary RoPE to
/// ALL `T` rows of a `[T, n_heads, head_dim]` Q or K tensor in ONE dispatch,
/// each row `t` rotated at `positions[t]` (one absolute sequence position per
/// token row) instead of a single scalar `position` shared by every row.
/// Replaces the T-loop of per-token `rope_yarn_partial` calls in the Laguna
/// prefill attention path. Same math/params as `rope_yarn_partial` (pass
/// `n_rot == head_dim`, `freq_scale = 1.0`, `ext_factor = 0.0`,
/// `attn_factor = 1.0` for plain, non-YaRN RoPE: Laguna's sliding-window
/// layers use exactly this). Grid `[T, n_heads, 1]`, block `[1, 1, head_dim/2]`
/// (covers both the rotated-pair range and the passthrough-tail range, same
/// as the scalar-position kernel). f32 only.
#[allow(clippy::too_many_arguments)]
pub fn rope_yarn_partial_many(
    dev: &dyn Device,
    qk: &Tensor,
    positions: &Tensor,
    n_heads: usize,
    n_rot: u32,
    theta_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    n_orig_ctx: u32,
) -> Result<Tensor> {
    let head_dim = *qk.shape.last().ok_or_else(|| Error::Msg("rope_yarn_partial_many: scalar input".into()))?;
    let t = qk.elem_count() / (n_heads * head_dim);
    if positions.elem_count() != t {
        return Err(Error::Msg(format!(
            "rope_yarn_partial_many: positions len {} != T {t}", positions.elem_count()
        )));
    }
    if n_rot == 0 || n_rot as usize > head_dim || n_rot % 2 != 0 {
        return Err(Error::Msg(format!(
            "rope_yarn_partial_many: n_rot {n_rot} invalid for head_dim {head_dim}"
        )));
    }
    // Same YaRN corr-dims derivation as `rope_yarn_partial`, depends only on
    // the (per-layer-type, chunk-invariant) RoPE constants, never on position.
    let corr_dim = |n_rotations: f32| -> f32 {
        n_rot as f32 * (n_orig_ctx as f32 / (n_rotations * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * theta_base.ln())
    };
    let corr_low = corr_dim(beta_fast).floor().max(0.0);
    let corr_high = corr_dim(beta_slow).ceil().min(n_rot as f32 - 1.0);

    let out = Tensor::empty(dev, qk.shape.clone(), DType::F32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    let f = |x: f32| x.to_le_bytes().to_vec();
    let half = (head_dim / 2) as u32;
    dev.dispatch_raw_cuda(
        ROPE_YARN_PARTIAL_MANY_SRC, "rope_yarn_partial_many.cu", "butter_rope_yarn_partial_many",
        &[
            (qk.buffer.as_ref(), qk.offset),
            (out.buffer.as_ref(), 0),
            (positions.buffer.as_ref(), positions.offset),
        ],
        &[
            i(head_dim as i32), i(n_rot as i32), i(n_heads as i32),
            f(theta_base), f(freq_scale), f(ext_factor), f(attn_factor),
            f(corr_low), f(corr_high),
        ],
        [t as u32, n_heads as u32, 1], [1, 1, half], 0, false,
    )?;
    Ok(out)
}

/// Write a single `u32` scalar into an existing 1-element device buffer, in
/// place, without reallocating. Laguna's CUDA-graph decode path
/// (`BUTTER_LAGUNA_GRAPH=1`) keeps a small persistent "workspace" of such
/// buffers (token id, RoPE position, KV length) whose device POINTERS get
/// captured into the graph and must never move; only their contents change
/// from token to token. Call this OUTSIDE any graph-capture region, it must
/// run eagerly every step, right before the (captured or replayed) decode
/// work that reads the buffer.
pub fn write_u32(dev: &dyn Device, buf: &Tensor, val: u32) -> Result<()> {
    dev.dispatch_raw_cuda(
        WRITE_U32_SRC, "write_u32.cu", "butter_write_u32",
        &[(buf.buffer.as_ref(), buf.offset)],
        &[val.to_le_bytes().to_vec()],
        [1, 1, 1], [1, 1, 1], 0, false,
    )
}

const WRITE_U32_SRC: &str = r#"
extern "C" __global__ void butter_write_u32(unsigned int* __restrict__ buf, unsigned int val)
{
    buf[0] = val;
}
"#;

// ── Laguna per-head softplus attention gate (raw CUDA) ───────────────────
//
// `g` holds one raw (pre-activation) gate scalar per head, projected from
// the same normed hidden state that feeds Q/K/V (`g_proj(attn_norm(x))`,
// NOT the attention output). This multiplies `softplus(g[h])` into every
// element of head `h`'s attention output, broadcasting the scalar across
// `head_dim`, applied BEFORE `o_proj`.
const GATE_SOFTPLUS_MUL_PERHEAD_SRC: &str = r#"
extern "C" __global__ void butter_gate_softplus_mul_perhead(
    const float* __restrict__ attn, const float* __restrict__ g,
    float* __restrict__ out, int head_dim, int n_heads)
{
    int h = blockIdx.x;
    int i = threadIdx.x;
    if (h >= n_heads || i >= head_dim) return;
    float gv = g[h];
    // softplus(x) = log1p(exp(x)); linear for large x where exp(x) overflows.
    float sp = gv > 20.0f ? gv : log1pf(expf(gv));
    out[h * head_dim + i] = attn[h * head_dim + i] * sp;
}
"#;

/// Per-head softplus attention gate: `out[h, i] = attn[h, i] * softplus(g[h])`
/// for `attn` `[n_heads, head_dim]` and `g` `[n_heads]` (one raw gate scalar
/// per head, broadcast over `head_dim`). Laguna's XS.2/S.2 attention-output
/// gate, applied before `o_proj`. f32 only.
pub fn gate_softplus_mul_perhead(dev: &dyn Device, attn: &Tensor, g: &Tensor) -> Result<Tensor> {
    let head_dim = *attn.shape.last().ok_or_else(|| Error::Msg("gate_softplus_mul_perhead: scalar attn".into()))?;
    let n_heads = attn.elem_count() / head_dim;
    if g.elem_count() != n_heads {
        return Err(Error::Msg(format!(
            "gate_softplus_mul_perhead: g has {} elems, expected {n_heads} (one per head)",
            g.elem_count()
        )));
    }
    let out = Tensor::empty(dev, attn.shape.clone(), DType::F32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(
        GATE_SOFTPLUS_MUL_PERHEAD_SRC, "gate_softplus_mul_perhead.cu", "butter_gate_softplus_mul_perhead",
        &[(attn.buffer.as_ref(), attn.offset), (g.buffer.as_ref(), g.offset), (out.buffer.as_ref(), 0)],
        &[i(head_dim as i32), i(n_heads as i32)],
        [n_heads as u32, 1, 1], [head_dim as u32, 1, 1], 0, false,
    )?;
    Ok(out)
}

const GATE_SOFTPLUS_MUL_PERHEAD_MANY_SRC: &str = r#"
extern "C" __global__ void butter_gate_softplus_mul_perhead_many(
    const float* __restrict__ attn, const float* __restrict__ g,
    float* __restrict__ out, int head_dim)
{
    long row = (long)blockIdx.x;   // flat (t * n_heads + h)
    int i = threadIdx.x;
    if (i >= head_dim) return;
    float gv = g[row];
    float sp = gv > 20.0f ? gv : log1pf(expf(gv));
    out[row * head_dim + i] = attn[row * head_dim + i] * sp;
}
"#;

/// Batched [`gate_softplus_mul_perhead`]: `out[t, h, i] = attn[t, h, i] *
/// softplus(g[t, h])` for `attn` `[T, n_heads, head_dim]` and `g` `[T,
/// n_heads]` (one raw gate scalar per token-head, broadcast over
/// `head_dim`). Replaces the T-loop of per-token `gate_softplus_mul_perhead`
/// calls in the Laguna prefill attention path, applied before `o_proj`,
/// same as the decode path. f32 only.
pub fn gate_softplus_mul_perhead_many(dev: &dyn Device, attn: &Tensor, g: &Tensor, n_heads: usize) -> Result<Tensor> {
    let head_dim = *attn.shape.last().ok_or_else(|| Error::Msg("gate_softplus_mul_perhead_many: scalar attn".into()))?;
    let rows = attn.elem_count() / head_dim;
    if g.elem_count() != rows {
        return Err(Error::Msg(format!(
            "gate_softplus_mul_perhead_many: g has {} elems, expected {rows} (T*n_heads)",
            g.elem_count()
        )));
    }
    let _ = n_heads; // rows already folds T*n_heads; kept for call-site clarity/shape docs
    let out = Tensor::empty(dev, attn.shape.clone(), DType::F32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(
        GATE_SOFTPLUS_MUL_PERHEAD_MANY_SRC, "gate_softplus_mul_perhead_many.cu", "butter_gate_softplus_mul_perhead_many",
        &[(attn.buffer.as_ref(), attn.offset), (g.buffer.as_ref(), g.offset), (out.buffer.as_ref(), 0)],
        &[i(head_dim as i32)],
        [rows as u32, 1, 1], [head_dim as u32, 1, 1], 0, false,
    )?;
    Ok(out)
}

/// One-time compaction of a linear prefill K/V scratch buffer into a
/// fixed-size decode ring: copies the LAST `copy_len` absolute positions
/// `[copy_start, copy_start+copy_len)` from `src` `[n_kv_heads, src_cap,
/// head_dim]` into `dst` `[n_kv_heads, ring_cap, head_dim]` at ring slot
/// `abs_pos % ring_cap`. Used once per sliding-window layer after a Laguna
/// prefill finishes the whole prompt, to hand off from the prefill-only
/// linear scratch (`LagunaPrefillScratch`, sized to the prompt) to the
/// persistent decode ring (`LagunaKvCache`'s fixed `sliding_window`-slot
/// buffer) so decode can continue exactly where prefill left off. f32 only.
#[allow(clippy::too_many_arguments)]
pub fn kv_compact_ring(
    dev: &dyn Device,
    src: &Tensor,
    dst: &Tensor,
    n_kv_heads: usize,
    src_cap: usize,
    head_dim: usize,
    ring_cap: usize,
    copy_start: usize,
    copy_len: usize,
) -> Result<()> {
    if copy_len == 0 {
        return Ok(());
    }
    let i = |x: i32| x.to_le_bytes().to_vec();
    let total = (n_kv_heads * copy_len * head_dim) as u32;
    dev.dispatch_raw_cuda(
        KV_COMPACT_RING_SRC, "kv_compact_ring.cu", "butter_kv_compact_ring",
        &[(src.buffer.as_ref(), src.offset), (dst.buffer.as_ref(), dst.offset)],
        &[
            i(n_kv_heads as i32), i(src_cap as i32), i(head_dim as i32),
            i(ring_cap as i32), i(copy_start as i32), i(copy_len as i32),
        ],
        [total.div_ceil(256).max(1), 1, 1], [256, 1, 1], 0, false,
    )
}

const KV_COMPACT_RING_SRC: &str = r#"
extern "C" __global__ void butter_kv_compact_ring(
    const float* __restrict__ src, float* __restrict__ dst,
    int n_kv_heads, int src_cap, int hd, int ring_cap, int copy_start, int copy_len)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)n_kv_heads * copy_len * hd;
    if (i >= total) return;
    int d = (int)(i % hd);
    long t = i / hd;
    int c = (int)(t % copy_len);   // 0..copy_len
    int h = (int)(t / copy_len);   // kv head
    int src_pos = copy_start + c;  // absolute position within the scratch
    int dst_slot = src_pos % ring_cap;
    dst[((long)h * ring_cap + dst_slot) * hd + d] = src[((long)h * src_cap + src_pos) * hd + d];
}
"#;

/// Softmax over the last dim, row-wise. Dispatches `iron_softmax`. Row width
/// `n` must be a multiple of 1024 (the kernel's 4-elems/thread loop).
pub fn softmax(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    let n = *x.shape.last().ok_or_else(|| Error::Msg("softmax: scalar input".into()))?;
    if n % 1024 != 0 {
        return Err(Error::Msg(format!("softmax: row width {n} must be a multiple of 1024")));
    }
    let rows = x.elem_count() / n;
    let k = lookup("iron_softmax", x.dtype)?;
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let grid = Grid { grid: [rows as u32, 1, 1], block: [256, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((n as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Llama-style rotary position embedding applied to a `[n_heads, head_dim]`
/// Q or K tensor at sequence `position`. Dispatches `iron_rope_llama`. Pass
/// the freq-band knobs disabled (`scale_factor=1`, `low=1`, `high=1`,
/// `orig_max=1e9`) for vanilla RoPE.
#[allow(clippy::too_many_arguments)]
pub fn rope_llama(
    dev: &dyn Device,
    qk: &Tensor,
    position: u32,
    theta_base: f32,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: f32,
) -> Result<Tensor> {
    let head_dim = *qk.shape.last().ok_or_else(|| Error::Msg("rope: scalar input".into()))?;
    let n_heads = qk.elem_count() / head_dim;
    let half = head_dim / 2;
    let k = lookup("iron_rope_banded", qk.dtype)?;
    let out = Tensor::empty(dev, qk.shape.clone(), qk.dtype)?;
    // iron_rope_banded reads `position` from a `positions: [T]` u32 buffer and
    // selects the row via an outer grid axis; decode is the `T = 1` case (a
    // 1-element positions buffer). `row_stride` is the element stride between
    // consecutive rows (= n_heads*head_dim for a contiguous, non-fused-QKV qk).
    let pos_buf = scalar_buf(dev, position)?;
    let row_stride = (n_heads * head_dim) as u32;
    // Grid3D `[T, n_heads, half_dim]`: program_id<0>=row, <1>=head, <2>=i.
    // TIER-1 launch-geometry fix: move `half` from grid.z into block.z
    // (product unchanged → byte-identical gid_z under Grid3D codegen).
    let grid = Grid { grid: [1, n_heads as u32, 1], block: [1, 1, half as u32] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(qk.buffer.clone()),
            Binding::Buffer(pos_buf),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((head_dim as u32).to_le_bytes().to_vec()),
            Binding::Scalar((half as u32).to_le_bytes().to_vec()),
            Binding::Scalar(row_stride.to_le_bytes().to_vec()),
            Binding::Scalar(theta_base.to_le_bytes().to_vec()),
            Binding::Scalar(scale_factor.to_le_bytes().to_vec()),
            Binding::Scalar(low_freq_factor.to_le_bytes().to_vec()),
            Binding::Scalar(high_freq_factor.to_le_bytes().to_vec()),
            Binding::Scalar(original_max_position.to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Batched Llama-style RoPE: apply position-dependent rotation to ALL T rows in
/// ONE dispatch. Replaces the T-loop of per-token `rope_llama` calls in the
/// prefill attention path, eliminating the dl(q_all)/dl(k_all)/dl(v_all) + S×dl
/// overhead for the attention layers.
///
/// `qk`: [T, n_heads, head_dim] float  
/// `positions`: [T] u32 positions (one per token row)  
/// returns: [T, n_heads, head_dim] rotated (same dtype as qk)
pub fn rope_llama_many(
    dev: &dyn Device,
    qk: &Tensor,
    positions: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta_base: f32,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: f32,
) -> Result<Tensor> {
    use wh_iron_core::ir::KernelMode;
    let t = qk.elem_count() / (n_heads * head_dim);
    let half = head_dim / 2;
    let k = cached_ir("iron_rope_llama_many", qk.dtype, || {
        let mut kk = wh_iron_std::kernels::rope::rope_banded::iron_rope_banded::kernel_ir_for(qk.dtype);
        kk.mode = KernelMode::Grid3D;
        kk
    });
    let out = Tensor::empty(dev, qk.shape.clone(), qk.dtype)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let f = |v: f32| Binding::Scalar(v.to_le_bytes().to_vec());
    let row_stride = (n_heads * head_dim) as u32;
    // TIER-1 launch-geometry fix: move `half` from grid.z into block.z
    // (product unchanged → byte-identical gid_z under Grid3D codegen).
    let grid = Grid { grid: [t as u32, n_heads as u32, 1], block: [1, 1, half as u32] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(qk.buffer.clone()),
            Binding::Buffer(positions.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(head_dim as u32),
            u(half as u32),
            u(row_stride),
            f(theta_base),
            f(scale_factor),
            f(low_freq_factor),
            f(high_freq_factor),
            f(original_max_position),
        ],
        grid,
    )?;
    Ok(out)
}

/// Batched KV-cache append: write T tokens' K (or V) rows into the per-head
/// cache in ONE dispatch. Replaces the T-loop of per-token `kv_append` calls.
///
/// `src`: [T, n_kv_heads, head_dim]  
/// `positions`: [T] u32 positions  
/// `cache`: [n_kv_heads, max_seq, head_dim] (pre-allocated)
pub fn kv_append_many(
    dev: &dyn Device,
    src: &Tensor,
    positions: &Tensor,
    cache: &Tensor,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) -> Result<()> {
    use wh_iron_core::ir::KernelMode;
    let t = src.elem_count() / (n_kv_heads * head_dim);
    let k = cached_ir("kv_cache_update_many", src.dtype, || {
        let mut kk = wh_iron_std::kernels::kv_cache::update_many::iron_kv_cache_update_many::kernel_ir_for(src.dtype);
        kk.mode = KernelMode::Grid3D;
        kk
    });
    let total = t * n_kv_heads * head_dim;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(src.buffer.clone()),
            Binding::Buffer(positions.buffer.clone()),
            Binding::Buffer(cache.buffer.clone()),
            u(head_dim as u32),
            u(max_seq as u32),
            u((n_kv_heads * head_dim) as u32),
        ],
        // PERF: pack threads/block (was 1). Global linear element id; total =
        // t*n_kv_heads*head_dim. Pick largest of {256,64,1} dividing total exactly.
        {
            let b = if total % 256 == 0 { 256u32 } else if total % 64 == 0 { 64u32 } else { 1u32 };
            Grid { grid: [(total as u32) / b, 1, 1], block: [b, 1, 1] }
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{dsv4_mhc_sinkhorn_split, permute_q4_from_marlin, permute_q4_to_marlin, quantize_q4, quantize_nvfp4, dec_ue4m3, E2M1_LUT};

    /// NVFP4 quantize → dequantize round-trip reconstruction error must be on par
    /// with Q4 (the existing argmax-correct path). Validated standalone at ~9.5% on
    /// GB10; here we just gate that the Rust quantizer reconstructs the same way and
    /// stays comfortably under Q4's error ceiling for a representative weight.
    #[test]
    fn nvfp4_reconstruction_on_par_with_q4() {
        let (m, k) = (64usize, 2688usize); // k % 16 == 0 and % 32 == 0
        // deterministic pseudo-gaussian weights
        let w: Vec<f32> = (0..m * k).map(|i| {
            let x = ((i * 2654435761usize) & 0xffff) as f32 / 65535.0 - 0.5;
            let y = (((i * 40503usize) >> 3) & 0xffff) as f32 / 65535.0 - 0.5;
            (x + y) * 0.08 // roughly normal-ish, small magnitude
        }).collect();

        // NVFP4 reconstruct
        let (codes, scales, global) = quantize_nvfp4(&w, m, k);
        let nblk = k / 16;
        let mut sse_nv = 0f64; let mut ss = 0f64;
        for r in 0..m {
            for b in 0..nblk {
                let bs = dec_ue4m3(scales[r * nblk + b]) * global;
                for i in 0..16 {
                    let elem = b * 16 + i;
                    let word = elem / 8; let nib = elem % 8;
                    let code = (codes[r * (k / 8) + word] >> (nib * 4)) & 0xf;
                    let sign = if code & 8 != 0 { -1.0 } else { 1.0 };
                    let rec = sign * E2M1_LUT[(code & 7) as usize] * bs;
                    let orig = w[r * k + elem];
                    let d = (rec - orig) as f64; sse_nv += d * d; ss += (orig as f64) * (orig as f64);
                }
            }
        }
        // Q4 reconstruct (block-32 amax/7) for the same weight
        let (qs, qsc) = quantize_q4(&w, m, k);
        let bpr = k / 32; let mut sse_q4 = 0f64;
        for r in 0..m {
            for b in 0..bpr {
                let d = qsc[r * bpr + b];
                for word in 0..4 {
                    for i in 0..8 {
                        let packed = qs[r * bpr * 4 + b * 4 + word];
                        let mut q = ((packed >> (i * 4)) & 0xf) as i32;
                        if q >= 8 { q -= 16; } // sign-extend nibble
                        let rec = q as f32 * d;
                        let orig = w[r * k + b * 32 + word * 8 + i];
                        let e = (rec - orig) as f64; sse_q4 += e * e;
                    }
                }
            }
        }
        let rmse_nv = (sse_nv / ss).sqrt();
        let rmse_q4 = (sse_q4 / ss).sqrt();
        println!("nvfp4 rel_rmse={:.4}  q4 rel_rmse={:.4}", rmse_nv, rmse_q4);
        // NVFP4 must be within 1.5× of Q4 (it was actually slightly better in C++).
        assert!(rmse_nv < rmse_q4 * 1.5, "NVFP4 recon {rmse_nv:.4} >> Q4 {rmse_q4:.4}");
    }

    /// Verify `permute_q4_to_marlin` + `permute_q4_from_marlin` is a lossless round-trip.
    #[test]
    fn marlin_permute_roundtrip() {
        let (n_exp, n_out, k_in) = (4, 128, 64); // n_out and k_in are multiples of 64/32
        let bpr = k_in / 32;
        let total_rows = n_exp * n_out;
        // Use a deterministic weight pattern to expose nibble-mapping bugs.
        let wv: Vec<f32> = (0..total_rows * k_in)
            .map(|i| (i as f32 * 0.019 - 0.5).sin() * 7.0)
            .collect();
        let (qs_std, _scales) = quantize_q4(&wv, total_rows, k_in);

        // Round-trip: std → Marlin → std
        let qs_mar = permute_q4_to_marlin(&qs_std, n_exp, n_out, k_in);
        let qs_back = permute_q4_from_marlin(&qs_mar, n_exp, n_out, k_in);

        assert_eq!(qs_std.len(), qs_mar.len(), "Marlin layout must have same u32 count");
        assert_eq!(qs_std, qs_back, "round-trip must be lossless");

        // Also verify individual nibble values are preserved at a few spot checks.
        // Standard layout: qs_std[exp*n_out*bpr*4 + row*bpr*4 + blk*4 + word]
        // nibble at k=5: blk=0, word=0, nib_in_word=5
        for e in 0..n_exp {
            for row in 0..n_out {
                for k in [0usize, 7, 15, 16, 31] {
                    let blk = k / 32;
                    let word = (k % 32) / 8;
                    let nib_in = (k % 32) % 8;
                    let src_word = qs_std[e * n_out * bpr * 4 + row * bpr * 4 + blk * 4 + word];
                    let nib_std = (src_word >> (nib_in * 4)) & 0xf;
                    let dst_word = qs_back[e * n_out * bpr * 4 + row * bpr * 4 + blk * 4 + word];
                    let nib_back = (dst_word >> (nib_in * 4)) & 0xf;
                    assert_eq!(nib_std, nib_back,
                        "nibble mismatch at e={e} row={row} k={k}: std={nib_std} back={nib_back}");
                }
            }
        }
    }

    #[test]
    fn sinkhorn_comb_is_doubly_stochastic() {
        // mixes/base arbitrary; after enough Sinkhorn iters the 4×4 comb
        // matrix must be (near) doubly-stochastic: every row and column ≈ 1.
        let mixes: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.2).collect();
        let base: Vec<f32> = (0..24).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let (pre, post, comb) = dsv4_mhc_sinkhorn_split(&mixes, [1.0, 1.0, 1.0], &base, 1e-6, 20);
        assert_eq!((pre.len(), post.len(), comb.len()), (4, 4, 16));
        for i in 0..4 {
            let row: f32 = (0..4).map(|j| comb[i * 4 + j]).sum();
            let col: f32 = (0..4).map(|j| comb[j * 4 + i]).sum();
            assert!((row - 1.0).abs() < 1e-2, "row {i} sum {row}");
            assert!((col - 1.0).abs() < 1e-2, "col {i} sum {col}");
        }
        // pre is sigmoid+eps ∈ (0,1+eps); post is 2·sigmoid ∈ (0,2).
        assert!(pre.iter().all(|&x| x > 0.0 && x < 1.01));
        assert!(post.iter().all(|&x| x > 0.0 && x < 2.0));
    }
}

// ── Fused Mamba projection split (Part-B: 3 strided_col_copy → 1 dispatch) ─
//
// Splits proj [s, in_proj_out] into z [s, di], xbc [s, conv_dim], dt_raw [s, m_nh]
// in ONE device dispatch. Replaces the 3 sequential `strided_col_copy` launches
// that were 20-37% of prefill time at S=512-2048 (138 dispatches/forward).
pub fn mamba_split_proj(
    dev: &dyn Device,
    proj: &Tensor,
    s: usize,
    in_proj_out: usize,
    di: usize,
    conv_dim: usize,
    m_nh: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let kernel = cached_ir("mamba_split_proj", DType::F32, || {
        use wh_iron_core::ir::KernelMode;
        let mut k = wh_iron_std::kernels::ssm::scan::iron_mamba_split_proj::kernel_ir_for();
        k.mode = KernelMode::Grid3D;
        k
    });
    let z_out  = Tensor::empty(dev, vec![s * di], DType::F32)?;
    let xbc_out = Tensor::empty(dev, vec![s * conv_dim], DType::F32)?;
    let dt_out  = Tensor::empty(dev, vec![s * m_nh], DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(proj.buffer.clone()),
            Binding::Buffer(z_out.buffer.clone()),
            Binding::Buffer(xbc_out.buffer.clone()),
            Binding::Buffer(dt_out.buffer.clone()),
            u(in_proj_out as u32),
            u(di as u32),
            u(conv_dim as u32),
            u(m_nh as u32),
        ],
        // PERF: pack threads/block (was block:[1,1,1] = 1 thread/block, 1/32 of a
        // warp, this split was ~its full cost in launch overhead, not memory). The
        // kernel uses a global linear element id (gid_x), so block size is transparent
        // as long as the grid covers exactly s*in_proj_out threads. in_proj_out (10304
        // = 64*161) is a multiple of 64 → s*in_proj_out always divisible by 64.
        {
            let nt = (s * in_proj_out) as u32;
            let b = if nt % 512 == 0 { 512u32 } else if nt % 256 == 0 { 256u32 } else if nt % 64 == 0 { 64u32 } else if nt % 32 == 0 { 32u32 } else { 1u32 };
            Grid { grid: [nt / b, 1, 1], block: [b, 1, 1] }
        },
    )?;
    Ok((z_out, xbc_out, dt_out))
}
const MAMBA_SPLIT_V4_SRC: &str = r#"
#include <cuda_runtime.h>

extern "C" __global__ void mamba_split_proj_v4(
    const float* __restrict__ proj,
    float* __restrict__ z_out,
    float* __restrict__ xbc_out,
    float* __restrict__ dt_out,
    int in_proj_out4, int di4, int conv_dim4, int m_nh4, int total4)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total4) return;
    int row4 = di4 + conv_dim4 + m_nh4;
    int ti = idx / row4;
    int ci4 = idx - ti * row4;
    const float4* __restrict__ src4 = reinterpret_cast<const float4*>(proj);
    float4 v = src4[(size_t)ti * in_proj_out4 + ci4];
    if (ci4 < di4) {
        reinterpret_cast<float4*>(z_out)[(size_t)ti * di4 + ci4] = v;
    } else if (ci4 < di4 + conv_dim4) {
        int j = ci4 - di4;
        reinterpret_cast<float4*>(xbc_out)[(size_t)ti * conv_dim4 + j] = v;
    } else {
        int j = ci4 - di4 - conv_dim4;
        reinterpret_cast<float4*>(dt_out)[(size_t)ti * m_nh4 + j] = v;
    }
}

__device__ __forceinline__ float mamba_softplus(float x) {
    return x > 20.0f ? x : logf(1.0f + expf(x));
}

extern "C" __global__ void mamba_split_proj_softplus_v4(
    const float* __restrict__ proj,
    const float* __restrict__ bias,
    float* __restrict__ z_out,
    float* __restrict__ xbc_out,
    float* __restrict__ dt_out,
    int in_proj_out4, int di4, int conv_dim4, int m_nh4, int total4)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total4) return;
    int row4 = di4 + conv_dim4 + m_nh4;
    int ti = idx / row4;
    int ci4 = idx - ti * row4;
    const float4* __restrict__ src4 = reinterpret_cast<const float4*>(proj);
    float4 v = src4[(size_t)ti * in_proj_out4 + ci4];
    if (ci4 < di4) {
        reinterpret_cast<float4*>(z_out)[(size_t)ti * di4 + ci4] = v;
    } else if (ci4 < di4 + conv_dim4) {
        int j = ci4 - di4;
        reinterpret_cast<float4*>(xbc_out)[(size_t)ti * conv_dim4 + j] = v;
    } else {
        int j = ci4 - di4 - conv_dim4;
        int b = j << 2;
        v.x = mamba_softplus(v.x + bias[b + 0]);
        v.y = mamba_softplus(v.y + bias[b + 1]);
        v.z = mamba_softplus(v.z + bias[b + 2]);
        v.w = mamba_softplus(v.w + bias[b + 3]);
        reinterpret_cast<float4*>(dt_out)[(size_t)ti * m_nh4 + j] = v;
    }
}
"#;

/// Fused raw-float4 in_proj split (z/xbc/dt) WITH dt-bias+softplus folded into
/// the dt band. Replaces mamba_split_proj + softplus_add_rows. Escape with
/// NEMOTRON_MAMBA_SPLIT_RAW_OFF=1 or NEMOTRON_MAMBA_SPLIT_SOFTPLUS_OFF=1.
pub fn mamba_split_proj_softplus(
    dev: &dyn Device,
    proj: &Tensor,
    bias: &Tensor,
    s: usize,
    in_proj_out: usize,
    di: usize,
    conv_dim: usize,
    m_nh: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let raw_ok = std::env::var("NEMOTRON_MAMBA_SPLIT_RAW_OFF").is_err()
        && std::env::var("NEMOTRON_MAMBA_SPLIT_SOFTPLUS_OFF").is_err()
        && proj.dtype == DType::F32 && bias.dtype == DType::F32
        && proj.offset % 16 == 0 && bias.offset % 4 == 0
        && di % 4 == 0 && conv_dim % 4 == 0 && m_nh % 4 == 0 && in_proj_out % 4 == 0;
    if raw_ok {
        let z_out  = Tensor::empty(dev, vec![s * di], DType::F32)?;
        let xbc_out = Tensor::empty(dev, vec![s * conv_dim], DType::F32)?;
        let dt_out  = Tensor::empty(dev, vec![s * m_nh], DType::F32)?;
        let i = |v: i32| v.to_le_bytes().to_vec();
        let total4 = (s * (di + conv_dim + m_nh) / 4) as u32;
        if dev.dispatch_raw_cuda(
            MAMBA_SPLIT_V4_SRC,
            "mamba_split_v4.cu",
            "mamba_split_proj_softplus_v4",
            &[
                (proj.buffer.as_ref(), proj.offset),
                (bias.buffer.as_ref(), bias.offset),
                (z_out.buffer.as_ref(), 0),
                (xbc_out.buffer.as_ref(), 0),
                (dt_out.buffer.as_ref(), 0),
            ],
            &[i((in_proj_out / 4) as i32), i((di / 4) as i32), i((conv_dim / 4) as i32), i((m_nh / 4) as i32), i(total4 as i32)],
            [total4.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            false,
        ).is_ok() {
            return Ok((z_out, xbc_out, dt_out));
        }
    }
    let (z, xbc, dt_raw) = mamba_split_proj(dev, proj, s, in_proj_out, di, conv_dim, m_nh)?;
    let dt = softplus_add_rows(dev, &dt_raw, bias, s, m_nh)?;
    Ok((z, xbc, dt))
}


// ── Fused Mamba conv-output split (Part-B: 3 strided_col_copy → 1 dispatch) ─
//
// Splits yc_silu [s, conv_dim] into x [s, di], b [s, ng*ds], c [s, ng*ds]
// in ONE device dispatch.
pub fn mamba_split_conv(
    dev: &dyn Device,
    yc_silu: &Tensor,
    s: usize,
    conv_dim: usize,
    di: usize,
    ng_ds: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let kernel = cached_ir("mamba_split_conv", DType::F32, || {
        use wh_iron_core::ir::KernelMode;
        let mut k = wh_iron_std::kernels::ssm::scan::iron_mamba_split_conv::kernel_ir_for();
        k.mode = KernelMode::Grid3D;
        k
    });
    let x_out  = Tensor::empty(dev, vec![s * di], DType::F32)?;
    let b_out  = Tensor::empty(dev, vec![s * ng_ds], DType::F32)?;
    let c_out  = Tensor::empty(dev, vec![s * ng_ds], DType::F32)?;
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(yc_silu.buffer.clone()),
            Binding::Buffer(x_out.buffer.clone()),
            Binding::Buffer(b_out.buffer.clone()),
            Binding::Buffer(c_out.buffer.clone()),
            u(conv_dim as u32),
            u(di as u32),
            u(ng_ds as u32),
        ],
        // PERF: 256 threads/block (was 1). Global linear element id; conv_dim (6144)
        // is a multiple of 256 → s*conv_dim divisible by 256, exact grid, no OOB.
        { let n = (s * conv_dim) as u32; let b = 256u32; Grid { grid: [n / b, 1, 1], block: [b, 1, 1] } },
    )?;
    Ok((x_out, b_out, c_out))
}

// ── Batched gated group RMSNorm (Mamba2, prefill) ─────────────────────────
//
// Processes S tokens in one dispatch.  Each thread-group handles one (token,
// norm-group) pair.  Replaces the host dl+compute+upload loop in the
// CONV_DEVICE prefill path (TODO comment at line ~1543 of modeltests/lib.rs).
/// RMSNorm with f16 output only. Equivalent to `cast_f32_f16(rms_norm(...))`,
/// but avoids the intermediate f32 norm tensor and the separate cast launch.
/// TODO(rebrand-merge): the fast-path fused kernel (`iron_rms_norm_f16out`)
/// isn't in `thewafflehaus/iron@dev` yet, so this always takes the
/// `rms_norm` + `cast_f32_f16` fallback below (same numerics, one extra
/// dispatch) instead of the single-pass fused kernel.
pub fn rms_norm_f16out(dev: &dyn Device, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let out = rms_norm(dev, x, weight, eps)?;
    cast_f32_f16(dev, &out)
}

/// TODO(rebrand-merge): the fused kernel (`iron_gated_group_rmsnorm_batched_f16out`)
/// isn't in `thewafflehaus/iron@dev` yet (only the plain, non-f16out
/// `iron_gated_group_rmsnorm_batched` is) — always takes the
/// `gated_group_rmsnorm_batched` + `cast_f32_f16` fallback below.
pub fn gated_group_rmsnorm_batched_f16out(
    dev: &dyn Device,
    y: &Tensor,
    z: &Tensor,
    w: &Tensor,
    eps: f32,
    s: usize,
    di: usize,
    gs: usize,
) -> Result<Tensor> {
    let out = gated_group_rmsnorm_batched(dev, y, z, w, eps, s, di, gs)?;
    cast_f32_f16(dev, &out)
}

pub fn gated_group_rmsnorm_batched(
    dev: &dyn Device,
    y: &Tensor,
    z: &Tensor,
    w: &Tensor,
    eps: f32,
    s: usize,
    di: usize,
    gs: usize,
) -> Result<Tensor> {
    let ng = di / gs;
    let kernel = cached_ir("gated_group_rmsnorm_batched", DType::F32, || {
        use wh_iron_core::ir::KernelMode;
        let mut k = wh_iron_std::kernels::ssm::scan::iron_gated_group_rmsnorm_batched::kernel_ir_for();
        k.mode = KernelMode::Reduction;
        k
    });
    let out = Tensor::empty(dev, vec![s * di], DType::F32)?;
    let eps_buf = Tensor::new(scalar_buf(dev, eps.to_bits())?, vec![1], DType::F32);
    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    dev.dispatch(
        &kernel,
        &[
            Binding::Buffer(y.buffer.clone()),
            Binding::Buffer(z.buffer.clone()),
            Binding::Buffer(w.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Buffer(eps_buf.buffer.clone()),
            u(gs as u32),
            u(ng as u32),
        ],
        Grid { grid: [(s * ng) as u32, 1, 1], block: [(gs / 4) as u32, 1, 1] },
    )?;
    Ok(out)
}

const BATCHED_DEQUANT_Q4_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void moe_batched_dequant_q4(
    const unsigned int* __restrict__ qs,
    const __half*       __restrict__ sc,
    const unsigned int* __restrict__ eid,
    __half*             __restrict__ out,
    int n_active, int n_out, int k_in, int bpr_in
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_active * n_out * k_in) return;
    int slot   = i / (n_out * k_in);
    int local  = i - slot * n_out * k_in;
    int r      = local / k_in;
    int c      = local - r * k_in;
    int expert = (int)eid[slot];
    int b      = c / 32;
    int j      = c % 32;
    int w_idx  = j / 8;
    int nib_idx= j % 8;
    int global_blk = expert * n_out * bpr_in + r * bpr_in + b;
    unsigned int word = qs[global_blk * 4 + w_idx];
    unsigned int nib  = (word >> (nib_idx * 4)) & 0xFu;
    int q_signed  = (nib >= 8u) ? (int)nib - 16 : (int)nib;
    float scale   = __half2float(sc[global_blk]);
    __half val    = __float2half((float)q_signed * scale);
    out[i] = val;
}
"#;

/// Fused grouped-GEMM MoE expert pass (CUDA 13+, `NEMOTRON_GROUPED_GEMM=1`).
///
/// ONE batched NVRTC dequant kernel (`BATCHED_DEQUANT_Q4_SRC`) dequants ALL active
/// experts' weights into a contiguous per-slot scratch buffer in a SINGLE dispatch
/// (thread `i` → slot `i/(n_out*k_in)` → correct slot offset; reads `eid[slot]` for
/// the global expert id). Then per-expert `gemm_tc_off` reads from each slot. The
/// per-slot writes are what the plain `dispatch`+`iron_dequant_q4` path could NOT do
/// (that kernel always writes `out[0..n_elem]`, so every expert clobbered slot 0).
///
/// Device `relu2_scale_f16` eliminates the host relu² round-trip. No host syncs
/// between experts; everything stays on device.
///
/// Uses pre-allocated scratch (`up_scratch`/`dn_scratch`) to avoid per-layer
/// `cuMemAlloc`. Slot `i` of the UP scratch is `i*inter*hid*2` bytes; DN is
/// `i*hid*inter*2`. Scratch must hold at least `n_active` slots.
///
/// Returns `[mt, hid]` f16 output (relu² applied, 1/256 scaled). Caller scatter-adds.
const MOE_W4A16_SRC: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void moe_w4a16(
    const unsigned int* __restrict__ qs,      // Q4 weights: n_exp*n_out*(k_in/32)*4
    const __half*       __restrict__ scales,  // scales: n_exp*n_out*(k_in/32)
    const __half*       __restrict__ x,       // [m_total, k_in] f16, sorted by expert
    const unsigned int* __restrict__ indices, // [m_total] expert ids
    __half*             __restrict__ out,     // [m_total, n_out] f16
    int m_total, int n_out, int k_in)
{
    // ── Tile / warp coordinates ────────────────────────────────────────────
    const int n_tile_base = blockIdx.x * 64;
    const int m_tile_base = blockIdx.y * 64;
    const int warp_id     = threadIdx.x >> 5;
    // 2×2 warp layout: warp_id 0→(m=0,n=0), 1→(m=0,n=32), 2→(m=32,n=0), 3→(m=32,n=32)
    (void)(threadIdx.x & 31); // lane, unused in this kernel (warp_id sufficient)
    const int warp_m_base = (warp_id >> 1) * 32; // 0 or 32
    const int warp_n_base = (warp_id  & 1) * 32; // 0 or 32

    // ── Shared memory layout ───────────────────────────────────────────────
    // smem_X  [64×32] f16  = 4096 bytes, row-major [m_local, k_local]
    // smem_WT [32×64] f16  = 4096 bytes, row-major [k_local, n_local] = W transposed
    // smem_C  [64×64] f32  = 16384 bytes, output accumulation tile
    __shared__ __half smem_X [64 * 32]; // [m_local][k_local]
    __shared__ __half smem_WT[32 * 64]; // [k_local][n_local]  ← W^T
    __shared__ float  smem_C [64 * 64];

    const int bpr             = k_in / 32;
    const int nblk_per_expert = n_out * bpr;

    // ── WMMA accumulators: 2 m-tiles × 2 n-tiles (each 16×16) per warp ───
    // warp_m_base covers 32 rows  = 2 × m16, warp_n_base covers 32 cols = 2 × n16.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];

    // ── Per-expert sub-run loop (same as moe_bgemm_q4_bm64) ───────────────
    int sub_offset = 0;
    for (int _sub = 0; _sub < 64; _sub++) {
        int cur_row = m_tile_base + sub_offset;
        if (sub_offset >= 64 || cur_row >= m_total) break;
        unsigned cur_exp = indices[cur_row];

        // Find sub-run end.
        int sub_end = sub_offset;
        for (int ii = sub_offset; ii < 64; ii++) {
            int r = m_tile_base + ii;
            if (r >= m_total || indices[r] != cur_exp) break;
            sub_end = ii + 1;
        }

        // Reset accumulators.
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::fill_fragment(c_frag[mi][ni], 0.0f);

        // ── K-loop: BK=32 per step ─────────────────────────────────────────
        for (int kb = 0; kb < k_in; kb += 32) {

            // Stage X: 128 threads load 64×32 = 2048 halves (16 each).
            // Thread t → row = t/2, k_half = (t&1)*16 → 16 consecutive k-values.
            {
                int t          = threadIdx.x;
                int local_row  = t >> 1;
                int k_half     = (t & 1) << 4;     // 0 or 16
                int global_row = m_tile_base + local_row;
                bool in_run    = (local_row >= sub_offset) & (local_row < sub_end) & (global_row < m_total);
                // Clamp global_row to avoid OOB pointer arithmetic when out of run.
                int safe_row   = in_run ? global_row : 0;
                const __half* xsrc = x + (size_t)safe_row * k_in + kb + k_half;
                __half*       xdst = smem_X + local_row * 32 + k_half;
                __half zero = __float2half(0.f);
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    xdst[i] = in_run ? xsrc[i] : zero;
            }

            // Dequant W → smem_WT[k_local * 64 + n_local].
            // 128 threads × 16 elements = 2048 halves covering [32, 64].
            // Thread t → flat=t*16+i → k_local=flat/64, n_local=flat%64.
            {
                int t = threadIdx.x;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int flat    = t * 16 + i;           // 0..2047
                    int k_local = flat >> 6;            // 0..31
                    int n_local = flat & 63;            // 0..63
                    int global_n = n_tile_base + n_local;
                    int k        = kb + k_local;
                    // Q4 block for (global_n, k):
                    int row_blk  = global_n * bpr + k / 32;         // within-expert row block
                    int blk      = (int)cur_exp * nblk_per_expert + row_blk;
                    int lane_in_blk = k & 31;
                    unsigned word = qs[(size_t)blk * 4 + lane_in_blk / 8];
                    unsigned nib  = (word >> ((lane_in_blk & 7) * 4)) & 0xfu;
                    int q_s       = (int)(nib >= 8u ? nib - 16u : nib);
                    float sc_f    = __half2float(scales[blk]);
                    smem_WT[k_local * 64 + n_local] = __float2half((float)q_s * sc_f);
                }
            }
            __syncthreads();

            // ── WMMA: 2 k-steps × 2 m-tiles × 2 n-tiles per warp ─────────
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {         // 2 k-steps of 16 in BK=32
                int k_off = ki * 16;
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    // A: smem_X[(warp_m_base + mi*16)*32 + k_off], ld=32
                    wmma::load_matrix_sync(a_frag,
                        smem_X + (warp_m_base + mi * 16) * 32 + k_off,
                        32);
                    #pragma unroll
                    for (int ni = 0; ni < 2; ni++) {
                        // B: smem_WT[k_off * 64 + (warp_n_base + ni*16)], ld=64
                        wmma::load_matrix_sync(b_frag,
                            smem_WT + k_off * 64 + (warp_n_base + ni * 16),
                            64);
                        wmma::mma_sync(c_frag[mi][ni], a_frag, b_frag, c_frag[mi][ni]);
                    }
                }
            }
            __syncthreads();
        } // end K-loop

        // ── Store C frags → smem_C ─────────────────────────────────────────
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                wmma::store_matrix_sync(
                    smem_C + (warp_m_base + mi * 16) * 64 + (warp_n_base + ni * 16),
                    c_frag[mi][ni], 64, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ── Write smem_C → global out (only rows in this expert's run) ─────
        {
            int t = threadIdx.x;
            #pragma unroll
            for (int e = 0; e < 32; e++) {
                int flat      = t * 32 + e;
                int local_row = flat >> 6;   // 0..63
                int local_col = flat & 63;   // 0..63
                int global_m  = m_tile_base + local_row;
                int global_n  = n_tile_base + local_col;
                bool valid    = (local_row >= sub_offset) & (local_row < sub_end)
                              & (global_m < m_total) & (global_n < n_out);
                if (valid)
                    out[(size_t)global_m * n_out + global_n] =
                        __float2half(smem_C[local_row * 64 + local_col]);
            }
        }
        __syncthreads();

        sub_offset = sub_end;
    } // end per-expert sub-run loop
}
"#;

// ── Marlin-permuted Q4 layout ─────────────────────────────────────────────
//
// `permute_q4_to_marlin` converts the standard row-major Q4 layout produced
// by `quantize_q4` into a tile-major "Marlin" layout that enables fully
// coalesced 128-bit global loads in `moe_w4a16_marlin`.
//
// Standard layout (per expert, row r = output neuron):
//   qs[expert * n_out * bpr * 4  +  r * bpr * 4  +  b * 4  +  word]
//   where bpr = k_in/32, each u32 holds 8 nibbles for k-positions word*8..word*8+8
//   of block b (k-range b*32..(b+1)*32).
//
// Marlin tile layout (what `moe_w4a16_marlin` expects):
//   For each expert e, each 64-row n-tile nt (n_tile_base = nt*64), each k-tile
//   kt (k_tile_base = kt*32 = the kb value):
//     tile_base = e*(n_tiles*k_tiles*256) + nt*(k_tiles*256) + kt*256
//     u32 at tile_base + j  (j ∈ 0..256):
//       k_local   = j % 32          (position within BK=32 tile)
//       row_group = j / 32          (0..7, each group = 8 consecutive rows)
//       nibble i  → output row (row_group*8 + i), k = k_tile_base + k_local
//
// This layout makes thread `t` load tile_base+t and tile_base+t+128, two
// consecutive addresses, covering the full 64×32 tile with 4 cache lines
// per warp per load, vs ~64 scattered cache lines in the standard layout.
//
// `n_exp`  = total experts in the weight pool (all experts, packed contiguously)
// `n_out`  = output features per expert (must be multiple of 64)
// `k_in`   = input features (must be multiple of 32)
// `qs_std` = standard `quantize_q4` output: `[n_exp * n_out * bpr * 4]` u32
// Returns: `[n_exp * (n_out/64) * (k_in/32) * 256]` u32 (same total byte count)
pub fn permute_q4_to_marlin(
    qs_std: &[u32],
    n_exp: usize,
    n_out: usize,
    k_in: usize,
) -> Vec<u32> {
    assert_eq!(n_out % 64, 0, "permute_q4_to_marlin: n_out must be multiple of 64");
    assert_eq!(k_in  % 32, 0, "permute_q4_to_marlin: k_in must be multiple of 32");
    let bpr     = k_in  / 32; // Q4 blocks per weight row
    let n_tiles = n_out / 64; // number of 64-row n-tiles per expert
    let k_tiles = k_in  / 32; // = bpr

    // Total u32s is identical to input: n_exp * n_out * bpr * 4 = n_exp * n_tiles * k_tiles * 256
    // Verify: n_tiles * 256 = (n_out/64) * 256 = n_out * 4; k_tiles = bpr → same total ✓
    let total = n_exp * n_tiles * k_tiles * 256;
    assert_eq!(qs_std.len(), n_exp * n_out * bpr * 4);
    let mut out = vec![0u32; total];

    for e in 0..n_exp {
        let exp_std_base = e * n_out * bpr * 4;
        let exp_mar_base = e * n_tiles * k_tiles * 256;
        for nt in 0..n_tiles {
            for kt in 0..k_tiles {
                let tile_mar_base = exp_mar_base + nt * k_tiles * 256 + kt * 256;
                // Tile covers output rows [nt*64 .. (nt+1)*64) and
                // k-positions [kt*32 .. (kt+1)*32).
                // Fill 256 u32s: u32 at j → k_local=j%32, row_group=j/32.
                for j in 0..256usize {
                    let k_local   = j % 32;
                    let row_group = j / 32; // 0..7
                    // Pack 8 nibbles for consecutive rows row_group*8..row_group*8+8
                    // at k-position k_tile_base + k_local.
                    let k_abs = kt * 32 + k_local;   // absolute k-position
                    let k_blk = k_abs / 32;           // = kt (k_abs is already kt*32+k_local)
                    debug_assert_eq!(k_blk, kt);
                    let k_in_blk = k_abs % 32;        // = k_local (position within block)
                    let word_in_blk = k_in_blk / 8;  // which of the 4 u32s in the block
                    let nib_in_word = k_in_blk % 8;  // which nibble in that u32

                    let mut packed = 0u32;
                    for i in 0..8usize {
                        let row = nt * 64 + row_group * 8 + i; // absolute output row
                        // Standard layout address for this (row, k_abs):
                        // qs_std[exp_std_base + row * bpr * 4 + k_blk * 4 + word_in_blk]
                        let src_word = qs_std[exp_std_base + row * bpr * 4 + k_blk * 4 + word_in_blk];
                        let nib = (src_word >> (nib_in_word * 4)) & 0xf;
                        packed |= nib << (i * 4);
                    }
                    out[tile_mar_base + j] = packed;
                }
            }
        }
    }
    out
}

// ── Inverse: Marlin → standard layout (for validation) ─────────────────────
pub fn permute_q4_from_marlin(
    qs_mar: &[u32],
    n_exp: usize,
    n_out: usize,
    k_in: usize,
) -> Vec<u32> {
    assert_eq!(n_out % 64, 0);
    assert_eq!(k_in  % 32, 0);
    let bpr     = k_in  / 32;
    let n_tiles = n_out / 64;
    let k_tiles = bpr;
    let mut out = vec![0u32; n_exp * n_out * bpr * 4];

    for e in 0..n_exp {
        let exp_std_base = e * n_out * bpr * 4;
        let exp_mar_base = e * n_tiles * k_tiles * 256;
        for nt in 0..n_tiles {
            for kt in 0..k_tiles {
                let tile_mar_base = exp_mar_base + nt * k_tiles * 256 + kt * 256;
                for j in 0..256usize {
                    let k_local   = j % 32;
                    let row_group = j / 32;
                    let k_abs    = kt * 32 + k_local;
                    let k_blk    = kt;
                    let k_in_blk = k_local;
                    let word_in_blk = k_in_blk / 8;
                    let nib_in_word = k_in_blk % 8;
                    let src_word = qs_mar[tile_mar_base + j];
                    for i in 0..8usize {
                        let row = nt * 64 + row_group * 8 + i;
                        let nib = (src_word >> (i * 4)) & 0xf;
                        let dst_idx = exp_std_base + row * bpr * 4 + k_blk * 4 + word_in_blk;
                        // Clear the nibble slot and set it.
                        out[dst_idx] &= !(0xf << (nib_in_word * 4));
                        out[dst_idx] |=  nib  << (nib_in_word * 4);
                    }
                    let _ = k_abs; // used only for assertion above
                }
            }
        }
    }
    out
}

// ── Marlin-layout W4A16 WMMA kernel ───────────────────────────────────────
//
// Same contract as `moe_w4a16` but expects `qs` in Marlin tile-major layout
// (as produced by `permute_q4_to_marlin`). Replaces the per-nibble scattered
// read loop with two coalesced u32 loads per thread covering the full 64×32
// tile, cutting Q4 global reads from ~64 scattered cache lines to 4 cache
// lines per warp per K-step.
//
// Tiling: 64(M) × 64(N) × 32(K) per block, 4 warps (128 threads), 2×2 warp
// layout, double-buffer not used (smem fits in one phase: X=4KB, WT=4KB, C=16KB).
// Grid: [n_out/64, ceil(m_total/64)].
const MOE_W4A16_MARLIN_SRC: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// ── Kernel: moe_w4a16_marlin ───────────────────────────────────────────────
// qs layout: Marlin tile-major. For expert e, n-tile nt, k-tile kt (= kb/32):
//   tile_base = e*(n_tiles*k_tiles*256) + nt*(k_tiles*256) + kt*256
//   u32 at tile_base+j (j in 0..256):
//     k_local = j%32, row_group = j/32
//     nibble i → output row row_group*8+i at k-position kt*32+k_local
//
// Thread t loads tile_base+t and tile_base+t+128 → perfectly coalesced.
extern "C" __global__ void moe_w4a16_marlin(
    const unsigned int* __restrict__ qs,      // Marlin Q4: n_exp*(n_out/64)*(k_in/32)*256
    const __half*       __restrict__ scales,  // standard: n_exp*n_out*(k_in/32) f16
    const __half*       __restrict__ x,       // [m_total, k_in] f16, sorted by expert
    const unsigned int* __restrict__ indices, // [m_total] expert ids
    __half*             __restrict__ out,     // [m_total, n_out] f16
    int m_total, int n_out, int k_in)
{
    // ── Tile / warp coordinates ────────────────────────────────────────────
    const int n_tile_base = blockIdx.x * 64;
    const int m_tile_base = blockIdx.y * 64;
    const int warp_id     = threadIdx.x >> 5;
    const int lane        = threadIdx.x & 31;
    // 2×2 warp layout: warp 0→(m=0,n=0), 1→(m=0,n=32), 2→(m=32,n=0), 3→(m=32,n=32)
    const int warp_m_base = (warp_id >> 1) * 32;
    const int warp_n_base = (warp_id  & 1) * 32;
    const int t           = threadIdx.x;  // 0..127

    // ── Shared memory layout ───────────────────────────────────────────────
    // smem_X  [64×32] f16  = 4096 bytes, row-major [m_local, k_local]
    // smem_WT [32×64] f16  = 4096 bytes, row-major [k_local, n_local] (W^T)
    // smem_C  [64×64] f32  = 16384 bytes, output accumulation tile
    __shared__ __half smem_X [64 * 32];
    __shared__ __half smem_WT[32 * 64];
    __shared__ float  smem_C [64 * 64];

    const int bpr             = k_in / 32;   // Q4 blocks per weight row (= k_tiles)
    const int n_tiles         = n_out / 64;
    const int nblk_per_expert = n_out * bpr; // for scale indexing (unchanged layout)
    const int nt              = n_tile_base / 64; // n-tile index for this block

    // ── WMMA accumulators ─────────────────────────────────────────────────
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];

    // ── Per-expert sub-run loop ────────────────────────────────────────────
    int sub_offset = 0;
    for (int _sub = 0; _sub < 64; _sub++) {
        int cur_row = m_tile_base + sub_offset;
        if (sub_offset >= 64 || cur_row >= m_total) break;
        unsigned cur_exp = indices[cur_row];

        // Find sub-run end.
        int sub_end = sub_offset;
        for (int ii = sub_offset; ii < 64; ii++) {
            int r = m_tile_base + ii;
            if (r >= m_total || indices[r] != cur_exp) break;
            sub_end = ii + 1;
        }

        // Reset accumulators.
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::fill_fragment(c_frag[mi][ni], 0.0f);

        // Base of this expert's Marlin-packed Q4 in qs[]:
        // expert e owns n_tiles * bpr * 256 u32s.
        const unsigned int* qs_exp = qs + (size_t)cur_exp * n_tiles * bpr * 256;

        // ── K-loop: BK=32 per step ─────────────────────────────────────────
        for (int kb = 0; kb < k_in; kb += 32) {
            const int kt = kb / 32; // k-tile index

            // ── Stage X: 128 threads load 64×32 = 2048 halves ────────────
            // Thread t: row = t/2, k_half = (t&1)*16, loads 16 halves.
            {
                int local_row  = t >> 1;
                int k_half     = (t & 1) << 4;
                int global_row = m_tile_base + local_row;
                bool in_run    = (local_row >= sub_offset) & (local_row < sub_end) & (global_row < m_total);
                int safe_row   = in_run ? global_row : 0;
                const __half* xsrc = x + (size_t)safe_row * k_in + kb + k_half;
                __half*       xdst = smem_X + local_row * 32 + k_half;
                __half zero = __float2half(0.f);
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    xdst[i] = in_run ? xsrc[i] : zero;
            }

            // ── Stage WT: coalesced Marlin tile load + dequant ────────────
            // Tile base in qs_exp: nt * bpr * 256 + kt * 256
            // Thread t loads u32s at positions t and t+128 in the 256-u32 tile.
            // Each u32 encodes 8 nibbles: nibble i → output row row_group*8+i
            // at k_local within this k-tile, where row_group = j/32, k_local = j%32.
            {
                const unsigned int* tile = qs_exp + (size_t)nt * bpr * 256 + (size_t)kt * 256;

                // Load 0: j = t  (covers row_groups 0..3 for t in 0..127)
                {
                    unsigned word = tile[t];
                    int k_local   = t % 32;
                    int row_group = t / 32;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        int n_local = row_group * 8 + i;  // output row within tile
                        unsigned nib = (word >> (i * 4)) & 0xf;
                        int q_s = (int)(nib >= 8u ? nib - 16u : nib);
                        int blk = (n_tile_base + n_local) * bpr + kt;  // scale block
                        float sc = __half2float(scales[(size_t)cur_exp * nblk_per_expert + blk]);
                        smem_WT[k_local * 64 + n_local] = __float2half((float)q_s * sc);
                    }
                }

                // Load 1: j = t + 128
                {
                    unsigned word = tile[t + 128];
                    int j2        = t + 128;
                    int k_local   = j2 % 32;
                    int row_group = j2 / 32;  // 4..7 for t in 0..127
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        int n_local = row_group * 8 + i;
                        unsigned nib = (word >> (i * 4)) & 0xf;
                        int q_s = (int)(nib >= 8u ? nib - 16u : nib);
                        int blk = (n_tile_base + n_local) * bpr + kt;
                        float sc = __half2float(scales[(size_t)cur_exp * nblk_per_expert + blk]);
                        smem_WT[k_local * 64 + n_local] = __float2half((float)q_s * sc);
                    }
                }
            }
            __syncthreads();

            // ── WMMA: 2 k-steps × 2 m-tiles × 2 n-tiles per warp ─────────
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                int k_off = ki * 16;
                #pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    wmma::load_matrix_sync(a_frag,
                        smem_X + (warp_m_base + mi * 16) * 32 + k_off, 32);
                    #pragma unroll
                    for (int ni = 0; ni < 2; ni++) {
                        wmma::load_matrix_sync(b_frag,
                            smem_WT + k_off * 64 + (warp_n_base + ni * 16), 64);
                        wmma::mma_sync(c_frag[mi][ni], a_frag, b_frag, c_frag[mi][ni]);
                    }
                }
            }
            __syncthreads();
        } // end K-loop

        // ── Store C frags → smem_C ─────────────────────────────────────────
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::store_matrix_sync(
                    smem_C + (warp_m_base + mi * 16) * 64 + (warp_n_base + ni * 16),
                    c_frag[mi][ni], 64, wmma::mem_row_major);
        __syncthreads();

        // ── Write smem_C → global out ──────────────────────────────────────
        #pragma unroll
        for (int e = 0; e < 32; e++) {
            int flat      = t * 32 + e;
            int local_row = flat >> 6;
            int local_col = flat & 63;
            int global_m  = m_tile_base + local_row;
            int global_n  = n_tile_base + local_col;
            bool valid    = (local_row >= sub_offset) & (local_row < sub_end)
                          & (global_m < m_total) & (global_n < n_out);
            if (valid)
                out[(size_t)global_m * n_out + global_n] =
                    __float2half(smem_C[local_row * 64 + local_col]);
        }
        __syncthreads();

        sub_offset = sub_end;
    } // end per-expert sub-run loop
    (void)lane;
}
"#;

// ── 128×128×32 Marlin W4A16 WMMA kernel ───────────────────────────────────
//
// Upgrades from 64×64×32 (4-warp) to 128×128×32 (8-warp) tile:
//   - 4× accumulator area → 4× compute density at the same global bandwidth
//   - Scales preloaded into smem at the start of each K-iteration (128 f32
//     values, avoids 8 __half2float per dequant inner loop)
//   - Double-buffered X and WT staging (ping-pong, 2 smem slots each)
//
// Grid: [n_out/128, ceil(m_total/128)], block: [256, 1, 1] (8 warps).
// Warp layout (2 rows × 4 cols):
//   warp_m_base = (warp_id >> 2) * 64   → row 0 or 64 within 128-row M-tile
//   warp_n_base = (warp_id  & 3) * 32   → col 0, 32, 64, or 96 in 128-col N-tile
//   c_frag[4][2]: 4 mi × 2 ni = 8 WMMA frags covering 64×32 output per warp
//
// smem (per block):
//   smem_X [2][128*32] f16  = 16384 bytes  double-buffer
//   smem_WT[2][32*128] f16  = 16384 bytes  double-buffer
//   smem_SC[128]       f32  =   512 bytes  scale preload (128 × f32)
//   smem_C [128*128]   f32  = 65536 bytes  accumulator
//   Total                   = 98816 bytes  (~96KB, within GB10 228KB/SM)
//
// WT tile load (Marlin layout, two consecutive 64-N tiles per 128-N block):
//   First half  (n_local 0..63):  tile = qs_exp + nt*bpr*256 + kt*256
//   Second half (n_local 64..127): tile = qs_exp + (nt+1)*bpr*256 + kt*256
//   256 threads × 1 u32 each per tile → fully coalesced single-pass.
//   Each u32 → 8 nibbles → 8 rows → written to smem_WT[k_local*128 + n_local].
//
// `n_out % 128 == 0` required; `k_in % 32 == 0` required.
const MOE_W4A16_MARLIN128_SRC: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// ── Kernel: moe_w4a16_marlin128 ────────────────────────────────────────────
//
// 128(M)×128(N)×32(K) tile, 256 threads (8 warps), 2-row × 4-col warp layout.
//
// Three improvements over moe_w4a16_marlin (64×64×32):
//   1. 4× larger tile: 128×128 accumulator → 4× compute density.
//   2. Scale preload: 128 f32 scales loaded into smem at each K-step,
//      eliminating __half2float() inside the per-nibble dequant loop.
//   3. Double-buffered X and WT staging: next tile's loads overlap
//      current tile's WMMA compute (ping-pong smem slots).
//   4. Padded smem_WT row stride (128->136 halfwords) to break the
//      power-of-two bank-conflict alignment: a plain 128-halfword
//      (256-byte) row stride is an exact multiple of the 32-bank×4-byte
//      conflict period, so a warp's 32 threads (which vary the ROW index
//      while the column/bank-determining index stays fixed within an
//      unrolled dequant iteration) ALL alias the same bank on every
//      LOAD_WT store — confirmed via `ncu` (l1tex bank-conflict counters
//      vs wavefront counters): ~18x store-side overhead vs the
//      conflict-free ideal on this kernel. 136 is the smallest multiple
//      of 8 (WMMA's row-major `ldm` alignment requirement for __half)
//      that reduces the conflict: the achievable minimum under that
//      constraint is a 4-way conflict (down from 32-way); modular
//      arithmetic (row_wt/2 mod 32) shows 8-alignment forbids getting to
//      fully conflict-free for this 32-row/128-bank shape. smem_X (the
//      other bank-conflict offender ncu found, ~3.5x overhead, load-side)
//      is NOT padded: this device's actual opt-in dynamic-shared-memory
//      cap is 101376 bytes (queried via
//      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 101376,
//      NOT the ~228KB this comment previously assumed by analogy to a
//      different GPU) and there is no budget left to also pad X (16384
//      bytes at the padded WT size already leaves only ~1.5KB of
//      headroom) without exceeding it and failing `cuFuncSetAttribute`
//      at launch. Left as a documented follow-up if the tile/accumulator
//      layout is ever revisited to free up smem headroom.
//
// qs layout: Marlin tile-major, same 64-row tiles as moe_w4a16_marlin.
// A 128-N-tile block straddles two consecutive 64-N Marlin tiles (nt0, nt0+1).
// 256 threads × 1 u32/tile → both tiles load in a single fully-coalesced pass.
//
// smem breakdown (per block; WT row PADDED to 136 halfwords, X unpadded):
//   smem_X [2][128*32] f16 = 16384 B  (double-buffered X, unpadded)
//   smem_WT[2][32*136] f16 = 17408 B  (double-buffered W^T, padded)
//   smem_SC[128]       f32 =   512 B  (preloaded scales for this K-step)
//   smem_C [128*128]   f32 = 65536 B  (accumulator)
//   Total                  = 99840 B  (~97.5 KB; GB10 opt-in cap is 101376 B)
//
// n_out % 128 == 0 and k_in % 32 == 0 required.
extern "C" __global__ void moe_w4a16_marlin128(
    const unsigned int* __restrict__ qs,      // Marlin Q4: n_exp*(n_out/64)*(k_in/32)*256
    const __half*       __restrict__ scales,  // f16 scales: n_exp*n_out*(k_in/32)
    const __half*       __restrict__ x,       // [m_total, k_in] f16
    const unsigned int* __restrict__ indices, // [m_total] expert ids
    __half*             __restrict__ out,     // [m_total, n_out] f16
    int m_total, int n_out, int k_in)
{
    const int n_tile_base = blockIdx.x * 128;
    const int m_tile_base = blockIdx.y * 128;
    const int warp_id     = threadIdx.x >> 5;   // 0..7
    const int t           = threadIdx.x;         // 0..255

    // Warp covers [warp_m_base..+64) × [warp_n_base..+32) of the 128×128 tile.
    const int warp_m_base = (warp_id >> 2) * 64; // 0 or 64
    const int warp_n_base = (warp_id  & 3) * 32; // 0, 32, 64, or 96

    // smem layout (all offsets in bytes; WT row PADDED to 136, X unpadded):
    //   [0       .. 16384) smem_X [2][128*32] __half
    //   [16384   .. 33792) smem_WT[2][32*136] __half
    //   [33792   .. 34304) smem_SC[128]       float
    //   [34304   .. 99840) smem_C [128*128]   float
    extern __shared__ char smem_raw[];
    __half* smem_X  = (__half*)(smem_raw);
    __half* smem_WT = (__half*)(smem_raw + 16384);
    float*  smem_SC = (float* )(smem_raw + 33792);
    float*  smem_C  = (float* )(smem_raw + 34304);

    const int bpr             = k_in / 32;
    const int n_tiles64       = n_out / 64;
    const int nblk_per_expert = n_out * bpr;
    const int nt0             = blockIdx.x * 2;  // first 64-N Marlin tile index

    // 4×2 WMMA accumulator per warp: 4 m-frags × 2 n-frags = 64(m) × 32(n).
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][2];

    // ── Helper: load scales for kt into smem_SC ────────────────────────────
    // Threads 0..127 each load one scale; threads 128..255 idle.
    // Caller must __syncthreads() after this before reading smem_SC.
    #define LOAD_SCALES(kt_)                                                   \
        if (t < 128) {                                                         \
            int n_col = n_tile_base + t;                                       \
            smem_SC[t] = __half2float(                                         \
                scales[(size_t)cur_exp * nblk_per_expert + n_col * bpr + (kt_)]); \
        }

    // ── Helper: load X for kb_ into smem_X[slot_] ─────────────────────────
    // Thread t: local_row = t>>1, k_half = (t&1)<<4 → 16 halves.
    #define LOAD_X(slot_, kb_)                                                 \
        {                                                                      \
            int lr  = t >> 1, kh = (t & 1) << 4;                              \
            int gr  = m_tile_base + lr;                                        \
            bool ok = (lr >= sub_offset) & (lr < sub_end) & (gr < m_total);   \
            int sr  = ok ? gr : 0;                                             \
            const __half* xs = x + (size_t)sr * k_in + (kb_) + kh;           \
            __half*       xd = smem_X + (slot_) * (128*32) + lr * 32 + kh;   \
            __half zero = __float2half(0.f);                                   \
            _Pragma("unroll")                                                  \
            for (int _i = 0; _i < 16; _i++) xd[_i] = ok ? xs[_i] : zero;    \
        }

    // ── Helper: load WT for kt_ into smem_WT[slot_] ───────────────────────
    // Two loads: half-0 (n_local 0..63 from nt0) and half-1 (64..127 from nt0+1).
    // 256 threads × 1 u32 per tile (256 u32s/tile) → perfectly coalesced.
    // smem_SC must already be populated for kt_.
    #define LOAD_WT(slot_, kt_)                                                \
        {                                                                      \
            /* half-0: n_local 0..63, Marlin tile nt0 */                       \
            {                                                                  \
                unsigned w0 = qs_exp[(size_t)nt0 * bpr * 256 + (size_t)(kt_) * 256 + t]; \
                int kl = t % 32, rg = t / 32;                                 \
                _Pragma("unroll")                                               \
                for (int _i = 0; _i < 8; _i++) {                              \
                    int nl = rg * 8 + _i;                                      \
                    unsigned nib = (w0 >> (_i*4)) & 0xf;                      \
                    int qs_val = (int)(nib >= 8u ? nib - 16u : nib);          \
                    smem_WT[(slot_)*(32*136) + kl*136 + nl] =                 \
                        __float2half((float)qs_val * smem_SC[nl]);             \
                }                                                              \
            }                                                                  \
            /* half-1: n_local 64..127, Marlin tile nt0+1 */                   \
            {                                                                  \
                unsigned w1 = qs_exp[(size_t)(nt0+1) * bpr * 256 + (size_t)(kt_) * 256 + t]; \
                int kl = t % 32, rg = t / 32;                                 \
                _Pragma("unroll")                                               \
                for (int _i = 0; _i < 8; _i++) {                              \
                    int nl = 64 + rg * 8 + _i;                                \
                    unsigned nib = (w1 >> (_i*4)) & 0xf;                      \
                    int qs_val = (int)(nib >= 8u ? nib - 16u : nib);          \
                    smem_WT[(slot_)*(32*136) + kl*136 + nl] =                 \
                        __float2half((float)qs_val * smem_SC[nl]);             \
                }                                                              \
            }                                                                  \
        }

    // ── Per-expert sub-run loop ────────────────────────────────────────────
    int sub_offset = 0;
    for (int _sub = 0; _sub < 128; _sub++) {
        int cur_row = m_tile_base + sub_offset;
        if (sub_offset >= 128 || cur_row >= m_total) break;
        unsigned cur_exp = indices[cur_row];

        int sub_end = sub_offset;
        for (int ii = sub_offset; ii < 128; ii++) {
            int r = m_tile_base + ii;
            if (r >= m_total || indices[r] != cur_exp) break;
            sub_end = ii + 1;
        }

        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::fill_fragment(c_frag[mi][ni], 0.0f);

        const unsigned int* qs_exp = qs + (size_t)cur_exp * n_tiles64 * bpr * 256;

        // ── Prologue: fill smem_SC[kt=0], then load buf=0 X + WT ──────────
        LOAD_SCALES(0)
        __syncthreads();
        LOAD_X(0, 0)
        LOAD_WT(0, 0)
        __syncthreads();

        // ── Main K-loop: compute buf, prefetch next into 1-buf ─────────────
        for (int kb = 0; kb < k_in; kb += 32) {
            const int kt       = kb / 32;
            const int buf      = kt & 1;
            const int next_buf = buf ^ 1;
            const int next_kt  = kt + 1;
            const int next_kb  = kb + 32;

            // Start prefetching scales for next K-tile while we compute.
            // The __syncthreads() at the end of this iteration makes them visible
            // before the next LOAD_WT reads smem_SC.
            if (next_kb < k_in) {
                LOAD_SCALES(next_kt)
            }

            // WMMA compute on buf: 2 ki-steps × 4 mi-frags × 2 ni-frags.
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                const int k_off = ki * 16;
                #pragma unroll
                for (int mi = 0; mi < 4; mi++) {
                    wmma::load_matrix_sync(a_frag,
                        smem_X + buf * (128*32) + (warp_m_base + mi*16) * 32 + k_off,
                        32);
                    #pragma unroll
                    for (int ni = 0; ni < 2; ni++) {
                        wmma::load_matrix_sync(b_frag,
                            smem_WT + buf * (32*136) + k_off * 136 + (warp_n_base + ni*16),
                            136);
                        wmma::mma_sync(c_frag[mi][ni], a_frag, b_frag, c_frag[mi][ni]);
                    }
                }
            }
            __syncthreads(); // scales prefetch + compute done; safe to write next_buf

            if (next_kb < k_in) {
                LOAD_X(next_buf, next_kb)
                LOAD_WT(next_buf, next_kt)
                __syncthreads(); // next_buf ready for next iteration
            }
        } // end K-loop

        // ── Store accumulators → smem_C ────────────────────────────────────
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::store_matrix_sync(
                    smem_C + (warp_m_base + mi*16) * 128 + (warp_n_base + ni*16),
                    c_frag[mi][ni], 128, wmma::mem_row_major);
        __syncthreads();

        // ── Write smem_C → global out ──────────────────────────────────────
        // 256 threads × 64 elements = 16384 = 128×128 ✓
        #pragma unroll
        for (int e = 0; e < 64; e++) {
            int flat      = t * 64 + e;
            int local_row = flat >> 7;
            int local_col = flat & 127;
            int global_m  = m_tile_base + local_row;
            int global_n  = n_tile_base + local_col;
            bool valid    = (local_row >= sub_offset) & (local_row < sub_end)
                          & (global_m < m_total) & (global_n < n_out);
            if (valid)
                out[(size_t)global_m * n_out + global_n] =
                    __float2half(smem_C[local_row * 128 + local_col]);
        }
        __syncthreads();

        sub_offset = sub_end;
    } // end per-expert loop

    #undef LOAD_SCALES
    #undef LOAD_X
    #undef LOAD_WT
}
"#;

/// W4A16 MoE grouped GEMM, Marlin coalesced layout.
///
/// Drop-in replacement for `moe_w4a16` but requires `qs` to be in Marlin
/// tile-major layout (from `permute_q4_to_marlin`). Achieves coalesced
/// 128-bit global reads for Q4 weights: 4 cache lines per warp per K-step
/// vs ~64 in the standard nibble-scattered layout.
///
/// Dispatches the 128×128×32 tile (256-thread, 8-warp) kernel when
/// `n_out % 128 == 0` for higher compute density; falls back to the
/// 64×64×32 tile kernel otherwise (`n_out % 64 == 0` required).
/// `n_out % 128 == 0` for higher compute density; falls back to the
/// 64×64×32 tile kernel otherwise (`n_out % 64 == 0` required).
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16_marlin(
    dev: &dyn Device,
    x: &Tensor,       // [m_total, k_in] f16, sorted by expert
    qs: &Tensor,      // Marlin Q4 weights (permute_q4_to_marlin output)
    scales: &Tensor,  // f16 scales (unchanged from quantize_q4)
    indices: &Tensor, // [m_total] u32 expert ids
    m_total: usize,
    n_out: usize,
    k_in: usize,
) -> Result<Tensor> {
    if n_out % 64 != 0 {
        return Err(Error::Msg(format!("moe_w4a16_marlin: n_out {n_out} must be multiple of 64")));
    }
    let out = Tensor::empty(dev, vec![m_total, n_out], DType::F16)?;
    let m_total_i = m_total as i32;
    let n_out_i   = n_out   as i32;
    let k_in_i    = k_in    as i32;

    // Use the 128×128 tile only when BOTH n_out and k_in are multiples of 128
    // (the 128-tile strides K by 128; a non-128 k_in -- e.g. inter=1856 on the
    // down-GEMM -- would read past the weight buffer). Else fall to the 64-tile.
    if n_out % 128 == 0 && k_in % 128 == 0 {
        dev.dispatch_raw_cuda(
            MOE_W4A16_MARLIN128_SRC,
            "moe_w4a16_marlin128.cu",
            "moe_w4a16_marlin128",
            &[
                (qs.buffer.as_ref(),      qs.offset),
                (scales.buffer.as_ref(),  scales.offset),
                (x.buffer.as_ref(),       x.offset),
                (indices.buffer.as_ref(), indices.offset),
                (out.buffer.as_ref(),     0),
            ],
            &[
                m_total_i.to_le_bytes().to_vec(),
                n_out_i.to_le_bytes().to_vec(),
                k_in_i.to_le_bytes().to_vec(),
            ],
            [(n_out as u32) / 128, (m_total as u32).div_ceil(128), 1],
            [256, 1, 1],
            // smem: X_db(16384, unpadded) + WT_db(17408, padded 128->136) + SC(512)
            // + C(65536) = 99840 bytes (~97.5 KB). GB10's opt-in dynamic-smem cap is
            // 101376 bytes (queried via CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),
            // so this is the largest WT padding this device's budget allows without also
            // shrinking something else; see the kernel doc for the full rationale.
            99840,
            false,
        )?;
    } else {
        dev.dispatch_raw_cuda(
            MOE_W4A16_MARLIN_SRC,
            "moe_w4a16_marlin.cu",
            "moe_w4a16_marlin",
            &[
                (qs.buffer.as_ref(),      qs.offset),
                (scales.buffer.as_ref(),  scales.offset),
                (x.buffer.as_ref(),       x.offset),
                (indices.buffer.as_ref(), indices.offset),
                (out.buffer.as_ref(),     0),
            ],
            &[
                m_total_i.to_le_bytes().to_vec(),
                n_out_i.to_le_bytes().to_vec(),
                k_in_i.to_le_bytes().to_vec(),
            ],
            [(n_out as u32) / 64, (m_total as u32).div_ceil(64), 1],
            [128, 1, 1],
            // smem: X(4096) + WT(4096) + C(16384) = 24576 bytes
            24576,
            false,
        )?;
    }
    Ok(out)
}

/// W4A16 MoE grouped GEMM (Marlin-style, WMMA tensor cores, inline Q4 dequant).
///
/// Drop-in replacement for `moe_bgemm_q4_bm64`. Uses `mma.sync m16n16k16` with
/// weights staged as W^T in smem, no f16 weight scratch in global memory, no
/// 2× weight bandwidth. `n_out % 64 == 0` required.
///
/// Returns `[m_total, n_out]` f16 (same contract as `moe_bgemm_q4_bm64`).
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a16(
    dev: &dyn Device,
    x: &Tensor,       // [m_total, k_in] f16, sorted by expert
    qs: &Tensor,      // Q4 weights
    scales: &Tensor,  // f16 scales
    indices: &Tensor, // [m_total] u32 expert ids
    m_total: usize,
    n_out: usize,
    k_in: usize,
) -> Result<Tensor> {
    if n_out % 64 != 0 {
        return Err(Error::Msg(format!("moe_w4a16: n_out {n_out} must be multiple of 64")));
    }
    let out = Tensor::empty(dev, vec![m_total, n_out], DType::F16)?;
    let m_total_i = m_total as i32;
    let n_out_i   = n_out   as i32;
    let k_in_i    = k_in    as i32;
    dev.dispatch_raw_cuda(
        MOE_W4A16_SRC,
        "moe_w4a16.cu",
        "moe_w4a16",
        &[
            (qs.buffer.as_ref(),      qs.offset),
            (scales.buffer.as_ref(),  scales.offset),
            (x.buffer.as_ref(),       x.offset),
            (indices.buffer.as_ref(), indices.offset),
            (out.buffer.as_ref(),     0),
        ],
        &[
            m_total_i.to_le_bytes().to_vec(),
            n_out_i.to_le_bytes().to_vec(),
            k_in_i.to_le_bytes().to_vec(),
        ],
        [(n_out as u32) / 64, (m_total as u32).div_ceil(64), 1],
        [128, 1, 1],
        // smem: X(4096) + WT(4096) + C(16384) = 24576 bytes
        24576,
        false,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gemm(
    dev: &dyn Device,
    qs_up: &Tensor,
    sc_up: &Tensor,
    qs_dn: &Tensor,
    sc_dn: &Tensor,
    xs_f16: &Tensor,        // [mt, hid] f16: rows sorted by expert id
    g_starts: &[usize],     // length = n_active + 1
    expert_ids: &[usize],   // expert id for each group
    hid: usize,
    inter: usize,
    bpr_up: usize,          // hid / 32
    bpr_dn: usize,          // inter / 32
    up_scratch: Option<&Tensor>, // Pre-allocated [n_exp*inter, hid] f16
    dn_scratch: Option<&Tensor>, // Pre-allocated [n_exp*hid, inter] f16
) -> Result<Tensor> {
    let n_active = expert_ids.len();
    let mt = g_starts.last().copied().unwrap_or(0);
    if n_active == 0 || mt == 0 {
        return Tensor::empty(dev, vec![mt.max(1), hid], DType::F16);
    }

    // Experts are processed in chunks of `chunk`: each chunk's weights are
    // dequanted (one batched NVRTC launch) into distinct slots of the scratch,
    // then GEMM'd. Smaller chunks keep the just-dequanted f16 weight L2-cache-hot
    // for its GEMM (a full-slab dequant evicts the cache → the GEMM re-reads from
    // HBM, ~2× weight bandwidth). Measured best at chunk=1 on GB10 (per-expert
    // dequant→GEMM, cache-hot). NEMOTRON_MOE_CHUNK overrides (default 1).
    // Note: chunks do NOT overlap on the GPU (single ordered stream); the win is
    // purely cache locality. At large S (high tokens/expert) the GEMM dominates
    // and this path beats the baseline per-expert loop (+6% @ S=4096, +17% @ S=8192).
    let chunk: usize = std::env::var("NEMOTRON_MOE_CHUNK").ok()
        .and_then(|v| v.parse().ok()).filter(|&c| c >= 1).unwrap_or(1);
    let n_chunks = n_active.div_ceil(chunk);

    // Per-slot weight sizes (f16 elements).
    let up_slot = inter * hid;
    let dn_slot = hid * inter;

    // Defensive bounds check on caller-supplied scratch pools. The chunked
    // batched-dequant writes f16 expert weights to slot `c0..c1` of the scratch
    // at byte offset `c0*{up,dn}_slot*2`, and the per-expert GEMM reads/writes
    // `slot*{up,dn}_slot*2`, so the scratch MUST hold `n_active` full slots
    // (`n_active*{up,dn}_slot` f16 elems). A caller binding an under-sized pool
    // (e.g. sized for fewer active experts than this token batch routed to)
    // triggers an out-of-bounds WRITE, silent corruption on Metal, a
    // deterministic MMU-fault on CUDA. Fail loudly here, naming the shapes.
    if let Some(scratch) = up_scratch {
        let need = n_active * up_slot;
        let have = scratch.buffer.len() / DType::F16.size_bytes();
        if need > have {
            return Err(Error::Msg(format!(
                "moe_grouped_gemm OOB: up_scratch needs {need} f16 elems \
                 (n_active={n_active} * inter={inter} * hid={hid}) but pool holds {have}"
            )));
        }
    }
    if let Some(scratch) = dn_scratch {
        let need = n_active * dn_slot;
        let have = scratch.buffer.len() / DType::F16.size_bytes();
        if need > have {
            return Err(Error::Msg(format!(
                "moe_grouped_gemm OOB: dn_scratch needs {need} f16 elems \
                 (n_active={n_active} * hid={hid} * inter={inter}) but pool holds {have}"
            )));
        }
    }

    // NOTE: A grouped-batched cuBLAS variant (one cublasGemmGroupedBatchedEx per
    // pass instead of the per-expert gemm_tc_off loop) was measured on GB10 @
    // S=2048 and found to be both INCORRECT (output garbage) and SLOWER than the
    // per-expert loop: with ~107 active experts each group has only ~19 tokens, so
    // the grouped GEMM is a skinny m≈19 GEMM that tiles poorly (≈10 TFLOP/s vs the
    // per-expert path's 27), and the per-call cuMemAlloc/free of the device pointer
    // arrays serializes the stream. Per-expert cuBLAS (27 TFLOP/s) remains best.

    // ── UP scratch: full `n_active` slots (distinct regions per chunk so chunk
    // N+1's dequant overlaps chunk N's GEMMs, same-buffer reuse would serialize). ──
    let up_w_owned: Option<Tensor>;
    let up_w = if let Some(scratch) = up_scratch {
        Tensor { buffer: scratch.buffer.clone(), offset: 0, shape: vec![n_active * inter, hid], dtype: DType::F16 }
    } else {
        up_w_owned = Some(Tensor::empty(dev, vec![n_active * inter, hid], DType::F16)?);
        up_w_owned.as_ref().unwrap().clone()
    };

    // [mt, inter] f16 output for UP pass.
    let up_out = Tensor::empty(dev, vec![mt, inter], DType::F16)?;

    // ── UP pass: chunked dequant → GEMM ─────────────────────────────────────
    for ci in 0..n_chunks {
        let c0 = ci * chunk;
        let c1 = (c0 + chunk).min(n_active);
        let cn = c1 - c0;
        // Upload this chunk's expert ids.
        let eid_bytes: Vec<u8> = expert_ids[c0..c1].iter().flat_map(|&e| (e as u32).to_le_bytes()).collect();
        let eid_dev = dev.upload(&eid_bytes)
            .map_err(|e| Error::Msg(format!("moe_grouped_gemm: upload eid UP chunk {ci}: {e:?}")))?;
        // Batched dequant for this chunk → slots 0..cn of up_w.
        let total = (cn * up_slot) as u32;
        let scalars = [
            (cn as i32).to_le_bytes().to_vec(),
            (inter as i32).to_le_bytes().to_vec(),
            (hid as i32).to_le_bytes().to_vec(),
            (bpr_up as i32).to_le_bytes().to_vec(),
        ];
        // Dequant writes to slots c0..c1 of the full scratch (distinct per chunk).
        dev.dispatch_raw_cuda(
            BATCHED_DEQUANT_Q4_SRC, "moe_batched_dequant_q4.cu", "moe_batched_dequant_q4",
            &[
                (qs_up.buffer.as_ref(), qs_up.offset),
                (sc_up.buffer.as_ref(), sc_up.offset),
                (eid_dev.as_ref(), 0),
                (up_w.buffer.as_ref(), c0 * up_slot * 2),
            ],
            &scalars,
            [total.div_ceil(256), 1, 1], [256, 1, 1], 0, false,
        ).map_err(|e| Error::Msg(format!("moe_grouped_gemm: batched dequant UP chunk {ci}: {e:?}")))?;
        // GEMM each expert in the chunk (reads its global slot in up_w).
        for j in 0..cn {
            let slot = c0 + j;
            let rows = g_starts[slot + 1] - g_starts[slot];
            if rows == 0 { continue; }
            dev.gemm_tc_off(
                xs_f16.buffer.as_ref(), xs_f16.offset + g_starts[slot] * hid * 2,
                up_w.buffer.as_ref(), slot * up_slot * 2,
                up_out.buffer.as_ref(), g_starts[slot] * inter * 2,
                rows, inter, hid, DType::F16,
            ).map_err(|e| Error::Msg(format!("moe_grouped_gemm: UP gemm slot={slot}: {e:?}")))?;
        }
    }

    // Device relu² + scale 1/256 (no host round-trip).
    let up_relu2 = relu2_scale_f16(dev, &up_out, 1.0f32 / 256.0)?;
    drop(up_out);
    if up_scratch.is_none() { drop(up_w); let _ = up_w_owned; }

    // ── DN scratch: full `n_active` slots (distinct regions per chunk). ──────
    let dn_w_owned: Option<Tensor>;
    let dn_w = if let Some(scratch) = dn_scratch {
        Tensor { buffer: scratch.buffer.clone(), offset: 0, shape: vec![n_active * hid, inter], dtype: DType::F16 }
    } else {
        dn_w_owned = Some(Tensor::empty(dev, vec![n_active * hid, inter], DType::F16)?);
        dn_w_owned.as_ref().unwrap().clone()
    };

    let dn_out = Tensor::empty(dev, vec![mt, hid], DType::F16)?;

    // ── DN pass: chunked dequant → GEMM ─────────────────────────────────────
    for ci in 0..n_chunks {
        let c0 = ci * chunk;
        let c1 = (c0 + chunk).min(n_active);
        let cn = c1 - c0;
        let eid_bytes: Vec<u8> = expert_ids[c0..c1].iter().flat_map(|&e| (e as u32).to_le_bytes()).collect();
        let eid_dev = dev.upload(&eid_bytes)
            .map_err(|e| Error::Msg(format!("moe_grouped_gemm: upload eid DN chunk {ci}: {e:?}")))?;
        let total = (cn * dn_slot) as u32;
        let scalars = [
            (cn as i32).to_le_bytes().to_vec(),
            (hid as i32).to_le_bytes().to_vec(),
            (inter as i32).to_le_bytes().to_vec(),
            (bpr_dn as i32).to_le_bytes().to_vec(),
        ];
        dev.dispatch_raw_cuda(
            BATCHED_DEQUANT_Q4_SRC, "moe_batched_dequant_q4.cu", "moe_batched_dequant_q4",
            &[
                (qs_dn.buffer.as_ref(), qs_dn.offset),
                (sc_dn.buffer.as_ref(), sc_dn.offset),
                (eid_dev.as_ref(), 0),
                (dn_w.buffer.as_ref(), c0 * dn_slot * 2),
            ],
            &scalars,
            [total.div_ceil(256), 1, 1], [256, 1, 1], 0, false,
        ).map_err(|e| Error::Msg(format!("moe_grouped_gemm: batched dequant DN chunk {ci}: {e:?}")))?;
        for j in 0..cn {
            let slot = c0 + j;
            let rows = g_starts[slot + 1] - g_starts[slot];
            if rows == 0 { continue; }
            dev.gemm_tc_off(
                up_relu2.buffer.as_ref(), up_relu2.offset + g_starts[slot] * inter * 2,
                dn_w.buffer.as_ref(), slot * dn_slot * 2,
                dn_out.buffer.as_ref(), g_starts[slot] * hid * 2,
                rows, hid, inter, DType::F16,
            ).map_err(|e| Error::Msg(format!("moe_grouped_gemm: DN gemm slot={slot}: {e:?}")))?;
        }
    }
    if dn_scratch.is_none() { drop(dn_w); let _ = dn_w_owned; }

    Ok(dn_out)
}

// ════════════════════════════════════════════════════════════════════════════
// NVFP4 (mxf4nvf4) grouped MoE GEMM, native Blackwell block-scaled FP4 mma.
// Fragment-packed layout: weights repacked once at load (nvfp4_pack_w), acts
// quantized+packed per layer (moe_fp4_actq_pack), one fixed-launch grouped mma
// over BM=16 tile descriptors (graph-safe). Requires IRON_CUDA_ARCH=
// compute_121a (or _120a) — base-arch PTX rejects the mma.
// ════════════════════════════════════════════════════════════════════════════

const MOE_FP4_SRC: &str = r#"
#include <cuda_fp16.h>

// UE4M3 codebook (monotonic): e=0 subnormal m/8*2^-7? NO, matches quantize_nvfp4:
// e==0 → (m/8)*0.0078125 (=2^-7), else (1+m/8)*2^(e-7).
__device__ __forceinline__ float dec_ue4m3(unsigned b){
    int e=(b>>3)&0xF, m=b&7;
    return (e==0) ? (m/8.f)*0.0078125f : (1.f+m/8.f)*exp2f((float)(e-7));
}
__device__ __forceinline__ unsigned enc_ue4m3(float s){
    if(!(s>0.f)) return 0u;
    unsigned c;
    if(s>=480.f) return 127u;
    else {
        int e; float fr=frexpf(s,&e);          // s = fr*2^e, fr in [0.5,1)
        int E=e+6;                              // s = (2*fr)*2^(E-7)
        if(E<1){                                // subnormal: v=(m/8)*2^-7
            int m=(int)rintf(s*1024.f);
            c = (m>7) ? 8u : (unsigned)m;
        } else {
            int m=(int)rintf((2.f*fr-1.f)*8.f);
            if(m>7){m=0;E+=1;}
            if(E>15){E=15;m=7;}
            c=(unsigned)((E<<3)|m);
        }
    }
    // neighbor refinement: closed-form rounding can land one code high at
    // regime boundaries (subnormal->normal, exponent rollover); nearest of
    // {c-1, c} matches the exhaustive-argmin reference exactly.
    if(c>0){
        float vc=dec_ue4m3(c), vp=dec_ue4m3(c-1);
        if(fabsf(s-vp)<=fabsf(s-vc)) c=c-1;
    }
    return c;
}
__device__ __forceinline__ unsigned enc_e2m1(float x, float inv_bs){
    float v=x*inv_bs; unsigned sg=(v<0.f)?8u:0u; v=fabsf(v);
    // levels 0,.5,1,1.5,2,3,4,6 ; midpoints .25,.75,1.25,1.75,2.5,3.5,5
    unsigned m = (v>=.25f)+(v>=.75f)+(v>=1.25f)+(v>=1.75f)+(v>=2.5f)+(v>=3.5f)+(v>=5.f);
    return sg|m;
}
__device__ __forceinline__ float dec_e2m1(unsigned code){
    const float L[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    float mag=L[code&7u]; return (code&8u)?-mag:mag;
}
__device__ __forceinline__ int b4b_map(int blk){ const int t[4]={0,2,1,3}; return t[blk]; }

// amax over slabs: x f16 [n_slabs * slab_len]; atomicMax on positive float bits.
__device__ __forceinline__ float fp4_act(float v, int mode, float fs){
    if(mode==1){ float r=v>0.f?v:0.f; return r*r*fs; }
    return v;
}
extern "C" __global__ void nvfp4_amax_slab(
    const __half* __restrict__ x, unsigned* __restrict__ amax_bits,
    long slab_len, int n_slabs, int act_mode, float act_scale)
{
    long total=slab_len*(long)n_slabs;
    long stride=(long)gridDim.x*blockDim.x;
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(n_slabs==1){
        // hot path (per-call activation global): block-reduce, 1 atomic/block.
        float m=0.f;
        for(long p=i;p<total;p+=stride) m=fmaxf(m,fabsf(fp4_act(__half2float(x[p]),act_mode,act_scale)));
        #pragma unroll
        for(int o=16;o;o>>=1) m=fmaxf(m,__shfl_down_sync(0xffffffffu,m,o));
        __shared__ float wm[32];
        if((threadIdx.x&31)==0) wm[threadIdx.x>>5]=m;
        __syncthreads();
        if(threadIdx.x<32){
            float v=(threadIdx.x<(blockDim.x>>5))?wm[threadIdx.x]:0.f;
            #pragma unroll
            for(int o=16;o;o>>=1) v=fmaxf(v,__shfl_down_sync(0xffffffffu,v,o));
            if(threadIdx.x==0&&v>0.f) atomicMax(&amax_bits[0],__float_as_uint(v));
        }
        return;
    }
    // setup-only path (per-expert weight slabs): local-run accumulation.
    float m=0.f; int cur=-1;
    for(long p=i;p<total;p+=stride){
        int slab=(int)(p/slab_len);
        float v=fabsf(__half2float(x[p]));
        if(slab!=cur){
            if(cur>=0&&m>0.f) atomicMax(&amax_bits[cur],__float_as_uint(m));
            cur=slab; m=v;
        } else m=fmaxf(m,v);
    }
    if(cur>=0&&m>0.f) atomicMax(&amax_bits[cur],__float_as_uint(m));
}

// finalize: g[i] = amax/6/448 (per-tensor NVFP4 global), floor tiny.
extern "C" __global__ void nvfp4_globals(
    const unsigned* __restrict__ amax_bits, float* __restrict__ g, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float a=__uint_as_float(amax_bits[i]);
    float v=a/6.f/448.f; g[i]=(v>0.f)?v:1e-9f;
}

// Weight quant+pack: w f16 [n_exp*N, K] row-major → fragment layout
// Wp [e][NT][KT][32][2] u32, Wsc [e][NT][KT][8] u32. One warp = (e,nt,kt).
extern "C" __global__ void nvfp4_pack_w(
    const __half* __restrict__ w, const float* __restrict__ gw,
    unsigned* __restrict__ Wp, unsigned* __restrict__ Wsc,
    int n_exp, int N, int K)
{
    int KT=K/64, NT=N/8;
    long warp=((long)blockIdx.x*blockDim.x+threadIdx.x)>>5;
    long total=(long)n_exp*NT*KT; if(warp>=total) return;
    int lane=threadIdx.x&31, gid=lane>>2, tig=lane&3;
    int kt=(int)(warp%KT); long t2=warp/KT; int nt=(int)(t2%NT); int e=(int)(t2/NT);
    float g=gw[e]; float inv_g=1.f/g;
    int row=nt*8+gid; const __half* wr=w+((long)e*N+row)*K;
    unsigned b0=0,b1=0;
    // lane covers (blk in {tig>=2?2,3:0,1}, w in {tig&1?8..15:0..7}) of the 64-K tile
    int blk0=(tig>=2)?2:0, w0=(tig&1)?8:0;
    for(int bb=0;bb<2;bb++){
        int blk=blk0+bb; int bstart=kt*64+blk*16;
        float amax=0.f;
        #pragma unroll
        for(int j=0;j<16;j++) amax=fmaxf(amax,fabsf(__half2float(wr[bstart+j])));
        unsigned sb=enc_ue4m3((amax/6.f)*inv_g);
        float bs=dec_ue4m3(sb)*g; float inv_bs=(bs>0.f)?1.f/bs:0.f;
        unsigned reg=0;
        #pragma unroll
        for(int j=0;j<8;j++){
            int kk=blk*16+w0+j; int gK=kt*64+kk; (void)kk;
            unsigned code=enc_e2m1(__half2float(wr[gK]),inv_bs);
            reg|=code<<(j*4);
        }
        if(blk&1) b1=reg; else b0=reg;
    }
    long fbase=((long)((long)e*NT+nt)*KT+kt);
    Wp[(fbase*32+lane)*2+0]=b0; Wp[(fbase*32+lane)*2+1]=b1;
    if(tig==0){
        unsigned s=0;
        #pragma unroll
        for(int blk=0;blk<4;blk++){
            int bstart=kt*64+blk*16; float amax=0.f;
            for(int j=0;j<16;j++) amax=fmaxf(amax,fabsf(__half2float(wr[bstart+j])));
            s|=enc_ue4m3((amax/6.f)*inv_g)<<(b4b_map(blk)*8);
        }
        Wsc[fbase*8+gid]=s;
    }
}

// Activation quant+pack: a f16 [mt, K] (sorted rows) + tile descs → fragment
// layout Ap [tile][KT][32][4] u32 + Asc [tile][KT][32] u32. Rows past gend → 0.
// One warp = (tile, kt). gA read from device (dynamic per-call global).
extern "C" __global__ void moe_fp4_actq_pack(
    const __half* __restrict__ a, const int* __restrict__ tok0,
    const int* __restrict__ gend, const float* __restrict__ gA,
    unsigned* __restrict__ Ap, unsigned* __restrict__ Asc,
    const unsigned* __restrict__ st,
    int maxt, int K, int butter_total, int act_mode, float act_scale, int use_st)
{
    // Warp-per-(tile,kt): stage the 16x64 f16 tile to smem with coalesced
    // half2 loads (each lane: 2 rows x 16B), compute the 64 block-16 amaxes
    // cooperatively (one per (row,blk) pair on lanes 0..31 x2), then encode.
    int KT=K/64;
    int wid=threadIdx.x>>5;
    long warp=((long)blockIdx.x*(blockDim.x>>5))+wid;
    if(warp>=(long)maxt*KT) return;
    int lane=threadIdx.x&31, gid=lane>>2, tig=lane&3;
    int kt=(int)(warp%KT); int t=(int)(warp/KT);
    int base=tok0[t], end=gend[t];
    float g=gA[0]; float inv_g=1.f/g;
    __shared__ __half smem_all[8][16*64];    // 8 warps/block x 2KB
    __shared__ float bsmem_all[8][16*4];     // reconstructed block scales
    __half* sm=smem_all[wid];
    float* bsm=bsmem_all[wid];
    // stage: lane loads rows (lane>>1) halves: 32 halves = 4x half4(8B). row-major.
    {
        int r0=lane>>1, c0=(lane&1)*32;      // each lane: row r0, cols c0..c0+31
        int row=base+r0;
        bool live=(r0 < end-base)&&(row<butter_total);
        long src_row=(use_st&&live)?(long)st[row]:(long)row;  // gather-on-read
        const __half* src=a+src_row*K+kt*64+c0;
        #pragma unroll
        for(int j=0;j<32;j+=8){
            if(live){
                uint4 v=*(const uint4*)&src[j];
                if(act_mode){
                    const __half* hv=(const __half*)&v;
                    #pragma unroll
                    for(int q=0;q<8;q++)
                        sm[r0*64+c0+j+q]=__float2half(fp4_act(__half2float(hv[q]),act_mode,act_scale));
                } else *(uint4*)&sm[r0*64+c0+j]=v;
            } else {
                uint4 z={0,0,0,0}; *(uint4*)&sm[r0*64+c0+j]=z;
            }
        }
    }
    __syncwarp();
    // block scales: 64 (row,blk) pairs; lanes do 2 each.
    #pragma unroll
    for(int pp=0;pp<2;pp++){
        int pair=lane+pp*32;                 // 0..63
        int r=pair>>2, blk=pair&3;
        const __half* bp=&sm[r*64+blk*16];
        float amax=0.f;
        #pragma unroll
        for(int j=0;j<16;j++) amax=fmaxf(amax,fabsf(__half2float(bp[j])));
        unsigned sb=enc_ue4m3((amax/6.f)*inv_g);
        bsm[r*4+blk]=dec_ue4m3(sb)*g;
    }
    __syncwarp();
    unsigned regs[4]={0,0,0,0};
    int blk0=(tig>=2)?2:0, w0=(tig&1)?8:0;
    #pragma unroll
    for(int half=0;half<2;half++){
        int r=gid+half*8;
        #pragma unroll
        for(int bb=0;bb<2;bb++){
            int blk=blk0+bb;
            float bs=bsm[r*4+blk]; float inv_bs=(bs>0.f)?1.f/bs:0.f;
            unsigned reg=0;
            #pragma unroll
            for(int j=0;j<8;j++)
                reg|=enc_e2m1(__half2float(sm[r*64+blk*16+w0+j]),inv_bs)<<(j*4);
            int ri=(blk&1)?(half?3:2):(half?1:0);
            regs[ri]|=reg;
        }
    }
    long fbase=(long)t*KT+kt;
    #pragma unroll
    for(int r=0;r<4;r++) Ap[(fbase*32+lane)*4+r]=regs[r];
    // scale word (re-encode from reconstructed bs: enc(dec(x))==x for codebook values)
    unsigned sw=0;
    if(tig==0||tig==1){
        int r=gid+(tig==1?8:0);
        #pragma unroll
        for(int blk=0;blk<4;blk++)
            sw|=enc_ue4m3(bsm[r*4+blk]*inv_g)<<(b4b_map(blk)*8);
    }
    Asc[fbase*32+lane]=sw;
}

// VARIANT: each warp computes 4 consecutive n8-tiles, A frags loaded once,
// reused 4x; W streams per-subtile. Cuts A traffic 4x, raises mma density.
// 16-byte async global->shared copy (per-warp staging; no block syncs).
__device__ __forceinline__ void butter_cp16(void* sm, const void* gm){
  unsigned saddr=(unsigned)__cvta_generic_to_shared(sm);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(saddr),"l"(gm));
}
extern "C" __global__ void moe_fp4_grouped_mma_x4(
    const unsigned* __restrict__ Ap, const unsigned* __restrict__ Asc,
    const unsigned* __restrict__ Wp, const unsigned* __restrict__ Wsc,
    const int* __restrict__ tok0, const int* __restrict__ eidv,
    const int* __restrict__ gend, const float* __restrict__ gA,
    const float* __restrict__ gw, __half* __restrict__ out,
    unsigned* amax_out,
    int N, int K, int butter_total, int am_mode, float am_scale)
{
    // Per-warp cp.async 2-stage smem W pipeline + A register prefetch: the
    // direct global->register W loads left the mma latency-bound (~42 TF);
    // double-buffering the 4 W tiles per warp while the mma consumes the
    // other stage measures 47.9 TF standalone on GB10 (+14%). Each warp owns
    // its smem slot, per-warp wait_group + syncwarp only, NO __syncthreads
    // (the cross-warp shared-W variant with block syncs measured SLOWER).
    __shared__ unsigned smW[8][2][256]; // [warp][stage][4 tiles x 64 u32]
    __shared__ unsigned smS[8][2][32];  // [warp][stage][4 tiles x 8 u32]
    int KT=K/64, NT=N/8, NG=(NT+3)>>2;
    int t=blockIdx.x;
    int w=threadIdx.x>>5;
    int ng=blockIdx.y*(blockDim.x>>5)+w;
    if(ng>=NG) return;
    int base=tok0[t], end=gend[t];
    if(end<=base) return;
    int lane=threadIdx.x&31, gid=lane>>2, tig=lane&3;
    unsigned eid=(unsigned)eidv[t];
    const unsigned* A =Ap +((long)t*KT*32)*4+lane*4;
    const unsigned* As=Asc+ (long)t*KT*32+lane;
    float c[4][4]={};
    int nt0=ng*4;
    #define X4_ISSUE(kt_,st_) do{                                            \
        int c0=lane, c1=lane+32;                                              \
        int j0=c0>>4, j1=c1>>4;                                               \
        int nt_a=nt0+j0, nt_b=nt0+j1;                                         \
        if(nt_a<NT) butter_cp16(&smW[w][st_][j0*64+(c0&15)*4],                    \
            Wp+(((long)eid*NT+nt_a)*KT+(kt_))*64+(c0&15)*4);                  \
        if(nt_b<NT) butter_cp16(&smW[w][st_][j1*64+(c1&15)*4],                    \
            Wp+(((long)eid*NT+nt_b)*KT+(kt_))*64+(c1&15)*4);                  \
        if(lane<8){ int js=lane>>1, hf=lane&1; int nt_s=nt0+js;               \
            if(nt_s<NT) butter_cp16(&smS[w][st_][js*8+hf*4],                      \
                Wsc+(((long)eid*NT+nt_s)*KT+(kt_))*8+hf*4); }                 \
        asm volatile("cp.async.commit_group;");                               \
    }while(0)
    X4_ISSUE(0,0);
    unsigned a0=A[0],a1=A[1],a2=A[2],a3=A[3],sA=As[0];
    for(int kt=0;kt<KT;kt++){
        int st=kt&1;
        unsigned na0=0,na1=0,na2=0,na3=0,nsA=0;
        if(kt+1<KT){
            X4_ISSUE(kt+1,st^1);
            const unsigned* An=A+(long)(kt+1)*32*4;
            na0=An[0];na1=An[1];na2=An[2];na3=An[3];nsA=As[(long)(kt+1)*32];
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncwarp();
        #pragma unroll
        for(int j=0;j<4;j++){
            int nt=nt0+j; if(nt>=NT) break;
            unsigned b0=smW[w][st][j*64+lane*2], b1=smW[w][st][j*64+lane*2+1];
            unsigned sB=smS[w][st][j*8+gid];
            asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
             "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
             :"+f"(c[j][0]),"+f"(c[j][1]),"+f"(c[j][2]),"+f"(c[j][3])
             :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sA),"r"(sB));
        }
        a0=na0;a1=na1;a2=na2;a3=na3;sA=nsA;
        __syncwarp();
    }
    #undef X4_ISSUE
    float g=gA[0]*gw[eid];
    int valid=end-base;
    float am=0.f;
    #pragma unroll
    for(int j=0;j<4;j++){
        int nt=nt0+j; if(nt>=NT) break;
        int col=nt*8+tig*2;
        if(gid<valid){
            long r=(long)(base+gid)*N+col;
            float v0=c[j][0]*g, v1=c[j][1]*g;
            out[r]=__float2half(v0); out[r+1]=__float2half(v1);
            if(am_mode>=0){am=fmaxf(am,fabsf(fp4_act(v0,am_mode,am_scale)));am=fmaxf(am,fabsf(fp4_act(v1,am_mode,am_scale)));}
        }
        if(gid+8<valid){
            long r=(long)(base+gid+8)*N+col;
            float v2=c[j][2]*g, v3=c[j][3]*g;
            out[r]=__float2half(v2); out[r+1]=__float2half(v3);
            if(am_mode>=0){am=fmaxf(am,fabsf(fp4_act(v2,am_mode,am_scale)));am=fmaxf(am,fabsf(fp4_act(v3,am_mode,am_scale)));}
        }
    }
    if(am_mode>=0){
        // fused next-GEMM activation amax: warp-reduce then 1 atomic/warp.
        #pragma unroll
        for(int o=16;o;o>>=1) am=fmaxf(am,__shfl_down_sync(0xffffffffu,am,o));
        if(lane==0&&am>0.f) atomicMax(amax_out,__float_as_uint(am));
    }
}

// Grouped mma: one warp = (tile, nt). FIXED launch grid, padding tiles
// (gend==0) early-out → graph-safe. out f16 [mt, N] in sorted-row space.
extern "C" __global__ void moe_fp4_grouped_mma(
    const unsigned* __restrict__ Ap, const unsigned* __restrict__ Asc,
    const unsigned* __restrict__ Wp, const unsigned* __restrict__ Wsc,
    const int* __restrict__ tok0, const int* __restrict__ eidv,
    const int* __restrict__ gend, const float* __restrict__ gA,
    const float* __restrict__ gw, __half* __restrict__ out,
    int N, int K, int butter_total)
{
    int KT=K/64, NT=N/8;
    int t=blockIdx.x;                       // tile index
    int nt=blockIdx.y*(blockDim.x>>5)+(threadIdx.x>>5); // n8-tile
    if(nt>=NT) return;
    int base=tok0[t], end=gend[t];
    if(end<=base) return;                   // padding tile
    int lane=threadIdx.x&31, gid=lane>>2, tig=lane&3;
    unsigned eid=(unsigned)eidv[t];
    const unsigned* A =Ap +((long)t*KT*32)*4+lane*4;
    const unsigned* As=Asc+ (long)t*KT*32+lane;
    const unsigned* W =Wp +(((long)eid*NT+nt)*KT*32)*2+lane*2;
    const unsigned* Ws=Wsc+(((long)eid*NT+nt)*KT)*8+gid;
    float c0=0,c1=0,c2=0,c3=0;
    for(int kt=0;kt<KT;kt++){
        const unsigned* Ak=A+(long)kt*32*4;
        const unsigned* Wk=W+(long)kt*32*2;
        unsigned a0=Ak[0],a1=Ak[1],a2=Ak[2],a3=Ak[3],b0=Wk[0],b1=Wk[1];
        unsigned sA=As[(long)kt*32], sB=Ws[(long)kt*8];
        asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
         :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
         :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sA),"r"(sB));
    }
    float g=gA[0]*gw[eid];
    int valid=end-base; int col=nt*8+tig*2;
    if(gid<valid){
        long r=(long)(base+gid)*N+col;
        out[r]=__float2half(c0*g); out[r+1]=__float2half(c1*g);
    }
    if(gid+8<valid){
        long r=(long)(base+gid+8)*N+col;
        out[r]=__float2half(c2*g); out[r+1]=__float2half(c3*g);
    }
}

// Extend the routed MoE group descriptors with TWO identity groups covering
// all s tokens (the shared expert folded into the grouped GEMM as two
// stacked always-selected experts: up [2*inter,hid] slabs / down K-split).
// out st/sw get [mt..mt+2s) = identity tokens with weight 1.0; offsets gain
// off[n_exp+1] = mt+s, off[n_exp+2] = mt+2s. One launch, graph-safe.
extern "C" __global__ void moe_extend_groups(
    const unsigned* __restrict__ st_in, const float* __restrict__ sw_in,
    const unsigned* __restrict__ off_in,
    unsigned* __restrict__ st_out, float* __restrict__ sw_out,
    unsigned* __restrict__ off_out,
    int mt, int s, int n_exp)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int total=mt+2*s;
    if(i<total){
        if(i<mt){ st_out[i]=st_in[i]; sw_out[i]=sw_in[i]; }
        else { st_out[i]=(unsigned)((i-mt)%s); sw_out[i]=1.0f; }
    }
    if(i<=n_exp){ off_out[i]=off_in[i]; }
    else if(i==n_exp+1){ off_out[i]=(unsigned)(mt+s); }
    else if(i==n_exp+2){ off_out[i]=(unsigned)(mt+2*s); }
}

// FP8 e4m3 per-tensor quant. Pass 1 = nvfp4_amax_slab (reused, mode 0).
// fp8_scale: amax bits -> scale = amax/448 and inv (one thread).
extern "C" __global__ void fp8_scale(
    const unsigned* __restrict__ amax_bits, float* __restrict__ sc, float clip)
{
    float am=__uint_as_float(*amax_bits)*((clip>0.f)?clip:1.f);
    float s=(am>0.f)?(am/448.f):1.f;
    sc[0]=s;        // dequant scale (consumed by the GEMM)
    sc[1]=1.f/s;    // quant scale (consumed by fp8_quant)
}
// fp8_quant: x f16 [n] -> e4m3 bytes, v*inv clamped to ±448, round-nearest.
// Per-row e4m3 quant: each of R rows -> e4m3 + per-row dequant scale[R]
// (amax/448). Per-channel (weight rows=N) or per-token (act rows=M).
extern "C" __global__ void fp8_quant_perrow(
    const __half* __restrict__ x, unsigned char* __restrict__ q,
    float* __restrict__ scale, int R, int K)
{
    int r=blockIdx.x; if(r>=R) return;
    const __half* xr=x+(long)r*K;
    __shared__ float sm[256];
    float am=0.f;
    for(int i=threadIdx.x;i<K;i+=blockDim.x) am=fmaxf(am,fabsf(__half2float(xr[i])));
    sm[threadIdx.x]=am; __syncthreads();
    for(int o=blockDim.x>>1;o>0;o>>=1){ if(threadIdx.x<o) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+o]); __syncthreads(); }
    float sc=(sm[0]>0.f)?sm[0]/448.f:1.f; float inv=1.f/sc;
    if(threadIdx.x==0) scale[r]=sc;
    unsigned char* qr=q+(long)r*K;
    for(int i=threadIdx.x;i<K;i+=blockDim.x){
        float v=__half2float(xr[i])*inv; v=fminf(fmaxf(v,-448.f),448.f);
        unsigned s2=(v<0.f)?0x80u:0u; float a=fabsf(v); unsigned c;
        if(!(a>0.f)) c=0u; else if(a>=448.f) c=0x7Eu;
        else { int e; float fr=frexpf(a,&e); int E=e+6;
            if(E<1){ int m=(int)rintf(a*512.f); if(m>7)m=7; c=(unsigned)m; }
            else { int m=(int)rintf((2.f*fr-1.f)*8.f); if(m>7){m=0;E+=1;} if(E>15){E=15;m=6;} c=(unsigned)((E<<3)|m); } }
        qr[i]=(unsigned char)(s2|c);
    }
}
extern "C" __global__ void fp8_quant_perrow_f32(
    const float* __restrict__ x, unsigned char* __restrict__ q,
    float* __restrict__ scale, int R, int K)
{
    int r=blockIdx.x; if(r>=R) return;
    const float* xr=x+(long)r*K;
    __shared__ float sm[256];
    float am=0.f;
    for(int i=threadIdx.x;i<K;i+=blockDim.x) am=fmaxf(am,fabsf(xr[i]));
    sm[threadIdx.x]=am; __syncthreads();
    for(int o=blockDim.x>>1;o>0;o>>=1){ if(threadIdx.x<o) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+o]); __syncthreads(); }
    float sc=(sm[0]>0.f)?sm[0]/448.f:1.f; float inv=1.f/sc;
    if(threadIdx.x==0) scale[r]=sc;
    unsigned char* qr=q+(long)r*K;
    for(int i=threadIdx.x;i<K;i+=blockDim.x){
        float v=xr[i]*inv; v=fminf(fmaxf(v,-448.f),448.f);
        unsigned s2=(v<0.f)?0x80u:0u; float a=fabsf(v); unsigned c;
        if(!(a>0.f)) c=0u; else if(a>=448.f) c=0x7Eu;
        else { int e; float fr=frexpf(a,&e); int E=e+6;
            if(E<1){ int m=(int)rintf(a*512.f); if(m>7)m=7; c=(unsigned)m; }
            else { int m=(int)rintf((2.f*fr-1.f)*8.f); if(m>7){m=0;E+=1;} if(E>15){E=15;m=6;} c=(unsigned)((E<<3)|m); } }
        qr[i]=(unsigned char)(s2|c);
    }
}
// post-GEMM rowcol dequant: D[r,c] *= xsc[r] * wsc[c]  (D [M,N] row-major).
extern "C" __global__ void fp8_rowcol_scale_f32(
    float* __restrict__ o, const float* __restrict__ xsc, const float* __restrict__ wsc, int M, int N)
{
    long e=(long)blockIdx.x*blockDim.x+threadIdx.x; long n=(long)M*N;
    long st=(long)gridDim.x*blockDim.x;
    for(;e<n;e+=st){ int r=(int)(e/N), c=(int)(e%N); o[e]*=xsc[r]*wsc[c]; }
}

// Fused block-Hadamard(128) along K + global amax of |rotated| (atomicMax on
// the float bit-pattern, monotonic for positives). Collapses the separate
// fwht + amax passes for the per-tensor fp8 proj path. K % 128 == 0.
extern "C" __global__ void fwht128_amax(
    const __half* __restrict__ x, __half* __restrict__ y,
    unsigned* __restrict__ amax, long rows, int K)
{
    int nb = K / 128;
    long bid = (long)blockIdx.x;
    if (bid >= rows * (long)nb) return;
    int t = threadIdx.x;
    __shared__ float s[128];
    long base = bid * 128;
    s[t] = __half2float(x[base + t]);
    __syncthreads();
    for (int len = 1; len < 128; len <<= 1) {
        float a = s[t]; float b = s[t ^ len];
        __syncthreads();
        s[t] = (t & len) ? (b - a) : (a + b);
        __syncthreads();
    }
    float r = s[t] * 0.08838834764831845f;
    y[base + t] = __float2half(r);
    // block-reduce |r| then one atomicMax to the global amax
    s[t] = fabsf(r);
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) { if (t < o) s[t] = fmaxf(s[t], s[t + o]); __syncthreads(); }
    if (t == 0) atomicMax(amax, __float_as_uint(s[0]));
}

// Block Walsh-Hadamard transform (size 128) along the K dim, normalized by
// 1/sqrt(128). Each CUDA block handles one (row, kblock) of 128 contiguous
// elems. Orthonormal: rotating both activation and weight along K leaves the
// GEMM x.Wt unchanged, but spreads per-channel activation outliers so a
// per-tensor fp8 quant holds quality (QuaRot/Hadamard recipe). K % 128 == 0.
extern "C" __global__ void fwht128(
    const __half* __restrict__ x, __half* __restrict__ y, long rows, int K)
{
    int nb = K / 128;
    long bid = (long)blockIdx.x;
    if (bid >= rows * (long)nb) return;
    int t = threadIdx.x; // 0..127
    __shared__ float s[128];
    long base = bid * 128;
    s[t] = __half2float(x[base + t]);
    __syncthreads();
    for (int len = 1; len < 128; len <<= 1) {
        float a = s[t]; float b = s[t ^ len];
        __syncthreads();
        s[t] = (t & len) ? (b - a) : (a + b);
        __syncthreads();
    }
    y[base + t] = __float2half(s[t] * 0.08838834764831845f); // 1/sqrt(128)
}

extern "C" __global__ void fp8_quant(
    const __half* __restrict__ x, unsigned char* __restrict__ q,
    const float* __restrict__ sc, long n)
{
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    long stride=(long)gridDim.x*blockDim.x;
    float inv=sc[1];
    for(;i<n;i+=stride){
        float v=__half2float(x[i])*inv;
        v=fminf(fmaxf(v,-448.f),448.f);
        // e4m3: sign + 4-bit exp (bias 7) + 3-bit mantissa, no inf, nan=0x7f
        unsigned s=(v<0.f)?0x80u:0u; float a=fabsf(v);
        unsigned c;
        if(a<0.0009765625f){ // subnormal threshold 2^-10... min normal 2^-6
            int m=(int)rintf(a*512.f); // subnormals: m/8 * 2^-6 -> a*2^9... (m = a/2^-9)
            if(m>7)m=7; c=(unsigned)m;
        } else {
            int e; float fr=frexpf(a,&e); // a = fr*2^e, fr in [0.5,1)
            int E=e+6;                    // a = (2fr)*2^(E-7)
            int m=(int)rintf((2.f*fr-1.f)*8.f);
            if(m>7){m=0;E+=1;}
            if(E<1){ int ms=(int)rintf(a*512.f); if(ms>7)ms=7; c=(unsigned)ms; }
            else { if(E>15)E=15; if(E==15&&m>6)m=6; c=(unsigned)((E<<3)|m); }
        }
        q[i]=(unsigned char)(s|c);
    }
}

// Pass-1 of the fused fp8 act path: block-Hadamard(128) along K + global amax,
// WITHOUT materializing the rotated tensor (no y write). Halves the HBM of the
// old fwht128_amax (which wrote a full [rows,K] f16 that fp8_quant then re-read).
extern "C" __global__ void fwht128_amaxonly(
    const __half* __restrict__ x, unsigned* __restrict__ amax, long rows, int K)
{
    int nb = K / 128; long bid = (long)blockIdx.x;
    if (bid >= rows * (long)nb) return;
    int t = threadIdx.x; __shared__ float s[128]; long base = bid * 128;
    s[t] = __half2float(x[base + t]); __syncthreads();
    for (int len = 1; len < 128; len <<= 1) {
        float a = s[t]; float b = s[t ^ len]; __syncthreads();
        s[t] = (t & len) ? (b - a) : (a + b); __syncthreads();
    }
    float r = s[t] * 0.08838834764831845f;
    s[t] = fabsf(r); __syncthreads();
    for (int o = 64; o > 0; o >>= 1) { if (t < o) s[t] = fmaxf(s[t], s[t + o]); __syncthreads(); }
    if (t == 0) atomicMax(amax, __float_as_uint(s[0]));
}

// Pass-2 of the fused fp8 act path: re-rotate x (Hadamard 128) AND quant to e4m3
// in one kernel reading x directly (no rotated-tensor round-trip). Uses the
// precomputed scale sc[1]. Same e4m3 encoding as fp8_quant.
extern "C" __global__ void fwht128_quant(
    const __half* __restrict__ x, unsigned char* __restrict__ q,
    const float* __restrict__ sc, long rows, int K)
{
    int nb = K / 128; long bid = (long)blockIdx.x;
    if (bid >= rows * (long)nb) return;
    int t = threadIdx.x; __shared__ float s[128]; long base = bid * 128;
    s[t] = __half2float(x[base + t]); __syncthreads();
    for (int len = 1; len < 128; len <<= 1) {
        float a = s[t]; float b = s[t ^ len]; __syncthreads();
        s[t] = (t & len) ? (b - a) : (a + b); __syncthreads();
    }
    float v = s[t] * 0.08838834764831845f * sc[1];
    v = fminf(fmaxf(v, -448.f), 448.f);
    unsigned sgn=(v<0.f)?0x80u:0u; float a=fabsf(v); unsigned c;
    if(a<0.0009765625f){ int m=(int)rintf(a*512.f); if(m>7)m=7; c=(unsigned)m; }
    else { int e; float fr=frexpf(a,&e); int E=e+6; int m=(int)rintf((2.f*fr-1.f)*8.f);
        if(m>7){m=0;E+=1;} if(E<1){ int ms=(int)rintf(a*512.f); if(ms>7)ms=7; c=(unsigned)ms; }
        else { if(E>15)E=15; if(E==15&&m>6)m=6; c=(unsigned)((E<<3)|m); } }
    q[base + t]=(unsigned char)(sgn|c);
}

// f32-input variant of lt_fp4_quant, reads the residual-stream f32 tensor
// directly, killing the cast_f32_f16 + f16 intermediate round-trip that
// preceded every dense-projection quant (f32 -> f16 -> e2m1 became
// f32 -> e2m1; the e2m1 grid subsumes the f16 rounding).
extern "C" __global__ void lt_fp4_quant_f32(
    const float* __restrict__ x, unsigned char* __restrict__ xp,
    unsigned char* __restrict__ sf, int R, int K)
{
    int KB=K/16;
    long nb=(long)R*KB;
    long b=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=nb) return;
    int r=(int)(b/KB), kb=(int)(b%KB);
    const float* xb=x+(long)r*K+kb*16;
    float v[16]; float am=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ v[i]=xb[i]; am=fmaxf(am,fabsf(v[i])); }
    unsigned sc=enc_ue4m3(am/6.f);
    if(sc==0u&&am>0.f) sc=1u;
    float sd=dec_ue4m3(sc);
    float inv=(sd>0.f)?(1.f/sd):0.f;
    int KB4=(KB+3)>>2;
    long soff=(long)(r>>7)*(512L*KB4)+(long)(kb>>2)*512+(r&31)*16+((r>>5)&3)*4+(kb&3);
    sf[soff]=(unsigned char)sc;
    unsigned char* o=xp+((long)r*K+kb*16)/2;
    #pragma unroll
    for(int i=0;i<8;i++){
        unsigned lo=enc_e2m1(v[2*i],inv), hi=enc_e2m1(v[2*i+1],inv);
        o[i]=(unsigned char)(lo|(hi<<4));
    }
}

// Quantize f16 [R,K] row-major to the cuBLASLt block-scaled NVFP4 format:
// packed e2m1 (2 elems/byte, linear row-major) + one ue4m3 scale per
// 16-element K-block stored in the hardware's swizzled 512-byte-block layout.
// One thread per 16-element block; per-block self-scaling (amax/6 -> ue4m3),
// no global amax pass.
extern "C" __global__ void lt_fp4_quant(
    const __half* __restrict__ x, unsigned char* __restrict__ xp,
    unsigned char* __restrict__ sf, int R, int K)
{
    int KB=K/16;
    long nb=(long)R*KB;
    long b=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=nb) return;
    int r=(int)(b/KB), kb=(int)(b%KB);
    const __half* xb=x+(long)r*K+kb*16;
    float v[16]; float am=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ v[i]=__half2float(xb[i]); am=fmaxf(am,fabsf(v[i])); }
    unsigned sc=enc_ue4m3(am/6.f);
    if(sc==0u&&am>0.f) sc=1u;
    float sd=dec_ue4m3(sc);
    float inv=(sd>0.f)?(1.f/sd):0.f;
    int KB4=(KB+3)>>2;
    long soff=(long)(r>>7)*(512L*KB4)+(long)(kb>>2)*512+(r&31)*16+((r>>5)&3)*4+(kb&3);
    sf[soff]=(unsigned char)sc;
    unsigned char* o=xp+((long)r*K+kb*16)/2;
    #pragma unroll
    for(int i=0;i<8;i++){
        unsigned lo=enc_e2m1(v[2*i],inv), hi=enc_e2m1(v[2*i+1],inv);
        o[i]=(unsigned char)(lo|(hi<<4));
    }
}

// nvfp4_gscale: amax bits -> [global, inv_global] (one thread). global =
// amax/(6*448) (same convention as nvfp4_globals); 1.0 when amax == 0.
// Word 0 feeds the GEMM's per-group alpha; word 1 feeds the quant kernels.
extern "C" __global__ void nvfp4_gscale(
    const unsigned* __restrict__ amax_bits, float* __restrict__ g)
{
    float a=__uint_as_float(*amax_bits);
    float v=a/6.f/448.f;
    v=(v>0.f)?v:1.f;
    g[0]=v; g[1]=1.f/v;
}

// fp4_sfa_offsets: device build of the per-group SFA byte offsets from the
// router offsets, sfa_off[g] = (sum_{j<g} ceil(M_j/128)) * 512 * ceil(K/64).
// Single block, serial prefix (n_groups <= a few hundred).
extern "C" __global__ void fp4_sfa_offsets(
    const unsigned* __restrict__ off, long long* __restrict__ sfa_off,
    int n_groups, int kb4)
{
    if(threadIdx.x!=0||blockIdx.x!=0) return;
    long long acc=0;
    for(int g=0; g<n_groups; ++g){
        sfa_off[g]=acc*512LL*kb4;
        unsigned m=off[g+1]-off[g];
        acc+=(long long)((m+127)/128);
    }
}

// fp4_group_alpha_id: identity-expert variant, alpha[g] = gA[0] * gw[g].
extern "C" __global__ void fp4_group_alpha_id(
    const float* __restrict__ gA, const float* __restrict__ gw,
    float* __restrict__ alpha, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) alpha[i]=gA[0]*gw[i];
}

// fp4_group_alpha: per-group GEMM output scale alpha[g] = gA[0] * gw[eid[g]]
// (activation per-call global x expert weight global), folds both per-tensor
// globals back into the grouped GEMM epilogue, all device-side.
extern "C" __global__ void fp4_group_alpha(
    const float* __restrict__ gA, const float* __restrict__ gw,
    const int* __restrict__ eids, float* __restrict__ alpha, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) alpha[i]=gA[0]*gw[eids[i]];
}

// lt_fp4_quant + per-tensor global: identical layout to lt_fp4_quant but the
// block scale is encoded RELATIVE to gs[0] (sc = enc_ue4m3((amax/6)/global)),
// keeping the ue4m3 codes centered in the normal range instead of the
// subnormal floor (the no-global recipe loses scale precision when block
// amaxes are small, e.g. Q4-dequanted weights). Effective block scale =
// dec_ue4m3(sc) * global, the GEMM multiplies global back in via alpha.
extern "C" __global__ void lt_fp4_quant_gs(
    const __half* __restrict__ x, unsigned char* __restrict__ xp,
    unsigned char* __restrict__ sf, const float* __restrict__ gs,
    int R, int K)
{
    int KB=K/16;
    long nb=(long)R*KB;
    long b=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=nb) return;
    int r=(int)(b/KB), kb=(int)(b%KB);
    const __half* xb=x+(long)r*K+kb*16;
    float v[16]; float am=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ v[i]=__half2float(xb[i]); am=fmaxf(am,fabsf(v[i])); }
    float ig=1.f/gs[0];
    unsigned sc=enc_ue4m3((am/6.f)*ig);
    if(sc==0u&&am>0.f) sc=1u;
    float sd=dec_ue4m3(sc);
    float inv=(sd>0.f)?(ig/sd):0.f;
    int KB4=(KB+3)>>2;
    long soff=(long)(r>>7)*(512L*KB4)+(long)(kb>>2)*512+(r&31)*16+((r>>5)&3)*4+(kb&3);
    sf[soff]=(unsigned char)sc;
    unsigned char* o=xp+((long)r*K+kb*16)/2;
    #pragma unroll
    for(int i=0;i<8;i++){
        unsigned lo=enc_e2m1(v[2*i],inv), hi=enc_e2m1(v[2*i+1],inv);
        o[i]=(unsigned char)(lo|(hi<<4));
    }
}

extern "C" __global__ void lt_fp4_quant_gs_46(
    const __half* __restrict__ x, unsigned char* __restrict__ xp,
    unsigned char* __restrict__ sf, const float* __restrict__ gs, int R, int K)
{
    int KB=K/16; long nb=(long)R*KB;
    long b=(long)blockIdx.x*blockDim.x+threadIdx.x; if(b>=nb) return;
    int r=(int)(b/KB), kb=(int)(b%KB);
    const __half* xb=x+(long)r*K+kb*16;
    float v[16]; float am=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ v[i]=__half2float(xb[i]); am=fmaxf(am,fabsf(v[i])); }
    float ig=1.f/gs[0];
    // Four-Over-Six: scale-to-4 (denser, no 4-6 hole) vs scale-to-6 (full range), pick min-MSE per block.
    unsigned sc6=enc_ue4m3((am/6.f)*ig); if(sc6==0u&&am>0.f) sc6=1u; float sd6=dec_ue4m3(sc6); float inv6=(sd6>0.f)?(ig/sd6):0.f;
    unsigned sc4=enc_ue4m3((am/4.f)*ig); if(sc4==0u&&am>0.f) sc4=1u; float sd4=dec_ue4m3(sc4); float inv4=(sd4>0.f)?(ig/sd4):0.f;
    float e6=0.f,e4=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){
        float r6=(inv6>0.f)?dec_e2m1(enc_e2m1(v[i],inv6))/inv6:0.f; float d6=r6-v[i]; e6+=d6*d6;
        float r4=(inv4>0.f)?dec_e2m1(enc_e2m1(v[i],inv4))/inv4:0.f; float d4=r4-v[i]; e4+=d4*d4;
    }
    // ue4m3 code 127 (e=15,m=7) is NaN in hardware e4m3 (max normal=448=code126).
    // scale-to-4 can push the block scale >480 → enc_ue4m3 returns 127 → cuBLASLt
    // reads a NaN block scale → garbage. scale-to-6 maxes at 448 (code126, safe).
    // So only pick scale-to-4 when it stays a valid (non-NaN) ue4m3 code.
    unsigned sc; float inv;
    if(e4<e6 && sc4<127u){ sc=sc4; inv=inv4; } else { sc=sc6; inv=inv6; }
    int KB4=(KB+3)>>2;
    long soff=(long)(r>>7)*(512L*KB4)+(long)(kb>>2)*512+(r&31)*16+((r>>5)&3)*4+(kb&3);
    sf[soff]=(unsigned char)sc;
    unsigned char* o=xp+((long)r*K+kb*16)/2;
    #pragma unroll
    for(int i=0;i<8;i++){
        unsigned lo=enc_e2m1(v[2*i],inv), hi=enc_e2m1(v[2*i+1],inv);
        o[i]=(unsigned char)(lo|(hi<<4));
    }
}
// f32-input global-aware variant of lt_fp4_quant_gs (dense activations read
// straight from the f32 residual stream; block scales relative to gs[0]).
extern "C" __global__ void lt_fp4_quant_gs_f32(
    const float* __restrict__ x, unsigned char* __restrict__ xp,
    unsigned char* __restrict__ sf, const float* __restrict__ gs,
    int R, int K)
{
    int KB=K/16;
    long nb=(long)R*KB;
    long b=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=nb) return;
    int r=(int)(b/KB), kb=(int)(b%KB);
    const float* xb=x+(long)r*K+kb*16;
    float v[16]; float am=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ v[i]=xb[i]; am=fmaxf(am,fabsf(v[i])); }
    float ig=1.f/gs[0];
    unsigned sc=enc_ue4m3((am/6.f)*ig);
    if(sc==0u&&am>0.f) sc=1u;
    float sd=dec_ue4m3(sc);
    float inv=(sd>0.f)?(ig/sd):0.f;
    int KB4=(KB+3)>>2;
    long soff=(long)(r>>7)*(512L*KB4)+(long)(kb>>2)*512+(r&31)*16+((r>>5)&3)*4+(kb&3);
    sf[soff]=(unsigned char)sc;
    unsigned char* o=xp+((long)r*K+kb*16)/2;
    #pragma unroll
    for(int i=0;i<8;i++){
        unsigned lo=enc_e2m1(v[2*i],inv), hi=enc_e2m1(v[2*i+1],inv);
        o[i]=(unsigned char)(lo|(hi<<4));
    }
}
// D-scale fold: dscale[0] = act_global * weight_global (both gs[0]).
extern "C" __global__ void fp4_dscale(
    const float* __restrict__ ga, const float* __restrict__ gw, float* __restrict__ d)
{ if(threadIdx.x==0&&blockIdx.x==0) d[0]=ga[0]*gw[0]; }
// post-GEMM per-tensor output scale by a device scalar (f32 or f16 out).
extern "C" __global__ void fp4_apply_dscale_f32(float* __restrict__ o, const float* __restrict__ d, long n)
{ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; long st=(long)gridDim.x*blockDim.x; float s=d[0]; for(;i<n;i+=st) o[i]*=s; }
extern "C" __global__ void fp4_apply_dscale_f16(__half* __restrict__ o, const float* __restrict__ d, long n)
{ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; long st=(long)gridDim.x*blockDim.x; float s=d[0]; for(;i<n;i+=st) o[i]=__float2half(__half2float(o[i])*s); }

// Grouped variant of lt_fp4_quant_gs for the grouped block-scaled GEMM: the
// packed e2m1 stays GLOBAL row-major (per-group A ptr = base + start*K/2),
// but the ue4m3 scales land in a PER-GROUP swizzled blob addressed by the
// group-LOCAL row (each group's SF tensor must start at row 0 of its own
// 128-row-stripe layout). g_starts = device [n_groups] start rows
// (contiguous coverage of 0..R, sorted ascending); sfa_off = device
// [n_groups] BYTE offsets of each group's blob inside sf. gs = [global,
// inv_global] from nvfp4_gscale (per-call activation global).
extern "C" __global__ void lt_fp4_quant_grouped(
    const __half* __restrict__ x, unsigned char* __restrict__ xp,
    unsigned char* __restrict__ sf, const int* __restrict__ g_starts,
    const long long* __restrict__ sfa_off, const float* __restrict__ gs,
    const unsigned* __restrict__ st,
    int n_groups, int R, int K, int act_mode, float act_scale, int use_st)
{
    int KB=K/16;
    long nb=(long)R*KB;
    long b=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=nb) return;
    int r=(int)(b/KB), kb=(int)(b%KB);
    // largest g with g_starts[g] <= r (binary search; groups contiguous)
    int gl=0, gh=n_groups-1;
    while(gl<gh){ int gm=(gl+gh+1)>>1; if(g_starts[gm]<=r) gl=gm; else gh=gm-1; }
    int lr=r-g_starts[gl];
    int sr = use_st ? (int)st[r] : r;   // gather-on-read from the ungathered source
    const __half* xb=x+(long)sr*K+kb*16;
    float v[16]; float am=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ v[i]=fp4_act(__half2float(xb[i]),act_mode,act_scale); am=fmaxf(am,fabsf(v[i])); }
    float ig=gs[1];
    unsigned sc=enc_ue4m3((am/6.f)*ig);
    if(sc==0u&&am>0.f) sc=1u;
    float sd=dec_ue4m3(sc);
    float inv=(sd>0.f)?(ig/sd):0.f;
    int KB4=(KB+3)>>2;
    long soff=(long long)sfa_off[gl]+(long)(lr>>7)*(512L*KB4)+(long)(kb>>2)*512+(lr&31)*16+((lr>>5)&3)*4+(kb&3);
    sf[soff]=(unsigned char)sc;
    unsigned char* o=xp+((long)r*K+kb*16)/2;
    #pragma unroll
    for(int i=0;i<8;i++){
        unsigned lo=enc_e2m1(v[2*i],inv), hi=enc_e2m1(v[2*i+1],inv);
        o[i]=(unsigned char)(lo|(hi<<4));
    }
}

// Routed-act Four-Over-Six: same as lt_fp4_quant_grouped but adaptive 4/6 block
// scale (NVFP4 quality). code-127 guard: scale-to-4 can hit ue4m3 NaN code.
extern "C" __global__ void lt_fp4_quant_grouped_46(
    const __half* __restrict__ x, unsigned char* __restrict__ xp,
    unsigned char* __restrict__ sf, const int* __restrict__ g_starts,
    const long long* __restrict__ sfa_off, const float* __restrict__ gs,
    const unsigned* __restrict__ st,
    int n_groups, int R, int K, int act_mode, float act_scale, int use_st)
{
    int KB=K/16; long nb=(long)R*KB;
    long b=(long)blockIdx.x*blockDim.x+threadIdx.x; if(b>=nb) return;
    int r=(int)(b/KB), kb=(int)(b%KB);
    int gl=0, gh=n_groups-1;
    while(gl<gh){ int gm=(gl+gh+1)>>1; if(g_starts[gm]<=r) gl=gm; else gh=gm-1; }
    int lr=r-g_starts[gl];
    int sr = use_st ? (int)st[r] : r;
    const __half* xb=x+(long)sr*K+kb*16;
    float v[16]; float am=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ v[i]=fp4_act(__half2float(xb[i]),act_mode,act_scale); am=fmaxf(am,fabsf(v[i])); }
    float ig=gs[1];
    unsigned sc6=enc_ue4m3((am/6.f)*ig); if(sc6==0u&&am>0.f) sc6=1u; float sd6=dec_ue4m3(sc6); float inv6=(sd6>0.f)?(ig/sd6):0.f;
    unsigned sc4=enc_ue4m3((am/4.f)*ig); if(sc4==0u&&am>0.f) sc4=1u; float sd4=dec_ue4m3(sc4); float inv4=(sd4>0.f)?(ig/sd4):0.f;
    float e6=0.f,e4=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){
        float r6=(inv6>0.f)?dec_e2m1(enc_e2m1(v[i],inv6))/inv6:0.f; float d6=r6-v[i]; e6+=d6*d6;
        float r4=(inv4>0.f)?dec_e2m1(enc_e2m1(v[i],inv4))/inv4:0.f; float d4=r4-v[i]; e4+=d4*d4;
    }
    unsigned sc; float inv;
    if(e4<e6 && sc4<127u){ sc=sc4; inv=inv4; } else { sc=sc6; inv=inv6; }
    int KB4=(KB+3)>>2;
    long soff=(long long)sfa_off[gl]+(long)(lr>>7)*(512L*KB4)+(long)(kb>>2)*512+(lr&31)*16+((lr>>5)&3)*4+(kb&3);
    sf[soff]=(unsigned char)sc;
    unsigned char* o=xp+((long)r*K+kb*16)/2;
    #pragma unroll
    for(int i=0;i<8;i++){
        unsigned lo=enc_e2m1(v[2*i],inv), hi=enc_e2m1(v[2*i+1],inv);
        o[i]=(unsigned char)(lo|(hi<<4));
    }
}
extern "C" __global__ void pad_rows_f16(const __half* src, __half* dst, int k, int kpad, int rows){
    long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    long total = (long)rows*(long)kpad;
    if(i>=total) return;
    int c = (int)(i % (long)kpad); long r = i / (long)kpad;
    dst[i] = (c<k) ? src[r*(long)k + c] : __float2half(0.f);
}
"#;

/// Extend routed MoE descriptors (st/sw/offsets) with two identity groups
/// covering all `s` tokens, the shared expert folded into the grouped GEMM.
/// Returns (st_ext [mt+2s], sw_ext [mt+2s], off_ext [n_exp+3]).
pub fn moe_extend_groups(
    dev: &dyn Device, st: &Tensor, sw: &Tensor, off: &Tensor,
    mt: usize, s: usize, n_exp: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let i = |v: i32| v.to_le_bytes().to_vec();
    let total = mt + 2 * s;
    let st_e = Tensor::new(dev.alloc(total * 4)?, vec![total], DType::U32);
    let sw_e = Tensor::new(dev.alloc(total * 4)?, vec![total], DType::F32);
    let off_e = Tensor::new(dev.alloc((n_exp + 3) * 4)?, vec![n_exp + 3], DType::U32);
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "moe_extend_groups",
        &[(st.buffer.as_ref(), st.offset), (sw.buffer.as_ref(), sw.offset), (off.buffer.as_ref(), off.offset),
          (st_e.buffer.as_ref(), 0), (sw_e.buffer.as_ref(), 0), (off_e.buffer.as_ref(), 0)],
        &[i(mt as i32), i(s as i32), i(n_exp as i32)],
        [(total as u32).div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
    Ok((st_e, sw_e, off_e))
}

/// Quantize an f16 `[r, k]` row-major tensor to cuBLASLt block-scaled NVFP4:
/// returns `(pack, sf)`, packed e2m1 `[r*k/2]` bytes + the swizzled ue4m3
/// scale tensor sized `ceil(r/128)*512*ceil((k/16)/4)` bytes (zero-seeded so
/// padding rows decode to 0). Used for both weights (once, at setup) and
/// activations (per call). `k % 32 == 0` required.
pub fn lt_fp4_quant(dev: &dyn Device, x: &Tensor, r: usize, k: usize) -> Result<(Tensor, Tensor)> {
    if k % 32 != 0 { return Err(Error::Msg(format!("lt_fp4_quant: k {k} %32!=0"))); }
    let i = |v: i32| v.to_le_bytes().to_vec();
    let kb = k / 16;
    let kb4 = kb.div_ceil(4);
    let sf_bytes = r.div_ceil(128) * 512 * kb4;
    let pack = Tensor::new(dev.alloc((r * k) / 2)?, vec![(r * k) / 2], DType::U32);
    let sf = Tensor::new(dev.alloc_zeroed(sf_bytes)?, vec![sf_bytes], DType::U32);
    let nb = (r * kb) as u32;
    // f32 input quantizes directly (no f16 staging round-trip).
    let kname = if x.dtype == DType::F32 { "lt_fp4_quant_f32" } else { "lt_fp4_quant" };
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", kname,
        &[(x.buffer.as_ref(), x.offset), (pack.buffer.as_ref(), 0), (sf.buffer.as_ref(), 0)],
        &[i(r as i32), i(k as i32)],
        [nb.div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
    Ok((pack, sf))
}

/// Grouped-activation variant of [`lt_fp4_quant`] feeding the CUTLASS grouped
/// block-scaled GEMM: quantizes the sorted-token f16 `x [mt, k]` to packed
/// e2m1 `[mt*k/2]` bytes (global row-major) + a per-group swizzled ue4m3
/// scale pool where group g's blob (group-LOCAL rows, padded to 128-row
/// stripes, zero-seeded) starts at byte `sfa_off[g]`. A per-call activation
/// GLOBAL (amax/(6·448), dynamic, device-computed: no host sync) centers the
/// block scales in the ue4m3 normal range; the GEMM folds it back in via the
/// per-group alpha (see [`moe_grouped_gemm_cutlass_fp4`]). `g_starts` has
/// `n_groups+1` entries (`g_starts[n_groups] == mt`). Returns
/// `(pack, sf, sfa_off, gs)`, `gs` = device `[global, inv_global]` f32 pair.
pub fn lt_fp4_quant_grouped(
    dev: &dyn Device, x: &Tensor, mt: usize, k: usize, g_starts: &[usize],
) -> Result<(Tensor, Tensor, Vec<i64>, Tensor)> {
    if k % 32 != 0 { return Err(Error::Msg(format!("lt_fp4_quant_grouped: k {k} %32!=0"))); }
    let n_groups = g_starts.len() - 1;
    let kb = k / 16;
    let kb4 = kb.div_ceil(4);
    let mut sfa_off: Vec<i64> = Vec::with_capacity(n_groups);
    let mut sf_bytes: i64 = 0;
    for g in 0..n_groups {
        sfa_off.push(sf_bytes);
        let m = g_starts[g + 1] - g_starts[g];
        sf_bytes += (m.div_ceil(128) * 512 * kb4) as i64;
    }
    let i = |v: i32| v.to_le_bytes().to_vec();
    let l = |v: i64| v.to_le_bytes().to_vec();
    // per-call activation global: amax over the whole [mt,k] then amax→scale.
    let amax = Tensor::new(dev.alloc_zeroed(4)?, vec![1], DType::U32);
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_amax_slab",
        &[(x.buffer.as_ref(), x.offset), (amax.buffer.as_ref(), 0)],
        &[l((mt * k) as i64), i(1), i(0), 0f32.to_le_bytes().to_vec()],
        [((mt * k) as u32).div_ceil(256 * 64).max(1), 1, 1], [256, 1, 1], 0, false)?;
    let gs = Tensor::new(dev.alloc(8)?, vec![2], DType::F32);
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_gscale",
        &[(amax.buffer.as_ref(), 0), (gs.buffer.as_ref(), 0)],
        &[], [1, 1, 1], [1, 1, 1], 0, false)?;
    let pack = Tensor::new(dev.alloc((mt * k) / 2)?, vec![(mt * k) / 2], DType::U32);
    let sf = Tensor::new(dev.alloc_zeroed(sf_bytes as usize)?, vec![sf_bytes as usize], DType::U32);
    let starts_b: Vec<u8> = g_starts[..n_groups].iter().flat_map(|&s| (s as i32).to_le_bytes()).collect();
    let offs_b: Vec<u8> = sfa_off.iter().flat_map(|&o| o.to_le_bytes()).collect();
    let starts_d = Tensor::new(dev.upload(&starts_b)?, vec![n_groups], DType::U32);
    let offs_d = Tensor::new(dev.upload(&offs_b)?, vec![n_groups * 2], DType::U32);
    let nb = (mt * kb) as u32;
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "lt_fp4_quant_grouped",
        &[(x.buffer.as_ref(), x.offset), (pack.buffer.as_ref(), 0), (sf.buffer.as_ref(), 0),
          (starts_d.buffer.as_ref(), 0), (offs_d.buffer.as_ref(), 0), (gs.buffer.as_ref(), 0),
          (gs.buffer.as_ref(), 0)],
        &[i(n_groups as i32), i(mt as i32), i(k as i32), i(0), 0f32.to_le_bytes().to_vec(), i(0)],
        [nb.div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
    Ok((pack, sf, sfa_off, gs))
}

/// Quantize a contiguous f16 expert slab `[n_exp*n, k]` to per-expert
/// block-scaled NVFP4 in the CUTLASS/cuBLASLt canonical layout: per-expert
/// packed e2m1 (`n*k/2` bytes each, contiguous) + per-expert swizzled ue4m3
/// scales (`ceil(n/128)*512*ceil(k/64)` bytes each, contiguous, zero-seeded)
/// + per-expert GLOBALS `gw [n_exp]` f32 (amax/(6·448), keeps the block
/// scales in the ue4m3 normal range; the GEMM folds `gw` back in via the
/// per-group alpha). One quant dispatch per expert so every expert's SF blob
/// is laid out from local row 0 (an `[n_exp*n, k]` whole-slab quant would
/// misalign stripes whenever `n % 128 != 0`). Setup-time only.
pub fn lt_fp4_quant_slab(
    dev: &dyn Device, w: &Tensor, n_exp: usize, n: usize, k: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    if k % 32 != 0 { return Err(Error::Msg(format!("lt_fp4_quant_slab: k {k} %32!=0"))); }
    let i = |v: i32| v.to_le_bytes().to_vec();
    let l = |v: i64| v.to_le_bytes().to_vec();
    let kb = k / 16;
    let kb4 = kb.div_ceil(4);
    let sf_exp = n.div_ceil(128) * 512 * kb4;
    let pack_exp = n * k / 2;
    // per-expert globals (same two passes as the fragment-pack setup)
    let amax = Tensor::new(dev.alloc_zeroed(n_exp * 4)?, vec![n_exp], DType::U32);
    let gw = Tensor::empty(dev, vec![n_exp], DType::F32)?;
    let slab = (n * k) as i64;
    let total_elems = (n_exp * n * k) as u32;
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_amax_slab",
        &[(w.buffer.as_ref(), w.offset), (amax.buffer.as_ref(), 0)],
        &[l(slab), i(n_exp as i32), i(0), 0f32.to_le_bytes().to_vec()],
        [total_elems.div_ceil(256 * 8), 1, 1], [256, 1, 1], 0, false)?;
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_globals",
        &[(amax.buffer.as_ref(), 0), (gw.buffer.as_ref(), 0)],
        &[i(n_exp as i32)],
        [(n_exp as u32).div_ceil(64), 1, 1], [64, 1, 1], 0, false)?;
    let pack = Tensor::new(dev.alloc(n_exp * pack_exp)?, vec![n_exp * pack_exp], DType::U32);
    let sf = Tensor::new(dev.alloc_zeroed(n_exp * sf_exp)?, vec![n_exp * sf_exp], DType::U32);
    let nb = (n * kb) as u32;
    // B8a: routed-expert WEIGHT 4/6 (Four-Over-Six) under NEMOTRON_FP4_46, free
    // (setup-time, covers ~90% of params). Same per-expert global gw[e] (gs[0]).
    let kname = if std::env::var("NEMOTRON_FP4_46").is_ok() { "lt_fp4_quant_gs_46" } else { "lt_fp4_quant_gs" };
    for e in 0..n_exp {
        dev.dispatch_raw_cuda(
            MOE_FP4_SRC, "moe_fp4.cu", kname,
            &[(w.buffer.as_ref(), w.offset + e * n * k * 2),
              (pack.buffer.as_ref(), e * pack_exp), (sf.buffer.as_ref(), e * sf_exp),
              (gw.buffer.as_ref(), e * 4)],
            &[i(n as i32), i(k as i32)],
            [nb.div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
    }
    Ok((pack, sf, gw))
}

/// Device-descriptor variant of [`lt_fp4_quant_grouped`]: per-group SFA
/// offsets are computed ON DEVICE from the router offsets (`off_dev`, u32
/// `[n_groups+1]`), no host offsets, no uploads, graph-safe. The SFA pool is
/// sized worst-case (`mt/128 + n_groups` stripes). Returns `(pack, sf, gs)`;
/// the GEMM run derives the same per-group offsets from `off_dev`.
pub fn lt_fp4_quant_grouped_dev(
    dev: &dyn Device, x: &Tensor, mt: usize, k: usize, off_dev: &Tensor, n_groups: usize,
    act_mode: i32, act_scale: f32,
    gather_st: Option<(&Tensor, usize)>,
) -> Result<(Tensor, Tensor, Tensor)> {
    if k % 32 != 0 { return Err(Error::Msg(format!("lt_fp4_quant_grouped_dev: k {k} %32!=0"))); }
    let i = |v: i32| v.to_le_bytes().to_vec();
    let l = |v: i64| v.to_le_bytes().to_vec();
    let kb = k / 16;
    let kb4 = kb.div_ceil(4);
    let sf_bytes = (mt.div_ceil(128) + n_groups) * 512 * kb4; // worst-case stripes
    let amax_rows = gather_st.map(|(_, s_rows)| s_rows).unwrap_or(mt);
    let amax = Tensor::new(dev.alloc_zeroed(4)?, vec![1], DType::U32);
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_amax_slab",
        &[(x.buffer.as_ref(), x.offset), (amax.buffer.as_ref(), 0)],
        &[l((amax_rows * k) as i64), i(1), i(act_mode), act_scale.to_le_bytes().to_vec()],
        [((amax_rows * k) as u32).div_ceil(256 * 64).max(1), 1, 1], [256, 1, 1], 0, false)?;
    let gs = Tensor::new(dev.alloc(8)?, vec![2], DType::F32);
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_gscale",
        &[(amax.buffer.as_ref(), 0), (gs.buffer.as_ref(), 0)],
        &[], [1, 1, 1], [1, 1, 1], 0, false)?;
    let sfa_off = Tensor::new(dev.alloc(n_groups * 8)?, vec![n_groups * 2], DType::U32);
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "fp4_sfa_offsets",
        &[(off_dev.buffer.as_ref(), off_dev.offset), (sfa_off.buffer.as_ref(), 0)],
        &[i(n_groups as i32), i(kb4 as i32)],
        [1, 1, 1], [1, 1, 1], 0, false)?;
    let pack = Tensor::new(dev.alloc((mt * k) / 2)?, vec![(mt * k) / 2], DType::U32);
    let sf = Tensor::new(dev.alloc_zeroed(sf_bytes)?, vec![sf_bytes], DType::U32);
    let nb = (mt * kb) as u32;
    // off_dev doubles as g_starts (u32 values, n_groups+1 entries; the kernel
    // binary-searches the first n_groups). B8a: routed-ACT 4/6 under NEMOTRON_FP4_46.
    let gkname = if std::env::var("NEMOTRON_FP4_46_ACT").is_ok() { "lt_fp4_quant_grouped_46" } else { "lt_fp4_quant_grouped" };
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", gkname,
        &[(x.buffer.as_ref(), x.offset), (pack.buffer.as_ref(), 0), (sf.buffer.as_ref(), 0),
          (off_dev.buffer.as_ref(), off_dev.offset), (sfa_off.buffer.as_ref(), 0), (gs.buffer.as_ref(), 0),
          match gather_st { Some((t, _)) => (t.buffer.as_ref(), t.offset), None => (gs.buffer.as_ref(), 0) }],
        &[i(n_groups as i32), i(mt as i32), i(k as i32), i(act_mode), act_scale.to_le_bytes().to_vec(),
          i(if gather_st.is_some() { 1 } else { 0 })],
        [nb.div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
    Ok((pack, sf, gs))
}

/// Per-group alpha for the identity-expert device path: alpha[g] = gs[0]*gw[g],
/// written into the PERSISTENT `alpha` buffer the prepared handle points at.
pub fn fp4_group_alpha_id(dev: &dyn Device, gs: &Tensor, gw: &Tensor, alpha: &Tensor, n: usize) -> Result<()> {
    let i = |v: i32| v.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "fp4_group_alpha_id",
        &[(gs.buffer.as_ref(), gs.offset), (gw.buffer.as_ref(), gw.offset), (alpha.buffer.as_ref(), 0)],
        &[i(n as i32)],
        [(n as u32).div_ceil(64), 1, 1], [64, 1, 1], 0, false)
}

/// CUTLASS grouped block-scaled NVFP4 MoE GEMM, `out[t,n](f16) = Σ_k
/// A[t,k]·W[eid][n,k]` over sorted-token groups with BOTH operands in packed
/// e2m1 + swizzled ue4m3 block-16 scales. `a_pack`/`a_sf`/`sfa_off`/`a_gs`
/// from [`lt_fp4_quant_grouped`]; `w_pack`/`w_sf`/`w_gw` from
/// [`lt_fp4_quant_slab`]. The per-tensor globals are folded back in through
/// the GEMM's per-group alpha (`gA·gw[eid]`, built device-side: no host
/// sync). `g_starts[g]..g_starts[g+1]` = group g's token rows;
/// `expert_ids[g]` selects the weight slab. ONE GEMM launch for all groups.
/// Returns `out [mt, n]` f16. CUDA + CUTLASS_DIR sm_120a/121a build only.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gemm_cutlass_fp4(
    dev: &dyn Device,
    a_pack: &Tensor, a_sf: &Tensor, sfa_off: &[i64], a_gs: &Tensor,
    w_pack: &Tensor, w_sf: &Tensor, w_gw: &Tensor,
    g_starts: &[usize], expert_ids: &[usize],
    n: usize, k: usize,
) -> Result<Tensor> {
    let n_active = expert_ids.len();
    let mt = g_starts.last().copied().unwrap_or(0);
    let out = Tensor::empty(dev, vec![mt.max(1), n], DType::F16)?;
    if n_active == 0 || mt == 0 { return Ok(out); }
    let group_rows: Vec<i32> = (0..n_active).map(|g| (g_starts[g + 1] - g_starts[g]) as i32).collect();
    let eids: Vec<i32> = expert_ids.iter().map(|&e| e as i32).collect();
    // device per-group alpha = gA[0] * gw[eid[g]]
    let i = |v: i32| v.to_le_bytes().to_vec();
    let eids_b: Vec<u8> = eids.iter().flat_map(|&e| e.to_le_bytes()).collect();
    let eids_d = Tensor::new(dev.upload(&eids_b)?, vec![n_active], DType::U32);
    let alpha = Tensor::empty(dev, vec![n_active], DType::F32)?;
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "fp4_group_alpha",
        &[(a_gs.buffer.as_ref(), a_gs.offset), (w_gw.buffer.as_ref(), w_gw.offset),
          (eids_d.buffer.as_ref(), 0), (alpha.buffer.as_ref(), 0)],
        &[i(n_active as i32)],
        [(n_active as u32).div_ceil(64), 1, 1], [64, 1, 1], 0, false)?;
    dev.moe_grouped_cutlass_fp4(
        a_pack.buffer.as_ref(), a_sf.buffer.as_ref(),
        w_pack.buffer.as_ref(), w_sf.buffer.as_ref(),
        out.buffer.as_ref(), &group_rows, &eids, sfa_off,
        Some(alpha.buffer.as_ref()), n, k)?;
    Ok(out)
}

/// Zero-pad the K (column) dim of an f16 `[rows,k]` tensor to `[rows,kpad]`.
/// No-op clone when `kpad==k`. Used to satisfy the mxf8f6f4 SF atom-K=128
/// constraint (e.g. down-proj K=1856 -> 1920); padded cols add 0 to the dot.
pub fn pad_rows_f16(dev: &dyn Device, x: &Tensor, rows: usize, k: usize, kpad: usize) -> Result<Tensor> {
    if kpad == k { return Ok(x.clone()); }
    let i = |v: i32| v.to_le_bytes().to_vec();
    let dst = Tensor::new(dev.alloc(rows * kpad * 2)?, vec![rows * kpad], DType::F16);
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "pad_rows_f16",
        &[(x.buffer.as_ref(), x.offset), (dst.buffer.as_ref(), 0)],
        &[i(k as i32), i(kpad as i32), i(rows as i32)],
        [((rows * kpad) as u32).div_ceil(256).max(1), 1, 1], [256, 1, 1], 0, false)?;
    Ok(dst)
}

/// W4A8 per-group SFA byte offsets (atom-padded, ue8m0 = 1 byte/elem) for the
/// grouped act-quant + GEMM. MX per-32 blocks: kb = k/32. Per group the SFA
/// section is `ceil(m/128)*512*ceil(kb/4)` bytes (same swizzle atom as NVFP4,
/// only the block size differs). Returns `(offsets[n_groups], total_bytes)`.
pub fn w4a8_sfa_offsets(g_starts: &[usize], k: usize) -> (Vec<i64>, usize) {
    let n_groups = g_starts.len() - 1;
    let kb4 = (k / 32).div_ceil(4);
    let mut offs = Vec::with_capacity(n_groups);
    let mut acc: i64 = 0;
    for g in 0..n_groups {
        offs.push(acc);
        let m = g_starts[g + 1] - g_starts[g];
        acc += (m.div_ceil(128) * 512 * kb4) as i64;
    }
    (offs, acc as usize)
}

/// W4A8 grouped MoE GEMM: fp8 e4m3 acts x mxfp4 weights (cutlass mxf8f6f4).
/// Both SF are per-32 ue8m0 (full-range), so NO per-tensor global scale / alpha
/// (alpha=1.0). `sfa_off` = host per-group SFA byte offsets (atom-padded);
/// `sfb_exp_bytes` = per-expert SFB stride in bytes (from `w4a8_packw`).
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gemm_w4a8(
    dev: &dyn Device,
    a_pack: &Tensor, a_sf: &Tensor, sfa_off: &[i64],
    w_pack: &Tensor, w_sf: &Tensor, sfb_exp_bytes: i64,
    g_starts: &[usize], expert_ids: &[usize],
    n: usize, k: usize,
) -> Result<Tensor> {
    let n_active = expert_ids.len();
    let mt = g_starts.last().copied().unwrap_or(0);
    let out = Tensor::empty(dev, vec![mt.max(1), n], DType::F16)?;
    if n_active == 0 || mt == 0 { return Ok(out); }
    let group_rows: Vec<i32> = (0..n_active).map(|g| (g_starts[g + 1] - g_starts[g]) as i32).collect();
    let eids: Vec<i32> = expert_ids.iter().map(|&e| e as i32).collect();
    dev.moe_grouped_cutlass_w4a8(
        a_pack.buffer.as_ref(), a_sf.buffer.as_ref(),
        w_pack.buffer.as_ref(), w_sf.buffer.as_ref(),
        out.buffer.as_ref(), &group_rows, &eids, sfa_off,
        sfb_exp_bytes, None, n, k)?;
    Ok(out)
}

/// W4A8 fp8 act-quant: f16 `x[mt,k]` -> e4m3 bytes `[mt*k]` + ue8m0 SF (per-32,
/// group-aware SFA). `sfa_off` = host per-group SFA byte offsets (atom-padded).
/// Returns `(q, sf)`. `sf_total_bytes` = caller-sized SFA buffer length.
pub fn w4a8_actquant_grouped(
    dev: &dyn Device,
    x: &Tensor, group_rows: &[i32], sfa_off: &[i64], sf_total_bytes: usize,
    mt: usize, k: usize,
) -> Result<(Tensor, Tensor)> {
    let q = Tensor::new(dev.alloc(mt * k)?, vec![mt * k], DType::U32);
    let sf = Tensor::new(dev.alloc_zeroed(sf_total_bytes.max(1))?, vec![sf_total_bytes.max(1)], DType::U32);
    dev.w4a8_actquant(x.buffer.as_ref(), q.buffer.as_ref(), sf.buffer.as_ref(),
        group_rows, sfa_off, mt, k)?;
    Ok((q, sf))
}

/// W4A8 mxfp4 weight pack: f16 `w[n_exp,n,k]` -> e2m1 bytes + ue8m0 SF (per-32).
/// Returns `(packed, sf, sfb_exp_elems)`; the GEMM's `sfb_exp_bytes` = sfb_exp_elems.
pub fn w4a8_packw(
    dev: &dyn Device,
    w: &Tensor, n_exp: usize, n: usize, k: usize,
) -> Result<(Tensor, Tensor, i64)> {
    let packed = Tensor::new(dev.alloc(n_exp * n * k / 2)?, vec![n_exp * n * k / 2], DType::U32);
    // SFB size: per-expert atom-padded element count, queried from the first pack.
    // Over-allocate generously (per-32 -> n*k/32 + atom padding) then trust the
    // returned per-expert element count for the GEMM stride.
    let sf_cap = n_exp * ((n * k / 32) + 4 * 128 * 4);
    let sf = Tensor::new(dev.alloc_zeroed(sf_cap.max(1))?, vec![sf_cap.max(1)], DType::U32);
    let sfb_exp_elems = dev.w4a8_packw(w.buffer.as_ref(), packed.buffer.as_ref(), sf.buffer.as_ref(),
        n_exp, n, k)?;
    Ok((packed, sf, sfb_exp_elems))
}

/// `sfb_exp_bytes` = per-expert SFB stride in bytes (from `w8a8_packw`).
#[allow(clippy::too_many_arguments)]
/// Validated NVFP4 (f16-act) grouped-MoE GEMM (marlin). `a` f16 `[m,K]`, `b` fp4
/// marlin-tile u32 `[n_exp,K/16,N*2]`, `b_scales` S0E5M3 fp8 (u32-viewed),
/// `global_scale` f32 `[n_exp]`, `topk_weights` f32 `[m*top_k]`; routing
/// (`sorted_token_ids`/`expert_ids`/`num_tokens_past_padded`) device i32.
/// Returns f16 `[m*top_k, N]`. mul_topk_weights + fp32 reduce on.
#[allow(clippy::too_many_arguments)]
pub fn moe_marlin_gemm(
    dev: &dyn Device,
    a: &Tensor, b: &Tensor, b_scales: &Tensor, global_scale: &Tensor, topk_weights: &Tensor,
    sorted_token_ids: &Tensor, expert_ids: &Tensor, num_tokens_past_padded: &Tensor,
    prob_m: usize, prob_n: usize, prob_k: usize, num_experts: usize, top_k: usize,
    group_size: usize, moe_block: usize, sms: i32,
) -> Result<Tensor> {
    let mt = (prob_m * top_k).max(1);
    let out = Tensor::empty(dev, vec![mt, prob_n.max(1)], DType::F16)?;
    let num_groups = prob_k / group_size;
    let ctmp = ((sms as usize) * 4 * moe_block * 256).max(1);
    let c_tmp = Tensor::new(dev.alloc_zeroed(ctmp * 4)?, vec![ctmp], DType::F32);
    let wsn = ((sms as usize) * 4).max(1);
    let ws = Tensor::new(dev.alloc_zeroed(wsn * 4)?, vec![wsn], DType::U32);
    dev.moe_marlin_gemm(
        a.buffer.as_ref(), b.buffer.as_ref(), out.buffer.as_ref(), c_tmp.buffer.as_ref(),
        b_scales.buffer.as_ref(), global_scale.buffer.as_ref(), topk_weights.buffer.as_ref(),
        sorted_token_ids.buffer.as_ref(), expert_ids.buffer.as_ref(), num_tokens_past_padded.buffer.as_ref(),
        ws.buffer.as_ref(),
        moe_block, num_experts, top_k, true, prob_m, prob_n, prob_k, num_groups, group_size, sms, true,
    )?;
    Ok(out)
}

/// Repack one expert's GPTQ-packed `[K/8,N]` u32 weights -> marlin tile `[K/16,N*2]` u32 (device).
pub fn marlin_repack(dev: &dyn Device, b_q: &Tensor, size_k: usize, size_n: usize, sms: i32, max_shared_mem: i32) -> Result<Tensor> {
    let out_elems = ((size_k / 16) * (size_n * 2)).max(1);
    let out = Tensor::new(dev.alloc(out_elems * 4)?, vec![out_elems], DType::U32);
    dev.marlin_repack(b_q.buffer.as_ref(), out.buffer.as_ref(), size_k, size_n, sms, max_shared_mem)?;
    Ok(out)
}

/// On-device marlin routing build from per-expert offsets `off` [n_exp+1] (device i32).
pub fn marlin_build_routing(dev:&dyn Device, off:&Tensor, n_exp:usize, blk:usize, mt:usize)->Result<(Tensor,Tensor,Tensor)>{
    let max_blocks = mt/blk + n_exp + 1;
    let stid=Tensor::new(dev.alloc((max_blocks*blk).max(1)*4)?, vec![(max_blocks*blk).max(1)], DType::U32);
    let eid=Tensor::new(dev.alloc(max_blocks.max(1)*4)?, vec![max_blocks.max(1)], DType::U32);
    let ntpp=Tensor::new(dev.alloc_zeroed(4)?, vec![1], DType::U32);
    dev.marlin_build_routing(off.buffer.as_ref(), n_exp, blk, mt, stid.buffer.as_ref(), eid.buffer.as_ref(), ntpp.buffer.as_ref())?;
    Ok((stid,eid,ntpp))
}

pub fn moe_grouped_gemm_w8a8(
    dev: &dyn Device,
    a_pack: &Tensor, a_sf: &Tensor, sfa_off: &[i64],
    w_pack: &Tensor, w_sf: &Tensor, sfb_exp_bytes: i64,
    g_starts: &[usize], expert_ids: &[usize],
    n: usize, k: usize,
) -> Result<Tensor> {
    let n_active = expert_ids.len();
    let mt = g_starts.last().copied().unwrap_or(0);
    let out = Tensor::empty(dev, vec![mt.max(1), n], DType::F16)?;
    if n_active == 0 || mt == 0 { return Ok(out); }
    let group_rows: Vec<i32> = (0..n_active).map(|g| (g_starts[g + 1] - g_starts[g]) as i32).collect();
    let eids: Vec<i32> = expert_ids.iter().map(|&e| e as i32).collect();
    dev.moe_grouped_cutlass_w8a8(
        a_pack.buffer.as_ref(), a_sf.buffer.as_ref(),
        w_pack.buffer.as_ref(), w_sf.buffer.as_ref(),
        out.buffer.as_ref(), &group_rows, &eids, sfa_off,
        sfb_exp_bytes, None, n, k)?;
    Ok(out)
}

/// W4A8 fp8 act-quant: f16 `x[mt,k]` -> e4m3 bytes `[mt*k]` + ue8m0 SF (per-32,
/// group-aware SFA). `sfa_off` = host per-group SFA byte offsets (atom-padded).
/// Returns `(q, sf)`. `sf_total_bytes` = caller-sized SFA buffer length.
pub fn w8a8_actquant_grouped(
    dev: &dyn Device,
    x: &Tensor, group_rows: &[i32], sfa_off: &[i64], sf_total_bytes: usize,
    mt: usize, k: usize,
) -> Result<(Tensor, Tensor)> {
    let q = Tensor::new(dev.alloc(mt * k)?, vec![mt * k], DType::U32);
    let sf = Tensor::new(dev.alloc_zeroed(sf_total_bytes.max(1))?, vec![sf_total_bytes.max(1)], DType::U32);
    dev.w8a8_actquant(x.buffer.as_ref(), q.buffer.as_ref(), sf.buffer.as_ref(),
        group_rows, sfa_off, mt, k)?;
    Ok((q, sf))
}

/// W4A8 mxfp4 weight pack: f16 `w[n_exp,n,k]` -> e2m1 bytes + ue8m0 SF (per-32).
/// Returns `(packed, sf, sfb_exp_elems)`; the GEMM's `sfb_exp_bytes` = sfb_exp_elems.
pub fn w8a8_packw(
    dev: &dyn Device,
    w: &Tensor, n_exp: usize, n: usize, k: usize,
) -> Result<(Tensor, Tensor, i64)> {
    let packed = Tensor::new(dev.alloc(n_exp * n * k)?, vec![n_exp * n * k], DType::U32);
    // SFB size: per-expert atom-padded element count, queried from the first pack.
    // Over-allocate generously (per-32 -> n*k/32 + atom padding) then trust the
    // returned per-expert element count for the GEMM stride.
    let sf_cap = n_exp * ((n * k / 32) + 4 * 128 * 4);
    let sf = Tensor::new(dev.alloc_zeroed(sf_cap.max(1))?, vec![sf_cap.max(1)], DType::U32);
    let sfb_exp_elems = dev.w8a8_packw(w.buffer.as_ref(), packed.buffer.as_ref(), sf.buffer.as_ref(),
        n_exp, n, k)?;
    Ok((packed, sf, sfb_exp_elems))
}


/// Quantize an f16 tensor to e4m3 with a dynamic per-tensor scale.
/// Returns `(q, sc)`, e4m3 bytes `[n]` + a 2-f32 scale buffer
/// `[dequant_scale, quant_scale]` (the GEMM consumes word 0).
pub fn fp8_quant(dev: &dyn Device, x: &Tensor, n: usize, clip: f32) -> Result<(Tensor, Tensor)> {
    let i = |v: i32| v.to_le_bytes().to_vec();
    let l = |v: i64| v.to_le_bytes().to_vec();
    let amax = Tensor::new(dev.alloc_zeroed(4)?, vec![1], DType::U32);
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_amax_slab",
        &[(x.buffer.as_ref(), x.offset), (amax.buffer.as_ref(), 0)],
        &[l(n as i64), i(1), i(0), 0f32.to_le_bytes().to_vec()],
        [((n as u32).div_ceil(256 * 64)).max(1), 1, 1], [256, 1, 1], 0, false)?;
    let sc = Tensor::new(dev.alloc(8)?, vec![2], DType::F32);
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "fp8_scale",
        &[(amax.buffer.as_ref(), 0), (sc.buffer.as_ref(), 0)],
        &[clip.to_le_bytes().to_vec()], [1, 1, 1], [1, 1, 1], 0, false)?;
    let q = Tensor::new(dev.alloc(n)?, vec![n], DType::U32);
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "fp8_quant",
        &[(x.buffer.as_ref(), x.offset), (q.buffer.as_ref(), 0), (sc.buffer.as_ref(), 0)],
        &[l(n as i64)],
        [(n as u32).div_ceil(256 * 4).max(1), 1, 1], [256, 1, 1], 0, false)?;
    Ok((q, sc))
}

/// Fused block-Hadamard(128) along K + global amax. Returns `(rotated [rows,k]
/// f16, amax [1] u32 bits)` for a subsequent fp8 quant that skips its own amax.
pub fn fwht128_amax(dev: &dyn Device, x: &Tensor, rows: usize, k: usize) -> Result<(Tensor, Tensor)> {
    if k % 128 != 0 { return Err(Error::Msg(format!("fwht128_amax: k {k} %128 != 0"))); }
    let y = Tensor::new(dev.alloc(rows * k * 2)?, vec![rows * k], DType::F16);
    let amax = Tensor::new(dev.alloc_zeroed(4)?, vec![1], DType::U32);
    let nb = (rows * (k / 128)) as u32;
    let i = |v: i32| v.to_le_bytes().to_vec();
    let l = |v: i64| v.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "fwht128_amax",
        &[(x.buffer.as_ref(), x.offset), (y.buffer.as_ref(), 0), (amax.buffer.as_ref(), 0)],
        &[l(rows as i64), i(k as i32)],
        [nb, 1, 1], [128, 1, 1], 0, false)?;
    Ok((y.reshaped(vec![rows, k]), amax))
}

/// Per-tensor fp8 quant using a PRECOMPUTED amax (from `fwht128_amax`), with
/// optional clip. Returns `(q bytes, sc [2] f32)`. Skips the amax pass.
pub fn fp8_quant_pre(dev: &dyn Device, x: &Tensor, n: usize, amax: &Tensor, clip: f32) -> Result<(Tensor, Tensor)> {
    let i = |v: i32| v.to_le_bytes().to_vec();
    let l = |v: i64| v.to_le_bytes().to_vec();
    let sc = Tensor::new(dev.alloc(8)?, vec![2], DType::F32);
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "fp8_scale",
        &[(amax.buffer.as_ref(), 0), (sc.buffer.as_ref(), 0)],
        &[clip.to_le_bytes().to_vec()], [1, 1, 1], [1, 1, 1], 0, false)?;
    let q = Tensor::new(dev.alloc(n)?, vec![n], DType::U32);
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "fp8_quant",
        &[(x.buffer.as_ref(), x.offset), (q.buffer.as_ref(), 0), (sc.buffer.as_ref(), 0)],
        &[l(n as i64)],
        [(n as u32).div_ceil(256 * 4).max(1), 1, 1], [256, 1, 1], 0, false)?;
    Ok((q, sc))
}

/// Fused-path pass 1: block-Hadamard(128) along K + global amax, NO rotated
/// output (saves the [rows,k] f16 write+read vs `fwht128_amax`). Returns amax bits.
pub fn fwht128_amaxonly(dev: &dyn Device, x: &Tensor, rows: usize, k: usize) -> Result<Tensor> {
    if k % 128 != 0 { return Err(Error::Msg(format!("fwht128_amaxonly: k {k} %128 != 0"))); }
    let amax = Tensor::new(dev.alloc_zeroed(4)?, vec![1], DType::U32);
    let nb = (rows * (k / 128)) as u32;
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "fwht128_amaxonly",
        &[(x.buffer.as_ref(), x.offset), (amax.buffer.as_ref(), 0)],
        &[(rows as i64).to_le_bytes().to_vec(), (k as i32).to_le_bytes().to_vec()],
        [nb, 1, 1], [128, 1, 1], 0, false)?;
    Ok(amax)
}

/// Fused-path pass 2: re-rotate `x` (Hadamard 128 along K) AND quant to e4m3 in
/// one kernel reading `x` directly (no rotated-tensor round-trip), using a
/// precomputed amax. Returns `(q bytes, sc [2] f32)`.
pub fn fwht128_quant_pre(dev: &dyn Device, x: &Tensor, rows: usize, k: usize, amax: &Tensor, clip: f32) -> Result<(Tensor, Tensor)> {
    if k % 128 != 0 { return Err(Error::Msg(format!("fwht128_quant_pre: k {k} %128 != 0"))); }
    let sc = Tensor::new(dev.alloc(8)?, vec![2], DType::F32);
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "fp8_scale",
        &[(amax.buffer.as_ref(), 0), (sc.buffer.as_ref(), 0)],
        &[clip.to_le_bytes().to_vec()], [1, 1, 1], [1, 1, 1], 0, false)?;
    let n = rows * k;
    let q = Tensor::new(dev.alloc(n)?, vec![n], DType::U32);
    let nb = (rows * (k / 128)) as u32;
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "fwht128_quant",
        &[(x.buffer.as_ref(), x.offset), (q.buffer.as_ref(), 0), (sc.buffer.as_ref(), 0)],
        &[(rows as i64).to_le_bytes().to_vec(), (k as i32).to_le_bytes().to_vec()],
        [nb, 1, 1], [128, 1, 1], 0, false)?;
    Ok((q, sc))
}

/// Block Walsh-Hadamard transform (128) along K of an f16 `[rows, k]` tensor.
/// k must be a multiple of 128. Orthonormal; used to spread activation outliers
/// before a per-tensor fp8 quant (and applied identically to the weight offline).
pub fn fwht128(dev: &dyn Device, x: &Tensor, rows: usize, k: usize) -> Result<Tensor> {
    if k % 128 != 0 { return Err(Error::Msg(format!("fwht128: k {k} %128 != 0"))); }
    let y = Tensor::new(dev.alloc(rows * k * 2)?, vec![rows * k], DType::F16);
    let nb = (rows * (k / 128)) as u32;
    let i = |v: i32| v.to_le_bytes().to_vec();
    let l = |v: i64| v.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "fwht128",
        &[(x.buffer.as_ref(), x.offset), (y.buffer.as_ref(), 0)],
        &[l(rows as i64), i(k as i32)],
        [nb, 1, 1], [128, 1, 1], 0, false)?;
    Ok(y.reshaped(vec![rows, k]))
}

/// FP8 e4m3 dense GEMM with per-tensor scales (vendor-recipe precision for
/// the Mamba in/out projections + shared expert on this model family).
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8(
    dev: &dyn Device,
    x_q: &Tensor, x_sc: &Tensor,
    w_q: &Tensor, w_sc: &Tensor,
    m: usize, n: usize, k: usize, out_f32: bool,
) -> Result<Tensor> {
    let out = Tensor::empty(dev, vec![m, n], if out_f32 { DType::F32 } else { DType::F16 })?;
    dev.gemm_fp8(
        x_q.buffer.as_ref(), x_sc.buffer.as_ref(),
        w_q.buffer.as_ref(), w_sc.buffer.as_ref(),
        out.buffer.as_ref(), m, n, k, out_f32)?;
    Ok(out)
}

/// Per-row e4m3 quant of an f16/f32 `[r,k]` tensor → `(q [r*k] bytes, scale [r] f32)`.
/// Per-channel (weights, rows=n) / per-token (acts, rows=m), the quality-
/// preserving FP8 granularity (per-tensor crushed cosine to ~0.94).
pub fn fp8_quant_perrow(dev: &dyn Device, x: &Tensor, r: usize, k: usize) -> Result<(Tensor, Tensor)> {
    let i = |v: i32| v.to_le_bytes().to_vec();
    let q = Tensor::new(dev.alloc(r * k)?, vec![r * k], DType::U32);
    let sc = Tensor::new(dev.alloc(r * 4)?, vec![r], DType::F32);
    let kn = if x.dtype == DType::F32 { "fp8_quant_perrow_f32" } else { "fp8_quant_perrow" };
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", kn,
        &[(x.buffer.as_ref(), x.offset), (q.buffer.as_ref(), 0), (sc.buffer.as_ref(), 0)],
        &[i(r as i32), i(k as i32)], [r as u32, 1, 1], [256, 1, 1], 0, false)?;
    Ok((q, sc))
}

/// FP8 e4m3 GEMM with PER-CHANNEL weight + PER-TOKEN act scales, folded as a
/// post-pass (`D[r,c] *= xsc[r]·wsc[c]`), sidesteps cuBLASLt's unsupported
/// OUTER_VEC FP8 path. `x_q`/`x_sc` per-token (rows=m), `w_q`/`w_sc`
/// per-channel (rows=n). Quality-preserving (cosine >0.98 typical). out[m,n] f32.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_perchan(
    dev: &dyn Device,
    x_q: &Tensor, x_sc: &Tensor,
    w_q: &Tensor, w_sc: &Tensor,
    m: usize, n: usize, k: usize,
) -> Result<Tensor> {
    let out = Tensor::empty(dev, vec![m, n], DType::F32)?;
    // identity per-tensor scale (1.0), the real per-channel/per-token scales
    // are folded in the post-pass below. D_raw = Wq·Xq in f32.
    let one = scalar_buf(dev, (1.0f32).to_bits())?;
    dev.gemm_fp8(
        x_q.buffer.as_ref(), one.as_ref(),
        w_q.buffer.as_ref(), one.as_ref(),
        out.buffer.as_ref(), m, n, k, true)?;
    let i = |v: i32| v.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "fp8_rowcol_scale_f32",
        &[(out.buffer.as_ref(), 0), (x_sc.buffer.as_ref(), x_sc.offset), (w_sc.buffer.as_ref(), w_sc.offset)],
        &[i(m as i32), i(n as i32)],
        [((m*n) as u32).div_ceil(256*4).max(1), 1, 1], [256, 1, 1], 0, false)?;
    Ok(out)
}

/// Block-scaled NVFP4 dense GEMM: `out[m,n](f16) = X·Wᵀ` with both sides in
/// the `lt_fp4_quant` format. Routes through the backend's cublasLt FP4 path
/// (~4x the f16 tensor-core GEMM on GB10).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp4(
    dev: &dyn Device,
    x_pack: &Tensor, x_sf: &Tensor,
    w_pack: &Tensor, w_sf: &Tensor,
    m: usize, n: usize, k: usize, out_f32: bool,
    d_scale: Option<&Tensor>,
) -> Result<Tensor> {
    let out = Tensor::empty(dev, vec![m, n], if out_f32 { DType::F32 } else { DType::F16 })?;
    dev.gemm_fp4(
        x_pack.buffer.as_ref(), x_sf.buffer.as_ref(),
        w_pack.buffer.as_ref(), w_sf.buffer.as_ref(),
        out.buffer.as_ref(), m, n, k, out_f32, None)?;
    if let Some(ds) = d_scale {
        let l = |v: i64| v.to_le_bytes().to_vec();
        let ne = (m * n) as i64;
        let kn = if out_f32 { "fp4_apply_dscale_f32" } else { "fp4_apply_dscale_f16" };
        dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", kn,
            &[(out.buffer.as_ref(), 0), (ds.buffer.as_ref(), ds.offset)],
            &[l(ne)], [((ne as u32).div_ceil(256*4)).max(1),1,1],[256,1,1],0,false)?;
    }
    Ok(out)
}

/// Global-aware dense FP4 quant: per-TENSOR global keeps the ue4m3 block
/// scales in normal range (block-amax/6 relative to the global) instead of
/// the subnormal floor that wrecks small-magnitude dense weights (cosine
/// 0.44 -> ~0.97). Returns (pack, sf, gs[2]=[global,unused]); the GEMM folds
/// the act·weight globals back via its D-scale. f16 or f32 input.
/// Four-Over-Six variant (adaptive 4/6 block scale; NVFP4 quality, GEMM-identical = free). f16 in.
pub fn lt_fp4_quant_g_46(dev: &dyn Device, x: &Tensor, r: usize, k: usize) -> Result<(Tensor, Tensor, Tensor)> {
    if k % 32 != 0 { return Err(Error::Msg(format!("lt_fp4_quant_g_46: k {k} %32!=0"))); }
    let i = |v: i32| v.to_le_bytes().to_vec();
    let l = |v: i64| v.to_le_bytes().to_vec();
    let kb = k / 16; let kb4 = kb.div_ceil(4);
    let sf_bytes = r.div_ceil(128) * 512 * kb4;
    let amax = Tensor::new(dev.alloc_zeroed(4)?, vec![1], DType::U32);
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_amax_slab",
        &[(x.buffer.as_ref(), x.offset), (amax.buffer.as_ref(), 0)],
        &[l((r*k) as i64), i(1), i(0), 0f32.to_le_bytes().to_vec()],
        [((r*k) as u32).div_ceil(256*64).max(1),1,1],[256,1,1],0,false)?;
    let gs = Tensor::new(dev.alloc(8)?, vec![2], DType::F32);
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_gscale",
        &[(amax.buffer.as_ref(),0),(gs.buffer.as_ref(),0)], &[], [1,1,1],[1,1,1],0,false)?;
    let pack = Tensor::new(dev.alloc((r*k)/2)?, vec![(r*k)/2], DType::U32);
    let sf = Tensor::new(dev.alloc_zeroed(sf_bytes)?, vec![sf_bytes], DType::U32);
    let nb = (r*kb) as u32;
    let kname = if x.dtype == DType::F32 { "lt_fp4_quant_gs_f32" } else { "lt_fp4_quant_gs_46" };
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", kname,
        &[(x.buffer.as_ref(), x.offset), (pack.buffer.as_ref(),0), (sf.buffer.as_ref(),0), (gs.buffer.as_ref(),0)],
        &[i(r as i32), i(k as i32)],
        [nb.div_ceil(256),1,1],[256,1,1],0,false)?;
    Ok((pack, sf, gs))
}

pub fn lt_fp4_quant_g(dev: &dyn Device, x: &Tensor, r: usize, k: usize) -> Result<(Tensor, Tensor, Tensor)> {
    if k % 32 != 0 { return Err(Error::Msg(format!("lt_fp4_quant_g: k {k} %32!=0"))); }
    let i = |v: i32| v.to_le_bytes().to_vec();
    let l = |v: i64| v.to_le_bytes().to_vec();
    let kb = k / 16; let kb4 = kb.div_ceil(4);
    let sf_bytes = r.div_ceil(128) * 512 * kb4;
    // per-tensor global = amax over [r,k] -> nvfp4_gscale (gs[0]=amax/(6*448)).
    let amax = Tensor::new(dev.alloc_zeroed(4)?, vec![1], DType::U32);
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_amax_slab",
        &[(x.buffer.as_ref(), x.offset), (amax.buffer.as_ref(), 0)],
        &[l((r*k) as i64), i(1), i(0), 0f32.to_le_bytes().to_vec()],
        [((r*k) as u32).div_ceil(256*64).max(1),1,1],[256,1,1],0,false)?;
    let gs = Tensor::new(dev.alloc(8)?, vec![2], DType::F32);
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_gscale",
        &[(amax.buffer.as_ref(),0),(gs.buffer.as_ref(),0)], &[], [1,1,1],[1,1,1],0,false)?;
    let pack = Tensor::new(dev.alloc((r*k)/2)?, vec![(r*k)/2], DType::U32);
    let sf = Tensor::new(dev.alloc_zeroed(sf_bytes)?, vec![sf_bytes], DType::U32);
    let nb = (r*kb) as u32;
    let kname = if x.dtype == DType::F32 { "lt_fp4_quant_gs_f32" } else { "lt_fp4_quant_gs" };
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", kname,
        &[(x.buffer.as_ref(), x.offset), (pack.buffer.as_ref(),0), (sf.buffer.as_ref(),0), (gs.buffer.as_ref(),0)],
        &[i(r as i32), i(k as i32)],
        [nb.div_ceil(256),1,1],[256,1,1],0,false)?;
    Ok((pack, sf, gs))
}

/// No-global FP4 quant: per-16 ue4m3 block scales absorb the FULL magnitude
/// (global gs=[1,1]), so the dequant stays ~true-scale and the f16-output GEMM
/// (out_f32=false) does NOT overflow, fast f16 path AND correct (vs the
/// two-level global variant whose globally-normalized raw blows past f16 max).
pub fn lt_fp4_quant_g_ng(dev: &dyn Device, x: &Tensor, r: usize, k: usize) -> Result<(Tensor, Tensor, Tensor)> {
    if k % 32 != 0 { return Err(Error::Msg(format!("lt_fp4_quant_g_ng: k {k} %32!=0"))); }
    let i = |v: i32| v.to_le_bytes().to_vec();
    let kb = k / 16; let kb4 = kb.div_ceil(4);
    let sf_bytes = r.div_ceil(128) * 512 * kb4;
    let one_bytes: Vec<u8> = [1f32, 1f32].iter().flat_map(|f| f.to_le_bytes()).collect();
    let gs = Tensor::new(dev.upload(&one_bytes)?, vec![2], DType::F32); // gs=[1, 1]: no global
    let pack = Tensor::new(dev.alloc((r*k)/2)?, vec![(r*k)/2], DType::U32);
    let sf = Tensor::new(dev.alloc_zeroed(sf_bytes)?, vec![sf_bytes], DType::U32);
    let nb = (r*kb) as u32;
    let kname = if x.dtype == DType::F32 { "lt_fp4_quant_gs_f32" } else { "lt_fp4_quant_gs" };
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", kname,
        &[(x.buffer.as_ref(), x.offset), (pack.buffer.as_ref(),0), (sf.buffer.as_ref(),0), (gs.buffer.as_ref(),0)],
        &[i(r as i32), i(k as i32)],
        [nb.div_ceil(256),1,1],[256,1,1],0,false)?;
    Ok((pack, sf, gs))
}

/// dscale[0] = act_global * weight_global (device, one f32) for the FP4 D-scale fold.
pub fn fp4_dscale(dev: &dyn Device, ga: &Tensor, gw: &Tensor) -> Result<Tensor> {
    let d = Tensor::new(dev.alloc(4)?, vec![1], DType::F32);
    dev.dispatch_raw_cuda(MOE_FP4_SRC, "moe_fp4.cu", "fp4_dscale",
        &[(ga.buffer.as_ref(), ga.offset), (gw.buffer.as_ref(), gw.offset), (d.buffer.as_ref(),0)],
        &[], [1,1,1],[1,1,1],0,false)?;
    Ok(d)
}

/// Quantize+fragment-pack MoE expert weights to NVFP4 on device.
/// `w_f16`: f16 `[n_exp*n, k]` row-major (concatenated experts). Returns
/// `(wp, wsc, gw)`, fragment codes, scale words, per-expert globals (device).
pub fn moe_fp4_pack_weights(
    dev: &dyn Device, w_f16: &Tensor, n_exp: usize, n: usize, k: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    if k % 64 != 0 { return Err(Error::Msg(format!("moe_fp4_pack_weights: K {k} %64!=0"))); }
    if n % 8 != 0 { return Err(Error::Msg(format!("moe_fp4_pack_weights: N {n} %8!=0"))); }
    let (kt, ntl) = (k / 64, n / 8);
    let amax = Tensor::new(dev.alloc_zeroed(n_exp * 4).map_err(|e| Error::Msg(format!("{e:?}")))?, vec![n_exp], DType::U32);
    let gw = Tensor::empty(dev, vec![n_exp], DType::F32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    let l = |x: i64| x.to_le_bytes().to_vec();
    let slab = (n * k) as i64;
    let total_elems = (n_exp * n * k) as u32;
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_amax_slab",
        &[(w_f16.buffer.as_ref(), w_f16.offset), (amax.buffer.as_ref(), 0)],
        &[l(slab), i(n_exp as i32), i(0), 0f32.to_le_bytes().to_vec()],
        [total_elems.div_ceil(256 * 8), 1, 1], [256, 1, 1], 0, false)?;
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_globals",
        &[(amax.buffer.as_ref(), 0), (gw.buffer.as_ref(), 0)],
        &[i(n_exp as i32)],
        [(n_exp as u32).div_ceil(64), 1, 1], [64, 1, 1], 0, false)?;
    let wp = Tensor::empty(dev, vec![n_exp * ntl * kt * 32 * 2], DType::U32)?;
    let wsc = Tensor::empty(dev, vec![n_exp * ntl * kt * 8], DType::U32)?;
    let warps = (n_exp * ntl * kt) as u32;
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_pack_w",
        &[(w_f16.buffer.as_ref(), w_f16.offset), (gw.buffer.as_ref(), 0),
          (wp.buffer.as_ref(), 0), (wsc.buffer.as_ref(), 0)],
        &[i(n_exp as i32), i(n as i32), i(k as i32)],
        [(warps * 32).div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
    Ok((wp, wsc, gw))
}

/// Fully-on-device NVFP4 grouped MoE GEMM (W4A4): quantizes the f16 sorted
/// activations `a [mt,k]` to NVFP4 (dynamic per-call global), packs fragments,
/// and runs the Blackwell block-scaled mma over BM=16 tile descriptors built
/// from `offsets` (device `[n_exp+1]`). Fixed launch, graph-safe, no host
/// syncs. Returns `out [mt, n]` f16.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn moe_fp4_grouped_mma_dev(
    dev: &dyn Device, a: &Tensor, wp: &Tensor, wsc: &Tensor, gw: &Tensor,
    offsets: &Tensor, n_exp: usize, mt: usize, n: usize, k: usize,
    act_mode: i32, act_scale: f32,
    amax_in: Option<&Tensor>, amax_out: Option<(&Tensor, i32, f32)>,
    gather_idx: Option<&Tensor>,
) -> Result<Tensor> {
    if k % 64 != 0 { return Err(Error::Msg(format!("moe_fp4_grouped_mma_dev: K {k} %64!=0"))); }
    let out = Tensor::empty(dev, vec![mt.max(1), n], DType::F16)?;
    if mt == 0 { return Ok(out); }
    let (kt, ntl) = (k / 64, n / 8);
    let maxt = mt.div_ceil(16) + n_exp;
    let t0d = Tensor::empty(dev, vec![maxt], DType::U32)?;
    let eidd = Tensor::empty(dev, vec![maxt], DType::U32)?;
    let ged = Tensor::empty(dev, vec![maxt], DType::U32)?;
    let i = |x: i32| x.to_le_bytes().to_vec();
    let l = |x: i64| x.to_le_bytes().to_vec();
    dev.dispatch_raw_cuda(
        MOE_BUILD_TILES_SRC, "moe_build_tiles.cu", "moe_build_tiles",
        &[(offsets.buffer.as_ref(), offsets.offset), (t0d.buffer.as_ref(), 0),
          (eidd.buffer.as_ref(), 0), (ged.buffer.as_ref(), 0)],
        &[i(n_exp as i32), i(16), i(maxt as i32)],
        [1, 1, 1], [1, 1, 1], 0, false)?;
    // dynamic activation global: use the producer-GEMM's fused amax when
    // provided (skips a full re-read of the activation tensor).
    let amax_owned;
    let amax: &Tensor = match amax_in {
        Some(t) => t,
        None => {
            amax_owned = Tensor::new(dev.alloc_zeroed(4).map_err(|e| Error::Msg(format!("{e:?}")))?, vec![1], DType::U32);
            dev.dispatch_raw_cuda(
                MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_amax_slab",
                &[(a.buffer.as_ref(), a.offset), (amax_owned.buffer.as_ref(), 0)],
                &[l(a.elem_count() as i64), i(1), i(act_mode), act_scale.to_le_bytes().to_vec()],
                [(a.elem_count() as u32).div_ceil(256 * 64).max(1), 1, 1], [256, 1, 1], 0, false)?;
            &amax_owned
        }
    };
    let ga = Tensor::empty(dev, vec![1], DType::F32)?;
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "nvfp4_globals",
        &[(amax.buffer.as_ref(), 0), (ga.buffer.as_ref(), 0)],
        &[i(1)],
        [1, 1, 1], [64, 1, 1], 0, false)?;
    // act quant+pack
    let ap = Tensor::empty(dev, vec![maxt * kt * 32 * 4], DType::U32)?;
    let asc = Tensor::empty(dev, vec![maxt * kt * 32], DType::U32)?;
    let warps = (maxt * kt) as u32;
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "moe_fp4_actq_pack",
        &[(a.buffer.as_ref(), a.offset), (t0d.buffer.as_ref(), 0), (ged.buffer.as_ref(), 0),
          (ga.buffer.as_ref(), 0), (ap.buffer.as_ref(), 0), (asc.buffer.as_ref(), 0),
          match gather_idx { Some(t) => (t.buffer.as_ref(), t.offset), None => (t0d.buffer.as_ref(), 0) }],
        &[i(maxt as i32), i(k as i32), i(mt as i32), i(act_mode), act_scale.to_le_bytes().to_vec(),
          i(if gather_idx.is_some() { 1 } else { 0 })],
        [(warps * 32).div_ceil(256), 1, 1], [256, 1, 1], 0, false)?;
    // grouped mma: grid [maxt, NT/8] × 8 warps/block
    // x4 variant: 4 n8-tiles per warp (A frags register-reused), +49% on
    // GB10 vs 1 nt/warp (L2-poorer than desktop Blackwell, where x1 wins).
    let warps_per_block = 8u32;
    let ng = ntl.div_ceil(4) as u32;
    dev.dispatch_raw_cuda(
        MOE_FP4_SRC, "moe_fp4.cu", "moe_fp4_grouped_mma_x4",
        &[(ap.buffer.as_ref(), 0), (asc.buffer.as_ref(), 0),
          (wp.buffer.as_ref(), wp.offset), (wsc.buffer.as_ref(), wsc.offset),
          (t0d.buffer.as_ref(), 0), (eidd.buffer.as_ref(), 0), (ged.buffer.as_ref(), 0),
          (ga.buffer.as_ref(), 0), (gw.buffer.as_ref(), gw.offset),
          (out.buffer.as_ref(), 0),
          match &amax_out { Some((t, _, _)) => (t.buffer.as_ref(), t.offset), None => (ga.buffer.as_ref(), 0) }],
        &[i(n as i32), i(k as i32), i(mt as i32),
          i(amax_out.as_ref().map(|(_, m, _)| *m).unwrap_or(-1)),
          amax_out.as_ref().map(|(_, _, sc)| *sc).unwrap_or(0.0).to_le_bytes().to_vec()],
        [maxt as u32, ng.div_ceil(warps_per_block), 1],
        [warps_per_block * 32, 1, 1], 0, false)?;
    Ok(out)
}

/// Fused residual-add + RMS norm: `residual_out = a + b`, `normed_out =
/// rms_norm(a + b, w)`, one pass instead of an `add` kernel plus a
/// `rms_norm` kernel re-reading the sum (the per-layer residual hot path).
pub fn add_rms_norm(
    dev: &dyn Device, a: &Tensor, b: &Tensor, weight: &Tensor, eps: f32,
) -> Result<(Tensor, Tensor)> {
    let n = *a.shape.last().ok_or_else(|| Error::Msg("add_rms_norm: scalar input".into()))?;
    if n % 4 != 0 || n > 4096 * 4 {
        return Err(Error::Msg(format!("add_rms_norm: n {n} unsupported (need %4, fits one block)")));
    }
    let rows = a.elem_count() / n;
    let residual = Tensor::empty(dev, a.shape.clone(), a.dtype)?;
    let normed = Tensor::empty(dev, a.shape.clone(), a.dtype)?;
    let eps_buf = scalar_buf(dev, eps.to_bits())?;
    // Reduction mode: one threadgroup per row, `tid` = lane within the row
    // (the same TG=n/4 contract as iron_rms_norm).
    let k = cached_ir("iron_add_rms_norm", a.dtype, || {
        let mut k = wh_iron_std::kernels::ops::unary::iron_add_rms_norm::kernel_ir_for(a.dtype);
        k.mode = wh_iron_core::ir::KernelMode::Reduction;
        k
    });
    let grid = Grid { grid: [rows as u32, 1, 1], block: [(n / 4) as u32, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(a.buffer.clone()),
            Binding::Buffer(b.buffer.clone()),
            Binding::Buffer(weight.buffer.clone()),
            Binding::Buffer(eps_buf),
            Binding::Buffer(residual.buffer.clone()),
            Binding::Buffer(normed.buffer.clone()),
            Binding::Scalar((n as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok((residual, normed))
}

#[cfg(test)]
mod marlin_prep_validate {
    use super::*;
    use std::fs;
    fn rf32(p:&str)->Vec<f32>{let b=fs::read(p).unwrap(); b.chunks(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()}
    #[test]
    fn marlin_weight_prep_matches_python(){
        // Reference fixtures come from env, never a hardcoded path; unset -> quiet skip.
        let Ok(dir)=std::env::var("BUTTER_MARLIN_FIXTURE_DIR") else {
            println!("[skipped] BUTTER_MARLIN_FIXTURE_DIR unset");
            return;
        };
        let (n,k)=(1856usize,2688usize); let ng=k/16; let kp=k/8;
        let w0=rf32(&format!("{dir}/W0.bin")); assert_eq!(w0.len(), n*k);
        let (codes,uesc,global)=quantize_nvfp4(&w0,n,k);
        // codes[N,kp] -> transpose [kp,N]
        let mut qw=vec![0u32; kp*n];
        for r in 0..n { for c in 0..kp { qw[c*n+r]=codes[r*kp+c]; } }
        let qref=fs::read(format!("{dir}/qweight.bin")).unwrap();
        let qref0:Vec<u32>=qref[..kp*n*4].chunks(4).map(|c|u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let qm=qw.iter().zip(&qref0).filter(|(a,b)|a==b).count();
        // scales: dec_ue4m3[N,ng] -> values, transpose [ng,N], s0e5m3
        let mut sv=vec![0f32; ng*n];
        for r in 0..n { for g in 0..ng { sv[g*n+r]=dec_ue4m3(uesc[r*ng+g]); } }
        let s0=marlin_s0e5m3_scales(&sv,ng,n);
        let bref=fs::read(format!("{dir}/bscales.bin")).unwrap();
        let bm=s0.iter().zip(&bref[..ng*n]).filter(|(a,b)|a==b).count();
        let gref=rf32(&format!("{dir}/global.bin"));
        let grust=global*128.0;
        eprintln!("QW match {}/{}  BSCALE match {}/{}  global rust {} py {} (ratio {})", qm, kp*n, bm, ng*n, grust, gref[0], grust/gref[0]);
        assert!(qm==kp*n, "qweight mismatch");
        assert!(bm>=(ng*n*999/1000), "bscale mismatch");
        assert!((grust/gref[0]-1.0).abs()<0.02, "global mismatch");
    }
}
