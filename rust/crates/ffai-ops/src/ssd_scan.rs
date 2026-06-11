// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # Mamba2 SSD chunked-MATMUL prefill scan (state-space duality)
//!
//! Replaces the sequential-in-T `ssm_prefill_scan` with the chunk-scan form
//! (Dao & Gu, "Transformers are SSMs") that runs the scan as cuBLAS tensor-core
//! GEMMs. Split T into `nc = ceil(T/L)` chunks of length L; per (chunk, head):
//!
//! ```text
//!   A       = -exp(a_log[h])
//!   Lcs[i]  = Σ_{k≤i} A*dt[k]                (inclusive cumsum within chunk)
//!   Lmask[i,j] = exp(Lcs[i]-Lcs[j])  (i≥j)   else 0
//!
//!   1. CB        = C · Bᵀ                              [L,L]   (G1, batched)
//!      M[i,j]    = CB[i,j]·Lmask[i,j]·dt[j]            (custom kernel)
//!   2. y_intra   = M · x                               [L,dh]  (G2, batched)
//!   3. S_chunk   = (decay·dt·B)ᵀ · x                   [ds,dh] (G3, batched)
//!      decay[j]  = exp(Lcs[L-1]-Lcs[j])
//!   4. recurrence (serial over nc): S_in[c+1]=αc·S_in[c]+S_chunk[c]
//!      αc        = exp(Lcs[L-1]) of chunk c
//!   5. CS        = C · S_in                            [L,dh]  (G4, batched)
//!      y[i]      = y_intra[i] + exp(Lcs[i])·CS[i] + x[i]·D[h]  (custom kernel)
//! ```
//!
//! Steps 1,2,3,5 are `cublasGemmStridedBatchedEx` over batch = nc*H. B/C are
//! shared across heads-per-group (group g = h/hpg) — the build kernels fan the
//! group out to each head's batch slot. FP16 GEMM in / FP32 accumulate; all the
//! decay / segsum / elementwise math stays FP32. Gated by `NEMOTRON_SSD_MATMUL`.
//!
//! Fixed for NemotronH: dh=64, ds=128, H=64, G=8 (hpg=8). L ∈ {128,256}.
//!
//! Lives in its own module (not `lib.rs`) to avoid edit contention with the
//! concurrent tensor-core attention work on the same branch.

use ffai_core::{DType, Device, Error, Result, Tensor};

/// CUDA: inclusive cumsum of A*dt within each chunk → Lcs (f32), one thread per
/// (chunk, head). A = -exp(a_log[h]). dt layout [T, H]. Lcs layout [nc*H, L].
const SSD_LCS_SRC: &str = r#"
extern "C" __global__ void ssd_lcs(
    const float* __restrict__ dt,     // [T, H]
    const float* __restrict__ a_log,  // [H]
    float*       __restrict__ lcs,    // [nc*H, L]  inclusive cumsum of A*dt
    unsigned int T, unsigned int H, unsigned int L, unsigned int nc)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // bh = c*H + h
    if (idx >= nc * H) return;
    unsigned int c = idx / H;
    unsigned int h = idx % H;
    float a = -expf(a_log[h]);
    float acc = 0.f;
    for (unsigned int i = 0; i < L; ++i) {
        unsigned int t = c * L + i;
        float dtv = (t < T) ? dt[t * H + h] : 0.f;
        acc += a * dtv;
        lcs[idx * L + i] = acc;
    }
}
"#;

/// CUDA: gather/cast C,B from [T,G,ds] into [nc*H, L, ds] (head h uses group
/// h/hpg), broadcasting the group operand into every head's batch slot. f16.
const SSD_GATHER_BC_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_gather_bc(
    const float* __restrict__ b_mat,  // [T, G, ds]
    const float* __restrict__ c_mat,  // [T, G, ds]
    __half*      __restrict__ b_out,  // [nc*H, L, ds] f16
    __half*      __restrict__ c_out,  // [nc*H, L, ds] f16
    unsigned int T, unsigned int H, unsigned int G, unsigned int hpg,
    unsigned int L, unsigned int ds, unsigned int nc)
{
    unsigned long long n = (unsigned long long)nc * H * L * ds;
    unsigned long long e = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    unsigned int s   = (unsigned int)(e % ds);
    unsigned int i   = (unsigned int)((e / ds) % L);
    unsigned int bh  = (unsigned int)(e / ((unsigned long long)ds * L));
    unsigned int c   = bh / H;
    unsigned int h   = bh % H;
    unsigned int g   = h / hpg;
    unsigned int t   = c * L + i;
    float bv = 0.f, cv = 0.f;
    if (t < T) {
        bv = b_mat[(t * G + g) * ds + s];
        cv = c_mat[(t * G + g) * ds + s];
    }
    b_out[e] = __float2half(bv);
    c_out[e] = __float2half(cv);
}
"#;

/// CUDA (FUSED path): gather/cast C,B from [T,G,ds] into PER-GROUP [nc*G, L, ds]
/// f16 — NO 8× head broadcast (that 8× write is replaced by a device-pointer
/// array in the batched GEMM, where head h's batch slot points at group h/hpg's
/// slice). 8× fewer writes / 8× smaller scratch than `ssd_gather_bc`.
const SSD_GATHER_BC_G_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_gather_bc_g(
    const float* __restrict__ b_mat,  // B base (row-strided; see rs/b_off)
    const float* __restrict__ c_mat,  // C base
    __half*      __restrict__ b_out,  // [nc*G, L, ds] f16  (per group)
    __half*      __restrict__ c_out,  // [nc*G, L, ds] f16  (per group)
    unsigned int T, unsigned int G,
    unsigned int L, unsigned int ds, unsigned int nc,
    unsigned int rs, unsigned int b_off, unsigned int c_off) // row stride + col offsets
{
    unsigned long long n = (unsigned long long)nc * G * L * ds;
    unsigned long long e = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    unsigned int s   = (unsigned int)(e % ds);
    unsigned int i   = (unsigned int)((e / ds) % L);
    unsigned int cg  = (unsigned int)(e / ((unsigned long long)ds * L)); // c*G + g
    unsigned int c   = cg / G;
    unsigned int g   = cg % G;
    unsigned int t   = c * L + i;
    float bv = 0.f, cv = 0.f;
    if (t < T) {
        bv = b_mat[(unsigned long long)t * rs + b_off + g * ds + s];
        cv = c_mat[(unsigned long long)t * rs + c_off + g * ds + s];
    }
    b_out[e] = __float2half(bv);
    c_out[e] = __float2half(cv);
}
"#;

/// CUDA: transpose x [T,H,dh] → xT [nc*H, dh, L] (f16). Head h, chunk c.
const SSD_XT_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_xt(
    const float* __restrict__ x,     // x base (row-strided)
    __half*      __restrict__ xt,    // [nc*H, dh, L]
    unsigned int T, unsigned int H, unsigned int dh, unsigned int L, unsigned int nc,
    unsigned int rs, unsigned int x_off)
{
    unsigned long long n = (unsigned long long)nc * H * dh * L;
    unsigned long long e = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    unsigned int i  = (unsigned int)(e % L);             // position within chunk
    unsigned int p  = (unsigned int)((e / L) % dh);      // head_dim index
    unsigned int bh = (unsigned int)(e / ((unsigned long long)L * dh));
    unsigned int c  = bh / H;
    unsigned int h  = bh % H;
    unsigned int t  = c * L + i;
    float v = (t < T) ? x[(unsigned long long)t * rs + x_off + h * dh + p] : 0.f;
    xt[e] = __float2half(v);
}
"#;

/// CUDA: M[i,j] = CB[i,j]·exp(Lcs[i]-Lcs[j])·dt[j], causal (i≥j) else 0. f16.
/// CB is the f16 output of G1 ([nc*H, L, L]). dt layout [T,H]. One thread/elem.
const SSD_MMASK_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_mmask(
    const __half* __restrict__ cb,    // [nc*H, L, L] = C·Bᵀ
    const float*  __restrict__ lcs,   // [nc*H, L]
    const float*  __restrict__ dt,    // [T, H]
    __half*       __restrict__ m_out, // [nc*H, L, L] f16
    unsigned int T, unsigned int H, unsigned int L, unsigned int nc)
{
    unsigned long long n = (unsigned long long)nc * H * L * L;
    unsigned long long e = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    unsigned int j  = (unsigned int)(e % L);
    unsigned int i  = (unsigned int)((e / L) % L);
    unsigned int bh = (unsigned int)(e / ((unsigned long long)L * L));
    if (i < j) { m_out[e] = __float2half(0.f); return; }
    unsigned int c  = bh / H;
    unsigned int h  = bh % H;
    unsigned int tj = c * L + j;
    float dtj = (tj < T) ? dt[tj * H + h] : 0.f;
    float decay = expf(lcs[bh * L + i] - lcs[bh * L + j]);
    float v = __half2float(cb[e]) * decay * dtj;
    m_out[e] = __float2half(v);
}
"#;

/// CUDA: build BdT[s,j] = exp(Lcs[L-1]-Lcs[j])·dt[j]·B[j,s] → [nc*H, ds, L] f16.
/// (the decayed, dt-weighted, transposed B for chunk-state G3.) B is [T,G,ds].
const SSD_BDT_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_bdt(
    const float* __restrict__ b_mat, // B base (row-strided)
    const float* __restrict__ lcs,   // [nc*H, L]
    const float* __restrict__ dt,    // [T, H]
    __half*      __restrict__ bdt,   // [nc*H, ds, L]
    unsigned int T, unsigned int H, unsigned int G, unsigned int hpg,
    unsigned int L, unsigned int ds, unsigned int nc,
    unsigned int rs, unsigned int b_off)
{
    unsigned long long n = (unsigned long long)nc * H * ds * L;
    unsigned long long e = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    unsigned int j  = (unsigned int)(e % L);            // position
    unsigned int s  = (unsigned int)((e / L) % ds);     // state index
    unsigned int bh = (unsigned int)(e / ((unsigned long long)L * ds));
    unsigned int c  = bh / H;
    unsigned int h  = bh % H;
    unsigned int g  = h / hpg;
    unsigned int tj = c * L + j;
    float bv = 0.f, dtj = 0.f;
    if (tj < T) { bv = b_mat[(unsigned long long)tj * rs + b_off + g * ds + s]; dtj = dt[tj * H + h]; }
    float decay = expf(lcs[bh * L + (L - 1)] - lcs[bh * L + j]);
    bdt[e] = __float2half(decay * dtj * bv);
}
"#;

/// CUDA: serial inter-chunk recurrence. For each (head, state s, dh p):
///   S_in[c] = state at START of chunk c; S_in[0] = state_in[h,p,s]
///   S_in[c+1] = αc·S_in[c] + S_chunk[c],  αc = exp(Lcs[bh,L-1])
/// Emits SinT[bh] = S_in[c]ᵀ as [nc*H, dh, ds] (transposed for G4), all f16,
/// and writes the FINAL state (after chunk nc-1) to state_out [H, dh, ds] f32.
/// Grid: one thread per (head, s, p) = H*ds*dh threads; loops nc serially.
const SSD_RECUR_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_recur(
    const __half* __restrict__ s_chunk,  // [nc*H, ds, dh]
    const float*  __restrict__ lcs,      // [nc*H, L]
    const float*  __restrict__ state_in, // [H, dh, ds] f32
    __half*       __restrict__ sin_t,    // [nc*H, dh, ds] f16  (S_inᵀ per chunk)
    float*        __restrict__ state_out,// [H, dh, ds] f32
    unsigned int H, unsigned int dh, unsigned int ds, unsigned int L, unsigned int nc)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long tot = (unsigned long long)H * ds * dh;
    if (idx >= tot) return;
    unsigned int p  = (unsigned int)(idx % dh);            // head_dim
    unsigned int s  = (unsigned int)((idx / dh) % ds);     // state
    unsigned int h  = (unsigned int)(idx / ((unsigned long long)dh * ds));
    // initial state for chunk 0
    float st = state_in[(h * dh + p) * ds + s];
    for (unsigned int c = 0; c < nc; ++c) {
        unsigned int bh = c * H + h;
        // emit S_inᵀ for this chunk BEFORE applying it: SinT[bh, p, s] = st
        sin_t[(bh * dh + p) * ds + s] = __float2half(st);
        float alpha = expf(lcs[bh * L + (L - 1)]);
        float sc = __half2float(s_chunk[(bh * ds + s) * dh + p]);
        st = alpha * st + sc;
    }
    state_out[(h * dh + p) * ds + s] = st;
}
"#;

/// Transposed-coalesced inter-chunk recurrence: consumes S_chunkᵀ
/// `[nc*H, dh, ds]` (G3 emitted with swapped operands) with an s-fastest
/// thread map, so BOTH the chunk-state reads and the SinT/state writes are
/// fully coalesced — the original layout scattered one side 32-way across
/// the serial nc loop. Same math, same per-thread accumulation order.
const SSD_RECUR_T_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_recur_t(
    const __half* __restrict__ s_chunk_t, // [nc*H, dh, ds]
    const float*  __restrict__ lcs,       // [nc*H, L]
    const float*  __restrict__ state_in,  // [H, dh, ds] f32
    __half*       __restrict__ sin_t,     // [nc*H, dh, ds] f16
    float*        __restrict__ state_out, // [H, dh, ds] f32
    unsigned int H, unsigned int dh, unsigned int ds, unsigned int L, unsigned int nc)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long tot = (unsigned long long)H * ds * dh;
    if (idx >= tot) return;
    unsigned int sct = (unsigned int)(idx % ds);          // state (fastest)
    unsigned int p   = (unsigned int)((idx / ds) % dh);   // head_dim
    unsigned int h   = (unsigned int)(idx / ((unsigned long long)ds * dh));
    float st = state_in[(h * dh + p) * ds + sct];
    for (unsigned int c = 0; c < nc; ++c) {
        unsigned int bh = c * H + h;
        sin_t[(bh * dh + p) * ds + sct] = __float2half(st);
        float alpha = expf(lcs[bh * L + (L - 1)]);
        float sc = __half2float(s_chunk_t[((unsigned long long)bh * dh + p) * ds + sct]);
        st = alpha * st + sc;
    }
    state_out[(h * dh + p) * ds + sct] = st;
}
"#;

/// Varlen inter-chunk recurrence. Identical to `ssd_recur` but resets the
/// carried state to zero at every chunk flagged in `seg_reset[c]` — i.e. the
/// first chunk of each packed sequence starts fresh. Requires each packed
/// segment length to be a multiple of `L` (chunk_len) so no chunk straddles a
/// boundary. With an all-zero `seg_reset` this is bit-identical to `ssd_recur`.
const SSD_RECUR_VARLEN_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_recur_varlen(
    const __half* __restrict__ s_chunk,  // [nc*H, ds, dh]
    const float*  __restrict__ lcs,      // [nc*H, L]
    const float*  __restrict__ state_in, // [H, dh, ds] f32
    const unsigned int* __restrict__ seg_reset, // [nc] 1 → chunk starts a new sequence
    __half*       __restrict__ sin_t,    // [nc*H, dh, ds] f16
    float*        __restrict__ state_out,// [H, dh, ds] f32
    unsigned int H, unsigned int dh, unsigned int ds, unsigned int L, unsigned int nc)
{
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long tot = (unsigned long long)H * ds * dh;
    if (idx >= tot) return;
    unsigned int p  = (unsigned int)(idx % dh);
    unsigned int s  = (unsigned int)((idx / dh) % ds);
    unsigned int h  = (unsigned int)(idx / ((unsigned long long)dh * ds));
    float st = state_in[(h * dh + p) * ds + s];
    for (unsigned int c = 0; c < nc; ++c) {
        if (seg_reset[c]) st = 0.0f;       // packed boundary → fresh recurrent state
        unsigned int bh = c * H + h;
        sin_t[(bh * dh + p) * ds + s] = __float2half(st);
        float alpha = expf(lcs[bh * L + (L - 1)]);
        float sc = __half2float(s_chunk[(bh * ds + s) * dh + p]);
        st = alpha * st + sc;
    }
    state_out[(h * dh + p) * ds + s] = st;
}
"#;

/// CUDA: final combine. y[t,h,p] = y_intra[bh,i,p] + exp(Lcs[bh,i])·CS[bh,i,p]
///   + x[t,h,p]·D[h]. y_intra, CS are f16 [nc*H, L, dh]; output y f32 [T,H,dh].
const SSD_COMBINE_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_combine(
    const __half* __restrict__ y_intra, // [nc*H, L, dh]
    const __half* __restrict__ cs,      // [nc*H, L, dh]
    const float*  __restrict__ lcs,     // [nc*H, L]
    const float*  __restrict__ x,       // x base (row-strided)
    const float*  __restrict__ d_skip,  // [H]
    float*        __restrict__ y,       // [T, H, dh]
    unsigned int T, unsigned int H, unsigned int dh, unsigned int L, unsigned int nc,
    unsigned int rs, unsigned int x_off)
{
    unsigned long long e = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long n = (unsigned long long)nc * H * L * dh;
    if (e >= n) return;
    unsigned int p  = (unsigned int)(e % dh);
    unsigned int i  = (unsigned int)((e / dh) % L);
    unsigned int bh = (unsigned int)(e / ((unsigned long long)L * dh));
    unsigned int c  = bh / H;
    unsigned int h  = bh % H;
    unsigned int t  = c * L + i;
    if (t >= T) return;
    float yi = __half2float(y_intra[e]);
    float decay = expf(lcs[bh * L + i]);
    float ci = __half2float(cs[e]) * decay;
    float xv = x[(unsigned long long)t * rs + x_off + h * dh + p];
    y[(t * H + h) * dh + p] = yi + ci + xv * d_skip[h];
}
"#;

/// Mamba2 SSD **chunked-matmul prefill scan** for the NemotronH cell
/// (dh=64, ds=128, H=64, G=8). Runs the scan as cuBLAS tensor-core GEMMs
/// (state-space duality). Returns `(state_out, y)` — same signature/layout as
/// `ssm_prefill_scan`: `y` is `[T·H·dh]` f32, `state_out` matches `state_in`.
/// Inputs are f32 (cast internally to f16 for GEMMs). `t_total` need not be a
/// multiple of L — the tail chunk is zero-padded. `chunk_len` is typically 128
/// or 256. Gated by `NEMOTRON_SSD_MATMUL` at the call site.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_scan_ssd_strided(
    dev: &dyn Device,
    // ONE row-strided base holding [x | B | C] per token row (e.g. the conv
    // output read in place — no mamba_split_conv materialization). For the
    // compact 3-tensor form pass the same tensors with rs = their natural
    // widths via `ssm_prefill_scan_ssd`.
    x: &Tensor,
    x_rs: u32, x_off: u32,
    b_mat: &Tensor,
    c_mat: &Tensor,
    bc_rs: u32, b_off: u32, c_off: u32,
    a_log: &Tensor,
    d_skip: &Tensor,
    dt: &Tensor,
    state_in: &Tensor,
    t_total: u32,
    dh: u32,
    ds: u32,
    n_heads: u32,
    n_groups: u32,
    chunk_len: u32,
    // Varlen packed-prefill: `[nc]` u32 buffer, 1 where a chunk starts a new
    // packed sequence (resets the recurrent state). `None` = single sequence
    // (dense path, bit-identical to before).
    seg_reset: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    if (dh, ds, n_heads, n_groups) != (64, 128, 64, 8) {
        return Err(Error::Msg(format!(
            "ssm_prefill_scan_ssd: only Nemotron cell (64,128,64,8) wired, got ({dh},{ds},{n_heads},{n_groups})"
        )));
    }
    if x.dtype != DType::F32 {
        return Err(Error::Msg("ssm_prefill_scan_ssd: only F32 input wired (cast first)".into()));
    }
    let (t, h, dhu, dsu, g) = (
        t_total as usize, n_heads as usize, dh as usize, ds as usize, n_groups as usize,
    );
    let hpg = h / g; // heads per group
    let l = chunk_len as usize;
    let nc = t.div_ceil(l); // number of chunks (tail zero-padded)
    let bhc = nc * h;       // batch count for the GEMMs
    let bgc = nc * g;       // per-GROUP batch count (fused path: 8× smaller B/C)

    // FUSED path (DEFAULT-ON; escape NEMOTRON_SSD_FUSED_OFF=1): materialize B/C
    // only PER-GROUP [nc*G, L, ds] (8× smaller / 8× fewer writes than the
    // broadcast gather), and fan the head→group slice into the G1/G4 GEMMs via a
    // per-batch DEVICE POINTER ARRAY (cublasGemmBatchedEx) instead of the 8×
    // redundant write. cuBLAS tensor cores preserved; argmax bit-identical to the
    // strided path (same f16 GEMM inputs, just read from the shared group slice).
    // (NEMOTRON_SSD_FUSED still force-enables it for explicit A/B.)
    let fused = std::env::var("NEMOTRON_SSD_FUSED_OFF").is_err();

    // ── Output + final state (f32, matching the sequential scan). ─────────
    let y = Tensor::empty(dev, vec![t * h * dhu], DType::F32)?;
    let state_out = Tensor::empty(dev, state_in.shape.clone(), DType::F32)?;

    // ── Scratch buffers. ──────────────────────────────────────────────────
    // b/c sized per-GROUP in the fused path (bgc), per-HEAD otherwise (bhc).
    let bc_batch = if fused { bgc } else { bhc };
    let lcs = Tensor::empty(dev, vec![bhc * l], DType::F32)?;            // [nc*H, L]
    let b_f16 = Tensor::empty(dev, vec![bc_batch * l * dsu], DType::F16)?; // [nc*{G|H}, L, ds]
    let c_f16 = Tensor::empty(dev, vec![bc_batch * l * dsu], DType::F16)?; // [nc*{G|H}, L, ds]
    let xt = Tensor::empty(dev, vec![bhc * dhu * l], DType::F16)?;       // [nc*H, dh, L]
    let cb = Tensor::empty(dev, vec![bhc * l * l], DType::F16)?;         // [nc*H, L, L]
    let mmask = Tensor::empty(dev, vec![bhc * l * l], DType::F16)?;      // [nc*H, L, L]
    let y_intra = Tensor::empty(dev, vec![bhc * l * dhu], DType::F16)?;  // [nc*H, L, dh]
    let bdt = Tensor::empty(dev, vec![bhc * dsu * l], DType::F16)?;      // [nc*H, ds, L]
    let s_chunk = Tensor::empty(dev, vec![bhc * dsu * dhu], DType::F16)?;// [nc*H, ds, dh]
    let sin_t = Tensor::empty(dev, vec![bhc * dhu * dsu], DType::F16)?;  // [nc*H, dh, ds]
    let cs = Tensor::empty(dev, vec![bhc * l * dhu], DType::F16)?;       // [nc*H, L, dh]

    let raw = |src: &str, name: &str, fnn: &str,
               ptrs: &[(&dyn ffai_core::DeviceBuffer, usize)],
               scalars: &[u32], n_threads: usize| -> Result<()> {
        let block = 256u32;
        let grid = [((n_threads as u32) + block - 1) / block, 1, 1];
        let sc: Vec<Vec<u8>> = scalars.iter().map(|v| v.to_le_bytes().to_vec()).collect();
        dev.dispatch_raw_cuda(src, name, fnn, ptrs, &sc, grid, [block, 1, 1], 0, false)
    };
    fn bb(t: &Tensor) -> &dyn ffai_core::DeviceBuffer { t.buffer.as_ref() }
    let (tt, hh, gg, ll, dsd, dhd, ncn, hpgn) = (
        t_total, n_heads, n_groups, l as u32, ds, dh, nc as u32, hpg as u32,
    );

    // 1. Lcs cumsum.
    raw(SSD_LCS_SRC, "ssd_lcs.cu", "ssd_lcs",
        &[(bb(dt), 0), (bb(a_log), 0), (bb(&lcs), 0)],
        &[tt, hh, ll, ncn], bhc)?;
    // 2. Gather B/C → f16. Fused: per-GROUP [nc*G,L,ds] (no 8× broadcast).
    if fused {
        raw(SSD_GATHER_BC_G_SRC, "ssd_gather_bc_g.cu", "ssd_gather_bc_g",
            &[(bb(b_mat), 0), (bb(c_mat), 0), (bb(&b_f16), 0), (bb(&c_f16), 0)],
            &[tt, gg, ll, dsd, ncn, bc_rs, b_off, c_off], bgc * l * dsu)?;
    } else {
        if (bc_rs, b_off, c_off) != ((g * dsu) as u32, 0, 0) {
            return Err(Error::Msg("ssm_prefill_scan_ssd: strided B/C requires the fused gather (unset NEMOTRON_SSD_FUSED_OFF)".into()));
        }
        raw(SSD_GATHER_BC_SRC, "ssd_gather_bc.cu", "ssd_gather_bc",
            &[(bb(b_mat), 0), (bb(c_mat), 0), (bb(&b_f16), 0), (bb(&c_f16), 0)],
            &[tt, hh, gg, hpgn, ll, dsd, ncn], bhc * l * dsu)?;
    }
    // 3. Transpose x → xT [nc*H, dh, L] f16.
    raw(SSD_XT_SRC, "ssd_xt.cu", "ssd_xt",
        &[(bb(x), 0), (bb(&xt), 0)],
        &[tt, hh, dhd, ll, ncn, x_rs, x_off], bhc * dhu * l)?;

    // Byte size of one [L,ds] f16 group slice — the broadcast stride for the
    // device-pointer arrays (head h → group h/hpg slice c*G + h/hpg).
    let el = 2i64; // f16 element size (bytes)
    let st_lds = (l * dsu) as i64 * el; // stride of one [L,ds] matrix
    let st_ll = (l * l) as i64 * el;
    // head→group slice byte offset for batch bh = c*H + h.
    let grp_off = |bh: usize| -> usize {
        let c = bh / h;
        let hh_ = bh % h;
        (c * g + hh_ / hpg) * (l * dsu) * (el as usize)
    };

    // ── G1: CB = C · Bᵀ  [L,L] = C[L,ds]·B[L,ds]ᵀ. ─────────────────────────
    // primitive C[m,n]=X[m,k]·W[n,k]ᵀ → m=L, n=L, k=ds, X=c_f16, W=b_f16.
    if fused {
        // Device-ptr-array batched GEMM: X=C, W=B both point at the per-group
        // slice (broadcast), out=cb per-head batch slot. m=L,n=L,k=ds.
        let c_offs: Vec<usize> = (0..bhc).map(grp_off).collect();
        let b_offs: Vec<usize> = c_offs.clone();
        let cb_offs: Vec<usize> = (0..bhc).map(|bh| bh * (l * l) * (el as usize)).collect();
        dev.gemm_batched(
            bb(&c_f16), &c_offs, bb(&b_f16), &b_offs, bb(&cb), &cb_offs,
            l, l, dsu, DType::F16,
        )?;
    } else {
        dev.gemm_strided_batched(
            bb(&c_f16), st_lds, bb(&b_f16), st_lds, bb(&cb), st_ll,
            l, l, dsu, bhc, DType::F16,
        )?;
    }

    // 4. M = CB ⊙ Lmask ⊙ dt[j], causal.
    raw(SSD_MMASK_SRC, "ssd_mmask.cu", "ssd_mmask",
        &[(bb(&cb), 0), (bb(&lcs), 0), (bb(dt), 0), (bb(&mmask), 0)],
        &[tt, hh, ll, ncn], bhc * l * l)?;

    // ── G2: y_intra = M · x  [L,dh]. m=L, n=dh, k=L, X=mmask, W=xt[dh,L]. ───
    let st_ldh = (l * dhu) as i64 * el;
    let st_dhl = (dhu * l) as i64 * el;
    dev.gemm_strided_batched(
        bb(&mmask), st_ll, bb(&xt), st_dhl, bb(&y_intra), st_ldh,
        l, dhu, l, bhc, DType::F16,
    )?;

    // 5. BdT[s,j] = decay·dt·B  [ds,L] f16.
    raw(SSD_BDT_SRC, "ssd_bdt.cu", "ssd_bdt",
        &[(bb(b_mat), 0), (bb(&lcs), 0), (bb(dt), 0), (bb(&bdt), 0)],
        &[tt, hh, gg, hpgn, ll, dsd, ncn, bc_rs, b_off], bhc * dsu * l)?;

    // ── G3: S_chunk = BdT · xᵀ-form  [ds,dh]. m=ds, n=dh, k=L,
    //        X=bdt[ds,L], W=xt[dh,L]. ─────────────────────────────────────
    // NEMOTRON_SSD_RECUR_T=1: emit S_chunkᵀ [dh,ds] instead (operands
    // swapped — same products, GEMM-internal accumulation order may differ)
    // and run the fully-coalesced transposed recurrence. Quant-regime lever;
    // default off to preserve the exact path's bit pattern.
    let recur_t = seg_reset.is_none() && std::env::var("NEMOTRON_SSD_RECUR_T").is_ok();
    let st_dsl = (dsu * l) as i64 * el;
    let st_dsdh = (dsu * dhu) as i64 * el;
    if recur_t {
        dev.gemm_strided_batched(
            bb(&xt), st_dhl, bb(&bdt), st_dsl, bb(&s_chunk), st_dsdh,
            dhu, dsu, l, bhc, DType::F16,
        )?;
    } else {
        dev.gemm_strided_batched(
            bb(&bdt), st_dsl, bb(&xt), st_dhl, bb(&s_chunk), st_dsdh,
            dsu, dhu, l, bhc, DType::F16,
        )?;
    }

    // 6. Serial inter-chunk recurrence → SinT [nc*H, dh, ds] + state_out.
    //    Varlen: reset state at packed-sequence boundaries (seg_reset[c]).
    if recur_t {
        raw(SSD_RECUR_T_SRC, "ssd_recur_t.cu", "ssd_recur_t",
            &[(bb(&s_chunk), 0), (bb(&lcs), 0), (bb(state_in), 0), (bb(&sin_t), 0), (bb(&state_out), 0)],
            &[hh, dhd, dsd, ll, ncn], h * dsu * dhu)?;
    } else {
    match seg_reset {
        Some(sr) => raw(SSD_RECUR_VARLEN_SRC, "ssd_recur_varlen.cu", "ssd_recur_varlen",
            &[(bb(&s_chunk), 0), (bb(&lcs), 0), (bb(state_in), 0), (bb(sr), 0), (bb(&sin_t), 0), (bb(&state_out), 0)],
            &[hh, dhd, dsd, ll, ncn], h * dsu * dhu)?,
        None => raw(SSD_RECUR_SRC, "ssd_recur.cu", "ssd_recur",
            &[(bb(&s_chunk), 0), (bb(&lcs), 0), (bb(state_in), 0), (bb(&sin_t), 0), (bb(&state_out), 0)],
            &[hh, dhd, dsd, ll, ncn], h * dsu * dhu)?,
    };
    }

    // ── G4: CS = C · S_in  [L,dh]. m=L, n=dh, k=ds, X=c_f16, W=sinT[dh,ds]. ─
    let st_dhds = (dhu * dsu) as i64 * el;
    if fused {
        // X=C reads the per-group slice (broadcast via grp_off); W=sin_t and
        // out=cs stay per-head batch slots. m=L, n=dh, k=ds.
        let c_offs: Vec<usize> = (0..bhc).map(grp_off).collect();
        let sin_offs: Vec<usize> = (0..bhc).map(|bh| bh * (dhu * dsu) * (el as usize)).collect();
        let cs_offs: Vec<usize> = (0..bhc).map(|bh| bh * (l * dhu) * (el as usize)).collect();
        dev.gemm_batched(
            bb(&c_f16), &c_offs, bb(&sin_t), &sin_offs, bb(&cs), &cs_offs,
            l, dhu, dsu, DType::F16,
        )?;
    } else {
        dev.gemm_strided_batched(
            bb(&c_f16), st_lds, bb(&sin_t), st_dhds, bb(&cs), st_ldh,
            l, dhu, dsu, bhc, DType::F16,
        )?;
    }

    // 7. Combine → y [T,H,dh] f32.
    raw(SSD_COMBINE_SRC, "ssd_combine.cu", "ssd_combine",
        &[(bb(&y_intra), 0), (bb(&cs), 0), (bb(&lcs), 0), (bb(x), 0), (bb(d_skip), 0), (bb(&y), 0)],
        &[tt, hh, dhd, ll, ncn, x_rs, x_off], bhc * l * dhu)?;

    Ok((state_out, y))
}

/// Compact 3-tensor form of [`ssm_prefill_scan_ssd_strided`] — original
/// signature preserved for existing callers (x [T,H·dh], B/C [T,G·ds]).
#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_scan_ssd(
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
    chunk_len: u32,
    seg_reset: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    ssm_prefill_scan_ssd_strided(
        dev,
        x, n_heads * dh, 0,
        b_mat, c_mat, n_groups * ds, 0, 0,
        a_log, d_skip, dt, state_in,
        t_total, dh, ds, n_heads, n_groups, chunk_len, seg_reset,
    )
}
