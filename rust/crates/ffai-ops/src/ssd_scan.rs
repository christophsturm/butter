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

/// CUDA (GLUE-FUSE front): one launch that does ssd_lcs + ssd_gather_bc_g +
/// ssd_xt. These three are independent (distinct source tensors, distinct
/// outputs) so this merges 3 kernel launches → 1 and lets the scheduler overlap
/// the three traffic streams. BIT-IDENTICAL to the three separate kernels (same
/// math, same writes). A grid-stride sweep handles each output's element count
/// independently. Gated by NEMOTRON_SSD_GLUE_FUSE.
const SSD_FRONT_FUSED_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_front_fused(
    const float* __restrict__ dt,     // [T, H]
    const float* __restrict__ a_log,  // [H]
    float*       __restrict__ lcs,    // [nc*H, L]
    const float* __restrict__ b_mat,  // B base (row-strided)
    const float* __restrict__ c_mat,  // C base
    __half*      __restrict__ b_out,  // [nc*G, L, ds] f16
    __half*      __restrict__ c_out,  // [nc*G, L, ds] f16
    const float* __restrict__ x,      // x base (row-strided)
    __half*      __restrict__ xt,     // [nc*H, dh, L] f16
    unsigned int T, unsigned int H, unsigned int G,
    unsigned int L, unsigned int ds, unsigned int dh, unsigned int nc,
    unsigned int bc_rs, unsigned int b_off, unsigned int c_off,
    unsigned int x_rs, unsigned int x_off)
{
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long gstride = (unsigned long long)gridDim.x * blockDim.x;

    // --- (1) lcs cumsum: one thread per (chunk, head) = nc*H ---
    for (unsigned long long idx = gid; idx < (unsigned long long)nc * H; idx += gstride) {
        unsigned int c = (unsigned int)(idx / H);
        unsigned int h = (unsigned int)(idx % H);
        float a = -expf(a_log[h]);
        float acc = 0.f;
        for (unsigned int i = 0; i < L; ++i) {
            unsigned int t = c * L + i;
            float dtv = (t < T) ? dt[t * H + h] : 0.f;
            acc += a * dtv;
            lcs[idx * L + i] = acc;
        }
    }

    // --- (2) gather B/C per-group: nc*G*L*ds elements ---
    unsigned int hpg = H / G;
    unsigned long long nbc = (unsigned long long)nc * G * L * ds;
    for (unsigned long long e = gid; e < nbc; e += gstride) {
        unsigned int s   = (unsigned int)(e % ds);
        unsigned int i   = (unsigned int)((e / ds) % L);
        unsigned int cg  = (unsigned int)(e / ((unsigned long long)ds * L));
        unsigned int c   = cg / G;
        unsigned int g   = cg % G;
        unsigned int t   = c * L + i;
        float bv = 0.f, cv = 0.f;
        if (t < T) {
            bv = b_mat[(unsigned long long)t * bc_rs + b_off + g * ds + s];
            cv = c_mat[(unsigned long long)t * bc_rs + c_off + g * ds + s];
        }
        b_out[e] = __float2half(bv);
        c_out[e] = __float2half(cv);
    }
    (void)hpg;

    // --- (3) transpose x → xt [nc*H, dh, L]: nc*H*dh*L elements ---
    unsigned long long nx = (unsigned long long)nc * H * dh * L;
    for (unsigned long long e = gid; e < nx; e += gstride) {
        unsigned int i  = (unsigned int)(e % L);
        unsigned int p  = (unsigned int)((e / L) % dh);
        unsigned int bh = (unsigned int)(e / ((unsigned long long)L * dh));
        unsigned int c  = bh / H;
        unsigned int h  = bh % H;
        unsigned int t  = c * L + i;
        float v = (t < T) ? x[(unsigned long long)t * x_rs + x_off + h * dh + p] : 0.f;
        xt[e] = __float2half(v);
    }
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

/// CUDA (GLUE-FUSE): in-place variant of `ssd_mmask` — masks `cb` IN PLACE
/// (single inout buffer, no separate `mmask` materialization). Each thread reads
/// and writes only its own element `e`, so the in-place update is hazard-free.
/// Dropping the distinct `mmask` buffer keeps the just-read tile L2-resident for
/// the write, cutting effective DRAM traffic. Bit-identical to `ssd_mmask`.
const SSD_MMASK_INPLACE_SRC: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void ssd_mmask_inplace(
    __half*       __restrict__ cb,    // [nc*H, L, L]  (masked in place)
    const float*  __restrict__ lcs,   // [nc*H, L]
    const float*  __restrict__ dt,    // [T, H]
    unsigned int T, unsigned int H, unsigned int L, unsigned int nc)
{
    unsigned long long n = (unsigned long long)nc * H * L * L;
    unsigned long long e = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    unsigned int j  = (unsigned int)(e % L);
    unsigned int i  = (unsigned int)((e / L) % L);
    unsigned int bh = (unsigned int)(e / ((unsigned long long)L * L));
    if (i < j) { cb[e] = __float2half(0.f); return; }
    unsigned int c  = bh / H;
    unsigned int h  = bh % H;
    unsigned int tj = c * L + j;
    float dtj = (tj < T) ? dt[tj * H + h] : 0.f;
    float decay = expf(lcs[bh * L + i] - lcs[bh * L + j]);
    float v = __half2float(cb[e]) * decay * dtj;
    cb[e] = __float2half(v);
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

/// FUSED tensor-core (wmma) SSD intra-chunk kernel: replaces G1 (CB=C·Bᵀ) +
/// `ssd_mmask` + G2 (y_intra=M·x) with ONE kernel — keeps both matmuls on the
/// tensor cores (wmma 16×16×16 f16→f32) and round-trips the L×L score tile
/// through SHARED only (never HBM), killing the `cb` + `mmask` [nc*H,L,L]
/// materializations. The SSD intra path is masked attention with NO softmax:
///   M[i,j] = (C[i]·B[j]) · exp(Lcs[i]-Lcs[j]) · dt[j]   (causal i≥j, else 0)
///   y_intra[i,p] = Σ_j M[i,j]·x[j,p]
/// The decay's within-chunk `Lcs[i]-Lcs[j]` subtraction stays in fp32 in-kernel
/// (stability-critical — exp(individual Lcs) underflows). One block per
/// (chunk, head, 16-row tile); reads the per-GROUP c_f16/b_f16 slices directly
/// (fused gather path). Fixed: ds=128, dh=64 (NemotronH). f16 in C/B/x, f16 out.
const SSD_INTRA_WMMA_SRC: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
// FUSED tensor-core SSD intra-chunk: ONE block per (chunk,head), 8 warps (=L/16
// row-tiles), C/B wmma frags loaded DIRECT FROM GLOBAL (L2-cached; matrix_b
// col_major handles the Bᵀ), x+lcs staged in ~28KB dynamic smem (->3 blocks/SM),
// O kept in register accumulator frags, output stored via the reused Ss tile.
// Replaces cuBLAS G1(CB)+ssd_mmask+G2(y_intra) with no cb/mmask HBM round-trip.
// Beats cuBLAS on GB10 (0.75 vs ~0.91 ms/layer @ S=2048). f16 CB round matches
// the chunked reference. Fixed NemotronH cell ds=128,dh=64; L=128.
extern "C" __global__ void ssd_intra_wmma(
    const __half* __restrict__ c_f16, const __half* __restrict__ b_f16,
    const float* __restrict__ x_mat, const float* __restrict__ lcs,
    const float* __restrict__ dt, __half* __restrict__ y_intra,
    unsigned T,unsigned H,unsigned G,unsigned hpg,unsigned L,unsigned ds,unsigned dh,unsigned nc,unsigned x_rs,unsigned x_off)
{
    const int BR=16,BC=16; int ds_nc=ds/16, dh_nc=dh/16;
    int bh=blockIdx.x; int c=bh/H,h=bh%H,g=h/hpg; long cg=(long)c*G+g;
    int warp=threadIdx.x>>5, lane=threadIdx.x&31, R0=warp*BR; int nwarp=L/BR;
    extern __shared__ char smem[];
    __half* Xs=(__half*)smem;
    float*  Lc=(float*)(Xs + (size_t)L*dh);
    float*  Ssw=(float*)(Lc + L);
    __half* Psw=(__half*)(Ssw + (size_t)nwarp*BR*BC);
    float (*Ss)[BC]=(float(*)[BC])(Ssw + (size_t)warp*BR*BC);
    __half (*Ps)[BC]=(__half(*)[BC])(Psw + (size_t)warp*BR*BC);
    int tid=threadIdx.x,nth=blockDim.x;
    for(int e=tid;e<(int)(L*dh);e+=nth){int p=e/(int)dh,pp=e%(int)dh;int t=c*(int)L+p;Xs[p*dh+pp]=(t<(int)T)?__float2half(x_mat[(long)t*x_rs+x_off+h*dh+pp]):__float2half(0.f);}
    for(int e=tid;e<(int)L;e+=nth)Lc[e]=lcs[(long)bh*L+e];
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> oacc[4];
    for(int t=0;t<dh_nc;t++) wmma::fill_fragment(oacc[t],0.f);
    int maxj=R0+BR-1; if(maxj>=(int)L)maxj=(int)L-1;
    const __half* Cbase=&c_f16[(cg*L+R0)*ds];
    wmma::fragment<wmma::accumulator,16,16,16,float> s_acc;
    for(int kj0=0;kj0<=maxj;kj0+=BC){
        wmma::fill_fragment(s_acc,0.f);
        const __half* Bbase=&b_f16[(cg*L+kj0)*ds];
        for(int cc=0;cc<ds_nc;cc++){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major>a;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major>b;
            wmma::load_matrix_sync(a,Cbase+cc*16,ds);
            wmma::load_matrix_sync(b,Bbase+cc*16,ds);
            wmma::mma_sync(s_acc,a,b,s_acc);
        }
        wmma::store_matrix_sync(&Ss[0][0],s_acc,BC,wmma::mem_row_major); __syncwarp();
        if(lane<BR){int i=lane,ri=R0+i;float li=Lc[ri];
            for(int j=0;j<BC;j++){int pj=kj0+j,tj=c*(int)L+pj;float w=0.f;
                if(ri>=pj){float dtj=dt[(long)tj*H+h];float dcay=__expf(li-Lc[pj]);float cbv=__half2float(__float2half(Ss[i][j]));w=cbv*dcay*dtj;}
                Ps[i][j]=__float2half(w);}}
        __syncwarp();
        for(int dc=0;dc<dh_nc;dc++){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major>pa;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major>vb;
            wmma::load_matrix_sync(pa,&Ps[0][0],BC);
            wmma::load_matrix_sync(vb,&Xs[kj0*dh+dc*16],dh);
            wmma::mma_sync(oacc[dc],pa,vb,oacc[dc]);
        }
        __syncwarp();
    }
    for(int dc=0;dc<dh_nc;dc++){
        wmma::store_matrix_sync(&Ss[0][0],oacc[dc],BC,wmma::mem_row_major); __syncwarp();
        for(int e=lane;e<BR*16;e+=32){int i=e/16,p=e%16,r=R0+i;if(r<(int)L)y_intra[((long)bh*L+r)*dh+dc*16+p]=__float2half(Ss[i][p]);}
        __syncwarp();
    }
}
"#;


/// HYBRID fused mask+PV: keep cuBLAS G1 (cb=C·Bᵀ, exact accumulation → cosine
/// preserved) but read `cb` and fuse the decay-mask + y_intra=M·x into ONE
/// kernel — killing the separate `ssd_mmask` materialization + the G2 cuBLAS
/// call. Only the PV matmul is hand-wmma (the CB stays cuBLAS-exact). One block
/// per (chunk,head), L/16 warps, ~28KB dynamic smem. cb is [nc*H,L,L] f16.
const SSD_MASKPV_SRC: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void ssd_maskpv(
    const __half* __restrict__ cb, const float* __restrict__ x_mat,
    const float* __restrict__ lcs, const float* __restrict__ dt, __half* __restrict__ y_intra,
    unsigned T,unsigned H,unsigned L,unsigned dh,unsigned nc,unsigned x_rs,unsigned x_off)
{
    const int BR=16,BC=16; int dh_nc=dh/16;
    int bh=blockIdx.x; int c=bh/H,h=bh%H;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,R0=warp*BR; int nwarp=L/BR;
    extern __shared__ char smem[];
    __half* Xs=(__half*)smem;
    float*  Lc=(float*)(Xs+(size_t)L*dh);
    __half* Psw=(__half*)(Lc+L);
    float*  Ssw=(float*)(Psw+(size_t)nwarp*BR*BC);
    __half (*Ps)[BC]=(__half(*)[BC])(Psw+(size_t)warp*BR*BC);
    float (*Ss)[BC]=(float(*)[BC])(Ssw+(size_t)warp*BR*BC);
    int tid=threadIdx.x,nth=blockDim.x;
    for(int e=tid;e<(int)(L*dh);e+=nth){int p=e/(int)dh,pp=e%(int)dh;int t=c*(int)L+p;Xs[p*dh+pp]=(t<(int)T)?__float2half(x_mat[(long)t*x_rs+x_off+h*dh+pp]):__float2half(0.f);}
    for(int e=tid;e<(int)L;e+=nth)Lc[e]=lcs[(long)bh*L+e];
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> oacc[4];
    for(int t=0;t<dh_nc;t++)wmma::fill_fragment(oacc[t],0.f);
    int maxj=R0+BR-1;if(maxj>=(int)L)maxj=(int)L-1;
    for(int kj0=0;kj0<=maxj;kj0+=BC){
        if(lane<BR){int i=lane,ri=R0+i;float li=Lc[ri];
            for(int j=0;j<BC;j++){int pj=kj0+j,tj=c*(int)L+pj;float w=0.f;
                if(ri>=pj){float cbv=__half2float(cb[((long)bh*L+ri)*L+pj]);float dtj=dt[(long)tj*H+h];float dcay=__expf(li-Lc[pj]);w=cbv*dcay*dtj;}
                Ps[i][j]=__float2half(w);}}
        __syncwarp();
        for(int dc=0;dc<dh_nc;dc++){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major>pa;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major>vb;
            wmma::load_matrix_sync(pa,&Ps[0][0],BC);
            wmma::load_matrix_sync(vb,&Xs[kj0*dh+dc*16],dh);
            wmma::mma_sync(oacc[dc],pa,vb,oacc[dc]);
        }
        __syncwarp();
    }
    for(int dc=0;dc<dh_nc;dc++){
        wmma::store_matrix_sync(&Ss[0][0],oacc[dc],BC,wmma::mem_row_major);__syncwarp();
        for(int e=lane;e<BR*16;e+=32){int i=e/16,p=e%16,r=R0+i;if(r<(int)L)y_intra[((long)bh*L+r)*dh+dc*16+p]=__float2half(Ss[i][p]);}
        __syncwarp();
    }
}
"#;

/// FUSED inter-chunk SSD (agent-built, validated 5090): one block per head,
/// S_in[ds,dh] fp32 resident in SRAM across all chunks — replaces bdt + G3 +
/// recurrence + G4 + combine (5 ops + 4 cuBLAS GEMMs) with ONE launch, deleting
/// the states/sin_t/CS HBM tensors. Consumes y_intra; produces final y.
const SSD_INTER_FUSED2_SRC: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define LOG2E 1.4426950408889634f
__device__ __forceinline__ float fast_exp(float x){ return expf(x); }

#define DS 128
#define DH 64
#define L  128

extern "C" __global__ void __launch_bounds__(128,1)
ssd_inter_fused2(
    const __half* __restrict__ c_f16,
    const __half* __restrict__ b_f16,
    const float*  __restrict__ x_mat,
    const float*  __restrict__ lcs,
    const float*  __restrict__ dt,
    const __half* __restrict__ y_intra,
    const float*  __restrict__ Dvec,
    float* __restrict__ y_out,
    unsigned int T, unsigned int H, unsigned int G, unsigned int nc)
{
    const int h   = blockIdx.x;
    const int hpg = H / G;
    const int g   = h / hpg;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;     // 0..3 (each warp owns 2 of the 8 row-tiles)

    extern __shared__ char smem[];
    // Layout (81KB, fits 99KB sm_120 optin):
    //   S_in [DS][DH] fp32  = 32KB  (persistent recurrence state; row=d, col=p)
    //   big0  32KB          = C tile [L][DS] half / B^T tile [DS][L] half;
    //                         reused as CS [L][DH] fp32 (8192 floats) after the mma.
    //   Xtile 16KB          = S_in-half mirror [DS][DH] / X tile [L][DH] half.
    //   sh_scale[L], sh_lcs[L] fp32.
    float*  S_in     = (float*)smem;                 // 32KB
    __half* big0     = (__half*)(S_in + DS*DH);      // 32KB (16384 halfs)
    float*  CS       = (float*)big0;                 // alias: CS fp32 [L][DH] = 8192 floats = 32KB
    __half* Xtile    = big0 + 16384;                 // 16KB (8192 halfs)
    float*  sh_scale = (float*)(Xtile + L*DH);       // [L]
    float*  sh_lcs   = sh_scale + L;                 // [L]

    // zero S_in
    for(int e=tid; e<DS*DH; e+=128) S_in[e]=0.f;
    __syncthreads();

    for(unsigned c=0;c<nc;c++){
        const long cg=(long)c*G+g;
        const long bh=(long)c*H+h;

        // load lcs + scale
        if(tid<L) sh_lcs[tid]=lcs[bh*L+tid];
        __syncthreads();
        float lcs_last = sh_lcs[L-1];
        if(tid<L){
            float dtj=dt[(c*L+tid)*H+h];
            sh_scale[tid]=fast_exp(fminf(lcs_last - sh_lcs[tid],0.f))*dtj;
        }
        __syncthreads();

        // ===== phase1: y_off = exp(lcs[i]) * (C @ S_in), combine =====
        // big0 := C tile [L][DS] half;  Xtile := S_in half mirror [DS][DH]
        for(int e=tid;e<L*DS;e+=128){ int i=e/DS,d=e%DS; big0[i*DS+d]=c_f16[(cg*L+i)*DS+d]; }
        for(int e=tid;e<DS*DH;e+=128){ Xtile[e]=__float2half(S_in[e]); }
        __syncthreads();

        // CS[L][DH] = C[L][DS] @ S_in[DS][DH].  8 row-tiles (L/16) x 4 col (dh/16).
        // Each warp owns row-tiles {warp, warp+4}; hold 2x4 frags in registers so we
        // can overwrite big0 (=CS fp32 alias) only after all reads of big0 are done.
        {
            wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][DH/16];
            for(int ri=0; ri<2; ri++){ int rt=warp+ri*4;
                for(int ct=0; ct<DH/16; ct++){
                    wmma::fill_fragment(acc[ri][ct],0.f);
                    for(int kt=0; kt<DS/16; kt++){
                        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
                        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
                        wmma::load_matrix_sync(a, &big0[(rt*16)*DS + kt*16], DS);
                        wmma::load_matrix_sync(b, &Xtile[(kt*16)*DH + ct*16], DH);
                        wmma::mma_sync(acc[ri][ct],a,b,acc[ri][ct]);
                    }
                }
            }
            __syncthreads();  // all big0 reads done; safe to overwrite big0 as CS fp32
            for(int ri=0; ri<2; ri++){ int rt=warp+ri*4;
                for(int ct=0; ct<DH/16; ct++)
                    wmma::store_matrix_sync(&CS[(rt*16)*DH + ct*16], acc[ri][ct], DH, wmma::mem_row_major);
            }
        }
        __syncthreads();
        // combine: y = y_intra + exp(lcs[i])*CS + x*D
        {
            float Dh=Dvec[h];
            for(int e=tid;e<L*DH;e+=128){
                int i=e/DH,p=e%DH;
                float yoff=fast_exp(sh_lcs[i])*CS[i*DH+p];
                int t=c*L+i;
                float xv=x_mat[(long)t*H*DH+h*DH+p];
                float yi=__half2float(y_intra[(bh*L+i)*DH+p]);
                y_out[(long)t*H*DH+h*DH+p]=yi+yoff+xv*Dh;
            }
        }
        __syncthreads();

        // ===== phase2: S_chunk = Bsc^T @ X, recurrence =====
        // big0 := B^T scaled  [DS][L]  half:  big0[d*L+j] = scale[j]*B[j][d]
        for(int e=tid;e<L*DS;e+=128){ int j=e/DS,d=e%DS;
            float bv=__half2float(b_f16[(cg*L+j)*DS+d]);
            big0[d*L+j]=__float2half(sh_scale[j]*bv);
        }
        // Xtile := X[L][DH] half
        for(int e=tid;e<L*DH;e+=128){ int j=e/DH,p=e%DH; int t=c*L+j;
            Xtile[e]=__float2half(x_mat[(long)t*H*DH+h*DH+p]);
        }
        __syncthreads();
        float decay=fast_exp(lcs_last);
        // Sc[DS][DH] = big0(B^T)[DS][L] @ Xtile[L][DH].  8 row-tiles (ds/16) x 4 col.
        // Register-hold frags so big0 (=CS alias) is overwritten only after reads done.
        {
            wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][DH/16];
            for(int ri=0; ri<2; ri++){ int rt=warp+ri*4;
                for(int ct=0; ct<DH/16; ct++){
                    wmma::fill_fragment(acc[ri][ct],0.f);
                    for(int kt=0; kt<L/16; kt++){
                        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
                        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
                        wmma::load_matrix_sync(a,&big0[(rt*16)*L + kt*16], L);
                        wmma::load_matrix_sync(b,&Xtile[(kt*16)*DH + ct*16], DH);
                        wmma::mma_sync(acc[ri][ct],a,b,acc[ri][ct]);
                    }
                }
            }
            __syncthreads();  // all big0 reads done
            for(int ri=0; ri<2; ri++){ int rt=warp+ri*4;
                for(int ct=0; ct<DH/16; ct++)
                    wmma::store_matrix_sync(&CS[(rt*16)*DH + ct*16], acc[ri][ct], DH, wmma::mem_row_major);
            }
        }
        __syncthreads();
        // recurrence: S_in = decay*S_in + Sc   (note Sc rows index 'd', cols 'p';
        // S_in is [DS][DH] row=d col=p — matches Sc layout exactly)
        for(int e=tid;e<DS*DH;e+=128) S_in[e]=decay*S_in[e]+CS[e];
        __syncthreads();
    }
}
"#;


/// FULLY-FUSED tensor-core chunked SSD (agent-built): three launches
/// (phaseA_tc + phaseB + phaseC) replace the WHOLE SSD trio for the prefill
/// path. The intra-chunk L×L score is STREAMED per 16×16 tile through SHARED
/// only (never HBM) using warp wmma (HMMA.16816 f16->f32) — fits the 99 KB
/// consumer-Blackwell smem cap at L=128 and keeps M/Mm off HBM. NO cp.async /
/// TMA — plain global->smem + EXPLICIT __syncthreads — so the sm_121
/// async-barrier divergence cannot occur. FP32 accumulate. I/O matches the live
/// compact call site: X token-major row-strided x[t*x_rs+x_off+h*dh+p]; Bg/Cg
/// token-major bc[(t*G+grp)*ds+n]; dt token-major dt[t*H+h]; a_log[H] RAW
/// (A_h=-exp(a_log[h]) negated in-kernel); D[H] residual; Y token-major
/// y[(t*H+h)*dh+p] f32; state_out[H,dh,ds] state_out[(h*dh+d)*ds+n]. Internal
/// scratch (Yintra[H,S,dh], stateC[H,G,ds,dh], cumChunk[H,G], Sin[H,G,ds,dh])
/// stays head-major/private. Fixed NemotronH H=64,dh=64,ds=128,G=8,L=128.
/// Gated by NEMOTRON_SSD_FUSED_TC at the call site.
const SSD_FUSED_TC_SRC: &str = r#"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
#define H_  64
#define DS  128
#define DH  64
#define GG  8
#define L_  128

extern "C" __global__ void ssd_fused_phaseA_tc(
    const float* __restrict__ X,
    const float* __restrict__ Bg,
    const float* __restrict__ Cg,
    const float* __restrict__ a_log,
    const float* __restrict__ dt,
    float* __restrict__ Yintra,
    float* __restrict__ stateC,
    float* __restrict__ cumChunk,
    unsigned int S, unsigned int x_rs, unsigned int x_off)
{
    const int g = blockIdx.x, h = blockIdx.y;
    const int G = (int)(S / L_) + ((S % L_) ? 1 : 0);
    const int grp = h / (H_ / GG);
    const int tid = threadIdx.x, nth = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31, nwarp = nth >> 5;
    const float Ah = -__expf(a_log[h]);
    const int chunk0 = g * L_;
    extern __shared__ char smc[];
    float* s_cum = (float*)smc;
    half*  s_Xb  = (half*)(s_cum + L_);
    half*  s_Bh  = s_Xb + L_*DH;
    half*  s_Ch  = s_Bh + L_*DS;
    float* s_Msc = (float*)(s_Ch + L_*DS);
    half*  s_Mh  = (half*)(s_Msc + nwarp*16*16);
    if (tid == 0) { float acc = 0.f;
      for (int i = 0; i < L_; ++i) { int t = chunk0 + i;
        float dtv = (t < (int)S) ? dt[(size_t)t*H_ + h] : 0.f; acc += dtv; s_cum[i] = Ah*acc; }
      cumChunk[(size_t)h*G + g] = Ah*acc; }
    __syncthreads();
    for (int i = tid; i < L_*DS; i += nth) { int r = i / DS, c = i % DS; int t = chunk0 + r;
      float bv = (t < (int)S) ? Bg[((size_t)t*GG + grp)*DS + c] : 0.f;
      float cv = (t < (int)S) ? Cg[((size_t)t*GG + grp)*DS + c] : 0.f;
      s_Bh[i] = __float2half(bv); s_Ch[i] = __float2half(cv); }
    for (int i = tid; i < L_*DH; i += nth) { int r = i / DH, c = i % DH; int t = chunk0 + r;
      float dtv = (t < (int)S) ? dt[(size_t)t*H_ + h] : 0.f;
      float xv  = (t < (int)S) ? X[(size_t)t*x_rs + x_off + (size_t)h*DH + c] : 0.f;
      s_Xb[i] = __float2half(xv * dtv); }
    __syncthreads();
    float* myMsc = s_Msc + warp*16*16; half* myMh = s_Mh + warp*16*16;
    for (int mi = warp*16; mi < L_; mi += nwarp*16) {
      wmma::fragment<wmma::accumulator,16,16,16,float> yacc[DH/16];
      #pragma unroll
      for (int q = 0; q < DH/16; ++q) wmma::fill_fragment(yacc[q], 0.f);
      for (int nj = 0; nj <= mi; nj += 16) {
        wmma::fragment<wmma::accumulator,16,16,16,float> macc; wmma::fill_fragment(macc, 0.f);
        for (int k = 0; k < DS; k += 16) {
          wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> fb;
          wmma::load_matrix_sync(fa, s_Ch + mi*DS + k, DS);
          wmma::load_matrix_sync(fb, s_Bh + nj*DS + k, DS);
          wmma::mma_sync(macc, fa, fb, macc);
        }
        wmma::store_matrix_sync(myMsc, macc, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) { int r = e >> 4, c = e & 15; int gr = mi+r, gc = nj+c;
          float v = 0.f; if (gc <= gr) v = myMsc[e] * __expf(s_cum[gr] - s_cum[gc]); myMh[e] = __float2half(v); }
        __syncwarp();
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fM;
        wmma::load_matrix_sync(fM, myMh, 16);
        #pragma unroll
        for (int q = 0; q < DH/16; ++q) {
          wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fX;
          wmma::load_matrix_sync(fX, s_Xb + nj*DH + q*16, DH);
          wmma::mma_sync(yacc[q], fM, fX, yacc[q]);
        }
      }
      #pragma unroll
      for (int q = 0; q < DH/16; ++q)
        wmma::store_matrix_sync(Yintra + ((size_t)h*S + chunk0 + mi)*DH + q*16, yacc[q], DH, wmma::mem_row_major);
    }
    __syncthreads();
    float cumlast = s_cum[L_-1];
    for (int i = tid; i < DS*DH; i += nth) { int n = i / DH, d = i % DH; float acc = 0.f;
      for (int j = 0; j < L_; ++j) { float w = __expf(cumlast - s_cum[j]);
        acc += w * __half2float(s_Bh[j*DS + n]) * __half2float(s_Xb[j*DH + d]); }
      stateC[(((size_t)h*G + g)*DS + n)*DH + d] = acc; }
}

extern "C" __global__ void ssd_fused_phaseB(
    const float* __restrict__ stateC,
    const float* __restrict__ cumChunk,
    float* __restrict__ Sin,
    float* __restrict__ state_out,
    unsigned int S)
{
    int h = blockIdx.x; int G = (int)(S / L_) + ((S % L_) ? 1 : 0);
    int tid = threadIdx.x, nth = blockDim.x;
    extern __shared__ float run[];
    for (int i = tid; i < DS*DH; i += nth) run[i] = 0.f; __syncthreads();
    for (int g = 0; g < G; ++g) {
      for (int i = tid; i < DS*DH; i += nth) Sin[(((size_t)h*G + g)*DS*DH) + i] = run[i];
      __syncthreads();
      float decay = __expf(cumChunk[(size_t)h*G + g]);
      for (int i = tid; i < DS*DH; i += nth) run[i] = decay*run[i] + stateC[(((size_t)h*G + g)*DS*DH) + i];
      __syncthreads();
    }
    for (int i = tid; i < DS*DH; i += nth) { int n = i / DH, d = i % DH;
      state_out[((size_t)h*DH + d)*DS + n] = run[i]; }
}

extern "C" __global__ void ssd_fused_phaseC(
    const float* __restrict__ Cg,
    const float* __restrict__ a_log,
    const float* __restrict__ dt,
    const float* __restrict__ X,
    const float* __restrict__ D,
    const float* __restrict__ Sin,
    const float* __restrict__ Yintra,
    float* __restrict__ Y,
    unsigned int S, unsigned int x_rs, unsigned int x_off)
{
    int g = blockIdx.x, h = blockIdx.y; int G = (int)(S / L_) + ((S % L_) ? 1 : 0);
    int grp = h / (H_ / GG); int tid = threadIdx.x, nth = blockDim.x;
    const float Ah = -__expf(a_log[h]); const int chunk0 = g * L_;
    extern __shared__ float s_cum[];
    if (tid == 0) { float acc = 0.f;
      for (int i = 0; i < L_; ++i) { int t = chunk0 + i;
        float dtv = (t < (int)S) ? dt[(size_t)t*H_ + h] : 0.f; acc += dtv; s_cum[i] = Ah*acc; } }
    __syncthreads();
    const float* SinG = Sin + (((size_t)h*G + g)*DS*DH);
    for (int idx = tid; idx < L_*DH; idx += nth) { int r = idx / DH, d = idx % DH; int t = chunk0 + r;
      if (t >= (int)S) continue;
      const float* Cr = &Cg[((size_t)t*GG + grp)*DS];
      float inc = 0.f; for (int n = 0; n < DS; ++n) inc += Cr[n] * SinG[(size_t)n*DH + d];
      float w = __expf(s_cum[r]);
      float xv = X[(size_t)t*x_rs + x_off + (size_t)h*DH + d];
      float yi = Yintra[((size_t)h*S + t)*DH + d];
      Y[((size_t)t*H_ + h)*DH + d] = yi + w*inc + D[h]*xv;
    }
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
    // Fused tensor-core intra-chunk (NEMOTRON_SSD_INTRA_WMMA=1, requires fused
    // per-group gather): ONE wmma kernel replaces G1(CB)+mmask+G2(y_intra),
    // killing the cb+mmask [nc*H,L,L] HBM materializations. Fixed Nemotron cell.
    let intra_wmma = fused && std::env::var("NEMOTRON_SSD_INTRA_WMMA").is_ok();
    let hybrid_pv = fused && !intra_wmma && std::env::var("NEMOTRON_SSD_HYBRID").is_ok();

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

    // FULLY-FUSED tensor-core SSD (NEMOTRON_SSD_FUSED_TC=1, A/B-able, default
    // OFF) — 3 launches replace the WHOLE trio. Fixed NemotronH cell at L=128.
    // Reads X/B/C/dt in the live token-major compact layout; writes Y [T,H,dh]
    // f32 + state_out [H,dh,ds]. No async copy (sm_121-safe). See SSD_FUSED_TC_SRC.
    let fused_tc = std::env::var("NEMOTRON_SSD_FUSED_TC").is_ok();
    if fused_tc {
        if l != 128 {
            return Err(Error::Msg("NEMOTRON_SSD_FUSED_TC requires chunk_len=128 (set NEMOTRON_SSD_L=128)".into()));
        }
        let yintra = Tensor::empty(dev, vec![h * t * dhu], DType::F32)?;
        let state_c = Tensor::empty(dev, vec![h * nc * dsu * dhu], DType::F32)?;
        let cum_chunk = Tensor::empty(dev, vec![h * nc], DType::F32)?;
        let sin_buf = Tensor::empty(dev, vec![h * nc * dsu * dhu], DType::F32)?;
        let smem_a = (l*4 + l*dhu*2 + 2*l*dsu*2 + 8*16*16*4 + 8*16*16*2) as u32;
        let smem_b = (dsu*dhu*4) as u32;
        let smem_c = (l*4) as u32;
        let sc_a: Vec<Vec<u8>> = [tt, x_rs, x_off].iter().map(|v| v.to_le_bytes().to_vec()).collect();
        dev.dispatch_raw_cuda(
            SSD_FUSED_TC_SRC, "ssd_fused_tc.cu", "ssd_fused_phaseA_tc",
            &[(bb(x), 0), (bb(b_mat), 0), (bb(c_mat), 0), (bb(a_log), 0), (bb(dt), 0),
              (bb(&yintra), 0), (bb(&state_c), 0), (bb(&cum_chunk), 0)],
            &sc_a, [ncn, hh, 1], [256, 1, 1], smem_a, false,
        )?;
        let sc_b: Vec<Vec<u8>> = [tt].iter().map(|v| v.to_le_bytes().to_vec()).collect();
        dev.dispatch_raw_cuda(
            SSD_FUSED_TC_SRC, "ssd_fused_tc.cu", "ssd_fused_phaseB",
            &[(bb(&state_c), 0), (bb(&cum_chunk), 0), (bb(&sin_buf), 0), (bb(&state_out), 0)],
            &sc_b, [hh, 1, 1], [256, 1, 1], smem_b, false,
        )?;
        let sc_c: Vec<Vec<u8>> = [tt, x_rs, x_off].iter().map(|v| v.to_le_bytes().to_vec()).collect();
        dev.dispatch_raw_cuda(
            SSD_FUSED_TC_SRC, "ssd_fused_tc.cu", "ssd_fused_phaseC",
            &[(bb(c_mat), 0), (bb(a_log), 0), (bb(dt), 0), (bb(x), 0), (bb(d_skip), 0),
              (bb(&sin_buf), 0), (bb(&yintra), 0), (bb(&y), 0)],
            &sc_c, [ncn, hh, 1], [256, 1, 1], smem_c, false,
        )?;
        return Ok((state_out, y));
    }

    // GLUE-FUSE front: merge ssd_lcs + ssd_gather_bc_g + ssd_xt into ONE launch
    // (bit-identical; 3 launches → 1, overlapped traffic streams). Requires the
    // per-group fused gather path. Escape: leave NEMOTRON_SSD_GLUE_FUSE unset.
    let glue_fuse = fused && std::env::var("NEMOTRON_SSD_GLUE_FUSE").is_ok();
    // Front-fusion (lcs+gather+xt → 1 launch) is a separate experiment flag —
    // measured ~5% SLOWER (HBM-neutral; loses per-kernel grid tuning), kept off.
    let front_fuse = fused && std::env::var("NEMOTRON_SSD_FRONT_FUSE").is_ok();
    if front_fuse {
        // Grid sized to the biggest output (B/C gather: nc*G*L*ds); grid-stride
        // loops inside the kernel cover the smaller lcs/xt element counts too.
        let n_front = (bgc * l * dsu).max(bhc * dhu * l);
        let sc: Vec<Vec<u8>> = [tt, hh, gg, ll, dsd, dhd, ncn, bc_rs, b_off, c_off, x_rs, x_off]
            .iter().map(|v| v.to_le_bytes().to_vec()).collect();
        let block = 256u32;
        let grid = [((n_front as u32) + block - 1) / block, 1, 1];
        dev.dispatch_raw_cuda(
            SSD_FRONT_FUSED_SRC, "ssd_front_fused.cu", "ssd_front_fused",
            &[(bb(dt), 0), (bb(a_log), 0), (bb(&lcs), 0),
              (bb(b_mat), 0), (bb(c_mat), 0), (bb(&b_f16), 0), (bb(&c_f16), 0),
              (bb(x), 0), (bb(&xt), 0)],
            &sc, grid, [block, 1, 1], 0, false,
        )?;
    } else {
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
    }

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

    let st_ldh = (l * dhu) as i64 * el;
    let st_dhl = (dhu * l) as i64 * el;
    if intra_wmma {
        // ── FUSED: one wmma kernel does CB + mask + M·x → y_intra directly,
        // reading the per-group c_f16/b_f16 slices + x in place (no cb/mmask). ──
        // grid: one block per (chunk,head); block = L/16 warps (256 thr); ~28KB
        // dynamic smem (x+lcs staged, per-warp Ss/Ps) → 3 blocks/SM. Beats cuBLAS.
        let nwarp = (l / 16) as u32;
        let smem_b = (l * dhu * 2 + l * 4 + (l/16) * 16 * 16 * 4 + (l/16) * 16 * 16 * 2) as u32;
        let sc: Vec<Vec<u8>> = [tt, hh, gg, hpgn, ll, dsd, dhd, ncn, x_rs, x_off]
            .iter().map(|v| v.to_le_bytes().to_vec()).collect();
        dev.dispatch_raw_cuda(
            SSD_INTRA_WMMA_SRC, "ssd_intra_wmma.cu", "ssd_intra_wmma",
            &[(bb(&c_f16), 0), (bb(&b_f16), 0), (bb(x), 0), (bb(&lcs), 0), (bb(dt), 0), (bb(&y_intra), 0)],
            &sc, [bhc as u32, 1, 1], [nwarp * 32, 1, 1], smem_b, false,
        )?;
    } else {
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

    if hybrid_pv {
        // HYBRID: cb (cuBLAS, exact) → fused mask+PV (no mmask materialization,
        // no G2). One block per (chunk,head), L/16 warps, ~28KB dynamic smem.
        let nwarp = (l / 16) as u32;
        let smem_b = (l * dhu * 2 + l * 4 + (l/16) * 16 * 16 * 2 + (l/16) * 16 * 16 * 4) as u32;
        let sc2: Vec<Vec<u8>> = [tt, hh, ll, dhd, ncn, x_rs, x_off]
            .iter().map(|v| v.to_le_bytes().to_vec()).collect();
        dev.dispatch_raw_cuda(
            SSD_MASKPV_SRC, "ssd_maskpv.cu", "ssd_maskpv",
            &[(bb(&cb), 0), (bb(x), 0), (bb(&lcs), 0), (bb(dt), 0), (bb(&y_intra), 0)],
            &sc2, [bhc as u32, 1, 1], [nwarp * 32, 1, 1], smem_b, false,
        )?;
    } else {
    // 4. M = CB ⊙ Lmask ⊙ dt[j], causal.
    // GLUE-FUSE: write the mask IN-PLACE over `cb` (which is dead after this
    // read) instead of a fresh `mmask` buffer — kills the separate 33MB mmask
    // materialization and keeps the freshly-read `cb` tile L2-resident for the
    // in-place write, cutting effective DRAM write+read of mmask. Bit-identical
    // (same elementwise math; `m_out` aliases `cb`). The kernel reads cb[e] then
    // writes the SAME index e, so the alias is hazard-free per element.
    let mmask_buf: &Tensor = if glue_fuse { &cb } else { &mmask };
    if glue_fuse {
        // in-place: mask cb directly (no separate mmask buffer / round-trip).
        raw(SSD_MMASK_INPLACE_SRC, "ssd_mmask_inplace.cu", "ssd_mmask_inplace",
            &[(bb(&cb), 0), (bb(&lcs), 0), (bb(dt), 0)],
            &[tt, hh, ll, ncn], bhc * l * l)?;
    } else {
        raw(SSD_MMASK_SRC, "ssd_mmask.cu", "ssd_mmask",
            &[(bb(&cb), 0), (bb(&lcs), 0), (bb(dt), 0), (bb(&mmask), 0)],
            &[tt, hh, ll, ncn], bhc * l * l)?;
    }

    // ── G2: y_intra = M · x  [L,dh]. m=L, n=dh, k=L, X=mmask, W=xt[dh,L]. ───
    dev.gemm_strided_batched(
        bb(mmask_buf), st_ll, bb(&xt), st_dhl, bb(&y_intra), st_ldh,
        l, dhu, l, bhc, DType::F16,
    )?;
    }
    }


    // FUSED inter-chunk (NEMOTRON_SSD_INTER): ONE launch (one block/head, S_in
    // resident in SRAM across all chunks) replaces bdt+G3+recur+G4+combine and
    // the states/sin_t/CS HBM round-trips. Consumes y_intra; writes final y.
    // Prefill-only (no state_out carry); requires the compact x layout (x_rs=H*dh).
    let inter_fused = seg_reset.is_none() && std::env::var("NEMOTRON_SSD_INTER").is_ok();
    if inter_fused {
        let sc3: Vec<Vec<u8>> = [tt, hh, gg, ncn].iter().map(|v| v.to_le_bytes().to_vec()).collect();
        dev.dispatch_raw_cuda(
            SSD_INTER_FUSED2_SRC, "ssd_inter_fused2.cu", "ssd_inter_fused2",
            &[(bb(&c_f16), 0), (bb(&b_f16), 0), (bb(x), 0), (bb(&lcs), 0), (bb(dt), 0),
              (bb(&y_intra), 0), (bb(d_skip), 0), (bb(&y), 0)],
            &sc3, [hh, 1, 1], [128, 1, 1], 82944, false,
        )?;
        return Ok((state_out, y));
    }

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
