// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # Mamba2 SSD chunked-MATMUL prefill scan — PORTABLE backend.
//!
//! Same state-space-duality chunked-matmul algorithm as the CUDA
//! `ssm_prefill_scan_ssd` (in `ssd_scan.rs`), but built ENTIRELY from portable
//! registry kernels that codegen to MSL / HIP / SPIR-V — no `dispatch_raw_cuda`,
//! no cuBLAS `cublasGemmStridedBatchedEx`. Runs on Apple hardware MMA (and HIP /
//! Vulkan / RDNA4) via the portable `ffai_gemm_batched` Reduction-mode GEMM.
//!
//! The 4 batched GEMMs (steps 1/2/4/5 below) run on `ffai_gemm_batched`
//! (batch = nc·H, the SAME `out = input · weightᵀ` contraction `ffai_gemm`
//! uses, replicated over the batch axis with per-matrix strides). The small
//! segsum / Lmask / decay / transpose / recurrence / combine kernels run on
//! the portable `ssd_*` `#[kernel]` ops in `metaltile-std::ffai::ssm`.
//!
//! Everything stays f32 (no f16 GEMM) — portability + correctness first; the
//! sequential scan also accumulates in f32, so this matches it tightly.
//!
//! Gated by `NEMOTRON_SSD_PORTABLE` at the call site (A/B-able vs the
//! sequential `ssm_prefill_scan`). The CUDA `ssm_prefill_scan_ssd` is left
//! byte-for-byte unchanged.
//!
//! Fixed for NemotronH: dh=64, ds=128, H=64, G=8 (hpg=8). L ∈ {128,256}.

use ffai_core::{Binding, DType, Device, Error, Grid, Result, Tensor};
use ffai_kernels_core::ir::KernelMode;

use crate::cached_ir;

/// Portable Mamba2 SSD **chunked-matmul prefill scan** for the NemotronH cell
/// (dh=64, ds=128, H=64, G=8). Drop-in replacement for the sequential
/// `ssm_prefill_scan`: returns `(state_out, y)` with `y` `[T·H·dh]` f32 and
/// `state_out` matching `state_in`. Inputs are f32. `t_total` need not be a
/// multiple of `chunk_len` — the tail chunk is zero-padded. `chunk_len` is
/// typically 128 or 256. Backend-portable (runs on Metal/HIP/Vulkan).
#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_scan_ssd_portable(
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
) -> Result<(Tensor, Tensor)> {
    if (dh, ds, n_heads, n_groups) != (64, 128, 64, 8) {
        return Err(Error::Msg(format!(
            "ssm_prefill_scan_ssd_portable: only Nemotron cell (64,128,64,8) wired, got ({dh},{ds},{n_heads},{n_groups})"
        )));
    }
    if x.dtype != DType::F32 {
        return Err(Error::Msg(
            "ssm_prefill_scan_ssd_portable: only F32 input wired (cast first)".into(),
        ));
    }
    let (t, h, dhu, dsu, g) = (
        t_total as usize,
        n_heads as usize,
        dh as usize,
        ds as usize,
        n_groups as usize,
    );
    let hpg = h / g; // heads per group
    let l = chunk_len as usize;
    let nc = t.div_ceil(l); // number of chunks (tail zero-padded)
    let bhc = nc * h; // batch count for the GEMMs

    // ── Output + final state (f32, matching the sequential scan). ─────────
    let y = Tensor::empty(dev, vec![t * h * dhu], DType::F32)?;
    let state_out = Tensor::empty(dev, state_in.shape.clone(), DType::F32)?;

    // ── Scratch buffers (all f32). ────────────────────────────────────────
    // NOTE: b_g/c_g ([nc*H, L, ds]) and cb ([nc*H, L, L]) are GONE — the G1
    // (CB→M) and G4 (CS) GEMMs now read B/C straight from [T, G, ds] with the
    // 8× per-head broadcast folded into the tile load (ssd_g1_cb / ssd_g4_cs),
    // and G1 applies the decay-mask in its epilogue (ssd_mmask fused away).
    // That kills the 8× redundant gather_bc write+read AND the [L,L] CB
    // round-trip per chunk.
    let lcs = Tensor::empty(dev, vec![bhc * l], DType::F32)?; // [nc*H, L]
    let xt = Tensor::empty(dev, vec![bhc * dhu * l], DType::F32)?; // [nc*H, dh, L]
    let mmask = Tensor::empty(dev, vec![bhc * l * l], DType::F32)?; // [nc*H, L, L] (M from G1)
    let y_intra = Tensor::empty(dev, vec![bhc * l * dhu], DType::F32)?; // [nc*H, L, dh]
    let bdt = Tensor::empty(dev, vec![bhc * dsu * l], DType::F32)?; // [nc*H, ds, L]
    let s_chunk = Tensor::empty(dev, vec![bhc * dsu * dhu], DType::F32)?; // [nc*H, ds, dh]
    let sin_t = Tensor::empty(dev, vec![bhc * dhu * dsu], DType::F32)?; // [nc*H, dh, ds]
    let cs = Tensor::empty(dev, vec![bhc * l * dhu], DType::F32)?; // [nc*H, L, dh]

    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let (tt, hh, gg, ll, dsd, dhd, ncn, hpgn) = (
        t_total,
        n_heads,
        n_groups,
        l as u32,
        ds,
        dh,
        nc as u32,
        hpg as u32,
    );

    // Grid3D elementwise dispatch helper: one thread per output element.
    let elem = |name: &str, bufs: &[&Tensor], scalars: &[Binding], n_threads: usize| -> Result<()> {
        let kern = cached_ir(name, DType::F32, || {
            let mut k = build_ssd_ir(name);
            k.mode = KernelMode::Grid3D;
            k
        });
        let mut bindings: Vec<Binding> =
            bufs.iter().map(|t| Binding::Buffer(t.buffer.clone())).collect();
        bindings.extend_from_slice(scalars);
        // 1-D grid; cap block at 256 threads per group.
        let block = 256u32;
        let grid = Grid {
            grid: [(n_threads as u32).div_ceil(block), 1, 1],
            block: [block, 1, 1],
        };
        dev.dispatch(&kern, &bindings, grid)
    };

    // Batched GEMM helper: out[bh][r,o] = Σ_k weight[bh][o,k]·input[bh][r,k].
    // Reduction-mode, grid [(out_dim/32),(n_rows/32), batch]. All operands f32.
    let gemm_b = |weight: &Tensor,
                  input: &Tensor,
                  out: &Tensor,
                  m: usize, // n_rows
                  n: usize, // out_dim
                  k: usize, // in_dim
                  w_stride: usize,
                  x_stride: usize,
                  o_stride: usize|
     -> Result<()> {
        let kern = cached_ir("ffai_gemm_batched", DType::F32, || {
            let mut k = ffai_kernels_std::kernels::gemm::dense::mt_gemm_batched::kernel_ir_for(DType::F32);
            k.mode = KernelMode::Reduction;
            k
        });
        let bindings = vec![
            Binding::Buffer(weight.buffer.clone()),
            Binding::Buffer(input.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(k as u32),        // in_dim
            u(n as u32),        // out_dim
            u(m as u32),        // n_rows
            u(w_stride as u32),
            u(x_stride as u32),
            u(o_stride as u32),
        ];
        let grid = Grid {
            grid: [
                (n as u32).div_ceil(32),
                (m as u32).div_ceil(32),
                bhc as u32,
            ],
            block: [1024, 1, 1],
        };
        dev.dispatch(&kern, &bindings, grid)
    };

    // Fused gather-GEMM dispatch (ssd_g1_cb / ssd_g4_cs). Reduction-mode,
    // grid [(out_dim/32),(n_rows/32), bhc]; bindings vary per kernel (broadcast
    // index math is baked in-kernel), so the caller passes the binding list.
    let gemm_fused = |name: &str, bindings: Vec<Binding>, n: usize, m: usize| -> Result<()> {
        let kern = cached_ir(name, DType::F32, || {
            let mut k = build_ssd_ir(name);
            k.mode = KernelMode::Reduction;
            k
        });
        let grid = Grid {
            grid: [(n as u32).div_ceil(32), (m as u32).div_ceil(32), bhc as u32],
            block: [1024, 1, 1],
        };
        dev.dispatch(&kern, &bindings, grid)
    };

    // Optional per-phase profiler (env SSD_PHASE_PROF=1): synchronize + time
    // each phase, bucketed into elementwise / intra-GEMM / inter-recur /
    // state-GEMM. Zero cost when the env var is unset (default).
    let phase_prof = std::env::var("SSD_PHASE_PROF").is_ok();
    let t_pre = std::cell::Cell::new(0.0f64); // prologue elementwise (lcs/gather/xt)
    let t_lcs = std::cell::Cell::new(0.0f64); // sub: lcs cumsum
    let t_gbc = std::cell::Cell::new(0.0f64); // sub: gather_bc
    let t_xt = std::cell::Cell::new(0.0f64); // sub: xt transpose
    let t_g1 = std::cell::Cell::new(0.0f64); // CB = C·Bᵀ  (intra)
    let t_mmask = std::cell::Cell::new(0.0f64); // M = CB⊙mask (elementwise)
    let t_g2 = std::cell::Cell::new(0.0f64); // y_intra = M·x (intra)
    let t_bdt = std::cell::Cell::new(0.0f64); // BdT (elementwise)
    let t_g3 = std::cell::Cell::new(0.0f64); // S_chunk = BdT·x (state)
    let t_recur = std::cell::Cell::new(0.0f64); // serial inter-chunk recurrence
    let t_g4 = std::cell::Cell::new(0.0f64); // CS = C·S_in (state)
    let t_comb = std::cell::Cell::new(0.0f64); // combine (elementwise)
    macro_rules! phase {
        ($cell:expr, $body:expr) => {{
            if phase_prof {
                dev.synchronize()?;
                let t0 = std::time::Instant::now();
                let r = $body;
                dev.synchronize()?;
                $cell.set($cell.get() + t0.elapsed().as_secs_f64() * 1e6);
                r
            } else {
                $body
            }
        }};
    }

    // 1. Lcs cumsum: one thread per (chunk, head).
    phase!(t_lcs, elem(
        "ssd_lcs",
        &[dt, a_log, &lcs],
        &[u(tt), u(hh), u(ll), u(ncn)],
        bhc,
    )?);
    // 2. (gather_bc REMOVED — fused into G1/G4 as broadcast tile-loads.)
    // 3. Transpose x → xt [nc*H, dh, L].
    phase!(t_xt, elem(
        "ssd_xt",
        &[x, &xt],
        &[u(tt), u(hh), u(dhd), u(ll), u(ncn)],
        bhc * dhu * l,
    )?);

    // ── G1+mmask FUSED: M = (C·Bᵀ) ⊙ decay-mask  [L,L]. ──────────────────
    // ssd_g1_cb reads B/C straight from [T,G,ds] (8× head broadcast folded
    // into the tile load → no b_g/c_g), computes CB[i,j]=Σ_s C[t_i,g,s]·
    // B[t_j,g,s], and applies the causal decay·dt mask in the epilogue,
    // writing M directly (no CB scratch, no separate mmask round-trip).
    phase!(t_g1, gemm_fused(
        "ssd_g1_cb",
        vec![
            Binding::Buffer(b_mat.buffer.clone()),
            Binding::Buffer(c_mat.buffer.clone()),
            Binding::Buffer(lcs.buffer.clone()),
            Binding::Buffer(dt.buffer.clone()),
            Binding::Buffer(mmask.buffer.clone()),
            u(tt), u(hh), u(gg), u(hpgn), u(ll), u(dsd),
        ],
        l, // out_dim = L
        l, // n_rows  = L
    )?);

    // ── G2: y_intra = M · x  [L,dh]. out[i,p]=Σ_j mmask[i,j]·xt[p,j],
    //        weight=xt[dh,L], input=mmask[L,L], k=L. ─────────────────────────
    phase!(t_g2, gemm_b(&xt, &mmask, &y_intra, l, dhu, l, dhu * l, l * l, l * dhu)?);

    // 5. BdT[s,j] = decay·dt·B  [ds,L].
    phase!(t_bdt, elem(
        "ssd_bdt",
        &[b_mat, &lcs, dt, &bdt],
        &[u(tt), u(hh), u(gg), u(hpgn), u(ll), u(dsd), u(ncn)],
        bhc * dsu * l,
    )?);

    // ── G3: S_chunk = BdT · xt-form  [ds,dh]. out[s,p]=Σ_j bdt[s,j]·xt[p,j],
    //        weight=xt[dh,L], input=bdt[ds,L], k=L. ─────────────────────────
    phase!(t_g3, gemm_b(&xt, &bdt, &s_chunk, dsu, dhu, l, dhu * l, dsu * l, dsu * dhu)?);

    // 6. Serial inter-chunk recurrence → SinT [nc*H, dh, ds] + state_out.
    phase!(t_recur, elem(
        "ssd_recur",
        &[&s_chunk, &lcs, state_in, &sin_t, &state_out],
        &[u(hh), u(dhd), u(dsd), u(ll), u(ncn)],
        h * dsu * dhu,
    )?);

    // ── G4 FUSED: CS = C · S_in  [L,dh]. out[i,p]=Σ_s C[t_i,g,s]·sin_t[p,s].
    // ssd_g4_cs reads C straight from [T,G,ds] (broadcast tile-load → no c_g);
    // sin_t is already per-batch [nc*H,dh,ds]. weight=sin_t, input=C, k=ds.
    phase!(t_g4, gemm_fused(
        "ssd_g4_cs",
        vec![
            Binding::Buffer(sin_t.buffer.clone()),
            Binding::Buffer(c_mat.buffer.clone()),
            Binding::Buffer(cs.buffer.clone()),
            u(tt), u(hh), u(gg), u(hpgn), u(ll), u(dsd), u(dhd),
        ],
        dhu, // out_dim = dh
        l,   // n_rows  = L
    )?);

    // 7. Combine → y [T,H,dh] f32.
    phase!(t_comb, elem(
        "ssd_combine",
        &[&y_intra, &cs, &lcs, x, d_skip, &y],
        &[u(tt), u(hh), u(dhd), u(ll), u(ncn)],
        bhc * l * dhu,
    )?);

    if phase_prof {
        eprintln!("    [SSD pre-sub L={l}] lcs={:.1} gather_bc={:.1}(fused) xt={:.1}", t_lcs.get(), t_gbc.get(), t_xt.get());
        t_pre.set(t_lcs.get() + t_gbc.get() + t_xt.get());
        let intra = t_g1.get() + t_g2.get();
        let state = t_g3.get() + t_g4.get();
        let elemw = t_pre.get() + t_mmask.get() + t_bdt.get() + t_comb.get();
        let tot = intra + state + elemw + t_recur.get();
        eprintln!(
            "    [SSD phase L={l} nc={nc}] intra-GEMM(G1+G2)={:.1}us({:.0}%) state-GEMM(G3+G4)={:.1}us({:.0}%) elementwise={:.1}us({:.0}%) inter-recur={:.1}us({:.0}%) | G1(CB)={:.1} G2(y)={:.1} G3(Sc)={:.1} G4(CS)={:.1} pre={:.1} mmask={:.1} bdt={:.1} comb={:.1}",
            intra, intra / tot * 100.0, state, state / tot * 100.0,
            elemw, elemw / tot * 100.0, t_recur.get(), t_recur.get() / tot * 100.0,
            t_g1.get(), t_g2.get(), t_g3.get(), t_g4.get(),
            t_pre.get(), t_mmask.get(), t_bdt.get(), t_comb.get(),
        );
    }

    Ok((state_out, y))
}

/// Resolve one of the `ssd_*` portable elementwise kernels by name. They are
/// f32-fixed (`kernel_ir_for()` with no dtype arg), Grid3D-dispatched.
fn build_ssd_ir(name: &str) -> ffai_core::Kernel {
    use ffai_kernels_std::kernels::ssm;
    match name {
        "ssd_lcs" => ssm::scan::mt_ssd_lcs::kernel_ir_for(),
        "ssd_gather_bc" => ssm::scan::mt_ssd_gather_bc::kernel_ir_for(),
        "ssd_xt" => ssm::scan::mt_ssd_xt::kernel_ir_for(),
        "ssd_mmask" => ssm::scan::mt_ssd_mmask::kernel_ir_for(),
        "ssd_bdt" => ssm::scan::mt_ssd_bdt::kernel_ir_for(),
        "ssd_recur" => ssm::scan::mt_ssd_recur::kernel_ir_for(),
        "ssd_combine" => ssm::scan::mt_ssd_combine::kernel_ir_for(),
        "ssd_g1_cb" => ssm::scan::mt_ssd_g1_cb::kernel_ir_for(),
        "ssd_g4_cs" => ssm::scan::mt_ssd_g4_cs::kernel_ir_for(),
        other => panic!("build_ssd_ir: unknown ssd kernel {other}"),
    }
}
