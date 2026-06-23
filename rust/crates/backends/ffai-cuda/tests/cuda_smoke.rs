// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Prove the shared engine seam runs a kernel end-to-end on real CUDA
//! hardware: build a metaltile `Kernel` IR, then drive it entirely through
//! the backend-neutral `ffai_core::Device` trait (alloc / upload / dispatch
//! / download / synchronize) — no CUDA-specific code in sight. This is the
//! concrete proof that CUDA consumes the shared layer.
//!
//! Runs only with `--features cuda` on a CUDA host. Skips (no failure) when
//! no device is present.
#![cfg(feature = "cuda")]

use ffai_core::{Binding, DType, Grid, Tensor};
use ffai_cuda::CudaDevice;
use metaltile_core::{
    ir::{BinOpKind, IndexExpr, Kernel, Op, Param, ParamKind, ValueId},
    shape::Shape,
};

fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// out[i] = a[i] + b[i] — Elementwise f32.
fn vector_add_ir() -> Kernel {
    let mut k = Kernel::new("vector_add");
    for (name, is_out) in [("a", false), ("b", false), ("c", true)] {
        k.params.push(Param {
            name: name.into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.name_value(ValueId::new(0), "idx");
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
        Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
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

#[test]
fn vector_add_through_shared_device_trait() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };
    eprintln!("device: {}", dev.name());

    const N: usize = 4096;
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();

    let abuf = dev.upload(&to_bytes(&a)).unwrap();
    let bbuf = dev.upload(&to_bytes(&b)).unwrap();
    let cbuf = dev.alloc(N * 4).unwrap();

    let k = vector_add_ir();
    let grid = Grid::d1((N as u32).div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[Binding::Buffer(abuf), Binding::Buffer(bbuf), Binding::Buffer(cbuf.clone())],
        grid,
    )
    .unwrap();
    dev.synchronize().unwrap();

    let mut out = vec![0u8; N * 4];
    dev.download(cbuf.as_ref(), &mut out).unwrap();
    let c = from_bytes(&out);

    let mut max_err = 0.0f32;
    for i in 0..N {
        max_err = max_err.max((c[i] - (a[i] + b[i])).abs());
    }
    assert!(max_err <= 1e-6, "vector_add via ffai Device mismatch: max|Δ|={max_err:.3e}");
    eprintln!("vector_add through ffai_core::Device on CUDA: OK (max|Δ|={max_err:.1e})");
}

/// Drive the real op layer (`ffai_ops::add` / `mul`) on CUDA — the same
/// calls model code makes, executed through the shared Device trait.
#[test]
fn ffai_ops_elementwise_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    const N: usize = 2048;
    let a: Vec<f32> = (0..N).map(|i| (i % 17) as f32 - 8.0).collect();
    let b: Vec<f32> = (0..N).map(|i| (i % 5) as f32 + 1.0).collect();

    let ta = Tensor::new(dev.upload(&to_bytes(&a)).unwrap(), vec![N], DType::F32);
    let tb = Tensor::new(dev.upload(&to_bytes(&b)).unwrap(), vec![N], DType::F32);

    let sum = ffai_ops::add(dev.as_ref(), &ta, &tb).unwrap();
    let prod = ffai_ops::mul(dev.as_ref(), &ta, &tb).unwrap();
    dev.synchronize().unwrap();

    let mut sbytes = vec![0u8; N * 4];
    let mut pbytes = vec![0u8; N * 4];
    dev.download(sum.buffer.as_ref(), &mut sbytes).unwrap();
    dev.download(prod.buffer.as_ref(), &mut pbytes).unwrap();
    let s = from_bytes(&sbytes);
    let p = from_bytes(&pbytes);

    let mut err = 0.0f32;
    for i in 0..N {
        err = err.max((s[i] - (a[i] + b[i])).abs());
        err = err.max((p[i] - (a[i] * b[i])).abs());
    }
    assert!(err <= 1e-6, "ffai_ops add/mul on CUDA mismatch: max|Δ|={err:.3e}");
    eprintln!("ffai_ops::add + ffai_ops::mul on CUDA: OK (max|Δ|={err:.1e})");
}

/// Heavier ops via the registered-kernel lookup: rms_norm (mt_rms_norm) and
/// gemv (mt_gemv), driven through the shared Device trait on CUDA and
/// checked against a CPU reference. This is the mechanism every transformer
/// op rides on.
#[test]
fn ffai_ops_rms_norm_and_gemv_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    // ── rms_norm: [rows, n] ──────────────────────────────────────────
    const ROWS: usize = 4;
    const N: usize = 512;
    let x: Vec<f32> = (0..ROWS * N).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let w: Vec<f32> = (0..N).map(|j| 1.0 + (j % 7) as f32 * 0.05).collect();
    let eps = 1e-5f32;

    let tx = Tensor::new(dev.upload(&to_bytes(&x)).unwrap(), vec![ROWS, N], DType::F32);
    let tw = Tensor::new(dev.upload(&to_bytes(&w)).unwrap(), vec![N], DType::F32);
    let ty = ffai_ops::rms_norm(dev.as_ref(), &tx, &tw, eps).unwrap();
    dev.synchronize().unwrap();
    let mut yb = vec![0u8; ROWS * N * 4];
    dev.download(ty.buffer.as_ref(), &mut yb).unwrap();
    let y = from_bytes(&yb);

    let mut rms_err = 0.0f32;
    for r in 0..ROWS {
        let row = &x[r * N..(r + 1) * N];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / N as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for j in 0..N {
            let want = row[j] * scale * w[j];
            rms_err = rms_err.max((y[r * N + j] - want).abs());
        }
    }
    assert!(rms_err <= 1e-4, "rms_norm on CUDA mismatch: max|Δ|={rms_err:.3e}");
    eprintln!("ffai_ops::rms_norm (mt_rms_norm) on CUDA: OK (max|Δ|={rms_err:.1e})");

    // ── gemv: [M,K] @ [K] ────────────────────────────────────────────
    const M: usize = 64;
    const K: usize = 512;
    let mat: Vec<f32> = (0..M * K).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
    let vecd: Vec<f32> = (0..K).map(|j| ((j % 9) as f32 - 4.0) * 0.05).collect();

    let tmat = Tensor::new(dev.upload(&to_bytes(&mat)).unwrap(), vec![M, K], DType::F32);
    let tvec = Tensor::new(dev.upload(&to_bytes(&vecd)).unwrap(), vec![K], DType::F32);
    let tout = ffai_ops::gemv(dev.as_ref(), &tmat, &tvec).unwrap();
    dev.synchronize().unwrap();
    let mut ob = vec![0u8; M * 4];
    dev.download(tout.buffer.as_ref(), &mut ob).unwrap();
    let got = from_bytes(&ob);

    let mut gemv_err = 0.0f32;
    for r in 0..M {
        let want: f32 = (0..K).map(|j| mat[r * K + j] * vecd[j]).sum();
        gemv_err = gemv_err.max((got[r] - want).abs());
    }
    assert!(gemv_err <= 1e-3, "gemv on CUDA mismatch: max|Δ|={gemv_err:.3e}");
    eprintln!("ffai_ops::gemv (mt_gemv) on CUDA: OK (max|Δ|={gemv_err:.1e})");
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The remaining transformer ops via registered kernels: silu, swiglu,
/// gather (embedding), softmax — all through the shared Device trait on CUDA.
#[test]
fn ffai_ops_transformer_ops_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    // ── silu ─────────────────────────────────────────────────────────
    let g: Vec<f32> = (0..1024).map(|i| (i % 21) as f32 * 0.1 - 1.0).collect();
    let tg = Tensor::new(dev.upload(&to_bytes(&g)).unwrap(), vec![1024], DType::F32);
    let ts = ffai_ops::silu(dev.as_ref(), &tg).unwrap();
    dev.synchronize().unwrap();
    let mut sb = vec![0u8; 1024 * 4];
    dev.download(ts.buffer.as_ref(), &mut sb).unwrap();
    let s = from_bytes(&sb);
    let mut e = 0.0f32;
    for i in 0..1024 {
        e = e.max((s[i] - g[i] * sigmoid(g[i])).abs());
    }
    assert!(e <= 1e-5, "silu mismatch: {e:.2e}");
    eprintln!("ffai_ops::silu (mt_silu) on CUDA: OK (max|Δ|={e:.1e})");

    // ── swiglu ───────────────────────────────────────────────────────
    let up: Vec<f32> = (0..1024).map(|i| (i % 13) as f32 * 0.05).collect();
    let tu = Tensor::new(dev.upload(&to_bytes(&up)).unwrap(), vec![1024], DType::F32);
    let tw = ffai_ops::swiglu(dev.as_ref(), &tg, &tu).unwrap();
    dev.synchronize().unwrap();
    let mut wb = vec![0u8; 1024 * 4];
    dev.download(tw.buffer.as_ref(), &mut wb).unwrap();
    let w = from_bytes(&wb);
    e = 0.0;
    for i in 0..1024 {
        e = e.max((w[i] - g[i] * sigmoid(g[i]) * up[i]).abs());
    }
    assert!(e <= 1e-5, "swiglu mismatch: {e:.2e}");
    eprintln!("ffai_ops::swiglu (mt_swiglu) on CUDA: OK (max|Δ|={e:.1e})");

    // ── gather (embedding) ───────────────────────────────────────────
    const VOCAB: usize = 8;
    const DIM: usize = 512;
    let table: Vec<f32> = (0..VOCAB * DIM).map(|i| i as f32 * 0.001).collect();
    let ids: [u32; 4] = [2, 5, 0, 7];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tt = Tensor::new(dev.upload(&to_bytes(&table)).unwrap(), vec![VOCAB, DIM], DType::F32);
    let ti = Tensor::new(dev.upload(&id_bytes).unwrap(), vec![4], DType::U32);
    let tge = ffai_ops::gather(dev.as_ref(), &tt, &ti).unwrap();
    dev.synchronize().unwrap();
    let mut gb = vec![0u8; 4 * DIM * 4];
    dev.download(tge.buffer.as_ref(), &mut gb).unwrap();
    let go = from_bytes(&gb);
    e = 0.0;
    for (t, &id) in ids.iter().enumerate() {
        for d in 0..DIM {
            e = e.max((go[t * DIM + d] - table[id as usize * DIM + d]).abs());
        }
    }
    assert!(e == 0.0, "gather mismatch: {e:.2e}");
    eprintln!("ffai_ops::gather (ffai_gather) on CUDA: OK (exact)");

    // ── softmax ──────────────────────────────────────────────────────
    const ROWS: usize = 2;
    const NW: usize = 1024;
    let xs: Vec<f32> = (0..ROWS * NW).map(|i| ((i % 37) as f32 - 18.0) * 0.1).collect();
    let txs = Tensor::new(dev.upload(&to_bytes(&xs)).unwrap(), vec![ROWS, NW], DType::F32);
    let tsm = ffai_ops::softmax(dev.as_ref(), &txs).unwrap();
    dev.synchronize().unwrap();
    let mut smb = vec![0u8; ROWS * NW * 4];
    dev.download(tsm.buffer.as_ref(), &mut smb).unwrap();
    let sm = from_bytes(&smb);
    e = 0.0;
    for r in 0..ROWS {
        let row = &xs[r * NW..(r + 1) * NW];
        let m = row.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for j in 0..NW {
            e = e.max((sm[r * NW + j] - exps[j] / sum).abs());
        }
    }
    assert!(e <= 1e-5, "softmax mismatch: {e:.2e}");
    eprintln!("ffai_ops::softmax (mt_softmax) on CUDA: OK (max|Δ|={e:.1e})");
}

/// Decode-time attention (sdpa_decode) on CUDA vs a CPU reference, single
/// q-head / single kv-head, head_dim=64. The gatekeeper op for a real
/// transformer forward.
#[test]
fn ffai_ops_sdpa_decode_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    const HD: usize = 64;
    const NKV: usize = 128;
    let scale = 1.0f32 / (HD as f32).sqrt();

    let q: Vec<f32> = (0..HD).map(|d| ((d % 9) as f32 - 4.0) * 0.05).collect();
    let kc: Vec<f32> = (0..NKV * HD).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
    let vc: Vec<f32> = (0..NKV * HD).map(|i| ((i % 11) as f32 - 5.0) * 0.03).collect();

    let tq = Tensor::new(dev.upload(&to_bytes(&q)).unwrap(), vec![1, HD], DType::F32);
    let tk = Tensor::new(dev.upload(&to_bytes(&kc)).unwrap(), vec![NKV, HD], DType::F32);
    let tv = Tensor::new(dev.upload(&to_bytes(&vc)).unwrap(), vec![NKV, HD], DType::F32);

    // n_kv=128, kv_stride=128 (cache exactly filled), heads_per_group=1.
    let tout =
        ffai_ops::sdpa_decode(dev.as_ref(), &tq, &tk, &tv, HD, NKV as u32, NKV as u32, 1, scale)
            .unwrap();
    dev.synchronize().unwrap();
    let mut ob = vec![0u8; HD * 4];
    dev.download(tout.buffer.as_ref(), &mut ob).unwrap();
    let got = from_bytes(&ob);

    // CPU reference: scores = scale·(q·k[t]); softmax; out = Σ p[t]·v[t].
    let mut scores = vec![0.0f32; NKV];
    for t in 0..NKV {
        let dot: f32 = (0..HD).map(|d| q[d] * kc[t * HD + d]).sum();
        scores[t] = scale * dot;
    }
    let m = scores.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut want = vec![0.0f32; HD];
    for t in 0..NKV {
        let p = exps[t] / sum;
        for d in 0..HD {
            want[d] += p * vc[t * HD + d];
        }
    }

    let mut err = 0.0f32;
    for d in 0..HD {
        err = err.max((got[d] - want[d]).abs());
    }
    assert!(err <= 1e-4, "sdpa_decode on CUDA mismatch: max|Δ|={err:.3e}");
    eprintln!("ffai_ops::sdpa_decode (ffai_sdpa_decode_d64) on CUDA: OK (max|Δ|={err:.1e})");
}
