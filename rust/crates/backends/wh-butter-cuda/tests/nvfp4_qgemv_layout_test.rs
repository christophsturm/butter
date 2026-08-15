// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! GPU correctness test for the NVFP4 checkpoint -> `iron` kernel layout
//! mapping documented in `NVFP4_LAYOUT.md`.
//!
//! Packs a small synthetic weight matrix on the host exactly the way
//! `wh-butter-loader::nvfp4::pack_nvfp4_host` says a modelopt checkpoint is
//! laid out (2× E2M1 nibbles/byte, E4M3 per-16-block scale, one f32 global),
//! uploads the raw bytes UNCHANGED (no host-side re-swizzle — that's the
//! claim under test), dispatches `wh_butter_ops::gemv_nvfp4` (which calls
//! iron's real `iron_nvfp4_qgemv` kernel), and compares against an f32
//! reference matvec computed by dequantizing the same packed bytes on the
//! host. This is the closest thing to an end-to-end proof, short of the real
//! checkpoint (blocked — see `NVFP4_LAYOUT.md`), that the loader's
//! `Nvfp4Tensor{packed, scales, scale2}` byte layout is exactly what the GPU
//! kernel expects.

use wh_butter_core::{DType, Device, Tensor};
use wh_butter_cuda::CudaDevice;
use wh_butter_loader::nvfp4::{e2m1_decode, e4m3_decode, pack_nvfp4_host};
use wh_butter_ops::gemv_nvfp4;

fn rng(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0f32; n];
    let mut s = seed;
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 3.0; // spread over the E2M1 grid's range
    }
    v
}

fn upload_bytes(d: &dyn Device, b: &[u8]) -> Tensor {
    Tensor::new(d.upload(b).unwrap(), vec![b.len()], DType::U8)
}

/// Host reference matvec through the exact decode `iron_nvfp4_qgemv` uses
/// (transcribed in `wh_butter_loader::nvfp4::{e2m1_decode, e4m3_decode}`,
/// which are themselves kept in lockstep with iron's `kernels/primitives.rs`
/// — see that module's doc comments).
fn host_ref_matvec(packed: &[u8], scales: &[u8], global: f32, x: &[f32], out_dim: usize, in_dim: usize, block_size: usize) -> Vec<f32> {
    let n_blocks = in_dim / block_size;
    let mut out = vec![0f32; out_dim];
    for row in 0..out_dim {
        let mut acc = 0f32;
        for col in 0..in_dim {
            let byte = packed[row * (in_dim / 2) + col / 2];
            let nibble = if col % 2 == 0 { byte & 0xf } else { byte >> 4 };
            let scale_byte = scales[row * n_blocks + col / block_size];
            let w = e2m1_decode(nibble) * e4m3_decode(scale_byte) * global;
            acc += w * x[col];
        }
        out[row] = acc;
    }
    out
}

#[test]
fn nvfp4_qgemv_matches_host_reference() {
    let Some(d) = CudaDevice::create().expect("cuda") else {
        eprintln!("no CUDA device visible, skip");
        return;
    };
    let d = d.as_ref();

    // Small but non-trivial shape: out_dim=48 (odd multiple of 32, so the
    // Reduction grid isn't a single warp-aligned coincidence), in_dim=64
    // (4 blocks of 16 -- exercises multi-block-per-row scale indexing).
    let out_dim = 48usize;
    let in_dim = 64usize;
    let block_size = 16usize;

    let w = rng(out_dim * in_dim, 11);
    let x = rng(in_dim, 22);

    let (packed, scales, global) = pack_nvfp4_host(&w, out_dim, in_dim, block_size);
    assert_eq!(packed.len(), out_dim * in_dim / 2);
    assert_eq!(scales.len(), out_dim * (in_dim / block_size));

    let expect = host_ref_matvec(&packed, &scales, global, &x, out_dim, in_dim, block_size);

    // Upload the RAW packed/scale bytes unchanged -- the point of the test is
    // that no host-side re-swizzle is needed (NVFP4_LAYOUT.md §3).
    let w_dev = upload_bytes(d, &packed);
    let s_dev = upload_bytes(d, &scales);
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_dev = Tensor::new(d.upload(&x_bytes).unwrap(), vec![in_dim], DType::F32);

    let out_dev = gemv_nvfp4(d, &w_dev, &s_dev, &x_dev, out_dim, in_dim, block_size, global).expect("gemv_nvfp4 dispatch");

    let mut out_bytes = vec![0u8; out_dim * 4];
    d.download(out_dev.buffer.as_ref(), &mut out_bytes).unwrap();
    let got: Vec<f32> = out_bytes.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let (mut max_abs, mut max_rel) = (0f32, 0f32);
    for (g, e) in got.iter().zip(expect.iter()) {
        let a = (g - e).abs();
        let rel = a / e.abs().max(1e-3);
        max_abs = max_abs.max(a);
        max_rel = max_rel.max(rel);
    }
    eprintln!("nvfp4_qgemv out_dim={out_dim} in_dim={in_dim} block_size={block_size}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
    // Both sides run the identical decode math on identical bytes -- this
    // should match to float rounding, not NVFP4 quantization error (that
    // error already happened once, in `pack_nvfp4_host`, and is baked into
    // both `expect` and what the kernel decodes). A generous 1e-2 rel bound
    // per the task brief; in practice this should land near float epsilon.
    assert!(max_rel < 1e-2, "nvfp4_qgemv: max_rel {max_rel:.3e} exceeds 1e-2 (max_abs {max_abs:.3e})");
    eprintln!("nvfp4_qgemv_matches_host_reference: PASS");
}
