// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Laguna-S-2.1 decode smoke on CUDA. Env-gated: set BUTTER_LAGUNA_GGUF to the
//! model's GGUF path (the 71G checkpoint), otherwise the test skips.
use wh_butter_cuda::CudaDevice;
#[test]
fn laguna_host_dequant() {
    wh_butter_modeltests::verify_laguna_host_dequant();
}

#[test]
fn laguna_host_layer0() {
    wh_butter_modeltests::verify_laguna_host_layer0();
}

/// Isolated rope_yarn_partial check with synthetic data. At position 0 the
/// rotation angle is zero for every dim, so rotated dims must equal
/// input * mscale (mscale = attn_factor exactly; the wrapper applies no
/// internal re-multiply) and passthrough dims must be bit-identical.
#[test]
fn laguna_rope_yarn_unit() {
    use wh_butter_core::{DType, Tensor};
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let n_heads = 3usize;
    let hd = 8usize;
    let n_rot = 4u32;
    let n: usize = n_heads * hd;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = Tensor::new(d.upload(&xb).unwrap(), vec![n_heads, hd], DType::F32);

    let attn_factor = 1.4852030263919618f32;
    let freq_scale = 1.0f32 / 128.0;
    let out = wh_butter_ops::rope_yarn_partial(d, &t, 0, n_rot, 500000.0, freq_scale, 1.0, attn_factor, 32.0, 1.0, 8192).unwrap();
    let mut ob = vec![0u8; n * 4];
    d.synchronize().unwrap();
    d.download(out.buffer.as_ref(), &mut ob).unwrap();
    let o: Vec<f32> = ob.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    let mscale = attn_factor; // attn_factor is the final mscale, no internal re-multiply
    let mut bad = 0;
    for h in 0..n_heads {
        for i in 0..hd {
            let idx = h * hd + i;
            let expect = if (i as u32) < n_rot { x[idx] * mscale } else { x[idx] };
            if (o[idx] - expect).abs() > 1e-4 * expect.abs().max(1.0) {
                eprintln!("pos0 MISMATCH h={h} i={i}: got {} expect {expect}", o[idx]);
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "rope_yarn_partial wrong at position 0 ({bad} bad elems)");
    eprintln!("rope pos-0 identity*mscale: OK (mscale={mscale})");

    // Exact model geometry: 48 heads x 128 dims, n_rot 64, position 0. The
    // same identity*mscale invariant must hold; this catches launch-geometry
    // bugs the tiny shape above cannot (block y = 64, grid x = 48).
    let n_heads2 = 48usize;
    let hd2 = 128usize;
    let n2 = n_heads2 * hd2;
    let x2: Vec<f32> = (0..n2).map(|i| ((i * 2654435761usize) % 1000) as f32 * 0.002 - 1.0).collect();
    let xb2: Vec<u8> = x2.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t2 = Tensor::new(d.upload(&xb2).unwrap(), vec![n_heads2, hd2], DType::F32);
    let out2 = wh_butter_ops::rope_yarn_partial(d, &t2, 0, 64, 500000.0, freq_scale, 1.0, attn_factor, 32.0, 1.0, 8192).unwrap();
    let mut ob2 = vec![0u8; n2 * 4];
    d.synchronize().unwrap();
    d.download(out2.buffer.as_ref(), &mut ob2).unwrap();
    let o2: Vec<f32> = ob2.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut bad2 = 0;
    for h in 0..n_heads2 {
        for i in 0..hd2 {
            let idx = h * hd2 + i;
            let expect = if i < 64 { x2[idx] * mscale } else { x2[idx] };
            if (o2[idx] - expect).abs() > 1e-4 * expect.abs().max(1.0) {
                if bad2 < 6 { eprintln!("model-geom pos0 MISMATCH h={h} i={i}: got {} expect {expect}", o2[idx]); }
                bad2 += 1;
            }
        }
    }
    assert_eq!(bad2, 0, "rope_yarn_partial wrong at model geometry pos 0 ({bad2} bad elems)");
    eprintln!("rope model-geometry pos-0: OK");
}

/// Isolated per-head softplus gate check with synthetic data.
#[test]
fn laguna_gate_unit() {
    use wh_butter_core::{DType, Tensor};
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let n_heads = 4usize;
    let hd = 8usize;
    let attn: Vec<f32> = (0..n_heads * hd).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let g: Vec<f32> = vec![-2.0, 0.0, 1.5, 30.0];
    let up = |v: &[f32]| -> Tensor {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Tensor::new(d.upload(&b).unwrap(), vec![v.len()], DType::F32)
    };
    let ta = up(&attn).reshaped(vec![n_heads, hd]);
    let tg = up(&g);
    let out = wh_butter_ops::gate_softplus_mul_perhead(d, &ta, &tg).unwrap();
    let mut ob = vec![0u8; attn.len() * 4];
    d.synchronize().unwrap();
    d.download(out.buffer.as_ref(), &mut ob).unwrap();
    let o: Vec<f32> = ob.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut bad = 0;
    for h in 0..n_heads {
        let sp = if g[h] > 20.0 { g[h] } else { (1.0 + g[h].exp()).ln() };
        for i in 0..hd {
            let idx = h * hd + i;
            let expect = attn[idx] * sp;
            if (o[idx] - expect).abs() > 1e-5 * expect.abs().max(1.0) {
                eprintln!("gate MISMATCH h={h} i={i}: got {} expect {expect}", o[idx]);
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "gate_softplus_mul_perhead wrong ({bad} bad elems)");
    eprintln!("gate softplus per-head: OK");
}

/// Batched-prefill `rope_yarn_partial_many` must match a T-loop of the
/// scalar-position `rope_yarn_partial` (the proven decode kernel), one call
/// per token row at that row's absolute position. Exercises both a YaRN
/// full-attention-layer shape (n_rot < head_dim, ext_factor != 0) and a
/// plain sliding-layer shape (n_rot == head_dim, ext_factor == 0), matching
/// Laguna's two RoPE configs.
#[test]
fn laguna_rope_yarn_many_matches_looped() {
    use wh_butter_core::{DType, Tensor};
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let up = |v: &[f32]| -> Tensor {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Tensor::new(d.upload(&b).unwrap(), vec![v.len()], DType::F32)
    };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> {
        let mut b = vec![0u8; n * 4];
        d.synchronize().unwrap();
        d.download(t.buffer.as_ref(), &mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };

    // (n_rot, theta, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, n_orig_ctx, label)
    let configs: [(u32, f32, f32, f32, f32, f32, f32, u32, &str); 2] = [
        (64, 500000.0, 1.0 / 128.0, 1.0, 1.4852030263919618, 32.0, 1.0, 8192, "full/yarn"),
        (128, 10000.0, 1.0, 0.0, 1.0, 32.0, 1.0, 8192, "sliding/plain"),
    ];

    for (n_rot, theta, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, n_orig_ctx, label) in configs {
        let t = 5usize;
        let n_heads = 6usize;
        let hd = 128usize;
        let n = t * n_heads * hd;
        let x: Vec<f32> = (0..n).map(|i| ((i * 2654435761usize) % 2000) as f32 * 0.001 - 1.0).collect();
        let positions: Vec<u32> = (0..t).map(|i| (7 + i * 13) as u32).collect(); // non-contiguous, non-zero-based

        // Batched: one dispatch over [T, n_heads, hd].
        let xt = up(&x).reshaped(vec![t, n_heads, hd]);
        let posb = Tensor::new(
            d.upload(&positions.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>()).unwrap(),
            vec![t], DType::U32,
        );
        let got = wh_butter_ops::rope_yarn_partial_many(
            d, &xt, &posb, n_heads, n_rot, theta, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, n_orig_ctx,
        ).unwrap();
        let gv = dl(&got, n);

        // Looped reference: one rope_yarn_partial call per token row at its
        // own absolute position, over that row's [n_heads, hd] slab.
        let mut ev = vec![0f32; n];
        for (ti, &pos) in positions.iter().enumerate() {
            let row = &x[ti * n_heads * hd..(ti + 1) * n_heads * hd];
            let rt = up(row).reshaped(vec![n_heads, hd]);
            let ro = wh_butter_ops::rope_yarn_partial(
                d, &rt, pos, n_rot, theta, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, n_orig_ctx,
            ).unwrap();
            let rov = dl(&ro, n_heads * hd);
            ev[ti * n_heads * hd..(ti + 1) * n_heads * hd].copy_from_slice(&rov);
        }

        let mut bad = 0;
        for i in 0..n {
            if (gv[i] - ev[i]).abs() > 1e-4 * ev[i].abs().max(1.0) {
                if bad < 6 {
                    eprintln!("[{label}] MISMATCH i={i}: many={} looped={}", gv[i], ev[i]);
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "rope_yarn_partial_many mismatch vs looped rope_yarn_partial ({label}, {bad} bad elems)");
        eprintln!("rope_yarn_partial_many vs looped [{label}]: OK");
    }
}

/// Batched-prefill `gate_softplus_mul_perhead_many` must match a T-loop of
/// the scalar-row `gate_softplus_mul_perhead` (the proven decode kernel),
/// one call per token row.
#[test]
fn laguna_gate_many_matches_looped() {
    use wh_butter_core::{DType, Tensor};
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let up = |v: &[f32]| -> Tensor {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Tensor::new(d.upload(&b).unwrap(), vec![v.len()], DType::F32)
    };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> {
        let mut b = vec![0u8; n * 4];
        d.synchronize().unwrap();
        d.download(t.buffer.as_ref(), &mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };

    let t = 4usize;
    let n_heads = 5usize; // deliberately not 64-aligned (Laguna's real head counts: 48/72)
    let hd = 8usize;
    let n = t * n_heads * hd;
    let attn: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 2.0).collect();
    let g: Vec<f32> = (0..t * n_heads).map(|i| (i as f32) * 3.7 - 6.0).collect(); // spans <0, ~0, >20 branches

    let at = up(&attn).reshaped(vec![t, n_heads, hd]);
    let gt = up(&g);
    let got = wh_butter_ops::gate_softplus_mul_perhead_many(d, &at, &gt, n_heads).unwrap();
    let gv = dl(&got, n);

    let mut ev = vec![0f32; n];
    for ti in 0..t {
        let arow = &attn[ti * n_heads * hd..(ti + 1) * n_heads * hd];
        let grow = &g[ti * n_heads..(ti + 1) * n_heads];
        let art = up(arow).reshaped(vec![n_heads, hd]);
        let grt = up(grow);
        let ro = wh_butter_ops::gate_softplus_mul_perhead(d, &art, &grt).unwrap();
        let rov = dl(&ro, n_heads * hd);
        ev[ti * n_heads * hd..(ti + 1) * n_heads * hd].copy_from_slice(&rov);
    }

    let mut bad = 0;
    for i in 0..n {
        if (gv[i] - ev[i]).abs() > 1e-5 * ev[i].abs().max(1.0) {
            if bad < 6 {
                eprintln!("MISMATCH i={i}: many={} looped={}", gv[i], ev[i]);
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "gate_softplus_mul_perhead_many mismatch vs looped gate_softplus_mul_perhead ({bad} bad elems)");
    eprintln!("gate_softplus_mul_perhead_many vs looped: OK");
}

/// `sdpa_decode_nbuf` (the CUDA-graph-capturable raw-CUDA fallback used by
/// Laguna's decode path under `BUTTER_LAGUNA_GRAPH=1`) must match the registry
/// `ops::sdpa_decode` kernel's output — same dense-causal single-pass
/// online-softmax math, just reading `n_kv` from a device buffer at kernel
/// runtime instead of a launch-time baked-in scalar. Checked at `n_kv` in
/// {1, 7, 512} (kv_stride fixed at 512) x `heads_per_group` in {6, 9}:
/// Laguna's actual GQA ratios (48 full-attention / 72 sliding-window Q heads
/// over 8 KV heads).
#[test]
fn laguna_sdpa_decode_nbuf_matches_registry() {
    use wh_butter_core::{DType, Tensor};
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let hd = 128usize;
    let n_kv_heads = 4usize;
    let kv_stride = 512u32;
    let scale = 1.0 / (hd as f32).sqrt();

    let ramp = |n: usize, step: f32, start: f32| -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    };
    let up = |v: &[f32]| -> Tensor {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Tensor::new(d.upload(&b).unwrap(), vec![v.len()], DType::F32)
    };

    for &heads_per_group in &[6usize, 9usize] {
        let n_q_heads = n_kv_heads * heads_per_group;
        let q = ramp(n_q_heads * hd, 0.013, -0.4);
        let k = ramp(n_kv_heads * kv_stride as usize * hd, 0.011, -0.5);
        let v = ramp(n_kv_heads * kv_stride as usize * hd, 0.007, -0.3);
        let tq = up(&q).reshaped(vec![n_q_heads, hd]);
        let tk = up(&k).reshaped(vec![n_kv_heads, kv_stride as usize, hd]);
        let tv = up(&v).reshaped(vec![n_kv_heads, kv_stride as usize, hd]);

        for &n_kv in &[1u32, 7, 512] {
            let expect = wh_butter_ops::sdpa_decode(d, &tq, &tk, &tv, hd, n_kv, kv_stride, heads_per_group as u32, scale).unwrap();
            let mut eb = vec![0u8; n_q_heads * hd * 4];
            d.synchronize().unwrap();
            d.download(expect.buffer.as_ref(), &mut eb).unwrap();
            let ev: Vec<f32> = eb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

            let n_kv_buf = Tensor::new(d.upload(&n_kv.to_le_bytes()).unwrap(), vec![1], DType::U32);
            let got = wh_butter_ops::sdpa_decode_nbuf(d, &tq, &tk, &tv, hd, &n_kv_buf, kv_stride, heads_per_group as u32, scale).unwrap();
            let mut gb = vec![0u8; n_q_heads * hd * 4];
            d.synchronize().unwrap();
            d.download(got.buffer.as_ref(), &mut gb).unwrap();
            let gv: Vec<f32> = gb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

            let mut bad = 0;
            for i in 0..ev.len() {
                if (ev[i] - gv[i]).abs() > 1e-4 * ev[i].abs().max(1.0) {
                    if bad < 6 {
                        eprintln!("MISMATCH hpg={heads_per_group} n_kv={n_kv} i={i}: nbuf={} registry={}", gv[i], ev[i]);
                    }
                    bad += 1;
                }
            }
            assert_eq!(bad, 0, "sdpa_decode_nbuf mismatch vs registry sdpa_decode (heads_per_group={heads_per_group}, n_kv={n_kv}, {bad} bad elems)");
        }
    }
    eprintln!("sdpa_decode_nbuf matches registry sdpa_decode: OK");
}

/// `iron_moe_gather_q4_swiglu` (the fused gate+up+SwiGLU MoE gather kernel
/// used by Laguna's decode path under `BUTTER_LAGUNA_FUSE=1`) must match the
/// unfused compose: `moe_gather_q4` run twice (gate stack, up stack) then a
/// host-side SwiGLU on the downloaded pair. Synthetic 4-expert stack, top_k
/// 3, inter 8, hid 32 (small but exercises the same Q4 block-32 / f16
/// per-block-scale layout the real Laguna MoE stacks use). Modeled on
/// `laguna_gate_unit` above: synthetic buffers, upload, compare with a
/// relative tolerance.
///
/// TODO(rebrand-merge): `wh_butter_ops::moe_gather_q4` / `moe_gather_q4_swiglu`
/// dispatch `iron_moe_gather_q4` / `iron_moe_gather_q4_swiglu` via
/// `wh_iron_std`'s kernel IR, but the pinned `thewafflehaus/iron@dev` kernel
/// library only ships `iron_moe_gather_q4_relu2/_down/_down_accum` — the
/// plain (non-relu2) and `_swiglu` variants this test needs are not present
/// upstream yet. Ignored until those kernels land in the iron repo; do not
/// silently unignore without confirming they exist.
#[ignore = "blocked: iron_moe_gather_q4 / iron_moe_gather_q4_swiglu kernels not yet in thewafflehaus/iron@dev"]
#[test]
fn laguna_moe_swiglu_fused_unit() {
    use wh_butter_core::{DType, Tensor};
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();

    let n_exp = 4usize;
    let inter = 8usize;
    let hid = 32usize;
    let top_k = 3usize;

    // f32 -> f16 bit conversion, exactly as the Laguna GGUF loader converts
    // `quantize_q4`'s host f32 per-block scales before upload (the gather
    // kernels read scales as f16; uploading raw f32 bits would silently
    // reinterpret them as garbage f16s).
    fn half_bits(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 112;
        if e <= 0 { return sign; }
        if e >= 0x1f { return sign | 0x7c00; }
        let m = (x >> 13) & 0x3ff;
        let round = (x >> 12) & 1;
        let v = ((e as u32) << 10) | m;
        sign | ((v + round) as u16)
    }
    let f16_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&f| half_bits(f).to_le_bytes()).collect() };
    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let u32_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    let mut s = 0x9e3779b9u32;
    let mut rng = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as f32 / u32::MAX as f32) - 0.5 };

    let gate_w: Vec<f32> = (0..n_exp * inter * hid).map(|_| rng()).collect();
    let up_w: Vec<f32> = (0..n_exp * inter * hid).map(|_| rng()).collect();
    let x: Vec<f32> = (0..hid).map(|_| rng()).collect();

    let (gate_qs, gate_sc) = wh_butter_ops::quantize_q4(&gate_w, n_exp * inter, hid);
    let (up_qs, up_sc) = wh_butter_ops::quantize_q4(&up_w, n_exp * inter, hid);

    let gate_qs_t = Tensor::new(d.upload(&u32_bytes(&gate_qs)).unwrap(), vec![gate_qs.len()], DType::U32);
    let gate_sc_t = Tensor::new(d.upload(&f16_bytes(&gate_sc)).unwrap(), vec![gate_sc.len()], DType::F16);
    let up_qs_t = Tensor::new(d.upload(&u32_bytes(&up_qs)).unwrap(), vec![up_qs.len()], DType::U32);
    let up_sc_t = Tensor::new(d.upload(&f16_bytes(&up_sc)).unwrap(), vec![up_sc.len()], DType::F16);
    let x_t = Tensor::new(d.upload(&f32_bytes(&x)).unwrap(), vec![hid], DType::F32);

    let idx: Vec<u32> = vec![2, 0, 3];
    assert_eq!(idx.len(), top_k);
    let idx_t = Tensor::new(d.upload(&u32_bytes(&idx)).unwrap(), vec![top_k], DType::U32);

    // Fused: one dispatch, gate+up+silu(gate)*up in the kernel.
    let fused = wh_butter_ops::moe_gather_q4_swiglu(d, &gate_qs_t, &gate_sc_t, &up_qs_t, &up_sc_t, &x_t, &idx_t, top_k, inter, hid).unwrap();
    let mut fb = vec![0u8; top_k * inter * 4];
    d.synchronize().unwrap();
    d.download(fused.buffer.as_ref(), &mut fb).unwrap();
    let fv: Vec<f32> = fb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    // Unfused compose: two plain gathers + host-side SwiGLU on the download.
    let gates = wh_butter_ops::moe_gather_q4(d, &gate_qs_t, &gate_sc_t, &x_t, &idx_t, top_k, inter, hid).unwrap();
    let ups = wh_butter_ops::moe_gather_q4(d, &up_qs_t, &up_sc_t, &x_t, &idx_t, top_k, inter, hid).unwrap();
    let mut gb = vec![0u8; top_k * inter * 4];
    let mut ub = vec![0u8; top_k * inter * 4];
    d.synchronize().unwrap();
    d.download(gates.buffer.as_ref(), &mut gb).unwrap();
    d.download(ups.buffer.as_ref(), &mut ub).unwrap();
    let gv: Vec<f32> = gb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let uv: Vec<f32> = ub.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    let mut bad = 0;
    for i in 0..top_k * inter {
        let silu_g = gv[i] / (1.0 + (-gv[i]).exp());
        let expect = silu_g * uv[i];
        if (fv[i] - expect).abs() > 1e-3 * expect.abs().max(1.0) {
            eprintln!("swiglu-fused MISMATCH i={i}: fused={} expect={expect} (gate={} up={})", fv[i], gv[i], uv[i]);
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "iron_moe_gather_q4_swiglu wrong vs unfused compose ({bad} bad elems)");
    eprintln!("moe gather q4 fused swiglu vs unfused compose: OK");
}

/// `sdpa_multi_tc_varlen`'s new sliding-window block skip (`win != 0`) must
/// match the `win = 0` full-range/mask-only path exactly, the skip only
/// shrinks which query rows each KV block's GEMMs cover, the per-element
/// `seg_lo` mask inside `sdpa_tc_softmax_varlen` is still the correctness
/// source of truth either way, so both should reach the same result.
///
/// `T = 16` query rows, sliding window 4, but `base_kv = 4080` so this
/// chunk continues a much longer prefix (`kv_stride = n_kv = 4096`). That
/// makes `n_kv` exceed the kernel's fixed 2048-wide KV block (`bk =
/// 2048.min(n_kv)`), so the KV range splits into two blocks: block 0
/// `[0, 2048)` and block 1 `[2048, 4096)`. With window 4 every query's
/// `seg_lo` sits at `4077..4092`, entirely inside block 1, so under `win=4`
/// block 0 should be skipped ENTIRELY (q_cnt=0) while block 1 is computed
/// for every row, genuinely exercising the row-range skip, not just
/// threading the new parameter through as a no-op. Modeled on
/// `laguna_sdpa_decode_nbuf_matches_registry`'s synthetic-ramp-data +
/// download + relative-tolerance-compare harness.
#[test]
fn laguna_sdpa_varlen_window_skip_matches_full_range() {
    use wh_butter_core::{DType, Tensor};
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();

    let hd = 8usize;
    let n_kv_heads = 1usize;
    let heads_per_group = 2usize;
    let n_q_heads = n_kv_heads * heads_per_group;
    let t = 16usize;              // n_query (this chunk's rows)
    let base_kv = 4080u32;        // this chunk continues a long prefix
    let kv_stride = 4096u32;      // n_kv = base_kv + t = 4096 > bk(2048) => 2 blocks
    let window = 4u32;
    let scale = 1.0 / (hd as f32).sqrt();

    let ramp = |n: usize, step: f32, start: f32| -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    };
    let up = |v: &[f32]| -> Tensor {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Tensor::new(d.upload(&b).unwrap(), vec![v.len()], DType::F32)
    };

    let q = ramp(t * n_q_heads * hd, 0.013, -0.4);
    let k = ramp(n_kv_heads * kv_stride as usize * hd, 0.011, -0.5);
    let v = ramp(n_kv_heads * kv_stride as usize * hd, 0.007, -0.3);
    let tq = up(&q).reshaped(vec![t, n_q_heads, hd]);
    let tk = up(&k).reshaped(vec![n_kv_heads, kv_stride as usize, hd]);
    let tv = up(&v).reshaped(vec![n_kv_heads, kv_stride as usize, hd]);

    // Exactly Laguna's sliding-layer prefill convention:
    // seg_lo[i] = max(0, (base_kv+i) - (window-1)).
    let seg_lo_h: Vec<u32> = (0..t)
        .map(|i| ((base_kv as i64 + i as i64) - (window as i64 - 1)).max(0) as u32)
        .collect();
    let seg_lo = Tensor::new(
        d.upload(&seg_lo_h.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap(),
        vec![t], DType::U32,
    );

    let out_full = wh_butter_ops::sdpa_multi_tc_varlen(
        d, &tq, &tk, &tv, &seg_lo, 0, hd, n_q_heads as u32,
        base_kv, t as u32, kv_stride, heads_per_group as u32, true, scale, 0,
    ).unwrap();
    let out_win = wh_butter_ops::sdpa_multi_tc_varlen(
        d, &tq, &tk, &tv, &seg_lo, 0, hd, n_q_heads as u32,
        base_kv, t as u32, kv_stride, heads_per_group as u32, true, scale, window,
    ).unwrap();

    let n = t * n_q_heads * hd;
    let dl = |ten: &Tensor| -> Vec<f32> {
        let mut b = vec![0u8; n * 4];
        d.synchronize().unwrap();
        d.download(ten.buffer.as_ref(), &mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let fv = dl(&out_full);
    let wv = dl(&out_win);

    let mut bad = 0;
    for i in 0..n {
        if (fv[i] - wv[i]).abs() > 1e-4 * fv[i].abs().max(1.0) {
            if bad < 6 {
                eprintln!("MISMATCH i={i}: win=0 {} vs win={window} {}", fv[i], wv[i]);
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "sdpa_multi_tc_varlen window skip (win={window}) mismatch vs win=0 full-range ({bad} bad elems)");
    eprintln!("sdpa_multi_tc_varlen window skip vs full-range: OK");
}

#[test]
fn laguna_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    wh_butter_modeltests::verify_laguna(dev.as_ref(), "GB10 sm_121");
}
