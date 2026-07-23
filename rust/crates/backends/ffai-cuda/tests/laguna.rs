// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Laguna-S-2.1 decode smoke on CUDA. Env-gated: set FFAI_LAGUNA_GGUF to the
//! model's GGUF path (the 71G checkpoint), otherwise the test skips.
use ffai_cuda::CudaDevice;
#[test]
fn laguna_host_dequant() {
    ffai_modeltests::verify_laguna_host_dequant();
}

#[test]
fn laguna_host_layer0() {
    ffai_modeltests::verify_laguna_host_layer0();
}

/// Isolated rope_yarn_partial check with synthetic data. At position 0 the
/// rotation angle is zero for every dim, so rotated dims must equal
/// input * mscale (mscale = attn_factor exactly; the wrapper applies no
/// internal re-multiply) and passthrough dims must be bit-identical.
#[test]
fn laguna_rope_yarn_unit() {
    use ffai_core::{DType, Tensor};
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
    let out = ffai_ops::rope_yarn_partial(d, &t, 0, n_rot, 500000.0, freq_scale, 1.0, attn_factor, 32.0, 1.0, 8192).unwrap();
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
    let out2 = ffai_ops::rope_yarn_partial(d, &t2, 0, 64, 500000.0, freq_scale, 1.0, attn_factor, 32.0, 1.0, 8192).unwrap();
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
    use ffai_core::{DType, Tensor};
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
    let out = ffai_ops::gate_softplus_mul_perhead(d, &ta, &tg).unwrap();
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

/// `sdpa_decode_nbuf` (the CUDA-graph-capturable raw-CUDA fallback used by
/// Laguna's decode path under `FFAI_LAGUNA_GRAPH=1`) must match the registry
/// `ops::sdpa_decode` kernel's output — same dense-causal single-pass
/// online-softmax math, just reading `n_kv` from a device buffer at kernel
/// runtime instead of a launch-time baked-in scalar. Checked at `n_kv` in
/// {1, 7, 512} (kv_stride fixed at 512) x `heads_per_group` in {6, 9} —
/// Laguna's actual GQA ratios (48 full-attention / 72 sliding-window Q heads
/// over 8 KV heads).
#[test]
fn laguna_sdpa_decode_nbuf_matches_registry() {
    use ffai_core::{DType, Tensor};
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
            let expect = ffai_ops::sdpa_decode(d, &tq, &tk, &tv, hd, n_kv, kv_stride, heads_per_group as u32, scale).unwrap();
            let mut eb = vec![0u8; n_q_heads * hd * 4];
            d.synchronize().unwrap();
            d.download(expect.buffer.as_ref(), &mut eb).unwrap();
            let ev: Vec<f32> = eb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

            let n_kv_buf = Tensor::new(d.upload(&n_kv.to_le_bytes()).unwrap(), vec![1], DType::U32);
            let got = ffai_ops::sdpa_decode_nbuf(d, &tq, &tk, &tv, hd, &n_kv_buf, kv_stride, heads_per_group as u32, scale).unwrap();
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

/// `ffai_moe_gather_q4_swiglu` (the fused gate+up+SwiGLU MoE gather kernel
/// used by Laguna's decode path under `FFAI_LAGUNA_FUSE=1`) must match the
/// unfused compose: `moe_gather_q4` run twice (gate stack, up stack) then a
/// host-side SwiGLU on the downloaded pair. Synthetic 4-expert stack, top_k
/// 3, inter 8, hid 32 (small but exercises the same Q4 block-32 / f16
/// per-block-scale layout the real Laguna MoE stacks use). Modeled on
/// `laguna_gate_unit` above: synthetic buffers, upload, compare with a
/// relative tolerance.
#[test]
fn laguna_moe_swiglu_fused_unit() {
    use ffai_core::{DType, Tensor};
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

    let (gate_qs, gate_sc) = ffai_ops::quantize_q4(&gate_w, n_exp * inter, hid);
    let (up_qs, up_sc) = ffai_ops::quantize_q4(&up_w, n_exp * inter, hid);

    let gate_qs_t = Tensor::new(d.upload(&u32_bytes(&gate_qs)).unwrap(), vec![gate_qs.len()], DType::U32);
    let gate_sc_t = Tensor::new(d.upload(&f16_bytes(&gate_sc)).unwrap(), vec![gate_sc.len()], DType::F16);
    let up_qs_t = Tensor::new(d.upload(&u32_bytes(&up_qs)).unwrap(), vec![up_qs.len()], DType::U32);
    let up_sc_t = Tensor::new(d.upload(&f16_bytes(&up_sc)).unwrap(), vec![up_sc.len()], DType::F16);
    let x_t = Tensor::new(d.upload(&f32_bytes(&x)).unwrap(), vec![hid], DType::F32);

    let idx: Vec<u32> = vec![2, 0, 3];
    assert_eq!(idx.len(), top_k);
    let idx_t = Tensor::new(d.upload(&u32_bytes(&idx)).unwrap(), vec![top_k], DType::U32);

    // Fused: one dispatch, gate+up+silu(gate)*up in the kernel.
    let fused = ffai_ops::moe_gather_q4_swiglu(d, &gate_qs_t, &gate_sc_t, &up_qs_t, &up_sc_t, &x_t, &idx_t, top_k, inter, hid).unwrap();
    let mut fb = vec![0u8; top_k * inter * 4];
    d.synchronize().unwrap();
    d.download(fused.buffer.as_ref(), &mut fb).unwrap();
    let fv: Vec<f32> = fb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    // Unfused compose: two plain gathers + host-side SwiGLU on the download.
    let gates = ffai_ops::moe_gather_q4(d, &gate_qs_t, &gate_sc_t, &x_t, &idx_t, top_k, inter, hid).unwrap();
    let ups = ffai_ops::moe_gather_q4(d, &up_qs_t, &up_sc_t, &x_t, &idx_t, top_k, inter, hid).unwrap();
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
    assert_eq!(bad, 0, "ffai_moe_gather_q4_swiglu wrong vs unfused compose ({bad} bad elems)");
    eprintln!("moe gather q4 fused swiglu vs unfused compose: OK");
}

#[test]
fn laguna_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    ffai_modeltests::verify_laguna(dev.as_ref(), "GB10 sm_121");
}
