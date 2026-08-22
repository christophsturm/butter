// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

use wh_butter_core::{DType, Device, Tensor};
use wh_butter_cuda::CudaDevice;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn download_f32(device: &dyn Device, tensor: &Tensor) -> Vec<f32> {
    let mut bytes = vec![0u8; tensor.elem_count() * 4];
    device.synchronize().unwrap();
    device.download(tensor.buffer.as_ref(), &mut bytes).unwrap();
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect()
}

fn e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 0x07;
    if exponent == 0 {
        sign * (mantissa as f32) * 2.0f32.powi(-9)
    } else {
        sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    }
}

#[test]
fn resident_fp8_kv_preserves_physical_stride() {
    let Some(device) = CudaDevice::create().expect("create CUDA device") else { return };
    let device = device.as_ref();
    let (n_q, n_kv, head_dim, tokens, cap, group) = (4usize, 2usize, 256usize, 5usize, 11usize, 16usize);
    let source = |salt: usize| -> Vec<f32> {
        (0..tokens * n_kv * head_dim)
            .map(|index| (((index * 37 + salt) % 257) as f32 - 128.0) / 41.0)
            .collect()
    };
    let k = source(11);
    let v = source(73);
    let q: Vec<f32> = (0..n_q * head_dim)
        .map(|index| (((index * 29 + 17) % 251) as f32 - 125.0) / 53.0)
        .collect();
    let positions: Vec<u8> = (0..tokens as u32).flat_map(|value| value.to_le_bytes()).collect();
    let make = |values: &[f32], shape: Vec<usize>| {
        Tensor::new(device.upload(&f32_bytes(values)).unwrap(), shape, DType::F32)
    };
    let positions = Tensor::new(device.upload(&positions).unwrap(), vec![tokens], DType::U32);
    let k_src = make(&k, vec![tokens, n_kv, head_dim]);
    let v_src = make(&v, vec![tokens, n_kv, head_dim]);
    let q = make(&q, vec![n_q, head_dim]);
    let code_count = n_kv * cap * head_dim;
    let scale_count = code_count / group;
    let k_codes = Tensor::new(device.alloc_zeroed(code_count).unwrap(), vec![n_kv, cap, head_dim], DType::U8);
    let v_codes = Tensor::new(device.alloc_zeroed(code_count).unwrap(), vec![n_kv, cap, head_dim], DType::U8);
    let k_scales = Tensor::new(device.alloc_zeroed(scale_count * 4).unwrap(), vec![n_kv, cap, head_dim / group], DType::F32);
    let v_scales = Tensor::new(device.alloc_zeroed(scale_count * 4).unwrap(), vec![n_kv, cap, head_dim / group], DType::F32);
    let sinks = Tensor::new(device.alloc_zeroed(n_q * 4).unwrap(), vec![n_q], DType::F32);
    wh_butter_ops::kv_append_fp8_e4m3_many(device, &k_src, &positions, &k_codes, &k_scales, n_kv, head_dim, cap, group).unwrap();
    wh_butter_ops::kv_append_fp8_e4m3_many(device, &v_src, &positions, &v_codes, &v_scales, n_kv, head_dim, cap, group).unwrap();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let run = |stride: usize| wh_butter_ops::sdpa_decode_fp8_e4m3(
        device, &q, &k_codes, &k_scales, &v_codes, &v_scales, &sinks,
        head_dim, tokens as u32, stride as u32, (n_q / n_kv) as u32, group as u32, scale,
    ).unwrap();
    let output = run(cap);
    device.synchronize().unwrap();
    let mut kc = vec![0u8; code_count];
    let mut vc = vec![0u8; code_count];
    device.download(k_codes.buffer.as_ref(), &mut kc).unwrap();
    device.download(v_codes.buffer.as_ref(), &mut vc).unwrap();
    let ks = download_f32(device, &k_scales);
    let vs = download_f32(device, &v_scales);
    let q_host = download_f32(device, &q);
    let actual = download_f32(device, &output);
    let mut expected = vec![0.0f32; n_q * head_dim];
    for qh in 0..n_q {
        let kvh = qh / (n_q / n_kv);
        let mut scores = vec![0.0f32; tokens];
        for token in 0..tokens {
            for dim in 0..head_dim {
                let row = (kvh * cap + token) * head_dim + dim;
                let sg = (kvh * cap + token) * (head_dim / group) + dim / group;
                scores[token] += q_host[qh * head_dim + dim] * e4m3(kc[row]) * ks[sg] * scale;
            }
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|score| (*score - max).exp()).collect();
        let denom: f32 = weights.iter().sum();
        for dim in 0..head_dim {
            for token in 0..tokens {
                let row = (kvh * cap + token) * head_dim + dim;
                let sg = (kvh * cap + token) * (head_dim / group) + dim / group;
                expected[qh * head_dim + dim] += weights[token] / denom * e4m3(vc[row]) * vs[sg];
            }
        }
    }
    let max_abs = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 2e-3, "resident FP8 attention max_abs={max_abs}");
    let wrong = download_f32(device, &run(tokens));
    let wrong_max = wrong.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(wrong_max > 0.1, "negative control missed collapsed KV stride: {wrong_max}");
}
