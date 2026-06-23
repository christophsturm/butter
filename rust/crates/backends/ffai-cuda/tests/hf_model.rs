// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Generic dense-LLM verification harness on CUDA. Mirror of the Metal
//! harness: MODEL_DIR (HF dir) → load_hf → forward_single(TOK) → argmax.
//!   MODEL_DIR=/path TOK=9707 EXPECT=21806 \
//!     cargo test -p ffai-cuda --features cuda --test hf_model -- --nocapture
#![cfg(feature = "cuda")]

use ffai_core::DType;
use ffai_cuda::CudaDevice;
use ffai_models::llama::{forward_single, load_hf};

fn decode(b: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        DType::BF16 => {
            b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
        }
        DType::F16 => b.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        _ => vec![],
    }
}
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign << 31 } else {
            let mut e = -1i32; let mut m = mant;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            (sign << 31) | (((e + 127 - 15) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[test]
fn hf_model_forward_on_cuda() {
    let Some(dir) = std::env::var("MODEL_DIR").ok() else {
        eprintln!("set MODEL_DIR — skipping");
        return;
    };
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };
    eprintln!("device: {} — load_hf {dir}", dev.name());
    let m = load_hf(dev.as_ref(), &dir).expect("load_hf");
    eprintln!(
        "cfg: hidden={} heads={}/{} hd={} inter={} layers={} vocab={} qk_norm={} bias={}",
        m.cfg.hidden, m.cfg.n_q_heads, m.cfg.n_kv_heads, m.cfg.head_dim, m.cfg.intermediate,
        m.n_layers, m.vocab, m.cfg.qk_norm, m.cfg.attn_bias
    );

    let token: u32 = std::env::var("TOK").ok().and_then(|s| s.parse().ok()).unwrap_or(9707);
    let logits = forward_single(dev.as_ref(), &m.cfg, &m.weights, token).expect("forward");
    dev.synchronize().unwrap();
    let dt = logits.dtype;
    let mut lb = vec![0u8; m.vocab * dt.size_bytes()];
    dev.download(logits.buffer.as_ref(), &mut lb).unwrap();
    let l = decode(&lb, dt);

    let mut idx: Vec<usize> = (0..m.vocab).collect();
    idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
    eprintln!("token {token} → top-5 (CUDA): {:?}", &idx[..5]);
    eprintln!("ARGMAX (CUDA) = {}", idx[0]);
    if let Some(exp) = std::env::var("EXPECT").ok().and_then(|s| s.parse::<usize>().ok()) {
        assert_eq!(idx[0], exp, "CUDA argmax {} != expected {exp}", idx[0]);
        eprintln!("✅ argmax matches expected {exp}");
    }
}
