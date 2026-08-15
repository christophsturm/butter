// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Empirical layout sanity check against the REAL Qwen3.6-27B NVFP4
//! checkpoint (`~/models/q27nvfp4-nokv/`) — dequants one small tensor and
//! asserts no-NaN + a plausible magnitude range.
//!
//! `#[ignore]`d: as of this writing the checkpoint's `.safetensors` shards
//! are gone (symlinks into `~/models/qwen3.6-27b-nvfp4/`, which no longer
//! exists on disk — see `NVFP4_LAYOUT.md` § blocker note at the top). Run
//! this the moment the checkpoint (or an equivalent Qwen3.6-27B modelopt
//! NVFP4 export) is back on disk:
//!
//!   cargo test -p wh-butter-loader --test nvfp4_checkpoint_smoke -- --ignored --nocapture

use wh_butter_loader::SafeTensors;

#[test]
#[ignore = "checkpoint weights missing on disk as of 2026-08-14 — see NVFP4_LAYOUT.md"]
fn dequant_one_real_tensor_sanity() {
    let dir = format!("{}/models/q27nvfp4-nokv", std::env::var("HOME").unwrap());
    let st = SafeTensors::open_dir(&dir).expect("open checkpoint dir (is it restored?)");

    // A mid-size, non-huge FFN tensor: layer 0's down_proj (W4A16_NVFP4 per
    // hf_quant_config.json, group_size 16). hidden=5120, intermediate=17408
    // -> down_proj is [5120, 17408].
    let base = "model.language_model.layers.0.mlp.down_proj";
    let t = st.load_nvfp4_tensor(base).unwrap_or_else(|e| panic!("load_nvfp4_tensor('{base}') failed: {e}"));
    println!("{base}: rows={} cols={} block_size={} scale2={}", t.rows, t.cols, t.block_size, t.scale2);
    assert_eq!(t.block_size, 16, "expected group_size=16 per hf_quant_config.json");

    let (w, shape) = st.dequant_nvfp4_tensor_f32(base).unwrap();
    assert_eq!(shape, vec![t.rows, t.cols]);

    let mut n_nan = 0usize;
    let mut n_zero = 0usize;
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    for &v in &w {
        if v.is_nan() { n_nan += 1; continue; }
        if v == 0.0 { n_zero += 1; }
        max_abs = max_abs.max(v.abs());
        sum_abs += v.abs() as f64;
    }
    let mean_abs = sum_abs / w.len() as f64;
    println!(
        "dequant stats: n={} n_nan={n_nan} n_zero={n_zero} ({:.1}%) max_abs={max_abs:.4} mean_abs={mean_abs:.4}",
        w.len(),
        100.0 * n_zero as f64 / w.len() as f64
    );

    assert_eq!(n_nan, 0, "NVFP4 dequant produced NaN — layout mismatch (see NVFP4_LAYOUT.md §3 caveat)");
    // Plausible range for a trained LLM's down_proj weights: not all zero,
    // not blown up. NVFP4's E2M1 grid tops out at 6 * block_scale * scale2 —
    // any value near f32::MAX would mean the scale2/scale-byte read landed on
    // garbage bytes (wrong tensor, wrong offset, or the u8/f8e4m3 dtype-tag
    // assumption in NVFP4_LAYOUT.md §1 was wrong).
    assert!(max_abs > 1e-6, "all-zero dequant — likely reading the wrong bytes");
    assert!(max_abs < 1e3, "implausibly large dequant ({max_abs}) — likely a layout mismatch");
}
