// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Validate the MLX 4-bit affine dequant + name-remap path against MLX ground
//! truth captured from `mx.dequantize(...)` on the Nemotron-Cascade checkpoint.
//! Needs the local MLX dir (skips if absent).
use ffai_loader::SafeTensors;

const DIR: &str = "/Users/tom/models/Nemotron-Cascade-2-30B-A3B-4bit";

#[test]
fn mlx_in_proj_row0_matches_mlx_groundtruth() {
    let Ok(st) = SafeTensors::open_dir(DIR) else { eprintln!("no MLX dir — skip"); return; };
    // backbone.layers.0.mixer.in_proj.weight → [10304, 2688] dense f32.
    let (w, sh) = st.tensor_f32("backbone.layers.0.mixer.in_proj.weight").unwrap();
    assert_eq!(sh, vec![10304, 2688], "dequant shape");
    // MLX ground truth: mx.dequantize(W,S,B,gs=64,bits=4,affine) row0[:8].
    let gt = [
        -0.0361328125f32, 0.030029296875, 0.0419921875, 0.01806640625,
        0.01806640625, -0.048095703125, 0.0240478515625, 0.0361328125,
    ];
    for (i, &g) in gt.iter().enumerate() {
        // Our scales/biases are bf16-decoded → tiny diffs vs MLX's bf16 path; 1e-3 is plenty.
        assert!((w[i] - g).abs() < 1e-3, "col {i}: got {} want {g}", w[i]);
    }
}

#[test]
fn mlx_prefix_strip_and_expert_slice() {
    let Ok(st) = SafeTensors::open_dir(DIR) else { eprintln!("no MLX dir — skip"); return; };
    // language_model. prefix strip + MLX dequant.
    let (_w, sh) = st.tensor_f32("language_model.backbone.layers.0.mixer.in_proj.weight").unwrap();
    assert_eq!(sh, vec![10304, 2688]);
    // Per-expert remap: layer 8 is an E-layer. experts.{e}.up_proj → switch_mlp.fc1[e].
    let (up0, ush) = st.tensor_f32("language_model.backbone.layers.8.mixer.experts.0.up_proj.weight").unwrap();
    assert_eq!(ush, vec![1856, 2688], "expert up slab shape [inter, hid]");
    let (dn0, dsh) = st.tensor_f32("language_model.backbone.layers.8.mixer.experts.0.down_proj.weight").unwrap();
    assert_eq!(dsh, vec![2688, 1856], "expert down slab shape [hid, inter]");
    // Expert 0 slab must equal the first slab of the full packed dequant.
    let (full_up, fsh) = st.tensor_f32("backbone.layers.8.mixer.switch_mlp.fc1.weight").unwrap();
    assert_eq!(fsh[0], 128 * 1856);
    let n = 1856 * 2688;
    for i in (0..n).step_by(9973) {
        assert!((up0[i] - full_up[i]).abs() < 1e-6, "expert0 up mismatch at {i}");
    }
    let _ = dn0;
}
