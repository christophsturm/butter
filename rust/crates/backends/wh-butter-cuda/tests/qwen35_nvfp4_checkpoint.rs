// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;
use wh_butter_core::{DType, Tensor};
use wh_butter_cuda::CudaDevice;
use wh_butter_loader::{nvfp4::{e2m1_decode, e4m3_decode}, SafeTensors};
use wh_butter_ops::{
    cast_bf16_f32, fp8_quant_perrow, gemm_fp4, gemm_fp8_perchan, gemv, gemv_nvfp4,
    gemv_q8, lt_fp4_quant_with_global, quantize_bf16_q8,
};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16)
        })
        .collect()
}

fn f32_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn interleave_scales(scales: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(rows % 128, 0);
    assert_eq!(cols % 4, 0);
    let mut output = vec![0u8; scales.len()];
    let col_blocks = cols / 4;
    for row in 0..rows {
        for col in 0..cols {
            let dst = (row / 128) * (512 * col_blocks)
                + (col / 4) * 512
                + (row % 32) * 16
                + ((row / 32) % 4) * 4
                + col % 4;
            output[dst] = scales[row * cols + col];
        }
    }
    output
}

#[test]
fn real_checkpoint_w4a4_m1_matches_quantized_host_reference() {
    let Ok(dir) = std::env::var("BUTTER_QWEN35_NVFP4") else {
        eprintln!("BUTTER_QWEN35_NVFP4 is unset, skipping checkpoint test");
        return;
    };
    let Some(device) = CudaDevice::create().expect("create CUDA device") else {
        eprintln!("no CUDA device, skipping checkpoint test");
        return;
    };
    let device = device.as_ref();
    let st = SafeTensors::open_dir(&dir).expect("open checkpoint");
    let matrix = st
        .load_nvfp4_tensor("model.language_model.layers.0.mlp.gate_proj")
        .expect("load NVFP4 gate projection");
    let x: Vec<f32> = (0..matrix.cols)
        .map(|index| ((index % 31) as f32 - 15.0) / 310.0)
        .collect();
    let input = Tensor::new(
        device.upload(&f32_bytes(&x)).unwrap(),
        vec![matrix.cols],
        DType::F32,
    );
    let (x_packed, x_scales, _) = lt_fp4_quant_with_global(
        device,
        &input,
        1,
        matrix.cols,
        matrix.input_multiplier,
    )
    .expect("quantize activation");
    device.synchronize().unwrap();
    let mut x_packed_host = vec![0u8; matrix.cols / 2];
    let activation_scale_cols = matrix.cols / matrix.block_size;
    let activation_scale_bytes = 512 * (activation_scale_cols / 4);
    let mut x_scales_host = vec![0u8; activation_scale_bytes];
    device
        .download(x_packed.buffer.as_ref(), &mut x_packed_host)
        .unwrap();
    device
        .download(x_scales.buffer.as_ref(), &mut x_scales_host)
        .unwrap();
    assert_eq!(
        &x_packed_host[..16],
        &[255, 239, 238, 222, 205, 188, 170, 9, 33, 50, 68, 85, 102, 102, 119, 247],
        "packed activation prefix changed"
    );
    let mut x_quantized = vec![0.0f32; matrix.cols];
    for col in 0..matrix.cols {
        let code_byte = x_packed_host[col / 2];
        let code = if col & 1 == 0 {
            code_byte & 0xf
        } else {
            code_byte >> 4
        };
        let block = col / matrix.block_size;
        let scale_offset = (block / 4) * 512 + block % 4;
        x_quantized[col] = e2m1_decode(code)
            * e4m3_decode(x_scales_host[scale_offset])
            * matrix.input_multiplier;
    }
    let checked_rows = 48;
    let mut expected = vec![0.0f32; checked_rows];
    for row in 0..checked_rows {
        for col in 0..matrix.cols {
            let byte = matrix.packed[row * (matrix.cols / 2) + col / 2];
            let code = if col & 1 == 0 { byte & 0xf } else { byte >> 4 };
            let scale = matrix.scales
                [row * (matrix.cols / matrix.block_size) + col / matrix.block_size];
            expected[row] += e2m1_decode(code)
                * e4m3_decode(scale)
                * matrix.weight_multiplier
                * x_quantized[col];
        }
    }
    let weight_scales = interleave_scales(
        matrix.scales,
        matrix.rows,
        matrix.cols / matrix.block_size,
    );
    let weight = Tensor::new(
        device.upload(matrix.packed).unwrap(),
        vec![matrix.packed.len()],
        DType::U8,
    );
    let scales = Tensor::new(
        device.upload(&weight_scales).unwrap(),
        vec![weight_scales.len()],
        DType::U8,
    );
    let alpha = matrix.input_multiplier * matrix.weight_multiplier;
    let alpha = Tensor::new(
        device.upload(&alpha.to_le_bytes()).unwrap(),
        vec![1],
        DType::F32,
    );
    let output = gemm_fp4(
        device,
        &x_packed,
        &x_scales,
        &weight,
        &scales,
        1,
        matrix.rows,
        matrix.cols,
        true,
        Some(&alpha),
    )
    .expect("dispatch checkpoint W4A4 M=1 GEMM");
    device.synchronize().unwrap();
    let mut output_bytes = vec![0u8; matrix.rows * 4];
    device
        .download(output.buffer.as_ref(), &mut output_bytes)
        .unwrap();
    let actual: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let max_abs = actual[..checked_rows]
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 0.02, "checkpoint W4A4 M=1 max_abs={max_abs}");
}

#[test]
fn real_checkpoint_w4a4_m5_matches_independent_rows() {
    let Ok(dir) = std::env::var("BUTTER_QWEN35_NVFP4") else {
        eprintln!("BUTTER_QWEN35_NVFP4 is unset, skipping checkpoint test");
        return;
    };
    let Some(device) = CudaDevice::create().expect("create CUDA device") else {
        eprintln!("no CUDA device, skipping checkpoint test");
        return;
    };
    let device = device.as_ref();
    let st = SafeTensors::open_dir(&dir).expect("open checkpoint");
    let matrix = st
        .load_nvfp4_tensor("model.language_model.layers.0.mlp.gate_proj")
        .expect("load NVFP4 gate projection");
    let rows = 5;
    let values: Vec<f32> = (0..rows)
        .flat_map(|row| {
            (0..matrix.cols).map(move |col| {
                (((col + row * 7) % 31) as f32 - 15.0) / (310.0 + row as f32)
            })
        })
        .collect();
    let input = Tensor::new(
        device.upload(&f32_bytes(&values)).unwrap(),
        vec![rows, matrix.cols],
        DType::F32,
    );
    let (x_packed, x_scales, _) = lt_fp4_quant_with_global(
        device,
        &input,
        rows,
        matrix.cols,
        matrix.input_multiplier,
    )
    .expect("quantize batched activation");
    let weight_scales = interleave_scales(
        matrix.scales,
        matrix.rows,
        matrix.cols / matrix.block_size,
    );
    let weight = Tensor::new(
        device.upload(matrix.packed).unwrap(),
        vec![matrix.packed.len()],
        DType::U8,
    );
    let scales = Tensor::new(
        device.upload(&weight_scales).unwrap(),
        vec![weight_scales.len()],
        DType::U8,
    );
    let alpha = matrix.input_multiplier * matrix.weight_multiplier;
    let alpha = Tensor::new(
        device.upload(&alpha.to_le_bytes()).unwrap(),
        vec![1],
        DType::F32,
    );
    let batched = gemm_fp4(
        device,
        &x_packed,
        &x_scales,
        &weight,
        &scales,
        rows,
        matrix.rows,
        matrix.cols,
        true,
        Some(&alpha),
    )
    .expect("dispatch checkpoint W4A4 M=5 GEMM");
    let mut expected = Vec::with_capacity(rows * matrix.rows);
    for row in 0..rows {
        let row_input = Tensor::new(
            device
                .upload(&f32_bytes(
                    &values[row * matrix.cols..(row + 1) * matrix.cols],
                ))
                .unwrap(),
            vec![matrix.cols],
            DType::F32,
        );
        let (row_packed, row_scales, _) = lt_fp4_quant_with_global(
            device,
            &row_input,
            1,
            matrix.cols,
            matrix.input_multiplier,
        )
        .expect("quantize independent activation row");
        let row_output = gemm_fp4(
            device,
            &row_packed,
            &row_scales,
            &weight,
            &scales,
            1,
            matrix.rows,
            matrix.cols,
            true,
            Some(&alpha),
        )
        .expect("dispatch independent W4A4 row");
        device.synchronize().unwrap();
        let mut bytes = vec![0u8; matrix.rows * 4];
        device
            .download(row_output.buffer.as_ref(), &mut bytes)
            .unwrap();
        expected.extend(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())),
        );
    }
    device.synchronize().unwrap();
    let mut bytes = vec![0u8; rows * matrix.rows * 4];
    device.download(batched.buffer.as_ref(), &mut bytes).unwrap();
    let actual = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()));
    let max_abs = actual
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 0.02, "checkpoint W4A4 M=5 max_abs={max_abs}");
}

#[test]
fn real_checkpoint_bf16_lm_head_rows_match_host_reference() {
    let Ok(dir) = std::env::var("BUTTER_QWEN35_NVFP4") else {
        eprintln!("BUTTER_QWEN35_NVFP4 is unset, skipping checkpoint test");
        return;
    };
    let Some(device) = CudaDevice::create().expect("create CUDA device") else {
        eprintln!("no CUDA device, skipping checkpoint test");
        return;
    };
    let device = device.as_ref();
    let st = SafeTensors::open_dir(&dir).expect("open checkpoint");
    let (all_weights, dtype, shape) = st.tensor("lm_head.weight").expect("load lm_head");
    assert_eq!(dtype, DType::BF16);
    assert_eq!(shape, [248_320, 5_120]);

    let rows = 48;
    let cols = shape[1];
    let weight_bytes = &all_weights[..rows * cols * 2];
    let weights = bf16_to_f32(weight_bytes);
    let input_values: Vec<f32> = (0..cols)
        .map(|i| ((i % 31) as f32 - 15.0) / 310.0)
        .collect();
    let input_bf16 = f32_to_bf16_bytes(&input_values);
    let rounded_input = bf16_to_f32(&input_bf16);
    let expected: Vec<f32> = weights
        .chunks_exact(cols)
        .map(|row| row.iter().zip(&rounded_input).map(|(w, x)| w * x).sum())
        .collect();

    let weight = Tensor::new(
        device.upload(weight_bytes).unwrap(),
        vec![rows, cols],
        DType::BF16,
    );
    let input_bf16_tensor = Tensor::new(
        device.upload(&input_bf16).unwrap(),
        vec![cols],
        DType::BF16,
    );
    let output =
        gemv(device, &weight, &input_bf16_tensor).expect("dispatch BF16 lm_head GEMV");
    device.synchronize().unwrap();
    let mut output_bytes = vec![0u8; rows * 2];
    device
        .download(output.buffer.as_ref(), &mut output_bytes)
        .unwrap();
    let actual = bf16_to_f32(&output_bytes);
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 0.01, "checkpoint BF16 lm_head GEMV max_abs={max_abs}");

    let (q8, q8_scales) = quantize_bf16_q8(weight_bytes, rows, cols).unwrap();
    let blocks_per_row = cols / 32;
    let expected_q8: Vec<f32> = (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| {
                    let block = col / 32;
                    let packed = q8[row * blocks_per_row * 8 + block * 8 + (col % 32) / 4];
                    let code = ((packed >> ((col % 4) * 8)) as u8) as i8 as f32;
                    code * q8_scales[row * blocks_per_row + block] * input_values[col]
                })
                .sum()
        })
        .collect();
    let q8_bytes: Vec<u8> = q8.iter().flat_map(|value| value.to_le_bytes()).collect();
    let q8_tensor = Tensor::new(
        device.upload(&q8_bytes).unwrap(),
        vec![q8.len()],
        DType::U32,
    );
    let q8_scale_tensor = Tensor::new(
        device
            .upload(
                &q8_scales
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        vec![q8_scales.len()],
        DType::F32,
    );
    let input_f32 = Tensor::new(
        device.upload(&f32_bytes(&input_values)).unwrap(),
        vec![cols],
        DType::F32,
    );
    let q8_output = gemv_q8(
        device,
        &q8_tensor,
        &q8_scale_tensor,
        &input_f32,
        rows,
        cols,
        rows,
    )
    .expect("dispatch Q8 lm_head GEMV");
    device.synchronize().unwrap();
    let mut q8_bytes = vec![0u8; rows * 4];
    device.download(q8_output.buffer.as_ref(), &mut q8_bytes).unwrap();
    let actual_q8: Vec<f32> = q8_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let q8_oracle_max_abs = actual_q8
        .iter()
        .zip(&expected_q8)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let q8_vs_bf16_max_abs = actual_q8
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "checkpoint Q8 lm_head oracle max_abs={q8_oracle_max_abs}, versus BF16 max_abs={q8_vs_bf16_max_abs}"
    );
    assert!(
        q8_oracle_max_abs < 1e-3,
        "checkpoint Q8 lm_head oracle max_abs={q8_oracle_max_abs}"
    );
    assert!(
        q8_vs_bf16_max_abs < 0.1,
        "checkpoint Q8 versus BF16 lm_head max_abs={q8_vs_bf16_max_abs}"
    );
}

fn check_real_checkpoint_fp8_projection(name: &str, label: &str) {
    let Ok(dir) = std::env::var("BUTTER_QWEN35_NVFP4") else {
        eprintln!("BUTTER_QWEN35_NVFP4 is unset, skipping checkpoint test");
        return;
    };
    let Some(device) = CudaDevice::create().unwrap() else {
        eprintln!("CUDA is unavailable, skipping checkpoint test");
        return;
    };
    let device = device.as_ref();
    let st = SafeTensors::open_dir(&dir).expect("open checkpoint");
    let (all_weights, dtype, shape) = st.tensor(name).expect("load FP8 projection source");
    assert_eq!(dtype, DType::BF16);
    assert_eq!(shape.len(), 2);

    let rows = 48;
    let cols = shape[1];
    let weight_bytes = &all_weights[..rows * cols * 2];
    let weights = bf16_to_f32(weight_bytes);
    let input_values: Vec<f32> = (0..cols)
        .map(|index| ((index % 31) as f32 - 15.0) / 310.0)
        .collect();
    let expected: Vec<f32> = weights
        .chunks_exact(cols)
        .map(|row| row.iter().zip(&input_values).map(|(w, x)| w * x).sum())
        .collect();

    let weight_bf16 = Tensor::new(
        device.upload(weight_bytes).unwrap(),
        vec![rows, cols],
        DType::BF16,
    );
    let weight_f32 = cast_bf16_f32(device, &weight_bf16).unwrap();
    let (weight_fp8, weight_scales) = fp8_quant_perrow(device, &weight_f32, rows, cols).unwrap();
    let input = Tensor::new(
        device.upload(&f32_bytes(&input_values)).unwrap(),
        vec![1, cols],
        DType::F32,
    );
    let (input_fp8, input_scales) = fp8_quant_perrow(device, &input, 1, cols).unwrap();
    let output = gemm_fp8_perchan(
        device,
        &input_fp8,
        &input_scales,
        &weight_fp8,
        &weight_scales,
        1,
        rows,
        cols,
    )
    .unwrap();
    device.synchronize().unwrap();
    let mut output_bytes = vec![0u8; rows * 4];
    device
        .download(output.buffer.as_ref(), &mut output_bytes)
        .unwrap();
    let actual: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let dot: f64 = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| *actual as f64 * *expected as f64)
        .sum();
    let actual_norm: f64 = actual.iter().map(|value| (*value as f64).powi(2)).sum();
    let expected_norm: f64 = expected.iter().map(|value| (*value as f64).powi(2)).sum();
    let cosine = dot / (actual_norm * expected_norm).sqrt();
    eprintln!("checkpoint FP8 {label} projection max_abs={max_abs}, cosine={cosine:.8}");
    assert!(max_abs < 0.02, "checkpoint FP8 {label} max_abs={max_abs}");
    assert!(cosine > 0.999, "checkpoint FP8 {label} cosine={cosine}");

    let mut scale_bytes = vec![0u8; rows * 4];
    device
        .download(weight_scales.buffer.as_ref(), &mut scale_bytes)
        .unwrap();
    let mut wrong_scales: Vec<f32> = scale_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    wrong_scales.rotate_left(1);
    let wrong_scale_tensor = Tensor::new(
        device.upload(&f32_bytes(&wrong_scales)).unwrap(),
        vec![rows],
        DType::F32,
    );
    let wrong = gemm_fp8_perchan(
        device,
        &input_fp8,
        &input_scales,
        &weight_fp8,
        &wrong_scale_tensor,
        1,
        rows,
        cols,
    )
    .unwrap();
    device.synchronize().unwrap();
    let mut wrong_bytes = vec![0u8; rows * 4];
    device
        .download(wrong.buffer.as_ref(), &mut wrong_bytes)
        .unwrap();
    let wrong_max_abs = wrong_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .zip(&actual)
        .map(|(wrong, actual)| (wrong - actual).abs())
        .fold(0.0f32, f32::max);
    assert!(
        wrong_max_abs > 0.01,
        "rotated per-channel scales should diverge, max_abs={wrong_max_abs}"
    );
}

#[test]
fn real_checkpoint_fp8_gdn_projection_matches_host_reference() {
    check_real_checkpoint_fp8_projection(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "GDN",
    );
}

#[test]
fn real_checkpoint_fp8_attention_projection_matches_host_reference() {
    check_real_checkpoint_fp8_projection(
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "attention",
    );
}

#[test]
fn real_checkpoint_rows_match_host_reference() {
    let Ok(dir) = std::env::var("BUTTER_QWEN35_NVFP4") else {
        eprintln!("BUTTER_QWEN35_NVFP4 is unset, skipping checkpoint test");
        return;
    };
    let Some(device) = CudaDevice::create().expect("create CUDA device") else {
        eprintln!("no CUDA device, skipping checkpoint test");
        return;
    };
    let device = device.as_ref();
    let st = SafeTensors::open_dir(&dir).expect("open checkpoint");
    let base = "model.language_model.layers.0.mlp.gate_proj";
    let matrix = st.load_nvfp4_tensor(base).expect("load NVFP4 gate projection");
    assert_eq!((matrix.rows, matrix.cols, matrix.block_size), (17_408, 5_120, 16));

    let rows = 48;
    let packed_len = rows * matrix.cols / 2;
    let scale_len = rows * matrix.cols / matrix.block_size;
    let packed = &matrix.packed[..packed_len];
    let scales = &matrix.scales[..scale_len];
    let x: Vec<f32> = (0..matrix.cols).map(|i| ((i % 31) as f32 - 15.0) / 31.0).collect();

    let mut expected = vec![0.0f32; rows];
    for row in 0..rows {
        for col in 0..matrix.cols {
            let byte = packed[row * (matrix.cols / 2) + col / 2];
            let code = if col & 1 == 0 { byte & 0xf } else { byte >> 4 };
            let scale = scales[row * (matrix.cols / matrix.block_size) + col / matrix.block_size];
            expected[row] += e2m1_decode(code) * e4m3_decode(scale) * matrix.weight_multiplier * x[col];
        }
    }

    let weight = Tensor::new(device.upload(packed).unwrap(), vec![packed_len], DType::U8);
    let scale = Tensor::new(device.upload(scales).unwrap(), vec![scale_len], DType::U8);
    let input = Tensor::new(device.upload(&f32_bytes(&x)).unwrap(), vec![matrix.cols], DType::F32);
    let output = gemv_nvfp4(
        device,
        &weight,
        &scale,
        &input,
        rows,
        matrix.cols,
        matrix.block_size,
        matrix.weight_multiplier,
    )
    .expect("dispatch NVFP4 GEMV");
    device.synchronize().unwrap();
    let mut bytes = vec![0u8; rows * 4];
    device.download(output.buffer.as_ref(), &mut bytes).unwrap();
    let actual: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 2e-4, "checkpoint NVFP4 GEMV max_abs={max_abs}");

    let full_weight = Tensor::new(
        device.upload(matrix.packed).unwrap(),
        vec![matrix.packed.len()],
        DType::U8,
    );
    let full_scales = Tensor::new(
        device.upload(matrix.scales).unwrap(),
        vec![matrix.scales.len()],
        DType::U8,
    );
    let full_output = gemv_nvfp4(
        device,
        &full_weight,
        &full_scales,
        &input,
        matrix.rows,
        matrix.cols,
        matrix.block_size,
        matrix.weight_multiplier,
    )
    .unwrap();
    device.synchronize().unwrap();
    let mut full_bytes = vec![0u8; matrix.rows * 4];
    device
        .download(full_output.buffer.as_ref(), &mut full_bytes)
        .unwrap();
    let full_actual: Vec<f32> = full_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(full_actual.len(), matrix.rows);
    assert!(
        full_actual.iter().all(|value| value.is_finite()),
        "full checkpoint NVFP4 GEMV produced a non-finite value"
    );
    let full_prefix_max_abs = full_actual[..rows]
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        full_prefix_max_abs < 2e-4,
        "full checkpoint NVFP4 GEMV prefix max_abs={full_prefix_max_abs}"
    );
    let iterations = 64;
    let started = Instant::now();
    for _ in 0..iterations {
        let _ = gemv_nvfp4(
            device,
            &full_weight,
            &full_scales,
            &input,
            matrix.rows,
            matrix.cols,
            matrix.block_size,
            matrix.weight_multiplier,
        )
        .unwrap();
    }
    device.synchronize().unwrap();
    let micros = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
    let bytes = (matrix.packed.len() + matrix.scales.len() + matrix.cols * 4) as f64;
    eprintln!(
        "checkpoint gate projection: {micros:.1} us, {:.1} GB/s",
        bytes / (micros * 1e-6) / 1e9
    );
}
