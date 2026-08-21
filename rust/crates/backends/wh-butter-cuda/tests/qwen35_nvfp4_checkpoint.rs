// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;
use wh_butter_core::{DType, Tensor};
use wh_butter_cuda::CudaDevice;
use wh_butter_loader::{nvfp4::{e2m1_decode, e4m3_decode}, SafeTensors};
use wh_butter_ops::gemv_nvfp4;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
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
    device.synchronize().unwrap();
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
