// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Reader for compressed NVFP4 linear weights.

use crate::{DType, Error, Result, SafeTensors};

/// One row-major NVFP4 matrix borrowed from a SafeTensors mapping.
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4Tensor<'a> {
    /// Two E2M1 values per byte, low nibble first.
    pub packed: &'a [u8],
    /// One raw E4M3 scale byte per 16-column block.
    pub scales: &'a [u8],
    /// Multiplier applied after the E2M1 and E4M3 values are decoded.
    pub weight_multiplier: f32,
    /// Static activation multiplier recorded by the checkpoint.
    pub input_multiplier: f32,
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
}

/// Decode the OCP E2M1 code used by NVFP4.
pub fn e2m1_decode(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = MAG[(code & 0x7) as usize];
    if code & 0x8 != 0 { -value } else { value }
}

/// Decode an E4M3 finite-number scale byte.
pub fn e4m3_decode(code: u8) -> f32 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = ((code >> 3) & 0xf) as i32;
    let mantissa = (code & 0x7) as f32;
    let value = if exp == 0 {
        mantissa * 0.001_953_125
    } else {
        (1.0 + mantissa * 0.125) * 2f32.powi(exp - 7)
    };
    sign * value
}

fn scalar_f32(st: &SafeTensors, name: &str) -> Result<f32> {
    let (bytes, dtype, shape) = st.tensor(name)?;
    if dtype != DType::F32 || !(shape.is_empty() || shape == [1]) || bytes.len() != 4 {
        return Err(Error::Msg(format!(
            "nvfp4: {name} must be an F32 scalar, got {dtype} {shape:?}"
        )));
    }
    Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
}

impl SafeTensors {
    /// Load either ModelOpt's standard `weight` / `weight_scale` /
    /// `weight_scale_2` / `input_scale` names or the explicit
    /// `weight_packed` / `weight_global_scale` conversion names without
    /// copying the packed matrix.
    pub fn load_nvfp4_tensor(&self, base: &str) -> Result<Nvfp4Tensor<'_>> {
        let packed_name = if self.info(&format!("{base}.weight_packed")).is_some() {
            format!("{base}.weight_packed")
        } else {
            format!("{base}.weight")
        };
        let (packed, packed_dtype, packed_shape) = self.tensor(&packed_name)?;
        if packed_dtype != DType::U8 || packed_shape.len() != 2 {
            return Err(Error::Msg(format!(
                "nvfp4: {packed_name} must be U8 [rows, cols/2], got {packed_dtype} {packed_shape:?}"
            )));
        }
        let rows = packed_shape[0];
        let cols = packed_shape[1] * 2;

        let (scales, scale_dtype, scale_shape) = self.tensor(&format!("{base}.weight_scale"))?;
        if scale_dtype != DType::U8 || scale_shape.len() != 2 || scale_shape[0] != rows {
            return Err(Error::Msg(format!(
                "nvfp4: {base}.weight_scale must be raw FP8 [rows, blocks], got {scale_dtype} {scale_shape:?}"
            )));
        }
        let blocks = scale_shape[1];
        if blocks == 0 || cols % blocks != 0 {
            return Err(Error::Msg(format!(
                "nvfp4: {base} has {cols} columns and {blocks} scale blocks"
            )));
        }
        let block_size = cols / blocks;
        if block_size != 16 {
            return Err(Error::Msg(format!(
                "nvfp4: {base} block size {block_size}, expected 16"
            )));
        }

        let explicit_globals = self.info(&format!("{base}.weight_global_scale")).is_some();
        let weight_global_name = if explicit_globals {
            format!("{base}.weight_global_scale")
        } else {
            format!("{base}.weight_scale_2")
        };
        let input_global_name = if explicit_globals {
            format!("{base}.input_global_scale")
        } else {
            format!("{base}.input_scale")
        };
        let weight_global = scalar_f32(self, &weight_global_name)?;
        let input_global = scalar_f32(self, &input_global_name)?;
        if !weight_global.is_finite()
            || weight_global <= 0.0
            || !input_global.is_finite()
            || input_global <= 0.0
        {
            return Err(Error::Msg(format!(
                "nvfp4: {base} global scales must be finite and positive"
            )));
        }

        Ok(Nvfp4Tensor {
            packed,
            scales,
            // The explicit conversion stores inverse globals. Standard
            // ModelOpt exports store the dequant multipliers directly.
            weight_multiplier: if explicit_globals {
                weight_global.recip()
            } else {
                weight_global
            },
            input_multiplier: if explicit_globals {
                input_global.recip()
            } else {
                input_global
            },
            rows,
            cols,
            block_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_standard_nvfp4_fixture() -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("butter-nvfp4-{}.safetensors", std::process::id()));
        let header = serde_json::json!({
            "layer.weight": {"dtype":"U8", "shape":[1,8], "data_offsets":[0,8]},
            "layer.weight_scale": {"dtype":"F8_E4M3", "shape":[1,1], "data_offsets":[8,9]},
            "layer.weight_scale_2": {"dtype":"F32", "shape":[], "data_offsets":[9,13]},
            "layer.input_scale": {"dtype":"F32", "shape":[], "data_offsets":[13,17]}
        })
        .to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0x10; 8]);
        bytes.push(0x38);
        bytes.extend_from_slice(&0.25f32.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn fp_codebooks_have_expected_landmarks() {
        assert_eq!(e2m1_decode(0x0), 0.0);
        assert_eq!(e2m1_decode(0x7), 6.0);
        assert_eq!(e2m1_decode(0xf), -6.0);
        assert_eq!(e4m3_decode(0x38), 1.0);
        assert_eq!(e4m3_decode(0xb8), -1.0);
    }

    #[test]
    fn standard_modelopt_layout_stays_packed_and_uses_direct_scale() {
        let path = write_standard_nvfp4_fixture();
        let st = SafeTensors::open(path.to_str().unwrap()).unwrap();
        let matrix = st.load_nvfp4_tensor("layer").unwrap();
        assert_eq!((matrix.rows, matrix.cols, matrix.block_size), (1, 16, 16));
        assert_eq!(matrix.packed, &[0x10; 8]);
        assert_eq!(matrix.scales, &[0x38]);
        assert_eq!(matrix.weight_multiplier, 0.25);
        assert_eq!(matrix.input_multiplier, 0.5);
        std::fs::remove_file(path).unwrap();
    }
}
