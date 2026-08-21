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
    if dtype != DType::F32 || shape != [1] || bytes.len() != 4 {
        return Err(Error::Msg(format!(
            "nvfp4: {name} must be an F32 [1] tensor, got {dtype} {shape:?}"
        )));
    }
    Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
}

impl SafeTensors {
    /// Load `<base>.weight_packed`, `.weight_scale`,
    /// `.weight_global_scale`, and `.input_global_scale` without copying.
    pub fn load_nvfp4_tensor(&self, base: &str) -> Result<Nvfp4Tensor<'_>> {
        let (packed, packed_dtype, packed_shape) = self.tensor(&format!("{base}.weight_packed"))?;
        if packed_dtype != DType::U8 || packed_shape.len() != 2 {
            return Err(Error::Msg(format!(
                "nvfp4: {base}.weight_packed must be U8 [rows, cols/2], got {packed_dtype} {packed_shape:?}"
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

        // The checkpoint records inverse global scales. The dequantized
        // multiplier is their reciprocal.
        let weight_global = scalar_f32(self, &format!("{base}.weight_global_scale"))?;
        let input_global = scalar_f32(self, &format!("{base}.input_global_scale"))?;
        if !weight_global.is_finite() || weight_global <= 0.0 || !input_global.is_finite() || input_global <= 0.0 {
            return Err(Error::Msg(format!(
                "nvfp4: {base} global scales must be finite and positive"
            )));
        }

        Ok(Nvfp4Tensor {
            packed,
            scales,
            weight_multiplier: weight_global.recip(),
            input_multiplier: input_global.recip(),
            rows,
            cols,
            block_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp_codebooks_have_expected_landmarks() {
        assert_eq!(e2m1_decode(0x0), 0.0);
        assert_eq!(e2m1_decode(0x7), 6.0);
        assert_eq!(e2m1_decode(0xf), -6.0);
        assert_eq!(e4m3_decode(0x38), 1.0);
        assert_eq!(e4m3_decode(0xb8), -1.0);
    }
}
