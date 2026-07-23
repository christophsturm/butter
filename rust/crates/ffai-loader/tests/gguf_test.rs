// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! GGUF v3 parser + Q8_0 dequant vs gguf-py reference (DSv4-Flash checkpoint).
use ffai_loader::gguf::Gguf;

#[test]
fn gguf_parse_and_q8_0_dequant_match_ggufpy() {
    let path = std::env::var("GGUF_PATH").unwrap_or_else(|_| {
        "/Users/tom/models/ds4-model/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf".to_string()
    });
    let Ok(g) = Gguf::open(&path) else {
        eprintln!("no GGUF at {path} — skipping");
        return;
    };
    let bc = g.metadata_u32.get("deepseek4.block_count").copied();
    eprintln!("deepseek4.block_count = {bc:?}, tensors = {}", g.tensor_names().count());
    assert_eq!(bc, Some(43), "block_count metadata");

    let d = g.dequant_f32("blk.0.attn_kv.weight").expect("dequant attn_kv");
    let want = [-0.002768f32, 0.039214, 0.010611, 0.004613, -0.000923, 0.05859, 0.014763, 0.042905];
    eprintln!("rust first8 = {:?}", &d[..8]);
    let mut e = 0.0f32;
    for i in 0..8 {
        e = e.max((d[i] - want[i]).abs());
    }
    assert!(e <= 1e-4, "Q8_0 dequant mismatch vs gguf-py: max|Δ|={e:.3e}");
    eprintln!("✅ GGUF v3 parse + Q8_0 dequant match gguf-py (max|Δ|={e:.1e})");

    // Q2_K dequant vs gguf-py.
    let q2 = g.dequant_f32("blk.0.ffn_down_exps.weight").expect("dequant down_exps");
    let want2 = [-0.017595f32, 0.015516, 0.032072, -0.017595, 0.015516, -0.017595, 0.015516, -0.00104];
    eprintln!("rust Q2_K first8 = {:?}", &q2[..8]);
    let mut e2 = 0.0f32;
    for i in 0..8 {
        e2 = e2.max((q2[i] - want2[i]).abs());
    }
    assert!(e2 <= 1e-4, "Q2_K dequant mismatch vs gguf-py: max|Δ|={e2:.3e}");
    eprintln!("✅ GGUF Q2_K dequant matches gguf-py (max|Δ|={e2:.1e})");

    // IQ2_XXS dequant vs gguf-py.
    let iq = g.dequant_f32("blk.0.ffn_gate_exps.weight").expect("dequant gate_exps");
    let want3 = [-0.006656f32, -0.006656, 0.006656, 0.020801, 0.006656, 0.006656, 0.006656, 0.020801];
    eprintln!("rust IQ2_XXS first8 = {:?}", &iq[..8]);
    let mut e3 = 0.0f32;
    for i in 0..8 {
        e3 = e3.max((iq[i] - want3[i]).abs());
    }
    assert!(e3 <= 1e-4, "IQ2_XXS dequant mismatch vs gguf-py: max|Δ|={e3:.3e}");
    eprintln!("✅ GGUF IQ2_XXS dequant matches gguf-py (max|Δ|={e3:.1e}) — all DSv4 quant types covered");
}

// ── Synthetic round-trip unit tests for the k-quants (no model file needed) ──
//
// We construct a one-block GGUF in memory whose data section holds a single
// Q4_K/Q5_K/Q6_K super-block built with the canonical GGML packing, then assert
// `dequant_f32` reproduces the values implied by that packing. This pins the
// bit layout (sub-block scale unpack, nibble order, high-bit planes) without an
// external reference.

/// f32 → IEEE-754 half (round-to-nearest-even), enough for test scales.
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        return sign; // flush tiny to signed zero (test values avoid this)
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Build a minimal GGUF v3 byte image with a single named tensor whose data is
/// `data` and type id `ggml_type`, returning the bytes to write to a temp file.
fn build_gguf(name: &str, dims: &[u64], ggml_type: u32, data: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // magic "GGUF"
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
    b.extend_from_slice(&0u64.to_le_bytes()); // n_kv
    // tensor info
    b.extend_from_slice(&(name.len() as u64).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&ggml_type.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // offset within data section
    // align data section to 32
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.extend_from_slice(data);
    b
}

fn write_temp(bytes: &[u8], stem: &str) -> String {
    let path = std::env::temp_dir().join(format!("ffai_kquant_{stem}.gguf"));
    std::fs::write(&path, bytes).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn q4_k_dequant_synthetic_roundtrip() {
    // One QK_K=256 super-block. d/dmin and 8 (scale,min) 6-bit pairs chosen
    // small/representable; 4-bit codes = (index % 16).
    let d = 0.5f32;
    let dmin = 0.125f32;
    let sc: [u8; 8] = [3, 7, 12, 1, 20, 33, 5, 9];
    let mn: [u8; 8] = [2, 5, 8, 0, 14, 21, 4, 6];

    // Pack the 12-byte scales array per GGML get_scale_min_k4 inverse.
    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = sc[j] & 63;
        scales[j + 4] = mn[j] & 63;
    }
    for j in 4..8 {
        scales[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4);
        scales[j - 4] |= (sc[j] >> 4) << 6;
        scales[j] |= (mn[j] >> 4) << 6;
    }

    // 4-bit codes: element i in sub-block s has code (i % 16). qs holds 128
    // bytes: chunk j (32 bytes) packs sub-block 2j in low nibbles, 2j+1 in high.
    let code = |elem: usize| (elem % 16) as u8;
    let mut qs = [0u8; 128];
    for j in 0..4 {
        for l in 0..32 {
            let lo = code(2 * j * 32 + l);
            let hi = code((2 * j + 1) * 32 + l);
            qs[j * 32 + l] = (lo & 0xF) | (hi << 4);
        }
    }

    let mut data = Vec::new();
    data.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    data.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
    data.extend_from_slice(&scales);
    data.extend_from_slice(&qs);
    assert_eq!(data.len(), 144);

    let bytes = build_gguf("w", &[256], 12, &data);
    let path = write_temp(&bytes, "q4k");
    let g = Gguf::open(&path).unwrap();
    let out = g.dequant_f32("w").unwrap();
    std::fs::remove_file(&path).ok();

    let df = f16_to_f32_t(f32_to_f16(d));
    let dminf = f16_to_f32_t(f32_to_f16(dmin));
    let mut maxe = 0.0f32;
    for sb in 0..8 {
        for l in 0..32 {
            let elem = sb * 32 + l;
            let want = df * sc[sb] as f32 * code(elem) as f32 - dminf * mn[sb] as f32;
            maxe = maxe.max((out[elem] - want).abs());
        }
    }
    assert!(maxe < 1e-3, "Q4_K synthetic mismatch max|Δ|={maxe:.3e}");
    eprintln!("✅ Q4_K synthetic round-trip ok (max|Δ|={maxe:.1e})");
}

#[test]
fn q5_k_dequant_synthetic_roundtrip() {
    let d = 0.25f32;
    let dmin = 0.0625f32;
    let sc: [u8; 8] = [4, 9, 15, 2, 30, 40, 7, 11];
    let mn: [u8; 8] = [1, 6, 10, 3, 18, 25, 5, 8];

    let mut scales = [0u8; 12];
    for j in 0..4 {
        scales[j] = sc[j] & 63;
        scales[j + 4] = mn[j] & 63;
    }
    for j in 4..8 {
        scales[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4);
        scales[j - 4] |= (sc[j] >> 4) << 6;
        scales[j] |= (mn[j] >> 4) << 6;
    }

    // 5-bit codes: low 4 bits = (elem % 16), high bit alternates by elem parity.
    let lo_code = |elem: usize| (elem % 16) as u8;
    let hi_bit = |elem: usize| (elem % 3 == 0) as u8; // 0 or 1

    let mut qs = [0u8; 128];
    let mut qh = [0u8; 32];
    for j in 0..4 {
        for l in 0..32 {
            let e_lo = 2 * j * 32 + l;
            let e_hi = (2 * j + 1) * 32 + l;
            qs[j * 32 + l] = (lo_code(e_lo) & 0xF) | (lo_code(e_hi) << 4);
            // high-bit plane: bit (2j) for low sub-block lane l, bit (2j+1) for high.
            if hi_bit(e_lo) != 0 {
                qh[l] |= 1 << (2 * j);
            }
            if hi_bit(e_hi) != 0 {
                qh[l] |= 1 << (2 * j + 1);
            }
        }
    }

    let mut data = Vec::new();
    data.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    data.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
    data.extend_from_slice(&scales);
    data.extend_from_slice(&qh);
    data.extend_from_slice(&qs);
    assert_eq!(data.len(), 176);

    let bytes = build_gguf("w", &[256], 13, &data);
    let path = write_temp(&bytes, "q5k");
    let g = Gguf::open(&path).unwrap();
    let out = g.dequant_f32("w").unwrap();
    std::fs::remove_file(&path).ok();

    let df = f16_to_f32_t(f32_to_f16(d));
    let dminf = f16_to_f32_t(f32_to_f16(dmin));
    let mut maxe = 0.0f32;
    for sb in 0..8 {
        for l in 0..32 {
            let elem = sb * 32 + l;
            let q = lo_code(elem) as i32 + 16 * hi_bit(elem) as i32;
            let want = df * sc[sb] as f32 * q as f32 - dminf * mn[sb] as f32;
            maxe = maxe.max((out[elem] - want).abs());
        }
    }
    assert!(maxe < 1e-3, "Q5_K synthetic mismatch max|Δ|={maxe:.3e}");
    eprintln!("✅ Q5_K synthetic round-trip ok (max|Δ|={maxe:.1e})");
}

#[test]
fn q6_k_dequant_synthetic_roundtrip() {
    // 16 int8 scales, d(f16). 6-bit codes 0..63 stored as ql(low4)+qh(high2).
    let d = 0.03125f32;
    let scales: [i8; 16] = [1, -2, 3, -4, 5, -6, 7, -8, 2, -3, 4, -5, 6, -7, 8, -1];

    // We choose, for output element index `e`, a 6-bit code = (e % 64).
    // Reconstruct the on-disk ql/qh from the canonical interleave the dequant
    // uses (two 128-halves; 4 groups per lane).
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    // Mirror the reader's index math so the round-trip is exact.
    for n2 in 0..2 {
        for l in 0..32 {
            // group base output indices in this half
            let e1 = n2 * 128 + l;
            let e2 = n2 * 128 + 32 + l;
            let e3 = n2 * 128 + 64 + l;
            let e4 = n2 * 128 + 96 + l;
            let c1 = (e1 % 64) as u8; // 0..63
            let c2 = (e2 % 64) as u8;
            let c3 = (e3 % 64) as u8;
            let c4 = (e4 % 64) as u8;
            // ql_h[l]   low nibble = c1&0xF ; high nibble = c3&0xF
            // ql_h[l+32] low nibble = c2&0xF; high nibble = c4&0xF
            ql[n2 * 64 + l] = (c1 & 0xF) | ((c3 & 0xF) << 4);
            ql[n2 * 64 + l + 32] = (c2 & 0xF) | ((c4 & 0xF) << 4);
            // qh_h[l]: bits0-1=c1>>4, bits2-3=c2>>4, bits4-5=c3>>4, bits6-7=c4>>4
            qh[n2 * 32 + l] = ((c1 >> 4) & 3)
                | (((c2 >> 4) & 3) << 2)
                | (((c3 >> 4) & 3) << 4)
                | (((c4 >> 4) & 3) << 6);
        }
    }

    let mut data = Vec::new();
    data.extend_from_slice(&ql);
    data.extend_from_slice(&qh);
    for s in scales {
        data.push(s as u8);
    }
    data.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    assert_eq!(data.len(), 210);

    let bytes = build_gguf("w", &[256], 14, &data);
    let path = write_temp(&bytes, "q6k");
    let g = Gguf::open(&path).unwrap();
    let out = g.dequant_f32("w").unwrap();
    std::fs::remove_file(&path).ok();

    let df = f16_to_f32_t(f32_to_f16(d));
    // scale index `is` = l/16, offset by group within the half; matches reader.
    let mut maxe = 0.0f32;
    for n2 in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let groups = [
                (n2 * 128 + l, scales[n2 * 8 + is]),
                (n2 * 128 + 32 + l, scales[n2 * 8 + is + 2]),
                (n2 * 128 + 64 + l, scales[n2 * 8 + is + 4]),
                (n2 * 128 + 96 + l, scales[n2 * 8 + is + 6]),
            ];
            for (e, sca) in groups {
                let code = (e % 64) as i32 - 32;
                let want = df * sca as f32 * code as f32;
                maxe = maxe.max((out[e] - want).abs());
            }
        }
    }
    assert!(maxe < 1e-3, "Q6_K synthetic mismatch max|Δ|={maxe:.3e}");
    eprintln!("✅ Q6_K synthetic round-trip ok (max|Δ|={maxe:.1e})");
}

#[test]
fn split_gguf_reads_tensors_across_parts() {
    // Build a 2-part split GGUF (`*-00001-of-00002.gguf` / `*-00002-...`) where
    // part 1 holds tensor "a" and part 2 holds tensor "b", then confirm a single
    // `Gguf::open` on part 1 exposes BOTH tensors with correct data.
    let a: Vec<u8> = (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let b: Vec<u8> = (0..4u32).flat_map(|i| ((i + 10) as f32).to_le_bytes()).collect();
    let p1 = build_gguf("a", &[4], 0, &a); // F32
    let p2 = build_gguf("b", &[4], 0, &b);

    let dir = std::env::temp_dir();
    let path1 = dir.join("ffai_split_test-00001-of-00002.gguf");
    let path2 = dir.join("ffai_split_test-00002-of-00002.gguf");
    std::fs::write(&path1, &p1).unwrap();
    std::fs::write(&path2, &p2).unwrap();

    let g = Gguf::open(&path1.to_string_lossy()).unwrap();
    let da = g.dequant_f32("a").unwrap();
    let db = g.dequant_f32("b").unwrap();
    std::fs::remove_file(&path1).ok();
    std::fs::remove_file(&path2).ok();

    assert_eq!(da, vec![0.0, 1.0, 2.0, 3.0], "part-1 tensor");
    assert_eq!(db, vec![10.0, 11.0, 12.0, 13.0], "part-2 tensor");
    eprintln!("✅ split GGUF: opened part 1, read tensors from BOTH parts");
}

/// Local copy of the half→f32 conversion (the loader's is private).
fn f16_to_f32_t(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((e + 127 - 15) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}


/// Q4_K + Q6_K dequant vs gguf-py reference, on a real Qwen2.5-7B-Q4_K_M GGUF
/// (the common k-quant). Reference first-16 values were dumped with
/// `gguf.quants.dequantize`. Also exercises the split-GGUF open path (the model
/// ships as `*-00001-of-00002.gguf`). Skips if the model isn't present.
#[test]
fn q4k_q6k_dequant_match_ggufpy_qwen7b() {
    let path = std::env::var("QWEN25_7B_GGUF").unwrap_or_else(|_| {
        "/Users/tom/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf".to_string()
    });
    if !std::path::Path::new(&path).exists() {
        eprintln!("model not found at {path} — skipping");
        return;
    }
    let g = Gguf::open(&path).expect("open split gguf");

    // gguf-py reference (gguf 0.18.0, dequantize()).
    let q4k_ref: [f32; 16] = [
        -0.000382, -0.004206, 0.011090, 0.001530, -0.002294, -0.006118, 0.003442, -0.008030,
        0.001530, -0.002294, -0.008030, -0.004206, -0.008030, 0.013002, -0.002294, 0.001530,
    ];
    let q6k_ref: [f32; 16] = [
        0.000422, 0.008435, 0.003796, -0.004218, 0.000422, 0.013496, 0.001265, -0.004218,
        -0.000422, -0.009279, 0.003796, 0.004639, 0.009279, 0.002109, -0.003796, 0.000422,
    ];

    let q4 = g.dequant_f32("blk.0.attn_q.weight").expect("dequant Q4_K");
    let q6 = g.dequant_f32("blk.0.ffn_down.weight").expect("dequant Q6_K");

    let max_dev = |out: &[f32], want: &[f32; 16]| {
        (0..16).fold(0.0f32, |a, i| a.max((out[i] - want[i]).abs()))
    };
    let e4 = max_dev(&q4, &q4k_ref);
    let e6 = max_dev(&q6, &q6k_ref);
    assert!(e4 < 1e-5, "Q4_K vs gguf-py mismatch: max|Δ|={e4:.3e}");
    assert!(e6 < 1e-5, "Q6_K vs gguf-py mismatch: max|Δ|={e6:.3e}");
    eprintln!("✅ Q4_K (max|Δ|={e4:.1e}) + Q6_K (max|Δ|={e6:.1e}) dequant match gguf-py on real Qwen2.5-7B-Q4_K_M split GGUF");
}
