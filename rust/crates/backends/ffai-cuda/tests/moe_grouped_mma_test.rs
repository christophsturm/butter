// Copyright 2026 Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Correctness for the grouped MoE GEMM (mma.sync register-O):
//! out[t,n] = Σ_k A[t,k]·W[eid(t)][n,k], sorted tokens, contiguous expert slab.
//! Reference computed on host (f32 accumulate of the same f16 inputs).

use ffai_cuda::CudaDevice;
use ffai_core::{DType, Device, Tensor};
use ffai_ops::{gemm_cublas_f32out, fwht128, gemm_fp8, fp8_quant, moe_q4_grouped_mma_dev, quantize_q4, moe_grouped_gemm_mma, moe_grouped_gemm_cutlass, cast_f32_f16, cast_f16_f32, lt_fp4_quant_g, lt_fp4_quant_g_46, dec_ue4m3,
    moe_grouped_gemm_w4a8, w4a8_actquant_grouped, w4a8_packw, w4a8_sfa_offsets,
    moe_grouped_gemm_w8a8, w8a8_actquant_grouped, w8a8_packw,
    lt_fp4_quant_slab, lt_fp4_quant_grouped, moe_grouped_gemm_cutlass_fp4, rms_norm, gemm_cublas, cast_f32_bf16, cast_bf16_f32};

fn rng(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0f32; n];
    let mut s = seed;
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.2;
    }
    v
}
fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }
fn dl(d: &dyn Device, t: &Tensor) -> Vec<f32> {
    let n = t.elem_count(); let mut b = vec![0u8; n*4];
    d.download(t.buffer.as_ref(), &mut b).unwrap();
    b.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
}
fn run(d: &dyn Device, n_exp: usize, k: usize, n: usize, groups: &[(usize, usize)]) {
    // groups: (expert_id, rows)
    let mut g_starts = vec![0usize];
    let mut expert_ids = vec![];
    for &(e, r) in groups { expert_ids.push(e); g_starts.push(g_starts.last().unwrap() + r); }
    let mt = *g_starts.last().unwrap();

    let a_f = rng(mt*k, 11);
    let w_f = rng(n_exp*n*k, 22);
    let (a16, w16) = (&a_f, &w_f); // f16 rounding folded into the 3e-2 tol

    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let a_dev = cast_f32_f16(d, &up(&a_f, vec![mt, k])).unwrap().reshaped(vec![mt, k]);
    let w_dev = cast_f32_f16(d, &up(&w_f, vec![n_exp, n, k])).unwrap().reshaped(vec![n_exp, n, k]);

    let out = moe_grouped_gemm_mma(d, &a_dev, &w_dev, &g_starts, &expert_ids, n, k).unwrap();
    let got = dl(d, &cast_f16_f32(d, &out).unwrap());

    // host reference
    let mut exp = vec![0f32; mt*n];
    for g in 0..groups.len() {
        let (eid, _) = groups[g];
        for t in g_starts[g]..g_starts[g+1] {
            for nn in 0..n {
                let mut acc = 0f32;
                for kk in 0..k { acc += a16[t*k+kk] * w16[(eid*n+nn)*k+kk]; }
                exp[t*n+nn] = acc;
            }
        }
    }
    let (mut maxa, mut maxr) = (0f32, 0f32);
    for (r, t) in exp.iter().zip(got.iter()) {
        let a = (r-t).abs(); let rel = a / r.abs().max(1e-3);
        if a>maxa {maxa=a;} if rel>maxr {maxr=rel;}
    }
    eprintln!("moe_grouped_mma mt={mt} N={n} K={k}: max_abs={maxa:.2e} max_rel={maxr:.2e}");
    assert!(maxr < 3e-2, "moe_grouped_mma: max_rel {maxr:.2e} > 3e-2 (max_abs {maxa:.2e})");
    eprintln!("moe_grouped_mma mt={mt} N={n} K={k}: PASS ✓");
}

fn run_cutlass(d: &dyn Device, n_exp: usize, k: usize, n: usize, groups: &[(usize, usize)]) {
    let mut g_starts = vec![0usize]; let mut expert_ids = vec![];
    for &(e, r) in groups { expert_ids.push(e); g_starts.push(g_starts.last().unwrap() + r); }
    let mt = *g_starts.last().unwrap();
    let a_f = rng(mt*k, 11); let w_f = rng(n_exp*n*k, 22);
    let (a16, w16) = (&a_f, &w_f);
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let a_dev = cast_f32_f16(d, &up(&a_f, vec![mt, k])).unwrap().reshaped(vec![mt, k]);
    let w_dev = cast_f32_f16(d, &up(&w_f, vec![n_exp, n, k])).unwrap().reshaped(vec![n_exp, n, k]);
    let out = match moe_grouped_gemm_cutlass(d, &a_dev, &w_dev, &g_starts, &expert_ids, n, k) {
        Ok(o) => o,
        Err(e) => { eprintln!("cutlass unavailable ({e:?}) — skip"); return; }
    };
    let got = dl(d, &cast_f16_f32(d, &out).unwrap());
    let mut exp = vec![0f32; mt*n];
    for g in 0..groups.len() { let (eid,_) = groups[g];
        for t in g_starts[g]..g_starts[g+1] { for nn in 0..n {
            let mut acc=0f32; for kk in 0..k { acc += a16[t*k+kk]*w16[(eid*n+nn)*k+kk]; }
            exp[t*n+nn]=acc; } } }
    let mut maxr=0f32; for (r,t) in exp.iter().zip(got.iter()) { let rel=(r-t).abs()/r.abs().max(1e-2); if rel>maxr {maxr=rel;} }
    eprintln!("moe_grouped_CUTLASS mt={mt} N={n} K={k}: max_rel={maxr:.2e}");
    assert!(maxr < 5e-2, "cutlass max_rel {maxr:.2e}");
    eprintln!("moe_grouped_CUTLASS mt={mt} N={n} K={k}: PASS ✓");
}

#[test]
fn moe_grouped_cutlass_nemotron() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run_cutlass(d.as_ref(), 8, 2688, 1856, &[(0,96),(3,40),(5,128),(7,17)]);
}

fn run_w4a8(d: &dyn Device, n_exp: usize, k: usize, n: usize, groups: &[(usize, usize)]) {
    let mut g_starts = vec![0usize]; let mut expert_ids = vec![];
    for &(e, r) in groups { expert_ids.push(e); g_starts.push(g_starts.last().unwrap() + r); }
    let mt = *g_starts.last().unwrap();
    let a_f = rng(mt*k, 11); let w_f = rng(n_exp*n*k, 22);
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let a_dev = cast_f32_f16(d, &up(&a_f, vec![mt, k])).unwrap().reshaped(vec![mt, k]);
    let w_dev = cast_f32_f16(d, &up(&w_f, vec![n_exp, n, k])).unwrap().reshaped(vec![n_exp, n, k]);

    // W4A8: fp8 e4m3 acts (group-aware per-32 SFA) x mxfp4 weights (per-32 SFB).
    let (sfa_off, sfa_bytes) = w4a8_sfa_offsets(&g_starts, k);
    let (a_q, a_sf) = match w4a8_actquant_grouped(d, &a_dev, &g_starts.iter().zip(g_starts.iter().skip(1)).map(|(s,e)|(e-s) as i32).collect::<Vec<_>>(), &sfa_off, sfa_bytes, mt, k) {
        Ok(o) => o, Err(e) => { eprintln!("w4a8 actquant unavailable ({e:?}) — skip"); return; } };
    let (w_pack, w_sf, sfb_eb) = match w4a8_packw(d, &w_dev, n_exp, n, k) {
        Ok(o) => o, Err(e) => { eprintln!("w4a8 packw unavailable ({e:?}) — skip"); return; } };
    let out = match moe_grouped_gemm_w4a8(d, &a_q, &a_sf, &sfa_off, &w_pack, &w_sf, sfb_eb, &g_starts, &expert_ids, n, k) {
        Ok(o) => o, Err(e) => { eprintln!("w4a8 gemm unavailable ({e:?}) — skip"); return; } };
    let got = dl(d, &cast_f16_f32(d, &out).unwrap());

    // host f32 reference + cosine
    let mut exp = vec![0f32; mt*n];
    for g in 0..groups.len() { let (eid,_) = groups[g];
        for t in g_starts[g]..g_starts[g+1] { for nn in 0..n {
            let mut acc=0f32; for kk in 0..k { acc += a_f[t*k+kk]*w_f[(eid*n+nn)*k+kk]; }
            exp[t*n+nn]=acc; } } }
    let cosf = |g: &[f32]| -> f64 {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (r,t) in exp.iter().zip(g.iter()) { dot+=(*r as f64)*(*t as f64); na+=(*r as f64).powi(2); nb+=(*t as f64).powi(2); }
        dot/(na.sqrt()*nb.sqrt()+1e-12) };
    let cos = cosf(&got);
    let rms = |v: &[f32]| (v.iter().map(|x|(*x as f64)*(*x as f64)).sum::<f64>()/v.len() as f64).sqrt();
    eprintln!("  MAG: ref_rms={:.5} w4a8_rms={:.5} ratio={:.4}", rms(&exp), rms(&got), rms(&got)/rms(&exp).max(1e-30));
    // W4A4 NVFP4 (per-16 ue4m3 weights + global, fp4 acts) on the SAME inputs.
    let (wp4, wsf4, gw4) = lt_fp4_quant_slab(d, &w_dev, n_exp, n, k).unwrap();
    let (ap4, asf4, aoff4, ags4) = lt_fp4_quant_grouped(d, &a_dev, mt, k, &g_starts).unwrap();
    let out4 = moe_grouped_gemm_cutlass_fp4(d, &ap4, &asf4, &aoff4, &ags4, &wp4, &wsf4, &gw4, &g_starts, &expert_ids, n, k).unwrap();
    let got4 = dl(d, &cast_f16_f32(d, &out4).unwrap());
    let cos4 = cosf(&got4);
    eprintln!("moe_grouped_W4A8 mt={mt} N={n} K={k}: cosine={cos:.6}  (W4A4 nvfp4 same-input cosine={cos4:.6})");
    assert!(cos >= cos4 - 1e-4, "w4a8 cosine {cos:.6} worse than W4A4 {cos4:.6}");
    eprintln!("moe_grouped_W4A8 mt={mt} N={n} K={k}: PASS ✓");
}

fn run_w8a8(d: &dyn Device, n_exp: usize, k: usize, n: usize, groups: &[(usize, usize)]) {
    let mut g_starts = vec![0usize]; let mut expert_ids = vec![];
    for &(e, r) in groups { expert_ids.push(e); g_starts.push(g_starts.last().unwrap() + r); }
    let mt = *g_starts.last().unwrap();
    let a_f = rng(mt*k, 11); let w_f = rng(n_exp*n*k, 22);
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let a_dev = cast_f32_f16(d, &up(&a_f, vec![mt, k])).unwrap().reshaped(vec![mt, k]);
    let w_dev = cast_f32_f16(d, &up(&w_f, vec![n_exp, n, k])).unwrap().reshaped(vec![n_exp, n, k]);

    // W4A8: fp8 e4m3 acts (group-aware per-32 SFA) x mxfp4 weights (per-32 SFB).
    let (sfa_off, sfa_bytes) = w4a8_sfa_offsets(&g_starts, k);
    let (a_q, a_sf) = match w8a8_actquant_grouped(d, &a_dev, &g_starts.iter().zip(g_starts.iter().skip(1)).map(|(s,e)|(e-s) as i32).collect::<Vec<_>>(), &sfa_off, sfa_bytes, mt, k) {
        Ok(o) => o, Err(e) => { eprintln!("w8a8 actquant unavailable ({e:?}) — skip"); return; } };
    let (w_pack, w_sf, sfb_eb) = match w8a8_packw(d, &w_dev, n_exp, n, k) {
        Ok(o) => o, Err(e) => { eprintln!("w8a8 packw unavailable ({e:?}) — skip"); return; } };
    let out = match moe_grouped_gemm_w8a8(d, &a_q, &a_sf, &sfa_off, &w_pack, &w_sf, sfb_eb, &g_starts, &expert_ids, n, k) {
        Ok(o) => o, Err(e) => { eprintln!("w8a8 gemm unavailable ({e:?}) — skip"); return; } };
    let got = dl(d, &cast_f16_f32(d, &out).unwrap());

    // host f32 reference + cosine
    let mut exp = vec![0f32; mt*n];
    for g in 0..groups.len() { let (eid,_) = groups[g];
        for t in g_starts[g]..g_starts[g+1] { for nn in 0..n {
            let mut acc=0f32; for kk in 0..k { acc += a_f[t*k+kk]*w_f[(eid*n+nn)*k+kk]; }
            exp[t*n+nn]=acc; } } }
    let cosf = |g: &[f32]| -> f64 {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (r,t) in exp.iter().zip(g.iter()) { dot+=(*r as f64)*(*t as f64); na+=(*r as f64).powi(2); nb+=(*t as f64).powi(2); }
        dot/(na.sqrt()*nb.sqrt()+1e-12) };
    let cos = cosf(&got);
    let rms = |v: &[f32]| (v.iter().map(|x|(*x as f64)*(*x as f64)).sum::<f64>()/v.len() as f64).sqrt();
    eprintln!("  MAG: ref_rms={:.5} w8a8_rms={:.5} ratio={:.4}", rms(&exp), rms(&got), rms(&got)/rms(&exp).max(1e-30));
    // W4A4 NVFP4 (per-16 ue4m3 weights + global, fp4 acts) on the SAME inputs.
    let (wp4, wsf4, gw4) = lt_fp4_quant_slab(d, &w_dev, n_exp, n, k).unwrap();
    let (ap4, asf4, aoff4, ags4) = lt_fp4_quant_grouped(d, &a_dev, mt, k, &g_starts).unwrap();
    let out4 = moe_grouped_gemm_cutlass_fp4(d, &ap4, &asf4, &aoff4, &ags4, &wp4, &wsf4, &gw4, &g_starts, &expert_ids, n, k).unwrap();
    let got4 = dl(d, &cast_f16_f32(d, &out4).unwrap());
    let cos4 = cosf(&got4);
    eprintln!("moe_grouped_W8A8 mt={mt} N={n} K={k}: cosine={cos:.6}  (W4A4 nvfp4 same-input cosine={cos4:.6})");
    assert!(cos >= cos4 - 1e-4, "w8a8 cosine {cos:.6} worse than W4A4 {cos4:.6}");
    eprintln!("moe_grouped_W8A8 mt={mt} N={n} K={k}: PASS ✓");
}


#[test]
fn moe_grouped_w8a8_nemotron() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run_w8a8(d.as_ref(), 8, 2688, 1856, &[(0,96),(3,40),(5,128),(7,17)]);
}
#[test]
fn moe_grouped_w8a8_down() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run_w8a8(d.as_ref(), 8, 1856, 2688, &[(0,96),(3,40),(5,128),(7,17)]);
}

fn run_w4a8_const(d: &dyn Device, k: usize, n: usize, av: f32, wv: f32) {
    let n_exp = 1; let groups = [(0usize, 128usize)];
    let mut g_starts = vec![0usize]; let mut expert_ids = vec![];
    for &(e, r) in groups.iter() { expert_ids.push(e); g_starts.push(g_starts.last().unwrap() + r); }
    let mt = *g_starts.last().unwrap();
    let a_f = vec![av; mt*k]; let w_f = vec![wv; n_exp*n*k];
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(d.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let a_dev = cast_f32_f16(d, &up(&a_f, vec![mt, k])).unwrap().reshaped(vec![mt, k]);
    let w_dev = cast_f32_f16(d, &up(&w_f, vec![n_exp, n, k])).unwrap().reshaped(vec![n_exp, n, k]);
    let (sfa_off, sfa_bytes) = w4a8_sfa_offsets(&g_starts, k);
    let grows: Vec<i32> = g_starts.windows(2).map(|w|(w[1]-w[0]) as i32).collect();
    let (a_q, a_sf) = w4a8_actquant_grouped(d, &a_dev, &grows, &sfa_off, sfa_bytes, mt, k).unwrap();
    let (w_pack, w_sf, eb) = w4a8_packw(d, &w_dev, n_exp, n, k).unwrap();
    let out = moe_grouped_gemm_w4a8(d, &a_q, &a_sf, &sfa_off, &w_pack, &w_sf, eb, &g_starts, &expert_ids, n, k).unwrap();
    let got = dl(d, &cast_f16_f32(d, &out).unwrap());
    let expv = k as f32 * av * wv;
    let gm = got.iter().sum::<f32>()/got.len() as f32;
    eprintln!("  CONST k={k} a={av} w={wv}: expect={expv:.4} got_mean={gm:.4} ratio={:.5}", gm/expv);
}

#[test]
fn moe_grouped_w4a8_const() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run_w4a8_const(d.as_ref(), 2688, 1856, 0.1, 0.1);
    run_w4a8_const(d.as_ref(), 2688, 1856, 0.05, 0.05);
    run_w4a8_const(d.as_ref(), 2688, 1856, 0.2, 0.05);
}

#[test]
fn moe_grouped_w4a8_nemotron() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run_w4a8(d.as_ref(), 8, 2688, 1856, &[(0,96),(3,40),(5,128),(7,17)]);
}

#[test]
fn moe_grouped_w4a8_stress() {
    // 30B-like: many active experts, varied group sizes (incl m=1), up shape.
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let sizes = [1usize,2,7,13,31,64,96,128,200,256,17,40,55,88,3,9];
    let groups: Vec<(usize,usize)> = sizes.iter().enumerate().map(|(e,&r)| (e, r)).collect();
    run_w4a8(d.as_ref(), 16, 2688, 1856, &groups);
}

#[test]
fn moe_grouped_w4a8_stress_down() {
    // 30B-like DOWN shape (K=inter padded to 1920), varied groups.
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let sizes = [1usize,2,7,13,31,64,96,128,200,256,17,40,55,88,3,9];
    let groups: Vec<(usize,usize)> = sizes.iter().enumerate().map(|(e,&r)| (e, r)).collect();
    run_w4a8(d.as_ref(), 16, 1920, 2688, &groups);
}

#[test]
fn moe_grouped_w4a8_down() {
    // DOWN-proj shape: K=inter=1856 (NOT a multiple of MX atom-K=128), N=hid=2688.
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run_w4a8(d.as_ref(), 8, 1856, 2688, &[(0,96),(3,40),(5,128),(7,17)]);
}

#[test]
fn moe_grouped_w4a8_down_pad() {
    // DOWN-proj K padded to 1920 (=15*128) — verifies K%128 is the constraint.
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run_w4a8(d.as_ref(), 8, 1920, 2688, &[(0,96),(3,40),(5,128),(7,17)]);
}


#[test]
fn moe_grouped_mma_small() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    run(d.as_ref(), 4, 64, 128, &[(0,20),(2,16),(3,8)]);
}
#[test]
fn moe_grouped_mma_nemotron_shape() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    // Nemotron up-proj shape: K=hid=2688, N=inter=1856, a few experts w/ ~96 rows.
    run(d.as_ref(), 8, 2688, 1856, &[(0,96),(3,40),(5,128),(7,17)]);
}

#[test]
fn moe_grouped_mma_bench() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let dev = d.as_ref();
    // Full S=2048 MoE up-proj: 128 experts × 96 rows, N=inter=1856, K=hid=2688.
    let n_exp: usize = std::env::var("NEXP").ok().and_then(|v|v.parse().ok()).unwrap_or(128); let (n, k, rows) = (1856usize, 2688usize, 96usize);
    let mut g_starts = vec![0usize]; let mut eids = vec![];
    for e in 0..n_exp { eids.push(e); g_starts.push(g_starts.last().unwrap() + rows); }
    let mt = *g_starts.last().unwrap();
    let a = cast_f32_f16(dev, &Tensor::new(dev.upload(&tb(&rng(mt*k, 1))).unwrap(), vec![mt*k], DType::F32)).unwrap().reshaped(vec![mt, k]);
    let w = cast_f32_f16(dev, &Tensor::new(dev.upload(&tb(&rng(n_exp*n*k, 2))).unwrap(), vec![n_exp*n*k], DType::F32)).unwrap().reshaped(vec![n_exp, n, k]);
    // warm
    let _ = moe_grouped_gemm_mma(dev, &a, &w, &g_starts, &eids, n, k).unwrap();
    dev.synchronize().ok();
    let iters = 30;
    let t = std::time::Instant::now();
    for _ in 0..iters { let _ = moe_grouped_gemm_mma(dev, &a, &w, &g_starts, &eids, n, k).unwrap(); }
    dev.synchronize().ok();
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let tflops = 2.0 * mt as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12;
    eprintln!("moe_grouped_mma_bench: mt={mt} N={n} K={k} → {ms:.3} ms/call, {tflops:.1} TFLOP/s (cuBLAS per-expert ref: 15.7)");
}

const E2M1L: [f32;8] = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0];
// host dequant of the lt_fp4_quant_g(_46) swizzled NVFP4 layout: v = e2m1 * dec_ue4m3(sf[soff]) * global.
fn dequant_ltg(d: &dyn Device, pack: &Tensor, sf: &Tensor, gs: &Tensor, r: usize, k: usize) -> Vec<f32> {
    let kb_=k/16; let kb4_=(kb_+3)/4; let sf_bytes=r.div_ceil(128)*512*kb4_;
    let pb = { let mut b=vec![0u8;(r*k)/2]; d.download(pack.buffer.as_ref(),&mut b).unwrap(); b };
    let sfb = { let mut b=vec![0u8;sf_bytes]; d.download(sf.buffer.as_ref(),&mut b).unwrap(); b };
    let gsb = { let mut b=vec![0u8;8]; d.download(gs.buffer.as_ref(),&mut b).unwrap(); b };
    let global = f32::from_le_bytes([gsb[0],gsb[1],gsb[2],gsb[3]]);
    let kb = k/16; let kb4 = (kb+3)/4;
    let mut out = vec![0f32; r*k];
    for rr in 0..r { for b in 0..kb {
        let soff = (rr>>7)*(512*kb4)+(b>>2)*512+(rr&31)*16+((rr>>5)&3)*4+(b&3);
        let bs = dec_ue4m3(sfb[soff]) * global;
        for i in 0..16 {
            let elem = b*16+i; let byte = (rr*k+elem)/2; let lonib = (elem%2)==0;
            let code = if lonib { pb[byte] & 0xf } else { (pb[byte]>>4) & 0xf };
            let sign = if code & 8 != 0 { -1.0 } else { 1.0 };
            out[rr*k+elem] = sign * E2M1L[(code&7) as usize] * bs;
        }
    }}
    out
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dt,mut na,mut nb)=(0f64,0f64,0f64);
    for (x,y) in a.iter().zip(b.iter()){ dt+=(*x as f64)*(*y as f64); na+=(*x as f64).powi(2); nb+=(*y as f64).powi(2); }
    dt/(na.sqrt()*nb.sqrt()+1e-30)
}
#[test]
fn lt_fp4_46_unit() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let (r,k)=(256usize,2688usize);
    let xf = rng(r*k, 7);
    let x = cast_f32_f16(d.as_ref(), &Tensor::new(d.upload(&tb(&xf)).unwrap(), vec![r*k], DType::F32).reshaped(vec![r,k])).unwrap().reshaped(vec![r,k]);
    let (p0,sf0,g0) = lt_fp4_quant_g(d.as_ref(), &x, r, k).unwrap();
    let (p6,sf6,g6) = lt_fp4_quant_g_46(d.as_ref(), &x, r, k).unwrap();
    let rec0 = dequant_ltg(d.as_ref(), &p0,&sf0,&g0, r,k);
    let rec6 = dequant_ltg(d.as_ref(), &p6,&sf6,&g6, r,k);
    eprintln!("  lt_fp4 baseline cosine-vs-input = {:.6}", cos(&xf,&rec0));
    eprintln!("  lt_fp4 _46(4/6)  cosine-vs-input = {:.6}", cos(&xf,&rec6));
    let rms=|v:&[f32]|(v.iter().map(|x|(*x as f64)*(*x as f64)).sum::<f64>()/v.len() as f64).sqrt();
    eprintln!("  rms: input={:.4} base={:.4} _46={:.4}", rms(&xf), rms(&rec0), rms(&rec6));
}

#[test]
fn moe_grouped_cutlass_bench() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let dev = d.as_ref();
    let n_exp: usize = std::env::var("NEXP").ok().and_then(|v|v.parse().ok()).unwrap_or(128); let (n, k, rows) = (1856usize, 2688usize, 96usize);
    let mut g_starts = vec![0usize]; let mut eids = vec![];
    for e in 0..n_exp { eids.push(e); g_starts.push(g_starts.last().unwrap() + rows); }
    let mt = *g_starts.last().unwrap();
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(dev.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let a_dev = cast_f32_f16(dev, &up(&rng(mt*k, 1), vec![mt, k])).unwrap().reshaped(vec![mt, k]);
    let w_dev = cast_f32_f16(dev, &up(&rng(n_exp*n*k, 2), vec![n_exp, n, k])).unwrap().reshaped(vec![n_exp, n, k]);
    let (wp4, wsf4, gw4) = lt_fp4_quant_slab(dev, &w_dev, n_exp, n, k).unwrap();
    let (ap4, asf4, aoff4, ags4) = lt_fp4_quant_grouped(dev, &a_dev, mt, k, &g_starts).unwrap();
    let _ = moe_grouped_gemm_cutlass_fp4(dev, &ap4, &asf4, &aoff4, &ags4, &wp4, &wsf4, &gw4, &g_starts, &eids, n, k).unwrap();
    dev.synchronize().ok();
    let iters = 50;
    let t = std::time::Instant::now();
    for _ in 0..iters { let _ = moe_grouped_gemm_cutlass_fp4(dev, &ap4, &asf4, &aoff4, &ags4, &wp4, &wsf4, &gw4, &g_starts, &eids, n, k).unwrap(); }
    dev.synchronize().ok();
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let tflops = 2.0 * mt as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12;
    eprintln!("moe_grouped_CUTLASS_bench: mt={mt} N={n} K={k} -> {ms:.3} ms/call, {tflops:.1} TFLOP/s (dense-fp4 ref ~240-306)");
}

#[test]
fn f16_gemm_ceiling_bench() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let dev = d.as_ref();
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(dev.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    for (m,n,k) in [(2048usize,2048usize,2048usize),(4096,4096,4096),(2048,10304,2688),(8192,8192,8192)] {
        let x = cast_f32_f16(dev, &up(&rng(m*k, 1), vec![m,k])).unwrap().reshaped(vec![m,k]);
        let w = cast_f32_f16(dev, &up(&rng(n*k, 2), vec![n,k])).unwrap().reshaped(vec![n,k]);
        let _ = gemm_cublas_f32out(dev, &x, &w, m, n, k).unwrap();
        dev.synchronize().ok();
        let iters=30; let t=std::time::Instant::now();
        for _ in 0..iters { let _=gemm_cublas_f32out(dev,&x,&w,m,n,k).unwrap(); }
        dev.synchronize().ok();
        let ms=t.elapsed().as_secs_f64()*1000.0/iters as f64;
        let tf=2.0*m as f64*n as f64*k as f64/(ms*1e-3)/1e12;
        eprintln!("f16_ceiling m={m} n={n} k={k} -> {ms:.3} ms, {tf:.1} TFLOP/s");
    }
}

#[test]
fn fwht_rotate_invariance() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let dev = d.as_ref();
    let (m,n,k) = (64usize, 64usize, 256usize); // k multiple of 128
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(dev.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let x = cast_f32_f16(dev, &up(&rng(m*k,1), vec![m,k])).unwrap().reshaped(vec![m,k]);
    let w = cast_f32_f16(dev, &up(&rng(n*k,2), vec![n,k])).unwrap().reshaped(vec![n,k]);
    let y1 = dl(dev, &gemm_cublas_f32out(dev, &x, &w, m, n, k).unwrap());
    let xh = fwht128(dev, &x, m, k).unwrap();
    let wh = fwht128(dev, &w, n, k).unwrap();
    let y2 = dl(dev, &gemm_cublas_f32out(dev, &xh, &wh, m, n, k).unwrap());
    let c = cos(&y1, &y2);
    eprintln!("fwht_rotate_invariance: cos(y1,y2)={c:.6} (want >0.999)");
    assert!(c > 0.999, "rotate-invariance broken cos={c}");
}

#[test]
fn moe_q4_grouped_bench() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let dev = d.as_ref();
    let n_exp: usize = std::env::var("NEXP").ok().and_then(|v|v.parse().ok()).unwrap_or(128); let (n, k, rows) = (1856usize, 2688usize, 96usize);
    let mt = n_exp * rows;
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(dev.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let a = cast_f32_f16(dev, &up(&rng(mt*k, 1), vec![mt,k])).unwrap().reshaped(vec![mt,k]);
    let (qs_v, sc_v) = quantize_q4(&rng(n_exp*n*k, 2), n_exp*n, k);
    let qs_b: Vec<u8> = qs_v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let qs = Tensor::new(dev.upload(&qs_b).unwrap(), vec![qs_v.len()], DType::U32);
    let sc = cast_f32_f16(dev, &up(&sc_v, vec![sc_v.len()])).unwrap();
    let offs: Vec<u8> = (0..=n_exp).flat_map(|e| ((e*rows) as u32).to_le_bytes()).collect();
    let off = Tensor::new(dev.upload(&offs).unwrap(), vec![n_exp+1], DType::U32);
    let _ = moe_q4_grouped_mma_dev(dev, &a, &qs, &sc, &off, n_exp, mt, n, k).unwrap();
    dev.synchronize().ok();
    let iters = 50;
    let t = std::time::Instant::now();
    for _ in 0..iters { let _ = moe_q4_grouped_mma_dev(dev, &a, &qs, &sc, &off, n_exp, mt, n, k).unwrap(); }
    dev.synchronize().ok();
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let tf = 2.0 * mt as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12;
    eprintln!("moe_q4_grouped_bench: mt={mt} N={n} K={k} -> {ms:.3} ms, {tf:.1} TFLOP/s (f16 ceiling ~85, cutlass-fp4 ref 60)");
}

#[test]
fn moe_cutlass_f16_bench() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let dev = d.as_ref();
    let n_exp: usize = std::env::var("NEXP").ok().and_then(|v|v.parse().ok()).unwrap_or(128); let (n, k, rows) = (1856usize, 2688usize, 96usize);
    let mut g_starts = vec![0usize]; let mut eids = vec![];
    for e in 0..n_exp { eids.push(e); g_starts.push(g_starts.last().unwrap() + rows); }
    let mt = *g_starts.last().unwrap();
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(dev.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let a = cast_f32_f16(dev, &up(&rng(mt*k, 1), vec![mt,k])).unwrap().reshaped(vec![mt,k]);
    let w = cast_f32_f16(dev, &up(&rng(n_exp*n*k, 2), vec![n_exp*n, k])).unwrap().reshaped(vec![n_exp*n, k]);
    let _ = moe_grouped_gemm_cutlass(dev, &a, &w, &g_starts, &eids, n, k).unwrap();
    dev.synchronize().ok();
    let iters = 50;
    let t = std::time::Instant::now();
    for _ in 0..iters { let _ = moe_grouped_gemm_cutlass(dev, &a, &w, &g_starts, &eids, n, k).unwrap(); }
    dev.synchronize().ok();
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let tf = 2.0 * mt as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12;
    eprintln!("moe_cutlass_f16_bench: mt={mt} N={n} K={k} -> {ms:.3} ms, {tf:.1} TFLOP/s (q4-grouped ref 13.7, cutlass-fp4 60)");
}

// Isolates the bf16-stream NaN: run rms_norm + gemm_tc in f32 vs bf16, report cosine + NaN.
#[test]
fn bf16_kernel_isolate() {
    let Some(d) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let dev = d.as_ref();
    let (rows, hid) = (2048usize, 2688usize);
    let up = |v: &[f32], sh: Vec<usize>| -> Tensor {
        Tensor::new(dev.upload(&tb(v)).unwrap(), vec![v.len()], DType::F32).reshaped(sh) };
    let x = up(&rng(rows*hid, 7), vec![rows, hid]);
    let w = up(&rng(hid, 9), vec![hid]);
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let (mut dot, mut na, mut nb) = (0f64,0f64,0f64);
        for i in 0..a.len() { dot+=a[i] as f64*b[i] as f64; na+=(a[i] as f64).powi(2); nb+=(b[i] as f64).powi(2); }
        (dot/(na.sqrt()*nb.sqrt())) as f32 };
    let has_nan = |v: &[f32]| v.iter().any(|x| x.is_nan());

    // --- rms_norm f32 vs bf16(wide) ---
    let n_f32 = dl(dev, &rms_norm(dev, &x, &w, 1e-5).unwrap());
    let xb = cast_f32_bf16(dev, &x).unwrap();
    let wb = cast_f32_bf16(dev, &w).unwrap();
    let n_bf = rms_norm(dev, &xb, &wb, 1e-5).unwrap();
    let n_bf_f = dl(dev, &cast_bf16_f32(dev, &n_bf).unwrap());
    eprintln!("RMS_NORM bf16-wide: NaN={}  cosine_vs_f32={:.6}", has_nan(&n_bf_f), cos(&n_f32, &n_bf_f));

    // --- gemm_tc f32-ref vs bf16 ---  out[rows,hid] = x[rows,hid] . W[hid,hid]^T
    let wsq = up(&rng(hid*hid, 11), vec![hid, hid]);
    let xh = cast_f32_f16(dev, &x).unwrap();
    let wh = cast_f32_f16(dev, &wsq).unwrap();
    let g_f16 = dl(dev, &cast_f16_f32(dev, &gemm_cublas(dev, &xh, &wh, rows, hid, hid).unwrap()).unwrap());
    let xbb = cast_f32_bf16(dev, &x).unwrap();
    let wbb = cast_f32_bf16(dev, &wsq).unwrap();
    let g_bf = dl(dev, &cast_bf16_f32(dev, &gemm_cublas(dev, &xbb, &wbb, rows, hid, hid).unwrap()).unwrap());
    eprintln!("GEMM_TC bf16: NaN={}  cosine_vs_f16={:.6}", has_nan(&g_bf), cos(&g_f16, &g_bf));
}

#[test]
fn fp8_rate_bench() {
    let Some(d)=CudaDevice::create().expect("cuda") else { eprintln!("no CUDA"); return; };
    let dev=d.as_ref();
    let up=|v:&[f32],sh:Vec<usize>|->Tensor{ Tensor::new(dev.upload(&tb(v)).unwrap(),vec![v.len()],DType::F32).reshaped(sh) };
    for (m,n,k) in [(2048usize,2688usize,2688usize),(2048,8192,2688),(4096,4096,4096)] {
        let x=up(&rng(m*k,1),vec![m,k]); let w=up(&rng(n*k,2),vec![n,k]);
        // fp8 path
        let (xq,xsc)=fp8_quant(dev,&cast_f32_f16(dev,&x).unwrap(),m*k,1.0).unwrap();
        let (wq,wsc)=fp8_quant(dev,&cast_f32_f16(dev,&w).unwrap(),n*k,1.0).unwrap();
        let _=gemm_fp8(dev,&xq,&xsc,&wq,&wsc,m,n,k,true).unwrap(); dev.synchronize().ok();
        let it=50; let t=std::time::Instant::now();
        for _ in 0..it { let _=gemm_fp8(dev,&xq,&xsc,&wq,&wsc,m,n,k,true).unwrap(); }
        dev.synchronize().ok(); let ms8=t.elapsed().as_secs_f64()*1000.0/it as f64;
        let tf8=2.0*m as f64*n as f64*k as f64/(ms8*1e-3)/1e12;
        // f16 path
        let xh=cast_f32_f16(dev,&x).unwrap(); let wh=cast_f32_f16(dev,&w).unwrap();
        let _=gemm_cublas(dev,&xh,&wh,m,n,k).unwrap(); dev.synchronize().ok();
        let t=std::time::Instant::now();
        for _ in 0..it { let _=gemm_cublas(dev,&xh,&wh,m,n,k).unwrap(); }
        dev.synchronize().ok(); let ms16=t.elapsed().as_secs_f64()*1000.0/it as f64;
        let tf16=2.0*m as f64*n as f64*k as f64/(ms16*1e-3)/1e12;
        eprintln!("fp8_rate_bench m={m} n={n} k={k}: fp8 {ms8:.3}ms {tf8:.1} TFLOP/s | f16 {ms16:.3}ms {tf16:.1} TFLOP/s | fp8/f16 speedup {:.2}x", tf8/tf16);
    }
}
