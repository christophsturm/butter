// F-85 round-10 lever 2: isolated marlin-vs-gemv_q4 microbench at the
// small_proj shapes (attn wk/wv/wo, gdn wz/wo) -- per the round-10 brief,
// "marlin only wins where measured": route only shapes where this shows a
// real win, since marlin's fixed per-call kernel overhead may dominate at
// these smaller N (unlike the already-routed 5 big shapes).
use wh_butter_core::{DType, Tensor};
use wh_butter_ops as ops;
use wh_butter_cuda::CudaDevice;
use std::time::Instant;

fn tbf(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tbu(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
// Verbatim copy of qwen35.rs's private helper (same rounding behavior as
// the real load path -- this microbench must match production bit-for-bit
// in how it builds inputs, only the timing loop differs).
fn f32s_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    fn half_bits(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 112;
        if e <= 0 {
            return sign;
        }
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        let m = (x >> 13) & 0x3ff;
        let round = (x >> 12) & 1;
        let v = ((e as u32) << 10) | m;
        sign | ((v + round) as u16)
    }
    v.iter().flat_map(|&f| half_bits(f).to_le_bytes()).collect()
}

const GB10_SMS: i32 = 48;
const MARLIN_MAX_SHARED_MEM: i32 = 101376;
const MARLIN_GROUP_SIZE: usize = 32;

#[test]
fn qwen35_marlin_smallproj_microbench() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();
    let shapes: &[(usize, usize, &str)] = &[
        (1024, 5120, "full_wk_or_wv"),
        (5120, 6144, "full_wo"),
        (6144, 5120, "gdn_wz"),
        (5120, 6144, "gdn_wo"),
    ];
    // Routing triple for M=1 (same for every shape).
    let off_bytes: Vec<u8> = [0i32, 1i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let off = Tensor::new(d.upload(&off_bytes).unwrap(), vec![2], DType::I32);
    let routing = ops::marlin_build_routing(d, &off, 1, 8, 1).unwrap();

    for &(m, k, lbl) in shapes {
        let mut s = 0x1u32;
        let mut rng = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as f32 / u32::MAX as f32) - 0.5 };
        let w: Vec<f32> = (0..m * k).map(|_| rng()).collect();
        let x: Vec<f32> = (0..k).map(|_| rng()).collect();

        // --- gemv_q4 baseline ---
        let (qs, sc) = ops::quantize_q4(&w, m, k);
        let qt = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
        let st = Tensor::new(d.upload(&tbf(&sc)).unwrap(), vec![sc.len()], DType::F32);
        let xt = Tensor::new(d.upload(&tbf(&x)).unwrap(), vec![k], DType::F32);
        for _ in 0..5 { let _ = ops::gemv_q4(d, &qt, &st, &xt, m, k, m).unwrap(); }
        d.synchronize().unwrap();
        let it = 200;
        let t = Instant::now();
        for _ in 0..it { let _ = ops::gemv_q4(d, &qt, &st, &xt, m, k, m).unwrap(); }
        d.synchronize().unwrap();
        let gemv_us = t.elapsed().as_secs_f64() * 1e6 / it as f64;

        // --- marlin U4B8 f16 ---
        let (bqw, scales_f32) = ops::quantize_u4b8_gptq_g32(&w, m, k);
        let scales_bytes = f32s_to_f16_bytes(&scales_f32);
        let bqw_t = Tensor::new(d.upload(&tbu(&bqw)).unwrap(), vec![bqw.len()], DType::U32);
        let scales_raw = Tensor::new(d.upload(&scales_bytes).unwrap(), vec![scales_bytes.len() / 2], DType::F16);
        let b_repacked = ops::marlin_repack(d, &bqw_t, k, m, GB10_SMS, MARLIN_MAX_SHARED_MEM).unwrap();
        let num_groups = k / MARLIN_GROUP_SIZE;
        let b_scales = ops::marlin_permute_scales(d, &scales_raw, num_groups, m).unwrap();
        let (thread_k, thread_n, blocks_per_sm) =
            ops::marlin_pick_config(d, 1, m, k, MARLIN_GROUP_SIZE, GB10_SMS).unwrap();
        let (workspace, c_tmp) = ops::marlin_gemm_u4b8_workspace(d, m, GB10_SMS).unwrap();
        let x16_bytes = f32s_to_f16_bytes(&x);
        let x16 = Tensor::new(d.upload(&x16_bytes).unwrap(), vec![k], DType::F16);

        let call = || {
            ops::marlin_gemm_u4b8_f16(
                d, &x16, &b_repacked, &b_scales,
                &routing.0, &routing.1, &routing.2,
                &workspace, &c_tmp,
                1, m, k, MARLIN_GROUP_SIZE, GB10_SMS,
                thread_k, thread_n, blocks_per_sm,
            ).unwrap()
        };
        for _ in 0..5 { let _ = call(); }
        d.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..it { let _ = call(); }
        d.synchronize().unwrap();
        let marlin_us = t.elapsed().as_secs_f64() * 1e6 / it as f64;

        let verdict = if marlin_us < gemv_us { "MARLIN WINS" } else { "gemv wins" };
        eprintln!(
            "SMALLPROJ {lbl:16} m={m:6} k={k:6}  gemv_q4={gemv_us:7.1}us  marlin={marlin_us:7.1}us  ({:+.1}%)  {verdict}",
            (marlin_us - gemv_us) / gemv_us * 100.0
        );
    }
}
