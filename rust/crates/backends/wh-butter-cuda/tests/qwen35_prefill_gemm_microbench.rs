// F-85 prefill wave: per-shape GEMM microbench for the batched chunked-
// prefill path -- `gemm_q4_mpp` (the general-M/K batched Q4 GEMM,
// `laguna.rs`'s established non-marlin prefill path) vs a batched Marlin
// U4B8 GEMM (the same kernel family qwen35's decode-M=1 path already uses,
// `ops::marlin_gemm_u4b8_f16`, run at prefill M instead of 1) at M=512 and
// M=2048 (the task's pp512/pp2048 operating points), across every distinct
// dense-projection shape this model's prefill hot path calls
// (`Q4Dense::gemm` in `qwen35.rs`).
//
// Self-contained (does not reach into `qwen35.rs`'s private `Q4Dense`/
// `MarlinWeight` types -- mirrors `qwen35_marlin_smallproj_microbench.rs`'s
// pattern of building its own isolated marlin buffers via the public
// `ops::*` API only).
use wh_butter_core::{DType, Tensor};
use wh_butter_ops as ops;
use wh_butter_cuda::CudaDevice;
use std::time::Instant;

fn tbf(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn tbu(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
// Verbatim copy of qwen35.rs's private helper -- same rounding behavior as
// the real load path.
fn f32s_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    fn half_bits(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 112;
        if e <= 0 { return sign; }
        if e >= 0x1f { return sign | 0x7c00; }
        let m = (x >> 13) & 0x3ff;
        let round = (x >> 12) & 1;
        let v = ((e as u32) << 10) | m;
        sign | ((v + round) as u16)
    }
    v.iter().flat_map(|&f| half_bits(f).to_le_bytes()).collect()
}

const GB10_SMS: i32 = 48;
const MARLIN_MAX_SHARED_MEM: i32 = 101376;

fn rng_vec(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    let mut next = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s as f32 / u32::MAX as f32) - 0.5 };
    (0..n).map(|_| next()).collect()
}

#[test]
fn qwen35_prefill_gemm_microbench() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA, skip"); return; };
    let d = dev.as_ref();

    // (m, k, label) -- every distinct dense-projection shape prefill's
    // Q4Dense::gemm calls (see qwen35.rs's GdnWeights/FullAttnWeights/
    // DenseFfn field docs for the (m,k) provenance of each).
    let shapes: &[(usize, usize, &str)] = &[
        (12288, 5120, "attn_wq"),
        (1024, 5120, "attn_wk_wv"),
        (5120, 6144, "attn_wo__gdn_wo"),
        (10240, 5120, "gdn_wqkv"),
        (6144, 5120, "gdn_wz"),
        (17408, 5120, "ffn_gate_up"),
        (5120, 17408, "ffn_down"),
    ];
    let ms: &[usize] = &[512, 2048];

    println!("shape,m,k,M,gemm_q4_mpp_us,marlin_u4b8_us,winner");
    for &(m, k, lbl) in shapes {
        let w = rng_vec(m * k, 0x9e3779b9u32.wrapping_add(m as u32).wrapping_add(k as u32));

        // ---- gemm_q4_mpp operand (plain Q4, same layout Q4Dense.qs/scales use) ----
        let (qs, sc_f32) = ops::quantize_q4(&w, m, k);
        let sc_bytes = f32s_to_f16_bytes(&sc_f32);
        let qs_t = Tensor::new(d.upload(&tbu(&qs)).unwrap(), vec![qs.len()], DType::U32);
        let sc_t = Tensor::new(d.upload(&sc_bytes).unwrap(), vec![sc_bytes.len() / 2], DType::F16);

        // ---- marlin U4B8 operand (shape-only; reused across every M) ----
        let (bqw, mscales_f32) = ops::quantize_u4b8_gptq_g32(&w, m, k);
        // NOTE: quantize_u4b8_gptq_g32 is hardwired GS=32; the production
        // g=64 record config (QWEN35_MARLIN_G=64) uses the generalized
        // group-size quantizer instead. Group size does not change the GEMM
        // kernel's per-call cost profile materially (same tile geometry,
        // different #scale-groups read) -- GS=32 here isolates the M-scaling
        // question this microbench is actually asking, kept simple/self-
        // contained rather than re-deriving the g=64 quantizer.
        let sc_bytes_m = f32s_to_f16_bytes(&mscales_f32);
        let bqw_t = Tensor::new(d.upload(&tbu(&bqw)).unwrap(), vec![bqw.len()], DType::U32);
        let scales_raw = Tensor::new(d.upload(&sc_bytes_m).unwrap(), vec![sc_bytes_m.len() / 2], DType::F16);
        let b_repacked = ops::marlin_repack(d, &bqw_t, k, m, GB10_SMS, MARLIN_MAX_SHARED_MEM).unwrap();
        let num_groups = k / 32;
        let b_scales = ops::marlin_permute_scales(d, &scales_raw, num_groups, m).unwrap();
        let (workspace, c_tmp) = ops::marlin_gemm_u4b8_workspace(d, m, GB10_SMS).unwrap();

        for &mt in ms {
            let x: Vec<f32> = rng_vec(mt * k, 0xdeadbeefu32.wrapping_add(mt as u32));
            let x_t = Tensor::new(d.upload(&tbf(&x)).unwrap(), vec![mt, k], DType::F32);
            let x16 = Tensor::new(d.upload(&f32s_to_f16_bytes(&x)).unwrap(), vec![mt, k], DType::F16);

            // ---- gemm_q4_mpp timing ----
            for _ in 0..3 { let _ = ops::gemm_q4_mpp(d, &x_t, &qs_t, &sc_t, mt, m, k).unwrap(); }
            d.synchronize().unwrap();
            let it = 20;
            let t0 = Instant::now();
            for _ in 0..it { let _ = ops::gemm_q4_mpp(d, &x_t, &qs_t, &sc_t, mt, m, k).unwrap(); }
            d.synchronize().unwrap();
            let mpp_us = t0.elapsed().as_secs_f64() * 1e6 / it as f64;

            // ---- marlin U4B8 timing (routing + auto-config resolved for THIS mt) ----
            let off_bytes: Vec<u8> = [0i32, mt as i32].iter().flat_map(|v| v.to_le_bytes()).collect();
            let off = Tensor::new(d.upload(&off_bytes).unwrap(), vec![2], DType::I32);
            let routing = ops::marlin_build_routing(d, &off, 1, 8, mt).unwrap();
            let (thread_k, thread_n, blocks_per_sm) =
                ops::marlin_pick_config(d, mt, m, k, 32, GB10_SMS).unwrap();
            let call = || {
                ops::marlin_gemm_u4b8_f16(
                    d, &x16, &b_repacked, &b_scales,
                    &routing.0, &routing.1, &routing.2,
                    &workspace, &c_tmp,
                    mt, m, k, 32, GB10_SMS,
                    thread_k, thread_n, blocks_per_sm,
                ).unwrap()
            };
            for _ in 0..3 { let _ = call(); }
            d.synchronize().unwrap();
            let t1 = Instant::now();
            for _ in 0..it { let _ = call(); }
            d.synchronize().unwrap();
            let marlin_us = t1.elapsed().as_secs_f64() * 1e6 / it as f64;

            let winner = if marlin_us < mpp_us { "marlin" } else { "gemm_q4_mpp" };
            println!("{lbl},{m},{k},{mt},{mpp_us:.1},{marlin_us:.1},{winner}");
        }
    }
}
