// Mamba2 kernel microbench — ssm_step + conv1d at NemotronH dims.
use ffai_core::{DType, Tensor};
use ffai_ops::{conv1d_causal_step, silu, ssm_step};
use ffai_cuda::CudaDevice;
use std::time::Instant;
fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
#[test]
fn mamba_kernels() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { return; };
    let d = dev.as_ref();
    let (di, m_nh, m_dh, ds, ng, kc) = (4096usize, 64usize, 64usize, 128usize, 8usize, 4usize);
    let conv_dim = di + 2 * ng * ds;
    let up = |n: usize| Tensor::new(d.upload(&tb(&vec![0.1f32; n])).unwrap(), vec![n], DType::F32);
    let (x, al, bm, cm, dk, dt, stt) = (up(di), up(m_nh), up(ng*ds), up(ng*ds), up(m_nh), up(m_nh), up(m_nh*m_dh*ds));
    let (cw, cb, cs, xbc) = (up(kc*conv_dim), up(conv_dim), up((kc-1)*conv_dim), up(conv_dim));
    for _ in 0..20 {
        let _ = ssm_step(d, &x, &al, &bm, &cm, &dk, &dt, &stt, m_dh as u32, ds as u32, m_nh as u32, (m_nh/ng) as u32).unwrap();
        let _ = conv1d_causal_step(d, &xbc, &cw, &cb, &cs, conv_dim as u32, kc as u32).unwrap();
        let _ = silu(d, &x).unwrap();
    }
    d.synchronize().unwrap();
    let it = 200;
    let t = Instant::now();
    for _ in 0..it { let _ = ssm_step(d, &x, &al, &bm, &cm, &dk, &dt, &stt, m_dh as u32, ds as u32, m_nh as u32, (m_nh/ng) as u32).unwrap(); }
    d.synchronize().unwrap();
    eprintln!("ssm_step: {:.1} us/call", t.elapsed().as_secs_f64() * 1e6 / it as f64);
    d.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..it { let _ = conv1d_causal_step(d, &xbc, &cw, &cb, &cs, conv_dim as u32, kc as u32).unwrap(); }
    d.synchronize().unwrap();
    eprintln!("conv1d_causal_step: {:.1} us/call", t.elapsed().as_secs_f64() * 1e6 / it as f64);
    let t = Instant::now();
    for _ in 0..it { let _ = silu(d, &x).unwrap(); }
    d.synchronize().unwrap();
    eprintln!("silu[4096]: {:.1} us/call", t.elapsed().as_secs_f64() * 1e6 / it as f64);
}
