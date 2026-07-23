#![cfg(feature = "cuda")]
//! Isolate gemm_batched (device-ptr-array, broadcast offsets) vs
//! gemm_strided_batched on the SSD G1 shape: CB = C·Bᵀ, [L,L]=C[L,ds]·B[L,ds]ᵀ,
//! plus a multi-chunk fused-vs-strided regression for the null-stream race.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;

fn tb_f16(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&f| half_bits(f).to_le_bytes()).collect()
}
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
fn f16_to_f32(b: u16) -> f32 {
    let sign = ((b & 0x8000) as u32) << 16;
    let exp = ((b >> 10) & 0x1f) as u32;
    let man = (b & 0x3ff) as u32;
    if exp == 0 { return f32::from_bits(sign); }
    if exp == 0x1f { return f32::from_bits(sign | 0x7f800000 | (man << 13)); }
    f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
}
fn fill(n: usize, s: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (((i * 31 + s * 977) % 251) as f32 / 251.0 - 0.5) * 2.0 * scale).collect()
}

#[test]
fn grouped_broadcast_matches_strided() {
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();
    // SSD G1: L=128, ds=128, H=64, G=8, nc=4 → bhc=256, bgc=32 (multi-chunk).
    let nc: usize = 4;
    let (l, ds, h, g) = (128usize, 128usize, 64usize, 8usize);
    let hpg = h / g;
    let bhc = nc * h;
    let bgc = nc * g;
    let el = 2i64;

    // Per-HEAD C/B (strided reference) and per-GROUP C/B (grouped broadcast).
    // Build per-group data first, then expand to per-head by broadcast.
    let cg = fill(bgc * l * ds, 4, 3.0);
    let bg = fill(bgc * l * ds, 3, 3.0);
    // per-head expanded
    let mut ch = vec![0f32; bhc * l * ds];
    let mut bh_ = vec![0f32; bhc * l * ds];
    for bh in 0..bhc {
        let c = bh / h; let hh = bh % h; let grp = c * g + hh / hpg;
        ch[bh*l*ds..(bh+1)*l*ds].copy_from_slice(&cg[grp*l*ds..(grp+1)*l*ds]);
        bh_[bh*l*ds..(bh+1)*l*ds].copy_from_slice(&bg[grp*l*ds..(grp+1)*l*ds]);
    }

    let mk = |v: &[f32]| Tensor::new(d.upload(&tb_f16(v)).unwrap(), vec![v.len()], DType::F16);
    let cg_t = mk(&cg); let bg_t = mk(&bg);
    let ch_t = mk(&ch); let bh_t = mk(&bh_);
    let cb_strided = Tensor::new(d.upload(&vec![0u8; bhc*l*l*2]).unwrap(), vec![bhc*l*l], DType::F16);
    let cb_grouped = Tensor::new(d.upload(&vec![0u8; bhc*l*l*2]).unwrap(), vec![bhc*l*l], DType::F16);

    // Strided (per-head): X=C, W=B, out=cb. m=L,n=L,k=ds.
    let st_lds = (l*ds) as i64 * el; let st_ll = (l*l) as i64 * el;
    d.gemm_strided_batched(ch_t.buffer.as_ref(), st_lds, bh_t.buffer.as_ref(), st_lds,
        cb_strided.buffer.as_ref(), st_ll, l, l, ds, bhc, DType::F16).unwrap();

    // Grouped (broadcast offsets into per-group buffers).
    let grp_off = |bh: usize| -> usize { let c=bh/h; let hh=bh%h; (c*g+hh/hpg)*(l*ds)*(el as usize) };
    let c_offs: Vec<usize> = (0..bhc).map(grp_off).collect();
    let b_offs = c_offs.clone();
    let cb_offs: Vec<usize> = (0..bhc).map(|bh| bh*(l*l)*(el as usize)).collect();
    d.gemm_batched(cg_t.buffer.as_ref(), &c_offs, bg_t.buffer.as_ref(), &b_offs,
        cb_grouped.buffer.as_ref(), &cb_offs, l, l, ds, DType::F16).unwrap();
    d.synchronize().unwrap();

    let mut sb = vec![0u8; bhc*l*l*2];
    d.download(cb_strided.buffer.as_ref(), &mut sb).unwrap();
    let s_ref: Vec<f32> = sb.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0],c[1]]))).collect();
    d.download(cb_grouped.buffer.as_ref(), &mut sb).unwrap();
    let s_got: Vec<f32> = sb.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0],c[1]]))).collect();

    let mut emax = 0f32; let mut refmag = 0f32; let mut nanG = 0; let mut nanR = 0;
    for i in 0..s_ref.len() {
        if !s_got[i].is_finite() { nanG += 1; }
        if !s_ref[i].is_finite() { nanR += 1; }
        emax = emax.max((s_got[i]-s_ref[i]).abs());
        refmag = refmag.max(s_ref[i].abs());
    }
    eprintln!("G1 grouped-vs-strided: max|Δ|={emax:.3e} refmag={refmag:.3e} nan(grouped)={nanG} nan(strided)={nanR}");
    eprintln!("  cb_ref[0..4]={:?}", &s_ref[..4]);
    eprintln!("  cb_got[0..4]={:?}", &s_got[..4]);
    assert_eq!(nanG, 0, "grouped produced {nanG} non-finite values");
    assert!(emax/refmag.max(1e-6) < 1e-2, "grouped mismatch");
    eprintln!("✅ grouped broadcast matches strided");
}

/// Regression guard for the multi-chunk null-stream race: the fused SSD path
/// (per-group B/C + cublasGemmBatchedEx device-ptr-array) must equal the
/// strided non-fused path at nc≥4 (T=512, L=128 → 4 chunks). The bug corrupted
/// ONLY the last chunk (the recycled ptr-array buffer was overwritten by a
/// synchronous null-stream H2D before the prior GEMM read it on self.stream).
/// The fix issues that H2D on self.stream; this asserts every chunk matches.
#[test]
fn fused_vs_nonfused_multichunk() {
    use ffai_ops::ssm_prefill_scan_ssd;
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();
    let (t, h, dh, ds, ng, l) = (512usize, 64usize, 64usize, 128usize, 8usize, 128u32);
    let x = fill(t*h*dh, 1, 4.0);
    let a_log: Vec<f32> = (0..h).map(|i| -3.0 + 6.0*(i as f32/h as f32)).collect();
    let b = fill(t*ng*ds, 3, 3.0); let c = fill(t*ng*ds, 4, 3.0);
    let dsk = fill(h, 5, 1.0);
    let dt: Vec<f32> = (0..t*h).map(|i| 0.05 + 1.4*(((i*13)%11) as f32/11.0)).collect();
    let si = vec![0.0f32; h*dh*ds];
    let mkf = |v: &[f32], n: usize| Tensor::new(d.upload(&v.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<u8>>()).unwrap(), vec![n], DType::F32);
    let xt=mkf(&x,t*h*dh); let at=mkf(&a_log,h); let bt=mkf(&b,t*ng*ds); let ct=mkf(&c,t*ng*ds);
    let dtk=mkf(&dsk,h); let dtt=mkf(&dt,t*h); let sit=mkf(&si,h*dh*ds);
    let call = || ssm_prefill_scan_ssd(d,&xt,&at,&bt,&ct,&dtk,&dtt,&sit,t as u32,dh as u32,ds as u32,h as u32,ng as u32,l,None).unwrap();
    // Reference: non-fused strided path (NEMOTRON_SSD_FUSED_OFF disables fusion).
    unsafe { std::env::set_var("NEMOTRON_SSD_FUSED_OFF", "1"); }
    let (_, y_ref) = call(); d.synchronize().unwrap();
    unsafe { std::env::remove_var("NEMOTRON_SSD_FUSED_OFF"); }
    // Fused path (now the default).
    let (_, y_got) = call(); d.synchronize().unwrap();
    let mut rb = vec![0u8; t*h*dh*4]; d.download(y_ref.buffer.as_ref(), &mut rb).unwrap();
    let yr: Vec<f32> = rb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut gb = vec![0u8; t*h*dh*4]; d.download(y_got.buffer.as_ref(), &mut gb).unwrap();
    let yg: Vec<f32> = gb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let nbad = (0..yr.len()).filter(|&i| (yr[i]-yg[i]).abs() > 1.0 || !yg[i].is_finite()).count();
    eprintln!("fused vs non-fused (T=512, 4 chunks): {nbad}/{} mismatches", yr.len());
    assert_eq!(nbad, 0, "fused SSD path diverged from strided at nc=4 — null-stream race regression?");
    eprintln!("✅ fused == non-fused across all 4 chunks");
}
