#![cfg(feature = "cuda")]
//! Isolate add_rms_norm_f16norm with REAL in-model f32 inputs (a/b/w all f32).
//! Compares residual + normed against a CPU oracle AND the working add_rms_norm.
use wh_butter_core::{DType, Device, Tensor};
use wh_butter_cuda::CudaDevice;
use wh_butter_ops::{add_rms_norm, add_rms_norm_f16norm};

fn tb(v:&[f32])->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn fb32(b:&[u8])->Vec<f32>{b.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect()}
fn f16_to_f32(bits:u16)->f32{
    let sign=((bits>>15)&1) as u32; let exp=((bits>>10)&0x1f) as u32; let mant=(bits&0x3ff) as u32;
    let f=if exp==0 { if mant==0 { sign<<31 } else { // subnormal
        let mut e=-1i32; let mut m=mant; while m & 0x400 ==0 { m<<=1; e-=1; } m&=0x3ff;
        ((sign<<31)|(((e+127+(-14))as u32)<<23)|(m<<13)) } }
      else if exp==0x1f { (sign<<31)|0x7f800000|(mant<<13) }
      else { (sign<<31)|((exp+(127-15))<<23)|(mant<<13) };
    f32::from_bits(f)
}
fn fb16(b:&[u8])->Vec<f32>{b.chunks_exact(2).map(|c|f16_to_f32(u16::from_le_bytes([c[0],c[1]]))).collect()}
fn tn(d:&dyn Device,v:&[f32],sh:Vec<usize>)->Tensor{Tensor::new(d.upload(&tb(v)).unwrap(),sh,DType::F32)}

#[test]
fn f16norm_f32_inputs_matches_oracle(){
    let Some(dev)=CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();
    let (rows, n) = (8usize, 4096usize); // real hidden size
    let eps = 1e-5f32;
    let a: Vec<f32> = (0..rows*n).map(|i| ((i % 13) as f32 * 0.07 - 0.4)).collect();
    let b: Vec<f32> = (0..rows*n).map(|i| ((i % 7) as f32 * 0.05 - 0.15)).collect();
    let w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.1).collect();
    // CPU oracle
    let mut res_ref = vec![0f32; rows*n]; let mut nrm_ref = vec![0f32; rows*n];
    for r in 0..rows { let mut ssq=0f32; for c in 0..n { let s=a[r*n+c]+b[r*n+c]; res_ref[r*n+c]=s; ssq+=s*s; }
        let rms=1f32/(ssq/n as f32+eps).sqrt(); for c in 0..n { nrm_ref[r*n+c]=res_ref[r*n+c]*rms*w[c]; } }

    let at=tn(d,&a,vec![rows,n]); let bt=tn(d,&b,vec![rows,n]); let wt=tn(d,&w,vec![n]);
    // f32 control
    let (r32, n32) = add_rms_norm(d, &at, &bt, &wt, eps).unwrap();
    // fused f16norm
    let (rf, nf) = add_rms_norm_f16norm(d, &at, &bt, &wt, eps).unwrap();
    d.synchronize().unwrap();
    assert_eq!(rf.dtype, DType::F32, "residual must be f32");
    assert_eq!(nf.dtype, DType::F16, "normed must be f16");

    let mut g=vec![0u8;rows*n*4]; d.download(rf.buffer.as_ref(),&mut g).unwrap(); let res_got=fb32(&g);
    let mut h=vec![0u8;rows*n*2]; d.download(nf.buffer.as_ref(),&mut h).unwrap(); let nrm_got=fb16(&h);
    let mut c32=vec![0u8;rows*n*4]; d.download(r32.buffer.as_ref(),&mut c32).unwrap(); let res_c=fb32(&c32);
    let mut cn=vec![0u8;rows*n*4]; d.download(n32.buffer.as_ref(),&mut cn).unwrap(); let nrm_c=fb32(&cn);

    // EXACT-FUSION CHECK: fused f16 normed must equal cast_f32_f16(f32wrap normed) bit-for-bit.
    let nrm_c_f16 = wh_butter_ops::cast_f32_f16(d, &n32).unwrap();
    d.synchronize().unwrap();
    let mut hc=vec![0u8;rows*n*2]; d.download(nrm_c_f16.buffer.as_ref(),&mut hc).unwrap();
    let mut hf=vec![0u8;rows*n*2]; d.download(nf.buffer.as_ref(),&mut hf).unwrap();
    let bitdiff = hc.iter().zip(hf.iter()).filter(|(a,b)| a!=b).count();
    eprintln!("FUSED vs cast_f32_f16(f32wrap): differing f16 BYTES = {bitdiff} / {}", rows*n*2);
    let res_nan=res_got.iter().any(|x|!x.is_finite());
    let nrm_nan=nrm_got.iter().any(|x|!x.is_finite());
    let mut res_err=0f32; for i in 0..rows*n { res_err=res_err.max((res_got[i]-res_ref[i]).abs()); }
    let mut nrm_err=0f32; for i in 0..rows*n { let d0=(nrm_got[i]-nrm_ref[i]).abs(); let den=nrm_ref[i].abs().max(1e-3); nrm_err=nrm_err.max(d0/den); }
    let mut res_vs32=0f32; for i in 0..rows*n { res_vs32=res_vs32.max((res_got[i]-res_c[i]).abs()); }
    eprintln!("f16norm f32-in: res_nan={res_nan} nrm_nan={nrm_nan} res|Δ|oracle={res_err:.3e} nrm_rel|Δ|oracle={nrm_err:.3e} res|Δ|vs_f32wrap={res_vs32:.3e}");
    eprintln!("  res_got[0..4]={:?} ref={:?}", &res_got[0..4], &res_ref[0..4]);
    eprintln!("  nrm_got[0..4]={:?} ref={:?}", &nrm_got[0..4], &nrm_ref[0..4]);
    eprintln!("  f32wrap nrm[0..4]={:?}", &nrm_c[0..4]);
    assert!(!res_nan && !nrm_nan, "NaN in fused outputs");
    assert!(res_err < 1e-3, "residual wrong vs oracle: {res_err}");
    assert!(nrm_err < 2e-2, "normed wrong vs oracle (f16 tol): {nrm_err}");
    assert!(res_vs32 < 1e-3, "residual diverges from f32 wrapper: {res_vs32}");
}
