#![cfg(feature = "cuda")]
//! Isolate slice() on f16 with a large element offset (the lm_head last-token slice).
use wh_butter_core::{DType, Device, Tensor};
use wh_butter_cuda::CudaDevice;
use wh_butter_ops::{slice, cast_f32_f16, cast_f16_f32};

fn tb(v:&[f32])->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn fb32(b:&[u8])->Vec<f32>{b.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect()}
fn tn(d:&dyn Device,v:&[f32])->Tensor{Tensor::new(d.upload(&tb(v)).unwrap(),vec![v.len()],DType::F32)}

#[test]
fn slice_f16_large_offset(){
    let Some(dev)=CudaDevice::create().expect("cuda") else { return; };
    let d=dev.as_ref();
    let (s,hid)=(2048usize,4096usize);
    let full:Vec<f32>=(0..s*hid).map(|i| ((i%101) as f32*0.013-0.5)).collect();
    let f32t=tn(d,&full);
    let f16t=cast_f32_f16(d,&f32t).unwrap();
    d.synchronize().unwrap();
    let off=(s-1)*hid;
    // slice on f32 (baseline path)
    let sl32=slice(d,&f32t,off,hid).unwrap();
    // slice on f16 then cast (fused path)
    let sl16=slice(d,&f16t,off,hid).unwrap();
    let sl16_f32=cast_f16_f32(d,&sl16).unwrap();
    d.synchronize().unwrap();
    // reference path: cast full f16->f32 FIRST then slice (the working fix)
    let f16_full_f32=cast_f16_f32(d,&f16t).unwrap();
    let sl_ref=slice(d,&f16_full_f32,off,hid).unwrap();
    d.synchronize().unwrap();
    let mut a=vec![0u8;hid*4]; d.download(sl32.buffer.as_ref(),&mut a).unwrap(); let g32=fb32(&a);
    let mut b=vec![0u8;hid*4]; d.download(sl16_f32.buffer.as_ref(),&mut b).unwrap(); let g16=fb32(&b);
    let mut c=vec![0u8;hid*4]; d.download(sl_ref.buffer.as_ref(),&mut c).unwrap(); let gr=fb32(&c);
    let mut e_sl16=0f32; for i in 0..hid { e_sl16=e_sl16.max((g16[i]-full[off+i]).abs()); }
    let mut e_ref=0f32;  for i in 0..hid { e_ref=e_ref.max((gr[i]-full[off+i]).abs()); }
    let mut e_32=0f32;   for i in 0..hid { e_32=e_32.max((g32[i]-full[off+i]).abs()); }
    eprintln!("slice@off={off} hid={hid}: f32|Δ|={e_32:.3e}  slice-f16-then-cast|Δ|={e_sl16:.3e}  cast-then-slice|Δ|={e_ref:.3e}");
    eprintln!("  f32   [0..4]={:?}", &g32[0..4]);
    eprintln!("  sl16  [0..4]={:?}", &g16[0..4]);
    eprintln!("  ref   [0..4]={:?}", &gr[0..4]);
    eprintln!("  truth [0..4]={:?}", &full[off..off+4]);
}
