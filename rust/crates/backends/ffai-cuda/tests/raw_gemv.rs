//! Raw-NVRTC GEMV microbench: does an EXPLICIT uint4 weight load (impossible in
//! the #[kernel] DSL — its auto-vectorizer won't emit it) beat the DSL's scalar
//! strided load? Both kernels hand-written CUDA-C, compiled via the same NVRTC
//! path, timed in one process on identical buffers. If uint4 ≫ scalar, raw
//! kernels are the real path past the DSL plateau.
//!
//! cargo test --release -p ffai-cuda --features cuda --test raw_gemv -- --nocapture

#![cfg(feature = "cuda")]

use metaltile_runtime::CudaDevice;
use std::os::raw::c_void;
use std::time::Instant;

const SCALAR: &str = r#"
extern "C" __global__ void rawscalar(const unsigned* qs,const float* d,const float* x,float* out,int k_in,int rpg){
  int row=blockIdx.x;int lane=threadIdx.x;int bpr=k_in>>5;int nwords=bpr*4;
  const unsigned* qrow=qs+(size_t)row*nwords;const float* drow=d+(size_t)row*bpr;
  const float* xrow=x+(size_t)(row/rpg)*k_in;float dot=0.f;
  for(int j=lane;j<nwords;j+=32){int blk=j>>2;int sub=j&3;unsigned p=qrow[j];float sc=drow[blk];
    const float* xb=xrow+(blk<<5)+(sub<<3);float a=0.f;
    #pragma unroll
    for(int i=0;i<8;i++){int nb=(p>>(i*4))&0xf;a+=(float)(nb>7?nb-16:nb)*xb[i];}
    dot+=sc*a;}
  #pragma unroll
  for(int o=16;o;o>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,o);
  if(lane==0)out[row]=dot;
}
"#;

const UINT4: &str = r#"
extern "C" __global__ void rawuint4(const unsigned* qs,const float* d,const float* x,float* out,int k_in,int rpg){
  int row=blockIdx.x;int lane=threadIdx.x;int bpr=k_in>>5;
  const unsigned* qrow=qs+(size_t)row*bpr*4;const float* drow=d+(size_t)row*bpr;
  const float* xrow=x+(size_t)(row/rpg)*k_in;float dot=0.f;
  for(int b=lane;b<bpr;b+=32){
    uint4 w=*reinterpret_cast<const uint4*>(qrow+b*4); // explicit 128-bit coalesced load
    float sc=drow[b];const float* xb=xrow+(b<<5);
    unsigned ws[4]={w.x,w.y,w.z,w.w};float a=0.f;
    #pragma unroll
    for(int wi=0;wi<4;wi++){unsigned p=ws[wi];
      #pragma unroll
      for(int i=0;i<8;i++){int nb=(p>>(i*4))&0xf;a+=(float)(nb>7?nb-16:nb)*xb[wi*8+i];}}
    dot+=sc*a;}
  #pragma unroll
  for(int o=16;o;o>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,o);
  if(lane==0)out[row]=dot;
}
"#;

const F16SCALE: &str = r#"
#include <cuda_fp16.h>
extern "C" __global__ void rawf16s(const unsigned* qs,const __half* d,const float* x,float* out,int k_in,int rpg){
  int row=blockIdx.x;int lane=threadIdx.x;int bpr=k_in>>5;int nwords=bpr*4;
  const unsigned* qrow=qs+(size_t)row*nwords;const __half* drow=d+(size_t)row*bpr;
  const float* xrow=x+(size_t)(row/rpg)*k_in;float dot=0.f;
  for(int j=lane;j<nwords;j+=32){int blk=j>>2;int sub=j&3;unsigned p=qrow[j];float sc=__half2float(drow[blk]);
    const float* xb=xrow+(blk<<5)+(sub<<3);float a=0.f;
    #pragma unroll
    for(int i=0;i<8;i++){int nb=(p>>(i*4))&0xf;a+=(float)(nb>7?nb-16:nb)*xb[i];}
    dot+=sc*a;}
  #pragma unroll
  for(int o=16;o;o>>=1)dot+=__shfl_down_sync(0xffffffffu,dot,o);
  if(lane==0)out[row]=dot;
}
"#;

fn bench(dev: &CudaDevice, name: &str) {
    // Representative decode GEMVs (m_out, k_in).
    for &(m, k, lbl) in &[(10304usize, 2688usize, "in_proj"), (131072usize, 2688usize, "lm_head"), (2688usize, 2688usize, "o_proj")] {
        let bpr = k / 32;
        let nwords = bpr * 4;
        let scale_sz = if name == "f16scale" { 2 } else { 4 };
        let qbytes = vec![0u8; m * nwords * 4];
        let dbytes = vec![0u8; m * bpr * scale_sz];
        let xbytes = vec![1u8; k * 4];
        let qp = dev.alloc_raw(qbytes.len()).unwrap(); dev.htod(qp, &qbytes).unwrap();
        let dp = dev.alloc_raw(dbytes.len()).unwrap(); dev.htod(dp, &dbytes).unwrap();
        let xp = dev.alloc_raw(xbytes.len()).unwrap(); dev.htod(xp, &xbytes).unwrap();
        let op = dev.alloc_raw(m * 4).unwrap();

        let (src, fname) = if name == "uint4" { (UINT4, "rawuint4") } else if name == "f16scale" { (F16SCALE, "rawf16s") } else { (SCALAR, "rawscalar") };
        let module = dev.compile(src, "raw.cu").unwrap();
        let func = module.function(fname).unwrap();

        let (mut a_q, mut a_d, mut a_x, mut a_o) = (qp, dp, xp, op);
        let (mut a_k, mut a_r) = (k as i32, m as i32);
        let mut args: [*mut c_void; 6] = [
            &mut a_q as *mut _ as *mut c_void, &mut a_d as *mut _ as *mut c_void,
            &mut a_x as *mut _ as *mut c_void, &mut a_o as *mut _ as *mut c_void,
            &mut a_k as *mut _ as *mut c_void, &mut a_r as *mut _ as *mut c_void,
        ];
        for _ in 0..5 { dev.launch_async(func, [m as u32,1,1], [32,1,1], 0, &mut args).unwrap(); }
        dev.synchronize().unwrap();
        let it = 100;
        let t = Instant::now();
        for _ in 0..it { dev.launch_async(func, [m as u32,1,1], [32,1,1], 0, &mut args).unwrap(); }
        dev.synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / it as f64;
        let gb = (m * nwords * 4) as f64 / 1e9;
        eprintln!("  [{name}] {lbl} ({m}x{k}): {us:.1} us/call, {:.0} GB/s", gb / (us / 1e6));
    }
}

#[test]
fn raw_scalar_vs_uint4() {
    let dev = match CudaDevice::create().unwrap() { Some(d) => d, None => { eprintln!("no cuda"); return; } };
    eprintln!("──── raw GEMV: scalar (DSL-equivalent) vs explicit uint4 ────");
    bench(&dev, "scalar");
    bench(&dev, "f16scale");
    bench(&dev, "uint4");
}
