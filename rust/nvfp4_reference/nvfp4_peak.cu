#include <cstdio>
__global__ void peak(float* out,int iters){
  unsigned a0=0x24242424,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,sA=0x38383838,sB=0x38383838;
  float c0=0,c1=0,c2=0,c3=0;
  for(int i=0;i<iters;i++){
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
     :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sA),"r"(sB));
    a0+=c0>1e30f?1:0; // fake dep to prevent hoist, never taken
  }
  if(c0==1234.5f)out[threadIdx.x]=c0+c1+c2+c3;
}
int main(){
  int blocks=264, tpb=256, iters=20000; // 264*8 warps
  float* d;cudaMalloc(&d,4096);
  peak<<<blocks,tpb>>>(d,5); cudaDeviceSynchronize(); // warmup
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
  cudaEventRecord(s); peak<<<blocks,tpb>>>(d,iters); cudaEventRecord(e);cudaEventSynchronize(e);
  float ms=0;cudaEventElapsedTime(&ms,s,e);
  double warps=(double)blocks*(tpb/32);
  double flop=warps*(double)iters*16*8*64*2;
  printf("mxf4nvf4 m16n8k64 peak: %.1f TFLOP/s  (%.2f ms, %.0f warps x %d iters)\n",flop/(ms*1e-3)/1e12,ms,warps,iters);
  // also int4 m16n8k64 for comparison (s32 acc)
  return 0;
}
