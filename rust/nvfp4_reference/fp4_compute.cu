#include <cstdio>
// Validate the NVFP4 mma COMPUTES (not just assembles) on GB10/sm_121a.
// A[16x64]=B[8x64]=e2m1(1.0)=0x2 nibble → 0x22222222/reg; scales=ue4m3(1.0)=0x38 → 0x38383838.
// D[i][j] = sum_{k0..63} 1*1 * sA*sB = 64. Layout-independent (all elems equal).
__global__ void k(float* D){
  unsigned a0=0x22222222,a1=0x22222222,a2=0x22222222,a3=0x22222222;
  unsigned b0=0x22222222,b1=0x22222222;
  unsigned sA=0x38383838, sB=0x38383838;
  float c0=0,c1=0,c2=0,c3=0;
  asm volatile(
   "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
   : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
   : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sA),"r"(sB));
  int lane=threadIdx.x;
  // C/D layout: c0,c1 -> row=gid, col=tig*2+{0,1}; c2,c3 -> row=gid+8
  int gid=lane>>2, tig=lane&3;
  D[(gid)*8 + tig*2+0]=c0; D[(gid)*8 + tig*2+1]=c1;
  D[(gid+8)*8 + tig*2+0]=c2; D[(gid+8)*8 + tig*2+1]=c3;
}
int main(){
  float* d; cudaMalloc(&d,16*8*4); cudaMemset(d,0,16*8*4);
  k<<<1,32>>>(d);
  cudaError_t e=cudaDeviceSynchronize();
  if(e){printf("LAUNCH ERR: %s\n",cudaGetErrorString(e));return 1;}
  float h[128]; cudaMemcpy(h,d,16*8*4,cudaMemcpyDeviceToHost);
  int ok=0,bad=0; float mn=1e9,mx=-1e9;
  for(int i=0;i<128;i++){ if(h[i]==64.0f)ok++; else bad++; if(h[i]<mn)mn=h[i]; if(h[i]>mx)mx=h[i]; }
  printf("D[0..7]: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
  printf("==64: %d/128   range[%.3f,%.3f]\n",ok,mn,mx);
  printf(ok==128?"COMPUTE OK (all 64)\n":"MISMATCH (expected all 64)\n");
  return 0;
}
