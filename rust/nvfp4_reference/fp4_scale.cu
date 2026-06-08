#include <cstdio>
#include <cstring>
// Derive scale-block map. A=B=e2m1 1.0. sB=all 1.0. sA: base all 1.0(0x38), except thread TL byte TB=2.0(0x40).
// D[m][n]=sum_{b=0..3} 16 * sA_eff[m][b]  (A=B=1,sB=1). Baseline all = 64. The (m,b) that (TL,TB) drives -> D[m]+=16.
__global__ void k(float* D,int TL,int TB){
  unsigned a0=0x22222222,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0;
  unsigned sA=0x38383838, sB=0x38383838;
  if((int)threadIdx.x==TL){ sA = (sA & ~(0xFFu<<(TB*8))) | (0x40u<<(TB*8)); }
  float c0=0,c1=0,c2=0,c3=0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
   :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sA),"r"(sB));
  int g=threadIdx.x>>2,t=threadIdx.x&3;
  D[g*8+t*2]=c0; D[g*8+t*2+1]=c1; D[(g+8)*8+t*2]=c2; D[(g+8)*8+t*2+1]=c3;
}
int main(){
  float* d; cudaMalloc(&d,128*4); float h[128];
  printf("scaleA: (thread TL, byte TB) -> rows where D jumped 64->80 (delta from 2.0x scale)\n");
  for(int TL=0;TL<32;TL++) for(int TB=0;TB<4;TB++){
    cudaMemset(d,0,128*4); k<<<1,32>>>(d,TL,TB); cudaDeviceSynchronize();
    cudaMemcpy(h,d,128*4,cudaMemcpyDeviceToHost);
    char buf[128]=""; for(int r=0;r<16;r++){ if(h[r*8]>64.5f){char t[24];sprintf(t,"r%d(%.0f) ",r,h[r*8]);strcat(buf,t);} }
    if((TL<8)||TL==31) printf("TL%-2d B%d -> %s\n",TL,TB,buf[0]?buf:"(no change)");
  }
  return 0;
}
