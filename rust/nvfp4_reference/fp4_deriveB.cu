#include <cstdio>
#include <cstring>
// Derive B fragment row-map: one-hot B at (tl,tr,tn)=e2m1 1.0, A=all-ones, scales=1.
// D[m][n]=sum_k A[m][k]*B[n][k]=sum_k B[n][k] = 1.0 only at B's row n (all m). Read which D col lights.
__global__ void k(float* D,int tl,int tr,int tn){
  unsigned a0=0x22222222,a1=0x22222222,a2=0x22222222,a3=0x22222222, b[2]={0,0}, sA=0x38383838,sB=0x38383838;
  if((int)threadIdx.x==tl) b[tr] |= (0x2u<<(tn*4));
  float c0=0,c1=0,c2=0,c3=0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
   :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b[0]),"r"(b[1]),"r"(sA),"r"(sB));
  int g=threadIdx.x>>2,t=threadIdx.x&3;
  D[g*8+t*2]=c0; D[g*8+t*2+1]=c1; D[(g+8)*8+t*2]=c2; D[(g+8)*8+t*2+1]=c3;
}
int main(){
  float* d; cudaMalloc(&d,128*4); float h[128];
  printf("B: lane reg -> cols_with_nonzero_D\n");
  for(int tl=0;tl<32;tl++) for(int tr=0;tr<2;tr++){
    cudaMemset(d,0,128*4); k<<<1,32>>>(d,tl,tr,0); cudaDeviceSynchronize();
    cudaMemcpy(h,d,128*4,cudaMemcpyDeviceToHost);
    char buf[64]=""; for(int c=0;c<8;c++){int nz=0; for(int r=0;r<16;r++) if(h[r*8+c]!=0)nz++; if(nz){char t[8];sprintf(t,"c%d ",c);strcat(buf,t);}}
    if(tl<8||tl==31) printf("L%-2d reg%d -> %s\n",tl,tr,buf[0]?buf:"(none)");
  }
  return 0;
}
