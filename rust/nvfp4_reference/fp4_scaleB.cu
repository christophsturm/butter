#include <cstdio>
#include <cstring>
// Derive sB map: A=B=1, sA=1. one-hot sB byte TB of thread TL =2.0. D[m][n]=sum_blk 16*sB[n][blk]. find col n that jumps.
__global__ void k(float* D,int TL,int TB){
  unsigned a0=0x22222222,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,sA=0x38383838,sB=0x38383838;
  if((int)threadIdx.x==TL) sB=(sB&~(0xFFu<<(TB*8)))|(0x40u<<(TB*8));
  float c0=0,c1=0,c2=0,c3=0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
   :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sA),"r"(sB));
  int g=threadIdx.x>>2,t=threadIdx.x&3;
  D[g*8+t*2]=c0;D[g*8+t*2+1]=c1;D[(g+8)*8+t*2]=c2;D[(g+8)*8+t*2+1]=c3;
}
int main(){
  float* d;cudaMalloc(&d,128*4);float h[128];
  printf("sB: (TL,TB) -> cols that jumped\n");
  for(int TL=0;TL<32;TL++)for(int TB=0;TB<4;TB++){
    cudaMemset(d,0,128*4);k<<<1,32>>>(d,TL,TB);cudaDeviceSynchronize();cudaMemcpy(h,d,128*4,cudaMemcpyDeviceToHost);
    char buf[64]="";for(int c=0;c<8;c++)if(h[c]>64.5f){char t[8];sprintf(t,"c%d ",c);strcat(buf,t);}
    if(TL<12||TL==31)printf("TL%-2d B%d -> %s\n",TL,TB,buf[0]?buf:"-");
  }
  return 0;
}
