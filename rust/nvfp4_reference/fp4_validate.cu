#include <cstdio>
#include <cstdlib>
// Full random m16n8k64 tile validation. A[16][64],B[8][64] random e2m1 nibbles, scales=1.
// Hypothesis map: gid=lane>>2,tig=lane&3.
//  A reg0:row gid,k=tig*16+0..7 ; reg2:row gid,k=tig*16+8..15 ; reg1:row gid+8,k=tig*16+0..7 ; reg3:row gid+8,k=tig*16+8..15
//  B reg0:row gid,k=tig*16+0..7 ; reg1:row gid,k=tig*16+8..15
__device__ __host__ float dec(int nib){
  int s=(nib>>3)&1,e=(nib>>1)&3,m=nib&1; float v;
  if(e==0) v=0.5f*m; else v=(1.0f+0.5f*m)*(1<<(e-1));
  return s? -v:v;
}
__constant__ unsigned char Ac[16*64], Bc[8*64];
__global__ void k(float* D){
  int lane=threadIdx.x,gid=lane>>2,tig=lane&3;
  unsigned a[4]={0,0,0,0}, b[2]={0,0};
  for(int j=0;j<8;j++){
    a[0]|=(unsigned)Ac[(gid)*64    + tig*16+0+j]<<(j*4);
    a[2]|=(unsigned)Ac[(gid)*64    + tig*16+8+j]<<(j*4);
    a[1]|=(unsigned)Ac[(gid+8)*64  + tig*16+0+j]<<(j*4);
    a[3]|=(unsigned)Ac[(gid+8)*64  + tig*16+8+j]<<(j*4);
    b[0]|=(unsigned)Bc[(gid)*64    + tig*16+0+j]<<(j*4);
    b[1]|=(unsigned)Bc[(gid)*64    + tig*16+8+j]<<(j*4);
  }
  unsigned sA=0x38383838,sB=0x38383838; float c0=0,c1=0,c2=0,c3=0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
   :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sA),"r"(sB));
  D[gid*8+tig*2]=c0; D[gid*8+tig*2+1]=c1; D[(gid+8)*8+tig*2]=c2; D[(gid+8)*8+tig*2+1]=c3;
}
int main(){
  unsigned char A[16*64],B[8*64]; srand(7);
  for(int i=0;i<16*64;i++)A[i]=rand()&15; for(int i=0;i<8*64;i++)B[i]=rand()&15;
  cudaMemcpyToSymbol(Ac,A,sizeof(A)); cudaMemcpyToSymbol(Bc,B,sizeof(B));
  float* d; cudaMalloc(&d,128*4); cudaMemset(d,0,128*4); k<<<1,32>>>(d);
  if(cudaDeviceSynchronize()){printf("launch err\n");return 1;}
  float h[128]; cudaMemcpy(h,d,128*4,cudaMemcpyDeviceToHost);
  int bad=0; float maxerr=0;
  for(int m=0;m<16;m++)for(int n=0;n<8;n++){
    float ref=0; for(int kk=0;kk<64;kk++) ref+=dec(A[m*64+kk])*dec(B[n*64+kk]);
    float got=h[m*8+n]; float e=ref-got; if(e<0)e=-e; if(e>maxerr)maxerr=e;
    if(e>1e-2f){bad++; if(bad<=4)printf("MISS D[%d][%d] ref=%.3f got=%.3f\n",m,n,ref,got);}
  }
  printf("bad=%d/128  maxerr=%.5f  %s\n",bad,maxerr,bad?"LAYOUT WRONG":"LAYOUT CONFIRMED");
  return 0;
}
