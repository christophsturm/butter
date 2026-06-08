#include <cstdio>
#include <cstdlib>
__device__ __host__ float dec4(int nib){int s=(nib>>3)&1,e=(nib>>1)&3,m=nib&1;float v;if(e==0)v=0.5f*m;else v=(1.0f+0.5f*m)*(1<<(e-1));return s?-v:v;}
// ue4m3 decode (unsigned e4m3: 4 exp bias7, 3 mant)
__device__ __host__ float decS(unsigned char b){int e=(b>>3)&0xF,m=b&7;if(e==0)return m/8.0f*0.0078125f;return (1.0f+m/8.0f)*exp2f((float)(e-7));}
__constant__ unsigned char Ac[16*64],Bc[8*64],SAc[16*4],SBc[8*4];
__global__ void k(float* D){
  int lane=threadIdx.x,gid=lane>>2,tig=lane&3;
  unsigned a[4]={0,0,0,0},b[2]={0,0};
  for(int j=0;j<8;j++){
    a[0]|=(unsigned)Ac[gid*64+tig*16+0+j]<<(j*4);  a[2]|=(unsigned)Ac[gid*64+tig*16+8+j]<<(j*4);
    a[1]|=(unsigned)Ac[(gid+8)*64+tig*16+0+j]<<(j*4); a[3]|=(unsigned)Ac[(gid+8)*64+tig*16+8+j]<<(j*4);
    b[0]|=(unsigned)Bc[gid*64+tig*16+0+j]<<(j*4); b[1]|=(unsigned)Bc[gid*64+tig*16+8+j]<<(j*4);
  }
  // scales: tig0 -> row gid (A) / row gid (B); tig1 -> row gid+8 (A); else 1.0(0x38)
  unsigned sA=0x38383838,sB=0x38383838;
  if(tig==0){ sA=0; for(int x=0;x<4;x++) sA|=(unsigned)SAc[gid*4+x]<<(x*8); }
  else if(tig==1){ sA=0; for(int x=0;x<4;x++) sA|=(unsigned)SAc[(gid+8)*4+x]<<(x*8); }
  if(tig==0){ sB=0; for(int x=0;x<4;x++) sB|=(unsigned)SBc[gid*4+x]<<(x*8); }
  float c0=0,c1=0,c2=0,c3=0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
   :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sA),"r"(sB));
  D[gid*8+tig*2]=c0;D[gid*8+tig*2+1]=c1;D[(gid+8)*8+tig*2]=c2;D[(gid+8)*8+tig*2+1]=c3;
}
int main(){
  unsigned char A[16*64],B[8*64],SA[16*4],SB[8*4];srand(11);
  for(int i=0;i<16*64;i++)A[i]=rand()&15; for(int i=0;i<8*64;i++)B[i]=rand()&15;
  // scales: exp in [6,8], mant random -> decode 0.5..~3.75
  for(int i=0;i<16*4;i++)SA[i]=((6+(rand()%3))<<3)|(rand()&7);
  for(int i=0;i<8*4;i++)SB[i]=((6+(rand()%3))<<3)|(rand()&7);
  cudaMemcpyToSymbol(Ac,A,sizeof A);cudaMemcpyToSymbol(Bc,B,sizeof B);
  cudaMemcpyToSymbol(SAc,SA,sizeof SA);cudaMemcpyToSymbol(SBc,SB,sizeof SB);
  float* d;cudaMalloc(&d,128*4);cudaMemset(d,0,128*4);k<<<1,32>>>(d);
  if(cudaDeviceSynchronize()){printf("launch err\n");return 1;}
  float h[128];cudaMemcpy(h,d,128*4,cudaMemcpyDeviceToHost);
  int bad=0;float maxre=0;
  for(int m=0;m<16;m++)for(int n=0;n<8;n++){
    float ref=0; for(int blk=0;blk<4;blk++){float sub=0;for(int kk=0;kk<16;kk++){int K=blk*16+kk;sub+=dec4(A[m*64+K])*dec4(B[n*64+K]);} ref+=sub*decS(SA[m*4+blk])*decS(SB[n*4+blk]);}
    float got=h[m*8+n];float re=ref!=0?(ref-got)/ref:(ref-got);if(re<0)re=-re;if(re>maxre)maxre=re;
    if(re>1e-2f){bad++;if(bad<=4)printf("MISS D[%d][%d] ref=%.4f got=%.4f\n",m,n,ref,got);}
  }
  printf("bad=%d/128  max_rel_err=%.5f  %s\n",bad,maxre,bad?"SCALE/LAYOUT WRONG":"FULL LAYOUT+SCALE CONFIRMED");
  return 0;
}
