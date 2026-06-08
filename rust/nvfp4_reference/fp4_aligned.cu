#include <cstdio>
#include <cstdlib>
__device__ __host__ float dec4(int nib){int s=(nib>>3)&1,e=(nib>>1)&3,m=nib&1;float v;if(e==0)v=0.5f*m;else v=(1.0f+0.5f*m)*(1<<(e-1));return s?-v:v;}
__device__ __host__ float decS(unsigned char b){int e=(b>>3)&0xF,m=b&7;if(e==0)return m/8.0f*0.0078125f;return (1.0f+m/8.0f)*exp2f((float)(e-7));}
__constant__ unsigned char Ac[16*64],Bc[8*64],SAc[16*4],SBc[8*4];
__device__ int byteForBlock(int blk){int t[4]={0,2,1,3};return t[blk];}
__global__ void kr(float* D){
  int lane=threadIdx.x,gid=lane>>2,tig=lane&3;
  unsigned a[4]={0,0,0,0},b[2]={0,0},sA=0x38383838,sB=0x38383838;
  // ALIGNED load: for each weight k, compute its fragment slot from (block,w)
  // B row = gid (this lane). place B[gid][K]
  for(int K=0;K<64;K++){int blk=K>>4,w=K&15; int ttig=(blk>=2?2:0)+(w>=8?1:0); if(ttig!=tig)continue; int reg=blk&1; int nib=w&7; b[reg]|=(unsigned)Bc[gid*64+K]<<(nib*4);}
  // A rows gid (regs0/2) and gid+8 (regs1/3)
  for(int half=0;half<2;half++){int row=gid+half*8;
    for(int K=0;K<64;K++){int blk=K>>4,w=K&15;int ttig=(blk>=2?2:0)+(w>=8?1:0);if(ttig!=tig)continue;int nib=w&7;
      int reg = half==0 ? (blk&1?2:0) : (blk&1?3:1);
      a[reg]|=(unsigned)Ac[row*64+K]<<(nib*4);}}
  // scales: A scale lane tig0->row gid, tig1->row gid+8; byte=byteForBlock(blk)
  if(tig==0||tig==1){int row=gid+(tig==1?8:0);sA=0;for(int blk=0;blk<4;blk++)sA|=(unsigned)SAc[row*4+blk]<<(byteForBlock(blk)*8);}
  if(tig==0){sB=0;for(int blk=0;blk<4;blk++)sB|=(unsigned)SBc[gid*4+blk]<<(byteForBlock(blk)*8);}
  float c0=0,c1=0,c2=0,c3=0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
   :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sA),"r"(sB));
  D[gid*8+tig*2]=c0;D[gid*8+tig*2+1]=c1;D[(gid+8)*8+tig*2]=c2;D[(gid+8)*8+tig*2+1]=c3;
}
int main(){
  unsigned char A[16*64],B[8*64],SA[16*4],SB[8*4];srand(11);
  for(int i=0;i<16*64;i++)A[i]=rand()&15;for(int i=0;i<8*64;i++)B[i]=rand()&15;
  for(int i=0;i<16*4;i++)SA[i]=((6+(rand()%3))<<3)|(rand()&7);for(int i=0;i<8*4;i++)SB[i]=((6+(rand()%3))<<3)|(rand()&7);
  cudaMemcpyToSymbol(Ac,A,sizeof A);cudaMemcpyToSymbol(Bc,B,sizeof B);cudaMemcpyToSymbol(SAc,SA,sizeof SA);cudaMemcpyToSymbol(SBc,SB,sizeof SB);
  float* d;cudaMalloc(&d,128*4);cudaMemset(d,0,128*4);kr<<<1,32>>>(d);
  if(cudaDeviceSynchronize()){printf("err\n");return 1;}
  float h[128];cudaMemcpy(h,d,128*4,cudaMemcpyDeviceToHost);
  int bad=0;float mre=0;
  for(int m=0;m<16;m++)for(int n=0;n<8;n++){float ref=0;for(int blk=0;blk<4;blk++){float s=0;for(int kk=0;kk<16;kk++){int K=blk*16+kk;s+=dec4(A[m*64+K])*dec4(B[n*64+K]);}ref+=s*decS(SA[m*4+blk])*decS(SB[n*4+blk]);}float re=ref!=0?(ref-h[m*8+n])/ref:ref-h[m*8+n];if(re<0)re=-re;if(re>mre)mre=re;if(re>1e-2f){bad++;if(bad<=3)printf("MISS[%d][%d] ref=%.3f got=%.3f\n",m,n,ref,h[m*8+n]);}}
  printf("bad=%d/128 mre=%.5f %s\n",bad,mre,bad?"WRONG":"<<<< NVFP4 mma FULLY SOLVED (data+scale)");
  return 0;
}
