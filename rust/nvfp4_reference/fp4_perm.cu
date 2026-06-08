#include <cstdio>
#include <cstdlib>
__device__ __host__ float dec4(int nib){int s=(nib>>3)&1,e=(nib>>1)&3,m=nib&1;float v;if(e==0)v=0.5f*m;else v=(1.0f+0.5f*m)*(1<<(e-1));return s?-v:v;}
__device__ __host__ float decS(unsigned char b){int e=(b>>3)&0xF,m=b&7;if(e==0)return m/8.0f*0.0078125f;return (1.0f+m/8.0f)*exp2f((float)(e-7));}
__constant__ unsigned char Ac[16*64],Bc[8*64],SAc[16*4],SBc[8*4];
__constant__ int PERM[4];
__global__ void k(float* D){
  int lane=threadIdx.x,gid=lane>>2,tig=lane&3;
  unsigned a[4]={0,0,0,0},b[2]={0,0};
  for(int j=0;j<8;j++){
    a[0]|=(unsigned)Ac[gid*64+tig*16+j]<<(j*4); a[2]|=(unsigned)Ac[gid*64+tig*16+8+j]<<(j*4);
    a[1]|=(unsigned)Ac[(gid+8)*64+tig*16+j]<<(j*4); a[3]|=(unsigned)Ac[(gid+8)*64+tig*16+8+j]<<(j*4);
    b[0]|=(unsigned)Bc[gid*64+tig*16+j]<<(j*4); b[1]|=(unsigned)Bc[gid*64+tig*16+8+j]<<(j*4);
  }
  // pack scale: byte PERM[blk] gets block blk's scale
  unsigned sA=0x38383838,sB=0x38383838; int row=-1;
  if(tig==0)row=gid; else if(tig==1)row=gid+8;
  if(row>=0){sA=0;for(int blk=0;blk<4;blk++)sA|=(unsigned)SAc[row*4+blk]<<(PERM[blk]*8);}
  if(tig==0){sB=0;for(int blk=0;blk<4;blk++)sB|=(unsigned)SBc[gid*4+blk]<<(PERM[blk]*8);}
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
  float ref[128];
  for(int m=0;m<16;m++)for(int n=0;n<8;n++){float r=0;for(int blk=0;blk<4;blk++){float s=0;for(int kk=0;kk<16;kk++){int K=blk*16+kk;s+=dec4(A[m*64+K])*dec4(B[n*64+K]);}r+=s*decS(SA[m*4+blk])*decS(SB[n*4+blk]);}ref[m*8+n]=r;}
  int perms[24][4],np=0;int p[4]={0,1,2,3};
  // generate all 24 permutations
  for(int a=0;a<4;a++)for(int bb=0;bb<4;bb++)for(int c=0;c<4;c++)for(int dd=0;dd<4;dd++){if(a!=bb&&a!=c&&a!=dd&&bb!=c&&bb!=dd&&c!=dd){perms[np][0]=a;perms[np][1]=bb;perms[np][2]=c;perms[np][3]=dd;np++;}}
  float* d;cudaMalloc(&d,128*4);float h[128];
  for(int pi=0;pi<np;pi++){cudaMemcpyToSymbol(PERM,perms[pi],sizeof(int)*4);cudaMemset(d,0,128*4);k<<<1,32>>>(d);cudaDeviceSynchronize();cudaMemcpy(h,d,128*4,cudaMemcpyDeviceToHost);
    int bad=0;float mre=0;for(int i=0;i<128;i++){float re=ref[i]!=0?(ref[i]-h[i])/ref[i]:ref[i]-h[i];if(re<0)re=-re;if(re>mre)mre=re;if(re>1e-2f)bad++;}
    if(bad<8)printf("PERM[%d %d %d %d] bad=%d mre=%.4f %s\n",perms[pi][0],perms[pi][1],perms[pi][2],perms[pi][3],bad,mre,bad==0?"<<< MATCH":"");
  }
  return 0;
}
