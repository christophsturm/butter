#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
static const float E2M1[8]={0,.5f,1,1.5f,2,3,4,6};
int enc(float x,float s){float v=x/s;int sg=v<0;v=fabsf(v);int b=0;float bd=1e30f;for(int i=0;i<8;i++){float d=fabsf(v-E2M1[i]);if(d<bd){bd=d;b=i;}}return (sg<<3)|b;}
unsigned char encS(float s){if(s<=0)return 0;int be=0,bm=0;float bd=1e30f;for(int e=0;e<16;e++)for(int m=0;m<8;m++){float v=(e==0)?(m/8.f)*0.0078125f:(1.f+m/8.f)*exp2f((float)(e-7));float d=fabsf(s-v);if(d<bd){bd=d;be=e;bm=m;}}return (be<<3)|bm;}
// quantize_nvfp4 layout: codes[row*(K/8)+elem/8] nibble elem%8; scales[row*(K/16)+blk] ue4m3; global
void qnv(const std::vector<float>&W,int rows,int K,std::vector<unsigned>&codes,std::vector<unsigned char>&scal,float glob){
  codes.assign((size_t)rows*(K/8),0); scal.assign((size_t)rows*(K/16),0);
  for(int r=0;r<rows;r++)for(int b=0;b<K/16;b++){int base=r*K+b*16;float amax=0;for(int j=0;j<16;j++)amax=std::max(amax,fabsf(W[base+j]));
    unsigned char sb=encS((amax/6.f)/glob); scal[r*(K/16)+b]=sb;
    int e=(sb>>3)&0xF,m=sb&7; float bs=((e==0)?(m/8.f)*0.0078125f:(1.f+m/8.f)*exp2f((float)(e-7)))*glob; if(bs<=0)bs=1e-9f;
    for(int j=0;j<16;j++){int elem=b*16+j;int code=enc(W[base+j],bs);codes[(size_t)r*(K/8)+elem/8]|=(unsigned)code<<((elem%8)*4);}}
}
__device__ int b4b(int blk){int t[4]={0,2,1,3};return t[blk];}
// In-kernel gather from packed quantize_nvfp4 layout. One warp -> C[16x8] over K.
__global__ void gemm(const unsigned* Acode,const unsigned char* Asc,const unsigned* Wcode,const unsigned char* Wsc,
                     float* C,int K,float g){
  int lane=threadIdx.x,gid=lane>>2,tig=lane&3; int KT=K/64; float c0=0,c1=0,c2=0,c3=0;
  for(int t=0;t<KT;t++){
    unsigned a[4]={0,0,0,0},b[2]={0,0},sA=0,sB=0;
    for(int kk=0;kk<64;kk++){int blk=kk>>4,w=kk&15,ttig=(blk>=2?2:0)+(w>=8?1:0);if(ttig!=tig)continue;int nib=w&7;int gK=t*64+kk;
      // W row gid
      {int e=gK,word=e>>3,sh=(e&7)*4;unsigned code=(Wcode[(size_t)gid*(K/8)+word]>>sh)&0xf; b[blk&1]|=code<<(nib*4);}
      // A rows gid (regs0/2) and gid+8 (regs1/3)
      {int e=gK,word=e>>3,sh=(e&7)*4;
        unsigned ca=(Acode[(size_t)gid*(K/8)+word]>>sh)&0xf; a[(blk&1?2:0)]|=ca<<(nib*4);
        unsigned cb=(Acode[(size_t)(gid+8)*(K/8)+word]>>sh)&0xf; a[(blk&1?3:1)]|=cb<<(nib*4);}
    }
    // scales: 4 blocks of this tile = global blocks t*4..t*4+3
    if(tig==0||tig==1){int row=gid+(tig==1?8:0);for(int blk=0;blk<4;blk++)sA|=(unsigned)Asc[(size_t)row*(K/16)+t*4+blk]<<(b4b(blk)*8);}
    if(tig==0){for(int blk=0;blk<4;blk++)sB|=(unsigned)Wsc[(size_t)gid*(K/16)+t*4+blk]<<(b4b(blk)*8);}
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
     :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sA),"r"(sB));
  }
  C[gid*8+tig*2]=c0*g;C[gid*8+tig*2+1]=c1*g;C[(gid+8)*8+tig*2]=c2*g;C[(gid+8)*8+tig*2+1]=c3*g;
}
int main(){
  int K=2688; std::vector<float> A(16*K),W(8*K);srand(5);
  for(auto&x:A){float u1=(rand()+1.)/(RAND_MAX+2.),u2=(rand()+1.)/(RAND_MAX+2.);x=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2)*0.05f;}
  for(auto&x:W){float u1=(rand()+1.)/(RAND_MAX+2.),u2=(rand()+1.)/(RAND_MAX+2.);x=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2)*0.05f;}
  float gam=0,gwm=0;for(float x:A)gam=std::max(gam,fabsf(x));for(float x:W)gwm=std::max(gwm,fabsf(x));
  float gA=gam/6.f/448.f,gW=gwm/6.f/448.f;
  std::vector<unsigned> Ac,Wc;std::vector<unsigned char> As,Ws; qnv(A,16,K,Ac,As,gA); qnv(W,8,K,Wc,Ws,gW);
  unsigned *dAc,*dWc;unsigned char *dAs,*dWs;float* dC;
  cudaMalloc(&dAc,Ac.size()*4);cudaMalloc(&dWc,Wc.size()*4);cudaMalloc(&dAs,As.size());cudaMalloc(&dWs,Ws.size());cudaMalloc(&dC,16*8*4);
  cudaMemcpy(dAc,Ac.data(),Ac.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(dWc,Wc.data(),Wc.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dAs,As.data(),As.size(),cudaMemcpyHostToDevice);cudaMemcpy(dWs,Ws.data(),Ws.size(),cudaMemcpyHostToDevice);
  gemm<<<1,32>>>(dAc,dAs,dWc,dWs,dC,K,gA*gW);
  if(cudaDeviceSynchronize()){printf("err\n");return 1;}
  float h[128];cudaMemcpy(h,dC,16*8*4,cudaMemcpyDeviceToHost);
  double sse=0,ss=0;for(int m=0;m<16;m++)for(int n=0;n<8;n++){double ref=0;for(int k=0;k<K;k++)ref+=(double)A[m*K+k]*W[n*K+k];double e=ref-h[m*8+n];sse+=e*e;ss+=ref*ref;}
  printf("in-kernel packed gather GEMM [16x8x%d]: rel_rmse=%.2f%%  %s\n",K,100*sqrt(sse/ss),sqrt(sse/ss)<0.2?"<<< CORRECT (matches quantize_nvfp4 layout)":"WRONG");
  return 0;
}
