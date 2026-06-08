#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
static const float E2M1[8]={0,.5f,1,1.5f,2,3,4,6};
__host__ int enc(float x,float s){float v=x/s;int sg=v<0;v=fabsf(v);int b=0;float bd=1e30f;for(int i=0;i<8;i++){float d=fabsf(v-E2M1[i]);if(d<bd){bd=d;b=i;}}return (sg<<3)|b;}
__host__ float qS(float s){if(s<=0)return 1e-9f;float bv=1e-9f,bd=1e30f;for(int e=0;e<16;e++)for(int m=0;m<8;m++){float v=(e==0)?(m/8.f)*0.0078125f:(1.f+m/8.f)*exp2f((float)(e-7));float d=fabsf(s-v);if(d<bd){bd=d;bv=v;}}return bv;}
__host__ unsigned char encS(float s){if(s<=0)return 0;int be=0,bm=0;float bd=1e30f;for(int e=0;e<16;e++)for(int m=0;m<8;m++){float v=(e==0)?(m/8.f)*0.0078125f:(1.f+m/8.f)*exp2f((float)(e-7));float d=fabsf(s-v);if(d<bd){bd=d;be=e;bm=m;}}return (be<<3)|bm;}
// device: nibble codes + ue4m3 scale bytes, in ALIGNED layout per /tmp/fp4_aligned.cu
// Buffers: per ktile, A nibbles[16*64], W nibbles[8*64]; scales packed per row blk.
#define NT 4                  // K = NT*64
__constant__ unsigned char Ac[16*64*NT],Wc[8*64*NT],SAc[16*4*NT],SWc[8*4*NT];
__device__ int b4b(int blk){int t[4]={0,2,1,3};return t[blk];}
__global__ void gemm(float* C,float gA,float gW){
  int lane=threadIdx.x,gid=lane>>2,tig=lane&3; float c0=0,c1=0,c2=0,c3=0;
  for(int t=0;t<NT;t++){
    const unsigned char* A=Ac+t*16*64;const unsigned char* W=Wc+t*8*64;
    const unsigned char* SA=SAc+t*16*4;const unsigned char* SW=SWc+t*8*4;
    unsigned a[4]={0,0,0,0},b[2]={0,0},sA=0,sB=0;
    for(int K=0;K<64;K++){int blk=K>>4,w=K&15,ttig=(blk>=2?2:0)+(w>=8?1:0);if(ttig!=tig)continue;int nib=w&7,reg=blk&1;
      b[reg]|=(unsigned)W[gid*64+K]<<(nib*4);
      a[(blk&1?2:0)]|=(unsigned)A[gid*64+K]<<(nib*4);      // row gid -> regs0/2
      a[(blk&1?3:1)]|=(unsigned)A[(gid+8)*64+K]<<(nib*4);  // row gid+8 -> regs1/3
    }
    if(tig==0||tig==1){int row=gid+(tig==1?8:0);for(int blk=0;blk<4;blk++)sA|=(unsigned)SA[row*4+blk]<<(b4b(blk)*8);}
    if(tig==0){for(int blk=0;blk<4;blk++)sB|=(unsigned)SW[gid*4+blk]<<(b4b(blk)*8);}
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
     :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sA),"r"(sB));
  }
  float g=gA*gW;
  C[gid*8+tig*2]=c0*g;C[gid*8+tig*2+1]=c1*g;C[(gid+8)*8+tig*2]=c2*g;C[(gid+8)*8+tig*2+1]=c3*g;
}
int main(){
  int M=16,N=8,K=NT*64; std::vector<float> A(M*K),W(N*K);
  srand(3);for(auto&x:A){float u1=(rand()+1.)/(RAND_MAX+2.),u2=(rand()+1.)/(RAND_MAX+2.);x=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2)*0.05f;}
  for(auto&x:W){float u1=(rand()+1.)/(RAND_MAX+2.),u2=(rand()+1.)/(RAND_MAX+2.);x=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2)*0.05f;}
  // quantize A,W -> nibbles + ue4m3 block16 scales + global, ALIGNED per ktile
  float gAmax=0,gWmax=0;for(float x:A)gAmax=std::max(gAmax,fabsf(x));for(float x:W)gWmax=std::max(gWmax,fabsf(x));
  float gA=gAmax/6.f/448.f, gW=gWmax/6.f/448.f;
  std::vector<unsigned char> An(M*64*NT),Wn(N*64*NT),SAb(M*4*NT),SWb(N*4*NT);
  auto Q=[&](std::vector<float>&Wm,int rows,std::vector<unsigned char>&nib,std::vector<unsigned char>&sb,float glob){
    for(int r=0;r<rows;r++)for(int t=0;t<NT;t++)for(int blk=0;blk<4;blk++){int k0=t*64+blk*16;float amax=0;for(int j=0;j<16;j++)amax=std::max(amax,fabsf(Wm[r*K+k0+j]));
      float bs=qS((amax/6.f)/glob)*glob; if(bs<=0)bs=1e-9f; sb[t*rows*4 + r*4+blk]=encS((amax/6.f)/glob);
      for(int j=0;j<16;j++)nib[t*rows*64 + r*64+blk*16+j]=enc(Wm[r*K+k0+j],bs);}
  };
  Q(A,16,An,SAb,gA); Q(W,8,Wn,SWb,gW);
  cudaMemcpyToSymbol(Ac,An.data(),An.size());cudaMemcpyToSymbol(Wc,Wn.data(),Wn.size());
  cudaMemcpyToSymbol(SAc,SAb.data(),SAb.size());cudaMemcpyToSymbol(SWc,SWb.data(),SWb.size());
  float* d;cudaMalloc(&d,M*N*4);gemm<<<1,32>>>(d,gA,gW);
  if(cudaDeviceSynchronize()){printf("err\n");return 1;}
  std::vector<float> h(M*N);cudaMemcpy(h.data(),d,M*N*4,cudaMemcpyDeviceToHost);
  double sse=0,ss=0;for(int m=0;m<M;m++)for(int n=0;n<N;n++){double ref=0;for(int k=0;k<K;k++)ref+=(double)A[m*K+k]*W[n*K+k];double e=ref-h[m*N+n];sse+=e*e;ss+=ref*ref;}
  printf("NVFP4 GEMM [%dx%dx%d]  rel_rmse vs f32 = %.3f%%  (quant-noise expected ~10%%)\n",M,N,K,100*sqrt(sse/ss));
  printf("C[0][0..3]=%.4f %.4f %.4f %.4f\n",h[0],h[1],h[2],h[3]);
  return 0;
}
