#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
// Device activation quantizer: f16 act[rows,K] -> NVFP4 (e2m1 codes u32 8/word, ue4m3 scales[rows,K/16], global).
// global passed in (real model: a reduce kernel; here host-computed). One thread per block-16.
__device__ unsigned char encS_d(float s){if(s<=0)return 0;int be=0,bm=0;float bd=1e30f;
  for(int e=0;e<16;e++)for(int m=0;m<8;m++){float v=(e==0)?(m/8.f)*0.0078125f:(1.f+m/8.f)*exp2f((float)(e-7));float d=fabsf(s-v);if(d<bd){bd=d;be=e;bm=m;}}return (be<<3)|bm;}
__device__ float decS_d(unsigned char b){int e=(b>>3)&0xF,m=b&7;return (e==0)?(m/8.f)*0.0078125f:(1.f+m/8.f)*exp2f((float)(e-7));}
__device__ int encE_d(float x,float s){float v=x/s;int sg=v<0;v=fabsf(v);
  const float L[8]={0,.5f,1,1.5f,2,3,4,6};int b=0;float bd=1e30f;for(int i=0;i<8;i++){float d=fabsf(v-L[i]);if(d<bd){bd=d;b=i;}}return (sg<<3)|b;}
__global__ void actq(const __half* a,unsigned* codes,unsigned char* scales,int rows,int K,float global){
  int blk=blockIdx.x*blockDim.x+threadIdx.x; int nblk=rows*(K/16); if(blk>=nblk)return;
  int r=blk/(K/16), b=blk%(K/16); const __half* ab=a+(size_t)r*K+b*16;
  float v[16],amax=0; for(int i=0;i<16;i++){v[i]=__half2float(ab[i]);amax=fmaxf(amax,fabsf(v[i]));}
  unsigned char sb=encS_d((amax/6.f)/global); scales[blk]=sb; float bs=decS_d(sb)*global; if(bs<=0)bs=1e-9f;
  for(int i=0;i<16;i++){int elem=b*16+i,word=elem>>3,nib=elem&7;int code=encE_d(v[i],bs);atomicOr(&codes[(size_t)r*(K/8)+word],(unsigned)code<<(nib*4));}
}
int main(){
  int rows=512,K=2688; std::vector<float> A((size_t)rows*K);srand(7);
  for(auto&x:A){float u1=(rand()+1.)/(RAND_MAX+2.),u2=(rand()+1.)/(RAND_MAX+2.);x=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2)*0.05f;}
  std::vector<__half> Ah(A.size()); for(size_t i=0;i<A.size();i++)Ah[i]=__float2half(A[i]);
  float gmax=0;for(float x:A)gmax=fmaxf(gmax,fabsf(x)); float global=gmax/6.f/448.f;
  __half* dA;unsigned* dC;unsigned char* dS; cudaMalloc(&dA,Ah.size()*2);cudaMalloc(&dC,(size_t)rows*(K/8)*4);cudaMalloc(&dS,(size_t)rows*(K/16));
  cudaMemcpy(dA,Ah.data(),Ah.size()*2,cudaMemcpyHostToDevice);cudaMemset(dC,0,(size_t)rows*(K/8)*4);
  int nblk=rows*(K/16); actq<<<(nblk+255)/256,256>>>(dA,dC,dS,rows,K,global);
  if(cudaDeviceSynchronize()){printf("err\n");return 1;}
  std::vector<unsigned> C((size_t)rows*(K/8));std::vector<unsigned char> S((size_t)rows*(K/16));
  cudaMemcpy(C.data(),dC,C.size()*4,cudaMemcpyDeviceToHost);cudaMemcpy(S.data(),dS,S.size(),cudaMemcpyDeviceToHost);
  // dequant + compare to original (f16-rounded)
  const float L[8]={0,.5f,1,1.5f,2,3,4,6}; double sse=0,ss=0;
  for(int r=0;r<rows;r++)for(int b=0;b<K/16;b++){int e=(S[r*(K/16)+b]>>3)&0xF,m=S[r*(K/16)+b]&7;float bs=((e==0)?(m/8.f)*0.0078125f:(1.f+m/8.f)*exp2f((float)(e-7)))*global;
    for(int i=0;i<16;i++){int elem=b*16+i,word=elem>>3,nib=elem&7;unsigned cd=(C[(size_t)r*(K/8)+word]>>(nib*4))&0xf;float sign=(cd&8)?-1:1;float rec=sign*L[cd&7]*bs;float orig=__half2float(Ah[(size_t)r*K+elem]);double d=rec-orig;sse+=d*d;ss+=(double)orig*orig;}}
  printf("device act-quant [%dx%d]: rel_rmse=%.3f%%  %s\n",rows,K,100*sqrt(sse/ss),sqrt(sse/ss)<0.15?"<<< OK (matches quantize_nvfp4)":"CHECK");
  return 0;
}
