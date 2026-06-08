#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
// e2m1 codebook (magnitudes), index 0..7
static const float E2M1[8]={0,0.5f,1,1.5f,2,3,4,6};
float enc_e2m1(float x,float s,int&code){ // quantize x with scale s -> nearest e2m1*s
  float v=x/s; int sign=v<0; v=fabsf(v); int best=0; float bd=1e30f;
  for(int i=0;i<8;i++){float d=fabsf(v-E2M1[i]); if(d<bd){bd=d;best=i;}}
  code=(sign<<3)|best; return (sign?-1:1)*E2M1[best]*s;
}
// ue4m3 quantize a positive scale value -> nearest representable (4exp bias7,3mant)
float quant_ue4m3(float s){ if(s<=0)return 1e-9f; float best=1e30f,bd=1e30f;
  for(int e=0;e<16;e++)for(int m=0;m<8;m++){float v=(e==0)?(m/8.0f)*0.0078125f:(1.0f+m/8.0f)*exp2f((float)(e-7)); float d=fabsf(s-v); if(d<bd){bd=d;best=v;}} return best; }
int main(){
  srand(1); int K=2688, N=64; // a representative weight block [N rows x K]
  std::vector<float> W(N*K); double e=0;
  // realistic-ish weights: gaussian * small
  for(auto&w:W){ float u1=(rand()+1.0)/(RAND_MAX+2.0),u2=(rand()+1.0)/(RAND_MAX+2.0); w=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2)*0.02f; }
  // --- NVFP4: per-row block-16 ue4m3 scale, e2m1 codes, + per-tensor global ---
  double sse_nv=0,ss=0; float gmax=0; for(float w:W)gmax=std::max(gmax,fabsf(w));
  float global=gmax/6.0f/448.0f; // global so per-block amax/6 fits ue4m3 range (~448)
  for(int n=0;n<N;n++)for(int b=0;b<K/16;b++){
    float amax=0; for(int j=0;j<16;j++)amax=std::max(amax,fabsf(W[n*K+b*16+j]));
    float bs=quant_ue4m3((amax/6.0f)/global)*global; if(bs<=0)bs=1e-9f;
    for(int j=0;j<16;j++){int c; float r=enc_e2m1(W[n*K+b*16+j],bs,c); double d=r-W[n*K+b*16+j]; sse_nv+=d*d; ss+=(double)W[n*K+b*16+j]*W[n*K+b*16+j]; }
  }
  // --- Q4: block-32 signed int4 amax/7, f16 scale (approx as f32) ---
  double sse_q4=0; for(int n=0;n<N;n++)for(int b=0;b<K/32;b++){
    float amax=0; for(int j=0;j<32;j++)amax=std::max(amax,fabsf(W[n*K+b*32+j]));
    float s=amax/7.0f; if(s<=0)s=1e-9f;
    for(int j=0;j<32;j++){int q=(int)lroundf(W[n*K+b*32+j]/s); q=std::max(-7,std::min(7,q)); float r=q*s; double d=r-W[n*K+b*32+j]; sse_q4+=d*d; }
  }
  printf("weight [%dx%d]  ||W||^2=%.4g\n",N,K,ss);
  printf("NVFP4 (e2m1 blk16 ue4m3+global) : rel_rmse = %.4f%%\n",100*sqrt(sse_nv/ss));
  printf("Q4    (int4 blk32 amax/7 f16)   : rel_rmse = %.4f%%\n",100*sqrt(sse_q4/ss));
  printf(sse_nv<=sse_q4*1.5?"NVFP4 accuracy ~comparable-or-better to Q4 -> argmax should hold\n":"NVFP4 noticeably worse -> argmax RISK, validate carefully\n");
  return 0;
}
