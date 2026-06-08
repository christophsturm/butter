#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
static const float E2M1[8]={0,.5f,1,1.5f,2,3,4,6};
int enc(float x,float s){float v=x/s;int sg=v<0;v=fabsf(v);int b=0;float bd=1e30f;for(int i=0;i<8;i++){float d=fabsf(v-E2M1[i]);if(d<bd){bd=d;b=i;}}return (sg<<3)|b;}
float qS(float s){if(s<=0)return 1e-9f;float bv=1e-9f,bd=1e30f;for(int e=0;e<16;e++)for(int m=0;m<8;m++){float v=(e==0)?(m/8.f)*0.0078125f:(1.f+m/8.f)*exp2f((float)(e-7));float d=fabsf(s-v);if(d<bd){bd=d;bv=v;}}return bv;}
unsigned char encS(float s){if(s<=0)return 0;int be=0,bm=0;float bd=1e30f;for(int e=0;e<16;e++)for(int m=0;m<8;m++){float v=(e==0)?(m/8.f)*0.0078125f:(1.f+m/8.f)*exp2f((float)(e-7));float d=fabsf(s-v);if(d<bd){bd=d;be=e;bm=m;}}return (be<<3)|bm;}
__device__ int b4b(int blk){int t[4]={0,2,1,3};return t[blk];}
// MT=M/16 tiles, NT=N/8, KT=K/64. Apack[mt][kt][lane][4], Wpack[nt][kt][lane][2], Asc[mt][kt][lane], Wsc[nt][kt][gid]
__global__ void gemm(const unsigned* Ap,const unsigned* Wp,const unsigned* Asc,const unsigned* Wsc,
                     float* C,int MT,int NT,int KT,int N,float g){
  int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int lane=threadIdx.x&31;
  if(warp>=MT*NT)return; int mt=warp/NT, nt=warp%NT; int gid=lane>>2,tig=lane&3;
  float c0=0,c1=0,c2=0,c3=0;
  for(int kt=0;kt<KT;kt++){
    const unsigned* A=Ap+((size_t)(mt*KT+kt)*32+lane)*4;
    const unsigned* W=Wp+((size_t)(nt*KT+kt)*32+lane)*2;
    unsigned a0=A[0],a1=A[1],a2=A[2],a3=A[3],b0=W[0],b1=W[1];
    unsigned sA=Asc[(size_t)(mt*KT+kt)*32+lane], sB=Wsc[(size_t)(nt*KT+kt)*8+gid];
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
     :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sA),"r"(sB));
  }
  int r0=mt*16,col=nt*8+tig*2;
  C[(r0+gid)*N+col]=c0*g;C[(r0+gid)*N+col+1]=c1*g;C[(r0+gid+8)*N+col]=c2*g;C[(r0+gid+8)*N+col+1]=c3*g;
}
int main(){
  int M=96,N=1856,K=2688,MT=M/16,NT=N/8,KT=K/64;
  std::vector<float> A((size_t)M*K),W((size_t)N*K);srand(3);
  for(auto&x:A){float u1=(rand()+1.)/(RAND_MAX+2.),u2=(rand()+1.)/(RAND_MAX+2.);x=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2)*0.05f;}
  for(auto&x:W){float u1=(rand()+1.)/(RAND_MAX+2.),u2=(rand()+1.)/(RAND_MAX+2.);x=sqrtf(-2*logf(u1))*cosf(6.2831853f*u2)*0.05f;}
  float gam=0,gwm=0;for(float x:A)gam=std::max(gam,fabsf(x));for(float x:W)gwm=std::max(gwm,fabsf(x));
  float gA=gam/6.f/448.f,gW=gwm/6.f/448.f;
  std::vector<unsigned> Ap((size_t)MT*KT*32*4,0),Wp((size_t)NT*KT*32*2,0),Asc((size_t)MT*KT*32,0),Wsc((size_t)NT*KT*8,0);
  // pack A
  for(int mt=0;mt<MT;mt++)for(int kt=0;kt<KT;kt++)for(int lane=0;lane<32;lane++){int gid=lane>>2,tig=lane&3;unsigned a[4]={0,0,0,0},sA=0;
    for(int half=0;half<2;half++){int row=mt*16+gid+half*8;for(int Kk=0;Kk<64;Kk++){int blk=Kk>>4,w=Kk&15,tt=(blk>=2?2:0)+(w>=8?1:0);if(tt!=tig)continue;int nib=w&7;int gK=kt*64+Kk;
      // block scale (block16) for this element
      int k0=gK-(w);/*block start*/ int bstart=(gK/16)*16; float amax=0;for(int j=0;j<16;j++)amax=std::max(amax,fabsf(A[(size_t)row*K+bstart+j]));
      float bs=qS((amax/6.f)/gA)*gA;if(bs<=0)bs=1e-9f; a[(blk&1?(half?3:2):(half?1:0))]|=(unsigned)enc(A[(size_t)row*K+gK],bs)<<(nib*4);}}
    if(tig==0||tig==1){int row=mt*16+gid+(tig==1?8:0);for(int blk=0;blk<4;blk++){int bstart=(kt*64+blk*16);float amax=0;for(int j=0;j<16;j++)amax=std::max(amax,fabsf(A[(size_t)row*K+bstart+j]));sA|=(unsigned)encS((amax/6.f)/gA)<<(b4b(blk)*8);}}
    Ap[((size_t)(mt*KT+kt)*32+lane)*4+0]=a[0];Ap[((size_t)(mt*KT+kt)*32+lane)*4+1]=a[1];Ap[((size_t)(mt*KT+kt)*32+lane)*4+2]=a[2];Ap[((size_t)(mt*KT+kt)*32+lane)*4+3]=a[3];Asc[(size_t)(mt*KT+kt)*32+lane]=sA;}
  // pack W
  for(int nt=0;nt<NT;nt++)for(int kt=0;kt<KT;kt++)for(int lane=0;lane<32;lane++){int gid=lane>>2,tig=lane&3;unsigned b[2]={0,0};int row=nt*8+gid;
    for(int Kk=0;Kk<64;Kk++){int blk=Kk>>4,w=Kk&15,tt=(blk>=2?2:0)+(w>=8?1:0);if(tt!=tig)continue;int nib=w&7,reg=blk&1;int gK=kt*64+Kk;int bstart=(gK/16)*16;float amax=0;for(int j=0;j<16;j++)amax=std::max(amax,fabsf(W[(size_t)row*K+bstart+j]));float bs=qS((amax/6.f)/gW)*gW;if(bs<=0)bs=1e-9f;b[reg]|=(unsigned)enc(W[(size_t)row*K+gK],bs)<<(nib*4);}
    Wp[((size_t)(nt*KT+kt)*32+lane)*2+0]=b[0];Wp[((size_t)(nt*KT+kt)*32+lane)*2+1]=b[1];
    if(tig==0){unsigned sW=0;for(int blk=0;blk<4;blk++){int bstart=(kt*64+blk*16);float amax=0;for(int j=0;j<16;j++)amax=std::max(amax,fabsf(W[(size_t)row*K+bstart+j]));sW|=(unsigned)encS((amax/6.f)/gW)<<(b4b(blk)*8);}Wsc[(size_t)(nt*KT+kt)*8+gid]=sW;}}
  unsigned *dAp,*dWp,*dAsc,*dWsc;float* dC;
  cudaMalloc(&dAp,Ap.size()*4);cudaMalloc(&dWp,Wp.size()*4);cudaMalloc(&dAsc,Asc.size()*4);cudaMalloc(&dWsc,Wsc.size()*4);cudaMalloc(&dC,(size_t)M*N*4);
  cudaMemcpy(dAp,Ap.data(),Ap.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(dWp,Wp.data(),Wp.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dAsc,Asc.data(),Asc.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(dWsc,Wsc.data(),Wsc.size()*4,cudaMemcpyHostToDevice);
  int warps=MT*NT,tpb=256,blocks=(warps*32+tpb-1)/tpb;
  float g=gA*gW;
  gemm<<<blocks,tpb>>>(dAp,dWp,dAsc,dWsc,dC,MT,NT,KT,N,g);cudaDeviceSynchronize();
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);int rep=50;
  cudaEventRecord(s);for(int i=0;i<rep;i++)gemm<<<blocks,tpb>>>(dAp,dWp,dAsc,dWsc,dC,MT,NT,KT,N,g);cudaEventRecord(e);cudaEventSynchronize(e);
  float ms=0;cudaEventElapsedTime(&ms,s,e);ms/=rep;
  std::vector<float> h((size_t)M*N);cudaMemcpy(h.data(),dC,(size_t)M*N*4,cudaMemcpyDeviceToHost);
  double sse=0,ss=0;for(int m=0;m<M;m++)for(int n=0;n<8;n++){double ref=0;for(int k=0;k<K;k++)ref+=(double)A[(size_t)m*K+k]*W[(size_t)n*K+k];double er=ref-h[(size_t)m*N+n];sse+=er*er;ss+=ref*ref;}
  double flop=2.0*M*N*K;
  printf("NVFP4 tiled GEMM [%dx%dx%d]: %.3f ms = %.1f TFLOP/s  rel_rmse(first8col)=%.2f%%\n",M,N,K,ms,flop/(ms*1e-3)/1e12,100*sqrt(sse/ss));
  return 0;
}
