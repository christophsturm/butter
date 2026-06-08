#include <cuda_fp4.h>
#include <mma.h>
// Probe: does sm_121a assemble a block-scaled FP4 (e2m1 x mxfp4) mma.sync?
__global__ void probe(const unsigned* a,const unsigned* b,float* d,const unsigned* sfa,const unsigned* sfb){
  unsigned ra[4]={a[0],a[1],a[2],a[3]};
  unsigned rb[2]={b[0],b[1]};
  float rc[4]={0,0,0,0};
  // Blackwell block-scaled MXF4 mma (sm_120a/121a): m16n8k64, e2m1 x e2m1, ue8m0 scales
  asm volatile(
   "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
   : "+f"(rc[0]),"+f"(rc[1]),"+f"(rc[2]),"+f"(rc[3])
   : "r"(ra[0]),"r"(ra[1]),"r"(ra[2]),"r"(ra[3]),"r"(rb[0]),"r"(rb[1]),"r"(sfa[0]),"r"(sfb[0]));
  d[threadIdx.x]=rc[0];
}
int main(){return 0;}
