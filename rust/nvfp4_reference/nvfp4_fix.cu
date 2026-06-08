#include <cuda_fp4.h>
__global__ void probe(const unsigned* a,const unsigned* b,float* d,const unsigned* sa,const unsigned* sb){
  unsigned ra[4]={a[0],a[1],a[2],a[3]}; unsigned rb[2]={b[0],b[1]};
  float rc[4]={0,0,0,0};
  asm volatile(
   "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
   : "+f"(rc[0]),"+f"(rc[1]),"+f"(rc[2]),"+f"(rc[3])
   : "r"(ra[0]),"r"(ra[1]),"r"(ra[2]),"r"(ra[3]),"r"(rb[0]),"r"(rb[1]),"r"(sa[0]),"r"(sb[0]));
  d[threadIdx.x]=rc[0];
}
int main(){return 0;}
