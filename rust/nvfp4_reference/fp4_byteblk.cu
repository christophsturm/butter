#include <cstdio>
// A=1.0 everywhere; B=1.0 ONLY in data k-block bd (k in [bd*16,bd*16+16)), else 0.
// sB=1; sA: lane(gid0,tig0) byte TB=2.0, rest 1.0. D[0][0]=16*sA[block of bd]. If byte TB scales block bd -> 32 else 16.
__global__ void k(float* D,int bd,int TB){
  int lane=threadIdx.x,gid=lane>>2,tig=lane&3;
  unsigned a0=0x22222222,a1=a0,a2=a0,a3=a0;       // A all 1.0
  // B = 1.0 only where data k-block==bd. lane tig holds k=tig*16..+15 = block tig. regs b0(k tig*16+0..7),b1(+8..15)
  unsigned b0=(tig==bd)?0x22222222:0, b1=(tig==bd)?0x22222222:0;
  unsigned sA=0x38383838,sB=0x38383838;
  if(tig==0 && gid==0) sA=(sA&~(0xFFu<<(TB*8)))|(0x40u<<(TB*8));
  float c0=0,c1=0,c2=0,c3=0;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
   :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sA),"r"(sB));
  D[gid*8+tig*2]=c0;D[gid*8+tig*2+1]=c1;D[(gid+8)*8+tig*2]=c2;D[(gid+8)*8+tig*2+1]=c3;
}
int main(){
  float* d;cudaMalloc(&d,128*4);float h[128];
  printf("      bd0   bd1   bd2   bd3   (D[0][0]; 32=byte scales that block)\n");
  for(int TB=0;TB<4;TB++){ printf("byte%d ",TB);
    for(int bd=0;bd<4;bd++){cudaMemset(d,0,128*4);k<<<1,32>>>(d,bd,TB);cudaDeviceSynchronize();cudaMemcpy(h,d,128*4,cudaMemcpyDeviceToHost);printf("%5.0f ",h[0]);}
    printf("\n");
  }
  return 0;
}
