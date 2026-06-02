#include <metal_stdlib>
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
#include <metal_simdgroup_matrix>
using namespace metal;

// Plain f16 GEMM C[M,N]=A[M,K]·B[K,N] (B row-major, not transposed). 64×64 tile,
// 4 simdgroups. Two impls to measure M5 Neural-Accelerator (matmul2d) vs
// simdgroup_8x8 raw throughput — no dequant, pure MMA.

kernel void mm_sg(
    const device half *A [[buffer(0)]], const device half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nt = tgid.x*64u, mt = tgid.y*64u;
    threadgroup half As[64*32], Bs[64*32];
    const uint sg_m=(sgitg/2u)*32u, sg_n=(sgitg&1u)*32u;
    const uint ar=tiitg/2u, ak=(tiitg&1u)*16u;     // A stage 64×32
    const uint br=tiitg/2u, bk=(tiitg&1u)*16u;     // B stage 64×32 (B is [N,K]-like here: we store B^T tile)
    simdgroup_matrix<float,8,8> c00(0.f),c01(0.f),c02(0.f),c03(0.f),c10(0.f),c11(0.f),c12(0.f),c13(0.f),
                                c20(0.f),c21(0.f),c22(0.f),c23(0.f),c30(0.f),c31(0.f),c32(0.f),c33(0.f);
    simdgroup_matrix<half,8,8> a0,a1,a2,a3,b0,b1,b2,b3;
    for (uint kb=0u; kb<K; kb+=32u) {
        threadgroup half4 *as=(threadgroup half4*)(As+ar*32u+ak);
        const device half4 *ap=(const device half4*)(A+(mt+ar)*K+kb+ak); for(uint i=0;i<4;++i) as[i]=ap[i];
        threadgroup half4 *bs=(threadgroup half4*)(Bs+br*32u+bk);
        const device half4 *bp=(const device half4*)(B+(nt+br)*K+kb+bk); for(uint i=0;i<4;++i) bs[i]=bp[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint ki=0u; ki<32u; ki+=8u) {
            simdgroup_load(a0,As+(sg_m+0u)*32u+ki,32u); simdgroup_load(a1,As+(sg_m+8u)*32u+ki,32u);
            simdgroup_load(a2,As+(sg_m+16u)*32u+ki,32u); simdgroup_load(a3,As+(sg_m+24u)*32u+ki,32u);
            simdgroup_load(b0,Bs+(sg_n+0u)*32u+ki,32u,0,true); simdgroup_load(b1,Bs+(sg_n+8u)*32u+ki,32u,0,true);
            simdgroup_load(b2,Bs+(sg_n+16u)*32u+ki,32u,0,true); simdgroup_load(b3,Bs+(sg_n+24u)*32u+ki,32u,0,true);
            simdgroup_multiply_accumulate(c00,a0,b0,c00);simdgroup_multiply_accumulate(c01,a0,b1,c01);simdgroup_multiply_accumulate(c02,a0,b2,c02);simdgroup_multiply_accumulate(c03,a0,b3,c03);
            simdgroup_multiply_accumulate(c10,a1,b0,c10);simdgroup_multiply_accumulate(c11,a1,b1,c11);simdgroup_multiply_accumulate(c12,a1,b2,c12);simdgroup_multiply_accumulate(c13,a1,b3,c13);
            simdgroup_multiply_accumulate(c20,a2,b0,c20);simdgroup_multiply_accumulate(c21,a2,b1,c21);simdgroup_multiply_accumulate(c22,a2,b2,c22);simdgroup_multiply_accumulate(c23,a2,b3,c23);
            simdgroup_multiply_accumulate(c30,a3,b0,c30);simdgroup_multiply_accumulate(c31,a3,b1,c31);simdgroup_multiply_accumulate(c32,a3,b2,c32);simdgroup_multiply_accumulate(c33,a3,b3,c33);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float Ct[64*64];
    simdgroup_store(c00,Ct+(sg_m+0u)*64u+sg_n+0u,64u);simdgroup_store(c01,Ct+(sg_m+0u)*64u+sg_n+8u,64u);simdgroup_store(c02,Ct+(sg_m+0u)*64u+sg_n+16u,64u);simdgroup_store(c03,Ct+(sg_m+0u)*64u+sg_n+24u,64u);
    simdgroup_store(c10,Ct+(sg_m+8u)*64u+sg_n+0u,64u);simdgroup_store(c11,Ct+(sg_m+8u)*64u+sg_n+8u,64u);simdgroup_store(c12,Ct+(sg_m+8u)*64u+sg_n+16u,64u);simdgroup_store(c13,Ct+(sg_m+8u)*64u+sg_n+24u,64u);
    simdgroup_store(c20,Ct+(sg_m+16u)*64u+sg_n+0u,64u);simdgroup_store(c21,Ct+(sg_m+16u)*64u+sg_n+8u,64u);simdgroup_store(c22,Ct+(sg_m+16u)*64u+sg_n+16u,64u);simdgroup_store(c23,Ct+(sg_m+16u)*64u+sg_n+24u,64u);
    simdgroup_store(c30,Ct+(sg_m+24u)*64u+sg_n+0u,64u);simdgroup_store(c31,Ct+(sg_m+24u)*64u+sg_n+8u,64u);simdgroup_store(c32,Ct+(sg_m+24u)*64u+sg_n+16u,64u);simdgroup_store(c33,Ct+(sg_m+24u)*64u+sg_n+24u,64u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tiitg;i<64u*64u;i+=128u){ uint r=i/64u,cc=i%64u; if(mt+r<M&&nt+cc<N) C[(mt+r)*N+nt+cc]=(half)Ct[i]; }
}

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
// matmul2d at simdgroup scope (coop_tile config: 32×32×32, B transposed).
kernel void mm_mpp(
    const device half *A [[buffer(0)]], const device half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]], constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nt=tgid.x*64u, mt=tgid.y*64u;
    threadgroup half As[64*32], Bs[64*32];
    threadgroup float Cs[4*32*32];   // per-sg contiguous 32×32
    const uint sg_m=(sgitg/2u)*32u, sg_n=(sgitg&1u)*32u;
    const uint ar=tiitg/2u, ak=(tiitg&1u)*16u;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(32,32,32,false,true,false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca=op.get_left_input_cooperative_tensor<half,half,float>();
    auto cb=op.get_right_input_cooperative_tensor<half,half,float>();
    auto cc=op.get_destination_cooperative_tensor<decltype(ca),decltype(cb),float>();
    for (uint16_t _i=0; _i<cc.get_capacity(); ++_i) cc[_i]={};   // zero accumulator
    for (uint kb=0u; kb<K; kb+=32u) {
        threadgroup half4 *as=(threadgroup half4*)(As+ar*32u+ak);
        const device half4 *ap=(const device half4*)(A+(mt+ar)*K+kb+ak); for(uint i=0;i<4;++i) as[i]=ap[i];
        threadgroup half4 *bs=(threadgroup half4*)(Bs+ar*32u+ak);
        const device half4 *bp=(const device half4*)(B+(nt+ar)*K+kb+ak); for(uint i=0;i<4;++i) bs[i]=bp[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        metal::tensor<threadgroup half, metal::extents<int,32,32>, metal::tensor_inline> tA(As+sg_m*32u, metal::extents<int,32,32>{}); ca.load(tA);
        metal::tensor<threadgroup half, metal::extents<int,32,32>, metal::tensor_inline> tB(Bs+sg_n*32u, metal::extents<int,32,32>{}); cb.load(tB);
        op.run(ca,cb,cc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *csg = Cs + sgitg*1024u;
    metal::tensor<threadgroup float, metal::extents<int,32,32>, metal::tensor_inline> tC(csg, metal::extents<int,32,32>{}); cc.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scatter this sg's 32×32 (rows sg_m.., cols sg_n..) — 32 threads of the sg do 32 elems each
    for (uint e=tiitg%32u; e<1024u; e+=32u){ uint ii=e/32u, jj=e%32u; uint gr=mt+sg_m+ii, gc=nt+sg_n+jj; if(gr<M&&gc<N) C[gr*N+gc]=(half)csg[e]; }
}
#endif
