#include <metal_stdlib>
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

// Production MoE IQ2_XXS GEMM on M5 Neural Accelerators (matmul2d) + ragged
// multi-expert run-detection. grid/signs preloaded to threadgroup; per
// same-expert run: dequant W→Ws, matmul2d (4 sg × 32×32×32, ~3× simdgroup), store
// the run's rows. out[m,n]=Σ_k x[m,k]·W_e[n,k], e=indices[m]. FFAI bm64 bindings.

kernel void ffai_moe_bgemm_iq2xxs_view_u16_bm64_f16(
    const device half   *x          [[buffer(0)]],
    const device ushort *view_u16   [[buffer(1)]],
    const device half   *view_f16   [[buffer(2)]],
    const device uchar  *grid       [[buffer(3)]],
    const device uchar  *signs      [[buffer(4)]],
    const device uint   *indices    [[buffer(5)]],
    device half         *out        [[buffer(6)]],
    constant uint &m_total          [[buffer(7)]],
    constant uint &n_out            [[buffer(8)]],
    constant uint &k_in             [[buffer(9)]],
    constant uint &tensor_byte_off  [[buffer(10)]],
    constant uint &expert_byte_stride [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
    const uint n_tile = tgid.x*64u, m_tile = tgid.y*64u;
    threadgroup half  Xs[64*32];
    threadgroup half  Ws[64*32];
    threadgroup uchar Gs[256*8];
    threadgroup uchar Ss[128];
    threadgroup float Cs[4*32*32];
    for (uint i=tiitg;i<2048u;i+=128u) Gs[i]=grid[i];
    for (uint i=tiitg;i<128u;i+=128u) Ss[i]=signs[i];
    const uint sg_m=(sgitg/2u)*32u, sg_n=(sgitg&1u)*32u;
    const uint x_row=tiitg/2u, x_k0=(tiitg&1u)*16u;
    const uint w_flat=tiitg*16u, w_row=w_flat/32u, w_k0=w_flat&31u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub_offset=0u;
    while (sub_offset<64u) {
        uint cur_row=m_tile+sub_offset;
        if (cur_row>=m_total) break;
        uint expert=indices[cur_row];
        uint sub_end=64u;
        for (uint p=sub_offset+1u;p<64u;++p){ uint pr=m_tile+p; if(pr>=m_total||indices[pr]!=expert){sub_end=p;break;} }
        const uint eu16=(tensor_byte_off+expert*expert_byte_stride)/2u;

        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(32,32,32,false,true,false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        auto ca=op.get_left_input_cooperative_tensor<half,half,float>();
        auto cb=op.get_right_input_cooperative_tensor<half,half,float>();
        auto cc=op.get_destination_cooperative_tensor<decltype(ca),decltype(cb),float>();
        for (uint16_t _i=0;_i<cc.get_capacity();++_i) cc[_i]={};

        for (uint kb=0u; kb<k_in; kb+=32u) {
            {   uint gr=m_tile+x_row; bool inr=(x_row>=sub_offset)&&(x_row<sub_end)&&(gr<m_total);
                threadgroup half4 *xs=(threadgroup half4*)(Xs+x_row*32u+x_k0);
                if (inr){ const device half4 *xp=(const device half4*)(x+gr*k_in+kb+x_k0); for(uint i=0;i<4;++i) xs[i]=xp[i]; }
                else { for(uint i=0;i<4;++i) xs[i]=half4(0); }
            }
            {   uint vidx0=(n_tile+w_row)*k_in+kb+w_k0;
                uint block=vidx0/256u, group=(vidx0&255u)/32u;
                uint b16=eu16+block*33u;
                float d=(float)view_f16[b16];
                uint qd=b16+1u+group*4u;
                uint aux_idx=(uint)view_u16[qd]|((uint)view_u16[qd+1u]<<16);
                uint aux_sgn=(uint)view_u16[qd+2u]|((uint)view_u16[qd+3u]<<16);
                float db=d*(((float)(aux_sgn>>28)+0.5f)*0.25f);
                threadgroup half *ws=Ws+w_row*32u;
                #pragma unroll
                for (uint i=0;i<16;++i){ uint kl=w_k0+i; uint oct=(kl&31u)/8u, lo=kl&7u;
                    uint gkey=(aux_idx>>(oct*8u))&0xffu; float o=(float)(int8_t)Gs[gkey*8u+lo];
                    uint sx=(aux_sgn>>(oct*7u))&0x7fu; float sgn=((Ss[sx]>>lo)&1u)?-1.f:1.f;
                    ws[kl]=(half)(db*sgn*o); }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            metal::tensor<threadgroup half, metal::extents<int,32,32>, metal::tensor_inline> tA(Xs+sg_m*32u, metal::extents<int,32,32>{}); ca.load(tA);
            metal::tensor<threadgroup half, metal::extents<int,32,32>, metal::tensor_inline> tB(Ws+sg_n*32u, metal::extents<int,32,32>{}); cb.load(tB);
            op.run(ca,cb,cc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup float *csg=Cs+sgitg*1024u;
        metal::tensor<threadgroup float, metal::extents<int,32,32>, metal::tensor_inline> tC(csg, metal::extents<int,32,32>{}); cc.store(tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e=tiitg%32u; e<1024u; e+=32u){ uint ii=e/32u, jj=e%32u; uint r=sg_m+ii;
            if (r>=sub_offset && r<sub_end){ uint gr=m_tile+r, gc=n_tile+sg_n+jj; if(gr<m_total&&gc<n_out) out[gr*n_out+gc]=(half)csg[e]; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sub_offset=sub_end;
    }
#endif
}
