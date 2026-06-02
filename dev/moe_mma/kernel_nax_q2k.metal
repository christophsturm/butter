#include <metal_stdlib>
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

// MoE Q2_K down-proj on M5 Neural Accelerators (matmul2d) + ragged run-detection.
// Q2_K block = 84B = 42 u16: scales[0..16], qs[16..80] (2-bit), d@80(u16 40),
// dmin@82(u16 41). w = d*scale*q - dmin*min. matmul2d 4 sg × 32×32×32 (~3× sg).

kernel void ffai_moe_bgemm_q2k_view_u16_bm64_f16(
    const device half   *x          [[buffer(0)]],
    const device ushort *view_u16   [[buffer(1)]],
    const device half   *view_f16   [[buffer(2)]],
    const device uint   *indices    [[buffer(3)]],
    device half         *out        [[buffer(4)]],
    constant uint &m_total          [[buffer(5)]],
    constant uint &n_out            [[buffer(6)]],
    constant uint &k_in             [[buffer(7)]],
    constant uint &tensor_byte_off  [[buffer(8)]],
    constant uint &expert_byte_stride [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
    const uint n_tile = tgid.x*64u, m_tile = tgid.y*64u;
    threadgroup half  Xs[64*32];
    threadgroup half  Ws[64*32];
    threadgroup float Cs[4*32*32];
    const uint sg_m=(sgitg/2u)*32u, sg_n=(sgitg&1u)*32u;
    const uint x_row=tiitg/2u, x_k0=(tiitg&1u)*16u;
    const uint w_flat=tiitg*16u, w_row=w_flat/32u, w_k0=w_flat&31u;

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
            {   uint global_row=n_tile+w_row;
                uint block=(global_row*k_in+kb+w_k0)/256u;
                uint blk=eu16+block*42u;
                float d=(float)view_f16[blk+40u], dmin=(float)view_f16[blk+41u];
                threadgroup half *ws=Ws+w_row*32u;
                #pragma unroll
                for (uint i=0;i<16;++i){ uint kl=w_k0+i; uint vidx=global_row*k_in+kb+kl;
                    uint inb=vidx&255u; uint hf=inb>>7u; uint yh=inb-hf*128u; uint jg=yh>>5u; uint yg=yh-jg*32u; uint shf=yg>>4u; uint l=yg-shf*16u;
                    uint shift=jg*2u; uint q_byte=hf*32u+shf*16u+l; uint sub=hf*8u+jg*2u+shf;
                    uint word_idx=q_byte>>2u; uint biw=q_byte&3u; uint qw=blk+8u+word_idx*2u;
                    uint word=(uint)view_u16[qw]|((uint)view_u16[qw+1u]<<16); uint qs_byte=(word>>(biw*8u))&0xffu; uint q2=(qs_byte>>shift)&0x3u;
                    uint sc_word=(uint)view_u16[blk+(sub>>1u)]; uint scb=(sc_word>>((sub&1u)*8u))&0xffu;
                    float sc4=(float)(scb&0xfu), mn4=(float)((scb>>4u)&0xfu);
                    ws[kl]=(half)(d*sc4*(float)q2 - dmin*mn4); }
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
