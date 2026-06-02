#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Production hand-tuned MoE IQ2_XXS register-block GEMM with ragged multi-expert
// run handling. Rows are expert-sorted; a 64-row tile may straddle an expert
// boundary, so we walk contiguous same-expert sub-runs within the tile, each
// with its own expert weights. 64 tok x 64 out/threadgroup, 4 simdgroups, 16
// register simdgroup_float8x8 accumulators. grid+signs preloaded to threadgroup.
// out[m,n] = sum_k x[m,k] * W_e[n,k], e = indices[m]. Matches FFAI bm64 bindings.

kernel void ffai_moe_bgemm_iq2xxs_handmma_f16(
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
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tiitg [[thread_index_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint n_tile = tgid.x * 64u;
    const uint m_tile = tgid.y * 64u;

    threadgroup half  Xs[64*32];
    threadgroup half  Ws[64*32];
    threadgroup uchar Gs[256*8];
    threadgroup uchar Ss[128];
    threadgroup float Ct[64*64];
    for (uint i = tiitg; i < 2048u; i += 128u) Gs[i] = grid[i];
    for (uint i = tiitg; i < 128u;  i += 128u) Ss[i] = signs[i];

    const uint sg_m = (sgitg / 2u) * 32u;
    const uint sg_n = (sgitg & 1u) * 32u;
    const uint x_row = tiitg / 2u;
    const uint x_k0  = (tiitg & 1u) * 16u;
    const uint w_flat = tiitg * 16u;
    const uint w_row  = w_flat / 32u;
    const uint w_k0   = w_flat & 31u;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub_offset = 0u;
    while (sub_offset < 64u) {
        uint cur_row = m_tile + sub_offset;
        if (cur_row >= m_total) break;
        uint expert = indices[cur_row];
        // sub_end: first row in [sub_offset+1,64) with a different expert / OOR.
        uint sub_end = 64u;
        for (uint p = sub_offset + 1u; p < 64u; ++p) {
            uint pr = m_tile + p;
            if (pr >= m_total || indices[pr] != expert) { sub_end = p; break; }
        }
        const uint eu16 = (tensor_byte_off + expert * expert_byte_stride) / 2u;

        simdgroup_matrix<float,8,8> c00(0.f),c01(0.f),c02(0.f),c03(0.f),c10(0.f),c11(0.f),c12(0.f),c13(0.f),
                                    c20(0.f),c21(0.f),c22(0.f),c23(0.f),c30(0.f),c31(0.f),c32(0.f),c33(0.f);
        simdgroup_matrix<half,8,8> a0,a1,a2,a3,b0,b1,b2,b3;

        for (uint kb = 0u; kb < k_in; kb += 32u) {
            {
                uint gr = m_tile + x_row;
                bool in_run = (x_row >= sub_offset) && (x_row < sub_end) && (gr < m_total);
                threadgroup half4 *xs = (threadgroup half4 *)(Xs + x_row * 32u + x_k0);
                if (in_run) {
                    const device half4 *xp = (const device half4 *)(x + gr * k_in + kb + x_k0);
                    #pragma unroll
                    for (uint i=0;i<4;++i) xs[i]=xp[i];
                } else {
                    #pragma unroll
                    for (uint i=0;i<4;++i) xs[i]=half4(0);
                }
            }
            {
                uint vidx0 = (n_tile + w_row) * k_in + kb + w_k0;
                uint block = vidx0 / 256u, group = (vidx0 & 255u) / 32u;
                uint b16 = eu16 + block * 33u;
                float d = (float)view_f16[b16];
                uint q = b16 + 1u + group * 4u;
                uint ai = (uint)view_u16[q] | ((uint)view_u16[q+1u]<<16);
                uint as = (uint)view_u16[q+2u] | ((uint)view_u16[q+3u]<<16);
                float db = d * (((float)(as>>28)+0.5f)*0.25f);
                threadgroup half *ws = Ws + w_row*32u;
                #pragma unroll
                for (uint i=0;i<16;++i){
                    uint kl=w_k0+i; uint oct=(kl&31u)/8u; uint lo=kl&7u;
                    uint gk=(ai>>(oct*8u))&0xffu; float o=(float)(int8_t)Gs[gk*8u+lo];
                    uint sx=(as>>(oct*7u))&0x7fu; float sgn=((Ss[sx]>>lo)&1u)?-1.f:1.f;
                    ws[kl]=(half)(db*sgn*o);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint ki=0u; ki<32u; ki+=8u) {
                simdgroup_load(a0, Xs+(sg_m+ 0u)*32u+ki,32u); simdgroup_load(a1, Xs+(sg_m+ 8u)*32u+ki,32u);
                simdgroup_load(a2, Xs+(sg_m+16u)*32u+ki,32u); simdgroup_load(a3, Xs+(sg_m+24u)*32u+ki,32u);
                simdgroup_load(b0, Ws+(sg_n+ 0u)*32u+ki,32u,0,true); simdgroup_load(b1, Ws+(sg_n+ 8u)*32u+ki,32u,0,true);
                simdgroup_load(b2, Ws+(sg_n+16u)*32u+ki,32u,0,true); simdgroup_load(b3, Ws+(sg_n+24u)*32u+ki,32u,0,true);
                simdgroup_multiply_accumulate(c00,a0,b0,c00); simdgroup_multiply_accumulate(c01,a0,b1,c01);
                simdgroup_multiply_accumulate(c02,a0,b2,c02); simdgroup_multiply_accumulate(c03,a0,b3,c03);
                simdgroup_multiply_accumulate(c10,a1,b0,c10); simdgroup_multiply_accumulate(c11,a1,b1,c11);
                simdgroup_multiply_accumulate(c12,a1,b2,c12); simdgroup_multiply_accumulate(c13,a1,b3,c13);
                simdgroup_multiply_accumulate(c20,a2,b0,c20); simdgroup_multiply_accumulate(c21,a2,b1,c21);
                simdgroup_multiply_accumulate(c22,a2,b2,c22); simdgroup_multiply_accumulate(c23,a2,b3,c23);
                simdgroup_multiply_accumulate(c30,a3,b0,c30); simdgroup_multiply_accumulate(c31,a3,b1,c31);
                simdgroup_multiply_accumulate(c32,a3,b2,c32); simdgroup_multiply_accumulate(c33,a3,b3,c33);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(c00,Ct+(sg_m+ 0u)*64u+sg_n+ 0u,64u); simdgroup_store(c01,Ct+(sg_m+ 0u)*64u+sg_n+ 8u,64u);
        simdgroup_store(c02,Ct+(sg_m+ 0u)*64u+sg_n+16u,64u); simdgroup_store(c03,Ct+(sg_m+ 0u)*64u+sg_n+24u,64u);
        simdgroup_store(c10,Ct+(sg_m+ 8u)*64u+sg_n+ 0u,64u); simdgroup_store(c11,Ct+(sg_m+ 8u)*64u+sg_n+ 8u,64u);
        simdgroup_store(c12,Ct+(sg_m+ 8u)*64u+sg_n+16u,64u); simdgroup_store(c13,Ct+(sg_m+ 8u)*64u+sg_n+24u,64u);
        simdgroup_store(c20,Ct+(sg_m+16u)*64u+sg_n+ 0u,64u); simdgroup_store(c21,Ct+(sg_m+16u)*64u+sg_n+ 8u,64u);
        simdgroup_store(c22,Ct+(sg_m+16u)*64u+sg_n+16u,64u); simdgroup_store(c23,Ct+(sg_m+16u)*64u+sg_n+24u,64u);
        simdgroup_store(c30,Ct+(sg_m+24u)*64u+sg_n+ 0u,64u); simdgroup_store(c31,Ct+(sg_m+24u)*64u+sg_n+ 8u,64u);
        simdgroup_store(c32,Ct+(sg_m+24u)*64u+sg_n+16u,64u); simdgroup_store(c33,Ct+(sg_m+24u)*64u+sg_n+24u,64u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tiitg; i < 64u*64u; i += 128u) {
            uint r=i/64u, cc=i%64u;
            if (r >= sub_offset && r < sub_end) {
                uint gr=m_tile+r, gc=n_tile+cc;
                if (gr<m_total && gc<n_out) out[gr*n_out+gc] = (half)Ct[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sub_offset = sub_end;
    }
}
