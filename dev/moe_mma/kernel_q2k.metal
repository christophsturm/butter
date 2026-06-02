#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Hand-tuned MoE Q2_K register-block GEMM (down projection). Reads RAW Q2_K
// blocks via a no-copy view (no deinterleave pool). Register-resident
// simdgroup_float8x8 accumulators, 64x64 tile, vectorized half4 X-load, ragged
// multi-expert run-detection. Q2_K block = 84B = 42 u16: scales[0..16] (4-bit
// d-scale low + 4-bit min-scale high per 16-val sub-block), qs[16..80] (2-bit
// quants), d f16@80 (u16 40), dmin f16@82 (u16 41). w = d*scale*q - dmin*min.
// out[m,n] = sum_k x[m,k]*W_e[n,k], e=indices[m]. Matches q2k_view bindings.

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
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tiitg [[thread_index_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint n_tile = tgid.x * 64u;
    const uint m_tile = tgid.y * 64u;
    threadgroup half  Xs[64*32];
    threadgroup half  Ws[64*32];
    threadgroup float Ct[64*64];
    const uint sg_m = (sgitg / 2u) * 32u;
    const uint sg_n = (sgitg & 1u) * 32u;
    const uint x_row = tiitg / 2u;
    const uint x_k0  = (tiitg & 1u) * 16u;
    const uint w_flat = tiitg * 16u;
    const uint w_row  = w_flat / 32u;
    const uint w_k0   = w_flat & 31u;

    uint sub_offset = 0u;
    while (sub_offset < 64u) {
        uint cur_row = m_tile + sub_offset;
        if (cur_row >= m_total) break;
        uint expert = indices[cur_row];
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
                uint global_row = n_tile + w_row;
                uint block = (global_row * k_in + kb + w_k0) / 256u;
                uint blk = eu16 + block * 42u;
                float d = (float)view_f16[blk + 40u];
                float dmin = (float)view_f16[blk + 41u];
                threadgroup half *ws = Ws + w_row*32u;
                #pragma unroll
                for (uint i=0;i<16;++i){
                    uint kl = w_k0 + i;
                    uint vidx = global_row * k_in + kb + kl;
                    uint inb = vidx & 255u;
                    uint hf = inb >> 7u;            // /128
                    uint yh = inb - hf*128u;
                    uint jg = yh >> 5u;             // /32
                    uint yg = yh - jg*32u;
                    uint shf = yg >> 4u;            // sub_half /16
                    uint l = yg - shf*16u;
                    uint shift = jg*2u;
                    uint q_byte = hf*32u + shf*16u + l;
                    uint sub = hf*8u + jg*2u + shf;
                    uint word_idx = q_byte >> 2u;
                    uint biw = q_byte & 3u;
                    uint qw = blk + 8u + word_idx*2u;
                    uint word = (uint)view_u16[qw] | ((uint)view_u16[qw+1u] << 16);
                    uint qs_byte = (word >> (biw*8u)) & 0xffu;
                    uint q2 = (qs_byte >> shift) & 0x3u;
                    uint sc_word = (uint)view_u16[blk + (sub>>1u)];
                    uint scb = (sc_word >> ((sub&1u)*8u)) & 0xffu;
                    float sc4 = (float)(scb & 0xfu);
                    float mn4 = (float)((scb >> 4u) & 0xfu);
                    ws[kl] = (half)(d * sc4 * (float)q2 - dmin * mn4);
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
