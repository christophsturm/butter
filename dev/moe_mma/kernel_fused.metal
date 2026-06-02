#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Fused gate+up+swiglu MoE IQ2_XXS GEMM (dev harness). One pass stages the
// activation X ONCE per K-tile and feeds BOTH the gate and up expert matmuls,
// then applies swiglu on store -> inner[m,n]. Halves the X re-read vs running
// gate and up as two separate GEMMs, and folds away the standalone swiglu pass.
// 64 tokens x 32 out per threadgroup; 4 simdgroups; each owns 32 tok x 16 out
// = 4 tok-frags x 2 out-frags = 8 gate-acc + 8 up-acc. silu(g)*u (limit clamp).

kernel void moe_mma_iq2_fused(
    const device half   *x          [[buffer(0)]],
    const device ushort *g_u16      [[buffer(1)]],  const device half *g_f16 [[buffer(2)]],
    const device ushort *u_u16      [[buffer(3)]],  const device half *u_f16 [[buffer(4)]],
    const device uchar  *grid       [[buffer(5)]],
    const device uchar  *signs      [[buffer(6)]],
    const device uint   *indices    [[buffer(7)]],
    device half         *inner      [[buffer(8)]],  // [m_total, n_out]
    constant uint &m_total          [[buffer(9)]],
    constant uint &n_out            [[buffer(10)]],
    constant uint &k_in             [[buffer(11)]],
    constant uint &tensor_byte_off  [[buffer(12)]],
    constant uint &expert_byte_stride [[buffer(13)]],
    constant float &swiglu_limit    [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tiitg [[thread_index_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint n_tile = tgid.x * 64u;
    const uint m_tile = tgid.y * 64u;

    threadgroup half  Xs[64*32];
    threadgroup half  gWs[64*32];
    threadgroup half  uWs[64*32];
    threadgroup uchar Gs[256*8];
    threadgroup uchar Ss[128];
    for (uint i = tiitg; i < 2048u; i += 128u) Gs[i] = grid[i];
    for (uint i = tiitg; i < 128u;  i += 128u) Ss[i] = signs[i];

    const uint sg_m = (sgitg / 2u) * 32u;
    const uint sg_n = (sgitg & 1u) * 32u;
    const uint x_row = tiitg / 2u;
    const uint x_k0  = (tiitg & 1u) * 16u;
    const uint w_flat = tiitg * 16u;
    const uint w_row  = w_flat / 32u;
    const uint w_k0   = w_flat & 31u;

    const uint expert = indices[m_tile];
    const uint eu16 = (tensor_byte_off + expert * expert_byte_stride) / 2u;

    simdgroup_matrix<float,8,8> g00(0.f),g01(0.f),g02(0.f),g03(0.f),g10(0.f),g11(0.f),g12(0.f),g13(0.f),
                                g20(0.f),g21(0.f),g22(0.f),g23(0.f),g30(0.f),g31(0.f),g32(0.f),g33(0.f);
    simdgroup_matrix<float,8,8> u00(0.f),u01(0.f),u02(0.f),u03(0.f),u10(0.f),u11(0.f),u12(0.f),u13(0.f),
                                u20(0.f),u21(0.f),u22(0.f),u23(0.f),u30(0.f),u31(0.f),u32(0.f),u33(0.f);
    simdgroup_matrix<half,8,8> a0,a1,a2,a3,gb0,gb1,gb2,gb3,ub0,ub1,ub2,ub3;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kb = 0u; kb < k_in; kb += 32u) {
        // X once.
        {
            uint gr = m_tile + x_row;
            threadgroup half4 *xs = (threadgroup half4 *)(Xs + x_row * 32u + x_k0);
            const device half4 *xp = (const device half4 *)(x + gr * k_in + kb + x_k0);
            if (m_tile + 64u <= m_total) { for (uint i=0;i<4;++i) xs[i]=xp[i]; }
            else { half4 z=half4(0); for (uint i=0;i<4;++i) xs[i]=(gr<m_total)?xp[i]:z; }
        }
        // Dequant gate + up weights for w_row (shared lane map, two sources).
        {
            uint vidx0 = (n_tile + w_row) * k_in + kb + w_k0;
            uint block = vidx0 / 256u, group = (vidx0 & 255u) / 32u;
            uint b16 = eu16 + block * 33u;
            uint q = b16 + 1u + group * 4u;
            float gd = (float)g_f16[b16];
            uint gai = (uint)g_u16[q] | ((uint)g_u16[q+1u]<<16); uint gas = (uint)g_u16[q+2u]|((uint)g_u16[q+3u]<<16);
            float gdb = gd * (((float)(gas>>28)+0.5f)*0.25f);
            float ud = (float)u_f16[b16];
            uint uai = (uint)u_u16[q] | ((uint)u_u16[q+1u]<<16); uint uas = (uint)u_u16[q+2u]|((uint)u_u16[q+3u]<<16);
            float udb = ud * (((float)(uas>>28)+0.5f)*0.25f);
            threadgroup half *gws = gWs + w_row*32u;
            threadgroup half *uws = uWs + w_row*32u;
            #pragma unroll
            for (uint i=0;i<16;++i){
                uint kl=w_k0+i; uint oct=(kl&31u)/8u; uint lo=kl&7u;
                uint gk=(gai>>(oct*8u))&0xffu; float go=(float)(int8_t)Gs[gk*8u+lo];
                uint gs=(gas>>(oct*7u))&0x7fu; float gsg=((Ss[gs]>>lo)&1u)?-1.f:1.f;
                gws[kl]=(half)(gdb*gsg*go);
                uint uk=(uai>>(oct*8u))&0xffu; float uo=(float)(int8_t)Gs[uk*8u+lo];
                uint us=(uas>>(oct*7u))&0x7fu; float usg=((Ss[us]>>lo)&1u)?-1.f:1.f;
                uws[kl]=(half)(udb*usg*uo);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint ki=0u; ki<32u; ki+=8u) {
            simdgroup_load(a0, Xs+(sg_m+ 0u)*32u+ki,32u); simdgroup_load(a1, Xs+(sg_m+ 8u)*32u+ki,32u);
            simdgroup_load(a2, Xs+(sg_m+16u)*32u+ki,32u); simdgroup_load(a3, Xs+(sg_m+24u)*32u+ki,32u);
            simdgroup_load(gb0, gWs+(sg_n+ 0u)*32u+ki,32u,0,true); simdgroup_load(gb1, gWs+(sg_n+ 8u)*32u+ki,32u,0,true);
            simdgroup_load(gb2, gWs+(sg_n+16u)*32u+ki,32u,0,true); simdgroup_load(gb3, gWs+(sg_n+24u)*32u+ki,32u,0,true);
            simdgroup_load(ub0, uWs+(sg_n+ 0u)*32u+ki,32u,0,true); simdgroup_load(ub1, uWs+(sg_n+ 8u)*32u+ki,32u,0,true);
            simdgroup_load(ub2, uWs+(sg_n+16u)*32u+ki,32u,0,true); simdgroup_load(ub3, uWs+(sg_n+24u)*32u+ki,32u,0,true);
            simdgroup_multiply_accumulate(g00,a0,gb0,g00); simdgroup_multiply_accumulate(g01,a0,gb1,g01);
            simdgroup_multiply_accumulate(g02,a0,gb2,g02); simdgroup_multiply_accumulate(g03,a0,gb3,g03);
            simdgroup_multiply_accumulate(g10,a1,gb0,g10); simdgroup_multiply_accumulate(g11,a1,gb1,g11);
            simdgroup_multiply_accumulate(g12,a1,gb2,g12); simdgroup_multiply_accumulate(g13,a1,gb3,g13);
            simdgroup_multiply_accumulate(g20,a2,gb0,g20); simdgroup_multiply_accumulate(g21,a2,gb1,g21);
            simdgroup_multiply_accumulate(g22,a2,gb2,g22); simdgroup_multiply_accumulate(g23,a2,gb3,g23);
            simdgroup_multiply_accumulate(g30,a3,gb0,g30); simdgroup_multiply_accumulate(g31,a3,gb1,g31);
            simdgroup_multiply_accumulate(g32,a3,gb2,g32); simdgroup_multiply_accumulate(g33,a3,gb3,g33);
            simdgroup_multiply_accumulate(u00,a0,ub0,u00); simdgroup_multiply_accumulate(u01,a0,ub1,u01);
            simdgroup_multiply_accumulate(u02,a0,ub2,u02); simdgroup_multiply_accumulate(u03,a0,ub3,u03);
            simdgroup_multiply_accumulate(u10,a1,ub0,u10); simdgroup_multiply_accumulate(u11,a1,ub1,u11);
            simdgroup_multiply_accumulate(u12,a1,ub2,u12); simdgroup_multiply_accumulate(u13,a1,ub3,u13);
            simdgroup_multiply_accumulate(u20,a2,ub0,u20); simdgroup_multiply_accumulate(u21,a2,ub1,u21);
            simdgroup_multiply_accumulate(u22,a2,ub2,u22); simdgroup_multiply_accumulate(u23,a2,ub3,u23);
            simdgroup_multiply_accumulate(u30,a3,ub0,u30); simdgroup_multiply_accumulate(u31,a3,ub1,u31);
            simdgroup_multiply_accumulate(u32,a3,ub2,u32); simdgroup_multiply_accumulate(u33,a3,ub3,u33);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // swiglu element-wise on register frag-pairs: each gate frag g[fr][fc] and
    // up frag u[fr][fc] share the SAME lane->(tok,out) map, so combine them with
    // NO full 64x64 store tiles (that would overflow threadgroup). Store each
    // 8x8 pair to a tiny per-simdgroup scratch (2KB total), apply swiglu, scatter.
    threadgroup float sc[4*128];   // per-sg: 8x8 gate + 8x8 up
    threadgroup float *scg = sc + sgitg * 128u;
    threadgroup float *scu = scg + 64u;
#define STORE_SW(GF, UF, FR, FC) \
    simdgroup_store(GF, scg, 8u); simdgroup_store(UF, scu, 8u); \
    simdgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint e=0u;e<2u;++e){ uint idx=lane*2u+e; uint ii=idx/8u, jj=idx&7u; \
        uint gr=m_tile+sg_m+(FR)*8u+ii, gc=n_tile+sg_n+(FC)*8u+jj; \
        if (gr<m_total && gc<n_out){ float gg=scg[idx], uu=scu[idx]; \
            if(swiglu_limit>0.f){gg=min(gg,swiglu_limit);uu=clamp(uu,-swiglu_limit,swiglu_limit);} \
            float s=gg/(1.f+exp(-gg)); inner[gr*n_out+gc]=(half)(s*uu);} } \
    simdgroup_barrier(mem_flags::mem_threadgroup);
    STORE_SW(g00,u00,0u,0u) STORE_SW(g01,u01,0u,1u) STORE_SW(g02,u02,0u,2u) STORE_SW(g03,u03,0u,3u)
    STORE_SW(g10,u10,1u,0u) STORE_SW(g11,u11,1u,1u) STORE_SW(g12,u12,1u,2u) STORE_SW(g13,u13,1u,3u)
    STORE_SW(g20,u20,2u,0u) STORE_SW(g21,u21,2u,1u) STORE_SW(g22,u22,2u,2u) STORE_SW(g23,u23,2u,3u)
    STORE_SW(g30,u30,3u,0u) STORE_SW(g31,u31,3u,1u) STORE_SW(g32,u32,3u,2u) STORE_SW(g33,u33,3u,3u)
#undef STORE_SW
}
