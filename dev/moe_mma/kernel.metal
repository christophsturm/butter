#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Original hand-tuned MoE IQ2_XXS register-block GEMM (development harness).
// out[m,n] = sum_k x[m,k] * W_e[n,k], expert e = indices[m]. Register-resident
// 8x8 simdgroup accumulators (no threadgroup C scratch → room to preload the
// IQ2 grid+signs tables into threadgroup, so the inner dequant reads from L1).
// 64 token-rows x 32 out-cols per threadgroup, 128 threads = 4 simdgroups.
// Each simdgroup owns 32 tokens x 16 out = 4 token-frags x 2 out-frags = 8 acc.
//
// IQ2_XXS block view: u16[33] per 256-value block = d(f16) + qs[32]u16. Group g
// (0..7, 32 vals each): aux_idx=qs[g*4]|qs[g*4+1]<<16, aux_sgn=qs[g*4+2]|qs[..3]
// <<16; scale4=aux_sgn>>28; db=d*(scale4+0.5)*0.25. Val k in group: oct=(k&31)/8,
// lane=k&7, gkey=(aux_idx>>(oct*8))&0xff, o=grid[gkey*8+lane](i8), sidx=(aux_sgn>>
// (oct*7))&0x7f, sign=(signs[sidx]>>lane)&1?-1:1; w=db*sign*o.

kernel void moe_mma_iq2(
    const device half   *x          [[buffer(0)]],   // [m_total, k_in]
    const device ushort *view_u16   [[buffer(1)]],   // raw blocks (u16 view)
    const device half   *view_f16   [[buffer(2)]],   // same buffer, f16 view (d)
    const device uchar  *grid       [[buffer(3)]],   // 256*8 signed octets
    const device uchar  *signs      [[buffer(4)]],   // 128
    const device uint   *indices    [[buffer(5)]],   // [m_total] expert per row
    device half         *out        [[buffer(6)]],   // [m_total, n_out]
    constant uint &m_total          [[buffer(7)]],
    constant uint &n_out            [[buffer(8)]],
    constant uint &k_in             [[buffer(9)]],
    constant uint &tensor_byte_off  [[buffer(10)]],
    constant uint &expert_byte_stride [[buffer(11)]],
    uint3 tgid   [[threadgroup_position_in_grid]],
    uint  tiitg  [[thread_index_in_threadgroup]],
    uint  sgitg  [[simdgroup_index_in_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]])
{
    const uint n_tile = tgid.x * 64u;     // out tile base (64 wide)
    const uint m_tile = tgid.y * 64u;     // token tile base (64)

    threadgroup half  Xs[64*32];          // 4KB
    threadgroup half  Ws[64*32];          // 4KB
    threadgroup uchar Gs[256*8];          // 2KB grid preload
    threadgroup uchar Ss[128];            // signs preload

    // Cooperative preload of grid (2048B) + signs (128B): 128 threads.
    for (uint i = tiitg; i < 2048u; i += 128u) Gs[i] = grid[i];
    for (uint i = tiitg; i < 128u;  i += 128u) Ss[i] = signs[i];

    const uint sg_m = (sgitg / 2u) * 32u; // token half 0/32
    const uint sg_n = (sgitg & 1u) * 32u; // out half 0/32 (each sg = 32 out)

    // staging lane maps
    const uint x_row = tiitg / 2u;        // 0..63
    const uint x_k0  = (tiitg & 1u) * 16u;// 0/16
    const uint w_flat = tiitg * 16u;      // 64*32 = 2048 = 128 lanes * 16
    const uint w_row  = w_flat / 32u;     // 0..63
    const uint w_k0   = w_flat & 31u;     // 0 / 16

    // For the dev harness: single expert across the whole tile (indices[m_tile]).
    const uint expert = indices[m_tile];
    const uint eu16 = (tensor_byte_off + expert * expert_byte_stride) / 2u;

    simdgroup_matrix<float,8,8> c00(0.f),c01(0.f),c02(0.f),c03(0.f),
                                c10(0.f),c11(0.f),c12(0.f),c13(0.f),
                                c20(0.f),c21(0.f),c22(0.f),c23(0.f),
                                c30(0.f),c31(0.f),c32(0.f),c33(0.f);
    simdgroup_matrix<half,8,8> a0,a1,a2,a3,b0,b1,b2,b3;

    threadgroup_barrier(mem_flags::mem_threadgroup);  // grid/signs ready

    for (uint kb = 0u; kb < k_in; kb += 32u) {
        // X stage: row x_row, cols [x_k0 .. x_k0+16) — 64 tokens x 32 K.
        // Vectorized half4 copy; full-tile fast path skips the per-row bounds
        // branch (the common case — only the last token-tile is ragged).
        {
            uint gr = m_tile + x_row;
            threadgroup half4 *xs = (threadgroup half4 *)(Xs + x_row * 32u + x_k0);
            if (m_tile + 64u <= m_total) {
                const device half4 *xp = (const device half4 *)(x + gr * k_in + kb + x_k0);
                #pragma unroll
                for (uint i = 0; i < 4; ++i) xs[i] = xp[i];
            } else {
                const device half4 *xp = (const device half4 *)(x + gr * k_in + kb + x_k0);
                half4 z = half4(0);
                #pragma unroll
                for (uint i = 0; i < 4; ++i) xs[i] = (gr < m_total) ? xp[i] : z;
            }
        }
        // W dequant: row w_row (0..63), 16 K from w_k0 (one 32-group). 128 lanes.
        {
            uint vidx0 = (n_tile + w_row) * k_in + kb + w_k0;
            uint block = vidx0 / 256u;
            uint group = (vidx0 & 255u) / 32u;
            uint b16 = eu16 + block * 33u;
            float d = (float)view_f16[b16];
            uint q = b16 + 1u + group * 4u;
            uint aux_idx = (uint)view_u16[q]     | ((uint)view_u16[q+1u] << 16);
            uint aux_sgn = (uint)view_u16[q+2u]  | ((uint)view_u16[q+3u] << 16);
            uint scale4 = aux_sgn >> 28;
            float db = d * (((float)scale4 + 0.5f) * 0.25f);
            threadgroup half *ws = Ws + w_row * 32u;
            #pragma unroll
            for (uint i = 0; i < 16; ++i) {
                uint kl = w_k0 + i;
                uint oct = (kl & 31u) / 8u;
                uint lo  = kl & 7u;
                uint gkey = (aux_idx >> (oct * 8u)) & 0xffu;
                float o = (float)(int8_t)Gs[gkey * 8u + lo];
                uint sidx = (aux_sgn >> (oct * 7u)) & 0x7fu;
                float sign = ((Ss[sidx] >> lo) & 1u) ? -1.f : 1.f;
                ws[kl] = (half)(db * sign * o);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint ki = 0u; ki < 32u; ki += 8u) {
            simdgroup_load(a0, Xs + (sg_m +  0u) * 32u + ki, 32u);
            simdgroup_load(a1, Xs + (sg_m +  8u) * 32u + ki, 32u);
            simdgroup_load(a2, Xs + (sg_m + 16u) * 32u + ki, 32u);
            simdgroup_load(a3, Xs + (sg_m + 24u) * 32u + ki, 32u);
            simdgroup_load(b0, Ws + (sg_n +  0u) * 32u + ki, 32u, 0, true);
            simdgroup_load(b1, Ws + (sg_n +  8u) * 32u + ki, 32u, 0, true);
            simdgroup_load(b2, Ws + (sg_n + 16u) * 32u + ki, 32u, 0, true);
            simdgroup_load(b3, Ws + (sg_n + 24u) * 32u + ki, 32u, 0, true);
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

    // Store: simdgroup_store all 16 frags into a 64×64 threadgroup tile, scatter.
    threadgroup float Ct[64*64];
    simdgroup_store(c00, Ct+(sg_m+ 0u)*64u+sg_n+ 0u,64u); simdgroup_store(c01, Ct+(sg_m+ 0u)*64u+sg_n+ 8u,64u);
    simdgroup_store(c02, Ct+(sg_m+ 0u)*64u+sg_n+16u,64u); simdgroup_store(c03, Ct+(sg_m+ 0u)*64u+sg_n+24u,64u);
    simdgroup_store(c10, Ct+(sg_m+ 8u)*64u+sg_n+ 0u,64u); simdgroup_store(c11, Ct+(sg_m+ 8u)*64u+sg_n+ 8u,64u);
    simdgroup_store(c12, Ct+(sg_m+ 8u)*64u+sg_n+16u,64u); simdgroup_store(c13, Ct+(sg_m+ 8u)*64u+sg_n+24u,64u);
    simdgroup_store(c20, Ct+(sg_m+16u)*64u+sg_n+ 0u,64u); simdgroup_store(c21, Ct+(sg_m+16u)*64u+sg_n+ 8u,64u);
    simdgroup_store(c22, Ct+(sg_m+16u)*64u+sg_n+16u,64u); simdgroup_store(c23, Ct+(sg_m+16u)*64u+sg_n+24u,64u);
    simdgroup_store(c30, Ct+(sg_m+24u)*64u+sg_n+ 0u,64u); simdgroup_store(c31, Ct+(sg_m+24u)*64u+sg_n+ 8u,64u);
    simdgroup_store(c32, Ct+(sg_m+24u)*64u+sg_n+16u,64u); simdgroup_store(c33, Ct+(sg_m+24u)*64u+sg_n+24u,64u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tiitg; i < 64u * 64u; i += 128u) {
        uint r = i / 64u, cc = i % 64u;
        uint gr = m_tile + r, gc = n_tile + cc;
        if (gr < m_total && gc < n_out) out[gr * n_out + gc] = (half)Ct[i];
    }
}
