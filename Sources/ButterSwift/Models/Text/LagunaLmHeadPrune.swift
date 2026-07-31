// Copyright 2026 Tom Turney (@TheTom)
//
// Certified two-pass lm_head elision for Laguna decode (mlxfast notes/68 /
// DARKBLOOM). Stock decode lm_head reads full BF16 [100352, 2048] (~411 MB)
// per token. With an MXFP8 sidecar present this replaces the full GEMV with:
//
//   1. COARSE: fused MXFP8 GEMV → coarse c_i, certified bound delta_i,
//      and BF16 pre-fill of every row.
//   2. SELECTION: mask rows with c_i + delta_i >= thr, where
//      L = max(c - delta) and thr = L - |L|/64 (BF16 cast safety margin).
//   3. EXACT: fixed 4-row simdgroup blocks run full BF16 GEMV only when any
//      row is a candidate; non-candidates keep coarse_bf (provably below
//      the winner). Candidate slots are bit-identical to stock gemv.
//
// DEFAULT ON when MXFP8 sidecar + BF16 lm_head are available.
// Escape: BUTTER_LAGUNA_LM_HEAD_PRUNE=0
// Requires: lm_head_mxfp8_codes.u8 + scales + BF16 lm_head.weight
//
// T29: inline candidate mask. Folds SELECTION into EXACT:
// the exact kernel recomputes `coarse[r] + delta[r] >= thr` itself instead
// of reading a precomputed `is_cand` mask, so the coarse pass only needs to
// emit `coarse`/`delta` (no `coarse_bf` store) and the standalone select
// dispatch + its `is_cand` buffer disappear. Textually the same predicate
// and the same `bfloat(coarse[r])` cast as the retained path, so NaN/signed-
// zero behavior and the certificate are unchanged; this removes one of four
// GPU dispatches per decode step. This is a fixed per-token dispatch-count
// win independent of candidate-set size, so unlike the coarse-pass cost it
// helps the flat-logit (large-|C|) case too, not just confident-text.
// DEFAULT ON. Escape: BUTTER_LAGUNA_LMHEAD_INLINE_MASK=0 (restores the
// original four-dispatch coarse_bf/select/exact path above, byte-identical
// to pre-T29).

import Foundation
import Metal
import IronSwift

enum LagunaLmHeadPrune {
    /// DEFAULT ON: unset or any value other than `"0"` enables the certified
    /// two-pass decode head when the MXFP8 sidecar + BF16 weight exist.
    /// Set `BUTTER_LAGUNA_LM_HEAD_PRUNE=0` to restore stock full lm_head.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_LM_HEAD_PRUNE"] != "0"
    }

    /// T29: DEFAULT ON. Set `BUTTER_LAGUNA_LMHEAD_INLINE_MASK=0` to restore
    /// the pre-T29 four-dispatch coarse_bf/select/exact path (kept verbatim
    /// below for the kill switch and same-binary A/B).
    static var inlineMaskEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_LMHEAD_INLINE_MASK"] != "0"
    }

    /// T29 diagnosis-only probe: prints `[laguna-lmhead-diag] candidates=K`
    /// (K of 100352 rows passing the certified bound, i.e. the size of the
    /// candidate set |C| that the exact pass must full-GEMV) per decode
    /// step to stderr. Adds one extra atomic-counter dispatch plus a
    /// blocking GPU readback, so it is OFF the hot path used for the A/B
    /// timing gates above -- opt-in only, never default ON.
    static var diagEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_LMHEAD_DIAG"] == "1"
    }

    private static let vocab = 100_352
    private static let kDim = 2048
    // Protected by `lock`; same pattern as other host-side Metal PSO caches.
    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static let header = """
    #include <metal_stdlib>
    using namespace metal;

    // e4m3fn decode (fp8.h): half bit pattern (b&127)<<7 * 256, sign bit 7.
    static inline float laguna_e4m3_decode(uint8_t b) {
        half converted = as_type<half>(ushort(uint(b & 127u) << 7));
        converted = converted * (half)256.0f;
        return (b & 128u) ? -float(converted) : float(converted);
    }

    // e8m0 decode (fp8.h): bits<<23 as f32; bits==0 -> 2^-127.
    static inline float laguna_e8m0_decode(uint8_t b) {
        if (b == 0u) {
            return as_type<float>(0x00400000u);
        }
        return as_type<float>(uint(b) << 23);
    }

    // Certified |ratio - code| half-cell bound (notes/68). Denormal/normal
    // collapse to 2^(max(e,1)-11); top code 0x7E uses 186 (448*(√2-1) ceil).
    static inline float laguna_hs8(uint8_t b) {
        uint mag = uint(b) & 127u;
        uint e = mag >> 3;
        float h = as_type<float>((metal::max(e, 1u) + 116u) << 23);
        return (mag == 126u) ? 186.0f : h;
    }

    // Bit-parallel e4m3 decode of one packed word → 4 floats (byte order).
    static inline float4 laguna_e4m3_decode4(uint w) {
        uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
        uint hs = w >> 8;
        uint hi = ((hs & 0x007F007Fu) << 7) | ((hs & 0x00800080u) << 8);
        half2 h02 = as_type<half2>(lo) * half2((half)256.0f);
        half2 h13 = as_type<half2>(hi) * half2((half)256.0f);
        return float4(float(h02.x), float(h13.x), float(h02.y), float(h13.y));
    }
    """

    // Fused MXFP8 coarse + certified delta + BF16 pre-fill.
    // One simdgroup per vocab row; each of 32 lanes covers 64 elements (2×32).
    // delta_i = d_i*(1+γ) + 2*γ*m_i covers quantization error and float
    // rounding (γ = 2^-15; depth << γ relative).
    private static let coarseSrc = header + """
    kernel void laguna_lmhead_coarse_v2(
        device const bfloat *x [[buffer(0)]],
        device const uchar *codes [[buffer(1)]],
        device const uchar *scales [[buffer(2)]],
        device float *coarse [[buffer(3)]],
        device float *delta [[buffer(4)]],
        device bfloat *coarse_bf [[buffer(5)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
        constexpr float GAMMA = 0x1p-15f;
        uint row = tgid * 8u + sgid;
        if (row >= 100352u) return;

        const device uchar* crow = codes + size_t(row) * 2048u;
        const device uchar* srow = scales + size_t(row) * 64u;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        float m_acc = 0.0f;
        for (uint gg = 0; gg < 2u; ++gg) {
            uint g = 2u * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32u);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            float mg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32u);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8u; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf = as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 = metal::select(hsf, float4(186.0f), mag4 == 126u);
                float4 acv4 = metal::abs(cv4);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4u; ++k) {
                    float cv = cv4[k];
                    float xv = xv4[k];
                    float ax = ax4[k];
                    cg += xv * cv;
                    dg += ax * hs4[k];
                    mg += ax * acv4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
            m_acc += sd * mg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        m_acc = simd_sum(m_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
            coarse_bf[row] = bfloat(c_acc);
        }
    }
    """

    // L = max(coarse - delta); thr = L - |L|/64. Single TG reduction.
    // Associative max → tree shape does not change the float result.
    private static let maxThrSrc = header + """
    kernel void laguna_lmhead_max_thr(
        device const float *coarse [[buffer(0)]],
        device const float *delta [[buffer(1)]],
        device float *thr [[buffer(2)]],
        threadgroup float *sh [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]],
        uint lid [[thread_index_in_threadgroup]]) {
        constexpr uint VOCAB = 100352u;
        constexpr uint TG = 256u;
        float local = -INFINITY;
        for (uint i = tid; i < VOCAB; i += TG) {
            local = max(local, coarse[i] - delta[i]);
        }
        sh[lid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG / 2u; s > 0u; s >>= 1u) {
            if (lid < s) sh[lid] = max(sh[lid], sh[lid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) {
            float L = sh[0];
            thr[0] = L - metal::abs(L) * (1.0f / 64.0f);
        }
    }
    """

    private static let selectSrc = header + """
    kernel void laguna_lmhead_select(
        device const float *coarse [[buffer(0)]],
        device const float *delta [[buffer(1)]],
        device const float *thr [[buffer(2)]],
        device uchar *is_cand [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
        if (i >= 100352u) return;
        is_cand[i] = (coarse[i] + delta[i] >= thr[0]) ? uchar(1) : uchar(0);
    }
    """

    private static let exactSrc = header + """
    // Fixed 4-row blocks per simdgroup; early-out when no candidates.
    // Per-row arithmetic mirrors stock gemv_al_bfloat16 (bm8/bn1/sm1/sn32/tm4).
    kernel void laguna_lmhead_exact_block(
        device const bfloat *coarse_bf [[buffer(0)]],
        device const bfloat *lm_head [[buffer(1)]],
        device const bfloat *x [[buffer(2)]],
        device const uchar *is_cand [[buffer(3)]],
        device bfloat *assembled [[buffer(4)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
        constexpr uint VOCAB = 100352u;
        constexpr uint K = 2048u;
        uint base = tgid * 32u + sgid * 4u;
        bool any_c = false;
        #pragma unroll
        for (uint tm = 0; tm < 4u; ++tm) {
            uint r = base + tm;
            any_c = any_c || (r < VOCAB && is_cand[r] != 0);
        }
        if (!any_c) {
            if (lane < 4u && base + lane < VOCAB)
                assembled[base + lane] = coarse_bf[base + lane];
            return;
        }
        thread float result[4] = {0.f, 0.f, 0.f, 0.f};
        uint bn = lane * 4u;
        for (uint i = 0; i < 16u; ++i) {
            vec<bfloat, 4> xv = *((const device vec<bfloat, 4>*)(x + bn));
            float v0 = float(xv.x), v1 = float(xv.y);
            float v2 = float(xv.z), v3 = float(xv.w);
            #pragma unroll
            for (uint tm = 0; tm < 4u; ++tm) {
                const device bfloat* mrow = lm_head + size_t(base + tm) * K;
                vec<bfloat, 4> mv = *((const device vec<bfloat, 4>*)(mrow + bn));
                result[tm] += float(mv.x)*v0 + float(mv.y)*v1
                            + float(mv.z)*v2 + float(mv.w)*v3;
            }
            bn += 128u;
        }
        #pragma unroll
        for (uint tm = 0; tm < 4u; ++tm)
            for (ushort sn = 16; sn >= 1; sn >>= 1)
                result[tm] += simd_shuffle_down(result[tm], sn);
        if (lane == 0) {
            #pragma unroll
            for (uint tm = 0; tm < 4u; ++tm) {
                uint r = base + tm;
                if (r < VOCAB)
                    assembled[r] = (is_cand[r] != 0)
                        ? bfloat(result[tm]) : coarse_bf[r];
            }
        }
    }
    """

    // T29 inline-mask coarse: identical accumulation/bound sequence to
    // `coarseSrc` above, minus the `coarse_bf[row] = bfloat(c_acc)` store.
    // The inline exact kernel below recomputes that same cast from the f32
    // `coarse` buffer it already reads, so dropping the write here costs
    // nothing and removes one bf16 store per row from the coarse dispatch.
    private static let inlineCoarseSrc = header + """
    kernel void laguna_lmhead_inline_coarse_v2(
        device const bfloat *x [[buffer(0)]],
        device const uchar *codes [[buffer(1)]],
        device const uchar *scales [[buffer(2)]],
        device float *coarse [[buffer(3)]],
        device float *delta [[buffer(4)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
        constexpr float GAMMA = 0x1p-15f;
        uint row = tgid * 8u + sgid;
        if (row >= 100352u) return;

        const device uchar* crow = codes + size_t(row) * 2048u;
        const device uchar* srow = scales + size_t(row) * 64u;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        float m_acc = 0.0f;
        for (uint gg = 0; gg < 2u; ++gg) {
            uint g = 2u * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            const device uint4* cptr = (const device uint4*)(crow + g * 32u);
            uint4 packed0 = cptr[0];
            uint4 packed1 = cptr[1];
            float cg = 0.0f;
            float dg = 0.0f;
            float mg = 0.0f;
            const device ushort4* xrow = (const device ushort4*)(x + g * 32u);
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 8u; ++w) {
                uint word = (w < 4u) ? packed0[w & 3u] : packed1[w & 3u];
                float4 cv4 = laguna_e4m3_decode4(word);
                float4 xv4 = as_type<float4>(uint4(xrow[w]) << 16);
                float4 ax4 = metal::abs(xv4);
                uint4 b4 = (uint4(word) >> uint4(0u, 8u, 16u, 24u)) & 255u;
                uint4 mag4 = b4 & 127u;
                uint4 e4 = mag4 >> 3;
                float4 hsf = as_type<float4>((metal::max(e4, uint4(1u)) + 116u) << 23);
                float4 hs4 = metal::select(hsf, float4(186.0f), mag4 == 126u);
                float4 acv4 = metal::abs(cv4);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4u; ++k) {
                    float cv = cv4[k];
                    float xv = xv4[k];
                    float ax = ax4[k];
                    cg += xv * cv;
                    dg += ax * hs4[k];
                    mg += ax * acv4[k];
                }
            }
            c_acc += sd * cg;
            d_acc += sd * dg;
            m_acc += sd * mg;
        }
        c_acc = simd_sum(c_acc);
        d_acc = simd_sum(d_acc);
        m_acc = simd_sum(m_acc);
        if (lane == 0) {
            coarse[row] = c_acc;
            delta[row] = d_acc * (1.0f + GAMMA) + (2.0f * GAMMA) * m_acc;
        }
    }
    """

    // T29 inline-mask exact pass. Folds SELECTION into EXACT: each
    // simdgroup recomputes `coarse[r] + delta[r] >= thr[0]` itself for its
    // fixed 4-row block instead of reading a precomputed `is_cand[r]` byte,
    // so the standalone select dispatch and its buffer disappear. The
    // predicate text and the `bfloat(coarse[r])` non-candidate cast are
    // copied verbatim from `selectSrc`/`exactSrc` above (same FP32 reload,
    // same comparison, same cast), and the stock-gemv replica body is
    // untouched -- so candidate rows stay bit-identical to `exactSrc`'s and
    // the certificate is unaffected.
    private static let inlineExactSrc = header + """
    kernel void laguna_lmhead_exact_inline_mask_block(
        device const float *coarse [[buffer(0)]],
        device const float *delta [[buffer(1)]],
        device const float *thr [[buffer(2)]],
        device const bfloat *lm_head [[buffer(3)]],
        device const bfloat *x [[buffer(4)]],
        device bfloat *assembled [[buffer(5)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
        constexpr uint VOCAB = 100352u;
        constexpr uint K = 2048u;
        uint base = tgid * 32u + sgid * 4u;
        bool any_c = false;
        #pragma unroll
        for (uint tm = 0; tm < 4u; ++tm) {
            uint r = base + tm;
            any_c = any_c || (r < VOCAB && coarse[r] + delta[r] >= thr[0]);
        }
        if (!any_c) {
            if (lane < 4u && base + lane < VOCAB)
                assembled[base + lane] = bfloat(coarse[base + lane]);
            return;
        }
        thread float result[4] = {0.f, 0.f, 0.f, 0.f};
        uint bn = lane * 4u;
        for (uint i = 0; i < 16u; ++i) {
            vec<bfloat, 4> xv = *((const device vec<bfloat, 4>*)(x + bn));
            float v0 = float(xv.x), v1 = float(xv.y);
            float v2 = float(xv.z), v3 = float(xv.w);
            #pragma unroll
            for (uint tm = 0; tm < 4u; ++tm) {
                const device bfloat* mrow = lm_head + size_t(base + tm) * K;
                vec<bfloat, 4> mv = *((const device vec<bfloat, 4>*)(mrow + bn));
                result[tm] += float(mv.x)*v0 + float(mv.y)*v1
                            + float(mv.z)*v2 + float(mv.w)*v3;
            }
            bn += 128u;
        }
        #pragma unroll
        for (uint tm = 0; tm < 4u; ++tm)
            for (ushort sn = 16; sn >= 1; sn >>= 1)
                result[tm] += simd_shuffle_down(result[tm], sn);
        if (lane == 0) {
            #pragma unroll
            for (uint tm = 0; tm < 4u; ++tm) {
                uint r = base + tm;
                if (r < VOCAB)
                    assembled[r] = (coarse[r] + delta[r] >= thr[0])
                        ? bfloat(result[tm]) : bfloat(coarse[r]);
            }
        }
    }
    """

    // T29 diagnosis-only: counts rows passing the certified candidate bound
    // (|C| in the exact pass's terms). Never on the perf-gated hot path,
    // see `diagEnabled` above.
    private static let candCountSrc = header + """
    kernel void laguna_lmhead_cand_count(
        device const float *coarse [[buffer(0)]],
        device const float *delta [[buffer(1)]],
        device const float *thr [[buffer(2)]],
        device atomic_uint *count [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
        if (i >= 100352u) return;
        if (coarse[i] + delta[i] >= thr[0]) {
            atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
        }
    }
    """

    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        let opts = MTLCompileOptions()
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    /// Pruned decode head. `normed` is post-finalNorm hidden [hidden] bf16.
    /// `mx` is MXFP8 sidecar; `bf16Weight` is full BF16 lm_head [vocab, hidden].
    static func logits(
        normed: Tensor,
        mx: Mxfp8Linear,
        bf16Weight: Tensor,
        on cmd: MTLCommandBuffer,
        device: Device
    ) -> Tensor {
        precondition(normed.dtype == .bf16 && bf16Weight.dtype == .bf16)
        precondition(bf16Weight.shape == [vocab, kDim])
        let mtl = device.mtlDevice

        let coarse = Tensor.empty(shape: [vocab], dtype: .f32, device: device)
        let delta = Tensor.empty(shape: [vocab], dtype: .f32, device: device)

        if inlineMaskEnabled {
            // T29: 1) inline coarse (coarse, delta only, no coarse_bf store)
            let coarsePSO = pso(
                "laguna_lmhead_inline_coarse_v2", source: inlineCoarseSrc, device: mtl)
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(coarsePSO)
                enc.setBuffer(normed.buffer, offset: normed.offset, index: 0)
                enc.setBuffer(mx.weight.buffer, offset: mx.weight.offset, index: 1)
                enc.setBuffer(mx.scales.buffer, offset: mx.scales.offset, index: 2)
                enc.setBuffer(coarse.buffer, offset: coarse.offset, index: 3)
                enc.setBuffer(delta.buffer, offset: delta.offset, index: 4)
                enc.dispatchThreadgroups(
                    MTLSize(width: (vocab + 7) / 8, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }

            // 2) thr = max(coarse - delta) - |L|/64 (single TG reduction)
            let thrBuf = Tensor.empty(shape: [1], dtype: .f32, device: device)
            let maxPSO = pso("laguna_lmhead_max_thr", source: maxThrSrc, device: mtl)
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(maxPSO)
                enc.setBuffer(coarse.buffer, offset: coarse.offset, index: 0)
                enc.setBuffer(delta.buffer, offset: delta.offset, index: 1)
                enc.setBuffer(thrBuf.buffer, offset: thrBuf.offset, index: 2)
                enc.setThreadgroupMemoryLength(256 * 4, index: 0)
                enc.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }

            // T29 diagnosis-only probe (BUTTER_LAGUNA_LMHEAD_DIAG=1): count
            // |C| via an extra atomic dispatch, read back on cmd completion.
            // Off the perf-gated path entirely when unset.
            if diagEnabled {
                // Bypass scratch-slab allocation: this buffer must stay
                // valid until `cmd`'s completion handler runs (after the
                // scratch slab may already have been reset for later
                // layers/tokens), so it gets its own dedicated MTLBuffer.
                let countBuf = Tensor(
                    buffer: device.mtlDevice.makeBuffer(
                        length: 4, options: .storageModeShared)!,
                    offset: 0, shape: [1], dtype: .u32)
                countBuf.zero()
                let countPSO = pso("laguna_lmhead_cand_count", source: candCountSrc, device: mtl)
                if let enc = cmd.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(countPSO)
                    enc.setBuffer(coarse.buffer, offset: coarse.offset, index: 0)
                    enc.setBuffer(delta.buffer, offset: delta.offset, index: 1)
                    enc.setBuffer(thrBuf.buffer, offset: thrBuf.offset, index: 2)
                    enc.setBuffer(countBuf.buffer, offset: countBuf.offset, index: 3)
                    enc.dispatchThreads(
                        MTLSize(width: vocab, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    enc.endEncoding()
                }
                cmd.addCompletedHandler { _ in
                    let k = countBuf.buffer.contents()
                        .advanced(by: countBuf.offset)
                        .assumingMemoryBound(to: UInt32.self).pointee
                    FileHandle.standardError.write(
                        Data("[laguna-lmhead-diag] candidates=\(k)/\(vocab)\n".utf8))
                }
            }

            // 3) Exact block with the candidate test folded in, no separate
            // select dispatch, no is_cand buffer, no coarse_bf buffer.
            let assembled = Tensor.empty(shape: [vocab], dtype: .bf16, device: device)
            let exactPSO = pso(
                "laguna_lmhead_exact_inline_mask_block", source: inlineExactSrc, device: mtl)
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(exactPSO)
                enc.setBuffer(coarse.buffer, offset: coarse.offset, index: 0)
                enc.setBuffer(delta.buffer, offset: delta.offset, index: 1)
                enc.setBuffer(thrBuf.buffer, offset: thrBuf.offset, index: 2)
                enc.setBuffer(bf16Weight.buffer, offset: bf16Weight.offset, index: 3)
                enc.setBuffer(normed.buffer, offset: normed.offset, index: 4)
                enc.setBuffer(assembled.buffer, offset: assembled.offset, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: vocab / 32, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }
            return assembled
        }

        // Kill-switch fallback (BUTTER_LAGUNA_LMHEAD_INLINE_MASK=0): original
        // four-dispatch coarse_bf/select/exact path, byte-identical to pre-T29.

        // 1) Fused MXFP8 coarse → f32 coarse + f32 delta + bf16 coarse
        let coarseBF = Tensor.empty(shape: [vocab], dtype: .bf16, device: device)
        let coarsePSO = pso("laguna_lmhead_coarse_v2", source: coarseSrc, device: mtl)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(coarsePSO)
            enc.setBuffer(normed.buffer, offset: normed.offset, index: 0)
            enc.setBuffer(mx.weight.buffer, offset: mx.weight.offset, index: 1)
            enc.setBuffer(mx.scales.buffer, offset: mx.scales.offset, index: 2)
            enc.setBuffer(coarse.buffer, offset: coarse.offset, index: 3)
            enc.setBuffer(delta.buffer, offset: delta.offset, index: 4)
            enc.setBuffer(coarseBF.buffer, offset: coarseBF.offset, index: 5)
            // 8 simdgroups per TG (256 threads) → vocab/8 TGs
            enc.dispatchThreadgroups(
                MTLSize(width: (vocab + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 2) thr = max(coarse - delta) - |L|/64 (single TG reduction)
        let thrBuf = Tensor.empty(shape: [1], dtype: .f32, device: device)
        let maxPSO = pso("laguna_lmhead_max_thr", source: maxThrSrc, device: mtl)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(maxPSO)
            enc.setBuffer(coarse.buffer, offset: coarse.offset, index: 0)
            enc.setBuffer(delta.buffer, offset: delta.offset, index: 1)
            enc.setBuffer(thrBuf.buffer, offset: thrBuf.offset, index: 2)
            enc.setThreadgroupMemoryLength(256 * 4, index: 0)
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 3) Candidate mask: c_i + delta_i >= thr
        let isCand = Tensor.empty(shape: [vocab], dtype: .u8, device: device)
        let selectPSO = pso("laguna_lmhead_select", source: selectSrc, device: mtl)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(selectPSO)
            enc.setBuffer(coarse.buffer, offset: coarse.offset, index: 0)
            enc.setBuffer(delta.buffer, offset: delta.offset, index: 1)
            enc.setBuffer(thrBuf.buffer, offset: thrBuf.offset, index: 2)
            enc.setBuffer(isCand.buffer, offset: isCand.offset, index: 3)
            enc.dispatchThreads(
                MTLSize(width: vocab, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }

        // 4) Exact block: 3136 TGs × 256 threads, early-out non-candidates
        let assembled = Tensor.empty(shape: [vocab], dtype: .bf16, device: device)
        let exactPSO = pso("laguna_lmhead_exact_block", source: exactSrc, device: mtl)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(exactPSO)
            enc.setBuffer(coarseBF.buffer, offset: coarseBF.offset, index: 0)
            enc.setBuffer(bf16Weight.buffer, offset: bf16Weight.offset, index: 1)
            enc.setBuffer(normed.buffer, offset: normed.offset, index: 2)
            enc.setBuffer(isCand.buffer, offset: isCand.offset, index: 3)
            enc.setBuffer(assembled.buffer, offset: assembled.offset, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: vocab / 32, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        return assembled
    }
}
