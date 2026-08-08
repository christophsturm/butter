// Copyright 2026 Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Attention-side group-16 NVFP4 (E2M1 + E4M3 micro-scale, `global=1.0`)
// requant for LATE layers only: same decode-only side-layout idea as
// `LagunaAttnInt8` (T20), one notch narrower. T20 requanted every layer's
// Q/K/V(+gate)/o_proj to group-32 affine INT8 and explicitly left NVFP4
// group-16 for late layers unimplemented ("the added complexity wasn't
// worth another kernel" at the time). T39's per-kernel GPU-time probe
// changed that calculus: the two INT8-banked attention QMVs
// (`oproj_gated` + `qkv_qmv`) are ~61% of measured decode GPU-busy work,
// by far the largest lever left on the table, see
// benchmarks/laguna-xs/README.md T39/T41.
//
// Bit layout matches `OpsNVFP4.swift` / `MoELayer.StackedNvfp4Experts`
// exactly, so this bank's packed weight + scale bytes are byte-for-byte
// interchangeable with any other NVFP4 consumer in this codebase:
// `packed` is `[outDim, inDim/8]` u32 (8 E2M1 nibbles/word, nibble `i`
// at bits `[4i, 4i+4)`), `scales` is `[outDim, inDim/groupSize]` u8
// (E4M3 magnitude byte, `global=1.0`, the per-group scale absorbs the
// whole factor, same convention `nvfp4Qgemv`'s doc comment calls out).
// `groupSize` is 16 (vs `LagunaAttnInt8`'s 32), NVFP4 is a finer-grained
// per-block scale to compensate for the much coarser 4-bit code alphabet
// ({0, 0.5, 1, 1.5, 2, 3, 4, 6} magnitudes + sign, 16 codes total).
//
// Unlike `LagunaAttnInt8`'s affine (scale + zero-point) INT8, NVFP4 here
// is a *symmetric* (scale-only, no bias) quantization: E2M1 already
// straddles zero, so there is no zero-point term to carry and no `xs`
// (activation-sum) correction term in the QMV inner loop.
//
// Threshold: `BUTTER_LAGUNA_ATTN_NVFP4_FROM` (default 32), layers with
// `layerIndex >= fromLayer` build an NVFP4 g16 bank INSTEAD OF the T20
// INT8 g32 bank for their QKV(+gate)/o_proj weights; layers below the
// threshold are unaffected (still INT8, if `LagunaAttnInt8.enabled`).
// Setting the threshold to 99 disables NVFP4 entirely (no layer index in
// any Laguna-family model reaches it), which restores this file's
// contribution to a no-op: every layer falls back to the existing T20
// INT8 (or stock BF16) path. This composes with, and does not replace,
// `BUTTER_LAGUNA_ATTN_INT8`: that flag is the single kill switch for the
// whole attention-side-requant mechanism (INT8 and NVFP4-tail alike),
// since `LagunaText.ensureAttnInt8` gates on `LagunaAttnInt8.enabled`
// before deciding which bank flavor to build per layer.
//
// Every accumulator in both the quantize kernel and the QMV kernels below
// stays in `float` end to end (dequant, multiply, and reduce all in
// `float`; only the final `out[row]` write casts down to `bfloat`). This
// deliberately sidesteps the T23 half-staging overflow class documented
// in benchmarks/laguna-xs/README.md ("scale-deferred staging fix... the
// whole MPP gather kernel family"): that bug came from staging the
// dequantized weight and activation into Metal `half` *before* the
// multiply, where a single product can round to `Inf` at real late-layer
// (SwiGLU-amplified) activation magnitude even though the accumulator
// itself was `float`. Nothing here ever touches `half`, so there is no
// `2^-14`-class exponent-bias fold to carry here: the T23 lesson is
// "don't stage into half at all if you can afford not to," and a
// per-row GEMV (versus MPP's cooperative-tensor tile GEMM) can afford
// not to.
//
// Both the quantize kernel and the batched QMVs are hand-written
// live-compiled Metal (`pso()`, same pattern as `LagunaAttnInt8` and
// `LagunaNvfp4Fuse`); there is no existing Swift/Metal BF16->NVFP4
// encode kernel anywhere in this codebase to reuse (every other NVFP4
// weight in Laguna arrives pre-quantized from the checkpoint; this is
// the first runtime BF16->NVFP4 quantize path). The QMV/QMV-gated
// kernels mirror `LagunaAttnInt8.qmvSrc`/`qmvGatedSrc`'s exact geometry
// (TPG=64, 2 simdgroups x 4 rows/simdgroup, row-bounds-checked so
// `outDim` need not be a multiple of 8) with the INT8 affine dequant
// swapped for NVFP4 E2M1/E4M3 dequant.
//
// Escape: `BUTTER_LAGUNA_ATTN_NVFP4_FROM=99` (or `BUTTER_LAGUNA_ATTN_INT8=0`,
// which disables both flavors) restores the stock T20 INT8-everywhere
// behavior.

import Foundation
import Metal
import IronSwift

enum LagunaAttnNvfp4Tail {
    static let groupSize = 16

    /// Decode-attention QMV V2 (`BUTTER_LAGUNA_DECODE_QMV_V2=1`,
    /// default OFF). One flag, three effects, all decode-only:
    ///
    /// 1. `fromLayer` defaults to 0 instead of 99: EVERY attention layer
    ///    builds the group-16 NVFP4 bank instead of the T20 group-32
    ///    affine INT8 bank, halving attention weight bytes/token
    ///    (0.5625 B/param vs 1.125). T42's +0.83% verdict against this
    ///    was measured in the ~81 tok/s HOST-bound era (pre the launch-flag hoist);
    ///    post-lever-0 decode is GPU-bandwidth-bound in exactly these
    ///    two buckets (`qkv_qmv` + `oproj_gated` ≈ 52% of the token), so
    ///    the byte halving now lands (§3e of the campaign compendium).
    /// 2. The QKV site uses the norm-folded V2 QMV (`normQmvV2` below):
    ///    RMSNorm reduction + QKV(+gate) QMV in ONE dispatch, the NVFP4
    ///    twin of `LagunaAttnInt8.qmvNormFolded` (bit-for-bit the same
    ///    `iron_rms_norm_bf16` virtual-thread emulation).
    /// 3. NVFP4 dequant switches to the V2 fast path (split-nibble E2M1
    ///    decode + E4M3 bit-decode + `2^22` scale fold, see
    ///    `helperV2Src`) — bit-exact vs the V1 LUT/branch decode
    ///    (exhaustively locked by `LagunaAttnNvfp4V2Tests`), just fewer
    ///    integer/select ops per loaded byte. V1's measured effective
    ///    bandwidth at Laguna decode shapes was 334 GB/s (qkv) /
    ///    272 GB/s (gated o_proj) vs the INT8 kernels' ~400-440 GB/s —
    ///    the dequant ALU was eating the byte win; V2 exists to close
    ///    exactly that.
    ///
    /// Explicit `BUTTER_LAGUNA_ATTN_NVFP4_FROM` still wins over the V2
    /// default in `fromLayer` (so the tail-only configuration stays
    /// reachable under V2 for ablation). Live read (not a static let):
    /// tests toggle this per-case via setenv.
    /// DEFAULT ON (Tom's flip 2026-08-08, +32% decode): all-layer NVFP4
    /// g16 attention banks. Output is coherent but NOT byte-identical to
    /// the prior INT8-g32 default — a dequant-association reorder, the
    /// same FP-reorder class the reference engine ships NVFP4-attn
    /// everywhere through 1,344 exact-token checks. Escape `=0` restores
    /// the INT8 path. Live read (tests toggle per-case via setenv).
    static var v2Enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_DECODE_QMV_V2"] != "0"
    }

    /// Default 99 (OFF, w2-integration cycle 3 verdict): layers
    /// `>= fromLayer` build an NVFP4 g16 bank instead of
    /// `LagunaAttnInt8`'s group-32 INT8 bank; `99` means no
    /// Laguna-family layer index reaches it, so every layer stays on
    /// the T20 INT8 path. Authored default was 32 (laguna-xs-2.1-nvfp4
    /// has 40 layers, so 8 late layers would go NVFP4). Validated:
    /// output stays coherent/correct at 32 (a distinct-but-coherent
    /// continuation from the INT8-only baseline, expected per the
    /// 4-bit tail's own late-layer numerics shift), but the fib 80-tok
    /// T=0 decode A/B (3 runs/side, 5s cooldowns, same host window)
    /// only measured +0.83% (32: 80.02/81.84/82.59 tok/s median 81.84
    /// vs 99: 80.41/81.17/81.57 tok/s median 81.17) - short of the 2%
    /// bar needed to justify a numerics-changing default. Kernels and
    /// `BUTTER_LAGUNA_ATTN_NVFP4_FROM=32` (or lower) are kept as an
    /// opt-in for smaller/more-aggressive tail configurations or a
    /// future re-A/B.
    static var fromLayer: Int {
        guard let raw = ProcessInfo.processInfo.environment["BUTTER_LAGUNA_ATTN_NVFP4_FROM"],
            let v = Int(raw)
        else { return v2Enabled ? 0 : 99 }
        return v
    }

    /// `true` when `layerIndex` should use the NVFP4 tail bank rather
    /// than the T20 INT8 bank. `layerIndex < 0` (untagged / test-built
    /// layers) never qualifies, matching `LagunaAttnInt8`'s existing
    /// "diagnostic logging only, no functional dependency" treatment of
    /// a missing layer index, so an un-indexed layer stays on the T20
    /// INT8 path rather than silently defaulting into the NVFP4 tail.
    static func appliesToLayer(_ layerIndex: Int) -> Bool {
        layerIndex >= 0 && layerIndex >= fromLayer
    }

    /// One group-16 NVFP4 side layout: `packed` is `[outDim, inDim/8]`
    /// u32 (8 E2M1 codes per word), `scales` is `[outDim,
    /// inDim/groupSize]` u8 (E4M3 magnitude byte, `global` folded into
    /// the byte itself; call sites always use `global=1.0`, matching
    /// `StackedNvfp4Experts`'s own convention). Dequant: `e2m1(code) *
    /// e4m3(scale)`.
    struct Bank {
        let packed: Tensor
        let scales: Tensor
        let outDim: Int
        let inDim: Int
    }

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

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

    // Shared decode/encode helpers, textually inlined into each
    // self-contained kernel source string below (each `makeLibrary`
    // call compiles an independent translation unit: no cross-string
    // symbol sharing, same discipline `LagunaAttnInt8` and
    // `LagunaNvfp4Fuse` already follow for their own live-compiled
    // sources).
    //
    // `iron_nvfp4_e4m3_scale`: bit-exact vs `OpsNVFP4`'s own
    // `iron_nvfp4_e4m3_scale` (see
    // `Sources/IronSwift/Resources/kernel_overlays/
    // iron_nvfp4_dequant_gemv_expert_indexed_bf16.metal`), same
    // exp4/mant3/sign decomposition, bias 7, subnormal step `2^-9`.
    // `iron_nvfp4_nibble_value`: bit-exact vs the naive (non-bit-twiddle)
    // E2M1 LUT `butter_nvfp4_gather_steel_bm16_bgemm_bf16.metal`'s
    // `nvfp4_decode_nibble` uses: magnitude class `{0, 0.5, 1, 1.5, 2,
    // 3, 4, 6}`, sign in bit 3.
    // `iron_nvfp4_encode_e4m3_mag` / `iron_nvfp4_encode_e2m1_nibble` are
    // new (no prior Swift/Metal NVFP4 *encode* path in this codebase):
    // exhaustive nearest-representable search over all 128 E4M3
    // magnitudes (16 exponents x 8 mantissas, O(16) via per-exponent
    // closed-form best mantissa rather than a 128-way brute force) and
    // all 8 E2M1 magnitude classes respectively; both run once per
    // group / once per element at load time, off the decode hot path.
    private static let helperSrc = """
        inline float iron_nvfp4_e4m3_scale(uchar bits) {
            uint b = uint(bits);
            uint exp4 = (b >> 3) & 15u;
            float mant3 = float(b & 7u);
            float subnormal = mant3 * 0.001953125f;
            float normal = (1.0f + mant3 * 0.125f) * exp2(float(exp4) - 7.0f);
            float mag = select(normal, subnormal, exp4 < 1u);
            bool negative = ((b >> 7) & 1u) != 0u;
            return select(mag, -mag, negative);
        }

        inline float iron_nvfp4_nibble_value(uint nib) {
            constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            float mag = lut[nib & 0x7u];
            bool negative = (nib & 0x8u) != 0u;
            return select(mag, -mag, negative);
        }

        inline uchar iron_nvfp4_encode_e4m3_mag(float m) {
            if (!(m > 0.0f)) { return 0u; }
            float best = INFINITY;
            uint bestByte = 0u;
            {
                float mant = round(m / 0.001953125f);
                mant = clamp(mant, 0.0f, 7.0f);
                float cand = mant * 0.001953125f;
                float diff = fabs(cand - m);
                if (diff < best) { best = diff; bestByte = uint(mant); }
            }
            for (uint e = 1u; e <= 15u; ++e) {
                float scale2 = exp2(float(e) - 7.0f);
                float mant = round((m / scale2 - 1.0f) * 8.0f);
                mant = clamp(mant, 0.0f, 7.0f);
                float cand = (1.0f + mant * 0.125f) * scale2;
                float diff = fabs(cand - m);
                if (diff < best) { best = diff; bestByte = (e << 3) | uint(mant); }
            }
            return uchar(bestByte);
        }

        inline uint iron_nvfp4_encode_e2m1_nibble(float v, float scale) {
            if (!(scale > 0.0f)) { return 0u; }
            constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            float t = v / scale;
            float at = fabs(t);
            uint best = 0u;
            float bestDiff = INFINITY;
            for (uint i = 0u; i < 8u; ++i) {
                float diff = fabs(lut[i] - at);
                if (diff < bestDiff) { bestDiff = diff; best = i; }
            }
            uint nib = best;
            if (t < 0.0f) { nib |= 0x8u; }
            return nib;
        }
        """

    // DECODE_QMV_V2 fast-dequant helpers. Bit-exact vs the V1
    // helpers above (locked exhaustively over all 256 E4M3 bytes x 16
    // E2M1 nibbles by `LagunaAttnNvfp4V2Tests.dequantEquivalence...`),
    // in a fraction of the integer/select ops:
    //
    // * `iron_nvfp4_e4m3_scale22_v2(bits)`: the E4M3 byte's exp4 field
    //   lands directly in a half's exponent field via `raw = (bits +
    //   (bits & 128)) << 7` (the `+ (bits & 128)` carries the sign into
    //   half bit 15; mant3 fills the top three half-mantissa bits). The
    //   reinterpreted half is the true scale x 2^-8 (E4M3 bias 7 vs half
    //   bias 15), and half subnormals reproduce the E4M3 subnormal
    //   ladder (step 2^-9) exactly. The returned value folds BOTH the
    //   2^8 half-bias correction AND the 2^14 E2M1-pattern correction
    //   below into one exact power-of-two multiply: x 4194304 (2^22).
    // * `iron_nvfp4_pack_dot_v2(p, xf)`: split-nibble E2M1 decode. Even
    //   nibbles (`xe`) and odd nibbles (`yo`) of the 8-code word are
    //   spread so each nibble's {sign, exp2, man1} bits land at half
    //   bit positions {15, 11:10, 9} (mask 0x8E00), giving the exact
    //   half bit pattern of the E2M1 value x 2^-14 — the same 8 bit
    //   patterns per word as the V1 LUT, reinterpreted rather than
    //   selected. Accumulation ORDER inside the pack is identical to
    //   V1's `qd` chain (indices 0..7, sequential adds), and the 2^-14
    //   is an exact power-of-two scaling of every partial, so
    //   `scale22 * qd_v2 == scale * qd_v1` bit-for-bit.
    //
    // FTZ caveat (documented, not load-bearing): a product |x * value|
    // below ~2^-112 could flush to zero under the 2^-14 staging where V1
    // would keep it. Real normalized-row activations are >= 1e-4-ish;
    // the equivalence tests cover the realistic range, and the shipped
    // V2 path's stream gate is vs the INT8 baseline (different numerics
    // anyway), not vs V1.
    static let helperV2Src = """
        inline float iron_nvfp4_e4m3_scale22_v2(uchar bits) {
            ushort raw = ushort((uint(bits) + (uint(bits) & 128u)) << 7);
            return float(as_type<half>(raw)) * 4194304.0f;
        }

        inline float iron_nvfp4_pack_dot_v2(uint p, thread const float *xf) {
            uint xe = p & 0x0F0F0F0Fu;
            uint ge = xe | (xe << 3u);
            uint yo = p & 0xF0F0F0F0u;
            uint go = yo | (yo >> 3u);
            uint q0 = (ge << 9u) & 0x8E008E00u;
            uint q1 = (go << 8u) & 0x8E008E00u;
            uint q2 = (ge << 1u) & 0x8E008E00u;
            uint q3 = go & 0x8E008E00u;
            float2 v04 = float2(as_type<half2>(q0));
            float2 v15 = float2(as_type<half2>(q1));
            float2 v26 = float2(as_type<half2>(q2));
            float2 v37 = float2(as_type<half2>(q3));
            float qd = 0.0f;
            qd += v04.x * xf[0];
            qd += v15.x * xf[1];
            qd += v26.x * xf[2];
            qd += v37.x * xf[3];
            qd += v04.y * xf[4];
            qd += v15.y * xf[5];
            qd += v26.y * xf[6];
            qd += v37.y * xf[7];
            return qd;
        }
        """

    // Group-16 NVFP4 quantize: one threadgroup (16 lanes) per group.
    // Amax reduction goes through explicit threadgroup memory + a
    // barrier, not `simd_max`/`simd_broadcast` over a partial
    // simdgroup. `GROUP` (16) is half the native 32-wide simdgroup on
    // every Apple GPU this targets, and threadgroup-memory reduction is
    // unambiguously correct at any dispatch width, unlike relying on
    // simdgroup intrinsics over a sub-simdgroup-width dispatch. Packing
    // re-reads `w[base+idx]` directly (mirrors
    // `LagunaAttnInt8.quantizeSrc`'s own packing lanes) rather than
    // shuffling another lane's register: no inter-lane communication
    // needed beyond the one broadcast scale byte.
    private static let quantizeSrc =
        """
        #include <metal_stdlib>
        using namespace metal;

        \(helperSrc)

        kernel void laguna_attn_quantize_nvfp4_g16_bf16(
            const device bfloat *w [[buffer(0)]],
            device uint *out [[buffer(1)]],
            device uchar *scales [[buffer(2)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_threadgroup]])
        {
            constexpr uint GROUP = 16;
            threadgroup float tgAbs[GROUP];
            threadgroup uchar tgScaleByte;

            uint g = tgid.x;
            uint base = g * GROUP;

            float v = float(w[base + lane]);
            tgAbs[lane] = fabs(v);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lane == 0) {
                float amax = 0.0f;
                for (uint i = 0; i < GROUP; ++i) {
                    amax = max(amax, tgAbs[i]);
                }
                float idealScale = amax > 0.0f ? (amax / 6.0f) : 0.0f;
                uchar scaleByte = iron_nvfp4_encode_e4m3_mag(idealScale);
                tgScaleByte = scaleByte;
                scales[g] = scaleByte;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 2 words/group (16 elements / 8 per word); lanes 0..1 pack.
            if (lane < 2) {
                float scale = iron_nvfp4_e4m3_scale(tgScaleByte);
                uint packed = 0;
                for (uint i = 0; i < 8; ++i) {
                    uint idx = lane * 8 + i;
                    float x = float(w[base + idx]);
                    uint nib = iron_nvfp4_encode_e2m1_nibble(x, scale);
                    packed |= (nib << (4u * i));
                }
                out[g * 2 + lane] = packed;
            }
        }
        """

    // Batched group-16 NVFP4 GEMV over a (possibly row-concatenated)
    // bank. Same geometry as `LagunaAttnInt8.qmvSrc`: TPG=64 (2
    // simdgroups x 32 lanes, full native simdgroup width, no partial-
    // simdgroup concern here, unlike the quantize kernel above), 4
    // output rows per simdgroup => 8 rows/threadgroup; grid =
    // ceil(outDim/8). `packsPerRow` = inDim/8 (8 E2M1 codes/word);
    // `groupsPerRow` = inDim/16, so each 8-wide pack covers exactly
    // half an E4M3 scale group (`g = xBase / GROUP`). Symmetric
    // (scale-only) dequant: no zero-point/`xs` term, unlike the INT8
    // affine QMV this mirrors.
    private static let qmvSrc =
        """
        #include <metal_stdlib>
        using namespace metal;

        \(helperSrc)

        kernel void laguna_attn_qmv_nvfp4_g16_bf16(
            const device uint *w [[buffer(0)]],
            const device uchar *scales [[buffer(1)]],
            const device bfloat *x [[buffer(2)]],
            device bfloat *out [[buffer(3)]],
            constant uint &inDim [[buffer(4)]],
            constant uint &outDim [[buffer(5)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 16;
            constexpr uint ROWS_PER_SG = 4;
            uint packsPerRow = inDim / 8u;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * (2u * ROWS_PER_SG) + sg * ROWS_PER_SG;

            float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (uint block = 0; block < packsPerRow; block += 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 8u;
                    bfloat4 xv0 = *((const device bfloat4 *)(x + xBase));
                    bfloat4 xv1 = *((const device bfloat4 *)(x + xBase + 4u));
                    float xf0 = float(xv0[0]);
                    float xf1 = float(xv0[1]);
                    float xf2 = float(xv0[2]);
                    float xf3 = float(xv0[3]);
                    float xf4 = float(xv1[0]);
                    float xf5 = float(xv1[1]);
                    float xf6 = float(xv1[2]);
                    float xf7 = float(xv1[3]);
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < ROWS_PER_SG; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float scale = iron_nvfp4_e4m3_scale(scales[row * groupsPerRow + g]);
                            float qd = 0.0f;
                            qd += iron_nvfp4_nibble_value(p & 0xFu) * xf0;
                            qd += iron_nvfp4_nibble_value((p >> 4u) & 0xFu) * xf1;
                            qd += iron_nvfp4_nibble_value((p >> 8u) & 0xFu) * xf2;
                            qd += iron_nvfp4_nibble_value((p >> 12u) & 0xFu) * xf3;
                            qd += iron_nvfp4_nibble_value((p >> 16u) & 0xFu) * xf4;
                            qd += iron_nvfp4_nibble_value((p >> 20u) & 0xFu) * xf5;
                            qd += iron_nvfp4_nibble_value((p >> 24u) & 0xFu) * xf6;
                            qd += iron_nvfp4_nibble_value((p >> 28u) & 0xFu) * xf7;
                            acc[r] += scale * qd;
                        }
                    }
                }
            }

            for (uint r = 0; r < ROWS_PER_SG; ++r) {
                float total = simd_sum(acc[r]);
                if (lane == 0) {
                    uint row = rowBase + r;
                    if (row < outDim) {
                        out[row] = bfloat(total);
                    }
                }
            }
        }
        """

    // Gate-folded variant of `qmvSrc` for the o_proj bank, mirroring
    // `LagunaAttnInt8.qmvGatedSrc` exactly (same softplus formula, same
    // `packsPerHead`-based head lookup) with the dequant swapped for
    // NVFP4. `packsPerHead` here is `headDim/8` (vs INT8's `headDim/4`)
    // since a pack is 8-wide, not 4-wide.
    private static let qmvGatedSrc =
        """
        #include <metal_stdlib>
        using namespace metal;

        \(helperSrc)

        kernel void laguna_attn_qmv_nvfp4_g16_gated_bf16(
            const device uint *w [[buffer(0)]],
            const device uchar *scales [[buffer(1)]],
            const device bfloat *x [[buffer(2)]],
            const device bfloat *gate_logits [[buffer(3)]],
            device bfloat *out [[buffer(4)]],
            constant uint &inDim [[buffer(5)]],
            constant uint &outDim [[buffer(6)]],
            constant uint &packsPerHead [[buffer(7)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 16;
            constexpr uint ROWS_PER_SG = 4;
            uint packsPerRow = inDim / 8u;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * (2u * ROWS_PER_SG) + sg * ROWS_PER_SG;

            float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (uint block = 0; block < packsPerRow; block += 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 8u;
                    uint head = packIdx / packsPerHead;
                    float gl = float(gate_logits[head]);
                    bool pos = gl > 0.0f;
                    float ax = pos ? gl : (0.0f - gl);
                    float gate = (pos ? gl : 0.0f) + log(1.0f + exp(0.0f - ax));

                    bfloat4 xv0 = *((const device bfloat4 *)(x + xBase));
                    bfloat4 xv1 = *((const device bfloat4 *)(x + xBase + 4u));
                    float xf0 = float(xv0[0]) * gate;
                    float xf1 = float(xv0[1]) * gate;
                    float xf2 = float(xv0[2]) * gate;
                    float xf3 = float(xv0[3]) * gate;
                    float xf4 = float(xv1[0]) * gate;
                    float xf5 = float(xv1[1]) * gate;
                    float xf6 = float(xv1[2]) * gate;
                    float xf7 = float(xv1[3]) * gate;
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < ROWS_PER_SG; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float scale = iron_nvfp4_e4m3_scale(scales[row * groupsPerRow + g]);
                            float qd = 0.0f;
                            qd += iron_nvfp4_nibble_value(p & 0xFu) * xf0;
                            qd += iron_nvfp4_nibble_value((p >> 4u) & 0xFu) * xf1;
                            qd += iron_nvfp4_nibble_value((p >> 8u) & 0xFu) * xf2;
                            qd += iron_nvfp4_nibble_value((p >> 12u) & 0xFu) * xf3;
                            qd += iron_nvfp4_nibble_value((p >> 16u) & 0xFu) * xf4;
                            qd += iron_nvfp4_nibble_value((p >> 20u) & 0xFu) * xf5;
                            qd += iron_nvfp4_nibble_value((p >> 24u) & 0xFu) * xf6;
                            qd += iron_nvfp4_nibble_value((p >> 28u) & 0xFu) * xf7;
                            acc[r] += scale * qd;
                        }
                    }
                }
            }

            for (uint r = 0; r < ROWS_PER_SG; ++r) {
                float total = simd_sum(acc[r]);
                if (lane == 0) {
                    uint row = rowBase + r;
                    if (row < outDim) {
                        out[row] = bfloat(total);
                    }
                }
            }
        }
        """

    // Split-K variant of the gated o_proj NVFP4 QMV. T39 finding (see
    // benchmarks/laguna-xs/README.md section 14, "The actionable lever:
    // NVFP4 gated o_proj kernel is missing split-K"): with the NVFP4
    // tail ON end to end (`BUTTER_LAGUNA_ATTN_NVFP4_FROM=0`), the gated
    // o_proj kernel measured 2443us GPU-busy vs INT8's 1992us (+23%)
    // purely because `qmvGatedSrc` above never received the K-parallel
    // restructure `LagunaAttnInt8.qmvGatedSplitK` already has, while the
    // plain NVFP4 `qkv_qmv` kernel (no split-K needed, already ahead of
    // INT8 at -35%) shows the byte-width win NVFP4 should deliver
    // everywhere once this gap closes. This kernel ports that same
    // `NR0=2`/`NSG=4` two-stage-reduce geometry from
    // `LagunaAttnInt8.qmvGatedSplitKSrc` (mirror it 1:1 for the
    // restructure) with the affine INT8 dequant swapped for this file's
    // symmetric NVFP4 g16 dequant (no zero-point, no `xs` correction
    // term) and the pack width widened from 4 codes/word (INT8 u32) to
    // 8 codes/word (NVFP4 u32), so `packIdx` strides `8*NSG` elements
    // per simdgroup burst instead of `4*NSG`, and each simdgroup loads
    // two `bfloat4`s (`xv0`/`xv1`) per K-chunk instead of one.
    //
    // ROUNDING FIX (same trap as INT8's `qmvGatedSplitK`, commit
    // `1614148f`): the softplus GATE VALUE is rounded through bf16
    // before use (`gate = float(bfloat((pos ? gl : 0.0f) + log(1.0f +
    // exp(0.0f - ax))))`), matching what a separate
    // gate-product-softplus dispatch would materialize into a real bf16
    // tensor before a downstream QMV reads it back. Skipping this round
    // and carrying the softplus result in float32 end to end reproduces
    // the INT8 kernel's exact failure mode (a systematic ~0.5
    // max-abs-diff vs the reference, not a near-ULP reduction-order
    // gap) because it is a genuine value substitution, not just a
    // different summation order. The gated activation PRODUCT
    // (`xf0..xf7`, gate times each of the 8 activation lanes) is
    // likewise rounded through bf16 per lane, mirroring INT8's second
    // rounding point for the same reason (the fused kernel stands in
    // for a materialized bf16 intermediate).
    private static let qmvGatedSplitKSrc =
        """
        #include <metal_stdlib>
        using namespace metal;

        \(helperSrc)

        kernel void laguna_attn_qmv_nvfp4_g16_gated_splitk_bf16(
            const device uint *w [[buffer(0)]],
            const device uchar *scales [[buffer(1)]],
            const device bfloat *x [[buffer(2)]],
            const device bfloat *gate_logits [[buffer(3)]],
            device bfloat *out [[buffer(4)]],
            constant uint &inDim [[buffer(5)]],
            constant uint &outDim [[buffer(6)]],
            constant uint &packsPerHead [[buffer(7)]],
            threadgroup float *shmem [[threadgroup(0)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sgitg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 16;
            constexpr uint NR0 = 2;
            constexpr uint NSG = 4;
            uint packsPerRow = inDim / 8u;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * NR0;

            float acc[NR0] = {0.0f, 0.0f};

            for (uint block = sgitg * 32u; block < packsPerRow; block += NSG * 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 8u;
                    uint head = packIdx / packsPerHead;
                    float gl = float(gate_logits[head]);
                    bool pos = gl > 0.0f;
                    float ax = pos ? gl : (0.0f - gl);
                    // ROUNDING POINT 1 (softplus gate value -> bf16): see
                    // the header comment above this kernel.
                    float gate = float(bfloat((pos ? gl : 0.0f) + log(1.0f + exp(0.0f - ax))));

                    bfloat4 xv0 = *((const device bfloat4 *)(x + xBase));
                    bfloat4 xv1 = *((const device bfloat4 *)(x + xBase + 4u));
                    // ROUNDING POINT 2 (gate*activation product -> bf16):
                    // see the header comment above this kernel.
                    float xf0 = float(bfloat(float(xv0[0]) * gate));
                    float xf1 = float(bfloat(float(xv0[1]) * gate));
                    float xf2 = float(bfloat(float(xv0[2]) * gate));
                    float xf3 = float(bfloat(float(xv0[3]) * gate));
                    float xf4 = float(bfloat(float(xv1[0]) * gate));
                    float xf5 = float(bfloat(float(xv1[1]) * gate));
                    float xf6 = float(bfloat(float(xv1[2]) * gate));
                    float xf7 = float(bfloat(float(xv1[3]) * gate));
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < NR0; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float scale = iron_nvfp4_e4m3_scale(scales[row * groupsPerRow + g]);
                            float qd = 0.0f;
                            qd += iron_nvfp4_nibble_value(p & 0xFu) * xf0;
                            qd += iron_nvfp4_nibble_value((p >> 4u) & 0xFu) * xf1;
                            qd += iron_nvfp4_nibble_value((p >> 8u) & 0xFu) * xf2;
                            qd += iron_nvfp4_nibble_value((p >> 12u) & 0xFu) * xf3;
                            qd += iron_nvfp4_nibble_value((p >> 16u) & 0xFu) * xf4;
                            qd += iron_nvfp4_nibble_value((p >> 20u) & 0xFu) * xf5;
                            qd += iron_nvfp4_nibble_value((p >> 24u) & 0xFu) * xf6;
                            qd += iron_nvfp4_nibble_value((p >> 28u) & 0xFu) * xf7;
                            acc[r] += scale * qd;
                        }
                    }
                }
            }

            for (uint r = 0; r < NR0; ++r) {
                float partial = simd_sum(acc[r]);
                if (lane == 0) {
                    shmem[32u * r + sgitg] = partial;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                for (uint r = 0; r < NR0; ++r) {
                    float v = lane < NSG ? shmem[32u * r + lane] : 0.0f;
                    float total = simd_sum(v);
                    if (lane == 0) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            out[row] = bfloat(total);
                        }
                    }
                }
            }
        }
        """

    // Split-K variant of `qmvSrc` (see `qmvGatedSplitKSrc` above for the
    // full T39 rationale + geometry writeup; this is the ungated
    // sibling, same `NR0=2`/`NSG=4` two-stage reduce, same 8-wide NVFP4
    // pack stride, mirroring `LagunaAttnInt8.qmvSplitKSrc` with the
    // symmetric NVFP4 dequant swapped in for the affine INT8 one (no
    // bias/`xs` term). Measured mainly to close the `oproj_gated`
    // regression above; ported here too since `qkv_qmv` already wins
    // and split-K should only help further, and INT8 ships both
    // variants behind the one flag for the same reason.
    private static let qmvSplitKSrc =
        """
        #include <metal_stdlib>
        using namespace metal;

        \(helperSrc)

        kernel void laguna_attn_qmv_nvfp4_g16_splitk_bf16(
            const device uint *w [[buffer(0)]],
            const device uchar *scales [[buffer(1)]],
            const device bfloat *x [[buffer(2)]],
            device bfloat *out [[buffer(3)]],
            constant uint &inDim [[buffer(4)]],
            constant uint &outDim [[buffer(5)]],
            threadgroup float *shmem [[threadgroup(0)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sgitg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 16;
            constexpr uint NR0 = 2;
            constexpr uint NSG = 4;
            uint packsPerRow = inDim / 8u;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * NR0;

            float acc[NR0] = {0.0f, 0.0f};

            for (uint block = sgitg * 32u; block < packsPerRow; block += NSG * 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 8u;
                    bfloat4 xv0 = *((const device bfloat4 *)(x + xBase));
                    bfloat4 xv1 = *((const device bfloat4 *)(x + xBase + 4u));
                    float xf0 = float(xv0[0]);
                    float xf1 = float(xv0[1]);
                    float xf2 = float(xv0[2]);
                    float xf3 = float(xv0[3]);
                    float xf4 = float(xv1[0]);
                    float xf5 = float(xv1[1]);
                    float xf6 = float(xv1[2]);
                    float xf7 = float(xv1[3]);
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < NR0; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float scale = iron_nvfp4_e4m3_scale(scales[row * groupsPerRow + g]);
                            float qd = 0.0f;
                            qd += iron_nvfp4_nibble_value(p & 0xFu) * xf0;
                            qd += iron_nvfp4_nibble_value((p >> 4u) & 0xFu) * xf1;
                            qd += iron_nvfp4_nibble_value((p >> 8u) & 0xFu) * xf2;
                            qd += iron_nvfp4_nibble_value((p >> 12u) & 0xFu) * xf3;
                            qd += iron_nvfp4_nibble_value((p >> 16u) & 0xFu) * xf4;
                            qd += iron_nvfp4_nibble_value((p >> 20u) & 0xFu) * xf5;
                            qd += iron_nvfp4_nibble_value((p >> 24u) & 0xFu) * xf6;
                            qd += iron_nvfp4_nibble_value((p >> 28u) & 0xFu) * xf7;
                            acc[r] += scale * qd;
                        }
                    }
                }
            }

            for (uint r = 0; r < NR0; ++r) {
                float partial = simd_sum(acc[r]);
                if (lane == 0) {
                    shmem[32u * r + sgitg] = partial;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                for (uint r = 0; r < NR0; ++r) {
                    float v = lane < NSG ? shmem[32u * r + lane] : 0.0f;
                    float total = simd_sum(v);
                    if (lane == 0) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            out[row] = bfloat(total);
                        }
                    }
                }
            }
        }
        """

    /// V1 helper source exposed for `LagunaAttnNvfp4V2Tests`'s exhaustive
    /// dequant-equivalence probe kernel (V1 vs V2 over all 256 E4M3 bytes
    /// x 16 E2M1 nibbles). Not read by any production dispatch.
    static var helperSrcForTesting: String { helperSrc }

    // DECODE_QMV_V2 kernel 1: norm-folded QKV(+gate) NVFP4 QMV.
    // The NVFP4 twin of `LagunaAttnInt8.normQmvSrc` (T-series
    // BUTTER_LAGUNA_QKV_NORMFOLD): folds the block's input RMSNorm into
    // the bank's own QMV so the decode QKV head is ONE dispatch instead
    // of two (separate `Ops.rmsNorm` + `qmv`), and the normalized row
    // never round-trips through device memory.
    //
    // The RMS reduction section is copied VERBATIM from `normQmvSrc`
    // (same 64-real-thread emulation of `iron_rms_norm_bf16`'s
    // `inDim/4`-thread virtual threadgroup, same phase-1/2/3 combine,
    // same single `bfloat(x * rms * w)` rounding point) — see that
    // kernel's header for the full bit-exactness argument; the K loop
    // below is `qmvSrc`'s geometry (2 simdgroups x 4 rows, full-K walk
    // per simdgroup, so the per-row accumulation ORDER matches the
    // plain non-split-K QMV exactly) with the affine INT8 dequant
    // swapped for the V2 fast NVFP4 dequant. Bit-exactness vs the
    // separate `Ops.rmsNorm` + plain-`qmv` chain is asserted by
    // `LagunaAttnNvfp4V2Tests.normFoldMatchesSeparateChain`.
    private static let normQmvV2Src = """
        #include <metal_stdlib>
        using namespace metal;

        \(helperV2Src)

        kernel void laguna_attn_norm_qmv_nvfp4_g16_v2_bf16(
            const device uint *w [[buffer(0)]],
            const device uchar *scales [[buffer(1)]],
            const device bfloat *residual [[buffer(2)]],
            const device bfloat *norm_weight [[buffer(3)]],
            device bfloat *out [[buffer(4)]],
            constant uint &inDim [[buffer(5)]],
            constant uint &outDim [[buffer(6)]],
            constant float &eps [[buffer(7)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 16;
            constexpr uint ROWS_PER_SG = 4;
            constexpr uint REAL_SIMDGROUPS = 2;
            uint packsPerRow = inDim / 8u;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * (2u * ROWS_PER_SG) + sg * ROWS_PER_SG;

            // --- rms reduction: verbatim from laguna_attn_norm_qmv_int8_g32_bf16
            // (64 real lanes emulate the inDim/4 virtual lanes
            // `iron_rms_norm_bf16` would dispatch). ---
            threadgroup float partial_sg[32];
            uint nSimdVirtual = inDim / 128u;
            uint virtualPerThread = nSimdVirtual / REAL_SIMDGROUPS;

            for (uint j = 0; j < virtualPerThread; ++j) {
                uint vsg = sg + REAL_SIMDGROUPS * j;
                uint base = (vsg * 32u + lane) * 4u;
                float x0 = float(residual[base]);
                float x1 = float(residual[base + 1u]);
                float x2 = float(residual[base + 2u]);
                float x3 = float(residual[base + 3u]);
                float partial = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
                float sv = simd_sum(partial);
                if (lane == 0) {
                    partial_sg[vsg] = sv;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float tg_ssq[1];
            if (sg == 0) {
                float wv = lane < nSimdVirtual ? partial_sg[lane] : 0.0f;
                float total = simd_sum(wv);
                if (lane == 0) {
                    tg_ssq[0] = total;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float rms = rsqrt(tg_ssq[0] / float(inDim) + eps);

            // --- NVFP4 g16 QMV over the inline-recomputed normalized row
            // (`bfloat(x * rms * w)`, ONE bf16 round, matching
            // `iron_rms_norm_bf16`'s own rounding). ---
            float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (uint block = 0; block < packsPerRow; block += 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 8u;
                    float xf[8];
                    for (uint i = 0; i < 8u; ++i) {
                        xf[i] = float(bfloat(
                            float(residual[xBase + i]) * rms *
                            float(norm_weight[xBase + i])));
                    }
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < ROWS_PER_SG; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float scale22 = iron_nvfp4_e4m3_scale22_v2(
                                scales[row * groupsPerRow + g]);
                            acc[r] += scale22 * iron_nvfp4_pack_dot_v2(p, xf);
                        }
                    }
                }
            }

            for (uint r = 0; r < ROWS_PER_SG; ++r) {
                float total = simd_sum(acc[r]);
                if (lane == 0) {
                    uint row = rowBase + r;
                    if (row < outDim) {
                        out[row] = bfloat(total);
                    }
                }
            }
        }
        """

    // DECODE_QMV_V2 kernel 2: gated o_proj split-K NVFP4 QMV, V2.
    // Two changes vs `qmvGatedSplitKSrc` (V1), both bit-exact:
    // 1. The per-head softplus gate is computed ONCE per head into a
    //    threadgroup gate table (the reference stack's o_proj kernel
    //    does exactly this) instead of being recomputed by every lane
    //    on every K block — same formula, same ROUNDING POINT 1 bf16
    //    round, so the VALUE each lane reads is identical; only the
    //    redundant exp/log recomputation goes away.
    // 2. V2 fast dequant (`helperV2Src`) replaces the LUT/branch V1
    //    dequant; accumulation order unchanged.
    // Geometry (NR0=2 rows/TG, NSG=4 simdgroups jointly striding K,
    // two-stage reduce) is byte-for-byte V1's. Bit-exactness vs V1 is
    // asserted by `LagunaAttnNvfp4V2Tests.gatedV2MatchesGatedV1`.
    // `gate_table` is sized for the largest head count in the Laguna
    // family (64); the Swift wrapper declines past that.
    private static let qmvGatedSplitKV2Src = """
        #include <metal_stdlib>
        using namespace metal;

        \(helperV2Src)

        kernel void laguna_attn_qmv_nvfp4_g16_gated_splitk_v2_bf16(
            const device uint *w [[buffer(0)]],
            const device uchar *scales [[buffer(1)]],
            const device bfloat *x [[buffer(2)]],
            const device bfloat *gate_logits [[buffer(3)]],
            device bfloat *out [[buffer(4)]],
            constant uint &inDim [[buffer(5)]],
            constant uint &outDim [[buffer(6)]],
            constant uint &packsPerHead [[buffer(7)]],
            constant uint &nHeads [[buffer(8)]],
            threadgroup float *shmem [[threadgroup(0)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sgitg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 16;
            constexpr uint NR0 = 2;
            constexpr uint NSG = 4;
            uint packsPerRow = inDim / 8u;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * NR0;

            threadgroup float gate_table[64];
            uint lid = sgitg * 32u + lane;
            if (lid < nHeads) {
                float gl = float(gate_logits[lid]);
                bool pos = gl > 0.0f;
                float ax = pos ? gl : (0.0f - gl);
                // ROUNDING POINT 1 (softplus gate value -> bf16), the
                // exact expression V1 evaluates per lane per K block.
                gate_table[lid] =
                    float(bfloat((pos ? gl : 0.0f) + log(1.0f + exp(0.0f - ax))));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float acc[NR0] = {0.0f, 0.0f};

            for (uint block = sgitg * 32u; block < packsPerRow; block += NSG * 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 8u;
                    uint head = packIdx / packsPerHead;
                    float gate = gate_table[head];

                    bfloat4 xv0 = *((const device bfloat4 *)(x + xBase));
                    bfloat4 xv1 = *((const device bfloat4 *)(x + xBase + 4u));
                    float xf[8];
                    // ROUNDING POINT 2 (gate*activation product -> bf16),
                    // same as V1.
                    xf[0] = float(bfloat(float(xv0[0]) * gate));
                    xf[1] = float(bfloat(float(xv0[1]) * gate));
                    xf[2] = float(bfloat(float(xv0[2]) * gate));
                    xf[3] = float(bfloat(float(xv0[3]) * gate));
                    xf[4] = float(bfloat(float(xv1[0]) * gate));
                    xf[5] = float(bfloat(float(xv1[1]) * gate));
                    xf[6] = float(bfloat(float(xv1[2]) * gate));
                    xf[7] = float(bfloat(float(xv1[3]) * gate));
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < NR0; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float scale22 = iron_nvfp4_e4m3_scale22_v2(
                                scales[row * groupsPerRow + g]);
                            acc[r] += scale22 * iron_nvfp4_pack_dot_v2(p, xf);
                        }
                    }
                }
            }

            for (uint r = 0; r < NR0; ++r) {
                float partial = simd_sum(acc[r]);
                if (lane == 0) {
                    shmem[32u * r + sgitg] = partial;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                for (uint r = 0; r < NR0; ++r) {
                    float v = lane < NSG ? shmem[32u * r + lane] : 0.0f;
                    float total = simd_sum(v);
                    if (lane == 0) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            out[row] = bfloat(total);
                        }
                    }
                }
            }
        }
        """

    // DECODE_QMV_V2 kernel 1b: split-K norm-folded QKV(+gate) NVFP4
    // QMV. Same fold as `normQmvV2Src` but with `qmvSplitKSrc`'s NR0=2 /
    // NSG=4 K-parallel geometry (128 threads jointly striding K for a
    // 2-row pair, two-stage reduce): measured in-situ, the 8-rows/TG
    // norm-fold geometry left the qkv bucket at ~348 GB/s effective
    // (1296 us/tok) while the split-K gated o_proj hit ~389 GB/s — the
    // serial full-K walk per simdgroup is latency-bound at Laguna's
    // K=2048. The RMS emulation section generalizes `normQmvV2Src`'s to
    // REAL_SIMDGROUPS=4: virtual simdgroup `vsg = sg + 4*j` contributes
    // the identical per-vsg `simd_sum` value into the identical
    // `partial_sg[vsg]` slot, and the phase-3 combine (simdgroup 0
    // summing the 32-slot array in lane order) is unchanged, so the
    // computed `rms` is bit-identical to `iron_rms_norm_bf16`'s (and to
    // `normQmvV2Src`'s). The K loop's accumulation order matches
    // `qmvSplitKSrc` (V1 split-K) exactly, so the bit-exact reference
    // chain is `Ops.rmsNorm` + `qmvSplitK`
    // (`LagunaAttnNvfp4V2Tests.normFoldSplitKMatchesSeparateChain`).
    private static let normQmvSplitKV2Src = """
        #include <metal_stdlib>
        using namespace metal;

        \(helperV2Src)

        kernel void laguna_attn_norm_qmv_nvfp4_g16_splitk_v2_bf16(
            const device uint *w [[buffer(0)]],
            const device uchar *scales [[buffer(1)]],
            const device bfloat *residual [[buffer(2)]],
            const device bfloat *norm_weight [[buffer(3)]],
            device bfloat *out [[buffer(4)]],
            constant uint &inDim [[buffer(5)]],
            constant uint &outDim [[buffer(6)]],
            constant float &eps [[buffer(7)]],
            threadgroup float *shmem [[threadgroup(0)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sgitg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 16;
            constexpr uint NR0 = 2;
            constexpr uint NSG = 4;
            uint packsPerRow = inDim / 8u;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * NR0;

            // --- rms reduction: `normQmvV2Src`'s emulation at
            // REAL_SIMDGROUPS=4 (see header). ---
            threadgroup float partial_sg[32];
            uint nSimdVirtual = inDim / 128u;
            uint virtualPerThread = nSimdVirtual / NSG;

            for (uint j = 0; j < virtualPerThread; ++j) {
                uint vsg = sgitg + NSG * j;
                uint base = (vsg * 32u + lane) * 4u;
                float x0 = float(residual[base]);
                float x1 = float(residual[base + 1u]);
                float x2 = float(residual[base + 2u]);
                float x3 = float(residual[base + 3u]);
                float partial = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
                float sv = simd_sum(partial);
                if (lane == 0) {
                    partial_sg[vsg] = sv;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float tg_ssq[1];
            if (sgitg == 0) {
                float wv = lane < nSimdVirtual ? partial_sg[lane] : 0.0f;
                float total = simd_sum(wv);
                if (lane == 0) {
                    tg_ssq[0] = total;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float rms = rsqrt(tg_ssq[0] / float(inDim) + eps);

            // --- split-K NVFP4 QMV over the inline-recomputed normalized
            // row (`bfloat(x * rms * w)`, ONE bf16 round). ---
            float acc[NR0] = {0.0f, 0.0f};

            for (uint block = sgitg * 32u; block < packsPerRow; block += NSG * 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 8u;
                    float xf[8];
                    for (uint i = 0; i < 8u; ++i) {
                        xf[i] = float(bfloat(
                            float(residual[xBase + i]) * rms *
                            float(norm_weight[xBase + i])));
                    }
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < NR0; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float scale22 = iron_nvfp4_e4m3_scale22_v2(
                                scales[row * groupsPerRow + g]);
                            acc[r] += scale22 * iron_nvfp4_pack_dot_v2(p, xf);
                        }
                    }
                }
            }

            for (uint r = 0; r < NR0; ++r) {
                float partial = simd_sum(acc[r]);
                if (lane == 0) {
                    shmem[32u * r + sgitg] = partial;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                for (uint r = 0; r < NR0; ++r) {
                    float v = lane < NSG ? shmem[32u * r + lane] : 0.0f;
                    float total = simd_sum(v);
                    if (lane == 0) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            out[row] = bfloat(total);
                        }
                    }
                }
            }
        }
        """

    /// Quantize a `[outDim, inDim]` BF16 weight to a group-16 NVFP4 side
    /// layout. Declines (returns `nil`) unless `inDim` is a multiple of
    /// 16 and `weight` is 2D BF16, matching the group-16 alignment the
    /// packed layout requires.
    static func quantizeG16(
        weight: Tensor, device: Device, cmd: MTLCommandBuffer
    ) -> Bank? {
        guard weight.dtype == .bf16, weight.shape.count == 2,
            weight.shape[1].isMultiple(of: groupSize)
        else { return nil }
        let outDim = weight.shape[0]
        let inDim = weight.shape[1]
        let packsPerRow = inDim / 8
        let groupsPerRow = inDim / groupSize
        let packed = Tensor.empty(shape: [outDim, packsPerRow], dtype: .u32, device: device)
        let scales = Tensor.empty(shape: [outDim, groupsPerRow], dtype: .u8, device: device)
        let nGroups = outDim * groupsPerRow
        let p = pso(
            "laguna_attn_quantize_nvfp4_g16_bf16", source: quantizeSrc,
            device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(weight.buffer, offset: weight.offset, index: 0)
            enc.setBuffer(packed.buffer, offset: packed.offset, index: 1)
            enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
            enc.dispatchThreadgroups(
                MTLSize(width: nGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: groupSize, height: 1, depth: 1))
            enc.endEncoding()
        }
        return Bank(packed: packed, scales: scales, outDim: outDim, inDim: inDim)
    }

    /// Row-concatenate banks that share `inDim` (e.g. Q/K/V + per-head
    /// gate) into one bank a single QMV can read. Pure CPU memcpy; the
    /// GPU quantize dispatches that produced `banks` must already have
    /// completed (caller commits + waits the setup command buffer
    /// first), same contract as `LagunaAttnInt8.concat`.
    static func concat(_ banks: [Bank], device: Device) -> Bank? {
        guard let first = banks.first,
            banks.allSatisfy({ $0.inDim == first.inDim })
        else { return nil }
        let totalOut = banks.reduce(0) { $0 + $1.outDim }
        let packsPerRow = first.inDim / 8
        let groupsPerRow = first.inDim / groupSize
        let packedBuf = device.makeBuffer(length: totalOut * packsPerRow * 4)
        let scalesBuf = device.makeBuffer(length: totalOut * groupsPerRow)
        var packedOff = 0
        var scalesOff = 0
        for b in banks {
            let packedBytes = b.outDim * packsPerRow * 4
            let scaleBytes = b.outDim * groupsPerRow
            memcpy(
                packedBuf.contents().advanced(by: packedOff),
                b.packed.buffer.contents().advanced(by: b.packed.offset), packedBytes)
            memcpy(
                scalesBuf.contents().advanced(by: scalesOff),
                b.scales.buffer.contents().advanced(by: b.scales.offset), scaleBytes)
            packedOff += packedBytes
            scalesOff += scaleBytes
        }
        // Row-concatenated bank replaces the T20 INT8 (or stock BF16)
        // Q/K/V(+gate) / o_proj weights for every decode token from
        // here on for this layer; pin it in the residency set like any
        // other long-lived weight buffer.
        device.markWeightsResident([packedBuf, scalesBuf])
        return Bank(
            packed: Tensor(
                buffer: packedBuf, offset: 0, shape: [totalOut, packsPerRow], dtype: .u32),
            scales: Tensor(
                buffer: scalesBuf, offset: 0, shape: [totalOut, groupsPerRow], dtype: .u8),
            outDim: totalOut, inDim: first.inDim)
    }

    private static func gatedShapesValid(
        input: Tensor, gateLogits: Tensor, bank: Bank, heads: Int, headDim: Int, out: Tensor
    ) -> Bool {
        input.dtype == .bf16 && gateLogits.dtype == .bf16 && out.dtype == .bf16
            && input.elementCount == bank.inDim
            && gateLogits.elementCount == heads
            && out.elementCount == bank.outDim
            && bank.inDim == heads * headDim
            && headDim % 8 == 0
    }

    /// Split-K variant of `qmvGated` (see `qmvGatedSplitKSrc` above for
    /// the T39 rationale + design, and
    /// `LagunaAttnInt8.qmvGatedSplitK`/`splitKEnabled` for the geometry
    /// this mirrors). Same shape/dtype contract as the non-split
    /// `qmvGated`, plus declines (returns `false`, `out` left untouched)
    /// unless `bank.outDim` is a multiple of 2 (fixed `NR0 = 2` tiling,
    /// no partial-pair tail).
    static func qmvGatedSplitK(
        input: Tensor, gateLogits: Tensor, bank: Bank,
        heads: Int, headDim: Int,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard LagunaAttnInt8.gatedEnabled,
            gatedShapesValid(
                input: input, gateLogits: gateLogits, bank: bank, heads: heads, headDim: headDim,
                out: out),
            bank.outDim.isMultiple(of: 2)
        else { return false }

        // V2 twin (threadgroup gate table + fast dequant, bit-exact
        // vs V1 — see `qmvGatedSplitKV2Src`). The V2 kernel's gate table
        // is fixed at 64 slots, the largest Laguna-family head count;
        // anything wider falls through to V1.
        let useV2 = v2Enabled && heads <= 64

        let packsPerHead = headDim / 8
        let p = useV2
            ? pso(
                "laguna_attn_qmv_nvfp4_g16_gated_splitk_v2_bf16",
                source: qmvGatedSplitKV2Src, device: device.mtlDevice)
            : pso(
                "laguna_attn_qmv_nvfp4_g16_gated_splitk_bf16", source: qmvGatedSplitKSrc,
                device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
        enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
        enc.setBuffer(input.buffer, offset: input.offset, index: 2)
        enc.setBuffer(gateLogits.buffer, offset: gateLogits.offset, index: 3)
        enc.setBuffer(out.buffer, offset: out.offset, index: 4)
        var inDimV = UInt32(bank.inDim)
        var outDimV = UInt32(bank.outDim)
        var packsPerHeadV = UInt32(packsPerHead)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&outDimV, length: 4, index: 6)
        enc.setBytes(&packsPerHeadV, length: 4, index: 7)
        if useV2 {
            var nHeadsV = UInt32(heads)
            enc.setBytes(&nHeadsV, length: 4, index: 8)
        }
        enc.setThreadgroupMemoryLength(2 * 32 * MemoryLayout<Float>.stride, index: 0)
        let tiles = bank.outDim / 2
        enc.dispatchThreadgroups(
            MTLSize(width: tiles, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

    /// Fused gate-product-softplus + o_proj NVFP4 QMV in one dispatch:
    /// the NVFP4 sibling of `LagunaAttnInt8.qmvGated`. Gated behind the same
    /// `LagunaAttnInt8.gatedEnabled` toggle (`BUTTER_LAGUNA_INT8_OPROJ_GATED`)
    /// since it is the same fusion decision regardless of which
    /// quantization scheme the bank underneath uses. Declines (returns
    /// `false`, `into` left untouched) unless every shape/dtype
    /// precondition holds; caller falls back to the separate-dispatch
    /// chain feeding stock `qmv`. Tries `qmvGatedSplitK` first when
    /// `LagunaAttnInt8.splitKEnabled` (same `BUTTER_LAGUNA_QMV_SPLITK`
    /// flag the INT8 bank uses, one kill switch for split-K across both
    /// quantization schemes), falling through to this file's original
    /// (pre-T39) geometry on any decline.
    static func qmvGated(
        input: Tensor, gateLogits: Tensor, bank: Bank,
        heads: Int, headDim: Int,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard LagunaAttnInt8.gatedEnabled else { return false }
        if LagunaAttnInt8.splitKEnabled,
            qmvGatedSplitK(
                input: input, gateLogits: gateLogits, bank: bank, heads: heads, headDim: headDim,
                on: cmd, device: device, into: out)
        {
            return true
        }
        guard gatedShapesValid(
            input: input, gateLogits: gateLogits, bank: bank, heads: heads, headDim: headDim,
            out: out)
        else { return false }

        let packsPerHead = headDim / 8
        let p = pso(
            "laguna_attn_qmv_nvfp4_g16_gated_bf16", source: qmvGatedSrc,
            device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
        enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
        enc.setBuffer(input.buffer, offset: input.offset, index: 2)
        enc.setBuffer(gateLogits.buffer, offset: gateLogits.offset, index: 3)
        enc.setBuffer(out.buffer, offset: out.offset, index: 4)
        var inDimV = UInt32(bank.inDim)
        var outDimV = UInt32(bank.outDim)
        var packsPerHeadV = UInt32(packsPerHead)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&outDimV, length: 4, index: 6)
        enc.setBytes(&packsPerHeadV, length: 4, index: 7)
        let rowsPerTG = 8
        let tiles = (bank.outDim + rowsPerTG - 1) / rowsPerTG
        enc.dispatchThreadgroups(
            MTLSize(width: tiles, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

    /// Split-K variant of `qmv` (see `qmvSplitKSrc` above for the T39
    /// rationale + design, and
    /// `LagunaAttnInt8.qmvSplitK`/`splitKEnabled` for the geometry this
    /// mirrors). Same shape/dtype contract as `qmv`, plus declines
    /// (returns `false`, `out` left untouched) unless `bank.outDim` is a
    /// multiple of 2 (fixed `NR0 = 2` tiling, no partial-pair tail).
    static func qmvSplitK(
        input: Tensor, bank: Bank, on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard input.dtype == .bf16, out.dtype == .bf16,
            input.elementCount == bank.inDim,
            out.elementCount == bank.outDim,
            bank.outDim.isMultiple(of: 2)
        else { return false }

        let p = pso(
            "laguna_attn_qmv_nvfp4_g16_splitk_bf16", source: qmvSplitKSrc,
            device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
        enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
        enc.setBuffer(input.buffer, offset: input.offset, index: 2)
        enc.setBuffer(out.buffer, offset: out.offset, index: 3)
        var inDimV = UInt32(bank.inDim)
        var outDimV = UInt32(bank.outDim)
        enc.setBytes(&inDimV, length: 4, index: 4)
        enc.setBytes(&outDimV, length: 4, index: 5)
        enc.setThreadgroupMemoryLength(2 * 32 * MemoryLayout<Float>.stride, index: 0)
        let tiles = bank.outDim / 2
        enc.dispatchThreadgroups(
            MTLSize(width: tiles, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

    /// Single-token (`x = [inDim]` bf16) batched QMV over `bank`,
    /// writing `[bank.outDim]` bf16 into `out`. Tries `qmvSplitK` first
    /// when `LagunaAttnInt8.splitKEnabled`, falling through to this
    /// file's original geometry on any decline (shape mismatch or odd
    /// `bank.outDim`).
    static func qmv(
        input: Tensor, bank: Bank, on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) {
        precondition(input.dtype == .bf16, "LagunaAttnNvfp4Tail.qmv: input must be bf16")
        precondition(
            input.elementCount == bank.inDim,
            "LagunaAttnNvfp4Tail.qmv: input.elementCount \(input.elementCount) != bank.inDim \(bank.inDim)"
        )
        precondition(
            out.elementCount == bank.outDim,
            "LagunaAttnNvfp4Tail.qmv: out.elementCount \(out.elementCount) != bank.outDim \(bank.outDim)"
        )
        if LagunaAttnInt8.splitKEnabled,
            qmvSplitK(input: input, bank: bank, on: cmd, device: device, into: out)
        {
            return
        }
        qmvPlain(input: input, bank: bank, on: cmd, device: device, into: out)
    }

    /// The original (pre-T39, non-split-K) QMV geometry: 2 simdgroups x
    /// 4 rows, each simdgroup walking the full K dimension. Split out of
    /// `qmv` so `LagunaAttnNvfp4V2Tests` can address it directly: the
    /// norm-folded V2 kernel's K loop reproduces exactly THIS kernel's
    /// per-row accumulation order, so `Ops.rmsNorm` + `qmvPlain` is its
    /// bit-exact two-dispatch reference regardless of the process-wide
    /// `BUTTER_LAGUNA_QMV_SPLITK` state.
    static func qmvPlain(
        input: Tensor, bank: Bank, on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) {
        let p = pso(
            "laguna_attn_qmv_nvfp4_g16_bf16", source: qmvSrc, device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
            enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
            enc.setBuffer(input.buffer, offset: input.offset, index: 2)
            enc.setBuffer(out.buffer, offset: out.offset, index: 3)
            var inDimV = UInt32(bank.inDim)
            var outDimV = UInt32(bank.outDim)
            enc.setBytes(&inDimV, length: 4, index: 4)
            enc.setBytes(&outDimV, length: 4, index: 5)
            let rowsPerTG = 8
            let tiles = (bank.outDim + rowsPerTG - 1) / rowsPerTG
            enc.dispatchThreadgroups(
                MTLSize(width: tiles, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
        }
    }

    /// DECODE_QMV_V2: norm-folded QKV(+gate) NVFP4 QMV — RMSNorm
    /// reduction + QMV in one dispatch (see `normQmvV2Src`). Declines
    /// (returns `false`, `out` untouched) unless `v2Enabled` and the
    /// same fast-path envelope `LagunaAttnInt8.qmvNormFolded` requires
    /// (`inDim <= 4096`, multiple of 256 so 64 real lanes evenly cover
    /// the `inDim/128` virtual simdgroups); caller falls back to the
    /// separate `inputNorm` + `qmv` chain. Also honors the shared
    /// `BUTTER_LAGUNA_QKV_NORMFOLD=0` escape hatch (it is the same fold
    /// decision, independent of bank flavor).
    static func normQmvV2(
        residual: Tensor, normWeight: Tensor, eps: Float, bank: Bank,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard v2Enabled, LagunaAttnInt8.normFoldEnabled,
            residual.dtype == .bf16, normWeight.dtype == .bf16, out.dtype == .bf16,
            residual.elementCount == bank.inDim,
            normWeight.elementCount == bank.inDim,
            out.elementCount == bank.outDim,
            bank.inDim <= 4096, bank.inDim.isMultiple(of: 256)
        else { return false }

        // Split-K geometry when the shape allows it (see
        // `normQmvSplitKV2Src`: in-situ probe had the 8-rows/TG variant
        // at ~348 GB/s effective vs split-K's ~389 on the gated o_proj
        // twin; K-parallelism is the difference at K=2048). The split-K
        // rms emulation needs `inDim/128` virtual simdgroups to divide
        // evenly over 4 real ones (`inDim % 512 == 0`) and the NR0=2
        // tiling needs an even `outDim`. Escape sub-flag
        // `BUTTER_LAGUNA_QMV_V2_NORMFOLD_SPLITK=0` pins the 8-rows/TG
        // variant for ablation.
        if bank.outDim.isMultiple(of: 2), bank.inDim.isMultiple(of: 512),
            ProcessInfo.processInfo.environment["BUTTER_LAGUNA_QMV_V2_NORMFOLD_SPLITK"] != "0"
        {
            let p = pso(
                "laguna_attn_norm_qmv_nvfp4_g16_splitk_v2_bf16", source: normQmvSplitKV2Src,
                device: device.mtlDevice)
            guard let enc = cmd.makeComputeCommandEncoder() else { return false }
            enc.setComputePipelineState(p)
            enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
            enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
            enc.setBuffer(residual.buffer, offset: residual.offset, index: 2)
            enc.setBuffer(normWeight.buffer, offset: normWeight.offset, index: 3)
            enc.setBuffer(out.buffer, offset: out.offset, index: 4)
            var inDimV = UInt32(bank.inDim)
            var outDimV = UInt32(bank.outDim)
            var epsV = eps
            enc.setBytes(&inDimV, length: 4, index: 5)
            enc.setBytes(&outDimV, length: 4, index: 6)
            enc.setBytes(&epsV, length: 4, index: 7)
            enc.setThreadgroupMemoryLength(2 * 32 * MemoryLayout<Float>.stride, index: 0)
            let tiles = bank.outDim / 2
            enc.dispatchThreadgroups(
                MTLSize(width: tiles, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
            return true
        }

        let p = pso(
            "laguna_attn_norm_qmv_nvfp4_g16_v2_bf16", source: normQmvV2Src,
            device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
        enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
        enc.setBuffer(residual.buffer, offset: residual.offset, index: 2)
        enc.setBuffer(normWeight.buffer, offset: normWeight.offset, index: 3)
        enc.setBuffer(out.buffer, offset: out.offset, index: 4)
        var inDimV = UInt32(bank.inDim)
        var outDimV = UInt32(bank.outDim)
        var epsV = eps
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&outDimV, length: 4, index: 6)
        enc.setBytes(&epsV, length: 4, index: 7)
        let rowsPerTG = 8
        let tiles = (bank.outDim + rowsPerTG - 1) / rowsPerTG
        enc.dispatchThreadgroups(
            MTLSize(width: tiles, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }
}
