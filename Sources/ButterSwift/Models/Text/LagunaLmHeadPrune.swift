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
//
// w21: INT5-planar alternative coarse copy. DEFAULT ON (A/B'd, escape
// BUTTER_LAGUNA_LMHEAD_INT5=0). The int5 coarse copy carries the SAME
// certified-conservative bound + exact-assembly stage, so it is argmax-
// identical to the MXFP8 arm by construction; confirmed byte-for-byte
// over a 128-tok greedy stream and measured +2.17% decode (183.9 vs 180.0
// tok/s, 5/5 env-balanced paired wins on laguna-xs-2.1-nvfp4, base arm
// BUTTER_LAGUNA_DECODE_QMV_V2=1). Where the MXFP8 arm above reads an offline
// sidecar file (2112 B/row: 2048 e4m3 code bytes + 64 e8m0 scale bytes),
// this arm builds a 1344 B/row copy AT MODEL LOAD directly from the
// resident BF16 `lm_head` weight (`LagunaInt5Coarse.build`, no extra file
// needed): a 4-bit nibble plane (1024 B/row) + a 1-bit high plane
// (256 B/row) + the same per-32-element e8m0 group scale (64 B/row),
// -36.4% coarse bytes. Each element is a signed 5-bit offset-binary code
// (`nibble | (highbit << 4)`, decoded as `q = u - 16`) instead of a full
// e4m3 byte.
//
// The int5 coarse kernel (`int5InlineCoarseSrc`) emits the SAME two f32
// outputs (coarse, delta) with the SAME `d*(1+GAMMA) + 2*GAMMA*m` bound as
// the MXFP8 inline coarse kernel -- only the per-element quantization terms
// change to the int5 alphabet (flat half-cell 0.5 in ratio units, and
// |what| = |q|; see `int5InlineCoarseSrc`'s doc for the re-derived, still
// certified-conservative bound). The inline-mask EXACT pass (step 3) is
// reused UNCHANGED, since it only consumes the (coarse, delta, thr,
// lm_head, x) contract both arms satisfy identically.
//
// Step 2 (the threshold) is NOT reused as-is: int5's `delta` is
// systematically wider than MXFP8's (a flat 0.5 half-cell vs the tighter
// per-code `laguna_hs8` table), and the shared `L = max(coarse - delta)`
// reduction folds every row's OWN delta into the threshold, so a wider
// delta alone would widen the candidate set well beyond what the same
// `|L|/64` margin gives the MXFP8 arm. `int5WinnerThresholdSrc` swaps in
// an EXACT-WINNER threshold instead: locate a row via a coarse-only
// argmax, then recompute that ONE row's exact BF16 logit in the same
// dispatch and threshold against it directly -- removing every row's
// delta from the threshold computation, not just widening the margin
// around it. See `LagunaInt5Coarse` and its `build` for the planar layout
// and the init-time range guard that declines back to the MXFP8 arm if
// the tensor ever produces |q| > 15.

import Foundation
import Metal
import IronSwift

enum LagunaLmHeadPrune {
    /// DEFAULT ON: unset or any value other than `"0"` enables the certified
    /// two-pass decode head when the MXFP8 sidecar + BF16 weight exist.
    /// Set `BUTTER_LAGUNA_LM_HEAD_PRUNE=0` to restore stock full lm_head.
    static let enabled: Bool = ButterEnv["BUTTER_LAGUNA_LM_HEAD_PRUNE"] != "0"

    /// T29: DEFAULT ON. Set `BUTTER_LAGUNA_LMHEAD_INLINE_MASK=0` to restore
    /// the pre-T29 four-dispatch coarse_bf/select/exact path (kept verbatim
    /// below for the kill switch and same-binary A/B).
    static let inlineMaskEnabled: Bool = ButterEnv["BUTTER_LAGUNA_LMHEAD_INLINE_MASK"] != "0"

    /// w21: DEFAULT ON. Set `BUTTER_LAGUNA_LMHEAD_INT5=0` to restore the
    /// 1 byte/value MXFP8 sidecar coarse copy. The int5 planar copy (nibble
    /// plane + 1-bit plane + e8m0 group scale = 1344 B/row, about -36% coarse
    /// bytes; see `LagunaInt5Coarse`) leaves the certified bound, the
    /// L/threshold reduction and the exact-assembly stage UNCHANGED -- only the
    /// coarse GEMV reads a smaller weight copy and folds the int5 half-cell
    /// into the same `d*(1+GAMMA) + 2*GAMMA*m` delta, so it is argmax-identical
    /// to the MXFP8 arm by construction (byte-identical 128-tok stream,
    /// +2.17% decode). Only consulted on the inline-mask path and only when the
    /// coarse copy was built at init (the range guard can decline back to the
    /// MXFP8 arm if the tensor ever produces |q| > 15); otherwise MXFP8 runs.
    static let int5Enabled: Bool = ButterEnv["BUTTER_LAGUNA_LMHEAD_INT5"] != "0"

    /// w24 isolation probe (NOT for shipping): forces the int5 arm to reuse
    /// the shared `laguna_lmhead_max_thr` reduction (the same kernel the
    /// MXFP8 arm uses) instead of `int5WinnerThresholdSrc`, even though the
    /// wider flat int5 delta will widen the candidate set beyond what that
    /// kernel gives the MXFP8 arm (see `int5WinnerThresholdSrc`'s doc). Only
    /// meaningful with `int5Enabled` set. Exists to isolate whether the
    /// extra single-TG winner-threshold dispatch (argmax reduction + serial
    /// one-row BF16 GEMV) is the source of the fused+int5 latency
    /// regression, independent of candidate-set width. Set
    /// `BUTTER_LAGUNA_LMHEAD_INT5_SHAREDTHR=1` to enable.
    static let int5SharedThresholdEnabled: Bool = ButterEnv["BUTTER_LAGUNA_LMHEAD_INT5_SHAREDTHR"] == "1"

    /// T29 diagnosis-only probe: prints `[laguna-lmhead-diag] candidates=K`
    /// (K of 100352 rows passing the certified bound, i.e. the size of the
    /// candidate set |C| that the exact pass must full-GEMV) per decode
    /// step to stderr. Adds one extra atomic-counter dispatch plus a
    /// blocking GPU readback, so it is OFF the hot path used for the A/B
    /// timing gates above -- opt-in only, never default ON.
    static let diagEnabled: Bool = ButterEnv["BUTTER_LAGUNA_LMHEAD_DIAG"] == "1"

    /// T40 follow-up opt-in: when a token routes through CPU sampling (see
    /// `SamplingMode` below), keep the certified prune's cheap GPU pass
    /// instead of falling back to the full unpruned lm_head, and restrict
    /// the CPU sampling distribution to the certified candidate set
    /// instead. DEFAULT OFF -- `samplingMode(...)` returns `.exactFullVocab`
    /// for CPU-sampled tokens unless this is set. See `maskNonCandidates`
    /// for the statistical justification and its excluded-mass bound.
    static var candidateRestrictedSamplingEnabled: Bool {
        ButterEnv["BUTTER_LMHEAD_PRUNE_SAMPLE_CANDIDATES"] == "1"
    }

    // ── T40 follow-up: sampling-path awareness ─────────────────────────
    //
    // The certificate above only protects the ARGMAX identity: every
    // non-candidate row is provably below the true max (see the
    // SELECTION step in the header comment), so greedy decode and the
    // fused-GPU categorical/argmax kernels -- which only ever consume the
    // WINNING logit's identity -- are unaffected by which rows got the
    // coarse approximation. CPU sampling (`Sampling.swift`, the
    // `cpuSample` path in `Generate.swift`) is different: temperature > 0
    // runs a full softmax over every row's *value*, and repetition
    // penalty rescales arbitrary rows before argmax/sampling even sees
    // them. Feeding either of those the coarse-approximated non-candidate
    // rows as if they were exact logits produces the wrong distribution
    // (README T40 follow-up). This section gives the CPU sampling path a
    // way to know (a) whether it must fall back to the full unpruned
    // lm_head for this token, and (b) when the certified candidate set
    // can safely stand in for the full vocab instead.

    /// Which strategy a single decode token's lm_head chain should use.
    enum SamplingMode: Equatable {
        /// Certified two-pass prune, unrestricted. Non-candidate rows
        /// carry the coarse MXFP8 approximation. Correct whenever only
        /// the argmax identity matters (greedy decode, GPU categorical
        /// sampling, and -- per `samplingMode(...)` below -- CPU
        /// sampling at temperature 0 with no repetition penalty, since
        /// that path also reduces to a plain argmax).
        case pruneCertified
        /// Full BF16 GEMV, every row exact. Required whenever CPU
        /// sampling reads more than the winning logit's identity.
        case exactFullVocab
        /// Opt-in: keep the certified prune's cheap GPU pass, but
        /// restrict the CPU sampling distribution to the certified
        /// candidate set and renormalize over just those rows (see
        /// `maskNonCandidates(...)`).
        case pruneCandidateRestricted
    }

    /// Pure decision, no GPU/Metal dependency -- exercised directly by
    /// `Tests/ButterTests/Generation/LmHeadSamplingPathTests.swift`.
    /// Mirrors (and is the source of truth for) the routing
    /// `Generate.swift`'s `sampleNext` applies before calling
    /// `engine.forward(...)` on the `cpuSample` path.
    ///
    /// - `temperature == 0 && repetitionPenalty == 1.0`: this is a plain
    ///   argmax regardless of which code path reaches it (the GPU fast
    ///   path, or the legacy CPU `Sampling.sample` short-circuit -- see
    ///   that function's "Greedy short-circuit" step, which runs before
    ///   temperature scaling and returns `argmaxOf` directly). Keep the
    ///   certified prune.
    /// - `repetitionPenalty != 1.0`: repetition penalty rescales
    ///   arbitrary rows' logits *before* argmax/sampling runs (Sampling
    ///   step 1), independent of logit magnitude. The certificate was
    ///   only checked against the UNPENALIZED logits, so a penalty can
    ///   promote a row the coarse pass never vetted above the true max.
    ///   Forces `exactFullVocab` unconditionally -- the opt-in flag does
    ///   NOT override this (see `maskNonCandidates` doc: its soundness
    ///   argument is stated purely in terms of unpenalized logits).
    /// - `temperature > 0`, no repetition penalty: default
    ///   `exactFullVocab` (correct, costs the prune's win); opt-in
    ///   `pruneCandidateRestricted` when
    ///   `BUTTER_LMHEAD_PRUNE_SAMPLE_CANDIDATES=1`.
    static func samplingMode(
        temperature: Float,
        repetitionPenalty: Float,
        candidateRestrictedSamplingEnabled: Bool =
            LagunaLmHeadPrune.candidateRestrictedSamplingEnabled
    ) -> SamplingMode {
        guard temperature > 0 || repetitionPenalty != 1.0 else {
            return .pruneCertified
        }
        guard repetitionPenalty == 1.0 else {
            return .exactFullVocab
        }
        return candidateRestrictedSamplingEnabled ? .pruneCandidateRestricted : .exactFullVocab
    }

    /// T40 follow-up: per-token override the generation driver sets immediately
    /// before calling `engine.forward(...)` on the `cpuSample` path, and
    /// restores to `.pruneCertified` immediately after (see
    /// `Generate.swift`'s `sampleNext`). Read once at the top of
    /// `lagunaFinalNormLmHead`'s prune-selection branch.
    ///
    /// A static context var rather than a `LanguageModel.forward(...)`
    /// parameter: `forward` is a protocol requirement every text family
    /// implements (Gemma, Qwen3, Llama, Mamba2, ...), not just Laguna;
    /// threading a Laguna-only concern through that shared signature
    /// would touch every conformer for a flag only one family's lm_head
    /// chain reads. This mirrors `LagunaDecodeKernelTiming`'s existing
    /// `beginToken`/`endToken` pattern for the same reason: decode is a
    /// strictly sequential, one-token-in-flight host loop (`sampleNext`
    /// calls `engine.forward` synchronously and nothing else touches the
    /// GPU concurrently with it), so a plain static is sound here
    /// without extra synchronization -- the same assumption
    /// `LagunaDecodeKernelTiming` already relies on.
    ///
    /// DEFAULT `.pruneCertified`: any call site that never sets this
    /// (tests, `butter bench`, the greedy/gpu-categorical fast paths)
    /// gets the certified-safe default, unchanged from behavior before
    /// this follow-up.
    nonisolated(unsafe) static var currentSamplingMode: SamplingMode = .pruneCertified

    /// T40 follow-up candidate-restricted support: the f32 `coarse`/`delta`
    /// certificate arrays and threshold scalar the two-pass kernels
    /// already compute, stashed by `logits(...)` on every call
    /// (regardless of `currentSamplingMode` -- retaining three Tensor
    /// handles is free; only reading them back to CPU costs anything).
    ///
    /// `Generate.swift` reads this ONLY after `engine.forward(...)` has
    /// returned for that same token, i.e. after the command buffer
    /// wrapping `logits(...)`'s dispatches has committed and
    /// `waitUntilCompleted()` returned -- at that point the GPU has
    /// finished writing these buffers and nothing else has been queued
    /// since (lm_head is the last op in a token's forward pass), so the
    /// values are stable to read. This is a narrower safety window than
    /// the `diagEnabled` probe's completion-handler read above (which is
    /// why that probe needs its own dedicated non-pooled buffer): here
    /// the read is synchronous and immediate, not deferred to a
    /// completion handler, so the shared scratch allocator backing
    /// `Tensor.empty` is not at risk of having been recycled out from
    /// under it yet. Callers should still defensively check the vocab
    /// size matches before trusting a stale value across model/family
    /// boundaries (e.g. a previous generation call on a different,
    /// non-Laguna model in the same process never touches this var, so
    /// it would otherwise carry an unrelated Laguna vocab size forward).
    nonisolated(unsafe) private(set) static var lastCandidateCertificate:
        (coarse: Tensor, delta: Tensor, thr: Tensor)?

    /// Clears `lastCandidateCertificate`. `Generate.swift` calls this
    /// once at the start of each `driveGeneration` call so a stale
    /// certificate from a previous generation (possibly on a different,
    /// non-Laguna model in the same process) can never be mistaken for
    /// this generation's.
    static func resetCandidateCertificate() {
        lastCandidateCertificate = nil
    }

    /// Restricts `assembled` (the prune's bf16 output: exact on
    /// candidate rows, coarse-approximated elsewhere) to just the
    /// certified candidate set by forcing every non-candidate row to
    /// `-infinity`, then hands the result to the existing
    /// `Sampling.sample(values:...)` pipeline unchanged. `-infinity`
    /// softmaxes to exactly zero probability, so the remaining
    /// candidates renormalize to sum to 1 automatically -- softmax
    /// already IS the renormalization step "restrict + renormalize"
    /// asks for; no separate rescale pass is needed. Top-K / top-P /
    /// min-P also treat `-infinity` correctly for free (always sorts
    /// last, contributes zero to the cumulative-softmax walk, always
    /// fails the min-P threshold).
    ///
    /// Recomputes the same predicate the exact kernel used
    /// (`coarse[i] + delta[i] >= thr`) in identical operand order --
    /// written as `!(coarse[i] + delta[i] >= thr)` rather than
    /// `coarse[i] + delta[i] < thr` so a NaN certificate value (should
    /// never happen under normal operation, but IEEE-754 comparisons
    /// with NaN are all false) is treated as non-candidate exactly like
    /// the kernel's `is_cand` boolean would be, instead of silently
    /// passing through a `<` comparison that is *also* false for NaN.
    ///
    /// ─── Statistical justification (the certificate, notes/68) ───
    /// `delta[i]` is a certified bound on the coarse approximation
    /// error: `coarse[i] - delta[i] <= exact[i] <= coarse[i] + delta[i]`
    /// for every row `i` (see the COARSE step doc above `coarseSrc`).
    /// Two things follow:
    ///
    ///  1. `L := max_j (coarse[j] - delta[j])` is a certified LOWER
    ///     bound on the true max: for every `i`,
    ///     `coarse[i] - delta[i] <= exact[i] <= true_max`, so taking the
    ///     max over `i` gives `L <= true_max`.
    ///  2. For an excluded row `i` (`coarse[i] + delta[i] < thr`, where
    ///     `thr = L - |L|/64`): `exact[i] <= coarse[i] + delta[i] < thr
    ///     = L - |L|/64 <= true_max - |L|/64`.
    ///
    /// So every excluded row's exact logit sits more than `|L|/64` below
    /// the true max. At temperature `T`, that row's softmax weight
    /// relative to the max row is bounded by `exp(-|L|/(64*T))`. This is
    /// a WORST-CASE PER-ROW bound, not a bound on the total excluded
    /// mass (which also depends on how many non-candidate rows there are
    /// and how close each sits to the `thr` boundary) -- the T29
    /// diagnosis probe (`BUTTER_LAGUNA_LMHEAD_DIAG=1`, prints `|C|`
    /// candidates/100352 per token) is the empirical, measured version
    /// of this argument and should be preferred over this static bound
    /// when tuning `T`. Illustrative numbers: for `|L| ~= 20` (a
    /// representative unnormalized bf16 lm_head magnitude for a
    /// confident decode step) and `T = 0.7`, the guaranteed gap is
    /// `20/64/0.7 ~= 0.446` nats, i.e. each excluded row's weight is
    /// bounded by `exp(-0.446) ~= 0.64x` the max row's -- a real but not
    /// overwhelming per-row exclusion at that `T`; the bound tightens
    /// sharply as `T` drops (at `T = 0.2`: `exp(-1.56) ~= 0.21x`) and
    /// loosens as `T` rises. Because it is only a per-row (not
    /// aggregate) bound, this mode trades a quantifiable-but-not-hard
    /// exactness guarantee for throughput; callers that need the hard
    /// guarantee should stay on the default `exactFullVocab`.
    static func maskNonCandidates(
        assembled: [Float], coarse: [Float], delta: [Float], thr: Float
    ) -> [Float] {
        precondition(
            assembled.count == coarse.count && coarse.count == delta.count,
            "maskNonCandidates: assembled/coarse/delta length mismatch "
                + "(\(assembled.count)/\(coarse.count)/\(delta.count))")
        var out = assembled
        for i in 0 ..< out.count where !(coarse[i] + delta[i] >= thr) {
            out[i] = -Float.infinity
        }
        return out
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

    // w21: int5-planar coarse GEMV. Same per-row shape as the MXFP8
    // `inlineCoarseSrc` above (one simdgroup per row, 32 lanes, 2 groups per
    // lane, `simd_sum` reduction, `laguna_e8m0_decode` group scale) and the
    // SAME two f32 outputs (coarse, delta) with the SAME
    // `d*(1+GAMMA) + 2*GAMMA*m` bound -- so the L/threshold reduction and the
    // inline exact pass below consume it unchanged. Only the coarse copy's
    // byte layout differs: instead of one e4m3 byte per value it reads a 4-bit
    // nibble plane (codes_lo, 1024 B/row, uint4 per lane-group at g*16) plus a
    // 1-bit plane (codes_hi, 256 B/row, one uint per lane-group at g*4),
    // decoding each value as offset-binary u-16 in [-15, 15] (see
    // `LagunaInt5Coarse.build`). Because the int5 alphabet is the flat integer
    // lattice q in ratio (w/sd) units, the half-cell is a constant 0.5 (vs the
    // MXFP8 per-code `laguna_hs8`) and |decode| in ratio units is |q| (vs
    // `laguna_e4m3_decode`): the certificate is identical in structure with
    // those two substitutions. d_i = sum_g sd_g * sum_{j in g} |x_j| * 0.5
    // >= sum_j |x_j| * |w_ij - sd_g*q_ij| (exact flat half-cell, sd a power of
    // two, |q| <= 15 by the init guard), and m_i = sum_g sd_g * sum_j |x_j| *
    // |q_ij| = sum_j |x_j| * |what_ij|, so delta_i = d_i*(1+GAMMA) +
    // 2*GAMMA*m_i is the same certified upper bound on |exact_i - coarse_i|
    // (int5 accumulation depth ~45 roundings/element-path << the GAMMA = 2^-15
    // budget, exactly as on the MXFP8 arm).
    private static let int5InlineCoarseSrc = header + """
    kernel void laguna_lmhead_int5_inline_coarse(
        device const bfloat *x [[buffer(0)]],
        device const uchar *codes_lo [[buffer(1)]],
        device const uchar *codes_hi [[buffer(2)]],
        device const uchar *scales [[buffer(3)]],
        device float *coarse [[buffer(4)]],
        device float *delta [[buffer(5)]],
        uint tgid [[threadgroup_position_in_grid]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
        constexpr float GAMMA = 0x1p-15f;
        uint row = tgid * 8u + sgid;
        if (row >= 100352u) return;

        const device uchar* lorow = codes_lo + size_t(row) * 1024u;
        const device uchar* hirow = codes_hi + size_t(row) * 256u;
        const device uchar* srow = scales + size_t(row) * 64u;

        float c_acc = 0.0f;
        float d_acc = 0.0f;
        float m_acc = 0.0f;
        for (uint gg = 0; gg < 2u; ++gg) {
            uint g = 2u * lane + gg;
            float sd = laguna_e8m0_decode(srow[g]);
            uint4 lo4 = ((const device uint4*)(lorow + g * 16u))[0];
            uint hb = ((const device uint*)(hirow + g * 4u))[0];
            const device ushort4* xrow = (const device ushort4*)(x + g * 32u);
            float cg = 0.0f;
            float dg = 0.0f;
            float mg = 0.0f;
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 4u; ++w) {
                // Word w: elements 8w..8w+7 of the group. Nibble plane byte b
                // holds elements 2b (low) / 2b+1 (high); 1-bit plane bit j of
                // the group's word holds element j's bit 4.
                uint lw = lo4[w];
                uint hw = hb >> (8u * w);
                uint4 ne = (uint4(lw) >> uint4(0u, 8u, 16u, 24u)) & 15u;
                uint4 no = (uint4(lw) >> uint4(4u, 12u, 20u, 28u)) & 15u;
                uint4 he = (uint4(hw) >> uint4(0u, 2u, 4u, 6u)) & 1u;
                uint4 ho = (uint4(hw) >> uint4(1u, 3u, 5u, 7u)) & 1u;
                // Offset-binary decode: value = u - 16 in [-15, 15], exact.
                float4 ve = float4(ne | (he << 4u)) - 16.0f;
                float4 vo = float4(no | (ho << 4u)) - 16.0f;
                // bf16 -> f32 is exactly bits<<16 for every value class.
                float4 xa = as_type<float4>(uint4(xrow[2 * w]) << 16);
                float4 xb = as_type<float4>(uint4(xrow[2 * w + 1]) << 16);
                float4 xe = float4(xa.x, xa.z, xb.x, xb.z);
                float4 xo = float4(xa.y, xa.w, xb.y, xb.w);
                float4 axe = metal::abs(xe);
                float4 axo = metal::abs(xo);
                #pragma clang loop unroll(full)
                for (uint k = 0; k < 4u; ++k) {
                    cg += xe[k] * ve[k];
                    cg += xo[k] * vo[k];
                    // Flat int5 half-cell: |w/sd - q| <= 0.5 for every element.
                    dg += axe[k] * 0.5f;
                    dg += axo[k] * 0.5f;
                    // Rounding-cover term: |what| in ratio units is |q|.
                    mg += axe[k] * metal::abs(ve[k]);
                    mg += axo[k] * metal::abs(vo[k]);
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

    // w21: EXACT-WINNER threshold, int5 arm only. The int5 coarse kernel's
    // `delta` is systematically wider than the MXFP8 arm's (flat 0.5 half-
    // cell vs the tighter per-code `laguna_hs8` table), and `maxThrSrc`
    // above folds every row's OWN delta into `L = max(coarse - delta)`
    // before subtracting the safety margin -- so a wider delta pulls `L`
    // down and widens `thr`'s candidate set for every row, not just the
    // ones that actually need the slack. This kernel removes `delta` from
    // the threshold entirely: single-TG strided argmax over `coarse` alone
    // (same reduction shape as `laguna_lmhead_max_thr`, just tracking an
    // index alongside the value) locates a row `r`, then the SAME dispatch
    // recomputes `r`'s exact BF16 logit `e_r` with a textual replica of
    // `laguna_lmhead_exact_inline_mask_block`'s per-row GEMV body (same
    // 32-lane partition, same 16 strided vec4 loads, same sequential f32
    // accumulation, same simd_shuffle_down tree) and emits
    // `thr = e_r - |e_r|/64` -- the same BF16-cast safety margin as
    // `laguna_lmhead_max_thr`, just anchored on one exact row instead of
    // every row's coarse-minus-delta.
    //
    // Soundness holds for ANY row `r`, not just the argmax-of-coarse pick:
    // `e_r <= max_i e_i` trivially, since `e_r` is one term of that max, and
    // that is the ONLY property of the old `L` the downstream candidacy
    // proof uses (see `maskNonCandidates`'s doc: `exact[i] <= coarse[i] +
    // delta[i] < thr = L - |L|/64 <= true_max - |L|/64`, substitute `e_r`
    // for `L` verbatim). So the coarse-argmax choice here is purely a
    // TIGHTNESS heuristic -- a row whose exact logit is likely close to the
    // true winner gives a threshold close to the ideal `L` -- never a
    // correctness requirement; a NaN or duplicate-max `coarse` still yields
    // SOME valid row index and the proof goes through unchanged.
    private static let int5WinnerThresholdSrc = header + """
    kernel void laguna_lmhead_int5_winner_threshold(
        device const float *coarse [[buffer(0)]],
        device const bfloat *lm_head [[buffer(1)]],
        device const bfloat *x [[buffer(2)]],
        device float *thr [[buffer(3)]],
        threadgroup float *sh_val [[threadgroup(0)]],
        threadgroup uint *sh_idx [[threadgroup(1)]],
        uint tid [[thread_position_in_threadgroup]],
        uint sgid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
        constexpr uint VOCAB = 100352u;
        constexpr uint K = 2048u;
        constexpr uint TG = 256u;

        float bestV = -metal::numeric_limits<float>::infinity();
        uint bestI = 0u;
        for (uint i = tid; i < VOCAB; i += TG) {
            float v = coarse[i];
            if (v > bestV) {
                bestV = v;
                bestI = i;
            }
        }
        sh_val[tid] = bestV;
        sh_idx[tid] = bestI;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG / 2u; s > 0u; s >>= 1u) {
            if (tid < s && sh_val[tid + s] > sh_val[tid]) {
                sh_val[tid] = sh_val[tid + s];
                sh_idx[tid] = sh_idx[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint r = sh_idx[0];

        // Exact GEMV of row r, simdgroup 0 only -- textually the same
        // per-row body as `laguna_lmhead_exact_inline_mask_block`'s
        // candidate path above.
        if (sgid == 0u) {
            float result = 0.0f;
            uint bn = lane * 4u;
            const device bfloat* mrow = lm_head + size_t(r) * K;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                vec<bfloat, 4> xv = *((const device vec<bfloat, 4>*)(x + bn));
                vec<bfloat, 4> mv = *((const device vec<bfloat, 4>*)(mrow + bn));
                result += float(mv.x) * float(xv.x) + float(mv.y) * float(xv.y)
                        + float(mv.z) * float(xv.z) + float(mv.w) * float(xv.w);
                bn += 128u;
            }
            #pragma clang loop unroll(full)
            for (ushort sn = 16; sn >= 1; sn >>= 1) {
                result += simd_shuffle_down(result, sn);
            }
            if (lane == 0u) {
                float e_r = result;
                thr[0] = e_r - metal::abs(e_r) * (1.0f / 64.0f);
            }
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

    /// w21: planar int5 coarse copy of the lm_head weight. Same certified
    /// two-pass role as the MXFP8 `Mxfp8Linear` sidecar, at ~0.66 byte/value:
    /// a 4-bit nibble plane (`codesLo`, 1024 B/row) + a 1-bit plane
    /// (`codesHi`, 256 B/row) + a per-32-element e8m0 group scale byte
    /// (`scales`, 64 B/row) = 1344 B/row, vs the MXFP8 copy's 2048 code bytes +
    /// 64 scale bytes = 2112 B/row (about -36% coarse bytes). Built once from
    /// the BF16 weight at init and only when `int5Enabled`; the MXFP8 sidecar
    /// stays the default/fallback. Consumed by `int5InlineCoarseSrc` above.
    struct LagunaInt5Coarse {
        let codesLo: Tensor  // u8 [vocab, kDim/2]  nibble plane
        let codesHi: Tensor  // u8 [vocab, kDim/8]  1-bit plane
        let scales: Tensor   // u8 [vocab, kDim/32] e8m0 group scale bytes

        /// Symmetric int5 quantize + planar pack, laid out so
        /// `int5InlineCoarseSrc` decodes it bit-for-bit (this is the
        /// CPU/Tensor translation of the reference `buildInt5Planes` MLX ops).
        ///
        /// Scale rule (per 32-element group, gmax = max|w|): sd = 2^e with
        /// e = floor_exp(gmax) - 3, bumped one exponent when the gmax mantissa
        /// is >= 1.9375 (bit test mant >= 0x780000), so gmax/sd < 15.5 EXACTLY
        /// (8m < 15.5 when m < 1.9375; 4m < 8 otherwise). Then q = round(w/sd)
        /// -- the quotient is EXACT because sd is a power of two -- has
        /// |q| <= 15 with a flat half-cell |w - sd*q| <= sd/2. The scale byte
        /// carries e8m0 semantics (decoded by the shared `laguna_e8m0_decode`).
        ///
        /// RANGE GUARD (mirrors the reference): the no-overflow property is
        /// re-verified on the actual tensor; if any element would need |q| > 15
        /// (a 6th magnitude bit, which would make u = q + 16 leave [1, 31] and
        /// break the kernel's certified bound) the build declines and returns
        /// nil, so the caller keeps the MXFP8 arm.
        ///
        /// Untimed init and gated OFF by default, so this per-row CPU pass over
        /// the 100352 x 2048 weight is paid only when `BUTTER_LAGUNA_LMHEAD_INT5=1`.
        static func build(from weight: Tensor, device: Device) -> LagunaInt5Coarse? {
            precondition(weight.dtype == .bf16 && weight.shape == [vocab, kDim])
            let groups = kDim / 32  // 64
            let loBytes = kDim / 2  // 1024
            let hiBytes = kDim / 8  // 256

            let loBuf = device.makeBuffer(length: vocab * loBytes)
            let hiBuf = device.makeBuffer(length: vocab * hiBytes)
            let scBuf = device.makeBuffer(length: vocab * groups)

            let wPtr = weight.buffer.contents().advanced(by: weight.offset)
                .assumingMemoryBound(to: UInt16.self)
            let loPtr = loBuf.contents().assumingMemoryBound(to: UInt8.self)
            let hiPtr = hiBuf.contents().assumingMemoryBound(to: UInt8.self)
            let scPtr = scBuf.contents().assumingMemoryBound(to: UInt8.self)

            var u = [UInt8](repeating: 0, count: kDim)  // offset-binary, one row
            var overflow = false
            for row in 0 ..< vocab {
                let wRow = wPtr + row * kDim
                let scRow = scPtr + row * groups
                for g in 0 ..< groups {
                    let base = g * 32
                    var gmax: Float = 0
                    for j in 0 ..< 32 {
                        let a = abs(Float(bitPattern: UInt32(wRow[base + j]) << 16))
                        if a > gmax { gmax = a }
                    }
                    // e8m0 scale byte: exponent(gmax) - 3, +1 when mant >= 1.9375.
                    let gbits = gmax.bitPattern
                    let biasedE = Int(gbits >> 23)
                    let mant = gbits & 0x007F_FFFF
                    let bump = (mant >= 0x0078_0000) ? 1 : 0
                    var sdByte = biasedE - 3 + bump
                    if sdByte < 0 { sdByte = 0 }
                    if sdByte > 255 { sdByte = 255 }
                    scRow[g] = UInt8(sdByte)
                    let sd =
                        (sdByte == 0)
                        ? Float(bitPattern: 0x0040_0000)  // 2^-127, e8m0 semantics
                        : Float(bitPattern: UInt32(sdByte) << 23)
                    for j in 0 ..< 32 {
                        let f = Float(bitPattern: UInt32(wRow[base + j]) << 16)
                        let q = (f / sd).rounded(.toNearestOrEven)
                        if abs(q) > 15 { overflow = true }
                        u[base + j] = UInt8(truncatingIfNeeded: Int(q) + 16)
                    }
                }
                if overflow { break }
                // Planar pack this row. Nibble plane: byte b = low 4 bits of the
                // even element (2b) in the low nibble, of the odd element (2b+1)
                // in the high nibble. 1-bit plane: byte s = bit 4 of elements
                // 8s..8s+7, one per bit position. (Global-linear layout; the
                // group scale is the only per-group quantity.)
                let loRow = loPtr + row * loBytes
                for b in 0 ..< loBytes {
                    loRow[b] = (u[2 * b] & 0x0F) | ((u[2 * b + 1] & 0x0F) << 4)
                }
                let hiRow = hiPtr + row * hiBytes
                for s in 0 ..< hiBytes {
                    var byte: UInt8 = 0
                    for p in 0 ..< 8 {
                        byte |= ((u[8 * s + p] >> 4) & 1) << p
                    }
                    hiRow[s] = byte
                }
            }
            guard !overflow else {
                FileHandle.standardError.write(
                    Data(
                        "[laguna] lm_head int5: code overflow (|q|>15); using MXFP8 arm\n"
                            .utf8))
                return nil
            }
            // Loaded straight into a host-built buffer (not the GGUF mmap-view
            // residency path), read every decode-prune pass -- pin it, exactly
            // like the MXFP8 sidecar in `lagunaLoadMxfp8LmHead`.
            device.markWeightsResident([loBuf, hiBuf, scBuf])
            return LagunaInt5Coarse(
                codesLo: Tensor(
                    buffer: loBuf, offset: 0, shape: [vocab, loBytes], dtype: .u8),
                codesHi: Tensor(
                    buffer: hiBuf, offset: 0, shape: [vocab, hiBytes], dtype: .u8),
                scales: Tensor(
                    buffer: scBuf, offset: 0, shape: [vocab, groups], dtype: .u8))
        }
    }

    /// Pruned decode head. `normed` is post-finalNorm hidden [hidden] bf16.
    /// `mx` is MXFP8 sidecar; `bf16Weight` is full BF16 lm_head [vocab, hidden].
    /// `int5` is the optional w21 planar int5 coarse copy: when present and
    /// `int5Enabled`, the inline-mask path reads it for the coarse GEMV instead
    /// of `mx`, producing the same `coarse`/`delta` the rest of the pass
    /// consumes. `mx` is still the fallback and drives every other path.
    static func logits(
        normed: Tensor,
        mx: Mxfp8Linear,
        bf16Weight: Tensor,
        int5: LagunaInt5Coarse? = nil,
        on cmd: MTLCommandBuffer,
        device: Device
    ) -> Tensor {
        precondition(normed.dtype == .bf16 && bf16Weight.dtype == .bf16)
        precondition(bf16Weight.shape == [vocab, kDim])
        let mtl = device.mtlDevice

        let coarse = Tensor.empty(shape: [vocab], dtype: .f32, device: device)
        let delta = Tensor.empty(shape: [vocab], dtype: .f32, device: device)

        if inlineMaskEnabled {
            // T29: 1) inline coarse (coarse, delta only, no coarse_bf store).
            // w21: when the int5 sidecar is built and enabled, read the smaller
            // planar int5 coarse copy instead of the MXFP8 codes/scales; both
            // emit the SAME f32 `coarse`/`delta` so steps 2/3 below are
            // untouched. The MXFP8 kernel is the default/fallback arm.
            if int5Enabled, let i5 = int5 {
                let coarsePSO = pso(
                    "laguna_lmhead_int5_inline_coarse",
                    source: int5InlineCoarseSrc, device: mtl)
                if let enc = cmd.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(coarsePSO)
                    enc.setBuffer(normed.buffer, offset: normed.offset, index: 0)
                    enc.setBuffer(i5.codesLo.buffer, offset: i5.codesLo.offset, index: 1)
                    enc.setBuffer(i5.codesHi.buffer, offset: i5.codesHi.offset, index: 2)
                    enc.setBuffer(i5.scales.buffer, offset: i5.scales.offset, index: 3)
                    enc.setBuffer(coarse.buffer, offset: coarse.offset, index: 4)
                    enc.setBuffer(delta.buffer, offset: delta.offset, index: 5)
                    enc.dispatchThreadgroups(
                        MTLSize(width: (vocab + 7) / 8, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    enc.endEncoding()
                }
            } else {
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
            }

            // 2) threshold. w21: the int5 arm swaps in the EXACT-WINNER
            // threshold (`int5WinnerThresholdSrc`) instead of the shared
            // `L = max(coarse - delta)` reduction below -- int5's wider,
            // flat-half-cell `delta` would otherwise get folded into `L`
            // for every row, not just the ones that need the slack, and
            // widen the candidate set well beyond what the MXFP8 arm sees
            // at the same `thr` margin (see the kernel's doc above). Both
            // arms still produce the same single-scalar `thrBuf` that step
            // 3 below consumes identically.
            let thrBuf = Tensor.empty(shape: [1], dtype: .f32, device: device)
            if int5Enabled, int5 != nil, !int5SharedThresholdEnabled {
                let winnerPSO = pso(
                    "laguna_lmhead_int5_winner_threshold",
                    source: int5WinnerThresholdSrc, device: mtl)
                if let enc = cmd.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(winnerPSO)
                    enc.setBuffer(coarse.buffer, offset: coarse.offset, index: 0)
                    enc.setBuffer(bf16Weight.buffer, offset: bf16Weight.offset, index: 1)
                    enc.setBuffer(normed.buffer, offset: normed.offset, index: 2)
                    enc.setBuffer(thrBuf.buffer, offset: thrBuf.offset, index: 3)
                    enc.setThreadgroupMemoryLength(256 * 4, index: 0)
                    enc.setThreadgroupMemoryLength(256 * 4, index: 1)
                    enc.dispatchThreadgroups(
                        MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    enc.endEncoding()
                }
            } else {
                // thr = max(coarse - delta) - |L|/64 (single TG reduction)
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
            // T40 follow-up: stash the certificate for the candidate-restricted CPU
            // sampling opt-in (see `lastCandidateCertificate` doc above).
            // Retaining three Tensor handles is free; nothing reads them
            // back to CPU unless `samplingMode(...)` picked
            // `.pruneCandidateRestricted` for this token.
            lastCandidateCertificate = (coarse: coarse, delta: delta, thr: thrBuf)
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
        // T40 follow-up: same certificate stash as the inline-mask branch above,
        // kept here too so the kill-switch (BUTTER_LAGUNA_LMHEAD_INLINE_MASK=0)
        // A/B path still supports the candidate-restricted CPU sampling opt-in.
        lastCandidateCertificate = (coarse: coarse, delta: delta, thr: thrBuf)
        return assembled
    }
}
