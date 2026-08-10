// Copyright 2026 Tom Turney (@TheTom)
//
// W57: MoE combine-tail fusion — 3 elementwise `[T, hidden]` dispatches
// down to 1.
//
// A dispatch inventory of butter's Laguna prefill MoE tail vs mlx.fast's
// (W56) found butter running THREE separate `[T, hidden]` elementwise
// passes where mlx.fast fuses into one kernel:
//   1. `Ops.moeUnpermute` — combine: gather routed-expert outputs +
//      weighted top-k sum, writes the MoE output.
//   2. `Ops.add` — shared-expert add (`MoELayer.decodeManyNvfp4`'s tail,
//      when a shared expert is present and computed as a single
//      `[T,hidden]` tensor — dual-queue MPP or batched QMM, NOT the
//      per-row swiGLU fallback).
//   3. `Ops.add` — residual add (`LagunaText.forwardMulti`'s `.moe` case,
//      `h2 + ffnOutFlat`).
//
// This eliminates dispatches 2 and 3 (and their two extra full
// `[T,hidden]` read-modify-write round-trips) by having `Ops.moeUnpermute`'s
// generated kernel (`iron_moe_unpermute`) grow a `_tail` variant
// (`iron_moe_unpermute_tail`, `crates/wh-iron-std/src/kernels/moe/
// moe_unpermute_tail.rs` in the sibling iron repo) that takes the shared-
// expert output and residual as two extra inputs and folds `+ shared
// [+residual]` into the SAME f32 accumulator the weighted-sum loop already
// builds, before the single final cast+store. Same left-to-right
// accumulation order as the unfused call sites (unpermute, += shared,
// residual + that) — not a reassociation, so results should be
// byte-identical, not just within-tolerance (verified via iron's
// `test_moe_unpermute_tail_with_shared` / `_no_shared` correctness tests,
// CPU oracle, f32/f16/bf16, plus a full `moe_` suite regression pass).
//
// Only applies on the sorted/MPP `Ops.moeUnpermute` path (`sortedInvPerm`
// present) and only when the shared-expert output (if any) is already a
// single `[T,hidden]` tensor — NOT the `LagunaSortedCombine` numerics-audit
// alternative (a different, orthogonal experiment; mutually exclusive by
// construction, `LagunaSortedCombine.enabled` takes priority when both are
// set) and NOT the per-row swiGLU shared-expert fallback (rare/slow path;
// would need its own scratch-buffer restructuring to produce one
// `[T,hidden]` tensor before the combine, out of scope for this bounded
// lever).
//
// Opt-in: BUTTER_LAGUNA_FUSED_MOE_TAIL=1 (default OFF for the W57 A/B;
// zero behavior change unless set).
enum LagunaFusedMoeTail {
    static var enabled: Bool {
        ButterEnv["BUTTER_LAGUNA_FUSED_MOE_TAIL"] != "0"
    }
}
