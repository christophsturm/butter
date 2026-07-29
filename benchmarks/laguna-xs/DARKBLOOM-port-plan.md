# DARKBLOOM → butter+iron port plan (Laguna XS NVFP4)

Research-only. Sources: mlxfast `LagunaRuntimeModel.swift` / `LagunaLmHeadPrune.swift` / `LagunaRuntimeWeights.swift`; butter `LagunaNvfp4Fuse.swift`, `LagunaLmHeadPrune.swift`, `LagunaText.swift`, `MoELayer.swift` Path B, `benchmarks/laguna-xs/README.md`.

**Do not modify butter in this note.** Absolute paths below are read references.

---

## Absolute targets (Yukon / board)

| Metric | Butter board memo (older) | User: Yukon #1 (2026-07-29) | Butter clean NVFP4 (T10–T11) |
|--------|---------------------------|-----------------------------|-----------------------------|
| Decode tok/s | ~102 | **~131** | **~85** (host-sensitive; thrash → 55–58) |
| Prefill tok/s (T≈512) | ~3651 | **~4030** | **~450–520** (MPP + NAX + MMA era) |

### What #1 (~131 / ~4030) implies

- **Decode gap:** 131/85 ≈ **1.54×**. Yukon serial score weights decode **0.75** → closing this gap is high leverage *if* prefill is not a floor.
- **Prefill gap:** 4030/500 ≈ **8×**. Prefill is still the **score floor** for butter off-board chase. Even perfect decode (speedup 1.0 vs 102) with prefill at 500 yields score ~0.55 vs baseline 102/3651 math; vs #1 the absolute bar is higher.
- **MoE dominates prefill:** butter timing (`BUTTER_LAGUNA_TIMING=1`) shows MoE ~95% of T=512 wall (~30 ms/layer). Board needs ~3.5 ms/layer → **~8.5× MoE kernel gap**, not attention polish.
- **mlxfast/Yukon score:** `score = decode_speedup^0.75 * prefill_speedup^0.25`, floors 0.95, **acceptance band ≈ ±5% per submission** (chunk large wins). Butter’s internal score uses the same exponents against a pinned MLX baseline (~72 dec / ~2721 pref) or board constants (102 / 3651).
- **Implication for port priority:**  
  - **Sprint 1 (decode):** iron-bitexact MoE fusions + QK-norm+RoPE + residual/router + dual-queue/pipe hygiene → claw toward 100–120 t/s.  
  - **Sprint 2 (prefill):** gather-NAX / BK=64 / fused sorted gate+up / MoE tail → only path to multi-k t/s.  
  - **Sprint 3:** Yukon only if quality gates + paired band strategy exist; do not chase micro-fusions that ranked −0.2% after rot.

---

## ROI ranking (for butter+iron today)

Rough expected impact on **butter** (not mlxfast’s incremental %). Higher = do first.

| Rank | Trick | Est. decode | Est. prefill | Butter status | Quality risk |
|------|-------|-------------|--------------|---------------|--------------|
| 1 | Iron-bitexact **routed** SwiGLU QMV + **down-reduce** (DARKBLOOM_FUSED_ROUTED_*) | **+3–8%** if wins | small | **Partial** (opt-in kernels lose / not bitexact) | Med (dequant mismatch) |
| 2 | **Fused sliding QK-norm + RoPE** (+ full YaRN twin) | **+1–2%** (mlxfast’s biggest single ranked win +1.73%) | prefill twin +few % | **Missing** (separate rmsNormRows + rope) | Low if stock math |
| 3 | Prefill **gather MoE kernel class** (NAX / BK64 / fused gate+up sorted) | none | **+10–50%+** toward board | **Partial** (MPP BM16, gather-NAX spike, concurrent gate+up) | High if wrong tile |
| 4 | **Shared** fused gate+up SwiGLU QMV (decode) + iron match | +0.5–1.5% | shared ~12% of MoE; prefill fused bank already ON | **Partial** (opt-in; drift vs iron) | Med |
| 5 | **Fused gated o_proj** (softplus × O GEMV) | +0.5–1.5% | small | **Missing** (softplus expand + separate o_proj) | Low–med |
| 6 | **Residual + RMS + router GEMV** fold | +0.5–1.5% | small | **Partial** (`addAndRmsNorm`; no router in-kernel) | Low |
| 7 | **Decode async / CB ladder** (overlap) | +5–10% when GPU-idle (mlxfast ladder history) | prefill ladder | **Partial** (layer pipe depth 4, dual-queue) | Low (scheduling) |
| 8 | **Wired residency / ZH** (driver residency) | +0–4% | **large** on MLX path | **Partial** (`Device.markWeightsResident`) | Low if capacity-safe |
| 9 | **NVFP4 scale-fold** in dequant ALU | small per kernel, multiplies across L×experts | small–med | **Missing / iron-native** (different dequant) | Low if identity |
| 10 | **lm_head prune** (certified two-pass) | +0–2% butter; mlxfast default ON | none | **Exists opt-in OFF** | Low with cert / med without delta |
| 11 | Prefill **MoE tail** (weighted reduce + residual) | none | +1–4% | **Missing** | Low |
| 12 | Dense **L0** gate-up SwiGLU + down residual | tiny (1 layer) | tiny | **Missing** | Low |
| 13 | **Fused QKV projection** (one dispatch, no bank) | small | none | **Exists** (stacked dense GEMV default) | Low |
| 14 | Prefill **router tournament** | none | small | **Missing** (GPU top-k exists) | Low |
| — | Fused QKV **bank** (`FUSED_QKV`) | **null** | mild cost | Skip | — |
| — | RoPE **angle atlas** | **−0.23%** decode default | was −0.79% ranked | Skip | — |
| — | Prefill **router top8** ALU select | — | **−0.68%** ranked | Skip | — |

---

## Per-trick detail

### 1. Routed NVFP4 gate/up bank + SwiGLU QMV + down reduce (+ shared joint)

**What it does (mlxfast)**  
- Load-time row-concat NVFP4 `[gate; up]` per expert bank (`DARKBLOOM_FUSED_ROUTED_GATE_UP`, default ON).  
- Decode: one gather-QMV-style dispatch → `[1,1,8,1,512]` SwiGLU mids (`FUSED_ROUTED_SWIGLU_QMV`).  
- Decode: eight downs + router weight + scale + residual in fewer dispatches (`FUSED_ROUTED_DOWN_REDUCE`, `FUSED_ROUTED_SHARED_DOWN_RESIDUAL`, `FUSED_ROUTED_SHARED_SWIGLU_QMV`).  
- Claim: bit-exact per-row K-loop vs separate banks; decode ~+1.9% for gate/up bank alone historically.

**Default:** ON (all major routed/shared flags `!= "0"`).

**Impact:** Decode-primary; prefill uses separate flags / stock banks for older path, with `PREFILL_FUSED_GATE_UP` for sorted M≥64 (~4% prefill post-RUNSKIP).

**Butter today:** **Partial / quality-fragile**  
- Files: `/Users/tom/dev/butter-dev/Sources/ButterSwift/Models/Text/LagunaNvfp4Fuse.swift`, `MoELayer.swift` `decodeGPUPathBNvfp4`.  
- Opt-in: `BUTTER_LAGUNA_PATHB_GATEUP_FUSE`, `PATHB_NVFP4_FUSED` (chain8 — **slow**), `PATHB_NVFP4_SLOT_FUSE`, `FUSED_SHARED_SWIGLU`.  
- T11: serial chain8 and slot fuse **lost** tok/s; gateup fuse ~parity; shared SwiGLU **continuation drift** vs iron dequant.

**Port steps**  
1. Diff mlxfast `laguna_*_swiglu*` / down-reduce Metal against `iron_nvfp4_qgemv` / eidx many (scale, nibble order, global, block=16).  
2. Prefer **wrapping iron** (shared X load, multi-slot) over hand-rolled e2m1 if iron is the quality oracle.  
3. Ship `BUTTER_LAGUNA_PATHB_ROUTED_SWIGLU=1` iron-bitexact; A/B cos vs stock Path B eidx on real residual.  
4. Then fold weighted down + residual (`scalarFMAChain8` + adds) only if TG occupancy beats concurrent eidx.  
5. Keep chain8 serial **off**; do not default until clean-host decode ↑.

**Quality risk:** Medium–high until bitexact to iron; Paris-OK ≠ long-gen OK (shared fuse already showed drift).

---

### 2. Fused QK-norm + RoPE (sliding + full YaRN)

**What it does**  
One kernel: per-head RMSNorm(q/k) + rotary (plain sliding or partial YaRN full). Replaces 4–6 dispatches/layer.  
mlxfast history: barrier-on-broadcast bug caused −0.19% ranked; **local rsqrt, no TG barrier** → **+1.19% steady / +1.73% ranked** (`DARKBLOOM_FUSED_SLIDING_QK_NORM_ROPE`, default ON). Prefill twin `DARKBLOOM_PREFILL_QK_NORM_ROPE` default ON. Full: `DARKBLOOM_FUSED_FULL_QK_NORM_YARN`.

**Butter today:** **Missing**  
- Decode: `Ops.rmsNormRows` ×2 then `applyRopeInPlace` (`LagunaText.swift` ~1099–1109).  
- Prefill: same pattern in `forwardMulti`.  
- Iron has `iron_rms_norm_small`, `iron_rope_*`, `iron_partial_rope_*` separately (`Ops.swift` / `OpsFused.swift` / `OpsDSv4.swift`).

**Port steps**  
1. New iron or Butter Metal: `laguna_sliding_qk_norm_rope_bf16` (hd=128, Laguna shapes).  
2. **No** threadgroup barrier for inv-RMS after `simd_sum` (mlxfast lesson).  
3. Full layers: partial rotary dim + YaRN factors from same tables as stock.  
4. Wire decode then prefill multi-token grids.  
5. Escape: `BUTTER_LAGUNA_FUSED_QK_ROPE=0`.

**Quality risk:** Low if cos/sin and RMS match stock; YaRN partial is the sharp edge.

---

### 3. Prefill sorted fused gate+up gather-GEMM (+ expert-aligned / NAX)

**What it does**  
`DARKBLOOM_PREFILL_FUSED_GATE_UP` (default ON): one N=1024 sorted gather-QMM vs two N=512 (~4% prefill).  
`DARKBLOOM_EXPERT_ALIGNED_GATHER` (default ON): interleaved gate/up bank layout for gather.  
Board path (research): MLX `fp_gather_qmm_rhs_nax` BM=BN=BK=**64**, `tile_matmad_nax`.

**Butter today:** **Partial**  
- Concurrent gate+up MPP BM16 default; gather-NAX spike opt-in; dense NAX for shared ~+10%.  
- Expert-group QMM slower (host segment scan).  
- Still ~8× off #1 on MoE.

**Port steps**  
1. Finish iron gather **BK=64** / real NAXTile if available; else maximize BM16 MPP.  
2. Fused `[gate;up]` **sorted** gather once + interleaved SwiGLU (mlxfast layout).  
3. Prefill MoE tail (trick 11).  
4. Kill host waits on expert-group path or drop it.

**Quality risk:** High for new tiles (historical all-NaN / empty gen); gate with cos=1 vs eidx layer scan.

---

### 4. Shared expert fused gate+up + SwiGLU QMV / prefill QMM

**What it does**  
`DARKBLOOM_FUSED_SHARED_GATE_UP` + `FUSED_SHARED_SWIGLU_QMV` + optional `FUSED_SHARED_DOWN_RESIDUAL` (all default ON decode). Prefill keeps smaller shapes on stock or separate flags.

**Butter today:** **Partial**  
- Prefill fused bank + iron QMM: **default ON** (`BUTTER_LAGUNA_SHARED_PREFILL_FUSED_GATEUP`).  
- Decode shared SwiGLU: **opt-in OFF** (quality drift).

**Port steps**  
1. Make `sharedSwigluQmv` call **identical** scale/nibble path as `iron_nvfp4_qgemv`.  
2. Re-enable default only after Paris+fib+long + logit cos.  
3. Optional: fuse shared down + residual with routed acc.

**Quality risk:** Medium until iron-identical.

---

### 5. Fused gated output projection

**What it does**  
`DARKBLOOM_FUSED_GATED_OUTPUT` (default ON): softplus head-gate × o_proj in one GEMV variant per attn family.

**Butter today:** **Missing**  
- `applySoftplusHeadGate` → expand → `oProj` (`LagunaText.swift`).

**Port steps**  
1. Kernel: g_proj GEMV (N=64 heads) or consume precomputed gates; scale o rows by softplus(g_h) without full expand materialize.  
2. Note mlxfast: **g_proj never row-fused into QKV** (different split-K).  
3. Escape flag; A/B decode.

**Quality risk:** Low–medium (softplus + bf16 order).

---

### 6. Residual + RMSNorm (+ router)

**What it does**  
`DARKBLOOM_FUSED_RESIDUAL_RMS` / `PREFILL_FUSED_RESIDUAL_RMS` (ON): residual add + RMS one kernel.  
`DARKBLOOM_FUSED_RESIDUAL_RMS_ROUTER` (ON): same + router GEMV `[256,2048]` with `ROUTER_ROWS_PER_GROUP=8` for occupancy (bit-exact row ownership).

**Butter today:** **Partial**  
- Decode already uses `Ops.addAndRmsNorm` when `hidden≤4096` (`LagunaText.swift` ~1135–1147).  
- Router is separate BF16 gemv + Path B top-k.

**Port steps**  
1. Extend fused kernel output: write residual, normed **and** router logits (rows_per_group=8).  
2. Or chain iron_add_rms_norm → iron gemv with persistent ffnIn.  
3. Prefill: already multi-row RMS paths; fuse residual only if bandwidth-bound.

**Quality risk:** Low (row-independent).

---

### 7. Decode async stage / prefill async ladder

**What it does**  
`DARKBLOOM_DECODE_ASYNC_STAGE` default `at:1,7,15,23,31,39`: `asyncEval` after boundaries — **bit-exact**, pure overlap. History: ladder8 +9.7% vs off; refined schedule +~1.7% more.  
`DARKBLOOM_PREFILL_ASYNC_LADDER` default 8.

**Butter today:** **Partial**  
- Layer pipe depth 4, `LAYERS_PER_CB` 2; dual-queue shared; no MLX-style asyncEval.  
- Dual-queue costs 1 CB + 2 events × ~39 MoE/token — can hurt under host thrash.

**Port steps**  
1. Clean-host A/B: `DUAL_QUEUE=0` early-encode vs dual-queue.  
2. Align pipe boundaries with mlxfast’s front-heavy schedule (fire early after layer 1).  
3. Avoid host waits on NVFP4 Path B (already fence-free on quality path).

**Quality risk:** Low.

---

### 8. Wired zero-headroom residency (`DARKBLOOM_WIRED_ZH`)

**What it does**  
Wire live weights once at load (fraction×active + slack); capacity so scored scratch never fits → no per-CB residency thrash. Measured up to **−28% prefill, −4% decode** at full wire on MLX path. Default ON ≥96 GB machines.

**Butter today:** **Partial**  
- `Device.markWeightsResident` / `MTLResidencySet` for weight buffers.  
- Not the same zero-headroom capacity discipline as mlxfast notes/47.

**Port steps**  
1. Audit whether Laguna load calls markWeightsResident for full 21.6 GB + fused banks.  
2. Optionally size residency set to live+slack only (mlxfast dose table).  
3. Measure prefill primarily.

**Quality risk:** Low; risk is OOM / OS wired cap on <128 GB.

---

### 9. NVFP4 scale fold (`DARKBLOOM_NVFP4_SCALE_FOLD`)

**What it does**  
Hoist `2^14` out of 16-wide qdot into scale (`× 4194304` path); −16 multiplies per qdot_16; claimed bit-exact. Default ON.

**Butter today:** Iron dequant is a different implementation (`iron_nvfp4_e4m3_scale` + e2m1). Fold may already be irrelevant or partially present in iron.

**Port steps**  
1. Profile iron qgemv ALU; apply same identity if multiplies remain.  
2. Enumerate 256 E4M3 codes (mlxfast notes/58).

**Quality risk:** Low if proven identity.

---

### 10. lm_head prune (certified two-pass)

**What it does**  
`DARKBLOOM_LM_HEAD_PRUNE` **default ON**: MXFP8 coarse GEMV + certified delta + mask + exact BF16 block GEMV (stock `gemv_al_bfloat16` replica for candidates). Cuts 411 MB BF16 traffic when candidates sparse.

**Butter today:** **Exists, default OFF**  
- `/Users/tom/dev/butter-dev/Sources/ButterSwift/Models/Text/LagunaLmHeadPrune.swift`  
- `BUTTER_LAGUNA_LM_HEAD_PRUNE=1`; beta thr = max − beta (default 2.0) — **not full certified delta**.  
- T8: ~+1 t/s when warm; full MXFP8 replace **slower**.

**Port steps**  
1. Port mlxfast **delta_i** certificate + select kernel (not just beta thr).  
2. Speed coarse below BF16 gemv (v2 vectorized already; still ≈ BF16 cost on butter).  
3. Keep default OFF until coarse is net win on clean host.

**Quality risk:** Low with certificate; medium with beta-only thr.

---

### 11. Prefill MoE tail (+ sorted tail)

**What it does**  
`DARKBLOOM_PREFILL_MOE_TAIL` (ON): weighted expert combine + routed scale + shared + residual in one elementwise kernel.  
`PREFILL_SORTED_MOE_TAIL` (ON): avoid scatter unsort copy via inverse perm.

**Butter today:** **Missing** (explicit multi-op Path B / MPP unpermute).

**Port steps**  
1. After gather MPP outs: single kernel reduce 8 slots + shared + residual.  
2. Optional sorted-order path if scatter shows up in capture.

**Quality risk:** Low if reduction order fixed (slot order).

---

### 12. Dense L0 gate-up SwiGLU + down residual

**What it does**  
`DARKBLOOM_FUSED_DENSE_GATE_UP_SWIGLU` / `FUSED_DENSE_DOWN_RESIDUAL` (ON): layer-0 BF16 MLP only (not NVFP4).

**Butter today:** **Missing** (generic dense MLP).

**Port steps**  
1. Fuse only layer 0 in `LagunaText` dense branch.  
2. Low priority (1 of 40 layers).

**Quality risk:** Low.

---

### 13. Fused QKV (bank vs projection)

| Flag | Default | Notes |
|------|---------|-------|
| `DARKBLOOM_FUSED_QKV` | **OFF** | Concat bank; mild prefill cost, no decode gain |
| `DARKBLOOM_FUSED_QKV_PROJECTION` | **ON** | One dispatch over three stock weights; no bank |

**Butter:** stacked `[Q;K;V]` dense GEMV default (`BUTTER_LAGUNA_SERIAL_QKV` escape). **Roughly covered.**

**Port:** optional projection-style without doubling memory; skip bank concat.

---

### 14. Router fusion family

| Flag | Default | Notes |
|------|---------|-------|
| `DARKBLOOM_FUSED_ROUTER` | ON | top-8 selection one kernel |
| `FUSED_ROUTER_CAST` / `NORM` | ON | sink cast + renorm |
| `PREFILL_ROUTER_TOP8` | **OFF** | ranked −0.68% |
| `PREFILL_ROUTER_TOURNAMENT` | ON | better prefill shape |

**Butter:** `Ops.moeRouterTopKBiased` + Path B GPU already. Tournament / cast sink optional polish.

---

### 15. Layout / engine (not pure fusion)

| Item | mlxfast default | Butter |
|------|-----------------|--------|
| `POST_WIRE_COMMAND_BUFFER` (MLX 512 MB / 50 ops) | ON | N/A (butter CB model) |
| Compiled decode | ON (DARKBLOOM_COMPILED_DECODE) | N/A (hand Metal) |
| TRACE_FUSION | opt-in | add similar for A/B |
| ROPE_ANGLE_ATLAS | **OFF** | skip |

---

## Butter baseline map (quick)

| Area | Default behavior | Escape |
|------|------------------|--------|
| Path B NVFP4 | full eidx concurrent, fence-free | `BUTTER_LAGUNA_PATHB=0/hybrid` |
| Dual-queue shared | ON | `BUTTER_LAGUNA_DUAL_QUEUE=0` |
| Layer pipe | depth 4, layers/CB 2 | `LAYER_PIPE=0` |
| Prefill MoE | hybrid bf16/f32 MPP BM16 | `USE_GATHER_MPP=0` |
| Shared prefill fused gateup | ON | `SHARED_PREFILL_FUSED_GATEUP=0` |
| Dense NAX shared QMM | ON | `BUTTER_NVFP4_QMM_NAX=0` |
| SDPA prefill MMA | ON (T pad 32) | `SDPA_PREFILL_MMA=0` |
| Decode NVFP4 fuses | **OFF** | `PATHB_*` / `FUSED_SHARED_SWIGLU` |
| lm_head prune | **OFF** | `LM_HEAD_PRUNE=1` |
| Residual+RMS | ON (addAndRmsNorm) | width guard |
| Stacked QKV | ON | `SERIAL_QKV=1` |

---

## 3-sprint plan

### Sprint 1 — Decode (target: clean **≥100 t/s**, stretch 110–120 toward #1 131)

**Goal:** Close 85 → 100+ without breaking Paris/fib/long. Clean host: swap used ≪ 4 GB.

1. **Iron-bitexact shared + routed SwiGLU fuse**  
   - Fix `LagunaNvfp4Fuse` dequant to iron oracle.  
   - Implement concurrent multi-slot routed gate+up→SwiGLU that **beats** `nvfp4DequantGemvExpertIndexedMany` + `swigluMany`.  
   - Do **not** ship serial chain8 as default.

2. **QK-norm + RoPE fuse** (sliding + full YaRN)  
   - Port mlxfast barrier-free design; wire decode first.

3. **Gated o_proj fuse** + **residual+RMS+router** occupancy  
   - Finish softplus×O; router rows-per-group if router shows in GPU capture.

4. **Scheduling hygiene**  
   - A/B dual-queue vs single-queue early; pipe schedule with early fire after L1.  
   - Confirm residency set covers full NVFP4 tower.

5. **lm_head**  
   - Only if coarse kernel is faster than BF16 gemv; else leave opt-in.

**Exit criteria:** decode ≥100 clean; quality gates green; optional certified prune behind flag.

---

### Sprint 2 — Prefill kernels (target: **≥1000 t/s** interim, path to multi-k)

**Goal:** Attack MoE ~8× cliff (not ±5% attention nits).

1. **Gather kernel class:** iron BK=64 / true NAXTile or best BM16 MPP; fused sorted gate+up single gather.  
2. **Prefill MoE tail** fusion (weighted + scale + shared + residual).  
3. **Prefill QK-norm+RoPE** multi-token.  
4. **Shared expert** already ~12% — keep NAX + fused bank; avoid thrashing with dual-queue if MPP BW-bound.  
5. Capture/compare vs mlxfast gather path semantics (RUNSKIP-era fused gate/up).

**Exit criteria:** T=512 prefill ≥1000 t/s quality-OK; layer scan cos≈1 vs eidx; no late NaN.

---

### Sprint 3 — Submit-to-Yukon (optional)

Only if butter+iron is the submission vehicle (or a thin Yukon wrapper). mlxfast is already the Yukon-native stack.

1. **Correctness:** token-exact / golden gates matching serial-v2 contract.  
2. **Band strategy:** chunk ≤5% per ranked submission; don’t dump full wired ZH + all fusions at once.  
3. **Do not submit:** `FUSED_QKV` bank, `ROPE_ANGLE_ATLAS`, `PREFILL_ROUTER_TOP8`, unproven custom dequant.  
4. **Do submit stack:** barrier-free QK-RoPE, iron MoE fuses, residual/router, async ladder, dosed wired ZH, prefill fused gate/up + MoE tail.  
5. **Instrumentation:** `TRACE_FUSION`-style first-hit logs so silent fallback is visible.  
6. **Score realism:** beating #1 (131/4030) requires MoE kernels in the same league as MLX gather-NAX — Sprint 2 is the real Yukon gate; Sprint 1 alone cannot win score if prefill stays ~500.

**Exit criteria:** local paired score ≥1.0 vs on-box baseline with quality green; ranked band plan documented.

---

## Explicit non-goals / anti-patterns (from mlxfast scars)

- Do not re-introduce **TG barrier** after `simd_sum` for inv-RMS.  
- Do not assume prefill fuse that hurt pre-RUNSKIP still hurts — re-measure independently.  
- Do not default butter prune without coarse faster than BF16.  
- Do not treat dual-queue as free under swap thrash.  
- Serial 8-slot down fuse lost on M5 — concurrency > mega-kernel unless occupancy proven.

---

## Key source anchors

| Topic | Path |
|-------|------|
| Flag block + comments | `/Users/tom/dev/mlxfast-challenge/Sources/MLXFastModel/LagunaRuntimeModel.swift` ~90–476, ~2384–2419, ~4021–4072 |
| lm_head prune cert | `/Users/tom/dev/mlxfast-challenge/Sources/MLXFastModel/LagunaLmHeadPrune.swift` |
| Wired ZH | `/Users/tom/dev/mlxfast-challenge/Sources/MLXFastModel/LagunaRuntimeWeights.swift` ~417–560 |
| Butter fuse experiments | `/Users/tom/dev/butter-dev/Sources/ButterSwift/Models/Text/LagunaNvfp4Fuse.swift` |
| Butter prune | `/Users/tom/dev/butter-dev/Sources/ButterSwift/Models/Text/LagunaLmHeadPrune.swift` |
| Path B decode | `/Users/tom/dev/butter-dev/Sources/ButterSwift/Models/MoELayer.swift` ~3549–4134 |
| Layer residual/QK/RoPE | `/Users/tom/dev/butter-dev/Sources/ButterSwift/Models/Text/LagunaText.swift` ~1060–1186 |
| Campaign numbers | `/Users/tom/dev/butter-dev/benchmarks/laguna-xs/README.md` |
| Yukon score contract | `/Users/tom/dev/mlxfast-challenge/AGENTS.md` |

---

*Generated 2026-07-29. Research only; butter not modified.*
