# Laguna XS 2.1 — Butter chase board vs mlx.fast

Off-board absolute-speed chase on M5 Max. Not a Yukon submission.

## Model

- Primary (affine 4-bit, loadable today): `mlx-community/Laguna-XS-2.1-4bit` → `~/models/laguna-xs-2.1-4bit`
- Competition quant (NVFP4, native path TBD): `poolside/Laguna-XS-2.1-NVFP4-mlx`

## Window (match mlx.fast serial v2)

| Phase | Work |
|-------|------|
| Prefill | T=512 cold |
| Decode | 512 seed + 128 teacher-forced steps |

## MLX official baseline (pinned constants)

| Phase | s/tok | tok/s |
|-------|-------|-------|
| Prefill | 0.0003675 | ~2721 |
| Decode | 0.013856 | ~72.2 |

Internal score: `(base_dec/our_dec)^0.75 * (base_pref/our_pref)^0.25`

## Smoke

```bash
# after kernels regenerated + release build
swift run -c release butter generate \
  --model ~/models/laguna-xs-2.1-4bit \
  --prompt "Write a rust fib function" \
  --max-tokens 64 --stats
```

## Bench (once generate works)

Use `butter bench` methods that report prefill/decode tok/s, or a dedicated
window script. Record absolute tok/s + score vs baseline in a dated
`benchmarks/m5-max-YYYY-MM-DD.md` row labeled **affine4**, not NVFP4.

## Campaign log (M5 Max, affine4, `tom/feat/laguna-swift-xs`)

| When | Change | Prefill t/s | Decode t/s | Notes |
|------|--------|-------------|------------|-------|
| T0 baseline | per-token forwardMulti | 18.5 | 14.2 | simple bench, 13-tok prompt |
| T1 | batched forwardMulti + MoE decodeMany | **442** @370–701 tok | **~19–21** | simple short still ~29/20 (tiny T) |
| Board #1 (MLX NVFP4) | — | 3651 | 102 | other stack |
| MLX baseline | — | 2721 | 72 | score 1.0 |

Gaps after T1: prefill ~6–8× off board, decode ~5× off board.
Next: decode host-sync (softplus expand + sigmoid router), SWA sdpaMulti path, NVFP4.

### T2 session (decode Path B / MoE many)

| Config | Prefill | Decode | Quality |
|--------|---------|--------|---------|
| T0 serial everything | 18.5 | 14.2 | OK Paris |
| T1 decodeMany prefill | ~442 | ~20 | **BROKEN** empty |
| T2 batched attn + serial MoE (default) | ~29 short / ~27 @400 | **~21** | **OK Paris** |
| Path B GPU (`BUTTER_LAGUNA_PATHB=1`) | — | ~30–42 | **BROKEN** "skills" loop |
| Board #1 | 3651 | 102 | — |

**Blockers left:** fix `MoELayer.decodeMany` for sigmoid+int4 (prefill cliff), fix Path B GPU scores (decode 2×).

### T3 session (Path B full + gather prefill + depth-1 pipe)

| Config | Prefill | Decode | Quality |
|--------|---------|--------|---------|
| GPU softplus expand (no host) | ~30 short | ~25 | OK |
| Path B full expert-indexed + mid-MoE wait | — | ~26 | OK Paris |
| Path B full + ffnCopy residual + depth-1 pipe | ~86 short / **~510 @800** | **~60** | **OK Paris + fib** |
| decodeMany + MPP (default gather) | ~500 | — | **BROKEN** EOS |
| decodeMany + `BUTTER_MOE_BGEMM_NO_MPP=1` | **~490–510 @480–800** | ~60 | **OK** |
| Board #1 (MLX NVFP4) | 3651 | 102 | — |

**Defaults now:** Path B `full` (expert-indexed int4 + shared expert), decodeMany ON, force `NO_MPP` for Laguna, depth-1 layer pipeline, GPU softplus expand.

**Root causes fixed this session:**
1. Path B full garbage = residual reading `accumulatorScratch` while next use overwrites → **ffnCopy** before residual.
2. decodeMany garbage = auto **MPP** gather on M5 for Laguna affine4 → force simdgroup (`NO_MPP`).

**Gaps to board:** decode ~1.7× (60→102), prefill ~7× (510→3651). Next: drop more host fences, fix MPP for Laguna or NVFP4 path, lm_head prune, fused softplus×O, async CB overlap.

### T3b (Path B device-plan prefill)

| Config | Prefill | Decode | Quality |
|--------|---------|--------|---------|
| Path B d2 plan + tileplan | ~570 @800 | ~90 | **BROKEN** empty |
| Path B d2 plan + bm16 (no tileplan) | **~102 short / ~538 @800** | **~60** | **OK** |
| Board #1 | 3651 | 102 | — |

**Still shipping:** full Path B decode + ffnCopy + pipe depth 2, decodeMany ON, NO_MPP, Path B GPU device plan (no host route), tileplan off for Path B.

### T3c (concurrent expert-indexed GEMV)

| Config | Prefill | Decode | Quality |
|--------|---------|--------|---------|
| Concurrent int4 expert GEMV (default) | ~99 short / ~538 long | **~66.4** | **OK Paris** |
| Serial expert GEMV | ~100 | ~60 | OK |
| Board #1 | 3651 | 102 | — |

Gap decode: **1.54×** (66→102). Prefill: **~6.8×** (538→3651).

### T3d (fused finalNorm+lm_head + concurrent GEMV)

| Config | Prefill | Decode | Quality |
|--------|---------|--------|---------|
| + `rmsNormQgemvInt4Fast` lm_head | ~97 short / ~522 @800 | **~67** | **OK Paris** |
| Board #1 | 3651 | 102 | — |

Gap decode: **1.52×** (67→102). Prefill: **~7.0×** (522→3651).

### Latest (loop still open)

| Metric | Ours (affine4 quality OK) | Board #1 (NVFP4) | Gap |
|--------|---------------------------|------------------|-----|
| Decode | **~67–69 t/s** | 102 | **~1.5×** |
| Prefill @~800 | **~741 t/s** | 3651 | **~4.9×** |
| Prefill short | **~111 t/s** | — | — |

**T0 → now:** decode 14→68 (**4.9×**), long prefill 18→741 (**41×**). Quality: Paris + coherent fib.

**T3e:** prefill layer pipe + decodeMany `commitAtEnd:false` → **741 t/s @800**.

**Shipped this loop:** Path B full expert-indexed + ffnCopy, concurrent expert GEMVs, dual-queue shared expert, depth-2 CB pipe, GPU softplus expand, decodeMany + Path B device plan (NO_MPP, no tileplan), fused finalNorm+lm_head, fused residual+RMS.

**Next rocks (still looping):**
1. Certified lm_head prune (competition notes/68 — big decode)
2. Fix MPP gather for Laguna affine4 (prefill)
3. NVFP4 native path to match board quant

### T4 overnight (research + NVFP4 wiring)

Research (survey of fast open inference stacks): **#1 gap is NVFP4 vs affine4 bandwidth** + lm_head prune + single-cmdbuf polish.

Shipped:
- Fused QKV int4 decode (`batchedQkvQgemvInt4Fast`)
- NVFP4 Ops (`OpsNVFP4`, `Nvfp4Linear`, expert-indexed GEMV, gather BM16)
- MoELayer `StackedNvfp4Experts` + Path B NVFP4 decode
- Laguna loader NVFP4 switch_mlp / shared_expert (await weights download ~21GB)

Affine4 quality OK: **~70 decode / ~736 prefill@800** (still short of 102/3651).

### T5 overnight (NVFP4 board quant)

Weights: `poolside/Laguna-XS-2.1-NVFP4-mlx` → `~/models/laguna-xs-2.1-nvfp4` (21.6 GB; MoE experts NVFP4 U32+U8, attn/dense0/lm_head BF16).

| Config | Prefill | Decode | Quality |
|--------|---------|--------|---------|
| Path B NVFP4 + serial MoE prefill (default) | ~60 short / **~22 @482** | **~84** | **OK Paris + fib** |
| Batched MPP gather QMM | NaN logits | — | BROKEN |
| Host-plan gather QMM | finite, wrong top | — | BROKEN garbage |
| Expert-indexed sorted rows | finite, wrong top | ~84 | BROKEN |
| Board #1 (MLX NVFP4) | 3651 | 102 | — |

**Decode gap now 1.21× (84→102).** Prefill still ~5–100× off until gather is correct.

Shipped:
- NVFP4 load (`Nvfp4Linear`, stacked experts, Path B decode, dual-queue shared)
- NVFP4 QMM for shared-expert `callMany`
- Gather dispatch fixed to `_threadgroups` (was silent coop-tensor corruption) — still not quality-correct
- Default `BUTTER_NVFP4_GATHER!=1` → serial Path B prefill (quality)

**Next:**
1. Fix NVFP4 MoE gather quality (vs Path B GEMV reference)
2. MXFP8 lm_head sidecar files ready (`lm_head_mxfp8_*.u8`, cos≈0.998 vs BF16);
   load path SEGV under investigation — opt-in `BUTTER_LAGUNA_LM_HEAD_MXFP8=1`
3. Full certified prune (mlxfast `LagunaLmHeadPrune` / notes/68) after #2
4. Fuse shared/routed SwiGLU QMV like mlxfast DARKBLOOM kernels

**Score vs board (quality OK NVFP4 serial MoE):**
decode_speedup = 84/102 ≈ 0.82; prefill_speedup = 22/3651 ≈ 0.006
score ≈ 0.82^0.75 * 0.006^0.25 ≈ **0.17** (far below 1.0 — prefill dominates floor)

### Night baseline 2026-07-28 late (keep looping)

| Metric | Butter NVFP4 quality OK | Board | Gap |
|--------|-------------------------|-------|-----|
| Decode | **83.5–84.2 t/s** | 102 | 1.21× |
| Prefill short | ~62 | — | — |
| Prefill @482 | **~22 t/s** | 3651 @512 | ~170× |
| Quality | Paris + fib OK | — | — |

Env: default Path B full, serial MoE prefill (`BUTTER_NVFP4_GATHER!=1`), dual-queue shared.
MXFP8 lm_head sidecar: quality OK when forced, **slower** than BF16 gemv (80 vs 84) — need prune not full replace.
Gather QMM / sorted eidx: still quality-broken (not just dispatch).

### T6 concurrent token-order Path B (quality breakthrough)

**Root cause of garbage multi-token MoE:** `moeUnpermute` path and/or NVFP4 shared QMM — not the expert-indexed GEMVs themselves.

**Fix:** concurrent token-order Path B:
1. `gate.callMany` once
2. GPU `moeSigmoidBiasRows` + `moeRouterTopKBiasedMany`
3. Concurrent expert-indexed gate/up/down over T·topK (token order, no sort)
4. Per-token `scalarFMAChain8` (not unpermute)
5. Per-token shared GEMV (not QMM — QMM still breaks quality)

| Metric | Before (serial Path B) | **Now** | Board |
|--------|------------------------|---------|-------|
| Decode | ~84 | **~84** | 102 |
| Prefill @482 | ~22 | **~231** | 3651 |
| Prefill @802 | — | **~228** | — |
| Quality | Paris+fib | **Paris+fib+long OK** | — |

**Prefill 10× win**, still ~16× off board. Decode gap unchanged 1.21×.

Env: concurrent Path B default; `BUTTER_NVFP4_SERIAL_PATHB=1` escape; `BUTTER_NVFP4_USE_GATHER_QMM=1` still NaNs.

### T7 QMM grid fix + non-MPP gather

**Root cause of shared QMM / dense QMM garbage:** `dispatchThreads` grid was
`m*outDim` instead of `m*outDim*tpg` — Metal collapsed TGs so `tgid` was wrong.

Fixed `Ops.nvfp4Qmm` grid. Non-MPP `iron_nvfp4_gather_qmm` wired as
`Ops.nvfp4GatherQmm` (quality cos≈1.0 vs serial). MPP gather still NaNs.

| Path | Prefill @~1200 | Quality |
|------|----------------|---------|
| Expert-indexed concurrent (default) | **~243 t/s** | OK |
| Non-MPP gather QMM | ~228 t/s | OK |
| MPP gather | slower + NaN | BROKEN |

Shared QMM default ON after grid fix (quality OK).

| Metric | Now | Board | Gap |
|--------|-----|-------|-----|
| Decode | **~85** | 102 | 1.20× |
| Prefill @482–1200 | **~230–245** | 3651 | ~15× |
| Quality | Paris+fib+long | — | — |

**T7b:** Stacked dense QKV single GEMV for BF16 attn (~+0.5–1 t/s decode).  
**T7c:** Earlier dual-queue shared (overlap router+experts); pipe depth 3; max cmdbufs 256.

### T8 local grind (2026-07-29) — prune + SWA concurrent + expert-group QMM

| Change | Decode | Prefill @682 | Quality | Keep? |
|--------|--------|--------------|---------|-------|
| Baseline (T7) | ~84–85 | ~247 | OK | — |
| Fused MXFP8 lm_head prune (mlxfast-style coarse v2) | ~85–86 when warm | same | Paris+fib OK | opt-in `BUTTER_LAGUNA_LM_HEAD_PRUNE=1` (not default — coarse ≈ BF16 cost) |
| Full MXFP8 replace | ~80 | — | OK | worse |
| Expert-group dense QMM prefill | — | **slower** (~230 vs 247) | OK | opt-in only (`BUTTER_NVFP4_EXPERT_GROUP_QMM=1`) — host wait kills win |
| Non-MPP gather QMM | — | ~232 | OK | slightly slower than eidx |
| MPP BM16/BM64 | — | NaN logits (100% nan) | BROKEN | blocked |
| Concurrent SWA sdpaDecode prefill | ~85 | ~247 (no change) | OK | kept (encode cleaner; MoE still dominates) |
| Softplus expand-id cache (prefill) | — | small host save | OK | kept |
| Pipe depth 4 default (was 3) | ~85–86 | — | OK | **default** |

**Defaults now:** concurrent Path B eidx, dual-queue shared, pipe depth **4**, shared QMM ON, prune OFF, expert-group OFF, MPP OFF.

**Blockers to board:**
1. **Prefill ~15×** — need working MPP (or CSR-on-GPU expert QMM without host wait). MPP still all-NaN after threadgroups fix.
2. **Decode ~1.20×** — lm_head prune only ~+1 t/s; need fused NVFP4 expert SwiGLU / less per-layer CB fencing.

**Score vs board (quality OK):**
decode_speedup ≈ 85/102 ≈ 0.83; prefill_speedup ≈ 245/3651 ≈ 0.067
score ≈ 0.83^0.75 * 0.067^0.25 ≈ **0.35** (prefill still floors score).

### T8b fused shared SwiGLU QMV (mlxfast port)

`LagunaNvfp4Fuse` + `BUTTER_LAGUNA_FUSED_SHARED_SWIGLU=1`:
- Compiles; Paris still says Paris but **continuation drifts** (dequant not bit-identical to iron `nvfp4_qgemv` scale path).
- Decode ~same as baseline (~85) — shared expert is a small slice of MoE time.
- **Not default.** Next: match iron E4M3 scale decode exactly, then port **routed** fused SwiGLU + down-reduce (mlxfast DARKBLOOM_FUSED_ROUTED_*) for the real decode lever.

### T9 MPP kernel proven; full-model multi-token still broken

**Unit tests** (`Tests/ButterTests/Ops/Nvfp4MppSmokeTests.swift`):
- Synthetic same-expert BM16: cos=1 vs eidx
- Multi-expert sorted BM16: cos=1
- **Real Laguna gate weights** BM16 + BM64: cos=1
- Real down proj (kIn=512): cos=1
- Full gate+up+swiglu+down chain: finite, good range
- **moeSortPlan + gather + MPP** on real weights: cos=1

**Root cause of historical all-NaN:** BM64 auto-select on live short expert runs
(default now BM16 only; `BUTTER_NVFP4_GATHER_BM=64` opt-in).

**Full-model multi-token MPP** (`BUTTER_NVFP4_USE_GATHER_MPP=1`):
- Prefill **faster** (~0.45s vs 0.69s @162 tok ≈ **360 vs 235 t/s**)
- Generation text **empty** (tokens emitted but decode to nothing)
- Dump logits often all-NaN without host fences; with per-layer wait, MoE
  `outFlat` is finite every layer but quality still wrong
- Isolated MPP command buffer + classic sort + zeroed outs: still empty gen

**Default stays eidx concurrent Path B** (quality Paris+fib+long OK).

| Metric | Default (eidx) | MPP opt-in | Board |
|--------|----------------|------------|-------|
| Decode | **~85** | ~85 | 102 |
| Prefill @162–700 | **~235–245** | ~360 (broken quality) | 3651 |
| Quality | OK | **BROKEN** empty gen | — |

**Live A/B (BUTTER_NVFP4_MPP_DIAG=1):** first MoE layer `outFlat` vs eidx
**cos=1.0** on real residual — unpermute + Path B weights OK for layer 1.

### T10 MPP f32-promote — DEFAULT ON (quality OK)

**Late-layer NaN root cause:** bf16 MPP half coop tiles overflow when residual
`|h|` grows past ~400 by L37 (eidx float GEMV stays finite at `|h|~1e6`).
Layer scan: L38 first Inf/NaN on bf16 MPP; eidx L38 maxAbs~1e6 finite.

**Fix:** bf16 → cast f32 → f32 MPP → cast bf16 (`BUTTER_NVFP4_MPP_BF16=1` old path).

**Default:** MPP ON for T≥4 (`BUTTER_NVFP4_USE_GATHER_MPP=0` → eidx).

| Metric | eidx | **MPP f32 default** | Board | Gap |
|--------|------|---------------------|-------|-----|
| Decode | ~85 | **~85** | 102 | 1.20× |
| Prefill @162 | ~235 | **~329** | 3651 | ~11× |
| Prefill @482 | ~245 | **~402** | 3651 | ~9× |
| Quality | OK | **Paris+fib+long OK** | — | — |

Score ≈ 0.83^0.75 × 0.11^0.25 ≈ **0.50** (was ~0.35).

**Next:** decode fuse (mlxfast DARKBLOOM routed); more prefill (attn/SWA, BM64 only if wins).

### T11 local grind (2026-07-29 cont) — fuses + MPP cast + concurrent gate/up

**Machine note:** `ai.useraven.mlx-server` (Gemma-4-26B MLX on :8081) was thrashing
unified mem (swap ~17–33 GB). Unloaded for fair Laguna benches:
`launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/ai.useraven.mlx-server.plist`.
Decode numbers under residual swap: **~56–58 t/s** (not comparable to T10 ~85;
prefill still healthy). Re-check decode after reboot / swap drain.

| Change | Prefill @~500 | Decode | Quality | Keep? |
|--------|---------------|--------|---------|-------|
| Serial chain8 fused down (all slots in one TG) | — | **~54** (loss) | OK | opt-in `PATHB_NVFP4_FUSED=1` |
| Per-slot swiglu+down fuse | — | loss | OK | opt-in `SLOT_FUSE=1` |
| Gate+up→swiglu eidx fuse | — | ~parity/slight loss | OK | opt-in `GATEUP_FUSE=1` |
| MPP f32 scratch (no per-call `Tensor.empty`) | ~414 | — | OK | **default** |
| Shared cast + f32 gate/up/down + concurrent gate+up MPP | **~419–421** | — | Paris+fib OK | **default** |
| BM64 MPP | ~330 (slower) | — | OK | stay BM16 |
| eidx (no MPP) | ~249 | — | OK | MPP stays default |
| Long prefill @722 | **~425** | — | OK | — |
| + dual-queue shared expert prefill | **~430** @502 / **~431** @722 | ~58 | Paris+fib OK | **default** |
| dual0 (serial shared) | ~418 | — | OK | escape `DUAL_QUEUE=0` |
| blit.fill zero MPP outs (vs host memset) | ~430 (parity) | — | OK | **default** (cleaner CB) |
| Prefill gate/up/down/outFlat grow-only scratch | ~429 (parity) | — | OK | **default** (less driver thrash) |
| Early dual-queue shared (before router/sort) | ~429 (parity; shared already hidden) | — | OK | **default** |
| Prefill router/sort/token-src grow-only scratch | ~429 (parity) | — | OK | **default** |
| Hybrid MPP: bf16 layers ≤36, f32 late | ~429 (cast not the bottleneck) | OK Rayleigh+Paris+fib | **default** `MPP_BF16_MAX_LAYER=36` |
| All-f32 MPP (`MAX_LAYER=-1`) | ~427–428 | OK | escape |

**Defaults now:** hybrid early-bf16 MPP (≤L36) + late f32 promote, concurrent
gate+up MPP, early dual-queue shared prefill, blit zero outs, full prefill
scratch reuse, experimental decode fuses **OFF**.

| Metric | Now (T11, thrashy host) | Board | Gap |
|--------|-------------------------|-------|-----|
| Decode | **~57–58** (host-bound under swap; was ~85 clean) | 102 | ~1.8× |
| Prefill @500–1100 | **~426–431** | 3651 | ~8.5× |
| Quality | Paris+fib+Rayleigh OK | — | — |

Score (using clean decode 85 if host free, pref 430): ≈ 0.83^0.75 × 0.118^0.25 ≈ **0.52**.

**Shipped:** `LagunaNvfp4Fuse` experimental kernels; MPP intermediateScratch;
`nvfp4MoeGatherBm16Pair` concurrent; MoELayer f32 shared-cast + dual-queue
shared prefill; blit zero MPP outs; prefill matmul + router scratch reuse;
per-layer hybrid bf16/f32 MPP.

**Finding:** f32 promote cast is **not** prefill bottleneck (~same tok/s as
native bf16 early). MoE MPP / attn still dominate. mlxfast uses fused
sorted gate+up gather-QMM (~4% there); our concurrent pair is analogous.

**Restore after benches:**  
`launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.useraven.mlx-server.plist`

### Clean-host decode hygiene (swap thrash → 56–58 instead of ~85)

Decode is **host-sensitive** under unified-mem pressure. Do **not** kill
Claude/Chrome. Prefer unload of known heavy agents, then reboot if swap
stays sticky:

```bash
# 1) mlx-server (Gemma MLX on :8081) — OK to unload for Laguna benches
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/ai.useraven.mlx-server.plist 2>/dev/null
# restore: launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.useraven.mlx-server.plist

# 2) Optional local LLM servers (only if YOU started them for other work)
# ds4-server (M5 DeepSeek): pkill -9 -x ds4-server
# never pkill -f llama-server from inline ssh (self-kill trap)

# 3) Check pressure
sysctl vm.swapusage          # want used << 4 GB for clean decode numbers
memory_pressure | head -20   # free pages should stay high during generate

# 4) Sticky swap after unload: macOS often keeps swap *slots* allocated
# even with free RAM. A reboot is the reliable drain. After boot, re-check
# swap used before trusting decode t/s.
```

Fair decode gate used in this campaign: **swap used < ~4 GB** before
`butter generate --stats`. Prefill is less sensitive (still ~430–490 under
thrash); decode drops ~85 → ~56–58 when swap thrashing.

**Next:** clean-host rebench decode (reboot / drain swap — residual 17 GB
swapouts stick); wire `sdpaPrefillMma` (d128) for full/SWA prefill; GPU
segment-scan expert-group QMM; iron-bitexact routed fuse.

### T12 decode-path audit (2026-07-29) — host fences / prune readiness

Code inspection (no default flip without clean-host A/B):

| Item | Status | Notes |
|------|--------|-------|
| Path B NVFP4 decode host wait | **none** | `decodeGPUPathBNvfp4` stays on caller's CB; no mid-decode `waitUntilCompleted` |
| Hybrid Path B | host wait | only `BUTTER_LAGUNA_PATHB=hybrid` |
| CPU router fallback | host wait | when stacked experts / Path B gate fails |
| Concurrent expert GEMVs | **default ON** | `nvfp4DequantGemvExpertIndexedMany` concurrent encoder; escape `BUTTER_MOE_SERIAL_EXPERT_GEMV=1` |
| Dual-queue shared expert | **default ON** | +1 CB + 2 events × ~39 MoE/token; escape `BUTTER_LAGUNA_DUAL_QUEUE=0` (single-queue early encode) |
| Residual ffnCopy blit | **default OFF** | write residual into `ffnOutScratch`; escape `BUTTER_LAGUNA_FFN_COPY=1` |
| Layer pipe | depth **4**, `LAYERS_PER_CB` **2** | `LAYER_PIPE=0` → single-CB serial wait |
| Experimental NVFP4 fuses | **opt-in OFF** | GATEUP/SLOT/chain8/shared-swiglu — parity or loss on M5 so far |
| `lm_head` prune | **opt-in only** | `BUTTER_LAGUNA_LM_HEAD_PRUNE=1`; code: "wins are small until coarse is faster"; full MXFP8 replace slower (~80 vs ~84). **Do not default.** |
| Fused finalNorm+int4 lm_head | affine4 only | NVFP4 board uses BF16 `lm_head` gemv (or prune) |

**Decode gap to board (clean ~85 → 102):** residual CB/event overhead (dual-queue + pipe) + missing fused routed SwiGLU/down (mlxfast DARKBLOOM) + lm_head BW. Not a missing host-fence bug on the NVFP4 quality path.

**Memory (this session):** mlx-server already unloaded; no ds4-server. Swap used ~10–12 GB sticky after prior thrash — skip fair decode smoke until reboot drains swap below ~4 GB (see hygiene section above).

### T11c — `sdpaPrefillMma` for unwindowed prefill (T%32==0)

`LagunaSdpa.prefillMmaBf16`: bf16→f16 cast Q, compact bf16 KV→f16, Hy3-style
`Ops.sdpaPrefillMma` (d128), cast out back to bf16. Used when
`windowStart==0 && sinkEnd==0` (cold T=512 and full layers).

| Config | Prefill @512 (warm) | Quality | Keep? |
|--------|---------------------|---------|-------|
| **MMA ON (default)** | **~485–487** | Paris+fib+sky OK | **default** |
| MMA OFF (`SDPA_PREFILL_MMA=0`) | ~443–461 | OK | escape |
| Prior MPP-only ~502-tok | ~429 | OK | — |

**~5–9% prefill win** at board-shaped T=512. Escape: `BUTTER_LAGUNA_SDPA_PREFILL_MMA=0`.

| Metric | Now | Board | Gap |
|--------|-----|-------|-----|
| Prefill @512 | **~486** | 3651 | ~7.5× |
| Decode (thrashy) | ~57–58 | 102 | clean rebench needed |
| Quality | OK | — | — |

Score (clean dec 85, pref 486): ≈ 0.83^0.75 × 0.133^0.25 ≈ **0.54**.

### T11d — pad T to 32 for MMA

When `nRows ≥ 32` but not multiple of 32, pad Q/K to next 32 and discard
pad query outputs. Real queries keep causal range over real keys.

| Config | Prefill (warm) |
|--------|----------------|
| T=512 aligned MMA | **~490** |
| T=481 pad→512 MMA | **~471** (was ~450 without MMA) |
| Quality | Paris+fib OK |

Slight aligned win vs T11c (~490 vs ~486); unaligned shapes now get MMA.

### T11e — SDPA concurrent K/V compact + ring scratch

Concurrent compact of K‖V + ring-of-8 intermediateScratch temps (pipe-safe).
Prefill **~490** @512 (parity — MMA/MoE dominate, not compact). Quality OK.
Kept for cleaner encode + less driver alloc thrash.

### T11f — skip MPP out zero when mTotal%16==0

Most prefill shapes (T·8) are 16-aligned → skip 3 blit fills × 39 MoE layers.
Prefill still **~490** @512 (parity). Quality OK.

### T11g — prefill timing + shared expert cost (resume)

`BUTTER_LAGUNA_TIMING=1` (depth-1 wall):

| Bucket | ms @ T=512 | Share |
|--------|------------|-------|
| **MoE** | **~1157** | **~95%** |
| dense L0 | ~58 | ~5% |
| SWA layers | ~910 | (subset of MoE+attn) |
| full layers | ~306 | |

**Per MoE layer ~30ms.** Board 3651 t/s needs ~3.5ms/layer → **~8.5× MoE kernel gap**.

Shared expert alone: `NO_SHARED=1` → **~537 t/s** vs **~480** with shared (~12%). Dual-queue cannot fully hide BW-bound shared work (competes with MPP for memory). Concurrent shared gate+up QMM slight help (~480).

All-native-bf16 MPP (`MPP_BF16=1`): empty gen (late overflow). Hybrid ≤L36 stays.

BM64 / expert-group QMM still slower than BM16 MPP.

### T11h — dense NVFP4 QMM NAX for shared expert (default ON)

Wire `iron_nvfp4_qmm_nax_{f32,f16,bf16}` into `Ops.nvfp4Qmm` / `nvfp4QmmMany` when
`m % 32 == 0 && outDim % 32 == 0` (BM=BN=32, TPG=128, grid `[N/32, M/32]`).
Escape: `BUTTER_NVFP4_QMM_NAX=0`. Falls back to reduction TPG=64 otherwise.

| Config | Prefill @512 | MoE wall | Quality |
|--------|--------------|----------|---------|
| **NAX ON (default)** | **~447–452** | ~1062 ms | Paris + fib OK |
| NAX OFF | ~409 | ~1183 ms | OK |

**~10% prefill win** from shared-expert dense path (shared ≈12% of MoE). Dense L0
unchanged (~63 ms). Routed gather still BM16 MPP — **still the 8.5× cliff**.

**Root-cause research (parallel agents):** board uses MLX `fp_gather_qmm_rhs_nax`
BM=BN=BK=**64**, TPG=128, `tile_matmad_nax`. Iron has no NVFP4 gather-NAX twin
(`iron_nvfp4_qmm_nax_*` is MPP matmul2d, misnamed). Plan: `/tmp/laguna-moe-nax-plan.md`.
Live Metal gather-NAX spike + expert-group async dense NAX in flight.

| Metric | Now | Board | Gap |
|--------|-----|-------|-----|
| Prefill @512 | **~450** | 3651 | ~8.1× |
| Decode (thrashy swap ~10 GB) | ~40–50 | 102 | clean host ~85 needed |
| Quality | OK | — | — |

### T12 — decode thrash audit (no code change)

Default decode path already fence-free (Path B concurrent eidx, dual-queue shared,
FFN copy OFF, pipe depth 4). lm_head prune stays opt-in. Sticky swap is the
decode limiter on this host; reboot/drain for clean ~85 rebench.

### T13 — gather NAX-class spike + agent panel (2026-07-29)

**Shipped:**
1. Dense `iron_nvfp4_qmm_nax_*` default ON for shared expert (~10% prefill when clean).
2. `Nvfp4GatherNax.swift` + live Metal `butter_nvfp4_gather_qmm_rhs_nax_mpp_{bf16,f32}`  
   (BM=BN=64, TPG=128, MPP matmul2d 32×32×32 — **honest: not MLX steel NAXTile**).
3. Env: `BUTTER_NVFP4_GATHER_NAX=1` (opt-in). Pair path LIVE by default;  
   serial down needs `BUTTER_NVFP4_GATHER_NAX_LIVE=1` (quality gate).
4. BM64 full gate+up+down wiring fix (`BUTTER_NVFP4_MOE_BM64=1`).
5. Expert-group async dual-CB pipe (still host segment scan; slower than MPP).
6. Research plan: `/tmp/laguna-moe-nax-plan.md` — board is MLX `fp_gather_qmm_rhs_nax`  
   BK=**64** + `tile_matmad_nax`; iron BM16 is 8× more TGs + 4× more K steps.

**Measured (thrashy host, swap ~10 GB):**

| Config | Prefill @512 | Quality |
|--------|--------------|---------|
| Default BM16 MPP | ~330–380 | Paris OK |
| GATHER_NAX=1 (LIVE pair + iron down) | **~365–371** | Paris OK |
| Expert-group dense NAX | ~161 | Paris OK (slow) |

**Still the cliff:** MoE ~30ms/layer vs board ~3.5ms. Closing requires true  
board-class gather NAX (or iron twin with BK=64 + better dequant staging), not  
more dispatch micro-opts. Decode thrash ~50; clean host historically ~85 vs 102.

### T14 — iron `bm16_bk32` + hybrid BM64 down (iron + butter)

**Iron** (`iron-dev`): new kernel `iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp`
(BM=16 / BN=32 / **BK=32**, matmul2d 16×32×32). Iron unit tests f32/f16/bf16
PASS (cos err ≤2e-3). Emitted into `butter-dev/Sources/IronSwift`.

**Butter wiring (default ON):**
- Gate/up: materialise gather + concurrent **BK32** pair (escape `MPP_XROWS=1`)
- Down: hybrid **BM64** when `nOut ≥ 1024` (escape `HYBRID_BM=0`)
- Classic BK16: `BUTTER_NVFP4_GATHER_BK=16`

| Config | Prefill @512 (warm, thrashy) | Quality |
|--------|------------------------------|---------|
| **T14 BK32 + hybrid down** | **~406–408** | Paris OK |
| BK16 + xrows (escape) | ~419 | OK |
| Prior campaign clean | ~490 | OK |

**~parity under thrash** — BK32 not yet a clear win vs BK16+xrows on this host;
iron correctness is solid. Next: BK=64 iron gather, fused gate+up iron, clean-host
rebench, and sync kernel into `../iron` so `make regenerate-kernels` keeps it.

**Envs:**
| Env | Default | Meaning |
|-----|---------|---------|
| (BK32) | ON | iron `bm16_bk32` for BM16 path |
| `BUTTER_NVFP4_GATHER_BK=16` | off | classic BK16 |
| `BUTTER_NVFP4_HYBRID_BM` | ON | BM64 when nOut≥1024 |
| `BUTTER_NVFP4_MPP_XROWS=1` | off | prior xrows gate+up |
| `BUTTER_NVFP4_QMM_NAX` | ON | dense shared NAX |
| `BUTTER_NVFP4_GATHER_BK64=1` | **off** | iron BM64×BK64 stage (unit PASS; **full-model empty gen** — keep opt-in) |

### T15 — iron BM64 BK64 + shared scratch recovery

**Iron** `iron_nvfp4_moe_gather_qmm_bm64_bk64_mpp_*` (9/9 unit PASS):
stages BK=64 (dual SK=32 matmul2d; MPP forbids coop K=64). Wired opt-in
`BUTTER_NVFP4_GATHER_BK64=1`. Full Laguna: **empty Paris gen** — not default.

**Shared expert agent (scratch reuse):** prefill **~452 → ~520** by killing
per-layer `Tensor.empty` thrash (`prefillSharedExpertNvfp4` grow-only scratches).
Shared no longer a ~12% tax when warm. Still in tree.

| Config | Prefill @512 | Quality |
|--------|--------------|---------|
| Default (BK32 + hybrid + shared scratch) | ~360–410 thrashy | Paris OK |
| Shared-scratch session (agent) | **~516–522** | Paris+fib OK |
| BK64 all gather | ~320–340 | **BROKEN empty** |
| Board | 3651 | — |

### T16 — stabilize prefill defaults (BK32 + shared scratch + hybrid BM64)

**Code confirm (default path, no env flips):**
1. Shared expert grow-only scratches in `prefillSharedExpertNvfp4` (mid/down/gu/gate/up).
2. BM16 path uses iron `bm16_bk32` when `kIn%32==0`; escape `BUTTER_NVFP4_GATHER_BK=16`.
3. Hybrid BM64 when `nOut≥1024` (down proj); escape `BUTTER_NVFP4_HYBRID_BM=0`.
4. `BUTTER_NVFP4_GATHER_BK64` remains **OFF** (full-model empty gen).

**Build:** `swift build -c release --build-path /tmp/butter-prefill-bench-build --product butter`

**Quality (default):** Paris ✅ ("The capital of France is Paris.") + fib ✅ (coherent Rust `fn fib`).

**Prefill T=512** (`('hello world ')*255` → 512 toks), `BUTTER_LAGUNA_TIMING=1`, thrashy host (swap used ~10 GB, mlx-server up):

| Config | Run | Prefill t/s | MoE ms | layers ms |
|--------|-----|-------------|--------|-----------|
| **Default** BK32+hybrid+shared-scratch | 1 (cold-ish) | 388.0 | 1196 | 1312 |
| | 2 | **406.7** | 1183 | 1251 |
| | 3 | **403.0** | 1196 | 1263 |
| **A/B** `GATHER_BK=16 MPP_XROWS=1 HYBRID_BM=0` | 1 (cold) | 273.7 | 1779 | 1862 |
| | 2 | **422.9** | 1129 | 1202 |
| | 3 | **410.4** | 1171 | 1239 |

Warm default **~403–407 t/s**, MoE **~1183–1196 ms**. Warm A/B (classic BK16+xrows, all-BM16) **~410–423 t/s** — slight edge under thrash (same pattern as T14). Default stays BK32+hybrid for path stability / future BK64 iron twin; GATHER_BK64 still opt-in only.

| Metric | Now (T16 thrashy) | Board | Gap |
|--------|-------------------|-------|-----|
| Prefill @512 warm | **~405** | 3651 | ~9× |
| Decode short (thrash) | ~48–50 | 102 | clean-host ~85 needed |
| Quality | Paris+fib OK | — | — |

**Best env defaults (ship as code defaults; no env required)**
*(superseded by T19: hybrid BM64 is now OFF by default — see the current-defaults table at the end):*

| Knob | Default | Escape |
|------|---------|--------|
| BM16 BK | **32** (iron `bm16_bk32`) | `BUTTER_NVFP4_GATHER_BK=16` |
| Hybrid BM | **ON** (BM64 if nOut≥1024) | `BUTTER_NVFP4_HYBRID_BM=0` |
| Gate/up pair | materialise + BK32 concurrent | `BUTTER_NVFP4_MPP_XROWS=1` |
| GATHER_BK64 | **OFF** | `BUTTER_NVFP4_GATHER_BK64=1` (empty gen) |
| Shared expert scratches | grow-only ON | — |
| Dense shared NAX | ON | `BUTTER_NVFP4_QMM_NAX=0` |
| MPP gather | ON (T≥4) | `BUTTER_NVFP4_USE_GATHER_MPP=0` |

### T17 — multi-worktree agent swarm (mlxfast DARKBLOOM port)

Worktrees: `butter-wts/{butter-prune,butter-decode-fuse,butter-prefill-bench}`  
+ iron-dev BK64 quality agent. Merged into `butter-dev`.

| Port | Default | Merged measure (thrashy host) |
|------|---------|-------------------------------|
| **Certified lm_head 2-pass prune** | **ON** (`PRUNE=0` off) | Paris OK; decode ~55–56 |
| **Path B ROUTED_SWIGLU** | **ON** | fuse-off ~45 → fused ~54 (**+20%**) |
| **Path B DOWN_REDUCE** | **ON** | (bundled in fuse A/B) |
| Prefill BK32+hybrid+scratch | ON | ~403–407 @512 (worktree) |
| Iron BK64 dual-slab | n/a | **reverted to BM64 body** (empty gen fixed; not board-class BK anymore) |

**Port plan:** `benchmarks/laguna-xs/DARKBLOOM-port-plan.md` (from mlxfast mine).

**Yukon #1 reference:** ~131 decode / ~4030 prefill — still ~2.3× decode / ~10× prefill off.

**Next:** QK-norm+RoPE fuse; iron-bitexact routed fuse metallib; clean-host rebench.

### T18 — QK-norm+RoPE fuse (mlxfast DARKBLOOM port)

`LagunaFusedQKNormRoPE` in `LagunaSdpa.swift` — live Metal, one TG/head,
no threadgroup barrier (simd_sum inv-RMS). Wired for SWA decode
`head_dim=128` bf16/f16. Default ON (`BUTTER_LAGUNA_FUSED_QK_NORM_ROPE=0` escape).

| Config | Decode (agent, 128 tok) | Quality |
|--------|-------------------------|---------|
| fuse OFF | ~58.6 | OK |
| fuse ON | ~59.5 (**+1.5%**) | Paris+fib OK |

Matches mlxfast's small ranked gain (+1.7%) order of magnitude. Prefill/YaRN
untouched. Combined defaults (prune + Path B fuse + QK fuse) on thrashy host
~40–56 decode depending on swap.

### T19 — prefill hunt: hybrid BM64 OFF default (+5.5%)

Full report: `reports/butter-prefill-hunt.md`.

T14 hybrid (BM64 down when nOut≥1024) was a **prefill regression** vs all-BM16
at Laguna shapes. Flip: hybrid **opt-in** (`BUTTER_NVFP4_HYBRID_BM=1`); default
all-BM16 + BK32. Also grow-only scratch for gate/up materialise gather.

| Config | Prefill @512 warm | MoE ms | Quality |
|--------|-------------------|--------|---------|
| **Default all-BM16** | **~495–497** | ~981 | Paris+fib OK |
| Control `HYBRID_BM=1` | ~469–470 | ~1038 | OK |
| T16 thrashy hybrid-on | ~403–407 | ~1190 | OK |
| Board / Yukon #1 | ~4030 | — | — |

**+5.5% prefill** vs hybrid-on control. MoE still **~96%** of layer wall.
XROWS / DUAL0 / gather-scratch alone = warm parity. GATHER_BK64 still OFF.

**Next iron:** BM16×BK64 stage for mid=512 (not default BM64 path); true gather-NAX.
---

## Current defaults (as shipped — no env required)

| Feature | Default | Escape |
|---------|---------|--------|
| lm_head certified 2-pass prune | ON (MXFP8 sidecar + BF16 head present) | `BUTTER_LAGUNA_LM_HEAD_PRUNE=0` |
| Path B routed SwiGLU+QMV (8 experts) | ON | `BUTTER_LAGUNA_PATHB_ROUTED_SWIGLU=0` |
| Path B down-reduce | ON | `BUTTER_LAGUNA_PATHB_DOWN_REDUCE=0` |
| QK-norm+RoPE fuse (SWA decode) | ON | `BUTTER_LAGUNA_FUSED_QK_NORM_ROPE=0` |
| Dense NVFP4 QMM NAX (shared expert) | ON when m,out %32==0 | `BUTTER_NVFP4_QMM_NAX=0` |
| MoE gather MPP | ON for T≥4 | `BUTTER_NVFP4_USE_GATHER_MPP=0` |
| Gather tile BK | BK32 (iron `bm16_bk32`) when k%32==0 | `BUTTER_NVFP4_GATHER_BK=16` |
| Hybrid BM64 (nOut≥1024) | **OFF** (T19: prefill regression) | `BUTTER_NVFP4_HYBRID_BM=1` |
| GATHER_BK64 | OFF | `BUTTER_NVFP4_GATHER_BK64=1` |
| Gate/up xrows path | OFF (materialise + BK32 pair) | `BUTTER_NVFP4_MPP_XROWS=1` |
| Shared expert prefill scratches | grow-only + fused gate/up bank | `BUTTER_LAGUNA_SHARED_PREFILL_FUSED_GATEUP=0` |
| Shared SwiGLU fuse (decode) | OFF (regresses) | `BUTTER_LAGUNA_FUSED_SHARED_SWIGLU=1` |
| Serial chain8 / slot fuse | OFF | `BUTTER_LAGUNA_PATHB_NVFP4_FUSED=1` / `..._SLOT_FUSE=1` |

Agent reports for T14–T19 live under `reports/`.
