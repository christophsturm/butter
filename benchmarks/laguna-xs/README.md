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

### T20 - attention INT8 requant

Adds group-32 affine INT8 requant for the attention Q/K/V(+gate)/o_proj
weights into `LagunaLayer`. New file
`Sources/ButterSwift/Models/Text/LagunaAttnInt8.swift`:

* Load-time (lazy, once per layer, on the first decode call):
  re-quantize `q_proj`/`k_proj`/`v_proj` (+`g_proj` per-head gate, when
  its shape matches) BF16 -> group-32 affine INT8, then row-concatenate
  into ONE bank. `o_proj` gets its own standalone group-32 INT8 bank
  (different input width - it reads the post-SDPA output). Original BF16
  weights stay resident and authoritative; prefill (`forwardMulti`) is
  untouched.
* Decode: one batched group-32 INT8 QMV dispatch over the QKV(+gate)
  bank replaces the stock dense BF16 fused-GEMV path (and the separate
  `g_proj` GEMV, folded into the same dispatch); the gate's raw logits
  are sliced from the bank's tail and softplus-activated exactly as
  before. `o_proj` becomes a second INT8 QMV over the gated attention
  output.
* Both the quantize kernel and the batched QMV are hand-written
  live-compiled Metal (`LagunaAttnInt8.pso`, same pattern as
  `LagunaNvfp4Fuse`) - `iron_qmv_int8_fast_*` and the iron
  `quantizeAffine` quantize kernels both hardcode group_size=64 in
  their generated inner loops (see
  `QuantizedOpsValidation.validateAffineQuantize`), so neither is
  reusable for group-32.
* NVFP4 group-16 for layers ≥32 was **not implemented this pass** - the
  INT8 stage alone gave a modest, noisy-but-real
  win and the model-scale reasoning below suggested the added complexity
  wasn't worth another kernel; INT8 stays on all 40 layers.

Default ON. Escape: `BUTTER_LAGUNA_ATTN_INT8=0` (restores the stock BF16
QKV/o_proj decode path byte-for-byte).

| Run (fib prompt, 80 tok, `--stats`) | Decode (steady) | Quality |
|--------------------------------------|------------------|---------|
| INT8 ON (run 1) | 61.92 t/s | Paris + coherent rust discussion |
| INT8 ON (run 2) | 62.51 t/s | OK |
| INT8 OFF (run 1) | 61.40 t/s | OK |
| INT8 OFF (run 2) | 61.62 t/s | OK |

| Metric | Value |
|--------|-------|
| Decode ON (avg of 2) | **62.2 t/s** |
| Decode OFF (avg of 2) | 61.5 t/s |
| Decode delta | **+1.1%** |
| Prefill @512, INT8 ON (`BUTTER_LAGUNA_TIMING=1`) | 464.6 t/s (no regression vs ~411 floor) |
| GPU mem delta (banks resident) | +~1.5 GB (matches the computed ~1.6 GB for 40 layers of INT8 QKV+gate+o_proj banks - confirms all layers built) |

**Below the hoped-for decode win, and the honest reason why:** back-of-
envelope per-layer attention BF16 weight traffic (~38–76 MB depending on
`num_attention_heads_per_layer`) is actually larger than one layer's NVFP4
MoE traffic (~25–28 MB) on Laguna XS, so halving it via INT8 should matter
a lot on a bandwidth-bound decode step - but the measured wall-clock gain
is far short of that. M5 Max unified bandwidth (~400+ GB/s) comfortably
covers even the un-requanted ~1.5 GB/token attention traffic at ~62 tok/s
(~90 GB/s), which points at decode being latency- (dispatch-count / sync-
chain) bound rather than bandwidth-bound at this token rate - consistent
with the gain tracking the ~1 dispatch/layer removed (gate folded into the
QKV bank) more than the ~2x byte reduction. Left as a note for the next
lever: dispatch-count reduction (deeper fusion across layers, or
front-loaded async-commit boundary scheduling) likely has more
headroom here than further weight-traffic cuts.

Quality gates: `What is the capital of France?` -> "The capital of France
is Paris." (48 tok, coherent). Fib prompt -> coherent on-topic rust
discussion (not literal code within 80 tokens, same behavior with the
flag off - not a regression). No token-level correctness issue observed.
---

## Current defaults (as shipped — no env required)

| Feature | Default | Escape |
|---------|---------|--------|
| lm_head certified 2-pass prune | ON (MXFP8 sidecar + BF16 head present) | `BUTTER_LAGUNA_LM_HEAD_PRUNE=0` |
| Path B routed SwiGLU+QMV (8 experts) | ON | `BUTTER_LAGUNA_PATHB_ROUTED_SWIGLU=0` |
| Path B down-reduce | ON | `BUTTER_LAGUNA_PATHB_DOWN_REDUCE=0` |
| QK-norm+RoPE fuse (SWA decode) | ON | `BUTTER_LAGUNA_FUSED_QK_NORM_ROPE=0` |
| Attention group-32 INT8 requant (T20) | ON | `BUTTER_LAGUNA_ATTN_INT8=0` |
| Dense NVFP4 QMM NAX (shared expert) | ON when m,out %32==0 | `BUTTER_NVFP4_QMM_NAX=0` |
| MoE gather MPP | ON for T≥4 | `BUTTER_NVFP4_USE_GATHER_MPP=0` |
| Gather tile BK | BK32 (iron `bm16_bk32`) when k%32==0 | `BUTTER_NVFP4_GATHER_BK=16` |
| Hybrid BM64 (nOut≥1024) | **OFF** (T19: prefill regression) | `BUTTER_NVFP4_HYBRID_BM=1` |
| GATHER_BK64 | OFF | `BUTTER_NVFP4_GATHER_BK64=1` |
| Gate/up xrows path | OFF (materialise + BK32 pair) | `BUTTER_NVFP4_MPP_XROWS=1` |
| Shared expert prefill scratches | grow-only + fused gate/up bank | `BUTTER_LAGUNA_SHARED_PREFILL_FUSED_GATEUP=0` |
| Shared SwiGLU fuse (decode) | OFF (regresses) | `BUTTER_LAGUNA_FUSED_SHARED_SWIGLU=1` |
| Serial chain8 / slot fuse | OFF | `BUTTER_LAGUNA_PATHB_NVFP4_FUSED=1` / `..._SLOT_FUSE=1` |
| Gated o_proj fusion (T24) | ON | `BUTTER_LAGUNA_FUSED_GATED_OPROJ=0` |
| Decode SDPA GQA-pair + ALPHASKIP (T24) | ON when shape allows | `BUTTER_LAGUNA_SDPA_GQA2=0` |

Agent reports for T14–T19 live under `reports/`.

### T21 - residency/CB batching

Investigated whether Butter's residency-set story has a coverage gap
(full weight-wiring is worth real prefill/decode percentage points when
missing) and whether raising command-buffer batching budgets helps
prefill.

**Residency coverage - gaps found and fixed.** `Device.markWeightsResident`
(pins buffers into one `MTLResidencySet` on `commandQueue`) was already
correctly applied to: the main GGUF/safetensors weight bundle load
(`Model.load`, `GGUFTensorBundle.gpuTensorView`), and every `KVCache` /
`GDNStateCache` / `ConvStateCache` buffer. Four **decode-path fused
weight buffers** were allocated via raw `device.makeBuffer` /
`Tensor.empty` at load-time but never registered, so the driver
per-allocation-validated them on every decode dispatch:

1. `LagunaText.swift` - `qkvFusedWeight` (BF16 stacked Q/K/V bank for
   single-GEMV decode).
2. `LagunaText.swift` `lagunaLoadMxfp8LmHead` - the lm_head MXFP8
   sidecar `codes`/`scales` buffers (loaded straight from the sidecar
   `.u8` files, bypassing the GGUF mmap-view path that auto-wires).
3. `LagunaAttnInt8.concat` - the row-concatenated group-32 INT8 QKV(+gate)
   and o_proj banks (T20's attention requant, default ON).
4. `LagunaNvfp4Fuse.fuseGateUp` - the fused `[gate; up]` NVFP4 bank for
   the shared-expert SwiGLU path.

Also found `commandQueueB` (the secondary queue used by MoE gate‖up
parallel dispatch and Hy3's dual-stream decode) never had the residency
set attached at all - only `commandQueue` did, via `addResidencySet`.
Any weight-touching dispatch submitted on queue B was unpinned even
after `markWeightsResident` ran.

Fix: added `device.markWeightsResident([...])` at each of the four
buffer-creation sites (same pattern as the existing KV-cache call
sites), and `commandQueueB.addResidencySet(set)` alongside the existing
`commandQueue.addResidencySet(set)` in `Device.markWeightsResident`.
`kernels_extra.metallib` and the scratch slab / dequant-intermediate
pools were checked and are **not** weight data (compiled PSOs and
transient per-token scratch respectively) - no gap there.

**Command-buffer batching.** Butter's only prefill-relevant knob is
`BUTTER_MAX_COMMAND_BUFFERS` (`IronLibrary.defaultMaxCommandBufferCount`,
default 256) - it caps *in-flight command-buffer count* on the queue
(backpressure), which is architecturally different from an
auto-commit-by-size/op-count buffer policy (Butter's own code comment already
notes this is a deliberate, orthogonal choice since Butter fires many
small per-layer command buffers rather than one large batched buffer).
`BUTTER_LAGUNA_TIMING=1` at T=512 shows MoE dominates prefill wall time
(~95%, 1007.7ms of a 1060ms per-forward layer total) - CB-count is not
the prefill bottleneck at this scale.

A/B on prefill @512 (`BUTTER_LAGUNA_TIMING=1`, `python3 -c "print('hello
world '*255)"`, 2 runs each, same session): `BUTTER_MAX_COMMAND_BUFFERS`
= 64 / 256 (default) / 512 all landed at **~560–564 tok/s**, no
measurable difference - confirms prefill isn't in-flight-CB-depth bound
here. No code change from this axis.

| Metric (fib prompt, 80 tok, decode steady) | Pre-fix (2 runs) | Post-fix (4 runs across 2 build sessions) |
|---|---|---|
| tok/s | 60.48, 61.72 (avg 61.10) | 61.93, 61.98, 62.20, 62.07 (avg 62.05) |
| Delta | - | **+1.6%** |

| Prefill @512 (`BUTTER_LAGUNA_TIMING=1`) | tok/s |
|---|---|
| Session A (cold-ish) | 479.05–482.59 |
| Session B (same code, later, warmer) | 559.46–563.83 |

Prefill numbers swing ~15% run-to-run within the *same* build depending
on session warmth/disk-cache state, swamping any signal from the
residency fix (expected - the four fixed buffers are decode-only;
`LagunaAttnInt8`'s own file header notes prefill/`forwardMulti` never
touches that path). No prefill regression observed either way; both
sessions comfortably clear the ~460 t/s floor.

Quality gates: `What is the capital of France?` (48 tok) -> "The capital
of France is Paris." repeated, coherent. Fib prompt (80 tok) -> coherent
on-topic rust discussion, same behavior as baseline. No correctness
regression.

**Verdict:** small, real decode win (+1.6%) from closing a genuine
residency gap (Butter's residency-set discipline was ~90% there already
- KV/state caches and main weights were correctly wired; only the newer
decode-fusion side-buffers from T20/T21-era work had been missed). CB
batching-depth axis is a dead end at this scale - no knob change shipped.

### T22 - NAX gather QMM (true BK=64 staging; correctness-blocked, OFF)

Attempted a true BK=64 NAX-class gather: amortize NVFP4 dequant + barrier
overhead over a 64-wide K tile instead of 32, into butter's MPP
`matmul2d`-based MoE gather kernel family.
**Ships env-gated OFF** (`BUTTER_NVFP4_GATHER_NAX64=1`); default
behavior is byte-for-byte unchanged. This section documents what was
built, what's verified, and the real wall hit at full-model scale.

**What shipped:**

- Two new hand kernels, `butter_nvfp4_gather_qmm_rhs_nax64_mpp_{bf16,f32}`
  (`Sources/IronSwift/Resources/kernels/`). Each 64-wide K tile stages
  TWO independent contiguous 64×32 half-blocks (`Xs[0]`/`Ws[0]` cols
  [0,32), `Xs[1]`/`Ws[1]` cols [32,64)) under **one** barrier pair, then
  runs both SK=32 `matmul2d` calls back-to-back - vs the existing BK=32
  spike kernel's one block under two barrier pairs. The K-offset
  sequence into the accumulator (0, 32, 64, 96, ...) is unchanged, so
  this is a scheduling change, not a numerical one relative to that
  spike.
- Memory-budget fix folded into the same kernels: the first version
  declared `Xs[2][2048]` + `Ws[2][2048]` + `OutScratch[4096]` as
  separate threadgroup arrays (32768B - exactly the 32KB Apple-GPU
  compute-threadgroup ceiling, zero headroom for whatever `matmul2d`
  needs internally). `Xs`/`Ws` are only live before the K-loop's
  closing barrier; `OutScratch` is only live after it - so they now
  alias the same 16384-byte backing store (`(threadgroup half*)
  OutScratch`), halving total explicit threadgroup usage to 16384B.
  Verified NOT the root cause of the correctness wall below (identical
  failure before and after); kept anyway as a strict improvement.
- Swift wiring: `Ops.nvfp4MoeGatherNax64` / `...Pair`
  (`Sources/ButterSwift/Ops/Nvfp4GatherNax.swift`), threaded into the
  existing `Ops.nvfp4MoeGatherBm16(Pair)` dispatch - the same entry
  points `MoELayer` already calls for gate/up/down - so no `MoELayer`
  changes were needed to cover all three projections generically
  (`kIn % 64 == 0` gate/up=2048, down=512 on Laguna XS).
- Repo hygiene fix: `Sources/IronSwift/Resources/kernels/` was
  *entirely* gitignored (regenerated by `iron build --emit all`), which
  silently dropped every hand-written `butter_*.metal` MPP live-compile
  source from git - including the pre-existing T13 NAX spike and fused
  gate/up kernels, not just this work. Added a `!butter_*.metal`
  allowlist exception; those 8 kernels are now tracked.

**Kernel-level correctness - thorough, all passing.**
`swift test --filter Nvfp4GatherNax64` (11 tests, 4 suites, ~23s):
synthetic same-expert; real Laguna gate/down weight slices with expert
runs straddling the 64-row tile boundary (rows 63/64, 127/128, 191/192
checked individually) and zero-count experts; the concurrent Pair
dispatch with both bf16 and f32 input (the two code paths `MoELayer`
actually calls for native-bf16 vs f32-promoted layers); a poisoned-
buffer write-coverage check (NaN-fills the output first, confirms every
element gets overwritten); a 20-layer chained-command-buffer run with
real `device.intermediateScratch` reuse (matches `MoELayer`'s scratch
pattern); and an all-39-real-layers × gate/up/down NaN scan. Every one
of these: cos > 0.9999, zero NaN, vs the expert-indexed GEMV oracle.

**Full-model wall: real 39-layer Laguna forward goes all-NaN.**
`butter generate --model ~/models/laguna-xs-2.1-nvfp4 --prompt "What is
the capital of France?" --dump-prefill-logits` with
`BUTTER_NVFP4_GATHER_NAX64=1` produces **all-NaN prefill logits**
(confirmed via the CLI's `--dump-prefill-logits` diagnostic; the
`--verbose` top-5 diagnostic is a different, non-representative code
path and misleadingly showed matching finite logits - don't trust it
for this class of bug). Reproduced identically with
`BUTTER_NVFP4_MPP_BF16_MAX_LAYER=-1` (forces f32-promote on every
layer), ruling out a bf16-only cause.

**Root-caused, not just localized.** A temporary env-gated debug hook
in `MoELayer.decodeManyNvfp4` (`BUTTER_NVFP4_NAX64_DEBUG=1`, forced a
sync commit+wait+readback per layer; reverted before commit - see git
history of this section if it's ever needed again) pinpointed the
break to **exactly one row** of layer 38's `down_proj` output: layer
1–37 clean (`nan=0 inf=0` every layer), layer 38 shows `nan=122 inf=1926`
(= exactly 2048 = one full `hidden`-wide row), layer 39 is fully
corrupted (cascade). The one bad row was routed to **expert 161**.

A magnitude sweep through layer 38's *real* expert-161 gate/up/down
weights (`Tests/ButterTests/Ops/Nvfp4GatherNax64MagnitudeTests.swift`)
reproduces it directly: NAX64 goes non-finite once input activation
magnitude reaches **~50**, while the expert-indexed GEMV oracle stays
perfectly finite (cos=1.0 on the finite subset) up to at least 1600.

**The critical result: the OLD, already-shipped BK=32 NAX spike kernel
(`butter_nvfp4_gather_qmm_rhs_nax_mpp_*`, live via
`BUTTER_NVFP4_GATHER_NAX_LIVE=1`) shows byte-for-byte IDENTICAL
NaN/Inf counts at every magnitude in the same sweep.** This is not a
NAX64-specific bug - every MPP kernel in this family (BM16, BM64,
BK32-NAX, and this BK64 attempt) stages both the activation and the
dequantized weight into Metal `half` for the actual `matmul2d` call
(the `f32` kernel variants only change the *buffer* dtype at the
kernel boundary, not the internal tile precision), and `half`'s
~65504 ceiling is easy to blow past mid-product at NVFP4's dequant
range × real late-layer residual magnitude - a fragility the existing
code comments already flagged (`OpsNVFP4.swift`: "late Laguna layers
grow residual |h| past ~400 and half A*B overflows"). The *existing*
default pipeline (BM16 gate/up, GEMV-heavy) apparently never happens
to land a token on this specific cliff for tested prompts; substituting
**any** differently-rounded kernel for 37 layers of gate/up/down changes
the accumulated residual-stream trajectory by enough floating-point
noise to occasionally tip one token over it by layer 38. The BK=32
spike surviving this exact prompt is not evidence it's immune - it's
evidence its particular rounding didn't happen to cross the line
*this time*, on a shared cliff both kernels stand on.

**Verdict:** the BK=64 restructuring itself is correct (11/11 kernel
tests, matching the already-shipped kernel's numerics exactly at every
tested magnitude). The blocker is a **pre-existing, shared numerical
fragility** in the whole `half`-staged MPP gather-kernel family that
this port exposes rather than causes. Shipping NAX64 default-on would
be gambling on which specific rounding trajectory a given prompt
crosses the cliff on - not acceptable. Kept **env-gated OFF**
(`BUTTER_NVFP4_GATHER_NAX64=1`, opt-in only); default `butter generate`
behavior (Paris/fib gates, release build) reconfirmed byte-identical
to pre-T22 after all changes above landed.

**Real fix (future work, out of scope for this pass):** stop staging
into `half` for the matmul tiles - either (a) verify MPP `matmul2d`
supports `float,float->float` cooperative tensors and use them for
late layers (the ones already flagged for f32-promote), which would
need a second kernel variant and a perf re-check (f32 tiles are 2×
the register/bandwidth cost of half), or (b) clamp/rescale the
dequantized `half` values into a safe range before the matmul with a
compensating rescale on the accumulator (cheaper, but is an
approximation, not exact). Either fix belongs to the *entire* MPP
gather family (BM16/BM64/BK32/BK64), not just this port - worth its
own investigation pass rather than folding into T22.

**Files:** kernels + `Nvfp4GatherNax.swift` wiring in the Stage A/B
commit; kernel tests in
`Tests/ButterTests/Ops/Nvfp4GatherNax64{,Scan,Chain,Magnitude}Tests.swift`
(all `.serialized` - they share process-global env-var dispatch
switches via `setenv`, which raced under Swift Testing's default
concurrent execution before that trait was added).


### T23 - scale-deferred staging fix (removes the half-overflow cliff family-wide)

Follow-up to T22. Fixes the half-staging overflow that blocked the whole
MPP gather kernel family at full-model scale, and flips **on**
`BUTTER_NVFP4_GATHER_NAX64=1` as a working, correctness-verified opt-in
(default behavior unchanged - see the perf verdict below for why it
stays opt-in rather than becoming the new default).

**Root cause (recap from T22):** every MPP gather kernel in this family
(`butter_nvfp4_gather_qmm_rhs_nax_mpp_{bf16,f32}`,
`butter_nvfp4_gather_qmm_rhs_nax64_mpp_{bf16,f32}`, and the default
`butter_nvfp4_moe_gather_bm16_mpp_xrows_bf16` - **not** its `_f32`
sibling, which turns out to already use genuine `threadgroup float`
tiles internally, see below) stages both the dequantized NVFP4 weight
value and the raw activation value into Metal `half` before the
`matmul2d` cooperative-tensor call. Once real late-layer (SwiGLU-
amplified) activation magnitude grows large enough, an individual `A*B`
product rounds to `Inf` inside MPP's half-compute-type multiply, even
though the accumulator itself is `float`.

**Fix shipped (Option A from the plan - exponent-bias fold, generalized
to both operands):** fold a fixed power-of-two bias into *both* the
staged weight value and the staged activation value before the `half`
cast, and multiply the compensation back into the `float` accumulator
once at the final output write. Because the bias is a compile-time
constant (not data- or group-dependent), it factors out of the entire
K-reduction exactly:
`sum_k((a_k·B)·(b_k·B)) == B² · sum_k(a_k·b_k)`. This is a standard
`metal::tensor`-level staging technique in spirit (stage narrow,
compensate at the boundary), derived and tuned independently for this
kernel family's own overflow behavior.

```metal
constant float kNvfp4StageBias = 0.00006103515625f;  // 2^-14
constant float kNvfp4StageComp = 268435456.0f;        // 2^28 == 1/(bias*bias)
// staging:
Xs[...] = half(xv * kNvfp4StageBias);
Ws[...] = half(dequant(w) * scale * kNvfp4StageBias);
// output:
out[...] = bfloat(OutScratch[...] * kNvfp4StageComp);
```

**The bias value was reached empirically, not derived a priori, and by
balancing two gates that pull in opposite directions:**

1. The real-weight magnitude sweep (new permanent test, see below)
   needs bias large enough (more headroom) to stay finite at high
   activation magnitude.
2. The *pre-existing* real-Laguna-weight tests in
   `Nvfp4GatherNax64Tests.swift` (gate/down/sort-plan, real trained
   weights at normal/typical activation magnitude) need bias small
   enough (less precision loss) to stay above their own `cos>0.9999`
   gate.

A bias small enough to preserve full precision at typical magnitude
(`2^-12`) still overflows past activation magnitude ~400–800. A bias
big enough to survive activation magnitude 1600 (`2^-15` or smaller)
pushes the real gate/down weights' typically-small dequantized values
into `half`'s subnormal range and drops the pre-existing tests from
`cos~1.0` to `~0.9998` - a real regression, not just a synthetic-sweep
concern (first attempt shipped `2^-15`, which flipped two pre-existing
`Nvfp4GatherNax64Tests` real-weight tests from `cos~1.0` to `0.999843`
/ `0.9998533`, both below their `0.9999` gate - caught by running the
existing suite, not assumed passing). `2^-14` is the largest bias that
keeps every real-weight test at `cos~1.0` while extending the
finite-safe range from T22's observed **~50** cliff to **at least 800**
- 16x headroom over the actual bug. A single fixed bias cannot cover
unbounded activation magnitude without this precision tradeoff; full
coverage to 1600+ would need per-tile/per-magnitude adaptive scaling
(Option C from the plan - f32 tiles for late layers - remains the
documented fallback for that last stretch, out of scope for this pass).

**`butter_nvfp4_moe_gather_bm16_mpp_xrows_f32` needed no fix.**
Inspecting it while patching its `bf16` sibling: this Iron-generated
kernel already declares `threadgroup float xs`/`ws` and instantiates
`matmul2d`'s cooperative tensors as `<float, float, float>` - genuine
f32 tiles throughout, not a `half`-staged kernel with a wider buffer
dtype at the boundary (unlike the NAX/NAX64 family, where the `_f32`
variant *only* changes the device-buffer dtype and still stages into
`half` internally - exactly the trap T22 flagged). This is why the
existing "promote bf16 → f32 MPP for late layers" escape in
`OpsNVFP4.swift` genuinely dodges the overflow for the *default* BM16
path already; the NAX/NAX64 family's superficially-similar "f32
promote" doesn't, because its f32 variant was never actually f32 inside
the kernel. Left `_f32` untouched.

**Permanent regression test added:**
`Tests/ButterTests/Ops/Nvfp4GatherNax64MagnitudeTests.swift`'s sweep
now asserts (previously only printed) `nan==0`, `inf==0`, and
`cos>0.9999` at every swept magnitude, for both the NAX64 kernel and
the old BK=32 NAX kernel, plus sanity-checks the f32 GEMV oracle itself
stays finite over the same range. Ramp capped at `[1, 10, 50, 100, 200,
400, 800]` (dropped 1600 from T22's exploratory sweep) - see the
bias-tuning tradeoff above for why. Sweep table at the shipped bias
(`2^-14`), real layer-38 expert-161 weights:

| magnitude | NAX64 nan/inf | OLD(BK32) nan/inf | oracle nan/inf | cos(finite) |
|---|---|---|---|---|
| 1 | 0/0 | 0/0 | 0/0 | 0.9999986 |
| 10 | 0/0 | 0/0 | 0/0 | 1.0000001 |
| 50 | 0/0 | 0/0 | 0/0 | 1.0 |
| 100 | 0/0 | 0/0 | 0/0 | 0.99999994 |
| 200 | 0/0 | 0/0 | 0/0 | 1.0 |
| 400 | 0/0 | 0/0 | 0/0 | 0.99999994 |
| 800 | 0/0 | 0/0 | 0/0 | 1.0 |

NAX64 and OLD(BK32) show byte-identical nan/inf counts at every
magnitude throughout the tuning sweep (as in T22) - confirms the fix is
a property of the shared staging pattern, not kernel-specific.

**Kernel-level test suite: 18/18 passing, zero regressions.** This
machine's `swift test --filter Nvfp4` combined run is flaky
independent of this change - Swift Testing's default concurrency
starts 5–6 real-weight-loading tests at once, several of which mmap
sizeable slices of the ~41GB NVFP4 checkpoint; on a shared workstation
this intermittently exceeds available memory and the test process is
killed with no crash log (reproduced identically on pre-fix/unmodified
code - confirmed via `git stash` A/B - so this is a pre-existing
environment/harness limitation, not something this change caused).
Worked around by running each suite individually instead of the
combined filter; every suite passed cleanly that way, including with
all 5 real-weight `NVFP4 MPP smoke` tests running concurrently (their
own default within-suite behavior) in 19.8s:

| Suite | Tests | Result |
|---|---|---|
| `NVFP4 MPP smoke` | 7 | ✔ all pass |
| `NVFP4 NAX64 gather QMM` | 6 | ✔ all pass |
| `NVFP4 NAX64 chained multi-layer diagnostic` | 2 | ✔ all pass |
| `NVFP4 NAX64 all-layer NaN scan` | 2 | ✔ all pass, `bad=[]` across real 39 layers × gate/up/down |
| `NVFP4 NAX64 magnitude sweep (T22 repro)` | 1 | ✔ pass (new permanent assertions) |

**Full-model validation, `BUTTER_NVFP4_GATHER_NAX64=1`, release build:**

- `--dump-prefill-logits` on `"What is the capital of France?"`: 100352
  logits, **0 NaN, 0 Inf**, range `[-10.06, 21.13]` - the exact
  diagnostic T22 reported as all-NaN now comes back fully finite.
- Paris (48 tok): *"The capital of France is Paris. Paris is located in
  the north-central part of the country. Paris is also the most
  populous city in France. It is a major center for finance, culture,
  and communication.\n\nAnswer: Paris."* - coherent, correct.
- Fib (80 tok): complete, correct, idiomatic iterative Rust
  `fn fibonacci(n: u32) -> u64`.
- 552-token filler prompt (`"hello world "*255`-style, 512+ tokens) +
  20-token continuation: coherent, on-topic, no repetition collapse or
  garbage.
- `BUTTER_NVFP4_GATHER_NAX=1` (old BK=32 spike, same staging fix
  applied): Paris also correct and coherent at the full model - this
  kernel was equally blocked per T22's "byte-for-byte identical
  NaN/Inf" finding and is equally fixed now.

**Default-path byte-identical, confirmed properly (not just claimed):**
first A/B used default `--temperature` sampling and looked different
pre/post-fix - turned out to be sampling non-determinism (no seed / T>0
by default), not a real change: repeating the *same* build twice at
default temperature gave two different continuations. Redid the A/B
with `--temperature 0` (greedy/deterministic): pre-fix and post-fix
Paris output is **byte-identical**, and pre-fix and post-fix fib output
is **byte-identical**, both via `git checkout` isolating just the
`bm16_mpp_xrows_bf16` kernel file and rebuilding. Confirms the fix is a
true no-op on tested default-path prompts (their residual magnitude
never reaches the cliff this fix protects against) while removing the
underlying fragility.

**Perf A/B, prefill @512** (`"hello world "*255`, 512 tokens,
`BUTTER_LAGUNA_TIMING=1`, 2 runs each, same session):

| Config | Run 1 | Run 2 | tok/s (avg) |
|---|---|---|---|
| Default (BM16) | 478.5 | 474.1 | ~476.3 |
| `BUTTER_NVFP4_GATHER_NAX64=1` | 449.1 | 506.9 | ~478.0 |

**No measurable prefill improvement** - the two configs land within
~1% of each other, well inside the ~15% run-to-run noise floor T21
already documented on this machine. The theoretical win NAX64 targets
(amortize dequant + halve barrier count over a 64-wide K tile) doesn't
show up at this scale; MoE compute, not staging/barrier overhead,
dominates the prefill wall (consistent with T21's own `layers=X moe=Y`
breakdown showing MoE ~95% of per-layer time). Per the plan's own gate
("flip default only if quality passes AND prefill improves >10%"): **default
NOT flipped.**

**Verdict:** the half-staging overflow is fixed across the entire MPP
gather kernel family (NAX BK=32, NAX64 BK=64, default BM16-bf16) with a
single exact, well-tested technique; both previously-blocked kernels
now produce coherent, correct, full-model output with zero regression
on 18/18 existing + new kernel tests and byte-identical default-path
generation. `BUTTER_NVFP4_GATHER_NAX64=1` and `BUTTER_NVFP4_GATHER_NAX=1`
are both real, usable, quality-verified opt-ins now (previously
correctness-blocked, unusable at any setting). They stay opt-in rather
than default because - now that correctness is no longer the blocker -
the measured prefill win is ~0%, not the hoped-for step toward the ~8x
throughput gap to close. Closing that gap needs a different lever than this
kernel-geometry change; NAX64's BK=64 restructuring itself was already
confirmed correct in T22, this pass only removed the precision wall
that had been masking the (non-)result.

**Files:** kernel changes in
`Sources/IronSwift/Resources/kernels/butter_nvfp4_gather_qmm_rhs_nax{,64}_mpp_{bf16,f32}.metal`
and `butter_nvfp4_moe_gather_bm16_mpp_xrows_bf16.metal`; permanent
regression assertions in
`Tests/ButterTests/Ops/Nvfp4GatherNax64MagnitudeTests.swift`.

### T24 - gated o_proj fusion (overnight dispatch-count pass)

Follow-up to T20's finding that decode is latency/dispatch-count bound,
not bandwidth-bound, on this box. Adds two gated o_proj fusions into a
new file, `Sources/ButterSwift/Models/Text/LagunaGatedOProjFuse.swift`, hand-written
live-compiled Metal via the same `pso()` pattern as `LagunaNvfp4Fuse` /
`LagunaAttnInt8` (per those two files' own precedent - butter's `iron`
DSL has no primitive for this shape, so this stays runtime Metal rather
than a DSL port).

Two kernels, routed to whichever `o_proj` path a layer's T20
`ensureAttnInt8()` build landed on:

1. **`laguna_gate_product_softplus_bf16_h{48,64}`** - one dispatch computes
   the numerically-stable softplus of the *raw* per-head gate logit (same
   op stream as `iron_softplus_bf16`), rounds to BF16 where `Ops.softplus`
   already rounds, then broadcast-multiplies against the raw SDPA row and
   rounds to BF16 where `Ops.mul` already rounds - reproducing the stock
   `softplus → gather-expand → mul` chain (3 dispatches) bit-exactly in 1.
   Feeds the T20 INT8 `o_proj` QMV bank unchanged. **This is the path the
   default config (`BUTTER_LAGUNA_ATTN_INT8` ON) actually exercises.**
2. **`laguna_gated_output_projection_bf16_h{H}_u2`** - plain BF16 `o_proj`
   path only (INT8 bank absent/declined, i.e. `BUTTER_LAGUNA_ATTN_INT8=0`):
   folds the per-head gate multiply directly into the o_proj GEMV's column
   loads (unroll depth 2), so the
   `[nHeads·headDim]` gated row never materializes. Needs an already-
   activated (post-softplus) `[nHeads]` gate - a cheap `Ops.softplus` over
   `nHeads` elements runs first, replacing the O(hidden)-wide gather+mul.
   This kernel's own K-reduction order differs from the stock GEMV's, so
   - like T20's INT8 QMV - it is **not** claimed byte-identical, only
   correctness/coherence-verified (see below); kernel 1 above *is* claimed
   and verified byte-identical.

Escape: `BUTTER_LAGUNA_FUSED_GATED_OPROJ=0` restores the stock chain.
Default ON.

**Correctness (temperature 0, greedy, same prompt, byte comparison):**

| Config | Fused ON vs OFF |
|---|---|
| Default (`BUTTER_LAGUNA_ATTN_INT8` ON → kernel 1 path) | **byte-identical** output, "What is the capital of France?" (48 tok) |
| `BUTTER_LAGUNA_ATTN_INT8=0` (→ kernel 2 path) | Diverges after "The capital of France is Paris." - expected FP non-associativity from a different GEMV reduction order (same class of difference as T20's INT8 QMV vs stock BF16 GEMV); both continuations stay coherent, on-topic, factually correct, no repetition collapse |

**Quality gates** (`--temperature 0` unless noted):

- Paris (48 tok, default config): *"The capital of France is Paris.
  Okay, so I need to figure out the capital of France. Let me start by
  recalling what I know. France is a country in Europe, right? I
  remember that some major cities in Europe"* - states Paris, coherent.
- Fib (80 tok, default config, `--stats`): coherent, on-topic Rust/
  Fibonacci discussion, no repetition collapse (matches T20's own
  "not literal code within 80 tokens, not a regression" bar).
- `BUTTER_LAGUNA_ATTN_INT8=0` (kernel 2 path) Paris/fib: also coherent,
  states Paris correctly, on-topic - see divergence note above.

**Decode A/B, fib prompt (80 tok, `--stats`, 2 runs each side):**

| Config | Run 1 | Run 2 | Decode steady (avg) |
|---|---|---|---|
| Default (`ATTN_INT8` ON), FUSED ON (kernel 1) | 62.48 t/s | 62.48 t/s | **62.48 t/s** |
| Default (`ATTN_INT8` ON), FUSED OFF | 61.82 t/s | 61.54 t/s | 61.68 t/s |
| `ATTN_INT8=0`, FUSED ON (kernel 2) | 61.76 t/s | - | 61.76 t/s |
| `ATTN_INT8=0`, FUSED OFF (stock) | 58.66 t/s | - | 58.66 t/s |

Default-path delta: **+1.3%** (62.48 vs 61.68 t/s) - small but clean,
consistent with T20's "dispatch count, not bandwidth" reading (kernel 1
removes 2 of 3 gate dispatches on the path that's actually live by
default). Kernel-2-only path delta: **+5.3%** (61.76 vs 58.66 t/s) - the
bigger win, but only reachable with `BUTTER_LAGUNA_ATTN_INT8=0`, a
non-default config. Baseline referenced in the task brief (~62 t/s
"record" comparison point) matches the FUSED-OFF default-path number
here (61.68 t/s) within run-to-run noise.

**Prefill @512 unaffected** (`"hello world "*255`, 512 tok,
`BUTTER_LAGUNA_TIMING=1`; T24 only touches decode `forward`, not
`forwardMulti`):

| Config | `layers=` (512 tok) | tok/s |
|---|---|---|
| FUSED ON, run 1 | 1053.4ms | 486.1 |
| FUSED ON, run 2 | 1061.8ms | 482.3 |
| FUSED OFF | 1055.3ms | 485.2 |

All three land within ~1% of each other - no regression vs. the ~476
t/s floor T23 established, well inside this machine's documented
run-to-run noise.

**L37 132ms anomaly (optional check, no fix attempted):** the task
brief flagged a prior overnight observation of `L37` at ~132ms vs
~29ms for sibling MoE layers. Did not reproduce here - this session's
`BUTTER_LAGUNA_TIMING=1` runs show `L37 moe+swa` at 29.47–30.27ms,
in line with `L38`/`L39`/`L01` (27.6–29.6ms) every time. Likely a
one-off (thermal, first-touch page fault, or a since-fixed condition)
rather than a standing issue - left as a note, not investigated further
per the brief.

**Verdict:** both gated-o_proj fusions are implemented,
default-verified-safe, and shipped as opt-out (`BUTTER_LAGUNA_FUSED_GATED_OPROJ=0`).
Kernel 1 (the path the default config actually takes) is a small,
byte-identical, dispatch-count win. Kernel 2 is a bigger win but
currently only reachable behind the non-default `ATTN_INT8=0` escape -
worth revisiting whether the INT8 o_proj bank (T20) and the fused-GEMV
o_proj (T24 kernel 2) can be combined in a future pass (an INT8-native
version of kernel 2's gate-folded GEMV), rather than being mutually
exclusive alternatives as shipped here.

**Files:** `Sources/ButterSwift/Models/Text/LagunaGatedOProjFuse.swift`
(new); wiring in `Sources/ButterSwift/Models/Text/LagunaText.swift`
(`LagunaLayer.forward`, decode-only).

### T24 continued - decode SDPA: GQA head-pairing + ALPHASKIP

Second half of the overnight dispatch-count pass, taken up after the
gated-o_proj fusion above was committed with time still on the clock.
Two named techniques - GQA pair-heads and ALPHASKIP - applied directly
to butter's own decode SDPA algorithm, mirrored line-for-
line off the iron-DSL-generated baseline
(`Sources/IronSwift/Resources/kernels/iron_sdpa_decode_bf16.metal`) so
the only real deltas are the two techniques, not an independent
reimplementation. New file:
`Sources/ButterSwift/Models/Text/LagunaSdpaDecodeGqa2.swift`, same
hand-written `pso()` pattern as `LagunaNvfp4Fuse` / `LagunaAttnInt8` /
`LagunaGatedOProjFuse` (this is a fixed-shape decode-only kernel, not a
generic primitive - iron DSL editing was out of scope for an overnight
timebox: it lives in a sibling repo with its own `make test` gate).

- **GQA head-pairing**: 1 threadgroup now serves 2 adjacent Q heads
  instead of 1 (`nHeads/2` threadgroups, 1024 threads each - same TG
  size as the stock kernel, half the dispatch count). Each lane reads a
  K/V element once per step and reuses it for both heads' dot products
  and weighted-value accumulation. Valid only when `heads_per_group`
  (`nHeads/nKVHeads`) is even, so a consecutive-index pair never
  straddles two different KV heads - true for both Laguna head counts
  (48/8=6, 64/8=8).
- **ALPHASKIP**: `factor = exp(oldMax - newMax)` is bit-exact `1.0f`
  whenever the running max didn't change; `os[i] * 1.0f == os[i]`
  exactly under IEEE 754, so `if (factor == 1.0f)` skips the multiply
  entirely on that (common) step. Shipped only the exact-comparison
  form - no `fma`-fused rewrite of the
  general branch, which would round differently than the original
  two-op `mul` then `add` chain.
- `has_sink` (GPT-OSS-style learned sink logit) is not modeled: Laguna's
  `Ops.sdpaDecode` call site never sets it, and it's algebraically a
  no-op at `has_sink=0` in the generated kernel this mirrors (verified
  by hand-tracing the generated MSL's `select()` chain before writing
  the simplified version). `sinkEnd`/`windowStart` (the *positional*
  sink+sliding-window fast path - unrelated to the logit sink) are
  fully supported and covered by the correctness suite below.

Escape: `BUTTER_LAGUNA_SDPA_GQA2=0`. Default ON when `headDim==128`,
dtype bf16, `heads_per_group` even (both Laguna head counts qualify).

**Correctness - new permanent test,
`Tests/ButterTests/Ops/LagunaSdpaDecodeGqa2Tests.swift`, 7/7 passing:**
synthetic random Q/K/V at Laguna's exact decode shapes (h48/kv8,
h64/kv8; dense small/mid `nKV`; sink+sliding-window; `nKV==1` edge
case), fused kernel vs. `Ops.sdpaDecode` (the proven stock kernel),
asserting both max-abs-diff `< 0.05` and cosine similarity `> 0.9999`
per case - not bit-exact-asserted (different source text can still
reorder a handful of float ops even when algebraically identical), but
every case actually landed far inside tolerance. Run via
`swift test --filter LagunaSdpaDecodeGqa2Tests` (targeted, not
`make test-unit`/`regenerate-kernels` - nothing iron-generated changed,
so the DSL regen step is a no-op this diff doesn't need to pay for).

Given this repo's own documented 2026-05-19 GPU-freeze postmortem (bad
dispatch shape from an unchecked wrapper), the correctness suite was
run and green *before* this kernel was ever wired into the live decode
path - same 1024-thread threadgroup size as the proven stock kernel,
half as many threadgroups, no new threadgroup-memory pressure (the 4
`tg_out*` scratch arrays are reused sequentially between the two paired
heads' combine steps rather than doubled).

**Quality gates** (`--temperature 0`, default config - both T24 fusions
ON): Paris (48 tok) and fib (80 tok) - **byte-identical** output with
`BUTTER_LAGUNA_SDPA_GQA2=0`, same as kernel 1's own byte-identical
result. A 150-token longer-form prompt ("TCP vs UDP") stayed coherent,
correct, no repetition collapse or garbage.

**Decode A/B, fib prompt (80 tok, `--stats`, 2 runs each side):**

| Config | Run 1 | Run 2 | Decode steady (avg) |
|---|---|---|---|
| `SDPA_GQA2` ON (+ gated-o_proj fusion, default) | 62.44 t/s | 62.37 t/s | **62.41 t/s** |
| `SDPA_GQA2` OFF (gated-o_proj fusion still ON) | 61.27 t/s | 61.51 t/s | 61.39 t/s |

Isolated delta: **+1.7%**. Compound with the gated-o_proj fusion above
(both ON, i.e. shipped defaults, vs. both escapes `=0`, i.e. full
stock):

| Config | Run 1 | Run 2 | Decode steady (avg) |
|---|---|---|---|
| Both T24 fusions ON (shipped default) | 62.13 t/s | 62.35 t/s | **62.24 t/s** |
| Both T24 fusions OFF (full stock) | 59.74 t/s | 59.70 t/s | 59.72 t/s |

**Compound win: +4.2%** (62.24 vs 59.72 t/s) - both dispatch-count
reductions stack cleanly, consistent with T20's "dispatch count, not
bandwidth" reading of this decode step.

**Prefill @512 unaffected** (decode-only change; `forwardMulti` never
calls `LagunaSdpaDecodeGqa2`): first back-to-back A/B read 1311ms
(GQA2 ON) vs. 1057ms (OFF) - a spurious ~24% gap that a 5s cooldown
between runs made vanish (1052–1056ms both ways across 3 repeat runs).
Logged as a reminder that this machine needs cooldown gaps between
back-to-back release-build timing runs, not a real regression - see
T21's own documented ~15% run-to-run noise floor, this was thermal on
top of that, not a code effect.

**Verdict:** both techniques (GQA head-pairing, exact-alpha rescale
skip) shipped, default ON, verified via a dedicated GPU-correctness
test *and* byte-identical default-path generation, +1.7% isolated /
+4.2% compound with T24's other fusion. Confirms the same thesis as
T20 and the gated-o_proj fusion: this decode step has more headroom in
dispatch-count/launch-overhead reduction than in raw bandwidth or ALU
work, at least at Laguna XS's current scale.

**Files:** `Sources/ButterSwift/Models/Text/LagunaSdpaDecodeGqa2.swift`
(new); `Tests/ButterTests/Ops/LagunaSdpaDecodeGqa2Tests.swift` (new);
wiring in `Sources/ButterSwift/Models/Text/LagunaText.swift`
(`LagunaLayer.forward`, decode-only).


### T25 - prefill MoE diagnosis: attention, not the NVFP4 gather-QMM GEMMs, dominates the wall

Overnight mission: close the prefill MoE kernel-efficiency gap by
porting a steel-class BK=64 gather pipeline into butter's NVFP4 MoE
gather kernels. Diagnosing first changed the plan: the ~29ms/MoE-layer
figure T21/T22/T23 worked from turns out to be dominated by
**attention**, not the MoE GEMMs, so the steel-pipeline port was not
attempted this session. Documenting why, with numbers, instead.

**New instrumentation (kept, env-gated OFF, zero default-path behavior
change):** `BUTTER_LAGUNA_MOE_PHASE_TIMING=1` inserts forced
commit+wait checkpoints inside `MoELayer.decodeManyNvfp4`
(`Sources/ButterSwift/Models/MoELayer.swift`) splitting each MoE layer's
wall-clock into `pre` (whatever attention/norm work the caller already
encoded on the shared per-layer command buffer before MoE started) /
`route` (router GEMM + sigmoid-bias top-k) / `sort` (device sort-plan) /
`gateup` (gate+up GEMM, fused or paired) / `down` (down-proj GEMM) /
`tail` (unpermute + shared-expert add), logged per layer to stderr as
`[laguna-moe-phase] L<n> total=...ms pre=...ms route=...ms ...`. Same
technique as the existing `BUTTER_LAGUNA_TIMING` layer splitter (forces
`prefDepth=1`-style full serialization for the duration of the
measurement) - diagnostic only, `work` had to become `var` inside
`decodeManyNvfp4` to support the mid-function command-buffer rotation,
but the checkpoint function is a no-op (`guard moePhaseTiming else {
return }`) when the env var is unset, so the change is inert by default.
Verified inert: `butter generate` on "What is the capital of France?"
(`--temperature 0`) still returns the same coherent, correct
"The capital of France is Paris..." completion with the env var unset,
and the existing `MoELayerTests` suite (9/9, synthetic 4-expert
route→SwiGLU→combine coverage) passes unchanged.

**Measurement - 512-token prefill (`"hello world "*255`), release
build, `--temperature 0`, `BUTTER_LAGUNA_MOE_PHASE_TIMING=1`, summed
over the model's 39 MoE-type layers (L01–L39; L00 is the one dense
layer and isn't covered by this instrumentation):**

| Phase | Sum (39 layers) | Share |
|---|---|---|
| `pre` (attention: QKV, QK-norm, RoPE, SDPA, o_proj, add+postAttnNorm) | 722.0ms | **74.9%** |
| `gateup` (NVFP4 gather-QMM gate+up GEMM) | 103.1ms | 10.7% |
| `down` (NVFP4 gather-QMM down GEMM) | 64.5ms | 6.7% |
| `route` (router GEMM + sigmoid-bias top-k) | 41.2ms | 4.3% |
| `sort` (device sort-plan) | 19.9ms | 2.1% |
| `tail` (unpermute + shared-expert add) | 12.9ms | 1.3% |
| **Total** | **963.7ms** (~24.7ms/layer avg) | 100% |

Cross-checked in the same run against the pre-existing
`BUTTER_LAGUNA_TIMING` layer-level splitter: `layers=1127.7ms
moe=1076.9 dense=50.8` (L00 dense=50.8ms + 39 moe layers=1076.9ms). The
39-layer sums roughly agree (963.7ms vs 1076.9ms; the ~113ms delta is
the extra sync overhead from phase-level vs layer-level serialization -
more forced commit/wait round trips), and both runs' per-layer totals
land in the same ~22–30ms band T21/T23 already reported. This is the
same measurement methodology (forced per-layer serialization), just
with one more level of decomposition.

**The correction:** T21's own "MoE ~95% of per-layer time" and this
mission's "MoE = 93% of wall at ~29ms per MoE layer" are both true
statements about *which layers* dominate total forward wall-clock
(39 of Laguna XS's 40 layers use MoE FFN, 1 is dense - so "moe-tagged"
layers trivially account for ~95% of total time by layer count). Neither
number was ever a claim about what happens *inside* a single MoE-tagged
layer. This session's phase split shows that within a MoE-tagged layer,
**attention (`pre`) is ~75% of the wall-clock and the NVFP4 gather-QMM
GEMMs (`gateup`+`down`) are only ~17.4% combined.** The "29ms/layer, 93%
MoE implies ~3ms/layer, kernel efficiency not orchestration"
framing in the mission brief conflated "layer classified as MoE-type" with
"time spent in MoE compute" - they are not the same thing once attention
is properly separated out.

**What this means for the steel-pipeline-port ask:** even a *perfect*
BK=64/wide-K gather kernel - eliminating 100%
of `gateup`+`down` time - would save at most ~17% of a MoE layer's
wall-clock (and less of full prefill, since L00's dense layer and the
lm_head/embed steps aren't touched at all). That's already below the
original ">15% win" bar taken in isolation, and nowhere near a
"stretch: 2-3x" framing, which was implicitly a whole-layer target based
on the mis-attributed 29ms figure. The dominant lever for closing the
throughput gap is **attention/SDPA prefill**, not
the NVFP4 gather-QMM kernels - a different, larger piece of work outside
this mission's scope (and outside this session's remaining time budget).

**Existing-lever sanity check (zero new code, just re-ran the isolated
`gateup`+`down` phase timing against an env flag that already
exists):** `BUTTER_NVFP4_MOE_BM64=1` (the BM64 gate/up kernels from
T14–T16, already built and shipped, off by default per T19's "+5.5%
with BM64 OFF") - re-measuring with the new phase-timing isolation
confirms the prior finding cleanly, now on the *isolated* GEMM time
instead of noisy whole-layer time: `gateup` 103.1ms→341.1ms (3.3x
worse), `down` 64.5ms→123.4ms (1.9x worse), `pre` unaffected (722.9ms,
noise-level delta from 722.0ms, as expected - BM64 only touches the FFN
GEMMs). Corroborates T19's default-OFF call from a cleaner vantage
point; no new lever found here.

**Rough roofline context (order-of-magnitude, not load-bearing):**
gate+up+down FLOPs per layer at T=512 (`mTotal=t·topK=4096`,
`hidden=2048`, `moe_intermediate=512`) = `3 · 2·4096·2048·512` ≈ 25.8
GFLOP. Measured combined `gateup`+`down` = 167.6ms / 39 layers ≈ 4.3ms/
layer → ~6.0 TFLOP/s effective. That's well under M5 Max's peak, so
there likely *is* still a real, if smaller, kernel-efficiency gap in the
NVFP4 gather-QMM GEMMs themselves - just not the dominant one, and not
big enough in absolute ms to be this mission's highest-value target
tonight.

**Structural note for whoever picks up the GEMM-kernel gap next**
(analysis only, not applied or tested this session - flagging as a
lead, not a verified fix): comparing butter's Iron-DSL-generated
`butter_nvfp4_moe_gather_bm16_mpp_xrows_bf16.metal` against a leaner
staging strategy surfaces two structural deltas neither T22 nor T23
tested (both sessions only varied K-tile width, BK32→BK64, at fixed
BM=16 - a 0% result):
1. An alternative kernel design could stage only the **weight** into
   `bfloat` (full float32-width exponent) for the `mpp::tensor_ops::matmul2d`
   cooperative-tensor call and skip staging the **activation** into
   threadgroup memory or a narrowed type at all - loading it directly
   from device memory into per-thread NAX tile registers instead.
   Butter's kernel stages *both* operands into `half`
   via scalar per-lane loops in threadgroup memory - the exact
   fragility T22/T23 spent a full session patching with the `2^-14`
   exponent-bias fold. Adopting that asymmetric-staging design (bfloat
   weight stage, no activation staging) would likely have sidestepped
   that whole overflow investigation, though it's an open question
   whether MPP's `matmul2d` on this hardware/Metal-version accepts
   `bfloat` cooperative-tensor operands the same way it accepts `half`
   - not verified this session.
2. Butter's Iron-generated kernel loops the K-reduction in **16-wide**
   steps (2 `threadgroup_barrier` calls per step; K=2048 gate/up →
   128 iterations → 256 barriers), each re-decoding the NVFP4 block
   scale from scratch. A leaner design would block K in 32/64-wide
   chunks with only 2 barriers per chunk, `SK=32`/`UK=16` subtile
   `matmul2d` calls issued *without* intervening barriers, and the C
   accumulator held in per-thread registers across the entire
   K-reduction (matches butter's existing accumulator handling - that
   part isn't the gap). The T22 BK64 hand-kernel *did* widen K but kept
   BM=16 and the Iron-style scalar per-lane staging loops, which may be
   why it saw no win - the barrier-count reduction alone wasn't the
   bottleneck, and this session's BM64 re-test (above) shows widening
   the *other* dimension doesn't help either via the existing kernel
   family. A genuine win would likely need that combination (wide K,
   no activation staging, vectorized weight loader), not one axis at a
   time - a real rewrite, not a parameter tweak, and per the corrected
   diagnosis above, worth at most ~17% of a MoE layer even if fully
   realized.

**Verdict / where this stopped:** stage 1 (diagnose) is done and
changed the plan. Stages 2–4 (steel-pipeline port, correctness ladder,
perf A/B) were not attempted - the diagnosis showed the originally
targeted kernel family accounts for too little of the wall-clock to be
this session's highest-value use of the remaining time, and a from-
scratch Metal kernel rewrite carries real stall risk (as T22's own
"correctness-blocked" detour demonstrated) that isn't worth taking on
for a ≤17%-of-one-layer ceiling this close to a hard deadline. Recommend
next session point the "close the throughput gap" effort at
**attention/SDPA prefill** (`pre`, ~75% of MoE-layer wall-clock,
already using the T11c `sdpaPrefillMma` fast path as its floor) instead
of the NVFP4 gather-QMM GEMMs.

**Files:** `Sources/ButterSwift/Models/MoELayer.swift`
(`BUTTER_LAGUNA_MOE_PHASE_TIMING` instrumentation, diagnostic-only, off
by default).

### T26 - attention phase breakdown finds the real culprit (naive dense
GEMM, not SDPA) - 1.78x prefill win, default ON

T25 handed off "attention/SDPA prefill" as the next lever, with a
specific hypothesis in the follow-up brief: the sliding-window per-row
decode SDPA path was assumed to be running at T=512 and costing
~27ms/layer where a flash-style kernel should cost ~1-2ms. Stage 1
(diagnose first, same rule T25 used) extended the phase-timing
instrumentation *inside* the attention phase itself and found that
hypothesis wrong on both counts - worth stating plainly since it
changed where Stage 2 landed.

**New instrumentation (kept, env-gated OFF, zero default-path behavior
change):** `BUTTER_LAGUNA_ATTN_PHASE_TIMING=1` adds the same
forced-commit+wait checkpoint technique as T25's
`BUTTER_LAGUNA_MOE_PHASE_TIMING`, this time inside
`LagunaLayer.forwardMulti` (`Sources/ButterSwift/Models/Text/LagunaText.swift`),
splitting the attention phase into `norm` / `qkv_proj` / `qknorm` /
`rope_kv` / `kv_prep` / `sdpa[<path>]` (further broken into
`sdpa_qpad` / `sdpa_qcast_transpose` / `sdpa_kv_compact` /
`sdpa_mma_kernel` / `sdpa_out_transpose_cast` via a `phaseCheckpoint`
hook threaded into `LagunaSdpa.prefillMmaBf16`,
`Sources/ButterSwift/Models/Text/LagunaSdpa.swift`) / `gate` / `oproj`
/ `addnorm`, logged per layer as `[laguna-attn-phase] L<n> total=...
<phase>=...ms ...`. The checkpoint function is a no-op (`guard
attnPhaseTiming else { return }`) when the env var is unset, and the
one live behavioral difference from before - `prefillMmaBf16`'s `cmd`
parameter became `inout` so its internal stages can checkpoint too -
resolves to the exact same command buffer identity on every call when
the hook is `nil` (the default), so this is inert on the default path.
Also logs which SDPA branch each layer actually took
(`windowed-decode-loop` / `prefillMma` / `sdpaMulti-fallback`) rather
than assuming from `isSliding`.

**First finding - the "sliding-window path" hypothesis was wrong.**
`KVCacheProtocol.sdpaSinkWindow(nKV:)` has exactly one implementation
in the codebase (`Sources/ButterSwift/KVCache/KVCache.swift`, a
protocol-extension default) and it unconditionally returns `(sinkEnd:
0, windowStart: 0)` - no cache class overrides it. `LagunaLayer.forwardMulti`'s
windowed-decode-loop branch only fires when `windowStart > 0 ||
sinkEnd > 0`, which is therefore never true today, regardless of
`isSliding`. Confirmed empirically: all 40 layers' phase logs report
`sdpa[prefillMma]` - **zero layers ever take the per-row windowed-decode
path at T=512.** Every layer, sliding or full, already goes through the
`LagunaSdpa.prefillMmaBf16` MMA path. (Laguna XS's `sliding_window` is
512, exactly equal to this benchmark's T - real long-context prefill
chunks *would* need the window to actually engage, so this is a latent
gap worth flagging for whoever next works on >512-token contexts, but
it is not the T=512 story.)

**Second finding - SDPA itself was never the bottleneck.** Summed
over all 40 layers (`BUTTER_LAGUNA_ATTN_PHASE_TIMING=1`, same
`"hello world "*255` / 512-token / `--temperature 0` setup as T25):

| Phase | Sum (40 layers) | Share |
|---|---|---|
| `qkv_proj` (Q/K/V dense BF16 GEMMs) | 390.2ms | **35.1%** |
| `oproj` (O dense BF16 GEMM) | 297.9ms | **26.8%** |
| `norm` (input RMSNorm) | 155.9ms | 14.0% |
| `gate` (softplus headwise gate) | 125.4ms | 11.3% |
| `rope_kv` (RoPE + KV append) | 37.0ms | 3.3% |
| `sdpa_mma_kernel` (the actual MMA attention kernel) | 24.1ms | 2.2% |
| `qknorm` | 16.1ms | 1.4% |
| `sdpa_out_transpose_cast` | 16.0ms | 1.4% |
| `sdpa_qcast_transpose` | 13.2ms | 1.2% |
| `addnorm` | 12.9ms | 1.2% |
| `sdpa_qpad` | 12.3ms | 1.1% |
| `sdpa_kv_compact` | 9.0ms | 0.8% |
| `kv_prep` | 1.3ms | 0.1% |
| `sdpa[prefillMma]` (residual/label bucket) | 0.8ms | 0.1% |
| **Total** | **1112.2ms** (~27.8ms/layer) | 100% |

All five SDPA sub-stages combined (`sdpa_qpad` + `sdpa_qcast_transpose`
+ `sdpa_kv_compact` + `sdpa_mma_kernel` + `sdpa_out_transpose_cast` +
`kv_prep`) sum to **~76.7ms, 6.9% of the attention wall-clock** - the
`sdpa_mma_kernel` bucket alone (the actual attention math) is 0.60ms/layer
average, already inside the ~1-2ms/layer range the mission brief wanted
from a "steel-class flash kernel." The dominant cost is the plain dense
BF16 `qProj`/`kProj`/`vProj`/`oProj` GEMMs - 61.9% combined - which is
where Stage 2 went instead of a flash-attention port.

**Root cause, found by reading the dispatch, not guessing:**
`AnyLinear.callMany` (`Sources/ButterSwift/Layers.swift`) routes plain
BF16 Linear layers through `Ops.gemm`
(`Sources/ButterSwift/Ops/Ops.swift`), which dispatches
`iron_gemm_bf16` - a 32×32-output-tile kernel
(`Sources/IronSwift/Resources/kernels/iron_gemm_bf16.metal`) where
**each thread computes one output element via a fully scalar
per-K-element FMA loop**, staged through threadgroup memory but never
touching the GPU's matrix/tensor units. Meanwhile the Iron kernel
library already ships `iron_steel_gemm_64x64x16_2x2_bf16.metal` - a
proper `simdgroup_matrix`-based MMA GEMM (64×64 threadgroup tile, 2×2
8×8-fragment sub-tiling) - generated and compiled into the metallib but
with **zero callers and zero test coverage anywhere in the repo**
(`grep -rln "_64x64x16_2x2" Sources/` returns nothing outside the
generated kernel sources). It's the same simdgroup_matrix MMA structure
NVFP4's gather kernels use, just never wired up for the *dense*
(non-gather) case that attention's Q/K/V/O projections need.

**Stage 2 fix: `LagunaSteelGemm.swift`** wires that unused steel kernel
into Laguna's prefill Q/K/V/O projections specifically (not the shared
`Ops.gemm`, to keep blast radius contained to Laguna). Semantics were
pinned down empirically before touching any production path - see
`Tests/ButterTests/Ops/LagunaSteelGemmProbeTests.swift`, a standalone
correctness probe against a CPU reference that confirms the kernel's
`a`/`b` layout convention: `a` is `[M,K]` row-major (K contiguous,
matches activations as-is), `b` is `[K,N]` row-major (N contiguous) -
the *transpose* of how Butter stores Linear weights (`[outDim,
inDim]`, inDim contiguous). Weights are transposed once at
`LagunaLayer.init` time (`LagunaSteelGemm.transposeWeightBf16`, its own
tiny inline Metal kernel, own commit+wait - load-time only, never the
hot path) and cached as `qWeightT`/`kWeightT`/`vWeightT`/`oWeightT`.
`forwardMulti` calls a new `projOrSteel` helper that dispatches the
steel path when `LagunaSteelGemm.eligible(t:outDim:inDim:dtype:)` holds
and falls back to the existing `AnyLinear.callMany` path otherwise.

**Eligibility is conservative and load-bearing for correctness, not
just perf:** `iron_steel_gemm_64x64x16_2x2_bf16` has **no bounds
masking** on its output writes (unlike the naive kernel, which checks
`n_rows`/`out_dim` per-thread before writing) - if `t`, `outDim`, or
`inDim` aren't exact multiples of 64, the kernel reads/writes past
tensor bounds. `eligible(...)` requires all three `% 64 == 0`. Laguna
XS's prefill shapes all qualify (`hidden=2048`, Q `outDim=6144`, K/V
`outDim=1024`, T=512 in this benchmark), but decode (`T=1`) and any
non-64-aligned prefill chunk correctly fall back - no risk of the OOB
path firing unexpectedly.

**Gates (release build, `swift build -c release`):**
- Build: clean, no new warnings beyond pre-existing ones.
- Targeted tests (`swift test --filter
  "LagunaSteelGemmProbeTests|LagunaRecognitionTests|LagunaSdpaDecodeGqa2Tests|MoELayerTests"`):
  **24/24 pass**, including the new probe test.
- Correctness A/B, `--temperature 0` (greedy/deterministic), steel ON
  vs OFF (`BUTTER_LAGUNA_ATTN_STEEL_GEMM=1` vs `=0`):
  - Paris (8-token prompt, T<64 so steel is ineligible regardless):
    identical, unaffected as expected.
  - Fib (6-token prompt, 80-token continuation, T<64): byte-identical
    output both configs - coherent, idiomatic Rust `fn fib`.
  - 512-token `"hello world "*255` filler + 20-token continuation
    (T=512, steel path *is* eligible and engaged here): **generated
    continuation text byte-identical** between steel ON and OFF.
- Decode-steady tok/s (fib prompt, tokens 11+, 3 runs each, 5s
  cooldown): OFF 91.68 / 91.78 / 92.23 (avg 91.9) vs ON 91.01 / 91.37 /
  90.78 (avg 91.05) - **no regression** (within run-to-run noise;
  decode is always T=1, never eligible for the steel path, so this is
  expected to be a pure no-op and measured as one).
- Prefill @512 A/B (3 runs each, 5s cooldown, `--max-tokens 1
  --temperature 0`):

  | Config | Run 1 | Run 2 | Run 3 | Median |
  |---|---|---|---|---|
  | OFF (naive `iron_gemm_bf16`) | 569.11 | 569.59 | 568.89 | 569.11 tok/s |
  | ON (steel `iron_steel_gemm_64x64x16_2x2_bf16`) | 1013.08 | 1012.68 | 1014.32 | 1013.08 tok/s |

  **1.78x prefill speedup**, <0.3% run-to-run variance on both sides
  (much cleaner signal than T21's ~15% noise floor on this shared
  machine - this run happened to be quiet). A later default-path
  sanity re-run (no env var, i.e. the new default) landed at
  1010-1016 tok/s across another 3 runs, confirming the flip.

**Default flipped ON** (`BUTTER_LAGUNA_ATTN_STEEL_GEMM` now defaults to
enabled; `=0` escapes to the old naive-GEMM path) per the gate: quality
green (byte-identical generations) and a clear, reproducible win.

**What's left (flagged, not chased - see "norm"/"gate" caveat below):**
re-running the phase breakdown with the steel path on shows `qkv_proj`
dropped from 390.2ms to 178.1ms (2.19x) and `oproj` from 297.9ms to
121.0ms (2.46x) sum-over-40-layers, matching the end-to-end win. The
now-largest remaining buckets are `norm` (156.5ms) and `gate`
(122.5ms) - but both are suspect as measurement artifacts rather than
real targets: the *instrumented* per-layer total (721.7ms after the
fix) is ~216ms higher than the clean uninstrumented prefill time
(~505ms, back-computed from the 1013 tok/s median), i.e. the 13
forced-checkpoint round trips per layer add roughly 0.4ms/checkpoint of
host-sync overhead each - enough to fully explain a cheap single
`rmsNormRows` call over 512×2048 elements reporting ~3.9ms. `gate`
(a small `hidden→nHeads` GEMM + softplus + gather-expand + mul) likely
has more real cost mixed in than `norm` does, but wasn't isolated
further this session - real target for whoever picks this up next:
re-run with per-sub-op checkpoints *inside* `applySoftplusHeadGateManyHost`
before assuming `gate` needs a fix, the same "diagnose before porting"
discipline this session and T25 both leaned on.

**Files:** `Sources/ButterSwift/Models/Text/LagunaText.swift`
(`BUTTER_LAGUNA_ATTN_PHASE_TIMING` instrumentation + `projOrSteel`
wiring), `Sources/ButterSwift/Models/Text/LagunaSdpa.swift`
(`prefillMmaBf16` gained the optional `phaseCheckpoint` hook, `cmd` is
now `inout`), `Sources/ButterSwift/Models/Text/LagunaSteelGemm.swift`
(new - steel MMA GEMM wrapper + weight-transpose helper),
`Tests/ButterTests/Ops/LagunaSteelGemmProbeTests.swift` (new -
standalone kernel-semantics correctness probe).

### T27 - round 2: re-diagnose with steel ON, extend steel-GEMM coverage
(L00 dense FFN + MoE router), re-A/B the NAX64 gather lever

Round 2 of the prefill attack, continuing directly from T26's default-ON
steel MMA GEMM for attention Q/K/V/O. Same "diagnose before porting"
discipline as T25/T26.

**Stage 1 - re-diagnosis with steel ON (512-token `"hello world "*255`
prefill, release build, `--temperature 0`, current default config i.e.
`BUTTER_LAGUNA_ATTN_STEEL_GEMM` unset = ON):**

`BUTTER_LAGUNA_ATTN_PHASE_TIMING=1` (40 layers, attention sub-phases):

| Phase | Sum (40 layers) | Share |
|---|---|---|
| `qkv_proj` (now steel MMA GEMM) | 177.72ms | 24.5% |
| `norm` (input RMSNorm - mostly checkpoint-sync overhead, see T26) | 156.62ms | 21.6% |
| `gate` (softplus headwise gate - small GEMM + gather/mul, mostly overhead) | 124.20ms | 17.1% |
| `oproj` (now steel MMA GEMM) | 121.03ms | 16.7% |
| `rope_kv` | 37.84ms | 5.2% |
| `sdpa_mma_kernel` | 25.76ms | 3.5% |
| `sdpa_out_transpose_cast` | 17.29ms | 2.4% |
| `qknorm` | 16.61ms | 2.3% |
| `sdpa_qcast_transpose` | 13.18ms | 1.8% |
| `addnorm` | 12.40ms | 1.7% |
| `sdpa_qpad` | 12.11ms | 1.7% |
| `sdpa_kv_compact` | 9.29ms | 1.3% |
| `kv_prep` | 1.34ms | 0.2% |
| `sdpa[prefillMma]` (residual/label) | 0.89ms | 0.1% |
| **Total** | **726.29ms** | 100% |

Matches T26's own "what's left" projection almost exactly (`qkv_proj`
178.1→177.72, `oproj` 121.0→121.03, `norm` 156.5→156.62, `gate`
122.5→124.20) - confirms T26's numbers were reproducible, not a
one-off. `norm`+`gate` together are 38.7% of this *instrumented* total,
but per T26's own analysis (13 forced checkpoints/layer × ~0.4ms
host-sync overhead ≈ 5.2ms/layer ≈ 208ms/40 layers fully explains a
cheap `rmsNormRows` call reporting multi-ms) neither is a real GEMM
target - `norm` has no GEMM at all, and `gate`'s GEMM is `hidden(2048)
→ nHeads(48)`, non-64-aligned (48 % 64 ≠ 0) and computationally trivial
(512×2048×48×2 ≈ 100 MFLOP, sub-microsecond at any reasonable
throughput) - its wall-clock is softplus+gather+mul host-dispatch
overhead, not a GEMM-efficiency problem, so it's out of scope for the
steel-GEMM lever specifically.

`BUTTER_LAGUNA_MOE_PHASE_TIMING=1` (39 MoE layers):

| Phase | Sum (39 layers) | Share |
|---|---|---|
| `pre` (attention, now steel-accelerated) | 332.27ms | 57.9% |
| `gateup` (NVFP4 gather-QMM gate+up GEMM) | 102.86ms | 17.9% |
| `down` (NVFP4 gather-QMM down GEMM) | 64.71ms | 11.3% |
| `route` (router GEMM + sigmoid-bias top-k) | 41.65ms | 7.3% |
| `sort` (device sort-plan) | 19.44ms | 3.4% |
| `tail` (unpermute + shared-expert add) | 12.75ms | 2.2% |
| **Total** | **573.67ms** | 100% |

`gateup`+`down` unchanged in absolute terms from T25 (167.6ms→167.57ms
- steel never touched them) but their **share** jumped from 17.4%
(T25, `pre`-dominated denominator) to **29.2%** (T25's own
"if >25% of wall, attempt NAX64 hybridization" trigger from the
mission brief) now that `pre` shrank 2.17× from T26's fix. `pre` is
still the single largest bucket (57.9%) but is already steel-accelerated
- its remaining cost is the `norm`/`gate`/SDPA-glue buckets covered
above, not a fresh GEMM-efficiency gap.

**Stage 2 - extend steel-GEMM coverage.** Candidate audit (code-read,
not guessed) of every remaining dense BF16 GEMM on the prefill path:

| Candidate | Shape | Eligible? | Verdict |
|---|---|---|---|
| L00 dense FFN (`LagunaDenseMLP.gate/up/down`) | hidden=2048, intermediate=8192 (both %64==0) | Yes, plain BF16 Linear | **Wired** |
| MoE router (`MoELayer.gate`, `decodeManyNvfp4`'s `route` phase) | hidden=2048 → nExperts=256 (both %64==0) | Yes, plain BF16 Linear (confirmed via safetensors header: `model.layers.N.mlp.gate.weight`, dtype BF16, shape [256,2048], no `.scales` sibling) | **Wired** |
| Shared expert (`MoELayer.sharedGateProj/UpProj/DownProj`) | hidden=2048 ↔ shared_expert_intermediate=512 (both %64==0 - shape-eligible) | **No** - on the NVFP4 board (`isNvfp4Mode` true in `buildLagunaLayer`), `prefillSharedExpertNvfp4` (`Sources/ButterSwift/Models/MoELayer.swift:3258`) loads it as `Nvfp4Linear` via `lagunaLoadNvfp4Linear`, dispatching `Ops.nvfp4Qmm` (dequant-QMM), not `Ops.gemm`/`AnyLinear.callMany` - confirmed via safetensors index (`shared_expert.gate_proj.scales` sibling present). Not a BF16 dense GEMM on this checkpoint at all. | Not applicable - ruled out by reading the loader, not measurement |
| `gProj` (attention softplus gate) | hidden=2048 → nHeads=48 | No - 48 % 64 ≠ 0, and per Stage 1, computationally trivial regardless | Skipped |
| `lm_head` final-row prefill path | single row | No - brief's own call, confirmed not worth pursuing | Skipped (per brief) |

Wiring: factored T26's inline `steelWeightT` closure out of
`LagunaLayer.init` into a shared `LagunaSteelGemm.weightT(_:dtype:device:)`
helper (`Sources/ButterSwift/Models/Text/LagunaSteelGemm.swift`) so
`LagunaDenseMLP` and `MoELayer` can reuse the same
plain-bf16-Linear-or-nil guard instead of re-deriving it.
`LagunaDenseMLP` gained `gateWeightT`/`upWeightT`/`downWeightT` (computed
at `LagunaLayer.init` time, load-time only) and a `projOrSteel` closure
in `forwardMany` mirroring `LagunaLayer.forwardMulti`'s. `MoELayer`
gained `gateWeightT` (same pattern, computed in `MoELayer.init`, which
now takes optional `dtype`/`device` params defaulting to `.bf16`/`.shared`
to preserve every existing call site) and the router-GEMM call site in
`decodeManyNvfp4` (`Sources/ButterSwift/Models/MoELayer.swift`, the
`route` phase) now tries the steel path before falling back to
`gate.callMany`. Both reuse `LagunaSteelGemm.eligible(...)` - same
%64==0 + bf16 + `BUTTER_LAGUNA_ATTN_STEEL_GEMM` gate as T26, no new
env var (this is additional coverage under the existing T26 flag/default,
not a separate feature).

**Gates (release build, `swift build -c release`, clean, no new
warnings):**
- Targeted tests (`swift test --filter
  "LagunaSteelGemmProbeTests|LagunaRecognitionTests|LagunaSdpaDecodeGqa2Tests|MoELayerTests"`):
  **24/24 pass**, unchanged.
- Correctness A/B, `--temperature 0`, steel ON (default) vs OFF
  (`BUTTER_LAGUNA_ATTN_STEEL_GEMM=0`):
  - Paris (8-token prompt, T<64): byte-identical.
  - Fib (6-token prompt, 80-token continuation, T<64): byte-identical.
  - 512-token filler + 20-token continuation (T=512, exercises the new
    L00-FFN and router-GEMM steel paths in addition to T26's QKV/O):
    **generated continuation byte-identical** between ON and OFF.
- Decode-steady tok/s (fib prompt, 3 runs each, 5s cooldown - T=1
  always, never eligible for any steel path, expected pure no-op):
  OFF 90.68 / 90.16 / 90.15 (avg 90.33) vs ON 92.26 / 91.12 / 91.68
  (avg 91.69) - **no regression** (within run-to-run noise, same as
  T26).
- Prefill @512 A/B (3 runs each, 5s cooldown, `--max-tokens 1
  --temperature 0`):

  | Config | Run 1 | Run 2 | Run 3 | Median (s) | tok/s |
  |---|---|---|---|---|---|
  | OFF (naive `Ops.gemm` everywhere) | 0.90s | 0.90s | 0.90s | 0.90s | 568.9 |
  | ON (T26 QKV/O + T27 L00-FFN + router steel) | 0.50s | 0.50s | 0.49s | 0.50s | ~1030.7 |

  `--stats`' 2-decimal-second resolution is too coarse to cleanly
  separate T26's already-shipped 1013 tok/s from this session's
  incremental delta at the whole-`generate`-run level, so the
  isolating measurement is `BUTTER_LAGUNA_TIMING=1`'s per-forward
  breakdown instead, same build, same prompt, `--max-tokens 1`:

  | | OFF (`=0`) | ON (default) | Speedup |
  |---|---|---|---|
  | `layers` (total layer wall) | 1061.4ms | 650.4ms | 1.63x |
  | `moe` (39 MoE-tagged layers, incl. attention `pre` + route + gather GEMMs) | 1008.6ms | 620.3ms | 1.63x |
  | `dense` (L00: attention + FFN combined) | 52.8ms | 30.1ms | **1.75x** |

  L00's dense-layer wall dropped 1.75x (52.8→30.1ms) from adding FFN
  coverage on top of T26's attention-only win - a small absolute
  number (1 of 40 layers) but a clean, real, isolated confirmation the
  new wiring is doing real work, not noise. `route`'s contribution is
  too small in absolute ms (~1ms/layer) to isolate cleanly from
  `BUTTER_LAGUNA_TIMING`'s layer-level granularity; its correctness is
  covered by the byte-identical 512-token gate above (which exercises
  the router-GEMM path 39 times) and by `MoELayerTests`.

**Verdict:** small, real, low-risk win - ships default ON under the
existing T26 flag (no new escape needed; `BUTTER_LAGUNA_ATTN_STEEL_GEMM=0`
already covers both the T26 and T27 additions). Files:
`Sources/ButterSwift/Models/Text/LagunaSteelGemm.swift` (new
`weightT(_:dtype:device:)` shared helper), `LagunaText.swift`
(`LagunaDenseMLP` steel wiring, `LagunaLayer.init` refactored to use
the shared helper), `Sources/ButterSwift/Models/MoELayer.swift`
(`MoELayer.gateWeightT` + router-GEMM steel dispatch in
`decodeManyNvfp4`).

**Stage 3 - attack the largest remaining bucket.** Per Stage 1, the
NVFP4 gather-QMM `gateup`+`down` GEMMs are now 29.2% of the MoE-layer
wall (crossing the brief's own >25% trigger for a NAX64 re-A/B / possible
hybridization attempt). Re-A/B first, same discipline as the rest of
this campaign.

**`BUTTER_NVFP4_GATHER_NAX64=1` re-A/B under the new steel-ON baseline**
(`BUTTER_LAGUNA_MOE_PHASE_TIMING=1`, 512-token prefill, 39 MoE layers):

| Phase | OFF (default) | ON (`NAX64=1`) | Delta |
|---|---|---|---|
| `gateup` | 103.38ms | 289.99ms | **2.80x worse** |
| `down` | 64.00ms | 49.56ms | 1.29x better |
| `route`/`sort`/`tail`/`pre` | unchanged (noise-level) | unchanged | - |
| **Total (39 layers)** | 581.89ms | 758.34ms | **1.30x worse overall** |

Matches T25's own re-confirmation of T19's BM64 finding: widening the
gather-QMM tile geometry is a net loss on this board's shapes with the
existing kernel structure, `gateup` (mid=512, the narrower N) losing
far more than `down` (n=2048, the wider N) gains.

**Correctness re-check** (same "What is the capital of France?" prompt,
20 tokens, `--temperature 0`, `mTotal = t·topK = 8·8 = 64` so NAX64
*is* eligible even on this short prompt):

- OFF: `"The capital of France is Paris.\nOkay, so I need to figure out
  the capital of"`
- `NAX64=1`: `"The capital of France is Paris. Paris is located in the
  northern part of the country. It"`

**Diverges after "Paris."** - this is T22's original "correctness-
blocked" finding (`### T22 - NAX gather QMM (true BK=64 staging;
correctness-blocked, OFF)`), independently re-confirmed here 5 sessions
later, still unresolved. The perf regression above is almost moot next
to this: a kernel producing wrong logits isn't shippable regardless of
speed.

**Decision: hybridization NOT attempted this session.** The brief's own
gate for this stretch goal is ">25% of wall AND time permits" - the
25% bar is met, but every existing "wide-tile" NVFP4 gather variant in
this codebase is now confirmed either correctness-broken (`GATHER_NAX64`
above, `GATHER_BK64` - "full-model empty gen", never fixed since first
flagged) or a measured performance regression on Laguna XS's specific
shapes (`GATHER_NAX` ~365-371 t/s vs a ~410+ floor, `HYBRID_BM=1` a
documented T19 regression) - five separate attempts across T14-T23, all
negative. Per T25's own structural analysis, a real fix needs the
*combination* of bfloat weight staging with **no** activation staging, wide (32/64-tile)
K-reduction with only 2 barriers per chunk instead of butter's current
16-wide/256-barrier loop - which is a from-scratch Metal kernel rewrite,
not a flag flip or parameter tweak. That class of change is exactly what
T22 spent a full session on and landed "correctness-blocked" on; attempting
it again in the remaining window of a round-2 session that has already
shipped two verified wins, with no time buffer left to debug a bad
kernel before the deadline, is the "real stall risk" the brief itself
warns against. Per the brief's own honest-fallback clause - "diagnosis +
env-gated partials + notes beats silent stalls" - this session stops
here on the GEMM-kernel-gap axis and documents the recommendation
instead of attempting a rushed rewrite.

**Recommendation for whoever picks this up next:** treat the
gather-QMM K-reduction rewrite as its own dedicated session (not a
stretch goal bolted onto a round-2 prefill pass) - implement T25's two
structural deltas together (bfloat weight stage / no activation stage,
wide-K barrier-light reduction) against `butter_nvfp4_moe_gather_bm16_mpp_xrows_bf16.metal`,
gate correctness against the CPU reference the same way
`LagunaSteelGemmProbeTests` did for the dense steel GEMM (a standalone
kernel-semantics probe *before* wiring it into any production path),
and budget for a correctness-debugging pass given the NAX/NAX64/BK64
family's unbroken 0-for-5 track record on this exact class of kernel.
Ceiling if fully realized: ~29% of a MoE-tagged layer's wall (Stage 1's
`gateup`+`down` share) - real, but smaller than either T26's or this
session's attention/dense-GEMM wins in absolute terms, since attention
already dominates and is now steel-accelerated.

**T27 overall verdict:** two shipped, gated, default-ON wins (L00 dense
FFN + MoE router steel GEMM, Stage 2) built on a clean re-diagnosis
(Stage 1) that correctly predicted where the next real lever was *and
wasn't* (Stage 3) - same "diagnose before porting" discipline as
T25/T26, applied to itself. No regressions, no new escape hatches
needed (both ride T26's existing `BUTTER_LAGUNA_ATTN_STEEL_GEMM` flag).
The NVFP4 gather-QMM kernel-efficiency gap remains open, now with a
concrete rewrite spec (T25 structural note) and 5 sessions of negative
results on the "just flip a tile-geometry flag" family ruled out -
next session should treat it as a from-scratch kernel task with its own
budget, not another A/B sweep of existing flags.

### T28 - overnight decode levers

Overnight mission: implement four highest-expected-impact
remaining decode levers into Laguna's Path B decode, in priority order,
gating each on release-build correctness + A/B before committing, and
skipping honestly where the structure doesn't map or the numbers don't
support it. Baseline going in: ~90-92 tok/s decode (fresh process); target
floor 140.3 tok/s on the same M5 Max.

**Session result: decode steady 79.3 -> 102.3 tok/s (+29.0%)** from the two
levers that shipped default-ON (lever 1 dominates; lever 4 is a smaller
top-up on the same measurement), fib prompt, 3 runs/side, 5s cooldowns,
same build:

| Config | Run 1 | Run 2 | Run 3 | Avg |
|---|---|---|---|---|
| Pre-T28 (`NINE_SLOT=0`, `LAYERS_PER_CB=2`) | 80.01 | 78.99 | 78.86 | 79.29 tok/s |
| Post-T28 (both shipped defaults) | 102.41 | 102.20 | 102.32 | 102.31 tok/s |

#### Lever 1 - nine-slot merged MoE dispatch (shipped, default ON)

Implements a nine-slot merged dispatch-count idea (8 routed + 1
shared expert slot in one threadgroup family) - **not** a
rows4/256-threadgroup hyper-optimized geometry (that layout regressed
in earlier experimentation) - targeting instead the
proven, earlier 1152-threadgroup shape. New kernels in
`Sources/ButterSwift/Models/Text/LagunaNvfp4Fuse.swift`:
`laguna_nine_slot_nvfp4_swiglu_qmv_bf16` (1152 TGs, TG=64 - 8 routed + 1
shared expert slots, tile-major, one dispatch) and
`laguna_nine_slot_nvfp4_down_reduce_bf16` (512 TGs, TG=288 - 9-slot down
QMV + weighted reduce, shared added unconditionally at weight 1.0, same
accumulation order as the stock routed-then-shared chain). Both reuse the
exact `iron_nvfp4_qdot_16` fast bitpath + `bfloat(g*sig*u)` rounding already
proven in `moeRoutedSwigluQmv`/`moeRoutedDownReduce` and
`laguna_shared_nvfp4_swiglu_qmv` - a dispatch-count merge, not new
numerics. Replaces the routed swiglu dispatch + the separate shared
gate/up/down chain (dual-queue event pair, or single-queue early-encode) +
the final `Ops.add(acc, sharedOut)` - up to 5-7 dispatches collapsed to 2.

Wired into `MoELayer.decodeGPUPathBNvfp4` as an early-exit branch, gated on
the exact Laguna XS Path B shape (mid=512, hidden=2048, topK=8, bf16,
block=16) plus the shared expert being NVFP4-backed with a `fuseGateUp`-
compatible gate/up pair (reuses the existing `ensureFusedSharedGateUp`
cache; a new `ensureNineSlotShared` adds the raw shared-down NVFP4 fields
alongside it). Default ON; escape `BUTTER_LAGUNA_NINE_SLOT=0`.

Gates (release build, `--temperature 0`):
- Paris (48 tok): **byte-identical** ON vs OFF.
- Fib (80 tok): diverges after ~20 tokens - the shared expert now takes the
  fast `qdot_16` path instead of the stock `Ops.nvfp4Qgemv`/iron GEMV it
  used before (dual-queue's default `BUTTER_LAGUNA_FUSED_SHARED_SWIGLU=0`
  meant the shared leg was never on the fast path pre-T28), an expected
  FP-reorder; both continuations are coherent, on-topic Rust/Fibonacci
  discussion, no repetition collapse.
- 150-token TCP/UDP prompt: coherent, factually correct, no repetition
  collapse or garbage at longer horizon.
- Decode steady (fib, 3 runs/side, 5s cooldown): **ON avg 100.6 tok/s vs OFF
  avg 85.1 tok/s (+18.2%)**.
- Prefill @512 unaffected (decode-only path): 645.9ms vs 657.4ms layers,
  within noise.

#### Lever 2 - residual+RMS+router fold (shipped, default OFF - regresses)

Implements a fold of the
post-attention residual add + RMSNorm + MoE router GEMV into one dispatch,
reusing the normed row (staged in threadgroup memory) for the router GEMV
instead of a separate `gate(h, on: cmd)` call. New file
`Sources/ButterSwift/Models/Text/LagunaResidualRmsRouterFuse.swift`.
`MoELayer.decode` gained a sibling `decodeWithPrecomputedLogits` entry
point (the existing 4-arg `decode` is `DecoderLayer` protocol conformance
and can't grow a defaulted 5th parameter - protocol witness matching
requires the exact declared signature, confirmed by a build error on the
first attempt).

Two attempts, both regressions on decode steady (fib, 3 runs/side, 5s
cooldown):

| Attempt | Threadgroups | Decode steady | vs stock (100.1-100.2 avg) |
|---|---|---|---|
| 1. Single TG (normed row "free" once resident) | 1 | 49.4 tok/s | **-50.6%** |
| 2. `rows_per_group=8` layout (redundant residual+norm recompute per TG, 8-row router slice each) | 32 | 94.9 tok/s | **-5.3%** |

Root cause for both: the stock separate router GEMV (`Ops.gemv`, called via
plain `Linear.callAsFunction`) already dispatches **one threadgroup per
output row** - 256 threadgroups for Laguna's `[256,2048]` router - spreading
the GEMV across up to 256 GPU cores. Attempt 1 collapsed all 256 rows onto
one threadgroup (one core) to make the normed-row reuse "free," which
serialized the entire router GEMV and cost far more than the one dispatch
it saved. Attempt 2 (using a `rows_per_group=8` default)
recovered most of the loss by parallelizing across `256/8 = 32`
threadgroups, but 32 is still 8x less parallel than stock's 256, and this fold's per-row reduction (32 lanes/row) is narrower than `iron_gemv_bf16`'s
(256 threads/row) - net still a regression, just a much smaller one.

Residual/normed correctness verified: Paris (T=0) states "Paris" reachable
identically at both configs; full text diverges a few tokens later from the
router-GEMV FP-reorder (same class as every other decode fuse in this file
family) - coherent, not garbage.

**Kept as a correctness-verified, tested opt-in**
(`BUTTER_LAGUNA_RESIDUAL_RMS_ROUTER=1`) rather than deleted, per this
repo's own precedent for correct-but-slower kernels
(`BUTTER_LAGUNA_FUSED_SHARED_SWIGLU`, `BUTTER_NVFP4_MOE_BM64`). Default
OFF - byte-for-byte the pre-existing `addAndRmsNorm` + separate `gate()`
path. A real win would need matching `Ops.gemv`'s per-row threadgroup count
(~256, not 32), which changes the redundant-recompute-vs-parallelism
tradeoff enough to be its own investigation, not a same-session follow-up.

#### Lever 3 - embedding+RoPE-atlas fold (skipped)

Diagnosed before porting anything, per this campaign's "diagnose before
porting" rule. Existing measurements (not independently reverified this
session, but taken at face value) already settle the question: with
every other lever at default, folding this in measures roughly -0.23%
steady decode (se 0.03%, 0/2 ABBA pairs favoring ON, quiescent-machine
rig) - the fused kernel's fixed cost exceeds the two tiny probe
dispatches it would remove, the same promoted-era-value rot its prefill
sibling showed. The fold depends on a load-time "RoPE angle atlas" (a
precomputed FP32 cos/sin table per position, built once and consumed by
both prefill and decode fused kernels) that reads angle rows instead of
recomputing per call.

Butter has no equivalent infrastructure - `lagunaRopeYaRNPartial`
(`Sources/ButterSwift/Models/Text/LagunaText.swift`) computes YaRN cos/sin
**analytically inside the kernel** from `position` + `theta_base` every
call, per layer; there is no atlas table anywhere in the codebase to fold
the embedding gather against. Porting this lever would mean building the
entire atlas-table prerequisite from scratch, in a session's remaining time,
to reach a fold that measurements already say costs
more than it saves. Skipped - not attempted, no code written. Recommend
against revisiting unless butter independently grows RoPE-angle-atlas
infrastructure for an unrelated (e.g. prefill) reason first.

#### Lever 4 - decode CB pipe sweep (shipped, default `LAYERS_PER_CB` 2 -> 1)

Butter's per-token decode pipeline
(`BUTTER_LAGUNA_LAYERS_PER_CB`/`BUTTER_LAGUNA_PIPE_DEPTH`,
`LagunaText.swift`'s `forward(tokenId:...)`) is this codebase's own
front-loaded async-commit boundary scheduling mechanism, distinct from a
lazy-op-graph `asyncEval` scheduler (imperative per-layer command buffers
vs lazy-graph eval), so this session A/B'd butter's own knobs rather than
adopting a boundary-set config format with no direct translation onto a
non-lazy dispatch pipeline.

`BUTTER_LAGUNA_PIPE_DEPTH` swept 1..8 (fib prompt, `--stats`, 5s cooldowns,
single run/value): flat 100.08-100.48 tok/s across the whole range - already
past its saturation point at the existing default (4). No change.

`BUTTER_LAGUNA_LAYERS_PER_CB` swept 1..20 (same harness): the *old* default
(2) was already past its own sweet spot -

| layersPerCB | decode steady |
|---|---|
| 1 | 102.27 tok/s |
| 2 (old default) | 100.53 tok/s |
| 3 | 98.35 tok/s |
| 4 | 96.47 tok/s |
| 5 | 95.05 tok/s |
| 8 | 91.50 tok/s |
| 10 | 88.06 tok/s |
| 20 | 75.29 tok/s |

Confirmed with 3 runs/side, 5s cooldown: **1 avg 102.11 tok/s vs 2 avg
100.11 tok/s (+1.9%)**, monotonic falloff beyond that. Packing more layers
into one command buffer trades dispatch overlap for bigger, less-parallel
buffers on this per-layer-CB architecture - plausibly shifted by lever 1's
own dispatch-count cut shrinking each layer's work enough that the finest
CB granularity now wins outright (the value that picked `layersPerCB=2` as
default predates tonight's session).

Flipped default to 1. Pure scheduling change, zero numerics delta: Paris
(T=0) byte-identical to pre-change. Prefill unaffected (decode-only knob,
`forwardMulti` never reads it): 651.7ms layers @512, within the ~646-660ms
noise band measured earlier this session.

**T28 overall verdict:** two shipped, default-ON wins (lever 1 dispatch
merge, lever 4 CB-packing retune) compounding to **+29.0% decode steady**
(79.3 -> 102.3 tok/s) this session, against a ~90-92 tok/s starting point
and a 140.3 tok/s target floor on the same hardware. One lever
shipped correct-but-off with a clear root cause and a scoped follow-up
(lever 2 - needs per-row threadgroup count parity with `Ops.gemv`, not a
quick tweak). One lever skipped before writing any code, because the
existing measurements already answered the question
(lever 3). Same "diagnose before porting, gate honestly, document losses"
discipline as T20-T27.

### T29

Overnight target: a prior finding said the certified 2-pass lm_head prune
(`LagunaLmHeadPrune.swift`) gives **+58% decode on confident-text
workloads (Paris ~102 tok/s) but ~0% on flat-logit workloads (rust fib
~65 tok/s, identical with `BUTTER_LAGUNA_LM_HEAD_PRUNE=0`)** - the claim
being that when the certified candidate set |C| blows up, the exact
pass's full/coarse cost dominates per-token time and swamps the prune's
win. Task: implement two new lm_head candidate-mask levers and, time
permitting, diagnose where the fib case's per-token time actually goes.

**Model correction (read this before the numbers below).** The prune
only fires when an MXFP8 lm_head sidecar is loaded
(`lm_head_mxfp8_codes.u8` / `_scales.u8`); those files exist only under
`~/models/laguna-xs-2.1-nvfp4`, **not** under this board's usual smoke-test
model `~/models/laguna-xs-2.1-4bit` (affine4's `lm_head` is itself int4
and takes `rmsNormQgemvInt4Fast`'s fused fast path in
`lagunaFinalNormLmHead`, before `LagunaLmHeadPrune` is ever reached - see
`Sources/ButterSwift/Models/Text/LagunaText.swift:2415-2442`). First pass
of this session's gates ran against affine4 by mistake (copying the T28
smoke-test habit) and "passed" with a suspicious ~1% ON/OFF delta on
*both* prompts - a tell that the toggled code was never executing. All
numbers below are re-measured against `laguna-xs-2.1-nvfp4`, which is
also the model whose Paris number (~102 tok/s) matches the task's own
baseline figure exactly.

#### Item 1 - coarse pass v2 (word-parallel e4m3) - already shipped, no new work

Checked `Sources/ButterSwift/Models/Text/LagunaLmHeadPrune.swift`'s
existing `laguna_lmhead_coarse_v2` kernel: MXFP8-word-parallel
`laguna_e4m3_decode4` decode, `ushort4` bf16 activation loads, a
fully-unrolled 2x8x4 loop structure, fixed accumulation order. This lever
landed in an earlier session (`3b7c0826`, "NVFP4 Laguna XS runtime ...
certified lm_head prune") already at its intended v2 form - nothing to
do here.

#### Item 2 - inline candidate mask (shipped, default ON)

Adds an inline candidate-mask lever for the lm_head exact pass.
Folds SELECTION into EXACT: the exact-pass kernel recomputes
`coarse[r] + delta[r] >= thr[0]` itself per fixed 4-row simdgroup block
instead of reading a precomputed `is_cand` mask, so the standalone select
dispatch and its `uint8` buffer disappear, and the coarse pass drops its
`coarse_bf` store (the exact kernel reloads the f32 `coarse` value and
applies the identical `bfloat()` cast - same NaN/signed-zero behavior,
same certificate). New kernels `laguna_lmhead_inline_coarse_v2` (2
outputs, no `coarse_bf`) and `laguna_lmhead_exact_inline_mask_block`
(candidate test inlined). One fewer GPU dispatch per decode token (3 vs
4) - a **fixed per-token cost cut independent of |C|**, so unlike the
coarse-pass cost itself it should help flat-logit workloads too, not
just confident-text ones. Gated `BUTTER_LAGUNA_LMHEAD_INLINE_MASK`,
default ON; `=0` restores the original four-dispatch
`coarse_bf`/select/exact path verbatim in the same binary (kill switch +
same-binary A/B).

**Gates (release build, M5 Max, `laguna-xs-2.1-nvfp4`, `--temperature 0`):**
- Paris (48 tok): **byte-identical** ON vs OFF.
- Fib (80 tok): **byte-identical** ON vs OFF; coherent real Rust
  (`fn fibonacci(n: u32) -> u64 { match n { 0 => 0, 1 => 1, _ => { let
  mut a = 0; let mut b = 1; ...`), no repetition collapse.
- Decode A/B (3 runs/side, 5s cooldown):

  | Prompt | OFF (kill switch) | ON (T29 default) | Δ |
  |---|---|---|---|
  | Paris | 101.63, 98.08, 101.82 (avg **100.51**) | 101.96, 101.92, 101.72 (avg **101.87**) | **+1.35%** |
  | Fib | 101.64, 102.28, 102.00 (avg **101.97**) | 100.92, 102.62, 102.39 (avg **101.98**) | +0.01% (flat, no regression) |

- Prefill @512 (`BUTTER_LAGUNA_TIMING=1`, 2 runs/side after a noisy
  first-run outlier): OFF 648.8ms / ON 650.1ms layers - flat, no
  regression (lm_head fires once per prefill call on a single row; its
  cost is noise against 512-token attention/MoE).

#### Item 3 - diagnosis: where does the fib case's time actually go?

**Data first, per the task brief's own instruction, and the data says
the premise doesn't reproduce here.** Added an opt-in diagnostic-only
probe (`BUTTER_LAGUNA_LMHEAD_DIAG=1`, new `laguna_lmhead_cand_count`
atomic-counter kernel + a `cmd.addCompletedHandler` readback; never on
the perf-gated path above) that prints the certified candidate-set size
`|C|` per decode token. Run against the exact fib prompt from the task
brief, 80 tokens, `--temperature 0`, default (ON) config:

| | value |
|---|---|
| `|C|` range | 2 to 3229 (out of 100352 vocab rows) |
| `|C|` at token 1 (highest-uncertainty token) | 3229 (3.2% of vocab) |
| `|C|` steady-state (tokens ~10+) | single/low-double digits, several < 10 |
| max `|C|` anywhere in the 80-token run | 3229 (3.2%) |

The candidate set never comes close to "blowing up" for this prompt on
this build: even its worst token touches ~3% of vocab rows, and the
`laguna_lmhead_exact_*_block` kernel's early-out is per **4-row block**
(25088 blocks total), so a few thousand scattered candidates still skip
the overwhelming majority of blocks. Consistent with this, **fib and
Paris decode-steady numbers are statistically indistinguishable in this
session's measurements** (~100-102 tok/s, both prompts, both ON and
OFF) - not the ~102-vs-~65 split the task brief opened with. Two
non-exclusive explanations, neither chased further given the time
budget: (a) the original finding may have used a longer/more
code-heavy continuation than 80 tokens (deeper into real Rust syntax,
more near-tied whitespace/punctuation candidates) where |C| grows later
than this run reached; (b) something about tonight's build (T25-T28
landed several unrelated attention/MoE/dispatch levers between the
finding's measurement and this session) shifted the steady-state
baseline enough that lm_head is no longer the fib-case bottleneck it
was when the finding was taken "minutes ago" relative to a different
commit.

**No certification-threshold/two-level-coarse tuning attempted.** The
task brief's own fallback instruction was conditional on the data
showing the full uncertified path dominating - it doesn't, here, so
loosening the certified bound (which is the correctness contract) would
be an unmotivated risk to the certificate for a problem this session's
own measurements can't reproduce. If a future session reproduces the
slow fib case, `BUTTER_LAGUNA_LMHEAD_DIAG=1` is the fastest way to
confirm whether |C| is actually large before touching the threshold.

**T29 overall verdict:** item 1 was already shipped pre-session (no
work needed, verified against the intended design). Item 2
shipped, default ON: byte-identical correctness on both gate prompts,
+1.35% decode on Paris, flat (no regression) on fib. Item 3 diagnosed
honestly with a new reusable opt-in probe (`BUTTER_LAGUNA_LMHEAD_DIAG=1`)
rather than a forced fix - this session's data does not show the
flat-logit blowup the task opened with, so no threshold change was
made. Corrected a mid-session model-selection mistake (affine4 has no
MXFP8 sidecar, so the prune path was never exercised) before any of
the numbers above were taken as real.

### T30 - prefill fused QK-norm+RoPE (sliding + YaRN full-attention),
single-dispatch prefill kernels

Final overnight task: build single-dispatch fused QK-norm+RoPE prefill
kernels - one dispatch each replacing the stock chain of 4-6 separate
kernels (Q/K RMSNorm, RoPE, plus for the sliding case the SDPA
pad/cast/transpose sub-stages T26/T27 measured individually) - into
butter's prefill path.
T27's own "what's left" list flagged this as the next attention lever
(`qkv_proj` still 24.5% of attn wall even on steel; the remaining
`sdpa_qpad`+`sdpa_qcast_transpose`+`qknorm`+Q's share of `rope_kv` add up
to a smaller but real slice on top of that).

**Sliding layers (33/40) - shipped, default ON.** New
`LagunaSdpa.fusedQKNormRopeForPrefill` (`Sources/ButterSwift/Models/Text/
LagunaSdpa.swift`): one TG-per-(head,token) dispatch (128 threads = 4
simdgroups, one simdgroup per head, grid.y = nRows) fuses Q/K RMSNorm +
plain RoPE (full rotary, headDim=128, matching the already-shipped decode
twin `LagunaFusedQKNormRoPE`'s formula) AND writes Q **directly** into the
padded `[nHeads, tPad, headDim]` f16 layout `LagunaSdpa.prefillMmaBf16`
otherwise builds via a separate pad→cast→transpose chain - so this kills
`sdpa_qpad` + `sdpa_qcast_transpose` too, not just the norm/RoPE
dispatches. K is written in the same `[nRows, nKVHeads, headDim]` bf16
layout `KVCache.appendRangeOnGPUMany` already expects, so cache-append
code is untouched. Two-phase API (`fusedQKNormRopeForPrefill` - pure, no
cache mutation - then `prefillMmaTailFromHTD` - KV compact + MMA +
transpose/cast back to bf16) so a caller's eligibility pre-check (shape/
dtype/`kPad<=maxSeq`/no active sink-window, mirroring `prefillMmaBf16`'s
own guard) can run *before* committing to the path - once K is appended
to the cache there's no cheap rollback to the stock chain. Wired into
`LagunaLayer.forwardMulti` (`Sources/ButterSwift/Models/Text/
LagunaText.swift`) as a new branch ahead of the existing QK-norm/RoPE/SDPA
selection, which becomes the `else`. Escape: `BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE=0`.

**Correctness - near-ULP FP-reorder, not a bug (verified via a standalone
probe, not just eyeballing).** A new `Tests/ButterTests/Ops/
LagunaPrefillFusedQKRopeProbeTests.swift` (same spirit as
`LagunaSteelGemmProbeTests`) runs the fused kernel against a manual
replica of the stock chain (`Ops.rmsNormRows` x2 + `Ops.ropePartialManyTwo`
+ `Ops.castBf16ToF16` + `Ops.transposeTHD_f16`) on synthetic data: at
nRows=32 the two are byte-identical (0 mismatches). At the realistic
nRows=512 (512 distinct RoPE angles, some with large `position·inv_freq`
arguments to `cos`/`sin`), a handful of elements differ by ~1 bf16 ULP -
39/4,194,304 Q elements (maxAbs 0.0039) and 5/524,288 K elements (maxAbs
6.1e-5). Two independently-compiled Metal kernels evaluating the same
transcendental expression can legitimately round the last bit differently;
this is the same class of divergence already accepted through T20-T29
(steel GEMM, nine-slot MoE dispatch, etc.), confirmed here with an actual
isolated measurement instead of assumed. At full-model scale (40 layers,
greedy argmax) this tiny per-layer perturbation does cascade into a
different but still coherent generation - see gates below.

**Gates (release build, `swift build -c release`, clean, no new
warnings):**
- Targeted tests (`swift test --filter
  "LagunaPrefillFusedQKRopeProbe|LagunaSteelGemmProbeTests|LagunaRecognitionTests|LagunaSdpaDecodeGqa2Tests|MoELayerTests"`):
  **25/25 pass** (24 pre-existing + the new probe).
- Correctness A/B, `--temperature 0`, ON (default) vs OFF
  (`BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE=0`), `laguna-xs-2.1-4bit`:
  - Paris (48 tok, T=8 < 32 → path ineligible regardless): byte-identical.
  - Fib (80 tok, T=6 < 32): byte-identical.
  - 512-token `"hello world "*255` filler + 20-token continuation (T=512,
    path *is* eligible for the 33 sliding layers): generated continuation
    **diverges** after the literal echoed prompt - expected, per the
    probe's own near-ULP finding above, not a regression. Both
    continuations stay in-register English (no repetition collapse, no
    token salad).
  - A second, non-degenerate 512-token real-English prompt (a paragraph on
    computing history, repeated to length, + "summarize in one sentence")
    diverges the same way but is a useful coherence tell: OFF's
    continuation gets stuck in a repetition loop ("The text appears to
    discuss the development of software." x5); ON's continuation is a
    clean, accurate one-sentence summary of the passage. Not a claim that
    ON is "better" in general (T=0 greedy + a coincidentally-favorable
    logit nudge, nothing more) - just further evidence this is ordinary
    FP-reorder noise, not corruption.
- Decode-steady tok/s (fib prompt, T=1 always → never eligible for this
  prefill-only path, expected pure no-op; 3 runs/side, 5s cooldown): OFF
  98.55 / 99.28 / 98.62 (avg 98.82) vs ON 100.16 / 99.66 / 100.59 (avg
  100.14) - **no regression** (within run-to-run noise, as expected).
- Prefill @512 A/B (`--max-tokens 1 --temperature 0 --stats`, 5 runs/side
  **interleaved** - OFF/ON/OFF/ON/... - with 5s cooldowns, to control for
  host drift rather than relying on two back-to-back blocks):

  | Run | OFF | ON |
  |---|---|---|
  | 1 | 889.52 | 896.61 |
  | 2 | 890.27 | 897.18 |
  | 3 | 888.33 | 895.70 |
  | 4 | 888.90 | 897.87 |
  | 5 | 889.41 | 899.07 |
  | **Median** | **889.41** | **897.18** |

  **+0.87%**, and notably clean for this host (zero overlap between the
  two distributions across 5 interleaved trials - OFF always 888-890, ON
  always 895-899). Small, as expected: T27's own phase breakdown put
  `sdpa_qpad`+`sdpa_qcast_transpose` at ~2.9% of the *attention* wall and
  `qknorm` at ~2.3% (plus an unmeasured Q-only slice of `rope_kv`'s 5.2%),
  and attention itself is ~57.9% of a MoE-tagged layer's wall (T27) - so
  the addressable ceiling for this specific lever was always a few percent
  of whole-prefill, not a multiplier. A separate `BUTTER_LAGUNA_TIMING=1`
  per-layer-forced-sync read (2 runs/side) showed the *opposite* sign
  (OFF ~825ms vs ON ~851ms `layers`) - not trusted as the production
  signal: that instrumentation forces a commit+wait at every layer
  boundary regardless of config, and per T26/T27's own notes about
  `BUTTER_LAGUNA_ATTN_PHASE_TIMING`, forced-sync round trips are known to
  read differently than the natural async-pipelined default path
  (`--stats`, un-instrumented, is what a real caller experiences).
  Flagging the discrepancy rather than picking whichever number looked
  better - worth a follow-up if someone wants to fully reconcile it, but
  the interleaved `--stats` numbers above are both the more
  production-representative measurement and the higher-confidence one
  (tight, non-overlapping, 5-sample).

**YaRN full-attention twin (7/40 layers) - shipped, default ON, same
flag.** Time permitted the stretch target after all. New
`LagunaSdpa.fusedQKNormYarnForPrefill` (same file): same TG-per-(head,
token) structure, but partial rotary (`rotaryDim=64` of `headDim=128`,
lanes `[0,8)` rotate both halves of the 64-wide rotary range, lanes
`[16,32)` pass the RMSNorm'd tail `[64,128)` through untouched, lanes
`[8,16)` idle) and YaRN's correction-ramp frequency blend instead of
plain RoPE. **Matched against butter's *own* stock kernel
(`iron_rope_yarn_bf16`, what `lagunaRopeYaRNPartial` actually dispatches)**,
rather than pre-scaling the *input* through an extra bf16
round before applying the YaRN `attn_factor` (butter's own kernel
instead folds it into `cos`/`sin` once), and are only
equivalent in exact arithmetic, not under bf16 rounding, so matching
butter's actual chain is the correct target. Reuses
`prefillQkRopeFuseEnabled`/`BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE` - no new
env var - and `prefillMmaTailFromHTD` (generic, doesn't care about
rotaryDim). The addressable win here is structurally *larger* per-layer
despite touching only 7/40 layers:
`lagunaBatchedRopeAndKV`'s `isFullYarn` branch calls
`lagunaRopeYaRNPartial` **once per row**, not batched like the sliding
path's `Ops.ropePartialManyTwo` - 2×nRows tiny dispatches per
full-attention layer (1024 at T=512) collapse to one.

**Bug found and fixed en route (in existing shipped code, not new
code):** `Ops.rmsNormRows`+`Ops.ropePartialManyTwo`+`Ops.castBf16ToF16`+
`Ops.transposeTHD_f16` is the *exact* stock chain
`fusedQKNormYarnForPrefill` replaces, so the new
`LagunaPrefillFusedQKRopeProbeTests.matchesStockYarnChain` probe built it
by hand to compare against - and hit an apparent 24448/24576-element
all-zero mismatch that turned out to be a real, pre-existing bug in
`Ops.castBf16ToF16` (`Sources/ButterSwift/Ops/Ops.swift`): its
`iron_cast_bf16_to_f16_f32` kernel reads a bounds-check
`constant uint &n [[buffer(2)]]`, but the Swift call site never bound
buffer index 2 - Metal does not zero-initialize an unbound slot, so the
kernel's `n > tid` guard was silently reading whatever an *unrelated*
prior encoder happened to leave there, truncating the cast whenever that
residual value was smaller than the real element count. This call site is
also what `LagunaSdpa.prefillMmaBf16` (T26, default-on, shipped for
weeks) uses for its own Q pad→cast→transpose chain - it likely never
tripped in production because the dispatches that typically precede it in
this codebase's real usage (batched `rmsNormRows`/`ropePartialManyTwo`
GEMV-scale dispatches) happen to leave a large-enough residual `n`, but
that was luck, not a guarantee, and the per-row YaRN loop's much smaller
dispatches broke that luck immediately. One-line fix: bind `n` via
`enc.setBytes` like every other dispatch in this file already does.
Verified fixed against the probe (0 mismatches after the fix vs
24448/24576 before) and against the full targeted suite (still 26/26
green - the fix makes `Ops.castBf16ToF16` strictly more correct, not
behaviorally different for any call site that was already working by
luck).

**Gates (release build, same methodology as the sliding twin above,
`laguna-xs-2.1-4bit`, both fuses default ON together):**
- Targeted tests: **26/26 pass** (25 prior + the new YaRN probe).
- Paris/fib (T<32): byte-identical, as before (path ineligible
  regardless of which fuse).
- 512-token `"hello world "*255` filler: diverges post-prompt (same
  near-ULP class as the sliding-only result), stays coherent English, no
  repetition collapse or garbage.
- The real-English "computing history" long-prompt coherence check
  (T=559, exercises *both* fuses across all 40 layers): **byte-identical**
  continuation between OFF and ON this time - the castBf16ToF16 fix
  plus this specific prompt's logit landscape didn't tip any argmax
  crossover, unlike the degenerate filler prompt above. Not a claim that
  it's always byte-identical at T>=32 (the sliding-only run already
  showed it isn't, in general) - just further evidence of "ordinary
  noise, not corruption."
- Decode-steady (fib, T=1, never eligible, 3 runs/side, 5s cooldown): OFF
  101.05 / 100.87 / 100.86 (avg 100.93) vs ON 103.04 / 100.92 / 102.36
  (avg 102.11) - no regression.
- Prefill @512 A/B (5 interleaved runs, 5s cooldowns, same protocol as
  the sliding-only measurement):

  | Run | OFF | ON |
  |---|---|---|
  | 1 | 877.42 | 917.51 |
  | 2 | 877.06 | 914.34 |
  | 3 | 869.00 | 914.87 |
  | 4 | 877.36 | 916.12 |
  | 5 | 876.17 | 914.76 |
  | **Median** | **877.06** | **914.87** |

  **+4.31%** combined (sliding + YaRN together) - noticeably bigger than
  the sliding-only +0.87%, consistent with the YaRN twin's per-row-dispatch
  collapse being a structurally larger win per touched layer than the
  sliding fuse's already-batched stock chain. (A cold-process single-shot
  `--stats` run showed the *opposite* sign, ON slower - a fresh CLI
  process pays both new kernels' one-time Metal shader compile inside the
  very first prefill it measures; the interleaved-with-cooldown protocol
  avoids that artifact, same lesson as the `BUTTER_LAGUNA_TIMING`
  discrepancy noted above.)

**Files:** `Sources/ButterSwift/Models/Text/LagunaSdpa.swift`
(`fusedQKNormRopeForPrefill` + `fusedQKNormYarnForPrefill` +
`prefillMmaTailFromHTD` + the new Metal kernel sources,
`prefillQkRopeFuseEnabled`/`prefillQkRopeFuseEligible`/
`prefillQkYarnFuseEligible`), `Sources/ButterSwift/Models/Text/
LagunaText.swift` (`LagunaLayer.forwardMulti`'s two new fused branches -
sliding and YaRN), `Sources/ButterSwift/Ops/Ops.swift`
(`castBf16ToF16`'s missing-buffer-binding fix), `Tests/ButterTests/Ops/
LagunaPrefillFusedQKRopeProbeTests.swift` (new - standalone kernel probes
for both the sliding and YaRN fuses).

**T30 overall verdict:** both the sliding (33/40 layers) and YaRN
full-attention (7/40 layers) prefill QK-norm+RoPE fuses shipped, default
ON, sharing one flag. Combined prefill @512 win **+4.31%** (877.06 →
914.87 tok/s median, 5 interleaved runs). Correctness verified via two
standalone kernel probes (not just end-to-end eyeballing) plus a real
pre-existing bug found and fixed in `Ops.castBf16ToF16` along the way.
Both gate suites green, no regressions on decode or the Paris/fib
short-prompt quality checks. Smaller than the session's earlier T26 win
(1.78x) by design - T27's own phase breakdown had already sized this
lever's ceiling at a few percent of whole-prefill, not a multiplier, and
the measured result lands right where that estimate predicted.

---

## Overnight session summary (2026-07-29 → 07-30, T20–T29)

Branch `tom/feat/laguna-overnight-perf` (local only), 21 commits (T20-T30). Targets: record 140.3 decode / 4080.3 prefill on this same M5 Max.

| Metric | Session start | Session end | Record | Remaining gap |
|--------|--------------|-------------|--------|---------------|
| Prefill @512 | ~411 tok/s | **~1058–1063 tok/s (2.6x)** | 4080 | ~3.8x |
| Decode steady | ~57.8 tok/s | **~102 tok/s fast-window (+77%); see caveat** | 140.3 | ~1.4x |
| Quality | Paris + fib | unchanged (byte-identical T=0 on every default-ON change) | - | - |

**Shipped default-ON:** T20 attn group-32 INT8 requant + batched QKV QMV (+1.1%); T21 residency-set gap fixes (+1.6%); T24 fused softplus gate product + gated o_proj and GQA-paired+exact-ALPHASKIP SDPA (+4.2% compound); T26 steel MMA GEMM for Q/K/V/O prefill projections (**1.78x prefill** - the naive scalar-FMA `iron_gemm_bf16` was the whole attention story; `iron_steel_gemm_64x64x16_2x2_bf16` existed with zero callers); T27 steel coverage for L00 dense FFN + router (L00 wall 1.75x); T28 nine-slot merged routed+shared MoE dispatch (**+18.2% decode**) + layersPerCB 2→1 (+1.9%); T29 lm_head inline-mask exact pass (+1.35%); T30 prefill fused QK-norm+RoPE, sliding + YaRN twins (+4.3%, kills the per-row YaRN dispatch storm and the SDPA pad/transpose stages) + fixed a latent castBf16ToF16 missing bounds-buffer bind that had worked by luck since T26.

**Shipped opt-in / fixed-but-parked:** T22 BK=64 NAX gather (correct after T23, but T27 re-A/B shows net regression - all wide-tile gather flags confirmed non-wins); T23 **2^-14 scale-fold staging fix for the whole MPP gather family** (removes a real NaN cliff that explains every historical "unit-green, full-model-broken" gather failure; permanent magnitude regression test added); T28 residual+RMS+router fold (correct, regresses, OFF).

**Diagnosis corrections that reordered everything:** T25 phase timing proved attention was 75% of prefill wall (MoE GEMMs only 17%) - the campaign had been attacking the wrong 8x for days; T26/T27 then proved the GEMM kernel *class* (scalar-FMA vs simdgroup MMA) was the real attention story.

**Open items for next session (ranked):**
1. **Decode/host bimodality - leading hypothesis: compositor GPU contention.** The box oscillates between ~65 and ~102 tok/s decode windows (both prompts, same binary; prefill unaffected; no thermal flags). 07:00 probe: every slow-window run coincided with WindowServer pinned at 50-54% CPU (Hermes desktop helpers actively compositing); the ~102 fast windows all fell in dead hours. Fits the signature: compositor steals GPU time from latency-sensitive decode dispatches but not from large batched prefill. Test: quit Hermes/browsers or let the display sleep, decode should jump to ~102. All overnight A/Bs were within-window so relative gains hold.
2. Prefill next 2x: fuse norm into the steel-GEMM epilogue (QK-norm+RoPE fusions landed in T30); MoE gather GEMMs now ~29% of MoE wall - needs the from-scratch simdgroup-tile NVFP4 gather kernel (spec in T27 notes), not more mpp-staging variants.
3. Decode last 1.4x: the remaining edge is likely compiled decode + async-stage boundaries + kernel-level unrolls; butter analogs partially explored (T28 lever 4), rest undocumented.
