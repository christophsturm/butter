# butter-prefill-hunt — T18 one prefill win (hybrid BM64 OFF default)

**Worktree:** `/Users/tom/dev/butter-dev` (`tom/feat/laguna-swift-xs`)  
**When:** 2026-07-29  
**Host:** M5 Max, swap used ~9.8–10 GB sticky (mlx thrash residual). Prefill still measurable.  
**Build:** `swift build -c release --build-path /tmp/butter-prefill-hunt --product butter`  
**Binary:** `/tmp/butter-prefill-hunt/release/butter`

## Goal

Find and land **one** prefill win that preserves Paris quality. Floor: >3% at T=512.

## Candidates tried

| # | Candidate | Result |
|---|-----------|--------|
| 1 | Shared expert scratches + dense NAX + BK32 all active | **Confirmed ON** as defaults. Warm baseline with hybrid still ON: ~456–470 t/s. |
| 2 | `BUTTER_NVFP4_MPP_XROWS=1` with BK32 (skip materialise) | **Parity** with materialise+BK32 once warm (~465). Not the win. |
| 3 | Dual-queue shared + concurrent MPP contention (`DUAL_QUEUE=0`) | **Parity** (~465). Dual-queue not the limiter when warm. |
| 4 | `BUTTER_LAGUNA_TIMING=1` profile | **MoE still ~95–96%** of layer wall. |
| — | Gate/up materialise grow-only scratch | Hygiene ship (avoids per-layer `Tensor.empty`); **not** the headline warm win. |
| **WIN** | **Hybrid BM64 for wide-N OFF by default** | **+5.5% prefill** vs hybrid ON. |

## THE WIN — hybrid BM64 opt-in only

T14 made hybrid default: BM16 gate/up (nOut=512), BM64 down (nOut=2048).  
Re-measure at T=512: **BM64 down is slower than all-BM16**.

### Code change

`Sources/ButterSwift/Ops/OpsNVFP4.swift` — `nvfp4MoeGatherPrefersBm64`:

- **Before:** hybrid ON unless `BUTTER_NVFP4_HYBRID_BM=0`
- **After:** all-BM16 default; hybrid only if `BUTTER_NVFP4_HYBRID_BM=1` (or force `MOE_BM64=1` / `GATHER_BM=64`)

Also shipped (hygiene, same PR surface):

- Gate/up materialise gather → `Device.intermediateScratch` + `Ops.gather(..., into:)`
- MoELayer non-fused materialise path → same grow-only pattern

**GATHER_BK64 remains OFF** (empty-gen; not touched).

### Quality (new default)

| Prompt | Result |
|--------|--------|
| "What is the capital of France?" | **OK** — "The capital of France is Paris. …" |
| "Write a short rust fibonacci function" | **OK** — coherent `fn fibonacci(n: i32) -> i32 { … }` |

### Prefill T=512 (warm, 3 runs)

Prompt: `('hello world ')*255` → 512 tokens. `BUTTER_LAGUNA_TIMING=1`, `--max-tokens 1`.

| Config | Run | Prefill t/s | MoE ms | layers ms | dense ms |
|--------|-----|-------------|--------|-----------|----------|
| **Default (all-BM16)** | 1 | **496.57** | 981.2 | 1022.9 | 41.7 |
| | 2 | **496.19** | 981.1 | 1023.4 | 42.3 |
| | 3 | **495.39** | 982.7 | 1025.0 | 42.3 |
| Control `HYBRID_BM=1` | 1 (cold-ish) | 365.39 | 1348.8 | 1391.3 | 42.5 |
| | 2 | **470.21** | 1038.2 | 1080.6 | 42.4 |
| | 3 | **469.44** | 1038.6 | 1080.8 | 42.2 |

**Warm default: ~495–497 t/s**  
**Warm hybrid-on: ~469–470 t/s**  
**Delta: +5.5% prefill; MoE wall −5.5% (1038 → 981 ms)**

### Timing profile (default warm)

| Bucket | ms @ T=512 | Share |
|--------|------------|-------|
| **MoE** | **~981** | **~96%** |
| dense L0 | ~42 | ~4% |
| layers total | ~1023 | 100% |

MoE still dominates. Per-layer ~25 ms (981/39) vs board target ~3.5 ms → **~7× MoE kernel gap** remaining.

### Other A/B (same host, prior NEW binary)

| Config | Warm prefill t/s | Notes |
|--------|------------------|-------|
| XROWS=1 | ~465–467 | parity vs materialise+BK32 |
| DUAL_QUEUE=0 | ~465 | parity |
| NAX=0 | ~416–421 | dense shared NAX still ~10% of MoE |
| MPP=0 (eidx) | ~212–242 | MPP remains critical |
| GATHER_BK=16 | ~457–465 | BK32 not slower; keep BK32 default |
| FUSED_QK_NORM_ROPE=0 | ~462–465 | prefill parity under thrash |

## Defaults after this hunt

| Knob | Default | Escape |
|------|---------|--------|
| BM16 BK | **32** | `BUTTER_NVFP4_GATHER_BK=16` |
| Hybrid BM64 wide-N | **OFF** | `BUTTER_NVFP4_HYBRID_BM=1` |
| Force all-BM64 | off | `BUTTER_NVFP4_MOE_BM64=1` / `GATHER_BM=64` |
| Gate/up pair | materialise (scratch) + BK32 concurrent | `BUTTER_NVFP4_MPP_XROWS=1` |
| GATHER_BK64 | **OFF** | `=1` (empty gen) |
| Shared expert scratches | grow-only | — |
| Dense shared NAX | ON | `BUTTER_NVFP4_QMM_NAX=0` |
| MPP gather | ON (T≥4) | `BUTTER_NVFP4_USE_GATHER_MPP=0` |
| Dual-queue shared | ON | `BUTTER_LAGUNA_DUAL_QUEUE=0` |

## Score snapshot (thrashy host)

| Metric | Now | Yukon #1 | Gap |
|--------|-----|----------|-----|
| Prefill @512 warm | **~496** | ~4030 | **~8.1×** |
| Decode short (thrash) | ~60 | ~131 | clean-host rebench needed |
| Quality | Paris + fib OK | — | — |

Score floor still prefill-bound: even with clean decode ~85–100, prefill ~496 → board needs MoE kernel class change.

## Why no bigger win

1. **MoE is 96% of wall** — micro-opts (xrows, dual-queue, gather alloc) are ≤noise once warm.
2. **Hybrid BM64 was a regression** relative to all-BM16 at Laguna mid=512 / down=2048 tile geometry (more TGs + BK32 BM16 occupancy beats BM64 TG packing here).
3. Board uses MLX gather-NAX BM=BN=BK=**64** + `tile_matmad_nax`. Iron has no true NVFP4 gather-NAX twin; `GATHER_BK64` full-model still empty-gen.

## Next iron kernel target

**Primary:** iron **BM16 × BK64** stage for mid=512 (gate/up nOut=512 stays BM16 occupancy, fewer K steps) — quality-first, do **not** default GATHER_BK64 BM64 path until Paris green.

**Secondary:** real gather-NAX (or iron twin of MLX `fp_gather_qmm_rhs_nax` BK=64 + proper dequant staging). Only path to multi-k t/s toward Yukon #1.

**Do not:** re-enable hybrid BM64 as default without a clean-host rebench that beats all-BM16; do not default GATHER_BK64.

## Files touched

1. `Sources/ButterSwift/Ops/OpsNVFP4.swift` — hybrid default OFF; gate/up gather scratch
2. `Sources/ButterSwift/Models/MoELayer.swift` — non-fused materialise grow-only scratch
