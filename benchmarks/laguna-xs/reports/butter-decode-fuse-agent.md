# butter-decode-fuse agent report

**Worktree:** `/Users/tom/dev/butter-wts/butter-decode-fuse`  
**Branch:** `tom/feat/laguna-decode-fuse`  
**Build:** `/tmp/butter-decode-fuse-build/release/butter`  
**No commit/push.**

## Goal

Port mlxfast **decode-only** NVFP4 MoE fusions (DARKBLOOM-style) to Butter Path B to move Laguna XS 2.1 decode toward 100+ tok/s without quality loss.

## What landed

### 1. Fast iron-exact NVFP4 qdot (shared by all fuse kernels)

In `LagunaNvfp4Fuse.swift` header:

- mlxfast half bit-twiddle E2M1 pack decode (`* 16384`) — **bit-exact** vs `iron_nvfp4_e2m1` for all 16 nibbles / 1000 random packs
- iron E4M3 scale path unchanged (matches mlx nofold scale, ratio 1.0)

### 2. `FUSED_ROUTED_SWIGLU_QMV` → default **ON**

- Kernel: `laguna_moe_routed_swiglu_qmv_nvfp4_bf16`
- One dispatch: 8 expert slots × 128 tiles, TPG 64
- Separate stacked gate/up banks (no fused-bank build)
- Tile-major expert interleave (mlxfast scheduling)
- Emits 8 mid `[512]` BF16 activations (SwiGLU in-kernel)
- Escape: `BUTTER_LAGUNA_PATHB_ROUTED_SWIGLU=0`

### 3. `FUSED_ROUTED_DOWN_REDUCE` → default **ON**

- Kernel: `laguna_moe_routed_down_reduce_nvfp4_bf16`
- One TG = 8 SGs (one per topK slot), 512 tiles × 4 rows
- Concurrent down QMV + BF16 weighted reduce
- Router weights already include `routedScalingFactor` (no hard-coded 2.5)
- Escape: `BUTTER_LAGUNA_PATHB_DOWN_REDUCE=0`

### 4. Shared SwiGLU QMV → remains **opt-in**

- Existing `BUTTER_LAGUNA_FUSED_SHARED_SWIGLU=1`
- Measured **regression** on M5 vs iron dual QMV (~−10% when combined)
- Kept opt-in only

### 5. Residual + RMS

- Already present in `LagunaText` decode via `Ops.addAndRmsNorm` (unchanged)

### Prefill

- Untouched. All new fuses are T=1 / Path B NVFP4 decode only.

## Gate results

### Official Paris gate (defaults)

```
/tmp/butter-decode-fuse-build/release/butter generate \
  --model ~/models/laguna-xs-2.1-nvfp4 \
  --prompt "What is the capital of France?" --max-tokens 48 --stats
```

| Metric | Value |
|--------|-------|
| Answer | **Paris** ✓ (`The capital of France is Paris.`) |
| decode | **53.87 tok/s** |
| decode (steady, tokens 11+) | **53.98 tok/s** |
| prefill | 20.95 tok/s (8-token prompt) |
| load | 3.72s |

### A/B (same binary, env escapes)

| Config | decode t/s | Paris? |
|--------|------------|--------|
| **defaults** (routed swiglu + down reduce) | **~54** | yes |
| stock (both OFF) | ~47 | yes |
| routed swiglu only | ~51 | yes |
| down reduce only | ~46 | yes |
| + shared swiglu ON | ~45–46 | yes (slower) |

Approx **+15%** decode vs stock concurrent iron eidx + swigluMany + FMA.

### Fib smoke

Prompt `Write a rust fib function` / 96 tokens: **~54.6 t/s** steady; coherent `fn fib(n: i32) -> i32` discussion. Not a strict unit test.

## Files touched

- `Sources/ButterSwift/Models/Text/LagunaNvfp4Fuse.swift` — kernels + flags
- `Sources/ButterSwift/Models/MoELayer.swift` — `decodeGPUPathBNvfp4` wiring

## Env reference (decode)

| Env | Default | Meaning |
|-----|---------|---------|
| `BUTTER_LAGUNA_PATHB_ROUTED_SWIGLU` | ON (`!=0`) | fused gate+up QMV+SwiGLU |
| `BUTTER_LAGUNA_PATHB_DOWN_REDUCE` | ON (`!=0`) | fused down+weighted reduce |
| `BUTTER_LAGUNA_FUSED_SHARED_SWIGLU` | OFF (`==1` opt-in) | shared fused bank QMV |
| `BUTTER_LAGUNA_PATHB_GATEUP_FUSE` | OFF | older per-row gateup (superseded) |
| `BUTTER_LAGUNA_PATHB_NVFP4_FUSED` | OFF | serial chain8 (slow) |
| `BUTTER_LAGUNA_PATHB_NVFP4_SLOT_FUSE` | OFF | per-slot swiglu+down |

## Gap to 100+ tok/s

Still ~**2× short** of 100 tok/s. These fusions cut dispatch + mid materialization and speed dequant ALU, but:

1. Runtime-compiled Metal still trails precompiled iron metallib for some shapes
2. Shared expert iron path + dual-queue still dominate part of the MoE step
3. Attention / residual / router / 40-layer host CB pipe not fused here
4. mlxfast’s larger stack (residual+RMS+router, QK-norm+RoPE, shared-down residual, scale-fold in iron metallib) is not fully ported

### Highest-ROI next steps (not done)

1. Port fast half-bitpath qdot into **iron** `iron_nvfp4_dequant_gemv_expert_indexed_*` / `iron_nvfp4_qgemv_*` metallib (universal win)
2. mlxfast `FUSED_RESIDUAL_RMS_ROUTER` (residual+RMS+router proj one kernel)
3. Default dual-queue off or shared-on-main-queue early with iron only
4. Decode `layersPerCB` tuning + event reduction
5. mlxfast routed+shared 9-slot SwiGLU single dispatch (if shared fuse ever wins)

## Quality notes

- Paris gate: **pass**
- Arithmetic intent: iron-exact e2m1/e4m3; SiLU in float then BF16 store (matches `Ops.swiglu`)
- Down reduce BF16 weighted sum approximates `scalarFMAChain8` slot order 0..7
- If quality ever drifts: set both fuse escapes to `0` for stock Path B
