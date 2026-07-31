# butter QK-norm + RoPE fuse agent report

**Worktree:** `/Users/tom/dev/butter-dev`  
**Branch:** `tom/feat/laguna-swift-xs` (dirty main tree)  
**Build:** `/tmp/butter-qk-fuse-build/release/butter`  
**No commit/push.**

## Goal

Port mlxfast’s highest-ROI remaining **decode** win: fused QK RMSNorm + plain
RoPE for sliding-attention layers (`DARKBLOOM_FUSED_SLIDING_QK_NORM_ROPE`).
Decode-only; prefill left alone.

## What landed

### Live-Metal kernel `LagunaFusedQKNormRoPE`

File: `Sources/ButterSwift/Models/Text/LagunaSdpa.swift`

- Kernels:
  - `laguna_sliding_qk_norm_rope_bf16_128`
  - `laguna_sliding_qk_norm_rope_f16_128`
- Shape: **one threadgroup per head**, TPG=32 (1 simdgroup), head_dim=128
- Grid: `nQHeads + nKVHeads` (XS SWA: 64+8 = 72 TGs/layer)
- RMSNorm: 4 els/lane, `simd_sum` SSQ, **local `rsqrt` — no threadgroup barrier**
  (mlxfast lesson: barrier was free waste when only 1 SG; 72×30 = 2160/token)
- Norm rounding: iron-order `bfloat((x * inv) * w)` then back to f32 for RoPE
  so intermediate boundary matches stock `rmsNorm → rope`
- RoPE: plain rotate-half pairs `p ↔ p+64`, inv_freq =
  `exp2((-p) * log2(theta) / 64)` — matches `iron_rope_banded` + `RoPEScaling.none`
- Cos/sin computed in-kernel from `position` + `theta_base` (no angle atlas)

### Wire-up in decode path

File: `Sources/ButterSwift/Models/Text/LagunaText.swift` — `LagunaLayer.forward`

```
if LagunaFusedQKNormRoPE.canFuse(isSliding, rotaryDim==headDim, headDim==128, bf16|f16):
    one dispatch → (qN, kN)
else:
    stock rmsNormRows ×2 + applyRopeInPlace
```

- **Default ON** (`BUTTER_LAGUNA_FUSED_QK_NORM_ROPE != "0"`)
- Escape: `BUTTER_LAGUNA_FUSED_QK_NORM_ROPE=0`
- Guards: sliding only, full rotary (`rotaryDim == headDim`), head_dim 128
- Full-attn YaRN / partial rotary / prefill: **unchanged stock path**

### Dispatch savings (XS, 30 SWA layers)

| Path | Dispatches / token (SWA QK-norm+RoPE) |
|------|--------------------------------------|
| Stock | 30 × 4 = **120** (q-norm, k-norm, rope-q, rope-k) |
| Fused | 30 × 1 = **30** |

Full layers (10×) still use stock YaRN partial RoPE.

## Quality gates

### Paris (default fuse ON)

```
/tmp/butter-qk-fuse-build/release/butter generate \
  --model ~/models/laguna-xs-2.1-nvfp4 \
  --prompt "What is the capital of France?" --max-tokens 48 --stats
```

| Metric | Value |
|--------|-------|
| Answer | **Paris** ✓ |
| decode | **59.40 tok/s** (first run) |
| decode (steady) | 59.27 tok/s |

### Fib (default fuse ON)

Prompt: `Write a Python function that returns the nth Fibonacci number.`

| Metric | Value |
|--------|-------|
| Quality | Coherent Fibonacci discussion ✓ |
| decode | **59.01 tok/s** |
| decode (steady) | 59.00 tok/s |

## A/B: fuse ON vs OFF (128 gen tokens, 3 runs each)

| Config | decode t/s (3 runs) | steady mean |
|--------|---------------------|-------------|
| **fuse OFF** (`=0`) | 59.17 / 58.06 / 58.35 | **~58.6** |
| **fuse ON** (default) | 59.59 / 59.70 / 59.13 | **~59.5** |

Approx **+1.5%** decode vs stock 4-dispatch chain — same ballpark as mlxfast’s
reported +1.19% / +1.73% after the barrier fix. All Paris answers correct.

## Prefill

Untouched. Fuse gated on decode `forward` only; `forwardMany` prefill path
still uses batched `rmsNormRows` + `ropePartialManyTwo` / YaRN.

## Files touched

| File | Change |
|------|--------|
| `Sources/ButterSwift/Models/Text/LagunaSdpa.swift` | `LagunaFusedQKNormRoPE` live Metal kernels + API |
| `Sources/ButterSwift/Models/Text/LagunaText.swift` | Decode branch behind env flag |

## Escape / ablation

```bash
# stock path
BUTTER_LAGUNA_FUSED_QK_NORM_ROPE=0 butter generate ...

# fused (default)
butter generate ...
# or explicit:
BUTTER_LAGUNA_FUSED_QK_NORM_ROPE=1 butter generate ...
```

## Not done (intentionally)

- Prefill multi-token QK-norm+RoPE twin (`DARKBLOOM_PREFILL_QK_NORM_ROPE`)
- Full-attn fused QK-norm + YaRN (`DARKBLOOM_FUSED_FULL_QK_NORM_YARN`)
- Angle atlas preload (kernel derives cos/sin; stock-matched formula)
- Commit/push
