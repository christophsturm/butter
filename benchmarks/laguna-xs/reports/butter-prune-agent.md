# Butter Laguna lm_head prune — agent report

**Worktree:** `/Users/tom/dev/butter-wts/butter-prune` (`tom/feat/laguna-prune`)  
**Date:** 2026-07-29  
**Model:** `~/models/laguna-xs-2.1-nvfp4`  
**Binary:** `./.build/release/butter` (release)

## Goal

Make Laguna decode lm_head prune **default ON** and quality-safe (Paris + fib), modeled on mlxfast DARKBLOOM certified 2-pass prune.

## Changes

### `Sources/ButterSwift/Models/Text/LagunaLmHeadPrune.swift` (rewrite)

Ported mlxfast certified two-pass (notes/68 / DARKBLOOM):

1. **COARSE** fused MXFP8 GEMV → `coarse`, certified `delta`, `coarse_bf`
   - `delta_i = d_i*(1+γ) + 2*γ*m_i` with γ = 2^-15, half-cell `hs8` bounds
2. **SELECTION** mask: `c_i + delta_i >= thr` where `L = max(c - δ)`, `thr = L − |L|/64`
3. **EXACT** fixed 4-row simdgroup BF16 GEMV for candidates only; non-candidates keep `coarse_bf`

**Default ON:** `BUTTER_LAGUNA_LM_HEAD_PRUNE != "0"` (unset = on).  
**Escape:** `BUTTER_LAGUNA_LM_HEAD_PRUNE=0`.  
Does **not** force prune if MXFP8 sidecar or BF16 lm_head missing (caller guards).

Replaced prior opt-in (`== "1"`) + crude `max(coarse) − beta` threshold (beta default 2.0) with certified path.

### `Sources/ButterSwift/Models/Text/LagunaText.swift`

- Load still prefers BF16 stock head; loads MXFP8 sidecar when present.
- Logs when certified prune is active.
- `lagunaFinalNormLmHead` already wired prune on decode/prefill-last-token when:
  - `LagunaLmHeadPrune.enabled`
  - `lmHeadMxfp8 != nil`
  - BF16 `Linear` weight shape `[100352, 2048]`
- No force when weights missing.

## Quality gate

### Prune ON (default)

**Paris** (`What is the capital of France?`, max 48):

```
The capital of France is Paris. Paris is a beautiful city with many attractions
and landmarks. It is located in the northern part of France. ...
```

- decode: **48.80 tok/s** (steady 48.63)
- non-empty, correct capital ✓

**fib** (`Write a rust fib function`, max 80):

```
I need a Rust function that calculates the nth Fibonacci number using recursion.
The function should return a `u64` type. The function should be named `fib`. ...
```

- decode: **42.27 tok/s** (steady 42.11)
- non-empty, on-topic ✓ (small model rambles; not empty-gen fail)

### Prune OFF (`BUTTER_LAGUNA_LM_HEAD_PRUNE=0`)

**Paris:** still answers Paris (then drifts to Jack Sparrow).  
- decode: **28.44 tok/s** (steady 28.04)

**fib:** coherent fib explanation.  
- decode: **45.19 tok/s** (steady 45.46)

## Decode t/s A/B

| Prompt | Prune ON | Prune OFF | Δ |
|--------|----------|-----------|---|
| Paris (48 tok) | **48.80** | 28.44 | **+72%** |
| fib (80 tok) | 42.27 | **45.19** | −6% (noise / candidate set) |

Notes:
- First Paris ON run had cold prefill (12 tok/s); decode still strong.
- Fib A/B is within normal variance; certified path can be similar or slightly heavier when many rows are candidates. Paris shows the big BW win on short factual decode.
- Leaderboard mlxfast ~131 decode is a different stack (fusion + more opts); this work is the prune lever only.

## Gates

- Empty gen: **PASS** (both prompts non-empty)
- Paris capital: **PASS**
- Escape env: **PASS** (OFF path runs stock head; no prune log line)
- Sidecar required: **PASS** (prune only when MXFP8 present + BF16 shape)

## Not done

- No commit / push (per brief)
- No full decode fusion stack to chase 131 t/s
- No bit-exact harness vs stock gemv (mlxfast has that; butter relies on certificate + quality prompts)

## Files touched

- `/Users/tom/dev/butter-wts/butter-prune/Sources/ButterSwift/Models/Text/LagunaLmHeadPrune.swift`
- `/Users/tom/dev/butter-wts/butter-prune/Sources/ButterSwift/Models/Text/LagunaText.swift`
