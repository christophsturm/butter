# butter-prefill-bench agent report — T16 stabilize prefill

**Worktree:** `/Users/tom/dev/butter-wts/butter-prefill-bench` (`tom/feat/laguna-prefill-bench`)  
**When:** 2026-07-29 ~16:30 UTC  
**Host:** M5 Max, swap used ~10.1 GB sticky thrash; `ai.useraven.mlx-server` still active. Prefill less sensitive than decode.

## 1. Defaults confirmed (code)

### Shared expert grow-only scratches — ON default path
`MoELayer.prefillSharedExpertNvfp4` (`Sources/ButterSwift/Models/MoELayer.swift` ~3199+):
- `prefSharedMidScratch`, `prefSharedDownScratch`, `prefSharedGuScratch`, `prefSharedGateScratch` / `prefSharedUpScratch`
- Grow-only: reallocate only when dtype mismatch or `elementCount < need`
- Called from default prefill shared path when `BUTTER_NVFP4_NO_SHARED` unset

### BK32 default for BM16 path — ON
`Ops.nvfp4MoeGatherBm16` (`OpsNVFP4.swift` ~320–332):
- `useBk32 = !forceBk16 && kIn % 32 == 0`
- `forceBk16` only if `BUTTER_NVFP4_GATHER_BK=16`
- Dispatches `iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_*`

Gate/up pair default (`nvfp4MoeGatherBm16Pair` path ~476–504):
- Materialise gather once + concurrent iron BK32 pair
- Escape xrows: `BUTTER_NVFP4_MPP_XROWS=1`

### Hybrid BM64 for nOut ≥ 1024 — ON
`nvfp4MoeGatherPrefersBm64` (~140–163):
- Hybrid default: `!hybridOff && nOut >= 1024` → BM64 (down proj n=2048)
- Escape: `BUTTER_NVFP4_HYBRID_BM=0` or `GATHER_BM=16`

### GATHER_BK64 — OFF (empty gen)
`nvfp4MoeGatherPrefersBm64Bk64` (~165–173): only if `BUTTER_NVFP4_GATHER_BK64=1`. Keep opt-in.

## 2. Build

```bash
cd /Users/tom/dev/butter-wts/butter-prefill-bench
swift build -c release --build-path /tmp/butter-prefill-bench-build --product butter
# OK in ~281s → /tmp/butter-prefill-bench-build/release/butter
```

No kernel regen needed (BK32 + related kernels already in metallib / IronKernels.swift).

## 3. Quality

| Prompt | Result | Decode t/s (thrash) |
|--------|--------|---------------------|
| "What is the capital of France?" | **OK** — "The capital of France is Paris." | ~47.9 |
| "Write a rust fib function" | **OK** — coherent Rust `fn fib` | ~49.5 |

## 4. Prefill T=512 (default)

Prompt: `python3 -c "open('/tmp/p512.txt','w').write(('hello world ')*255)"` → **512** tokens.

```bash
BUTTER_LAGUNA_TIMING=1 /tmp/butter-prefill-bench-build/release/butter generate \
  --model ~/models/laguna-xs-2.1-nvfp4 --prompt "$(cat /tmp/p512.txt)" --max-tokens 1 --stats
```

| Run | Prefill t/s | MoE ms | layers ms | dense ms |
|-----|-------------|--------|-----------|----------|
| 1 | 388.02 | 1196.0 | 1311.6 | 115.6 |
| 2 | **406.69** | 1183.1 | 1250.7 | 67.6 |
| 3 | **402.98** | 1196.4 | 1262.5 | 66.1 |

**Warm default: ~403–407 t/s; MoE ~1183–1196 ms** (~95% of layer wall).

## 5. A/B escape path

Env: `BUTTER_NVFP4_GATHER_BK=16 BUTTER_NVFP4_MPP_XROWS=1 BUTTER_NVFP4_HYBRID_BM=0`

| Run | Prefill t/s | MoE ms | layers ms |
|-----|-------------|--------|-----------|
| 1 (cold) | 273.71 | 1778.9 | 1862.2 |
| 2 | **422.89** | 1128.9 | 1202.0 |
| 3 | **410.39** | 1171.2 | 1239.3 |

Warm A/B **~410–423 t/s** — slight edge vs default under thrash (matches T14). Not a quality regression signal either way; keep BK32+hybrid as product default.

## 6. Docs

Appended **T16** section to `benchmarks/laguna-xs/README.md` with numbers + best env defaults table.

## Best env defaults (no env = this)

| Knob | Default | Escape |
|------|---------|--------|
| BM16 BK | **32** | `BUTTER_NVFP4_GATHER_BK=16` |
| Hybrid BM | **ON** (nOut≥1024 → BM64) | `BUTTER_NVFP4_HYBRID_BM=0` |
| Gate/up | gather + BK32 pair | `BUTTER_NVFP4_MPP_XROWS=1` |
| GATHER_BK64 | **OFF** | `=1` (empty gen) |
| Shared scratches | grow-only | — |
| Dense QMM NAX | ON | `BUTTER_NVFP4_QMM_NAX=0` |
| MPP gather | ON | `BUTTER_NVFP4_USE_GATHER_MPP=0` |

## Notes / gaps

- Prefill gap to board still ~9× (405 vs 3651). MoE still ~30 ms/layer vs ~3.5 ms board target.
- Decode thrash ~48–50; clean-host historical ~85 vs board 102. Drain swap / unload mlx-server for fair decode.
- No commit/push (per brief).
