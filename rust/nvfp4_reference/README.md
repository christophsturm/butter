# NVFP4 (mxf4nvf4) reference harnesses — GB10 / sm_121a

Standalone CUDA probes that reverse-engineered + validated the native Blackwell
**NVFP4 block-scaled tensor-core mma** for the Nemotron-H prefill MoE GEMM
(closing the gap to vLLM's 6395 tok/s @S2048; the Q4 grouped path is ~10 TFLOP/s,
NVFP4 mma peaks ~386 TFLOP/s throttled on GB10).

The instruction (validated computing bit-exact):
```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
```
Build (accelerated-arch-only — base sm_121 PTX is rejected by ptxas):
```
nvcc -gencode arch=compute_121a,code=sm_121a <file>.cu -o <out>
```

## Files (build order = discovery order)
| file | what it establishes |
|------|---------------------|
| `fp4_probe.cu` / `nvfp4_fix.cu` | the mxf4nvf4 mma **assembles** on sm_121a; the scale token is `ue4m3` (unsigned), not `e4m3` |
| `fp4_compute.cu` | mma **computes correctly** (all-ones → D=64); C/D accumulator layout (c0,c1→row gid,col tig*2+{0,1}; c2,c3→row gid+8); e2m1 1.0=0x2, ue4m3 1.0=0x38 |
| `fp4_derive.cu` / `fp4_deriveB.cu` | empirical A/B fragment **row-map** (one-hot sweep): A reg0,2→gid / reg1,3→gid+8; B reg0,1→gid |
| `fp4_validate.cu` | full random-tile **data layout** bit-exact vs f32 (scales=1) |
| `fp4_scale.cu` / `fp4_scaleB.cu` / `fp4_byteblk.cu` / `fp4_perm.cu` | derive the **scale operand** layout (scale_vec::4X, byte→block, lane mapping) |
| **`fp4_aligned.cu`** | **THE reference loader** — full layout + random ue4m3 scales bit-exact vs f32 (bad=0/128). Mirror this in the kernel. |
| `qnvfp4_test.cpp` | weight quant accuracy: NVFP4 ≈ Q4 rel-rmse (host) |
| `nvfp4_gemm.cu` | multi-K-tile accumulation + per-tensor global-scale epilogue |
| `nvfp4_peak.cu` | raw mma throughput microbench (386 TFLOP/s throttled @539MHz) |
| `nvfp4_tiled.cu` | tiled GEMM at real expert dims [96×1856×2688] = 58 TFLOP/s (~6× Q4) |
| **`nvfp4_packed.cu`** | **in-kernel gather** from the `quantize_nvfp4` packed `[N, codes(K/8 u32)+scales(K/16 u8)]` layout — the core of the production kernel |
| `nvfp4_actq.cu` | device activation quantizer: f16 `[rows,K]` → NVFP4 codes+scales+global |

These are dev/reference, NOT part of the cargo build. The production path is
`quantize_nvfp4` (ffai-ops, committed) + the forthcoming `MOE_FP4_GROUPED_MMA_SRC`.
Full state + the exact fragment/scale layout recipe: memory note
`nemotron-grouped-moe-gemm.md` (2026-06-08 sections).
