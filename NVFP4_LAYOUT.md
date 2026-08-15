# Qwen3.6-27B NVFP4 checkpoint — layout + kernel-compatibility map

Source checkpoint: `~/models/q27nvfp4-nokv/` (symlinks into `~/models/qwen3.6-27b-nvfp4/`).
Arch: `qwen3_5` / `Qwen3_5ForConditionalGeneration` — 64-layer hybrid GDN
(`linear_attention`) + full-attention (`full_attention`, every 4th layer),
hidden 5120, intermediate 17408, 24 attn heads / 4 KV heads. Produced by
**modelopt 0.45.0** (`hf_quant_config.json`, `producer.name = "modelopt"`).

> ⚠️ **CHECKPOINT DATA MISSING AT TIME OF WRITING.** `~/models/qwen3.6-27b-nvfp4/`
> (the symlink target for all four `.safetensors` shards + most sidecar files)
> no longer exists on disk — only `~/models/q27nvfp4-nokv/config.json` and
> `hf_quant_config.json` are real files; everything else in that directory is a
> **broken symlink**. `du`/`find` across `~/models/` turned up no remnant of the
> raw weights; disk is at 92% (71G free / 916G). A **newer** checkpoint,
> `~/models/qwen38-nvfp4/` (single-shard, "Qwen3.8-27B-NVFP4" per the HF cache
> dir name), appeared today and is almost certainly what displaced it — looks
> like the model family moved 3.6→3.8 and someone reclaimed the space. This is
> **not this task's checkpoint** (different shard layout, different HF repo).
>
> Practical effect: §1 below (tensor naming / dtype / shape) is documented from
> `hf_quant_config.json` + the public, well-established **modelopt NVFP4 HF
> export convention** — NOT verified against an actual safetensors header,
> because there is no safetensors file left to read. §5 (dequant-one-real-tensor
> empirical sanity check) is **blocked** for the same reason — see the bottom of
> this doc. Everything derived from `iron`'s kernel source (§2) IS verified
> (read directly from `~/iron`, which is intact). Re-run the header dump in §1
> the moment the checkpoint (or an equivalent Qwen3.6-27B modelopt NVFP4 export)
> is back on disk — the exact `python3`/struct-based header parser used to
tead the JSON in this doc is in the loader test harness described in §4.

## 0. Correction to task framing: this is NOT full dense-stack NVFP4

The task brief described the checkpoint as "FULL dense-stack coverage:
attention, GDN projections, FFN, lm_head all NVFP4" with `exclude_modules =
[mtp*]` as the only carve-out. **That's not what `hf_quant_config.json` says.**
It's `quant_algo: MIXED_PRECISION`:

| Module class | quant_algo | Notes |
|---|---|---|
| `mlp.gate_proj` / `up_proj` / `down_proj` (all 64 layers) | `W4A16_NVFP4`, `group_size: 16` | dense FFN — **this is the NVFP4 weight path** |
| `lm_head` | `W4A16_NVFP4`, `group_size: 16` | also NVFP4 |
| `self_attn.{q,k,v,o}_proj` (full-attention layers) | `FP8` | **not NVFP4** — no `group_size`, standard per-tensor FP8 |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (GDN layers) | `FP8` | **not NVFP4** |
| `mtp*`, `mtp.layers.0*` | (excluded) | left unquantized, per `exclude_modules` |

401 quantized-layer entries total: 193 `W4A16_NVFP4`, 208 `FP8`, 0 MoE-style
`experts.*` names (this model is dense, not MoE — `dequant_gemv_expert_indexed_block_scaled`
in iron's kernel suite is not applicable to this checkpoint). Confirmed via
`hf_quant_config.json["quantization"]["quantized_layers"]` (grouped by
layer-index-normalized suffix — every layer follows the same pattern; `lm_head`
is the only non-layer entry and is also NVFP4).

**This loader module targets the NVFP4 slice only** (FFN gate/up/down +
lm_head — the params that dominate B/param budget at hidden=5120,
intermediate=17408, 64 layers: `3 × 5120 × 17408 × 64 ≈ 17.1B` of the model's
~27B total). The FP8 attention/GDN tensors are a **separate, smaller** loader
concern (standard per-tensor FP8, `weight` fp8_e4m3 + one `weight_scale` fp32
scalar, no block structure) — flagged for the correctness-port agent
(`qwen36-spark-port`), not built here.

## 1. Checkpoint tensor layout (modelopt NVFP4 convention, per `group_size: 16`)

For each `W4A16_NVFP4` linear `<base>` (e.g.
`model.language_model.layers.0.mlp.gate_proj`), modelopt's HF export emits
three tensors:

| Tensor | dtype (safetensors tag) | Shape | Meaning |
|---|---|---|---|
| `<base>.weight` | `U8` (packed FP4, 2×E2M1 nibbles/byte) | `[out_features, in_features / 2]` | packed 4-bit weight |
| `<base>.weight_scale` | `F8_E4M3` (raw byte codes) | `[out_features, in_features / 16]` | per-16-column block scale, E4M3 |
| `<base>.weight_scale_2` | `F32` | `[1]` or scalar | one **per-tensor** global scale (`amax/(6·448)` convention — matches the `nvfp4_globals` formula already in `wh-butter-ops`' CUDA path) |

Byte packing within `<base>.weight`: **low nibble = even column index, high
nibble = odd column index** within each byte — the standard modelopt/TRT-LLM
NVFP4 convention (same as OCP MX packing). `in_features` a multiple of 16
(`group_size`) guarantees `in_features/2` a whole number of bytes and
`in_features/16` a whole number of scale bytes.

`weight_scale` dtype: safetensors added the `F8_E4M3`/`F8_E5M2` tags some time
back; if this producer instead emits raw `U8` for the scale tensor too (some
modelopt export paths do, treating fp8 as an opaque byte type) the *bytes* are
identical either way — this loader treats `weight_scale` as raw `u8` E4M3 codes
regardless of which tag the header carries (see §3, `parse_dtype` extension).

## 2. iron kernel expectations (read from `~/iron`, verified against source)

Every block-scaled kernel in the suite the task named
(`gemm/block_scaled_matmul.rs::iron_nvfp4_qgemv`,
`gemm/batched_4_block_scaled_qgemv.rs::iron_nvfp4_batched_4_qgemv`,
`gemm/block_scaled_qmm.rs::iron_nvfp4_qmm`,
`gemm/block_scaled_mma.rs::iron_nvfp4_qmm_mma`,
`norm/rms_norm_block_scaled_qgemv.rs`, `norm/gated_rms_norm_block_scaled_qgemv.rs`)
uses **the same flat layout** for the `nvfp4` format variant
(`BITS=4, WT=u32, ST=u8, WDEC=0 (E2M1 nibble), SKIND=1 (E4M3 micro × global)`):

```
weight  [rows, in_dim/8]        u32   (8 E2M1 nibbles/word, little-endian:
                                        nibble i = bits[4i..4i+4), element
                                        index = i, i.e. increasing column
                                        with increasing nibble/bit position)
scales  [rows, in_dim/block_size] u8  (raw E4M3 byte code)
global  f32                            (ONE scalar per weight matrix — a
                                        `#[constexpr(only_when="SKIND==1")]`
                                        kernel-dispatch parameter, not a buffer)
```

Decode (from `kernels/primitives.rs`, transcribed):
```
element = E2M1_decode(nibble)             // {0,0.5,1,1.5,2,3,4,6} × sign bit
block_scale = E4M3_decode(scale_byte)     // standard OCP FP8 E4M3, bias 7
w = element * block_scale * global
```
`block_size` is a runtime dispatch parameter (any multiple of 8); no kernel
hard-codes 16, but the checkpoint's `group_size: 16` must be passed through
at dispatch time.

`flash_block_scaled_sdpa.rs` (quantized-KV-cache flash attention) is **not
applicable** to this checkpoint: this checkpoint is the `-nokv` variant
(KV cache is not NVFP4-quantized) and, per §0, attention itself is FP8 here,
not NVFP4.

## 3. Kernel-compatibility verdict: DIRECT, no re-swizzle

The task brief flagged a possible "scale re-swizzle flat→CUTLASS atom layout"
requirement (per the Laguna scoping notes). **That applies to a different code
path** — `wh-butter-ops`' own native-CUDA CUTLASS/cuBLASLt grouped-MoE NVFP4
kernels (`moe_grouped_gemm_cutlass`, `nvfp4_pack_w`, `moe_fp4.cu`) use a CUTLASS
canonical swizzle atom (`ceil(m/128)*512*ceil(kb/4)` bytes, per the comment at
`wh-butter-ops/src/lib.rs:9593`). **iron's own DSL kernels
(`iron_<fmt>_qgemv`, `_qmm`, `_qmm_mma`, `_batched_4_qgemv`) do not use that
swizzle** — they read the flat `[rows, in_dim/block_size]` row-major scale
layout directly, confirmed identically across the GEMV, fused-RMSNorm-GEMV,
and MMA-GEMM variants (§2).

Mapping checkpoint → kernel:

| Checkpoint tensor | → | Kernel input | Transform |
|---|---|---|---|
| `<base>.weight` `[out, in/2]` u8 | → | `weight` `[out, in/8]` u32 | **zero-copy reinterpret**: 4 consecutive packed bytes = 1 u32 (both little-endian on GB10/ARM and on the safetensors-standard byte order; nibble-within-byte order already matches iron's within-word nibble order — see caveat below) |
| `<base>.weight_scale` `[out, in/16]` u8/f8e4m3 | → | `scales` `[out, in/16]` u8 | **direct** — same shape, same raw byte semantics (E4M3 code) |
| `<base>.weight_scale_2` scalar f32 | → | `global: f32` constexpr dispatch param | **direct** |
| n/a | → | `block_size: u32` constexpr dispatch param | pass `16` (from `hf_quant_config.json` `group_size`) |

**Caveat requiring empirical verification once the checkpoint is restored:**
the u8→u32 reinterpret assumes iron's "nibble i occupies bits `[4i,4i+4)` of
the u32, element index = i" convention lines up with modelopt's "low nibble =
even column, high nibble = odd column" *at the byte→u32 boundary* (i.e. byte 0
of the packed row → bits 0-7 of the u32 → nibbles 0,1 → columns 0,1; byte 1 →
bits 8-15 → nibbles 2,3 → columns 2,3; etc.). This is the standard
little-endian byte-to-word layout and is extremely likely correct (it's the
same assumption `wh-butter-loader`'s existing MLX-quant reinterpret already
makes for the unrelated int4 MLX path — see `mlx_dequant_f32` in `lib.rs`,
which does exactly this `u32::from_le_bytes` reinterpret of a nibble-packed
byte buffer). Flagged, not proven, because the source bytes to test it against
don't exist right now (§ blocker note at top).

**Verdict: DIRECT compatibility, no kernel-side or loader-side re-swizzle
needed for the FFN/lm_head NVFP4 path**, modulo the one-line caveat above.

## 4. Loader interface (built, `wh-butter-loader`)

`crates/wh-butter-loader/src/nvfp4.rs` (new module), on `SafeTensors`:

```rust
pub struct Nvfp4Tensor<'a> {
    pub packed: &'a [u8],   // [rows, cols/2] raw bytes (2 nibbles/byte)
    pub scales: &'a [u8],   // [rows, cols/block_size] raw E4M3 byte codes
    pub scale2: f32,        // per-tensor global scale
    pub rows: usize,
    pub cols: usize,        // = in_features
    pub block_size: usize,  // from weight_scale's shape (cols / scales_row_len)
}

impl SafeTensors {
    /// Zero-copy: reads `<base>.weight` / `.weight_scale` / `.weight_scale_2`.
    pub fn load_nvfp4_tensor(&self, base: &str) -> Result<Nvfp4Tensor<'_>>;

    /// Host dequant to a dense `[rows, cols]` f32 matrix — for sanity checks
    /// and non-perf-critical CPU paths. Mirrors the existing `mlx_dequant_f32`.
    pub fn dequant_nvfp4_tensor_f32(&self, base: &str) -> Result<(Vec<f32>, Vec<usize>)>;
}
```

`packed`/`scales` are borrowed directly from the mmap — no copy, no upload
until a caller decides to. `block_size` is read back from the tensor shapes
(`cols / scales.len()/rows`) rather than hard-coded, so it stays correct if a
future checkpoint uses a different `group_size`. The caller (`qwen35.rs`, once
the port agent gets there) is expected to `d.upload(tensor.packed)` /
`d.upload(tensor.scales)` and dispatch `iron_nvfp4_qgemv` /
`iron_nvfp4_qmm_mma` directly against those buffers — no intermediate
transform, per §3.

`parse_dtype` in `lib.rs` gained `"F8_E4M3"` / `"F8_E5M2"` → `DType::U8`
(byte-identical reinterpret; no float decode happens in the dtype layer,
kernels/host code decode the E4M3 bit pattern themselves) so that
`weight_scale` tensors aren't silently dropped from the shard index the way
unsupported dtypes are today (see the `// Omni/multimodal checkpoints...`
comment in `open_shard` — anything `parse_dtype` doesn't recognize gets
skipped rather than erroring, which would have made `weight_scale` invisible
to `tensor()` lookups without this change).

## 5. Empirical dequant sanity check — BLOCKED

Step 5 of the task ("dequant ONE real checkpoint tensor... validates the
layout doc empirically") **could not be performed** — there is no checkpoint
data left to dequant (see the blocker note at the top of this doc). The
loader code (`dequant_nvfp4_tensor_f32`) and a ready-to-run check are in place
(`crates/wh-butter-loader/tests/nvfp4_checkpoint_smoke.rs`, `#[ignore]`d with
a comment) — pointed at `~/models/q27nvfp4-nokv`, it loads
`model.language_model.layers.0.mlp.down_proj`, dequants it, and asserts
no-NaN + a plausible magnitude range. Run it the moment the checkpoint (or an
equivalent) is back:
```
cargo test -p wh-butter-loader --test nvfp4_checkpoint_smoke -- --ignored --nocapture
```

## 6. Unit test — synthetic tensor, GPU-verified

`crates/backends/wh-butter-cuda/tests/nvfp4_qgemv_layout_test.rs`: packs a
small synthetic `[16, 32]` weight (in_dim=32, block_size=16 → 2 blocks/row) on
the host exactly per §1/§3's convention (2 nibbles/byte, E4M3 block scale,
one f32 global), uploads it, dispatches `iron_nvfp4_qgemv`, and compares
against an f32 host reference computed by the same decode math as §2. See
report for the pass/fail result and measured max relative error.
