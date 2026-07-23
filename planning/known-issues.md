# Known Issues

Open bugs + rough notes from the model-family port. Each entry carries
enough context to pick up cold. Resolved-work history lives in
`planning/session-plan.md` / `plan.md`, not here.

## Qwen3-VL-tower VIDEO path degenerate — surfaced 2026-06-04

`Qwen35VisionIntegrationTests.videoTextGeneration` (`mlx-community/Qwen3.5-2B-4bit`,
the Qwen3-VL ViT tower via `Qwen35VL.load`) greedy-decodes **degenerate**
output on cat.mp4: 200 tokens at only **24 % unique**, stuck emitting
`<think>` with no content — the cat is never described. The IMAGE path on the
SAME tower is healthy (54 % unique, correctly describes the dog), and the
video-token COUNT matches (the splice doesn't throw), so the per-token
**video FEATURES** are wrong, not the placeholder geometry. Prime suspects:
the Qwen3 tower's `encode(frames:)` temporal-patch fold (`temporal_patch_size`
= 2) or the video M-RoPE position assignment. First time the Qwen3-VL-tower
video path has had integration coverage (the suite previously pointed at the
wrong model — Qwen3-VL-30B-A3B). Likely also affects
`Qwen3VisionIntegrationTests.videoTextGeneration` (same tower); the full
integration bisect will confirm. Debug via an mlx-vlm video-feature parity
dump (same method that fixed LFM2-VL). NOT a placeholder/count bug.

## GPU LSTM (`iron_lstm`) — context-dependent NaN in the full Kokoro chain

**Symptom.** With the Kokoro front-end wired to run its LSTMs on the GPU
(`KokoroFrontEndGPU.biLstmGPU` → `Ops.lstm`), the full `synthesize`
chain produces an all-NaN waveform — first NaN is the `predictor.shared`
biLSTM output at T≈65. The CPU `KokoroMath.biLSTM` on the **identical**
input (`dAligned`, range ±2.27) is finite; proven with a side-by-side
diagnostic (`shGPUnan=true` / `shCPUnan=false`).

**Why it's not numerical.** `predictor.shared` weights are bounded
(absmax ≈ 1.26) and an LSTM cell is mathematically bounded
(`|c_new| ≤ |c_old| + 1`, gates ∈ [0,1], tanh ∈ [-1,1]) — so a correct
LSTM **cannot** emit NaN from finite, bounded input. The NaN must be a
kernel/resource fault (uninitialised threadgroup/buffer read), not
overflow.

**Why it's elusive.** The kernel is correct **in isolation**: two
regression tests at the real shape (hidden 256 = 8 simdgroups, seqLen 65)
both pass — `lstmLongSeqHidden256` (plain) and
`lstmBidirectionalSharedOutput` (the exact `outStride`/`outOffset`/
`reverse` shared-output pattern `biLstmGPU` uses), each 12–16 repeated
dispatches, finite + bit-deterministic + matching the CPU oracle. The
NaN only appears **after the full chain's many prior GPU dispatches**
(PLBERT 12 layers + bert/text encoders + per-block decoder/generator
work) — a context/state interaction, possibly iron buffer-pool
aliasing or a Metal threadgroup-memory hazard that needs that much prior
GPU traffic to surface. Not reproducible standalone, so not yet bisectable.

**Mitigation in place.** Kokoro's front-end LSTMs (text_encoder /
durationEncoder / predictor.lstm / shared) run on the **CPU** — negligible
at L ≤ ~70. Everything matrix-heavy (PLBERT, convs, F0/N AdaIN curves, the
whole decoder + generator) stays GPU-resident; full GPU chain output is
then identical to the verified CPU path (energy 75.23).

**To re-enable.** Find an isolation repro (replay the exact failing
`predictor.shared` input+weights captured from a full-chain run; or stress
hundreds of back-to-back `Ops.lstm` dispatches interleaved with gemm/sdpa
to churn the buffer pool), then bisect lstm.rs (threadgroup `h`/`c` init +
barriers — they look correct on inspection) or the FFAI Tensor buffer
pool. Re-point `biLstmGPU` back to `Ops.lstm` once fixed.

## GPU pin — broad vision SDPA pin ELIMINATED; residual d32 + watch item

**Status (updated 2026-06-05).** The broad GPU pin is **gone.** The
2026-05-30 iron production-shape work added `(nQuery=576, nQHeads=16)`
GPU-correctness cells for `iron_sdpa_bidirectional` at d{64,72,80,96} and
fixed the dispatch-geometry bug, so those dims are now **GPU
production-safe** (`OpsValidation.sdpaBidirectionalProductionSafeHeadDims
= {64,72,80,96}`; `sdpaBidirectionalIsProductionSafe(_:)` gates the choice).
The vision towers that used to pin (GlmOcr, MiniCPM-V, Qwen2/2.5/3-VL,
Gemma4, SmolVLM2 — all d64–d96) run their bidirectional attention on the
GPU. (`defaultMaxCommandBufferCount` is 64 since PR #14 / J.7.)

**Residual — the only CPU-guarded bidirectional path:**
- **d32 — FastViT-HD (FastVLM).** Has a kernel (`sdpaBidirectionalSupportedHeadDims`
  includes 32) but is **not production-verified**
  (`sdpaBidirectionalIsProductionSafe(32) == false`), so `VisionEncoderLayer`
  falls back to the CPU `cpuAttention` path for d32. To close: add an
  `sdpa_bidirectional_d32_*` production-shape correctness cell in
  wh-iron-std, then add 32 to `sdpaBidirectionalProductionSafeHeadDims`.
- **d128** (Pixtral / Mistral3 / GlmOcr full-attention) routes through
  `Ops.sdpaMulti(causal:false)`, not the bidirectional kernel — unaffected.

**Watch item (vision + audio).** The validate-or-fall-back guard
(`sdpaBidirectionalIsProductionSafe`) means a **new, never-benched shape**
(a tower with an unlisted head_dim, or a much larger `nQuery`) cleanly
falls to CPU rather than pinning — but that's exactly where a future
regression would hide, so keep watching new vision/audio attention shapes.
The old "every dim CPU-guarded, only d72 GPU" state no longer applies.

## CPU cost remaining on audio encoders + FastVLM (vision now mostly GPU)

**Status (updated 2026-06-05).** The vision-tower bidirectional attention
moved back to the GPU at d64–d96 (see the GPU-pin entry), so VLM tests no
longer saturate CPU on attention. The genuine CPU perf cost that remains:

- **Audio encoders (Whisper, Qwen-Omni)** — `AudioEncoder.swift::cpuAttention`
  runs ~24 blocks of bidirectional attention over ~1500 frames on CPU
  (~70 GFLOP/utterance); `WhisperIntegrationTests` /
  `QwenOmniIntegrationTests` can hit the test timeout. Whisper's decoder
  also re-pays cross-attention CPU cost per token (no cross-side KV cache).
  These towers are mostly d≥64, so the production-safe `sdpaBidirectional`
  kernels now exist — the migration is wiring, not a missing kernel.
- **FastVLM** — d32 FastViT-HD attention is CPU-guarded (see GPU-pin
  residual) and its depthwise convs run CPU (no `Ops.conv2dDepthwise`).

**Next step.** Migrate the audio encoder to `Ops.sdpaBidirectional` and the
decoder self-attn to `Ops.sdpaMulti(causal:true)` + a precomputed-K/V
cross-attention dispatch; add the d32 production cell + a depthwise-conv2d
kernel for FastVLM.

## SDPA head-dim + AURA kernel coverage

What's wired vs outstanding (verified 2026-05-28).

**Text decode — `Ops.sdpaDecode`** (`OpsValidation.supportedSdpaHeadDims`):

| head_dim | Status | Notes |
|---|---|---|
| 64, 96, 128, 256, 512 | ✅ wired | d96 added this session (Phi-3); d128 generic; d256 two-phase; d512 dedicated |

**Sliding-window decode** (`slidingWindowSdpaHeadDims`):

| head_dim | Status |
|---|---|
| 128 | ✅ |
| 64, 256, … | ❌ sliding-window only supported at d128 today |

**Vision bidirectional — `Ops.sdpaBidirectional`** (`sdpaBidirectionalSupportedHeadDims = {32,64,72,80,96}`):

| head_dim | Kernel | GPU production-safe? |
|---|---|---|
| 64, 72, 80, 96 | ✅ | ✅ verified at production shape (576×16) — `sdpaBidirectionalProductionSafeHeadDims` |
| 32 | ✅ | ❌ has a kernel but not production-verified → CPU-guarded (FastViT-HD / FastVLM) |
| 128 | n/a | routed through `Ops.sdpaMulti(causal:false)` (Pixtral, Mistral3, GlmOcr) |

**AURA KV (Π-rotated quantized cache):**

- Family wiring: ✅ **universal** — every attention family rotates Q by Π
  / un-rotates the output (generalised this session).
- Schemes: ✅ `AURACodebook.supportedBits = {2,3,4,8}`; any `aura{kb}v{vb}`
  combo (aura4v4 default, aura4v2, aura8v8, …); encode kernels exist for
  int2/3/4/8.
- head_dim: ⚠️ **power-of-two only** (factory guard) — works at
  64/128/256, excludes 96 (Phi-3) / 80.
- Performance: ❌ decode still uses the dequant B-path; compressed-domain
  `aura_flash` fast decode is Phase H (H.1/H.2 open in session-plan).

## Gemma4-31B-4bit degenerate output — likely sampling config, unverified

`mlx-community/gemma-4-31b-it-4bit` (`Gemma4Dense (31B)`) greedy-decodes at
~10 % diversity while E2B / E4B / 26B-A4B pass. Likely the **same class** as
the (now-resolved) GPT-OSS-20B case — model default sampling on a RAW prompt
(flat low-probability tail at temp 1.0 → word-salad), not a kernel/forward
bug. Re-test with constrained sampling (`min_p 0.05`) or the chat template,
and add a forward-parity guard vs mlx-lm (like `GPTOSSTextIntegrationTests`
now has) before assuming a kernel bug.

## Qwen3-1.7B-3bit degenerate output — model quality floor, NOT a kernel bug

`Quantized3bitIntegrationTests` fails coherence (19 unique tokens / 200,
real-but-repetitive). **Confirmed not a kernel bug:** the only int3
kernels on the path (`dequant_gather_int3`, `dequant_gemv_int3`) pass new
production-shape GPU-correctness cells (iron `5cca5ac`), and the
emitted MSL is behaviorally identical before/after the `1522fbd` codegen
change. Pure 3-bit on a 1.7B model just sits at its greedy quality floor
— same class as Qwen3.5-0.8B (which relaxed its floor to 0.05) and the
2-bit test (no coherence assert).

**Plan (to rule out false positives).** Migrate the test off the tiny
1.7B-3bit checkpoint — convert + use **Qwen3.5-2B** quantized so the
suite asserts real coherence on a model that can actually hold up at low
bit-width. (Real-usage note: even Qwen3.5-0.8B-4bit is marginal, so
very-small + very-low-bit is expected to degrade.) Alternative: relax
the 3-bit floor to the 0.8B precedent. Left failing-on-purpose until
decided.

## Open integration-test triage (carried, unverified this session)

Brief — full per-item diagnoses are in `plan.md` / conversation history:

- **DeepSeekR1Distill-Qwen-1.5B** — greedy loop at temp=0; likely a
  Qwen2-base model property, not a kernel bug (needs mlx-lm baseline
  compare). R1-Distill-Llama-8B passes.
- **FastVLM** — 4-bit vision tower unsupported in `loadPW2D`; test points
  at the bf16 checkpoint.
- **FireRedASR2** — non-2D weight where a `Linear` is expected (depthwise
  axis pick at load).
- **GLMASR** — audio adapter feeds quantized weights into `Ops.gemm`
  without dequant.
- **FishSpeech** — codec Conv1d (dilated/transposed) still CPU until
  iron kernels land.

## Quantization — mixed-precision + format gaps

2-bit affine + per-tensor `QuantSpec` shipped (Phase E): `bits ∈
{2,3,4,5,6,8}` load + `ffai convert --bits/--embedding-bits/--lm-head-bits
/--vision-bits`; the loader derives each tensor's width from its packed
shape. Remaining gaps:

- **Config-advertised mixed-precision** (`mixed_3_6`, Unsloth UD recipes)
  — shape-derived loading works; needs verification on real checkpoints.
  `Qwen36UnslothMixedTextIntegrationTests` exists, gated by
  `IntegrationGroupGating.enableMixedPrecisionSuites` (default off) — flip
  + bisect.
- **Per-LAYER bit budgets** — `ffai convert` can't yet drive a
  layer-by-layer recipe (per-role only); loader-side per-tensor dispatch
  already supports arbitrary widths.
- **String-typed `bits`** — `dolphin-2.9.3-qwen2-*-2bit` ship `"bits":"2"`
  (string); `ModelConfig` parser expects Int → silently drops the quant
  path.
- **GGUF** — no parser (Phase 10).

## iron `test_mel_spectrogram` / `test_mel_spectrogram_magnitude` [f16] — flaky GPU correctness cell

**Symptom.** The iron `kernel_tests_harness` umbrella intermittently
fails one of the two f16 Mel `#[test_kernel]`s (which one, and the delta,
vary run-to-run — e.g. `max|Δ|≈16` then `≈8`), so `make test` is
flaky-red. Reproduces on **clean `dev`** with none of our kernels present —
**pre-existing, not ours**, and macOS-gated so it does not run on
iron's Linux CI.

**Why.** The Mel kernels accumulate the DFT in f32, but the **f16 inputs**
of a near-tonal test sine leave most bins at the log-floor, where `log`
amplifies tiny magnitude differences; the flakiness points at a too-tight
f16 tolerance and/or an under-padded test audio buffer (the frame walk reads
past the buffer on the last frames → nondeterministic). It does **not**
affect FFAI: the Gemma4 audio front-end uses the **f32** Mel path (validated
to 4.4e-4 vs the feature-extractor oracle + a coherent audio→text decode).

**Next step (upstream iron):** loosen the f16 Mel tolerance to the
log-floor regime, switch the test signal to broadband noise, or pad the test
audio to `(n_frames-1)*hop + n_fft`.

## Qwen3-VL image→text degenerate — non-square native-resolution grid

Qwen3-VL loads + runs and **video→text is coherent** (validates the GPU
`Ops.patchUnfoldQwen` multi-frame path), but **image→text still
degenerates**: `Qwen3 runForward`/`unfoldPatches` + the position-embedding /
M-RoPE / merger assume one square `gridSide`, while the native-resolution
image processor produces non-square grids. The GPU `patch_unfold_qwen`
kernel already supports non-square; the surrounding geometry does not.
Tracked under the smart_resize native-resolution work (`updated-architecture-overview.md`
§13 M1; the in-progress Qwen×4 processor task). (The earlier
`Ops.gather: unsupported dtype u32` load crash on this family is RESOLVED —
commit `14f127d` injects the top-level `quantization` into the re-wrapped
text config.)

## Orphaned GPU kernels still awaiting a consumer

`Ops.frameDiffLuma` (video luma frame-diff motion gate) has a iron
kernel + FFAI Ops wrapper + unit test but **no model call site**: it is
infrastructure for the planned real-time video loop (iron
`multi-modal-support.md` §7c / K7) — a cheap motion gate that skips
near-static frames before the vision tower. No model currently decodes
multi-frame video through such a gate, so wiring it is net-new feature work
(threshold + which frames to skip), not a mechanical swap. Left until the
real-time video pipeline lands. `Ops.patchEmbed` (fused unfold+embed) is
functionally superseded for Gemma4 by `Ops.im2colPatchInterleaved` + the
existing `Ops.gemm`.

## Tokenizers — InternLM2Tokenizer not registered

Loading InternLM2 logs `InternLM2Tokenizer is not registered, falling
back to BPE`. Low priority — the BPE fallback decodes coherently; only
special-token / chat-template edges are at risk. Fix: register
`InternLM2Tokenizer` against the SentencePiece/BPE handler where
`LlamaTokenizer` / `Qwen2Tokenizer` etc. are mapped. Surfaced 2026-05-28
with the InternLM2 loader work.

## Downloader — Xet-backed HF repos 404 (`ffai download` / `convert` / `Model.load`)

Surfaced 2026-05-29 adding `LiquidAI/LFM2.5-8B-A1B`. Its `model.safetensors`
is **Xet-backed** (resolve redirects to `cas-bridge.xethub.hf.co`,
`x-xet-hash` header); FFAI's runtime downloader (the `swift-huggingface`
/ `swift-transformers` `HubClient.downloadSnapshot` path in
`ModelDownloader.swift`) does not follow the Xet CAS redirect and fails
with `Status 404: Entry not found`. HF has been migrating new repos to
Xet by default, so this blocks `ffai download`, `ffai convert <repo>`
(its `ModelLocator.resolve`), **and** `Model.load` on any Xet repo —
including the `ekryski/*` conversions we upload (new uploads are
Xet-backed too). **Workaround in use:** download via the `hf` CLI (native
`hf_xet` support) into the HF cache, then point `ffai convert` at the
local snapshot dir and pre-cache uploaded repos with `hf download` before
the integration test loads them (`Model.load` hits cache, skips the
network path). **Real fix (open):** `Package.swift` already bumps
`swift-transformers` 1.3.0 → 1.3.3 (uncommitted, likely a prior Xet-fix
attempt — resolve was still 1.3.2); confirm whether 1.3.3+ adds Xet
support (or wire `hf_xet` / the CAS redirect into the download path),
then drop the pre-cache workaround.
