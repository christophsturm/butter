# Known Issues

Open bugs + rough notes from the model-family port. Each entry carries
enough context to pick up cold. Resolved-work history lives in
`planning/session-plan.md` / `plan.md`, not here.

## GPU LSTM (`ffai_lstm`) — context-dependent NaN in the full Kokoro chain

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
work) — a context/state interaction, possibly metaltile buffer-pool
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

## GPU pin — vision / audio bidirectional SDPA at production shape

**Symptom.** Some VLM / audio runs peg the GPU at 100 % and never
return (worst case seen: WindowServer freeze → hard reboot). Text decode
is unaffected — `ffai inspect` confirms the same checkpoints load +
text-decode cleanly, so the pin is isolated to the **vision-tower /
audio-encoder forward**.

**Leading hypothesis.** `mt_sdpa_bidirectional_d64_*` (and likely the
other bidirectional dims) has an inner-loop / bounds pathology at
production geometry — first hit on GlmOcr + MiniCPM-V vision towers at
`(nQHeads=16, nQuery=576, kvStride=576)`. No metaltile GPU-correctness
test has ever exercised these production shapes. Same class as the
wrong-dispatch freeze in
`papers/post-mortem-2026-05-19-dispatch-shape-gpu-freeze.md`.

**Mitigation in place.** Every `Ops.sdpaBidirectional` consumer at
d ∈ {32,64,80,96} routes through a pure-CPU `concurrentPerform` softmax
loop; `VisionEncoder.forward` takes the GPU path only at the one
confirmed-safe dim (d72). Qwen2.5-VL d80 GPU path is behind
`FFAI_QWEN25VL_GPU_SDPA=1`. Slow but no pin. (`defaultMaxCommandBufferCount`
is 64 since PR #14 / J.7; the old `FFAI_MAX_COMMAND_BUFFERS=16` is still
an override.)

**Next step.** Reproduce under Metal System Trace / `MTLCaptureManager`
to read the hung dispatch, OR add per-dim
`sdpa_bidirectional_d{N}_gpu_correctness.rs` in metaltile-std at the real
`(nQHeads,nQuery,kvStride)` shapes vs a CPU oracle, fix the kernel
arithmetic, then drop the CPU guards (`VisionEncoder.swift` + per-family
`*Vision.swift`). Use `make integration-bisect` + `ffai inspect` to
localise which family/shape pins.

## CPU saturation on vision / audio towers — mostly the intentional guard now

**Status.** Largely EXPECTED today: vision/audio bidirectional attention
runs on CPU **on purpose** (the GPU-pin guard above). So "all cores at
100 %, GPU idle" during VLM/audio tests is the guard working, not a new
bug. The genuine perf cost that remains:

- **Audio encoders (Whisper, Qwen-Omni)** — `AudioEncoder.swift::cpuAttention`
  runs ~24 blocks of bidirectional attention over 1500 frames on CPU
  (~70 GFLOP/utterance); `WhisperIntegrationTests` /
  `QwenOmniIntegrationTests` can hit the test timeout. Whisper's decoder
  also re-pays cross-attention CPU cost per token (no cross-side KV cache).
- **FastVLM 1024px** — FastViT-HD depthwise convs run CPU (no
  `Ops.conv2dDepthwise`).

**Next step.** Same fix as the GPU-pin entry: once `sdpaBidirectional`
is GPU-safe at production shape, vision towers move back to GPU. Audio
encoder/decoder migrate to `Ops.sdpaBidirectional` (encoder) +
`Ops.sdpaMulti(causal:true)` (decoder self-attn) + a precomputed-K/V
cross-attention dispatch.

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

| head_dim | Kernel | GPU-safe? |
|---|---|---|
| 72 | ✅ | ✅ only confirmed-safe dim |
| 32, 64, 80, 96 | ✅ | ❌ pins GPU at production shape → CPU-guarded (see GPU-pin entry) |
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

## Qwen3.6-27B-4bit degenerate output — fix landed, pending test confirmation

**Was:** greedy decode emitted `"!"` every step at 27B (hidden=5120)
while Qwen3.5-0.8B on the same engine was coherent. **Fix:**
verified-shape gates now exclude the 27B shape from both fast paths —
`Ops.batchedQkvQmmFast` (`Qwen3xText.swift:2032-2049`) and
`Ops.rmsNormQgemvInt4Fast` (`:2759-2773`) — routing to the safe path.
Env overrides `FFAI_NO_FUSED_QKV` / `FFAI_NO_FUSED_LM_HEAD` retained.

**Confirmed 2026-05-29.** `Qwen36TextIntegrationTests` PASSED (123 s) in
the full bisect — the fix holds. **RESOLVED** — stale
`KNOWN FAILURE (2026-05-27)` comment deleted from `Qwen36TextIntegrationTests.swift`.

## LFM2-MoE — quantized router loaded as plain Linear — RESOLVED 2026-05-29

**Was:** `LFM2TextIntegrationTests` LFM2-MoE case crashed —
`Ops.swift:406: Precondition failed: gemv: in_dim mismatch 512 vs 2048`.
**Root cause (the original "router-shape mismatch" framing was wrong):**
`mlx-community/LFM2-8B-A1B-4bit` ships the router `feed_forward.gate` as
an **8-bit quantized triplet** (`gate.weight` packed U32 `[32, 512]` +
`gate.scales`/`gate.biases` `[32, 32]`; logical in=2048 → out=32,
group_size 64). `buildLFM2MoE` loaded it as a plain `Linear` on the
packed U32 tensor, so the gemv read in_dim from the packed width (512)
instead of the logical hidden (2048). It is NOT a bad conversion — the
gate is deliberately quantized at higher precision than the 4-bit body.
**Fix:** route the gate through `loadLinear` so the quantized triplet is
detected and the bit-width derived (commit on `ek/fix-all-the-things`);
raw checkpoints still fall back to a plain `Linear`. Guarded by
`ConvertDriverTests.quantizedRouterLoadsViaLoadLinear` (round-trip) +
the LFM2-MoE integration test. Stale `unsupportedConfig` comment at
`LFM2Text.swift:198-200` corrected. Separately, `ffai convert` now keeps
MoE routers full-precision by default (`--router-bits` opts in).

## GPT-OSS-20B-4bit + Gemma4-31B-4bit degenerate output — newly surfaced (2026-05-29)

Surfaced once the `FFAI_BUILD_MACHINE` gate was removed and these heavy
models ran in the bisect for the first time:
- **GPT-OSS-20B-4bit** (`loan-star/gpt-oss-20b-mlx-4Bit`) —
  `GPTOSSTextIntegrationTests` greedy-decodes 200 tokens at **12 %
  diversity** (24 unique): real tokens, then loops. Default raw KV cache +
  host-side sink correction.
- **Gemma4-31B-4bit** (`mlx-community/gemma-4-31b-it-4bit`) — the
  `Gemma4Dense (31B)` case at **10 % diversity**. Gemma4 **E2B / E4B /
  26B-A4B pass** — only the 31B dense degenerates.

Same class as the Qwen3.6-27B-4bit case above (large hidden / fused-path
kernel-shape sensitivity at 4-bit). **NOT a regression** from the
2026-05-29 cache / spec-decode / ungating work — the default raw-decode
paths these hit were untouched; the suites had simply never run before
(build-machine-gated). Investigation: separate a genuine 4-bit
quant-quality floor from a fast-path kernel-shape bug by A/B-ing the
fused-path env overrides (`FFAI_NO_FUSED_QKV` / `FFAI_NO_FUSED_LM_HEAD`
for GPT-OSS; the analogous Gemma4 fast paths), exactly as was done for
Qwen3.6-27B.

## Qwen3-1.7B-3bit degenerate output — model quality floor, NOT a kernel bug

`Quantized3bitIntegrationTests` fails coherence (19 unique tokens / 200,
real-but-repetitive). **Confirmed not a kernel bug:** the only int3
kernels on the path (`dequant_gather_int3`, `dequant_gemv_int3`) pass new
production-shape GPU-correctness cells (metaltile `5cca5ac`), and the
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
  metaltile kernels land.

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

## metaltile `moe_gather_down_q2k_correctness` — RESOLVED (test oracle, not the kernel)

**Was:** `make test` in the sibling **metaltile** repo failed
`moe_gather_down_q2k_correctness` (the fused MoE down-projection gather over
a Q2_K-quantized expert weight disagreed with its CPU reference at ~0.71
relative error), landing with #243.

**Root cause + fix (branch `ek/fix-q2k-gather-down-test`, commit
`c685e4ce`).** The **kernel was correct** — byte-identical Q2_K dequant math
to the validated `gguf_dequant_q2_k`. The **test oracle was wrong**: it
unpacked the 2-bit quants with the naive 4-consecutive-per-byte mapping
(`in_block/4` byte, `(in_block%4)*2` shift) while the canonical layout (the
kernel + `gguf_dequant_q2_k`) is the llama.cpp `dequantize_row_q2_K` order
(2 halves of 128, 4 j-groups of 32, two 16-runs at a shared `jg*2` shift).
Fix: gave the oracle a `q2_k_qpos` helper identical to `gguf_dequant_q2_k`.
Both `gather_down_q2k_{f32,f16}` now pass. (Pending PR back to metaltile
`dev`.)

## metaltile `test_mel_spectrogram` / `test_mel_spectrogram_magnitude` [f16] — flaky GPU correctness cell

**Symptom.** The metaltile `kernel_tests_harness` umbrella intermittently
fails one of the two f16 Mel `#[test_kernel]`s (which one, and the delta,
vary run-to-run — e.g. `max|Δ|≈16` then `≈8`), so `make test` is
flaky-red. Reproduces on **clean `dev`** with none of our kernels present —
**pre-existing, not ours**, and macOS-gated so it does not run on
metaltile's Linux CI.

**Why.** The Mel kernels accumulate the DFT in f32, but the **f16 inputs**
of a near-tonal test sine leave most bins at the log-floor, where `log`
amplifies tiny magnitude differences; the flakiness points at a too-tight
f16 tolerance and/or an under-padded test audio buffer (the frame walk reads
past the buffer on the last frames → nondeterministic). It does **not**
affect FFAI: the Gemma4 audio front-end uses the **f32** Mel path (validated
to 4.4e-4 vs the feature-extractor oracle + a coherent audio→text decode).

**Next step (upstream metaltile):** loosen the f16 Mel tolerance to the
log-floor regime, switch the test signal to broadband noise, or pad the test
audio to `(n_frames-1)*hop + n_fft`.

## Qwen3-VL vision — `Ops.gather: unsupported dtype u32` crash — RESOLVED

**Was:** the Qwen3-VL suite aborted at load (`Model.prewarm` → first forward)
with `FFAI/Ops.swift:371: Fatal error: Ops.gather: unsupported dtype u32`.
**Root cause:** `Qwen3Vision.load` re-wrapped the checkpoint's `text_config`
into a fresh `ModelConfig` but dropped the **top-level** `quantization` block
(MLX VL checkpoints declare it once at the top level, not inside
`text_config`). The Qwen3Dense backbone then loaded its embedding as a plain
`Embedding` over the packed-u32 weight (and would have mis-loaded every
quantized linear). **Fix (commit `14f127d`):** inject the top-level
`quantization` into the re-wrapped text config. Qwen2/2.5-VL thread
`quantization` through as a parameter and never had the bug.

**Result.** Qwen3-VL now loads + runs; **video→text is coherent** (the cat
clip is described correctly), which validates the GPU `Ops.patchUnfoldQwen`
wiring end-to-end (the multi-frame temporal path). Remaining: **image→text
still degenerates** — a separate, pre-existing square-grid issue (Qwen3
`runForward`/`unfoldPatches` assume one square `gridSide`, while the
native-resolution image processor produces non-square grids; the GPU
`patch_unfold_qwen` kernel already supports non-square, but the position
embedding / M-RoPE / merger still assume square). Tracked under the
smart_resize native-resolution work (the in-progress Qwen×4 processor task).

## Orphaned GPU kernels still awaiting a consumer

`Ops.frameDiffLuma` (video luma frame-diff motion gate) has a metaltile
kernel + FFAI Ops wrapper + unit test but **no model call site**: it is
infrastructure for the planned real-time video loop (metaltile
`multi-modal-support.md` §7c / K7) — a cheap motion gate that skips
near-static frames before the vision tower. No model currently decodes
multi-frame video through such a gate, so wiring it is net-new feature work
(threshold + which frames to skip), not a mechanical swap. Left until the
real-time video pipeline lands. `Ops.patchEmbed` (fused unfold+embed) is
functionally superseded for Gemma4 by `Ops.im2colPatchInterleaved` + the
existing `Ops.gemm`.

**Resolved:** `Ops.gatedRmsNormQgemvInt4Fast` is now wired (commit
`8cc1635`) into the **Qwen3.5/3.6 Gated-DeltaNet** decode tail — fusing the
gated mixer RMSNorm + int4 out_proj when out_proj is 4-bit (group 64) and
the GDN value dim fits. It was previously mis-attributed to Mamba2; Mamba2's
global `n_groups=1` norm is incompatible with this per-head gated RMSNorm.
Validated by `Qwen35TextIntegrationTests` (coherent decode).

## Tokenizers — InternLM2Tokenizer not registered

Loading InternLM2 logs `InternLM2Tokenizer is not registered, falling
back to BPE`. Low priority — the BPE fallback decodes coherently; only
special-token / chat-template edges are at risk. Fix: register
`InternLM2Tokenizer` against the SentencePiece/BPE handler where
`LlamaTokenizer` / `Qwen2Tokenizer` etc. are mapped. Surfaced 2026-05-28
with the InternLM2 loader work.

## LFM2.5 "forward bug" — RESOLVED: instruct model needs its chat template

Investigated 2026-05-30. **There is no forward bug.** `LiquidAI/LFM2.5-8B-A1B`
looked degenerate on a RAW prompt ("…when when when … que que"), but it is an
**instruct / reasoning model** that requires its chat template; LFM2-8B-A1B is
a base model, which is why raw prompts worked there. Fed a chat-formatted
prompt, LFM2.5 reasons and answers correctly ("…The answer: Paris…"). The
conversion (per-expert + stacked), loader, quantization, and forward pass are
all correct.

**Root cause of the bad first impression:** LFM2.5 ships its chat template as a
standalone `chat_template.jinja` file, NOT a `chat_template` field in
`tokenizer_config.json`. swift-transformers' `AutoTokenizer` only loads the
config field, so `Model.generate(messages:)` had no template and callers fell
back to raw prompts. **Fix:** `Model.standaloneChatTemplate` reads the
`.jinja` from the model directory when the tokenizer carries no built-in
template, and `renderChatTemplate` passes it as a `ChatTemplateArgument.literal`
override. `Model.supportsChatTemplate` reports availability.
`LFM2TextIntegrationTests` LFM2.5 case now generates via the chat API and
asserts coherence + the "Paris" answer. Also wired `InspectTap` into LFM2's
forward so `ffai inspect --layer-trace` works for the family (it was a silent
no-op before — used to confirm activations were healthy during this triage).

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
