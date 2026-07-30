# Known Issues — to investigate after the model-implementation wave lands

Open bugs and rough notes captured during the model-family port. Each entry has enough context that a future session can pick it up cold.

> **Revalidation note (2026-07-29).** The entries below were captured during the 2026-05 model-family port and predate the `ek/fix-all-the-things-dev-merge` reconciliation (Butter rename + Laguna engine). Two have since been resolved and moved to [§ Resolved](#resolved-2026-07-29): the **GPU-pin-at-100%** investigation and the **quantization 2-bit + mixed-precision gap**. The remaining entries (CPU pin, audio CPU-attention, integration-bisect failures) still need re-validation against the current branch before they're actioned — kernel names, line numbers, and symptoms may have shifted with the reconciliation.

## CPU pin — vision / audio models burn all CPU cores

**Symptom.** During some integration tests (observed via `mactop`), all CPU cores spike to 100 % while the GPU stays well below saturation. GPU never sitting at 100 % during these runs is the giveaway — work that should be on the GPU is silently running on the CPU instead.

**Status.** Cause unknown. The vision (Phase 6.5) and audio (Phase 7) model families landed without a CPU-fallback audit, so the suspect surface is anything those pipelines touch where:

- A model op silently falls through to the parallel-CPU `concurrentPerform` core because no GPU wrapper was wired up (e.g. unknown SDPA `head_dim`, missing `Ops.conv2dDepthwise`, a custom audio op).
- A test or harness allocates / keeps tensors on CPU instead of dispatching to GPU.
- A test fixture / oracle is running its CPU reference path (parallelised across cores) and we're misreading that as the model itself pinning the CPU.

**Live signal.** `mactop` shows full CPU saturation across all cores during some currently-running tests; GPU has not been seen pegged during the same window. Indicates CPU is doing real work the GPU should be doing — or that a test-side CPU oracle dwarfs the GPU dispatch.

**Next steps when this lands as a priority.** Deferred until the current integration-test sweep finishes — that pass will surface more failures and give a fuller picture of which model families need attention. Then:

1. Re-run vision / audio model tests under `mactop` (or `powermetrics --samplers cpu_power,gpu_power`) to confirm the CPU pin reproduces and pin down which families.
2. For each pinning family, instrument the forward pass to log every op dispatch — find the calls that hit `concurrentPerform` / `cpuFallback` paths instead of `Ops.*` GPU dispatch.
3. Cross-reference against the FastVLM CPU depthwise-conv outlier in [§ Vision tower SDPA](#vision-tower-sdpa--head_dim-coverage) — likely the same class of gap, just not yet enumerated for the other vision / audio families.
4. File missing `Ops.*` wrappers as separate iron / Butter follow-ups (kernel exists in iron but no Swift wrapper, or kernel itself missing).

## Vision tower SDPA — head_dim coverage

VLM vision towers run bidirectional multi-head attention; Butter now ships GPU kernels for every head_dim in the cached zoo:

| head_dim | Kernel | Models |
|---|---|---|
| 32  | `Ops.sdpaBidirectional` (d32) | FastVLM (FastViT-HD) |
| 64  | `Ops.sdpaBidirectional` (d64) | SigLIP-base, CLIP-L, Mistral3, Gemma 4 E2B/E4B, Qwen3-VL 2B/4B |
| 72  | `Ops.sdpaBidirectional` (d72, ragged) | SigLIP-So400m → Paligemma, Gemma 3 VL, Gemma 4 26B/31B, Idefics3, Qwen3-VL-30B-A3B |
| 80  | `Ops.sdpaBidirectional` (d80, ragged) | Qwen2.5-VL |
| 96  | `Ops.sdpaBidirectional` (d96) | Qwen2-VL |
| 128 | `Ops.sdpaMulti(causal: false)` | Pixtral, Mistral3 Pixtral-base, GlmOcr |

`Sources/ButterSwift/VisionEncoder.swift::forward` dispatches based on the tower's `headDim`. Unknown head_dims (e.g. d80 at SigLIP-So-400m variants not yet seen) fall back to the parallel CPU `concurrentPerform` core. No tracked head_dim gaps as of 2026-05-24.

The remaining known cold-inference outlier is FastVLM at 1024px: its FastViT-HD stem has 256×256×96 + 128×128×192 depthwise convs that still run CPU because no `Ops.conv2dDepthwise(...)` wrapper exists. Tracked as a separate iron follow-up.

## Integration bisect — 2026-05-23/24 first-run findings

Per-suite serialised bisect run via `make integration-bisect` (commit `6047340`+) surfacing failures across all 74 integration tests. Fixed during the run:

- **ChatterboxIntegrationTests** — was failing on `unsupported dtype I64 for tensor "s3gen.speaker_encoder.xvector.block1.tdnnd6. nonlinear2.batchnorm.num_batches_tracked"`. Fixed in commit `af74ce7` by adding `.i64` / `.u64` to `DType` + making `SafeTensorsBundle` skip tensors with unsupported dtypes (Debug log only). Now PASSes in 21s.

- **CohereTranscribeIntegrationTests** — HF 404. Original `mlx-community/c4ai-aya-expanse-transcribe-mlx` archived; Cohere re-released as `cohere-transcribe-03-2026`. Fixed in commit `6ad9e73` pointing test at `mlx-community/cohere-transcribe-03-2026-mlx-8bit`.

Triage queue from same run (root-cause TBD — most look pre-existing; **all need re-validation against the current branch**):

- **DeepSeekR1DistillIntegrationTests** — R1-Distill-Llama-8B PASSes; R1-Distill-Qwen-1.5B produces degenerate output (token 15 repeated). Manually verified `iron_add_rms_norm` at hidden=1536 with `maxResidErr=0`, `maxNormedErr=1.4e-6` against CPU reference — fusion is NOT the regressor. Likely a pre-existing Qwen2 (R1-Distill base architecture) issue.

- **FastVLMIntegrationTests** — `Swift/ContiguousArrayBuffer.swift: 692: Fatal error: Index out of range` during the `load` test. Crash happens before the GPU-SDPA path runs, so the VLM agent's migration is not the cause. Suspect the SafeTensors-skip change or a pre-existing array index in the FastViT-HD loader.

- **FireRedASR2IntegrationTests** — `Butter/Layers.swift:19: Precondition failed: Linear: weight must be 2D`. Some weight tensor is non-2D where a `Linear` is expected — checkpoint shape vs loader-side reshape mismatch.

- **FishSpeechIntegrationTests** — `safetensors file not found: models--mlx-community--fish-audio-s2-pro-8bit/snapshots/.../ model.safetensors`. Snapshot directory exists, file is missing — incomplete cache. Needs re-download or upstream check.

- **GLMASRIntegrationTests** — `Ops.swift:1072 Precondition failed: dequantGemv: input 65280 ≠ in_dim 1280`. Caller is feeding the wrong-rank tensor (looks like 51 rows of in_dim=1280 flat). Pre-existing shape bug in the GLMASR encoder/decoder.

Bisect continues; this list will be appended-to as more suites complete.

## Audio model CPU bottlenecks — Whisper + QwenOmni timeout

**Symptom.** `WhisperIntegrationTests` and `QwenOmniIntegrationTests` both hit the 900 s `gtimeout` cap during the integration bisect; the log captures only the build banner — the test process was SIGKILL'd before any test boundary was reported.

**Root cause — shared audio-encoder CPU attention.** Both families route through `Sources/ButterSwift/AudioEncoder.swift::cpuAttention` (line ~165). The Whisper-style encoder runs N encoder blocks (24 for Whisper-large-v3 / Qwen2.5-Omni) of multi-head bidirectional attention over `nAudioCtx = 1500` frames; each block performs `O(nFrames² · nHeads · headDim)` work on the CPU via `DispatchQueue.concurrentPerform`. With 1500² · 20 · 64 ≈ 2.9 G FLOPs per layer × 24 layers ≈ 70 G FLOPs of pure CPU work per `encodeAudio` call. That dominates the wall clock and pegs all cores (matches the broader CPU-pin symptom in [§ CPU pin](#cpu-pin--vision--audio-models-burn-all-cpu-cores)). The Q/K/V projections + LayerNorm + GELU + final GEMM are already on GPU; only the SDPA-equivalent attention core falls back to CPU.

**Whisper additional site — decoder.** `Sources/ButterSwift/Models/Whisper.swift::cpuAttention` (line ~424) runs the SAME CPU multi-head attention for the **decoder's self- AND cross-attention**, once per generated token. `generateTranscript` then loops `decoderLogits` for up to 224 tokens. The cross-attention K/V is recomputed against the full 1500-frame audio every step (no KV cache on the cross side), so each generated token re-pays a `nQuery · nKV · headDim` matmul that should be a single GPU SDPA dispatch. Whisper's transcribe test pays the encoder cost once + the decoder cost ≈ 224 times.

**QwenOmni site.** `QwenOmniModel.encodeAudio` (Sources/ButterSwift/Models/QwenOmni.swift:163) calls into the same shared `AudioEncoder`, so it inherits the encoder-side bottleneck. QwenOmni has no decoder cpuAttention of its own (it splices feature tokens into the Qwen3 text backbone, which is already on GPU). So the expected wall-clock improvement after migration is proportional to the time spent in the audio tower alone — but for a multi-second input that's still tens of seconds of pure CPU at the encoder.

**Migration needed.** Replace both `cpuAttention` cores (`AudioEncoder.swift::cpuAttention` and `Whisper.swift::cpuAttention`) with `Ops.sdpaBidirectional` (encoder) and `Ops.sdpaMulti(causal: true)` (decoder self-attn, already cached) + a cross-attention GPU dispatch. Whisper's head_dim is 64 (small) / 80 (medium) / 80 (large-v3) — all already covered by [§ Vision tower SDPA](#vision-tower-sdpa--head_dim-coverage). The cross-attention path also needs a precomputed-K/V variant: encoder K/V should be projected once per utterance, then reused for every decoder step (eliminates the per-token K/V recompute). Mirrors the VLM splice pattern.

**Status.** Diagnosed only; not in scope for the current bisect pass. Tracked here so the audio-tower migration can pick this up as the unified "audio CPU-attention → GPU SDPA" port.

## AURA spec-decode rollback — stale atomic_or bits after truncate + re-append

**Symptom.** `AURAQuantizedKVCacheTests.truncateReappendMatchesClean` fails: after a spec-decode `truncate` (rolling back a rejected speculative draft) and re-appending the accepted continuation, the re-decoded AURA slots come back as garbage (`maxErr ≈ 2e38` vs the `< 1e-3` a clean rollback yields). The other AURA cache tests (append / decode round-trips) pass, so the codebook + encode/decode path is sound in general — the bug is specific to the re-write-after-truncate case.

**Status.** Surfaced 2026-07-29, the first time the unit-test target compiled on `ek/fix-all-the-things-dev-merge` (it was blocked by the reconciliation fossils until then, so this test had never run here). The Swift-side zero-on-rewrite guard in `AURAQuantizedKVCache.appendOnGPU` (zero `packed[pos]` when `pos < highWaterSlot`) *looks* correct, so the leak is likely in the `zeroPackedSlot` GPU dispatch or the regenerated AURA encode kernel's `atomic_or` semantics. Not yet determined whether it is purely pre-existing or an interaction with the iron-dev kernel regen.

**Impact.** Spec-decode (draft-then-verify) with AURA-compressed KV caches only. Non-spec-decode AURA generation is unaffected (those cache tests pass). AURA is opt-in.

**Next steps** (defer into the AURA / `AttentionCacheFactory` work):
1. Confirm `zeroPackedSlot` actually clears the slot on the GPU — audit its dispatch shape / kernel.
2. Check whether the current iron AURA encode kernel still `atomic_or`-accumulates (vs overwrite); if the semantics changed in the regen, the Swift zero-on-rewrite may be redundant or mistimed.
3. Compare against `origin/dev`'s behavior to localize pre-existing vs regen-introduced.

## Resolved (2026-07-29)

Closed out during the `ek/fix-all-the-things-dev-merge` reconciliation review. Kept as a short record so the investigation isn't re-opened from scratch.

- **GPU pin at 100% (long-running).** The 2026-05 investigation (a freshly-regenerated MPP/coop-tile kernel with a pathological dispatch shape at production sizes) is no longer relevant on the current branch and has been dropped. `BUTTER_MAX_COMMAND_BUFFERS=16` remains the production default. If a GPU pin resurfaces, re-open with a fresh Metal System Trace capture rather than the stale hypothesis stack.

- **Quantization — 2-bit + mixed-precision.** Both shipped. 2-bit is live end-to-end: `iron_dequant_gemv_int2_{f32,f16,bf16}`, `iron_dequant_gather_int2_*`, and `iron_qmm_mma_int2_*` kernels are emitted by iron and dispatched from `Sources/ButterSwift/Ops/Ops.swift`; the roadmap lists 2/3/4/5/6/8-bit as shipped. Mixed-precision (per-tensor-class bit-width) is implemented via `Sources/ButterSwift/Converter/QuantSpec.swift` + `ConvertDriver.swift` (per-role `--bits` / `--embedding-bits` / `--lm-head-bits` / `--vision-bits` overrides) and per-tensor bit-width derivation on load. Test coverage tracked with the quant integration matrix.
