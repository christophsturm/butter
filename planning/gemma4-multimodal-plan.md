# Gemma 4 — full multimodal (text + image + audio + video)

Status: **architecture mapped, feasibility confirmed, implementation phased below.**
Owner directive (2026-05-30): "We have a Gemma 4 reference model. Build it."

## Feasibility (confirmed 2026-05-30)

- **Reference checkpoint:** `google/gemma-4-E2B-it` — **ungated** (`gated: false`). Also `gemma-4-E4B`, `gemma-4-26B`, `gemma-4-31B`.
- **4-bit weights already carry every modality:** `mlx-community/gemma-4-e2b-4bit` (ungated, single index, 2649 tensors) ships the FULL `audio_tower` (754 keys) + `embed_vision` (661 keys) + `language_model` (1234 keys). **The user's worry that the MLX export "dropped audio/video" is wrong at the weight level — the tensors are present.** mlx-vlm may not *run* audio, but FFAI can load these weights directly. **→ no conversion needed to bring up + test E2B.** (Convert our own 4-bit for E4B / 26B / 31B only if/when we want them and an MLX export doesn't already exist.)
- The gap is **purely FFAI code**: FFAI's `Gemma4` is text + image today; it has no audio encoder and no audio/video token splice.

## Architecture (from `google/gemma-4-E2B-it` config + `audio_tower.*` tensor names)

`Gemma4ForConditionalGeneration` (`model_type: gemma4`), three encoders spliced into the text stream by placeholder tokens:

| token | id | meaning |
|---|---|---|
| `image_token_id` | 258880 | image soft-token placeholder (`vision_soft_tokens_per_image: 280`) |
| `audio_token_id` | 258881 | audio frame placeholder |
| `boa_token_id` / `eoa_token_id` | 256000 / 258883 | begin/end audio span |
| `boi_token_id` / `eoi_token_id` | — | begin/end image span |
| `video_token_id` | 258884 | video frame placeholder (frames → vision tower) |

**Text** (`gemma4_text`, FFAI HAS): 35 layers, MoE + Per-Layer-Embeddings + KV-sharing — already loads via `Gemma4E` / `Gemma4Dense` / `Gemma4MoE`.

**Vision** (`gemma4_vision`, FFAI HAS): SigLIP, hidden 768, 16 layers, attention-pool → 280 soft tokens — `Gemma4VLVisionModel`. **Video** = N frames each through this tower at `video_token_id` slots (mirror the Qwen2-VL video path).

**Audio** (`gemma4_audio`, FFAI MISSING — the build) — a **Conformer / USM encoder**:
- `audio_config`: hidden 1024, 12 layers, 8 heads, `output_proj_dims: 1536`, `rms_norm_eps: 1e-6`.
- **Front-end:** mel spectrogram (feature_size/mel-bins + window/hop from `processor_config.json` / `feature_extractor_config.json` — pull these) → `subsample_conv_projection`: `input_proj_linear` + 2 conv-subsampling layers (`subsampling_conv_channels: [128, 32]`, each `layerK.conv` + `layerK.norm`) → downsampled audio frames.
- **Per Conformer layer** (`audio_tower.layers.N.*`, 62 tensors each), macaron sandwich:
  1. `feed_forward1` — `pre_layer_norm` → `ffw_layer_1` (clipped linear) → SiLU → `ffw_layer_2` (clipped linear) → `post_layer_norm`, residual scaled by `residual_weight: 0.5`.
  2. **Local chunked self-attention** — `norm_pre_attn` → MHA → `norm_post_attn`. Chunked: `attention_chunk_size: 12`, `attention_context_left: 13`, `attention_context_right: 0` (causal local window), `attention_logit_cap: 50.0` (tanh-softcap), `attention_invalid_logits_value: -1e9`. (q/k/v/o proj tensors are in the per-layer set beyond the 40 sampled — enumerate them.)
  3. `lconv1d` (conv module) — `pre_layer_norm` → `linear_start` (clipped) → GLU? → `depthwise_conv1d` (`conv_kernel_size: 5`) → `conv_norm` → SiLU → `linear_end` (clipped).
  4. `feed_forward2` — same as ff1.
  5. `norm_out`.
- **`output_proj`** 1024 → 1536 (`output_proj_dims`) into the text hidden dim.
- **"Clipped linears"** (`use_clipped_linears: true`): every audio linear ships `input_min/input_max/output_min/output_max` tensors — activation clipping ranges (clip input to [in_min,in_max], matmul, clip output to [out_min,out_max]). A Gemma-4-audio-specific quantization-robustness scheme. Must be applied or audio output is wrong. Confirm exact semantics against the `transformers` Gemma4 audio modeling code.

## Kernels likely needed (metaltile)
- **Local/chunked SDPA** with left-context window + logit softcap (chunk 12 / left 13 / right 0). Possibly expressible via the existing sliding-window SDPA + softcap; confirm.
- **Depthwise conv1d** (kernel 5) for the conv module — check `audio_conv1d` / `conv1d_causal_step` coverage; this is non-causal within the chunk.
- **Conv subsampling** (strided conv2d/1d over mel) — may reuse existing conv kernels.
- **Clipped linear** — a clamp-matmul-clamp; can be CPU/host-composed first, kernel later.
Each new kernel ships a paired `*_gpu_correctness` test (metaltile) per the repo standard.

## Phased build (each phase independently committable + testable)

- **Phase 1 — audio config + front-end.** Parse `audio_config` + `processor`/`feature_extractor` (mel params). Implement the mel spectrogram extractor (reuse `Sources/FFAI/Audio/` STT mel front-ends — Whisper/Parakeet already do mel) + `subsample_conv_projection`. Unit test: config parse + mel shape on a known clip.
- **Phase 2 — Conformer layer.** One layer (macaron FF + local attn + conv module + clipped linears), host-composed where kernels are missing. Forward-shape unit test vs the loaded E2B layer-0 weights; numeric spot-check vs a tiny CPU oracle.
- **Phase 3 — full audio encoder + output_proj.** Stack 12 layers; `Gemma4AudioEncoder.encode(waveform) -> [audioFrames, 1536]`. Wire `InspectTap`.
- **Phase 4 — splice + capability.** Extend `Gemma4VL` to a 3-way cross-modal splice: replace `audio_token_id` spans (between boa/eoa) with audio frames, `video_token_id` with per-frame vision soft tokens. Declare `.audioIn` / `.videoIn` on `Gemma4VL.availableCapabilities`. (`Capability` already has `.audioIn`/`.videoIn`.)
- **Phase 5 — integration tests.** `Tests/ModelIntegrationTests/Vision/Gemma4MultimodalIntegrationTests.swift`: load `mlx-community/gemma-4-e2b-4bit`, run (a) image+text (already works), (b) **audio+text** (a short speech clip → coherent response, `expectCoherentOutput`), (c) **video+text**. Use the **model's default params** (not greedy) per the new standard.
- **Phase 6 — the other sizes.** E4B / 26B / 31B: confirm each loads; convert our own 4-bit (`ffai convert`, stacked experts for the MoE 26B) + upload to `ekryski/` only if no usable MLX export exists. One integration test each.

## Confirmed 2026-05-30 — fully de-risked, ready to build the encoder

- **Kernels: ALL needed ops already exist** (regenerated off metaltile `dev` @ `dcbe860`, 411 ok). **No new metaltile branch required for pass 1.** Reuse: `Ops.conv2d` (SSCP subsampling), `Ops.audioConv1d` (depthwise conv module), `Ops.gemm` (attention scores / projections), `Ops.softmax`, `Ops.layerNorm` (SSCP channel norm), `Ops.rmsNormRows`, `Ops.silu`/`sigmoid`/`softplus`/`add`/`mul`, plus the mel front-end kernels `mel_spectrogram` / `mel_filterbank` / `mel_stft_window` / FFT. `clip`/`tanh`/relative-shift do on CPU (encoder-scale).
- **Validation oracle ready:** `mlx-community/gemma-4-e2b-4bit` downloaded (3.4 GB, ungated) at `~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-4bit`.
- **Reference (port target):** `mlx_vlm/models/gemma4/audio.py` (509 lines, complete) at `~/Development/personal/ai/gemma4-benchmark/.venv/lib/python3.10/site-packages/mlx_vlm/models/gemma4/` — plus `vision.py::ClippableLinear`, `config.py::AudioConfig`, `processing_gemma4.py`, and `gemma4.py` (the `masked_scatter` splice). Mirror FFAI's `Gemma4VLVisionBlock` style (token-major rows, GPU GEMMs via `projectRows`, CPU attention).
- **Exact algorithm** (from the reference):
  - **ClippableLinear** = `clip(x, input_min, input_max)` → `linear` → `clip(y, output_min, output_max)`; 4 scalar bounds/tensor (±inf when unused).
  - **SSCP**: 2× [zero-invalid → pad(1,1,1,1) → Conv2d(3×3, stride 2) → LayerNorm(channels) → ReLU], then flatten `F·C` → `input_proj_linear` → hidden. 128 mel bins → freq 128→64→32, `proj_input_dim = 32·32 = 1024` → 1024.
  - **ConformerFeedForward** (macaron): `residual + 0.5 · post_norm(ff2(silu(ff1(pre_norm(clip(x))))))`.
  - **AudioAttention**: ClippableLinear q/k/v/post (head_dim 128 = 1024/8) + `relative_k_proj` + `per_dim_scale` (softplus); `q_scale = d^-0.5/ln2`, `k_scale = ln(1+e)/ln2`; chunk into blocks (chunk 12, context = 12+13+0=25); sinusoidal **relative position embedding + relative-shift**; `logits = tanh(logits/50)·50`; mask by local causal-valid; softmax; `einsum bnuwc,bucnh→buwnh`; post.
  - **ConformerLightConv1d**: `pre_norm` → `linear_start`(→2×) → GLU → causal-pad(k-1=4) → depthwise conv1d(k5) → clip → conv_norm → silu → linear_end → +residual.
  - **ConformerBlock**: ff1 → [clip→norm_pre_attn→attn→clip→ +norm_post_attn] → zero-invalid → lconv1d → ff2 → clip → norm_out.
  - **AudioEncoder**: SSCP → 12 blocks → `output_proj` (1024→1536, +bias). Then top-level `embed_audio.embedding_projection` (a **quantized** Linear) → text dim → `masked_scatter` at `audio_token_id` (258881).
- **Mel front-end params** (`feature_extraction_gemma4`): 128 mels, 16 kHz, frame 20 ms (320), hop 10 ms (160), HTK preemphasis flavor, dither 0, mel_floor 1e-3. Wire FFAI's existing mel kernels with these.
- **Tensor map** (per `audio_tower.layers.N.`): `feed_forward1/2.{pre_layer_norm,ffw_layer_1,ffw_layer_2,post_layer_norm}`, `self_attn.{q_proj,k_proj,v_proj,post}` (clipped: `.linear.weight` + `.input_min/max` + `.output_min/max`), `self_attn.relative_k_proj.weight`, `self_attn.per_dim_scale`, `lconv1d.{pre_layer_norm,linear_start,depthwise_conv1d,conv_norm,linear_end}`, `norm_{pre_attn,post_attn,out}`. Top: `audio_tower.subsample_conv_projection.{input_proj_linear,layer0/1.{conv,norm}}`, `audio_tower.output_proj.{weight,bias}`, `embed_audio.embedding_projection.{weight,scales,biases}`.

## Validation approach
Numerically diff FFAI's audio-tower output against `mlx_vlm` for the same mel input (dump mlx-vlm's `audio_tower(audio_features, mask)` to a fixture), iterating until within tolerance — THEN wire the splice + the coherence integration test. The chunked relative-shift attention is the piece most likely to need oracle-debugging; do it against the fixture, not blind.
