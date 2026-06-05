# Qwen3-Omni-30B-A3B-Thinking — Port Plan

Status: **scoped + quant kicked off**, implementation pending the 4-bit quant.
Branch: `ek/fix-all-the-things`. Tracked under task #66 / #65.

Model: `Qwen/Qwen3-Omni-30B-A3B-Thinking`, `model_type: qwen3_omni_moe`,
arch `Qwen3OmniMoeForConditionalGeneration`. Omni = text + image + video +
audio-IN → text (TTS half ignored, `enable_audio_output: false`). All weights
under `thinker.*`. 19,743 tensors, 59 GB bf16 → converting to ekryski 4-bit.

## Reuse map (verified)

| Component | Verdict | Where |
|---|---|---|
| Text (30B-A3B MoE) | REUSE | `Qwen35Hybrid.loadModel(config:weights:options:device:) -> Qwen35Model` (`Qwen3xText.swift:165`). Detects MoE via `text_config.num_experts>0`; shared-expert is optional (`shared_expert_intermediate_size ?? 0` → Omni has 0 → skipped). Stacked `switch_mlp.{gate,up,down}_proj` experts, router `mlp.gate`. |
| Vision (Qwen3-VL ViT) | REUSE (verify names) | `Qwen3VLVisionModel.load(visionConfig:textHidden:weights:dtype:device:)` (`Qwen3Vision.swift:421`) + `.asVisionEncoder()` (:607). Reference wiring: `Qwen35VL.load` (:904/:997). |
| Audio (conv2d encoder) | **NEW** | No existing match — see below. |
| VisionModel splice | REUSE | `VisionModel.init(visionEncoder:engine:imageTokenId:videoTokenId:normalization:imageTokenCount:audioTokenId:audioEncode:)`. Pattern: `MiniCPMO.swift:160`. |
| conv2d / sinusoidal pos | REUSE | `Ops.conv2d` (`Ops/Ops.swift`); `MediaUtils.sinusoidalPositions(length:dim:)` (`AudioPreprocessing.swift:188`). |

`QwenOmni.swift` (audio-only, conv1d Whisper stem + Qwen3**Dense**) does NOT
fit (needs conv2d stem + MoE + vision) — build a fresh `Qwen3Omni` family
entry mirroring `MiniCPMO`, do NOT retrofit QwenOmni.

## Config (thinker_config)

- text_config (`qwen3_omni_moe_text`): hidden 2048, 48L, **128 experts / 8 active**,
  moe_intermediate 768, heads 32, kv 4, head_dim 128, qk_norm, rope_theta 1e6,
  rms_eps 1e-6, vocab 152064, shared_expert_intermediate 0, no top-level `text_config`
  → **must construct a ModelConfig whose `text_config` = thinker_config.text_config**,
  and pass `weights.prefixed("thinker.")` so `model.layers.*` + `lm_head` resolve.
- vision_config (`qwen3_omni_moe_vision_encoder`): patch 16, depth 27, hidden 1152,
  heads 16, intermediate 4304, spatial_merge 2, temporal_patch 2, out_hidden 2048,
  image_size 768, gelu_pytorch_tanh, deepstack `merger_list`. Weights `thinker.visual.*`.
- audio_config (`qwen3_omni_moe_audio_encoder`): d_model 1280, 32L, 20 heads,
  ffn 5120, mel 128, output_dim 2048, downsample_hidden 480, n_window 50,
  max_source_positions 1500. Weights `thinker.audio_tower.*`.

Token ids: image 151655, video 151656, audio 151675, audio_start 151669,
audio_end 151670, vision_start 151652, vision_end 151653, im_start 151644,
im_end 151645.

## Audio encoder forward (reference: transformers `modeling_qwen3_omni_moe.py`)

Weight shapes (bf16 base):
- `conv2d1.weight [480,1,3,3]`, `conv2d2.weight [480,480,3,3]`, `conv2d3.weight [480,480,3,3]` (+biases)
- `conv_out.weight [1280,7680]`
- `layers.N.{self_attn.{q,k,v,out_proj}(+bias), self_attn_layer_norm, fc1[5120,1280], fc2, final_layer_norm}` ×32
- `ln_post [1280]`, `proj1.weight [1280,1280]`, `proj2.weight [2048,1280]` (+biases)

Forward:
1. mel `[mel=128, time]` → unsqueeze `[chunks,1,128,time]`.
2. `gelu(conv2d1)` → `gelu(conv2d2)` → `gelu(conv2d3)`, each **k3 s2 p1** (downsamples
   freq 128→~16 and time by 2 each). Out `[b,480,~16,t]`.
3. `permute(0,3,1,2)` → `[b,t,480,16]` → view `[b,t,7680]` → `conv_out` Linear 7680→1280.
4. `+ sinusoidal positional_embedding[:T]` (runtime, dim 1280).
5. 32 transformer layers, **post-norm**, **full bidirectional** attention
   (n_window=50 chunks the CNN, NOT attention) → reuse `AudioEncoderLayer`
   (note k_proj has NO bias in Whisper layers — CONFIRM against these weights:
   the dump shows k_proj.bias present, so all of q/k/v/out carry bias here).
6. `ln_post` → `proj1` (1280→1280) → **gelu** → `proj2` (1280→2048).
7. Output ~13 tokens / 100-frame chunk, 2048-dim (== text hidden). No extra avg-pool.

Output length: `feat = (L-1)//2+1` three times.

## Milestones

1. **Audio encoder** `Sources/FFAI/Models/Audio/Omni/Qwen3OmniAudio.swift`:
   conv2d stem (3× `Ops.conv2d` k3s2p1 + gelu), permute+reshape, `conv_out`,
   sinusoidal pos, 32× `AudioEncoderLayer`, `ln_post`, `proj1`→gelu→`proj2`.
   `encode(waveform:device:) -> [nTok,2048]`. Mel front-end: 128-bin log-mel
   (reuse `AudioPreprocessing`; confirm hop/window/n_fft vs Whisper).
2. **Family entry** `Sources/FFAI/Models/Qwen3Omni.swift`: build text via
   `Qwen35Hybrid.loadModel` (synthesized text_config + `thinker.` prefix);
   vision via `Qwen3VLVisionModel.load(thinker.visual.*)` → `asVisionEncoder()`;
   audio via milestone 1; assemble `VisionModel(... audioTokenId:151675,
   audioEncode:)`. `handles`: arch `Qwen3OmniMoeForConditionalGeneration` /
   model_type `qwen3_omni_moe`.
3. **Route** in `Model.swift` (before the generic Qwen35VL block; gate on
   `Qwen3Omni.handles`). defaultGenerationParameters = `Qwen35Hybrid`'s.
4. **Test** `Tests/ModelIntegrationTests/Omni/Qwen3OmniIntegrationTests.swift`
   (gated `enableOmniSuites`, load-once): load / text / image / video / audio.
   Use `ekryski/Qwen3-Omni-30B-A3B-Thinking-4bit`.

## Open questions / risks

- **Vision name drift**: loader expects `pos_embed.weight` + `merger.linear_fc1`;
  Omni dump shows `thinker.visual.pos_embed` (bare?) + `merger`/`merger_list`
  groups. Verify exact tensor names; the Omni ViT may need a small adapter vs
  the stock Qwen3-VL tower (esp. deepstack `merger_list`).
- **Mel front-end params** for the conv2d encoder (n_fft/hop) — confirm vs
  processor (`preprocessor_config.json`).
- **Quant**: `ffai convert Qwen/Qwen3-Omni-30B-A3B-Thinking --bits 4
  --embedding-bits 4 --upload-repo ekryski/Qwen3-Omni-30B-A3B-Thinking-4bit`
  auto-starts after the MiniCPM-V-4.5 convert frees memory (watcher pid logged
  to `/tmp/qwen3omni-convert.log`). Routers stay full-precision (default),
  vision/audio towers stay bf16 (default) — matches the loaders (`loadLinear`
  probes `isQuantized` per-module).
- Each 30B-MoE decode is multi-minute; budget the milestone runs accordingly.
