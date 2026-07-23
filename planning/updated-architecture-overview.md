# FFAI — Updated Architecture Overview & Consolidation Opportunities

**Status:** Analysis / planning doc. Not a behavior-change spec — an opportunity map.
**Date:** 2026-06-05
**Scope:** Whole `Sources/FFAI` + `Tests`. Complements (does not replace)
[`architecture.md`](architecture.md) (build-time pipeline + token flow visuals)
and [`model-registry-and-inheritance-plan.md`](model-registry-and-inheritance-plan.md)
(the registry-unification plan that just landed — commits `be2a852`…`d1dd07a`).

This document answers ten questions the team asked:

1. Where to consolidate duplicate code.
2. Where CPU-side inference remains (GPU-everywhere gaps) — Ops vs iron kernels.
3. Whether model loading is consistent across families.
4. How to normalize integration tests to *load + per-modality inference only*.
5. Dead code.
6. Inefficient / redundant code.
7. Integration tests doing more than #4.
8. `Sources/` files missing test coverage.
9. The model inheritance/composition trace + diagram (6 representative models).
10. Where to apply Swift OO / event-oriented best practices.

Every finding cites `file:line`. A prioritized roadmap is at the end (§12).

---

## 1. How the system actually works (the layer stack)

One public type — `Model` ([Model.swift:86](../Sources/FFAI/Model.swift)) — wraps
**any** modality's concrete engine behind `engine: any ModelEngine`. The
modality taxonomy is overlapping *sets* (omni = text + audio-in; ocr ≈ vlm), which
single inheritance can't model, so each modality is a **protocol that refines the
`ModelEngine` marker** and a family conforms to one or several.

```
                         PUBLIC SURFACE
   ┌───────────────────────────────────────────────────────────────┐
   │ Model  (final class, @unchecked Sendable)   Model.swift:86      │
   │   .engine : any ModelEngine                                     │
   │   .kind   : ModelKind   (.llm/.vlm/.ocr/.omni/.stt/.tts/.sts/.vad)│
   │   .vlModel: VisionModel?      .tokenizer  .config  .capabilities │
   │   .generate(_:ModelInput) / .generateStream(_:)  ModelInference  │
   │   .events : AsyncStream<ModelLifecycleEvent>                     │
   └───────────────────────────────┬───────────────────────────────┘
                                    │ load()
                   ┌────────────────┴─────────────────┐
                   ▼                                   ▼
        ModelRegistry.entry(for:config)      (two load paths diverge here)
        ModelRegistryTable.swift             Model.swift:406
        ┌──────────────────────┐             ┌────────────────────────────┐
        │ TEXT / VL tier        │            │ AUDIO / VAD tier             │
        │ entry.load(c,w,o,d)   │            │ entry.loadFromDirectory(...) │
        │ • central SafeTensors │            │ • family builds own bundle   │
        │   bundle + residency  │            │ • async, attaches tokenizer  │
        │   pin (Model:424-436) │            │ • NO residency pin           │
        └──────────┬───────────┘             └──────────────┬──────────────┘
                   ▼                                         ▼
        ModelRegistry.Loaded{engine, vlModel,    ModelRegistry.Loaded{engine,
          defaultGenParams, capabilities}          tokenizer, capabilities}

                         ENGINE PROTOCOL LAYER   ModelType/ModelEngine.swift
   protocol ModelEngine {}                          (pure marker)
     ├─ TextEngine          → LanguageModel: Module, TextEngine  (LLM/VLM backbone)
     ├─ Transcribing        → STT / omni-audio-in  (refines StreamingTranscriber)
     ├─ Synthesizing        → TTS                  (refines StreamingSynthesizer)
     ├─ SpeechTransforming  → STS denoise/separate
     └─ Detecting           → VAD / diarization / endpointing

                         INFERENCE DISPATCH   ModelType/ModelInference.swift
   Model.generate(.text)      → generate(messages:) → Generation/Generate.swift
   Model.generate(.transcribe)→ engine as? any Transcribing  → transcribeStream
   Model.generate(.speak)     → engine as? any Synthesizing   → synthesizeStream
   Model.generate(.transform) → engine as? any SpeechTransforming → process
   Model.generate(.detect)    → engine as? any Detecting      → detect
```

Conformances for the audio families are wired in one file
([EngineConformances.swift](../Sources/FFAI/ModelType/EngineConformances.swift)).
Shared building blocks: `ModelLayers/` (`DecoderLayer`, `Linear`, `RMSNorm`,
`LayerNorm`, `Embedding`, `MoELayer`, `VisionEncoderLayer`, `AudioEncoderLayer`),
`Vision/VisionEncoder`, `Audio/AudioEncoder`, cross-modal splice
`VisionModel.swift`, GPU ops `Ops/`, media facade `Media/MediaUtils.swift`.

**Key architectural truth:** there are **three** "transformer block" patterns,
by modality, and they do **not** share a base:

| Pattern | Type | Used by |
|---|---|---|
| Text **dense** | homogeneous `[ConcreteLayer]`, layer is a plain `Module`, model calls `layer.forward(...)` | Llama, Qwen2/3, Mistral, Phi, Gemma 2/3, … |
| Text **hybrid** | heterogeneous `[any DecoderLayer]`, uniform `layer.decode(...)` loop + per-index cache downcast | Qwen 3.5, NemotronH, Jamba, Granite4, FalconH1, LFM2 |
| Vision/audio **encoder** | `VisionEncoderLayer` / `AudioEncoderLayer` — near-identical pre-norm bidirectional blocks | all VLM towers / all speech encoders |

`DecoderLayer` is text-stack-only; vision/audio decoder layers do **not** conform.

---

## 2. Model inheritance & composition trace (Q9)

Six representative models, traced load → inference, with the *type/protocol* at
each hop. The point: see which files a model passes through and how its interface
morphs from "directory on disk" to "coherent text out."

### 2.1 Universal hops (every model)

```
"id-or-path"
  → Model.load                       Model.swift:381    (String)
  → ModelLocator.resolve             Loader/ModelLocator        (URL dir)
  → ModelConfig.load                 Loader/ModelConfig         (ModelConfig)
  → ModelRegistry.entry(for:)        ModelRegistryTable:579     (ModelEntry?)
  → entry.load | entry.loadFromDirectory                        (Loaded)
  → Model(engine: any ModelEngine, kind: ModelKind, …)          (Model)
  → Model.generate(_:ModelInput)     ModelInference.swift:36    (ModelOutput)
```

### 2.2 MiniCPM-o (omni — the most-composed model)

A `VisionModel` wrapping a **reused Qwen3 dense backbone** + SigLIP resampler
tower + Whisper-medium audio tower. (Note: Omni uses `MiniCPMOVision.swift` +
`MiniCPMOResampler.swift` + `MiniCPM4Audio.swift`; `MiniCPM4Vision.swift` is the
*4.6* vit-merger tower, a common point of confusion.)

```
ModelRegistryTable.swift:249  entry "MiniCPM-V 4.x", kind=.vlm, sniff: MiniCPM4.handles
  → ModelRegistry.dispatchAndLoad → isVisionLanguage → entry.load
  → MiniCPM4.variant(for:)              Models/MiniCPM4.swift:137   → MiniCPM4Omni
  → MiniCPM4Omni.load → MiniCPM4.loadResampler(withAudio:true)
        ├─ TEXT  : builds synthetic qwen3 config → Qwen3Dense.loadModel
        │          Models/Text/Qwen3Text.swift   → Qwen3Model: LanguageModel
        ├─ VISION: MiniCPMOVision: VisionEncoder (reuses VisionEncoderLayer)
        │          + MiniCPMOResampler (perceiver → 64 tokens/image)
        └─ AUDIO : MiniCPM4Audio (reuses AudioEncoder/AudioEncoderLayer)
                   + 2× AnyLinear projector + avg-pool(5)
  → VisionModel.init(visionEncoder, engine: any LanguageModel,
        imageTokenId/videoTokenId/audioTokenId, audioEncode closure)
  → Loaded(engine: Qwen3Model, vlModel: VisionModel, kind: .omni)

INFERENCE (multimodal):  model.vlModel.generate(promptTokens:audio:image:videoFrames:)
  → encodeAudio/encodeImage → spliceMultimodal (substitute soft-token rows)
  → engine.forwardMulti(inputEmbeddings:) prefill → engine.forward(tokenId:) decode
INFERENCE (text-only):   Model.generate(.text) → textEngine (= Qwen3Model)
```

> **Interface inconsistency (flagged):** Omni is `kind=.omni` but its audio/image
> path is reached **only** through `model.vlModel.generate(...)` — there is **no
> `.omni` / image / audio-in case in `ModelInput`** ([ModelIO.swift:94](../Sources/FFAI/ModelType/ModelIO.swift)).
> The unified `Model.generate(_:)` sum type covers text/transcribe/speak/transform/detect
> but not vision/omni multimodal. See §11.3.

### 2.3 Qwen 3.5 (text hybrid GDN ↔ attention)

```
ModelRegistryTable.swift:508  entry "Qwen 3.5", kind=.llm
  → ModelRegistry.loadQwen35            ModelRegistry.swift:490
  → Qwen35.variant(for:)                Models/Qwen35.swift   → Qwen35Hybrid
  → Qwen35Hybrid.loadModel              Models/Text/Qwen3xText.swift
       builds [any DecoderLayer] = Qwen35GDNLayer | Qwen35AttentionLayer per layer_types
       each wraps a mixer + Qwen35FFN enum (.dense | .moe → shared MoELayer)
  → Qwen35Model: LanguageModel (+ SpeculativeVerifier)
INFERENCE: Model.generate(.text) → Generate.driveGeneration
  → Qwen35Model.forwardMulti (prefill) / forwardSample (decode)
  → layer.decode(h, position:, cache:, cmd:, device:)   [DecoderLayer loop]
```

Reuses `DecoderLayer`/`StatelessLayerCache`, `RMSNorm`, `Linear`/`QuantizedLinear`,
`MoELayer`. Reimplements only the GDN mixer (`GDNStateCache` double-buffer) +
sigmoid-gated shared expert. Uses the "Jamba pattern" command-buffer discipline
(layers commit `cmd`; final norm+lm_head queued on a pristine caller `cmd`).

Contrast **Qwen3 dense** ([Qwen3Text.swift](../Sources/FFAI/Models/Text/Qwen3Text.swift)):
homogeneous `[Qwen3Layer]`, `Qwen3Layer` is a plain `Module` (not `DecoderLayer`),
direct `layer.forward(...)`. Same family root, different block pattern.

### 2.4 / 2.5 / 2.6 — Kokoro (TTS), Whisper (STT), DeepFilterNet (STS), Silero (VAD)

| Model | Registry entry | Engine type | Conformance | Inference entry | Shared infra reused |
|---|---|---|---|---|---|
| **Kokoro** TTS | Table:723 `.tts` | `KokoroModel: @unchecked Sendable` | `Synthesizing` **marker** (EngineConformances:50) | `synthesizeStream(text:voice:speed:)` → acoustic (GPU or CPU) → `KokoroVocoder` iSTFT | `AudioModel` voice surface, `Ops`/`Tensor` vocoder; **reimplements** whole StyleTTS2 acoustic stack |
| **Whisper** STT | Table:685 `.stt` | `WhisperModel` | `Transcribing` **marker** (EngineConformances:34) | `transcribeStream(_:…)` → `encodeAudio` (GPU) → AR `decoderLogits` (mix GPU + `cpuAttention`) | `AudioEncoder`/`AudioEncoderLayer`, `Linear`, `LayerNorm`; **reimplements** cross-attending decoder |
| **DeepFilterNet** STS | Table:863 `.sts` | `DeepFilterNetModel` | `SpeechTransforming` **adapter body** `process→enhance` (EngineConformances:68) | `process` → `enhance(waveform:)` — STFT→ERB→net→iSTFT, **entirely CPU/DSP** | only `SafeTensorsBundle`; **bypasses all shared infra** |
| **Silero** VAD | Table:905 `.vad` | `SileroVADModel` | `Detecting` **adapter body** `detect→detect(audio:)` + output-type mapping (EngineConformances:124) | `detect(audio:sampleRate:)` → `VADOutput` → `[DetectionEvent]`, **entirely CPU** (`VADCompute`) | only `SafeTensorsBundle`/`Tensor`; bespoke CPU compute |

**Conformance consistency tiers** (the interface inconsistency to normalize, §11.4):

- **Pure marker** (cleanest): LLM (`LanguageModel`), STT (`Transcribing`), TTS (`Synthesizing`).
- **Adapter body** (carries a real method body in EngineConformances): STS (`process`→`enhance`), VAD (`detect`→`detect(audio:)` + 3 output-type→`DetectionEvent` maps).
- **Outliers:** SAM-Audio is `.sts` but **not** `SpeechTransforming` (reached via `engine as?`, EngineConformances:80); omni multimodal has **no generic-dispatch case** (reached via `model.vlModel`).

---

## 3. Consolidate duplicate code (Q1)

| # | Cluster | Files / lines | What's duplicated | Action | Effort |
|---|---|---|---|---|---|
| **D1** | Registry `loadXxx` fns | `ModelRegistry.swift:195–552` — **23 near-identical** 8–10-line fns | `variant(for:)` → `loadModel` → `Loaded(...)` boilerplate, differs only by family enum name | Generic `loadVariant<V: ModelVariant>(...)` **or** fold into the `ModelEntry.load` closures (which already just call `loadXxx`, a redundant indirection). Keep `loadOLMo2` (no variant) + `loadLFM2` (vocab override) bespoke | ~280 LOC removed. **High value, low risk** |
| **D2** | Family `enum` + `variant(for:)` + `XxxVariant` + `XxxError` | 33 `Models/*.swift` roots | 29 structurally-identical per-family variant protocols (all declare the same 3 members); ~28 `variant(for:)` ignore config and return `.self`; ~26 single-case `XxxError.missingConfig` with identical descriptions | Shared `ModelVariant` protocol + default `variant(for:)->Self`; shared `ModelLoadError.missingConfig(family:)`. Only Gemma4/LFM2/MiniCPM4 override (genuinely multi-variant) | ~29 protocols + ~26 enums removed |
| **D3** | STT/omni streaming-decode loop | `GraniteSpeech:2161`, `Qwen3ASR:875`, `GLMASR:873`, `CohereTranscribe:1476`, `VoxtralRealtime:1124`, `LFMAudio:1382`, `Qwen3.swift:498` (QwenOmni) | **~900 LOC** — each reimplements `prepareTranscribe`→TranscribeState, the per-token loop, the *verbatim* "decode prefix, emit suffix" delta idiom, a private `sampleTok`, and a synchronous `forwardDecodeToken` (commit+wait) | Shared `ARTranscribeDriver` (or default method on `Transcribing`) parameterized by `(prepareState, forwardToken, eosIds, tokenizer)`. **Also fixes the O(n²) decode + the CPU-sample, §5/§6** | **~900 LOC removed. Highest-leverage consolidation** |
| **D4** | Audio `handles(_:)` prefix | ~29 audio roots | First two lines (`modelTypes.contains` / `architectures.contains`) duplicate `ModelEntry.matches` (ModelEntry.swift:98) which already ORs the structural sniff | Reduce each `handles` to just the structural sniff; let `matches` do membership | ~58 LOC removed |
| **D5** | Mel/STFT front-ends | mostly shared via `MediaUtils.logMelSpectrogram`; local copies in `Parakeet`, `MossFormer2SE`, some TTS | Verify Parakeet/MossFormer2-SE aren't re-rolling shared log-Mel (TTS forward-STFT/iSTFT tails are legitimately different) | Targeted; partial | small |

`VisionEncoderLayer` vs `AudioEncoderLayer` are near-identical pre-norm
bidirectional blocks (same structure, same dual GPU-`sdpaBidirectional`/`cpuAttention`
shape) — a *borderline* shared-base candidate; the divergence (GELU, a couple of
head-dim kernel choices) is small enough that consolidation is optional.

---

## 4. GPU-everywhere — remove CPU-side inference (Q2)

**Headline:** No Ops are missing a backing kernel. The 159-family iron kernel
tree (`~/Development/personal/ai/iron/crates/wh-iron-std/src/ffai/`) covers
the entire `Ops.*` surface; every unusual op (lstm, vocoder, mel_spectrogram,
adain1d, snake1d, audio_conv1d, resize_normalize, im2col_patch, patch_unfold_qwen,
frame_diff_luma) maps to a real kernel. So the gap is **wiring**, not kernels.

> The generated kernels themselves are build artifacts (gitignored under
> `Sources/IronSwift/Resources/kernels/*.metal` + `Generated/`), emitted by
> `make regenerate-kernels`. `OpsCoverageNotes.swift` documents the ~30 kernel
> families *intentionally* not Ops-wrapped (internal composite blocks, bit-width
> variants auto-routed inside a wrapper, FP4/FP8 blocked on the `DType` enum,
> FFT/sort/scan with no caller, MLX bench-only).

### 4.1 Sanctioned CPU paths — leave as-is

One-time-per-utterance codec/DSP (`Audio/AudioPrimitives.swift` `AudioMath` — no
iron general strided/grouped/transposed conv1d; backs BigVGAN/Encodec/Mimi/
SNAC/DAC/Vocos), `Audio/Mimi*.swift`, `Audio/DeepFilterNetDSP.swift` (vDSP FFT),
`VADCompute.swift` (tiny models), Kokoro/Soprano/FishSpeech acoustic stacks
(GPU variants already exist alongside), and **load-time** weight folds
(Gemma `1+w` RMSNorm fold, GPT-OSS MXFP4→f16 transcode) + trivial token-id /
4-byte-sample I/O.

### 4.2 CPU compute on inference paths — candidates to move to GPU

| # | Location | Computes | Hot path? | GPU op status | Priority |
|---|---|---|---|---|---|
| **C1** | `Generate.swift:281–309` `.cpuSample` + `Sampling.swift` | filtered sampling: full-vocab readback + CPU rep-penalty + **CPU `indexed.sort` over the whole vocab (151,936 for Qwen3)** + top-K/P/min-P + softmax + categorical | **YES — per decode token** whenever `topK>0 ‖ topP<1 ‖ minP>0 ‖ repPenalty≠1` | **Exists, ZERO callers.** `Ops.logitsTopKMask/TopPMask/MinPMask/RepetitionPenalty/Temperature` + `softmaxCategoricalSample` all shipped + validated | **#1 — biggest hot-path win, ops already exist** |
| **C2** | `MoELayer.swift:724–792, 1026` | MoE host-side permutation plan: readback router idx/weights, build gather idx, `planTuples.sort`, host gather copy, `cmd.waitUntilCompleted()` barrier | per MoE step (decode t=1 trivial; prefill t·topK) | partial — GPU router (`moeRouterTopKMany`) + `moeUnpermute` exist; sort/plan/gather still CPU (no GPU sort wired) | low (index-shuffle; GEMMs already GPU) |
| **C3** | `PaligemmaVision.swift:742–766` | patch-embed conv via `DispatchQueue.concurrentPerform` + manual accumulate | once/image | `Ops.patchEmbed`/`conv2d` exist | low-med |
| **C4** | `VisionEncoderLayer.swift:136–142` `cpuAttention` | bidirectional MHA fallback for **head_dim 32** (FastViT-HD) | once/image, gated | kernel exists (`sdpaBidirectional`) but `OpsValidation` returns false for d32 — needs a verified production cell | low |
| **C5** | `Parakeet:337/398`, `SenseVoice:1024` (hand-rolled radix-2 FFT), `CohereTranscribe:485–618` | mel front-ends on CPU (STFT/filterbank/log) | once/clip | `Ops.melSpectrogram`/`melSpectrogramMagnitude` exist + used by Whisper/GLMASR/Qwen3ASR/Voxtral/Gemma4; these 3 diverge (NeMo dither, Kaldi pre-emphasis, Cohere norm) | med |
| **C6** | `GraniteSpeech:1003` `TODO(perf)`, padded last block | encoder SDPA padded-block fallback | once/clip, gated | d128 `sdpaRelPosConformer` already used 10 lines below — **the TODO is STALE** | trivial (update comment) |

**Action list:** (1) wire C1 — `Generate.swift`'s `cpuSample` through the five GPU
logits ops + `softmaxCategoricalSample` (removes the only per-token CPU compute in
text decode); (2) C5 — route Parakeet/SenseVoice/Cohere through `Ops.melSpectrogram`
+ a GPU post-norm; (3) C3 — Paligemma patch-embed → `Ops.patchEmbed`; (4) update the
stale C6 TODO; (5) C2/C4 are low-priority verification/sort work.

---

## 5. Model-loading consistency (Q3)

The text/VL ↔ audio split is *mostly* clean (text/VL receive a central
residency-pinned bundle; audio self-bundles from a directory). The leaks:

| Issue | Detail | Fix |
|---|---|---|
| **3 entry-point names for one job** | `load(directory:)` (TTS/STT/STS), `loadFromDirectory(_:)` (all 5 VAD), `load(from:)` (DeepFilterNet:363, CohereTranscribe:1773 — which **also** has `load(directory:):1903`) | Pick one — `load(directory:device:)` — adapt VAD + DeepFilterNet |
| **Arbitrary async/sync split** | 7 audio families `async` (Qwen3TTS, Marvis, LlamaTTS, Qwen3TTSBase, Soprano, GraniteSpeech, CohereTranscribe) vs rest sync, no modality reason; closure type is `async throws` for all anyway | Normalize signature; let sync loaders just not `await` |
| **`handles` vs `matches`** | audio families expose `handles(_:)` (wired as `sniff:`) duplicating `ModelEntry.matches` membership; text/VL families have no `handles`, rely on entry sets | Drop the membership half of `handles` (D4) |
| **Bundle ownership leaky** | text/VL build bundle centrally + **residency-pin** (Model:424–436); audio builds its own bundle inside each loader (×25) and **bypasses residency pin** (Model:402–405). SAM-Audio is a half-exception — the *registry closure* builds its bundle (Table:897) | Decide: either give audio the residency-pin path, or document the opt-out uniformly |
| **Capabilities — consistent ✓** | every audio entry declares the right `Capability.*` set; VL gates `.audioIn`/`.videoIn` dynamically in its `load` closure | none |

---

## 6. Dead code (Q5)

| Confidence | Item | Location | Note |
|---|---|---|---|
| High | `OpsCoverageNotes` enum (293 LOC) | `Ops/OpsCoverageNotes.swift` | **Zero references** in `Sources/` or `Tests/` outside the file. Docs-as-code with a public API nothing reads → convert to markdown/comment or delete |
| High (test-only) | 15 typed accessors `llama`/`qwen3`/`qwen35`/… | `Model.swift:187–241` | Each `engine as? XModel`; **zero production callers**, only `Tests/` use them. Test-only API on the public surface — see §11.1 |
| Medium | `Qwen36.variant(for:)` pure delegation | `Qwen36.swift:62` | Just calls `Qwen35.variant`; verify it carries unique state or fold into the Qwen35 entry's model_types |
| Medium | PocketTTS stub | `PocketTTS.swift:67, 416` | "not yet wired in" / "Stage 3: wire Mimi codec" — a registered family that can't synthesize; incomplete-feature catalog row, not strictly dead |

No `#if false`, no large commented-out blocks (7 single-line code-like comments, all
legit docs). ~50 TODO/FIXME, mostly legit roadmap pointers (+ the stale C6 one).

---

## 7. Inefficient / redundant code (Q6)

| # | Location | Problem | Fix |
|---|---|---|---|
| **I1** | STT streaming delta — `GraniteSpeech:2211`, `Qwen3ASR`, `CohereTranscribe`, `LFMAudio` (4 sites) | **O(n²)**: every decode token calls `tokenizer.decode(generated)` **and** `tokenizer.decode(generated.dropLast())` over the whole growing prefix + an `Array(...dropLast())` copy/token | Keep `prev` = last step's `cur`; decode only `[nextToken]` (the text path already does — `Generate.swift:409`). Folds into the D3 shared driver |
| **I2** | Per-token sampler buffer allocs — `LanguageModel.swift:343` (`forwardSample` 1×), `:373/378/383` (`forwardSampleCategorical` **3×** per token) | thousands of transient 4-byte `MTLBuffer`s on the hottest path; temperature buffer is constant across the whole generation | Hoist 3 reusable scratch buffers into the engine/cache; `memcpy` per step |
| **I3** | per-token `commit();waitUntilCompleted()` in audio decode (340 wait sites; `GraniteSpeech:2288`, `Qwen3.swift:425`) | fully serializes CPU↔GPU each token, no pipelining; audio AR decoders read back **full logits** then CPU-sample instead of the fused 4-byte `forwardSample` readback the text path uses | route audio AR decode through `LanguageModel.forwardSample/Categorical` (folds into D3) |
| **I4** | default `forwardMulti(inputEmbeddings:)` — `LanguageModel.swift:266–277` | per-position self-committing cmdbuf + `waitUntilCompleted` + `slicedRows().reshaped()` alloc per row; any VL family not overriding pays full serialization on image prefill | audit which VL towers still inherit the default; override or batch |
| **I5** | `gpuEmbedLookup` — `GraniteSpeech:2239` | a full CPU↔GPU roundtrip just to gather one embedding row, then a second for the forward — 2 syncs/token where 1 suffices | fuse embed+forward; folds into D3 |

---

## 8. Integration-test normalization (Q4 + Q7)

### 8.1 The gold standard

[`MiniCPM45OmniIntegrationTests.swift`](../Tests/ModelIntegrationTests/Omni/MiniCPM45OmniIntegrationTests.swift):
one `@Suite`, `.serialized`, gated; **one static memoized `Task` load via
`loadModel(modelId)`** (the canonical `Model.load` wrapper behind `ModelLoadLock`);
**one `@Test` per supported modality** (`load` capability-surface assert, `text`,
`image`, `audio`, `video`), each ending in the **behavioral** `expectCoherentOutput`
+ content-mention helpers (`expectMentionsDog`/`…cat`). Media fixtures come from
`VisionTestHelpers`/`AudioTestHelpers`. **Its one blemish:** the `sine(seconds:)`
generator is inline at lines 42–47.

> Worth noting even the gold standard hand-builds the multimodal **prompt framing**
> (`<image>`+placeholder-run+`</image>`, lines 94–101) in the test. Because there's
> no unified multimodal `ModelInput` case (§2.2), every VLM test must do this. If
> that framing is *model-required normalization*, it belongs inside the model before
> `generate` (§11.3); if it's caller prompt-authoring, it's acceptable in the test.

### 8.2 The `sine` fix (explicitly requested)

There is **no shared sine helper** — it is copy-pasted across **~25 test files**
(inline loops, `static func sine`, `syntheticWaveform`, `syntheticClip`,
`sineWaveform`) at varying freqs/rates. `AudioTestHelpers.swift` today provides
`clean001Waveform()`/`conversationalAWaveform()` which **already call
`MediaUtils.loadWaveform`** — so the helper→MediaUtils pattern is established.
`AudioPreprocessing.swift` already owns the front-end math (`hannWindow`,
`melFilterbank`, `resample`, `logMelSpectrogram`).

**Action:**
1. Add `MediaUtils.sineWave(frequency:seconds:sampleRate:amplitude:)` to
   `AudioPreprocessing.swift` (production owns signal generation).
2. Add a thin `AudioTestHelpers.tone(...)` + `clip(...)` (silence+tone+silence for
   VAD) that **delegate to `MediaUtils`** — mirroring `clean001Waveform`.
3. Delete the ~25 inline copies (start with the two `static func sine` in the Omni
   gold-standard files); fold the Kokoro/StyleTTS2 complex-spectrogram fixtures into
   a sibling `syntheticSpectrogram` helper.

### 8.3 Tests doing more than load + per-modality inference (Q7)

Two systemic deviation classes, concentrated in **Audio/**:

- **D-load** — bespoke loading (`WhisperModel.load(directory:)`, `KokoroModel.load`,
  `SileroVADModel.fromPretrained`, `ModelLocator().resolve` + concrete `.load`)
  instead of `Model.load`. Bypasses the registry.
- **D-internal** — numeric/structural assertions (config fields, `.layers.count`,
  `Registry.handles`, finite-feature/variance floors, **GPU↔CPU parity**, shape
  math) instead of `expectCoherentOutput`/content checks.
- **D-preproc** — inline signal generation (§8.2).
- **D-decode** — hand-rolled prefill+sample loop instead of `generate`.

| Test file | Modalities | Deviations (worst first) |
|---|---|---|
| `Audio/TTS/KokoroIntegrationTests.swift` | speak | **Worst: `plbertGPUParity` (235-276) + `generatorGPUParity` (278-336) are op-level GPU↔CPU parity tests on internal submodules**, read voices safetensors by hand (267). **Belongs in `FFAITests`.** + D-load + inline sine (83) |
| `Vision/PaligemmaIntegrationTests.swift` | image→text (OCR) | **D-decode: hand-rolls the entire greedy decode loop + private `argmaxOnHost` (109-151)**; 6 config asserts (58-63) |
| `Audio/STS/MossFormer2SEIntegrationTests.swift` | enhance | dedicated `config` (96-113) + `registry` (115-126) **unit tests inside an integration suite**; D-load; `syntheticWaveform` (83-92) |
| `Audio/STT/WhisperIntegrationTests.swift` (pattern shared by all 9 STT) | transcribe | D-load (51-56); config + `encoder.layers.count`, finite/variance, degenerate-logit (58-117); separate `encode`/`decode` sub-stage tests; inline sine (78,95) |
| `Audio/VAD/SileroVADIntegrationTests.swift` | detect | D-load `.fromPretrained` (70-74); branch-geometry + exact frame-count math (81-96); `syntheticClip` (49-65). (detect-behavior asserts are OK) |
| `Audio/Omni/QwenOmniIntegrationTests.swift` | transcribe | mixes both load styles **in one file** (`QwenOmniModel.load` for most, `Model.load` for `transcribeStreaming` 116-165); `audioEncoder.layers.count` (62-64); inline sine (75) |
| `Text/*` + `Quantized*bit*` (pattern) | text | compliant shape (load via `loadModel`, end in `expectCoherentOutput`) but pile on engine-routing + config-shape + cache-type + raw-`forward` asserts (e.g. `Qwen2Text:48-75`). Low severity |

**Net:** Text + Vision + Omni broadly follow the standard (differ mainly in extra
config-shape asserts). **All of Audio deviates structurally** and is the bulk of the
cleanup. Move op-level/parity/config/registry tests **down to `FFAITests`**; in the
integration suite keep load-via-`Model.load` + one coherent inference per modality.

**Legit exceptions** (cross-cutting infra suites, keep): `KVCacheGrowth`,
`ModelInspection`, `ModelDeterminism`, `ModelKVCacheMatrix`, `SlidingWindow`,
`EmbeddingPrefill` — they test infra, not a single model's modality.

---

## 9. Source files missing test coverage (Q8)

> Caveat: vision/audio/omni integration groups are **gated `false`** in
> `IntegrationGroupGating`, so many model files nominally "covered by an integration
> test" are dark on a default `make test`. The ModelType/KVCache/Ops gaps below are
> covered by **no** suite, gated or not.

**HIGH risk (no covering test at all):**
- `ModelType/ModelInference.swift` (224 L) — the unified `Model.generate` dispatch. No test references it.
- `ModelType/ModelIO.swift` (136 L) — `ModelInput`/`ModelOutput`/`ModelOutputChunk` + request structs. Untested.
- `ModelType/EngineConformances.swift` (148 L) — the audio-engine conformance bridge. Untested.
- `KVCache/AttentionCacheFactory.swift` (298 L) — per-layer cache geometry. No direct test.
- `KVCache/CacheSnapshot.swift` (119 L) — `LayerCacheSnapshot` serialization round-trip (exactly what CLAUDE.md says to test). No direct test.
- `Ops/OpsMath.swift` — untested public ops: `arange`, `argmin`, `div`, `sub`, `neg`, `recip`, `logsumexp`, `maxElem`, `minElem`, `copyKernel` (empty-body-MSL→all-zeros hazard makes untested kernels high-risk).

**MEDIUM risk (real logic, only gated-off integration coverage):**
- `Audio/FishS1DACQuantization.swift` (879 L) — no test references any of its types. Largest dark file.
- `Models/Vision/MiniCPMOResampler.swift` (472 L), `MiniCPMOVision.swift` (228 L), `Models/Audio/Omni/MiniCPM4Audio.swift` (259 L) — only via gated-off omni/vision suites.
- `Audio/SNACBlocks.swift` (311 L) — confirm `SNACCodecTests` reaches encoder/decoder block types.

**LOW (thin wrappers / indirectly covered, no action):** `Models/Text/{FastVLM,MiniCPM4,Mistral3,Pixtral}Text.swift`, `KokoroFrontEndGPU.swift`, `ModelRegistryTable.swift` (data, covered via routing tests), `OpsCoverageNotes.swift` (delete per §6).

---

## 10. Swift OO & event-oriented architecture (Q10)

| # | Opportunity | Location | Recommendation |
|---|---|---|---|
| **O1** | 15 typed `engine as? XModel` accessors — anti-pattern, couple public `Model` to every concrete family, **test-only** | `Model.swift:187–241` | Delete; tests downcast `model.engine as? Llama3Model` directly (or one generic `model.engine(as:)`). Removes `Model`'s compile dependency on every family type |
| **O2** | `@unchecked Sendable` 122× across 77 files, **no inline rationale** (CLAUDE.md requires one). Asymmetry: 60 audio engines self-mark; text `LanguageModel` engines do **not** (ride on `Model`'s blanket mark; `Module` is plain `AnyObject`) — two strategies for one safety story | grep `@unchecked Sendable` | Make `ModelEngine: Sendable` (or `Module: Sendable`) with one documented `@unchecked` at the conformance level + shared rationale; or at minimum add the required one-line reason to each + pick one convention |
| **O3** | **Lifecycle events underused.** `Model.events` is emitted only from `enable`/`disable` (Model:354,371) + load `.ready` (Model:475). **Generation bypasses it** — `driveGeneration` (Generate.swift:228) drives its own `AsyncThrowingStream<GenerationChunk>` and never emits `.generating`/`.idle`, despite the doc comment (Model:255-259) claiming that lifecycle. Load progress (download %, weight-load %) also not evented | `Model.swift`, `Generate.swift` | Emit `.generating`/`.idle` around `driveGeneration`; add a load-progress event. The architecture exists; the two highest-value paths don't use it |
| **O4** | ~26 single-case `XxxError.missingConfig` enums | family roots | Shared `ModelLoadError.missingConfig(family:)` (keep richer enums like `Qwen3Error`). Same as D2 from the OO angle |
| **O5** | 33 family `enum`-of-statics + per-family variant protocol where one `ModelVariant`/`ModelFamily` protocol with default `variant(for:)->Self` fits (28 are single-variant, config-ignoring) | `Models/*.swift` | Single protocol; only Gemma4/LFM2/MiniCPM4 override dispatch |
| **O6** | `ModelInference` engine-resolution helpers fine as-is | `ModelInference.swift:192-215` | Optional: one generic `engine<T>(_:for:)`; low priority |

---

## 11. Cross-cutting architectural notes

1. **`Model`'s family coupling (O1)** is the single clearest OO smell — 15 concrete
   types named in the public wrapper, all for tests.
2. **The registry indirection is double-layered (D1+D2):** `ModelRegistryTable`
   `load` closure → `ModelRegistry.loadXxx` → `Xxx.variant(for:)` → `variant.loadModel`.
   Two of those three hops are mechanical. Collapsing to closure → `loadVariant<V>`
   removes ~300 LOC.
3. **No unified multimodal input (§2.2):** `ModelInput` covers text/transcribe/speak/
   transform/detect but image/video/omni go through `model.vlModel`. Either add
   `.vision`/`.omni` cases to the sum type, or document that VLM/omni is a separate
   surface. This also drives the integration-test prompt-framing question (§8.1).
4. **Conformance tiers (§2.6):** marker (LLM/STT/TTS) vs adapter-body (STS/VAD) vs
   outlier (SAM-Audio/omni). Normalizing the audio engine entry-point names (D3/§5)
   would let STS/VAD become markers too.

---

## 12. Prioritized roadmap

Ranked by (impact × confidence) ÷ effort. Each is independently shippable.

| Rank | Action | Refs | Impact | Effort |
|---|---|---|---|---|
| 1 | **Wire `Generate.cpuSample` through the 5 GPU logits ops + `softmaxCategoricalSample`** — removes the only per-token CPU compute in text decode | C1 §4.2 | High (perf, GPU-everywhere) | S — ops exist + validated |
| 2 | **Extract `ARTranscribeDriver` for STT/omni decode** — kills ~900 LOC dup, the O(n²) decode, the full-logits readback, and per-token sync | D3 §3, I1/I3/I5 §7 | High | M |
| 3 | **Collapse the 23 `loadXxx` + redundant table closures into `loadVariant<V>`** | D1 §3 | Med (−280 LOC, clarity) | S |
| 4 | **Normalize audio integration tests to the gold standard**; move Kokoro parity / Whisper sub-stage / MossFormer config+registry tests down to `FFAITests` | §8.3 | Med-High (test integrity) | M |
| 5 | **Centralize `sineWave` in `MediaUtils`/`AudioTestHelpers`, delete ~25 inline copies** | §8.2 | Med (the explicit ask) | S |
| 6 | **Add tests for the dark core: `ModelInference`, `ModelIO`, `EngineConformances`, `AttentionCacheFactory`, `CacheSnapshot`, the 10 untested `OpsMath` ops** | §9 | High (risk) | M |
| 7 | **Shared `ModelVariant` protocol + `ModelLoadError.missingConfig` — drop ~29 protocols + ~26 error enums** | D2 §3, O4/O5 §10 | Med (−LOC, consistency) | M |
| 8 | **Emit generation lifecycle (`.generating`/`.idle`) + load progress on `Model.events`** | O3 §10 | Med (UX/observability) | S |
| 9 | **Delete the 15 typed `Model` accessors; add one-line `@unchecked Sendable` rationales / lift to protocol** | O1/O2 §10 | Med (OO hygiene) | S |
| 10 | **Hoist per-token sampler scratch buffers; audit default `forwardMulti(inputEmbeddings:)` overrides** | I2/I4 §7 | Low-Med (perf) | S |
| 11 | **GPU-ify remaining once-per-clip/image CPU: STT mel front-ends (C5), Paligemma patch-embed (C3); update stale C6 TODO; verify d32 SDPA (C4)** | §4.2 | Low-Med | M |
| 12 | **Normalize audio load entry-points to one `load(directory:)`; decide audio residency-pin; reduce `handles` to sniff-only (D4)** | §5, D4 §3 | Low-Med (consistency) | M |
| 13 | **Delete/convert `OpsCoverageNotes` (293 LOC, unconsumed); resolve PocketTTS stub** | §6 | Low | S |

**Sequencing note:** #2 (ARTranscribeDriver) absorbs I1/I3/I5 and depends on the
audio decoders being able to reach `LanguageModel.forwardSample` — do #1's wiring
mindset first, then #2. #4 (test normalization) is independent and can run in
parallel with everything.

---

## 13. Multi-modal support — status & remaining gaps (Q vs [`multi-modal-support.md`](multi-modal-support.md))

Cross-checked the multi-modal spec against the code on `ek/fix-all-the-things`
(2026-06-05). **Most of the big workstreams have landed since the spec was
written** — streaming, the kernel backlog, MediaUtils, in-core media decode are
done. The spec's §1 "does not work yet" list and §10 roadmap are **stale** in
those areas. What's verified done vs what genuinely remains:

### 13.1 Verified DONE (spec/roadmap now stale)

| Spec item | Status | Evidence |
|---|---|---|
| **Streaming output across ALL modalities** (spec §7a, roadmap #9; Phase K) | ✅ DONE | `VisionModel.generateStream` + `generateMultimodalStream` (VisionModel.swift:527/544/564/738) stream text-out from image/video; `runSampled` (703) replaced greedy-only `runGreedy`. **11 `transcribeStream`** + **13 `synthesizeStream`** engine impls; unified `Model.generateStream(_:ModelInput)` covers text/STT/TTS/STS/VAD. The spec's "VLM streaming output — does NOT work" / "blocked on embedding-prefill entry point" and Phase K's "Streaming is text-only today" are both **outdated.** |
| **VLM respects generation params** (roadmap #2) | ✅ DONE | `runGreedy`→`runSampled`; sampling (temp/top-p/top-k/min-p/rep-pen/seed) on the multimodal decode. |
| **Kernel backlog K1–K7** (spec §12) | ✅ kernels exist | All 7 have Ops wrappers: `resizeNormalize` (K1/K2), `melSpectrogramMagnitude` (K3), `avgPool2dNHWC` (K4), `im2colPatch`/`patchUnfoldQwen` (K5), `posEmb2dAdd` (K6), `frameDiffLuma` (K7). The *kernels* are no longer a backlog — see 13.2 for the residual **wiring** gap. |
| **Consolidated `MediaUtils`** (spec §4) | ✅ exists | `Media/MediaUtils.swift` is the GPU-first facade; geometry helpers (`smartResize`, `aspectPreservingSize`, video token math) live there. |
| **In-core media decode** (spec §5, roadmap #7) | ✅ DONE | `VideoPreprocessing.decodeVideoFrames` (AVAssetImageGenerator, timestamps), image/audio decode in core — no longer test-only. |
| **Registry consolidation** (session-plan I.2) | ✅ files done | `Loader/ModelRegistry.swift` + `ModelRegistryTable.swift` canonical; old `VADModelRegistry.swift` / `AudioModelRegistry.swift` removed. (CLI rename to `ffai model list` still pending — see 13.3.) |

### 13.2 Genuinely REMAINING functional gaps (verified)

| # | Gap | Verified state | Spec ref |
|---|---|---|---|
| **M1** | **`smart_resize` rollout incomplete** — only **3 towers** wired (`Qwen2Vision`, `Gemma4Vision`, `LFM2Vision` call `smartResize`). **Qwen2.5/3/3.5-VL, Pixtral, Mistral3, GLM-OCR still force fixed-square** → aspect distortion + token truncation (quality loss, not crash) | grep: `smartResize` only in those 3 vision files | §3, roadmap #4 (partial) |
| **M2** | **Tile/slice splitting — NONE.** MiniCPM-V, Idefics3, SmolVLM2, LFM2-VL run single-tile (lose detail on large/non-square); FastVLM has no letterbox pad | grep: no `tileSplit`/`do_image_splitting` anywhere | §3, roadmap #5 |
| **M3** | **No unified multimodal `ModelInput` case.** Image/video/omni reach inference only via `model.vlModel.generate(...)`; `ModelInput` (ModelIO.swift) covers text/transcribe/speak/transform/detect but **not vision/omni**. Each VLM caller hand-builds placeholder framing (see §8.1). Combined image+video+audio in one call (spec §8) also not surfaced — the 4 `VisionModel.generate` overloads each take one modality | VisionModel.swift:369/398/442/499 | §8, §2.2 |
| **M4** | **GPU-preprocessing wiring incomplete** — kernels exist (13.1) but some models still CPU: Paligemma patch-embed (`concurrentPerform`, §4.2 C3), STT mel front-ends Parakeet/SenseVoice/Cohere (§4.2 C5), VisionEncoderLayer d32 `cpuAttention` (§4.2 C4) | §4.2 of this doc | §12, roadmap #8 (partial) |
| **M5** | **Streaming audio INPUT — none.** Input is always a full `[Float]` waveform; no chunked/real-time ingest (sliding-window encoder + cached state) | — | §7b, roadmap #11 |
| **M6** | **Live camera/mic capture — none.** No `AVCaptureSession`/`AVCaptureDevice` anywhere | grep: zero hits | §6, roadmap #11 |
| **M7** | **Real-time video loop — none.** No KV-continuation + motion-gated frame loop; the `frameDiffLuma` (K7) kernel exists but is unwired into a streaming pipeline | — | §7c, roadmap #12 |
| **M8** | **CLI raw-input `ffai model …` namespace** — per-modality subcommands + streaming-by-default exist (`SpeakCommand`/`TranscribeCommand`/`GenerateMediaCommands`), but the catalog is still `ffai models`/`ffai download`, not the `ffai model list/add/remove/search` namespace | CLI file list | §9, session-plan I.1/I.2/I.3–I.6 |

**Net:** the streaming + kernel + decode foundations are done; the remaining
multimodal work is **per-model processor fidelity** (M1 smart_resize rollout,
M2 tiling — both quality, not crashes), **finishing the GPU-preprocess wiring**
(M4), the **unified multimodal input surface** (M3), and the **real-time/live**
features (M5–M7) which are net-new. M1 is the highest-impact correctness item
(7 VLMs silently distort non-square images today).

---

## 14. Plan-doc reconciliation (Q vs three named plans)

Verified [`model-registry-and-inheritance-plan.md`](model-registry-and-inheritance-plan.md),
[`qwen3-omni-30b-port-plan.md`](qwen3-omni-30b-port-plan.md), and
[`gemma4-multimodal-plan.md`](gemma4-multimodal-plan.md) against code on
`ek/fix-all-the-things` (2026-06-05).

### 14.1 Registry / inheritance plan — Phases 0–4 DONE, Phase 5 PARTIAL, one functional deviation

| Phase | Status | Evidence |
|---|---|---|
| 0 — pure moves | ✅ | `Loader/ModelRegistry.swift`, `FFAI/Model.swift` |
| 1 — `ModelKind` + data-driven dispatch | ✅ | `ModelKind.swift`; `ModelEntry` table in `ModelRegistryTable.swift` (find-first-match replaces the ~75 branches) |
| 2 — single catalog → CLI | ✅ | `CatalogEntry`/`CatalogGroup` deleted from `ModelsCommand.swift` (0 refs); CLI renders from `ModelRegistry.catalog` |
| 3 — engine protocol layer | ✅ | `ModelType/` (`ModelEngine` + `TextEngine`/`Transcribing`/`Synthesizing`/`SpeechTransforming`/`Detecting`); `ModelInput`/`ModelOutput` sum types |
| 4 — fold audio + VAD | ✅ | `audioEntries` in the table; `LoadedAudioModel`/`LoadedVADModel` **deleted** (only comment refs remain in `AudioStreaming.swift`) |
| 5 — test-layout reconciliation + docs | ⚠️ **PARTIAL** | Docs done (`architecture.md` + `documentation/developing/adding-a-model.md` both reference `ModelEntry`/`ModelKind`/`ModelEngine`). **Outstanding: the `Tests/`-mirrors-`Sources/` sweep + dead-test removal** — overlaps directly with §8 (audio integration tests still deviate from the gold standard) |

**Deviations to review** (most are documented/intentional; one is a real functional gap):

- **[Functional gap — decide] No `.imageText` / `.video` case in `ModelInput`.** The Phase 3 API sketch (plan lines 200–209) proposed `.imageText` + `.video`; the shipped `ModelInput` (`ModelType/ModelIO.swift`) is `.text/.transcribe/.speak/.transform/.detect` only. Vision/omni multimodal is reached via `model.vlModel.generate(...)`, **not** the unified sum type — so `Model.generate(_:)` cannot service an image/video/omni request. (Same as §13 **M3**.) The plan's `.audio` also shipped as `.transcribe`. **This is the one substantive deviation worth a decision** — add the cases, or formally bless `vlModel` as the multimodal surface.
- **[Intentional — T7] `ModelEngine` is a bare marker** (no structural requirements), vs the target-arch diagram's `ModelEngine: Module { capabilities; dtype }`. Capabilities moved to `Model` (`availableCapabilities` + runtime `enable`/`disable`), not the engine. This is what lets the 30+ heterogeneous audio/VAD classes conform for free — documented in resolved-decision T7.
- **[Intentional — T6] `ModelOutput` / `ModelOutputChunk`** names (the plan's `GenerationResult`/`GenerationChunk` collided with existing text structs).
- Typed convenience `generateText` was proposed; text actually uses the existing `generate(messages:)`. Minor.

Net: the core goal — one `Model.load`, one registry, one `any ModelEngine` existential — **is achieved**. Residuals = Phase 5's test sweep + the missing multimodal `ModelInput` cases.

### 14.2 Qwen3-Omni-30B port — NOT wired (milestones 1–4 outstanding)

Only the quant (the plan's "Open questions / risk") was started — there is a local
4-bit conversion. **None of the code milestones exist:**

| Milestone | Artifact | State |
|---|---|---|
| 1 — audio encoder | `Models/Audio/Omni/Qwen3OmniAudio.swift` (conv2d stem + 32× `AudioEncoderLayer`) | **ABSENT** |
| 2 — family entry | `Models/Qwen3Omni.swift` (text via `Qwen35Hybrid` + Qwen3-VL tower + audio) | **ABSENT** |
| 3 — registry route | a `Qwen3Omni` entry ahead of the generic path | **ABSENT** |
| 4 — integration test | `Tests/ModelIntegrationTests/Omni/Qwen3OmniIntegrationTests.swift` | **ABSENT** |

**⚠️ Routing hazard:** `qwen3_omni` / `qwen3_omni_moe` / `Qwen3OmniMoeForConditionalGeneration`
are **already claimed by the legacy `QwenOmniModel`** (`Qwen3.swift:71-87`, `isOmni(_:)`).
The plan explicitly says that engine does **not** fit the 30B — it's a conv1d Whisper
stem + Qwen3**Dense**, with no conv2d stem, no MoE backbone, no vision tower. So loading
the quantized 30B-A3B omni checkpoint **today routes to the wrong handler** → expect a
tensor-name/shape failure or incoherent output, not a clean "unsupported." Milestones 2/3
specifically call for a fresh `Qwen3Omni` routed **before** the legacy claim.

→ **Action:** build milestones 1–4 (the plan has the verified reuse map + tensor names +
audio-encoder algorithm). If instead the intent is for the legacy `QwenOmniModel` to
absorb the 30B, that needs an explicit decision + the conv2d/MoE/vision wiring added to it
— today it silently claims a checkpoint it can't load.

### 14.3 Gemma 4 multimodal — built + integrated; tests need consolidation

The hard, novel piece (Conformer/USM audio encoder) is implemented + numerically validated,
the 3-way splice + `.audioIn`/`.videoIn` capabilities landed, and the mel front-end is now
calibrated (a `mel-front-end-vs-feature-extractor` test exists; session-plan notes it
"bit-exact"). Integration tests cover every modality — but are **spread across 4 files with
inconsistent load patterns**, which is the consolidation requested:

| File | Checkpoint | Tests | Load pattern | Issue |
|---|---|---|---|---|
| `Omni/Gemma4OmniIntegrationTests.swift` (216 L) | omni (`gemma4_unified`) | load, text, encode-shapes, image, audio, video | ✅ load-once static `Task` (gold-standard shape) | inline `sine` (§8.2); `encode shapes` is a sub-stage test |
| `Vision/Gemma4VisionIntegrationTests.swift` (255 L) | `gemma-4-e2b-it-4bit` | load, enable/disable, image, **vision-tower-vs-oracle**, videoIn, video | ❌ each `@Test` re-loads | numeric oracle parity → belongs in `FFAITests` |
| `Vision/Gemma4MultimodalIntegrationTests.swift` (294 L) | `gemma-4-e2b-it-4bit` **(same)** | load, **audio-encode shapes**, **audio-encoder-vs-oracle**, **mel-vs-extractor**, audio | ❌ each `@Test` re-loads | 3 numeric oracle/sub-stage tests; inline sine |
| `Text/Gemma4TextIntegrationTests.swift` (178 L) | E2B + 31B Dense + 26B-A4B MoE | text-gen per size | per-checkpoint | legitimately multi-checkpoint — keep |

**Consolidation (→ MiniCPM45Omni shape):** merge `Gemma4VisionIntegrationTests` +
`Gemma4MultimodalIntegrationTests` (both load the **same** `gemma-4-e2b-it-4bit`) into ONE
suite that loads the e2b checkpoint **once** (static memoized `Task`) and runs
`load → text → image → audio → video` coherence `@Test`s. Move the three numeric oracle
tests (vision-tower, audio-encoder, mel-front-end) **down to `FFAITests`** — they are
component-correctness tests, not integration coherence (§8 gold standard). Keep
`Gemma4OmniIntegrationTests` as the separate **omni-checkpoint** suite (it already follows
the pattern — just hoist its inline `sine` per §8.2). Keep `Gemma4TextIntegrationTests`
for the 31B/26B size matrix (optionally fold its E2B-text case into the consolidated suite).

**Plan residual:** audio/video coherence on the larger sizes (E4B/26B/31B) isn't separately
tested — only their text path is (in `Gemma4TextIntegrationTests`).

> If you want, I can do the §14.3 Gemma 4 test consolidation + the §8.2 `sine` move as a
> follow-up change (it's a contained test-only refactor); say the word.
