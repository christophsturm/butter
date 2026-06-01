# Multi-Modal Support — design + roadmap

> Scope: how FFAI ingests and runs **image / video / audio** across the
> VL / audio model families — resolution handling, media decode, the
> multi-modal `generate` surface, live capture, and streaming (in + out).
> Grounded in the current code (2026-05-31). Companion to
> [`gemma4-multimodal-plan.md`](gemma4-multimodal-plan.md) (Gemma 4 specifics,
> now done end-to-end) and [`architecture.md`](architecture.md).

---

## 1. Current state (what works / what doesn't)

### Works
- **Gemma 4 all three modalities** end-to-end (image / audio / video → text),
  including **variable-resolution** images (aspect-ratio preserving).
- **Text-only streaming output** — `Generate.generateStream(...)` yields
  `GenerationChunk`s over an `AsyncStream` (the buffered path is built on it).
- **Audio file decode + resample** — `AudioPreprocessing.loadWaveform(url:targetRate:)`
  uses **AVFoundation** (`AVAudioFile`); any OS-decodable container → mono
  `[Float]` at the target rate. Plus an in-memory linear `resample`.
- **Image preprocess** — `ImagePreprocessing.preprocess` (CPU bilinear resize
  → normalize → NCHW) + `resize`.
- **On-the-fly preprocessing is cheap** — a bilinear resize is sub-millisecond
  vs hundreds-of-ms–seconds of tower+LLM inference. No separate fast path needed.

### Does NOT work yet (the gaps this doc plans)
- **VLM streaming output** — `VisionModel.generate(...)` and `runGreedy` return
  buffered `[Int]`; the VLM path does NOT route through `Generate`'s streaming.
  (Blocked on the engine exposing a public *embedding-prefill* entry point — see
  the TODO at `VisionModel.swift` `generate` doc.)
- **VLM is greedy-only — and IGNORES generation parameters (a correctness
  bug, not just a missing feature).** `runGreedy` decodes via `Ops.argmax`
  every step (`VisionModel.swift:~485`), so EVERY VLM / Gemma 4 multimodal
  generation (image / audio / video → text) discards the model-card defaults
  (temperature / top-p / top-k / seed) that the text path respects. Models
  should respect their `defaultGenerationParameters` by default, including on
  the multimodal path. This is fixed by routing the VLM decode through
  `Generate` (§7a) — the same change that unlocks streaming.
- **Multiple modalities per call** — `spliceMultimodal` / `runGreedy` already
  accept image+video+audio tokens *simultaneously*, but the three public
  `generate(...)` methods each pass only ONE (the other two `nil`). So no
  single call combining e.g. **audio + video**.
- **Streaming audio INPUT** — input is always a full `[Float]` waveform; no
  chunked/real-time audio ingest.
- **Video-file decode in CORE** — there is NONE. Frame decode lives only in the
  TEST helper (`VisionTestHelpers.catVideoFrames` via `AVAssetImageGenerator`).
  Callers must hand `VisionModel` an `[RGBImage]` themselves.
- **Live camera capture** — no `AVCaptureSession` anywhere.
- **Per-model faithful resolution** — ONLY Gemma 4 does the real
  (variable-resolution) processor. Every other VL model — *including
  natively-variable ones like Qwen2/2.5/3-VL* — is force-resized to a fixed
  square (`gridSide·patchSize`), distorting aspect ratio (quality loss, not a
  crash). We do NOT error on "wrong" sizes; we always resize (matching how the
  reference processors behave).

---

## 2. Design principles

1. **Auto-resize, don't reject.** transformers / mlx-vlm processors accept any
   input size and resize/pad/tile internally. FFAI matches this — the user
   never pre-sizes, and we never throw "unsupported size". The fix for quality
   is implementing each model's *real* processor, not adding size errors.
2. **On-the-fly preprocessing — but it must still be optimized.** Doing
   resize/normalize/patchify as part of inference is the right shape (no
   pre-baking). For *single-shot* inference the cost is dwarfed by the forward
   pass, so it's not a latency problem TODAY. BUT the current helpers are
   **single-threaded CPU and unoptimized** (`ImagePreprocessing.resize` is a
   scalar triple `for`-loop; the Gemma 4 patch-unfold / pool / embedder and the
   audio magnitude-mel front-end are CPU `toFloatArray` loops / a CPU DFT). At
   **real-time / streaming video fps** these CPU passes become a per-frame tax
   on the hot path. So: parallelize (`DispatchQueue.concurrentPerform`) at
   minimum, and **move resize/normalize/patchify + the mel front-end onto the
   GPU** (an `Ops.resizeBilinear` / GPU normalize, a magnitude-mel kernel) so
   the whole pre→tower→LLM path is GPU-resident. See §12.
3. **AVFoundation over ffmpeg (for now).** FFAI targets Apple Silicon;
   AVFoundation is already used for audio, is hardware-accelerated, and avoids a
   heavy native dep. Revisit ffmpeg only for formats AVFoundation can't decode
   (MKV, some AVI codecs) or true cross-platform (Linux).
4. **Source-level, tested helpers.** Preprocessing/decode live in `Sources/FFAI`
   with their own unit tests; the *same* helpers are used by the models AND by
   the integration tests (no test-only decode path that drifts from production).

---

## 3. Per-model resolution / processor AUDIT (results, 2026-05-31)

Audited every VL/audio model's input handling against the mlx-vlm reference.
**Only Gemma 4 overrides `preprocessImage`** — every other VLM routes through
the shared `VisionModel.encodeImage` → generic **fixed-square**
`preprocess(targetW: imageSize, targetH: imageSize)` (`VisionModel.swift:189-197`).
There is **zero** smart-resize / tiling / pan-and-scan anywhere in
`Sources/FFAI/Vision/`. Worst offenders first:

| Model | Modality | Reference processor | Native | Patch | Token count | FFAI today | Verdict |
|---|---|---|---|---|---|---|---|
| **Qwen2-VL** | img+vid | native `smart_resize` (min/max-pixels, aspect) | any | 14, merge 2 | variable | fixed 28×28→392² (`Qwen2Vision.swift:334` "fixed-grid for now") | ❌ distorts+truncates |
| **Qwen2.5-VL** | img+vid | native `smart_resize` | any | 14, merge 2 | variable | fixed 28×28→392² (`Qwen25Vision.swift:446,545`) | ❌ distorts+truncates |
| **Qwen3-VL** | img+vid | native `smart_resize` | any | 16, merge 2 | variable | fixed 32×32² (`Qwen3Vision.swift:342`) | ❌ distorts+truncates |
| **Qwen3.5-VL** | img+vid | native `smart_resize` | any | 16, merge 2 | variable | fixed grid (Qwen3 tower, `Qwen3xVision.swift:147`) | ❌ distorts+truncates |
| **Pixtral** | image | native aspect (longest-edge cap) | ≤1024 | 16 | variable | fixed 336² (`PixtralVision.swift:104`) | ❌ distorts+truncates |
| **Mistral3** | image | native aspect (Pixtral-style) | ≤1540 | 16, merge 2 | variable | fixed `imageSize²` (`Mistral3Vision.swift:329,341`) | ❌ distorts+truncates |
| **FastVLM** | image | pad-to-square (letterbox) then resize | 1024 | conv ÷4 | fixed | plain square resize, no pad (`FastVLMVision.swift:49`) | ❌ mild (lost aspect) |
| **MiniCPM-V** | img+vid | tile/slice split (`scale_res 448` + grid) | 448/slice | 14 | post-merge | single square, no slicing (`MiniCPMVVision.swift:648,685`) | ⚠️ incomplete |
| **Idefics3** | image | tile split (`do_image_splitting` + thumbnail) | 364/tile | 14, shuffle 2 | 169/tile × N | single 364², no split (`Idefics3Vision.swift:51`) | ⚠️ incomplete |
| **SmolVLM2** | img+vid | tile split + thumbnail | 512/tile | 16, shuffle 4 | 64/tile × N | single 512², no split (`SmolVLM2Vision.swift:54`) | ⚠️ incomplete |
| **LFM2-VL** | image | tile split (`tile 512`, ≤256 tokens) | 512/tile | 16, unshuffle | ≤256 pooled | single square, no tiling (`LFM2Vision.swift:62`) | ⚠️ incomplete |
| **GLM-OCR** | image | native `smart_resize` (Qwen family) | any | 16, merge 2 | variable | aspect-preserving at encode (`GlmOcrVision.swift:745`); but **caller must pre-resize** | ✅ encode-faithful (⚠️ no in-tree smart_resize) |
| **PaliGemma** | image | fixed-square SigLIP | 224/448/896 | 14 | fixed | fixed square (`PaligemmaVision.swift:733`) | ✅ faithful |
| **NemotronH** | image | fixed-square SigLIP | config | config | fixed | fixed square (`NemotronHVision.swift:54`) | ✅ faithful |
| **Gemma 4 VL** | img+vid | aspect-ratio variable | any | 16 | variable | overrides `preprocessImage` → `[1,3,H,W]` | ✅ faithful (reference row) |

**Audio** — all 16 kHz, standard log-mel/FBANK, nothing unusual: Whisper (80/128
bins), Parakeet (NeMo 128), Qwen3-ASR (128), SenseVoice (Kaldi FBANK→LFR 80),
GLM-ASR (128), Gemma 4 audio (USM, done). No action.

**Net:** **7 distort** (Qwen ×4 + Pixtral + Mistral3 force native/aspect into a
square = highest impact; FastVLM mild), **4 incomplete** (tile/slice models run
single-tile → lose detail on large/non-square images), **4 faithful**
(PaliGemma, NemotronH, GLM-OCR, Gemma 4). Audio uniformly fine.

### Faithful-fix per family
- **Qwen2/2.5/3/3.5-VL** — replace the `defaultGridSide` constant with
  `smart_resize` (round H,W to `patch·merge` under min/max-pixel bounds) →
  feed the native grid into the already-variable `encode(frames:)` temporal
  unfold; facade must report the *runtime* merged token count.
- **Pixtral / Mistral3** — override `preprocessImage`: aspect-preserving resize
  to longest-edge cap (336 / 1540) rounded to `patch·merge`, emit `[1,3,H,W]`;
  thread `image_sizes` through (encoder already has a per-image `numPatches`).
- **MiniCPM-V / Idefics3 / SmolVLM2 / LFM2-VL** — implement
  `do_image_splitting`/slicing: cut into tiles + a global thumbnail, run the
  existing single-tile encoder per tile, concat tokens (pixel-shuffle is already
  per-tile correct). Today they emit exactly one tile's worth.
- **FastVLM** — add letterbox pad-to-square (`expand2square`) before the resize
  via a `preprocessImage` override. Cheap.
- **GLM-OCR** — add an in-tree `smart_resize` so the path is self-contained
  (encode is already aspect-correct).
- **PaliGemma / NemotronH / Gemma 4** — no change.

The native-resolution `smart_resize` (Qwen + GLM-OCR + Pixtral/Mistral3) and the
tile-splitting (4 models) are the two reusable helpers to build (§4) — each
covers a whole family. Highest-impact first build: **Qwen `smart_resize`** (4
models) using the Gemma 4 `preprocessImage`-hook pattern.

*(Cells are the starting hypothesis; the audit confirms each against the
reference image/video/feature processor.)* Highest-impact fix after Gemma 4:
**Qwen2-VL native-resolution** processor (3 models share it).

---

## 4. Source-level helpers (build + test)

Centralize in `Sources/FFAI` with unit tests; reuse from models + integration
tests:
- `ImagePreprocessing`: keep `resize` (bilinear) + `preprocess`; add the
  per-model processor variants the audit demands (aspect-ratio-preserving /
  letterbox-pad / tile-split / native-resolution smart-resize). Each gets a
  geometry unit test like `Gemma4VisionGeometryTests` (pure, fast, no model).
- A small **`MediaDecoding`** module (AVFoundation) — see §5.
- Integration tests per modality per model use the real helpers (the Gemma 4
  image/audio/video integration tests are the template).

---

## 5. Media decode in core (AVFoundation) — files + formats

Add an AVFoundation-backed decode API to **core** (not just tests):
- `MediaDecoding.image(url:) -> RGBImage` (still images: png/jpeg/heic/…).
- `MediaDecoding.audio(url:targetRate:) -> [Float]` (wrap existing
  `loadWaveform`).
- `MediaDecoding.videoFrames(url:fps:maxFrames:) -> [(RGBImage, timestampSec)]`
  via `AVAssetImageGenerator` (sampling) or `AVAssetReader` (full decode). Must
  also surface **timestamps** (Gemma 4 video prompts need `mm:ss`).

### Container/codec support matrix (AVFoundation, Apple Silicon)
| Format | Decode | Notes |
|---|---|---|
| `.mp4` / `.mov` / `.m4v` (H.264/HEVC) | ✅ native | hardware-accelerated |
| `.m4a` / `.wav` / `.mp3` / `.aac` (audio) | ✅ native | via `AVAudioFile` |
| `.avi` | ⚠️ codec-dependent | many AVI codecs unsupported by AVFoundation |
| `.mkv` (Matroska) | ❌ not supported | AVFoundation has no MKV demuxer |

**For AVI/MKV** we either (a) document "remux/transcode to mp4 first"
(`AVAssetExportSession` can remux some; users can `ffmpeg -c copy`), or (b) add
an **optional ffmpeg/VLCKit backend** behind a protocol so the core stays
AVFoundation-only. Recommendation: ship AVFoundation (mp4/mov/m4v + common
audio) first; add the ffmpeg backend later only if AVI/MKV demand is real.

---

## 6. Live camera / capture

For "enable the Mac camera, stream a source, model handles it":
- `AVCaptureSession` + `AVCaptureDevice` (camera) / `AVAudioEngine` (mic) →
  `AVCaptureVideoDataOutput` delivers `CMSampleBuffer` frames in real time.
- A `CameraSource` (and `MicSource`) that yields `(RGBImage, timestamp)` /
  audio chunks over an `AsyncStream`, with frame-rate throttling (sample to the
  model's target fps, e.g. 1–2 fps for a VLM — you can't run the tower at 30 fps).
- Permissions (`NSCameraUsageDescription` / `NSMicrophoneUsageDescription`).

This is gated on **streaming inference** (§7) — capturing frames is easy;
*running the model continuously* on a growing stream is the hard part.

---

## 7. Streaming (in + out) — the big workstream

Three independent pieces, in dependency order:

### 7a. VLM streaming OUTPUT (smallest, do first)
Route the VLM decode through `Generate`'s streaming instead of the bespoke
`runGreedy`. Requires the text engine to expose a **public embedding-prefill
entry point** (prefill from a spliced `[nTokens, hidden]` stream, return the
KV-state) so `Generate.generateStream` can take over the decode loop. Also lets
VLMs reuse **sampling** (temperature/top-p) instead of greedy-only. Unblocks
streaming captions/transcripts.

### 7b. Streaming audio INPUT
Chunked waveform ingest → incremental mel + encoder. For Conformer/USM
(Gemma 4) and Whisper-style encoders this means a sliding window + cached
encoder state; for true real-time STT, the `voxtral_realtime` family is the
purpose-built target. Output streams via 7a.

### 7c. Streaming VIDEO + real-time loop (largest)
Continuous frames (file or camera) → VLM → streaming text.

**Avoiding O(n²) — the key technique is KV-cache continuation, NOT
re-prefilling.** The naive loop re-encodes the whole frame history each step →
O(n²). Instead, treat the video as one growing token stream:
- Encode each **new** frame's patches through the vision tower **once**, append
  its soft tokens to the running KV cache, and decode forward. Old frames are
  never re-encoded and their attention K/V are already cached → **per-frame cost
  is O(window)**, not O(history).
- A **sliding window** evicts the oldest frames' tokens (rotating-eviction KV
  cache — the same machinery the text sliding-window models use) so the context
  + per-step cost stay bounded no matter how long the stream runs.
- So the steady-state cost per processed frame ≈ one tower encode + a bounded
  attention over the window. The expensive, unavoidable part is **the vision
  tower encode per processed frame**.

**Can we beat 1–2 fps? Yes — but not by running the tower at 30 fps.** The tower
encode is the floor; you don't need it on every frame. Strategy (this is what I
would do, and your instinct is right):
1. **Capture at 30 fps, process adaptively at ~1–5 fps.** Buffer the camera at
   native fps; sample frames for the tower. A model understands a scene from a
   few fps — 30 fps of *tower* work is wasted.
2. **Drop redundant frames by motion / scene-change.** Cheap frame-diff (downscaled
   luma SAD) → only encode keyframes where the scene changed; hold the last
   tokens otherwise. Static scenes cost ~0; action scenes get more frames.
3. **Batch frames through the tower.** Multiple sampled frames in one set of
   GEMMs/attention dispatches amortizes launch overhead (the vision attention is
   already GPU after the migration).
4. **Shrink the per-frame tower cost** — smaller frame resolution (fewer
   patches), and GPU-optimize the still-CPU tower stages (Gemma 4 patch-unfold /
   pool / embedder) per §12.
5. **Backpressure**: if the tower can't keep up, drop frames (never queue
   unboundedly) so latency stays flat.

Realistic target: real-time *interaction* at 30 fps **capture** with **2–8 fps
effective tower throughput** (adaptive, motion-gated) — which is plenty for
"describe what the camera sees" while staying real-time. True 30 fps *tower*
processing is neither needed nor achievable for a real VLM tower; the win is
smart sampling + KV continuation + a GPU-resident pipeline.

- **Temporal prompt management**: timestamps + frame markers maintained as the
  window slides.

Depends on 7a (+ the rotating-eviction KV cache, already in the plan).

---

## 8. Multi-modal combined `generate` (audio + video + image together)

Small additive change: a `generate(promptTokens:image:videoFrames:audio:...)`
(or a `MultiModalInput` struct) that encodes each present modality and threads
all three token sets into the existing `runGreedy(imageTokens:videoTokens:
audioTokens:)` — which already supports it. The caller interleaves the spans
(`boa…eoa` for audio, `{ts} boi…eoi` per video frame, `boi…eoi` for image) in
prompt order. One forward, one text output grounded on everything. Add a
Gemma 4 audio+video integration test.

---

## 9. CLI (raw input → auto-preprocess → inference)

A `ffai` surface that takes a **raw file path or a live source** and does
decode → model-correct preprocess → inference internally, so users never
pre-size:
- `ffai generate --image dog.jpeg --prompt "..."`
- `ffai generate --video clip.mp4 --prompt "..." [--fps 1 --max-frames 8]`
- `ffai generate --audio speech.wav --prompt "Transcribe this."`
- `ffai generate --camera --prompt "Describe what you see" --stream` (needs §6+§7)
Decode via the §5 core `MediaDecoding`; preprocess via the model's processor;
stream output once §7a lands.

---

## 10. Prioritized roadmap

1. **Resolution/processor AUDIT + docs** (§3) — cheap, high value, prevents
   silent quality loss; informs the Qwen rollout.
2. **VLM respects generation params** (§7a / §1) — route `runGreedy` through
   `Generate` so the multimodal path samples per `defaultGenerationParameters`
   instead of always-argmax. Correctness fix; also the prerequisite for VLM
   streaming output. Do early.
3. **`smart_resize` native-resolution helper** (§3/§4) — highest-impact: one
   helper fixes **Qwen2/2.5/3/3.5-VL** (4 models, currently force-squared) and
   self-contains **GLM-OCR**; the aspect-cap variant fixes **Pixtral + Mistral3**.
   Same `preprocessImage`-hook pattern as Gemma 4.
4. **Tile/slice-splitting helper** (§3/§4) — fixes the 4 incomplete tile models
   (MiniCPM-V, Idefics3, SmolVLM2, LFM2-VL) — cut into tiles + global thumbnail,
   encode per tile, concat. Plus FastVLM letterbox-pad (cheap).
5. **Combined multi-modal `generate`** (§8) — small; unlocks audio+video.
6. **Core `MediaDecoding`** (§5) — mp4/mov + audio + frame-sampling-with-timestamps.
7. **GPU + parallel preprocessing** (§12) — needed before real-time fps.
8. **VLM streaming output** (§7a) — builds on #2.
9. **CLI raw-input commands** (§9).
10. **Streaming audio input** (§7b) + **camera source** (§6).
11. **Real-time video loop** (§7c) — depends on #8 + rotating-eviction KV cache.
12. *(Later)* ffmpeg/VLCKit backend for AVI/MKV if demanded.

---

## 12. Preprocessing performance (GPU + parallelize)

The preprocessing/front-end helpers are currently **single-threaded CPU and
unoptimized**. Fine for single-shot latency, but a per-frame tax for real-time
video and just wasteful. To make the whole pre→tower→LLM path GPU-resident:
- **`ImagePreprocessing.resize`** — scalar triple loop (`ImagePreprocessing.swift`
  `resize`). Parallelize now (`concurrentPerform`); GPU later (an
  `Ops.resizeBilinear` sampling kernel, or `MPSImageScale`).
- **normalize + NCHW pack** — CPU loop in `preprocess`; fold into a GPU
  normalize/transpose (or the resize kernel's output).
- **Gemma 4 vision tower CPU stages** — patch-unfold, attention-pooling, and the
  multimodal-embedder RMSNorm still run on CPU via `toFloatArray` loops
  (`Gemma4Vision.swift`). GPU these (unfold → a gather/copy kernel; pool → an
  avg-pool kernel; embed RMSNorm → `Ops.rmsNormRows`).
- **Gemma 4 audio magnitude-mel front-end** — a CPU DFT (`Gemma4AudioFrontend.swift`).
  Add a **magnitude-mel GPU kernel** (the existing `Ops.melSpectrogram` is
  power-only); big win for long clips and streaming audio.
- **AVFoundation decode** is already hardware-accelerated; the CPU cost there is
  only the `CMSampleBuffer`/`CGImage` → `RGBImage` copy — keep it tight (avoid
  extra allocations per frame in the streaming loop).

Each gets a correctness test (GPU vs the CPU reference) like the kernel work
already landed. Sequenced before the real-time video loop (#10).

---

## 11. Open questions
- KV-cache continuation for the VLM decode loop — reuse the text path's cache,
  or a VLM-specific one for the sliding video window?
- Sampling on the VLM path — fold into 7a or keep greedy for determinism in
  tests?
- Per-model processor abstraction — extend the `preprocessImage` hook into a
  fuller `ImageProcessor` protocol (resize mode + tiling + token-count formula)
  so the audit's findings drop in cleanly?
- AVI/MKV: document-only ("transcode first") vs an optional decode backend?
