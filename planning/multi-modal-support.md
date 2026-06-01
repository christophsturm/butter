# Multi-Modal Support — design + roadmap

> Scope: how FFAI ingests and runs **image / video / audio** across the
> VL / audio model families — resolution handling, media decode, the
> multi-modal `generate` surface, live capture, and streaming (in + out).
> Grounded in the current code (2026-05-31). Companion to
> [`gemma4-multimodal-plan.md`](gemma4-multimodal-plan.md) (Gemma 4 specifics,
> now done end-to-end) and [`architecture.md`](architecture.md).

> Filename note: requested as `mulit-modal-support.md`; created with the
> corrected spelling `multi-modal-support.md`.

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
- **VLM is greedy-only** — no sampling/temperature on the multi-modal path.
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
2. **On-the-fly preprocessing.** Resize/normalize/patchify as part of inference;
   it's negligible vs the forward pass. No GPU resize / no pre-baking needed.
3. **AVFoundation over ffmpeg (for now).** FFAI targets Apple Silicon;
   AVFoundation is already used for audio, is hardware-accelerated, and avoids a
   heavy native dep. Revisit ffmpeg only for formats AVFoundation can't decode
   (MKV, some AVI codecs) or true cross-platform (Linux).
4. **Source-level, tested helpers.** Preprocessing/decode live in `Sources/FFAI`
   with their own unit tests; the *same* helpers are used by the models AND by
   the integration tests (no test-only decode path that drifts from production).

---

## 3. Per-model resolution / processor AUDIT (to produce)

Build a table — one row per VL + audio model — and document it (in
`documentation/` + a capability table). This surfaces every model silently
distorting input and feeds the SmolVLM/Qwen rollout.

| Model | Modality | Reference processor | Native sizes | FFAI today | Faithful? |
|---|---|---|---|---|---|
| Gemma 4 VL | image/video | aspect-ratio variable (≤ maxPatches, ×48) | any | variable ✅ | ✅ |
| Gemma 4 audio | audio | 16 kHz mel, USM Conformer | 16 kHz | ✅ | ✅ |
| Qwen2-VL | image/video | **native resolution** (smart-resize) | any | **fixed square** | ❌ distorts |
| Qwen2.5-VL | image/video | native resolution + windows | any | **fixed square** | ❌ distorts |
| Qwen3-VL | image/video | native resolution | any | **fixed square** | ❌ distorts |
| SmolVLM2 | image | SigLIP fixed + image splitting | 384 tiles | fixed square | ⚠️ verify tiling |
| Idefics3 | image | SigLIP 364 + splitting | 364 tiles | fixed square | ⚠️ verify tiling |
| PaliGemma | image | SigLIP fixed 224/448 | fixed | fixed square | likely ✅ |
| Pixtral | image | native-ish | any | fixed square | ❌ verify |
| MiniCPM-V | image | tiling / slices | tiles | fixed square | ⚠️ |
| Mistral3 | image | Pixtral-style | any | fixed square | ⚠️ |
| FastVLM | image | fixed | fixed | fixed square | likely ✅ |
| GLM-OCR | image | fixed 336 | fixed | fixed square | likely ✅ |
| Whisper / Parakeet / … | audio | 16 kHz log-mel | 16 kHz | ✅ (verify mel) | ⚠️ |

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
Continuous frames (file or camera) → VLM → streaming text. The hard problems:
- **Context growth**: re-encoding all frames every step is O(n²). Need a
  sliding window of recent frames + KV-cache continuation (don't re-prefill).
- **Frame-rate decoupling**: capture at 30 fps, sample to model fps, drop
  backpressure frames.
- **Temporal prompt management**: timestamps + frame markers maintained as the
  window slides.

Depends on 7a (+ the KV-cache/spec-decode work already in the plan).

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
2. **Qwen2-VL native-resolution processor** (§3/§4) — highest-impact faithful
   fix (3 models). Same `preprocessImage`-hook pattern as Gemma 4.
3. **Combined multi-modal `generate`** (§8) — small; unlocks audio+video.
4. **Core `MediaDecoding`** (§5) — mp4/mov + audio + frame-sampling-with-timestamps.
5. **VLM streaming output** (§7a) — needs the engine embedding-prefill API.
6. **CLI raw-input commands** (§9).
7. **Streaming audio input** (§7b) + **camera source** (§6).
8. **Real-time video loop** (§7c) — depends on 5 + KV-cache work.
9. *(Later)* ffmpeg/VLCKit backend for AVI/MKV if demanded.

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
