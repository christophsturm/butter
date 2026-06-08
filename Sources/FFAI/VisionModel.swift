// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// VisionModel — vision-language model composition.
//
// A VLM is a vision encoder + a text backbone joined by a cross-modal
// token splice: the prompt contains image-placeholder tokens, the vision
// encoder turns an image into a run of patch-token embeddings, and the
// splice replaces the placeholder positions in the text embedding stream
// with those vision tokens. The fused stream then flows through the text
// backbone exactly as a text-only prompt would.
//
// This file is the family-agnostic core: the splice + a self-contained
// greedy generate. Each VL family file (Gemma3VL, Qwen25VL, …) builds a
// `VisionModel` from its checkpoint — a `VisionEncoder` (.5
// `VisionEncoder.swift`) plus the already-shipped text engine — and
// declares its image-placeholder token id.
//
// The text backbone is driven through `TextGenerating.forward(
// inputEmbedding:...)` — the embedding-input primitive the VL-target
// families implement. Decode after prefill is ordinary token-id forward.

import Foundation
import Metal

public enum VisionModelError: Error, CustomStringConvertible {
    case engineLacksEmbeddingInput(String)
    case placeholderCountMismatch(expected: Int, found: Int)
    case videoTokenIdMissing(String)
    case videoPlaceholderCountMismatch(expected: Int, found: Int)
    case audioUnsupported(String)
    case audioPlaceholderCountMismatch(expected: Int, found: Int)

    public var description: String {
        switch self {
        case .engineLacksEmbeddingInput(let name):
            return "VisionModel: text engine \(name) does not support "
                + "embedding-input forward — VLM splice impossible"
        case .placeholderCountMismatch(let expected, let found):
            return "VisionModel: prompt has \(found) image-placeholder tokens "
                + "but the vision encoder produced \(expected) image tokens"
        case .videoTokenIdMissing(let family):
            return "VisionModel: \(family) was built without a videoTokenId; "
                + "generate(promptTokens:videoFrames:...) needs the family "
                + "loader to thread `video_token_index` through to VisionModel.init"
        case .videoPlaceholderCountMismatch(let expected, let found):
            return "VisionModel: prompt has \(found) video-placeholder tokens "
                + "but the vision encoder produced \(expected) video tokens"
        case .audioUnsupported(let family):
            return "VisionModel: \(family) was built without an audio tower; "
                + "generate(promptTokens:audio:...) needs the family loader to "
                + "thread `audio_token_id` + an audioEncode hook through to "
                + "VisionModel.init (the checkpoint lacks `audio_config` / "
                + "`audio_tower.*`)"
        case .audioPlaceholderCountMismatch(let expected, let found):
            return "VisionModel: prompt has \(found) audio-placeholder tokens "
                + "but the audio encoder produced \(expected) audio tokens"
        }
    }
}

/// A loaded vision-language model: a `VisionEncoder` + a text
/// `TextGenerating` engine + the image-placeholder token id its chat
/// template uses.
public final class VisionModel: @unchecked Sendable {
    /// The vision tower.
    public let visionEncoder: VisionEncoder
    /// The text backbone (Gemma3Model, Qwen3Model, …).
    public let engine: any TextGenerating
    /// Token id the chat template emits as an image placeholder. Each
    /// occurrence is replaced by one vision token.
    public let imageTokenId: Int
    /// Token id the chat template emits as a video-frame placeholder
    /// (`<|video_pad|>` on Qwen 2/2.5/3 VL). Each occurrence is replaced
    /// by one of the per-temporal-patch merged vision tokens emitted by
    /// the vision tower's `encode(frames:)`. `nil` for families that
    /// don't wire the multi-frame temporal-patch path.
    public let videoTokenId: Int?
    /// Per-channel image normalization the checkpoint expects.
    public let normalization: ImageNormalization
    /// Number of vision tokens one image contributes to the text stream.
    /// Usually `visionEncoder.config.numPatches`, but a family whose
    /// projector pools the patch grid (Gemma 3 VL: 4096 patches → 256
    /// tokens) passes the pooled count explicitly.
    public let imageTokenCount: Int
    /// Token id the chat template emits as an audio placeholder
    /// (`<audio_soft_token>`, id 258881 on Gemma 4). Each occurrence is
    /// replaced by one of the audio soft tokens emitted by `audioEncode`.
    /// `nil` for VLMs without an audio tower.
    public let audioTokenId: Int?
    /// Audio encode hook: a raw 16 kHz mono waveform → audio soft tokens
    /// `[nAudioTokens, hidden]` already projected into the text hidden dim
    /// (front-end → audio encoder → multi-modal embedder). `nil` for VLMs
    /// without an audio tower. Family-supplied so `VisionModel` stays
    /// decoupled from the Gemma 4 audio types.
    public let audioEncode: (@Sendable (_ waveform: [Float], _ device: Device) -> Tensor)?

    /// Whether this VLM wired an audio tower — true exactly when both an
    /// `audioTokenId` and an `audioEncode` hook are present.
    public var supportsAudio: Bool { audioTokenId != nil && audioEncode != nil }

    public init(
        visionEncoder: VisionEncoder, engine: any TextGenerating,
        imageTokenId: Int,
        videoTokenId: Int? = nil,
        normalization: ImageNormalization,
        imageTokenCount: Int? = nil,
        audioTokenId: Int? = nil,
        audioEncode: (@Sendable (_ waveform: [Float], _ device: Device) -> Tensor)? = nil
    ) throws {
        guard engine.supportsEmbeddingInput else {
            throw VisionModelError.engineLacksEmbeddingInput(String(describing: type(of: engine)))
        }
        self.visionEncoder = visionEncoder
        self.engine = engine
        self.imageTokenId = imageTokenId
        self.videoTokenId = videoTokenId
        self.normalization = normalization
        self.imageTokenCount = imageTokenCount ?? visionEncoder.config.numPatches
        self.audioTokenId = audioTokenId
        self.audioEncode = audioEncode
    }

    // ─── Cross-modal splice ──────────────────────────────────────────

    /// Build the spliced prompt-embedding stream.
    ///
    /// `promptTokens` is the tokenized prompt, with exactly one
    /// `imageTokenId` per image-token slot. `imageTokens` is the
    /// vision-encoder output, `[numImages * numPatches, hidden]`. Every
    /// `imageTokenId` position in the prompt takes the next vision row;
    /// every other position takes its text-token embedding.
    ///
    /// Returns one `[hidden]` `Tensor` per prompt position — the input
    /// the text backbone's `forward(inputEmbedding:...)` consumes.
    ///
    /// CPU-driven (the spec marks the splice "CPU-fine"); the per-token
    /// embedding lookups + the vision encode are the only GPU work.
    public func splice(
        promptTokens: [Int], imageTokens: Tensor,
        device: Device = .shared
    ) throws -> [Tensor] {
        let placeholderCount = promptTokens.filter { $0 == imageTokenId }.count
        let visionRows = imageTokens.shape[0]
        guard placeholderCount == visionRows else {
            throw VisionModelError.placeholderCountMismatch(
                expected: visionRows, found: placeholderCount)
        }
        let hidden = engine.hidden
        precondition(
            imageTokens.shape == [visionRows, hidden],
            "VisionModel.splice: image tokens \(imageTokens.shape) "
                + "≠ [\(visionRows), \(hidden)]")

        let imageRowBytes = hidden * imageTokens.dtype.byteSize
        var stream: [Tensor] = []
        stream.reserveCapacity(promptTokens.count)
        var visionCursor = 0
        for tok in promptTokens {
            if tok == imageTokenId {
                // Slice the next vision-encoder row as a [hidden] view.
                let row = Tensor(
                    buffer: imageTokens.buffer,
                    offset: imageTokens.offset + visionCursor * imageRowBytes,
                    shape: [hidden], dtype: imageTokens.dtype)
                stream.append(row)
                visionCursor += 1
            } else {
                stream.append(engine.textEmbedding(tokenId: tok, device: device))
            }
        }
        return stream
    }

    /// Encode an image into vision tokens in the text hidden dim. A vision
    /// tower may supply its own preprocessing (Gemma 4's variable-resolution
    /// processor); otherwise the generic fixed-square preprocess is used.
    public func encodeImage(_ image: RGBImage, device: Device = .shared) -> Tensor {
        if let pixels = visionEncoder.preprocessImage(image, device: device) {
            return visionEncoder.encode(image: pixels, device: device)
        }
        let cfg = visionEncoder.config
        let pixels = MediaUtils.preprocess(
            image, targetW: cfg.imageSize, targetH: cfg.imageSize,
            normalization: normalization, dtype: visionEncoder.dtype,
            device: device)
        return visionEncoder.encode(image: pixels, device: device)
    }

    /// Encode a sequence of video frames into vision tokens in the text
    /// hidden dim. Each frame is preprocessed (resize → normalize) the
    /// same way `encodeImage(...)` is — the vision tower's temporal
    /// folding (one temporal patch = `temporal_patch_size` consecutive
    /// frames) happens inside `VisionEncoder.encode(frames:)`. Throws
    /// `VisionEncoderError.videoUnsupported` for towers that don't wire
    /// the multi-frame path.
    public func encodeVideo(
        _ frames: [RGBImage],
        device: Device = .shared
    ) throws -> Tensor {
        precondition(
            !frames.isEmpty,
            "VisionModel.encodeVideo: expected at least one frame")
        let cfg = visionEncoder.config
        let pixels: [Tensor] = frames.map { frame in
            // Honour a family-specific processor (Gemma 4's variable
            // resolution) per frame; otherwise the generic fixed square.
            visionEncoder.preprocessImage(frame, device: device)
                ?? MediaUtils.preprocess(
                    frame, targetW: cfg.imageSize, targetH: cfg.imageSize,
                    normalization: normalization, dtype: visionEncoder.dtype,
                    device: device)
        }
        return try visionEncoder.encode(frames: pixels, device: device)
    }

    /// Encode a raw 16 kHz mono waveform into audio soft tokens in the
    /// text hidden dim (front-end → audio encoder → multi-modal embedder).
    /// Throws `VisionModelError.audioUnsupported` for VLMs that didn't wire
    /// an audio tower.
    public func encodeAudio(
        _ waveform: [Float], device: Device = .shared
    ) throws -> Tensor {
        guard let audioEncode else {
            throw VisionModelError.audioUnsupported(
                String(describing: type(of: engine)))
        }
        return audioEncode(waveform, device)
    }

    /// Cross-modal splice driver — builds the embedding stream by
    /// walking `promptTokens` and substituting vision-encoder rows at
    /// each image-placeholder / video-placeholder position. The image
    /// and video token streams are consumed independently (each carries
    /// its own cursor) so a prompt may interleave both.
    private func spliceMultimodal(
        promptTokens: [Int],
        imageTokens: Tensor?,
        videoTokens: Tensor?,
        audioTokens: Tensor? = nil,
        device: Device
    ) throws -> [Tensor] {
        let hidden = engine.hidden
        // Validate placeholder counts up-front so we fail fast on
        // mismatched prompts.
        let imagePlaceholderCount = promptTokens.filter { $0 == imageTokenId }.count
        let imageVisionRows = imageTokens?.shape[0] ?? 0
        guard imagePlaceholderCount == imageVisionRows else {
            throw VisionModelError.placeholderCountMismatch(
                expected: imageVisionRows, found: imagePlaceholderCount)
        }
        let videoPlaceholderCount: Int
        if let videoTokenId {
            videoPlaceholderCount = promptTokens.filter { $0 == videoTokenId }.count
        } else {
            videoPlaceholderCount = 0
        }
        let videoVisionRows = videoTokens?.shape[0] ?? 0
        guard videoPlaceholderCount == videoVisionRows else {
            throw VisionModelError.videoPlaceholderCountMismatch(
                expected: videoVisionRows, found: videoPlaceholderCount)
        }
        let audioPlaceholderCount: Int
        if let audioTokenId {
            audioPlaceholderCount = promptTokens.filter { $0 == audioTokenId }.count
        } else {
            audioPlaceholderCount = 0
        }
        let audioRows = audioTokens?.shape[0] ?? 0
        guard audioPlaceholderCount == audioRows else {
            throw VisionModelError.audioPlaceholderCountMismatch(
                expected: audioRows, found: audioPlaceholderCount)
        }
        if let imageTokens {
            precondition(
                imageTokens.shape == [imageVisionRows, hidden],
                "VisionModel.splice: image tokens \(imageTokens.shape) "
                    + "≠ [\(imageVisionRows), \(hidden)]")
        }
        if let videoTokens {
            precondition(
                videoTokens.shape == [videoVisionRows, hidden],
                "VisionModel.splice: video tokens \(videoTokens.shape) "
                    + "≠ [\(videoVisionRows), \(hidden)]")
        }
        if let audioTokens {
            precondition(
                audioTokens.shape == [audioRows, hidden],
                "VisionModel.splice: audio tokens \(audioTokens.shape) "
                    + "≠ [\(audioRows), \(hidden)]")
        }
        // Byte stride for slicing a single [hidden] row out of a modality
        // token buffer — same for image, video, and audio.
        let imageRowBytes =
            imageTokens.map {
                hidden * $0.dtype.byteSize
            } ?? 0
        let videoRowBytes =
            videoTokens.map {
                hidden * $0.dtype.byteSize
            } ?? 0
        let audioRowBytes =
            audioTokens.map {
                hidden * $0.dtype.byteSize
            } ?? 0

        var stream: [Tensor] = []
        stream.reserveCapacity(promptTokens.count)
        var imageCursor = 0
        var videoCursor = 0
        var audioCursor = 0
        for tok in promptTokens {
            if tok == imageTokenId, let imageTokens {
                // VALIDATE: image splice index — each `imageTokenId` slot
                // takes the next vision-encoder row in order.
                let row = Tensor(
                    buffer: imageTokens.buffer,
                    offset: imageTokens.offset + imageCursor * imageRowBytes,
                    shape: [hidden], dtype: imageTokens.dtype)
                stream.append(row)
                imageCursor += 1
            } else if let videoTokenId, tok == videoTokenId, let videoTokens {
                let row = Tensor(
                    buffer: videoTokens.buffer,
                    offset: videoTokens.offset + videoCursor * videoRowBytes,
                    shape: [hidden], dtype: videoTokens.dtype)
                stream.append(row)
                videoCursor += 1
            } else if let audioTokenId, tok == audioTokenId, let audioTokens {
                // VALIDATE: audio splice index — each `audioTokenId` slot
                // (the `boa … eoa` span's interior) takes the next audio
                // soft-token row in order. Mirrors the reference
                // `masked_scatter` at `audio_token_id` positions.
                let row = Tensor(
                    buffer: audioTokens.buffer,
                    offset: audioTokens.offset + audioCursor * audioRowBytes,
                    shape: [hidden], dtype: audioTokens.dtype)
                stream.append(row)
                audioCursor += 1
            } else {
                stream.append(engine.textEmbedding(tokenId: tok, device: device))
            }
        }
        return stream
    }

    // ─── Generation ──────────────────────────────────────────────────

    /// Greedy multi-modal generation. Runs prefill over the spliced
    /// embedding stream (image tokens injected at the placeholder
    /// positions), then decodes `maxTokens` tokens greedily.
    ///
    /// Returns the generated token ids. Coherence-first: this is the
    /// minimal path the integration tests exercise; sampling
    /// filters / streaming are a later pass — the text-only `Generate`
    /// already has them and `VisionModel` will route through it once the
    /// engine grows a public embedding-prefill entry point.
    public func generate(
        promptTokens: [Int], image: RGBImage?,
        maxTokens: Int, parameters: GenerationParameters? = nil,
        eosTokenId: Int?,
        eosTokenIds: [Int] = [],
        device: Device = .shared
    ) throws -> [Int] {
        let imageTokens = image.map { encodeImage($0, device: device) }
        return try runSampled(
            promptTokens: promptTokens,
            imageTokens: imageTokens, videoTokens: nil, audioTokens: nil,
            maxTokens: maxTokens, parameters: parameters,
            eosTokenId: eosTokenId,
            eosTokenIds: eosTokenIds, device: device)
    }

    /// Greedy multi-modal generation with a video prompt. `videoFrames`
    /// is the ordered list of RGB frames (the caller decides sampling —
    /// `VLMTestSupport.catVideoFrames(...)` produces 1–N evenly spaced
    /// frames); the vision tower folds them into temporal patches and
    /// the splice substitutes one vision-token row at every
    /// `<|video_pad|>` placeholder.
    ///
    /// Throws `VisionModelError.videoTokenIdMissing` for VLMs that were
    /// built without a `videoTokenId` (`Gemma3VL`, `Paligemma`,
    /// `FastVLM`, …), or
    /// `VisionModelError.videoPlaceholderCountMismatch` if the prompt
    /// doesn't contain exactly `(T/temporal_patch_size) × spatial`
    /// `<|video_pad|>` placeholders.
    public func generate(
        promptTokens: [Int], videoFrames: [RGBImage],
        maxTokens: Int, parameters: GenerationParameters? = nil,
        eosTokenId: Int?,
        eosTokenIds: [Int] = [],
        device: Device = .shared
    ) throws -> [Int] {
        guard videoTokenId != nil else {
            throw VisionModelError.videoTokenIdMissing(
                String(describing: type(of: visionEncoder)))
        }
        let videoTokens = try encodeVideo(videoFrames, device: device)
        return try runSampled(
            promptTokens: promptTokens,
            imageTokens: nil, videoTokens: videoTokens, audioTokens: nil,
            maxTokens: maxTokens, parameters: parameters,
            eosTokenId: eosTokenId,
            eosTokenIds: eosTokenIds, device: device)
    }

    #if canImport(AVFoundation)
        /// Greedy multi-modal generation from a VIDEO FILE.
        ///
        /// The model owns the whole video path: it decodes + samples up to
        /// `maxFrames` frames from `videoURL` (via
        /// `MediaUtils.decodeVideoFrames`), folds them into temporal patches,
        /// and builds the prompt by inserting EXACTLY the right number of
        /// `videoTokenId` placeholders between the caller-supplied
        /// `preTokens` / `postTokens` (the family's chat-template markers).
        /// The caller no longer decodes frames or computes the placeholder
        /// count — the model derives it from the encoded video-token rows, so
        /// it always matches the tower's `temporal_patch_size` / resolution.
        ///
        /// Optional `framePrefix` / `frameSuffix` wrap EACH decoded frame's
        /// placeholder run — for families whose video prompt interleaves
        /// per-frame markers (Gemma 4 emits a `"MM:SS "` timestamp +
        /// begin/end-of-image tokens around every frame). When BOTH are nil
        /// (Qwen / MiniCPM), the placeholders are one flat run. When either
        /// is provided, the rows are split evenly across the decoded frames
        /// (`totalRows / frameCount` per frame — valid for towers without
        /// temporal folding, i.e. one frame → one contiguous row block).
        ///
        /// Throws `VisionModelError.videoTokenIdMissing` for VLMs that were
        /// built without a `videoTokenId`.
        public func generate(
            preTokens: [Int], videoURL: URL, maxFrames: Int, postTokens: [Int],
            framePrefix: ((Int) -> [Int])? = nil,
            frameSuffix: ((Int) -> [Int])? = nil,
            maxTokens: Int, parameters: GenerationParameters? = nil,
            eosTokenId: Int?,
            eosTokenIds: [Int] = [],
            device: Device = .shared
        ) throws -> [Int] {
            guard let videoTokenId else {
                throw VisionModelError.videoTokenIdMissing(
                    String(describing: type(of: visionEncoder)))
            }
            let frames = try MediaUtils.decodeVideoFrames(
                url: videoURL, maxFrames: maxFrames)
            let videoTokens = try encodeVideo(frames, device: device)
            let totalRows = videoTokens.shape[0]
            // Build the placeholder span. The splice substitutes one video
            // row per `videoTokenId` placeholder, IN ORDER, so the total
            // placeholder count must equal `totalRows` for both layouts.
            let videoSpan: [Int]
            if framePrefix != nil || frameSuffix != nil {
                // Per-frame layout: one row block per decoded frame, wrapped
                // by the caller's per-frame markers.
                precondition(
                    frames.count > 0 && totalRows % frames.count == 0,
                    "VisionModel.generate(videoURL:): per-frame layout needs "
                        + "the video-token rows (\(totalRows)) divisible by "
                        + "the frame count (\(frames.count))")
                let perFrame = totalRows / frames.count
                var span: [Int] = []
                for i in 0 ..< frames.count {
                    span += framePrefix?(i) ?? []
                    span += Array(repeating: videoTokenId, count: perFrame)
                    span += frameSuffix?(i) ?? []
                }
                videoSpan = span
            } else {
                // Flat layout: one placeholder per encoded video-token row.
                videoSpan = Array(repeating: videoTokenId, count: totalRows)
            }
            let promptTokens = preTokens + videoSpan + postTokens
            return try runSampled(
                promptTokens: promptTokens,
                imageTokens: nil, videoTokens: videoTokens, audioTokens: nil,
                maxTokens: maxTokens, parameters: parameters,
                eosTokenId: eosTokenId,
                eosTokenIds: eosTokenIds, device: device)
        }
    #endif

    /// Greedy multi-modal generation with an audio prompt. `waveform` is a
    /// raw 16 kHz mono signal; the audio front-end + encoder + multi-modal
    /// embedder turn it into audio soft tokens, and the splice substitutes
    /// one soft-token row at every `audioTokenId` placeholder (the interior
    /// of the `boa … eoa` span). Throws `VisionModelError.audioUnsupported`
    /// for VLMs without an audio tower.
    public func generate(
        promptTokens: [Int], audio waveform: [Float],
        maxTokens: Int, parameters: GenerationParameters? = nil,
        eosTokenId: Int?,
        eosTokenIds: [Int] = [],
        device: Device = .shared
    ) throws -> [Int] {
        let audioTokens = try encodeAudio(waveform, device: device)
        return try runSampled(
            promptTokens: promptTokens,
            imageTokens: nil, videoTokens: nil, audioTokens: audioTokens,
            maxTokens: maxTokens, parameters: parameters,
            eosTokenId: eosTokenId,
            eosTokenIds: eosTokenIds, device: device)
    }

    // ─── Streaming multimodal generation ─────────────────────────────
    //
    // Token-by-token mirrors of the batch `generate(...)` methods above:
    // each yields one `GenerationChunk` per decoded token (token id only —
    // the `Model` layer fills in `text` via the tokenizer), then a final
    // stats chunk. A buffered collect of any of these is bit-identical to
    // the corresponding batch call (shared `prepareMultimodalDecode` +
    // `decodeOneToken`). The image encode is non-throwing; the video / audio
    // encodes can throw, so those entry points are `throws` (the failure
    // surfaces before the stream rather than inside it).

    /// Streaming image + text generation. See `generate(promptTokens:image:)`.
    public func generateStream(
        promptTokens: [Int], image: RGBImage?,
        maxTokens: Int, parameters: GenerationParameters? = nil,
        eosTokenId: Int?,
        eosTokenIds: [Int] = [],
        device: Device = .shared
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        let imageTokens = image.map { encodeImage($0, device: device) }
        return generateMultimodalStream(
            promptTokens: promptTokens,
            imageTokens: imageTokens, videoTokens: nil, audioTokens: nil,
            maxTokens: maxTokens, parameters: parameters,
            eosTokenId: eosTokenId, eosTokenIds: eosTokenIds, device: device)
    }

    /// Streaming video + text generation. See
    /// `generate(promptTokens:videoFrames:)`.
    public func generateStream(
        promptTokens: [Int], videoFrames: [RGBImage],
        maxTokens: Int, parameters: GenerationParameters? = nil,
        eosTokenId: Int?,
        eosTokenIds: [Int] = [],
        device: Device = .shared
    ) throws -> AsyncThrowingStream<GenerationChunk, Error> {
        guard videoTokenId != nil else {
            throw VisionModelError.videoTokenIdMissing(
                String(describing: type(of: visionEncoder)))
        }
        let videoTokens = try encodeVideo(videoFrames, device: device)
        return generateMultimodalStream(
            promptTokens: promptTokens,
            imageTokens: nil, videoTokens: videoTokens, audioTokens: nil,
            maxTokens: maxTokens, parameters: parameters,
            eosTokenId: eosTokenId, eosTokenIds: eosTokenIds, device: device)
    }

    /// Streaming audio + text generation. See `generate(promptTokens:audio:)`.
    public func generateStream(
        promptTokens: [Int], audio waveform: [Float],
        maxTokens: Int, parameters: GenerationParameters? = nil,
        eosTokenId: Int?,
        eosTokenIds: [Int] = [],
        device: Device = .shared
    ) throws -> AsyncThrowingStream<GenerationChunk, Error> {
        let audioTokens = try encodeAudio(waveform, device: device)
        return generateMultimodalStream(
            promptTokens: promptTokens,
            imageTokens: nil, videoTokens: nil, audioTokens: audioTokens,
            maxTokens: maxTokens, parameters: parameters,
            eosTokenId: eosTokenId, eosTokenIds: eosTokenIds, device: device)
    }

    /// Shared multimodal decode driver — builds the prefill embedding
    /// stream (with optional pre-encoded image / video / audio tokens),
    /// runs prefill to fill the KV cache, then decodes one token at a time,
    /// **sampling each step per `parameters`** via the same
    /// `Sampling.sample` the text path uses (not hard-wired argmax).
    /// Per-token decode state captured right after the multimodal prefill
    /// — the shared seam between the batch `runSampled` and the streaming
    /// `generateMultimodalStream`. The prefill (splice + batched embedding
    /// forward) is the expensive part; both decode loops continue from the
    /// same state, so the two paths stay bit-identical.
    private struct MultimodalDecodeState {
        let caches: [any LayerCacheProtocol]
        var nextToken: Int
        var rng: AnyRandomNumberGenerator
        let params: GenerationParameters
        var tokenHistory: [Int]
        let stopSet: Set<Int>
        var pos: Int
        let maxTokens: Int
        let promptCount: Int
        let prefillTimeS: Double
    }

    /// Run the multimodal prefill (splice → batched embedding forward) and
    /// return the decode state poised to emit the first token. Shared by the
    /// batch + streaming decode loops.
    private func prepareMultimodalDecode(
        promptTokens: [Int],
        imageTokens: Tensor?, videoTokens: Tensor?, audioTokens: Tensor?,
        maxTokens: Int, parameters: GenerationParameters?,
        eosTokenId: Int?, eosTokenIds: [Int],
        device: Device
    ) throws -> MultimodalDecodeState {
        // Respect the caller's generation parameters (temperature / top-p /
        // top-k / min-p / repetition-penalty / seed). Default — when a
        // caller passes none — is deterministic greedy (temperature 0), but
        // the real call paths (tests, `Model`/CLI) pass the model's
        // `defaultGenerationParameters`. The same `Sampling.sample` the
        // text path uses drives every step, so the multimodal decode is no
        // longer hard-wired to argmax.
        let params =
            parameters ?? GenerationParameters().with { $0.temperature = 0 }
        var rng = params.makeRNG()
        let caches = engine.makeLayerCaches(device: device)
        let prefillStart = Date()

        // Build the prefill embedding stream.
        let stream: [Tensor]
        if imageTokens != nil || videoTokens != nil || audioTokens != nil {
            stream = try spliceMultimodal(
                promptTokens: promptTokens,
                imageTokens: imageTokens, videoTokens: videoTokens,
                audioTokens: audioTokens,
                device: device)
        } else {
            // Text-only prompt — every position is a text-token embedding.
            stream = promptTokens.map {
                engine.textEmbedding(tokenId: $0, device: device)
            }
        }

        // Multimodal-placeholder predicate — these positions get token id
        // `0` for any token-id-derived per-position state (Gemma 4 PLE),
        // matching the reference `where(text_mask, input_ids, 0)`. Every
        // other position uses its real token id.
        func pleTokenId(for tok: Int) -> Int {
            if tok == imageTokenId { return 0 }
            if let videoTokenId, tok == videoTokenId { return 0 }
            if let audioTokenId, tok == audioTokenId { return 0 }
            return tok
        }

        // `tokenHistory` feeds the repetition penalty (prompt + decoded so
        // far), mirroring `Generate.driveGeneration`.
        let tokenHistory = promptTokens

        // Prefill: BATCHED. Assemble the spliced stream into a contiguous
        // `[seqLen, hidden]` block and feed it to the engine's batched
        // embedding-prefill (`forwardMulti(inputEmbeddings:)`) in
        // `prefillStepSize`-sized chunks. Only the FINAL position's logits
        // matter — `forwardMulti` returns the chunk tail, so the last chunk
        // yields the first decoded token's logits, sampled via the shared
        // `Sampling.sample` (temperature 0 → greedy, plus top-k / top-p /
        // min-p / rep-penalty).
        let embeddingBlock = assembleEmbeddingBlock(stream, device: device)
        let plePositionIds = promptTokens.map { pleTokenId(for: $0) }

        var nextToken = 0
        let seqLen = stream.count
        let chunkSize = max(
            1, params.prefillStepSize ?? engine.defaultPrefillStepSize)
        var prefillPos = 0
        while prefillPos < seqLen {
            let end = min(prefillPos + chunkSize, seqLen)
            let rows = end - prefillPos
            let chunkBlock = embeddingBlock.slicedRows(start: prefillPos, count: rows)
            let chunkPLE = Array(plePositionIds[prefillPos ..< end])
            let cmd = device.makeCommandBuffer()
            let logits = engine.forwardMulti(
                inputEmbeddings: chunkBlock, startingAt: prefillPos,
                pleTokenIds: chunkPLE,
                caches: caches, on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            if end == seqLen {
                nextToken = Sampling.sample(
                    logits, parameters: params, rng: &rng,
                    tokenHistory: tokenHistory)
            }
            prefillPos = end
        }

        // Stop on any EOS — Gemma 3+ etc. publish `eos_token_id` as a list;
        // the caller passes both `eosTokenId` (single) and `eosTokenIds`.
        var stopSet: Set<Int> = Set(eosTokenIds)
        if let single = eosTokenId { stopSet.insert(single) }

        return MultimodalDecodeState(
            caches: caches, nextToken: nextToken, rng: rng, params: params,
            tokenHistory: tokenHistory, stopSet: stopSet, pos: stream.count,
            maxTokens: maxTokens, promptCount: promptTokens.count,
            prefillTimeS: Date().timeIntervalSince(prefillStart))
    }

    private func runSampled(
        promptTokens: [Int],
        imageTokens: Tensor?, videoTokens: Tensor?, audioTokens: Tensor?,
        maxTokens: Int, parameters: GenerationParameters?,
        eosTokenId: Int?,
        eosTokenIds: [Int],
        device: Device
    ) throws -> [Int] {
        var st = try prepareMultimodalDecode(
            promptTokens: promptTokens,
            imageTokens: imageTokens, videoTokens: videoTokens,
            audioTokens: audioTokens,
            maxTokens: maxTokens, parameters: parameters,
            eosTokenId: eosTokenId, eosTokenIds: eosTokenIds, device: device)
        // Decode: token-id forward from the prefill tail, sampling each step.
        var generated: [Int] = []
        for _ in 0 ..< st.maxTokens {
            if st.stopSet.contains(st.nextToken) { break }
            generated.append(st.nextToken)
            st.tokenHistory.append(st.nextToken)
            st.nextToken = decodeOneToken(
                prior: st.nextToken, pos: st.pos,
                caches: st.caches, params: st.params, rng: &st.rng,
                tokenHistory: st.tokenHistory, device: device)
            st.pos += 1
        }
        return generated
    }

    /// Streaming multimodal decode — yields one `GenerationChunk` per
    /// decoded token (token id only; the `Model` layer fills in `text` via
    /// the tokenizer), then a final chunk carrying `GenerationStats`. The
    /// same prefill + per-token loop as `runSampled`, so a buffered collect
    /// of this stream is bit-identical to the batch path. Cancelling the
    /// consuming task stops decode at the next step.
    func generateMultimodalStream(
        promptTokens: [Int],
        imageTokens: Tensor?, videoTokens: Tensor?, audioTokens: Tensor?,
        maxTokens: Int, parameters: GenerationParameters?,
        eosTokenId: Int?,
        eosTokenIds: [Int],
        device: Device
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var st = try prepareMultimodalDecode(
                        promptTokens: promptTokens,
                        imageTokens: imageTokens, videoTokens: videoTokens,
                        audioTokens: audioTokens,
                        maxTokens: maxTokens, parameters: parameters,
                        eosTokenId: eosTokenId, eosTokenIds: eosTokenIds,
                        device: device)
                    let decodeStart = Date()
                    var generatedCount = 0
                    for _ in 0 ..< st.maxTokens {
                        try Task.checkCancellation()
                        if st.stopSet.contains(st.nextToken) { break }
                        st.tokenHistory.append(st.nextToken)
                        generatedCount += 1
                        continuation.yield(
                            GenerationChunk(
                                text: "", tokens: [st.nextToken],
                                position: st.pos + 1, stats: nil))
                        st.nextToken = decodeOneToken(
                            prior: st.nextToken, pos: st.pos,
                            caches: st.caches, params: st.params, rng: &st.rng,
                            tokenHistory: st.tokenHistory, device: device)
                        st.pos += 1
                    }
                    let stats = Self.minimalStats(
                        promptTokens: st.promptCount,
                        generatedTokens: generatedCount,
                        contextSize: engine.maxContextWindow,
                        prefillTimeS: st.prefillTimeS,
                        decodeTimeS: Date().timeIntervalSince(decodeStart))
                    continuation.yield(
                        GenerationChunk(
                            text: "", tokens: [], position: st.pos, stats: stats))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// One decode step: token-id forward → sample the next token. SYNC (it
    /// blocks on `waitUntilCompleted` internally), so it is safe to call
    /// from the streaming `Task` — the async context forbids a DIRECT
    /// `waitUntilCompleted`, but a sync helper that wraps it is fine (same
    /// trick the text path uses via `engine.forwardSample`).
    private func decodeOneToken(
        prior: Int, pos: Int,
        caches: [any LayerCacheProtocol],
        params: GenerationParameters,
        rng: inout AnyRandomNumberGenerator,
        tokenHistory: [Int],
        device: Device
    ) -> Int {
        let cmd = device.makeCommandBuffer()
        let logits = engine.forward(
            tokenId: prior, position: pos,
            caches: caches, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        return Sampling.sample(
            logits, parameters: params, rng: &rng, tokenHistory: tokenHistory)
    }

    /// Minimal `GenerationStats` for the multimodal stream's final chunk —
    /// counts + timing only (the memory / energy / perplexity probes the
    /// text path wires are not plumbed through the multimodal decode yet).
    private static func minimalStats(
        promptTokens: Int, generatedTokens: Int, contextSize: Int,
        prefillTimeS: Double, decodeTimeS: Double
    ) -> GenerationStats {
        GenerationStats(
            promptTokens: promptTokens, generatedTokens: generatedTokens,
            contextSize: contextSize, prefillTimeS: prefillTimeS,
            decodeTimeS: decodeTimeS, timeToFirstTokenMs: prefillTimeS * 1000,
            steadyTokensPerSecond: nil,
            baselineGPUBytes: 0, postPrefillGPUBytes: 0, postDecodeGPUBytes: 0,
            prefillPeakGPUBytes: 0, decodePeakGPUBytes: 0, wiredTicketBytes: 0,
            weightsBytes: 0, kvCacheAllocatedBytes: 0, kvCacheUsedBytes: 0,
            thinkPerplexity: nil, genPerplexity: nil, thinkKLDivergence: nil,
            genKLDivergence: nil, thinkTokenCount: nil, genTokenCount: nil)
    }

    /// Pack the spliced embedding stream — one `[hidden]` row per prompt
    /// position (text-token embeddings + vision / audio soft tokens) — into
    /// one contiguous `[seqLen, hidden]` block for the engine's batched
    /// embedding-prefill.
    ///
    /// CPU memcpy: every source row is already host-valid — `textEmbedding`
    /// and the vision / audio encoders each commit + wait before returning,
    /// so the splice rows' backing buffers hold completed GPU output. The
    /// assembled block (shared-storage `MTLBuffer`) is then read by the
    /// prefill command buffer committed afterwards, so the writes are
    /// visible to the GPU. All rows must share the stream's dtype.
    private func assembleEmbeddingBlock(
        _ stream: [Tensor], device: Device
    ) -> Tensor {
        precondition(!stream.isEmpty, "assembleEmbeddingBlock: empty stream")
        let hidden = engine.hidden
        let dtype = stream[0].dtype
        let rowBytes = hidden * dtype.byteSize
        let block = Tensor.empty(
            shape: [stream.count, hidden], dtype: dtype, device: device)
        let dstBase = block.buffer.contents().advanced(by: block.offset)
        for (i, row) in stream.enumerated() {
            precondition(
                row.elementCount == hidden && row.dtype == dtype,
                "assembleEmbeddingBlock: row \(i) is \(row.shape) "
                    + "(\(row.dtype)), expected [\(hidden)] (\(dtype))")
            let src = row.buffer.contents().advanced(by: row.offset)
            memcpy(dstBase.advanced(by: i * rowBytes), src, rowBytes)
        }
        return block
    }
}

// MARK: - Seeing conformance

// VisionModel is the composed-VLM handler behind `Model.vlm`. It OVERRIDES the
// shared `Seeing` decode with its tuned batched-embedding prefill path — so it
// never reaches `prefillVisual` (which keeps the trapping default). The unified
// `Seeing.generate`/`generateStream` just route to the existing per-media
// methods by which media is present.
extension VisionModel: Seeing {
    /// The composed text backbone runs the decode.
    public var textEngine: any TextGenerating { engine }

    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        engine.makeLayerCaches(maxSeq: maxSeq, device: device)
    }

    public func generate(
        promptTokens: [Int], image: RGBImage?, videoFrames: [RGBImage],
        maxTokens: Int, parameters: GenerationParameters?,
        eosTokenId: Int?, eosTokenIds: [Int], device: Device
    ) throws -> [Int] {
        if !videoFrames.isEmpty {
            return try generate(
                promptTokens: promptTokens, videoFrames: videoFrames,
                maxTokens: maxTokens, parameters: parameters,
                eosTokenId: eosTokenId, eosTokenIds: eosTokenIds, device: device)
        }
        return try generate(
            promptTokens: promptTokens, image: image,
            maxTokens: maxTokens, parameters: parameters,
            eosTokenId: eosTokenId, eosTokenIds: eosTokenIds, device: device)
    }

    public func generateStream(
        promptTokens: [Int], image: RGBImage?, videoFrames: [RGBImage],
        maxTokens: Int, parameters: GenerationParameters?,
        eosTokenId: Int?, eosTokenIds: [Int], device: Device
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        if !videoFrames.isEmpty {
            do {
                return try generateStream(
                    promptTokens: promptTokens, videoFrames: videoFrames,
                    maxTokens: maxTokens, parameters: parameters,
                    eosTokenId: eosTokenId, eosTokenIds: eosTokenIds, device: device)
            } catch {
                return AsyncThrowingStream { $0.finish(throwing: error) }
            }
        }
        return generateStream(
            promptTokens: promptTokens, image: image,
            maxTokens: maxTokens, parameters: parameters,
            eosTokenId: eosTokenId, eosTokenIds: eosTokenIds, device: device)
    }
}
