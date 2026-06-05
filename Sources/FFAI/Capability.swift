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
// Capability — what a model can do. Multi-modal models declare which
// capabilities they support via their family file; users pick which to
// enable at load time. Disabled modalities skip weight allocation.
//
// textIn / textOut are universal for LLMs and always implicitly enabled.

import Foundation

public enum Capability: String, Sendable, Hashable, CaseIterable, Codable {
    case textIn
    case textOut
    /// Image input — image-only VL models (Gemma 3/4-VL, Nemotron-VL,
    /// Idefics3, GlmOcr, FastVLM, Mistral3, Paligemma, …). Video-capable
    /// families declare `.videoIn` in addition to `.imageIn`.
    case imageIn
    /// A model that can consume video — i.e. a temporally-ordered
    /// sequence of frames. Distinct from `imageIn` because not every
    /// vision-language model wires the multi-frame temporal-patch
    /// path: a `videoIn` model accepts `[Tensor]` (one frame each) via
    /// `VisionEncoder.encode(frames:device:)` and folds them into the
    /// vision-token stream in temporal-patch chunks, while an
    /// `imageIn`-only model takes a single image and treats the
    /// temporal axis as a degenerate repeat.
    case videoIn
    case audioIn
    case audioOut
    /// Image output — a model that synthesizes images (text-to-image /
    /// image-to-image generators). No FFAI family wires this yet; the case
    /// exists so the capability vocabulary and the `textToImage` /
    /// `imageToImage` sets below are in place ahead of the first generator.
    case imageOut
    /// Video output — a model that synthesizes video (text-to-video /
    /// video-to-video generators). Future-proofing alongside `imageOut`;
    /// no FFAI family wires this yet.
    case videoOut
    case toolCalling
    /// Model supports chain-of-thought / "thinking" generation — emits
    /// a private reasoning trace before the final answer (Qwen 3 thinking,
    /// DeepSeek-R1, Claude extended thinking). The trace is typically
    /// fenced in `<think>…</think>` (or family-specific tokens) and
    /// stripped by `ThinkingSplit` before being shown to the user.
    case thinking
    /// Model supports a user-tunable reasoning-effort dial (none / low /
    /// medium / high / extra-high / max), distinct from just having
    /// `.thinking`. The selected level is set on `GenerationParameters`,
    /// not here — this capability just advertises that the model
    /// honours it.
    case reasoningLevel
}

/// User-tunable reasoning effort for models that advertise the
/// `Capability.reasoningLevel` capability. Models without it ignore
/// the setting.
///
/// Values follow the Claude Opus convention (`none` → `max`) that
/// other reasoning-tuned model families are adopting. `.none` disables
/// the reasoning trace entirely; `.max` lets the model spend as long
/// as it needs to.
///
/// FFAI exposes the **full** enum as the common user-facing dial.
/// Models that natively only understand a subset (e.g. GPT-OSS-20B
/// supports `{.low, .medium, .high}`) declare their native set via
/// `ReasoningCapable.supportedReasoningLevels` and the resolver
/// `clamped(to:)` below maps the user's request to the nearest native
/// value. `.none` always maps to `.none` — explicit disable wins.
public enum ReasoningLevel: String, Sendable, Hashable, CaseIterable, Codable {
    case none
    case low
    case medium
    case high
    /// Raw value `"extra-high"` to match the hyphenated convention
    /// other model families use on the wire.
    case extraHigh = "extra-high"
    case max
}

extension ReasoningLevel {
    /// Canonical "more reasoning" ordering. Used by `clamped(to:)` to
    /// pick the nearest native level by index distance.
    public static let canonicalOrder: [ReasoningLevel] = [
        .none, .low, .medium, .high, .extraHigh, .max,
    ]

    /// Map this user-requested level to the nearest value the model
    /// actually understands. `.none` is always honoured (returns
    /// `.none` regardless of `supported`) — it's an explicit
    /// "disable" signal. For any other value:
    ///
    ///   • If `supported` already contains it, return it unchanged.
    ///   • Otherwise pick the nearest member of `supported` by
    ///     canonical-order distance. Ties break toward the **lower**
    ///     (cheaper) level — better to under-reason than to silently
    ///     burn extra tokens.
    ///
    /// Example — GPT-OSS-20B has `supported = {.low, .medium, .high}`:
    ///   - `.none` → `.none`              (explicit disable)
    ///   - `.low / .medium / .high`        unchanged
    ///   - `.extraHigh / .max`            → `.high` (clamped)
    public func clamped(to supported: Set<ReasoningLevel>) -> ReasoningLevel {
        if self == .none { return .none }
        if supported.contains(self) { return self }
        let order = ReasoningLevel.canonicalOrder
        guard let myIdx = order.firstIndex(of: self) else { return self }
        let candidates = order.enumerated().filter { supported.contains($0.element) }
        guard !candidates.isEmpty else { return self }
        let best = candidates.min { lhs, rhs in
            let dl = abs(lhs.offset - myIdx)
            let dr = abs(rhs.offset - myIdx)
            if dl != dr { return dl < dr }
            // Tie-break toward the lower (cheaper) level.
            return lhs.offset < rhs.offset
        }
        return best!.element
    }
}

/// Conformance marker for models that advertise
/// `Capability.reasoningLevel`. Declares which native levels the
/// model recognises so the runtime can clamp user requests.
///
/// Conformance is by family variant struct (e.g.
/// `GPTOSSMoEVariant`) — the same shape as
/// `defaultGenerationParameters`. Non-reasoning families simply
/// don't conform.
public protocol ReasoningCapable {
    /// Levels the model natively recognises (excluding `.none`,
    /// which every reasoning-capable model honours implicitly as
    /// "disable reasoning"). For GPT-OSS-20B this is
    /// `[.low, .medium, .high]`; new models add whatever they
    /// support.
    static var supportedReasoningLevels: Set<ReasoningLevel> { get }
}

extension Capability {
    public static let textOnly: Set<Capability> = [.textIn, .textOut]
    public static let textWithTools: Set<Capability> = [.textIn, .textOut, .toolCalling]

    /// Speech-to-text models (Whisper): audio in, text out.
    public static let speechToText: Set<Capability> = [.audioIn, .textOut]
    /// Image-to-text models (image captioning / VQA): image in, text out.
    public static let imageToText: Set<Capability> = [.imageIn, .textOut]
    /// Video-to-text models (video captioning / understanding): video in,
    /// text out.
    public static let videoToText: Set<Capability> = [.videoIn, .textOut]
    /// Text-to-speech models (Kokoro): text in, audio out.
    public static let textToSpeech: Set<Capability> = [.textIn, .audioOut]
    /// Omni-modal models: every input modality (text + image + video +
    /// audio in), text out. The full omni capability set — a model that
    /// consumes all four modalities. (`omniAudio` below is the narrower
    /// audio-only-omni subset FFAI's Qwen-Omni path currently wires.)
    public static let omni: Set<Capability> = [
        .textIn, .imageIn, .videoIn, .audioIn, .textOut,
    ]
    /// Audio-only omni models (Qwen-Omni as wired today): text + audio in,
    /// text out. Subset of `omni` for families whose non-audio modalities
    /// aren't spliced yet.
    public static let omniAudio: Set<Capability> = [.textIn, .audioIn, .textOut]
    /// Speech enhancement / source separation / audio segmentation
    /// (DeepFilterNet, MossFormer2-SE, SAMAudio): audio in, audio out.
    public static let speechToSpeech: Set<Capability> = [.audioIn, .audioOut]

    // ── Generative output sets (future-proofing) ─────────────────────
    // No FFAI family wires an image/video generator yet; these sets are
    // declared ahead of the first one so the capability vocabulary is
    // stable when it lands.

    /// Text-to-image generators (diffusion / autoregressive): text in,
    /// image out.
    public static let textToImage: Set<Capability> = [.textIn, .imageOut]
    /// Image-to-image generators (edit / style / super-resolution): image
    /// in, image out.
    public static let imageToImage: Set<Capability> = [.imageIn, .imageOut]
    /// Text-to-video generators: text in, video out.
    public static let textToVideo: Set<Capability> = [.textIn, .videoOut]
    /// Image-to-video generators: image in, video out.
    public static let imageToVideo: Set<Capability> = [.imageIn, .videoOut]
    /// Omni-modal video models: every input modality (text + image + video),
    /// video out. The full omni video capability set — a model that consumes
    /// all three modalities.
    public static let omniVideo: Set<Capability> = [.textIn, .imageIn, .videoIn, .videoOut]
    /// Video-to-video generators (edit / interpolation): video in, video
    /// out.
    public static let videoToVideo: Set<Capability> = [.videoIn, .videoOut]
}
