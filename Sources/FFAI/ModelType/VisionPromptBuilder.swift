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
// VisionPromptBuilder — turns a message-based `VisionRequest` (chat
// messages whose content parts carry media slots) into the framed
// prompt-token sequence a `VisionModel` decodes.
//
// Primary path (template-driven): render the family's own chat template
// with the message content-parts. A VL chat template iterates the parts
// (`content is sequence` / `content.type == 'image'`) and emits its own
// vision markers plus ONE placeholder token per media slot — exactly the
// HF-processor model. FFAI then expands each placeholder to the media's
// encoded vision-row count (the count `VisionModel.splice` validates
// against). Verified against cached Qwen*-VL / Gemma4 / SmolVLM2 / FastVLM
// templates; swift-jinja supports the `is string`/`is sequence`/`in`/
// `.type` constructs those templates use.
//
// Fallback: when the template emits no placeholder for the supplied media
// (it doesn't understand content parts), throw `templateFramingUnavailable`
// — per-family FFAI-controlled fallback framing is a follow-up; callers can
// use the token-based `VisionRequest` escape hatch meanwhile.

import Foundation

/// Which media modality a placeholder token / media item is.
enum MediaSlotModality: Sendable {
    case image, video, audio
}

extension Model {
    /// Build the framed prompt tokens for a message-based vision request.
    /// The raw `media` still flow onward to the `VisionModel.generateStream`
    /// overload for encoding + splice; this only produces the token
    /// scaffold with each media placeholder expanded to its row count.
    func buildVisionPrompt(_ req: VisionRequest) throws -> [Int] {
        guard let vlm = vlModel else {
            throw ModelError.unsupportedInput(kind: kind, input: "vision")
        }
        let rendered = try renderChatTemplate(
            messages: req.messages, options: req.templateOptions)

        // Text-only VLM turn (no media): the rendered template is the prompt.
        if req.media.isEmpty { return rendered }

        var expanded: [Int] = []
        expanded.reserveCapacity(rendered.count)
        var mediaIdx = 0
        for tok in rendered {
            guard let slot = placeholderModality(tok, vlm: vlm) else {
                expanded.append(tok)
                continue
            }
            guard mediaIdx < req.media.count else {
                // More placeholders in the prompt than media supplied.
                throw VisionModelError.placeholderCountMismatch(
                    expected: req.media.count, found: mediaIdx + 1)
            }
            let media = req.media[mediaIdx]
            try requireModalityMatch(slot: slot, media: media, vlm: vlm)
            // Expand this one placeholder to the media's encoded row count.
            let n = try visionRowCount(media, vlm: vlm)
            expanded.append(contentsOf: repeatElement(tok, count: n))
            mediaIdx += 1
        }
        guard mediaIdx == req.media.count else {
            // The template emitted fewer placeholders than media — it
            // didn't understand the content parts. Fall back (not yet
            // implemented per-family).
            throw VisionModelError.templateFramingUnavailable(
                String(describing: type(of: vlm.engine)))
        }
        return expanded
    }

    /// The modality of a placeholder token, or nil if `tok` is ordinary
    /// text. Special-token ids don't collide with text ids, so an exact
    /// match is safe.
    private func placeholderModality(_ tok: Int, vlm: VisionModel)
        -> MediaSlotModality?
    {
        if tok == vlm.imageTokenId { return .image }
        if let v = vlm.videoTokenId, tok == v { return .video }
        if let a = vlm.audioTokenId, tok == a { return .audio }
        return nil
    }

    /// Ensure the media at this slot matches the placeholder's modality —
    /// an image slot must pair with `.image`, etc. Guards against callers
    /// ordering `media` differently from the message's slots.
    private func requireModalityMatch(
        slot: MediaSlotModality, media: VisionMedia, vlm: VisionModel
    ) throws {
        let matches: Bool
        switch (slot, media) {
        case (.image, .image), (.video, .video), (.audio, .audio):
            matches = true
        default:
            matches = false
        }
        guard matches else {
            throw VisionModelError.templateFramingUnavailable(
                "media order does not match the message's media slots for "
                    + String(describing: type(of: vlm.engine)))
        }
    }

    /// Encode `media` to learn how many vision rows (placeholder tokens) it
    /// contributes. NOTE: the media is encoded again inside
    /// `VisionModel.generateStream` — an encode-once optimization (pass the
    /// pre-encoded rows through) is a follow-up.
    private func visionRowCount(_ media: VisionMedia, vlm: VisionModel) throws
        -> Int
    {
        switch media {
        case .image(let img): return vlm.encodeImage(img).shape[0]
        case .video(let frames): return try vlm.encodeVideo(frames).shape[0]
        case .audio(let wave): return try vlm.encodeAudio(wave).shape[0]
        }
    }
}
