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
// MultimodalPrompt — a family-agnostic builder for VLM prompts in the
// CLI. The integration tests hand-build each family's prompt (BOS +
// chat markers + the exact media-placeholder run); the CLI can't, so we
// derive the framing generically from the model's chat template and
// splice the media run in at the start of the user content.

import FFAI
import Foundation

enum MultimodalPrompt {
    /// Build a VLM prompt: the model's chat-template framing for a single
    /// user turn, with `count` copies of `mediaToken` inserted immediately
    /// before the user's text.
    ///
    /// **How the insertion point is found** — we render the user turn
    /// twice through the chat template: once with the real `userText`,
    /// once with empty content. The two token streams share a common
    /// prefix (the BOS + role header); they first diverge exactly where
    /// the user content begins. That divergence index is where the media
    /// placeholders belong (the question references the media that
    /// precedes it). This reproduces the `[…header…] + [media×N] + [text]
    /// + [suffix]` layout every family's hand-built test prompt uses,
    /// without hard-coding per-family markers.
    ///
    /// Falls back to `[bos] + [media×N] + textTokens` when the model
    /// ships no chat template (rare for VLMs, but Paligemma-style base
    /// checkpoints exist). Per-family wrappers like Gemma's
    /// `<start_of_image>` boi/eoi are NOT added here — for those families
    /// the placeholder run alone is accepted by the splice, and the
    /// integration tests remain the high-fidelity reference.
    static func build(
        model: Model,
        userText: String,
        mediaToken: Int,
        count: Int
    ) throws -> [Int] {
        precondition(count > 0, "MultimodalPrompt.build: media token count must be > 0")
        let mediaRun = Array(repeating: mediaToken, count: count)

        guard model.supportsChatTemplate else {
            // No template — best-effort raw framing.
            let bos = model.textTokenizer.bosTokenId ?? model.config.bosTokenId
            let textTokens = model.textTokenizer.encode(text: userText)
            return (bos.map { [$0] } ?? []) + mediaRun + textTokens
        }

        let withText = try model.renderChatTemplate(
            messages: [ChatMessage(role: .user, content: userText)])
        let empty = try model.renderChatTemplate(
            messages: [ChatMessage(role: .user, content: "")])

        // First index where the two renders diverge == start of the user
        // content. Clamp to `withText` bounds defensively.
        var insertAt = 0
        let limit = Swift.min(withText.count, empty.count)
        while insertAt < limit && withText[insertAt] == empty[insertAt] {
            insertAt += 1
        }
        insertAt = Swift.min(insertAt, withText.count)

        var prompt = Array(withText[..<insertAt])
        prompt.append(contentsOf: mediaRun)
        prompt.append(contentsOf: withText[insertAt...])
        return prompt
    }
}
