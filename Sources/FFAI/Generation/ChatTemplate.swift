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
// ChatTemplate — typed wrappers around swift-transformers'
// `Tokenizer.applyChatTemplate(...)`.
//
// FFAI accepts chat-formatted input through `Model.generate(messages:)`
// / `Model.generateStream(messages:)`. The tokenizer's bundled Jinja
// template (loaded from `tokenizer_config.json`) does the rendering;
// our job is to give callers a typed Swift surface for the inputs.
//
// The thinking / reasoning hooks below pass through to the template's
// well-known variable names (`enable_thinking`, `reasoning_effort`)
// rather than us implementing per-family rendering. That way new
// models that adopt the same conventions just work.

import Foundation
import Tokenizers

/// One piece of a chat message's content. A text-only turn is a single
/// `.text`; a multimodal turn interleaves `.text` with media slots
/// (`.image` / `.video` / `.audio`) that bind positionally, in order, to
/// the `media` array carried alongside the messages on a vision request.
/// The slot itself carries no pixels — it marks WHERE the media's
/// placeholder run belongs in the rendered prompt.
public enum ContentPart: Sendable, Equatable {
    case text(String)
    case image
    case video
    case audio
}

public struct ChatMessage: Sendable, Equatable {
    public enum Role: String, Sendable, Codable, Equatable {
        case system, user, assistant, tool
    }

    public var role: Role
    public var content: [ContentPart]
    /// Reasoning trace for an assistant message. Only relevant in
    /// multi-turn conversations where the prior assistant turn included
    /// a thinking segment that the template wants to re-emit.
    public var thinking: String?

    /// Text-only message (the common case). Content becomes a single
    /// `.text` part, so every text chat template is unaffected.
    public init(role: Role, content: String, thinking: String? = nil) {
        self.role = role
        self.content = [.text(content)]
        self.thinking = thinking
    }

    /// Multimodal message: interleave `.text` with media slots.
    public init(role: Role, content: [ContentPart], thinking: String? = nil) {
        self.role = role
        self.content = content
        self.thinking = thinking
    }

    /// Concatenated text of all `.text` parts (media slots omitted).
    public var textContent: String {
        content.compactMap {
            if case .text(let s) = $0 { return s } else { return nil }
        }.joined()
    }

    /// True if any part is a media slot.
    public var hasMedia: Bool {
        content.contains {
            if case .text = $0 { return false } else { return true }
        }
    }

    /// The media slots in order — binds positionally to the request's
    /// `media` array.
    public var mediaSlots: [ContentPart] {
        content.filter {
            if case .text = $0 { return false } else { return true }
        }
    }

    /// Convert to the `[String: any Sendable]` shape swift-transformers
    /// passes into the Jinja template. Text-only turns emit a plain
    /// `String` content (byte-identical to prior behavior — text
    /// templates unaffected); multimodal turns emit an HF-style content
    /// list (`[{"type":"text","text":…}, {"type":"image"}, …]`) that VL
    /// chat templates iterate.
    public var asTemplateMessage: [String: any Sendable] {
        var dict: [String: any Sendable] = ["role": role.rawValue]
        if hasMedia {
            dict["content"] = content.map { part -> [String: any Sendable] in
                switch part {
                case .text(let s): return ["type": "text", "text": s]
                case .image: return ["type": "image"]
                case .video: return ["type": "video"]
                case .audio: return ["type": "audio"]
                }
            }
        } else {
            dict["content"] = textContent
        }
        if let t = thinking { dict["thinking"] = t }
        return dict
    }
}

/// GPT-OSS Harmony reasoning-effort hints. Surfaced as a typed enum
/// here even though we don't ship a GPT-OSS family yet — the template
/// variable name is well-known so other reasoning-tuned models can pick
/// it up too.
public enum ReasoningEffort: String, Sendable, Codable, Equatable {
    case low, medium, high
}

public struct ChatTemplateOptions: Sendable, Equatable {
    /// Append the assistant prompt suffix that invites the model to
    /// respond. `true` for the typical "render this conversation, now
    /// generate the assistant reply" use case.
    public var addGenerationPrompt: Bool

    /// Set the template's `enable_thinking` variable. Qwen 3 toggles
    /// the `<think>...</think>` segment on/off via this flag. Harmless
    /// for templates that don't reference the variable.
    public var enableThinking: Bool

    /// Set the template's `reasoning_effort` variable (Harmony /
    /// GPT-OSS). `nil` skips it.
    public var reasoningEffort: ReasoningEffort?

    /// Cap the templated token sequence length (truncates leading
    /// system / user turns to fit). `nil` = no cap.
    public var maxLength: Int?

    /// Whether to truncate when `maxLength` is hit. `true` truncates;
    /// `false` throws on overflow.
    public var truncation: Bool

    /// Additional template variables not covered by the typed fields
    /// above. Merged into `additionalContext` after the typed fields.
    public var extraContext: [String: any Sendable]

    public init(
        addGenerationPrompt: Bool = true,
        enableThinking: Bool = false,
        reasoningEffort: ReasoningEffort? = nil,
        maxLength: Int? = nil,
        truncation: Bool = false,
        extraContext: [String: any Sendable] = [:]
    ) {
        self.addGenerationPrompt = addGenerationPrompt
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.maxLength = maxLength
        self.truncation = truncation
        self.extraContext = extraContext
    }

    public static func == (lhs: ChatTemplateOptions, rhs: ChatTemplateOptions) -> Bool {
        lhs.addGenerationPrompt == rhs.addGenerationPrompt
            && lhs.enableThinking == rhs.enableThinking
            && lhs.reasoningEffort == rhs.reasoningEffort && lhs.maxLength == rhs.maxLength
            && lhs.truncation == rhs.truncation
        // extraContext intentionally not compared — `any Sendable` isn't
        // Equatable. Two options with different `extraContext` will read
        // as equal here; callers that care should compare the dicts
        // themselves.
    }
}

public enum ChatTemplateError: Error, CustomStringConvertible {
    case noTemplateOnTokenizer
    case renderFailed(any Error)

    public var description: String {
        switch self {
        case .noTemplateOnTokenizer:
            return
                "Tokenizer has no chat template — render the prompt manually or load a different tokenizer."
        case .renderFailed(let e):
            return "Chat template render failed: \(e)"
        }
    }
}

extension Model {
    /// Render `messages` through the tokenizer's chat template and
    /// return the resulting token ids — same input the model would
    /// receive if you'd encoded the rendered string yourself.
    public func renderChatTemplate(
        messages: [ChatMessage],
        options: ChatTemplateOptions = .init()
    ) throws -> [Int] {
        // Prefer the tokenizer's built-in template (from
        // tokenizer_config.json); fall back to a standalone
        // `chat_template.jinja` shipped alongside the model, passed as a
        // literal override. One of the two must exist.
        let standalone = standaloneChatTemplate
        guard tokenizer?.hasChatTemplate == true || standalone != nil else {
            throw ChatTemplateError.noTemplateOnTokenizer
        }
        var additional: [String: any Sendable] = options.extraContext
        additional["enable_thinking"] = options.enableThinking
        if let effort = options.reasoningEffort {
            additional["reasoning_effort"] = effort.rawValue
        }
        let templateMessages = messages.map { $0.asTemplateMessage }
        do {
            return try textTokenizer.applyChatTemplate(
                messages: templateMessages,
                chatTemplate: standalone.map { ChatTemplateArgument.literal($0) },
                addGenerationPrompt: options.addGenerationPrompt,
                truncation: options.truncation,
                maxLength: options.maxLength,
                tools: nil,
                additionalContext: additional
            )
        } catch {
            throw ChatTemplateError.renderFailed(error)
        }
    }

    /// Buffered chat-shaped generation. Renders `messages` through the
    /// tokenizer's chat template, then drives the standard
    /// generate loop.
    public func generate(
        messages: [ChatMessage],
        templateOptions: ChatTemplateOptions = .init(),
        parameters: GenerationParameters? = nil,
        profile: Profile = .shared
    ) async throws -> GenerationResult {
        let promptTokens = try renderChatTemplate(
            messages: messages,
            options: templateOptions)
        let params = parameters ?? defaultGenerationParameters
        let stream = generateStreamInternal(
            promptTokens: promptTokens,
            parameters: params, profile: profile)
        return try await collectStream(stream, promptTokens: promptTokens)
    }

    /// Streaming chat-shaped generation. Same shape as
    /// `generateStream(prompt:parameters:)` but the prompt is built by
    /// rendering `messages` through the tokenizer's chat template.
    public func generateStream(
        messages: [ChatMessage],
        templateOptions: ChatTemplateOptions = .init(),
        parameters: GenerationParameters? = nil,
        profile: Profile = .shared
    )
        throws -> AsyncThrowingStream<GenerationChunk, Error>
    {
        let promptTokens = try renderChatTemplate(
            messages: messages,
            options: templateOptions)
        let params = parameters ?? defaultGenerationParameters
        return generateStreamInternal(
            promptTokens: promptTokens,
            parameters: params, profile: profile)
    }
}
