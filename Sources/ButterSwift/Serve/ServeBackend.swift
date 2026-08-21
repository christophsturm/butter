// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ServeGenerationBackend : swap a real Model+session for a stub in tests.

import Foundation

public protocol ServeGenerationBackend: Sendable {
    var modelId: String { get }
    var maxModelLen: Int? { get }
    var defaultGenerationParameters: GenerationParameters { get }
    /// Prefix tokens reused on the last completion (0 if unknown).
    var lastCachedTokens: Int { get }
    func completeChat(
        messages: [ChatMessage],
        tools: [ChatToolDefinition]?,
        parameters: GenerationParameters,
        templateOptions: ChatTemplateOptions
    ) throws -> AsyncThrowingStream<GenerationChunk, Error>
    /// Stop in-flight prefill/decode. Default is a no-op.
    func abort()
}

extension ServeGenerationBackend {
    public func abort() {}
}

/// Production backend: one `GenerationSession` over a loaded `Model`.
public final class ButterServeBackend: ServeGenerationBackend, @unchecked Sendable {
    public let modelId: String
    public let model: Model
    public let session: GenerationSession

    public init(model: Model, modelId: String) {
        self.model = model
        self.modelId = modelId
        self.session = GenerationSession(model: model)
    }

    public var maxModelLen: Int? { model.engine.maxSeq }
    public var lastCachedTokens: Int { session.lastCachedTokens }
    public var defaultGenerationParameters: GenerationParameters {
        model.defaultGenerationParameters
    }

    public func completeChat(
        messages: [ChatMessage],
        tools: [ChatToolDefinition]?,
        parameters: GenerationParameters,
        templateOptions: ChatTemplateOptions
    ) throws -> AsyncThrowingStream<GenerationChunk, Error> {
        var options = templateOptions
        if tools != nil { options.tools = tools }
        let promptTokens: [Int]
        if model.tokenizer.hasChatTemplate {
            promptTokens = try model.renderChatTemplate(messages: messages, options: options)
        } else {
            let text = messages.map { "\($0.role.rawValue): \($0.content)" }.joined(separator: "\n")
            promptTokens = model.encodePrompt(text)
        }
        return session.complete(
            promptTokens: promptTokens, parameters: parameters)
    }

    public func abort() { session.abort() }
}
