// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed as an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// One resident checkpoint. Completions `model` may swap it.

import Foundation
import os

public final class ResidentServeBackend: ServeGenerationBackend, @unchecked Sendable {
    public typealias Loader = @Sendable (String) throws -> ButterServeBackend

    private let inner: OSAllocatedUnfairLock<ButterServeBackend>
    private let loader: Loader
    private let loads = OSAllocatedUnfairLock(initialState: 1)

    public init(initial: ButterServeBackend, loader: @escaping Loader) {
        self.inner = OSAllocatedUnfairLock(initialState: initial)
        self.loader = loader
    }

    public var modelId: String { inner.withLock { $0.modelId } }
    public var maxModelLen: Int? { inner.withLock { $0.maxModelLen } }
    public var lastCachedTokens: Int { inner.withLock { $0.lastCachedTokens } }
    public var defaultGenerationParameters: GenerationParameters {
        inner.withLock { $0.defaultGenerationParameters }
    }
    public var loadCount: Int { loads.withLock { $0 } }

    public func abort() { inner.withLock { $0.abort() } }

    public func prepare(requestedModel: String?) throws {
        let want = requestedModel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if want.isEmpty { return }
        let current = inner.withLock { $0.modelId }
        if want == current { return }
        let next: ButterServeBackend
        do {
            next = try loader(want)
        } catch {
            throw ServeError.notFound
        }
        let old = inner.withLock { slot -> ButterServeBackend in
            let prev = slot
            slot = next
            return prev
        }
        old.abort()
        loads.withLock { $0 += 1 }
    }

    public func completeChat(
        messages: [ChatMessage],
        tools: [ChatToolDefinition]?,
        parameters: GenerationParameters,
        templateOptions: ChatTemplateOptions
    ) throws -> AsyncThrowingStream<GenerationChunk, Error> {
        try inner.withLock {
            try $0.completeChat(
                messages: messages, tools: tools, parameters: parameters,
                templateOptions: templateOptions)
        }
    }
}
