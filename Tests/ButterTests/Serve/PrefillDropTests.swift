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
// Live instrument: GenerationSession prefill on a tiny hybrid, TCP
// close mid-prefill, leftover chunks must not commit.

import Foundation
import Network
import os
import Testing
import Tokenizers

@testable import ButterSwift

@Suite("Serve live prefill drop", .serialized)
struct PrefillDropTests {

    @Test("TCP close mid-prefill stops leftover GPU chunks")
    func tcpCloseStopsRemainingPrefillChunks() async throws {
        PrefillAbortProbe.reset()
        PrefillAbortProbe.enabled = true
        defer {
            PrefillAbortProbe.enabled = false
            PrefillAbortProbe.reset()
        }

        let engine = makeTinyQwen35(kv: .raw, maxSeq: 128)
        let promptLen = 48
        let tokenizer = ProbeTokenizer(count: promptLen)
        let model = Model(
            engine: engine,
            tokenizer: tokenizer,
            config: ModelConfig(architecture: "Qwen3_5ForCausalLM", modelType: "qwen3_5", raw: [:]),
            modelDirectory: URL(fileURLWithPath: "/tmp/butter-prefill-drop"),
            availableCapabilities: Capability.textOnly,
            enabledCapabilities: Capability.textOnly,
            defaultGenerationParameters: GenerationParameters(
                maxTokens: 1, prefillStepSize: 1, temperature: 0))
        let backend = ButterServeBackend(model: model, modelId: "tiny-hybrid")
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let body =
            #"{"stream":true,"max_tokens":1,"messages":[{"role":"user","content":"prefill-drop"}]}"#
        let req =
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            + "Content-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\n"
            + "Connection: keep-alive\r\n\r\n\(body)"

        let conn = try await prefillDropOpen(port: server.boundPort, send: req)
        try await waitUntil(timeout: 8) { PrefillAbortProbe.committed >= 1 }
        conn.send(
            content: nil,
            contentContext: .finalMessage,
            isComplete: true,
            completion: .contentProcessed { _ in conn.cancel() })
        try await Task.sleep(for: .seconds(2))

        #expect(PrefillAbortProbe.committed >= 1)
        #expect(
            PrefillAbortProbe.committed < promptLen,
            "leftover prefill completed: committed=\(PrefillAbortProbe.committed) begun=\(PrefillAbortProbe.begun)")
    }
}

private func waitUntil(timeout: TimeInterval, _ pred: @escaping () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if pred() { return }
        try await Task.sleep(for: .milliseconds(20))
    }
    throw PrefillDropTimeout()
}

private struct PrefillDropTimeout: Error {}

private func prefillDropOpen(port: UInt16, send: String) async throws -> NWConnection {
    try await withCheckedThrowingContinuation { (cont: CheckedContinuation<NWConnection, Error>) in
        let once = PrefillDropOnce()
        let conn = NWConnection(
            host: NWEndpoint.Host("127.0.0.1"),
            port: NWEndpoint.Port(rawValue: port)!,
            using: .tcp)
        conn.stateUpdateHandler = { state in
            switch state {
            case .ready:
                conn.send(
                    content: Data(send.utf8),
                    completion: .contentProcessed { error in
                        if let error {
                            once.finish(cont, .failure(error))
                        } else {
                            once.finish(cont, .success(conn))
                        }
                    })
            case .failed(let e):
                once.finish(cont, .failure(e))
            default:
                break
            }
        }
        conn.start(queue: DispatchQueue(label: "butter.prefillDrop"))
    }
}

private final class PrefillDropOnce: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock(initialState: false)
    func finish(
        _ cont: CheckedContinuation<NWConnection, Error>,
        _ result: Result<NWConnection, Error>
    ) {
        let first = lock.withLock { done -> Bool in
            if done { return false }
            done = true
            return true
        }
        guard first else { return }
        cont.resume(with: result)
    }
}

/// Fixed-length encode so HTTP `encodePrompt` yields `count` token ids.
private final class ProbeTokenizer: Tokenizer, @unchecked Sendable {
    let count: Int
    init(count: Int) { self.count = count }

    func tokenize(text: String) -> [String] { Array(repeating: "t", count: count) }
    func encode(text: String) -> [Int] { Array(repeating: 1, count: count) }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { encode(text: text) }
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { 1 }
    func convertIdToToken(_ id: Int) -> String? { "t" }
    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { 0 }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    func applyChatTemplate(messages: [Message]) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
}
