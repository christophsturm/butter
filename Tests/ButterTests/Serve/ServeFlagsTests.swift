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
// Serve load flags, thinking kwargs, and cached_tokens.

import Foundation
import os
import Testing

@testable import ButterSwift

@Suite("Serve flags")
struct ServeFlagsTests {

    @Test("aura kv-cache is AURA not raw : mutation: ignore --kv-cache")
    func auraKVCache() throws {
        let opts = try LoadOptions.fromCLI(kvCache: "aura")
        guard case .auraQuantized = opts.kvCache else {
            Issue.record("expected auraQuantized, got \(opts.kvCache)")
            return
        }
    }

    @Test("unknown kv-cache is an error : mutation: silently fall back to raw")
    func unknownKVCache() {
        #expect(throws: LoadOptions.CLIError.unknownKV("nope")) {
            _ = try LoadOptions.fromCLI(kvCache: "nope")
        }
    }

    @Test("max-context sizes the cache : mutation: drop the field")
    func maxContext() throws {
        let opts = try LoadOptions.fromCLI(maxContextLength: 8192)
        #expect(opts.maxContextLength == 8192)
    }

    @Test("max-context 0 is rejected")
    func maxContextZero() {
        #expect(throws: LoadOptions.CLIError.maxContextMustBePositive) {
            _ = try LoadOptions.fromCLI(maxContextLength: 0)
        }
    }

    @Test("window keep without size is rejected")
    func windowKeepNeedsSize() {
        #expect(throws: LoadOptions.CLIError.windowKeepNeedsSize) {
            _ = try LoadOptions.fromCLI(kvWindowKeep: 4)
        }
    }

    @Test("aura decode path compressed vs mirror")
    func auraDecodePath() throws {
        let a = try LoadOptions.fromCLI(kvCache: "aura", auraDecodePath: "compressed")
        #expect(a.auraDecodePath == .compressed)
        let b = try LoadOptions.fromCLI(kvCache: "aura", auraDecodePath: "dequant-mirror")
        #expect(b.auraDecodePath == .dequantMirror)
    }

    @Test("chat_template_kwargs.enable_thinking is parsed : Claims")
    func thinkingKwargs() throws {
        let json = """
            {"messages":[{"role":"user","content":"hi"}],\
            "chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"high"}}
            """
        let req = try ChatCompletionHTTPRequest.parse(Data(json.utf8))
        #expect(req.chatTemplateKwargs?.enableThinking == true)
        #expect(req.chatTemplateKwargs?.reasoningEffort == .high)
    }

    @Test("prepare records lastCachedTokens as nPast : Product")
    func lastCachedTokensFromPrepare() {
        let kv = FakeKVCache(length: 4, maxSeq: 32)
        let session = GenerationSession(makeCaches: { [FakeKVCache(length: 0, maxSeq: 32)] })
        session.testingReplaceCaches([kv], tokens: [1, 2, 3, 4])
        let n = session.prepare(for: [1, 2, 3, 4, 5, 6])
        #expect(n == 4)
        #expect(session.lastCachedTokens == 4)
    }

    @Test("tool round replays structured calls and reports a matched prefix")
    func usageCachedTokensFromToolRound() async throws {
        let backend = SessionStubBackend(modelId: "stub-model")
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        func post(_ messages: [[String: Any]]) async throws -> [String: Any] {
            var req = URLRequest(
                url: URL(string: "http://127.0.0.1:\(server.boundPort)/v1/chat/completions")!)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "stream": false,
                "messages": messages,
            ])
            let (data, resp) = try await URLSession.shared.data(for: req)
            #expect((resp as! HTTPURLResponse).statusCode == 200)
            return try #require(
                JSONSerialization.jsonObject(with: data) as? [String: Any])
        }
        var messages: [[String: Any]] = [["role": "user", "content": "read it"]]
        let first = try await post(messages)
        let firstUsage = first["usage"] as? [String: Any]
        #expect(firstUsage?["cached_tokens"] as? Int == 0)
        let firstChoice = try #require((first["choices"] as? [[String: Any]])?.first)
        #expect(firstChoice["finish_reason"] as? String == "tool_calls")
        let assistant = try #require(firstChoice["message"] as? [String: Any])
        #expect(assistant["content"] is NSNull)
        let calls = try #require(assistant["tool_calls"] as? [[String: Any]])
        let callId = try #require(calls.first?["id"] as? String)
        messages.append(assistant)
        messages.append([
            "role": "tool", "tool_call_id": callId, "content": "file body",
        ])

        let second = try await post(messages)
        let secondUsage = second["usage"] as? [String: Any]
        let cached = try #require(secondUsage?["cached_tokens"] as? Int)
        #expect(cached > 0)
        #expect(cached == backend.lastCachedTokens)

        let observed = backend.lastMessages
        #expect(observed.count == 3)
        #expect(observed[1].role == .assistant)
        #expect(observed[1].toolCalls?.first?.name == "read")
        #expect(observed[1].toolCalls?.first?.arguments.contains("path") == true)
        #expect(observed[2].role == .tool)
        #expect(observed[2].toolCallId == callId)
    }

    @Test("tool-round cached_tokens is nPast of system+tools stem")
    func usageCachedTokensOnToolRound() async throws {
        let backend = SessionStubBackend(modelId: "stub-model")
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let toolsJSON =
            #"[{"type":"function","function":{"name":"read","description":"r","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]"#
        func post(_ messages: String) async throws -> Int? {
            var req = URLRequest(
                url: URL(string: "http://127.0.0.1:\(server.boundPort)/v1/chat/completions")!)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = Data(
                #"{"stream":false,"tools":\#(toolsJSON),"messages":\#(messages)}"#.utf8)
            let (data, resp) = try await URLSession.shared.data(for: req)
            #expect((resp as! HTTPURLResponse).statusCode == 200)
            let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            return (obj?["usage"] as? [String: Any])?["cached_tokens"] as? Int
        }
        let firstMsgs =
            #"[{"role":"system","content":"You are a helper."},{"role":"user","content":"read /tmp/x"}]"#
        #expect(try await post(firstMsgs) == 0)
        let secondMsgs =
            #"[{"role":"system","content":"You are a helper."},{"role":"user","content":"read /tmp/x"},{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"ok"}]"#
        let second = try await post(secondMsgs)
        let toolsDef = [ChatToolDefinition(name: "read", description: "r")]
        let firstIds = SessionStubBackend.promptIds(
            messages: [
                ChatMessage(role: .system, content: "You are a helper."),
                ChatMessage(role: .user, content: "read /tmp/x"),
            ], tools: toolsDef)
        let secondIds = SessionStubBackend.promptIds(
            messages: [
                ChatMessage(role: .system, content: "You are a helper."),
                ChatMessage(role: .user, content: "read /tmp/x"),
                ChatMessage(
                    role: .assistant, content: "",
                    toolCalls: [ChatMessageToolCall(id: "call_1", name: "read", arguments: "{}")]),
                ChatMessage(role: .tool, content: "ok", toolCallId: "call_1"),
            ], tools: toolsDef)
        let expected = GenerationSession.nPast(cached: firstIds, new: secondIds)
        #expect(second == expected)
        #expect(expected > 0, "tool-round stem must overlap")
    }
}

/// Fake session: message bytes are token ids, so overlapping prompts
/// produce a real nPast instead of a planted usage field.
final class SessionStubBackend: ServeGenerationBackend, @unchecked Sendable {
    let modelId: String
    var maxModelLen: Int? = 2048
    var defaultGenerationParameters = GenerationParameters()
    private let session = GenerationSession(makeCaches: {
        [FakeKVCache(length: 0, maxSeq: 256)]
    })
    private let observed = OSAllocatedUnfairLock(initialState: [ChatMessage]())

    init(modelId: String) { self.modelId = modelId }

    var lastCachedTokens: Int { session.lastCachedTokens }
    var lastMessages: [ChatMessage] { observed.withLock { $0 } }

    static func promptIds(messages: [ChatMessage], tools: [ChatToolDefinition]?) -> [Int] {
        var s = ""
        if let tools {
            for t in tools { s += "tool:\(t.name)\n" }
        }
        for m in messages {
            s += "\(m.role.rawValue):\(m.content)"
            if let id = m.toolCallId { s += "#\(id)" }
            if let tcs = m.toolCalls {
                for tc in tcs { s += ">\(tc.id ?? ""):\(tc.name):\(tc.arguments)" }
            }
            s += "\n"
        }
        return Array(s.utf8.map { Int($0) })
    }

    func completeChat(
        messages: [ChatMessage],
        tools: [ChatToolDefinition]?,
        parameters: GenerationParameters,
        templateOptions: ChatTemplateOptions
    ) throws -> AsyncThrowingStream<GenerationChunk, Error> {
        observed.withLock { $0 = messages }
        let ids = Self.promptIds(messages: messages, tools: tools)
        _ = session.prepare(for: ids)
        let isToolRound = messages.contains { $0.role == .tool }
        let output =
            isToolRound
            ? "done"
            : "<tool_call><function=read><parameter=path>note.md</parameter></function></tool_call>"
        let committed = ids + output.utf8.map { Int($0) }
        let kv = FakeKVCache(length: committed.count, maxSeq: 256)
        session.testingReplaceCaches([kv], tokens: committed)
        let stats = GenerationStats(
            promptTokens: ids.count, generatedTokens: 1, contextSize: 0,
            prefillTimeS: 0, decodeTimeS: 0, timeToFirstTokenMs: 0,
            steadyTokensPerSecond: nil,
            baselineGPUBytes: 0, postPrefillGPUBytes: 0, postDecodeGPUBytes: 0,
            prefillPeakGPUBytes: 0, decodePeakGPUBytes: 0,
            wiredTicketBytes: 0, weightsBytes: 0,
            kvCacheAllocatedBytes: 0, kvCacheUsedBytes: 0,
            thinkPerplexity: nil, genPerplexity: nil,
            thinkKLDivergence: nil, genKLDivergence: nil,
            thinkTokenCount: nil, genTokenCount: nil)
        return AsyncThrowingStream { continuation in
            continuation.yield(
                GenerationChunk(text: output, tokens: [1], position: ids.count))
            continuation.yield(
                GenerationChunk(text: "", tokens: [], position: ids.count, stats: stats))
            continuation.finish()
        }
    }

    private func renderStubPrompt(_ messages: [ChatMessage]) -> String {
        var rendered = messages.map { message in
            var part = "<\(message.role.rawValue)>\(message.content)"
            for call in message.toolCalls ?? [] {
                part += "<tool_call><function=\(call.name)>"
                if let data = call.arguments.data(using: .utf8),
                    let args = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                {
                    for key in args.keys.sorted() {
                        part += "<parameter=\(key)>\(args[key]!)</parameter>"
                    }
                }
                part += "</function></tool_call>"
            }
            part += "</\(message.role.rawValue)>"
            return part
        }.joined()
        rendered += "<assistant>"
        return rendered
    }
}
