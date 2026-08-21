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
// Hybrid prefix reuse. Kept off the TCP-drop suite so a live prefill
// abort is not queued behind other GPU work.

import Foundation
import Testing
import Tokenizers

@testable import ButterSwift

@Suite("Serve hybrid prefix")
struct HybridPrefixTests {

    @Test("hybrid session restores the prior prompt boundary")
    func hybridSessionRestoresPromptBoundary() async throws {
        let engine = makeTinyQwen35(kv: .raw, maxSeq: 128)
        let model = Model(
            engine: engine,
            tokenizer: ProbeTokenizer(count: 1),
            config: ModelConfig(
                architecture: "Qwen3_5ForCausalLM", modelType: "qwen3_5", raw: [:]),
            modelDirectory: URL(fileURLWithPath: "/tmp/butter-prefix-boundary"),
            availableCapabilities: Capability.textOnly,
            enabledCapabilities: Capability.textOnly,
            defaultGenerationParameters: GenerationParameters(
                maxTokens: 1, prefillStepSize: 2, temperature: 0))
        let session = GenerationSession(model: model)

        for try await _ in session.complete(promptTokens: [1, 2, 3, 4]) {}
        #expect(session.lastCachedTokens == 0)
        var reusedOutput: [Int] = []
        for try await chunk in session.complete(promptTokens: [1, 2, 3, 4, 9]) {
            reusedOutput.append(contentsOf: chunk.tokens)
        }

        #expect(session.lastCachedTokens == 4)

        let fresh = GenerationSession(model: model)
        var freshOutput: [Int] = []
        for try await chunk in fresh.complete(promptTokens: [1, 2, 3, 4, 9]) {
            freshOutput.append(contentsOf: chunk.tokens)
        }
        #expect(reusedOutput == freshOutput)
    }

    @Test("identical hybrid prompt rebuilds to recompute seed logits")
    func identicalHybridPromptRebuilds() async throws {
        let engine = makeTinyQwen35(kv: .raw, maxSeq: 128)
        let model = Model(
            engine: engine,
            tokenizer: ProbeTokenizer(count: 1),
            config: ModelConfig(
                architecture: "Qwen3_5ForCausalLM", modelType: "qwen3_5", raw: [:]),
            modelDirectory: URL(fileURLWithPath: "/tmp/butter-identical-boundary"),
            availableCapabilities: Capability.textOnly,
            enabledCapabilities: Capability.textOnly,
            defaultGenerationParameters: GenerationParameters(
                maxTokens: 1, prefillStepSize: 2, temperature: 0))
        let session = GenerationSession(model: model)
        let prompt = [1, 2, 3, 4]

        var firstOutput: [Int] = []
        for try await chunk in session.complete(promptTokens: prompt) {
            firstOutput.append(contentsOf: chunk.tokens)
        }
        var repeatedOutput: [Int] = []
        for try await chunk in session.complete(promptTokens: prompt) {
            repeatedOutput.append(contentsOf: chunk.tokens)
        }

        #expect(session.lastCachedTokens == 0)
        #expect(repeatedOutput == firstOutput)
    }

    @Test("production tool round reports tokenizer-matched hybrid prefix")
    func productionToolRoundUsesRenderedPrefix() async throws {
        let tokenizer = try await loadToolRoundTokenizer()
        let engine = makeTinyQwen35(kv: .raw, maxSeq: 2048, vocab: 512)
        let model = Model(
            engine: engine,
            tokenizer: tokenizer,
            config: ModelConfig(
                architecture: "Qwen3_5ForCausalLM", modelType: "qwen3_5", raw: [:]),
            modelDirectory: URL(fileURLWithPath: "/tmp/butter-tool-round-prefix"),
            availableCapabilities: Capability.textOnly,
            enabledCapabilities: Capability.textOnly,
            defaultGenerationParameters: GenerationParameters(
                maxTokens: 1, prefillStepSize: 2, temperature: 0))
        let templateTools = [
            ChatToolDefinition(
                name: "read_note", description: "Read a note from disk",
                parametersJSON:
                    #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#)
        ]
        let firstMessages = [
            ChatMessage(
                role: .system,
                content:
                    "Use tools when needed. Return concise answers. Check every path before reading."),
            ChatMessage(
                role: .user, content: "Read note.md and summarize it."),
        ]
        let secondMessages =
            firstMessages + [
                ChatMessage(
                    role: .assistant, content: "",
                    toolCalls: [
                        ChatToolCall(
                            id: "call_exact_1", name: "read_note",
                            arguments: #"{"path":"note.md"}"#)
                    ]),
                ChatMessage(
                    role: .tool, content: "Prefix reuse confirmed now.",
                    toolCallId: "call_exact_1"),
            ]
        let options = ChatTemplateOptions(tools: templateTools)
        let firstRendered = try model.renderChatTemplate(
            messages: firstMessages, options: options)
        let secondRendered = try model.renderChatTemplate(
            messages: secondMessages, options: options)
        #expect(firstRendered.count == 277, "first=\(firstRendered.count)")
        #expect(secondRendered.count == 325, "second=\(secondRendered.count)")
        #expect(secondRendered.starts(with: firstRendered))
        let missingCalls = try model.renderChatTemplate(
            messages: firstMessages + [
                ChatMessage(role: .assistant, content: ""),
                secondMessages[3],
            ], options: options)
        #expect(missingCalls != secondRendered)
        #expect(secondMessages[3].asTemplateMessage["tool_call_id"] as? String == "call_exact_1")

        let backend = ButterServeBackend(model: model, modelId: "tiny-hybrid")
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let tools: [[String: Any]] = [
            [
                "type": "function",
                "function": [
                    "name": "read_note", "description": "Read a note from disk",
                    "parameters": [
                        "type": "object",
                        "properties": ["path": ["type": "string"]],
                        "required": ["path"],
                    ],
                ],
            ]
        ]
        let first = try await postToolRound(
            port: server.boundPort,
            body: [
                "stream": false, "max_tokens": 1, "tools": tools,
                "messages": [
                    [
                        "role": "system",
                        "content":
                            "Use tools when needed. Return concise answers. Check every path before reading.",
                    ],
                    ["role": "user", "content": "Read note.md and summarize it."],
                ],
            ])
        let firstPrompt = try #require((first["usage"] as? [String: Any])?["prompt_tokens"] as? Int)
        #expect(firstPrompt == firstRendered.count)
        #expect((first["usage"] as? [String: Any])?["cached_tokens"] as? Int == 0)

        let second = try await postToolRound(
            port: server.boundPort,
            body: [
                "stream": false, "max_tokens": 1, "tools": tools,
                "messages": [
                    [
                        "role": "system",
                        "content":
                            "Use tools when needed. Return concise answers. Check every path before reading.",
                    ],
                    ["role": "user", "content": "Read note.md and summarize it."],
                    [
                        "role": "assistant", "content": NSNull(),
                        "tool_calls": [
                            [
                                "id": "call_exact_1", "type": "function",
                                "function": [
                                    "name": "read_note", "arguments": #"{"path":"note.md"}"#,
                                ],
                            ]
                        ],
                    ],
                    [
                        "role": "tool", "tool_call_id": "call_exact_1",
                        "content": "Prefix reuse confirmed now.",
                    ],
                ],
            ])
        let secondUsage = try #require(second["usage"] as? [String: Any])
        #expect(secondUsage["prompt_tokens"] as? Int == secondRendered.count)
        #expect(secondUsage["cached_tokens"] as? Int == firstRendered.count)
    }
}

private func loadToolRoundTokenizer() async throws -> Tokenizer {
    let fixture = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/tool-round-tokenizer")
    return try await TokenizerLoader().load(from: fixture)
}

private func postToolRound(port: UInt16, body: [String: Any]) async throws -> [String: Any] {
    var request = URLRequest(
        url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    let (data, response) = try await URLSession.shared.data(for: request)
    #expect((response as? HTTPURLResponse)?.statusCode == 200)
    return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}
