// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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

import Foundation
import TestHelpers
import Testing
import Tokenizers

@testable import ButterSwift

@Suite(
    "Qwen3.8 tool-round prompt integration", .serialized,
    .enabled(if: modelTestsOptedIn))
struct Qwen38ToolRoundPromptIntegrationTests {
    @Test("checkpoint tokenizer renders the 277 to 325 matched prefix")
    func checkpointTokenizerRendersExactMatchedPrefix() async throws {
        let environment = ProcessInfo.processInfo.environment
        let path = try #require(
            environment["BUTTER_QWEN38_MODEL_PATH"],
            "set BUTTER_QWEN38_MODEL_PATH to the Qwen3.8 27B checkpoint directory")
        #expect(FileManager.default.fileExists(atPath: path))

        let tokenizer = try await TokenizerLoader().load(
            from: URL(fileURLWithPath: path, isDirectory: true))
        let tools = [
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
            ChatMessage(role: .user, content: "Read note.md and summarize it."),
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

        let first = try render(
            tokenizer: tokenizer, messages: firstMessages, tools: tools)
        let second = try render(
            tokenizer: tokenizer, messages: secondMessages, tools: tools)
        let missingCalls = try render(
            tokenizer: tokenizer,
            messages: firstMessages + [
                ChatMessage(role: .assistant, content: ""),
                secondMessages[3],
            ], tools: tools)

        #expect(first.count == 277, "first=\(first.count)")
        #expect(second.count == 325, "second=\(second.count)")
        #expect(second.starts(with: first))
        #expect(missingCalls != second)

        let model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(
                path,
                options: LoadOptions(prewarm: false, maxContextLength: 2048))
        }
        let backend = ButterServeBackend(model: model, modelId: "qwen38-exact")
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let requestTools: [[String: Any]] = [
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
        let firstResponse = try await post(
            port: server.boundPort,
            body: [
                "stream": false, "max_tokens": 1, "tools": requestTools,
                "messages": [
                    [
                        "role": "system",
                        "content":
                            "Use tools when needed. Return concise answers. Check every path before reading.",
                    ],
                    ["role": "user", "content": "Read note.md and summarize it."],
                ],
            ])
        let firstUsage = try #require(firstResponse["usage"] as? [String: Any])
        #expect(firstUsage["prompt_tokens"] as? Int == 277)
        #expect(firstUsage["cached_tokens"] as? Int == 0)

        let secondResponse = try await post(
            port: server.boundPort,
            body: [
                "stream": false, "max_tokens": 1, "tools": requestTools,
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
        let secondUsage = try #require(secondResponse["usage"] as? [String: Any])
        #expect(secondUsage["prompt_tokens"] as? Int == 325)
        #expect(secondUsage["cached_tokens"] as? Int == 277)
    }

    private func render(
        tokenizer: Tokenizer, messages: [ChatMessage], tools: [ChatToolDefinition]
    ) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: messages.map(\.asTemplateMessage),
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: tools.map(\.asTemplateTool),
            additionalContext: ["enable_thinking": false]
        )
    }

    private func post(port: UInt16, body: [String: Any]) async throws -> [String: Any] {
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
