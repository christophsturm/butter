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
// ChatTemplateTests — type shape (ChatMessage / ChatTemplateOptions /
// ReasoningEffort / ChatTemplateError). End-to-end render-through-the-
// tokenizer is exercised by the integration tests with real models.

import Foundation
import Testing

@testable import ButterSwift

@Suite("ChatTemplate types")
struct ChatTemplateTests {

    @Test("ChatMessage.Role rawValues are stable")
    func roleRawValues() {
        #expect(ChatMessage.Role.system.rawValue == "system")
        #expect(ChatMessage.Role.user.rawValue == "user")
        #expect(ChatMessage.Role.assistant.rawValue == "assistant")
        #expect(ChatMessage.Role.tool.rawValue == "tool")
    }

    @Test("ChatMessage round-trips through asTemplateMessage")
    func asTemplateMessage() {
        let m1 = ChatMessage(role: .user, content: "Hi")
        let dict1 = m1.asTemplateMessage
        #expect(dict1["role"] as? String == "user")
        #expect(dict1["content"] as? String == "Hi")
        #expect(dict1["thinking"] == nil)

        let m2 = ChatMessage(
            role: .assistant, content: "Hello",
            thinking: "<reasoning here>")
        let dict2 = m2.asTemplateMessage
        #expect(dict2["thinking"] as? String == "<reasoning here>")
        #expect(dict2["reasoning_content"] as? String == "<reasoning here>")

        let call = ChatToolCall(
            id: "call_1", name: "read",
            arguments: #"{"path":"/tmp/x","lines":2}"#)
        let m3 = ChatMessage(
            role: .assistant, content: "", toolCalls: [call])
        let dict3 = m3.asTemplateMessage
        let calls = dict3["tool_calls"] as? [[String: any Sendable]]
        let function = calls?.first?["function"] as? [String: any Sendable]
        let arguments = function?["arguments"] as? [String: any Sendable]
        #expect(calls?.first?["id"] as? String == "call_1")
        #expect(function?["name"] as? String == "read")
        #expect(arguments?["path"] as? String == "/tmp/x")
        #expect(arguments?["lines"] as? Int == 2)
    }

    @Test("ChatMessage Equatable")
    func messageEquatable() {
        let a = ChatMessage(role: .user, content: "Hi")
        let b = ChatMessage(role: .user, content: "Hi")
        #expect(a == b)
        let c = ChatMessage(role: .user, content: "Hi", thinking: "x")
        #expect(a != c)
    }

    @Test("ChatTemplateOptions defaults")
    func optionsDefaults() {
        let o = ChatTemplateOptions()
        #expect(o.addGenerationPrompt == true)
        #expect(o.enableThinking == false)
        #expect(o.reasoningEffort == nil)
        #expect(o.maxLength == nil)
        #expect(o.truncation == false)
        #expect(o.extraContext.isEmpty)
        #expect(o.tools == nil)
    }

    @Test("ChatTemplateOptions custom init + Equatable on typed fields")
    func optionsCustomInit() {
        let o = ChatTemplateOptions(
            addGenerationPrompt: false,
            enableThinking: true,
            reasoningEffort: .high,
            maxLength: 4096,
            truncation: true
        )
        #expect(o.addGenerationPrompt == false)
        #expect(o.enableThinking == true)
        #expect(o.reasoningEffort == .high)
        #expect(o.maxLength == 4096)
        #expect(o.truncation == true)

        let same = ChatTemplateOptions(
            addGenerationPrompt: false, enableThinking: true,
            reasoningEffort: .high, maxLength: 4096, truncation: true
        )
        #expect(o == same)

        let different = ChatTemplateOptions(
            enableThinking: true,
            reasoningEffort: .low)
        #expect(o != different)
    }

    @Test("Serve ChatToolDefinition template shape is OpenAI function")
    func toolDefinitionTemplateShape() {
        let tool = ChatToolDefinition(
            name: "read",
            description: "Read a file",
            parametersJSON: #"{"type":"object","properties":{"path":{"type":"string"}}}"#)
        let dict = tool.asTemplateTool
        #expect(dict["type"] as? String == "function")
        let fn = dict["function"] as? [String: any Sendable]
        #expect(fn?["name"] as? String == "read")
        #expect(fn?["description"] as? String == "Read a file")
        let params = fn?["parameters"] as? [String: any Sendable]
        #expect(params?["type"] as? String == "object")
    }

    @Test("ReasoningEffort rawValues align with Harmony convention")
    func reasoningEffort() {
        #expect(ReasoningEffort.low.rawValue == "low")
        #expect(ReasoningEffort.medium.rawValue == "medium")
        #expect(ReasoningEffort.high.rawValue == "high")
    }

    @Test("ChatTemplateError descriptions are non-empty")
    func errorDescriptions() {
        let e1 = ChatTemplateError.noTemplateOnTokenizer
        #expect(!String(describing: e1).isEmpty)
        struct Boom: Error {}
        let e2 = ChatTemplateError.renderFailed(Boom())
        #expect(String(describing: e2).contains("Boom"))
    }

    @Test("ThinkingFormat — every case has a non-empty rawValue")
    func thinkingFormatRawValues() {
        for f in ThinkingFormat.allCases {
            #expect(!f.rawValue.isEmpty)
        }
    }
}
