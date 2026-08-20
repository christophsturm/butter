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
// MessagesCodec : Anthropic /v1/messages request parse + response encode.
// Unknown body fields are ignored. Auth is not enforced here.

import Foundation

struct MessagesHTTPRequest: Sendable {
    var model: String?
    var messages: [ChatMessage]
    var stream: Bool
    var tools: [ChatToolDefinition]?
    var maxTokens: Int
    var temperature: Float?
    var topP: Float?

    static func parse(_ data: Data) throws -> MessagesHTTPRequest {
        let obj: [String: Any]
        do {
            guard let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw ServeError.badRequest("JSON object required")
            }
            obj = parsed
        } catch let e as ServeError {
            throw e
        } catch {
            throw ServeError.badRequest("invalid JSON")
        }

        guard obj["max_tokens"] != nil else {
            throw ServeError.badRequest("max_tokens required")
        }
        let maxTokens =
            (obj["max_tokens"] as? Int)
            ?? (obj["max_tokens"] as? NSNumber)?.intValue
        guard let maxTokens else {
            throw ServeError.badRequest("max_tokens required")
        }

        guard let rawMessages = obj["messages"] as? [Any], !rawMessages.isEmpty else {
            throw ServeError.badRequest("messages required")
        }

        var messages: [ChatMessage] = []
        if let system = obj["system"] {
            let text = extractAnthropicSystem(system)
            if !text.isEmpty {
                messages.append(ChatMessage(role: .system, content: text))
            }
        }

        for raw in rawMessages {
            guard let m = raw as? [String: Any] else { continue }
            let roleStr = (m["role"] as? String) ?? "user"
            appendAnthropicMessage(role: roleStr, content: m["content"], into: &messages)
        }
        if messages.isEmpty { throw ServeError.badRequest("messages required") }

        var tools: [ChatToolDefinition]?
        if let rawTools = obj["tools"] as? [Any] {
            var parsed: [ChatToolDefinition] = []
            for t in rawTools {
                guard let dict = t as? [String: Any],
                    let name = dict["name"] as? String
                else { continue }
                let desc = (dict["description"] as? String) ?? ""
                let paramsJSON: String
                if let schema = dict["input_schema"] ?? dict["parameters"] {
                    if let s = schema as? String {
                        paramsJSON = s
                    } else if JSONSerialization.isValidJSONObject(schema),
                        let d = try? JSONSerialization.data(withJSONObject: schema),
                        let s = String(data: d, encoding: .utf8)
                    {
                        paramsJSON = s
                    } else {
                        paramsJSON = "{}"
                    }
                } else {
                    paramsJSON = "{}"
                }
                parsed.append(
                    ChatToolDefinition(name: name, description: desc, parametersJSON: paramsJSON))
            }
            tools = parsed
        }

        let stream = obj["stream"] as? Bool ?? false
        let temperature =
            (obj["temperature"] as? NSNumber)?.floatValue ?? (obj["temperature"] as? Float)
        let topP = (obj["top_p"] as? NSNumber)?.floatValue ?? (obj["top_p"] as? Float)

        return MessagesHTTPRequest(
            model: obj["model"] as? String,
            messages: messages,
            stream: stream,
            tools: tools,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )
    }
}

enum MessagesCodec {
    static func stopReason(finish: String) -> String {
        switch finish {
        case "tool_calls": return "tool_use"
        case "length": return "max_tokens"
        default: return "end_turn"
        }
    }

    static func toolUseId(from recoveredId: String, index: Int) -> String {
        if recoveredId.hasPrefix("toolu_") { return recoveredId }
        return String(format: "toolu_%012d", index)
    }

    static func messageStartEvent(id: String, model: String) -> String {
        jsonObject([
            "type": "message_start",
            "message": [
                "id": id,
                "type": "message",
                "role": "assistant",
                "content": [] as [Any],
                "model": model,
                "stop_reason": NSNull(),
                "stop_sequence": NSNull(),
                "usage": ["input_tokens": 0, "output_tokens": 0],
            ] as [String: Any],
        ])
    }

    static func contentBlockStartText(index: Int) -> String {
        jsonObject([
            "type": "content_block_start",
            "index": index,
            "content_block": ["type": "text", "text": ""],
        ])
    }

    static func contentBlockStartToolUse(index: Int, id: String, name: String) -> String {
        jsonObject([
            "type": "content_block_start",
            "index": index,
            "content_block": [
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": [:] as [String: Any],
            ] as [String: Any],
        ])
    }

    static func contentBlockDeltaText(index: Int, text: String) -> String {
        jsonObject([
            "type": "content_block_delta",
            "index": index,
            "delta": ["type": "text_delta", "text": text],
        ])
    }

    static func contentBlockDeltaInputJSON(index: Int, partialJSON: String) -> String {
        jsonObject([
            "type": "content_block_delta",
            "index": index,
            "delta": ["type": "input_json_delta", "partial_json": partialJSON],
        ])
    }

    static func contentBlockStop(index: Int) -> String {
        jsonObject(["type": "content_block_stop", "index": index])
    }

    static func messageDeltaEvent(stopReason: String, outputTokens: Int) -> String {
        jsonObject([
            "type": "message_delta",
            "delta": [
                "stop_reason": stopReason,
                "stop_sequence": NSNull(),
            ] as [String: Any],
            "usage": ["output_tokens": outputTokens],
        ])
    }

    static func messageStopEvent() -> String {
        jsonObject(["type": "message_stop"])
    }

    static func encodeMessage(
        id: String, model: String,
        text: String?, toolCalls: [ParsedToolCall]?,
        stopReason: String,
        inputTokens: Int, outputTokens: Int
    ) -> String {
        var content: [[String: Any]] = []
        if let text, !text.isEmpty {
            content.append(["type": "text", "text": text])
        }
        if let toolCalls {
            for (i, call) in toolCalls.enumerated() {
                let inputObj: Any
                if let data = call.arguments.data(using: .utf8),
                    let parsed = try? JSONSerialization.jsonObject(with: data)
                {
                    inputObj = parsed
                } else {
                    inputObj = [String: Any]()
                }
                content.append([
                    "type": "tool_use",
                    "id": toolUseId(from: call.id, index: i),
                    "name": call.name,
                    "input": inputObj,
                ])
            }
        }
        if content.isEmpty {
            content.append(["type": "text", "text": ""])
        }
        return jsonObject([
            "id": id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": model,
            "stop_reason": stopReason,
            "stop_sequence": NSNull(),
            "usage": [
                "input_tokens": inputTokens,
                "output_tokens": outputTokens,
            ],
        ])
    }
}

// MARK: - Request helpers

private func extractAnthropicSystem(_ value: Any) -> String {
    if let s = value as? String { return s }
    return extractContent(value)
}

private func appendAnthropicMessage(role: String, content: Any?, into messages: inout [ChatMessage]) {
    let mappedRole: ChatMessage.Role
    switch role {
    case "assistant": mappedRole = .assistant
    case "tool": mappedRole = .tool
    default: mappedRole = .user
    }

    if let s = content as? String {
        messages.append(ChatMessage(role: mappedRole, content: s))
        return
    }
    guard let parts = content as? [Any] else {
        messages.append(ChatMessage(role: mappedRole, content: ""))
        return
    }

    var textBuf = ""
    func flushText() {
        if !textBuf.isEmpty {
            messages.append(ChatMessage(role: mappedRole, content: textBuf))
            textBuf = ""
        }
    }

    for part in parts {
        guard let d = part as? [String: Any] else {
            if let s = part as? String { textBuf += s }
            continue
        }
        let type = (d["type"] as? String) ?? "text"
        switch type {
        case "text":
            textBuf += (d["text"] as? String) ?? ""
        case "tool_result":
            flushText()
            let body = extractContent(d["content"])
            messages.append(ChatMessage(role: .tool, content: body))
        case "tool_use":
            flushText()
            let name = (d["name"] as? String) ?? ""
            let inputJSON: String
            if let input = d["input"] {
                if let s = input as? String {
                    inputJSON = s
                } else if JSONSerialization.isValidJSONObject(input),
                    let data = try? JSONSerialization.data(withJSONObject: input),
                    let s = String(data: data, encoding: .utf8)
                {
                    inputJSON = s
                } else {
                    inputJSON = "{}"
                }
            } else {
                inputJSON = "{}"
            }
            // No structured tool_use on ChatMessage : fold into assistant text.
            messages.append(ChatMessage(role: .assistant, content: "\(name) \(inputJSON)"))
        default:
            if let t = d["text"] as? String { textBuf += t }
        }
    }
    flushText()
}
