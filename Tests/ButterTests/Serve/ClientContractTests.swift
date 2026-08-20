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
// Completions and Messages HTTP contracts against a stub backend.

import Foundation
import Testing

@testable import ButterSwift

@Suite("Serve clients")
struct ClientContractTests {

    // MARK: - Chat Completions

    @Test("completions stream ends with finish, usage, then done")
    func completionsStreamHasFinishThenUsageThenDone() async throws {
        let system = String(repeating: "S", count: 2048)
        let stats = contractStats(prompt: 8, generated: 3)
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: "hello", tokens: [1], position: 1),
                GenerationChunk(text: "!", tokens: [2], position: 2),
                GenerationChunk(text: "", tokens: [], position: 2, stats: stats),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let body = """
            {"model":"stub-model","stream":true,"stream_options":{"include_usage":true},\
            "messages":[{"role":"system","content":\(jsonString(system))},\
            {"role":"user","content":[{"type":"text","text":"reminder"},{"type":"text","text":"hi"}]}],\
            "tools":[{"type":"function","function":{"name":"read","description":"r",\
            "parameters":{"type":"object","properties":{"path":{"type":"string"},"i":{"type":"integer"}},\
            "required":["path","i"],"additionalProperties":false}}}]}
            """
        let resp = try await contractPost(
            "http://127.0.0.1:\(server.boundPort)/v1/chat/completions",
            json: body,
            headers: ["Authorization": "Bearer dummy"])
        #expect(resp.status == 200)
        let text = String(data: resp.body, encoding: .utf8) ?? ""
        #expect(text.contains("data: [DONE]"), "body was: \(text)")

        var ids = Set<String>()
        var sawFinish = false
        var sawUsageAfterFinish = false
        var sawFinishAlready = false
        for line in text.split(whereSeparator: \.isNewline) {
            let line = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6)).trimmingCharacters(in: .whitespacesAndNewlines)
            if payload == "[DONE]" || payload.isEmpty { continue }
            guard let data = payload.data(using: .utf8),
                let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            if let id = obj["id"] as? String { ids.insert(id) }
            let choices = obj["choices"] as? [Any] ?? []
            if let first = choices.first as? [String: Any],
                let fr = first["finish_reason"] as? String,
                fr == "stop" || fr == "tool_calls"
            {
                sawFinish = true
                sawFinishAlready = true
            }
            if choices.isEmpty, obj["usage"] != nil, sawFinishAlready {
                sawUsageAfterFinish = true
            }
        }
        #expect(ids.count == 1, "same id on every chunk; got \(ids)")
        #expect(sawFinish, "need finish_reason stop|tool_calls; body: \(text)")
        #expect(sawUsageAfterFinish, "usage-only chunk after finish; body: \(text)")
    }

    @Test("xml leak becomes tool_calls")
    func xmlLeakBecomesToolCalls() async throws {
        let xml = """
            <tool_call>
            <function=read>
            <parameter=path>
            /tmp/x
            </parameter>
            <parameter=i>
            1
            </parameter>
            </function>
            </tool_call>
            """
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: xml, tokens: [1, 2, 3], position: 3),
                GenerationChunk(text: "", tokens: [], position: 3, stats: contractStats(prompt: 2, generated: 3)),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let body = """
            {"model":"stub-model","stream":true,"messages":[{"role":"user","content":"go"}],\
            "tools":[{"type":"function","function":{"name":"read","description":"r",\
            "parameters":{"type":"object","properties":{"path":{"type":"string"},"i":{"type":"integer"}},\
            "required":["path","i"],"additionalProperties":false}}}]}
            """
        let resp = try await contractPost(
            "http://127.0.0.1:\(server.boundPort)/v1/chat/completions", json: body)
        #expect(resp.status == 200)
        let text = String(data: resp.body, encoding: .utf8) ?? ""
        #expect(text.contains("tool_calls"), "body: \(text)")
        #expect(
            text.contains("\"finish_reason\":\"tool_calls\"")
                || text.contains("\"finish_reason\": \"tool_calls\""),
            "body: \(text)")
        // Client must not be forced to parse XML from content alone.
        var sawStructuredName = false
        for line in text.split(whereSeparator: \.isNewline) {
            let line = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6))
            guard let data = payload.data(using: .utf8),
                let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let choices = obj["choices"] as? [[String: Any]],
                let delta = choices.first?["delta"] as? [String: Any],
                let tcs = delta["tool_calls"] as? [[String: Any]],
                let fn = tcs.first?["function"] as? [String: Any],
                let name = fn["name"] as? String, name == "read",
                let args = fn["arguments"] as? String
            else { continue }
            #expect(args.contains("path") || args.contains("/tmp"))
            sawStructuredName = true
        }
        #expect(sawStructuredName, "structured delta.tool_calls missing; body: \(text)")
    }

    // MARK: - Completions with Authorization

    @Test("bearer completions non-stream")
    func bearerCompletionsNonStream() async throws {
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: "pong", tokens: [1], position: 1),
                GenerationChunk(text: "", tokens: [], position: 1, stats: contractStats(prompt: 2, generated: 1)),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let body = """
            {"model":"stub-model","stream":false,\
            "messages":[{"role":"system","content":"sys"},{"role":"user","content":"ping"}]}
            """
        let resp = try await contractPost(
            "http://127.0.0.1:\(server.boundPort)/v1/chat/completions",
            json: body,
            headers: ["Authorization": "Bearer secret"])
        #expect(resp.status == 200)
        let obj = try JSONSerialization.jsonObject(with: resp.body) as? [String: Any]
        let choices = obj?["choices"] as? [[String: Any]]
        let message = choices?.first?["message"] as? [String: Any]
        #expect(message?["content"] as? String == "pong")
        #expect(choices?.first?["finish_reason"] as? String != nil)
        #expect(obj?["usage"] != nil)
    }

    @Test("json leak becomes tool_calls")
    func jsonLeakBecomesToolCalls() async throws {
        let leak =
            #"<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>"#
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: leak, tokens: [1], position: 1),
                GenerationChunk(text: "", tokens: [], position: 1, stats: contractStats(prompt: 1, generated: 1)),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let body = #"{"stream":false,"messages":[{"role":"user","content":"run"}]}"#
        let resp = try await contractPost(
            "http://127.0.0.1:\(server.boundPort)/v1/chat/completions",
            json: body,
            headers: ["Authorization": "Bearer secret"])
        #expect(resp.status == 200)
        let obj = try JSONSerialization.jsonObject(with: resp.body) as? [String: Any]
        let message =
            ((obj?["choices"] as? [[String: Any]])?.first?["message"]) as? [String: Any]
        let toolCalls = message?["tool_calls"] as? [[String: Any]]
        let fn = toolCalls?.first?["function"] as? [String: Any]
        #expect(fn?["name"] as? String == "bash")
        #expect(fn?["arguments"] is String)
        #expect((fn?["arguments"] as? String)?.contains("ls") == true)
        #expect(
            (obj?["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String
                == "tool_calls")
    }

    // MARK: - Messages

    @Test("messages stream uses event lines")
    func messagesStreamEvents() async throws {
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: "yo", tokens: [1], position: 1),
                GenerationChunk(text: "", tokens: [], position: 1, stats: contractStats(prompt: 2, generated: 1)),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let body = """
            {"model":"stub-model","max_tokens":256,"stream":true,"temperature":0.2,\
            "system":"Be terse.","messages":[{"role":"user","content":"hi"}]}
            """
        let resp = try await contractPost(
            "http://127.0.0.1:\(server.boundPort)/v1/messages",
            json: body,
            headers: [
                "anthropic-version": "2023-06-01",
                "x-api-key": "dummy",
            ])
        #expect(resp.status == 200)
        let text = String(data: resp.body, encoding: .utf8) ?? ""
        #expect(text.contains("event: message_start"), "body: \(text)")
        #expect(text.contains("event: content_block_start"), "body: \(text)")
        #expect(text.contains("text_delta"), "body: \(text)")
        #expect(text.contains("event: content_block_stop"), "body: \(text)")
        #expect(
            text.contains("\"stop_reason\":\"end_turn\"")
                || text.contains("\"stop_reason\": \"end_turn\""),
            "body: \(text)")
        #expect(text.contains("event: message_delta"), "body: \(text)")
        #expect(text.contains("event: message_stop"), "body: \(text)")
    }

    @Test("messages tool_use from json leak")
    func messagesToolUseFromJsonLeak() async throws {
        let leak =
            #"<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>"#
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: leak, tokens: [1], position: 1),
                GenerationChunk(text: "", tokens: [], position: 1, stats: contractStats(prompt: 1, generated: 1)),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let body = """
            {"model":"stub-model","max_tokens":128,"stream":true,\
            "messages":[{"role":"user","content":[{"type":"text","text":"run"}]}],\
            "tools":[{"name":"bash","description":"shell",\
            "input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}]}
            """
        let resp = try await contractPost(
            "http://127.0.0.1:\(server.boundPort)/v1/messages",
            json: body,
            headers: [
                "anthropic-version": "2023-06-01",
                "Authorization": "Bearer dummy",
            ])
        #expect(resp.status == 200)
        let text = String(data: resp.body, encoding: .utf8) ?? ""
        #expect(text.contains("tool_use"), "body: \(text)")
        #expect(
            text.contains("\"stop_reason\":\"tool_use\"")
                || text.contains("\"stop_reason\": \"tool_use\""),
            "body: \(text)")
    }

    @Test("messages requires max_tokens")
    func messagesRequiresMaxTokens() async throws {
        let server = ServeServer(
            host: "127.0.0.1", port: 0, backend: StubServeBackend(modelId: "stub-model"))
        try await server.start()
        defer { server.stop() }
        let body = #"{"model":"stub-model","messages":[{"role":"user","content":"hi"}]}"#
        let resp = try await contractPost(
            "http://127.0.0.1:\(server.boundPort)/v1/messages", json: body)
        #expect(resp.status == 400)
    }

    @Test("messages ignores unknown fields")
    func messagesIgnoresUnknownFields() async throws {
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: "ok", tokens: [1], position: 1),
                GenerationChunk(text: "", tokens: [], position: 1, stats: contractStats(prompt: 1, generated: 1)),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }

        let body = """
            {"model":"stub-model","max_tokens":32,"stream":false,\
            "thinking":{"type":"enabled"},"metadata":{"user_id":"u"},\
            "cache_control":{"type":"ephemeral"},"tool_choice":"auto","strict":true,\
            "messages":[{"role":"user","content":"hi"}]}
            """
        let resp = try await contractPost(
            "http://127.0.0.1:\(server.boundPort)/v1/messages", json: body)
        #expect(resp.status == 200)
    }

    @Test("models list includes context_length")
    func modelsListHasContextLength() async throws {
        let backend = StubServeBackend(modelId: "stub-model", maxModelLen: 8192)
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(server.boundPort)/v1/models")!)
        req.httpMethod = "GET"
        let (data, resp) = try await URLSession.shared.data(for: req)
        #expect((resp as! HTTPURLResponse).statusCode == 200)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let entry = (obj?["data"] as? [[String: Any]])?.first
        #expect(entry?["id"] as? String == "stub-model")
        let hasLen =
            entry?["max_model_len"] != nil || entry?["context_length"] != nil
        #expect(hasLen)
    }
}

// MARK: - Helpers

private func contractStats(prompt: Int, generated: Int) -> GenerationStats {
    GenerationStats(
        promptTokens: prompt, generatedTokens: generated, contextSize: 0,
        prefillTimeS: 0, decodeTimeS: 0, timeToFirstTokenMs: 0,
        steadyTokensPerSecond: nil,
        baselineGPUBytes: 0, postPrefillGPUBytes: 0, postDecodeGPUBytes: 0,
        prefillPeakGPUBytes: 0, decodePeakGPUBytes: 0,
        wiredTicketBytes: 0, weightsBytes: 0,
        kvCacheAllocatedBytes: 0, kvCacheUsedBytes: 0,
        thinkPerplexity: nil, genPerplexity: nil,
        thinkKLDivergence: nil, genKLDivergence: nil,
        thinkTokenCount: nil, genTokenCount: nil)
}

private struct ContractHTTP {
    var status: Int
    var body: Data
}

private func contractPost(
    _ url: String, json: String, headers: [String: String] = [:]
) async throws -> ContractHTTP {
    var req = URLRequest(url: URL(string: url)!)
    req.httpMethod = "POST"
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    for (k, v) in headers { req.setValue(v, forHTTPHeaderField: k) }
    req.httpBody = json.data(using: .utf8)
    let (data, resp) = try await URLSession.shared.data(for: req)
    return ContractHTTP(status: (resp as! HTTPURLResponse).statusCode, body: data)
}

private func jsonString(_ s: String) -> String {
    let data = try! JSONSerialization.data(withJSONObject: [s])
    let wrapped = String(data: data, encoding: .utf8)!
    // "[\"...\"]" → "\"...\""
    return String(wrapped.dropFirst().dropLast())
}
