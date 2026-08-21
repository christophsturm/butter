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

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("Serve")
struct HTTPServerTests {

    @Test("JSON leak recovers structured call")
    func jsonMarkupRecovers() {
        let content =
            #"<tool_call>{"name": "bash", "arguments": {"command": "ls /tmp/sweep-test"}}</tool_call>"#
        let recovered = ToolCallParser.recover(from: content)
        #expect(recovered != nil)
        #expect(recovered?.calls.count == 1)
        #expect(recovered?.calls.first?.name == "bash")
        #expect(recovered?.calls.first?.arguments.contains("sweep-test") == true)
        #expect(recovered?.residual == "")
    }

    @Test("Qwen3-coder XML leak recovers structured call")
    func qwen3CoderXML() {
        let content = """

            <tool_call>
            <function=bash>
            <parameter=command>
            ls -l /tmp/nemotron-test
            </parameter>
            </function>
            </tool_call>
            """
        let recovered = ToolCallParser.recover(from: content)
        #expect(recovered != nil)
        #expect(recovered?.calls.first?.name == "bash")
        #expect(recovered?.calls.first?.arguments.contains("nemotron-test") == true)
    }

    @Test("markup inside think is not a tool call")
    func thinkBlockHidesMarkup() {
        let inner =
            #"<tool_call>{"name":"bash","arguments":{"command":"rm"}}</tool_call>"#
        let content = "<think>\nplanning \(inner)\n</think>\njust a reply"
        #expect(ToolCallParser.recover(from: content) == nil)
        #expect(ToolCallParser.stripThinkBlocks(content).contains("just a reply"))
    }

    @Test("commentary mentioning a tag does not recover")
    func commentaryDoesNotRecover() {
        let content =
            "I would call <tool_call> if I had tools, but I'll just answer instead. "
            + String(repeating: "x", count: 80)
        #expect(ToolCallParser.recover(from: content) == nil)
    }

    @Test("GET /health and /v1/models")
    func healthAndModels() async throws {
        let backend = StubServeBackend(modelId: "stub-model", maxModelLen: 4096)
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let port = server.boundPort
        #expect(port > 0)

        let health = try await get("http://127.0.0.1:\(port)/health")
        #expect(health.status == 200)
        let cors = health.headers.first { $0.key.lowercased() == "access-control-allow-origin" }?.value
        #expect(cors == "*")

        let models = try await get("http://127.0.0.1:\(port)/v1/models")
        #expect(models.status == 200)
        let obj = try JSONSerialization.jsonObject(with: models.body) as? [String: Any]
        let data = obj?["data"] as? [[String: Any]]
        #expect(data?.first?["id"] as? String == "stub-model")
        #expect(data?.first?["max_model_len"] as? Int == 4096)
    }

    @Test("SSE finish_reason is present and usage chunk follows when requested")
    func sseFinishAndUsage() async throws {
        let stats = stubStats(prompt: 4, generated: 2)
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: "hello", tokens: [11], position: 5),
                GenerationChunk(text: " world", tokens: [12], position: 6),
                GenerationChunk(text: "", tokens: [], position: 6, stats: stats),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let port = server.boundPort

        let body = """
            {"model":"stub-model","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}
            """
        let resp = try await post(
            "http://127.0.0.1:\(port)/v1/chat/completions", json: body)
        #expect(resp.status == 200)
        let text = String(data: resp.body, encoding: .utf8) ?? ""
        #expect(text.contains("data: [DONE]"))

        var sawFinish = false
        var sawUsage = false
        var sawContent = false
        for line in text.split(whereSeparator: \.isNewline) {
            let line = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6)).trimmingCharacters(in: .whitespacesAndNewlines)
            if payload.isEmpty || payload == "[DONE]" { continue }
            guard let data = payload.data(using: .utf8),
                let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            let choices = obj["choices"] as? [Any] ?? []
            if let first = choices.first as? [String: Any] {
                if let delta = first["delta"] as? [String: Any],
                    let c = delta["content"] as? String, !c.isEmpty
                {
                    sawContent = true
                }
                if let fr = first["finish_reason"] as? String, fr == "stop" {
                    sawFinish = true
                }
            }
            if choices.isEmpty, let usage = obj["usage"] as? [String: Any] {
                sawUsage = true
                #expect((usage["prompt_tokens"] as? Int) ?? (usage["prompt_tokens"] as? NSNumber)?.intValue == 4)
                #expect((usage["completion_tokens"] as? Int) ?? (usage["completion_tokens"] as? NSNumber)?.intValue == 2)
            }
        }
        #expect(sawContent, "body was: \(text)")
        #expect(sawFinish, "body was: \(text)")
        #expect(sawUsage, "body was: \(text)")
    }

    @Test("array message content is accepted")
    func arrayContent() async throws {
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: "ok", tokens: [1], position: 1),
                GenerationChunk(text: "", tokens: [], position: 1, stats: stubStats(prompt: 1, generated: 1)),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let body = """
            {"stream":false,"messages":[{"role":"user","content":[{"type":"text","text":"ping"}]}]}
            """
        let resp = try await post(
            "http://127.0.0.1:\(server.boundPort)/v1/chat/completions", json: body)
        #expect(resp.status == 200)
        let obj = try JSONSerialization.jsonObject(with: resp.body) as? [String: Any]
        let choices = obj?["choices"] as? [[String: Any]]
        let message = choices?.first?["message"] as? [String: Any]
        #expect(message?["content"] as? String == "ok")
        #expect(choices?.first?["finish_reason"] as? String == "stop")
    }
}

struct StubServeBackend: ServeGenerationBackend, Sendable {
    var modelId: String
    var maxModelLen: Int? = 2048
    var lastCachedTokens: Int = 0
    var defaultGenerationParameters: GenerationParameters = .init()
    var chunks: [GenerationChunk] = []

    func completeChat(
        messages: [ChatMessage],
        tools: [ChatToolDefinition]?,
        parameters: GenerationParameters,
        templateOptions: ChatTemplateOptions
    ) throws -> AsyncThrowingStream<GenerationChunk, Error> {
        let chunks = self.chunks
        return AsyncThrowingStream { continuation in
            let task = Task {
                for c in chunks {
                    try Task.checkCancellation()
                    continuation.yield(c)
                }
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}

private func stubStats(prompt: Int, generated: Int) -> GenerationStats {
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

private struct HTTPResult {
    var status: Int
    var headers: [String: String]
    var body: Data
}

private func get(_ url: String) async throws -> HTTPResult {
    var req = URLRequest(url: URL(string: url)!)
    req.httpMethod = "GET"
    return try await perform(req)
}

private func post(_ url: String, json: String) async throws -> HTTPResult {
    var req = URLRequest(url: URL(string: url)!)
    req.httpMethod = "POST"
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.httpBody = json.data(using: .utf8)
    return try await perform(req)
}

private func perform(_ req: URLRequest) async throws -> HTTPResult {
    let (data, resp) = try await URLSession.shared.data(for: req)
    let http = resp as! HTTPURLResponse
    var headers: [String: String] = [:]
    for (k, v) in http.allHeaderFields {
        headers["\(k)"] = "\(v)"
    }
    return HTTPResult(status: http.statusCode, headers: headers, body: data)
}

final class FakeLayerCache: LayerCacheProtocol, @unchecked Sendable {
    var length: Int
    var maxSeq: Int
    var bytesAllocated: Int { 0 }
    var bytesInUse: Int { 0 }
    init(length: Int, maxSeq: Int = 32) {
        self.length = length
        self.maxSeq = maxSeq
    }
    func reset() { length = 0 }
}

final class FakeKVCache: KVCacheProtocol, @unchecked Sendable {
    var length: Int
    let maxSeq: Int
    let nKVHeads = 1
    let headDim = 1
    let dtype: DType = .f32
    var bytesAllocated: Int { 0 }
    var bytesInUse: Int { 0 }
    var eviction: KVEviction
    var absolutePosition: Int
    var effectiveMaxSize: Int { maxSeq }

    init(
        length: Int, maxSeq: Int,
        eviction: KVEviction = .unbounded,
        absolutePosition: Int? = nil
    ) {
        self.length = length
        self.maxSeq = maxSeq
        self.eviction = eviction
        self.absolutePosition = absolutePosition ?? length
    }

    func reset() {
        length = 0
        absolutePosition = 0
    }

    func truncate(toLength length: Int) {
        self.length = length
        self.absolutePosition = length
    }

    func appendOnGPU(kFlat: Tensor, vFlat: Tensor, on cmd: MTLCommandBuffer) {
        length += 1
        absolutePosition += 1
    }

    func prepareForAttention(on cmd: MTLCommandBuffer) -> (k: Tensor, v: Tensor) {
        preconditionFailure("FakeKVCache is not used for attention")
    }
}
