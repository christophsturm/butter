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
// One resident model: same id does not reload; A->B drops A; bad id
// keeps A; swap during generate aborts leftover A work.

import Foundation
import os
import Testing

@testable import ButterSwift

@Suite("Serve resident swap", .serialized)
struct ResidentSwapTests {

    @Test("same model twice does not reload")
    func sameModelDoesNotReload() async throws {
        let backend = SwapStubBackend(initial: "model-a")
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        _ = try await swapPost(port: server.boundPort, model: "model-a", content: "one")
        _ = try await swapPost(port: server.boundPort, model: "model-a", content: "two")
        #expect(backend.loadCount == 1)
        #expect(backend.modelId == "model-a")
    }

    @Test("swap A to B drops A session")
    func swapDropsPreviousSession() async throws {
        let backend = SwapStubBackend(initial: "model-a")
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        _ = try await swapPost(port: server.boundPort, model: "model-a", content: "aa")
        let b = try await swapPost(port: server.boundPort, model: "model-b", content: "bb")
        #expect(backend.modelId == "model-b")
        #expect(backend.loadCount == 2)
        #expect(backend.dropped.contains("model-a"))
        #expect(b.contains("model-b") || b.contains("ok"))
        let listed = try await swapModels(port: server.boundPort)
        #expect(listed.contains("model-b"))
        #expect(!listed.contains("model-a"))
    }

    @Test("swap during generate aborts leftover A")
    func swapDuringGenerateAbortsA() async throws {
        let backend = SwapStubBackend(initial: "model-a", slowA: true)
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let streamTask = Task {
            _ = try? await swapPost(
                port: server.boundPort, model: "model-a", content: "slow", stream: true)
        }
        try await waitUntil(timeout: 4) { backend.aDidStart }
        _ = try await swapPost(port: server.boundPort, model: "model-b", content: "fast")
        try await Task.sleep(for: .milliseconds(300))
        #expect(backend.aCancelled)
        #expect(backend.modelId == "model-b")
        streamTask.cancel()
    }

    @Test("bad id keeps previous resident")
    func badIdKeepsResident() async throws {
        let backend = SwapStubBackend(initial: "model-a")
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        _ = try await swapPost(port: server.boundPort, model: "model-a", content: "hi")
        let resp = try await swapPostRaw(
            port: server.boundPort, model: "missing", content: "nope")
        #expect(resp.status == 404 || resp.status == 400)
        #expect(backend.modelId == "model-a")
        #expect(backend.loadCount == 1)
        let listed = try await swapModels(port: server.boundPort)
        #expect(listed.contains("model-a"))
    }
}

private func swapPost(
    port: UInt16, model: String, content: String, stream: Bool = false
) async throws -> String {
    let raw = try await swapPostRaw(port: port, model: model, content: content, stream: stream)
    return String(data: raw.body, encoding: .utf8) ?? ""
}

private struct SwapHTTP {
    var status: Int
    var body: Data
}

private func swapPostRaw(
    port: UInt16, model: String, content: String, stream: Bool = false
) async throws -> SwapHTTP {
    var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
    req.httpMethod = "POST"
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.httpBody = Data(
        #"{"model":"\#(model)","stream":\#(stream),"messages":[{"role":"user","content":"\#(content)"}]}"#
            .utf8)
    let (data, resp) = try await URLSession.shared.data(for: req)
    return SwapHTTP(status: (resp as! HTTPURLResponse).statusCode, body: data)
}

private func swapModels(port: UInt16) async throws -> String {
    let (data, _) = try await URLSession.shared.data(
        from: URL(string: "http://127.0.0.1:\(port)/v1/models")!)
    return String(data: data, encoding: .utf8) ?? ""
}

private func waitUntil(timeout: TimeInterval, _ pred: @escaping () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if pred() { return }
        try await Task.sleep(for: .milliseconds(20))
    }
    throw SwapWaitTimeout()
}

private struct SwapWaitTimeout: Error {}

final class SwapStubBackend: ServeGenerationBackend, @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock(initialState: SwapState())
    private let slowA: Bool
    private let aSlow: SlowStubBackend?

    init(initial: String, slowA: Bool = false) {
        self.slowA = slowA
        let slow = slowA ? SlowStubBackend() : nil
        self.aSlow = slow
        lock.withLock {
            $0.modelId = initial
            $0.inner = SwapStubBackend.makeInner(id: initial, slow: slow)
        }
    }

    var modelId: String { lock.withLock { $0.modelId } }
    var loadCount: Int { lock.withLock { $0.loadCount } }
    var dropped: [String] { lock.withLock { $0.dropped } }
    var aDidStart: Bool { aSlow?.didStart ?? false }
    var aCancelled: Bool { aSlow?.cancelled ?? false }
    var maxModelLen: Int? { 128 }
    var lastCachedTokens: Int { 0 }
    var defaultGenerationParameters = GenerationParameters()

    func abort() { lock.withLock { $0.inner.abort() } }

    func prepare(requestedModel: String?) throws {
        try lock.withLock { s in
            let want = requestedModel ?? s.modelId
            if want == s.modelId { return }
            if want == "missing" { throw ServeError.notFound }
            s.inner.abort()
            s.dropped.append(s.modelId)
            s.modelId = want
            s.loadCount += 1
            s.inner = SwapStubBackend.makeInner(id: want, slow: aSlow)
        }
    }

    func completeChat(
        messages: [ChatMessage],
        tools: [ChatToolDefinition]?,
        parameters: GenerationParameters,
        templateOptions: ChatTemplateOptions
    ) throws -> AsyncThrowingStream<GenerationChunk, Error> {
        try lock.withLock {
            try $0.inner.completeChat(
                messages: messages, tools: tools, parameters: parameters,
                templateOptions: templateOptions)
        }
    }

    private static func makeInner(id: String, slow: SlowStubBackend?) -> any ServeGenerationBackend {
        if id == "model-a", let slow { return slow }
        return SessionStubBackend(modelId: id)
    }
}

private struct SwapState {
    var modelId = "model-a"
    var loadCount = 1
    var dropped: [String] = []
    var inner: any ServeGenerationBackend = SessionStubBackend(modelId: "model-a")
}
