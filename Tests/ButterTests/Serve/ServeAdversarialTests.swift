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
// Serve adversarial checks: cache reuse, HTTP leftovers, parser false recover,
// and single-flight admit.

import Foundation
import Network
import Testing

@testable import ButterSwift

@Suite("Serve adversarial")
struct ServeAdversarialTests {

    // MARK: - Charter 1 : cache reuse (Risk / Product)
    // Imagine: hybrid serve always rebuilds, or rewind keeps stale GDN.

    @Test("nPast decrements when new is fully cached : mutation: drop the -1")
    func nPastWhenEqual() {
        #expect(GenerationSession.nPast(cached: [1, 2, 3], new: [1, 2, 3]) == 2)
        #expect(GenerationSession.nPast(cached: [1, 2, 3, 4], new: [1, 2, 3]) == 2)
        #expect(GenerationSession.nPast(cached: [1, 2], new: [1, 9]) == 1)
        #expect(GenerationSession.nPast(cached: [], new: [1, 2]) == 0)
    }

    @Test("append-only keeps recurrent : mutation: require every layer be KV")
    func prepareKeepsRecurrentOnAppend() {
        let layer = FakeLayerCache(length: 5)
        let session = GenerationSession(makeCaches: { [FakeLayerCache(length: 0)] })
        session.testingReplaceCaches([layer], tokens: [1, 2, 3, 4, 5])
        let n = session.prepare(for: [1, 2, 3, 4, 5, 6])
        #expect(n == 5)
        #expect(layer.length == 5)
        #expect(session.snapshotCachedTokens() == [1, 2, 3, 4, 5])
    }

    @Test("hybrid append keeps KV + recurrent : Product (Qwen-class serve)")
    func prepareHybridAppendKeepsBoth() {
        let kv = FakeKVCache(length: 4, maxSeq: 32)
        let layer = FakeLayerCache(length: 4)
        let session = GenerationSession(makeCaches: {
            [FakeKVCache(length: 0, maxSeq: 32), FakeLayerCache(length: 0)]
        })
        session.testingReplaceCaches([kv, layer], tokens: [1, 2, 3, 4])
        let n = session.prepare(for: [1, 2, 3, 4, 5, 6])
        #expect(n == 4)
        #expect(kv.length == 4)
        #expect(layer.length == 4)
    }

    @Test("recurrent rewind rebuilds : mutation: keep GDN when nPast < length")
    func prepareRebuildsRecurrentOnRewind() {
        let layer = FakeLayerCache(length: 5)
        let session = GenerationSession(makeCaches: { [FakeLayerCache(length: 0)] })
        session.testingReplaceCaches([layer], tokens: [1, 2, 3, 4, 5])
        let n = session.prepare(for: [1, 2])
        #expect(n == 0)
        #expect(session.snapshotCachedTokens().isEmpty)
    }

    @Test("equal prompt is a rewind : Product (nPast-1 vs GDN length)")
    func prepareEqualPromptRebuildsRecurrent() {
        let layer = FakeLayerCache(length: 5)
        let session = GenerationSession(makeCaches: { [FakeLayerCache(length: 0)] })
        session.testingReplaceCaches([layer], tokens: [1, 2, 3, 4, 5])
        let n = session.prepare(for: [1, 2, 3, 4, 5])
        #expect(n == 0)
        #expect(session.snapshotCachedTokens().isEmpty)
    }

    @Test("KV equal-prompt truncates to nPast : Product")
    func prepareTruncates() {
        let kv = FakeKVCache(length: 5, maxSeq: 16)
        let session = GenerationSession(makeCaches: { [FakeKVCache(length: 0, maxSeq: 16)] })
        session.testingReplaceCaches([kv], tokens: [1, 2, 3, 4, 5])
        let n = session.prepare(for: [1, 2, 3, 4, 5])
        #expect(n == 4)
        #expect(kv.length == 4)
        #expect(session.snapshotCachedTokens() == [1, 2, 3, 4])
    }

    @Test("rotated window cannot truncate : Product")
    func prepareRebuildsRotatedWindow() {
        let kv = FakeKVCache(
            length: 4, maxSeq: 8,
            eviction: .window(maxSize: 4, keep: 0),
            absolutePosition: 10)
        let session = GenerationSession(makeCaches: {
            [FakeKVCache(length: 0, maxSeq: 8)]
        })
        session.testingReplaceCaches([kv], tokens: [1, 2, 3, 4])
        let n = session.prepare(for: [1, 2])
        #expect(n == 0)
        #expect(session.snapshotCachedTokens().isEmpty)
    }

    // MARK: - Charter 2 : HTTP/1.1 a Completions client actually sends
    // Imagine: leftover unread, missing CL treated as empty, chunked ignored.

    @Test("pipelined GET leftover is answered : Standards; instrument known-bad = no stub-model")
    func pipelinedGets() async throws {
        let backend = StubServeBackend(modelId: "stub-model", maxModelLen: 128)
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let req =
            "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
            + "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
        let raw = try await rawTCP(port: server.boundPort, send: req) {
            $0.contains("ok") && $0.contains("stub-model")
        }
        #expect(raw.contains("ok"))
        #expect(raw.contains("stub-model"))
    }

    @Test("POST without Content-Length reads until close : World")
    func postUntilClose() async throws {
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: "ok", tokens: [1], position: 1),
                GenerationChunk(text: "", tokens: [], position: 1, stats: rstStubStats()),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let json = #"{"stream":false,"messages":[{"role":"user","content":"hi"}]}"#
        let req =
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            + "Content-Type: application/json\r\nConnection: close\r\n\r\n" + json
        let raw = try await rawTCP(
            port: server.boundPort, send: req, closeAfterSend: true
        ) { $0.contains("finish_reason") }
        #expect(raw.contains("finish_reason"))
        #expect(raw.contains("ok"))
    }

    @Test("chunked POST still completes : World; mutation: treat missing CL as empty body")
    func chunkedPost() async throws {
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: "ok", tokens: [1], position: 1),
                GenerationChunk(text: "", tokens: [], position: 1, stats: rstStubStats()),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let json = #"{"stream":false,"messages":[{"role":"user","content":"hi"}]}"#
        let bytes = Array(json.utf8)
        let mid = 20
        func chunk(_ part: ArraySlice<UInt8>) -> String {
            String(part.count, radix: 16) + "\r\n" + String(bytes: part, encoding: .utf8)! + "\r\n"
        }
        let req =
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            + "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            + chunk(bytes[..<mid]) + chunk(bytes[mid...]) + "0\r\n\r\n"
        let raw = try await rawTCP(port: server.boundPort, send: req) {
            $0.contains("finish_reason")
        }
        #expect(raw.contains("finish_reason"))
        #expect(raw.contains("ok"))
    }

    @Test("invalid JSON is 400 not 500 : Claims; mutation: let JSONSerialization throw")
    func invalidJSON() async throws {
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: StubServeBackend(modelId: "x"))
        try await server.start()
        defer { server.stop() }
        let resp = try await rstPost(
            "http://127.0.0.1:\(server.boundPort)/v1/chat/completions",
            json: "{not-json")
        #expect(resp.status == 400)
    }

    @Test("empty messages is 400 : Claims")
    func emptyMessages() async throws {
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: StubServeBackend(modelId: "x"))
        try await server.start()
        defer { server.stop() }
        let resp = try await rstPost(
            "http://127.0.0.1:\(server.boundPort)/v1/chat/completions",
            json: #"{"messages":[]}"#)
        #expect(resp.status == 400)
    }

    @Test("SSE finish_reason is the string stop : Claims; mutation: assert only non-empty")
    func sseFinishIsStop() async throws {
        let backend = StubServeBackend(
            modelId: "stub-model",
            chunks: [
                GenerationChunk(text: "hi", tokens: [1], position: 1),
                GenerationChunk(text: "", tokens: [], position: 1, stats: rstStubStats()),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        let resp = try await rstPost(
            "http://127.0.0.1:\(server.boundPort)/v1/chat/completions",
            json: #"{"stream":true,"messages":[{"role":"user","content":"x"}]}"#)
        #expect(resp.status == 200)
        let text = String(data: resp.body, encoding: .utf8) ?? ""
        #expect(
            text.contains("\"finish_reason\":\"stop\"")
                || text.contains("\"finish_reason\": \"stop\""))
    }

    // MARK: - Charter 3 : parser false recover (Domain)
    // Imagine: commentary or truncated XML becomes a tool_call.

    @Test("truncated XML does not recover : Domain")
    func truncatedXML() {
        #expect(ToolCallParser.recover(from: "<tool_call><function=bash><parameter=c>ls") == nil)
    }

    @Test("short markup in long chatter does not recover : Domain (ratio)")
    func ratioRejects() {
        let markup = #"<tool_call>{"name":"bash","arguments":{}}</tool_call>"#
        let chatter = String(repeating: "word ", count: 80)
        #expect(ToolCallParser.recover(from: chatter + markup) == nil)
    }

    @Test("markup-majority recovers : Domain")
    func ratioAccepts() {
        let markup = #"<tool_call>{"name":"bash","arguments":{"command":"x"}}</tool_call>"#
        #expect(ToolCallParser.recover(from: markup + " ok")?.calls.first?.name == "bash")
    }

    // MARK: - Charter 4 : two POSTs cannot share one KV (Time)

    @Test("AsyncLock peak is 1 : Time; mutation: no-op acquire")
    func admitSerial() async {
        let gate = AsyncLock()
        let box = PeakBox()
        await withTaskGroup(of: Void.self) { group in
            for _ in 0 ..< 8 {
                group.addTask {
                    await gate.acquire()
                    box.enter()
                    try? await Task.sleep(for: .milliseconds(15))
                    box.leave()
                    gate.release()
                }
            }
        }
        #expect(box.peak == 1)
    }
}

private final class PeakBox: @unchecked Sendable {
    private let sem = DispatchSemaphore(value: 1)
    private var live = 0
    private(set) var peak = 0
    func enter() {
        sem.wait()
        live += 1
        peak = max(peak, live)
        sem.signal()
    }
    func leave() {
        sem.wait()
        live -= 1
        sem.signal()
    }
}

private func rstStubStats() -> GenerationStats {
    GenerationStats(
        promptTokens: 1, generatedTokens: 1, contextSize: 0,
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

private struct RSTHTTP {
    var status: Int
    var body: Data
}

private func rstPost(_ url: String, json: String) async throws -> RSTHTTP {
    var req = URLRequest(url: URL(string: url)!)
    req.httpMethod = "POST"
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.httpBody = json.data(using: .utf8)
    let (data, resp) = try await URLSession.shared.data(for: req)
    return RSTHTTP(status: (resp as! HTTPURLResponse).statusCode, body: data)
}

/// Instrument for charter 2. Known-bad: first HTTP response only.
private func rawTCP(
    port: UInt16,
    send: String,
    closeAfterSend: Bool = false,
    until predicate: @escaping @Sendable (String) -> Bool
) async throws -> String {
    final class Session: @unchecked Sendable {
        private let sem = DispatchSemaphore(value: 1)
        private var _conn: NWConnection?
        private var acc = Data()
        let once = OnceResume()
        let predicate: @Sendable (String) -> Bool

        init(predicate: @escaping @Sendable (String) -> Bool) {
            self.predicate = predicate
        }

        var conn: NWConnection? {
            get {
                sem.wait(); defer { sem.signal() }
                return _conn
            }
            set {
                sem.wait(); defer { sem.signal() }
                _conn = newValue
            }
        }

        func append(_ data: Data) {
            sem.wait()
            acc.append(data)
            sem.signal()
        }

        func snapshot() -> String {
            sem.wait()
            defer { sem.signal() }
            return String(data: acc, encoding: .utf8) ?? ""
        }

        func receiveLoop(
            _ conn: NWConnection, cont: CheckedContinuation<String, Error>
        ) {
            conn.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) {
                [self] data, _, isComplete, error in
                if let error {
                    self.once.finish(cont, .failure(error))
                    return
                }
                if let data { self.append(data) }
                let s = self.snapshot()
                if self.predicate(s) || isComplete {
                    conn.cancel()
                    self.once.finish(cont, .success(s))
                    return
                }
                self.receiveLoop(conn, cont: cont)
            }
        }
    }

    let session = Session(predicate: predicate)
    do {
        return try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask {
                try await withCheckedThrowingContinuation {
                    (cont: CheckedContinuation<String, Error>) in
                    let conn = NWConnection(
                        host: NWEndpoint.Host("127.0.0.1"),
                        port: NWEndpoint.Port(rawValue: port)!,
                        using: .tcp)
                    session.conn = conn
                    let queue = DispatchQueue(label: "butter.rst.rawTCP.\(port)")
                    conn.stateUpdateHandler = { state in
                        switch state {
                        case .ready:
                            conn.send(
                                content: Data(send.utf8),
                                completion: .contentProcessed { _ in
                                    if closeAfterSend {
                                        conn.send(
                                            content: nil,
                                            contentContext: .finalMessage,
                                            isComplete: true,
                                            completion: .contentProcessed { _ in })
                                    }
                                })
                            session.receiveLoop(conn, cont: cont)
                        case .failed(let e):
                            session.once.finish(cont, .failure(e))
                        case .cancelled:
                            let s = session.snapshot()
                            if session.predicate(s) {
                                session.once.finish(cont, .success(s))
                            } else {
                                session.once.finish(cont, .failure(RSTTimeout()))
                            }
                        default:
                            break
                        }
                    }
                    conn.start(queue: queue)
                }
            }
            group.addTask {
                try await Task.sleep(for: .seconds(3))
                session.conn?.cancel()
                throw RSTTimeout()
            }
            let first = try await group.next()!
            group.cancelAll()
            session.conn?.cancel()
            return first
        }
    } catch {
        session.conn?.cancel()
        throw error
    }
}

private struct RSTTimeout: Error {}

private final class OnceResume: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func finish(
        _ cont: CheckedContinuation<String, Error>,
        _ result: Result<String, Error>
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard !done else { return }
        done = true
        cont.resume(with: result)
    }
}
