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
// Tiny HTTP/1.1 + SSE server (Network.framework). OpenAI Chat Completions.

import Foundation
import Network

public final class ServeServer: @unchecked Sendable {
    public let host: String
    public let requestedPort: UInt16
    public let backend: any ServeGenerationBackend
    public var defaultTemplateOptions: ChatTemplateOptions

    private var listener: NWListener?
    let queue = DispatchQueue(label: "butter.serve")
    private let stateLock = NSLock()
    private var _boundPort: UInt16 = 0
    private var connections: [ObjectIdentifier: ConnectionHandler] = [:]

    public var boundPort: UInt16 { stateLock.withLock { _boundPort } }

    public init(
        host: String = "127.0.0.1",
        port: UInt16 = 8000,
        backend: any ServeGenerationBackend,
        defaultTemplateOptions: ChatTemplateOptions = .init()
    ) {
        self.host = host
        self.requestedPort = port
        self.backend = backend
        self.defaultTemplateOptions = defaultTemplateOptions
    }

    public func start() async throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        params.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: NWEndpoint.Host(host),
            port: NWEndpoint.Port(rawValue: requestedPort) ?? .any)
        let listener = try NWListener(using: params)
        self.listener = listener
        listener.newConnectionHandler = { [weak self] conn in
            self?.accept(conn)
        }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let once = Once()
            listener.stateUpdateHandler = { [weak self] state in
                switch state {
                case .ready:
                    let port = listener.port?.rawValue ?? 0
                    self?.stateLock.withLock { self?._boundPort = port }
                    once.resume { cont.resume() }
                case .failed(let err):
                    once.resume { cont.resume(throwing: err) }
                default:
                    break
                }
            }
            listener.start(queue: queue)
        }
    }

    public func stop() {
        listener?.cancel()
        listener = nil
        // Cancel outside the lock: handler.onClose also takes stateLock.
        let handlers = stateLock.withLock { () -> [ConnectionHandler] in
            let vals = Array(connections.values)
            connections.removeAll()
            return vals
        }
        for h in handlers { h.cancel() }
    }

    /// Start and park until the current task is cancelled.
    public func runForever() async throws {
        try await start()
        let port = boundPort
        print("butter serve listening on http://\(host):\(port)")
        print("  GET  /health")
        print("  GET  /v1/models")
        print("  POST /v1/chat/completions")
        print("  POST /v1/messages")
        while !Task.isCancelled {
            try await Task.sleep(for: .seconds(3600))
        }
        stop()
    }

    private func accept(_ connection: NWConnection) {
        let handler = ConnectionHandler(connection: connection, server: self)
        let id = ObjectIdentifier(handler)
        stateLock.withLock { connections[id] = handler }
        handler.onClose = { [weak self] in
            self?.stateLock.withLock { _ = self?.connections.removeValue(forKey: id) }
        }
        handler.start()
    }
}

private final class Once: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func resume(_ body: () -> Void) {
        lock.lock()
        defer { lock.unlock() }
        guard !done else { return }
        done = true
        body()
    }
}

private final class ConnectionHandler: @unchecked Sendable {
    let connection: NWConnection
    weak var server: ServeServer?
    var onClose: (() -> Void)?
    private var buffer = Data()
    private var generationTask: Task<Void, Never>?
    private var closed = false
    private let lock = NSLock()
    private var writeQueue: [Data] = []
    private var writing = false
    private var closeWhenIdle = false
    /// HTTP/1.1 keep-alive unless the client asked to close (POST generation forces close).
    private var responseKeepAlive = false
    /// Stop reading further requests (one-shot generation / closing).
    private var readingPaused = false

    init(connection: NWConnection, server: ServeServer) {
        self.connection = connection
        self.server = server
    }

    func start() {
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                self?.cancel()
            default:
                break
            }
        }
        connection.start(queue: server?.queue ?? .main)
        receive()
    }

    func cancel() {
        lock.lock()
        let already = closed
        closed = true
        let task = generationTask
        generationTask = nil
        lock.unlock()
        guard !already else { return }
        task?.cancel()
        connection.cancel()
        onClose?()
    }

    private func shouldStopReading() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return closed || closeWhenIdle
    }

    private func isReadingPaused() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return readingPaused
    }

    private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) {
            [weak self] data, _, isComplete, error in
            guard let self else { return }
            if let error {
                self.fail(error)
                return
            }
            if let data, !data.isEmpty {
                self.buffer.append(data)
                if self.buffer.count > 64 * 1024 * 1024 {
                    self.responseKeepAlive = false
                    self.reply(status: 413, body: "payload too large", contentType: "text/plain")
                    return
                }
            }
            var handled = self.tryHandleRequest(connectionComplete: isComplete)
            while handled && !self.shouldStopReading() && !self.isReadingPaused()
                && self.tryHandleRequest(connectionComplete: false)
            {
                handled = true
            }
            if self.shouldStopReading() { return }
            if handled {
                // A finished request (no Content-Length uses this FIN as
                // the body delimiter) must keep generating. A later
                // receive that cannot parse a request and is complete
                // is a mid-prefill drop and cancel()s below.
                if !isComplete { self.receive() }
                return
            }
            if isComplete {
                self.cancel()
                return
            }
            self.receive()
        }
    }

    private func tryHandleRequest(connectionComplete: Bool = false) -> Bool {
        guard let headerEnd = findHeaderEnd(buffer) else { return false }
        let headerData = buffer.subdata(in: 0 ..< headerEnd)
        guard let headerText = String(data: headerData, encoding: .utf8) else {
            responseKeepAlive = false
            buffer.removeSubrange(0 ..< headerEnd)
            reply(status: 400, body: "bad request", contentType: "text/plain")
            return true
        }
        let parsed = parseHeaders(headerText)
        responseKeepAlive = wantsKeepAlive(parsed)
        let bodyStart = headerEnd
        let have = buffer.count - bodyStart

        let body: Data
        if parsed.transferEncodingChunked {
            switch decodeChunkedBody(buffer, bodyStart: bodyStart) {
            case .needMore:
                if connectionComplete {
                    responseKeepAlive = false
                    buffer.removeAll(keepingCapacity: false)
                    reply(status: 400, body: "incomplete chunked body", contentType: "text/plain")
                    return true
                }
                return false
            case .error:
                responseKeepAlive = false
                buffer.removeAll(keepingCapacity: false)
                reply(status: 400, body: "bad chunked body", contentType: "text/plain")
                return true
            case .done(let decoded, let endOffset):
                body = decoded
                buffer.removeSubrange(0 ..< endOffset)
            }
        } else if let cl = parsed.contentLength {
            if have < cl { return false }
            body = buffer.subdata(in: bodyStart ..< (bodyStart + cl))
            buffer.removeSubrange(0 ..< (bodyStart + cl))
        } else if parsed.method == "POST" || parsed.method == "PUT" {
            // No Content-Length: wait for peer close, then take remaining bytes.
            if !connectionComplete { return false }
            body = buffer.subdata(in: bodyStart ..< buffer.count)
            buffer.removeAll(keepingCapacity: false)
        } else {
            body = Data()
            buffer.removeSubrange(0 ..< bodyStart)
        }
        dispatch(method: parsed.method, path: parsed.path, body: body)
        return true
    }

    private func dispatch(method: String, path: String, body: Data) {
        let cors: [(String, String)] = [
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Headers", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ]
        if method == "OPTIONS" {
            reply(status: 204, body: Data(), headers: cors)
            return
        }
        let pathOnly = path.split(separator: "?", maxSplits: 1).first.map(String.init) ?? path
        do {
            switch (method, pathOnly) {
            case ("GET", "/health"):
                reply(status: 200, body: "ok\n", contentType: "text/plain", headers: cors)
            case ("GET", "/v1/models"):
                let json = modelsJSON(backend: server?.backend)
                reply(status: 200, body: json, contentType: "application/json", headers: cors)
            case ("POST", "/v1/chat/completions"):
                try handleChatCompletions(body: body, extraHeaders: cors)
            case ("POST", "/v1/messages"), ("POST", "/v1/messages/"):
                try handleMessages(body: body, extraHeaders: cors)
            default:
                if method != "GET" && method != "POST" {
                    reply(status: 405, body: "method not allowed", contentType: "text/plain", headers: cors)
                } else {
                    reply(status: 404, body: "not found", contentType: "text/plain", headers: cors)
                }
            }
        } catch let e as ServeError {
            let msg = jsonObject(["error": ["message": e.description, "type": "invalid_request_error"]])
            reply(status: 400, body: msg, contentType: "application/json", headers: cors)
        } catch {
            let msg = jsonObject(["error": ["message": "\(error)", "type": "server_error"]])
            reply(status: 500, body: msg, contentType: "application/json", headers: cors)
        }
    }

    private func handleChatCompletions(body: Data, extraHeaders: [(String, String)]) throws {
        guard let server else { throw ServeError.badRequest("server gone") }
        let req = try ChatCompletionHTTPRequest.parse(body)
        // Generation is one-shot; URLSession clients expect the connection to end.
        responseKeepAlive = false
        readingPaused = true
        var params = server.backend.defaultGenerationParameters
        if let mt = req.maxTokens { params.maxTokens = mt }
        if let t = req.temperature { params.temperature = t }
        if let p = req.topP { params.topP = p }
        if let s = req.seed { params.seed = s }
        var template = server.defaultTemplateOptions
        if let kw = req.chatTemplateKwargs {
            if let v = kw.enableThinking { template.enableThinking = v }
            if let v = kw.reasoningEffort { template.reasoningEffort = v }
            if !kw.extra.isEmpty { template.extraContext = kw.extra }
        }
        let stream = try server.backend.completeChat(
            messages: req.messages,
            tools: req.tools,
            parameters: params,
            templateOptions: template
        )
        let modelId = req.model ?? server.backend.modelId
        let completionId = "chatcmpl-\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
        let created = Int(Date().timeIntervalSince1970)

        if req.stream {
            var headers = extraHeaders
            headers.append(("Content-Type", "text/event-stream"))
            headers.append(("Cache-Control", "no-cache"))
            headers.append(("Connection", "close"))
            writeRaw(status: 200, headers: headers, body: Data())
            let task = Task { [weak self] in
                guard let self else { return }
                await self.streamCompletion(
                    stream: stream, id: completionId, created: created,
                    model: modelId, includeUsage: req.includeUsage,
                    maxTokens: params.maxTokens)
            }
            lock.lock()
            generationTask = task
            lock.unlock()
        } else {
            let task = Task { [weak self] in
                guard let self else { return }
                await self.bufferedCompletion(
                    stream: stream, id: completionId, created: created,
                    model: modelId, extraHeaders: extraHeaders,
                    maxTokens: params.maxTokens)
            }
            lock.lock()
            generationTask = task
            lock.unlock()
        }
    }

    private func handleMessages(body: Data, extraHeaders: [(String, String)]) throws {
        guard let server else { throw ServeError.badRequest("server gone") }
        let req = try MessagesHTTPRequest.parse(body)
        responseKeepAlive = false
        readingPaused = true
        var params = server.backend.defaultGenerationParameters
        params.maxTokens = req.maxTokens
        if let t = req.temperature { params.temperature = t }
        if let p = req.topP { params.topP = p }
        let stream = try server.backend.completeChat(
            messages: req.messages,
            tools: req.tools,
            parameters: params,
            templateOptions: server.defaultTemplateOptions
        )
        let modelId = req.model ?? server.backend.modelId
        let messageId = "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"

        if req.stream {
            var headers = extraHeaders
            headers.append(("Content-Type", "text/event-stream"))
            headers.append(("Cache-Control", "no-cache"))
            headers.append(("Connection", "close"))
            writeRaw(status: 200, headers: headers, body: Data())
            let task = Task { [weak self] in
                guard let self else { return }
                await self.streamMessages(
                    stream: stream, id: messageId, model: modelId,
                    maxTokens: params.maxTokens)
            }
            lock.lock()
            generationTask = task
            lock.unlock()
        } else {
            let task = Task { [weak self] in
                guard let self else { return }
                await self.bufferedMessages(
                    stream: stream, id: messageId, model: modelId,
                    extraHeaders: extraHeaders, maxTokens: params.maxTokens)
            }
            lock.lock()
            generationTask = task
            lock.unlock()
        }
    }

    private func streamCompletion(
        stream: AsyncThrowingStream<GenerationChunk, Error>,
        id: String, created: Int, model: String, includeUsage: Bool,
        maxTokens: Int
    ) async {
        var assembled = ""
        var emittedChars = 0
        var promptTokens = 0
        var completionTokens = 0
        do {
            for try await chunk in stream {
                try Task.checkCancellation()
                if let stats = chunk.stats {
                    promptTokens = stats.promptTokens
                    completionTokens = stats.generatedTokens
                    continue
                }
                if chunk.tokens.isEmpty && chunk.text.isEmpty { continue }
                assembled += chunk.text
                if let tagStart = toolTagStart(assembled) {
                    if tagStart > emittedChars {
                        let extra = substring(assembled, from: emittedChars, to: tagStart)
                        if !extra.isEmpty {
                            writeSSE(id: id, created: created, model: model, content: extra)
                        }
                        emittedChars = tagStart
                    }
                    continue
                }
                if assembled.count > emittedChars {
                    let extra = substring(assembled, from: emittedChars, to: assembled.count)
                    if !extra.isEmpty {
                        writeSSE(id: id, created: created, model: model, content: extra)
                    }
                    emittedChars = assembled.count
                }
            }
            try Task.checkCancellation()
            let held = substring(assembled, from: emittedChars, to: assembled.count)
            let recovered = ToolCallParser.recover(from: assembled)
            var finish =
                (maxTokens > 0 && completionTokens >= maxTokens) ? "length" : "stop"
            if let recovered {
                // Leading residual was already streamed before the tool tag.
                // Emit only unemitted residual held past markup (trailing text).
                if let heldRec = ToolCallParser.recover(from: held), !heldRec.residual.isEmpty {
                    writeSSE(id: id, created: created, model: model, content: heldRec.residual)
                } else if emittedChars == 0, !recovered.residual.isEmpty {
                    writeSSE(id: id, created: created, model: model, content: recovered.residual)
                }
                writeSSE(id: id, created: created, model: model, toolCalls: recovered.calls)
                finish = "tool_calls"
            } else if !held.isEmpty {
                writeSSE(id: id, created: created, model: model, content: held)
            }
            writeSSE(id: id, created: created, model: model, finishReason: finish)
            if includeUsage {
                writeSSEUsage(
                    id: id, created: created, model: model,
                    prompt: promptTokens, completion: completionTokens,
                    cached: server?.backend.lastCachedTokens ?? 0)
            }
            writeSSEDone()
            finishAndClose()
        } catch is CancellationError {
            cancel()
        } catch {
            writeSSEError("\(error)")
            writeSSEDone()
            finishAndClose()
        }
    }

    private func bufferedCompletion(
        stream: AsyncThrowingStream<GenerationChunk, Error>,
        id: String, created: Int, model: String, extraHeaders: [(String, String)],
        maxTokens: Int
    ) async {
        var text = ""
        var promptTokens = 0
        var completionTokens = 0
        do {
            for try await chunk in stream {
                try Task.checkCancellation()
                text += chunk.text
                if let stats = chunk.stats {
                    promptTokens = stats.promptTokens
                    completionTokens = stats.generatedTokens
                }
            }
            var finish =
                (maxTokens > 0 && completionTokens >= maxTokens) ? "length" : "stop"
            var toolCalls: [ParsedToolCall]?
            var content: String? = text
            if let recovered = ToolCallParser.recover(from: text) {
                toolCalls = recovered.calls
                content = recovered.residual.isEmpty ? nil : recovered.residual
                finish = "tool_calls"
            }
            let body = encodeChatCompletion(
                id: id, created: created, model: model,
                content: content, toolCalls: toolCalls, finish: finish,
                promptTokens: promptTokens, completionTokens: completionTokens,
                cachedTokens: server?.backend.lastCachedTokens ?? 0)
            reply(status: 200, body: body, contentType: "application/json", headers: extraHeaders)
        } catch is CancellationError {
            cancel()
        } catch {
            let msg = jsonObject(["error": ["message": "\(error)", "type": "server_error"]])
            reply(status: 500, body: msg, contentType: "application/json", headers: extraHeaders)
        }
    }

    // MARK: - Anthropic Messages codec handlers

    private func streamMessages(
        stream: AsyncThrowingStream<GenerationChunk, Error>,
        id: String, model: String, maxTokens: Int
    ) async {
        writeAnthropEvent(
            "message_start",
            MessagesCodec.messageStartEvent(id: id, model: model))
        var assembled = ""
        var emittedChars = 0
        var textStarted = false
        var completionTokens = 0

        func emitTextDelta(_ text: String) {
            guard !text.isEmpty else { return }
            if !textStarted {
                writeAnthropEvent(
                    "content_block_start", MessagesCodec.contentBlockStartText(index: 0))
                textStarted = true
            }
            writeAnthropEvent(
                "content_block_delta",
                MessagesCodec.contentBlockDeltaText(index: 0, text: text))
        }

        do {
            for try await chunk in stream {
                try Task.checkCancellation()
                if let stats = chunk.stats {
                    completionTokens = stats.generatedTokens
                    continue
                }
                if chunk.tokens.isEmpty && chunk.text.isEmpty { continue }
                assembled += chunk.text
                if let tagStart = toolTagStart(assembled) {
                    if tagStart > emittedChars {
                        emitTextDelta(substring(assembled, from: emittedChars, to: tagStart))
                        emittedChars = tagStart
                    }
                    continue
                }
                if assembled.count > emittedChars {
                    emitTextDelta(substring(assembled, from: emittedChars, to: assembled.count))
                    emittedChars = assembled.count
                }
            }
            try Task.checkCancellation()
            let held = substring(assembled, from: emittedChars, to: assembled.count)
            let recovered = ToolCallParser.recover(from: assembled)
            var stopReason = MessagesCodec.stopReason(
                finish: (maxTokens > 0 && completionTokens >= maxTokens) ? "length" : "stop")

            if let recovered {
                let trailing: String
                if let heldRec = ToolCallParser.recover(from: held) {
                    trailing = heldRec.residual
                } else if emittedChars == 0 {
                    trailing = recovered.residual
                } else {
                    trailing = ""
                }
                emitTextDelta(trailing)
                if textStarted {
                    writeAnthropEvent("content_block_stop", MessagesCodec.contentBlockStop(index: 0))
                }
                let toolBase = textStarted ? 1 : 0
                for (i, call) in recovered.calls.enumerated() {
                    let idx = toolBase + i
                    let toolId = MessagesCodec.toolUseId(from: call.id, index: i)
                    writeAnthropEvent(
                        "content_block_start",
                        MessagesCodec.contentBlockStartToolUse(
                            index: idx, id: toolId, name: call.name))
                    writeAnthropEvent(
                        "content_block_delta",
                        MessagesCodec.contentBlockDeltaInputJSON(
                            index: idx, partialJSON: call.arguments))
                    writeAnthropEvent(
                        "content_block_stop", MessagesCodec.contentBlockStop(index: idx))
                }
                stopReason = "tool_use"
            } else {
                emitTextDelta(held)
                if !textStarted {
                    writeAnthropEvent(
                        "content_block_start", MessagesCodec.contentBlockStartText(index: 0))
                    textStarted = true
                }
                writeAnthropEvent("content_block_stop", MessagesCodec.contentBlockStop(index: 0))
            }

            writeAnthropEvent(
                "message_delta",
                MessagesCodec.messageDeltaEvent(
                    stopReason: stopReason, outputTokens: completionTokens))
            writeAnthropEvent("message_stop", MessagesCodec.messageStopEvent())
            finishAndClose()
        } catch is CancellationError {
            cancel()
        } catch {
            writeAnthropEvent(
                "error", jsonObject(["type": "error", "error": ["message": "\(error)"]]))
            writeAnthropEvent("message_stop", MessagesCodec.messageStopEvent())
            finishAndClose()
        }
    }

    private func bufferedMessages(
        stream: AsyncThrowingStream<GenerationChunk, Error>,
        id: String, model: String, extraHeaders: [(String, String)],
        maxTokens: Int
    ) async {
        var text = ""
        var promptTokens = 0
        var completionTokens = 0
        do {
            for try await chunk in stream {
                try Task.checkCancellation()
                text += chunk.text
                if let stats = chunk.stats {
                    promptTokens = stats.promptTokens
                    completionTokens = stats.generatedTokens
                }
            }
            var finish =
                (maxTokens > 0 && completionTokens >= maxTokens) ? "length" : "stop"
            var residual: String? = text
            var toolCalls: [ParsedToolCall]?
            if let recovered = ToolCallParser.recover(from: text) {
                toolCalls = recovered.calls
                residual = recovered.residual.isEmpty ? nil : recovered.residual
                finish = "tool_calls"
            }
            let body = MessagesCodec.encodeMessage(
                id: id, model: model, text: residual, toolCalls: toolCalls,
                stopReason: MessagesCodec.stopReason(finish: finish),
                inputTokens: promptTokens, outputTokens: completionTokens)
            reply(status: 200, body: body, contentType: "application/json", headers: extraHeaders)
        } catch is CancellationError {
            cancel()
        } catch {
            let msg = jsonObject(["error": ["message": "\(error)", "type": "server_error"]])
            reply(status: 500, body: msg, contentType: "application/json", headers: extraHeaders)
        }
    }

    private func writeAnthropEvent(_ event: String, _ data: String) {
        writeRawBody(Data("event: \(event)\r\ndata: \(data)\r\n\r\n".utf8))
    }

    private func writeSSE(
        id: String, created: Int, model: String,
        content: String? = nil,
        toolCalls: [ParsedToolCall]? = nil,
        finishReason: String? = nil
    ) {
        var delta: [String: Any] = [:]
        if let content { delta["content"] = content }
        if let toolCalls {
            delta["tool_calls"] = toolCalls.enumerated().map { i, tc in
                [
                    "index": i,
                    "id": tc.id,
                    "type": "function",
                    "function": ["name": tc.name, "arguments": tc.arguments],
                ] as [String: Any]
            }
        }
        let payload: [String: Any] = [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                [
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finishReason.map { $0 as Any } ?? NSNull(),
                ] as [String: Any]
            ],
        ]
        writeSSEData(jsonObject(payload))
    }

    private func writeSSEUsage(
        id: String, created: Int, model: String, prompt: Int, completion: Int,
        cached: Int = 0
    ) {
        let payload: [String: Any] = [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [] as [Any],
            "usage": [
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "cached_tokens": cached,
            ],
        ]
        writeSSEData(jsonObject(payload))
    }

    private func writeSSEDone() { writeRawBody(Data("data: [DONE]\r\n\r\n".utf8)) }
    private func writeSSEError(_ message: String) {
        writeSSEData(jsonObject(["error": ["message": message]]))
    }
    private func writeSSEData(_ json: String) {
        writeRawBody(Data("data: \(json)\r\n\r\n".utf8))
    }

    private func writeRaw(status: Int, headers: [(String, String)], body: Data) {
        var head = "HTTP/1.1 \(status) \(statusText(status))\r\n"
        var hasContentLength = false
        var isEventStream = false
        for (k, v) in headers {
            head += "\(k): \(v)\r\n"
            let key = k.lowercased()
            if key == "content-length" { hasContentLength = true }
            if key == "content-type" && v.lowercased().hasPrefix("text/event-stream") {
                isEventStream = true
            }
        }
        // Always emit Content-Length for non-SSE responses (including empty bodies).
        // 204 No Content stays without a body/length.
        if !hasContentLength && !isEventStream && status != 204 {
            head += "Content-Length: \(body.count)\r\n"
        }
        head += "\r\n"
        var data = Data(head.utf8)
        data.append(body)
        writeRawBody(data)
    }

    private func writeRawBody(_ data: Data) {
        lock.lock()
        writeQueue.append(data)
        if writing {
            lock.unlock()
            return
        }
        writing = true
        lock.unlock()
        pumpWrites()
    }

    private func pumpWrites() {
        lock.lock()
        guard !writeQueue.isEmpty else {
            writing = false
            let shouldClose = closeWhenIdle
            lock.unlock()
            if shouldClose {
                connection.send(
                    content: nil,
                    contentContext: .finalMessage,
                    isComplete: true,
                    completion: .contentProcessed { [weak self] _ in self?.cancel() })
            }
            return
        }
        let data = writeQueue.removeFirst()
        lock.unlock()
        connection.send(content: data, completion: .contentProcessed { [weak self] _ in
            self?.pumpWrites()
        })
    }

    private func finishAndClose() {
        lock.lock()
        closeWhenIdle = true
        if writing || !writeQueue.isEmpty {
            lock.unlock()
            return
        }
        lock.unlock()
        connection.send(
            content: nil,
            contentContext: .finalMessage,
            isComplete: true,
            completion: .contentProcessed { [weak self] _ in self?.cancel() })
    }

    private func reply(
        status: Int, body: String, contentType: String,
        headers: [(String, String)] = []
    ) {
        reply(status: status, body: Data(body.utf8), contentType: contentType, headers: headers)
    }

    private func reply(
        status: Int, body: Data, contentType: String? = nil,
        headers: [(String, String)] = []
    ) {
        var headers = headers
        if let contentType { headers.append(("Content-Type", contentType)) }
        let keepAlive = responseKeepAlive
        headers.append(("Connection", keepAlive ? "keep-alive" : "close"))
        // Content-Length is added once in writeRaw (avoid a duplicate header).
        writeRaw(status: status, headers: headers, body: body)
        if keepAlive {
            return
        }
        finishAndClose()
    }

    private func fail(_ error: Error) {
        _ = error
        cancel()
    }
}

private func toolTagStart(_ s: String) -> Int? {
    let markers = ["<tool_call", "<|tool_call", "[TOOL_CALLS]"]
    var best: Int?
    for m in markers {
        if let r = s.range(of: m) {
            let i = s.distance(from: s.startIndex, to: r.lowerBound)
            best = min(best ?? i, i)
        }
    }
    return best
}

private func substring(_ s: String, from: Int, to: Int) -> String {
    guard from >= 0, to <= s.count, from < to else { return "" }
    let a = s.index(s.startIndex, offsetBy: from)
    let b = s.index(s.startIndex, offsetBy: to)
    return String(s[a ..< b])
}

// MARK: - HTTP parse + JSON

private func findHeaderEnd(_ data: Data) -> Int? {
    if let r = data.range(of: Data("\r\n\r\n".utf8)) { return r.upperBound }
    if let r = data.range(of: Data("\n\n".utf8)) { return r.upperBound }
    return nil
}

private struct ParsedHTTPRequest {
    var method: String
    var path: String
    var httpVersion: String
    var contentLength: Int?
    var transferEncodingChunked: Bool
    var connectionClose: Bool
    var connectionKeepAlive: Bool
}

private func parseHeaders(_ text: String) -> ParsedHTTPRequest {
    let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
    let lines = normalized.split(separator: "\n", omittingEmptySubsequences: false)
    let requestLine = lines.first?.split(separator: " ") ?? []
    let method = requestLine.count > 0 ? String(requestLine[0]) : "GET"
    let path = requestLine.count > 1 ? String(requestLine[1]) : "/"
    let httpVersion = requestLine.count > 2 ? String(requestLine[2]) : "HTTP/1.1"
    var contentLength: Int?
    var transferEncodingChunked = false
    var connectionClose = false
    var connectionKeepAlive = false
    for line in lines.dropFirst() {
        let parts = line.split(separator: ":", maxSplits: 1)
        guard parts.count == 2 else { continue }
        let name = parts[0].lowercased()
        let value = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
        switch name {
        case "content-length":
            contentLength = Int(value)
        case "transfer-encoding":
            transferEncodingChunked = value.lowercased().split(separator: ",").contains {
                $0.trimmingCharacters(in: .whitespaces) == "chunked"
            }
        case "connection":
            for token in value.lowercased().split(separator: ",") {
                let t = token.trimmingCharacters(in: .whitespaces)
                if t == "close" { connectionClose = true }
                if t == "keep-alive" { connectionKeepAlive = true }
            }
        default:
            break
        }
    }
    return ParsedHTTPRequest(
        method: method,
        path: path,
        httpVersion: httpVersion,
        contentLength: contentLength,
        transferEncodingChunked: transferEncodingChunked,
        connectionClose: connectionClose,
        connectionKeepAlive: connectionKeepAlive
    )
}

private func wantsKeepAlive(_ parsed: ParsedHTTPRequest) -> Bool {
    if parsed.connectionClose { return false }
    if parsed.connectionKeepAlive { return true }
    // HTTP/1.1 defaults to keep-alive; HTTP/1.0 defaults to close.
    return parsed.httpVersion.uppercased().hasPrefix("HTTP/1.1")
}

private enum ChunkedBodyResult {
    case needMore
    case done(body: Data, endOffset: Int)
    case error
}

/// Decode an HTTP/1.1 chunked body starting at `bodyStart` in `buffer`.
private func decodeChunkedBody(_ buffer: Data, bodyStart: Int) -> ChunkedBodyResult {
    var offset = bodyStart
    var body = Data()
    while true {
        guard let line = readChunkLine(buffer, from: &offset) else { return .needMore }
        let sizeToken = line.split(separator: ";", maxSplits: 1).first
            .map(String.init)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard let size = Int(sizeToken, radix: 16) else { return .error }
        if size == 0 {
            // Final chunk: optional trailers terminated by a blank line.
            if let end = findTrailingBlankLine(buffer, from: offset) {
                return .done(body: body, endOffset: end)
            }
            return .needMore
        }
        let dataEnd = offset + size
        guard buffer.count >= dataEnd else { return .needMore }
        body.append(buffer.subdata(in: offset ..< dataEnd))
        offset = dataEnd
        // Chunk data is followed by CRLF or LF.
        if offset + 1 < buffer.count, buffer[offset] == 0x0D, buffer[offset + 1] == 0x0A {
            offset += 2
        } else if offset < buffer.count, buffer[offset] == 0x0A {
            offset += 1
        } else if offset >= buffer.count {
            return .needMore
        } else {
            return .error
        }
    }
}

private func readChunkLine(_ buffer: Data, from offset: inout Int) -> String? {
    if let r = buffer.range(of: Data("\r\n".utf8), in: offset ..< buffer.count) {
        let line = String(data: buffer.subdata(in: offset ..< r.lowerBound), encoding: .utf8)
        offset = r.upperBound
        return line
    }
    if let r = buffer.range(of: Data("\n".utf8), in: offset ..< buffer.count) {
        let line = String(data: buffer.subdata(in: offset ..< r.lowerBound), encoding: .utf8)
        offset = r.upperBound
        return line
    }
    return nil
}

private func findTrailingBlankLine(_ buffer: Data, from offset: Int) -> Int? {
    if let r = buffer.range(of: Data("\r\n\r\n".utf8), in: offset ..< buffer.count) {
        return r.upperBound
    }
    if offset < buffer.count, buffer[offset] == 0x0D,
        offset + 1 < buffer.count, buffer[offset + 1] == 0x0A
    {
        // Zero trailers: lone CRLF after the 0-size line already consumed by readChunkLine…
        // Caller left `offset` at the first byte after `0\r\n`. A final `\r\n` ends the body.
        return offset + 2
    }
    if let r = buffer.range(of: Data("\n\n".utf8), in: offset ..< buffer.count) {
        return r.upperBound
    }
    if offset < buffer.count, buffer[offset] == 0x0A {
        return offset + 1
    }
    return nil
}

private func statusText(_ code: Int) -> String {
    switch code {
    case 200: return "OK"
    case 204: return "No Content"
    case 400: return "Bad Request"
    case 404: return "Not Found"
    case 405: return "Method Not Allowed"
    case 413: return "Payload Too Large"
    case 500: return "Internal Server Error"
    default: return "OK"
    }
}

private func modelsJSON(backend: (any ServeGenerationBackend)?) -> String {
    var entry: [String: Any] = [
        "id": backend?.modelId ?? "unknown",
        "object": "model",
        "owned_by": "butter",
    ]
    if let n = backend?.maxModelLen {
        entry["max_model_len"] = n
        entry["context_length"] = n
    }
    return jsonObject(["object": "list", "data": [entry]])
}

func jsonObject(_ obj: Any) -> String {
    guard JSONSerialization.isValidJSONObject(obj),
        let data = try? JSONSerialization.data(withJSONObject: obj, options: [])
    else { return "{}" }
    return String(data: data, encoding: .utf8) ?? "{}"
}

struct ChatCompletionHTTPRequest: Sendable {
    var model: String?
    var messages: [ChatMessage]
    var stream: Bool
    var includeUsage: Bool
    var tools: [ChatToolDefinition]?
    var maxTokens: Int?
    var temperature: Float?
    var topP: Float?
    var seed: UInt64?
    var chatTemplateKwargs: ChatTemplateKwargs?

    struct ChatTemplateKwargs: Sendable {
        var enableThinking: Bool?
        var reasoningEffort: ReasoningEffort?
        var extra: [String: any Sendable]
    }

    static func parse(_ data: Data) throws -> ChatCompletionHTTPRequest {
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
        guard let rawMessages = obj["messages"] as? [Any], !rawMessages.isEmpty else {
            throw ServeError.badRequest("messages required")
        }
        var messages: [ChatMessage] = []
        for raw in rawMessages {
            guard let m = raw as? [String: Any] else { continue }
            let roleStr = (m["role"] as? String) ?? "user"
            let role: ChatMessage.Role
            switch roleStr {
            case "system", "developer": role = .system
            case "assistant": role = .assistant
            case "tool": role = .tool
            default: role = .user
            }
            let content = extractContent(m["content"])
            let thinking = m["thinking"] as? String
            messages.append(ChatMessage(role: role, content: content, thinking: thinking))
        }
        if messages.isEmpty { throw ServeError.badRequest("messages required") }

        var tools: [ChatToolDefinition]?
        if let rawTools = obj["tools"] as? [Any] {
            var parsed: [ChatToolDefinition] = []
            for t in rawTools {
                guard let dict = t as? [String: Any] else { continue }
                let fn = (dict["function"] as? [String: Any]) ?? dict
                guard let name = fn["name"] as? String else { continue }
                let desc = (fn["description"] as? String) ?? ""
                let paramsJSON: String
                if let p = fn["parameters"] {
                    if let s = p as? String {
                        paramsJSON = s
                    } else if JSONSerialization.isValidJSONObject(p),
                        let d = try? JSONSerialization.data(withJSONObject: p),
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
        var includeUsage = false
        if let so = obj["stream_options"] as? [String: Any] {
            includeUsage = (so["include_usage"] as? Bool) ?? false
        }

        var kwargs: ChatTemplateKwargs?
        if let raw = obj["chat_template_kwargs"] as? [String: Any] {
            var extra: [String: any Sendable] = [:]
            var enable: Bool?
            var effort: ReasoningEffort?
            for (k, v) in raw {
                if k == "enable_thinking" {
                    enable = (v as? Bool) ?? (v as? NSNumber)?.boolValue
                } else if k == "reasoning_effort", let s = v as? String {
                    effort = ReasoningEffort(rawValue: s)
                    extra[k] = s
                } else {
                    extra[k] = jsonToSendable(v)
                }
            }
            kwargs = ChatTemplateKwargs(
                enableThinking: enable, reasoningEffort: effort, extra: extra)
        }

        let maxTokens =
            (obj["max_tokens"] as? Int) ?? (obj["max_completion_tokens"] as? Int)
            ?? (obj["max_tokens"] as? NSNumber)?.intValue
        let temperature =
            (obj["temperature"] as? NSNumber)?.floatValue ?? (obj["temperature"] as? Float)
        let topP = (obj["top_p"] as? NSNumber)?.floatValue ?? (obj["top_p"] as? Float)
        let seed: UInt64?
        if let n = obj["seed"] as? UInt64 {
            seed = n
        } else if let n = obj["seed"] as? Int, n >= 0 {
            seed = UInt64(n)
        } else {
            seed = nil
        }

        return ChatCompletionHTTPRequest(
            model: obj["model"] as? String,
            messages: messages,
            stream: stream,
            includeUsage: includeUsage,
            tools: tools,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            seed: seed,
            chatTemplateKwargs: kwargs
        )
    }
}

func extractContent(_ value: Any?) -> String {
    if let s = value as? String { return s }
    if let arr = value as? [Any] {
        return arr.compactMap { part -> String? in
            if let s = part as? String { return s }
            guard let d = part as? [String: Any] else { return nil }
            if let t = d["text"] as? String { return t }
            return nil
        }.joined()
    }
    return ""
}

func encodeChatCompletion(
    id: String, created: Int, model: String,
    content: String?, toolCalls: [ParsedToolCall]?, finish: String,
    promptTokens: Int, completionTokens: Int,
    cachedTokens: Int = 0
) -> String {
    var message: [String: Any] = ["role": "assistant"]
    message["content"] = content.map { $0 as Any } ?? NSNull()
    if let toolCalls {
        message["tool_calls"] = toolCalls.map { tc in
            [
                "id": tc.id,
                "type": "function",
                "function": ["name": tc.name, "arguments": tc.arguments],
            ] as [String: Any]
        }
    }
    let payload: [String: Any] = [
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            [
                "index": 0,
                "message": message,
                "finish_reason": finish,
            ] as [String: Any]
        ],
        "usage": [
            "prompt_tokens": promptTokens,
            "completion_tokens": completionTokens,
            "total_tokens": promptTokens + completionTokens,
            "cached_tokens": cachedTokens,
        ],
    ]
    return jsonObject(payload)
}
