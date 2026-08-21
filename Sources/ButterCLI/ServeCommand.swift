// Copyright 2026 Eric Kryski (@ekryski)
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
// `butter serve`: Chat Completions HTTP server (one model, one KV slot).

import ArgumentParser
import ButterSwift
import Foundation
import os

struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Serve an OpenAI-compatible Chat Completions API (one loaded model)."
    )

    @Option(name: .shortAndLong, help: "HuggingFace repo id or local model path.")
    var model: String

    @Option(name: .long, help: "TCP port. Default 8000.")
    var port: Int = 8000

    @Option(name: .long, help: "Bind address. Default 127.0.0.1.")
    var host: String = "127.0.0.1"

    @Flag(name: .long, help: "Default chat_template_kwargs.enable_thinking when the client omits it.")
    var enableThinking: Bool = false

    @Option(name: .long, help: "Default reasoning_effort (low/medium/high) when the client omits it.")
    var reasoningEffort: String?

    @Option(
        name: .long,
        help:
            "KV cache scheme: \"raw\", \"affine8\", \"affine4\", \"aura\" (AURA packed KV), or any \"auraNvM\" recipe."
    )
    var kvCache: String?

    @Option(
        name: .long,
        help: "AURA decode path: \"compressed\" (default) or \"dequant-mirror\". Ignored unless --kv-cache is an aura scheme."
    )
    var auraDecodePath: String?

    @Option(
        name: .long,
        help: "Size the KV cache for this many positions. Default: model family pick."
    )
    var maxContext: Int?

    @Option(
        name: .long,
        help: "FIFO KV window size. 0 / unset = unbounded up to --max-context."
    )
    var kvWindowSize: Int?

    @Option(name: .long, help: "Attention-sink slots pinned across FIFO eviction. Needs --kv-window-size.")
    var kvWindowKeep: Int?

    func validate() throws {
        if port < 0 || port > 65535 {
            throw ValidationError("--port must be in 0...65535 (got \(port))")
        }
        if let raw = reasoningEffort, ReasoningEffort(rawValue: raw) == nil {
            throw ValidationError(
                "--reasoning-effort must be low, medium, or high (got \(raw))")
        }
    }

    func run() async throws {
        print("butter \(Butter.version) : loading \(model)…")
        let loadStart = Date()
        let loadOpts: LoadOptions
        do {
            loadOpts = try LoadOptions.fromCLI(
                kvCache: kvCache,
                kvWindowSize: kvWindowSize,
                kvWindowKeep: kvWindowKeep,
                auraDecodePath: auraDecodePath,
                maxContextLength: maxContext)
        } catch {
            throw ValidationError("\(error)")
        }
        let m = try await Model.load(model, options: loadOpts)
        print("loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStart)))s")

        var template = ChatTemplateOptions()
        template.enableThinking = enableThinking
        if let raw = reasoningEffort, let effort = ReasoningEffort(rawValue: raw) {
            template.reasoningEffort = effort
        }

        let initial = ButterServeBackend(model: m, modelId: model)
        let opts = loadOpts
        let backend = ResidentServeBackend(initial: initial) { id in
            let box = OSAllocatedUnfairLock<Result<ButterServeBackend, Error>?>(initialState: nil)
            let sem = DispatchSemaphore(value: 0)
            Task {
                do {
                    let loaded = try await Model.load(id, options: opts)
                    box.withLock {
                        $0 = .success(ButterServeBackend(model: loaded, modelId: id))
                    }
                } catch {
                    box.withLock { $0 = .failure(error) }
                }
                sem.signal()
            }
            sem.wait()
            switch box.withLock({ $0 }) {
            case .success(let b): return b
            default: throw ServeError.notFound
            }
        }
        let server = ServeServer(
            host: host,
            port: UInt16(port),
            backend: backend,
            defaultTemplateOptions: template)
        try await server.runForever()
    }
}
