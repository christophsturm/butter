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
// Serve load flags, thinking kwargs, and cached_tokens.

import Foundation
import Testing

@testable import ButterSwift

@Suite("Serve flags")
struct ServeFlagsTests {

    @Test("aura kv-cache is AURA not raw : mutation: ignore --kv-cache")
    func auraKVCache() throws {
        let opts = try LoadOptions.fromCLI(kvCache: "aura")
        guard case .auraQuantized = opts.kvCache else {
            Issue.record("expected auraQuantized, got \(opts.kvCache)")
            return
        }
    }

    @Test("unknown kv-cache is an error : mutation: silently fall back to raw")
    func unknownKVCache() {
        #expect(throws: LoadOptions.CLIError.unknownKV("nope")) {
            _ = try LoadOptions.fromCLI(kvCache: "nope")
        }
    }

    @Test("max-context sizes the cache : mutation: drop the field")
    func maxContext() throws {
        let opts = try LoadOptions.fromCLI(maxContextLength: 8192)
        #expect(opts.maxContextLength == 8192)
    }

    @Test("max-context 0 is rejected")
    func maxContextZero() {
        #expect(throws: LoadOptions.CLIError.maxContextMustBePositive) {
            _ = try LoadOptions.fromCLI(maxContextLength: 0)
        }
    }

    @Test("window keep without size is rejected")
    func windowKeepNeedsSize() {
        #expect(throws: LoadOptions.CLIError.windowKeepNeedsSize) {
            _ = try LoadOptions.fromCLI(kvWindowKeep: 4)
        }
    }

    @Test("aura decode path compressed vs mirror")
    func auraDecodePath() throws {
        let a = try LoadOptions.fromCLI(kvCache: "aura", auraDecodePath: "compressed")
        #expect(a.auraDecodePath == .compressed)
        let b = try LoadOptions.fromCLI(kvCache: "aura", auraDecodePath: "dequant-mirror")
        #expect(b.auraDecodePath == .dequantMirror)
    }

    @Test("chat_template_kwargs.enable_thinking is parsed : Claims")
    func thinkingKwargs() throws {
        let json = """
            {"messages":[{"role":"user","content":"hi"}],\
            "chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"high"}}
            """
        let req = try ChatCompletionHTTPRequest.parse(Data(json.utf8))
        #expect(req.chatTemplateKwargs?.enableThinking == true)
        #expect(req.chatTemplateKwargs?.reasoningEffort == .high)
    }

    @Test("prepare records lastCachedTokens as nPast : Product")
    func lastCachedTokensFromPrepare() {
        let kv = FakeKVCache(length: 4, maxSeq: 32)
        let session = GenerationSession(makeCaches: { [FakeKVCache(length: 0, maxSeq: 32)] })
        session.testingReplaceCaches([kv], tokens: [1, 2, 3, 4])
        let n = session.prepare(for: [1, 2, 3, 4, 5, 6])
        #expect(n == 4)
        #expect(session.lastCachedTokens == 4)
    }

    @Test("usage includes cached_tokens : mutation: omit the field")
    func usageCachedTokens() async throws {
        let backend = StubServeBackend(
            modelId: "stub-model",
            lastCachedTokens: 17,
            chunks: [
                GenerationChunk(text: "ok", tokens: [1], position: 1),
                GenerationChunk(
                    text: "", tokens: [], position: 1,
                    stats: GenerationStats(
                        promptTokens: 20, generatedTokens: 1, contextSize: 0,
                        prefillTimeS: 0, decodeTimeS: 0, timeToFirstTokenMs: 0,
                        steadyTokensPerSecond: nil,
                        baselineGPUBytes: 0, postPrefillGPUBytes: 0, postDecodeGPUBytes: 0,
                        prefillPeakGPUBytes: 0, decodePeakGPUBytes: 0,
                        wiredTicketBytes: 0, weightsBytes: 0,
                        kvCacheAllocatedBytes: 0, kvCacheUsedBytes: 0,
                        thinkPerplexity: nil, genPerplexity: nil,
                        thinkKLDivergence: nil, genKLDivergence: nil,
                        thinkTokenCount: nil, genTokenCount: nil)),
            ])
        let server = ServeServer(host: "127.0.0.1", port: 0, backend: backend)
        try await server.start()
        defer { server.stop() }
        var req = URLRequest(
            url: URL(string: "http://127.0.0.1:\(server.boundPort)/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = Data(#"{"stream":false,"messages":[{"role":"user","content":"hi"}]}"#.utf8)
        let (data, resp) = try await URLSession.shared.data(for: req)
        #expect((resp as! HTTPURLResponse).statusCode == 200)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let usage = obj?["usage"] as? [String: Any]
        #expect(usage?["cached_tokens"] as? Int == 17)
    }
}

