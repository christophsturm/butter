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
//
// ContextOverflowGuardTests — D9: `driveGeneration`'s fail-fast check
// (`promptTokens.count + params.maxTokens > engine.maxSeq`). Pins the
// exact boundary: a request that just fits the KV cache's physical
// capacity must proceed unmodified; one token past it must throw
// `GenerationError.contextOverflow` before any prefill work runs,
// instead of the old behaviour (run the full prefill, then hit
// `KVEvictionState.reserveNextSlot()`'s precondition mid-decode).
//
// `OverflowProbeModel` is a from-scratch minimal `LanguageModel` (same
// shape as `LanguageModelDefaultsTests.CountingModel`, kept local so
// this file has no cross-file test-fixture coupling): it implements
// only the handful of `LanguageModel` requirements without a default
// extension implementation, returns a fixed peaked-logits tensor from
// every `forward` call, and reports zero KV caches — real GPU tensor
// ops, no model weights, no HuggingFace load.

import Foundation
import Metal
import Testing
import Tokenizers

@testable import ButterSwift

@Suite("D9 context-overflow guard")
struct ContextOverflowGuardTests {

    /// Minimal `LanguageModel`: constant peaked logits (argmax always
    /// picks token 0), no real weights, no KV caches.
    private final class OverflowProbeModel: LanguageModel, @unchecked Sendable {
        let hidden = 4
        let nLayers = 0
        let nHeads = 1
        let nKVHeads = 1
        let headDim = 4
        let vocab = 8
        let maxSeq: Int
        let dtype: DType = .f32

        init(maxSeq: Int) { self.maxSeq = maxSeq }

        func parameters() -> [(String, Tensor)] { [] }

        func makeLayerCaches(maxSeq _: Int? = nil, device _: Device) -> [any LayerCacheProtocol] {
            []
        }

        func forward(
            tokenId _: Int, position _: Int,
            caches _: [any LayerCacheProtocol],
            on cmd: MTLCommandBuffer, device: Device
        ) -> Tensor {
            let t = Tensor.empty(shape: [vocab], dtype: .f32)
            var values = [Float](repeating: 1, count: vocab)
            values[0] = 100
            t.copyIn(from: values)
            return t
        }
    }

    /// `count` fixed-id tokens per `encode` call — gives an exact,
    /// deterministic `promptTokens.count` without a real tokenizer.
    private final class FixedLengthTokenizer: Tokenizer, @unchecked Sendable {
        let count: Int
        init(count: Int) { self.count = count }

        func tokenize(text: String) -> [String] { Array(repeating: "t", count: count) }
        func encode(text: String) -> [Int] { Array(repeating: 1, count: count) }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { encode(text: text) }
        func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { 1 }
        func convertIdToToken(_ id: Int) -> String? { "t" }
        var bosToken: String? { nil }
        var bosTokenId: Int? { nil }
        var eosToken: String? { nil }
        var eosTokenId: Int? { nil }
        var unknownToken: String? { nil }
        var unknownTokenId: Int? { nil }
        var hasChatTemplate: Bool { false }

        func applyChatTemplate(messages: [Message]) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
        func applyChatTemplate(
            messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
        func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws
            -> [Int]
        { throw TokenizerError.missingChatTemplate }
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
        func applyChatTemplate(
            messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
            truncation: Bool, maxLength: Int?, tools: [ToolSpec]?
        ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
        func applyChatTemplate(
            messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
            truncation: Bool, maxLength: Int?, tools: [ToolSpec]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
    }

    private func makeModel(maxSeq: Int, promptCount: Int) -> Model {
        Model(
            engine: OverflowProbeModel(maxSeq: maxSeq),
            tokenizer: FixedLengthTokenizer(count: promptCount),
            config: ModelConfig(architecture: "Qwen3_5ForCausalLM", modelType: "qwen3_5", raw: [:]),
            modelDirectory: URL(fileURLWithPath: "/tmp/butter-d9-overflow-test"),
            availableCapabilities: Capability.textOnly,
            enabledCapabilities: Capability.textOnly,
            defaultGenerationParameters: GenerationParameters(maxTokens: 0, temperature: 0))
    }

    @Test("prompt + maxTokens exactly at capacity does not throw")
    func atBoundaryPasses() async throws {
        let cap = 16
        let maxNew = 4
        let promptCount = cap - maxNew  // exact fit: 12 + 4 == 16
        let model = makeModel(maxSeq: cap, promptCount: promptCount)

        let result = try await model.generate(
            prompt: "irrelevant — FixedLengthTokenizer ignores the text",
            parameters: GenerationParameters(maxTokens: maxNew, stopOnEOS: false, temperature: 0))

        #expect(result.promptTokens.count == promptCount)
        #expect(result.generatedTokens.count == maxNew)
    }

    @Test("prompt + maxTokens one over capacity throws contextOverflow")
    func overBoundaryThrows() async throws {
        let cap = 16
        let maxNew = 4
        let promptCount = cap - maxNew + 1  // one too many: 13 + 4 == 17

        let model = makeModel(maxSeq: cap, promptCount: promptCount)

        do {
            _ = try await model.generate(
                prompt: "irrelevant — FixedLengthTokenizer ignores the text",
                parameters: GenerationParameters(
                    maxTokens: maxNew, stopOnEOS: false, temperature: 0))
            Issue.record("expected GenerationError.contextOverflow, but generate() succeeded")
        } catch let error as GenerationError {
            guard
                case .contextOverflow(
                    let gotPrompt, let gotMaxNew, let gotCap) = error
            else {
                Issue.record("expected .contextOverflow, got \(error)")
                return
            }
            #expect(gotPrompt == promptCount)
            #expect(gotMaxNew == maxNew)
            #expect(gotCap == cap)
        }
    }

    @Test("a .window-evicting cache is exempt from the capacity guard")
    func windowedCacheSkipsGuard() async throws {
        // A cache reporting `.window` eviction is designed to run past
        // its physical `maxSeq` forever (FIFO rotation) — the guard
        // must not fire even when prompt + maxTokens exceeds `maxSeq`.
        // `WindowedProbeModel` reuses the same forward primitive but
        // returns one real `KVCache` wired to `.window` eviction so
        // `driveGeneration`'s `isWindowed` scan sees it.
        final class WindowedProbeModel: LanguageModel, @unchecked Sendable {
            let hidden = 4
            let nLayers = 1
            let nHeads = 1
            let nKVHeads = 1
            let headDim = 4
            let vocab = 8
            let maxSeq: Int
            let dtype: DType = .f32
            init(maxSeq: Int) { self.maxSeq = maxSeq }
            func parameters() -> [(String, Tensor)] { [] }
            func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
                [
                    KVCache(
                        nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq ?? self.maxSeq,
                        dtype: dtype, eviction: .window(maxSize: 4, keep: 0), device: device)
                ]
            }
            func forward(
                tokenId _: Int, position _: Int,
                caches _: [any LayerCacheProtocol],
                on cmd: MTLCommandBuffer, device: Device
            ) -> Tensor {
                let t = Tensor.empty(shape: [vocab], dtype: .f32)
                var values = [Float](repeating: 1, count: vocab)
                values[0] = 100
                t.copyIn(from: values)
                return t
            }
        }

        let cap = 8
        let maxNew = 4
        let promptCount = cap  // prompt alone already == cap; +maxNew overflows cap
        let model = Model(
            engine: WindowedProbeModel(maxSeq: cap),
            tokenizer: FixedLengthTokenizer(count: promptCount),
            config: ModelConfig(architecture: "Qwen3_5ForCausalLM", modelType: "qwen3_5", raw: [:]),
            modelDirectory: URL(fileURLWithPath: "/tmp/butter-d9-overflow-window-test"),
            availableCapabilities: Capability.textOnly,
            enabledCapabilities: Capability.textOnly,
            defaultGenerationParameters: GenerationParameters(maxTokens: 0, temperature: 0))

        let result = try await model.generate(
            prompt: "irrelevant",
            parameters: GenerationParameters(maxTokens: maxNew, stopOnEOS: false, temperature: 0))
        #expect(result.generatedTokens.count == maxNew)
    }

    @Test("a mixed .unbounded + .window cache set still throws contextOverflow")
    func mixedCacheStillThrows() async throws {
        // Regression for the verifier finding on the original D9 patch:
        // hybrid families (Gemma 2/3/4, GPT-OSS, Laguna, MuseGlimmer)
        // mix `.window` sliding-local layers with `.unbounded` global
        // layers *within the same engine*. The old `caches.contains {
        // .window }` exemption fired — and silently disabled the
        // guard — the moment ANY layer was windowed, even though the
        // `.unbounded` layer sitting right next to it has exactly the
        // same fixed physical ceiling as a plain non-hybrid model and
        // still overflows. The fix only exempts when EVERY
        // attention-cache layer is `.window`; this model returns one
        // of each; the `.unbounded` layer must be sufficient to make
        // the guard fire.
        final class MixedProbeModel: LanguageModel, @unchecked Sendable {
            let hidden = 4
            let nLayers = 2
            let nHeads = 1
            let nKVHeads = 1
            let headDim = 4
            let vocab = 8
            let maxSeq: Int
            let dtype: DType = .f32
            init(maxSeq: Int) { self.maxSeq = maxSeq }
            func parameters() -> [(String, Tensor)] { [] }
            func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
                let cap = maxSeq ?? self.maxSeq
                return [
                    // Global layer: `.unbounded`, the fixed ceiling
                    // this test exists to make sure still gets checked.
                    KVCache(
                        nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, eviction: .unbounded, device: device),
                    // Sliding-local layer: `.window`, same shape as
                    // `windowedCacheSkipsGuard` above.
                    KVCache(
                        nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, eviction: .window(maxSize: 4, keep: 0), device: device),
                ]
            }
            func forward(
                tokenId _: Int, position _: Int,
                caches _: [any LayerCacheProtocol],
                on cmd: MTLCommandBuffer, device: Device
            ) -> Tensor {
                let t = Tensor.empty(shape: [vocab], dtype: .f32)
                var values = [Float](repeating: 1, count: vocab)
                values[0] = 100
                t.copyIn(from: values)
                return t
            }
        }

        let cap = 8
        let maxNew = 4
        let promptCount = cap  // prompt alone already == cap; +maxNew overflows cap
        let model = Model(
            engine: MixedProbeModel(maxSeq: cap),
            tokenizer: FixedLengthTokenizer(count: promptCount),
            config: ModelConfig(architecture: "Qwen3_5ForCausalLM", modelType: "qwen3_5", raw: [:]),
            modelDirectory: URL(fileURLWithPath: "/tmp/butter-d9-overflow-mixed-test"),
            availableCapabilities: Capability.textOnly,
            enabledCapabilities: Capability.textOnly,
            defaultGenerationParameters: GenerationParameters(maxTokens: 0, temperature: 0))

        do {
            _ = try await model.generate(
                prompt: "irrelevant",
                parameters: GenerationParameters(
                    maxTokens: maxNew, stopOnEOS: false, temperature: 0))
            Issue.record("expected GenerationError.contextOverflow, but generate() succeeded")
        } catch let error as GenerationError {
            guard
                case .contextOverflow(
                    let gotPrompt, let gotMaxNew, let gotCap) = error
            else {
                Issue.record("expected .contextOverflow, got \(error)")
                return
            }
            #expect(gotPrompt == promptCount)
            #expect(gotMaxNew == maxNew)
            #expect(gotCap == cap)
        }
    }
}
