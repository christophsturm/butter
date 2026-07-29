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
// Seeing — the image/video → text capability protocol. The handler behind
// `Model.vlm`. TOKEN-based: callers pass `promptTokens` already framed with
// the family's media-placeholder run (the `Model` layer owns message →
// promptTokens framing and the tokenizer, so this protocol carries no
// tokenizer, no token ids, and no chat messages). Audio-in is NOT here — it
// is the `Transcribing` capability (an omni model conforms to both).
//
// Two ways to conform, mirroring `TextGenerating`'s forward/default split:
//
//  • Implement the `prefillVisual` primitive (encode + splice the media,
//    prefill the caches, return the tail logits) and get `generate` /
//    `generateStream` for FREE from the shared default below. This is how the
//    engine-internal-vision families (Idefics3, PaliGemma, GLM-OCR, SmolVLM2)
//    conform — replacing their hand-rolled decode loops.
//
//  • Or override `generate` / `generateStream` directly and skip the
//    primitive. `VisionModel` does this — it keeps its tuned batched-embedding
//    prefill rather than the per-token default.
//
// The decode yields `GenerationChunk(text: "", tokens: [id])` — empty text,
// just token ids; the `Model` layer fills in `.text` via the tokenizer.

import Foundation

// `: Sendable` so the shared `generateStream` default can spawn its decode
// `Task` capturing `self` — every conformer is a load-immutable model object
// (`@unchecked Sendable`), matching `StreamingTranscriber: Sendable`.
public protocol Seeing: Sendable {
    /// The text backbone that runs the decode after the visual prefill.
    var textEngine: any TextGenerating { get }

    /// Build the per-layer caches for a generation (sized for the model's
    /// defaults unless `maxSeq` overrides).
    func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol]

    /// Encode + splice the present visual media into `promptTokens`, prefill
    /// `caches`, and return the last-position logits + the prompt length. The
    /// ONE method a vision family must implement to use the shared decode — it
    /// is the only place the family's encode signatures + injection style live.
    /// `promptTokens` already carries the placeholder run (framed by `Model`).
    /// The default traps; override `generate`/`generateStream` instead if you
    /// don't want the shared decode.
    func prefillVisual(
        promptTokens: [Int], image: RGBImage?, videoFrames: [RGBImage],
        caches: [any LayerCacheProtocol], device: Device
    ) throws -> (lastLogits: Tensor, promptLength: Int)

    /// Buffered image/video → text generation. Returns the generated token
    /// ids (the `Model` layer decodes them to text).
    func generate(
        promptTokens: [Int], image: RGBImage?, videoFrames: [RGBImage],
        maxTokens: Int, parameters: GenerationParameters?,
        eosTokenId: Int?, eosTokenIds: [Int], device: Device
    ) throws -> [Int]

    /// Streaming counterpart — one `GenerationChunk` per decoded token
    /// (`text` empty; `tokens` carries the id). Errors surface through the
    /// stream.
    func generateStream(
        promptTokens: [Int], image: RGBImage?, videoFrames: [RGBImage],
        maxTokens: Int, parameters: GenerationParameters?,
        eosTokenId: Int?, eosTokenIds: [Int], device: Device
    ) -> AsyncThrowingStream<GenerationChunk, Error>
}

extension Seeing {
    /// Default `prefillVisual`: traps. A conformer either implements this (and
    /// uses the shared `generate`/`generateStream` defaults) or overrides
    /// `generate`/`generateStream` (and never reaches here).
    public func prefillVisual(
        promptTokens: [Int], image: RGBImage?, videoFrames: [RGBImage],
        caches: [any LayerCacheProtocol], device: Device
    ) throws -> (lastLogits: Tensor, promptLength: Int) {
        preconditionFailure(
            "\(type(of: self)) must implement prefillVisual(...) or override "
                + "generate/generateStream")
    }

    /// Shared streaming decode: prefill via `prefillVisual`, then sample one
    /// token per step (the same `Sampling.sample` pipeline the text path uses —
    /// temperature 0 ⇒ greedy, plus top-k / top-p / min-p / rep-penalty) until
    /// a stop token or `maxTokens`. Yields token-only chunks.
    public func generateStream(
        promptTokens: [Int], image: RGBImage? = nil, videoFrames: [RGBImage] = [],
        maxTokens: Int = 512, parameters: GenerationParameters? = nil,
        eosTokenId: Int? = nil, eosTokenIds: [Int] = [], device: Device = .shared
    ) -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let caches = makeLayerCaches(maxSeq: nil, device: device)
                    let (lastLogits, promptLength) = try prefillVisual(
                        promptTokens: promptTokens, image: image,
                        videoFrames: videoFrames, caches: caches, device: device)
                    var stopSet = Set(eosTokenIds)
                    if let single = eosTokenId { stopSet.insert(single) }
                    let params = parameters ?? GenerationParameters()
                    var rng = params.makeRNG()
                    var history = promptTokens
                    var next = Sampling.sample(
                        lastLogits, parameters: params, rng: &rng, tokenHistory: history)
                    for step in 0 ..< maxTokens {
                        if stopSet.contains(next) { break }
                        try Task.checkCancellation()
                        continuation.yield(
                            GenerationChunk(
                                text: "", tokens: [next], position: promptLength + step))
                        history.append(next)
                        let logits = textEngine.forward(
                            tokenId: next, position: promptLength + step,
                            caches: caches, device: device)
                        next = Sampling.sample(
                            logits, parameters: params, rng: &rng, tokenHistory: history)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Buffered counterpart — runs the shared decode to completion.
    public func generate(
        promptTokens: [Int], image: RGBImage? = nil, videoFrames: [RGBImage] = [],
        maxTokens: Int = 512, parameters: GenerationParameters? = nil,
        eosTokenId: Int? = nil, eosTokenIds: [Int] = [], device: Device = .shared
    ) throws -> [Int] {
        let caches = makeLayerCaches(maxSeq: nil, device: device)
        let (lastLogits, promptLength) = try prefillVisual(
            promptTokens: promptTokens, image: image,
            videoFrames: videoFrames, caches: caches, device: device)
        var stopSet = Set(eosTokenIds)
        if let single = eosTokenId { stopSet.insert(single) }
        let params = parameters ?? GenerationParameters()
        var rng = params.makeRNG()
        var history = promptTokens
        var next = Sampling.sample(
            lastLogits, parameters: params, rng: &rng, tokenHistory: history)
        var generated: [Int] = []
        for step in 0 ..< maxTokens {
            if stopSet.contains(next) { break }
            generated.append(next)
            history.append(next)
            let logits = textEngine.forward(
                tokenId: next, position: promptLength + step,
                caches: caches, device: device)
            next = Sampling.sample(
                logits, parameters: params, rng: &rng, tokenHistory: history)
        }
        return generated
    }
}
