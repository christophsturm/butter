# Streaming

Streaming is the primitive in Butter's generation API; the buffered `Model.generate(...)` is a thin collector over the same stream so there's one source of truth for the prefill + decode loop.

```swift
let stream = model.generateStream(prompt: "Why is the sky blue?")
for try await chunk in stream {
    print(chunk.text, terminator: "")
}
```

Each chunk carries the decoded delta `text`, the new token id(s) since the last yield, and the absolute sequence position. The **final** chunk has empty `text` / `tokens` and a non-nil `stats: GenerationStats` with the full memory + timing numbers — same shape `--stats` prints.

## `GenerationChunk` shape

```swift
public struct GenerationChunk: Sendable {
    public let text: String              // decoded delta since last chunk
    public let tokens: [Int]             // new token id(s) in this chunk
    public let position: Int             // absolute sequence position after this chunk
    public let stats: GenerationStats?   // populated only on the final chunk
    public var isFinal: Bool { stats != nil }
}
```

## Cancellation

The stream-producing task honors Swift's structured cancellation. Drop the consuming task (or call `task.cancel()`) and the producer notices at the next token boundary, flushes the stream cleanly, and finishes — no zombie command buffers, no leaked KV cache.

```swift
let task = Task {
    for try await chunk in model.generateStream(prompt: "...") {
        if userInterrupted { break }
        print(chunk.text, terminator: "")
    }
}
// elsewhere:
task.cancel()
```

## Chat / multi-turn streaming

`generateStream` has a chat-templated overload that takes `[ChatMessage]` and renders through the tokenizer's chat template. Same chunk shape on the output side:

```swift
let messages: [ChatMessage] = [
    .init(role: .system, content: "You are concise."),
    .init(role: .user, content: "Why is the sky blue?"),
]
let stream = try model.generateStream(messages: messages)
for try await chunk in stream {
    print(chunk.text, terminator: "")
}
```

See [`chat-templates.md`](chat-templates.md) for how `ChatMessage`
+ `ChatTemplateOptions` map onto the tokenizer's Jinja template.

## Multimodal streaming (image / video / audio → text)

`VisionModel` mirrors the text streaming surface — one `generateStream`
overload per input modality, each returning the same
`AsyncThrowingStream<GenerationChunk, Error>`. A buffered collect of any of
them is bit-identical to the corresponding `generate(...)` call (they share
the prefill + per-token decode seam). Chunks carry token ids only —
`VisionModel` has no tokenizer, so the caller detokenizes:

```swift
let vlm = try #require(model.vlModel)
let stream = vlm.generateStream(
    promptTokens: promptTokens, image: image,
    maxTokens: 200,
    eosTokenId: model.config.eosTokenId, eosTokenIds: model.config.eosTokenIds)

var collected: [Int] = []
for try await chunk in stream {
    collected.append(contentsOf: chunk.tokens)
    // Robust incremental detokenization: decode the full running buffer
    // each step and print only the newly-revealed suffix (survives BPE
    // merges and multi-token glyphs a per-token decode would mangle).
    let text = model.tokenizer.decode(tokens: collected, skipSpecialTokens: true)
    // … print the delta vs what you printed last step …
}
```

`generateStream(videoFrames:)` and `generateStream(audio:)` are the video
and omni-audio analogues (both `throws`, because the frame / waveform
encode can fail before the stream starts). The image overload is
non-throwing.

## Speech streaming (STT / TTS)

The speech models stream on the same `AsyncThrowingStream` contract, with a
modality-appropriate element type:

```swift
// STT — yields incremental transcript TEXT (one delta per decoded token).
for try await chunk in sttModel.transcribeStream(waveform, maxNewTokens: 256) {
    print(chunk, terminator: "")
}

// TTS — yields one waveform Tensor per segment (audio chunks instead of
// tokens), so playback can start on segment 1 while segment 2 synthesizes.
for try await chunk in ttsModel.synthesizeStream(text: "Hello there.", voice: nil) {
    play(chunk)   // [samples] f32 @ the model's sample rate
}
```

Concatenating the STT deltas reproduces the batch transcript (modulo the
batch's trailing whitespace trim); concatenating the TTS chunks equals
`synthesize` over the joined segments. Autoregressive encoders stream
per-token; single-pass (CTC / RNN-T) encoders yield once — the best
"streaming" possible without chunked audio input.

## CLI streaming

Streaming is the `ffai` CLI's **primary route** for every command — text,
`generate image|video|audio`, `transcribe`, and `speak` all stream by
default. Pass `--no-streaming` to buffer and print once (bit-identical to
the buffered API). See [using-the-cli.md](using-the-cli.md).

## Buffered collection

`Model.generate(prompt:parameters:)` is implemented as roughly:

```swift
public func generate(prompt: String,
                     parameters: GenerationParameters? = nil) async throws -> GenerationResult {
    var generated: [Int] = [], text = ""
    var stats: GenerationStats?
    for try await chunk in generateStream(prompt: prompt, parameters: parameters) {
        generated.append(contentsOf: chunk.tokens)
        text += chunk.text
        if let s = chunk.stats { stats = s }
    }
    return GenerationResult(
        promptTokens: tokenizer.encode(text: prompt),
        generatedTokens: generated, text: text, stats: stats!
    )
}
```

So you get the same `GenerationResult` shape — including all the stats — whether you call the buffered API or build your own collector around the stream.

## Why streaming is the primitive

Three reasons:

1. **The most common UI pattern.** Token-as-it-arrives chat / completion demos can't wait for the full response.
2. **Fits batch decoding naturally.** When Phase 8+ batching lands, `generateStream(...)` for a single sequence becomes one of N sequences sharing the same kernel dispatch loop; the consumer side stays unchanged.
3. **One loop.** Buffered + streaming sharing the same producer means the prefill / decode / stats path is exercised by every call shape — no parallel "almost-the-same" implementations to keep in sync.

## See also

- [Quick start](quickstart.md) — basic usage including the streaming example.
- [Chat templates](chat-templates.md) — how `messages:` overloads render through the tokenizer.
- [`generation-parameters.md`](generation-parameters.md) — the knobs that control generation; same parameters apply to streaming and buffered.
- [Observability](observability.md) — the `[STATS]` block streaming yields on its final chunk.
