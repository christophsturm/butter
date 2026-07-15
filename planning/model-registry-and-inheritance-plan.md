# Model Registry + Model Inheritance Consolidation Plan

| Field | Value |
|---|---|
| Status | Implemented (Phases 0–5 shipped) |
| Scope | Loader, Model wrapper, all 3 registries, model-type abstraction, CLI catalog, unit-test layout |
| Approach | Protocol composition (NOT class inheritance). Phased, behavior-preserving first. |
| Branch | `ek/model-registry-consolidation` (off `dev`) |

## Goal

One `Model.load()` entry → one generic `Model` for every modality (llm, vlm, stt, tts, sts, vad, ocr, omni; future tti/ttv/vtv/itv/segmentation). One `ModelRegistry` as the single source of truth for "what we support + how to load it", read by both the loader and the `ffai models` CLI. Kill the duplicated arch/modelType dispatch branches and the 3 parallel registries.

## Current state (facts)

- `Loader/Model.swift` = 1697 lines. Holds `ModelError` (24-75), `ModelRegistry` enum (164-1254, ~1090 lines incl. `Loaded` struct + `dispatchAndLoad` + ~45 `loadXxx` helpers + ~75 if-branches), `Model` final class (1258-1697).
- `Model.load(_:options:device:)` is ALREADY `async throws` (1511). `dispatchAndLoad(config:weights:options:device:)` is sync.
- 3 separate registries, incompatible return types:
  - text/VL: `ModelRegistry.dispatchAndLoad(...) -> Loaded` (`engine: any LanguageModel`, sync, weights pre-loaded).
  - audio: `AudioModelRegistry.load(directory:device:) async -> LoadedAudioModel` (26-case enum, NOT LanguageModel).
  - vad: `VADModelRegistry.loadFromDirectory(_:) -> LoadedVADModel` (5-case enum).
- CLI `FFAICLI/ModelsCommand.swift` = a SECOND hand-maintained catalog (`CatalogEntry{family,modelType,summary,repos}` + `CatalogGroup`), drifts from the dispatch chain.
- Protocols: `LanguageModel: Module` (forward/forwardMulti/forward(inputEmbedding:)/textEmbedding/shape props), `AudioModel: Module` (synthesize). `VisionModel` is a concrete composing class. Family roots = `enum <F>` (modelTypes/architectures/variant(for:)) + `<F>Variant` protocol + `<F>Error`.
- **CONFIRMED**: family roots self-validate. `loadModel`/`load` guard config fields → throw `<F>Error.missingConfig`; `weights.tensor(named:)` throws on missing tensor. Registry only does arch/modelType ROUTING; it does not validate.
- No `ModelKind` taxonomy exists; "type" is implied by the capability set + which registry loads it.

## Target architecture

### Model-type layers = protocols + default impls (not base classes)

Rationale: the taxonomy is overlapping sets, not a tree (omni = text+image+video+audio-in; ocr ≈ vlm caps). Swift single inheritance can't model that; siblings (vlm/omni share vision, stt/omni share audio) must share behavior via protocol extensions. Matches the value-type/composition convention.

```
protocol ModelEngine: Module {            // base every engine refines
    var capabilities: Set<Capability> { get }
    var dtype: DType { get }
    // metadata/lifecycle shared by all engines
}
protocol TextEngine: ModelEngine { forward(tokenId:)/forwardMulti(tokenIds:)/forward(inputEmbedding:)/textEmbedding }   // today's LanguageModel
protocol Transcribing: ModelEngine { transcribe(waveform)/transcribeStream }            // STT
protocol Synthesizing: ModelEngine { synthesize(text)/synthesizeStream }                // TTS
protocol SpeechTransforming: ModelEngine { process(waveform) -> waveform }              // STS
protocol Detecting: ModelEngine { detect(waveform) -> [Event] }                         // VAD
// future: ImageGenerating, VideoGenerating, Segmenting …
```
`LanguageModel` becomes `: TextEngine` (keep the name to limit churn). A family conforms to ONE or SEVERAL (omni: TextEngine + Transcribing). Shared per-type behavior lives in protocol extensions.

> ✅ Decided (D1): `forward`/`forwardMany` live on each modality protocol with consistent NAMES but modality-typed I/O; `ModelEngine` base carries only metadata/lifecycle. `Model.generate` dispatches by kind. (Rejected: one uniform `forward` signature — would force a concrete generic `Model` and break the single-`load()`-returns-one-`Model` goal.)

### `Model` = one generic wrapper

```
final class Model {                       // in FFAI/Model.swift (moved up)
    let engine: any ModelEngine
    let kind: ModelKind
    let capabilities: Set<Capability>
    let config: ModelConfig
    let tokenizer: (any Tokenizer)?       // nil for VAD/STS/pure-gen
    let vlModel: VisionModel?
    static func load(_:options:device:) async throws -> Model          // ONE entry, all kinds
    func generate(_ input: ModelInput) async throws -> GenerationResult
    func generateStream(_ input: ModelInput) -> AsyncThrowingStream<GenerationChunk, Error>
}
```
- `ModelInput` = tagged-union enum, one case per modality, each carrying a dedicated STRUCT payload: `.text(TextGenerationInput)`, `.imageText(ImageTextInput)`, `.audio(AudioTranscribeInput)`, `.speak(SpeakInput)`, … (future `.imageGen`/`.videoGen`/`.segment`).
- `GenerationResult` / `GenerationChunk` = same tagged-enum style: `.textDelta(String)`, `.audioSamples([Float])`, `.detections([Event])`, `.image(...)`. Streaming for ALL kinds.
- Enum (not open protocol) because FFAI owns the whole curated modality set: exhaustive `switch` forces every site to handle a new modality (the safety we want); matches the codebase's enum idiom. Struct payloads keep each modality's fields independently evolvable.
- Thin typed convenience methods on `Model` (`generateText`/`transcribe`/`synthesize`/`detect`) wrap the generic path + throw `ModelError.unsupportedInput` on kind mismatch — callers who know the kind skip the switch.
- `Model` validates input vs `capabilities`/`kind`. Text generation still routed through `Generation/Generate`; audio through the engine's synthesize/transcribe.

### `ModelKind` + `ModelEntry` (the registry table)

```
enum ModelKind { case llm, vlm, ocr, omni, stt, tts, sts, vad   // + reserved tti, ttv, vtv, itv, imageEdit, segmentation }
struct ModelEntry {
    let family: String            // "Qwen 3"
    let kind: ModelKind
    let modelTypes: Set<String>; let architectures: Set<String>
    let capabilities: Set<Capability>       // declared/max
    let summary: String; let repos: [String]    // catalog info (was CatalogEntry)
    let load: (ModelConfig, SafeTensorsBundle?, URL, LoadOptions, Device) async throws -> Model
    func handles(_ config: ModelConfig) -> Bool   // default: modelTypes∪architectures match (+ optional custom sniff closure)
}
let registry: [ModelEntry] = [ … ]      // single source of truth
```
- Loader: `find first entry where handles(config) → load`. Replaces ~75 branches + the if-arch/if-mt duplication.
- CLI `ffai models`: group `registry` by `kind`, print family/summary/repos. Deletes the hand-maintained catalog.
- `ModelInfo` (inspect) gains `kind`.

## Phased plan (each phase = its own commit(s), gated)

Cross-cutting rule (Q4): **every Source move/rename mirrors into `Tests/`** — test file name, folder, and what it tests track `Sources/`. Each phase updates the corresponding tests in the SAME commit; Phase 5 is a final reconciliation sweep.

### Phase 0 — Pure moves (zero logic)
- Extract `ModelRegistry` (enum + `Loaded` + `dispatchAndLoad` + all `loadXxx` + VL predicates) → `Loader/ModelRegistry.swift`.
- Move `Model` class + `ModelError` → `FFAI/Model.swift`.
- Mirror tests: split `Tests/FFAITests/Loader/ModelTests*` → `Loader/ModelRegistryTests.swift` + top-level `ModelTests.swift`.
- Risk: low. Build + unit gate. Commit.

### Phase 1 — `ModelKind` + data-driven text/VL dispatch
- Add `FFAI/ModelKind.swift`. Add `ModelEntry` (text/VL only this phase).
- Convert the ~75 text/VL branches → `[ModelEntry]` table + find-first-match loop. Families unchanged (still self-validate). Behavior-preserving.
- Keep `Loaded` internally for now (engine still `any LanguageModel`).
- Tests: `ModelRegistryTests` asserts table coverage (every family matchable), dispatch correctness, no-match → `unsupportedArchitecture`.
- Risk: medium (touches all text/VL routing). Gate. Commit.

### Phase 2 — Single catalog → CLI
- Move `summary`/`repos` into `ModelEntry`. `ffai models` renders from `registry` grouped by `kind`. Delete `CatalogEntry`/`CatalogGroup` from `ModelsCommand.swift`.
- Tests: `ModelsCommandTests` (or new `ModelRegistryCatalogTests`) — every kind represented, no empty groups, sample lookups.
- Risk: low. Gate. Commit.

### Phase 3 — Engine protocol layer
- Add `FFAI/ModelType/` protocols: `ModelEngine` (base) + `TextEngine`/`Transcribing`/`Synthesizing`/`SpeechTransforming`/`Detecting` (+ default impls for shared per-type behavior). `LanguageModel: TextEngine`. Refactor `AudioModel`/omni/VAD engines to conform.
- Add `ModelInput`/`GenerationResult`/`GenerationChunk` sum types.
- `Model` gains `engine: any ModelEngine` + `kind` + top-level `generate`/`generateStream` (text path wired; audio/vad wired in Phase 4). Keep `model.engine as? any LanguageModel` shim for existing call sites.
- Tests: protocol-conformance tests per modality; `Model.generate` text-path test.
- Risk: high (new public surface). Likely 2-3 sub-commits. Gate each.

### Phase 4 — Fold audio + VAD into the unified registry
- Add audio/VAD families as `ModelEntry`s (kind = stt/tts/sts/vad/omni). `ModelRegistry` becomes the only registry. `Model.load()` is the single async entry for ALL kinds, returning one `Model`; `generate`/`generateStream` dispatch by kind.
- Make all loads async + directory-uniform (text load already async at the public level). Retire `LoadedAudioModel`/`LoadedVADModel` enums (or keep internal-only behind the entry closures).
- Update call sites: CLI `transcribe`/`speak`/`generate`, `AudioModelRegistry`/`VADModelRegistry` consumers, tests.
- Risk: highest (big blast radius across CLI + audio/vad tests). Sub-commits per modality. Gate each.

### Phase 5 — Test-layout reconciliation + docs
- Sweep `Tests/` so every file mirrors its `Sources/` counterpart (name, folder, target-of-test). Remove dead test files.
- Update `documentation/architecture.md` (Converter/ already added; add ModelType/, ModelKind, unified registry), `documentation/developing/adding-a-model.md` (new "register a family = add a `ModelEntry`" flow).
- Full `make test` (unit + integration smoke). Commit.

## Risks / watch-items

- **Blast radius**: `model.engine: any LanguageModel` is used widely; Phase 3 shim limits churn, Phase 4 removes it.
- **New public API** (`ModelInput`/`GenerationChunk`): design once, carefully — hardest-to-reverse surface.
- **Audio/VAD load is directory-based**, text is config+weights-based; the `ModelEntry.load` closure must accept both (pass `URL` + optional pre-decoded config/bundle).
- **`forward`/`forwardMany` uniformity** — see Q1; do not force one signature across modalities.
- **Integration tests** (ModelTests/, serialized, multi-GB) must still pass — run `make test-integration` at Phase 4 end, not just unit.

## Resolved decisions

- **D1 — `forward`/`forwardMany`**: live on the modality protocols (consistent names, modality-typed I/O); base = metadata only. No uniform signature.
- **D2 — input/output types**: tagged-union enums (`ModelInput`/`GenerationChunk`) with struct payloads per case + thin typed convenience methods (`generateText`/`transcribe`/`synthesize`/`detect`). Not open protocols. (See Target architecture.)
- **D3 — `LoadedAudioModel`/`LoadedVADModel`**: DELETE outright once folded into the registry.
- **D4 — convert / model-card family lists**: fold into `ModelEntry` (single source of truth).
- **D5 — execution**: land Phases 0-2 first on one branch (sequential gated commits), then checkpoint with the user before 3-5.

## Open questions (revisit at Phase 3)

- `ModelInput` struct-payload field design (messages/tools/images vs waveform vs voice) — design when Phase 3 lands, not before.
- Whether `Model.load` for directory-only audio/vad needs a pre-decode step shared with text (Phase 4 detail).

## Phase 3 API sketch (proposed — awaiting sign-off)

### Engine protocol layer (`FFAI/ModelType/`)

```swift
// Base — every loadable engine. Metadata/lifecycle only (D1: no forward here).
public protocol ModelEngine: Module {
    var capabilities: Set<Capability> { get }
    var dtype: DType { get }
}

// Text generation (LLM/VLM). The existing `LanguageModel` becomes this (rename or
// `LanguageModel: TextEngine` alias — decision T1). Keeps today's forward surface.
public protocol TextEngine: ModelEngine {
    var hidden: Int { get }; var maxContextWindow: Int { get }
    func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol]
    func forward(tokenId: Int, position: Int, caches: [any LayerCacheProtocol],
                 on: MTLCommandBuffer, device: Device) -> Tensor
    func forwardMulti(tokenIds: [Int], startingAt: Int, caches: [any LayerCacheProtocol],
                      on: MTLCommandBuffer, device: Device) -> Tensor   // keep name (T2)
    func forward(inputEmbedding: Tensor, position: Int, …) -> Tensor    // VLM splice
    func textEmbedding(tokenId: Int, device: Device) -> Tensor
}
public protocol Transcribing: ModelEngine {                            // STT
    func transcribe(_ w: [Float], parameters: TranscribeParameters, device: Device)
        async throws -> TranscriptionResult
    func transcribeStream(_ w: [Float], parameters: TranscribeParameters)
        -> AsyncThrowingStream<String, Error>
}
public protocol Synthesizing: ModelEngine {                            // TTS
    var sampleRate: Int { get }
    func synthesize(_ text: String, parameters: AudioGenerationParameters, device: Device)
        async throws -> [Float]
    func synthesizeStream(_ text: String, parameters: AudioGenerationParameters)
        -> AsyncThrowingStream<[Float], Error>
}
public protocol SpeechTransforming: ModelEngine {                      // STS
    var sampleRate: Int { get }
    func process(_ w: [Float], device: Device) async throws -> [Float]
}
public protocol Detecting: ModelEngine {                              // VAD
    func detect(_ w: [Float], device: Device) async throws -> [DetectionEvent]
}
```
A family conforms to one or several (omni → `TextEngine` + `Transcribing`).

### Input / output sum types (D2: tagged enums + struct payloads)

```swift
public enum ModelInput: Sendable {
    case text(TextRequest)             // messages + params + tools
    case imageText(ImageTextRequest)   // images + messages
    case video(VideoTextRequest)
    case audio(AudioRequest)           // waveform [+ prompt]  (STT / omni)
    case speak(SpeakRequest)           // text + voice         (TTS)
    case transform(AudioWaveform)      // STS
    case detect(AudioWaveform)         // VAD
    // future: .imageGen, .videoGen, .segment
}
public struct TextRequest: Sendable {
    public var messages: [ChatMessage]; public var parameters: GenerationParameters
    public var tools: [Tool]?
}
// …ImageTextRequest / AudioRequest / SpeakRequest analogous (struct per case).

public enum GenerationChunk: Sendable {           // streaming
    case textDelta(String); case audioSamples([Float]); case detections([DetectionEvent])
}
public enum GenerationResult: Sendable {          // batch
    case text(String); case transcript(TranscriptionResult)
    case audio([Float]); case detections([DetectionEvent])
}
```

### Unified `Model` (FFAI/Model.swift)

```swift
public final class Model {
    public let engine: any ModelEngine
    public let kind: ModelKind
    public let capabilities: Set<Capability>
    public let tokenizer: (any Tokenizer)?
    public let vlModel: VisionModel?

    public static func load(_ id: String, options:, device:) async throws -> Model  // ALL kinds

    // Generic surface — validates input vs kind, throws ModelError.unsupportedInput on mismatch.
    public func generate(_ input: ModelInput) async throws -> GenerationResult
    public func generateStream(_ input: ModelInput) -> AsyncThrowingStream<GenerationChunk, Error>

    // Typed convenience (wrap generate; same unsupportedInput guard).
    public func generateText(_ messages: [ChatMessage], parameters: GenerationParameters) async throws -> String
    public func transcribe(_ w: [Float], parameters: TranscribeParameters) async throws -> TranscriptionResult
    public func synthesize(_ text: String, voice: String?, parameters: AudioGenerationParameters) async throws -> [Float]
    public func detect(_ w: [Float]) async throws -> [DetectionEvent]
}
```
`generate` dispatches by `engine as? <ModalityProtocol>`; the existing
`Generation/Generate.driveGeneration` backs the text path.

### Decisions — RESOLVED at implementation (2026-06-05)

- **T1 — RESOLVED: keep `LanguageModel`, refine to `: TextEngine`.** `protocol LanguageModel: Module, TextEngine`. Zero churn on the ~40 conformers + call sites; `LanguageModel` stays the name everywhere. `TextEngine: ModelEngine` is the modality marker.
- **T2 — RESOLVED: keep `forwardMulti`.** Existing name across all engines; no rename.
- **T3 — RESOLVED: reuse `Generation/ChatTemplate.ChatMessage`.** No new chat type. `Tool` does not exist yet (chat template passes `tools: nil`); `TextRequest` omits tools for now — added when tool-calling is wired. The unified text path delegates to the existing `generate(messages:)` / `generateStream(messages:)`.
- **T4 — RESOLVED:** introduced `TranscribeParameters { maxNewTokens, temperature }` (maps 1:1 onto `StreamingTranscriber`'s args) and `DetectionEvent { startSample, endSample, sampleRate, label, score, speaker? }` (the common shape the three VAD output types — `VADOutput`, `VADEndpointOutput`, `DiarizationOutput` — map down to). `SynthesizeParameters { voice?, speed }` added for symmetry on the TTS side.
- **T5 — RESOLVED: typed-convenience methods are the primary CLI entry.** `transcribe`/`transcribeStream`/`synthesize`/`synthesizeStream`/`detect` on `Model` are what `ffai transcribe` / `ffai speak` call; the generic `generate(_:)` / `generateStream(_:)` sum-type surface is the programmatic API. CLI never switches on the output enum.
- **T6 (new, naming collision) — RESOLVED:** the plan's proposed `GenerationResult` / `GenerationChunk` *enum* names already exist as **structs** in `Generation/Generate.swift` (the text-generation batch/stream result). To avoid clobbering the deeply-wired text path, the unified modality sum types are named **`ModelOutput`** (batch) and **`ModelOutputChunk`** (streaming); input is **`ModelInput`**. The existing `GenerationResult`/`GenerationChunk` structs are untouched.
- **T7 (new, engine type) — RESOLVED:** `ModelEngine` is a minimal marker protocol (no structural requirements) so all 30+ heterogeneous audio/VAD concrete types conform for free. `Model.engine` retypes to `any ModelEngine`; a `Model.languageModel: (any LanguageModel)?` accessor (and internal `requireLanguageModel()`) backs the text/VL call sites. `Model.tokenizer` becomes optional (VAD/STS carry none). `ModelInfo` gains `kind` and tolerates a non-text engine (shape fields read from `languageModel` when present).
- **STS/SAMAudio — RESOLVED:** `SpeechTransforming { process(_:) async throws -> [Float] }` covers the waveform→waveform denoisers (DeepFilterNet, MossFormer2-SE). SAMAudio's promptable `segment(waveform:description:)` does not fit a uniform waveform→waveform contract, so it is reached via `model.engine as? SAMAudioModel` (kind `.sts`, capabilities `speechToSpeech`) rather than forced into `SpeechTransforming`.
