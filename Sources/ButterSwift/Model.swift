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
// Model — public entry point users interact with. Resolves a model
// id-or-path, downloads via HF if needed, decodes config, dispatches to
// the right family file, loads weights, and exposes a forward()/generate()
// surface.

import Foundation
import Metal
import Tokenizers

public enum ModelError: Error, CustomStringConvertible {
    case unsupportedArchitecture(String)
    case unsupportedModelType(String)
    case capabilityNotAvailable(Capability)
    case visionModelNotIntegrated(String)
    /// A `ModelInput` was passed to `Model.generate` / a typed convenience
    /// method that the loaded model's `kind` can't service — e.g. `.speak`
    /// to an LLM, or `.text` to a VAD model. Carries the model's kind and a
    /// short description of the rejected input.
    case unsupportedInput(kind: ModelType, input: String)
    /// A speculative-decoding draft / assistant model (e.g. the Gemma 4
    /// Omni `gemma4_unified_assistant` EAGLE drafter) was loaded as a
    /// standalone generation model. It cannot generate on its own — it
    /// drafts tokens conditioned on the target model's hidden state, which
    /// the target then verifies. Wiring it into the spec-decode path is
    /// tracked as Phase 8.7.
    case draftModelNotGeneratable(String)
    /// The model's weights + a minimal KV cache exceed the wired-memory
    /// budget — it can't run at any usable context on this machine /
    /// budget. Raised by the over-allocation guard (`MemoryBudget`).
    case insufficientMemory(weightBytes: Int, budgetBytes: Int, detail: String)
    /// An explicit `LoadOptions.wiredLimitBytes` exceeds the safe maximum
    /// (physical RAM − the OS reserve). Raised at load so a deliberate
    /// over-request is reported rather than silently clamped.
    case wiredLimitTooHigh(requestedBytes: Int, safeMaxBytes: Int, osReserveBytes: Int)

    public var description: String {
        switch self {
        case .unsupportedArchitecture(let a): return "Unsupported architecture: \(a)"
        case .unsupportedModelType(let m): return "Unsupported model_type: \(m)"
        case .capabilityNotAvailable(let c): return "Capability not available: \(c)"
        case .unsupportedInput(let kind, let input):
            return "Unsupported input for a \(kind.rawValue) model: \(input)"
        case .visionModelNotIntegrated(let a):
            return "Vision-language checkpoint '\(a)' detected. The Butter "
                + "vision foundation (VisionEncoder, ImagePreprocessing, "
                + "VisionModel cross-modal splice, conv2d/patch_embed/rope_2d "
                + "Ops) is in tree, but this VL family is not yet wired to "
                + "a checkpoint loader. Load the text-only checkpoint, or "
                + "compose a VisionModel directly from VisionEncoder + the text "
                + "engine."
        case .draftModelNotGeneratable(let a):
            return "Speculative-decoding draft model '\(a)' cannot generate "
                + "standalone — it is an EAGLE/MTP drafter that predicts tokens "
                + "from the target model's hidden state for the target to verify. "
                + "Load the target model (e.g. the Gemma 4 Omni 12B), or use the "
                + "typed `Gemma4Draft.load(...)` for spec-decode wiring "
                + "(Phase 8.7)."
        case .insufficientMemory(_, _, let detail):
            return "Insufficient memory: \(detail)"
        case .wiredLimitTooHigh(let requested, let safeMax, let reserve):
            let gb = { (b: Int) in String(format: "%.2f GB", Double(b) / 1_073_741_824.0) }
            return "LoadOptions.wiredLimitBytes (\(gb(requested))) exceeds the safe "
                + "maximum \(gb(safeMax)) (physical RAM minus a \(gb(reserve)) OS "
                + "reserve). Lower wiredLimitBytes to \(gb(safeMax)) or less, or free "
                + "memory / use a machine with more RAM."
        }
    }
}

/// High-level loaded model with tokenizer attached. The public API users
/// touch.
public final class Model: @unchecked Sendable {
    /// The concrete model engine, type-erased to the unified `ModelEngine`
    /// existential — a `TextGenerating` (LLM / VLM text backbone), a
    /// `Transcribing` (STT), a `Speaking` (TTS), a `VoiceActivating` (VAD), or
    /// a `SpeechTransforming` (STS). For a text/VL model this is the text
    /// engine; use `languageModel` / `textEngine` to recover the typed text
    /// surface, or the typed convenience accessors (`llama`, `qwen3`, …) for
    /// a specific family.
    public let engine: any ModelEngine
    /// The catalog kind the registry filed this model under. The dispatch
    /// key for `generate(_:)` / `generateStream(_:)`.
    public let kind: ModelType
    /// The model's tokenizer. `nil` for kinds that don't tokenize text
    /// (VAD, STS, and some pure audio transforms).
    public let tokenizer: (any Tokenizer)?
    public let config: ModelConfig
    public let modelDirectory: URL
    public let availableCapabilities: Set<Capability>

    /// The engine viewed as a text `TextGenerating`, or `nil` for non-text
    /// kinds (STT / TTS / STS / VAD). The shim every existing text/VL call
    /// site recovers the forward / cache surface through.
    public var languageModel: (any TextGenerating)? { engine as? any TextGenerating }

    /// The text engine, for the text/VL code paths. Precondition-fails on a
    /// non-text kind — a caller bug; guard on `kind` / `languageModel` first.
    /// The convenience shim text commands (generate / inspect / bench) reach
    /// the forward / cache surface through; programmatic callers that may hold
    /// a non-text model should use the optional `languageModel` instead.
    public var textEngine: any TextGenerating {
        guard let lm = engine as? any TextGenerating else {
            preconditionFailure(
                "Model.textEngine on a non-text engine (kind=\(kind.rawValue))")
        }
        return lm
    }

    /// The tokenizer, for the text/VL code paths. Precondition-fails when the
    /// model carries none (VAD / STS) — a caller bug; text generation is only
    /// reached for kinds that tokenize. Programmatic callers that may hold a
    /// non-text model should use the optional `tokenizer` instead.
    public var textTokenizer: any Tokenizer {
        guard let t = tokenizer else {
            preconditionFailure(
                "Model.textTokenizer on a model with no tokenizer "
                    + "(kind=\(kind.rawValue))")
        }
        return t
    }

    /// The model's chat template, when it ships as a standalone
    /// `chat_template.jinja` file (the newer HF convention) rather than a
    /// `chat_template` field inside `tokenizer_config.json` (which
    /// swift-transformers loads into the tokenizer itself). `nil` when the
    /// tokenizer already carries a template or no `.jinja` file is present.
    /// `renderChatTemplate` passes this as a literal override so instruct /
    /// reasoning models that only ship the standalone file (e.g.
    /// LiquidAI/LFM2.5) still format correctly through `generate(messages:)`.
    public var standaloneChatTemplate: String? {
        guard tokenizer?.hasChatTemplate != true else { return nil }
        let url = modelDirectory.appendingPathComponent("chat_template.jinja")
        return try? String(contentsOf: url, encoding: .utf8)
    }

    /// True when chat-formatted generation is available — either the
    /// tokenizer carries a template or a standalone `chat_template.jinja`
    /// was found alongside the model.
    public var supportsChatTemplate: Bool {
        tokenizer?.hasChatTemplate == true || standaloneChatTemplate != nil
    }

    /// The `LoadOptions` this model was loaded with. Retained so the
    /// memory-management path (`makeManagedCaches`) can honor
    /// `maxContextLength` / `preallocateKVCache` / `initialKVCacheCapacity`
    /// / `wiredLimitBytes` at cache-creation time, uniformly across
    /// every family.
    public let loadOptions: LoadOptions

    /// The composed vision-language model — `nil` unless the checkpoint
    /// is a VLM. Use `vlModel.generate(...)` for an image+text prompt;
    /// available only when `availableCapabilities` contains `.imageIn`.
    public let vlModel: VisionModel?

    /// Currently-enabled capabilities. Mutated via `enable(_:)` /
    /// `disable(_:)`; guarded by `capabilityLock` for thread safety.
    private var _enabledCapabilities: Set<Capability>
    private let capabilityLock = NSLock()

    /// Snapshot of the enabled-capability set.
    public var enabledCapabilities: Set<Capability> {
        capabilityLock.lock()
        defer { capabilityLock.unlock() }
        return _enabledCapabilities
    }
    /// Default generation parameters declared by the model's family
    /// variant. Use as-is, or call `.with { $0.maxTokens = ... }` to
    /// tweak a field without losing the family-tuned baseline.
    public let defaultGenerationParameters: GenerationParameters

    /// Convenience accessor for tests + tools that want the Llama-typed
    /// model. Returns nil if the loaded engine isn't Llama3.
    public var llama: Llama3Model? { engine as? Llama3Model }

    /// Convenience accessor for the Qwen3 engine.
    public var qwen3: Qwen3Model? { engine as? Qwen3Model }

    /// Convenience accessor for the Mamba 2 engine.
    public var mamba2: Mamba2Model? { engine as? Mamba2Model }

    /// Convenience accessor for the SmolVLM2 engine.
    public var smolVLM2: SmolVLM2Model? { engine as? SmolVLM2Model }

    /// Convenience accessor for the FalconH1 hybrid engine.
    public var falconH1: FalconH1Model? { engine as? FalconH1Model }

    /// Convenience accessor for the NemotronH hybrid engine.
    public var nemotronH: NemotronHModel? { engine as? NemotronHModel }

    /// Convenience accessor for the Granite4 hybrid engine.
    public var graniteMoeHybrid: Granite4Model? { engine as? Granite4Model }

    /// Convenience accessor for the Jamba hybrid engine.
    public var jamba: JambaModel? { engine as? JambaModel }

    /// Convenience accessor for the Qwen3.5 hybrid engine.
    public var qwen35: Qwen35Model? { engine as? Qwen35Model }

    /// Convenience accessor for the Gemma 2 engine.
    public var gemma2: Gemma2Model? { engine as? Gemma2Model }

    /// Convenience accessor for the GPT-OSS MoE engine.
    public var gptOSS: GPTOSSModel? { engine as? GPTOSSModel }

    /// Convenience accessor for the LFM2 hybrid engine.
    public var lfm2: LFM2Model? { engine as? LFM2Model }

    /// Convenience accessor for the Nemotron-Labs-Diffusion engine.
    public var nemotronLabsDiffusion: NemotronDiffusionModel? { engine as? NemotronDiffusionModel }

    /// Convenience accessor for the Starcoder 2 engine. Returns nil if
    /// the loaded engine isn't Starcoder 2. Note: Starcoder 2 was
    /// previously misrouted through the Llama loader and `m.llama`
    /// returned a (broken) Llama wrapper — `m.starcoder2` is the
    /// correct accessor for Starcoder2ForCausalLM checkpoints.
    public var starcoder2: Starcoder2Model? { engine as? Starcoder2Model }

    /// Convenience accessor for the OLMo 2 engine. Returns nil if the
    /// loaded engine isn't OLMo 2. OLMo 2 (`Olmo2ForCausalLM`) is
    /// post-norm with q/k-norm and routes to its own decoder — not the
    /// Llama path — so `m.llama` is nil for OLMo 2 checkpoints. (OLMo 1
    /// stays Llama-shaped and does surface through `m.llama`.)
    public var olmo2: OLMo2Model? { engine as? OLMo2Model }

    private let stateLock = NSLock()
    private var _currentState: ModelLifecycleState = .ready

    public var currentState: ModelLifecycleState {
        stateLock.lock()
        defer { stateLock.unlock() }
        return _currentState
    }

    /// Maximum number of lifecycle events buffered when no consumer is
    /// reading from `events`. The default `AsyncStream` policy is
    /// `.unbounded`, which leaks events forever if nobody subscribes
    /// (the common case — most callers don't attach an `events` task).
    /// 64 is well above the typical event count per generation
    /// (~6: idle → loading → ready → generating → idle, plus a few
    /// capability flips) but small enough that the unconsumed-events
    /// retention is bounded.
    public static let eventsBufferCapacity = 64

    public let events: AsyncStream<ModelLifecycleEvent>
    private let eventsContinuation: AsyncStream<ModelLifecycleEvent>.Continuation

    /// Weight buffers pinned into the device's residency set at load,
    /// released in `deinit` so the wired memory doesn't accumulate across
    /// models loaded in one process. Empty for models built outside the
    /// `Model.load` path (e.g. tests) — `unmarkWeightsResident([])` no-ops.
    private let residentWeightBuffers: [MTLBuffer]
    /// The device whose residency set holds `residentWeightBuffers`.
    private let residencyDevice: Device

    init(
        engine: any ModelEngine, kind: ModelType = .llm,
        tokenizer: (any Tokenizer)?, config: ModelConfig,
        modelDirectory: URL,
        availableCapabilities: Set<Capability>,
        enabledCapabilities: Set<Capability>,
        defaultGenerationParameters: GenerationParameters,
        loadOptions: LoadOptions = LoadOptions(),
        vlModel: VisionModel? = nil,
        residentWeightBuffers: [MTLBuffer] = [],
        residencyDevice: Device = .shared
    ) {
        self.engine = engine
        self.kind = kind
        self.tokenizer = tokenizer
        self.config = config
        self.modelDirectory = modelDirectory
        self.availableCapabilities = availableCapabilities
        self.loadOptions = loadOptions
        self.vlModel = vlModel
        self.residentWeightBuffers = residentWeightBuffers
        self.residencyDevice = residencyDevice
        // textIn / textOut are universal — always enabled. Other
        // requested capabilities are honored only if the model declares
        // them available.
        self._enabledCapabilities =
            enabledCapabilities
            .union(Capability.textOnly)
            .intersection(availableCapabilities.union(Capability.textOnly))
        self.defaultGenerationParameters = defaultGenerationParameters
        // Bounded buffer — when no consumer is reading, keep the most
        // recent `eventsBufferCapacity` events and drop older ones.
        // Avoids the unbounded-growth leak from the default policy.
        let (stream, cont) = AsyncStream<ModelLifecycleEvent>.makeStream(
            bufferingPolicy: .bufferingNewest(Self.eventsBufferCapacity)
        )
        self.events = stream
        self.eventsContinuation = cont
    }

    deinit {
        // Release this model's weights from the device residency set so
        // the wired memory is freed — otherwise every model loaded in a
        // process stays resident and memory accumulates across loads.
        residencyDevice.unmarkWeightsResident(residentWeightBuffers)
        eventsContinuation.finish()
    }

    fileprivate func emit(_ event: ModelLifecycleEvent) {
        stateLock.lock()
        _currentState = event.state
        stateLock.unlock()
        eventsContinuation.yield(event)
    }

    // ─── Capability enable / disable ─────────────────────────────────

    /// Whether a capability is currently enabled.
    public func isEnabled(_ capability: Capability) -> Bool {
        enabledCapabilities.contains(capability)
    }

    /// Enable a capability at runtime — e.g. `enable(.imageIn)` lights
    /// up the vision path on a model loaded text-only. No-op if the
    /// capability isn't in `availableCapabilities` (a text-only model
    /// can't gain vision) or is already enabled. Emits a lifecycle
    /// event tagged with the capability so consumers can react.
    ///
    /// `textIn` / `textOut` are universal and always enabled — calling
    /// `enable` / `disable` on them is a harmless no-op.
    @discardableResult
    public func enable(_ capability: Capability) -> Bool {
        guard
            availableCapabilities.contains(capability)
                || Capability.textOnly.contains(capability)
        else { return false }
        capabilityLock.lock()
        let changed = !_enabledCapabilities.contains(capability)
        _enabledCapabilities.insert(capability)
        capabilityLock.unlock()
        if changed {
            eventsContinuation.yield(
                ModelLifecycleEvent(capability: capability, state: currentState))
        }
        return changed
    }

    /// Disable a capability at runtime. `textIn` / `textOut` are
    /// universal and cannot be disabled — those calls are a no-op.
    /// Emits a capability-tagged lifecycle event when the set changes.
    @discardableResult
    public func disable(_ capability: Capability) -> Bool {
        guard !Capability.textOnly.contains(capability) else { return false }
        capabilityLock.lock()
        let changed = _enabledCapabilities.contains(capability)
        _enabledCapabilities.remove(capability)
        capabilityLock.unlock()
        if changed {
            eventsContinuation.yield(
                ModelLifecycleEvent(capability: capability, state: currentState))
        }
        return changed
    }

    // ─── Top-level loader ────────────────────────────────────────────

    /// Resolve an id-or-path, download if needed, decode config, load
    /// weights, build the family-specific model, attach tokenizer.
    public static func load(
        _ idOrPath: String,
        options: LoadOptions = LoadOptions(),
        device: Device = .shared
    ) async throws -> Model {
        Debug.log(.load, "Model.load id-or-path=\(idOrPath)")
        // Fail fast on an explicit wired-memory ticket that would starve
        // the OS (physical RAM minus the 8 GB reserve), before any heavy
        // load work. No-op when wiredLimitBytes isn't set.
        try MemoryBudget.validateWiredLimit(options: options, device: device)
        let model = try await Profile.timeAsync("model_load") {
            try await Profile.signpostAsync("model_load") {
                let locator = ModelLocator(
                    downloader: ModelDownloader(cacheDirectory: options.cacheDirectory))
                let dir = try await locator.resolve(idOrPath: idOrPath, revision: options.revision)
                Debug.log(.loader, "resolved snapshot dir: \(dir.path)")
                let config = try ModelConfig.load(from: dir)
                Debug.log(
                    .load,
                    "config: arch=\(config.architecture ?? "?") model_type=\(config.modelType ?? "?") hidden=\(config.hiddenSize ?? 0) layers=\(config.numLayers ?? 0)"
                )
                // Audio / VAD families load from the directory (they build
                // their own weight bundles internally — some async — and
                // attach their own tokenizer), so they bypass the top-level
                // bundle build + residency pin and the LoRA hook below.
                if let entry = ModelRegistry.entry(for: config),
                    let dirLoad = entry.loadFromDirectory
                {
                    var loaded = try await dirLoad(dir, config, options, device)
                    loaded.kind = entry.kind
                    return Model(
                        engine: loaded.engine, kind: loaded.kind,
                        tokenizer: loaded.tokenizer, config: config,
                        modelDirectory: dir,
                        availableCapabilities: loaded.availableCapabilities,
                        enabledCapabilities: options.capabilities,
                        defaultGenerationParameters: loaded.defaultGenerationParameters,
                        loadOptions: options,
                        vlModel: loaded.vlModel,
                        residencyDevice: device)
                }

                // Text / VL path: pre-built, residency-pinned weights.
                let bundle = try SafeTensorsBundle(directory: dir, device: device)
                // Pin every weight buffer in a persistent MTLResidencySet
                // so prefill / decode dispatches skip per-allocation
                // residency tracking. macOS 15+ / iOS 18+; older OSes
                // and `BUTTER_NO_RESIDENCY_SET=1` no-op via
                // `Device.markWeightsResident`.
                var weightBuffers: [MTLBuffer] = []
                for file in bundle.files {
                    for entry in file.entries.values {
                        weightBuffers.append(entry.buffer)
                    }
                }
                device.markWeightsResident(weightBuffers)
                let loaded = try ModelRegistry.dispatchAndLoad(
                    config: config, weights: bundle, options: options, device: device
                )
                // Nemotron-Labs-Diffusion ships an optional
                // `linear_spec_lora` adapter that sharpens the
                // self-speculation diffusion drafter. Generalized: any
                // LoRA-adaptable engine auto-attaches its bundled adapter here
                // — `attachLoRA` no-ops when the checkpoint shipped none.
                if let adaptable = loaded.engine as? any LoRAAdaptable {
                    adaptable.attachLoRA(from: dir, device: device)
                }
                let tokenizer = try await TokenizerLoader().load(from: dir)
                return Model(
                    engine: loaded.engine, kind: loaded.kind, tokenizer: tokenizer,
                    config: config,
                    modelDirectory: dir,
                    availableCapabilities: loaded.availableCapabilities,
                    enabledCapabilities: options.capabilities,
                    defaultGenerationParameters: loaded.defaultGenerationParameters,
                    loadOptions: options,
                    vlModel: loaded.vlModel,
                    residentWeightBuffers: weightBuffers,
                    residencyDevice: device
                )
            }
        }

        // Prewarm just touches the embedding lookup once so the PSO is
        // compiled before the first user-visible decode. Captured as a
        // separate phase so `--profiling 1` shows it broken out from
        // model_load.
        if options.prewarm {
            await Profile.timeAsync("prewarm") {
                await Profile.signpostAsync("prewarm") {
                    await model.prewarm()
                }
            }
        }

        model.emit(ModelLifecycleEvent(state: .ready))
        return model
    }

    /// Compile PSOs for the kernels we'll need during decode by running
    /// one no-op forward step. Costs ~100ms-1s on first load.
    ///
    /// The cache only ever holds the single position-0 token this
    /// forward appends, so it is sized to a tiny fixed depth rather
    /// than the model's full context window. Without the cap, a
    /// long-context model (e.g. Qwen3.6-27B at max_position_embeddings
    /// = 262144) would pre-allocate ~17 GB of KV cache here — at LOAD
    /// time, before the caller has even asked to generate — and could
    /// exhaust unified memory on its own. PSO compilation depends on
    /// kernel shapes, not cache depth, so a depth of a few slots
    /// compiles exactly the same pipeline states.
    public func prewarm() async {
        // Prewarm is a text-engine concept (compile the decode PSOs). Audio /
        // VAD engines have no token-forward path — skip.
        guard let lm = languageModel else { return }
        let cache = lm.makeLayerCaches(maxSeq: Self.prewarmCacheDepth)
        _ = lm.forward(tokenId: 0, position: 0, caches: cache)
    }

    /// Cache depth used by `prewarm()`. A single position-0 forward
    /// needs only one slot; the small margin is defensive headroom for
    /// any family whose forward touches position+1 internally.
    private static let prewarmCacheDepth = 8

    // ─── Managed KV-cache creation (the memory-management chokepoint) ──

    /// Resolve the KV-cache context ceiling for a session: the model's
    /// `max_position_embeddings`, narrowed by `LoadOptions.maxContextLength`,
    /// then by the optional `generationBudget` (prompt + maxTokens), then
    /// clamped by the over-allocation guard so weights + max-KV + a
    /// working-memory margin fit the wired-memory budget. Throws
    /// `ModelError.insufficientMemory` if the model can't run at any
    /// usable context on this machine / budget.
    ///
    /// Every cache-creation path (generation, CLI inspect/generate)
    /// routes through here, so the cap + guard apply uniformly across
    /// every model family without per-family wiring.
    public func resolvedKVContextCeiling(
        generationBudget: Int? = nil, device: Device = .shared
    ) throws -> Int {
        let lm = textEngine
        var ceiling = lm.maxContextWindow
        if let mc = loadOptions.maxContextLength { ceiling = Swift.min(ceiling, mc) }
        if let gb = generationBudget { ceiling = Swift.min(ceiling, gb) }
        ceiling = try MemoryBudget.clampContext(
            requestedCeiling: ceiling, engine: lm,
            options: loadOptions, device: device)
        return Swift.max(1, ceiling)
    }

    /// Build the per-layer caches for a session with memory management
    /// applied: the context ceiling is resolved + guarded (see
    /// `resolvedKVContextCeiling`), and `LoadOptions.preallocateKVCache`
    /// / `initialKVCacheCapacity` are honored by setting the KVCache
    /// initial-capacity knob around the (synchronous) construction.
    ///
    /// `preallocateKVCache` collapses to "start at the full ceiling"
    /// (no growth). The knob is saved + restored around the build so it
    /// doesn't leak into other models' cache creation.
    public func makeManagedCaches(
        generationBudget: Int? = nil, device: Device = .shared
    ) throws -> [any LayerCacheProtocol] {
        let lm = textEngine
        let ceiling = try resolvedKVContextCeiling(
            generationBudget: generationBudget, device: device)

        // Common path: no per-load initial-capacity override → build at
        // the global default, touching no shared state.
        guard loadOptions.preallocateKVCache || loadOptions.initialKVCacheCapacity != nil
        else {
            return lm.makeLayerCaches(maxSeq: ceiling, device: device)
        }

        // Override path: `preallocateKVCache` collapses to "start at the
        // full ceiling"; `initialKVCacheCapacity` sets a custom start.
        // We express both through the `KVCache.defaultInitialCapacity`
        // knob around the SYNCHRONOUS build, saving + restoring it so the
        // override doesn't leak to other models. (The save/restore window
        // is synchronous — no `await` — so a single call is self-consistent;
        // simultaneous generations on *different* models that both set an
        // override could still race the global, but that only affects the
        // initial allocation SIZE, never correctness, since growth
        // reconstructs the cache faithfully regardless of where it starts.)
        let saved = KVCache.defaultInitialCapacity
        KVCache.defaultInitialCapacity =
            loadOptions.preallocateKVCache
            ? ceiling
            : Swift.max(1, loadOptions.initialKVCacheCapacity ?? saved)
        defer { KVCache.defaultInitialCapacity = saved }
        return lm.makeLayerCaches(maxSeq: ceiling, device: device)
    }
}

// ─── Hot LoRA adapter management ─────────────────────────────────────

extension Model {
    /// The loaded engine viewed as `LoRAAdaptable`, or `nil` when this model's
    /// family doesn't support adapters. The single delegation point so the
    /// `Model` wrapper never special-cases a concrete family.
    private var loraAdaptable: (any LoRAAdaptable)? { engine as? any LoRAAdaptable }

    /// Whether a LoRA adapter is currently attached. `false` for families that
    /// don't conform to `LoRAAdaptable`.
    public var hasLoRA: Bool { loraAdaptable?.hasLoRA ?? false }

    /// Hot-load a LoRA adapter at runtime. `directory` may be the model
    /// directory (the adapter is resolved under the family's adapter
    /// subfolder) or a directory holding the adapter outright — so the same
    /// call swaps in the bundled adapter or an external one. Any
    /// currently-attached adapter is replaced. No-op on families that don't
    /// support adapters. Do not call during an active generate.
    public func loadLoRA(from directory: URL, device: Device = .shared) {
        guard let adaptable = loraAdaptable else { return }
        adaptable.detachLoRA()
        adaptable.attachLoRA(from: directory, device: device)
    }

    /// Hot-unload the current LoRA adapter. No-op when none is attached.
    public func unloadLoRA() { loraAdaptable?.detachLoRA() }
}
