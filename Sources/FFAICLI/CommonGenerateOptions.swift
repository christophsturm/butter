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
// CommonGenerateOptions — the model-loading, KV-cache, sampling, and
// streaming flags shared by every generation subcommand (`generate
// text/image/video/audio`). Factored into a single `@OptionGroup` so the
// four commands stay DRY: the load-option parsing and the
// family-default + override merge live here once, not copy-pasted.

import ArgumentParser
import FFAI
import Foundation

/// Flags common to all generation subcommands. Embed with
/// `@OptionGroup var common: CommonGenerateOptions`.
struct CommonGenerateOptions: ParsableArguments {
    @Option(name: .shortAndLong, help: "HuggingFace repo id or local model path.")
    var model: String = "unsloth/Llama-3.2-1B"

    @Option(name: .long, help: "Maximum tokens to generate. Defaults to the model family's value.")
    var maxTokens: Int?

    @Flag(
        name: .long,
        help: "Print a [STATS] block (per-phase memory, tok/s, TTFT, KV cache, wired ticket).")
    var stats: Bool = false

    /// Stream tokens to stdout as they're generated. Default on — the
    /// CLI's primary route. Disable with `--no-streaming` to print the
    /// full text once at the end (bit-identical to the buffered API).
    @Flag(
        name: .long, inversion: .prefixedNo,
        help: "Stream tokens to stdout as they're generated (default). Disable with --no-streaming."
    )
    var streaming: Bool = true

    @Flag(
        name: .long,
        help: "Enable debug logging for every FFAI subsystem (loader, kernels, generate, ...).")
    var debug: Bool = false

    @Option(
        name: .long,
        help: "Profiling level: 0 (off), 1 (wallclock breakdown), 2 (level 1 + os_signpost).")
    var profiling: Int = 0

    // ─── Sampling knobs (override the model-family defaults) ─────────

    @Option(
        name: .long, help: "Sampling temperature. 0 = greedy argmax (deterministic, GPU fast path)."
    )
    var temperature: Float?

    @Option(name: .long, help: "Top-K cutoff. 0 = disabled.")
    var topK: Int?

    @Option(name: .long, help: "Top-P / nucleus cutoff. 1.0 = disabled.")
    var topP: Float?

    @Option(name: .long, help: "Min-P cutoff (Qwen-style). 0 = disabled.")
    var minP: Float?

    @Option(name: .long, help: "Repetition penalty. 1.0 = disabled.")
    var repetitionPenalty: Float?

    @Option(name: .long, help: "PRNG seed for reproducible sampling.")
    var seed: UInt64?

    // ─── KV cache / memory management ────────────────────────────────

    @Option(
        name: .long,
        help:
            "KV cache scheme: \"raw\" (default fp16/bf16), \"affine8\" (~45% smaller), \"affine4\" (~70% smaller), \"aura\" (Phase 5d default aura4v4 ~5x smaller), or any \"auraNvM\" recipe."
    )
    var kvCache: String?

    @Option(
        name: .long,
        help:
            "Maximum positions retained per attention layer. Past this, the cache evicts in FIFO order with the first --kv-window-keep slots pinned. 0 / unset means unbounded."
    )
    var kvWindowSize: Int?

    @Option(
        name: .long,
        help:
            "Number of initial positions kept across FIFO eviction (attention sinks). Default 0. Only meaningful when --kv-window-size is set."
    )
    var kvWindowKeep: Int?

    @Option(
        name: .long,
        help:
            "AURA decode-time attention path: \"compressed\" (default) or \"dequant-mirror\" (Stage 1a A/B). Ignored when --kv-cache isn't an aura scheme."
    )
    var auraDecodePath: String?

    // ─── Derived configuration ───────────────────────────────────────

    /// Apply `--debug` + `--profiling` global side effects. Call once at
    /// the top of `run()` before any FFAI work so the load path is
    /// captured.
    func applyGlobalFlags() throws {
        if debug { Debug.enableAll() }
        guard let lvl = ProfileLevel(rawValue: profiling) else {
            throw ValidationError("Invalid --profiling level \(profiling). Use 0, 1, or 2.")
        }
        Profile.shared.level = lvl
        Profile.shared.resetPhases()
    }

    /// Translate the CLI KV-cache flags into `LoadOptions`. Shared by all
    /// generation subcommands so the scheme parsing lives in one place.
    func loadOptions() throws -> LoadOptions {
        var loadOpts = LoadOptions()
        let rawKVKind = (kvCache ?? "raw").lowercased()
        switch rawKVKind {
        case "raw":
            loadOpts.kvCache = .raw
        case "affine8":
            loadOpts.kvCache = .affineQuantized(bits: 8, groupSize: 64)
        case "affine4":
            // group_size=16 at 4-bit — finer groups than affine8's 64.
            // gs64 decode is degenerate; gs16 is the first coherent size
            // (see KVCacheSchemeIntegrationTests). TurboQuant rotation
            // would let gs64 work at 4-bit; that's Phase 5d (AURA).
            loadOpts.kvCache = .affineQuantized(bits: 4, groupSize: 16)
        case _ where rawKVKind.hasPrefix("aura"):
            guard let scheme = AURAScheme.parse(rawKVKind) else {
                throw ValidationError(
                    "Unknown AURA recipe \"\(rawKVKind)\". Try \"aura\", \"aura4\", \"aura4v2\", \"aura3\", \"aura8\"."
                )
            }
            loadOpts.kvCache = .auraQuantized(scheme: scheme)
        default:
            throw ValidationError(
                "Unknown --kv-cache \"\(kvCache ?? "")\". Use \"raw\", \"affine8\", \"affine4\", or any \"auraNvM\" recipe."
            )
        }

        if let size = kvWindowSize, size > 0 {
            let keep = kvWindowKeep ?? 0
            loadOpts.kvEviction = .window(maxSize: size, keep: keep)
        } else if (kvWindowKeep ?? 0) != 0 {
            throw ValidationError("--kv-window-keep requires --kv-window-size to be set.")
        }

        if let raw = auraDecodePath?.lowercased() {
            switch raw {
            case "compressed":
                loadOpts.auraDecodePath = .compressed
            case "dequant-mirror", "dequant_mirror", "mirror":
                loadOpts.auraDecodePath = .dequantMirror
            default:
                throw ValidationError(
                    "Unknown --aura-decode-path \"\(auraDecodePath ?? "")\". Use \"compressed\" or \"dequant-mirror\"."
                )
            }
        }
        return loadOpts
    }

    /// Merge the family defaults with any explicit CLI sampling
    /// overrides. The model's `defaultGenerationParameters` is the base;
    /// every flag the user passed wins.
    func resolvedParameters(for model: Model) -> GenerationParameters {
        model.defaultGenerationParameters.with {
            if let n = maxTokens { $0.maxTokens = n }
            if let t = temperature { $0.temperature = t }
            if let k = topK { $0.topK = k }
            if let p = topP { $0.topP = p }
            if let mp = minP { $0.minP = mp }
            if let rp = repetitionPenalty { $0.repetitionPenalty = rp }
            if let s = seed { $0.seed = s }
        }
    }

    /// Load the model with the resolved KV-cache options, printing the
    /// standard "loading… / loaded in Xs" banner.
    func loadModel() async throws -> Model {
        print("ffai \(FFAI.version) — loading \(model)…")
        let loadStart = Date()
        let m = try await Model.load(model, options: try loadOptions())
        print("loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStart)))s")
        return m
    }

    /// Print the trailing prompt/generation stats footer shared by the
    /// generation subcommands.
    func printFooter(_ result: GenerationResult) {
        print("---")
        print(
            "prompt: \(result.promptTokens.count) tokens "
                + "(\(String(format: "%.2f", result.prefillTimeS))s prefill)")
        print(
            "generated: \(result.generatedTokens.count) tokens "
                + "in \(String(format: "%.2f", result.decodeTimeS))s "
                + "(\(String(format: "%.2f", result.tokensPerSecond)) tok/s)")
        if stats {
            print(result.stats.formatted())
        }
        if Profile.shared.level >= .wallclock {
            print(Profile.shared.phases.formatted())
        }
    }
}
