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
// GenerateCommand — the default `butter` subcommand. Runs one
// prompt → text generation, optionally streaming, with --stats /
// --debug / --profiling instrumentation.

import ArgumentParser
import ButterSwift
import Foundation

struct GenerateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate text from a single prompt."
    )

    // Parameter sanity. Without this, `--max-tokens=-1` (the `=` form
    // bypasses argument-parser range niceties) reached the generation
    // loop's `0 ..< maxTokens` range and died on a messageless SIGTRAP;
    // negative temperatures produced special-token garbage.
    func validate() throws {
        if let n = maxTokens, n < 0 {
            throw ValidationError("--max-tokens must be >= 0 (got \(n))")
        }
        if let t = temperature, t < 0 {
            throw ValidationError("--temperature must be >= 0 (got \(t))")
        }
    }

    @Option(name: .shortAndLong, help: "HuggingFace repo id or local model path.")
    var model: String = "unsloth/Llama-3.2-1B"

    @Option(name: .shortAndLong, help: "Prompt to generate from.")
    var prompt: String = "Once upon a time"

    @Option(name: .long, help: "Maximum tokens to generate. Defaults to the model family's value.")
    var maxTokens: Int?

    @Flag(name: .long, help: "Print top-5 next-token distribution instead of generating.")
    var verbose: Bool = false

    @Option(
        name: .long,
        help: "Diagnostic. Run the chunked prefill path over the prompt and write the final-position logits as raw little-endian f32 to this path, then exit.")
    var dumpPrefillLogits: String?

    @Flag(
        name: .long,
        help: "Print a [STATS] block (per-phase memory, tok/s, TTFT, KV cache, wired ticket).")
    var stats: Bool = false

    /// Stream tokens to stdout as they're generated. Default on.
    /// Disable with `--no-streaming` to print the full text once at the
    /// end (matches the buffered API exactly).
    @Flag(
        name: .long, inversion: .prefixedNo,
        help: "Stream tokens to stdout as they're generated (default). Disable with --no-streaming."
    )
    var streaming: Bool = true

    @Flag(
        name: .long,
        help: "Enable debug logging for every Butter subsystem (loader, kernels, generate, ...).")
    var debug: Bool = false

    @Option(
        name: .long,
        help:
            "Profiling level: 0 (off), 1 (wallclock breakdown), 2 (level 1 + os_signpost intervals)."
    )
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

    @Flag(
        name: .long,
        help:
            "Print the raw generated token ids as a JSON array after generation (scripted statistical-harness consumption, e.g. Loop E)."
    )
    var printTokenIds: Bool = false

    @Option(
        name: .long,
        help:
            "KV cache scheme: \"raw\" (default fp16/bf16), \"affine8\" (~45% smaller), \"affine4\" (~70% smaller), \"aura\" (Phase 5d default aura4v4 ~5x smaller), or any \"auraNvM\" recipe."
    )
    var kvCache: String?

    @Option(
        name: .long,
        help:
            "Maximum positions retained per attention layer. Past this, the cache evicts in FIFO order with the first --kv-keep slots pinned. 0 / unset means unbounded (cap at model max_position_embeddings)."
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
            "AURA decode-time attention path: \"compressed\" (default — attend directly on packed K/V codes via aura_flash_p1/pass2; realises AURA's ~4x memory savings) or \"dequant-mirror\" (Stage 1a — bulk-dequant the compressed cache into a full-precision mirror buffer per layer and use sdpaDecode; loses the memory savings but matches the Stage 1a code path for A/B benching). Ignored when --kv-cache isn't an aura scheme."
    )
    var auraDecodePath: String?

    @Option(
        name: .long,
        help: "Size the KV cache for this many positions. Default: model family pick."
    )
    var maxContext: Int?

    // ─── Nemotron-Labs-Diffusion tri-mode decoding ───────────────────

    @Option(
        name: .long,
        help:
            "Decoding mode: \"ar\" (autoregressive, default), \"diffusion\" (block-wise parallel), or \"self-spec\" (diffusion draft + AR verify). diffusion / self-spec require a Nemotron-Labs-Diffusion model."
    )
    var mode: String = "ar"

    @Option(name: .long, help: "Block length for diffusion / self-spec decoding. Default 32.")
    var blockLength: Int = 32

    @Option(
        name: .long,
        help:
            "Confidence threshold for diffusion-mode denoising (0..1). Unset uses an even per-step transfer budget."
    )
    var confidenceThreshold: Float?

    func run() async throws {
        // Apply --debug + --profiling before any Butter work so the
        // model-load path is captured.
        if debug { Debug.enableAll() }
        guard let lvl = ProfileLevel(rawValue: profiling) else {
            throw ValidationError("Invalid --profiling level \(profiling). Use 0, 1, or 2.")
        }
        Profile.shared.level = lvl
        Profile.shared.resetPhases()

        print("butter \(Butter.version) — loading \(model)…")
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

        if let dumpPath = dumpPrefillLogits {
            // Mirror driveGeneration's chunk loop so this exercises the exact
            // prefill path (forwardMulti in prefillStepSize chunks), not the
            // per-token forward the verbose flag uses.
            let promptTokens = m.tokenizer.encode(text: prompt)
            let caches = m.engine.makeLayerCaches()
            // Resolve chunk size through the SAME chain as the real generate
            // path (Generate.swift): env override, then the family/param-level
            // override, then the engine default. Skipping the middle link here
            // used to let a param-only chunk-size override go unseen by this
            // diagnostic, so a quality gate run through the dump could
            // validate a different chunk configuration than generation
            // actually uses.
            let chunkSize = max(
                1,
                ProcessInfo.processInfo.environment["BUTTER_PREFILL_STEP"].flatMap(Int.init)
                    ?? m.defaultGenerationParameters.prefillStepSize
                    ?? m.engine.defaultPrefillStepSize)
            var pos = 0
            var lastLogits: Tensor?
            while pos < promptTokens.count {
                let end = min(pos + chunkSize, promptTokens.count)
                let chunk = Array(promptTokens[pos ..< end])
                let cmd = Device.shared.makeCommandBuffer()
                lastLogits = m.engine.forwardMulti(
                    tokenIds: chunk, startingAt: pos,
                    caches: caches, on: cmd, device: Device.shared)
                cmd.commit()
                await cmd.completed()
                pos = end
            }
            guard let l = lastLogits else { return }
            let floats = l.toFloatArray()
            var data = Data(count: floats.count * 4)
            data.withUnsafeMutableBytes { raw in
                floats.withUnsafeBytes { src in
                    raw.copyMemory(from: src)
                }
            }
            try data.write(to: URL(fileURLWithPath: dumpPath))
            print("wrote \(floats.count) f32 logits to \(dumpPath)")
            return
        }

        if verbose {
            // Run prefill once, print top-5 next tokens with their logits.
            // Useful for sanity-checking distributions without committing
            // to a sampling strategy.
            let promptTokens = m.tokenizer.encode(text: prompt)
            print("prompt tokens: \(promptTokens)")

            let caches = m.engine.makeLayerCaches()
            var lastLogits: Tensor?
            for (i, t) in promptTokens.enumerated() {
                lastLogits = m.engine.forward(tokenId: t, position: i, caches: caches)
            }
            if let l = lastLogits {
                print("top-5 next tokens:")
                for (id, v) in Sampling.topN(l, n: 5) {
                    let s = m.tokenizer.decode(tokens: [id], skipSpecialTokens: false)
                    print("  \(id) (\(String(format: "%.4f", v)))  \"\(s)\"")
                }
            }
            return
        }

        // Non-autoregressive modes — block-wise diffusion or linear
        // self-speculation (Nemotron-Labs-Diffusion only).
        let genMode = mode.lowercased()
        if genMode != "ar" {
            guard genMode == "diffusion" || genMode == "self-spec" || genMode == "selfspec"
            else {
                throw ValidationError(
                    "Unknown --mode \"\(mode)\". Use ar, diffusion, or self-spec.")
            }
            guard m.nemotronLabsDiffusion != nil else {
                throw ValidationError(
                    "--mode \(mode) requires a Nemotron-Labs-Diffusion model "
                        + "(\(model) does not support it).")
            }
            let isDiffusion = genMode == "diffusion"
            // Diffusion requires maxNewTokens to be a multiple of the
            // block length; round down (min one block).
            let requested = maxTokens ?? 128
            let maxNew =
                isDiffusion
                ? max(blockLength, (requested / blockLength) * blockLength)
                : requested
            let diffParams = DiffusionParameters(
                maxNewTokens: maxNew, blockLength: blockLength,
                confidenceThreshold: isDiffusion ? confidenceThreshold : nil)

            print("---")
            print(prompt)
            let start = Date()
            let result =
                isDiffusion
                ? m.generateDiffusion(prompt: prompt, parameters: diffParams)
                : m.generateSelfSpeculative(prompt: prompt, parameters: diffParams)
            let elapsed = Date().timeIntervalSince(start)
            print(result.text)
            print("---")
            let tps = Double(result.generatedTokens.count) / max(elapsed, 1e-9)
            print(
                "generated: \(result.generatedTokens.count) tokens in "
                    + "\(String(format: "%.2f", elapsed))s "
                    + "(\(String(format: "%.2f", tps)) tok/s, "
                    + "\(result.forwardPasses) forward passes, "
                    + "\(String(format: "%.2f", Double(result.generatedTokens.count) / Double(result.forwardPasses))) tokens/forward)"
            )
            return
        }

        // Family defaults + any explicit CLI overrides.
        let params = m.defaultGenerationParameters.with {
            if let n = maxTokens { $0.maxTokens = n }
            if let t = temperature { $0.temperature = t }
            if let k = topK { $0.topK = k }
            if let p = topP { $0.topP = p }
            if let mp = minP { $0.minP = mp }
            if let rp = repetitionPenalty { $0.repetitionPenalty = rp }
            if let s = seed { $0.seed = s }
        }

        print("---")
        print(prompt, terminator: "")

        let result = try await runGenerate(model: m, params: params)

        print("---")
        print(
            "prompt: \(result.promptTokens.count) tokens "
                + "(\(String(format: "%.2f", result.prefillTimeS))s prefill)")
        print(
            "generated: \(result.generatedTokens.count) tokens "
                + "in \(String(format: "%.2f", result.decodeTimeS))s "
                + "(\(String(format: "%.2f", result.tokensPerSecond)) tok/s)")
        if printTokenIds {
            let ids = result.generatedTokens.map(String.init).joined(separator: ",")
            print("token ids: [\(ids)]")
        }
        if stats {
            print(result.stats.formatted())
        }
        if Profile.shared.level >= .wallclock {
            print(Profile.shared.phases.formatted())
        }
        // T39: BUTTER_LAGUNA_DECODE_KERNEL_TIMING=1 per-kernel decode probe.
        // This run's own `--stats` tok/s is NOT the real anchor: forcing a
        // commit+wait at every checkpoint inflates it (same class of
        // inflation T26/T31 already flagged for prefill's forced-sync phase
        // timing). Pass the real (separately-measured, plain --stats,
        // timing OFF) decode tok/s via BUTTER_LAGUNA_DECODE_REAL_TOKS for
        // the share-projection column; omit it to see raw forced-sync
        // numbers only.
        // The per-N-token probe (BUTTER_LAGUNA_DECODE_PROBE=N, see
        // GpuTokenLoop) toggles the KERNEL_TIMING var around each probe
        // token, so also gate the report on the probe var itself.
        let decodeProbeOn =
            (Int(ButterEnv["BUTTER_LAGUNA_DECODE_PROBE"] ?? "") ?? 0) > 0
        if LagunaDecodeKernelTiming.enabled || decodeProbeOn {
            let realUsPerToken: Double?
            if let realToks = ProcessInfo.processInfo.environment["BUTTER_LAGUNA_DECODE_REAL_TOKS"]
                .flatMap(Double.init), realToks > 0
            {
                realUsPerToken = 1_000_000.0 / realToks
            } else {
                realUsPerToken = nil
            }
            LagunaDecodeKernelTiming.report(realUsPerToken: realUsPerToken)
        }
    }

    /// Drive generation either by consuming the stream chunk-by-chunk
    /// (printing as we go) or by collecting once at the end. The
    /// producer loop is identical either way; the only difference is
    /// who prints when. Stream consumption with per-token print adds
    /// ~µs of stdio per token vs ~ms of decode — stats are unaffected.
    private func runGenerate(
        model: Model,
        params: GenerationParameters
    ) async throws -> GenerationResult {
        if streaming {
            var generated: [Int] = []
            var text = ""
            var stats: GenerationStats?
            for try await chunk in model.generateStream(prompt: prompt, parameters: params) {
                if !chunk.text.isEmpty {
                    print(chunk.text, terminator: "")
                    fflush(stdout)
                }
                generated.append(contentsOf: chunk.tokens)
                text += chunk.text
                if let s = chunk.stats { stats = s }
            }
            print("")  // newline after the streamed text
            return GenerationResult(
                promptTokens: model.encodePrompt(prompt),
                generatedTokens: generated, text: text,
                stats: stats!  // streamEndedWithoutFinalChunk would have thrown above
            )
        } else {
            let result = try await model.generate(prompt: prompt, parameters: params)
            print(result.text)
            return result
        }
    }
}
