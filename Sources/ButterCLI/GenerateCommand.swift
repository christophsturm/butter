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
// GenerateCommand — the `butter generate` command group. It dispatches to
// one subcommand per input modality:
//
//   butter generate text  "<prompt>"     text → text (default)
//   butter generate image <file>         image + prompt → text
//   butter generate video <file>         video + prompt → text
//   butter generate audio <file>         audio + prompt → text (omni)
//
// `generate` with no subcommand runs `text` (so `butter generate "hi"` and
// the root-level `butter "hi"` still work). Streaming is the primary route
// for every modality; pass `--no-streaming` to buffer.

import ArgumentParser
import ButterSwift
import Foundation

struct GenerateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate text from a prompt and optional image / video / audio input.",
        subcommands: [
            GenerateTextCommand.self,
            GenerateImageCommand.self,
            GenerateVideoCommand.self,
            GenerateAudioCommand.self,
        ],
        defaultSubcommand: GenerateTextCommand.self
    )
}

/// How `--prompt` / the positional prompt is fed to the model.
enum GenerateMode: String, ExpressibleByArgument, CaseIterable {
    /// Feed the prompt verbatim (base / completion models).
    case raw
    /// Wrap the prompt as a single user turn and render it through the
    /// model's chat template (instruct / reasoning models).
    case chat
}

/// `butter generate text` — the original single-prompt text generator.
struct GenerateTextCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "text",
        abstract: "Generate text from a single prompt."
    )

    @OptionGroup var common: CommonGenerateOptions

    @Argument(help: "Prompt to generate from. Omit to use a sample prompt.")
    var prompt: String = "Once upon a time"

    @Option(
        name: .long,
        help: ArgumentHelp(
            "How the prompt is fed: 'raw' (default, verbatim — base/completion models) or "
                + "'chat' (wrap as a user turn through the model's chat template — instruct/"
                + "reasoning models like LFM2.5, which degenerate on raw input)."))
    var mode: GenerateMode = .raw

    @Flag(name: .long, help: "Print top-5 next-token distribution instead of generating.")
    var verbose: Bool = false

    // Generate-specific diagnostic; the shared sampling / KV / debug
    // knobs live in `CommonGenerateOptions` above.
    @Option(
        name: .long,
        help:
            "Diagnostic. Run the chunked prefill path over the prompt and write the final-position logits as raw little-endian f32 to this path, then exit."
    )
    var dumpPrefillLogits: String?

    // ─── Nemotron-Labs-Diffusion tri-mode decoding ───────────────────

    @Option(
        name: .long,
        help:
            "Decode algorithm: \"ar\" (autoregressive, default), \"diffusion\" (block-wise parallel), or \"self-spec\" (diffusion draft + AR verify). diffusion / self-spec require a Nemotron-Labs-Diffusion model."
    )
    var decodeMode: String = "ar"

    @Option(name: .long, help: "Block length for diffusion / self-spec decoding. Default 32.")
    var blockLength: Int = 32

    @Option(
        name: .long,
        help:
            "Confidence threshold for diffusion-mode denoising (0..1). Unset uses an even per-step transfer budget."
    )
    var confidenceThreshold: Float?

    func run() async throws {
        try common.applyGlobalFlags()
        let m = try await common.loadModel()

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
            try runVerbose(model: m)
            return
        }

        // Non-autoregressive modes — block-wise diffusion or linear
        // self-speculation (Nemotron-Labs-Diffusion only).
        let genMode = decodeMode.lowercased()
        if genMode != "ar" {
            try await runDiffusion(model: m, genMode: genMode)
            return
        }

        let params = common.resolvedParameters(for: m)
        print("---")
        print(prompt, terminator: "")
        let result = try await runGenerate(model: m, params: params)
        common.printFooter(result)
    }

    /// `--verbose`: run prefill once and print the top-5 next tokens.
    private func runVerbose(model m: Model) throws {
        let promptTokens = m.textTokenizer.encode(text: prompt)
        print("prompt tokens: \(promptTokens)")
        // Size the cache to the probe prompt, not the model's full
        // context window (long-context models would otherwise allocate
        // tens of GB for this sanity check).
        let verifyCacheDepth = max(1, Swift.min(m.textEngine.maxContextWindow, promptTokens.count + 1))
        let caches = m.textEngine.makeLayerCaches(maxSeq: verifyCacheDepth)
        var lastLogits: Tensor?
        for (i, t) in promptTokens.enumerated() {
            lastLogits = m.textEngine.forward(tokenId: t, position: i, caches: caches)
        }
        if let l = lastLogits {
            print("top-5 next tokens:")
            for (id, v) in Sampling.topN(l, n: 5) {
                let s = m.textTokenizer.decode(tokens: [id], skipSpecialTokens: false)
                print("  \(id) (\(String(format: "%.4f", v)))  \"\(s)\"")
            }
        }
    }

    /// `--decode-mode diffusion|self-spec` (Nemotron-Labs-Diffusion).
    private func runDiffusion(model m: Model, genMode: String) async throws {
        guard genMode == "diffusion" || genMode == "self-spec" || genMode == "selfspec" else {
            throw ValidationError("Unknown --decode-mode \"\(decodeMode)\". Use ar, diffusion, or self-spec.")
        }
        guard m.nemotronLabsDiffusion != nil else {
            throw ValidationError(
                "--decode-mode \(decodeMode) requires a Nemotron-Labs-Diffusion model "
                    + "(\(common.model) does not support it).")
        }
        let isDiffusion = genMode == "diffusion"
        let requested = common.maxTokens ?? 128
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
    }

    /// Autoregressive generation, streaming by default.
    private func runGenerate(
        model: Model, params: GenerationParameters
    ) async throws -> GenerationResult {
        let chatMessages: [ChatMessage]? =
            mode == .chat ? [ChatMessage(role: .user, content: prompt)] : nil
        if mode == .chat && !model.supportsChatTemplate {
            throw ValidationError(
                "--mode chat requires a chat template, but \(common.model) ships none "
                    + "(neither tokenizer_config.json nor a standalone chat_template.jinja). "
                    + "Use --mode raw, or load an instruct model.")
        }

        if common.streaming {
            var generated: [Int] = []
            var text = ""
            var stats: GenerationStats?
            let stream =
                try chatMessages != nil
                ? model.generateStream(messages: chatMessages!, parameters: params)
                : model.generateStream(prompt: prompt, parameters: params)
            for try await chunk in stream {
                if !chunk.text.isEmpty {
                    print(chunk.text, terminator: "")
                    fflush(stdout)
                }
                generated.append(contentsOf: chunk.tokens)
                text += chunk.text
                if let s = chunk.stats { stats = s }
            }
            print("")
            let promptTokens =
                chatMessages.flatMap { try? model.renderChatTemplate(messages: $0) }
                ?? model.textTokenizer.encode(text: prompt)
            return GenerationResult(
                promptTokens: promptTokens,
                generatedTokens: generated, text: text,
                stats: stats!)
        } else {
            let result: GenerationResult
            if let messages = chatMessages {
                result = try await model.generate(messages: messages, parameters: params)
            } else {
                result = try await model.generate(prompt: prompt, parameters: params)
            }
            print(result.text)
            return result
        }
    }
}
