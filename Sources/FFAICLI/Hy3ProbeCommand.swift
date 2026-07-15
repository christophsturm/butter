// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// One-load Hy3 throughput probe: full dual + FREE + pure-MoE without
// reloading the ~92G checkpoint (avoids multi× thrash of cold loads).

import ArgumentParser
import FFAI
import Foundation

/// Load Hy3 once; run several generate passes with different probe envs.
struct Hy3ProbeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "hy3probe",
        abstract:
            "One-load Hy3 dual + FREE/pure-MoE prefill probes (avoids multi× 92G reloads)."
    )

    @Option(name: .shortAndLong, help: "Local Hy3 model path.")
    var model: String = "/Users/tom/models/Hy3-oQ2"

    @Option(name: .shortAndLong, help: "Prompt text (aim for ≥128 tokens).")
    var prompt: String =
        String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 25)
        + String(repeating: "x ", count: 60)

    @Option(name: .long, help: "Max new tokens for full dual runs.")
    var maxTokens: Int = 32

    @Option(name: .long, help: "Max new tokens for FREE / pure-MoE probes.")
    var probeTokens: Int = 8

    func run() async throws {
        // Clear probe envs so load/defaults are clean.
        for k in [
            "FFAI_HY3_SKIP_MOE", "FFAI_HY3_SKIP_SHARED", "FFAI_HY3_SKIP_ATTN",
            "FFAI_HY3_SKIP_SDPA", "FFAI_HY3_MOE_BM32", "FFAI_HY3_SCRATCH",
            "FFAI_HY3_PROFILE", "FFAI_HY3_ACTIVE_EG",
        ] {
            unsetenv(k)
        }
        setenv("FFAI_MAX_COMMAND_BUFFERS", "256", 1)

        print("ffai \(FFAI.version) hy3probe — loading \(model)…")
        let loadStart = Date()
        let m = try await Model.load(model, options: LoadOptions())
        print(
            String(
                format: "loaded in %.2fs", Date().timeIntervalSince(loadStart)))

        let params = GenerationParameters(
            maxTokens: maxTokens, stopOnEOS: false, temperature: 0)

        // ── Full dual ────────────────────────────────────────────────
        print("\n=== FULL dual-1 (gen=\(maxTokens)) ===")
        let r1 = try await runOnce(m, prompt: prompt, params: params, label: "full-1")
        print("\n=== FULL dual-2 (gen=\(maxTokens)) ===")
        let r2 = try await runOnce(m, prompt: prompt, params: params, label: "full-2")

        // ── FREE: skip MoE + shared (attn-only ceiling) ─────────────
        setenv("FFAI_HY3_SKIP_MOE", "1", 1)
        setenv("FFAI_HY3_SKIP_SHARED", "1", 1)
        let freeParams = GenerationParameters(
            maxTokens: probeTokens, stopOnEOS: false, temperature: 0)
        print("\n=== FREE SKIP_MOE+SKIP_SHARED (gen=\(probeTokens)) ===")
        let rFree = try await runOnce(
            m, prompt: prompt, params: freeParams, label: "free")

        // ── Pure MoE: skip attn + shared ────────────────────────────
        unsetenv("FFAI_HY3_SKIP_MOE")
        setenv("FFAI_HY3_SKIP_ATTN", "1", 1)
        setenv("FFAI_HY3_SKIP_SHARED", "1", 1)
        print("\n=== PURE_MOE SKIP_ATTN+SKIP_SHARED (gen=\(probeTokens)) ===")
        let rMoE = try await runOnce(
            m, prompt: prompt, params: freeParams, label: "pure-moe")

        unsetenv("FFAI_HY3_SKIP_ATTN")
        unsetenv("FFAI_HY3_SKIP_SHARED")
        unsetenv("FFAI_HY3_SKIP_MOE")

        // ── Summary ─────────────────────────────────────────────────
        print("\n=== SUMMARY (one load) ===")
        printRates("full-1", r1)
        printRates("full-2", r2)
        printRates("free  ", rFree)
        printRates("moe   ", rMoE)
        let floorsOK =
            r1.decodeTPS >= 25 && r2.decodeTPS >= 25
            && r1.prefillTPS >= 1000 && r2.prefillTPS >= 1000
        print(
            floorsOK
                ? "FLOORS: prefill≥1k decode≥25 — PASS"
                : "FLOORS: prefill≥1k decode≥25 — NOT MET "
                    + String(
                        format: "(prefill %.1f/%.1f decode %.1f/%.1f)",
                        r1.prefillTPS, r2.prefillTPS, r1.decodeTPS, r2.decodeTPS)
        )
    }

    private struct Rates {
        var prefillTPS: Double
        var decodeTPS: Double
        var promptN: Int
        var genN: Int
    }

    private func runOnce(
        _ m: Model, prompt: String, params: GenerationParameters, label: String
    ) async throws -> Rates {
        let result = try await m.generate(
            prompt: prompt, parameters: params)
        let s = result.stats
        let pre = s.prefillTokensPerSecond
        let dec = s.decodeTokensPerSecond
        let pn = result.promptTokens.count
        let gn = result.generatedTokens.count
        print(
            String(
                format:
                    "[%@] prompt=%d gen=%d prefill=%.2f tok/s decode=%.2f tok/s",
                label, pn, gn, pre, dec))
        // Keep text quiet — stats only.
        return Rates(
            prefillTPS: pre, decodeTPS: dec, promptN: pn, genN: gn)
    }

    private func printRates(_ name: String, _ r: Rates) {
        print(
            String(
                format: "  %@: prefill %7.2f  decode %6.2f  (T=%d gen=%d)",
                name, r.prefillTPS, r.decodeTPS, r.promptN, r.genN))
    }
}
