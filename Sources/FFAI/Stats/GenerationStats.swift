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
// GenerationStats — every numeric field one `generate(...)` call
// produces. The CLI's --stats prints this; the bench harness writes
// the same fields into the markdown / JSON sidecar so analysis tooling
// stays cross-compatible with mlx-swift-lm reports.
//
// Field naming and grouping mirrors mlx-swift-lm's `ResultRow` so
// porting analysis scripts is mechanical. Fields not yet meaningful
// (batch decode, speculative decode) are scaffolded as commented
// blocks below — uncomment when the underlying capability lands.

import Foundation

public struct GenerationStats: Sendable {
    // ─── Counts ──────────────────────────────────────────────────────

    /// Number of prompt tokens fed to prefill.
    public let promptTokens: Int
    /// Number of tokens emitted by the decode loop.
    public let generatedTokens: Int
    /// Maximum context window the model supports.
    public let contextSize: Int

    // ─── Latency / throughput ────────────────────────────────────────

    public let prefillTimeS: Double
    public let decodeTimeS: Double
    public let timeToFirstTokenMs: Double

    public var prefillTokensPerSecond: Double {
        prefillTimeS > 0 ? Double(promptTokens) / prefillTimeS : 0
    }
    public var decodeTokensPerSecond: Double {
        decodeTimeS > 0 ? Double(generatedTokens) / decodeTimeS : 0
    }

    /// Per-token decode rate computed from token 11 onward — drops the
    /// first ~10 tokens so PSO compilation, autorelease pool warm-up,
    /// and Metal pipeline-cache misses don't depress the steady-state
    /// number. `nil` when fewer than 11 generated tokens.
    public let steadyTokensPerSecond: Double?

    // ─── Memory (Apple Silicon GPU + wired ticket) ───────────────────

    /// `MTLDevice.currentAllocatedSize` immediately before prefill.
    public let baselineGPUBytes: Int
    /// `MTLDevice.currentAllocatedSize` after the last prefill token.
    public let postPrefillGPUBytes: Int
    /// `MTLDevice.currentAllocatedSize` after the last decode token.
    public let postDecodeGPUBytes: Int
    /// Max GPU bytes seen during the prefill phase.
    public let prefillPeakGPUBytes: Int
    /// Max GPU bytes seen during the decode phase.
    public let decodePeakGPUBytes: Int
    /// `MTLDevice.recommendedMaxWorkingSetSize` — the wired-memory
    /// ticket the OS will try to hold resident.
    public let wiredTicketBytes: Int
    /// Resident weight-tensor bytes (sum of every model parameter
    /// buffer). Computed once at construction.
    public let weightsBytes: Int
    /// Sum of `bytesAllocated` across every per-layer KV cache —
    /// the *capacity*, allocated up-front at the model's max context.
    public let kvCacheAllocatedBytes: Int
    /// Sum of `bytesInUse` across every per-layer KV cache — the
    /// *live* slice (length out of maxSeq).
    public let kvCacheUsedBytes: Int

    public var peakGPUBytes: Int { max(prefillPeakGPUBytes, decodePeakGPUBytes) }
    public var prefillGrowthBytes: Int { postPrefillGPUBytes - baselineGPUBytes }
    public var decodeGrowthBytes: Int { postDecodeGPUBytes - postPrefillGPUBytes }

    // ─── Quality (opt-in; bench harness fills these via Perplexity) ──

    /// Perplexity over the *thinking* segment (tokens between
    /// `<think>` … `</think>`). `nil` when the model didn't emit a
    /// thinking segment, or when the caller didn't request perplexity.
    public let thinkPerplexity: Double?
    /// Perplexity over the post-thinking generation segment.
    public let genPerplexity: Double?
    /// KL divergence vs a reference distribution (e.g. bf16 vs the
    /// quantized variant). Requires a paired reference run.
    public let thinkKLDivergence: Double?
    public let genKLDivergence: Double?

    /// Token counts inside the think / gen segments (sum to
    /// `generatedTokens` when both are non-nil).
    public let thinkTokenCount: Int?
    public let genTokenCount: Int?

    // ─── Inter-token latency distribution (decode, ms) ───────────────
    //
    // Percentiles of the gap between consecutive decode tokens. Mean
    // tok/s hides the tail; p95 / p99 surface the worst-case stalls a
    // user actually feels. `nil` until the decode loop produces tokens.

    public let interTokenLatencyP50Ms: Double?
    public let interTokenLatencyP95Ms: Double?
    public let interTokenLatencyP99Ms: Double?

    /// Decode-throughput retention: late-window tok/s ÷ early-window
    /// tok/s over the run. 1.0 ≈ steady, <1.0 ≈ thermal / KV drift.
    /// `nil` for runs too short to judge (see `DecodeTimeline`).
    public let sustainedDecodeRatio: Double?

    // ─── Energy + thermal ────────────────────────────────────────────
    //
    // Energy is the kernel's per-task counter (CPU+GPU, no sudo); see
    // `EnergyStats`. `nil` when the Mach read is unavailable.

    /// Joules attributed to this process during prefill.
    public let prefillEnergyJoules: Double?
    /// Joules attributed to this process during decode.
    public let decodeEnergyJoules: Double?

    /// Process thermal state at the start of generation, the worst
    /// (peak) state seen during it, and the state at the end. `.unknown`
    /// when not captured (hand-built stats / empty prompt).
    public let thermalStateAtStart: ThermalState
    public let thermalStatePeak: ThermalState
    public let thermalStateAtEnd: ThermalState

    /// Energy per generated token over the decode phase — the headline
    /// efficiency number. `nil` when energy was unavailable or no tokens
    /// were generated.
    public var joulesPerToken: Double? {
        guard let e = decodeEnergyJoules, generatedTokens > 0 else { return nil }
        return e / Double(generatedTokens)
    }

    // ─── Scaffold for future modes ───────────────────────────────────
    //
    // Uncomment + populate when the corresponding capability ships.
    // Kept here so the formatted() printer + bench writer schema only
    // grow additively when those land.
    //
    // Batch decoding:
    //   public let batchSize: Int
    //   public let perSequenceDecodeTokensPerSecond: Double?
    //
    // Speculative decoding:
    //   public let acceptanceRate: Double?
    //   public let draftTokensPerSecond: Double?
    //   public let draftAcceptedTokens: Int?

    // ─── Init ─────────────────────────────────────────────────────────
    //
    // Explicit so the new telemetry params can default — every existing
    // call site predates them and passes only through `genTokenCount:`.
    // Keep the new params trailing + defaulted to preserve that.

    public init(
        promptTokens: Int,
        generatedTokens: Int,
        contextSize: Int,
        prefillTimeS: Double,
        decodeTimeS: Double,
        timeToFirstTokenMs: Double,
        steadyTokensPerSecond: Double?,
        baselineGPUBytes: Int,
        postPrefillGPUBytes: Int,
        postDecodeGPUBytes: Int,
        prefillPeakGPUBytes: Int,
        decodePeakGPUBytes: Int,
        wiredTicketBytes: Int,
        weightsBytes: Int,
        kvCacheAllocatedBytes: Int,
        kvCacheUsedBytes: Int,
        thinkPerplexity: Double?,
        genPerplexity: Double?,
        thinkKLDivergence: Double?,
        genKLDivergence: Double?,
        thinkTokenCount: Int?,
        genTokenCount: Int?,
        interTokenLatencyP50Ms: Double? = nil,
        interTokenLatencyP95Ms: Double? = nil,
        interTokenLatencyP99Ms: Double? = nil,
        sustainedDecodeRatio: Double? = nil,
        prefillEnergyJoules: Double? = nil,
        decodeEnergyJoules: Double? = nil,
        thermalStateAtStart: ThermalState = .unknown,
        thermalStatePeak: ThermalState = .unknown,
        thermalStateAtEnd: ThermalState = .unknown
    ) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.contextSize = contextSize
        self.prefillTimeS = prefillTimeS
        self.decodeTimeS = decodeTimeS
        self.timeToFirstTokenMs = timeToFirstTokenMs
        self.steadyTokensPerSecond = steadyTokensPerSecond
        self.baselineGPUBytes = baselineGPUBytes
        self.postPrefillGPUBytes = postPrefillGPUBytes
        self.postDecodeGPUBytes = postDecodeGPUBytes
        self.prefillPeakGPUBytes = prefillPeakGPUBytes
        self.decodePeakGPUBytes = decodePeakGPUBytes
        self.wiredTicketBytes = wiredTicketBytes
        self.weightsBytes = weightsBytes
        self.kvCacheAllocatedBytes = kvCacheAllocatedBytes
        self.kvCacheUsedBytes = kvCacheUsedBytes
        self.thinkPerplexity = thinkPerplexity
        self.genPerplexity = genPerplexity
        self.thinkKLDivergence = thinkKLDivergence
        self.genKLDivergence = genKLDivergence
        self.thinkTokenCount = thinkTokenCount
        self.genTokenCount = genTokenCount
        self.interTokenLatencyP50Ms = interTokenLatencyP50Ms
        self.interTokenLatencyP95Ms = interTokenLatencyP95Ms
        self.interTokenLatencyP99Ms = interTokenLatencyP99Ms
        self.sustainedDecodeRatio = sustainedDecodeRatio
        self.prefillEnergyJoules = prefillEnergyJoules
        self.decodeEnergyJoules = decodeEnergyJoules
        self.thermalStateAtStart = thermalStateAtStart
        self.thermalStatePeak = thermalStatePeak
        self.thermalStateAtEnd = thermalStateAtEnd
    }

    // ─── Pretty printer for `--stats` ────────────────────────────────

    public func formatted() -> String {
        var out = "[STATS]\n"
        out += "  prompt:           \(promptTokens) tokens\n"
        out += "  generated:        \(generatedTokens) tokens\n"
        out += "  context:          \(contextSize) tokens\n"
        out += String(format: "  ttft:             %.2f ms\n", timeToFirstTokenMs)
        out += String(
            format: "  prefill:          %.2fs (%.2f tok/s)\n",
            prefillTimeS, prefillTokensPerSecond)
        out += String(
            format: "  decode:           %.2fs (%.2f tok/s)\n",
            decodeTimeS, decodeTokensPerSecond)
        if let s = steadyTokensPerSecond {
            out += String(format: "  decode (steady):  %.2f tok/s   (tokens 11+)\n", s)
        }
        out += String(format: "  baseline GPU:     %@\n", Self.fmt(baselineGPUBytes))
        out += String(
            format: "  post-prefill GPU: %@   (+ %@)\n",
            Self.fmt(postPrefillGPUBytes), Self.fmt(prefillGrowthBytes))
        out += String(
            format: "  post-decode  GPU: %@   (+ %@)\n",
            Self.fmt(postDecodeGPUBytes), Self.fmt(decodeGrowthBytes))
        out += String(format: "  prefill peak:     %@\n", Self.fmt(prefillPeakGPUBytes))
        out += String(format: "  decode  peak:     %@\n", Self.fmt(decodePeakGPUBytes))
        out += String(format: "  weights:          %@\n", Self.fmt(weightsBytes))
        out += String(format: "  KV cache (alloc): %@\n", Self.fmt(kvCacheAllocatedBytes))
        out += String(format: "  KV cache (used):  %@\n", Self.fmt(kvCacheUsedBytes))
        out += String(format: "  wired ticket:     %@\n", Self.fmt(wiredTicketBytes))
        if let pp = genPerplexity {
            out += String(format: "  gen perplexity:   %.3f\n", pp)
        }
        if let pp = thinkPerplexity {
            out += String(format: "  think perplexity: %.3f\n", pp)
        }
        if let kl = genKLDivergence {
            out += String(format: "  gen KLD:          %.4f\n", kl)
        }
        if let kl = thinkKLDivergence {
            out += String(format: "  think KLD:        %.4f\n", kl)
        }
        if let t = thinkTokenCount, let g = genTokenCount {
            out += "  think / gen split: \(t) / \(g) tokens\n"
        }
        if let p50 = interTokenLatencyP50Ms {
            out += String(
                format: "  inter-token (ms): p50 %.2f  p95 %.2f  p99 %.2f\n",
                p50, interTokenLatencyP95Ms ?? p50, interTokenLatencyP99Ms ?? p50)
        }
        if let r = sustainedDecodeRatio {
            out += String(
                format: "  decode retention: %.1f%%   (late tok/s ÷ early)\n", r * 100)
        }
        if let jpt = joulesPerToken {
            out += String(
                format: "  energy:           %.4f J/token   (decode %.2f J)\n",
                jpt, decodeEnergyJoules ?? 0)
        }
        if thermalStateAtStart != .unknown || thermalStatePeak != .unknown {
            out +=
                "  thermal:          \(thermalStateAtStart.rawValue)"
                + " → peak \(thermalStatePeak.rawValue)"
                + " → \(thermalStateAtEnd.rawValue)\n"
        }
        return out
    }

    private static func fmt(_ bytes: Int) -> String {
        let mb = Double(bytes) / 1024.0 / 1024.0
        if mb >= 1024 { return String(format: "%.2f GB", mb / 1024) }
        return String(format: "%.1f MB", mb)
    }
}
