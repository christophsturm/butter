# Telemetry: Quality-Metrics Consolidation (PPL / KLD / NIAH)

> Status: **proposal / for discussion** (2026-06-03). Design options for moving
> model-quality metrics behind a telemetry interface that models implement, so
> they're uniform and inspectable. No code landed yet — this is the strawman to
> settle before implementation.
>
> Author: TheTom · Reviewer: @ekryski

## 1. Why

Quality metrics (perplexity, KL-divergence vs a reference, and the planned
needle-in-a-haystack retrieval eval) are the numbers that tell us whether a
quant format or an architecture port is actually correct — not just fast. Right
now they're reachable only through the bench runner, computed by a standalone
helper that reaches into the model. We want them:

1. **Uniform** — every model family scored the same way, through one interface.
2. **Inspectable** — surfaced via the existing inspect surface (`InspectTap`),
   not bolted onto `BenchRunner` alone.
3. **Consolidated** — interface *and* math (PPL, KLD, NIAH) live together under
   `Telemetry/`, instead of split across `Stats/` (PPL/KLD) and `Benchmark/`
   (NIAH stub).

## 2. Current organization (what exists today)

| Area | File(s) | Role |
|------|---------|------|
| **Stats/** | `Perplexity.swift` (holds both `compute(...)` **and** `klDivergence(...)`), `GenerationStats`, `MemoryStats`, `ThinkingSplit` | the metric math |
| **Telemetry/** | `Profiling.swift` (phase timings), `InspectTap.swift` (layer-boundary tensor dumps), `Debug.swift`, `TokenizerInspection.swift` | instrumentation + inspect surface |
| **Benchmark/** | `BenchRunner.swift`, `BenchMethod.swift`, `BenchmarkWriter.swift` | drives benches; `NIAH` is a plumbed-but-unimplemented `BenchMethod` case |
| **Model** | `LanguageModel: Module` (`LanguageModel.swift`); concrete `Model` exposes `.engine` (`makeLayerCaches`, `forward(tokenId:position:caches:device:)`) | the thing being scored |

Key observation: `Perplexity.compute` / `klDivergence` already take a concrete
`Model` and score it **only** through `model.engine.forward(...)` +
`model.engine.makeLayerCaches(...)`. They are already engine-uniform — the
metrics don't special-case any family. So "models implement the metrics" is
really about **formalizing that uniformity into a protocol** the inspect surface
can call, not about teaching each model new math.

```
Today:   BenchRunner ──► Stats.Perplexity.compute(model) ──► model.engine.forward(...)
                          Stats.Perplexity.klDivergence(ref, cand)
         InspectTap ──► (no access to quality metrics)
```

## 3. Goals / non-goals

**Goals**
- One telemetry interface for quality scoring; models conform; `InspectTap` (or
  an `inspect`/`telemetry` subcommand) and `BenchRunner` both consume it.
- **Consolidate all quality-metric code — interface *and* math (PPL, KLD,
  future NIAH) — under `Telemetry/`.** Today PPL/KLD live in `Stats/` and NIAH
  is stubbed in `Benchmark/`; pull every quality metric into one home. The
  non-quality runtime stats (`GenerationStats`, `MemoryStats`, `ThinkingSplit`)
  stay in `Stats/`.
- Pin quality benches to **release** build config (forced-decode over a corpus
  is far too slow in debug).

**Non-goals**
- Changing how PPL / KLD are *computed*. The existing formulas — perplexity =
  `exp(mean negative-log-likelihood)`, KLD = mean per-position `KL(ref‖cand)` —
  carry over unchanged. This is a relocation + interface refactor, not a
  re-derivation of the metrics.
- Implementing NIAH here (only fixing its home as `Telemetry/` and that it rides
  the same interface).

## 4. The layering decision

- **`Telemetry/`** = everything quality: the interface (protocol) + the math
  (`Perplexity.swift`, KLD, NIAH-later) + the inspect glue.
- **`Stats/`** = non-quality runtime stats only — `GenerationStats`,
  `MemoryStats`, `ThinkingSplit`. The quality metrics move *out* of here.
- Model conforms to the telemetry protocol → hands its distributions →
  `Telemetry/` crunches → result is inspectable.

Concretely: `Stats/Perplexity.swift` → `Telemetry/Perplexity.swift` (KLD split
to `Telemetry/KLDivergence.swift`, or kept co-located — see §8), and NIAH lands
in `Telemetry/` when built. The whole quality surface ends up under one folder.

## 5. Design — one flag-gated tap

There is **one** telemetry tap: `InspectTap`. It is already the right shape —
env-configured (`InspectTap.fromEnvironment`), value-type, predicate-gated
(`shouldDump(layer:)`). We extend that one tap to be the single capture point for
*everything* telemetry — the existing layer-boundary tensor dumps **and** the
quality metrics (PPL, KLD, NIAH). No second surface, no per-metric subcommand.

Two principles:

1. **One tap, all metrics.** Every metric we want to capture is a capability of
   `InspectTap`, selected by a flag. The tap is threaded through the existing
   generate / forced-decode path; whoever holds it decides what gets recorded.
2. **Flag-gated, default off, zero cost when off.** Each metric is an independent
   on/off flag. A disabled metric does **no work** — no retained logits, no extra
   forward, no softmax/KL math — so telemetry never taxes the hot path unless you
   asked for it. Enabling is opt-in via env / config, mirroring how `InspectTap`
   already reads its layer-dump config from the environment.

```swift
// Telemetry/InspectTap.swift  (extended)
public struct QualityMetrics: OptionSet, Sendable {
    public let rawValue: Int
    public static let perplexity = QualityMetrics(rawValue: 1 << 0)
    public static let kld        = QualityMetrics(rawValue: 1 << 1)
    public static let niah       = QualityMetrics(rawValue: 1 << 2)
}

public struct InspectTap: Sendable {
    // ... existing layer-boundary dump config ...
    public var metrics: QualityMetrics            // default []  → nothing captured
    public var isCapturingMetrics: Bool { !metrics.isEmpty }

    // fromEnvironment also parses e.g. FFAI_TELEMETRY=ppl,kld → .perplexity ∪ .kld
}
```

The hot path only does metric work behind the flag — off ⇒ the branch is never
entered:

```swift
// in the generate / forced-decode loop
if tap.metrics.contains(.perplexity) { /* accumulate NLL from this position's logits */ }
if tap.metrics.contains(.kld)        { /* accumulate KL(ref‖cand) */ }
// off → none of the above runs; no logits retained, no softmax
```

Models feed the tap through one small contract so every family is scored
identically — the model only supplies its distributions; the tap + `Telemetry/`
math own everything else:

```swift
// Telemetry/QualityScorable.swift
/// Force-decode `tokens`; return next-token logits at each scored position
/// (length = tokens.count - 1). PPL (own NLL), KLD (vs a reference's rows), and
/// NIAH all derive from these — handing back logits keeps the math in one place.
public protocol QualityScorable {
    func forcedDecodeLogits(tokens: [Int]) throws -> [Tensor]
}

// Default conformance for the existing concrete Model — every current family
// works for free (it already has engine.forward); a family with a cheaper
// batched forced-decode can override.
extension Model: QualityScorable {
    public func forcedDecodeLogits(tokens: [Int]) throws -> [Tensor] {
        let caches = engine.makeLayerCaches(maxSeq: nil, device: .shared)
        return (0 ..< tokens.count - 1).map { t in
            engine.forward(tokenId: tokens[t], position: t, caches: caches, device: .shared)
        }
    }
}
```

Layout after:

```
Telemetry/InspectTap.swift        // THE tap — layer dumps + flag-gated quality metrics
Telemetry/QualityScorable.swift   // the one model-facing contract (forcedDecodeLogits)
Telemetry/Perplexity.swift        // PPL math — moved from Stats/
Telemetry/KLDivergence.swift      // KLD math — split from Perplexity.swift (or co-located, §8)
Telemetry/NIAH.swift              // later — needle harness over forcedDecodeLogits
Benchmark/BenchRunner.swift       // sets the tap's metric flags, reads results
Stats/                            // unchanged: GenerationStats, MemoryStats, ThinkingSplit
```

## 6. Methodology — generating *proper* PPL / KLD

The per-position math already in `Stats/Perplexity.swift` is correct:
full-vocab, max-subtraction logsumexp accumulated in **float64**
(`negLogSoftmaxAt`), and KLD as full-vocab `Σ p_ref·(log p_ref − log p_cand)`
in log-space with the right direction (`klAtPosition`). The work to make these
*trustworthy quant-quality numbers* is in the harness, not the formulas:

1. **Context windowing + stride.** `compute` today runs one growing KV cache
   from position 0 over the whole token list and scores every token — so tokens
   past `n_ctx` are RoPE-extrapolated (garbage) and the first tokens have ~no
   left-context (noise). Use a fixed `n_ctx` window, **strided with overlap**
   (stride = `n_ctx/2`), scoring only the second half so every scored token has
   ≥`n_ctx/2` of context; count each token once.
2. **BOS / first-token handling.** Prepend BOS consistently; don't score BOS
   itself or the first corpus token (no context).
3. **KLD: cache the reference distribution.** Don't co-run reference + candidate
   live (infeasible to hold two 80 GB-class models; wasteful to re-run the
   reference per quant). Two-phase: run the full-precision reference **once**,
   dump per-position log-probs to disk (store **f16** to halve size), then score
   each candidate against the cached file. This is the shape the `--ref-model`
   flag should grow into (`--save-ref-logits` → `--kld --ref-logits <file>`).
4. **Report the distribution, not just the mean.** Mean KLD hides tail
   distortion — exactly where bad quants live. Emit **mean / median / p90 / p99 /
   max** KLD plus **top-1 agreement %** (`argmax(ref) == argmax(cand)`); top-1
   flip-rate is the most legible "is this quant broken" signal.
5. **Fixed corpus + tokenizer.** Raw **WikiText-2 test** (the plumbed `wikitext2`
   method); tokenize once; feed identical tokens + identical windowing to
   reference and candidate so KLD positions align.
6. **lm_head precision.** Compute the final logits projection in **f32** where
   feasible so PPL/KLD measure *quantization* distortion, not f16-logit rounding.
7. **Determinism + release.** Forced/greedy decode (already), deterministic
   packed weights, release build (see §7).

## 7. Release-only benches (separate but related gap)

Quality benches forced-decode an entire corpus; in **debug** this is painfully
slow. iron already runs its bench tooling in release (`cargo run --release`).
FFAI's bench path is not pinned today. Proposed:

- `ffai bench` (and any bench CI step) builds/runs with `swift run -c release` /
  `swift build -c release`.
- Optionally a guard that warns (or refuses) if a quality bench is invoked from a
  debug build, so we never publish debug-speed numbers by accident.

## 8. Usage — how to run

Quality metrics run through the `ffai bench` CLI. PPL and KLD work **today**;
the `FFAI_TELEMETRY` live-capture flag and NIAH are the follow-ups this design
adds the surface for.

**Perplexity** (WikiText-2 forced-decode) — works today:

```bash
# debug (slow — quick smoke only)
ffai bench --method wikitext2 --model <hf-repo-or-local-path>

# release — use this for real numbers (see §7)
swift run -c release ffai bench --method wikitext2 \
    --model <hf-repo-or-local-path> \
    --wikitext2-max-tokens 4096            # corpus token cap (default 2048)
```

**KL-divergence** vs a full-precision reference — works today. Pass the bf16
(unquantized) variant of the same architecture as `--ref-model`:

```bash
swift run -c release ffai bench --method wikitext2 \
    --model     <quantized-repo-or-path> \
    --ref-model <bf16-reference-repo-or-path>
# emits Gen PPL + Gen KLD columns into ./benchmarks (override with --report-dir)
```

**Telemetry flag surface** (the new on/off gates; default off, zero cost when
unset) — the env var `InspectTap` parses, consumed by the live-capture
follow-up:

```bash
FFAI_TELEMETRY=ppl,kld   ffai <generate|serve> ...   # capture PPL + KLD live
FFAI_TELEMETRY=ppl       ffai <generate|serve> ...   # perplexity only
# unset → nothing captured, no overhead (mirrors FFAI_INSPECT / FFAI_INSPECT_LAYERS)
```

**NIAH** (needle-in-a-haystack retrieval) — *target*; lands as a follow-up
(`Telemetry/NIAH.swift`) over the same contract:

```bash
swift run -c release ffai bench --method niah --model <long-ctx-repo>
```

## 9. Settled vs open

**Settled (this discussion):**
- **One tap.** `InspectTap` is the single capture point; no separate quality
  subcommand. ✔
- **Flag-gated, default off, zero cost when disabled.** ✔
- **One model-facing contract** (`QualityScorable.forcedDecodeLogits`) — not a
  protocol-per-metric. ✔
- **Consolidate quality math under `Telemetry/`**; `Stats/` keeps non-quality
  stats. ✔

**Open for the call:**
1. **Handoff type:** logits (math centralized — recommended) vs log-probs (model
   does its own softmax).
2. **Flag config surface:** env var name + grammar (e.g. `FFAI_TELEMETRY=ppl,kld`),
   and whether `BenchRunner` sets the flags programmatically vs via env.
3. **KLD file split** — own `Telemetry/KLDivergence.swift` or stay co-located in
   `Telemetry/Perplexity.swift` (shared `negLogSoftmaxAt` / `decodeLogSoftmax`).
4. **Reference model for KLD** — require two `QualityScorable`s (clean; matches
   the existing `--ref-model` flag)?
5. **Release-bench enforcement** — warn vs hard-refuse on debug builds.

## 10. Rough migration

1. Add `Telemetry/QualityScorable.swift` (protocol) + `extension Model: QualityScorable`.
2. Move `Stats/Perplexity.swift` → `Telemetry/Perplexity.swift`, re-type
   `compute`/`klDivergence` from `Model` to `some QualityScorable` (no math
   change). Optionally split KLD into `Telemetry/KLDivergence.swift`.
3. Extend `InspectTap` with the `QualityMetrics` flags (+ `fromEnvironment`
   parsing); gate the per-position metric work in the generate/forced-decode loop
   behind those flags.
4. Point `BenchRunner` at the tap — set flags, read results.
5. Pin the bench path to release.
6. (Later) `Telemetry/NIAH.swift` over the same contract + flag.
