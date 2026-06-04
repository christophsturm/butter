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
// DecodeTimeline — pure reductions over the per-token decode wall-clock
// series the generate loop already records. Two axes the external
// Apple-Silicon benchmarks report that a single mean tok/s number can't:
//
//   • Inter-token latency percentiles (p50/p95/p99). Mean throughput
//     hides the tail; a smooth 60 tok/s average with an occasional
//     200 ms stall reads fine on paper but janks the UI. p95/p99 are
//     where that shows up.
//
//   • Sustained-decode retention. On a long run thermal throttling and
//     cache growth slow decode over time. We compare the late window's
//     tok/s to the early window's: 1.0 = rock-steady, <1.0 = drift.
//
// Both operate on the `[Double]` of per-token decode durations (seconds)
// that `driveGeneration` collects, so they add no hot-path cost — just
// arithmetic over an array that already exists.

import Foundation

enum DecodeTimeline {
    /// Tokens dropped from the front before measuring *steady-state*
    /// retention. Matches the warm-up window `steadyTokensPerSecond`
    /// already discards (PSO compile, autorelease warm-up, pipeline
    /// cache misses) so drift reflects thermals, not cold start.
    static let warmupTokens = 10

    /// Minimum generated tokens before retention is meaningful: we need
    /// the warm-up dropped plus two non-trivial windows (≥ 8 each) to
    /// compare. Below this, retention is `nil`. The signal sharpens on
    /// the 512+-token runs the benchmarks use for sustained decode.
    static let minTokensForRetention = warmupTokens + 16

    /// Linear-interpolated `p`-th percentile (`p` in 0...1) of the
    /// per-token durations, returned in **milliseconds**. `nil` for an
    /// empty series. Uses the same interpolation convention as NumPy's
    /// default so cross-tool comparisons line up.
    static func percentileMs(_ perTokenSeconds: [Double], _ p: Double) -> Double? {
        guard !perTokenSeconds.isEmpty else { return nil }
        let sorted = perTokenSeconds.sorted()
        if sorted.count == 1 { return sorted[0] * 1000 }
        let clamped = min(max(p, 0), 1)
        let rank = clamped * Double(sorted.count - 1)
        let lower = Int(rank.rounded(.down))
        let upper = Int(rank.rounded(.up))
        let frac = rank - Double(lower)
        let value = sorted[lower] + (sorted[upper] - sorted[lower]) * frac
        return value * 1000
    }

    /// Decode-throughput retention: late-window tok/s ÷ early-window
    /// tok/s, after dropping the warm-up window and splitting the rest
    /// into halves. 1.0 ≈ steady; 0.8 ≈ 20 % slower by the end (thermal
    /// throttle or growing KV cost). `nil` when too few tokens to judge.
    static func sustainedRetention(_ perTokenSeconds: [Double]) -> Double? {
        guard perTokenSeconds.count >= minTokensForRetention else { return nil }
        let body = Array(perTokenSeconds.dropFirst(warmupTokens))
        let half = body.count / 2
        let early = body.prefix(half)
        let late = body.suffix(half)
        let earlyMean = early.reduce(0, +) / Double(early.count)  // s / token
        let lateMean = late.reduce(0, +) / Double(late.count)
        guard earlyMean > 0, lateMean > 0 else { return nil }
        // tok/s ratio late÷early == s/token ratio early÷late.
        return earlyMean / lateMean
    }
}
