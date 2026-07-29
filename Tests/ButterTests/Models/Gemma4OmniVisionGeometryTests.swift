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
// Pure-geometry unit tests for the Gemma 4 Omni encoder-free vision
// embedder's soft-token budget. The omni tower shares the variable-
// resolution `aspectPreservingSize` geometry (patch 16, pool 3 ⇒
// sideMult 48), but exposes a configurable budget {70,140,280,560,1120}.
// One soft token = one 48×48 patch, so the token count is
// `(H/48)·(W/48)` and must stay ≤ the budget with dims divisible by 48.

import Testing

@testable import ButterSwift

@Suite("Gemma4 Omni Vision Geometry")
struct Gemma4OmniVisionGeometryTests {

    private func size(_ w: Int, _ h: Int, budget: Int) -> (h: Int, w: Int) {
        Gemma4VLVisionModel.aspectPreservingSize(
            width: w, height: h,
            patchSize: 16, poolingKernelSize: 3, maxSoftTokens: budget)
    }

    /// Soft tokens for the omni tower = number of 48×48 patches.
    private func softTokens(_ s: (h: Int, w: Int)) -> Int {
        (s.h / 48) * (s.w / 48)
    }

    @Test("square image hits each budget without exceeding it")
    func squareBudgets() {
        // Exact targets for a 700×700 source across the published budgets.
        let cases: [(budget: Int, dim: Int, tokens: Int)] = [
            (70, 384, 64),
            (140, 528, 121),
            (280, 768, 256),
            (560, 1104, 529),
            (1120, 1584, 1089),
        ]
        for c in cases {
            let s = size(700, 700, budget: c.budget)
            #expect(
                s == (h: c.dim, w: c.dim),
                "budget \(c.budget): expected \(c.dim)² got \(s)")
            #expect(softTokens(s) == c.tokens)
            #expect(softTokens(s) <= c.budget, "budget \(c.budget) exceeded")
            #expect(s.h % 48 == 0 && s.w % 48 == 0)
        }
    }

    @Test("a bigger budget never yields fewer soft tokens")
    func monotoneInBudget() {
        let budgets = [70, 140, 280, 560, 1120]
        var prev = 0
        for b in budgets {
            let t = softTokens(size(900, 600, budget: b))
            #expect(t >= prev, "soft tokens dropped from \(prev) to \(t) at budget \(b)")
            #expect(t <= b)
            prev = t
        }
    }

    @Test("non-square + degenerate strips stay divisible by 48 and within budget")
    func nonSquareValid() {
        for (w, h) in [(1600, 400), (400, 1600), (2000, 30), (100, 50)] {
            let s = size(w, h, budget: 280)
            #expect(s.h % 48 == 0 && s.w % 48 == 0, "\(w)×\(h) → \(s) not /48")
            #expect(softTokens(s) <= 280, "\(w)×\(h) → \(softTokens(s)) > 280")
            #expect(s.h > 0 && s.w > 0)
        }
    }
}
