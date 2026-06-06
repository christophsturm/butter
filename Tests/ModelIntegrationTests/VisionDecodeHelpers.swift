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
// Shared greedy-decode helper for the engine-internal-vision integration
// suites. Idefics3 / PaliGemma / SmolVLM2 have NO composed `VisionModel`
// (`Model.vlModel` is nil) — they splice image features inside the engine's
// own `forward(...)` / `prefillWithImage(...)`, so their image tests can't go
// through `vlm.generate(...)` and must drive the decode loop by hand. This
// centralizes the *identical* tail of those loops (argmax the prefill's
// last-position logits, then `forwardSample` until a stop token / cap) so the
// suites stop each re-rolling it.
//
// NOTE: when the planned `VisionModel` adapter for these engine-internal
// families lands (see `planning/session-plan.md` "VLModel adapter"), these
// tests can move to the same `generate`-based shape as the other VLM suites
// and this helper can retire.

import Foundation

@testable import FFAI

/// Greedy-decode after an image prefill: take the argmax of the prefill's
/// last-position `lastLogits` as the first token, then `forwardSample` one
/// token per step (`position = promptLength + step`) until a `stopTokens` id
/// is produced or `maxTokens` is reached. Returns the generated token ids
/// (excluding the stop token).
func greedyDecodeAfterPrefill(
    lastLogits: Tensor,
    textEngine: any LanguageModel,
    promptLength: Int,
    stopTokens: Set<Int>,
    maxTokens: Int,
    caches: [any LayerCacheProtocol]
) -> [Int] {
    var generated: [Int] = []
    var next = Sampling.argmax(lastLogits)
    for step in 0 ..< maxTokens {
        if stopTokens.contains(next) { break }
        generated.append(next)
        next = textEngine.forwardSample(
            tokenId: next, position: promptLength + step, caches: caches)
    }
    return generated
}
