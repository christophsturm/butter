// Copyright 2026 Tom Turney (@TheTom)
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
// KV-cache sizing gap fix: `MuseGlimmerDense.loadModel` used to hardcode
// `config.int("max_position_embeddings") ?? 131072` for `maxSeq`, never
// consulting `LoadOptions.maxContextLength` — unlike `LagunaText`/
// `Hy3Text`/`NemotronDiffusionText`. That `maxSeq` sizes the UNBOUNDED
// KV cache on the 13 full-attention NoPE layers (sliding layers stay
// bounded by `slidingWindow` regardless — see
// `MuseGlimmerModel.makeLayerCaches`), so every load paid for the full
// 131072-token window with no public-API way to shrink it. These tests
// cover the extracted `MuseGlimmerDense.resolveMaxSeq` helper directly
// (no weights bundle / model load needed) — synthetic config decode,
// same style as `LagunaRecognitionTests`.

import Foundation
import Testing

@testable import ButterSwift

@Suite("MuseGlimmer maxSeq resolution")
struct MuseGlimmerMaxSeqTests {

    @Test("Default (no LoadOptions override) preserves the checkpoint's advertised window")
    func defaultUsesCheckpointWindow() {
        let cfg = ModelConfig(
            architecture: "MuseGlimmerForConditionalGeneration",
            modelType: "museglimmer",
            raw: ["max_position_embeddings": 131_072])
        let maxSeq = MuseGlimmerDense.resolveMaxSeq(config: cfg, options: LoadOptions())
        #expect(maxSeq == 131_072)
    }

    @Test("Missing max_position_embeddings falls back to 131072")
    func fallsBackTo131072WhenConfigOmitsIt() {
        let cfg = ModelConfig(
            architecture: "MuseGlimmerForConditionalGeneration",
            modelType: "museglimmer", raw: [:])
        let maxSeq = MuseGlimmerDense.resolveMaxSeq(config: cfg, options: LoadOptions())
        #expect(maxSeq == 131_072)
    }

    @Test("LoadOptions.maxContextLength overrides the checkpoint window")
    func optionOverridesCheckpointWindow() {
        let cfg = ModelConfig(
            architecture: "MuseGlimmerForConditionalGeneration",
            modelType: "museglimmer",
            raw: ["max_position_embeddings": 131_072])
        let maxSeq = MuseGlimmerDense.resolveMaxSeq(
            config: cfg, options: LoadOptions(maxContextLength: 4096))
        #expect(maxSeq == 4096)
    }

    @Test("LoadOptions.maxContextLength can also raise past the advertised window")
    func optionCanRaiseAboveCheckpointWindow() {
        let cfg = ModelConfig(
            architecture: "MuseGlimmerForConditionalGeneration",
            modelType: "museglimmer",
            raw: ["max_position_embeddings": 32_768])
        let maxSeq = MuseGlimmerDense.resolveMaxSeq(
            config: cfg, options: LoadOptions(maxContextLength: 131_072))
        #expect(maxSeq == 131_072)
    }
}
