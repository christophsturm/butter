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
// DeepSeek-V4 model integration. Mirrors the common model-integration
// pattern (loads / shapes + configs / default parameters / coherent
// output). GGUF-loader-specific coverage lives in
// `Tests/ModelIntegrationTests/Loader/GGUFLoaderTests.swift`.
//
// Skipped by default — the DSv4-Flash checkpoint is ~86 GB, so every
// test guard-returns unless a checkpoint is staged at
// `$FFAI_DSV4_GGUF_PATH` (default `~/models/deepseek-v4-flash`). The
// GGUF path is a parallel loader (`DeepSeekV4Flash.loadModelFromGGUF`),
// not the standard `Model.load` safetensors flow, so these construct
// the model directly.

import Foundation
import Testing

@testable import FFAI

@Suite("DeepSeekV4 integration", .serialized)
struct DeepSeekV4IntegrationTests {

    /// Local checkpoint dir, or `nil` to skip (model too big for CI).
    private var modelPath: String? {
        let env =
            ProcessInfo.processInfo.environment["FFAI_DSV4_GGUF_PATH"]
            ?? NSString("~/models/deepseek-v4-flash").expandingTildeInPath
        guard FileManager.default.fileExists(atPath: env) else { return nil }
        return env
    }

    /// Minimal config synthesized from GGUF hparams (the GGUF loader
    /// reads the rest of the structure off the file itself).
    private func ggufConfig(_ bundle: GGUFTensorBundle) -> ModelConfig {
        let hidden = Int(bundle.reader.metadataUInt32("deepseek4.embedding_length") ?? 4096)
        let nLayers = Int(bundle.reader.metadataUInt32("deepseek4.block_count") ?? 43)
        let vocab = Int(bundle.reader.metadataUInt32("deepseek4.vocab_size") ?? 129_280)
        let nHeads = Int(bundle.reader.metadataUInt32("deepseek4.attention.head_count") ?? 64)
        return ModelConfig(
            architecture: "DeepSeekV4ForCausalLM", modelType: "deepseek4",
            raw: [
                "hidden_size": hidden, "num_hidden_layers": nLayers,
                "vocab_size": vocab, "num_attention_heads": nHeads,
            ])
    }

    // 1. Loader + expected shapes / config.
    @Test("Loads from GGUF and exposes the expected layer geometry")
    func loadsAndHasExpectedShapes() throws {
        guard let dir = modelPath else {
            print("DeepSeekV4IntegrationTests: skipping (no model at FFAI_DSV4_GGUF_PATH)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        let model = try DeepSeekV4Flash.loadModelFromGGUF(
            config: ggufConfig(bundle), gguf: bundle, options: LoadOptions(), device: .shared)

        #expect(model.textConfig.nLayers == 43)
        // compress_ratios carries one extra entry for the MTP slot.
        #expect(model.layerCompressRatios.count >= model.textConfig.nLayers)

        // Layer 0 is full-attention (compress_ratio 0); loading it dequants
        // the layer tensors and exposes per-head learnable attn sinks.
        let layer0 = try model.layer(0)
        #expect(layer0.compressRatio == 0)
        #expect(layer0.layerIndex == 0)
        #expect(layer0.attnSinks.shape.map { Int($0) } == [64])
        model.releaseLayer(0)
    }

    // 2. Default generation parameters.
    @Test("Default generation parameters match the DSv4 recipe")
    func defaultGenerationParameters() {
        let p = DeepSeekV4Flash.defaultGenerationParameters
        // No model-specific maxTokens override — uses the framework default.
        #expect(p.maxTokens == GenerationParameters().maxTokens)
        #expect(p.prefillStepSize == 4096)
        #expect(p.temperature == 0.6)
        #expect(p.topP == 0.95)
        #expect(p.topK == 64)
    }

    // 3. Coherent output — one forward must produce finite, NaN-free logits
    //    and a real argmax (the model-too-big-to-CI proxy for "generates
    //    sane text"; full multi-token generate lands when the GGUF path
    //    wires into `Model.generate`).
    @Test("Forward from BOS produces finite, NaN-free logits")
    func forwardProducesSaneLogits() throws {
        guard let dir = modelPath else {
            print("DeepSeekV4IntegrationTests: skipping (no model at FFAI_DSV4_GGUF_PATH)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        let model = try DeepSeekV4Flash.loadModelFromGGUF(
            config: ggufConfig(bundle), gguf: bundle, options: LoadOptions(), device: .shared)
        let state = model.makeDecodeState()
        let bos = Int(bundle.reader.metadataUInt32("tokenizer.ggml.bos_token_id") ?? 0)

        let logits = try model.forwardAllLayers(inputTokenId: bos, state: state)
        let host = logits.toFloatArray()

        let nNaN = host.reduce(0) { $0 + ($1.isNaN ? 1 : 0) }
        #expect(nNaN == 0, "logits has \(nNaN) NaN values")

        var maxIdx = 0
        var maxVal: Float = -.infinity
        for (i, v) in host.enumerated() where v > maxVal { maxVal = v; maxIdx = i }
        #expect(maxVal.isFinite, "argmax logit not finite: \(maxVal)")
        #expect(maxIdx >= 0 && maxIdx < host.count)
    }
}
