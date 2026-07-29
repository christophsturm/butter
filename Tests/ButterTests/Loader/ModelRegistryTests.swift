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
import Foundation
import Testing

@testable import ButterSwift

@Suite("ModelRegistry")
struct ModelRegistryTests {
    static func writeBundle(in dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Empty safetensors file (header == "{}"). Bundle.dispatchAndLoad
        // will throw before touching tensor data when arch is unknown.
        let header = "{}"
        let headerBytes = Array(header.utf8)
        var headerLen = UInt64(headerBytes.count)
        var data = Data()
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(contentsOf: headerBytes)
        try data.write(to: dir.appendingPathComponent("model.safetensors"))
    }

    @Test("unknown architecture is rejected with ModelError.unsupportedArchitecture")
    func unknownArchitecture() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("butter-reg-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeBundle(in: dir)

        let cfgURL = dir.appendingPathComponent("config.json")
        try #"""
        {"architectures": ["NotAFamily"], "model_type": "alien"}
        """#.write(to: cfgURL, atomically: true, encoding: .utf8)

        let cfg = try ModelConfig.load(from: dir)
        let bundle = try SafeTensorsBundle(directory: dir)
        do {
            _ = try ModelRegistry.dispatchAndLoad(
                config: cfg, weights: bundle,
                options: LoadOptions(), device: .shared
            )
            Issue.record("expected throw")
        } catch let e as ModelError {
            if case .unsupportedArchitecture(let a) = e {
                #expect(a == "NotAFamily")
            } else {
                Issue.record("expected .unsupportedArchitecture, got \(e)")
            }
        }
    }

    @Test("Llama3Error.missingConfig description")
    func llamaMissingConfig() {
        let e = Llama3Error.missingConfig
        #expect(String(describing: e).contains("missing"))
    }

    @Test("Llama3.variant returns Llama3Dense for any llama config")
    func variantDispatch() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("butter-variant-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try #"{"model_type": "llama"}"#.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8
        )
        let cfg = try ModelConfig.load(from: dir)
        let v = try Llama3.variant(for: cfg)
        #expect(v is Llama3Dense.Type)
    }

    // ─── Data-driven dispatch table routing ──────────────────────────
    // `entry(for:)` is the routing predicate behind `dispatchAndLoad` —
    // testing it validates the table wiring without loading any weights.

    private func cfg(
        architecture: String? = nil, modelType: String? = nil,
        raw: [String: Any] = [:]
    ) -> ModelConfig {
        ModelConfig(architecture: architecture, modelType: modelType, raw: raw)
    }

    @Test("text tier routes each family's model_type to the right entry")
    func textTierRouting() {
        // Build configs from the SAME static sets the table dispatches on,
        // so this stays correct as families evolve their strings.
        #expect(
            ModelRegistry.entry(for: cfg(modelType: Llama3.modelTypes.first))?.family
                == "Llama 3.x")
        #expect(
            ModelRegistry.entry(for: cfg(modelType: Qwen2.modelTypes.first))?.family
                == "Qwen 2 / 2.5")
        #expect(
            ModelRegistry.entry(for: cfg(modelType: Qwen3.modelTypes.first))?.family
                == "Qwen 3")
        #expect(
            ModelRegistry.entry(for: cfg(modelType: Gemma2.modelTypes.first))?.family
                == "Gemma 2")
        #expect(
            ModelRegistry.entry(for: cfg(modelType: GPTOSS.modelTypes.first))?.family
                == "GPT-OSS")
        #expect(
            ModelRegistry.entry(for: cfg(modelType: Granite4.modelTypes.first))?.family
                == "Granite 4")
    }

    @Test("vision tier routes each VL architecture to the right entry")
    func visionTierRouting() {
        #expect(
            ModelRegistry.entry(
                for: cfg(architecture: "Gemma3ForConditionalGeneration"))?.family
                == "Gemma 3 VL")
        #expect(
            ModelRegistry.entry(
                for: cfg(architecture: "Qwen2VLForConditionalGeneration"))?.family
                == "Qwen 2-VL")
        #expect(
            ModelRegistry.entry(
                for: cfg(architecture: "Qwen2_5_VLForConditionalGeneration"))?.family
                == "Qwen 2.5-VL")
        // GLM-OCR is filed under the OCR kind, distinct from a plain VLM.
        let glm = ModelRegistry.entry(
            for: cfg(architecture: GlmOcr.architectures.first))
        #expect(glm?.family == "GLM-OCR")
        #expect(glm?.kind == .ocr)
    }

    @Test("ordering: MiniCPM5 wins over generic Llama for a llama-arch config")
    func miniCPM5BeforeLlama() {
        let mini = cfg(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["_name_or_path": "MiniCPM5-1B-MLX"])
        #expect(ModelRegistry.entry(for: mini)?.family == "MiniCPM5")
        // A plain llama config (no marker) still routes to Llama 3.x.
        #expect(
            ModelRegistry.entry(for: cfg(modelType: "llama"))?.family == "Llama 3.x")
    }

    @Test("ordering: Gemma 4 drafter wins over Gemma 4")
    func gemma4DrafterBeforeGemma4() {
        #expect(
            ModelRegistry.entry(
                for: cfg(modelType: Gemma4.drafterModelTypes.first))?.family
                == "Gemma 4 drafter")
        #expect(
            ModelRegistry.entry(for: cfg(modelType: Gemma4.modelTypes.first))?.family
                == "Gemma 4")
    }

    @Test("a VL config with no handler resolves to no entry (→ visionModelNotIntegrated)")
    func unhandledVisionConfig() {
        let c = cfg(
            architecture: "TotallyUnknownVLForConditionalGeneration",
            raw: ["vision_config": [:]])
        #expect(ModelRegistry.entry(for: c) == nil)
    }

    @Test("an unknown non-VL config resolves to no entry (→ unsupportedArchitecture)")
    func unhandledTextConfig() {
        #expect(
            ModelRegistry.entry(
                for: cfg(architecture: "NotAFamily", modelType: "alien")) == nil)
    }

    // ─── Catalog (single source for `ffai models`) ───────────────────

    @Test("catalog covers every implemented kind, each non-empty")
    func catalogCoversImplementedKinds() {
        let kinds = Set(ModelRegistry.catalog.map(\.kind))
        for kind in ModelType.allCases where kind.isImplemented {
            #expect(kinds.contains(kind), "catalog missing kind \(kind)")
        }
        for (_, entries) in ModelRegistry.catalog {
            #expect(!entries.isEmpty)
            for e in entries { #expect(!e.family.isEmpty) }
        }
    }

    @Test("catalog includes the audio + VAD families folded into the registry")
    func catalogIncludesAudioAndVAD() {
        let families = Set(ModelRegistry.catalogEntries.map(\.family))
        #expect(families.contains("Whisper STT"))      // stt
        #expect(families.contains("Kokoro TTS"))       // tts
        #expect(families.contains("Qwen-Omni"))        // omni
        #expect(families.contains("DeepFilterNet"))    // sts
        #expect(families.contains("Silero VAD"))       // vad
        // …alongside the text/VL families that drive dispatch.
        #expect(families.contains("Llama 3.x"))
        #expect(families.contains("Qwen 2-VL"))
    }

    @Test("only text/VL entries carry a load closure; audio rows are catalog-only")
    func loadablitySplit() {
        for e in ModelRegistry.visionEntries + ModelRegistry.textEntries {
            #expect(e.load != nil, "\(e.family) should be loadable")
        }
        for e in ModelRegistry.audioEntries {
            #expect(e.load == nil, "\(e.family) is catalog-only for now")
        }
    }
}
