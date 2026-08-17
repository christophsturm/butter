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
// Qwen3xTextTests — unit coverage for `Sources/ButterSwift/Models/Text/Qwen3xText.swift`.
//
// Offline. The "Qwen3x" file covers BOTH Qwen 3.5 and Qwen 3.6 (same
// stack-interleaved GDN + attention architecture, same `qwen3_5*`
// model_type strings; the 3.6 family root is a doc-only anchor). This
// file covers:
//   • `Qwen35Hybrid` variant surface (capabilities + greedy defaults),
//   • `Qwen35LayerKind.init(from:)` — the `layer_types` entry parser
//     (`"linear_attention"` = GDN / `"full_attention"` = attention,
//     plus the unknown-name rejection path).

import Foundation
import Testing

@testable import ButterSwift

@Suite("Qwen35Hybrid Variant Surface")
struct Qwen35HybridTests {

    @Test("Qwen35Hybrid advertises text in/out capabilities")
    func capabilities() {
        #expect(Qwen35Hybrid.availableCapabilities.contains(.textIn))
        #expect(Qwen35Hybrid.availableCapabilities.contains(.textOut))
        #expect(!Qwen35Hybrid.availableCapabilities.contains(.imageIn))
    }

    /// Qwen 3.5 ships base + instruction-tuned checkpoints. Greedy by
    /// default keeps the integration suite deterministic.
    @Test("Qwen35Hybrid default generation parameters are greedy")
    func defaultGenerationParameters() {
        let p = Qwen35Hybrid.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature == 0.0)
        #expect(p.topP == 1.0)
        #expect(p.topK == 0)
        // `nil` on purpose: this family used to hardcode 1024, silently
        // overriding `Qwen35Model.defaultPrefillStepSize` (4096, this
        // hybrid family, including its dense 27B variant) on every real
        // generation call. A
        // chunk-size sweep found no numerical difference between chunk
        // sizes at this range (bit-identical dumped prefill logits) and
        // a throughput win from deferring to the tuned engine default,
        // so the family-level override was removed rather than bumped
        // to a second hardcoded number.
        #expect(p.prefillStepSize == nil)
    }
}

@Suite("Qwen35Hybrid MTP sidecar discovery")
struct Qwen35HybridMTPSidecarTests {
    @Test("GGUF companions precede the shared legacy directory")
    func ggufCandidateOrder() {
        let root = URL(fileURLWithPath: "/models", isDirectory: true)
        let gguf = root.appendingPathComponent("Qwen3.8-27B-Q4_K_M.gguf")
        let candidates = Qwen35Hybrid.mtpSidecarCandidates(
            weightsDirectory: root, weightURLs: [gguf], overridePath: nil)
        #expect(
            candidates.map(\.path) == [
                "/models/Qwen3.8-27B-Q4_K_M.gguf.mtp-sidecar",
                "/models/Qwen3.8-27B-Q4_K_M.mtp-sidecar",
                "/models/mtp-sidecar",
            ])
    }

    @Test("explicit sidecar override wins and duplicates are removed")
    func explicitOverride() {
        let root = URL(fileURLWithPath: "/models", isDirectory: true)
        let candidates = Qwen35Hybrid.mtpSidecarCandidates(
            weightsDirectory: root,
            weightURLs: [root.appendingPathComponent("model.safetensors")],
            overridePath: "/models/mtp-sidecar")
        #expect(candidates.map(\.path) == ["/models/mtp-sidecar"])
    }

    @Test("MTP loader accepts old namespaced and Qwen3.8 standalone head keys")
    func tensorPrefixes() throws {
        #expect(try Qwen3xMTPHead.tensorPrefix(for: ["mtp.fc.weight"]) == "mtp.")
        #expect(try Qwen3xMTPHead.tensorPrefix(for: ["fc.weight"]) == "")
        #expect(
            try Qwen3xMTPHead.tensorPrefix(
                for: ["fc.weight", "mtp.fc.weight"]) == "mtp.")
        #expect(throws: Qwen35Error.self) {
            _ = try Qwen3xMTPHead.tensorPrefix(for: ["norm.weight"])
        }
    }
}

@Suite("Qwen35LayerKind layer_types Parser")
struct Qwen35LayerKindTests {

    @Test("layer_types entries map to GDN (linear_attention) and attention (full_attention)")
    func validNames() throws {
        #expect(try Qwen35LayerKind(from: "linear_attention") == .gdn)
        #expect(try Qwen35LayerKind(from: "full_attention") == .attention)
    }

    @Test("unknown layer_types entry throws unsupportedConfig")
    func unknownRejected() {
        #expect(throws: Qwen35Error.self) {
            _ = try Qwen35LayerKind(from: "mamba")
        }
        #expect(throws: Qwen35Error.self) {
            _ = try Qwen35LayerKind(from: "sliding_attention")
        }
    }
}
