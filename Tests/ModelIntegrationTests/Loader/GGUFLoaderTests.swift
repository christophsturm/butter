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
// GGUF loader integration — open a real GGUF, read its metadata, decode
// a representative tensor of each quant type, and build a tokenizer from
// the embedded vocab. Focused on the loader (`GGUFTensorBundle`), not on
// any one model family.
//
// Skipped by default. Set `$FFAI_GGUF_PATH` to any GGUF checkpoint dir
// (a small one is ideal — these tests only exercise the parser + dequant
// pipeline). Falls back to the DSv4-Flash path (`$FFAI_DSV4_GGUF_PATH` /
// `~/models/deepseek-v4-flash`) when set, so it doubles as DSv4 coverage
// where that large file is staged.

import Foundation
import Testing
import Tokenizers

@testable import FFAI

@Suite("GGUF loader integration", .serialized)
struct GGUFLoaderTests {

    private var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        let candidate =
            env["FFAI_GGUF_PATH"]
            ?? env["FFAI_DSV4_GGUF_PATH"]
            ?? NSString("~/models/deepseek-v4-flash").expandingTildeInPath
        guard FileManager.default.fileExists(atPath: candidate) else { return nil }
        return candidate
    }

    private func open() throws -> GGUFTensorBundle? {
        guard let dir = modelPath else {
            print("GGUFLoaderTests: skipping (no GGUF at FFAI_GGUF_PATH / FFAI_DSV4_GGUF_PATH)")
            return nil
        }
        return try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
    }

    @Test("Open a GGUF: header + architecture + non-trivial tensor table")
    func opensCheckpoint() throws {
        guard let bundle = try open() else { return }
        #expect(bundle.architecture != nil, "GGUF must carry general.architecture")
        #expect(bundle.reader.tensorInfos.count > 0, "tensor info table is empty")
    }

    @Test("Dequant a Q8_0 tensor → correct shape, finite, bounded")
    func dequantQ8_0() throws {
        guard let bundle = try open() else { return }
        guard let info = bundle.reader.tensorInfos.first(where: { $0.type == .q8_0 }) else {
            print("GGUFLoaderTests: no Q8_0 tensors — skipping")
            return
        }
        try assertDequantSane(bundle, info)
    }

    @Test("Dequant a Q2_K tensor → correct shape, finite, bounded")
    func dequantQ2_K() throws {
        guard let bundle = try open() else { return }
        guard let info = bundle.reader.tensorInfos.first(where: { $0.type == .q2_K }) else {
            print("GGUFLoaderTests: no Q2_K tensors — skipping")
            return
        }
        try assertDequantSane(bundle, info)
    }

    @Test("Dequant an IQ2_XXS tensor → correct shape, finite, bounded, non-zero")
    func dequantIQ2_XXS() throws {
        guard let bundle = try open() else { return }
        guard let info = bundle.reader.tensorInfos.first(where: { $0.type == .iq2_xxs }) else {
            print("GGUFLoaderTests: no IQ2_XXS tensors — skipping")
            return
        }
        try assertDequantSane(bundle, info, requireNonZero: true)
    }

    @Test("Build a tokenizer from the GGUF metadata block")
    func buildTokenizer() throws {
        guard let bundle = try open() else { return }
        let tokenizer: any Tokenizer
        do {
            tokenizer = try GGUFTokenizerAdapter.build(reader: bundle.reader)
        } catch GGUFTokenizerAdapter.Error.unsupportedKind(let kind) {
            print("GGUFLoaderTests: tokenizer kind '\(kind)' not in the supported BPE set yet — skipping")
            return
        }
        let ids = tokenizer.encode(text: "The history of the printing press began when")
        #expect(!ids.isEmpty, "encode returned an empty token list")
        #expect(!tokenizer.decode(tokens: ids).isEmpty, "decode returned an empty string")
    }

    /// Dequant a tensor to f32 and assert shape match + finite, bounded
    /// values over a sample (exact numerical checks live in the metaltile
    /// kernel correctness tests; this is the loader-pipeline sanity).
    private func assertDequantSane(
        _ bundle: GGUFTensorBundle, _ info: GGUFTensorInfo, requireNonZero: Bool = false
    ) throws {
        let t = try bundle.tensor(named: info.name, outDtype: .f32)
        #expect(t.shape.map { Int($0) } == info.dimensions.map { Int($0) })
        let sample = t.toArray(as: Float.self).prefix(1024)
        var anyNonZero = false
        for v in sample {
            #expect(v.isFinite, "\(info.type) dequant produced non-finite value")
            #expect(abs(v) < 1e3, "\(info.type) dequant magnitude unreasonable (\(v))")
            if v != 0 { anyNonZero = true }
        }
        if requireNonZero {
            #expect(anyNonZero, "\(info.type) dequant produced all-zero sample")
        }
    }
}
