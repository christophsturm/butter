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
// SpeakCommand — `ffai speak "<text>"`. Loads a text-to-speech model and
// synthesizes a `.wav`. The model streams one waveform chunk per segment;
// we append each chunk as it arrives (so a future player could start on
// segment 1) and write the joined signal to the output file at the end.

import ArgumentParser
import FFAI
import Foundation

struct SpeakCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speak",
        abstract: "Synthesize speech from text to a .wav file (text-to-speech)."
    )

    // Default to Soprano: it synthesizes from RAW text (BPE → LLM → Vocos)
    // with no external grapheme-to-phoneme step. Phoneme-based models
    // (Kokoro, StyleTTS2) need IPA input — FFAI bundles no g2p — and most
    // other TTS families are staged ports whose synthesis isn't wired yet.
    @Option(name: .shortAndLong, help: "HuggingFace repo id or local path to a TTS model.")
    var model: String = "mlx-community/Soprano-80M-4bit"

    @Argument(help: "Text to speak.")
    var text: String

    @Option(name: .long, help: "Speaker / voice id for multi-speaker models.")
    var voice: String?

    @Option(name: .long, help: "Speaking-rate multiplier (1.0 = normal).")
    var speed: Float = 1.0

    @Option(name: .long, help: "Output .wav path.")
    var out: String = "out.wav"

    @Flag(
        name: .long, inversion: .prefixedNo,
        help: "Stream synthesis segment-by-segment (default). Disable with --no-streaming.")
    var streaming: Bool = true

    @Flag(name: .long, help: "Enable debug logging for every FFAI subsystem.")
    var debug: Bool = false

    func run() async throws {
        if debug { Debug.enableAll() }

        print("ffai \(FFAI.version) — loading \(model)…")
        let loadStart = Date()
        let dir = try await ModelLocator().resolve(idOrPath: model)
        let loaded = try await AudioModelRegistry.load(directory: dir)
        print("loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStart)))s")

        guard let synthesizer = loaded.synthesizer else {
            throw ValidationError(
                "\(model) is not a text-to-speech model (no synthesis path).")
        }

        print("---")
        let start = Date()
        var samples: [Float] = []
        var segments = 0
        let stream = synthesizer.synthesizeStream(text: text, voice: voice, speed: speed)
        for try await chunk in stream {
            samples.append(contentsOf: chunk.toFloatArray())
            segments += 1
            if streaming {
                print(
                    "segment \(segments): +\(chunk.shape.first ?? 0) samples "
                        + "(\(samples.count) total)")
            }
        }

        let outURL = URL(fileURLWithPath: out)
        try WavWriter.write(
            samples: samples, sampleRate: synthesizer.sampleRate, to: outURL)
        print("---")
        let seconds = Double(samples.count) / Double(synthesizer.sampleRate)
        print(
            "wrote \(outURL.path) — \(String(format: "%.2f", seconds))s of audio "
                + "(\(samples.count) samples @ \(synthesizer.sampleRate) Hz) "
                + "in \(String(format: "%.2f", Date().timeIntervalSince(start)))s")
    }
}
