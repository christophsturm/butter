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
// TranscribeCommand — `ffai transcribe <audio>`. Loads a speech-to-text
// model and streams the transcript to stdout as it decodes. Streaming is
// the primary route; `--no-streaming` prints once at the end.

import ArgumentParser
import FFAI
import Foundation

/// Sample rate STT front-ends expect (16 kHz mono).
private let kSTTSampleRate = 16000

struct TranscribeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe an audio file to text (speech-to-text), streaming."
    )

    @Option(name: .shortAndLong, help: "HuggingFace repo id or local path to an STT model.")
    var model: String = "mlx-community/granite-4.0-1b-speech-5bit"

    @Argument(help: "Path to an audio file (WAV / any AVFoundation-decodable format).")
    var audioFile: String

    @Option(name: .long, help: "Maximum tokens to decode.")
    var maxNewTokens: Int = 512

    @Option(name: .long, help: "Sampling temperature. 0 = greedy (deterministic).")
    var temperature: Float = 0.0

    @Flag(
        name: .long, inversion: .prefixedNo,
        help: "Stream the transcript as it decodes (default). Disable with --no-streaming.")
    var streaming: Bool = true

    @Flag(name: .long, help: "Enable debug logging for every FFAI subsystem.")
    var debug: Bool = false

    func run() async throws {
        if debug { Debug.enableAll() }

        print("ffai \(FFAI.version) — loading \(model)…")
        let loadStart = Date()
        let m = try await Model.load(model)
        print("loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStart)))s")

        guard m.engine is any Transcribing else {
            throw ValidationError(
                "\(model) is not a speech-to-text model (no transcribe path).")
        }

        let waveform = try MediaUtils.loadWaveform(
            url: URL(fileURLWithPath: audioFile), targetRate: kSTTSampleRate)

        print("---")
        let start = Date()
        let stream = m.transcribeStream(
            waveform, maxNewTokens: maxNewTokens, temperature: temperature)
        var text = ""
        for try await chunk in stream {
            text += chunk
            if streaming {
                print(chunk, terminator: "")
                fflush(stdout)
            }
        }
        if streaming {
            print("")
        } else {
            print(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        print("---")
        print("transcribed in \(String(format: "%.2f", Date().timeIntervalSince(start)))s")
    }
}
