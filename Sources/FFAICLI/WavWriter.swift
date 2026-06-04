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
// WavWriter — minimal 16-bit PCM WAV encoder for `ffai speak` output.
// Pure Foundation (no AVFoundation dependency): build the 44-byte
// canonical RIFF/WAVE header, then the little-endian int16 sample frames.

import Foundation

enum WavWriter {
    /// Number of bytes in the canonical PCM WAV header.
    private static let headerBytes = 44
    /// 16-bit samples → 2 bytes per sample.
    private static let bytesPerSample = 2
    /// WAVE format tag for uncompressed PCM.
    private static let pcmFormatTag: UInt16 = 1

    /// Write mono `samples` (f32 in roughly [-1, 1]) as a 16-bit PCM WAV
    /// at `sampleRate`. Samples are clamped to [-1, 1] before scaling to
    /// the int16 range so an over-driven vocoder can't wrap around.
    static func write(samples: [Float], sampleRate: Int, to url: URL) throws {
        let channels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(channels) * UInt32(bytesPerSample)
        let blockAlign = channels * UInt16(bytesPerSample)
        let dataBytes = UInt32(samples.count * bytesPerSample)

        var data = Data(capacity: headerBytes + Int(dataBytes))

        func appendLE<T: FixedWidthInteger>(_ value: T) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }

        // RIFF chunk descriptor.
        data.append(contentsOf: Array("RIFF".utf8))
        appendLE(UInt32(headerBytes - 8) + dataBytes)  // overall file size − 8
        data.append(contentsOf: Array("WAVE".utf8))

        // "fmt " sub-chunk.
        data.append(contentsOf: Array("fmt ".utf8))
        appendLE(UInt32(16))  // PCM fmt chunk size
        appendLE(pcmFormatTag)
        appendLE(channels)
        appendLE(UInt32(sampleRate))
        appendLE(byteRate)
        appendLE(blockAlign)
        appendLE(bitsPerSample)

        // "data" sub-chunk.
        data.append(contentsOf: Array("data".utf8))
        appendLE(dataBytes)
        let scale = Float(Int16.max)
        for s in samples {
            let clamped = Swift.max(-1.0, Swift.min(1.0, s))
            appendLE(Int16(clamped * scale))
        }

        try data.write(to: url)
    }
}
