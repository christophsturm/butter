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
// TTSPhonemeText — shared text helpers for the phoneme-based TTS
// families (Kokoro / StyleTTS2).
//
// These models synthesize from a PHONEME alphabet (IPA + stress marks),
// not from raw graphemes, and FFAI ships no bundled grapheme-to-phoneme
// (G2P) front-end. The `StreamingSynthesizer.synthesizeStream(text:)`
// path therefore needs two pure-text utilities:
//
//   • `looksPhonemized(_:)` — distinguish an already-phonemized IPA
//     string from raw English. The discriminator is the presence of
//     IPA-specific characters (ɛ, ʊ, ˈ, ː …): the Kokoro / StyleTTS2
//     symbol tables ALSO contain plain ASCII letters, so "Hello world"
//     would tokenize to *something* — just the wrong something. Without
//     this check the stream would silently ship garbage audio.
//   • `sentenceSegments(_:)` — split an utterance into sentence-sized
//     chunks so streaming can begin on sentence 1 while later sentences
//     synthesize, the TTS analogue of token-streaming.

import Foundation

/// Pure-text helpers shared by the phoneme-based TTS models. No model
/// state — just string analysis used by the streaming front-ends.
enum TTSPhonemeText {

    /// IPA / phoneme-specific characters that do NOT occur in ordinary
    /// English orthography. Their presence is strong evidence that a
    /// string is already phonemized (the Kokoro / StyleTTS2 symbol tables
    /// publish exactly this alphabet). Mirrors `StyleTTS2TextCleaner`'s
    /// `lettersIPA` plus the stress / length suprasegmentals so the check
    /// stays in sync with what the tokenizers actually accept.
    private static let ipaCharacters: Set<Character> = {
        // IPA letters plus suprasegmentals (clicks, stress, length, tone,
        // and the combining marks Kokoro / Misaki emit).
        let ipa =
            "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ"
            + "ǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\u{2018}\u{0329}\u{2019}ᵻ"
        return Set(ipa)
    }()

    /// Whether `text` looks like an already-phonemized string the
    /// Kokoro / StyleTTS2 acoustic models can consume directly.
    ///
    /// Returns `true` when the string contains at least one IPA /
    /// suprasegmental character (see `ipaCharacters`). Plain English text
    /// (ASCII letters + punctuation only) returns `false` — those
    /// graphemes need a real G2P pass first, which FFAI does not bundle.
    /// Empty / whitespace-only input returns `false`.
    static func looksPhonemized(_ text: String) -> Bool {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return false }
        return text.contains { ipaCharacters.contains($0) }
    }

    /// Sentence-terminating punctuation used to chunk an utterance for
    /// streaming. Covers ASCII and the inverted / full-width marks the
    /// TTS symbol tables include.
    private static let sentenceTerminators: Set<Character> = [
        ".", "!", "?", "…", "。", "！", "？",
    ]

    /// Split `text` into sentence-sized segments, keeping each segment's
    /// terminating punctuation attached (the acoustic model is trained
    /// with end-of-sentence punctuation, which shapes final prosody).
    ///
    /// Splitting on phoneme strings is safe: the IPA alphabet does not
    /// reuse `.`/`!`/`?` for anything other than sentence boundaries, and
    /// concatenating the per-segment syntheses reproduces a single synth
    /// of the whole string (each chunk is synthesized independently at the
    /// same per-segment shape the batch path uses).
    ///
    /// Whitespace-only segments are dropped. A string with no terminator
    /// yields a single segment (the whole trimmed string).
    static func sentenceSegments(_ text: String) -> [String] {
        var segments: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if sentenceTerminators.contains(ch) {
                let trimmed = current.trimmingCharacters(
                    in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { segments.append(trimmed) }
                current = ""
            }
        }
        // Trailing run with no terminator (or the whole string when there
        // is no sentence punctuation at all).
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { segments.append(tail) }
        return segments
    }
}
