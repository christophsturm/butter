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
// TTSPhonemeTextTests — fast, checkpoint-free unit tests for the shared
// phoneme-text helpers the Kokoro / StyleTTS2 streaming front-ends use to
// decide whether text is phonemized and to chunk it into sentences.

import Foundation
import Testing

@testable import FFAI

@Suite("TTSPhonemeText")
struct TTSPhonemeTextTests {

    @Test("looksPhonemized — IPA strings are accepted")
    func phonemizedAccepted() {
        // Real Kokoro / StyleTTS2 phoneme strings carry IPA + stress marks.
        #expect(TTSPhonemeText.looksPhonemized("hɛlˈoʊ wˈɜːld"))
        #expect(TTSPhonemeText.looksPhonemized("ðə kwˈɪk braʊn fˈɑks"))
        // A single IPA character is enough signal.
        #expect(TTSPhonemeText.looksPhonemized("hello ʊ"))
    }

    @Test("looksPhonemized — raw English is rejected")
    func rawTextRejected() {
        // ASCII-only graphemes need a G2P pass FFAI doesn't bundle.
        #expect(!TTSPhonemeText.looksPhonemized("Hello world."))
        #expect(!TTSPhonemeText.looksPhonemized("The quick brown fox."))
        #expect(!TTSPhonemeText.looksPhonemized("12345 !?,."))
    }

    @Test("looksPhonemized — empty / whitespace is rejected")
    func emptyRejected() {
        #expect(!TTSPhonemeText.looksPhonemized(""))
        #expect(!TTSPhonemeText.looksPhonemized("   \n\t "))
    }

    @Test("sentenceSegments — splits on terminators, keeps punctuation")
    func splitsSentences() {
        let segs = TTSPhonemeText.sentenceSegments("hɛlˈoʊ. wˈɜːld!")
        #expect(segs == ["hɛlˈoʊ.", "wˈɜːld!"])
    }

    @Test("sentenceSegments — no terminator yields one trimmed segment")
    func singleSegment() {
        #expect(TTSPhonemeText.sentenceSegments("  hɛlˈoʊ  ") == ["hɛlˈoʊ"])
    }

    @Test("sentenceSegments — trailing run with no terminator is kept")
    func trailingRun() {
        let segs = TTSPhonemeText.sentenceSegments("ˈeɪ. biː? siː")
        #expect(segs == ["ˈeɪ.", "biː?", "siː"])
    }

    @Test("sentenceSegments — empty / whitespace yields no segments")
    func emptyNoSegments() {
        #expect(TTSPhonemeText.sentenceSegments("").isEmpty)
        #expect(TTSPhonemeText.sentenceSegments("   ").isEmpty)
        #expect(TTSPhonemeText.sentenceSegments("\n\t  \n").isEmpty)
    }
}
