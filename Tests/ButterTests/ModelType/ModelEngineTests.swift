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
// ModelEngineTests — the engine protocol layer's value types + the VAD
// output → `DetectionEvent` mapping adapters (the only non-marker logic in
// the conformances). Mirrors `Sources/Butter/ModelType/`.

import Foundation
import Testing

@testable import ButterSwift

@Suite("Engine protocol layer")
struct ModelEngineTests {

    // MARK: - Parameter defaults

    @Test("TranscribeParameters / SynthesizeParameters defaults")
    func parameterDefaults() {
        let t = TranscribeParameters()
        #expect(t.maxNewTokens == 512)
        #expect(t.temperature == 0.0)

        let s = SynthesizeParameters()
        #expect(s.voice == nil)
        #expect(s.speed == 1.0)
    }

    // MARK: - DetectionEvent seconds math

    @Test("DetectionEvent maps samples to seconds")
    func detectionEventSeconds() {
        let e = DetectionEvent(
            startSample: 16000, endSample: 32000, sampleRate: 16000,
            label: "speech", score: 1.0)
        #expect(e.startSeconds == 1.0)
        #expect(e.endSeconds == 2.0)
        #expect(e.speaker == nil)
    }

    // MARK: - VAD output adapters

    @Test("VADOutput segments become speech events")
    func vadOutputToEvents() {
        let out = VADOutput(
            probabilities: [0.9, 0.8],
            frameStrideSamples: 512,
            sampleRate: 16000,
            segments: [
                VADSpeechSegment(startSample: 0, endSample: 8000, sampleRate: 16000),
                VADSpeechSegment(startSample: 16000, endSample: 24000, sampleRate: 16000),
            ])
        let events = out.detectionEvents
        #expect(events.count == 2)
        #expect(events.allSatisfy { $0.label == "speech" })
        #expect(events[0].startSample == 0 && events[0].endSample == 8000)
        #expect(events[1].sampleRate == 16000)
    }

    @Test("DiarizationOutput segments carry speaker index")
    func diarizationToEvents() {
        let out = DiarizationOutput(
            speakerProbabilities: [[0.9, 0.1]],
            frameStrideSamples: 512,
            sampleRate: 16000,
            segments: [
                DiarizationSegment(startSeconds: 0.0, endSeconds: 1.0, speaker: 0),
                DiarizationSegment(startSeconds: 1.0, endSeconds: 2.0, speaker: 1),
            ],
            numSpeakers: 2)
        let events = out.detectionEvents
        #expect(events.count == 2)
        #expect(events.allSatisfy { $0.label == "speaker" })
        #expect(events[0].speaker == 0 && events[1].speaker == 1)
        // 1.0s @ 16 kHz = sample 16000.
        #expect(events[1].startSample == 16000)
    }

    @Test("VADEndpointOutput yields one event only when triggered")
    func endpointToEvents() {
        let fired = VADEndpointOutput(probability: 0.92, prediction: 1)
        let firedEvents = fired.detectionEvents(sampleRate: 16000)
        #expect(firedEvents.count == 1)
        #expect(firedEvents[0].label == "endpoint")
        #expect(firedEvents[0].score == 0.92)

        let quiet = VADEndpointOutput(probability: 0.1, prediction: 0)
        #expect(quiet.detectionEvents(sampleRate: 16000).isEmpty)
    }
}
