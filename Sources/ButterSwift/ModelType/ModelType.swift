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
// ModelType — the primary, human-facing category of a model. The registry
// tags every supported family with a `ModelType`; it's the grouping key for
// the `butter models` catalog and `butter inspect`.
//
// `ModelType` is NOT a substitute for `Capability` — capabilities are the
// fine-grained runtime truth (and overlap: an OCR model has the same
// `imageIn`/`textOut` capabilities as a VLM). `ModelType` is the single
// catalog label a model is filed under. Detection-only kinds (vad) have no
// output `Capability` at all, so the kind cannot be derived from
// capabilities alone — the registry assigns it explicitly.

import Foundation

public enum ModelType: String, Sendable, Hashable, CaseIterable, Codable {
    // ── Understanding (X → text) ─────────────────────────────────────
    /// Text-only large language model.
    case llm
    /// Vision-language model — image/video + text in, text out.
    case vlm
    /// OCR / document understanding — image in, text out (same
    /// capabilities as a VLM; filed separately as a catalog label).
    case ocr
    /// Omni-modal — text + image + video + audio in, text out.
    case omni
    /// Speech-to-text (transcription).
    case stt

    // ── Audio generation / transformation ────────────────────────────
    /// Text-to-speech.
    case tts
    /// Speech-to-speech — enhancement / separation / segmentation.
    case sts
    /// Voice-activity detection / diarization / endpointing (detection
    /// only — no generative output).
    case vad

    // ── Reserved (future, no family wires these yet) ──────────────────
    // Declared ahead of the first generator so the catalog vocabulary is
    // stable when it lands — mirrors the `Capability.imageOut`/`videoOut`
    // future-proofing.
    /// Text-to-image generation.
    case tti
    /// Image-to-image generation (edit / style / super-resolution).
    case iti
    /// Text-to-video generation.
    case ttv
    /// Video-to-video generation (edit / interpolation).
    case vtv
    /// Image-to-video generation.
    case itv
    /// Image segmentation.
    case segmentation

    /// Catalog group header for `butter models`.
    public var title: String {
        switch self {
        case .llm: return "Large language models"
        case .vlm: return "Vision-language models"
        case .ocr: return "OCR / document understanding"
        case .omni: return "Omni-modal models"
        case .stt: return "Speech-to-text"
        case .tts: return "Text-to-speech"
        case .sts: return "Speech-to-speech / audio"
        case .vad: return "Voice activity & diarization"
        case .tti: return "Text-to-image"
        case .iti: return "Image-to-image"
        case .ttv: return "Text-to-video"
        case .vtv: return "Video-to-video"
        case .itv: return "Image-to-video"
        case .segmentation: return "Image segmentation"
        }
    }

    /// Whether any Butter family currently ships for this kind. The reserved
    /// generative kinds are `false` until their first family lands.
    public var isImplemented: Bool {
        switch self {
        case .llm, .vlm, .ocr, .omni, .stt, .tts, .sts, .vad: return true
        case .tti, .iti, .ttv, .vtv, .itv, .segmentation: return false
        }
    }
}
