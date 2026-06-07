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
// Kokoro — a StyleTTS2-family text-to-speech model. Kokoro turns a
// phonemized text sequence into a waveform through the StyleTTS2
// pipeline:
//
//   phonemes ──text encoder (PLBert)──▶ token features
//            ──prosody predictor (+ voice style)──▶ durations + F0/N
//            ──length-regulated decoder──▶ predicted STFT (re, im)
//            ──iSTFTNet vocoder──▶ waveform
//
// FFAI ships the `vocoder` (iSTFT overlap-add) Metal kernel — the
// waveform-synthesis tail — and `Ops.vocoderISTFT` wraps it. This file
// is the FFAI Kokoro model: a `KokoroVocoder` that runs the iSTFTNet
// tail on the GPU, plus the surrounding model scaffold.
//
// ## Scope note
//
// The full StyleTTS2 acoustic stack (the PLBert text encoder, the
// duration / F0 / N prosody predictors, the AdaIN-residual decoder) is
// a large port. FFAI's contribution today is the GPU vocoder tail
// and the model-level plumbing: `Kokoro.synthesize` consumes a
// predicted complex spectrogram and produces a waveform. The acoustic
// front-end is loaded from the checkpoint when present; when it is
// not yet wired, `synthesizeFromSpectrogram` is the supported entry
// point and `synthesize(phonemes:)` reports the front-end as
// unavailable.
//
// ## Voices
//
// The default voice is `af_heart` — the standard Kokoro v1.0
// reference voice. Override per call via
// `AudioGenerationParameters.voice` or persist a different active
// voice with `setVoice(_:)`. The full Kokoro v1.0 voice catalogue is
// exposed via `availableVoices`; each voice is a 256-dim style vector
// the prosody predictor + decoder consume. Voice vectors are not
// bundled with FFAI — they're resolved on first use from the
// HuggingFace cache (the Kokoro repo ships them alongside the model
// weights) so we don't redistribute weights under a different licence.
//
// ## Phonemizer (text-side front-end)
//
// Kokoro takes *phonemes*, not raw text. The text-to-phoneme step
// runs through the `Phonemizer` protocol (`Audio/Phonemizer.swift`);
// callers register a provider via `PhonemizerRegistry.shared`. Misaki
// G2P (Apache 2.0) is the upstream Kokoro choice for English; for
// other languages users typically plug in espeak-ng (GPL — not
// bundled) or a neural G2P model. FFAI ships the registry surface and
// the AudioModel-level `voice` plumbing so the acoustic-stack port
// (when it lands) drops straight in.

import Foundation
import Metal

// ─── Configuration ───────────────────────────────────────────────────

/// Kokoro / iSTFTNet hyper-parameters, decoded from `config.json`.
public struct KokoroConfig: Sendable {
    /// Phoneme vocabulary size (`n_token`).
    public let nToken: Int
    /// Acoustic hidden dimension (`hidden_dim`).
    public let hidden: Int
    /// Mel bins the acoustic decoder predicts (`n_mels`).
    public let nMels: Int
    /// Output waveform sample rate (24 kHz for Kokoro).
    public let sampleRate: Int
    /// iSTFT FFT length — Kokoro's iSTFTNet head uses a tiny 20-sample
    /// FFT with hop 5 (`gen_istft_n_fft`, `gen_istft_hop_size`).
    public let istftNFFT: Int
    /// iSTFT hop length.
    public let istftHop: Int

    public init(
        nToken: Int, hidden: Int, nMels: Int,
        sampleRate: Int = 24_000, istftNFFT: Int = 20,
        istftHop: Int = 5
    ) {
        self.nToken = nToken
        self.hidden = hidden
        self.nMels = nMels
        self.sampleRate = sampleRate
        self.istftNFFT = istftNFFT
        self.istftHop = istftHop
    }

    /// Build a `KokoroConfig` from a decoded `config.json`. Kokoro's
    /// config nests the iSTFTNet block under `istftnet`.
    public static func from(_ config: ModelConfig) -> KokoroConfig? {
        guard let nToken = config.int("n_token"),
            let hidden = config.int("hidden_dim")
        else { return nil }
        let nMels = config.int("n_mels") ?? 80
        let sampleRate = config.int("sample_rate") ?? 24_000
        var nFFT = 20
        var hop = 5
        if let istft = config.nested("istftnet") {
            if let f = istft["gen_istft_n_fft"] as? Int { nFFT = f }
            if let h = istft["gen_istft_hop_size"] as? Int { hop = h }
        }
        return KokoroConfig(
            nToken: nToken, hidden: hidden, nMels: nMels,
            sampleRate: sampleRate, istftNFFT: nFFT,
            istftHop: hop)
    }
}

// ─── iSTFTNet vocoder ────────────────────────────────────────────────

/// The iSTFTNet vocoder tail — turns a predicted complex spectrogram
/// back into a time-domain waveform. This is the GPU-accelerated part
/// of Kokoro: `Ops.vocoderISTFT` wraps the fused iSTFT overlap-add
/// kernel.
public final class KokoroVocoder: @unchecked Sendable {
    /// FFT length of the iSTFT.
    public let nFFT: Int
    /// Hop length of the iSTFT.
    public let hopLength: Int
    /// Synthesis window `[nFFT]` (Hann). Built once at init.
    public let window: Tensor

    public init(
        nFFT: Int, hopLength: Int, dtype: DType = .f32,
        device: Device = .shared
    ) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        let win = MediaUtils.hannWindow(nFFT)
        let w = Tensor.empty(shape: [nFFT], dtype: dtype, device: device)
        MediaUtils.copyFloats(win, into: w)
        self.window = w
    }

    /// Reconstruct a waveform from a predicted STFT. `specRe` / `specIm`
    /// are `[nFrames, nFreq]` real / imaginary planes
    /// (`nFreq = nFFT/2 + 1`). Returns the `[outLen]` waveform with
    /// `outLen = (nFrames - 1) * hopLength + nFFT`.
    public func synthesize(
        specRe: Tensor, specIm: Tensor,
        device: Device = .shared
    ) -> Tensor {
        let nFreq = nFFT / 2 + 1
        precondition(
            specRe.shape.count == 2 && specRe.shape[1] == nFreq,
            "KokoroVocoder.synthesize: specRe must be [nFrames, nFreq]")
        let nFrames = specRe.shape[0]
        let cmd = device.makeCommandBuffer()
        let waveform = Ops.vocoderISTFT(
            specRe: specRe, specIm: specIm, window: window,
            nFrames: nFrames, nFFT: nFFT, hopLength: hopLength, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return waveform
    }
}

// ─── Kokoro model ────────────────────────────────────────────────────

public enum KokoroError: Error, CustomStringConvertible {
    /// The acoustic front-end (text encoder + prosody predictor +
    /// decoder) is not yet wired in this build — `synthesize(phonemes:)`
    /// needs it. `synthesizeFromSpectrogram` works regardless.
    case acousticFrontEndUnavailable
    /// `synthesizeStream(text:...)` was handed raw graphemes (plain
    /// English / unphonemized text) but Kokoro has no in-tree
    /// grapheme-to-phoneme (G2P) front-end. The acoustic model consumes
    /// *phonemes*, so the caller must phonemize the text first. Carries
    /// the offending text for diagnostics.
    case needsPhonemizedInput(text: String)

    public var description: String {
        switch self {
        case .acousticFrontEndUnavailable:
            return "Kokoro: the StyleTTS2 acoustic front-end is not wired "
                + "in this build; drive the vocoder via "
                + "synthesizeFromSpectrogram or supply a predicted "
                + "spectrogram from an external acoustic model"
        case .needsPhonemizedInput(let text):
            return "Kokoro: synthesizeStream(text:) received raw text "
                + "(\"\(text)\") but Kokoro synthesizes from PHONEMES and "
                + "FFAI ships no bundled grapheme-to-phoneme (G2P) front-end. "
                + "Phonemize the text to an IPA string first (e.g. via Misaki "
                + "G2P or espeak-ng), then call `phonemeIds(for:)` + "
                + "`synthesize(phonemeIds:)` / `synthesizeStream(phonemeSegments:)`, "
                + "or register a `Phonemizer` on `PhonemizerRegistry.shared`."
        }
    }
}

/// A loaded Kokoro TTS model. Owns the iSTFTNet vocoder tail (always
/// available — it is the FFAI-accelerated part) and, when the
/// checkpoint's acoustic stack is wired, the text encoder + prosody
/// predictor + decoder.
public final class KokoroModel: @unchecked Sendable {
    public let config: KokoroConfig
    public let vocoder: KokoroVocoder
    /// The phoneme→id table from the checkpoint's `config.json` `vocab`.
    public let phonemeVocab: [String: Int]
    /// Acoustic-stack weights (PLBERT + text/duration encoders + predictor +
    /// decoder + generator). `nil` when built config-only (vocoder tests).
    let acoustic: KokoroWeights?
    /// GPU weight cache for the iSTFTNet generator tail (uploads each conv
    /// once, lazily on first synthesis). `nil` when there's no acoustic stack.
    private lazy var gpuWeights: KokoroGPUWeights? = acoustic.map { KokoroGPUWeights($0) }
    /// Snapshot directory — used to resolve per-voice style packs lazily.
    let directory: URL?
    let dtype: DType
    /// Currently-active voice name (mutates via `setVoice(_:)`).
    /// Defaults to `Kokoro.defaultVoice` at init.
    public private(set) var currentVoice: String

    public init(
        config: KokoroConfig, vocoder: KokoroVocoder,
        phonemeVocab: [String: Int], dtype: DType,
        acoustic: KokoroWeights? = nil, directory: URL? = nil
    ) {
        self.config = config
        self.vocoder = vocoder
        self.phonemeVocab = phonemeVocab
        self.acoustic = acoustic
        self.directory = directory
        self.dtype = dtype
        self.currentVoice = KokoroModel.defaultVoice
    }

    // ─── Voice catalogue ─────────────────────────────────────────────

    /// Default voice — Kokoro v1.0's reference female "American Heart"
    /// voice. Picked because it's the most-cited demo voice in the
    /// Kokoro paper + community samples.
    public static let defaultVoice = "af_heart"

    /// Kokoro v1.0 voice catalogue. Prefix convention:
    ///   a* = American English   b* = British English
    ///   e* = Spanish            f* = French
    ///   h* = Hindi              i* = Italian
    ///   j* = Japanese           p* = Portuguese (BR)
    ///   z* = Mandarin Chinese
    /// Second letter: f = female, m = male.
    ///
    /// Voice vectors (256-dim style vectors per voice) are not
    /// bundled here — Kokoro resolves them from the HF cache on first
    /// `setVoice(_:)` once the acoustic stack ships.
    public static let availableVoices: [String] = [
        // American English
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
        // British English
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        // Other languages — abbreviated prefix only; full list grows
        // with Kokoro upstream voice packs.
        "ef_dora", "em_alex", "em_santa",
        "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola",
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
        "jm_kumo",
        "pf_dora", "pm_alex", "pm_santa",
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ]

    /// Activate `name` as the model's current voice. Validates against
    /// `availableVoices`; throws `AudioGenerationError.voiceNotAvailable`
    /// for unknown names. The actual style-vector load (from the HF
    /// cache) lands with the acoustic stack — this is the API surface.
    public func setVoice(_ name: String) throws {
        let resolved = (name == "default") ? KokoroModel.defaultVoice : name
        guard KokoroModel.availableVoices.contains(resolved) else {
            throw AudioGenerationError.voiceNotAvailable(
                requested: name, available: KokoroModel.availableVoices)
        }
        currentVoice = resolved
    }

    /// Synthesize a waveform from a predicted complex spectrogram.
    /// This is the GPU vocoder path — the FFAI-accelerated tail of
    /// Kokoro. `specRe` / `specIm` are `[nFrames, nFreq]`.
    public func synthesizeFromSpectrogram(
        specRe: Tensor, specIm: Tensor, device: Device = .shared
    ) -> Tensor {
        vocoder.synthesize(specRe: specRe, specIm: specIm, device: device)
    }

    /// Map a phoneme string into checkpoint token ids using the
    /// `config.json` `vocab` table. An external Misaki G2P produces the
    /// phoneme string; this is the id mapping Kokoro applies before the
    /// acoustic encoder. Unknown phonemes are dropped.
    public func phonemeIds(for phonemes: String) -> [Int] {
        phonemes.compactMap { phonemeVocab[String($0)] }
    }

    /// Full text→waveform synthesis. Requires the acoustic front-end;
    /// throws `KokoroError.acousticFrontEndUnavailable` when it is not
    /// wired (see the scope note at the top of this file).
    public func synthesize(
        phonemeIds: [Int],
        speed: Float = 1.0,
        device: Device = .shared
    ) throws -> Tensor {
        guard let acoustic, let dir = directory else {
            throw KokoroError.acousticFrontEndUnavailable
        }
        // Per-voice style pack: [N, 1, 256] indexed by phoneme count. Split
        // into the predictor (prosody, second 128) and decoder (acoustic,
        // first 128) halves.
        let voiceURL = dir.appendingPathComponent("voices/\(currentVoice).safetensors")
        // Load the SPECIFIC voice file — NOT the `voices/` directory. Every
        // voice pack stores its tensor under the same key "voice", so a
        // directory-level `SafeTensorsBundle` would index all ~50 packs and
        // resolve "voice" to whichever file sorts last (wrong speaker → wrong
        // style vector → wrong prosody). A single-file read is unambiguous.
        let vf = try SafeTensorsFile(url: voiceURL)
        // Kokoro voice packs store a single `[N, 1, 256]` tensor keyed "voice".
        let voice = ((try? vf.tensor(named: "voice"))?.toFloatArray()) ?? []
        guard voice.count >= 256 else { throw KokoroError.acousticFrontEndUnavailable }
        let row = min(max(phonemeIds.count - 1, 0), voice.count / 256 - 1)
        let refS = Array(voice[(row * 256) ..< (row * 256 + 256)])
        let sDec = Array(refS[0 ..< 128])
        let sPred = Array(refS[128 ..< 256])

        // StyleTTS2 wraps the phoneme ids with a leading/trailing pad token
        // (id 0) — the BOS/EOS the encoders + aligner were trained with.
        let ids = [0] + phonemeIds + [0]
        // GPU front-end (PLBERT → bert_encoder → text_encoder → predictor) when
        // available; the CPU path is the reference.
        let dEn: [[Float]]
        let asr: [[Float]]
        let dur: [Int], f0: [Float], n: [Float]
        if let gw = gpuWeights {
            dEn = acoustic.bertEncodeGPU(acoustic.plbertEncodeGPU(ids, gw: gw), gw: gw)
            asr = acoustic.textEncodeGPU(ids, gw: gw)
            (dur, f0, n, _) = acoustic.predictGPU(dEn: dEn, style: sPred, speed: speed, gw: gw)
        } else {
            dEn = acoustic.bertEncode(KokoroPLBert(w: acoustic).encode(ids))
            asr = acoustic.textEncode(ids)
            (dur, f0, n, _) = acoustic.predict(dEn: dEn, style: sPred, speed: speed)
        }
        // Length-regulate the acoustic features by the predicted durations.
        var asrAligned = [[Float]]()
        for (i, r) in asr.enumerated() { for _ in 0 ..< dur[min(i, dur.count - 1)] { asrAligned.append(r) } }
        let (re, im) = acoustic.decode(
            asr: asrAligned, f0: f0, n: n, style: sDec, vocoder: vocoder, gpu: gpuWeights)
        let nFreq = vocoder.nFFT / 2 + 1
        let frames = re.count
        var reFlat = [Float](repeating: 0, count: frames * nFreq)
        var imFlat = reFlat
        for t in 0 ..< frames {
            for k in 0 ..< nFreq { reFlat[t * nFreq + k] = re[t][k]; imFlat[t * nFreq + k] = im[t][k] }
        }
        let reT = Tensor.empty(shape: [frames, nFreq], dtype: .f32, device: device)
        reT.copyIn(from: reFlat)
        let imT = Tensor.empty(shape: [frames, nFreq], dtype: .f32, device: device)
        imT.copyIn(from: imFlat)
        return vocoder.synthesize(specRe: reT, specIm: imT, device: device)
    }

    /// Streaming text→audio synthesis — yields ONE waveform `Tensor` per
    /// phoneme segment (the TTS analogue of token-streaming: audio chunks
    /// instead of tokens). The caller segments the utterance (typically one
    /// segment per sentence / pause) so playback can start on segment 1 while
    /// segment 2 synthesizes. Concatenating the yielded chunks equals
    /// `synthesize` over the joined segments. Cancelling the consumer stops
    /// synthesis at the next segment boundary.
    public func synthesizeStream(
        phonemeSegments: [[Int]],
        speed: Float = 1.0,
        device: Device = .shared
    ) -> AsyncThrowingStream<Tensor, Error> {
        let model = self
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for segment in phonemeSegments {
                        try Task.checkCancellation()
                        if segment.isEmpty { continue }
                        let audio = try model.synthesize(
                            phonemeIds: segment, speed: speed, device: device)
                        continuation.yield(audio)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Streaming text→audio synthesis (the `StreamingSynthesizer`
    /// contract the `ffai speak` CLI drives).
    ///
    /// **Kokoro consumes phonemes, not graphemes**, and FFAI ships no
    /// bundled grapheme-to-phoneme (G2P) front-end. So `text` must already
    /// be a *phonemized IPA string* (the alphabet `config.json`'s `vocab`
    /// table publishes — see `phonemeIds(for:)`). The stream:
    ///
    ///   1. applies `voice` (when non-nil) via `setVoice(_:)`;
    ///   2. verifies `text` looks phonemized — if it's raw graphemes the
    ///      stream finishes with `KokoroError.needsPhonemizedInput`
    ///      (we do NOT fabricate a phonemizer and ship wrong audio);
    ///   3. splits the phoneme string into sentence/segment chunks and
    ///      reuses the per-segment `synthesize(phonemeIds:)` path, yielding
    ///      one waveform `Tensor` per segment.
    ///
    /// Concatenating the yielded chunks equals concatenating a
    /// `synthesize(phonemeIds:)` over each sentence segment — the stream
    /// drives the exact same per-segment acoustic path as the batch API,
    /// so streamed and buffered audio stay sample-identical per segment.
    ///
    /// - Parameters:
    ///   - text: A phonemized IPA string (NOT raw English). Plain text is
    ///           rejected with `needsPhonemizedInput`.
    ///   - voice: Voice name (e.g. `af_heart`); `nil` keeps `currentVoice`.
    ///   - speed: Synthesis speed multiplier.
    public func synthesizeStream(
        text: String,
        voice: String?,
        speed: Float
    ) -> AsyncThrowingStream<Tensor, Error> {
        let model = self
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    // Apply the requested voice before any synthesis so the
                    // whole utterance uses one consistent speaker.
                    if let voice { try model.setVoice(voice) }

                    // Kokoro needs phonemes. Reject raw graphemes loudly
                    // rather than mapping ASCII letters as if they were
                    // phonemes (which would ship garbage audio).
                    guard TTSPhonemeText.looksPhonemized(text) else {
                        throw KokoroError.needsPhonemizedInput(text: text)
                    }

                    // One segment per sentence so playback can start on the
                    // first sentence while later ones synthesize. Each
                    // segment is mapped to phoneme ids independently.
                    let segments = TTSPhonemeText.sentenceSegments(text)
                        .map { model.phonemeIds(for: $0) }
                    for segment in segments {
                        try Task.checkCancellation()
                        if segment.isEmpty { continue }
                        let audio = try model.synthesize(
                            phonemeIds: segment, speed: speed, device: .shared)
                        continuation.yield(audio)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

// ─── StreamingSynthesizer conformance ────────────────────────────────

// `synthesizeStream(text:voice:speed:)` above has the exact protocol
// shape; the text path requires phonemized input (Kokoro has no G2P) —
// see that method's doc comment. The model's output rate lives on its
// config, so surface it as `sampleRate` for the protocol.
extension KokoroModel: StreamingSynthesizer {
    public var sampleRate: Int { config.sampleRate }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension KokoroModel {
    public static let modelTypes: Set<String> = ["kokoro", "style_tts2"]

    /// Whether a decoded `config.json` describes a Kokoro checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        return config.has("istftnet") && config.has("n_token")
    }

    /// Load a Kokoro checkpoint from a resolved snapshot directory. The
    /// vocoder tail is always constructed; the acoustic stack is wired
    /// when present (see scope note).
    public static func load(directory: URL, device: Device = .shared)
        throws -> KokoroModel
    {
        let config = try ModelConfig.load(from: directory)
        guard let kc = KokoroConfig.from(config) else {
            throw ModelError.unsupportedModelType(
                "config.json is not a Kokoro config")
        }
        // The phoneme vocabulary lives inline in config.json.
        let phonemeVocab = (config.raw["vocab"] as? [String: Int]) ?? [:]
        let vocoder = KokoroVocoder(
            nFFT: kc.istftNFFT, hopLength: kc.istftHop, dtype: .f32, device: device)
        // Acoustic stack: read the model safetensors when present.
        let acoustic = (try? SafeTensorsBundle(directory: directory)).map { KokoroWeights(bundle: $0) }
        return KokoroModel(
            config: kc, vocoder: vocoder, phonemeVocab: phonemeVocab, dtype: .f32,
            acoustic: acoustic, directory: directory)
    }

    /// Assemble a `KokoroModel` from a decoded config. Factored out so
    /// tests can drive the vocoder path without a checkpoint.
    public static func build(
        config kc: KokoroConfig,
        phonemeVocab: [String: Int] = [:],
        dtype: DType = .f32,
        device: Device = .shared
    ) -> KokoroModel {
        let vocoder = KokoroVocoder(
            nFFT: kc.istftNFFT, hopLength: kc.istftHop,
            dtype: dtype, device: device)
        return KokoroModel(
            config: kc, vocoder: vocoder,
            phonemeVocab: phonemeVocab, dtype: dtype)
    }
}

// MARK: - Unified-engine modality conformance (ModelType/ModelEngine.swift)
extension KokoroModel: Speaking {}
