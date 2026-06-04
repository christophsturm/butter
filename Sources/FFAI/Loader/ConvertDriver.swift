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
// ConvertDriver — quantize a bf16/fp16 HuggingFace checkpoint to MLX
// 4-bit (or 8-bit / 2-bit) affine format using FFAI's own GPU kernels.
//
// Replaces `mlx_lm.convert` for the common case:
//   ConvertDriver.convert(sourceDir: src, destDir: dst, options: opts)
//
// Output is a safetensors directory with the same layout that FFAI's
// SafeTensorsBundle + quantized-weight loaders expect:
//   model.safetensors       — packed u32 weights + bf16/f16 scales/biases
//   config.json             — original config + "quantization" block added
//   tokenizer.json / *.txt  — copied unchanged
//
// ## Quantization selection
//
// A weight tensor is quantized when ALL of these hold:
//   1. It is 2D (Linear weight shape [out, in]).
//   2. `in` is divisible by `group_size` (64 — kernel invariant).
//   3. `in` is divisible by `pack_factor` (8 for int4, 4 for int8).
//   4. Its key ends in `.weight` but not `norm.weight` (LayerNorm /
//      RMSNorm weights are not quantized — they're tiny and sensitive).
//   5. It is not `embed_tokens.weight` unless `quantizeEmbeddings=true`.
//   6. It is not `lm_head.weight` unless `quantizeLmHead=true`.
//
// Everything else is copied through unchanged, preserving the original
// dtype and raw bytes.

import Foundation
import Metal

/// Options controlling quantization behaviour.
///
/// Each "role" in the checkpoint — main linear projections, token
/// embedding, lm_head, vision tower — can independently take any
/// `QuantSpec`. Affine `.bits(2 / 3 / 4 / 5 / 6 / 8)` triggers
/// `QuantizedOps.quantizeAffine` to produce a `(weight, scales,
/// biases)` triplet; `.fp16` / `.bf16` skip quantization and emit
/// the source weight cast to the target dtype.
///
/// `nil` on a role override means "fall through to the default" —
/// for embeddings / lm_head / vision tower the default is "leave
/// unchanged" (mlx-lm convention: skip the embedding and lm_head;
/// vision towers stay full-precision because FFAI's VL towers run
/// plain `Linear`, not `QuantizedLinear`). The main `bits` always
/// applies and has no nil case.
///
/// A single conversion can freely mix specs across roles — FFAI's
/// `loadLinear` / `loadEmbedding` derive the per-tensor bit-width
/// from the saved shapes via `deriveAffineQuantBits`, and the dtype
/// is part of every tensor's safetensors header, so mixed checkpoints
/// load correctly without per-tensor `config.json` entries.
/// Output layout for MoE expert projections.
public enum ExpertLayout: String, Sendable, CaseIterable {
    /// Gather per-expert weights into one `switch_mlp.<proj>` tensor of
    /// shape `[E, out, …]` (mlx-lm `QuantizedSwitchLinear` layout). Default.
    case stacked
    /// Keep the source per-expert `experts.<e>.<proj>` tensors as-is.
    case perExpert = "per-expert"
}

public struct ConvertOptions: Sendable {
    /// Spec for the main linear projections (attention + MLP — i.e.
    /// q/k/v/o, gate/up/down, MoE experts). Default `.bits(4)`.
    public var bits: QuantSpec = .bits(4)
    /// Group size — must be 64 (kernel invariant). Do not change.
    public var groupSize: Int = 64
    /// Dtype for tensors that aren't quantized AND don't carry an
    /// explicit downcast spec (norms, biases, conv1d kernels, RoPE
    /// tables). `nil` (default) preserves each tensor's source dtype.
    /// When the spec on a tensor is `.fp16` / `.bf16` that takes
    /// precedence over this fallback for that role's tensors.
    public var dtype: DType? = nil
    /// Spec for the token embedding table. `nil` (default) keeps it
    /// full-precision (mlx-lm convention).
    public var embeddingSpec: QuantSpec? = nil
    /// Spec for the `lm_head` projection. `nil` (default) keeps it
    /// full-precision; when `lm_head` is tied to the embedding the
    /// loader reuses the embedding triplet, so this knob only matters
    /// for untied heads (Qwen 3.6, some Gemma).
    public var lmHeadSpec: QuantSpec? = nil
    /// Spec for vision-tower weights. `nil` (default) keeps the tower
    /// full-precision — FFAI's VL towers (Qwen3-VL / Qwen3.5-VL,
    /// Pixtral, SigLIP, Idefics3, MiniCPM-V, FastVLM) use plain
    /// `Linear`, not `QuantizedLinear`, so a quantized tower would
    /// crash the loader. Set only when wiring a new VL tower that
    /// consumes `QuantizedLinear`.
    public var visionSpec: QuantSpec? = nil

    /// Spec for audio-tower / audio-projector weights (keys matching
    /// `audio_tower.` / `audio_embedder` / `embed_audio` / the shared
    /// `multimodal_embedder`). `nil` (default) keeps them full-precision —
    /// the same rationale as `visionSpec`: encoder-free projectors (e.g.
    /// Gemma 4 Omni's `embed_audio` 640→3840) are tiny and precision-
    /// sensitive (the sole audio pathway), so bf16 is the safe default.
    /// Set via `--audio-bits` to quantize them anyway.
    public var audioSpec: QuantSpec? = nil

    /// Spec for MoE router gates — the small `[num_experts, hidden]`
    /// projection that scores experts for top-k routing (keys ending in
    /// `.gate.weight` / `.router.weight`: `feed_forward.gate`,
    /// `block_sparse_moe.gate`, `mlp.gate`, …). `nil` (default) keeps
    /// the router full-precision: quantizing it risks flipping discrete
    /// top-k expert selections for a negligible memory saving (the gate
    /// is tiny — e.g. 32×2048 ≈ 64K params), which is the standard
    /// high-quality MoE-quant convention. Set (via `--router-bits`) to
    /// quantize routers anyway, e.g. to mirror an upstream conversion
    /// that quantized the gate (mlx-community ships some at 8-bit).
    public var routerSpec: QuantSpec? = nil

    /// How MoE expert projections are written. `.stacked` (default)
    /// gathers the per-expert `experts.<e>.{w1,w2,w3}` source tensors into
    /// one stacked `switch_mlp.{gate,up,down}_proj` tensor of shape
    /// `[E, out, …]` — mlx-lm's `QuantizedSwitchLinear` layout, which the
    /// loader dispatches through a single fused expert-indexed kernel
    /// (faster than 32 separate `Linear` calls). `.perExpert` preserves
    /// the source per-expert tensors. Non-MoE checkpoints ignore this.
    public var expertLayout: ExpertLayout = .stacked

    /// Spec for MoE expert projections (`switch_mlp.*` / `experts.<e>.*`).
    /// `nil` (default) inherits the main `bits`. Set to `.fp16` / `.bf16`
    /// to leave experts full-precision (the experts hold most of a MoE's
    /// weight mass, so this is the big size/quality dial), or to a
    /// different width to quantize them apart from the dense path.
    public var expertSpec: QuantSpec? = nil

    /// Original source identifier — HF repo id (`org/repo`) or local
    /// path — used to fill in the `base_model:` field of the emitted
    /// README.md model card and the human-readable conversion command.
    /// `nil` (default) suppresses the card entirely; set to the same
    /// value the caller would pass to `Model.load` so downstream users
    /// can trace provenance.
    public var sourceID: String? = nil

    /// Optional target HF repo id for the conversion (the `--upload-repo`
    /// value when used via the CLI). Surfaced in the emitted model card's
    /// example conversion command so a reader can reproduce the upload
    /// step. Has no side-effect on the conversion itself.
    public var uploadRepo: String? = nil

    public init() {}
}

/// Errors surfaced by ConvertDriver.
public enum ConvertDriverError: Error, CustomStringConvertible {
    case missingConfigJSON(URL)
    case configJSONMalformed(URL)
    case configJSONWriteFailed(URL, Error)
    case noSafetensorsFound(URL)
    case unsupportedBits(Int)
    case mkdirFailed(URL, Error)

    public var description: String {
        switch self {
        case .missingConfigJSON(let d):
            return "config.json missing in source dir: \(d.path)"
        case .configJSONMalformed(let u):
            return "config.json is not a JSON object: \(u.path)"
        case .configJSONWriteFailed(let u, let e):
            return "failed to write config.json to \(u.path): \(e)"
        case .noSafetensorsFound(let d):
            return "no .safetensors files found in: \(d.path)"
        case .unsupportedBits(let b):
            return "unsupported bits=\(b) — must be 2 / 3 / 4 / 5 / 6 / 8"
        case .mkdirFailed(let u, let e):
            return "failed to create output directory \(u.path): \(e)"
        }
    }
}

public enum ConvertDriver {

    // ─── Public entry point ──────────────────────────────────────────

    /// Quantize the checkpoint at `sourceDir` and write the result to
    /// `destDir`. `progress` is called with a human-readable status line
    /// for each tensor processed (useful for a CLI progress display).
    public static func convert(
        sourceDir: URL,
        destDir: URL,
        options: ConvertOptions = ConvertOptions(),
        progress: (@Sendable (String) -> Void)? = nil
    ) throws {
        // Validate every quantized spec up front so a bad bit-width
        // surfaces before we touch the GPU or open the writer. Downcast
        // specs (`.fp16` / `.bf16`) bypass quantization entirely so they
        // need no validation here.
        let allSpecs: [QuantSpec?] = [
            options.bits, options.embeddingSpec, options.lmHeadSpec, options.visionSpec,
            options.audioSpec, options.routerSpec, options.expertSpec,
        ]
        for spec in allSpecs {
            if case .bits(let b) = spec, !QuantSpec.supportedBits.contains(b) {
                throw ConvertDriverError.unsupportedBits(b)
            }
        }

        // Create the output directory (including parents).
        do {
            try FileManager.default.createDirectory(
                at: destDir, withIntermediateDirectories: true)
        } catch {
            throw ConvertDriverError.mkdirFailed(destDir, error)
        }

        // ─── Load source tensors ─────────────────────────────────────
        let bundle = try SafeTensorsBundle(directory: sourceDir, device: .shared)
        guard !bundle.files.isEmpty else {
            throw ConvertDriverError.noSafetensorsFound(sourceDir)
        }

        // ─── Prepare GPU command infrastructure ──────────────────────
        let device = Device.shared

        // ─── Build output writer ─────────────────────────────────────
        let outputURL = destDir.appendingPathComponent("model.safetensors")
        let writer = SafeTensorsWriter(url: outputURL)

        // Build the conversion plan (sorted source keys → output steps),
        // then execute it. The plan is the extensibility seam for future
        // targeted / per-layer / per-expert / ablation recipes (see
        // `planning/plan.md` § "Targeted & mixed quantization"): a richer
        // planner emits new `ConversionStep` cases; the executors below
        // stay the stable core.
        let allKeys = bundle.allKeys.sorted()
        let plan = buildPlan(keys: allKeys, options: options)
        for step in plan {
            switch step {
            case .single(let key):
                try executeSingle(
                    key: key, bundle: bundle, writer: writer,
                    options: options, device: device, progress: progress)
            case .stackExperts(let outBase, let expertKeys, let spec):
                try executeStackExperts(
                    outBase: outBase, expertWeightKeys: expertKeys, spec: spec,
                    bundle: bundle, writer: writer, options: options,
                    device: device, progress: progress)
            }
        }

        try writer.finalize()
        progress?("wrote \(outputURL.lastPathComponent)")

        // ─── Write updated config.json ───────────────────────────────
        try writeConfig(sourceDir: sourceDir, destDir: destDir, options: options)

        // ─── Copy tokenizer + auxiliary files ────────────────────────
        copyAuxiliaryFiles(from: sourceDir, to: destDir, progress: progress)

        // ─── Emit README.md (HF model card) ──────────────────────────
        // Only when the caller supplied `sourceID` — without it the
        // card would be missing the most important piece (the base
        // model link), so it's better to skip than to publish a
        // partial card.
        if let sourceID = options.sourceID {
            try writeModelCard(
                destDir: destDir, sourceID: sourceID, options: options,
                progress: progress)
        }
    }

    // ─── Quantization eligibility ────────────────────────────────────

    /// Pick the `QuantSpec` to use for `key`. `nil` means "pass this
    /// tensor through unchanged in its source dtype" — every norm /
    /// bias / 1D tensor lands here, plus any role whose spec is `nil`.
    ///
    /// Routing:
    ///   * `embed_tokens.weight` / `embeddings.weight` → `embeddingSpec`
    ///   * `lm_head.weight`                            → `lmHeadSpec`
    ///   * vision-tower keys (matches `.visual.` / `visual.` /
    ///     `vision_tower.` / `vision_model.` prefixes used across
    ///     Qwen3-VL / Qwen3.5-VL / Pixtral / SigLIP / Idefics3 /
    ///     MiniCPM-V / FastVLM, plus Gemma 4 Omni's encoder-free
    ///     `vision_embedder.*` / `embed_vision`)       → `visionSpec`
    ///   * audio-tower keys (`audio_tower.` / `audio_embedder` /
    ///     `embed_audio`)                              → `audioSpec`
    ///   * everything else (attention + MLP)           → `bits`
    ///
    /// For a `.bits(N)` spec we further check that the tensor is 2D,
    /// ends in `.weight`, and its inner dim is divisible by both
    /// `group_size` and the chosen bit-width's storage stride — if not,
    /// the tensor falls through to pass-through (`nil`) rather than
    /// crashing the quantize kernel. `.fp16` / `.bf16` downcast applies
    /// to any tensor that the role rule matches, regardless of shape.
    /// True when `key` names a MoE router/gate projection — the small
    /// `[num_experts, hidden]` weight that scores experts for top-k
    /// routing. Matches the exact `.gate.weight` / `.router.weight`
    /// suffixes (`feed_forward.gate` in LFM2 / DeepSeek,
    /// `block_sparse_moe.gate` in Mixtral, `mlp.gate` in Qwen MoE) and
    /// deliberately does NOT match dense-MLP projections like
    /// `.gate_proj.weight` / `.gate_up_proj.weight` (those end in
    /// `proj.weight`, not `.gate.weight`).
    static func isMoERouterKey(_ key: String) -> Bool {
        key.hasSuffix(".gate.weight") || key.hasSuffix(".router.weight")
    }

    // ─── Conversion plan ─────────────────────────────────────────────

    /// One output unit of a conversion. `buildPlan` turns the source keys
    /// + options into a `[ConversionStep]`; the executors below realise it.
    /// This is the extensibility seam for future targeted / per-layer /
    /// per-expert / ablation recipes (planning/plan.md § "Targeted & mixed
    /// quantization"): a richer planner emits new cases without disturbing
    /// the execution core.
    enum ConversionStep {
        /// Quantize / downcast / pass-through one source tensor → same key.
        case single(key: String)
        /// Gather `expertWeightKeys` (one `.weight` per expert, in expert
        /// order) for one (layer, projection) into a single stacked
        /// `switch_mlp.<proj>` tensor under `outBase`. `spec` is the
        /// resolved expert spec (`expertSpec ?? bits`).
        case stackExperts(outBase: String, expertWeightKeys: [String], spec: QuantSpec)
    }

    /// Parse a per-expert MoE weight key
    /// `<prefix>.experts.<e>.<wName>.weight` → the stacked output base
    /// `<prefix>.switch_mlp.<proj>` + the expert index. `wName` ∈
    /// {`w1`/`w3`/`w2`, `gate_proj`/`up_proj`/`down_proj`} maps to the
    /// stacked `gate_proj` / `up_proj` / `down_proj` the loader's
    /// `sliceStackedExperts` expects. Returns nil for any non-expert key.
    static func parseExpertWeightKey(_ key: String) -> (stackedBase: String, expert: Int)? {
        guard key.hasSuffix(".weight"), let r = key.range(of: ".experts.") else {
            return nil
        }
        let prefix = String(key[..<r.lowerBound])  // <...>.feed_forward
        let rest = String(key[r.upperBound...])  // <e>.<wName>.weight
        let parts = rest.split(separator: ".")
        guard parts.count == 3, let e = Int(parts[0]) else { return nil }
        let proj: String
        switch parts[1] {
        case "w1", "gate_proj": proj = "gate_proj"
        case "w3", "up_proj": proj = "up_proj"
        case "w2", "down_proj": proj = "down_proj"
        default: return nil
        }
        return ("\(prefix).switch_mlp.\(proj)", e)
    }

    /// Build the output plan from sorted source keys. In `.stacked` layout,
    /// per-expert weights are grouped by `(layer, projection)` into one
    /// `stackExperts` step each and everything else is a `single`. In
    /// `.perExpert` layout every tensor is a `single` (experts pick up
    /// `expertSpec` through `effectiveSpec`).
    static func buildPlan(keys: [String], options: ConvertOptions) -> [ConversionStep] {
        guard options.expertLayout == .stacked else {
            return keys.map { .single(key: $0) }
        }
        var groups: [String: [(Int, String)]] = [:]
        var consumed = Set<String>()
        for key in keys {
            if let (base, e) = parseExpertWeightKey(key) {
                groups[base, default: []].append((e, key))
                consumed.insert(key)
            }
        }
        var steps: [ConversionStep] =
            keys.filter { !consumed.contains($0) }.map { .single(key: $0) }
        let spec = options.expertSpec ?? options.bits
        for base in groups.keys.sorted() {
            let ordered = groups[base]!.sorted { $0.0 < $1.0 }.map { $0.1 }
            steps.append(
                .stackExperts(outBase: base, expertWeightKeys: ordered, spec: spec))
        }
        return steps
    }

    // ─── Step executors ──────────────────────────────────────────────

    /// Quantize / downcast / pass-through a single tensor — the original
    /// per-tensor path, lifted out of the main loop.
    private static func executeSingle(
        key: String, bundle: SafeTensorsBundle, writer: SafeTensorsWriter,
        options: ConvertOptions, device: Device,
        progress: (@Sendable (String) -> Void)?
    ) throws {
        let entry = try bundle.tensor(named: key)
        switch effectiveSpec(key: key, tensor: entry, options: options) {
        case .bits(let bits)?:
            // Per-tensor bit-width — may differ from `options.bits` when an
            // embedding / lm_head / vision / expert override applied.
            progress?("quantizing \(key) \(entry.shape) @ \(bits)bit")
            let (packedBytes, scalesBytes, biasesBytes) = try quantizeTensor(
                entry, key: key, options: options, bits: bits, device: device)
            let base = String(key.dropLast(".weight".count))
            let weightShape = packedShape(
                original: entry.shape, numel: entry.elementCount, bits: bits)
            let groupShape = scalesShape(original: entry.shape, groupSize: options.groupSize)
            try writer.append(
                name: "\(base).weight", dtype: .u32, shape: weightShape, bytes: packedBytes)
            try writer.append(
                name: "\(base).scales", dtype: entry.dtype, shape: groupShape, bytes: scalesBytes)
            try writer.append(
                name: "\(base).biases", dtype: entry.dtype, shape: groupShape, bytes: biasesBytes)

        case .fp16?, .bf16?:
            let targetDtype = effectiveSpec(key: key, tensor: entry, options: options)!
                .downcastDtype!
            progress?("downcasting \(key) \(entry.shape) → \(targetDtype.rawValue)")
            let bytes = downcastBytes(from: entry, to: targetDtype)
            try writer.append(name: key, dtype: targetDtype, shape: entry.shape, bytes: bytes)

        case nil:
            progress?("copying   \(key) \(entry.shape)")
            let bytes = rawBytes(from: entry)
            try writer.append(name: key, dtype: entry.dtype, shape: entry.shape, bytes: bytes)
        }
    }

    /// Gather per-expert weights and emit one stacked `switch_mlp.<proj>`
    /// tensor. Each expert is quantized/downcast independently, then the
    /// per-expert byte blobs are concatenated along a new leading expert
    /// axis — the row-major `[E, …]` layout `sliceStackedExperts` slices.
    /// Quantized (`spec == .bits`) → `[E, out, packed]` u32 + `[E, out,
    /// groups]` scales/biases. Downcast/raw (`.fp16`/`.bf16`) → `[E, out,
    /// in]` weight only.
    private static func executeStackExperts(
        outBase: String, expertWeightKeys: [String], spec: QuantSpec,
        bundle: SafeTensorsBundle, writer: SafeTensorsWriter,
        options: ConvertOptions, device: Device,
        progress: (@Sendable (String) -> Void)?
    ) throws {
        let numExperts = expertWeightKeys.count
        let first = try bundle.tensor(named: expertWeightKeys[0])
        precondition(first.shape.count == 2, "stackExperts: expert weight must be 2D")
        let perExpertShape = first.shape  // [out, in]

        switch spec {
        case .bits(let bits):
            progress?(
                "stacking+quantizing \(outBase) "
                    + "[\(numExperts), \(perExpertShape[0]), \(perExpertShape[1])] @ \(bits)bit")
            var wData = Data(), sData = Data(), bData = Data()
            for k in expertWeightKeys {
                let t = try bundle.tensor(named: k)
                let (p, s, b) = try quantizeTensor(
                    t, key: k, options: options, bits: bits, device: device)
                wData.append(p)
                sData.append(s)
                bData.append(b)
            }
            let pe = packedShape(
                original: perExpertShape, numel: first.elementCount, bits: bits)  // [out, pCols]
            let ge = scalesShape(original: perExpertShape, groupSize: options.groupSize)  // [out, gCols]
            try writer.append(
                name: "\(outBase).weight", dtype: .u32, shape: [numExperts] + pe, bytes: wData)
            try writer.append(
                name: "\(outBase).scales", dtype: first.dtype, shape: [numExperts] + ge,
                bytes: sData)
            try writer.append(
                name: "\(outBase).biases", dtype: first.dtype, shape: [numExperts] + ge,
                bytes: bData)

        case .fp16, .bf16:
            let target = spec.downcastDtype!
            progress?(
                "stacking \(outBase) "
                    + "[\(numExperts), \(perExpertShape[0]), \(perExpertShape[1])] → \(target.rawValue)")
            var wData = Data()
            for k in expertWeightKeys {
                wData.append(downcastBytes(from: try bundle.tensor(named: k), to: target))
            }
            try writer.append(
                name: "\(outBase).weight", dtype: target,
                shape: [numExperts] + perExpertShape, bytes: wData)
        }
    }

    private static func effectiveSpec(
        key: String, tensor: Tensor, options: ConvertOptions
    ) -> QuantSpec? {
        // Norms always pass through (never quantized; never downcast
        // implicitly — the kernel-side RMSNorm needs the original
        // precision for stability).
        if key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") {
            return nil
        }

        // Pick the role-specific spec.
        let roleSpec: QuantSpec?
        if key.contains("embed_tokens.weight") || key.contains("embeddings.weight") {
            roleSpec = options.embeddingSpec
        } else if key.contains("lm_head.weight") {
            roleSpec = options.lmHeadSpec
        } else if key.contains(".visual.") || key.hasPrefix("visual.")
            || key.contains("vision_tower.") || key.contains("vision_model.")
            || key.contains("vision_embedder") || key.contains("embed_vision")
            || key.contains("patch_embedder")
        {
            // Vision-tower / encoder-free vision projector (Gemma 4 Omni
            // `vision_embedder.*` + `embed_vision`). Defaults to bf16.
            roleSpec = options.visionSpec
        } else if key.contains("audio_tower.") || key.contains("audio_embedder")
            || key.contains("embed_audio")
        {
            // Audio-tower / encoder-free audio projector (Gemma 4 Omni
            // `embed_audio`). Defaults to bf16; falls back to the vision
            // knob when `--audio-bits` is unset so one flag can cover both.
            roleSpec = options.audioSpec ?? options.visionSpec
        } else if isMoERouterKey(key) {
            // MoE router gates default to full-precision (`routerSpec`
            // nil → pass-through); the `--router-bits` flag opts in.
            roleSpec = options.routerSpec
        } else if parseExpertWeightKey(key) != nil {
            // Per-expert MoE weight (reached only in `.perExpert` layout —
            // in `.stacked` layout these are consumed by `stackExperts` and
            // never hit `effectiveSpec`). `expertSpec` nil inherits `bits`.
            roleSpec = options.expertSpec ?? options.bits
        } else {
            // The main `bits` is always non-nil; wrap as Optional for
            // uniform handling below.
            roleSpec = options.bits
        }
        guard let spec = roleSpec else { return nil }

        switch spec {
        case .bits(let bits):
            // Only 2D `.weight` tensors are eligible for affine quant.
            guard tensor.shape.count == 2, key.hasSuffix(".weight") else { return nil }
            // The bit-stream packing requires `inDim * bits` to be a
            // multiple of 32 AND `inDim` to be a multiple of group_size.
            let inDim = tensor.shape[1]
            guard inDim.isMultiple(of: options.groupSize) else { return nil }
            guard (inDim * bits) % 32 == 0 else { return nil }
            return .bits(bits)

        case .fp16, .bf16:
            // Skip the source-already-target no-op. If the tensor's
            // source dtype already matches the requested downcast,
            // emit it unchanged through the pass-through path.
            if tensor.dtype == spec.downcastDtype { return nil }
            return spec
        }
    }

    // ─── Per-tensor quantization ─────────────────────────────────────

    /// GPU-quantize a single 2D weight tensor to `bits` bits per code.
    /// Returns raw bytes for the packed weight, scales, and biases
    /// buffers — ready for the writer.
    ///
    /// The kernel operates on a flat `[numel]` view of the weight; the
    /// original `[out, in]` shape is preserved at the caller level via
    /// `packedShape` / `scalesShape`. `bits` here may differ from
    /// `options.bits.bits` when an embedding / lm_head / vision override
    /// applied — every per-tensor knob runs through this same path,
    /// just with a different `bits`. Storage size is computed via
    /// `QuantizedOpsValidation.packedUInt32Count`, which generalises
    /// `numel / packFactor` to the odd-width packings 3 / 5 / 6.
    private static func quantizeTensor(
        _ src: Tensor, key: String,
        options: ConvertOptions, bits: Int,
        device: Device
    ) throws -> (packed: Data, scales: Data, biases: Data) {
        let numel = src.elementCount
        let nGroups = numel / options.groupSize
        guard let nPacks = QuantizedOpsValidation.packedUInt32Count(numel: numel, bits: bits)
        else {
            throw ConvertDriverError.unsupportedBits(bits)
        }

        // Allocate output buffers.
        let packed = Tensor.empty(shape: [nPacks], dtype: .u32, device: device)
        let scales = Tensor.empty(shape: [nGroups], dtype: src.dtype, device: device)
        let biases = Tensor.empty(shape: [nGroups], dtype: src.dtype, device: device)
        packed.zero()
        scales.zero()
        biases.zero()

        // Reshape src to flat [numel] so the kernel sees one contiguous row.
        let flat = src.reshaped(to: [numel])

        // Encode + commit + wait for the GPU to finish.
        let cb = device.makeCommandBuffer()
        QuantizedOps.quantizeAffine(
            weight: flat,
            packed: packed, scales: scales, biases: biases,
            bits: bits, groupSize: options.groupSize,
            on: cb)
        cb.commit()
        cb.waitUntilCompleted()

        return (
            rawBytes(from: packed),
            rawBytes(from: scales),
            rawBytes(from: biases)
        )
    }

    // ─── Shape helpers ───────────────────────────────────────────────

    /// Shape of the packed u32 tensor given the original `[out, in]`
    /// shape and the chosen `bits`. The pack collapses the trailing dim
    /// down to its `(numel * bits / 32) / out` width — which equals
    /// `in / packFactor` for the clean widths 2 / 4 / 8 and the
    /// `in * bits / 32` byte-stream stride for 3 / 5 / 6.
    private static func packedShape(original: [Int], numel: Int, bits: Int) -> [Int] {
        var s = original
        let outerCount = numel / s[s.count - 1]
        let packCount = QuantizedOpsValidation.packedUInt32Count(numel: numel, bits: bits) ?? 0
        s[s.count - 1] = packCount / outerCount
        return s
    }

    /// Shape of the scales / biases tensor: [out, in / group_size].
    private static func scalesShape(original: [Int], groupSize: Int) -> [Int] {
        var s = original
        s[s.count - 1] = s[s.count - 1] / groupSize
        return s
    }

    // ─── Raw byte extraction ─────────────────────────────────────────

    /// Copy a tensor's bytes out of its MTLBuffer into a Data value.
    /// The buffer uses shared storage, so this is a plain memcpy — no GPU
    /// sync needed (we've already waited on the command buffer above).
    private static func rawBytes(from tensor: Tensor) -> Data {
        let ptr = tensor.buffer.contents().advanced(by: tensor.offset)
        return Data(bytes: ptr, count: tensor.byteCount)
    }

    /// Convert a tensor's bytes to the requested floating-point dtype
    /// (`.fp16` / `.bf16` / `.f32`). Used when a role explicitly opts
    /// into `.fp16` / `.bf16` downcast via its `QuantSpec`. CPU-side
    /// conversion through the canonical `Tensor.toFloatArray()` path —
    /// fast enough for offline conversion (no GPU dispatch overhead
    /// per-tensor) and avoids needing a dedicated `Ops.cast` kernel
    /// for every dtype pair.
    private static func downcastBytes(from src: Tensor, to target: DType) -> Data {
        // Source-already-target shouldn't reach here (see `effectiveSpec`),
        // but handle it gracefully if it does.
        if src.dtype == target {
            return rawBytes(from: src)
        }
        let floats = src.toFloatArray()
        switch target {
        case .f32:
            return floats.withUnsafeBufferPointer { Data(buffer: $0) }
        case .f16:
            let halves = floats.map { Float16($0) }
            return halves.withUnsafeBufferPointer { Data(buffer: $0) }
        case .bf16:
            let bits = floats.map { v -> UInt16 in
                let b = v.bitPattern
                let rounded = b &+ 0x7FFF &+ ((b >> 16) & 1)
                return UInt16(rounded >> 16)
            }
            return bits.withUnsafeBufferPointer { Data(buffer: $0) }
        default:
            fatalError("ConvertDriver.downcastBytes: unsupported target dtype \(target)")
        }
    }

    // ─── config.json ─────────────────────────────────────────────────

    /// Read config.json from source, inject quantization blocks, write to
    /// dest. Both `quantization` and `quantization_config` are written for
    /// compatibility: mlx-lm checks `quantization`, Transformers checks
    /// `quantization_config`.
    private static func writeConfig(
        sourceDir: URL, destDir: URL, options: ConvertOptions
    ) throws {
        let srcURL = sourceDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: srcURL.path) else {
            throw ConvertDriverError.missingConfigJSON(sourceDir)
        }

        let data = try Data(contentsOf: srcURL)
        let parsed: Any
        if #available(macOS 12.0, iOS 15.0, *) {
            parsed = try JSONSerialization.jsonObject(with: data, options: [.json5Allowed])
        } else {
            parsed = try JSONSerialization.jsonObject(with: data)
        }
        guard var obj = parsed as? [String: Any] else {
            throw ConvertDriverError.configJSONMalformed(srcURL)
        }

        // Write the quantization block only when something is actually
        // affine-quantized. A pure downcast (e.g. `--bits fp16`)
        // produces no `(weight, scales, biases)` triplets, so leaving
        // the block out keeps the output a plain-precision checkpoint
        // that loaders treat as bf16/fp16 by default.
        //
        // For mixed-spec conversions (e.g. `--bits 4 --embedding-bits 8`)
        // the top-level `quantization.bits` records the MAIN spec's
        // bit-width; per-tensor widths are recovered at load time from
        // the saved shapes via `deriveAffineQuantBits`. The
        // `quantization_config` mirror is for Transformers compat.
        if let mainBits = options.bits.bits {
            let quantBlock: [String: Any] = [
                "bits": mainBits,
                "group_size": options.groupSize,
                "mode": "affine",
            ]
            obj["quantization"] = quantBlock
            obj["quantization_config"] = quantBlock
        } else {
            // Pure downcast conversion — make sure we don't leave a
            // stale quantization block from the source config.
            obj.removeValue(forKey: "quantization")
            obj.removeValue(forKey: "quantization_config")
        }

        // Sanitize before NSJSONSerialization: Python's `json` module emits
        // `Infinity` / `-Infinity` / `NaN` literals when `allow_nan=True`
        // (the default), and several upstream configs use them
        // (e.g. NemotronH's `time_step_limit: [0.0, Infinity]`).
        // `JSONSerialization` is strict JSON — it throws on non-finite
        // doubles. Replace each with a JSON-legal sentinel that the
        // model code won't trip on (`1e308` ≈ DBL_MAX; NaN → null).
        let sanitized = sanitizeForJSON(obj)

        let outData = try JSONSerialization.data(
            withJSONObject: sanitized, options: [.sortedKeys, .prettyPrinted])
        let destURL = destDir.appendingPathComponent("config.json")
        do {
            try outData.write(to: destURL, options: .atomic)
        } catch {
            throw ConvertDriverError.configJSONWriteFailed(destURL, error)
        }
    }

    /// Walk a JSON-shaped tree and replace non-finite `Double` / `Float`
    /// values with JSON-legal stand-ins. `+Inf` → `1e308`, `-Inf` →
    /// `-1e308`, `NaN` → `NSNull`. NSJSONSerialization rejects non-finite
    /// numbers; Python's `json` module writes them as `Infinity` / `NaN`
    /// literals (a non-spec extension), which is what upstream configs
    /// like NemotronH ship.
    private static func sanitizeForJSON(_ value: Any) -> Any {
        if let n = value as? Double {
            if n.isNaN { return NSNull() }
            if n == .infinity { return 1e308 }
            if n == -.infinity { return -1e308 }
            return n
        }
        if let n = value as? Float {
            if n.isNaN { return NSNull() }
            if n == .infinity { return Float(1e30) }
            if n == -.infinity { return Float(-1e30) }
            return n
        }
        if let dict = value as? [String: Any] {
            return dict.mapValues(sanitizeForJSON)
        }
        if let arr = value as? [Any] {
            return arr.map(sanitizeForJSON)
        }
        return value
    }

    // ─── Auxiliary file copy ─────────────────────────────────────────

    /// Copy tokenizer and other non-weight files that must travel with the
    /// checkpoint. Missing files are silently skipped (not every model has
    /// every optional file).
    private static func copyAuxiliaryFiles(
        from sourceDir: URL, to destDir: URL,
        progress: (@Sendable (String) -> Void)?
    ) {
        // Explicit-name files to try copying.
        let namedFiles = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "tokenizer.model",
            "vocab.txt",
            "merges.txt",
        ]

        let fm = FileManager.default
        for name in namedFiles {
            let src = sourceDir.appendingPathComponent(name)
            // fileExists follows symlinks, so this correctly handles HF
            // cache blobs that are symlinked from the snapshot directory.
            guard fm.fileExists(atPath: src.path) else { continue }
            let dst = destDir.appendingPathComponent(name)
            // Overwrite if already exists (e.g. re-running the convert).
            try? fm.removeItem(at: dst)
            do {
                // HF snapshot directories present files as relative
                // symlinks pointing into the blobs store
                // (e.g. `../../blobs/<sha>`). `copyItem` would copy the
                // symlink, not the target, so the destination would be a
                // broken relative symlink. Resolve to the real blob path
                // first so we copy the actual bytes.
                let resolved = src.resolvingSymlinksInPath()
                try fm.copyItem(at: resolved, to: dst)
                progress?("copied    \(name)")
            } catch {
                // Non-fatal — log but continue so the tokenizer-less
                // model weight file is still written.
                progress?("warning: could not copy \(name): \(error)")
            }
        }

        // Also copy any remaining *.txt and *.json files that are not
        // safetensors-adjacent (e.g. generation_config.json).
        let copyExtensions: Set<String> = ["txt", "json"]
        if let contents = try? fm.contentsOfDirectory(
            at: sourceDir, includingPropertiesForKeys: nil)
        {
            for file in contents where copyExtensions.contains(file.pathExtension.lowercased()) {
                let name = file.lastPathComponent
                // Skip files we already handled above, and config.json
                // (written separately with quantization blocks added).
                if namedFiles.contains(name) || name == "config.json" { continue }
                // Skip safetensors index files.
                if name.hasPrefix("model.safetensors") { continue }
                let dst = destDir.appendingPathComponent(name)
                try? fm.removeItem(at: dst)
                do {
                    // Resolve symlinks for the same reason as above.
                    let resolved = file.resolvingSymlinksInPath()
                    try fm.copyItem(at: resolved, to: dst)
                    progress?("copied    \(name)")
                } catch {
                    progress?("warning: could not copy \(name): \(error)")
                }
            }
        }
    }

    // ─── Model card (README.md) ──────────────────────────────────────

    /// Write a minimal HF-style README.md to `destDir`, describing the
    /// conversion: YAML frontmatter (license / base_model / language /
    /// tags), title, one-line provenance summary linking back to the
    /// source repo and to FFAI, an example `ffai convert` command the
    /// reader can copy verbatim, and the FFAI version that produced
    /// this artifact. Mirrors the format of
    /// https://huggingface.co/ekryski/Qwen3.5-0.8B-2bit.
    ///
    /// The card is best-effort — a failure to write doesn't fail the
    /// conversion (the model artifacts on disk are still valid). We
    /// surface a warning through `progress` so the user sees it.
    private static func writeModelCard(
        destDir: URL, sourceID: String, options: ConvertOptions,
        progress: (@Sendable (String) -> Void)?
    ) throws {
        let cardURL = destDir.appendingPathComponent("README.md")
        let card = renderModelCard(sourceID: sourceID, options: options)
        do {
            try card.write(to: cardURL, atomically: true, encoding: .utf8)
            progress?("wrote     README.md")
        } catch {
            // Convert artifacts on disk are still fine — surface and continue.
            progress?("warning: could not write README.md: \(error)")
        }
    }

    /// Build the README.md body. Pure-function — pulled out of
    /// `writeModelCard` so it's covered by unit tests without touching
    /// the filesystem AND so the `ffai card` CLI surface can render
    /// the same body for a repo that's already on HF.
    public static func renderModelCard(
        sourceID: String, options: ConvertOptions
    ) -> String {
        let isHFRepo = sourceID.contains("/") && !sourceID.hasPrefix("/")
        let sourceLink =
            isHFRepo
            ? "[\(sourceID)](https://huggingface.co/\(sourceID))"
            : "`\(sourceID)`"

        // Display title — prefer the upload-repo's last segment (matches
        // what HF shows in the page header); fall back to the source's
        // last segment.
        let title: String = {
            if let upload = options.uploadRepo,
                let last = upload.split(separator: "/").last
            {
                return String(last)
            }
            if let last = sourceID.split(separator: "/").last {
                return String(last)
            }
            return "Converted model"
        }()

        // Tags: always mlx + ffai; add quantization-mode tags so HF
        // filters surface this checkpoint under the right buckets.
        var tags = ["mlx", "ffai"]
        switch options.bits {
        case .bits(let n):
            tags.append("quantized")
            tags.append("\(n)bit")
            tags.append("affine")
        case .fp16:
            tags.append("fp16")
        case .bf16:
            tags.append("bf16")
        }

        // Per-role overrides also show up as tags so a reader can spot
        // mixed-precision conversions at a glance.
        if let e = options.embeddingSpec, e != options.bits {
            tags.append("embed-\(e.label)")
        }
        if let h = options.lmHeadSpec, h != options.bits {
            tags.append("lmhead-\(h.label)")
        }
        if let v = options.visionSpec, v != options.bits {
            tags.append("vision-\(v.label)")
        }
        if let a = options.audioSpec, a != options.bits {
            tags.append("audio-\(a.label)")
        }

        let tagsBlock = tags.map { "  - \($0)" }.joined(separator: "\n")

        // Human-readable spec description for the lead sentence.
        let specDesc: String
        switch options.bits {
        case .bits(let n): specDesc = "\(n)-bit affine quantization"
        case .fp16: specDesc = "fp16 downcast"
        case .bf16: specDesc = "bf16 downcast"
        }

        // Example `ffai convert` command — uses the same source id +
        // every non-default spec the conversion actually used, plus the
        // upload-repo when set. `ffai convert` accepts both `--bits 4`
        // and `--bits fp16` so the same field reconstructs both forms.
        var cmd = "ffai convert \(sourceID) --bits \(cliBitsValue(options.bits))"
        if let e = options.embeddingSpec {
            cmd += " --embedding-bits \(cliBitsValue(e))"
        }
        if let h = options.lmHeadSpec {
            cmd += " --lm-head-bits \(cliBitsValue(h))"
        }
        if let v = options.visionSpec {
            cmd += " --vision-bits \(cliBitsValue(v))"
        }
        if let a = options.audioSpec {
            cmd += " --audio-bits \(cliBitsValue(a))"
        }
        if let r = options.uploadRepo {
            cmd += " \\\n    --upload-repo \(r)"
        }

        // FFAI version — pinned in `Sources/FFAI/FFAI.swift`. Reading
        // it through the public `FFAI.version` constant keeps the card
        // and the `ffai --version` output in sync.
        let ffaiVersion = FFAI.version

        return """
            ---
            license: apache-2.0
            base_model: \(sourceID)
            language:
              - en
            tags:
            \(tagsBlock)
            ---

            # \(title)

            \(specDesc) of \(sourceLink), produced with [FFAI](https://github.com/thewafflehaus/FFAI) \(ffaiVersion)'s `ffai convert` (mlx-affine format, `group_size=64`).

            ## Conversion

            ```bash
            \(cmd)
            ```

            ## See also

            - [FFAI](https://github.com/thewafflehaus/FFAI) — fast Apple Silicon LLM inference. `Model.load("\(options.uploadRepo ?? sourceID)")` runs this checkpoint end-to-end.
            - [FFAI quickstart](https://github.com/thewafflehaus/FFAI/blob/main/documentation/quickstart.md)
            - [FFAI quantization docs](https://github.com/thewafflehaus/FFAI/blob/main/documentation/quantization.md)
            """
    }

    /// Render a `QuantSpec` as the string the `--bits` flag accepts on
    /// the CLI side: `"4"` / `"fp16"` / `"bf16"`. This differs from
    /// `QuantSpec.label` (`"4bit"` / `"fp16"` / `"bf16"`) — the label
    /// is the log-line rendering; this is the CLI argument rendering.
    private static func cliBitsValue(_ spec: QuantSpec) -> String {
        switch spec {
        case .bits(let n): return "\(n)"
        case .fp16: return "fp16"
        case .bf16: return "bf16"
        }
    }
}
