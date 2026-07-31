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
// LanguageModel — common surface every text-generating model conforms
// to. Lets Generate.swift and the CLI work against any model family
// without knowing the concrete type.

import Foundation
import Metal

public protocol LanguageModel: Module {
    var hidden: Int { get }
    var nLayers: Int { get }
    var nHeads: Int { get }
    var nKVHeads: Int { get }
    var headDim: Int { get }
    var vocab: Int { get }
    var maxSeq: Int { get }
    var dtype: DType { get }

    /// Default prefill chunk size — the number of prompt tokens this
    /// family's `forwardMulti(...)` consumes per dispatch when the
    /// caller doesn't override `GenerationParameters.prefillStepSize`.
    /// Tuned per family in mlx-swift-lm; defaults to 1024 (the
    /// generic dense Llama/Mistral/Phi/Qwen3 value). Larger values
    /// trade peak memory (the [N, hidden] activation chunk) for
    /// fewer dispatches; smaller values keep memory flatter at the
    /// cost of more commit cycles.
    ///
    /// Currently-tuned overrides:
    /// - GPT-OSS 20B: 2048
    /// - Qwen 3.5 MoE: 4096
    /// - Gemma 4 (sliding window every-other-layer): 4096
    var defaultPrefillStepSize: Int { get }

    /// Whether this family requires a leading `<bos>` token on every
    /// prompt for coherent generation, *and* cannot rely on the
    /// tokenizer's post-processor to add it.
    ///
    /// Most families either don't need a BOS or get one "for free" from
    /// their `tokenizer.json` `TemplateProcessing` post-processor (its
    /// `single` template lists `<bos>` as a special token). Gemma 4 is
    /// the exception: it is BOS-critical — running it without a leading
    /// `<bos>` yields degraded, incoherent output — yet its
    /// `tokenizer.json` post-processor's `single` template is bare
    /// (`[Sequence A]`, no special token), so `Tokenizer.encode` returns
    /// no BOS. Gemma 3, by contrast, lists `<bos>` in its post-processor
    /// and works for free. Families that set this to `true` get an
    /// explicit BOS prepended by `Generate.swift` when `encode` did not
    /// already produce one. Default `false`.
    var requiresLeadingBOS: Bool { get }

    /// One per-layer state cache, sized for the model's defaults. The
    /// concrete type returned depends on the family (raw `KVCache` /
    /// `AffineQuantizedKVCache` for attention models, `Mamba2LayerCache`
    /// for Mamba 2). The engine knows its own cache type and casts back
    /// internally; callers pass the array through as
    /// `[any LayerCacheProtocol]`.
    func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol]

    /// Queue a single-token forward pass onto an existing command
    /// buffer. **Does not commit.** Returns the logits Tensor whose
    /// contents become valid after `cmd` is committed and completes.
    ///
    /// This is the primitive every higher-level entry point composes:
    /// `forward(...)` creates its own cmdbuf around this, `forwardSample`
    /// queues the argmax on the same cmdbuf, `forwardSampleCategorical`
    /// queues the categorical sampler on the same cmdbuf. Family files
    /// implement this once and everything else gets the 1-cmdbuf
    /// behaviour automatically — historically Llama and Qwen3 had to
    /// hand-roll fused forwardSampleCategorical overrides to avoid the
    /// 2-cmdbuf default path; with this protocol method, the default
    /// is fast and overrides are unnecessary.
    func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor

    /// Queue a **multi-token** forward pass — process `tokenIds.count`
    /// positions in one logical call. Updates each cache for positions
    /// `[position, position + tokenIds.count)`. Returns the logits at
    /// the **last** position only (the chunk's tail — the only thing
    /// prefill needs).
    ///
    /// The default implementation loops `forward(tokenId:)` one token
    /// at a time, so every family gets this surface for free. Family
    /// files override it with a chunked path that batches the QKV
    /// projection + a single `Ops.sdpaMulti(causal: true)` dispatch +
    /// batched MLP, eliminating the per-token kernel-launch and
    /// command-buffer overhead. That override is the TTFT
    /// win on long prompts; the default loop is correct-but-slow.
    ///
    /// `tokenIds.isEmpty == false` is a precondition (an empty chunk
    /// is a caller bug — `Generate.driveGeneration` never produces
    /// one).
    func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor

    /// Forward + GPU argmax in one command buffer. Returns just the
    /// chosen token id (4-byte readback) — no full logits transfer.
    func forwardSample(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol], device: Device
    ) -> Int

    /// Whether this engine supports `forward(inputEmbedding:...)` — the
    /// embedding-input forward path a VLM needs to splice image tokens
    /// into the text stream. Most text-only families return `false`; the
    /// VL-target families (Gemma 3 / 4, Qwen3 / 3.5, Nemotron-Labs-
    /// Diffusion) override this to `true`. Default `false`.
    var supportsEmbeddingInput: Bool { get }

    /// Queue a single-token forward pass that takes a precomputed
    /// `[hidden]` embedding row instead of a token id — the primitive a
    /// `VisionModel` uses to inject vision-encoder tokens at image-placeholder
    /// positions. Everything after the embedding lookup (norm scale,
    /// layer stack, lm_head) is identical to `forward(tokenId:...)`.
    ///
    /// Families that don't support this trap; check `supportsEmbeddingInput`
    /// first. The default implementation traps.
    func forward(
        inputEmbedding: Tensor, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor

    /// Look up the raw `[hidden]` embedding row for a text token — the
    /// table gather *without* any family-specific post-scale (Gemma's
    /// embed-scale is applied inside `forward(inputEmbedding:...)`, not
    /// here). A `VisionModel` uses this to build the spliced prompt-embedding
    /// stream: text tokens get `textEmbedding(...)`, image-placeholder
    /// positions get vision-encoder rows. VL-target families implement
    /// it; the default traps. Check `supportsEmbeddingInput` first.
    func textEmbedding(tokenId: Int, device: Device) -> Tensor

    /// Whether this family implements `forward(tokenIdBuffer:...)`: a
    /// GPU-resident decode step that takes the token id as a
    /// **device-resident** `Tensor` instead of a host `Int`. This is
    /// the primitive `GpuTokenLoop` (Generation/GpuTokenLoop.swift)
    /// needs to chain decode steps back-to-back on the GPU: step N+1's
    /// embedding gather reads step N's `Ops.argmax` output buffer
    /// directly, so issuing step N+1 never waits on the host having
    /// read step N's chosen token id back first. Default `false`;
    /// families opt in by overriding both this and
    /// `forward(tokenIdBuffer:...)`.
    var supportsGPUTokenFeedback: Bool { get }

    /// GPU-resident decode step: the same computation as
    /// `forward(tokenId:...)`, but the token id arrives as a
    /// single-element `.u32` device `Tensor`, typically a previous
    /// decode step's `Ops.argmax` output buffer, instead of a host
    /// `Int`. This is what lets `GpuTokenLoop` commit step N+1's work
    /// without waiting on step N's completion: the embedding gather
    /// only depends on a GPU buffer write that the same command queue
    /// already orders correctly, not on a host readback.
    ///
    /// Families that don't support this trap; check
    /// `supportsGPUTokenFeedback` before calling. The default
    /// implementation traps.
    func forward(
        tokenIdBuffer: Tensor, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor
}

extension LanguageModel {
    /// Default prefill chunk size matches mlx-swift-lm's generic
    /// `LanguageModel` default (1024 — Llama / Mistral / Phi / Qwen3
    /// dense). GPT-OSS, Qwen 3.5 MoE, Gemma 4 override this per the
    /// values benched in mlx-swift-lm.
    public var defaultPrefillStepSize: Int { 1024 }

    /// Default: no explicit BOS prefixing. Families that are BOS-critical
    /// and whose tokenizer post-processor does not add one (Gemma 4)
    /// override this to `true`.
    public var requiresLeadingBOS: Bool { false }

    /// Default: embedding-input forward unsupported. VL-target families
    /// override both this and `forward(inputEmbedding:...)`.
    public var supportsEmbeddingInput: Bool { false }

    /// Default `forward(inputEmbedding:...)`: traps. VL-target families
    /// provide a real implementation.
    public func forward(
        inputEmbedding: Tensor, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        preconditionFailure(
            "\(type(of: self)) does not support embedding-input forward — "
                + "check supportsEmbeddingInput before calling")
    }

    /// Default `textEmbedding(...)`: traps. VL-target families provide a
    /// real implementation.
    public func textEmbedding(tokenId: Int, device: Device) -> Tensor {
        preconditionFailure(
            "\(type(of: self)) does not support textEmbedding — "
                + "check supportsEmbeddingInput before calling")
    }

    /// Default: no GPU-resident token feedback. Families that
    /// implement `forward(tokenIdBuffer:...)` override this to `true`.
    public var supportsGPUTokenFeedback: Bool { false }

    /// Default `forward(tokenIdBuffer:...)`: traps. Families that
    /// support GPU-resident token feedback provide a real
    /// implementation and override `supportsGPUTokenFeedback`; callers
    /// (`GpuTokenLoop`) check that flag before ever calling this.
    public func forward(
        tokenIdBuffer: Tensor, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        preconditionFailure(
            "\(type(of: self)) does not support GPU-resident token feedback; "
                + "check supportsGPUTokenFeedback before calling")
    }

    public func makeLayerCaches(maxSeq: Int? = nil, device: Device = .shared)
        -> [any LayerCacheProtocol]
    {
        makeLayerCaches(maxSeq: maxSeq, device: device)
    }

    /// Default `forward(...)`: wraps `forward(...on cmd:)` in a fresh
    /// command buffer, commits, waits. Returns logits.
    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol], device: Device = .shared
    ) -> Tensor {
        let cmd = device.makeCommandBuffer()
        let logits = forward(
            tokenId: tokenId, position: position,
            caches: caches, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        return logits
    }

    /// Default `forwardMulti(...)`: loops `forward(tokenId:)` one
    /// position at a time on the supplied command buffer, returning
    /// the tail logits. Correct-but-slow — every family inherits this
    /// for free, optimised families override to batch the chunk in
    /// one `Ops.sdpaMulti(causal: true)` dispatch.
    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            !tokenIds.isEmpty,
            "forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, token) in tokenIds.enumerated() {
            logits = forward(
                tokenId: token, position: position + i,
                caches: caches, on: cmd, device: device)
        }
        return logits
    }

    /// Default `forwardSample(...)`: queues forward + argmax on the
    /// same command buffer; returns the chosen token id.
    public func forwardSample(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        device: Device = .shared
    ) -> Int {
        let dkTiming = LagunaDecodeKernelTiming.enabled
        var cmd = device.makeCommandBuffer()
        // `forward(...on cmd:)` runs its own internal layer-stack pipe (own
        // command buffers, already fully waited-on by the time it returns)
        // and only uses this outer `cmd` for the final norm + lm_head chain.
        let logits = forward(
            tokenId: tokenId, position: position,
            caches: caches, on: cmd, device: device)
        // T39: isolate the lm_head chain (final RMSNorm + coarse/exact/
        // select) from the argmax/sample dispatch that follows. No-op when
        // BUTTER_LAGUNA_DECODE_KERNEL_TIMING is unset.
        if dkTiming {
            cmd = LagunaDecodeKernelTiming.checkpoint("lmhead_chain", cmd: cmd, device: device)
        }
        // T21 hygiene: was a fresh 4-byte `device.makeBuffer` every decode
        // token (this is the default greedy-sample path every model
        // family inherits), pooled via the tag-keyed reusable-scratch
        // cache `Device` already uses for GGUF dequant intermediates.
        // Safe to reuse across tokens: the CPU read below happens only
        // after `waitUntilCompleted`/the dkTiming checkpoint, both
        // synchronous, so there's no cross-token overlap on one instance.
        let outBuf = device.intermediateScratch(tag: "lm_argmax_out", minBytes: 4)
        let outT = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)
        Ops.argmax(logits, into: outT, on: cmd)
        if dkTiming {
            _ = LagunaDecodeKernelTiming.checkpoint("argmax_sample", cmd: cmd, device: device)
        } else {
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        return Int(outBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }

    /// Forward + GPU softmax-categorical-sample for the pure-temperature
    /// sampling path (T > 0, no top-K / top-P / min-P / rep-penalty).
    /// Logits never cross to CPU; only the chosen token id (4 bytes)
    /// flows back.
    ///
    /// Queues forward + sampler on the SAME command buffer for a
    /// 1-commit-per-token decode step. Mamba 2 (and any future model
    /// family) gets this fused path automatically by implementing only
    /// the primitive `forward(...on cmd:)`. Previously this default
    /// used 2 cmdbufs (forward inside its own commit, then a separate
    /// commit for the sampler); Llama and Qwen3 worked around it with
    /// hand-rolled overrides which are now redundant.
    public func forwardSampleCategorical(
        tokenId: Int, position: Int, caches: [any LayerCacheProtocol],
        temperature: Float, uniformDraw: Float,
        device: Device = .shared
    ) -> Int {
        let cmd = device.makeCommandBuffer()
        let logits = forward(
            tokenId: tokenId, position: position,
            caches: caches, on: cmd, device: device)

        // T21 hygiene: temperature is ~constant across one generation's
        // token loop, so it fits `Device.scalarBuffer`'s cache-by-value
        // design exactly (same rationale as that cache's rmsNorm-eps /
        // RoPE-arg use). `uniformDraw` is a fresh random value every
        // token by design, so it can't be value-cached (that would grow
        // `scalarBufCache` unbounded), it gets the tag-keyed reusable
        // scratch instead, same as the argmax output buffer above.
        let tBuf = device.scalarBuffer(temperature)
        let temperatureT = Tensor(buffer: tBuf, offset: 0, shape: [1], dtype: .f32)

        let uBuf = device.intermediateScratch(tag: "lm_sample_uniform", minBytes: 4)
        var uVal = uniformDraw
        memcpy(uBuf.contents(), &uVal, 4)
        let uniformT = Tensor(buffer: uBuf, offset: 0, shape: [1], dtype: .f32)

        let outBuf = device.intermediateScratch(tag: "lm_categorical_out", minBytes: 4)
        let outT = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)
        Ops.softmaxCategoricalSample(
            logits, into: outT,
            temperature: temperatureT,
            uniform: uniformT, on: cmd)

        cmd.commit()
        cmd.waitUntilCompleted()
        return Int(outBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }
}
