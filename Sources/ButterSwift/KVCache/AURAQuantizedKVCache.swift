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
// AURAQuantizedKVCache — KVCacheProtocol conformance that stores K and V
// as AURA-codec packed indices + per-position norm corrections.
//
// Storage layout (per layer, both K and V):
//   packed  [nKVHeads, maxSeq, packedWidth] u32   — bit-packed codebook indices
//   norms   [nKVHeads, maxSeq]              f32   — per-position norm correction
//   working [nKVHeads, maxSeq, headDim]     dtype — shared across layers; bulk-dequant target
//
// `packedWidth = ceil(headDim * bits / 32)`. K and V use independent bit
// widths (the production aura4v2 recipe is K=4-bit, V=2-bit per
// `papers/aura-compression-algorithm.md` §2.5).
//
// ─── Per-layer SRHT rotation Π ──────────────────────────────────────
//
// Π is the random orthogonal rotation the AURA codec assumes its inputs
// have been multiplied by (Lloyd-Max boundaries were trained against
// Beta-distributed rotated coordinates — see
// `papers/aura-compression-algorithm.md` §2.2). The encode kernel
// applies Π internally before quantising and `prepareForAttention`
// returns K and V *still rotated*, so the rest of the forward pass has
// to compensate:
//
//   * Q is rotated by Π post-RoPE so the SDPA score
//     `(Π·Q)·(Π·K)^T = Q·K^T` recovers the original attention scores
//     (orthogonality + transpose of an orthogonal matrix is its
//     inverse).
//   * The SDPA output is in Π-rotated space (`Σ softmax · (Π·V) = Π·O`)
//     so the layer un-rotates by Π^T before `oProj`.
//
// RoPE doesn't commute with arbitrary orthogonal rotations, so the
// per-token call order MUST be project → RMSNorm → RoPE → Π·.
//
// Π is built per-layer (deterministic seed = layer index) by
// `AURARotation.srhtMatrix` and is shared across heads within a layer.
// Π^T is precomputed at cache build time so the runtime path is two
// per-head gemvs (no transpose dispatch). For an orthogonal Π,
// Π^T = Π^(-1); the encode kernel only needs Π itself.
//
// Stage 1a (this implementation) applies Π via per-head Ops.gemv
// dispatches; Stage 1b folds the rotation into the compressed-domain
// `aura_flash_p1` / pass2 kernels so the cost drops to one fused
// dispatch.
//
// ─── Dispatch granularity ───────────────────────────────────────────
//
// The append paths encode every K row in one dispatch and every V row in
// one dispatch. Reusable contiguous scratch rows receive the atomic output,
// then one ordered blit scatters those rows into the permanent head-major
// cache. This preserves the attention layout while avoiding one encode
// dispatch per head or token.

import Foundation
import Metal

public final class AURAQuantizedKVCache: KVCacheProtocol, @unchecked Sendable {
    private let device: Device
    public let nKVHeads: Int
    public let headDim: Int
    public let maxSeq: Int
    public let dtype: DType
    public let scheme: AURAScheme
    /// Which decode-time attention path this cache uses. Set at load
    /// time from `LoadOptions.auraDecodePath`; default `.compressed`.
    /// Today both paths take the dequant-mirror code in
    /// `prepareForAttention(on:)` — the `.compressed` path's
    /// `Ops.auraFlashP1` / `auraFlashPass2` wrappers haven't been
    /// authored yet; until they ship, `.compressed` quietly falls back
    /// to the same mirror dispatch as `.dequantMirror`. The flag is
    /// wired now so callers can opt-in to the compressed path the
    /// moment the wrappers land — no API change needed at that point.
    public let decodePath: AURADecodePath

    public let kPackedWidth: Int
    public let vPackedWidth: Int

    // Codec data — built per-layer by makeLayerCaches and passed in here.
    // (`rotation` / `rotationT` are per-layer; the codebooks are shared
    // across layers because Lloyd-Max levels are dim-only — no layer
    // statistics are baked in yet.)
    public let rotation: Tensor  // [headDim, headDim] f32 — Π, encode kernel input
    public let rotationT: Tensor  // [headDim, headDim] f32 — Π^T, for un-rotation kernels / future use
    /// Π in the activation dtype, used by `Ops.auraRotatePerHead` on the
    /// Q post-RoPE side. Aliases `rotation` when `dtype == .f32`.
    public let rotationDtype: Tensor
    /// Π^T in the activation dtype, used to un-rotate the SDPA output
    /// before `oProj`. Aliases `rotationT` when `dtype == .f32`.
    public let rotationDtypeT: Tensor
    /// Codebook in the cache dtype. Encode + decode kernels read
    /// directly with no per-call cast — the dtype unification landed
    /// when the single-pass `aura_flash_sdpa` kernel was migrated to
    /// `Tensor<T>` (matches the production C++ TQ+ fork pattern: fp16-
    /// stored norms / codebook, f32-at-use via cast-at-load).
    public let kCodebook: Tensor  // [2^keyBits]   dtype
    public let kBoundaries: Tensor  // [2^keyBits-1] dtype — Lloyd-Max thresholds (Tensor<T> per iron #226)
    public let vCodebook: Tensor  // [2^valueBits] dtype
    public let vBoundaries: Tensor  // [2^valueBits-1] dtype

    // Per-cache compressed storage.
    public let kPacked: Tensor  // [nKVHeads, maxSeq, kPackedWidth] u32
    public let vPacked: Tensor  // [nKVHeads, maxSeq, vPackedWidth] u32
    public let kNorms: Tensor  // [nKVHeads, maxSeq] dtype — encode writes T, decode reads T
    public let vNorms: Tensor  // [nKVHeads, maxSeq] dtype

    // Reusable contiguous output for single-step and range appends. The encode
    // kernel uses atomic OR for packed codes, so these packed scratches are
    // cleared before each append. Norm scratches are overwritten directly.
    private var kAppendPackedScratch: Tensor
    private var vAppendPackedScratch: Tensor
    private var kAppendNormsScratch: Tensor
    private var vAppendNormsScratch: Tensor
    private var appendScratchRows: Int

    // Shared working buffers — bulk-dequant target; reused across layers.
    public let sharedWorkingK: Tensor  // [nKVHeads, maxSeq, headDim] dtype
    public let sharedWorkingV: Tensor

    /// Lock-protected fill state. Same Phase-8 concurrent-decode prep
    /// reasoning as `KVCache.lengthLock` — see `Sources/ButterSwift/KVCache.swift`.
    private let lengthLock = NSLock()
    private var _evictionState: KVEvictionState
    public var length: Int { lengthLock.withLock { _evictionState.length } }
    public var absolutePosition: Int { lengthLock.withLock { _evictionState.absolutePosition } }
    public var eviction: KVEviction { _evictionState.policy }
    public var effectiveMaxSize: Int {
        switch _evictionState.policy {
        case .unbounded: return maxSeq
        case .window(let m, _): return m
        }
    }

    public convenience init(
        nKVHeads: Int, headDim: Int, maxSeq: Int, dtype: DType,
        scheme: AURAScheme,
        rotation: Tensor, rotationT: Tensor,
        rotationDtype: Tensor, rotationDtypeT: Tensor,
        kCodebook: Tensor, kBoundaries: Tensor,
        vCodebook: Tensor, vBoundaries: Tensor,
        sharedWorkingK: Tensor, sharedWorkingV: Tensor,
        decodePath: AURADecodePath = .compressed,
        device: Device = .shared
    ) {
        self.init(
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
            dtype: dtype, scheme: scheme,
            rotation: rotation, rotationT: rotationT,
            rotationDtype: rotationDtype, rotationDtypeT: rotationDtypeT,
            kCodebook: kCodebook, kBoundaries: kBoundaries,
            vCodebook: vCodebook, vBoundaries: vBoundaries,
            sharedWorkingK: sharedWorkingK, sharedWorkingV: sharedWorkingV,
            eviction: .unbounded, decodePath: decodePath,
            device: device)
    }

    public init(
        nKVHeads: Int, headDim: Int, maxSeq: Int, dtype: DType,
        scheme: AURAScheme,
        rotation: Tensor, rotationT: Tensor,
        rotationDtype: Tensor, rotationDtypeT: Tensor,
        kCodebook: Tensor, kBoundaries: Tensor,
        vCodebook: Tensor, vBoundaries: Tensor,
        sharedWorkingK: Tensor, sharedWorkingV: Tensor,
        eviction: KVEviction,
        decodePath: AURADecodePath = .compressed,
        device: Device = .shared
    ) {
        precondition(
            rotation.shape == [headDim, headDim],
            "AURAQuantizedKVCache: rotation must be [headDim, headDim]")
        precondition(
            rotation.dtype == .f32,
            "AURAQuantizedKVCache: rotation must be f32")
        precondition(
            rotationT.shape == [headDim, headDim],
            "AURAQuantizedKVCache: rotationT must be [headDim, headDim]")
        precondition(
            rotationT.dtype == .f32,
            "AURAQuantizedKVCache: rotationT must be f32")
        precondition(
            rotationDtype.shape == [headDim, headDim]
                && rotationDtypeT.shape == [headDim, headDim],
            "AURAQuantizedKVCache: rotationDtype/rotationDtypeT must be [headDim, headDim]")
        precondition(
            rotationDtype.dtype == dtype && rotationDtypeT.dtype == dtype,
            "AURAQuantizedKVCache: rotationDtype/rotationDtypeT dtype must match cache dtype \(dtype)"
        )
        precondition(
            kCodebook.dtype == dtype,
            "AURAQuantizedKVCache: K codebook dtype must match cache dtype \(dtype)")
        precondition(
            kBoundaries.dtype == dtype,
            "AURAQuantizedKVCache: K boundaries dtype must match cache dtype \(dtype) "
                + "— iron #226 unified rotation/boundaries to Tensor<T>")
        precondition(
            vCodebook.dtype == dtype,
            "AURAQuantizedKVCache: V codebook dtype must match cache dtype \(dtype)")
        precondition(
            vBoundaries.dtype == dtype,
            "AURAQuantizedKVCache: V boundaries dtype must match cache dtype \(dtype)")
        precondition(
            sharedWorkingK.shape == [nKVHeads, maxSeq, headDim],
            "AURAQuantizedKVCache: sharedWorkingK shape mismatch")
        precondition(
            sharedWorkingV.shape == [nKVHeads, maxSeq, headDim],
            "AURAQuantizedKVCache: sharedWorkingV shape mismatch")
        precondition(
            sharedWorkingK.dtype == dtype && sharedWorkingV.dtype == dtype,
            "AURAQuantizedKVCache: sharedWorking dtype mismatch")

        self.device = device
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.scheme = scheme
        self.decodePath = decodePath
        self.rotation = rotation
        self.rotationT = rotationT
        self.rotationDtype = rotationDtype
        self.rotationDtypeT = rotationDtypeT
        self.kCodebook = kCodebook
        self.kBoundaries = kBoundaries
        self.vCodebook = vCodebook
        self.vBoundaries = vBoundaries
        self.sharedWorkingK = sharedWorkingK
        self.sharedWorkingV = sharedWorkingV

        self.kPackedWidth = AURACodebook.packedWidth(dim: headDim, bits: scheme.keyBits)
        self.vPackedWidth = AURACodebook.packedWidth(dim: headDim, bits: scheme.valueBits)

        self.kPacked = Tensor.empty(
            shape: [nKVHeads, maxSeq, kPackedWidth], dtype: .u32, device: device)
        self.vPacked = Tensor.empty(
            shape: [nKVHeads, maxSeq, vPackedWidth], dtype: .u32, device: device)
        self.kNorms = Tensor.empty(
            shape: [nKVHeads, maxSeq], dtype: dtype, device: device)
        self.vNorms = Tensor.empty(
            shape: [nKVHeads, maxSeq], dtype: dtype, device: device)
        self.kAppendPackedScratch = Tensor.empty(
            shape: [nKVHeads, kPackedWidth], dtype: .u32, device: device)
        self.vAppendPackedScratch = Tensor.empty(
            shape: [nKVHeads, vPackedWidth], dtype: .u32, device: device)
        self.kAppendNormsScratch = Tensor.empty(
            shape: [nKVHeads], dtype: dtype, device: device)
        self.vAppendNormsScratch = Tensor.empty(
            shape: [nKVHeads], dtype: dtype, device: device)
        self.appendScratchRows = nKVHeads

        // Codec is purely additive in atomic_or terms, so packed slots
        // MUST start zeroed. Norms slots get overwritten per encode but
        // zero is a safe default.
        kPacked.zero()
        vPacked.zero()
        kNorms.zero()
        vNorms.zero()

        self._evictionState = KVEvictionState(policy: eviction, bufferCapacity: maxSeq)
    }

    public func reset() { lengthLock.withLock { _evictionState.reset() } }

    public func truncate(toLength length: Int) {
        lengthLock.withLock { _evictionState.truncate(toLength: length) }
    }

    /// Encode the current step's K + V into the compressed storage.
    /// `kFlat` and `vFlat` come in as [nKVHeads, headDim] in the model's
    /// dtype (bf16/f16/f32) — the encode kernel is dtype-generic, so no
    /// upcast is needed.
    public func appendOnGPU(
        kFlat: Tensor, vFlat: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            kFlat.dtype == dtype && vFlat.dtype == dtype,
            "AURAQuantizedKVCache: kFlat/vFlat dtype must match cache dtype")
        precondition(
            kFlat.elementCount == nKVHeads * headDim,
            "AURAQuantizedKVCache: kFlat shape mismatch")
        precondition(
            vFlat.elementCount == nKVHeads * headDim,
            "AURAQuantizedKVCache: vFlat shape mismatch")

        lengthLock.withLock {
            let pos = _evictionState.reserveNextSlot()
            clearAppendPackedScratch(rows: nKVHeads, on: cmd)
            Ops.auraEncode(
                input: kFlat.reshaped(to: [nKVHeads, headDim]),
                rotation: rotationDtype,
                boundaries: kBoundaries, codebook: kCodebook,
                packedOut: kAppendPackedScratch, normsOut: kAppendNormsScratch,
                rows: nKVHeads, dim: headDim, packedWidth: kPackedWidth,
                bits: scheme.keyBits, on: cmd)
            Ops.auraEncode(
                input: vFlat.reshaped(to: [nKVHeads, headDim]),
                rotation: rotationDtype,
                boundaries: vBoundaries, codebook: vCodebook,
                packedOut: vAppendPackedScratch, normsOut: vAppendNormsScratch,
                rows: nKVHeads, dim: headDim, packedWidth: vPackedWidth,
                bits: scheme.valueBits, on: cmd)
            scatterAppendScratch(positions: [pos], on: cmd)
        }
    }

    /// Encode a contiguous `[T, nKVHeads, headDim]` range with one K
    /// dispatch and one V dispatch, then scatter into the head-major cache.
    public func appendManyOnGPU(
        kFlat: Tensor, vFlat: Tensor, t: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(t > 0, "AURAQuantizedKVCache: appendMany T must be positive")
        precondition(
            kFlat.dtype == dtype && vFlat.dtype == dtype,
            "AURAQuantizedKVCache: appendMany dtype mismatch")
        let rows = t * nKVHeads
        precondition(
            kFlat.elementCount == rows * headDim
                && vFlat.elementCount == rows * headDim,
            "AURAQuantizedKVCache: appendMany shape mismatch")

        lengthLock.withLock {
            var positions = [Int]()
            positions.reserveCapacity(t)
            for _ in 0 ..< t {
                positions.append(_evictionState.reserveNextSlot())
            }
            ensureAppendScratchCapacity(rows: rows)
            clearAppendPackedScratch(rows: rows, on: cmd)
            Ops.auraEncode(
                input: kFlat.reshaped(to: [rows, headDim]),
                rotation: rotationDtype,
                boundaries: kBoundaries, codebook: kCodebook,
                packedOut: kAppendPackedScratch, normsOut: kAppendNormsScratch,
                rows: rows, dim: headDim, packedWidth: kPackedWidth,
                bits: scheme.keyBits, on: cmd)
            Ops.auraEncode(
                input: vFlat.reshaped(to: [rows, headDim]),
                rotation: rotationDtype,
                boundaries: vBoundaries, codebook: vCodebook,
                packedOut: vAppendPackedScratch, normsOut: vAppendNormsScratch,
                rows: rows, dim: headDim, packedWidth: vPackedWidth,
                bits: scheme.valueBits, on: cmd)
            scatterAppendScratch(positions: positions, on: cmd)
        }
    }

    private func ensureAppendScratchCapacity(rows: Int) {
        guard rows > appendScratchRows else { return }
        kAppendPackedScratch = Tensor.empty(
            shape: [rows, kPackedWidth], dtype: .u32, device: device)
        vAppendPackedScratch = Tensor.empty(
            shape: [rows, vPackedWidth], dtype: .u32, device: device)
        kAppendNormsScratch = Tensor.empty(shape: [rows], dtype: dtype, device: device)
        vAppendNormsScratch = Tensor.empty(shape: [rows], dtype: dtype, device: device)
        appendScratchRows = rows
    }

    private func clearAppendPackedScratch(rows: Int, on cmd: MTLCommandBuffer) {
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure("AURAQuantizedKVCache: append scratch clear failed")
        }
        blit.fill(
            buffer: kAppendPackedScratch.buffer,
            range: kAppendPackedScratch.offset
                ..< (kAppendPackedScratch.offset + rows * kPackedWidth * 4),
            value: 0)
        blit.fill(
            buffer: vAppendPackedScratch.buffer,
            range: vAppendPackedScratch.offset
                ..< (vAppendPackedScratch.offset + rows * vPackedWidth * 4),
            value: 0)
        blit.endEncoding()
    }

    /// Bulk-dequant the entire filled prefix of K and V into the shared
    /// working buffers. The output lives in **Π-rotated** codec space.
    /// The caller (e.g. `Qwen3Layer.forward`) is expected to apply Π to
    /// Q before SDPA so the scores cancel correctly, and apply Π^T to
    /// the SDPA output before `oProj` so the residual stream is in the
    /// original activation space. See file header for the math.
    public func prepareForAttention(on cmd: MTLCommandBuffer) -> (k: Tensor, v: Tensor) {
        guard length > 0 else { return (sharedWorkingK, sharedWorkingV) }
        // kPacked / kNorms / sharedWorkingK are all laid out
        // [nKVHeads, maxSeq, …], so the dequant kernel must use `maxSeq`
        // as the per-head row stride — `length` (the fill count) only
        // sets how many rows to process. Passing `length` as the stride
        // mis-offsets every head past head 0, with error growing as the
        // cache fills (the AURA index-50 collapse).
        Ops.auraDequantRotated(
            packed: kPacked, norms: kNorms, codebook: kCodebook,
            into: sharedWorkingK,
            nKVHeads: nKVHeads, dim: headDim, packedWidth: kPackedWidth,
            tokens: length, bits: scheme.keyBits, cacheStride: maxSeq, on: cmd)
        Ops.auraDequantRotated(
            packed: vPacked, norms: vNorms, codebook: vCodebook,
            into: sharedWorkingV,
            nKVHeads: nKVHeads, dim: headDim, packedWidth: vPackedWidth,
            tokens: length, bits: scheme.valueBits, cacheStride: maxSeq, on: cmd)
        return (sharedWorkingK, sharedWorkingV)
    }

    // MARK: - bytesAllocated / bytesInUse

    public var bytesAllocated: Int {
        // Per-cache storage. Shared working buffers and codec tensors are
        // accounted once by the caller.
        let packedBytes = nKVHeads * maxSeq * (kPackedWidth + vPackedWidth) * 4
        let normBytes = nKVHeads * maxSeq * 2 * dtype.byteSize
        let scratchPackedBytes = appendScratchRows * (kPackedWidth + vPackedWidth) * 4
        let scratchNormBytes = appendScratchRows * 2 * dtype.byteSize
        return packedBytes + normBytes + scratchPackedBytes + scratchNormBytes
    }

    public var bytesInUse: Int {
        let filled = length
        let packedBytes = nKVHeads * filled * (kPackedWidth + vPackedWidth) * 4
        let normBytes = nKVHeads * filled * 2 * dtype.byteSize
        return packedBytes + normBytes
    }

    // MARK: - Internals

    private func scatterAppendScratch(positions: [Int], on cmd: MTLCommandBuffer) {
        let normByteSize = dtype.byteSize
        let normBytesPerHead = maxSeq * normByteSize
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure("AURAQuantizedKVCache: append scratch scatter failed")
        }
        func scatter(
            packedScratch: Tensor, normsScratch: Tensor,
            packed: Tensor, norms: Tensor, packedWidth: Int
        ) {
            let packedBytesPerSlot = packedWidth * 4
            let packedBytesPerHead = maxSeq * packedBytesPerSlot
            for r in positions.indices {
                for h in 0 ..< nKVHeads {
                    let sourceRow = r * nKVHeads + h
                    blit.copy(
                        from: packedScratch.buffer,
                        sourceOffset: packedScratch.offset + sourceRow * packedBytesPerSlot,
                        to: packed.buffer,
                        destinationOffset: packed.offset
                            + h * packedBytesPerHead + positions[r] * packedBytesPerSlot,
                        size: packedBytesPerSlot)
                    blit.copy(
                        from: normsScratch.buffer,
                        sourceOffset: normsScratch.offset + sourceRow * normByteSize,
                        to: norms.buffer,
                        destinationOffset: norms.offset
                            + h * normBytesPerHead + positions[r] * normByteSize,
                        size: normByteSize)
                }
            }
        }
        scatter(
            packedScratch: kAppendPackedScratch, normsScratch: kAppendNormsScratch,
            packed: kPacked, norms: kNorms, packedWidth: kPackedWidth)
        scatter(
            packedScratch: vAppendPackedScratch, normsScratch: vAppendNormsScratch,
            packed: vPacked, norms: vNorms, packedWidth: vPackedWidth)
        blit.endEncoding()
    }
}

/// Build the four rotation tensors AURAQuantizedKVCache needs for a
/// single layer. Π is generated via SRHT seeded by `layerIndex` so
/// every layer gets its own rotation but the model is fully
/// deterministic across runs (and across model loads). Π^T is computed
/// CPU-side; activation-dtype copies are pre-cast so the runtime path
/// can call Ops.auraRotatePerHead without a cast dispatch.
///
/// Lives outside the cache class so both Qwen3 and Llama (and any
/// future host model that wires AURAQuantizedKVCache) can build the
/// per-layer rotations the same way.
public enum AURAQuantizedKVCacheRotations {
    public struct Bundle {
        public let rotation: Tensor  // [headDim, headDim] f32
        public let rotationT: Tensor  // [headDim, headDim] f32
        public let rotationDtype: Tensor  // [headDim, headDim] activationDtype
        public let rotationDtypeT: Tensor  // [headDim, headDim] activationDtype
    }

    public static func build(
        headDim: Int, layerIndex: Int,
        activationDtype: DType, device: Device
    ) -> Bundle {
        // CPU-side rotation. SRHT requires power-of-2 headDim; transpose
        // is a straightforward row/column swap of the row-major buffer.
        let piData = AURARotation.srhtMatrix(dim: headDim, seed: UInt64(layerIndex))
        var piTData = [Float](repeating: 0, count: headDim * headDim)
        for i in 0 ..< headDim {
            for j in 0 ..< headDim {
                piTData[j * headDim + i] = piData[i * headDim + j]
            }
        }

        // f32 copies — required by the encode kernel and kept around for
        // any future kernel that expects f32 rotations.
        let rotation = Tensor.empty(shape: [headDim, headDim], dtype: .f32, device: device)
        rotation.copyIn(from: piData)
        let rotationT = Tensor.empty(shape: [headDim, headDim], dtype: .f32, device: device)
        rotationT.copyIn(from: piTData)

        // Activation-dtype copies — alias when dtype is f32 (no cast
        // needed), otherwise cast bit-pattern-correctly into a fresh
        // tensor. Small (headDim^2 * dtype bytes) so the cast cost is
        // negligible vs the inference loop.
        let rotationDtype: Tensor
        let rotationDtypeT: Tensor
        switch activationDtype {
        case .f32:
            rotationDtype = rotation
            rotationDtypeT = rotationT
        case .f16:
            rotationDtype = Tensor.empty(shape: [headDim, headDim], dtype: .f16, device: device)
            rotationDtype.copyIn(from: piData.map { Float16($0) })
            rotationDtypeT = Tensor.empty(shape: [headDim, headDim], dtype: .f16, device: device)
            rotationDtypeT.copyIn(from: piTData.map { Float16($0) })
        case .bf16:
            // bf16 is the top 16 bits of fp32 — truncating cast.
            rotationDtype = Tensor.empty(shape: [headDim, headDim], dtype: .bf16, device: device)
            rotationDtype.copyIn(
                from: piData.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })
            rotationDtypeT = Tensor.empty(shape: [headDim, headDim], dtype: .bf16, device: device)
            rotationDtypeT.copyIn(
                from: piTData.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })
        default:
            preconditionFailure(
                "AURAQuantizedKVCacheRotations: unsupported activation dtype \(activationDtype)")
        }

        return Bundle(
            rotation: rotation, rotationT: rotationT,
            rotationDtype: rotationDtype, rotationDtypeT: rotationDtypeT)
    }
}
