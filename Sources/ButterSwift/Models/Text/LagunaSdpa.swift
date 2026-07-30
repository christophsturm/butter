// Copyright 2026 Tom Turney (@TheTom)
//
// Laguna prefill SDPA: optional `sdpaPrefillMma` path (Hy3-style) via
// bf16→f16 cast + compact KV. Opt-in: BUTTER_LAGUNA_SDPA_PREFILL_MMA=1
// (default ON when nRows≥32 and nRows%32==0 for full-attn / unwindowed).
// Escape: BUTTER_LAGUNA_SDPA_PREFILL_MMA=0
//
// Decode QK-norm + plain RoPE fuse for sliding layers (T=1):
//   BUTTER_LAGUNA_FUSED_QK_NORM_ROPE != "0" (DEFAULT ON). Escape `=0`.
// Port of mlxfast DARKBLOOM_FUSED_SLIDING_QK_NORM_ROPE — one TG per head,
// one simdgroup, no threadgroup barrier (simd_sum already broadcasts).

import Foundation
import Metal
import IronSwift

enum LagunaSdpa {
    static var prefillMmaEnabled: Bool {
        let env = ProcessInfo.processInfo.environment["BUTTER_LAGUNA_SDPA_PREFILL_MMA"]
        if env == "0" { return false }
        if env == "1" { return true }
        return true  // default ON when shapes allow
    }

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        let lib = try! device.makeLibrary(source: source, options: MTLCompileOptions())
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    private static let compactSrc = """
    #include <metal_stdlib>
    using namespace metal;
    // Compact strided bf16 KV [nKV, maxSeq, D] → packed f16 [nKV, kLen, D].
    kernel void laguna_compact_kv_bf16_to_f16(
        device const bfloat *src [[buffer(0)]],
        device half *dst [[buffer(1)]],
        constant uint &n_kv [[buffer(2)]],
        constant uint &max_seq [[buffer(3)]],
        constant uint &k_len [[buffer(4)]],
        constant uint &head_dim [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint h = gid.x;
        uint td = gid.y;
        if (h >= n_kv || td >= k_len * head_dim) return;
        uint t = td / head_dim;
        uint d = td % head_dim;
        uint srcIdx = ((h * max_seq) + t) * head_dim + d;
        uint dstIdx = ((h * k_len) + t) * head_dim + d;
        dst[dstIdx] = half(float(src[srcIdx]));
    }
    """

    private static let castF16ToBf16Src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void laguna_cast_f16_to_bf16(
        device const half *inp [[buffer(0)]],
        device bfloat *out [[buffer(1)]],
        uint i [[thread_position_in_grid]]
    ) {
        out[i] = bfloat(float(inp[i]));
    }
    """

    static func compactKvBf16ToF16(
        src: Tensor, nKV: Int, maxSeq: Int, kLen: Int, headDim: Int,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor? = nil,
        encoder: MTLComputeCommandEncoder? = nil
    ) -> Tensor {
        precondition(src.dtype == .bf16 && kLen > 0 && kLen <= maxSeq)
        let result = out ?? Tensor.empty(shape: [nKV, kLen, headDim], dtype: .f16, device: device)
        let p = pso(
            "laguna_compact_kv_bf16_to_f16", source: compactSrc, device: device.mtlDevice)
        let owns = encoder == nil
        guard let enc = encoder ?? cmd.makeComputeCommandEncoder() else { return result }
        enc.setComputePipelineState(p)
        enc.setBuffer(src.buffer, offset: src.offset, index: 0)
        enc.setBuffer(result.buffer, offset: result.offset, index: 1)
        var nKVU = UInt32(nKV), maxU = UInt32(maxSeq), kU = UInt32(kLen), dU = UInt32(headDim)
        enc.setBytes(&nKVU, length: 4, index: 2)
        enc.setBytes(&maxU, length: 4, index: 3)
        enc.setBytes(&kU, length: 4, index: 4)
        enc.setBytes(&dU, length: 4, index: 5)
        let td = kLen * headDim
        enc.dispatchThreads(
            MTLSize(width: nKV, height: td, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: min(td, 256), depth: 1))
        if owns { enc.endEncoding() }
        return result
    }

    /// Ring of scratch slots so prefill layer-pipe (depth 4) does not
    /// overwrite an in-flight layer's SDPA temps.
    private static let scratchRing = 8
    nonisolated(unsafe) private static var scratchSlot: Int = 0
    private static let scratchSlotLock = NSLock()

    private static func nextSlot() -> Int {
        scratchSlotLock.lock()
        let s = scratchSlot
        scratchSlot = (scratchSlot + 1) % scratchRing
        scratchSlotLock.unlock()
        return s
    }

    /// Grow-only scratch via Device.intermediateScratch (ring-tagged).
    private static func scratch(
        tag: String, shape: [Int], dtype: DType, device: Device, slot: Int
    ) -> Tensor {
        let n = shape.reduce(1, *)
        let bytes = n * dtype.byteSize
        let buf = device.intermediateScratch(
            tag: "\(tag)_s\(slot)", minBytes: bytes)
        return Tensor(buffer: buf, offset: 0, shape: shape, dtype: dtype)
    }

    static func castF16ToBf16(
        _ input: Tensor, into out: Tensor, on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(input.dtype == .f16 && out.dtype == .bf16)
        precondition(input.elementCount == out.elementCount)
        let n = input.elementCount
        let p = pso(
            "laguna_cast_f16_to_bf16", source: castF16ToBf16Src, device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(out.buffer, offset: out.offset, index: 1)
        enc.dispatchThreads(
            MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(n, 256), height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Causal prefill SDPA via `sdpaPrefillMma` (f16 MMA). Returns bf16
    /// `[nRows, nHeads*headDim]`. headDim=128, nRows≥32. Pads Q/K to next
    /// multiple of 32 when needed (pad queries discarded after).
    /// - Parameter phaseCheckpoint: T26 diagnostic hook only (nil in the
    ///   default/hot path, zero behavior change when unset). When
    ///   provided (`BUTTER_LAGUNA_ATTN_PHASE_TIMING=1`), called at each
    ///   internal stage boundary with `(label, currentBuffer)` and must
    ///   return the buffer to keep encoding onto (a forced commit+wait
    ///   checkpoint reassigns `cmd` to a fresh buffer, same technique as
    ///   the outer `LagunaLayer.forwardMulti` breakdown). Lets Stage 1's
    ///   "SDPA MMA kernel itself vs. layout transposes/copies" question
    ///   get a real per-stage answer instead of one lumped "sdpa" bucket.
    ///   See benchmarks/laguna-xs/README.md T26.
    static func prefillMmaBf16(
        qTHD: Tensor,  // [nRows, nHeads, headDim] bf16
        cacheK: Tensor, cacheV: Tensor,
        nRows: Int, position: Int,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        maxSeq: Int, scale: Float,
        on cmd: inout MTLCommandBuffer, device: Device,
        phaseCheckpoint: ((String, MTLCommandBuffer) -> MTLCommandBuffer)? = nil
    ) -> Tensor? {
        guard prefillMmaEnabled,
            qTHD.dtype == .bf16,
            headDim == 128,
            nRows >= 32
        else { return nil }

        // Pad T up to next BQ=32 tile (kernel requirement).
        let tPad = ((nRows + 31) / 32) * 32
        let kReal = position + nRows
        let kPad = position + tPad
        precondition(kReal <= maxSeq && kPad <= maxSeq + 32)  // pad may exceed cache
        // If pad would need K beyond maxSeq, fall back (rare at ctx clamp).
        if kPad > maxSeq { return nil }

        let qElems = nRows * nHeads * headDim
        let qPadElems = tPad * nHeads * headDim
        let slot = nextSlot()

        // Scratch temps (ring of 8 — safe under prefill pipe depth 4).
        let qPadBf = scratch(
            tag: "laguna_sdpa_qpad_bf", shape: [tPad, nHeads, headDim],
            dtype: .bf16, device: device, slot: slot)
        let qF16 = scratch(
            tag: "laguna_sdpa_qf16", shape: [tPad, nHeads, headDim],
            dtype: .f16, device: device, slot: slot)
        let qHTD = scratch(
            tag: "laguna_sdpa_qhtd", shape: [nHeads, tPad, headDim],
            dtype: .f16, device: device, slot: slot)
        let outHTD = scratch(
            tag: "laguna_sdpa_outhtd", shape: [nHeads, tPad, headDim],
            dtype: .f16, device: device, slot: slot)
        let outTHD = scratch(
            tag: "laguna_sdpa_outthd", shape: [tPad, nHeads, headDim],
            dtype: .f16, device: device, slot: slot)
        // Owned output — must not share ring (caller holds across layers).
        let outBf = Tensor.empty(
            shape: [nRows, nHeads * headDim], dtype: .bf16, device: device)
        let kPacked = scratch(
            tag: "laguna_sdpa_kpack", shape: [nKVHeads, kPad, headDim],
            dtype: .f16, device: device, slot: slot)
        let vPacked = scratch(
            tag: "laguna_sdpa_vpack", shape: [nKVHeads, kPad, headDim],
            dtype: .f16, device: device, slot: slot)

        // Q pad: copy real rows, zero pad tail.
        if let blit = cmd.makeBlitCommandEncoder() {
            blit.copy(
                from: qTHD.buffer, sourceOffset: qTHD.offset,
                to: qPadBf.buffer, destinationOffset: qPadBf.offset,
                size: qElems * 2)
            if tPad > nRows {
                let padOff = qPadBf.offset + qElems * 2
                blit.fill(
                    buffer: qPadBf.buffer,
                    range: padOff ..< (padOff + (qPadElems - qElems) * 2),
                    value: 0)
            }
            blit.endEncoding()
        }
        if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_qpad", cmd) }

        // Q: bf16 [Tpad,H,D] → f16 → transpose → [H,Tpad,D]
        _ = Ops.castBf16ToF16(
            qPadBf.reshaped(to: [qPadElems]),
            into: qF16.reshaped(to: [qPadElems]), on: cmd)
        _ = Ops.transposeTHD_f16(qF16, t: tPad, h: nHeads, d: headDim, on: cmd, into: qHTD)
        if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_qcast_transpose", cmd) }

        // K/V: concurrent compact (common path kPad==kReal).
        if kPad == kReal {
            if let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) {
                _ = compactKvBf16ToF16(
                    src: cacheK, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: kPacked, encoder: enc)
                _ = compactKvBf16ToF16(
                    src: cacheV, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: vPacked, encoder: enc)
                enc.endEncoding()
            } else {
                _ = compactKvBf16ToF16(
                    src: cacheK, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: kPacked)
                _ = compactKvBf16ToF16(
                    src: cacheV, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: vPacked)
            }
        } else {
            let kRealPacked = scratch(
                tag: "laguna_sdpa_kreal", shape: [nKVHeads, kReal, headDim],
                dtype: .f16, device: device, slot: slot)
            let vRealPacked = scratch(
                tag: "laguna_sdpa_vreal", shape: [nKVHeads, kReal, headDim],
                dtype: .f16, device: device, slot: slot)
            if let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) {
                _ = compactKvBf16ToF16(
                    src: cacheK, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: kRealPacked, encoder: enc)
                _ = compactKvBf16ToF16(
                    src: cacheV, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: vRealPacked, encoder: enc)
                enc.endEncoding()
            }
            if let blit = cmd.makeBlitCommandEncoder() {
                let padBytes = nKVHeads * kPad * headDim * 2
                blit.fill(
                    buffer: kPacked.buffer,
                    range: kPacked.offset ..< (kPacked.offset + padBytes), value: 0)
                blit.fill(
                    buffer: vPacked.buffer,
                    range: vPacked.offset ..< (vPacked.offset + padBytes), value: 0)
                for h in 0 ..< nKVHeads {
                    let kSrcOff = kRealPacked.offset + h * kReal * headDim * 2
                    let vSrcOff = vRealPacked.offset + h * kReal * headDim * 2
                    let kDstOff = kPacked.offset + h * kPad * headDim * 2
                    let vDstOff = vPacked.offset + h * kPad * headDim * 2
                    let nBytes = kReal * headDim * 2
                    blit.copy(
                        from: kRealPacked.buffer, sourceOffset: kSrcOff,
                        to: kPacked.buffer, destinationOffset: kDstOff, size: nBytes)
                    blit.copy(
                        from: vRealPacked.buffer, sourceOffset: vSrcOff,
                        to: vPacked.buffer, destinationOffset: vDstOff, size: nBytes)
                }
                blit.endEncoding()
            }
        }
        if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_kv_compact", cmd) }

        _ = Ops.sdpaPrefillMma(
            q: qHTD, k: kPacked, v: vPacked,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            qLen: tPad, kLen: kPad, scale: scale, on: cmd, into: outHTD)
        if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_mma_kernel", cmd) }

        _ = Ops.transposeHTD_f16(outHTD, t: tPad, h: nHeads, d: headDim, on: cmd, into: outTHD)

        // Cast only the real T rows back to bf16.
        let realOutF16 = Tensor(
            buffer: outTHD.buffer, offset: outTHD.offset,
            shape: [nRows * nHeads * headDim], dtype: .f16)
        castF16ToBf16(
            realOutF16,
            into: outBf.reshaped(to: [nRows * nHeads * headDim]),
            on: cmd, device: device)
        if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_out_transpose_cast", cmd) }
        return outBf
    }

    // T30: prefill fused QK RMSNorm + plain RoPE (sliding layers).
    //
    // One TG-per-(head,token) dispatch (128 threads = 4 simdgroups, one
    // simdgroup per head) replaces FOUR stock kernels (q RMSNorm, k
    // RMSNorm, RoPE q, RoPE k) for the whole prefill chunk AND writes Q
    // directly into the `[nHeads, tPad, headDim]` f16 layout that
    // `prefillMmaBf16` otherwise builds via a separate pad->cast->transpose
    // chain (`sdpa_qpad` + `sdpa_qcast_transpose`, T26/T27's own phase
    // breakdown: together ~2.9% of the attention wall at T=512, on top of
    // `qkv_proj`'s 24.5%/`norm`+`gate`'s mostly-overhead 38.7%), so this
    // kills those two dispatches for good, not just the norm/RoPE ones. K
    // stays `[nRows, nKVHeads, headDim]` bf16 (unchanged layout: what
    // `KVCache.appendRangeOnGPUMany` expects), so cache-append code is
    // untouched.
    //
    // Exactness, matching the decode twin `LagunaFusedQKNormRoPE` (which
    // itself matches `iron_rms_norm_bf16` / `iron_rope_banded_bf16` under
    // `RoPEScaling.none`) element for element: RMSNorm rounding
    // `bfloat((x*inv)*w)`, rotate-half pairs `p ↔ p+64`,
    // `inv_freq = exp2(-p·log2(theta)/64)`. Q's HTD-f16 write additionally
    // reproduces the *two* roundings the stock chain pays (bf16 RoPE
    // output, then `Ops.castBf16ToF16`) via `half(float(bfloat(...)))`
    // instead of writing `half(...)` directly from the float rotation:
    // dropping that first rounding would be a numerically different
    // (if probably immaterial) result from the un-fused path.
    //
    // Sliding layers only (full rotary, headDim==128). The YaRN
    // full-attention twin is not ported this session, see
    // benchmarks/laguna-xs/README.md T30.
    //
    // Two-phase API by design: Phase 1 (`fusedQKNormRopeForPrefill`) is
    // pure (no cache mutation) so a caller's eligibility pre-check
    // (mirroring `prefillMmaBf16`'s own `kPad <= maxSeq` guard) is safe to
    // do BEFORE calling it; Phase 2 (`prefillMmaTailFromHTD`) runs after
    // the caller has appended the fused K to the KV cache, so neither
    // phase needs to return an Optional / partially-mutate-then-bail.
    // Escape: BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE=0.

    static var prefillQkRopeFuseEnabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE"] != "0"
    }

    /// Shape/dtype/env eligibility only. Caller must ALSO check
    /// `position + tPad <= maxSeq` (`tPad = ((nRows+31)/32)*32`) and that
    /// the sliding-window path isn't actually engaged
    /// (`cache.sdpaSinkWindow` both zero) before committing, see the
    /// call site in `LagunaLayer.forwardMulti`.
    static func prefillQkRopeFuseEligible(
        isSliding: Bool, rotaryDim: Int, headDim: Int, dtype: DType,
        nHeads: Int, nKVHeads: Int, nRows: Int
    ) -> Bool {
        prefillQkRopeFuseEnabled
            && isSliding
            && rotaryDim == headDim
            && headDim == 128
            && dtype == .bf16
            && (nHeads + nKVHeads) % 4 == 0
            && prefillMmaEnabled
            && nRows >= 32
    }

    private static let prefillFusedQKRopeSrc = """
    #include <metal_stdlib>
    using namespace metal;

    // One threadgroup (128 threads = 4 simdgroups) per (head-block, token);
    // each simdgroup owns one (token, head) row. grid.y = nRows (real
    // tokens only, the padded tail is zeroed by a separate tiny kernel).
    kernel void laguna_prefill_sliding_qk_norm_rope_direct_bf16_128(
        device const bfloat *raw_q [[buffer(0)]],
        device const bfloat *raw_k [[buffer(1)]],
        device const bfloat *q_weight [[buffer(2)]],
        device const bfloat *k_weight [[buffer(3)]],
        device half *out_q_htd [[buffer(4)]],
        device bfloat *out_k [[buffer(5)]],
        constant uint &n_q_heads [[buffer(6)]],
        constant uint &n_kv_heads [[buffer(7)]],
        constant uint &t_pad [[buffer(8)]],
        constant uint &position_base [[buffer(9)]],
        constant float &theta_base [[buffer(10)]],
        constant float &eps [[buffer(11)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint simd_index [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        constexpr uint head_dim = 128u;
        constexpr uint half_dim = 64u;

        uint t = tgpig.y;
        uint head = tgpig.x * 4u + simd_index;
        bool is_q = head < n_q_heads;

        const device bfloat *input;
        const device bfloat *weight;
        if (is_q) {
            input = raw_q + (t * n_q_heads + head) * head_dim;
            weight = q_weight;
        } else {
            uint kh = head - n_q_heads;
            input = raw_k + (t * n_kv_heads + kh) * head_dim;
            weight = k_weight;
        }

        uint base = lane * 4u;
        float x[4];
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            x[i] = float(input[base + i]);
            sum += x[i] * x[i];
        }
        // simd_sum returns the total to every lane, no threadgroup barrier.
        sum = simd_sum(sum);
        float inv_rms = rsqrt(sum / float(head_dim) + eps);

        thread bfloat normalized[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            float wi = float(weight[base + i]);
            normalized[i] = bfloat((x[i] * inv_rms) * wi);
        }

        // Partner of pair p lives at p+64 → 16 lanes away (64/4).
        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16u);
        }

        // Lower 16 lanes own all 64 pairs; write both halves.
        if (lane < 16u) {
            float log_theta = log2(theta_base);
            float half_f = float(half_dim);
            float pos_f = float(position_base + t);
            if (is_q) {
                // HTD-f16, padded layout: [head][t][d].
                device half *out = out_q_htd + (head * t_pad + t) * head_dim;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    uint pair = base + i;
                    float first = float(normalized[i]);
                    float second = paired[i];
                    float inv_freq = exp2(((-float(pair)) * log_theta) / half_f);
                    float th = pos_f * inv_freq;
                    float c = cos(th);
                    float s = sin(th);
                    // Double-round (bf16 then f16) to match the un-fused
                    // chain's RoPE-bf16-store → castBf16ToF16 sequence.
                    bfloat r0 = bfloat(first * c - second * s);
                    bfloat r1 = bfloat(first * s + second * c);
                    out[pair] = half(float(r0));
                    out[pair + half_dim] = half(float(r1));
                }
            } else {
                // THD-bf16 layout: [t][kh][d], matches KV-cache append.
                uint kh = head - n_q_heads;
                device bfloat *out = out_k + (t * n_kv_heads + kh) * head_dim;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    uint pair = base + i;
                    float first = float(normalized[i]);
                    float second = paired[i];
                    float inv_freq = exp2(((-float(pair)) * log_theta) / half_f);
                    float th = pos_f * inv_freq;
                    float c = cos(th);
                    float s = sin(th);
                    out[pair] = bfloat(first * c - second * s);
                    out[pair + half_dim] = bfloat(first * s + second * c);
                }
            }
        }
    }
    """

    private static let prefillQhtdZeroPadSrc = """
    #include <metal_stdlib>
    using namespace metal;
    // Zero the [n_heads, t_pad-real_t, head_dim] tail of a
    // [n_heads, t_pad, head_dim] f16 buffer starting at row `real_t`.
    kernel void laguna_prefill_qhtd_zero_pad_f16(
        device half *out_q_htd [[buffer(0)]],
        constant uint &t_pad [[buffer(1)]],
        constant uint &real_t [[buffer(2)]],
        constant uint &head_dim [[buffer(3)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint head = gid.x;
        uint t = real_t + gid.y;
        uint d = gid.z;
        out_q_htd[(head * t_pad + t) * head_dim + d] = half(0.0);
    }
    """

    /// Phase 1 (pure, no cache mutation). Builds `qHTD` `[nHeads, tPad,
    /// headDim]` f16 (zero-padded past `nRows`) and `kOut` `[nRows,
    /// nKVHeads, headDim]` bf16 in one dispatch (plus a tiny zero-pad
    /// dispatch when `tPad > nRows`). The returned `slot` must be passed
    /// to `prefillMmaTailFromHTD` so both share the same scratch-ring
    /// generation.
    static func fusedQKNormRopeForPrefill(
        rawQ: Tensor, rawK: Tensor,
        qWeight: Tensor, kWeight: Tensor,
        nRows: Int, tPad: Int, position: Int,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        thetaBase: Float, eps: Float,
        on cmd: MTLCommandBuffer, device: Device
    ) -> (qHTD: Tensor, kOut: Tensor, slot: Int) {
        precondition(headDim == 128, "fusedQKNormRopeForPrefill: headDim must be 128")
        precondition(
            (nHeads + nKVHeads) % 4 == 0,
            "fusedQKNormRopeForPrefill: nHeads+nKVHeads must be a multiple of 4")
        precondition(rawQ.dtype == .bf16 && rawK.dtype == .bf16)
        precondition(rawQ.elementCount == nRows * nHeads * headDim)
        precondition(rawK.elementCount == nRows * nKVHeads * headDim)

        let slot = nextSlot()
        let qHTD = scratch(
            tag: "laguna_sdpa_qhtd", shape: [nHeads, tPad, headDim],
            dtype: .f16, device: device, slot: slot)
        // Owned (not ring): caller holds this across the KV-append step.
        let kOut = Tensor.empty(
            shape: [nRows, nKVHeads * headDim], dtype: .bf16, device: device)

        let p = pso(
            "laguna_prefill_sliding_qk_norm_rope_direct_bf16_128",
            source: prefillFusedQKRopeSrc, device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(rawQ.buffer, offset: rawQ.offset, index: 0)
            enc.setBuffer(rawK.buffer, offset: rawK.offset, index: 1)
            enc.setBuffer(qWeight.buffer, offset: qWeight.offset, index: 2)
            enc.setBuffer(kWeight.buffer, offset: kWeight.offset, index: 3)
            enc.setBuffer(qHTD.buffer, offset: qHTD.offset, index: 4)
            enc.setBuffer(kOut.buffer, offset: kOut.offset, index: 5)
            var nQU = UInt32(nHeads)
            var nKvU = UInt32(nKVHeads)
            var tPadU = UInt32(tPad)
            var posU = UInt32(position)
            var theta = thetaBase
            var epsV = eps
            enc.setBytes(&nQU, length: 4, index: 6)
            enc.setBytes(&nKvU, length: 4, index: 7)
            enc.setBytes(&tPadU, length: 4, index: 8)
            enc.setBytes(&posU, length: 4, index: 9)
            enc.setBytes(&theta, length: 4, index: 10)
            enc.setBytes(&epsV, length: 4, index: 11)
            let groups = (nHeads + nKVHeads) / 4
            enc.dispatchThreadgroups(
                MTLSize(width: groups, height: nRows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
        }

        if tPad > nRows {
            let zp = pso(
                "laguna_prefill_qhtd_zero_pad_f16",
                source: prefillQhtdZeroPadSrc, device: device.mtlDevice)
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(zp)
                enc.setBuffer(qHTD.buffer, offset: qHTD.offset, index: 0)
                var tPadU = UInt32(tPad)
                var realTU = UInt32(nRows)
                var hdU = UInt32(headDim)
                enc.setBytes(&tPadU, length: 4, index: 1)
                enc.setBytes(&realTU, length: 4, index: 2)
                enc.setBytes(&hdU, length: 4, index: 3)
                enc.dispatchThreads(
                    MTLSize(width: nHeads, height: tPad - nRows, depth: headDim),
                    threadsPerThreadgroup: MTLSize(
                        width: 1, height: 1, depth: min(headDim, 64)))
                enc.endEncoding()
            }
        }
        return (qHTD, kOut, slot)
    }

    /// Phase 2: KV compaction + MMA + transpose/cast back to bf16, given a
    /// precomputed `qHTD` (from `fusedQKNormRopeForPrefill`). Same tail
    /// `prefillMmaBf16` runs after its own pad/cast/transpose stages,
    /// duplicated rather than factored out of `prefillMmaBf16` to keep
    /// this session's diff small and the existing (well-gated) function
    /// untouched.
    static func prefillMmaTailFromHTD(
        qHTD: Tensor, tPad: Int, slot: Int,
        cacheK: Tensor, cacheV: Tensor,
        nRows: Int, position: Int,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        maxSeq: Int, scale: Float,
        on cmd: inout MTLCommandBuffer, device: Device,
        phaseCheckpoint: ((String, MTLCommandBuffer) -> MTLCommandBuffer)? = nil
    ) -> Tensor {
        let kReal = position + nRows
        let kPad = position + tPad
        precondition(kPad <= maxSeq, "prefillMmaTailFromHTD: kPad \(kPad) > maxSeq \(maxSeq)")

        let outHTD = scratch(
            tag: "laguna_sdpa_outhtd", shape: [nHeads, tPad, headDim],
            dtype: .f16, device: device, slot: slot)
        let outTHD = scratch(
            tag: "laguna_sdpa_outthd", shape: [tPad, nHeads, headDim],
            dtype: .f16, device: device, slot: slot)
        let outBf = Tensor.empty(
            shape: [nRows, nHeads * headDim], dtype: .bf16, device: device)
        let kPacked = scratch(
            tag: "laguna_sdpa_kpack", shape: [nKVHeads, kPad, headDim],
            dtype: .f16, device: device, slot: slot)
        let vPacked = scratch(
            tag: "laguna_sdpa_vpack", shape: [nKVHeads, kPad, headDim],
            dtype: .f16, device: device, slot: slot)

        if kPad == kReal {
            if let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) {
                _ = compactKvBf16ToF16(
                    src: cacheK, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: kPacked, encoder: enc)
                _ = compactKvBf16ToF16(
                    src: cacheV, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: vPacked, encoder: enc)
                enc.endEncoding()
            } else {
                _ = compactKvBf16ToF16(
                    src: cacheK, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: kPacked)
                _ = compactKvBf16ToF16(
                    src: cacheV, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: vPacked)
            }
        } else {
            let kRealPacked = scratch(
                tag: "laguna_sdpa_kreal", shape: [nKVHeads, kReal, headDim],
                dtype: .f16, device: device, slot: slot)
            let vRealPacked = scratch(
                tag: "laguna_sdpa_vreal", shape: [nKVHeads, kReal, headDim],
                dtype: .f16, device: device, slot: slot)
            if let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent) {
                _ = compactKvBf16ToF16(
                    src: cacheK, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: kRealPacked, encoder: enc)
                _ = compactKvBf16ToF16(
                    src: cacheV, nKV: nKVHeads, maxSeq: maxSeq, kLen: kReal,
                    headDim: headDim, on: cmd, device: device, into: vRealPacked, encoder: enc)
                enc.endEncoding()
            }
            if let blit = cmd.makeBlitCommandEncoder() {
                let padBytes = nKVHeads * kPad * headDim * 2
                blit.fill(
                    buffer: kPacked.buffer,
                    range: kPacked.offset ..< (kPacked.offset + padBytes), value: 0)
                blit.fill(
                    buffer: vPacked.buffer,
                    range: vPacked.offset ..< (vPacked.offset + padBytes), value: 0)
                for h in 0 ..< nKVHeads {
                    let kSrcOff = kRealPacked.offset + h * kReal * headDim * 2
                    let vSrcOff = vRealPacked.offset + h * kReal * headDim * 2
                    let kDstOff = kPacked.offset + h * kPad * headDim * 2
                    let vDstOff = vPacked.offset + h * kPad * headDim * 2
                    let nBytes = kReal * headDim * 2
                    blit.copy(
                        from: kRealPacked.buffer, sourceOffset: kSrcOff,
                        to: kPacked.buffer, destinationOffset: kDstOff, size: nBytes)
                    blit.copy(
                        from: vRealPacked.buffer, sourceOffset: vSrcOff,
                        to: vPacked.buffer, destinationOffset: vDstOff, size: nBytes)
                }
                blit.endEncoding()
            }
        }
        if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_kv_compact", cmd) }

        _ = Ops.sdpaPrefillMma(
            q: qHTD, k: kPacked, v: vPacked,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            qLen: tPad, kLen: kPad, scale: scale, on: cmd, into: outHTD)
        if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_mma_kernel", cmd) }

        _ = Ops.transposeHTD_f16(outHTD, t: tPad, h: nHeads, d: headDim, on: cmd, into: outTHD)
        let realOutF16 = Tensor(
            buffer: outTHD.buffer, offset: outTHD.offset,
            shape: [nRows * nHeads * headDim], dtype: .f16)
        castF16ToBf16(
            realOutF16,
            into: outBf.reshaped(to: [nRows * nHeads * headDim]),
            on: cmd, device: device)
        if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_out_transpose_cast", cmd) }
        return outBf
    }

    // T30 stretch: prefill fused QK RMSNorm + partial-rotary YaRN
    // RoPE (full-attention layers).
    //
    // Twin of the sliding fuse above for full-attention layers (partial
    // rotary, `rotaryDim < headDim`, YaRN mscale). The ROTATION MATH
    // matches butter's *own* stock kernel (`iron_rope_yarn_bf16`, used
    // per-row by `lagunaRopeYaRNPartial`): it folds the YaRN `attn_factor`
    // into `cos`/`sin` once, rather than pre-scaling the rotation input
    // through an extra bf16 rounding step first. The two approaches are
    // mathematically equivalent in exact arithmetic, NOT bit-identical
    // under bf16 rounding, so matching butter's own kernel is the correct
    // target here, confirmed against a probe test before wiring in (see
    // `LagunaPrefillFusedQKRopeProbeTests`).
    //
    // The addressable win here is structurally larger per-layer than the
    // sliding fuse despite touching only 7/40 layers: `lagunaBatchedRopeAndKV`'s
    // `isFullYarn` branch calls `lagunaRopeYaRNPartial` ONCE PER ROW (not
    // batched like the sliding path's `Ops.ropePartialManyTwo`), so 2×nRows
    // tiny dispatches per full-attention layer (1024 at T=512) collapse to
    // one dispatch here.
    //
    // Same two-phase split as the sliding twin; escape shares the same
    // `BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE` flag (no new env var).

    static func prefillQkYarnFuseEligible(
        isSliding: Bool, rotaryDim: Int, headDim: Int, dtype: DType,
        nHeads: Int, nKVHeads: Int, nRows: Int
    ) -> Bool {
        prefillQkRopeFuseEnabled
            && !isSliding
            && rotaryDim < headDim
            && rotaryDim % 2 == 0
            && headDim == 128
            && dtype == .bf16
            && (nHeads + nKVHeads) % 4 == 0
            && prefillMmaEnabled
            && nRows >= 32
    }

    private static let prefillFusedQKYarnSrc = """
    #include <metal_stdlib>
    using namespace metal;

    // One threadgroup (128 threads = 4 simdgroups) per (head-block, token);
    // each simdgroup owns one (token, head) row. `rotary_pairs` = rotaryDim/2
    // (< 64 for partial rotary); pairs [0,rotary_pairs) rotate, elements
    // [2*rotary_pairs, head_dim) pass through unrotated (already RMSNorm'd).
    kernel void laguna_prefill_full_qk_norm_yarn_direct_bf16_128(
        device const bfloat *raw_q [[buffer(0)]],
        device const bfloat *raw_k [[buffer(1)]],
        device const bfloat *q_weight [[buffer(2)]],
        device const bfloat *k_weight [[buffer(3)]],
        device half *out_q_htd [[buffer(4)]],
        device bfloat *out_k [[buffer(5)]],
        constant uint &n_q_heads [[buffer(6)]],
        constant uint &n_kv_heads [[buffer(7)]],
        constant uint &t_pad [[buffer(8)]],
        constant uint &position_base [[buffer(9)]],
        constant float &theta_base [[buffer(10)]],
        constant float &eps [[buffer(11)]],
        constant uint &rotary_pairs [[buffer(12)]],
        constant float &yarn_factor [[buffer(13)]],
        constant float &yarn_low [[buffer(14)]],
        constant float &yarn_high [[buffer(15)]],
        constant float &yarn_attn_factor [[buffer(16)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint simd_index [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        constexpr uint head_dim = 128u;

        uint t = tgpig.y;
        uint head = tgpig.x * 4u + simd_index;
        bool is_q = head < n_q_heads;

        const device bfloat *input;
        const device bfloat *weight;
        if (is_q) {
            input = raw_q + (t * n_q_heads + head) * head_dim;
            weight = q_weight;
        } else {
            uint kh = head - n_q_heads;
            input = raw_k + (t * n_kv_heads + kh) * head_dim;
            weight = k_weight;
        }

        uint base = lane * 4u;
        float x[4];
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            x[i] = float(input[base + i]);
            sum += x[i] * x[i];
        }
        sum = simd_sum(sum);
        float inv_rms = rsqrt(sum / float(head_dim) + eps);

        thread bfloat normalized[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            float wi = float(weight[base + i]);
            normalized[i] = bfloat((x[i] * inv_rms) * wi);
        }

        // rotary_pairs elements per half; partner of pair p lives at
        // p+rotary_pairs → rotary_pairs/4 lanes away.
        uint shuffle_dist = rotary_pairs / 4u;
        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ shuffle_dist);
        }

        uint rotary_lanes = rotary_pairs / 4u;  // lanes [0, rotary_lanes) rotate
        if (lane < rotary_lanes) {
            float log_theta = log2(theta_base);
            float half_f = float(rotary_pairs);
            float pos_f = float(position_base + t);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                uint pair = base + i;
                float pair_f = float(pair);
                float inv_freq_extrap = exp2(((-pair_f) * log_theta) / half_f);
                float t_ramp = (pair_f - yarn_low) / (yarn_high - yarn_low);
                float ramp = select(t_ramp, 1.0f, bool(t_ramp > 1.0f));
                ramp = select(ramp, 0.0f, bool(0.0f > ramp));
                float inv_freq =
                    (inv_freq_extrap / yarn_factor) * ramp
                    + inv_freq_extrap * (1.0f - ramp);
                float theta = pos_f * inv_freq;
                float cos_t = cos(theta) * yarn_attn_factor;
                float sin_t = sin(theta) * yarn_attn_factor;
                float first = float(normalized[i]);
                float second = paired[i];
                bfloat r0 = bfloat(first * cos_t - second * sin_t);
                bfloat r1 = bfloat(first * sin_t + second * cos_t);
                if (is_q) {
                    device half *out = out_q_htd + (head * t_pad + t) * head_dim;
                    out[pair] = half(float(r0));
                    out[pair + rotary_pairs] = half(float(r1));
                } else {
                    uint kh = head - n_q_heads;
                    device bfloat *out = out_k + (t * n_kv_heads + kh) * head_dim;
                    out[pair] = r0;
                    out[pair + rotary_pairs] = r1;
                }
            }
        } else if (lane >= 2u * rotary_lanes) {
            // Tail elements [2*rotary_pairs, head_dim): pass through
            // the RMSNorm'd value verbatim, no rotation.
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                if (is_q) {
                    device half *out = out_q_htd + (head * t_pad + t) * head_dim;
                    out[base + i] = half(float(normalized[i]));
                } else {
                    uint kh = head - n_q_heads;
                    device bfloat *out = out_k + (t * n_kv_heads + kh) * head_dim;
                    out[base + i] = normalized[i];
                }
            }
        }
    }
    """

    /// Phase 1 (pure) for full-attention (YaRN, partial-rotary) layers,
    /// same contract as `fusedQKNormRopeForPrefill`.
    static func fusedQKNormYarnForPrefill(
        rawQ: Tensor, rawK: Tensor,
        qWeight: Tensor, kWeight: Tensor,
        nRows: Int, tPad: Int, position: Int,
        nHeads: Int, nKVHeads: Int, headDim: Int, rotaryDim: Int,
        thetaBase: Float, eps: Float, yarn: Ops.RoPEYaRN,
        on cmd: MTLCommandBuffer, device: Device
    ) -> (qHTD: Tensor, kOut: Tensor, slot: Int) {
        precondition(headDim == 128, "fusedQKNormYarnForPrefill: headDim must be 128")
        precondition(rotaryDim > 0 && rotaryDim < headDim && rotaryDim % 2 == 0)
        precondition(
            (nHeads + nKVHeads) % 4 == 0,
            "fusedQKNormYarnForPrefill: nHeads+nKVHeads must be a multiple of 4")
        precondition(rawQ.dtype == .bf16 && rawK.dtype == .bf16)
        precondition(rawQ.elementCount == nRows * nHeads * headDim)
        precondition(rawK.elementCount == nRows * nKVHeads * headDim)
        let rotaryPairs = rotaryDim / 2
        precondition(
            rotaryPairs % 4 == 0,
            "fusedQKNormYarnForPrefill: rotaryDim/2 must be a multiple of 4")

        let slot = nextSlot()
        let qHTD = scratch(
            tag: "laguna_sdpa_qhtd", shape: [nHeads, tPad, headDim],
            dtype: .f16, device: device, slot: slot)
        let kOut = Tensor.empty(
            shape: [nRows, nKVHeads * headDim], dtype: .bf16, device: device)

        let p = pso(
            "laguna_prefill_full_qk_norm_yarn_direct_bf16_128",
            source: prefillFusedQKYarnSrc, device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(rawQ.buffer, offset: rawQ.offset, index: 0)
            enc.setBuffer(rawK.buffer, offset: rawK.offset, index: 1)
            enc.setBuffer(qWeight.buffer, offset: qWeight.offset, index: 2)
            enc.setBuffer(kWeight.buffer, offset: kWeight.offset, index: 3)
            enc.setBuffer(qHTD.buffer, offset: qHTD.offset, index: 4)
            enc.setBuffer(kOut.buffer, offset: kOut.offset, index: 5)
            var nQU = UInt32(nHeads)
            var nKvU = UInt32(nKVHeads)
            var tPadU = UInt32(tPad)
            var posU = UInt32(position)
            var theta = thetaBase
            var epsV = eps
            var rotaryPairsU = UInt32(rotaryPairs)
            var yarnFactor = yarn.factor
            var yarnLow = yarn.low
            var yarnHigh = yarn.high
            var yarnAttnFactor = yarn.attnFactor
            enc.setBytes(&nQU, length: 4, index: 6)
            enc.setBytes(&nKvU, length: 4, index: 7)
            enc.setBytes(&tPadU, length: 4, index: 8)
            enc.setBytes(&posU, length: 4, index: 9)
            enc.setBytes(&theta, length: 4, index: 10)
            enc.setBytes(&epsV, length: 4, index: 11)
            enc.setBytes(&rotaryPairsU, length: 4, index: 12)
            enc.setBytes(&yarnFactor, length: 4, index: 13)
            enc.setBytes(&yarnLow, length: 4, index: 14)
            enc.setBytes(&yarnHigh, length: 4, index: 15)
            enc.setBytes(&yarnAttnFactor, length: 4, index: 16)
            let groups = (nHeads + nKVHeads) / 4
            enc.dispatchThreadgroups(
                MTLSize(width: groups, height: nRows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
        }

        if tPad > nRows {
            let zp = pso(
                "laguna_prefill_qhtd_zero_pad_f16",
                source: prefillQhtdZeroPadSrc, device: device.mtlDevice)
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(zp)
                enc.setBuffer(qHTD.buffer, offset: qHTD.offset, index: 0)
                var tPadU2 = UInt32(tPad)
                var realTU = UInt32(nRows)
                var hdU = UInt32(headDim)
                enc.setBytes(&tPadU2, length: 4, index: 1)
                enc.setBytes(&realTU, length: 4, index: 2)
                enc.setBytes(&hdU, length: 4, index: 3)
                enc.dispatchThreads(
                    MTLSize(width: nHeads, height: tPad - nRows, depth: headDim),
                    threadsPerThreadgroup: MTLSize(
                        width: 1, height: 1, depth: min(headDim, 64)))
                enc.endEncoding()
            }
        }
        return (qHTD, kOut, slot)
    }
}

// ─── Decode fused QK RMSNorm + plain RoPE (sliding layers, T=1) ──────
//
// Replaces four stock dispatches (q RMSNorm, k RMSNorm, RoPE q, RoPE k)
// with one TG-per-head kernel. Mirrors mlxfast `laguna_sliding_qk_norm_rope`
// (head_dim=128, full rotary, rotate-half pairs p ↔ p+64):
//   • RMSNorm: 4 elements/lane, simd_sum SSQ, local rsqrt — NO barrier
//     (simd_sum already returns the total to every lane).
//   • Norm rounding matches iron_rms_norm: bfloat((x * inv) * w).
//   • RoPE matches iron_rope_banded with RoPEScaling.none: inv_freq =
//     exp2((-p) * log2(theta) / 64), pair layout [p, p+64].
// Escape: BUTTER_LAGUNA_FUSED_QK_NORM_ROPE=0.

enum LagunaFusedQKNormRoPE {
    /// Default ON (`!= "0"`). Set `BUTTER_LAGUNA_FUSED_QK_NORM_ROPE=0` to ablate.
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BUTTER_LAGUNA_FUSED_QK_NORM_ROPE"] != "0"
    }

    /// Sliding-layer decode only: full rotary, head_dim 128, bf16/f16.
    static func canFuse(
        isSliding: Bool, rotaryDim: Int, headDim: Int, dtype: DType
    ) -> Bool {
        enabled
            && isSliding
            && rotaryDim == headDim
            && headDim == 128
            && (dtype == .bf16 || dtype == .f16)
    }

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        let lib = try! device.makeLibrary(source: source, options: MTLCompileOptions())
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    private static let bf16Src = """
    #include <metal_stdlib>
    using namespace metal;

    // One threadgroup (32 lanes = 1 simdgroup) per Q or K head.
    // head 0..n_q_heads-1 → Q; head n_q_heads.. → K.
    kernel void laguna_sliding_qk_norm_rope_bf16_128(
        device const bfloat *raw_q [[buffer(0)]],
        device const bfloat *raw_k [[buffer(1)]],
        device const bfloat *q_weight [[buffer(2)]],
        device const bfloat *k_weight [[buffer(3)]],
        device bfloat *out_q [[buffer(4)]],
        device bfloat *out_k [[buffer(5)]],
        constant uint &n_q_heads [[buffer(6)]],
        constant uint &position [[buffer(7)]],
        constant float &theta_base [[buffer(8)]],
        constant float &eps [[buffer(9)]],
        uint head [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        constexpr uint head_dim = 128u;
        constexpr uint half_dim = 64u;

        const device bfloat *input;
        const device bfloat *weight;
        device bfloat *output;
        if (head < n_q_heads) {
            input = raw_q + head * head_dim;
            weight = q_weight;
            output = out_q + head * head_dim;
        } else {
            uint kh = head - n_q_heads;
            input = raw_k + kh * head_dim;
            weight = k_weight;
            output = out_k + kh * head_dim;
        }

        // 4 contiguous elements per lane (128 / 32).
        uint base = lane * 4u;
        float x[4];
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            x[i] = float(input[base + i]);
            sum += x[i] * x[i];
        }
        // simd_sum returns the total to every lane, no threadgroup barrier.
        sum = simd_sum(sum);
        float inv_rms = rsqrt(sum / float(head_dim) + eps);

        // Iron-order intermediate: bfloat((x * inv) * w), then back to f32
        // so RoPE sees the same rounding stock rmsNorm → rope would.
        thread bfloat normalized[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            float wi = float(weight[base + i]);
            normalized[i] = bfloat((x[i] * inv_rms) * wi);
        }

        // Partner of pair p lives at p+64 → 16 lanes away (64/4).
        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16u);
        }

        // Lower 16 lanes own all 64 pairs; write both halves.
        if (lane < 16u) {
            float log_theta = log2(theta_base);
            float half_f = float(half_dim);
            float pos_f = float(position);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                uint pair = base + i;
                float first = float(normalized[i]);
                float second = paired[i];
                // iron_rope_banded inv_freq with RoPEScaling.none (scale=1):
                // inv = exp2((-pair) * log2(theta) / half_dim)
                float inv_freq = exp2(((-float(pair)) * log_theta) / half_f);
                float th = pos_f * inv_freq;
                float c = cos(th);
                float s = sin(th);
                output[pair] = bfloat(first * c - second * s);
                output[pair + half_dim] = bfloat(first * s + second * c);
            }
        }
    }
    """

    private static let f16Src = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void laguna_sliding_qk_norm_rope_f16_128(
        device const half *raw_q [[buffer(0)]],
        device const half *raw_k [[buffer(1)]],
        device const half *q_weight [[buffer(2)]],
        device const half *k_weight [[buffer(3)]],
        device half *out_q [[buffer(4)]],
        device half *out_k [[buffer(5)]],
        constant uint &n_q_heads [[buffer(6)]],
        constant uint &position [[buffer(7)]],
        constant float &theta_base [[buffer(8)]],
        constant float &eps [[buffer(9)]],
        uint head [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        constexpr uint head_dim = 128u;
        constexpr uint half_dim = 64u;

        const device half *input;
        const device half *weight;
        device half *output;
        if (head < n_q_heads) {
            input = raw_q + head * head_dim;
            weight = q_weight;
            output = out_q + head * head_dim;
        } else {
            uint kh = head - n_q_heads;
            input = raw_k + kh * head_dim;
            weight = k_weight;
            output = out_k + kh * head_dim;
        }

        uint base = lane * 4u;
        float x[4];
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            x[i] = float(input[base + i]);
            sum += x[i] * x[i];
        }
        sum = simd_sum(sum);
        float inv_rms = rsqrt(sum / float(head_dim) + eps);

        thread half normalized[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            float wi = float(weight[base + i]);
            normalized[i] = half((x[i] * inv_rms) * wi);
        }

        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16u);
        }

        if (lane < 16u) {
            float log_theta = log2(theta_base);
            float half_f = float(half_dim);
            float pos_f = float(position);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                uint pair = base + i;
                float first = float(normalized[i]);
                float second = paired[i];
                float inv_freq = exp2(((-float(pair)) * log_theta) / half_f);
                float th = pos_f * inv_freq;
                float c = cos(th);
                float s = sin(th);
                output[pair] = half(first * c - second * s);
                output[pair + half_dim] = half(first * s + second * c);
            }
        }
    }
    """

    /// Fused per-head Q/K RMSNorm + plain RoPE for sliding decode (T=1).
    /// Writes flat `[nQHeads·128]` / `[nKVHeads·128]` outputs.
    @discardableResult
    static func apply(
        rawQ: Tensor, rawK: Tensor,
        qWeight: Tensor, kWeight: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        position: Int, thetaBase: Float, eps: Float,
        on cmd: MTLCommandBuffer, device: Device,
        intoQ: Tensor? = nil, intoK: Tensor? = nil
    ) -> (q: Tensor, k: Tensor) {
        precondition(headDim == 128)
        precondition(rawQ.elementCount == nQHeads * headDim)
        precondition(rawK.elementCount == nKVHeads * headDim)
        precondition(qWeight.elementCount == headDim && kWeight.elementCount == headDim)
        precondition(rawQ.dtype == qWeight.dtype && rawK.dtype == kWeight.dtype)
        precondition(rawQ.dtype == rawK.dtype)

        let qOut = intoQ ?? Tensor.empty(
            shape: [nQHeads * headDim], dtype: rawQ.dtype, device: device)
        let kOut = intoK ?? Tensor.empty(
            shape: [nKVHeads * headDim], dtype: rawK.dtype, device: device)
        precondition(qOut.elementCount == nQHeads * headDim)
        precondition(kOut.elementCount == nKVHeads * headDim)

        let (name, source): (String, String)
        switch rawQ.dtype {
        case .bf16:
            (name, source) = (
                "laguna_sliding_qk_norm_rope_bf16_128", bf16Src)
        case .f16:
            (name, source) = (
                "laguna_sliding_qk_norm_rope_f16_128", f16Src)
        default:
            fatalError("LagunaFusedQKNormRoPE: unsupported dtype \(rawQ.dtype)")
        }

        let p = pso(name, source: source, device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return (qOut, kOut) }
        enc.setComputePipelineState(p)
        enc.setBuffer(rawQ.buffer, offset: rawQ.offset, index: 0)
        enc.setBuffer(rawK.buffer, offset: rawK.offset, index: 1)
        enc.setBuffer(qWeight.buffer, offset: qWeight.offset, index: 2)
        enc.setBuffer(kWeight.buffer, offset: kWeight.offset, index: 3)
        enc.setBuffer(qOut.buffer, offset: qOut.offset, index: 4)
        enc.setBuffer(kOut.buffer, offset: kOut.offset, index: 5)
        var nQU = UInt32(nQHeads)
        var posU = UInt32(position)
        var theta = thetaBase
        var epsV = eps
        enc.setBytes(&nQU, length: 4, index: 6)
        enc.setBytes(&posU, length: 4, index: 7)
        enc.setBytes(&theta, length: 4, index: 8)
        enc.setBytes(&epsV, length: 4, index: 9)
        enc.dispatchThreadgroups(
            MTLSize(width: nQHeads + nKVHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        return (qOut, kOut)
    }
}
