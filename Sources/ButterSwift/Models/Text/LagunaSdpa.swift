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
        let env = ButterEnv["BUTTER_LAGUNA_SDPA_PREFILL_MMA"]
        if env == "0" { return false }
        if env == "1" { return true }
        return true  // default ON when shapes allow
    }

    // Fuse the SDPA-output layout round trip. The MMA kernel emits
    // head-major f16 `[nHeads, tPad, headDim]` (`outHTD`); o_proj consumes
    // token-major bf16 `[nRows, nHeads*headDim]`. The stock tail pays two
    // dispatches AND a full `outTHD` f16 intermediate round trip for this:
    // `transposeHTD_f16` (HTD→THD, pure layout) then `castF16ToBf16`
    // (f16→bf16, real rows only). This lever fuses both into a single
    // `laguna_transpose_htd_cast_bf16` dispatch that reads `outHTD` and
    // writes `outBf` directly — deleting one dispatch/layer and the entire
    // `outTHD` scratch write+read (tPad·nHeads·headDim f16 each way).
    //
    // BYTE-IDENTICAL: the transpose is a pure address remap (no rounding)
    // and the fused kernel reproduces the exact cast the stock chain paid,
    // `bfloat(float(outHTD[(h·tPad+t)·headDim+d]))`, over the same real
    // `[0, nRows)` rows. Default ON; escape `BUTTER_LAGUNA_SDPA_NATIVE_LAYOUT=0`
    // restores the two-dispatch transpose+cast path (for A/B / fallback).
    static var sdpaNativeLayoutEnabled: Bool {
        ButterEnv["BUTTER_LAGUNA_SDPA_NATIVE_LAYOUT"] != "0"
    }

    // ─── W58: fused SDPA output epilogue (opt-in A/B lever) ───────────
    //
    // One level deeper than `sdpaNativeLayoutEnabled`: instead of a SEPARATE
    // `laguna_transpose_htd_cast_bf16` dispatch that reads the MMA output
    // `outHTD` and writes the transposed bf16 `outBf`, the MMA prefill kernel
    // itself writes O straight to the transposed bf16 layout in its store
    // epilogue (in-register f32→f16→bf16 narrowing). This keeps the attn
    // output ON-CHIP — the `[nHeads, tPad, headDim]` f16 `outHTD` intermediate
    // is never written to (or read back from) global memory at all
    // (~2 MB/layer read+write eliminated).
    //
    // BYTE-IDENTICAL to the `sdpaNativeLayoutEnabled` path: the fused store
    // does `bfloat(static_cast<half>(o·is_row))`, the exact double-round the
    // stock chain paid (`.cast::<f16>()` in the MMA store + `bfloat(float(x))`
    // in the transpose-cast). Default OFF; opt-in
    // `BUTTER_LAGUNA_PREFILL_FUSED_EPILOGUE=1`.
    static var prefillFusedEpilogueEnabled: Bool {
        ButterEnv["BUTTER_LAGUNA_PREFILL_FUSED_EPILOGUE"] == "1"
    }

    // ─── Prefill SDPA mask block-map (opt-in) ────────────────────────
    //
    // `prefillMmaBf16`'s causal MMA path (`Ops.sdpaPrefillMma`) always
    // scans the *entire* `[0, kPad)` K range for every 32-row query
    // tile and relies on the kernel's own per-element causal test to
    // zero out the masked-off future. At T=512 causal, roughly half of
    // every (query-tile, key) pair is in the upper-triangular dead
    // region (`ki_min > qi_max`): work that can be skipped outright
    // rather than computed-then-masked, and the surviving live region
    // itself splits cleanly into a small causal-diagonal "edge" band
    // that needs the per-row test and a large "full" middle band that
    // provably doesn't.
    //
    // This section computes that closed form host-side per 32-row
    // query tile (`queryBlockBounds`) and dispatches a new
    // `laguna_sdpa_prefill_blockmap_f16_128` kernel restructured into
    // exactly those phases: dead K skipped entirely (the loop bounds
    // are `[mainStart, kvEnd)`, nothing outside is ever touched), the
    // always-live sink prefix `[0, sinkEnd)` and the two causal/window
    // edge bands run the per-row mask test, and the full middle band
    // runs with no branch at all.
    //
    // FORWARD-LOOKING CAPABILITY, not just a preserved-behavior lever:
    // sliding-window prefill is *inert* today outside this flag:
    // `KVCacheProtocol.sdpaSinkWindow` unconditionally returns `(0, 0)`
    // (see `LagunaText.swift`'s T26 comment), so `prefillMmaBf16`'s
    // causal MMA path already runs fully unwindowed for every prefill
    // layer, sliding or not. That's numerically harmless at T=512 (the
    // window is 4096, wider than the whole chunk) but would be silently
    // wrong at longer prompts. Passing a nonzero `window` through this
    // path (see `LagunaText.swift`'s call site) is what actually turns
    // sliding-window masking on at prefill for the first time, rather
    // than just re-deriving the causal-only result faster.
    //
    // SCOPE NOTE: `prefillMmaBf16`'s K/V *compaction* step
    // (`compactKvBf16ToF16`) still copies the entire `[0, kReal)` cache
    // prefix regardless of window; this lever removes dead work from
    // the attention inner loop only, not from compaction. A
    // window-bounded compaction is a natural follow-up, out of scope
    // here.
    //
    // Deliberately a hand-written *scalar* (no simdgroup_matrix) kernel,
    // one simdgroup per query row (32 simdgroups = the 32 rows of one
    // query tile, 32 lanes each over `headDim`), mirroring
    // `laguna_sdpa_decode_gqa2_bf16_128`'s per-row online-softmax math.
    // A from-scratch simdgroup_matrix MMA port of the same block-map
    // bounds would out-throughput this, but is materially higher-risk
    // to hand-write correctly without a build/test loop; this trades
    // some raw throughput for a kernel whose correctness is easy to
    // read off directly. Opt-in: `BUTTER_LAGUNA_PREFILL_BLOCKMAP=1`.
    // Default OFF: `prefillMmaBf16` keeps calling `Ops.sdpaPrefillMma`
    // unless this is set, so default behavior is unchanged.
    static var prefillBlockmapEnabled: Bool {
        ButterEnv["BUTTER_LAGUNA_PREFILL_BLOCKMAP"] == "1"
    }

    /// Closed-form per-query-block K bounds for the block-map prefill
    /// kernel. All fields are absolute KV column indices into the
    /// compacted `[nKVHeads, kPad, headDim]` buffers, forming the
    /// non-decreasing chain
    /// `0 <= sinkEnd <= mainStart <= fullLo <= fullHi <= kvEnd <= kPad`:
    ///   * `[0, sinkEnd)`         : always-live sink prefix, causal-
    ///                              masked per row (empty when sink is
    ///                              disabled).
    ///   * `[sinkEnd, mainStart)` : dead, never visited. Only nonempty
    ///                              when a small sink leaves a gap
    ///                              before the window's earliest live
    ///                              column.
    ///   * `[mainStart, fullLo)`  : edge, per-row causal+window test.
    ///   * `[fullLo, fullHi)`     : full, valid for every row in the
    ///                              tile, no per-row test.
    ///   * `[fullHi, kvEnd)`      : edge, per-row causal+window test
    ///                              (the causal diagonal band).
    ///   * `[kvEnd, kPad)`        : dead, never visited (strictly
    ///                              future).
    struct PrefillBlockmapBounds: Equatable {
        var sinkEnd: Int
        var mainStart: Int
        var fullLo: Int
        var fullHi: Int
        var kvEnd: Int
    }

    /// Row validity, matching the convention already used throughout
    /// Butter's SDPA stack (`KVEvictionState`/`Ops.sdpaDecode`'s
    /// `windowStart = max(0, nKV - window)`, `nKV = q + 1`): query row
    /// at absolute position `q` may attend to key column `k` iff
    /// `k <= q` (causal) AND (`k < sinkEnd` (sink override) OR
    /// `window <= 0` (no window) OR `k >= q - window + 1` (in-window)).
    /// `window <= 0` disables the sliding-window rule entirely (pure
    /// causal); `sinkEnd <= 0` disables the sink exception.
    ///
    /// For one query tile spanning absolute rows `[qiMin, qiMax]`
    /// (`qiMax == qiMin + 31` for a full 32-row tile), returns the
    /// tightest bounds such that: every `(q, k)` pair with `k` outside
    /// `[0, sinkEnd) ∪ [mainStart, kvEnd)` is invalid for every row in
    /// the tile (safe to skip), and every `(q, k)` pair with `k` inside
    /// `[fullLo, fullHi)` is valid for every row in the tile (safe to
    /// skip the mask test). See
    /// `Tests/ButterTests/Ops/LagunaPrefillBlockmapTests.swift` for the
    /// brute-force proof of both claims.
    static func queryBlockBounds(
        qiMin: Int, qiMax: Int, kPad: Int, window: Int, sinkEnd rawSinkEnd: Int
    ) -> PrefillBlockmapBounds {
        precondition(qiMin <= qiMax, "queryBlockBounds: qiMin \(qiMin) > qiMax \(qiMax)")
        func clamp(_ x: Int, _ lo: Int, _ hi: Int) -> Int { min(max(x, lo), hi) }

        // Causal upper bound: no row in the tile can see past its own
        // position, and no row's position exceeds `qiMax`.
        let kvEnd = max(0, min(kPad, qiMax + 1))
        let sinkEnd = max(0, min(rawSinkEnd, kvEnd))
        // Window lower bound computed from the tile's SMALLEST row
        // (`qiMin`): that row has the loosest (earliest) window start
        // of anyone in the tile, so it's the correct "might still be
        // live for someone" floor.
        let windowStart = window > 0 ? max(0, qiMin - window + 1) : 0
        let mainStart = clamp(max(windowStart, sinkEnd), sinkEnd, kvEnd)

        // fullLo: window becomes unconditionally satisfied for every
        // row once `k >= qiMax - window + 1` (the tile's LARGEST row,
        // i.e. tightest/latest window start, is satisfied, which
        // implies every looser row is too).
        var fullLo = window > 0 ? clamp(qiMax - window + 1, mainStart, kvEnd) : mainStart
        // fullHi: causal becomes unconditionally satisfied for every
        // row only while `k <= qiMin` (the tile's SMALLEST row is the
        // tightest causal constraint).
        var fullHi = clamp(qiMin + 1, mainStart, kvEnd)
        if fullLo > fullHi { fullLo = fullHi }  // narrow window vs. wide tile

        return PrefillBlockmapBounds(
            sinkEnd: sinkEnd, mainStart: mainStart, fullLo: fullLo, fullHi: fullHi, kvEnd: kvEnd)
    }

    /// Coarse per-(query-block, key-block) classification at an
    /// arbitrary key-tile size: the "block (qi,ki) fully dead iff..."
    /// form callers reason about at a glance. Deliberately independent
    /// of `queryBlockBounds` (different granularity, different
    /// derivation) so the two can cross-check each other against the
    /// same brute-force per-element scan in tests.
    enum BlockClass: Equatable { case dead, full, edge }

    static func classifyBlock(
        qiMin: Int, qiMax: Int, kiMin: Int, kiMax: Int, window: Int, sinkEnd: Int
    ) -> BlockClass {
        let overlapsSink = sinkEnd > 0 && kiMin < sinkEnd
        // Dead: strictly future (causal), or strictly window-expired
        // AND not exempted by an overlapping sink prefix.
        if kiMin > qiMax { return .dead }
        if window > 0, kiMax < qiMin - window + 1, !overlapsSink { return .dead }
        // Full: causal-safe for every row (largest k <= smallest q)
        // AND (window disabled, OR window-safe for every row (smallest
        // k satisfies the tightest/largest-q bound), OR the whole tile
        // sits inside the always-live sink prefix).
        let causalSafe = kiMax <= qiMin
        let windowSafe = window <= 0 || kiMin >= qiMax - window + 1
        if causalSafe && (windowSafe || (overlapsSink && kiMax < sinkEnd)) {
            return .full
        }
        return .edge
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

    // Fused transpose `[nHeads, tPad, headDim]` f16 (HTD, MMA output)
    // → `[nRows, nHeads*headDim]` bf16 (THD token-major, o_proj input),
    // real rows only. One dispatch replaces `transposeHTD_f16` + the
    // `castF16ToBf16` real-row copy, and skips the `outTHD` f16 scratch
    // entirely. Byte-identical: `dst` linear index `tid` decomposes to
    // (t, h, d) in token-major order; the read `bfloat(float(inp[src]))`
    // is the exact expression `castF16ToBf16` applied post-transpose.
    private static let transposeHtdCastBf16Src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void laguna_transpose_htd_cast_bf16(
        device const half *inp [[buffer(0)]],   // [n_heads, t_pad, head_dim]
        device bfloat *out [[buffer(1)]],        // [n_rows, n_heads*head_dim]
        constant uint &t_pad [[buffer(2)]],
        constant uint &n_heads [[buffer(3)]],
        constant uint &head_dim [[buffer(4)]],
        constant uint &total [[buffer(5)]],
        uint tid [[thread_position_in_grid]]
    ) {
        if (tid >= total) return;
        uint d = tid % head_dim;
        uint hh = tid / head_dim;
        uint h = hh % n_heads;
        uint t = hh / n_heads;
        uint srcIdx = (h * t_pad + t) * head_dim + d;
        out[tid] = bfloat(float(inp[srcIdx]));
    }
    """

    /// Dispatch `laguna_transpose_htd_cast_bf16` (fused out-layout).
    /// `input` = `[nHeads, tPad, headDim]` f16; `out` = `[nRows,
    /// nHeads*headDim]` bf16 (real rows). Deletes the stock
    /// transpose+cast pair (and the `outTHD` round trip). Byte-identical.
    static func transposeHTDCastToBf16(
        _ input: Tensor, nRows: Int, tPad: Int, nHeads: Int, headDim: Int,
        into out: Tensor, on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(input.dtype == .f16 && out.dtype == .bf16)
        let total = nRows * nHeads * headDim
        precondition(out.elementCount == total)
        let p = pso(
            "laguna_transpose_htd_cast_bf16", source: transposeHtdCastBf16Src,
            device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(out.buffer, offset: out.offset, index: 1)
        var tPadU = UInt32(tPad), hU = UInt32(nHeads)
        var dU = UInt32(headDim), totU = UInt32(total)
        enc.setBytes(&tPadU, length: 4, index: 2)
        enc.setBytes(&hU, length: 4, index: 3)
        enc.setBytes(&dU, length: 4, index: 4)
        enc.setBytes(&totU, length: 4, index: 5)
        enc.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(total, 256), height: 1, depth: 1))
        enc.endEncoding()
    }

    // ─── W48: direct-KV prefill SDPA (sdpa_kv_compact elimination) ──
    //
    // See Laguna Parity compendium §16-TargetA. `prefillMmaBf16`'s hot
    // path pays a `compactKvBf16ToF16` dispatch (strided bf16 KV cache
    // → packed f16) purely so the MMA kernel can address K/V by a
    // fixed, tightly-packed stride. iron's `iron_sdpa_prefill_mma` now
    // takes the addressing stride (`kLen`) and the real KV length
    // (`kReal`) as independent runtime params (W48 iron change), so it
    // can read the RAW strided bf16 cache directly — `kLen = maxSeq`,
    // `kReal` masks the `[kReal, maxSeq)` tail to -inf alongside the
    // existing causal test. This drops the compact pass entirely.
    //
    // Q still needs the HTD layout the MMA kernel expects, so the
    // pad+transpose step stays, but goes STRAIGHT to bf16 (no more
    // `Ops.castBf16ToF16` en route to f16) — the MMA runs the bf16
    // monomorphization of `iron_sdpa_prefill_mma` (`T=bf16`) instead of
    // the f16 one, per the compendium's route (ii) ("all-bf16 MMA").
    //
    // NOT byte-identical vs the f16-compact default: bf16-native Q/K/V
    // operands are a small precision DELTA from today's f16-packed
    // path (bf16 has fewer mantissa bits than f16 but a wider exponent
    // — neither strictly dominates elementwise, but bf16 avoids an
    // extra intermediate rounding step the f16 path pays, so this is
    // arguably no worse and sometimes more accurate, same class as
    // W44's QMV_V2). Opt-in pending an A/B: BUTTER_LAGUNA_SDPA_DIRECT_KV=1
    // (default OFF).
    static var directKvEnabled: Bool {
        ButterEnv["BUTTER_LAGUNA_SDPA_DIRECT_KV"] == "1"
    }

    // `[T, H, D]` bf16 → `[H, T, D]` bf16. Pure layout permute (no
    // rounding) — same index math as `iron_transpose_thd_f16`, bf16
    // in/out instead of f16.
    private static let transposeThdBf16Src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void laguna_transpose_thd_bf16(
        device const bfloat *inp [[buffer(0)]],   // [t, h, d]
        device bfloat *out [[buffer(1)]],          // [h, t, d]
        constant uint &t_len [[buffer(2)]],
        constant uint &h_len [[buffer(3)]],
        constant uint &d_len [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint t = gid.x;
        uint hd = gid.y;
        if (t >= t_len || hd >= h_len * d_len) return;
        uint h = hd / d_len;
        uint d = hd % d_len;
        out[(h * t_len + t) * d_len + d] = inp[(t * h_len + h) * d_len + d];
    }
    """

    /// Dispatch `laguna_transpose_thd_bf16`: `[tPad, nHeads, headDim]`
    /// bf16 (THD, padded Q) → `[nHeads, tPad, headDim]` bf16 (HTD, MMA
    /// input). Byte-identical layout permute, no cast.
    static func transposeTHDToBf16(
        _ input: Tensor, t: Int, h: Int, d: Int,
        into out: Tensor, on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(input.dtype == .bf16 && out.dtype == .bf16)
        let p = pso(
            "laguna_transpose_thd_bf16", source: transposeThdBf16Src, device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(out.buffer, offset: out.offset, index: 1)
        var tU = UInt32(t), hU = UInt32(h), dU = UInt32(d)
        enc.setBytes(&tU, length: 4, index: 2)
        enc.setBytes(&hU, length: 4, index: 3)
        enc.setBytes(&dU, length: 4, index: 4)
        let hd = h * d
        enc.dispatchThreads(
            MTLSize(width: t, height: hd, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(t, 32), height: min(hd, 32), depth: 1))
        enc.endEncoding()
    }

    // Fused transpose `[nHeads, tPad, headDim]` bf16 (HTD, direct-KV MMA
    // output) → `[nRows, nHeads*headDim]` bf16 (THD token-major, o_proj
    // input), real rows only. Same index math as
    // `laguna_transpose_htd_cast_bf16` (W10) MINUS the cast — the
    // direct-KV MMA already emits bf16, so this is a pure layout
    // permute, not a narrowing.
    private static let transposeHtdToBf16Src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void laguna_transpose_htd_to_bf16(
        device const bfloat *inp [[buffer(0)]],  // [n_heads, t_pad, head_dim]
        device bfloat *out [[buffer(1)]],         // [n_rows, n_heads*head_dim]
        constant uint &t_pad [[buffer(2)]],
        constant uint &n_heads [[buffer(3)]],
        constant uint &head_dim [[buffer(4)]],
        constant uint &total [[buffer(5)]],
        uint tid [[thread_position_in_grid]]
    ) {
        if (tid >= total) return;
        uint d = tid % head_dim;
        uint hh = tid / head_dim;
        uint h = hh % n_heads;
        uint t = hh / n_heads;
        out[tid] = inp[(h * t_pad + t) * head_dim + d];
    }
    """

    /// Dispatch `laguna_transpose_htd_to_bf16` (direct-KV out-layout,
    /// no cast). `input` = `[nHeads, tPad, headDim]` bf16; `out` =
    /// `[nRows, nHeads*headDim]` bf16 (real rows).
    static func transposeHTDToBf16Direct(
        _ input: Tensor, nRows: Int, tPad: Int, nHeads: Int, headDim: Int,
        into out: Tensor, on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(input.dtype == .bf16 && out.dtype == .bf16)
        let total = nRows * nHeads * headDim
        precondition(out.elementCount == total)
        let p = pso(
            "laguna_transpose_htd_to_bf16", source: transposeHtdToBf16Src,
            device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        enc.setBuffer(input.buffer, offset: input.offset, index: 0)
        enc.setBuffer(out.buffer, offset: out.offset, index: 1)
        var tPadU = UInt32(tPad), hU = UInt32(nHeads)
        var dU = UInt32(headDim), totU = UInt32(total)
        enc.setBytes(&tPadU, length: 4, index: 2)
        enc.setBytes(&hU, length: 4, index: 3)
        enc.setBytes(&dU, length: 4, index: 4)
        enc.setBytes(&totU, length: 4, index: 5)
        enc.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(total, 256), height: 1, depth: 1))
        enc.endEncoding()
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

    // One simdgroup (32 lanes) per query row; one threadgroup = one
    // 32-row query tile x one Q head (1024 threads = 32 simdgroups x 32
    // lanes, the same threadgroup geometry `laguna_sdpa_decode_gqa2_bf16_128`
    // and the stock `iron_sdpa_decode_*` kernels already use). Unlike
    // those decode kernels, each simdgroup here owns a DIFFERENT query
    // row end-to-end rather than cooperating on one shared row, so there
    // is no cross-simdgroup reduction / threadgroup-memory combine step
    // at the end: every simdgroup just finalizes and writes its own row.
    //
    // `bounds` is the host-computed `queryBlockBounds(...)` output, laid
    // out as `[nQBlocks, 4]` u32 (`mainStart, fullLo, fullHi, kvEnd`, one
    // row per 32-query tile); `sink_end`/`window` are call-wide scalars
    // (sink/window don't vary per tile, only the derived bounds do).
    private static let blockmapSrc = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void laguna_sdpa_prefill_blockmap_f16_128(
        const device half *q [[buffer(0)]],       // [n_heads, t_pad, 128]
        const device half *k [[buffer(1)]],       // [n_kv_heads, k_pad, 128]
        const device half *v [[buffer(2)]],       // [n_kv_heads, k_pad, 128]
        device half *out [[buffer(3)]],           // [n_heads, t_pad, 128]
        const device uint *bounds [[buffer(4)]],  // [n_q_blocks, 4]
        constant uint &t_pad [[buffer(5)]],
        constant uint &k_pad [[buffer(6)]],
        constant uint &position [[buffer(7)]],
        constant uint &sink_end [[buffer(8)]],
        constant uint &window [[buffer(9)]],
        constant uint &heads_per_group [[buffer(10)]],
        constant float &scale [[buffer(11)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]])
    {
        constexpr uint head_dim = 128u;
        uint qBlock = tgid.x;
        uint qHead = tgid.y;
        uint kvHead = qHead / heads_per_group;
        uint qRow = qBlock * 32u + sg;            // local row in [0, t_pad)
        uint qAbs = position + qRow;              // absolute causal position
        uint d0 = lane * 4u;

        uint bBase = qBlock * 4u;
        uint mainStart = bounds[bBase + 0u];
        uint fullLo = bounds[bBase + 1u];
        uint fullHi = bounds[bBase + 2u];
        uint kvEnd = bounds[bBase + 3u];

        float qv[4];
        {
            const device half *qP = q + (qHead * t_pad + qRow) * head_dim + d0;
            for (uint i = 0u; i < 4u; ++i) { qv[i] = float(qP[i]) * scale; }
        }

        float runMax = -INFINITY;
        float runSum = 0.0f;
        float os[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint kvBase = kvHead * k_pad * head_dim;

        // One K/V element step: dot product via simd_sum across this
        // simdgroup's 4-lane partial products, then an online-softmax
        // update. A macro (not a helper function) so every phase loop
        // below stays a single, independently-readable block; there is
        // no thread-array-reference plumbing to double-check.
    #define BM_STEP(kv)                                                        \\
        {                                                                      \\
            uint kv0 = kvBase + (kv) * head_dim + d0;                          \\
            const device half *kP = k + kv0;                                   \\
            const device half *vP = v + kv0;                                   \\
            float kk[4];                                                       \\
            for (uint j = 0u; j < 4u; ++j) { kk[j] = float(kP[j]); }            \\
            float partial = 0.0f;                                              \\
            for (uint j = 0u; j < 4u; ++j) { partial += qv[j] * kk[j]; }        \\
            float score = simd_sum(partial);                                   \\
            float newMax = max(runMax, score);                                 \\
            float factor = exp(runMax - newMax);                               \\
            float weight = exp(score - newMax);                                \\
            runSum = runSum * factor + weight;                                 \\
            runMax = newMax;                                                   \\
            float vv[4];                                                       \\
            for (uint j = 0u; j < 4u; ++j) { vv[j] = float(vP[j]); }            \\
            if (factor == 1.0f) {                                              \\
                for (uint j = 0u; j < 4u; ++j) { os[j] = os[j] + weight * vv[j]; } \\
            } else {                                                           \\
                for (uint j = 0u; j < 4u; ++j) { os[j] = os[j] * factor + weight * vv[j]; } \\
            }                                                                  \\
        }

        // Phase 0: always-live sink prefix, exempt from the window
        // rule, never from causality.
        for (uint kv = 0u; kv < sink_end; ++kv) {
            if (kv <= qAbs) { BM_STEP(kv); }
        }
        // Phase 1: edge-lo, per-row causal + window test.
        for (uint kv = mainStart; kv < fullLo; ++kv) {
            bool causalOk = kv <= qAbs;
            bool windowOk = (window == 0u) || (kv + window > qAbs);
            if (causalOk && windowOk) { BM_STEP(kv); }
        }
        // Phase 2: full middle, provably valid for every row in this
        // query tile (see `LagunaSdpa.queryBlockBounds`); no mask test.
        for (uint kv = fullLo; kv < fullHi; ++kv) { BM_STEP(kv); }
        // Phase 3: edge-hi (causal diagonal band), per-row masked.
        for (uint kv = fullHi; kv < kvEnd; ++kv) {
            bool causalOk = kv <= qAbs;
            bool windowOk = (window == 0u) || (kv + window > qAbs);
            if (causalOk && windowOk) { BM_STEP(kv); }
        }
    #undef BM_STEP

        float invSum = runSum > 0.0f ? (1.0f / runSum) : 0.0f;
        uint outOff = (qHead * t_pad + qRow) * head_dim + d0;
        for (uint j = 0u; j < 4u; ++j) { out[outOff + j] = half(os[j] * invSum); }
    }
    """

    /// Dispatch `laguna_sdpa_prefill_blockmap_f16_128`. Same call
    /// contract as `Ops.sdpaPrefillMma`: `q`/`out` are
    /// `[nHeads, tPad, 128]` f16 (HTD layout), `k`/`v` are
    /// `[nKVHeads, kPad, 128]` f16. `window <= 0` disables the sliding
    /// window (pure causal, same masking `Ops.sdpaPrefillMma` computes,
    /// just via a different, scalar kernel); `sinkEnd <= 0` disables the
    /// sink exception. Builds and uploads the per-query-tile bounds
    /// buffer host-side (`queryBlockBounds`), then dispatches one
    /// `(nQBlocks, nHeads, 1)` threadgroup grid, 1024 threads/group.
    static func sdpaPrefillBlockmap(
        qHTD: Tensor, kPacked: Tensor, vPacked: Tensor,
        nHeads: Int, nKVHeads: Int,
        tPad: Int, kPad: Int, position: Int,
        window: Int, sinkEnd: Int, scale: Float,
        on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor
    ) {
        precondition(tPad > 0 && tPad % 32 == 0, "sdpaPrefillBlockmap: tPad must be a positive multiple of 32")
        precondition(
            nHeads % nKVHeads == 0, "sdpaPrefillBlockmap: nHeads must be a multiple of nKVHeads")
        precondition(
            qHTD.dtype == .f16 && kPacked.dtype == .f16 && vPacked.dtype == .f16 && out.dtype == .f16,
            "sdpaPrefillBlockmap: q/k/v/out must be f16")

        let nQBlocks = tPad / 32
        let clampedSinkEnd = max(0, min(sinkEnd, kPad))
        let clampedWindow = max(0, window)

        var boundsHost = [UInt32](repeating: 0, count: nQBlocks * 4)
        for qi in 0 ..< nQBlocks {
            let qiMin = position + qi * 32
            let qiMax = qiMin + 31
            let b = queryBlockBounds(
                qiMin: qiMin, qiMax: qiMax, kPad: kPad,
                window: clampedWindow, sinkEnd: clampedSinkEnd)
            boundsHost[qi * 4 + 0] = UInt32(b.mainStart)
            boundsHost[qi * 4 + 1] = UInt32(b.fullLo)
            boundsHost[qi * 4 + 2] = UInt32(b.fullHi)
            boundsHost[qi * 4 + 3] = UInt32(b.kvEnd)
        }
        // Fresh (non-ring) allocation: this buffer is written directly
        // from the CPU via `copyIn` below, unlike the rest of
        // `prefillMmaBf16`'s scratch temps, which are only ever written
        // by GPU kernels/blits already ordered on `cmd`. A CPU write has
        // no ordering relationship with a prior in-flight command
        // buffer's GPU reads of a shared ring slot, so this one must not
        // borrow the `scratch(...)` ring; it's small (`nQBlocks * 16`
        // bytes), so a fresh allocation per call is cheap.
        let boundsBuf = Tensor.empty(shape: [nQBlocks, 4], dtype: .u32, device: device)
        boundsBuf.copyIn(from: boundsHost)

        let headsPerGroup = nHeads / nKVHeads
        let p = pso(
            "laguna_sdpa_prefill_blockmap_f16_128", source: blockmapSrc,
            device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        enc.setBuffer(qHTD.buffer, offset: qHTD.offset, index: 0)
        enc.setBuffer(kPacked.buffer, offset: kPacked.offset, index: 1)
        enc.setBuffer(vPacked.buffer, offset: vPacked.offset, index: 2)
        enc.setBuffer(out.buffer, offset: out.offset, index: 3)
        enc.setBuffer(boundsBuf.buffer, offset: boundsBuf.offset, index: 4)
        var tPadU = UInt32(tPad)
        var kPadU = UInt32(kPad)
        var positionU = UInt32(position)
        var sinkEndU = UInt32(clampedSinkEnd)
        var windowU = UInt32(clampedWindow)
        var headsPerGroupU = UInt32(headsPerGroup)
        var scaleV = scale
        enc.setBytes(&tPadU, length: 4, index: 5)
        enc.setBytes(&kPadU, length: 4, index: 6)
        enc.setBytes(&positionU, length: 4, index: 7)
        enc.setBytes(&sinkEndU, length: 4, index: 8)
        enc.setBytes(&windowU, length: 4, index: 9)
        enc.setBytes(&headsPerGroupU, length: 4, index: 10)
        enc.setBytes(&scaleV, length: 4, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: nQBlocks, height: nHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
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
    /// - Parameters window/sinkEnd: block-map opt-in only (see
    ///   `prefillBlockmapEnabled`'s header comment above); ignored
    ///   entirely unless `BUTTER_LAGUNA_PREFILL_BLOCKMAP=1`, in which
    ///   case `window > 0` newly enables real sliding-window masking at
    ///   prefill (previously inert here) and `sinkEnd > 0` keeps an
    ///   always-live sink prefix. Both default to `0` (pure causal,
    ///   identical to every existing call site) for source compatibility.
    static func prefillMmaBf16(
        qTHD: Tensor,  // [nRows, nHeads, headDim] bf16
        cacheK: Tensor, cacheV: Tensor,
        nRows: Int, position: Int,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        maxSeq: Int, scale: Float,
        on cmd: inout MTLCommandBuffer, device: Device,
        window: Int = 0, sinkEnd: Int = 0,
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

        // W48 direct-KV path (see the header comment above
        // `directKvEnabled`): skip `compactKvBf16ToF16` entirely and
        // read `cacheK`/`cacheV` raw+strided by `maxSeq`. Q goes
        // straight to bf16 HTD (no f16 cast) so the MMA runs its bf16
        // monomorphization end to end. Mutually exclusive with the
        // scalar block-map path (that kernel is f16-only) — falls
        // through to the default compact path if both are requested.
        if directKvEnabled && !prefillBlockmapEnabled {
            let qHTDBf = scratch(
                tag: "laguna_sdpa_qhtd_bf", shape: [nHeads, tPad, headDim],
                dtype: .bf16, device: device, slot: slot)
            transposeTHDToBf16(
                qPadBf, t: tPad, h: nHeads, d: headDim, into: qHTDBf, on: cmd, device: device)
            if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_qcast_transpose", cmd) }
            // No compact/blit pass — this checkpoint is kept so phase-
            // timing probes (BUTTER_LAGUNA_ATTN_PHASE_TIMING=1) still see
            // the same stage sequence, now ~free.
            if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_kv_compact", cmd) }

            let outHTDBf = scratch(
                tag: "laguna_sdpa_outhtd_bf", shape: [nHeads, tPad, headDim],
                dtype: .bf16, device: device, slot: slot)
            _ = Ops.sdpaPrefillMma(
                q: qHTDBf, k: cacheK, v: cacheV,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                qLen: tPad, kLen: maxSeq, scale: scale, on: cmd, into: outHTDBf,
                position: position, kReal: kReal)
            if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_mma_kernel", cmd) }

            transposeHTDToBf16Direct(
                outHTDBf, nRows: nRows, tPad: tPad, nHeads: nHeads, headDim: headDim,
                into: outBf, on: cmd, device: device)
            if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_out_transpose_cast", cmd) }
            return outBf
        }

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

        if prefillBlockmapEnabled {
            // T-blockmap opt-in path: same qHTD/kPacked/vPacked inputs,
            // same outHTD output as the MMA path below, so the two are
            // directly A/B-able (see LagunaPrefillBlockmapTests' GPU
            // equivalence cases). Restructures the loop into skip-dead /
            // full-no-mask / edge-masked phases instead of scanning
            // `[0, kPad)` unconditionally, and, new here, actually
            // enforces `window`/`sinkEnd` instead of ignoring them.
            sdpaPrefillBlockmap(
                qHTD: qHTD, kPacked: kPacked, vPacked: vPacked,
                nHeads: nHeads, nKVHeads: nKVHeads,
                tPad: tPad, kPad: kPad, position: position,
                window: window, sinkEnd: sinkEnd, scale: scale,
                on: cmd, device: device, into: outHTD)
        } else {
            _ = Ops.sdpaPrefillMma(
                q: qHTD, k: kPacked, v: vPacked,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                qLen: tPad, kLen: kPad, scale: scale, on: cmd, into: outHTD)
        }
        if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_mma_kernel", cmd) }

        if sdpaNativeLayoutEnabled {
            // Single fused transpose+cast dispatch (byte-identical).
            transposeHTDCastToBf16(
                outHTD, nRows: nRows, tPad: tPad, nHeads: nHeads, headDim: headDim,
                into: outBf, on: cmd, device: device)
        } else {
            let outTHD = scratch(
                tag: "laguna_sdpa_outthd", shape: [tPad, nHeads, headDim],
                dtype: .f16, device: device, slot: slot)
            _ = Ops.transposeHTD_f16(outHTD, t: tPad, h: nHeads, d: headDim, on: cmd, into: outTHD)
            // Cast only the real T rows back to bf16.
            let realOutF16 = Tensor(
                buffer: outTHD.buffer, offset: outTHD.offset,
                shape: [nRows * nHeads * headDim], dtype: .f16)
            castF16ToBf16(
                realOutF16,
                into: outBf.reshaped(to: [nRows * nHeads * headDim]),
                on: cmd, device: device)
        }
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
        ButterEnv["BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE"] != "0"
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

    // F-85 prefill pack, item 2: H1 threadgroup geometry for the T30 fused
    // QK-norm+RoPE prefill kernels. T30 shipped one TG (128 threads = 4
    // simdgroups) per (head-block, token) with 4 heads packed per TG, one
    // simdgroup per head. H1 dispatches one TG (32 threads = 1 simdgroup)
    // per (head, token) instead, same per-lane RMSNorm/RoPE math and the
    // same intra-simdgroup `simd_shuffle` rotary-partner lookup, just a
    // narrower/more-numerous threadgroup grid (`nHeads+nKVHeads` in the
    // head dimension instead of `(nHeads+nKVHeads)/4`). Bit-exact vs the
    // 4-head/128-thread kernel: identical scalar/transcendental expression
    // per lane, `simd_shuffle` result does not depend on which simdgroup
    // index a lane's 32-lane group happens to run as. Opt-in, independent
    // of `BUTTER_LAGUNA_PREFILL_QK_ROPE_FUSE` (still gates whether the
    // fused path runs at all): this only chooses which of the two
    // dispatch geometries the fused kernel uses.
    // Escape/opt-in: BUTTER_LAGUNA_PREFILL_QKROPE_H1=1 (default OFF, unverified
    // this session, see benchmarks/laguna-xs/README.md F-85 prefill pack).
    static var prefillQkRopeH1Enabled: Bool {
        ButterEnv["BUTTER_LAGUNA_PREFILL_QKROPE_H1"] == "1"
    }

    // F-85 prefill pack, item 3: fp32 RoPE angle atlas for prefill (sliding
    // layers only; decode and the YaRN full-attention twin are explicitly
    // out of scope this session, see the file-level note above
    // `fusedQKNormRopeForPrefill`). Precomputes `cos`/`sin` for every
    // `(position, pair)` up to `ropeAtlasMaxPositions` once (fp32), and the
    // fused kernel indexes the table by absolute position instead of
    // evaluating `exp2`/`cos`/`sin` analytically per lane per call. Rotary
    // partner is still resolved via the existing intra-simdgroup
    // `simd_shuffle(lane ^ 16)` (full rotary, headDim 128): the atlas only
    // replaces the angle computation, not the shuffle.
    // Opt-in: BUTTER_LAGUNA_ROPE_ATLAS=1 (default OFF, unverified this
    // session). Falls back to the analytic kernel (H1 or base, per the flag
    // above) whenever `position + tPad` exceeds the atlas's cached
    // position range.
    static var prefillRopeAtlasEnabled: Bool {
        ButterEnv["BUTTER_LAGUNA_ROPE_ATLAS"] == "1"
    }

    static let ropeAtlasMaxPositions = 4096

    static func prefillQkRopeAtlasEligible(position: Int, tPad: Int) -> Bool {
        prefillRopeAtlasEnabled && (position + tPad) <= ropeAtlasMaxPositions
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

    // H1 variant of `prefillFusedQKRopeSrc` (F-85 prefill pack item 2): one
    // threadgroup (32 threads, 1 simdgroup) per (head, token) instead of
    // one TG (128 threads, 4 simdgroups) per (head-block, token). Same
    // per-lane body verbatim otherwise: `head` is just `tgpig.x` directly
    // (no simdgroup-index offset, always the sole simdgroup in its TG).
    private static let prefillFusedQKRopeSrcH1 = """
    #include <metal_stdlib>
    using namespace metal;

    // One threadgroup (32 threads = 1 simdgroup) per (head, token);
    // grid.x = n_q_heads+n_kv_heads, grid.y = nRows.
    kernel void laguna_prefill_sliding_qk_norm_rope_direct_bf16_128_h1(
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
        uint lane [[thread_index_in_simdgroup]]
    ) {
        constexpr uint head_dim = 128u;
        constexpr uint half_dim = 64u;

        uint t = tgpig.y;
        uint head = tgpig.x;
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

        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16u);
        }

        if (lane < 16u) {
            float log_theta = log2(theta_base);
            float half_f = float(half_dim);
            float pos_f = float(position_base + t);
            if (is_q) {
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
                    bfloat r0 = bfloat(first * c - second * s);
                    bfloat r1 = bfloat(first * s + second * c);
                    out[pair] = half(float(r0));
                    out[pair + half_dim] = half(float(r1));
                }
            } else {
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

    // F-85 prefill pack item 3: fp32 angle atlas. Build kernel fills
    // `cos_atlas`/`sin_atlas` `[maxPositions, rotaryPairs]` (fp32, row per
    // absolute position) once; the read kernel below indexes it instead of
    // evaluating `exp2`/`cos`/`sin` per lane. Same H1 (32-thread, 1
    // simdgroup per head) dispatch geometry as item 2's kernel; the two
    // levers compose in this one kernel (atlas replaces the angle math,
    // H1 is the dispatch shape). A base-geometry (4-head/128-thread) atlas
    // variant was not authored this session (scope: prefill lever
    // combination left for a follow-up, see the mission report).
    private static let ropeAtlasBuildSrc = """
    #include <metal_stdlib>
    using namespace metal;
    // grid.x = rotary_pairs, grid.y = maxPositions.
    kernel void laguna_rope_atlas_build_f32(
        device float *cos_atlas [[buffer(0)]],
        device float *sin_atlas [[buffer(1)]],
        constant uint &rotary_pairs [[buffer(2)]],
        constant float &theta_base [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint pair = gid.x;
        uint pos = gid.y;
        if (pair >= rotary_pairs) return;
        float log_theta = log2(theta_base);
        float half_f = float(rotary_pairs);
        float inv_freq = exp2(((-float(pair)) * log_theta) / half_f);
        float th = float(pos) * inv_freq;
        uint idx = pos * rotary_pairs + pair;
        cos_atlas[idx] = cos(th);
        sin_atlas[idx] = sin(th);
    }
    """

    private static let prefillFusedQKRopeSrcAtlas = """
    #include <metal_stdlib>
    using namespace metal;

    // Atlas-read twin of the H1 kernel above: same one-simdgroup-per-head
    // dispatch, but `cos`/`sin` come from a precomputed `[maxPositions, 64]`
    // fp32 table instead of an analytic `exp2`/`cos`/`sin` evaluation.
    // Rotary partner is still the intra-simdgroup `simd_shuffle(lane^16)`
    // lookup; the atlas only replaces the angle source.
    kernel void laguna_prefill_sliding_qk_norm_rope_atlas_bf16_128(
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
        constant float &eps [[buffer(10)]],
        device const float *cos_atlas [[buffer(11)]],
        device const float *sin_atlas [[buffer(12)]],
        constant uint &atlas_stride [[buffer(13)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        constexpr uint head_dim = 128u;
        constexpr uint half_dim = 64u;

        uint t = tgpig.y;
        uint head = tgpig.x;
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

        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ 16u);
        }

        if (lane < 16u) {
            uint pos = position_base + t;
            uint atlas_row = pos * atlas_stride;
            if (is_q) {
                device half *out = out_q_htd + (head * t_pad + t) * head_dim;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    uint pair = base + i;
                    float first = float(normalized[i]);
                    float second = paired[i];
                    float c = cos_atlas[atlas_row + pair];
                    float s = sin_atlas[atlas_row + pair];
                    bfloat r0 = bfloat(first * c - second * s);
                    bfloat r1 = bfloat(first * s + second * c);
                    out[pair] = half(float(r0));
                    out[pair + half_dim] = half(float(r1));
                }
            } else {
                uint kh = head - n_q_heads;
                device bfloat *out = out_k + (t * n_kv_heads + kh) * head_dim;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    uint pair = base + i;
                    float first = float(normalized[i]);
                    float second = paired[i];
                    float c = cos_atlas[atlas_row + pair];
                    float s = sin_atlas[atlas_row + pair];
                    out[pair] = bfloat(first * c - second * s);
                    out[pair + half_dim] = bfloat(first * s + second * c);
                }
            }
        }
    }
    """

    /// Lazily-built fp32 RoPE angle atlas, cached by `(rotaryPairs,
    /// thetaBase)`. `cos`/`sin` are separate `[maxPositions, rotaryPairs]`
    /// buffers built via `laguna_rope_atlas_build_f32` (see source above).
    /// Built once per distinct `thetaBase` (Laguna XS's sliding layers all
    /// share one rope theta, so in practice this builds once for the whole
    /// model run); grown or rebuilt is never needed within
    /// `ropeAtlasMaxPositions` since prefill chunks that would exceed it
    /// are routed back to the analytic kernel by
    /// `prefillQkRopeAtlasEligible` before this is ever called.
    private struct RopeAtlasKey: Hashable {
        let rotaryPairs: Int
        let thetaBits: UInt32
    }
    nonisolated(unsafe) private static var ropeAtlasCache: [RopeAtlasKey: (cos: Tensor, sin: Tensor)] = [:]
    private static let ropeAtlasLock = NSLock()

    private static func ropeAtlas(
        rotaryPairs: Int, thetaBase: Float, on cmd: MTLCommandBuffer, device: Device
    ) -> (cos: Tensor, sin: Tensor) {
        let key = RopeAtlasKey(rotaryPairs: rotaryPairs, thetaBits: thetaBase.bitPattern)
        ropeAtlasLock.lock()
        if let cached = ropeAtlasCache[key] {
            ropeAtlasLock.unlock()
            return cached
        }
        ropeAtlasLock.unlock()

        let cosT = Tensor.empty(shape: [ropeAtlasMaxPositions, rotaryPairs], dtype: .f32, device: device)
        let sinT = Tensor.empty(shape: [ropeAtlasMaxPositions, rotaryPairs], dtype: .f32, device: device)
        // Build dispatch is encoded onto the caller's `cmd`, same as every
        // other kernel in this file: relies on same-`MTLCommandQueue`
        // commit-order execution (no explicit fence). The first eligible
        // call's build dispatch and its own atlas-read dispatch share one
        // command buffer (hazard-tracked automatically), and later cache
        // hits from later layers/chunks only ever READ these buffers, on
        // command buffers submitted after the build's, to the same queue.
        let p = pso(
            "laguna_rope_atlas_build_f32", source: ropeAtlasBuildSrc, device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(cosT.buffer, offset: cosT.offset, index: 0)
            enc.setBuffer(sinT.buffer, offset: sinT.offset, index: 1)
            var rp = UInt32(rotaryPairs)
            var theta = thetaBase
            enc.setBytes(&rp, length: 4, index: 2)
            enc.setBytes(&theta, length: 4, index: 3)
            enc.dispatchThreads(
                MTLSize(width: rotaryPairs, height: ropeAtlasMaxPositions, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(rotaryPairs, 32), height: 8, depth: 1))
            enc.endEncoding()
        }
        ropeAtlasLock.lock()
        ropeAtlasCache[key] = (cosT, sinT)
        ropeAtlasLock.unlock()
        return (cosT, sinT)
    }

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

        // F-85 prefill pack: atlas (item 3) takes priority over H1 (item 2)
        // when both are opted in. Atlas eligibility already implies the
        // H1-style one-simdgroup-per-head geometry (see
        // `prefillFusedQKRopeSrcAtlas`), so there is nothing left for the
        // H1-only kernel to add on top. Neither flag changes anything by
        // default (both OFF); the base T30 4-head/128-thread analytic
        // kernel remains what ships.
        let useAtlas = prefillQkRopeAtlasEligible(position: position, tPad: tPad)
        if useAtlas {
            let (cosAtlas, sinAtlas) = ropeAtlas(
                rotaryPairs: headDim / 2, thetaBase: thetaBase, on: cmd, device: device)
            let p = pso(
                "laguna_prefill_sliding_qk_norm_rope_atlas_bf16_128",
                source: prefillFusedQKRopeSrcAtlas, device: device.mtlDevice)
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
                var epsV = eps
                enc.setBytes(&nQU, length: 4, index: 6)
                enc.setBytes(&nKvU, length: 4, index: 7)
                enc.setBytes(&tPadU, length: 4, index: 8)
                enc.setBytes(&posU, length: 4, index: 9)
                enc.setBytes(&epsV, length: 4, index: 10)
                enc.setBuffer(cosAtlas.buffer, offset: cosAtlas.offset, index: 11)
                enc.setBuffer(sinAtlas.buffer, offset: sinAtlas.offset, index: 12)
                var atlasStride = UInt32(headDim / 2)
                enc.setBytes(&atlasStride, length: 4, index: 13)
                enc.dispatchThreadgroups(
                    MTLSize(width: nHeads + nKVHeads, height: nRows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.endEncoding()
            }
        } else if prefillQkRopeH1Enabled {
            let p = pso(
                "laguna_prefill_sliding_qk_norm_rope_direct_bf16_128_h1",
                source: prefillFusedQKRopeSrcH1, device: device.mtlDevice)
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
                enc.dispatchThreadgroups(
                    MTLSize(width: nHeads + nKVHeads, height: nRows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.endEncoding()
            }
        } else {
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

        // W58 fused epilogue eligibility: f16 MMA path, head_dim=128.
        let useFusedEpilogue =
            prefillFusedEpilogueEnabled && qHTD.dtype == .f16 && headDim == 128
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

        if useFusedEpilogue {
            // W58: the MMA kernel writes O straight to the transposed bf16
            // layout — no `outHTD` round trip, no separate transpose+cast
            // dispatch. The kernel iterates all `tPad` query rows, so
            // allocate a padded output and return a view of the first
            // `nRows` rows.
            let outBfPad = Tensor.empty(
                shape: [tPad, nHeads * headDim], dtype: .bf16, device: device)
            Ops.sdpaPrefillMmaTransBf16(
                q: qHTD, k: kPacked, v: vPacked,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                qLen: tPad, kLen: kPad, scale: scale, on: cmd, into: outBfPad)
            if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_mma_fused_epilogue", cmd) }
            return Tensor(
                buffer: outBfPad.buffer, offset: outBfPad.offset,
                shape: [nRows, nHeads * headDim], dtype: .bf16)
        }

        let outHTD = scratch(
            tag: "laguna_sdpa_outhtd", shape: [nHeads, tPad, headDim],
            dtype: .f16, device: device, slot: slot)
        let outBf = Tensor.empty(
            shape: [nRows, nHeads * headDim], dtype: .bf16, device: device)

        _ = Ops.sdpaPrefillMma(
            q: qHTD, k: kPacked, v: vPacked,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            qLen: tPad, kLen: kPad, scale: scale, on: cmd, into: outHTD)
        if let phaseCheckpoint { cmd = phaseCheckpoint("sdpa_mma_kernel", cmd) }

        if sdpaNativeLayoutEnabled {
            // Single fused transpose+cast dispatch (byte-identical),
            // no `outTHD` round trip.
            transposeHTDCastToBf16(
                outHTD, nRows: nRows, tPad: tPad, nHeads: nHeads, headDim: headDim,
                into: outBf, on: cmd, device: device)
        } else {
            let outTHD = scratch(
                tag: "laguna_sdpa_outthd", shape: [tPad, nHeads, headDim],
                dtype: .f16, device: device, slot: slot)
            _ = Ops.transposeHTD_f16(outHTD, t: tPad, h: nHeads, d: headDim, on: cmd, into: outTHD)
            let realOutF16 = Tensor(
                buffer: outTHD.buffer, offset: outTHD.offset,
                shape: [nRows * nHeads * headDim], dtype: .f16)
            castF16ToBf16(
                realOutF16,
                into: outBf.reshaped(to: [nRows * nHeads * headDim]),
                on: cmd, device: device)
        }
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

    // H1 variant of `prefillFusedQKYarnSrc` (F-85 prefill pack item 2),
    // same relationship as `prefillFusedQKRopeSrcH1` to the sliding
    // kernel: one threadgroup (32 threads, 1 simdgroup) per (head, token),
    // `head = tgpig.x` directly, otherwise byte-for-byte the same body.
    private static let prefillFusedQKYarnSrcH1 = """
    #include <metal_stdlib>
    using namespace metal;

    // One threadgroup (32 threads = 1 simdgroup) per (head, token);
    // grid.x = n_q_heads+n_kv_heads, grid.y = nRows.
    kernel void laguna_prefill_full_qk_norm_yarn_direct_bf16_128_h1(
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
        uint lane [[thread_index_in_simdgroup]]
    ) {
        constexpr uint head_dim = 128u;

        uint t = tgpig.y;
        uint head = tgpig.x;
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

        uint shuffle_dist = rotary_pairs / 4u;
        thread float paired[4];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            paired[i] = simd_shuffle(float(normalized[i]), lane ^ shuffle_dist);
        }

        uint rotary_lanes = rotary_pairs / 4u;
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

        // F-85 prefill pack item 2 (H1 geometry). No atlas twin for YaRN
        // this session (out of scope, see the mission report).
        if prefillQkRopeH1Enabled {
            let p = pso(
                "laguna_prefill_full_qk_norm_yarn_direct_bf16_128_h1",
                source: prefillFusedQKYarnSrcH1, device: device.mtlDevice)
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
                enc.dispatchThreadgroups(
                    MTLSize(width: nHeads + nKVHeads, height: nRows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.endEncoding()
            }
        } else {
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
        ButterEnv["BUTTER_LAGUNA_FUSED_QK_NORM_ROPE"] != "0"
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
