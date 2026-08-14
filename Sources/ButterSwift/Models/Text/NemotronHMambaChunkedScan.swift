// Copyright 2026 Tom Turney (@TheTom)
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
// Chunked selective-scan prefill for `NemotronHMambaLayer` (Nemotron
// 3.5 Lightning's 23 mamba2 mixer layers).
//
// ─── Motivation ────────────────────────────────────────────────────
//
// PR #132 batched attention/dense-MLP/NVFP4-MoE prefill (pp512 42.6 ->
// 110.1 tok/s), but left Mamba 2 on `DecoderLayer`'s default
// `decodeMulti` — a per-row loop that still dispatches `conv1dCausalStep`
// + the grouped SSM step once EACH for every one of the T prompt
// tokens (see `NemotronHMambaLayer.decode`'s step (3)/(6)). At pp512
// the batched layers hide most of that behind the (now much cheaper)
// attention/MLP/MoE dispatches; at pp2048 the still-sequential mamba
// fraction dominates (measured speedup collapses from ~2.58x at pp512
// to ~1.28x at pp2048 — the wider T amplifies a per-token cost that
// never got a batched path).
//
// This file adds that batched path: `NemotronHMambaLayer.decodeMulti`
// (in `NemotronHText.swift`) now runs the WHOLE per-layer mixer batched
// over T rows, with the recurrence itself split into
// `ceil(T / chunkSize)` dispatches instead of T.
//
// ─── Algorithm (v1: chunked, sequential-inside-the-chunk) ───────────
//
// The per-(head, channel) selective-scan recurrence is
//
//     h_t = a_t * h_{t-1} + b_t * x_t         (a_t = exp(A_eff * dt_t))
//     y_t = c_t . h_t
//
// which unrolls, within a chunk of C tokens starting from carry-in
// state h_in, to
//
//     h_t = (prod_{k<=t} a_k) * h_in + sum_{j<=t} (prod_{j<k<=t} a_k) * b_j * x_j
//     y_t = c_t . h_t
//
// i.e. a per-chunk cumulative decay product times the carried-in state,
// plus an intra-chunk contribution from every earlier token in the
// chunk. The fully parallel form computes the cumulative products and
// the intra-chunk contraction with `c_t` as prefix-sum-shaped work
// spread across the C dimension. This file does NOT implement that yet
// — per the perf-wave scope, v1 is the simpler, still-correct
// reduction: keep the token loop *sequential*, but move it INSIDE one
// kernel dispatch per chunk instead of one dispatch per token. That
// alone removes `chunkSize - 1` of every `chunkSize` dispatches (the
// documented dominant cost at pp2048); a future v2 can replace
// `scanChunk`'s inner token loop with the parallel cumulative-product
// form without touching any call site here.
//
// `scanChunk` is therefore a mechanical transcription of
// `NemotronHMambaGroupedSSMFuse.ssmStepGrouped`'s single-token kernel
// (itself a transcription of the shipped `iron_ssm_step_*` kernel; see
// that file's header for the address-computation-only bit-exactness
// argument) wrapped in an outer `for t in 0..<chunkLen` loop. The
// state buffer `h` is read/written through GLOBAL memory on every
// token exactly as the unfused per-token dispatch chain does — no
// register-resident carry across tokens in v1, trading some memory
// bandwidth for zero new correctness surface (no compile-time
// array-size baking, no register-pressure/spill risk to reason about).
// Chunk boundaries carry state for free: consecutive chunk dispatches
// on the SAME command buffer read-after-write the SAME `h` buffer,
// which Metal already serializes correctly across encoders without any
// host round-trip — so `ceil(T / chunkSize)` dispatches queue back to
// back with no CPU sync between them.
//
// Because every floating-point operation a thread performs is
// unchanged in expression, order, and operand values from
// `ssmStepGrouped`'s per-token body (only the token-stride added to
// the `x`/`b`/`c`/`dt`/`y` addressing, and `a` demoted to a
// loop-invariant per-head read) — expect (and test for) BIT-EXACT
// parity against calling `ssmStepGrouped` `T` times in a row, same
// class of claim `ssmStepGrouped` itself makes vs the stock kernel.
// See `NemotronHMambaChunkedScanTests` for the oracle.
//
// ─── Batched mixer plumbing around the scan ──────────────────────────
//
// The rest of the mixer (RMSNorm, in_proj, conv1d+silu, dt=softplus,
// D-skip, gated mixer RMSNorm, out_proj) is turned into a real
// batched-T implementation too, reusing existing primitives wherever
// they already support arbitrary row counts:
//
//   - RMSNorm / gated mixer RMSNorm: `Ops.rmsNormRows` already
//     broadcasts a single `[rowSize]` weight across `nRows` — no new
//     kernel needed (`decode`'s single-token call already exercises
//     the same broadcast for `nGroups` rows).
//   - in_proj / out_proj: `AnyLinear.callMany` (existing batched GEMM
//     path used by the attention/MLP F-132 `decodeMulti`s).
//   - conv1d + SiLU: NemotronH's `in_proj` is ONE combined GEMM
//     (`z | xBC | dt_raw`), unlike Qwen3.5 GDN's four separate
//     projections — so the combined `[T, projDim]` output needs a
//     COLUMN slice (contiguous per row, strided across rows) to
//     recover `z`/`xBC`/`dt_raw`, which no existing op does. `Ops.
//     conv1dCausalStepSiluCastMany` (the T-tokens-in-one-dispatch
//     conv+silu kernel Qwen3.5 GDN already uses) covers the actual
//     conv sweep, but only emits f32 — cast back to the activation
//     dtype so the rest of the pipeline matches `decode`'s native-dtype
//     convention exactly.
//   - `D`/`mixer.norm.weight`/`dt_bias` are all `[dInner]`/`[nHeads]`
//     per-CHANNEL constants that need broadcasting across `T` ROWS
//     before they can feed `Ops.add`/`Ops.mul` (which require exact
//     shape match, no implicit broadcast) — a plain byte-repeat tile,
//     zero floating-point math, so it can't introduce numerical drift
//     on its own; the actual add/mul math after tiling reuses the
//     SAME `Ops.add`/`Ops.mul`/`Ops.softplus`/`NemotronHMambaMixerFuse.
//     addMul` kernels the per-token path already calls (deliberately
//     NOT re-deriving a fused add+softplus here — see
//     `NemotronHMambaMixerFuse`'s header for why that fusion stays
//     opt-in-only: a CI run on different GPU/driver measured 142/512
//     bf16 mismatches for exactly that class of transcendental fusion).
//
// Two new byte-generic (dtype-agnostic) data-movement kernels cover
// both needs — `sliceColumns` (strided column extract) and
// `rowBroadcastTile` (row repeat) — plus one small per-dtype cast
// kernel (`castF32ToNative`) for the conv output. None of the three do
// any floating-point arithmetic, so none of them can be the source of
// a numerical mismatch against the sequential path; only `scanChunk`
// touches the recurrence math.
//
// ─── Env gate ──────────────────────────────────────────────────────
//
// `BUTTER_NEMOTRON_CHUNKED_SCAN=0` reverts `NemotronHMambaLayer.
// decodeMulti` to the exact per-row default (duplicated inline since
// Swift can't dispatch to a protocol extension's default method from
// an overriding conformance). `BUTTER_NEMOTRON_CHUNK_SIZE` overrides
// the chunk length `C` (default 128) for sweeping 64/128/256 etc.

import Foundation
import Metal
import IronSwift

enum NemotronHMambaChunkedScan {
    /// Default ON — see the file header for the correctness argument
    /// and `NemotronHMambaChunkedScanTests` for the oracle.
    /// `BUTTER_NEMOTRON_CHUNKED_SCAN=0` reverts `NemotronHMambaLayer.
    /// decodeMulti` to the exact per-row default.
    static let enabled: Bool =
        ButterEnv["BUTTER_NEMOTRON_CHUNKED_SCAN"] != "0"

    /// Chunk length `C` — tokens processed per `scanChunk` dispatch.
    /// `BUTTER_NEMOTRON_CHUNK_SIZE` overrides for perf sweeps (64/128/
    /// 256); default 128.
    static let chunkSize: Int = {
        if let raw = ButterEnv["BUTTER_NEMOTRON_CHUNK_SIZE"], let n = Int(raw), n > 0 {
            return n
        }
        return 128
    }()

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pso(
        _ name: String, source: String, device: MTLDevice, fastMath: Bool
    ) -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        let cacheKey = fastMath ? name : "\(name)#strict"
        if let p = psoCache[cacheKey] { return p }
        let opts = MTLCompileOptions()
        opts.fastMathEnabled = fastMath
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[cacheKey] = p
        return p
    }

    // ─── (a) chunked selective scan ───────────────────────────────────
    //
    // Mechanical transcription of `NemotronHMambaGroupedSSMFuse.
    // ssmStepGrouped`'s per-token kernel body (see that file for the
    // per-line correspondence to the shipped `iron_ssm_step_*` kernel)
    // wrapped in an outer `for t_i in 0..<chunk_len` loop. `a` (aEff)
    // is loop-invariant (a load-time model parameter, not per-token);
    // `x`/`b`/`c`/`dt`/`y` gain a token stride; `h` keeps the EXACT
    // same global-memory addressing every token, read-modify-written
    // once per token exactly as the unfused per-token dispatch chain
    // would — this is the "v1: sequential inside one dispatch" design,
    // not the fully parallel cumulative-product form (see file header).
    private static let ssmScanChunkSrc = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void nemotron_mixer_ssm_scan_chunk_f32(
        const device float *x [[buffer(0)]],
        const device float *a [[buffer(1)]],
        const device float *b [[buffer(2)]],
        const device float *c [[buffer(3)]],
        const device float *dt [[buffer(4)]],
        device float *h [[buffer(5)]],
        device float *y [[buffer(6)]],
        constant uint &head_dim [[buffer(7)]],
        constant uint &state_dim [[buffer(8)]],
        constant uint &heads_per_group [[buffer(9)]],
        constant uint &chunk_len [[buffer(10)]],
        constant uint &d_inner [[buffer(11)]],
        constant uint &bc_stride [[buffer(12)]],
        constant uint &n_heads [[buffer(13)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint v_idx = gid.x;
        uint v1 = head_dim;
        uint v_h_id = v_idx / v1;
        uint v3 = v_h_id * v1;
        uint v_d = v_idx - v3;
        uint v_g_id = v_h_id / heads_per_group;
        uint v_bc_base0 = v_g_id * state_dim;
        uint v_h_base = (v_h_id * state_dim) * v1;
        float v_a = a[v_h_id];
        for (uint t_i = 0; t_i < chunk_len; t_i += 1) {
            auto v5 = dt[t_i * n_heads + v_h_id];
            float v_dt_val = static_cast<float>(v5);
            auto v9 = (v_a * v_dt_val);
            auto v_decay = exp(v9);
            auto v13 = x[t_i * d_inner + v3 + v_d];
            float v_x_d = static_cast<float>(v13);
            float __ml_y_d = 0.0f;
            uint v_bc_row = t_i * bc_stride + v_bc_base0;
            for (uint i_0 = 0; i_0 < state_dim; i_0 += 1) {
                uint v_h_idx = v_h_base + i_0 * v1 + v_d;
                auto v_h_old = h[v_h_idx];
                uint v_bc_idx = v_bc_row + i_0;
                auto v25 = b[v_bc_idx];
                float v_b_n = static_cast<float>(v25);
                auto v29 = ((v_dt_val * v_b_n) * v_x_d);
                auto v_new_h = ((v_decay * v_h_old) + v29);
                h[v_h_idx] = v_new_h;
                auto v31 = c[v_bc_idx];
                auto v34 = (static_cast<float>(v31) * v_new_h);
                __ml_y_d = __ml_y_d + v34;
            }
            y[t_i * d_inner + v3 + v_d] = static_cast<float>(__ml_y_d);
        }
    }

    kernel void nemotron_mixer_ssm_scan_chunk_f16(
        const device half *x [[buffer(0)]],
        const device half *a [[buffer(1)]],
        const device half *b [[buffer(2)]],
        const device half *c [[buffer(3)]],
        const device half *dt [[buffer(4)]],
        device float *h [[buffer(5)]],
        device half *y [[buffer(6)]],
        constant uint &head_dim [[buffer(7)]],
        constant uint &state_dim [[buffer(8)]],
        constant uint &heads_per_group [[buffer(9)]],
        constant uint &chunk_len [[buffer(10)]],
        constant uint &d_inner [[buffer(11)]],
        constant uint &bc_stride [[buffer(12)]],
        constant uint &n_heads [[buffer(13)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint v_idx = gid.x;
        uint v1 = head_dim;
        uint v_h_id = v_idx / v1;
        uint v3 = v_h_id * v1;
        uint v_d = v_idx - v3;
        uint v_g_id = v_h_id / heads_per_group;
        uint v_bc_base0 = v_g_id * state_dim;
        uint v_h_base = (v_h_id * state_dim) * v1;
        float v_a = static_cast<float>(a[v_h_id]);
        for (uint t_i = 0; t_i < chunk_len; t_i += 1) {
            auto v5 = dt[t_i * n_heads + v_h_id];
            float v_dt_val = static_cast<float>(v5);
            auto v9 = (v_a * v_dt_val);
            auto v_decay = exp(v9);
            auto v13 = x[t_i * d_inner + v3 + v_d];
            float v_x_d = static_cast<float>(v13);
            float __ml_y_d = 0.0f;
            uint v_bc_row = t_i * bc_stride + v_bc_base0;
            for (uint i_0 = 0; i_0 < state_dim; i_0 += 1) {
                uint v_h_idx = v_h_base + i_0 * v1 + v_d;
                auto v_h_old = h[v_h_idx];
                uint v_bc_idx = v_bc_row + i_0;
                auto v25 = b[v_bc_idx];
                float v_b_n = static_cast<float>(v25);
                auto v29 = ((v_dt_val * v_b_n) * v_x_d);
                auto v_new_h = ((v_decay * v_h_old) + v29);
                h[v_h_idx] = v_new_h;
                auto v31 = c[v_bc_idx];
                auto v34 = (static_cast<float>(v31) * v_new_h);
                __ml_y_d = __ml_y_d + v34;
            }
            half v39 = static_cast<half>(__ml_y_d);
            y[t_i * d_inner + v3 + v_d] = v39;
        }
    }

    kernel void nemotron_mixer_ssm_scan_chunk_bf16(
        const device bfloat *x [[buffer(0)]],
        const device bfloat *a [[buffer(1)]],
        const device bfloat *b [[buffer(2)]],
        const device bfloat *c [[buffer(3)]],
        const device bfloat *dt [[buffer(4)]],
        device float *h [[buffer(5)]],
        device bfloat *y [[buffer(6)]],
        constant uint &head_dim [[buffer(7)]],
        constant uint &state_dim [[buffer(8)]],
        constant uint &heads_per_group [[buffer(9)]],
        constant uint &chunk_len [[buffer(10)]],
        constant uint &d_inner [[buffer(11)]],
        constant uint &bc_stride [[buffer(12)]],
        constant uint &n_heads [[buffer(13)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint v_idx = gid.x;
        uint v1 = head_dim;
        uint v_h_id = v_idx / v1;
        uint v3 = v_h_id * v1;
        uint v_d = v_idx - v3;
        uint v_g_id = v_h_id / heads_per_group;
        uint v_bc_base0 = v_g_id * state_dim;
        uint v_h_base = (v_h_id * state_dim) * v1;
        float v_a = static_cast<float>(a[v_h_id]);
        for (uint t_i = 0; t_i < chunk_len; t_i += 1) {
            auto v5 = dt[t_i * n_heads + v_h_id];
            float v_dt_val = static_cast<float>(v5);
            auto v9 = (v_a * v_dt_val);
            auto v_decay = exp(v9);
            auto v13 = x[t_i * d_inner + v3 + v_d];
            float v_x_d = static_cast<float>(v13);
            float __ml_y_d = 0.0f;
            uint v_bc_row = t_i * bc_stride + v_bc_base0;
            for (uint i_0 = 0; i_0 < state_dim; i_0 += 1) {
                uint v_h_idx = v_h_base + i_0 * v1 + v_d;
                auto v_h_old = h[v_h_idx];
                uint v_bc_idx = v_bc_row + i_0;
                auto v25 = b[v_bc_idx];
                float v_b_n = static_cast<float>(v25);
                auto v29 = ((v_dt_val * v_b_n) * v_x_d);
                auto v_new_h = ((v_decay * v_h_old) + v29);
                h[v_h_idx] = v_new_h;
                auto v31 = c[v_bc_idx];
                auto v34 = (static_cast<float>(v31) * v_new_h);
                __ml_y_d = __ml_y_d + v34;
            }
            bfloat v39 = bfloat(__ml_y_d);
            y[t_i * d_inner + v3 + v_d] = v39;
        }
    }
    """

    /// One dispatch covering `chunkLen` tokens' worth of selective
    /// scan for every `(head, channel)` pair. `x`/`b`/`c`/`dt`/`y` are
    /// `[chunkLen, ...]` row-major slices (a `Tensor.slicedRows` view
    /// into the layer's full `[T, ...]` buffers — no copy). `state`
    /// (`layerCache.ssm.h`, `[nHeads, stateDim, headDim]`, always f32)
    /// carries in AND out — call once per chunk, in increasing
    /// token-offset order, on the SAME `cmd`, and Metal's automatic
    /// read-after-write hazard tracking across encoders serializes the
    /// chunks correctly with no host sync needed between them.
    static func scanChunk(
        x: Tensor, a: Tensor, b: Tensor, c: Tensor, dt: Tensor,
        state h: Tensor, into y: Tensor,
        nHeads: Int, headDim: Int, stateDim: Int, headsPerGroup: Int,
        chunkLen: Int, dInner: Int, bcStride: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) {
        precondition(h.dtype == .f32, "NemotronHMambaChunkedScan.scanChunk: state h must be f32")
        precondition(x.dtype == y.dtype, "NemotronHMambaChunkedScan.scanChunk: x/y dtype must match")
        precondition(
            a.dtype == x.dtype && b.dtype == x.dtype && c.dtype == x.dtype && dt.dtype == x.dtype,
            "NemotronHMambaChunkedScan.scanChunk: a/b/c/dt dtype must match x")
        let name: String
        switch x.dtype {
        case .f32: name = "nemotron_mixer_ssm_scan_chunk_f32"
        case .f16: name = "nemotron_mixer_ssm_scan_chunk_f16"
        case .bf16: name = "nemotron_mixer_ssm_scan_chunk_bf16"
        default: fatalError("NemotronHMambaChunkedScan.scanChunk: unsupported dtype \(x.dtype)")
        }
        // Matches `ssmStepGrouped`'s fast-math choice (iron's own
        // default) — needed for the two to reproduce the same compiled
        // arithmetic; verified in `NemotronHMambaChunkedScanTests`.
        let p = pso(name, source: ssmScanChunkSrc, device: device.mtlDevice, fastMath: true)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        enc.setBuffer(x.buffer, offset: x.offset, index: 0)
        enc.setBuffer(a.buffer, offset: a.offset, index: 1)
        enc.setBuffer(b.buffer, offset: b.offset, index: 2)
        enc.setBuffer(c.buffer, offset: c.offset, index: 3)
        enc.setBuffer(dt.buffer, offset: dt.offset, index: 4)
        enc.setBuffer(h.buffer, offset: h.offset, index: 5)
        enc.setBuffer(y.buffer, offset: y.offset, index: 6)
        var headDimV = UInt32(headDim)
        enc.setBytes(&headDimV, length: 4, index: 7)
        var stateDimV = UInt32(stateDim)
        enc.setBytes(&stateDimV, length: 4, index: 8)
        var headsPerGroupV = UInt32(headsPerGroup)
        enc.setBytes(&headsPerGroupV, length: 4, index: 9)
        var chunkLenV = UInt32(chunkLen)
        enc.setBytes(&chunkLenV, length: 4, index: 10)
        var dInnerV = UInt32(dInner)
        enc.setBytes(&dInnerV, length: 4, index: 11)
        var bcStrideV = UInt32(bcStride)
        enc.setBytes(&bcStrideV, length: 4, index: 12)
        var nHeadsV = UInt32(nHeads)
        enc.setBytes(&nHeadsV, length: 4, index: 13)
        let n = nHeads * headDim
        let tg = MTLSize(width: min(256, n), height: 1, depth: 1)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    // ─── (b) byte-generic column slice ────────────────────────────────
    //
    // Extracts `[t, colCount]` from a combined `[t, totalCols]` row-major
    // tensor at column offset `colStart`. Pure byte copy (no dtype
    // dispatch needed) — used to split the combined `in_proj` output
    // (`z | xBC | dt_raw`) and the combined conv output (`x | B | C`)
    // into contiguous per-piece buffers.
    private static let sliceColumnsSrc = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void nemotron_mixer_slice_columns(
        const device uchar *src [[buffer(0)]],
        device uchar *dst [[buffer(1)]],
        constant uint &total_row_bytes [[buffer(2)]],
        constant uint &col_start_bytes [[buffer(3)]],
        constant uint &col_bytes [[buffer(4)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint b = gid.x;
        uint row = gid.y;
        dst[row * col_bytes + b] = src[row * total_row_bytes + col_start_bytes + b];
    }
    """

    static func sliceColumns(
        _ src: Tensor, colStart: Int, colCount: Int, totalCols: Int,
        t: Int, device: Device, on cmd: MTLCommandBuffer
    ) -> Tensor {
        precondition(
            src.elementCount == t * totalCols,
            "NemotronHMambaChunkedScan.sliceColumns: shape mismatch "
                + "(\(src.elementCount) vs \(t)*\(totalCols))")
        let elemSize = src.dtype.byteSize
        let dst = Tensor.empty(shape: [t, colCount], dtype: src.dtype, device: device)
        let p = pso(
            "nemotron_mixer_slice_columns", source: sliceColumnsSrc,
            device: device.mtlDevice, fastMath: true)
        guard let enc = cmd.makeComputeCommandEncoder() else { return dst }
        enc.setComputePipelineState(p)
        enc.setBuffer(src.buffer, offset: src.offset, index: 0)
        enc.setBuffer(dst.buffer, offset: dst.offset, index: 1)
        var totalRowBytes = UInt32(totalCols * elemSize)
        enc.setBytes(&totalRowBytes, length: 4, index: 2)
        var colStartBytes = UInt32(colStart * elemSize)
        enc.setBytes(&colStartBytes, length: 4, index: 3)
        var colBytes = UInt32(colCount * elemSize)
        enc.setBytes(&colBytes, length: 4, index: 4)
        let grid = MTLSize(width: colCount * elemSize, height: t, depth: 1)
        let tg = MTLSize(width: min(256, colCount * elemSize), height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        return dst
    }

    // ─── (c) byte-generic row broadcast tile ──────────────────────────
    //
    // Repeats a `[rowLen]` vector into `[t, rowLen]`. Pure byte copy —
    // used to broadcast per-channel constants (`D`, `mixer.norm.weight`,
    // `dt_bias`) across T rows so they can feed the exact-shape-only
    // `Ops.add`/`Ops.mul`/`NemotronHMambaMixerFuse.addMul` the
    // sequential path already uses (no new arithmetic kernel — the
    // fusion class that needs care, e.g. addMul, is reused as-is on the
    // tiled result, so it inherits that fusion's existing bit-exactness
    // verification unchanged).
    private static let rowBroadcastTileSrc = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void nemotron_mixer_row_bcast_tile(
        const device uchar *src [[buffer(0)]],
        device uchar *dst [[buffer(1)]],
        constant uint &row_bytes [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint b = gid.x;
        uint row = gid.y;
        dst[row * row_bytes + b] = src[b];
    }
    """

    static func rowBroadcastTile(
        _ src: Tensor, t: Int, device: Device, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let rowLen = src.elementCount
        let elemSize = src.dtype.byteSize
        let dst = Tensor.empty(shape: [t, rowLen], dtype: src.dtype, device: device)
        let p = pso(
            "nemotron_mixer_row_bcast_tile", source: rowBroadcastTileSrc,
            device: device.mtlDevice, fastMath: true)
        guard let enc = cmd.makeComputeCommandEncoder() else { return dst }
        enc.setComputePipelineState(p)
        enc.setBuffer(src.buffer, offset: src.offset, index: 0)
        enc.setBuffer(dst.buffer, offset: dst.offset, index: 1)
        var rowBytes = UInt32(rowLen * elemSize)
        enc.setBytes(&rowBytes, length: 4, index: 2)
        let grid = MTLSize(width: rowLen * elemSize, height: t, depth: 1)
        let tg = MTLSize(width: min(256, rowLen * elemSize), height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        return dst
    }

    // ─── (d) batched conv1d, native-dtype output ──────────────────────
    //
    // NOT `Ops.conv1dCausalStepSiluCastMany` — that existing kernel
    // fuses conv AND silu with a SINGLE f32 accumulator and only casts
    // to the activation dtype at the very end. `decode`'s per-token
    // path does NOT do that: `Ops.conv1dCausalStep` rounds the conv
    // accumulator down to the native dtype FIRST (matching the shipped
    // `iron_conv1d_causal_step_*` kernel — see e.g. `iron_conv1d_
    // causal_step_bf16.metal`'s `y[v_d] = bfloat(v34)`), and only THEN
    // does a SEPARATE `Ops.silu` dispatch read that already-rounded
    // value back. Skipping that intermediate round (as the fused-many
    // kernel does) is exactly the "naive fused kernel that only rounds
    // once at the very end" hazard `NemotronHMambaMixerFuse`'s header
    // warns about — measured here: it visibly shortened the chunked-
    // vs-sequential greedy continuation match well past what pure
    // recurrence re-association explains.
    //
    // This kernel is `iron_conv1d_causal_step_silu_cast_many_*`'s
    // struct (per-channel register-resident conv-tap state across a
    // T-token sweep, one dispatch) with the silu fusion REMOVED and the
    // output cast point moved to match `iron_conv1d_causal_step_*`
    // exactly (round once, right after the conv accumulate — same f32
    // accumulator, same per-dtype output cast). `Ops.silu` (the exact
    // kernel `decode` already calls) runs as a SEPARATE dispatch
    // afterward on the whole `[t, convDim]` result, which reproduces
    // `decode`'s per-dtype silu precision exactly (native `half`
    // compute for f16, f32-round-trip for bf16 — see `Ops.silu`'s own
    // kernel) with no new kernel needed for that half.
    //
    // Verified bit-exact vs `t` sequential `Ops.conv1dCausalStep` calls
    // at f16/bf16 (`NemotronHMambaChunkedScanTests`) — the dtypes every
    // shipped NemotronH checkpoint actually runs its mixer in. f32
    // alone lands ~1 ULP off on about half the elements: a GPU
    // FMA-contraction artifact (the accumulation statements are
    // textually identical to the reference kernel's, compiled under
    // the same fast-math setting, but the surrounding loop shape still
    // lets the compiler pick a different fused-multiply-add schedule)
    // — the same "no source-level lever moves it" class of hazard
    // `NemotronHMambaMixerFuse`'s header documents for `mulSilu`. Not
    // chased further given f16/bf16 (the dtypes that matter here) are
    // exact; f32 checks a tight tolerance in the test instead of
    // strict equality.
    private static let conv1dManyNativeSrc = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void nemotron_mixer_conv1d_many_native_f32(
        const device float *src [[buffer(0)]],
        const device float *w [[buffer(1)]],
        const device float *b [[buffer(2)]],
        const device float *state_in [[buffer(3)]],
        device float *out [[buffer(4)]],
        device float *state_out [[buffer(5)]],
        constant uint &t_len [[buffer(6)]],
        constant uint &conv_dim [[buffer(7)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint v_d = gid.x;
        float v_b_d = float(b[v_d]);
        uint v4 = conv_dim;
        float v_w0 = float(w[v_d]);
        float v_w1 = float(w[v4 + v_d]);
        float v_w2 = float(w[2u * v4 + v_d]);
        float v_w3 = float(w[3u * v4 + v_d]);
        float __ml_s0 = float(state_in[v_d]);
        float __ml_s1 = float(state_in[v4 + v_d]);
        float __ml_s2 = float(state_in[2u * v4 + v_d]);
        for (uint i_0 = 0; i_0 < t_len; i_0 += 1) {
            float v_x_r = float(src[i_0 * v4 + v_d]);
            float v_acc = v_b_d + v_w3 * v_x_r;
            v_acc = v_acc + v_w0 * __ml_s0;
            v_acc = v_acc + v_w1 * __ml_s1;
            v_acc = v_acc + v_w2 * __ml_s2;
            out[i_0 * v4 + v_d] = v_acc;
            __ml_s0 = __ml_s1;
            __ml_s1 = __ml_s2;
            __ml_s2 = v_x_r;
        }
        state_out[v_d] = __ml_s0;
        state_out[v4 + v_d] = __ml_s1;
        state_out[2u * v4 + v_d] = __ml_s2;
    }

    kernel void nemotron_mixer_conv1d_many_native_f16(
        const device half *src [[buffer(0)]],
        const device half *w [[buffer(1)]],
        const device half *b [[buffer(2)]],
        const device half *state_in [[buffer(3)]],
        device half *out [[buffer(4)]],
        device half *state_out [[buffer(5)]],
        constant uint &t_len [[buffer(6)]],
        constant uint &conv_dim [[buffer(7)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint v_d = gid.x;
        float v_b_d = float(b[v_d]);
        uint v4 = conv_dim;
        float v_w0 = float(w[v_d]);
        float v_w1 = float(w[v4 + v_d]);
        float v_w2 = float(w[2u * v4 + v_d]);
        float v_w3 = float(w[3u * v4 + v_d]);
        float __ml_s0 = float(state_in[v_d]);
        float __ml_s1 = float(state_in[v4 + v_d]);
        float __ml_s2 = float(state_in[2u * v4 + v_d]);
        for (uint i_0 = 0; i_0 < t_len; i_0 += 1) {
            float v_x_r = float(src[i_0 * v4 + v_d]);
            float v_acc = v_b_d + v_w3 * v_x_r;
            v_acc = v_acc + v_w0 * __ml_s0;
            v_acc = v_acc + v_w1 * __ml_s1;
            v_acc = v_acc + v_w2 * __ml_s2;
            out[i_0 * v4 + v_d] = static_cast<half>(v_acc);
            __ml_s0 = __ml_s1;
            __ml_s1 = __ml_s2;
            __ml_s2 = v_x_r;
        }
        state_out[v_d] = static_cast<half>(__ml_s0);
        state_out[v4 + v_d] = static_cast<half>(__ml_s1);
        state_out[2u * v4 + v_d] = static_cast<half>(__ml_s2);
    }

    kernel void nemotron_mixer_conv1d_many_native_bf16(
        const device bfloat *src [[buffer(0)]],
        const device bfloat *w [[buffer(1)]],
        const device bfloat *b [[buffer(2)]],
        const device bfloat *state_in [[buffer(3)]],
        device bfloat *out [[buffer(4)]],
        device bfloat *state_out [[buffer(5)]],
        constant uint &t_len [[buffer(6)]],
        constant uint &conv_dim [[buffer(7)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint v_d = gid.x;
        float v_b_d = float(b[v_d]);
        uint v4 = conv_dim;
        float v_w0 = float(w[v_d]);
        float v_w1 = float(w[v4 + v_d]);
        float v_w2 = float(w[2u * v4 + v_d]);
        float v_w3 = float(w[3u * v4 + v_d]);
        float __ml_s0 = float(state_in[v_d]);
        float __ml_s1 = float(state_in[v4 + v_d]);
        float __ml_s2 = float(state_in[2u * v4 + v_d]);
        for (uint i_0 = 0; i_0 < t_len; i_0 += 1) {
            float v_x_r = float(src[i_0 * v4 + v_d]);
            float v_acc = v_b_d + v_w3 * v_x_r;
            v_acc = v_acc + v_w0 * __ml_s0;
            v_acc = v_acc + v_w1 * __ml_s1;
            v_acc = v_acc + v_w2 * __ml_s2;
            out[i_0 * v4 + v_d] = bfloat(v_acc);
            __ml_s0 = __ml_s1;
            __ml_s1 = __ml_s2;
            __ml_s2 = v_x_r;
        }
        state_out[v_d] = bfloat(__ml_s0);
        state_out[v4 + v_d] = bfloat(__ml_s1);
        state_out[2u * v4 + v_d] = bfloat(__ml_s2);
    }
    """

    /// Batched depthwise causal conv1d (kernel size 4 — the only size
    /// NemotronH/Qwen3.5/Mamba2 ship) over `t` tokens in ONE dispatch,
    /// state held in per-channel registers across the sweep (state read
    /// once at the top, written once at the bottom — same shape as
    /// `Ops.conv1dCausalStepSiluCastMany`). Output stays in the SAME
    /// dtype as the input (unlike that kernel, which always emits f32)
    /// — see the file comment above for why that distinction is
    /// load-bearing here. Caller still needs a separate `Ops.silu` call
    /// on the result to match `decode`'s conv1d-then-silu two-step.
    static func conv1dMany(
        src: Tensor, w: Tensor, b: Tensor, stateIn: Tensor, stateOut: Tensor,
        t: Int, convDim: Int, convKernel: Int,
        device: Device, on cmd: MTLCommandBuffer
    ) -> Tensor {
        precondition(
            convKernel == 4,
            "NemotronHMambaChunkedScan.conv1dMany: requires conv_kernel = 4")
        precondition(
            src.dtype == w.dtype && src.dtype == b.dtype && src.dtype == stateIn.dtype,
            "NemotronHMambaChunkedScan.conv1dMany: dtype mismatch")
        let dst = Tensor.empty(shape: [t, convDim], dtype: src.dtype, device: device)
        let name: String
        switch src.dtype {
        case .f32: name = "nemotron_mixer_conv1d_many_native_f32"
        case .f16: name = "nemotron_mixer_conv1d_many_native_f16"
        case .bf16: name = "nemotron_mixer_conv1d_many_native_bf16"
        default: fatalError("NemotronHMambaChunkedScan.conv1dMany: unsupported dtype \(src.dtype)")
        }
        let p = pso(
            name, source: conv1dManyNativeSrc, device: device.mtlDevice, fastMath: true)
        guard let enc = cmd.makeComputeCommandEncoder() else { return dst }
        enc.setComputePipelineState(p)
        enc.setBuffer(src.buffer, offset: src.offset, index: 0)
        enc.setBuffer(w.buffer, offset: w.offset, index: 1)
        enc.setBuffer(b.buffer, offset: b.offset, index: 2)
        enc.setBuffer(stateIn.buffer, offset: stateIn.offset, index: 3)
        enc.setBuffer(dst.buffer, offset: dst.offset, index: 4)
        enc.setBuffer(stateOut.buffer, offset: stateOut.offset, index: 5)
        var tLen = UInt32(t)
        enc.setBytes(&tLen, length: 4, index: 6)
        var convDimV = UInt32(convDim)
        enc.setBytes(&convDimV, length: 4, index: 7)
        let grid = MTLSize(width: convDim, height: 1, depth: 1)
        let tg = MTLSize(width: min(256, convDim), height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        return dst
    }
}
