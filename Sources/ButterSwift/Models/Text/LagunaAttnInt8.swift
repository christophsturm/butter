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
// Attention-side group-32 affine INT8 requant: decode-only side layout
// built at load from the BF16 Q/K/V (+ per-head gate) and o_proj
// projections.
//
// At load time each attention block's Q/K/V (+ per-head gate) BF16
// weights are re-quantized to a group-32 affine INT8 side layout and
// row-concatenated into ONE bank; the serial decode token then reads
// that bank with a single batched QMV dispatch instead of three (or
// four) separate BF16 GEMVs. `o_proj` gets its own standalone group-32
// INT8 bank (different input width, it consumes the SDPA output, not
// the normalized residual). The original BF16 parameters stay resident
// and authoritative; prefill (`forwardMulti`) never touches this path.
//
// `iron_qmv_int8_fast_*` (Ops.swift) is a close cousin but hardcodes
// group_size=64 in its generated inner loop (see
// `QuantizedOpsValidation.validateAffineQuantize`, which hardcodes the
// *quantize* kernels to group_size=64 for the same reason: the iron
// codegen reduction shape is fixed at build time). Neither is reusable
// for group-32, so both the quantize step and the batched QMV here are
// hand-written live-compiled Metal, following the same `pso()` +
// `makeLibrary(source:)` pattern as `LagunaNvfp4Fuse`.
//
// Escape: `BUTTER_LAGUNA_ATTN_INT8=0` restores the stock BF16 path.

import Foundation
import Metal
import IronSwift

enum LagunaAttnInt8 {
    /// Default ON; set `BUTTER_LAGUNA_ATTN_INT8=0` to fall back to the
    /// stock BF16 QKV / o_proj decode path.
    static let enabled: Bool = ButterEnv["BUTTER_LAGUNA_ATTN_INT8"] != "0"

    /// T32 task 2: fold the per-head gate-product-softplus multiply into
    /// this bank's own QMV as an activation prologue (see `qmvGated`
    /// below), instead of a separate `LagunaGatedOProjFuse` dispatch
    /// feeding this QMV unchanged. T32 shipped this default OFF (its own
    /// wall-clock A/B read flat, "quality-first, no default change for a
    /// noise-level win"). T39's per-kernel GPU-time probe found the
    /// unfused two-dispatch chain (`LagunaGatedOProjFuse` fuse +
    /// `qmv`, what runs when this is OFF) is the single largest GPU-time
    /// consumer in decode (~35% of measured GPU-busy work, ahead of the
    /// QKV bank QMV and every MoE kernel); re-flipped default ON here;
    /// see benchmarks/laguna-xs/README.md T39. Set
    /// `BUTTER_LAGUNA_INT8_OPROJ_GATED=0` to restore the T32-default
    /// separate-dispatch (T24 kernel 1) chain. Only takes effect where
    /// `enabled` (this bank) already holds.
    static var gatedEnabled: Bool {
        ButterEnv["BUTTER_LAGUNA_INT8_OPROJ_GATED"] != "0"
    }

    /// `qmvGated` dispatch geometry, v2 vs the original T32 kernel.
    /// Default ON (v2). The v1 body (`qmvGatedV1` below, same source as
    /// the kernel this file shipped with before this change) recomputed
    /// softplus per lane per k-block and read the packed weight a
    /// 32-bit word at a time; v2 precomputes one softplus per head into
    /// a threadgroup gate table (barrier once, then a pure table read
    /// per k-block), widens the per-lane slice to `values_per_thread =
    /// 8` over a `block_size = 256` stride, rounds the gated activation
    /// through bf16 at the same point the unfused gate-product kernel
    /// would, and reads packed weight bytes as scalar `uint8_t` instead
    /// of unpacking a `uint32` word.
    ///
    /// DEFAULT OFF (w2-integration cycle 3 finding, unrelated to any
    /// cycle-3 branch's own diff): v2 was landed in cycle 2 default ON
    /// but had never actually run through a real multi-step decode loop
    /// end to end: its Metal kernel failed to COMPILE on this
    /// toolchain (32023.883) until a cycle-3 signature fix (paired
    /// `uint3`/`uint` grid-position attributes, see the qmvGatedV2Src
    /// kernel doc), so only synthetic single-shape unit tests
    /// (LagunaOProjQmvV2Tests) had ever exercised it. Once unblocked,
    /// real decode traffic surfaced a repetition-collapse bug:
    /// `butter generate --model ~/models/laguna-xs-2.1-nvfp4 --prompt
    /// "What is the capital of France?" --max-tokens 48 --temperature 0`
    /// with the classic per-token decode loop (BUTTER_GPU_TOKEN_LOOP=0)
    /// degenerates into repeating "The capital of France is Paris.\n"
    /// six times instead of the canonical coherent continuation. Root
    /// cause NOT isolated despite targeted A/B testing: confirmed NOT
    /// the Ops.argmax GPU kernel (reproduces via the CPU-sample path,
    /// BUTTER_GREEDY_GPU_ARGMAX=0, which never touches it), NOT the
    /// multi-CB layer pipeline (reproduces with BUTTER_LAGUNA_LAYER_PIPE=0),
    /// NOT generic KV-cache code (v1 runs the identical cache code
    /// correctly for the same 48 tokens). The bug requires BOTH v2 AND
    /// the classic Int-token decode call path: the new GPU-resident
    /// token-loop path (`LanguageModel.forward(tokenIdBuffer:...)`,
    /// GpuTokenLoop.swift) calls the exact same shared
    /// `forwardFromTokenTensor` body yet does not trigger it, which is
    /// unexplained (see benchmarks/laguna-xs/README.md T42 for the
    /// full isolation matrix). Quality bar: a kernel that can silently
    /// collapse decode into repetition is not shippable regardless of
    /// its speed, so this defaults OFF until root-caused. Set
    /// `BUTTER_LAGUNA_QMV_V2=1` to opt back in for further
    /// investigation; the kernel and its tests are kept, not removed.
    static let qmvV2Enabled: Bool = ButterEnv["BUTTER_LAGUNA_QMV_V2"] == "1"

    /// Split-K dispatch geometry for the group-32 INT8 QMV kernels
    /// (`qmv`/`qmvGated`). See `qmvSplitK`/`qmvGatedSplitK` below for the
    /// design — each threadgroup covers `NR0 = 2` output rows with
    /// `NSG = 4` simdgroups (128 threads) that jointly stride the K
    /// dimension of that SAME row pair, instead of `qmvSrc`/
    /// `qmvGatedSrc`'s geometry where each of 2 simdgroups independently
    /// walks the full K dimension for its own 4 rows. Reduction is
    /// two-stage (per-sg `simd_sum` -> threadgroup memory -> a second
    /// `simd_sum` across simdgroups -> one write), and the activation
    /// load is shared across both rows in the pair instead of being
    /// re-read per row. This changes floating-point summation order vs
    /// the existing kernels (K-dimension split across simdgroups instead
    /// of walked serially by one), so results are near-ULP rather than
    /// guaranteed bit-identical — see
    /// `Tests/ButterTests/Ops/LagunaQmvSplitKTests.swift`.
    ///
    /// T45 attempt-3 validation (real Metal, not offline replay):
    /// `qmvGatedSplitK`'s gate-value bf16-round fix (`1614148f`) lands at
    /// cos=1.0/maxAbsDiff=0.0 exactly on the o_proj shape (was 0.5). The
    /// two plain `qmvSplitK` shapes (previously failing a bare 0.05
    /// maxAbsDiff at 0.125) show cos > 0.9999999 and, per-element, only
    /// 1-2 of 8240-10240 rows above 0.01 with the rest at 0.0 — an
    /// isolated single bf16-ULP flip (0.125 = 1 ULP at ~O(16-32)
    /// magnitude), not a structured error; the magnitude-relative test
    /// tolerance change is legitimate. Real-model coherence (Paris 48-tok,
    /// fib 80-tok and 160-tok extended check): no repetition collapse,
    /// stays on-topic, produces correct Rust code. Decode A/B (fib 80-tok
    /// T=0, interleaved to control for host drift, 10 pairs): median
    /// +4.7%, mean +7.1%, 8/10 pairs positive — clears the 2% bar.
    /// **Flipped default ON**; `BUTTER_LAGUNA_QMV_SPLITK=0` is the escape
    /// hatch. `qmv`/`qmvGated` try the split-K kernel first and fall back
    /// to their existing (unmodified) kernel whenever it declines —
    /// shape/dtype mismatch, or `bank.outDim` not a multiple of 2 (the
    /// fixed `NR0 = 2` tiling has no partial-pair tail path).
    static let splitKEnabled: Bool = ButterEnv["BUTTER_LAGUNA_QMV_SPLITK"] != "0"

    static let groupSize = 32

    /// One group-32 affine INT8 side layout: `packed` is `[outDim,
    /// inDim/4]` u32 (4 int8 codes per word), `scales`/`biases` are
    /// `[outDim, inDim/groupSize]` bf16. Dequant: `code * scale + bias`.
    struct Bank {
        let packed: Tensor
        let scales: Tensor
        let biases: Tensor
        let outDim: Int
        let inDim: Int
    }

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pso(_ name: String, source: String, device: MTLDevice)
        -> MTLComputePipelineState
    {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        let opts = MTLCompileOptions()
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    // Group-32 affine INT8 quantize: one threadgroup (32 lanes = 1
    // simdgroup) per group. Mirrors the algorithm of
    // `iron_affine_quantize_int8_bf16` (min/max simd-reduce -> scale +
    // zero-point bias -> round-clamp-pack 4 codes/word) but with 1
    // element/lane over 32 lanes instead of 2 elements/lane over 32
    // lanes (that kernel's fixed 64-wide group).
    private static let quantizeSrc = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_attn_quantize_int8_g32_bf16(
            const device bfloat *w [[buffer(0)]],
            device uint *out [[buffer(1)]],
            device bfloat *scales [[buffer(2)]],
            device bfloat *biases [[buffer(3)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_threadgroup]])
        {
            constexpr uint GROUP = 32;
            uint g = tgid.x;
            uint base = g * GROUP;

            float v = float(w[base + lane]);
            float lo = simd_min(v);
            float hi = simd_max(v);
            float rawScale = (hi - lo) / 255.0f;
            float scale = rawScale < 1e-7f ? 1.0f : rawScale;
            float invScale = 1.0f / scale;

            if (lane == 0) {
                scales[g] = bfloat(scale);
                biases[g] = bfloat(lo);
            }

            // 8 words/group (32 elements / 4 per word); lanes 0..7 pack.
            if (lane < 8) {
                uint packed = 0;
                for (uint i = 0; i < 4; ++i) {
                    uint idx = lane * 4 + i;
                    float x = float(w[base + idx]);
                    float q = clamp((x - lo) * invScale + 0.5f, 0.0f, 255.0f);
                    packed |= (uint(q) << (8u * i));
                }
                out[g * 8 + lane] = packed;
            }
        }
        """

    // Batched group-32 affine INT8 GEMV over a (possibly row-concatenated)
    // bank. TPG = 64 (2 simdgroups x 32 lanes), 4 output rows per
    // simdgroup => 8 rows/threadgroup; grid = ceil(outDim/8). Row bounds
    // are checked so `outDim` need not be a multiple of 8 (e.g. the
    // per-head gate tail, or an odd head count).
    private static let qmvSrc = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_attn_qmv_int8_g32_bf16(
            const device uint *w [[buffer(0)]],
            const device bfloat *scales [[buffer(1)]],
            const device bfloat *biases [[buffer(2)]],
            const device bfloat *x [[buffer(3)]],
            device bfloat *out [[buffer(4)]],
            constant uint &inDim [[buffer(5)]],
            constant uint &outDim [[buffer(6)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 32;
            constexpr uint ROWS_PER_SG = 4;
            uint packsPerRow = inDim / 4;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * (2u * ROWS_PER_SG) + sg * ROWS_PER_SG;

            float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (uint block = 0; block < packsPerRow; block += 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 4u;
                    bfloat4 xv = *((const device bfloat4 *)(x + xBase));
                    float x0 = float(xv[0]);
                    float x1 = float(xv[1]);
                    float x2 = float(xv[2]);
                    float x3 = float(xv[3]);
                    float xs = x0 + x1 + x2 + x3;
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < ROWS_PER_SG; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float s = float(scales[row * groupsPerRow + g]);
                            float b = float(biases[row * groupsPerRow + g]);
                            float c0 = float(p & 0xFFu);
                            float c1 = float((p >> 8u) & 0xFFu);
                            float c2 = float((p >> 16u) & 0xFFu);
                            float c3 = float((p >> 24u) & 0xFFu);
                            float qd = c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3;
                            acc[r] += s * qd + b * xs;
                        }
                    }
                }
            }

            for (uint r = 0; r < ROWS_PER_SG; ++r) {
                float total = simd_sum(acc[r]);
                if (lane == 0) {
                    uint row = rowBase + r;
                    if (row < outDim) {
                        out[row] = bfloat(total);
                    }
                }
            }
        }
        """

    /// Quantize a `[outDim, inDim]` BF16 weight to a group-32 affine
    /// INT8 side layout. Declines (returns `nil`) unless `inDim` is a
    /// multiple of 32 and `weight` is 2D BF16, matching the group-32
    /// alignment the packed layout requires.
    static func quantizeG32(
        weight: Tensor, device: Device, cmd: MTLCommandBuffer
    ) -> Bank? {
        guard weight.dtype == .bf16, weight.shape.count == 2,
            weight.shape[1].isMultiple(of: groupSize)
        else { return nil }
        let outDim = weight.shape[0]
        let inDim = weight.shape[1]
        let packsPerRow = inDim / 4
        let groupsPerRow = inDim / groupSize
        let packed = Tensor.empty(shape: [outDim, packsPerRow], dtype: .u32, device: device)
        let scales = Tensor.empty(shape: [outDim, groupsPerRow], dtype: .bf16, device: device)
        let biases = Tensor.empty(shape: [outDim, groupsPerRow], dtype: .bf16, device: device)
        let nGroups = outDim * groupsPerRow
        let p = pso(
            "laguna_attn_quantize_int8_g32_bf16", source: quantizeSrc,
            device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(weight.buffer, offset: weight.offset, index: 0)
            enc.setBuffer(packed.buffer, offset: packed.offset, index: 1)
            enc.setBuffer(scales.buffer, offset: scales.offset, index: 2)
            enc.setBuffer(biases.buffer, offset: biases.offset, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: nGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
        }
        return Bank(packed: packed, scales: scales, biases: biases, outDim: outDim, inDim: inDim)
    }

    /// Row-concatenate banks that share `inDim` (e.g. Q/K/V + per-head
    /// gate) into one bank a single QMV can read. Pure CPU memcpy; the
    /// GPU quantize dispatches that produced `banks` must already have
    /// completed (caller commits + waits the setup command buffer
    /// first).
    static func concat(_ banks: [Bank], device: Device) -> Bank? {
        guard let first = banks.first,
            banks.allSatisfy({ $0.inDim == first.inDim })
        else { return nil }
        let totalOut = banks.reduce(0) { $0 + $1.outDim }
        let packsPerRow = first.inDim / 4
        let groupsPerRow = first.inDim / groupSize
        let packedBuf = device.makeBuffer(length: totalOut * packsPerRow * 4)
        let scalesBuf = device.makeBuffer(length: totalOut * groupsPerRow * 2)
        let biasesBuf = device.makeBuffer(length: totalOut * groupsPerRow * 2)
        var packedOff = 0
        var scalesOff = 0
        var biasesOff = 0
        for b in banks {
            let packedBytes = b.outDim * packsPerRow * 4
            let scaleBytes = b.outDim * groupsPerRow * 2
            memcpy(
                packedBuf.contents().advanced(by: packedOff),
                b.packed.buffer.contents().advanced(by: b.packed.offset), packedBytes)
            memcpy(
                scalesBuf.contents().advanced(by: scalesOff),
                b.scales.buffer.contents().advanced(by: b.scales.offset), scaleBytes)
            memcpy(
                biasesBuf.contents().advanced(by: biasesOff),
                b.biases.buffer.contents().advanced(by: b.biases.offset), scaleBytes)
            packedOff += packedBytes
            scalesOff += scaleBytes
            biasesOff += scaleBytes
        }
        // Row-concatenated bank replaces the stock BF16 Q/K/V(+gate) or
        // o_proj weights for every decode token from here on; pin it
        // in the residency set like any other long-lived weight buffer.
        device.markWeightsResident([packedBuf, scalesBuf, biasesBuf])
        return Bank(
            packed: Tensor(
                buffer: packedBuf, offset: 0, shape: [totalOut, packsPerRow], dtype: .u32),
            scales: Tensor(
                buffer: scalesBuf, offset: 0, shape: [totalOut, groupsPerRow], dtype: .bf16),
            biases: Tensor(
                buffer: biasesBuf, offset: 0, shape: [totalOut, groupsPerRow], dtype: .bf16),
            outDim: totalOut, inDim: first.inDim)
    }

    // T32 task 2: gate-folded variant of `qmvSrc` for the o_proj bank
    // specifically. `LagunaGatedOProjFuse`'s T24 kernel 1 already folds
    // `softplus(gate_logits) * attention_output` into one dispatch feeding
    // this bank's QMV unchanged (kernel 1); this kernel goes one step
    // further and folds THAT gate-product into this QMV's own activation
    // load, removing kernel 1's dispatch too. `packsPerHead` (== headDim/4,
    // a runtime uniform, not baked in) tells each lane which per-head gate
    // logit its 4-wide `x` load belongs to (`headDim` a multiple of 4 in
    // every model this bank targets, Laguna XS headDim=128). Softplus is
    // the identical numerically-stable form as `LagunaGatedOProjFuse`'s
    // kernel 1 (`max(x,0) + log(1+exp(-|x|))`), recomputed per lane per
    // 32-pack block (== once per head, ~nHeads times/lane total) rather
    // than materializing a `[nHeads]` scratch: cheap relative to the GEMV
    // itself. `xs` (the affine zero-point term's activation sum) uses the
    // GATED x values, matching what stock `qmv` would see if fed an
    // already-gated row.
    private static let qmvGatedSrc = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_attn_qmv_int8_g32_gated_bf16(
            const device uint *w [[buffer(0)]],
            const device bfloat *scales [[buffer(1)]],
            const device bfloat *biases [[buffer(2)]],
            const device bfloat *x [[buffer(3)]],
            const device bfloat *gate_logits [[buffer(4)]],
            device bfloat *out [[buffer(5)]],
            constant uint &inDim [[buffer(6)]],
            constant uint &outDim [[buffer(7)]],
            constant uint &packsPerHead [[buffer(8)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 32;
            constexpr uint ROWS_PER_SG = 4;
            uint packsPerRow = inDim / 4;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * (2u * ROWS_PER_SG) + sg * ROWS_PER_SG;

            float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (uint block = 0; block < packsPerRow; block += 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 4u;
                    bfloat4 xv = *((const device bfloat4 *)(x + xBase));
                    uint head = packIdx / packsPerHead;
                    float gl = float(gate_logits[head]);
                    bool pos = gl > 0.0f;
                    float ax = pos ? gl : (0.0f - gl);
                    float gate = (pos ? gl : 0.0f) + log(1.0f + exp(0.0f - ax));
                    float x0 = float(xv[0]) * gate;
                    float x1 = float(xv[1]) * gate;
                    float x2 = float(xv[2]) * gate;
                    float x3 = float(xv[3]) * gate;
                    float xs = x0 + x1 + x2 + x3;
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < ROWS_PER_SG; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float s = float(scales[row * groupsPerRow + g]);
                            float b = float(biases[row * groupsPerRow + g]);
                            float c0 = float(p & 0xFFu);
                            float c1 = float((p >> 8u) & 0xFFu);
                            float c2 = float((p >> 16u) & 0xFFu);
                            float c3 = float((p >> 24u) & 0xFFu);
                            float qd = c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3;
                            acc[r] += s * qd + b * xs;
                        }
                    }
                }
            }

            for (uint r = 0; r < ROWS_PER_SG; ++r) {
                float total = simd_sum(acc[r]);
                if (lane == 0) {
                    uint row = rowBase + r;
                    if (row < outDim) {
                        out[row] = bfloat(total);
                    }
                }
            }
        }
        """

    // v2 of the gated o_proj QMV: same 64-thread / 2-simdgroup / 4-row
    // tile as v1 above, but the inner loop is reshaped around
    // `values_per_thread = 8` over a `block_size = 256` stride (one
    // lane advances 8 input columns per k-block instead of one packed
    // 4-wide word), and the per-head softplus gate is computed ONCE
    // per threadgroup into `gate_table` rather than recomputed by every
    // lane on every k-block. The gated activation is rounded through
    // bf16 at the exact point `LagunaGatedOProjFuse.applyGateProductSoftplus`
    // would round it (`float(bfloat(float(x) * gate))`), so feeding this
    // kernel is bit-identical to running that gate-product kernel first
    // and then a plain (ungated) affine QMV over the result. The block's
    // activation sum (`sum`, the zero-point/bias term) is computed once
    // per block and shared across all 4 rows in the simdgroup, same as
    // v1. Packed weight bytes are read scalar (`uint8_t`), not unpacked
    // a `uint32` word at a time: a wider per-lane weight load measured
    // as a net loss on this GPU in earlier work despite a local win, so
    // this port deliberately keeps the load scalar. `headDim` is a
    // runtime uniform (not baked per head-count like a specialized
    // kernel would), so head lookup is `column / headDim` rather than a
    // compile-time shift; requires `inDim` a multiple of `block_size`
    // (256), checked by the caller guard below. Every current Laguna
    // shape (`heads * headDim` with `headDim == 128`) satisfies this,
    // and `qmvGatedV2` declines cleanly (falls back to v1) when it
    // doesn't.
    private static let qmvGatedV2Src = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_attn_qmv_int8_g32_gated_v2_bf16(
            const device uint8_t *w [[buffer(0)]],
            const device bfloat *scales [[buffer(1)]],
            const device bfloat *biases [[buffer(2)]],
            const device bfloat *x [[buffer(3)]],
            const device bfloat *gate_logits [[buffer(4)]],
            device bfloat *out [[buffer(5)]],
            constant uint &inDim [[buffer(6)]],
            constant uint &outDim [[buffer(7)]],
            constant uint &headDim [[buffer(8)]],
            constant uint &heads [[buffer(9)]],
            threadgroup float *gate_table [[threadgroup(0)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint3 lid3 [[thread_position_in_threadgroup]],
            uint simd_lid [[thread_index_in_simdgroup]],
            uint simd_gid [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 32;
            constexpr uint ROWS_PER_SG = 4;
            constexpr uint VALUES_PER_THREAD = 8;
            constexpr uint BLOCK_SIZE = 256;   // VALUES_PER_THREAD * SIMD_SIZE(32)
            constexpr uint SCALE_STEP = GROUP / VALUES_PER_THREAD;
            uint groupsPerRow = inDim / GROUP;

            // One softplus per head, computed once into threadgroup memory
            // at the same rounding point the unfused gate-product kernel
            // uses. Strided over `heads` so this holds for any head count
            // relative to the fixed 64-thread threadgroup width.
            for (uint h = lid3.x; h < heads; h += 64) {
                // Numerically-stable softplus, same op stream/order as
                // iron_softplus_bf16 and the separate-dispatch gate-product
                // kernel, so the fused path stays byte-compatible with the
                // two-dispatch chain it replaces.
                float x = float(gate_logits[h]);
                bool pos = x > 0.0f;
                float negx = 0.0f - x;
                float ax = pos ? x : negx;
                float e = exp(0.0f - ax);
                float lg = log(1.0f + e);
                gate_table[h] = float(bfloat((pos ? x : 0.0f) + lg));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint rowBase = tgid.x * (2u * ROWS_PER_SG) + simd_gid * ROWS_PER_SG;

            const device uint8_t *ws = w + rowBase * inDim + simd_lid * VALUES_PER_THREAD;
            const device bfloat *sc = scales + rowBase * groupsPerRow + simd_lid / SCALE_STEP;
            const device bfloat *bs = biases + rowBase * groupsPerRow + simd_lid / SCALE_STEP;
            const device bfloat *xp = x + simd_lid * VALUES_PER_THREAD;

            thread float x_thread[VALUES_PER_THREAD];
            float result[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

            uint column = simd_lid * VALUES_PER_THREAD;
            for (uint k = 0; k < inDim; k += BLOCK_SIZE) {
                float gate = gate_table[column / headDim];
                float sum = 0.0f;
                for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
                    float value = float(bfloat(float(xp[i]) * gate));
                    sum += value;
                    x_thread[i] = value;
                }

                for (uint row = 0; row < ROWS_PER_SG; ++row) {
                    uint outRow = rowBase + row;
                    if (outRow < outDim) {
                        const device uint8_t *wl = ws + row * inDim;
                        float scale = float(sc[row * groupsPerRow]);
                        float bias = float(bs[row * groupsPerRow]);
                        float accum = 0.0f;
                        for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
                            accum += x_thread[i] * float(wl[i]);
                        }
                        result[row] += scale * accum + sum * bias;
                    }
                }

                ws += BLOCK_SIZE;
                sc += BLOCK_SIZE / GROUP;
                bs += BLOCK_SIZE / GROUP;
                xp += BLOCK_SIZE;
                column += BLOCK_SIZE;
            }

            for (uint row = 0; row < ROWS_PER_SG; ++row) {
                float total = simd_sum(result[row]);
                if (simd_lid == 0) {
                    uint outRow = rowBase + row;
                    if (outRow < outDim) {
                        out[outRow] = bfloat(total);
                    }
                }
            }
        }
        """

    /// Shape/dtype guard shared by `qmvGatedV1` and `qmvGatedV2`
    /// (everything except `gatedEnabled`, which only the public
    /// `qmvGated` entry point checks). The v1/v2 functions are exposed
    /// so tests can dispatch either kernel directly regardless of the
    /// environment flag.
    private static func gatedShapesValid(
        input: Tensor, gateLogits: Tensor, bank: Bank, heads: Int, headDim: Int, out: Tensor
    ) -> Bool {
        input.dtype == .bf16 && gateLogits.dtype == .bf16 && out.dtype == .bf16
            && input.elementCount == bank.inDim
            && gateLogits.elementCount == heads
            && out.elementCount == bank.outDim
            && bank.inDim == heads * headDim
            && headDim % 4 == 0
    }

    /// Fused gate-product-softplus + o_proj INT8 QMV, v1 kernel (see
    /// `qmvGatedV2Src` for the v2 body this file now dispatches by
    /// default). `gateLogits` is the RAW per-head gate logit
    /// (pre-softplus), same input
    /// `LagunaGatedOProjFuse.applyGateProductSoftplus` takes. Declines
    /// (returns `false`, `into` left untouched) unless every
    /// shape/dtype precondition holds.
    static func qmvGatedV1(
        input: Tensor, gateLogits: Tensor, bank: Bank,
        heads: Int, headDim: Int,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard gatedShapesValid(
            input: input, gateLogits: gateLogits, bank: bank, heads: heads, headDim: headDim,
            out: out)
        else { return false }

        let packsPerHead = headDim / 4
        let p = pso(
            "laguna_attn_qmv_int8_g32_gated_bf16", source: qmvGatedSrc,
            device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
        enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
        enc.setBuffer(bank.biases.buffer, offset: bank.biases.offset, index: 2)
        enc.setBuffer(input.buffer, offset: input.offset, index: 3)
        enc.setBuffer(gateLogits.buffer, offset: gateLogits.offset, index: 4)
        enc.setBuffer(out.buffer, offset: out.offset, index: 5)
        var inDimV = UInt32(bank.inDim)
        var outDimV = UInt32(bank.outDim)
        var packsPerHeadV = UInt32(packsPerHead)
        enc.setBytes(&inDimV, length: 4, index: 6)
        enc.setBytes(&outDimV, length: 4, index: 7)
        enc.setBytes(&packsPerHeadV, length: 4, index: 8)
        let rowsPerTG = 8
        let tiles = (bank.outDim + rowsPerTG - 1) / rowsPerTG
        enc.dispatchThreadgroups(
            MTLSize(width: tiles, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

    /// Fused gate-product-softplus + o_proj INT8 QMV, v2 kernel (default
    /// path, see `qmvGatedV2Src` header comment for the design). Same
    /// contract as `qmvGatedV1`, plus an additional decline: `bank.inDim`
    /// must be a multiple of 256 (the kernel's `block_size`), since the
    /// per-lane 8-wide stride assumes full blocks and does not carry a
    /// partial-block tail path the way `qmv`/`qmvGatedV1`'s 32-wide
    /// per-pack loop does.
    static func qmvGatedV2(
        input: Tensor, gateLogits: Tensor, bank: Bank,
        heads: Int, headDim: Int,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard gatedShapesValid(
            input: input, gateLogits: gateLogits, bank: bank, heads: heads, headDim: headDim,
            out: out),
            bank.inDim % 256 == 0
        else { return false }

        let p = pso(
            "laguna_attn_qmv_int8_g32_gated_v2_bf16", source: qmvGatedV2Src,
            device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
        enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
        enc.setBuffer(bank.biases.buffer, offset: bank.biases.offset, index: 2)
        enc.setBuffer(input.buffer, offset: input.offset, index: 3)
        enc.setBuffer(gateLogits.buffer, offset: gateLogits.offset, index: 4)
        enc.setBuffer(out.buffer, offset: out.offset, index: 5)
        var inDimV = UInt32(bank.inDim)
        var outDimV = UInt32(bank.outDim)
        var headDimV = UInt32(headDim)
        var headsV = UInt32(heads)
        enc.setBytes(&inDimV, length: 4, index: 6)
        enc.setBytes(&outDimV, length: 4, index: 7)
        enc.setBytes(&headDimV, length: 4, index: 8)
        enc.setBytes(&headsV, length: 4, index: 9)
        enc.setThreadgroupMemoryLength(heads * MemoryLayout<Float>.stride, index: 0)
        let rowsPerTG = 8
        let tiles = (bank.outDim + rowsPerTG - 1) / rowsPerTG
        enc.dispatchThreadgroups(
            MTLSize(width: tiles, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

    // Split-K variant of the gated o_proj QMV: same `NR0=2`/`NSG=4`
    // two-stage-reduce restructure as `qmvSplitKSrc` below, with the
    // same numerically-stable softplus form `qmvGatedSrc` (v1) uses,
    // recomputed per lane per block, folded in before the affine dot so
    // the gated activation feeds both rows in the pair from one register
    // load. `packsPerHead` has the same meaning as in `qmvGatedSrc`: each
    // lane's own `packIdx` determines which per-head gate logit its
    // 4-wide `x` slice belongs to.
    //
    // ROUNDING FIX (TWO points, both mirroring `qmvGatedV2`): the gated
    // activation this kernel feeds into the affine dot stands in for what
    // a SEPARATE `LagunaGatedOProjFuse.applyGateProductSoftplus` dispatch
    // would materialize into a real bf16 tensor before a downstream plain
    // QMV reads it, so BOTH bf16 quantization steps that separate dispatch
    // performs have to happen here, or the kernel computes a DIFFERENT
    // quantity (a real value substitution), not just a differently-ordered
    // sum of the same one:
    //   1. The softplus GATE VALUE is rounded to bf16 before the multiply,
    //      matching v2's threadgroup gate_table
    //      (`gate_table[h] = float(bfloat((pos ? x : 0.0f) + lg))`), which
    //      stores softplus bf16-rounded and reads it back for the product.
    //   2. The gate*activation PRODUCT is rounded to bf16 right after the
    //      multiply, matching v2's `float(bfloat(float(xp[i]) * gate))`.
    // The o_proj Double CPU reference (`cpuReferenceGated`) double-rounds
    // at exactly these two points (`bf16Round(stableSoftplus(...))` then
    // `bf16Round(x[j] * gate[head])`). An earlier version carried BOTH in
    // float32 (copying v1's pattern); adding only point 2 still left point
    // 1 in float32 and the kernel stuck at maxAbsDiff 0.5 on real hardware
    // (a systematic per-row error that scales with the row's magnitude,
    // not the near-ULP gap a pure reduction restructure gives). Only
    // rounding the gate value too (point 1) collapses that quantity
    // mismatch. NB: this was misdiagnosed once by an offline replay that
    // reported point 2 alone reaching 0.0 -- the real Metal run is the
    // authority here, and it needed both points.
    private static let qmvGatedSplitKSrc = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_attn_qmv_int8_g32_gated_splitk_bf16(
            const device uint *w [[buffer(0)]],
            const device bfloat *scales [[buffer(1)]],
            const device bfloat *biases [[buffer(2)]],
            const device bfloat *x [[buffer(3)]],
            const device bfloat *gate_logits [[buffer(4)]],
            device bfloat *out [[buffer(5)]],
            constant uint &inDim [[buffer(6)]],
            constant uint &outDim [[buffer(7)]],
            constant uint &packsPerHead [[buffer(8)]],
            threadgroup float *shmem [[threadgroup(0)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sgitg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 32;
            constexpr uint NR0 = 2;
            constexpr uint NSG = 4;
            uint packsPerRow = inDim / 4;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * NR0;

            float acc[NR0] = {0.0f, 0.0f};

            for (uint block = sgitg * 32u; block < packsPerRow; block += NSG * 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 4u;
                    bfloat4 xv = *((const device bfloat4 *)(x + xBase));
                    uint head = packIdx / packsPerHead;
                    float gl = float(gate_logits[head]);
                    bool pos = gl > 0.0f;
                    float ax = pos ? gl : (0.0f - gl);
                    // ROUNDING POINT 1 (softplus gate value -> bf16):
                    // qmvGatedV2 stores the softplus result bf16-rounded in
                    // its threadgroup gate_table
                    // (`gate_table[h] = float(bfloat((pos ? x : 0.0f) + lg))`)
                    // and reads that bf16 value back for the product, so the
                    // gate this kernel multiplies by MUST be the same
                    // bf16-rounded scalar, not the full-float32 softplus. The
                    // separate gate-product-softplus dispatch this fused path
                    // stands in for materializes softplus into a real bf16
                    // tensor before the product too, and the o_proj CPU
                    // reference (`cpuReferenceGated`) double-rounds
                    // identically (`bf16Round(stableSoftplus(...))`). Carrying
                    // the gate in float32 here is a value substitution, not a
                    // reduction-order change, which is what left this kernel
                    // at maxAbsDiff 0.5 even after ROUNDING POINT 2 below was
                    // already in place.
                    float gate = float(bfloat((pos ? gl : 0.0f) + log(1.0f + exp(0.0f - ax))));
                    // ROUNDING POINT 2 (gate*activation product -> bf16):
                    // same point qmvGatedV2 rounds at
                    // (`float(bfloat(float(xp[i]) * gate))`): this gated
                    // activation stands in for what a separate gate-product
                    // dispatch would materialize as a real bf16 tensor before
                    // a downstream plain QMV reads it, so a plain float32
                    // carry here is a math change, not just a reduction-order
                    // change (see the header comment above this kernel's
                    // declaration for the failure this produced before the
                    // fix).
                    float x0 = float(bfloat(float(xv[0]) * gate));
                    float x1 = float(bfloat(float(xv[1]) * gate));
                    float x2 = float(bfloat(float(xv[2]) * gate));
                    float x3 = float(bfloat(float(xv[3]) * gate));
                    float xs = x0 + x1 + x2 + x3;
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < NR0; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float s = float(scales[row * groupsPerRow + g]);
                            float b = float(biases[row * groupsPerRow + g]);
                            float c0 = float(p & 0xFFu);
                            float c1 = float((p >> 8u) & 0xFFu);
                            float c2 = float((p >> 16u) & 0xFFu);
                            float c3 = float((p >> 24u) & 0xFFu);
                            float qd = c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3;
                            acc[r] += s * qd + b * xs;
                        }
                    }
                }
            }

            for (uint r = 0; r < NR0; ++r) {
                float partial = simd_sum(acc[r]);
                if (lane == 0) {
                    shmem[32u * r + sgitg] = partial;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                for (uint r = 0; r < NR0; ++r) {
                    float v = lane < NSG ? shmem[32u * r + lane] : 0.0f;
                    float total = simd_sum(v);
                    if (lane == 0) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            out[row] = bfloat(total);
                        }
                    }
                }
            }
        }
        """

    /// Split-K variant of `qmvGatedV1`/`qmvGatedV2` (see `splitKEnabled`
    /// and `qmvGatedSplitKSrc` above for the design). Same shape/dtype
    /// contract as `qmvGatedV1` (`gatedShapesValid`), plus `bank.outDim`
    /// must be a multiple of 2 (fixed `NR0 = 2` tiling, no partial-pair
    /// tail).
    static func qmvGatedSplitK(
        input: Tensor, gateLogits: Tensor, bank: Bank,
        heads: Int, headDim: Int,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard gatedShapesValid(
            input: input, gateLogits: gateLogits, bank: bank, heads: heads, headDim: headDim,
            out: out),
            bank.outDim.isMultiple(of: 2)
        else { return false }

        let packsPerHead = headDim / 4
        let p = pso(
            "laguna_attn_qmv_int8_g32_gated_splitk_bf16", source: qmvGatedSplitKSrc,
            device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
        enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
        enc.setBuffer(bank.biases.buffer, offset: bank.biases.offset, index: 2)
        enc.setBuffer(input.buffer, offset: input.offset, index: 3)
        enc.setBuffer(gateLogits.buffer, offset: gateLogits.offset, index: 4)
        enc.setBuffer(out.buffer, offset: out.offset, index: 5)
        var inDimV = UInt32(bank.inDim)
        var outDimV = UInt32(bank.outDim)
        var packsPerHeadV = UInt32(packsPerHead)
        enc.setBytes(&inDimV, length: 4, index: 6)
        enc.setBytes(&outDimV, length: 4, index: 7)
        enc.setBytes(&packsPerHeadV, length: 4, index: 8)
        enc.setThreadgroupMemoryLength(2 * 32 * MemoryLayout<Float>.stride, index: 0)
        let tiles = bank.outDim / 2
        enc.dispatchThreadgroups(
            MTLSize(width: tiles, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

    /// Fused gate-product-softplus + o_proj INT8 QMV in one dispatch.
    /// `gateLogits` is the RAW per-head gate logit (pre-softplus), same
    /// input `LagunaGatedOProjFuse.applyGateProductSoftplus` takes.
    /// Declines (returns `false`, `into` left untouched) unless
    /// `gatedEnabled` and every shape/dtype precondition holds; caller
    /// falls back to the separate-dispatch (T24 kernel 1) chain feeding
    /// stock `qmv`. Tries the split-K kernel first when `splitKEnabled`
    /// (falling through on decline), then dispatches `qmvGatedV2` by
    /// default (`BUTTER_LAGUNA_QMV_V2=0` selects `qmvGatedV1` instead);
    /// if v2 is selected but declines on shape (the `inDim % 256` guard),
    /// falls through to v1 rather than surfacing the outer two-dispatch
    /// chain. v1 has no such restriction.
    static func qmvGated(
        input: Tensor, gateLogits: Tensor, bank: Bank,
        heads: Int, headDim: Int,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard gatedEnabled else { return false }
        if splitKEnabled,
            qmvGatedSplitK(
                input: input, gateLogits: gateLogits, bank: bank, heads: heads, headDim: headDim,
                on: cmd, device: device, into: out)
        {
            return true
        }
        if qmvV2Enabled {
            if qmvGatedV2(
                input: input, gateLogits: gateLogits, bank: bank, heads: heads, headDim: headDim,
                on: cmd, device: device, into: out)
            {
                return true
            }
        }
        return qmvGatedV1(
            input: input, gateLogits: gateLogits, bank: bank, heads: heads, headDim: headDim,
            on: cmd, device: device, into: out)
    }

    // Split-K variant of `qmvSrc` (see `splitKEnabled` above for the
    // design). Each threadgroup is `NSG = 4` simdgroups (128 threads)
    // covering `NR0 = 2` output rows; every simdgroup strides a disjoint
    // slice of the K dimension for that SAME row pair (`block` starts at
    // `sgitg * 32` and advances by `NSG * 32` each iteration, so the 4
    // simdgroups' 32-lane bursts interleave to cover the full row
    // coalesced), instead of `qmvSrc` where 2 simdgroups each own 4 rows
    // and walk the whole K dimension alone. The activation 4-wide load
    // (`x0..x3`, `xs`) is read once per K-chunk into registers and
    // reused across both rows in `acc[NR0]`, and the group-32
    // scale/bias affine term (`s * qd + b * xs`) is applied once per
    // 4-wide raw dot, same as `qmvSrc` — this restructures the
    // reduction, not the math. After the K loop, each simdgroup's
    // partial (`simd_sum(acc[r])`) is written to `shmem[32*r + sgitg]`
    // by lane 0 (stage 1); after a barrier, simdgroup 0 re-`simd_sum`s
    // the `NSG` partials per row (zero-padded past `NSG`) and lane 0
    // writes `out[row]` (stage 2). `shmem` is sized `[NSG, NR0]`-worth
    // via the `32*r` stride so per-row slots don't collide even though
    // only 4 of each 32-wide span are used.
    private static let qmvSplitKSrc = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_attn_qmv_int8_g32_splitk_bf16(
            const device uint *w [[buffer(0)]],
            const device bfloat *scales [[buffer(1)]],
            const device bfloat *biases [[buffer(2)]],
            const device bfloat *x [[buffer(3)]],
            device bfloat *out [[buffer(4)]],
            constant uint &inDim [[buffer(5)]],
            constant uint &outDim [[buffer(6)]],
            threadgroup float *shmem [[threadgroup(0)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sgitg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 32;
            constexpr uint NR0 = 2;
            constexpr uint NSG = 4;
            uint packsPerRow = inDim / 4;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * NR0;

            float acc[NR0] = {0.0f, 0.0f};

            for (uint block = sgitg * 32u; block < packsPerRow; block += NSG * 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 4u;
                    bfloat4 xv = *((const device bfloat4 *)(x + xBase));
                    float x0 = float(xv[0]);
                    float x1 = float(xv[1]);
                    float x2 = float(xv[2]);
                    float x3 = float(xv[3]);
                    float xs = x0 + x1 + x2 + x3;
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < NR0; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float s = float(scales[row * groupsPerRow + g]);
                            float b = float(biases[row * groupsPerRow + g]);
                            float c0 = float(p & 0xFFu);
                            float c1 = float((p >> 8u) & 0xFFu);
                            float c2 = float((p >> 16u) & 0xFFu);
                            float c3 = float((p >> 24u) & 0xFFu);
                            float qd = c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3;
                            acc[r] += s * qd + b * xs;
                        }
                    }
                }
            }

            for (uint r = 0; r < NR0; ++r) {
                float partial = simd_sum(acc[r]);
                if (lane == 0) {
                    shmem[32u * r + sgitg] = partial;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sgitg == 0) {
                for (uint r = 0; r < NR0; ++r) {
                    float v = lane < NSG ? shmem[32u * r + lane] : 0.0f;
                    float total = simd_sum(v);
                    if (lane == 0) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            out[row] = bfloat(total);
                        }
                    }
                }
            }
        }
        """

    /// Split-K variant of `qmv` (see `splitKEnabled`/`qmvSplitKSrc` above
    /// for the design). Same shape/dtype contract as `qmv`, plus declines
    /// (returns `false`, `out` left untouched) unless `bank.outDim` is a
    /// multiple of 2 (fixed `NR0 = 2` tiling, no partial-pair tail).
    static func qmvSplitK(
        input: Tensor, bank: Bank, on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard input.dtype == .bf16, out.dtype == .bf16,
            input.elementCount == bank.inDim,
            out.elementCount == bank.outDim,
            bank.outDim.isMultiple(of: 2)
        else { return false }

        let p = pso(
            "laguna_attn_qmv_int8_g32_splitk_bf16", source: qmvSplitKSrc,
            device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
        enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
        enc.setBuffer(bank.biases.buffer, offset: bank.biases.offset, index: 2)
        enc.setBuffer(input.buffer, offset: input.offset, index: 3)
        enc.setBuffer(out.buffer, offset: out.offset, index: 4)
        var inDimV = UInt32(bank.inDim)
        var outDimV = UInt32(bank.outDim)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&outDimV, length: 4, index: 6)
        enc.setThreadgroupMemoryLength(2 * 32 * MemoryLayout<Float>.stride, index: 0)
        let tiles = bank.outDim / 2
        enc.dispatchThreadgroups(
            MTLSize(width: tiles, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

    /// Single-token (`x = [inDim]` bf16) batched QMV over `bank`,
    /// writing `[bank.outDim]` bf16 into `out`. Tries the split-K kernel
    /// first when `splitKEnabled` (falling through to the geometry below
    /// on any decline — shape mismatch or odd `bank.outDim`).
    static func qmv(
        input: Tensor, bank: Bank, on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) {
        precondition(input.dtype == .bf16, "LagunaAttnInt8.qmv: input must be bf16")
        precondition(
            input.elementCount == bank.inDim,
            "LagunaAttnInt8.qmv: input.elementCount \(input.elementCount) != bank.inDim \(bank.inDim)"
        )
        precondition(
            out.elementCount == bank.outDim,
            "LagunaAttnInt8.qmv: out.elementCount \(out.elementCount) != bank.outDim \(bank.outDim)"
        )
        if splitKEnabled,
            qmvSplitK(input: input, bank: bank, on: cmd, device: device, into: out)
        {
            return
        }
        let p = pso(
            "laguna_attn_qmv_int8_g32_bf16", source: qmvSrc, device: device.mtlDevice)
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(p)
            enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
            enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
            enc.setBuffer(bank.biases.buffer, offset: bank.biases.offset, index: 2)
            enc.setBuffer(input.buffer, offset: input.offset, index: 3)
            enc.setBuffer(out.buffer, offset: out.offset, index: 4)
            var inDimV = UInt32(bank.inDim)
            var outDimV = UInt32(bank.outDim)
            enc.setBytes(&inDimV, length: 4, index: 5)
            enc.setBytes(&outDimV, length: 4, index: 6)
            let rowsPerTG = 8
            let tiles = (bank.outDim + rowsPerTG - 1) / rowsPerTG
            enc.dispatchThreadgroups(
                MTLSize(width: tiles, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
        }
    }

// `qmvNormFolded` further folds the block's input RMSNorm into the QKV
// bank's own QMV (escape: `BUTTER_LAGUNA_QKV_NORMFOLD=0`); see its doc
// comment and `normQmvSrc` for the fold's exactness argument.
    /// Default ON; set `BUTTER_LAGUNA_QKV_NORMFOLD=0` to restore the
    /// separate `inputNorm` + `qmv` chain. Only takes effect where
    /// `enabled` (this bank) already holds.
    static var normFoldEnabled: Bool {
        ButterEnv["BUTTER_LAGUNA_QKV_NORMFOLD"] != "0"
    }

    // Norm-folded variant of `qmvSrc`: folds the block's input RMSNorm
    // into this bank's own QMV so the attention head reads one dispatch
    // instead of two. Takes the PRE-norm `residual` (the layer's `h`)
    // plus the norm weight instead of an already-normalized `x`.
    //
    // NO GRID-WIDE REDUCTION IS NEEDED. Every threadgroup already reads
    // the whole `inDim`-wide row for its own K loop below, so each one
    // re-derives the row's mean square privately: the redundancy is one
    // small re-read (4-8 KB) and one `inDim`-element reduction against
    // the multi-MB weight-bank traffic the K loop already does.
    //
    // EXACTNESS. `Ops.rmsNorm`'s fast path (`dispatchRmsNorm`, `n <= 4096`
    // and a multiple of 128) dispatches `iron_rms_norm_bf16` at
    // `TPG = inDim/4` real threads: each thread owns 4 elements, folds a
    // per-thread sum-of-squares, and the codegen's threadgroup-scope
    // `reduce_sum` (see `wh-iron-codegen/src/msl/reduce.rs`) combines
    // it in three phases: per-simdgroup `simd_sum`, lane 0 of each
    // simdgroup stores to a 32-slot threadgroup array, then simdgroup 0
    // re-`simd_sum`s those slots (zero-padded past `n_simd`). This bank's
    // threadgroup is fixed at 64 real threads (2 simdgroups) regardless of
    // `inDim`, so it CANNOT just run that dispatch geometry directly. It
    // emulates it instead: real thread `lid` (simdgroup `sg`, lane `lane`)
    // stands in for `virtualPerThread = (inDim/128) / 2` virtual
    // simdgroups `vsg = sg + 2*j`, each contributing exactly the same
    // per-thread sum-of-squares and `simd_sum` a real 64-lane-threadgroup
    // dispatch of `iron_rms_norm_bf16` would compute for that vsg, same
    // values, same lane order, so the phase-1/2/3 combine and the final
    // `rsqrt(ssq/inDim + eps)` land bit-identical. `metal::rsqrt` (not
    // `precise::rsqrt`; `iron_rms_norm_bf16` doesn't use the precise
    // variant either, see `unary_intrinsic` in
    // `wh-iron-codegen/src/backend.rs`).
    //
    // The K loop then recomputes each lane's normalized values inline
    // from `residual`/`norm_weight` (both L1-resident after the reduction
    // pass) instead of staging a normalized row to threadgroup memory:
    // measured a large loss for the staged variant on the sibling
    // norm+affine-QKV prototype (extra threadgroup memory caps resident
    // threadgroups per core); inline recompute costs one extra multiply
    // per lane per K-loop iteration and keeps threadgroup memory at the
    // ~132 bytes the reduction itself needs. Matches
    // `iron_rms_norm_bf16`'s own rounding: `bfloat(x * rms * w)`, ONE
    // round to bf16 (not two, unlike a naive
    // `w * bfloat(x * inv_mean)` staged reconstruction).
    private static let normQmvSrc = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void laguna_attn_norm_qmv_int8_g32_bf16(
            const device uint *w [[buffer(0)]],
            const device bfloat *scales [[buffer(1)]],
            const device bfloat *biases [[buffer(2)]],
            const device bfloat *residual [[buffer(3)]],
            const device bfloat *norm_weight [[buffer(4)]],
            device bfloat *out [[buffer(5)]],
            constant uint &inDim [[buffer(6)]],
            constant uint &outDim [[buffer(7)]],
            constant float &eps [[buffer(8)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_simdgroup]],
            uint sg [[simdgroup_index_in_threadgroup]])
        {
            constexpr uint GROUP = 32;
            constexpr uint ROWS_PER_SG = 4;
            constexpr uint REAL_SIMDGROUPS = 2;
            uint packsPerRow = inDim / 4;
            uint groupsPerRow = inDim / GROUP;
            uint rowBase = tgid.x * (2u * ROWS_PER_SG) + sg * ROWS_PER_SG;

            // --- rms reduction: 64 real lanes emulate the inDim/4 virtual
            // lanes `iron_rms_norm_bf16` would dispatch, `virtualPerThread`
            // virtual simdgroups at a time per real simdgroup. ---
            threadgroup float partial_sg[32];
            uint nSimdVirtual = inDim / 128u;
            uint virtualPerThread = nSimdVirtual / REAL_SIMDGROUPS;

            for (uint j = 0; j < virtualPerThread; ++j) {
                uint vsg = sg + REAL_SIMDGROUPS * j;
                uint base = (vsg * 32u + lane) * 4u;
                float x0 = float(residual[base]);
                float x1 = float(residual[base + 1u]);
                float x2 = float(residual[base + 2u]);
                float x3 = float(residual[base + 3u]);
                float partial = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
                float sv = simd_sum(partial);
                if (lane == 0) {
                    partial_sg[vsg] = sv;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float tg_ssq[1];
            if (sg == 0) {
                float wv = lane < nSimdVirtual ? partial_sg[lane] : 0.0f;
                float total = simd_sum(wv);
                if (lane == 0) {
                    tg_ssq[0] = total;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float rms = rsqrt(tg_ssq[0] / float(inDim) + eps);

            // --- affine_qmv over the inline-recomputed normalized row ---
            float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (uint block = 0; block < packsPerRow; block += 32u) {
                uint packIdx = block + lane;
                if (packIdx < packsPerRow) {
                    uint xBase = packIdx * 4u;
                    float x0 = float(bfloat(
                        float(residual[xBase]) * rms * float(norm_weight[xBase])));
                    float x1 = float(bfloat(
                        float(residual[xBase + 1u]) * rms * float(norm_weight[xBase + 1u])));
                    float x2 = float(bfloat(
                        float(residual[xBase + 2u]) * rms * float(norm_weight[xBase + 2u])));
                    float x3 = float(bfloat(
                        float(residual[xBase + 3u]) * rms * float(norm_weight[xBase + 3u])));
                    float xs = x0 + x1 + x2 + x3;
                    uint g = xBase / GROUP;
                    for (uint r = 0; r < ROWS_PER_SG; ++r) {
                        uint row = rowBase + r;
                        if (row < outDim) {
                            uint p = w[row * packsPerRow + packIdx];
                            float s = float(scales[row * groupsPerRow + g]);
                            float b = float(biases[row * groupsPerRow + g]);
                            float c0 = float(p & 0xFFu);
                            float c1 = float((p >> 8u) & 0xFFu);
                            float c2 = float((p >> 16u) & 0xFFu);
                            float c3 = float((p >> 24u) & 0xFFu);
                            float qd = c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3;
                            acc[r] += s * qd + b * xs;
                        }
                    }
                }
            }

            for (uint r = 0; r < ROWS_PER_SG; ++r) {
                float total = simd_sum(acc[r]);
                if (lane == 0) {
                    uint row = rowBase + r;
                    if (row < outDim) {
                        out[row] = bfloat(total);
                    }
                }
            }
        }
        """

    /// Norm-folded batched QKV(+gate) QMV: folds `Ops.rmsNorm`'s fast-path
    /// reduction into this bank's own dispatch (see `normQmvSrc` above for
    /// the exactness argument). `residual` is the PRE-norm block input
    /// (`h`), not an already-normalized row. Declines (returns `false`,
    /// `out` left untouched) unless `normFoldEnabled` and every
    /// shape/dtype/alignment precondition holds; caller then falls back to
    /// the standalone `inputNorm` + `qmv` chain.
    static func qmvNormFolded(
        residual: Tensor, normWeight: Tensor, eps: Float, bank: Bank,
        on cmd: MTLCommandBuffer, device: Device, into out: Tensor
    ) -> Bool {
        guard normFoldEnabled,
            residual.dtype == .bf16, normWeight.dtype == .bf16, out.dtype == .bf16,
            residual.elementCount == bank.inDim,
            normWeight.elementCount == bank.inDim,
            out.elementCount == bank.outDim,
            // Matches `dispatchRmsNorm`'s fast-path envelope (`n <= 4096`,
            // `n % 128 == 0`) plus this kernel's own alignment: 64 real
            // lanes (2 simdgroups) must divide the virtual simdgroup count
            // (`inDim / 128`) evenly, i.e. `inDim` a multiple of 256.
            bank.inDim <= 4096, bank.inDim.isMultiple(of: 256)
        else { return false }

        let p = pso(
            "laguna_attn_norm_qmv_int8_g32_bf16", source: normQmvSrc, device: device.mtlDevice)
        guard let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(p)
        enc.setBuffer(bank.packed.buffer, offset: bank.packed.offset, index: 0)
        enc.setBuffer(bank.scales.buffer, offset: bank.scales.offset, index: 1)
        enc.setBuffer(bank.biases.buffer, offset: bank.biases.offset, index: 2)
        enc.setBuffer(residual.buffer, offset: residual.offset, index: 3)
        enc.setBuffer(normWeight.buffer, offset: normWeight.offset, index: 4)
        enc.setBuffer(out.buffer, offset: out.offset, index: 5)
        var inDimV = UInt32(bank.inDim)
        var outDimV = UInt32(bank.outDim)
        var epsV = eps
        enc.setBytes(&inDimV, length: 4, index: 6)
        enc.setBytes(&outDimV, length: 4, index: 7)
        enc.setBytes(&epsV, length: 4, index: 8)
        let rowsPerTG = 8
        let tiles = (bank.outDim + rowsPerTG - 1) / rowsPerTG
        enc.dispatchThreadgroups(
            MTLSize(width: tiles, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

}
