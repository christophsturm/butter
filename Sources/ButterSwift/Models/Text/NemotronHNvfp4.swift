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
// NemotronH mixed-precision quantization — Nemotron-3.5-Lightning-
// 30B-A3B-NVFP4 port.
//
// The published checkpoint is NOT uniformly quantized. Per
// `hf_quant_config.json` (`modelopt` MIXED_PRECISION):
//
//   • MoE routed experts (`experts.N.{up,down}_proj`), the MoE shared
//     expert (`shared_experts.{up,down}_proj`), and `lm_head` — NVFP4
//     weight-only, group_size 16. Raw HF layout per tensor:
//       `<base>.weight`         u8  [out, in/2]   — 2 e2m1 codes/byte
//       `<base>.weight_scale`   f8_e4m3 [out, in/16] — per-block scale
//       `<base>.weight_scale_2` f32 []            — per-tensor global scale
//   • Mamba mixer `in_proj`/`out_proj` (all 23 mamba layers) — static
//     FP8 W8A8. Raw HF layout:
//       `<base>.weight`       f8_e4m3 [out, in]
//       `<base>.weight_scale` f32 []              — single scalar
//       `<base>.input_scale`  f32 [1]             — activation scale
//                                                    (unused — Butter
//                                                    keeps activations
//                                                    in bf16 end to end)
//   • Everything else (attention q/k/v/o proj, embeddings, mamba SSM
//     params + conv1d + gated-norm weight, MoE router gate +
//     correction bias, all RMSNorm weights) — plain bf16, untouched.
//     Attention `k_proj`/`v_proj` additionally carry `.k_scale`/
//     `.v_scale` (KV-cache FP8 scales for the reference TensorRT-LLM
//     serving stack) — Butter ignores these; its KV cache stays bf16.
//
// Butter's own NVFP4 kernels (`Nvfp4Linear`, `Ops.nvfp4DequantGemvExpertIndexedMany`,
// `Ops.nvfp4GatherSteel`) expect a DIFFERENT on-disk shape than HF ships:
//   `<base>.weight`  u32 [out, in/8]  — 8 e2m1 codes/lane
//   `<base>.scales`  u8  [out, in/16] — per-block E4M3 scale, with any
//                                        per-tensor global scale ALREADY
//                                        folded in (every existing
//                                        caller in this repo passes
//                                        `global: 1.0`; there is no
//                                        live path that reads a second,
//                                        per-tensor scale tensor).
// This file bridges the two: `repackNvfp4Weight` reinterprets the
// nibble-packed HF byte layout as Butter's u32-lane layout (pure
// bit-reinterpretation — HF's `torch.uint8 [out, in/2]` and Butter's
// `u32 [out, in/8]` cover the identical bytes in the identical
// little-endian order, four u8 nibble-pairs concatenated into one
// u32 lane), and `foldGlobalIntoScales` bakes each tensor's own
// `weight_scale_2` into its per-block `weight_scale` bytes via an
// E4M3 decode → multiply → re-encode round trip, so every downstream
// kernel call can pass `global: 1.0` unmodified.
//
// FP8 mamba projections have no Butter kernel to dispatch through at
// all (`Ops/OpsCoverageNotes.swift` — Butter's `DType` has no f8 case,
// and the one FP8 kernel that exists, `dsv4Fp8BlockDequant`, is
// unwired 128×128-block-scale scaffolding that doesn't fit this
// scalar-scale scheme). `dequantFp8ToDType` decodes the 46 mamba
// in_proj/out_proj tensors to bf16 on the CPU at load time instead —
// a one-time cost, and a small fraction of the checkpoint's total
// weight (the MoE experts dominate both parameter count and compute).

import Foundation
import Metal

// ─── E4M3 (float8_e4m3fn) scalar codec ────────────────────────────────
//
// 1 sign / 4 exponent (bias 7) / 3 mantissa, no infinities, NaN only at
// S.1111.111 (0x7F / 0xFF). Max finite magnitude 448 (exp=15, mant=6),
// min positive subnormal 2^-9 (exp=0, mant=1).

enum E4M3 {
    /// Decode one E4M3 byte to Float32. Built as a 256-entry LUT the
    /// first time it's needed — decode is called per-weight-element for
    /// the FP8 mamba dequant path (millions of calls), so paying the
    /// closed-form formula once per byte value (not per element) matters.
    static let decodeLUT: [Float] = (0 ..< 256).map { decodeSlow(UInt8($0)) }

    @inline(__always)
    static func decode(_ byte: UInt8) -> Float { decodeLUT[Int(byte)] }

    private static func decodeSlow(_ byte: UInt8) -> Float {
        let negative = (byte & 0x80) != 0
        let exp = Int((byte >> 3) & 0x0F)
        let mant = Int(byte & 0x07)
        var mag: Float
        if exp == 0 {
            // Subnormal: 2^-6 * (mant/8).
            mag = Float(mant) / 8.0 * exp2Float(-6)
        } else if exp == 15 && mant == 7 {
            return Float.nan
        } else {
            // Normal: 2^(exp-7) * (1 + mant/8).
            mag = (1.0 + Float(mant) / 8.0) * exp2Float(exp - 7)
        }
        return negative ? -mag : mag
    }

    @inline(__always)
    private static func exp2Float(_ e: Int) -> Float {
        Float(sign: .plus, exponent: e, significand: 1.0)
    }

    /// Encode a Float32 to the nearest representable E4M3 byte
    /// (round-half-away-from-zero on the 3 mantissa bits; ties don't
    /// materially matter here — this only re-quantizes values that are
    /// already E4M3-precision, folding in a per-tensor global scale).
    /// Overflow saturates to the max finite magnitude (448) rather than
    /// producing NaN, since a saturated micro-scale is a far smaller
    /// error than corrupting the whole block with NaN.
    static func encodeNearest(_ v: Float) -> UInt8 {
        if v.isNaN { return 0x7F }
        let signBit: UInt8 = v < 0 ? 0x80 : 0x00
        let av = Double(abs(v))
        if av == 0 { return signBit }
        let maxFinite = 448.0
        if av >= maxFinite {
            return signBit | 0x7E  // exp=15 (0b1111), mant=6 (0b110) = 448
        }
        var e = Int(floor(log2(av)))
        if e < -6 {
            // Subnormal range: value = 2^-6 * (m/8), m in 0...7.
            let m = (av / exp2Double(-6) * 8.0).rounded()
            let mInt = max(0, min(7, Int(m)))
            return signBit | UInt8(mInt)
        }
        if e > 8 { e = 8 }
        let sig = av / exp2Double(e)
        var mInt = Int(((sig - 1.0) * 8.0).rounded())
        if mInt >= 8 {
            mInt = 0
            e += 1
        }
        if e > 8 {
            // Overflowed while rounding up — saturate to max finite.
            return signBit | 0x7E
        }
        if e + 7 == 15 && mInt == 7 {
            // Would land exactly on the NaN pattern — clamp to max finite.
            mInt = 6
        }
        let expStored = UInt8(e + 7)
        return signBit | (expStored << 3) | UInt8(mInt & 0x07)
    }

    @inline(__always)
    private static func exp2Double(_ e: Int) -> Double { Foundation.exp2(Double(e)) }
}

// ─── Byte-level tensor helpers ────────────────────────────────────────

/// Reinterpret an HF NVFP4 packed weight — `u8 [out, in/2]` (2 e2m1
/// codes/byte, sequential input-channel order) — as Butter's expected
/// layout — `u32 [out, in/8]` (8 codes/lane). Every row's `in/2` bytes
/// are already contiguous and little-endian on Apple Silicon, so this
/// is a pure reinterpretation: no code needs to move, only the dtype
/// tag and the trailing-dimension size change. `outDim`/`inDim` are the
/// LOGICAL (unpacked) dimensions.
func repackNvfp4Weight(
    _ hfWeight: Tensor, outDim: Int, inDim: Int, device: Device
) -> Tensor {
    precondition(hfWeight.dtype == .u8, "repackNvfp4Weight: expected u8, got \(hfWeight.dtype)")
    precondition(
        hfWeight.elementCount == outDim * (inDim / 2),
        "repackNvfp4Weight: element count \(hfWeight.elementCount) != \(outDim)*\(inDim / 2)")
    precondition(inDim % 8 == 0, "repackNvfp4Weight: inDim \(inDim) must be a multiple of 8")
    let packed = inDim / 8
    let out = Tensor.empty(shape: [outDim, packed], dtype: .u32, device: device)
    // Byte-for-byte copy — same underlying bytes, different dtype tag.
    let byteCount = outDim * (inDim / 2)
    memcpy(
        out.buffer.contents().advanced(by: out.offset),
        hfWeight.buffer.contents().advanced(by: hfWeight.offset),
        byteCount)
    return out
}

/// Fold a per-tensor global scale (HF's `weight_scale_2`, f32 scalar)
/// into a per-block E4M3 micro-scale tensor (HF's `weight_scale`, read
/// as raw `.u8` bytes via the `F8_E4M3`→`.u8` DType mapping), producing
/// a new `.u8` tensor of the same shape whose values already include
/// the global factor — so every kernel call site can pass `global: 1.0`.
func foldNvfp4GlobalIntoScales(
    _ blockScaleU8: Tensor, global: Float, device: Device
) -> Tensor {
    precondition(
        blockScaleU8.dtype == .u8,
        "foldNvfp4GlobalIntoScales: expected u8 (raw E4M3 bytes), got \(blockScaleU8.dtype)")
    if global == 1.0 {
        // No-op fold — still materialize a fresh tensor so callers never
        // alias the source bundle's backing buffer.
        let out = Tensor.empty(shape: blockScaleU8.shape, dtype: .u8, device: device)
        memcpy(
            out.buffer.contents().advanced(by: out.offset),
            blockScaleU8.buffer.contents().advanced(by: blockScaleU8.offset),
            blockScaleU8.byteCount)
        return out
    }
    let n = blockScaleU8.elementCount
    let src = blockScaleU8.toArray(as: UInt8.self)
    var dst = [UInt8](repeating: 0, count: n)
    for i in 0 ..< n {
        let folded = E4M3.decode(src[i]) * global
        dst[i] = E4M3.encodeNearest(folded)
    }
    let out = Tensor.empty(shape: blockScaleU8.shape, dtype: .u8, device: device)
    out.copyIn(from: dst)
    return out
}

/// Dequantize a static-FP8 weight (`F8_E4M3` read as raw `.u8` bytes,
/// scaled by one per-tensor scalar) to a dense tensor in `dtype`
/// (bf16/f16/f32). Used for the 46 NemotronH-Lightning mamba mixer
/// `in_proj`/`out_proj` tensors — Butter has no FP8 GEMM/GEMV kernel
/// (see file header), so these get dequantized once at load time and
/// run through the ordinary bf16 `Linear` path.
func dequantFp8ToDType(
    _ f8AsU8: Tensor, scale: Float, outDim: Int, inDim: Int,
    dtype: DType, device: Device
) -> Tensor {
    precondition(f8AsU8.dtype == .u8, "dequantFp8ToDType: expected u8 (raw E4M3 bytes)")
    precondition(f8AsU8.elementCount == outDim * inDim, "dequantFp8ToDType: shape mismatch")
    let bytes = f8AsU8.toArray(as: UInt8.self)
    var floats = [Float](repeating: 0, count: bytes.count)
    for i in 0 ..< bytes.count {
        floats[i] = E4M3.decode(bytes[i]) * scale
    }
    let out = Tensor.empty(shape: [outDim, inDim], dtype: dtype, device: device)
    switch dtype {
    case .f32:
        out.copyIn(from: floats)
    case .f16:
        out.copyIn(from: floats.map { Float16($0) })
    case .bf16:
        out.copyIn(
            from: floats.map { v -> UInt16 in
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            })
    default:
        preconditionFailure("dequantFp8ToDType: unsupported output dtype \(dtype)")
    }
    return out
}

/// Write a host `[Float]` array into a fresh tensor of the requested
/// dtype — same round-to-nearest-even bf16 scheme as
/// `dequantFp8ToDType` above (and `NemotronHText.swift`'s private
/// `writeFloatsNH`, which this mirrors but can't call directly across
/// files). Used to build the GPU-resident router correction-bias and
/// routed-scaling-factor tensors the NVFP4 MoE layer's GPU router
/// needs (see `NemotronHNvfp4MoELayer.decode`).
private func writeFloatsAsDType(
    _ values: [Float], shape: [Int], dtype: DType, device: Device
) -> Tensor {
    let out = Tensor.empty(shape: shape, dtype: dtype, device: device)
    switch dtype {
    case .f32:
        out.copyIn(from: values)
    case .f16:
        out.copyIn(from: values.map { Float16($0) })
    case .bf16:
        out.copyIn(
            from: values.map { v -> UInt16 in
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            })
    default:
        preconditionFailure("writeFloatsAsDType: unsupported dtype \(dtype)")
    }
    return out
}

/// Load a projection weight that MAY be static-FP8 quantized (HF's
/// `<base>.weight` (f8_e4m3, read as raw `.u8`) + `<base>.weight_scale`
/// (f32 scalar) + `<base>.input_scale` (f32, unused — Butter keeps
/// activations bf16)), dequantizing to `dtype` at load time. Falls
/// back to a plain tensor read when the checkpoint doesn't quantize
/// this base (every other NemotronH checkpoint, and any future
/// Lightning weight that ships unquantized).
func loadMaybeFp8WeightNH(
    base: String, outDim: Int, inDim: Int,
    weights: SafeTensorsBundle, dtype: DType, device: Device
) throws -> Tensor {
    guard weights.has("\(base).weight_scale"), weights.has("\(base).input_scale"),
        !weights.has("\(base).weight_scale_2")  // NVFP4 uses this suffix too — disambiguate.
    else {
        return try weights.tensor(named: "\(base).weight")
    }
    let f8Bytes = try weights.tensor(named: "\(base).weight")
    let scaleT = try weights.tensor(named: "\(base).weight_scale")
    precondition(
        scaleT.elementCount == 1,
        "loadMaybeFp8WeightNH: \(base).weight_scale expected a scalar, got \(scaleT.shape)")
    let scale = scaleT.toFloatArray()[0]
    return dequantFp8ToDType(
        f8Bytes, scale: scale, outDim: outDim, inDim: inDim, dtype: dtype, device: device)
}

// ─── F-128 bandwidth wave: mamba in_proj/out_proj FP8 → int4 requant ─
//
// `loadMaybeFp8WeightNH` above dequantizes the 46 Lightning mamba
// mixer `in_proj`/`out_proj` tensors to dense `dtype` (bf16) at load
// time — 2 bytes/param, read fresh every decode token (Butter has no
// FP8 GEMV kernel, see that function's doc). That is the largest
// attributed dense per-token weight-traffic slice left in the decode
// path (F-127's fused-expert wave exhausted the MoE dispatch-fusion
// lever; these mamba projections were untouched).
//
// The reference TensorRT-LLM serving stack for this exact checkpoint
// is oracle-proven at 4-bit (Q4_K_M) for these same weights — Butter's
// own greedy-decode outputs already match that oracle word-for-word
// at bf16 precision, so re-quantizing this slice from bf16 down to
// 4-bit is *matching* the oracle's own precision, not degrading below
// a proven-good baseline. See `Tests/.../NemotronHNvfp4Tests.swift`
// for the roundtrip-tolerance unit coverage and the port's decode-gate
// notes for the end-to-end coherence check.
//
// Reuses Butter's own affine-int4 quantize kernels
// (`QuantizedOps.quantizeAffine` — the same primitive `butter convert`
// uses to build every other int4 checkpoint in this repo) rather than
// inventing a new kernel: `QuantizedLinear` already has a decode GEMV
// (`Ops.dequantGemv`, one dispatch) and a batched-prefill GEMM
// (`Ops.dequantGemmDynamicM`), so wiring these two tensors through it
// costs zero new Metal.
enum NemotronHMambaProjQ4 {
    /// Default ON. Set `BUTTER_NEMOTRON_MAMBA_PROJ_Q4=0` to fall back
    /// to the original dense-bf16 `Linear` path (F-125's initial port)
    /// for A/B comparison or quality-regression triage.
    static let enabled = ButterEnv["BUTTER_NEMOTRON_MAMBA_PROJ_Q4"] != "0"
}

/// Load one NemotronH mamba mixer projection (`in_proj` or `out_proj`),
/// auto-detecting the checkpoint's FP8 static quantization (identical
/// detection to `loadMaybeFp8WeightNH`) and, when found and
/// `NemotronHMambaProjQ4.enabled`, requantizing the dequantized dense
/// weight down to Butter's affine int4 `QuantizedLinear` format
/// (group_size 64 — see `requantizeDenseToInt4NH`'s doc for why 64 is
/// the only width available) instead of leaving it as a dense bf16
/// `Linear`.
///
/// Falls back to plain `AnyLinear(Linear(...))` when: the checkpoint
/// doesn't ship this base as FP8 (every non-Lightning NemotronH
/// checkpoint), the escape flag is off, or `inDim` isn't a multiple of
/// 64 (the group-alignment requirement — never true for the shipped
/// Lightning shapes, `hidden=2688`/`d_inner=4096`, both multiples of
/// 64, but guarded rather than assumed for a future checkpoint with
/// different dimensions).
func loadMambaProjNH(
    base: String, outDim: Int, inDim: Int,
    weights: SafeTensorsBundle, dtype: DType, device: Device
) throws -> AnyLinear {
    let wasFp8 =
        weights.has("\(base).weight_scale") && weights.has("\(base).input_scale")
        && !weights.has("\(base).weight_scale_2")
    let dense = try loadMaybeFp8WeightNH(
        base: base, outDim: outDim, inDim: inDim,
        weights: weights, dtype: dtype, device: device)
    guard wasFp8, NemotronHMambaProjQ4.enabled, inDim % 64 == 0 else {
        return AnyLinear(Linear(weight: dense))
    }
    return AnyLinear(
        requantizeDenseToInt4NH(dense, outDim: outDim, inDim: inDim, device: device))
}

/// Affine-int4-requantize a dense `[outDim, inDim]` weight into a
/// `QuantizedLinear` (group_size 64 — `QuantizedOps.quantizeAffine`'s
/// underlying `iron_affine_quantize_int4_*` kernels hardcode a
/// one-simdgroup × 2-elements/lane reduction shape, so 64 is the only
/// group size those kernels emit; see
/// `QuantizedOpsValidation.validateAffineQuantize`'s doc). One GPU
/// dispatch + a commit/wait — load-time only (one call per tensor per
/// checkpoint load), replacing the FP8→bf16 CPU dequant loop this
/// layers on top of at the same call site (`loadMambaProjNH`).
func requantizeDenseToInt4NH(
    _ dense: Tensor, outDim: Int, inDim: Int, device: Device
) -> QuantizedLinear {
    precondition(
        dense.shape == [outDim, inDim],
        "requantizeDenseToInt4NH: shape mismatch \(dense.shape) vs [\(outDim), \(inDim)]")
    let bits = 4
    let groupSize = 64
    precondition(
        inDim % groupSize == 0,
        "requantizeDenseToInt4NH: inDim \(inDim) must be a multiple of groupSize \(groupSize)")
    let packedCols = inDim * bits / 32
    let scaleCols = inDim / groupSize
    let packed = Tensor.empty(shape: [outDim, packedCols], dtype: .u32, device: device)
    let scales = Tensor.empty(shape: [outDim, scaleCols], dtype: dense.dtype, device: device)
    let biases = Tensor.empty(shape: [outDim, scaleCols], dtype: dense.dtype, device: device)
    let cmd = device.makeCommandBuffer()
    QuantizedOps.quantizeAffine(
        weight: dense, packed: packed, scales: scales, biases: biases,
        bits: bits, groupSize: groupSize, on: cmd)
    cmd.commit()
    cmd.waitUntilCompleted()
    return QuantizedLinear(
        weight: packed, scales: scales, biases: biases, bits: bits, groupSize: groupSize)
}

// ─── NVFP4 linear / stacked-expert loaders (HF raw layout) ───────────

/// Load one NVFP4-quantized dense projection (lm_head, or a single
/// non-batched expert like the MoE shared expert) from HF's raw
/// `<base>.weight` / `<base>.weight_scale` / `<base>.weight_scale_2`
/// triplet into a Butter `Nvfp4Linear`.
func loadNemotronHNvfp4Linear(
    base: String, outDim: Int, inDim: Int,
    weights: SafeTensorsBundle, device: Device
) throws -> Nvfp4Linear {
    let hfWeight = try weights.tensor(named: "\(base).weight")
    let blockScale = try weights.tensor(named: "\(base).weight_scale")
    let globalScaleT = try weights.tensor(named: "\(base).weight_scale_2")
    precondition(
        globalScaleT.elementCount == 1,
        "loadNemotronHNvfp4Linear: \(base).weight_scale_2 expected a scalar, "
            + "got shape \(globalScaleT.shape)")
    let global = globalScaleT.toFloatArray()[0]
    let repacked = repackNvfp4Weight(hfWeight, outDim: outDim, inDim: inDim, device: device)
    let folded = foldNvfp4GlobalIntoScales(blockScale, global: global, device: device)
    return Nvfp4Linear(weight: repacked, scales: folded, blockSize: 16, global: 1.0)
}

/// Stacked NVFP4 routed experts for a NemotronH MoE layer — the
/// up_proj/down_proj analogue of `MoELayer.StackedNvfp4Experts`, minus
/// the gate projection (NemotronH experts are `down(relu(up(x))^2)`,
/// no SwiGLU gate). Each expert's own `weight_scale_2` is folded into
/// its slice of the stacked per-block scale tensor at load time (see
/// file header) — `global` is therefore always `1.0` at the call site.
public struct NemotronHStackedNvfp4Experts: Sendable {
    public let upWeight: Tensor  // [E, moeIntermediate, hidden/8]      u32
    public let upScales: Tensor  // [E, moeIntermediate, hidden/16]     u8
    public let downWeight: Tensor  // [E, hidden, moeIntermediate/8]    u32
    public let downScales: Tensor  // [E, hidden, moeIntermediate/16]   u8
    public let numExperts: Int
    public let moeIntermediate: Int
    public let hidden: Int
    public let blockSize: Int
    public let dtype: DType
}

/// Load all `nExperts` routed experts for one NemotronH MoE layer from
/// HF's per-expert tensor layout into one stacked pair of buffers, so
/// `Ops.nvfp4DequantGemvExpertIndexedMany` can gather-index by expert
/// ID in a single dispatch. This is a load-time-only host loop (one
/// pass per checkpoint load, not per token) — copies + E4M3 fold each
/// expert's weight/scale slice into its offset in the stacked tensor.
func loadNemotronHStackedNvfp4Experts(
    prefix p: String, nExperts: Int, moeIntermediate: Int, hidden: Int,
    weights: SafeTensorsBundle, dtype: DType, device: Device
) throws -> NemotronHStackedNvfp4Experts {
    let blockSize = 16
    let upPacked = hidden / 8
    let upScaleCols = hidden / blockSize
    let downPacked = moeIntermediate / 8
    let downScaleCols = moeIntermediate / blockSize

    let upWeight = Tensor.empty(
        shape: [nExperts, moeIntermediate, upPacked], dtype: .u32, device: device)
    let upScales = Tensor.empty(
        shape: [nExperts, moeIntermediate, upScaleCols], dtype: .u8, device: device)
    let downWeight = Tensor.empty(
        shape: [nExperts, hidden, downPacked], dtype: .u32, device: device)
    let downScales = Tensor.empty(
        shape: [nExperts, hidden, downScaleCols], dtype: .u8, device: device)

    let upWeightRowBytes = moeIntermediate * upPacked * DType.u32.byteSize
    let upScaleRowBytes = moeIntermediate * upScaleCols
    let downWeightRowBytes = hidden * downPacked * DType.u32.byteSize
    let downScaleRowBytes = hidden * downScaleCols

    for e in 0 ..< nExperts {
        let ep = "\(p).experts.\(e)"

        let upLinear = try loadNemotronHNvfp4Linear(
            base: "\(ep).up_proj", outDim: moeIntermediate, inDim: hidden,
            weights: weights, device: device)
        precondition(upLinear.weight.byteCount == upWeightRowBytes)
        memcpy(
            upWeight.buffer.contents().advanced(by: upWeight.offset + e * upWeightRowBytes),
            upLinear.weight.buffer.contents().advanced(by: upLinear.weight.offset),
            upWeightRowBytes)
        precondition(upLinear.scales.byteCount == upScaleRowBytes)
        memcpy(
            upScales.buffer.contents().advanced(by: upScales.offset + e * upScaleRowBytes),
            upLinear.scales.buffer.contents().advanced(by: upLinear.scales.offset),
            upScaleRowBytes)

        let downLinear = try loadNemotronHNvfp4Linear(
            base: "\(ep).down_proj", outDim: hidden, inDim: moeIntermediate,
            weights: weights, device: device)
        precondition(downLinear.weight.byteCount == downWeightRowBytes)
        memcpy(
            downWeight.buffer.contents().advanced(by: downWeight.offset + e * downWeightRowBytes),
            downLinear.weight.buffer.contents().advanced(by: downLinear.weight.offset),
            downWeightRowBytes)
        precondition(downLinear.scales.byteCount == downScaleRowBytes)
        memcpy(
            downScales.buffer.contents().advanced(by: downScales.offset + e * downScaleRowBytes),
            downLinear.scales.buffer.contents().advanced(by: downLinear.scales.offset),
            downScaleRowBytes)
    }

    return NemotronHStackedNvfp4Experts(
        upWeight: upWeight, upScales: upScales,
        downWeight: downWeight, downScales: downScales,
        numExperts: nExperts, moeIntermediate: moeIntermediate, hidden: hidden,
        blockSize: blockSize, dtype: dtype)
}

// ─── NemotronHNvfp4MoELayer — "E" with NVFP4 routed/shared experts ───
//
// Same routing math and residual/shared-expert composition as the
// plain `NemotronHMoELayer` (`NemotronHText.swift`) — only the expert
// FFN math is quantized. Deliberately does NOT reuse any of
// `MoELayer.swift`'s Laguna-tuned fused SwiGLU dispatch chains (those
// are shaped for a gate+up SwiGLU topology and hard-code Laguna's own
// `hidden`/`moeIntermediate`/`topK`); this layer calls the generic
// `Ops.nvfp4DequantGemvExpertIndexedMany` primitive directly, twice
// per token (once for up_proj across the topK selected experts, once
// for down_proj), which is correct for ANY per-expert weight shape —
// prioritizing correctness-first port fidelity over decode throughput.
public final class NemotronHNvfp4MoELayer: Module, DecoderLayer {
    let norm: RMSNorm
    let gate: AnyLinear
    let eScoreCorrectionBiasHost: [Float]
    let stacked: NemotronHStackedNvfp4Experts
    /// Shared expert — NVFP4, but a single (non-batched) expert, so a
    /// plain `Nvfp4Linear` GEMV pair suffices (no gather-by-index).
    let sharedUp: Nvfp4Linear?
    let sharedDown: Nvfp4Linear?

    public let hidden, nRouted, topK: Int
    public let routedScalingFactor: Float
    public let normTopkProb: Bool
    let dtype: DType

    // ─── GPU-resident routing (F-126 perf wave) ───────────────────────
    // Allocated once at load time — topK/nRouted/dtype are fixed for
    // the lifetime of a loaded model, so these are plain `let` scratch
    // buffers rather than the lazily-reallocated pattern `MoELayer`
    // uses for its more configuration-varied Laguna paths. See
    // `decode` for how they replace the CPU commit+wait router.
    //
    // The whole routing chain (logits, bias, biased-score comparison,
    // unbiased-weight sum) runs in **f32**, not the model's bf16
    // activation dtype — measured, not assumed: this checkpoint's
    // `e_score_correction_bias` sits on a ~40.7 common-mode offset
    // with per-expert deltas of ~0.01 riding on top (an auxiliary-
    // load-balancing bias that drifted well off zero during
    // training). bf16 has ~8 mantissa bits (~0.4% relative precision)
    // — at a base of ~40.7 that's an LSB spacing around 0.16-0.3,
    // completely swallowing the ~0.01 deltas that actually
    // discriminate between experts, so the bf16-typed fused router
    // kernel (which itself rounds the biased score to `bfloat` before
    // comparing — see its module doc) was picking essentially
    // *arbitrary* experts. mlx-lm and this file's own CPU path both
    // keep the correction bias in f32 for exactly this reason (see
    // `loadNemotronHStackedNvfp4Experts`'s prefix doc / the "kept in
    // fp32" comment on `NemotronHMoELayer.eScoreCorrectionBias`) — the
    // GPU path has to match that precision, not just the activation
    // dtype, to select the SAME experts. The f32 `iron_moe_router
    // _topk_biased_rows_fused_f32` kernel variant has no such
    // intermediate rounding.
    let eScoreCorrectionBiasGPU: Tensor  // [nRouted] f32.
    let routerLogitsF32Scratch: Tensor  // [nRouted] f32 — upcast of the bf16 router GEMV output.
    let routerIndicesScratch: Tensor  // [topK] u32 — GPU top-K output.
    let routerWeightsF32Scratch: Tensor  // [topK] f32 — GPU top-K output (pre-scaling).
    let routerWeightsScratch: Tensor  // [topK], dtype-matched — final scaled weights, downcast for the combine.
    let routedScaleVecF32: Tensor?  // [topK] f32, filled with routedScalingFactor; nil when 1.0.

    /// GPU-resident sigmoid+bias top-K routing, default ON. Set
    /// `BUTTER_NEMOTRON_NVFP4_GPU_ROUTER=0` to fall back to the
    /// original per-layer CPU commit+wait router (46 waits/token on
    /// the published Lightning 23-MoE-layer stack) for A/B comparison
    /// or regression triage. `decode`'s branch and `commitsMidDecode`
    /// below must stay in lockstep with this flag — the outer decode
    /// loop (`NemotronHModel.forward`) uses `commitsMidDecode` to
    /// decide whether it's safe to batch this layer onto the same
    /// command buffer as its neighbours.
    static let gpuRouterEnabled = ButterEnv["BUTTER_NEMOTRON_NVFP4_GPU_ROUTER"] != "0"

    /// Diagnostic only — `BUTTER_NEMOTRON_DEBUG_ROUTING=1` prints each
    /// layer's chosen expert-id set (GPU or CPU path, whichever is
    /// active) to stderr in call order, for direct A/B diffing between
    /// a GPU-routed run and a `BUTTON_NEMOTRON_NVFP4_GPU_ROUTER=0`
    /// CPU-routed run of the identical prompt. Forces an extra
    /// mid-decode commit+wait when on the GPU path (see
    /// `decodeGPURouted`) — never enable for a perf measurement.
    static let debugRoutingEnabled = ButterEnv["BUTTER_NEMOTRON_DEBUG_ROUTING"] == "1"
    nonisolated(unsafe) static var debugCallCounter = 0

    /// Whether this instance actually takes the GPU-router branch.
    /// `iron_moe_router_topk_biased_rows_fused` always renormalizes the
    /// chosen weights to sum-to-1 (see its kernel doc) — the exact
    /// behaviour of the CPU path's `if topK > 1 && normTopkProb`
    /// branch. Every shipped Lightning checkpoint sets `normTopkProb`
    /// and `topK == 6`, but guard both explicitly rather than assume:
    /// a future checkpoint with `normTopkProb == false` or `topK == 1`
    /// falls back to the CPU router instead of silently renormalizing
    /// weights the CPU reference wouldn't.
    var usesGpuRouter: Bool { Self.gpuRouterEnabled && normTopkProb && topK > 1 }

    /// Whether this instance takes the fused routed-expert dispatch
    /// path (`NemotronLightningNvfp4Fuse.moeRoutedRelu2Qmv` +
    /// `moeRoutedDownReduce`) instead of the per-slot GEMV/relu/square/
    /// GEMV/headGateMul/add loop. The two fused kernels hardcode
    /// Lightning's shapes (hidden=2688, moeIntermediate=1856, topK=6,
    /// bf16) — guard every dimension explicitly so a future NemotronH
    /// NVFP4 checkpoint with different shapes safely falls back to the
    /// generic per-slot loop instead of reading out of bounds.
    var usesFusedExperts: Bool {
        NemotronLightningNvfp4Fuse.fusedExpertsEnabled
            && dtype == .bf16 && topK == 6 && hidden == 2688
            && stacked.moeIntermediate == 1856
    }

    /// `DecoderLayer.commitsMidDecode` — see the protocol doc. `true`
    /// only when the CPU-router fallback is active (it still needs a
    /// host readback of the router logits, exactly like the sibling
    /// `NemotronHMoELayer`); the default GPU-routed path never commits
    /// `cmd` internally.
    public var commitsMidDecode: Bool { !usesGpuRouter || Self.debugRoutingEnabled }

    init(
        norm: RMSNorm, gate: AnyLinear, eScoreCorrectionBias: [Float],
        stacked: NemotronHStackedNvfp4Experts,
        sharedUp: Nvfp4Linear?, sharedDown: Nvfp4Linear?,
        hidden: Int, topK: Int,
        routedScalingFactor: Float, normTopkProb: Bool, dtype: DType,
        device: Device
    ) {
        precondition(
            topK > 0 && topK <= stacked.numExperts,
            "NemotronHNvfp4MoELayer: topK \(topK) out of range 1…\(stacked.numExperts)")
        self.norm = norm
        self.gate = gate
        self.eScoreCorrectionBiasHost = eScoreCorrectionBias
        self.stacked = stacked
        self.sharedUp = sharedUp
        self.sharedDown = sharedDown
        self.hidden = hidden
        self.nRouted = stacked.numExperts
        self.topK = topK
        self.routedScalingFactor = routedScalingFactor
        self.normTopkProb = normTopkProb
        self.dtype = dtype
        self.eScoreCorrectionBiasGPU = writeFloatsAsDType(
            eScoreCorrectionBias, shape: [stacked.numExperts], dtype: .f32, device: device)
        self.routerLogitsF32Scratch = Tensor.empty(shape: [stacked.numExperts], dtype: .f32, device: device)
        self.routerIndicesScratch = Tensor.empty(shape: [topK], dtype: .u32, device: device)
        self.routerWeightsF32Scratch = Tensor.empty(shape: [topK], dtype: .f32, device: device)
        self.routerWeightsScratch = Tensor.empty(shape: [topK], dtype: dtype, device: device)
        self.routedScaleVecF32 =
            routedScalingFactor == 1.0
            ? nil
            : Tensor.filled(routedScalingFactor, shape: [topK], dtype: .f32, device: device)
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in norm.parameters() { out.append(("norm.\(k)", v)) }
        for (k, v) in gate.parameters() { out.append(("mixer.gate.\(k)", v)) }
        out.append(("mixer.experts.up_proj.weight", stacked.upWeight))
        out.append(("mixer.experts.up_proj.scales", stacked.upScales))
        out.append(("mixer.experts.down_proj.weight", stacked.downWeight))
        out.append(("mixer.experts.down_proj.scales", stacked.downScales))
        return out
    }

    /// `DecoderLayer` conformance. Dispatches to the GPU-routed fast
    /// path (default) or the CPU-routed fallback
    /// (`BUTTER_NEMOTRON_NVFP4_GPU_ROUTER=0`) — see `usesGpuRouter`.
    public func decode(
        _ h: Tensor, position _: Int,
        cache _: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        if usesGpuRouter {
            return decodeGPURouted(h, cmd: cmd, device: device)
        }
        return decodeCPURouted(h, cmd: cmd, device: device)
    }

    /// F-126 perf wave: sigmoid+bias top-K routing stays entirely on
    /// the GPU — no host readback, no mid-decode commit. Everything
    /// queues onto the caller's `cmd`; the caller (the outer
    /// `NemotronHModel.forward` loop, or a unit test) commits once at
    /// the end of the token. Mirrors the Mamba/Attention/MLP layers'
    /// contract exactly (see `DecoderLayer.decode`'s doc: "does not
    /// commit").
    ///
    /// Routing chain:
    ///   1. `Ops.moeRouterTopKBiasedRowsFused` — one dispatch computes
    ///      `sigmoid(logit) + bias` per expert, top-K by the biased
    ///      score, and writes `topK` indices/weights (weights =
    ///      unbiased sigmoid, renormalized to sum-to-1 — same
    ///      `normTopkProb` semantics as the CPU path's `finalScores /
    ///      sum` branch; see `usesGpuRouter`'s doc for why that guard
    ///      exists).
    ///   2. Multiply the weights buffer in place by
    ///      `routedScalingFactor` (`Ops.mul`, GPU-resident constant
    ///      vector) — replaces the CPU path's
    ///      `finalScores.map { $0 * routedScalingFactor }`.
    ///   3. Per-slot expert-id tensors are zero-copy VIEWS into
    ///      `routerIndicesScratch` (different byte offset, same
    ///      buffer) — `Ops.nvfp4DequantGemvExpertIndexedMany` reads
    ///      the expert id device-side per dispatch, so no per-slot
    ///      host `memcpy` into a fresh 4-byte buffer is needed (the
    ///      CPU path's `idxTensors` loop).
    ///   4. The weighted combine reads each slot's scalar weight the
    ///      same way — a `[1]`-shaped view into `routerWeightsScratch`
    ///      — via `Ops.headGateMul` (broadcast multiply: `head_dim ==
    ///      hidden` makes every element of the `[hidden]` down output
    ///      read the same single gate scalar), instead of the CPU
    ///      path's `Tensor.filled(hostFloat, shape: [hidden], ...)`.
    private func decodeGPURouted(
        _ h: Tensor, cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let xNorm = norm(h, on: cmd)
        let logitsTensor = gate(xNorm, on: cmd)

        // Routing math runs in f32 throughout (logits upcast, bias
        // already f32, biased-score compare + unbiased-weight sum both
        // in the f32 kernel variant) — see the file-scope doc on
        // `eScoreCorrectionBiasGPU` for why bf16 silently breaks
        // selection on this checkpoint's correction-bias distribution.
        Ops.castToF32(logitsTensor, into: routerLogitsF32Scratch, on: cmd)
        Ops.moeRouterTopKBiasedRowsFused(
            logits: routerLogitsF32Scratch, bias: eScoreCorrectionBiasGPU,
            indicesOut: routerIndicesScratch, weightsOut: routerWeightsF32Scratch,
            t: 1, nExperts: nRouted, k: topK, on: cmd)
        if let scaleVec = routedScaleVecF32 {
            _ = Ops.mul(
                routerWeightsF32Scratch, scaleVec, on: cmd, into: routerWeightsF32Scratch)
        }
        // Final combine weights only need to match `dtype` (the down-
        // projection output's precision) — downcast once here, after
        // all the precision-sensitive routing math is done.
        if dtype == .f32 {
            Ops.copy(routerWeightsF32Scratch, into: routerWeightsScratch, on: cmd)
        } else if dtype == .bf16 {
            Ops.castF32ToBf16(routerWeightsF32Scratch, into: routerWeightsScratch, on: cmd)
        } else {
            fatalError(
                "NemotronHNvfp4MoELayer.decodeGPURouted: unsupported activation dtype \(dtype) "
                    + "(only f32/bf16 have a routing-weights downcast wired)")
        }

        // DEBUG ONLY (`BUTTER_NEMOTRON_DEBUG_ROUTING=1`): force a sync
        // to print the chosen expert-id set for cross-checking against
        // `decodeCPURouted`'s CPU router, call-for-call. This is the
        // ONE place the GPU path ever commits `cmd` — doing so forces
        // `commitsMidDecode` to report `true` while the flag is set
        // (see below), so the outer forward loop hands every following
        // layer a fresh buffer, exactly like the CPU-routed layer kind.
        // `workCmd` is `cmd` unchanged when the flag is off (the normal,
        // zero-commit fast path).
        var workCmd = cmd
        if Self.debugRoutingEnabled {
            workCmd.commit()
            workCmd.waitUntilCompleted()
            let n = Self.debugCallCounter
            Self.debugCallCounter += 1
            let ids = routerIndicesScratch.toArray(as: UInt32.self).sorted()
            FileHandle.standardError.write("[route-gpu] call=\(n) ids=\(ids)\n".data(using: .utf8)!)
            if n == 0 {
                let l8 = logitsTensor.toFloatArray().prefix(8)
                let b8 = eScoreCorrectionBiasGPU.toFloatArray().prefix(8)
                let w6 = routerWeightsScratch.toFloatArray()
                FileHandle.standardError.write(
                    "[route-gpu] call=0 logits[0..8]=\(Array(l8)) bias[0..8]=\(Array(b8)) weights=\(w6)\n"
                        .data(using: .utf8)!)
            }
            workCmd = device.makeCommandBuffer()
        }

        let moeIntermediate = stacked.moeIntermediate
        var accumulator: Tensor?

        if usesFusedExperts {
            // ── F-127 perf wave: fused routed-expert dispatch pair ────
            // Replaces the 6x up-GEMV + 6x relu + 6x square + 6x
            // down-GEMV + 6x headGateMul + 5x accumulate-add per-slot
            // loop (35 dispatches) with 2 — see
            // NemotronLightningNvfp4Fuse.swift. `routerIndicesScratch`/
            // `routerWeightsScratch` are consumed device-side as whole
            // [topK] buffers (no per-slot view tensors needed).
            var upOuts: [Tensor] = []
            upOuts.reserveCapacity(topK)
            for _ in 0 ..< topK {
                upOuts.append(Tensor.empty(shape: [moeIntermediate], dtype: .bf16, device: device))
            }
            NemotronLightningNvfp4Fuse.moeRoutedRelu2Qmv(
                input: xNorm, upWeight: stacked.upWeight, upScales: stacked.upScales,
                expertIndices: routerIndicesScratch, outputs: upOuts,
                global: 1.0, on: workCmd, device: device)

            let routedCombine = Tensor.empty(shape: [hidden], dtype: .bf16, device: device)
            NemotronLightningNvfp4Fuse.moeRoutedDownReduce(
                mids: upOuts, expertIndices: routerIndicesScratch,
                slotWeights: routerWeightsScratch,
                downWeight: stacked.downWeight, downScales: stacked.downScales,
                output: routedCombine, global: 1.0, on: workCmd, device: device)
            accumulator = routedCombine
        } else {
            // ── Escape path (BUTTER_NEMOTRON_NVFP4_FUSED_EXPERTS=0, or
            //    a checkpoint shape the fused kernels don't cover) —
            //    original per-slot loop, byte-for-byte. ────────────────
            var idxTensors: [Tensor] = []
            idxTensors.reserveCapacity(topK)
            for slot in 0 ..< topK {
                idxTensors.append(
                    Tensor(
                        buffer: routerIndicesScratch.buffer,
                        offset: routerIndicesScratch.offset + slot * DType.u32.byteSize,
                        shape: [1], dtype: .u32))
            }

            // ── Routed experts — two batched gather-GEMV dispatches ──────
            let upIns = Array(repeating: xNorm, count: topK)
            var upOuts: [Tensor] = []
            for _ in 0 ..< topK {
                upOuts.append(Tensor.empty(shape: [moeIntermediate], dtype: dtype, device: device))
            }
            let upWeights = Array(repeating: stacked.upWeight, count: topK)
            let upScalesArr = Array(repeating: stacked.upScales, count: topK)
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: upWeights, scalesStacked: upScalesArr,
                inputs: upIns, expertIndices: idxTensors, outputs: upOuts,
                blockSize: stacked.blockSize, global: 1.0, on: workCmd)

            // relu(up)^2 per slot, in place into a fresh "inner" buffer.
            var innerOuts: [Tensor] = []
            for slot in 0 ..< topK {
                let r = Ops.relu(upOuts[slot], on: workCmd)
                innerOuts.append(Ops.mul(r, r, on: workCmd))
            }

            var downOuts: [Tensor] = []
            for _ in 0 ..< topK {
                downOuts.append(Tensor.empty(shape: [hidden], dtype: dtype, device: device))
            }
            let downWeights = Array(repeating: stacked.downWeight, count: topK)
            let downScalesArr = Array(repeating: stacked.downScales, count: topK)
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: downWeights, scalesStacked: downScalesArr,
                inputs: innerOuts, expertIndices: idxTensors, outputs: downOuts,
                blockSize: stacked.blockSize, global: 1.0, on: workCmd)

            // ── Weighted sum over the topK selected experts — per-slot
            //    scale reads straight from the GPU router-weights buffer,
            //    no host float involved. ─────────────────────────────────
            for slot in 0 ..< topK {
                let wSlot = Tensor(
                    buffer: routerWeightsScratch.buffer,
                    offset: routerWeightsScratch.offset + slot * dtype.byteSize,
                    shape: [1], dtype: dtype)
                let scaled = Ops.headGateMul(
                    attn: downOuts[slot], gate: wSlot, headDim: hidden, on: workCmd)
                accumulator =
                    accumulator.map { Ops.add($0, scaled, on: workCmd) } ?? scaled
            }
        }

        // ── Shared expert — always-on, plain NVFP4 GEMV pair ──────────
        if let su = sharedUp, let sd = sharedDown {
            let sUp = su(xNorm, on: workCmd)
            let sR = Ops.relu(sUp, on: workCmd)
            let sInner = Ops.mul(sR, sR, on: workCmd)
            let sharedOut = sd(sInner, on: workCmd)
            accumulator =
                accumulator.map { Ops.add($0, sharedOut, on: workCmd) } ?? sharedOut
        }

        let routedOut = accumulator!
        let result = Ops.add(h, routedOut, on: workCmd)
        if Self.debugRoutingEnabled {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }
        return result
    }

    /// Original CPU-routed path — commits `cmd` and returns a
    /// fully-resident tensor produced on a fresh internal buffer, same
    /// command-buffer contract as `NemotronHMoELayer.decode`. Kept as
    /// the `BUTTER_NEMOTRON_NVFP4_GPU_ROUTER=0` fallback (see
    /// `usesGpuRouter`) — untouched by the F-126 perf wave, so it
    /// remains a ground-truth reference for A/B token-match checks
    /// against `decodeGPURouted` above.
    private func decodeCPURouted(
        _ h: Tensor, cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // ── Pre-mixer RMSNorm + router GEMV on the caller's buffer ───
        let xNorm = norm(h, on: cmd)
        let logitsTensor = gate(xNorm, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── CPU routing — identical to `NemotronHMoELayer.decode` ────
        let logits = logitsTensor.toFloatArray()
        var origScores = [Float](repeating: 0, count: nRouted)
        var selectionScores = [Float](repeating: 0, count: nRouted)
        for i in 0 ..< nRouted {
            let s = 1.0 / (1.0 + Foundation.exp(-logits[i]))
            origScores[i] = s
            selectionScores[i] = s + eScoreCorrectionBiasHost[i]
        }
        let order = (0 ..< nRouted).sorted { a, b in
            if selectionScores[a] != selectionScores[b] {
                return selectionScores[a] > selectionScores[b]
            }
            return a < b
        }
        let inds = Array(order.prefix(topK))
        if Self.debugRoutingEnabled {
            let n = Self.debugCallCounter
            Self.debugCallCounter += 1
            FileHandle.standardError.write(
                "[route-cpu] call=\(n) ids=\(inds.sorted())\n".data(using: .utf8)!)
            if n == 0 {
                let msg =
                    "[route-cpu] call=0 logits[0..8]=\(Array(logits.prefix(8))) "
                    + "bias[0..8]=\(Array(eScoreCorrectionBiasHost.prefix(8)))\n"
                FileHandle.standardError.write(msg.data(using: .utf8)!)
            }
        }
        var finalScores = inds.map { origScores[$0] }
        if topK > 1 && normTopkProb {
            let sum = finalScores.reduce(0, +) + 1e-20
            finalScores = finalScores.map { $0 / sum }
        }
        finalScores = finalScores.map { $0 * routedScalingFactor }

        // ── Routed experts — two batched gather-GEMV dispatches ──────
        let workCmd = device.makeCommandBuffer()
        let moeIntermediate = stacked.moeIntermediate

        var idxTensors: [Tensor] = []
        idxTensors.reserveCapacity(topK)
        for ei in inds {
            var v = UInt32(ei)
            let buf = device.makeBuffer(length: 4)
            memcpy(buf.contents(), &v, 4)
            idxTensors.append(Tensor(buffer: buf, offset: 0, shape: [1], dtype: .u32))
        }
        let upIns = Array(repeating: xNorm, count: topK)
        var upOuts: [Tensor] = []
        for _ in 0 ..< topK {
            upOuts.append(Tensor.empty(shape: [moeIntermediate], dtype: dtype, device: device))
        }
        let upWeights = Array(repeating: stacked.upWeight, count: topK)
        let upScalesArr = Array(repeating: stacked.upScales, count: topK)
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: upWeights, scalesStacked: upScalesArr,
            inputs: upIns, expertIndices: idxTensors, outputs: upOuts,
            blockSize: stacked.blockSize, global: 1.0, on: workCmd)

        // relu(up)^2 per slot, in place into a fresh "inner" buffer.
        var innerOuts: [Tensor] = []
        for slot in 0 ..< topK {
            let r = Ops.relu(upOuts[slot], on: workCmd)
            innerOuts.append(Ops.mul(r, r, on: workCmd))
        }

        var downOuts: [Tensor] = []
        for _ in 0 ..< topK {
            downOuts.append(Tensor.empty(shape: [hidden], dtype: dtype, device: device))
        }
        let downWeights = Array(repeating: stacked.downWeight, count: topK)
        let downScalesArr = Array(repeating: stacked.downScales, count: topK)
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: downWeights, scalesStacked: downScalesArr,
            inputs: innerOuts, expertIndices: idxTensors, outputs: downOuts,
            blockSize: stacked.blockSize, global: 1.0, on: workCmd)

        // ── Weighted sum over the topK selected experts ───────────────
        var accumulator: Tensor?
        for slot in 0 ..< topK {
            let w = Tensor.filled(finalScores[slot], shape: [hidden], dtype: dtype, device: device)
            let scaled = Ops.mul(downOuts[slot], w, on: workCmd)
            accumulator =
                accumulator.map { Ops.add($0, scaled, on: workCmd) } ?? scaled
        }

        // ── Shared expert — always-on, plain NVFP4 GEMV pair ──────────
        if let su = sharedUp, let sd = sharedDown {
            let sUp = su(xNorm, on: workCmd)
            let sR = Ops.relu(sUp, on: workCmd)
            let sInner = Ops.mul(sR, sR, on: workCmd)
            let sharedOut = sd(sInner, on: workCmd)
            accumulator =
                accumulator.map { Ops.add($0, sharedOut, on: workCmd) } ?? sharedOut
        }

        let routedOut = accumulator!
        let result = Ops.add(h, routedOut, on: workCmd)
        workCmd.commit()
        workCmd.waitUntilCompleted()
        return result
    }

    /// F-132 batched prefill: T-batched NVFP4 MoE. `h` is `[T, hidden]`;
    /// returns `[T, hidden]`. Only takes the batched GPU-router path
    /// when `usesGpuRouter` (matching `decode`'s guard — the same
    /// checkpoint-shape / config conditions apply, T-independent);
    /// otherwise falls back to the per-row default `decodeMulti` loop
    /// (`DecoderLayer`'s protocol extension) reusing `decode`'s existing
    /// CPU-routed path unchanged.
    ///
    /// Reuses the batched-prefill MoE primitives the reusable
    /// `MoELayer.swift` gather-GEMM / sort-route pipeline already
    /// exercises for other model families — no new kernel families:
    ///
    ///   1. `Ops.moeRouterTopKBiasedRowsFused(t: T, ...)` — same fused
    ///      sigmoid+bias top-K router as `decodeGPURouted`, run once
    ///      over all T rows instead of T times.
    ///   2. `Ops.moeSortPlanFused` — GPU counting-sort of the `T·topK`
    ///      (token, slot) pairs by chosen expert id, so the gather-GEMM
    ///      below sees expert-contiguous row runs.
    ///   3. `Ops.gather` — expands the T-row `xNorm` into `T·topK`
    ///      sorted rows (one row per selected expert slot).
    ///   4. `Ops.nvfp4MoeGatherBm16` (×2 — up_proj, down_proj) — the
    ///      same gather-GEMM primitive `NemotronLightningNvfp4Fuse`'s
    ///      decode-time fused kernels wrap for T=1; here it runs
    ///      natively at T·topK rows in one dispatch per projection.
    ///   5. `Ops.moeUnpermuteTail` — un-sorts the down-projection output
    ///      back to token order, weighted-sums the `topK` slots, adds
    ///      the (batched, densely-computed) shared expert, and adds the
    ///      residual — all in one dispatch.
    ///
    /// All work queues onto `cmd`; no commit inside on the GPU-router
    /// branch (matches `commitsMidDecode == false` for that case).
    /// Set once, the first time `decodeMulti` falls back to the per-row
    /// loop because `Ops.mppAutoCapable` is false — avoids spamming
    /// stderr once per prefill chunk. See the guard in `decodeMulti`
    /// below for the full story.
    nonisolated(unsafe) private static var mppFallbackLogged = false

    /// Shared per-row `decode()` loop — both fallback guards in
    /// `decodeMulti` below (CPU-routed router, MPP-incapable hardware)
    /// use this exact body. It's `DecoderLayer`'s default `decodeMulti`,
    /// replicated here because a type that provides its own
    /// `decodeMulti` can't fall through to the protocol extension's
    /// default from inside itself.
    private static func perRowDecodeMultiFallback(
        layer: NemotronHNvfp4MoELayer, _ h: Tensor, startingAt position: Int,
        cache: any LayerCacheProtocol, cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let t = h.shape[0]
        let rowSize = h.shape.last!
        let result = Tensor.empty(shape: h.shape, dtype: h.dtype, device: device)
        result.zero()
        for i in 0 ..< t {
            let row = h.slicedRows(start: i, count: 1).reshaped(to: [rowSize])
            let out = layer.decode(
                row, position: position + i, cache: cache, cmd: cmd, device: device)
            let dst = result.slicedRows(start: i, count: 1).reshaped(to: [rowSize])
            _ = Ops.add(out, dst, on: cmd, into: dst)
        }
        return result
    }

    public func decodeMulti(
        _ h: Tensor, startingAt position: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        guard usesGpuRouter else {
            // CPU-routed fallback (BUTTER_NEMOTRON_NVFP4_GPU_ROUTER=0, or
            // a checkpoint with normTopkProb==false / topK==1).
            return Self.perRowDecodeMultiFallback(
                layer: self, h, startingAt: position, cache: cache, cmd: cmd, device: device)
        }
        guard Ops.mppAutoCapable else {
            // `Ops.nvfp4MoeGatherBm16`'s default dispatch path (when none
            // of the shape/env-gated non-MPP variants — LagunaNvfp4ExpertGather
            // / LagunaNvfp4MppGather's own `.capable` check, opt-in Steel —
            // fire) unconditionally requests a `matmul2d` MPP cooperative-
            // tensor PSO, which needs Apple GPU Family 9+ / macOS 26+ (same
            // gate as `Ops.mppAutoCapable`; see its doc). Every existing
            // MPP-touching caller in this codebase only ever ran on capable
            // hardware; this layer's batched `decodeMulti` is default-ON,
            // so on incapable hardware (older Macs, CI paravirt GPUs) it
            // must not even attempt the gather-GEMM path — degrade to the
            // per-row loop instead of a Metal-compiler fatalError on an
            // uncompileable PSO.
            if !Self.mppFallbackLogged {
                Self.mppFallbackLogged = true
                FileHandle.standardError.write(
                    ("[nemotronh] Ops.mppAutoCapable == false — batched NVFP4 MoE "
                        + "prefill falling back to the per-token decode() loop (MPP "
                        + "cooperative-tensor kernels need Apple GPU Family 9+ / "
                        + "macOS 26+).\n")
                        .data(using: .utf8)!)
            }
            return Self.perRowDecodeMultiFallback(
                layer: self, h, startingAt: position, cache: cache, cmd: cmd, device: device)
        }

        let t = h.shape[0]
        let rowSize = h.shape.last!
        precondition(
            rowSize == hidden,
            "NemotronHNvfp4MoELayer.decodeMulti: row size \(rowSize) != hidden \(hidden)")

        let xNorm = Ops.rmsNormRows(
            h, weight: norm.weight, eps: norm.eps,
            nRows: t, rowSize: hidden, on: cmd)

        // ── Batched router (f32 throughout — see `eScoreCorrectionBiasGPU`'s
        //    doc for why bf16 silently breaks selection on this
        //    checkpoint's correction-bias distribution). ──────────────────
        let logitsAll = gate.callMany(xNorm, t: t, on: cmd, device: device)
        let logitsF32 = Tensor.empty(shape: [t * nRouted], dtype: .f32, device: device)
        Ops.castToF32(logitsAll.reshaped(to: [t * nRouted]), into: logitsF32, on: cmd)

        let mTotal = t * topK
        let indicesTok = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        let weightsF32 = Tensor.empty(shape: [mTotal], dtype: .f32, device: device)
        Ops.moeRouterTopKBiasedRowsFused(
            logits: logitsF32, bias: eScoreCorrectionBiasGPU,
            indicesOut: indicesTok, weightsOut: weightsF32,
            t: t, nExperts: nRouted, k: topK, on: cmd)
        if routedScalingFactor != 1.0 {
            let scaleVec = Tensor.filled(
                routedScalingFactor, shape: [mTotal], dtype: .f32, device: device)
            _ = Ops.mul(weightsF32, scaleVec, on: cmd, into: weightsF32)
        }
        let weightsAct = Tensor.empty(shape: [mTotal], dtype: dtype, device: device)
        if dtype == .f32 {
            Ops.copy(weightsF32, into: weightsAct, on: cmd)
        } else if dtype == .bf16 {
            Ops.castF32ToBf16(weightsF32, into: weightsAct, on: cmd)
        } else {
            fatalError(
                "NemotronHNvfp4MoELayer.decodeMulti: unsupported activation dtype \(dtype)")
        }

        // ── Sort (token, slot) pairs by chosen expert ─────────────────────
        let sortedExperts = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        let sourceTokens = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        let invPerm = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        Ops.moeSortPlanFused(
            topkIds: indicesTok, sortedExperts: sortedExperts,
            sourceTokens: sourceTokens, invPerm: invPerm,
            mTotal: mTotal, k: topK, nExperts: nRouted, blockSize: 256, on: cmd)

        // ── Gather xNorm rows into expert-sorted (token, slot) order ──────
        let xGathered = Ops.gather(table: xNorm, tokenIds: sourceTokens, on: cmd)

        // ── Routed experts: two gather-GEMM dispatches (up, down) ─────────
        let moeIntermediate = stacked.moeIntermediate
        let upOut = Tensor.empty(shape: [mTotal, moeIntermediate], dtype: dtype, device: device)
        Ops.nvfp4MoeGatherBm16(
            input: xGathered, weight: stacked.upWeight, scales: stacked.upScales,
            indices: sortedExperts, mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
            blockSize: stacked.blockSize, global: 1.0, on: cmd, into: upOut)

        let r = Ops.relu(upOut, on: cmd)
        let inner = Ops.mul(r, r, on: cmd)

        let downOut = Tensor.empty(shape: [mTotal, hidden], dtype: dtype, device: device)
        Ops.nvfp4MoeGatherBm16(
            input: inner, weight: stacked.downWeight, scales: stacked.downScales,
            indices: sortedExperts, mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
            blockSize: stacked.blockSize, global: 1.0, on: cmd, into: downOut)

        // ── Shared expert — always-on, dense (not gathered/routed),
        //    batched over all T rows via Nvfp4Linear.callMany. ─────────────
        var sharedOut: Tensor?
        if let su = sharedUp, let sd = sharedDown {
            let sUp = su.callMany(xNorm, t: t, on: cmd, device: device)
            let sR = Ops.relu(sUp, on: cmd)
            let sInner = Ops.mul(sR, sR, on: cmd)
            sharedOut = sd.callMany(sInner, t: t, on: cmd, device: device)
        }

        // ── Un-sort + weighted combine + shared-expert add + residual add,
        //    all in one dispatch. ───────────────────────────────────────────
        let result = Tensor.empty(shape: [t, hidden], dtype: dtype, device: device)
        Ops.moeUnpermuteTail(
            expertOutputs: downOut, invPerm: invPerm, topKWeights: weightsAct,
            sharedExpert: sharedOut, residual: h, into: result,
            nRows: t, hidden: hidden, k: topK, on: cmd)
        return result
    }
}
