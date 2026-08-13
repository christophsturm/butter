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
// Host-side codec coverage for the Nemotron-3.5-Lightning-30B-A3B-NVFP4
// port. These tests exercise the pure CPU math (E4M3 codec, weight
// repacking, global-scale folding) with synthetic data — no GPU
// checkpoint download required, unlike the full load/decode/coherence
// gate which needs the real ~17 GB checkpoint.

import Foundation
import Metal
import Testing

@testable import ButterSwift

@Suite("NemotronH NVFP4/FP8 mixed-precision loader helpers")
struct NemotronHNvfp4Tests {

    // ─── E4M3 (float8_e4m3fn) scalar codec ────────────────────────────

    @Test("E4M3 decode matches hand-computed reference values")
    func e4m3DecodeKnownValues() {
        // 0x00 = S0 E0000 M000 -> +0.0
        #expect(E4M3.decode(0x00) == 0.0)
        // 0x80 = S1 E0000 M000 -> -0.0
        #expect(E4M3.decode(0x80) == 0.0 || E4M3.decode(0x80) == -0.0)
        // 0x38 = 0b0_0111_000 -> sign 0, exp 7 (stored), mant 0
        //   -> 2^(7-7) * (1 + 0/8) = 1.0
        #expect(abs(E4M3.decode(0x38) - 1.0) < 1e-6)
        // 0xB8 = 0b1_0111_000 -> same magnitude, negative -> -1.0
        #expect(abs(E4M3.decode(0xB8) - (-1.0)) < 1e-6)
        // 0x40 = 0b0_1000_000 -> exp 8 (stored), mant 0 -> 2^1 * 1 = 2.0
        #expect(abs(E4M3.decode(0x40) - 2.0) < 1e-6)
        // 0x7E = 0b0_1111_110 -> exp 15 (stored), mant 6 -> max finite 448.
        #expect(abs(E4M3.decode(0x7E) - 448.0) < 1e-3)
        // 0x7F = 0b0_1111_111 -> NaN.
        #expect(E4M3.decode(0x7F).isNaN)
        #expect(E4M3.decode(0xFF).isNaN)
        // 0x01 = smallest positive subnormal = 2^-6 * (1/8) = 2^-9.
        #expect(abs(E4M3.decode(0x01) - Float(exactly: 1)! / 512.0) < 1e-9)
    }

    @Test("E4M3 encode(decode(byte)) round-trips for every representable byte")
    func e4m3RoundTripsAllBytes() {
        for raw in 0 ..< 256 {
            let byte = UInt8(raw)
            let exp = (byte >> 3) & 0x0F
            let mant = byte & 0x07
            if exp == 0x0F && mant == 0x07 { continue }  // NaN — no round trip.
            let value = E4M3.decode(byte)
            let reencoded = E4M3.encodeNearest(value)
            #expect(
                E4M3.decode(reencoded) == value,
                Comment(
                    rawValue: "byte \(String(byte, radix: 2)) -> \(value) -> re-encoded "
                        + "\(reencoded) -> \(E4M3.decode(reencoded))"))
        }
    }

    @Test("E4M3 encode saturates overflow to max finite instead of NaN")
    func e4m3EncodeSaturatesOverflow() {
        let byte = E4M3.encodeNearest(1_000_000.0)
        #expect(!E4M3.decode(byte).isNaN)
        #expect(abs(E4M3.decode(byte) - 448.0) < 1e-3)
        let negByte = E4M3.encodeNearest(-1_000_000.0)
        #expect(abs(E4M3.decode(negByte) - (-448.0)) < 1e-3)
    }

    // ─── NVFP4 weight repack (HF u8[out,in/2] -> Butter u32[out,in/8]) ─

    @Test("repackNvfp4Weight preserves bytes exactly (pure reinterpretation)")
    func repackPreservesBytes() {
        autoreleasepool {
            let device = Device.shared
            let outDim = 4
            let inDim = 32  // in/2 = 16 bytes/row, in/8 = 4 u32 lanes/row.
            var bytes = [UInt8](repeating: 0, count: outDim * (inDim / 2))
            for i in 0 ..< bytes.count { bytes[i] = UInt8((i * 37 + 5) % 256) }
            let hfWeight = Tensor.empty(shape: [outDim, inDim / 2], dtype: .u8, device: device)
            hfWeight.copyIn(from: bytes)

            let repacked = repackNvfp4Weight(hfWeight, outDim: outDim, inDim: inDim, device: device)
            #expect(repacked.dtype == .u32)
            #expect(repacked.shape == [outDim, inDim / 8])

            // Byte-for-byte identity: `Tensor.toArray(as:)` sizes its
            // read by `elementCount` in the TENSOR's own dtype (16 u32
            // lanes here, not 64 u8 bytes), so read as u32 and
            // decompose each lane back to little-endian bytes by hand
            // — matching how `repackNvfp4Weight`'s memcpy actually laid
            // them out on this (little-endian, Apple Silicon) platform.
            let repackedLanes = repacked.toArray(as: UInt32.self)
            var repackedBytes: [UInt8] = []
            repackedBytes.reserveCapacity(repackedLanes.count * 4)
            for lane in repackedLanes {
                repackedBytes.append(UInt8(lane & 0xFF))
                repackedBytes.append(UInt8((lane >> 8) & 0xFF))
                repackedBytes.append(UInt8((lane >> 16) & 0xFF))
                repackedBytes.append(UInt8((lane >> 24) & 0xFF))
            }
            #expect(repackedBytes == bytes)
        }
    }

    // ─── Global-scale folding into per-block E4M3 micro-scales ────────

    @Test("foldNvfp4GlobalIntoScales with global=1.0 is a byte-identical copy")
    func foldIdentityScale() {
        autoreleasepool {
            let device = Device.shared
            let bytes: [UInt8] = [0x38, 0x40, 0x00, 0x30, 0x2C]  // assorted E4M3 codes.
            let scaleT = Tensor.empty(shape: [bytes.count], dtype: .u8, device: device)
            scaleT.copyIn(from: bytes)
            let folded = foldNvfp4GlobalIntoScales(scaleT, global: 1.0, device: device)
            #expect(folded.toArray(as: UInt8.self) == bytes)
        }
    }

    @Test("foldNvfp4GlobalIntoScales(global: 2.0) doubles the decoded scale value")
    func foldDoublesScale() {
        autoreleasepool {
            let device = Device.shared
            // 0x38 decodes to 1.0; folding in a global of 2.0 should
            // land on (or very near) the E4M3 code for 2.0 (0x40).
            let scaleT = Tensor.empty(shape: [1], dtype: .u8, device: device)
            scaleT.copyIn(from: [UInt8(0x38)])
            let folded = foldNvfp4GlobalIntoScales(scaleT, global: 2.0, device: device)
            let decoded = E4M3.decode(folded.toArray(as: UInt8.self)[0])
            #expect(abs(decoded - 2.0) < 1e-3)
        }
    }

    // ─── FP8 -> bf16 dense dequantization (mamba in_proj/out_proj) ────

    @Test("dequantFp8ToDType reproduces scale * E4M3(byte) within bf16 rounding")
    func fp8DequantMatchesReference() {
        autoreleasepool {
            let device = Device.shared
            let outDim = 2
            let inDim = 4
            // 0x38 (=1.0), 0x40 (=2.0), 0x00 (=0.0), 0xB8 (=-1.0), ...
            let bytes: [UInt8] = [0x38, 0x40, 0x00, 0xB8, 0x38, 0x38, 0x40, 0x00]
            let f8 = Tensor.empty(shape: [outDim, inDim], dtype: .u8, device: device)
            f8.copyIn(from: bytes)
            let scale: Float = 0.5
            let out = dequantFp8ToDType(
                f8, scale: scale, outDim: outDim, inDim: inDim,
                dtype: .bf16, device: device)
            let result = out.toFloatArray()
            for i in 0 ..< bytes.count {
                let expected = E4M3.decode(bytes[i]) * scale
                #expect(
                    abs(result[i] - expected) < 5e-3,
                    Comment(rawValue: "index \(i): \(result[i]) vs \(expected)"))
            }
        }
    }

    // ─── F-128: mamba in_proj/out_proj FP8→int4 requant ───────────────

    @Test("requantizeDenseToInt4NH round-trips a dense weight within int4 quantization tolerance")
    func mambaProjInt4RoundTrips() {
        autoreleasepool {
            let device = Device.shared
            let outDim = 8
            let inDim = 64  // exactly one group_size=64 group per row.

            // Synthetic weight with a realistic spread of values per
            // row (not all zero/uniform, so the per-group affine
            // scale+bias is actually exercised) plus a small per-row
            // offset so rows don't share identical quantization
            // parameters.
            var floats = [Float](repeating: 0, count: outDim * inDim)
            for i in 0 ..< floats.count {
                let row = i / inDim
                let x = Float(i % 37) - 18.0
                floats[i] = x * 0.037 + Float(row) * 0.01
            }
            func toBF16(_ v: Float) -> UInt16 {
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            }
            let dense = Tensor.empty(shape: [outDim, inDim], dtype: .bf16, device: device)
            dense.copyIn(from: floats.map(toBF16))

            let quant = requantizeDenseToInt4NH(dense, outDim: outDim, inDim: inDim, device: device)
            #expect(quant.bits == 4)
            #expect(quant.groupSize == 64)
            #expect(quant.weight.shape == [outDim, inDim * 4 / 32])
            #expect(quant.scales.shape == [outDim, inDim / 64])
            #expect(quant.biases.shape == [outDim, inDim / 64])

            // Dequantize back with the same kernel family
            // `QuantizedLinear`'s decode GEMV drives
            // (`QuantizedOps.dequantizeAffine`'s int4 path), and check
            // the per-element error stays within a typical int4
            // quantization step relative to this synthetic weight's
            // spread (~1.33 per row / 15 int4 steps ≈ 0.089, plus
            // negligible bf16 rounding on top).
            let dequant = Tensor.empty(shape: [outDim, inDim], dtype: .bf16, device: device)
            let cmd = device.makeCommandBuffer()
            QuantizedOps.dequantizeAffine(
                weight: quant.weight, scales: quant.scales, biases: quant.biases,
                into: dequant, bits: 4, groupSize: 64, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            let original = dense.toFloatArray()
            let roundtrip = dequant.toFloatArray()
            var maxAbsErr: Float = 0
            for i in 0 ..< original.count {
                maxAbsErr = max(maxAbsErr, abs(original[i] - roundtrip[i]))
            }
            #expect(
                maxAbsErr < 0.15,
                Comment(rawValue: "max abs roundtrip error \(maxAbsErr) exceeds int4 tolerance"))
        }
    }

    @Test(
        "loadMambaProjNH's inDim%64 guard is load-bearing: bypassing it lets groups span row boundaries"
    )
    func mambaProjInt4GuardPreventsRowSpillover() {
        autoreleasepool {
            // `loadMambaProjNH` only takes the int4 requant path when
            // `inDim % 64 == 0` — NOT because the flattened `numel`
            // fails `QuantizedOps.quantizeAffine`'s own validation
            // (numel can still be a clean multiple of 64 even when a
            // single row isn't, e.g. outDim=2/inDim=96 -> numel=192 =
            // 3*64), but because group_size=64 groups the WEIGHT'S
            // FLAT BUFFER, and a non-64-aligned row width lets one
            // group's 64 elements straddle two logically unrelated
            // rows. This test bypasses `loadMambaProjNH`'s guard on
            // purpose to prove that danger is real, not defensive
            // paranoia: row 0 oscillates tightly around +1.0, row 1
            // tightly around -1.0 (each row's OWN natural quantization
            // step would be tiny, ~0.0004). The second group (elements
            // 64..127) straddles row boundary 96, forcing that group's
            // shared affine scale/bias to cover the full [-1,1] span —
            // ~330x coarser than either row's natural step — so values
            // in that straddled group must read back far worse than
            // the equivalent elements in a properly-aligned group
            // (NOT just the two exact extremes, which affine quant
            // always reproduces exactly regardless of grouping — this
            // uses oscillating, non-extremal values specifically so
            // that trivial case can't mask the corruption).
            let device = Device.shared
            let outDim = 2
            let inDim = 96  // NOT a multiple of 64 — the guarded case.
            let numel = outDim * inDim
            #expect(inDim % 64 != 0)
            #expect(numel % 64 == 0, "numel alone looks aligned — that's the trap")

            func toBF16(_ v: Float) -> UInt16 {
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            }
            var floats = [Float](repeating: 0, count: numel)
            for i in 0 ..< numel {
                // ±0.2 wobble — well above bf16's own ~0.008 ulp near
                // magnitude 1.0, so bf16 storage rounding doesn't
                // dominate the signal this test is isolating.
                let osc = Float(i % 7 - 3) * 0.0667
                floats[i] = (i < inDim ? 1.0 : -1.0) + osc
            }
            let dense = Tensor.empty(shape: [outDim, inDim], dtype: .bf16, device: device)
            dense.copyIn(from: floats.map(toBF16))

            let groupSize = 64
            let nGroups = numel / groupSize
            let packed = Tensor.empty(shape: [nGroups * groupSize * 4 / 32], dtype: .u32, device: device)
            let scales = Tensor.empty(shape: [nGroups], dtype: .bf16, device: device)
            let biases = Tensor.empty(shape: [nGroups], dtype: .bf16, device: device)
            let cmd = device.makeCommandBuffer()
            QuantizedOps.quantizeAffine(
                weight: dense, packed: packed, scales: scales, biases: biases,
                bits: 4, groupSize: groupSize, on: cmd)
            let dequant = Tensor.empty(shape: [outDim, inDim], dtype: .bf16, device: device)
            QuantizedOps.dequantizeAffine(
                weight: packed, scales: scales, biases: biases,
                into: dequant, bits: 4, groupSize: groupSize, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            let original = floats
            let roundtrip = dequant.toFloatArray()
            var perElemErr = [Float](repeating: 0, count: numel)
            for i in 0 ..< numel { perElemErr[i] = abs(roundtrip[i] - original[i]) }

            // Group 0 (elements 0..63) sits entirely inside row 0 —
            // clean, tightly-clustered ~1.0±0.003 values, so its
            // natural quantization step is tiny.
            let cleanGroupErr = perElemErr[0 ..< 64].max()!
            // Group 1 (elements 64..127) straddles the row-96
            // boundary — its shared scale/bias must span roughly
            // [-1.003, 1.003] instead of either row's true ~0.006
            // range, so its worst-case error should be dramatically
            // larger than the clean group's.
            let straddledGroupErr = perElemErr[64 ..< 128].max()!
            #expect(
                straddledGroupErr > cleanGroupErr * 3,
                Comment(
                    rawValue: "expected the row-96-straddling group's max error (\(straddledGroupErr)) "
                        + "to dwarf the clean group's (\(cleanGroupErr)) — if this fails, "
                        + "group_size=64 stopped requiring inDim%64==0 alignment and "
                        + "loadMambaProjNH's guard may be stale"))
        }
    }

    // ─── F-126: GPU-resident router vs the CPU reference ──────────────
    //
    // `NemotronHNvfp4MoELayer.decodeGPURouted` replaces the per-layer
    // CPU commit+wait router (`decodeCPURouted`) with
    // `Ops.moeRouterTopKBiasedRowsFused` — this test exercises just the
    // routing math (sigmoid + correction-bias top-K, sum-to-1 renorm,
    // routed_scaling_factor) in isolation, at the real Lightning shape
    // (128 experts, top-6), without needing a full NVFP4 expert stack.

    @Test("GPU sigmoid+bias top-K router matches the CPU reference (ids as a set, weights cos-tight)")
    func gpuRouterMatchesCPUReference() {
        autoreleasepool {
            let device = Device.shared
            let nExperts = 128
            let topK = 6
            let routedScalingFactor: Float = 2.5

            // Deterministic pseudo-random logits/bias — reproducible
            // without a PRNG dependency, and spread enough to exercise
            // the sort (including near-ties).
            func pseudoRandom(_ seed: Int) -> Float {
                var x = UInt64(bitPattern: Int64(seed)) &* 2_654_435_761 &+ 0x9E37_79B9_7F4A_7C15
                x ^= x >> 33
                x &*= 0xff51_afd7_ed55_8ccd
                x ^= x >> 33
                let u = Double(x % 1_000_000) / 1_000_000.0
                return Float(u * 6.0 - 3.0)  // spread over [-3, 3]
            }
            let logitsHost = (0 ..< nExperts).map { pseudoRandom($0) }
            let biasHost = (0 ..< nExperts).map { pseudoRandom($0 + 10_000) * 0.1 }

            func toBF16(_ v: Float) -> UInt16 {
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            }
            func fromBF16(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }

            let logitsBF = logitsHost.map(toBF16)
            let biasBF = biasHost.map(toBF16)

            let logitsT = Tensor.empty(shape: [nExperts], dtype: .bf16, device: device)
            logitsT.copyIn(from: logitsBF)
            let biasT = Tensor.empty(shape: [nExperts], dtype: .bf16, device: device)
            biasT.copyIn(from: biasBF)

            // CPU reference — identical math to
            // `NemotronHNvfp4MoELayer.decodeCPURouted`, reading back the
            // SAME bf16-rounded values the GPU kernel sees (matches
            // `logitsTensor.toFloatArray()` in the real decode path).
            let logitsRef = logitsBF.map(fromBF16)
            let biasRef = biasBF.map(fromBF16)
            var origScores = [Float](repeating: 0, count: nExperts)
            var selectionScores = [Float](repeating: 0, count: nExperts)
            for i in 0 ..< nExperts {
                let s = 1.0 / (1.0 + Foundation.exp(-logitsRef[i]))
                origScores[i] = s
                selectionScores[i] = s + biasRef[i]
            }
            let order = (0 ..< nExperts).sorted { a, b in
                if selectionScores[a] != selectionScores[b] {
                    return selectionScores[a] > selectionScores[b]
                }
                return a < b
            }
            let indsRef = Array(order.prefix(topK))
            var weightsRef = indsRef.map { origScores[$0] }
            let sum = weightsRef.reduce(0, +) + 1e-20
            weightsRef = weightsRef.map { $0 / sum * routedScalingFactor }

            // GPU — the exact dispatch chain `decodeGPURouted` uses.
            let idxOut = Tensor.empty(shape: [topK], dtype: .u32, device: device)
            let wOut = Tensor.empty(shape: [topK], dtype: .bf16, device: device)
            let scaleVec = Tensor.filled(
                routedScalingFactor, shape: [topK], dtype: .bf16, device: device)
            let cmd = device.makeCommandBuffer()
            Ops.moeRouterTopKBiasedRowsFused(
                logits: logitsT, bias: biasT,
                indicesOut: idxOut, weightsOut: wOut,
                t: 1, nExperts: nExperts, k: topK, on: cmd)
            _ = Ops.mul(wOut, scaleVec, on: cmd, into: wOut)
            cmd.commit()
            cmd.waitUntilCompleted()

            let idxGPU = idxOut.toArray(as: UInt32.self).map { Int($0) }
            let wGPU = wOut.toFloatArray()

            // ids: same SET. Slot order may legitimately differ from
            // the CPU stable sort (the summed weighted contribution
            // across experts is order-independent), but the chosen
            // experts themselves must match exactly.
            #expect(
                Set(idxGPU) == Set(indsRef),
                Comment(
                    rawValue: "GPU ids \(idxGPU.sorted()) vs CPU ids \(indsRef.sorted())"))

            // weights: match by expert id, cosine-tight — the fused
            // kernel's bf16 intermediate rounding (see its module doc)
            // can differ slightly from the CPU path's pure-f32 sigmoid,
            // even though both read the identical bf16 input tensor.
            var refByID: [Int: Float] = [:]
            for (i, id) in indsRef.enumerated() { refByID[id] = weightsRef[i] }
            for (i, id) in idxGPU.enumerated() {
                guard let expected = refByID[id] else { continue }
                #expect(
                    abs(wGPU[i] - expected) < 5e-3,
                    Comment(rawValue: "expert \(id): gpu \(wGPU[i]) vs cpu \(expected)"))
            }
        }
    }

    /// Regression test for the exact bug the F-126 perf wave hit
    /// against the real Lightning checkpoint: `e_score_correction_bias`
    /// sits on a large common-mode offset (~40.7, an auxiliary-load-
    /// balancing bias that drifted well off zero during training) with
    /// per-expert deltas of only ~0.01 riding on top. bf16 (~8 mantissa
    /// bits) can't resolve deltas that small at that magnitude — every
    /// bias value rounds to the SAME bf16 bucket, and top-K selection
    /// degenerates to near-arbitrary. Reproduced live: the bf16-typed
    /// router (`gpuRouterMatchesCPUReference` above, whose synthetic
    /// bias is small/zero-mean) matched the CPU reference perfectly in
    /// isolation, yet a real-checkpoint decode picked a completely
    /// different expert set at layer 0 of token 0 — `bias[0..8]` read
    /// back as `[40.75, 40.75, ...]` (all equal) instead of the true
    /// `[40.687637, 40.693634, ...]`.
    ///
    /// This test exercises the ACTUAL production dispatch chain
    /// (`NemotronHNvfp4MoELayer.decodeGPURouted`'s routing math): logits
    /// upcast bf16→f32 (`Ops.castToF32`), bias held in f32 throughout
    /// (never bf16), the f32 kernel variant of
    /// `Ops.moeRouterTopKBiasedRowsFused` (no internal `bfloat(...)`
    /// rounding — see that kernel's `.metal` source vs. the bf16
    /// variant's), scaling in f32, and only the FINAL combine weight
    /// downcast to bf16 (`Ops.castF32ToBf16`) — precision-sensitive
    /// math never touches bf16 until after the expert selection is
    /// already locked in.
    @Test("GPU router survives a large common-mode bias offset (the real Lightning bias shape)")
    func gpuRouterMatchesCPUReferenceWithLargeCommonModeBias() {
        autoreleasepool {
            let device = Device.shared
            let nExperts = 128
            let topK = 6
            let routedScalingFactor: Float = 2.5
            let commonModeOffset: Float = 40.69  // matches the real checkpoint's magnitude.

            func pseudoRandom(_ seed: Int) -> Float {
                var x = UInt64(bitPattern: Int64(seed)) &* 2_654_435_761 &+ 0x9E37_79B9_7F4A_7C15
                x ^= x >> 33
                x &*= 0xff51_afd7_ed55_8ccd
                x ^= x >> 33
                let u = Double(x % 1_000_000) / 1_000_000.0
                return Float(u * 6.0 - 3.0)
            }
            let logitsHost = (0 ..< nExperts).map { pseudoRandom($0) }
            // Deltas of ~0.01 on top of the common-mode offset — the
            // exact shape that broke bf16 (an LSB spacing of ~0.16-0.3
            // at this magnitude swallows deltas this small).
            let biasHost = (0 ..< nExperts).map {
                commonModeOffset + pseudoRandom($0 + 10_000) * 0.01
            }

            func toBF16(_ v: Float) -> UInt16 {
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            }
            func fromBF16(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }

            // Router GEMV output is bf16 in production (the activation
            // dtype) — round-trip the logits through bf16 exactly as
            // the real `gate(xNorm, on: cmd)` GEMV would.
            let logitsBF = logitsHost.map(toBF16)
            let logitsT = Tensor.empty(shape: [nExperts], dtype: .bf16, device: device)
            logitsT.copyIn(from: logitsBF)
            // Bias stays f32 end to end — this IS the fix under test.
            let biasT = Tensor.empty(shape: [nExperts], dtype: .f32, device: device)
            biasT.copyIn(from: biasHost)

            // CPU reference — identical math to `decodeCPURouted`.
            let logitsRef = logitsBF.map(fromBF16)
            var origScores = [Float](repeating: 0, count: nExperts)
            var selectionScores = [Float](repeating: 0, count: nExperts)
            for i in 0 ..< nExperts {
                let s = 1.0 / (1.0 + Foundation.exp(-logitsRef[i]))
                origScores[i] = s
                selectionScores[i] = s + biasHost[i]
            }
            let order = (0 ..< nExperts).sorted { a, b in
                if selectionScores[a] != selectionScores[b] {
                    return selectionScores[a] > selectionScores[b]
                }
                return a < b
            }
            let indsRef = Array(order.prefix(topK))
            var weightsRef = indsRef.map { origScores[$0] }
            let sum = weightsRef.reduce(0, +) + 1e-20
            weightsRef = weightsRef.map { $0 / sum * routedScalingFactor }

            // GPU — mirrors `decodeGPURouted` exactly: upcast logits,
            // f32 router kernel, f32 scale, THEN downcast the weight.
            let logitsF32 = Tensor.empty(shape: [nExperts], dtype: .f32, device: device)
            let idxOut = Tensor.empty(shape: [topK], dtype: .u32, device: device)
            let wOutF32 = Tensor.empty(shape: [topK], dtype: .f32, device: device)
            let wOutBF16 = Tensor.empty(shape: [topK], dtype: .bf16, device: device)
            let scaleVec = Tensor.filled(
                routedScalingFactor, shape: [topK], dtype: .f32, device: device)
            let cmd = device.makeCommandBuffer()
            Ops.castToF32(logitsT, into: logitsF32, on: cmd)
            Ops.moeRouterTopKBiasedRowsFused(
                logits: logitsF32, bias: biasT,
                indicesOut: idxOut, weightsOut: wOutF32,
                t: 1, nExperts: nExperts, k: topK, on: cmd)
            _ = Ops.mul(wOutF32, scaleVec, on: cmd, into: wOutF32)
            Ops.castF32ToBf16(wOutF32, into: wOutBF16, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            let idxGPU = idxOut.toArray(as: UInt32.self).map { Int($0) }
            let wGPU = wOutBF16.toFloatArray()

            #expect(
                Set(idxGPU) == Set(indsRef),
                Comment(
                    rawValue: "GPU ids \(idxGPU.sorted()) vs CPU ids \(indsRef.sorted()) "
                        + "(large common-mode bias case)"))

            var refByID: [Int: Float] = [:]
            for (i, id) in indsRef.enumerated() { refByID[id] = weightsRef[i] }
            for (i, id) in idxGPU.enumerated() {
                guard let expected = refByID[id] else { continue }
                #expect(
                    abs(wGPU[i] - expected) < 5e-3,
                    Comment(rawValue: "expert \(id): gpu \(wGPU[i]) vs cpu \(expected)"))
            }
        }
    }
}
