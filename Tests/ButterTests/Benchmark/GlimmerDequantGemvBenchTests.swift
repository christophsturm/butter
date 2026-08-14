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
// GlimmerDequantGemvBenchTests — isolated per-shape GB/s of the shipped
// `Ops.dequantGemv` (bits=4, group_size=64, bf16 scales/biases/input)
// decode kernel (`iron_dequant_gemv_int4_bf16`) at Muse Glimmer 30B's
// EXACT dense-decoder shapes (hidden=6656, nHeads=32, nKVHeads=2,
// headDim=128, intermediate=19968 — read live from
// ~/models/muse-glimmer-30b-4bit/config.json's text_config +
// quantization block, 2026-08-14).
//
// Investigates the decode gap flagged by the #137 census
// ([[glimmer-lightning-ports-shipped]]): dense decode 25.2-25.4 tok/s
// vs the model vendor's 26.6 M5 reference, proven GEMV-bandwidth-bound
// (dispatch-count cut moved <1%). 8 dequant-GEMV projections/layer x 52
// layers stream ~14.6 GB/token; this file measures the GB/s each
// projection shape actually achieves, GPU-only clock (same convention
// as `LagunaNax2GpuOnlyBenchTests` / iron's `runner/gpu.rs::measure`:
// GPUEndTime-GPUStartTime, warmup discarded, median of N).
//
// Also includes a raw MTLBlitCommandEncoder copy at comparable transfer
// size — the "pure memcpy" practical-bandwidth ceiling this Mac can hit,
// independent of any compute kernel — and one comparison shape pair
// from Phi3's MHA-dense decoder (hidden=3072, intermediate=8192, no
// GQA K/V narrowing) to see whether Glimmer's narrow-K/V (nKVHeads=2)
// shapes land at a different GB/s than a square/MHA shape at the same
// bits/group_size.
//
//   swift test -c release --filter GlimmerDequantGemvBenchTests

import Foundation
import Metal
import IronSwift
import Testing

@testable import ButterSwift

@Suite("Glimmer dequant-GEMV bandwidth (W137 gap hunt)", .serialized)
struct GlimmerDequantGemvBenchTests {

    // MARK: - GPU-only median clock (matches LagunaNax2GpuOnlyBenchTests)

    static func gpuOnlyMedianMs(warmup: Int, iters: Int, _ body: (MTLCommandBuffer) -> Void) -> Double {
        func runOnce() -> Double {
            let cmd = Device.shared.makeCommandBuffer()
            body(cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            return (cmd.gpuEndTime - cmd.gpuStartTime) * 1000.0  // ms
        }
        for _ in 0 ..< warmup { _ = runOnce() }
        var samples: [Double] = []
        for _ in 0 ..< iters { samples.append(runOnce()) }
        samples.sort()
        return samples[samples.count / 2]
    }

    // MARK: - random fill helpers

    static func randomU32(count n: Int, seed: inout UInt32) -> [UInt32] {
        var out = [UInt32](repeating: 0, count: n)
        for i in 0 ..< n {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            out[i] = seed
        }
        return out
    }

    // Small-magnitude bf16 bit patterns (as raw UInt16), bounded so the
    // dequant accumulation can't hit inf — bandwidth timing doesn't care
    // about the numeric result, but a clean finite run avoids any NaN/Inf
    // corner-case the ALU dequant path might special-case.
    static func randomBf16Bits(count n: Int, seed: inout UInt32, scale: Float) -> [UInt16] {
        var bits = [UInt16](repeating: 0, count: n)
        for i in 0 ..< n {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            let u = Float(Int(seed % 2000)) / 1000.0 - 1.0
            bits[i] = UInt16((( u * scale).bitPattern >> 16) & 0xFFFF)
        }
        return bits
    }

    // MARK: - byte model (weight + scales + biases; input/output are
    // negligible at these out/in dims — e.g. the biggest MLP shape moves
    // ~74.7 MB of weight+scale+bias vs ~40 KB of input+output).

    static func gemvBytes(outDim: Int, inDim: Int, bits: Int, groupSize: Int) -> Int {
        let packedPerRow = inDim * bits / 32
        let weightBytes = outDim * packedPerRow * 4  // u32
        let nGroups = inDim / groupSize
        let scaleBiasBytes = outDim * nGroups * 2 * 2  // bf16 scales + bf16 biases
        let inputBytes = inDim * 2
        let outputBytes = outDim * 2
        return weightBytes + scaleBiasBytes + inputBytes + outputBytes
    }

    static func gbps(bytes: Int, ms: Double) -> Double {
        Double(bytes) / (ms / 1000.0) / 1e9
    }

    // MARK: - one dequant-gemv shape, GPU-only median GB/s

    static func benchShape(
        label: String, outDim: Int, inDim: Int, bits: Int = 4, groupSize: Int = 64,
        warmup: Int = 20, iters: Int = 20
    ) -> (ms: Double, gbps: Double) {
        let device = Device.shared
        var seed: UInt32 = UInt32(label.utf8.reduce(0) { $0 &+ UInt32($1) }) &+ 7
        let packedPerRow = inDim * bits / 32
        let nGroups = inDim / groupSize

        let weight = Tensor.empty(shape: [outDim, packedPerRow], dtype: .u32, device: device)
        weight.copyIn(from: Self.randomU32(count: outDim * packedPerRow, seed: &seed))

        let scales = Tensor.empty(shape: [outDim, nGroups], dtype: .bf16, device: device)
        scales.copyIn(from: Self.randomBf16Bits(count: outDim * nGroups, seed: &seed, scale: 0.05))

        let biases = Tensor.empty(shape: [outDim, nGroups], dtype: .bf16, device: device)
        biases.copyIn(from: Self.randomBf16Bits(count: outDim * nGroups, seed: &seed, scale: 0.05))

        let input = Tensor.empty(shape: [inDim], dtype: .bf16, device: device)
        input.copyIn(from: Self.randomBf16Bits(count: inDim, seed: &seed, scale: 1.0))

        let out = Tensor.empty(shape: [outDim], dtype: .bf16, device: device)

        let ms = Self.gpuOnlyMedianMs(warmup: warmup, iters: iters) { cmd in
            _ = Ops.dequantGemv(
                weight: weight, scales: scales, biases: biases,
                input: input, bits: bits, groupSize: groupSize,
                on: cmd, into: out)
        }
        let bytes = Self.gemvBytes(outDim: outDim, inDim: inDim, bits: bits, groupSize: groupSize)
        return (ms, Self.gbps(bytes: bytes, ms: ms))
    }

    // MARK: - pure blit "practical peak" — no compute, just HW-coalesced
    // GPU-side copy. Bandwidth = (read + write) bytes / time, the
    // conventional memcpy-bandwidth convention.

    static func blitPracticalPeakGbps(bytes: Int, warmup: Int = 10, iters: Int = 20) -> Double {
        let mtlDevice = Device.shared.mtlDevice
        guard let src = mtlDevice.makeBuffer(length: bytes, options: .storageModePrivate),
            let dst = mtlDevice.makeBuffer(length: bytes, options: .storageModePrivate)
        else {
            fatalError("blitPracticalPeakGbps: makeBuffer failed")
        }
        func runOnce() -> Double {
            let cmd = Device.shared.makeCommandBuffer()
            guard let enc = cmd.makeBlitCommandEncoder() else {
                fatalError("blitPracticalPeakGbps: makeBlitCommandEncoder failed")
            }
            enc.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: bytes)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            return (cmd.gpuEndTime - cmd.gpuStartTime) * 1000.0
        }
        for _ in 0 ..< warmup { _ = runOnce() }
        var samples: [Double] = []
        for _ in 0 ..< iters { samples.append(runOnce()) }
        samples.sort()
        let ms = samples[samples.count / 2]
        // read + write, the standard memcpy-bandwidth convention.
        return Self.gbps(bytes: bytes * 2, ms: ms)
    }

    @Test("pure-blit practical bandwidth ceiling (M5 Max, this box)")
    func blitPracticalPeak() {
        // 512 MiB — big enough to amortize blit-encoder fixed overhead
        // (the smallest dequant-gemv shape here moves ~1MB, the biggest
        // ~75MB; 512MB sits comfortably above both so the ceiling number
        // isn't itself dispatch-overhead-bound).
        let bytes = 512 * 1024 * 1024
        let peak = Self.blitPracticalPeakGbps(bytes: bytes)
        print(String(format: "[glimmer-gemv-bw] blit practical peak: %.1f GB/s (512 MiB copy)", peak))
        #expect(peak > 0)
    }

    @Test("Glimmer 30B dense-decoder dequant-GEMV GB/s per projection shape")
    func glimmerShapes() {
        // hidden=6656, nHeads=32, nKVHeads=2, headDim=128,
        // intermediate=19968, bits=4, group_size=64, bf16 activations —
        // read live from ~/models/muse-glimmer-30b-4bit/config.json
        // (text_config + top-level quantization block), 2026-08-14.
        let hidden = 6656
        let qOut = 32 * 128  // 4096
        let kvOut = 2 * 128  // 256
        let intermediate = 19968

        let shapes: [(label: String, outDim: Int, inDim: Int)] = [
            ("q_proj", qOut, hidden),
            ("k_proj", kvOut, hidden),
            ("v_proj", kvOut, hidden),
            ("o_proj", hidden, qOut),
            ("attn_gate", qOut, hidden),  // sigmoid attention-output gate, same shape as q_proj
            ("mlp_gate", intermediate, hidden),
            ("mlp_up", intermediate, hidden),
            ("mlp_down", hidden, intermediate),
        ]

        var totalBytesPerLayer = 0
        var totalMsPerLayer = 0.0
        for (label, outDim, inDim) in shapes {
            let (ms, gbps) = Self.benchShape(label: label, outDim: outDim, inDim: inDim)
            let bytes = Self.gemvBytes(outDim: outDim, inDim: inDim, bits: 4, groupSize: 64)
            totalBytesPerLayer += bytes
            totalMsPerLayer += ms
            print(String(
                format: "[glimmer-gemv-bw] %@ out=%d in=%d  %.4fms  %.1f GB/s  (%.2f MB)",
                label, outDim, inDim, ms, gbps, Double(bytes) / 1e6))
        }
        let layerGbps = Self.gbps(bytes: totalBytesPerLayer, ms: totalMsPerLayer)
        let perTokenBytes = totalBytesPerLayer * 52
        let modeledTokPerS = 1000.0 / (totalMsPerLayer * 52.0)
        print(String(
            format: "[glimmer-gemv-bw] SUM/layer: %.4fms  %.1f GB/s  (%.2f MB); x52 layers = %.2f GB/token, isolated-kernel-only model = %.1f tok/s",
            totalMsPerLayer, layerGbps, Double(totalBytesPerLayer) / 1e6,
            Double(perTokenBytes) / 1e9, modeledTokPerS))
    }

    @Test("Phi3 MHA-dense comparison shapes (hidden=3072, no GQA K/V narrowing)")
    func phi3ComparisonShapes() {
        // hidden=3072, nHeads=nKVHeads=32 (MHA — no narrow K/V), headDim=96,
        // intermediate=8192 (Tests/ModelIntegrationTests/Text/Phi3IntegrationTests.swift).
        // Same bits=4/group_size=64/bf16 kernel family as Glimmer; the point
        // is to see whether GB/s moves with shape (narrow-K/V vs square,
        // smaller hidden vs Glimmer's 6656) or stays flat — i.e. whether
        // Glimmer's numbers are kernel-generic or shape-specific.
        let hidden = 3072
        let intermediate = 8192

        let shapes: [(label: String, outDim: Int, inDim: Int)] = [
            ("phi3_qkvo_square", hidden, hidden),  // MHA: q/k/v/o all hidden x hidden
            ("phi3_mlp_gate_up", intermediate, hidden),
            ("phi3_mlp_down", hidden, intermediate),
        ]
        for (label, outDim, inDim) in shapes {
            let (ms, gbps) = Self.benchShape(label: label, outDim: outDim, inDim: inDim)
            print(String(
                format: "[glimmer-gemv-bw] %@ out=%d in=%d  %.4fms  %.1f GB/s", label, outDim, inDim, ms, gbps))
        }
    }
}
