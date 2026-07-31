// Copyright 2026 Tom Turney (@TheTom)
//
// Diagnostic: every isolated NAX64 test (single dispatch, single command
// buffer, waitUntilCompleted immediately after) passes cleanly, yet
// `butter generate --dump-prefill-logits` with BUTTER_NVFP4_GATHER_NAX64=1
// produces all-NaN logits after a real 39-layer forward pass. This chains
// many NAX64 dispatches (gate/up Pair + down, repeated across simulated
// "layers" using DIFFERENT real weight tensors) on ONE command buffer with
// ONE commit+wait at the end, reusing scratch buffers via
// `device.intermediateScratch` the same way MoELayer does across layers:
// the one structural condition none of the other tests replicate.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("NVFP4 NAX64 chained multi-layer diagnostic", .serialized)
struct Nvfp4GatherNax64ChainTests {
    private var mppAvailable: Bool {
        guard let dev = MTLCreateSystemDefaultDevice() else { return false }
        return dev.supportsFamily(.apple9)
    }

    @Test("NAX64 chained gate+up+down across many layers, one command buffer", .enabled(if: true))
    func chainedManyLayers() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)

        let mTotal = 64
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        var seed: UInt32 = 21
        for i in 0 ..< mTotal {
            seed = seed &* 1664525 &+ 1013904223
            idxHost[i] = seed % 256
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let hidden = 2048
        let moeIntermediate = 512
        var xHost0 = [UInt16](repeating: 0, count: mTotal * hidden)
        for r in 0 ..< mTotal {
            for k in 0 ..< hidden {
                let v: Float = 0.25 * sin(Float(r * 13 + k) * 0.03) + 0.05
                xHost0[r * hidden + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf0 = device.makeBuffer(length: xHost0.count * 2)
        xHost0.withUnsafeBytes { _ = memcpy(xBuf0.contents(), $0.baseAddress!, xHost0.count * 2) }
        var currentH = Tensor(buffer: xBuf0, offset: 0, shape: [mTotal, hidden], dtype: .bf16)

        let cmd = device.makeCommandBuffer()
        var layerFinal: [(Int, Bool)] = []  // (layer, hasNaN)

        for layer in 1 ... 20 {
            let gw = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.gate_proj.weight")
            let gs = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.gate_proj.scales")
            let uw = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.up_proj.weight")
            let us = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.up_proj.scales")
            let dw = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.down_proj.weight")
            let ds = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.down_proj.scales")

            // Same scratch tags every layer -> real MoELayer's grow-only reuse pattern.
            let gBuf = device.intermediateScratch(
                tag: "chain_test_gate", minBytes: mTotal * moeIntermediate * 2)
            let uBuf = device.intermediateScratch(
                tag: "chain_test_up", minBytes: mTotal * moeIntermediate * 2)
            let dBuf = device.intermediateScratch(
                tag: "chain_test_down", minBytes: mTotal * hidden * 2)
            let gateOut = Tensor(buffer: gBuf, offset: 0, shape: [mTotal, moeIntermediate], dtype: .bf16)
            let upOut = Tensor(buffer: uBuf, offset: 0, shape: [mTotal, moeIntermediate], dtype: .bf16)
            let downOut = Tensor(buffer: dBuf, offset: 0, shape: [mTotal, hidden], dtype: .bf16)

            // Gate+up Pair (raw bf16, matches MoELayer's native-bf16 branch).
            Ops.nvfp4MoeGatherNax64Pair(
                input: currentH,
                weight0: gw, scales0: gs, out0: gateOut,
                weight1: uw, scales1: us, out1: upOut,
                indices: indices,
                mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                blockSize: 16, global: 1.0, on: cmd)
            let inner = Ops.swiglu(gate: gateOut, up: upOut, on: cmd, into: gateOut)
            Ops.nvfp4MoeGatherNax64(
                input: inner, weight: dw, scales: ds, indices: indices,
                mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                blockSize: 16, global: 1.0, preferNativeBf16: true,
                on: cmd, into: downOut)

            // Feed this layer's down-proj output as next layer's input
            // (residual-stream stand-in), a fresh buffer per layer so we
            // can inspect each layer's output independently after the run.
            let nextBuf = device.makeBuffer(length: mTotal * hidden * 2)
            let nextH = Tensor(buffer: nextBuf, offset: 0, shape: [mTotal, hidden], dtype: .bf16)
            _ = Ops.add(currentH, downOut, on: cmd, into: nextH)
            currentH = nextH
        }

        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "chain cmd status \(cmd.status.rawValue)")

        let n = currentH.elementCount
        let ptr = currentH.buffer.contents().advanced(by: currentH.offset)
            .bindMemory(to: UInt16.self, capacity: n)
        var nanCount = 0
        for i in 0 ..< n {
            let bits = UInt32(ptr[i]) << 16
            if Float(bitPattern: bits).isNaN { nanCount += 1 }
        }
        print("[nax64-chain] after 20 chained layers: nan=\(nanCount)/\(n)")
        #expect(nanCount == 0, "chained NAX64 across 20 layers produced \(nanCount) NaNs")
    }
}

extension Nvfp4GatherNax64ChainTests {
    /// The real MoELayer f32-promote branch calls `nvfp4MoeGatherBm16Pair`
    /// (-> my `nvfp4MoeGatherNax64Pair`) with F32 input. Every other Pair
    /// test in this repo used bf16 input; this is the first f32-input Pair
    /// test.
    @Test("NAX64 Pair with F32 input (f32-promote branch path)", .enabled(if: true))
    func pairF32Input() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        let uw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.up_proj.weight")
        let us = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.up_proj.scales")
        let nOut = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 64 == 0)

        let mTotal = 64
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        var seed: UInt32 = 71
        for i in 0 ..< mTotal {
            seed = seed &* 1664525 &+ 1013904223
            idxHost[i] = seed % 256
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var xHost = [Float](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            let v: Float = 0.2 + Float(r % 9) * 0.08
            for k in 0 ..< kIn { xHost[r * kIn + k] = v }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 4)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 4) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .f32)

        let outGate = Tensor.empty(shape: [mTotal, nOut], dtype: .f32, device: device)
        let outUp = Tensor.empty(shape: [mTotal, nOut], dtype: .f32, device: device)
        let nanBits: UInt32 = 0x7FC0_0000
        var poison = [UInt32](repeating: nanBits, count: mTotal * nOut)
        poison.withUnsafeBytes {
            _ = memcpy(outGate.buffer.contents().advanced(by: outGate.offset), $0.baseAddress!, mTotal * nOut * 4)
            _ = memcpy(outUp.buffer.contents().advanced(by: outUp.offset), $0.baseAddress!, mTotal * nOut * 4)
        }

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4MoeGatherNax64Pair(
            input: x,
            weight0: gw, scales0: gs, out0: outGate,
            weight1: uw, scales1: us, out1: outUp,
            indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed, "f32 Pair cmd status \(cmd.status.rawValue)")

        func readF32(_ t: Tensor) -> [Float] {
            let n = t.elementCount
            let ptr = t.buffer.contents().advanced(by: t.offset).bindMemory(to: Float.self, capacity: n)
            return (0 ..< n).map { ptr[$0] }
        }
        let gate = readF32(outGate)
        let up = readF32(outUp)
        let gateNan = gate.filter(\.isNaN).count
        let upNan = up.filter(\.isNaN).count
        print("[nax64-pair-f32] gate nan=\(gateNan)/\(gate.count) sample=\(gate.prefix(6))")
        print("[nax64-pair-f32] up   nan=\(upNan)/\(up.count) sample=\(up.prefix(6))")
        #expect(gateNan == 0, "F32 Pair gate produced \(gateNan) NaNs (poison=\(poison.count))")
        #expect(upNan == 0, "F32 Pair up produced \(upNan) NaNs")
        _ = poison
    }
}
