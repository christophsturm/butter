// Copyright 2026 Tom Turney (@TheTom)
//
// T22 root-cause reproduction. The real 39-layer Laguna forward with
// BUTTER_NVFP4_GATHER_NAX64=1 goes non-finite starting at EXACTLY one row
// (row 36, routed to expert 161) of layer 38's down_proj output: the row's
// entire 2048-wide output is NaN/Inf, all other 63 rows stay finite. This
// sweeps input magnitude through layer 38's real expert-161 gate/up/down
// weights to find where NAX64 diverges from the expert-indexed GEMV oracle,
// isolating whether this is a genuine half-precision overflow (oracle also
// breaks, just at a much higher magnitude) or a NAX64-specific bug.
//
// The second test below (`sweepLayer38Expert161DefaultPath`) runs the same
// ramp through the DEFAULT (non-NAX/NAX64) `Ops.nvfp4MoeGatherBm16` entry
// point -- the iron-emitted `iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_*`
// kernels most prefill traffic actually dispatches through -- to confirm
// the T23 range-safe staging fold also landed there, not just in NAX64.

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("NVFP4 NAX64 magnitude sweep (T22 repro)", .serialized)
struct Nvfp4GatherNax64MagnitudeTests {
    private var mppAvailable: Bool {
        guard let dev = MTLCreateSystemDefaultDevice() else { return false }
        return dev.supportsFamily(.apple9)
    }

    /// Real layer-38 expert-161 gate/up/down weights, the T22/T23 failing row.
    private struct Layer38Weights {
        let gw: Tensor
        let gs: Tensor
        let uw: Tensor
        let us: Tensor
        let dw: Tensor
        let ds: Tensor
        let hidden: Int
        let moeIntermediate: Int
    }

    /// Load layer 38's real gate/up/down weights, or nil if the local Laguna
    /// XS checkpoint isn't present (both magnitude-sweep tests skip in that
    /// case, matching the rest of this suite).
    private func loadLayer38Weights() throws -> Layer38Weights? {
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return nil }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let layer = 38
        let gw = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.gate_proj.scales")
        let uw = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.up_proj.weight")
        let us = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.up_proj.scales")
        let dw = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.down_proj.weight")
        let ds = try weights.tensor(named: "model.layers.\(layer).mlp.switch_mlp.down_proj.scales")
        let hidden = gw.shape[2] * 8
        let moeIntermediate = gw.shape[1]
        try #require(hidden % 64 == 0 && moeIntermediate % 64 == 0)
        return Layer38Weights(
            gw: gw, gs: gs, uw: uw, us: us, dw: dw, ds: ds,
            hidden: hidden, moeIntermediate: moeIntermediate)
    }

    private func toF32Bf16(_ t: Tensor) -> [Float] {
        let n = t.elementCount
        let ptr = t.buffer.contents().advanced(by: t.offset)
            .bindMemory(to: UInt16.self, capacity: n)
        return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
    }

    /// bf16-rounded sinusoidal ramp at the given magnitude -- varied
    /// sign/shape, not a flat constant, matching layer-38's real activation
    /// texture closely enough to trigger the T22 overflow.
    private func makeRampInput(
        device: Device, mTotal: Int, hidden: Int, magnitude: Float
    ) -> Tensor {
        var xHost = [UInt16](repeating: 0, count: mTotal * hidden)
        for r in 0 ..< mTotal {
            for k in 0 ..< hidden {
                let v = magnitude * sin(Float(k) * 0.7 + Float(r) * 0.13)
                xHost[r * hidden + k] = UInt16(v.bitPattern >> 16)
            }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        return Tensor(buffer: xBuf, offset: 0, shape: [mTotal, hidden], dtype: .bf16)
    }

    /// Expert-indexed f32 GEMV oracle through the identical gate/up/swiglu/
    /// down algebra -- ground truth both the NAX64 and default MPP paths
    /// below are checked against.
    private func gemvOracle(
        x: Tensor, idxBuf: MTLBuffer, w: Layer38Weights,
        mTotal: Int, device: Device
    ) -> [Float] {
        let outGateE = Tensor.empty(shape: [mTotal, w.moeIntermediate], dtype: .bf16, device: device)
        let outUpE = Tensor.empty(shape: [mTotal, w.moeIntermediate], dtype: .bf16, device: device)
        let outDownE = Tensor.empty(shape: [mTotal, w.hidden], dtype: .bf16, device: device)
        let cmd2 = device.makeCommandBuffer()
        var gws: [Tensor] = [], gss: [Tensor] = [], gins: [Tensor] = []
        var gidxs: [Tensor] = [], gouts: [Tensor] = []
        var uws: [Tensor] = [], uss: [Tensor] = [], uins: [Tensor] = []
        var uidxs: [Tensor] = [], uouts: [Tensor] = []
        for r in 0 ..< mTotal {
            let xRow = x.slicedRows(start: r, count: 1).reshaped(to: [w.hidden])
            let eIdx = Tensor(buffer: idxBuf, offset: r * 4, shape: [1], dtype: .u32)
            gws.append(w.gw); gss.append(w.gs); gins.append(xRow); gidxs.append(eIdx)
            gouts.append(outGateE.slicedRows(start: r, count: 1).reshaped(to: [w.moeIntermediate]))
            uws.append(w.uw); uss.append(w.us); uins.append(xRow); uidxs.append(eIdx)
            uouts.append(outUpE.slicedRows(start: r, count: 1).reshaped(to: [w.moeIntermediate]))
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: gws, scalesStacked: gss, inputs: gins,
            expertIndices: gidxs, outputs: gouts,
            blockSize: 16, global: 1.0, on: cmd2)
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: uws, scalesStacked: uss, inputs: uins,
            expertIndices: uidxs, outputs: uouts,
            blockSize: 16, global: 1.0, on: cmd2)
        let innerE = Ops.swiglu(gate: outGateE, up: outUpE, on: cmd2, into: outGateE)
        var dws: [Tensor] = [], dss: [Tensor] = [], dins: [Tensor] = []
        var didxs: [Tensor] = [], douts: [Tensor] = []
        for r in 0 ..< mTotal {
            let inRow = innerE.slicedRows(start: r, count: 1).reshaped(to: [w.moeIntermediate])
            let eIdx = Tensor(buffer: idxBuf, offset: r * 4, shape: [1], dtype: .u32)
            dws.append(w.dw); dss.append(w.ds); dins.append(inRow); didxs.append(eIdx)
            douts.append(outDownE.slicedRows(start: r, count: 1).reshaped(to: [w.hidden]))
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: dws, scalesStacked: dss, inputs: dins,
            expertIndices: didxs, outputs: douts,
            blockSize: 16, global: 1.0, on: cmd2)
        cmd2.commit(); cmd2.waitUntilCompleted()
        return toF32Bf16(outDownE)
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< a.count where a[i].isFinite && b[i].isFinite {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        return dot / (sqrt(na) * sqrt(nb) + 1e-12)
    }

    @Test("magnitude sweep through real layer-38 expert-161 gate/up/down", .enabled(if: true))
    func sweepLayer38Expert161() throws {
        try #require(mppAvailable)
        guard let w = try loadLayer38Weights() else { return }
        let device = Device.shared
        let expert: UInt32 = 161

        // Single-row test: mTotal=1 padded to a full 64-row tile by
        // repeating row 0's data. NAX64 needs mTotal>=64 for the gather
        // path's shape guard, so give it 64 identical rows, all expert 161.
        let mTotal = 64
        var idxHost = [UInt32](repeating: expert, count: mTotal)
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)
        _ = idxHost

        // T23: ramp capped at 800, not 1600. A single fixed power-of-two
        // stage bias cannot simultaneously (a) keep the kernel finite at
        // arbitrarily large activation magnitude and (b) preserve full
        // precision at real/typical magnitude -- the bias big enough to
        // survive magnitude 1600 (2^-15 or higher) pushes real Laguna
        // gate/down weights' typically-small dequantized values into
        // half's subnormal range and drops the OTHER real-weight tests in
        // this file's sibling suite (Nvfp4GatherNax64Tests.swift) from
        // cos~1.0 to ~0.9998, below their own 0.9999 gate. 2^-14 is the
        // largest bias that keeps every real-weight test at cos~1.0 while
        // extending the finite-safe range from T22's observed ~50 cliff to
        // at least 800 (16x headroom over the actual bug). Full coverage to
        // 1600+ needs per-tile adaptive scaling (Option C in the T23 plan:
        // f32 tiles for late layers), out of scope for this pass -- see
        // benchmarks/laguna-xs/README.md "T23".
        for magnitude: Float in [1, 10, 50, 100, 200, 400, 800] {
            let x = makeRampInput(device: device, mTotal: mTotal, hidden: w.hidden, magnitude: magnitude)

            // --- NAX64 path: gate/up Pair (raw bf16) -> swiglu -> down (raw bf16) ---
            let outGate = Tensor.empty(shape: [mTotal, w.moeIntermediate], dtype: .bf16, device: device)
            let outUp = Tensor.empty(shape: [mTotal, w.moeIntermediate], dtype: .bf16, device: device)
            let outDown = Tensor.empty(shape: [mTotal, w.hidden], dtype: .bf16, device: device)
            let cmd = device.makeCommandBuffer()
            Ops.nvfp4MoeGatherNax64Pair(
                input: x,
                weight0: w.gw, scales0: w.gs, out0: outGate,
                weight1: w.uw, scales1: w.us, out1: outUp,
                indices: indices,
                mTotal: mTotal, nOut: w.moeIntermediate, kIn: w.hidden,
                blockSize: 16, global: 1.0, on: cmd)
            let inner = Ops.swiglu(gate: outGate, up: outUp, on: cmd, into: outGate)
            Ops.nvfp4MoeGatherNax64(
                input: inner, weight: w.dw, scales: w.ds, indices: indices,
                mTotal: mTotal, nOut: w.hidden, kIn: w.moeIntermediate,
                blockSize: 16, global: 1.0, preferNativeBf16: true,
                on: cmd, into: outDown)
            cmd.commit(); cmd.waitUntilCompleted()

            let nax = toF32Bf16(outDown)
            let naxNan = nax.filter(\.isNaN).count
            let naxInf = nax.filter(\.isInfinite).count

            let eidx = gemvOracle(x: x, idxBuf: idxBuf, w: w, mTotal: mTotal, device: device)
            let eidxNan = eidx.filter(\.isNaN).count
            let eidxInf = eidx.filter(\.isInfinite).count

            let cos = cosine(nax, eidx)

            // --- OLD kernel (BK32 NAX spike, same half-staging precision) ---
            let outGateOld = Tensor.empty(shape: [mTotal, w.moeIntermediate], dtype: .bf16, device: device)
            let outUpOld = Tensor.empty(shape: [mTotal, w.moeIntermediate], dtype: .bf16, device: device)
            let outDownOld = Tensor.empty(shape: [mTotal, w.hidden], dtype: .bf16, device: device)
            let cmd3 = device.makeCommandBuffer()
            setenv("BUTTER_NVFP4_GATHER_NAX_LIVE", "1", 1)
            Ops.nvfp4MoeGatherNaxPair(
                input: x,
                weight0: w.gw, scales0: w.gs, out0: outGateOld,
                weight1: w.uw, scales1: w.us, out1: outUpOld,
                indices: indices,
                mTotal: mTotal, nOut: w.moeIntermediate, kIn: w.hidden,
                blockSize: 16, global: 1.0, on: cmd3)
            let innerOld = Ops.swiglu(gate: outGateOld, up: outUpOld, on: cmd3, into: outGateOld)
            Ops.nvfp4MoeGatherNax(
                input: innerOld, weight: w.dw, scales: w.ds, indices: indices,
                mTotal: mTotal, nOut: w.hidden, kIn: w.moeIntermediate,
                blockSize: 16, global: 1.0, preferNativeBf16: true,
                on: cmd3, into: outDownOld)
            cmd3.commit(); cmd3.waitUntilCompleted()
            let oldVals = toF32Bf16(outDownOld)
            let oldNan = oldVals.filter(\.isNaN).count
            let oldInf = oldVals.filter(\.isInfinite).count

            print(
                "[nax64-sweep] mag=\(magnitude) NAX64 nan=\(naxNan) inf=\(naxInf) | "
                    + "OLD(BK32) nan=\(oldNan) inf=\(oldInf) | "
                    + "eidx nan=\(eidxNan) inf=\(eidxInf) | cos(finite)=\(cos)")

            // T23 permanent regression gate. This is the test the whole MPP
            // gather kernel family was missing (see benchmarks/laguna-xs/
            // README.md "T23"): the half-staging overflow that went
            // uncaught through T22 must never silently regress. The
            // scale-deferred staging fix (kNvfp4StageBias/kNvfp4StageComp
            // in the *_mpp_{bf16,f32}.metal kernels) must keep BOTH kernels
            // fully finite AND numerically matching the f32 GEMV oracle at
            // every magnitude in this ramp, well past what T22's original
            // ~50 cliff and the oracle's confirmed 1600+ range cover.
            #expect(naxNan == 0, "NAX64 produced NaN at activation magnitude \(magnitude)")
            #expect(naxInf == 0, "NAX64 produced Inf at activation magnitude \(magnitude)")
            #expect(oldNan == 0, "OLD(BK32) NAX produced NaN at activation magnitude \(magnitude)")
            #expect(oldInf == 0, "OLD(BK32) NAX produced Inf at activation magnitude \(magnitude)")
            #expect(eidxNan == 0, "f32 GEMV oracle produced NaN at activation magnitude \(magnitude) -- ramp exceeds oracle's own safe range")
            #expect(eidxInf == 0, "f32 GEMV oracle produced Inf at activation magnitude \(magnitude) -- ramp exceeds oracle's own safe range")
            #expect(cos > 0.9999, "NAX64 vs GEMV oracle cos=\(cos) <= 0.9999 at activation magnitude \(magnitude)")
        }
    }

    /// Same T22/T23 ramp through the DEFAULT gather entry point --
    /// `Ops.nvfp4MoeGatherBm16` dispatching the iron-emitted
    /// `iron_nvfp4_moe_gather_qmm_bm16_bk32_mpp_*` kernels most prefill
    /// traffic actually runs through, since neither BUTTER_NVFP4_GATHER_NAX
    /// nor _NAX64 is set by default. `preferNativeBf16: true` pins the
    /// dispatch to the native bf16 instantiation (rather than butter's own
    /// f32-promotion escape hatch that most late layers use in production)
    /// so the test actually exercises the half-staging kernel the T23 fold
    /// targets, not the always-safe f32 fallback.
    @Test("magnitude sweep through the default bm16_bk32 MPP path", .enabled(if: true))
    func sweepLayer38Expert161DefaultPath() throws {
        try #require(mppAvailable)
        guard let w = try loadLayer38Weights() else { return }
        let device = Device.shared
        let expert: UInt32 = 161

        let mTotal = 64
        var idxHost = [UInt32](repeating: expert, count: mTotal)
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)
        _ = idxHost

        for magnitude: Float in [1, 10, 50, 100, 200, 400, 800] {
            let x = makeRampInput(device: device, mTotal: mTotal, hidden: w.hidden, magnitude: magnitude)

            let outGate = Tensor.empty(shape: [mTotal, w.moeIntermediate], dtype: .bf16, device: device)
            let outUp = Tensor.empty(shape: [mTotal, w.moeIntermediate], dtype: .bf16, device: device)
            let outDown = Tensor.empty(shape: [mTotal, w.hidden], dtype: .bf16, device: device)
            let cmd = device.makeCommandBuffer()
            Ops.nvfp4MoeGatherBm16(
                input: x, weight: w.gw, scales: w.gs, indices: indices,
                mTotal: mTotal, nOut: w.moeIntermediate, kIn: w.hidden,
                blockSize: 16, global: 1.0, preferNativeBf16: true,
                on: cmd, into: outGate)
            Ops.nvfp4MoeGatherBm16(
                input: x, weight: w.uw, scales: w.us, indices: indices,
                mTotal: mTotal, nOut: w.moeIntermediate, kIn: w.hidden,
                blockSize: 16, global: 1.0, preferNativeBf16: true,
                on: cmd, into: outUp)
            let inner = Ops.swiglu(gate: outGate, up: outUp, on: cmd, into: outGate)
            Ops.nvfp4MoeGatherBm16(
                input: inner, weight: w.dw, scales: w.ds, indices: indices,
                mTotal: mTotal, nOut: w.hidden, kIn: w.moeIntermediate,
                blockSize: 16, global: 1.0, preferNativeBf16: true,
                on: cmd, into: outDown)
            cmd.commit(); cmd.waitUntilCompleted()

            let dflt = toF32Bf16(outDown)
            let dfltNan = dflt.filter(\.isNaN).count
            let dfltInf = dflt.filter(\.isInfinite).count

            let eidx = gemvOracle(x: x, idxBuf: idxBuf, w: w, mTotal: mTotal, device: device)
            let eidxNan = eidx.filter(\.isNaN).count
            let eidxInf = eidx.filter(\.isInfinite).count

            let cos = cosine(dflt, eidx)

            print(
                "[bm16-default-sweep] mag=\(magnitude) default nan=\(dfltNan) inf=\(dfltInf) | "
                    + "eidx nan=\(eidxNan) inf=\(eidxInf) | cos(finite)=\(cos)")

            // T23 permanent regression gate for the DEFAULT dispatch path.
            // This is the codepath most prefill traffic actually runs
            // (NAX/NAX64 above are both opt-in via env var) -- it must
            // carry the same range-safe half-staging fold as the NAX64
            // kernels, since `make regenerate-kernels` rewrites the whole
            // iron-emitted kernel family together.
            #expect(dfltNan == 0, "default bm16_bk32 MPP produced NaN at activation magnitude \(magnitude)")
            #expect(dfltInf == 0, "default bm16_bk32 MPP produced Inf at activation magnitude \(magnitude)")
            #expect(eidxNan == 0, "f32 GEMV oracle produced NaN at activation magnitude \(magnitude) -- ramp exceeds oracle's own safe range")
            #expect(eidxInf == 0, "f32 GEMV oracle produced Inf at activation magnitude \(magnitude) -- ramp exceeds oracle's own safe range")
            #expect(cos > 0.9999, "default bm16_bk32 MPP vs GEMV oracle cos=\(cos) <= 0.9999 at activation magnitude \(magnitude)")
        }
    }
}
