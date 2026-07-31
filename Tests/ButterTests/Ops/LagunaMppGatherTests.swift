// Copyright 2026 Tom Turney (@TheTom)
//
// w4/mpp-moegather correctness gate for `LagunaNvfp4MppGather` -- the fused
// NVFP4-dequant + MPP tensor-ops gather-GEMM with EG-style (true
// 1-expert/threadgroup) ownership. Mirrors the pattern the sibling NAX/
// NAX64 suites already established (Nvfp4GatherNax64Tests.swift,
// Nvfp4GatherNax64MagnitudeTests.swift): expert-indexed GEMV oracle via
// `Ops.nvfp4DequantGemvExpertIndexedMany`, real Laguna XS checkpoint
// weights where available (tests skip cleanly if the local checkpoint
// isn't present), synthetic weights for the shape/geometry-only cases.
//
// This suite specifically targets the failure classes prior sessions
// documented for the adjacent problem (see `Nvfp4GatherEG.swift`'s header
// and `LagunaNvfp4MppGather.swift`'s own header): tile-boundary straddling,
// zero-count experts, ragged (non-64-multiple) expert run lengths, and the
// T23 half-staging overflow at real SwiGLU-amplified activation magnitude.
// Every GPU-touching test is guarded on `Ops.mppAutoCapable` (skip, not
// fail, on incapable hardware -- e.g. CI paravirt GPUs below Apple GPU
// Family 9 or pre-macOS-26).

import Foundation
import Metal
import Testing
@testable import ButterSwift
import IronSwift

@Suite("LagunaNvfp4MppGather (w4/mpp-moegather EG-style fused NVFP4 MPP gather-GEMM)", .serialized)
struct LagunaMppGatherTests {
    private var mppAvailable: Bool { Ops.mppAutoCapable }

    // MARK: - Shared helpers (mirrors Nvfp4GatherNax64Tests.swift's own)

    private func toF32(_ t: Tensor) -> [Float] {
        let n = t.elementCount
        let ptr = t.buffer.contents().advanced(by: t.offset)
            .bindMemory(to: UInt16.self, capacity: n)
        return (0 ..< n).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> (cos: Float, nan: Int) {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        var nanCount = 0
        for i in 0 ..< a.count {
            if a[i].isNaN { nanCount += 1; continue }
            guard b[i].isFinite else { continue }
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        return (dot / (sqrt(na) * sqrt(nb) + 1e-12), nanCount)
    }

    /// Oracle: one expert-indexed GEMV per row (same helper pattern as
    /// Nvfp4GatherNax64Tests.swift's `eidxOracle`).
    private func eidxOracle(
        x: Tensor, weight: Tensor, scales: Tensor, idxBuf: MTLBuffer,
        mTotal: Int, kIn: Int, nOut: Int, device: Device, cmd: MTLCommandBuffer
    ) -> Tensor {
        let outEidx = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outEidx.buffer.contents().advanced(by: outEidx.offset), 0, mTotal * nOut * 2)
        var ws: [Tensor] = [], ss: [Tensor] = [], ins: [Tensor] = []
        var idxs: [Tensor] = [], outs: [Tensor] = []
        for r in 0 ..< mTotal {
            ws.append(weight); ss.append(scales)
            ins.append(x.slicedRows(start: r, count: 1).reshaped(to: [kIn]))
            idxs.append(Tensor(buffer: idxBuf, offset: r * 4, shape: [1], dtype: .u32))
            outs.append(outEidx.slicedRows(start: r, count: 1).reshaped(to: [nOut]))
        }
        Ops.nvfp4DequantGemvExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss,
            inputs: ins, expertIndices: idxs, outputs: outs,
            blockSize: 16, global: 1.0, on: cmd)
        return outEidx
    }

    private func loadModelDir() -> URL? {
        let modelDir = URL(fileURLWithPath: NSString(
            string: "~/models/laguna-xs-2.1-nvfp4").expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: modelDir.path) else { return nil }
        return modelDir
    }

    // MARK: - Eligibility / opt-in gating (no GPU dispatch needed)

    @Test("eligible() respects BUTTER_NVFP4_MPP_GATHER opt-in + shape/dtype gates")
    func eligibilityGating() throws {
        // Shape-only gates, independent of the env flag.
        #expect(!LagunaNvfp4MppGather.shapeEligible(dtype: .f32, nOut: 2048, kIn: 2048, blockSize: 16),
            "f32 must decline -- kernel is bf16-only")
        #expect(!LagunaNvfp4MppGather.shapeEligible(dtype: .bf16, nOut: 2000, kIn: 2048, blockSize: 16),
            "nOut not %64==0 must decline")
        #expect(!LagunaNvfp4MppGather.shapeEligible(dtype: .bf16, nOut: 2048, kIn: 2000, blockSize: 16),
            "kIn not %32==0 must decline")
        #expect(!LagunaNvfp4MppGather.shapeEligible(dtype: .bf16, nOut: 2048, kIn: 2048, blockSize: 32),
            "blockSize != 16 must decline (only Poolside Laguna's group-16 NVFP4 layout is supported)")
        #expect(LagunaNvfp4MppGather.shapeEligible(dtype: .bf16, nOut: 2048, kIn: 2048, blockSize: 16),
            "aligned bf16 shape must be shape-eligible")

        // T44 re-validation flip: default ON, escape-to-disable
        // (`BUTTER_NVFP4_MPP_GATHER=0`), matching `LagunaMppGemm.enabled`'s
        // convention -- see `LagunaNvfp4MppGather.enabled`'s header for the
        // correctness/coherence/A-B bar this cleared. Unless the env var is
        // explicitly set to "0", `enabled` (and therefore `eligible`, for an
        // otherwise-eligible shape) must be true.
        if ProcessInfo.processInfo.environment["BUTTER_NVFP4_MPP_GATHER"] == nil {
            #expect(LagunaNvfp4MppGather.enabled, "must default ON")
            if mppAvailable {
                #expect(
                    LagunaNvfp4MppGather.eligible(dtype: .bf16, nOut: 2048, kIn: 2048, blockSize: 16),
                    "must accept by default for an eligible shape on MPP-capable hardware")
            }
        }
    }

    @Test("maxTiles() bound holds across a range of (mTotal, nExperts)")
    func maxTilesBoundIsSafe() throws {
        // sum_e ceil(count_e / 64) <= ceil(mTotal/64) + nExperts (see
        // LagunaNvfp4MppGather.swift's "MAX-TILES BOUND" header). Check the
        // bound directly against a worst-case run-length distribution
        // (every expert gets exactly 1 row, maximizing wasted partial
        // tiles) for a handful of (mTotal, nExperts) pairs.
        for (mTotal, nExperts) in [(64, 4), (100, 16), (4096, 256), (80, 256), (1, 1)] {
            let used = min(mTotal, nExperts)
            var worstCaseTiles = 0
            var remaining = mTotal
            for e in 0 ..< used {
                let count = (e == used - 1) ? remaining : 1
                remaining -= count
                worstCaseTiles += (count + 63) / 64
            }
            let bound = LagunaNvfp4MppGather.maxTiles(mTotal: mTotal, nExperts: nExperts)
            #expect(bound >= worstCaseTiles,
                "maxTiles(\(mTotal),\(nExperts))=\(bound) must cover worst-case tile count \(worstCaseTiles)")
        }
    }

    // MARK: - Synthetic same-expert sanity (fast, no checkpoint needed)

    @Test("EG-MPP synthetic same-expert matches eidx oracle", .enabled(if: true))
    func synthSameExpertMatchesEidx() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        let device = Device.shared
        let mTotal = 16
        let nOut = 512
        let kIn = 2048
        let nExperts = 4
        let blockSize = 16
        let packs = kIn / 8
        let groups = kIn / blockSize
        var wHost = [UInt32](repeating: 0, count: nExperts * nOut * packs)
        for e in 0 ..< nExperts {
            for r in 0 ..< nOut {
                for p in 0 ..< packs {
                    wHost[(e * nOut + r) * packs + p] = UInt32(0x1111_1111 &+ UInt32(e))
                }
            }
        }
        let sHost = [UInt8](repeating: 0x48, count: nExperts * nOut * groups)
        let wBuf = device.makeBuffer(length: wHost.count * 4)
        let sBuf = device.makeBuffer(length: sHost.count)
        wHost.withUnsafeBytes { _ = memcpy(wBuf.contents(), $0.baseAddress!, wHost.count * 4) }
        sHost.withUnsafeBytes { _ = memcpy(sBuf.contents(), $0.baseAddress!, sHost.count) }
        let weight = Tensor(buffer: wBuf, offset: 0, shape: [nExperts, nOut, packs], dtype: .u32)
        let scales = Tensor(buffer: sBuf, offset: 0, shape: [nExperts, nOut, groups], dtype: .u8)

        let xHost = [UInt16](repeating: 0x3F80, count: mTotal * kIn)
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        // All rows expert 0 -- exercises a single tile with a single
        // ragged (mTotal=16 < bm=64) run.
        let idxHost = [UInt32](repeating: 0, count: mTotal)
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let outEg = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outEg.buffer.contents().advanced(by: outEg.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        LagunaNvfp4MppGather.callGather(
            input: x, weight: weight, scales: scales, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: blockSize, global: 1.0, on: cmd, device: device, into: outEg)
        let outEidx = eidxOracle(
            x: x, weight: weight, scales: scales, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        let eg = toF32(outEg)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(eg, eidx)
        print("[eg-mpp-synth] nan=\(nan) cos=\(cos) sample eg=\(eg.prefix(4)) eidx=\(eidx.prefix(4))")
        #expect(nan == 0, "EG-MPP produced \(nan) NaNs")
        #expect(cos > 0.9999, "EG-MPP vs eidx cos \(cos)")
    }

    // MARK: - Tile-straddle boundary + zero-count experts (real Laguna gate weights)

    /// Same run-length construction as
    /// `Nvfp4GatherNax64Tests.nax64RealGateBoundaryStraddle`: runs crafted
    /// to straddle 64-row tile boundaries under the OLD flat-M-tile
    /// schemes, plus experts that never appear at all. For THIS kernel
    /// "straddle" is a non-event by construction (the tile plan gives every
    /// run its own tile boundary regardless of where it falls in the flat
    /// row order) -- this test's job is to prove that equivalence holds,
    /// not to find a straddle bug the EG design already precludes.
    @Test("EG-MPP real Laguna gate: former-straddle runs + zero-count experts", .enabled(if: true))
    func realGateFormerStraddleRuns() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        guard let modelDir = loadModelDir() else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        let nOut = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 32 == 0, "gate kIn \(kIn) must be %32==0 for EG-MPP")
        try #require(nOut % 64 == 0)

        // Same runs as the NAX64 sibling test: expert 2 spans [50,80)
        // (straddles the old tile0|tile1 seam @64), expert 9 spans
        // [120,150) (straddles tile1|tile2 @128), expert 20 spans
        // [180,210) (straddles tile2|tile3 @192); experts 5, 15 never
        // appear (zero-count).
        struct Run { let expert: UInt32; let start: Int; let end: Int }
        let runs = [
            Run(expert: 0, start: 0, end: 50),
            Run(expert: 2, start: 50, end: 80),
            Run(expert: 7, start: 80, end: 120),
            Run(expert: 9, start: 120, end: 150),
            Run(expert: 12, start: 150, end: 180),
            Run(expert: 20, start: 180, end: 210),
            Run(expert: 30, start: 210, end: 224),
        ]
        let mTotal = runs.last!.end
        let nExperts = gw.shape[0]
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for run in runs {
            for r in run.start ..< run.end { idxHost[r] = run.expert }
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            let v: Float = 0.5 + Float(r % 7) * 0.1
            let bf16 = UInt16(v.bitPattern >> 16)
            for k in 0 ..< kIn { xHost[r * kIn + k] = bf16 }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outEg = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outEg.buffer.contents().advanced(by: outEg.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        LagunaNvfp4MppGather.callGather(
            input: x, weight: gw, scales: gs, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, device: device, into: outEg)
        _ = nExperts
        let outEidx = eidxOracle(
            x: x, weight: gw, scales: gs, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)

        let eg = toF32(outEg)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(eg, eidx)
        print("[eg-mpp-gate] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "EG-MPP real gate produced \(nan) NaNs")
        #expect(cos > 0.9999, "EG-MPP real gate former-straddle cos \(cos)")

        // Per-row check at every former-seam row, same spirit as the NAX64
        // sibling test -- localizes a tile-plan-boundary bug rather than
        // only seeing it diluted in the aggregate cos.
        for boundary in [64, 128, 192] {
            for r in [boundary - 1, boundary] {
                guard r < mTotal else { continue }
                let egRow = Array(eg[(r * nOut) ..< ((r + 1) * nOut)])
                let eidxRow = Array(eidx[(r * nOut) ..< ((r + 1) * nOut)])
                let (rowCos, rowNan) = cosine(egRow, eidxRow)
                print("[eg-mpp-gate] row \(r) (expert \(idxHost[r])) cos=\(rowCos) nan=\(rowNan)")
                #expect(rowNan == 0, "row \(r) NaN")
                #expect(rowCos > 0.999, "row \(r) at former tile seam cos \(rowCos)")
            }
        }
    }

    // MARK: - Ragged segments (real Laguna down proj, non-64-multiple runs)

    /// Real Laguna down proj (kIn=moeIntermediate, nOut=hidden): 3 experts,
    /// 32 rows each -- every one of the resulting 3 tiles is RAGGED
    /// (row_count=32 < bm=64), the exact case the tile plan's
    /// `min(bm, end - row_start)` masking exists for.
    @Test("EG-MPP real Laguna down proj: ragged (32-row) segments", .enabled(if: true))
    func realDownProjRaggedSegments() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        guard let modelDir = loadModelDir() else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let dw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.down_proj.weight")
        let ds = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.down_proj.scales")
        let nOut = dw.shape[1]
        let packs = dw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 32 == 0, "down kIn \(kIn) must be %32==0 for EG-MPP")
        try #require(nOut % 64 == 0)

        let mTotal = 96
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        for i in 0 ..< mTotal { idxHost[i] = UInt32(i / 32) }  // experts 0,1,2, 32 rows each -- all ragged
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            let v: Float = 0.3 + Float(r % 5) * 0.15
            let bf16 = UInt16(v.bitPattern >> 16)
            for k in 0 ..< kIn { xHost[r * kIn + k] = bf16 }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let outEg = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outEg.buffer.contents().advanced(by: outEg.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        LagunaNvfp4MppGather.callGather(
            input: x, weight: dw, scales: ds, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, device: device, into: outEg)
        let outEidx = eidxOracle(
            x: x, weight: dw, scales: ds, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let eg = toF32(outEg)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(eg, eidx)
        print("[eg-mpp-down] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "EG-MPP down proj (ragged) produced \(nan) NaNs")
        #expect(cos > 0.9999, "EG-MPP down proj (ragged) cos \(cos)")
    }

    // MARK: - Zero-count experts at production scale (real moeSortPlan output)

    /// Full sort-plan-shaped integration: real `moeSortPlan` output with
    /// mTotal=80 « nExperts=256, so the overwhelming majority of experts
    /// are zero-count -- exactly the case `Nvfp4GatherEG.expertOffsets`'s
    /// binary search and the tileplan prepass's per-expert loop both have
    /// to skip over cheaply and correctly.
    @Test("EG-MPP after moeSortPlan+gather (256 experts, mostly zero-count)", .enabled(if: true))
    func afterSortPlanGatherManyZeroCountExperts() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        guard let modelDir = loadModelDir() else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        let mid = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 32 == 0)
        let t = 10
        let topK = 8
        let mTotal = t * topK  // 80
        let nExperts = 256

        var hHost = [UInt16](repeating: 0, count: t * kIn)
        for r in 0 ..< t {
            let v: Float = 0.4 + Float(r) * 0.05
            let bf16 = UInt16(v.bitPattern >> 16)
            for k in 0 ..< kIn { hHost[r * kIn + k] = bf16 }
        }
        let hBuf = device.makeBuffer(length: hHost.count * 2)
        hHost.withUnsafeBytes { _ = memcpy(hBuf.contents(), $0.baseAddress!, hHost.count * 2) }
        let hRows = Tensor(buffer: hBuf, offset: 0, shape: [t, kIn], dtype: .bf16)

        var topkHost = [UInt32](repeating: 0, count: mTotal)
        var seed: UInt32 = 13
        for r in 0 ..< t {
            var used = Set<UInt32>()
            var s = 0
            while s < topK {
                seed = seed &* 1_664_525 &+ 1_013_904_223
                let e = seed % UInt32(nExperts)
                if used.insert(e).inserted {
                    topkHost[r * topK + s] = e
                    s += 1
                }
            }
        }
        let topkBuf = device.makeBuffer(length: mTotal * 4)
        topkHost.withUnsafeBytes { _ = memcpy(topkBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let topkIds = Tensor(buffer: topkBuf, offset: 0, shape: [mTotal], dtype: .u32)

        let se = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        let st = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)
        let ip = Tensor.empty(shape: [mTotal], dtype: .u32, device: device)

        let cmd = device.makeCommandBuffer()
        Ops.moeSortPlan(
            topkIds: topkIds, sortedExperts: se, sourceTokens: st, invPerm: ip,
            mTotal: mTotal, k: topK, on: cmd)
        let xGathered = Ops.gather(table: hRows, tokenIds: st, on: cmd)

        let outEg = Tensor.empty(shape: [mTotal, mid], dtype: .bf16, device: device)
        memset(outEg.buffer.contents().advanced(by: outEg.offset), 0, mTotal * mid * 2)
        LagunaNvfp4MppGather.callGather(
            input: xGathered, weight: gw, scales: gs, indices: se,
            mTotal: mTotal, nOut: mid, kIn: kIn,
            blockSize: 16, global: 1.0, on: cmd, device: device, into: outEg)

        let outEidx = eidxOracle(
            x: xGathered, weight: gw, scales: gs, idxBuf: se.buffer,
            mTotal: mTotal, kIn: kIn, nOut: mid, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let seHost = se.toArray(as: UInt32.self)
        let uniqueExperts = Set(seHost).count
        print("[eg-mpp-sort] unique experts used = \(uniqueExperts)/\(nExperts) (rest zero-count)")
        try #require(uniqueExperts < nExperts, "test setup should exercise real zero-count experts")

        let eg = toF32(outEg)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(eg, eidx)
        print("[eg-mpp-sort] mTotal=\(mTotal) nan=\(nan) cos=\(cos)")
        #expect(nan == 0)
        #expect(cos > 0.9999, "EG-MPP after sort+gather cos \(cos)")
    }

    // MARK: - T23 magnitude fold (layer-38 / expert-161, the T22/T23 repro row)

    /// Mirrors `Nvfp4GatherNax64MagnitudeTests.sweepLayer38Expert161`: real
    /// layer-38 expert-161 gate/up/down weights, activation magnitude
    /// ramped 1...800 (16x past the T22-observed ~50 overflow cliff),
    /// checked for zero NaN/Inf and cos>0.9999 against the f32 GEMV oracle
    /// through the identical gate/up/swiglu/down algebra.
    @Test("EG-MPP stays finite through activation magnitude 800 (T23 fold, real layer-38/expert-161)", .enabled(if: true))
    func magnitudeFoldRealLayer38Expert161() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        guard let modelDir = loadModelDir() else { return }
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
        try #require(hidden % 32 == 0 && moeIntermediate % 64 == 0 && hidden % 64 == 0)

        let expert: UInt32 = 161
        let mTotal = 64  // one full tile, all rows the T22/T23 failing expert
        let idxHost = [UInt32](repeating: expert, count: mTotal)
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        for magnitude: Float in [1, 10, 50, 100, 200, 400, 800] {
            var xHost = [UInt16](repeating: 0, count: mTotal * hidden)
            for r in 0 ..< mTotal {
                for k in 0 ..< hidden {
                    let v = magnitude * sin(Float(k) * 0.7 + Float(r) * 0.13)
                    xHost[r * hidden + k] = UInt16(v.bitPattern >> 16)
                }
            }
            let xBuf = device.makeBuffer(length: xHost.count * 2)
            xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
            let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, hidden], dtype: .bf16)

            let outGate = Tensor.empty(shape: [mTotal, moeIntermediate], dtype: .bf16, device: device)
            let outUp = Tensor.empty(shape: [mTotal, moeIntermediate], dtype: .bf16, device: device)
            let outDown = Tensor.empty(shape: [mTotal, hidden], dtype: .bf16, device: device)
            let cmd = device.makeCommandBuffer()
            LagunaNvfp4MppGather.callGatherPair(
                input: x,
                weight0: gw, scales0: gs, out0: outGate,
                weight1: uw, scales1: us, out1: outUp,
                indices: indices,
                mTotal: mTotal, nOut: moeIntermediate, kIn: hidden,
                blockSize: 16, global: 1.0, on: cmd, device: device)
            let inner = Ops.swiglu(gate: outGate, up: outUp, on: cmd, into: outGate)
            LagunaNvfp4MppGather.callGather(
                input: inner, weight: dw, scales: ds, indices: indices,
                mTotal: mTotal, nOut: hidden, kIn: moeIntermediate,
                blockSize: 16, global: 1.0, on: cmd, device: device, into: outDown)
            cmd.commit(); cmd.waitUntilCompleted()

            let eg = toF32(outDown)
            let egNan = eg.filter(\.isNaN).count
            let egInf = eg.filter(\.isInfinite).count

            // f32 expert-indexed GEMV oracle through the identical
            // gate/up/swiglu/down algebra.
            let outGateE = Tensor.empty(shape: [mTotal, moeIntermediate], dtype: .bf16, device: device)
            let outUpE = Tensor.empty(shape: [mTotal, moeIntermediate], dtype: .bf16, device: device)
            let outDownE = Tensor.empty(shape: [mTotal, hidden], dtype: .bf16, device: device)
            let cmd2 = device.makeCommandBuffer()
            var gws: [Tensor] = [], gss: [Tensor] = [], gins: [Tensor] = []
            var gidxs: [Tensor] = [], gouts: [Tensor] = []
            var uws: [Tensor] = [], uss: [Tensor] = [], uins: [Tensor] = []
            var uidxs: [Tensor] = [], uouts: [Tensor] = []
            for r in 0 ..< mTotal {
                let xRow = x.slicedRows(start: r, count: 1).reshaped(to: [hidden])
                let eIdx = Tensor(buffer: idxBuf, offset: r * 4, shape: [1], dtype: .u32)
                gws.append(gw); gss.append(gs); gins.append(xRow); gidxs.append(eIdx)
                gouts.append(outGateE.slicedRows(start: r, count: 1).reshaped(to: [moeIntermediate]))
                uws.append(uw); uss.append(us); uins.append(xRow); uidxs.append(eIdx)
                uouts.append(outUpE.slicedRows(start: r, count: 1).reshaped(to: [moeIntermediate]))
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
                let inRow = innerE.slicedRows(start: r, count: 1).reshaped(to: [moeIntermediate])
                let eIdx = Tensor(buffer: idxBuf, offset: r * 4, shape: [1], dtype: .u32)
                dws.append(dw); dss.append(ds); dins.append(inRow); didxs.append(eIdx)
                douts.append(outDownE.slicedRows(start: r, count: 1).reshaped(to: [hidden]))
            }
            Ops.nvfp4DequantGemvExpertIndexedMany(
                weightsStacked: dws, scalesStacked: dss, inputs: dins,
                expertIndices: didxs, outputs: douts,
                blockSize: 16, global: 1.0, on: cmd2)
            cmd2.commit(); cmd2.waitUntilCompleted()
            let eidx = toF32(outDownE)
            let eidxNan = eidx.filter(\.isNaN).count
            let eidxInf = eidx.filter(\.isInfinite).count

            let (cos, _) = cosine(eg, eidx)
            print("[eg-mpp-magnitude] mag=\(magnitude) eg nan=\(egNan) inf=\(egInf) | eidx nan=\(eidxNan) inf=\(eidxInf) | cos=\(cos)")

            #expect(egNan == 0, "EG-MPP produced NaN at activation magnitude \(magnitude)")
            #expect(egInf == 0, "EG-MPP produced Inf at activation magnitude \(magnitude)")
            #expect(eidxNan == 0, "f32 GEMV oracle produced NaN at magnitude \(magnitude) -- ramp exceeds oracle's own safe range")
            #expect(eidxInf == 0, "f32 GEMV oracle produced Inf at magnitude \(magnitude) -- ramp exceeds oracle's own safe range")
            #expect(cos > 0.9999, "EG-MPP vs GEMV oracle cos=\(cos) <= 0.9999 at activation magnitude \(magnitude)")
        }
    }

    // MARK: - End-to-end wiring: nvfp4MoeGatherBm16 dispatch chain

    /// Confirms the opt-in wiring in `OpsNVFP4.nvfp4MoeGatherBm16` actually
    /// routes to `LagunaNvfp4MppGather` when `BUTTER_NVFP4_MPP_GATHER=1`,
    /// and that the result still agrees with the eidx oracle through the
    /// public dispatch entry point (not just the direct
    /// `LagunaNvfp4MppGather.callGather` call the other tests use).
    @Test("nvfp4MoeGatherBm16 routes through EG-MPP when BUTTER_NVFP4_MPP_GATHER=1", .enabled(if: true))
    func endToEndDispatchWiring() throws {
        guard mppAvailable else { return }  // MPP coop tensors unavailable (e.g. CI paravirt GPU): skip, not fail
        guard let modelDir = loadModelDir() else { return }
        let device = Device.shared
        let weights = try SafeTensorsBundle(directory: modelDir, device: device)
        let gw = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.weight")
        let gs = try weights.tensor(named: "model.layers.1.mlp.switch_mlp.gate_proj.scales")
        let nOut = gw.shape[1]
        let packs = gw.shape[2]
        let kIn = packs * 8
        try #require(kIn % 32 == 0 && nOut % 64 == 0)

        let mTotal = 48  // ragged: below one full 64-row tile
        var idxHost = [UInt32](repeating: 0, count: mTotal)
        var seed: UInt32 = 7
        for i in 0 ..< mTotal {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            idxHost[i] = seed % 256
        }
        let idxBuf = device.makeBuffer(length: mTotal * 4)
        idxHost.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, mTotal * 4) }
        let indices = Tensor(buffer: idxBuf, offset: 0, shape: [mTotal], dtype: .u32)

        var xHost = [UInt16](repeating: 0, count: mTotal * kIn)
        for r in 0 ..< mTotal {
            let v: Float = 0.2 + Float(r % 11) * 0.07
            let bf16 = UInt16(v.bitPattern >> 16)
            for k in 0 ..< kIn { xHost[r * kIn + k] = bf16 }
        }
        let xBuf = device.makeBuffer(length: xHost.count * 2)
        xHost.withUnsafeBytes { _ = memcpy(xBuf.contents(), $0.baseAddress!, xHost.count * 2) }
        let x = Tensor(buffer: xBuf, offset: 0, shape: [mTotal, kIn], dtype: .bf16)

        let previous = ProcessInfo.processInfo.environment["BUTTER_NVFP4_MPP_GATHER"]
        setenv("BUTTER_NVFP4_MPP_GATHER", "1", 1)
        defer {
            if let previous { setenv("BUTTER_NVFP4_MPP_GATHER", previous, 1) }
            else { unsetenv("BUTTER_NVFP4_MPP_GATHER") }
        }
        #expect(LagunaNvfp4MppGather.enabled, "env flag must be observed as ON for this test")

        let outDispatch = Tensor.empty(shape: [mTotal, nOut], dtype: .bf16, device: device)
        memset(outDispatch.buffer.contents().advanced(by: outDispatch.offset), 0, mTotal * nOut * 2)

        let cmd = device.makeCommandBuffer()
        Ops.nvfp4MoeGatherBm16(
            input: x, weight: gw, scales: gs, indices: indices,
            mTotal: mTotal, nOut: nOut, kIn: kIn,
            blockSize: 16, global: 1.0, preferNativeBf16: true,
            on: cmd, into: outDispatch)
        let outEidx = eidxOracle(
            x: x, weight: gw, scales: gs, idxBuf: idxBuf,
            mTotal: mTotal, kIn: kIn, nOut: nOut, device: device, cmd: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let dispatch = toF32(outDispatch)
        let eidx = toF32(outEidx)
        let (cos, nan) = cosine(dispatch, eidx)
        print("[eg-mpp-e2e] mTotal=\(mTotal) nOut=\(nOut) kIn=\(kIn) nan=\(nan) cos=\(cos)")
        #expect(nan == 0, "nvfp4MoeGatherBm16 (EG-MPP routed) produced \(nan) NaNs")
        #expect(cos > 0.9999, "nvfp4MoeGatherBm16 (EG-MPP routed) vs eidx cos \(cos)")
    }
}
