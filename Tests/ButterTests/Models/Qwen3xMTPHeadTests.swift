// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// Qwen3xMTPHeadTests - spec 030 phase 1 coverage for the MTP draft
// head (`Sources/ButterSwift/Models/Text/Qwen3xMTPHead.swift`).
//
// Three tiers, all gated on local checkpoint artifacts existing:
//
//   (a) sidecar-only load + shape audit (needs just `mtp-sidecar/`,
//       dummy shared embedding/lm_head references),
//   (b) fc-fusion arithmetic vs a CPU reference (per-element RMSNorm
//       check + int4-dequant dot products on sampled fc rows),
//   (c) end-to-end draftStep on the real 27B backbone + sidecar:
//       finite draft logits, determinism across a cache reset, and the
//       backbone `forwardManyHiddenAndLogits` hidden-state feed.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import ButterSwift

private let mtpBackbonePath = "/Users/tom/models/qwen36-27b-4bit-mlx"
private let mtpSidecarPath = "\(mtpBackbonePath)/mtp-sidecar"
private let mtpSidecarAvailable =
    FileManager.default.fileExists(atPath: mtpSidecarPath)
private let mtpBackboneAvailable =
    FileManager.default.fileExists(
        atPath: "\(mtpBackbonePath)/model.safetensors.index.json")

// Geometry of the Qwen3.6-27B checkpoint the sidecar belongs to.
private let mtpHiddenDim = 5120
private let mtpVocab = 248_320
private let mtpNHeads = 24
private let mtpNKVHeads = 4
private let mtpHeadDim = 256
private let mtpRotaryDim = 64  // headDim * partial_rotary_factor 0.25
private let mtpRopeTheta: Float = 10_000_000
private let mtpEps: Float = 1e-6

@Suite("Qwen3x MTP cache capacity")
struct Qwen3xMTPCacheCapacityTests {
    @Test("uses configured model context without an override")
    func usesConfiguredModelContext() {
        #expect(
            Qwen3xMTPHead.resolvedCacheCapacity(
                configuredMaxSeq: 262_144, override: nil) == 262_144)
    }

    @Test("valid override takes precedence")
    func validOverrideTakesPrecedence() {
        #expect(
            Qwen3xMTPHead.resolvedCacheCapacity(
                configuredMaxSeq: 262_144, override: "65536") == 65_536)
    }

    @Test("invalid override falls back to configured model context")
    func invalidOverrideFallsBack() {
        #expect(
            Qwen3xMTPHead.resolvedCacheCapacity(
                configuredMaxSeq: 262_144, override: "invalid") == 262_144)
    }
}

// ─── Host-side bf16 helpers ──────────────────────────────────────────

/// Round-to-nearest-even Float -> bf16 bit pattern (matches the Metal
/// `bfloat(...)` cast and the loader's host conversion).
private func bf16Bits(_ v: Float) -> UInt16 {
    let bits = v.bitPattern
    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
    return UInt16(rounded >> 16)
}

private func bf16Round(_ v: Float) -> Float {
    Float(bitPattern: UInt32(bf16Bits(v)) << 16)
}

/// Deterministic pseudo-random vector in [-1, 1) (LCG; fixed seed per
/// call site) already rounded through bf16 so host + GPU see the exact
/// same values.
private func deterministicBF16Vector(count: Int, seed: UInt64) -> [Float] {
    var state = seed
    var out = [Float](repeating: 0, count: count)
    for i in 0 ..< count {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let u = Float(state >> 40) / Float(1 << 24)  // [0, 1)
        out[i] = bf16Round(u * 2 - 1)
    }
    return out
}

private func makeBF16Tensor(_ values: [Float], device: Device) -> Tensor {
    let t = Tensor.empty(shape: [values.count], dtype: .bf16, device: device)
    t.copyIn(from: values.map(bf16Bits))
    return t
}

/// CPU mirror of the iron RMSNorm kernel: fp32 sum of squares over the
/// bf16 inputs, `x * rsqrt(mean + eps) * w`, bf16-rounded output.
private func cpuRMSNormBF16(_ x: [Float], weight: [Float], eps: Float) -> [Float] {
    var ssq: Float = 0
    for v in x { ssq += v * v }
    let inv = 1.0 / (ssq / Float(x.count) + eps).squareRoot()
    return (0 ..< x.count).map { bf16Round(x[$0] * inv * weight[$0]) }
}

private func argmax(_ v: [Float]) -> Int {
    var best = 0
    for i in 1 ..< v.count where v[i] > v[best] { best = i }
    return best
}

/// Build a head from the sidecar with tiny DUMMY shared embedding +
/// lm_head references - enough for the load/shape and fc-fusion tests
/// without pulling the 27B backbone into memory.
private func loadHeadWithDummySharedModules(device: Device) throws -> Qwen3xMTPHead {
    let embedTable = Tensor.empty(
        shape: [4, mtpHiddenDim], dtype: .bf16, device: device)
    embedTable.zero()
    let lmWeight = Tensor.empty(
        shape: [4, mtpHiddenDim], dtype: .bf16, device: device)
    lmWeight.zero()
    return try Qwen3xMTPHead.load(
        sidecarDirectory: URL(fileURLWithPath: mtpSidecarPath),
        embedTokens: AnyEmbedding(Embedding(weight: embedTable)),
        lmHead: AnyLinear(Linear(weight: lmWeight)),
        hidden: mtpHiddenDim, vocab: mtpVocab,
        nHeads: mtpNHeads, nKVHeads: mtpNKVHeads, headDim: mtpHeadDim,
        rotaryDim: mtpRotaryDim, ropeTheta: mtpRopeTheta,
        attnOutputGate: true,
        eps: mtpEps, dtype: .bf16, device: device)
}

// ─── (a) + (b): sidecar-only tests ───────────────────────────────────

@Suite(
    "Qwen3x MTP Head (sidecar)", .serialized,
    .enabled(
        if: mtpSidecarAvailable,
        "requires the MTP sidecar at \(mtpSidecarPath)")
)
struct Qwen3xMTPHeadSidecarTests {

    @Test("sidecar loads with correct shapes and quantization")
    func sidecarLoadShapes() throws {
        let device = Device.shared
        let head = try loadHeadWithDummySharedModules(device: device)

        // fc: [hidden, 2*hidden] int4 g64 (packed u32 cols = 2*hidden/8).
        let fcQ = try #require(head.fc.inner as? QuantizedLinear)
        #expect(fcQ.weight.shape == [mtpHiddenDim, 2 * mtpHiddenDim / 8])
        #expect(fcQ.scales.shape == [mtpHiddenDim, 2 * mtpHiddenDim / 64])
        #expect(fcQ.bits == 4)
        #expect(fcQ.groupSize == 64)

        // Attention projections through the wrapped layer's mixer.
        let mixer = head.layer.mixer
        #expect(mixer.nHeads == mtpNHeads)
        #expect(mixer.nKVHeads == mtpNKVHeads)
        #expect(mixer.headDim == mtpHeadDim)
        #expect(mixer.rotaryDim == mtpRotaryDim)
        #expect(mixer.attnOutputGate)
        #expect(!mixer.useTwoPassDecode)
        let qProj = try #require(mixer.qProj.inner as? QuantizedLinear)
        #expect(qProj.weight.shape[0] == 2 * mtpNHeads * mtpHeadDim)  // 12288
        let kProj = try #require(mixer.kProj.inner as? QuantizedLinear)
        #expect(kProj.weight.shape[0] == mtpNKVHeads * mtpHeadDim)  // 1024

        // Norms (folded +1 at load; the raw sidecar tensors are
        // centered): 5120-wide outer norms, 256-wide per-head q/k norms.
        #expect(head.preFcNormHidden.weight.elementCount == mtpHiddenDim)
        #expect(head.preFcNormEmbedding.weight.elementCount == mtpHiddenDim)
        #expect(head.mtpNorm.weight.elementCount == mtpHiddenDim)
        #expect(head.layer.mixer.qNorm.weight.elementCount == mtpHeadDim)

        // Own KV cache: 4 KV heads x 256 dims, unbounded eviction.
        #expect(head.kvCache.nKVHeads == mtpNKVHeads)
        #expect(head.kvCache.headDim == mtpHeadDim)
        #expect(head.kvCache.length == 0)

        // 15 sidecar tensors -> 31 parameter entries (3 per quantized
        // linear x 8 linears, 1 per norm x 7).
        #expect(head.parameters().count == 31)
    }

    @Test("fc-fusion matches a CPU reference (norms + int4 dequant dot)")
    func fcFusionCPUReference() async throws {
        let device = Device.shared
        let head = try loadHeadWithDummySharedModules(device: device)

        let hHost = deterministicBF16Vector(count: mtpHiddenDim, seed: 0x1234_5678)
        let eHost = deterministicBF16Vector(count: mtpHiddenDim, seed: 0x9ABC_DEF0)
        let hTensor = makeBF16Tensor(hHost, device: device)
        let eTensor = makeBF16Tensor(eHost, device: device)

        let cmd = device.makeCommandBuffer()
        let fused = head.fuseInputs(
            prevHidden: hTensor, embedding: eTensor, on: cmd, device: device)
        cmd.commit()
        await cmd.awaitCompletion()
        let gpuOut = fused.toFloatArray()
        #expect(gpuOut.count == mtpHiddenDim)

        // CPU reference: the two pre-fc norms, bf16-rounded, then the
        // EMBEDDING-first concat. The order is pinned to the reference
        // speculative-decode implementation of this head (fc input =
        // concat(normEmbedding(embed), normHidden(hidden)) - embedding
        // in columns 0..hidden-1, hidden state in hidden..2*hidden-1);
        // a swapped concat drafts silent garbage, so this test fails
        // loudly if fuseInputs ever flips it.
        let wH = head.preFcNormHidden.weight.toFloatArray()
        let wE = head.preFcNormEmbedding.weight.toFloatArray()
        let normH = cpuRMSNormBF16(hHost, weight: wH, eps: mtpEps)
        let normE = cpuRMSNormBF16(eHost, weight: wE, eps: mtpEps)
        let concat = normE + normH

        // Per-element norm check against the GPU concat halves is
        // implicit in the fc dot check below; also sanity-check the
        // norm math itself is not degenerate.
        #expect(normH.contains { $0 != 0 })

        // Int4 affine dequant dot on sampled fc rows:
        //   w[o][i] = ((packed[o][i/8] >> 4*(i%8)) & 0xF) * scale[o][i/64]
        //             + bias[o][i/64]
        let fcQ = try #require(head.fc.inner as? QuantizedLinear)
        let inDim = 2 * mtpHiddenDim
        let packsPerRow = inDim / 8
        let groupsPerRow = inDim / 64
        let packed = fcQ.weight.buffer.contents()
            .advanced(by: fcQ.weight.offset)
            .bindMemory(to: UInt32.self, capacity: mtpHiddenDim * packsPerRow)
        let scales = fcQ.scales.toFloatArray()
        let biases = fcQ.biases.toFloatArray()

        for o in [0, 1, 137, 2048, mtpHiddenDim - 1] {
            var acc: Float = 0
            for i in 0 ..< inDim {
                let pack = packed[o * packsPerRow + i / 8]
                let nib = Float((pack >> (UInt32(i % 8) * 4)) & 0xF)
                let g = o * groupsPerRow + i / 64
                acc += concat[i] * (nib * scales[g] + biases[g])
            }
            let got = gpuOut[o]
            let tol = max(0.05, abs(acc) * 0.02)
            #expect(
                abs(got - acc) <= tol,
                "fc row \(o): gpu=\(got) cpu=\(acc) tol=\(tol)")
        }
    }
}

// ─── (c): end-to-end on the real backbone ────────────────────────────

@Suite(
    "Qwen3x MTP Head (27B end-to-end)", .serialized,
    .enabled(
        if: mtpSidecarAvailable && mtpBackboneAvailable,
        "requires the 27B backbone + sidecar under \(mtpBackbonePath)")
)
struct Qwen3xMTPHeadEndToEndTests {

    @Test("draftStep on the real checkpoint: finite, deterministic draft logits")
    func draftStepRealCheckpoint() async throws {
        let device = Device.shared
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(mtpBackbonePath, options: opts)
        }
        let q = try #require(m.qwen35, "expected Qwen35Model engine")
        let head = try #require(
            q.mtpHead, "expected the mtp-sidecar to attach an MTP head")
        #expect(head.hidden == q.hidden)
        #expect(head.vocab == q.vocab)

        // Backbone hidden states for a fixed short prompt.
        let tokenIds = m.tokenizer.encode(text: "The capital of France is")
        let t = tokenIds.count
        let caches = q.makeLayerCaches(maxSeq: 256, device: device)
        let cmd = device.makeCommandBuffer()
        let (hiddenAll, logitsAll) = q.forwardManyHiddenAndLogits(
            tokenIds: tokenIds, startPosition: 0,
            caches: caches, on: cmd, device: device)
        cmd.commit()
        await cmd.awaitCompletion()

        #expect(hiddenAll.shape == [t, q.hidden])
        #expect(logitsAll.elementCount == t * q.vocab)
        let hiddenHost = hiddenAll.toFloatArray()
        #expect(hiddenHost.allSatisfy { $0.isFinite }, "backbone hidden must be finite")
        let backboneLogits = logitsAll.toFloatArray()
        #expect(backboneLogits.allSatisfy { $0.isFinite }, "backbone logits must be finite")

        // Last-position hidden row feeds the draft head, paired with
        // the token the backbone greedily samples for position t (the
        // MTP pairing: hidden state at position i + token at i+1 draft
        // the token at i+2).
        let dtBytes = hiddenAll.dtype.byteSize
        let prevHidden = Tensor(
            buffer: hiddenAll.buffer,
            offset: hiddenAll.offset + (t - 1) * q.hidden * dtBytes,
            shape: [q.hidden], dtype: hiddenAll.dtype)
        let backboneTop = argmax(Array(backboneLogits[(t - 1) * q.vocab ..< t * q.vocab]))

        // Two identical draftSteps with a cache reset between them.
        var draftTops: [Int] = []
        var draftLogitsRuns: [[Float]] = []
        for _ in 0 ..< 2 {
            head.resetCache()
            let stepCmd = device.makeCommandBuffer()
            let (draftHidden, draftLogits) = head.draftStep(
                prevHidden: prevHidden, prevToken: backboneTop, position: t,
                on: stepCmd, device: device)
            stepCmd.commit()
            await stepCmd.awaitCompletion()

            #expect(draftHidden.elementCount == q.hidden)
            #expect(head.kvCache.length == 1)
            let logitsHost = draftLogits.toFloatArray()
            #expect(logitsHost.count == q.vocab)
            #expect(logitsHost.allSatisfy { $0.isFinite }, "draft logits must be finite")
            draftTops.append(argmax(logitsHost))
            draftLogitsRuns.append(logitsHost)
        }

        #expect(
            draftTops[0] == draftTops[1],
            "draft top-1 must be deterministic across a cache reset")
        var maxAbsDiff: Float = 0
        for i in 0 ..< draftLogitsRuns[0].count {
            maxAbsDiff = max(maxAbsDiff, abs(draftLogitsRuns[0][i] - draftLogitsRuns[1][i]))
        }
        #expect(maxAbsDiff < 1e-3, "draft logits drifted: maxAbsDiff=\(maxAbsDiff)")

        // Smoke output for the mission report.
        print("MTP SMOKE prompt=\(tokenIds)")
        print(
            "MTP SMOKE backbone top-1 @last: \(backboneTop) "
                + "(\(m.tokenizer.decode(tokens: [backboneTop])))")
        print(
            "MTP SMOKE draft top-1 run1=\(draftTops[0]) run2=\(draftTops[1]) "
                + "(\(m.tokenizer.decode(tokens: [draftTops[0]]))) "
                + "maxAbsDiff=\(maxAbsDiff)")
    }
}
