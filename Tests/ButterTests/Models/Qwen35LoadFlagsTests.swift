// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed as an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Qwen 3.5/3.6 hybrid attention honors LoadOptions (AURA, maxSeq, window).

import Foundation
import Testing

@testable import ButterSwift

@Suite("Qwen3.5 load flags")
struct Qwen35LoadFlagsTests {

    @Test("attention caches honor AURA")
    func qwen35AuraCaches() {
        let model = makeTinyQwen35(kv: .auraQuantized(scheme: .default), maxSeq: 64)
        let caches = model.makeLayerCaches(maxSeq: nil, device: .shared)
        let attn = caches.compactMap { $0 as? AURAQuantizedKVCache }
        #expect(!attn.isEmpty)
        #expect(attn.allSatisfy { $0.maxSeq == 64 })
        #expect(caches.contains { $0 is Qwen35GDNLayerCache })
    }

    @Test("AURA decode does not trap on a raw KVCache cast")
    func qwen35AuraDecodeDoesNotTrap() {
        let model = makeTinyQwen35(kv: .auraQuantized(scheme: .default), maxSeq: 64)
        let caches = model.makeLayerCaches(maxSeq: nil, device: .shared)
        let cmd = Device.shared.makeCommandBuffer()
        _ = model.forward(
            tokenId: 1, position: 0, caches: caches,
            on: cmd, device: .shared)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.status == .completed)
    }

    @Test("maxSeq is the context cap")
    func qwen35MaxSeqFromCap() {
        let model = makeTinyQwen35(kv: .raw, maxSeq: 4096)
        #expect(model.maxSeq == 4096)
        let caches = model.makeLayerCaches(maxSeq: nil, device: .shared)
        let kv = caches.compactMap { $0 as? KVCache }
        #expect(kv.allSatisfy { $0.maxSeq == 4096 })
    }

    @Test("window eviction is stored on attention caches")
    func qwen35WindowEviction() {
        let model = makeTinyQwen35(
            kv: .raw, maxSeq: 128,
            eviction: .window(maxSize: 32, keep: 4))
        let caches = model.makeLayerCaches(maxSeq: nil, device: .shared)
        let kv = caches.compactMap { $0 as? KVCache }
        #expect(kv.allSatisfy {
            if case .window(let s, let k) = $0.eviction { s == 32 && k == 4 } else { false }
        })
    }
}

/// 1 GDN + 1 full-attention dense hybrid so `makeLayerCaches` actually
/// builds an attention KV (the all-GDN tiny used by prefill tests never
/// hits `--kv-cache aura`).
private func makeTinyQwen35(
    kv: KVCacheKind,
    maxSeq: Int,
    eviction: KVEviction = .unbounded
) -> Qwen35Model {
    let vocab = 64
    let hidden = 512
    let nHeads = 4
    let nKVHeads = 1
    let headDim = 128
    let dk = 128, dv = 128, hk = 1, hv = 2
    let keyDim = hk * dk
    let valueDim = hv * dv
    let convDim = 2 * keyDim + valueDim
    let convKernel = 4
    let eps: Float = 1e-6
    let invKeyScale = Foundation.pow(Float(dk), -0.5)

    func bf16Bits(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(rounded >> 16)
    }
    func ones(_ shape: [Int]) -> Tensor {
        let n = shape.reduce(1, *)
        let t = Tensor.empty(shape: shape, dtype: .bf16)
        t.copyIn(from: [UInt16](repeating: bf16Bits(1), count: n))
        return t
    }
    func rand(_ shape: [Int], scale: Float) -> Tensor {
        let n = shape.reduce(1, *)
        var state: UInt64 = 0xC0FFEE
        let vals = (0 ..< n).map { _ -> UInt16 in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u = Float(state >> 33) / Float(UInt32.max)
            return bf16Bits((u * 2 - 1) * scale)
        }
        let t = Tensor.empty(shape: shape, dtype: .bf16)
        t.copyIn(from: vals)
        return t
    }

    let gdnMixer = Qwen35GDNMixer(
        inProjQKV: AnyLinear(Linear(weight: rand([convDim, hidden], scale: 0.05))),
        inProjZ: AnyLinear(Linear(weight: rand([valueDim, hidden], scale: 0.05))),
        inProjB: AnyLinear(Linear(weight: rand([hv, hidden], scale: 0.05))),
        inProjA: AnyLinear(Linear(weight: rand([hv, hidden], scale: 0.05))),
        outProj: AnyLinear(Linear(weight: rand([hidden, valueDim], scale: 0.05))),
        convW: rand([convKernel, convDim], scale: 0.2),
        convB: {
            let b = Tensor.empty(shape: [convDim], dtype: .bf16)
            b.zero()
            return b
        }(),
        mixerNorm: RMSNorm(weight: ones([dv]), eps: eps),
        aLog: [Float](repeating: -0.5, count: hv),
        dtBias: [Float](repeating: 0, count: hv),
        hidden: hidden,
        numKeyHeads: hk, numValueHeads: hv,
        keyHeadDim: dk, valueHeadDim: dv,
        keyDim: keyDim, valueDim: valueDim, convDim: convDim,
        convKernel: convKernel, eps: eps, invKeyScale: invKeyScale,
        dtype: .bf16)
    func denseFFN() -> Qwen35FFN {
        .dense(
            Qwen35DenseMLP(
                gateProj: AnyLinear(Linear(weight: rand([768, hidden], scale: 0.05))),
                upProj: AnyLinear(Linear(weight: rand([768, hidden], scale: 0.05))),
                downProj: AnyLinear(Linear(weight: rand([hidden, 768], scale: 0.05)))))
    }
    let gdn = Qwen35GDNLayer(
        inputNorm: RMSNorm(weight: ones([hidden]), eps: eps),
        postNorm: RMSNorm(weight: ones([hidden]), eps: eps),
        mixer: gdnMixer, ffn: denseFFN(), hidden: hidden)

    let attnMixer = Qwen35AttentionMixer(
        qProj: AnyLinear(Linear(weight: rand([nHeads * headDim, hidden], scale: 0.05))),
        kProj: AnyLinear(Linear(weight: rand([nKVHeads * headDim, hidden], scale: 0.05))),
        vProj: AnyLinear(Linear(weight: rand([nKVHeads * headDim, hidden], scale: 0.05))),
        oProj: AnyLinear(Linear(weight: rand([hidden, nHeads * headDim], scale: 0.05))),
        qNorm: RMSNorm(weight: ones([headDim]), eps: eps),
        kNorm: RMSNorm(weight: ones([headDim]), eps: eps),
        nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, rotaryDim: 32,
        ropeTheta: 10_000, attnOutputGate: false)
    let attn = Qwen35AttentionLayer(
        inputNorm: RMSNorm(weight: ones([hidden]), eps: eps),
        postNorm: RMSNorm(weight: ones([hidden]), eps: eps),
        mixer: attnMixer, ffn: denseFFN(), hidden: hidden)

    let embedW = rand([vocab, hidden], scale: 0.5)
    return Qwen35Model(
        embedTokens: AnyEmbedding(Embedding(weight: embedW)),
        layers: [gdn, attn],
        finalNorm: RMSNorm(weight: ones([hidden]), eps: eps),
        lmHead: AnyLinear(Linear(weight: rand([vocab, hidden], scale: 0.1))),
        hidden: hidden, nLayers: 2, nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
        numKeyHeads: hk, numValueHeads: hv,
        keyHeadDim: dk, valueHeadDim: dv,
        convDim: convDim, convKernel: convKernel,
        vocab: vocab, maxSeq: maxSeq, dtype: .bf16,
        kvCacheKind: kv, kvEviction: eviction)
}
